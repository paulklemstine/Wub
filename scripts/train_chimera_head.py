#!/usr/bin/env python3
"""Closed-loop DAgger Roundtrip Trainer for CHIMERA 50-D -> 21-D Head Synthesis.

Architecture (`chimera_head` inside singing_head_model.json):
  Input (50-d):
    [0..17]  18-d browser-exact acoustic features
    [18..49] 32-band normalized spectral envelope (80..9000 Hz, 2x [1,2,1]/4 smoothed)
  Network:
    50 -> 256 (GELU) -> 256 (GELU, residual skip from h1) -> 21:
      [0..13]  sigmoid -> 14 normalized CHIMERA articulators u[0..13]
      [14]     sigmoid -> pressure   [2.8 .. 15.5]
      [15]     sigmoid -> breath     [0.00 .. 0.96]
      [16]     sigmoid -> tension    [0.04 .. 0.96]
      [17]     sigmoid -> brightness [0.02 .. 1.00]
      [18]     sigmoid -> voicedProb [0.00 .. 1.00]
      [19]     sigmoid -> roughness  [0.00 .. 0.55]
      [20]     tanh*1.5 -> f0_log_pred (normalized log2(f0 / 220))

Pipeline:
  1. Synthesize 65,536 diverse random 21-D CHIMERA waveforms via native `soundbank_gen`
     (single unified BodyMix(0.5, 0.0) + Studio-Mic EQ).
  2. Train high-capacity ForwardCHIMERANet(21-D -> 32-band env + 18-D feats).
  3. Train ChimeraHeadInverseNet across all real 60 Hz vocal stem sequences (864k frames)
     + random CHIMERA soundbank with direct spectral + multi-band + feature roundtrip loss.
  4. Closed-Loop DAgger Round 2: evaluate Round-1 policy on 32,768 held-out/active real
     track frames, render actual CHIMERA waveforms for those predictions, add on-policy
     samples to the replay buffer (98,304 total), retrain ForwardCHIMERANet, and fine-tune
     ChimeraHeadInverseNet for exact on-policy CHIMERA parity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_tract_soundbank import (
    TARGET_SR,
    FFT_SIZE,
    TRACT_BOUNDS_NP,
    _apply_studio_mic_eq_torch,
    _extract_gpu_features_and_env,
    _node_compile_and_sample,
)
from scripts.train_singing_head import _smooth_env_bands


def _resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def _render_native(in_21: np.ndarray, chimera_bin: Path) -> np.ndarray:
    """Render (N, 2048) raw CHIMERA waveforms from float32 (N, 21) inputs."""
    n_items = len(in_21)
    tmp_dir = Path(tempfile.mkdtemp(prefix="wub_chi21_"))
    try:
        in_p = tmp_dir / "in.bin"
        out_p = tmp_dir / "out.bin"
        in_21.astype(np.float32).tofile(in_p)
        subprocess.run([str(chimera_bin), str(in_p), str(out_p), str(TARGET_SR)], check=True)
        return np.fromfile(out_p, dtype=np.float32).reshape(n_items, FFT_SIZE).copy()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_features_batch_eq(
    raw_waves: np.ndarray,
    target_rms: np.ndarray,
    dev: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Studio-Mic EQ, match target_rms, extract 18-D feats and smoothed 32-band env."""
    n = len(raw_waves)
    out_feats = np.zeros((n, 18), dtype=np.float32)
    out_env = np.zeros((n, 32), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            w_t = torch.from_numpy(raw_waves[s:e]).to(dev)
            w_eq = _apply_studio_mic_eq_torch(w_t, TARGET_SR)
            w_rms = torch.sqrt(torch.mean(w_eq * w_eq, dim=1, keepdim=True) + 1e-12)
            t_rms = torch.from_numpy(target_rms[s:e]).to(dev).unsqueeze(1)
            audio_b = (w_eq / w_rms) * t_rms
            feats_t, env_t, _ = _extract_gpu_features_and_env(audio_b, TARGET_SR)
            out_feats[s:e] = feats_t.cpu().numpy()
            out_env[s:e] = env_t.cpu().numpy()
    return out_feats, _smooth_env_bands(out_env)


def generate_full21_soundbank(
    n_samples: int,
    chimera_bin: Path,
    dev: torch.device,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `n_samples` exact (in_21, feats_18, env_32) triples from CHIMERA."""
    rng = np.random.default_rng(seed)
    n_postures = max(1024, n_samples // 2)
    exc = 2
    print(f"[CHIMERA-SB] Sampling {n_postures} tract postures x {exc} excitations...", flush=True)
    art_14, areas_44, _, lip_area, min_area, _ = _node_compile_and_sample(n_postures)
    valid = np.all(np.isfinite(areas_44), axis=1) & np.all(np.isfinite(art_14), axis=1) & (lip_area >= 0.01)
    art_rep = np.repeat(art_14[valid], exc, axis=0)[:n_samples].astype(np.float32)
    min_a_rep = np.repeat(min_area[valid], exc, axis=0)[:n_samples].astype(np.float32)
    m = len(art_rep)

    lo = TRACT_BOUNDS_NP[:, 0][None, :]
    hi = TRACT_BOUNDS_NP[:, 1][None, :]
    u14 = np.clip((art_rep - lo) / np.maximum(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)
    # Full continuous speaker pitch/tract scaling via u14[:, 13] (larynx) across [-1.45..+1.45]
    mix_mask = rng.random(m) < 0.35
    u14[mix_mask, 13] = rng.uniform(0.0, 1.0, size=int(np.sum(mix_mask))).astype(np.float32)

    log_f0 = rng.uniform(math.log2(78.0), math.log2(640.0), size=m).astype(np.float32)
    f0_hz = np.clip(2.0 ** log_f0, 75.0, 680.0).astype(np.float32)

    tight = np.clip((0.30 - min_a_rep) / 0.28, 0.0, 1.0)
    liptuck = np.clip(art_rep[:, 11], 0.0, 1.0)
    tuck_w = np.clip((liptuck - 0.25) / 0.65, 0.0, 1.0)
    is_fric = rng.random(m) < (0.14 + 0.45 * np.maximum(tight, tuck_w))
    voiced_flag = np.where(is_fric, 0.0, 1.0).astype(np.float32)

    pressure = rng.uniform(3.2, 15.2, size=m).astype(np.float32)
    breath = np.where(
        is_fric,
        rng.uniform(0.35, 0.96, size=m),
        np.where(rng.random(m) < 0.35, rng.uniform(0.25, 0.85, size=m), rng.uniform(0.01, 0.32, size=m)),
    ).astype(np.float32)
    tension = rng.uniform(0.05, 0.96, size=m).astype(np.float32)
    bright = rng.uniform(0.02, 1.00, size=m).astype(np.float32)
    roughness = np.where(rng.random(m) < 0.35, rng.uniform(0.12, 0.55, size=m), rng.uniform(0.0, 0.14, size=m)).astype(np.float32)

    in_21 = np.zeros((m, 21), dtype=np.float32)
    in_21[:, 0:14] = u14
    in_21[:, 14] = f0_hz
    in_21[:, 15] = pressure
    in_21[:, 16] = breath
    in_21[:, 17] = tension
    in_21[:, 18] = bright
    in_21[:, 19] = voiced_flag
    in_21[:, 20] = roughness

    print(f"[CHIMERA-SB] Rendering {m} real CHIMERA waveforms via native binary...", flush=True)
    raw_waves = _render_native(in_21, chimera_bin)
    target_rms = np.exp(rng.uniform(math.log(0.015), math.log(0.28), size=m)).astype(np.float32)
    feats_18, env_32 = _extract_features_batch_eq(raw_waves, target_rms, dev=dev)
    feats_18[:, 8] = rng.uniform(0.0, 0.35, size=m).astype(np.float32)
    return in_21, feats_18, env_32


def _in21_to_ctrl_tensor(in_21: np.ndarray) -> np.ndarray:
    """Convert raw in_21 [u14, f0_hz, prs, breath, tens, bright, vf, rough] to normalized 21-d controls."""
    out = np.zeros_like(in_21, dtype=np.float32)
    out[:, 0:14] = np.clip(in_21[:, 0:14], 0.0, 1.0)
    out[:, 14] = np.clip(np.log2(np.maximum(65.0, in_21[:, 14]) / 220.0), -1.5, 1.5)
    out[:, 15] = np.clip((in_21[:, 15] - 2.8) / (15.5 - 2.8), 0.0, 1.0)
    out[:, 16] = np.clip(in_21[:, 16] / 0.96, 0.0, 1.0)
    out[:, 17] = np.clip((in_21[:, 17] - 0.04) / (0.96 - 0.04), 0.0, 1.0)
    out[:, 18] = np.clip((in_21[:, 18] - 0.02) / (1.00 - 0.02), 0.0, 1.0)
    out[:, 19] = np.clip(in_21[:, 19], 0.0, 1.0)
    out[:, 20] = np.clip(in_21[:, 20] / 0.55, 0.0, 1.0)
    return out


def _ctrl_to_in21_np(ctrl: np.ndarray) -> np.ndarray:
    """Inverse of `_in21_to_ctrl_tensor`: normalized 21-d controls -> raw in_21 for soundbank_gen."""
    out = np.zeros_like(ctrl, dtype=np.float32)
    out[:, 0:14] = np.clip(ctrl[:, 0:14], 0.0, 1.0)
    out[:, 14] = np.clip(220.0 * (2.0 ** np.clip(ctrl[:, 14], -1.5, 1.5)), 70.0, 760.0)
    out[:, 15] = 2.8 + np.clip(ctrl[:, 15], 0.0, 1.0) * (15.5 - 2.8)
    out[:, 16] = np.clip(ctrl[:, 16], 0.0, 1.0) * 0.96
    out[:, 17] = 0.04 + np.clip(ctrl[:, 17], 0.0, 1.0) * (0.96 - 0.04)
    out[:, 18] = 0.02 + np.clip(ctrl[:, 18], 0.0, 1.0) * (1.00 - 0.02)
    out[:, 19] = (ctrl[:, 19] > 0.5).astype(np.float32)
    out[:, 20] = np.clip(ctrl[:, 20], 0.0, 1.0) * 0.55
    return out


class ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + 0.5 * self.net(x))


class ForwardCHIMERANet(nn.Module):
    """High-accuracy differentiable CHIMERA surrogate: 21-D controls -> (32-band env, 18-D feats)."""

    def __init__(self, in_dim: int = 21, hid: int = 384) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU())
        self.b1 = ResBlock(hid)
        self.b2 = ResBlock(hid)
        self.b3 = ResBlock(hid)
        self.head = nn.Sequential(nn.Linear(hid, 256), nn.GELU())
        self.env_out = nn.Linear(256, 32)
        self.feat_out = nn.Linear(256, 18)

    def forward(self, ctrl_21: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.b3(self.b2(self.b1(self.stem(ctrl_21))))
        h = self.head(h)
        raw_env = self.env_out(h)
        c_env = raw_env - raw_env.mean(dim=-1, keepdim=True)
        env = F.normalize(c_env, p=2, dim=-1, eps=1e-6)
        feats = self.feat_out(h)
        return env, feats


class ChimeraHeadInverseNet(nn.Module):
    """Fast browser-exportable ResMLP: 50-D [18 feats + 32 env] -> 21-D normalized CHIMERA controls."""

    def __init__(self, in_dim: int = 50, hid: int = 256, out_dim: int = 21) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, out_dim)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(h1 + self.fc2(h1))
        return self.fc3(h2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(x)
        u14_and_src = torch.sigmoid(raw[:, :19])          # u14 (0..13), prs, breath, tens, bright, voiced_prob
        rough_n = 0.40 * torch.sigmoid(raw[:, 19:20])     # keep roughness in clean [0..0.22] regime (no diplophonia)
        f0_log = 1.5 * torch.tanh(raw[:, 20:21])          # normalized log2(f0 / 220)
        return torch.cat([u14_and_src, rough_n, f0_log], dim=1)


def train_forward_chimera(
    ctrl_all: np.ndarray,
    env_all: np.ndarray,
    feats_norm_all: np.ndarray,
    dev: torch.device,
    epochs: int = 120,
    batch_size: int = 2048,
    seed: int = 42,
) -> Tuple[ForwardCHIMERANet, float, float]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ctrl_all))
    cut = int(0.92 * len(perm))
    tr, va = perm[:cut], perm[cut:]

    xc = torch.from_numpy(ctrl_all[tr]).to(dev)
    ye = torch.from_numpy(env_all[tr]).to(dev)
    yf = torch.from_numpy(feats_norm_all[tr]).to(dev)
    xvc = torch.from_numpy(ctrl_all[va]).to(dev)
    yve = torch.from_numpy(env_all[va]).to(dev)
    yvf = torch.from_numpy(feats_norm_all[va]).to(dev)

    net = ForwardCHIMERANet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best_score = -1e9
    best_sd = None
    best_cos = 0.0
    best_r2 = 0.0
    n_tr = len(xc)
    for ep in range(epochs):
        net.train()
        p = torch.randperm(n_tr, device=dev)
        for s in range(0, n_tr, batch_size):
            idx = p[s : s + batch_size]
            pe, pf = net(xc[idx])
            cos_l = (1.0 - (pe * ye[idx]).sum(dim=-1)).mean()
            loss = cos_l + 1.5 * F.smooth_l1_loss(pe, ye[idx]) + 0.55 * F.smooth_l1_loss(pf, yf[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                ve, vf = net(xvc)
                cos_m = float((ve * yve).sum(dim=-1).mean().item())
                res = float(((ve - yve) ** 2).sum().item())
                tot = float(((yve - yve.mean(dim=0, keepdim=True)) ** 2).sum().item())
                r2 = 1.0 - res / max(1e-8, tot)
                if cos_m + 0.5 * r2 > best_score:
                    best_score = cos_m + 0.5 * r2
                    best_cos, best_r2 = cos_m, r2
                    best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if best_sd is not None:
        net.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    net.eval()
    for p_param in net.parameters():
        p_param.requires_grad = False
    return net, best_cos, best_r2


def train_inverse_roundtrip(
    inv: ChimeraHeadInverseNet,
    fwd: ForwardCHIMERANet,
    seqs_data: dict,
    sb_ctrl: np.ndarray,
    sb_x50_norm: np.ndarray,
    sb_env: np.ndarray,
    sb_feats_norm: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    dev: torch.device,
    epochs: int = 85,
    lr: float = 2.0e-3,
    seed: int = 42,
) -> Tuple[ChimeraHeadInverseNet, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    x50_seq = seqs_data["x50"]
    env_seq = seqs_data["env"]
    feats_raw_seq = seqs_data["feats"]
    rms_seq = seqs_data["rms"]
    valid_seq = seqs_data["valid"]
    is_tr = seqs_data["is_train"]
    is_va = seqs_data["is_val"]

    x50_norm_seq = ((x50_seq - x_mean) / x_std).astype(np.float32)
    feats_norm_seq = ((feats_raw_seq - feat_mean) / feat_std).astype(np.float32)

    xt = torch.from_numpy(x50_norm_seq[is_tr]).to(dev)
    ft_norm = torch.from_numpy(feats_norm_seq[is_tr]).to(dev)
    ft_raw = torch.from_numpy(feats_raw_seq[is_tr]).to(dev)
    et = torch.from_numpy(env_seq[is_tr]).to(dev)
    rt = torch.from_numpy(rms_seq[is_tr]).to(dev)
    vt = torch.from_numpy(valid_seq[is_tr]).to(dev)

    xv = torch.from_numpy(x50_norm_seq[is_va]).to(dev)
    fv_norm = torch.from_numpy(feats_norm_seq[is_va]).to(dev)
    fv_raw = torch.from_numpy(feats_raw_seq[is_va]).to(dev)
    ev = torch.from_numpy(env_seq[is_va]).to(dev)
    rv = torch.from_numpy(rms_seq[is_va]).to(dev)
    vv = torch.from_numpy(valid_seq[is_va]).to(dev)

    sb_x_t = torch.from_numpy(sb_x50_norm).to(dev)
    sb_c_t = torch.from_numpy(sb_ctrl).to(dev)
    sb_e_t = torch.from_numpy(sb_env).to(dev)
    sb_f_t = torch.from_numpy(sb_feats_norm).to(dev)
    n_sb = len(sb_x_t)

    # Indices of tracked spectral/timbral features inside 18-d feats (skipping flux=8 and loud=0/peak=1 RMS scaling)
    spec_feat_idx = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17], device=dev, dtype=torch.long)

    opt = torch.optim.AdamW(inv.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_tr_seq, t_len, _ = xt.shape
    batch_seqs = 64
    best_val_cos = -1.0
    best_sd = None
    best_metrics: Dict[str, float] = {}

    for ep in range(epochs):
        inv.train()
        perm = torch.randperm(n_tr_seq, device=dev)
        for s in range(0, n_tr_seq, batch_seqs):
            idx = perm[s : s + batch_seqs]
            xb = xt[idx]             # (B, T, 50)
            fb_norm = ft_norm[idx]   # (B, T, 18)
            fb_raw = ft_raw[idx]     # (B, T, 18)
            eb = et[idx]             # (B, T, 32)
            rb = rt[idx]             # (B, T)
            vb = vt[idx]             # (B, T)

            bs = xb.shape[0]
            flat = bs * t_len
            pred_ctrl = inv(xb.reshape(flat, 50)).reshape(bs, t_len, 21)

            # Active frame weighting (focus on actual audible audio >= 0.004 RMS)
            act_w = (torch.clamp((rb - 0.003) / 0.020, 0.0, 1.0) * vb).reshape(flat, 1)
            denom_w = torch.clamp(act_w.sum(), min=1.0)

            # Blend exact measured F0 when cleanly voiced so forward surrogate evaluates true pitch regime
            flat_ctrl = pred_ctrl.reshape(flat, 21)
            real_voic = fb_raw.reshape(flat, 18)[:, 3:4]
            real_f0_log = fb_raw.reshape(flat, 18)[:, 4:5]
            f0_for_fwd = torch.where(real_voic > 0.35, real_f0_log, flat_ctrl[:, 20:21])
            # Straight-Through Estimator (STE) so ForwardCHIMERANet sees exact binary voiced 0/1 like CHIMERA
            v_soft = flat_ctrl[:, 18:19]
            v_hard = (v_soft > 0.5).float()
            v_ste = v_soft + (v_hard - v_soft).detach()
            fwd_in_ordered = torch.cat(
                [flat_ctrl[:, 0:14], f0_for_fwd, flat_ctrl[:, 14:18], v_ste, flat_ctrl[:, 19:20]],
                dim=1,
            )

            p_env, p_feats = fwd(fwd_in_ordered)
            real_env_f = eb.reshape(flat, 32)
            cos_real = (p_env * real_env_f).sum(dim=-1, keepdim=True)
            loss_rt_env = ((3.50 * (1.0 - cos_real) + 2.50 * F.smooth_l1_loss(p_env, real_env_f, reduction="none").mean(dim=-1, keepdim=True)) * act_w).sum() / denom_w

            f_err = F.smooth_l1_loss(p_feats[:, spec_feat_idx], fb_norm.reshape(flat, 18)[:, spec_feat_idx], reduction="none").mean(dim=-1, keepdim=True)
            loss_rt_feat = (f_err * act_w).sum() / denom_w

            # Pitch & voicing alignment on active frames
            v_f0_mask = ((real_voic > 0.45).float() * act_w)
            loss_f0 = (F.smooth_l1_loss(flat_ctrl[:, 20:21], real_f0_log, reduction="none") * v_f0_mask).sum() / torch.clamp(v_f0_mask.sum(), min=1.0)
            target_vf = (real_voic > 0.34).float()
            loss_vf = (F.smooth_l1_loss(v_soft, target_vf, reduction="none") * act_w).sum() / denom_w

            # Temporal smoothness on controls (keep syllables expressive yet free of 60Hz jitter)
            m1 = (vb[:, 1:] * vb[:, :-1]).unsqueeze(-1)
            d1 = pred_ctrl[:, 1:, :20] - pred_ctrl[:, :-1, :20]
            loss_vel = ((d1 * d1) * m1).sum() / torch.clamp(m1.sum() * 20.0, min=1.0)
            m2 = (vb[:, 2:] * vb[:, 1:-1] * vb[:, :-2]).unsqueeze(-1)
            d2 = pred_ctrl[:, 2:, :20] - 2.0 * pred_ctrl[:, 1:-1, :20] + pred_ctrl[:, :-2, :20]
            loss_acc = ((d2 * d2) * m2).sum() / torch.clamp(m2.sum() * 20.0, min=1.0)

            # Direct soundbank parameter + roundtrip supervision (2048 CHIMERA frames/step)
            sb_i = torch.randint(0, n_sb, (2048,), device=dev)
            sb_pred = inv(sb_x_t[sb_i])
            sb_tgt = sb_c_t[sb_i]
            # sb_pred order: [u14(0..13), prs(14), br(15), tens(16), brt(17), vprob(18), rgh(19), f0_log(20)]
            # sb_tgt order:  [u14(0..13), f0_log(14), prs(15), br(16), tens(17), brt(18), vf(19), rgh(20)]
            sb_pred_ordered = torch.cat([sb_pred[:, 0:14], sb_pred[:, 20:21], sb_pred[:, 14:20]], dim=1)
            loss_sb_param = F.smooth_l1_loss(sb_pred_ordered[:, :14], sb_tgt[:, :14]) + 0.85 * F.smooth_l1_loss(sb_pred_ordered[:, 15:21], sb_tgt[:, 15:21])
            sb_fwd_in = torch.cat([sb_pred[:, 0:14], sb_tgt[:, 14:15], sb_pred[:, 14:20]], dim=1)
            sb_pe, sb_pf = fwd(sb_fwd_in)
            loss_sb_rt = (1.0 - (sb_pe * sb_e_t[sb_i]).sum(dim=-1)).mean() + 0.40 * F.smooth_l1_loss(sb_pf[:, spec_feat_idx], sb_f_t[sb_i][:, spec_feat_idx])

            loss = (
                loss_rt_env
                + 0.75 * loss_rt_feat
                + 0.45 * loss_f0
                + 0.55 * loss_vf
                + 3.50 * loss_vel
                + 6.00 * loss_acc
                + 0.35 * loss_sb_param
                + 0.85 * loss_sb_rt
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inv.parameters(), 2.0)
            opt.step()
        sched.step()

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            inv.eval()
            with torch.no_grad():
                nv = xv.shape[0]
                flat_v = nv * t_len
                pc_v = inv(xv.reshape(flat_v, 50))
                rv_f = fv_raw.reshape(flat_v, 18)[:, 3:4]
                rf0_f = fv_raw.reshape(flat_v, 18)[:, 4:5]
                f0_use = torch.where(rv_f > 0.35, rf0_f, pc_v[:, 20:21])
                vh_v = (pc_v[:, 18:19] > 0.5).float()
                fwd_v = torch.cat([pc_v[:, 0:14], f0_use, pc_v[:, 14:18], vh_v, pc_v[:, 19:20]], dim=1)
                pe_v, pf_v = fwd(fwd_v)
                act_v = (rv.reshape(flat_v) > 0.015) & (vv.reshape(flat_v) > 0.5)
                val_cos = float((pe_v[act_v] * ev.reshape(flat_v, 32)[act_v]).sum(dim=-1).mean().item())
                vf_t = fv_norm.reshape(flat_v, 18)[act_v][:, spec_feat_idx]
                pf_t = pf_v[act_v][:, spec_feat_idx]
                f_res = float(((pf_t - vf_t) ** 2).sum().item())
                f_tot = float(((vf_t - vf_t.mean(dim=0, keepdim=True)) ** 2).sum().item())
                val_fr2 = 1.0 - f_res / max(1e-8, f_tot)
                pc_seq = pc_v.reshape(nv, t_len, 21)[:, :, :20]
                vel_v = float((pc_seq[:, 1:] - pc_seq[:, :-1]).abs().mean().item())
                acc_v = float((pc_seq[:, 2:] - 2.0 * pc_seq[:, 1:-1] + pc_seq[:, :-2]).abs().mean().item())
                if val_cos > best_val_cos:
                    best_val_cos = val_cos
                    best_sd = {k: v.detach().cpu().clone() for k, v in inv.state_dict().items()}
                    best_metrics = {
                        "surrogate_val_env_cos": round(val_cos, 4),
                        "surrogate_val_feat_r2": round(val_fr2, 4),
                        "val_vel": round(vel_v, 5),
                        "val_accel": round(acc_v, 5),
                    }
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                print(
                    f"[chimera_head] ep {ep+1}/{epochs} val_cos={val_cos:.4f} feat_r2={val_fr2:.4f} "
                    f"vel={vel_v:.4f} acc={acc_v:.4f}",
                    flush=True,
                )

    if best_sd is not None:
        inv.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    return inv, best_metrics


def evaluate_real_chimera_audio(
    inv: ChimeraHeadInverseNet,
    seqs_data: dict,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    chimera_bin: Path,
    dev: torch.device,
    n_eval: int = 2048,
    seed: int = 99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run inv on `n_eval` held-out active validation frames and render ACTUAL CHIMERA audio."""
    inv.eval()
    rng = np.random.default_rng(seed)
    va_mask = seqs_data["is_val"]
    x50_va = seqs_data["x50"][va_mask].reshape(-1, 50)
    feats_va = seqs_data["feats"][va_mask].reshape(-1, 18)
    env_va = seqs_data["env"][va_mask].reshape(-1, 32)
    rms_va = seqs_data["rms"][va_mask].reshape(-1)
    valid_va = seqs_data["valid"][va_mask].reshape(-1) > 0.5
    active = valid_va & (rms_va > 0.015)
    act_idx = np.nonzero(active)[0]
    sel = rng.choice(act_idx, size=min(n_eval, len(act_idx)), replace=False)

    x_sel = ((x50_va[sel] - x_mean) / x_std).astype(np.float32)
    with torch.no_grad():
        pred = inv(torch.from_numpy(x_sel).to(dev)).cpu().numpy()

    voic_sel = feats_va[sel, 3]
    f0_sel = feats_va[sel, 4]
    f0_log_use = np.where(voic_sel > 0.35, f0_sel, pred[:, 20])
    ctrl_ordered = np.concatenate([pred[:, 0:14], f0_log_use[:, None], pred[:, 14:20]], axis=1)
    in_21 = _ctrl_to_in21_np(ctrl_ordered)

    waves = _render_native(in_21, chimera_bin)
    s_feats, s_env = _extract_features_batch_eq(waves, rms_va[sel], dev=dev)
    env_cos = np.sum(s_env * env_va[sel], axis=1)

    r_f = feats_va[sel]
    corrs = []
    for col in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
        a0 = s_feats[:, col] - s_feats[:, col].mean()
        b0 = r_f[:, col] - r_f[:, col].mean()
        den = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
        corrs.append(float((a0 * b0).sum() / den) if den > 1e-8 else 0.0)
    return in_21, s_feats, s_env, float(env_cos.mean()), float(np.mean(corrs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train closed-loop 50-D -> 21-D CHIMERA Singing Head.")
    parser.add_argument("--seq-cache", type=Path, default=Path("/tmp/wub_vocal_seqs_cache.pt"))
    parser.add_argument("--chimera-bin", type=Path, default=Path("/tmp/wub_head_train/soundbank_gen"))
    parser.add_argument("--sb-cache", type=Path, default=Path("/tmp/wub_soundbank_chimera21.npz"))
    parser.add_argument("--base-model", type=Path, default=Path("/tmp/wub_head_train/singing_head_model.json"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/wub_head_train/singing_head_model_with_chimera.json"))
    parser.add_argument("--sb-samples", type=int, default=65536)
    parser.add_argument("--dagger-samples", type=int, default=32768)
    parser.add_argument("--epochs-r1", type=int, default=75)
    parser.add_argument("--epochs-r2", type=int, default=45)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[CHIMERA-TRAIN] Loading sequences from {args.seq_cache}...", flush=True)
    all_seqs = torch.load(args.seq_cache, weights_only=False)
    feats_seq = np.stack([s["feats"] for s in all_seqs], axis=0).astype(np.float32)
    env_seq = _smooth_env_bands(np.stack([s["env"] for s in all_seqs], axis=0).astype(np.float32))
    rms_seq = np.stack([s["rms"] for s in all_seqs], axis=0).astype(np.float32)
    valid_seq = np.stack([s["valid"] for s in all_seqs], axis=0).astype(np.float32)
    x50_seq = np.concatenate([feats_seq, env_seq], axis=-1).astype(np.float32)

    tids = [s["track_id"] for s in all_seqs]
    utids = sorted(set(tids))
    rng.shuffle(utids)
    val_tids = set(utids[: max(1, int(round(0.14 * len(utids))))])
    is_val = np.array([t in val_tids for t in tids], dtype=bool)
    is_train = ~is_val

    tr_mask = valid_seq[is_train] > 0.5
    x50_tr_flat = x50_seq[is_train][tr_mask]
    x_mean = np.mean(x50_tr_flat, axis=0).astype(np.float32)
    x_std = np.maximum(np.std(x50_tr_flat, axis=0), 1e-4).astype(np.float32)
    feat_mean = x_mean[:18]
    feat_std = x_std[:18]

    seqs_data = {
        "x50": x50_seq,
        "env": env_seq,
        "feats": feats_seq,
        "rms": rms_seq,
        "valid": valid_seq,
        "is_train": is_train,
        "is_val": is_val,
    }

    if args.sb_cache.exists():
        print(f"[CHIMERA-TRAIN] Loading cached soundbank from {args.sb_cache}...", flush=True)
        d = np.load(args.sb_cache)
        sb_in21, sb_feats, sb_env = d["in_21"], d["feats"], d["env"]
    else:
        sb_in21, sb_feats, sb_env = generate_full21_soundbank(
            args.sb_samples, args.chimera_bin, dev=dev, seed=args.seed
        )
        np.savez_compressed(args.sb_cache, in_21=sb_in21, feats=sb_feats, env=sb_env)

    sb_ctrl = _in21_to_ctrl_tensor(sb_in21)
    sb_x50_norm = ((np.concatenate([sb_feats, sb_env], axis=1) - x_mean) / x_std).astype(np.float32)
    sb_feats_norm = ((sb_feats - feat_mean) / feat_std).astype(np.float32)

    print(f"[CHIMERA-TRAIN] Stage 1: Training ForwardCHIMERANet on {len(sb_ctrl)} samples...", flush=True)
    fwd, fwd_cos, fwd_r2 = train_forward_chimera(sb_ctrl, sb_env, sb_feats_norm, dev=dev, epochs=110, seed=args.seed)
    print(f"  ForwardCHIMERANet val_cos={fwd_cos:.4f} val_env_r2={fwd_r2:.4f}", flush=True)

    print("[CHIMERA-TRAIN] Stage 2 (Round 1): Training ChimeraHeadInverseNet (50->256->256->21)...", flush=True)
    inv = ChimeraHeadInverseNet().to(dev)
    inv, r1_metrics = train_inverse_roundtrip(
        inv, fwd, seqs_data, sb_ctrl, sb_x50_norm, sb_env, sb_feats_norm,
        x_mean, x_std, feat_mean, feat_std, dev=dev, epochs=args.epochs_r1, seed=args.seed,
    )
    _, _, _, r1_real_cos, r1_real_fc = evaluate_real_chimera_audio(
        inv, seqs_data, x_mean, x_std, args.chimera_bin, dev=dev
    )
    print(f"  [Round 1 ACTUAL CHIMERA EVAL (n=2048 val active)] env_cos={r1_real_cos:.4f} feat_corr={r1_real_fc:.4f}", flush=True)

    # Stage 3: Closed-Loop DAgger Round 2 (on-policy CHIMERA replay)
    if args.dagger_samples > 0:
        print(f"[CHIMERA-TRAIN] Stage 3 (DAgger Round 2): Generating {args.dagger_samples} on-policy CHIMERA renders...", flush=True)
        inv.eval()
        tr_x50 = x50_seq[is_train].reshape(-1, 50)
        tr_feats = feats_seq[is_train].reshape(-1, 18)
        tr_rms = rms_seq[is_train].reshape(-1)
        tr_v = valid_seq[is_train].reshape(-1) > 0.5
        act_tr = np.nonzero(tr_v & (tr_rms > 0.012))[0]
        dag_sel = rng.choice(act_tr, size=min(args.dagger_samples, len(act_tr)), replace=False)
        with torch.no_grad():
            dag_in_n = torch.from_numpy(((tr_x50[dag_sel] - x_mean) / x_std).astype(np.float32)).to(dev)
            dag_pred = inv(dag_in_n).cpu().numpy()
        dag_v = tr_feats[dag_sel, 3]
        dag_f0 = tr_feats[dag_sel, 4]
        dag_f0_use = np.where(dag_v > 0.35, dag_f0, dag_pred[:, 20])
        dag_ctrl_ord = np.concatenate([dag_pred[:, 0:14], dag_f0_use[:, None], dag_pred[:, 14:20]], axis=1)
        dag_in21 = _ctrl_to_in21_np(dag_ctrl_ord)
        dag_waves = _render_native(dag_in21, args.chimera_bin)
        dag_feats, dag_env = _extract_features_batch_eq(dag_waves, tr_rms[dag_sel], dev=dev)
        dag_feats[:, 8] = tr_feats[dag_sel, 8]
        dag_ctrl = _in21_to_ctrl_tensor(dag_in21)
        dag_x50_n = ((np.concatenate([dag_feats, dag_env], axis=1) - x_mean) / x_std).astype(np.float32)
        dag_f_n = ((dag_feats - feat_mean) / feat_std).astype(np.float32)

        comb_ctrl = np.concatenate([sb_ctrl, dag_ctrl], axis=0)
        comb_env = np.concatenate([sb_env, dag_env], axis=0)
        comb_x50_n = np.concatenate([sb_x50_norm, dag_x50_n], axis=0)
        comb_f_n = np.concatenate([sb_feats_norm, dag_f_n], axis=0)

        print(f"[CHIMERA-TRAIN] Retraining ForwardCHIMERANet on {len(comb_ctrl)} combined samples...", flush=True)
        fwd, fwd_cos2, fwd_r2_2 = train_forward_chimera(comb_ctrl, comb_env, comb_f_n, dev=dev, epochs=95, seed=args.seed + 7)
        print(f"  DAgger ForwardCHIMERANet val_cos={fwd_cos2:.4f} val_env_r2={fwd_r2_2:.4f}", flush=True)

        inv, r2_metrics = train_inverse_roundtrip(
            inv, fwd, seqs_data, comb_ctrl, comb_x50_n, comb_env, comb_f_n,
            x_mean, x_std, feat_mean, feat_std, dev=dev, epochs=args.epochs_r2, lr=1.0e-3, seed=args.seed + 7,
        )
        _, _, _, r2_real_cos, r2_real_fc = evaluate_real_chimera_audio(
            inv, seqs_data, x_mean, x_std, args.chimera_bin, dev=dev
        )
        print(f"  [Round 2 ACTUAL CHIMERA EVAL (n=2048 val active)] env_cos={r2_real_cos:.4f} feat_corr={r2_real_fc:.4f}", flush=True)
    else:
        r2_metrics = r1_metrics
        r2_real_cos, r2_real_fc = r1_real_cos, r1_real_fc

    with open(args.base_model, encoding="utf-8") as f:
        payload = json.load(f)

    sd = inv.cpu().state_dict()
    payload["chimera_head"] = {
        "version": 1,
        "format": "wub_chimera_head_v1",
        "in_dim": 50,
        "hidden_dim": 256,
        "out_dim": 21,
        "x_mean": [round(float(v), 6) for v in x_mean.tolist()],
        "x_std": [round(float(v), 6) for v in x_std.tolist()],
        "stats": {
            "forward_cos": round(fwd_cos, 4),
            "r1_real_env_cos": round(r1_real_cos, 4),
            "r1_real_feat_corr": round(r1_real_fc, 4),
            "real_env_cos": round(r2_real_cos, 4),
            "real_feat_corr": round(r2_real_fc, 4),
            **r2_metrics,
        },
        "weights": {
            "w1": np.round(sd["fc1.weight"].numpy(), 6).tolist(),
            "b1": np.round(sd["fc1.bias"].numpy(), 6).tolist(),
            "w2": np.round(sd["fc2.weight"].numpy(), 6).tolist(),
            "b2": np.round(sd["fc2.bias"].numpy(), 6).tolist(),
            "w3": np.round(sd["fc3.weight"].numpy(), 6).tolist(),
            "b3": np.round(sd["fc3.bias"].numpy(), 6).tolist(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"[CHIMERA-TRAIN] Saved {args.out} ({args.out.stat().st_size / 1024.0:.1f} KB).", flush=True)


if __name__ == "__main__":
    main()
