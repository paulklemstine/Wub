#!/usr/bin/env python3
"""Build and train all runtime sidecar artifacts for Wub's per-stem remix engine.

Generates and validates:
  1. arc_context_features.json + arc_model.json
  2. progression_context.json + progression_model.json
  3. rhythm_scores.json + rhythm_flux.f32 + rhythm_mesh_model.json
  4. vocal_mix.json
  5. syntax_successors.json
  6. sidecars.json (all flags enabled)

All schemas strictly match frontend runtime loaders in:
  - frontend/src/stem_remix.ts
  - frontend/src/arc_inference.ts
  - frontend/src/progression_inference.ts
  - frontend/src/rhythm_inference.ts
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_taste_model import PairwiseRanker, encode_ranker_candidates, encode_ranker_contexts  # noqa: E402

SEED = 505
STEMS = ("drums", "bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
SEC_TO_IDX = {s: i for i, s in enumerate(SECTIONS)}
PITCH_NAMES = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}
RHYTHM_FEAT_NAMES = [
    "da", "db", "simult", "union", "neither",
    "p_mid_given_first", "p_first_given_mid",
    "offbeat_diff", "syncop_diff", "beat_diff", "call_response",
    "density_prod", "density_absdiff", "pos_profile_corr",
    "logflux_corr", "joint_peak", "lag_corr_sum",
    "mean_logflux_first", "mean_logflux_mid",
]
RHYTHM_PAIRS = [
    ("drums", "bass"),
    ("drums", "other"),
    ("drums", "vocals"),
    ("bass", "other"),
    ("bass", "vocals"),
    ("other", "vocals"),
]


def parse_key(key_str: str) -> tuple[int, str]:
    """Parse 'F# minor' / 'C major' -> (root 0..11, 'minor'|'major')."""
    parts = str(key_str or "C minor").strip().split()
    if not parts:
        return 0, "minor"
    root = PITCH_NAMES.get(parts[0].upper(), 0)
    mode = "major" if len(parts) > 1 and "MAJ" in parts[1].upper() else "minor"
    return root, mode


def rotate12(vec: list[float] | np.ndarray, shift: int) -> list[float]:
    arr = [float(x) for x in vec[:12]]
    if len(arr) < 12:
        arr.extend([0.0] * (12 - len(arr)))
    s = shift % 12
    return [round(arr[(i - s) % 12], 6) for i in range(12)]


class ArcTransformer(nn.Module):
    """Exact tensor layout matching frontend/src/arc_inference.ts."""

    def __init__(self, feature_dim: int = 48, d_model: int = 96, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 6, n_sections: int = 5, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.proj = nn.Linear(feature_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head_section = nn.Linear(d_model, n_sections)
        self.head_energy = nn.Linear(d_model, 1)
        self.head_delta = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, L, F), mask: bool (B, L) where True = valid token
        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.proj(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        normed = self.ln(pooled)
        return self.head_section(normed), self.head_energy(normed).squeeze(-1), self.head_delta(normed).squeeze(-1)


def build_arc_sidecars(dataset: dict, out_dir: Path, device: torch.device, epochs: int = 18) -> dict[str, Any]:
    """Build arc_context_features.json and train arc_model.json."""
    by_pid: dict[str, list[float]] = {}
    for row in dataset.get("kit_phrases", []):
        feat = [round(float(x), 6) for x in row["features"][:48]]
        by_pid[row["phrase_id"]] = feat

    arc_ctx = {"version": 1, "feature_dim": 48, "byPhraseId": by_pid}
    (out_dir / "arc_context_features.json").write_text(json.dumps(arc_ctx, separators=(",", ":")))

    # Build consecutive sequence samples across aligned corpus tracks
    rng = random.Random(SEED)
    max_len = 6
    track_ids = [t["track_id"] for t in dataset["tracks"]]
    rng_np = np.random.default_rng(SEED)
    perm_tids = list(track_ids)
    rng.shuffle(perm_tids)
    n_val = max(1, int(0.1 * len(perm_tids)))
    val_set = set(perm_tids[:n_val])

    tr_ctx, tr_mask, tr_sec, tr_en, tr_del = [], [], [], [], []
    va_ctx, va_mask, va_sec, va_en, va_del = [], [], [], [], []

    for track in dataset["tracks"]:
        tid = track["track_id"]
        wins = sorted([w for w in track["windows"] if w.get("aligned", True)], key=lambda w: w["start_bar"])
        if len(wins) < 2:
            continue
        is_val = tid in val_set
        for i in range(1, len(wins)):
            # Check continuity backwards up to max_len steps (4-bar stride)
            chain = [wins[i - 1]]
            k = i - 2
            while k >= 0 and len(chain) < max_len and chain[-1]["start_bar"] - wins[k]["start_bar"] == 4:
                chain.append(wins[k])
                k -= 1
            chain.reverse()
            if wins[i]["start_bar"] - chain[-1]["start_bar"] != 4:
                continue
            # Also subsample random prefix lengths so variable-length context generalizes
            if len(chain) > 1 and rng.random() < 0.35:
                keep = rng.randint(1, len(chain))
                chain = chain[-keep:]

            x_pad = np.zeros((max_len, 48), dtype=np.float32)
            m_pad = np.zeros(max_len, dtype=bool)
            off = max_len - len(chain)
            for j, cw in enumerate(chain):
                x_pad[off + j] = np.asarray(cw["features"][:48], dtype=np.float32)
                m_pad[off + j] = True

            nxt = wins[i]
            prev_en = float(chain[-1]["features"][32])
            nxt_en = float(nxt["features"][32])
            sec_idx = SEC_TO_IDX.get(str(nxt.get("section", "GROOVE_A")).upper(), 1)
            if is_val:
                va_ctx.append(x_pad); va_mask.append(m_pad)
                va_sec.append(sec_idx); va_en.append(nxt_en); va_del.append(nxt_en - prev_en)
            else:
                tr_ctx.append(x_pad); tr_mask.append(m_pad)
                tr_sec.append(sec_idx); tr_en.append(nxt_en); tr_del.append(nxt_en - prev_en)

    X_tr = np.stack(tr_ctx)
    M_tr = np.stack(tr_mask)
    X_va = np.stack(va_ctx)
    M_va = np.stack(va_mask)

    valid_rows = X_tr[M_tr]
    feat_mean = valid_rows.mean(axis=0).astype(np.float32)
    feat_std = valid_rows.std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0

    # Notice arc_inference.ts standardizes BOTH valid tokens AND zero-pad slots: (raw - mean) / std
    X_tr_n = (X_tr - feat_mean) / feat_std
    X_va_n = (X_va - feat_mean) / feat_std

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_n), torch.from_numpy(M_tr),
        torch.tensor(tr_sec, dtype=torch.long),
        torch.tensor(tr_en, dtype=torch.float32),
        torch.tensor(tr_del, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.from_numpy(X_va_n), torch.from_numpy(M_va),
        torch.tensor(va_sec, dtype=torch.long),
        torch.tensor(va_en, dtype=torch.float32),
        torch.tensor(va_del, dtype=torch.float32),
    )
    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=512, shuffle=False)

    torch.manual_seed(SEED)
    model = ArcTransformer(feature_dim=48, d_model=96, n_layers=2, n_heads=4, max_len=max_len, n_sections=5, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()

    best_loss = float("inf")
    best_state = None
    best_metrics = {}
    for ep in range(epochs):
        model.train()
        for xb, mb, sb, eb, db in tr_dl:
            xb, mb, sb, eb, db = xb.to(device), mb.to(device), sb.to(device), eb.to(device), db.to(device)
            opt.zero_grad(set_to_none=True)
            s_log, e_log, d_pred = model(xb, mb)
            loss = ce(s_log, sb) + 2.0 * bce(e_log, eb.clamp(0.0, 1.0)) + 2.0 * huber(d_pred, db)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        v_loss, correct, tot = 0.0, 0, 0
        en_err = []
        with torch.no_grad():
            for xb, mb, sb, eb, db in va_dl:
                xb, mb, sb, eb, db = xb.to(device), mb.to(device), sb.to(device), eb.to(device), db.to(device)
                s_log, e_log, d_pred = model(xb, mb)
                loss = ce(s_log, sb) + 2.0 * bce(e_log, eb.clamp(0.0, 1.0)) + 2.0 * huber(d_pred, db)
                v_loss += float(loss.item()) * len(sb)
                correct += int((s_log.argmax(dim=-1) == sb).sum().item())
                tot += len(sb)
                en_err.append(torch.abs(torch.sigmoid(e_log) - eb).cpu().numpy())
        v_mean = v_loss / max(1, tot)
        acc = correct / max(1, tot)
        mae = float(np.mean(np.concatenate(en_err))) if en_err else 0.0
        if v_mean < best_loss:
            best_loss = v_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val_loss": v_mean, "val_section_acc": acc, "val_energy_mae": mae}
        if (ep + 1) % 3 == 0 or ep == epochs - 1:
            print(f"[sidecar:arc] ep {ep+1}/{epochs} val_loss={v_mean:.4f} sec_acc={acc:.4f} en_mae={mae:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    weights_json = {}
    for k, v in model.state_dict().items():
        arr = v.detach().cpu().numpy().astype(np.float32)
        weights_json[k] = arr.tolist()

    arc_doc = {
        "format": "wub-arc-model-v1",
        "version": 1,
        "config": {
            "feature_dim": 48,
            "d_model": 96,
            "n_layers": 2,
            "n_heads": 4,
            "dim_ff": 384,
            "max_len": max_len,
            "dropout": 0.1,
            "activation": "gelu",
            "norm_first": True,
            "n_sections": 5,
        },
        "sections": list(SECTIONS),
        "feat_mean": [round(float(x), 6) for x in feat_mean.tolist()],
        "feat_std": [round(float(x), 6) for x in feat_std.tolist()],
        "weights": weights_json,
        "metrics": best_metrics,
    }
    (out_dir / "arc_model.json").write_text(json.dumps(arc_doc, separators=(",", ":")))
    return {"phrases": len(by_pid), **best_metrics}


def _prog_prev_vec(prev: dict, prev2: dict | None) -> np.ndarray:
    """Match frontend/src/progression_inference.ts::buildPrevVector (56-d) verbatim."""
    v = np.zeros(56, dtype=np.float32)
    if 0 <= prev["rel_chord"] < 12:
        v[prev["rel_chord"]] = 1.0
    if prev2 is not None and 0 <= prev2["rel_chord"] < 12:
        v[12 + prev2["rel_chord"]] = 1.0
    v[24:36] = np.asarray(prev["rel_chroma"], dtype=np.float32)[:12]
    v[36:48] = np.asarray(prev["rel_bass_chroma"], dtype=np.float32)[:12]
    v[48] = float(prev["energy"])
    si = SEC_TO_IDX.get(prev["section"], 1)
    v[49 + si] = 1.0
    v[54] = 1.0 if prev["mode"] == "minor" else 0.0
    v[55] = 1.0 if prev["mode"] == "major" else 0.0
    return v


def build_progression_sidecars(manifest: dict, dataset: dict, out_dir: Path, device: torch.device, epochs: int = 24) -> dict[str, Any]:
    """Build progression_context.json and train progression_model.json."""
    row_by_pid = {r["phrase_id"]: r for r in dataset.get("kit_phrases", [])}
    by_pid = {}
    for kit in manifest.get("kits", []):
        key_str = kit.get("key", "C minor")
        kroot, mode = parse_key(key_str)
        for p in kit.get("phrases", []):
            pid = p["id"]
            row = row_by_pid.get(pid)
            feat = row["features"] if row else None
            abs_chroma = feat[0:12] if feat is not None else (p.get("chroma") or [0.0] * 12)
            bass_chroma = feat[112:124] if feat is not None else [0.0] * 12
            abs_bass_root = int(p.get("bass_root", 0) or 0) % 12
            if feat is not None and sum(feat[148:160]) > 0:
                abs_chord_root = int(np.argmax(feat[148:160]))
            else:
                abs_chord_root = int(np.argmax(abs_chroma)) if sum(abs_chroma) > 0 else abs_bass_root
            rel_chroma = rotate12(abs_chroma, -kroot)
            rel_bass_chroma = rotate12(bass_chroma, -kroot)
            rel_chord = (abs_chord_root - kroot) % 12
            rel_bass = (abs_bass_root - kroot) % 12
            by_pid[pid] = {
                "rel_chord": int(rel_chord),
                "rel_bass": int(rel_bass),
                "rel_chroma": rel_chroma,
                "rel_bass_chroma": rel_bass_chroma,
                "abs_chroma": [round(float(x), 6) for x in abs_chroma[:12]],
                "abs_bass_root": int(abs_bass_root),
                "abs_chord_root": int(abs_chord_root),
                "key": key_str,
                "key_root": int(kroot),
                "mode": mode,
                "section": str(p.get("section", "GROOVE_A")).upper(),
                "energy": round(float(p.get("energy", 0.5)), 6),
            }

    ctx_doc = {
        "format": "wub-progression-context-v1",
        "version": 1,
        "n_phrases": len(by_pid),
        "byPhraseId": by_pid,
    }
    (out_dir / "progression_context.json").write_text(json.dumps(ctx_doc, separators=(",", ":")))

    # Mine corpus windows for Markov transitions + 56->64->12 MLP
    alpha = 1.0
    cnt_uni = np.full((12, 12), alpha, dtype=np.float64)
    cnt_min = np.full((12, 12), alpha, dtype=np.float64)
    cnt_maj = np.full((12, 12), alpha, dtype=np.float64)
    unigram = np.full(12, alpha, dtype=np.float64)

    rng = random.Random(SEED)
    tids = [t["track_id"] for t in dataset["tracks"]]
    rng.shuffle(tids)
    val_set = set(tids[: max(1, int(0.1 * len(tids)))])

    X_tr, Y_tr, X_va, Y_va = [], [], [], []
    for track in dataset["tracks"]:
        tid = track["track_id"]
        kroot, mode = parse_key(track.get("key", "C minor"))
        wins = sorted([w for w in track["windows"] if w.get("aligned", True)], key=lambda w: w["start_bar"])
        entries = []
        for w in wins:
            f = w["features"]
            chord_oh = f[148:160]
            if sum(chord_oh) > 0:
                abs_chord = int(np.argmax(chord_oh))
            else:
                abs_chord = int(np.argmax(f[0:12])) if sum(f[0:12]) > 0 else 0
            entries.append({
                "start_bar": w["start_bar"],
                "rel_chord": (abs_chord - kroot) % 12,
                "rel_chroma": rotate12(f[0:12], -kroot),
                "rel_bass_chroma": rotate12(f[112:124], -kroot),
                "energy": float(f[32]),
                "section": str(w.get("section", "GROOVE_A")).upper(),
                "mode": mode,
            })
        is_val = tid in val_set
        for i, cur in enumerate(entries):
            if not is_val:
                unigram[cur["rel_chord"]] += 1.0
            if i >= 1 and cur["start_bar"] - entries[i - 1]["start_bar"] == 4:
                prev = entries[i - 1]
                prev2 = entries[i - 2] if (i >= 2 and prev["start_bar"] - entries[i - 2]["start_bar"] == 4) else None
                if not is_val:
                    cnt_uni[prev["rel_chord"], cur["rel_chord"]] += 1.0
                    if mode == "minor":
                        cnt_min[prev["rel_chord"], cur["rel_chord"]] += 1.0
                    else:
                        cnt_maj[prev["rel_chord"], cur["rel_chord"]] += 1.0
                vec = _prog_prev_vec(prev, prev2)
                if is_val:
                    X_va.append(vec); Y_va.append(cur["rel_chord"])
                else:
                    X_tr.append(vec); Y_tr.append(cur["rel_chord"])

    def _log_rows(mat: np.ndarray) -> list[list[float]]:
        p = mat / mat.sum(axis=1, keepdims=True)
        return np.log(np.clip(p, 1e-12, 1.0)).tolist()

    uni_log = np.log(np.clip(unigram / unigram.sum(), 1e-12, 1.0)).tolist()

    X_tr_a = np.stack(X_tr)
    Y_tr_a = np.asarray(Y_tr, dtype=np.int64)
    X_va_a = np.stack(X_va)
    Y_va_a = np.asarray(Y_va, dtype=np.int64)
    feat_mean = X_tr_a.mean(axis=0)
    feat_std = X_tr_a.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    X_tr_n = ((X_tr_a - feat_mean) / feat_std).astype(np.float32)
    X_va_n = ((X_va_a - feat_mean) / feat_std).astype(np.float32)

    torch.manual_seed(SEED)
    mlp = nn.Sequential(nn.Linear(56, 64), nn.ReLU(), nn.Linear(64, 12)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    ce = nn.CrossEntropyLoss()
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_a)), batch_size=512, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(Y_va_a)), batch_size=1024)

    best_loss = float("inf")
    best_state = None
    best_acc = 0.0
    for ep in range(epochs):
        mlp.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(mlp(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        mlp.eval()
        tot_loss, correct, tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = mlp(xb)
                tot_loss += float(ce(logits, yb).item()) * len(yb)
                correct += int((logits.argmax(dim=-1) == yb).sum().item())
                tot += len(yb)
        v_loss = tot_loss / max(1, tot)
        acc = correct / max(1, tot)
        if v_loss < best_loss:
            best_loss = v_loss
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
    if best_state:
        mlp.load_state_dict(best_state)
    print(f"[sidecar:prog] best val_ce={best_loss:.4f} top1_acc={best_acc:.4f}", flush=True)

    sd = mlp.state_dict()
    model_doc = {
        "format": "wub-progression-v1",
        "version": 1,
        "config": {
            "n_classes": 12,
            "mlp_in": 56,
            "mlp_hidden": 64,
            "activation": "relu",
            "mode_split": True,
            "alpha": alpha,
        },
        "markov": {
            "unified_log": _log_rows(cnt_uni),
            "minor_log": _log_rows(cnt_min),
            "major_log": _log_rows(cnt_maj),
            "unigram_log": uni_log,
        },
        "mlp": {
            "feat_mean": [round(float(x), 6) for x in feat_mean.tolist()],
            "feat_std": [round(float(x), 6) for x in feat_std.tolist()],
            "fc1_weight": sd["0.weight"].numpy().astype(np.float32).tolist(),
            "fc1_bias": sd["0.bias"].numpy().astype(np.float32).tolist(),
            "fc2_weight": sd["2.weight"].numpy().astype(np.float32).tolist(),
            "fc2_bias": sd["2.bias"].numpy().astype(np.float32).tolist(),
        },
        "metrics": {"val_ce": best_loss, "val_top1": best_acc},
    }
    (out_dir / "progression_model.json").write_text(json.dumps(model_doc, separators=(",", ":")))
    return {"phrases": len(by_pid), "val_ce": best_loss, "val_top1": best_acc}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    sx = float(x.std())
    sy = float(y.std())
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    mx = float(x.mean())
    my = float(y.mean())
    return float(np.mean((x - mx) * (y - my)) / (sx * sy))


def rhythm_pair_features(ga: np.ndarray, fa: np.ndarray, gb: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Exact 19-feature match to frontend/src/rhythm_inference.ts::rhythmMeshFeatures."""
    ga_f = ga.astype(np.float64)
    gb_f = gb.astype(np.float64)
    da = float(ga_f.mean())
    db = float(gb_f.mean())
    both = float(np.sum((ga > 0) & (gb > 0))) / 64.0
    union = float(np.sum((ga > 0) | (gb > 0))) / 64.0
    neither = float(np.sum((ga == 0) & (gb == 0))) / 64.0
    pba = both / da if da > 0 else 0.0
    pab = both / db if db > 0 else 0.0
    off_mask = (np.arange(64) % 4) == 2
    sync_mask = np.isin(np.arange(64) % 16, [3, 6, 11, 14])
    beat_mask = (np.arange(64) % 4) == 0
    da_off = float(ga_f[off_mask].mean())
    db_off = float(gb_f[off_mask].mean())
    da_sync = float(ga_f[sync_mask].mean())
    db_sync = float(gb_f[sync_mask].mean())
    da_beat = float(ga_f[beat_mask].mean())
    db_beat = float(gb_f[beat_mask].mean())

    resp_n = float(np.sum(ga[:-1] & gb[1:]) + np.sum(gb[:-1] & ga[1:]))
    resp = resp_n / 2.0 / 63.0

    pa = ga_f.reshape(4, 16).mean(axis=0)
    pb = gb_f.reshape(4, 16).mean(axis=0)
    pc = 0.0 if (pa.std() < 1e-6 or pb.std() < 1e-6) else _corr(pa, pb)

    la = np.log1p(fa.astype(np.float64))
    lb = np.log1p(fb.astype(np.float64))
    c = _corr(la, lb)
    ma, mb = float(la.mean()), float(lb.mean())
    sa, sb = float(la.std()), float(lb.std())
    na = (la - ma) / (sa + 1e-9)
    nb = (lb - mb) / (sb + 1e-9)
    joint = float(np.mean(np.minimum(na, nb)))
    lag1 = _corr(la[:63], lb[1:64])
    lag2 = _corr(lb[:63], la[1:64])

    return np.array([
        da, db, both, union, neither, pba, pab,
        da_off - db_off, da_sync - db_sync, da_beat - db_beat,
        resp, da * db, abs(da - db), pc,
        c, joint, lag1 + lag2, ma, mb,
    ], dtype=np.float32)


def build_rhythm_and_vocal_sidecars(manifest: dict, onsets_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Build rhythm_scores.json, rhythm_flux.f32, rhythm_mesh_model.json, and vocal_mix.json."""
    onsets_meta = json.loads((onsets_dir / "mix_compat_onsets.json").read_text())
    flux_all = np.fromfile(onsets_dir / "mix_compat_onsets.f32", dtype=np.float32).reshape(-1, 4, 64)
    onset_idx = onsets_meta["index"]

    phrases_order = []
    phrase_tids = []
    phrase_rows = []
    for kit in manifest.get("kits", []):
        tid = kit["track_id"]
        for p in kit.get("phrases", []):
            pid = p["id"]
            phrases_order.append(pid)
            phrase_tids.append(tid)
            phrase_rows.append(onset_idx[pid])

    flux = flux_all[np.asarray(phrase_rows, dtype=np.int64)]  # (N, 4, 64)
    (out_dir / "rhythm_flux.f32").write_bytes(flux.astype(np.float32).tobytes())

    pct = 62.0
    thresholds: dict[str, float] = {}
    grids = np.zeros_like(flux, dtype=np.uint8)
    for si, stem in enumerate(STEMS):
        vals = flux[:, si, :].reshape(-1)
        pos = vals[vals > 1e-4]
        th = float(np.percentile(pos, pct)) if len(pos) > 0 else 0.1
        thresholds[stem] = th
        grids[:, si, :] = (flux[:, si, :] >= th).astype(np.uint8)

    rhythm_phrases = []
    for i, pid in enumerate(phrases_order):
        entry = {"id": pid}
        for si, stem in enumerate(STEMS):
            entry[stem] = "".join("1" if x else "0" for x in grids[i, si].tolist())
        rhythm_phrases.append(entry)
    (out_dir / "rhythm_scores.json").write_text(json.dumps({
        "format": "wub-rhythm-scores-v1",
        "phrases": rhythm_phrases,
    }, separators=(",", ":")))

    # Build vocal_mix.json directly from vocals channel flux & grid
    vocal_doc = {}
    v_flux = flux[:, 3, :]
    v_grid = grids[:, 3, :]
    for i, pid in enumerate(phrases_order):
        vf = v_flux[i]
        vg = v_grid[i]
        tot = float(vf.mean())
        if tot < 1e-4:
            vocal_doc[pid] = {
                "vocal_edge_strength": [1.0, 1.0],
                "vocal_activity16": "0" * 64,
            }
            continue
        start_edge = float(vf[:2].mean())
        end_edge = float(vf[-2:].mean())
        s_clean = max(0.0, min(1.0, 1.0 - start_edge / (tot * 2.0 + 1e-6)))
        e_clean = max(0.0, min(1.0, 1.0 - end_edge / (tot * 2.0 + 1e-6)))
        vocal_doc[pid] = {
            "vocal_edge_strength": [round(s_clean, 4), round(e_clean, 4)],
            "vocal_activity16": "".join("1" if x else "0" for x in vg.tolist()),
        }
    (out_dir / "vocal_mix.json").write_text(json.dumps(vocal_doc, separators=(",", ":")))

    # Train 6 pairwise logistic models (real same-song pairs vs cross-song random pairs)
    rng = np.random.default_rng(SEED)
    tids_arr = np.asarray(phrase_tids)
    uniq_tids = sorted(set(phrase_tids))
    rng.shuffle(uniq_tids)
    val_tids = set(uniq_tids[: max(1, int(0.15 * len(uniq_tids)))])
    tr_idx = np.array([i for i, t in enumerate(phrase_tids) if t not in val_tids])
    va_idx = np.array([i for i, t in enumerate(phrase_tids) if t in val_tids])

    pair_models = {}
    pair_aucs = {}
    for sa, sb in RHYTHM_PAIRS:
        ia, ib = STEMS.index(sa), STEMS.index(sb)

        def _make_split(indices: np.ndarray, n_neg_mult: int = 3):
            xs, ys = [], []
            for p_i in indices:
                # positive: real same phrase if both stems non-silent (or any)
                xs.append(rhythm_pair_features(grids[p_i, ia], flux[p_i, ia], grids[p_i, ib], flux[p_i, ib]))
                ys.append(1)
                # cross-song negatives
                cand = indices[tids_arr[indices] != tids_arr[p_i]]
                if len(cand) > 0:
                    negs = rng.choice(cand, size=min(n_neg_mult, len(cand)), replace=False)
                    for n_i in negs:
                        xs.append(rhythm_pair_features(grids[p_i, ia], flux[p_i, ia], grids[n_i, ib], flux[n_i, ib]))
                        ys.append(0)
            return np.stack(xs), np.asarray(ys, dtype=np.int64)

        X_tr, y_tr = _make_split(tr_idx, 3)
        X_va, y_va = _make_split(va_idx, 3)
        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        std[std < 1e-6] = 1.0
        X_tr_n = (X_tr - mean) / std
        X_va_n = (X_va - mean) / std
        clf = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
        clf.fit(X_tr_n, y_tr)
        val_prob = clf.predict_proba(X_va_n)[:, 1]
        auc = float(roc_auc_score(y_va, val_prob)) if len(np.unique(y_va)) > 1 else 0.5
        pkey = f"{sa}-{sb}"
        pair_aucs[pkey] = round(auc, 4)
        pair_models[pkey] = {
            "coef": [round(float(v), 6) for v in clf.coef_[0].tolist()],
            "intercept": round(float(clf.intercept_[0]), 6),
            "feature_mean": [round(float(v), 6) for v in mean.tolist()],
            "feature_std": [round(float(v), 6) for v in std.tolist()],
        }

    mesh_doc = {
        "format": "wub-rhythm-mesh-model-v1",
        "feature_names": RHYTHM_FEAT_NAMES,
        "density_idx": [0, 1],
        "thresholds": thresholds,
        "pct": pct,
        "pairs": pair_models,
        "val_aucs": pair_aucs,
    }
    (out_dir / "rhythm_mesh_model.json").write_text(json.dumps(mesh_doc, separators=(",", ":")))
    print(f"[sidecar:rhythm] phrases={len(phrases_order)} pair_aucs={pair_aucs}", flush=True)
    return {"phrases": len(phrases_order), "pair_aucs": pair_aucs}


@torch.no_grad()
def build_syntax_successors(manifest: dict, dataset: dict, ranker_ckpt: Path, out_dir: Path, device: torch.device, top_k: int = 24) -> dict[str, Any]:
    """Build syntax_successors.json from trained taste_ranker cosine affinity."""
    ckpt = torch.load(ranker_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    mean = np.asarray(cfg["mean"], dtype=np.float32)
    std = np.asarray(cfg["std"], dtype=np.float32)
    fdim = len(mean)

    model = PairwiseRanker(
        feature_dim=fdim,
        d_model=int(cfg.get("d_model", 128)),
        n_layers=int(cfg.get("n_layers", 3)),
        n_heads=int(cfg.get("n_heads", 4)),
        max_len=int(cfg.get("max_len", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        init_scale=float(cfg.get("init_scale", 10.0)),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = {r["phrase_id"]: r for r in dataset.get("kit_phrases", [])}
    meta = []
    feats = []
    for kit in manifest.get("kits", []):
        for p in kit.get("phrases", []):
            pid = p["id"]
            if pid not in rows:
                continue
            f = np.nan_to_num(np.asarray(rows[pid]["features"][:fdim], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(f)
            meta.append((pid, kit["track_id"], kit.get("style", ""), str(p.get("section", "GROOVE_A")).upper()))

    X = torch.from_numpy(np.stack(feats)).to(device)
    # Encode as single-step context and as candidate
    ctx_emb = F.normalize(model.encode_context(X.unsqueeze(1), torch.ones((len(X), 1), dtype=torch.bool, device=device)), dim=-1)
    cand_emb = F.normalize(model.encode_candidate(X), dim=-1)
    sim = (ctx_emb @ cand_emb.T).cpu().numpy()  # (N, N) in [-1, 1]
    prob = 1.0 / (1.0 + np.exp(-3.0 * sim))  # monotonic [0, 1]

    styles = [m[2] for m in meta]
    sections = [m[3] for m in meta]
    tids = [m[1] for m in meta]
    pids = [m[0] for m in meta]

    successors = {}
    for i, pid in enumerate(pids):
        row_scores = prob[i]
        # Only score valid cross-track same-style successors
        same_style_idx = [j for j in range(len(pids)) if tids[j] != tids[i] and styles[j] == styles[i]]
        same_sec_idx = [j for j in same_style_idx if sections[j] == sections[i]]
        same_sec_idx.sort(key=lambda j: row_scores[j], reverse=True)
        same_style_idx.sort(key=lambda j: row_scores[j], reverse=True)
        successors[pid] = {
            "same_section_top": [{"to": pids[j], "score": round(float(row_scores[j]), 4)} for j in same_sec_idx[:top_k]],
            "overall_top": [{"to": pids[j], "score": round(float(row_scores[j]), 4)} for j in same_style_idx[:top_k]],
        }
    (out_dir / "syntax_successors.json").write_text(json.dumps({"successors": successors}, separators=(",", ":")))
    print(f"[sidecar:syntax] anchors={len(successors)}", flush=True)
    return {"anchors": len(successors)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--onsets-dir", required=True)
    ap.add_argument("--ranker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arc-epochs", type=int, default=18)
    ap.add_argument("--prog-epochs", type=int, default=24)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    dataset = json.loads(Path(args.dataset).read_text())
    device = torch.device(args.device)

    arc_res = build_arc_sidecars(dataset, out_dir, device, epochs=args.arc_epochs)
    prog_res = build_progression_sidecars(manifest, dataset, out_dir, device, epochs=args.prog_epochs)
    rhy_res = build_rhythm_and_vocal_sidecars(manifest, Path(args.onsets_dir), out_dir)
    syn_res = build_syntax_successors(manifest, dataset, Path(args.ranker), out_dir, device)

    sidecars_flag = {
        "version": 1,
        "arc_model": True,
        "arc_context_features": True,
        "syntax_successors": True,
        "rhythm_scores": True,
        "rhythm_mesh_model": True,
        "rhythm_flux": True,
        "progression_model": True,
        "progression_context": True,
        "vocal_mix": True,
        "section_model": True,
    }
    (out_dir / "sidecars.json").write_text(json.dumps(sidecars_flag, indent=2) + "\n")
    summary = {"arc": arc_res, "progression": prog_res, "rhythm_vocal": rhy_res, "syntax": syn_res}
    (out_dir / "sidecars_report.json").write_text(json.dumps(summary, indent=2))
    print(f"[sidecar:done] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
