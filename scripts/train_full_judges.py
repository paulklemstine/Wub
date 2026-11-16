#!/usr/bin/env python3
"""Train full-corpus Mix + Per-Stem AudioJudges (and full-corpus RhythmMesh) on GPU.

Mines 4-bar segments directly from `<stems>/<tid>/{drums,bass,other,vocals}.wav`
aligned to `processed/<tid>.json` downbeats (zero throwaway .opus encoding),
extracts 64x128 log-mel patches matching `scripts/build_stem_bars.py`, trains five
two-tower `AudioJudge` CNNs (`audio_judge.pt` mix teacher + 4 per-stem judges)
with hard same-style/close-BPM cross-song negatives + SpecAugment on a strict
track-disjoint split, and optionally fits `rhythm_mesh_model.json` on active
full-corpus 64-step onset flux pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audio_judge_model import AudioJudge  # noqa: E402
from scripts.build_sidecars import (  # noqa: E402
    RHYTHM_FEAT_NAMES,
    RHYTHM_PAIRS,
    rhythm_pair_features,
)
from wub_core.section_arcs import bar_features, segment_sections  # noqa: E402

STEMS = ("drums", "bass", "other", "vocals")
ALL_CHANNELS = ("drums", "bass", "other", "vocals", "mix")
MEL_CFG = {
    "sr": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 64,
    "frames": 128,
}
MIN_BAR_SEC = 3.0
MAX_BAR_SEC = 20.0
MIN_RMS_DB = -52.0


def _resample_np(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(y, sr_out // g, sr_in // g).astype(np.float32)


def _logmel_from_pcm(y22: np.ndarray) -> np.ndarray | None:
    if y22.size < MEL_CFG["sr"] // 4:
        return None
    mel = librosa.feature.melspectrogram(
        y=y22,
        sr=MEL_CFG["sr"],
        n_fft=MEL_CFG["n_fft"],
        hop_length=MEL_CFG["hop_length"],
        n_mels=MEL_CFG["n_mels"],
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=1.0)
    frames = db.shape[1]
    if frames == 0:
        return None
    edges = np.linspace(0, frames, MEL_CFG["frames"] + 1)
    pooled = np.empty((MEL_CFG["n_mels"], MEL_CFG["frames"]), dtype=np.float32)
    for i in range(MEL_CFG["frames"]):
        lo = int(edges[i])
        hi = max(lo + 1, int(edges[i + 1]))
        pooled[:, i] = db[:, lo:hi].mean(axis=1)
    return pooled.astype(np.float16)


def _flux64_from_pcm(y22: np.ndarray) -> np.ndarray:
    if y22.size < 1024:
        return np.zeros(64, dtype=np.float32)
    S = np.abs(librosa.stft(y22, n_fft=1024, hop_length=256))
    mag = np.log1p(S)
    diff = np.diff(mag, axis=1, prepend=mag[:, :1])
    diff[diff < 0] = 0.0
    flux = diff.mean(axis=0)
    edges = np.linspace(0, len(flux), 65)
    out = np.zeros(64, dtype=np.float32)
    for k in range(64):
        lo = int(edges[k])
        hi = max(lo + 1, int(edges[k + 1]))
        out[k] = float(flux[lo:min(hi, len(flux))].mean())
    return out


def _extract_track_worker(job: tuple[str, str, int]) -> dict[str, Any] | None:
    proc_path_str, stems_dir_str, max_phrases = job
    proc_path = Path(proc_path_str)
    try:
        doc = json.loads(proc_path.read_text())
    except Exception:
        return None

    tid = str(doc.get("track_id") or proc_path.stem)
    stem_dir = Path(stems_dir_str) / tid
    stem_files: dict[str, Path] = {}
    for st in STEMS:
        p = stem_dir / f"{st}.wav"
        if not p.is_file():
            # support mp3/flac/opus fallback
            found = None
            for ext in (".flac", ".opus", ".ogg", ".mp3"):
                cand = stem_dir / f"{st}{ext}"
                if cand.is_file():
                    found = cand
                    break
            if found is None:
                return None
            p = found
        stem_files[st] = p

    tatums = doc.get("tatums") or []
    n_bars = len(tatums) // 16
    if n_bars < 8:
        return None

    try:
        bars = bar_features(tatums)
        spans = segment_sections(bars)
    except Exception:
        spans = []

    sec_at_bar: dict[int, tuple[str, float]] = {}
    for sp in spans:
        sec_name = str(sp.get("section", "GROOVE_A")).upper()
        en = float(sp.get("mean_energy", 0.5))
        for b in range(int(sp.get("start_bar", 0)), int(sp.get("end_bar", 0))):
            sec_at_bar[b] = (sec_name, en)

    # Candidate 4-bar downbeats across all section boundaries + 4-bar grid
    starts = sorted({int(sp.get("start_bar", 0)) for sp in spans if int(sp.get("start_bar", 0)) + 4 <= n_bars}
                    | set(range(0, n_bars - 3, 4)))
    if not starts:
        return None
    if len(starts) > max_phrases:
        idxs = np.linspace(0, len(starts) - 1, max_phrases, dtype=int)
        starts = [starts[int(i)] for i in idxs]

    bpm = float(doc.get("bpm", 124.0) or 124.0)
    key = str(doc.get("key", "C minor") or "C minor")
    style = str(doc.get("style", "unknown") or "unknown")

    metas = []
    mels: dict[str, list[np.ndarray]] = {ch: [] for ch in ALL_CHANNELS}
    fluxes: list[np.ndarray] = []

    infos = {}
    try:
        for st, p in stem_files.items():
            infos[st] = sf.info(str(p))
    except Exception:
        return None

    for sb in starts:
        i0 = sb * 16
        i1 = (sb + 4) * 16
        if i1 >= len(tatums):
            continue
        t0 = float(tatums[i0]["start"])
        t1 = float(tatums[i1]["start"])
        dur = t1 - t0
        if dur < MIN_BAR_SEC or dur > MAX_BAR_SEC:
            continue

        stem_mels: dict[str, np.ndarray] = {}
        stem_flux: list[np.ndarray] = []
        mix_pcm: np.ndarray | None = None
        ok = True
        for st in STEMS:
            info = infos[st]
            sr_file = int(info.samplerate)
            f0 = max(0, int(round(t0 * sr_file)))
            n_fr = int(round(dur * sr_file))
            if f0 + n_fr > info.frames:
                ok = False
                break
            pcm, _ = sf.read(str(stem_files[st]), start=f0, frames=n_fr, dtype="float32", always_2d=True)
            mono = pcm.mean(axis=1)
            rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(mono * mono))) + 1e-9)
            y22 = _resample_np(mono, sr_file, MEL_CFG["sr"])
            lm = _logmel_from_pcm(y22)
            if lm is None:
                ok = False
                break
            stem_mels[st] = lm
            stem_flux.append(_flux64_from_pcm(y22))
            if mix_pcm is None:
                mix_pcm = y22.copy()
            else:
                n_min = min(len(mix_pcm), len(y22))
                mix_pcm = mix_pcm[:n_min] + y22[:n_min]
            if rms_db < MIN_RMS_DB:
                # Keep silent flag cleanly in meta
                pass

        if not ok or mix_pcm is None:
            continue
        mix_lm = _logmel_from_pcm(mix_pcm)
        if mix_lm is None:
            continue

        sec_name, sec_en = sec_at_bar.get(sb, ("GROOVE_A", 0.5))
        pid = f"{tid}_b{sb:03d}"
        metas.append({
            "phrase_id": pid,
            "track_id": tid,
            "start_bar": int(sb),
            "bpm": bpm,
            "key": key,
            "style": style,
            "section": sec_name,
            "energy": float(sec_en),
            "duration": float(dur),
        })
        for st in STEMS:
            mels[st].append(stem_mels[st])
        mels["mix"].append(mix_lm)
        fluxes.append(np.stack(stem_flux, axis=0))  # (4, 64)

    if len(metas) < 2:
        return None
    return {"metas": metas, "mels": mels, "flux": fluxes}


def extract_full_corpus(
    processed_dir: Path,
    stems_dir: Path,
    out_dir: Path,
    max_tracks: int,
    max_phrases_per_track: int,
    workers: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ready_flag = out_dir / "_READY.json"
    if ready_flag.is_file():
        print(f"[full_bars] reusing cached extraction at {out_dir}", flush=True)
        return out_dir

    proc_files = sorted(processed_dir.glob("*.json"))
    if max_tracks > 0:
        proc_files = proc_files[:max_tracks]
    jobs = [(str(p), str(stems_dir), max_phrases_per_track) for p in proc_files]
    print(f"[full_bars] scanning {len(jobs)} processed tracks with {workers} workers...", flush=True)

    all_metas: list[dict[str, Any]] = []
    all_mels: dict[str, list[np.ndarray]] = {ch: [] for ch in ALL_CHANNELS}
    all_flux: list[np.ndarray] = []
    t0 = time.time()
    tracks_ok = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_extract_track_worker, jobs, chunksize=8)):
            if res is not None:
                tracks_ok += 1
                all_metas.extend(res["metas"])
                for ch in ALL_CHANNELS:
                    all_mels[ch].extend(res["mels"][ch])
                all_flux.extend(res["flux"])
            if (i + 1) % 200 == 0:
                print(f"[full_bars] {i + 1}/{len(jobs)} tracks scanned -> {tracks_ok} usable, {len(all_metas)} phrases ({time.time() - t0:.1f}s)", flush=True)

    print(f"[full_bars] extracted {len(all_metas)} phrases across {tracks_ok} tracks in {time.time() - t0:.1f}s", flush=True)

    # Save each channel in train_stem_judge-compatible layout
    bar_index = {
        "mel": MEL_CFG,
        "mel_file": "mel.npy",
        "count": len(all_metas),
        "phrase_id": [m["phrase_id"] for m in all_metas],
        "track_id": [m["track_id"] for m in all_metas],
        "start_bar": [m["start_bar"] for m in all_metas],
        "bpm": [m["bpm"] for m in all_metas],
        "key": [m["key"] for m in all_metas],
        "style": [m["style"] for m in all_metas],
        "section": [m["section"] for m in all_metas],
        "energy": [m["energy"] for m in all_metas],
        "duration": [m["duration"] for m in all_metas],
    }
    for ch in ALL_CHANNELS:
        ch_dir = out_dir / ch
        ch_dir.mkdir(parents=True, exist_ok=True)
        arr = np.stack(all_mels[ch]).astype(np.float16)
        np.save(ch_dir / "mel.npy", arr)
        (ch_dir / "bar_index.json").write_text(json.dumps(bar_index))

    np.savez(
        out_dir / "full_rhythm_flux.npz",
        flux=np.stack(all_flux).astype(np.float32),
        track_id=np.asarray(bar_index["track_id"]),
    )
    ready_flag.write_text(json.dumps({"tracks": tracks_ok, "phrases": len(all_metas)}))
    return out_dir


def _gpu_spec_aug(x: torch.Tensor) -> torch.Tensor:
    """Vectorized SpecAugment on (B, 1, 64, 128) mel tensor on GPU."""
    B, _, F_dim, T_dim = x.shape
    out = x.clone()
    fills = out.mean(dim=(2, 3), keepdim=True)
    # Mask one frequency band on ~65% of items
    mask_f = (torch.rand(B, device=x.device) < 0.65)
    if mask_f.any():
        f0 = torch.randint(0, 56, (B,), device=x.device)
        fw = torch.randint(2, 9, (B,), device=x.device)
        f_idx = torch.arange(F_dim, device=x.device).view(1, 1, F_dim, 1)
        m_f = mask_f.view(B, 1, 1, 1) & (f_idx >= f0.view(B, 1, 1, 1)) & (f_idx < (f0 + fw).view(B, 1, 1, 1))
        out = torch.where(m_f, fills, out)
    # Mask one time block on ~65% of items
    mask_t = (torch.rand(B, device=x.device) < 0.65)
    if mask_t.any():
        t0 = torch.randint(0, 114, (B,), device=x.device)
        tw = torch.randint(4, 15, (B,), device=x.device)
        t_idx = torch.arange(T_dim, device=x.device).view(1, 1, 1, T_dim)
        m_t = mask_t.view(B, 1, 1, 1) & (t_idx >= t0.view(B, 1, 1, 1)) & (t_idx < (t0 + tw).view(B, 1, 1, 1))
        out = torch.where(m_t, fills, out)
    return out


def train_one_judge(
    ch_dir: Path,
    channel: str,
    out_ckpt: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    bank_tids: set[str],
    seed: int = 404,
) -> dict[str, Any]:
    index = json.loads((ch_dir / "bar_index.json").read_text())
    mels_np = np.load(ch_dir / "mel.npy")
    mels_dev = torch.from_numpy(mels_np).to(device)  # float16 (N, 64, 128) ~287 MB fits easily in VRAM
    del mels_np

    tids = index["track_id"]
    bars = index["start_bar"]
    styles = index["style"]
    bpms = index["bpm"]

    by_track: dict[str, list[tuple[int, int]]] = {}
    for row_i, (tid, sb) in enumerate(zip(tids, bars)):
        by_track.setdefault(str(tid), []).append((int(sb), row_i))
    for v in by_track.values():
        v.sort()

    unique_tids = sorted(by_track.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_tids)
    n_val = max(4, int(round(0.12 * len(unique_tids))))
    val_tids = set(unique_tids[:n_val])

    def build_split_arrays(split_tids: set[str], rng_loc: random.Random):
        pos_pairs = []
        pool_all = []
        style_buckets: dict[str, list[int]] = {}
        for tid in sorted(split_tids):
            seq = by_track[tid]
            for j in range(len(seq) - 1):
                pos_pairs.append((seq[j][1], seq[j + 1][1]))
            for _, row_i in seq:
                pool_all.append(row_i)
                style_buckets.setdefault(str(styles[row_i]), []).append(row_i)

        n_pos = len(pos_pairs)
        a_pos = np.array([p[0] for p in pos_pairs], dtype=np.int64)
        b_pos = np.array([p[1] for p in pos_pairs], dtype=np.int64)
        # Fixed-width candidate tables per anchor (32 hard + 32 any) for O(1) vectorized negative sampling
        K = 32
        hard_mat = np.empty((n_pos, K), dtype=np.int64)
        any_mat = np.empty((n_pos, K), dtype=np.int64)
        for idx_p, a_idx in enumerate(a_pos):
            t_a = tids[a_idx]
            st_a = str(styles[a_idx])
            bpm_a = float(bpms[a_idx])
            sb = style_buckets.get(st_a, [])
            same_st = []
            if sb:
                sample_sb = sb if len(sb) <= 64 else rng_loc.sample(sb, 64)
                same_st = [r for r in sample_sb if tids[r] != t_a and abs(float(bpms[r]) - bpm_a) <= 10.0]
                if not same_st:
                    same_st = [r for r in sample_sb if tids[r] != t_a]
            cand_any = []
            for _ in range(48):
                r = pool_all[rng_loc.randrange(len(pool_all))]
                if tids[r] != t_a:
                    cand_any.append(r)
                if len(cand_any) >= K:
                    break
            if not cand_any:
                cand_any = [int(a_idx)]
            if not same_st:
                same_st = cand_any
            hard_mat[idx_p] = np.array([same_st[k % len(same_st)] for k in range(K)], dtype=np.int64)
            any_mat[idx_p] = np.array([cand_any[k % len(cand_any)] for k in range(K)], dtype=np.int64)
        return (
            torch.from_numpy(a_pos).to(device),
            torch.from_numpy(b_pos).to(device),
            torch.from_numpy(hard_mat).to(device),
            torch.from_numpy(any_mat).to(device),
        )

    tr_tids = set(unique_tids) - val_tids
    tr_a, tr_b, tr_hard, tr_any = build_split_arrays(tr_tids, random.Random(seed + 7))
    va_a, va_b, va_hard, va_any = build_split_arrays(val_tids, random.Random(seed + 999))

    # Deterministic fixed validation pairs (1 pos + 2 neg per val pair) for zero-variance tracking
    with torch.no_grad():
        torch.manual_seed(seed + 1234)
        va_n = len(va_a)
        va_neg1 = va_hard[torch.arange(va_n, device=device), torch.randint(0, 32, (va_n,), device=device)]
        va_neg2 = va_any[torch.arange(va_n, device=device), torch.randint(0, 32, (va_n,), device=device)]
        va_pair_a = torch.cat([va_a, va_a, va_a], dim=0)
        va_pair_b = torch.cat([va_b, va_neg1, va_neg2], dim=0)
        va_pair_y = torch.cat([
            torch.ones(va_n, dtype=torch.float32),
            torch.zeros(va_n * 2, dtype=torch.float32),
        ]).numpy()

    torch.manual_seed(seed)
    model = AudioJudge(emb_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    bce = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"

    best_auc = -1.0
    best_state = None
    tr_n = len(tr_a)
    pair_batch = max(32, batch_size // 2)  # 1 pos + 2 neg -> 3*pair_batch items per optimizer step

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(tr_n, device=device)
        tot_loss = 0.0
        tot_items = 0
        for s in range(0, tr_n, pair_batch):
            idx = perm[s:s + pair_batch]
            cur_bs = len(idx)
            a_pos_i = tr_a[idx]
            b_pos_i = tr_b[idx]
            use_h1 = torch.rand(cur_bs, device=device) < 0.55
            col1 = torch.randint(0, 32, (cur_bs,), device=device)
            b_neg1_i = torch.where(use_h1, tr_hard[idx, col1], tr_any[idx, col1])
            use_h2 = torch.rand(cur_bs, device=device) < 0.55
            col2 = torch.randint(0, 32, (cur_bs,), device=device)
            b_neg2_i = torch.where(use_h2, tr_hard[idx, col2], tr_any[idx, col2])

            ba_idx = torch.cat([a_pos_i, a_pos_i, a_pos_i], dim=0)
            bb_idx = torch.cat([b_pos_i, b_neg1_i, b_neg2_i], dim=0)
            by = torch.cat([
                torch.ones(cur_bs, device=device),
                torch.zeros(cur_bs * 2, device=device),
            ])

            xa = _gpu_spec_aug(mels_dev[ba_idx].unsqueeze(1).float())
            xb = _gpu_spec_aug(mels_dev[bb_idx].unsqueeze(1).float())

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xa, xb)
                loss = bce(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.item()) * len(by)
            tot_items += len(by)
        sched.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for s in range(0, len(va_pair_a), 512):
                xa = mels_dev[va_pair_a[s:s + 512]].unsqueeze(1).float()
                xb = mels_dev[va_pair_b[s:s + 512]].unsqueeze(1).float()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xa, xb)
                scores.append(torch.sigmoid(logits).float().cpu().numpy())
        auc = float(roc_auc_score(va_pair_y, np.concatenate(scores)))
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"[judge:{channel}] ep {ep+1}/{epochs} loss={tot_loss/max(1,tot_items):.4f} val_auc={auc:.4f} best={best_auc:.4f}", flush=True)

    del mels_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "val_auc": best_auc,
        "cfg": {"emb_dim": 128, "channel": channel, "epochs": epochs, "tracks": len(unique_tids), "phrases": len(tids)},
    }, out_ckpt)
    print(f"[judge:{channel}] SAVED {out_ckpt} best_val_auc={best_auc:.4f} (tracks={len(unique_tids)}, phrases={len(tids)})", flush=True)
    return {"channel": channel, "val_auc": best_auc, "tracks": len(unique_tids), "phrases": len(tids)}


def fit_full_rhythm_mesh(
    flux_npz: Path,
    bank_onsets_dir: Path,
    bank_manifest_path: Path,
    out_mesh_path: Path,
    seed: int = 123,
) -> dict[str, Any]:
    """Fit 6 pairwise logistic models on full-corpus active stems and eval on held-out bank tracks."""
    data = np.load(flux_npz)
    flux_full = data["flux"]  # (N, 4, 64)
    tids_full = data["track_id"]

    # Bank threshold alignment (use the shipped thresholds from bank_onsets so runtime matches exact pct!)
    bank_manifest = json.loads(bank_manifest_path.read_text())
    bank_meta = json.loads((bank_onsets_dir / "mix_compat_onsets.json").read_text())
    bank_flux = np.fromfile(bank_onsets_dir / "mix_compat_onsets.f32", dtype=np.float32).reshape(-1, 4, 64)
    idx_map = bank_meta["index"]
    bank_rows, bank_tids = [], []
    for kit in bank_manifest.get("kits", []):
        for p in kit.get("phrases", []):
            pid = p["id"]
            if pid in idx_map:
                bank_rows.append(idx_map[pid])
                bank_tids.append(kit["track_id"])
    b_flux = bank_flux[np.asarray(bank_rows, dtype=np.int64)]
    pct = 62.0
    thresholds: dict[str, float] = {}
    for si, stem in enumerate(STEMS):
        vals = b_flux[:, si, :].reshape(-1)
        pos = vals[vals > 1e-4]
        thresholds[stem] = float(np.percentile(pos, pct)) if len(pos) > 0 else 0.1

    # Full-corpus flux scale calibration: rescale each stem channel so percentile pct matches bank threshold
    grids_full = np.zeros_like(flux_full, dtype=np.uint8)
    for si, stem in enumerate(STEMS):
        vals = flux_full[:, si, :].reshape(-1)
        pos = vals[vals > 1e-4]
        full_th = float(np.percentile(pos, pct)) if len(pos) > 0 else thresholds[stem]
        scale = thresholds[stem] / max(full_th, 1e-6)
        flux_full[:, si, :] *= scale
        grids_full[:, si, :] = (flux_full[:, si, :] >= thresholds[stem]).astype(np.uint8)

    b_grids = np.zeros_like(b_flux, dtype=np.uint8)
    for si, stem in enumerate(STEMS):
        b_grids[:, si, :] = (b_flux[:, si, :] >= thresholds[stem]).astype(np.uint8)

    # Disjoint val split matching build_sidecars seed=404
    rng = np.random.default_rng(404)
    uniq_bank = sorted(set(bank_tids))
    rng.shuffle(uniq_bank)
    val_bank_tids = set(uniq_bank[: max(1, int(0.15 * len(uniq_bank)))])
    tr_full_mask = np.array([str(t) not in val_bank_tids for t in tids_full])
    tr_full_idx = np.nonzero(tr_full_mask)[0]
    va_bank_idx = np.array([i for i, t in enumerate(bank_tids) if t in val_bank_tids])
    btids_arr = np.asarray(bank_tids)

    pair_models, pair_aucs = {}, {}
    for sa, sb in RHYTHM_PAIRS:
        ia, ib = STEMS.index(sa), STEMS.index(sb)
        act_tr = (grids_full[tr_full_idx, ia].mean(axis=1) >= 0.02) & (grids_full[tr_full_idx, ib].mean(axis=1) >= 0.02)
        tr_pool = tr_full_idx[act_tr]
        xs_tr, ys_tr = [], []
        for p_i in tr_pool:
            xs_tr.append(rhythm_pair_features(grids_full[p_i, ia], flux_full[p_i, ia], grids_full[p_i, ib], flux_full[p_i, ib]))
            ys_tr.append(1)
            cand = tr_pool[tids_full[tr_pool] != tids_full[p_i]]
            if len(cand) > 0:
                negs = rng.choice(cand, size=min(4, len(cand)), replace=False)
                for n_i in negs:
                    xs_tr.append(rhythm_pair_features(grids_full[p_i, ia], flux_full[p_i, ia], grids_full[n_i, ib], flux_full[n_i, ib]))
                    ys_tr.append(0)
        X_tr = np.stack(xs_tr)
        y_tr = np.asarray(ys_tr, dtype=np.int64)

        # Validate on the EXACT held-out bank tracks (active subset)
        act_va = (b_grids[va_bank_idx, ia].mean(axis=1) >= 0.02) & (b_grids[va_bank_idx, ib].mean(axis=1) >= 0.02)
        va_pool = va_bank_idx[act_va] if act_va.sum() >= 20 else va_bank_idx
        xs_va, ys_va = [], []
        for p_i in va_pool:
            xs_va.append(rhythm_pair_features(b_grids[p_i, ia], b_flux[p_i, ia], b_grids[p_i, ib], b_flux[p_i, ib]))
            ys_va.append(1)
            cand = va_pool[btids_arr[va_pool] != btids_arr[p_i]]
            if len(cand) > 0:
                negs = rng.choice(cand, size=min(4, len(cand)), replace=False)
                for n_i in negs:
                    xs_va.append(rhythm_pair_features(b_grids[p_i, ia], b_flux[p_i, ia], b_grids[n_i, ib], b_flux[n_i, ib]))
                    ys_va.append(0)
        X_va = np.stack(xs_va)
        y_va = np.asarray(ys_va, dtype=np.int64)

        mean = X_tr.mean(axis=0)
        std = X_tr.std(axis=0)
        std[std < 1e-6] = 1.0
        X_tr_n = (X_tr - mean) / std
        X_va_n = (X_va - mean) / std
        clf = LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X_tr_n, y_tr)
        prob = clf.predict_proba(X_va_n)[:, 1]
        auc = float(roc_auc_score(y_va, prob)) if len(np.unique(y_va)) > 1 else 0.5
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
        "train_phrases": int(len(tr_full_idx)),
    }
    out_mesh_path.write_text(json.dumps(mesh_doc, separators=(",", ":")))
    print(f"[rhythm_mesh:full] val_aucs={pair_aucs}", flush=True)
    return pair_aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--stems", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--bars-out", type=Path, required=True)
    ap.add_argument("--models-out", type=Path, required=True)
    ap.add_argument("--rhythm-out", type=Path, default=None)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--max-phrases", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=38)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    extract_full_corpus(
        args.processed,
        args.stems,
        args.bars_out,
        max_tracks=args.max_tracks,
        max_phrases_per_track=args.max_phrases,
        workers=args.workers,
    )

    bank_manifest = json.loads((args.bank_dir / "kit_manifest.json").read_text())
    bank_tids = {str(k["track_id"]) for k in bank_manifest.get("kits", [])}
    device = torch.device(args.device)

    results = {}
    for ch, out_name in [
        ("mix", "audio_judge.pt"),
        ("drums", "stem_judge_drums.pt"),
        ("bass", "stem_judge_bass.pt"),
        ("other", "stem_judge_other.pt"),
        ("vocals", "stem_judge_vocals.pt"),
    ]:
        res = train_one_judge(
            args.bars_out / ch,
            ch,
            args.models_out / out_name,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            bank_tids=bank_tids,
        )
        results[ch] = res

    if args.rhythm_out is not None:
        results["rhythm_mesh"] = fit_full_rhythm_mesh(
            args.bars_out / "full_rhythm_flux.npz",
            args.bank_dir,
            args.bank_dir / "kit_manifest.json",
            args.rhythm_out,
        )

    (args.models_out / "full_judges_report.json").write_text(json.dumps(results, indent=2))
    print("[full_judges] SUMMARY:", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
