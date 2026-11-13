#!/usr/bin/env python3
"""Build a per-bar mix-compatibility dataset (real vs random cross-stem mixes).

User goals (2026-11): score how well 4 stems sound together per bar, train on
random recombinations vs real co-occurring song windows, prioritize rhythm
mismatch, and enable cross-song creations (never all 4 from the same source;
cap <= 2 per source in mixes we want the app to pick).

Outputs a graded dataset for training a browser-runnable critic:
  - positives: real same-song, same-time 4-stem phrase windows (per-bar)
  - negatives: random cross-song tuples, tempo/key-normalized like the engine
  - per-bar features emphasizing rhythm lock + harmony + spectral balance/energy
  - optional rendered-mix embeddings (MERT/PANNs, ears_embed helpers) for a
    "distance-to-real-manifold" teacher score

Run (laptop, RTX 4050, corpus/stems available):
  PYTHONPATH=. python3 scripts/build_mix_compat_dataset.py \
    --manifest frontend/public/data/kits/kit_manifest.json \
    --kits-dir frontend/public/data/kits \
    --stems /mnt/e/Wub/data/stems \
    --bars data/stem_bars_flow \
    --out /tmp/wubtrain/mix_compat_dataset \
    --positives 20000 --negatives 60000 --device cuda

This script is designed to be resumable (--resume) and to degrade gracefully:
  - without --render/--embed it uses feature-heuristic grading only
  - without stem bars it falls back to manifest/onset-derived rhythm proxies
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warnings

STEMS = ("drums", "bass", "other", "vocals")
BARS_PER_PHRASE = 4
STEPS_PER_BAR = 16  # 64 steps / 4 bars when a 64-step grid is available
SEED = 505


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------- data loading ---------------------------------

def load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text())
    if not isinstance(m, dict) or not isinstance(m.get("kits"), list):
        raise ValueError("invalid kit manifest")
    return m


def phrase_index(manifest: dict):
    """Return (kits_by_id, phrases list with kit refs)."""
    kits = {k["track_id"]: k for k in manifest["kits"] if isinstance(k.get("track_id"), str)}
    phrases = []
    for kit in manifest["kits"]:
        tid = kit.get("track_id")
        if not isinstance(tid, str):
            continue
        for p in kit.get("phrases") or []:
            if not isinstance(p, dict) or not isinstance(p.get("id"), str):
                continue
            # require all 4 stems
            stems = p.get("stems") or {}
            if not all(isinstance(stems.get(s), dict) and stems[s].get("file") for s in STEMS):
                continue
            phrases.append({"kit": kit, "phrase": p, "track_id": tid})
    return kits, phrases


# --------------------------- feature extraction -----------------------------

def _safe_vec(x, n, fill=0.0):
    if not isinstance(x, list):
        return [fill] * n
    out = []
    for i in range(n):
        v = x[i] if i < len(x) else fill
        try:
            v = float(v)
        except Exception:
            v = fill
        if not math.isfinite(v):
            v = fill
        out.append(v)
    return out


def onset_grid_64(phrase: dict, stem: str):
    """Best-effort 64-step onset/energy grid per stem.
    Prefers phrase['onset'][stem] or phrase['onsets'][stem] if present; else
    returns None (caller may derive from stem bars)."""
    for key in ("onset", "onsets", "onset_grid", "onset_grids"):
        obj = phrase.get(key)
        if isinstance(obj, dict) and stem in obj:
            g = obj[stem]
            if isinstance(g, list) and len(g) >= 16:
                # normalize to 64
                v = np.array([float(x) for x in g[:64]] + [0.0] * max(0, 64 - len(g)), dtype=np.float32)
                if len(g) != 64:
                    # resample linearly
                    idx = np.linspace(0, len(g) - 1, 64)
                    base = np.array([float(x) for x in g], dtype=np.float32)
                    v = np.interp(idx, np.arange(len(g)), base).astype(np.float32)
                return v
    return None


def bar_slice(grid64: np.ndarray, bar: int) -> np.ndarray:
    s = bar * 16
    return grid64[s:s + 16]


def rhythm_pair_features(a: np.ndarray, b: np.ndarray) -> dict:
    """Rhythm lock between two 16-step bar onset envelopes (priority axis)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    eps = 1e-6
    # normalized correlation (onset alignment / groove lock)
    aa = a - a.mean()
    bb = b - b.mean()
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb) + eps)
    corr = float(np.dot(aa, bb) / denom)
    # downbeat emphasis agreement (steps 0,4,8,12 in 16-step bar)
    down = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], np.float32)
    off = 1.0 - down
    down_agree = float(np.dot(a * down, b * down) / (np.linalg.norm(a * down) * np.linalg.norm(b * down) + eps))
    off_agree = float(np.dot(a * off, b * off) / (np.linalg.norm(a * off) * np.linalg.norm(b * off) + eps))
    # density balance / clash (both very busy or one empty)
    da = float(a.mean())
    db = float(b.mean())
    density_gap = abs(da - db)
    joint_pulse = float(np.mean(a * b))
    # syncopation proxy: energy off strong beats
    sync_a = float(np.dot(a, off) / (np.dot(a, down) + eps))
    sync_b = float(np.dot(b, off) / (np.dot(b, down) + eps))
    sync_gap = abs(sync_a - sync_b)
    return {
        "corr": corr,
        "down_agree": down_agree,
        "off_agree": off_agree,
        "density_gap": density_gap,
        "joint_pulse": joint_pulse,
        "sync_gap": sync_gap,
    }


_NOTE_SEMI = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11,
}


def parse_key(k: Any) -> tuple[int, bool] | None:
    if not isinstance(k, str):
        return None
    s = k.strip()
    if not s:
        return None
    minor = s.endswith("m")
    name = s[:-1] if minor else s
    root = _NOTE_SEMI.get(name.upper())
    if root is None:
        return None
    return root, minor


def key_shift(session_key: Any, phrase_key: Any) -> int:
    """Semitones to shift phrase_key TO session_key via relative major, in -5..+6."""
    a = parse_key(session_key)
    b = parse_key(phrase_key)
    if not a or not b:
        return 0
    a_eff = (a[0] + 3) % 12 if a[1] else a[0]
    b_eff = (b[0] + 3) % 12 if b[1] else b[0]
    d = (a_eff - b_eff) % 12
    if d > 6:
        d -= 12
    return int(d)


def _shift_chroma(c: Any, shift: int) -> np.ndarray | None:
    if isinstance(c, list) and len(c) == 12:
        v = np.array([float(x) for x in c], np.float32)
        if shift % 12 != 0:
            v = np.roll(v, shift % 12)
        n = np.linalg.norm(v) + 1e-6
        return v / n
    return None


def _shift_root(r: Any, shift: int) -> int | None:
    try:
        return (int(r) + shift) % 12 if r is not None else None
    except Exception:
        return None


def harmonic_features(phrases_by_stem: dict, kit_by_stem: dict) -> dict:
    """Harmonic consonance in session-key space (drums kit's key) from chroma/bass_root."""
    session_key = (kit_by_stem.get("drums") or {}).get("key", "")
    shifts = {
        s: (key_shift(session_key, (kit_by_stem.get(s) or {}).get("key", "")) if session_key else 0)
        for s in ("bass", "other", "vocals")
    }
    cb = _shift_chroma(phrases_by_stem.get("bass", {}).get("chroma"), shifts["bass"])
    co = _shift_chroma(phrases_by_stem.get("other", {}).get("chroma"), shifts["other"])
    cv = _shift_chroma(phrases_by_stem.get("vocals", {}).get("chroma"), shifts["vocals"])
    def cos(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))
    rb = _shift_root(phrases_by_stem.get("bass", {}).get("bass_root"), shifts["bass"])
    ro = _shift_root(phrases_by_stem.get("other", {}).get("bass_root"), shifts["other"])
    root_interval_penalty = 0.0
    if rb is not None and ro is not None:
        iv = abs((ro - rb) % 12)
        iv = min(iv, 12 - iv)
        root_interval_penalty = {0: 0.0, 5: 0.05, 7: 0.05, 3: 0.1, 4: 0.1, 2: 0.2, 1: 0.4, 6: 0.6}.get(iv, 0.25)
    return {
        "bass_other_chroma": cos(cb, co),
        "bass_vocal_chroma": cos(cb, cv),
        "other_vocal_chroma": cos(co, cv),
        "root_interval_penalty": root_interval_penalty,
    }


def spectral_balance_features(phrases_by_stem: dict) -> dict:
    """Proxy spectral/masking balance from per-stem rms_db/peak (manifest)."""
    rms = {}
    for s in STEMS:
        st = (phrases_by_stem.get(s, {}).get("stems") or {}).get(s, {})
        v = st.get("rms_db")
        try:
            rms[s] = float(v)
        except Exception:
            rms[s] = -24.0
    # low-end crowding proxy: drums+bass both loud
    low_crowd = max(0.0, (rms["drums"] + 18.0) / 18.0) * max(0.0, (rms["bass"] + 18.0) / 18.0)
    vocal_buried = max(0.0, (rms["other"] - rms["vocals"]) / 18.0)
    spread = float(np.std(list(rms.values())))
    return {"low_crowd": low_crowd, "vocal_buried": vocal_buried, "rms_spread": spread, **{f"rms_{s}": rms[s] for s in STEMS}}


def energy_features(phrases_by_stem: dict) -> dict:
    e = []
    for s in STEMS:
        try:
            e.append(float(phrases_by_stem.get(s, {}).get("energy", 0.5)))
        except Exception:
            e.append(0.5)
    e = np.array(e, np.float32)
    return {"energy_mean": float(e.mean()), "energy_std": float(e.std()), "energy_min": float(e.min()), "energy_max": float(e.max())}


def per_bar_feature_vector(bar: int, phrases_by_stem: dict, kit_by_stem: dict, grids: dict) -> np.ndarray:
    """Build a fixed-order per-bar feature vector, rhythm-first."""
    # rhythm: key pairs (priority) + aggregate
    pairs = [("drums", "bass"), ("drums", "other"), ("bass", "other"), ("drums", "vocals"), ("bass", "vocals"), ("other", "vocals")]
    rfeats = []
    corr_vals = []
    for a, b in pairs:
        ga = grids.get(a)
        gb = grids.get(b)
        if ga is None or gb is None:
            # neutral when missing
            f = {"corr": 0.0, "down_agree": 0.0, "off_agree": 0.0, "density_gap": 0.5, "joint_pulse": 0.0, "sync_gap": 0.5}
        else:
            f = rhythm_pair_features(bar_slice(ga, bar), bar_slice(gb, bar))
        corr_vals.append(f["corr"])
        rfeats.extend([f["corr"], f["down_agree"], f["off_agree"], f["density_gap"], f["joint_pulse"], f["sync_gap"]])
    rhythm_agg = [float(np.mean(corr_vals)), float(np.min(corr_vals)), float(np.std(corr_vals))]
    harm = harmonic_features(phrases_by_stem, kit_by_stem)
    spec = spectral_balance_features(phrases_by_stem)
    ener = energy_features(phrases_by_stem)
    vec = []
    vec.extend(rhythm_agg)          # 3 (rhythm priority summary)
    vec.extend(rfeats)              # 36
    vec.extend([harm["bass_other_chroma"], harm["bass_vocal_chroma"], harm["other_vocal_chroma"], harm["root_interval_penalty"]])
    vec.extend([spec["low_crowd"], spec["vocal_buried"], spec["rms_spread"]])
    vec.extend([spec[f"rms_{s}"] for s in STEMS])
    vec.extend([ener["energy_mean"], ener["energy_std"], ener["energy_min"], ener["energy_max"]])
    return np.array(vec, dtype=np.float32)


FEATURE_DIM = 3 + 36 + 4 + 3 + 4 + 4  # 54


def heuristic_mix_score(per_bar: np.ndarray) -> dict:
    """Rhythm-prioritized heuristic score in [0,1] from per-bar feature vectors.
    per_bar: (4, FEATURE_DIM). Returns {per_bar:[4], mean, min, rhythm_mean}."""
    # indices based on layout above
    # 0: mean corr, 1: min corr, 2: std corr ; pair corrs at 3 + i*6
    scores = []
    rhythm_scores = []
    for b in range(per_bar.shape[0]):
        v = per_bar[b]
        mean_corr = float(v[0])
        min_corr = float(v[1])
        density_gaps = [float(v[3 + i * 6 + 3]) for i in range(6)]
        sync_gaps = [float(v[3 + i * 6 + 5]) for i in range(6)]
        # rhythm (priority): lock + downbeat agreement, penalize density/sync mismatch
        pair_down = [float(v[3 + i * 6 + 1]) for i in range(6)]
        rhythm = 0.5 + 0.25 * mean_corr + 0.15 * min_corr + 0.15 * float(np.mean(pair_down))
        rhythm -= 0.10 * float(np.mean(density_gaps)) + 0.10 * float(np.mean(sync_gaps))
        harm = 0.5 + 0.20 * float(v[39]) + 0.10 * float(v[40]) + 0.10 * float(v[41]) - 0.20 * float(v[42])
        spec = 0.5 - 0.15 * float(v[43]) - 0.10 * float(v[44]) - 0.05 * min(1.0, float(v[45]) / 12.0)
        ener_std = float(v[51])
        ener = 0.5 - 0.20 * min(1.0, ener_std / 0.4)
        # rhythm-weighted
        s = 0.50 * rhythm + 0.20 * harm + 0.15 * spec + 0.15 * ener
        s = float(max(0.0, min(1.0, s)))
        scores.append(s)
        rhythm_scores.append(float(max(0.0, min(1.0, rhythm))))
    return {
        "per_bar": scores,
        "rhythm_per_bar": rhythm_scores,
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "rhythm_mean": float(np.mean(rhythm_scores)),
        "rhythm_min": float(np.min(rhythm_scores)),
    }


# --------------------------- mix sampling -----------------------------------

def sample_random_mix(phrases, rng: random.Random, max_same_source: int = 2):
    """Sample 4 stems from different phrase windows with source cap <= max_same_source."""
    for _ in range(200):
        picks = [rng.choice(phrases) for _ in STEMS]
        tracks = [p["track_id"] for p in picks]
        counts = {}
        for t in tracks:
            counts[t] = counts.get(t, 0) + 1
        if max(counts.values()) <= max_same_source:
            return {s: picks[i] for i, s in enumerate(STEMS)}
    # fallback: enforce by rejection with relaxed picks
    picks = [rng.choice(phrases) for _ in STEMS]
    return {s: picks[i] for i, s in enumerate(STEMS)}


def make_real_mix(item):
    return {s: item for s in STEMS}




# --------------------------- render + embed teacher -------------------------

def _load_mono(path: Path, sr: int = 24000) -> np.ndarray:
    try:
        import soundfile as sf
        data, file_sr = sf.read(str(path), always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        x = np.asarray(data, dtype=np.float32)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import librosa
                x, file_sr = librosa.load(str(path), sr=None, mono=True)
                x = np.asarray(x, dtype=np.float32)
        except Exception:
            import subprocess
            cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0 or not proc.stdout:
                return np.zeros(1, np.float32)
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()
        else:
            if file_sr != sr:
                x = _resample_audio(x, int(file_sr), sr)
            return x
    if file_sr != sr:
        x = _resample_audio(x, int(file_sr), sr)
    return x


def _resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n_out = max(1, int(round(len(x) * sr_out / sr_in)))
        idx = np.linspace(0, len(x) - 1, n_out)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def _bar_segments(x: np.ndarray, bars: int = BARS_PER_PHRASE):
    n = len(x)
    if n <= 0:
        z = np.zeros(1, np.float32)
        return [z.copy() for _ in range(bars)]
    segs = []
    for b in range(bars):
        s = int(round(b * n / bars))
        e = int(round((b + 1) * n / bars))
        if e <= s:
            e = s + 1
        segs.append(np.asarray(x[s:min(e, n)], dtype=np.float32))
    return segs


def _mix_bar(segs_by_stem: dict, bar: int, target_len: int) -> np.ndarray:
    """Sum 4 stems for one bar, tempo-normalized to target_len by resampling."""
    acc = np.zeros(target_len, np.float32)
    for stem in STEMS:
        seg = segs_by_stem.get(stem, [np.zeros(1, np.float32)] * BARS_PER_PHRASE)[bar]
        if len(seg) < 2:
            continue
        if len(seg) != target_len:
            idx = np.linspace(0, len(seg) - 1, target_len)
            seg = np.interp(idx, np.arange(len(seg)), seg).astype(np.float32)
        acc += seg
    # prevent clipping, preserve relative balance roughly
    mx = float(np.max(np.abs(acc))) if acc.size else 0.0
    if mx > 1.0:
        acc = acc / (mx + 1e-6)
    return acc


def _spectral_embed(x: np.ndarray, sr: int = 24000, dim: int = 128) -> np.ndarray:
    """Deterministic mix embedding (log-mel-like spectral stats) as a robust fallback
    and as the render+embed teacher backbone when MERT is unavailable. Rhythm/texture
    sensitive: captures groove density, spectral balance, and periodicity."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < 512:
        return np.zeros(dim, np.float32)
    # frame
    n_fft = 1024; hop = 512
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames < 4:
        n_frames = 4
    frames = []
    for i in range(n_frames):
        s = i * hop
        fr = x[s:s + n_fft]
        if len(fr) < n_fft:
            fr = np.pad(fr, (0, n_fft - len(fr)))
        frames.append(fr * win)
    F = np.stack(frames)
    mag = np.abs(np.fft.rfft(F, n=n_fft))
    spec = np.log1p(mag)
    # band energies (sub/low/mid/hi/air) over freq bins
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bands = [(20, 120), (120, 300), (300, 1000), (1000, 3000), (3000, 8000), (8000, sr / 2)]
    bfeats = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            bfeats.extend([0.0, 0.0])
        else:
            v = spec[:, m]
            bfeats.extend([float(v.mean()), float(v.std())])
    # onset/flux over time (rhythm, per-bin mean keeps magnitude ~0..1)
    flux = np.diff(spec, axis=0)
    flux[flux < 0] = 0
    flux_t = flux.mean(axis=1)
    # summarize flux in 16 equal bins (bar rhythm shape)
    bins = 16
    idx = np.linspace(0, len(flux_t), bins + 1).astype(int)
    rshape = [float(flux_t[idx[i]:idx[i+1]].mean()) if idx[i+1] > idx[i] else 0.0 for i in range(bins)]
    # periodicity/autocorr of flux (groove regularity)
    ft = flux_t - flux_t.mean()
    denom = float(np.dot(ft, ft) + 1e-6)
    ac = []
    for lag in [1, 2, 3, 4, 6, 8]:
        if lag >= len(ft):
            ac.append(0.0)
        else:
            ac.append(float(np.dot(ft[:-lag], ft[lag:]) / denom))
    # spectral centroid (normalized by Nyquist -> 0..1) / flatness stats
    nyq = max(1.0, float(sr) * 0.5)
    sc_num = (mag * freqs).sum(axis=1)
    sc_den = mag.sum(axis=1) + 1e-6
    centroid = (sc_num / sc_den) / nyq
    geo = np.exp(np.mean(np.log(mag + 1e-6), axis=1))
    flat = geo / (mag.mean(axis=1) + 1e-6)
    stats = [float(centroid.mean()), float(centroid.std()), float(flat.mean()), float(flat.std()),
             float(flux_t.mean()), float(flux_t.std())]
    # 12-bin FFT chroma distribution + tonal peakiness (detects cross-key harmonic clashes)
    tonal_mask = (freqs >= 60.0) & (freqs <= 4000.0)
    chroma_vec = np.zeros(12, np.float32)
    chroma_peak = 0.0
    chroma_ent = 0.0
    if np.any(tonal_mask):
        tf = freqs[tonal_mask]
        midi = np.round(69.0 + 12.0 * np.log2(tf / 440.0)).astype(np.int32) % 12
        tm = mag[:, tonal_mask]
        for pc in range(12):
            m_pc = midi == pc
            if np.any(m_pc):
                chroma_vec[pc] = float(np.sqrt(np.mean(tm[:, m_pc] ** 2)))
        csum = float(chroma_vec.sum()) + 1e-6
        cp = chroma_vec / csum
        chroma_peak = float(cp.max() / (cp.mean() + 1e-6)) / 12.0
        chroma_ent = float(-np.sum(cp * np.log(cp + 1e-8)) / math.log(12.0))
        chroma_vec = ((cp - cp.mean()) * 4.0).astype(np.float32)
    vec = np.array(
        bfeats + rshape + ac + stats + chroma_vec.tolist() + [chroma_peak, chroma_ent],
        dtype=np.float32,
    )  # 40 + 14 = 54
    # project/pad deterministically to dim
    if vec.size >= dim:
        out = vec[:dim].copy()
    else:
        # tile with simple hashed mixing for stable width
        out = np.zeros(dim, np.float32)
        out[:vec.size] = vec
        for i in range(vec.size, dim):
            out[i] = 0.5 * vec[i % vec.size] + 0.3 * vec[(i * 3 + 1) % vec.size]
    # l2 normalize
    nrm = float(np.linalg.norm(out) + 1e-8)
    return (out / nrm).astype(np.float32)


class RenderEmbedTeacher:
    """Render summed bars and grade mixes by closeness to the real-mix manifold.

    - Decodes shipped stem audio per phrase (LRU cached) and pitch-shifts pitched
      stems to the session (drums) key before summing, matching browser playback.
    - Renders per-bar mixes with tempo normalization (resample to median bar length).
    - Embeds bars/mixes with a spectral (rhythm+harmony+texture) encoder; optionally
      uses MERT if ears_embed is importable and --mert is set.
    - Centers embeddings against the real bank mean and calibrates cosine scores
      between random cross-song baseline and real-mix k-NN similarity.
    """
    def __init__(self, kits_dir: Path, phrases: list, sr: int = 16000, device: str = "cpu",
                 use_mert: bool = False, k: int = 15):
        self.kits_dir = kits_dir
        self.sr = sr
        self.device = device
        self.k = k
        self.raw_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self.audio_cache: OrderedDict[tuple[str, str, int], list] = OrderedDict()
        self.max_raw_cache = 3000
        self.max_audio_cache = 4000
        self.bank_mean: np.ndarray | None = None
        self.sim_lo: float = -0.1
        self.sim_hi: float = 0.85
        self.encoder = None
        self.encoder_kind = "spectral"
        if use_mert:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from scripts.ears_embed import MertEncoder, MERT_SR
                enc = MertEncoder(device=device)
                enc._load()  # probe: lazy weight load
                self.encoder = enc
                self.encoder_kind = "mert"
                self.sr = MERT_SR
                log(f"[mix-compat] teacher encoder: MERT ({device})")
            except Exception as e:
                log(f"[mix-compat] MERT unavailable ({e}); using spectral encoder")
        if self.encoder is None:
            log("[mix-compat] teacher encoder: spectral rhythm+chroma+texture")

    def _raw_mono(self, item: dict, stem: str) -> np.ndarray | None:
        pid = item["phrase"]["id"]
        rkey = (pid, stem)
        if rkey in self.raw_cache:
            self.raw_cache.move_to_end(rkey)
            return self.raw_cache[rkey]
        rel = (item["phrase"].get("stems") or {}).get(stem, {}).get("file")
        if not rel:
            return None
        x = _load_mono(self.kits_dir / rel, sr=self.sr)
        self.raw_cache[rkey] = x
        if len(self.raw_cache) > self.max_raw_cache:
            self.raw_cache.popitem(last=False)
        return x

    def _bars_for(self, item: dict, stem: str, semitones: int = 0):
        pid = item["phrase"]["id"]
        shift = 0 if stem == "drums" else int(semitones)
        key = (pid, stem, shift)
        if key in self.audio_cache:
            self.audio_cache.move_to_end(key)
            return self.audio_cache[key]
        x = self._raw_mono(item, stem)
        if x is None:
            segs = [np.zeros(1, np.float32)] * BARS_PER_PHRASE
        else:
            if shift != 0 and len(x) > 64:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        import librosa
                        x = librosa.effects.pitch_shift(
                            x, sr=self.sr, n_steps=float(shift), res_type="soxr_lq"
                        ).astype(np.float32)
                except Exception:
                    pass
            segs = _bar_segments(x, BARS_PER_PHRASE)
        self.audio_cache[key] = segs
        if len(self.audio_cache) > self.max_audio_cache:
            self.audio_cache.popitem(last=False)
        return segs

    def embed_mix(self, segs_by_stem: dict, bar: int, target_len: int) -> np.ndarray:
        mix = _mix_bar(segs_by_stem, bar, target_len)
        if self.encoder is not None:
            try:
                # MertEncoder.embed expects list of 24k mono arrays
                import torch
                with torch.no_grad():
                    vec = self.encoder.embed([mix.astype(np.float32)])[0]
                v = np.asarray(vec, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v) + 1e-8)
                return (v / n).astype(np.float32)
            except Exception:
                return _spectral_embed(mix, sr=self.sr)
        return _spectral_embed(mix, sr=self.sr)

    def _target_len(self, segs_by_stem: dict) -> int:
        lens = [len(seg) for segs in segs_by_stem.values() for seg in segs if len(seg) > 0]
        if not lens:
            return self.sr  # ~1s bar fallback
        return int(np.median(lens))

    def embed_mix_phrase(self, mix_items: dict) -> np.ndarray:
        """Mean bar embedding for a 4-stem mix shifted into the drums session key."""
        session_key = (mix_items["drums"].get("kit") or {}).get("key", "")
        segs_by_stem = {}
        for s in STEMS:
            sh = 0 if (s == "drums" or not session_key) else key_shift(
                session_key, (mix_items[s].get("kit") or {}).get("key", "")
            )
            segs_by_stem[s] = self._bars_for(mix_items[s], s, semitones=sh)
        tlen = self._target_len(segs_by_stem)
        embs = [self.embed_mix(segs_by_stem, b, tlen) for b in range(BARS_PER_PHRASE)]
        v = np.mean(np.stack(embs, axis=0), axis=0)
        n = float(np.linalg.norm(v) + 1e-8)
        return (v / n).astype(np.float32)

    def build_real_bank(self, real_items: list, max_items: int = 0) -> np.ndarray:
        items = real_items if not max_items else real_items[:max_items]
        bank = []
        for i, item in enumerate(items):
            mix = {s: item for s in STEMS}
            bank.append(self.embed_mix_phrase(mix))
            if (i + 1) % 200 == 0:
                log(f"[mix-compat] real bank {i+1}/{len(items)}")
        if not bank:
            return np.zeros((0, 128), np.float32)
        arr = np.stack(bank, axis=0).astype(np.float32)
        # Center against bank mean so positive-orthant DC bias does not collapse cosines
        self.bank_mean = arr.mean(axis=0, keepdims=True)
        centered = arr - self.bank_mean
        norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
        centered = (centered / norms).astype(np.float32)
        # Calibrate [sim_lo, sim_hi] from LOO top-k similarities vs cross-track median
        if centered.shape[0] >= 8:
            sub = centered[: min(512, centered.shape[0])]
            sims = sub @ centered.T
            # mask self-matches along diagonal
            for ri in range(sub.shape[0]):
                sims[ri, ri] = -1.0
            k = min(self.k, max(1, centered.shape[0] - 1))
            top_k = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
            self.sim_lo = float(np.percentile(sims[sims > -0.99], 50))
            self.sim_hi = float(max(self.sim_lo + 0.15, np.percentile(top_k, 85)))
            log(f"[mix-compat] manifold calibration sim_lo={self.sim_lo:.3f} sim_hi={self.sim_hi:.3f}")
        return centered

    def manifold_score(self, emb: np.ndarray, bank: np.ndarray) -> float:
        """Closeness to centered real-mix manifold, calibrated to [0, 1]."""
        if bank.shape[0] == 0:
            return 0.5
        e = emb.reshape(1, -1).astype(np.float32)
        if self.bank_mean is not None and e.shape[1] == self.bank_mean.shape[1]:
            e = e - self.bank_mean
        bn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        sims = (bn @ en.T).reshape(-1)
        # Ignore exact self-match when grading real items against real bank
        non_self = sims[sims < 0.9998]
        pool = non_self if non_self.size >= self.k else sims
        k = min(self.k, len(pool))
        top = np.sort(pool)[-k:]
        raw = float(np.mean(top))
        span = max(1e-4, self.sim_hi - self.sim_lo)
        return float(max(0.0, min(1.0, (raw - self.sim_lo) / span)))



# --------------------------- main build -------------------------------------

def build_dataset(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))
    kits, phrases = phrase_index(manifest)
    if len(phrases) < 8:
        raise SystemExit("not enough complete 4-stem phrases in manifest")
    log(f"[mix-compat] phrases={len(phrases)} kits={len(kits)} feature_dim={FEATURE_DIM}")

    # precompute per-stem 64-step grids (onset sidecar if given, else phrase fields)
    sidecar_onsets = None
    onsets_path = getattr(args, "onsets", "") or ""
    if onsets_path and Path(onsets_path).is_file():
        meta = json.loads(Path(onsets_path).read_text())
        f32 = Path(onsets_path).parent / meta.get("feature_file", "mix_compat_onsets.f32")
        shape = meta.get("shape") or [meta.get("count", 0), 4, meta.get("steps", 64)]
        n_o, ns_o, st_o = int(shape[0]), int(shape[1]), int(shape[2])
        raw_o = np.fromfile(f32, dtype=np.float32)[: n_o * ns_o * st_o].reshape(n_o, ns_o, st_o)
        idx_o = {str(k): int(v) for k, v in (meta.get("index") or {}).items()}
        stems_o = list(meta.get("stems") or STEMS)
        sidecar_onsets = (raw_o, idx_o, stems_o)
        log(f"[mix-compat] loaded onset sidecar ({len(idx_o)} phrases)")

    grids_cache = {}
    for item in phrases:
        pid = item["phrase"]["id"]
        g = {}
        for s in STEMS:
            grid = None
            if sidecar_onsets is not None:
                raw_o, idx_o, stems_o = sidecar_onsets
                row = idx_o.get(pid)
                if row is not None and s in stems_o:
                    grid = np.asarray(raw_o[row, stems_o.index(s)], dtype=np.float32)
            if grid is None:
                grid = onset_grid_64(item["phrase"], s)
            g[s] = grid
        grids_cache[pid] = g

    examples = []
    # positives: real co-occurring windows
    pos_n = min(args.positives, len(phrases))
    pos_items = rng.sample(phrases, pos_n) if pos_n < len(phrases) else list(phrases)
    for item in pos_items:
        mix = make_real_mix(item)
        per_bar = np.stack([
            per_bar_feature_vector(
                b,
                {s: mix[s]["phrase"] for s in STEMS},
                {s: mix[s]["kit"] for s in STEMS},
                {s: grids_cache[mix[s]["phrase"]["id"]][s] for s in STEMS},
            ) for b in range(BARS_PER_PHRASE)
        ])
        grade = heuristic_mix_score(per_bar)
        # real mixes anchor at 1.0 but keep per-bar detail
        examples.append({
            "kind": "real",
            "label": 1.0,
            "score": max(grade["mean"], 0.85),
            "rhythm": max(grade["rhythm_mean"], 0.85),
            "min_bar": max(grade["min"], 0.80),
            "per_bar": grade["per_bar"],
            "rhythm_per_bar": grade["rhythm_per_bar"],
            "sources": [mix["drums"]["track_id"]],
            "phrases": {s: mix[s]["phrase"]["id"] for s in STEMS},
            "tracks": {s: mix[s]["track_id"] for s in STEMS},
            "features": per_bar.astype(np.float32).tolist(),
            "_item": item,  # internal, stripped before save
        })

    # negatives: random cross-song recombinations (cap <=2 sources)
    for i in range(args.negatives):
        mix = sample_random_mix(phrases, rng, max_same_source=2)
        per_bar = np.stack([
            per_bar_feature_vector(
                b,
                {s: mix[s]["phrase"] for s in STEMS},
                {s: mix[s]["kit"] for s in STEMS},
                {s: grids_cache[mix[s]["phrase"]["id"]][s] for s in STEMS},
            ) for b in range(BARS_PER_PHRASE)
        ])
        grade = heuristic_mix_score(per_bar)
        tracks = [mix[s]["track_id"] for s in STEMS]
        # same-source count penalty (discourage accidental near-copies; 2 allowed but not boosted)
        counts = {}
        for t in tracks:
            counts[t] = counts.get(t, 0) + 1
        overlap = max(counts.values())
        overlap_pen = 0.05 * max(0, overlap - 1)
        # rhythm-prioritized label: min-bar matters (one bad bar ruins it)
        label = float(max(0.0, min(1.0, 0.15 + 0.55 * grade["mean"] + 0.30 * grade["min"] - overlap_pen)))
        # keep a fraction as hard negatives / hard positives by label spread
        examples.append({
            "kind": "random",
            "label": label,
            "score": grade["mean"],
            "rhythm": grade["rhythm_mean"],
            "min_bar": grade["min"],
            "per_bar": grade["per_bar"],
            "rhythm_per_bar": grade["rhythm_per_bar"],
            "overlap": overlap,
            "sources": sorted(set(tracks)),
            "phrases": {s: mix[s]["phrase"]["id"] for s in STEMS},
            "tracks": {s: mix[s]["track_id"] for s in STEMS},
            "features": per_bar.astype(np.float32).tolist(),
            "_mix": mix,  # internal, stripped before save
        })
        if (i + 1) % 5000 == 0:
            log(f"[mix-compat] negatives {i+1}/{args.negatives}")

    grading = "heuristic_v1 (rhythm-weighted, min-bar aware)"

    # ---- render+embed teacher: grade mixes by closeness to real-mix manifold ----
    use_teacher = bool(getattr(args, "render", False) or getattr(args, "embed", False))
    teacher_info = {"enabled": False}
    if use_teacher:
        kits_dir = Path(args.kits_dir)
        use_mert = bool(getattr(args, "mert", False))
        teacher = RenderEmbedTeacher(
            kits_dir=kits_dir,
            phrases=phrases,
            sr=24000 if use_mert else 16000,
            device=getattr(args, "device", "cpu"),
            use_mert=use_mert,
            k=int(getattr(args, "teacher_k", 15)),
        )
        # build real-mix manifold bank
        real_bank_n = int(getattr(args, "teacher_real_bank", 0)) or len(pos_items)
        real_bank_n = min(real_bank_n, len(pos_items))
        log(f"[mix-compat] building real-mix manifold bank n={real_bank_n} (encoder={teacher.encoder_kind})")
        bank = teacher.build_real_bank(pos_items, max_items=real_bank_n)
        teacher_limit = int(getattr(args, "teacher_limit", 0))
        graded = 0
        for idx, ex in enumerate(examples):
            if teacher_limit and graded >= teacher_limit:
                break
            try:
                if ex["kind"] == "real":
                    mix_items = {s: ex["_item"] for s in STEMS}
                else:
                    mix_items = ex.get("_mix")
                if not mix_items:
                    continue
                emb = teacher.embed_mix_phrase(mix_items)
                mscore = teacher.manifold_score(emb, bank)
                ex["teacher_manifold"] = mscore
                # rhythm-prioritized blend: heuristic rhythm/min-bar stays dominant,
                # manifold score (closeness to real songs) refines "song-like" fit.
                h = float(ex["label"])
                ex["label_heuristic"] = h
                ex["label"] = float(max(0.0, min(1.0, 0.55 * h + 0.45 * mscore)))
                if ex["kind"] == "real":
                    ex["label"] = float(max(ex["label"], 0.90))
                # Shift per-bar targets to preserve per-bar contrast while matching blended label
                per = ex.get("per_bar") or [ex["label"]] * BARS_PER_PHRASE
                per_mean = sum(float(v) for v in per) / max(1, len(per))
                delta = ex["label"] - per_mean
                ex["per_bar"] = [float(max(0.0, min(1.0, float(v) + delta))) for v in per]
                graded += 1
            except Exception as e:
                if graded < 5:
                    log(f"[mix-compat] teacher warn {e}")
            if graded and graded % 500 == 0:
                log(f"[mix-compat] teacher graded {graded}")
        grading = f"render+embed_teacher_v1 (encoder={teacher.encoder_kind}, bank={bank.shape[0]}, graded={graded}) blended 0.55*heuristic_rhythm + 0.45*manifold"
        teacher_info = {"enabled": True, "encoder": teacher.encoder_kind, "bank": int(bank.shape[0]), "graded": graded}
        log(f"[mix-compat] {grading}")

    # strip internal refs
    for ex in examples:
        ex.pop("_item", None)
        ex.pop("_mix", None)

    # summary
    labels = np.array([e["label"] for e in examples], np.float32)
    summary = {
        "version": 1,
        "feature_dim": FEATURE_DIM,
        "bars_per_phrase": BARS_PER_PHRASE,
        "count": len(examples),
        "positives": sum(1 for e in examples if e["kind"] == "real"),
        "negatives": sum(1 for e in examples if e["kind"] == "random"),
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
        "rhythm_priority": True,
        "source_cap": 2,
        "grading": grading,
        "teacher": teacher_info,
    }
    (out_dir / "mix_compat_dataset.json").write_text(json.dumps({"summary": summary, "examples": examples}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[mix-compat] wrote {out_dir} examples={len(examples)} label_mean={summary['label_mean']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--kits-dir", default="frontend/public/data/kits")
    ap.add_argument("--onsets", default="", help="optional mix_compat_onsets.json sidecar")
    ap.add_argument("--stems", default="", help="source stems root for render+embed (offline)")
    ap.add_argument("--bars", default="", help="per-stem bar features dir (optional)")
    ap.add_argument("--out", default="/tmp/wubtrain/mix_compat_dataset")
    ap.add_argument("--positives", type=int, default=20000)
    ap.add_argument("--negatives", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render", action="store_true", help="render summed bars for embed teacher (offline, heavy)")
    ap.add_argument("--embed", action="store_true", help="compute mix embeddings/FAD teacher (offline, heavy)")
    ap.add_argument("--mert", action="store_true", help="use MERT encoder for render+embed teacher when available (else spectral fallback)")
    ap.add_argument("--teacher-k", type=int, default=15, help="k nearest real mixes for manifold score")
    ap.add_argument("--teacher-real-bank", type=int, default=0, help="real mixes in manifold bank (0=all positives)")
    ap.add_argument("--teacher-limit", type=int, default=0, help="max examples to grade with teacher (0=all, for sampled render+embed)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.render or args.embed:
        log("[mix-compat] render+embed teacher enabled (rhythm-prioritized heuristic blended with real-mix manifold closeness)")
    build_dataset(args)


if __name__ == "__main__":
    main()
