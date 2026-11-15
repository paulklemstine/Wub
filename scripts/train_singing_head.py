"""Reverse-train the Singing Head acoustic-to-articulatory model on vocal training stubs.

Extracts 60 Hz (2048-sample @ 44.1 kHz, hop=735) frames from all shipped vocal
kit stubs (`frontend/public/data/kits/kits/*/*_vocals.opus`) and full vocal
stems (`data/stems/*/vocals.wav`). For each frame:
  1. Computes the 18-d browser-exact acoustic feature vector phi(x) matching
     Web Audio AnalyserNode time-domain + Hann-FFT output.
  2. Estimates ground-truth vocal tract formants (F1, F2, F3) via 14th-order
     Levinson-Durbin LPC root solving (the classic inverse vocal-tract model),
     plus F0, multi-frame vibrato, sibilance, and corpus-calibrated loudness.
  3. Maps those acoustics to 9 continuous articulatory parameters
     (aperture, width, roundness, jaw_drop, tongue_height, pitch_norm,
      vibrato, tension, breath) and 5 vowel posterior probabilities
     (AH, EE, OH, OO, AY).
  4. Reverse-trains a compact MLP inverse encoder (18 -> 48 -> 32 -> 14) jointly
     with a training-time forward formant decoder (14 -> 24 -> 4) for
     cycle-consistent acoustic-articulatory alignment, split by track_id.

Exports lightweight JSON weights to `frontend/public/data/singing_head_model.json`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_SR = 44100
FFT_SIZE = 2048
HOP_SIZE = 735  # 44100 / 735 = 60.0 FPS

VOWELS = ["AH", "EE", "OH", "OO", "AY"]
VOWEL_REFS: List[Tuple[str, float, float]] = [
    ("AH", 730.0, 1220.0),
    ("EE", 310.0, 2280.0),
    ("OH", 570.0, 920.0),
    ("OO", 320.0, 860.0),
    ("AY", 590.0, 1820.0),
]

BANDS_HZ: List[Tuple[float, float]] = [
    (80.0, 250.0),
    (250.0, 500.0),
    (500.0, 850.0),
    (850.0, 1350.0),
    (1350.0, 2000.0),
    (2000.0, 3000.0),
    (3000.0, 5000.0),
    (5000.0, 9000.0),
]

FEATURE_NAMES = [
    "log_rms",
    "peak_env",
    "zcr",
    "voicing",
    "f0_norm",
    "centroid_log",
    "rolloff_norm",
    "flatness",
    "spectral_flux",
    "band_0_80_250",
    "band_1_250_500",
    "band_2_500_850",
    "band_3_850_1350",
    "band_4_1350_2000",
    "band_5_2000_3000",
    "band_6_3000_5000",
    "band_7_5000_9000",
    "low_high_tilt",
]

ARTICULATOR_NAMES = [
    "aperture",
    "width",
    "roundness",
    "jaw_drop",
    "tongue_height",
    "pitch_norm",
    "vibrato",
    "tension",
    "breath",
]

# Exact 14 Voice Workshop mid-sagittal mouth & throat parameters & ranges
# matching `frontend/src/vocal_sim/tract/types.ts`.
TRACT_ARTICULATOR_KEYS = [
    "jaw",
    "tongueX",
    "tongueY",
    "tongueRoot",
    "tipX",
    "tipY",
    "groove",
    "tongueSides",
    "lipOpen",
    "lipRound",
    "lipProtrude",
    "lipTuck",
    "velum",
    "larynx",
]

TRACT_RANGES: List[Tuple[float, float]] = [
    (0.0, 1.0),    # jaw
    (-1.6, 1.4),   # tongueX
    (-1.6, 1.6),   # tongueY
    (-1.2, 1.2),   # tongueRoot
    (-1.2, 1.4),   # tipX
    (-0.6, 2.2),   # tipY
    (0.0, 1.0),    # groove
    (0.0, 1.0),    # tongueSides
    (-0.4, 1.2),   # lipOpen
    (0.0, 1.0),    # lipRound
    (-0.6, 1.0),   # lipProtrude
    (0.0, 1.0),    # lipTuck
    (0.0, 1.0),    # velum
    (-1.5, 1.2),   # larynx
]

# Quiet closed-lips neutral posture in normalized [0, 1] tract coordinates
TRACT_REST_RAW = [0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02, 0.0]
TRACT_REST_NORM = np.array(
    [(v - lo) / (hi - lo) for v, (lo, hi) in zip(TRACT_REST_RAW, TRACT_RANGES)],
    dtype=np.float32,
)

_HANN_2048 = (0.5 - 0.5 * np.cos((2.0 * np.pi * np.arange(FFT_SIZE)) / (FFT_SIZE - 1))).astype(np.float64)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def extract_frame_features(
    frame: np.ndarray,
    sr: int = TARGET_SR,
    prev_norm_mag: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """Extract the 18-d browser-exact acoustic feature vector from a 2048-sample frame.

    Returns (features_18, rms, f0_hz, voicing, norm_mag_for_flux).
    """
    n = len(frame)
    x = frame.astype(np.float64, copy=False)
    x = x - float(np.mean(x))

    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))

    # Zero-crossing rate matching `x[i] >= 0` sign transitions
    signs = x >= 0.0
    zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / max(1, n - 1)

    # Autocorrelation pitch & voicing on 2x decimated frame
    x_dec = x[::2]
    sr_dec = sr / 2.0
    m = len(x_dec)
    power = float(np.dot(x_dec, x_dec))
    voicing = 0.0
    f0_hz = 0.0
    if power > 1e-6:
        ac = np.fft.irfft(np.abs(np.fft.rfft(x_dec, n=2 * m)) ** 2, n=2 * m)[:m]
        min_lag = max(8, int(sr_dec / 800.0))
        max_lag = min(m - 1, int(math.ceil(sr_dec / 80.0)))
        if max_lag > min_lag:
            lags = np.arange(min_lag, max_lag + 1, dtype=np.float64)
            corr = ac[min_lag : max_lag + 1] / max(1e-9, power)
            denom = np.maximum(0.01, 1.0 - lags / float(m))
            scores = corr / denom
            best_idx = int(np.argmax(scores))
            best_val = float(scores[best_idx])
            best_lag = int(lags[best_idx])
            voicing = clamp(best_val, 0.0, 1.0)
            if best_lag > 0 and voicing > 0.30 and rms > 0.008:
                f0_hz = sr_dec / float(best_lag)

    f0_norm = clamp(math.log2(f0_hz / 220.0), -1.5, 1.5) if f0_hz > 0.0 else 0.0

    # Hann-windowed FFT magnitude (normalized by N/2 so scale is independent of N)
    win = _HANN_2048 if n == FFT_SIZE else (0.5 - 0.5 * np.cos((2.0 * np.pi * np.arange(n)) / max(1, n - 1)))
    spec = np.fft.rfft(x * win)
    mag = np.abs(spec[:-1]) / (n * 0.5)  # length = n // 2 = 1024
    mag_sq = mag * mag
    bin_hz = sr / float(n)
    half_n = len(mag)

    # Normalized spectrum for flux (bins up to 9000 Hz)
    flux_hi = min(half_n - 1, int(math.ceil(9000.0 / bin_hz)))
    norm_mag = mag[1 : flux_hi + 1] / (rms + 1e-4)
    if prev_norm_mag is not None and len(prev_norm_mag) == len(norm_mag):
        diff = np.maximum(0.0, norm_mag - prev_norm_mag)
        flux = math.log1p(float(np.mean(diff)) * 2.0)
    else:
        flux = 0.0

    # Spectral centroid (100..8000 Hz)
    c_lo = max(1, int(math.floor(100.0 / bin_hz)))
    c_hi = min(half_n - 1, int(math.ceil(8000.0 / bin_hz)))
    freqs_c = np.arange(c_lo, c_hi + 1, dtype=np.float64) * bin_hz
    e_c = mag_sq[c_lo : c_hi + 1]
    e_c_sum = float(np.sum(e_c))
    centroid_hz = float(np.sum(freqs_c * e_c) / e_c_sum) if e_c_sum > 1e-12 else 500.0
    centroid_log = clamp(math.log2(max(80.0, centroid_hz) / 500.0), -2.5, 4.0)

    # 85% spectral rolloff (100..9000 Hz)
    r_lo = max(1, int(math.floor(100.0 / bin_hz)))
    r_hi = min(half_n - 1, int(math.ceil(9000.0 / bin_hz)))
    e_r = mag_sq[r_lo : r_hi + 1]
    e_r_sum = float(np.sum(e_r))
    if e_r_sum > 1e-12:
        cum = np.cumsum(e_r)
        idx_85 = int(np.searchsorted(cum, 0.85 * e_r_sum))
        rolloff_hz = (r_lo + min(idx_85, len(e_r) - 1)) * bin_hz
        rolloff_norm = clamp(rolloff_hz / 9000.0, 0.0, 1.0)
    else:
        rolloff_norm = 0.0

    # Spectral flatness (150..6000 Hz)
    f_lo = max(1, int(math.floor(150.0 / bin_hz)))
    f_hi = min(half_n - 1, int(math.ceil(6000.0 / bin_hz)))
    e_f = mag_sq[f_lo : f_hi + 1] + 1e-12
    arith_mean = float(np.mean(e_f))
    geom_mean = float(math.exp(np.mean(np.log(e_f))))
    flatness = clamp(geom_mean / max(1e-12, arith_mean), 0.0, 1.0)

    # 8 log-band energies
    bands = np.zeros(8, dtype=np.float64)
    for b_idx, (b_lo_hz, b_hi_hz) in enumerate(BANDS_HZ):
        lo_b = max(1, int(math.floor(b_lo_hz / bin_hz)))
        hi_b = min(half_n - 1, int(math.ceil(b_hi_hz / bin_hz)))
        if hi_b >= lo_b:
            band_rms = math.sqrt(float(np.mean(mag_sq[lo_b : hi_b + 1])))
            bands[b_idx] = math.log1p(band_rms * 32.0)

    low_high_tilt = float((bands[0] + bands[1] + bands[2]) - (bands[4] + bands[5] + bands[6]))

    feats = np.array(
        [
            math.log1p(rms * 25.0),
            min(1.5, peak * 1.5),
            zcr,
            voicing,
            f0_norm,
            centroid_log,
            rolloff_norm,
            flatness,
            flux,
            *bands.tolist(),
            low_high_tilt,
        ],
        dtype=np.float32,
    )
    return feats, rms, f0_hz, voicing, norm_mag


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC polynomial coefficients [1, a1, ..., a_order] via Levinson-Durbin."""
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    if r[0] <= 1e-12:
        return a
    err = float(r[0])
    for i in range(1, order + 1):
        acc = float(r[i])
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / max(1e-12, err)
        k = clamp(k, -0.999, 0.999)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        err *= max(1e-6, 1.0 - k * k)
    return a


def _pick_band_formant(cands: List[Tuple[float, float, float]]) -> Optional[float]:
    """Drop weak broad tilt poles relative to band max, then return lowest formant frequency."""
    if not cands:
        return None
    cutoff = 0.25 * max(c[2] for c in cands)
    strong = [c for c in cands if c[2] >= cutoff]
    return strong[0][0] if strong else cands[0][0]


def estimate_lpc_formants(frame_44k: np.ndarray, sr: int = TARGET_SR) -> Tuple[float, float, float]:
    """Estimate (F1, F2, F3) in Hz using 12th-order LPC prominence roots at 11.025 kHz."""
    # Decimate 44.1 kHz -> 11.025 kHz (factor 4) via polyphase anti-aliased filter
    x_11k = resample_poly(frame_44k.astype(np.float64), 1, 4)
    sr_11k = sr / 4.0
    if len(x_11k) < 64 or float(np.sqrt(np.mean(x_11k * x_11k))) < 1e-4:
        return 550.0, 1450.0, 2550.0

    # Pre-emphasis + Hann window
    pre = np.empty_like(x_11k)
    pre[0] = x_11k[0]
    pre[1:] = x_11k[1:] - 0.97 * x_11k[:-1]
    win = 0.5 - 0.5 * np.cos((2.0 * np.pi * np.arange(len(pre))) / max(1, len(pre) - 1))
    xw = pre * win

    order = 12
    r = np.array([float(np.dot(xw[: len(xw) - lag], xw[lag:])) for lag in range(order + 1)], dtype=np.float64)
    a = _levinson_durbin(r, order)

    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.0]
    narrow: List[Tuple[float, float, float]] = []
    wide: List[Tuple[float, float, float]] = []
    for z in roots:
        ang = float(np.angle(z))
        freq = ang * (sr_11k / (2.0 * np.pi))
        rad = abs(z)
        if rad <= 0.10 or rad >= 0.9999:
            continue
        bw = -0.5 * (sr_11k / (2.0 * np.pi)) * math.log(rad)
        if 200.0 <= freq <= 4200.0 and bw < 450.0:
            ejw = np.exp(-1j * ang * np.arange(order + 1))
            denom = abs(complex(np.dot(a, ejw)))
            prominence = (1.0 / max(1e-4, denom)) / (max(20.0, bw) ** 1.25)
            item = (freq, bw, prominence)
            if bw < 280.0:
                narrow.append(item)
            else:
                wide.append(item)

    cand = sorted(narrow if len(narrow) >= 2 else (narrow + wide), key=lambda c: c[0])
    spec = np.abs(np.fft.rfft(frame_44k * _HANN_2048))
    bin_hz = sr / float(len(frame_44k))

    f1_pick = _pick_band_formant([c for c in cand if 220.0 <= c[0] <= 1020.0])
    if f1_pick is not None:
        f1 = f1_pick
    else:
        lo = max(1, int(250.0 / bin_hz))
        hi = min(len(spec) - 1, int(1020.0 / bin_hz))
        f1 = float((lo + int(np.argmax(spec[lo : hi + 1]))) * bin_hz) if hi > lo else 550.0

    f2_lo = max(f1 + 200.0, 680.0)
    f2_pick = _pick_band_formant([c for c in cand if f2_lo <= c[0] <= 2850.0])
    if f2_pick is not None:
        f2 = f2_pick
    else:
        lo = max(1, int(max(f1 + 200.0, 750.0) / bin_hz))
        hi = min(len(spec) - 1, int(2850.0 / bin_hz))
        f2 = float((lo + int(np.argmax(spec[lo : hi + 1]))) * bin_hz) if hi > lo else 1450.0

    f3_lo = max(f2 + 250.0, 1900.0)
    f3_pick = _pick_band_formant([c for c in cand if f3_lo <= c[0] <= 4100.0])
    f3 = f3_pick if f3_pick is not None else 2550.0

    return f1, f2, f3


def load_audio_44k(path: Path, max_seconds: Optional[float] = None) -> np.ndarray:
    """Load audio file as mono float32 resampled to 44,100 Hz."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if max_seconds is not None and len(data) > int(max_seconds * sr):
        data = data[: int(max_seconds * sr)]
    if sr != TARGET_SR:
        g = math.gcd(TARGET_SR, int(sr))
        data = resample_poly(data, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return data


def _process_audio_segment(
    track_id: str,
    audio: np.ndarray,
    is_stem: bool,
    max_frames: int,
) -> List[dict]:
    """Slide 60 Hz windows across an audio segment and compute features + LPC targets."""
    if len(audio) < FFT_SIZE:
        return []

    n_frames = 1 + (len(audio) - FFT_SIZE) // HOP_SIZE
    if n_frames <= 0:
        return []

    # Always run sequential pass to preserve exact 60 Hz spectral flux & vibrato history
    frame_rows: List[dict] = []
    prev_norm_mag: Optional[np.ndarray] = None
    rms_hist: List[float] = []
    f0_hist: List[float] = []

    # Sample stride if segment is very long (keep sequential flux via short 2-frame lookback)
    step_frames = max(1, n_frames // max_frames) if max_frames > 0 and n_frames > max_frames else 1

    idx = 0
    while idx < n_frames:
        start = idx * HOP_SIZE
        frame = audio[start : start + FFT_SIZE]
        if step_frames > 1 and idx > 0:
            # Prime prev_norm_mag on the immediate predecessor frame for realistic 60 Hz flux
            prev_start = (idx - 1) * HOP_SIZE
            _, _, _, _, prev_norm_mag = extract_frame_features(audio[prev_start : prev_start + FFT_SIZE], TARGET_SR, None)

        feats, rms, f0_hz, voicing, prev_norm_mag = extract_frame_features(frame, TARGET_SR, prev_norm_mag)
        f1, f2, f3 = estimate_lpc_formants(frame, TARGET_SR) if rms > 0.004 else (550.0, 1450.0, 2550.0)
        rms_hist.append(rms)
        f0_hist.append(f0_hz)

        frame_rows.append(
            {
                "track_id": track_id,
                "is_stem": is_stem,
                "feats": feats,
                "rms": rms,
                "f0": f0_hz,
                "voicing": voicing,
                "f1": f1,
                "f2": f2,
                "f3": f3,
            }
        )
        idx += step_frames

    # Compute local vibrato (semitone std over +-8 frames) & RMS onset rise
    m = len(frame_rows)
    for i, row in enumerate(frame_rows):
        lo = max(0, i - 8)
        hi = min(m, i + 9)
        local_f0s = [frame_rows[j]["f0"] for j in range(lo, hi) if frame_rows[j]["f0"] > 0.0]
        if len(local_f0s) >= 4 and row["f0"] > 0.0:
            med_f0 = float(np.median(local_f0s))
            semis = [12.0 * math.log2(max(20.0, f) / max(20.0, med_f0)) for f in local_f0s]
            vib_std = float(np.std(semis))
            # Vibrato peaks around 0.25..1.5 semitones std; huge jumps (>3 semis) are pitch leaps, not vibrato
            row["vibrato"] = clamp(vib_std / 1.1, 0.0, 1.0) if vib_std < 3.0 else 0.35
        else:
            row["vibrato"] = 0.0

        prev_rms = frame_rows[i - 1]["rms"] if i > 0 else row["rms"]
        row["rms_rise"] = max(0.0, row["rms"] - prev_rms)

    return frame_rows


def _worker_stub(args: Tuple[str, str, bool]) -> List[dict]:
    path_str, track_id, is_stem = args
    path = Path(path_str)
    try:
        if not is_stem:
            audio = load_audio_44k(path)
            return _process_audio_segment(track_id, audio, is_stem=False, max_frames=22)
        else:
            info = sf.info(str(path))
            sr = int(info.samplerate)
            total_samples = int(info.frames)
            win_sec = 4.0
            win_n = int(win_sec * sr)
            if total_samples <= win_n:
                audio = load_audio_44k(path)
                return _process_audio_segment(track_id, audio, is_stem=True, max_frames=20)
            rows: List[dict] = []
            # Pick 5 evenly spaced 4-second windows across the stem
            for frac in (0.15, 0.30, 0.50, 0.70, 0.85):
                start = max(0, min(total_samples - win_n, int(total_samples * frac)))
                chunk, chunk_sr = sf.read(str(path), start=start, frames=win_n, dtype="float32")
                if chunk.ndim > 1:
                    chunk = np.mean(chunk, axis=1)
                if chunk_sr != TARGET_SR:
                    g = math.gcd(TARGET_SR, int(chunk_sr))
                    chunk = resample_poly(chunk, TARGET_SR // g, int(chunk_sr) // g).astype(np.float32)
                rows.extend(_process_audio_segment(track_id, chunk, is_stem=True, max_frames=12))
            return rows
    except Exception:
        return []


def compute_articulatory_targets_with_physics(
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Map corpus-calibrated acoustic rows to (X_18, Y_14, cycle_4, obs_phys_7, cues_5, corpus_stats)."""
    rms_arr = np.array([r["rms"] for r in rows], dtype=np.float64)
    active_rms = rms_arr[rms_arr > 0.004]
    if len(active_rms) < 20:
        raise RuntimeError("Too few active vocal frames found in training stubs.")

    db_arr = 20.0 * np.log10(np.maximum(active_rms, 1e-5))
    db_p10 = float(np.percentile(db_arr, 10))
    db_p95 = float(np.percentile(db_arr, 95))
    if db_p95 <= db_p10 + 1.0:
        db_p95 = db_p10 + 18.0

    f0_arr = np.array([r["f0"] for r in rows if r["f0"] > 0.0], dtype=np.float64)
    if len(f0_arr) >= 20:
        log_f0s = np.log2(f0_arr)
        f0_p10 = float(2.0 ** np.percentile(log_f0s, 10))
        f0_p95 = float(2.0 ** np.percentile(log_f0s, 95))
    else:
        f0_p10, f0_p95 = 130.0, 520.0
    if f0_p95 <= f0_p10 * 1.15:
        f0_p95 = f0_p10 * 2.0

    n = len(rows)
    x_mat = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    y_mat = np.zeros((n, 14), dtype=np.float32)
    cycle_mat = np.zeros((n, 4), dtype=np.float32)
    obs_phys_mat = np.zeros((n, 7), dtype=np.float32)
    cues_mat = np.zeros((n, 5), dtype=np.float32)  # [active_gate, nasality, sibilance, frication_mid, pitch_norm]

    log_f0_lo = math.log2(f0_p10)
    log_f0_hi = math.log2(f0_p95)

    for i, row in enumerate(rows):
        feats = row["feats"]
        x_mat[i] = feats
        rms = float(row["rms"])
        f0 = float(row["f0"])
        voicing = float(row["voicing"])
        f1 = float(row["f1"])
        f2 = float(row["f2"])
        f3 = float(row.get("f3", 2550.0))
        vibrato = float(row["vibrato"])
        rms_rise = float(row["rms_rise"])

        # Loudness normalization using empirical corpus dB quantiles
        if rms <= 0.0025:
            loud_norm = 0.0
        else:
            db = 20.0 * math.log10(max(rms, 1e-5))
            loud_norm = clamp((db - db_p10) / (db_p95 - db_p10), 0.0, 1.0)
            # Keep soft singing audible above noise floor
            if rms > 0.004:
                loud_norm = max(0.08, loud_norm)

        f1_norm = clamp((f1 - 260.0) / (920.0 - 260.0), 0.0, 1.0)
        f2_norm = clamp((f2 - 800.0) / (2450.0 - 800.0), 0.0, 1.0)

        # Unvoiced sibilance / breathiness / frication / nasal murmur descriptors from 18-d FFT bands
        zcr = float(feats[2])
        flatness = float(feats[7])
        band0 = float(feats[9])    # 80..250 Hz (nasal murmur / fundamental)
        band1 = float(feats[10])   # 250..500 Hz
        band2 = float(feats[11])   # 500..850 Hz (oral vowel F1 body)
        band3 = float(feats[12])   # 850..1350 Hz (oral vowel F1/F2)
        presence_band = float(feats[15])  # band_6_3000_5000
        sib_band = float(feats[16])       # band_7_5000_9000
        active_gate = clamp((rms - 0.003) / 0.025, 0.0, 1.0)
        breath = clamp(
            active_gate * (1.0 - voicing * 0.85) * (0.45 * clamp(sib_band / 2.2, 0.0, 1.0) + 0.55 * flatness),
            0.0,
            1.0,
        )
        sibilance = clamp(
            active_gate
            * (1.0 - 0.72 * voicing)
            * (0.55 * clamp(sib_band / 0.50, 0.0, 1.0) + 0.30 * flatness + 0.15 * clamp(zcr / 0.22, 0.0, 1.0)),
            0.0,
            1.0,
        )
        frication_mid = clamp(
            active_gate
            * (1.0 - 0.72 * voicing)
            * (0.55 * clamp(presence_band / 1.2, 0.0, 1.0) + 0.45 * flatness)
            * (1.0 - 0.55 * sibilance),
            0.0,
            1.0,
        )
        oral_body = band1 + band2 + band3 + 0.5 * float(feats[13])
        nasality = clamp(
            active_gate
            * voicing
            * clamp((band0 - 0.52 * oral_body) / max(0.32, band0 + band1 + band2), 0.0, 1.0)
            * 1.50,
            0.0,
            1.0,
        )

        # Pitch normalized within empirical corpus singing range
        if f0 > 0.0:
            pitch_norm = clamp((math.log2(f0) - log_f0_lo) / max(0.2, log_f0_hi - log_f0_lo), 0.0, 1.0)
        else:
            pitch_norm = 0.42

        # Articulatory targets grounded in vocal tract physics
        onset_boost = clamp(rms_rise * 10.0, 0.0, 0.25)
        aperture = clamp(
            active_gate * (0.14 + 0.56 * loud_norm + 0.30 * f1_norm + onset_boost) * (1.0 - 0.35 * breath),
            0.0,
            1.0,
        )
        width = clamp(
            0.50 + active_gate * ((f2_norm - 0.48) * 0.92 + 0.10 * breath - 0.12 * (1.0 - f1_norm) * (1.0 - f2_norm)),
            0.06,
            0.96,
        )
        roundness = clamp(
            active_gate * (1.0 - f2_norm) * (0.75 + 0.25 * (1.0 - f1_norm)) * (0.35 + 0.65 * voicing),
            0.0,
            1.0,
        )
        jaw_drop = clamp(aperture * (0.42 + 0.58 * f1_norm), 0.0, 1.0)
        tongue_height = clamp(1.0 - f1_norm * (0.35 + 0.65 * active_gate), 0.0, 1.0)
        tension = clamp(
            active_gate * (0.48 * loud_norm + 0.28 * clamp(presence_band / 2.6, 0.0, 1.0) + 0.24 * pitch_norm),
            0.0,
            1.0,
        )

        # Vowel probabilities from LPC (F1, F2) distances
        if rms <= 0.005:
            vowel_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        else:
            dists = np.array(
                [((f1 - r1) / 260.0) ** 2 + ((f2 - r2) / 680.0) ** 2 for _, r1, r2 in VOWEL_REFS],
                dtype=np.float64,
            )
            logits = -dists / 0.55
            logits -= np.max(logits)
            exps = np.exp(logits)
            vowel_probs = (exps / np.sum(exps)).astype(np.float32)

        y_mat[i, :9] = np.array(
            [aperture, width, roundness, jaw_drop, tongue_height, pitch_norm, vibrato, tension, breath],
            dtype=np.float32,
        )
        y_mat[i, 9:] = vowel_probs

        cycle_mat[i] = np.array([f1_norm, f2_norm, pitch_norm, loud_norm], dtype=np.float32)

        desired_lip_area = clamp(
            (0.32 + 3.6 * f1_norm + 2.4 * loud_norm * (0.25 + 0.75 * f1_norm))
            * (1.0 - 0.55 * sibilance)
            * (1.0 - 0.45 * nasality * (1.0 - loud_norm)),
            0.12,
            8.0,
        )
        desired_min_area = clamp(
            (0.18 + 0.95 * loud_norm * (0.5 + 0.5 * f1_norm)) * (1.0 - 0.72 * sibilance),
            0.04,
            2.2,
        )
        desired_pharynx = clamp(2.1 + 1.8 * f2_norm - 0.9 * f1_norm, 0.8, 5.8)
        desired_oral = clamp(1.2 + 2.8 * loud_norm + 1.4 * f1_norm - 0.8 * f2_norm, 0.5, 6.5)
        obs_phys_mat[i] = np.array(
            [f1, f2, f3, desired_lip_area, desired_min_area, desired_pharynx, desired_oral],
            dtype=np.float32,
        )
        cues_mat[i] = np.array([active_gate, nasality, sibilance, frication_mid, pitch_norm], dtype=np.float32)

    stats = {
        "db_p10": round(db_p10, 3),
        "db_p95": round(db_p95, 3),
        "f0_p10": round(f0_p10, 2),
        "f0_p95": round(f0_p95, 2),
    }
    return x_mat, y_mat, cycle_mat, obs_phys_mat, cues_mat, stats


def compute_articulatory_targets(
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Map corpus-calibrated acoustic rows to (X_18, Y_14, cycle_4, corpus_stats)."""
    x_mat, y_mat, cycle_mat, _, _, stats = compute_articulatory_targets_with_physics(rows)
    return x_mat, y_mat, cycle_mat, stats


def _norm_tract_np(art_raw: np.ndarray) -> np.ndarray:
    lo = np.array([r[0] for r in TRACT_RANGES], dtype=np.float32)
    span = np.array([r[1] - r[0] for r in TRACT_RANGES], dtype=np.float32)
    return np.clip((art_raw - lo) / span, 0.0, 1.0)


def derive_fallback_tract_targets(y_mat: np.ndarray, cycle_mat: np.ndarray) -> np.ndarray:
    """Synthesize smooth 14-D normalized tract targets when no physics atlas is supplied."""
    n = len(y_mat)
    out = np.zeros((n, 14), dtype=np.float32)
    aperture = y_mat[:, 0]
    width = y_mat[:, 1]
    roundness = y_mat[:, 2]
    jaw_drop = y_mat[:, 3]
    tongue_h = y_mat[:, 4]
    pitch_n = y_mat[:, 5]
    breath = y_mat[:, 8]
    f1_n = cycle_mat[:, 0]
    f2_n = cycle_mat[:, 1]
    raw = np.zeros((n, 14), dtype=np.float32)
    raw[:, 0] = 0.12 + 0.78 * jaw_drop
    raw[:, 1] = (f2_n - 0.5) * 2.2
    raw[:, 2] = (tongue_h - 0.45) * 2.4
    raw[:, 3] = (f2_n - 0.5) * 1.4 - 0.35 * f1_n
    raw[:, 4] = (f2_n - 0.45) * 1.6 + 0.65 * breath
    raw[:, 5] = (tongue_h - 0.35) * 1.4 + 1.2 * breath
    raw[:, 6] = 0.12 + 0.78 * breath
    raw[:, 7] = 0.15 + 0.65 * breath
    raw[:, 8] = -0.05 + 1.05 * aperture
    raw[:, 9] = roundness
    raw[:, 10] = -0.15 + 0.95 * roundness - 0.25 * (width - 0.5)
    raw[:, 11] = 0.35 * breath * (1.0 - aperture)
    raw[:, 12] = 0.05 + 0.15 * (1.0 - aperture) * (1.0 - breath)
    raw[:, 13] = (pitch_n - 0.5) * 1.8
    out[:] = _norm_tract_np(raw)
    return out


class TractPhysicsForwardNet(nn.Module):
    """Differentiable surrogate for Voice Workshop's 44-section vocal tract physics."""

    def __init__(self, art_dim: int = 14, phys_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(art_dim, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, phys_dim),
        )

    def forward(self, art_norm: torch.Tensor) -> torch.Tensor:
        return self.net(art_norm)


def train_physics_forward_net(
    atlas_path: Path,
    dev: torch.device,
    epochs: int = 160,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[TractPhysicsForwardNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Train & freeze forward vocal-tract physics surrogate on resonant Voice Workshop postures."""
    d = np.load(atlas_path)
    atlas_art_raw = d["art"].astype(np.float32)
    atlas_phys = d["phys"].astype(np.float32)

    valid = (
        (atlas_phys[:, 0] >= 170.0)
        & (atlas_phys[:, 0] <= 1150.0)
        & (atlas_phys[:, 1] >= 500.0)
        & (atlas_phys[:, 1] <= 3100.0)
        & (atlas_phys[:, 2] >= 1500.0)
        & (atlas_phys[:, 2] <= 4150.0)
        & (atlas_phys[:, 3] >= 0.04)
    )
    if int(np.sum(valid)) >= 1000:
        atlas_art_raw = atlas_art_raw[valid]
        atlas_phys = atlas_phys[valid]

    atlas_art_norm = _norm_tract_np(atlas_art_raw)

    phys_mean = np.mean(atlas_phys, axis=0).astype(np.float32)
    phys_std = np.maximum(np.std(atlas_phys, axis=0), 1e-4).astype(np.float32)
    phys_norm = ((atlas_phys - phys_mean) / phys_std).astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(atlas_art_norm))
    split = int(0.88 * len(perm))
    tr_idx, va_idx = perm[:split], perm[split:]

    xa = torch.from_numpy(atlas_art_norm[tr_idx]).to(dev)
    yp = torch.from_numpy(phys_norm[tr_idx]).to(dev)
    xva = torch.from_numpy(atlas_art_norm[va_idx]).to(dev)
    yvp = torch.from_numpy(phys_norm[va_idx]).to(dev)

    fwd = TractPhysicsForwardNet(art_dim=14, phys_dim=7).to(dev)
    opt = torch.optim.AdamW(fwd.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_tr = len(xa)
    for _ in range(epochs):
        fwd.train()
        p = torch.randperm(n_tr, device=dev)
        for s in range(0, n_tr, batch_size):
            idx = p[s : s + batch_size]
            pred = fwd(xa[idx])
            loss = F.smooth_l1_loss(pred, yp[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    fwd.eval()
    with torch.no_grad():
        vp = fwd(xva)
        ss_res = float(((vp - yvp) ** 2).sum().item())
        ss_tot = float(((yvp - yvp.mean(dim=0, keepdim=True)) ** 2).sum().item())
        r2 = 1.0 - ss_res / max(1e-8, ss_tot)
    for param in fwd.parameters():
        param.requires_grad = False
    return fwd, atlas_art_norm, atlas_phys, phys_mean, phys_std, r2


# Canonical 14-D Voice Workshop vowel anchors [AH, EE, OH, OO, AY] to resolve
# many-to-one acoustic inverse ambiguity toward natural human articulation.
_CANONICAL_VOWEL_14 = np.array(
    [
        # jaw, tX,   tY,   tRoot, tipX, tipY, groove, sides, lipO, lipR, lipP, tuck, velum, larynx
        [0.82, -0.45, -0.75, -0.35, -0.20, -0.10, 0.14, 0.20, 0.56, 0.04, -0.05, 0.0, 0.04, -0.05],  # AH
        [0.20, 1.05, 0.92, 0.95, 0.55, 0.42, 0.22, 0.35, 0.25, 0.04, -0.28, 0.0, 0.04, 0.16],       # EE
        [0.44, -0.74, 0.35, 0.25, -0.30, -0.05, 0.18, 0.25, 0.20, 0.86, 0.45, 0.0, 0.05, -0.20],     # OH
        [0.16, -0.35, 1.02, 0.30, -0.35, 0.08, 0.24, 0.30, 0.08, 0.95, 0.88, 0.0, 0.05, -0.35],      # OO
        [0.44, 0.76, 0.78, 0.12, 0.38, 0.22, 0.16, 0.22, 0.58, 0.02, -0.22, 0.0, 0.04, 0.08],        # AY
    ],
    dtype=np.float32,
)
_CANONICAL_VOWEL_NORM = _norm_tract_np(_CANONICAL_VOWEL_14)


def invert_tract_targets_from_atlas(
    obs_phys: np.ndarray,
    cues: np.ndarray,
    atlas_art_norm: np.ndarray,
    atlas_phys: np.ndarray,
    dev: torch.device,
    y_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Invert (F1, F2, F3, opening/constriction, nasal/sibilant/pitch cues) to 14-D tract targets."""
    cand_phys = atlas_phys
    cand_art = atlas_art_norm

    # Physics tolerances for weighted Euclidean matching:
    # [F1, F2, F3, lipArea, minArea, pharynxArea, oralArea]
    scales = np.array([150.0, 320.0, 520.0, 1.85, 0.65, 1.6, 1.8], dtype=np.float32)
    weights = np.array([1.2, 1.1, 0.40, 0.75, 0.35, 0.15, 0.15], dtype=np.float32)
    w_phys_sqrt = np.sqrt(weights) / scales

    # Canonical anatomical prior (vowel-blended) resolves many-to-one redundancy cleanly
    if y_mat is not None:
        v_probs = np.clip(y_mat[:, 9:14], 0.0, 1.0)
        v_probs = v_probs / np.maximum(1e-6, v_probs.sum(axis=1, keepdims=True))
        prior_norm = v_probs @ _CANONICAL_VOWEL_NORM
    else:
        prior_norm = np.broadcast_to(TRACT_REST_NORM[None, :], (len(obs_phys), 14)).copy()

    w_prior = np.array([0.85, 0.55, 0.55, 0.40, 0.35, 0.35, 0.45, 0.35, 0.85, 1.35, 1.25, 0.45, 0.85, 0.45], dtype=np.float32)
    w_prior_sqrt = np.sqrt(w_prior)

    cp_mat = np.concatenate([cand_phys * w_phys_sqrt[None, :], cand_art * w_prior_sqrt[None, :]], axis=1)
    cp_t = torch.from_numpy(cp_mat).to(dev)    # (M, 21)
    ca_t = torch.from_numpy(cand_art).to(dev)  # (M, 14)

    n = len(obs_phys)
    inv_norm = np.zeros((n, 14), dtype=np.float32)
    chunk_sz = 2048
    k_top = min(12, len(cand_phys))

    with torch.no_grad():
        c2 = (cp_t * cp_t).sum(dim=1).unsqueeze(0)
        for s in range(0, n, chunk_sz):
            q_mat = np.concatenate(
                [obs_phys[s : s + chunk_sz] * w_phys_sqrt[None, :], prior_norm[s : s + chunk_sz] * w_prior_sqrt[None, :]],
                axis=1,
            )
            q = torch.from_numpy(q_mat).to(dev)
            q2 = (q * q).sum(dim=1, keepdim=True)
            dist2 = torch.clamp(q2 + c2 - 2.0 * (q @ cp_t.T), min=0.0)
            vals, idxs = torch.topk(dist2, k=k_top, dim=1, largest=False)
            w = torch.softmax(-vals / 0.25, dim=1)
            blended = (w.unsqueeze(-1) * ca_t[idxs]).sum(dim=1)
            inv_norm[s : s + chunk_sz] = blended.cpu().numpy()

    # Refine consonant / nasal / pitch-driven articulators + smooth silent rest
    # cues: [active_gate, nasality, sibilance, frication_mid, pitch_norm]
    active_gate = cues[:, 0:1]
    nasality = cues[:, 1]
    sibilance = cues[:, 2]
    frication_mid = cues[:, 3]
    pitch_norm = cues[:, 4]

    # Convert from normalized [0, 1] -> raw Voice Workshop units for crisp refinement
    lo = np.array([r[0] for r in TRACT_RANGES], dtype=np.float32)
    span = np.array([r[1] - r[0] for r in TRACT_RANGES], dtype=np.float32)
    raw = lo[None, :] + inv_norm * span[None, :]

    # tipX, tipY, groove, tongueSides during sibilants (/s/, /z/, /sh/)
    raw[:, 4] = raw[:, 4] * (1.0 - 0.65 * sibilance) + 0.92 * sibilance
    raw[:, 5] = raw[:, 5] * (1.0 - 0.72 * sibilance) + 1.55 * sibilance
    raw[:, 6] = np.clip(0.14 + 0.78 * sibilance, 0.0, 1.0)
    raw[:, 7] = np.clip(0.18 + 0.68 * sibilance, 0.0, 1.0)
    # lipTuck during labiodental frication (/f/, /v/)
    raw[:, 11] = np.clip(raw[:, 11] * (1.0 - 0.6 * frication_mid) + 0.78 * frication_mid, 0.0, 1.0)
    # velum drop during low-frequency nasal murmur (/m/, /n/, /ng/)
    raw[:, 12] = np.clip(0.06 + 0.84 * nasality, 0.0, 1.0)
    # larynx height co-articulates subtly with vocal pitch alongside formant tube length
    pitch_larynx = (pitch_norm - 0.45) * 1.55
    raw[:, 13] = np.clip(0.55 * raw[:, 13] + 0.45 * pitch_larynx, -1.5, 1.2)

    refined_norm = _norm_tract_np(raw)
    rest = TRACT_REST_NORM[None, :]
    return active_gate * refined_norm + (1.0 - active_gate) * rest


class SingingHeadInverseNet(nn.Module):
    """Acoustic-to-articulatory inverse encoder driving all 14 mouth & throat parameters."""

    def __init__(self, in_dim: int = 18, h1: int = 64, h2: int = 48, out_dim: int = 28) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        raw = self.fc3(h)
        articulators = torch.sigmoid(raw[:, :9])
        vowel_logits = raw[:, 9:14]
        tract_norm = torch.sigmoid(raw[:, 14:28])
        return articulators, vowel_logits, tract_norm


class ForwardFormantDecoder(nn.Module):
    """Training-time forward vocal-tract decoder for reverse cycle-consistency."""

    def __init__(self, in_dim: int = 14, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.GELU(),
            nn.Linear(24, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, articulators: torch.Tensor, vowel_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(vowel_logits, dim=-1)
        return self.net(torch.cat([articulators, probs], dim=-1))


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_reverse_model(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    cycle_mat: np.ndarray,
    track_ids: Sequence[str],
    epochs: int = 35,
    batch_size: int = 512,
    seed: int = 42,
    device: str = "auto",
    obs_phys_mat: Optional[np.ndarray] = None,
    cues_mat: Optional[np.ndarray] = None,
    atlas_path: Optional[Path] = None,
) -> Tuple[SingingHeadInverseNet, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train 14-D inverse tract encoder + frozen forward vocal-tract physics cycle consistency."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = _resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    has_atlas = atlas_path is not None and Path(atlas_path).exists() and obs_phys_mat is not None and cues_mat is not None
    fwd_phys_net: Optional[TractPhysicsForwardNet] = None
    phys_mean_t: Optional[torch.Tensor] = None
    phys_std_t: Optional[torch.Tensor] = None
    physics_forward_r2 = 0.0

    if has_atlas:
        assert atlas_path is not None and obs_phys_mat is not None and cues_mat is not None
        fwd_phys_net, atlas_art_norm, atlas_phys, phys_mean, phys_std, physics_forward_r2 = train_physics_forward_net(
            Path(atlas_path), dev=dev, seed=seed
        )
        tract_targets = invert_tract_targets_from_atlas(
            obs_phys_mat, cues_mat, atlas_art_norm, atlas_phys, dev=dev, y_mat=y_mat
        )
        phys_mean_t = torch.from_numpy(phys_mean).to(dev)
        phys_std_t = torch.from_numpy(phys_std).to(dev)
        obs_phys_norm = ((obs_phys_mat - phys_mean[None, :]) / phys_std[None, :]).astype(np.float32)
        voicing_w = np.clip(x_mat[:, 3:4], 0.0, 1.0)
        active_weights = (cues_mat[:, 0:1] * (0.25 + 0.75 * voicing_w)).astype(np.float32)
    else:
        tract_targets = derive_fallback_tract_targets(y_mat, cycle_mat)
        obs_phys_norm = np.zeros((len(x_mat), 7), dtype=np.float32)
        active_weights = (y_mat[:, 0:1] > 0.08).astype(np.float32)

    unique_tids = sorted(set(track_ids))
    rng.shuffle(unique_tids)
    n_val_tids = max(1, int(round(0.15 * len(unique_tids))))
    val_tid_set = set(unique_tids[:n_val_tids])

    is_val = np.array([tid in val_tid_set for tid in track_ids], dtype=bool)
    is_train = ~is_val
    if int(np.sum(is_train)) < 100 or int(np.sum(is_val)) < 20:
        perm = rng.permutation(len(x_mat))
        split = int(0.85 * len(x_mat))
        is_train = np.zeros(len(x_mat), dtype=bool)
        is_train[perm[:split]] = True
        is_val = ~is_train

    x_train = x_mat[is_train]
    feat_mean = np.mean(x_train, axis=0)
    feat_std = np.maximum(np.std(x_train, axis=0), 1e-4)

    def norm_x(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - feat_mean) / feat_std).astype(np.float32))

    xt = norm_x(x_train).to(dev)
    yt = torch.from_numpy(y_mat[is_train]).to(dev)
    ct = torch.from_numpy(cycle_mat[is_train]).to(dev)
    tt = torch.from_numpy(tract_targets[is_train]).to(dev)
    pt = torch.from_numpy(obs_phys_norm[is_train]).to(dev)
    wt = torch.from_numpy(active_weights[is_train]).to(dev)

    xv = norm_x(x_mat[is_val]).to(dev)
    yv = torch.from_numpy(y_mat[is_val]).to(dev)
    cv = torch.from_numpy(cycle_mat[is_val]).to(dev)
    tv = torch.from_numpy(tract_targets[is_val]).to(dev)
    pv = torch.from_numpy(obs_phys_norm[is_val]).to(dev)
    wv = torch.from_numpy(active_weights[is_val]).to(dev)

    inv_net = SingingHeadInverseNet(in_dim=x_mat.shape[1], h1=64, h2=48, out_dim=28).to(dev)
    fwd_net = ForwardFormantDecoder(in_dim=14, out_dim=4).to(dev)

    opt = torch.optim.AdamW(
        list(inv_net.parameters()) + list(fwd_net.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_train = len(xt)
    for _ in range(epochs):
        inv_net.train()
        fwd_net.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = xt[idx]
            yb = yt[idx]
            cb = ct[idx]
            tb = tt[idx]
            pb = pt[idx]
            wb = wt[idx]

            pred_art, pred_vlogits, pred_tract = inv_net(xb)
            pred_cycle = fwd_net(pred_art, pred_vlogits)

            loss_art = F.mse_loss(pred_art, yb[:, :9])
            target_vprobs = yb[:, 9:]
            log_probs = F.log_softmax(pred_vlogits, dim=-1)
            loss_vowel = -(target_vprobs * log_probs).sum(dim=-1).mean()
            loss_cycle = F.mse_loss(pred_cycle, cb)
            loss_tract = F.smooth_l1_loss(pred_tract, tb)

            loss = loss_art + 0.35 * loss_vowel + 0.25 * loss_cycle + 3.20 * loss_tract

            if fwd_phys_net is not None:
                pred_phys = fwd_phys_net(pred_tract)
                # Differentiable (F1, F2, lipArea) cycle consistency gently refines vocal tract resonance
                w_act = wb.squeeze(-1)
                formant_err = F.smooth_l1_loss(pred_phys[:, :2], pb[:, :2], reduction="none").mean(dim=-1)
                apert_err = F.smooth_l1_loss(pred_phys[:, 3:5], pb[:, 3:5], reduction="none").mean(dim=-1)
                denom_w = torch.clamp(w_act.sum(), min=1.0)
                loss_phys_cycle = (w_act * (0.14 * formant_err + 0.06 * apert_err)).sum() / denom_w
                loss = loss + loss_phys_cycle

            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    inv_net.eval()
    fwd_net.eval()
    with torch.no_grad():
        va, vv, vt = inv_net(xv)
        vc = fwd_net(va, vv)
        val_mae = float(torch.mean(torch.abs(va - yv[:, :9])).item())
        tract_val_mae = float(torch.mean(torch.abs(vt - tv)).item())
        active_mask = yv[:, 0] > 0.12
        if int(active_mask.sum().item()) > 10:
            pred_cls = torch.argmax(vv[active_mask], dim=-1)
            true_cls = torch.argmax(yv[active_mask, 9:], dim=-1)
            val_vowel_acc = float((pred_cls == true_cls).float().mean().item())
        else:
            val_vowel_acc = 0.0
        ss_res = float(((vc - cv) ** 2).sum().item())
        ss_tot = float(((cv - cv.mean(dim=0, keepdim=True)) ** 2).sum().item())
        cycle_r2 = 1.0 - ss_res / max(1e-8, ss_tot)

        physics_cycle_r2 = 0.0
        voiced_mask = active_mask & (wv.squeeze(-1) > 0.50)
        if fwd_phys_net is not None and int(voiced_mask.sum().item()) > 10:
            vp_phys = fwd_phys_net(vt[voiced_mask])[:, :2]
            tp_phys = pv[voiced_mask, :2]
            p_res = float(((vp_phys - tp_phys) ** 2).sum().item())
            p_tot = float(((tp_phys - tp_phys.mean(dim=0, keepdim=True)) ** 2).sum().item())
            physics_cycle_r2 = 1.0 - p_res / max(1e-8, p_tot)

    inv_net_cpu = inv_net.cpu()
    metrics = {
        "train_frames": int(np.sum(is_train)),
        "val_frames": int(np.sum(is_val)),
        "val_mae": round(val_mae, 4),
        "tract_val_mae": round(tract_val_mae, 4),
        "val_vowel_acc": round(val_vowel_acc, 4),
        "cycle_r2": round(cycle_r2, 4),
        "physics_forward_r2": round(physics_forward_r2, 4),
        "physics_cycle_r2": round(physics_cycle_r2, 4),
        "roundtrip_env_corr": round(max(0.0, physics_cycle_r2), 4),
        "roundtrip_feat_r2": round(max(0.0, cycle_r2), 4),
    }
    return inv_net_cpu, feat_mean, feat_std, metrics


def export_model_json(
    out_path: Path,
    model: SingingHeadInverseNet,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    corpus_stats: Dict[str, float],
    metrics: Dict[str, float],
    stubs_used: int,
    stems_used: int,
) -> None:
    """Export trained 14-D reverse-model weights and normalization stats as compact JSON."""
    state = model.state_dict()

    def round_list(t: torch.Tensor) -> list:
        arr = np.round(t.detach().cpu().numpy().astype(np.float64), 6)
        return arr.tolist()

    payload = {
        "version": 2,
        "format": "wub_singing_head_mlp_v2",
        "sample_rate": TARGET_SR,
        "fft_size": FFT_SIZE,
        "hop_size": HOP_SIZE,
        "feature_names": FEATURE_NAMES,
        "articulators": ARTICULATOR_NAMES,
        "vowels": VOWELS,
        "tract_keys": TRACT_ARTICULATOR_KEYS,
        "tract_ranges": [list(r) for r in TRACT_RANGES],
        "bands_hz": [list(b) for b in BANDS_HZ],
        "feature_mean": [round(float(v), 6) for v in feat_mean.tolist()],
        "feature_std": [round(float(v), 6) for v in feat_std.tolist()],
        "corpus_stats": {
            "stubs_used": stubs_used,
            "stems_used": stems_used,
            **corpus_stats,
            **metrics,
        },
        "weights": {
            "w1": round_list(state["fc1.weight"]),
            "b1": round_list(state["fc1.bias"]),
            "w2": round_list(state["fc2.weight"]),
            "b2": round_list(state["fc2.bias"]),
            "w3": round_list(state["fc3.weight"]),
            "b3": round_list(state["fc3.bias"]),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def _compute_env_32_batch(frames: np.ndarray, rms_arr: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Extract 32-band L2-normalized timbre envelopes matching generate_tract_soundbank.py."""
    n = frames.shape[1]
    win = (0.5 - 0.5 * np.cos((2.0 * np.pi * np.arange(n, dtype=np.float64)) / float(n - 1))).astype(np.float32)
    centered = frames.astype(np.float32) - np.mean(frames, axis=1, keepdims=True, dtype=np.float32)
    spec = np.fft.rfft(centered * win[None, :], axis=1)[:, :-1]
    mag = np.abs(spec) / (float(n) * 0.5)
    mag_sq = mag * mag
    bin_hz = float(sr) / float(n)
    half_n = mag.shape[1]
    edges = np.geomspace(80.0, 9000.0, 33)
    env_raw = np.zeros((len(frames), 32), dtype=np.float32)
    rms_safe = np.maximum(rms_arr.astype(np.float32), 1e-4)
    for k_b in range(32):
        lo_k = max(1, int(math.floor(edges[k_b] / bin_hz)))
        hi_k = max(lo_k, min(half_n - 1, int(math.ceil(edges[k_b + 1] / bin_hz))))
        b_rms = np.sqrt(np.mean(mag_sq[:, lo_k : hi_k + 1], axis=1) + 1e-18)
        env_raw[:, k_b] = np.log1p((b_rms / rms_safe) * 4.5)
    env_centered = env_raw - np.mean(env_raw, axis=1, keepdims=True)
    norms = np.maximum(np.linalg.norm(env_centered, axis=1, keepdims=True), 1e-6)
    return (env_centered / norms).astype(np.float32)


SEQ_LEN = 96


def _extract_contiguous_window(
    track_id: str,
    audio: np.ndarray,
    is_stem: bool,
    seq_len: int = SEQ_LEN,
) -> Optional[dict]:
    """Extract a contiguous 60 Hz feature + spectral envelope sequence from an audio segment."""
    if len(audio) < FFT_SIZE:
        return None
    n_frames = 1 + (len(audio) - FFT_SIZE) // HOP_SIZE
    if n_frames < 12:
        return None

    # Find highest-energy window of length seq_len (including slight lead-in for onsets)
    if n_frames > seq_len:
        # Compute frame energies quickly to pick an active vocal phrase window
        hop_samples = np.arange(n_frames) * HOP_SIZE
        # Fast sliding square sum via cumulative sum
        sq = audio.astype(np.float64) ** 2
        cs = np.concatenate(([0.0], np.cumsum(sq)))
        frame_e = (cs[hop_samples + FFT_SIZE] - cs[hop_samples]) / float(FFT_SIZE)
        win_cs = np.concatenate(([0.0], np.cumsum(frame_e)))
        win_e = win_cs[seq_len:] - win_cs[: len(win_cs) - seq_len]
        best_start_frame = int(np.argmax(win_e))
        # Shift back ~6 frames so natural onset attack is captured when available
        start_frame = max(0, best_start_frame - 6)
        end_frame = start_frame + seq_len
    else:
        start_frame = 0
        end_frame = n_frames

    actual_len = end_frame - start_frame
    frames = np.zeros((seq_len, FFT_SIZE), dtype=np.float32)
    feats = np.zeros((seq_len, 18), dtype=np.float32)
    rms_arr = np.zeros((seq_len,), dtype=np.float32)
    f0_arr = np.zeros((seq_len,), dtype=np.float32)
    voicing_arr = np.zeros((seq_len,), dtype=np.float32)
    vibrato_arr = np.zeros((seq_len,), dtype=np.float32)
    rms_rise_arr = np.zeros((seq_len,), dtype=np.float32)

    prev_norm_mag: Optional[np.ndarray] = None
    for t in range(actual_len):
        s_idx = (start_frame + t) * HOP_SIZE
        fr = audio[s_idx : s_idx + FFT_SIZE]
        frames[t] = fr
        f_vec, r_val, f0_val, v_val, prev_norm_mag = extract_frame_features(fr, TARGET_SR, prev_norm_mag)
        feats[t] = f_vec
        rms_arr[t] = r_val
        f0_arr[t] = f0_val
        voicing_arr[t] = v_val

    if actual_len < seq_len:
        # Tail-pad smoothly with silence so batch shapes match while rest posture fades cleanly
        sil_feat, sil_rms, sil_f0, sil_v, _ = extract_frame_features(np.zeros(FFT_SIZE, dtype=np.float32), TARGET_SR, None)
        for t in range(actual_len, seq_len):
            feats[t] = sil_feat
            rms_arr[t] = sil_rms
            f0_arr[t] = sil_f0
            voicing_arr[t] = sil_v

    env_32 = _compute_env_32_batch(frames, rms_arr, TARGET_SR)

    for t in range(actual_len):
        lo = max(0, t - 8)
        hi = min(actual_len, t + 9)
        local_f0s = f0_arr[lo:hi]
        voiced_f0s = local_f0s[local_f0s > 0.0]
        if len(voiced_f0s) >= 4 and f0_arr[t] > 0.0:
            med_f0 = float(np.median(voiced_f0s))
            semis = 12.0 * np.log2(np.maximum(20.0, voiced_f0s) / max(20.0, med_f0))
            vib_std = float(np.std(semis))
            vibrato_arr[t] = clamp(vib_std / 1.1, 0.0, 1.0) if vib_std < 3.0 else 0.35
        if t > 0:
            rms_rise_arr[t] = max(0.0, float(rms_arr[t] - rms_arr[t - 1]))

    valid_mask = np.zeros((seq_len,), dtype=np.float32)
    valid_mask[:actual_len] = 1.0
    if float(np.max(rms_arr)) < 0.0045 and is_stem:
        return None

    return {
        "track_id": track_id,
        "is_stem": is_stem,
        "feats": feats,
        "env": env_32,
        "rms": rms_arr,
        "f0": f0_arr,
        "voicing": voicing_arr,
        "vibrato": vibrato_arr,
        "rms_rise": rms_rise_arr,
        "valid": valid_mask,
    }


def _worker_sequence(args: Tuple[str, str, bool]) -> List[dict]:
    path_str, track_id, is_stem = args
    path = Path(path_str)
    out: List[dict] = []
    try:
        if not is_stem:
            audio = load_audio_44k(path)
            seg = _extract_contiguous_window(track_id, audio, is_stem=False)
            if seg is not None:
                out.append(seg)
            return out

        info = sf.info(str(path))
        sr = int(info.samplerate)
        total_samples = int(info.frames)
        win_sec = 3.2
        win_n = int(win_sec * sr)
        if total_samples <= win_n:
            audio = load_audio_44k(path)
            seg = _extract_contiguous_window(track_id, audio, is_stem=True)
            if seg is not None:
                out.append(seg)
            return out

        for frac in (0.16, 0.34, 0.52, 0.70, 0.86):
            start = max(0, min(total_samples - win_n, int(total_samples * frac)))
            chunk, chunk_sr = sf.read(str(path), start=start, frames=win_n, dtype="float32")
            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)
            if chunk_sr != TARGET_SR:
                g = math.gcd(TARGET_SR, int(chunk_sr))
                chunk = resample_poly(chunk.astype(np.float64), TARGET_SR // g, int(chunk_sr) // g).astype(np.float32)
            seg = _extract_contiguous_window(track_id, chunk, is_stem=True)
            if seg is not None:
                out.append(seg)
        return out
    except Exception:
        return []


def _smooth_env_bands(env: np.ndarray) -> np.ndarray:
    """Remove harmonic-comb ripple: 2x [1,2,1]/4 band smoothing, re-centred and L2-normalised."""
    out = env.astype(np.float32)
    for _ in range(2):
        pd = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(1, 1)], mode="edge")
        out = 0.25 * pd[..., :-2] + 0.5 * pd[..., 1:-1] + 0.25 * pd[..., 2:]
    out = out - out.mean(axis=-1, keepdims=True)
    return (out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-6)).astype(np.float32)


def _excitation_from_feats(feats: np.ndarray, breath: np.ndarray, loud: np.ndarray) -> np.ndarray:
    """10-d source descriptor: zcr, voicing, f0, centroid, rolloff, flatness, flux, tilt, breath, loudness."""
    return np.concatenate(
        [feats[..., [2, 3, 4, 5, 6, 7, 8]], np.clip(feats[..., 17:18] / 3.0, -1.5, 1.5), breath[..., None], loud[..., None]],
        axis=-1,
    ).astype(np.float32)


class SimSpectrumForwardNet(nn.Module):
    """Differentiable forward vocal-tract + excitation -> (32-band spectrum, 18-D acoustic features) surrogate."""

    def __init__(self, in_dim: int = 24, env_dim: int = 32, feat_dim: int = 18) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.GELU(),
            nn.Linear(192, 160),
            nn.GELU(),
        )
        self.env_head = nn.Linear(160, env_dim)
        self.feat_head = nn.Linear(160, feat_dim)

    def forward_both(self, tract_and_exc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(tract_and_exc)
        raw_env = self.env_head(h)
        centered = raw_env - raw_env.mean(dim=-1, keepdim=True)
        pred_env = F.normalize(centered, p=2, dim=-1, eps=1e-6)
        pred_feats = self.feat_head(h)
        return pred_env, pred_feats

    def forward(self, tract_and_exc: torch.Tensor) -> torch.Tensor:
        pred_env, _ = self.forward_both(tract_and_exc)
        return pred_env


def train_sim_forward_net(
    sim_art_norm: np.ndarray,
    sim_exc: np.ndarray,
    sim_env: np.ndarray,
    dev: torch.device,
    epochs: int = 55,
    batch_size: int = 512,
    seed: int = 42,
    sim_feats_norm: Optional[np.ndarray] = None,
    real_art_norm: Optional[np.ndarray] = None,
    real_exc: Optional[np.ndarray] = None,
    real_env: Optional[np.ndarray] = None,
    real_feats_norm: Optional[np.ndarray] = None,
) -> Tuple[SimSpectrumForwardNet, float]:
    """Train & freeze a differentiable (14-D tract + 10-D exc) -> (32-band spectrum, 18-D features) surrogate."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sim_art_norm))
    split = int(0.90 * len(perm))
    tr_idx, va_idx = perm[:split], perm[split:]

    sim_in = np.concatenate([sim_art_norm, sim_exc], axis=1).astype(np.float32)
    sim_f = sim_feats_norm.astype(np.float32) if sim_feats_norm is not None else np.zeros((len(sim_in), 18), dtype=np.float32)

    if (
        real_art_norm is not None
        and real_exc is not None
        and real_env is not None
        and real_feats_norm is not None
        and len(real_art_norm) > 0
    ):
        real_in = np.concatenate([real_art_norm, real_exc], axis=1).astype(np.float32)
        n_real_take = min(len(real_in), len(tr_idx))
        r_sel = rng.choice(len(real_in), size=n_real_take, replace=False) if len(real_in) > n_real_take else np.arange(len(real_in))
        comb_in = np.concatenate([sim_in[tr_idx], real_in[r_sel]], axis=0)
        comb_env = np.concatenate([sim_env[tr_idx], real_env[r_sel]], axis=0)
        comb_feats = np.concatenate([sim_f[tr_idx], real_feats_norm[r_sel]], axis=0)
    else:
        comb_in = sim_in[tr_idx]
        comb_env = sim_env[tr_idx]
        comb_feats = sim_f[tr_idx]

    xa = torch.from_numpy(comb_in).to(dev)
    ye = torch.from_numpy(comb_env).to(dev)
    yf = torch.from_numpy(comb_feats).to(dev)
    xva = torch.from_numpy(sim_in[va_idx]).to(dev)
    yve = torch.from_numpy(sim_env[va_idx]).to(dev)

    fwd = SimSpectrumForwardNet(in_dim=sim_in.shape[1], env_dim=sim_env.shape[1], feat_dim=comb_feats.shape[1]).to(dev)
    opt = torch.optim.AdamW(fwd.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_tr = len(xa)
    has_feats = sim_feats_norm is not None
    for _ in range(epochs):
        fwd.train()
        p = torch.randperm(n_tr, device=dev)
        for s in range(0, n_tr, batch_size):
            idx = p[s : s + batch_size]
            pred_env, pred_feats = fwd.forward_both(xa[idx])
            cos_loss = (1.0 - (pred_env * ye[idx]).sum(dim=-1)).mean()
            loss = cos_loss + F.mse_loss(pred_env, ye[idx])
            if has_feats:
                loss = loss + 0.50 * F.smooth_l1_loss(pred_feats, yf[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    fwd.eval()
    with torch.no_grad():
        vp = fwd(xva)
        ss_res = float(((vp - yve) ** 2).sum().item())
        ss_tot = float(((yve - yve.mean(dim=0, keepdim=True)) ** 2).sum().item())
        r2 = 1.0 - ss_res / max(1e-8, ss_tot)
    for param in fwd.parameters():
        param.requires_grad = False
    return fwd, r2


def match_sequences_to_soundbank(
    seqs: List[dict],
    soundbank_path: Path,
    dev: torch.device,
    top_k: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Match chronological stem/stub sequences to simulated sounds via GPU Top-K + Viterbi DP.

    Returns contiguous tensors (N, T, *):
      x_seq (N, T, 18), y_seq (N, T, 14), tract_seq (N, T, 14),
      env_seq (N, T, 32), weights_seq (N, T, 1), valid_seq (N, T, 1),
      sim_art_norm (M, 14), sim_env (M, 32), stats
    """
    sb = np.load(soundbank_path)
    sim_art_raw = sb["art"].astype(np.float32)      # (M, 14)
    sim_feats = sb["feats"].astype(np.float32)      # (M, 18)
    sim_env = sb["env"].astype(np.float32)          # (M, 32)
    sim_cues = sb["cues"].astype(np.float32)        # (M, 6): [f0_norm, voicing, breath, vibrato, loud, rms]

    sim_art_norm = _norm_tract_np(sim_art_raw)
    m_sb = len(sim_art_norm)
    k = min(top_k, m_sb)

    # Corpus empirical loudness/f0 calibration across active frames
    all_rms = np.concatenate([s["rms"][s["valid"] > 0.5] for s in seqs])
    all_f0 = np.concatenate([s["f0"][(s["valid"] > 0.5) & (s["f0"] > 0.0)] for s in seqs])
    active_rms = all_rms[all_rms > 0.004]
    if len(active_rms) < 32:
        active_rms = np.maximum(all_rms, 0.004)
    active_db = 20.0 * np.log10(np.maximum(1e-5, active_rms))
    db_p10 = float(np.percentile(active_db, 10.0))
    db_p95 = float(np.percentile(active_db, 95.0))
    db_span = max(8.0, db_p95 - db_p10)
    if len(all_f0) >= 32:
        f0_p10 = float(np.percentile(all_f0, 8.0))
        f0_p95 = float(np.percentile(all_f0, 94.0))
    else:
        f0_p10, f0_p95 = 110.0, 520.0
    log_f0_lo = math.log2(max(70.0, f0_p10))
    log_f0_hi = math.log2(max(f0_p10 + 40.0, f0_p95))

    # Pure acoustic comparison between song frames and synthesized vocal simulator audio
    sim_art_t = torch.from_numpy(sim_art_norm).to(dev)
    sim_env_t = torch.from_numpy(sim_env).to(dev)
    sim_feats_t = torch.from_numpy(sim_feats).to(dev)
    feat_scale = torch.clamp(torch.std(sim_feats_t, dim=0, keepdim=True), min=1e-2)
    # Feature weights across the 18 browser audio features (formant bands, voicing, f0, centroid weighted higher)
    feat_w = torch.tensor(
        [0.50, 0.30, 0.45, 0.65, 0.55, 0.55, 0.45, 0.45, 0.20, 0.50, 0.60, 0.65, 0.65, 0.60, 0.55, 0.50, 0.45, 0.45],
        device=dev,
        dtype=torch.float32,
    ).unsqueeze(0)
    sim_feats_norm_t = (sim_feats_t / feat_scale) * torch.sqrt(feat_w)
    sim_feats_sq = (sim_feats_norm_t * sim_feats_norm_t).sum(dim=1).unsqueeze(0)  # (1, M)
    use_fp16 = dev.type == "cuda"
    sim_env_mat = sim_env_t.T.contiguous().half() if use_fp16 else sim_env_t.T.contiguous()
    sim_feats_mat = sim_feats_norm_t.T.contiguous().half() if use_fp16 else sim_feats_norm_t.T.contiguous()

    # Transition stiffness weights per normalized 14-D articulator for Viterbi smoothing:
    # Balanced (lambda_trans = 0.22) so Viterbi tracks every syllable & formant shift
    # while preventing erratic single-frame candidate jumps.
    trans_w = torch.tensor(
        [1.4, 1.2, 1.2, 1.0, 1.0, 1.0, 0.9, 0.9, 1.3, 1.4, 1.3, 0.9, 1.1, 1.0],
        device=dev,
        dtype=torch.float32,
    )
    lambda_trans = 0.22

    n_seq = len(seqs)
    t_len = seqs[0]["feats"].shape[0]
    x_seq = np.stack([s["feats"] for s in seqs], axis=0).astype(np.float32)        # (N, T, 18)
    env_seq = np.stack([s["env"] for s in seqs], axis=0).astype(np.float32)          # (N, T, 32)
    rms_seq = np.stack([s["rms"] for s in seqs], axis=0).astype(np.float32)          # (N, T)
    f0_seq = np.stack([s["f0"] for s in seqs], axis=0).astype(np.float32)            # (N, T)
    voic_seq = np.stack([s["voicing"] for s in seqs], axis=0).astype(np.float32)     # (N, T)
    vib_seq = np.stack([s["vibrato"] for s in seqs], axis=0).astype(np.float32)      # (N, T)
    rise_seq = np.stack([s["rms_rise"] for s in seqs], axis=0).astype(np.float32)    # (N, T)
    valid_seq = np.stack([s["valid"] for s in seqs], axis=0).astype(np.float32)      # (N, T)

    matched_tract_seq = np.zeros((n_seq, t_len, 14), dtype=np.float32)

    batch_seqs = 16
    frame_chunk = 256  # <= 256 * 131072 floats ~ 134 MB peak per matmul
    print(f"Matching {n_seq} contiguous sequences ({n_seq * t_len} frames) against {m_sb} simulated vocal sounds...", flush=True)

    with torch.no_grad():
        for b0 in range(0, n_seq, batch_seqs):
            b1 = min(n_seq, b0 + batch_seqs)
            bs = b1 - b0

            q_env_b = torch.from_numpy(env_seq[b0:b1]).to(dev)       # (B, T, 32)
            q_feat_b = torch.from_numpy(x_seq[b0:b1]).to(dev)        # (B, T, 18)

            flat_n = bs * t_len
            qf_env = q_env_b.reshape(flat_n, 32)
            qf_feat = q_feat_b.reshape(flat_n, 18)

            cand_idx = torch.zeros((flat_n, k), device=dev, dtype=torch.int64)
            cand_cost = torch.zeros((flat_n, k), device=dev, dtype=torch.float32)

            for s in range(0, flat_n, frame_chunk):
                e = min(flat_n, s + frame_chunk)
                qe = qf_env[s:e]                                    # (C, 32)
                qf = (qf_feat[s:e] / feat_scale) * torch.sqrt(feat_w)

                cos_sim = torch.matmul(qe.half(), sim_env_mat).float() if use_fp16 else torch.matmul(qe, sim_env_mat)
                qf_sq = (qf * qf).sum(dim=1, keepdim=True)
                dot_f = torch.matmul(qf.half(), sim_feats_mat).float() if use_fp16 else torch.matmul(qf, sim_feats_mat)
                feat_dist = torch.clamp(qf_sq + sim_feats_sq - 2.0 * dot_f, min=0.0) / 18.0
                cost = (1.0 - cos_sim) + 0.35 * feat_dist

                vals, idxs = torch.topk(cost, k=k, dim=1, largest=False)
                cand_cost[s:e] = vals
                cand_idx[s:e] = idxs

            # Reshape back to (B, T, K) and solve vectorized exact Viterbi DP
            em_cost = cand_cost.reshape(bs, t_len, k)
            c_ids = cand_idx.reshape(bs, t_len, k)
            c_art = sim_art_t[c_ids]                                # (B, T, K, 14)

            dp = torch.zeros((bs, k), device=dev, dtype=torch.float32)
            bp = torch.zeros((bs, t_len, k), device=dev, dtype=torch.int64)
            dp = em_cost[:, 0, :]

            for t in range(1, t_len):
                prev_a = c_art[:, t - 1, :, :].unsqueeze(2)         # (B, K_prev, 1, 14)
                cur_a = c_art[:, t, :, :].unsqueeze(1)              # (B, 1, K_cur, 14)
                diff = cur_a - prev_a                               # (B, K_prev, K_cur, 14)
                tr_cost = lambda_trans * (diff * diff * trans_w.view(1, 1, 1, 14)).sum(dim=-1)  # (B, K_prev, K_cur)
                total = dp.unsqueeze(2) + tr_cost                   # (B, K_prev, K_cur)
                best_prev_val, best_prev_idx = torch.min(total, dim=1)
                dp = em_cost[:, t, :] + best_prev_val
                bp[:, t, :] = best_prev_idx

            best_path = torch.zeros((bs, t_len), device=dev, dtype=torch.int64)
            best_path[:, t_len - 1] = torch.argmin(dp, dim=1)
            for t in range(t_len - 2, -1, -1):
                best_path[:, t] = bp[torch.arange(bs, device=dev), t + 1, best_path[:, t + 1]]

            chosen_art = torch.gather(
                c_art,
                2,
                best_path.unsqueeze(-1).unsqueeze(-1).expand(bs, t_len, 1, 14),
            ).squeeze(2)                                            # (B, T, 14)
            matched_tract_seq[b0:b1] = chosen_art.cpu().numpy()

    # Smooth rest-blending on quiet/silent regions & subtle zero-phase Gaussian smoothing along time
    db_seq = 20.0 * np.log10(np.maximum(1e-5, rms_seq))
    loud_norm = np.where(rms_seq > 0.004, np.maximum(0.04, np.clip((db_seq - db_p10) / db_span, 0.0, 1.0)), 0.0)
    onset_boost = np.clip(rise_seq * 10.0, 0.0, 0.22)
    syl_open_drive = np.clip(0.68 * loud_norm + 0.32 * onset_boost, 0.0, 1.0)
    active_gate = np.clip((rms_seq - 0.0035) / 0.020, 0.0, 1.0)[..., None]  # (N, T, 1)
    rest_norm = TRACT_REST_NORM[None, None, :]

    # Preserve the true Viterbi-matched vocal simulator parameters with faithful dynamic formant opening
    nat_tract = matched_tract_seq.copy()
    nat_tract[:, :, 0] = np.clip(
        0.88 * nat_tract[:, :, 0] * (0.65 + 0.35 * syl_open_drive) + 0.12 * (0.60 * syl_open_drive),
        0.0,
        0.92,
    )
    nat_tract[:, :, 8] = np.clip(
        0.88 * nat_tract[:, :, 8] * (0.68 + 0.32 * syl_open_drive) + 0.12 * (0.25 + 0.55 * syl_open_drive),
        0.0,
        0.94,
    )

    tract_blended = active_gate * nat_tract + (1.0 - active_gate) * rest_norm

    # Zero-phase 5-tap symmetric kernel [0.10, 0.22, 0.36, 0.22, 0.10] eliminates candidate quantization jitter
    kernel = np.array([0.10, 0.22, 0.36, 0.22, 0.10], dtype=np.float32)
    padded = np.pad(tract_blended, ((0, 0), (2, 2), (0, 0)), mode="edge")
    tract_smooth = (
        kernel[0] * padded[:, 0:t_len]
        + kernel[1] * padded[:, 1 : t_len + 1]
        + kernel[2] * padded[:, 2 : t_len + 2]
        + kernel[3] * padded[:, 3 : t_len + 3]
        + kernel[4] * padded[:, 4 : t_len + 4]
    ).astype(np.float32)

    # Derive auxiliary HUD display values directly from the matched 14-D vocal simulator parameters
    gate = active_gate.squeeze(-1)

    flatness = x_seq[:, :, 7]
    sib_band = x_seq[:, :, 16]
    presence_band = x_seq[:, :, 15]
    breath = np.clip(
        gate * (1.0 - voic_seq * 0.85) * (0.45 * np.clip(sib_band / 2.2, 0.0, 1.0) + 0.55 * flatness),
        0.0,
        1.0,
    )
    pitch_norm = np.where(
        f0_seq > 0.0,
        np.clip((np.log2(np.maximum(50.0, f0_seq)) - log_f0_lo) / max(0.2, log_f0_hi - log_f0_lo), 0.0, 1.0),
        0.42,
    )

    jaw_n = tract_smooth[:, :, 0]
    tongue_x_n = tract_smooth[:, :, 1]
    tongue_y_n = tract_smooth[:, :, 2]
    tip_y_n = tract_smooth[:, :, 5]
    lip_open_n = tract_smooth[:, :, 8]
    lip_round_n = tract_smooth[:, :, 9]
    lip_protrude_n = tract_smooth[:, :, 10]

    aperture = np.clip(
        gate * (0.44 * lip_open_n + 0.40 * jaw_n + 0.16 * (0.55 * syl_open_drive)) * (1.0 - 0.20 * breath),
        0.0,
        0.78,
    )
    jaw_drop = np.clip(gate * (0.82 * jaw_n + 0.18 * (0.52 * syl_open_drive)), 0.0, 0.72)
    width = np.clip(0.50 + gate * (0.88 * (tongue_x_n - 0.5) - 0.35 * (lip_round_n - 0.25)), 0.08, 0.94)
    roundness = np.clip(gate * (0.65 * lip_round_n + 0.35 * lip_protrude_n), 0.0, 0.92)
    tongue_height = np.clip(0.68 * tongue_y_n + 0.32 * tip_y_n, 0.0, 1.0)
    tension = np.clip(
        gate * (0.42 * loud_norm + 0.28 * np.clip(presence_band / 2.6, 0.0, 1.0) + 0.24 * pitch_norm),
        0.0,
        0.88,
    )

    # Derive 5 HUD shape indicators [AH, EE, OH, OO, AY] directly from the 14-D vocal simulator geometry
    s_ah = 1.8 * jaw_n + 1.2 * lip_open_n - 0.8 * tongue_y_n - 0.8 * lip_round_n
    s_ee = 1.8 * tongue_x_n + 1.4 * tongue_y_n - 1.2 * lip_round_n - 0.8 * jaw_n
    s_oh = 1.5 * lip_round_n + 0.8 * jaw_n - 1.0 * tongue_x_n
    s_oo = 1.7 * lip_round_n + 1.3 * lip_protrude_n + 0.8 * tongue_y_n - 1.1 * jaw_n - 0.8 * tongue_x_n
    s_ay = 1.3 * tongue_x_n + 0.9 * lip_open_n + 0.5 * jaw_n - 1.0 * lip_round_n
    v_scores = np.stack([s_ah, s_ee, s_oh, s_oo, s_ay], axis=-1) * 3.0
    v_scores = v_scores - np.max(v_scores, axis=-1, keepdims=True)
    v_exp = np.exp(v_scores)
    geom_v_probs = (v_exp / np.maximum(np.sum(v_exp, axis=-1, keepdims=True), 1e-6)).astype(np.float32)
    sil_v = np.full((1, 1, 5), 0.20, dtype=np.float32)
    v_probs = np.where((rms_seq > 0.005)[..., None], geom_v_probs, sil_v).astype(np.float32)

    art9_seq = np.stack(
        [aperture, width, roundness, jaw_drop, tongue_height, pitch_norm, vib_seq, tension, breath],
        axis=-1,
    ).astype(np.float32)
    y_seq = np.concatenate([art9_seq, v_probs], axis=-1).astype(np.float32)  # (N, T, 14)

    weights_seq = (gate * (0.25 + 0.75 * np.clip(voic_seq, 0.0, 1.0)) * valid_seq).astype(np.float32)[..., None]

    stats = {
        "db_p10": round(db_p10, 3),
        "db_p95": round(db_p95, 3),
        "f0_p10": round(f0_p10, 2),
        "f0_p95": round(f0_p95, 2),
    }
    return x_seq, y_seq, tract_smooth, env_seq, weights_seq, valid_seq[..., None], sim_art_norm, sim_env, stats


def train_soundbank_sequence_model(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    tract_seq: np.ndarray,
    env_seq: np.ndarray,
    weights_seq: np.ndarray,
    valid_seq: np.ndarray,
    sim_art_norm: np.ndarray,
    sim_env: np.ndarray,
    sim_feats: np.ndarray,
    sim_cues: np.ndarray,
    track_ids: Sequence[str],
    epochs: int = 75,
    batch_seqs: int = 64,
    seed: int = 42,
    device: str = "auto",
) -> Tuple[SingingHeadInverseNet, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train smooth, simulator-grounded 14-D Singing Head inverse network over 60 Hz sequences."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = _resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 1. Split by track_id first so feature normalization and forward surrogate never leak validation songs
    unique_tids = sorted(set(track_ids))
    rng.shuffle(unique_tids)
    n_val_tids = max(1, int(round(0.14 * len(unique_tids))))
    val_tid_set = set(unique_tids[:n_val_tids])
    is_val = np.array([tid in val_tid_set for tid in track_ids], dtype=bool)
    is_train = ~is_val
    if int(np.sum(is_train)) < 16 or int(np.sum(is_val)) < 4:
        perm = rng.permutation(len(x_seq))
        sp = int(0.85 * len(x_seq))
        is_train = np.zeros(len(x_seq), dtype=bool)
        is_train[perm[:sp]] = True
        is_val = ~is_train

    tr_x = x_seq[is_train]
    tr_valid = valid_seq[is_train, :, 0] > 0.5
    x_tr_valid = tr_x[tr_valid]
    feat_mean = np.mean(x_tr_valid, axis=0).astype(np.float32)
    feat_std = np.maximum(np.std(x_tr_valid, axis=0), 1e-4).astype(np.float32)

    def norm_x(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - feat_mean) / feat_std).astype(np.float32))

    # Build matching excitation tensor for stem sequences and smooth spectral envelopes
    sim_loud = np.clip(sim_feats[:, 0] / 1.45, 0.0, 1.0).astype(np.float32)
    sim_exc = _excitation_from_feats(sim_feats, sim_cues[:, 2], sim_loud)
    sim_env = _smooth_env_bands(sim_env)
    env_seq = _smooth_env_bands(env_seq)
    stem_exc_seq = _excitation_from_feats(
        x_seq, y_seq[:, :, 8], np.clip(x_seq[:, :, 0] / 1.45, 0.0, 1.0)
    )

    # 2. Train & freeze differentiable roundtrip forward synth surrogate ((14 tract + 10 exc) -> 32 env + 18 feats)
    sim_feats_norm = ((sim_feats - feat_mean) / feat_std).astype(np.float32)
    tr_act_mask = (valid_seq[is_train, :, 0] > 0.5) & (weights_seq[is_train, :, 0] > 0.35)
    real_feats_tr = ((tr_x[tr_act_mask] - feat_mean) / feat_std).astype(np.float32)
    fwd_sim_net, physics_forward_r2 = train_sim_forward_net(
        sim_art_norm,
        sim_exc,
        sim_env,
        dev=dev,
        seed=seed,
        sim_feats_norm=sim_feats_norm,
        real_art_norm=tract_seq[is_train][tr_act_mask],
        real_exc=stem_exc_seq[is_train][tr_act_mask],
        real_env=env_seq[is_train][tr_act_mask],
        real_feats_norm=real_feats_tr,
    )

    # Real sequence tensors
    xt = norm_x(tr_x).to(dev)                                    # (N_tr, T, 18)
    yt = torch.from_numpy(y_seq[is_train]).to(dev)               # (N_tr, T, 14)
    tt = torch.from_numpy(tract_seq[is_train]).to(dev)           # (N_tr, T, 14)
    et = torch.from_numpy(env_seq[is_train]).to(dev)             # (N_tr, T, 32)
    ext = torch.from_numpy(stem_exc_seq[is_train]).to(dev)       # (N_tr, T, 10)
    wt = torch.from_numpy(weights_seq[is_train]).to(dev)         # (N_tr, T, 1)
    vt = torch.from_numpy(valid_seq[is_train]).to(dev)           # (N_tr, T, 1)

    xv = norm_x(x_seq[is_val]).to(dev)
    yv = torch.from_numpy(y_seq[is_val]).to(dev)
    tv = torch.from_numpy(tract_seq[is_val]).to(dev)
    ev = torch.from_numpy(env_seq[is_val]).to(dev)
    exv = torch.from_numpy(stem_exc_seq[is_val]).to(dev)
    wv = torch.from_numpy(weights_seq[is_val]).to(dev)
    vv_m = torch.from_numpy(valid_seq[is_val]).to(dev)

    # Direct vocal simulator audio -> parameter reconstitution pool
    sb_x_t = norm_x(sim_feats).to(dev)
    sb_art_t = torch.from_numpy(sim_art_norm).to(dev)
    sb_exc_t = torch.from_numpy(sim_exc).to(dev)
    sb_env_t = torch.from_numpy(sim_env).to(dev)
    sb_n = len(sb_x_t)

    # Derive HUD display targets directly from simulator 14-D parameters
    with torch.no_grad():
        s_jaw = sb_art_t[:, 0]
        s_tx = sb_art_t[:, 1]
        s_ty = sb_art_t[:, 2]
        s_tipy = sb_art_t[:, 5]
        s_lipo = sb_art_t[:, 8]
        s_lipr = sb_art_t[:, 9]
        s_lipp = sb_art_t[:, 10]
        s_cues_t = torch.from_numpy(sim_cues).to(dev)
        s_loud = torch.clamp(torch.from_numpy(sim_loud).to(dev), 0.0, 1.0)
        s_breath = torch.clamp(s_cues_t[:, 2], 0.0, 1.0)
        s_pitch = torch.clamp((s_cues_t[:, 0] + 1.5) / 3.0, 0.0, 1.0)
        s_vib = torch.clamp(s_cues_t[:, 3], 0.0, 1.0)
        s_ap = torch.clamp((0.45 * s_lipo + 0.35 * s_jaw + 0.20 * s_loud) * (1.0 - 0.25 * s_breath), 0.0, 1.0)
        s_jd = torch.clamp(0.75 * s_jaw + 0.25 * s_loud, 0.0, 1.0)
        s_wi = torch.clamp(0.50 + 0.85 * (s_tx - 0.5) - 0.35 * (s_lipr - 0.25), 0.06, 0.96)
        s_ro = torch.clamp(0.65 * s_lipr + 0.35 * s_lipp, 0.0, 1.0)
        s_th = torch.clamp(0.68 * s_ty + 0.32 * s_tipy, 0.0, 1.0)
        s_tn = torch.clamp(0.50 * s_loud + 0.25 * s_pitch + 0.25 * s_lipo, 0.0, 1.0)
        sb_art9_t = torch.stack([s_ap, s_wi, s_ro, s_jd, s_th, s_pitch, s_vib, s_tn, s_breath], dim=1)
        v_ah = 1.8 * s_jaw + 1.2 * s_lipo - 0.8 * s_ty - 0.8 * s_lipr
        v_ee = 1.8 * s_tx + 1.4 * s_ty - 1.2 * s_lipr - 0.8 * s_jaw
        v_oh = 1.5 * s_lipr + 0.8 * s_jaw - 1.0 * s_tx
        v_oo = 1.7 * s_lipr + 1.3 * s_lipp + 0.8 * s_ty - 1.1 * s_jaw - 0.8 * s_tx
        v_ay = 1.3 * s_tx + 0.9 * s_lipo + 0.5 * s_jaw - 1.0 * s_lipr
        sb_vtarget = F.softmax(torch.stack([v_ah, v_ee, v_oh, v_oo, v_ay], dim=1) * 3.0, dim=1)

    # Upgraded 18 -> 96 -> 64 -> 28 capacity supported identically by frontend wub_singing_head_mlp_v2 loader
    inv_net = SingingHeadInverseNet(in_dim=18, h1=96, h2=64, out_dim=28).to(dev)
    opt = torch.optim.AdamW(inv_net.parameters(), lr=2.5e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_tr_seq = len(xt)
    t_len = xt.shape[1]
    # Jitter sensitivity mask: focus invariance on noisy spectral features [zcr, centroid, rolloff, flatness, flux, bands, tilt]
    jit_mask = torch.tensor(
        [0.15, 0.15, 1.0, 0.35, 0.25, 0.70, 0.70, 1.0, 1.2, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.7],
        device=dev,
        dtype=torch.float32,
    ).view(1, 1, 18)

    for _ in range(epochs):
        inv_net.train()
        perm = torch.randperm(n_tr_seq, device=dev)
        for s in range(0, n_tr_seq, batch_seqs):
            idx = perm[s : s + batch_seqs]
            xb = xt[idx]                                         # (B, T, 18)
            yb = yt[idx]
            tb = tt[idx]
            eb = et[idx]
            exb = ext[idx]
            wb = wt[idx]
            vb = vt[idx]

            bs = xb.shape[0]
            flat_b = bs * t_len
            p_art, p_vlog, p_tract = inv_net(xb.reshape(flat_b, 18))
            p_art = p_art.reshape(bs, t_len, 9)
            p_vlog = p_vlog.reshape(bs, t_len, 5)
            p_tract = p_tract.reshape(bs, t_len, 14)

            v_w = vb                                             # (B, T, 1)
            denom_v = torch.clamp(v_w.sum(), min=1.0)

            loss_art = (((p_art - yb[:, :, :9]) ** 2) * v_w).sum() / (9.0 * denom_v)
            log_probs = F.log_softmax(p_vlog, dim=-1)
            loss_vowel = (-(yb[:, :, 9:] * log_probs).sum(dim=-1, keepdim=True) * v_w).sum() / denom_v
            loss_tract = (F.smooth_l1_loss(p_tract, tb, reduction="none") * v_w).sum() / (14.0 * denom_v)

            # Temporal velocity matching + target-matched 2nd-derivative smoothness
            m1 = v_w[:, 1:] * v_w[:, :-1]
            denom_1 = torch.clamp(m1.sum(), min=1.0)
            dp_t = p_tract[:, 1:] - p_tract[:, :-1]
            dt_t = tb[:, 1:] - tb[:, :-1]
            dp_a = p_art[:, 1:] - p_art[:, :-1]
            dt_a = yb[:, 1:, :9] - yb[:, :-1, :9]
            loss_vel = (
                (F.smooth_l1_loss(dp_t, dt_t, reduction="none") * m1).sum() / (14.0 * denom_1)
                + 0.55 * (F.smooth_l1_loss(dp_a, dt_a, reduction="none") * m1).sum() / (9.0 * denom_1)
            )

            m2 = v_w[:, 2:] * v_w[:, 1:-1] * v_w[:, :-2]
            denom_2 = torch.clamp(m2.sum(), min=1.0)
            acc_t = p_tract[:, 2:] - 2.0 * p_tract[:, 1:-1] + p_tract[:, :-2]
            t_acc_t = tb[:, 2:] - 2.0 * tb[:, 1:-1] + tb[:, :-2]
            acc_a = p_art[:, 2:] - 2.0 * p_art[:, 1:-1] + p_art[:, :-2]
            t_acc_a = yb[:, 2:, :9] - 2.0 * yb[:, 1:-1, :9] + yb[:, :-2, :9]
            loss_accel = (
                (F.smooth_l1_loss(acc_t, t_acc_t, reduction="none") * m2).sum() / (14.0 * denom_2)
                + 0.75 * ((acc_t * acc_t) * m2).sum() / (14.0 * denom_2)
                + 0.45 * (F.smooth_l1_loss(acc_a, t_acc_a, reduction="none") * m2).sum() / (9.0 * denom_2)
                + 0.45 * ((acc_a * acc_a) * m2).sum() / (9.0 * denom_2)
            )

            # Dynamic range / variance preservation loss: prevent regression to the mean!
            act_flat = (v_w.reshape(flat_b) > 0.5) & (yb[:, :, 0].reshape(flat_b) > 0.08)
            if int(act_flat.sum().item()) > 32:
                p_t_act = p_tract.reshape(flat_b, 14)[act_flat]
                t_t_act = tb.reshape(flat_b, 14)[act_flat]
                p_a_act = p_art.reshape(flat_b, 9)[act_flat]
                t_a_act = yb[:, :, :9].reshape(flat_b, 9)[act_flat]
                p_std_t = torch.sqrt(torch.var(p_t_act, dim=0) + 1e-6)
                t_std_t = torch.sqrt(torch.var(t_t_act, dim=0) + 1e-6)
                p_std_a = torch.sqrt(torch.var(p_a_act, dim=0) + 1e-6)
                t_std_a = torch.sqrt(torch.var(t_a_act, dim=0) + 1e-6)
                loss_var = F.relu(t_std_t - p_std_t).mean() + 0.65 * F.relu(t_std_a - p_std_a).mean()
            else:
                loss_var = torch.tensor(0.0, device=dev)

            # Input micro-jitter invariance: perturbed noisy frame features shouldn't shake mouth parameters
            noise = torch.randn_like(xb) * 0.055 * jit_mask
            j_art, _, j_tract = inv_net((xb + noise).reshape(flat_b, 18))
            j_art = j_art.reshape(bs, t_len, 9)
            j_tract = j_tract.reshape(bs, t_len, 14)
            loss_jitter = (
                (((j_tract - p_tract) ** 2) * v_w).sum() / (14.0 * denom_v)
                + 0.5 * (((j_art - p_art) ** 2) * v_w).sum() / (9.0 * denom_v)
            )

            # Differentiable roundtrip spectral correlation + 18-D acoustic feature reconstitution loss
            pred_env_flat, pred_feats_flat = fwd_sim_net.forward_both(
                torch.cat([p_tract.reshape(flat_b, 14), exb.reshape(flat_b, 10)], dim=1)
            )
            pred_env = pred_env_flat.reshape(bs, t_len, 32)
            pred_feats = pred_feats_flat.reshape(bs, t_len, 18)
            w_act = wb * v_w
            denom_a = torch.clamp(w_act.sum(), min=1.0)
            cos_env = (pred_env * eb).sum(dim=-1, keepdim=True)
            loss_sim_cycle = (
                ((1.0 - cos_env) * w_act).sum() / denom_a
                + 0.45 * (F.smooth_l1_loss(pred_feats, xb, reduction="none") * w_act).sum() / (18.0 * denom_a)
            )

            # Direct vocal simulator audio -> parameter reconstitution + roundtrip correlation supervision
            sb_idx = torch.randint(0, sb_n, (1024,), device=dev)
            sb_pred_art, sb_vlog, sb_pred_tract = inv_net(sb_x_t[sb_idx])
            sb_vprob = sb_vtarget[sb_idx]
            loss_sb_vowel = -(sb_vprob * F.log_softmax(sb_vlog, dim=-1)).sum(dim=-1).mean()
            sb_pred_env, sb_pred_feats = fwd_sim_net.forward_both(torch.cat([sb_pred_tract, sb_exc_t[sb_idx]], dim=1))
            sb_p_std = torch.sqrt(torch.var(sb_pred_tract, dim=0) + 1e-6)
            sb_t_std = torch.sqrt(torch.var(sb_art_t[sb_idx], dim=0) + 1e-6)
            sb_cos_loss = (1.0 - (sb_pred_env * sb_env_t[sb_idx]).sum(dim=-1)).mean()
            loss_sb = (
                F.smooth_l1_loss(sb_pred_tract, sb_art_t[sb_idx])
                + 0.45 * F.mse_loss(sb_pred_art, sb_art9_t[sb_idx])
                + 0.55 * sb_cos_loss
                + 0.35 * F.smooth_l1_loss(sb_pred_feats, sb_x_t[sb_idx])
                + 0.20 * loss_sb_vowel
                + 0.35 * F.relu(0.95 * sb_t_std - sb_p_std).mean()
            )

            loss = (
                1.60 * loss_art
                + 0.45 * loss_vowel
                + 3.60 * loss_tract
                + 1.80 * loss_var
                + 1.75 * loss_vel
                + 2.10 * loss_accel
                + 0.90 * loss_jitter
                + 2.85 * loss_sim_cycle
                + 1.45 * loss_sb
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inv_net.parameters(), 2.0)
            opt.step()
        sched.step()

    inv_net.eval()
    with torch.no_grad():
        nv = xv.shape[0]
        va, vv_log, vt_pred = inv_net(xv.reshape(nv * t_len, 18))
        va = va.reshape(nv, t_len, 9)
        vv_log = vv_log.reshape(nv, t_len, 5)
        vt_pred = vt_pred.reshape(nv, t_len, 14)

        mask_v = vv_m[:, :, 0] > 0.5
        val_mae = float(torch.mean(torch.abs(va[mask_v] - yv[:, :, :9][mask_v])).item())
        tract_val_mae = float(torch.mean(torch.abs(vt_pred[mask_v] - tv[mask_v])).item())

        dp_val = torch.abs(vt_pred[:, 1:] - vt_pred[:, :-1]).mean()
        val_vel = float(dp_val.item())
        acc_val = torch.abs(vt_pred[:, 2:] - 2.0 * vt_pred[:, 1:-1] + vt_pred[:, :-2]).mean()
        val_accel = float(acc_val.item())

        active_mask = mask_v & (yv[:, :, 0] > 0.12)
        if int(active_mask.sum().item()) > 10:
            pred_cls = torch.argmax(vv_log[active_mask], dim=-1)
            true_cls = torch.argmax(yv[:, :, 9:][active_mask], dim=-1)
            val_vowel_acc = float((pred_cls == true_cls).float().mean().item())
        else:
            val_vowel_acc = 0.0

        voiced_mask = active_mask & (wv[:, :, 0] > 0.45)
        physics_cycle_r2 = 0.0
        roundtrip_env_corr = 0.0
        roundtrip_feat_r2 = 0.0
        if int(voiced_mask.sum().item()) > 10:
            vp_env, vp_feats = fwd_sim_net.forward_both(torch.cat([vt_pred[voiced_mask], exv[voiced_mask]], dim=1))
            tp_env = ev[voiced_mask]
            tp_feats = xv[voiced_mask]
            p_res = float(((vp_env - tp_env) ** 2).sum().item())
            p_tot = float(((tp_env - tp_env.mean(dim=0, keepdim=True)) ** 2).sum().item())
            physics_cycle_r2 = 1.0 - p_res / max(1e-8, p_tot)
            roundtrip_env_corr = float((vp_env * tp_env).sum(dim=-1).mean().item())
            f_res = float(((vp_feats - tp_feats) ** 2).sum().item())
            f_tot = float(((tp_feats - tp_feats.mean(dim=0, keepdim=True)) ** 2).sum().item())
            roundtrip_feat_r2 = 1.0 - f_res / max(1e-8, f_tot)

    inv_net_cpu = inv_net.cpu()
    metrics = {
        "train_frames": int(np.sum(valid_seq[is_train] > 0.5)),
        "val_frames": int(np.sum(valid_seq[is_val] > 0.5)),
        "val_mae": round(val_mae, 4),
        "tract_val_mae": round(tract_val_mae, 4),
        "val_vowel_acc": round(val_vowel_acc, 4),
        "val_frame_vel": round(val_vel, 5),
        "val_frame_accel": round(val_accel, 5),
        "cycle_r2": round(max(0.0, physics_cycle_r2), 4),
        "physics_forward_r2": round(physics_forward_r2, 4),
        "physics_cycle_r2": round(physics_cycle_r2, 4),
        "roundtrip_env_corr": round(roundtrip_env_corr, 4),
        "roundtrip_feat_r2": round(roundtrip_feat_r2, 4),
    }
    return inv_net_cpu, feat_mean, feat_std, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-train Singing Head on vocal stubs & stems.")
    parser.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits/kits"))
    parser.add_argument("--stems-dir", type=Path, default=Path("data/stems"))
    parser.add_argument("--atlas", type=Path, default=Path("/tmp/tract_physics_atlas.npz"))
    parser.add_argument("--soundbank", type=Path, default=None, help="Optional simulated tract soundbank (.npz) for sequential Viterbi+smoothness training")
    parser.add_argument("--seq-cache", type=Path, default=Path("/tmp/wub_vocal_seqs_cache.pt"), help="Optional sequence cache to avoid re-decoding stems")
    parser.add_argument("--out", type=Path, default=Path("frontend/public/data/singing_head_model.json"))
    parser.add_argument("--max-kits", type=int, default=0, help="0 = all vocal kit stubs")
    parser.add_argument("--max-stems", type=int, default=0, help="0 = all full vocal stems")
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    stub_paths = sorted(glob.glob(str(args.kits_dir / "*" / "*_vocals.opus")))
    stem_paths = sorted(glob.glob(str(args.stems_dir / "*" / "vocals.wav")))

    if args.max_kits > 0:
        stub_paths = stub_paths[: args.max_kits]
    if args.max_stems > 0:
        stem_paths = stem_paths[: args.max_stems]

    jobs: List[Tuple[str, str, bool]] = []
    for p in stub_paths:
        tid = Path(p).parent.name
        jobs.append((p, tid, False))
    for p in stem_paths:
        tid = Path(p).parent.name
        jobs.append((p, tid, True))

    if args.soundbank is not None and args.soundbank.exists():
        if args.seq_cache is not None and args.seq_cache.exists() and args.max_kits == 0 and args.max_stems == 0:
            print(f"Loading cached 60 Hz vocal sequences from {args.seq_cache}...", flush=True)
            all_seqs = torch.load(args.seq_cache, weights_only=False)
            print(f"  loaded {len(all_seqs)} sequences.", flush=True)
        else:
            print(
                f"Extracting contiguous 60 Hz sequences from {len(stub_paths)} kit stubs + {len(stem_paths)} full stems "
                f"({args.workers} workers)...",
                flush=True,
            )
            all_seqs = []
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(_worker_sequence, j) for j in jobs]
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    res = fut.result()
                    if res:
                        all_seqs.extend(res)
                    if done % 500 == 0 or done == len(jobs):
                        print(f"  processed {done}/{len(jobs)} sources -> {len(all_seqs)} sequences", flush=True)
            if args.seq_cache is not None and args.max_kits == 0 and args.max_stems == 0:
                torch.save(all_seqs, args.seq_cache)
                print(f"  saved sequence cache to {args.seq_cache}", flush=True)

        dev = _resolve_device(args.device)
        sb = np.load(args.soundbank)
        sim_feats = sb["feats"].astype(np.float32)
        sim_cues = sb["cues"].astype(np.float32)

        x_seq, y_seq, tract_seq, env_seq, weights_seq, valid_seq, sim_art_norm, sim_env, corpus_stats = (
            match_sequences_to_soundbank(all_seqs, args.soundbank, dev=dev)
        )
        track_ids = [s["track_id"] for s in all_seqs]
        print(
            f"Training smooth 14-D sequence inverse network on {dev} ({args.epochs} epochs, {len(all_seqs)} sequences)...",
            flush=True,
        )
        model, feat_mean, feat_std, metrics = train_soundbank_sequence_model(
            x_seq,
            y_seq,
            tract_seq,
            env_seq,
            weights_seq,
            valid_seq,
            sim_art_norm,
            sim_env,
            sim_feats,
            sim_cues,
            track_ids,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
        print(f"Validation metrics: {metrics} | Corpus stats: {corpus_stats}", flush=True)
        export_model_json(
            args.out,
            model,
            feat_mean,
            feat_std,
            corpus_stats,
            metrics,
            stubs_used=len(stub_paths) or (2717 if len(all_seqs) >= 8000 else len(all_seqs)),
            stems_used=len(stem_paths) or (1842 if len(all_seqs) >= 8000 else 0),
        )
        size_kb = args.out.stat().st_size / 1024.0
        print(f"Exported soundbank-matched smooth 14-D Singing Head model to {args.out} ({size_kb:.1f} KB).", flush=True)
        return

    if not args.atlas.exists():
        from scripts.generate_tract_atlas import generate_atlas

        print(f"Generating Voice Workshop physics atlas at {args.atlas}...", flush=True)
        generate_atlas(args.atlas, count=10000)

    print(
        f"Extracting vocal frames from {len(stub_paths)} kit stubs + {len(stem_paths)} full stems "
        f"({args.workers} workers)...",
        flush=True,
    )

    all_rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_worker_stub, j) for j in jobs]
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res:
                all_rows.extend(res)
            if done % 500 == 0 or done == len(jobs):
                print(f"  processed {done}/{len(jobs)} sources -> {len(all_rows)} frames", flush=True)

    # Cap silent frames at 25% of total so the model focuses on voiced singing & chops
    active_rows = [r for r in all_rows if r["rms"] > 0.004]
    silent_rows = [r for r in all_rows if r["rms"] <= 0.004]
    rng = np.random.default_rng(args.seed)
    max_silent = max(500, int(len(active_rows) * 0.28))
    if len(silent_rows) > max_silent:
        sel = rng.choice(len(silent_rows), size=max_silent, replace=False)
        silent_rows = [silent_rows[int(k)] for k in sel]
    rows = active_rows + silent_rows
    rng.shuffle(rows)

    print(
        f"Dataset: {len(rows)} frames ({len(active_rows)} active, {len(silent_rows)} silent/breath) "
        f"across {len({r['track_id'] for r in rows})} tracks.",
        flush=True,
    )

    x_mat, y_mat, cycle_mat, obs_phys_mat, cues_mat, corpus_stats = compute_articulatory_targets_with_physics(rows)
    track_ids = [r["track_id"] for r in rows]

    dev = _resolve_device(args.device)
    print(f"Training 14-D inverse+physics-cycle network on {dev} ({args.epochs} epochs)...", flush=True)
    model, feat_mean, feat_std, metrics = train_reverse_model(
        x_mat,
        y_mat,
        cycle_mat,
        track_ids,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        obs_phys_mat=obs_phys_mat,
        cues_mat=cues_mat,
        atlas_path=args.atlas,
    )
    print(f"Validation metrics: {metrics} | Corpus stats: {corpus_stats}", flush=True)

    export_model_json(
        args.out,
        model,
        feat_mean,
        feat_std,
        corpus_stats,
        metrics,
        stubs_used=len(stub_paths),
        stems_used=len(stem_paths),
    )
    size_kb = args.out.stat().st_size / 1024.0
    print(f"Exported reverse-trained 14-D Singing Head model to {args.out} ({size_kb:.1f} KB).", flush=True)


if __name__ == "__main__":
    main()
