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
                rows.extend(_process_audio_segment(track_id, chunk, is_stem=True, max_frames=10))
            return rows
    except Exception:
        return []


def compute_articulatory_targets(
    rows: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Map corpus-calibrated acoustic rows to (X_18, Y_14, cycle_4, corpus_stats)."""
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

        # Unvoiced sibilance / breathiness (high 5-9 kHz + low voicing + active RMS)
        sib_band = float(feats[16])  # band_7_5000_9000
        presence_band = float(feats[15])  # band_6_3000_5000
        flatness = float(feats[7])
        active_gate = clamp((rms - 0.003) / 0.025, 0.0, 1.0)
        breath = clamp(active_gate * (1.0 - voicing * 0.85) * (0.45 * clamp(sib_band / 2.2, 0.0, 1.0) + 0.55 * flatness), 0.0, 1.0)

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

    stats = {
        "db_p10": round(db_p10, 3),
        "db_p95": round(db_p95, 3),
        "f0_p10": round(f0_p10, 2),
        "f0_p95": round(f0_p95, 2),
    }
    return x_mat, y_mat, cycle_mat, stats


class SingingHeadInverseNet(nn.Module):
    """Compact acoustic-to-articulatory inverse encoder (exports to browser JSON)."""

    def __init__(self, in_dim: int = 18, h1: int = 48, h2: int = 32, out_dim: int = 14) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        raw = self.fc3(h)
        articulators = torch.sigmoid(raw[:, :9])
        vowel_logits = raw[:, 9:]
        return articulators, vowel_logits


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
) -> Tuple[SingingHeadInverseNet, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train inverse encoder + forward formant cycle decoder with track split."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = _resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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

    xv = norm_x(x_mat[is_val]).to(dev)
    yv = torch.from_numpy(y_mat[is_val]).to(dev)
    cv = torch.from_numpy(cycle_mat[is_val]).to(dev)

    inv_net = SingingHeadInverseNet(in_dim=x_mat.shape[1], h1=48, h2=32, out_dim=14).to(dev)
    fwd_net = ForwardFormantDecoder(in_dim=14, out_dim=4).to(dev)

    opt = torch.optim.AdamW(
        list(inv_net.parameters()) + list(fwd_net.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    n_train = len(xt)
    for ep in range(epochs):
        inv_net.train()
        fwd_net.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = xt[idx]
            yb = yt[idx]
            cb = ct[idx]

            pred_art, pred_vlogits = inv_net(xb)
            pred_cycle = fwd_net(pred_art, pred_vlogits)

            loss_art = F.mse_loss(pred_art, yb[:, :9])
            target_vprobs = yb[:, 9:]
            log_probs = F.log_softmax(pred_vlogits, dim=-1)
            loss_vowel = -(target_vprobs * log_probs).sum(dim=-1).mean()
            loss_cycle = F.mse_loss(pred_cycle, cb)

            loss = loss_art + 0.35 * loss_vowel + 0.25 * loss_cycle
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    inv_net.eval()
    fwd_net.eval()
    with torch.no_grad():
        va, vv = inv_net(xv)
        vc = fwd_net(va, vv)
        val_mae = float(torch.mean(torch.abs(va - yv[:, :9])).item())
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

    inv_net_cpu = inv_net.cpu()
    metrics = {
        "train_frames": int(np.sum(is_train)),
        "val_frames": int(np.sum(is_val)),
        "val_mae": round(val_mae, 4),
        "val_vowel_acc": round(val_vowel_acc, 4),
        "cycle_r2": round(cycle_r2, 4),
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
    """Export trained reverse-model weights and normalization stats as compact JSON."""
    state = model.state_dict()

    def round_list(t: torch.Tensor) -> list:
        arr = np.round(t.detach().cpu().numpy().astype(np.float64), 6)
        return arr.tolist()

    payload = {
        "version": 1,
        "format": "wub_singing_head_mlp_v1",
        "sample_rate": TARGET_SR,
        "fft_size": FFT_SIZE,
        "hop_size": HOP_SIZE,
        "feature_names": FEATURE_NAMES,
        "articulators": ARTICULATOR_NAMES,
        "vowels": VOWELS,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-train Singing Head on vocal stubs & stems.")
    parser.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits/kits"))
    parser.add_argument("--stems-dir", type=Path, default=Path("data/stems"))
    parser.add_argument("--out", type=Path, default=Path("frontend/public/data/singing_head_model.json"))
    parser.add_argument("--max-kits", type=int, default=0, help="0 = all vocal kit stubs")
    parser.add_argument("--max-stems", type=int, default=0, help="0 = all full vocal stems")
    parser.add_argument("--epochs", type=int, default=35)
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

    x_mat, y_mat, cycle_mat, corpus_stats = compute_articulatory_targets(rows)
    track_ids = [r["track_id"] for r in rows]

    dev = _resolve_device(args.device)
    print(f"Training inverse+cycle network on {dev} ({args.epochs} epochs)...", flush=True)
    model, feat_mean, feat_std, metrics = train_reverse_model(
        x_mat, y_mat, cycle_mat, track_ids, epochs=args.epochs, seed=args.seed, device=args.device
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
    print(f"Exported reverse-trained Singing Head model to {args.out} ({size_kb:.1f} KB).", flush=True)


if __name__ == "__main__":
    main()
