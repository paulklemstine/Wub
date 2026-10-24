"""Click-free audio stitching utilities (equal-power micro-crossfades)."""
from __future__ import annotations

import numpy as np


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Crossfade tail of `a` with head of `b` using equal-power curves.

    Returns concatenated array with overlap of fade_samples.
    Both inputs are 1-D (mono) or 2-D (samples x channels).
    """
    if fade_samples <= 0:
        return np.concatenate([a, b], axis=0)
    fade_samples = min(fade_samples, len(a), len(b))
    t = np.linspace(0, np.pi / 2, fade_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    if a.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in = fade_in[:, None]
    head = a[:-fade_samples] if len(a) > fade_samples else np.zeros_like(b[:0])
    tail = b[fade_samples:] if len(b) > fade_samples else np.zeros_like(a[:0])
    mid = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    return np.concatenate([head, mid, tail], axis=0)


def stitch_with_crossfade(chunks: list[np.ndarray], sr: int, fade_ms: float = 3.0) -> np.ndarray:
    fade = int(sr * fade_ms / 1000.0)
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    out = chunks[0]
    for c in chunks[1:]:
        out = equal_power_crossfade(out, c, fade)
    return out


def boundary_click_metric(y: np.ndarray, sr: int, boundaries: list[int], window_ms: float = 5.0) -> float:
    """Max abs first-difference near boundaries, normalized by signal RMS.

    Low value => no clicks. Used by verify_slicing.py.
    """
    if len(y) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(y**2)) + 1e-9)
    w = int(sr * window_ms / 1000.0)
    worst = 0.0
    mono = y.mean(axis=1) if y.ndim == 2 else y
    diff = np.abs(np.diff(mono))
    for b in boundaries:
        lo, hi = max(0, b - w), min(len(diff), b + w)
        if hi > lo:
            worst = max(worst, float(np.max(diff[lo:hi])))
    return worst / rms
