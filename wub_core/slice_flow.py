"""Acoustic descriptors and graded supervision for assembled slice transitions.

No source identity is an input to the critic. Original chronological boundaries
are evidence, not a rule that cross-song combinations must be bad. All audio
operations here are analysis-only and never modify shipping assets.
"""
from __future__ import annotations

import math
import numpy as np

STEMS = ("drums", "bass", "other", "vocals")
FEATURES = ("level", "head_level", "tail_level", "crest", "centroid", "head_centroid",
            "tail_centroid", "sub", "low_mid", "mid", "high", "density", "offbeat",
            "downbeat", "timing_offset", "timing_spread", "width", "decay", "attack",
            "sustain") + tuple(f"chroma_{i}" for i in range(12))
DIM = len(FEATURES)
CONTEXT = ("section_boundary", "source_change", "phrase_boundary", "downbeat", "age",
           "repetition", "target_energy", "changed_fraction")
INPUT_DIM = DIM * 3 + len(STEMS) + len(CONTEXT)


def level(x):
    return float(np.clip((20 * np.log10(max(float(x), 1e-8)) + 60) / 60, 0, 1))


def describe_audio(audio: np.ndarray, sr: int, stem: str) -> np.ndarray:
    """32 finite, bounded descriptors of an actual (possibly rendered) cell."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if not len(a):
        return np.zeros(DIM, np.float32)
    mono = a.mean(axis=1)
    rms = float(np.sqrt(np.mean(a * a)))
    if rms < 1e-7:
        return np.zeros(DIM, np.float32)
    edge = max(1, len(a) // 8)
    energy = np.array([np.sqrt(np.mean(x * x)) if len(x) else 0 for x in np.array_split(mono, 128)])
    fft_size = min(4096, max(128, 2 ** int(math.floor(math.log2(max(128, len(mono)))))))
    def spectrum(x):
        if len(x) < fft_size:
            x = np.pad(x, (0, fft_size - len(x)))
        stride = max(fft_size, len(x) // 8)
        starts = list(range(0, max(1, len(x) - fft_size + 1), stride))[:8]
        frames = np.stack([x[i:i + fft_size] * np.hanning(fft_size) for i in starts])
        return np.mean(np.abs(np.fft.rfft(frames)) ** 2, axis=0)
    freq = np.fft.rfftfreq(fft_size, 1 / sr)
    spec = spectrum(mono)
    def centroid(s):
        return float(np.clip(np.dot(s, freq) / max(s.sum(), 1e-12) / 8000, 0, 1))
    out = np.zeros(DIM, np.float32)
    head = float(np.sqrt(np.mean(mono[:edge] ** 2)))
    tail = float(np.sqrt(np.mean(mono[-edge:] ** 2)))
    out[:7] = [level(rms), level(head), level(tail), np.clip(np.max(np.abs(mono)) / max(rms, 1e-8) / 12, 0, 1), centroid(spec), centroid(spectrum(mono[:edge])), centroid(spectrum(mono[-edge:]))]
    total = max(spec.sum(), 1e-12)
    for i, (lo, hi) in enumerate(((20, 150), (150, 500), (500, 2000), (2000, 12000))):
        out[7 + i] = spec[(freq >= lo) & (freq < hi)].sum() / total
    flux = np.maximum(0, np.diff(energy, prepend=energy[0]))
    peaks = np.flatnonzero((flux > max(1e-8, flux.mean() + 0.7 * flux.std())) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1)))
    positions = peaks / 8  # 128 envelope bins / 16 sixteenth positions
    offsets = positions - np.round(positions)
    out[11] = min(1, len(peaks) / 16)
    if len(peaks):
        steps = np.round(positions).astype(int) % 16
        out[12] = np.mean(np.isin(steps, [2, 6, 10, 14]))
        out[13] = np.mean(np.isin(steps, [0, 4, 8, 12]))
        out[14] = float(np.mean(offsets))
        out[15] = float(np.std(offsets))
    if a.shape[1] > 1:
        out[16] = min(1, np.sqrt(np.mean(((a[:, 0] - a[:, 1]) / 2) ** 2)) / max(rms, 1e-8))
    out[17:20] = [np.clip((head - tail) / max(rms, 1e-8), -1, 1), np.clip(flux.max() / max(energy.max(), 1e-8), 0, 1), np.mean(energy > rms * 0.5)]
    if stem != "drums":
        mask = (freq >= 45) & (freq <= 4000)
        pcs = np.round(69 + 12 * np.log2(np.maximum(freq[mask], 1) / 440)).astype(int) % 12
        chroma = np.bincount(pcs, weights=spec[mask], minlength=12)
        out[20:] = chroma / max(chroma.sum(), 1e-12)
    return np.nan_to_num(out)


def transform(features, semitones=0, gain=1):
    out = np.array(features, dtype=np.float32, copy=True)
    if gain <= 1e-5:
        return np.zeros(DIM, np.float32)
    out[:3] = np.clip(out[:3] + 20 * np.log10(gain) / 60, 0, 1)
    out[20:] = np.roll(out[20:], int(round(semitones)))
    return out


def transition_features(prev, nxt, stem, context=None):
    ctx = np.asarray(context if context is not None else [0, 1, 1, 1, .5, 0, .7, .25], np.float32)
    identity = np.zeros(4, np.float32)
    identity[STEMS.index(stem)] = 1
    return np.concatenate([prev, nxt, np.abs(np.asarray(prev) - nxt), identity, ctx]).astype(np.float32)


def acoustic_teacher(prev, nxt, stem, prepared=False):
    """Weak acoustic target, explicitly separate from human preference labels."""
    a, b = np.asarray(prev), np.asarray(nxt)
    level_jump = abs(float(a[2] - b[1]))
    texture = float(np.abs(a[7:11] - b[7:11]).mean())
    pocket = abs(float(a[11] - b[11])) + .5 * abs(float(a[14] - b[14]))
    harmonic = 0.
    if stem != 'drums' and a[20:].sum() > .1 and b[20:].sum() > .1:
        harmonic = 1 - float(np.dot(a[20:], b[20:]) / max(np.linalg.norm(a[20:]) * np.linalg.norm(b[20:]), 1e-8))
    dropout = .35 if stem in ('drums', 'bass') and a[0] > .35 and b[0] < .1 else 0
    penalty = 1.1 * level_jump + .8 * texture + .3 * pocket + .25 * harmonic + dropout
    return float(np.clip(.95 - penalty * (.7 if prepared else 1), .03, .97))


def mix_features(cells):
    a = np.asarray(cells)
    return a[:, [0, 7, 9, 11]].reshape(-1).astype(np.float32)


def mix_teacher(cells):
    """Weak masking/density target, not a claim of perceptual ground truth."""
    a = np.asarray(cells)
    power = 10 ** ((a[:, 0] * 60 - 60) / 10)
    if power.sum() < 1e-10:
        return .03
    vocal_mid = power[3] * a[3, 9]
    other_mid = power[2] * a[2, 9]
    buried = max(0., 1 - vocal_mid / max(other_mid * .15, 1e-10)) if a[3,0] > .25 else 0
    low = min(a[0,7], a[1,7]) * min(a[0,0], a[1,0])
    busy = max(0., float(a[:,11].mean()) - .55)
    return float(np.clip(.95 - .35*buried - .35*low - .5*busy, .03, .97))
