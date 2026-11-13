"""Unit tests for mix_compat key-alignment, teacher manifold scoring, and training targets."""

import numpy as np

from scripts.build_mix_compat_dataset import (
    BARS_PER_PHRASE,
    FEATURE_DIM,
    STEMS,
    _spectral_embed,
    harmonic_features,
    heuristic_mix_score,
    key_shift,
    per_bar_feature_vector,
)
from scripts.train_mix_compat import build_training_arrays


def test_key_shift_relative_major_and_direction():
    assert key_shift("Am", "C") == 0
    assert key_shift("C", "Am") == 0
    assert key_shift("F#m", "A") == 0
    assert key_shift("D", "C") == 2
    assert key_shift("C", "D") == -2
    assert key_shift("Am", "Gm") == 2
    assert key_shift("Gm", "Am") == -2
    for s in ("C", "C#", "D", "F#", "Am", "Bbm"):
        for p in ("C", "Eb", "G", "B", "Dm", "G#m"):
            d = key_shift(s, p)
            assert -6 <= d <= 6


def test_harmonic_features_aligns_stems_to_drums_session_key():
    # Bass in Dm with root=2 (D) & chroma peak at index 2 (D),
    # Other in Am with root=9 (A) & chroma peak at index 9 (A),
    # Drums in Am -> bass shifts by +7 (wraps to -5): (2 - 5) % 12 == 9!
    c_bass = [0.0] * 12
    c_bass[2] = 1.0
    c_other = [0.0] * 12
    c_other[9] = 1.0
    c_voc = [0.0] * 12
    c_voc[9] = 1.0
    phrases = {
        "drums": {"chroma": [0.0] * 12},
        "bass": {"chroma": c_bass, "bass_root": 2},
        "other": {"chroma": c_other, "bass_root": 9},
        "vocals": {"chroma": c_voc},
    }
    kits = {
        "drums": {"key": "Am"},
        "bass": {"key": "Dm"},
        "other": {"key": "Am"},
        "vocals": {"key": "Am"},
    }
    harm = harmonic_features(phrases, kits)
    assert np.isclose(harm["bass_other_chroma"], 1.0, atol=1e-5)
    assert np.isclose(harm["bass_vocal_chroma"], 1.0, atol=1e-5)
    assert harm["root_interval_penalty"] == 0.0


def test_build_training_arrays_incorporates_teacher_label_with_per_bar_contrast():
    feat = np.zeros((BARS_PER_PHRASE, FEATURE_DIM), dtype=np.float32).tolist()
    examples = [
        {
            "kind": "real",
            "label": 0.96,
            "per_bar": [0.46, 0.56, 0.66, 0.56],  # mean 0.56 -> delta +0.40
            "rhythm_per_bar": [0.70, 0.80, 0.90, 0.80],
            "features": feat,
        },
        {
            "kind": "random",
            "label": 0.30,
            "per_bar": [0.40, 0.50, 0.60, 0.50],  # mean 0.50 -> delta -0.20
            "rhythm_per_bar": [0.30, 0.40, 0.50, 0.40],
            "features": feat,
        },
    ]
    X, Y, groups = build_training_arrays(examples, BARS_PER_PHRASE, FEATURE_DIM)
    assert X.shape == (8, FEATURE_DIM)
    assert Y.shape == (8, 2)
    assert groups.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    # Compat targets incorporate phrase label (anchored mean, subject only to [0, 1] clamp) while preserving bar ordering.
    real_bars = Y[:4, 0]
    rand_bars = Y[4:, 0]
    assert np.allclose(real_bars, [0.86, 0.96, 1.0, 0.96], atol=1e-4)
    assert np.isclose(float(np.mean(rand_bars)), 0.30, atol=1e-4)
    assert real_bars[0] < real_bars[1] < real_bars[2]
    assert rand_bars[0] < rand_bars[1] < rand_bars[2]
    assert np.allclose(Y[:4, 1], [0.70, 0.80, 0.90, 0.80], atol=1e-5)


def test_spectral_embed_distinguishes_tonal_center_and_per_bar_features_vary():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    # Pure A440 harmonic tone vs dissonant tritone cluster
    tonal = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t)
    clash = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 622.25 * t)
    e_tonal = _spectral_embed(tonal, sr=sr, dim=128)
    e_clash = _spectral_embed(clash, sr=sr, dim=128)
    assert e_tonal.shape == (128,)
    assert e_clash.shape == (128,)
    assert float(np.dot(e_tonal, e_clash)) < 0.999

    # Different onset patterns across bar 0 vs bar 1 produce distinct feature vectors.
    g_drums = np.zeros(64, dtype=np.float32)
    g_bass = np.zeros(64, dtype=np.float32)
    g_drums[[0, 4, 8, 12]] = 1.0  # bar 0 locked
    g_bass[[0, 4, 8, 12]] = 1.0
    g_drums[[16, 20, 24, 28]] = 1.0  # bar 1 clashing
    g_bass[[18, 22, 26, 30]] = 1.0
    grids = {s: (g_drums if s == "drums" else g_bass) for s in STEMS}
    phrases = {s: {"chroma": [1.0] + [0.0] * 11, "bass_root": 0, "energy": 0.6, "stems": {s: {"rms_db": -12.0}}} for s in STEMS}
    kits = {s: {"key": "C"} for s in STEMS}
    per_bar = np.stack([per_bar_feature_vector(b, phrases, kits, grids) for b in range(BARS_PER_PHRASE)])
    grade = heuristic_mix_score(per_bar)
    assert grade["rhythm_per_bar"][0] > grade["rhythm_per_bar"][1]
    assert grade["per_bar"][0] > grade["per_bar"][1]
