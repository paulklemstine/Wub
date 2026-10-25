"""Unit tests for Acoustic Vocabulary Optimizer & Codebook Condensation."""

import numpy as np
import pytest

from wub_core.acoustic_vocabulary_optimizer import (
    AcousticVocabularyOptimizer,
    FEAT_DIM,
)


@pytest.fixture
def optimizer():
    return AcousticVocabularyOptimizer(random_state=42)


def test_feature_vector_extraction(optimizer):
    """Test 26-D acoustic feature extraction and normalization."""
    dummy_seg = {
        "timbre": [10.0, -20.0, 5.0, 0.0, -10.0, 15.0, -5.0, 2.0, -1.0, 0.5, -0.2, 0.1],
        "pitches": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "spectral_centroid": 2500.0,
        "loudness_max": -6.0,
    }

    vec = optimizer.extract_feature_vector(dummy_seg)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (FEAT_DIM,)
    assert vec.dtype == np.float32

    # Verify chroma component is unit normalized
    chroma_part = vec[12:24]
    assert np.isclose(np.linalg.norm(chroma_part), 1.0, atol=1e-4)

    # Verify log-centroid and normalized loudness bounds (0..1)
    assert 0.0 <= vec[24] <= 1.0
    assert 0.0 <= vec[25] <= 1.0


def test_compute_acoustic_clarity(optimizer):
    """Test slice clarity scoring and silence filtering."""
    silent_seg = {
        "is_silent": True,
        "loudness_max": -80.0,
        "duration": 0.2,
    }
    assert optimizer.compute_acoustic_clarity(silent_seg) == 0.0

    punchy_seg = {
        "is_silent": False,
        "loudness_start": -35.0,
        "loudness_max": -1.5,
        "duration": 0.30,
    }
    clarity = optimizer.compute_acoustic_clarity(punchy_seg)
    assert clarity > 0.65


def test_k_medoids_condensation(optimizer):
    """Test VQ cluster medoid selection and distortion reduction."""
    # Create 30 synthetic candidate segments across 3 distinct acoustic regions
    candidates = []
    for i in range(30):
        # 3 clusters: low centroid (bass), mid centroid, high centroid (hat)
        group = i % 3
        centroid = 150.0 if group == 0 else (1800.0 if group == 1 else 7500.0)
        pitch = 0 if group == 0 else (4 if group == 1 else 7)
        pitches = [0.0] * 12
        pitches[pitch] = 1.0

        candidates.append({
            "id": f"seg_{i}",
            "track_id": f"track_{i % 5}",
            "timbre": [float(group * 15.0)] * 12,
            "pitches": pitches,
            "spectral_centroid": centroid + np.random.normal(0, 10),
            "loudness_start": -30.0,
            "loudness_max": -4.0,
            "duration": 0.25,
        })

    # Condense 30 candidates into k=3 optimal medoids
    medoids, distortion = optimizer.condense_category_codebook(candidates, k_clusters=3)

    assert len(medoids) == 3
    assert distortion >= 0.0

    # Ensure all selected medoids have analysis metadata
    for m in medoids:
        assert "cluster_id" in m
        assert "cluster_size" in m
        assert "clarity_score" in m
        assert "timbre_vector" in m
        assert len(m["timbre_vector"]) == 12


def test_clean_slice_audio_dsp():
    """Test DSP audio normalization and click-free cosine micro-fades."""
    sr = 44100
    t = np.linspace(0, 0.30, int(sr * 0.30))
    raw = np.sin(2 * np.pi * 220.0 * t) * 2.0  # clipping

    cleaned = AcousticVocabularyOptimizer.clean_slice_audio(
        raw,
        sr=sr,
        target_dur=0.20,
        peak_target=0.88,
        fade_in_ms=2.0,
        fade_out_ms=25.0,
    )

    # 1. Truncated duration
    assert len(cleaned) == int(0.20 * sr)

    # 2. Peak normalization
    peak = float(np.max(np.abs(cleaned)))
    assert peak <= 0.88 + 1e-4

    # 3. Micro-fades prevent click pop on zero-crossing
    assert abs(cleaned[0]) < 0.005
    assert abs(cleaned[-1]) < 0.005


def test_condense_manifest_metadata():
    """Test recursive stripping of heavy base64 strings."""
    heavy_manifest = {
        "version": 3,
        "slices": {
            "drums": {
                "kick": [
                    {"name": "kick1.wav", "url": "data/slices/kick1.wav", "b64": "data:audio/wav;base64,AAAA..."},
                    {"name": "kick2.wav", "url": "data/slices/kick2.wav", "b64": "data:audio/wav;base64,BBBB..."},
                ]
            }
        },
        "style_kits": {
            "wave": {
                "stems": {
                    "bass": {
                        "sub": [
                            {"name": "sub1.wav", "url": "data/slices/sub1.wav", "b64": "data:audio/wav;base64,CCCC..."}
                        ]
                    }
                }
            }
        }
    }

    condensed = AcousticVocabularyOptimizer.condense_manifest_metadata(heavy_manifest)

    # Verify b64 is purged from all nested levels
    assert "b64" not in condensed["slices"]["drums"]["kick"][0]
    assert "b64" not in condensed["slices"]["drums"]["kick"][1]
    assert "b64" not in condensed["style_kits"]["wave"]["stems"]["bass"]["sub"][0]

    # Verify essential playback fields remain intact
    assert condensed["slices"]["drums"]["kick"][0]["url"] == "data/slices/kick1.wav"
