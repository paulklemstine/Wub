"""Tests for longer samples, slices, and feature adjustments across stems."""
import numpy as np
import pytest

from wub_core.analyzer import slice_stem, STEM_MIN_SEGMENT, STEM_MAX_SEGMENT
from wub_core.model_transformer import (
    FEAT_DIM,
    _STD_IDX,
    StemVQ,
    slice_feature_raw,
    STEM_ORDER,
)
from wub_core.model_markov import SliceRef
from wub_core.acoustic_vocabulary_optimizer import AcousticVocabularyOptimizer


def test_stem_specific_segmentation_bounds():
    """Verify that stem-specific bounds prevent micro-chopping and allow longer slices."""
    assert STEM_MIN_SEGMENT["drums"] >= 0.050
    assert STEM_MIN_SEGMENT["bass"] >= 0.100  # Bass cycles must not be micro-chopped
    assert STEM_MIN_SEGMENT["other"] >= 0.120  # Chords and pads need sustained acoustic presence
    assert STEM_MIN_SEGMENT["vocals"] >= 0.150

    assert STEM_MAX_SEGMENT["drums"] >= 4.0
    assert STEM_MAX_SEGMENT["bass"] >= 6.0
    assert STEM_MAX_SEGMENT["other"] >= 8.0
    assert STEM_MAX_SEGMENT["vocals"] >= 8.0


def test_26d_feature_vector_with_duration():
    """Verify that 26-D feature vectors incorporate log-duration for WubGPT."""
    assert FEAT_DIM == 26
    assert 24 in _STD_IDX and 25 in _STD_IDX  # Loudness and log-duration are standardized

    # Create short vs long slice
    short_slice = SliceRef(
        stem="bass",
        track_id="trk_1",
        segment_id="seg_short",
        start=0.0,
        duration=0.15,
        timbre=np.zeros(12, dtype=np.float32),
        chroma=np.zeros(12, dtype=np.float32),
        loud_max=-6.0,
        track_pos=0,
        track_bpm=120.0,
    )
    long_slice = SliceRef(
        stem="bass",
        track_id="trk_1",
        segment_id="seg_long",
        start=0.0,
        duration=2.50,
        timbre=np.zeros(12, dtype=np.float32),
        chroma=np.zeros(12, dtype=np.float32),
        loud_max=-6.0,
        track_pos=0,
        track_bpm=120.0,
    )

    f_short = slice_feature_raw(short_slice)
    f_long = slice_feature_raw(long_slice)

    assert f_short.shape == (26,)
    assert f_long.shape == (26,)
    # Log duration feature at index 25 should distinguish short vs long notes
    assert f_long[25] > f_short[25]


def test_stem_vq_fits_and_encodes_duration():
    """Verify StemVQ clusters distinct duration profiles."""
    vq = StemVQ(stem="bass", k=2)
    slices = [
        SliceRef(
            stem="bass",
            track_id="trk_1",
            segment_id=f"seg_{i}",
            start=float(i),
            duration=0.15 if i % 2 == 0 else 2.50,
            timbre=np.full(12, 0.1 * i, dtype=np.float32),
            chroma=np.zeros(12, dtype=np.float32),
            loud_max=-8.0,
            track_pos=i,
            track_bpm=120.0,
        )
        for i in range(8)
    ]
    vq.fit(slices)
    assert vq.fitted
    assert vq.n_clusters == 2

    # Verify short and long slices encode to valid token IDs
    tok_short = vq.encode(slices[0])
    tok_long = vq.encode(slices[1])
    assert 0 <= tok_short < vq.n_clusters
    assert 0 <= tok_long < vq.n_clusters

    # Centroids in raw feature space have dimension 26
    c0 = vq.centroid(tok_short)
    assert c0.shape == (26,)


def test_stem_aware_acoustic_clarity_for_sustained_slices():
    """Verify that longer slices for bass and other stems receive high clarity scores."""
    opt = AcousticVocabularyOptimizer()

    sustained_bass = {
        "stem": "bass",
        "is_silent": False,
        "loudness_start": -28.0,
        "loudness_max": -3.0,
        "duration": 1.5,
    }
    clarity_bass = opt.compute_acoustic_clarity(sustained_bass)
    assert clarity_bass > 0.60

    sustained_chord = {
        "stem": "other",
        "is_silent": False,
        "loudness_start": -26.0,
        "loudness_max": -4.0,
        "duration": 2.2,
    }
    clarity_chord = opt.compute_acoustic_clarity(sustained_chord)
    assert clarity_chord > 0.60
