"""Layout + metadata contract for the 160-d taste dataset.

The first 48 columns are the original v1 block.  The new columns are:

    48..111   4 stems x 16 onset bits (16-step bar grid, 4 bars folded)
   112..123   bass chroma
   124..135   other chroma
   136..147   bass-root one-hot
   148..159   chord-root one-hot

These tests build a tiny analysis JSON in memory so they need no corpus.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from scripts.build_taste_dataset import (
    BASE_FEATURE_DIM, CHROMA_DIM, FEATURE_DIM, ONSET_DIM,
    STEMS, TATUMS_PER_BAR, _WindowFeaturizer, _root_onehot,
    _validate,
)
from scripts.merge_kit_metadata import merge
from wub_core.section_arcs import bar_features, segment_sections

N_BARS = 20


def _make_analysis(seed: int = 505) -> dict:
    rng = random.Random(seed)
    tatums: list[dict] = []
    segments: list[dict] = []
    for stem in STEMS:
        for bar in range(N_BARS):
            pitches = [rng.random() if rng.random() < 0.35 else 0.0
                       for _ in range(CHROMA_DIM)]
            total = sum(pitches) or 1.0
            segments.append({
                "id": f"seg_{stem}_{bar:04d}",
                "track_id": "syn",
                "stem": stem,
                "start": round(bar * 2.0, 6),
                "duration": 2.0,
                "is_silent": False,
                "loudness_start": -30.0,
                "loudness_max": -18.0 - rng.random() * 6.0,
                "loudness_max_time": 0.4,
                "pitches": [v / total for v in pitches],
                "timbre": [rng.gauss(0, 1) for _ in range(12)],
            })
    # A bass note that enters at tatum 5 of bar 1 and is referenced by the
    # following tatums: it must set exactly one folded bit (position 5).
    segments.append({
        "id": "seg_bass_extra",
        "track_id": "syn",
        "stem": "bass",
        "start": round((TATUMS_PER_BAR + 5) * 0.125, 6),
        "duration": 0.5,
        "is_silent": False,
        "loudness_start": -30.0,
        "loudness_max": -16.0,
        "loudness_max_time": 0.1,
        "pitches": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0],
        "timbre": [0.0] * 12,
    })
    for bar in range(N_BARS):
        for tatum in range(TATUMS_PER_BAR):
            index = bar * TATUMS_PER_BAR + tatum
            ids = {stem: [f"seg_{stem}_{bar:04d}"] for stem in STEMS}
            if bar == 1 and 5 <= tatum <= 8:
                ids["bass"].append("seg_bass_extra")
            tatums.append({
                "id": f"tatum_{index:04d}",
                "index": index,
                "bar_position": bar,
                "beat_position": tatum // 4,
                "tatum_position": tatum,
                "start": round(index * 0.125, 6),
                "duration": 0.125,
                "confidence": 0.9,
                "stem_segments": ids,
            })
    return {
        "track_id": "syn", "filename": "syn.wav", "bpm": 120.0,
        "key": "A minor", "tatums": tatums, "segments": segments,
    }


@pytest.fixture(scope="module")
def featurizer():
    analysis = _make_analysis()
    bar_feats = bar_features(analysis["segments"], analysis["tatums"])
    spans = segment_sections(bar_feats)
    return _WindowFeaturizer(analysis, bar_feats, spans, N_BARS)


def test_layout_is_160_and_v1_tail_is_reserved(featurizer):
    assert FEATURE_DIM == 160
    assert BASE_FEATURE_DIM == 48
    assert ONSET_DIM == 64
    vector, _ = featurizer.describe(0)
    assert len(vector) == FEATURE_DIM
    assert all(math.isfinite(v) for v in vector)
    # v1 reserved tail stays zero; section one-hot still sums to one.
    assert vector[41:48] == [0.0] * 7
    assert abs(sum(vector[34:39]) - 1.0) < 1e-6


def test_onset_bits_are_folded_and_counted_once(featurizer):
    vector, _ = featurizer.describe(1)  # bars 1..4
    bits = vector[BASE_FEATURE_DIM:BASE_FEATURE_DIM + ONSET_DIM]
    assert set(bits) <= {0.0, 1.0}
    per_stem = [bits[i * 16:(i + 1) * 16] for i in range(len(STEMS))]
    drums, bass, other, vocals = per_stem
    # One bar-start segment per bar folds onto step 0 for every stem.
    assert sum(drums) == 1 and drums[0] == 1.0
    assert sum(other) == 1 and other[0] == 1.0
    assert sum(vocals) == 1 and vocals[0] == 1.0
    # The extra bass note at tatum 5 sets step 5 once, despite being
    # referenced by four consecutive tatums.
    assert sum(bass) == 2.0
    assert bass[0] == 1.0 and bass[5] == 1.0
    assert sum(featurizer.onset_bad.values()) == 0


def test_harmonic_blocks_and_metadata(featurizer):
    vector, meta = featurizer.describe(1)
    bass_chroma = vector[112:124]
    other_chroma = vector[124:136]
    bass_root = vector[136:148]
    chord_root = vector[148:160]
    for block in (bass_chroma, other_chroma):
        assert abs(sum(block) - 1.0) < 1e-4
    for block in (bass_root, chord_root):
        assert abs(sum(block) - 1.0) < 1e-4
        assert block.count(1.0) == 1
    assert meta["chroma"] == vector[:CHROMA_DIM]
    assert meta["bass_root"] == int(np.argmax(bass_chroma))
    assert bass_root[meta["bass_root"]] == 1.0
    # start_bar <= 4 is an intro; bar 8 of a 20-bar track is a middle window.
    assert meta["is_intro"] is True and meta["once"] is True
    mid, mid_meta = featurizer.describe(8)
    assert mid_meta["is_intro"] is False
    assert mid_meta["is_outro"] is False
    assert mid_meta["once"] is False
    late, late_meta = featurizer.describe(16)
    assert late_meta["is_outro"] is True and late_meta["once"] is True
    assert len(mid) == FEATURE_DIM and len(late) == FEATURE_DIM


def test_root_onehot_convention():
    onehot, root = _root_onehot([0.0] * CHROMA_DIM)
    assert root == 0 and onehot[0] == 1.0
    onehot, root = _root_onehot([0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0])
    assert root == 3 and onehot[3] == 1.0


def test_validate_accepts_a_full_window_and_merge_copies_fields(featurizer):
    windows = []
    for start_bar in (0, 4, 8, 16):
        vector, meta = featurizer.describe(start_bar)
        windows.append({
            "start_bar": start_bar,
            "section": featurizer.sec_of_bar[start_bar],
            "energy": 0.5,
            "aligned": True,
            **meta,
            "features": vector,
        })
    dataset = {
        "version": 2,
        "feature_dim": FEATURE_DIM,
        "tracks": [{"track_id": "syn", "style": "house", "bpm": 120.0,
                    "key": "A minor", "windows": windows}],
        "kit_phrases": [{
            "phrase_id": "phrase_08",
            "track_id": "syn",
            "start_bar": 8,
            "chroma": windows[2]["chroma"],
            "bass_root": windows[2]["bass_root"],
            "is_intro": windows[2]["is_intro"],
            "is_outro": windows[2]["is_outro"],
            "once": windows[2]["once"],
            "features": windows[2]["features"],
        }],
    }
    assert _validate(dataset, n_phrases=1, matched=1,
                     min_windows=1, min_frac=0.0) == []

    manifest = {"version": 1, "bars": 4, "kits": [{
        "track_id": "syn", "style": "house", "bpm": 120.0,
        "key": "A minor",
        "phrases": [{"id": "phrase_08", "start_bar": 8}],
    }]}
    stats = merge(dataset, manifest)
    assert stats["matched"] == 1 and stats["missing_count"] == 0
    phrase = manifest["kits"][0]["phrases"][0]
    for field in ("chroma", "bass_root", "is_intro", "is_outro", "once"):
        assert field in phrase
    assert phrase["chroma"] == windows[2]["chroma"]
    assert phrase["bass_root"] == windows[2]["bass_root"]
    assert phrase["once"] == (phrase["is_intro"] or phrase["is_outro"])
