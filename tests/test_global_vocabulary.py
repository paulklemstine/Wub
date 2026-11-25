"""Tests for GlobalVocabulary two-stage compression."""
import numpy as np

from wub_core.global_vocabulary import (
    GlobalVocabularyBuilder,
    best_pitch_shift,
    pitch_invariant_dist,
)


def _seg(stem, tid, sid, timbre0, pitch, loud=-8.0, dur=0.25, cent=1200.0):
    chroma = [0.0] * 12
    chroma[pitch] = 1.0
    return {
        "id": sid,
        "track_id": tid,
        "stem": stem,
        "start": 0.0,
        "duration": dur,
        "is_silent": False,
        "loudness_start": loud - 10.0,
        "loudness_max": loud,
        "pitches": chroma,
        "timbre": [timbre0] + [0.0] * 11,
        "spectral_centroid": cent,
    }


def test_pitch_invariant_and_shift():
    a = np.array([0.0] * 12 + [1.0] + [0.0] * 11 + [-8.0, 0.2], dtype=np.float32)
    b = np.array([0.0] * 12 + [0.0] * 7 + [1.0] + [0.0] * 4 + [-8.0, 0.2], dtype=np.float32)
    d_same = pitch_invariant_dist(a, a, "other")
    d_rot = pitch_invariant_dist(a, b, "other")
    assert d_same < 1e-6
    assert d_rot < 5.0  # rotation-invariant, small
    assert best_pitch_shift(0, 7) == -5
    assert best_pitch_shift(0, 5) == 5
    r = (7 - 0) % 12
    if r > 6:
        r -= 12
    assert r == -5


def test_two_stage_build_small():
    builder = GlobalVocabularyBuilder(
        target_counts={"drums": 4, "bass": 3, "other": 3, "vocals": 2}, k_per_song=2
    )
    # Fake 2 tracks worth of segments
    from pathlib import Path
    import json, tempfile

    with tempfile.TemporaryDirectory() as td:
        for ti in range(2):
            tid = f"track{ti}"
            segs = []
            for i in range(6):
                segs.append(_seg("drums", tid, f"seg_d_{ti}_{i}", 10.0 + i, 0, dur=0.12, cent=500.0))
            for i in range(4):
                segs.append(_seg("bass", tid, f"seg_b_{ti}_{i}", 5.0 + i, 0, dur=1.0, cent=200.0))
            for i in range(4):
                segs.append(_seg("other", tid, f"seg_o_{ti}_{i}", 4.0 + i, 0, dur=2.0, cent=1500.0))
            for i in range(3):
                segs.append(_seg("vocals", tid, f"seg_v_{ti}_{i}", 6.0 + i, 5, dur=0.4, cent=2000.0))
            Path(td, f"{tid}.json").write_text(
                json.dumps({"track_id": tid, "stems_available": ["drums", "bass", "other", "vocals"], "segments": segs})
            )
        cands = builder.collect_candidates(Path(td))
        assert sum(len(v) for v in cands.values()) > 0
        gv = builder.build(cands)
        assert gv.total() > 0
        assert len(gv.vocab_by_stem["drums"]) <= 4
        assert len(gv.vocab_by_stem["bass"]) <= 3
        assert len(gv.vocab_by_stem["other"]) <= 3
        assert len(gv.vocab_by_stem["vocals"]) <= 2
        # find_nearest works + pitch shift bounded
        import numpy as np

        feat = np.array([10.0] + [0.0] * 11 + [1.0] + [0.0] * 11 + [-8.0, 0.2], dtype=np.float32)
        m, shift = gv.find_nearest("drums", feat)
        assert m is not None and shift == 0
        feat2 = np.array([4.0] + [0.0] * 11 + [0.0] * 5 + [1.0] + [0.0] * 6 + [-8.0, 0.7], dtype=np.float32)
        m2, sh2 = gv.find_nearest("other", feat2)
        assert m2 is not None and -6 <= sh2 <= 6


def test_remap_bars_fidelity():
    builder = GlobalVocabularyBuilder(
        target_counts={"drums": 2, "bass": 2, "other": 2, "vocals": 2}, k_per_song=2
    )
    segs = [
        _seg("drums", "t1", "d0", 15.0, 0, dur=0.12, cent=500.0),
        _seg("bass", "t1", "b0", 5.0, 0, dur=1.0, cent=200.0),
    ]
    tatums = []
    for step in range(16):
        stem_segs = {}
        if step in (0, 4, 8, 12):
            stem_segs["drums"] = ["d0"]
        if step in (2, 6):
            stem_segs["bass"] = ["b0"]
        stem_segs.setdefault("other", [])
        stem_segs.setdefault("vocals", [])
        tatums.append({"stem_segments": stem_segs})
    import tempfile, json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        Path(td, "t1.json").write_text(
            json.dumps({"track_id": "t1", "stems_available": ["drums", "bass"], "segments": segs})
        )
        cands = builder.collect_candidates(Path(td))
        gv = builder.build(cands)
        bars, fid = builder.remap_bars(segs, tatums, gv)
        assert len(bars) == 1
        assert fid["drums"] >= 0.70
