"""Tests for the reference-free song-quality scorer (ordering + robustness)."""
from wub_core.song_quality import score_bar, score_song


def _bar(section="DROP", drums=(0, 4, 8, 12), bass=(), bpc=0, vel=0.95):
    grids = {"drums": list(drums), "bass": list(bass), "other": [], "vocals": []}
    sp = {"bass": {s: bpc for s in bass}, "other": {}, "vocals": {}}
    vels = {s: [vel if i in grids[s] else 0.0 for i in range(16)] for s in grids}
    return {"section": section, "chord_root": 0, "chord_minor": True,
            "onset_grids": grids, "step_pitches": sp, "velocities": vels}


def test_good_drop_beats_random_mush():
    good = [_bar("DROP", drums=(0, 4, 8, 12), bass=(2, 6, 10, 14), bpc=0) for _ in range(8)]
    bad = [_bar("DROP", drums=(1, 3, 5, 9, 13), bass=(0, 1, 3), bpc=1) for _ in range(8)]
    assert score_song(good)["score"] > score_song(bad)["score"] + 10.0


def test_deterministic_and_bounded():
    bars = [_bar() for _ in range(4)]
    a, b = score_song(bars)["score"], score_song(bars)["score"]
    assert a == b and 0.0 <= a <= 100.0


def test_empty_song_defined():
    assert score_song([])["score"] == 0.0
    silent = [_bar(drums=(), bass=()) for _ in range(4)]
    s = score_song(silent)["score"]
    assert 0.0 <= s <= 100.0


def test_score_bar_rerank_signal():
    prev = _bar()
    good = _bar(bass=(2, 6, 10, 14), bpc=0)
    bad = _bar(drums=(1, 5, 9), bass=(0, 1), bpc=6)  # semitone rub vs C + no downbeats
    assert score_bar(good, prev)["score"] > score_bar(bad, prev)["score"]
