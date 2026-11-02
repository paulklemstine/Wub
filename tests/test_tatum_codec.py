"""Tests for tatum-level helpers (rank map, fuzzy match, spans)."""
from wub_core.tatum_codec import nearest_rank, onset_spans, rank_index


def _protos():
    return {
        "drums": [
            {"signature": "a-.-.-.", "tokens": ["a", None, None, None]},
            {"signature": "b-.-c-.", "tokens": ["b", None, "c", None] + [None] * 12},
        ],
        "bass": [{"signature": ".", "tokens": [None] * 16}],
        "other": [{"signature": ".", "tokens": [None] * 16}],
        "vocals": [{"signature": ".", "tokens": [None] * 16}],
    }


def test_rank_index_exact():
    idx = rank_index(_protos())
    assert idx["drums"]["a-.-.-."] == 0
    assert idx["drums"].get("nope") is None


def test_nearest_rank_fuzzy_matches_grid():
    protos = _protos()
    grids = {("drums", r): tuple(i for i, t in enumerate(p["tokens"]) if t) for r, p in enumerate(protos["drums"])}
    cats = {("drums", r): tuple("kick" if t else "." for t in p["tokens"]) for r, p in enumerate(protos["drums"])}
    id_cat = {"a": "kick", "b": "snare", "c": "kick"}
    r, exact = nearest_rank("drums", "a-.-.-.", ["a", None, None, None], {"a-.-.-.": 0}, grids, cats, id_cat)
    assert (r, exact) == (0, True)
    # same grid, different slice id -> fuzzy match to rank 0
    r, exact = nearest_rank("drums", "z-.-.-.", ["z", None, None, None], {"a-.-.-.": 0}, grids, cats, {"z": "kick"})
    assert r == 0 and exact is False


def test_onset_spans_cap():
    steps = ["x"] + [None] * 15
    assert onset_spans(steps) == {0: 8}  # capped at 8
    steps = ["x", None, "y"] + [None] * 13
    assert onset_spans(steps) == {0: 2, 2: 8}
