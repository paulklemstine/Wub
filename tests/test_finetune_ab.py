"""Tests for A/B preference data plumbing (no GPU needed)."""
import json

from scripts.finetune_from_ab import load_pairs


def _bars(n, start=0):
    return [[(start + i + j) % 64 for j in range(6)] for i in range(n)]


def test_load_pairs_filters_and_splits(tmp_path):
    ab = {
        "results": [
            {"pair_id": "a", "winner": "X", "skipped": False,
             "mapping": {"X": "new", "Y": "old"},
             "bars": {"new": _bars(4), "old": _bars(4, 10)}},
            {"pair_id": "b", "winner": "Y", "skipped": False,
             "mapping": {"X": "new", "Y": "old"},
             "bars": {"new": _bars(4, 20), "old": _bars(4, 30)}},
            {"pair_id": "c", "winner": "X", "skipped": True,
             "bars": {"new": _bars(4), "old": _bars(4)}},
            {"pair_id": "d", "winner": "X", "skipped": False},  # no bars
        ]
    }
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(ab))
    winners, losers = load_pairs(p)
    assert len(winners) == 2 and len(losers) == 2
    assert all(len(w) == 24 for w in winners)  # 4 bars x 6 tokens
    # skipped / bar-less votes excluded
    assert all(len(w) >= 12 for w in winners + losers)


def test_seed_misuse_guard():
    # torch manual_seed path exercised in dry-run (see CI); here just check
    # that an empty results file yields empty pairs, not an exception.
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"results": []}, f)
        name = f.name
    try:
        w, l = load_pairs(__import__("pathlib").Path(name))
        assert w == [] and l == []
    finally:
        os.unlink(name)
