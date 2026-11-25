"""Checkpoint search order: conditioned model preferred, legacy fallbacks."""
from __future__ import annotations

from server.stream_manager import WUBGPT_CANDIDATES, get_cached_wubgpt


def test_cond_checkpoint_preferred_over_legacy():
    assert [p.as_posix() for p in WUBGPT_CANDIDATES] == [
        "data/wubgpt/wubgpt_cond.pt",
        "data/wubgpt/wubgpt.pt",
        "data/wubgpt.pt",
    ]


def test_get_cached_wubgpt_callable_without_models(tmp_path, monkeypatch):
    import server.stream_manager as sm

    monkeypatch.chdir(tmp_path)
    sm._WUBGPT_CACHE.clear()
    try:
        m, v = get_cached_wubgpt()
        assert (m, v) == (None, None)
    finally:
        sm._WUBGPT_CACHE.clear()
