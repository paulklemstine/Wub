"""Tests for the tatum dataset stem-coverage gate."""

from __future__ import annotations

from scripts.build_tatum_dataset import _track_has_stems

ALL = {"drums", "bass", "other", "vocals"}


def test_all_four_stems_in_analysis_accepted():
    assert _track_has_stems({"stems_available": sorted(ALL)}, "tid", None)


def test_missing_stems_rejected_without_list():
    assert not _track_has_stems({"stems_available": ["other"]}, "tid", None)


def test_stems_list_overrides_stale_field():
    assert _track_has_stems({"stems_available": ["other"]}, "tid", {"tid"})


def test_stems_list_does_not_accept_unknown_track():
    assert not _track_has_stems({"stems_available": ["other"]}, "tid", {"other-tid"})


def test_track_id_falls_back_to_filename():
    assert _track_has_stems({}, "file_stem", {"file_stem"})


def test_stems_list_accepts_analysis_without_field():
    assert _track_has_stems({"track_id": "abc"}, "unused", {"abc"})
