"""Tests for the fills/pocket/loudness prototype spec (pure functions)."""

from wub_core.fill_pocket_spec import (
    FILL_CATEGORIES,
    GAIN_MAX,
    GAIN_MIN,
    POCKET_BOUND_MS,
    detect_fill_bar,
    humanize_velocity,
    loudness_normalize_gain,
    pocket_nudge,
)


def test_fill_detector_fires_on_dense_drum_bar_with_fill_slices():
    onset_grid = [0, 4, 6, 8, 10, 12, 13, 14]
    cats = ["kick", "snare", "hat", "kick", "hat", "snare", "snare", "fill"]
    assert "fill" in FILL_CATEGORIES
    assert detect_fill_bar(onset_grid, cats) is True


def test_fill_detector_fires_on_canonical_turnaround_run():
    # Beat-4 run 12/13/14 counts even without labelled fill slices.
    assert detect_fill_bar([0, 4, 8, 12, 13, 14], ["kick"] * 6) is True


def test_fill_detector_quiet_on_sparse_groove():
    assert detect_fill_bar([0, 4, 8, 12], ["kick", "snare", "kick", "snare"]) is False
    assert detect_fill_bar([], []) is False


def test_pocket_nudges_match_priors_and_stay_bounded():
    assert pocket_nudge("kick", 0) == -2.0
    assert pocket_nudge("snare", 4) == 4.0
    assert pocket_nudge("hat", 3) == 8.0
    assert pocket_nudge("bass", 6) == 6.0
    for stem in ("kick", "snare", "hat", "hats", "bass", "drums", "other", "vocals"):
        for step in range(16):
            nudge = pocket_nudge(stem, step)
            assert -POCKET_BOUND_MS <= nudge <= POCKET_BOUND_MS


def test_humanize_velocity_within_eight_percent():
    assert humanize_velocity(1.0, 1.0) == 1.08
    assert humanize_velocity(1.0, -1.0) == 0.92
    assert humanize_velocity(0.9, 0.0) == 0.9


def test_gains_bounded_half_to_two():
    assert loudness_normalize_gain(-14.0) == 1.0
    for loud in (-30.0, -24.0, -20.0, -14.0, -8.0, -6.0, 0.0):
        g = loudness_normalize_gain(loud)
        assert GAIN_MIN <= g <= GAIN_MAX
    # Quiet slices boost, hot slices attenuate.
    assert loudness_normalize_gain(-20.0) > 1.0
    assert loudness_normalize_gain(-8.0) < 1.0
