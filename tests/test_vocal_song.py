"""Tests for vocal song arrangement: half the vocals, full verses/choruses."""

from wub_core.vocal_song import (
    backing_should_change,
    prefer_unit,
    presence_target,
    select_unit,
    should_rest,
    VocalUnit,
)


def test_presence_targets_are_half_wallpaper():
    # Old engine ~100% vocal blocks; vocal song targets ~40-50% overall.
    assert presence_target("INTRO") <= 0.15
    assert presence_target("BREAKDOWN") <= 0.25
    assert presence_target("DROP") >= 0.6  # chorus still sings
    assert presence_target("GROOVE_A") <= 0.5
    # Overall average across a 104-bar set with song blueprint weights
    # should be well under 100%.
    weights = {"INTRO": 8, "GROOVE_A": 32, "BUILDUP": 16, "DROP": 32, "BREAKDOWN": 16}
    total = sum(weights.values())
    avg = sum(presence_target(s) * n for s, n in weights.items()) / total
    assert 0.3 <= avg <= 0.55, f"avg vocal presence {avg:.2f} should be ~half"


def test_rests_are_deterministic_and_sparse_where_expected():
    # INTRO/BREAKDOWN rest most blocks; DROP sings most blocks.
    intro_rests = sum(should_rest("INTRO", b) for b in range(20))
    drop_rests = sum(should_rest("DROP", b) for b in range(20))
    assert intro_rests > drop_rests
    assert intro_rests >= 15  # 10% presence -> ~18 rests / 20
    assert drop_rests <= 10  # 70% presence -> ~6 rests / 20
    # Deterministic
    assert should_rest("DROP", 5) == should_rest("DROP", 5)


def test_long_units_win_verse_chorus():
    assert prefer_unit(8, "VERSE_1") > prefer_unit(4, "VERSE_1")
    assert prefer_unit(16, "CHORUS_1") >= prefer_unit(4, "CHORUS_1")
    assert prefer_unit(4, "BUILDUP_1") >= prefer_unit(16, "BUILDUP_1")


def test_section_holds_play_verse_chorus_before_switching():
    from wub_core.vocal_song import vocal_hold_for_bar
    assert vocal_hold_for_bar(10) == 16  # VERSE_1
    assert vocal_hold_for_bar(40) == 16  # CHORUS_1
    assert vocal_hold_for_bar(60) == 16  # VERSE_2
    assert vocal_hold_for_bar(85) == 16  # CHORUS_2
    assert vocal_hold_for_bar(0) == 8  # INTRO
    assert vocal_hold_for_bar(50) == 8  # BREAKDOWN


def test_backing_follows_vocal_entrances():
    # Drums hold under new vocals; bass/other move with them.
    assert backing_should_change("drums", "CHORUS") < 0.15
    assert backing_should_change("bass", "CHORUS") > 0.4
    assert backing_should_change("other", "VERSE") > 0.5


def test_theme_recall_wins():
    units = [
        VocalUnit("t1", ["a", "b"], 8, score=0.9),
        VocalUnit("t2", ["c", "d"], 8, score=0.1),
    ]
    picked = select_unit(units, "CHORUS_2", recalled_ids=["c", "d"])
    assert picked is not None and picked.ids == ["c", "d"]
    # Without recall, higher score + length preference wins.
    picked2 = select_unit(units, "CHORUS_1")
    assert picked2 is not None and picked2.ids == ["a", "b"]
