"""Tests for song foundation: repetition like training songs."""

from wub_core.song_foundation import (
    build_loop_order,
    is_compatible,
    movement_budget_song,
    repetition_bonus,
    section_for_bar,
    should_return,
)


def test_section_plan_has_song_durations():
    # Sections last 8-16 bars, not 4-bar churn.
    sec0, _ = section_for_bar(0)
    assert sec0 == "INTRO"
    sec8, _ = section_for_bar(8)
    assert sec8 == "GROOVE_A"
    sec32, e32 = section_for_bar(32)
    assert sec32 == "DROP"
    assert e32 > 0.9
    # Same section holds across an 8-bar span.
    assert section_for_bar(10)[0] == section_for_bar(14)[0] == "GROOVE_A"
    assert section_for_bar(34)[0] == section_for_bar(40)[0] == "DROP"


def test_movement_budget_holds_foundations():
    # Inside sections, 4-bar blocks mostly hold.
    b = movement_budget_song(10, 4, False)
    assert b["maxChanges"] <= 1
    assert b["maxHoldBars"] == 32
    # Section changes allow movement.
    b2 = movement_budget_song(32, 8, True)
    assert b2["maxChanges"] >= 2
    # Foundations last 32 bars (60s at 128 BPM), not 16.
    assert movement_budget_song(0, 16, False)["maxHoldBars"] == 32


def test_compatibility_rejects_extremes():
    ok, _ = is_compatible("drums", 1.02, 0, is_foundation=True)
    assert ok
    ok, reason = is_compatible("drums", 1.10, 0, is_foundation=True)
    assert not ok
    ok, _ = is_compatible("bass", 1.02, 2, chroma_cosine=0.75, root_interval=7)
    assert ok
    ok, _ = is_compatible("bass", 1.02, 2, chroma_cosine=0.75, root_interval=6)
    assert not ok  # tritone
    ok, _ = is_compatible("bass", 1.02, 6, chroma_cosine=0.75)
    assert not ok  # extreme shift
    ok, _ = is_compatible("other", 1.02, 0, chroma_cosine=0.40)
    assert not ok  # harmonic mush


def test_drums_return_more_than_vocals():
    assert should_return("drums", 8) > should_return("vocals", 8)
    assert should_return("drums", 8) > 0.4
    assert should_return("vocals", 8) < 0.15


def test_repetition_bonus_rewards_drums_return():
    # Fresh drums get small novelty, smaller than old 0.35 churn bonus.
    fresh_drums = repetition_bonus("drums", 0, None)
    assert 0 < fresh_drums < 0.15
    fresh_vocals = repetition_bonus("vocals", 0, None)
    assert fresh_vocals > fresh_drums  # vocals should be fresh
    # Return bonus beats novelty for drums.
    return_bonus = repetition_bonus("drums", 1, 8)
    assert return_bonus > 0.10
    # Third play penalized, vocals penalized most.
    assert repetition_bonus("vocals", 3, None) < repetition_bonus("drums", 3, None)


def test_loop_order_repeats_patterns():
    assert build_loop_order("AABB", 8) == [0, 0] or build_loop_order("AABB", 8) == [0, 1]
    # 16 bars = 4 phrase slots, AABB repeats pairs.
    order = build_loop_order("AABB", 16)
    assert len(order) == 4
    assert order[0] == order[1]  # AA
    assert order[2] == order[3]  # BB
    order_ab = build_loop_order("ABAB", 16)
    assert order_ab == [0, 1, 0, 1]
