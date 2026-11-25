"""Tests for wub_core.global_joint_cohesion on tiny synthetic bars."""
from __future__ import annotations

import math

from wub_core.global_joint_cohesion import (
    bar_stem_sig,
    joint_sig,
    mine_call_response,
    mine_joint_tuples,
    passes_theory,
    rerank_bars,
)

STEMS = ("drums", "bass", "other", "vocals")

D_KICK = "g_drums_00001"
D_HAT = "g_drums_00002"
B_SUB = "g_bass_00001"
B_STAB = "g_bass_00002"
O_CHORD = "g_other_00001"
O_PAD = "g_other_00002"
V_CHOP = "g_vocals_00001"
V_PHRASE = "g_vocals_00002"

VOCAB = {
    D_KICK: {"pitch": 0, "category": "drums:kick"},
    D_HAT: {"pitch": 0, "category": "drums:hat_closed"},
    B_SUB: {"pitch": 0, "category": "bass:sub"},
    B_STAB: {"pitch": 7, "category": "bass:stab"},
    O_CHORD: {"pitch": 0, "category": "other:chord"},
    O_PAD: {"pitch": 7, "category": "other:pad"},
    V_CHOP: {"pitch": 0, "category": "vocals:chop"},
    V_PHRASE: {"pitch": 4, "category": "vocals:phrase"},
}


def _blank():
    return {s: [None] * 16 for s in STEMS}


def _bar_J1():
    b = _blank()
    for st in (0, 4, 8, 12):
        b["drums"][st] = D_KICK
    for st in (2, 6, 10, 14):
        b["bass"][st] = B_STAB
    b["other"][0] = O_CHORD
    b["other"][8] = O_CHORD
    b["vocals"][4] = V_CHOP
    return b


def _bar_J2():
    b = _blank()
    for st in (2, 6, 10, 14):
        b["drums"][st] = D_HAT
    b["bass"][2] = B_SUB
    b["other"][0] = O_PAD
    b["vocals"][0] = V_PHRASE
    return b


def _bar_collision():
    b = _blank()
    b["drums"][0] = D_KICK
    b["bass"][0] = B_SUB  # same downbeat step as kick -> must fail
    b["other"][0] = O_CHORD
    b["vocals"][4] = V_CHOP
    return b


def test_top_joint_tuple_detected():
    j1 = _bar_J1()
    j2 = _bar_J2()
    # 2 tracks, J1 co-occurs 5x, J2 only 1x.
    tracks = [[dict(j1), dict(j1), dict(j1), dict(j1)], [dict(j1), dict(j2)]]
    # NOTE: dict(j1) is shallow; rebuild to avoid shared list mutation.
    tracks = [[_bar_J1() for _ in range(4)], [_bar_J1(), _bar_J2()]]
    tuples = mine_joint_tuples(tracks, top_n=2000)
    assert len(tuples) == 2
    top = tuples[0]
    assert top["count"] == 5
    assert top["sig"] == joint_sig(_bar_J1())
    # PMI fields present and finite.
    assert math.isfinite(float(top["pmi"]))
    assert float(top["pmi_ratio"]) > 0
    # Per-section counts recorded.
    assert isinstance(top["sections"], dict)
    assert sum(top["sections"].values()) == top["count"]


def test_theory_gate_rejects_kick_sub_collision():
    clean = _bar_J1()
    bad = _bar_collision()
    chord = {"root": 0}
    assert passes_theory(clean, chord, VOCAB) is True
    assert passes_theory(bad, chord, VOCAB) is False
    # Substring fallback without vocab: IDs containing kick/sub still reject.
    bad2 = _blank()
    bad2["drums"][4] = "my_kick_slice"
    bad2["bass"][4] = "my_sub_slice"
    assert passes_theory(bad2, chord, None) is False
    good2 = _blank()
    good2["drums"][4] = "my_kick_slice"
    good2["bass"][6] = "my_sub_slice"  # different steps -> no collision
    assert passes_theory(good2, chord, None) is True


def test_call_response_probs_sum_to_one():
    tracks = [[_bar_J1() for _ in range(4)], [_bar_J1(), _bar_J2()]]
    cr = mine_call_response(tracks, max_next=8)
    # Self transitions exist per stem.
    assert "drums" in cr and "bass" in cr
    # Cross-stem key e.g. vocals->drums recorded.
    assert "vocals->drums" in cr
    d_prev = bar_stem_sig(_bar_J1()["drums"])
    assert d_prev in cr["drums"]
    probs = cr["drums"][d_prev]
    # J1 drums -> J1 drums (3x) + J1 drums -> J2 drums (1x) = 4 total.
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert len(probs) <= 8
    # Cross probs also normalized.
    v_prev = bar_stem_sig(_bar_J1()["vocals"])
    assert v_prev in cr["vocals->drums"]
    assert abs(sum(cr["vocals->drums"][v_prev].values()) - 1.0) < 0.01


def test_rerank_prefers_theory_clean():
    clean = _bar_J1()
    bad = _bar_collision()
    ranked = rerank_bars(
        [bad, clean],
        chord={"root": 0},
        vocab=VOCAB,
        pmi_lookup={},
        audio_scores=[0.5, 0.5],
    )
    assert len(ranked) == 2
    # Clean passes theory (1.0) vs collision fails (0.0) -> clean first.
    assert ranked[0]["bar"] is clean
    assert ranked[0]["theory"] == 1.0
    assert ranked[1]["theory"] == 0.0
    assert ranked[0]["score"] > ranked[1]["score"]
