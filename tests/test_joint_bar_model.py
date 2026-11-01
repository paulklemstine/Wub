"""Tests for wub_core.joint_bar_model on tiny synthetic songs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wub_core import joint_bar_model as jm
from wub_core.schema import Tatum, TrackAnalysis


# ---------------------------------------------------------------------------
# synthetic song builders
# ---------------------------------------------------------------------------

BAR_A = {
    "drums": [0, 4, 8, 12],
    "bass": [2, 6, 10, 14],
    "other": [0, 8],
    "vocals": [0],
}
BAR_B = {
    "drums": [0, 2, 4, 6, 8, 10, 12, 14],
    "bass": [0, 4, 8, 12],
    "other": [0, 4, 8, 12],
    "vocals": [2, 6, 10, 14],
}
BAR_C = {
    "drums": [0],
    "bass": [0],
    "other": [],
    "vocals": [],
}
BAR_X = {
    "drums": [0, 4, 8, 12],
    "bass": [2, 6, 10, 14],
    "other": [0, 8],
    "vocals": [0],
}
BAR_Y = {
    "drums": [0, 2, 4, 8, 12],
    "bass": [2, 6, 10, 14],
    "other": [0, 8],
    "vocals": [0],
}
BAR_Z = {
    "drums": [0, 4, 8, 12],
    "bass": [0, 4, 8, 12],
    "other": [0, 8],
    "vocals": [0],
}


def make_track(track_id, bars, stems_available=("drums", "bass", "other", "vocals")):
    tatums = []
    idx = 0
    for b, bar in enumerate(bars):
        for step in range(16):
            seg = {}
            for stem in stems_available:
                onsets = bar.get(stem, [])
                seg[stem] = [f"{track_id}_{stem}_b{b}s{step}"] if step in onsets else []
            tatums.append(
                Tatum(
                    id=f"{track_id}_t{idx}",
                    index=idx,
                    bar_position=b,
                    beat_position=step // 4,
                    tatum_position=step % 4,
                    start=float(idx) * 0.1,
                    duration=0.1,
                    confidence=0.9,
                    stem_segments=seg,
                )
            )
            idx += 1
    return TrackAnalysis(
        track_id=track_id,
        filename=f"{track_id}.mp3",
        bpm=126.0,
        key="C major",
        duration=max(1.0, len(tatums) * 0.1),
        sample_rate=44100,
        stems_available=list(stems_available),
        tatums=tatums,
        segments=[],
    )


def _ids_of_first_song_tuple(seqs):
    return seqs[0]


# ---------------------------------------------------------------------------
# mining
# ---------------------------------------------------------------------------

def test_mine_tuples_basic():
    tracks = [
        make_track("song_a", [BAR_A, BAR_B, BAR_A, BAR_B]),
        make_track("song_b", [BAR_B, BAR_A, BAR_A, BAR_C]),
    ]
    seqs = jm.mine_bar_tuples(tracks)
    assert len(seqs) == 2
    for seq in seqs:
        assert len(seq) == 4
        for sec, tup in seq:
            assert sec in jm.SECTIONS
            assert isinstance(tup, tuple) and len(tup) == 4
            assert all(isinstance(c, int) and 0 <= c < jm.CODES_PER_STEM for c in tup)
    # <8 bars -> single phrase -> single section per song
    assert len({s for s, _ in seqs[0]}) == 1
    # identical bars mine identical tuples (joint signature is deterministic)
    assert seqs[0][0][1] == seqs[0][2][1]
    assert seqs[0][0][1] != seqs[0][1][1]


def test_mine_missing_stems_and_short_tracks():
    full = make_track("full", [BAR_A, BAR_B])
    partial = make_track("part", [BAR_A, BAR_B], stems_available=("drums", "bass"))
    seqs = jm.mine_bar_tuples([full, partial])
    assert len(seqs) == 2
    for _sec, tup in seqs[1]:
        for stem, code in zip(jm.STEMS, tup):
            groove, dens, mask = jm.decode_stem(code)
            if stem in ("other", "vocals"):
                assert groove == "missing"
                assert dens == 0 and mask == 0
    # tracks with <16 tatums are skipped
    tiny = make_track("tiny", [])
    assert jm.mine_bar_tuples([tiny]) == []


def test_mine_max_bars_cap():
    track = make_track("long", [BAR_A] * 70)
    seqs = jm.mine_bar_tuples([track], max_bars_per_song=64)
    assert len(seqs) == 1 and len(seqs[0]) == 64
    seqs2 = jm.mine_bar_tuples([track], max_bars_per_song=10)
    assert len(seqs2[0]) == 10


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _abc_model():
    tracks = [make_track(f"abc{i}", [BAR_A, BAR_B, BAR_C]) for i in range(3)]
    seqs = jm.mine_bar_tuples(tracks)
    sec = seqs[0][0][0]
    assert all(s == sec for tr in seqs for s, _ in tr)  # one shared section
    model = jm.train_joint_transitions(seqs, k=0.5)
    tups = [t for _, t in seqs[0]]
    assert len(set(tups)) == 3  # A, B, C distinct
    ids = [jm._resolve_id(model, t) for t in tups]
    return model, sec, ids


def test_transitions_deterministic_sequence():
    model, sec, (a, b, c) = _abc_model()
    assert model["vocab_size"] == 3
    assert model["transitions"][sec][str(a)][str(b)] == 3
    assert model["transitions"][sec][str(b)][str(c)] == 3
    # deterministic (temperature<=0) completion follows the corpus
    assert jm.sample_next(model, a, sec, rng=np.random.default_rng(0), temperature=0.0) == b
    assert jm.sample_next(model, b, sec, rng=np.random.default_rng(0), temperature=0.0) == c
    # learned next beats an unseen alternative
    assert jm.transition_prob(model, a, sec, b) > jm.transition_prob(model, a, sec, a)


def test_smoothing_gives_nonzero_prob_to_unseen():
    model, sec, (a, b, c) = _abc_model()
    # C never transitions anywhere in the corpus, A->A never observed
    assert jm.transition_prob(model, c, sec, a) > 0.0
    assert jm.transition_prob(model, a, sec, a) > 0.0
    # unseen section backs off to a uniform prior, still nonzero (smoothing)
    assert jm.prior_prob(model, "NO_SUCH_SECTION", a) == pytest.approx(1.0 / model["vocab_size"])
    assert jm.prior_prob(model, sec, a) > 0.0
    # unknown tuple ids resolve to nothing
    assert jm.transition_prob(model, a, sec, 9999) == 0.0
    # sampling from an unknown current tuple falls back instead of crashing
    got = jm.sample_next(model, 9999, sec, rng=np.random.default_rng(1))
    assert 0 <= got < model["vocab_size"]


def test_pruning_caps_size():
    tracks = []
    for i in range(30):
        bars = [
            {"drums": list(range((i % 15) + 1)), "bass": [2, 6, 10, 14],
             "other": [0, 8] if i % 2 else [0], "vocals": [0] if i % 3 else []},
            {"drums": [0, 4, 8, 12], "bass": list(range(i % 8)),
             "other": [0, 8], "vocals": [2, 6] if i % 2 else [0]},
        ]
        tracks.append(make_track(f"prune{i:02d}", bars))
    seqs = jm.mine_bar_tuples(tracks)
    model = jm.train_joint_transitions(seqs)
    assert model["vocab_size"] > 10
    compact = jm.to_top_compact(model, max_tuples=10, max_next=2)
    assert compact["vocab_size"] <= 10
    assert len(compact["tuples"]) <= 10
    for cur_table in compact["transitions"].values():
        for nxt_table in cur_table.values():
            assert len(nxt_table) <= 2
    for table in compact["priors"].values():
        assert len(table) <= 10
    blob = json.dumps(compact)
    assert len(blob) < 2_000_000
    # compact model still samples inside its vocab
    got = jm.sample_next(compact, 0, compact["sections"][0], rng=np.random.default_rng(0))
    assert 0 <= got < compact["vocab_size"]


# ---------------------------------------------------------------------------
# originality guard + temperature
# ---------------------------------------------------------------------------

def test_copy_guard_refuses_long_runs():
    assert jm.exceeds_copy_run([7, 7, 7], 7, 3) is True
    assert jm.exceeds_copy_run([7, 7, 7], 7, 4) is False
    assert jm.exceeds_copy_run([7, 7, 7], 5, 3) is False
    assert jm.exceeds_copy_run([1, 2, 2], 2, 2) is True
    assert jm.exceeds_copy_run([1, 2, 2], 1, 2) is False
    assert jm.exceeds_copy_run([], 9, 2) is False


def test_sample_next_copy_guard_redirects():
    tracks = [make_track(f"loop{i}", [BAR_A, BAR_A, BAR_A, BAR_C]) for i in range(3)]
    seqs = jm.mine_bar_tuples(tracks)
    sec = seqs[0][0][0]
    model = jm.train_joint_transitions(seqs)
    a = jm._resolve_id(model, seqs[0][0][1])
    c = jm._resolve_id(model, seqs[0][3][1])
    # corpus loops A->A, so unconstrained argmax stays on A ...
    plain = jm.sample_next(model, a, sec, rng=np.random.default_rng(0), temperature=0.0)
    assert plain == a
    # ... but with a 2-run history the guard refuses A and picks C
    guarded = jm.sample_next(
        model, a, sec, rng=np.random.default_rng(0), temperature=0.0,
        history=[a, a], max_copy_run=2,
    )
    assert guarded == c


def test_sample_next_temperature():
    tracks = [make_track(f"dom{i}", [BAR_X, BAR_Y]) for i in range(10)]
    tracks += [make_track("rare0", [BAR_X, BAR_Z])]
    seqs = jm.mine_bar_tuples(tracks)
    sec = seqs[0][0][0]
    model = jm.train_joint_transitions(seqs, k=0.5)
    x = jm._resolve_id(model, seqs[0][0][1])
    y = jm._resolve_id(model, seqs[0][1][1])

    rng_cold = np.random.default_rng(0)
    cold = [jm.sample_next(model, x, sec, rng=rng_cold, temperature=0.3)
            for _ in range(200)]
    rng_hot = np.random.default_rng(0)
    hot = [jm.sample_next(model, x, sec, rng=rng_hot, temperature=3.0)
           for _ in range(200)]
    cold_frac = sum(1 for g in cold if g == y) / len(cold)
    hot_frac = sum(1 for g in hot if g == y) / len(hot)
    assert cold_frac >= 0.9
    assert hot_frac < 0.9
    assert cold_frac > hot_frac


def test_save_load_roundtrip(tmp_path):
    model, _sec, _ids = _abc_model()
    p = tmp_path / "joint.json"
    jm.save_model(model, p)
    assert p.exists()
    back = jm.load_model(p)
    assert back["tuples"] == model["tuples"]
    assert back["tuple_counts"] == model["tuple_counts"]
    assert back["transitions"] == model["transitions"]
    assert back["priors"] == model["priors"]
    got = jm.sample_next(back, 0, back["sections"][0], rng=np.random.default_rng(0))
    assert 0 <= got < back["vocab_size"]
    compact = jm.to_top_compact(back)
    assert len(json.dumps(compact).encode()) < 2_000_000
