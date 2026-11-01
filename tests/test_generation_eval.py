"""Unit tests for wub_core.generation_eval (synthetic inputs only)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from wub_core.generation_eval import (
    bar_signature,
    build_corpus_stats,
    energy_curve_distance,
    evaluate_mix,
    groove_discontinuity,
    harmonic_collision_rate,
    originality_score,
    stem_interlock,
)
from wub_core.schema import Segment, Tatum, TrackAnalysis


# --------------------------------------------------------------------------
# Synthetic bar builder
# --------------------------------------------------------------------------

def _vel_grid(onsets, level=0.9) -> List[float]:
    v = [0.0] * 16
    for s in onsets:
        v[s % 16] = level
    return v


def _bar(
    drums=(0, 4, 8, 12),
    pitched_onsets=(0,),
    pitched_pcs=(0,),
    chord=(0, 4, 7),
    section: str = "DROP",
    bar_id=None,
    vel: float = 0.9,
) -> Dict[str, Any]:
    grids = {
        "drums": list(drums),
        "bass": list(pitched_onsets),
        "other": list(pitched_onsets),
        "vocals": list(pitched_onsets),
    }
    pitches = {
        "bass": list(pitched_pcs),
        "other": list(pitched_pcs),
        "vocals": list(pitched_pcs),
    }
    velocities = {stem: _vel_grid(g, vel) for stem, g in grids.items()}
    bar: Dict[str, Any] = {
        "section": section,
        "onset_grids": grids,
        "pitches": pitches,
        "chord": list(chord),
        "velocities": velocities,
    }
    if bar_id is not None:
        bar["bar_id"] = bar_id
    return bar


def _perfect_bars(n=4, **kw) -> List[Dict[str, Any]]:
    return [_bar(**kw) for _ in range(n)]


# --------------------------------------------------------------------------
# harmonic_collision_rate
# --------------------------------------------------------------------------

def test_harmonic_no_collision_for_consonant_bars():
    bars = _perfect_bars(4, pitched_pcs=(0, 4, 7), chord=(0, 4, 7))
    assert harmonic_collision_rate(bars) == pytest.approx(0.0)


def test_harmonic_collision_minor_second_and_major_seventh():
    # C# (1) is a minor 2nd above C (0); B (11) is a major 7th above C.
    assert harmonic_collision_rate(_perfect_bars(3, pitched_pcs=(1,), chord=(0, 4, 7))) == pytest.approx(1.0)
    assert harmonic_collision_rate(_perfect_bars(3, pitched_pcs=(11,), chord=(0, 4, 7))) == pytest.approx(1.0)


def test_harmonic_partial_collision_fraction():
    bars = [
        _bar(pitched_pcs=(0,), chord=(0, 4, 7)),
        _bar(pitched_pcs=(0,), chord=(0, 4, 7)),
        _bar(pitched_pcs=(1,), chord=(0, 4, 7)),
        _bar(pitched_pcs=(1,), chord=(0, 4, 7)),
    ]
    assert harmonic_collision_rate(bars) == pytest.approx(0.5)


def test_harmonic_empty_or_chordless_bars_score_zero():
    assert harmonic_collision_rate([]) == pytest.approx(0.0)
    bars = [{"section": "DROP", "onset_grids": {"drums": [0, 4, 8, 12]}}]
    assert harmonic_collision_rate(bars) == pytest.approx(0.0)
    # Chord present but no pitched notes -> clean, not a collision.
    bars = [{"section": "DROP",
             "onset_grids": {"drums": [0, 4, 8, 12]},
             "chord": [0, 4, 7]}]
    assert harmonic_collision_rate(bars) == pytest.approx(0.0)


def test_harmonic_chord_root_and_key_forms():
    bars = [_bar(pitched_pcs=(0, 4, 7))]
    for b in bars:
        del b["chord"]
        b["chord_root"] = 0
        b["chord_quality"] = "major"
    assert harmonic_collision_rate(bars) == pytest.approx(0.0)
    bars2 = _perfect_bars(2, pitched_pcs=(0, 4, 7))
    for b in bars2:
        del b["chord"]
        b["key"] = "C major"
    assert harmonic_collision_rate(bars2) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# groove_discontinuity
# --------------------------------------------------------------------------

def test_groove_perfect_continuity_scores_zero():
    bars = _perfect_bars(4, drums=(0, 4, 8, 12))
    assert groove_discontinuity(bars) == pytest.approx(0.0)


def test_groove_discontinuity_penalizes_breaks_and_missing_downbeats():
    bars = [
        _bar(drums=(0, 4, 8, 12)),
        _bar(drums=(1, 3, 5, 7)),  # disjoint + all downbeats missing
        _bar(drums=(0, 4, 8, 12)),
        _bar(drums=(1, 3, 5, 7)),
    ]
    assert groove_discontinuity(bars) > 0.5


def test_groove_single_or_empty_scores_zero():
    assert groove_discontinuity([]) == pytest.approx(0.0)
    assert groove_discontinuity(_perfect_bars(1)) == pytest.approx(0.0)


def test_groove_small_variation_scores_between():
    steady = groove_discontinuity(_perfect_bars(3, drums=(0, 4, 8, 12)))
    varied = groove_discontinuity([
        _bar(drums=(0, 4, 8, 12)),
        _bar(drums=(0, 4, 8, 12, 2, 6)),
        _bar(drums=(0, 4, 8, 12)),
    ])
    assert steady == pytest.approx(0.0)
    assert 0.0 < varied < 0.5


# --------------------------------------------------------------------------
# energy_curve_distance
# --------------------------------------------------------------------------

def _flat_bar(vel: float, section: str = "DROP") -> Dict[str, Any]:
    """Bar with an explicit flat 16-step velocity curve (no stem averaging)."""
    return {
        "section": section,
        "onset_grids": {"drums": [0, 4, 8, 12]},
        "velocities": [vel] * 16,
        "chord": [0, 4, 7],
        "pitches": {"bass": [0]},
    }


def test_energy_curve_match_scores_near_zero():
    corpus = {"DROP": [0.9] * 16}
    bars = [_flat_bar(0.9) for _ in range(4)]
    assert energy_curve_distance(bars, corpus) == pytest.approx(0.0, abs=1e-9)


def test_energy_curve_mismatch_scores_badly():
    corpus = {"DROP": [0.9] * 16}
    bars = [_flat_bar(0.1) for _ in range(4)]
    mse = energy_curve_distance(bars, corpus)
    assert mse == pytest.approx(0.64, abs=1e-9)
    assert mse > 0.5


def test_energy_curve_dict_grouped_input_and_stats_unwrap():
    stats = build_corpus_stats(_perfect_bars(4, vel=0.7))
    assert "DROP" in stats["section_curves"]
    grouped = {"DROP": _perfect_bars(2, vel=0.7)}
    assert energy_curve_distance(grouped, stats) == pytest.approx(0.0, abs=1e-9)
    assert energy_curve_distance([], stats) == pytest.approx(0.0)
    assert energy_curve_distance(grouped, {}) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# originality_score
# --------------------------------------------------------------------------

def test_originality_exact_copy_scores_zero():
    gen = ["a", "b", "c", "d"]
    corpus = {"a", "b", "c", "d", "x"}
    assert originality_score(gen, corpus) == pytest.approx(0.0)


def test_originality_novel_scores_one():
    assert originality_score(["n1", "n2"], {"a", "b"}) == pytest.approx(1.0)
    assert originality_score(["a"], set()) == pytest.approx(1.0)
    assert originality_score([], {"a"}) == pytest.approx(1.0)


def test_originality_partial_run():
    gen = ["a", "b", "n1", "n2"]
    assert originality_score(gen, {"a", "b"}) == pytest.approx(0.5)
    # Non-contiguous hits only give a run of 1.
    assert originality_score(["a", "x", "b"], {"a", "b"}) == pytest.approx(1.0 - 1.0 / 3.0)


# --------------------------------------------------------------------------
# build_corpus_stats
# --------------------------------------------------------------------------

def _make_track(track_id: str, n_bars: int = 2, seed: int = 0) -> TrackAnalysis:
    rng = np.random.default_rng(seed)
    stems = ["drums", "bass", "vocals", "other"]
    tatums: List[Tatum] = []
    segments: List[Segment] = []
    total = n_bars * 16
    for idx in range(total):
        bar = idx // 16
        beat = (idx % 16) // 4
        tpos = idx % 4
        tat_segs: Dict[str, List[str]] = {}
        for stem in stems:
            seg_id = f"seg_{stem}_{idx:04d}"
            is_silent = False
            if stem == "drums":
                is_silent = not (tpos == 0)
            elif stem == "bass":
                is_silent = not (tpos == 2)
            pitches = [0.0] * 12
            pitches[(idx + len(stem)) % 12] = 1.0
            segments.append(Segment(
                id=seg_id, track_id=track_id, stem=stem,
                start=idx * 0.12, duration=0.12, is_silent=is_silent,
                loudness_start=-20.0,
                loudness_max=-6.0 if not is_silent else -60.0,
                loudness_max_time=0.01, pitches=pitches,
                timbre=list(rng.normal(size=12)), spectral_centroid=1200.0,
                zero_crossing_start_offset=0.0,
            ))
            tat_segs[stem] = [seg_id]
        tatums.append(Tatum(
            id=f"tatum_{idx:04d}", index=idx, bar_position=bar,
            beat_position=beat, tatum_position=tpos,
            start=idx * 0.12, duration=0.12, confidence=0.8,
            stem_segments=tat_segs,
        ))
    return TrackAnalysis(
        track_id=track_id, filename=f"{track_id}.mp3", bpm=128.0,
        key="C major", duration=60.0, sample_rate=44100,
        stems_available=["drums", "bass", "vocals", "other"],
        tatums=tatums, segments=segments,
    )


def test_build_corpus_stats_from_tracks():
    tracks = [_make_track("t1", 2, 0), _make_track("t2", 2, 1)]
    stats = build_corpus_stats(tracks)
    assert stats["n_tracks"] == 2
    assert stats["n_bars"] == 4
    assert set(stats["section_curves"]) <= {"INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"}
    for curve in stats["section_curves"].values():
        assert len(curve) == 16
        assert all(0.0 <= v <= 1.0 for v in curve)
    assert len(stats["bar_ids"]) > 0
    cooc = np.asarray(stats["pitch_cooccurrence"])
    assert cooc.shape == (12, 12)
    assert cooc.sum() > 0
    assert sum(stats["section_bar_counts"].values()) == 4


def test_build_corpus_stats_from_symbolic_bars():
    stats = build_corpus_stats(_perfect_bars(3, vel=0.8))
    assert stats["n_bars"] == 3
    assert "DROP" in stats["section_curves"]
    assert energy_curve_distance(_perfect_bars(2, vel=0.8), stats) == pytest.approx(0.0, abs=1e-9)


def test_bar_signature_stable_and_distinct():
    a = _bar(drums=(0, 4, 8, 12))
    b = _bar(drums=(0, 4, 8, 12))
    c = _bar(drums=(0, 4, 8, 14))
    for d in (a, b, c):
        d.pop("bar_id", None)
    assert bar_signature(a) == bar_signature(b)
    assert bar_signature(a) != bar_signature(c)


# --------------------------------------------------------------------------
# evaluate_mix
# --------------------------------------------------------------------------

def test_evaluate_mix_perfect_scores_near_one():
    corpus = build_corpus_stats(_perfect_bars(4, vel=0.9, drums=(0, 2, 4, 6, 8, 10, 12, 14)))
    gen = [
        _bar(bar_id=f"fresh_{i}", vel=0.9, pitched_pcs=(0, 4, 7),
             chord=(0, 4, 7), drums=(0, 4, 8, 12))
        for i in range(4)
    ]
    report = evaluate_mix(gen, corpus)
    assert report["harmonic_collision_rate"] == pytest.approx(0.0)
    assert report["groove_discontinuity"] == pytest.approx(0.0)
    assert report["originality_score"] == pytest.approx(1.0)
    assert report["goodness"] == pytest.approx(1.0, abs=0.05)
    for key in ("n_bars", "energy_curve_distance", "energy_score",
                "longest_copy_run", "goodness"):
        assert key in report


def test_evaluate_mix_bad_scores_worse_than_perfect():
    corpus_ids = {f"stolen_{i}" for i in range(8)}
    corpus = {"section_curves": {"DROP": [0.9] * 16}, "bar_ids": corpus_ids}
    good = evaluate_mix(
        [_bar(bar_id=f"fresh_{i}", pitched_pcs=(0, 4, 7)) for i in range(4)],
        corpus,
    )
    bad = evaluate_mix(
        [
            _bar(bar_id=f"stolen_{i}", pitched_pcs=(1,), chord=(0, 4, 7),
                 drums=(1,) if i % 2 else (3,))
            for i in range(4)
        ],
        corpus,
    )
    assert bad["harmonic_collision_rate"] == pytest.approx(1.0)
    assert bad["originality_score"] == pytest.approx(0.0)
    assert bad["goodness"] < 0.5
    assert bad["goodness"] < good["goodness"]
    assert good["goodness"] > 0.9


def _mkbar(drums, bass):
    return {
        "bar_id": "t",
        "section": "GROOVE_A",
        "onset_grids": {
            "drums": drums, "bass": bass, "other": [0, 4, 8, 12], "vocals": [],
        },
        "velocities": {s: [0.85] * 16 for s in ("drums", "bass", "other", "vocals")},
        "chord": {"root": 0, "tones": [0, 4, 7]},
        "pitches": {"bass": [0], "other": [0, 4], "vocals": [7]},
    }


def test_interlock_locked_scores_high():
    bars = [_mkbar([0, 4, 8, 12], [0, 4, 8, 12]) for _ in range(4)]
    r = stem_interlock(bars)
    assert r["rate"] == 1.0
    assert r["score"] == 1.0


def test_interlock_random_scores_low_vs_corpus():
    import random
    rng = random.Random(11)
    locked = [_mkbar([0, 4, 8, 12], [0, 4, 8, 12]) for _ in range(16)]
    stats = {"interlock_mean": stem_interlock(locked)["rate"], "interlock_std": 0.05}
    loose = [_mkbar([0, 4, 8, 12], sorted(rng.sample(range(16), 5))) for _ in range(16)]
    assert stem_interlock(loose, stats)["score"] < stem_interlock(locked, stats)["score"]
    assert stem_interlock(locked, stats)["score"] > 0.9


def test_interlock_skips_bassless():
    bars = [_mkbar([0, 4, 8, 12], []) for _ in range(3)]
    assert stem_interlock(bars) == {"rate": 0.0, "score": 0.0}


def test_evaluate_mix_has_interlock_fields():
    bars = [_mkbar([0, 4, 8, 12], [0, 8]) for _ in range(4)]
    rep = evaluate_mix(bars, {"interlock_mean": 0.5, "interlock_std": 0.2})
    assert "interlock_rate" in rep and "interlock_score" in rep
    assert 0.0 <= rep["goodness"] <= 1.0
