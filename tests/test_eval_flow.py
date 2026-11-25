"""Tests for the Phase 0 flow evaluator (scripts/eval_flow.py).

The evaluator is the instrument that decides whether a transition change is an
improvement, so it must be deterministic and must rank musical structure above
shuffle.  No corpus is needed: the reference is synthesised in the test.
"""

import numpy as np

from scripts.eval_flow import (
    ARC_POINTS,
    BASE_SECTIONS,
    MOTIF_DIM,
    dtw_distance,
    base_section,
    build_reference,
    demo_trace,
    energy01,
    lag_profile,
    motif_matrix,
    motif_vector,
    ngram_counts,
    ngram_logprob,
    resample,
    score_trace,
    section_runs,
    self_test,
    source_stats,
    standardize,
    trace_matrix,
)


def _ref(trace):
    """A minimal reference built from a song-like demo trace."""
    profile = lag_profile(trace_matrix(trace["bars"]))
    return {
        "vocab": list(BASE_SECTIONS),
        "ngram": {
            "2": ngram_counts([["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"]], 2),
            "3": ngram_counts([["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"]], 3),
        },
        "duration_bars": {label: {"p10": 4, "p50": 16, "p90": 64} for label in BASE_SECTIONS},
        "lag_profile": profile,
        "motif_lag_profile": lag_profile(motif_matrix(trace["bars"])),
        "energy_envelope": {
            "mean": resample([b["energy"] for b in trace["bars"]]).tolist(),
            "std": [0.0] * ARC_POINTS,
            "points": ARC_POINTS,
        },
        "energy_arcs": [resample([b["energy"] for b in trace["bars"]]).tolist()],
        "metrics": {
            "form_ll": [-2.5, -2.0, -1.5],
            "structure_corr": [0.4, 0.6, 0.85],
            "structure_rmse": [0.04, 0.08, 0.16],
            "energy_dtw": [0.01, 0.04, 0.09],
        },
    }


def test_self_test_passes():
    checks = self_test()
    assert len(checks) >= 10


def test_section_label_mapping():
    assert base_section("GROOVE_A") == "GROOVE_A"
    assert base_section("GROOVE_HIGH") == "GROOVE_A"
    assert base_section("RISE") == "BUILDUP"
    assert base_section("PEAK") == "DROP"
    assert base_section("BRIDGE") == "BREAKDOWN"
    assert base_section("VERSE_2") == "GROOVE_A"
    assert base_section(None) == "GROOVE_A"


def test_energy_normalises_db_and_unit():
    assert energy01(-60.0) == 0.0
    assert abs(energy01(-30.0) - 0.5) < 1e-6
    assert energy01(0.72) == 0.72
    assert energy01(3.0) == 1.0


def test_standardize_zero_variance_columns_stay_zero():
    matrix = np.asarray([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    out = standardize(matrix)
    assert np.all(out[:, 1] == 0.0)
    assert abs(float(out[:, 0].mean())) < 1e-9


def test_ngram_logprob_prefers_seen_sequence():
    tables = {
        2: ngram_counts([["INTRO", "GROOVE_A", "DROP"]], 2),
        3: ngram_counts([["INTRO", "GROOVE_A", "DROP"]], 3),
    }
    seen = ngram_logprob(["INTRO", "GROOVE_A", "DROP"], tables, list(BASE_SECTIONS))
    unseen = ngram_logprob(["DROP", "BREAKDOWN", "DROP"], tables, list(BASE_SECTIONS))
    assert seen > unseen


def test_dtw_is_zero_for_identical_and_positive_otherwise():
    a = np.linspace(0.0, 1.0, 16)
    assert dtw_distance(a, a) < 1e-9
    # Same shape sampled at a different rate: near zero, not exactly zero.
    b = np.linspace(0.0, 1.0, 24)
    assert dtw_distance(a, b) < 0.02
    assert dtw_distance(a, 1.0 - b) > 0.1


def test_lag_profile_detects_loop_period():
    # A 4-bar cycle repeated: self-similar at lag 4 and 8, not at lag 1.
    rows = []
    for i in range(48):
        value = [0.2, 0.5, 0.8, 0.4][i % 4]
        rows.append({
            "energy": 0.5 + 0.1 * value,
            "section": "DROP",
            "stems": {
                "drums": {"phrase": f"p{i % 4}", "track": "t", "active": True,
                          "density": value},
                "bass": {"phrase": f"p{i % 4}", "track": "t", "active": True, "density": 0.3},
                "other": {"phrase": f"p{i % 4}", "track": "t", "active": True, "density": 0.2},
                "vocals": {"phrase": f"p{i % 4}", "track": "t", "active": True, "density": 0.1},
            },
        })
    profile = lag_profile(trace_matrix(rows))
    assert profile[4] > 0.99
    assert profile[8] > 0.99


def test_source_stats_measures_change_rate():
    bars = []
    for bar in range(32):
        source = "a" if bar < 16 else "b"
        bars.append({
            "energy": 0.6,
            "stems": {stem: {"phrase": f"{source}{stem}", "track": source,
                             "active": True, "density": 0.3} for stem in
                      ("drums", "bass", "other", "vocals")},
        })
    stats = source_stats(bars)
    assert stats["drums"]["changes"] == 1
    assert stats["drums"]["bars_per_change"] == 32.0
    assert stats["drums"]["distinct"] == 2


def test_demo_trace_songlike_beats_churn():
    song = demo_trace("song", bars=104, seed=11)
    churn = demo_trace("churn", bars=104, seed=11)
    ref = _ref(song)
    good = score_trace(song, ref)
    bad = score_trace(churn, ref)
    assert good["structure"]["corr"] > bad["structure"]["corr"]
    assert good["verdict"]["pass"] is True
    assert bad["verdict"]["pass"] is False
    assert any("source changes" in reason for reason in bad["verdict"]["reasons"])


def test_score_trace_reports_every_section():
    trace = demo_trace("song", bars=104, seed=3)
    report = score_trace(trace, _ref(trace))
    labels = [run["section"] for run in report["section_runs"]]
    assert labels == ["INTRO", "GROOVE_A", "DROP", "GROOVE_A", "BREAKDOWN",
                      "GROOVE_A", "BREAKDOWN", "GROOVE_A", "BREAKDOWN"]
    assert sum(run["bars"] for run in report["section_runs"]) == 104
    assert report["energy"]["envelope_rmse"] is not None


def test_trace_without_reference_still_scores():
    trace = demo_trace("song", bars=64, seed=5)
    report = score_trace(trace, None)
    assert report["verdict"]["pass"] is True  # only seam gates are active
    assert report["form"].get("ll") is None


def test_section_runs_merges_adjacent_labels():
    bars = [{"section": "DROP"} for _ in range(8)] + [{"section": "BREAKDOWN"}] * 4
    runs = section_runs(bars)
    assert runs == [("DROP", 8), ("BREAKDOWN", 4)]


def test_build_reference_requires_usable_corpus(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        build_reference(tmp_path, max_tracks=4)


def test_motif_vector_has_onsets_and_chroma():
    bar = {
        "energy": 0.7,
        "chroma": [0.1] * 12,
        "stems": {stem: {"onsets": [1 if i % 4 == 0 else 0 for i in range(16)]}
                  for stem in ("drums", "bass", "other", "vocals")},
    }
    vec = motif_vector(bar)
    assert vec.shape == (MOTIF_DIM,)
    assert vec[0] == 0.7
    assert vec[1] == 1.0 and vec[2] == 0.0
    assert vec[-1] == 0.1
    # Missing masks stay zero instead of inventing a fake motif.
    bare = motif_vector({"energy": 0.5})
    assert bare.shape == (MOTIF_DIM,)
    assert np.count_nonzero(bare[1:]) == 0


def test_motif_return_separates_groove_from_churn():
    song = demo_trace("song", 104, seed=19)
    churn = demo_trace("churn", 104, seed=19)
    good = lag_profile(motif_matrix(song["bars"]))
    bad = lag_profile(motif_matrix(churn["bars"]))
    assert good[8] > 0.2
    assert bad[8] < 0.1
    assert good[8] > bad[8]


def test_score_trace_reports_motif_gate():
    trace = demo_trace("song", 104, seed=23)
    report = score_trace(trace, _ref(trace))
    assert "corr" in report["motif"]
    assert report["motif"]["profile"]
