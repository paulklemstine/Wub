"""Unit tests for real section detection + full-arc mining (section_arcs)."""
import json

import numpy as np

from wub_core.section_arcs import (
    SectionArcModel,
    SectionSpan,
    bar_features,
    mine_arcs,
    mine_section_arcs_from_processed,
    segment_sections,
)


def _one_hot(pitch: int) -> list:
    v = [0.0] * 12
    v[pitch % 12] = 1.0
    return v


def _make_synth_track(track_id, sections):
    """Build raw processed-JSON-style (segments, tatums) for section specs.

    sections: list of dicts with keys:
      n_bars, drums (list of onset steps, or per-bar list of such lists),
      loud (float or per-bar list of floats), pitch (int).
    Every bar also gets one "other" onset at step 0 (the mix grid).
    """
    segments = []
    tatums = []
    t_idx = 0
    for steps, loud, pitch in _section_bar_specs(sections):
        drum_ids = {}
        for st in steps:
            sid = f"{track_id}_d_{t_idx + st}"
            drum_ids[st] = sid
            segments.append({
                "id": sid, "track_id": track_id, "stem": "drums",
                "start": float((t_idx + st) * 0.125),
                "duration": 0.20, "is_silent": False,
                "loudness_start": loud - 6.0, "loudness_max": loud,
                "loudness_max_time": 0.01,
                "pitches": _one_hot(0), "timbre": [1.0] + [0.0] * 11,
                "spectral_centroid": 800.0,
                "zero_crossing_start_offset": 0.0,
            })
        other_sid = f"{track_id}_o_{t_idx}"
        segments.append({
            "id": other_sid, "track_id": track_id, "stem": "other",
            "start": float(t_idx * 0.125),
            "duration": 0.40, "is_silent": False,
            "loudness_start": loud - 10.0, "loudness_max": loud - 4.0,
            "loudness_max_time": 0.01,
            "pitches": _one_hot(pitch), "timbre": [0.5] + [0.0] * 11,
            "spectral_centroid": 1200.0,
            "zero_crossing_start_offset": 0.0,
        })
        for st in range(16):
            tatums.append({
                "id": f"{track_id}_t_{t_idx + st}",
                "index": t_idx + st,
                "bar_position": t_idx // 16,
                "beat_position": st // 4,
                "tatum_position": st % 4,
                "start": float((t_idx + st) * 0.125),
                "duration": 0.125,
                "confidence": 0.5,
                "stem_segments": {
                    "drums": [drum_ids[st]] if st in drum_ids else [],
                    "bass": [],
                    "vocals": [],
                    "other": [other_sid] if st == 0 else [],
                },
            })
        t_idx += 16
    return segments, tatums


def _full_arc_sections():
    """6-section full arc: INTRO -> GROOVE_A -> BUILDUP -> DROP -> BREAKDOWN -> GROOVE_A."""
    return [
        {"n_bars": 4, "drums": [0], "loud": -26.0, "pitch": 0},            # INTRO
        {"n_bars": 4, "drums": [0, 4, 8, 12], "loud": -14.0, "pitch": 2},  # GROOVE_A
        {"n_bars": 6,  # BUILDUP: linear density + loudness ramp
         "drums": [[0, 8], [0, 4, 8], [0, 4, 8, 12], [0, 4, 6, 8, 12],
                   [0, 2, 4, 6, 8, 12], [0, 2, 4, 6, 8, 10, 12]],
         "loud": [-20.0, -18.0, -16.0, -14.0, -12.0, -10.0], "pitch": 4},
        {"n_bars": 8, "drums": [0, 2, 4, 6, 8, 10, 12, 14],                # DROP
         "loud": -8.0, "pitch": 7},
        {"n_bars": 4, "drums": [], "loud": -34.0, "pitch": 9},             # BREAKDOWN
        {"n_bars": 4, "drums": [0, 4, 8, 12], "loud": -14.0, "pitch": 5},  # GROOVE_A outro
    ]


def _section_bar_specs(sections):
    """Expand section specs to per-bar (drums, loud, pitch) honoring lists."""
    bars = []
    for sec in sections:
        for bi in range(sec["n_bars"]):
            drums = sec["drums"]
            drums = drums[bi] if isinstance(drums, list) and drums and isinstance(drums[0], list) else drums
            loud = sec["loud"]
            loud = loud[bi] if isinstance(loud, list) else loud
            bars.append((drums, loud, sec["pitch"]))
    return bars


def _short_sections():
    return [
        {"n_bars": 4, "drums": [0], "loud": -26.0, "pitch": 0},            # INTRO
        {"n_bars": 4, "drums": [0, 2, 4, 6, 8, 10, 12, 14],                # DROP
         "loud": -8.0, "pitch": 7},
        {"n_bars": 4, "drums": [], "loud": -34.0, "pitch": 9},             # BREAKDOWN
    ]


EXPECTED_ARC = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN", "GROOVE_A"]


def test_bar_features_density_energy_and_harmonic_change():
    segments, tatums = _make_synth_track("synth_big", _full_arc_sections())
    feats = bar_features(segments, tatums)
    assert len(feats) == 30

    intro = feats[0]
    assert intro["densities"]["drums"] == 1 / 16
    assert intro["energy"] < -20.0

    drop = feats[14]  # first DROP bar
    assert drop["densities"]["drums"] == 8 / 16
    assert drop["kick_downbeats"] is True
    assert drop["energy"] > -12.0
    # Harmonic rhythm: pitch changes 4 -> 7 at the BUILDUP|DROP boundary.
    assert drop["harmonic_change"] > 0.0
    assert feats[15]["harmonic_change"] == 0.0

    brk = feats[22]  # first BREAKDOWN bar: no drums onsets
    assert brk["densities"]["drums"] == 0.0
    assert brk["energy"] < -25.0


def test_segment_sections_detects_full_arc_in_order():
    segments, tatums = _make_synth_track("synth_big", _full_arc_sections())
    spans = segment_sections(bar_features(segments, tatums))
    labels = [s.label for s in spans]
    assert labels == EXPECTED_ARC
    # Required in-order subsequence.
    assert labels.index("INTRO") < labels.index("DROP") < labels.index("BREAKDOWN")
    # Section lengths honor the crafted layout.
    assert [s.n_bars for s in spans] == [4, 4, 6, 8, 4, 4]
    assert spans[0].start_bar == 0


def test_segment_sections_short_track_intro_drop_breakdown():
    segments, tatums = _make_synth_track("synth_small", _short_sections())
    spans = segment_sections(bar_features(segments, tatums))
    assert [s.label for s in spans] == ["INTRO", "DROP", "BREAKDOWN"]


def test_mine_arcs_counts_full_arcs_and_durations():
    big1 = [SectionSpan(l, st, n) for (l, n), st in
            zip([("INTRO", 4), ("GROOVE_A", 4), ("BUILDUP", 6),
                 ("DROP", 8), ("BREAKDOWN", 4), ("GROOVE_A", 4)],
                [0, 4, 8, 14, 22, 26])]
    big2 = [SectionSpan(s.label, s.start_bar, s.n_bars) for s in big1]
    short = [SectionSpan("INTRO", 0, 4), SectionSpan("DROP", 4, 4),
             SectionSpan("BREAKDOWN", 8, 4)]

    model = mine_arcs([big1, big2, short])
    assert model.total_tracks == 3
    # Only the two 6-section seqs contribute to arc counts.
    assert len(model.arcs) == 1
    assert model.arcs[0]["sections"] == EXPECTED_ARC
    assert model.arcs[0]["count"] == 2
    # Every span (including the short track) contributes durations.
    assert model.durations["INTRO"]["mean"] == 4.0
    assert model.durations["INTRO"]["count"] == 3
    assert model.durations["DROP"]["mean"] == (8 + 8 + 4) / 3
    assert model.durations["DROP"]["count"] == 3

    # Empty /degenerate inputs never crash.
    empty = mine_arcs([])
    assert empty.arcs == [] and empty.durations == {} and empty.total_tracks == 0
    assert segment_sections([]) == []


def test_end_to_end_mine_from_processed_dir(tmp_path):
    for tid in ("arc_a", "arc_b"):
        segments, tatums = _make_synth_track(tid, _full_arc_sections())
        doc = {
            "track_id": tid, "filename": f"{tid}.wav", "bpm": 126.0,
            "key": "C major", "duration": 60.0, "sample_rate": 22050,
            "stems_available": ["drums", "other"],
            "tatums": tatums, "segments": segments,
        }
        (tmp_path / f"{tid}.json").write_text(json.dumps(doc))
    model = mine_section_arcs_from_processed(tmp_path)
    assert model.total_tracks == 2
    assert model.arcs[0]["sections"] == EXPECTED_ARC
    assert model.arcs[0]["count"] == 2
    assert model.durations["DROP"]["mean"] == 8.0
    # JSON-serializable payload round-trips through the model.
    payload = model.to_dict()
    json.dumps(payload)
    assert SectionArcModel.from_dict(payload).to_dict() == payload
    assert len(payload["arcs"]) <= 50
