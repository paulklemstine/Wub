"""Tests for the compact harmony + transition models on synthetic tracks."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from wub_core.harmonic_compat import (
    build_harmonic_tables,
    compat_score,
    load as load_harmonic,
    to_compact as harmonic_compact,
)
from wub_core.schema import Segment, Tatum, TrackAnalysis
from wub_core.transition_ops import (
    build_transition_tables,
    energy_delta,
    load as load_trans,
    to_compact as trans_compact,
    transition_template,
)


def _chroma(pc: int, sharp: float = 1.0, floor: float = 0.02) -> List[float]:
    v = [floor] * 12
    v[pc % 12] = sharp
    return v


def _two_hot(pcs: List[int]) -> List[float]:
    v = [0.02] * 12
    for p in pcs:
        v[p % 12] = 1.0
    return v


def _seg(tid: str, stem: str, sid: str, start: float, dur: float,
         pitches: List[float], energy: float, centroid: float = 800.0) -> Segment:
    loud = -60.0 + 60.0 * energy
    return Segment(id=sid, track_id=tid, stem=stem, start=start, duration=dur,
                   is_silent=False, loudness_start=loud, loudness_max=loud,
                   loudness_max_time=0.0, pitches=pitches,
                   timbre=[0.0] * 12, spectral_centroid=centroid,
                   zero_crossing_start_offset=0.0)


def make_track(tid: str, key: str, bpm: float, bars: List[Dict[str, Any]]) -> TrackAnalysis:
    """One segment per pitched stem per bar; drums get one segment per onset run."""
    td = 60.0 / bpm / 4.0
    segments: List[Segment] = []
    tatums: List[Tatum] = []
    for b, spec in enumerate(bars):
        e = float(spec.get("energy", 0.5))
        bstart = b * 16 * td
        bar_ids: Dict[str, str] = {}
        for stem, pc, cent in (("bass", spec.get("bass_pc", 0), 200.0),
                               ("other", None, 1200.0),
                               ("vocals", spec.get("vocal_pc", 7), 1500.0)):
            sid = f"{tid}-{stem}-b{b}"
            if stem == "other":
                pitches = _two_hot(list(spec.get("other_pcs", [4, 7])))
            else:
                pitches = _chroma(int(pc if pc is not None else 0))
            segments.append(_seg(tid, stem, sid, bstart, 16 * td, pitches, e, cent))
            bar_ids[stem] = sid
        onsets = sorted(set(int(s) for s in spec.get("drum_onsets", [0, 4, 8, 12])))
        drum_runs: List[tuple] = []
        for i, s in enumerate(onsets):
            nxt = onsets[i + 1] if i + 1 < len(onsets) else 16
            drum_runs.append((s, nxt))
        for s, nxt in drum_runs:
            sid = f"{tid}-drums-b{b}-s{s}"
            segments.append(_seg(tid, "drums", sid, bstart + s * td,
                                 max(td, (nxt - s) * td),
                                 [0.1] * 12, e, 3000.0))
        for s in range(16):
            idx = b * 16 + s
            if onsets:
                cur = max(o for o in onsets if o <= s)
                did = f"{tid}-drums-b{b}-s{cur}"
            else:
                did = None
            stem_segs = {"bass": [bar_ids["bass"]], "other": [bar_ids["other"]],
                         "vocals": [bar_ids["vocals"]],
                         "drums": [did] if did else []}
            tatums.append(Tatum(id=f"{tid}-t{idx}", index=idx, bar_position=b,
                                beat_position=s // 4, tatum_position=s % 4,
                                start=bstart + s * td, duration=td,
                                confidence=0.9, stem_segments=stem_segs))
    return TrackAnalysis(track_id=tid, filename=f"{tid}.mp3", bpm=bpm, key=key,
                         duration=len(bars) * 16 * td, sample_rate=44100,
                         stems_available=["drums", "bass", "vocals", "other"],
                         tatums=tatums, segments=segments)


def harmony_tracks() -> List[TrackAnalysis]:
    out = []
    for t in range(3):
        bars = [{"energy": 0.5, "bass_pc": 0, "other_pcs": [4, 7], "vocal_pc": 7,
                 "drum_onsets": [0, 4, 8, 12]} for _ in range(8)]
        out.append(make_track(f"harm{t}", "C major", 126.0, bars))
    return out


def test_compat_consonant_beats_clash() -> None:
    tables = build_harmonic_tables(harmony_tracks())
    assert tables["total_bars"] == 24
    # Groove co-occurrences were mined too.
    assert any(tables["groove_pairs"].values())

    compact = harmonic_compact(tables)
    blob = json.dumps(compact)
    assert len(blob) < 1_000_000
    tables2 = load_harmonic(json.loads(blob))
    tables3 = load_harmonic(blob)  # JSON-string input also works

    cons = dict(bass=_chroma(0), other=_two_hot([4, 7]), vocal=0)
    clash = dict(bass=_chroma(1), other=_chroma(6), vocal=1)
    for tb in (tables2, tables3):
        s_cons = compat_score(cons["bass"], cons["other"], cons["vocal"],
                              0, False, "GROOVE_A", tb)
        s_clash = compat_score(clash["bass"], clash["other"], clash["vocal"],
                               0, False, "GROOVE_A", tb)
        assert 0.0 <= s_clash <= 1.0 and 0.0 <= s_cons <= 1.0
        assert s_cons > s_clash + 0.15, (s_cons, s_clash)
    # Unseen context still yields a valid score via fallback.
    s_fb = compat_score(cons["bass"], cons["other"], cons["vocal"],
                        9, True, "BREAKDOWN", tables2)
    assert 0.0 <= s_fb <= 1.0


def test_transition_mines_planted_fill() -> None:
    bars = ([{"energy": 0.2, "bass_pc": 0, "other_pcs": [4, 7], "vocal_pc": 0,
              "drum_onsets": [0, 8]} for _ in range(3)]
            + [{"energy": 0.2, "bass_pc": 0, "other_pcs": [4, 7], "vocal_pc": 0,
                "drum_onsets": list(range(16))}]  # planted fill, last bar before
            + [{"energy": 0.85, "bass_pc": 0, "other_pcs": [4, 7], "vocal_pc": 7,
                "drum_onsets": [0, 2, 4, 6, 8, 10, 12, 14]} for _ in range(4)])
    tracks = [make_track(f"tr{t}", "C major", 128.0, bars) for t in range(2)]
    labels = {f"tr{t}": ["BUILDUP"] * 4 + ["DROP"] * 4 for t in range(2)}

    tables = build_transition_tables(tracks, bar_sections=labels)
    assert "BUILDUP->DROP" in tables["transitions"]
    assert tables["transitions"]["BUILDUP->DROP"]["count"] == 2

    tpl = transition_template("BUILDUP", "DROP", tables)
    assert set(("pre_curve", "post_curve", "fill_grid", "entry_grid")) <= set(tpl)
    assert len(tpl["pre_curve"]) == 2 and len(tpl["post_curve"]) == 2
    assert float(np.mean(tpl["post_curve"])) > float(np.mean(tpl["pre_curve"]))
    # Planted fill: dense second half on the bar before the change.
    assert sum(tpl["fill_grid"][8:]) >= 6
    # Driving entry: downbeats firing on the first bar after.
    for step in (0, 4, 8, 12):
        assert tpl["entry_grid"][step] == 1
    assert energy_delta("BUILDUP", "DROP", tables) > 0.0
    assert tables["transitions"]["BUILDUP->DROP"]["groove_changes"]["drums"]

    compact = trans_compact(tables)
    blob = json.dumps(compact)
    assert len(blob) < 1_000_000
    tables2 = load_trans(json.loads(blob))
    tpl2 = transition_template("BUILDUP", "DROP", tables2)
    assert sum(tpl2["fill_grid"][8:]) >= 6
    assert energy_delta("BUILDUP", "DROP", tables2) > 0.0


def test_transition_energy_jump_inference_without_labels() -> None:
    bars = ([{"energy": 0.15, "bass_pc": 0, "other_pcs": [0], "vocal_pc": 0,
              "drum_onsets": [0]} for _ in range(4)]
            + [{"energy": 0.85, "bass_pc": 5, "other_pcs": [5, 9], "vocal_pc": 5,
                "drum_onsets": [0, 2, 4, 6, 8, 10, 12, 14]} for _ in range(4)])
    tracks = [make_track("jump0", "F major", 124.0, bars)]
    tables = build_transition_tables(tracks)  # no labels: energy-jump path
    assert tables["total_boundaries"] >= 1
    # Whatever the inferred names, some boundary rises into high energy.
    rising = [k for k in tables["transitions"]
              if energy_delta(k.split("->")[0], k.split("->")[1], tables) > 0.15]
    assert rising, list(tables["transitions"])
    tpl = transition_template(*rising[0].split("->"), tables)
    assert float(np.mean(tpl["post_curve"])) > float(np.mean(tpl["pre_curve"]))


def test_unknown_transition_falls_back_gracefully() -> None:
    tables = build_transition_tables(harmony_tracks())
    tpl = transition_template("BREAKDOWN", "DROP", tables)
    assert set(("pre_curve", "post_curve", "fill_grid", "entry_grid")) <= set(tpl)
    assert len(tpl["fill_grid"]) == 16 and len(tpl["entry_grid"]) == 16
    d = energy_delta("BUILDUP", "DROP", tables)
    assert isinstance(d, float) and d > 0  # prior-backed rise into the drop
