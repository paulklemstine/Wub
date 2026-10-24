"""Phase 1 tests: schema conformance + click-free stitching + DSP integration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_schema_matches_spec_example():
    from wub_core.schema import Segment, TrackAnalysis

    seg = Segment(
        id="seg_bass_0001", track_id="sha256_hash", stem="bass",
        start=0.452, duration=0.065, is_silent=False,
        loudness_start=-24.5, loudness_max=-6.2, loudness_max_time=0.012,
        pitches=[0.95, 0.12, 0.04, 0.01, 0.22, 0.05, 0.02, 0.45, 0.01, 0.03, 0.11, 0.08],
        timbre=[42.1, -12.4, 8.3, 1.2, -5.6, 3.1, -0.4, 2.2, -1.1, 0.8, -0.2, 0.5],
        spectral_centroid=450.5, zero_crossing_start_offset=0.0002,
    )
    assert len(seg.pitches) == 12 and len(seg.timbre) == 12


def test_equal_power_crossfade_is_click_free():
    from wub_core.audio_utils import boundary_click_metric, stitch_with_crossfade

    sr = 44100
    t = np.arange(sr // 4) / sr
    a = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    y = stitch_with_crossfade([a, b], sr, fade_ms=3.0)
    fade = int(sr * 0.003)
    assert len(y) == 2 * len(a) - fade
    m = boundary_click_metric(y, sr, [len(a) - fade])
    assert m < 8.0


def test_28d_vector_layout():
    from wub_core.indexer import DIM

    assert DIM == 28  # 12 timbre + 12 chroma + 1 loudness + 3 metrical


def _fake_track(track_id: str, stem: str = "other", n: int = 6) -> "TrackAnalysis":
    import hashlib

    from wub_core.schema import Segment, Tatum, TrackAnalysis

    stable = int(hashlib.sha256(track_id.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(stable)
    segs = [
        Segment(id=f"seg_{stem}_{i:04d}", track_id=track_id, stem=stem,
                start=i * 0.1, duration=0.1, is_silent=False,
                loudness_start=-20.0, loudness_max=-8.0, loudness_max_time=0.01,
                pitches=list(rng.random(12)), timbre=list(rng.normal(size=12)),
                spectral_centroid=800.0, zero_crossing_start_offset=0.0)
        for i in range(n)
    ]
    tat = Tatum(id="tatum_0000", index=0, bar_position=0, beat_position=0,
                tatum_position=0, start=0.0, duration=0.1, confidence=0.9,
                stem_segments={stem: [s.id for s in segs]})
    return TrackAnalysis(track_id=track_id, filename=f"{track_id}.mp3", bpm=128.0,
                         key="C major", duration=n * 0.1, sample_rate=44100,
                         stems_available=[stem], tatums=[tat], segments=segs)


def test_markov_chaos_zero_is_chronological():
    from wub_core import model_markov as mm

    pools = mm.build_pools([_fake_track("aa", n=12), _fake_track("bb", n=12)])
    seq = mm.generate(pools, steps=30, chaos={"other": 0.0}, seed=0)["other"]
    assert len(seq) == 30
    # chaos=0 means straight looping playback of a single track.
    assert {r.track_id for r in seq} == {"bb"} or {r.track_id for r in seq} == {"aa"}
    for a, b in zip(seq, seq[1:]):
        assert b.track_pos == (a.track_pos + 1) % 12


def test_markov_high_chaos_mixes_tracks():
    from wub_core import model_markov as mm

    pools = mm.build_pools([_fake_track("aa"), _fake_track("bb")])
    seq = mm.generate(pools, steps=60, chaos={"other": 0.9}, seed=3)
    assert len({r.track_id for r in seq["other"]}) == 2


def test_analyzer_end_to_end():
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    import warnings

    warnings.filterwarnings("ignore")
    from wub_core.analyzer import analyze_track
    from wub_core.indexer import build_indices
    from wub_core.schema import TrackAnalysis

    sr = 44100
    dur = 4.0
    N = int(sr * dur)
    t = np.arange(N) / sr
    beat_dur = 60 / 128.0
    drums = np.zeros(N)
    for b in np.arange(0, dur, beat_dur):
        i = int(b * sr)
        n = int(0.09 * sr)
        if i + n <= N:
            env = np.exp(-np.arange(n) / (0.012 * sr))
            drums[i:i + n] += env * np.sin(2 * np.pi * 55 * np.arange(n) / sr)
    bass = 0.4 * np.sign(np.sin(2 * np.pi * 46.25 * t)) * np.exp(-0.05 * t)
    other = 0.2 * np.sin(2 * np.pi * 220 * t)
    vocals = np.zeros(N)
    tmp = Path("/tmp/opencode/wub_pytest")
    sdir = tmp / "stems" / "t"
    sdir.mkdir(parents=True, exist_ok=True)
    for name, y in [("drums", drums), ("bass", bass), ("vocals", vocals), ("other", other)]:
        sf.write(str(sdir / (name + ".wav")), np.stack([y, y], axis=1).astype(np.float32), sr)
    mix = 0.25 * (drums + bass + other)
    sf.write(str(tmp / "mix.wav"), np.stack([mix, mix], axis=1).astype(np.float32), sr)

    out = tmp / "out.json"
    a = analyze_track(tmp / "mix.wav", sdir, out, track_id="t")
    assert 60 < a.bpm < 200
    assert len(a.tatums) > 8
    assert len(a.segments) > 4
    # every tatum references valid segment ids
    ids = {s.id for s in a.segments}
    for tat in a.tatums:
        for lst in tat.stem_segments.values():
            assert all(i in ids for i in lst)
    # JSON round-trips through strict schema
    rt = TrackAnalysis(**json.loads(out.read_text()))
    assert rt.track_id == "t"
    paths = build_indices([rt], tmp / "idx")
    assert paths, "expected at least one FAISS index"
