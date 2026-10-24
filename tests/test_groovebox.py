"""Unit tests for the Wub Groovebox engine: PatternBank and GrooveboxConductor."""
from __future__ import annotations

import numpy as np
import pytest

from wub_core.groovebox import GrooveboxConductor, SECTIONS
from wub_core.patterns import PatternBank, classify_drum_groove, classify_bass_groove
from wub_core.schema import Segment, Tatum, TrackAnalysis


def _make_dance_track(track_id: str, bpm: float = 128.0, n_bars: int = 4) -> TrackAnalysis:
    """Create a synthetic 4-stem dance track with 16 tatums per bar."""
    stems = ["drums", "bass", "vocals", "other"]
    tatums = []
    segments = []
    rng = np.random.default_rng(42)

    total_tatums = n_bars * 16

    for bar in range(n_bars):
        for beat in range(4):
            for tpos in range(4):
                idx = bar * 16 + beat * 4 + tpos
                tat_segs = {}
                for stem in stems:
                    seg_id = f"seg_{stem}_{idx:04d}"
                    # Simulate standard four-on-the-floor drum pattern
                    is_silent = False
                    if stem == "drums":
                        is_silent = not (tpos == 0)  # Onset on every beat
                    elif stem == "bass":
                        is_silent = not (tpos == 2)  # Offbeat bass

                    seg = Segment(
                        id=seg_id,
                        track_id=track_id,
                        stem=stem,
                        start=idx * 0.12,
                        duration=0.12,
                        is_silent=is_silent,
                        loudness_start=-20.0,
                        loudness_max=-6.0 if not is_silent else -60.0,
                        loudness_max_time=0.01,
                        pitches=list(rng.random(12)),
                        timbre=list(rng.normal(size=12)),
                        spectral_centroid=1200.0,
                        zero_crossing_start_offset=0.0,
                    )
                    segments.append(seg)
                    tat_segs[stem] = [seg_id]

                tatums.append(
                    Tatum(
                        id=f"tatum_{idx:04d}",
                        index=idx,
                        bar_position=bar,
                        beat_position=beat,
                        tatum_position=tpos,
                        start=idx * 0.12,
                        duration=0.12,
                        confidence=0.95,
                        stem_segments=tat_segs,
                    )
                )

    return TrackAnalysis(
        track_id=track_id,
        filename=f"{track_id}.mp3",
        bpm=bpm,
        key="C major",
        duration=total_tatums * 0.12,
        sample_rate=44100,
        stems_available=stems,
        tatums=tatums,
        segments=segments,
    )


def test_pattern_groove_classification():
    # Four on floor has onsets on 0, 4, 8, 12
    four_floor = [0, 4, 8, 12]
    assert classify_drum_groove(four_floor, 16) == "four_on_floor"

    # Fill has high density (e.g. 14 out of 16 steps active)
    fill = list(range(14))
    assert classify_drum_groove(fill, 16) == "fill"

    # Offbeat bass hits on 2, 6, 10, 14
    offbeat_bass = [2, 6, 10, 14]
    assert classify_bass_groove(offbeat_bass, 16) == "offbeat"

    # Rolling bass hits on many 16th steps
    rolling_bass = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert classify_bass_groove(rolling_bass, 16) == "rolling"


def test_pattern_bank_extraction():
    tr = _make_dance_track("test_tr_01", bpm=126.0, n_bars=4)
    bank = PatternBank()
    count = bank.load_from_tracks([tr])
    assert count > 0

    drum_pats = bank.get_patterns("drums")
    assert len(drum_pats) > 0
    assert drum_pats[0].length == 16
    assert drum_pats[0].groove_type == "four_on_floor"

    bass_pats = bank.get_patterns("bass")
    assert len(bass_pats) > 0
    assert bass_pats[0].groove_type == "offbeat"


def test_groovebox_conductor_steps_and_turnarounds():
    tr = _make_dance_track("test_tr_01", bpm=128.0, n_bars=8)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    conductor = GrooveboxConductor(bank, seed=42)
    assert conductor.state.section in SECTIONS

    # Step through 128 steps (8 full bars = 1 complete phrase)
    bars_seen = set()
    fill_seen = False

    for _ in range(128):
        slices, meta = conductor.next_step()
        bars_seen.add(meta["phrase_bar"])
        if meta["is_fill"]:
            fill_seen = True

        assert "drums" in slices
        assert "bass" in slices
        assert 1 <= meta["phrase_bar"] <= 8
        assert 1 <= meta["step_in_bar"] <= 16

    # Should have traversed all 8 bars of the phrase
    assert bars_seen == {1, 2, 3, 4, 5, 6, 7, 8}
    # Turnaround fill should have triggered on Bar 8
    assert fill_seen is True


def test_groovebox_palette_and_pattern_synthesis():
    tr = _make_dance_track("test_tr_02", bpm=128.0, n_bars=8)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    assert len(bank.slice_palette["drums"]["kick"]) > 0
    assert len(bank.slice_palette["bass"]["sub"]) > 0

    # Test synthesizing a high-energy drum fill pattern
    fill_pat = bank.create_groove_pattern("drums", "fill")
    assert fill_pat is not None
    assert fill_pat.length == 16
    assert fill_pat.groove_type == "fill"
    # Ensure active steps in the fill
    assert len(fill_pat.onset_grid) >= 4

    # Test synthesizing an offbeat bass pattern
    bass_pat = bank.create_groove_pattern("bass", "offbeat")
    assert bass_pat is not None
    assert bass_pat.length == 16
    assert set(bass_pat.onset_grid).intersection({2, 6, 10, 14})
