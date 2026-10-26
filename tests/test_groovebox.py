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


def test_multistem_song_phrase_cohesion():
    """Verify that MultiStemPatterns and SongPhrases lock all 4 stems to the same track."""
    tr = _make_dance_track("test_tr_cohesive", bpm=126.0, n_bars=8)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    assert len(bank.multistem_patterns) == 8
    assert len(bank.phrases) == 1

    phrase = bank.phrases[0]
    assert phrase.track_id == "test_tr_cohesive"
    assert len(phrase.bars) == 8

    # All bars must belong to the exact same track
    for b in phrase.bars:
        assert b.track_id == "test_tr_cohesive"
        assert "drums" in b.stems
        assert "bass" in b.stems
        assert "other" in b.stems
        assert "vocals" in b.stems
        assert len(b.stems["drums"]) == 16

    # Test playback with GrooveboxConductor
    conductor = GrooveboxConductor(bank, seed=42)
    assert conductor.current_phrase is not None
    assert conductor.current_phrase.track_id == "test_tr_cohesive"

    # Step through Bar 1
    slices_0, meta_0 = conductor.next_step()
    assert meta_0["phrase_bar"] == 1
    assert meta_0["step_in_bar"] == 1
    # On Step 0 (beat 1), drums and other stems are triggered
    assert slices_0["drums"] is not None
    assert slices_0["drums"].track_id == "test_tr_cohesive"


def test_user_feedback_and_cross_stem_reroll():
    tr1 = _make_dance_track("test_tr_a", bpm=128.0, n_bars=8)
    tr2 = _make_dance_track("test_tr_b", bpm=128.0, n_bars=8)
    bank = PatternBank()
    bank.load_from_tracks([tr1, tr2])

    conductor = GrooveboxConductor(bank, seed=42, cross_stem_mashup=True)
    assert conductor.active_patterns["drums"] is not None

    # Test Thumbs Up
    fb_up = conductor.record_feedback(stem="drums", rating=1, track_id="test_tr_a")
    assert fb_up["status"] == "ok"
    assert fb_up["action"] == "boosted"
    assert conductor.stem_weights["drums"].get("test_tr_a", 1.0) > 1.0

    # Test Thumbs Down and Immediate Reroll
    fb_down = conductor.record_feedback(stem="drums", rating=-1, track_id="test_tr_b")
    assert fb_down["status"] == "ok"
    assert fb_down["action"] == "rerolled"
    assert conductor.stem_weights["drums"].get("test_tr_b", 1.0) < 1.0

    # Next step meta includes pattern_ratings and pattern_track_ids
    _, meta = conductor.next_step()
    assert "pattern_track_ids" in meta
    assert "pattern_ratings" in meta
    assert "drums" in meta["pattern_ratings"]


def test_dynamic_cycle_modes_and_32_step_compound():
    """Verify selectable cycle modes (4x16, 8x16, 8x32, 16x32) and 32-step grid progression."""
    tr = _make_dance_track("test_tr_dyn", bpm=126.0, n_bars=16)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    conductor = GrooveboxConductor(bank, seed=42)

    # 1. Test 4x16 Mode (4 patterns, 16 steps each = 64 total tatums)
    assert conductor.set_cycle_mode("4x16") is True
    assert conductor.patterns_per_cycle == 4
    assert conductor.steps_per_pattern == 16
    assert conductor.cycle_total_steps == 64

    patterns_seen = set()
    fill_at_4 = False
    for step_i in range(64):
        slices, meta = conductor.next_step()
        patterns_seen.add(meta["pattern_in_cycle"])
        assert meta["cycle_mode"] == "4x16"
        assert meta["patterns_per_cycle"] == 4
        assert meta["steps_per_pattern"] == 16
        assert len(meta["pattern_grids"]["drums"]) == 16
        if meta["is_fill"] and meta["pattern_in_cycle"] == 4:
            fill_at_4 = True

    assert patterns_seen == {1, 2, 3, 4}
    assert fill_at_4 is True

    # 2. Test 8x32 Mode (8 patterns, 32 steps each = 256 total tatums)
    assert conductor.set_cycle_mode("8x32") is True
    assert conductor.patterns_per_cycle == 8
    assert conductor.steps_per_pattern == 32
    assert conductor.cycle_total_steps == 256

    slices_32, meta_32 = conductor.next_step()
    assert meta_32["cycle_mode"] == "8x32"
    assert meta_32["steps_per_pattern"] == 32
    # Verify 32-step grid length
    assert len(meta_32["pattern_grids"]["drums"]) == 32

    # 3. Test 16x32 Mode (16 patterns, 32 steps each = 512 total tatums)
    assert conductor.set_cycle_mode("16x32") is True
    assert conductor.patterns_per_cycle == 16
    assert conductor.steps_per_pattern == 32
    assert conductor.cycle_total_steps == 512

    # 4. Invalid cycle mode fails safely
    assert conductor.set_cycle_mode("invalid_mode") is False


def test_dense_pattern_infill_and_nonstop_beats():
    """Verify that sparse patterns are automatically infilled to ensure continuous driving beats."""
    tr = _make_dance_track("test_tr_dense", bpm=128.0, n_bars=8)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    conductor = GrooveboxConductor(bank, seed=42)

    # 1. Verify that active drum patterns have high onset density (16 active steps)
    active_drums = conductor.active_patterns["drums"]
    assert active_drums is not None
    assert len(active_drums.onset_grid) == 16
    assert active_drums.density == 1.0

    # 2. Verify that active bass patterns have rich pocket density (at least 8 onsets)
    active_bass = conductor.active_patterns["bass"]
    assert active_bass is not None
    assert len(active_bass.onset_grid) >= 8

    # 3. Verify step sequencer trigger: drums fire non-stop across full 16-step bar
    drum_slices_fired = 0
    for step_i in range(16):
        slices, meta = conductor.next_step()
        if slices["drums"] is not None:
            drum_slices_fired += 1

    # Every step of the bar has a driving drum slice!
    assert drum_slices_fired == 16


def test_empirical_pattern_velocities_and_dynamics():
    """Verify that PatternBank loads patterns with empirical velocity curves from real song loudness."""
    tr = _make_dance_track("test_tr_vels", bpm=126.0, n_bars=4)
    bank = PatternBank()
    n_pats = bank.load_from_tracks([tr])
    assert n_pats > 0

    # 1. MultiStemPattern has per-stem velocity lists
    msp = bank.multistem_patterns[0]
    assert hasattr(msp, "velocities")
    assert "drums" in msp.velocities
    assert len(msp.velocities["drums"]) == 16
    assert all(0.0 <= v <= 1.0 for v in msp.velocities["drums"])

    # 2. SequencerPattern has velocities curve
    for stem in ["drums", "bass", "other", "vocals"]:
        pats = bank.patterns_by_stem.get(stem, [])
        assert len(pats) > 0
        p = pats[0]
        assert p.velocities is not None
        assert len(p.velocities) == 16
        # Active steps have non-zero velocity
        for step in p.onset_grid:
            assert p.velocities[step] > 0.0


def test_statistical_groove_miner_and_mining_from_tracks():
    """Verify statistical groove fitting, empirical velocity/microtiming learning, and pattern mining."""
    from wub_core.groove_miner import StatisticalGrooveModel, mine_patterns_from_tracks

    tr1 = _make_dance_track("tr_mine_1", bpm=128.0, n_bars=8)
    tr2 = _make_dance_track("tr_mine_2", bpm=128.0, n_bars=8)

    model = StatisticalGrooveModel()
    n_bars = model.fit_from_tracks([tr1, tr2])
    assert n_bars > 0

    # Empirical velocities and microtimings are populated
    assert "drums" in model.empirical_velocities
    assert len(model.empirical_velocities["drums"]) > 0
    assert "drums" in model.empirical_microtiming

    # Mine top patterns from tracks
    mined = mine_patterns_from_tracks([tr1, tr2], min_recurrence=1, top_k=4)
    assert "drums" in mined
    assert len(mined["drums"]) > 0
    assert mined["drums"][0].velocities is not None


def test_stem_adaptive_onset_detection():
    """Verify stem-adaptive onset detection handles drums, bass, vocals, and other distinctly."""
    from wub_core.analyzer import detect_onsets

    sr = 22050
    # 1-second synthetic impulse signals
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Bass impulse (low sine chirp)
    y_bass = np.sin(2 * np.pi * 60 * t) * np.exp(-10 * (t % 0.25))
    # Drums click impulse
    y_drums = np.zeros(sr)
    for hit in [0, int(0.25 * sr), int(0.5 * sr), int(0.75 * sr)]:
        y_drums[hit : hit + 200] = 1.0

    onsets_drums = detect_onsets(y_drums, sr, stem="drums")
    onsets_bass = detect_onsets(y_bass, sr, stem="bass")
    assert isinstance(onsets_drums, np.ndarray)
    assert isinstance(onsets_bass, np.ndarray)
    assert len(onsets_drums) > 0
    assert len(onsets_bass) > 0

