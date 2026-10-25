"""Unit tests for Groove Dynamics Learner.

Tests micro-timing extraction, 4-stem coupling matrices, acoustic sidechain ducking,
and JSON schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from wub_core.groove_dynamics_learner import (
    GrooveDynamicsLearner,
    STYLES,
    STEMS,
    classify_track_style,
    get_style_sidechain_params,
    get_style_microtiming_offset,
    get_style_swing_ratio,
)


def test_classify_track_style():
    """Verify keyword and tempo style classification."""
    assert classify_track_style("CHVRN - Faded.mp3", 140.0) == "wave"
    assert classify_track_style("Boris Brejcha - Game Over.mp3", 126.0) == "cyber"
    assert classify_track_style("Booka Shade - Sacred.mp3", 123.0) == "house"
    assert classify_track_style("Alvaro - Shades.mp3", 129.0) == "breaks"
    # BPM fallbacks
    assert classify_track_style("unknown track", 140.0) == "cyber"
    assert classify_track_style("unknown track", 121.0) == "house"
    assert classify_track_style("unknown track", 110.0) == "wave"
    assert classify_track_style("unknown track", 128.0) == "breaks"


def test_helper_accessors():
    """Verify helper accessors return valid values for all styles."""
    for style in STYLES:
        depth, tau = get_style_sidechain_params(style)
        assert -20.0 <= depth <= -2.0
        assert 5.0 <= tau <= 120.0

        swing = get_style_swing_ratio(style)
        assert 0.45 <= swing <= 0.65

        for step in range(16):
            offset = get_style_microtiming_offset(style, step)
            assert isinstance(offset, float)
            assert -30.0 <= offset <= 30.0


def test_learner_initialization():
    """Verify learner initialization and defaults."""
    learner = GrooveDynamicsLearner()
    assert learner.processed_dir == Path("data/processed")
    assert learner.stems_dir == Path("data/stems")
    assert learner.sr == 44100


def test_microtiming_synthetic(tmp_path: Path):
    """Test micro-timing calculation against synthetic known offsets."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    bpm = 120.0  # 120 bpm -> t_step = 0.125s (125ms), bar = 2.0s
    t_step = 60.0 / bpm / 4.0
    known_swing_ms = 12.0  # +12ms on odd steps

    segments = []
    for bar in range(4):
        for step in range(16):
            expected_t = bar * 2.0 + step * t_step
            offset_sec = (known_swing_ms / 1000.0) if (step % 2 == 1) else 0.0
            actual_t = expected_t + offset_sec
            segments.append({
                "id": f"seg_{bar}_{step}",
                "stem": "drums",
                "start": actual_t,
                "duration": 0.1,
                "is_silent": False,
                "loudness_max": -6.0,
            })

    track_data = {
        "track_id": "synthetic_01",
        "filename": "Booka Shade - Test.mp3",  # style house
        "bpm": bpm,
        "key": "C major",
        "duration": 8.0,
        "sample_rate": 44100,
        "segments": segments,
        "tatums": [],
    }

    with open(processed_dir / "synthetic_01.json", "w") as f:
        json.dump(track_data, f)

    learner = GrooveDynamicsLearner(processed_dir=processed_dir, stems_dir=tmp_path / "stems")
    res = learner.learn_microtiming()

    house_mt = res["styles"]["house"]
    odd_offsets = [house_mt["step_offsets_ms"][s] for s in [1, 3, 5, 7, 9, 11, 13, 15]]
    even_offsets = [house_mt["step_offsets_ms"][s] for s in [0, 2, 4, 6, 8, 10, 12, 14]]

    assert np.allclose(odd_offsets, known_swing_ms, atol=0.5)
    assert np.allclose(even_offsets, 0.0, atol=0.5)
    assert house_mt["swing_offset_ms"] > 10.0


def test_inter_stem_coupling_synthetic(tmp_path: Path):
    """Test 4-stem co-occurrence and call-and-response coupling matrices."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    bpm = 120.0
    tatums = []
    # Create 32 tatums (2 bars)
    # Rhythmic rule: When vocals active (tatums 0..7), synths ('other') are silent.
    # When vocals silent (tatums 8..15), synths ('other') are active (call-and-response!).
    for idx in range(32):
        step = idx % 16
        vocal_act = (step < 8)
        synth_act = not vocal_act
        tatums.append({
            "id": f"tat_{idx}",
            "index": idx,
            "stem_segments": {
                "drums": ["seg_d"],
                "bass": ["seg_b"],
                "other": ["seg_o"] if synth_act else [],
                "vocals": ["seg_v"] if vocal_act else [],
            },
        })

    segments = [
        {"id": "seg_d", "stem": "drums", "start": 0.0, "duration": 0.5, "is_silent": False, "loudness_max": -6.0},
        {"id": "seg_b", "stem": "bass", "start": 0.0, "duration": 0.5, "is_silent": False, "loudness_max": -6.0},
        {"id": "seg_o", "stem": "other", "start": 0.0, "duration": 0.5, "is_silent": False, "loudness_max": -6.0},
        {"id": "seg_v", "stem": "vocals", "start": 0.0, "duration": 0.5, "is_silent": False, "loudness_max": -6.0},
    ]

    track_data = {
        "track_id": "coupling_01",
        "filename": "Boris Brejcha - Test.mp3",  # style cyber
        "bpm": bpm,
        "key": "A minor",
        "duration": 4.0,
        "sample_rate": 44100,
        "stems_available": ["drums", "bass", "other", "vocals"],
        "segments": segments,
        "tatums": tatums,
    }

    with open(processed_dir / "coupling_01.json", "w") as f:
        json.dump(track_data, f)

    learner = GrooveDynamicsLearner(processed_dir=processed_dir, stems_dir=tmp_path / "stems")
    res = learner.learn_inter_stem_coupling()

    cyber_cp = res["styles"]["cyber"]
    c_idx = cyber_cp["coupling_index_matrix"]
    # index 3 is vocals, index 2 is other/synths
    # P(other active | vocals silent) = 1.0, P(other active | vocals active) = 0.0
    # Coupling index = 1.0 - 0.0 = +1.0
    assert c_idx[3][2] > 0.8
    assert cyber_cp["vocal_pause_synth_lift"] > 5.0


def test_sidechain_ducking_synthetic(tmp_path: Path):
    """Test kick transient detection and bass attenuation measurement on synthetic waveforms."""
    stems_dir = tmp_path / "stems" / "track_01"
    stems_dir.mkdir(parents=True)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    sr = 44100
    dur_sec = 2.0
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)

    # Synthetic drums: 2 kicks at t=0.5s and t=1.5s (60 Hz decaying sine)
    drums = np.zeros_like(t)
    for kick_t in (0.5, 1.5):
        k_idx = int(kick_t * sr)
        k_len = int(0.15 * sr)
        tk = np.linspace(0, 0.15, k_len, endpoint=False)
        drums[k_idx : k_idx + k_len] = np.sin(2 * np.pi * 60 * tk) * np.exp(-30 * tk)

    # Synthetic bass: constant 80 Hz sine, ducked by -12 dB (factor 0.25) during kicks with 60ms exponential recovery
    bass = 0.8 * np.sin(2 * np.pi * 80 * t)
    for kick_t in (0.5, 1.5):
        k_idx = int(kick_t * sr)
        rec_len = int(0.15 * sr)
        tk = np.linspace(0, 0.15, rec_len, endpoint=False)
        # ducking envelope
        duck_env = 1.0 - 0.75 * np.exp(-tk / 0.060)
        bass[k_idx : k_idx + rec_len] *= duck_env

    import soundfile as sf
    sf.write(str(stems_dir / "drums.wav"), drums.astype(np.float32), sr)
    sf.write(str(stems_dir / "bass.wav"), bass.astype(np.float32), sr)

    track_data = {
        "track_id": "track_01",
        "filename": "Booka Shade - Sidechain.mp3",
        "bpm": 120.0,
        "key": "C major",
        "duration": 2.0,
        "sample_rate": sr,
        "stems_available": ["drums", "bass", "other", "vocals"],
        "segments": [],
        "tatums": [],
    }
    with open(processed_dir / "track_01.json", "w") as f:
        json.dump(track_data, f)

    learner = GrooveDynamicsLearner(processed_dir=processed_dir, stems_dir=tmp_path / "stems", sr=sr)
    res = learner.learn_sidechain_ducking(duration_per_track=2.0)

    house_sc = res["styles"]["house"]
    assert house_sc["sample_count"] >= 1
    # Measured depth should be around -10 to -14 dB
    assert -16.0 <= house_sc["attenuation_depth_db"] <= -8.0
    # Measured recovery tau should be around 40 to 80 ms
    assert 30.0 <= house_sc["recovery_time_ms"] <= 90.0


def test_exported_json_schema():
    """Verify frontend/public/data/groove_dynamics.json matches required schema and has valid ranges."""
    json_path = Path("frontend/public/data/groove_dynamics.json")
    assert json_path.exists(), "groove_dynamics.json was not created"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == 1
    assert "description" in data
    assert set(data["styles"]) == set(STYLES)
    assert set(data["stems"]) == set(STEMS)

    # 1. Micro-timing checks
    assert "microtiming" in data
    mt = data["microtiming"]
    assert "styles" in mt and "global" in mt
    for st in STYLES:
        st_mt = mt["styles"][st]
        assert len(st_mt["step_offsets_ms"]) == 16
        assert len(st_mt["step_jitter_ms"]) == 16
        assert len(st_mt["step_velocity"]) == 16
        assert 0.40 <= st_mt["swing_ratio"] <= 0.70
        for stem in STEMS:
            assert stem in st_mt["stems"]
            assert len(st_mt["stems"][stem]["step_offsets_ms"]) == 16

    # 2. Inter-stem coupling checks
    assert "inter_stem_coupling" in data
    cp = data["inter_stem_coupling"]
    for st in STYLES:
        st_cp = cp["styles"][st]
        assert len(st_cp["co_occurrence_matrix"]) == 4
        assert len(st_cp["co_occurrence_matrix"][0]) == 4
        assert len(st_cp["call_and_response_matrix"]) == 4
        assert len(st_cp["coupling_index_matrix"]) == 4
        assert len(st_cp["correlation_matrix"]) == 4
        assert len(st_cp["step_occupancy"]) == 16
        assert len(st_cp["step_occupancy"][0]) == 4
        assert len(st_cp["step_onset_density"]) == 16
        assert isinstance(st_cp["vocal_pause_synth_lift"], (int, float))
        assert isinstance(st_cp["kick_bass_syncopation_ratio"], (int, float))

    # 3. Sidechain ducking checks
    assert "sidechain_ducking" in data
    sc = data["sidechain_ducking"]
    for st in STYLES:
        st_sc = sc["styles"][st]
        assert -25.0 <= st_sc["attenuation_depth_db"] <= -1.0
        assert 1.0 <= st_sc["recovery_time_ms"] <= 200.0
        assert len(st_sc["recovery_curve"]) == 16
        assert st_sc["sample_count"] > 0
