"""Unit tests for 4-Style Acoustic Kits and Slice Harvesting."""

import json
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from scripts.harvest_style_kits import (
    clean_slice,
    classify_track_style,
    verify_audio_slices,
    STYLES,
    STYLE_METADATA,
)


def test_classify_track_style_keywords():
    """Verify keyword classification into distinct stylistic territories."""
    assert classify_track_style("CHVRN - Faded.mp3", 140.0) == "wave"
    assert classify_track_style("BITWVLF - To Dust.mp3", 120.0) == "wave"
    assert classify_track_style("Ambyion - New World.mp3", 136.0) == "wave"

    assert classify_track_style("Boris Brejcha - Game Over.mp3", 126.0) == "cyber"
    assert classify_track_style("Ann Clue - Acid Attack.mp3", 126.0) == "cyber"
    assert classify_track_style("Cushy - BMF.mp3", 130.0) == "cyber"

    assert classify_track_style("Booka Shade - Sacred.mp3", 123.0) == "house"
    assert classify_track_style("Amtrac - Identical.mp3", 126.0) == "house"
    assert classify_track_style("DAVI - In Deep.mp3", 126.0) == "house"

    assert classify_track_style("Alvaro - Shades.mp3", 129.0) == "breaks"
    assert classify_track_style("Aaren San - Apes From Space.mp3", 129.0) == "breaks"


def test_classify_track_style_bpm_fallbacks():
    """Verify BPM fallback classification."""
    assert classify_track_style("Unknown Track Fast.mp3", 140.0) == "cyber"
    assert classify_track_style("Unknown Track House.mp3", 122.0) == "house"
    assert classify_track_style("Unknown Track Slow.mp3", 100.0) == "wave"
    assert classify_track_style("Unknown Track Break.mp3", 130.0) == "breaks"


def test_clean_slice_dsp():
    """Test click-free micro-fades and peak normalization."""
    sr = 44100
    # Generate a loud sine wave with abrupt start and end (potential clicks)
    t = np.linspace(0, 0.25, int(sr * 0.25))
    raw = np.sin(2 * np.pi * 440.0 * t) * 1.5  # clipping level

    cleaned = clean_slice(raw, sr, peak_target=0.88, fade_in_ms=2.0, fade_out_ms=20.0)

    # 1. Peak normalization check
    peak = float(np.max(np.abs(cleaned)))
    assert peak <= 0.88 + 1e-4

    # 2. Click-free boundary check: first and last samples should be attenuated
    assert abs(cleaned[0]) < 0.005
    assert abs(cleaned[-1]) < 0.005

    # 3. Target duration truncation
    truncated = clean_slice(raw, sr, target_dur=0.10, peak_target=0.90)
    assert len(truncated) == int(0.10 * sr)


def test_slices_manifest_and_files():
    """Verify that slices directory contains valid, click-free audio files for all 4 styles."""
    slices_dir = Path("frontend/public/data/slices")
    assert slices_dir.exists()

    passed, failed = verify_audio_slices(slices_dir)
    assert failed == 0, f"{failed} slices failed audio verification"
    assert passed >= 200, f"Expected at least 200 valid slices, found {passed}"

    # Verify each style is represented in slice filenames
    for s in STYLES:
        matching = list(slices_dir.glob(f"*_{s}_*.wav"))
        assert len(matching) > 0, f"No slices found for style '{s}'"

        # Check chromatic bass sub coverage for each style
        for p in range(12):
            bass_sub = slices_dir / f"bass_sub_{s}_p{p}.wav"
            assert bass_sub.exists(), f"Missing chromatic slice {bass_sub.name}"

        # Check chromatic vocal coverage for chop, hook, and call across all styles
        for cat in ["chop", "hook", "call"]:
            for p in range(12):
                vocal_slice = slices_dir / f"vocals_{cat}_{s}_p{p}.wav"
                assert vocal_slice.exists(), f"Missing chromatic vocal slice {vocal_slice.name}"


def test_standalone_patterns_manifest_style_kits():
    """Verify standalone_patterns.json has valid style kits, multi-genre pattern banks, and vocal patterns."""
    patterns_json = Path("frontend/public/data/standalone_patterns.json")
    assert patterns_json.exists()

    with open(patterns_json) as f:
        data = json.load(f)

    assert "style_kits" in data
    assert "patterns_by_style" in data
    assert "slices" in data
    assert "vocal_patterns" in data
    assert "vocal_themes" in data

    # Check learned vocal patterns structure
    vp = data["vocal_patterns"]
    assert "themes" in vp
    assert len(vp["themes"]) == 4
    assert "corpus_stats" in vp
    assert "vowel_transitions" in vp["corpus_stats"]
    assert "pitch_contour_transitions" in vp["corpus_stats"]

    for s in STYLES:
        assert s in data["style_kits"]
        kit = data["style_kits"][s]
        assert "stems" in kit
        assert "drums" in kit["stems"]
        assert "bass" in kit["stems"]
        assert "other" in kit["stems"]
        assert "vocals" in kit["stems"]

        # Check chromatic pitches for bass sub
        assert len(kit["stems"]["bass"]["sub"]) == 12
        # Check chromatic pitches for vocals: chop, hook, call
        assert len(kit["stems"]["vocals"]["chop"]) == 12
        assert len(kit["stems"]["vocals"]["hook"]) == 12
        assert len(kit["stems"]["vocals"]["call"]) == 12

    # Check patterns_by_style has all styles
    for s in STYLES:
        assert s in data["patterns_by_style"]
        assert len(data["patterns_by_style"][s]["vocals"]) > 0


def test_vocal_patterns_and_weights_export():
    """Verify dedicated vocal_patterns.json and composing_ml_weights.json exports."""
    vocal_json = Path("frontend/public/data/vocal_patterns.json")
    assert vocal_json.exists()

    with open(vocal_json) as f:
        vd = json.load(f)

    assert "themes" in vd
    assert len(vd["themes"]) == 4
    assert "vowel_formants" in vd
    for v in ["AH", "EE", "OH", "OO", "AY"]:
        assert v in vd["vowel_formants"]

    weights_json = Path("frontend/public/data/composing_ml_weights.json")
    assert weights_json.exists()

    with open(weights_json) as wf:
        wd = json.load(wf)

    assert "vocal_vowel_transitions" in wd
    assert "vocal_pitch_transitions" in wd
    assert "vocal_step_onset_priors" in wd


def test_vocal_pattern_extractor_dsp():
    """Test vocal formant centroid classifier."""
    from wub_core.vocal_pattern_extractor import classify_vowel, choose_syllable

    assert classify_vowel(800.0) == "OO"
    assert classify_vowel(1500.0) == "OH"
    assert classify_vowel(2300.0) == "AH"
    assert classify_vowel(3200.0) == "AY"
    assert classify_vowel(4500.0) == "EE"

    syl = choose_syllable("wave", "EE", seed=0)
    assert isinstance(syl, str) and len(syl) > 0
