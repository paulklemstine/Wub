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


def test_standalone_patterns_manifest_style_kits():
    """Verify standalone_patterns.json has valid style kits and multi-genre pattern banks."""
    patterns_json = Path("frontend/public/data/standalone_patterns.json")
    assert patterns_json.exists()

    with open(patterns_json) as f:
        data = json.load(f)

    assert "style_kits" in data
    assert "patterns_by_style" in data
    assert "slices" in data

    for s in STYLES:
        assert s in data["style_kits"]
        kit = data["style_kits"][s]
        assert "stems" in kit
        assert "drums" in kit["stems"]
        assert "bass" in kit["stems"]
        assert "other" in kit["stems"]
        assert "vocals" in kit["stems"]

        # Check bass sub has 12 chromatic pitches
        assert len(kit["stems"]["bass"]["sub"]) == 12

    # Check patterns_by_style has all styles
    for s in STYLES:
        assert s in data["patterns_by_style"]
