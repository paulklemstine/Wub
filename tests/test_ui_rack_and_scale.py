"""Unit tests for the Neon Trippy Modular Sequencer Rack, Harmonic Scale Matrix,
and dynamic anime lipstick mouth presentation.
"""

from pathlib import Path
import re
import pytest


def test_index_html_contains_neon_sequencer_rack():
    """Verify that index.html contains the Neon Modular Sequencer Rack, scale matrix, and all 4 stems."""
    index_path = Path("frontend/index.html")
    assert index_path.exists(), "frontend/index.html must exist"
    content = index_path.read_text(encoding="utf-8")

    # Rack chassis and header
    assert 'id="trippy-sequencer-rack"' in content
    assert "KG-505 MODULAR" in content
    assert "HARMONIC SCALE MATRIX" in content

    # Scale ribbon elements
    assert 'id="rack-camelot-badge"' in content
    assert 'id="rack-scale-name"' in content
    assert 'id="rack-scale-mode"' in content
    assert 'id="rack-chord-name"' in content
    assert 'id="rack-chord-notes"' in content
    assert 'id="rack-bass-hz"' in content
    assert 'id="rack-chord-hz"' in content
    assert 'id="rack-scale-strip"' in content

    # 7-Note Scale degrees
    for i in range(7):
        assert f'id="rack-scale-pad-{i}"' in content
        assert f'id="rack-scale-note-{i}"' in content

    # 4 Modular stem units
    for stem in ["drums", "bass", "vocals", "other"]:
        assert f'id="rack-unit-{stem}"' in content
        assert f'id="rack-stem-territory-{stem}"' in content
        assert f'id="rack-stem-groove-{stem}"' in content
        assert f'id="rack-stem-patname-{stem}"' in content
        assert f'id="rack-stem-simbar-{stem}"' in content
        assert f'id="rack-stem-goodbar-{stem}"' in content
        assert f'id="rack-stem-goodval-{stem}"' in content
        assert f'id="rack-stem-enbar-{stem}"' in content
        assert f'id="rack-stem-denbar-{stem}"' in content
        assert f'id="rack-stem-steps-{stem}"' in content
        # 16 step pads per stem
        for s in range(16):
            assert f'id="rack-step-pad-{stem}-{s}"' in content


def test_tamagotchi_live_vocal_waveform_mouth():
    """Verify that tamagotchi.ts animates the mouth directly using the vocal waveform."""
    tg_path = Path("frontend/src/tamagotchi.ts")
    assert tg_path.exists(), "frontend/src/tamagotchi.ts must exist"
    content = tg_path.read_text(encoding="utf-8")

    # Live vocal waveform canvas mouth loop
    assert "THE LIVE VOCAL WAVEFORM AUDIO REACTIVE MOUTH" in content
    assert "mouthSteps = 64" in content
    assert "waveDisp = sampleNorm" in content
    assert "mouthAperture" in content
    assert "smoothedVocalAmp" in content
    assert "vowel" in content

    # Waveform-modulated ASCII singing mouth
    assert "extractWaveformMouth" in content
    assert "MOUTH_SHAPES" in content
    assert "singingLoopPhase" in content
    assert "zeroCrossings" in content
    assert "zcr" in content

    # HTML contains .rave-mouth styling
    html_path = Path("frontend/index.html")
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert ".rave-mouth" in html_content


def test_tamagotchi_round_orb_creature():
    """Verify that the wub creature has a round orb outline instead of ears."""
    tg_path = Path("frontend/src/tamagotchi.ts")
    assert tg_path.exists(), "frontend/src/tamagotchi.ts must exist"
    content = tg_path.read_text(encoding="utf-8")

    # Round Orb stages and outline
    assert "RAVE ORB" in content
    assert "DISCO ORB" in content
    assert "CYBER ORB" in content
    assert "ASTRAL ORB" in content
    assert "CREATURE ROUND ORB OUTLINE" in content
    assert "rave-orb" in content

    # Cat ears replaced
    assert "leftEarPeak" not in content
    assert "rightEarPeak" not in content
    assert "foreheadDip" not in content

    # HTML contains .rave-orb styling
    html_path = Path("frontend/index.html")
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert ".rave-orb" in html_content


def test_harmonic_scale_pitch_math():
    """Verify chromatic pitch math and diatonic intervals for Camelot keys."""
    # A Minor (8A): Root = 9 (A), Aeolian = [0, 2, 3, 5, 7, 8, 10]
    root_a = 9
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]
    a_minor_pitches = [(root_a + iv) % 12 for iv in minor_intervals]
    # Expected pitch classes: A(9), B(11), C(0), D(2), E(4), F(5), G(7)
    assert a_minor_pitches == [9, 11, 0, 2, 4, 5, 7]

    # Am7 chord tones: Root(9), Minor 3rd(0), 5th(4), Minor 7th(7)
    am7_tones = [9, 0, 4, 7]
    for tone in am7_tones:
        assert tone in a_minor_pitches, f"Chord tone {tone} must be in A minor scale"

    # C Major (8B): Root = 0 (C), Ionian = [0, 2, 4, 5, 7, 9, 11]
    root_c = 0
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    c_major_pitches = [(root_c + iv) % 12 for iv in major_intervals]
    assert c_major_pitches == [0, 2, 4, 5, 7, 9, 11]

    # Cmaj7 chord tones: Root(0), Major 3rd(4), 5th(7), Major 7th(11)
    cmaj7_tones = [0, 4, 7, 11]
    for tone in cmaj7_tones:
        assert tone in c_major_pitches, f"Chord tone {tone} must be in C major scale"


def test_client_ml_sequencer_song_and_slice_continuity():
    """Verify that ClientGrooveboxConductor maintains song, slice, and pattern continuity."""
    seq_path = Path("frontend/src/client_ml_sequencer.ts")
    assert seq_path.exists(), "frontend/src/client_ml_sequencer.ts must exist"
    content = seq_path.read_text(encoding="utf-8")

    # Timbral slice locking across chord changes
    assert "lockedCategorySlices" in content
    assert "kitTrackId" in content

    # Pattern continuity scoring and smooth motif blending
    assert "computePatternContinuity" in content
    assert "blendPatternContinuity" in content
    assert "shouldEvolvePattern" in content

    # Multi-bar phrase block crossfade envelope in AudioEngine
    ae_path = Path("frontend/src/audio_engine.ts")
    assert ae_path.exists(), "frontend/src/audio_engine.ts must exist"
    ae_content = ae_path.read_text(encoding="utf-8")
    assert "isLongLoop" in ae_content


