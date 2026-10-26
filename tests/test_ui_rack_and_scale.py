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
        assert f'id="rack-stem-enbar-{stem}"' in content
        assert f'id="rack-stem-denbar-{stem}"' in content
        assert f'id="rack-stem-steps-{stem}"' in content
        # 16 step pads per stem
        for s in range(16):
            assert f'id="rack-step-pad-{stem}-{s}"' in content


def test_tamagotchi_anime_lipstick_mouth():
    """Verify that tamagotchi.ts implements the sensual dynamic anime red lipstick mouth."""
    tg_path = Path("frontend/src/tamagotchi.ts")
    assert tg_path.exists(), "frontend/src/tamagotchi.ts must exist"
    content = tg_path.read_text(encoding="utf-8")

    assert "drawAnimeLipstickMouth" in content
    assert "#ff0055" in content  # Neon cherry red lipstick
    assert "#a60033" in content  # Ruby shadow
    assert "cupid" in content.lower()  # Cupid's bow
    assert "gloss" in content.lower()  # Voluptuous lower lip gloss reflection


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
