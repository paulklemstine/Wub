"""Unit tests for Groovy Pattern Assembler and Acoustic Slice Classifier."""

import numpy as np
import pytest

from wub_core.model_markov import SliceRef
from wub_core.groovy_pattern_assembler import (
    AcousticSliceClassifier,
    build_scale_chroma_template,
    get_chroma_template_for_key,
    harmonic_consonance,
    assemble_groovy_pattern,
    assemble_evolving_phrase,
    apply_sidechain_ducking,
    apply_hihat_envelope,
    GLOBAL_GROOVE_TEMPLATES,
)


def _make_dummy_slice(
    stem: str,
    track_id: str = "trk_1",
    seg_id: str = "seg_01",
    duration: float = 0.25,
    centroid: float = 1200.0,
    loud_max: float = -20.0,
    chroma: np.ndarray = None,
    tatum_in_bar: int = 0,
) -> SliceRef:
    if chroma is None:
        chroma = np.zeros(12, dtype=np.float32)
    return SliceRef(
        stem=stem,
        track_id=track_id,
        segment_id=seg_id,
        start=0.0,
        duration=duration,
        timbre=np.zeros(12, dtype=np.float32),
        chroma=chroma,
        loud_max=loud_max,
        track_pos=0,
        track_bpm=126.0,
        beat_pos=0,
        tatum_pos=0,
        tatum_in_bar=tatum_in_bar,
        spectral_centroid=centroid,
    )


def test_acoustic_slice_classifier_drums():
    kick = _make_dummy_slice("drums", centroid=600.0, loud_max=-18.0)
    assert AcousticSliceClassifier.classify(kick) == "kick"

    snare = _make_dummy_slice("drums", centroid=1800.0, loud_max=-20.0)
    assert AcousticSliceClassifier.classify(snare) == "snare"

    hat_closed = _make_dummy_slice("drums", centroid=4500.0, duration=0.08, loud_max=-25.0)
    assert AcousticSliceClassifier.classify(hat_closed) == "hat_closed"

    hat_open = _make_dummy_slice("drums", centroid=4200.0, duration=0.22, loud_max=-22.0)
    assert AcousticSliceClassifier.classify(hat_open) == "hat_open"

    fill = _make_dummy_slice("drums", centroid=1100.0, duration=0.35, loud_max=-24.0)
    assert AcousticSliceClassifier.classify(fill) == "fill"


def test_acoustic_slice_classifier_bass_other_vocals():
    sub = _make_dummy_slice("bass", centroid=220.0, duration=0.30)
    assert AcousticSliceClassifier.classify(sub) == "sub"

    rolling = _make_dummy_slice("bass", centroid=650.0, duration=0.18)
    assert AcousticSliceClassifier.classify(rolling) == "rolling"

    stab = _make_dummy_slice("bass", centroid=700.0, duration=0.10)
    assert AcousticSliceClassifier.classify(stab) == "stab"

    # Other: chord with polyphonic chroma
    poly_chroma = np.zeros(12, dtype=np.float32)
    poly_chroma[0] = 1.0  # C
    poly_chroma[4] = 0.8  # E
    poly_chroma[7] = 0.9  # G
    chord = _make_dummy_slice("other", chroma=poly_chroma, duration=0.25)
    assert AcousticSliceClassifier.classify(chord) == "chord"

    pad = _make_dummy_slice("other", duration=0.55)
    assert AcousticSliceClassifier.classify(pad) == "pad"

    arp = _make_dummy_slice("other", duration=0.12)
    assert AcousticSliceClassifier.classify(arp) == "stab" or AcousticSliceClassifier.classify(arp) == "arp"

    v_chop = _make_dummy_slice("vocals", duration=0.15)
    assert AcousticSliceClassifier.classify(v_chop) == "chop"

    v_phrase = _make_dummy_slice("vocals", duration=0.45)
    assert AcousticSliceClassifier.classify(v_phrase) == "phrase"


def test_harmonic_consonance():
    # C major chord in C major (8B) key
    c_maj_chroma = np.zeros(12, dtype=np.float32)
    c_maj_chroma[0] = 1.0  # C
    c_maj_chroma[4] = 0.8  # E
    c_maj_chroma[7] = 0.9  # G
    c_slice = _make_dummy_slice("other", chroma=c_maj_chroma)

    score_consonant = harmonic_consonance(c_slice, "8B")
    assert score_consonant > 0.70

    # F# / C# dissonant out-of-scale tritone
    tritone_chroma = np.zeros(12, dtype=np.float32)
    tritone_chroma[1] = 1.0  # C#
    tritone_chroma[6] = 1.0  # F#
    dissonant_slice = _make_dummy_slice("other", chroma=tritone_chroma)

    score_dissonant = harmonic_consonance(dissonant_slice, "8B")
    assert score_consonant > score_dissonant


def test_assemble_groovy_pattern():
    palette = {
        "drums": {
            "kick": [_make_dummy_slice("drums", centroid=500.0, loud_max=-18.0)],
            "snare": [_make_dummy_slice("drums", centroid=1500.0, loud_max=-20.0)],
            "hat_open": [_make_dummy_slice("drums", centroid=4000.0, duration=0.20)],
            "hat_closed": [_make_dummy_slice("drums", centroid=4500.0, duration=0.08)],
            "percussion": [_make_dummy_slice("drums", centroid=2200.0, duration=0.10)],
            "fill": [_make_dummy_slice("drums", centroid=1200.0, duration=0.35)],
        },
        "bass": {
            "sub": [_make_dummy_slice("bass", centroid=250.0, duration=0.25)],
            "rolling": [_make_dummy_slice("bass", centroid=600.0, duration=0.15)],
            "stab": [_make_dummy_slice("bass", centroid=700.0, duration=0.10)],
        },
        "other": {
            "chord": [_make_dummy_slice("other", duration=0.25)],
            "arp": [_make_dummy_slice("other", duration=0.12)],
            "pad": [_make_dummy_slice("other", duration=0.50)],
            "stab": [_make_dummy_slice("other", duration=0.10)],
        },
        "vocals": {
            "chop": [_make_dummy_slice("vocals", duration=0.18)],
            "phrase": [_make_dummy_slice("vocals", duration=0.40)],
        },
    }

    pat = assemble_groovy_pattern(
        stem="drums",
        template_name="four_on_floor_hats",
        palette=palette,
        target_key="8B",
    )

    assert pat is not None
    assert pat.length == 16
    assert pat.density >= 0.5
    # Kicks on 0, 4, 8, 12
    assert pat.steps[0] is not None
    assert pat.steps[4] is not None
    assert pat.steps[8] is not None
    assert pat.steps[12] is not None
    # Velocities populated
    assert pat.velocities is not None
    assert len(pat.velocities) == 16
    assert pat.velocities[0] >= 0.9


def test_assemble_evolving_phrase():
    palette = {
        "drums": {
            "kick": [_make_dummy_slice("drums", centroid=500.0)],
            "snare": [_make_dummy_slice("drums", centroid=1500.0)],
            "hat_open": [_make_dummy_slice("drums", centroid=4000.0, duration=0.2)],
            "hat_closed": [_make_dummy_slice("drums", centroid=4500.0, duration=0.08)],
            "percussion": [_make_dummy_slice("drums", centroid=2000.0)],
            "fill": [_make_dummy_slice("drums", centroid=1200.0, duration=0.35)],
        },
        "bass": {
            "sub": [_make_dummy_slice("bass", centroid=250.0)],
            "rolling": [_make_dummy_slice("bass", centroid=600.0)],
            "stab": [_make_dummy_slice("bass", centroid=700.0)],
        },
        "other": {
            "chord": [_make_dummy_slice("other", duration=0.25)],
            "arp": [_make_dummy_slice("other", duration=0.12)],
            "pad": [_make_dummy_slice("other", duration=0.50)],
            "stab": [_make_dummy_slice("other", duration=0.10)],
        },
        "vocals": {
            "chop": [_make_dummy_slice("vocals", duration=0.18)],
            "phrase": [_make_dummy_slice("vocals", duration=0.40)],
        },
    }

    # Test phrase bar 0 (Foundation)
    bar0 = assemble_evolving_phrase(phrase_bar=0, palette=palette, target_key="8A")
    assert bar0["drums"] is not None
    assert bar0["bass"] is not None
    assert bar0["drums"].groove_type == "four_on_floor_hats"

    # Test phrase bar 7 (Turnaround fill)
    bar7 = assemble_evolving_phrase(phrase_bar=7, palette=palette, target_key="8A")
    assert bar7["drums"] is not None
    assert bar7["drums"].groove_type == "turnaround_fill"
    # Step 15 silent breath before drop
    assert bar7["drums"].steps[15] is None


def test_sidechain_ducking():
    sr = 44100
    bpm = 126.0
    tatum_sec = 60.0 / bpm / 4.0
    total_samples = int(16 * tatum_sec * sr)

    # Constant amplitude bass waveform
    bass_orig = np.ones(total_samples, dtype=np.float32)
    kick_steps = [0, 4, 8, 12]

    ducked = apply_sidechain_ducking(bass_orig, kick_steps, bpm=bpm, sr=sr, duck_gain_db=-6.0)

    # Initial transient of step 0 should be attenuated to ~0.5 (-6 dB)
    assert ducked[0] < 0.60
    assert ducked[0] >= 0.45

    # Near the end of the tatum, should have recovered back towards 1.0
    step_samples = int(tatum_sec * sr)
    recovered_sample = ducked[int(step_samples * 0.9)]
    assert recovered_sample > 0.90


def test_hihat_envelope():
    sr = 44100
    hat_orig = np.ones(int(0.20 * sr), dtype=np.float32)
    shaped = apply_hihat_envelope(hat_orig, is_closed=True, sr=sr)

    # Start should be at full amplitude
    assert np.isclose(shaped[0], 1.0, atol=1e-3)
    # End of 200ms slice should be heavily attenuated
    assert shaped[-1] < 0.05
