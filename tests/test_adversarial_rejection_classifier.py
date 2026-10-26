"""Unit tests for the Adversarial Rejection Classifier, Pattern Audio Goodness Discriminator,
and continuous multi-minute vocal speech arrangement.
"""

from pathlib import Path
import numpy as np
import pytest

from wub_core.adversarial_rejection_classifier import (
    GoodnessEvaluationResult,
    PatternAudioGoodnessDiscriminator,
    AdversarialRejectionSampler,
)
from wub_core.patterns import SequencerPattern


def _make_sine_wave(freq: float, duration_sec: float = 0.5, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.7 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_test_pattern(
    pat_id: str,
    stem: str,
    groove_type: str,
    onset_grid: list,
) -> SequencerPattern:
    return SequencerPattern(
        id=pat_id,
        stem=stem,
        track_id="tr_synth_01",
        pattern_name=f"Test {stem.capitalize()} Pattern",
        length=16,
        steps=[None] * 16,
        energy=0.85,
        density=len(onset_grid) / 16.0,
        onset_grid=onset_grid,
        groove_type=groove_type,
    )


def test_discriminator_accepts_good_master_match():
    """Verify that audio closely matching authentic master audio scores high goodness and is accepted."""
    sr = 22050
    discriminator = PatternAudioGoodnessDiscriminator(rejection_threshold=0.65, sr=sr)

    # Master wave and rendered candidate wave (harmonic sine at A4 440Hz + harmonics)
    master = _make_sine_wave(440.0, 0.5, sr)
    rendered = master * 0.95 + 0.02 * np.random.randn(len(master)).astype(np.float32)

    pat = _make_test_pattern(
        pat_id="test_pat_good",
        stem="drums",
        groove_type="four_on_floor",
        onset_grid=[0, 4, 8, 12],
    )

    res = discriminator.evaluate(
        rendered_wave=rendered,
        master_wave=master,
        pattern=pat,
        stem="drums",
        groove_type="four_on_floor",
        chord_root=9,  # A
        is_minor=True,
    )

    assert isinstance(res, GoodnessEvaluationResult)
    assert res.goodness_score >= 0.65, f"Expected goodness >= 0.65, got {res.goodness_score}"
    assert res.is_accepted is True
    assert len(res.rejection_reasons) <= 1


def test_discriminator_rejects_silent_or_corrupt_audio():
    """Verify that silent, corrupt, or chaotic noise audio is rejected with clear diagnostic reasons."""
    sr = 22050
    discriminator = PatternAudioGoodnessDiscriminator(rejection_threshold=0.70, sr=sr)

    master = _make_sine_wave(440.0, 0.5, sr)
    # Silent or ultra-low amplitude wave
    silent_rendered = np.zeros(int(sr * 0.5), dtype=np.float32)

    pat = _make_test_pattern(
        pat_id="test_pat_silent",
        stem="drums",
        groove_type="four_on_floor",
        onset_grid=[],  # Empty onset grid
    )

    res = discriminator.evaluate(
        rendered_wave=silent_rendered,
        master_wave=master,
        pattern=pat,
        stem="drums",
        groove_type="four_on_floor",
    )

    assert res.is_accepted is False
    assert res.goodness_score < 0.60
    assert len(res.rejection_reasons) >= 1
    assert any("spectral" in r or "density" in r or "pocket" in r for r in res.rejection_reasons)


def test_adversarial_rejection_sampler_mutates_until_accepted():
    """Verify that AdversarialRejectionSampler resamples or mutates rejected candidates until accepted."""
    sr = 22050
    discriminator = PatternAudioGoodnessDiscriminator(rejection_threshold=0.68, sr=sr)
    sampler = AdversarialRejectionSampler(discriminator=discriminator, max_attempts=4)

    master = _make_sine_wave(220.0, 0.5, sr)
    attempt_counter = [0]

    def candidate_generator():
        attempt_counter[0] += 1
        if attempt_counter[0] == 1:
            # Bad candidate: erratic noise and empty onsets
            bad_wave = (0.01 * np.random.randn(int(sr * 0.5))).astype(np.float32)
            bad_pat = _make_test_pattern(pat_id="bad_1", stem="bass", groove_type="offbeat", onset_grid=[])
            return bad_pat, bad_wave
        else:
            # Good candidate: rich bass tone in pocket
            good_wave = master.copy()
            good_pat = _make_test_pattern(pat_id="good_1", stem="bass", groove_type="offbeat", onset_grid=[2, 6, 10, 14])
            return good_pat, good_wave

    pat, wave, eval_res = sampler.sample_accepted_pattern(
        candidate_generator=candidate_generator,
        master_reference_wave=master,
        stem="bass",
        groove_type="offbeat",
        chord_root=9,
    )

    assert eval_res.is_accepted is True
    assert eval_res.goodness_score >= 0.68
    assert sampler.rejection_stats["evaluated"] >= 2
    assert sampler.rejection_stats["accepted"] >= 1
    assert sampler.rejection_stats["mutated_success"] >= 1


def test_discriminator_weights_persistence(tmp_path, monkeypatch):
    """Verify that discriminator feature weights and rejection threshold can be saved and loaded."""
    test_weights_file = tmp_path / "test_classifier_weights.json"
    import wub_core.adversarial_rejection_classifier as arc
    monkeypatch.setattr(arc, "CLASSIFIER_WEIGHTS_FILE", test_weights_file)

    d1 = PatternAudioGoodnessDiscriminator(rejection_threshold=0.74)
    d1.weights["temporal"] = 0.30
    d1.save_learned_weights()

    assert test_weights_file.exists()

    d2 = PatternAudioGoodnessDiscriminator()
    assert d2.rejection_threshold == 0.74
    assert d2.weights["temporal"] == 0.30


def test_rejection_classifier_detailed_metrics_and_reasons():
    """Verify that evaluation produces comprehensive sub-metrics and diagnostic rejection reasons."""
    sr = 22050
    discriminator = PatternAudioGoodnessDiscriminator(rejection_threshold=0.70, sr=sr)

    master = _make_sine_wave(440.0, 0.5, sr)
    # Render with dissonant pitch (quarter-tone clash)
    dissonant = _make_sine_wave(465.0, 0.5, sr)

    pat = _make_test_pattern("pat_diss", "bass", "offbeat", [2, 6, 10, 14])
    res = discriminator.evaluate(
        rendered_wave=dissonant,
        master_wave=master,
        pattern=pat,
        stem="bass",
        groove_type="offbeat",
        chord_root=0,  # C
        is_minor=False,
    )

    d = res.to_dict()
    assert "goodness_score" in d
    assert "spectral_match" in d
    assert "temporal_match" in d
    assert "crest_match" in d
    assert "harmonic_match" in d
    assert "structural_match" in d
    assert 0.0 <= d["goodness_score"] <= 1.0


def test_extended_speech_manifest_and_stanzas():
    """Verify that vocal patterns contain extended speech capabilities for multi-minute arrangement."""
    from wub_core.vocal_pattern_extractor import learn_vocal_corpus_patterns
    res = learn_vocal_corpus_patterns(Path("processed"))
    themes = res["themes"]

    for theme in themes:
        assert len(theme["versePhrases"]) == 4
        assert len(theme["chorusPhrases"]) == 4
        # Verify speech phrases have forward-moving syllables
        all_words = [s["text"] for phrase in theme["versePhrases"] for s in phrase]
        assert len(all_words) >= 12, "Should have rich vocabulary across verse phrases"
