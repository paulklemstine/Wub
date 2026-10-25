"""Unit tests for pattern audio waveform assembly, acoustic scoring, and training feedback loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wub_core.model_markov import SliceRef
from wub_core.pattern_audio_trainer import (
    AudioMatchMetrics,
    GenericPatternOptimizer,
    PatternAudioFeedbackEngine,
    assemble_multistem_waveform,
    assemble_pattern_waveform,
    compute_audio_similarity,
)
from wub_core.patterns import PatternBank, SequencerPattern


def _make_dummy_slice(tid: str = "t1", seg_id: str = "s1", start: float = 0.0, dur: float = 0.25) -> SliceRef:
    return SliceRef(
        stem="drums",
        track_id=tid,
        segment_id=seg_id,
        start=start,
        duration=dur,
        timbre=np.zeros(12, dtype=np.float32),
        chroma=np.zeros(12, dtype=np.float32),
        loud_max=-12.0,
        track_pos=0,
        track_bpm=126.0,
    )


def test_compute_audio_similarity_identical_signals():
    sr = 22050
    duration = 1.905
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Rhythmic signal with transient pulses
    sig = 0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 200 * t)
    # Add pulses at 0, 4, 8, 12 beats
    for beat in (0, 0.476, 0.952, 1.428):
        idx = int(beat * sr)
        sig[idx : idx + 200] += 1.0

    m = compute_audio_similarity(sig, sig, sr=sr, stem="drums", groove_type="four_on_floor")
    assert isinstance(m, AudioMatchMetrics)
    assert 0.90 <= m.composite_score <= 1.0
    assert 0.90 <= m.spectral_similarity <= 1.0
    assert 0.85 <= m.onset_correlation <= 1.0


def test_compute_audio_similarity_different_signals():
    sr = 22050
    duration = 1.905
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Low frequency bass vs high frequency noise
    sig1 = 0.8 * np.sin(2 * np.pi * 50 * t)
    sig2 = 0.8 * np.sin(2 * np.pi * 8000 * t)

    m = compute_audio_similarity(sig1, sig2, sr=sr, stem="drums")
    assert m.composite_score < 0.65
    assert m.spectral_similarity < 0.40


def test_assemble_pattern_waveform_silence_fallback():
    # Slices with nonexistent audio files return silence without crashing
    steps = [_make_dummy_slice(seg_id=f"s_{i}") if i in (0, 4, 8, 12) else None for i in range(16)]
    wave = assemble_pattern_waveform(
        steps=steps,
        audio_for={},
        fallback_audio={},
        stem="drums",
        bpm=126.0,
        sr=22050,
    )
    assert isinstance(wave, np.ndarray)
    assert len(wave) == int(16 * (60.0 / 126.0 / 4.0) * 22050)
    assert np.all(wave == 0.0)


def test_feedback_engine_record_and_save(tmp_path: Path):
    fb_path = tmp_path / "test_feedback.json"
    engine = PatternAudioFeedbackEngine(feedback_path=fb_path)

    metrics = AudioMatchMetrics(
        composite_score=0.88,
        spectral_similarity=0.85,
        onset_correlation=0.92,
        chroma_similarity=0.80,
        dynamic_match=0.90,
        stem="drums",
        groove_type="four_on_floor_hats",
        ref_track_id="track_123",
    )
    steps = [_make_dummy_slice(tid="track_123", seg_id="slice_abc")]

    engine.record_metrics("pat_test_001", metrics, steps)
    engine._save()

    assert fb_path.exists()
    data = json.loads(fb_path.read_text())
    assert "pat_test_001" in data["patterns"]
    assert data["patterns"]["pat_test_001"]["composite_score"] == 0.88
    assert data["slices"]["slice_abc"] > 0.5
    assert data["grooves"]["drums"]["four_on_floor_hats"] > 0.5


def test_pattern_bank_weights_by_audio_similarity():
    bank = PatternBank(seed=123)
    p1 = SequencerPattern(
        id="pat_low",
        stem="drums",
        track_id="t1",
        pattern_name="Low Sim",
        length=16,
        steps=[None] * 16,
        energy=0.5,
        density=0.25,
        onset_grid=[0, 4, 8, 12],
        groove_type="four_on_floor",
        audio_similarity_score=0.2,
    )
    p2 = SequencerPattern(
        id="pat_high",
        stem="drums",
        track_id="t2",
        pattern_name="High Sim",
        length=16,
        steps=[None] * 16,
        energy=0.5,
        density=0.25,
        onset_grid=[0, 4, 8, 12],
        groove_type="four_on_floor",
        audio_similarity_score=0.95,
    )
    bank.add_pattern(p1)
    bank.add_pattern(p2)

    # Over 100 choices, high-similarity pattern should be selected the vast majority of times
    counts = {"pat_low": 0, "pat_high": 0}
    rng = np.random.default_rng(42)
    for _ in range(100):
        chosen = bank.pick_pattern(stem="drums", groove_type="four_on_floor", rng=rng)
        assert chosen is not None
        counts[chosen.id] += 1

    assert counts["pat_high"] > counts["pat_low"] * 3
