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
    MultiStemCohesionMetrics,
    PatternAudioFeedbackEngine,
    SongCompositionEconomyScorer,
    SongCompositionMetrics,
    assemble_multistem_waveform,
    assemble_pattern_waveform,
    compute_audio_similarity,
    compute_multistem_cohesion,
)
from wub_core.patterns import MultiStemPattern, PatternBank, SequencerPattern, SongPhrase


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


def test_compute_multistem_cohesion_balanced_mix():
    sr = 22050
    duration = 1.905
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # 1. Drums: Four on the floor kick (0, 0.476, 0.952, 1.428)
    drums = np.zeros(n_samples, dtype=np.float32)
    for beat in (0, 0.476, 0.952, 1.428):
        idx = int(beat * sr)
        drums[idx : idx + 300] += 0.8 * np.sin(2 * np.pi * 55 * t[:300])

    # 2. Bass: Offbeat pulses (0.238, 0.714, 1.190, 1.666) -> Clean pocket, no collision!
    bass = np.zeros(n_samples, dtype=np.float32)
    for offbeat in (0.238, 0.714, 1.190, 1.666):
        idx = int(offbeat * sr)
        bass[idx : idx + 400] += 0.7 * np.sin(2 * np.pi * 110 * t[:400])

    # 3. Other: Ambient chord pads
    other = 0.3 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)

    # 4. Vocals: Vocal verse singing with periodic pauses
    vocals = np.zeros(n_samples, dtype=np.float32)
    v_start, v_end = int(0.2 * sr), int(0.7 * sr)
    vocals[v_start:v_end] = 0.5 * np.sin(2 * np.pi * 880 * t[: (v_end - v_start)])

    synth_stems = {"drums": drums, "bass": bass, "other": other, "vocals": vocals}
    cohesion = compute_multistem_cohesion(
        synth_stems=synth_stems,
        sr=sr,
        stem_grooves={"drums": "four_on_floor_hats", "bass": "offbeat", "other": "chords", "vocals": "vocal_verse"},
    )

    assert isinstance(cohesion, MultiStemCohesionMetrics)
    assert cohesion.composite_cohesion >= 0.70
    assert cohesion.rhythmic_pocket_score >= 0.80
    assert cohesion.call_and_response_score >= 0.80


def test_compute_multistem_cohesion_detects_low_collision():
    sr = 22050
    duration = 1.905
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Simultaneous blasting sub-bass on kick downbeats without sidechain ducking -> collision!
    drums = np.zeros(n_samples, dtype=np.float32)
    bass = np.zeros(n_samples, dtype=np.float32)
    for beat in (0, 0.476, 0.952, 1.428):
        idx = int(beat * sr)
        drums[idx : idx + 500] += 1.0 * np.sin(2 * np.pi * 50 * t[:500])
        bass[idx : idx + 500] += 1.0 * np.sin(2 * np.pi * 50 * t[:500])

    synth_stems = {"drums": drums, "bass": bass}
    cohesion = compute_multistem_cohesion(synth_stems=synth_stems, sr=sr)
    # Frequency masking score should reflect low-end clash penalty
    assert cohesion.frequency_masking_score < 0.85


def test_export_locked_feedback_weights(tmp_path: Path):
    weights_p = tmp_path / "composing_ml_weights.json"
    public_weights_p = tmp_path / "public_composing_ml_weights.json"
    patterns_p = tmp_path / "standalone_patterns.json"

    # Seed mock standalone patterns
    mock_manifest = {
        "version": 3,
        "patterns": {
            "drums": [{"id": "p_d1", "groove_type": "four_on_floor_hats"}],
            "bass": [{"id": "p_b1", "groove_type": "offbeat"}],
        },
    }
    patterns_p.write_text(json.dumps(mock_manifest))

    fb_path = tmp_path / "pattern_audio_feedback.json"
    engine = PatternAudioFeedbackEngine(feedback_path=fb_path)
    engine.feedback_data["stem_pair_compatibility"] = {
        "drums_bass": {"four_on_floor_hats": {"offbeat": 0.96}},
    }
    engine.feedback_data["patterns"]["p_d1"] = {"spectral_similarity": 0.92, "composite_score": 0.94}
    engine.feedback_data["slices"]["slice_kick_1"] = 0.95

    res = engine.export_locked_feedback_weights(
        weights_path=weights_p,
        public_weights_path=public_weights_p,
        standalone_patterns_path=patterns_p,
    )
    assert res["status"] == "ok"
    assert weights_p.exists()
    assert public_weights_p.exists()

    w_data = json.loads(weights_p.read_text())
    assert "stem_cohesion_weights" in w_data
    sc = w_data["stem_cohesion_weights"]
    assert sc["stem_pair_compatibility"]["drums_bass"]["four_on_floor_hats"]["offbeat"] == 0.96
    assert sc["trained_slice_scores"]["slice_kick_1"] == 0.95

    pat_data = json.loads(patterns_p.read_text())
    assert "stem_cohesion_weights" in pat_data
    assert pat_data["patterns"]["drums"][0]["audio_similarity_score"] == 0.92
    assert pat_data["patterns"]["drums"][0]["cohesion_score"] == 0.94


def test_song_composition_economy_scorer():
    scorer = SongCompositionEconomyScorer()

    # Create a compact song (16 bars composed of 2 recurring patterns and 8 unique slices)
    s_kick = _make_dummy_slice(tid="track_a", seg_id="k1")
    s_snare = _make_dummy_slice(tid="track_a", seg_id="s1")
    s_hat = _make_dummy_slice(tid="track_a", seg_id="h1")
    s_bass = _make_dummy_slice(tid="track_a", seg_id="b1")

    # 16 bars
    bars = []
    for b in range(16):
        d_steps = [s_kick if i in (0, 8) else (s_snare if i in (4, 12) else (s_hat if i in (2, 6, 10, 14) else None)) for i in range(16)]
        b_steps = [s_bass if i in (2, 6, 10, 14) else None for i in range(16)]
        mp = MultiStemPattern(
            id=f"mp_{b}",
            track_id="track_a",
            track_title="Groovy Minimal Track",
            bar_num=b,
            phrase_bar=b % 8,
            section="GROOVE_A",
            stems={"drums": d_steps, "bass": b_steps},
            groove_types={"drums": "four_on_floor_hats", "bass": "offbeat"},
        )
        bars.append(mp)

    metrics = scorer.evaluate_song("track_a", "Groovy Minimal Track", bars, acoustic_fidelity=0.88)
    assert isinstance(metrics, SongCompositionMetrics)
    assert metrics.unique_slices_count == 4
    # With 16 bars * 32 stem steps = 512 tatums and only 4 unique slices, compression should be very high
    assert metrics.slice_compression_ratio >= 0.95
    # Only 1 unique pattern signature across 16 bars -> very high economy
    assert metrics.pattern_subset_economy >= 0.90
    assert metrics.composite_song_score >= 0.85
    assert metrics.to_dict()["unique_slices_count"] == 4


def test_feedback_engine_evaluate_song_composition_economy(tmp_path: Path):
    fb_path = tmp_path / "pattern_audio_feedback.json"
    engine = PatternAudioFeedbackEngine(feedback_path=fb_path)

    # Mock corpus store and bank
    class MockTrack:
        track_id = "tr_001"
        title = "Deep Electronic"

    class MockCorpusStore:
        analyses = {"tr_001": None}
        tracks = [MockTrack()]

    s_kick = _make_dummy_slice(tid="tr_001", seg_id="k1")
    bars = [
        MultiStemPattern(
            id=f"mp_{b}",
            track_id="tr_001",
            track_title="Deep Electronic",
            bar_num=b,
            phrase_bar=b % 8,
            section="GROOVE_A",
            stems={"drums": [s_kick if i % 4 == 0 else None for i in range(16)]},
            groove_types={"drums": "four_on_floor"},
        )
        for b in range(8)
    ]
    phrase = SongPhrase(
        id="phr_001",
        track_id="tr_001",
        track_title="Deep Electronic",
        phrase_index=0,
        section="GROOVE_A",
        bpm=126.0,
        key="C major",
        energy=0.8,
        bars=bars,
    )

    class MockBank:
        phrases = [phrase]

    res = engine.evaluate_song_composition_economy(MockCorpusStore(), MockBank(), max_tracks=5)
    assert res["status"] == "ok"
    assert res["scored_tracks"] == 1
    assert "song_composition_scores" in engine.feedback_data
    assert "tr_001" in engine.feedback_data["song_composition_scores"]
    assert engine.feedback_data["song_composition_scores"]["tr_001"]["slice_compression_ratio"] > 0.80


