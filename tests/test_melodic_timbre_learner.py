"""Unit tests for wub_core.melodic_timbre_learner."""

import json
from pathlib import Path
import numpy as np
import pytest

from wub_core.melodic_timbre_learner import (
    MelodicTimbreLearner,
    classify_motif_affinity,
    estimate_t60_schroeder,
    estimate_wet_dry_ratio,
    fit_biquad_cutoff_curve,
    fit_energy_crescendo_curve,
    fit_power_curve,
    fit_resonance_curve,
    MODAL_TEMPLATES,
    STYLES,
)


def test_fit_power_curve():
    u = np.linspace(0.0, 1.0, 20)
    # Test linear (alpha = 1.0)
    y_linear = 100.0 + 900.0 * u
    alpha_lin = fit_power_curve(u, y_linear, 100.0, 1000.0)
    assert 0.9 <= alpha_lin <= 1.1

    # Test quadratic (alpha = 2.0)
    y_quad = 100.0 + 900.0 * (u**2.0)
    alpha_quad = fit_power_curve(u, y_quad, 100.0, 1000.0)
    assert 1.8 <= alpha_quad <= 2.2


def test_fit_biquad_cutoff_curve():
    u = np.linspace(0.0, 1.0, 8)
    fc = np.geomspace(400.0, 16000.0, 8)
    f_start, f_end, alpha, samples_16 = fit_biquad_cutoff_curve(u, fc, "lowpass")

    assert f_start == pytest.approx(400.0, abs=1.0)
    assert f_end == pytest.approx(16000.0, abs=1.0)
    assert len(samples_16) == 16
    assert samples_16[0] == pytest.approx(400.0, abs=2.0)
    assert samples_16[-1] == pytest.approx(16000.0, abs=2.0)
    # Increasing curve
    assert samples_16[-1] > samples_16[0]


def test_fit_resonance_curve():
    u = np.linspace(0.0, 1.0, 8)
    q_st, q_en, q_peak, u_peak, samples_16 = fit_resonance_curve(
        u, q_start=1.0, q_end=1.1, q_peak=4.5, u_peak=0.88
    )

    assert q_st == 1.0
    assert q_en == 1.1
    assert q_peak == 4.5
    assert len(samples_16) == 16
    # Max value in samples should be near q_peak
    assert max(samples_16) == pytest.approx(4.5, abs=0.2)
    # Right at the end, it drops back toward q_end for drop gap
    assert samples_16[-1] < max(samples_16)


def test_fit_energy_crescendo_curve():
    u = np.linspace(0.0, 1.0, 8)
    rms = np.linspace(-30.0, -3.0, 8)
    l_start, l_end, gamma, tension_curve = fit_energy_crescendo_curve(u, rms)

    assert l_start == pytest.approx(-30.0, abs=0.5)
    assert l_end == pytest.approx(-3.0, abs=0.5)
    assert len(tension_curve) == 16
    assert tension_curve[0] == 0.0
    assert tension_curve[-1] == 1.0
    # Monotonically increasing tension
    for i in range(len(tension_curve) - 1):
        assert tension_curve[i] <= tension_curve[i + 1]


def test_classify_motif_affinity():
    aff_lyd, deg_lyd = classify_motif_affinity((0, 4, 6))
    assert "Lydian" in aff_lyd
    assert "1 - 3 - #4" == deg_lyd

    aff_phryg, deg_phryg = classify_motif_affinity((0, 1, 3))
    assert "Phrygian" in aff_phryg
    assert "1 - b2 - b3" == deg_phryg

    aff_dor, deg_dor = classify_motif_affinity((0, 3, 9))
    assert "Dorian" in aff_dor

    aff_pent, deg_pent = classify_motif_affinity((0, 3, 7, 10))
    assert "Pentatonic" in aff_pent


def test_estimate_t60_schroeder_synthetic():
    sr = 22050
    # Generate an exponential decay with exact T60 = 1.5 seconds
    # Amp(t) = exp(-t / tau), energy drops 60 dB (factor 1e-6) in 1.5s
    # log(1e-6) = -13.8155 -> tau = 1.5 / 6.9077 ~ 0.2171
    target_t60 = 1.5
    tau = target_t60 / 6.9077
    t = np.linspace(0.0, 2.0, int(2.0 * sr))
    envelope = np.exp(-t / tau)
    # Add noise carrier
    carrier = np.random.RandomState(42).randn(len(t))
    signal = envelope * carrier

    measured_t60 = estimate_t60_schroeder(signal, sr)
    assert measured_t60 is not None
    assert measured_t60 == pytest.approx(target_t60, rel=0.25)


def test_estimate_wet_dry_ratio():
    sr = 22050
    # Synthetic impulse + tail
    sig = np.zeros(int(1.0 * sr))
    # Impulse in direct window (first 20ms)
    sig[: int(0.02 * sr)] = 1.0
    # Small tail
    sig[int(0.05 * sr) : int(0.5 * sr)] = 0.1
    ratio = estimate_wet_dry_ratio(sig, sr)
    assert ratio is not None
    assert 0.15 <= ratio <= 0.65


def test_melodic_timbre_learner_execution(tmp_path):
    learner = MelodicTimbreLearner(processed_dir="data/processed", stems_dir="data/stems")
    out_file = tmp_path / "test_models.json"

    # Run quick execution with small track and stem limits
    payload = learner.learn_and_export(output_path=out_file, max_tracks=5, max_stems=2)

    assert out_file.exists()
    assert payload["version"] == "1.0.0"
    assert payload["styles"] == STYLES

    for style in STYLES:
        assert style in payload["filter_sweep_models"]
        assert "buildup_8bar" in payload["filter_sweep_models"][style]
        assert "cutoff" in payload["filter_sweep_models"][style]["buildup_8bar"]
        assert "resonance" in payload["filter_sweep_models"][style]["buildup_8bar"]
        assert "energy_crescendo" in payload["filter_sweep_models"][style]["buildup_8bar"]

        assert style in payload["melodic_hook_motifs"]
        for stem_key in ("synth_lead", "vocal_melody"):
            entry = payload["melodic_hook_motifs"][style][stem_key]
            assert "modal_distribution" in entry
            assert "dominant_mode" in entry
            # Modal distribution must sum to 1.0
            total_prob = sum(entry["modal_distribution"].values())
            assert total_prob == pytest.approx(1.0, abs=1e-3)
            assert "top_motifs_3note" in entry
            assert "top_motifs_4note" in entry
            assert "top_motifs_5note" in entry

        assert style in payload["reverb_space_profiles"]
        rev = payload["reverb_space_profiles"][style]
        assert rev["t60_decay_ms"] > 0
        assert 0.0 < rev["wet_dry_ratio"] < 1.0
        assert "stem_reverbs" in rev
