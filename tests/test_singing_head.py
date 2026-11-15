"""Tests for the reverse-trained Singing Head vocal articulator model and UI wiring."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_singing_head import (  # noqa: E402
    ARTICULATOR_NAMES,
    FEATURE_NAMES,
    FFT_SIZE,
    TARGET_SR,
    VOWELS,
    compute_articulatory_targets,
    estimate_lpc_formants,
    extract_frame_features,
    load_audio_44k,
    train_reverse_model,
)


def _synth_vowel(f0: float, f1: float, f2: float, f3: float = 2600.0, amp: float = 0.25, n: int = FFT_SIZE) -> np.ndarray:
    """Synthesize a voiced vowel frame using a harmonic buzz + cascade formant resonators."""
    t = np.arange(n, dtype=np.float64) / float(TARGET_SR)
    # Bandlimited glottal-like harmonic source (tilted -6 dB/oct)
    sig = np.zeros(n, dtype=np.float64)
    for h in range(1, 25):
        fh = f0 * h
        if fh >= 5500.0:
            break
        sig += (1.0 / (h ** 1.1)) * np.sin(2.0 * np.pi * fh * t)
    for freq, bw in ((f1, 70.0), (f2, 110.0), (f3, 160.0)):
        r = math.exp(-math.pi * bw / float(TARGET_SR))
        theta = 2.0 * math.pi * freq / float(TARGET_SR)
        a = np.array([1.0, -2.0 * r * math.cos(theta), r * r], dtype=np.float64)
        sig = lfilter([1.0 - r], a, sig)
    sig = sig / max(1e-6, float(np.sqrt(np.mean(sig * sig)))) * amp
    return sig.astype(np.float32)


def _forward_shipped_model(model_data: dict, feats: np.ndarray, rms: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the exported JSON MLP forward pass matching frontend/src/singing_head_model.ts."""
    mean = np.array(model_data["feature_mean"], dtype=np.float64)
    std = np.maximum(np.array(model_data["feature_std"], dtype=np.float64), 1e-4)
    x = (feats.astype(np.float64) - mean) / std

    w1 = np.array(model_data["weights"]["w1"], dtype=np.float64)
    b1 = np.array(model_data["weights"]["b1"], dtype=np.float64)
    w2 = np.array(model_data["weights"]["w2"], dtype=np.float64)
    b2 = np.array(model_data["weights"]["b2"], dtype=np.float64)
    w3 = np.array(model_data["weights"]["w3"], dtype=np.float64)
    b3 = np.array(model_data["weights"]["b3"], dtype=np.float64)

    def gelu_np(z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    h1 = gelu_np(w1 @ x + b1)
    h2 = gelu_np(w2 @ h1 + b2)
    raw = w3 @ h2 + b3

    sig = 1.0 / (1.0 + np.exp(-raw[:9]))
    silence_scale = max(0.0, min(1.0, (rms - 0.0015) / 0.004))
    art = sig.copy()
    art[0] *= silence_scale
    art[1] = 0.5 + (art[1] - 0.5) * silence_scale
    art[2] *= silence_scale
    art[3] *= silence_scale
    art[4] = 0.55 + (art[4] - 0.55) * silence_scale
    art[6] *= silence_scale
    art[7] *= silence_scale
    art[8] *= silence_scale

    logits = raw[9:] - np.max(raw[9:])
    exps = np.exp(logits)
    vprobs = exps / np.sum(exps)
    return art, vprobs


def test_train_script_cli_help() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_singing_head.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0
    assert "--kits-dir" in r.stdout


def test_lpc_formants_and_features_on_synthetic_vowels() -> None:
    ah = _synth_vowel(180.0, 740.0, 1220.0)
    ee = _synth_vowel(240.0, 310.0, 2260.0)
    oo = _synth_vowel(160.0, 320.0, 860.0)

    f1_ah, f2_ah, _ = estimate_lpc_formants(ah)
    f1_ee, f2_ee, _ = estimate_lpc_formants(ee)
    f1_oo, f2_oo, _ = estimate_lpc_formants(oo)

    assert f1_ah > f1_ee + 150.0
    assert f2_ee > f2_ah + 500.0
    assert f2_oo < f2_ah

    feats, rms, f0, voicing, norm_mag = extract_frame_features(ah, TARGET_SR, None)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))
    assert rms > 0.1
    assert 165.0 <= f0 <= 195.0
    assert voicing > 0.65
    assert len(norm_mag) > 50


def test_reverse_training_end_to_end_small() -> None:
    rows = []
    specs = [
        (180.0, 730.0, 1220.0, 0.25),
        (260.0, 310.0, 2280.0, 0.22),
        (150.0, 320.0, 860.0, 0.20),
        (210.0, 580.0, 1800.0, 0.18),
        (190.0, 560.0, 920.0, 0.16),
        (220.0, 700.0, 1200.0, 0.0005),
    ]
    base_frames = []
    for f0, f1_t, f2_t, amp in specs:
        sig = np.zeros(FFT_SIZE, dtype=np.float32) if amp < 0.002 else _synth_vowel(f0, f1_t, f2_t, amp=amp)
        base_frames.append(sig)

    rng = np.random.default_rng(7)
    for t_idx in range(12):
        tid = f"trk_{t_idx:02d}"
        for rep in range(3):
            for sig in base_frames:
                noisy = (sig + rng.normal(0, 0.0004, FFT_SIZE)).astype(np.float32)
                feats, rms, est_f0, voicing, _ = extract_frame_features(noisy, TARGET_SR, None)
                f1, f2, f3 = estimate_lpc_formants(noisy, TARGET_SR) if rms > 0.004 else (550.0, 1450.0, 2550.0)
                rows.append(
                    {
                        "track_id": tid,
                        "is_stem": False,
                        "feats": feats,
                        "rms": rms,
                        "f0": est_f0,
                        "voicing": voicing,
                        "f1": f1,
                        "f2": f2,
                        "f3": f3,
                        "vibrato": 0.1,
                        "rms_rise": 0.0,
                    }
                )

    x_mat, y_mat, c_mat, stats = compute_articulatory_targets(rows)
    assert x_mat.shape == (len(rows), 18)
    assert y_mat.shape == (len(rows), 14)
    assert c_mat.shape == (len(rows), 4)
    assert stats["db_p95"] > stats["db_p10"]

    model, feat_mean, feat_std, metrics = train_reverse_model(
        x_mat, y_mat, c_mat, [r["track_id"] for r in rows], epochs=35, batch_size=32, seed=7, device="cpu"
    )
    assert feat_mean.shape == (18,)
    assert feat_std.shape == (18,)
    assert metrics["val_mae"] < 0.20
    assert model is not None


def test_shipped_singing_head_model_and_responses() -> None:
    model_path = ROOT / "frontend/public/data/singing_head_model.json"
    assert model_path.exists(), "Shipped singing_head_model.json missing"
    with open(model_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["format"] == "wub_singing_head_mlp_v1"
    assert data["feature_names"] == FEATURE_NAMES
    assert data["articulators"] == ARTICULATOR_NAMES
    assert data["vowels"] == VOWELS
    assert data["corpus_stats"]["stubs_used"] >= 2000

    for key in ("w1", "b1", "w2", "b2", "w3", "b3"):
        arr = np.array(data["weights"][key], dtype=np.float64)
        assert np.all(np.isfinite(arr))

    # 1. Silence stays closed
    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    f_sil, rms_sil, _, _, _ = extract_frame_features(silence, TARGET_SR, None)
    art_sil, _ = _forward_shipped_model(data, f_sil, rms_sil)
    assert art_sil[0] < 0.12  # aperture
    assert art_sil[3] < 0.12  # jaw_drop

    # 2. Loud open AH opens wide and drops jaw
    ah = _synth_vowel(220.0, 760.0, 1200.0, amp=0.18)
    f_ah, rms_ah, _, _, _ = extract_frame_features(ah, TARGET_SR, None)
    art_ah, v_ah = _forward_shipped_model(data, f_ah, rms_ah)
    assert art_ah[0] > 0.50  # aperture
    assert art_ah[3] > 0.30  # jaw_drop

    # 3. Bright EE spreads wider than rounded OO
    ee = _synth_vowel(280.0, 300.0, 2320.0, amp=0.14)
    oo = _synth_vowel(160.0, 310.0, 820.0, amp=0.14)
    f_ee, rms_ee, _, _, _ = extract_frame_features(ee, TARGET_SR, None)
    f_oo, rms_oo, _, _, _ = extract_frame_features(oo, TARGET_SR, None)
    art_ee, v_ee = _forward_shipped_model(data, f_ee, rms_ee)
    art_oo, v_oo = _forward_shipped_model(data, f_oo, rms_oo)
    assert art_ee[1] > art_oo[1] + 0.10  # width: EE wider than OO
    assert art_oo[2] > art_ee[2] + 0.10  # roundness: OO rounder than EE
    assert v_ee[1] > v_oo[1]             # EE prob higher on EE
    assert (v_oo[2] + v_oo[3]) > (v_ee[2] + v_ee[3])  # OH/OO prob higher on OO

    # 4. Real vocal stub peak frame produces clear vocal aperture
    stub_paths = sorted((ROOT / "frontend/public/data/kits/kits").glob("*/*_vocals.opus"))
    assert stub_paths, "Expected at least one vocal kit stub on disk"
    best_aperture = 0.0
    for stub in stub_paths[:8]:
        wav = load_audio_44k(stub, max_seconds=6.0)
        if len(wav) < FFT_SIZE:
            continue
        peak_idx = int(np.argmax(np.abs(wav)))
        start = max(0, min(len(wav) - FFT_SIZE, peak_idx - FFT_SIZE // 2))
        frame = wav[start : start + FFT_SIZE]
        f_stub, rms_stub, _, _, _ = extract_frame_features(frame, TARGET_SR, None)
        art_stub, _ = _forward_shipped_model(data, f_stub, rms_stub)
        best_aperture = max(best_aperture, float(art_stub[0]))
    assert best_aperture > 0.25


def test_singing_head_bottom_of_page_dom_wiring() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    stages_pos = html.find('id="stages"')
    head_pos = html.find('id="singing-head-panel"')
    canvas_pos = html.find('id="singing-head-canvas"')
    footer_pos = html.find('id="status-line"')

    assert stages_pos != -1
    assert head_pos > stages_pos
    assert canvas_pos > head_pos
    assert footer_pos > canvas_pos
