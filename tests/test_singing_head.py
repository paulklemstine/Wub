"""Tests for the reverse-trained Singing Head vocal articulator model and UI wiring."""

from __future__ import annotations

import json
import math
import os
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
    TRACT_ARTICULATOR_KEYS,
    TRACT_REST_RAW,
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


def _forward_shipped_model(
    model_data: dict, feats: np.ndarray, rms: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    logits = raw[9:14] - np.max(raw[9:14])
    exps = np.exp(logits)
    vprobs = exps / np.sum(exps)

    tract = np.zeros(14, dtype=np.float64)
    if len(raw) >= 28:
        ranges = np.array(model_data["tract_ranges"], dtype=np.float64)
        lo = ranges[:, 0]
        hi = ranges[:, 1]
        norm_t = 1.0 / (1.0 + np.exp(-raw[14:28]))
        pred_raw = lo + norm_t * (hi - lo)
        rest = np.array(TRACT_REST_RAW, dtype=np.float64)
        tract = np.clip(rest + (pred_raw - rest) * silence_scale, lo, hi)

    return art, vprobs, tract


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


def _synth_nasal_murmur(f0_hz: float = 175.0, amp: float = 0.14) -> np.ndarray:
    """Synthesize a voiced bilabial/alveolar nasal murmur dominated by ~180-250 Hz."""
    t = np.arange(FFT_SIZE, dtype=np.float64) / TARGET_SR
    sig = 0.85 * np.sin(2.0 * math.pi * f0_hz * t) + 0.18 * np.sin(2.0 * math.pi * 2.0 * f0_hz * t)
    sig = sig / max(1e-6, float(np.sqrt(np.mean(sig * sig)))) * amp
    return sig.astype(np.float32)


def _synth_sibilant(amp: float = 0.12, seed: int = 19) -> np.ndarray:
    """Synthesize an unvoiced /s/ sibilant noise burst bandpassed around 6.5 kHz."""
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, FFT_SIZE)
    b, a = butter(4, [5200.0 / (0.5 * TARGET_SR), 8400.0 / (0.5 * TARGET_SR)], btype="band")
    sig = lfilter(b, a, noise)
    sig = sig / max(1e-6, float(np.sqrt(np.mean(sig * sig)))) * amp
    return sig.astype(np.float32)


def test_shipped_singing_head_model_and_responses() -> None:
    model_path = ROOT / "frontend/public/data/singing_head_model.json"
    assert model_path.exists(), "Shipped singing_head_model.json missing"
    with open(model_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["format"] == "wub_singing_head_mlp_v2"
    assert data["feature_names"] == FEATURE_NAMES
    assert data["articulators"] == ARTICULATOR_NAMES
    assert data["vowels"] == VOWELS
    assert data["tract_keys"] == TRACT_ARTICULATOR_KEYS
    assert len(data["tract_ranges"]) == 14
    assert data["corpus_stats"]["stubs_used"] >= 2000
    assert data["corpus_stats"]["stems_used"] >= 1500
    assert data["corpus_stats"]["physics_forward_r2"] > 0.65
    assert data["corpus_stats"]["physics_cycle_r2"] > 0.40
    assert data["corpus_stats"].get("roundtrip_env_corr", 0.0) > 0.80
    assert data["corpus_stats"].get("roundtrip_feat_r2", 0.0) > 0.50

    for key in ("w1", "b1", "w2", "b2", "w3", "b3"):
        arr = np.array(data["weights"][key], dtype=np.float64)
        assert np.all(np.isfinite(arr))

    idx_of = {k: i for i, k in enumerate(TRACT_ARTICULATOR_KEYS)}

    # 1. Silence stays gently closed at rest
    silence = np.zeros(FFT_SIZE, dtype=np.float32)
    f_sil, rms_sil, _, _, _ = extract_frame_features(silence, TARGET_SR, None)
    art_sil, _, tract_sil = _forward_shipped_model(data, f_sil, rms_sil)
    assert art_sil[0] < 0.12  # aperture
    assert art_sil[3] < 0.12  # jaw_drop
    assert tract_sil[idx_of["jaw"]] < 0.25
    assert tract_sil[idx_of["lipOpen"]] < 0.15

    # 2. Vocal simulator audio -> parameter reconstitution check
    import torch
    from scripts.generate_tract_soundbank import _node_compile_and_sample, synthesize_soundbank

    art_14, areas_44, dx, lip_area, min_area, min_idx_norm = _node_compile_and_sample(128)
    from scripts.train_singing_head import _norm_tract_np

    sim_art, sim_feats, _, _ = synthesize_soundbank(
        art_14,
        areas_44,
        dx,
        lip_area,
        min_area,
        min_idx_norm,
        excitations_per_posture=1,
        dev=torch.device("cpu"),
        batch_size=128,
        seed=17,
    )
    pred_tracts = []
    for i in range(len(sim_feats)):
        _, _, t_pred = _forward_shipped_model(data, sim_feats[i], 0.08)
        pred_tracts.append(t_pred)
    pred_tracts_arr = np.stack(pred_tracts, axis=0)
    assert np.all(np.isfinite(pred_tracts_arr))
    # Verify reconstituted parameters vary meaningfully across diverse simulator sounds
    assert float(np.std(pred_tracts_arr[:, idx_of["jaw"]])) > 0.03
    assert float(np.std(pred_tracts_arr[:, idx_of["tongueX"]])) > 0.04
    assert float(np.std(pred_tracts_arr[:, idx_of["lipOpen"]])) > 0.03
    # Verify roundtrip parameter correlation against true simulator tract postures
    true_n = _norm_tract_np(sim_art)
    pred_n = _norm_tract_np(pred_tracts_arr)
    assert float(np.corrcoef(true_n[:, idx_of["tongueX"]], pred_n[:, idx_of["tongueX"]])[0, 1]) > 0.65
    assert float(np.corrcoef(true_n[:, idx_of["jaw"]], pred_n[:, idx_of["jaw"]])[0, 1]) > 0.45
    assert float(np.corrcoef(true_n[:, idx_of["lipOpen"]], pred_n[:, idx_of["lipOpen"]])[0, 1]) > 0.45

    # 3. Real playing song vocal stubs produce clear mouth articulation AND smooth non-jittery trajectories
    stub_paths = sorted((ROOT / "frontend/public/data/kits/kits").glob("*/*_vocals.opus"))
    assert stub_paths, "Expected at least one vocal kit stub on disk"
    best_aperture = 0.0
    best_jaw = 0.0
    seq_vels = []
    seq_accels = []
    for stub in stub_paths[:10]:
        wav = load_audio_44k(stub, max_seconds=6.0)
        if len(wav) < FFT_SIZE + 40 * 735:
            continue
        peak_idx = int(np.argmax(np.abs(wav)))
        start0 = max(0, min(len(wav) - (FFT_SIZE + 32 * 735), peak_idx - 16 * 735))
        prev_mag = None
        seq_t = []
        for f_i in range(32):
            s = start0 + f_i * 735
            frame = wav[s : s + FFT_SIZE]
            f_stub, rms_stub, _, _, prev_mag = extract_frame_features(frame, TARGET_SR, prev_mag)
            art_stub, _, tract_stub = _forward_shipped_model(data, f_stub, rms_stub)
            best_aperture = max(best_aperture, float(art_stub[0]))
            best_jaw = max(best_jaw, float(tract_stub[idx_of["jaw"]]))
            # Normalize to [0, 1] range for uniform velocity/jerk check
            lo = np.array([r[0] for r in data["tract_ranges"]], dtype=np.float64)
            span = np.array([r[1] - r[0] for r in data["tract_ranges"]], dtype=np.float64)
            seq_t.append((tract_stub - lo) / span)
        seq_arr = np.stack(seq_t, axis=0)
        d1 = np.abs(seq_arr[1:] - seq_arr[:-1])
        d2 = np.abs(seq_arr[2:] - 2.0 * seq_arr[1:-1] + seq_arr[:-2])
        seq_vels.append(float(np.mean(d1)))
        seq_accels.append(float(np.mean(d2)))

    assert best_aperture > 0.25
    assert best_jaw > 0.25
    assert float(np.mean(seq_vels)) < 0.055, f"Trajectory velocity too jittery: {np.mean(seq_vels)}"
    assert float(np.mean(seq_accels)) < 0.065, f"Trajectory acceleration/jerk too high: {np.mean(seq_accels)}"


def test_singing_head_bottom_of_page_dom_wiring() -> None:
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    stages_pos = html.find('id="stages"')
    head_pos = html.find('id="singing-head-panel"')
    canvas_pos = html.find('id="singing-head-canvas"')
    host3d_pos = html.find('id="singing-head-3d-host"')
    footer_pos = html.find('id="status-line"')

    assert stages_pos != -1
    assert head_pos > stages_pos
    assert canvas_pos > head_pos
    assert host3d_pos > canvas_pos
    assert footer_pos > host3d_pos

    for token in (
        'id="sh-audio-track"',
        'id="sh-audio-synth"',
        'id="sh-mode-side"',
        'id="sh-mode-3d"',
        'data-sh-field="pressure"',
        'data-sh-field="velocity"',
        'data-sh-field="air"',
        'id="sh-toggle-labels"',
        'data-sh-preset="side"',
        'data-sh-preset="front"',
        'data-sh-preset="top"',
        'data-sh-preset="three"',
        'id="sh-toggle-cut"',
        'id="sh-toggle-tissues"',
    ):
        assert token in html, f"Missing expected control {token} in index.html"

    head_img = ROOT / "frontend/public/anatomy/head.jpg"
    assert head_img.exists() and head_img.stat().st_size > 100_000


def test_vocal_sim_geometry_and_meshes_bridge(tmp_path: Path) -> None:
    out_dir = tmp_path / "js"
    subprocess.run(
        [
            "npx",
            "--prefix",
            str(ROOT / "frontend"),
            "tsc",
            "--target",
            "es2022",
            "--module",
            "commonjs",
            "--moduleResolution",
            "node",
            "--esModuleInterop",
            "--outDir",
            str(out_dir),
            str(ROOT / "frontend/src/vocal_sim/articulation_bridge.ts"),
            str(ROOT / "frontend/src/vocal_sim/head/scene.ts"),
            str(ROOT / "frontend/src/vocal_sim/three/meshes.ts"),
            str(ROOT / "frontend/src/vocal_sim/tract_synth.ts"),
        ],
        check=True,
    )
    script = """
    const bridge = require(process.env.OUT + '/vocal_sim/articulation_bridge.js');
    const geomMod = require(process.env.OUT + '/vocal_sim/tract/geometry.js');
    const sceneMod = require(process.env.OUT + '/vocal_sim/head/scene.js');
    const meshMod = require(process.env.OUT + '/vocal_sim/three/meshes.js');
    const synthMod = require(process.env.OUT + '/vocal_sim/tract_synth.js');
    const typesMod = require(process.env.OUT + '/vocal_sim/tract/types.js');

    const an = bridge.defaultAnatomy();
    const teeth = bridge.defaultTeeth();
    const outArt = { ...typesMod.DEFAULT_ARTICULATORS };

    const ah = {
      aperture: 0.85, width: 0.5, roundness: 0.05, jawDrop: 0.82, tongueHeight: 0.2,
      pitchNorm: 0.4, vibrato: 0.1, tension: 0.8, breath: 0.1,
      vowelProbs: [0.85, 0.05, 0.04, 0.03, 0.03], topVowel: 'AH',
      rms: 0.16, f0Hz: 220, voicing: 0.9, centroidHz: 950, trained: true,
    };
    const ee = {
      aperture: 0.35, width: 0.88, roundness: 0.05, jawDrop: 0.18, tongueHeight: 0.9,
      pitchNorm: 0.5, vibrato: 0.1, tension: 0.75, breath: 0.1,
      vowelProbs: [0.03, 0.88, 0.03, 0.03, 0.03], topVowel: 'EE',
      rms: 0.14, f0Hz: 260, voicing: 0.9, centroidHz: 1850, trained: true,
    };
    const oo = {
      aperture: 0.25, width: 0.1, roundness: 0.9, jawDrop: 0.2, tongueHeight: 0.8,
      pitchNorm: 0.3, vibrato: 0.1, tension: 0.7, breath: 0.1,
      vowelProbs: [0.03, 0.03, 0.1, 0.81, 0.03], topVowel: 'OO',
      rms: 0.14, f0Hz: 180, voicing: 0.9, centroidHz: 600, trained: true,
    };

    const artAh = bridge.articulationToTract(ah, 1.0, { ...outArt });
    const artEe = bridge.articulationToTract(ee, 1.0, { ...outArt });
    const artOo = bridge.articulationToTract(oo, 1.0, { ...outArt });

    const gAh = geomMod.computeGeometry(artAh, an, teeth);
    const gEe = geomMod.computeGeometry(artEe, an, teeth);
    const gOo = geomMod.computeGeometry(artOo, an, teeth);
    const sAh = sceneMod.headScene(gAh, an, artAh);
    const mAh = meshMod.buildTractMeshes(gAh);

    const acAh = synthMod.evaluateTractAcoustics(artAh, gAh);
    const acEe = synthMod.evaluateTractAcoustics(artEe, gEe);
    const acOo = synthMod.evaluateTractAcoustics(artOo, gOo);

    if (!(gAh.lipArea > gOo.lipArea)) throw new Error('Expected AH lipArea > OO lipArea');
    if (!(gOo.length > gAh.length * 0.95)) throw new Error('Expected valid vocal tract length');
    if (!sAh.sprites || sAh.sprites.length < 3) throw new Error('Expected painted head scene sprites');
    if (!(mAh.air.positions.length > 1000 && mAh.teeth.length > 0)) throw new Error('Expected valid 3D meshes');
    if (!(acAh.openDrive > acOo.openDrive)) throw new Error('Expected AH openDrive > OO openDrive');
    if (!(acEe.formantHz[1] > acOo.formantHz[1])) throw new Error('Expected EE F2 > OO F2');
    console.log(JSON.stringify({
      lipAh: gAh.lipArea,
      lipOo: gOo.lipArea,
      sprites: sAh.sprites.length,
      teeth: mAh.teeth.length,
      f1Ah: acAh.formantHz[0],
      f2Ee: acEe.formantHz[1],
      f2Oo: acOo.formantHz[1],
    }));
    """
    env = {**os.environ, "OUT": str(out_dir)}
    proc = subprocess.run(["node", "-e", script], env=env, capture_output=True, text=True, check=True)
    parsed = json.loads(proc.stdout.strip())
    assert parsed["lipAh"] > parsed["lipOo"]
    assert parsed["teeth"] > 0


def test_python_typescript_14d_parity(tmp_path: Path) -> None:
    model_path = ROOT / "frontend/public/data/singing_head_model.json"
    with open(model_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = tmp_path / "js"
    subprocess.run(
        [
            "npx",
            "--prefix",
            str(ROOT / "frontend"),
            "tsc",
            "--target",
            "es2022",
            "--module",
            "commonjs",
            "--moduleResolution",
            "node",
            "--esModuleInterop",
            "--outDir",
            str(out_dir),
            str(ROOT / "frontend/src/singing_head_model.ts"),
            str(ROOT / "frontend/src/vocal_sim/articulation_bridge.ts"),
            str(ROOT / "frontend/src/vocal_sim/tract_synth.ts"),
        ],
        check=True,
    )

    probes = {
        "silence": np.zeros(FFT_SIZE, dtype=np.float32),
        "ah": _synth_vowel(220.0, 760.0, 1200.0, amp=0.18),
        "ee": _synth_vowel(280.0, 300.0, 2320.0, amp=0.14),
        "oo": _synth_vowel(160.0, 310.0, 820.0, amp=0.14),
        "nasal": _synth_nasal_murmur(),
        "sibilant": _synth_sibilant(),
    }

    py_out = {}
    samples_payload = {}
    for name, wav in probes.items():
        feats, rms, _, _, _ = extract_frame_features(wav, TARGET_SR, None)
        art, vprobs, tract = _forward_shipped_model(data, feats, rms)
        py_out[name] = {
            "art": art.tolist(),
            "vowelProbs": vprobs.tolist(),
            "tract": tract.tolist(),
        }
        samples_payload[name] = wav.tolist()

    payload_file = tmp_path / "probes.json"
    payload_file.write_text(json.dumps(samples_payload), encoding="utf-8")

    node_script = """
    const fs = require('fs');
    const mod = require(process.env.OUT + '/singing_head_model.js');
    const bridge = require(process.env.OUT + '/vocal_sim/articulation_bridge.js');
    const geomMod = require(process.env.OUT + '/vocal_sim/tract/geometry.js');
    const synthMod = require(process.env.OUT + '/vocal_sim/tract_synth.js');
    const types = require(process.env.OUT + '/vocal_sim/tract/types.js');
    const model = JSON.parse(fs.readFileSync(process.env.MODEL, 'utf8'));
    if (!mod.isValidSingingHeadModel(model)) throw new Error('isValidSingingHeadModel rejected v2');
    const probes = JSON.parse(fs.readFileSync(process.env.PROBES, 'utf8'));
    const inf = new mod.SingingHeadInference();
    inf.setModel(model);
    const an = bridge.defaultAnatomy();
    const teeth = bridge.defaultTeeth();
    const out = {};
    const acOut = {};
    for (const [name, arr] of Object.entries(probes)) {
      inf.resetFluxHistory();
      const pred = inf.infer(Float32Array.from(arr), 44100);
      out[name] = {
        art: [pred.aperture, pred.width, pred.roundness, pred.jawDrop, pred.tongueHeight, pred.pitchNorm, pred.vibrato, pred.tension, pred.breath],
        vowelProbs: Array.from(pred.vowelProbs),
        tract: types.ARTICULATOR_KEYS.map((k) => pred.tract[k]),
      };
      const tArt = bridge.articulationToTract(pred, 0.0, { ...types.DEFAULT_ARTICULATORS });
      const geom = geomMod.computeGeometry(tArt, an, teeth);
      const ac = synthMod.evaluateTractAcoustics(tArt, geom);
      acOut[name] = { f0Hz: pred.f0Hz, f1: ac.formantHz[0], f2: ac.formantHz[1], openDrive: ac.openDrive, fricGain: ac.fricGain };
    }
    if (!(acOut.ah.openDrive > acOut.oo.openDrive)) throw new Error('Roundtrip openDrive mismatch: ' + JSON.stringify(acOut));
    if (!(acOut.ah.f1 > acOut.ee.f1)) throw new Error('Roundtrip AH vs EE F1 mismatch: ' + JSON.stringify(acOut));
    if (!(acOut.ee.f2 > acOut.oo.f2)) throw new Error('Roundtrip EE vs OO F2 mismatch: ' + JSON.stringify(acOut));
    if (!(Math.abs(acOut.ah.f0Hz - 220) < 6 && Math.abs(acOut.oo.f0Hz - 160) < 6)) throw new Error('Roundtrip F0 tracking mismatch: ' + JSON.stringify(acOut));
    console.log(JSON.stringify(out));
    """
    env = {
        **os.environ,
        "OUT": str(out_dir),
        "MODEL": str(model_path),
        "PROBES": str(payload_file),
    }
    proc = subprocess.run(["node", "-e", node_script], env=env, capture_output=True, text=True, check=True)
    ts_out = json.loads(proc.stdout.strip())

    max_diff = 0.0
    for name in probes:
        for field in ("art", "vowelProbs", "tract"):
            diff = float(np.max(np.abs(np.array(py_out[name][field]) - np.array(ts_out[name][field]))))
            max_diff = max(max_diff, diff)
    assert max_diff < 1e-6, f"Python/TS parity drift too large: {max_diff}"
