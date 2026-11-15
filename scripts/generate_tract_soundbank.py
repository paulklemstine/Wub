#!/usr/bin/env python3
"""Generate a large acoustic soundbank from Voice Workshop's 14-D vocal simulator.

Stage A (Node + compiled TypeScript vocal tract):
  Samples `N` random & diverse 14-parameter postures across
  `frontend/src/vocal_sim/tract/{types,geometry,areaFunction}.ts` and computes
  exact 44-section oral tract areas, segment length `dx`, lip radiation area,
  narrowest constriction (`minArea`, `minIdx`), velum opening, and `lipTuck`.

Stage B (PyTorch GPU/CPU Acoustic Wave Synthesizer):
  For each 14-D tract posture, evaluates the 44-section Webster transmission-line
  transfer function `H_oral(f)` + parallel velum nasal branch `H_nasal(f)` +
  front-cavity constriction turbulence `H_fric(f)` across the Nyquist grid,
  synthesizes real 2048-sample @ 44.1 kHz vocal waveforms with varied random
  fundamental frequencies, open-quotient tilts, voicing levels, and breaths,
  and extracts both:
    - 18-d browser-exact acoustic feature vectors (`feats_18`), and
    - 32-band L2-normalized log-spectral timbre envelopes (`env_32`).

Saves the complete `(art_14, feats_18, env_32, cues_6)` soundbank to `.npz` so
every real vocal stem frame can be matched directly against real synthesized
simulator sounds via smooth Viterbi trajectory optimization.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

TARGET_SR = 44100
FFT_SIZE = 2048

NODE_TRACT_GEOM_SCRIPT = r"""
const fs = require('fs');
const path = require('path');

const tractDir = process.argv[2];
const outBinPath = process.argv[3];
const count = parseInt(process.argv[4] || '65536', 10);

const {
  ARTICULATOR_KEYS,
  ARTICULATOR_RANGES,
  DEFAULT_ANATOMY,
  DEFAULT_TEETH,
  DEFAULT_ARTICULATORS,
} = require(path.join(tractDir, 'types.js'));
const { computeGeometry } = require(path.join(tractDir, 'geometry.js'));
const { sectionsFromGeometry } = require(path.join(tractDir, 'areaFunction.js'));

let seed = 0x57554253;
function rand() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function randn() {
  const u = Math.max(1e-7, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function clampArt(a) {
  const out = {};
  for (const k of ARTICULATOR_KEYS) {
    const [lo, hi] = ARTICULATOR_RANGES[k];
    const v = a[k] ?? DEFAULT_ARTICULATORS[k];
    out[k] = Math.max(lo, Math.min(hi, v));
  }
  return out;
}
// Pack N x 63 float32 values per sample:
// [0..13] art_14, [14..57] areas_44, [58] dx, [59] lipArea, [60] minArea, [61] minIdxNorm, [62] totalLen
const STRIDE = 63;
const buf = new Float32Array(count * STRIDE);

let geom = null;
let sec = null;
for (let i = 0; i < count; i++) {
  // Pure random 14-D vocal simulator parameters across ARTICULATOR_RANGES.
  // We use stratified Sobol-like / uniform + beta-varied random sampling across
  // all 14 parameters so open, closed, constricted, rounded, spread, and nasal
  // vocal simulator configurations are densely covered without any preset vowels.
  const cand = {};
  const shapeMode = rand();
  for (const k of ARTICULATOR_KEYS) {
    const [lo, hi] = ARTICULATOR_RANGES[k];
    let u = rand();
    if (shapeMode < 0.35) {
      // Gently favor full-range extremes as well as interior values
      u = 0.5 + 0.5 * Math.sin((u - 0.5) * Math.PI);
    } else if (shapeMode < 0.65) {
      // Smooth interior distribution
      u = 0.5 * (u + rand());
    }
    cand[k] = lo + u * (hi - lo);
  }
  // Ensure physical variety across both open oral resonators and tight constrictions
  if (rand() < 0.50) {
    cand.velum = rand() * 0.15;
  }
  if (rand() < 0.55) {
    cand.lipTuck = rand() * 0.20;
  }

  const art = clampArt(cand);
  geom = computeGeometry(art, DEFAULT_ANATOMY, DEFAULT_TEETH, geom ? { reuse: geom } : undefined);
  sec = sectionsFromGeometry(geom, 44, sec);

  const areas = sec.areas;
  let minArea = 999.0;
  let minIdx = 0;
  for (let s = 0; s < 44; s++) {
    if (areas[s] < minArea) {
      minArea = areas[s];
      minIdx = s;
    }
  }

  const baseOff = i * STRIDE;
  for (let j = 0; j < 14; j++) {
    buf[baseOff + j] = art[ARTICULATOR_KEYS[j]];
  }
  for (let s = 0; s < 44; s++) {
    buf[baseOff + 14 + s] = areas[s];
  }
  buf[baseOff + 58] = sec.dx;
  buf[baseOff + 59] = Math.max(0.01, geom.lipArea);
  buf[baseOff + 60] = minArea;
  buf[baseOff + 61] = minIdx / 43.0;
  buf[baseOff + 62] = sec.length;
}

fs.writeFileSync(outBinPath, Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength));
process.stdout.write(`Wrote ${count} random Voice Workshop tract geometries to ${outBinPath}\n`);
"""

BANDS_HZ = [
    (80.0, 250.0),
    (250.0, 500.0),
    (500.0, 850.0),
    (850.0, 1350.0),
    (1350.0, 2000.0),
    (2000.0, 3000.0),
    (3000.0, 5000.0),
    (5000.0, 9000.0),
]


def _resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def _node_compile_and_sample(count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compile TS vocal tract and generate `count` random 44-section tract geometries."""
    tsc_bin = ROOT / "frontend/node_modules/.bin/tsc"
    tract_src = ROOT / "frontend/src/vocal_sim/tract"
    constants_src = ROOT / "frontend/src/vocal_sim/constants.ts"

    node_env = os.environ.copy()
    nvm_node = sorted((Path.home() / ".nvm/versions/node").glob("*/bin"))
    if nvm_node:
        node_env["PATH"] = f"{nvm_node[-1]}:{node_env.get('PATH', '')}"

    if shutil.which("node", path=node_env["PATH"]) is None:
        raise RuntimeError("Node.js not found on PATH.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="wub_soundbank_"))
    try:
        src_copy = tmp_dir / "src"
        tract_copy = src_copy / "tract"
        tract_copy.mkdir(parents=True, exist_ok=True)
        shutil.copy2(constants_src, src_copy / "constants.ts")
        for f in tract_src.glob("*.ts"):
            shutil.copy2(f, tract_copy / f.name)

        dist_dir = tmp_dir / "dist"
        tsc_cmd = (
            [str(tsc_bin)]
            if tsc_bin.exists()
            else ["npx", "--prefix", str(ROOT / "frontend"), "tsc"]
        )
        subprocess.run(
            tsc_cmd
            + [
                "--target",
                "es2022",
                "--module",
                "commonjs",
                "--moduleResolution",
                "node",
                "--esModuleInterop",
                "--skipLibCheck",
                "--outDir",
                str(dist_dir),
                str(src_copy / "constants.ts"),
                str(tract_copy / "types.ts"),
                str(tract_copy / "section3d.ts"),
                str(tract_copy / "geometry.ts"),
                str(tract_copy / "areaFunction.ts"),
            ],
            check=True,
            cwd=str(ROOT),
            env=node_env,
        )

        script_path = tmp_dir / "sample_tracts.js"
        bin_path = tmp_dir / "tracts.bin"
        script_path.write_text(NODE_TRACT_GEOM_SCRIPT, encoding="utf-8")
        subprocess.run(
            ["node", str(script_path), str(dist_dir / "tract"), str(bin_path), str(count)],
            check=True,
            env=node_env,
        )

        raw = np.fromfile(bin_path, dtype=np.float32).reshape(count, 63)
        art_14 = raw[:, :14].copy()
        areas_44 = raw[:, 14:58].copy()
        dx = raw[:, 58].copy()
        lip_area = raw[:, 59].copy()
        min_area = raw[:, 60].copy()
        min_idx_norm = raw[:, 61].copy()
        return art_14, areas_44, dx, lip_area, min_area, min_idx_norm
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _compute_transfer_spectra(
    areas_44: torch.Tensor,
    dx: torch.Tensor,
    lip_area: torch.Tensor,
    velum: torch.Tensor,
    min_area: torch.Tensor,
    min_idx_norm: torch.Tensor,
    lip_tuck: torch.Tensor,
    freqs_hz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute vectorized 44-section Webster oral+nasal transfer & frication spectra.

    Returns (|H_vocal(B, F)|, |H_fric(B, F)|) normalized smoothly.
    """
    rho = 1.14e-3
    c_snd = 35000.0
    alpha = 0.006

    # freqs_hz: (F,) -> omega: (1, F)
    w = (2.0 * math.pi * freqs_hz).unsqueeze(0)
    k = w / c_snd  # (1, F)
    dx_b = dx.unsqueeze(1)  # (B, 1)
    kdx = k * dx_b
    cos_k = torch.cos(kdx)
    sin_k = torch.sin(kdx)
    cosh_a = torch.cosh(alpha * dx_b)
    sinh_a = torch.sinh(alpha * dx_b)

    cosh_re = cosh_a * cos_k
    cosh_im = sinh_a * sin_k
    sinh_re = sinh_a * cos_k
    sinh_im = cosh_a * sin_k

    eff_lip = torch.clamp(lip_area, min=0.04).unsqueeze(1)  # (B, 1)
    r_lip = torch.sqrt(torch.clamp(eff_lip / math.pi, min=0.01))
    L_rad = (8.0 * rho) / (3.0 * (math.pi ** 2) * r_lip)
    p_re = 0.015 * ((rho * c_snd) / eff_lip) * ((w / 6000.0) ** 2)
    p_im = w * L_rad
    u_re = torch.ones_like(p_re)
    u_im = torch.zeros_like(p_re)

    # Traverse lips (43) -> glottis (0)
    clamped_areas = torch.clamp(areas_44, min=0.025)
    for s in range(43, -1, -1):
        Z = ((rho * c_snd) / clamped_areas[:, s]).unsqueeze(1)  # (B, 1)
        su_re = cosh_re * u_re - cosh_im * u_im
        su_im = cosh_re * u_im + cosh_im * u_re
        sp_re = (sinh_re * p_re - sinh_im * p_im) / Z
        sp_im = (sinh_re * p_im + sinh_im * p_re) / Z
        u_next_re = su_re + sp_re
        u_next_im = su_im + sp_im

        zsu_re = Z * (sinh_re * u_re - sinh_im * u_im)
        zsu_im = Z * (sinh_re * u_im + sinh_im * u_re)
        cp_re = cosh_re * p_re - cosh_im * p_im
        cp_im = cosh_re * p_im + cosh_im * p_re
        u_re, u_im = u_next_re, u_next_im
        p_re, p_im = zsu_re + cp_re, zsu_im + cp_im

    u_mag = torch.sqrt(torch.clamp(u_re * u_re + u_im * u_im, min=1e-18))
    h_oral = torch.clamp(1.0 / u_mag, max=40.0)  # (B, F)

    # Nasal resonator parallel coupling driven smoothly by velum [0..1]:
    # Adds low ~250 Hz nasal murmur + attenuates oral F1/F2 anti-resonance (~1000 Hz)
    v = torch.clamp(velum, 0.0, 1.0).unsqueeze(1)
    f_row = freqs_hz.unsqueeze(0)
    nasal_murmur = 2.6 * torch.exp(-0.5 * ((f_row - 245.0) / 115.0) ** 2) + 0.85 * torch.exp(
        -0.5 * ((f_row - 1050.0) / 260.0) ** 2
    )
    oral_antires = 1.0 - 0.55 * v * torch.exp(-0.5 * ((f_row - 960.0) / 280.0) ** 2)
    h_vocal = h_oral * oral_antires * (1.0 - 0.52 * v) + v * nasal_murmur * (
        0.5 + 0.5 * torch.clamp(h_oral.mean(dim=1, keepdim=True), 0.1, 4.0)
    )

    # Frication / sibilance spectrum controlled by constriction position & teeth obstacle:
    # Front constrictions (minIdx near lips ~0.75..0.95) -> high 5.5..7.5 kHz /s/ peak;
    # mid-palatal (~0.55..0.75) -> 2.8..4.5 kHz /sh/; lipTuck -> broad 1.8..6.5 kHz /f/v/.
    sib_center = 2400.0 + 4900.0 * torch.clamp((min_idx_norm - 0.45) / 0.50, 0.0, 1.0)
    sib_center = sib_center.unsqueeze(1)
    tightness = torch.clamp((0.36 - min_area) / 0.32, 0.0, 1.0).unsqueeze(1)
    tuck_amt = torch.clamp((lip_tuck - 0.20) / 0.70, 0.0, 1.0).unsqueeze(1)
    h_fric = tightness * torch.exp(-0.5 * ((f_row - sib_center) / 1150.0) ** 2) + 0.65 * tuck_amt * torch.exp(
        -0.5 * ((f_row - 3600.0) / 1950.0) ** 2
    )
    return h_vocal, h_fric


def _extract_gpu_features_and_env(
    waveforms: torch.Tensor,
    sr: int = TARGET_SR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-vectorized 18-d browser-exact features + 32-band L2 timbre envelope.

    Args:
      waveforms: (B, 2048) float32 tensor.
    Returns:
      (feats_18, env_32, rms)
    """
    dev = waveforms.device
    b_sz, n = waveforms.shape
    x = waveforms.to(torch.float32)
    x = x - x.mean(dim=1, keepdim=True)

    rms = torch.sqrt(torch.mean(x * x, dim=1) + 1e-18)
    peak = torch.amax(torch.abs(x), dim=1)

    signs = x >= 0.0
    zcr = (signs[:, 1:] != signs[:, :-1]).to(torch.float32).sum(dim=1) / float(max(1, n - 1))

    # Autocorrelation pitch & voicing on 2x decimated frame
    x_dec = x[:, ::2]
    m = x_dec.shape[1]
    sr_dec = sr / 2.0
    power = torch.sum(x_dec * x_dec, dim=1)
    spec_dec = torch.fft.rfft(x_dec, n=2 * m, dim=1)
    ac = torch.fft.irfft(spec_dec.real ** 2 + spec_dec.imag ** 2, n=2 * m, dim=1)[:, :m]
    min_lag = max(8, int(sr_dec / 800.0))
    max_lag = min(m - 1, int(math.ceil(sr_dec / 80.0)))
    lags = torch.arange(min_lag, max_lag + 1, device=dev, dtype=torch.float32)
    denom = torch.clamp(1.0 - lags / float(m), min=0.01).unsqueeze(0)
    scores = (ac[:, min_lag : max_lag + 1] / torch.clamp(power.unsqueeze(1), min=1e-9)) / denom
    best_val, best_off = torch.max(scores, dim=1)
    best_lag = lags[best_off]
    voicing = torch.where(power > 1e-6, torch.clamp(best_val, 0.0, 1.0), torch.zeros_like(best_val))
    voiced_ok = (power > 1e-6) & (voicing > 0.30) & (rms > 0.008) & (best_lag > 0)
    f0_hz = torch.where(voiced_ok, sr_dec / torch.clamp(best_lag, min=1.0), torch.zeros_like(best_lag))
    f0_norm = torch.where(
        f0_hz > 0.0,
        torch.clamp(torch.log2(torch.clamp(f0_hz, min=1.0) / 220.0), -1.5, 1.5),
        torch.zeros_like(f0_hz),
    )

    # Hann-windowed 2048 FFT -> 1024 magnitude bins
    idx = torch.arange(n, device=dev, dtype=torch.float32)
    win = 0.5 - 0.5 * torch.cos((2.0 * math.pi * idx) / float(n - 1))
    spec = torch.fft.rfft(x * win.unsqueeze(0), dim=1)[:, :-1]  # (B, 1024)
    mag = torch.abs(spec) / (float(n) * 0.5)
    mag_sq = mag * mag
    bin_hz = float(sr) / float(n)
    half_n = mag.shape[1]

    # Flux baseline (random temporal jitter near typical intra-vowel continuity)
    flux = torch.zeros_like(rms)

    # Spectral centroid (100..8000 Hz)
    c_lo = max(1, int(math.floor(100.0 / bin_hz)))
    c_hi = min(half_n - 1, int(math.ceil(8000.0 / bin_hz)))
    freqs_c = torch.arange(c_lo, c_hi + 1, device=dev, dtype=torch.float32) * bin_hz
    e_c = mag_sq[:, c_lo : c_hi + 1]
    e_c_sum = torch.sum(e_c, dim=1)
    centroid_hz = torch.where(
        e_c_sum > 1e-12,
        torch.sum(e_c * freqs_c.unsqueeze(0), dim=1) / torch.clamp(e_c_sum, min=1e-12),
        torch.full_like(e_c_sum, 500.0),
    )
    centroid_log = torch.clamp(torch.log2(torch.clamp(centroid_hz, min=80.0) / 500.0), -2.5, 4.0)

    # 85% rolloff (100..9000 Hz)
    r_lo = max(1, int(math.floor(100.0 / bin_hz)))
    r_hi = min(half_n - 1, int(math.ceil(9000.0 / bin_hz)))
    e_r = mag_sq[:, r_lo : r_hi + 1]
    e_r_sum = torch.sum(e_r, dim=1, keepdim=True)
    cum_r = torch.cumsum(e_r, dim=1)
    above = cum_r >= (0.85 * e_r_sum)
    idx_85 = torch.argmax(above.to(torch.int64), dim=1)
    rolloff_hz = (r_lo + idx_85.to(torch.float32)) * bin_hz
    rolloff_norm = torch.where(
        e_r_sum.squeeze(1) > 1e-12,
        torch.clamp(rolloff_hz / 9000.0, 0.0, 1.0),
        torch.zeros_like(rms),
    )

    # Flatness (150..6000 Hz)
    f_lo = max(1, int(math.floor(150.0 / bin_hz)))
    f_hi = min(half_n - 1, int(math.ceil(6000.0 / bin_hz)))
    e_f = mag_sq[:, f_lo : f_hi + 1] + 1e-12
    arith_mean = torch.mean(e_f, dim=1)
    geom_mean = torch.exp(torch.mean(torch.log(e_f), dim=1))
    flatness = torch.clamp(geom_mean / torch.clamp(arith_mean, min=1e-12), 0.0, 1.0)

    # 8 log-band energies
    bands_list = []
    for b_lo_hz, b_hi_hz in BANDS_HZ:
        lo_b = max(1, int(math.floor(b_lo_hz / bin_hz)))
        hi_b = min(half_n - 1, int(math.ceil(b_hi_hz / bin_hz)))
        band_rms = torch.sqrt(torch.mean(mag_sq[:, lo_b : hi_b + 1], dim=1))
        bands_list.append(torch.log1p(band_rms * 32.0))
    bands = torch.stack(bands_list, dim=1)  # (B, 8)
    low_high_tilt = (bands[:, 0] + bands[:, 1] + bands[:, 2]) - (bands[:, 4] + bands[:, 5] + bands[:, 6])

    feats = torch.stack(
        [
            torch.log1p(rms * 25.0),
            torch.clamp(peak * 1.5, max=1.5),
            zcr,
            voicing,
            f0_norm,
            centroid_log,
            rolloff_norm,
            flatness,
            flux,
            bands[:, 0],
            bands[:, 1],
            bands[:, 2],
            bands[:, 3],
            bands[:, 4],
            bands[:, 5],
            bands[:, 6],
            bands[:, 7],
            low_high_tilt,
        ],
        dim=1,
    )

    # 32-band L2-normalized log-spectral timbre envelope (80..9000 Hz)
    edges = np.geomspace(80.0, 9000.0, 33)
    env_cols = []
    for k_b in range(32):
        lo_k = max(1, int(math.floor(edges[k_b] / bin_hz)))
        hi_k = max(lo_k, min(half_n - 1, int(math.ceil(edges[k_b + 1] / bin_hz))))
        b_rms = torch.sqrt(torch.mean(mag_sq[:, lo_k : hi_k + 1], dim=1) + 1e-18)
        # Scale by rms so shape depends cleanly on spectral timbre, not volume
        rel = b_rms / torch.clamp(rms, min=1e-4)
        env_cols.append(torch.log1p(rel * 4.5))
    env_raw = torch.stack(env_cols, dim=1)
    env_centered = env_raw - env_raw.mean(dim=1, keepdim=True)
    env_32 = env_centered / torch.clamp(torch.norm(env_centered, dim=1, keepdim=True), min=1e-6)
    return feats, env_32, rms


def synthesize_soundbank(
    art_14: np.ndarray,
    areas_44: np.ndarray,
    dx: np.ndarray,
    lip_area: np.ndarray,
    min_area: np.ndarray,
    min_idx_norm: np.ndarray,
    excitations_per_posture: int,
    dev: torch.device,
    batch_size: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize real 44.1 kHz vocal sounds for every random tract posture and extract acoustics."""
    rng = np.random.default_rng(seed)
    n_postures = len(art_14)
    total_sounds = n_postures * excitations_per_posture

    # Pre-replicate postures across excitations
    art_rep = np.repeat(art_14, excitations_per_posture, axis=0).astype(np.float32)
    areas_rep = np.repeat(areas_44, excitations_per_posture, axis=0).astype(np.float32)
    dx_rep = np.repeat(dx, excitations_per_posture, axis=0).astype(np.float32)
    lip_rep = np.repeat(lip_area, excitations_per_posture, axis=0).astype(np.float32)
    min_a_rep = np.repeat(min_area, excitations_per_posture, axis=0).astype(np.float32)
    min_i_rep = np.repeat(min_idx_norm, excitations_per_posture, axis=0).astype(np.float32)

    out_feats = np.zeros((total_sounds, 18), dtype=np.float32)
    out_env = np.zeros((total_sounds, 32), dtype=np.float32)
    # cues_6: [f0_norm, voicing, breath, vibrato, loud_norm, rms]
    out_cues = np.zeros((total_sounds, 6), dtype=np.float32)

    n_fft_bins = FFT_SIZE // 2 + 1  # 1025 bins: 0..22050 Hz
    freqs_full = torch.linspace(0.0, TARGET_SR * 0.5, n_fft_bins, device=dev, dtype=torch.float32)
    freqs_eval = torch.clamp(freqs_full, min=65.0)
    t_axis = torch.arange(FFT_SIZE, device=dev, dtype=torch.float32) / float(TARGET_SR)

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + 13)

    harmonics = torch.arange(1, 56, device=dev, dtype=torch.float32).unsqueeze(0)  # (1, 55)

    for s in range(0, total_sounds, batch_size):
        e = min(total_sounds, s + batch_size)
        b = e - s

        a_t = torch.from_numpy(art_rep[s:e]).to(dev)
        ar_t = torch.from_numpy(areas_rep[s:e]).to(dev)
        dx_t = torch.from_numpy(dx_rep[s:e]).to(dev)
        lip_t = torch.from_numpy(lip_rep[s:e]).to(dev)
        mina_t = torch.from_numpy(min_a_rep[s:e]).to(dev)
        mini_t = torch.from_numpy(min_i_rep[s:e]).to(dev)

        velum_t = torch.clamp(a_t[:, 12], 0.0, 1.0)
        larynx_t = a_t[:, 13]
        liptuck_t = torch.clamp(a_t[:, 11], 0.0, 1.0)

        # Evaluate 44-section transfer function at all 1025 FFT bins
        h_vocal, h_fric = _compute_transfer_spectra(
            ar_t, dx_t, lip_t, velum_t, mina_t, mini_t, liptuck_t, freqs_eval
        )

        # Sample realistic singing excitation: f0 tied softly to larynx, random vocal effort tilt
        larynx_semis = larynx_t * 5.0
        log_f0 = torch.empty((b,), device=dev).uniform_(math.log2(95.0), math.log2(640.0), generator=gen)
        f0_hz = torch.clamp(2.0 ** (log_f0 + larynx_semis / 12.0 * 0.35), 82.0, 720.0)

        # Tight constrictions (<0.25 cm^2) or high lipTuck produce strong unvoiced frication/breath
        tightness = torch.clamp((0.28 - mina_t) / 0.26, 0.0, 1.0)
        tuck_w = torch.clamp((liptuck_t - 0.25) / 0.65, 0.0, 1.0)
        is_fric_dom = torch.rand((b,), device=dev, generator=gen) < (0.12 + 0.65 * torch.maximum(tightness, tuck_w))

        voicing_target = torch.where(
            is_fric_dom,
            torch.empty((b,), device=dev).uniform_(0.0, 0.35, generator=gen),
            torch.empty((b,), device=dev).uniform_(0.55, 0.98, generator=gen),
        )
        breath_amt = torch.clamp(
            torch.maximum(tightness, tuck_w) * 0.75 + (1.0 - voicing_target) * 0.55 + torch.rand((b,), device=dev, generator=gen) * 0.15,
            0.0,
            1.0,
        )
        tilt_exp = torch.empty((b,), device=dev).uniform_(0.95, 1.85, generator=gen)  # bright belt vs warm falsetto

        # Harmonic glottal source synthesized cleanly via nearest-bin spectral comb or harmonic sum, then filtered
        phase0 = torch.empty((b, 1), device=dev).uniform_(0.0, 2.0 * math.pi, generator=gen)
        harm_freqs = f0_hz.unsqueeze(1) * harmonics  # (B, 55)
        # Look up vocal tract transfer gain |H_vocal(h * f0)| at exact harmonic frequencies!
        bin_idx = torch.clamp(
            torch.round(harm_freqs / (TARGET_SR / float(FFT_SIZE))).to(torch.int64),
            0,
            n_fft_bins - 1,
        )
        h_gain_at_harm = torch.gather(h_vocal, 1, bin_idx)
        # Nyquist fade
        nyq_mask = (harm_freqs < (TARGET_SR * 0.46)).to(torch.float32)
        harm_amp = (1.0 / (harmonics ** tilt_exp.unsqueeze(1))) * h_gain_at_harm * nyq_mask
        angles = 2.0 * math.pi * harm_freqs.unsqueeze(2) * t_axis.view(1, 1, -1) + phase0.unsqueeze(2)
        voiced_wave = torch.sum(harm_amp.unsqueeze(2) * torch.sin(angles), dim=1)  # (B, 2048)

        # Normalize voiced component by its energy cleanly before mixing with shaped noise
        v_rms = torch.sqrt(torch.mean(voiced_wave * voiced_wave, dim=1, keepdim=True) + 1e-12)
        voiced_unit = voiced_wave / v_rms

        # Noise source filtered by oral + frication transfer spectra
        white = torch.randn((b, FFT_SIZE), device=dev, generator=gen)
        white_spec = torch.fft.rfft(white, dim=1)
        noise_filter = (0.35 * h_vocal + 2.8 * h_fric)
        noise_wave = torch.fft.irfft(white_spec * noise_filter, n=FFT_SIZE, dim=1)
        n_rms = torch.sqrt(torch.mean(noise_wave * noise_wave, dim=1, keepdim=True) + 1e-12)
        noise_unit = noise_wave / n_rms

        mix_v = voicing_target.unsqueeze(1)
        mix_n = (1.0 - mix_v) + breath_amt.unsqueeze(1) * 0.45
        mixed = mix_v * voiced_unit + mix_n * noise_unit

        # Target singing RMS across corpus range [-30 dB .. -6 dB] correlated with lip radiation opening
        loud_u = torch.empty((b,), device=dev).uniform_(0.08, 1.0, generator=gen)
        open_scale = torch.clamp((lip_t / 1.5) ** 0.25, 0.35, 1.15)
        target_rms = (0.008 * ((0.35 / 0.008) ** (loud_u * open_scale))).unsqueeze(1)
        m_rms = torch.sqrt(torch.mean(mixed * mixed, dim=1, keepdim=True) + 1e-12)
        audio_batch = (mixed / m_rms) * target_rms

        feats_t, env_t, rms_t = _extract_gpu_features_and_env(audio_batch, TARGET_SR)
        # Small realistic intra-frame spectral flux sample for natural singing variation
        feats_t[:, 8] = torch.from_numpy(rng.uniform(0.0, 0.35, size=(b,)).astype(np.float32)).to(dev)

        out_feats[s:e] = feats_t.cpu().numpy()
        out_env[s:e] = env_t.cpu().numpy()
        f0_n = torch.clamp((torch.log2(f0_hz) - math.log2(130.0)) / (math.log2(520.0) - math.log2(130.0)), 0.0, 1.0)
        vib_s = torch.clamp(torch.from_numpy(rng.beta(1.5, 4.5, size=(b,)).astype(np.float32)).to(dev) * voicing_target, 0.0, 1.0)
        out_cues[s:e] = torch.stack([f0_n, voicing_target, breath_amt, vib_s, loud_u, rms_t], dim=1).cpu().numpy()

    return art_rep, out_feats, out_env, out_cues


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Voice Workshop simulator random soundbank.")
    parser.add_argument("--postures", type=int, default=65536, help="Number of random 14-D tract postures.")
    parser.add_argument("--excitations", type=int, default=2, help="Synthesized sounds per posture.")
    parser.add_argument("--out", type=Path, default=Path("/tmp/wub_soundbank.npz"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = _resolve_device(args.device)
    print(f"[Soundbank] Sampling {args.postures} random 14-D postures from Voice Workshop geometry...")
    art_14, areas_44, dx, lip_area, min_area, min_idx_norm = _node_compile_and_sample(args.postures)

    # Filter unstable extreme area explosions while keeping broad vowels, nasals, sibilants
    valid = (
        np.all(np.isfinite(areas_44), axis=1)
        & np.all(np.isfinite(art_14), axis=1)
        & (lip_area >= 0.01)
    )
    art_14 = art_14[valid]
    areas_44 = areas_44[valid]
    dx = dx[valid]
    lip_area = lip_area[valid]
    min_area = min_area[valid]
    min_idx_norm = min_idx_norm[valid]

    print(
        f"[Soundbank] Synthesizing {len(art_14) * args.excitations} real 44.1kHz vocal waveforms on {dev}..."
    )
    art_rep, feats_18, env_32, cues_6 = synthesize_soundbank(
        art_14=art_14,
        areas_44=areas_44,
        dx=dx,
        lip_area=lip_area,
        min_area=min_area,
        min_idx_norm=min_idx_norm,
        excitations_per_posture=args.excitations,
        dev=dev,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        art=art_rep.astype(np.float32),
        feats=feats_18.astype(np.float32),
        env=env_32.astype(np.float32),
        cues=cues_6.astype(np.float32),
    )
    print(f"[Soundbank] Saved {len(art_rep)} simulated sounds -> {args.out}")


if __name__ == "__main__":
    main()
