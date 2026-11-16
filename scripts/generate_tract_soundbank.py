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
function natRand() {
  // Smooth, natural vocal distribution covering the full physiological singing range
  // without clustering at 0.5 or pinning to screaming extremes
  return 0.72 * rand() + 0.28 * 0.5 * (rand() + rand());
}
for (let i = 0; i < count; i++) {
  // Sample coupled, natural singing vocal simulator drives across ARTICULATOR_RANGES
  const uOpen = natRand();               // 0 = closed mouth, 1 = full sung vowel opening
  const uFront = natRand() * 2 - 1;      // -1 = back tongue, +1 = front tongue
  const uHigh = natRand() * 2 - 1;       // -1 = low oral cavity, +1 = high palatal arch
  const uRound = natRand();              // 0 = spread lips, 1 = rounded lips
  const uTip = natRand();                // tongue tip & alveolar constriction drive
  const uLarynx = natRand() * 2 - 1;

  const cand = {
    jaw: 0.02 + 0.64 * uOpen + randn() * 0.05,
    lipOpen: -0.22 + 0.88 * uOpen * (1.0 - 0.30 * uRound) + randn() * 0.06,
    lipRound: 0.88 * uRound * (1.0 - 0.25 * Math.max(0, uFront)) + randn() * 0.06,
    lipProtrude: -0.28 + 0.98 * uRound - 0.20 * Math.max(0, uFront) + randn() * 0.06,
    tongueX: 1.15 * uFront + randn() * 0.12,
    tongueY: 1.15 * uHigh * (1.0 - 0.35 * uOpen) - 0.38 * uOpen + randn() * 0.12,
    tongueRoot: 0.70 * uFront - 0.35 * uOpen + randn() * 0.12,
    tipX: 0.82 * uFront + 0.38 * (uTip - 0.4) + randn() * 0.12,
    tipY: -0.28 + 1.75 * uTip * (1.0 - 0.42 * uOpen) + randn() * 0.12,
    groove: 0.12 + 0.72 * uTip * Math.max(0, 1.0 - uOpen) + randn() * 0.06,
    tongueSides: 0.15 + 0.68 * Math.max(0, uHigh * 0.5 + uTip * 0.5) + randn() * 0.06,
    lipTuck: rand() < 0.18 ? (0.18 + 0.52 * natRand()) : (rand() * 0.10),
    velum: rand() < 0.20 ? (0.18 + 0.55 * natRand()) : (rand() * 0.08),
    larynx: 0.92 * uLarynx + randn() * 0.10,
  };

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


def _compute_oral_transfer_grid(
    areas_44: torch.Tensor,
    dx: torch.Tensor,
    lip_area: torch.Tensor,
    freqs_hz: torch.Tensor,
) -> torch.Tensor:
    """Compute pure 44-section Webster oral transfer magnitude |H_oral(B, F)| matching evaluateTractAcoustics."""
    rho = 1.14e-3
    c_snd = 35000.0
    alpha = 0.0028

    w = (2.0 * math.pi * freqs_hz).unsqueeze(0)
    k = w / c_snd
    dx_b = torch.clamp(dx, min=0.2).unsqueeze(1)
    kdx = k * dx_b
    cos_k = torch.cos(kdx)
    sin_k = torch.sin(kdx)
    cosh_a = torch.cosh(alpha * dx_b)
    sinh_a = torch.sinh(alpha * dx_b)

    cosh_re = cosh_a * cos_k
    cosh_im = sinh_a * sin_k
    sinh_re = sinh_a * cos_k
    sinh_im = cosh_a * sin_k

    eff_lip = torch.clamp(lip_area, min=0.04).unsqueeze(1)
    r_lip = torch.sqrt(torch.clamp(eff_lip / math.pi, min=0.01))
    L_rad = (8.0 * rho) / (3.0 * (math.pi ** 2) * r_lip)
    p_re = 0.015 * ((rho * c_snd) / eff_lip) * ((w / 6000.0) ** 2)
    p_im = w * L_rad
    u_re = torch.ones_like(p_re)
    u_im = torch.zeros_like(p_re)

    clamped_areas = torch.clamp(areas_44, min=0.025)
    for s in range(43, -1, -1):
        Z = ((rho * c_snd) / clamped_areas[:, s]).unsqueeze(1)
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
    return torch.clamp(1.0 / u_mag, max=40.0)


def _evaluate_tract_acoustics_batch(
    art_14: torch.Tensor,
    areas_44: torch.Tensor,
    dx: torch.Tensor,
    lip_area: torch.Tensor,
    min_area: torch.Tensor,
    min_idx_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized PyTorch match to frontend/src/vocal_sim/tract_synth.ts::evaluateTractAcoustics."""
    dev = art_14.device
    bsz = art_14.shape[0]
    n_grid = 80
    u = torch.linspace(0.0, 1.0, n_grid, device=dev, dtype=torch.float32)
    freq_grid = 75.0 * ((9200.0 / 75.0) ** u)
    h_vocal = _compute_oral_transfer_grid(areas_44, dx, lip_area, freq_grid)  # (B, 80)

    is_local_max = torch.zeros_like(h_vocal, dtype=torch.bool)
    is_local_max[:, 1:-1] = (h_vocal[:, 1:-1] >= h_vocal[:, :-2]) & (h_vocal[:, 1:-1] >= h_vocal[:, 2:])

    bands = (
        (190.0, 920.0, 520.0, 0.0),
        (680.0, 2750.0, 1480.0, 260.0),
        (1850.0, 3850.0, 2650.0, 320.0),
        (2950.0, 5100.0, 3650.0, 320.0),
    )
    f_hz = torch.zeros((bsz, 4), device=dev, dtype=torch.float32)
    f_q = torch.zeros((bsz, 4), device=dev, dtype=torch.float32)
    f_gain = torch.zeros((bsz, 4), device=dev, dtype=torch.float32)

    prev_hz = torch.zeros((bsz,), device=dev, dtype=torch.float32)
    for k, (b_lo, b_hi, fb_hz, min_sp) in enumerate(bands):
        lo_hz = torch.clamp(prev_hz + min_sp, min=b_lo).unsqueeze(1)  # (B, 1)
        in_band = (freq_grid.unsqueeze(0) >= lo_hz) & (freq_grid.unsqueeze(0) <= b_hi)
        loc_mask = in_band & is_local_max
        has_loc = loc_mask.any(dim=1)
        loc_scores = torch.where(loc_mask, h_vocal, torch.full_like(h_vocal, -1.0))
        any_scores = torch.where(in_band, h_vocal, torch.full_like(h_vocal, -1.0))
        scores = torch.where(has_loc.unsqueeze(1), loc_scores, any_scores)
        best_val, best_idx = torch.max(scores, dim=1)
        valid_peak = in_band.any(dim=1) & (best_val > -0.5)

        idx_c = torch.clamp(best_idx, 1, n_grid - 2)
        y0 = torch.log(torch.clamp(h_vocal.gather(1, (idx_c - 1).unsqueeze(1)).squeeze(1), min=1e-6))
        y1 = torch.log(torch.clamp(h_vocal.gather(1, idx_c.unsqueeze(1)).squeeze(1), min=1e-6))
        y2 = torch.log(torch.clamp(h_vocal.gather(1, (idx_c + 1).unsqueeze(1)).squeeze(1), min=1e-6))
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        interior = (best_idx > 0) & (best_idx < n_grid - 1) & (denom > 1e-6)
        delta = torch.where(interior, torch.clamp((y2 - y0) / torch.clamp(denom, min=1e-6), -0.5, 0.5), torch.zeros_like(best_val))
        f0_g = freq_grid[torch.clamp(best_idx - 1, 0, n_grid - 1)]
        f2_g = freq_grid[torch.clamp(best_idx + 1, 0, n_grid - 1)]
        f_peak_raw = freq_grid[best_idx] + delta * 0.5 * (f2_g - f0_g)
        f_peak = torch.where(
            valid_peak,
            torch.clamp(torch.maximum(f_peak_raw, lo_hz.squeeze(1)), max=b_hi),
            torch.clamp(lo_hz.squeeze(1), min=fb_hz),
        )

        m2 = h_vocal.gather(1, torch.clamp(best_idx - 2, 0, n_grid - 1).unsqueeze(1)).squeeze(1)
        p2 = h_vocal.gather(1, torch.clamp(best_idx + 2, 0, n_grid - 1).unsqueeze(1)).squeeze(1)
        neigh = 0.5 * (m2 + p2)
        sharp = torch.clamp(best_val / torch.clamp(neigh, min=0.05), 1.0, 5.0)
        q_val = torch.clamp((4.2 + k * 1.6) + (sharp - 1.0) * 2.2, 3.5, 16.0)
        g_val = torch.clamp(best_val / 10.0, 0.12, 1.65)

        f_hz[:, k] = f_peak
        f_q[:, k] = torch.where(valid_peak, q_val, torch.full_like(q_val, 5.5 + k * 1.6))
        f_gain[:, k] = torch.where(valid_peak, g_val, torch.full_like(g_val, 1.0 if k == 0 else 0.75 if k == 1 else 0.45 if k == 2 else 0.28))
        prev_hz = f_peak

    jaw_n = torch.clamp(art_14[:, 0], 0.0, 1.0)
    lip_open_n = torch.clamp((art_14[:, 8] + 0.4) / 1.6, 0.0, 1.0)
    lip_area_n = torch.clamp(torch.clamp(lip_area, min=0.04) / 4.0, 0.0, 1.0)
    open_drive = torch.clamp(0.50 * jaw_n + 0.35 * lip_open_n + 0.15 * lip_area_n, 0.02, 1.0)
    sib_center = 2400.0 + 4900.0 * torch.clamp((min_idx_norm - 0.45) / 0.50, 0.0, 1.0)
    tightness = torch.clamp((0.36 - min_area) / 0.32, 0.0, 1.0)
    tuck_amt = torch.clamp((torch.clamp(art_14[:, 11], 0.0, 1.0) - 0.20) / 0.70, 0.0, 1.0)
    fric_gain = torch.clamp(tightness + 0.65 * tuck_amt, 0.0, 1.2)
    nasal_gain = torch.clamp(art_14[:, 12], 0.0, 1.0)
    return f_hz, f_q, f_gain, open_drive, sib_center, fric_gain, nasal_gain


def _biquad_resp_torch(
    ftype: str,
    f0: torch.Tensor,
    q_val: torch.Tensor | float,
    gain_db: torch.Tensor | float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    sr: float = float(TARGET_SR),
) -> torch.Tensor:
    """Exact W3C Web Audio BiquadFilterNode complex frequency response on GPU."""
    f0_c = torch.clamp(f0, 20.0, 0.49 * sr).unsqueeze(1)
    w0 = (2.0 * math.pi / sr) * f0_c
    cw = torch.cos(w0)
    sw = torch.sin(w0)
    q_t = (torch.clamp(q_val, min=0.05).unsqueeze(1) if isinstance(q_val, torch.Tensor) else max(0.05, float(q_val)))
    alpha = sw / (2.0 * q_t)
    z1_r = z1.unsqueeze(0)
    z2_r = z2.unsqueeze(0)
    if ftype == "lowpass":
        b0 = (1.0 - cw) * 0.5
        b1 = 1.0 - cw
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
    elif ftype == "highpass":
        b0 = (1.0 + cw) * 0.5
        b1 = -(1.0 + cw)
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
    elif ftype == "bandpass":
        b0 = alpha
        b1 = torch.zeros_like(alpha)
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
    elif ftype == "peaking":
        g_t = (gain_db.unsqueeze(1) if isinstance(gain_db, torch.Tensor) else float(gain_db))
        A = torch.pow(torch.tensor(10.0, device=f0.device, dtype=torch.float32), g_t / 40.0)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cw
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cw
        a2 = 1.0 - alpha / A
    elif ftype == "highshelf":
        g_t = (gain_db.unsqueeze(1) if isinstance(gain_db, torch.Tensor) else float(gain_db))
        A = torch.pow(torch.tensor(10.0, device=f0.device, dtype=torch.float32), g_t / 40.0)
        al_s = sw * 0.5 * torch.sqrt(torch.clamp((A + 1.0 / A) * (1.0 / 0.7 - 1.0) + 2.0, min=1e-6))
        sq_a = 2.0 * torch.sqrt(A) * al_s
        b0 = A * ((A + 1.0) + (A - 1.0) * cw + sq_a)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cw)
        b2 = A * ((A + 1.0) + (A - 1.0) * cw - sq_a)
        a0 = (A + 1.0) - (A - 1.0) * cw + sq_a
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cw)
        a2 = (A + 1.0) - (A - 1.0) * cw - sq_a
    else:
        raise ValueError(f"Unsupported biquad type: {ftype}")
    num = b0.to(torch.complex64) + b1.to(torch.complex64) * z1_r + b2.to(torch.complex64) * z2_r
    den = a0.to(torch.complex64) + a1.to(torch.complex64) * z1_r + a2.to(torch.complex64) * z2_r
    return num / torch.where(den.abs() > 1e-8, den, torch.ones_like(den))


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
    h_oral = _compute_oral_transfer_grid(areas_44, dx, lip_area, freqs_hz)

    v = torch.clamp(velum, 0.0, 1.0).unsqueeze(1)
    f_row = freqs_hz.unsqueeze(0)
    nasal_murmur = 2.6 * torch.exp(-0.5 * ((f_row - 245.0) / 115.0) ** 2) + 0.85 * torch.exp(
        -0.5 * ((f_row - 1050.0) / 260.0) ** 2
    )
    oral_antires = 1.0 - 0.55 * v * torch.exp(-0.5 * ((f_row - 960.0) / 280.0) ** 2)
    h_vocal = h_oral * oral_antires * (1.0 - 0.52 * v) + v * nasal_murmur * (
        0.5 + 0.5 * torch.clamp(h_oral.mean(dim=1, keepdim=True), 0.1, 4.0)
    )

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
    batch_size: int = 512,
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
    w_grid = (2.0 * math.pi / float(TARGET_SR)) * freqs_full
    z1_grid = torch.complex(torch.cos(-w_grid), torch.sin(-w_grid))
    z2_grid = z1_grid * z1_grid
    t_axis = torch.arange(FFT_SIZE, device=dev, dtype=torch.float32) / float(TARGET_SR)

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + 13)

    harmonics = torch.arange(1, 64, device=dev, dtype=torch.float32)  # (63,)
    h_amps_base = 1.0 / (harmonics ** 0.72)
    h_phases_base = harmonics * 0.38 + torch.where((harmonics.to(torch.int64) & 1) == 1, 0.15, -0.15)
    h_body_base = _biquad_resp_torch("lowpass", torch.tensor([280.0], device=dev), 0.9, 0.0, z1_grid, z2_grid)  # (1, F)
    h_nasal_base = _biquad_resp_torch("bandpass", torch.tensor([245.0], device=dev), 3.2, 0.0, z1_grid, z2_grid)

    with torch.no_grad():
      for s in range(0, total_sounds, batch_size):
        e = min(total_sounds, s + batch_size)
        b = e - s

        a_t = torch.from_numpy(art_rep[s:e]).to(dev)
        ar_t = torch.from_numpy(areas_rep[s:e]).to(dev)
        dx_t = torch.from_numpy(dx_rep[s:e]).to(dev)
        lip_t = torch.from_numpy(lip_rep[s:e]).to(dev)
        mina_t = torch.from_numpy(min_a_rep[s:e]).to(dev)
        mini_t = torch.from_numpy(min_i_rep[s:e]).to(dev)

        larynx_t = a_t[:, 13]
        liptuck_t = torch.clamp(a_t[:, 11], 0.0, 1.0)

        # Exact evaluateTractAcoustics() parameters matching frontend/src/vocal_sim/tract_synth.ts
        f_hz, f_q, f_gain, open_drive, sib_center, fric_gain, nasal_gain = _evaluate_tract_acoustics_batch(
            a_t, ar_t, dx_t, lip_t, mina_t, mini_t
        )

        # Sample realistic singing excitation across full vocal pitch, brightness, and voicing range
        larynx_semis = larynx_t * 6.5
        log_f0 = torch.empty((b,), device=dev).uniform_(math.log2(95.0), math.log2(580.0), generator=gen)
        f0_hz = torch.clamp(2.0 ** (log_f0 + larynx_semis / 12.0 * 0.45), 75.0, 760.0)

        tightness = torch.clamp((0.30 - mina_t) / 0.28, 0.0, 1.0)
        tuck_w = torch.clamp((liptuck_t - 0.25) / 0.65, 0.0, 1.0)
        is_fric_dom = torch.rand((b,), device=dev, generator=gen) < (0.08 + 0.65 * torch.maximum(tightness, tuck_w))

        voicing_target = torch.where(
            is_fric_dom,
            torch.empty((b,), device=dev).uniform_(0.04, 0.35, generator=gen),
            torch.empty((b,), device=dev).uniform_(0.48, 0.98, generator=gen),
        )
        breath_amt = torch.clamp(
            torch.maximum(tightness, tuck_w) * 0.65 + (1.0 - voicing_target) * 0.55 + torch.rand((b,), device=dev, generator=gen) * 0.14,
            0.0,
            1.0,
        )
        tension = torch.clamp(0.65 * open_drive + 0.35 * torch.rand((b,), device=dev, generator=gen), 0.0, 1.0)
        front_drive = torch.clamp((a_t[:, 1] + 1.6) / 3.0, 0.0, 1.0)
        centroid_drive = torch.clamp(0.55 * open_drive + 0.25 * front_drive + 0.20 * torch.rand((b,), device=dev, generator=gen), 0.0, 1.0)

        # Exact VocalTractSynthesizer Web Audio filterbank complex transfer functions
        tilt_cutoff = torch.clamp(
            1150.0 + 2100.0 * open_drive + 3600.0 * centroid_drive + 1100.0 * tension, 850.0, 9200.0
        )
        H_tilt = _biquad_resp_torch("lowpass", tilt_cutoff, 0.7, 0.0, z1_grid, z2_grid)
        hp_hz = torch.clamp(130.0 + 400.0 * centroid_drive, 120.0, 560.0)
        H_hp = _biquad_resp_torch("highpass", hp_hz, 0.7, 0.0, z1_grid, z2_grid)
        H_cas = torch.ones((b, n_fft_bins), device=dev, dtype=torch.complex64)
        H_par = torch.zeros((b, n_fft_bins), device=dev, dtype=torch.complex64)
        for k_f in range(4):
            fk = f_hz[:, k_f]
            qk = f_q[:, k_f]
            gk = f_gain[:, k_f]
            cas_db = torch.clamp(6.0 + gk * 9.5 - k_f * 1.0, 2.0, 18.0)
            H_cas = H_cas * _biquad_resp_torch("peaking", fk, torch.clamp(qk * 0.65, 2.5, 11.0), cas_db, z1_grid, z2_grid)
            sgn = -1.0 if (k_f & 1) == 1 else 1.0
            par_w = (sgn * gk * (0.65 / (1.0 + k_f * 0.35))).unsqueeze(1).to(torch.complex64)
            H_par = H_par + par_w * _biquad_resp_torch("bandpass", fk, qk, 0.0, z1_grid, z2_grid)

        body_lvl = (0.06 * torch.clamp(1.0 - centroid_drive, 0.0, 1.0)).unsqueeze(1).to(torch.complex64)
        H_body = body_lvl * h_body_base
        H_nas = (nasal_gain * 0.35).unsqueeze(1).to(torch.complex64) * h_nasal_base
        H_notch = _biquad_resp_torch("peaking", torch.full((b,), 960.0, device=dev), 2.2, -8.5 * nasal_gain, z1_grid, z2_grid)
        rad_db = torch.clamp(-3.0 + 7.5 * open_drive + 12.0 * centroid_drive, -4.0, 15.5)
        H_rad = _biquad_resp_torch("highshelf", torch.full((b,), 650.0, device=dev), 0.7, rad_db, z1_grid, z2_grid)
        H_tract = H_hp * H_rad * (H_notch * (0.55 * H_cas + 0.85 * H_par + H_body) + H_nas)
        H_voiced = H_tilt * H_tract
        fric_amt = torch.clamp(fric_gain * (0.22 + 0.65 * breath_amt + 0.45 * (1.0 - voicing_target)), 0.0, 0.65)
        H_fric = fric_amt.unsqueeze(1).to(torch.complex64) * _biquad_resp_torch("bandpass", sib_center, 2.4, 0.0, z1_grid, z2_grid)

        # Synthesize glottal PeriodicWave + subOsc (63 harmonics) and filter through H_voiced + calibrated noise
        phase0 = torch.empty((b, 1), device=dev).uniform_(0.0, 2.0 * math.pi, generator=gen)
        harm_freqs = f0_hz.unsqueeze(1) * harmonics.unsqueeze(0)  # (B, 63)
        nyq_mask = (harm_freqs < (TARGET_SR * 0.46)).to(torch.float32)
        harm_amp = h_amps_base.unsqueeze(0) * nyq_mask
        angles = 2.0 * math.pi * harm_freqs.unsqueeze(2) * t_axis.view(1, 1, -1) + phase0.unsqueeze(2) + h_phases_base.view(1, -1, 1)
        glottal = torch.sum(harm_amp.unsqueeze(2) * torch.sin(angles), dim=1)
        g_peak = torch.clamp(torch.amax(torch.abs(glottal), dim=1, keepdim=True), min=1e-6)
        sub_lvl = (0.14 * torch.clamp(1.0 - 0.85 * centroid_drive, 0.05, 1.0)).unsqueeze(1)
        glottal_tot = glottal / g_peak + sub_lvl * torch.sin(2.0 * math.pi * f0_hz.unsqueeze(1) * t_axis.view(1, -1) + phase0)

        v_lvl = torch.clamp(0.10 + 0.90 * voicing_target, 0.08, 1.05).unsqueeze(1)
        asp_lvl = torch.clamp((1.0 - voicing_target) * 0.34 + breath_amt * 0.22, 0.01, 0.45).unsqueeze(1).to(torch.complex64)

        # Match Web Audio Uniform-pink noise RMS (~0.38) relative to PeriodicWave
        white = torch.randn((b, FFT_SIZE), device=dev, generator=gen) * 0.38
        g_spec = torch.fft.rfft(v_lvl * glottal_tot, dim=1)
        n_spec = torch.fft.rfft(white, dim=1)
        mixed = torch.fft.irfft(g_spec * H_voiced + n_spec * (asp_lvl * H_tract + H_fric), n=FFT_SIZE, dim=1)

        loud_u = torch.clamp(0.72 * open_drive + 0.28 * torch.rand((b,), device=dev, generator=gen), 0.03, 1.0)
        target_rms = (0.004 * ((0.36 / 0.004) ** loud_u)).unsqueeze(1)
        m_rms = torch.sqrt(torch.mean(mixed * mixed, dim=1, keepdim=True) + 1e-12)
        audio_batch = (mixed / m_rms) * target_rms

        feats_t, env_t, rms_t = _extract_gpu_features_and_env(audio_batch, TARGET_SR)
        feats_t[:, 8] = torch.from_numpy(rng.uniform(0.0, 0.35, size=(b,)).astype(np.float32)).to(dev)

        out_feats[s:e] = feats_t.cpu().numpy()
        out_env[s:e] = env_t.cpu().numpy()
        f0_n = torch.clamp((torch.log2(f0_hz) - math.log2(130.0)) / (math.log2(520.0) - math.log2(130.0)), 0.0, 1.0)
        vib_s = torch.clamp(torch.from_numpy(rng.beta(1.5, 4.5, size=(b,)).astype(np.float32)).to(dev) * voicing_target, 0.0, 1.0)
        out_cues[s:e] = torch.stack([f0_n, voicing_target, breath_amt, vib_s, loud_u, rms_t], dim=1).cpu().numpy()
        if e == total_sounds or ((s // batch_size) % 32 == 0):
            print(f"[Soundbank] {e}/{total_sounds} synthesized", flush=True)

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

    with torch.no_grad():
        pf_hz, pf_q, pf_gain, p_od, p_sib, p_fric, _ = _evaluate_tract_acoustics_batch(
            torch.from_numpy(art_14).to(dev),
            torch.from_numpy(areas_44).to(dev),
            torch.from_numpy(dx).to(dev),
            torch.from_numpy(lip_area).to(dev),
            torch.from_numpy(min_area).to(dev),
            torch.from_numpy(min_idx_norm).to(dev),
        )
        prof_1 = torch.cat(
            [pf_hz, pf_q, pf_gain, p_od.unsqueeze(1), p_sib.unsqueeze(1), p_fric.unsqueeze(1)], dim=1
        ).cpu().numpy().astype(np.float32)
    prof_rep = np.repeat(prof_1, args.excitations, axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        art=art_rep.astype(np.float32),
        feats=feats_18.astype(np.float32),
        env=env_32.astype(np.float32),
        cues=cues_6.astype(np.float32),
        profile=prof_rep,
    )
    print(f"[Soundbank] Saved {len(art_rep)} simulated sounds -> {args.out}")


if __name__ == "__main__":
    main()
