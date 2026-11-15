#!/usr/bin/env python3
"""Generate an exact 14-parameter vocal-tract physics atlas from Voice Workshop geometry.

Compiles the TypeScript vocal-tract geometry and 44-section area function
(`frontend/src/vocal_sim/tract/{types,geometry,areaFunction}.ts`) into a
temporary directory, samples ~10,000 anatomically valid vowel, consonant, and
smooth transitional postures across all 14 articulators, computes the exact
44-section Webster transmission-line resonance formants (F1, F2, F3) and
cross-sectional apertures (lip, minimum, pharynx, oral), and stores the
resulting `(articulators, physics)` pairs in an `.npz` file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NODE_ATLAS_SCRIPT = r"""
const fs = require('fs');
const path = require('path');

const tractDir = process.argv[2];
const outJsonPath = process.argv[3];
const count = parseInt(process.argv[4] || '10000', 10);

const {
  ARTICULATOR_KEYS,
  ARTICULATOR_RANGES,
  DEFAULT_ANATOMY,
  DEFAULT_TEETH,
  DEFAULT_ARTICULATORS,
} = require(path.join(tractDir, 'types.js'));
const { computeGeometry } = require(path.join(tractDir, 'geometry.js'));
const { sectionsFromGeometry } = require(path.join(tractDir, 'areaFunction.js'));

// Deterministic PRNG (Mulberry32)
let seed = 0x57554231;
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
function mixArt(a, b, t) {
  const out = {};
  for (const k of ARTICULATOR_KEYS) {
    const va = a[k] ?? DEFAULT_ARTICULATORS[k];
    const vb = b[k] ?? DEFAULT_ARTICULATORS[k];
    out[k] = va * (1 - t) + vb * t;
  }
  return out;
}

// Curated canonical phonetic anchors spanning vowels, nasals, fricatives, stops, liquids
const ANCHORS = [
  // Rest / neutral schwa
  { ...DEFAULT_ARTICULATORS },
  // /a/ AH (open back-central)
  { ...DEFAULT_ARTICULATORS, jaw: 0.72, tongueX: -0.78, tongueY: -0.92, tongueRoot: -0.52, tipX: -0.25, tipY: -0.10, groove: 0.14, lipOpen: 0.80, lipRound: 0.08, lipProtrude: 0.0, larynx: -0.10 },
  // /i/ EE (high front)
  { ...DEFAULT_ARTICULATORS, jaw: 0.22, tongueX: 0.92, tongueY: 1.02, tongueRoot: 0.62, tipX: 0.58, tipY: 0.45, groove: 0.22, lipOpen: 0.34, lipRound: 0.0, lipProtrude: -0.22, larynx: 0.18 },
  // /u/ OO (high back rounded)
  { ...DEFAULT_ARTICULATORS, jaw: 0.24, tongueX: -0.88, tongueY: 0.88, tongueRoot: 0.35, tipX: -0.42, tipY: 0.08, groove: 0.24, lipOpen: 0.16, lipRound: 0.92, lipProtrude: 0.82, larynx: -0.35 },
  // /o/ OH (mid-back rounded)
  { ...DEFAULT_ARTICULATORS, jaw: 0.46, tongueX: -0.82, tongueY: 0.05, tongueRoot: -0.18, tipX: -0.30, tipY: -0.05, groove: 0.18, lipOpen: 0.44, lipRound: 0.78, lipProtrude: 0.52, larynx: -0.20 },
  // /e/ AY (mid-front)
  { ...DEFAULT_ARTICULATORS, jaw: 0.38, tongueX: 0.68, tongueY: 0.48, tongueRoot: 0.30, tipX: 0.38, tipY: 0.24, groove: 0.16, lipOpen: 0.48, lipRound: 0.02, lipProtrude: -0.10, larynx: 0.08 },
  // /ae/ AE (low front)
  { ...DEFAULT_ARTICULATORS, jaw: 0.64, tongueX: 0.42, tongueY: -0.45, tongueRoot: -0.15, tipX: 0.18, tipY: -0.15, groove: 0.12, lipOpen: 0.68, lipRound: 0.0, lipProtrude: -0.12, larynx: 0.0 },
  // /m/ bilabial nasal
  { ...DEFAULT_ARTICULATORS, jaw: 0.12, tongueX: 0.0, tongueY: -0.05, tongueRoot: 0.05, tipX: 0.05, tipY: 0.05, lipOpen: -0.34, lipRound: 0.10, lipProtrude: 0.05, velum: 0.90, larynx: -0.12 },
  // /n/ alveolar nasal
  { ...DEFAULT_ARTICULATORS, jaw: 0.20, tongueX: 0.42, tongueY: 0.55, tongueRoot: 0.25, tipX: 0.85, tipY: 1.55, lipOpen: 0.30, lipRound: 0.05, velum: 0.90, larynx: 0.0 },
  // /ng/ velar nasal
  { ...DEFAULT_ARTICULATORS, jaw: 0.28, tongueX: -0.65, tongueY: 1.25, tongueRoot: 0.15, tipX: -0.25, tipY: 0.0, lipOpen: 0.34, lipRound: 0.05, velum: 0.92, larynx: -0.08 },
  // /s/ alveolar sibilant (grooved tip near alveolar ridge)
  { ...DEFAULT_ARTICULATORS, jaw: 0.14, tongueX: 0.55, tongueY: 0.72, tongueRoot: 0.32, tipX: 0.95, tipY: 1.62, groove: 0.88, tongueSides: 0.82, lipOpen: 0.24, lipRound: 0.04, lipProtrude: 0.0, velum: 0.0, larynx: 0.10 },
  // /sh/ postalveolar fricative
  { ...DEFAULT_ARTICULATORS, jaw: 0.18, tongueX: 0.35, tongueY: 0.88, tongueRoot: 0.42, tipX: 0.52, tipY: 1.25, groove: 0.65, tongueSides: 0.72, lipOpen: 0.28, lipRound: 0.58, lipProtrude: 0.56, velum: 0.0, larynx: 0.05 },
  // /f/ /v/ labiodental fricative
  { ...DEFAULT_ARTICULATORS, jaw: 0.16, tongueX: 0.05, tongueY: 0.05, tipX: 0.12, tipY: 0.05, lipOpen: 0.04, lipRound: 0.0, lipProtrude: -0.25, lipTuck: 0.88, velum: 0.0, larynx: 0.0 },
  // /l/ lateral approximant
  { ...DEFAULT_ARTICULATORS, jaw: 0.32, tongueX: 0.48, tongueY: 0.35, tongueRoot: 0.18, tipX: 0.88, tipY: 1.65, groove: 0.10, tongueSides: 0.08, lipOpen: 0.40, lipRound: 0.08, velum: 0.0, larynx: 0.0 },
  // /r/ rhotic approximant (tip curled, lowered jaw/lips slightly rounded)
  { ...DEFAULT_ARTICULATORS, jaw: 0.30, tongueX: -0.15, tongueY: 0.38, tongueRoot: -0.25, tipX: 0.35, tipY: 1.20, groove: 0.42, tongueSides: 0.65, lipOpen: 0.30, lipRound: 0.45, lipProtrude: 0.35, velum: 0.0, larynx: -0.15 },
  // /k/ /g/ velar constriction
  { ...DEFAULT_ARTICULATORS, jaw: 0.26, tongueX: -0.78, tongueY: 1.40, tongueRoot: 0.20, tipX: -0.45, tipY: -0.10, groove: 0.10, lipOpen: 0.36, lipRound: 0.10, velum: 0.0, larynx: 0.0 },
  // /b/ /p/ bilabial closure
  { ...DEFAULT_ARTICULATORS, jaw: 0.10, tongueX: 0.0, tongueY: 0.0, tipX: 0.0, tipY: 0.0, lipOpen: -0.36, lipRound: 0.08, lipProtrude: 0.05, velum: 0.0, larynx: 0.0 },
  // Belt / high larynx bright vowel
  { ...DEFAULT_ARTICULATORS, jaw: 0.85, tongueX: 0.25, tongueY: -0.25, tongueRoot: 0.45, tipX: 0.15, tipY: 0.05, groove: 0.15, lipOpen: 0.90, lipRound: 0.05, lipProtrude: -0.15, velum: 0.06, larynx: 0.85 },
  // Deep / low larynx dark vowel
  { ...DEFAULT_ARTICULATORS, jaw: 0.55, tongueX: -0.72, tongueY: 0.10, tongueRoot: -0.55, tipX: -0.35, tipY: -0.15, groove: 0.20, lipOpen: 0.40, lipRound: 0.65, lipProtrude: 0.60, velum: 0.0, larynx: -1.10 },
];

// Webster 2x2 chain-matrix transfer function -> [F1, F2, F3] in Hz
const RHO = 1.14e-3;
const C = 35000.0;
const ALPHA = 0.006;
const FREQ_MIN = 150;
const FREQ_MAX = 4300;
const FREQ_STEP = 10;
const N_FREQS = Math.floor((FREQ_MAX - FREQ_MIN) / FREQ_STEP) + 1;
const magBuf = new Float64Array(N_FREQS);

function computeFormantsHz(areas, dx, lipArea) {
  const n = areas.length;
  const effLipA = Math.max(0.05, lipArea);
  const rLip = Math.sqrt(Math.max(0.01, effLipA) / Math.PI);
  const Lrad = (8 * RHO) / (3 * Math.PI * Math.PI * rLip);
  const coshAlpha = Math.cosh(ALPHA * dx);
  const sinhAlpha = Math.sinh(ALPHA * dx);

  for (let fi = 0; fi < N_FREQS; fi++) {
    const f = FREQ_MIN + fi * FREQ_STEP;
    const w = 2 * Math.PI * f;
    const k = w / C;
    const cosK = Math.cos(k * dx);
    const sinK = Math.sin(k * dx);
    const coshRe = coshAlpha * cosK;
    const coshIm = sinhAlpha * sinK;
    const sinhRe = sinhAlpha * cosK;
    const sinhIm = coshAlpha * sinK;

    const pRe0 = 0.015 * ((RHO * C) / effLipA) * Math.pow(w / 6000, 2);
    const pIm0 = w * Lrad;
    let pRe = pRe0;
    let pIm = pIm0;
    let uRe = 1.0;
    let uIm = 0.0;

    for (let s = n - 1; s >= 0; s--) {
      const A = Math.max(0.03, areas[s]);
      const Z = (RHO * C) / A;
      // U_new = cosh*U + (sinh/Z)*P
      const suRe = coshRe * uRe - coshIm * uIm;
      const suIm = coshRe * uIm + coshIm * uRe;
      const spRe = (sinhRe * pRe - sinhIm * pIm) / Z;
      const spIm = (sinhRe * pIm + sinhIm * pRe) / Z;
      const uNextRe = suRe + spRe;
      const uNextIm = suIm + spIm;
      // P_new = (Z*sinh)*U + cosh*P
      const zsuRe = Z * (sinhRe * uRe - sinhIm * uIm);
      const zsuIm = Z * (sinhRe * uIm + sinhIm * uRe);
      const cpRe = coshRe * pRe - coshIm * pIm;
      const cpIm = coshRe * pIm + coshIm * pRe;
      uRe = uNextRe;
      uIm = uNextIm;
      pRe = zsuRe + cpRe;
      pIm = zsuIm + cpIm;
    }
    const uMag = Math.hypot(uRe, uIm);
    magBuf[fi] = 1.0 / Math.max(1e-12, uMag);
  }

  const peaks = [];
  for (let i = 1; i < N_FREQS - 1; i++) {
    if (magBuf[i] > magBuf[i - 1] && magBuf[i] >= magBuf[i + 1]) {
      const fPeak = FREQ_MIN + i * FREQ_STEP;
      if (peaks.length === 0 || fPeak - peaks[peaks.length - 1] >= 180) {
        peaks.push(fPeak);
        if (peaks.length >= 3) break;
      }
    }
  }
  while (peaks.length < 3) {
    peaks.push(peaks.length === 0 ? 500 : peaks.length === 1 ? 1500 : 2500);
  }
  return peaks;
}

const artRows = [];
const physRows = [];

for (let idx = 0; idx < count; idx++) {
  let art;
  if (idx < ANCHORS.length) {
    art = clampArt(ANCHORS[idx]);
  } else if (idx < Math.floor(count * 0.55)) {
    // Smooth interpolation between two anchors + small anatomical jitter
    const a = ANCHORS[Math.floor(rand() * ANCHORS.length)];
    const b = ANCHORS[Math.floor(rand() * ANCHORS.length)];
    const t = rand();
    art = mixArt(a, b, t);
    for (const k of ARTICULATOR_KEYS) {
      const [lo, hi] = ARTICULATOR_RANGES[k];
      const span = hi - lo;
      art[k] += randn() * 0.06 * span;
    }
    art = clampArt(art);
  } else if (idx < Math.floor(count * 0.92)) {
    // Anchor perturbation at medium scale (covers broad neighborhood around phonetic modes)
    const a = ANCHORS[Math.floor(rand() * ANCHORS.length)];
    art = { ...a };
    for (const k of ARTICULATOR_KEYS) {
      const [lo, hi] = ARTICULATOR_RANGES[k];
      const span = hi - lo;
      art[k] += randn() * 0.18 * span;
    }
    art = clampArt(art);
  } else {
    // Full uniform coverage across the 14-D hypercube
    art = {};
    for (const k of ARTICULATOR_KEYS) {
      const [lo, hi] = ARTICULATOR_RANGES[k];
      art[k] = lo + rand() * (hi - lo);
    }
  }

  const g = computeGeometry(art, DEFAULT_ANATOMY, DEFAULT_TEETH);
  const af = sectionsFromGeometry(g);
  const areas = af.areas;
  const nSec = areas.length;
  const [f1, f2, f3] = computeFormantsHz(areas, af.dx, g.lipArea);

  let minArea = 1e9;
  let pharynxSum = 0;
  let pharynxCnt = 0;
  let oralSum = 0;
  let oralCnt = 0;
  for (let s = 0; s < nSec; s++) {
    const ar = areas[s];
    if (ar < minArea) minArea = ar;
    if (s < Math.floor(nSec * 0.38)) {
      pharynxSum += ar;
      pharynxCnt++;
    } else if (s < Math.floor(nSec * 0.86)) {
      oralSum += ar;
      oralCnt++;
    }
  }
  const pharynxArea = pharynxSum / Math.max(1, pharynxCnt);
  const oralArea = oralSum / Math.max(1, oralCnt);

  artRows.push(ARTICULATOR_KEYS.map((k) => art[k]));
  physRows.push([f1, f2, f3, g.lipArea, minArea, pharynxArea, oralArea]);
}

fs.writeFileSync(
  outJsonPath,
  JSON.stringify({
    art_names: ARTICULATOR_KEYS,
    phys_names: ['f1_hz', 'f2_hz', 'f3_hz', 'lip_area', 'min_area', 'pharynx_area', 'oral_area'],
    art_ranges: ARTICULATOR_RANGES,
    art: artRows,
    phys: physRows,
  })
);
console.log(`Generated ${artRows.length} vocal-tract physics atlas samples.`);
"""


def generate_atlas(out_path: Path, count: int = 10000) -> Path:
    """Compile Voice Workshop TS tract modules and generate the `.npz` physics atlas."""
    import numpy as np

    frontend_dir = ROOT / "frontend"
    tract_ts_files = [
        frontend_dir / "src/vocal_sim/tract/types.ts",
        frontend_dir / "src/vocal_sim/tract/section3d.ts",
        frontend_dir / "src/vocal_sim/tract/geometry.ts",
        frontend_dir / "src/vocal_sim/tract/areaFunction.ts",
    ]

    with tempfile.TemporaryDirectory(prefix="wub_tract_atlas_") as tmp_str:
        tmp_dir = Path(tmp_str)
        js_out_dir = tmp_dir / "js"
        js_out_dir.mkdir(parents=True, exist_ok=True)
        node_script = tmp_dir / "eval_atlas.js"
        raw_json = tmp_dir / "atlas_raw.json"
        node_script.write_text(NODE_ATLAS_SCRIPT, encoding="utf-8")

        tsc_cmd = [
            "npx",
            "--prefix",
            str(frontend_dir),
            "tsc",
            "--outDir",
            str(js_out_dir),
            "--module",
            "commonjs",
            "--target",
            "es2020",
            "--moduleResolution",
            "node",
            "--esModuleInterop",
            "--skipLibCheck",
            *[str(p) for p in tract_ts_files],
        ]
        subprocess.run(tsc_cmd, check=True, cwd=str(ROOT))

        # Locate compiled tract/types.js regardless of tsconfig rootDir behavior
        compiled_tract_dir: Path | None = None
        for candidate in [js_out_dir, js_out_dir / "tract", js_out_dir / "src/vocal_sim/tract"]:
            if (candidate / "types.js").exists():
                compiled_tract_dir = candidate
                break
        if compiled_tract_dir is None:
            for p in js_out_dir.rglob("types.js"):
                compiled_tract_dir = p.parent
                break
        if compiled_tract_dir is None:
            raise FileNotFoundError(f"Could not locate compiled tract/types.js inside {js_out_dir}")

        subprocess.run(
            ["node", str(node_script), str(compiled_tract_dir), str(raw_json), str(count)],
            check=True,
            cwd=str(ROOT),
        )

        import json

        payload = json.loads(raw_json.read_text(encoding="utf-8"))
        art = np.asarray(payload["art"], dtype=np.float32)
        phys = np.asarray(payload["phys"], dtype=np.float32)
        art_names = np.asarray(payload["art_names"])
        phys_names = np.asarray(payload["phys_names"])
        art_ranges = np.asarray(
            [[payload["art_ranges"][k][0], payload["art_ranges"][k][1]] for k in payload["art_names"]],
            dtype=np.float32,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            art=art,
            phys=phys,
            art_names=art_names,
            phys_names=phys_names,
            art_ranges=art_ranges,
        )
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Voice Workshop 14-D vocal tract physics atlas")
    parser.add_argument("--out", type=Path, default=Path("/tmp/tract_physics_atlas.npz"))
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()

    out = generate_atlas(args.out, count=args.count)
    import numpy as np

    d = np.load(out)
    art = d["art"]
    phys = d["phys"]
    assert np.isfinite(art).all() and np.isfinite(phys).all(), "Non-finite values in atlas!"
    print(f"Saved {out}: art={art.shape}, phys={phys.shape}")
    print(f"F1 range: [{phys[:, 0].min():.0f}, {phys[:, 0].max():.0f}] Hz")
    print(f"F2 range: [{phys[:, 1].min():.0f}, {phys[:, 1].max():.0f}] Hz")
    print(f"F3 range: [{phys[:, 2].min():.0f}, {phys[:, 2].max():.0f}] Hz")


if __name__ == "__main__":
    main()
