// Shared mapping between Wub's 14-D Voice Workshop articulators + glottal state
// and CHIMERA's 17-D human airway articulators + 12-D glottal source vector.
// Matches crates/chimera/examples/soundbank_gen.rs bit-for-bit.

import type { SingingHeadArticulation } from "../singing_head_model";
import { ARTICULATOR_KEYS, ARTICULATOR_RANGES, type Articulators } from "./tract/types";

export const CHIMERA_ARTIC_LEN = 17;
export const CHIMERA_SOURCE_LEN = 12;

export const STUDIO_MIC_EQ = {
  highpassHz: 110.0,
  highpassQ: 0.7,
  peakHz: 2800.0,
  peakQ: 0.8,
  peakGainDb: 4.5,
  highshelfHz: 1200.0,
  highshelfQ: 0.7,
  highshelfGainDb: 3.0,
} as const;

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function lerpNorm(u: number, lo: number, hi: number): number {
  return lo + clamp(u, 0, 1) * (hi - lo);
}

/**
 * Convert Wub's 14-D Articulators + glottal/acoustic state into Chimera's
 * native `engine_set_artic` (17 values) and `engine_set_source` (12 values).
 */
export function mapWubToChimeraArrays(
  tractArt: Articulators,
  art: SingingHeadArticulation,
  isMuted: boolean,
  smoothedF0: number,
  centroidDrive: number,
  outArtic17: Float64Array,
  outSource12: Float64Array
): { gate: number; femaleMix: number; openDrive: number } {
  const u: number[] = new Array(14);
  for (let i = 0; i < 14; i++) {
    const key = ARTICULATOR_KEYS[i];
    const [lo, hi] = ARTICULATOR_RANGES[key];
    u[i] = clamp((tractArt[key] - lo) / Math.max(1e-6, hi - lo), 0, 1);
  }

  // Chimera ArticKey::ALL order:
  // 0 jaw, 1 tbX, 2 tbY, 3 ttX, 4 ttY, 5 ttCurl, 6 root, 7 lipAp, 8 lipProt,
  // 9 lipWidth, 10 labiodental, 11 velum, 12 larynx, 13 epilarynx,
  // 14 ventricular, 15 lateral, 16 dorsum
  outArtic17[0] = lerpNorm(u[0], 0.08, 1.95);        // Jaw
  outArtic17[1] = clamp(1.05 + 2.85 * u[1] - 0.48 * u[0] - 0.55 * u[9], 1.20, 3.82); // TbX
  outArtic17[2] = lerpNorm(u[2], 5.15, 7.95);        // TbY
  outArtic17[3] = lerpNorm(u[4], 4.80, 7.10);        // TtX
  outArtic17[4] = lerpNorm(u[5], 5.30, 8.10);        // TtY
  outArtic17[5] = 0.0;                               // TtCurl (neutral)
  outArtic17[6] = lerpNorm(u[3], -0.75, 0.75);       // Root
  outArtic17[7] = Math.max(-0.15, lerpNorm(u[8], 0.15, 2.05) - 0.35 * clamp(u[9], 0, 1)); // LipAp
  outArtic17[8] = lerpNorm(u[10], -0.15, 1.40);      // LipProt
  outArtic17[9] = 1.0 - clamp(u[9], 0, 1);           // LipWidth = 1 - lipRound
  outArtic17[10] = clamp(u[11], 0, 1) * 0.85;        // Labiodental
  outArtic17[11] = lerpNorm(u[12], -0.18, 0.95);     // Velum
  outArtic17[12] = lerpNorm(u[13], -1.10, 1.10);     // Larynx
  outArtic17[13] = clamp(u[6], 0, 1) * 0.55;         // Epilarynx (groove/twang)
  outArtic17[14] = 0.0;                              // Ventricular
  outArtic17[15] = clamp(u[7], 0, 1);                // Lateral (tongueSides)
  outArtic17[16] = 0.0;                              // Dorsum

  const openDrive = clamp(0.50 * u[0] + 0.35 * u[8] + 0.15 * (1.0 - u[9] * 0.4), 0.02, 1.0);

  const tension = clamp(art.tension, 0, 1);
  const breath = clamp(art.breath, 0, 1);
  const voicing = clamp(art.voicing, 0, 1);
  const vibGate = clamp(voicing * Math.max(0, (art.rms - 0.008) * 24), 0, 1);
  const vibDepthSt = 0.42 * clamp(art.vibrato, 0, 1) * vibGate;
  const f0 = clamp(smoothedF0, 65, 820);
  const energyDrive = clamp(Math.sqrt(clamp(art.rms * 6.5, 0, 1)), 0, 1);

  const active = !isMuted && art.rms >= 0.0025;
  const pressure = active
    ? clamp(4.2 + 6.8 * tension + 3.2 * energyDrive, 2.5, 15.5)
    : 0.0;
  const breathiness = clamp(0.02 + 0.68 * (1.0 - voicing) + 0.25 * breath + 0.22 * centroidDrive, 0.01, 0.96);
  const brightness = clamp(0.08 + 0.78 * centroidDrive + 0.14 * tension, 0.02, 1.0);
  const voiced = voicing > 0.36 ? 1.0 : 0.0;

  // Chimera source layout:
  // [0 f0, 1 glide, 2 pressure, 3 breathiness, 4 tension, 5 fry, 6 roughness,
  //  7 vibratoDepth, 8 vibratoRate, 9 drift, 10 voiced, 11 brightness]
  outSource12[0] = f0;
  outSource12[1] = 0.008;
  outSource12[2] = pressure;
  outSource12[3] = breathiness;
  outSource12[4] = clamp(0.15 + 0.70 * tension, 0.05, 0.95);
  outSource12[5] = 0.0;
  outSource12[6] = 0.06;
  outSource12[7] = vibDepthSt;
  outSource12[8] = 5.6;
  outSource12[9] = 0.03;
  outSource12[10] = voiced;
  outSource12[11] = brightness;

  const f0Mix = clamp((f0 - 135.0) / 145.0, 0.0, 1.0);
  const femaleMix = clamp(0.75 * f0Mix + 0.25 * centroidDrive, 0.0, 1.0);

  return {
    gate: active ? 1.0 : 0.0,
    femaleMix,
    openDrive,
  };
}
