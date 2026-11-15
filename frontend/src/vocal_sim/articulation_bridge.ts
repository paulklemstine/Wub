// Connects Wub's reverse-trained SingingHeadArticulation outputs to the
// Voice Workshop mid-sagittal & 3D vocal-tract geometry model.

import { SingingHeadArticulation } from "../singing_head_model";
import { N_SECTIONS } from "./tract/areaFunction";
import { TractGeometry } from "./tract/geometry";
import {
  ARTICULATOR_RANGES,
  Articulators,
  DEFAULT_ANATOMY,
  DEFAULT_ARTICULATORS,
  DEFAULT_TEETH,
  Anatomy,
  Teeth,
} from "./tract/types";

export interface VocalFieldState {
  pRms: Float32Array;
  uRms: Float32Array;
  ug: number;
  uMouth: number;
  uNose: number;
  nasalPort: number;
  glottalArea: number;
  constrictionIndex: number;
  voicingAmp: number;
  f0: number;
  phone: string;
}

// Canonical vowel postures fitted in Voice Workshop (src/dsp/speech/phonemes.ts)
// matched to SINGING_VOWELS = ["AH", "EE", "OH", "OO", "AY"] plus rest schwa.
const VOWEL_POSTURES: readonly Articulators[] = [
  // AH (ɑ / a)
  {
    ...DEFAULT_ARTICULATORS,
    jaw: 0.90,
    tongueX: 0.08,
    tongueY: -0.52,
    tongueRoot: -0.08,
    lipOpen: 0.56,
    lipRound: 0.0,
    lipProtrude: -0.10,
  },
  // EE (i)
  {
    ...DEFAULT_ARTICULATORS,
    jaw: 0.143,
    tongueX: 1.095,
    tongueY: 0.83,
    tongueRoot: 1.2,
    lipOpen: -0.05,
    lipRound: 0.19,
    lipProtrude: -0.35,
  },
  // OH (o)
  {
    ...DEFAULT_ARTICULATORS,
    jaw: 0.40,
    tongueX: -0.698,
    tongueY: 0.565,
    tongueRoot: 0.666,
    lipOpen: -0.012,
    lipRound: 0.99,
    lipProtrude: 0.40,
  },
  // OO (u)
  {
    ...DEFAULT_ARTICULATORS,
    jaw: 0.072,
    tongueX: 0.347,
    tongueY: 1.136,
    tongueRoot: 0.243,
    lipOpen: 0.01,
    lipRound: 1.0,
    lipProtrude: 0.963,
  },
  // AY (e)
  {
    ...DEFAULT_ARTICULATORS,
    jaw: 0.465,
    tongueX: 0.80,
    tongueY: 1.06,
    tongueRoot: -0.183,
    lipOpen: 0.779,
    lipRound: 0.0,
    lipProtrude: -0.346,
  },
];

const REST_POSTURE: Articulators = {
  ...DEFAULT_ARTICULATORS,
  jaw: 0.14,
  tongueX: 0.12,
  tongueY: 0.58,
  tongueRoot: 0.35,
  lipOpen: -0.04,
  lipRound: 0.08,
  lipProtrude: 0.02,
  velum: 0.08,
  larynx: -0.1,
};

function clampArt(key: keyof Articulators, v: number): number {
  const [lo, hi] = ARTICULATOR_RANGES[key];
  return v < lo ? lo : v > hi ? hi : v;
}

export function defaultAnatomy(): Anatomy {
  return { ...DEFAULT_ANATOMY };
}

export function defaultTeeth(): Teeth {
  return { ...DEFAULT_TEETH };
}

/**
 * Map reverse-trained 14-parameter Voice Workshop articulators (when available)
 * directly to the vocal tract, or fall back to 5-vowel posterior blending.
 */
export function articulationToTract(
  art: SingingHeadArticulation,
  timeSec: number,
  out: Articulators
): Articulators {
  const vibratoOsc = Math.sin(timeSec * 36) * 0.08 * art.vibrato;
  if (art.tract) {
    const t = art.tract;
    out.jaw = clampArt("jaw", t.jaw);
    out.tongueX = clampArt("tongueX", t.tongueX);
    out.tongueY = clampArt("tongueY", t.tongueY + vibratoOsc);
    out.tongueRoot = clampArt("tongueRoot", t.tongueRoot);
    out.tipX = clampArt("tipX", t.tipX);
    out.tipY = clampArt("tipY", t.tipY);
    out.groove = clampArt("groove", t.groove);
    out.tongueSides = clampArt("tongueSides", t.tongueSides);
    out.lipOpen = clampArt("lipOpen", t.lipOpen);
    out.lipRound = clampArt("lipRound", t.lipRound);
    out.lipProtrude = clampArt("lipProtrude", t.lipProtrude);
    out.lipTuck = clampArt("lipTuck", t.lipTuck);
    out.velum = clampArt("velum", t.velum);
    out.larynx = clampArt("larynx", t.larynx + vibratoOsc * 0.55);
    return out;
  }

  const probs = art.vowelProbs;
  let wSum = 0;
  let bJaw = 0;
  let bTx = 0;
  let bTy = 0;
  let bTr = 0;
  let bLipOpen = 0;
  let bLipRound = 0;
  let bLipPro = 0;

  for (let i = 0; i < VOWEL_POSTURES.length; i++) {
    const w = probs[i] ?? 0;
    const p = VOWEL_POSTURES[i];
    wSum += w;
    bJaw += w * p.jaw;
    bTx += w * p.tongueX;
    bTy += w * p.tongueY;
    bTr += w * p.tongueRoot;
    bLipOpen += w * p.lipOpen;
    bLipRound += w * p.lipRound;
    bLipPro += w * p.lipProtrude;
  }
  if (wSum > 1e-4) {
    const inv = 1 / wSum;
    bJaw *= inv;
    bTx *= inv;
    bTy *= inv;
    bTr *= inv;
    bLipOpen *= inv;
    bLipRound *= inv;
    bLipPro *= inv;
  } else {
    bJaw = REST_POSTURE.jaw;
    bTx = REST_POSTURE.tongueX;
    bTy = REST_POSTURE.tongueY;
    bTr = REST_POSTURE.tongueRoot;
    bLipOpen = REST_POSTURE.lipOpen;
    bLipRound = REST_POSTURE.lipRound;
    bLipPro = REST_POSTURE.lipProtrude;
  }

  // Activity weight blends smoothly from resting posture into full vowel articulation
  const act = Math.max(0, Math.min(1, art.aperture * 1.15 + art.tension * 0.55 + art.voicing * 0.35));

  const targetJaw = 0.55 * bJaw + 0.50 * art.jawDrop + 0.15 * art.aperture;
  const targetLipOpen = 0.55 * bLipOpen + 0.72 * (art.aperture - 0.18);
  const targetLipRound = 0.50 * bLipRound + 0.65 * art.roundness - 0.25 * art.width;
  const targetLipPro = 0.45 * bLipPro + 0.85 * art.roundness - 0.50 * art.width;
  const targetTy = bTy + (art.tongueHeight - 0.5) * 0.65 + vibratoOsc;

  out.jaw = clampArt("jaw", REST_POSTURE.jaw + act * (targetJaw - REST_POSTURE.jaw));
  out.tongueX = clampArt("tongueX", REST_POSTURE.tongueX + act * (bTx - REST_POSTURE.tongueX));
  out.tongueY = clampArt("tongueY", REST_POSTURE.tongueY + act * (targetTy - REST_POSTURE.tongueY));
  out.tongueRoot = clampArt("tongueRoot", REST_POSTURE.tongueRoot + act * (bTr - REST_POSTURE.tongueRoot));
  out.tipX = 0;
  out.tipY = clampArt("tipY", act * Math.max(0, art.breath - 0.35) * 0.45);
  out.groove = clampArt("groove", art.breath * 0.4);
  out.tongueSides = 0;
  out.lipOpen = clampArt("lipOpen", REST_POSTURE.lipOpen + act * (targetLipOpen - REST_POSTURE.lipOpen));
  out.lipRound = clampArt("lipRound", REST_POSTURE.lipRound + act * (targetLipRound - REST_POSTURE.lipRound));
  out.lipProtrude = clampArt("lipProtrude", REST_POSTURE.lipProtrude + act * (targetLipPro - REST_POSTURE.lipProtrude));
  out.lipTuck = 0;
  out.velum = clampArt("velum", 0.04 + 0.34 * art.breath * act + 0.08 * art.roundness * act);
  out.larynx = clampArt("larynx", (art.pitchNorm - 0.48) * 1.65 * act + vibratoOsc * 0.6);

  return out;
}

export function createVocalFieldState(): VocalFieldState {
  return {
    pRms: new Float32Array(N_SECTIONS),
    uRms: new Float32Array(N_SECTIONS),
    ug: 0,
    uMouth: 0,
    uNose: 0,
    nasalPort: 0,
    glottalArea: 0.06,
    constrictionIndex: 0,
    voicingAmp: 0,
    f0: 0,
    phone: "",
  };
}

const VOWEL_TO_IPA: Record<string, string> = {
  AH: "ɑ",
  EE: "i",
  OH: "o",
  OO: "u",
  AY: "e",
};

/**
 * Populate standing-wave pressure, volume-velocity & airflow telemetry from
 * live vocal audio + current vocal tract geometry so the air glow & particles match Voice Workshop.
 */
export function updateVocalFieldState(
  art: SingingHeadArticulation,
  tractArt: Articulators,
  geom: TractGeometry,
  timeSec: number,
  out: VocalFieldState
): VocalFieldState {
  const act = Math.max(0, Math.min(1, art.tension * 0.7 + art.aperture * 0.4 + Math.min(1, art.rms * 6.5) * 0.4));
  const voicing = Math.max(0, Math.min(1, art.voicing * act * 1.15));
  const velumPort = Math.max(0, tractArt.velum * 1.4);

  let minArea = Infinity;
  let minIdx = 0;
  for (let i = 0; i <= geom.lastRay; i++) {
    const a = geom.hits[i].area;
    if (a < minArea) {
      minArea = a;
      minIdx = Math.floor((geom.s[i] / Math.max(1e-3, geom.length)) * (N_SECTIONS - 1));
    }
  }

  const f0 = art.f0Hz > 40 ? art.f0Hz : 130;
  const f1Phase = 0.5 + 0.5 * Math.sin(timeSec * 8 + art.aperture * 2.5);
  for (let k = 0; k < N_SECTIONS; k++) {
    const frac = k / (N_SECTIONS - 1); // 0 = glottis, 1 = lips
    // Fundamental quarter-wave pressure max at glottis, velocity max at lips, plus F2 spatial ripple
    const pWave = Math.cos(frac * Math.PI * 0.5) * 0.72 + Math.abs(Math.cos(frac * Math.PI * 1.5)) * 0.28 * f1Phase;
    const uWave = Math.sin(frac * Math.PI * 0.5) * 0.75 + Math.abs(Math.sin(frac * Math.PI * 1.5)) * 0.25;
    out.pRms[k] = 0.08 + act * (0.25 + 0.70 * pWave);
    out.uRms[k] = 0.08 + act * (0.20 + 0.75 * uWave) * (0.65 + 0.35 * art.breath);
  }

  const flow = act * (90 + 260 * Math.max(voicing, art.breath * 0.8));
  out.ug = flow;
  const nasalShare = velumPort > 0.02 ? Math.min(0.65, velumPort / (velumPort + Math.max(0.2, geom.lipArea))) : 0;
  out.uMouth = flow * (1 - nasalShare * 0.7);
  out.uNose = flow * nasalShare * 1.3;
  out.nasalPort = velumPort;
  out.glottalArea = 0.05 + 0.19 * Math.max(voicing, art.breath * 0.7);
  out.constrictionIndex = minIdx;
  out.voicingAmp = voicing;
  out.f0 = f0;
  out.phone = act > 0.14 ? (VOWEL_TO_IPA[art.topVowel] ?? art.topVowel) : "";
  return out;
}
