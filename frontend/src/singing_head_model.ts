// -----------------------------------------------------------------------------
// Singing Head Reverse-Trained Acoustic-to-Articulatory Inference
// Exact TypeScript parity with scripts/train_singing_head.py.
// Converts live 2048-sample vocal AnalyserNode frames into 9 articulators +
// 5 vowel posteriors + 14 Voice Workshop mouth/throat parameters using the
// reverse-trained vocal stub/stem MLP weights.
// -----------------------------------------------------------------------------

import {
  Articulators,
  ARTICULATOR_KEYS,
  ARTICULATOR_RANGES,
  DEFAULT_ARTICULATORS,
} from "./vocal_sim/tract/types";

export const SINGING_VOWELS = ["AH", "EE", "OH", "OO", "AY"] as const;
export type SingingVowel = (typeof SINGING_VOWELS)[number];

export const SINGING_FFT_SIZE = 2048;

const BANDS_HZ: readonly (readonly [number, number])[] = [
  [80, 250],
  [250, 500],
  [500, 850],
  [850, 1350],
  [1350, 2000],
  [2000, 3000],
  [3000, 5000],
  [5000, 9000],
];

export interface SingingHeadModelWeights {
  version: number;
  format: string;
  sample_rate: number;
  fft_size: number;
  feature_mean: number[];
  feature_std: number[];
  corpus_stats?: {
    stubs_used?: number;
    stems_used?: number;
    train_frames?: number;
    val_frames?: number;
    val_mae?: number;
    tract_val_mae?: number;
    val_vowel_acc?: number;
    cycle_r2?: number;
    physics_forward_r2?: number;
    physics_cycle_r2?: number;
    roundtrip_env_corr?: number;
    roundtrip_feat_r2?: number;
    db_p10?: number;
    db_p95?: number;
    f0_p10?: number;
    f0_p95?: number;
  };
  tract_keys?: string[];
  tract_ranges?: [number, number][];
  weights: {
    w1: number[][];
    b1: number[];
    w2: number[][];
    b2: number[];
    w3: number[][];
    b3: number[];
  };
}

export interface SingingHeadArticulation {
  aperture: number;      // 0..1 vertical mouth opening
  width: number;         // 0..1 horizontal lip spread vs pucker
  roundness: number;     // 0..1 O/U lip protrusion
  jawDrop: number;       // 0..1 lower jaw hinge displacement
  tongueHeight: number;  // 0..1 high vs low tongue
  pitchNorm: number;     // 0..1 normalized pitch within corpus range
  vibrato: number;       // 0..1 vibrato / pitch modulation
  tension: number;       // 0..1 vocal belt / intensity
  breath: number;        // 0..1 unvoiced air / sibilance
  vowelProbs: readonly number[]; // [AH, EE, OH, OO, AY]
  topVowel: SingingVowel;
  /** All 14 reverse-trained Voice Workshop mouth & throat articulators (when available / inferred). */
  tract?: Articulators;
  rms: number;
  f0Hz: number;
  voicing: number;
  centroidHz: number;
  trained: boolean;
}

const TRACT_REST_RAW: readonly number[] = [
  0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02, 0.0,
];

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/** High-accuracy erf approximation (max error < 1.5e-7) for exact PyTorch GELU parity. */
function erfApprox(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t) *
      Math.exp(-ax * ax);
  return sign * y;
}

function gelu(x: number): number {
  return 0.5 * x * (1 + erfApprox(x * Math.SQRT1_2));
}

export function isValidSingingHeadModel(raw: unknown): raw is SingingHeadModelWeights {
  if (!raw || typeof raw !== "object") return false;
  const obj = raw as Record<string, unknown>;
  if (obj.format !== "wub_singing_head_mlp_v1" && obj.format !== "wub_singing_head_mlp_v2") {
    return false;
  }
  if (!Array.isArray(obj.feature_mean) || obj.feature_mean.length !== 18) return false;
  if (!Array.isArray(obj.feature_std) || obj.feature_std.length !== 18) return false;
  const w = obj.weights as Record<string, unknown> | undefined;
  if (!w || typeof w !== "object") return false;
  if (!Array.isArray(w.w1) || !Array.isArray(w.b1) || !Array.isArray(w.w2) || !Array.isArray(w.b2) || !Array.isArray(w.w3) || !Array.isArray(w.b3)) {
    return false;
  }
  const h1 = w.b1.length;
  const h2 = w.b2.length;
  const out = w.b3.length;
  if (w.w1.length !== h1 || w.w2.length !== h2 || w.w3.length !== out) return false;
  if (obj.format === "wub_singing_head_mlp_v1") {
    return h1 === 48 && h2 === 32 && out === 14;
  }
  return h1 >= 16 && h2 >= 16 && out === 28;
}

/**
 * Stateful, zero-allocation 60 FPS feature extractor + reverse-trained MLP evaluator.
 */
export class SingingHeadInference {
  private model: SingingHeadModelWeights | null = null;

  // Preallocated scratch buffers (2048-point FFT + decimated autocorrelation)
  private readonly hann = new Float64Array(SINGING_FFT_SIZE);
  private readonly centered = new Float64Array(SINGING_FFT_SIZE);
  private readonly decimated = new Float64Array(SINGING_FFT_SIZE >> 1);
  private readonly fftRe = new Float64Array(SINGING_FFT_SIZE);
  private readonly fftIm = new Float64Array(SINGING_FFT_SIZE);
  private readonly mag = new Float64Array(SINGING_FFT_SIZE >> 1);
  private readonly magSq = new Float64Array(SINGING_FFT_SIZE >> 1);
  private prevNormMag: Float64Array | null = null;
  private prevNormMagLen = 0;

  private readonly features = new Float32Array(18);
  private readonly xNorm = new Float64Array(18);
  private h1 = new Float64Array(64);
  private h2 = new Float64Array(48);
  private rawOut = new Float64Array(28);
  private readonly vowelProbs = new Float64Array(5);

  constructor() {
    for (let i = 0; i < SINGING_FFT_SIZE; i++) {
      this.hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (SINGING_FFT_SIZE - 1));
    }
  }

  public setModel(model: SingingHeadModelWeights | null): void {
    this.model = model;
    if (model) {
      if (this.h1.length < model.weights.b1.length) this.h1 = new Float64Array(model.weights.b1.length);
      if (this.h2.length < model.weights.b2.length) this.h2 = new Float64Array(model.weights.b2.length);
      if (this.rawOut.length < model.weights.b3.length) this.rawOut = new Float64Array(model.weights.b3.length);
    }
  }

  public getModel(): SingingHeadModelWeights | null {
    return this.model;
  }

  public isTrained(): boolean {
    return this.model !== null;
  }

  public resetFluxHistory(): void {
    this.prevNormMagLen = 0;
  }

  /**
   * Extract the 18-d acoustic feature vector matching scripts/train_singing_head.py.
   */
  public extractFeatures(
    samples: Float32Array,
    sampleRate: number
  ): {
    features: Float32Array;
    rms: number;
    f0Hz: number;
    voicing: number;
    centroidHz: number;
  } {
    const n = SINGING_FFT_SIZE;
    const srcLen = samples.length;

    let mean = 0;
    if (srcLen === n) {
      for (let i = 0; i < n; i++) mean += samples[i];
    } else if (srcLen > 0) {
      for (let i = 0; i < n; i++) {
        const idx = Math.min(srcLen - 1, Math.floor((i / n) * srcLen));
        mean += samples[idx];
      }
    }
    mean /= n;

    let sumSq = 0;
    let peak = 0;
    let signChanges = 0;
    let prevSign = false;

    for (let i = 0; i < n; i++) {
      const s =
        srcLen === n
          ? samples[i]
          : srcLen > 0
            ? samples[Math.min(srcLen - 1, Math.floor((i / n) * srcLen))]
            : 0;
      const v = s - mean;
      this.centered[i] = v;
      sumSq += v * v;
      const av = v < 0 ? -v : v;
      if (av > peak) peak = av;
      const sign = v >= 0;
      if (i > 0 && sign !== prevSign) signChanges++;
      prevSign = sign;
    }

    const rms = Math.sqrt(sumSq / n);
    const zcr = signChanges / (n - 1);

    // 2x decimated autocorrelation for voicing & F0
    const m = n >> 1;
    const srDec = sampleRate / 2;
    let power = 0;
    for (let i = 0; i < m; i++) {
      const v = this.centered[i << 1];
      this.decimated[i] = v;
      power += v * v;
    }

    let bestLag = 0;
    let bestVal = 0;
    if (power > 1e-6) {
      const minLag = Math.max(8, Math.floor(srDec / 800));
      const maxLag = Math.min(m - 1, Math.ceil(srDec / 80));
      const invPower = 1 / Math.max(1e-9, power);
      for (let lag = minLag; lag <= maxLag; lag++) {
        const len = m - lag;
        let acc = 0;
        for (let i = 0; i < len; i++) {
          acc += this.decimated[i] * this.decimated[i + lag];
        }
        const corr = acc * invPower;
        const denom = Math.max(0.01, 1 - lag / m);
        const score = corr / denom;
        if (score > bestVal) {
          bestVal = score;
          bestLag = lag;
        }
      }
    }

    const voicing = clamp(bestVal, 0, 1);
    const f0Hz = bestLag > 0 && voicing > 0.30 && rms > 0.008 ? srDec / bestLag : 0;
    const f0Norm = f0Hz > 0 ? clamp(Math.log2(f0Hz / 220), -1.5, 1.5) : 0;

    // Hann-windowed 2048-point radix-2 FFT
    for (let i = 0; i < n; i++) {
      this.fftRe[i] = this.centered[i] * this.hann[i];
      this.fftIm[i] = 0;
    }
    this.fftInPlace(this.fftRe, this.fftIm, n);

    const halfN = n >> 1;
    const invNorm = 1 / (n * 0.5);
    for (let k = 0; k < halfN; k++) {
      const re = this.fftRe[k];
      const im = this.fftIm[k];
      const mVal = Math.sqrt(re * re + im * im) * invNorm;
      this.mag[k] = mVal;
      this.magSq[k] = mVal * mVal;
    }

    const binHz = sampleRate / n;
    const fluxHi = Math.min(halfN - 1, Math.ceil(9000 / binHz));
    const fluxLen = Math.max(0, fluxHi);
    const scaleDenom = rms + 1e-4;
    if (!this.prevNormMag || this.prevNormMag.length < fluxLen) {
      this.prevNormMag = new Float64Array(fluxLen);
      this.prevNormMagLen = 0;
    }
    let flux = 0;
    if (this.prevNormMagLen === fluxLen && fluxLen > 0) {
      let fluxSum = 0;
      for (let k = 1; k <= fluxHi; k++) {
        const cur = this.mag[k] / scaleDenom;
        const prev = this.prevNormMag[k - 1];
        if (cur > prev) fluxSum += cur - prev;
        this.prevNormMag[k - 1] = cur;
      }
      flux = Math.log1p((fluxSum / fluxLen) * 2.0);
    } else {
      for (let k = 1; k <= fluxHi; k++) {
        this.prevNormMag[k - 1] = this.mag[k] / scaleDenom;
      }
      this.prevNormMagLen = fluxLen;
    }

    // Spectral centroid (100..8000 Hz)
    const cLo = Math.max(1, Math.floor(100 / binHz));
    const cHi = Math.min(halfN - 1, Math.ceil(8000 / binHz));
    let cNum = 0;
    let cDen = 0;
    for (let k = cLo; k <= cHi; k++) {
      const e = this.magSq[k];
      cNum += k * binHz * e;
      cDen += e;
    }
    const centroidHz = cDen > 1e-12 ? cNum / cDen : 500;
    const centroidLog = clamp(Math.log2(Math.max(80, centroidHz) / 500), -2.5, 4.0);

    // 85% spectral rolloff (100..9000 Hz)
    const rLo = Math.max(1, Math.floor(100 / binHz));
    const rHi = Math.min(halfN - 1, Math.ceil(9000 / binHz));
    let rSum = 0;
    for (let k = rLo; k <= rHi; k++) rSum += this.magSq[k];
    let rolloffNorm = 0;
    if (rSum > 1e-12) {
      const thresh = 0.85 * rSum;
      let acc = 0;
      let rBin = rHi;
      for (let k = rLo; k <= rHi; k++) {
        acc += this.magSq[k];
        if (acc >= thresh) {
          rBin = k;
          break;
        }
      }
      rolloffNorm = clamp((rBin * binHz) / 9000, 0, 1);
    }

    // Spectral flatness (150..6000 Hz)
    const fLo = Math.max(1, Math.floor(150 / binHz));
    const fHi = Math.min(halfN - 1, Math.ceil(6000 / binHz));
    let logSum = 0;
    let linSum = 0;
    const fCount = Math.max(1, fHi - fLo + 1);
    for (let k = fLo; k <= fHi; k++) {
      const e = this.magSq[k] + 1e-12;
      logSum += Math.log(e);
      linSum += e;
    }
    const flatness = clamp(Math.exp(logSum / fCount) / Math.max(1e-12, linSum / fCount), 0, 1);

    // 8 log-band energies
    const bands = this.features;
    for (let b = 0; b < 8; b++) {
      const [loHz, hiHz] = BANDS_HZ[b];
      const loB = Math.max(1, Math.floor(loHz / binHz));
      const hiB = Math.min(halfN - 1, Math.ceil(hiHz / binHz));
      let bSum = 0;
      const bCount = Math.max(1, hiB - loB + 1);
      for (let k = loB; k <= hiB; k++) bSum += this.magSq[k];
      const bandRms = Math.sqrt(bSum / bCount);
      bands[9 + b] = Math.log1p(bandRms * 32.0);
    }

    bands[0] = Math.log1p(rms * 25.0);
    bands[1] = Math.min(1.5, peak * 1.5);
    bands[2] = zcr;
    bands[3] = voicing;
    bands[4] = f0Norm;
    bands[5] = centroidLog;
    bands[6] = rolloffNorm;
    bands[7] = flatness;
    bands[8] = flux;
    bands[17] = (bands[9] + bands[10] + bands[11]) - (bands[13] + bands[14] + bands[15]);

    return { features: this.features, rms, f0Hz, voicing, centroidHz };
  }

  /**
   * Run reverse-trained MLP forward pass (or analytical fallback before model loads).
   */
  public infer(samples: Float32Array, sampleRate: number): SingingHeadArticulation {
    const { features, rms, f0Hz, voicing, centroidHz } = this.extractFeatures(samples, sampleRate);

    if (this.model !== null) {
      const { feature_mean, feature_std, weights, tract_ranges } = this.model;
      for (let i = 0; i < 18; i++) {
        this.xNorm[i] = (features[i] - feature_mean[i]) / Math.max(1e-4, feature_std[i]);
      }

      const h1Dim = weights.b1.length;
      const h2Dim = weights.b2.length;
      const outDim = weights.b3.length;

      // Layer 1: 18 -> h1Dim (GELU)
      for (let j = 0; j < h1Dim; j++) {
        const row = weights.w1[j];
        let acc = weights.b1[j];
        for (let i = 0; i < 18; i++) acc += row[i] * this.xNorm[i];
        this.h1[j] = gelu(acc);
      }

      // Layer 2: h1Dim -> h2Dim (GELU)
      for (let j = 0; j < h2Dim; j++) {
        const row = weights.w2[j];
        let acc = weights.b2[j];
        for (let i = 0; i < h1Dim; i++) acc += row[i] * this.h1[i];
        this.h2[j] = gelu(acc);
      }

      // Layer 3: h2Dim -> outDim (14 in v1, 28 in v2)
      for (let j = 0; j < outDim; j++) {
        const row = weights.w3[j];
        let acc = weights.b3[j];
        for (let i = 0; i < h2Dim; i++) acc += row[i] * this.h2[i];
        this.rawOut[j] = acc;
      }

      // Silence safety gate: when raw RMS is genuinely silent, gently suppress residual bias
      const silenceScale = clamp((rms - 0.0015) / 0.004, 0, 1);

      const aperture = sigmoid(this.rawOut[0]) * silenceScale;
      const width = 0.5 + (sigmoid(this.rawOut[1]) - 0.5) * silenceScale;
      const roundness = sigmoid(this.rawOut[2]) * silenceScale;
      const jawDrop = sigmoid(this.rawOut[3]) * silenceScale;
      const tongueHeight = 0.55 + (sigmoid(this.rawOut[4]) - 0.55) * silenceScale;
      const pitchNorm = sigmoid(this.rawOut[5]);
      const vibrato = sigmoid(this.rawOut[6]) * silenceScale;
      const tension = sigmoid(this.rawOut[7]) * silenceScale;
      const breath = sigmoid(this.rawOut[8]) * silenceScale;

      let maxLogit = -Infinity;
      for (let k = 0; k < 5; k++) {
        if (this.rawOut[9 + k] > maxLogit) maxLogit = this.rawOut[9 + k];
      }
      let expSum = 0;
      for (let k = 0; k < 5; k++) {
        const e = Math.exp(this.rawOut[9 + k] - maxLogit);
        this.vowelProbs[k] = e;
        expSum += e;
      }
      let bestK = 0;
      let bestP = -1;
      for (let k = 0; k < 5; k++) {
        const p = this.vowelProbs[k] / Math.max(1e-9, expSum);
        this.vowelProbs[k] = p;
        if (p > bestP) {
          bestP = p;
          bestK = k;
        }
      }

      let tract: Articulators | undefined;
      if (outDim >= 28) {
        const outTract: Articulators = { ...DEFAULT_ARTICULATORS };
        for (let k = 0; k < 14; k++) {
          const key = ARTICULATOR_KEYS[k];
          const [lo, hi] =
            tract_ranges && tract_ranges[k] ? tract_ranges[k] : ARTICULATOR_RANGES[key];
          const normVal = sigmoid(this.rawOut[14 + k]);
          const predRaw = lo + normVal * (hi - lo);
          const restRaw = TRACT_REST_RAW[k];
          outTract[key] = clamp(restRaw + (predRaw - restRaw) * silenceScale, lo, hi);
        }
        tract = outTract;
      }

      return {
        aperture,
        width,
        roundness,
        jawDrop,
        tongueHeight,
        pitchNorm,
        vibrato,
        tension,
        breath,
        vowelProbs: [
          this.vowelProbs[0],
          this.vowelProbs[1],
          this.vowelProbs[2],
          this.vowelProbs[3],
          this.vowelProbs[4],
        ],
        topVowel: SINGING_VOWELS[bestK],
        tract,
        rms,
        f0Hz,
        voicing,
        centroidHz,
        trained: true,
      };
    }

    // Analytical fallback before model JSON finishes loading
    const gate = clamp((rms - 0.003) / 0.06, 0, 1);
    const bright = clamp((centroidHz - 700) / 2400, 0, 1);
    const aperture = clamp(gate * 0.85, 0, 1);
    const width = clamp(0.35 + bright * 0.55, 0.1, 0.95);
    const roundness = clamp(gate * (1 - bright) * 0.8, 0, 1);
    const jawDrop = aperture * 0.8;
    const tongueHeight = clamp(0.3 + bright * 0.5, 0, 1);
    const pitchNorm = f0Hz > 0 ? clamp((Math.log2(f0Hz / 130)) / 2.0, 0, 1) : 0.42;
    const breath = clamp(gate * (1 - voicing) * 0.7, 0, 1);
    const topVowel: SingingVowel =
      bright > 0.7 ? "EE" : bright > 0.45 ? "AY" : bright > 0.25 ? "AH" : roundness > 0.45 ? "OO" : "OH";

    const tract: Articulators = {
      ...DEFAULT_ARTICULATORS,
      jaw: clamp(0.12 + jawDrop * 0.78, 0, 1),
      tongueX: clamp((bright - 0.45) * 2.1, -1.6, 1.4),
      tongueY: clamp((tongueHeight - 0.5) * 2.0, -1.6, 1.6),
      tongueRoot: clamp((bright - 0.5) * 1.3 - 0.3 * jawDrop, -1.2, 1.2),
      tipX: clamp((bright - 0.4) * 1.4 + breath * 0.6, -1.2, 1.4),
      tipY: clamp(breath * 1.1, -0.6, 2.2),
      groove: clamp(breath * 0.75, 0, 1),
      tongueSides: clamp(breath * 0.55, 0, 1),
      lipOpen: clamp(0.02 + aperture * 0.95 - 0.3 * breath, -0.4, 1.2),
      lipRound: roundness,
      lipProtrude: clamp(roundness * 0.85 - (width - 0.5) * 0.4, -0.6, 1),
      lipTuck: clamp(breath * (1 - aperture) * 0.4, 0, 1),
      velum: clamp(0.02 + 0.1 * gate * (1 - bright), 0, 1),
      larynx: clamp((pitchNorm - 0.48) * 1.4 * gate, -1.5, 1.2),
    };

    return {
      aperture,
      width,
      roundness,
      jawDrop,
      tongueHeight,
      pitchNorm,
      vibrato: 0,
      tension: aperture * 0.7,
      breath,
      vowelProbs: [0.2, 0.2, 0.2, 0.2, 0.2],
      topVowel,
      tract,
      rms,
      f0Hz,
      voicing,
      centroidHz,
      trained: false,
    };
  }

  private fftInPlace(re: Float64Array, im: Float64Array, n: number): void {
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
        tmp = im[i]; im[i] = im[j]; im[j] = tmp;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wLenRe = Math.cos(ang);
      const wLenIm = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let wRe = 1;
        let wIm = 0;
        for (let k = 0; k < half; k++) {
          const idx1 = i + k;
          const idx2 = idx1 + half;
          const uRe = re[idx1];
          const uIm = im[idx1];
          const vRe = re[idx2] * wRe - im[idx2] * wIm;
          const vIm = re[idx2] * wIm + im[idx2] * wRe;
          re[idx1] = uRe + vRe;
          im[idx1] = uIm + vIm;
          re[idx2] = uRe - vRe;
          im[idx2] = uIm - vIm;
          const nextWRe = wRe * wLenRe - wIm * wLenIm;
          wIm = wRe * wLenIm + wIm * wLenRe;
          wRe = nextWRe;
        }
      }
    }
  }
}
