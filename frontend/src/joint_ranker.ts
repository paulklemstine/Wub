// Joint four-stem transition ranker, distilled from the mix-level audio judge.
//
// The player assembles a 424-d state for any stem combination:
//   4 interleaved stems x 90-d phrase features + 64-d mix embedding
// and scores a candidate transition with the last two states, deltas, and the
// current/next block length (4/8/16 bars).  The weights and per-phrase
// features are produced by scripts/build_joint_ranker.py.  Older 400-d models
// (stemDim 84) still load: the per-stem width is derived from feature_dim.
import type { StemName } from "./types";

const STEMS: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const STEM_DIM = 90;
const STATE_DIM = 4 * STEM_DIM + 64;
// prev/current/candidate + two deltas, change mask, target, current/next bars.
const INPUT_DIM = STATE_DIM * 5 + 5 + 6 + 6;
const MIX_DIM = 64;
const BLOCK_INDEX: Record<number, number> = { 4: 0, 8: 1, 16: 2 };
const SEC_IDX: Record<string, number> = {
  INTRO: 0, GROOVE_A: 1, BUILDUP: 2, DROP: 3, BREAKDOWN: 4,
};

interface JointWeights {
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
  w3: number[];
  b3: number[];
}

interface JointManifest {
  version: number;
  model: string;
  val_auc: number;
  threshold: number;
  thresholds?: Record<string, number>;
  block_auc?: Record<string, { auc: number; teacher_auc: number; threshold: number; n: number }>;
  input_dim: number;
  state_dim: number;
  stem_dim?: number;
  feature_dim: number;
  phrase_ids: string[];
  feature_file: string;
  scaler: { mean: number[]; std: number[] };
  weights: JointWeights;
}

export class JointRanker {
  private manifest: JointManifest | null = null;
  private features: Float32Array | null = null;
  private rowOf = new Map<string, number>();
  private w1: Float32Array | null = null;
  private b1: Float32Array | null = null;
  private w2: Float32Array | null = null;
  private b2: Float32Array | null = null;
  private w3: Float32Array | null = null;
  private b3: Float32Array | null = null;
  private mean: Float32Array | null = null;
  private std: Float32Array | null = null;
  private inputDimValue = INPUT_DIM;
  private stateDimValue = STATE_DIM;
  private stemDimValue = STEM_DIM;
  private inputScratch = new Float32Array(INPUT_DIM);
  private readonly h1Scratch = new Float32Array(256);
  private readonly h2Scratch = new Float32Array(128);

  get ready(): boolean {
    return this.manifest !== null && this.features !== null && this.w1 !== null;
  }

  get threshold(): number {
    return this.manifest?.threshold ?? 0.5;
  }

  /** Hold threshold for the upcoming block length (4/8/16 bars). */
  thresholdFor(blockBars: number): number {
    const value = this.manifest?.thresholds?.[String(blockBars)];
    return typeof value === "number" ? value : this.threshold;
  }

  get valAuc(): number | null {
    return this.manifest?.val_auc ?? null;
  }

  get stateDim(): number {
    return this.stateDimValue;
  }

  /** Per-stem chunk width: 90 for the 424-d model, 84 for the old 400-d one. */
  get stemDim(): number {
    return this.stemDimValue;
  }

  async load(baseUrl: string): Promise<boolean> {
    const base = baseUrl.replace(/\/+$/, "");
    try {
      const response = await fetch(`${base}/joint_ranker.json`);
      if (!response.ok) return false;
      const manifest = (await response.json()) as JointManifest;
      if (!Array.isArray(manifest.phrase_ids) || !manifest.weights) return false;
      const stateDim = manifest.state_dim ?? STATE_DIM;
      const featureDim = manifest.feature_dim ?? stateDim;
      const stemDim = (featureDim - MIX_DIM) / 4;
      if (!Number.isInteger(stemDim) || stemDim < 58 || stemDim * 4 + MIX_DIM !== stateDim) {
        console.warn(
          `[JointRanker] feature_dim ${featureDim} / state_dim ${stateDim} do not ` +
          "describe four equal stem chunks; retrain the arc model"
        );
        return false;
      }
      if (typeof manifest.stem_dim === "number" && manifest.stem_dim !== stemDim) {
        console.warn(
          `[JointRanker] stem_dim ${manifest.stem_dim} disagrees with feature_dim; ` +
          `using ${stemDim}`
        );
      }
      const expectedInput = stateDim * 5 + 5 + 6 + 6;
      if (manifest.input_dim !== expectedInput) {
        console.warn(
          `[JointRanker] input_dim ${manifest.input_dim}, expected ${expectedInput}; ` +
          "retrain the arc model"
        );
        return false;
      }
      const featureResponse = await fetch(`${base}/${manifest.feature_file}`);
      if (!featureResponse.ok) return false;
      const features = new Float32Array(await featureResponse.arrayBuffer());
      const expected = manifest.phrase_ids.length * manifest.feature_dim;
      if (features.length < expected) {
        console.warn(`[JointRanker] feature file has ${features.length} floats, expected ${expected}`);
        return false;
      }
      this.manifest = manifest;
      this.features = features;
      this.stateDimValue = stateDim;
      this.stemDimValue = stemDim;
      this.inputDimValue = manifest.input_dim;
      this.inputScratch = new Float32Array(manifest.input_dim);
      this.rowOf.clear();
      manifest.phrase_ids.forEach((id, index) => this.rowOf.set(id, index));
      const w = manifest.weights;
      this.w1 = new Float32Array(w.w1);
      this.b1 = new Float32Array(w.b1);
      this.w2 = new Float32Array(w.w2);
      this.b2 = new Float32Array(w.b2);
      this.w3 = new Float32Array(w.w3);
      this.b3 = new Float32Array(w.b3);
      this.mean = new Float32Array(manifest.scaler.mean);
      this.std = new Float32Array(manifest.scaler.std);
      const perBlock = manifest.thresholds
        ? ` • thresholds 4/8/16 = ${[4, 8, 16]
          .map((bars) => (manifest.thresholds?.[String(bars)] ?? manifest.threshold).toFixed(3))
          .join(" / ")}`
        : "";
      console.info(
        `[JointRanker] loaded ${manifest.model} • val_auc ${manifest.val_auc.toFixed(4)} • ` +
        `${manifest.phrase_ids.length} phrases • threshold ${manifest.threshold.toFixed(3)}${perBlock}`
      );
      return true;
    } catch (err) {
      console.warn("[JointRanker] load failed; falling back to the per-stem selector", err);
      return false;
    }
  }

  /** A phrase's state, with the session key/tempo overrides applied. */
  stateFor(phraseId: string, keyShiftSemitones: number, rate: number): Float32Array | null {
    if (this.features === null || this.manifest === null) return null;
    const row = this.rowOf.get(phraseId);
    if (row === undefined) return null;
    const out = this.features.slice(
      row * this.manifest.feature_dim,
      (row + 1) * this.manifest.feature_dim,
    );
    const stemDim = this.stemDimValue;
    const rot = ((Math.round(keyShiftSemitones) % 12) + 12) % 12;
    const chromaTmp = new Float32Array(12);
    const rootTmp = new Float32Array(12);
    // The ranker was trained on near matches (|rate-1|<=2%, small shifts).
    // Clamp the overrides to the trained regime so far stretches/transposes
    // read as "very far" instead of extrapolating the MLP to absurd scores;
    // the caller's tiered costs differentiate far vs very far.
    for (let stem = 0; stem < 4; stem++) {
      const base = stem * stemDim;
      out[base + 50] = Math.max(-1, Math.min(1, keyShiftSemitones / 5));
      out[base + 51] = Math.max(-1.5, Math.min(1.5, (rate - 1) * 10));
      // Rotate pitched stems' chroma (16..27) and root (28..39) into session-key
      // space so the ranker & stemCosine see post-transposition harmonic alignment.
      if (stem > 0 && rot !== 0) {
        for (let i = 0; i < 12; i++) {
          chromaTmp[(i + rot) % 12] = out[base + 16 + i];
          rootTmp[(i + rot) % 12] = out[base + 28 + i];
        }
        out.set(chromaTmp, base + 16);
        out.set(rootTmp, base + 28);
      }
    }
    return out;
  }

  /**
   * Score a candidate transition. `changed` is the set of stems that move to
   * the candidate entries; the other stems in `candidate` are held current.
   */
  score(
    prev: Float32Array,
    current: Float32Array,
    candidate: Float32Array,
    changed: ReadonlySet<StemName>,
    targetSection: string,
    targetEnergy: number,
    currentBlockBars: number,
    nextBlockBars: number,
  ): number {
    if (
      this.w1 === null || this.b1 === null || this.w2 === null ||
      this.b2 === null || this.w3 === null || this.b3 === null ||
      this.mean === null || this.std === null
    ) {
      return 0;
    }
    const stateDim = this.stateDimValue;
    const input = this.inputScratch;
    input.set(prev.subarray(0, stateDim), 0);
    input.set(current.subarray(0, stateDim), stateDim);
    input.set(candidate.subarray(0, stateDim), stateDim * 2);
    for (let i = 0; i < stateDim; i++) {
      input[stateDim * 3 + i] = candidate[i] - current[i];
      input[stateDim * 4 + i] = current[i] - prev[i];
    }
    const maskOffset = stateDim * 5;
    input.fill(0, maskOffset, maskOffset + 5);
    for (const stem of changed) {
      input[maskOffset + STEMS.indexOf(stem)] = 1;
    }
    input[maskOffset + 4] = changed.size / 2;
    const targetOffset = maskOffset + 5;
    input.fill(0, targetOffset, targetOffset + 6);
    input[targetOffset + (SEC_IDX[targetSection.toUpperCase()] ?? 1)] = 1;
    input[targetOffset + 5] = targetEnergy;
    const blockOffset = targetOffset + 6;
    input.fill(0, blockOffset, blockOffset + 6);
    input[blockOffset + (BLOCK_INDEX[currentBlockBars] ?? 0)] = 1;
    input[blockOffset + 3 + (BLOCK_INDEX[nextBlockBars] ?? 0)] = 1;

    // Standardize.
    for (let i = 0; i < this.inputDimValue; i++) {
      input[i] = (input[i] - this.mean[i]) / this.std[i];
    }

    const w1 = this.w1;
    const b1 = this.b1;
    const h1 = this.h1Scratch;
    for (let j = 0; j < 256; j++) {
      let sum = b1[j];
      const offset = j * this.inputDimValue;
      for (let i = 0; i < this.inputDimValue; i++) sum += w1[offset + i] * input[i];
      h1[j] = sum > 0 ? sum : 0;
    }
    const w2 = this.w2;
    const b2 = this.b2;
    const h2 = this.h2Scratch;
    for (let j = 0; j < 128; j++) {
      let sum = b2[j];
      const offset = j * 256;
      for (let i = 0; i < 256; i++) sum += w2[offset + i] * h1[i];
      h2[j] = sum > 0 ? sum : 0;
    }
    const w3 = this.w3;
    const b3 = this.b3;
    let logit = b3[0];
    for (let i = 0; i < 128; i++) logit += w3[i] * h2[i];
    return 1 / (1 + Math.exp(-logit));
  }
}
