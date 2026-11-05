// Joint four-stem transition ranker, distilled from the mix-level audio judge.
//
// The player assembles a 400-d state for any stem combination:
//   4 stems x (52 phrase features + 32 timbre embedding) + 64-d mix embedding
// and scores a candidate transition with the last two states and deltas.
// The weights and per-phrase features are produced by
// scripts/build_joint_ranker.py.
import type { StemName } from "./types";

const STEMS: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const STEM_DIM = 84;
const STATE_DIM = 4 * STEM_DIM + 64;
const INPUT_DIM = STATE_DIM * 5 + 5 + 6;
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
  input_dim: number;
  state_dim: number;
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
  private readonly inputScratch = new Float32Array(INPUT_DIM);
  private readonly h1Scratch = new Float32Array(256);
  private readonly h2Scratch = new Float32Array(128);

  get ready(): boolean {
    return this.manifest !== null && this.features !== null && this.w1 !== null;
  }

  get threshold(): number {
    return this.manifest?.threshold ?? 0.5;
  }

  get valAuc(): number | null {
    return this.manifest?.val_auc ?? null;
  }

  get stateDim(): number {
    return this.manifest?.state_dim ?? STATE_DIM;
  }

  async load(baseUrl: string): Promise<boolean> {
    const base = baseUrl.replace(/\/+$/, "");
    try {
      const response = await fetch(`${base}/joint_ranker.json`);
      if (!response.ok) return false;
      const manifest = (await response.json()) as JointManifest;
      if (!Array.isArray(manifest.phrase_ids) || !manifest.weights) return false;
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
      console.info(
        `[JointRanker] loaded ${manifest.model} • val_auc ${manifest.val_auc.toFixed(4)} • ` +
        `${manifest.phrase_ids.length} phrases • threshold ${manifest.threshold.toFixed(3)}`
      );
      return true;
    } catch (err) {
      console.warn("[JointRanker] load failed; falling back to the per-stem selector", err);
      return false;
    }
  }

  /** A phrase's 400-d state, with the session key/tempo overrides applied. */
  stateFor(phraseId: string, keyShiftSemitones: number, rate: number): Float32Array | null {
    if (this.features === null || this.manifest === null) return null;
    const row = this.rowOf.get(phraseId);
    if (row === undefined) return null;
    const out = this.features.slice(
      row * this.manifest.feature_dim,
      (row + 1) * this.manifest.feature_dim,
    );
    for (let stem = 0; stem < 4; stem++) {
      out[stem * STEM_DIM + 50] = keyShiftSemitones / 5;
      out[stem * STEM_DIM + 51] = (rate - 1) * 10;
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
  ): number {
    if (
      this.w1 === null || this.b1 === null || this.w2 === null ||
      this.b2 === null || this.w3 === null || this.b3 === null ||
      this.mean === null || this.std === null
    ) {
      return 0;
    }
    const input = this.inputScratch;
    input.set(prev.subarray(0, STATE_DIM), 0);
    input.set(current.subarray(0, STATE_DIM), STATE_DIM);
    input.set(candidate.subarray(0, STATE_DIM), STATE_DIM * 2);
    for (let i = 0; i < STATE_DIM; i++) {
      input[STATE_DIM * 3 + i] = candidate[i] - current[i];
      input[STATE_DIM * 4 + i] = current[i] - prev[i];
    }
    const maskOffset = STATE_DIM * 5;
    input.fill(0, maskOffset, maskOffset + 5);
    for (const stem of changed) {
      input[maskOffset + STEMS.indexOf(stem)] = 1;
    }
    input[maskOffset + 4] = changed.size / 2;
    const targetOffset = maskOffset + 5;
    input.fill(0, targetOffset, targetOffset + 6);
    input[targetOffset + (SEC_IDX[targetSection.toUpperCase()] ?? 1)] = 1;
    input[targetOffset + 5] = targetEnergy;

    // Standardize.
    for (let i = 0; i < input.length; i++) {
      input[i] = (input[i] - this.mean[i]) / this.std[i];
    }

    const w1 = this.w1;
    const b1 = this.b1;
    const h1 = this.h1Scratch;
    for (let j = 0; j < 256; j++) {
      let sum = b1[j];
      const offset = j * INPUT_DIM;
      for (let i = 0; i < INPUT_DIM; i++) sum += w1[offset + i] * input[i];
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
