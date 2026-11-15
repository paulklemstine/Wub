/**
 * Runtime neural cell-compatibility scorer (`cell_compat.json`).
 *
 * Evaluates both pairwise stem cell affinity (via learned 32-d embeddings) and
 * full 4-stem bar mix quality using weights distilled & trained on GPU from the
 * Wub training corpus (`scripts/train_cell_compat.py`).
 * Falls back cleanly to a deterministic neutral 0.5 when not loaded.
 */

import {
  CellStemMeta,
  StemName,
  TriadQuality,
  popcount16,
  thirdClashPenalty,
} from "./theory_engine";

export interface CellCompatModelWeights {
  version: number;
  format: string;
  input_dim: number;
  embed_dim: number;
  enc_w1: number[][]; // [48][24]
  enc_b1: number[];   // [48]
  enc_w2: number[][]; // [32][48]
  enc_b2: number[];   // [32]
  mix_w1: number[][]; // [64][128]
  mix_b1: number[];   // [64]
  mix_w2: number[][]; // [1][64]
  mix_b2: number[];   // [1]
  arr_ctx_dim?: number;
  arr_samples?: number;
  arr_val_auc?: number;
  arr_val_acc?: number;
  essential_slices_mined?: number;
  tracks_mined?: number;
  mean_reconstitution_fidelity?: number;
  arr_w1?: number[][]; // [48][72]
  arr_b1?: number[];   // [48]
  arr_w2?: number[][]; // [1][48]
  arr_b2?: number[];   // [1]
}

export interface CellFeatureInput {
  stem: StemName;
  meta: CellStemMeta;
  barIdx: number;
  semitonesShift: number;
  targetQuality: "minor" | "major" | "sus";
}

const STEM_ORDER: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const DOWNBEAT_MASK = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12);
const OFFBEAT_MASK = (1 << 2) | (1 << 6) | (1 << 10) | (1 << 14);
const EVEN_MASK = 0b0101010101010101;

function gelu(x: number): number {
  // Fast tanh approximation matching PyTorch GELU
  const c = 0.7978845608028654; // sqrt(2 / pi)
  const inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}

function sigmoid(x: number): number {
  if (x >= 20) return 1;
  if (x <= -20) return 0;
  return 1 / (1 + Math.exp(-x));
}

export function buildCellFeatureVector(input: CellFeatureInput): Float32Array {
  const b = Math.max(0, Math.min(3, input.barIdx));
  const m = input.meta;
  const mask = (m.o[b] ?? 0) & 0xffff;
  const purity = m.p[b] ?? 0;
  const root = m.r[b] ?? -1;
  const qual: TriadQuality = m.q[b] ?? (input.stem === "drums" ? "perc" : "mono");

  const cnt = popcount16(mask);
  const v = new Float32Array(24);
  const stemIdx = STEM_ORDER.indexOf(input.stem);
  if (stemIdx >= 0) v[stemIdx] = 1;

  v[4] = purity;
  v[5] = cnt / 16;
  v[6] = popcount16(mask & DOWNBEAT_MASK) / 4;
  v[7] = popcount16(mask & OFFBEAT_MASK) / 4;
  v[8] = cnt > 0 ? popcount16(mask & (~EVEN_MASK & 0xffff)) / cnt : 0;

  if (qual === "minor") v[9] = 1;
  else if (qual === "major") v[10] = 1;
  else if (qual === "sus") v[11] = 1;
  else v[12] = 1;

  v[13] = popcount16(mask & 0x000f) / 4;
  v[14] = popcount16((mask >>> 4) & 0x000f) / 4;
  v[15] = popcount16((mask >>> 8) & 0x000f) / 4;
  v[16] = popcount16((mask >>> 12) & 0x000f) / 4;

  v[17] = Math.min(1, Math.abs(input.semitonesShift) / 6);
  v[18] = thirdClashPenalty(qual, input.targetQuality);
  v[19] = root >= 0 ? 1 : 0;
  v[20] = m.hook ? 1 : 0;
  v[21] = m.fill && b === 3 ? 1 : 0;
  v[22] = m.s === "AAAB" || m.s === "ABAB" ? 1 : 0;
  v[23] = m.s === "AABB" || m.s === "ABAC" ? 1 : 0;
  return v;
}

export class CellCompatScorer {
  private weights: CellCompatModelWeights | null = null;
  private loaded = false;

  isReady(): boolean {
    return this.loaded && this.weights !== null;
  }

  hasArrangementHead(): boolean {
    const w = this.weights;
    return Boolean(
      this.loaded &&
      w &&
      Array.isArray(w.arr_w1) &&
      Array.isArray(w.arr_b1) &&
      Array.isArray(w.arr_w2) &&
      Array.isArray(w.arr_b2),
    );
  }

  getTrainingStats(): {
    essentialSlicesMined: number;
    tracksMined: number;
    meanReconstitutionFidelity: number;
    arrValAuc: number;
  } | null {
    const w = this.weights;
    if (!w) return null;
    return {
      essentialSlicesMined: Number(w.essential_slices_mined ?? 0),
      tracksMined: Number(w.tracks_mined ?? 0),
      meanReconstitutionFidelity: Number(w.mean_reconstitution_fidelity ?? 0),
      arrValAuc: Number(w.arr_val_auc ?? 0),
    };
  }

  async load(url = "/data/kits/cell_compat.json"): Promise<boolean> {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return false;
      const raw = (await resp.json()) as Partial<CellCompatModelWeights>;
      if (
        !raw ||
        (raw.format !== "cell_compat_v1" && raw.format !== "cell_compat_v2") ||
        !Array.isArray(raw.enc_w1) ||
        !Array.isArray(raw.mix_w1)
      ) {
        return false;
      }
      this.weights = raw as CellCompatModelWeights;
      this.loaded = true;
      return true;
    } catch {
      return false;
    }
  }

  encode(input: CellFeatureInput): Float32Array | null {
    const w = this.weights;
    if (!w) return null;
    const x = buildCellFeatureVector(input);
    const h = new Float32Array(w.enc_w1.length);
    for (let i = 0; i < w.enc_w1.length; i++) {
      const row = w.enc_w1[i];
      let s = w.enc_b1[i] ?? 0;
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      h[i] = gelu(s);
    }
    const emb = new Float32Array(w.enc_w2.length);
    let normSq = 0;
    for (let i = 0; i < w.enc_w2.length; i++) {
      const row = w.enc_w2[i];
      let s = w.enc_b2[i] ?? 0;
      for (let j = 0; j < h.length; j++) s += row[j] * h[j];
      emb[i] = s;
      normSq += s * s;
    }
    const invNorm = normSq > 1e-8 ? 1 / Math.sqrt(normSq) : 1;
    for (let i = 0; i < emb.length; i++) emb[i] *= invNorm;
    return emb;
  }

  scorePair(a: CellFeatureInput, b: CellFeatureInput): number {
    const ea = this.encode(a);
    const eb = this.encode(b);
    if (!ea || !eb) return 0.5;
    let dot = 0;
    for (let i = 0; i < ea.length; i++) dot += ea[i] * eb[i];
    return Math.max(0, Math.min(1, 0.5 + 0.5 * dot));
  }

  scoreQuartet(inputs: Record<StemName, CellFeatureInput>): number {
    const w = this.weights;
    if (!w) return 0.5;
    const embs: Float32Array[] = [];
    for (const st of STEM_ORDER) {
      const e = this.encode(inputs[st]);
      if (!e) return 0.5;
      embs.push(e);
    }

    const embedDim = embs[0].length;
    const concat = new Float32Array(embedDim * 4);
    for (let s = 0; s < 4; s++) concat.set(embs[s], s * embedDim);

    const h = new Float32Array(w.mix_w1.length);
    for (let i = 0; i < w.mix_w1.length; i++) {
      const row = w.mix_w1[i];
      let sum = w.mix_b1[i] ?? 0;
      for (let j = 0; j < concat.length; j++) sum += row[j] * concat[j];
      h[i] = gelu(sum);
    }

    let logit = w.mix_b2[0] ?? 0;
    const row2 = w.mix_w2[0];
    for (let j = 0; j < h.length; j++) logit += row2[j] * h[j];
    return sigmoid(logit);
  }

  /** Score horizontal slice-to-slice continuation at `barPos` (0..3) and section boundary. */
  scoreTransition(args: {
    prev: CellFeatureInput;
    next: CellFeatureInput;
    barPos: number;
    isSectionBoundary: boolean;
  }): number {
    const w = this.weights;
    if (!w) return 0.5;
    const ePrev = this.encode(args.prev);
    const eNext = this.encode(args.next);
    if (!ePrev || !eNext) return 0.5;

    if (!w.arr_w1 || !w.arr_b1 || !w.arr_w2 || !w.arr_b2) {
      let dot = 0;
      for (let i = 0; i < ePrev.length; i++) dot += ePrev[i] * eNext[i];
      return Math.max(0, Math.min(1, 0.5 + 0.35 * dot));
    }

    const embedDim = ePrev.length;
    const pBar = Math.max(0, Math.min(3, args.prev.barIdx));
    const nBar = Math.max(0, Math.min(3, args.next.barIdx));
    const prevEs = Math.max(0, Math.min(1, args.prev.meta.es?.[pBar] ?? args.prev.meta.p[pBar] ?? 0.5));
    const nextEs = Math.max(0, Math.min(1, args.next.meta.es?.[nBar] ?? args.next.meta.p[nBar] ?? 0.5));
    const prevPur = args.prev.meta.p[pBar] ?? 0.5;
    const nextPur = args.next.meta.p[nBar] ?? 0.5;
    const delta = Math.max(-1, Math.min(1, nextPur - prevPur));

    const concat = new Float32Array(embedDim * 2 + 8);
    concat.set(ePrev, 0);
    concat.set(eNext, embedDim);
    const bPos = ((args.barPos % 4) + 4) % 4;
    concat[embedDim * 2 + bPos] = 1;
    concat[embedDim * 2 + 4] = args.isSectionBoundary ? 1 : 0;
    concat[embedDim * 2 + 5] = prevEs;
    concat[embedDim * 2 + 6] = nextEs;
    concat[embedDim * 2 + 7] = 0.5 + 0.5 * delta;

    const h = new Float32Array(w.arr_w1.length);
    for (let i = 0; i < w.arr_w1.length; i++) {
      const row = w.arr_w1[i];
      let sum = w.arr_b1[i] ?? 0;
      for (let j = 0; j < concat.length; j++) sum += row[j] * concat[j];
      h[i] = gelu(sum);
    }

    let logit = w.arr_b2[0] ?? 0;
    const row2 = w.arr_w2[0];
    for (let j = 0; j < h.length; j++) logit += row2[j] * h[j];
    return sigmoid(logit);
  }
}
