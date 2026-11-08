/**
 * Arc model reference inference (Phase 3, wub-arc-model-v1).
 *
 * Pure TypeScript math, no torch. Loads the exported JSON artifact
 * (`frontend/public/data/kits/arc_model.json`, same pattern as
 * taste_scores.json) and predicts the NEXT 4-bar window's section + energy
 * from the last <=6 played window feature vectors.
 *
 * Drop-in integration is specified in arc_integration_spec.md.  The engine
 * loads `arc_model.json` + the glance context sidecar off `base`; a missing
 * or invalid artifact keeps the planned arc wave (this model's blend is
 * gated by confidence and only nudges the plan).
 *
 * Forward pass (mirrors train_arc_model.py ArcModel exactly):
 *   1. standardise raw 48-d features with feat_mean / feat_std
 *   2. Linear proj 48->D + positional embedding (slot index 0..L-1)
 *   3. per TransformerEncoderLayer (norm_first, GELU exact, dropout OFF):
 *        h = h + SelfAttn(LayerNorm1(h), key padding mask)
 *        h = h + FFN(LayerNorm2(h))        (Linear D->4D, GELU, Linear 4D->D)
 *      SelfAttn: in_proj [3D,D] split q/k/v, 4 heads x (D/4), scores / sqrt(headDim),
 *      masked keys get -1e9 before softmax, concat heads, out_proj.
 *   4. masked mean-pool over real slots -> LayerNorm
 *   5. section head D->5 (softmax), energy head D->1 (sigmoid), delta head D->1 (raw)
 */

export const ARC_MODEL_FORMAT = "wub-arc-model-v1";
export const ARC_SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;

export interface ArcModelConfig {
  feature_dim: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  dim_ff: number;
  max_len: number;
  dropout: number;
  activation: string;
  norm_first: boolean;
  n_sections: number;
}

export interface ArcModelArtifact {
  format: string;
  version: number;
  config: ArcModelConfig;
  sections: string[];
  feat_mean: number[];
  feat_std: number[];
  weights: Record<string, number[][] | number[]>;
  meta?: unknown;
}

export interface ArcPrediction {
  /** Predicted next section name (one of ARC_SECTIONS). */
  section: string;
  /** Softmax probabilities aligned with ARC_SECTIONS. */
  sectionProbs: number[];
  /** Predicted next-window energy, clamped 0..1 (sigmoid head). */
  energy: number;
  /** Learned delta head output (auxiliary training head, raw value). */
  energyDeltaModel: number;
  /** energy - lastEnergy: the tension/release step the engine can use. */
  energyDelta: number;
  /** max(sectionProbs): gate section adoption on this. */
  confidence: number;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function erf(x: number): number {
  // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7 (matches PyTorch exact GELU closely).
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.sqrt(2)));
}

function matVec(w: number[][], b: number[], x: number[]): number[] {
  const out = new Array<number>(w.length);
  for (let i = 0; i < w.length; i++) {
    const row = w[i];
    let s = b[i];
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function layerNorm(x: number[], weight: number[], bias: number[]): number[] {
  const d = x.length;
  let mean = 0;
  for (const v of x) mean += v;
  mean /= d;
  let varSum = 0;
  for (const v of x) varSum += (v - mean) * (v - mean);
  const invStd = 1 / Math.sqrt(varSum / d + 1e-5);
  return x.map((v, i) => (v - mean) * invStd * weight[i] + bias[i]);
}

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function asMatrix(v: number[][] | number[] | undefined): number[][] {
  return v as number[][];
}

function asVector(v: number[][] | number[] | undefined): number[] {
  return v as number[];
}

export function isArcModelArtifact(value: unknown): value is ArcModelArtifact {
  if (!value || typeof value !== "object") return false;
  const m = value as Partial<ArcModelArtifact>;
  return (
    m.format === ARC_MODEL_FORMAT &&
    Array.isArray(m.feat_mean) &&
    Array.isArray(m.feat_std) &&
    typeof m.weights === "object" &&
    m.weights !== null
  );
}

/**
 * Predict the next window's section + energy from recent window features.
 *
 * @param model   parsed arc_model.json artifact (null -> null, use planned wave)
 * @param context 1..max_len raw 48-d window features, most-recent LAST.
 *                Pad slots are zero RAW vectors (they are masked everywhere).
 * @param lastEnergy energy of the most recent context window, 0..1.
 */
export function arcPredictNext(
  model: ArcModelArtifact | null,
  context: number[][],
  lastEnergy: number,
): ArcPrediction | null {
  if (!isArcModelArtifact(model)) return null;
  const cfg = model.config;
  const F = cfg.feature_dim;
  const D = cfg.d_model;
  const L = cfg.max_len;
  if (context.length === 0 || context.length > L) return null;
  for (const row of context) {
    if (!Array.isArray(row) || row.length !== F) return null;
    for (const v of row) if (!Number.isFinite(v)) return null;
  }
  if (!Number.isFinite(lastEnergy)) return null;
  if (model.feat_mean.length !== F || model.feat_std.length !== F) return null;

  const W = model.weights;
  const projW = asMatrix(W["proj.weight"]);
  const projB = asVector(W["proj.bias"]);
  const posW = asMatrix(W["pos.weight"]);

  // Standardise + right-align into L slots (pad raw zeros), build masks.
  const x: number[][] = [];
  const mask: boolean[] = [];
  const nPad = L - context.length;
  for (let s = 0; s < L; s++) {
    const src = s < nPad ? null : context[s - nPad];
    const row = new Array<number>(F);
    for (let f = 0; f < F; f++) {
      const raw = src ? src[f] : 0;
      row[f] = (raw - model.feat_mean[f]) / model.feat_std[f];
    }
    x.push(row);
    mask.push(src !== null);
  }

  // proj + pos
  let h: number[][] = x.map((row, s) => {
    const p = matVec(projW, projB, row);
    const pos = posW[s];
    for (let d = 0; d < D; d++) p[d] += pos[d];
    return p;
  });

  const H = cfg.n_heads;
  const headDim = D / H;
  const scale = 1 / Math.sqrt(headDim);

  for (let li = 0; li < cfg.n_layers; li++) {
    const pfx = `encoder.layers.${li}.`;
    const n1w = asVector(W[`${pfx}norm1.weight`]);
    const n1b = asVector(W[`${pfx}norm1.bias`]);
    const n2w = asVector(W[`${pfx}norm2.weight`]);
    const n2b = asVector(W[`${pfx}norm2.bias`]);
    const inW = asMatrix(W[`${pfx}self_attn.in_proj_weight`]);
    const inB = asVector(W[`${pfx}self_attn.in_proj_bias`]);
    const outW = asMatrix(W[`${pfx}self_attn.out_proj.weight`]);
    const outB = asVector(W[`${pfx}self_attn.out_proj.bias`]);
    const f1W = asMatrix(W[`${pfx}linear1.weight`]);
    const f1B = asVector(W[`${pfx}linear1.bias`]);
    const f2W = asMatrix(W[`${pfx}linear2.weight`]);
    const f2B = asVector(W[`${pfx}linear2.bias`]);

    // --- residual 1: h += attn(ln1(h)) ---
    const ln1 = h.map((row) => layerNorm(row, n1w, n1b));
    const qkv = ln1.map((row) => matVec(inW, inB, row)); // [L][3D]: q|k|v
    const attnOut = new Array<number[]>(L);
    for (let s = 0; s < L; s++) {
      const merged = new Array<number>(D);
      for (let head = 0; head < H; head++) {
        const off = head * headDim;
        const scores: number[] = [];
        for (let t = 0; t < L; t++) {
          if (!mask[t]) {
            scores.push(-1e9);
            continue;
          }
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += qkv[s][off + d] * qkv[t][D + off + d];
          }
          scores.push(dot * scale);
        }
        // Fully-masked query (pad slot): emit zeros, never NaN.
        const anyVisible = mask.some(Boolean);
        let probs: number[];
        if (!anyVisible) {
          probs = scores.map(() => 0);
        } else {
          probs = softmax(scores);
        }
        for (let d = 0; d < headDim; d++) {
          let acc = 0;
          for (let t = 0; t < L; t++) acc += probs[t] * qkv[t][2 * D + off + d];
          merged[off + d] = acc;
        }
      }
      attnOut[s] = matVec(outW, outB, merged);
    }
    for (let s = 0; s < L; s++) {
      for (let d = 0; d < D; d++) h[s][d] += attnOut[s][d];
    }

    // --- residual 2: h += ffn(ln2(h)) ---
    const ln2 = h.map((row) => layerNorm(row, n2w, n2b));
    for (let s = 0; s < L; s++) {
      const hidden = matVec(f1W, f1B, ln2[s]).map(gelu);
      const out = matVec(f2W, f2B, hidden);
      for (let d = 0; d < D; d++) h[s][d] += out[d];
    }
  }

  // Masked mean-pool -> LayerNorm -> heads.
  const count = mask.filter(Boolean).length;
  if (count === 0) return null;
  const pooled = new Array<number>(D).fill(0);
  for (let s = 0; s < L; s++) {
    if (!mask[s]) continue;
    for (let d = 0; d < D; d++) pooled[d] += h[s][d];
  }
  for (let d = 0; d < D; d++) pooled[d] /= count;
  const normed = layerNorm(pooled, asVector(W["ln.weight"]), asVector(W["ln.bias"]));

  const sectionLogits = matVec(asMatrix(W["head_section.weight"]), asVector(W["head_section.bias"]), normed);
  const energyLogit = matVec(asMatrix(W["head_energy.weight"]), asVector(W["head_energy.bias"]), normed)[0];
  const deltaModel = matVec(asMatrix(W["head_delta.weight"]), asVector(W["head_delta.bias"]), normed)[0];
  const probs = softmax(sectionLogits);
  const best = probs.indexOf(Math.max(...probs));
  const energy = clamp01(sigmoid(energyLogit));

  return {
    section: ARC_SECTIONS[best],
    sectionProbs: probs,
    energy,
    energyDeltaModel: deltaModel,
    energyDelta: energy - lastEnergy,
    confidence: probs[best],
  };
}
