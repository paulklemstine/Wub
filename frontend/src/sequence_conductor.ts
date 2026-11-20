/** History-aware phrase sequence conductor (browser inference).
 * Predicts the next phrase embedding from the last six phrases of the set.
 */
export const SEQ_HISTORY = 6;
export const SEQ_STEP_DIM = 214;
export const SEQ_STATE_DIM = SEQ_HISTORY * SEQ_STEP_DIM; // 6 * 214 = 1284
export const SEQ_TARGET_DIM = 74;
export const SEQ_EMBED_DIM = 64;
export const SEQ_SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;

interface Layer { w: number[][]; b: number[] }

export interface SequenceConductorArtifact {
  format: string;
  version: number;
  state_dim: number;
  target_dim: number;
  history: number;
  hidden: number;
  embed_dim: number;
  sections: string[];
  mean: number[];
  std: number[];
  layers: Layer[];
  report?: { val_embedding_cosine?: number; sequences?: number };
}

export interface SequenceConductorOutput {
  embedding: number[];
  change: number[];
  section: string;
  energy: number;
}

export function isSequenceConductorArtifact(v: unknown): v is SequenceConductorArtifact {
  const d = v as SequenceConductorArtifact;
  return Boolean(
    d && d.format === "wub-sequence-conductor-v1" && d.state_dim === SEQ_STATE_DIM
    && d.target_dim === SEQ_TARGET_DIM && d.history === SEQ_HISTORY
    && d.embed_dim === SEQ_EMBED_DIM && d.sections?.length === 5
    && d.mean?.length === SEQ_STATE_DIM && d.std?.length === SEQ_STATE_DIM
    && d.layers?.length === 3 && d.layers[0].w.length === d.hidden
    && d.layers[1].w.length === 128 && d.layers[2].w.length === SEQ_TARGET_DIM,
  );
}

export function sequenceState(steps: readonly number[][]): number[] {
  const out = new Array<number>(SEQ_STATE_DIM).fill(0);
  const recent = steps.slice(-SEQ_HISTORY);
  for (let s = 0; s < SEQ_HISTORY; s++) {
    const step = recent[s] ?? [];
    for (let i = 0; i < SEQ_STEP_DIM; i++) out[s * SEQ_STEP_DIM + i] = step[i] ?? 0;
  }
  return out;
}

export function inferSequence(model: SequenceConductorArtifact, raw: readonly number[]): SequenceConductorOutput {
  const x = new Float64Array(SEQ_STATE_DIM);
  for (let i = 0; i < SEQ_STATE_DIM; i++) {
    const s = model.std[i];
    x[i] = (raw[i] - model.mean[i]) / (Math.abs(s) > 1e-6 ? s : 1);
  }
  let a: Float64Array = x;
  for (let i = 0; i < 3; i++) {
    const layer = model.layers[i];
    const out = new Float64Array(layer.b.length);
    for (let j = 0; j < layer.b.length; j++) {
      let sum = layer.b[j];
      for (let k = 0; k < a.length; k++) sum += layer.w[j][k] * a[k];
      out[j] = i === 2 ? sum : Math.tanh(sum);
    }
    a = out;
  }
  const embedding = Array.from(a.slice(0, SEQ_EMBED_DIM));
  const change = Array.from(a.slice(64, 68), (v) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v)))));
  let best = 0;
  for (let i = 1; i < 5; i++) if (a[68 + i] > a[68 + best]) best = i;
  return { embedding, change, section: model.sections[best] ?? "GROOVE_A",
    energy: Math.max(0, Math.min(1, a[73])) };
}

export function sequenceScore(predicted: readonly number[], candidate: readonly number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < SEQ_EMBED_DIM; i++) {
    const p = predicted[i] ?? 0, c = candidate[i] ?? 0;
    dot += p * c; na += p * p; nb += c * c;
  }
  return na > 1e-8 && nb > 1e-8 ? Math.max(0, Math.min(1, dot / Math.sqrt(na * nb))) : 0.5;
}
