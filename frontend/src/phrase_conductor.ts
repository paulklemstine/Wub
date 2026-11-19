/** Phrase-space conductor: predicts the NEXT phrase embedding from the current
 * four-stem mix state, trained on real song chronology mapped to shipped ids.
 * Browser scoring is embedding cosine against candidate phrases, so every
 * decision maps to a phrase that actually exists in the bank.
 */
const clamp01 = (v: number): number => v < 0 ? 0 : v > 1 ? 1 : v;

export const PHRASE_STATE_DIM = 264;
export const PHRASE_TARGET_DIM = 70;
export const PHRASE_EMBED_DIM = 64;
export const PHRASE_PATTERN_DIM = 48;
export const PHRASE_SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;

interface Layer { w: number[][]; b: number[] }

export interface PhraseConductorArtifact {
  format: string;
  version: number;
  state_dim: number;
  target_dim: number;
  hidden: number;
  embed_dim: number;
  sections: string[];
  mean: number[];
  std: number[];
  layers: Layer[];
  report?: { val_embedding_cosine?: number; baseline_cosine?: number; pairs?: number };
}

export interface PhraseConductorInput {
  embedding: number[];
  patterns: number[][]; // four stems x 48
  section: string;
  energy: number;
  bpm: number;
  keyRoot: number;
}

export function isPhraseConductorArtifact(v: unknown): v is PhraseConductorArtifact {
  const d = v as PhraseConductorArtifact;
  return Boolean(
    d && d.format === "wub-phrase-conductor-v1" && d.state_dim === PHRASE_STATE_DIM
    && d.target_dim === PHRASE_TARGET_DIM && d.embed_dim === PHRASE_EMBED_DIM
    && d.sections?.length === 5 && d.hidden > 0
    && d.mean?.length === PHRASE_STATE_DIM && d.std?.length === PHRASE_STATE_DIM
    && d.layers?.length === 3
    && d.layers[0].w.length === d.hidden && d.layers[1].w.length === d.hidden
    && d.layers[2].w.length === PHRASE_TARGET_DIM,
  );
}

export function phraseConductorState(input: PhraseConductorInput): number[] {
  const v: number[] = [];
  for (let i = 0; i < PHRASE_EMBED_DIM; i++) v.push(input.embedding[i] ?? 0);
  for (let s = 0; s < 4; s++) {
    const vec = input.patterns[s] ?? [];
    for (let i = 0; i < PHRASE_PATTERN_DIM; i++) v.push(vec[i] ?? 0);
  }
  const idx = PHRASE_SECTIONS.indexOf(input.section as (typeof PHRASE_SECTIONS)[number]);
  for (let k = 0; k < 5; k++) v.push(k === Math.max(0, idx) ? 1 : 0);
  v.push(clamp01(input.energy));
  v.push(Math.min(1, input.bpm / 200));
  const d = Math.abs(input.keyRoot % 12) % 12;
  v.push(Math.min(d, 12 - d) / 6);
  return v.slice(0, PHRASE_STATE_DIM);
}

export function inferPhraseConductor(model: PhraseConductorArtifact, rawState: number[]): number[] {
  const x = new Float64Array(PHRASE_STATE_DIM);
  for (let i = 0; i < PHRASE_STATE_DIM; i++) {
    const s = model.std[i];
    x[i] = (rawState[i] - model.mean[i]) / (Math.abs(s) > 1e-6 ? s : 1);
  }
  let a: Float64Array = x;
  for (let l = 0; l < 3; l++) {
    const layer = model.layers[l];
    const out = new Float64Array(layer.b.length);
    for (let j = 0; j < layer.b.length; j++) {
      let sum = layer.b[j];
      for (let i = 0; i < a.length; i++) sum += layer.w[j][i] * a[i];
      out[j] = l === 2 ? sum : Math.tanh(sum);
    }
    a = out;
  }
  return Array.from(a.slice(0, PHRASE_EMBED_DIM));
}

/** Cosine similarity between the predicted and a candidate phrase embedding. */
export function phraseConductorScore(predicted: number[], candidateEmbedding: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < PHRASE_EMBED_DIM; i++) {
    const a = predicted[i] ?? 0;
    const b = candidateEmbedding[i] ?? 0;
    dot += a * b; na += a * a; nb += b * b;
  }
  return na > 1e-8 && nb > 1e-8 ? clamp01(dot / Math.sqrt(na * nb)) : 0.5;
}
