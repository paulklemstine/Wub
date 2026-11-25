/** Learned real-chronology phrase-transition prior in the shipped bank's space.
 *
 * Trained from 2,199 consecutive shipped-phrase pairs mined out of the real
 * songs (`scripts/train_bank_transition.py`). The browser scores a candidate
 * next phrase with the compact MLP and, when available, the mined top-K
 * successor probability. It is a bounded ranking bias only.
 */
const clamp01 = (v: number): number => v < 0 ? 0 : v > 1 ? 1 : v;
const sigmoid = (v: number): number => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v))));

export const BANK_INPUT_DIM = 136;
export const BANK_EMBED_DIM = 64;
export const BANK_SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;

interface Layer { w: number[][]; b: number[] }

export interface BankTransitionArtifact {
  format: string;
  version: number;
  input_dim: number;
  hidden: number;
  mean: number[];
  std: number[];
  layers: Layer[];
  embedding_dim: number;
  sections: string[];
  topk: Record<string, Array<{ id: string; p: number }>>;
  report?: { val_auc?: number; real_transitions?: number };
}

export interface BankPhrase {
  id: string;
  section?: string;
  embedding?: number[];
  energy: number;
  bpm: number;
  keyRoot: number;
}

export function isBankTransitionArtifact(v: unknown): v is BankTransitionArtifact {
  const d = v as BankTransitionArtifact;
  return Boolean(
    d && d.format === "wub-bank-transition-v1" && d.input_dim === BANK_INPUT_DIM
    && d.embedding_dim === BANK_EMBED_DIM && d.layers?.length === 2
    && d.mean?.length === BANK_INPUT_DIM && d.std?.length === BANK_INPUT_DIM
    && d.layers[0].w.length === d.hidden && d.layers[1].w.length === 1
    && d.topk && typeof d.topk === "object",
  );
}

function keyDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % 12;
  return Math.min(d, 12 - d) / 6;
}

/** 136-d feature vector: current+next embedding, section, energy delta, bpm/key distance. */
export function bankTransitionFeatures(
  current: BankPhrase,
  candidate: BankPhrase,
  section: string,
): number[] {
  const v = new Array<number>(BANK_INPUT_DIM).fill(0);
  const a = current.embedding ?? [];
  const b = candidate.embedding ?? [];
  for (let i = 0; i < BANK_EMBED_DIM; i++) {
    v[i] = a[i] ?? 0;
    v[BANK_EMBED_DIM + i] = b[i] ?? 0;
  }
  const idx = BANK_SECTIONS.indexOf(section as (typeof BANK_SECTIONS)[number]);
  v[128 + Math.max(0, idx)] = 1;
  v[133] = Math.max(-1, Math.min(1, (candidate.energy - current.energy) / 20));
  v[134] = Math.min(1, Math.abs(current.bpm - candidate.bpm) / 12);
  v[135] = keyDistance(current.keyRoot, candidate.keyRoot);
  return v;
}

export function inferBank(model: BankTransitionArtifact, features: readonly number[]): number {
  const x = new Float64Array(BANK_INPUT_DIM);
  for (let i = 0; i < BANK_INPUT_DIM; i++) {
    const s = model.std[i];
    x[i] = (features[i] - model.mean[i]) / (Math.abs(s) > 1e-6 ? s : 1);
  }
  const hidden = new Float64Array(model.hidden);
  for (let j = 0; j < model.hidden; j++) {
    let sum = model.layers[0].b[j];
    const row = model.layers[0].w[j];
    for (let i = 0; i < BANK_INPUT_DIM; i++) sum += row[i] * x[i];
    hidden[j] = Math.tanh(sum);
  }
  const out = model.layers[1];
  let logit = out.b[0];
  for (let j = 0; j < model.hidden; j++) logit += out.w[0][j] * hidden[j];
  return sigmoid(logit);
}

/** Combined learned score 0..1 (MLP plus mined real-successor probability). */
export function bankTransitionScore(
  model: BankTransitionArtifact,
  current: BankPhrase,
  candidate: BankPhrase,
  section: string,
): number {
  const mlp = inferBank(model, bankTransitionFeatures(current, candidate, section));
  const row = model.topk[current.id]?.find((item) => item.id === candidate.id);
  return row ? clamp01(0.6 * mlp + 0.4 * row.p) : mlp;
}
