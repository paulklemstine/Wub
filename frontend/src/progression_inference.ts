/**
 * Chord-progression scorer reference inference (wub-progression-v1).
 *
 * Pure TypeScript math, no torch. Loads `progression_model.json` and
 * `progression_context.json` (same static-data pattern as taste_scores.json)
 * and scores candidate 4-bar phrases by P(next chord | recent chord context)
 * in tonic-relative (session-key-relative) space.
 *
 * Drop-in integration is specified in progression_integration_spec.md.
 * This file is reference-only: it is NOT imported by the engine yet.
 *
 * Representation (see spec for justification):
 *   rel = (abs_pitch_class - keyRoot) % 12, tonic=0, C=0.
 *   rel_chord = chord_root_abs (argmax bass+other) relative to track key.
 * Canonical chroma = rotate(abs_chroma, -keyRoot) (tonic at 0).
 *
 * Forward pass (mirrors train_progression.py MLP exactly):
 *   1. buildPrevVector(prev, prev2) -> 56-d RAW (see contract below)
 *   2. standardise with feat_mean / feat_std (std floored at 0.05 at train)
 *   3. fc1: 56 -> 64 + ReLU (dropout OFF at inference)
 *   4. fc2: 64 -> 12 logits, log_softmax -> logprobs over next rel_chord
 * Markov fallback is a plain table lookup on log probs (also a one-hot
 * dot-product, parity-checked as such).
 */

export const PROGRESSION_MODEL_FORMAT = "wub-progression-v1";
export const PROGRESSION_CONTEXT_FORMAT = "wub-progression-context-v1";

export interface ProgressionModelArtifact {
  format: string;
  version: number;
  config: { n_classes: number; mlp_in: number; mlp_hidden: number; activation: string; mode_split: boolean; alpha: number };
  markov: { unified_log: number[][]; minor_log: number[][]; major_log: number[][]; unigram_log: number[] };
  mlp: { feat_mean: number[]; feat_std: number[]; fc1_weight: number[][]; fc1_bias: number[]; fc2_weight: number[][]; fc2_bias: number[] };
  meta?: unknown;
}

export interface ProgressionContextEntry {
  rel_chord: number;          // 0..11, tonic=0
  rel_bass: number;
  rel_chroma: number[];       // 12-d canonical overall chroma
  rel_bass_chroma: number[];  // 12-d canonical bass chroma
  abs_chroma: number[];
  abs_bass_root: number | null;
  abs_chord_root: number | null;
  key: string;
  key_root: number;           // 0..11
  mode: "minor" | "major" | string;
  section: string;
  energy: number;             // 0..1
}

export interface ProgressionContextArtifact {
  format: string;
  version: number;
  n_phrases: number;
  byPhraseId: Record<string, ProgressionContextEntry>;
}

const SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;

function isProgressionModel(v: unknown): v is ProgressionModelArtifact {
  if (!v || typeof v !== "object") return false;
  const m = v as Partial<ProgressionModelArtifact>;
  return (
    m.format === PROGRESSION_MODEL_FORMAT &&
    !!m.markov && !!m.mlp &&
    Array.isArray(m.markov.unified_log) &&
    Array.isArray(m.mlp.feat_mean) &&
    Array.isArray(m.mlp.fc1_weight)
  );
}

function isProgressionContext(v: unknown): v is ProgressionContextArtifact {
  if (!v || typeof v !== "object") return false;
  const m = v as Partial<ProgressionContextArtifact>;
  return (
    m.format === PROGRESSION_CONTEXT_FORMAT &&
    typeof m.byPhraseId === "object" && m.byPhraseId !== null
  );
}

function reluVec(x: number[]): number[] {
  return x.map((v) => (v > 0 ? v : 0));
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

function logSoftmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  let sum = 0;
  const exps = logits.map((v) => {
    const e = Math.exp(v - m);
    sum += e;
    return e;
  });
  const logZ = Math.log(sum);
  return logits.map((v) => v - m - logZ);
}

/**
 * Build the 56-d RAW prev-context vector (train order, exactly):
 *   [0..11]   prev rel_chord one-hot
 *   [12..23]  prev2 rel_chord one-hot (all-zero when no prev2)
 *   [24..35]  prev rel_chroma (12-d canonical)
 *   [36..47]  prev rel_bass_chroma (12-d canonical)
 *   [48]      prev energy
 *   [49..53]  prev section one-hot (INTRO/GROOVE_A/BUILDUP/DROP/BREAKDOWN)
 *   [54]      is_minor (1/0), [55] is_major (1/0)
 * Standardisation happens inside the scorer, never here.
 */
export function buildPrevVector(
  prev: ProgressionContextEntry,
  prev2: ProgressionContextEntry | null,
): number[] {
  const v = new Array<number>(56).fill(0);
  if (prev.rel_chord >= 0 && prev.rel_chord < 12) v[prev.rel_chord] = 1;
  if (prev2 && prev2.rel_chord >= 0 && prev2.rel_chord < 12) v[12 + prev2.rel_chord] = 1;
  for (let i = 0; i < 12; i++) {
    v[24 + i] = prev.rel_chroma[i] ?? 0;
    v[36 + i] = prev.rel_bass_chroma[i] ?? 0;
  }
  v[48] = prev.energy;
  const si = SECTIONS.indexOf(prev.section as (typeof SECTIONS)[number]);
  v[49 + (si >= 0 ? si : 1)] = 1;
  v[54] = prev.mode === "minor" ? 1 : 0;
  v[55] = prev.mode === "major" ? 1 : 0;
  return v;
}

/**
 * MLP log P(next_rel_chord | prevCtx). Returns 12 logprobs (higher better)
 * or null when the artifact is missing/invalid.
 */
export function progressionMlpLogProbs(
  model: ProgressionModelArtifact | null,
  prevVec: number[],
): number[] | null {
  if (!isProgressionModel(model)) return null;
  if (!Array.isArray(prevVec) || prevVec.length !== 56) return null;
  for (const x of prevVec) if (!Number.isFinite(x)) return null;
  const { feat_mean, feat_std, fc1_weight, fc1_bias, fc2_weight, fc2_bias } = model.mlp;
  if (feat_mean.length !== 56 || feat_std.length !== 56) return null;
  const std = prevVec.map((raw, i) => {
    const s = feat_std[i];
    const denom = Number.isFinite(s) && s > 1e-9 ? s : 1;
    return (raw - feat_mean[i]) / denom;
  });
  const h = reluVec(matVec(fc1_weight, fc1_bias, std));
  const logits = matVec(fc2_weight, fc2_bias, h);
  if (logits.length !== 12) return null;
  return logSoftmax(logits);
}

/** Convenience: MLP score for one candidate class (log prob, higher better). */
export function progressionMlpScore(
  model: ProgressionModelArtifact | null,
  prev: ProgressionContextEntry,
  prev2: ProgressionContextEntry | null,
  nextRelChord: number,
): number | null {
  if (!Number.isInteger(nextRelChord) || nextRelChord < 0 || nextRelChord > 11) return null;
  const vec = buildPrevVector(prev, prev2);
  const logp = progressionMlpLogProbs(model, vec);
  if (!logp) return null;
  return logp[nextRelChord];
}

/**
 * Markov fallback: log P(next | prev) from the mode-split table (higher better),
 * or null when invalid. Unified table is used when mode is unknown.
 * This is exactly one_hot(prev) . table . one_hot(next) — see parity harness.
 */
export function progressionMarkovScore(
  model: ProgressionModelArtifact | null,
  prevRelChord: number,
  nextRelChord: number,
  mode: string | null,
): number | null {
  if (!isProgressionModel(model)) return null;
  if (!Number.isInteger(prevRelChord) || !Number.isInteger(nextRelChord)) return null;
  if (prevRelChord < 0 || prevRelChord > 11 || nextRelChord < 0 || nextRelChord > 11) return null;
  const table =
    mode === "minor" ? model.markov.minor_log :
    mode === "major" ? model.markov.major_log :
    model.markov.unified_log;
  if (!table || !table[prevRelChord] || !Number.isFinite(table[prevRelChord][nextRelChord])) return null;
  return table[prevRelChord][nextRelChord];
}

/** Convenience: resolve entries by phrase id and score (MLP primary, Markov fallback). */
export function progressionScoreByPhraseId(
  model: ProgressionModelArtifact | null,
  context: ProgressionContextArtifact | null,
  prevPhraseId: string,
  prev2PhraseId: string | null,
  nextPhraseId: string,
): { score: number; source: "mlp" | "markov" | null } {
  if (!isProgressionContext(context)) return { score: 0, source: null };
  const prev = context.byPhraseId[prevPhraseId];
  const next = context.byPhraseId[nextPhraseId];
  if (!prev || !next) return { score: 0, source: null };
  const prev2 = prev2PhraseId ? context.byPhraseId[prev2PhraseId] ?? null : null;
  const mlp = progressionMlpScore(model, prev, prev2, next.rel_chord);
  if (mlp !== null && Number.isFinite(mlp)) return { score: mlp, source: "mlp" };
  const mk = progressionMarkovScore(model, prev.rel_chord, next.rel_chord, prev.mode);
  if (mk !== null && Number.isFinite(mk)) return { score: mk, source: "markov" };
  return { score: 0, source: null };
}

/** Rotate a 12-d chroma by +shift (engine rotateChroma). shift may be negative. */
export function rotateChroma12(chroma: readonly number[], shift: number): number[] {
  const out = new Array<number>(12).fill(0);
  for (let i = 0; i < 12; i++) out[(i + shift + 12) % 12] = chroma[i] ?? 0;
  return out;
}

export { isProgressionModel, isProgressionContext };
