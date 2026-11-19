/** Browser inference for the offline-trained song conductor.
 *
 * The model predicts, from the last two 4-bar cells of all four stems:
 *   - the next section (5-class),
 *   - the next target energy,
 *   - which stems change at the boundary (4 sigmoid logits),
 *   - the next 4-bar onset/chroma pattern of each stem (4 x 48).
 *
 * It is a decision source for the existing selector, not an audio renderer:
 * every bar it picks still comes from the shipped phrase bank. Hand-written
 * rules remain as fallback when this sidecar is absent or invalid.
 */
export const CONDUCTOR_STATE_DIM_V1 = 392;
export const CONDUCTOR_STATE_DIM = 1202;
export const CONDUCTOR_STATE_DIM_LEGACY = CONDUCTOR_STATE_DIM_V1;
export const CONDUCTOR_TARGET_DIM = 202;
export const CONDUCTOR_PATTERN_DIM = 48;
export const CONDUCTOR_STEMS = ["drums", "bass", "other", "vocals"] as const;
export const CONDUCTOR_SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;

const clamp01 = (v: number): number => v < 0 ? 0 : v > 1 ? 1 : v;
const sigmoid = (v: number): number => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v))));

interface Dense { w: number[][]; b: number[] }

export interface ConductorArtifact {
  format: string;
  version: number;
  history_cells?: number;
  state_dim: number;
  target_dim: number;
  hidden: number;
  sections: string[];
  stems: string[];
  pattern_dim: number;
  mean: number[];
  std: number[];
  layers: Dense[];
  authority: number;
  report?: {
    section_accuracy?: number;
    section_majority?: number;
    change_auc?: Record<string, number>;
    pattern_mse?: number;
    pattern_baseline_mse?: number;
  };
}

export interface ConductorDecision {
  section: string;
  sectionConfidence: number;
  energy: number;
  change: number[];
  patterns: number[][];
  authority: number;
}

export function isConductorArtifact(value: unknown): value is ConductorArtifact {
  const d = value as ConductorArtifact;
  const validDim = d?.state_dim === CONDUCTOR_STATE_DIM || d?.state_dim === CONDUCTOR_STATE_DIM_V1;
  return Boolean(
    d && (d.format === "wub-conductor-v1" || d.format === "wub-conductor-v2")
    && validDim && d.target_dim === CONDUCTOR_TARGET_DIM && d.pattern_dim === CONDUCTOR_PATTERN_DIM
    && d.sections?.length === 5 && d.stems?.length === 4
    && d.mean?.length === d.state_dim && d.std?.length === d.state_dim
    && d.layers?.length === 3 && Number.isFinite(d.authority)
    && d.layers[0].w.length === d.hidden && d.layers[1].w.length === d.hidden
    && d.layers[2].w.length === CONDUCTOR_TARGET_DIM,
  );
}

/** Assemble the 392-d state from two 4-stem, 4-bar cell summaries. */
export function conductorState(
  older: readonly number[][],
  newer: readonly number[][],
  section: string,
  energy: number,
  energyDelta: number,
  phase = 0,
): number[] {
  const state = new Array<number>(CONDUCTOR_STATE_DIM).fill(0);
  for (let s = 0; s < 4; s++) {
    for (let i = 0; i < CONDUCTOR_PATTERN_DIM; i++) {
      state[s * CONDUCTOR_PATTERN_DIM + i] = older[s]?.[i] ?? 0;
      state[192 + s * CONDUCTOR_PATTERN_DIM + i] = newer[s]?.[i] ?? 0;
    }
  }
  const idx = CONDUCTOR_SECTIONS.indexOf(section as (typeof CONDUCTOR_SECTIONS)[number]);
  state[384 + Math.max(0, idx)] = 1;
  state[389] = clamp01((energy + 60) / 60);
  state[390] = clamp01((energyDelta + 12) / 24);
  state[391] = clamp01(phase);
  return state;
}

/** Six-cell state used by conductor v2 (1202-d). */
export function conductorStateV2(
  history: readonly (readonly number[][])[],
  historySections: readonly string[],
  historyEnergyDb: readonly number[],
  section: string,
  energyDb: number,
  energyDeltaDb: number,
  phase: number,
): number[] {
  const state: number[] = [];
  for (let h = 0; h < 6; h++) {
    const cell = history[history.length - 6 + h];
    for (let s = 0; s < 4; s++) {
      const vec = cell?.[s] ?? [];
      for (let i = 0; i < CONDUCTOR_PATTERN_DIM; i++) state.push(vec[i] ?? 0);
    }
    const label = historySections[historySections.length - 6 + h] ?? section;
    const idx = CONDUCTOR_SECTIONS.indexOf(label as (typeof CONDUCTOR_SECTIONS)[number]);
    for (let k = 0; k < 5; k++) state.push(k === Math.max(0, idx) ? 1 : 0);
    const e = historyEnergyDb[historyEnergyDb.length - 6 + h] ?? energyDb;
    state.push(clamp01((e + 60) / 60));
  }
  const idx = CONDUCTOR_SECTIONS.indexOf(section as (typeof CONDUCTOR_SECTIONS)[number]);
  for (let k = 0; k < 5; k++) state.push(k === Math.max(0, idx) ? 1 : 0);
  state.push(clamp01((energyDb + 60) / 60));
  state.push(clamp01((energyDeltaDb + 12) / 24));
  state.push(clamp01(phase));

  // Harmonic alignment of the most recent cell: pairwise chroma cosine and
  // root-distance of the three pitched stems (bass, other, vocals).
  const last = history[history.length - 1] ?? [];
  const chroma = [1, 2, 3].map((s) => {
    const vec = last[s] ?? [];
    const out = new Array<number>(12).fill(0);
    for (let i = 0; i < 12; i++) out[i] = vec[35 + i] ?? 0;
    return out;
  });
  const roots = [1, 2, 3].map((s) => {
    const vec = last[s] ?? [];
    let best = 0, bestV = -1;
    for (let i = 0; i < 12; i++) if ((vec[23 + i] ?? 0) > bestV) { bestV = vec[23 + i] ?? 0; best = i; }
    return best;
  });
  const dot = (a: number[], b: number[]): number => a.reduce((s, v, i) => s + v * b[i], 0);
  const norm = (a: number[]): number => Math.sqrt(dot(a, a));
  for (let a = 0; a < 3; a++) {
    for (let b = a + 1; b < 3; b++) {
      const d = norm(chroma[a]) * norm(chroma[b]);
      state.push(d > 1e-8 ? dot(chroma[a], chroma[b]) / d : 0);
    }
  }
  for (let a = 0; a < 3; a++) {
    for (let b = a + 1; b < 3; b++) {
      const dist = Math.abs(roots[a] - roots[b]) % 12;
      state.push(1 - Math.min(dist, 12 - dist) / 6);
    }
  }
  return state.slice(0, CONDUCTOR_STATE_DIM);
}

/** Run the compact MLP and decode every head. */
export function inferConductor(model: ConductorArtifact, rawState: readonly number[]): ConductorDecision {
  if (rawState.length !== CONDUCTOR_STATE_DIM) {
    throw new Error(`conductor state dim ${rawState.length} != ${CONDUCTOR_STATE_DIM}`);
  }
  const x = new Float64Array(CONDUCTOR_STATE_DIM);
  for (let i = 0; i < CONDUCTOR_STATE_DIM; i++) {
    const s = model.std[i];
    x[i] = (rawState[i] - model.mean[i]) / (Math.abs(s) > 1e-6 ? s : 1);
  }
  let a = x;
  for (let l = 0; l < 3; l++) {
    const layer = model.layers[l];
    const out = new Float64Array(layer.b.length);
    for (let j = 0; j < layer.b.length; j++) {
      let sum = layer.b[j];
      const row = layer.w[j];
      for (let i = 0; i < a.length; i++) sum += row[i] * a[i];
      out[j] = l === 2 ? sum : Math.tanh(sum);
    }
    a = out;
  }
  let best = 0;
  for (let i = 1; i < 5; i++) if (a[i] > a[best]) best = i;
  const probs = Array.from(a.slice(0, 5));
  const max = Math.max(...probs);
  const exp = probs.map((p) => Math.exp(p - max));
  const denom = exp.reduce((s, v) => s + v, 0) || 1;
  const confidence = exp[best] / denom;
  const patterns: number[][] = [];
  for (let s = 0; s < 4; s++) {
    patterns.push(Array.from(a.slice(10 + s * CONDUCTOR_PATTERN_DIM,
      10 + (s + 1) * CONDUCTOR_PATTERN_DIM), clamp01));
  }
  return {
    section: model.sections[best] ?? "GROOVE_A",
    sectionConfidence: confidence,
    energy: clamp01(a[5]),
    change: [0, 1, 2, 3].map((i) => sigmoid(a[6 + i])),
    patterns,
    authority: clamp01(model.authority),
  };
}
