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
export const CONDUCTOR_STATE_DIM = 392;
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
  return Boolean(
    d && d.format === "wub-conductor-v1" && d.state_dim === CONDUCTOR_STATE_DIM
    && d.target_dim === CONDUCTOR_TARGET_DIM && d.pattern_dim === CONDUCTOR_PATTERN_DIM
    && d.sections?.length === 5 && d.stems?.length === 4
    && d.mean?.length === CONDUCTOR_STATE_DIM && d.std?.length === CONDUCTOR_STATE_DIM
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

/** Run the compact 392->128->128->202 tanh MLP and decode every head. */
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
