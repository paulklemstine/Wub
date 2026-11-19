/** Browser inference for the learned four-stem "song-ness" critic.
 *
 * Trained on real 4-bar windows from the analyzed songs versus cross-song
 * recombinations. Runs once on the assembled quartet at the final gate; the
 * adjustment is bounded and never renders or reschedules audio.
 */
const clamp01 = (v: number): number => v < 0 ? 0 : v > 1 ? 1 : v;
const sigmoid = (v: number): number => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v))));

interface Layer { w: number[][]; b: number[] }

export interface SongCriticArtifact {
  format: string;
  version: number;
  input_dim: number;
  hidden: number;
  mean: number[];
  std: number[];
  layers: Layer[];
  report?: { val_auc?: number; real_score_p10?: number };
}

export function isSongCriticArtifact(v: unknown): v is SongCriticArtifact {
  const d = v as SongCriticArtifact;
  return Boolean(
    d && d.format === "wub-song-critic-v1" && d.input_dim === 218
    && d.hidden > 0 && d.mean?.length === 218 && d.std?.length === 218
    && d.layers?.length === 3 && d.layers[0].w.length === d.hidden
    && d.layers[1].w.length === 32 && d.layers[2].w.length === 1,
  );
}

export function inferSongCritic(model: SongCriticArtifact, raw: readonly number[]): number {
  let a = new Float64Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const s = model.std[i];
    a[i] = (raw[i] - model.mean[i]) / (Math.abs(s) > 1e-6 ? s : 1);
  }
  for (const layer of model.layers) {
    const out = new Float64Array(layer.b.length);
    for (let j = 0; j < layer.b.length; j++) {
      let sum = layer.b[j];
      for (let i = 0; i < a.length; i++) sum += layer.w[j][i] * a[i];
      out[j] = layer === model.layers[2] ? sum : Math.tanh(sum);
    }
    a = out;
  }
  return clamp01(sigmoid(a[0]));
}
