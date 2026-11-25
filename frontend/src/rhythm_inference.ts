/* Rhythm-mesh reference inference (pure TS, no dependencies).
 * Implements the 19-feature pair vector exactly as the Python scorer
 * (scripts/score_rhythm_mesh.py::pair_features) and the logistic forward
 * pass from rhythm_mesh_model.json. Keep in sync with FEAT_NAMES order:
 *  0 da, 1 db, 2 simult, 3 union, 4 neither,
 *  5 p_mid_given_first, 6 p_first_given_mid,
 *  7 offbeat_diff, 8 syncop_diff, 9 beat_diff, 10 call_response,
 * 11 density_prod, 12 density_absdiff, 13 pos_profile_corr,
 * 14 logflux_corr, 15 joint_peak, 16 lag_corr_sum,
 * 17 mean_logflux_first, 18 mean_logflux_mid
 */

export type StemName = "drums" | "bass" | "other" | "vocals";
export type StemPairKey = "drums-bass" | "drums-other" | "drums-vocals" | "bass-other" | "bass-vocals" | "other-vocals";

export interface RhythmPairModel {
  coef: number[];          // 19
  intercept: number;
  feature_mean: number[];  // 19
  feature_std: number[];   // 19
}

export interface RhythmMeshModel {
  format: string;
  feature_names: string[];
  density_idx: number[];
  thresholds: Record<StemName, number>;
  pct: number;
  pairs: Record<StemPairKey, RhythmPairModel>;
}

export interface RhythmEntry {
  id: string;
  grid: Uint8Array;   // 64, 0/1
  flux: Float32Array; // 64 step fluxes
}

const SYNC_SET = new Set([3, 6, 11, 14]);

function mean(x: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}

function meanSelected(x: ArrayLike<number>, pred: (i: number) => boolean): number {
  let s = 0, n = 0;
  for (let i = 0; i < x.length; i++) if (pred(i)) { s += x[i]; n++; }
  return n === 0 ? 0 : s / n;
}

function std(x: ArrayLike<number>): number {
  const m = mean(x);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] - m) ** 2;
  return Math.sqrt(s / x.length);
}

function corr(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const sx = std(x), sy = std(y);
  if (sx < 1e-9 || sy < 1e-9) return 0;
  const mx = mean(x), my = mean(y);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] - mx) * (y[i] - my);
  return s / (x.length * sx * sy);
}

export function rhythmMeshFeatures(
  a: RhythmEntry, b: RhythmEntry,
): number[] {
  const ga = a.grid, gb = b.grid;
  if (ga.length !== 64 || gb.length !== 64) throw new Error("grids must be 64 steps");
  const da = mean(ga), db = mean(gb);
  let both = 0, union = 0, neither = 0;
  for (let k = 0; k < 64; k++) {
    if (ga[k] && gb[k]) both++;
    if (ga[k] || gb[k]) union++;
    if (!ga[k] && !gb[k]) neither++;
  }
  both /= 64; union /= 64; neither /= 64;
  const pba = da > 0 ? both / da : 0;
  const pab = db > 0 ? both / db : 0;
  const daOff = meanSelected(ga, (k) => k % 4 === 2);
  const dbOff = meanSelected(gb, (k) => k % 4 === 2);
  let respN = 0;
  for (let k = 0; k < 63; k++) respN += (ga[k] && gb[k + 1] ? 1 : 0) + (gb[k] && ga[k + 1] ? 1 : 0);
  const resp = respN / 2 / 63;
  const pa = new Array<number>(16).fill(0);
  const pb = new Array<number>(16).fill(0);
  for (let k = 0; k < 64; k++) { pa[k % 16] += ga[k]; pb[k % 16] += gb[k]; }
  for (let k = 0; k < 16; k++) { pa[k] /= 4; pb[k] /= 4; }
  const pc = (std(pa) < 1e-6 || std(pb) < 1e-6) ? 0 : corr(pa, pb);
  const fa = a.flux, fb = b.flux;
  const la = Array.from(fa, (v) => Math.log1p(v));
  const lb = Array.from(fb, (v) => Math.log1p(v));
  const c = corr(la, lb);
  const ma = mean(la), mb = mean(lb), sa = std(la), sb = std(lb);
  let joint = 0;
  for (let k = 0; k < 64; k++) {
    const na = (la[k] - ma) / (sa + 1e-9);
    const nb = (lb[k] - mb) / (sb + 1e-9);
    joint += Math.min(na, nb);
  }
  joint /= 64;
  const lag1 = corr(la.slice(0, 63), lb.slice(1, 64));
  const lag2 = corr(lb.slice(0, 63), la.slice(1, 64));
  return [
    da, db, both, union, neither, pba, pab,
    daOff - dbOff,
    meanSelected(ga, (k) => SYNC_SET.has(k % 16)) - meanSelected(gb, (k) => SYNC_SET.has(k % 16)),
    meanSelected(ga, (k) => k % 4 === 0) - meanSelected(gb, (k) => k % 4 === 0),
    resp, da * db, Math.abs(da - db), pc,
    c, joint, lag1 + lag2, ma, mb,
  ];
}

export function rhythmPairScore(
  model: RhythmMeshModel, pair: StemPairKey,
  entryA: RhythmEntry, entryB: RhythmEntry,
): number {
  const m = model.pairs[pair];
  if (!m) throw new Error(`unknown pair ${pair}`);
  const f = rhythmMeshFeatures(entryA, entryB);
  let z = m.intercept;
  for (let i = 0; i < 19; i++) {
    const sd = m.feature_std[i] > 0 ? m.feature_std[i] : 1;
    z += m.coef[i] * ((f[i] - m.feature_mean[i]) / sd);
  }
  return 1 / (1 + Math.exp(-z));
}

export function rhythmMixScore(
  model: RhythmMeshModel,
  stems: Record<StemName, RhythmEntry>,
): { pair: Record<string, number>; min: number; mean: number } {
  const pairs: Record<StemPairKey, [StemName, StemName]> = {
    "drums-bass": ["drums", "bass"], "drums-other": ["drums", "other"],
    "drums-vocals": ["drums", "vocals"], "bass-other": ["bass", "other"],
    "bass-vocals": ["bass", "vocals"], "other-vocals": ["other", "vocals"],
  };
  const pair: Record<string, number> = {};
  for (const [key, [x, y]] of Object.entries(pairs) as [StemPairKey, [StemName, StemName]][]) {
    pair[key] = rhythmPairScore(model, key, stems[x], stems[y]);
  }
  const vals = Object.values(pair);
  return { pair, min: Math.min(...vals), mean: vals.reduce((s, v) => s + v, 0) / vals.length };
}
