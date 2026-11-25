// Per-bar mix compatibility scorer (browser).  Scores how well a 4-stem
// combination fits together, emphasizing rhythm lock, with per-bar evaluation
// aggregated by min+mean so one bad bar pulls the mix down.
//
// Weights/shape come from scripts/train_mix_compat.py (mix_compat.json).  When
// the model or per-bar features are unavailable the scorer returns neutral
// (0.5) so it never blocks playback or changes behavior.
//
// Runtime features: the shipped bank does not yet carry a per-bar rhythm
// sidecar; until then we synthesize a coarse per-bar vector from phrase-level
// signals (chroma/energy/rms/bass_root) so the model can still contribute a
// soft prior.  Offline dataset/render+embed training (laptop) produces the
// high-fidelity teacher; per-bar onset grids can be added as a sidecar later.
import type { StemName } from "./types";

const STEMS: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const BARS = 4;

interface MixCompatWeights {
  w1: number[]; b1: number[];
  w2: number[]; b2: number[];
  w3: number[]; b3: number[];
}

interface MixCompatManifest {
  version: number;
  model: string;
  feature_dim: number;
  bars?: number;
  scaler: { mean: number[]; std: number[] };
  weights: MixCompatWeights;
}

export interface MixCompatPhrase {
  id: string;
  energy: number;
  chroma: number[] | null;
  bassRoot: number | null;
  keyShift?: number;
  rmsDbByStem: Record<StemName, number>;
  /** Optional 64-step onset/energy grid per stem (future sidecar). */
  onset?: Partial<Record<StemName, number[]>>;
}

export interface MixCompatScore {
  compat: number;       // 0..1 phrase-level (min/mean aggregate)
  rhythm: number;       // 0..1 rhythm subscore
  minBar: number;       // worst bar compat
  perBar: number[];     // 4 bars
  ready: boolean;       // false => neutral fallback
}

const NEUTRAL: MixCompatScore = { compat: 0.5, rhythm: 0.5, minBar: 0.5, perBar: [0.5, 0.5, 0.5, 0.5], ready: false };

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }

function relu(x: number): number { return x > 0 ? x : 0; }

function rotateChroma(c: number[] | null, shift: number | undefined): number[] | null {
  if (!c || c.length !== 12) return null;
  const s = Math.round(shift ?? 0);
  if (s === 0) return c;
  const out = new Array<number>(12);
  for (let i = 0; i < 12; i++) {
    out[i] = c[((i - s) % 12 + 12) % 12];
  }
  return out;
}

function shiftRoot(r: number | null, shift: number | undefined): number | null {
  if (r === null) return null;
  const s = Math.round(shift ?? 0);
  if (s === 0) return r;
  return ((r + s) % 12 + 12) % 12;
}

function cosine12(a: number[] | null, b: number[] | null): number {
  if (!a || !b || a.length !== 12 || b.length !== 12) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < 12; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

function rootPenalty(a: number | null, b: number | null): number {
  if (a === null || b === null) return 0.25;
  const iv = Math.min(Math.abs(((b - a) % 12 + 12) % 12), 12 - Math.abs(((b - a) % 12 + 12) % 12));
  return ({ 0: 0, 5: 0.05, 7: 0.05, 3: 0.1, 4: 0.1, 2: 0.2, 1: 0.4, 6: 0.6 } as Record<number, number>)[iv] ?? 0.25;
}

function rhythmPairFeatures(
  a: number[] | undefined,
  b: number[] | undefined,
  bar: number,
): [number, number, number, number, number, number] | null {
  if (!a || !b || a.length < 64 || b.length < 64) return null;
  const eps = 1e-6;
  const s = bar * 16;
  let ma = 0, mb = 0;
  for (let i = 0; i < 16; i++) {
    ma += a[s + i];
    mb += b[s + i];
  }
  ma /= 16;
  mb /= 16;

  let corrNum = 0, corrDa = 0, corrDb = 0;
  let downNum = 0, downNa = 0, downNb = 0;
  let offNum = 0, offNa = 0, offNb = 0;
  let jointSum = 0;
  let downSumA = 0, downSumB = 0, offSumA = 0, offSumB = 0;

  for (let i = 0; i < 16; i++) {
    const va = a[s + i];
    const vb = b[s + i];
    const da = va - ma;
    const db = vb - mb;
    corrNum += da * db;
    corrDa += da * da;
    corrDb += db * db;
    jointSum += va * vb;
    if (i % 4 === 0) {
      downNum += va * vb;
      downNa += va * va;
      downNb += vb * vb;
      downSumA += va;
      downSumB += vb;
    } else {
      offNum += va * vb;
      offNa += va * va;
      offNb += vb * vb;
      offSumA += va;
      offSumB += vb;
    }
  }

  const corr = corrNum / (Math.sqrt(corrDa) * Math.sqrt(corrDb) + eps);
  const downAgree = downNum / (Math.sqrt(downNa) * Math.sqrt(downNb) + eps);
  const offAgree = offNum / (Math.sqrt(offNa) * Math.sqrt(offNb) + eps);
  const densityGap = Math.abs(ma - mb);
  const jointPulse = jointSum / 16;
  const syncA = offSumA / (downSumA + eps);
  const syncB = offSumB / (downSumB + eps);
  const syncGap = Math.abs(syncA - syncB);
  return [corr, downAgree, offAgree, densityGap, jointPulse, syncGap];
}

/** Per-bar feature vector (FEATURE_DIM=54), rhythm-first (matches refresh_mix_compat_features.py). */
export function buildBarFeatures(mix: Record<StemName, MixCompatPhrase>, bar: number): number[] {
  const pairs: Array<[StemName, StemName]> = [
    ["drums", "bass"], ["drums", "other"], ["bass", "other"],
    ["drums", "vocals"], ["bass", "vocals"], ["other", "vocals"],
  ];
  const rfeats: number[] = [];
  const corrs: number[] = [];
  for (const [a, b] of pairs) {
    const rp = rhythmPairFeatures(mix[a].onset?.[a], mix[b].onset?.[b], bar);
    if (rp !== null) {
      corrs.push(rp[0]);
      rfeats.push(rp[0], rp[1], rp[2], rp[3], rp[4], rp[5]);
    } else {
      // Fallback when per-bar onset grids are unavailable.
      const corr = a === "drums" && b === "bass" ? 0.15 : 0.0;
      const ea = mix[a].energy, eb = mix[b].energy;
      const densityGap = Math.abs(ea - eb);
      const jointPulse = Math.min(ea, eb);
      const syncGap = densityGap * 0.5;
      corrs.push(corr);
      rfeats.push(corr, 0.0, 0.0, densityGap, jointPulse, syncGap);
    }
  }
  const meanCorr = corrs.reduce((s, x) => s + x, 0) / corrs.length;
  const minCorr = Math.min(...corrs);
  const varCorr = corrs.reduce((s, x) => s + (x - meanCorr) * (x - meanCorr), 0) / corrs.length;
  const stdCorr = Math.sqrt(varCorr);

  const cb = rotateChroma(mix.bass.chroma, mix.bass.keyShift);
  const co = rotateChroma(mix.other.chroma, mix.other.keyShift);
  const cv = rotateChroma(mix.vocals.chroma, mix.vocals.keyShift);
  const rb = shiftRoot(mix.bass.bassRoot, mix.bass.keyShift);
  const ro = shiftRoot(mix.other.bassRoot, mix.other.keyShift);
  const harm = [cosine12(cb, co), cosine12(cb, cv), cosine12(co, cv), rootPenalty(rb, ro)];

  const rms = STEMS.map((s) => mix[s].rmsDbByStem[s] ?? -24);
  const lowCrowd = Math.max(0, (rms[0] + 18) / 18) * Math.max(0, (rms[1] + 18) / 18);
  const vocalBuried = Math.max(0, (rms[2] - rms[3]) / 18);
  const rmsMean = rms.reduce((s, x) => s + x, 0) / 4;
  const rmsSpread = Math.sqrt(rms.reduce((s, x) => s + (x - rmsMean) * (x - rmsMean), 0) / 4);

  const es = STEMS.map((s) => mix[s].energy);
  const eMean = es.reduce((s, x) => s + x, 0) / 4;
  const eStd = Math.sqrt(es.reduce((s, x) => s + (x - eMean) * (x - eMean), 0) / 4);

  return [meanCorr, minCorr, stdCorr, ...rfeats, ...harm, lowCrowd, vocalBuried, rmsSpread, ...rms, eMean, eStd, Math.min(...es), Math.max(...es)];
}

export class MixCompatModel {
  private manifest: MixCompatManifest | null = null;
  private w1: Float32Array | null = null; private b1: Float32Array | null = null;
  private w2: Float32Array | null = null; private b2: Float32Array | null = null;
  private w3: Float32Array | null = null; private b3: Float32Array | null = null;
  private mean: Float32Array | null = null; private std: Float32Array | null = null;
  private d = 0;
  // Optional per-bar onset sidecar (recomputed from shipped kit audio):
  // phrase index -> 4 stems x 64 steps, float32 in 0..1.
  private onsetData: Float32Array | null = null;
  private onsetIndex = new Map<string, number>();

  get ready(): boolean { return this.manifest !== null && this.w1 !== null; }
  get hasOnsets(): boolean { return this.onsetData !== null && this.onsetIndex.size > 0; }

  /** 64-step onset/energy grid for one phrase x stem, or null if unavailable. */
  onsetGrid(phraseId: string, stem: StemName): number[] | null {
    if (!this.onsetData) return null;
    const row = this.onsetIndex.get(phraseId);
    if (row === undefined) return null;
    const si = STEMS.indexOf(stem);
    if (si < 0) return null;
    const base = (row * 4 + si) * 64;
    const out: number[] = new Array(64);
    for (let i = 0; i < 64; i++) out[i] = this.onsetData[base + i];
    return out;
  }

  async load(baseUrl: string): Promise<boolean> {
    const base = baseUrl.replace(/\/+$/, "");
    try {
      const res = await fetch(`${base}/mix_compat.json`);
      if (!res.ok) return false;
      const m = (await res.json()) as MixCompatManifest;
      if (!m || !m.weights || !m.scaler || !m.feature_dim) return false;
      this.manifest = m; this.d = m.feature_dim;
      const w = m.weights;
      this.w1 = new Float32Array(w.w1); this.b1 = new Float32Array(w.b1);
      this.w2 = new Float32Array(w.w2); this.b2 = new Float32Array(w.b2);
      this.w3 = new Float32Array(w.w3); this.b3 = new Float32Array(w.b3);
      this.mean = new Float32Array(m.scaler.mean); this.std = new Float32Array(m.scaler.std);
      console.info(`[MixCompat] loaded ${m.model} • feat ${m.feature_dim}`);
      // best-effort per-bar onset sidecar for true rhythm lock
      try {
        const oRes = await fetch(`${base}/mix_compat_onsets.json`);
        if (oRes.ok) {
          const om = await oRes.json() as { index?: Record<string, number>; feature_file?: string; count?: number };
          const fRes = await fetch(`${base}/${om.feature_file || "mix_compat_onsets.f32"}`);
          if (fRes.ok && om.index) {
            const buf = await fRes.arrayBuffer();
            const arr = new Float32Array(buf);
            const expected = (om.count || 0) * 4 * 64;
            if (expected === 0 || arr.length >= expected) {
              this.onsetData = arr;
              this.onsetIndex.clear();
              for (const [pid, idx] of Object.entries(om.index)) this.onsetIndex.set(pid, idx as number);
              console.info(`[MixCompat] onset sidecar loaded (${this.onsetIndex.size} phrases)`);
            }
          }
        }
      } catch { /* optional */ }
      return true;
    } catch {
      return false;
    }
  }

  private forward(x: number[]): [number, number] {
    if (!this.ready || !this.w1 || !this.b1 || !this.w2 || !this.b2 || !this.w3 || !this.b3 || !this.mean || !this.std) return [0.5, 0.5];
    const d = this.d;
    // scale
    const xs = new Float32Array(d);
    for (let i = 0; i < d; i++) xs[i] = (x[i] - this.mean[i]) / (this.std[i] || 1);
    // layer 1: 128
    const h1 = new Float32Array(128);
    for (let j = 0; j < 128; j++) { let s = this.b1[j]; for (let i = 0; i < d; i++) s += this.w1[j * d + i] * xs[i]; h1[j] = relu(s); }
    // layer 2: 64
    const h2 = new Float32Array(64);
    for (let j = 0; j < 64; j++) { let s = this.b2[j]; for (let i = 0; i < 128; i++) s += this.w2[j * 128 + i] * h1[i]; h2[j] = relu(s); }
    // out: 2 (compat, rhythm)
    const out: [number, number] = [0, 0];
    for (let j = 0; j < 2; j++) { let s = this.b3[j]; for (let i = 0; i < 64; i++) s += this.w3[j * 64 + i] * h2[i]; out[j] = 1 / (1 + Math.exp(-s)); }
    return out;
  }

  /** Score a 4-stem mix per bar (rhythm-prioritized, min-bar aware). */
  score(mix: Record<StemName, MixCompatPhrase>): MixCompatScore {
    if (!this.ready) return NEUTRAL;
    // Attach per-bar onset grids from the sidecar when available so rhythm
    // scoring uses true inter-stem lock (recomputed from shipped kit audio).
    if (this.hasOnsets) {
      for (const stem of STEMS) {
        const p = mix[stem];
        if (!p) continue;
        const grid = this.onsetGrid(p.id, stem);
        if (grid) {
          if (!p.onset) p.onset = {};
          p.onset[stem] = grid;
        }
      }
    }
    const perBar: number[] = [];
    const perRhythm: number[] = [];
    for (let b = 0; b < BARS; b++) {
      const feat = buildBarFeatures(mix, b);
      // pad/trim to model dim
      if (feat.length < this.d) while (feat.length < this.d) feat.push(0);
      const [c, r] = this.forward(feat);
      perBar.push(clamp01(c));
      perRhythm.push(clamp01(r));
    }
    const minBar = Math.min(...perBar);
    const meanBar = perBar.reduce((s, x) => s + x, 0) / perBar.length;
    const compat = 0.6 * minBar + 0.4 * meanBar;
    const rhythm = perRhythm.reduce((s, x) => s + x, 0) / perRhythm.length;
    return { compat, rhythm, minBar, perBar, ready: true };
  }
}
