/** Pure, dependency-free quality metrics for the assembled set.
 * Five dimensions: harmony, arrangement, repetition/churn, seams, and mix.
 * Nothing here renders audio or changes scheduling — it only measures.
 */
const clamp01 = (v: number): number => v < 0 ? 0 : v > 1 ? 1 : v;

export interface HarmonyBar { minCos: number; rootSpread: number }

export interface HarmonyQuality {
  p10: number;
  p50: number;
  clashRate: number;
  meanRootSpread: number;
}

export interface ArrangementEntry { bar: number; section: string; energy: number }

export interface ArrangementQuality {
  sectionChanges: number;
  uniqueSections: number;
  arcDeviation: number;
  meanEnergySlope: number;
}

export interface ChurnQuality {
  uniqueTracks: number;
  uniquePhrases: number;
  changes: number;
  repeatRate: number;
}

export interface SeamQuality {
  dipEvents: number;
  seamEvents: number;
  p95Jump: number;
}

export interface MixQuality {
  gainSpread: number;
  duckSpread: number;
  dominantStem: string | null;
}

export interface QualitySummary {
  harmony: HarmonyQuality;
  arrangement: ArrangementQuality;
  churn: ChurnQuality;
  seams: SeamQuality;
  mix: MixQuality;
  score: number;
  weakest: string;
}

function pct(values: number[], p: number): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(p / 100 * (s.length - 1))))];
}

const mean = (a: number[]): number => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;

export function harmonicQuality(bars: readonly HarmonyBar[], floor = 0.65): HarmonyQuality {
  const cos = bars.map((b) => b.minCos);
  return {
    p10: pct(cos, 10),
    p50: pct(cos, 50),
    clashRate: cos.length ? cos.filter((v) => v < floor).length / cos.length : 0,
    meanRootSpread: mean(bars.map((b) => b.rootSpread)),
  };
}

export function arrangementQuality(entries: readonly ArrangementEntry[], targets: readonly number[]): ArrangementQuality {
  const changes = entries.slice(1).filter((e, i) => e.section !== entries[i].section).length;
  const energies = entries.map((e) => e.energy);
  const slopes = energies.slice(1).map((e, i) => e - energies[i]);
  const deviation = entries.length
    ? mean(entries.map((e, i) => Math.abs(e.energy - (targets[Math.min(i, targets.length - 1)] ?? e.energy))))
    : 0;
  return {
    sectionChanges: changes,
    uniqueSections: new Set(entries.map((e) => e.section)).size,
    arcDeviation: deviation,
    meanEnergySlope: mean(slopes),
  };
}

export function churnQuality(
  stemHistories: Readonly<Record<string, readonly string[]>>,
  trackIds: readonly string[],
): ChurnQuality {
  const phrases = Object.values(stemHistories).flat();
  const changes = phrases.slice(1).filter((id, i) => id !== phrases[i]).length;
  return {
    uniqueTracks: new Set(trackIds).size,
    uniquePhrases: new Set(phrases).size,
    changes,
    repeatRate: phrases.length ? 1 - changes / phrases.length : 0,
  };
}

export function seamQuality(probe: {
  dipEvents?: number; seamEvents?: number; p95MaxJump?: number;
}): SeamQuality {
  return {
    dipEvents: probe.dipEvents ?? 0,
    seamEvents: probe.seamEvents ?? 0,
    p95Jump: probe.p95MaxJump ?? 0,
  };
}

export function mixQuality(gains: Partial<Record<string, number>>): MixQuality {
  const values = Object.values(gains).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!values.length) return { gainSpread: 0, duckSpread: 0, dominantStem: null };
  const lo = Math.min(...values), hi = Math.max(...values);
  const entries = Object.entries(gains).filter((e): e is [string, number] => typeof e[1] === "number");
  entries.sort((a, b) => b[1] - a[1]);
  return {
    gainSpread: hi - lo,
    duckSpread: hi - lo,
    dominantStem: entries[0]?.[0] ?? null,
  };
}

export function qualityScore(
  harmony: HarmonyQuality, arrangement: ArrangementQuality,
  churn: ChurnQuality, seams: SeamQuality, mix: MixQuality,
): { score: number; weakest: string } {
  const h = clamp01((harmony.p10 - 0.55) / 0.3) * (1 - harmony.clashRate * 0.5);
  const a = clamp01(1 - arrangement.arcDeviation / 0.25)
    * clamp01(1 - Math.abs(arrangement.sectionChanges - 4) / 8);
  const c = clamp01(churn.uniqueTracks / 8) * clamp01(1 - Math.abs(churn.repeatRate - 0.45) / 0.5);
  const s = clamp01(1 - seams.dipEvents / Math.max(1, seams.seamEvents) * 2)
    * clamp01(1 - seams.p95Jump / 0.5);
  const m = clamp01(1 - mix.gainSpread / 0.6);
  const parts: Array<[string, number]> = [["harmony", h], ["arrangement", a], ["churn", c], ["seams", s], ["mix", m]];
  parts.sort((x, y) => x[1] - y[1]);
  const score = parts.reduce((sum, p) => sum + p[1], 0) / parts.length;
  return { score: Math.round(score * 1000) / 1000, weakest: parts[0][0] };
}

export function buildQualitySummary(
  remix: Omit<QualitySummary, "score" | "weakest"> | null,
  probe: { dipEvents?: number; seamEvents?: number; p95MaxJump?: number },
  gains: Partial<Record<string, number>>,
): QualitySummary | null {
  if (!remix) return null;
  const seams = seamQuality(probe);
  const mix = mixQuality(gains);
  const { score, weakest } = qualityScore(remix.harmony, remix.arrangement, remix.churn, seams, mix);
  return { ...remix, seams, mix, score, weakest };
}
