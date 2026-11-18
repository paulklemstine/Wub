/** Sample a 104-bar section/energy arc from corpus-mined transition statistics.
 * Pure and deterministic for a given seed; no audio, no DOM. The fixed arc in
 * arc_planner.ts stays untouched — this is an optional overlay.
 */
import type { ArcTarget } from "./arc_planner";

export interface SongArcArtifact {
  format: string;
  version: number;
  tracks: number;
  transition: Record<string, Record<string, number>>;
  lengths: Record<string, { p10: number; p50: number; p90: number; mean: number; min_bars: number }>;
  section_energy: Record<string, { target: number; mean_db: number }>;
  section_slope: Record<string, { mean_db_per_bar: number }>;
  archetypes?: Array<{ sections: string[]; p: number }>;
  report?: { energy_scale?: { p5_db: number; p95_db: number } };
}

export interface SampledArcSegment {
  startBar: number;
  bars: number;
  section: string;
  energy: number;
  corpusTarget: number;
  slopeDbPerBar: number;
}

const BASE_ENERGY: Record<string, number> = {
  INTRO: 0.30, GROOVE_A: 0.72, BUILDUP: 0.82, DROP: 0.95,
  BREAKDOWN: 0.20, BRIDGE: 0.20,
};
const ORDER = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"] as const;
const clamp01 = (x: number): number => x < 0 ? 0 : x > 1 ? 1 : x;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickNext(row: Record<string, number> | undefined, rng: () => number, fallback: string): string {
  if (!row) return fallback;
  const entries = Object.entries(row).filter(([, p]) => Number.isFinite(p) && p > 0);
  if (!entries.length) return fallback;
  const total = entries.reduce((s, [, p]) => s + p, 0);
  let roll = rng() * total;
  for (const [section, p] of entries) {
    roll -= p;
    if (roll <= 0) return section;
  }
  return entries[entries.length - 1][0];
}

function sampleBars(model: SongArcArtifact, section: string, rng: () => number): number {
  const row = model.lengths?.[section];
  if (!row) return 8;
  // Triangular between p10 and p90 with a mode at p50, quantized to 4 bars.
  const lo = Math.max(4, row.p10 || 4);
  const hi = Math.max(lo, row.p90 || lo);
  const mid = Math.max(lo, Math.min(hi, row.p50 || lo));
  const u = rng(), v = rng();
  const raw = u < (mid - lo) / Math.max(1e-6, hi - lo + 1e-6)
    ? lo + (mid - lo) * Math.sqrt(u)
    : hi - (hi - mid) * Math.sqrt(1 - u);
  const bars = Math.round((raw + (v - 0.5) * 2) / 4) * 4;
  return Math.max(4, Math.min(32, bars));
}

function segmentEnergy(model: SongArcArtifact, section: string, bars: number): number {
  const base = BASE_ENERGY[section] ?? 0.6;
  const corpus = model.section_energy?.[section]?.target ?? 0.6;
  const slope = Math.max(-1.5, Math.min(1.5, model.section_slope?.[section]?.mean_db_per_bar ?? 0));
  // Small learned offset keeps the familiar musical order while the real
  // section slope supplies the build/fade shape inside the section.
  return clamp01(base + (corpus - 0.6) * 0.15 + slope * 0.006 * Math.max(0, bars / 2));
}

/** Sample a full set; the last 8 bars are always the drumless bridge. */
export function sampleSongArc(model: SongArcArtifact, seed: number, totalBars = 104): SampledArcSegment[] {
  const rng = mulberry32(seed);
  const segments: SampledArcSegment[] = [];
  let bar = 0;
  let section = "INTRO";
  const lastFull = Math.max(8, totalBars - 8);
  let guard = 0;
  while (bar < lastFull && guard++ < 24) {
    const requested = sampleBars(model, section, rng);
    const bars = Math.min(requested, lastFull - bar);
    segments.push({
      startBar: bar, bars, section,
      energy: segmentEnergy(model, section, bars),
      corpusTarget: model.section_energy?.[section]?.target ?? 0.6,
      slopeDbPerBar: model.section_slope?.[section]?.mean_db_per_bar ?? 0,
    });
    bar += bars;
    if (bar >= lastFull) break;
    const idx = ORDER.indexOf(section as (typeof ORDER)[number]);
    const fallback = idx >= 0 && idx + 1 < ORDER.length ? ORDER[idx + 1] : "GROOVE_A";
    section = pickNext(model.transition?.[section], rng, fallback);
  }
  if (bar < lastFull) {
    const bars = lastFull - bar;
    segments.push({ startBar: bar, bars, section, energy: segmentEnergy(model, section, bars),
      corpusTarget: model.section_energy?.[section]?.target ?? 0.6,
      slopeDbPerBar: model.section_slope?.[section]?.mean_db_per_bar ?? 0 });
  }
  // A set with no drop at all feels flat; guarantee one around the midpoint.
  if (!segments.some((s) => s.section === "DROP")) {
    const target = segments.find((s) => s.startBar <= 64 && s.startBar + s.bars > 64) ?? segments[segments.length - 1];
    if (target) {
      target.section = "DROP";
      target.energy = segmentEnergy(model, "DROP", target.bars);
      target.corpusTarget = model.section_energy?.DROP?.target ?? 0.6;
      target.slopeDbPerBar = model.section_slope?.DROP?.mean_db_per_bar ?? 0;
    }
  }
  segments.push({ startBar: lastFull, bars: totalBars - lastFull, section: "BRIDGE",
    energy: 0.20, corpusTarget: 0.2, slopeDbPerBar: -0.6 });
  return segments;
}

export function songArcTargetAt(arc: readonly SampledArcSegment[] | null, sessionBar: number): SampledArcSegment | null {
  if (!arc || !arc.length) return null;
  const bar = ((sessionBar % 104) + 104) % 104;
  for (const segment of arc) {
    if (bar >= segment.startBar && bar < segment.startBar + segment.bars) return segment;
  }
  return arc[arc.length - 1];
}

/** Convenience: an ArcTarget-shaped view for callers that want phase fields. */
export function songArcAsTarget(segment: SampledArcSegment, sessionBar: number): ArcTarget {
  return {
    sessionBar,
    section: segment.section,
    energy: segment.energy,
    phase: (segment.section === "BREAKDOWN" || segment.section === "BRIDGE"
      ? "BRIDGE" : segment.section) as ArcTarget["phase"],
    barInPhase: Math.max(0, sessionBar - segment.startBar),
    phaseBars: segment.bars,
    nextChangeBar: segment.startBar + segment.bars,
  };
}
