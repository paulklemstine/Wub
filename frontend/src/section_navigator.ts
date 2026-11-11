/**
 * Learned section navigator for infinite kit playback.
 *
 * Consumes `section_model.json` (mined from corpus section sequences and
 * transitions) to drive an ever-evolving section progression instead of a
 * single fixed 104-bar plan. Pure, dependency-free aside from types.
 */
import type { ArcTarget, MovementBudget } from "./arc_planner";
import { movementBudgetAt as arcMovementBudgetAt } from "./arc_planner";

export interface SectionModel {
  format: string;
  version: number;
  expanded_sections: string[];
  base_sections: string[];
  base_of: Record<string, string>;
  target_energy: Record<string, number>;
  durations: Record<string, { median_bars: number; min_bars: number; max_bars: number; mean_bars: number }>;
  transitions: Record<string, Array<{ to: string; p: number }>>;
  transitions3?: Record<string, Array<{ to: string; p: number }>>;
  start_probs: Array<{ to: string; p: number }>;
}

export interface LearnedTarget {
  /** Ranker-compatible base section (INTRO/GROOVE_A/BUILDUP/DROP/BREAKDOWN). */
  section: string;
  /** Expanded learned section kind (e.g. GROOVE_HIGH, RISE, PEAK). */
  kind: string;
  /** Target energy 0..1. */
  energy: number;
  /** Phase label for display (expanded kind). */
  phase: string;
  /** Bars elapsed inside the current learned section. */
  barInPhase: number;
  /** Total planned bars for the current learned section. */
  phaseBars: number;
  /** Session bar of the next section boundary (or +phaseBars). */
  nextChangeBar: number;
}

const FALLBACK_BASE = "GROOVE_A";

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0.5));
}

function chooseWeighted<T extends { to?: string; section?: string; p: number }>(
  items: readonly T[],
  rng: () => number,
): string | null {
  if (!items || items.length === 0) return null;
  let total = 0;
  for (const it of items) total += Math.max(0, Number(it.p) || 0);
  if (total <= 1e-9) return (items[0].to ?? (items[0] as any).section ?? null) as string | null;
  let r = rng() * total;
  for (const it of items) {
    r -= Math.max(0, Number(it.p) || 0);
    if (r <= 0) return (it.to ?? (it as any).section ?? null) as string | null;
  }
  const last = items[items.length - 1];
  return (last.to ?? (last as any).section ?? null) as string | null;
}

function pickDuration(model: SectionModel, kind: string, rng: () => number): number {
  const d = model.durations?.[kind];
  // Minimum 8 bars per section so targets/feel don't flip every 4 bars.
  // Bridge/outro can be shorter at boundaries, but navigator-level hold
  // keeps the infinite progression legible and less jumpy.
  const MIN_BARS = 8;
  if (!d) return MIN_BARS;
  const medRaw = Math.max(MIN_BARS, Math.round((d.median_bars || 8) / 4) * 4);
  const med = Math.max(MIN_BARS, medRaw);
  const lo = Math.max(MIN_BARS, Math.round((d.min_bars || MIN_BARS) / 4) * 4);
  const hi = Math.min(32, Math.max(med, Math.round((d.max_bars || 16) / 4) * 4));
  // Bias toward stable holds (median and longer) for smoother flow.
  const choices = [med, med, med, med, med + 4, med + 4, hi, lo].filter(
    (v) => v >= lo && v <= hi && v >= MIN_BARS,
  );
  const pick = choices[Math.floor(rng() * choices.length)] ?? med;
  return Math.max(MIN_BARS, Math.round(pick / 4) * 4);
}

/** Infinite, learned section progression state. */
export class SectionNavigator {
  private model: SectionModel | null = null;
  private currentKind = "GROOVE";
  private prevKind = "<START>";
  private barsInto = 0;
  private phaseBars = 8;
  private rng: () => number;

  constructor(rng: () => number = Math.random) {
    this.rng = rng;
  }

  load(model: unknown): boolean {
    if (!model || typeof model !== "object") return false;
    const m = model as Partial<SectionModel>;
    if (m.format !== "wub-section-model-v1" || !Array.isArray(m.expanded_sections)) return false;
    if (!m.transitions || !m.base_of || !m.target_energy) return false;
    this.model = m as SectionModel;
    this.reset();
    console.info(
      `[SectionNav] loaded learned section model • ${m.expanded_sections.length} sections • ` +
      `${(m as any).stats?.tracks ?? "?"} tracks`,
    );
    return true;
  }

  get ready(): boolean {
    return this.model !== null;
  }

  reset(): void {
    const m = this.model;
    if (!m) {
      this.currentKind = "GROOVE";
      this.prevKind = "<START>";
      this.barsInto = 0;
      this.phaseBars = 8;
      return;
    }
    const start = chooseWeighted(m.start_probs || [], this.rng) || "INTRO";
    this.currentKind = m.expanded_sections.includes(start) ? start : "INTRO";
    this.prevKind = "<START>";
    this.barsInto = 0;
    this.phaseBars = pickDuration(m, this.currentKind, this.rng);
  }

  /** Force the current section (e.g. align with cluster rotation/bridge). */
  force(kind: string, bars?: number): void {
    const m = this.model;
    if (m && m.expanded_sections.includes(kind)) {
      this.currentKind = kind;
    } else if (!m) {
      this.currentKind = kind;
    }
    this.barsInto = 0;
    this.phaseBars = bars ?? pickDuration(m!, this.currentKind, this.rng);
  }

  private sampleNext(): string {
    const m = this.model;
    if (!m) return "GROOVE";
    const key3 = `${this.prevKind}|${this.currentKind}`;
    const tri = m.transitions3?.[key3];
    const bi = m.transitions?.[this.currentKind];
    // Prefer trigram when available, else bigram.
    const table = (tri && tri.length > 0 ? tri : bi) || [];
    // Avoid immediate self-repeat unless it's the only option.
    const filtered = table.filter((x) => x.to !== this.currentKind);
    const pick = chooseWeighted(filtered.length > 0 ? filtered : table, this.rng);
    if (pick && m.expanded_sections.includes(pick)) return pick;
    // Fallback: any valid section by energy proximity.
    return "GROOVE";
  }

  /**
   * Advance the navigator to cover `sessionBar` and return the target for the
   * block starting at `sessionBar` with length `blockBars`. Call once per
   * scheduled block in order.
   */
  targetAt(sessionBar: number, blockBars: number): LearnedTarget | ArcTarget | null {
    const m = this.model;
    if (!m) return null;
    const bars = Math.max(4, Math.floor(blockBars || 4));

    // Advance through completed sections until current covers this bar.
    // barsInto tracks progress; we assume blocks are scheduled in order.
    let guard = 0;
    while (this.barsInto >= this.phaseBars && guard++ < 32) {
      const next = this.sampleNext();
      this.prevKind = this.currentKind;
      this.currentKind = next;
      this.barsInto -= this.phaseBars;
      if (this.barsInto < 0) this.barsInto = 0;
      this.phaseBars = pickDuration(m, this.currentKind, this.rng);
    }

    const kind = this.currentKind;
    const base = m.base_of?.[kind] || FALLBACK_BASE;
    const energy = clamp01(m.target_energy?.[kind] ?? 0.6);
    const target: LearnedTarget = {
      section: base,
      kind,
      energy,
      phase: kind,
      barInPhase: this.barsInto,
      phaseBars: this.phaseBars,
      nextChangeBar: Math.max(0, Math.floor(sessionBar)) + (this.phaseBars - this.barsInto),
    };

    // Consume this block.
    this.barsInto += bars;
    // If we landed exactly on/over boundary, next call will transition.
    return target;
  }

  /** Current expanded kind (for status/debug). */
  get current(): string {
    return this.currentKind;
  }
}

/** Map a base 5-class section name to a reasonable expanded kind for display/selection hints. */
export function expandBaseSection(
  base: string,
  energy: number,
  isIntro = false,
  isOutro = false,
): string {
  const b = String(base || "GROOVE_A").toUpperCase();
  const e = clamp01(energy);
  if (isOutro) return "OUTRO";
  if (isIntro || b === "INTRO") return e < 0.35 ? "INTRO" : "WARMUP";
  if (b === "DROP") return e >= 0.85 ? "PEAK" : "DROP";
  if (b === "BUILDUP") return e >= 0.7 ? "RISE" : "BUILDUP";
  if (b === "BREAKDOWN") return e >= 0.4 ? "BRIDGE" : "BREAKDOWN";
  // GROOVE_A
  if (e >= 0.78) return "GROOVE_HIGH";
  if (e >= 0.55) return "GROOVE";
  return "GROOVE_LOW";
}

/** Base section (5-class) for a possibly expanded kind. */
export function baseOf(kind: string, model: SectionModel | null): string {
  if (!kind) return FALLBACK_BASE;
  const up = String(kind).toUpperCase();
  if (model?.base_of?.[up]) return model.base_of[up];
  // static fallback
  const staticMap: Record<string, string> = {
    INTRO: "INTRO", WARMUP: "GROOVE_A", GROOVE_LOW: "GROOVE_A", GROOVE: "GROOVE_A",
    GROOVE_HIGH: "GROOVE_A", BUILDUP: "BUILDUP", RISE: "BUILDUP", DROP: "DROP",
    PEAK: "DROP", BREAKDOWN: "BREAKDOWN", BRIDGE: "BREAKDOWN", OUTRO: "BREAKDOWN",
  };
  return staticMap[up] || (["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"].includes(up) ? up : FALLBACK_BASE);
}
