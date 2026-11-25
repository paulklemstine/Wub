/**
 * Song foundation: stable 8-bar loops + 32-bar foundations that repeat
 * like real training songs. TypeScript mirror of `wub_core/song_foundation.py`.
 *
 * Mined corpus stats (156 tracks):
 * - within-section similarity at lag 8: coarse 0.425 / motif 0.346
 * - section durations: INTRO 8, GROOVE 12-16, BUILDUP 5, DROP 9, BREAKDOWN 15
 * - 8-bar return: drums 51%, bass 18%, other 15%, vocals 8%
 * - novelty: drums 41%, bass 79%, other 80%, vocals 90%
 */

import type { StemName } from "./types";

export const FOUNDATION_BARS = 32;
export const LOOP_BARS = 8;

export const RETURN_RATE_8: Record<StemName, number> = {
  drums: 0.51,
  bass: 0.18,
  other: 0.15,
  vocals: 0.08,
};

export const NOVELTY_RATE: Record<StemName, number> = {
  drums: 0.41,
  bass: 0.79,
  other: 0.80,
  vocals: 0.90,
};

export const MAX_TEMPO_FOUNDATION = 0.03;
export const MAX_TEMPO_DECORATIVE = 0.06;
export const MAX_SHIFT_FOUNDATION = 3;
export const MAX_SHIFT_DECORATIVE = 4;
export const MIN_CHROMA_COSINE = 0.60;

const FORBIDDEN_INTERVALS = new Set([1, 6, 11]);

export function sectionForBar(barInSet: number): { section: string; energy: number } {
  const b = ((barInSet % 104) + 104) % 104;
  if (b < 8) return { section: "INTRO", energy: 0.30 };
  if (b < 24) return { section: "GROOVE_A", energy: 0.68 };
  if (b < 32) return { section: "BUILDUP", energy: 0.84 };
  if (b < 48) return { section: "DROP", energy: 0.94 };
  if (b < 56) return { section: "BREAKDOWN", energy: 0.44 };
  if (b < 72) return { section: "GROOVE_A", energy: 0.74 };
  if (b < 80) return { section: "BUILDUP", energy: 0.88 };
  if (b < 96) return { section: "DROP", energy: 0.96 };
  return { section: "BREAKDOWN", energy: 0.25 };
}

export interface SongBudget {
  minChanges: number;
  maxChanges: number;
  maxHoldBars: number;
}

export function movementBudgetSong(
  _bar: number,
  blockBars: number,
  sectionChanged: boolean,
): SongBudget {
  if (blockBars <= 4) {
    if (sectionChanged) return { minChanges: 1, maxChanges: 2, maxHoldBars: 32 };
    return { minChanges: 0, maxChanges: 1, maxHoldBars: 32 };
  }
  if (sectionChanged) return { minChanges: 1, maxChanges: 3, maxHoldBars: 32 };
  return { minChanges: 0, maxChanges: 1, maxHoldBars: 32 };
}

export function isCompatible(
  stem: StemName,
  rate: number,
  keyShift: number,
  chromaCosine: number | null,
  rootInterval: number | null,
  isFoundation = true,
): { ok: boolean; reason: string } {
  if (stem === "drums") {
    const limit = isFoundation ? MAX_TEMPO_FOUNDATION : MAX_TEMPO_DECORATIVE;
    if (Math.abs(rate - 1) > limit + 1e-9) {
      return { ok: false, reason: `tempo ${rate.toFixed(3)} exceeds ${limit.toFixed(2)}` };
    }
    return { ok: true, reason: "ok" };
  }
  const maxRate = isFoundation ? MAX_TEMPO_FOUNDATION : MAX_TEMPO_DECORATIVE;
  if (Math.abs(rate - 1) > maxRate + 1e-9) {
    return { ok: false, reason: `tempo ${rate.toFixed(3)} exceeds ${maxRate.toFixed(2)}` };
  }
  const maxShift = isFoundation ? MAX_SHIFT_FOUNDATION : MAX_SHIFT_DECORATIVE;
  if (Math.abs(keyShift) > maxShift) {
    return { ok: false, reason: `key shift ${keyShift} exceeds ${maxShift}` };
  }
  if (rootInterval !== null && FORBIDDEN_INTERVALS.has(((rootInterval % 12) + 12) % 12)) {
    return { ok: false, reason: `forbidden interval ${rootInterval % 12}` };
  }
  if (chromaCosine !== null && chromaCosine < MIN_CHROMA_COSINE) {
    return { ok: false, reason: `chroma ${chromaCosine.toFixed(2)} below floor` };
  }
  return { ok: true, reason: "ok" };
}

export function shouldReturn(stem: StemName, ageBars: number): number {
  const base = RETURN_RATE_8[stem] ?? 0.15;
  if (ageBars === 8) return base;
  if (ageBars === 16) return base * 0.8;
  if (ageBars === 32) return base * 0.7;
  return 0;
}

export function repetitionBonus(
  stem: StemName,
  plays: number,
  ageBars: number | null,
): number {
  let bonus = 0;
  if (ageBars !== null && (ageBars === 8 || ageBars === 16 || ageBars === 32)) {
    bonus += shouldReturn(stem, ageBars) * 0.45;
  }
  if (plays === 0) {
    bonus += (NOVELTY_RATE[stem] ?? 0.7) * 0.20;
  } else if (plays === 1) {
    bonus -= ({ drums: 0.02, bass: 0.06, other: 0.06, vocals: 0.10 } as Record<string, number>)[stem] ?? 0.06;
  } else {
    bonus -= ({ drums: 0.08, bass: 0.15, other: 0.15, vocals: 0.25 } as Record<string, number>)[stem] ?? 0.15;
  }
  return bonus;
}

export function buildLoopOrder(schema: string, blockBars: number): number[] {
  const slots = Math.max(1, Math.floor(blockBars / 4));
  const out: number[] = [];
  if (schema === "ABAB") {
    for (let i = 0; i < slots; i++) out.push(i % 2);
    return out;
  }
  for (let i = 0; i < slots; i++) out.push(Math.floor(i / 2) % 2);
  return out;
}

export function selfTestSongFoundation(): void {
  const check = (cond: boolean, msg: string) => {
    if (!cond) throw new Error(`song_foundation: ${msg}`);
  };
  check(sectionForBar(0).section === "INTRO", "bar 0 intro");
  check(sectionForBar(10).section === "GROOVE_A", "bar 10 groove");
  check(sectionForBar(34).section === "DROP", "bar 34 drop");
  check(movementBudgetSong(10, 4, false).maxHoldBars === 32, "hold 32");
  check(!isCompatible("bass", 1.02, 2, 0.75, 6).ok, "tritone rejected");
  check(isCompatible("drums", 1.02, 0, null, null).ok, "drums ok");
  check(!isCompatible("drums", 1.10, 0, null, null).ok, "drums tempo rejected");
  check(shouldReturn("drums", 8) > shouldReturn("vocals", 8), "drums return more");
  check(buildLoopOrder("ABAB", 16).join(",") === "0,1,0,1", "abab order");
  check(buildLoopOrder("AABB", 16).join(",") === "0,0,1,1", "aabb order");
}
