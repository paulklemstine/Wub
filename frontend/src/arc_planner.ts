// Planned 104-bar set arc and per-stem movement policy.
//
// This module is deliberately pure and dependency-free: it contains no audio,
// DOM, manifest, or timer access.  The remix engine can therefore use it to
// choose a section/energy target, budget stem changes, and score a candidate
// transition without coupling the policy to playback state.
import type { StemName } from "./types";

export const ARC_BARS = 104;

export type ArcPhase =
  | "INTRO"
  | "GROOVE_A"
  | "BUILDUP"
  | "DROP"
  | "BREAKDOWN"
  | "BRIDGE";

/** One contiguous segment of the planned set.  Edit this table to retune the arc. */
export interface ArcSegment {
  /** First session bar in the segment (inclusive). */
  startBar: number;
  /** Number of bars in the segment. */
  bars: number;
  /** Section name understood by the joint ranker (SEC_IDX). */
  section: string;
  /** Target stem energy, 0..1. */
  energy: number;
  /** Human-facing phase; BRIDGE maps to section BREAKDOWN for the ranker. */
  phase: ArcPhase;
}

/**
 * The 104-bar arc, aligned to the block grid:
 *
 *   0-16   INTRO      0.30   (4-bar blocks)
 *  16-48   GROOVE_A   0.72   (4-bar blocks)
 *  48-64   BUILDUP    0.82   (two 8-bar blocks)
 *  64-96   DROP       0.95   (two 8-bar blocks + the 16-bar block)
 *  96-104  BRIDGE     0.20   (two 4-bar drumless blocks; section BREAKDOWN)
 *
 * The table must stay sorted, contiguous and exactly ARC_BARS long;
 * `selfTest()` verifies those invariants.
 */
export const ARC_SEGMENTS: readonly ArcSegment[] = [
  { startBar: 0, bars: 16, section: "INTRO", energy: 0.30, phase: "INTRO" },
  { startBar: 16, bars: 32, section: "GROOVE_A", energy: 0.72, phase: "GROOVE_A" },
  { startBar: 48, bars: 16, section: "BUILDUP", energy: 0.82, phase: "BUILDUP" },
  { startBar: 64, bars: 32, section: "DROP", energy: 0.95, phase: "DROP" },
  { startBar: 96, bars: 8, section: "BREAKDOWN", energy: 0.20, phase: "BRIDGE" },
];

export interface ArcTarget {
  /** Normalised session bar, 0..ARC_BARS-1. */
  sessionBar: number;
  /** Ranker-compatible section name.  The bridge reports BREAKDOWN. */
  section: string;
  /** Target energy, 0..1. */
  energy: number;
  /** Human-facing arc phase; distinguishes the bridge from a breakdown. */
  phase: ArcPhase;
  /** Bars elapsed inside this segment. */
  barInPhase: number;
  /** Total bars in this segment. */
  phaseBars: number;
  /** Session bar at which this segment ends (ARC_BARS at the set boundary). */
  nextChangeBar: number;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Wrap a session bar into 0..ARC_BARS-1.  Negative values wrap backwards and
 * ARC_BARS itself maps to bar 0 (the next set starts on the same intro shape).
 */
function normaliseBar(sessionBar: number): number {
  if (!Number.isFinite(sessionBar)) return 0;
  const whole = Math.floor(sessionBar);
  return ((whole % ARC_BARS) + ARC_BARS) % ARC_BARS;
}

/** The planned target for the block beginning at `sessionBar`. */
export function targetAt(sessionBar: number): ArcTarget {
  const bar = normaliseBar(sessionBar);
  let segment = ARC_SEGMENTS[ARC_SEGMENTS.length - 1]!;
  for (const candidate of ARC_SEGMENTS) {
    if (bar >= candidate.startBar && bar < candidate.startBar + candidate.bars) {
      segment = candidate;
      break;
    }
  }
  const end = segment.startBar + segment.bars;
  return {
    sessionBar: bar,
    section: segment.section,
    energy: clamp01(segment.energy),
    phase: segment.phase,
    barInPhase: Math.max(0, bar - segment.startBar),
    phaseBars: segment.bars,
    nextChangeBar: Math.min(ARC_BARS, end),
  };
}

/** Convenience accessor for callers that only need the human-facing phase. */
export function phaseAt(sessionBar: number): ArcPhase {
  return targetAt(sessionBar).phase;
}

export interface MovementBudget {
  /** Soft minimum number of stems that should change in this block. */
  minChanges: number;
  /** Hard-ish maximum number of stems that may change in this block. */
  maxChanges: number;
  /** Longest run of bars a stem may hold without being reconsidered. */
  maxHoldBars: number;
}

/**
 * Per-block stem movement policy.
 *
 * - 8/16-bar blocks ask for at least one change; 4-bar blocks may hold so the
 *   ranker can keep a good mix intact.
 * - The first 8 intro bars are exempt from the minimum so the anchor record can
 *   establish itself.
 * - The 96-104 bridge is a forced hold: only other + vocals play.
 */
export function movementBudgetAt(sessionBar: number, blockBars: number): MovementBudget {
  const target = targetAt(sessionBar);
  const bars = Math.max(1, Math.floor(finiteOr(blockBars, 4)));
  if (target.phase === "BRIDGE") {
    return { minChanges: 0, maxChanges: 0, maxHoldBars: 8 };
  }
  const intro = target.sessionBar < 8;
  return {
    minChanges: intro || bars < 8 ? 0 : 1,
    maxChanges: 2,
    maxHoldBars: 16,
  };
}

export interface HarmonyInput {
  stem: StemName;
  /** Bass root before the transition, chroma pitch class 0..11, or null. */
  fromRoot: number | null;
  /** Bass root after the transition, chroma pitch class 0..11, or null. */
  toRoot: number | null;
  /** 12-bin chroma before the transition, already in session-key space. */
  fromChroma: readonly number[] | null;
  /** 12-bin chroma after the transition, already in session-key space. */
  toChroma: readonly number[] | null;
  /** Semitones applied to the candidate to fit the session key. */
  keyShift: number;
}

// Interval classes: 0, 5 and 7 are consonant; 2/3/4/8/9/10 are mild colour;
// 1, 6 and 11 are the clash intervals.  Vocals get the stricter table.
const BASS_INTERVAL_PENALTY: readonly number[] = [
  0, 0.55, 0.12, 0.10, 0.10, 0, 0.85, 0, 0.15, 0.15, 0.20, 0.50,
];
const VOCAL_INTERVAL_PENALTY: readonly number[] = [
  0, 0.70, 0.16, 0.14, 0.12, 0, 0.95, 0, 0.18, 0.18, 0.24, 0.65,
];

function pitchClass(value: number | null): number | null {
  if (value === null || !Number.isFinite(value)) return null;
  return ((Math.round(value) % 12) + 12) % 12;
}

/** Absolute pitch-class interval, 0..11; null when either root is unknown. */
function rootInterval(fromRoot: number | null, toRoot: number | null): number | null {
  const from = pitchClass(fromRoot);
  const to = pitchClass(toRoot);
  if (from === null || to === null) return null;
  return Math.abs(to - from) % 12;
}

/** Cosine similarity of two 12-bin chroma vectors, or null if either is silent. */
function chromaCosine(
  fromChroma: readonly number[] | null,
  toChroma: readonly number[] | null,
): number | null {
  if (!fromChroma || !toChroma) return null;
  let dot = 0;
  let fromEnergy = 0;
  let toEnergy = 0;
  for (let i = 0; i < 12; i++) {
    const a = fromChroma[i] ?? 0;
    const b = toChroma[i] ?? 0;
    dot += a * b;
    fromEnergy += a * a;
    toEnergy += b * b;
  }
  if (fromEnergy <= 0 || toEnergy <= 0) return null;
  return dot / Math.sqrt(fromEnergy * toEnergy);
}

function keyShiftPenalty(keyShift: number): number {
  if (!Number.isFinite(keyShift)) return 0;
  const excess = Math.max(0, Math.abs(keyShift) - 3);
  return Math.min(0.16, excess * 0.04);
}

/**
 * Soft harmonic cost of moving a stem, 0 (ideal) .. 1 (worst).
 *
 * Drums are unpitched and always free.  Bass uses root-motion interval classes
 * plus a small penalty for a key shift beyond a minor third.  Vocals use a
 * stricter version of the same table and add chroma overlap when available.
 * Other uses chroma overlap only, at a deliberately mild weight.
 */
export function harmonyPenalty(input: HarmonyInput): number {
  const { stem } = input;
  if (stem === "drums") return 0;

  if (stem === "other") {
    const similarity = chromaCosine(input.fromChroma, input.toChroma);
    if (similarity === null) return 0;
    // 0..0.25: large changes in harmonic colour are discouraged, not banned.
    return clamp01(0.25 * (1 - similarity) / 2);
  }

  const isVocal = stem === "vocals";
  const interval = rootInterval(input.fromRoot, input.toRoot);
  let penalty = 0;
  if (interval !== null) {
    const table = isVocal ? VOCAL_INTERVAL_PENALTY : BASS_INTERVAL_PENALTY;
    penalty = table[interval] ?? 0;
  }
  if (!isVocal) {
    penalty += keyShiftPenalty(input.keyShift);
  }
  if (isVocal) {
    const similarity = chromaCosine(input.fromChroma, input.toChroma);
    if (similarity !== null) {
      // Reward overlap and penalise harmonic drift by up to 0.15.
      penalty += 0.15 * (1 - similarity) / 2;
    }
  }
  return clamp01(penalty);
}

/**
 * Vocal changes are sticky: the voice may move on a section change, after a
 * full 16-bar stretch, or on its very first assignment.  This is a soft policy;
 * callers may still move vocals and pay a score penalty instead.
 */
export function vocalChangeAllowed(
  sessionBar: number,
  lastVocalChangeBar: number,
  sectionChanged: boolean,
): boolean {
  if (sectionChanged) return true;
  if (!Number.isFinite(lastVocalChangeBar) || lastVocalChangeBar < 0) return true;
  return sessionBar - lastVocalChangeBar >= 16;
}

export interface TransitionScoreInput {
  /** Joint ranker score for the candidate mix, normally 0..1. */
  jointScore: number;
  /** Planned energy for this block, 0..1. */
  targetEnergy: number;
  /** Energy of the candidate mix, 0..1. */
  candidateEnergy: number;
  /** Novelty of the candidate, 0..1. */
  novelty: number;
  /** How recently/repeatedly this material has played, 0..1. */
  repetition: number;
  /** Harmonic cost from `harmonyPenalty`, 0..1. */
  harmony: number;
  /** Number of stems that would change at this boundary. */
  changedCount: number;
  budget: MovementBudget;
  /** True when the candidate change includes vocals. */
  includesVocals: boolean;
  /** Result of `vocalChangeAllowed` for this boundary. */
  vocalAllowed: boolean;
}

export interface TransitionScore {
  /** Final bounded score, 0..1. */
  score: number;
  /** Human-readable contribution log for status/debug output. */
  reasons: string[];
}

function signed(value: number, digits = 3): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

/**
 * Blend the joint ranker score with the planned arc and movement policy.
 *
 * The weights are intentionally small: the learned ranker remains the primary
 * signal, while energy, novelty, repetition, harmony and budget act as musical
 * guard rails around it.  Novelty/repetition are weighted strongly enough
 * that a never-played phrase can beat a slightly higher-ranked repeat;
 * without that, the flat top of the ranker distribution cycles the same
 * handful of tracks.
 */
export function scoreTransition(input: TransitionScoreInput): TransitionScore {
  const reasons: string[] = [];
  let score = clamp01(finiteOr(input.jointScore, 0));
  reasons.push(`joint ${score.toFixed(3)}`);

  const targetEnergy = clamp01(finiteOr(input.targetEnergy, 0.5));
  const candidateEnergy = clamp01(finiteOr(input.candidateEnergy, 0.5));
  const energyMatch = 1 - Math.abs(candidateEnergy - targetEnergy);
  const energyBonus = 0.08 * energyMatch;
  score += energyBonus;
  reasons.push(`energy ${signed(energyBonus)} (match ${energyMatch.toFixed(2)})`);

  const novelty = clamp01(finiteOr(input.novelty, 0));
  const noveltyBonus = 0.10 * novelty;
  score += noveltyBonus;
  reasons.push(`novelty ${signed(noveltyBonus)}`);

  const repetition = clamp01(finiteOr(input.repetition, 0));
  const repetitionPenalty = 0.15 * repetition;
  score -= repetitionPenalty;
  reasons.push(`repetition ${signed(-repetitionPenalty)}`);

  const harmony = clamp01(finiteOr(input.harmony, 0));
  const harmonyPenalty = 0.15 * harmony;
  score -= harmonyPenalty;
  reasons.push(`harmony ${signed(-harmonyPenalty)}`);

  if (input.includesVocals && !input.vocalAllowed) {
    score -= 0.05;
    reasons.push("vocals held -0.050");
  }

  const changedCount = Math.max(0, Math.floor(finiteOr(input.changedCount, 0)));
  const minChanges = Math.max(0, Math.floor(finiteOr(input.budget.minChanges, 0)));
  const maxChanges = Math.max(0, Math.floor(finiteOr(input.budget.maxChanges, 0)));
  if (changedCount > maxChanges) {
    const extra = changedCount - maxChanges;
    const penalty = 0.08 * extra;
    score -= penalty;
    reasons.push(`over budget ${changedCount} > ${maxChanges} (${signed(-penalty)})`);
  } else if (changedCount < minChanges && maxChanges > 0) {
    // A maxChanges of zero marks a bridge/hold block, which is exempt.
    score -= 0.03;
    reasons.push(`under budget ${changedCount} < ${minChanges} (-0.030)`);
  }

  return { score: clamp01(score), reasons };
}

/** Deterministic smoke test; an empty array means every assertion passed. */
export function selfTest(): string[] {
  const failures: string[] = [];
  const check = (condition: boolean, message: string): void => {
    if (!condition) failures.push(message);
  };
  const close = (a: number, b: number, epsilon = 1e-9): boolean =>
    Math.abs(a - b) <= epsilon;

  check(ARC_BARS === 104, `ARC_BARS should be 104, got ${ARC_BARS}`);

  let cursor = 0;
  for (const segment of ARC_SEGMENTS) {
    check(segment.startBar === cursor, `arc gap/overlap at bar ${cursor}`);
    check(segment.bars > 0, `arc segment at bar ${segment.startBar} has no bars`);
    cursor += segment.bars;
  }
  check(cursor === ARC_BARS, `arc covers ${cursor} bars, expected ${ARC_BARS}`);

  const t0 = targetAt(0);
  check(t0.sessionBar === 0 && t0.section === "INTRO" && t0.phase === "INTRO", "bar 0 target");
  check(close(t0.energy, 0.30), "bar 0 energy");
  check(t0.barInPhase === 0 && t0.phaseBars === 16 && t0.nextChangeBar === 16, "bar 0 phase bounds");

  const t16 = targetAt(16);
  check(t16.section === "GROOVE_A" && t16.barInPhase === 0 && t16.nextChangeBar === 48, "bar 16 target");
  check(close(t16.energy, 0.72), "bar 16 energy");

  const t48 = targetAt(48);
  check(t48.section === "BUILDUP" && t48.barInPhase === 0 && t48.nextChangeBar === 64, "bar 48 target");
  check(close(t48.energy, 0.82), "bar 48 energy");

  const t96 = targetAt(96);
  check(t96.section === "BREAKDOWN" && t96.phase === "BRIDGE", "bar 96 bridge target");
  check(close(t96.energy, 0.20), "bar 96 energy");
  check(t96.barInPhase === 0 && t96.phaseBars === 8 && t96.nextChangeBar === 104, "bar 96 phase bounds");

  const t103 = targetAt(103);
  check(t103.phase === "BRIDGE" && t103.barInPhase === 7 && t103.nextChangeBar === 104, "bar 103 bridge tail");

  const wrapped = targetAt(104);
  check(
    wrapped.sessionBar === 0 && wrapped.section === t0.section && wrapped.phase === t0.phase
      && wrapped.barInPhase === 0 && close(wrapped.energy, t0.energy),
    "bar 104 should wrap to bar 0",
  );
  const negative = targetAt(-1);
  check(
    negative.sessionBar === 103 && negative.phase === "BRIDGE" && negative.barInPhase === 7,
    "bar -1 should wrap to bar 103",
  );

  const introBudget = movementBudgetAt(0, 4);
  check(introBudget.minChanges === 0, "intro should not force a change");
  const eightBudget = movementBudgetAt(8, 8);
  check(
    eightBudget.minChanges === 1 && eightBudget.maxChanges === 2 && eightBudget.maxHoldBars === 16,
    "8-bar budget",
  );
  const longBudget = movementBudgetAt(80, 16);
  check(longBudget.maxChanges === 2 && longBudget.maxHoldBars === 16, "16-bar budget");
  const bridgeBudget = movementBudgetAt(96, 4);
  check(bridgeBudget.minChanges === 0 && bridgeBudget.maxChanges === 0, "bridge budget");

  const drums = harmonyPenalty({
    stem: "drums", fromRoot: 0, toRoot: 6, fromChroma: null, toChroma: null, keyShift: 5,
  });
  check(drums === 0, "drums harmony should be free");

  const bassSame = harmonyPenalty({
    stem: "bass", fromRoot: 0, toRoot: 0, fromChroma: null, toChroma: null, keyShift: 0,
  });
  const bassFifth = harmonyPenalty({
    stem: "bass", fromRoot: 0, toRoot: 7, fromChroma: null, toChroma: null, keyShift: 0,
  });
  const bassThird = harmonyPenalty({
    stem: "bass", fromRoot: 0, toRoot: 4, fromChroma: null, toChroma: null, keyShift: 0,
  });
  const bassSemitone = harmonyPenalty({
    stem: "bass", fromRoot: 0, toRoot: 1, fromChroma: null, toChroma: null, keyShift: 0,
  });
  const bassTritone = harmonyPenalty({
    stem: "bass", fromRoot: 0, toRoot: 6, fromChroma: null, toChroma: null, keyShift: 0,
  });
  check(bassSame === 0 && bassFifth === 0, "bass unison/fifth should be free");
  check(bassThird <= 0.30, "bass major third should be mild");
  check(bassSemitone >= 0.35 && bassSemitone < bassTritone, "bass semitone/tritone ordering");
  check(bassTritone >= 0.35 && bassTritone <= 1, "bass tritone should be penalised");

  const bassShift = harmonyPenalty({
    stem: "bass", fromRoot: 0, toRoot: 7, fromChroma: null, toChroma: null, keyShift: 5,
  });
  check(bassShift > bassFifth, "large key shift should add a small penalty");

  const vocalSemitone = harmonyPenalty({
    stem: "vocals", fromRoot: 0, toRoot: 1, fromChroma: null, toChroma: null, keyShift: 0,
  });
  const vocalTritone = harmonyPenalty({
    stem: "vocals", fromRoot: 0, toRoot: 6, fromChroma: null, toChroma: null, keyShift: 0,
  });
  check(vocalSemitone > bassSemitone, "vocals should be stricter on semitones");
  check(vocalTritone > bassTritone, "vocals should be stricter on tritones");

  const chromaA = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const chromaB = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const otherSame = harmonyPenalty({
    stem: "other", fromRoot: null, toRoot: null, fromChroma: chromaA, toChroma: chromaA, keyShift: 0,
  });
  const otherDifferent = harmonyPenalty({
    stem: "other", fromRoot: null, toRoot: null, fromChroma: chromaA, toChroma: chromaB, keyShift: 0,
  });
  check(otherSame === 0, "other should reward identical chroma");
  check(otherDifferent > otherSame, "other should penalise orthogonal chroma");

  const worstVocal = harmonyPenalty({
    stem: "vocals",
    fromRoot: 0,
    toRoot: 6,
    fromChroma: chromaA,
    toChroma: chromaB,
    keyShift: 5,
  });
  check(worstVocal >= 0 && worstVocal <= 1, "harmony must stay bounded");

  check(vocalChangeAllowed(10, -1, false), "first vocal change should be allowed");
  check(vocalChangeAllowed(10, 5, true), "section change should allow vocals");
  check(!vocalChangeAllowed(20, 5, false), "vocal change before 16 bars should be blocked");
  check(vocalChangeAllowed(21, 5, false), "vocal change at 16 bars should be allowed");
  check(vocalChangeAllowed(30, 5, false), "vocal change after 16 bars should be allowed");

  const base: TransitionScoreInput = {
    jointScore: 0.5,
    targetEnergy: 0.5,
    candidateEnergy: 0.5,
    novelty: 0.2,
    repetition: 0.2,
    harmony: 0.2,
    changedCount: 1,
    budget: { minChanges: 1, maxChanges: 2, maxHoldBars: 16 },
    includesVocals: false,
    vocalAllowed: true,
  };
  const baseScore = scoreTransition(base);
  check(baseScore.score >= 0 && baseScore.score <= 1, "base score should be bounded");
  check(baseScore.reasons.length > 0, "score should explain itself");
  check(baseScore.reasons.every((reason) => reason.length > 0), "reasons should be readable");
  check(scoreTransition(base).score === baseScore.score, "scoring should be deterministic");

  const goodEnergy = scoreTransition({ ...base, targetEnergy: 0.5, candidateEnergy: 0.5 });
  const badEnergy = scoreTransition({ ...base, targetEnergy: 0.5, candidateEnergy: 0.9 });
  check(goodEnergy.score > badEnergy.score, "energy match should be monotonic");

  const moreNovelty = scoreTransition({ ...base, novelty: 0.9 });
  check(moreNovelty.score > baseScore.score, "novelty should be monotonic");

  const moreRepetition = scoreTransition({ ...base, repetition: 0.9 });
  check(moreRepetition.score < baseScore.score, "repetition should be monotonic");

  const moreHarmony = scoreTransition({ ...base, harmony: 0.9 });
  check(moreHarmony.score < baseScore.score, "harmony penalty should be monotonic");

  const overBudget = scoreTransition({ ...base, changedCount: 4 });
  check(overBudget.score < baseScore.score, "over-budget changes should cost more");

  const underBudget = scoreTransition({ ...base, changedCount: 0 });
  check(underBudget.score < baseScore.score, "under-budget changes should cost a little");

  const blockedVocals = scoreTransition({
    ...base, includesVocals: true, vocalAllowed: false,
  });
  const allowedVocals = scoreTransition({
    ...base, includesVocals: true, vocalAllowed: true,
  });
  check(blockedVocals.score < allowedVocals.score, "held vocals should score lower");

  const clampsHigh = scoreTransition({ ...base, jointScore: 2, novelty: 1 });
  const clampsLow = scoreTransition({
    ...base,
    jointScore: -1,
    harmony: 1,
    repetition: 1,
    changedCount: 10,
    includesVocals: true,
    vocalAllowed: false,
  });
  check(clampsHigh.score <= 1 && clampsLow.score >= 0, "scores must clamp to 0..1");

  return failures;
}
