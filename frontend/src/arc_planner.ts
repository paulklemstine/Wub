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
  /**
   * Mean ears quality (0..1) of the changed stems' phrases.  Optional and
   * neutral (0.5) when no ears sidecar is shipped: good phrases win ties,
   * weak phrases stay eligible (quality-first soft prior, never a filter).
   */
  earQuality?: number;
  /**
   * Pre-gated signature bonus for this candidate (0..~0.28 for two stems).
   * Computed by `signatureBonus` in the caller; 0/absent disables the term.
   */
  signatureBonus?: number;
  /**
   * Pre-gated timbre-diversity term for this candidate, roughly
   * -0.08..+0.08.  Computed by `timbreBonus` in the caller.
   */
  timbreBonus?: number;
  /**
   * Mean stem-matched joint-feature cosine between the candidate mix and
   * the current mix, 0..1 (candidate vs current, per changed stem, keyShift
   * stamp slots excluded).  Truthfully near 0.7 for real continuations and
   * near or below 0 for donor mixes; a small centered weight turns it into a
   * tie-breaker, never a veto.  Neutral/untouched when undefined.
   */
  coherence?: number;
  /**
   * Mean learned phrase-syntax score (0..1) for continuity of each changed
   * stem: how well the incoming phrase continues the outgoing phrase's
   * learned phrase-grammar from intra-track transitions (present in the
   * gated successor table; absent entries are exactly neutral at 0.5).
   * Small weight, only re-orders near-best candidates.
   */
  syntax?: number;
  /**
   * Mean rhythm-mesh pair fit (0..1) of the candidate 4-stem mix: the six
   * learned stem-pair logistic models from `rhythm_inference` score whether
   * the stems' 64-step onset grids interlock.  Computed by the caller over
   * the candidate mix; undefined keeps the term exactly neutral.  Small
   * centered weight — a tie-breaker for rhythmic interlock, never a veto.
   */
  rhythmFit?: number;
  /**
   * Learned chord-progression fit (0..1) of the candidate next chord given
   * the recent harmonic context (MLP logprob mapped so uniform = 0.5
   * neutral).  Computed by the caller from the drums-anchored II-chord
   * window; undefined keeps the term exactly neutral.
   */
  progression?: number;
  /**
   * Vocal-edge cleanliness (0..1) of the outgoing/incoming vocal phrases:
   * mean of the outgoing end-edge and incoming start-edge strengths from
   * `vocal_mix.json`.  Only meaningful when `includesVocals` is true (the
   * term is ignored otherwise); undefined keeps the term exactly neutral.
   * Small centered weight — nudges vocal swaps toward clean phrase edges.
   */
  vocalEdge?: number;
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
  // Signature returns intentionally pay a discounted repetition cost (the
  // caller discounts `repetition` itself for signatures; the discount knob
  // lives in FLOW_SIGNATURE_REPETITION_DISCOUNT).  Nothing extra to add here.
  const repetitionPenalty = 0.15 * repetition;
  score -= repetitionPenalty;
  reasons.push(`repetition ${signed(-repetitionPenalty)}`);

  const harmony = clamp01(finiteOr(input.harmony, 0));
  const harmonyPenalty = 0.15 * harmony;
  score -= harmonyPenalty;
  reasons.push(`harmony ${signed(-harmonyPenalty)}`);

  const changedCount = Math.max(0, Math.floor(finiteOr(input.changedCount, 0)));
  const minChanges = Math.max(0, Math.floor(finiteOr(input.budget.minChanges, 0)));
  const maxChanges = Math.max(0, Math.floor(finiteOr(input.budget.maxChanges, 0)));

  const earQuality = clamp01(finiteOr(input.earQuality ?? 0.5, 0.5));
  const earBonus = 0.20 * (earQuality - 0.5);
  score += earBonus;
  reasons.push(`ears ${signed(earBonus)} (q ${earQuality.toFixed(2)})`);

  // Coherence gate (bar-level stem fit): the joint features already know
  // which stem meshes hold together — continuations sit around 0.7 cosine
  // while donors sit near or below 0. A small centered weight breaks ties
  // toward mixes whose layers coincide, never vetoes a ranked pick.
  const coherence = clamp01(finiteOr(input.coherence ?? 0.5, 0.5));
  const coherenceBonus = 0.15 * (coherence - 0.5);
  score += coherenceBonus;
  reasons.push(`coherence ${signed(coherenceBonus)} (${coherence.toFixed(2)})`);

  // Phrase-syntax bonus: continuity the learned phrase grammar assigns to
  // this transition. Per-change magnitude (0.12 solo, 0.08 pairs) keeps it a
  // tie-breaker among near-best candidates only, never a rank override.
  const syntax = clamp01(finiteOr(input.syntax ?? 0.5, 0.5));
  const syntaxWeight = changedCount > 1 ? 0.08 : 0.12;
  const syntaxBonus = syntaxWeight * (syntax - 0.5);
  score += syntaxBonus;
  reasons.push(`syntax ${signed(syntaxBonus)} (${syntax.toFixed(2)})`);

  // Rhythm-mesh bonus (bar-level onset interlock): the six stem-pair logistic
  // models know which hit patterns mesh; a candidate mix whose grids
  // coincide/answer across stems breaks ties, never vetoes a ranked pick.
  const rhythmFit = clamp01(finiteOr(input.rhythmFit ?? 0.5, 0.5));
  const rhythmBonus = 0.12 * (rhythmFit - 0.5);
  score += rhythmBonus;
  reasons.push(`rhythm ${signed(rhythmBonus)} (${rhythmFit.toFixed(2)})`);

  // Chord-progression bonus (4-bar grammar): whether the candidate next
  // chord follows the recent harmonic context the way the corpus taught it.
  const progression = clamp01(finiteOr(input.progression ?? 0.5, 0.5));
  const progressionBonus = 0.10 * (progression - 0.5);
  score += progressionBonus;
  reasons.push(`progression ${signed(progressionBonus)} (${progression.toFixed(2)})`);

  // Vocal-edge nudge (clean phrase boundaries): only meaningful when this
  // boundary actually swaps vocals; mid-phrase vocal splices get a small
  // soft penalty so swaps drift toward clean edges, never a hard block.
  if (input.includesVocals) {
    const vocalEdge = clamp01(finiteOr(input.vocalEdge ?? 0.5, 0.5));
    const vocalBonus = 0.15 * (vocalEdge - 0.5);
    score += vocalBonus;
    reasons.push(`vocal-edge ${signed(vocalBonus)} (${vocalEdge.toFixed(2)})`);
  }

  // Optional flow terms: both are pre-gated by their pure helpers below, so
  // scoreTransition only adds them (bounded, never a filter).
  const signatureBonus = clamp01(finiteOr(input.signatureBonus ?? 0, 0));
  if (signatureBonus > 0) {
    score += signatureBonus;
    reasons.push(`signature ${signed(signatureBonus)}`);
  }
  const timbreBonus = finiteOr(input.timbreBonus ?? 0, 0);
  const timbreClamped = Math.max(
    -FLOW_TIMBRE_OVERUSE_PENALTY * 2,
    Math.min(FLOW_TIMBRE_BONUS, timbreBonus),
  );
  if (Math.abs(timbreClamped) > 1e-9) {
    score += timbreClamped;
    reasons.push(`timbre ${signed(timbreClamped)}`);
  }

  if (input.includesVocals && !input.vocalAllowed) {
    score -= 0.05;
    reasons.push("vocals held -0.050");
  }

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

// ---------------------------------------------------------------------------
// Flow policy (A) signatures, (C) timbre budget, (B) transition techniques.
//
// This block is pure and dependency-free like the rest of arc_planner.ts: no
// audio, DOM, manifest or timer access.  All knobs are module constants so a
// listening rollback is a one-line flip.  The Python mirror in
// wub_core/flow_policy.py implements identical math for `pytest` coverage;
// keep the constants and formulas in sync when retuning.
//
// Every helper returns 0 / no-op when its gate is closed, so a missing signal
// (no taste_scores.json, no energy target, no reverb bus) always falls back
// to today's equal-gain hand-off and the pre-existing score.
// ---------------------------------------------------------------------------

/** Master gate for signature phrase returns.  Flipping restores prior policy. */
export const FLOW_SIGNATURES_ENABLED = true;
/** Signature phrases kept per stem (1-2).  One anchor is enough for the ear
 *  to learn; two gives the ranker a choice so the same window cannot loop. */
export const FLOW_SIGNATURES_PER_STEM = 2;
/** Planned energy at which signatures are fully eligible (DROP peak 0.95).
 *  Below 0.50 the bonus is zero; it ramps linearly up to this threshold. */
export const FLOW_SIGNATURE_PEAK_ENERGY = 0.90;
/** Max additive score bonus per changed signature stem at peak energy.
 *  Small beside the ranker (~1) so signatures win ties, never force a move. */
export const FLOW_SIGNATURE_BONUS = 0.14;
/** Bars a signature must rest between returns within a set (rotation resets
 *  the clock so the new anchor may return immediately).  16 = a half phrase
 *  of history; it is what keeps the return musical instead of a loop. */
export const FLOW_SIGNATURE_MIN_GAP_BARS = 16;
/** While a signature is returning, its repetition cost is multiplied by
 *  (1 - FLOW_SIGNATURE_REPETITION_DISCOUNT) = 0.4 in the caller, so the
 *  novelty/repetition budget still rotates the rest of the bank. */
export const FLOW_SIGNATURE_REPETITION_DISCOUNT = 0.6;

/** Timbre-diversity window in session bars (8 blocks).  Style changes are
 *  musical on this scale; banning repeats longer than this would fight the
 *  movement budget. */
export const FLOW_TIMBRE_WINDOW_BARS = 32;
/** Max soft bonus for a timbre unseen in the window.  An under-used kit drum
 *  kit gets this, never an exclusion. */
export const FLOW_TIMBRE_BONUS = 0.08;
/** Max soft penalty step for an over-used timbre (bounded, not a ban). */
export const FLOW_TIMBRE_OVERUSE_PENALTY = 0.04;

/** Master gate for DJ transition techniques.  False = equal-gain hand-off. */
export const FLOW_TRANSITIONS_ENABLED = true;
/** EQ-OUT tail fires when the incoming stem is at least this much hotter
 *  than the outgoing stem (mid-frequency masking becomes audible above ~0.15). */
export const FLOW_EQ_OUT_MIN_GAIN = 0.15;
/** EQ-OUT low-pass floor in Hz (keeps sub/kick punch, removes 300-3k mask). */
export const FLOW_EQ_OUT_FLOOR_HZ = 300;
/** Incoming vocal RMS (dBFS) at which the fade stretches to avoid mid-vowel
 *  collisions.  Louder than this gets FLOW_VOCAL_FADE_LOUD_SEC. */
export const FLOW_VOCAL_LOUD_RMS_DB = -18;
/** Slow incoming-vocal fade in seconds (within the 100-200 ms spec). */
export const FLOW_VOCAL_FADE_LOUD_SEC = 0.15;

/** Energy at or below which the signature bonus is exactly zero. */
export const FLOW_SIGNATURE_BONUS_FLOOR = 0.50;

/**
 * Signature energy scale in 0..1: 0 at INTRO/BREAKDOWN energies, 1 at/above
 * FLOW_SIGNATURE_PEAK_ENERGY (DROP).  Linear between, so GROOVE (0.72) earns
 * roughly half credit and BUILDUP (0.82) most of it.
 * Mirrors wub_core.flow_policy.signature_energy_scale.
 */
export function signatureEnergyScale(targetEnergy: number): number {
  const energy = clamp01(finiteOr(targetEnergy, 0.5));
  const peak = Math.max(0.51, FLOW_SIGNATURE_PEAK_ENERGY);
  return clamp01((energy - FLOW_SIGNATURE_BONUS_FLOOR) / (peak - FLOW_SIGNATURE_BONUS_FLOOR));
}

export interface SignatureCandidate {
  phraseId: string;
  trackId: string;
  /** Taste quality 0..1, or null/NaN when the sidecar is missing. */
  quality: number | null | undefined;
  /** Ears quality 0..1 (0.5 neutral when the sidecar is absent). */
  earQ: number;
  energy: number;
  section: string;
  isIntro?: boolean;
  isOutro?: boolean;
  once?: boolean;
}

/**
 * Pick 1-2 signature phrase ids from existing shipped signals only
 * (taste_scores quality + ears + energy/section in the manifest).
 *_intro/outro/once phrases are excluded: they are single moments, not
 * returns.  Distinct source tracks are preferred so both signatures do not
 * collapse onto one song.  Deterministic in (phraseId, trackId, quality);
 * missing quality falls back to 0.5 so taste-less banks still get anchors.
 * Mirrors wub_core.flow_policy.pick_signatures.
 */
export function pickSignatures(
  candidates: readonly SignatureCandidate[],
  perStem: number = FLOW_SIGNATURES_PER_STEM,
): Set<string> {
  const picked = new Set<string>();
  if (!Array.isArray(candidates) || candidates.length === 0) return picked;
  const count = Math.max(0, Math.floor(finiteOr(perStem, FLOW_SIGNATURES_PER_STEM)));
  if (count <= 0 || !FLOW_SIGNATURES_ENABLED) return picked;
  const legal = candidates.filter((candidate) => {
    if (!candidate || typeof candidate.phraseId !== "string" || candidate.phraseId === "") {
      return false;
    }
    if (candidate.isIntro === true || candidate.isOutro === true || candidate.once === true) {
      return false;
    }
    return true;
  });
  const scored = legal.map((candidate) => {
    const qualityRaw = candidate.quality;
    const quality = typeof qualityRaw === "number" && Number.isFinite(qualityRaw)
      ? clamp01(qualityRaw)
      : 0.5;
    const earQ = clamp01(finiteOr(candidate.earQ, 0.5));
    const energy = clamp01(finiteOr(candidate.energy, 0.5));
    const section = String(candidate.section ?? "").toUpperCase();
    const sectionBonus = section === "DROP" ? 0.10 : section === "GROOVE_A" ? 0.05 : 0;
    const score = 0.55 * quality + 0.15 * earQ + 0.20 * energy + sectionBonus;
    return { id: candidate.phraseId, track: String(candidate.trackId ?? ""), score };
  });
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    const byId = a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    if (byId !== 0) return byId;
    return a.track < b.track ? -1 : a.track > b.track ? 1 : 0;
  });
  const usedTracks = new Set<string>();
  for (const item of scored) {
    if (picked.size >= count) break;
    if (item.track !== "" && usedTracks.has(item.track)) continue;
    picked.add(item.id);
    if (item.track !== "") usedTracks.add(item.track);
  }
  // If distinct tracks run out (small cluster), fill with the next-best ids.
  if (picked.size < count) {
    for (const item of scored) {
      if (picked.size >= count) break;
      if (!picked.has(item.id)) picked.add(item.id);
    }
  }
  return picked;
}

export interface SignatureBonusInput {
  isSignature: boolean;
  targetEnergy: number;
  /** Session bars since this stem's signature last played (<0 = never). */
  gapBars: number;
  /** Pinned stems never get the bonus: user intent outranks signatures. */
  pinned?: boolean;
}

/**
 * Additive signature bonus for one candidate stem, 0..FLOW_SIGNATURE_BONUS.
 * Zero when disabled, not a signature, pinned, still inside the rest gap, or
 * below the energy floor.  Mirrors wub_core.flow_policy.signature_bonus.
 */
export function signatureBonus(input: SignatureBonusInput): number {
  if (!FLOW_SIGNATURES_ENABLED) return 0;
  if (!input.isSignature) return 0;
  if (input.pinned === true) return 0;
  const gap = Number(input.gapBars);
  // Unknown gap (never played) counts as rested.
  const rested = !Number.isFinite(gap) || gap < 0 || gap >= FLOW_SIGNATURE_MIN_GAP_BARS;
  if (!rested) return 0;
  return FLOW_SIGNATURE_BONUS * signatureEnergyScale(input.targetEnergy);
}

/**
 * Soft timbre-diversity term for one candidate: +FLOW_TIMBRE_BONUS when the
 * candidate's style has not been used in the window, stepping down linearly
 * to FLOW_TIMBRE_OVERUSE_PENALTY at three uses.  Never a ban: the worst
 * candidate still returns a small negative, not a filter.
 * `recentCount` = uses of this style in the 32-bar window; `windowBlocks` =
 * blocks elapsed inside the window (unused here beyond the zero guard, kept
 * for callers that scale by window length).
 * Mirrors wub_core.flow_policy.timbre_bonus.
 */
export function timbreBonus(recentCount: number, windowBlocks: number): number {
  const count = Math.max(0, Math.floor(finiteOr(recentCount, 0)));
  const blocks = Math.max(0, Math.floor(finiteOr(windowBlocks, 0)));
  if (blocks <= 0) return 0;
  const step = (FLOW_TIMBRE_BONUS + FLOW_TIMBRE_OVERUSE_PENALTY) / 3;
  const bonus = FLOW_TIMBRE_BONUS - step * count;
  const floor = -FLOW_TIMBRE_OVERUSE_PENALTY * 2;
  return Math.max(floor, Math.min(FLOW_TIMBRE_BONUS, bonus));
}

export interface TransitionTechniqueInput {
  stem: StemName;
  outgoingEnergy: number;
  incomingEnergy: number;
  incomingRmsDb: number;
  targetEnergy: number;
  isBridge: boolean;
  /** Vocal change performed at this boundary (false = vocal tail unused). */
  vocalChanged?: boolean;
}

export interface TransitionTechnique {
  /** Outgoing stem gets a low-pass sweep to FLOW_EQ_OUT_FLOOR_HZ. */
  eqOut: boolean;
  eqOutFloorHz: number;
  /** Outgoing vocal fades under an extended reverb send. */
  vocalReverbTail: boolean;
  /** Incoming-vocal fade override in seconds, or null for the default fade. */
  vocalFadeInSec: number | null;
}

/**
 * Pure DJ-technique selector.  At most two techniques are produced cleanly:
 * EQ-OUT tail (a pitched stem whose replacement is much hotter) and the
 * vocal treatment (reverb tail on the outgoing vocal, slow fade on a loud
 * incoming vocal).  Drums are deliberately never EQ'd to 300 Hz (it would
 * gut the kick); the drum mask/fill idea needs a separate voice-level click
 * and is left gated off so two clean techniques ship first.
 * All thresholds degrade to the plain equal-gain hand-off when disabled.
 * Mirrors wub_core.flow_policy.select_transition_technique.
 */
export function selectTransitionTechnique(
  input: TransitionTechniqueInput,
): TransitionTechnique {
  const off: TransitionTechnique = {
    eqOut: false,
    eqOutFloorHz: FLOW_EQ_OUT_FLOOR_HZ,
    vocalReverbTail: false,
    vocalFadeInSec: null,
  };
  if (!FLOW_TRANSITIONS_ENABLED) return off;
  if (input.isBridge) return off;
  const outgoing = clamp01(finiteOr(input.outgoingEnergy, 0.5));
  const incoming = clamp01(finiteOr(input.incomingEnergy, 0.5));
  const rmsDb = Number(input.incomingRmsDb);
  const stem = input.stem;
  const result: TransitionTechnique = { ...off };
  if (stem !== "drums" && incoming - outgoing >= FLOW_EQ_OUT_MIN_GAIN) {
    result.eqOut = true;
  }
  if (stem === "vocals") {
    const changed = input.vocalChanged !== false;
    if (changed && incoming >= 0.40) result.vocalReverbTail = true;
    if (Number.isFinite(rmsDb) && rmsDb >= FLOW_VOCAL_LOUD_RMS_DB && incoming >= 0.40) {
      result.vocalFadeInSec = FLOW_VOCAL_FADE_LOUD_SEC;
    }
  }
  return result;
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

  const goodEars = scoreTransition({ ...base, earQuality: 0.9 });
  const badEars = scoreTransition({ ...base, earQuality: 0.1 });
  check(goodEars.score > baseScore.score, "good ears should score above neutral");
  check(badEars.score < baseScore.score, "weak ears should score below neutral");

  const highCoherence = scoreTransition({ ...base, coherence: 0.9 });
  const donorCoherence = scoreTransition({ ...base, coherence: 0.05 });
  check(highCoherence.score > baseScore.score, "continuation-matched stems should score above neutral");
  check(donorCoherence.score < baseScore.score, "donor-like stems should score below neutral");
  check(highCoherence.score - donorCoherence.score < 0.15, "coherence must not dominate the ranker");

  const goodSyntax = scoreTransition({ ...base, syntax: 0.8 });
  const badSyntax = scoreTransition({ ...base, syntax: 0.2 });
  check(goodSyntax.score > baseScore.score, "good syntax should score above neutral");
  check(badSyntax.score < baseScore.score, "bad syntax should score below neutral");
  check(goodSyntax.score - badSyntax.score < 0.15, "syntax must not dominate the ranker");

  const goodRhythm = scoreTransition({ ...base, rhythmFit: 0.9 });
  const badRhythm = scoreTransition({ ...base, rhythmFit: 0.1 });
  check(goodRhythm.score > baseScore.score, "interlocking rhythm should score above neutral");
  check(badRhythm.score < baseScore.score, "colliding rhythm should score below neutral");
  check(goodRhythm.score - badRhythm.score < 0.15, "rhythm must not dominate the ranker");

  const goodProg = scoreTransition({ ...base, progression: 0.9 });
  const badProg = scoreTransition({ ...base, progression: 0.1 });
  check(goodProg.score > baseScore.score, "tonic-following chords should score above neutral");
  check(badProg.score < baseScore.score, "ungrammatical chords should score below neutral");
  check(goodProg.score - badProg.score < 0.15, "progression must not dominate the ranker");

  const cleanVocalEdge = scoreTransition({ ...base, includesVocals: true, vocalEdge: 0.9 });
  const midVocalEdge = scoreTransition({ ...base, includesVocals: true, vocalEdge: 0.1 });
  const heldVocals = scoreTransition({ ...base, includesVocals: true, vocalAllowed: true });
  check(cleanVocalEdge.score > heldVocals.score, "clean vocal edge should score above no-edge baseline");
  check(midVocalEdge.score < heldVocals.score, "mid-phrase vocal edge should score below baseline");
  check(cleanVocalEdge.score - midVocalEdge.score < 0.18, "vocal edge must not dominate the ranker");
  check(scoreTransition({ ...base }).score === baseScore.score, "missing coherence/syntax/rhythm/progression inputs stay neutral");
  check(scoreTransition({ ...base, includesVocals: false, vocalEdge: 0.1 }).score === baseScore.score,
    "vocalEdge is ignored when vocals are not changing");

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

  // Flow policy: signature picking, bonus gating, timbre budget, techniques.
  check(FLOW_SIGNATURES_PER_STEM === 2, "signatures per stem should be 2");
  check(FLOW_SIGNATURE_PEAK_ENERGY >= 0.85 && FLOW_SIGNATURE_PEAK_ENERGY <= 0.95, "signature peak near DROP");
  check(FLOW_SIGNATURE_MIN_GAP_BARS === 16, "signature gap in bars");
  check(FLOW_TIMBRE_WINDOW_BARS === 32, "timbre window in bars");

  const sigCandidates = [
    { phraseId: "a", trackId: "t1", quality: 0.9, earQ: 0.9, energy: 0.95, section: "DROP" },
    { phraseId: "b", trackId: "t2", quality: 0.8, earQ: 0.5, energy: 0.90, section: "DROP" },
    { phraseId: "c", trackId: "t1", quality: 0.6, earQ: 0.9, energy: 0.95, section: "DROP" },
    { phraseId: "intro", trackId: "t3", quality: 1, earQ: 1, energy: 0.95, section: "DROP", isIntro: true },
  ];
  const picked = pickSignatures(sigCandidates);
  check(picked.size === 2 && picked.has("a") && picked.has("b") && !picked.has("c") && !picked.has("intro"),
    "signatures prefer high quality, distinct tracks, no intro/outro/once");
  check(pickSignatures(sigCandidates, 1).size === 1, "perStem cap should be honoured");
  check(pickSignatures([]).size === 0, "no candidates -> no signatures");
  check(pickSignatures(sigCandidates.map((c) => ({ ...c, quality: null }))).size === 2,
    "missing quality still yields anchors");

  check(signatureBonus({ isSignature: true, targetEnergy: 0.95, gapBars: 20 }) > 0, "signature at peak");
  check(signatureBonus({ isSignature: true, targetEnergy: 0.20, gapBars: 20 }) === 0, "signature backs off at low energy");
  check(signatureBonus({ isSignature: true, targetEnergy: 0.95, gapBars: 4 }) === 0, "signature inside gap is zero");
  check(signatureBonus({ isSignature: false, targetEnergy: 0.95, gapBars: 20 }) === 0, "non-signature is zero");
  check(signatureBonus({ isSignature: true, targetEnergy: 0.95, gapBars: 20, pinned: true }) === 0, "pinned stem gets no signature bonus");
  check(signatureBonus({ isSignature: true, targetEnergy: 0.95, gapBars: -1 }) > 0, "never-played signature is rested");
  check(signatureEnergyScale(0.95) > signatureEnergyScale(0.72) && signatureEnergyScale(0.72) > signatureEnergyScale(0.30),
    "signature scale should be monotonic in energy");

  const withSig = scoreTransition({ ...base, signatureBonus: 0.1 });
  check(withSig.score > baseScore.score, "signature term should add");
  check(withSig.reasons.some((r) => r.includes("signature")), "signature term should explain itself");
  const withTimbre = scoreTransition({ ...base, timbreBonus: 0.05 });
  check(withTimbre.score > baseScore.score, "timbre term should add");
  const withTimbrePenalty = scoreTransition({ ...base, timbreBonus: -0.04 });
  check(withTimbrePenalty.score < baseScore.score, "over-used timbre should cost");

  check(timbreBonus(0, 8) === FLOW_TIMBRE_BONUS, "unseen timbre earns the full bonus");
  check(timbreBonus(3, 8) < 0, "thrice-used timbre is penalised");
  check(timbreBonus(0, 0) === 0, "empty window is neutral");
  check(timbreBonus(1, 8) < timbreBonus(0, 8) && timbreBonus(2, 8) < timbreBonus(1, 8),
    "timbre bonus should step down with use");

  const eqOut = selectTransitionTechnique({
    stem: "other", outgoingEnergy: 0.40, incomingEnergy: 0.80,
    incomingRmsDb: -20, targetEnergy: 0.95, isBridge: false,
  });
  check(eqOut.eqOut === true && eqOut.eqOutFloorHz === 300, "EQ-OUT when incoming is much hotter");
  const noEq = selectTransitionTechnique({
    stem: "other", outgoingEnergy: 0.70, incomingEnergy: 0.75,
    incomingRmsDb: -20, targetEnergy: 0.95, isBridge: false,
  });
  check(noEq.eqOut === false, "EQ-OUT needs a real energy gain");
  const drumEq = selectTransitionTechnique({
    stem: "drums", outgoingEnergy: 0.40, incomingEnergy: 0.90,
    incomingRmsDb: -20, targetEnergy: 0.95, isBridge: false,
  });
  check(drumEq.eqOut === false, "drums are never EQ'd to 300 Hz");
  const vocalTech = selectTransitionTechnique({
    stem: "vocals", outgoingEnergy: 0.60, incomingEnergy: 0.70,
    incomingRmsDb: -12, targetEnergy: 0.95, isBridge: false, vocalChanged: true,
  });
  check(vocalTech.vocalReverbTail === true, "vocal change gets a reverb tail");
  check(vocalTech.vocalFadeInSec !== null && vocalTech.vocalFadeInSec! >= 0.10 && vocalTech.vocalFadeInSec! <= 0.20,
    "loud incoming vocal fades in over 100-200 ms");
  const quietVocalTech = selectTransitionTechnique({
    stem: "vocals", outgoingEnergy: 0.60, incomingEnergy: 0.70,
    incomingRmsDb: -30, targetEnergy: 0.95, isBridge: false, vocalChanged: true,
  });
  check(quietVocalTech.vocalFadeInSec === null, "quiet vocal keeps the default fade");
  const bridgeTech = selectTransitionTechnique({
    stem: "vocals", outgoingEnergy: 0.40, incomingEnergy: 0.80,
    incomingRmsDb: -12, targetEnergy: 0.20, isBridge: true, vocalChanged: true,
  });
  check(bridgeTech.eqOut === false && bridgeTech.vocalReverbTail === false, "bridge stays plain");

  return failures;
}
