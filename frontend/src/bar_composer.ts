/**
 * Hierarchical Multi-Scale Bar Composer & 4-Mode Vocal Arranger.
 *
 * Uses `cell_index.json` + `theory_engine.ts` + optional `CellCompatScorer` to:
 * 1. Slice 4-bar donor phrases in memory (`SliceSpec[]`) and re-sequence them
 *    across tracks into musical motif schemas (`AAAB`, `ABAB`, `ABAC`, `AABB`)
 *    so simultaneous stems come from different songs AND no stem plays >1-2 bars
 *    verbatim from a single source.
 * 2. Lock every pitched bar (`bass`, `other`, `vocals`) to a Roman-numeral chord
 *    progression via chord-quality verification and per-bar `StretchVoice`
 *    semitone automation (`PitchAutomationPoint[]`).
 * 3. Enforce rhythmic counterpoint (kick-bass pocket & synth-vocal unmasking).
 * 4. Apply 4-mode Vocal Arrangement + 2-bar section transition operators
 *    (snare buildup rolls, pre-drop silence gap, drop punch, breakdown dropout).
 */

import { CellCompatScorer } from "./cell_compat";
import { SliceSpec } from "./kit_pcm";
import { SectionPlanContext } from "./song_architect";
import { PitchAutomationPoint } from "./stretch_voice";
import {
  CellIndexData,
  CellStemMeta,
  EssentialRole,
  MotifSchema,
  ProgressionChord,
  StemName,
  TriadQuality,
  popcount16,
  scoreKickBassPocket,
  scoreSynthVocalUnmasking,
  solveCellPitchShift,
} from "./theory_engine";

export interface ComposingPoolEntry {
  kit: { track_id: string; bpm: number; key: string; style: string };
  phrase: { id: string; section: string; start_bar: number; duration_sec: number };
  stem: StemName;
  keyShift: number;
  rate: number;
  energy: number;
  section: string;
  rmsDb: number;
}

export interface ComposedStemPlan<TEntry extends ComposingPoolEntry> {
  stem: StemName;
  /** Distinct donor pool entries loaded for slicing (`itemIndex` indexes into this array). */
  donors: TEntry[];
  /** Sub-phrase slice sequence whose `targetBars` sum exactly to `blockBars`. */
  slices: SliceSpec[];
  /** Semitone shift for the beginning of bar 0. */
  baseSemitones: number;
  /** Per-bar / per-beat semitone automation points across the block (output seconds). */
  pitchPoints: PitchAutomationPoint[];
  /** Motif schema followed by this stem. */
  motifSchema: MotifSchema;
  /** Active vocal arrangement strategy when stem === "vocals". */
  vocalStrategy?: "hook_loop" | "buildup_tighten" | "section_role" | "chord_chop" | "theme_recall";
  /** Theory + purity + pocket/unmasking quality score (0..1). */
  theoryScore: number;
  /** Track IDs contributing slices to this stem. */
  donorTrackIds: string[];
}

export interface ComposedBlockPlan<TEntry extends ComposingPoolEntry> {
  stems: Record<StemName, ComposedStemPlan<TEntry>>;
  meanTheoryScore: number;
  crossTrackCellCount: number;
}

interface SelectedCell<TEntry extends ComposingPoolEntry> {
  entry: TEntry;
  itemIndex: number;
  srcBar: number;
  purity: number;
  semitones: number;
  onsetMask: number;
  quality: TriadQuality;
  meta: CellStemMeta;
  essentialScore: number;
  essentialRole: EssentialRole;
}

interface PrevBarTransitionContext {
  meta: CellStemMeta;
  srcBar: number;
  semitones: number;
  quality: TriadQuality;
  role: EssentialRole;
  section: string;
}

const SCHEMAS: readonly MotifSchema[] = ["AAAB", "ABAB", "ABAC", "AABB"];

export class BarComposer {
  private index: CellIndexData | null = null;
  private loaded = false;
  private lastBlockTail: Partial<Record<StemName, PrevBarTransitionContext>> = {};

  isReady(): boolean {
    return this.loaded && this.index !== null;
  }

  getStyleProgressions(): CellIndexData["style_progressions"] | null {
    return this.index?.style_progressions ?? null;
  }

  async load(url = "/data/kits/cell_index.json"): Promise<boolean> {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return false;
      const raw = (await resp.json()) as Partial<CellIndexData>;
      if (
        !raw ||
        (raw.format !== "cell_index_v1" && raw.format !== "cell_index_v2") ||
        !raw.phrases
      ) {
        return false;
      }
      this.index = raw as CellIndexData;
      this.loaded = true;
      return true;
    } catch {
      return false;
    }
  }

  getPhraseMeta(phraseId: string, stem: StemName): CellStemMeta | null {
    return this.index?.phrases[phraseId]?.[stem] ?? null;
  }

  private pickMotifSchema(
    kitSection: string,
    stem: StemName,
    blockIdx: number,
  ): MotifSchema {
    if (stem === "drums") return blockIdx % 3 === 0 ? "ABAC" : "AAAB";
    const priors = this.index?.section_motif_priors?.[kitSection]?.[stem];
    if (priors) {
      let best: MotifSchema = "AAAB";
      let bestW = -1;
      for (const s of SCHEMAS) {
        const w = priors[s] ?? 0;
        // Deterministic rotation bias so not every block repeats the identical schema
        const variedW = w + ((s.charCodeAt(2) + blockIdx) % 4) * 0.04;
        if (variedW > bestW) {
          bestW = variedW;
          best = s;
        }
      }
      return best;
    }
    return SCHEMAS[blockIdx % SCHEMAS.length];
  }

  private chooseVocalPrimary<TEntry extends ComposingPoolEntry>(
    fallback: TEntry,
    pool: readonly TEntry[],
    ctx: SectionPlanContext,
    isPinned: boolean,
  ): { entry: TEntry; strategy: ComposedStemPlan<TEntry>["vocalStrategy"] } {
    if (isPinned || pool.length === 0 || !this.index) {
      return { entry: fallback, strategy: "section_role" };
    }

    // 1. Recall remembered hook/motif in Verse 2 / Chorus 2 / subsequent Chorus 1 blocks
    if (ctx.recalledVocalPhraseId) {
      const recalled = pool.find((e) => e.phrase.id === ctx.recalledVocalPhraseId);
      if (recalled) {
        return { entry: recalled, strategy: "theme_recall" };
      }
    }

    const rateRef = fallback.rate;
    const scored: Array<{ entry: TEntry; score: number; strategy: ComposedStemPlan<TEntry>["vocalStrategy"] }> = [];
    for (let i = 0; i < Math.min(pool.length, 60); i++) {
      const e = pool[i];
      if (Math.abs(e.rate - rateRef) > 0.04) continue;
      const meta = this.getPhraseMeta(e.phrase.id, "vocals");
      if (!meta) continue;
      const avgP = (meta.p[0] + meta.p[1] + meta.p[2] + meta.p[3]) / 4;
      if (avgP < 0.2) continue;

      const essMax = meta.es ? Math.max(...meta.es) : avgP;
      let s = avgP * 0.50 + essMax * 0.30 + (meta.fid ?? 0.65) * 0.12;
      let strat: ComposedStemPlan<TEntry>["vocalStrategy"] = "section_role";

      if (ctx.isBuildup) {
        strat = "buildup_tighten";
        if (meta.vr === "hook_1bar" || meta.vr === "chop") s += 0.35;
        if (meta.hook) s += 0.25;
      } else if (ctx.role === "CHORUS_1" || ctx.role === "CHORUS_2" || ctx.kitSection === "DROP") {
        strat = "hook_loop";
        if (meta.hook) s += 0.55;
        if (meta.vr === "hook_1bar" || meta.vr === "hook_2bar") s += 0.3;
      } else if (ctx.kitSection === "INTRO") {
        if (meta.vr === "teaser" || meta.vr === "airy") s += 0.45;
      } else if (ctx.kitSection === "BREAKDOWN") {
        if (meta.vr === "airy" || meta.vr === "hook_2bar") s += 0.45;
      } else {
        if (meta.vr === "verse" || meta.vr === "chop") s += 0.35;
      }
      if (e.phrase.id === fallback.phrase.id) s += 0.08;
      scored.push({ entry: e, score: s, strategy: strat });
    }

    if (scored.length === 0) return { entry: fallback, strategy: "section_role" };
    scored.sort((a, b) => b.score - a.score);
    return { entry: scored[0].entry, strategy: scored[0].strategy };
  }

  private findSecondaryDonor<TEntry extends ComposingPoolEntry>(
    stem: StemName,
    primary: TEntry,
    pool: readonly TEntry[],
    forbiddenTracks: Set<string>,
    ctx: SectionPlanContext,
  ): TEntry | null {
    if (!this.index || pool.length === 0) return null;
    const rateRef = primary.rate;
    let best: TEntry | null = null;
    let bestScore = -999;

    const maxScan = Math.min(pool.length, 48);
    for (let i = 0; i < maxScan; i++) {
      const cand = pool[i];
      if (cand.kit.track_id === primary.kit.track_id) continue;
      if (Math.abs(cand.rate - rateRef) > 0.038) continue;
      const meta = this.getPhraseMeta(cand.phrase.id, stem);
      if (!meta) continue;
      const maxBarP = Math.max(meta.p[0], meta.p[1], meta.p[2], meta.p[3]);
      if (maxBarP < 0.25) continue;

      const maxBarEs = meta.es ? Math.max(...meta.es) : maxBarP;
      let s = maxBarP * 0.55 + maxBarEs * 0.35 + (meta.fid ?? 0.65) * 0.12 - Math.abs(cand.rate - rateRef) * 4.0;
      if (!forbiddenTracks.has(cand.kit.track_id)) s += 0.35;
      if (cand.section.toUpperCase() === ctx.kitSection) s += 0.12;
      if (stem === "drums" && meta.fill) s += 0.18;
      if (stem === "vocals" && meta.hook && (ctx.role === "CHORUS_1" || ctx.role === "CHORUS_2")) s += 0.22;

      if (s > bestScore) {
        bestScore = s;
        best = cand;
      }
    }
    return best;
  }

  private pickBestCellForBar<TEntry extends ComposingPoolEntry>(
    stem: StemName,
    donors: readonly TEntry[],
    itemIdx: number,
    chord: ProgressionChord,
    sessionRoot: number,
    drumMaskHint: number | null,
    otherOrVocalMaskHint: number | null,
    preferFill: boolean,
    prevContext?: PrevBarTransitionContext,
    barPos?: number,
    isSectionBoundary?: boolean,
    desiredRole?: EssentialRole,
    compatScorer?: CellCompatScorer,
  ): SelectedCell<TEntry> {
    const entry = donors[itemIdx];
    const meta = this.getPhraseMeta(entry.phrase.id, stem) ?? {
      s: "AAAB",
      p: [0.5, 0.5, 0.5, 0.5],
      r: [sessionRoot, sessionRoot, sessionRoot, sessionRoot],
      q: stem === "drums" ? ["perc", "perc", "perc", "perc"] : ["mono", "mono", "mono", "mono"],
      o: [0, 0, 0, 0],
    };

    let bestBar = 0;
    let bestScore = -999;
    let bestShift = entry.keyShift;
    const roleTransitions = this.index?.essential_role_transitions;

    for (let b = 0; b < 4; b++) {
      const pur = meta.p[b] ?? 0;
      const mask = meta.o[b] ?? 0;
      const root = meta.r[b] ?? -1;
      const qual = meta.q[b] ?? (stem === "drums" ? "perc" : "mono");
      const ess = meta.es?.[b] ?? pur;
      const role: EssentialRole = meta.er?.[b] ?? (preferFill && b === 3 ? "T" : "A");

      const { semitones, penalty } = solveCellPitchShift(
        stem,
        root,
        qual,
        sessionRoot,
        chord.relRoot,
        chord.quality,
        6,
        true,
      );

      let score =
        pur * 0.75 +
        ess * 0.38 +
        (meta.fid ?? 0.65) * 0.08 -
        penalty * 1.15 -
        Math.abs(semitones) * 0.015;
      if (pur < 0.15) score -= 0.8; // Strongly avoid dead/silent bar cells
      if (desiredRole && role === desiredRole) score += 0.14;
      if (prevContext && roleTransitions?.[prevContext.role]?.[role] !== undefined) {
        score += 0.24 * (roleTransitions[prevContext.role][role] - 0.25);
      }

      if (stem === "drums") {
        const dense = popcount16(mask) / 16;
        score += dense * 0.15;
        if (preferFill && b === 3 && (meta.fill || meta.dfill)) score += 0.45;
        if (!preferFill && b < 3) score += 0.1;
      } else if (stem === "bass" && drumMaskHint !== null) {
        score += 0.45 * scoreKickBassPocket(drumMaskHint, mask);
      } else if (stem === "other" && otherOrVocalMaskHint !== null) {
        score += 0.3 * scoreSynthVocalUnmasking(mask, otherOrVocalMaskHint);
      } else if (stem === "vocals") {
        if (meta.cb === b) score += 0.2;
        if (otherOrVocalMaskHint !== null) {
          score += 0.25 * scoreSynthVocalUnmasking(otherOrVocalMaskHint, mask);
        }
      }

      if (prevContext && compatScorer?.hasArrangementHead()) {
        const arrProb = compatScorer.scoreTransition({
          prev: {
            stem,
            meta: prevContext.meta,
            barIdx: prevContext.srcBar,
            semitonesShift: prevContext.semitones,
            targetQuality: chord.quality,
          },
          next: {
            stem,
            meta,
            barIdx: b,
            semitonesShift: semitones,
            targetQuality: chord.quality,
          },
          barPos: barPos ?? b,
          isSectionBoundary: Boolean(isSectionBoundary),
        });
        score += 0.45 * (arrProb - 0.5);
      }

      if (score > bestScore) {
        bestScore = score;
        bestBar = b;
        bestShift = semitones;
      }
    }

    return {
      entry,
      itemIndex: itemIdx,
      srcBar: bestBar,
      purity: meta.p[bestBar] ?? 0.5,
      semitones: bestShift,
      onsetMask: meta.o[bestBar] ?? 0,
      quality: meta.q[bestBar] ?? (stem === "drums" ? "perc" : "mono"),
      meta,
      essentialScore: meta.es?.[bestBar] ?? (meta.p[bestBar] ?? 0.5),
      essentialRole: meta.er?.[bestBar] ?? (preferFill && bestBar === 3 ? "T" : "A"),
    };
  }

  private buildPatternSlots(
    schema: MotifSchema,
    donorsCount: number,
  ): Array<{ donorIdx: number; preferFill: boolean }> {
    const sec = donorsCount > 1 ? 1 : 0;
    if (schema === "ABAB") {
      return [
        { donorIdx: 0, preferFill: false },
        { donorIdx: sec, preferFill: false },
        { donorIdx: 0, preferFill: false },
        { donorIdx: sec, preferFill: true },
      ];
    }
    if (schema === "AABB") {
      return [
        { donorIdx: 0, preferFill: false },
        { donorIdx: 0, preferFill: false },
        { donorIdx: sec, preferFill: false },
        { donorIdx: sec, preferFill: true },
      ];
    }
    if (schema === "ABAC") {
      return [
        { donorIdx: 0, preferFill: false },
        { donorIdx: sec, preferFill: false },
        { donorIdx: 0, preferFill: false },
        { donorIdx: sec, preferFill: true },
      ];
    }
    // AAAB: A (donor 0) -> A (donor 0) -> A/B cross-track (donor 1) -> B turnaround (donor 1)
    return [
      { donorIdx: 0, preferFill: false },
      { donorIdx: 0, preferFill: false },
      { donorIdx: sec, preferFill: false },
      { donorIdx: sec, preferFill: true },
    ];
  }

  composeBlock<TEntry extends ComposingPoolEntry>(args: {
    baseEntries: Record<StemName, TEntry>;
    pools: Record<StemName, TEntry[]>;
    sectionCtx: SectionPlanContext;
    blockBars: number;
    blockDurationSec: number;
    blockIndex: number;
    sessionRoot: number;
    sessionMode: "minor" | "major";
    isBridgeBlock: boolean;
    pinnedStems: Set<StemName>;
    compatScorer?: CellCompatScorer;
  }): ComposedBlockPlan<TEntry> {
    const {
      baseEntries,
      pools,
      sectionCtx,
      blockBars,
      blockDurationSec,
      blockIndex,
      sessionRoot,
      isBridgeBlock,
      pinnedStems,
      compatScorer,
    } = args;

    const barOutSec = blockDurationSec / Math.max(1, blockBars);
    const activeTracks = new Set<string>([
      baseEntries.drums.kit.track_id,
      baseEntries.bass.kit.track_id,
      baseEntries.other.kit.track_id,
      baseEntries.vocals.kit.track_id,
    ]);

    // 1. Order composition: drums -> bass -> vocals -> other (for pocket + unmasking coupling)
    const drumOut: Partial<Record<StemName, ComposedStemPlan<TEntry>>> = {};
    const barDrumMasks: number[] = [];
    const barVocalMasks: number[] = [];
    const barSelectedCells: Record<StemName, SelectedCell<TEntry>[]> = {
      drums: [],
      bass: [],
      other: [],
      vocals: [],
    };

    const order: StemName[] = ["drums", "bass", "vocals", "other"];
    for (const stem of order) {
      let primary = baseEntries[stem];
      let vocalStrategy: ComposedStemPlan<TEntry>["vocalStrategy"] | undefined;
      if (stem === "vocals") {
        const chosen = this.chooseVocalPrimary(
          primary,
          pools.vocals ?? [],
          sectionCtx,
          pinnedStems.has("vocals"),
        );
        primary = chosen.entry;
        vocalStrategy = chosen.strategy;
      }

      const canSplit = !pinnedStems.has(stem) && !(isBridgeBlock && (stem === "drums" || stem === "bass"));
      const secondary = canSplit
        ? this.findSecondaryDonor(stem, primary, pools[stem] ?? [], activeTracks, sectionCtx)
        : null;
      const donors: TEntry[] = secondary ? [primary, secondary] : [primary];
      if (secondary) activeTracks.add(secondary.kit.track_id);

      const schema = this.pickMotifSchema(sectionCtx.kitSection, stem, blockIndex);
      const cycleSlots = this.buildPatternSlots(schema, donors.length);
      const slices: SliceSpec[] = [];
      const pitchPoints: PitchAutomationPoint[] = [];
      const selectedBars: SelectedCell<TEntry>[] = [];

      // Keep bar A / B cells cached so repeating letters (A or B) preserve motif memory!
      const motifCellCache = new Map<string, SelectedCell<TEntry>>();
      let baseSemitones = primary.keyShift;

      for (let b = 0; b < blockBars; b++) {
        const m = b % 4;
        const chord = sectionCtx.progression[m] ?? sectionCtx.progression[0];
        const slot = cycleSlots[m];
        const letter = schema[m];
        const prevCell = selectedBars[b - 1];
        const prevCtx: PrevBarTransitionContext | undefined = prevCell
          ? {
              meta: prevCell.meta,
              srcBar: prevCell.srcBar,
              semitones: prevCell.semitones,
              quality: prevCell.quality,
              role: prevCell.essentialRole,
              section: sectionCtx.kitSection,
            }
          : this.lastBlockTail[stem];
        const isSecBound = b === 0 && Boolean(prevCtx && prevCtx.section !== sectionCtx.kitSection);
        const desiredRole: EssentialRole =
          slot.preferFill || (m === 3 && blockBars >= 4)
            ? "T"
            : ctxIsChorus(sectionCtx) && (stem === "vocals" || stem === "other") && slot.donorIdx === 0
              ? "H"
              : letter === "A"
                ? "A"
                : "B";

        const cacheKey = `${letter}_${slot.donorIdx}`;
        let cell = motifCellCache.get(cacheKey);
        if (!cell) {
          cell = this.pickBestCellForBar(
            stem,
            donors,
            slot.donorIdx,
            chord,
            sessionRoot,
            stem === "bass" ? (barDrumMasks[b] ?? null) : null,
            stem === "other" ? (barVocalMasks[b] ?? null) : null,
            slot.preferFill,
            prevCtx,
            m,
            isSecBound,
            desiredRole,
            compatScorer,
          );
          motifCellCache.set(cacheKey, cell);
        } else {
          // Re-solve pitch shift of cached rhythmic cell against THIS bar's harmonic chord!
          const solved = solveCellPitchShift(
            stem,
            cell.meta.r[cell.srcBar] ?? -1,
            cell.quality,
            sessionRoot,
            chord.relRoot,
            chord.quality,
            6,
            true,
          );
          cell = { ...cell, semitones: solved.semitones };
        }

        selectedBars.push(cell);
        if (stem === "drums") barDrumMasks.push(cell.onsetMask);
        if (stem === "vocals") barVocalMasks.push(cell.onsetMask);
        if (b === 0) baseSemitones = cell.semitones;
        else if (stem !== "drums") {
          pitchPoints.push({ offsetSec: b * barOutSec, semitones: cell.semitones });
        }

        const isLastBar = b === blockBars - 1;
        const isFirstBar = b === 0;
        const barStartFrac = cell.srcBar * 0.25;
        const barEndFrac = (cell.srcBar + 1) * 0.25;

        // Apply 2-bar section transition & vocal arranger operators on slice geometry
        if (stem === "vocals" && sectionCtx.isBuildup && b >= blockBars - 2) {
          const cb = Math.max(0, Math.min(3, cell.meta.cb ?? cell.srcBar));
          const cStart = cb * 0.25;
          if (!isLastBar) {
            // Penultimate buildup bar: 2-beat loop tightening (2 x 0.5 bar)
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.125, targetBars: 0.5, gain: 0.92 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.125, targetBars: 0.5, gain: 1.0 });
          } else if (sectionCtx.isPreDrop) {
            // Final pre-drop bar: 1-beat x2 (0.50) + 1/16th stutter x6 (0.375) + 0.5-beat silence gap (0.125) = 1.000 bar
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.0625, targetBars: 0.25, gain: 0.98 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.0625, targetBars: 0.25, gain: 1.02 });
            for (let k = 0; k < 6; k++) {
              slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.015625, targetBars: 0.0625, gain: 0.85 + 0.04 * k });
            }
            slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.03125, targetBars: 0.125, gain: 0, endGain: 0 });
          } else {
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.0625, targetBars: 0.25, gain: 0.96 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.0625, targetBars: 0.25, gain: 1.0 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.0625, targetBars: 0.25, gain: 1.04 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.0625, targetBars: 0.25, gain: 1.06 });
          }
          continue;
        }

        if (stem === "vocals" && (ctxIsChorus(sectionCtx) || vocalStrategy === "hook_loop")) {
          // Chorus hook looper: tight A-A-A-B or A-B-A-C using 1-bar or 2-bar vocal hook cell
          const hb = cell.meta.hb === 2 ? 2 : 1;
          if (hb === 2 && m === 1) {
            const c0 = selectedBars[b - 1];
            const f0 = Math.min(0.75, c0.srcBar * 0.25 + 0.25);
            slices.push({ itemIndex: c0.itemIndex, startFrac: f0, endFrac: f0 + 0.25, targetBars: 1, gain: 1.02 });
            continue;
          }
          if (sectionCtx.kitSection === "INTRO") {
            // Intro teaser vocal sits cleanly in background without overpowering
            const g = m % 2 === 0 ? 0.8 : 0.35;
            slices.push({ itemIndex: cell.itemIndex, startFrac: barStartFrac, endFrac: barEndFrac, targetBars: 1, gain: g });
            continue;
          }
        }

        if (sectionCtx.isPreDrop && isLastBar) {
          if (stem === "drums") {
            // Drum snare buildup tightening + 0.5-beat (0.125 bar) pre-drop silence gap
            const fStart = cell.meta.fill ? 0.75 : barStartFrac;
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.0625, targetBars: 0.25, gain: 0.92 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.0625, targetBars: 0.25, gain: 0.98 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.03125, targetBars: 0.125, gain: 1.02 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.03125, targetBars: 0.125, gain: 1.05 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.03125, targetBars: 0.125, gain: 1.08 });
            slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.03125, targetBars: 0.125, gain: 0, endGain: 0 });
          } else {
            // Clean 0.5-beat silence breath across bass & other right before the Drop!
            slices.push({ itemIndex: cell.itemIndex, startFrac: barStartFrac, endFrac: barStartFrac + 0.21875, targetBars: 0.875, gain: 1.0, endGain: 0.85 });
            slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.03125, targetBars: 0.125, gain: 0, endGain: 0 });
          }
          continue;
        }

        if (sectionCtx.isBreakdownStart && isFirstBar && (stem === "drums" || stem === "bass")) {
          // Breakdown opens with a 2-beat dropout followed by gentle swell
          slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.125, targetBars: 0.5, gain: 0, endGain: 0 });
          slices.push({ itemIndex: cell.itemIndex, startFrac: barStartFrac + 0.125, endFrac: barEndFrac, targetBars: 0.5, gain: 0.35, endGain: 0.9 });
          continue;
        }

        const downbeatGain =
          sectionCtx.isDropDownbeat && isFirstBar && (stem === "drums" || stem === "other")
            ? 1.08
            : sectionCtx.kitSection === "INTRO" && stem === "vocals"
              ? 0.78
              : 1.0;

        slices.push({
          itemIndex: cell.itemIndex,
          startFrac: barStartFrac,
          endFrac: barEndFrac,
          targetBars: 1,
          gain: downbeatGain,
        });
      }

      barSelectedCells[stem] = selectedBars;
      const tailCell = selectedBars[selectedBars.length - 1];
      const incomingBoundary = Boolean(
        this.lastBlockTail[stem] && this.lastBlockTail[stem]!.section !== sectionCtx.kitSection,
      );

      let arrScoreSum = 0;
      let arrCount = 0;
      if (compatScorer?.hasArrangementHead() && selectedBars.length > 0) {
        if (this.lastBlockTail[stem]) {
          arrScoreSum += compatScorer.scoreTransition({
            prev: {
              stem,
              meta: this.lastBlockTail[stem]!.meta,
              barIdx: this.lastBlockTail[stem]!.srcBar,
              semitonesShift: this.lastBlockTail[stem]!.semitones,
              targetQuality: sectionCtx.progression[0]?.quality ?? "minor",
            },
            next: {
              stem,
              meta: selectedBars[0].meta,
              barIdx: selectedBars[0].srcBar,
              semitonesShift: selectedBars[0].semitones,
              targetQuality: sectionCtx.progression[0]?.quality ?? "minor",
            },
            barPos: 0,
            isSectionBoundary: incomingBoundary,
          });
          arrCount++;
        }
        for (let b = 1; b < selectedBars.length; b++) {
          const ch = sectionCtx.progression[b % 4] ?? sectionCtx.progression[0];
          arrScoreSum += compatScorer.scoreTransition({
            prev: {
              stem,
              meta: selectedBars[b - 1].meta,
              barIdx: selectedBars[b - 1].srcBar,
              semitonesShift: selectedBars[b - 1].semitones,
              targetQuality: ch.quality,
            },
            next: {
              stem,
              meta: selectedBars[b].meta,
              barIdx: selectedBars[b].srcBar,
              semitonesShift: selectedBars[b].semitones,
              targetQuality: ch.quality,
            },
            barPos: b % 4,
            isSectionBoundary: false,
          });
          arrCount++;
        }
      }
      if (tailCell) {
        this.lastBlockTail[stem] = {
          meta: tailCell.meta,
          srcBar: tailCell.srcBar,
          semitones: tailCell.semitones,
          quality: tailCell.quality,
          role: tailCell.essentialRole,
          section: sectionCtx.kitSection,
        };
      }

      const purities = selectedBars.map((c) => c.purity);
      const meanPurity = purities.reduce((a, b) => a + b, 0) / Math.max(1, purities.length);
      const meanEss =
        selectedBars.reduce((a, b) => a + b.essentialScore, 0) / Math.max(1, selectedBars.length);
      const meanArr = arrCount > 0 ? arrScoreSum / arrCount : meanEss;
      let extraTheory = 0;
      if (stem === "bass" && barSelectedCells.drums.length > 0) {
        let pSum = 0;
        for (let b = 0; b < selectedBars.length; b++) {
          pSum += scoreKickBassPocket(barSelectedCells.drums[b]?.onsetMask ?? 0, selectedBars[b].onsetMask);
        }
        extraTheory = pSum / selectedBars.length;
      } else if (stem === "other" && barSelectedCells.vocals.length > 0) {
        let uSum = 0;
        for (let b = 0; b < selectedBars.length; b++) {
          uSum += scoreSynthVocalUnmasking(selectedBars[b].onsetMask, barSelectedCells.vocals[b]?.onsetMask ?? 0);
        }
        extraTheory = uSum / selectedBars.length;
      } else {
        extraTheory = 0.6 * meanPurity + 0.4 * meanEss;
      }

      const donorTrackIds = Array.from(new Set(donors.map((d) => d.kit.track_id)));
      drumOut[stem] = {
        stem,
        donors,
        slices,
        baseSemitones,
        pitchPoints,
        motifSchema: schema,
        vocalStrategy,
        theoryScore: Math.max(0, Math.min(1, 0.42 * meanPurity + 0.36 * extraTheory + 0.22 * meanArr)),
        donorTrackIds,
      };
    }

    // Score full 4-stem mix compatibility on bar 0 if GPU-trained neural scorer is ready
    let neuralCompat = 0.5;
    if (
      compatScorer?.isReady() &&
      barSelectedCells.drums[0] &&
      barSelectedCells.bass[0] &&
      barSelectedCells.other[0] &&
      barSelectedCells.vocals[0]
    ) {
      const ch0 = sectionCtx.progression[0];
      neuralCompat = compatScorer.scoreQuartet({
        drums: { stem: "drums", meta: barSelectedCells.drums[0].meta, barIdx: barSelectedCells.drums[0].srcBar, semitonesShift: 0, targetQuality: ch0.quality },
        bass: { stem: "bass", meta: barSelectedCells.bass[0].meta, barIdx: barSelectedCells.bass[0].srcBar, semitonesShift: barSelectedCells.bass[0].semitones, targetQuality: ch0.quality },
        other: { stem: "other", meta: barSelectedCells.other[0].meta, barIdx: barSelectedCells.other[0].srcBar, semitonesShift: barSelectedCells.other[0].semitones, targetQuality: ch0.quality },
        vocals: { stem: "vocals", meta: barSelectedCells.vocals[0].meta, barIdx: barSelectedCells.vocals[0].srcBar, semitonesShift: barSelectedCells.vocals[0].semitones, targetQuality: ch0.quality },
      });
    }

    const stems = drumOut as Record<StemName, ComposedStemPlan<TEntry>>;
    const meanTheoryScore =
      0.75 *
        ((stems.drums.theoryScore +
          stems.bass.theoryScore +
          stems.other.theoryScore +
          stems.vocals.theoryScore) /
          4) +
      0.25 * neuralCompat;

    const distinctCells = new Set<string>();
    for (const s of order) {
      for (const tid of stems[s].donorTrackIds) distinctCells.add(`${s}:${tid}`);
    }

    return {
      stems,
      meanTheoryScore,
      crossTrackCellCount: distinctCells.size,
    };
  }
}

function ctxIsChorus(ctx: SectionPlanContext): boolean {
  return ctx.role === "CHORUS_1" || ctx.role === "CHORUS_2" || ctx.kitSection === "DROP";
}
