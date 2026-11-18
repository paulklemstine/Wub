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
import { isContiguousFullPhrase, SliceSpec } from "./kit_pcm";
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
  scoreBarSeamContinuity,
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

export interface VocalBarNet {
  mean: number[];
  std: number[];
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
}

export interface VocalBarModel {
  format: string;
  version: number;
  input_dim: number;
  next_dim: number;
  next: VocalBarNet;
  length: VocalBarNet;
}

export interface VocalFlowPriors {
  format: string;
  run_lengths: Record<string, number>;
  distinct_bars_per_active: Record<string, number>;
  variation_rate: number;
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
  /** True when this stem flows as an uncut contiguous phrase without sub-bar splices. */
  isContiguousUncut: boolean;
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

export interface VocalActivityInfo {
  frac: number;
  barFracs: readonly [number, number, number, number];
  activeBars: number;
}

const SCHEMAS: readonly MotifSchema[] = ["AAAB", "ABAB", "ABAC", "AABB"];

export class BarComposer {
  /** Optional learned next-vocal-bar model (activity/chroma + run length). */
  vocalBarModel: VocalBarModel | null = null;
  /** Optional learned vocal-development prior from real songs. */
  vocalFlow: VocalFlowPriors | null = null;

  private index: CellIndexData | null = null;
  private loaded = false;
  private lastBlockTail: Partial<Record<StemName, PrevBarTransitionContext>> = {};
  private vocalActivity: Map<string, VocalActivityInfo> | null = null;

  setVocalActivity(map: Map<string, VocalActivityInfo> | null): void {
    this.vocalActivity = map;
  }

  private isVocalBarActive(phraseId: string, barIdx: number, meta: CellStemMeta): boolean {
    const b = ((barIdx % 4) + 4) % 4;
    const pur = meta.p[b] ?? 0;
    const act = this.vocalActivity?.get(phraseId);
    if (act) {
      const bf = act.barFracs[b] ?? 0;
      const onsets = meta.o ? popcount16(meta.o[b] ?? 0) : 1;
      // Accept bar if vocal detector finds >=2 active 16ths AND (either onsets present or sustained >=25% singing),
      // while rejecting silent noise-floor bars even if chroma purity reads high.
      return bf >= 0.125 && (onsets > 0 || bf >= 0.25) && pur >= 0.12;
    }
    const onsets = meta.o ? popcount16(meta.o[b] ?? 0) : 1;
    const rmsOk = meta.hr ? (meta.hr[b] ?? 0) >= 0.12 : true;
    return pur >= 0.18 && onsets > 0 && rmsOk;
  }

  private vocalActivityMetrics(phraseId: string, meta: CellStemMeta): VocalActivityInfo {
    const known = this.vocalActivity?.get(phraseId);
    if (known) return known;
    const barFracs: [number, number, number, number] = [0, 0, 0, 0];
    let activeBars = 0;
    let sum = 0;
    for (let b = 0; b < 4; b++) {
      const active = this.isVocalBarActive(phraseId, b, meta);
      const density = active ? Math.min(1, popcount16(meta.o?.[b] ?? 0) / 10) : 0;
      barFracs[b] = active ? Math.max(0.25, density) : 0;
      if (active) activeBars++;
      sum += barFracs[b];
    }
    return { frac: sum / 4, barFracs, activeBars };
  }

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

  /** Commit only accepted audio; rejected proposals must not become context. */
  commitBoundary(stem: StemName, phraseId: string, srcBar: number, semitones: number, section: string): void {
    const meta = this.getPhraseMeta(phraseId, stem);
    if (!meta) return;
    this.lastBlockTail[stem] = { meta, srcBar, semitones,
      quality: meta.q[srcBar] ?? "mono", role: meta.er?.[srcBar] ?? "A", section };
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

  private inferVocal(net: VocalBarNet, x: readonly number[]): number[] {
    const hidden = new Array<number>(net.b1.length).fill(0);
    for (let j = 0; j < net.b1.length; j++) {
      let sum = net.b1[j];
      for (let i = 0; i < x.length; i++) {
        sum += net.w1[j][i] * (x[i] - net.mean[i]) / net.std[i];
      }
      hidden[j] = Math.tanh(sum);
    }
    return net.w2.map((row, j) => row.reduce((s, w, i) => s + w * hidden[i], net.b2[j]));
  }

  private vocalBarVec(meta: CellStemMeta, bar: number): number[] {
    const v = new Array<number>(30).fill(0);
    const mask = meta.o[bar] ?? 0;
    const count = popcount16(mask);
    v[0] = count >= 2 ? 1 : 0;
    v[1] = Math.min(1, count / 8);
    const root = meta.r[bar] ?? -1;
    v[2 + (root >= 0 ? root % 12 : 11)] = 1;
    for (let i = 0; i < 16; i++) v[14 + i] = (mask >> i) & 1;
    return v;
  }

  /**
   * Build a vocal bar schedule that develops instead of repeating one cell.
   * The distinct-bar target comes from real songs: a 4-bar vocal phrase with
   * four active bars uses about two different melodic bars, not one.
   */
  private developVocalBars<T extends { itemIndex: number; srcBar: number; meta?: CellStemMeta }>(
    pool: ReadonlyArray<T>,
    outputBars: number,
    rotation: number,
  ): T[] {
    if (pool.length === 0) return [];
    let attention = 1;
    let prediction: number[] | null = null;
    const model = this.vocalBarModel;
    if (model && pool[0].meta) {
      const meta = pool[0].meta;
      const here = pool[0].srcBar;
      const at = (offset: number): number[] => this.vocalBarVec(meta, Math.max(0, Math.min(3, here + offset)));
      // Training layout: [bar-2, bar, bar-1, bar].
      const input = [...at(-2), ...at(0), ...at(-1), ...at(0)];
      prediction = this.inferVocal(model.next, input);
      const runBars = Math.max(2, Math.min(16, Math.round((this.inferVocal(model.length, input)[0] ?? 0.25) * 16)));
      attention = Math.max(2, Math.round(runBars / 4));
    }
    // Real vocal passages are long, but a 16-bar hold still has to *develop*.
    // The mined "distinct chroma bars" prior is perceptually too conservative
    // (sustained notes collapse to one chroma), so use every active bar we
    // have and vary the 4-bar motif with the block rotation.  A single-bar
    // motif is never repeated verbatim across a long hold.
    const chosen = pool.slice(0, Math.max(1, Math.min(pool.length, 4)));
    if (prediction && chosen.some((item) => item.meta)) {
      const similarity = (item: T): number => {
        if (!item.meta) return 0;
        const vec = this.vocalBarVec(item.meta, item.srcBar);
        let score = 0;
        for (let i = 0; i < 12; i++) score += vec[2 + i] * prediction[i];
        for (let i = 0; i < 16; i++) score += vec[14 + i] * prediction[12 + i] * 0.5;
        return score;
      };
      chosen.sort((a, b) => similarity(b) - similarity(a));
    }
    const motifs = [
      [0, 1, 2, 3],
      [0, 1, 0, 2],
      [0, 2, 1, 3],
      [0, 0, 1, 2],
      [0, 1, 3, 2],
    ];
    const out: T[] = [];
    let last = -1;
    for (let b = 0; b < outputBars; b++) {
      const cycle = Math.floor(b / (4 * attention));
      const motif = motifs[(rotation + cycle) % motifs.length];
      let index = motif[b % 4] % chosen.length;
      if (index === last && chosen.length > 1) index = (index + 1) % chosen.length;
      out.push(chosen[index]);
      last = index;
    }
    return out;
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

    const fallbackMeta = this.getPhraseMeta(fallback.phrase.id, "vocals");
    const fallbackAct = fallbackMeta
      ? this.vocalActivityMetrics(fallback.phrase.id, fallbackMeta)
      : (this.vocalActivity?.get(fallback.phrase.id) ?? { frac: 0.5, barFracs: [0.5, 0.5, 0.5, 0.5] as const, activeBars: 4 });
    const fallbackIsSparse = fallbackAct.activeBars < 2 || fallbackAct.frac < 0.22;

    const rateRef = fallback.rate;
    const scored: Array<{ entry: TEntry; score: number; strategy: ComposedStemPlan<TEntry>["vocalStrategy"] }> = [];
    const maxCandidates = 240;
    const stride = Math.max(1, Math.floor(pool.length / maxCandidates));
    const seenIds = new Set<string>();
    const sampleIndices: number[] = [];
    for (let i = 0; i < pool.length && sampleIndices.length < maxCandidates; i += stride) {
      sampleIndices.push(i);
    }
    const fbIdx = pool.findIndex((e) => e.phrase.id === fallback.phrase.id);
    if (fbIdx >= 0 && !sampleIndices.includes(fbIdx)) sampleIndices.push(fbIdx);

    for (const i of sampleIndices) {
      const e = pool[i];
      if (seenIds.has(e.phrase.id)) continue;
      seenIds.add(e.phrase.id);
      if (Math.abs(e.rate - rateRef) > 0.04) continue;
      const meta = this.getPhraseMeta(e.phrase.id, "vocals");
      if (!meta) continue;
      const avgP = (meta.p[0] + meta.p[1] + meta.p[2] + meta.p[3]) / 4;
      if (avgP < 0.16) continue;
      const act = this.vocalActivityMetrics(e.phrase.id, meta);
      if (act.activeBars === 0 || act.frac < 0.18) continue;

      const essMax = meta.es ? Math.max(...meta.es) : avgP;
      const singingDensity = Math.min(1, act.frac / 0.60) * 0.65 + (act.activeBars / 4) * 0.55;
      let s = avgP * 0.40 + essMax * 0.25 + singingDensity + (meta.fid ?? 0.65) * 0.10;
      if (act.activeBars < 2) s -= 0.6;
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
      if (e.phrase.id === fallback.phrase.id) {
        s += fallbackIsSparse ? -0.4 : 0.20;
      }
      scored.push({ entry: e, score: s, strategy: strat });
    }

    if (scored.length === 0) return { entry: fallback, strategy: "section_role" };
    scored.sort((a, b) => b.score - a.score);
    return { entry: scored[0].entry, strategy: scored[0].strategy };
  }

  private defaultMeta(stem: StemName, sessionRoot: number): CellStemMeta {
    return {
      s: "AAAB",
      p: [0.5, 0.5, 0.5, 0.5],
      r: [sessionRoot, sessionRoot, sessionRoot, sessionRoot],
      q: stem === "drums" ? ["perc", "perc", "perc", "perc"] : ["mono", "mono", "mono", "mono"],
      o: [0, 0, 0, 0],
    };
  }

  private solvePhraseKeyShift<TEntry extends ComposingPoolEntry>(
    stem: StemName,
    primary: TEntry,
    meta: CellStemMeta,
    chord: ProgressionChord,
    sessionRoot: number,
  ): number {
    if (stem === "drums") return 0;
    let bestBar = 0;
    let bestW = -1;
    for (let b = 0; b < 4; b++) {
      const root = meta.r[b] ?? -1;
      const p = meta.p[b] ?? 0;
      const w = root >= 0 ? p + 0.1 : p * 0.25;
      if (w > bestW) {
        bestW = w;
        bestBar = b;
      }
    }
    const repRoot = meta.r[bestBar] ?? -1;
    if (repRoot < 0) return primary.keyShift;
    const repQual = meta.q[bestBar] ?? "mono";
    return solveCellPitchShift(
      stem,
      repRoot,
      repQual,
      sessionRoot,
      chord.relRoot,
      chord.quality,
      6,
      true,
    ).semitones;
  }

  private evaluateSeamGate(args: {
    stem: StemName;
    prevMeta: CellStemMeta;
    prevBar: number;
    prevSemitones: number;
    nextMeta: CellStemMeta;
    nextBar: number;
    nextSemitones: number;
    targetQuality: "minor" | "major" | "sus";
    barPos: number;
    isContiguousSamePhrase: boolean;
    isSectionBoundary: boolean;
    compatScorer?: CellCompatScorer;
  }): number {
    const acoustic = scoreBarSeamContinuity({
      stem: args.stem,
      prevMeta: args.prevMeta,
      prevBar: args.prevBar,
      nextMeta: args.nextMeta,
      nextBar: args.nextBar,
      isContiguousSamePhrase: args.isContiguousSamePhrase,
      isSectionBoundary: args.isSectionBoundary,
    });
    if (args.compatScorer?.hasArrangementHead()) {
      const neural = args.compatScorer.scoreTransition({
        prev: {
          stem: args.stem,
          meta: args.prevMeta,
          barIdx: args.prevBar,
          semitonesShift: args.prevSemitones,
          targetQuality: args.targetQuality,
        },
        next: {
          stem: args.stem,
          meta: args.nextMeta,
          barIdx: args.nextBar,
          semitonesShift: args.nextSemitones,
          targetQuality: args.targetQuality,
        },
        barPos: args.barPos,
        isSectionBoundary: args.isSectionBoundary,
        isContiguousSamePhrase: args.isContiguousSamePhrase,
      });
      return 0.55 * acoustic + 0.45 * neural;
    }
    return acoustic;
  }

  private buildContiguous4BarCells<TEntry extends ComposingPoolEntry>(
    primary: TEntry,
    meta: CellStemMeta,
    stem: StemName,
    blockBars: number,
    phraseKeyShift: number,
  ): SelectedCell<TEntry>[] {
    const out: SelectedCell<TEntry>[] = [];
    for (let b = 0; b < blockBars; b++) {
      const m = b % 4;
      const qual = meta.q[m] ?? (stem === "drums" ? "perc" : "mono");
      const pur = meta.p[m] ?? 0.5;
      out.push({
        entry: primary,
        itemIndex: 0,
        srcBar: m,
        purity: pur,
        semitones: phraseKeyShift,
        onsetMask: meta.o[m] ?? 0,
        quality: qual,
        meta,
        essentialScore: meta.es?.[m] ?? pur,
        essentialRole: meta.er?.[m] ?? (stem === "drums" && meta.fill && m === 3 ? "T" : "A"),
      });
    }
    return out;
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
    const bpmRef = Math.max(60, primary.kit.bpm || 120);
    const pMeta = this.getPhraseMeta(primary.phrase.id, stem);
    let best: TEntry | null = null;
    let bestScore = -999;

    const maxScan = Math.min(pool.length, 96);
    const stride = Math.max(1, Math.floor(pool.length / maxScan));
    for (let idx = 0, inspected = 0; idx < pool.length && inspected < maxScan; idx += stride, inspected++) {
      const cand = pool[idx];
      if (cand.kit.track_id === primary.kit.track_id) continue;
      const bpmDelta = Math.abs((cand.kit.bpm || bpmRef) - bpmRef) / bpmRef;
      if (bpmDelta > 0.04) continue;
      if (Math.abs(cand.rate - rateRef) > 0.035) continue;
      const meta = this.getPhraseMeta(cand.phrase.id, stem);
      if (!meta) continue;
      const maxBarP = Math.max(meta.p[0], meta.p[1], meta.p[2], meta.p[3]);
      if (maxBarP < 0.25) continue;
      let vocalBonus = 0;
      if (stem === "vocals") {
        const act = this.vocalActivityMetrics(cand.phrase.id, meta);
        if (act.activeBars < 2 || act.frac < 0.20) continue;
        vocalBonus = Math.min(1, act.frac / 0.6) * 0.35;
      }

      const targetBar = stem === "drums" ? 3 : 0;
      const seamCont = pMeta
        ? scoreBarSeamContinuity({
            stem,
            prevMeta: pMeta,
            prevBar: 2,
            nextMeta: meta,
            nextBar: targetBar,
            isContiguousSamePhrase: false,
            isSectionBoundary: false,
          })
        : 0.7;
      if (seamCont < 0.62) continue;

      const maxBarEs = meta.es ? Math.max(...meta.es) : maxBarP;
      let s =
        maxBarP * 0.40 +
        maxBarEs * 0.25 +
        seamCont * 0.45 +
        vocalBonus +
        (meta.fid ?? 0.65) * 0.10 -
        bpmDelta * 5.0 -
        Math.abs(cand.rate - rateRef) * 2.5;
      if (bpmDelta <= 0.015) s += 0.25;
      if (!forbiddenTracks.has(cand.kit.track_id)) s += 0.30;
      if (cand.section.toUpperCase() === ctx.kitSection) s += 0.10;
      if (stem === "drums" && meta.fill) s += 0.28;
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
    commit?: boolean;
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
      const pMeta = this.getPhraseMeta(primary.phrase.id, stem) ?? this.defaultMeta(stem, sessionRoot);
      const sMeta = secondary ? (this.getPhraseMeta(secondary.phrase.id, stem) ?? null) : null;

      const ch0 = sectionCtx.progression[0] ?? { relRoot: 0, quality: "minor" as const };
      let phraseKeyShift = this.solvePhraseKeyShift(stem, primary, pMeta, ch0, sessionRoot);
      const selectedBars = this.buildContiguous4BarCells(primary, pMeta, stem, blockBars, phraseKeyShift);

      let schema: MotifSchema = pMeta.s ?? "AAAB";
      let hasSpliceOrGainEdit = false;

      // 2. Selective surgical splices on top of contiguous 4-bar flow (gated by acoustic + neural seam score)
      if (stem === "drums") {
        const isTurnaround = (blockIndex % 2 === 1) || sectionCtx.isPreDrop || sectionCtx.isBuildup;
        if (!pinnedStems.has("drums") && !isBridgeBlock && isTurnaround && !pMeta.fill && secondary && sMeta?.fill) {
          const bFill = blockBars - 1;
          const prevCell = selectedBars[bFill - 1];
          const nextCellIn = this.pickBestCellForBar(
            stem,
            [primary, secondary],
            1,
            sectionCtx.progression[3] ?? ch0,
            sessionRoot,
            null,
            null,
            true,
            prevCell
              ? {
                  meta: prevCell.meta,
                  srcBar: prevCell.srcBar,
                  semitones: 0,
                  quality: "perc",
                  role: prevCell.essentialRole,
                  section: sectionCtx.kitSection,
                }
              : undefined,
            3,
            false,
            "T",
            compatScorer,
          );
          if (prevCell) {
            const seamIn = this.evaluateSeamGate({
              stem: "drums",
              prevMeta: prevCell.meta,
              prevBar: prevCell.srcBar,
              prevSemitones: 0,
              nextMeta: sMeta,
              nextBar: 3,
              nextSemitones: 0,
              targetQuality: ch0.quality,
              barPos: 3,
              isContiguousSamePhrase: false,
              isSectionBoundary: false,
              compatScorer,
            });
            if (seamIn >= 0.68) {
              selectedBars[bFill] = { ...nextCellIn, itemIndex: 1, srcBar: 3, semitones: 0 };
              schema = "AAAB";
              hasSpliceOrGainEdit = true;
            }
          }
        }
        if (sectionCtx.isBuildup && !pinnedStems.has("drums") && !hasSpliceOrGainEdit && blockBars >= 4) {
          const seamTight = this.evaluateSeamGate({
            stem: "drums",
            prevMeta: pMeta,
            prevBar: 0,
            prevSemitones: 0,
            nextMeta: pMeta,
            nextBar: 0,
            nextSemitones: 0,
            targetQuality: ch0.quality,
            barPos: 1,
            isContiguousSamePhrase: false,
            isSectionBoundary: false,
            compatScorer,
          });
          if (seamTight >= 0.70 && !pMeta.fill) {
            selectedBars[1] = { ...selectedBars[0] };
            selectedBars[2] = { ...selectedBars[0] };
            schema = "AAAB";
            hasSpliceOrGainEdit = true;
          }
        }
      } else if (stem === "vocals") {
        // First repair dead/silent bars in the primary vocal phrase so the
        // vocal track never plays empty silence gaps between lines.
        const activeSrcBars: number[] = [];
        for (let bIdx = 0; bIdx < 4; bIdx++) {
          if (this.isVocalBarActive(primary.phrase.id, bIdx, pMeta)) activeSrcBars.push(bIdx);
        }
        if (activeSrcBars.length > 0 && activeSrcBars.length < 4 && !pinnedStems.has("vocals")) {
          // Real songs develop a vocal line across several different bars.
          // A phrase with one active bar borrows active bars from the secondary
          // donor instead of copying that bar into every output slot.
          const pool: Array<{ itemIndex: number; srcBar: number; meta: CellStemMeta; semitones: number }> =
            activeSrcBars.map((bar) => ({ itemIndex: 0, srcBar: bar, meta: pMeta, semitones: phraseKeyShift }));
          if (pool.length < 4 && secondary && sMeta) {
            const secActive = [0, 1, 2, 3].filter((bar) => this.isVocalBarActive(secondary.phrase.id, bar, sMeta));
            const secShift = this.solvePhraseKeyShift("vocals", secondary, sMeta, ch0, sessionRoot);
            for (const bar of secActive) {
              pool.push({ itemIndex: 1, srcBar: bar, meta: sMeta, semitones: secShift });
            }
          }
          const schedule = this.developVocalBars(pool, blockBars, blockIndex);
          for (let b = 0; b < blockBars; b++) {
            const pick = schedule[b];
            if (!pick) continue;
            selectedBars[b] = {
              ...selectedBars[b],
              entry: pick.itemIndex === 0 ? primary : (secondary ?? primary),
              itemIndex: pick.itemIndex,
              srcBar: pick.srcBar,
              purity: pick.meta.p[pick.srcBar] ?? selectedBars[b].purity,
              semitones: pick.semitones,
              onsetMask: pick.meta.o[pick.srcBar] ?? selectedBars[b].onsetMask,
              quality: pick.meta.q[pick.srcBar] ?? selectedBars[b].quality,
              essentialScore: pick.meta.es?.[pick.srcBar] ?? (pick.meta.p[pick.srcBar] ?? selectedBars[b].essentialScore),
              essentialRole: pick.meta.er?.[pick.srcBar] ?? selectedBars[b].essentialRole,
              meta: pick.meta,
            };
          }
          hasSpliceOrGainEdit = true;
        } else if (activeSrcBars.length === 0 && secondary && sMeta && !pinnedStems.has("vocals")) {
          const secActive: number[] = [];
          for (let bIdx = 0; bIdx < 4; bIdx++) {
            if (this.isVocalBarActive(secondary.phrase.id, bIdx, sMeta)) secActive.push(bIdx);
          }
          if (secActive.length > 0) {
            const secShift = this.solvePhraseKeyShift("vocals", secondary, sMeta, ch0, sessionRoot);
            for (let b = 0; b < blockBars; b++) {
              const sb = secActive[b % secActive.length];
              selectedBars[b] = {
                entry: secondary,
                itemIndex: 1,
                srcBar: sb,
                purity: sMeta.p[sb] ?? 0.5,
                semitones: secShift,
                onsetMask: sMeta.o[sb] ?? 0,
                quality: sMeta.q[sb] ?? "mono",
                meta: sMeta,
                essentialScore: sMeta.es?.[sb] ?? (sMeta.p[sb] ?? 0.5),
                essentialRole: sMeta.er?.[sb] ?? "A",
              };
            }
            phraseKeyShift = secShift;
            hasSpliceOrGainEdit = true;
          }
        }

        const isChorus = ctxIsChorus(sectionCtx) || vocalStrategy === "hook_loop";
        if (isChorus && pMeta.hook && pMeta.hb === 2 && blockBars >= 4
          && this.isVocalBarActive(primary.phrase.id, 0, pMeta)
          && this.isVocalBarActive(primary.phrase.id, 1, pMeta)) {
          const loopSeam = this.evaluateSeamGate({
            stem: "vocals",
            prevMeta: pMeta,
            prevBar: 1,
            prevSemitones: phraseKeyShift,
            nextMeta: pMeta,
            nextBar: 0,
            nextSemitones: phraseKeyShift,
            targetQuality: ch0.quality,
            barPos: 2,
            isContiguousSamePhrase: false,
            isSectionBoundary: false,
            compatScorer,
          });
          if (loopSeam >= 0.68) {
            const hookPool = [0, 1]
              .filter((bar) => this.isVocalBarActive(primary.phrase.id, bar, pMeta))
              .map((bar) => ({ itemIndex: 0, srcBar: bar, meta: pMeta }));
            const hookBars = this.developVocalBars(hookPool, blockBars, blockIndex);
            const source = selectedBars.slice();
            for (let b = 0; b < blockBars; b++) {
              const bar = hookBars[b]?.srcBar ?? b % 2;
              selectedBars[b] = { ...source[bar], srcBar: bar };
            }
            schema = "ABAB";
            hasSpliceOrGainEdit = true;
          }
        } else if (sectionCtx.isBuildup && !pinnedStems.has("vocals")) {
          const cb = Math.max(0, Math.min(3, pMeta.cb ?? 0));
          if ((pMeta.p[cb] ?? 0) >= 0.30 && this.isVocalBarActive(primary.phrase.id, cb, pMeta)) {
            hasSpliceOrGainEdit = true;
          }
        }
      } else {
        // bass / other: contiguous 0->1->2->3 by default; repair dead dropout bars or seam-clean Buildups
        let bestActive = -1;
        let bestP = 0;
        for (let i = 0; i < 4; i++) {
          if ((pMeta.p[i] ?? 0) > bestP) {
            bestP = pMeta.p[i] ?? 0;
            bestActive = i;
          }
        }
        if (bestActive >= 0 && bestP >= 0.35) {
          for (let b = 0; b < blockBars; b++) {
            const m = b % 4;
            if ((pMeta.p[m] ?? 0.5) < 0.14) {
              const prevCell = b > 0 ? selectedBars[b - 1] : undefined;
              const seamFix = prevCell
                ? this.evaluateSeamGate({
                    stem,
                    prevMeta: prevCell.meta,
                    prevBar: prevCell.srcBar,
                    prevSemitones: phraseKeyShift,
                    nextMeta: pMeta,
                    nextBar: bestActive,
                    nextSemitones: phraseKeyShift,
                    targetQuality: ch0.quality,
                    barPos: m,
                    isContiguousSamePhrase: bestActive === prevCell.srcBar + 1,
                    isSectionBoundary: false,
                    compatScorer,
                  })
                : 0.72;
              if (seamFix >= 0.68) {
                selectedBars[b] = {
                  ...selectedBars[b],
                  srcBar: bestActive,
                  purity: pMeta.p[bestActive] ?? 0.5,
                  onsetMask: pMeta.o[bestActive] ?? 0,
                  quality: pMeta.q[bestActive] ?? "mono",
                  essentialScore: pMeta.es?.[bestActive] ?? (pMeta.p[bestActive] ?? 0.5),
                };
                hasSpliceOrGainEdit = true;
              }
            }
          }
        }
        if (sectionCtx.isBuildup && !pinnedStems.has(stem) && !hasSpliceOrGainEdit && blockBars >= 4) {
          const loopSeam = this.evaluateSeamGate({
            stem,
            prevMeta: pMeta,
            prevBar: 1,
            prevSemitones: phraseKeyShift,
            nextMeta: pMeta,
            nextBar: 0,
            nextSemitones: phraseKeyShift,
            targetQuality: ch0.quality,
            barPos: 2,
            isContiguousSamePhrase: false,
            isSectionBoundary: false,
            compatScorer,
          });
          if (loopSeam >= 0.70) {
            for (let b = 2; b < blockBars; b++) {
              const srcB = b % 2;
              selectedBars[b] = { ...selectedBars[srcB], srcBar: srcB };
            }
            schema = "AABB";
            hasSpliceOrGainEdit = true;
          }
        }
      }

      // Drop secondary donor when unused so audio cache never fetches an unused file
      const usedSecondary = Boolean(secondary && selectedBars.some((c) => c.itemIndex === 1));
      const donors: TEntry[] = usedSecondary && secondary ? [primary, secondary] : [primary];
      if (usedSecondary && secondary) activeTracks.add(secondary.kit.track_id);

      const slices: SliceSpec[] = [];
      const pitchPoints: PitchAutomationPoint[] = [];
      let baseSemitones = phraseKeyShift;

      // Single key shift per 4-bar phrase; allow per-bar pitchPoints ONLY when repeating a stationary 1-bar mono cell
      const firstCell = selectedBars[0];
      const allSameCell = Boolean(
        firstCell &&
        selectedBars.every((c) => c.itemIndex === firstCell.itemIndex && c.srcBar === firstCell.srcBar),
      );
      const allowPerBarPitch =
        allSameCell &&
        stem !== "drums" &&
        stem !== "vocals" &&
        firstCell.quality === "mono" &&
        firstCell.purity >= 0.72 &&
        (firstCell.meta.r[firstCell.srcBar] ?? -1) >= 0;

      if (allowPerBarPitch) {
        let lastSentShift = baseSemitones;
        for (let b = 0; b < blockBars; b++) {
          const chord = sectionCtx.progression[b % 4] ?? ch0;
          const solved = solveCellPitchShift(
            stem,
            firstCell.meta.r[firstCell.srcBar] ?? -1,
            firstCell.quality,
            sessionRoot,
            chord.relRoot,
            chord.quality,
            4,
            true,
          );
          if (b === 0) {
            baseSemitones = solved.semitones;
            lastSentShift = solved.semitones;
            selectedBars[b] = { ...selectedBars[b], semitones: solved.semitones };
          } else if (Math.abs(solved.semitones - baseSemitones) <= 4) {
            selectedBars[b] = { ...selectedBars[b], semitones: solved.semitones };
            if (solved.semitones !== lastSentShift) {
              pitchPoints.push({ offsetSec: b * barOutSec, semitones: solved.semitones });
              lastSentShift = solved.semitones;
            }
          }
        }
      }

      for (let b = 0; b < blockBars; b++) {
        const cell = selectedBars[b];
        if (stem === "drums") barDrumMasks.push(cell.onsetMask);
        if (stem === "vocals") barVocalMasks.push(cell.onsetMask);

        const isLastBar = b === blockBars - 1;
        const isFirstBar = b === 0;
        const barStartFrac = cell.srcBar * 0.25;
        const barEndFrac = (cell.srcBar + 1) * 0.25;

        if (stem === "vocals" && sectionCtx.isBuildup && !pinnedStems.has("vocals") && b >= blockBars - 2 && (cell.meta.p[cell.meta.cb ?? cell.srcBar] ?? 0) >= 0.30) {
          const cb = Math.max(0, Math.min(3, cell.meta.cb ?? cell.srcBar));
          const cStart = cb * 0.25;
          hasSpliceOrGainEdit = true;
          if (!isLastBar) {
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.125, targetBars: 0.5, gain: 0.92 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: cStart, endFrac: cStart + 0.125, targetBars: 0.5, gain: 1.0 });
          } else if (sectionCtx.isPreDrop) {
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

        if (sectionCtx.isPreDrop && isLastBar && !pinnedStems.has(stem)) {
          hasSpliceOrGainEdit = true;
          if (stem === "drums") {
            const fStart = cell.meta.fill ? 0.75 : barStartFrac;
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.0625, targetBars: 0.25, gain: 0.92 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.0625, targetBars: 0.25, gain: 0.98 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.03125, targetBars: 0.125, gain: 1.02 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.03125, targetBars: 0.125, gain: 1.05 });
            slices.push({ itemIndex: cell.itemIndex, startFrac: fStart, endFrac: fStart + 0.03125, targetBars: 0.125, gain: 1.08 });
            slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.03125, targetBars: 0.125, gain: 0, endGain: 0 });
          } else {
            slices.push({ itemIndex: cell.itemIndex, startFrac: barStartFrac, endFrac: barStartFrac + 0.21875, targetBars: 0.875, gain: 1.0, endGain: 0.85 });
            slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.03125, targetBars: 0.125, gain: 0, endGain: 0 });
          }
          continue;
        }

        if (sectionCtx.isBreakdownStart && isFirstBar && (stem === "drums" || stem === "bass") && !pinnedStems.has(stem)) {
          hasSpliceOrGainEdit = true;
          slices.push({ itemIndex: 0, startFrac: 0, endFrac: 0.125, targetBars: 0.5, gain: 0, endGain: 0 });
          slices.push({ itemIndex: cell.itemIndex, startFrac: barStartFrac + 0.125, endFrac: barEndFrac, targetBars: 0.5, gain: 0.35, endGain: 0.9 });
          continue;
        }

        const downbeatGain =
          sectionCtx.isDropDownbeat && isFirstBar && (stem === "drums" || stem === "other")
            ? 1.06
            : 1.0;
        if (Math.abs(downbeatGain - 1) > 0.01) hasSpliceOrGainEdit = true;

        slices.push({
          itemIndex: cell.itemIndex,
          startFrac: barStartFrac,
          endFrac: barEndFrac,
          targetBars: 1,
          gain: downbeatGain,
        });
      }

      const isContiguousBars = selectedBars.every((c, idx) => c.itemIndex === 0 && c.srcBar === (idx % 4));
      const isContiguousUncut =
        donors.length === 1 &&
        !hasSpliceOrGainEdit &&
        isContiguousBars &&
        (blockBars !== 4 || isContiguousFullPhrase(slices));

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
      if (tailCell && args.commit !== false) {
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
        isContiguousUncut,
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
