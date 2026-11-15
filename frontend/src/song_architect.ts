/**
 * Song Architect: manages Song Mode ("song") vs Infinite DJ Morph Mode ("morph"),
 * maps the 104-bar set onto structured pop/electronic sections, stores recurring
 * thematic memory (Chorus 1 -> Chorus 2, Verse 1 -> Verse 2), and emits
 * section transition flags (pre-drop buildup/gap, drop impact, breakdown dropout).
 */

import { ProgressionChord, StemName, getProgression } from "./theory_engine";

export type SongMode = "song" | "morph";

export type SongSectionRole =
  | "INTRO"
  | "VERSE_1"
  | "BUILDUP_1"
  | "CHORUS_1"
  | "BREAKDOWN"
  | "VERSE_2"
  | "BUILDUP_2"
  | "CHORUS_2"
  | "OUTRO_BRIDGE";

export interface SectionPlanContext {
  mode: SongMode;
  role: SongSectionRole;
  /** Mapped kit manifest section filter hint */
  kitSection: "INTRO" | "GROOVE_A" | "BUILDUP" | "DROP" | "BREAKDOWN";
  /** Target energy 0..1 */
  targetEnergy: number;
  /** Progression index within style bank (A=0 for verse/intro, B=1 for chorus/drop) */
  progressionIndex: number;
  /** Progression chords for this 4-bar cycle */
  progression: ProgressionChord[];
  /** Apply drum snare tightening + buildup vocal tightening + pre-drop silence gap at tail */
  isBuildup: boolean;
  isPreDrop: boolean;
  /** Apply full drop entry punch at bar 0 of this block */
  isDropDownbeat: boolean;
  /** Apply breakdown bass/kick drop-out at bar 0 of this block */
  isBreakdownStart: boolean;
  /** Remembered vocal phrase ID for Chorus or Verse recall (null if fresh) */
  recalledVocalPhraseId: string | null;
  /** Remembered motif phrases by stem for Chorus/Verse identity recall */
  recalledDonorPhraseIds: Partial<Record<StemName, string>>;
  /** Bar range within the 104-bar set */
  sectionStartBar: number;
  sectionEndBar: number;
}

export interface SongThemeMemory {
  style: string;
  sessionRoot: number;
  sessionMode: "minor" | "major";
  verseProgressionIdx: number;
  chorusProgressionIdx: number;
  verseDonors: Partial<Record<StemName, string>>;
  chorusDonors: Partial<Record<StemName, string>>;
  chorusVocalHookId: string | null;
  verseVocalId: string | null;
}

interface SongSegmentDef {
  startBar: number;
  endBar: number;
  role: SongSectionRole;
  kitSection: "INTRO" | "GROOVE_A" | "BUILDUP" | "DROP" | "BREAKDOWN";
  energy: number;
  progSlot: "verse" | "chorus";
}

// Aligned cleanly with 4->8->16->8 block boundaries (104 bars total):
// 4-bar blocks: 0..48 (boundaries 0, 8, 24, 32, 48)
// 8-bar blocks: 48..80 (boundaries 48, 56, 72, 80)
// 16-bar block: 80..96
// 8-bar bridge: 96..104
const SONG_BLUEPRINT: readonly SongSegmentDef[] = [
  { startBar: 0, endBar: 8, role: "INTRO", kitSection: "INTRO", energy: 0.34, progSlot: "verse" },
  { startBar: 8, endBar: 24, role: "VERSE_1", kitSection: "GROOVE_A", energy: 0.68, progSlot: "verse" },
  { startBar: 24, endBar: 32, role: "BUILDUP_1", kitSection: "BUILDUP", energy: 0.84, progSlot: "verse" },
  { startBar: 32, endBar: 48, role: "CHORUS_1", kitSection: "DROP", energy: 0.94, progSlot: "chorus" },
  { startBar: 48, endBar: 56, role: "BREAKDOWN", kitSection: "BREAKDOWN", energy: 0.44, progSlot: "verse" },
  { startBar: 56, endBar: 72, role: "VERSE_2", kitSection: "GROOVE_A", energy: 0.74, progSlot: "verse" },
  { startBar: 72, endBar: 80, role: "BUILDUP_2", kitSection: "BUILDUP", energy: 0.88, progSlot: "verse" },
  { startBar: 80, endBar: 96, role: "CHORUS_2", kitSection: "DROP", energy: 0.96, progSlot: "chorus" },
  { startBar: 96, endBar: 104, role: "OUTRO_BRIDGE", kitSection: "BREAKDOWN", energy: 0.25, progSlot: "verse" },
];

export class SongArchitect {
  private mode: SongMode = "song";
  private theme: SongThemeMemory | null = null;
  private setSeed = 0;

  getMode(): SongMode {
    return this.mode;
  }

  setMode(mode: SongMode): void {
    this.mode = mode;
  }

  resetForNewSet(style: string, sessionRoot: number, sessionMode: "minor" | "major"): void {
    this.setSeed = (this.setSeed + 1) % 10000;
    this.theme = {
      style,
      sessionRoot,
      sessionMode,
      verseProgressionIdx: this.setSeed,
      chorusProgressionIdx: this.setSeed + 1,
      verseDonors: {},
      chorusDonors: {},
      chorusVocalHookId: null,
      verseVocalId: null,
    };
  }

  private ensureTheme(style: string, sessionRoot: number, sessionMode: "minor" | "major"): SongThemeMemory {
    if (!this.theme || this.theme.style !== style) {
      this.resetForNewSet(style, sessionRoot, sessionMode);
    }
    return this.theme!;
  }

  describeBlock(
    barInSet: number,
    blockBars: number,
    style: string,
    sessionRoot: number,
    sessionMode: "minor" | "major",
    morphTargetEnergy: number,
    morphSectionLabel: string,
    customProgBank?: Parameters<typeof getProgression>[4],
  ): SectionPlanContext {
    const theme = this.ensureTheme(style, sessionRoot, sessionMode);
    const bar = ((barInSet % 104) + 104) % 104;

    if (this.mode === "morph") {
      const upper = (morphSectionLabel || "GROOVE_A").toUpperCase();
      const kitSection: SectionPlanContext["kitSection"] =
        upper === "INTRO"
          ? "INTRO"
          : upper === "BUILDUP"
            ? "BUILDUP"
            : upper === "DROP" || upper === "PEAK"
              ? "DROP"
              : upper === "BREAKDOWN" || upper === "BRIDGE"
                ? "BREAKDOWN"
                : "GROOVE_A";
      const progIdx = kitSection === "DROP" ? theme.chorusProgressionIdx : theme.verseProgressionIdx;
      const role: SongSectionRole =
        kitSection === "INTRO"
          ? "INTRO"
          : kitSection === "BUILDUP"
            ? "BUILDUP_1"
            : kitSection === "DROP"
              ? "CHORUS_1"
              : kitSection === "BREAKDOWN"
                ? "BREAKDOWN"
                : "VERSE_1";
      return {
        mode: "morph",
        role,
        kitSection,
        targetEnergy: morphTargetEnergy,
        progressionIndex: progIdx,
        progression: getProgression(style, sessionMode, 4, progIdx, customProgBank),
        isBuildup: kitSection === "BUILDUP",
        isPreDrop: kitSection === "BUILDUP" && bar >= 56,
        isDropDownbeat: kitSection === "DROP" && (bar === 64 || bar === 32),
        isBreakdownStart: kitSection === "BREAKDOWN" && bar >= 96,
        recalledVocalPhraseId: null,
        recalledDonorPhraseIds: {},
        sectionStartBar: bar,
        sectionEndBar: bar + blockBars,
      };
    }

    const seg =
      SONG_BLUEPRINT.find((s) => bar >= s.startBar && bar < s.endBar) ??
      SONG_BLUEPRINT[SONG_BLUEPRINT.length - 1];
    const blockEnd = bar + blockBars;
    const isBuildup = seg.role === "BUILDUP_1" || seg.role === "BUILDUP_2";
    const isPreDrop = isBuildup && blockEnd >= seg.endBar;
    const isDropDownbeat =
      (seg.role === "CHORUS_1" || seg.role === "CHORUS_2") && bar === seg.startBar;
    const isBreakdownStart = seg.role === "BREAKDOWN" && bar === seg.startBar;

    const progIdx =
      seg.progSlot === "chorus" ? theme.chorusProgressionIdx : theme.verseProgressionIdx;

    let recalledVocal: string | null = null;
    let recalledDonors: Partial<Record<StemName, string>> = {};
    if (seg.role === "CHORUS_2") {
      recalledVocal = theme.chorusVocalHookId;
      recalledDonors = { ...theme.chorusDonors };
    } else if (seg.role === "CHORUS_1" && bar > seg.startBar) {
      // Repeat the exact same Chorus 1 vocal hook across Chorus 1 sub-blocks
      recalledVocal = theme.chorusVocalHookId;
    } else if (seg.role === "VERSE_2") {
      recalledVocal = theme.verseVocalId;
      recalledDonors = { ...theme.verseDonors };
    }

    return {
      mode: "song",
      role: seg.role,
      kitSection: seg.kitSection,
      targetEnergy: seg.energy,
      progressionIndex: progIdx,
      progression: getProgression(style, sessionMode, 4, progIdx, customProgBank),
      isBuildup,
      isPreDrop,
      isDropDownbeat,
      isBreakdownStart,
      recalledVocalPhraseId: recalledVocal,
      recalledDonorPhraseIds: recalledDonors,
      sectionStartBar: seg.startBar,
      sectionEndBar: seg.endBar,
    };
  }

  recordSectionTheme(
    ctx: SectionPlanContext,
    chosenPhraseIds: Partial<Record<StemName, string>>,
  ): void {
    if (!this.theme || ctx.mode !== "song") return;
    if (ctx.role === "CHORUS_1") {
      if (!this.theme.chorusVocalHookId && chosenPhraseIds.vocals) {
        this.theme.chorusVocalHookId = chosenPhraseIds.vocals;
      }
      for (const stem of ["drums", "bass", "other", "vocals"] as const) {
        if (!this.theme.chorusDonors[stem] && chosenPhraseIds[stem]) {
          this.theme.chorusDonors[stem] = chosenPhraseIds[stem];
        }
      }
    } else if (ctx.role === "VERSE_1") {
      if (!this.theme.verseVocalId && chosenPhraseIds.vocals) {
        this.theme.verseVocalId = chosenPhraseIds.vocals;
      }
      for (const stem of ["drums", "bass", "other", "vocals"] as const) {
        if (!this.theme.verseDonors[stem] && chosenPhraseIds[stem]) {
          this.theme.verseDonors[stem] = chosenPhraseIds[stem];
        }
      }
    }
  }
}
