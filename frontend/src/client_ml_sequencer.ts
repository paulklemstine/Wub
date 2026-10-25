// Client-Side ML Pattern Sequencer and Autonomous Groovebox Engine
//
// Enables 100% standalone, in-browser operation without requiring a live WebSocket server.
// Rebuilds dance patterns learned from the training data with:
// 1. Authentic harmonic chord progressions across 8-bar musical phrase arcs
// 2. Multi-section arrangement intelligence (INTRO, BUILDUP with snare rush & drop gap, DROP, GROOVE, BREAKDOWN)
// 3. Dynamic sidechain ducking (dipping bass by -9dB on kick transients)
// 4. Distinct drum voices (kick, snare, open/closed hats with choke, syncopated percs)
// 5. Dual-engine playback: pitch-matched authentic audio slices + Web Audio synthesizer

import type { StemName } from "./types";
import { AudioEngine } from "./audio_engine";

export interface SerializedPattern {
  id: string;
  name: string;
  groove_type: string;
  energy: number;
  density: number;
  onset_grid: number[];
  velocities?: number[];
  length: number;
  key?: string;
  audio_similarity_score?: number;
}

export interface SerializedPhrase {
  id: string;
  section: string;
  bpm: number;
  key: string;
  energy: number;
  bars: Array<{
    bar_num: number;
    phrase_bar: number;
    section: string;
    onset_grids: Record<string, number[]>;
    groove_types: Record<string, string>;
    energy: number;
    key: string;
  }>;
}

export interface SliceManifestItem {
  name: string;
  url: string;
  b64?: string | null;
  duration: number;
  centroid: number;
  pitch?: number;
  loudness?: number;
  track_id: string;
  segment_id: string;
}

export interface SliceAsset {
  buffer: AudioBuffer;
  pitch: number;
  centroid: number;
  duration: number;
  loudness: number;
  name: string;
}

export interface LoopManifestItem {
  id: string;
  name: string;
  stem: StemName;
  category: string;
  url: string;
  source_bpm: number;
  key: string;
  dominant_pitch: number;
  duration: number;
  onset_grid: number[];
  track_id: string;
  track_title: string;
  suggested_sections: string[];
}

export interface LoadedLoop {
  meta: LoopManifestItem;
  buffer: AudioBuffer;
}

export interface CohesivePhraseManifestItem {
  id: string;
  track_id: string;
  track_title: string;
  key: string;
  source_bpm: number;
  section: "INTRO" | "GROOVE_A" | "BUILDUP" | "DROP" | "BREAKDOWN";
  energy: number;
  bar_length: number;
  duration: number;
  stems: Record<StemName, string>;
  onset_grids: Record<StemName, number[]>;
}

export interface LoadedPhraseBlock {
  meta: CohesivePhraseManifestItem;
  buffers: Partial<Record<StemName, AudioBuffer>>;
}

export interface StandaloneManifest {
  version: number;
  default_bpm: number;
  chord_progressions?: Array<{
    track_id: string;
    key: string;
    roots: number[];
  }>;
  patterns: Record<StemName, SerializedPattern[]>;
  phrases: SerializedPhrase[];
  slices: Record<StemName, Record<string, SliceManifestItem[]>>;
  loops?: Record<StemName, LoopManifestItem[]>;
  phrase_blocks?: CohesivePhraseManifestItem[];
}

export interface ComposingMLWeights {
  version: number;
  phrase_transitions: Record<string, Record<string, number>>;
  section_transitions: Record<string, Record<string, number>>;
  section_energy_targets: Record<string, number>;
}

// Semitone pitch constants: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
export const ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// -----------------------------------------------------------------------------
// 4 Map-Territory Quadrants & Sound Territories
// -----------------------------------------------------------------------------

export type StyleTerritory = "wave" | "cyber" | "house" | "breaks";

export function getStyleTerritory(x: number, y: number): StyleTerritory {
  if (x < 0.5 && y < 0.5) return "wave";
  if (x >= 0.5 && y < 0.5) return "cyber";
  if (x < 0.5 && y >= 0.5) return "house";
  return "breaks";
}

export const TERRITORY_INFO: Record<
  StyleTerritory,
  { name: string; short: string; color: string; glow: string; sub: string }
> = {
  wave: {
    name: "LIQUID WAVE / ETHER",
    short: "WAVE",
    color: "#06b6d4",
    glow: "rgba(6, 182, 212, 0.18)",
    sub: "Ethereal Diva • 808 Sub • Dream Pads",
  },
  cyber: {
    name: "CYBER / PHONK",
    short: "CYBER",
    color: "#d946ef",
    glow: "rgba(217, 70, 239, 0.18)",
    sub: "Vocoder Chants • Distorted Punch • Reese",
  },
  house: {
    name: "DEEP HOUSE / VELVET",
    short: "HOUSE",
    color: "#f59e0b",
    glow: "rgba(245, 158, 11, 0.18)",
    sub: "Soulful Hook • 909 Punch • Rolling Offbeat",
  },
  breaks: {
    name: "ELECTRO BREAKS / GRIT",
    short: "BREAKS",
    color: "#84cc16",
    glow: "rgba(132, 204, 22, 0.18)",
    sub: "Hype Chops • 2-Step Breaks • Funk Bass",
  },
};

export interface ChordStep {
  rootPitch: number; // 0..11
  isMinor: boolean;
  bassHz: number;
  chordHz: number;
  name: string;
}

// Camelot Harmonic Key Wheel Definitions
export interface CamelotKeyInfo {
  code: string;       // e.g. "8A", "8B"
  name: string;       // e.g. "Am", "C"
  rootPitch: number;  // 0..11
  isMinor: boolean;
}

export const CAMELOT_KEYS: Record<string, CamelotKeyInfo> = {
  "1A": { code: "1A", name: "G#m", rootPitch: 8, isMinor: true },
  "2A": { code: "2A", name: "D#m", rootPitch: 3, isMinor: true },
  "3A": { code: "3A", name: "A#m", rootPitch: 10, isMinor: true },
  "4A": { code: "4A", name: "Fm",  rootPitch: 5, isMinor: true },
  "5A": { code: "5A", name: "Cm",  rootPitch: 0, isMinor: true },
  "6A": { code: "6A", name: "Gm",  rootPitch: 7, isMinor: true },
  "7A": { code: "7A", name: "Dm",  rootPitch: 2, isMinor: true },
  "8A": { code: "8A", name: "Am",  rootPitch: 9, isMinor: true },
  "9A": { code: "9A", name: "Em",  rootPitch: 4, isMinor: true },
  "10A": { code: "10A", name: "Bm", rootPitch: 11, isMinor: true },
  "11A": { code: "11A", name: "F#m", rootPitch: 6, isMinor: true },
  "12A": { code: "12A", name: "C#m", rootPitch: 1, isMinor: true },

  "1B": { code: "1B", name: "B",   rootPitch: 11, isMinor: false },
  "2B": { code: "2B", name: "F#",  rootPitch: 6, isMinor: false },
  "3B": { code: "3B", name: "Db",  rootPitch: 1, isMinor: false },
  "4B": { code: "4B", name: "Ab",  rootPitch: 8, isMinor: false },
  "5B": { code: "5B", name: "Eb",  rootPitch: 3, isMinor: false },
  "6B": { code: "6B", name: "Bb",  rootPitch: 10, isMinor: false },
  "7B": { code: "7B", name: "F",   rootPitch: 5, isMinor: false },
  "8B": { code: "8B", name: "C",   rootPitch: 0, isMinor: false },
  "9B": { code: "9B", name: "G",   rootPitch: 7, isMinor: false },
  "10B": { code: "10B", name: "D",  rootPitch: 2, isMinor: false },
  "11B": { code: "11B", name: "A",  rootPitch: 9, isMinor: false },
  "12B": { code: "12B", name: "E",  rootPitch: 4, isMinor: false },
};

export interface HarmonicTemplateStep {
  semitoneOffset: number; // offset from key root
  isMinor: boolean;
}

export const HARMONIC_TEMPLATES: HarmonicTemplateStep[][] = [
  // 1. Epic Anthem: i - VI - III - VII (Am - F - C - G in 8A)
  [
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 8, isMinor: false },
    { semitoneOffset: 8, isMinor: false },
    { semitoneOffset: 3, isMinor: false },
    { semitoneOffset: 3, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
  ],
  // 2. Deep Melodic House: i - iv - v - i (Am - Dm - Em - Am in 8A)
  [
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 5, isMinor: true },
    { semitoneOffset: 5, isMinor: true },
    { semitoneOffset: 7, isMinor: true },
    { semitoneOffset: 7, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
  ],
  // 3. Dark Club Driving Pedal: i - i - VI - VII (Am - Am - F - G in 8A)
  [
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 8, isMinor: false },
    { semitoneOffset: 8, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
  ],
  // 4. Trance Uplift / Progressive: i - VII - VI - VII (Am - G - F - G in 8A)
  [
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 0, isMinor: true },
    { semitoneOffset: 10, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
    { semitoneOffset: 8, isMinor: false },
    { semitoneOffset: 8, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
    { semitoneOffset: 10, isMinor: false },
  ],
];

export function buildProgressionFromKey(key: CamelotKeyInfo, templateIdx: number): ChordStep[] {
  const tmpl = HARMONIC_TEMPLATES[templateIdx % HARMONIC_TEMPLATES.length];
  return tmpl.map((st) => {
    const root = (key.rootPitch + st.semitoneOffset) % 12;
    const isMinor = key.isMinor ? st.isMinor : !st.isMinor;
    const bassHz = 440 * Math.pow(2, (root - 9 - 24) / 12);
    const chordHz = 440 * Math.pow(2, (root - 9 - 12) / 12);
    return {
      rootPitch: root,
      isMinor,
      bassHz,
      chordHz,
      name: `${ROOT_NAMES[root]}${isMinor ? "m" : ""}`,
    };
  });
}

export function getAdjacentCamelotKey(currentCode: string): string {
  const num = parseInt(currentCode, 10);
  const letter = currentCode.endsWith("B") ? "B" : "A";
  const roll = Math.random();
  if (roll < 0.40) {
    // Fifth up (+1 Camelot hour)
    const nextNum = (num % 12) + 1;
    return `${nextNum}${letter}`;
  } else if (roll < 0.75) {
    // Fifth down (-1 Camelot hour)
    const prevNum = ((num - 2 + 12) % 12) + 1;
    return `${prevNum}${letter}`;
  } else if (roll < 0.90) {
    // Relative Major / Minor toggle
    const altLetter = letter === "A" ? "B" : "A";
    return `${num}${altLetter}`;
  } else {
    // Energy boost (+2 Camelot hours / whole tone step)
    const boostNum = ((num + 1) % 12) + 1;
    return `${boostNum}${letter}`;
  }
}

// Built-in authentic harmonic dance chord progressions learned from the corpus
const DEFAULT_PROGRESSIONS: ChordStep[][] = [
  buildProgressionFromKey(CAMELOT_KEYS["8A"], 0),
  buildProgressionFromKey(CAMELOT_KEYS["8A"], 1),
  buildProgressionFromKey(CAMELOT_KEYS["8A"], 2),
  buildProgressionFromKey(CAMELOT_KEYS["8A"], 3),
];

// -----------------------------------------------------------------------------
// Vocal Song Lyrics & Melodic Formant Engine
// -----------------------------------------------------------------------------

export interface VowelFormant {
  f1: number;
  f2: number;
  f3: number;
}

export const VOWEL_FORMANTS: Record<string, VowelFormant> = {
  AH: { f1: 820, f2: 1240, f3: 2500 }, // open back ("ah", "god", "father")
  EE: { f1: 280, f2: 2280, f3: 2950 }, // close front ("feel", "see", "beat")
  OH: { f1: 520, f2: 950, f3: 2400 },  // close-mid back ("hold", "go", "so")
  OO: { f1: 320, f2: 850, f3: 2200 },  // close back ("wub", "ooh", "move")
  AY: { f1: 540, f2: 1850, f3: 2550 }, // close-mid front ("yeah", "say", "baby")
};

export interface VocalSyllable {
  text: string;
  vowel: "AH" | "EE" | "OH" | "OO" | "AY";
  semitoneOffset: number; // relative to chord root (0=root, 3=m3, 5=4th, 7=5th, 10=m7, 12=octave)
  step: number;           // 0..15 in pattern
  durationSteps: number;  // 1, 2, 4
  velocity: number;
}

export interface VocalSongTheme {
  title: string;
  personaName: string;
  territory: StyleTerritory;
  chorusHook: VocalSyllable[];
  verseLineA: VocalSyllable[];
  verseLineB: VocalSyllable[];
  buildupChant: VocalSyllable[];
  breakdownSolo: VocalSyllable[];
}

export class VocalSongArranger {
  private currentThemeIndex: number = 0;
  private themes: VocalSongTheme[] = [
    // Persona 0 (Wave): Ethereal Diva ("DREAM BEYOND", floating soprano formants, angelic sustains)
    {
      title: "DREAM BEYOND",
      personaName: "Ethereal Diva",
      territory: "wave",
      chorusHook: [
        { text: "DREAM", vowel: "EE", semitoneOffset: 12, step: 0, durationSteps: 4, velocity: 0.95 },
        { text: "BEYOND", vowel: "OH", semitoneOffset: 10, step: 4, durationSteps: 3, velocity: 0.90 },
        { text: "THE", vowel: "EE", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.88 },
        { text: "LIGHT", vowel: "AY", semitoneOffset: 12, step: 10, durationSteps: 3, velocity: 0.96 },
        { text: "FLOAT", vowel: "OH", semitoneOffset: 7, step: 13, durationSteps: 3, velocity: 0.92 },
      ],
      verseLineA: [
        { text: "STARS", vowel: "AH", semitoneOffset: 7, step: 0, durationSteps: 4, velocity: 0.88 },
        { text: "DRIFT", vowel: "EE", semitoneOffset: 5, step: 4, durationSteps: 3, velocity: 0.85 },
        { text: "A", vowel: "AH", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.82 },
        { text: "WAY", vowel: "AY", semitoneOffset: 3, step: 11, durationSteps: 4, velocity: 0.90 },
      ],
      verseLineB: [
        { text: "PEACE", vowel: "EE", semitoneOffset: 12, step: 0, durationSteps: 4, velocity: 0.88 },
        { text: "IN", vowel: "EE", semitoneOffset: 10, step: 4, durationSteps: 3, velocity: 0.85 },
        { text: "MY", vowel: "AY", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.85 },
        { text: "SOUL", vowel: "OH", semitoneOffset: 5, step: 11, durationSteps: 4, velocity: 0.90 },
      ],
      buildupChant: [
        { text: "FLOAT", vowel: "OH", semitoneOffset: 5, step: 0, durationSteps: 3, velocity: 0.85 },
        { text: "HIGH", vowel: "AY", semitoneOffset: 7, step: 4, durationSteps: 3, velocity: 0.88 },
        { text: "SOAR", vowel: "OH", semitoneOffset: 10, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "FREE", vowel: "EE", semitoneOffset: 12, step: 12, durationSteps: 2, velocity: 0.96 },
        { text: "NOW", vowel: "AH", semitoneOffset: 14, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "OOH", vowel: "OO", semitoneOffset: 12, step: 0, durationSteps: 6, velocity: 0.95 },
        { text: "BEYOND", vowel: "OH", semitoneOffset: 10, step: 6, durationSteps: 4, velocity: 0.90 },
        { text: "CALM", vowel: "AH", semitoneOffset: 7, step: 10, durationSteps: 3, velocity: 0.92 },
        { text: "PEACE", vowel: "EE", semitoneOffset: 5, step: 13, durationSteps: 3, velocity: 0.90 },
      ],
    },

    // Persona 1 (Cyber): Cyberpunk Vocoder ("NEON SYSTEM OVERRIDE", gritty mechanical formants, dark minor stutters)
    {
      title: "NEON SYSTEM OVERRIDE",
      personaName: "Cyberpunk Vocoder",
      territory: "cyber",
      chorusHook: [
        { text: "NE", vowel: "EE", semitoneOffset: 0, step: 0, durationSteps: 1, velocity: 1.0 },
        { text: "ON", vowel: "OH", semitoneOffset: 0, step: 1, durationSteps: 1, velocity: 0.95 },
        { text: "SYS", vowel: "EE", semitoneOffset: 3, step: 3, durationSteps: 1, velocity: 0.95 },
        { text: "TEM", vowel: "EE", semitoneOffset: 3, step: 4, durationSteps: 1, velocity: 0.95 },
        { text: "O", vowel: "OH", semitoneOffset: 7, step: 6, durationSteps: 1, velocity: 0.98 },
        { text: "VER", vowel: "EE", semitoneOffset: 7, step: 7, durationSteps: 1, velocity: 0.98 },
        { text: "RIDE", vowel: "AY", semitoneOffset: 10, step: 8, durationSteps: 3, velocity: 1.0 },
        { text: "GLITCH", vowel: "EE", semitoneOffset: 3, step: 12, durationSteps: 2, velocity: 0.92 },
        { text: "PULSE", vowel: "OO", semitoneOffset: 0, step: 14, durationSteps: 2, velocity: 0.96 },
      ],
      verseLineA: [
        { text: "DARK", vowel: "AH", semitoneOffset: 0, step: 0, durationSteps: 2, velocity: 0.92 },
        { text: "MA", vowel: "AH", semitoneOffset: 3, step: 3, durationSteps: 1, velocity: 0.90 },
        { text: "TRIX", vowel: "EE", semitoneOffset: 3, step: 4, durationSteps: 2, velocity: 0.90 },
        { text: "FEED", vowel: "EE", semitoneOffset: 5, step: 8, durationSteps: 2, velocity: 0.92 },
        { text: "BURN", vowel: "OO", semitoneOffset: 0, step: 12, durationSteps: 3, velocity: 0.95 },
      ],
      verseLineB: [
        { text: "SYN", vowel: "EE", semitoneOffset: 7, step: 0, durationSteps: 2, velocity: 0.92 },
        { text: "APSE", vowel: "AH", semitoneOffset: 5, step: 3, durationSteps: 2, velocity: 0.88 },
        { text: "CRACK", vowel: "AH", semitoneOffset: 3, step: 6, durationSteps: 2, velocity: 0.92 },
        { text: "CODE", vowel: "OH", semitoneOffset: 0, step: 11, durationSteps: 4, velocity: 0.96 },
      ],
      buildupChant: [
        { text: "O", vowel: "OH", semitoneOffset: 0, step: 0, durationSteps: 1, velocity: 0.85 },
        { text: "VER", vowel: "EE", semitoneOffset: 3, step: 2, durationSteps: 1, velocity: 0.88 },
        { text: "LOAD", vowel: "OH", semitoneOffset: 5, step: 4, durationSteps: 1, velocity: 0.90 },
        { text: "THE", vowel: "EE", semitoneOffset: 7, step: 6, durationSteps: 1, velocity: 0.92 },
        { text: "SYS", vowel: "EE", semitoneOffset: 10, step: 8, durationSteps: 1, velocity: 0.95 },
        { text: "TEM", vowel: "EE", semitoneOffset: 12, step: 10, durationSteps: 1, velocity: 0.98 },
        { text: "GO", vowel: "OH", semitoneOffset: 14, step: 12, durationSteps: 2, velocity: 1.0 },
        { text: "HARD", vowel: "AH", semitoneOffset: 15, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "LOST", vowel: "OH", semitoneOffset: 7, step: 0, durationSteps: 4, velocity: 0.95 },
        { text: "IN", vowel: "EE", semitoneOffset: 5, step: 5, durationSteps: 2, velocity: 0.88 },
        { text: "THE", vowel: "EE", semitoneOffset: 3, step: 8, durationSteps: 2, velocity: 0.88 },
        { text: "WIRE", vowel: "AY", semitoneOffset: 0, step: 11, durationSteps: 4, velocity: 0.95 },
      ],
    },

    // Persona 2 (House): Soulful House ("FEEL THE LOVE", rich soulful open vowels, driving melodic hooks)
    {
      title: "FEEL THE LOVE",
      personaName: "Soulful House",
      territory: "house",
      chorusHook: [
        { text: "FEEL", vowel: "EE", semitoneOffset: 12, step: 0, durationSteps: 2, velocity: 1.0 },
        { text: "THE", vowel: "OH", semitoneOffset: 10, step: 3, durationSteps: 2, velocity: 0.92 },
        { text: "LOVE", vowel: "AH", semitoneOffset: 12, step: 6, durationSteps: 2, velocity: 0.98 },
        { text: "DEEP", vowel: "EE", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.90 },
        { text: "IN", vowel: "EE", semitoneOffset: 5, step: 10, durationSteps: 2, velocity: 0.92 },
        { text: "YOUR", vowel: "OH", semitoneOffset: 7, step: 12, durationSteps: 2, velocity: 0.96 },
        { text: "SOUL", vowel: "OH", semitoneOffset: 3, step: 14, durationSteps: 2, velocity: 1.0 },
      ],
      verseLineA: [
        { text: "BABY", vowel: "AY", semitoneOffset: 0, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "LET", vowel: "EE", semitoneOffset: 3, step: 4, durationSteps: 2, velocity: 0.88 },
        { text: "IT", vowel: "EE", semitoneOffset: 5, step: 7, durationSteps: 2, velocity: 0.90 },
        { text: "SHINE", vowel: "AY", semitoneOffset: 0, step: 10, durationSteps: 4, velocity: 0.95 },
      ],
      verseLineB: [
        { text: "HEART", vowel: "AH", semitoneOffset: 7, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "BEAT", vowel: "EE", semitoneOffset: 5, step: 4, durationSteps: 2, velocity: 0.88 },
        { text: "SO", vowel: "OH", semitoneOffset: 7, step: 7, durationSteps: 2, velocity: 0.90 },
        { text: "RIGHT", vowel: "AY", semitoneOffset: 3, step: 10, durationSteps: 4, velocity: 0.95 },
      ],
      buildupChant: [
        { text: "HIGH", vowel: "AY", semitoneOffset: 5, step: 0, durationSteps: 2, velocity: 0.82 },
        { text: "ER", vowel: "AH", semitoneOffset: 7, step: 2, durationSteps: 2, velocity: 0.85 },
        { text: "TAKE", vowel: "AY", semitoneOffset: 8, step: 4, durationSteps: 2, velocity: 0.88 },
        { text: "ME", vowel: "EE", semitoneOffset: 10, step: 6, durationSteps: 2, velocity: 0.90 },
        { text: "HIGH", vowel: "AY", semitoneOffset: 12, step: 8, durationSteps: 2, velocity: 0.94 },
        { text: "ER", vowel: "AH", semitoneOffset: 12, step: 10, durationSteps: 2, velocity: 0.96 },
        { text: "LOVE", vowel: "AH", semitoneOffset: 14, step: 12, durationSteps: 2, velocity: 0.98 },
        { text: "YEAH", vowel: "AY", semitoneOffset: 15, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "OOH", vowel: "OO", semitoneOffset: 7, step: 0, durationSteps: 5, velocity: 0.95 },
        { text: "HOLD", vowel: "OH", semitoneOffset: 5, step: 6, durationSteps: 2, velocity: 0.88 },
        { text: "ON", vowel: "AH", semitoneOffset: 3, step: 8, durationSteps: 4, velocity: 0.92 },
        { text: "LOVE", vowel: "AH", semitoneOffset: 0, step: 12, durationSteps: 4, velocity: 0.96 },
      ],
    },

    // Persona 3 (Breaks): Phonk/Hype Chant ("GET DOWN DROP THE BASS", rapid syncopated punchy syllables)
    {
      title: "GET DOWN DROP THE BASS",
      personaName: "Phonk Hype Chant",
      territory: "breaks",
      chorusHook: [
        { text: "GET", vowel: "EE", semitoneOffset: 0, step: 0, durationSteps: 1, velocity: 1.0 },
        { text: "DOWN", vowel: "AH", semitoneOffset: 3, step: 2, durationSteps: 2, velocity: 0.95 },
        { text: "DROP", vowel: "OH", semitoneOffset: 5, step: 5, durationSteps: 1, velocity: 0.98 },
        { text: "THE", vowel: "EE", semitoneOffset: 7, step: 6, durationSteps: 1, velocity: 0.92 },
        { text: "BASS", vowel: "AY", semitoneOffset: 10, step: 8, durationSteps: 2, velocity: 1.0 },
        { text: "HIT", vowel: "EE", semitoneOffset: 5, step: 10, durationSteps: 1, velocity: 0.94 },
        { text: "THAT", vowel: "AH", semitoneOffset: 7, step: 12, durationSteps: 1, velocity: 0.96 },
        { text: "BREAK", vowel: "AY", semitoneOffset: 0, step: 14, durationSteps: 2, velocity: 1.0 },
      ],
      verseLineA: [
        { text: "ROLL", vowel: "OH", semitoneOffset: 0, step: 0, durationSteps: 2, velocity: 0.95 },
        { text: "THE", vowel: "EE", semitoneOffset: 3, step: 3, durationSteps: 1, velocity: 0.90 },
        { text: "DRUM", vowel: "AH", semitoneOffset: 5, step: 6, durationSteps: 2, velocity: 0.95 },
        { text: "JUMP", vowel: "AH", semitoneOffset: 0, step: 11, durationSteps: 3, velocity: 0.98 },
      ],
      verseLineB: [
        { text: "FEEL", vowel: "EE", semitoneOffset: 7, step: 0, durationSteps: 2, velocity: 0.95 },
        { text: "THE", vowel: "EE", semitoneOffset: 5, step: 3, durationSteps: 1, velocity: 0.90 },
        { text: "PUNCH", vowel: "AH", semitoneOffset: 3, step: 6, durationSteps: 2, velocity: 0.95 },
        { text: "ROCK", vowel: "OH", semitoneOffset: 0, step: 11, durationSteps: 3, velocity: 0.98 },
      ],
      buildupChant: [
        { text: "DROP", vowel: "OH", semitoneOffset: 0, step: 0, durationSteps: 1, velocity: 0.88 },
        { text: "IT", vowel: "EE", semitoneOffset: 3, step: 2, durationSteps: 1, velocity: 0.90 },
        { text: "DROP", vowel: "OH", semitoneOffset: 5, step: 4, durationSteps: 1, velocity: 0.92 },
        { text: "IT", vowel: "EE", semitoneOffset: 7, step: 6, durationSteps: 1, velocity: 0.94 },
        { text: "DROP", vowel: "OH", semitoneOffset: 10, step: 8, durationSteps: 1, velocity: 0.96 },
        { text: "THE", vowel: "EE", semitoneOffset: 12, step: 10, durationSteps: 1, velocity: 0.98 },
        { text: "BASS", vowel: "AY", semitoneOffset: 14, step: 12, durationSteps: 2, velocity: 1.0 },
        { text: "NOW", vowel: "AH", semitoneOffset: 15, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "HOLD", vowel: "OH", semitoneOffset: 7, step: 0, durationSteps: 4, velocity: 0.95 },
        { text: "UP", vowel: "AH", semitoneOffset: 5, step: 5, durationSteps: 2, velocity: 0.90 },
        { text: "READY", vowel: "EE", semitoneOffset: 3, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "GO", vowel: "OH", semitoneOffset: 0, step: 12, durationSteps: 3, velocity: 0.98 },
      ],
    },
  ];

  public getPersonaForTerritory(territory: StyleTerritory): VocalSongTheme {
    const idx = this.themes.findIndex((t) => t.territory === territory);
    return idx >= 0 ? this.themes[idx] : this.themes[0];
  }

  public getSongTitle(territory?: StyleTerritory): string {
    const theme = territory ? this.getPersonaForTerritory(territory) : this.themes[this.currentThemeIndex];
    return theme.title;
  }

  public getPersonaName(territory?: StyleTerritory): string {
    const theme = territory ? this.getPersonaForTerritory(territory) : this.themes[this.currentThemeIndex];
    return theme.personaName;
  }

  public nextSong(): void {
    this.currentThemeIndex = (this.currentThemeIndex + 1) % this.themes.length;
  }

  public getSyllableForStep(section: string, phraseBar: number, step: number, territory?: StyleTerritory): VocalSyllable | undefined {
    const theme = territory ? this.getPersonaForTerritory(territory) : this.themes[this.currentThemeIndex];
    let list: VocalSyllable[] = [];

    if (section === "DROP") {
      list = theme.chorusHook;
    } else if (section === "GROOVE_A") {
      list = (phraseBar % 2 === 0) ? theme.verseLineA : theme.verseLineB;
    } else if (section === "BUILDUP") {
      list = theme.buildupChant;
    } else if (section === "BREAKDOWN") {
      list = theme.breakdownSolo;
    } else if (section === "INTRO") {
      if (phraseBar === 0 || phraseBar === 4) {
        if (step === 0) {
          const introSyl: VocalSyllable = theme.territory === "cyber"
            ? { text: "SYS", vowel: "EE", semitoneOffset: 0, step: 0, durationSteps: 4, velocity: 0.90 }
            : (theme.territory === "breaks"
              ? { text: "ROLL", vowel: "OH", semitoneOffset: 0, step: 0, durationSteps: 4, velocity: 0.90 }
              : { text: "OOH", vowel: "OO", semitoneOffset: 7, step: 0, durationSteps: 6, velocity: 0.90 });
          return introSyl;
        }
      }
      return undefined;
    }

    return list.find((s) => s.step === step);
  }

  public getCurrentLyricLine(section: string, phraseBar: number, territory?: StyleTerritory): string {
    const theme = territory ? this.getPersonaForTerritory(territory) : this.themes[this.currentThemeIndex];
    if (section === "DROP") {
      return theme.chorusHook.map((s) => s.text).join(" ");
    }
    if (section === "GROOVE_A") {
      const list = (phraseBar % 2 === 0) ? theme.verseLineA : theme.verseLineB;
      return list.map((s) => s.text).join(" ");
    }
    if (section === "BUILDUP") {
      return theme.buildupChant.map((s) => s.text).join(" ");
    }
    if (section === "BREAKDOWN") {
      return theme.breakdownSolo.map((s) => s.text).join(" ");
    }
    return `${theme.personaName.toUpperCase()} (AMBIENT CALL)`;
  }
}

// -----------------------------------------------------------------------------
// 1. Client-Side Web Audio Algorithmic Slice Synthesizer
// -----------------------------------------------------------------------------

export class WebAudioSliceSynthesizer {
  private ctx: AudioContext;
  private noiseBuffer: AudioBuffer | null = null;
  private loadedSliceBuffers: Map<string, AudioBuffer> = new Map();
  private lastOpenHatGain: GainNode | null = null;

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.initNoiseBuffer();
  }

  private initNoiseBuffer(): void {
    const bufferSize = this.ctx.sampleRate * 2;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const output = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }
    this.noiseBuffer = buffer;
  }

  public registerSliceBuffer(key: string, buffer: AudioBuffer): void {
    this.loadedSliceBuffers.set(key, buffer);
  }

  public getSliceBuffer(key: string): AudioBuffer | undefined {
    return this.loadedSliceBuffers.get(key);
  }

  // Punchy 909/808 Kick: 5ms click transient + exponential sub body sweep with territory personality
  public playKick(when: number, velocity: number = 1.0, destNode?: AudioNode, territory?: StyleTerritory): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // 1. Click transient (high-frequency snap)
    const clickOsc = this.ctx.createOscillator();
    const clickGain = this.ctx.createGain();
    clickOsc.type = (territory === "cyber") ? "sawtooth" : "triangle";
    const startClickHz = (territory === "cyber") ? 1800 : (territory === "wave" ? 1100 : 1400);
    clickOsc.frequency.setValueAtTime(startClickHz, t);
    clickOsc.frequency.exponentialRampToValueAtTime(120, t + 0.008);
    clickGain.gain.setValueAtTime(velocity * (territory === "cyber" ? 1.05 : 0.9), t);
    clickGain.gain.exponentialRampToValueAtTime(0.001, t + 0.012);
    clickOsc.connect(clickGain);
    clickGain.connect(dest);
    clickOsc.start(t);
    clickOsc.stop(t + 0.015);

    // 2. Sub Body sweep
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "sine";

    let startSubHz = 155;
    let endSubHz = 45;
    let decaySec = 0.32;
    if (territory === "wave") {
      // 808 deep long sub kick
      startSubHz = 115;
      endSubHz = 36;
      decaySec = 0.44;
    } else if (territory === "cyber") {
      // Hard phonk punch kick
      startSubHz = 175;
      endSubHz = 52;
      decaySec = 0.22;
    } else if (territory === "breaks") {
      // Tight crunchy breakbeat kick
      startSubHz = 170;
      endSubHz = 54;
      decaySec = 0.19;
    }

    osc.frequency.setValueAtTime(startSubHz, t);
    osc.frequency.exponentialRampToValueAtTime(endSubHz, t + (territory === "wave" ? 0.095 : 0.070));

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * (territory === "cyber" ? 1.45 : 1.35), t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + decaySec);

    osc.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + decaySec + 0.03);
  }

  // Crisp Snare / Clap: Tuned triangle body + stereo bandpassed noise snap
  public playSnare(when: number, velocity: number = 1.0, destNode?: AudioNode, territory?: StyleTerritory): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // Body tone
    const osc = this.ctx.createOscillator();
    const oscGain = this.ctx.createGain();
    osc.type = (territory === "breaks") ? "sawtooth" : "triangle";
    const bodyHz = territory === "breaks" ? 220 : (territory === "wave" ? 175 : 195);
    osc.frequency.setValueAtTime(bodyHz, t);
    osc.frequency.exponentialRampToValueAtTime(bodyHz * 0.5, t + 0.06);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    oscGain.gain.setValueAtTime(v * 0.75, t);
    oscGain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
    osc.connect(oscGain);
    oscGain.connect(dest);

    osc.start(t);
    osc.stop(t + 0.14);

    // Snare white noise wire snap
    if (this.noiseBuffer) {
      const noise = this.ctx.createBufferSource();
      noise.buffer = this.noiseBuffer;

      const filter = this.ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = territory === "cyber" ? 2400 : (territory === "wave" ? 1400 : 1900);
      filter.Q.value = territory === "breaks" ? 2.2 : 1.4;

      const noiseGain = this.ctx.createGain();
      const noiseDur = territory === "wave" ? 0.32 : (territory === "breaks" ? 0.18 : 0.22);
      noiseGain.gain.setValueAtTime(v * (territory === "cyber" ? 1.05 : 0.95), t);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, t + noiseDur);

      noise.connect(filter);
      filter.connect(noiseGain);
      noiseGain.connect(dest);

      noise.start(t);
      noise.stop(t + noiseDur + 0.02);
    }
  }

  // Sizzling Hi-Hat: 6-oscillator metallic frequency cluster with open-hat choke
  public playHat(when: number, velocity: number = 1.0, isOpen: boolean = false, destNode?: AudioNode, territory?: StyleTerritory): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // Choke previous open hat if a closed hat hits
    if (!isOpen && this.lastOpenHatGain) {
      try {
        this.lastOpenHatGain.gain.cancelScheduledValues(t);
        this.lastOpenHatGain.gain.setValueAtTime(this.lastOpenHatGain.gain.value, t);
        this.lastOpenHatGain.gain.exponentialRampToValueAtTime(0.001, t + 0.015);
      } catch {}
      this.lastOpenHatGain = null;
    }

    if (!this.noiseBuffer) return;

    const noise = this.ctx.createBufferSource();
    noise.buffer = this.noiseBuffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.value = territory === "cyber" ? 9000 : (territory === "wave" ? 6500 : 7500);

    const dur = isOpen ? (territory === "wave" ? 0.35 : 0.26) : 0.045;
    const v = Math.max(0.1, Math.min(1.0, velocity));

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(v * (isOpen ? 0.85 : 0.65), t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(dest);

    if (isOpen) {
      this.lastOpenHatGain = gain;
    }

    noise.start(t);
    noise.stop(t + dur + 0.02);
  }

  // Tuned Percussion / Bongo / Rimshot
  public playPerc(when: number, velocity: number = 1.0, pitchHz: number = 420.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(pitchHz * 1.5, t);
    osc.frequency.exponentialRampToValueAtTime(pitchHz, t + 0.04);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 0.8, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.09);

    osc.connect(gain);
    gain.connect(dest);
    osc.start(t);
    osc.stop(t + 0.10);
  }

  // Deep Sub / Rolling Bass with Harmonic Tuning
  public playSub(
    when: number,
    velocity: number = 1.0,
    pitchHz: number = 55.0,
    durationSec: number = 0.22,
    destNode?: AudioNode,
    territory?: StyleTerritory
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = territory === "wave" ? "sine" : "sawtooth";
    osc.frequency.setValueAtTime(pitchHz, t);

    // Warm resonant low-pass filter tracking pitch
    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    const cutoff = territory === "wave"
      ? Math.min(320, pitchHz * 3.5)
      : (territory === "cyber" ? Math.min(1400, pitchHz * 8.0) : Math.min(800, pitchHz * 5.5));
    const targetCutoff = territory === "breaks" ? cutoff * 0.25 : cutoff * 0.45;
    filter.frequency.setValueAtTime(cutoff, t);
    filter.frequency.exponentialRampToValueAtTime(targetCutoff, t + durationSec);
    filter.Q.value = territory === "cyber" ? 4.2 : (territory === "breaks" ? 3.5 : 2.5);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 0.95, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + durationSec);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + durationSec + 0.02);
  }

  // Melodic Detuned Sawtooth Synth Chords with Harmonic Scale Realization
  public playChord(
    when: number,
    velocity: number = 1.0,
    rootHz: number = 130.81, // C3
    isMinor: boolean = true,
    durationSec: number = 0.35,
    destNode?: AudioNode,
    territory?: StyleTerritory
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // Chord intervals: Root, 3rd, 5th, 7th (subtle 7th for lush modern dance feel)
    const thirdMult = isMinor ? Math.pow(2, 3 / 12) : Math.pow(2, 4 / 12);
    const fifthMult = Math.pow(2, 7 / 12);
    const seventhMult = isMinor ? Math.pow(2, 10 / 12) : Math.pow(2, 11 / 12);
    const freqs = [rootHz, rootHz * thirdMult, rootHz * fifthMult, rootHz * seventhMult * 0.5];

    const chordGain = this.ctx.createGain();
    const v = Math.max(0.1, Math.min(1.0, velocity));
    chordGain.gain.setValueAtTime(v * 0.42, t);
    chordGain.gain.exponentialRampToValueAtTime(0.001, t + durationSec);

    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    const startFreq = territory === "wave" ? 1800 : (territory === "cyber" ? 3400 : 2600);
    const endFreq = territory === "wave" ? 1200 : 800;
    filter.frequency.setValueAtTime(startFreq, t);
    filter.frequency.exponentialRampToValueAtTime(endFreq, t + durationSec);
    filter.Q.value = territory === "cyber" ? 3.2 : 1.8;

    chordGain.connect(filter);
    filter.connect(dest);

    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      // Slight detune for analog richness
      const osc = this.ctx.createOscillator();
      osc.type = territory === "wave" ? (i % 2 === 0 ? "sawtooth" : "triangle") : "sawtooth";
      osc.frequency.value = f * (1.0 + (i % 2 === 0 ? 0.003 : -0.003));
      osc.connect(chordGain);
      osc.start(t);
      osc.stop(t + durationSec + 0.02);
    }
  }

  // 16th Melodic Arpeggio Pluck
  public playArp(
    when: number,
    velocity: number = 1.0,
    pitchHz: number = 220.0,
    destNode?: AudioNode,
    territory?: StyleTerritory
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = territory === "cyber" ? "triangle" : "square"; // Phonk cowbell harmonic / square
    osc.frequency.setValueAtTime(pitchHz, t);

    const filter = this.ctx.createBiquadFilter();
    filter.type = territory === "cyber" ? "bandpass" : "lowpass";
    const startFilter = territory === "cyber" ? 820 : 3200;
    const endFilter = territory === "cyber" ? 780 : 600;
    filter.frequency.setValueAtTime(startFilter, t);
    filter.frequency.exponentialRampToValueAtTime(endFilter, t + 0.12);
    filter.Q.value = territory === "cyber" ? 6.5 : 3.0;

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 0.55, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.14);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + 0.16);
  }

  // Vocal Formant Chop
  public playVocalChop(
    when: number,
    velocity: number = 1.0,
    formantHz: number = 750,
    destNode?: AudioNode
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(220, t);
    osc.frequency.exponentialRampToValueAtTime(330, t + 0.15);

    const f1 = this.ctx.createBiquadFilter();
    f1.type = "bandpass";
    f1.frequency.value = formantHz;
    f1.Q.value = 4.0;

    const v = Math.max(0.1, Math.min(1.0, velocity));
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(v * 0.70, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);

    osc.connect(f1);
    f1.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + 0.20);
  }

  // Human Singing Voice Synthesizer with Formant Resonators, Pitch Scoop, Vibrato & Multi-Persona Formants
  public playSingingVoice(
    when: number,
    noteHz: number = 261.6,
    vowel: "AH" | "EE" | "OH" | "OO" | "AY" = "AH",
    durationSec: number = 0.28,
    velocity: number = 0.90,
    destNode?: AudioNode,
    territory: StyleTerritory = "wave"
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const baseFormants = VOWEL_FORMANTS[vowel] || VOWEL_FORMANTS.AH;
    let f1Hz = baseFormants.f1;
    let f2Hz = baseFormants.f2;
    let f3Hz = baseFormants.f3;
    let q1 = 5.5;
    let q2 = 6.0;
    let q3 = 4.5;
    let vibratoFreq = 5.2;
    let vibratoDepthCents = 0.015;
    let scoopSemis = -1.5;
    let attackTime = 0.020;
    let holdTime = Math.max(0.06, durationSec - 0.06);
    let breathGainVal = 0.15;
    let osc1Type: OscillatorType = "sawtooth";
    let osc2Type: OscillatorType = "sawtooth";

    if (territory === "wave") {
      // Ethereal Diva: Floating soprano formants (+22% frequency), airy breath, shimmering vibrato
      f1Hz *= 1.22;
      f2Hz *= 1.22;
      f3Hz *= 1.25;
      vibratoFreq = 5.8;
      vibratoDepthCents = 0.020;
      breathGainVal = 0.22;
      attackTime = 0.045; // Gentle ethereal attack
      scoopSemis = -2.0;  // Floating angelic scoop
      osc1Type = "sine";
      osc2Type = "triangle";
    } else if (territory === "cyber") {
      // Cyberpunk Vocoder: Gritty mechanical formants, sharp robotic Q, zero scoop, square modulator
      q1 = 8.5;
      q2 = 9.0;
      q3 = 6.0;
      vibratoFreq = 0.0; // Robotic pitch stability
      vibratoDepthCents = 0.0;
      scoopSemis = 0.0; // Rigid digital pitch quantization
      attackTime = 0.005; // Immediate transient
      holdTime = Math.max(0.03, durationSec - 0.03);
      breathGainVal = 0.08;
      osc1Type = "sawtooth";
      osc2Type = "square"; // Vocoder gritty pulse
    } else if (territory === "house") {
      // Soulful House: Warm open vowels, resonant chest tone, soulful vibrato
      vibratoFreq = 4.8;
      vibratoDepthCents = 0.018;
      attackTime = 0.015;
      breathGainVal = 0.12;
      osc1Type = "sawtooth";
      osc2Type = "sawtooth";
    } else if (territory === "breaks") {
      // Phonk/Hype Chant: Rapid syncopated punchy syllables, pitch scoop-down punch
      scoopSemis = 2.5; // Punchy scoop down from top
      attackTime = 0.006; // Ultra-punchy
      holdTime = Math.max(0.02, durationSec * 0.7);
      vibratoFreq = 6.2;
      vibratoDepthCents = 0.008;
      breathGainVal = 0.18;
      osc1Type = "sawtooth";
      osc2Type = "triangle";
    }

    // 1. Dual Oscillators: Glottal wave + detuned harmonic body
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    osc1.type = osc1Type;
    osc2.type = osc2Type;

    const scoopStartHz = scoopSemis !== 0 ? noteHz * Math.pow(2, scoopSemis / 12) : noteHz;
    osc1.frequency.setValueAtTime(scoopStartHz, t);
    if (scoopSemis !== 0) {
      osc1.frequency.exponentialRampToValueAtTime(noteHz, t + (territory === "breaks" ? 0.025 : 0.04));
    }
    osc2.frequency.setValueAtTime(scoopStartHz * 1.002, t);
    if (scoopSemis !== 0) {
      osc2.frequency.exponentialRampToValueAtTime(noteHz * 1.002, t + (territory === "breaks" ? 0.025 : 0.04));
    }

    // Human Vibrato LFO
    const vibratoOsc = this.ctx.createOscillator();
    const vibratoGain = this.ctx.createGain();
    if (vibratoFreq > 0) {
      vibratoOsc.frequency.setValueAtTime(vibratoFreq, t);
      vibratoGain.gain.setValueAtTime(0.0001, t);
      vibratoGain.gain.setValueAtTime(0.0001, t + 0.06);
      vibratoGain.gain.linearRampToValueAtTime(noteHz * vibratoDepthCents, t + 0.18);
      vibratoOsc.connect(vibratoGain);
      vibratoGain.connect(osc1.frequency);
      vibratoGain.connect(osc2.frequency);
    }

    // 2. Breath Transient (Filtered noise attack for consonant definition)
    let breathNode: AudioNode | null = null;
    if (this.noiseBuffer) {
      const breathSrc = this.ctx.createBufferSource();
      breathSrc.buffer = this.noiseBuffer;
      const breathFilter = this.ctx.createBiquadFilter();
      breathFilter.type = "highpass";
      breathFilter.frequency.setValueAtTime(3200, t);
      const breathGain = this.ctx.createGain();
      breathGain.gain.setValueAtTime(velocity * breathGainVal, t);
      breathGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.025);
      breathSrc.connect(breathFilter);
      breathFilter.connect(breathGain);
      breathNode = breathGain;
      breathSrc.start(t);
      breathSrc.stop(t + 0.03);
    }

    // 3. Parallel Triple-Formant Vocal Tract Filter Bank
    const oscMix = this.ctx.createGain();
    oscMix.gain.setValueAtTime(0.5, t);
    osc1.connect(oscMix);
    osc2.connect(oscMix);

    const f1 = this.ctx.createBiquadFilter();
    f1.type = "bandpass";
    f1.frequency.setValueAtTime(f1Hz, t);
    f1.Q.setValueAtTime(q1, t);

    const f2 = this.ctx.createBiquadFilter();
    f2.type = "bandpass";
    f2.frequency.setValueAtTime(f2Hz, t);
    f2.Q.setValueAtTime(q2, t);

    const f3 = this.ctx.createBiquadFilter();
    f3.type = "bandpass";
    f3.frequency.setValueAtTime(f3Hz, t);
    f3.Q.setValueAtTime(q3, t);

    const g1 = this.ctx.createGain();
    g1.gain.setValueAtTime(1.0, t);
    const g2 = this.ctx.createGain();
    g2.gain.setValueAtTime(0.70, t);
    const g3 = this.ctx.createGain();
    g3.gain.setValueAtTime(0.35, t);

    oscMix.connect(f1);
    f1.connect(g1);
    oscMix.connect(f2);
    f2.connect(g2);
    oscMix.connect(f3);
    f3.connect(g3);

    // Sum formants
    const vocalSum = this.ctx.createGain();
    g1.connect(vocalSum);
    g2.connect(vocalSum);
    g3.connect(vocalSum);
    if (breathNode) breathNode.connect(vocalSum);

    // Vocal Master Envelope (Soft attack, warm sustain, natural release)
    const masterGain = this.ctx.createGain();
    const v = Math.max(0.1, Math.min(1.0, velocity));

    masterGain.gain.setValueAtTime(0.0001, t);
    masterGain.gain.linearRampToValueAtTime(v * 0.85, t + attackTime);
    masterGain.gain.setValueAtTime(v * 0.80, t + holdTime);
    masterGain.gain.exponentialRampToValueAtTime(0.0001, t + durationSec + 0.07);

    vocalSum.connect(masterGain);
    masterGain.connect(dest);

    osc1.start(t);
    osc2.start(t);
    if (vibratoFreq > 0) vibratoOsc.start(t);

    const stopTime = t + durationSec + 0.08;
    osc1.stop(stopTime);
    osc2.stop(stopTime);
    if (vibratoFreq > 0) vibratoOsc.stop(stopTime);
  }

  // Buildup Pitch Riser
  public playRiser(when: number, durationSec: number = 2.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(150, t);
    osc.frequency.exponentialRampToValueAtTime(1800, t + durationSec);

    gain.gain.setValueAtTime(0.05, t);
    gain.gain.linearRampToValueAtTime(0.35, t + durationSec);
    gain.gain.exponentialRampToValueAtTime(0.001, t + durationSec + 0.05);

    osc.connect(gain);
    gain.connect(dest);
    osc.start(t);
    osc.stop(t + durationSec + 0.06);
  }
}

// -----------------------------------------------------------------------------
// 2. Client-Side Markov Phrase & Song Arranger ML
// -----------------------------------------------------------------------------

export class ClientMarkovPhraseModel {
  private transitions: Record<string, Record<string, number>> = {};

  constructor(weights?: ComposingMLWeights) {
    if (weights && weights.phrase_transitions) {
      this.transitions = weights.phrase_transitions;
    } else {
      this.loadDefaults();
    }
  }

  public updateWeights(weights: ComposingMLWeights): void {
    if (weights.phrase_transitions) {
      this.transitions = weights.phrase_transitions;
    }
  }

  public sampleNextGroove(
    phraseBar: number,
    stem: StemName,
    currentGroove: string,
    rng: () => number = Math.random
  ): string {
    const key = `${phraseBar}:${stem}:${currentGroove}`;
    const weights = this.transitions[key];

    if (!weights || Object.keys(weights).length === 0) {
      // Fallback sensible defaults
      if (phraseBar === 7 && stem === "drums") return "turnaround_fill";
      if (stem === "drums") return "four_on_floor_hats";
      if (stem === "bass") return "offbeat";
      if (stem === "vocals") {
        if (phraseBar === 7) return "vocal_stutter";
        if (phraseBar === 2 || phraseBar === 4 || phraseBar === 6) return "vocal_hook";
        return "vocal_verse";
      }
      return "chords";
    }

    const entries = Object.entries(weights);
    const total = entries.reduce((acc, [, w]) => acc + w, 0);
    if (total <= 0) return entries[0][0];

    const pick = rng() * total;
    let accum = 0;
    for (const [gType, w] of entries) {
      accum += w;
      if (pick <= accum) return gType;
    }
    return entries[0][0];
  }

  private loadDefaults(): void {
    this.transitions = {
      "0:drums:four_on_floor": { four_on_floor_hats: 8.0, four_on_floor: 5.0, breakbeat: 2.0 },
      "1:drums:four_on_floor_hats": { four_on_floor_hats: 9.0, four_on_floor_syncopated: 4.0 },
      "2:drums:four_on_floor_hats": { four_on_floor_syncopated: 7.0, four_on_floor_hats: 5.0 },
      "3:drums:four_on_floor_syncopated": { mini_turnaround: 8.0, four_on_floor_hats: 3.0 },
      "4:drums:four_on_floor_hats": { four_on_floor_syncopated: 6.0, four_on_floor_hats: 5.0 },
      "7:drums:four_on_floor_hats": { turnaround_fill: 10.0, four_on_floor_hats: 1.0 },
      "0:bass:offbeat": { offbeat: 8.0, rolling: 4.0, sustained: 3.0 },
      "1:bass:offbeat": { rolling: 6.0, offbeat: 6.0 },
      "2:bass:rolling": { stabs: 6.0, rolling: 5.0 },
      "7:bass:rolling": { sustained: 8.0, stabs: 4.0 },
      "0:vocals:vocal_verse": { vocal_verse: 7.0, vocal_call_response: 5.0, vocal_hook: 3.0 },
      "1:vocals:vocal_call_response": { vocal_call_response: 8.0, vocal_syncopated_chop: 4.0 },
      "2:vocals:vocal_hook": { vocal_hook: 8.0, vocal_syncopated_chop: 6.0 },
      "3:vocals:vocal_call_response": { vocal_call_response: 7.0, vocal_verse: 4.0 },
      "4:vocals:vocal_hook": { vocal_hook: 8.0, vocal_syncopated_chop: 5.0 },
      "6:vocals:vocal_stutter": { vocal_stutter: 9.0, vocal_hook: 4.0 },
      "7:vocals:vocal_stutter": { vocal_stutter: 9.0, vocal_hook: 5.0 },
    };
  }
}

export class ClientSongArranger {
  private currentSection: string = "GROOVE_A";
  private barInSection: number = 0;
  private totalBars: number = 0;
  private sectionDuration: number = 16;
  private sectionEnergyTargets: Record<string, number> = {
    INTRO: 0.25,
    BUILDUP: 0.65,
    DROP: 0.90,
    GROOVE_A: 0.83,
    BREAKDOWN: 0.20,
  };

  private sectionTransitions: Record<string, Record<string, number>> = {
    INTRO: { BUILDUP: 8.0, GROOVE_A: 2.0 },
    BUILDUP: { DROP: 10.0 },
    DROP: { GROOVE_A: 7.0, BREAKDOWN: 3.0 },
    GROOVE_A: { DROP: 6.0, BUILDUP: 4.0, BREAKDOWN: 2.0 },
    BREAKDOWN: { BUILDUP: 8.0, GROOVE_A: 3.0 },
  };

  constructor(weights?: ComposingMLWeights) {
    if (weights) {
      if (weights.section_energy_targets) {
        this.sectionEnergyTargets = { ...this.sectionEnergyTargets, ...weights.section_energy_targets };
      }
      if (weights.section_transitions) {
        this.sectionTransitions = { ...this.sectionTransitions, ...weights.section_transitions };
      }
    }
  }

  public getSection(): string {
    return this.currentSection;
  }

  public getBarInSection(): number {
    return this.barInSection;
  }

  public getTotalBars(): number {
    return this.totalBars;
  }

  public getSectionBarsLeft(): number {
    return Math.max(0, this.sectionDuration - this.barInSection);
  }

  public getSectionDuration(): number {
    return this.sectionDuration;
  }

  public getTargetEnergy(): number {
    return this.sectionEnergyTargets[this.currentSection] || 0.65;
  }

  public advanceBar(): { isSectionChange: boolean; section: string } {
    this.barInSection++;
    this.totalBars++;

    if (this.barInSection >= this.sectionDuration) {
      this.transitionToNextSection();
      return { isSectionChange: true, section: this.currentSection };
    }
    return { isSectionChange: false, section: this.currentSection };
  }

  public transitionToNextSection(): void {
    const cur = this.currentSection;
    const trans = this.sectionTransitions[cur] || { GROOVE_A: 5.0, DROP: 5.0 };
    const entries = Object.entries(trans);
    const total = entries.reduce((acc, [, w]) => acc + w, 0);

    let nextSec = "GROOVE_A";
    const pick = Math.random() * total;
    let accum = 0;
    for (const [sec, w] of entries) {
      accum += w;
      if (pick <= accum) {
        nextSec = sec;
        break;
      }
    }

    this.currentSection = nextSec;
    this.barInSection = 0;
    this.sectionDuration = (nextSec === "BUILDUP" || nextSec === "INTRO" || nextSec === "BREAKDOWN") ? 8 : 16;
  }

  public setSection(section: string): void {
    this.currentSection = section;
    this.barInSection = 0;
    this.sectionDuration = (section === "BUILDUP" || section === "INTRO" || section === "BREAKDOWN") ? 8 : 16;
  }
}

// -----------------------------------------------------------------------------
// 3. Client Autonomous Groovebox Conductor & Step Sequencer
// -----------------------------------------------------------------------------

export class ClientGrooveboxConductor {
  private engine: AudioEngine;
  private synth: WebAudioSliceSynthesizer;
  private phraseModel: ClientMarkovPhraseModel;
  private arranger: ClientSongArranger;
  private manifest: StandaloneManifest | null = null;

  // Master Clock & Sequencer State
  private tatumIndex: number = 0;
  private phraseBar: number = 0; // 0..7
  private stepInBar: number = 0;  // 0..15
  private patternInCycle: number = 1;
  private patternsPerCycle: number = 8;
  private stepsPerPattern: number = 16;
  private cycleMode: string = "8x16";

  // Harmonic Progression & Camelot Key State
  private currentCamelotCode: string = "8A";
  private currentTemplateIndex: number = 0;
  private barsInCurrentKey: number = 0;
  private activeProgression: ChordStep[] = DEFAULT_PROGRESSIONS[0];
  private progressionIndex: number = 0;

  // Real-time Reticule Steering State (guides timbre, filters, pattern sliding & slices)
  private steering: Record<StemName, { x: number; y: number }> = {
    drums: { x: 0.72, y: 0.65 },
    bass: { x: 0.28, y: 0.62 },
    vocals: { x: 0.68, y: 0.32 },
    other: { x: 0.32, y: 0.35 },
  };

  // Dynamic Sidechain Nodes
  private bassSidechainGain: GainNode | null = null;
  private synthSidechainGain: GainNode | null = null;

  // Active Patterns per Stem
  private activePatterns: Record<StemName, SerializedPattern | null> = {
    drums: null,
    bass: null,
    vocals: null,
    other: null,
  };

  private stemRatings: Record<StemName, number> = {
    drums: 0,
    bass: 0,
    vocals: 0,
    other: 0,
  };

  public playbackMode: "slices" | "phrase_blocks" = "slices";
  private sliceRegistry: Map<string, SliceAsset[]> = new Map();
  private loopRegistry: Map<StemName, LoadedLoop[]> = new Map();
  private activeLoops: Partial<Record<StemName, LoadedLoop>> = {};
  private phraseBlockRegistry: Map<string, LoadedPhraseBlock[]> = new Map();
  private activePhraseBlock: LoadedPhraseBlock | null = null;
  private phraseBarCount: number = 0;
  private vocalSongArranger: VocalSongArranger = new VocalSongArranger();
  private currentVocalMeta = {
    lyric: "",
    songTitle: "",
    syllable: "",
    vowel: "",
    note: "",
    sectionName: "",
  };
  private isRunning: boolean = false;
  private timer: number | null = null;
  private nextStepTime: number = 0;
  private currentBpm: number = 126;
  private onMetaCallback?: (meta: any, atAudioSec: number, step: number, totalSteps: number) => void;

  constructor(engine: AudioEngine) {
    this.engine = engine;
    const ctx = engine.ensureCtx();
    this.synth = new WebAudioSliceSynthesizer(ctx);
    this.phraseModel = new ClientMarkovPhraseModel();
    this.arranger = new ClientSongArranger();
    this.initSidechain(ctx);
    this.initDefaultPatterns();
  }

  private initSidechain(ctx: AudioContext): void {
    try {
      this.bassSidechainGain = ctx.createGain();
      this.synthSidechainGain = ctx.createGain();

      const bassInput = this.engine.getStemInput("bass");
      const otherInput = this.engine.getStemInput("other");

      this.bassSidechainGain.connect(bassInput);
      this.synthSidechainGain.connect(otherInput);
    } catch {
      // In case engine is still initializing
    }
  }

  /**
   * Guides and steers the timbre, texture, and pattern sliding of a stem
   * based on reticule coordinates (0..1).
   */
  public setReticleSteering(stem: StemName, x: number, y: number): void {
    const cx = Math.max(0, Math.min(1, x));
    const cy = Math.max(0, Math.min(1, y));
    this.steering[stem] = { x: cx, y: cy };

    // Real-time Audio Filter manipulation based on X
    if (stem === "bass") {
      // 70 Hz (deep sub) to 3200 Hz (gritty acid growl)
      const freq = 70 * Math.pow(3200 / 70, cx);
      const q = 0.7 + 3.8 * cx;
      this.engine.setStemFilter("bass", freq, q);
    } else if (stem === "drums") {
      // 1200 Hz (warm muffled punch) to 20000 Hz (crisp sizzle)
      const freq = 1200 * Math.pow(20000 / 1200, cx);
      const q = 0.7 + 1.2 * cx;
      this.engine.setStemFilter("drums", freq, q);
    } else if (stem === "other") {
      // 250 Hz (warm floating ambient pad) to 18000 Hz (cutting saw lead)
      const freq = 250 * Math.pow(18000 / 250, cx);
      const q = 0.7 + 2.8 * cx;
      this.engine.setStemFilter("other", freq, q);
    } else if (stem === "vocals") {
      // 600 Hz (warm telephone/muffled) to 15000 Hz (bright shimmering vocal)
      const freq = 600 * Math.pow(15000 / 600, cx);
      const q = 0.8 + 1.6 * cx;
      this.engine.setStemFilter("vocals", freq, q);
    }
  }

  public getReticleSteering(stem: StemName): { x: number; y: number } {
    const s = this.steering[stem];
    return { x: s.x, y: s.y };
  }

  /**
   * Computes the active Map-Territory cluster for a stem:
   * - Top-Left (x < 0.5, y < 0.5) -> 'wave' (Liquid Wave / Ether)
   * - Top-Right (x >= 0.5, y < 0.5) -> 'cyber' (Cyberpunk Phonk)
   * - Bottom-Left (x < 0.5, y >= 0.5) -> 'house' (Deep Melodic House)
   * - Bottom-Right (x >= 0.5, y >= 0.5) -> 'breaks' (Electro Breaks)
   */
  public getStemTerritory(stem: StemName): StyleTerritory {
    const s = this.steering[stem];
    return getStyleTerritory(s.x, s.y);
  }

  public get territories(): Record<StemName, StyleTerritory> {
    return {
      drums: this.getStemTerritory("drums"),
      bass: this.getStemTerritory("bass"),
      other: this.getStemTerritory("other"),
      vocals: this.getStemTerritory("vocals"),
    };
  }

  public async loadStaticBundle(basePath: string = "data"): Promise<boolean> {
    try {
      // 1. Fetch Composing ML Weights
      const wRes = await fetch(`${basePath}/composing_ml_weights.json`);
      if (wRes.ok) {
        const weights = (await wRes.json()) as ComposingMLWeights;
        this.phraseModel.updateWeights(weights);
        this.arranger = new ClientSongArranger(weights);
        console.info("[Client ML] Loaded pre-trained composing Markov weights into in-browser engine.");
      }

      // 2. Fetch Standalone Patterns & Slices Manifest
      const pRes = await fetch(`${basePath}/standalone_patterns.json`);
      if (pRes.ok) {
        this.manifest = (await pRes.json()) as StandaloneManifest;
        console.info(
          `[Client ML] Loaded standalone pattern bank (${Object.keys(this.manifest.patterns).length} stems, ${this.manifest.phrases.length} phrases).`
        );
        // Load corpus-learned chord progressions if available
        if (this.manifest.chord_progressions && this.manifest.chord_progressions.length > 0) {
          const p = this.manifest.chord_progressions[0];
          const steps: ChordStep[] = [];
          for (let i = 0; i < 8; i++) {
            const root = p.roots[i % p.roots.length];
            const isMinor = p.key.toLowerCase().includes("min");
            const bassHz = 440 * Math.pow(2, (root - 9 - 24) / 12);
            const chordHz = 440 * Math.pow(2, (root - 9 - 12) / 12);
            steps.push({
              rootPitch: root,
              isMinor,
              bassHz,
              chordHz,
              name: `${ROOT_NAMES[root]}${isMinor ? "m" : ""}`,
            });
          }
          this.activeProgression = steps;
        }
        // Pre-decode high-priority static audio slice WAVs
        void this.preloadAudioSlices(basePath);
        // Preload authentic 1-bar studio whole-pattern stem loops
        void this.preloadPatternLoops(basePath);
        // Preload authentic 4-bar cohesive multi-stem phrase blocks
        void this.preloadPhraseBlocks(basePath);
      }
      return true;
    } catch (err) {
      console.warn("[Client ML] Bundle load failed, using algorithmic synthesizer fallback:", err);
      return false;
    }
  }

  private async preloadAudioSlices(basePath: string): Promise<void> {
    if (!this.manifest || !this.manifest.slices) return;
    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const stemSlices = this.manifest.slices[stem];
      if (!stemSlices) continue;
      for (const [cat, list] of Object.entries(stemSlices)) {
        const assets: SliceAsset[] = [];
        for (let i = 0; i < Math.min(8, list.length); i++) {
          const item = list[i];
          try {
            const url = item.url.startsWith("http") ? item.url : `${basePath}/${item.url.replace(/^data\//, "")}`;
            const res = await fetch(url);
            if (res.ok) {
              const bytes = await res.arrayBuffer();
              const decoded = await this.engine.decodeWav(bytes);
              const asset: SliceAsset = {
                buffer: decoded,
                pitch: typeof item.pitch === "number" ? item.pitch : 0,
                centroid: typeof item.centroid === "number" ? item.centroid : 0,
                duration: typeof item.duration === "number" ? item.duration : decoded.duration,
                loudness: typeof item.loudness === "number" ? item.loudness : 1.0,
                name: item.name,
              };
              assets.push(asset);
              this.synth.registerSliceBuffer(`${stem}:${cat}_${i}`, decoded);
              if (i === 0) {
                this.synth.registerSliceBuffer(`${stem}:${cat}`, decoded);
              }
            }
          } catch {
            // Silently fall back to algorithmic synth
          }
        }
        if (assets.length > 0) {
          this.sliceRegistry.set(`${stem}:${cat}`, assets);
        }
      }
    }
  }

  private async preloadPatternLoops(basePath: string): Promise<void> {
    if (!this.manifest || !this.manifest.loops) return;
    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const list = this.manifest.loops[stem];
      if (!list || list.length === 0) continue;
      const loaded: LoadedLoop[] = [];
      for (const item of list) {
        try {
          const url = item.url.startsWith("http") ? item.url : `${basePath}/${item.url.replace(/^data\//, "")}`;
          const res = await fetch(url);
          if (res.ok) {
            const bytes = await res.arrayBuffer();
            const decoded = await this.engine.decodeWav(bytes);
            loaded.push({ meta: item, buffer: decoded });
          }
        } catch {
          // Graceful fallback
        }
      }
      if (loaded.length > 0) {
        this.loopRegistry.set(stem, loaded);
      }
    }
    const totalLoops = Array.from(this.loopRegistry.values()).reduce((sum, l) => sum + l.length, 0);
    console.info(`[Client ML] Preloaded ${totalLoops} studio 1-bar whole-pattern stem loops across ${this.loopRegistry.size} stems.`);
  }

  private async preloadPhraseBlocks(basePath: string): Promise<void> {
    if (!this.manifest || !this.manifest.phrase_blocks) return;
    for (const item of this.manifest.phrase_blocks) {
      const buffers: Partial<Record<StemName, AudioBuffer>> = {};
      for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
        const urlStr = item.stems[stem];
        if (!urlStr) continue;
        try {
          const url = urlStr.startsWith("http") ? urlStr : `${basePath}/${urlStr.replace(/^data\//, "")}`;
          const res = await fetch(url);
          if (res.ok) {
            const bytes = await res.arrayBuffer();
            const decoded = await this.engine.decodeWav(bytes);
            buffers[stem] = decoded;
          }
        } catch {
          // Graceful fallback
        }
      }
      if (Object.keys(buffers).length > 0) {
        const sec = item.section;
        const list = this.phraseBlockRegistry.get(sec) || [];
        list.push({ meta: item, buffers });
        this.phraseBlockRegistry.set(sec, list);
      }
    }
    const totalBlocks = Array.from(this.phraseBlockRegistry.values()).reduce((sum, l) => sum + l.length, 0);
    console.info(`[Client ML] Preloaded ${totalBlocks} studio 4-bar cohesive multi-stem phrase blocks across ${this.phraseBlockRegistry.size} sections.`);
  }

  public getBestPhraseBlock(section: string): LoadedPhraseBlock | null {
    const list = this.phraseBlockRegistry.get(section);
    if (!list || list.length === 0) {
      for (const fallbackSec of ["GROOVE_A", "DROP", "BUILDUP", "BREAKDOWN", "INTRO"]) {
        const fbList = this.phraseBlockRegistry.get(fallbackSec);
        if (fbList && fbList.length > 0) {
          return fbList[Math.floor(Math.random() * fbList.length)];
        }
      }
      return null;
    }
    return list[Math.floor(Math.random() * list.length)];
  }

  public getBestLoop(
    stem: StemName,
    section: string,
    phraseBar: number,
    targetPitch?: number
  ): { loop: LoadedLoop; rate: number } | null {
    const list = this.loopRegistry.get(stem);
    if (!list || list.length === 0) return null;

    let candidates = list;

    if (section === "DROP") {
      const dropCandidates = list.filter((l) =>
        l.meta.suggested_sections?.includes("DROP") ||
        ["drop", "rolling", "riff", "hook"].includes(l.meta.category)
      );
      if (dropCandidates.length > 0) candidates = dropCandidates;
    } else if (section === "BUILDUP") {
      const buildupCandidates = list.filter((l) =>
        l.meta.suggested_sections?.includes("BUILDUP") ||
        ["buildup", "turnaround", "stutter", "call"].includes(l.meta.category)
      );
      if (buildupCandidates.length > 0) candidates = buildupCandidates;
    } else if (section === "BREAKDOWN" || section === "INTRO") {
      const calmCandidates = list.filter((l) =>
        l.meta.suggested_sections?.includes("BREAKDOWN") ||
        l.meta.suggested_sections?.includes("INTRO") ||
        ["ambient", "verse", "pad", "chords", "minimal"].includes(l.meta.category)
      );
      if (calmCandidates.length > 0) candidates = calmCandidates;
    } else {
      // GROOVE_A
      const grooveCandidates = list.filter((l) =>
        l.meta.suggested_sections?.includes("GROOVE_A") ||
        ["groove_a", "sub_pump", "chords", "verse"].includes(l.meta.category)
      );
      if (grooveCandidates.length > 0) candidates = grooveCandidates;
    }

    if (stem === "drums" || targetPitch === undefined) {
      const pick = candidates[Math.floor(Math.random() * candidates.length)];
      const rate = this.currentBpm / pick.meta.source_bpm;
      return { loop: pick, rate };
    }

    let best = candidates[0];
    let bestDist = 999;
    let bestShift = 0;

    for (const c of candidates) {
      let shift = (targetPitch - c.meta.dominant_pitch) % 12;
      if (shift > 6) shift -= 12;
      if (shift < -6) shift += 12;
      const absShift = Math.abs(shift);
      if (absShift < bestDist) {
        bestDist = absShift;
        bestShift = shift;
        best = c;
      }
    }

    const pitchFactor = Math.pow(2, bestShift / 12);
    const rate = (this.currentBpm / best.meta.source_bpm) * pitchFactor;
    return { loop: best, rate };
  }

  private getBestSlice(
    stemCategory: string,
    targetPitch?: number,
    variationFraction?: number,
    territory?: StyleTerritory
  ): { buffer: AudioBuffer; rate: number } | null {
    // 1. Look for style-specific slices first (e.g. drums:kick:wave, bass:sub:cyber)
    let list: SliceAsset[] | undefined;
    if (territory) {
      list = this.sliceRegistry.get(`${stemCategory}:${territory}`);
    }
    // Fallback to general stem category
    if (!list || list.length === 0) {
      list = this.sliceRegistry.get(stemCategory);
    }
    if (!list || list.length === 0) {
      // Intelligent fallback within same stem
      const [stem, cat] = stemCategory.split(":");
      if (stem === "vocals") {
        for (const alt of ["hook", "call", "chop"]) {
          if (alt !== cat) {
            const altList = (territory ? this.sliceRegistry.get(`vocals:${alt}:${territory}`) : undefined) ||
              this.sliceRegistry.get(`vocals:${alt}`);
            if (altList && altList.length > 0) {
              list = altList;
              break;
            }
          }
        }
      } else if (stem === "bass") {
        for (const alt of ["sub", "rolling", "stab"]) {
          if (alt !== cat) {
            const altList = (territory ? this.sliceRegistry.get(`bass:${alt}:${territory}`) : undefined) ||
              this.sliceRegistry.get(`bass:${alt}`);
            if (altList && altList.length > 0) {
              list = altList;
              break;
            }
          }
        }
      } else if (stem === "other") {
        for (const alt of ["chord", "arp", "pad"]) {
          if (alt !== cat) {
            const altList = (territory ? this.sliceRegistry.get(`other:${alt}:${territory}`) : undefined) ||
              this.sliceRegistry.get(`other:${alt}`);
            if (altList && altList.length > 0) {
              list = altList;
              break;
            }
          }
        }
      }
    }

    if (!list || list.length === 0) {
      const fallbackBuf = (territory ? this.synth.getSliceBuffer(`${stemCategory}:${territory}`) : undefined) ||
        this.synth.getSliceBuffer(stemCategory);
      return fallbackBuf ? { buffer: fallbackBuf, rate: 1.0 } : null;
    }

    if (targetPitch === undefined) {
      // Pick with reticule sliding variation across available slices
      const v = variationFraction !== undefined ? Math.max(0, Math.min(0.999, variationFraction)) : Math.random();
      const idx = Math.floor(v * list.length) % list.length;
      return { buffer: list[idx].buffer, rate: 1.0 };
    }

    // For pitched stems (bass, other, vocals): find slice closest to target pitch
    let best = list[0];
    let bestDiff = 999;
    let bestShift = 0;
    const candidates: Array<{ item: SliceAsset; shift: number }> = [];

    for (const item of list) {
      let diff = (targetPitch - item.pitch) % 12;
      if (diff > 6) diff -= 12;
      if (diff < -6) diff += 12;
      const absDiff = Math.abs(diff);
      if (absDiff < bestDiff) {
        bestDiff = absDiff;
        bestShift = diff;
        best = item;
      }
      if (absDiff <= 1) {
        candidates.push({ item, shift: diff });
      }
    }

    // If multiple candidates have closest pitch, use variationFraction to slide through them
    if (candidates.length > 0 && variationFraction !== undefined) {
      const v = Math.max(0, Math.min(0.999, variationFraction));
      const cIdx = Math.floor(v * candidates.length) % candidates.length;
      best = candidates[cIdx].item;
      bestShift = candidates[cIdx].shift;
    }

    // If exact chromatic pitch exists (bestDiff == 0), rate is 1.0 (zero drift!).
    // Otherwise clamp shift to +/- 1 semitone max.
    const clampedShift = Math.max(-1, Math.min(1, bestShift));
    const rate = Math.pow(2, clampedShift / 12);
    return { buffer: best.buffer, rate };
  }

  public setCycleMode(mode: string): boolean {
    const valid = ["4x16", "8x16", "16x32"];
    if (!valid.includes(mode)) return false;
    this.cycleMode = mode;
    if (mode === "4x16") {
      this.patternsPerCycle = 4;
      this.stepsPerPattern = 16;
    } else if (mode === "8x16") {
      this.patternsPerCycle = 8;
      this.stepsPerPattern = 16;
    } else if (mode === "16x32") {
      this.patternsPerCycle = 16;
      this.stepsPerPattern = 32;
    }
    this.patternInCycle = 1;
    return true;
  }

  public setMetaCallback(cb: (meta: any, atAudioSec: number, step: number, totalSteps: number) => void): void {
    this.onMetaCallback = cb;
  }

  public setBpm(bpm: number): void {
    this.currentBpm = Math.max(60, Math.min(200, bpm));
  }

  public recordFeedback(stem: StemName, rating: 1 | -1): void {
    this.stemRatings[stem] += rating;
    if (rating < 0) {
      this.rerollStem(stem);
    }
  }

  public rerollStem(stem: StemName): void {
    if (this.activePhraseBlock && this.activePhraseBlock.buffers[stem]) {
      delete this.activePhraseBlock.buffers[stem];
    }
    const sec = this.arranger.getSection();
    const chord = this.activeProgression[this.phraseBar % this.activeProgression.length];
    const loopMatch = this.getBestLoop(stem, sec, this.phraseBar, stem === "drums" ? undefined : chord.rootPitch);
    if (loopMatch) {
      this.activeLoops[stem] = loopMatch.loop;
      const grid = loopMatch.loop.meta.onset_grid;
      const onsetIndices: number[] = [];
      const vels: number[] = new Array(16).fill(0.0);
      for (let i = 0; i < 16; i++) {
        if (grid[i] === 1) {
          onsetIndices.push(i);
          vels[i] = 0.95;
        }
      }
      this.activePatterns[stem] = {
        id: loopMatch.loop.meta.id,
        name: loopMatch.loop.meta.name,
        groove_type: loopMatch.loop.meta.category,
        energy: sec === "DROP" ? 0.95 : (sec === "BUILDUP" ? 0.80 : 0.70),
        density: onsetIndices.length / 16,
        onset_grid: onsetIndices,
        velocities: vels,
        length: 16,
        key: loopMatch.loop.meta.key,
      };
      return;
    }

    let targetGroove = "four_on_floor_hats";
    if (stem === "drums") {
      targetGroove = sec === "BUILDUP" ? "turnaround_fill" : (sec === "DROP" ? "four_on_floor_syncopated" : "breakbeat");
    } else if (stem === "bass") {
      targetGroove = sec === "DROP" ? "rolling" : "offbeat";
    } else if (stem === "other") {
      targetGroove = sec === "DROP" ? "riff" : "chords";
    } else if (stem === "vocals") {
      targetGroove = "chop";
    }
    this.activePatterns[stem] = this.synthesizePattern(stem, targetGroove);
  }

  public start(bpm: number): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.currentBpm = Math.max(60, Math.min(200, bpm));
    this.nextStepTime = this.engine.now() + 0.05;
    this.tatumIndex = 0;

    // Pick a progression
    this.progressionIndex = (this.progressionIndex + 1) % DEFAULT_PROGRESSIONS.length;
    this.activeProgression = DEFAULT_PROGRESSIONS[this.progressionIndex];

    this.composeBarPatterns(0);

    this.timer = window.setInterval(() => this.clockTick(), 20);
  }

  public stop(): void {
    this.isRunning = false;
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private clockTick(): void {
    if (!this.isRunning) return;
    const now = this.engine.now();
    const tatumSec = 60 / this.currentBpm / 4;

    // Clamp if nextStepTime fell behind current audio clock (e.g. background tab)
    if (this.nextStepTime < now) {
      this.nextStepTime = now + 0.02;
    }

    // Schedule 150ms ahead into Web Audio
    while (this.nextStepTime < now + 0.15) {
      this.processStep(this.nextStepTime);
      this.nextStepTime += tatumSec;
    }
  }

  private processStep(when: number): void {
    const tatum = this.tatumIndex;
    const stepInPattern = tatum % this.stepsPerPattern;
    const patternInCycle = Math.floor(tatum / this.stepsPerPattern) % this.patternsPerCycle;
    const stepInBar = tatum % 16;
    const phraseBar = Math.floor(tatum / 16) % 8;
    const isBarStart = (stepInBar === 0);

    if (isBarStart) {
      this.onBarStart();
      this.composeBarPatterns(phraseBar);
    }

    // Trigger audio hits for all active stems
    this.scheduleStepAudio(stepInPattern, when, phraseBar);

    // Emit live GrooveboxMeta with exact audio schedule timestamp
    if (this.onMetaCallback) {
      const meta = this.buildMeta(stepInPattern, patternInCycle, phraseBar);
      this.onMetaCallback(meta, when, stepInPattern, this.stepsPerPattern);
    }

    // Advance master monotonic tatum step
    this.tatumIndex++;
  }

  private onBarStart(): void {
    const { isSectionChange, section } = this.arranger.advanceBar();
    this.barsInCurrentKey++;

    if (isSectionChange || this.barsInCurrentKey >= 32) {
      // Modulate key on Camelot wheel for continuous infinite harmonic drift!
      const shouldModulate = isSectionChange ? (Math.random() < 0.50) : (this.barsInCurrentKey >= 32);
      if (shouldModulate) {
        this.currentCamelotCode = getAdjacentCamelotKey(this.currentCamelotCode);
        this.currentTemplateIndex = (this.currentTemplateIndex + 1) % HARMONIC_TEMPLATES.length;
        this.barsInCurrentKey = 0;
        const keyInfo = CAMELOT_KEYS[this.currentCamelotCode] || CAMELOT_KEYS["8A"];
        this.activeProgression = buildProgressionFromKey(keyInfo, this.currentTemplateIndex);
        console.info(`[KG-505 Harmonic Drift] Key shifted to ${keyInfo.code} (${keyInfo.name}) • Template #${this.currentTemplateIndex + 1}`);
      } else {
        // Switch progression template within current key
        this.currentTemplateIndex = (this.currentTemplateIndex + 1) % HARMONIC_TEMPLATES.length;
        const keyInfo = CAMELOT_KEYS[this.currentCamelotCode] || CAMELOT_KEYS["8A"];
        this.activeProgression = buildProgressionFromKey(keyInfo, this.currentTemplateIndex);
      }

      this.phraseBarCount = 0;
      this.activePhraseBlock = null;
      // Progress to next song composition when cycling into a new verse/song cycle
      if (section === "GROOVE_A" || section === "INTRO") {
        this.vocalSongArranger.nextSong();
      }
    } else {
      this.phraseBarCount = (this.phraseBarCount + 1) % 4;
    }
  }

  private schedulePhraseBlock(phrase: LoadedPhraseBlock, when: number, sec: string, barInSec: number): void {
    const isBuildupClimax = (sec === "BUILDUP" && barInSec >= 4);
    // On Bar 7 Step 15 of BUILDUP: Silent drop gap breath!
    const dropGapBreathSec = isBuildupClimax ? (14.85 * 60) / (this.currentBpm * 4) : undefined;

    // Tempo-only rate: NO PITCH RESAMPLING, ZERO DRIFT!
    const rate = this.currentBpm / phrase.meta.source_bpm;

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const buffer = phrase.buffers[stem];
      if (!buffer) continue;

      // Invariant section muting rules:
      if (sec === "BREAKDOWN" && (stem === "drums" || stem === "bass")) {
        continue;
      }
      if (sec === "INTRO" && stem === "bass") {
        continue;
      }
      if (sec === "BUILDUP" && stem === "bass" && barInSec >= 6) {
        continue;
      }

      this.engine.scheduleSlice(stem, buffer, when, {
        rate,
        duration: dropGapBreathSec,
        fadeMs: 6,
      });

      const grid = phrase.meta.onset_grids[stem] || [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
      const onsetIndices: number[] = [];
      const vels: number[] = new Array(16).fill(0.0);
      for (let i = 0; i < 16; i++) {
        if (grid[i] === 1) {
          onsetIndices.push(i);
          vels[i] = 0.95;
        }
      }

      this.activePatterns[stem] = {
        id: `${phrase.meta.id}_${stem}`,
        name: `${phrase.meta.track_title} [${sec}]`,
        groove_type: phrase.meta.section.toLowerCase(),
        energy: phrase.meta.energy,
        density: onsetIndices.length / 16,
        onset_grid: onsetIndices,
        velocities: vels,
        length: 16,
        key: phrase.meta.key,
      };

      if (stem === "vocals") {
        this.currentVocalMeta = {
          lyric: phrase.meta.track_title,
          songTitle: phrase.meta.track_title,
          syllable: sec,
          vowel: "EE",
          note: phrase.meta.key,
          sectionName: sec,
        };
      }
    }
  }

  private applySidechainDucking(when: number): void {
    // Dip bass by -9 dB (gain 0.35) and exponential release over 70ms
    if (this.bassSidechainGain) {
      const t = Math.max(this.engine.now(), when);
      this.bassSidechainGain.gain.cancelScheduledValues(t);
      this.bassSidechainGain.gain.setValueAtTime(0.35, t);
      this.bassSidechainGain.gain.exponentialRampToValueAtTime(1.0, t + 0.075);
    }
    // Dip synths slightly by -4 dB (gain 0.63) and release over 60ms
    if (this.synthSidechainGain) {
      const t = Math.max(this.engine.now(), when);
      this.synthSidechainGain.gain.cancelScheduledValues(t);
      this.synthSidechainGain.gain.setValueAtTime(0.63, t);
      this.synthSidechainGain.gain.exponentialRampToValueAtTime(1.0, t + 0.060);
    }
  }

  private scheduleStepAudio(step: number, when: number, phraseBar: number): void {
    const sec = this.arranger.getSection();
    const barInSec = this.arranger.getBarInSection();
    const chord = this.activeProgression[phraseBar % this.activeProgression.length];

    const dInput = this.engine.getStemInput("drums");
    const bDest = this.bassSidechainGain || this.engine.getStemInput("bass");
    const oDest = this.synthSidechainGain || this.engine.getStemInput("other");
    const vInput = this.engine.getStemInput("vocals");

    // Active Style Territories per stem
    const dTerr = this.getStemTerritory("drums");
    const bTerr = this.getStemTerritory("bass");
    const oTerr = this.getStemTerritory("other");
    const vTerr = this.getStemTerritory("vocals");

    // 0. Downbeat handling: phrase blocks mode vs authentic learned slice mode
    if (this.playbackMode === "phrase_blocks") {
      if (step % 16 === 0) {
        if (this.phraseBarCount === 0 || !this.activePhraseBlock) {
          this.activePhraseBlock = this.getBestPhraseBlock(sec);
          if (this.activePhraseBlock) {
            this.schedulePhraseBlock(this.activePhraseBlock, when, sec, barInSec);
          }
        }
        if (!this.activePhraseBlock) {
          this.scheduleBarLoops(when, sec, phraseBar, chord);
        }
      }
    } else {
      this.activePhraseBlock = null;
      this.activeLoops = {};
    }

    // -------------------------------------------------------------------------
    // 1. DRUMS (Section-Aware Arrangement with Real Acoustic Slices & Reticule Steering)
    // -------------------------------------------------------------------------
    const dPat = this.activePatterns.drums;
    const dSteer = this.steering.drums;
    if (this.playbackMode === "phrase_blocks" && (this.activePhraseBlock || this.activeLoops.drums)) {
      // Continuous studio drum phrase or loop is active: dynamic sidechain ducking on downbeats
      if (sec !== "BREAKDOWN" && [0, 4, 8, 12].includes(step % 16)) {
        this.applySidechainDucking(when);
      }
      // Trigger Pitch Riser on bar 6 step 0 of BUILDUP
      if (sec === "BUILDUP" && barInSec === 6 && step === 0) {
        this.synth.playRiser(when, (60 / this.currentBpm) * 8, dInput);
      }
    } else if (dPat) {
      const isBuildup = (sec === "BUILDUP");
      const isBreakdown = (sec === "BREAKDOWN");

      if (isBreakdown) {
        // Breakdown: Kicks completely muted, gentle closed hats on 2, 6, 10, 14
        if ([2, 6, 10, 14].includes(step % 16)) {
          const hatSlice = this.getBestSlice("drums:hat_closed", undefined, dSteer.y, dTerr);
          if (hatSlice) {
            this.engine.scheduleSlice("drums", hatSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.0,
              gain: 0.45,
              voiceKey: "drums:hat_closed",
            });
          } else {
            this.synth.playHat(when, 0.45, false, dInput, dTerr);
          }
        }
      } else if (isBuildup) {
        // Buildup: Snare acceleration & tension riser!
        let playSnare = false;
        let snareVel = 0.70;

        if (barInSec < 4) {
          // Bars 1-4: Quarter note snares on 4, 12
          playSnare = (step === 4 || step === 12);
        } else if (barInSec < 6) {
          // Bars 5-6: 8th note snares
          playSnare = (step % 2 === 0);
          snareVel = 0.85;
        } else {
          // Bars 7-8: 16th note continuous crescendo roll
          playSnare = (step < 15); // STEP 15 SILENT for drop breath!
          snareVel = 0.75 + (step / 16) * 0.25;
        }

        if (playSnare) {
          const snareSlice = this.getBestSlice("drums:snare", undefined, dSteer.y, dTerr);
          if (snareSlice) {
            this.engine.scheduleSlice("drums", snareSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: snareVel,
              voiceKey: "drums:snare",
            });
          } else {
            this.synth.playSnare(when, snareVel, dInput, dTerr);
          }
        }

        // Downbeat Kick pulse (bars 1-5 only)
        if (step % 4 === 0 && barInSec < 6) {
          const kickSlice = this.getBestSlice("drums:kick", undefined, dSteer.y, dTerr);
          if (kickSlice) {
            this.engine.scheduleSlice("drums", kickSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: 0.90,
              voiceKey: "drums:kick",
            });
          } else {
            this.synth.playKick(when, 0.90, dInput, dTerr);
          }
          this.applySidechainDucking(when);
        }

        // Trigger Pitch Riser on bar 6 step 0
        if (barInSec === 6 && step === 0) {
          this.synth.playRiser(when, (60 / this.currentBpm) * 8, dInput);
        }
      } else {
        // Standard Groove / Drop: Multi-territory rhythm driving forward
        const hasHit = dPat.onset_grid.includes(step);
        const vel = dPat.velocities ? dPat.velocities[step] : 1.0;

        // Kick triggers based on active Map-Territory quadrant
        let isKickStep = false;
        if (dTerr === "wave") {
          // Wave: Halftime / ambient dub kicks
          isKickStep = (step % 16 === 0 || step % 16 === 8) || ((step % 16 === 10) && dSteer.y > 0.55);
        } else if (dTerr === "cyber") {
          // Cyber: Phonk syncopated kicks on step 10 & 14 (+ 0, 4, 8)
          isKickStep = [0, 4, 8, 10, 14].includes(step % 16);
        } else if (dTerr === "breaks") {
          // Breaks: 2-step / syncopated breakbeat (kicks on 0, 6, 10)
          isKickStep = [0, 6, 10].includes(step % 16);
        } else {
          // House: Classic 4-on-the-floor
          isKickStep = [0, 4, 8, 12].includes(step % 16) || ((step % 16 === 10) && (dSteer.y > 0.65 || dPat.groove_type.includes("syncopated")));
        }

        if (isKickStep && (hasHit || isKickStep)) {
          const kickSlice = this.getBestSlice("drums:kick", undefined, dSteer.y, dTerr);
          if (kickSlice) {
            this.engine.scheduleSlice("drums", kickSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: vel,
              voiceKey: "drums:kick",
            });
          } else {
            this.synth.playKick(when, vel, dInput, dTerr);
          }
          this.applySidechainDucking(when);
        }

        // Snare / Clap triggers based on active territory
        let isSnareStep = false;
        if (dTerr === "wave") {
          isSnareStep = (step % 16 === 8) || (phraseBar === 7 && step % 16 >= 8 && step % 16 !== 15);
        } else if (dTerr === "breaks") {
          isSnareStep = (step % 16 === 4 || step % 16 === 12) || (dSteer.y > 0.65 && (step % 16 === 7 || (step % 16 === 15 && phraseBar !== 7)));
        } else {
          isSnareStep = ((step % 16) === 4 || (step % 16) === 12) ||
            (phraseBar === 7 && (step % 16) >= 8 && (step % 16) !== 15) ||
            (dSteer.y > 0.75 && (step % 16) === 14);
        }

        if (isSnareStep) {
          const snareSlice = this.getBestSlice("drums:snare", undefined, dSteer.y, dTerr);
          if (snareSlice) {
            this.engine.scheduleSlice("drums", snareSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: vel,
              voiceKey: "drums:snare",
            });
          } else {
            this.synth.playSnare(when, vel, dInput, dTerr);
          }
        }

        // Hi-Hats: Open hat on upbeats (2, 6, 10, 14) + closed 16th hats with choking
        if ([2, 6, 10, 14].includes(step % 16)) {
          const hatSlice = this.getBestSlice("drums:hat_open", undefined, dSteer.y, dTerr);
          if (hatSlice) {
            this.engine.scheduleSlice("drums", hatSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: 0.88,
              voiceKey: "drums:hat_open",
            });
          } else {
            this.synth.playHat(when, 0.90, true, dInput, dTerr);
          }
        } else if (step % 2 === 1 && sec !== "INTRO" && (dSteer.y > 0.25 || dTerr === "cyber")) {
          const hatSlice = this.getBestSlice("drums:hat_closed", undefined, dSteer.y, dTerr);
          const hatVel = (step % 4 === 1 || step % 4 === 3) ? 0.72 : 0.58;
          if (hatSlice) {
            this.engine.scheduleSlice("drums", hatSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.0,
              gain: hatVel,
              voiceKey: "drums:hat_closed",
              chokeKeys: ["drums:hat_open"],
            });
          } else {
            this.synth.playHat(when, hatVel, false, dInput, dTerr);
          }
        }

        // Syncopated Percussion
        if ([3, 7, 11].includes(step % 16) && (sec === "DROP" || sec === "GROOVE_A" || dSteer.y > 0.60 || dTerr === "cyber")) {
          const percSlice = this.getBestSlice("drums:percussion", undefined, dSteer.y, dTerr);
          if (percSlice) {
            this.engine.scheduleSlice("drums", percSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: 0.75,
              voiceKey: "drums:perc",
            });
          } else {
            const percHz = dTerr === "cyber" ? 820.0 : 420.0;
            this.synth.playPerc(when, 0.70, percHz, dInput);
          }
        }

        // Dramatic Turnaround Fill on Bar 7
        if (phraseBar === 7 && step === 14) {
          const fillSlice = this.getBestSlice("drums:fill", undefined, dSteer.y, dTerr);
          if (fillSlice) {
            this.engine.scheduleSlice("drums", fillSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 2.0,
              gain: 1.0,
              voiceKey: "drums:fill",
            });
          }
        }
      }
    }

    // -------------------------------------------------------------------------
    // 2. BASS (Steered by Reticule X: Sub/Rolling/Stab & Style Territory)
    // -------------------------------------------------------------------------
    const bPat = this.activePatterns.bass;
    const bSteer = this.steering.bass;
    if (bPat && sec !== "BREAKDOWN" && !(sec === "BUILDUP" && barInSec >= 6)) {
      const bassCat = bSteer.x < 0.38 ? "bass:sub" : (bSteer.x < 0.72 ? "bass:rolling" : "bass:stab");
      const isSparseBass = (bSteer.y < 0.35 && bTerr !== "cyber");
      const isDenseBass = (bSteer.y > 0.70 || bTerr === "cyber");

      const shouldPlayBass = isSparseBass
        ? (step % 16 === 0 || step % 16 === 8)
        : (isDenseBass
          ? ([0, 2, 3, 5, 6, 8, 10, 11, 13, 14].includes(step % 16) || bPat.onset_grid.includes(step))
          : bPat.onset_grid.includes(step));

      if (shouldPlayBass) {
        let vel = bPat.velocities ? (bPat.velocities[step] || 0.90) : 0.90;
        let dur = (bassCat === "bass:rolling") ? 0.14 : (bassCat === "bass:sub" ? 0.28 : 0.20);
        if (bTerr === "wave") dur = 0.38; // Spacious 808 sub
        else if (bTerr === "cyber") dur = 0.12; // Snappy acid/reese rolling

        const subSlice = this.getBestSlice(bassCat, chord.rootPitch, bSteer.y, bTerr);
        if (subSlice) {
          this.engine.scheduleSlice("bass", subSlice.buffer, when, {
            rate: subSlice.rate,
            fadeMs: 4,
            gain: vel,
            voiceKey: "bass",
          });
        } else {
          this.synth.playSub(when, vel, chord.bassHz, dur, bDest, bTerr);
        }
      }
    }

    // -------------------------------------------------------------------------
    // 3. SYNTHS / OTHER (Steered by Reticule X & Style Territory)
    // -------------------------------------------------------------------------
    const oPat = this.activePatterns.other;
    const oSteer = this.steering.other;
    if (oPat) {
      const otherCat = oSteer.x < 0.40 ? "other:pad" : (oSteer.x < 0.75 ? "other:chord" : "other:arp");
      const isArpMode = (otherCat === "other:arp" || sec === "DROP" || oSteer.y > 0.72 || oTerr === "cyber");
      const isSparsePad = ((otherCat === "other:pad" || oSteer.y < 0.35) && oTerr !== "cyber" && oTerr !== "breaks");

      const shouldPlayOther = isSparsePad
        ? (step % 16 === 0 || step % 16 === 8)
        : (isArpMode ? (step % 2 === 0 || oPat.onset_grid.includes(step)) : oPat.onset_grid.includes(step));

      if (shouldPlayOther) {
        const vel = oPat.velocities ? (oPat.velocities[step] || 0.85) : 0.85;

        if (isArpMode) {
          const arpSlice = this.getBestSlice("other:arp", chord.rootPitch, oSteer.y, oTerr);
          if (arpSlice) {
            this.engine.scheduleSlice("other", arpSlice.buffer, when, {
              rate: arpSlice.rate,
              fadeMs: 4,
              gain: vel,
              voiceKey: "other:arp",
            });
          } else {
            const thirdMult = chord.isMinor ? Math.pow(2, 3 / 12) : Math.pow(2, 4 / 12);
            const fifthMult = Math.pow(2, 7 / 12);
            const octMult = 2.0;
            const ninthMult = 2.0 * Math.pow(2, 2 / 12);

            let arpPool = [chord.chordHz, chord.chordHz * thirdMult, chord.chordHz * fifthMult, chord.chordHz * octMult];
            if (phraseBar === 2 || phraseBar === 3) {
              arpPool = [chord.chordHz * thirdMult, chord.chordHz * fifthMult, chord.chordHz * ninthMult, chord.chordHz * octMult];
            } else if (phraseBar === 4 || phraseBar === 5) {
              arpPool = [chord.chordHz * fifthMult, chord.chordHz * octMult, chord.chordHz * octMult * thirdMult, chord.chordHz * ninthMult];
            } else if (phraseBar >= 6) {
              arpPool = [chord.chordHz * octMult, chord.chordHz * fifthMult, chord.chordHz * thirdMult, chord.chordHz];
            }

            const arpFreq = arpPool[step % arpPool.length];
            this.synth.playArp(when, vel, arpFreq, oDest, oTerr);
          }
        } else {
          const dur = (sec === "BREAKDOWN" || sec === "INTRO" || isSparsePad || oTerr === "wave") ? 0.45 : 0.28;
          const targetSliceCat = isSparsePad ? "other:pad" : "other:chord";
          const chordSlice = this.getBestSlice(targetSliceCat, chord.rootPitch, oSteer.y, oTerr) ||
                             this.getBestSlice("other:chord", chord.rootPitch, oSteer.y, oTerr);
          if (chordSlice) {
            this.engine.scheduleSlice("other", chordSlice.buffer, when, {
              rate: chordSlice.rate,
              fadeMs: 4,
              gain: vel,
              voiceKey: "other:chord",
            });
          } else {
            this.synth.playChord(when, vel, chord.chordHz, chord.isMinor, dur, oDest, oTerr);
          }
        }
      }
    }

    // -------------------------------------------------------------------------
    // 4. VOCALS (Multi-Persona Singing Engine & Style Territory)
    // -------------------------------------------------------------------------
    const vPat = this.activePatterns.vocals;
    const vSteer = this.steering.vocals;
    const isDropGap = (sec === "BUILDUP" && phraseBar === 7 && (step % 16) === 15);

    if (!isDropGap && vPat) {
      const vocalCat = vSteer.x < 0.38 ? "vocals:call" : (vSteer.x < 0.72 ? "vocals:hook" : "vocals:chop");
      const isDenseChop = (vocalCat === "vocals:chop" || vSteer.y > 0.72 || vTerr === "breaks");
      const isSparseVocal = (vSteer.y < 0.35 && vTerr !== "breaks" && vTerr !== "cyber");

      const syllable = this.vocalSongArranger.getSyllableForStep(sec, phraseBar, step % 16, vTerr);
      const shouldPlayVocal = isSparseVocal
        ? ((step % 16 === 0 || step % 16 === 8) && (phraseBar % 2 === 0))
        : (isDenseChop ? ((step % 4 === 0 || step % 4 === 2) || syllable !== undefined) : (syllable || vPat.onset_grid.includes(step % 16)));

      if (shouldPlayVocal) {
        const syl: VocalSyllable = syllable || {
          text: (sec === "DROP") ? (vTerr === "cyber" ? "PULSE" : (vTerr === "breaks" ? "DROP" : "FEEL")) : "YEAH",
          vowel: (vTerr === "cyber" ? "EE" : (vTerr === "breaks" ? "OH" : "EE")),
          semitoneOffset: (step % 4 === 0) ? 12 : 7,
          step: step % 16,
          durationSteps: (sec === "BREAKDOWN" || sec === "INTRO") ? 4 : 2,
          velocity: vPat.velocities ? (vPat.velocities[step % 16] || 0.90) : 0.90,
        };

        const vel = syl.velocity;
        const melodyPitch = (chord.rootPitch + syl.semitoneOffset) % 12;
        const noteName = `${ROOT_NAMES[melodyPitch]}4`;
        const noteHz = 440 * Math.pow(2, (chord.rootPitch + syl.semitoneOffset - 9 - 12) / 12);
        const durSec = (syl.durationSteps * 60) / (this.currentBpm * 4);

        const activeTitle = this.vocalSongArranger.getSongTitle(vTerr);
        this.currentVocalMeta = {
          lyric: this.vocalSongArranger.getCurrentLyricLine(sec, phraseBar, vTerr),
          syllable: syl.text,
          vowel: syl.vowel,
          note: noteName,
          songTitle: activeTitle,
          sectionName: sec,
        };

        const slice = this.getBestSlice(vocalCat, melodyPitch, vSteer.y, vTerr) ||
                      this.getBestSlice("vocals:hook", melodyPitch, vSteer.y, vTerr) ||
                      this.getBestSlice("vocals:call", melodyPitch, vSteer.y, vTerr) ||
                      this.getBestSlice("vocals:chop", melodyPitch, vSteer.y, vTerr);

        if (slice) {
          this.engine.scheduleSlice("vocals", slice.buffer, when, {
            rate: slice.rate,
            fadeMs: 6,
            gain: vel,
            voiceKey: "vocals",
          });
        } else {
          this.synth.playSingingVoice(when, noteHz, syl.vowel, durSec, vel, vInput, vTerr);
        }
      }
    }
  }

  private scheduleBarLoops(when: number, sec: string, phraseBar: number, chord: ChordStep): void {
    const isBuildupDropBreath = (sec === "BUILDUP" && phraseBar === 7);
    const dropBreathDuration = isBuildupDropBreath ? (14.85 * 60) / (this.currentBpm * 4) : undefined;

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const loop = this.activeLoops[stem];
      if (!loop) continue;

      // Invariant section muting rules:
      if (sec === "BREAKDOWN" && (stem === "drums" || stem === "bass")) {
        continue;
      }
      if (sec === "INTRO" && stem === "bass") {
        continue;
      }
      if (sec === "BUILDUP" && stem === "bass" && phraseBar >= 6) {
        continue;
      }

      let rate = this.currentBpm / loop.meta.source_bpm;
      if (stem !== "drums") {
        let shift = (chord.rootPitch - loop.meta.dominant_pitch) % 12;
        if (shift > 6) shift -= 12;
        if (shift < -6) shift += 12;
        rate *= Math.pow(2, shift / 12);
      }

      this.engine.scheduleSlice(stem, loop.buffer, when, {
        rate,
        duration: dropBreathDuration,
        fadeMs: stem === "vocals" ? 8 : 4,
      });

      if (stem === "vocals") {
        this.currentVocalMeta = {
          lyric: `${loop.meta.track_title}`,
          songTitle: loop.meta.track_title,
          syllable: loop.meta.category.toUpperCase(),
          vowel: "EE",
          note: `${ROOT_NAMES[chord.rootPitch]}4`,
          sectionName: sec,
        };
      }
    }
  }

  private composeBarPatterns(phraseBar: number): void {
    const sec = this.arranger.getSection();
    const isTurnaround = (phraseBar === 7);
    const chord = this.activeProgression[phraseBar % this.activeProgression.length];

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      if (this.playbackMode === "phrase_blocks") {
        // 1. Try to select matching 1-bar studio loop from the authentic training corpus
        const loopMatch = this.getBestLoop(stem, sec, phraseBar, stem === "drums" ? undefined : chord.rootPitch);
        if (loopMatch) {
          this.activeLoops[stem] = loopMatch.loop;
          const grid = loopMatch.loop.meta.onset_grid;
          const onsetIndices: number[] = [];
          const vels: number[] = new Array(16).fill(0.0);
          for (let i = 0; i < 16; i++) {
            if (grid[i] === 1) {
              onsetIndices.push(i);
              vels[i] = 0.95;
            }
          }
          this.activePatterns[stem] = {
            id: loopMatch.loop.meta.id,
            name: loopMatch.loop.meta.name,
            groove_type: loopMatch.loop.meta.category,
            energy: sec === "DROP" ? 0.95 : (sec === "BUILDUP" ? 0.80 : 0.70),
            density: onsetIndices.length / 16,
            onset_grid: onsetIndices,
            velocities: vels,
            length: 16,
            key: loopMatch.loop.meta.key,
          };
          continue;
        }
      }

      this.activeLoops[stem] = undefined;
      const currG = this.activePatterns[stem]?.groove_type || "four_on_floor_hats";
      let nextG = isTurnaround && stem === "drums"
        ? "turnaround_fill"
        : this.phraseModel.sampleNextGroove(phraseBar, stem, currG);

      // Section-specific musical arrangement logic & Reticule Y steering
      const steerY = this.steering[stem].y;
      if (stem === "drums") {
        if (steerY < 0.35) {
          nextG = "minimal_downbeats";
        } else if (steerY > 0.72) {
          nextG = isTurnaround ? "turnaround_fill" : "syncopated_breaks";
        }
      } else if (stem === "bass") {
        if (steerY < 0.35) {
          nextG = "sub";
        } else if (steerY > 0.70) {
          nextG = "rolling";
        } else {
          nextG = "offbeat";
        }
      } else if (stem === "other") {
        if (steerY < 0.35) {
          nextG = "pad";
        } else if (steerY > 0.70) {
          nextG = "riff";
        } else {
          nextG = "chords";
        }
      } else if (stem === "vocals") {
        if (steerY < 0.35) {
          nextG = "vocal_sustained";
        } else if (steerY > 0.70) {
          nextG = "vocal_syncopated_chop";
        } else if (sec === "DROP") {
          nextG = (phraseBar === 7) ? "vocal_hook" : (phraseBar % 2 === 0 ? "vocal_hook" : "vocal_syncopated_chop");
        } else if (sec === "BUILDUP") {
          nextG = (phraseBar >= 6) ? "vocal_stutter" : "vocal_call_response";
        } else if (sec === "BREAKDOWN" || sec === "INTRO") {
          nextG = "vocal_sustained";
        } else if (sec === "GROOVE_A") {
          nextG = (phraseBar % 2 === 0) ? "vocal_verse" : "vocal_call_response";
        }
      }

      const stemTerr = this.getStemTerritory(stem);
      this.activePatterns[stem] = this.synthesizePattern(stem, nextG, stemTerr);
    }
  }

  private applyPatternMutation(stem: StemName, pat: SerializedPattern): SerializedPattern {
    const mutated = {
      ...pat,
      onset_grid: [...pat.onset_grid],
      velocities: pat.velocities ? [...pat.velocities] : new Array(16).fill(0.9),
    };
    const steerY = this.steering[stem].y;

    if (stem === "drums") {
      // High energy/Y: inject ghost kick on step 10
      if (steerY > 0.65 && !mutated.onset_grid.includes(10)) {
        mutated.onset_grid.push(10);
        mutated.velocities[10] = 0.72;
      }
      // Add ghost percussion on step 7
      if (steerY > 0.75 && !mutated.onset_grid.includes(7)) {
        mutated.onset_grid.push(7);
        mutated.velocities[7] = 0.65;
      }
      // Low Y: prune to minimal downbeats
      if (steerY < 0.35) {
        mutated.onset_grid = mutated.onset_grid.filter((s) => s === 0 || s === 8);
      }
    } else if (stem === "bass") {
      if (steerY > 0.70 && !mutated.onset_grid.includes(10)) {
        mutated.onset_grid.push(10);
        mutated.velocities[10] = 0.85;
      }
      if (steerY < 0.35) {
        mutated.onset_grid = mutated.onset_grid.filter((s) => s === 0 || s === 8);
      }
    }
    mutated.onset_grid.sort((a, b) => a - b);
    return mutated;
  }

  private synthesizePattern(stem: StemName, grooveType: string, territory?: StyleTerritory): SerializedPattern {
    const terr = territory ?? this.getStemTerritory(stem);

    // 1. Try to find matching authentic patterns in standalone manifest
    if (this.manifest && this.manifest.patterns && this.manifest.patterns[stem]) {
      const pats = this.manifest.patterns[stem];
      const matches = pats.filter((p) => {
        if (p.groove_type === `${grooveType}_${terr}` || p.groove_type === `${grooveType}:${terr}`) return true;
        if (p.groove_type === grooveType) return true;
        if (stem === "vocals") {
          if (grooveType === "vocal_hook" && (p.groove_type === "vocal_hook" || p.groove_type === "hook")) return true;
          if (grooveType === "vocal_verse" && (p.groove_type === "vocal_verse" || p.groove_type === "vocal_call_response" || p.groove_type === "phrase")) return true;
          if (grooveType === "vocal_call_response" && (p.groove_type === "vocal_call_response" || p.groove_type === "call")) return true;
          if (grooveType === "vocal_syncopated_chop" && (p.groove_type === "vocal_syncopated_chop" || p.groove_type === "chop")) return true;
          if (grooveType === "vocal_sustained" && (p.groove_type === "vocal_sustained" || p.groove_type === "phrase")) return true;
          if (grooveType === "vocal_stutter" && p.groove_type === "vocal_stutter") return true;
        } else if (stem === "drums") {
          if (grooveType === "turnaround_fill" && (p.groove_type === "turnaround_fill" || p.groove_type === "fill")) return true;
          if (grooveType === "mini_turnaround" && (p.groove_type === "mini_turnaround" || p.groove_type === "fill")) return true;
          if (grooveType === "four_on_floor_hats" && (p.groove_type === "four_on_floor_hats" || p.groove_type === "four_on_floor")) return true;
        } else if (stem === "bass") {
          if (grooveType === "offbeat" && (p.groove_type === "offbeat" || p.groove_type === "sub")) return true;
        } else if (stem === "other") {
          if (grooveType === "chords" && (p.groove_type === "chords" || p.groove_type === "pad")) return true;
        }
        return false;
      });

      if (matches.length > 0) {
        const v = Math.max(0, Math.min(0.999, this.steering[stem].y));
        const idx = Math.floor(v * matches.length) % matches.length;
        const pat = { ...matches[idx] };
        return this.applyPatternMutation(stem, pat);
      }
    }

    // 2. Synthesize dynamic groove pattern based on style territory
    const onset_grid: number[] = [];
    const velocities: number[] = new Array(16).fill(0.0);

    if (stem === "drums") {
      if (grooveType === "turnaround_fill") {
        for (const s of [0, 4, 8, 9, 10, 11, 12, 13, 14]) {
          onset_grid.push(s);
          velocities[s] = (s >= 8) ? 0.70 + (s - 8) * 0.05 : 1.0;
        }
        // Notice step 15 is explicitly SILENT for the drop breath
      } else if (grooveType === "mini_turnaround") {
        for (const s of [0, 2, 4, 6, 8, 10, 12, 13, 14]) {
          onset_grid.push(s);
          velocities[s] = (s >= 12) ? 0.92 : 0.85;
        }
      } else if (terr === "wave") {
        // Wave: Halftime / ambient dub kicks, spacious hats
        for (const s of [0, 8, 10]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 ? 1.0 : (s === 8 ? 0.92 : 0.65));
        }
        for (const s of [2, 6, 12, 14]) {
          if (!onset_grid.includes(s)) {
            onset_grid.push(s);
            velocities[s] = 0.60;
          }
        }
      } else if (terr === "cyber") {
        // Cyber: Phonk syncopated kicks on step 10 & 14, rapid rolling phonk hats
        for (const s of [0, 4, 8, 10, 14]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 4 || s === 8) ? 1.0 : 0.88;
        }
        for (const s of [2, 6, 12]) {
          if (!onset_grid.includes(s)) {
            onset_grid.push(s);
            velocities[s] = 0.75;
          }
        }
      } else if (terr === "breaks") {
        // Breaks: 2-step / syncopated breakbeat (kicks on 0, 6, 10, snares on 4, 12)
        for (const s of [0, 4, 6, 10, 12, 14, 15]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 4 || s === 12) ? 1.0 : (s === 6 || s === 10 ? 0.90 : 0.72);
        }
      } else {
        // House: Classic 4-on-the-floor, upbeat open hats
        for (let s = 0; s < 16; s++) {
          onset_grid.push(s);
          if (s % 4 === 0) velocities[s] = 1.0;       // Kicks
          else if (s % 2 === 0) velocities[s] = 0.88; // Open Hats
          else velocities[s] = 0.62;                  // 16th Closed Hats
        }
      }
    } else if (stem === "bass") {
      if (terr === "wave") {
        // Wave: Spacious 808 subs
        for (const s of [0, 6, 8]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 ? 1.0 : 0.85);
        }
      } else if (terr === "cyber") {
        // Cyber: Distorted rolling 16th bass with phonk acid syncopation
        for (let s = 0; s < 16; s++) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 3 || s === 6 || s === 10 || s === 12) ? 0.98 : 0.70;
        }
      } else if (terr === "breaks") {
        // Breaks: Rolling funk bass
        for (const s of [0, 3, 6, 8, 11, 14]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 8) ? 1.0 : 0.85;
        }
      } else {
        // House: Pumping offbeat bass on 2, 6, 10, 14
        for (const s of [2, 6, 10, 14, 3, 7, 11, 15]) {
          onset_grid.push(s);
          velocities[s] = (s % 4 === 2) ? 1.0 : 0.65;
        }
      }
    } else if (stem === "other") {
      if (terr === "wave") {
        // Wave: Dream pads / ambient chords
        for (const s of [0, 8]) {
          onset_grid.push(s);
          velocities[s] = 0.90;
        }
      } else if (terr === "cyber") {
        // Cyber: Phonk cowbell / rapid arp riffs
        for (const s of [0, 3, 6, 10, 12, 14]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 6 || s === 12) ? 1.0 : 0.85;
        }
      } else if (terr === "breaks") {
        // Breaks: Funk stabs & syncopated break riffs
        for (const s of [2, 5, 8, 11, 14]) {
          onset_grid.push(s);
          velocities[s] = 0.90;
        }
      } else {
        // House: Warm chord triads on upbeats 2, 6, 10, 14
        for (const s of [2, 6, 10, 14]) {
          onset_grid.push(s);
          velocities[s] = 0.90;
        }
      }
    } else {
      // vocals: multi-genre patterns
      if (terr === "wave") {
        for (const s of [0, 6, 10]) {
          onset_grid.push(s);
          velocities[s] = 0.92;
        }
      } else if (terr === "cyber") {
        for (const s of [0, 2, 6, 8, 10, 14]) {
          onset_grid.push(s);
          velocities[s] = 0.95;
        }
      } else if (terr === "breaks") {
        for (const s of [0, 2, 5, 8, 10, 12, 14]) {
          onset_grid.push(s);
          velocities[s] = 0.98;
        }
      } else {
        // house
        for (const s of [0, 3, 6, 8, 10, 12, 14]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 14) ? 1.0 : 0.92;
        }
      }
    }

    onset_grid.sort((a, b) => a - b);
    return {
      id: `pat_${stem}_${grooveType}_${terr}`,
      name: `${stem}_${grooveType}_${terr}`,
      groove_type: `${grooveType}_${terr}`,
      energy: 0.85,
      density: onset_grid.length / 16,
      onset_grid,
      velocities,
      length: 16,
      key: "8A",
    };
  }

  private initDefaultPatterns(): void {
    this.activePatterns.drums = this.synthesizePattern("drums", "four_on_floor_hats", this.getStemTerritory("drums"));
    this.activePatterns.bass = this.synthesizePattern("bass", "offbeat", this.getStemTerritory("bass"));
    this.activePatterns.other = this.synthesizePattern("other", "chords", this.getStemTerritory("other"));
    this.activePatterns.vocals = this.synthesizePattern("vocals", "vocal_verse", this.getStemTerritory("vocals"));
  }

  private buildMeta(stepInPattern: number, patternInCycle: number, phraseBar: number): any {
    const sec = this.arranger.getSection();
    const isFill = (phraseBar === 7);
    const chord = this.activeProgression[phraseBar % this.activeProgression.length];

    const dTerr = this.getStemTerritory("drums");
    const vTerr = this.getStemTerritory("vocals");
    const activePersona = this.vocalSongArranger.getPersonaName(vTerr);

    const patterns: Record<string, string> = {};
    const patternGrooves: Record<string, string> = {};
    const patternGrids: Record<string, number[]> = {};
    const patternAudioScores: Record<string, number> = {};

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const p = this.activePatterns[stem];
      if (p) {
        patterns[stem] = p.name ? `${p.name}` : `KG-505 [${chord.name}] (${p.groove_type})`;
        patternGrooves[stem] = p.groove_type;
        const grid = new Array(this.stepsPerPattern).fill(0);
        for (const idx of p.onset_grid) {
          if (idx < this.stepsPerPattern) grid[idx] = 1;
        }
        patternGrids[stem] = grid;
        patternAudioScores[stem] = 0.88;
      }
    }

    const stepInBar = (stepInPattern % 16);
    return {
      tatum_index: this.tatumIndex,
      tatumIndex: this.tatumIndex,
      section: sec,
      territory: dTerr,
      personaName: activePersona,
      territories: {
        drums: this.getStemTerritory("drums"),
        bass: this.getStemTerritory("bass"),
        other: this.getStemTerritory("other"),
        vocals: this.getStemTerritory("vocals"),
      },
      phrase_bar: phraseBar + 1,
      phraseBar: phraseBar + 1,
      step_in_bar: stepInBar + 1,
      stepInBar: stepInBar + 1,
      total_bars: this.arranger.getTotalBars(),
      totalBars: this.arranger.getTotalBars(),
      section_bars_elapsed: this.arranger.getBarInSection(),
      sectionBarsElapsed: this.arranger.getBarInSection(),
      section_total_bars: this.arranger.getSectionDuration(),
      sectionTotalBars: this.arranger.getSectionDuration(),
      section_bars_left: this.arranger.getSectionBarsLeft(),
      sectionBarsLeft: this.arranger.getSectionBarsLeft(),
      next_section: sec === "BUILDUP" ? "DROP" : (sec === "DROP" ? "GROOVE_A" : "BUILDUP"),
      nextSection: sec === "BUILDUP" ? "DROP" : (sec === "DROP" ? "GROOVE_A" : "BUILDUP"),
      is_fill: isFill,
      isFill: isFill,
      chord_name: chord.name,
      chordName: chord.name,
      camelot_code: this.currentCamelotCode,
      camelotCode: this.currentCamelotCode,
      cycle_mode: this.cycleMode,
      cycleMode: this.cycleMode,
      pattern_in_cycle: patternInCycle + 1,
      patternInCycle: patternInCycle + 1,
      patterns_per_cycle: this.patternsPerCycle,
      patternsPerCycle: this.patternsPerCycle,
      step_in_pattern: stepInPattern + 1,
      stepInPattern: stepInPattern + 1,
      steps_per_pattern: this.stepsPerPattern,
      stepsPerPattern: this.stepsPerPattern,
      cycle_total_steps: this.patternsPerCycle * this.stepsPerPattern,
      cycleTotalSteps: this.patternsPerCycle * this.stepsPerPattern,
      cycle_progress: (patternInCycle * this.stepsPerPattern + stepInPattern) / (this.patternsPerCycle * this.stepsPerPattern),
      cycleProgress: (patternInCycle * this.stepsPerPattern + stepInPattern) / (this.patternsPerCycle * this.stepsPerPattern),
      patterns,
      pattern_grooves: patternGrooves,
      patternGrooves,
      pattern_grids: patternGrids,
      patternGrids,
      pattern_ratings: this.stemRatings,
      patternRatings: this.stemRatings,
      pattern_audio_scores: patternAudioScores,
      patternAudioScores,
      vocal_lyric: this.currentVocalMeta.lyric,
      vocalLyric: this.currentVocalMeta.lyric,
      vocal_song_title: this.currentVocalMeta.songTitle,
      vocalSongTitle: this.currentVocalMeta.songTitle,
      vocal_syllable: this.currentVocalMeta.syllable,
      vocalSyllable: this.currentVocalMeta.syllable,
      vocal_vowel: this.currentVocalMeta.vowel,
      vocalVowel: this.currentVocalMeta.vowel,
      vocal_note: this.currentVocalMeta.note,
      vocalNote: this.currentVocalMeta.note,
      vocal_persona: activePersona,
      vocalPersona: activePersona,
      direct_ml_engine: true,
    };
  }
}
