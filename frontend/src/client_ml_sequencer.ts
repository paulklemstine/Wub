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
  cohesion_score?: number;
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
  vowel?: string;
  track_title?: string;
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
  vowel?: string;
  trackTitle?: string;
  trackId?: string;
  segmentId?: string;
  x?: number;
  y?: number;
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
  patterns_by_style?: Record<string, Record<string, SerializedPattern[]>>;
  phrases: SerializedPhrase[];
  slices: Record<StemName, Record<string, SliceManifestItem[]>>;
  loops?: Record<StemName, LoopManifestItem[]>;
  phrase_blocks?: CohesivePhraseManifestItem[];
  vocal_themes?: VocalSongTheme[];
  vocal_patterns?: Record<string, any>;
  style_kits?: Record<string, any>;
  stem_cohesion_weights?: StemCohesionWeights;
  song_composition_scores?: Record<string, any>;
}

export interface StemCohesionWeights {
  stem_pair_compatibility: {
    drums_bass?: Record<string, Record<string, number>>;
    other_vocals?: Record<string, Record<string, number>>;
    drums_vocals?: Record<string, Record<string, number>>;
    bass_other?: Record<string, Record<string, number>>;
  };
  frequency_masking_targets?: {
    kick_bass_low_cutoff_hz?: number;
    vocal_presence_band_hz?: [number, number];
    synth_ducking_on_vocal_db?: number;
    bass_sidechain_ducking_db?: number;
  };
  grooves?: Record<string, Record<string, number>>;
  trained_slice_scores?: Record<string, number>;
  song_composition_scores?: Record<string, any>;
  stem_mix_automation?: Record<string, any>;
}

export interface ComposingMLWeights {
  version: number;
  phrase_transitions: Record<string, Record<string, number>>;
  section_transitions: Record<string, Record<string, number>>;
  section_energy_targets: Record<string, number>;
  stem_cohesion_weights?: StemCohesionWeights;
  song_composition_scores?: Record<string, any>;
}

export interface CuratedSongKit {
  territory: StyleTerritory;
  keyRoot: number;
  kick: { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null;
  snare: { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null;
  hatClosed: { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null;
  hatOpen: { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null;
  percussion: { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null;
  fill: { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null;
  bassSubAsset: SliceAsset | null;
  bassRollingAsset: SliceAsset | null;
  chordAsset: SliceAsset | null;
  padAsset: SliceAsset | null;
  arpAsset: SliceAsset | null;
  vocalHookAsset: SliceAsset | null;
  vocalCallAsset: SliceAsset | null;
  vocalChopAsset: SliceAsset | null;
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

  public addThemes(newThemes: VocalSongTheme[]): void {
    if (!newThemes || newThemes.length === 0) return;
    for (const nt of newThemes) {
      const idx = this.themes.findIndex((t) => t.territory === nt.territory && t.title === nt.title);
      if (idx >= 0) {
        this.themes[idx] = nt;
      } else {
        this.themes.push(nt);
      }
    }
  }

  public setThemes(newThemes: VocalSongTheme[]): void {
    if (newThemes && newThemes.length > 0) {
      this.themes = newThemes;
      this.currentThemeIndex = 0;
    }
  }

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

  public randomizeSong(): void {
    if (this.themes.length > 0) {
      this.currentThemeIndex = Math.floor(Math.random() * this.themes.length);
    }
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
    let q1 = 3.2;
    let q2 = 3.6;
    let q3 = 2.8;
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

    // Vocal Presence Filter: Crisp vocal intelligibility and brightness
    const presenceFilter = this.ctx.createBiquadFilter();
    presenceFilter.type = "peaking";
    presenceFilter.frequency.setValueAtTime(3400, t);
    presenceFilter.Q.setValueAtTime(1.2, t);
    presenceFilter.gain.setValueAtTime(3.5, t); // +3.5 dB air & presence

    // Vocal Master Envelope (Soft attack, warm sustain, natural release)
    const masterGain = this.ctx.createGain();
    const v = Math.max(0.1, Math.min(1.0, velocity));

    masterGain.gain.setValueAtTime(0.0001, t);
    masterGain.gain.linearRampToValueAtTime(v * 1.45, t + attackTime);
    masterGain.gain.setValueAtTime(v * 1.35, t + holdTime);
    masterGain.gain.exponentialRampToValueAtTime(0.0001, t + durationSec + 0.07);

    vocalSum.connect(presenceFilter);
    presenceFilter.connect(masterGain);
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

  // Shimmering Stereo Crash Cymbal Splash for Drop Step 0 & Climaxes
  public playCrashCymbal(when: number, velocity: number = 1.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;
    const v = Math.max(0.1, Math.min(1.0, velocity));

    if (this.noiseBuffer) {
      const noise = this.ctx.createBufferSource();
      noise.buffer = this.noiseBuffer;

      const hp = this.ctx.createBiquadFilter();
      hp.type = "highpass";
      hp.frequency.setValueAtTime(4500, t);
      hp.frequency.exponentialRampToValueAtTime(7000, t + 1.8);

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(v * 0.95, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 2.2);

      noise.connect(hp);
      hp.connect(gain);
      gain.connect(dest);

      noise.start(t);
      noise.stop(t + 2.3);
    }

    // Metallic square overtone cluster for acoustic crash brightness
    const freqs = [587, 845, 1200, 3100];
    for (const f of freqs) {
      const osc = this.ctx.createOscillator();
      const oscGain = this.ctx.createGain();
      osc.type = "square";
      osc.frequency.setValueAtTime(f, t);

      oscGain.gain.setValueAtTime(v * 0.12, t);
      oscGain.gain.exponentialRampToValueAtTime(0.001, t + 1.2);

      osc.connect(oscGain);
      oscGain.connect(dest);
      osc.start(t);
      osc.stop(t + 1.25);
    }
  }

  // Physical 30Hz Sub Impact Boom on Drop Landings
  public playSubDropImpact(when: number, velocity: number = 1.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(75, t);
    osc.frequency.exponentialRampToValueAtTime(28, t + 0.55);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 1.35, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.70);

    osc.connect(gain);
    gain.connect(dest);
    osc.start(t);
    osc.stop(t + 0.75);
  }

  // Reverse Cymbal Swell into Downbeats & Drop Steps
  public playReverseCymbalSwell(when: number, durationSec: number = 1.2, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    if (this.noiseBuffer) {
      const noise = this.ctx.createBufferSource();
      noise.buffer = this.noiseBuffer;

      const bp = this.ctx.createBiquadFilter();
      bp.type = "bandpass";
      bp.frequency.setValueAtTime(1200, t);
      bp.frequency.exponentialRampToValueAtTime(8000, t + durationSec);
      bp.Q.value = 2.0;

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.exponentialRampToValueAtTime(0.85, t + durationSec);
      gain.gain.setValueAtTime(0.0001, t + durationSec + 0.02);

      noise.connect(bp);
      bp.connect(gain);
      gain.connect(dest);

      noise.start(t);
      noise.stop(t + durationSec + 0.03);
    }
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
  private sectionDuration: number = 48;
  private sectionEnergyTargets: Record<string, number> = {
    INTRO: 0.25,
    BUILDUP: 0.65,
    DROP: 0.90,
    GROOVE_A: 0.83,
    BREAKDOWN: 0.20,
  };

  private sectionTransitions: Record<string, Record<string, number>> = {
    INTRO: { GROOVE_A: 7.0, BUILDUP: 3.0 },
    BUILDUP: { DROP: 10.0 },
    DROP: { GROOVE_A: 5.0, BREAKDOWN: 5.0 },
    GROOVE_A: { BUILDUP: 6.5, DROP: 2.5, BREAKDOWN: 1.0 },
    BREAKDOWN: { BUILDUP: 8.5, GROOVE_A: 1.5 },
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
    // Section-Wide Groove Persistence:
    // Dance movements hold the feel for 48 to 64 bars (768 to 1024 tatums),
    // giving listeners an extended 2-3 minute steady, hypnotic groove that breathes and evolves
    if (nextSec === "GROOVE_A" || nextSec === "DROP") {
      this.sectionDuration = Math.random() < 0.5 ? 48 : 64; // 768 or 1024 tatums (2 to 3 minutes of steady hypnotic groove)
    } else if (nextSec === "INTRO") {
      this.sectionDuration = Math.random() < 0.5 ? 16 : 32; // 256 or 512 tatums
    } else if (nextSec === "BREAKDOWN") {
      this.sectionDuration = Math.random() < 0.5 ? 16 : 32; // 256 or 512 tatums
    } else {
      this.sectionDuration = 16; // BUILDUP is 16 bars (256 tatums) for a true gradual buildup arc!
    }
  }

  public setSection(section: string): void {
    this.currentSection = section;
    this.barInSection = 0;
    if (section === "GROOVE_A" || section === "DROP") {
      this.sectionDuration = 48;
    } else if (section === "INTRO" || section === "BREAKDOWN") {
      this.sectionDuration = 24;
    } else {
      this.sectionDuration = 16;
    }
  }

  public resetToSection(section: string): void {
    this.currentSection = section;
    this.barInSection = 0;
    if (section === "GROOVE_A" || section === "DROP") {
      this.sectionDuration = 48;
    } else if (section === "INTRO" || section === "BREAKDOWN") {
      this.sectionDuration = 24;
    } else {
      this.sectionDuration = 16;
    }
  }

  public randomizeStartingSection(): string {
    const choices = ["GROOVE_A", "GROOVE_A", "DROP"];
    const pick = choices[Math.floor(Math.random() * choices.length)];
    this.resetToSection(pick);
    return pick;
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
  private stemCohesionWeights: StemCohesionWeights | null = null;

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
    drums: { x: 0.35, y: 0.75 },
    bass: { x: 0.35, y: 0.75 },
    vocals: { x: 0.35, y: 0.72 },
    other: { x: 0.35, y: 0.72 },
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

  public playbackMode: "jukebox" | "slices" | "phrase_blocks" = "jukebox";
  private stemMixLevels: Record<StemName, number> = { drums: 1.0, bass: 1.0, other: 1.0, vocals: 1.0 };
  private sliceRegistry: Map<string, SliceAsset[]> = new Map();
  private segmentPositions: Map<string, { x: number; y: number }> = new Map();
  private trackStemPositions: Map<string, { x: number; y: number }> = new Map();
  private loopRegistry: Map<StemName, LoadedLoop[]> = new Map();
  private activeLoops: Partial<Record<StemName, LoadedLoop>> = {};
  private phraseBlockRegistry: Map<string, LoadedPhraseBlock[]> = new Map();
  private activePhraseBlock: LoadedPhraseBlock | null = null;
  private activeTrackId?: string;
  private pendingTrackId?: string;
  private phraseBarCount: number = 0;
  private totalBarsPlayed: number = 0;
  private recentLoops: Record<StemName, string[]> = {
    drums: [],
    bass: [],
    vocals: [],
    other: [],
  };
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
  private activeSongKit: CuratedSongKit | null = null;
  private kitKeyRoot: number = -1;
  private kitTerritory: StyleTerritory | null = null;
  private recentPhraseBlocks: string[] = [];

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

    // Pure acoustic slices: keep filters transparently wide open at 20 kHz for maximum punch, clarity and grit
    this.engine.setStemFilter(stem, 20000, 0.7);

    // Dynamic Reticule Slice Steering: immediately update the set of slices used by this stem's pattern!
    this.updateStemSlicesForReticle(stem, { x: cx, y: cy });
  }

  public setTrackStemPositions(positions: Map<string, { x: number; y: number }>): void {
    this.trackStemPositions = positions;
  }

  public setSegmentPositions(positions: Map<string, { x: number; y: number }>): void {
    this.segmentPositions = positions;
    for (const [, assets] of this.sliceRegistry) {
      for (const asset of assets) {
        if (asset.segmentId) {
          let pos = positions.get(asset.segmentId) ?? (asset.trackId ? positions.get(`${asset.trackId}:${asset.segmentId}`) : undefined);
          if (!pos && asset.trackId) {
            for (const [key, tPos] of this.trackStemPositions) {
              if (key.startsWith(asset.trackId)) {
                pos = tPos;
                break;
              }
            }
          }
          if (pos) {
            asset.x = pos.x;
            asset.y = pos.y;
          }
        }
      }
    }
    // Re-prime kit slices with updated UMAP map positions
    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      this.updateStemSlicesForReticle(stem, this.steering[stem]);
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
        if (weights.stem_cohesion_weights) {
          this.stemCohesionWeights = weights.stem_cohesion_weights;
          console.info("[Client ML] Locked in multi-stem audio feedback & cohesion weights from training.");
        }
        console.info("[Client ML] Loaded pre-trained composing Markov weights into in-browser engine.");
      }

      // 2. Fetch Standalone Patterns & Slices Manifest
      const pRes = await fetch(`${basePath}/standalone_patterns.json`);
      if (pRes.ok) {
        this.manifest = (await pRes.json()) as StandaloneManifest;
        if (this.manifest.stem_cohesion_weights && !this.stemCohesionWeights) {
          this.stemCohesionWeights = this.manifest.stem_cohesion_weights;
          console.info("[Client ML] Locked in multi-stem audio cohesion weights from manifest.");
        }
        console.info(
          `[Client ML] Loaded standalone pattern bank (${Object.keys(this.manifest.patterns).length} stems, ${this.manifest.phrases.length} phrases).`
        );
        // Load corpus-learned chord progressions if available
        if (this.manifest.chord_progressions && this.manifest.chord_progressions.length > 0) {
          const randProg = Math.floor(Math.random() * this.manifest.chord_progressions.length);
          this.progressionIndex = randProg;
          this.loadProgressionFromManifest(randProg);
        }
        // Load corpus-trained dynamic vocal song themes if available
        const vThemes = this.manifest.vocal_themes || (this.manifest.vocal_patterns?.themes as VocalSongTheme[]);
        if (vThemes && vThemes.length > 0) {
          this.vocalSongArranger.addThemes(vThemes);
          console.info(`[Client ML] Loaded ${vThemes.length} corpus-trained vocal themes.`);
        }
        // Initialize default active patterns with the nearest real patterns from the authentic manifest
        this.initDefaultPatterns();
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
        // Big Codebook: Load up to 128 slices per stem category from the authentic training corpus
        const maxSlices = 128;
        for (let i = 0; i < Math.min(maxSlices, list.length); i++) {
          const item = list[i];
          try {
            let bytes: ArrayBuffer | null = null;
            try {
              const url = item.url.startsWith("http") ? item.url : `${basePath}/${item.url.replace(/^data\//, "")}`;
              const res = await fetch(url);
              if (res.ok) {
                bytes = await res.arrayBuffer();
              }
            } catch {
              // Fetch failed, try b64 fallback below
            }
            if (!bytes && item.b64) {
              try {
                const base64Data = item.b64.split(",")[1] || item.b64;
                const binaryStr = atob(base64Data);
                const len = binaryStr.length;
                const u8 = new Uint8Array(len);
                for (let b = 0; b < len; b++) {
                  u8[b] = binaryStr.charCodeAt(b);
                }
                bytes = u8.buffer;
              } catch {
                // Ignore base64 decode failure
              }
            }
            if (bytes) {
              const decoded = await this.engine.decodeWav(bytes);
              let segPos = item.segment_id
                ? (this.segmentPositions.get(item.segment_id) ?? (item.track_id ? this.segmentPositions.get(`${item.track_id}:${item.segment_id}`) : undefined))
                : undefined;
              if (!segPos && item.track_id) {
                for (const [key, tPos] of this.trackStemPositions) {
                  if (key.startsWith(item.track_id)) {
                    segPos = tPos;
                    break;
                  }
                }
              }
              let x = segPos?.x;
              let y = segPos?.y;
              if (x === undefined || y === undefined) {
                const normCentroid = Math.max(0, Math.min(1, Math.log2(Math.max(50, item.centroid || 500) / 50) / 7.6));
                const isRight = cat.includes("cyber") || cat.includes("breaks");
                const isBottom = cat.includes("house") || cat.includes("breaks");
                x = (isRight ? 0.52 : 0.04) + 0.44 * normCentroid;
                y = (isBottom ? 0.52 : 0.04) + 0.44 * Math.max(0, Math.min(1, (item.duration || 0.2) / 0.8));
              }
              const asset: SliceAsset = {
                buffer: decoded,
                pitch: typeof item.pitch === "number" ? item.pitch : 0,
                centroid: typeof item.centroid === "number" ? item.centroid : 0,
                duration: typeof item.duration === "number" ? item.duration : decoded.duration,
                loudness: typeof item.loudness === "number" ? item.loudness : 1.0,
                name: item.name,
                vowel: item.vowel,
                trackTitle: item.track_title,
                trackId: item.track_id,
                segmentId: item.segment_id,
                x,
                y,
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
          // Also alias underscore style keys (e.g. 'bass:sub_wave' <-> 'bass:sub:wave')
          if (cat.includes("_")) {
            this.sliceRegistry.set(`${stem}:${cat.replace("_", ":")}`, assets);
          }
        }
      }
    }
    // Initialize default curated song kit with preloaded slices
    this.initSongKit("house", 0);
    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      this.updateStemSlicesForReticle(stem, this.steering[stem]);
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
    const blocks = this.manifest.phrase_blocks;

    const loadBlock = async (item: CohesivePhraseManifestItem): Promise<void> => {
      const buffers: Partial<Record<StemName, AudioBuffer>> = {};
      const stemEntries = Object.entries(item.stems) as Array<[StemName, string]>;
      await Promise.all(
        stemEntries.map(async ([stem, urlStr]) => {
          if (!urlStr) return;
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
        })
      );
      if (Object.keys(buffers).length > 0) {
        const sec = item.section;
        const list = this.phraseBlockRegistry.get(sec) || [];
        list.push({ meta: item, buffers });
        this.phraseBlockRegistry.set(sec, list);
      }
    };

    // Priority Tier: Load at least 1 block for EACH section (GROOVE_A, DROP, BUILDUP, INTRO, BREAKDOWN)
    const sectionsCovered = new Set<string>();
    const priorityBlocks: CohesivePhraseManifestItem[] = [];
    for (const b of blocks) {
      if (!sectionsCovered.has(b.section)) {
        sectionsCovered.add(b.section);
        priorityBlocks.push(b);
      }
    }
    // Fill up to 8 priority blocks
    for (const b of blocks) {
      if (priorityBlocks.length >= 8) break;
      if (!priorityBlocks.includes(b)) priorityBlocks.push(b);
    }

    await Promise.all(priorityBlocks.map(loadBlock));
    console.info(`[Client ML] Ready: ${priorityBlocks.length} priority studio 4-bar phrase blocks loaded in parallel across sections.`);

    // Background Tier: Stream the rest asynchronously
    const remainingBlocks = blocks.filter((b) => !priorityBlocks.includes(b));
    for (const item of remainingBlocks) {
      await loadBlock(item);
    }
    const totalBlocks = Array.from(this.phraseBlockRegistry.values()).reduce((sum, l) => sum + l.length, 0);
    console.info(`[Client ML] Preloaded ${totalBlocks} studio 4-bar cohesive multi-stem phrase blocks across ${this.phraseBlockRegistry.size} sections.`);
  }

  public getBestPhraseBlock(section: string, preferredTrackId?: string): LoadedPhraseBlock | null {
    let list = this.phraseBlockRegistry.get(section);
    if (!list || list.length === 0) {
      for (const fallbackSec of ["GROOVE_A", "DROP", "BUILDUP", "BREAKDOWN", "INTRO"]) {
        const fbList = this.phraseBlockRegistry.get(fallbackSec);
        if (fbList && fbList.length > 0) {
          list = fbList;
          break;
        }
      }
    }
    if (!list || list.length === 0) return null;

    // 1. Same-Track Cohesive Match: Prioritize blocks from the same authentic track
    const targetTrack = preferredTrackId || this.activeTrackId;
    if (targetTrack) {
      // First check exact section match
      const trackMatches = list.filter((p) => p.meta.track_id === targetTrack);
      if (trackMatches.length > 0) {
        return trackMatches[Math.floor(Math.random() * trackMatches.length)];
      }

      // Next check other sections from the SAME authentic track session:
      // A song's GROOVE_A or DROP block can easily play BUILDUP (with snare acceleration & bass cutoff),
      // or INTRO/BREAKDOWN (with bass & drum muting). This preserves 100% of the song's sonic DNA!
      for (const altSec of ["DROP", "GROOVE_A", "BUILDUP", "BREAKDOWN", "INTRO"]) {
        const altList = this.phraseBlockRegistry.get(altSec);
        if (altList) {
          const altMatches = altList.filter((p) => p.meta.track_id === targetTrack);
          if (altMatches.length > 0) {
            console.info(`[Client ML] Song Continuity: Kept active track ${altMatches[0].meta.track_title} for ${section} (adapted from ${altSec})`);
            return altMatches[0];
          }
        }
      }
    }

    // 2. Harmonic Affinity: Prefer phrase blocks in the active Camelot key or adjacent Camelot keys
    const activeKey = CAMELOT_KEYS[this.currentCamelotCode]?.name?.toLowerCase();
    let candidates = list;
    if (activeKey) {
      const harmonicMatches = list.filter((p) => p.meta.key.toLowerCase().includes(activeKey));
      if (harmonicMatches.length > 0) {
        candidates = harmonicMatches;
      }
    }

    // 3. Tempo Affinity: Prefer phrase blocks within +/- 8% of current BPM to avoid pitch warping
    const tempoMatches = candidates.filter(
      (p) => Math.abs(p.meta.source_bpm - this.currentBpm) / this.currentBpm <= 0.08
    );
    if (tempoMatches.length > 0) {
      candidates = tempoMatches;
    }

    // High variety across different tracks: avoid recently played phrase blocks
    const freshCandidates = candidates.filter((p) => !this.recentPhraseBlocks.includes(p.meta.id));
    const pool = freshCandidates.length > 0 ? freshCandidates : candidates;
    const pick = pool[Math.floor(Math.random() * pool.length)];

    this.recentPhraseBlocks = [...this.recentPhraseBlocks, pick.meta.id].slice(-6);
    return pick;
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

    // Same-Track Co-Stem Affinity Prior (Authentic Studio Session Glue)
    // Strictly prioritize loops from the SAME track across all stems
    const activeTrack = this.activeTrackId || this.activeLoops.drums?.meta.track_id;
    if (activeTrack) {
      const sameTrackMatches = candidates.filter((l) => l.meta.track_id === activeTrack);
      if (sameTrackMatches.length > 0) {
        candidates = sameTrackMatches;
      } else {
        const drumKey = this.activeLoops.drums?.meta.key || CAMELOT_KEYS[this.currentCamelotCode]?.name?.toLowerCase();
        if (drumKey) {
          const sameKeyMatches = candidates.filter((l) => l.meta.key.toLowerCase().includes(drumKey));
          if (sameKeyMatches.length > 0) {
            candidates = sameKeyMatches;
          }
        }
      }
    }

    const recents = this.recentLoops[stem] || [];
    const freshPool = candidates.filter((l) => !recents.slice(-4).includes(l.meta.id));
    const pool = freshPool.length > 0 ? freshPool : candidates;
    const pick = pool[Math.floor(Math.random() * pool.length)];

    this.recentLoops[stem] = [...recents, pick.meta.id].slice(-6);
    if (!this.activeTrackId && stem === "drums") {
      this.activeTrackId = pick.meta.track_id;
    }
    // Tempo-only playback rate: NEVER warp pitch with playbackRate to avoid tempo destruction!
    const rate = this.currentBpm / pick.meta.source_bpm;
    return { loop: pick, rate };
  }

  private getBestSlice(
    stemCategory: string,
    targetPitch?: number,
    variationFraction?: number,
    territory?: StyleTerritory
  ): { buffer: AudioBuffer; rate: number; asset?: SliceAsset } | null {
    // 1. Look for style-specific slices first (e.g. drums:kick:wave / drums:kick_wave)
    let list: SliceAsset[] | undefined;
    if (territory) {
      list = this.sliceRegistry.get(`${stemCategory}:${territory}`) ||
             this.sliceRegistry.get(`${stemCategory}_${territory}`);
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
            const altList = (territory ? (this.sliceRegistry.get(`vocals:${alt}:${territory}`) || this.sliceRegistry.get(`vocals:${alt}_${territory}`)) : undefined) ||
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
            const altList = (territory ? (this.sliceRegistry.get(`bass:${alt}:${territory}`) || this.sliceRegistry.get(`bass:${alt}_${territory}`)) : undefined) ||
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
            const altList = (territory ? (this.sliceRegistry.get(`other:${alt}:${territory}`) || this.sliceRegistry.get(`other:${alt}_${territory}`)) : undefined) ||
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
      // Prioritize slices with high acoustic scores from the trained feedback loop
      const trainedScores = this.stemCohesionWeights?.trained_slice_scores;
      let effectiveList = list;
      if (trainedScores && list.length > 2) {
        effectiveList = [...list].sort((a, b) => {
          const sA = trainedScores[a.name] ?? 0.8;
          const sB = trainedScores[b.name] ?? 0.8;
          return sB - sA;
        });
      }
      const v = variationFraction !== undefined ? Math.max(0, Math.min(0.999, variationFraction)) : Math.random();
      const idx = Math.floor(v * effectiveList.length) % effectiveList.length;
      return { buffer: effectiveList[idx].buffer, rate: 1.0, asset: effectiveList[idx] };
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

    // High-variety slice selection across qualified candidates
    if (candidates.length > 0) {
      const trainedScores = this.stemCohesionWeights?.trained_slice_scores;
      if (trainedScores) {
        candidates.sort((a, b) => {
          const sA = trainedScores[a.item.name] ?? 0.8;
          const sB = trainedScores[b.item.name] ?? 0.8;
          return sB - sA;
        });
      }
      const v = variationFraction !== undefined ? Math.max(0, Math.min(0.999, variationFraction)) : Math.random();
      const cIdx = Math.floor(v * candidates.length) % candidates.length;
      best = candidates[cIdx].item;
      bestShift = candidates[cIdx].shift;
    }

    // If exact chromatic pitch exists (bestDiff == 0), rate is 1.0 (zero drift!).
    // For vocals and melodic stems, allow up to +/- 6 semitones (Rule 6: Harmonic pitch shift)
    // to sing exact target melody notes while preserving timbre.
    const maxShift = stemCategory.startsWith("vocals") ? 6 : 3;
    const clampedShift = Math.max(-maxShift, Math.min(maxShift, bestShift));
    const rate = Math.pow(2, clampedShift / 12);
    return { buffer: best.buffer, rate, asset: best };
  }

  private selectBestAsset(
    stemCategory: string,
    territory?: StyleTerritory,
    targetPitch?: number,
    targetCoord?: { x: number; y: number }
  ): SliceAsset | null {
    let pool: SliceAsset[] = [];
    const [stem, cat] = stemCategory.split(":");

    if (targetCoord) {
      // Gather all slice assets available for this category across all territories
      for (const t of ["wave", "cyber", "house", "breaks"]) {
        const tList = this.sliceRegistry.get(`${stem}:${cat}:${t}`) || this.sliceRegistry.get(`${stem}:${cat}_${t}`);
        if (tList) pool.push(...tList);
      }
      const directList = this.sliceRegistry.get(stemCategory);
      if (directList) pool.push(...directList);

      // Fallback within stem if category pool is empty
      if (pool.length === 0 && stem === "drums") {
        for (const alt of ["kick", "snare", "hat_closed", "hat_open", "percussion", "fill"]) {
          for (const t of ["wave", "cyber", "house", "breaks"]) {
            const altList = this.sliceRegistry.get(`drums:${alt}:${t}`) || this.sliceRegistry.get(`drums:${alt}_${t}`);
            if (altList) pool.push(...altList);
          }
        }
      }
    } else {
      if (territory) {
        pool = this.sliceRegistry.get(`${stemCategory}:${territory}`) ||
               this.sliceRegistry.get(`${stemCategory}_${territory}`) || [];
      }
      if (pool.length === 0) {
        pool = this.sliceRegistry.get(stemCategory) || [];
      }
      if (pool.length === 0 && stem === "drums") {
        for (const alt of ["kick", "snare", "hat_closed", "hat_open", "percussion", "fill"]) {
          if (alt !== cat) {
            const altList = (territory ? (this.sliceRegistry.get(`drums:${alt}:${territory}`) || this.sliceRegistry.get(`drums:${alt}_${territory}`)) : undefined) ||
              this.sliceRegistry.get(`drums:${alt}`);
            if (altList && altList.length > 0) {
              pool = altList;
              break;
            }
          }
        }
      }
    }

    if (pool.length === 0) {
      return null;
    }

    // Deduplicate candidate pool by unique asset name
    const seen = new Set<string>();
    const list: SliceAsset[] = [];
    for (const item of pool) {
      if (!seen.has(item.name)) {
        seen.add(item.name);
        list.push(item);
      }
    }

    const trainedScores = this.stemCohesionWeights?.trained_slice_scores;
    let bestAsset = list[0];
    let bestScore = -99999;

    for (const item of list) {
      let score = (trainedScores && trainedScores[item.name]) ? (trainedScores[item.name] * 2.0) : 1.6;
      // Reticule Distance Steering: heavily favor slices physically closest to the reticule on the map
      if (targetCoord && typeof item.x === "number" && typeof item.y === "number") {
        const dx = item.x - targetCoord.x;
        const dy = item.y - targetCoord.y;
        const dist = Math.hypot(dx, dy);
        score -= dist * 10.0;
      }
      if (targetPitch !== undefined) {
        let diff = (targetPitch - item.pitch) % 12;
        if (diff > 6) diff -= 12;
        if (diff < -6) diff += 12;
        score -= Math.abs(diff) * 0.35;
      }
      if (score > bestScore) {
        bestScore = score;
        bestAsset = item;
      }
    }
    return bestAsset;
  }

  /**
   * Snaps reticule coordinates to the nearest real authentic pattern extracted
   * from the dance music corpus (manifest.patterns or manifest.patterns_by_style).
   */
  public selectNearestRealPattern(stem: StemName, coord: { x: number; y: number }): SerializedPattern | null {
    if (!this.manifest) return null;

    const candidatePatterns: Array<{ pat: SerializedPattern; px: number; py: number; style: StyleTerritory }> = [];
    const seen = new Set<string>();

    const terr = getStyleTerritory(coord.x, coord.y);
    const quadrants: Record<StyleTerritory, [number, number, number, number]> = {
      wave: [0.05, 0.45, 0.05, 0.45],
      cyber: [0.55, 0.95, 0.05, 0.45],
      house: [0.05, 0.45, 0.55, 0.95],
      breaks: [0.55, 0.95, 0.55, 0.95],
    };

    if (this.manifest.patterns_by_style) {
      for (const [st, stems] of Object.entries(this.manifest.patterns_by_style)) {
        const styleName = st as StyleTerritory;
        const pats = (stems as Record<string, SerializedPattern[]>)?.[stem] || [];
        const q = quadrants[styleName] || quadrants.house;
        for (const p of pats) {
          if (seen.has(p.id)) continue;
          seen.add(p.id);
          const trackPos = this.getTrackCoordForPattern(p.id, stem);
          let px = trackPos?.x;
          let py = trackPos?.y;
          if (px === undefined || py === undefined) {
            const e = typeof p.energy === "number" ? p.energy : 0.75;
            const d = typeof p.density === "number" ? p.density : 0.5;
            px = q[0] + (q[1] - q[0]) * Math.min(1.0, Math.max(0.0, (e - 0.5) / 0.5));
            py = q[2] + (q[3] - q[2]) * Math.min(1.0, Math.max(0.0, d));
          }
          candidatePatterns.push({ pat: p, px, py, style: styleName });
        }
      }
    }

    if (this.manifest.patterns?.[stem]) {
      for (const p of this.manifest.patterns[stem]) {
        if (seen.has(p.id)) continue;
        seen.add(p.id);
        const styleName = this.inferPatternStyle(p);
        const q = quadrants[styleName] || quadrants.house;
        const trackPos = this.getTrackCoordForPattern(p.id, stem);
        let px = trackPos?.x;
        let py = trackPos?.y;
        if (px === undefined || py === undefined) {
          const e = typeof p.energy === "number" ? p.energy : 0.75;
          const d = typeof p.density === "number" ? p.density : 0.5;
          px = q[0] + (q[1] - q[0]) * Math.min(1.0, Math.max(0.0, (e - 0.5) / 0.5));
          py = q[2] + (q[3] - q[2]) * Math.min(1.0, Math.max(0.0, d));
        }
        candidatePatterns.push({ pat: p, px, py, style: styleName });
      }
    }

    if (candidatePatterns.length === 0) return null;

    let bestPat: SerializedPattern | null = null;
    let bestScore = -Infinity;

    for (const item of candidatePatterns) {
      const dx = item.px - coord.x;
      const dy = item.py - coord.y;
      const dist = Math.hypot(dx, dy);

      const sim = item.pat.audio_similarity_score ?? 0.85;
      const coh = item.pat.cohesion_score ?? sim;
      const rec = (item.pat as any).recurrence_utility ?? 0.80;
      const isMatchingTerr = (item.style === terr) ? 0.40 : 0.0;
      const densityBonus = (item.pat.onset_grid && item.pat.onset_grid.length >= 3) ? 1.5 : (item.pat.onset_grid?.length <= 1 ? -3.0 : 0.0);

      const score = -12.0 * dist + 1.2 * sim + 0.8 * coh + 1.5 * rec + isMatchingTerr + densityBonus;
      if (score > bestScore) {
        bestScore = score;
        bestPat = item.pat;
      }
    }

    return bestPat;
  }

  private getTrackCoordForPattern(patternId: string, stem: StemName): { x: number; y: number } | null {
    const parts = patternId.split("_");
    if (parts.length < 3) return null;
    const prefix = parts[2];
    for (const [key, pos] of this.trackStemPositions) {
      if (key.startsWith(prefix) && key.endsWith(stem)) {
        return pos;
      }
    }
    for (const [key, pos] of this.trackStemPositions) {
      if (key.startsWith(prefix)) {
        return pos;
      }
    }
    return null;
  }

  private inferPatternStyle(pat: SerializedPattern): StyleTerritory {
    const gt = (pat.groove_type || "").toLowerCase();
    if (gt.includes("wave") || gt.includes("ambient") || gt.includes("chill")) return "wave";
    if (gt.includes("cyber") || gt.includes("phonk") || gt.includes("acid")) return "cyber";
    if (gt.includes("break") || gt.includes("jungle") || gt.includes("2step")) return "breaks";
    if (gt.includes("house") || gt.includes("four_on_floor") || gt.includes("disco")) return "house";

    const e = typeof pat.energy === "number" ? pat.energy : 0.75;
    const d = typeof pat.density === "number" ? pat.density : 0.5;

    if (e < 0.72 && d < 0.35) return "wave";
    if (e >= 0.85 && d < 0.50) return "cyber";
    if (d >= 0.60) return "breaks";
    return "house";
  }

  public initSongKit(territory: StyleTerritory, keyRoot: number, coord?: { x: number; y: number }): void {
    const dCoord = (coord && this.steering.drums) || this.steering.drums;
    const bCoord = (coord && this.steering.bass) || this.steering.bass;
    const oCoord = (coord && this.steering.other) || this.steering.other;
    const vCoord = (coord && this.steering.vocals) || this.steering.vocals;

    const dTerr = getStyleTerritory(dCoord.x, dCoord.y);
    const bTerr = getStyleTerritory(bCoord.x, bCoord.y);
    const oTerr = getStyleTerritory(oCoord.x, oCoord.y);
    const vTerr = getStyleTerritory(vCoord.x, vCoord.y);

    const kickAsset = this.selectBestAsset("drums:kick", dTerr, undefined, dCoord);
    const snareAsset = this.selectBestAsset("drums:snare", dTerr, undefined, dCoord);
    const hatClosedAsset = this.selectBestAsset("drums:hat_closed", dTerr, undefined, dCoord);
    const hatOpenAsset = this.selectBestAsset("drums:hat_open", dTerr, undefined, dCoord);
    const percAsset = this.selectBestAsset("drums:percussion", dTerr, undefined, dCoord);
    const fillAsset = this.selectBestAsset("drums:fill", dTerr, undefined, dCoord);

    const kickBuf = kickAsset ? kickAsset.buffer : this.synth.getSliceBuffer(`drums:kick:${dTerr}`) || this.synth.getSliceBuffer("drums:kick");
    const snareBuf = snareAsset ? snareAsset.buffer : this.synth.getSliceBuffer(`drums:snare:${dTerr}`) || this.synth.getSliceBuffer("drums:snare");
    const hatClosedBuf = hatClosedAsset ? hatClosedAsset.buffer : this.synth.getSliceBuffer(`drums:hat_closed:${dTerr}`) || this.synth.getSliceBuffer("drums:hat_closed");
    const hatOpenBuf = hatOpenAsset ? hatOpenAsset.buffer : this.synth.getSliceBuffer(`drums:hat_open:${dTerr}`) || this.synth.getSliceBuffer("drums:hat_open");
    const percBuf = percAsset ? percAsset.buffer : this.synth.getSliceBuffer(`drums:percussion:${dTerr}`) || this.synth.getSliceBuffer("drums:percussion");
    const fillBuf = fillAsset ? fillAsset.buffer : this.synth.getSliceBuffer(`drums:fill:${dTerr}`) || this.synth.getSliceBuffer("drums:fill");

    this.activeSongKit = {
      territory,
      keyRoot,
      kick: kickBuf ? { buffer: kickBuf, rate: 1.0, asset: kickAsset || undefined } : null,
      snare: snareBuf ? { buffer: snareBuf, rate: 1.0, asset: snareAsset || undefined } : null,
      hatClosed: hatClosedBuf ? { buffer: hatClosedBuf, rate: 1.0, asset: hatClosedAsset || undefined } : null,
      hatOpen: hatOpenBuf ? { buffer: hatOpenBuf, rate: 1.0, asset: hatOpenAsset || undefined } : null,
      percussion: percBuf ? { buffer: percBuf, rate: 1.0, asset: percAsset || undefined } : null,
      fill: fillBuf ? { buffer: fillBuf, rate: 1.0, asset: fillAsset || undefined } : null,

      bassSubAsset: this.selectBestAsset("bass:sub", bTerr, keyRoot, bCoord) || this.selectBestAsset("bass:sub", undefined, keyRoot, bCoord),
      bassRollingAsset: this.selectBestAsset("bass:rolling", bTerr, keyRoot, bCoord) || this.selectBestAsset("bass:rolling", undefined, keyRoot, bCoord),

      chordAsset: this.selectBestAsset("other:chord", oTerr, keyRoot, oCoord) || this.selectBestAsset("other:chord", undefined, keyRoot, oCoord),
      padAsset: this.selectBestAsset("other:pad", oTerr, keyRoot, oCoord) || this.selectBestAsset("other:pad", undefined, keyRoot, oCoord),
      arpAsset: this.selectBestAsset("other:arp", oTerr, keyRoot, oCoord) || this.selectBestAsset("other:arp", undefined, keyRoot, oCoord),

      vocalHookAsset: this.selectBestAsset("vocals:hook", vTerr, keyRoot, vCoord) || this.selectBestAsset("vocals:hook", undefined, keyRoot, vCoord),
      vocalCallAsset: this.selectBestAsset("vocals:call", vTerr, keyRoot, vCoord) || this.selectBestAsset("vocals:call", undefined, keyRoot, vCoord),
      vocalChopAsset: this.selectBestAsset("vocals:chop", vTerr, keyRoot, vCoord) || this.selectBestAsset("vocals:chop", undefined, keyRoot, vCoord),
    };
    this.kitKeyRoot = keyRoot;
    this.kitTerritory = territory;
    console.info(`[Client ML] Curated unified song kit locked in for territory: ${territory.toUpperCase()}, key root: ${ROOT_NAMES[keyRoot % 12]}`);
  }

  public updateStemSlicesForReticle(stem: StemName, coord: { x: number; y: number }): void {
    const terr = getStyleTerritory(coord.x, coord.y);
    const chordIndex = Math.floor(this.phraseBar / 2) % this.activeProgression.length;
    const chord = this.activeProgression[chordIndex] || this.activeProgression[0];
    const keyRoot = chord ? chord.rootPitch : 0;

    // 1. Immediately snap active pattern for this stem to nearest REAL extracted pattern from training data!
    const realPat = this.selectNearestRealPattern(stem, coord);
    if (realPat) {
      const prevVels = this.activePatterns[stem]?.velocities || new Array(16).fill(0.9);
      const targetVels = realPat.velocities ? [...realPat.velocities] : new Array(16).fill(0.9);
      // Smoothly blend velocities (75% target, 25% previous) for fluid interpolation
      const blendedVels = targetVels.map((tv, idx) => tv * 0.75 + (prevVels[idx] || 0.8) * 0.25);
      this.activePatterns[stem] = {
        ...realPat,
        onset_grid: [...realPat.onset_grid],
        velocities: blendedVels,
      };
      console.info(`[Client ML] Reticule locked nearest real pattern for ${stem}: ${realPat.name || realPat.id} (${realPat.groove_type})`);
    }

    if (!this.activeSongKit) {
      this.initSongKit(terr, keyRoot, coord);
      return;
    }

    if (stem === "drums") {
      const kickAsset = this.selectBestAsset("drums:kick", terr, undefined, coord);
      const snareAsset = this.selectBestAsset("drums:snare", terr, undefined, coord);
      const hatClosedAsset = this.selectBestAsset("drums:hat_closed", terr, undefined, coord);
      const hatOpenAsset = this.selectBestAsset("drums:hat_open", terr, undefined, coord);
      const percAsset = this.selectBestAsset("drums:percussion", terr, undefined, coord);
      const fillAsset = this.selectBestAsset("drums:fill", terr, undefined, coord);

      const kickBuf = kickAsset ? kickAsset.buffer : this.synth.getSliceBuffer(`drums:kick:${terr}`) || this.synth.getSliceBuffer("drums:kick");
      const snareBuf = snareAsset ? snareAsset.buffer : this.synth.getSliceBuffer(`drums:snare:${terr}`) || this.synth.getSliceBuffer("drums:snare");
      const hatClosedBuf = hatClosedAsset ? hatClosedAsset.buffer : this.synth.getSliceBuffer(`drums:hat_closed:${terr}`) || this.synth.getSliceBuffer("drums:hat_closed");
      const hatOpenBuf = hatOpenAsset ? hatOpenAsset.buffer : this.synth.getSliceBuffer(`drums:hat_open:${terr}`) || this.synth.getSliceBuffer("drums:hat_open");
      const percBuf = percAsset ? percAsset.buffer : this.synth.getSliceBuffer(`drums:percussion:${terr}`) || this.synth.getSliceBuffer("drums:percussion");
      const fillBuf = fillAsset ? fillAsset.buffer : this.synth.getSliceBuffer(`drums:fill:${terr}`) || this.synth.getSliceBuffer("drums:fill");

      this.activeSongKit.kick = kickBuf ? { buffer: kickBuf, rate: 1.0, asset: kickAsset || undefined } : null;
      this.activeSongKit.snare = snareBuf ? { buffer: snareBuf, rate: 1.0, asset: snareAsset || undefined } : null;
      this.activeSongKit.hatClosed = hatClosedBuf ? { buffer: hatClosedBuf, rate: 1.0, asset: hatClosedAsset || undefined } : null;
      this.activeSongKit.hatOpen = hatOpenBuf ? { buffer: hatOpenBuf, rate: 1.0, asset: hatOpenAsset || undefined } : null;
      this.activeSongKit.percussion = percBuf ? { buffer: percBuf, rate: 1.0, asset: percAsset || undefined } : null;
      this.activeSongKit.fill = fillBuf ? { buffer: fillBuf, rate: 1.0, asset: fillAsset || undefined } : null;
    } else if (stem === "bass") {
      this.activeSongKit.bassSubAsset = this.selectBestAsset("bass:sub", terr, keyRoot, coord) || this.selectBestAsset("bass:sub", undefined, keyRoot, coord);
      this.activeSongKit.bassRollingAsset = this.selectBestAsset("bass:rolling", terr, keyRoot, coord) || this.selectBestAsset("bass:rolling", undefined, keyRoot, coord);
    } else if (stem === "other") {
      this.activeSongKit.chordAsset = this.selectBestAsset("other:chord", terr, keyRoot, coord) || this.selectBestAsset("other:chord", undefined, keyRoot, coord);
      this.activeSongKit.padAsset = this.selectBestAsset("other:pad", terr, keyRoot, coord) || this.selectBestAsset("other:pad", undefined, keyRoot, coord);
      this.activeSongKit.arpAsset = this.selectBestAsset("other:arp", terr, keyRoot, coord) || this.selectBestAsset("other:arp", undefined, keyRoot, coord);
    } else if (stem === "vocals") {
      this.activeSongKit.vocalHookAsset = this.selectBestAsset("vocals:hook", terr, keyRoot, coord) || this.selectBestAsset("vocals:hook", undefined, keyRoot, coord);
      this.activeSongKit.vocalCallAsset = this.selectBestAsset("vocals:call", terr, keyRoot, coord) || this.selectBestAsset("vocals:call", undefined, keyRoot, coord);
      this.activeSongKit.vocalChopAsset = this.selectBestAsset("vocals:chop", terr, keyRoot, coord) || this.selectBestAsset("vocals:chop", undefined, keyRoot, coord);
    }

    // In Track-Centric mode: Reticule steers to nearest authentic track session in UMAP space!
    const nearestTrack = this.findNearestTrackForCoord(coord, stem);
    if (nearestTrack && nearestTrack !== this.activeTrackId) {
      this.pendingTrackId = nearestTrack;
      console.info(`[Client ML] Reticule steered nearest authentic track: ${nearestTrack} (queued for phrase boundary)`);
    }
  }

  private findNearestTrackForCoord(coord: { x: number; y: number }, stem: StemName): string | null {
    const candidateTracks = new Set<string>();
    for (const [, blocks] of this.phraseBlockRegistry) {
      for (const b of blocks) {
        candidateTracks.add(b.meta.track_id);
      }
    }
    if (candidateTracks.size === 0) return null;

    let bestTrack: string | null = null;
    let minDist = Infinity;

    for (const trackId of candidateTracks) {
      const stemPos = this.trackStemPositions.get(`${trackId}:${stem}`) ||
                      this.trackStemPositions.get(`${trackId}:drums`) ||
                      this.trackStemPositions.get(`${trackId}:other`);
      if (stemPos) {
        const d = Math.hypot(stemPos.x - coord.x, stemPos.y - coord.y);
        if (d < minDist) {
          minDist = d;
          bestTrack = trackId;
        }
      }
    }
    return bestTrack;
  }

  private getPitchedSliceFromAsset(
    asset: SliceAsset | null | undefined,
    targetPitch: number,
    maxShift: number = 6
  ): { buffer: AudioBuffer; rate: number; asset: SliceAsset } | null {
    if (!asset || !asset.buffer) return null;
    let diff = (targetPitch - asset.pitch) % 12;
    if (diff > 6) diff -= 12;
    if (diff < -6) diff += 12;
    const clampedShift = Math.max(-maxShift, Math.min(maxShift, diff));
    const rate = Math.pow(2, clampedShift / 12);
    return { buffer: asset.buffer, rate, asset };
  }

  public setPlaybackMode(mode: "jukebox" | "slices" | "phrase_blocks"): void {
    this.playbackMode = mode;
    this.activePhraseBlock = null;
    this.activeLoops = {};
    console.info(`[Client ML] Jukebox playback mode set to: ${mode}`);
  }

  public getPlaybackMode(): "jukebox" | "slices" | "phrase_blocks" {
    return this.playbackMode;
  }

  public getMovementPhase(): "FOUNDATION" | "DEVELOPMENT" | "CLIMAX" | "PRE_TRANSITION" {
    const bar = this.arranger.getBarInSection();
    const total = this.arranger.getSectionDuration();
    const left = total - bar;
    if (left <= 2 && total >= 8) {
      return "PRE_TRANSITION";
    }
    if (bar < Math.max(8, Math.floor(total * 0.25))) {
      return "FOUNDATION";
    }
    if (bar < Math.max(16, Math.floor(total * 0.65))) {
      return "DEVELOPMENT";
    }
    return "CLIMAX";
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
    const chordIndex = Math.floor(this.phraseBar / 2) % this.activeProgression.length;
    const chord = this.activeProgression[chordIndex];
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

  public shiftToNextProgression(): void {
    if (this.manifest?.chord_progressions && this.manifest.chord_progressions.length > 0) {
      this.progressionIndex = (this.progressionIndex + 1) % this.manifest.chord_progressions.length;
      this.loadProgressionFromManifest(this.progressionIndex);
    } else {
      this.progressionIndex = (this.progressionIndex + 1) % DEFAULT_PROGRESSIONS.length;
      this.activeProgression = DEFAULT_PROGRESSIONS[this.progressionIndex];
    }
  }

  public loadProgressionFromManifest(index: number): void {
    if (!this.manifest?.chord_progressions || this.manifest.chord_progressions.length === 0) return;
    const p = this.manifest.chord_progressions[index % this.manifest.chord_progressions.length];
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

  public start(bpm?: number): void {
    if (this.isRunning) return;
    this.isRunning = true;
    if (bpm) {
      this.currentBpm = Math.max(60, Math.min(200, bpm));
    }
    this.nextStepTime = this.engine.now() + 0.05;
    this.tatumIndex = 0;
    this.totalBarsPlayed = 0;
    this.recentLoops = { drums: [], bass: [], vocals: [], other: [] };

    // 1. Randomize starting section for rich variety (atmospheric INTRO, punchy GROOVE_A, or rising BUILDUP)
    const startingSec = this.arranger.randomizeStartingSection();

    // 2. Randomize starting chord progression from corpus or defaults
    if (this.manifest?.chord_progressions && this.manifest.chord_progressions.length > 0) {
      this.progressionIndex = Math.floor(Math.random() * this.manifest.chord_progressions.length);
      this.loadProgressionFromManifest(this.progressionIndex);
    } else {
      this.progressionIndex = Math.floor(Math.random() * DEFAULT_PROGRESSIONS.length);
      this.activeProgression = DEFAULT_PROGRESSIONS[this.progressionIndex];
    }

    // 3. Randomize initial vocal persona/song theme
    this.vocalSongArranger.randomizeSong();

    // 4. Prime curated unified song kit for starting territory and key
    const startTerritory = getStyleTerritory(this.steering.drums.x, this.steering.drums.y);
    const startKey = (this.activeProgression && this.activeProgression.length > 0) ? this.activeProgression[0].rootPitch : 0;
    this.initSongKit(startTerritory, startKey);

    // 5. Initial filters: pure acoustic slices wide open at 20 kHz
    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      this.engine.setStemFilter(stem, 20000, 0.7);
    }

    this.composeBarPatterns(0, true);

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
      let isSectionChange = false;
      if (tatum > 0) {
        const res = this.onBarStart();
        isSectionChange = res.isSectionChange;
      }
      this.composeBarPatterns(phraseBar, isSectionChange);
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

  private onBarStart(): { isSectionChange: boolean; section: string } {
    const { isSectionChange, section } = this.arranger.advanceBar();
    this.totalBarsPlayed++;
    this.barsInCurrentKey++;

    if (isSectionChange) {
      // Dynamic section transition filter staging
      // Keep all stem filters wide open at 20 kHz for pure, un-muffled slice clarity
      for (const s of ["drums", "bass", "other", "vocals"] as StemName[]) {
        this.engine.setStemFilter(s, 20000, 0.7);
      }

      // Section transition: reset active phrase block & loops so new section selects section-appropriate audio!
      this.activePhraseBlock = null;
      this.activeLoops = {};
    }

    // Key Modulation (Camelot Drift):
    // Invariant: NEVER modulate key on BUILDUP -> DROP transition!
    // A drop must always land with crushing resolve in the same key as the buildup.
    // Modulate key only after a full song cycle (barsInCurrentKey >= 48)
    // or when entering a reflective BREAKDOWN / INTRO.
    const isHarmonicResetPoint = (section === "BREAKDOWN" || section === "INTRO") && isSectionChange;
    const isSongCycleComplete = this.barsInCurrentKey >= 48 && section !== "BUILDUP" && section !== "DROP";

    if (isHarmonicResetPoint || isSongCycleComplete) {
      this.activeTrackId = undefined; // allow a fresh track identity on major song modulation!
      const shouldModulate = Math.random() < 0.65;
      if (shouldModulate) {
        this.currentCamelotCode = getAdjacentCamelotKey(this.currentCamelotCode);
        this.currentTemplateIndex = (this.currentTemplateIndex + 1) % HARMONIC_TEMPLATES.length;
        this.barsInCurrentKey = 0;
        const keyInfo = CAMELOT_KEYS[this.currentCamelotCode] || CAMELOT_KEYS["8A"];
        this.activeProgression = buildProgressionFromKey(keyInfo, this.currentTemplateIndex);
        console.info(`[KG-505 Harmonic Drift] Key shifted to ${keyInfo.code} (${keyInfo.name}) • Template #${this.currentTemplateIndex + 1}`);
      } else {
        this.shiftToNextProgression();
      }

      this.phraseBarCount = 0;
      this.activePhraseBlock = null;
      if (section === "GROOVE_A" || section === "INTRO") {
        this.vocalSongArranger.nextSong();
      }
    } else {
      this.phraseBarCount = (this.phraseBarCount + 1) % 8;
    }

    return { isSectionChange, section };
  }

  private schedulePhraseBlock(phrase: LoadedPhraseBlock, when: number, sec: string, barInSec: number): void {
    const totalSecBars = this.arranger.getSectionDuration();
    const isBuildupClimax = (sec === "BUILDUP" && barInSec >= totalSecBars - 4);
    // On the final bar Step 15 of BUILDUP (the 4th bar of the climax block): Silent drop gap breath!
    // In a 4-bar phrase, the 4th bar is bar index 3. 3 bars * 16 steps + 15 = 63 sixteenths.
    const dropGapBreathSec = isBuildupClimax ? (63 * 60) / (this.currentBpm * 4) : undefined;

    // Tempo-only rate: NO PITCH RESAMPLING, ZERO DRIFT!
    const rate = this.currentBpm / phrase.meta.source_bpm;

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const buffer = phrase.buffers[stem];
      if (!buffer) continue;

      // Boost acoustic presence for vocals and other (synths/chords) so they cut clearly through driving drums and bass
      const stemGain = stem === "vocals" ? 1.55 : (stem === "other" ? 1.40 : 1.0);
      this.engine.scheduleSlice(stem, buffer, when, {
        rate,
        duration: dropGapBreathSec,
        fadeMs: 24,
        gain: stemGain,
        voiceKey: `phrase_${stem}`,
      });

      let grid = phrase.meta.onset_grids[stem];
      if (!grid || grid.filter(g => g === 1).length <= 2) {
        if (stem === "other") {
          grid = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
        } else if (stem === "vocals") {
          grid = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        } else {
          grid = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
        }
      }
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
    // Subtle, transparent ducking: dip bass slightly by -2 dB (gain 0.78) and quick 38ms release so punch is preserved
    if (this.bassSidechainGain) {
      const t = Math.max(this.engine.now(), when);
      this.bassSidechainGain.gain.cancelScheduledValues(t);
      this.bassSidechainGain.gain.setValueAtTime(0.78, t);
      this.bassSidechainGain.gain.exponentialRampToValueAtTime(1.0, t + 0.038);
    }
    // Dip synths slightly by -3 dB (gain 0.72) and release over 45ms
    if (this.synthSidechainGain) {
      const t = Math.max(this.engine.now(), when);
      this.synthSidechainGain.gain.cancelScheduledValues(t);
      this.synthSidechainGain.gain.setValueAtTime(0.72, t);
      this.synthSidechainGain.gain.exponentialRampToValueAtTime(1.0, t + 0.045);
    }
  }

  private applyVocalDucking(when: number, durationSec: number): void {
    // Frequency Masking & Vocal Presence Carving:
    // Dip synths/chords by -2.5 dB while vocal formants sing to avoid mid-range frequency masking
    if (this.synthSidechainGain) {
      const t = Math.max(this.engine.now(), when);
      const duckDb = this.stemCohesionWeights?.frequency_masking_targets?.synth_ducking_on_vocal_db ?? -2.5;
      const duckGain = Math.pow(10, duckDb / 20);
      this.synthSidechainGain.gain.cancelScheduledValues(t);
      this.synthSidechainGain.gain.setValueAtTime(duckGain, t);
      this.synthSidechainGain.gain.linearRampToValueAtTime(1.0, t + durationSec + 0.05);
    }
  }

  /**
   * Automates the volume levels of the four instruments (drums, bass, other, vocals)
   * to gently and smoothly mix them in and out of the global audio wave across phrases,
   * section builds, drop gaps, and transcendent breakdowns.
   */
  private updateStemMixAutomation(
    when: number,
    sec: string,
    barInSec: number,
    totalSecBars: number,
    stepInBar: number
  ): void {
    const barsLeft = Math.max(0, totalSecBars - barInSec);
    const progressInSec = Math.max(0, Math.min(1, (barInSec * 16 + stepInBar) / (totalSecBars * 16)));
    const isTransitioning = barsLeft <= 2;
    const nextSec = sec === "BUILDUP" ? "DROP" : (sec === "DROP" ? "BREAKDOWN" : (sec === "BREAKDOWN" ? "GROOVE_A" : (sec === "INTRO" ? "GROOVE_A" : "BUILDUP")));

    // Baseline mix profiles per section: pure slices of goodness unmuted
    const secProfiles: Record<string, Record<StemName, number>> = {
      INTRO: { drums: 0.90, bass: 0.90, other: 1.15, vocals: 1.25 },
      GROOVE_A: { drums: 1.00, bass: 1.00, other: 1.12, vocals: 1.35 },
      BUILDUP: { drums: 1.00, bass: 0.95, other: 1.20, vocals: 1.40 },
      DROP: { drums: 1.08, bass: 1.08, other: 1.08, vocals: 1.38 },
      BREAKDOWN: { drums: 0.85, bass: 0.85, other: 1.25, vocals: 1.45 },
    };

    const currProfile = secProfiles[sec] || secProfiles.GROOVE_A;
    const nextProfile = secProfiles[nextSec] || secProfiles.GROOVE_A;

    let dTarget = currProfile.drums ?? 1.00;
    let bTarget = currProfile.bass ?? 1.00;
    let oTarget = currProfile.other ?? 1.12;
    let vTarget = currProfile.vocals ?? 1.35;

    if (sec === "INTRO") {
      dTarget = 0.85 + 0.15 * progressInSec;
      bTarget = 0.85 + 0.15 * progressInSec;
      oTarget = 0.95 + 0.25 * progressInSec;
      vTarget = 1.10 + 0.25 * progressInSec;
    } else if (sec === "BUILDUP") {
      dTarget = 0.95 + 0.10 * progressInSec;
      bTarget = 0.90 + 0.10 * progressInSec;
      oTarget = 1.05 + 0.25 * progressInSec;
      vTarget = 1.20 + 0.25 * progressInSec;

      // Drop Gap Breath: Bar 7 Step 15 is 100% SILENT for maximum drop impact!
      if (barInSec >= totalSecBars - 1 && stepInBar === 15) {
        dTarget = 0.0;
        bTarget = 0.0;
        oTarget = 0.0;
        vTarget = 0.0;
      }
    } else if (sec === "DROP") {
      if (barInSec === 0) {
        dTarget = 1.08;
        bTarget = 1.08;
        oTarget = 1.00;
        vTarget = 1.30;
      } else {
        dTarget = 1.00;
        bTarget = 1.00;
        oTarget = 1.08;
        vTarget = 1.38;
      }
    } else if (sec === "BREAKDOWN") {
      dTarget = 0.85;
      bTarget = 0.85;
      oTarget = 1.25;
      vTarget = 1.45;
    }

    // Smooth asymptotic transition blending in the penultimate 2 bars
    if (isTransitioning && sec !== "BUILDUP") {
      const blendFactor = Math.max(0, Math.min(1, ((2 - barsLeft) * 16 + stepInBar) / 32));
      const sCurve = 0.5 - 0.5 * Math.cos(blendFactor * Math.PI);
      dTarget = dTarget * (1.0 - sCurve) + (nextProfile.drums ?? dTarget) * sCurve;
      bTarget = bTarget * (1.0 - sCurve) + (nextProfile.bass ?? bTarget) * sCurve;
      oTarget = oTarget * (1.0 - sCurve) + (nextProfile.other ?? oTarget) * sCurve;
      vTarget = vTarget * (1.0 - sCurve) + (nextProfile.vocals ?? vTarget) * sCurve;
    }

    const rampTime = (sec === "BUILDUP" && barInSec >= totalSecBars - 1 && stepInBar === 15) ? 0.005 : 0.08;
    this.stemMixLevels.drums = dTarget;
    this.stemMixLevels.bass = bTarget;
    this.stemMixLevels.other = oTarget;
    this.stemMixLevels.vocals = vTarget;

    this.engine.setStemMixLevel("drums", dTarget, rampTime);
    this.engine.setStemMixLevel("bass", bTarget, rampTime);
    this.engine.setStemMixLevel("other", oTarget, rampTime);
    this.engine.setStemMixLevel("vocals", vTarget, rampTime);
  }

  private scheduleStepAudio(step: number, when: number, phraseBar: number): void {
    const sec = this.arranger.getSection();
    const barInSec = this.arranger.getBarInSection();
    const totalSecBars = this.arranger.getSectionDuration();
    const chordIndex = Math.floor(phraseBar / 2) % this.activeProgression.length;
    const chord = this.activeProgression[chordIndex];

    // Automate volume levels of the four instruments for transcendent flow
    this.updateStemMixAutomation(when, sec, barInSec, totalSecBars, step % 16);

    const dInput = this.engine.getStemInput("drums");
    const bDest = this.bassSidechainGain || this.engine.getStemInput("bass");
    const oDest = this.synthSidechainGain || this.engine.getStemInput("other");
    const vInput = this.engine.getStemInput("vocals");

    // Active Style Territories per stem
    const dTerr = this.getStemTerritory("drums");
    const bTerr = this.getStemTerritory("bass");
    const oTerr = this.getStemTerritory("other");
    const vTerr = this.getStemTerritory("vocals");

    // 0. Downbeat handling: phrase blocks mode vs jukebox studio loops mode vs live slice mode
    if (this.playbackMode === "phrase_blocks" || this.playbackMode === "jukebox") {
      if (step % 16 === 0) {
        // Cohesive 4-bar phrase blocks loop continuously across the section without jumping between random tracks!
        if (barInSec % 4 === 0) {
          // If a new track was steered via reticule, smoothly transition to it at phrase boundary!
          if (this.pendingTrackId && this.pendingTrackId !== this.activeTrackId) {
            const steeredBlock = this.getBestPhraseBlock(sec, this.pendingTrackId);
            if (steeredBlock) {
              this.activePhraseBlock = steeredBlock;
              this.activeTrackId = this.pendingTrackId;
              this.pendingTrackId = undefined;
              console.info(`[Client ML] DJ Crossfade: Steered to authentic track ${steeredBlock.meta.track_title} at phrase boundary.`);
            }
          }
          if (!this.activePhraseBlock) {
            this.activePhraseBlock = this.getBestPhraseBlock(sec, this.activeTrackId);
            if (this.activePhraseBlock && !this.activeTrackId) {
              this.activeTrackId = this.activePhraseBlock.meta.track_id;
            }
          }
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

    // =========================================================================
    // Master Section Transition Sound Effects & Movement Orchestration
    // =========================================================================
    const movement = this.getMovementPhase();
    const barsLeft = this.arranger.getSectionBarsLeft();
    const isTransitionBar = (barsLeft <= 1);

    // 1. DROP Explosive Impact on Step 0 (Crash Cymbal + Sub Boom)
    if (sec === "DROP" && barInSec === 0 && (step % 16) === 0) {
      this.synth.playCrashCymbal(when, 1.0, dInput);
      this.synth.playSubDropImpact(when, 1.0, bDest);
    }
    // 2. Reverse Cymbal Swell leading into Drop Gap Breath or Major Section Transition
    if ((sec === "BUILDUP" && isTransitionBar && (step % 16) === 11) ||
        (movement === "PRE_TRANSITION" && isTransitionBar && (step % 16) === 11)) {
      const stepDur = (60 / this.currentBpm / 4);
      this.synth.playReverseCymbalSwell(when, stepDur * 3.8, dInput);
    }
    // 3. Dynamic Filter Automation on Transitions
    if (movement === "PRE_TRANSITION") {
      const nextSec = sec === "BUILDUP" ? "DROP" : (sec === "DROP" ? "BREAKDOWN" : "BUILDUP");
      if (nextSec === "BREAKDOWN") {
        const progressInBar = (step % 16) / 16;
        const cutoff = 18000 - progressInBar * 14000;
        this.engine.setStemFilter("other", Math.max(3000, cutoff), 1.0);
      } else if (nextSec === "DROP" || nextSec === "BUILDUP") {
        const progressInBar = (step % 16) / 16;
        const cutoff = 4000 + progressInBar * 14000;
        this.engine.setStemFilter("other", Math.min(20000, cutoff), 1.1);
      }
    }

    // -------------------------------------------------------------------------
    // 1. DRUMS (Section-Aware Arrangement with Real Acoustic Slices & Reticule Steering)
    // -------------------------------------------------------------------------
    // Ensure unified CuratedSongKit is locked in for active territory & chord progression
    if (!this.activeSongKit || this.kitTerritory !== dTerr || ((phraseBar === 0 && (step % 16 === 0)) && this.kitKeyRoot !== chord.rootPitch && this.totalBarsPlayed % 16 === 0)) {
      this.initSongKit(dTerr, chord.rootPitch);
    }

    const dPat = this.activePatterns.drums;
    const dSteer = this.steering.drums;
    const isLoopDrumActive = Boolean((this.playbackMode === "jukebox" || this.playbackMode === "phrase_blocks") && (this.activePhraseBlock?.buffers?.drums || this.activeLoops.drums));
    if (isLoopDrumActive) {
      // Continuous studio drum phrase or loop is active: dynamic sidechain ducking on downbeats
      if (sec !== "BREAKDOWN" && [0, 4, 8, 12].includes(step % 16)) {
        this.applySidechainDucking(when);
      }
      // Trigger Pitch Riser on penultimate 2 bars of BUILDUP
      if (sec === "BUILDUP" && barsLeft === 2 && step === 0) {
        this.synth.playRiser(when, (60 / this.currentBpm) * 8, dInput);
      }
      // Groovy Slicer Turnaround on Bar 7 Step 14 or pre-transition turnaround
      if ((phraseBar === 7 || (movement === "PRE_TRANSITION" && isTransitionBar)) && (step % 16) === 14) {
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
      // Groovy Ghost Perc on Step 10 during DEVELOPMENT or CLIMAX
      if ((movement === "DEVELOPMENT" || movement === "CLIMAX" || dSteer.y > 0.70) && (step % 16) === 10) {
        const percSlice = this.getBestSlice("drums:percussion", undefined, dSteer.y, dTerr);
        if (percSlice) {
          this.engine.scheduleSlice("drums", percSlice.buffer, when, {
            rate: 1.0,
            fadeMs: 1.5,
            gain: 0.65,
            voiceKey: "drums:percussion",
          });
        }
      }
      // Crisp Open Hi-Hat on offbeats during CLIMAX (step 2 & 10)
      if (movement === "CLIMAX" && (step % 16 === 2 || step % 16 === 10)) {
        const hatSlice = this.getBestSlice("drums:hat_open", undefined, dSteer.y, dTerr);
        if (hatSlice) {
          this.engine.scheduleSlice("drums", hatSlice.buffer, when, {
            rate: 1.0,
            fadeMs: 1.0,
            gain: 0.45,
            voiceKey: "drums:hat_open",
          });
        }
      }
    } else if (dPat) {
      const isBuildup = (sec === "BUILDUP");
      const isBreakdown = (sec === "BREAKDOWN");

      if (isBreakdown) {
        // Breakdown: Minimal acoustic groove with deep warm kicks on 0 & 8 and gentle hats
        if (step % 16 === 0 || step % 16 === 8) {
          const kickSlice = this.activeSongKit?.kick || this.getBestSlice("drums:kick", undefined, 0, dTerr);
          if (kickSlice) {
            this.engine.scheduleSlice("drums", kickSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: 0.80,
              voiceKey: "drums:kick",
            });
          }
        }
        if ([2, 6, 10, 14].includes(step % 16)) {
          const hatSlice = this.activeSongKit?.hatClosed || this.getBestSlice("drums:hat_closed", undefined, 0, dTerr);
          if (hatSlice) {
            this.engine.scheduleSlice("drums", hatSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.0,
              gain: 0.55,
              voiceKey: "drums:hat_closed",
            });
          } else {
            this.synth.playHat(when, 0.55, false, dInput, dTerr);
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
          const snareSlice = this.activeSongKit?.snare || this.getBestSlice("drums:snare", undefined, 0, dTerr);
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
          const kickSlice = this.activeSongKit?.kick || this.getBestSlice("drums:kick", undefined, 0, dTerr);
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
        // Standard Groove / Drop: Rock-solid 4-on-the-floor dance pocket with acoustic clarity
        const vel = dPat.velocities ? dPat.velocities[step] : 1.0;

        // 1. Kick: Rock-solid 4-on-the-floor [0, 4, 8, 12] across dance sections
        let isKickStep = false;
        if (dTerr === "breaks") {
          // Breaks: 2-step / syncopated breakbeat (kicks on 0, 6, 10)
          isKickStep = [0, 6, 10].includes(step % 16);
        } else if (dTerr === "wave" && dSteer.y < 0.35) {
          // Ambient Wave: Halftime kicks on 0 & 8
          isKickStep = (step % 16 === 0 || step % 16 === 8);
        } else {
          // House / Cyber / Electro: Driving 4-on-the-floor [0, 4, 8, 12] + optional groove kick on 10
          isKickStep = [0, 4, 8, 12].includes(step % 16) ||
            ((step % 16 === 10) && (dSteer.y > 0.55 || dTerr === "cyber" || dPat.groove_type.includes("syncopated")));
        }

        if (isKickStep) {
          const kickSlice = this.activeSongKit?.kick || this.getBestSlice("drums:kick", undefined, 0, dTerr);
          const kickGain = (step % 16 === 10) ? vel * 0.75 : vel;
          if (kickSlice) {
            this.engine.scheduleSlice("drums", kickSlice.buffer, when, {
              rate: 1.0,
              fadeMs: 1.5,
              gain: kickGain,
              voiceKey: "drums:kick",
            });
          } else {
            this.synth.playKick(when, kickGain, dInput, dTerr);
          }
          this.applySidechainDucking(when);
        }

        // 2. Snare / Clap: Crisp snap on backbeats [4, 12] + turnaround rolls
        let isSnareStep = false;
        if (dTerr === "wave" && dSteer.y < 0.35) {
          isSnareStep = (step % 16 === 8) || (phraseBar === 7 && step % 16 >= 8 && step % 16 !== 15);
        } else if (dTerr === "breaks") {
          isSnareStep = (step % 16 === 4 || step % 16 === 12) || (dSteer.y > 0.65 && (step % 16 === 7 || (step % 16 === 15 && phraseBar !== 7)));
        } else {
          isSnareStep = ((step % 16) === 4 || (step % 16) === 12) ||
            (phraseBar === 7 && (step % 16) >= 8 && (step % 16) !== 15) ||
            (dSteer.y > 0.75 && (step % 16) === 14);
        }

        if (isSnareStep) {
          const snareSlice = this.activeSongKit?.snare || this.getBestSlice("drums:snare", undefined, 0, dTerr);
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

        // 3. Hi-Hats: Sizzling open hat on upbeats (2, 6, 10, 14)
        if ([2, 6, 10, 14].includes(step % 16)) {
          const hatSlice = this.activeSongKit?.hatOpen || this.getBestSlice("drums:hat_open", undefined, 0, dTerr);
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
        }

        // 4. Closed 16th Hi-Hats: Driving continuous groove with choking & micro-velocity accents
        if (sec !== "INTRO") {
          const isUpbeat = [2, 6, 10, 14].includes(step % 16);
          const isGhost16th = [1, 3, 5, 7, 9, 11, 13, 15].includes(step % 16);
          if (isGhost16th || (dSteer.y > 0.40 && !isUpbeat)) {
            const hatSlice = this.activeSongKit?.hatClosed || this.getBestSlice("drums:hat_closed", undefined, 0, dTerr);
            const hatVel = isGhost16th ? 0.68 : 0.52;
            if (hatSlice) {
              this.engine.scheduleSlice("drums", hatSlice.buffer, when, {
                rate: 1.0,
                fadeMs: 1.0,
                gain: hatVel,
                voiceKey: "drums:hat_closed",
                chokeKeys: isUpbeat ? [] : ["drums:hat_open"],
              });
            } else {
              this.synth.playHat(when, hatVel, false, dInput, dTerr);
            }
          }
        }

        // 5. Syncopated Percussion & Accents
        if ([3, 7, 11].includes(step % 16) && (sec === "DROP" || sec === "GROOVE_A" || dSteer.y > 0.55 || dTerr === "cyber")) {
          const percSlice = this.activeSongKit?.percussion || this.getBestSlice("drums:percussion", undefined, 0, dTerr);
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

        // 6. Dramatic Turnaround Fill on Bar 7 Step 14
        if (phraseBar === 7 && step === 14) {
          const fillSlice = this.activeSongKit?.fill || this.getBestSlice("drums:fill", undefined, 0, dTerr);
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
    const isLoopBassActive = Boolean((this.playbackMode === "jukebox" || this.playbackMode === "phrase_blocks") && (this.activeLoops.bass || this.activePhraseBlock?.buffers?.bass));
    if (bPat && !isLoopBassActive && !(sec === "BUILDUP" && phraseBar === 7 && (step % 16) === 15)) {
      const bassCat = bSteer.x < 0.38 ? "bass:sub" : (bSteer.x < 0.72 ? "bass:rolling" : "bass:stab");
      const isSparseBass = (bSteer.y < 0.35 && bTerr !== "cyber");
      const isDenseBass = (bSteer.y > 0.70 || bTerr === "cyber");

      const shouldPlayBass = isSparseBass
        ? (step % 16 === 0 || step % 16 === 8)
        : (isDenseBass
          ? ([0, 2, 3, 5, 6, 8, 10, 11, 13, 14].includes(step % 16) || bPat.onset_grid.includes(step))
          : ([2, 6, 10, 14].includes(step % 16) || bPat.onset_grid.includes(step)));

      if (shouldPlayBass) {
        let vel = bPat.velocities ? (bPat.velocities[step] || 0.90) : 0.90;
        let dur = (bassCat === "bass:rolling") ? 0.14 : (bassCat === "bass:sub" ? 0.28 : 0.20);
        if (bTerr === "wave") dur = 0.38; // Spacious 808 sub
        else if (bTerr === "cyber") dur = 0.12; // Snappy acid/reese rolling

        const subSlice = (bassCat === "bass:rolling")
          ? (this.getPitchedSliceFromAsset(this.activeSongKit?.bassRollingAsset, chord.rootPitch, 3) || this.getBestSlice(bassCat, chord.rootPitch, 0, bTerr))
          : (this.getPitchedSliceFromAsset(this.activeSongKit?.bassSubAsset, chord.rootPitch, 3) || this.getBestSlice(bassCat, chord.rootPitch, 0, bTerr));
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
    const isLoopOtherActive = Boolean((this.playbackMode === "jukebox" || this.playbackMode === "phrase_blocks") && (this.activeLoops.other || this.activePhraseBlock?.buffers?.other));
    if (oPat && !isLoopOtherActive) {
      const otherCat = oSteer.x < 0.40 ? "other:pad" : (oSteer.x < 0.75 ? "other:chord" : "other:arp");
      const isArpMode = (otherCat === "other:arp" || sec === "DROP" || oSteer.y > 0.72 || oTerr === "cyber");
      const isSparsePad = ((otherCat === "other:pad" || oSteer.y < 0.35) && oTerr !== "cyber" && oTerr !== "breaks");

      const shouldPlayOther = isSparsePad
        ? (step % 16 === 0 || step % 16 === 8)
        : (isArpMode ? (step % 2 === 0 || oPat.onset_grid.includes(step)) : ([2, 6, 10, 14].includes(step % 16) || oPat.onset_grid.includes(step)));

      if (shouldPlayOther) {
        const vel = oPat.velocities ? (oPat.velocities[step] || 0.85) : 0.85;

        if (isArpMode) {
          const arpSlice = this.getPitchedSliceFromAsset(this.activeSongKit?.arpAsset, chord.rootPitch, 3) || this.getBestSlice("other:arp", chord.rootPitch, 0, oTerr);
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
          const chordSlice = isSparsePad
            ? (this.getPitchedSliceFromAsset(this.activeSongKit?.padAsset, chord.rootPitch, 3) ||
               this.getPitchedSliceFromAsset(this.activeSongKit?.chordAsset, chord.rootPitch, 3) ||
               this.getBestSlice(targetSliceCat, chord.rootPitch, 0, oTerr))
            : (this.getPitchedSliceFromAsset(this.activeSongKit?.chordAsset, chord.rootPitch, 3) ||
               this.getBestSlice("other:chord", chord.rootPitch, 0, oTerr));
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
    const isLoopVocalActive = Boolean((this.playbackMode === "jukebox" || this.playbackMode === "phrase_blocks") && (this.activeLoops.vocals || this.activePhraseBlock?.buffers?.vocals));
    const isDropGap = (sec === "BUILDUP" && phraseBar === 7 && (step % 16) === 15);

    const syllable = this.vocalSongArranger.getSyllableForStep(sec, phraseBar, step % 16, vTerr);
    if (syllable) {
      const melodyPitch = (chord.rootPitch + syllable.semitoneOffset) % 12;
      const noteName = `${ROOT_NAMES[melodyPitch]}4`;
      const activeTitle = this.vocalSongArranger.getSongTitle(vTerr);
      this.currentVocalMeta = {
        lyric: this.vocalSongArranger.getCurrentLyricLine(sec, phraseBar, vTerr),
        syllable: syllable.text,
        vowel: syllable.vowel,
        note: noteName,
        songTitle: activeTitle,
        sectionName: sec,
      };
    }

    if (!isDropGap && vPat) {
      const vocalCat = vSteer.x < 0.38 ? "vocals:call" : (vSteer.x < 0.72 ? "vocals:hook" : "vocals:chop");
      const isDenseChop = (vocalCat === "vocals:chop" || vSteer.y > 0.72 || vTerr === "breaks");
      const isSparseVocal = (vSteer.y < 0.35 && vTerr !== "breaks" && vTerr !== "cyber");

      const shouldPlayVocal = isSparseVocal
        ? ((step % 16 === 0 || step % 16 === 8) && (phraseBar % 2 === 0))
        : (isDenseChop ? ((step % 4 === 0 || step % 4 === 2) || syllable !== undefined) : (syllable !== undefined || vPat.onset_grid.includes(step % 16)));

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

        // Dynamic Frequency Pocketing: Duck other/synths so vocal formants stand out
        this.applyVocalDucking(when, durSec);

        const activeTitle = this.vocalSongArranger.getSongTitle(vTerr);
        this.currentVocalMeta = {
          lyric: this.vocalSongArranger.getCurrentLyricLine(sec, phraseBar, vTerr),
          syllable: syl.text,
          vowel: syl.vowel,
          note: noteName,
          songTitle: activeTitle,
          sectionName: sec,
        };

        // Sings melody using authentic acoustic vocal slices whenever available
        const lockedVocalAsset = (vocalCat === "vocals:chop")
          ? (this.activeSongKit?.vocalChopAsset || this.activeSongKit?.vocalHookAsset)
          : (this.activeSongKit?.vocalHookAsset || this.activeSongKit?.vocalCallAsset);
        const slice = this.getPitchedSliceFromAsset(lockedVocalAsset, melodyPitch, 6) ||
          this.getBestSlice(vocalCat, melodyPitch, 0, vTerr) ||
          this.getBestSlice("vocals:hook", melodyPitch, 0, vTerr) ||
          this.getBestSlice("vocals:call", melodyPitch, 0, vTerr) ||
          this.getBestSlice("vocals:chop", melodyPitch, 0, vTerr);

        if (slice) {
          if (slice.asset?.trackTitle) {
            this.currentVocalMeta.songTitle = slice.asset.trackTitle;
          }
          if (slice.asset?.vowel) {
            this.currentVocalMeta.vowel = slice.asset.vowel as any;
          }
          // Sing current melody note using authentic acoustic vocal slice
          const sliceGain = isLoopVocalActive ? vel * 1.15 : vel * 1.35;
          this.engine.scheduleSlice("vocals", slice.buffer, when, {
            rate: slice.rate,
            fadeMs: 6,
            gain: sliceGain,
            voiceKey: "vocals",
          });
        } else {
          // Sing expressive vocaloid vocalization
          const synthVel = isLoopVocalActive ? vel * 0.95 : vel * 1.15;
          this.synth.playSingingVoice(when, noteHz, syl.vowel, durSec, synthVel, vInput, vTerr);
        }
      }
    }
  }

  private scheduleBarLoops(when: number, sec: string, phraseBar: number, chord: ChordStep): void {
    const isBuildupDropBreath = (sec === "BUILDUP" && phraseBar === 7);
    const dropBreathDuration = isBuildupDropBreath ? (14.85 * 60) / (this.currentBpm * 4) : undefined;
    const barInSec = this.arranger.getBarInSection();

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const loop = this.activeLoops[stem];
      if (!loop) continue;

      // Pure acoustic slice groove across all sections
      if (stem === "vocals" && sec === "INTRO" && phraseBar !== 0 && phraseBar !== 4) {
        continue; // Ethereal calls on bars 0 & 4 in ambient intro
      }

      // Tempo-only rate matching without pitch warping to preserve perfect rhythm pocket
      const rate = this.currentBpm / loop.meta.source_bpm;
      const stemGain = stem === "vocals" ? 1.50 : (stem === "other" ? 1.35 : 1.0);

      this.engine.scheduleSlice(stem, loop.buffer, when, {
        rate,
        duration: dropBreathDuration,
        fadeMs: stem === "vocals" ? 8 : 4,
        gain: stemGain,
        voiceKey: `loop_${stem}`,
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

  private composeBarPatterns(phraseBar: number, isSectionChange: boolean = false): void {
    const sec = this.arranger.getSection();
    const isNewPhrase = (phraseBar === 0);
    const isTurnaround = (phraseBar === 7);

    // Harmonic pacing: chords sustain for 2 bars (32 steps) across the phrase sequence
    const chordIndex = Math.floor(phraseBar / 2) % this.activeProgression.length;
    const chord = this.activeProgression[chordIndex];
    const isChordBoundary = (phraseBar % 2 === 0);

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      // Section-Wide Rhythm & Harmonic Lock:
      // - DRUMS, BASS & OTHER (synths/chords): Firmly lock for the ENTIRE SECTION!
      //   Never swap out the foundational groove or harmonic feel in the middle of a section or after 128 ticks!
      // - VOCALS: Stay locked to the current song theme for the section (re-phrase only on new phrase).
      const shouldLockStem = !isSectionChange &&
        ((stem === "drums" || stem === "bass" || stem === "other") ||
         (stem === "vocals" && !isNewPhrase));

      if (shouldLockStem && (this.activeLoops[stem] || this.activePatterns[stem])) {
        // Locked! Keep existing groove/pattern playing without frantic swapping
        continue;
      }

      if (this.playbackMode === "jukebox" || this.playbackMode === "phrase_blocks") {
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
          if (onsetIndices.length <= 1) {
            const enrich = (stem === "vocals") ? [0, 3, 6, 8, 10, 12, 14] : (stem === "other" ? [2, 6, 10, 14] : [0, 4, 8, 12]);
            for (const oi of enrich) {
              if (!onsetIndices.includes(oi)) {
                onsetIndices.push(oi);
                vels[oi] = 0.90;
              }
            }
            onsetIndices.sort((a, b) => a - b);
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
      const movement = this.getMovementPhase();

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
        } else if (movement === "FOUNDATION") {
          nextG = "pad";
        } else if (movement === "DEVELOPMENT") {
          nextG = "chords";
        } else if (movement === "CLIMAX") {
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
          if (movement === "FOUNDATION") {
            nextG = "vocal_call_response";
          } else if (movement === "DEVELOPMENT") {
            nextG = (phraseBar % 2 === 0) ? "vocal_verse" : "vocal_call_response";
          } else if (movement === "CLIMAX") {
            nextG = "vocal_syncopated_chop";
          } else {
            nextG = "vocal_call_response";
          }
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
    } else if (stem === "other") {
      if (mutated.onset_grid.length <= 1) {
        mutated.onset_grid = [2, 6, 10, 14];
        mutated.density = 4 / 16;
        for (const st of mutated.onset_grid) {
          mutated.velocities[st] = 0.90;
        }
      }
      if (steerY > 0.70) {
        for (const st of [0, 8]) {
          if (!mutated.onset_grid.includes(st)) {
            mutated.onset_grid.push(st);
            mutated.velocities[st] = 0.78;
          }
        }
      }
    } else if (stem === "vocals") {
      if (mutated.onset_grid.length <= 1) {
        const gt = mutated.groove_type || "vocal_verse";
        if (gt === "vocal_hook" || gt === "hook") {
          mutated.onset_grid = [0, 3, 6, 8, 10, 12, 14];
        } else if (gt === "vocal_stutter" || gt === "stutter") {
          mutated.onset_grid = [0, 2, 4, 6, 8, 10, 12, 14];
        } else if (gt === "vocal_syncopated_chop" || gt === "chop") {
          mutated.onset_grid = [0, 2, 5, 8, 11, 14];
        } else if (gt === "vocal_call_response" || gt === "call") {
          mutated.onset_grid = [0, 3, 6, 10, 14];
        } else {
          mutated.onset_grid = [0, 4, 7, 10, 12];
        }
        mutated.density = mutated.onset_grid.length / 16;
        for (const st of mutated.onset_grid) {
          mutated.velocities[st] = 0.92;
        }
      }
      if (steerY > 0.65) {
        for (const st of [2, 10]) {
          if (!mutated.onset_grid.includes(st)) {
            mutated.onset_grid.push(st);
            mutated.velocities[st] = 0.85;
          }
        }
      }
    }
    mutated.onset_grid.sort((a, b) => a - b);
    return mutated;
  }

  private computeInterStemAffinity(stem: StemName, candidateGroove: string): number {
    if (!this.stemCohesionWeights) return 0.85;
    const compat = this.stemCohesionWeights.stem_pair_compatibility;
    if (!compat) return 0.85;

    let score = 0.85;
    if (stem === "bass") {
      const dG = this.activePatterns.drums?.groove_type || "four_on_floor_hats";
      score = compat.drums_bass?.[dG]?.[candidateGroove] ??
              compat.drums_bass?.["four_on_floor_hats"]?.[candidateGroove] ?? 0.85;
    } else if (stem === "other") {
      const vG = this.activePatterns.vocals?.groove_type || "vocal_verse";
      score = compat.other_vocals?.[candidateGroove]?.[vG] ??
              compat.other_vocals?.["chords"]?.[vG] ?? 0.85;
    } else if (stem === "vocals") {
      const oG = this.activePatterns.other?.groove_type || "chords";
      score = compat.other_vocals?.[oG]?.[candidateGroove] ??
              compat.other_vocals?.["chords"]?.[candidateGroove] ?? 0.85;
    } else if (stem === "drums") {
      const bG = this.activePatterns.bass?.groove_type || "offbeat";
      score = compat.drums_bass?.[candidateGroove]?.[bG] ??
              compat.drums_bass?.[candidateGroove]?.["offbeat"] ?? 0.85;
    }
    return Math.max(0.2, Math.min(1.0, score));
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
        // ML Features Feedback Loop:
        // Rank and probabilistically select patterns based on:
        // 1. audio_similarity_score (from original song training data)
        // 2. cohesion_score (multi-stem mix synergy)
        // 3. inter-stem affinity (compatibility with currently active partner stems)
        const steerY = this.steering[stem]?.y ?? 0.5;
        const scores = matches.map((pat) => {
          const sim = pat.audio_similarity_score ?? 0.85;
          const coh = pat.cohesion_score ?? sim;
          const aff = this.computeInterStemAffinity(stem, pat.groove_type);
          return (0.45 * sim) + (0.35 * coh) + (0.20 * aff);
        });

        const temp = 0.12 + Math.abs(steerY - 0.5) * 0.20;
        const maxScore = Math.max(...scores);
        const exps = scores.map((s) => Math.exp((s - maxScore) / temp));
        const sumExp = exps.reduce((acc, val) => acc + val, 0);

        let r = Math.random() * sumExp;
        let chosenIdx = 0;
        for (let i = 0; i < exps.length; i++) {
          r -= exps[i];
          if (r <= 0) {
            chosenIdx = i;
            break;
          }
        }
        const pat = { ...matches[chosenIdx] };
        return this.applyPatternMutation(stem, pat);
      }

      // Fallback: select nearest real pattern from the authentic corpus
      const nearestReal = this.selectNearestRealPattern(stem, this.steering[stem]);
      if (nearestReal) {
        return this.applyPatternMutation(stem, { ...nearestReal });
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
    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const nearest = this.selectNearestRealPattern(stem, this.steering[stem]);
      if (nearest) {
        this.activePatterns[stem] = {
          ...nearest,
          onset_grid: [...nearest.onset_grid],
          velocities: nearest.velocities ? [...nearest.velocities] : new Array(16).fill(0.9),
        };
      } else {
        const defaultGrooves: Record<StemName, string> = {
          drums: "four_on_floor_hats",
          bass: "offbeat",
          other: "chords",
          vocals: "vocal_verse",
        };
        this.activePatterns[stem] = this.synthesizePattern(stem, defaultGrooves[stem], this.getStemTerritory(stem));
      }
    }
  }

  private buildMeta(stepInPattern: number, patternInCycle: number, phraseBar: number): any {
    const sec = this.arranger.getSection();
    const isFill = (phraseBar === 7);
    const chordIndex = Math.floor(phraseBar / 2) % this.activeProgression.length;
    const chord = this.activeProgression[chordIndex];

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
      movement_phase: this.getMovementPhase(),
      movementPhase: this.getMovementPhase(),
      bars_until_transition: this.arranger.getSectionBarsLeft(),
      barsUntilTransition: this.arranger.getSectionBarsLeft(),
      is_transition_bar: this.arranger.getSectionBarsLeft() <= 1,
      isTransitionBar: this.arranger.getSectionBarsLeft() <= 1,
      transition_type: `${sec}_TO_${sec === "BUILDUP" ? "DROP" : (sec === "DROP" ? "GROOVE_A" : "BUILDUP")}`,
      transitionType: `${sec}_TO_${sec === "BUILDUP" ? "DROP" : (sec === "DROP" ? "GROOVE_A" : "BUILDUP")}`,
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
      active_slice_coords: {
        drums: this.activeSongKit?.kick?.asset?.x !== undefined ? { x: this.activeSongKit.kick.asset.x, y: this.activeSongKit.kick.asset.y! } : undefined,
        bass: this.activeSongKit?.bassSubAsset?.x !== undefined ? { x: this.activeSongKit.bassSubAsset.x, y: this.activeSongKit.bassSubAsset.y! } : undefined,
        other: this.activeSongKit?.chordAsset?.x !== undefined ? { x: this.activeSongKit.chordAsset.x, y: this.activeSongKit.chordAsset.y! } : undefined,
        vocals: this.activeSongKit?.vocalHookAsset?.x !== undefined ? { x: this.activeSongKit.vocalHookAsset.x, y: this.activeSongKit.vocalHookAsset.y! } : undefined,
      },
      activeSliceCoords: {
        drums: this.activeSongKit?.kick?.asset?.x !== undefined ? { x: this.activeSongKit.kick.asset.x, y: this.activeSongKit.kick.asset.y! } : undefined,
        bass: this.activeSongKit?.bassSubAsset?.x !== undefined ? { x: this.activeSongKit.bassSubAsset.x, y: this.activeSongKit.bassSubAsset.y! } : undefined,
        other: this.activeSongKit?.chordAsset?.x !== undefined ? { x: this.activeSongKit.chordAsset.x, y: this.activeSongKit.chordAsset.y! } : undefined,
        vocals: this.activeSongKit?.vocalHookAsset?.x !== undefined ? { x: this.activeSongKit.vocalHookAsset.x, y: this.activeSongKit.vocalHookAsset.y! } : undefined,
      },
      stem_mix_levels: { ...this.stemMixLevels },
      stemMixLevels: { ...this.stemMixLevels },
      direct_ml_engine: true,
    };
  }
}
