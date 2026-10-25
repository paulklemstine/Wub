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
}

export interface ComposingMLWeights {
  version: number;
  phrase_transitions: Record<string, Record<string, number>>;
  section_transitions: Record<string, Record<string, number>>;
  section_energy_targets: Record<string, number>;
}

// Semitone pitch constants: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
export const ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

export interface ChordStep {
  rootPitch: number; // 0..11
  isMinor: boolean;
  bassHz: number;
  chordHz: number;
  name: string;
}

// Built-in authentic harmonic dance chord progressions learned from the corpus
const DEFAULT_PROGRESSIONS: ChordStep[][] = [
  // 1. Epic Anthem: i - VI - III - VII (Am - F - C - G)
  [
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 5, isMinor: false, bassHz: 43.65, chordHz: 174.6, name: "F" },
    { rootPitch: 5, isMinor: false, bassHz: 43.65, chordHz: 174.6, name: "F" },
    { rootPitch: 0, isMinor: false, bassHz: 65.4, chordHz: 130.8, name: "C" },
    { rootPitch: 0, isMinor: false, bassHz: 65.4, chordHz: 130.8, name: "C" },
    { rootPitch: 7, isMinor: false, bassHz: 49.0, chordHz: 196.0, name: "G" },
    { rootPitch: 7, isMinor: false, bassHz: 49.0, chordHz: 196.0, name: "G" },
  ],
  // 2. Deep Melodic House: i - iv - v - i (Am - Dm - Em - Am)
  [
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 2, isMinor: true, bassHz: 73.4, chordHz: 146.8, name: "Dm" },
    { rootPitch: 2, isMinor: true, bassHz: 73.4, chordHz: 146.8, name: "Dm" },
    { rootPitch: 4, isMinor: true, bassHz: 82.4, chordHz: 164.8, name: "Em" },
    { rootPitch: 4, isMinor: true, bassHz: 82.4, chordHz: 164.8, name: "Em" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
  ],
  // 3. Dark Club Driving Pedal: i - i - bVI - bVII (Am - Am - F - G)
  [
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 9, isMinor: true, bassHz: 55.0, chordHz: 110.0, name: "Am" },
    { rootPitch: 5, isMinor: false, bassHz: 43.65, chordHz: 174.6, name: "F" },
    { rootPitch: 5, isMinor: false, bassHz: 43.65, chordHz: 174.6, name: "F" },
    { rootPitch: 7, isMinor: false, bassHz: 49.0, chordHz: 196.0, name: "G" },
    { rootPitch: 7, isMinor: false, bassHz: 49.0, chordHz: 196.0, name: "G" },
  ],
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
  chorusHook: VocalSyllable[];
  verseLineA: VocalSyllable[];
  verseLineB: VocalSyllable[];
  buildupChant: VocalSyllable[];
  breakdownSolo: VocalSyllable[];
}

export class VocalSongArranger {
  private currentThemeIndex: number = 0;
  private themes: VocalSongTheme[] = [
    {
      title: "FEEL THE BEAT",
      chorusHook: [
        { text: "FEEL", vowel: "EE", semitoneOffset: 12, step: 0, durationSteps: 2, velocity: 1.0 },
        { text: "THE", vowel: "OH", semitoneOffset: 10, step: 3, durationSteps: 2, velocity: 0.92 },
        { text: "BEAT", vowel: "EE", semitoneOffset: 12, step: 6, durationSteps: 2, velocity: 0.98 },
        { text: "FEEL", vowel: "EE", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.90 },
        { text: "THE", vowel: "OH", semitoneOffset: 5, step: 10, durationSteps: 2, velocity: 0.92 },
        { text: "GROOVE", vowel: "OO", semitoneOffset: 7, step: 12, durationSteps: 2, velocity: 0.96 },
        { text: "NOW", vowel: "AH", semitoneOffset: 3, step: 14, durationSteps: 2, velocity: 1.0 },
      ],
      verseLineA: [
        { text: "NIGHT", vowel: "AY", semitoneOffset: 0, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "IS", vowel: "EE", semitoneOffset: 2, step: 3, durationSteps: 2, velocity: 0.85 },
        { text: "DEEP", vowel: "EE", semitoneOffset: 3, step: 6, durationSteps: 3, velocity: 0.90 },
        { text: "LOW", vowel: "OH", semitoneOffset: 0, step: 11, durationSteps: 4, velocity: 0.95 },
      ],
      verseLineB: [
        { text: "HEAT", vowel: "EE", semitoneOffset: 7, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "OF", vowel: "OH", semitoneOffset: 5, step: 3, durationSteps: 2, velocity: 0.85 },
        { text: "SOUND", vowel: "AH", semitoneOffset: 7, step: 6, durationSteps: 3, velocity: 0.92 },
        { text: "GO", vowel: "OH", semitoneOffset: 3, step: 11, durationSteps: 4, velocity: 0.95 },
      ],
      buildupChant: [
        { text: "HIGH", vowel: "AY", semitoneOffset: 5, step: 0, durationSteps: 2, velocity: 0.82 },
        { text: "ER", vowel: "AH", semitoneOffset: 7, step: 2, durationSteps: 2, velocity: 0.85 },
        { text: "TAKE", vowel: "AY", semitoneOffset: 8, step: 4, durationSteps: 2, velocity: 0.88 },
        { text: "ME", vowel: "EE", semitoneOffset: 10, step: 6, durationSteps: 2, velocity: 0.90 },
        { text: "HIGH", vowel: "AY", semitoneOffset: 12, step: 8, durationSteps: 2, velocity: 0.94 },
        { text: "ER", vowel: "AH", semitoneOffset: 12, step: 10, durationSteps: 2, velocity: 0.96 },
        { text: "NOW", vowel: "AH", semitoneOffset: 14, step: 12, durationSteps: 2, velocity: 0.98 },
        { text: "FEEL", vowel: "EE", semitoneOffset: 15, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "OOH", vowel: "OO", semitoneOffset: 7, step: 0, durationSteps: 5, velocity: 0.95 },
        { text: "HOLD", vowel: "OH", semitoneOffset: 5, step: 6, durationSteps: 2, velocity: 0.88 },
        { text: "ON", vowel: "AH", semitoneOffset: 3, step: 8, durationSteps: 4, velocity: 0.92 },
        { text: "YEAH", vowel: "AY", semitoneOffset: 0, step: 12, durationSteps: 4, velocity: 0.96 },
      ],
    },
    {
      title: "INFINITE WUB",
      chorusHook: [
        { text: "WUB", vowel: "OO", semitoneOffset: 0, step: 0, durationSteps: 2, velocity: 1.0 },
        { text: "THAT", vowel: "AH", semitoneOffset: 3, step: 3, durationSteps: 2, velocity: 0.92 },
        { text: "SOUND", vowel: "AH", semitoneOffset: 5, step: 6, durationSteps: 2, velocity: 0.98 },
        { text: "DEEP", vowel: "EE", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.90 },
        { text: "IN", vowel: "EE", semitoneOffset: 10, step: 10, durationSteps: 2, velocity: 0.94 },
        { text: "SIDE", vowel: "AY", semitoneOffset: 12, step: 12, durationSteps: 2, velocity: 0.96 },
        { text: "YEAH", vowel: "AY", semitoneOffset: 7, step: 14, durationSteps: 2, velocity: 1.0 },
      ],
      verseLineA: [
        { text: "BASS", vowel: "AY", semitoneOffset: 0, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "LINE", vowel: "AY", semitoneOffset: 3, step: 4, durationSteps: 3, velocity: 0.88 },
        { text: "PUMPS", vowel: "AH", semitoneOffset: 5, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "DOWN", vowel: "AH", semitoneOffset: 0, step: 12, durationSteps: 4, velocity: 0.95 },
      ],
      verseLineB: [
        { text: "MOVE", vowel: "OO", semitoneOffset: 7, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "YOUR", vowel: "OH", semitoneOffset: 5, step: 4, durationSteps: 3, velocity: 0.88 },
        { text: "FEET", vowel: "EE", semitoneOffset: 7, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "NOW", vowel: "AH", semitoneOffset: 3, step: 12, durationSteps: 4, velocity: 0.95 },
      ],
      buildupChant: [
        { text: "WUB", vowel: "OO", semitoneOffset: 0, step: 0, durationSteps: 2, velocity: 0.82 },
        { text: "WUB", vowel: "OO", semitoneOffset: 3, step: 2, durationSteps: 2, velocity: 0.85 },
        { text: "WUB", vowel: "OO", semitoneOffset: 5, step: 4, durationSteps: 2, velocity: 0.88 },
        { text: "WUB", vowel: "OO", semitoneOffset: 7, step: 6, durationSteps: 2, velocity: 0.90 },
        { text: "BASS", vowel: "AY", semitoneOffset: 10, step: 8, durationSteps: 2, velocity: 0.94 },
        { text: "DROP", vowel: "OH", semitoneOffset: 12, step: 10, durationSteps: 2, velocity: 0.96 },
        { text: "IT", vowel: "EE", semitoneOffset: 14, step: 12, durationSteps: 2, velocity: 0.98 },
        { text: "HARD", vowel: "AH", semitoneOffset: 15, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "LOST", vowel: "OH", semitoneOffset: 7, step: 0, durationSteps: 5, velocity: 0.95 },
        { text: "IN", vowel: "EE", semitoneOffset: 8, step: 6, durationSteps: 2, velocity: 0.88 },
        { text: "ECHO", vowel: "OH", semitoneOffset: 5, step: 8, durationSteps: 4, velocity: 0.92 },
        { text: "SOUND", vowel: "AH", semitoneOffset: 0, step: 12, durationSteps: 4, velocity: 0.96 },
      ],
    },
    {
      title: "ELECTRIC NIGHT",
      chorusHook: [
        { text: "LIGHT", vowel: "AY", semitoneOffset: 12, step: 0, durationSteps: 2, velocity: 1.0 },
        { text: "THE", vowel: "OH", semitoneOffset: 10, step: 3, durationSteps: 2, velocity: 0.92 },
        { text: "SKY", vowel: "AY", semitoneOffset: 12, step: 6, durationSteps: 2, velocity: 0.98 },
        { text: "YOU", vowel: "OO", semitoneOffset: 7, step: 8, durationSteps: 2, velocity: 0.90 },
        { text: "AND", vowel: "AH", semitoneOffset: 8, step: 10, durationSteps: 2, velocity: 0.94 },
        { text: "I", vowel: "AY", semitoneOffset: 10, step: 12, durationSteps: 2, velocity: 0.96 },
        { text: "FLY", vowel: "AY", semitoneOffset: 12, step: 14, durationSteps: 2, velocity: 1.0 },
      ],
      verseLineA: [
        { text: "STROBE", vowel: "OH", semitoneOffset: 0, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "LIGHT", vowel: "AY", semitoneOffset: 3, step: 4, durationSteps: 3, velocity: 0.88 },
        { text: "GLOW", vowel: "OH", semitoneOffset: 5, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "BRIGHT", vowel: "AY", semitoneOffset: 7, step: 12, durationSteps: 4, velocity: 0.95 },
      ],
      verseLineB: [
        { text: "WE", vowel: "EE", semitoneOffset: 7, step: 0, durationSteps: 3, velocity: 0.92 },
        { text: "ARE", vowel: "AH", semitoneOffset: 5, step: 4, durationSteps: 3, velocity: 0.88 },
        { text: "ONE", vowel: "AH", semitoneOffset: 3, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "TONIGHT", vowel: "AY", semitoneOffset: 0, step: 12, durationSteps: 4, velocity: 0.95 },
      ],
      buildupChant: [
        { text: "RISE", vowel: "AY", semitoneOffset: 5, step: 0, durationSteps: 3, velocity: 0.85 },
        { text: "UP", vowel: "AH", semitoneOffset: 7, step: 4, durationSteps: 3, velocity: 0.88 },
        { text: "FEEL", vowel: "EE", semitoneOffset: 10, step: 8, durationSteps: 3, velocity: 0.92 },
        { text: "IT", vowel: "EE", semitoneOffset: 12, step: 12, durationSteps: 2, velocity: 0.96 },
        { text: "NOW", vowel: "AH", semitoneOffset: 14, step: 14, durationSteps: 1, velocity: 1.0 },
      ],
      breakdownSolo: [
        { text: "HOLD", vowel: "OH", semitoneOffset: 12, step: 0, durationSteps: 5, velocity: 0.95 },
        { text: "MY", vowel: "AY", semitoneOffset: 10, step: 6, durationSteps: 2, velocity: 0.88 },
        { text: "HAND", vowel: "AH", semitoneOffset: 7, step: 8, durationSteps: 4, velocity: 0.92 },
        { text: "SHINE", vowel: "AY", semitoneOffset: 0, step: 12, durationSteps: 4, velocity: 0.96 },
      ],
    },
  ];

  public getSongTitle(): string {
    return this.themes[this.currentThemeIndex].title;
  }

  public nextSong(): void {
    this.currentThemeIndex = (this.currentThemeIndex + 1) % this.themes.length;
  }

  public getSyllableForStep(section: string, phraseBar: number, step: number): VocalSyllable | undefined {
    const theme = this.themes[this.currentThemeIndex];
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
          return { text: "OOH", vowel: "OO", semitoneOffset: 7, step: 0, durationSteps: 6, velocity: 0.90 };
        }
      }
      return undefined;
    }

    return list.find((s) => s.step === step);
  }

  public getCurrentLyricLine(section: string, phraseBar: number): string {
    const theme = this.themes[this.currentThemeIndex];
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
    return "OOH... (AMBIENT VOCAL CALL)";
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

  // Punchy 909/808 Kick: 5ms click transient + exponential sub body sweep
  public playKick(when: number, velocity: number = 1.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // 1. Click transient (high-frequency snap)
    const clickOsc = this.ctx.createOscillator();
    const clickGain = this.ctx.createGain();
    clickOsc.type = "triangle";
    clickOsc.frequency.setValueAtTime(1400, t);
    clickOsc.frequency.exponentialRampToValueAtTime(120, t + 0.008);
    clickGain.gain.setValueAtTime(velocity * 0.9, t);
    clickGain.gain.exponentialRampToValueAtTime(0.001, t + 0.012);
    clickOsc.connect(clickGain);
    clickGain.connect(dest);
    clickOsc.start(t);
    clickOsc.stop(t + 0.015);

    // 2. Sub Body sweep
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(155, t);
    osc.frequency.exponentialRampToValueAtTime(45, t + 0.075);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 1.35, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.32);

    osc.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + 0.35);
  }

  // Crisp Snare / Clap: Tuned triangle body + stereo bandpassed noise snap
  public playSnare(when: number, velocity: number = 1.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // Body tone
    const osc = this.ctx.createOscillator();
    const oscGain = this.ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(195, t);
    osc.frequency.exponentialRampToValueAtTime(95, t + 0.06);

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
      filter.frequency.value = 1900;
      filter.Q.value = 1.4;

      const noiseGain = this.ctx.createGain();
      noiseGain.gain.setValueAtTime(v * 0.95, t);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, t + 0.22);

      noise.connect(filter);
      filter.connect(noiseGain);
      noiseGain.connect(dest);

      noise.start(t);
      noise.stop(t + 0.24);
    }
  }

  // Sizzling Hi-Hat: 6-oscillator metallic frequency cluster with open-hat choke
  public playHat(when: number, velocity: number = 1.0, isOpen: boolean = false, destNode?: AudioNode): void {
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
    filter.frequency.value = 7500;

    const dur = isOpen ? 0.26 : 0.045;
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
    destNode?: AudioNode
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(pitchHz, t);

    // Warm resonant low-pass filter tracking pitch
    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    const cutoff = Math.min(800, pitchHz * 5.5);
    filter.frequency.setValueAtTime(cutoff, t);
    filter.frequency.exponentialRampToValueAtTime(cutoff * 0.45, t + durationSec);
    filter.Q.value = 2.5;

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
    destNode?: AudioNode
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
    filter.frequency.setValueAtTime(2600, t);
    filter.frequency.exponentialRampToValueAtTime(800, t + durationSec);
    filter.Q.value = 1.8;

    chordGain.connect(filter);
    filter.connect(dest);

    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      // Slight detune for analog richness
      const osc = this.ctx.createOscillator();
      osc.type = "sawtooth";
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
    destNode?: AudioNode
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "square";
    osc.frequency.setValueAtTime(pitchHz, t);

    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(3200, t);
    filter.frequency.exponentialRampToValueAtTime(600, t + 0.12);
    filter.Q.value = 3.0;

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

  // Human Singing Voice Synthesizer with Formant Resonators, Pitch Scoop, Vibrato & Breathiness
  public playSingingVoice(
    when: number,
    noteHz: number = 261.6,
    vowel: "AH" | "EE" | "OH" | "OO" | "AY" = "AH",
    durationSec: number = 0.28,
    velocity: number = 0.90,
    destNode?: AudioNode
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const formants = VOWEL_FORMANTS[vowel] || VOWEL_FORMANTS.AH;

    // 1. Dual Oscillators: Glottal wave + detuned harmonic body
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    osc1.type = "sawtooth";
    osc2.type = "sawtooth";

    // Natural vocal pitch scoop (scoops up ~1.5 semitones into target note)
    const scoopStartHz = noteHz * Math.pow(2, -1.5 / 12);
    osc1.frequency.setValueAtTime(scoopStartHz, t);
    osc1.frequency.exponentialRampToValueAtTime(noteHz, t + 0.04);
    osc2.frequency.setValueAtTime(scoopStartHz * 1.002, t);
    osc2.frequency.exponentialRampToValueAtTime(noteHz * 1.002, t + 0.04);

    // Human Vibrato LFO (~5.2 Hz, fades in after 60ms)
    const vibratoOsc = this.ctx.createOscillator();
    const vibratoGain = this.ctx.createGain();
    vibratoOsc.frequency.setValueAtTime(5.2, t);
    vibratoGain.gain.setValueAtTime(0.0001, t);
    vibratoGain.gain.setValueAtTime(0.0001, t + 0.06);
    vibratoGain.gain.linearRampToValueAtTime(noteHz * 0.015, t + 0.18); // ~25 cents vibrato depth
    vibratoOsc.connect(vibratoGain);
    vibratoGain.connect(osc1.frequency);
    vibratoGain.connect(osc2.frequency);

    // 2. Breath Transient (Filtered noise attack for consonant definition)
    let breathNode: AudioNode | null = null;
    if (this.noiseBuffer) {
      const breathSrc = this.ctx.createBufferSource();
      breathSrc.buffer = this.noiseBuffer;
      const breathFilter = this.ctx.createBiquadFilter();
      breathFilter.type = "highpass";
      breathFilter.frequency.setValueAtTime(3200, t);
      const breathGain = this.ctx.createGain();
      breathGain.gain.setValueAtTime(velocity * 0.15, t);
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
    f1.frequency.setValueAtTime(formants.f1, t);
    f1.Q.setValueAtTime(5.5, t);

    const f2 = this.ctx.createBiquadFilter();
    f2.type = "bandpass";
    f2.frequency.setValueAtTime(formants.f2, t);
    f2.Q.setValueAtTime(6.0, t);

    const f3 = this.ctx.createBiquadFilter();
    f3.type = "bandpass";
    f3.frequency.setValueAtTime(formants.f3, t);
    f3.Q.setValueAtTime(4.5, t);

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
    const attackTime = 0.020;
    const holdTime = Math.max(0.06, durationSec - 0.06);

    masterGain.gain.setValueAtTime(0.0001, t);
    masterGain.gain.linearRampToValueAtTime(v * 0.85, t + attackTime);
    masterGain.gain.setValueAtTime(v * 0.80, t + holdTime);
    masterGain.gain.exponentialRampToValueAtTime(0.0001, t + durationSec + 0.07);

    vocalSum.connect(masterGain);
    masterGain.connect(dest);

    osc1.start(t);
    osc2.start(t);
    vibratoOsc.start(t);

    const stopTime = t + durationSec + 0.08;
    osc1.stop(stopTime);
    osc2.stop(stopTime);
    vibratoOsc.stop(stopTime);
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

  // Harmonic Progression State
  private activeProgression: ChordStep[] = DEFAULT_PROGRESSIONS[0];
  private progressionIndex: number = 0;

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

  private sliceRegistry: Map<string, SliceAsset[]> = new Map();
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
        for (let i = 0; i < Math.min(4, list.length); i++) {
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

  private getBestSlice(stemCategory: string, targetPitch?: number): { buffer: AudioBuffer; rate: number } | null {
    const list = this.sliceRegistry.get(stemCategory);
    if (!list || list.length === 0) {
      const fallbackBuf = this.synth.getSliceBuffer(stemCategory);
      return fallbackBuf ? { buffer: fallbackBuf, rate: 1.0 } : null;
    }

    if (targetPitch === undefined) {
      // Pick with slight variation across available slices
      const idx = Math.floor(Math.random() * list.length);
      return { buffer: list[idx].buffer, rate: 1.0 };
    }

    // For pitched stems (bass, other, vocals): find slice closest to target pitch
    let best = list[0];
    let bestDiff = 999;
    let bestShift = 0;

    for (const item of list) {
      let diff = (targetPitch - item.pitch) % 12;
      if (diff > 6) diff -= 12;
      if (diff < -6) diff += 12;
      if (Math.abs(diff) < bestDiff) {
        bestDiff = Math.abs(diff);
        bestShift = diff;
        best = item;
      }
    }

    // Shift rate by semitones: 2^(shift / 12)
    const rate = Math.pow(2, bestShift / 12);
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
    const sec = this.arranger.getSection();
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
    if (isSectionChange) {
      // Pick next harmonic chord progression for fresh section
      this.progressionIndex = (this.progressionIndex + 1) % DEFAULT_PROGRESSIONS.length;
      this.activeProgression = DEFAULT_PROGRESSIONS[this.progressionIndex];
      // Progress to next song composition when cycling into a new verse/song cycle
      if (section === "GROOVE_A" || section === "INTRO") {
        this.vocalSongArranger.nextSong();
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

    // -------------------------------------------------------------------------
    // 1. DRUMS (Section-Aware Arrangement)
    // -------------------------------------------------------------------------
    const dPat = this.activePatterns.drums;
    if (dPat) {
      const isBuildup = (sec === "BUILDUP");
      const isBreakdown = (sec === "BREAKDOWN");

      if (isBreakdown) {
        // Breakdown: Kicks completely muted, gentle closed hats on 2, 6, 10, 14
        if ([2, 6, 10, 14].includes(step % 16)) {
          this.synth.playHat(when, 0.45, false, dInput);
        }
      } else if (isBuildup) {
        // Buildup: Snare acceleration & tension riser!
        // Steps 0..14: Accelerating snare roll
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
          this.synth.playSnare(when, snareVel, dInput);
        }

        // Downbeat Kick pulse
        if (step % 4 === 0 && barInSec < 6) {
          this.synth.playKick(when, 0.90, dInput);
          this.applySidechainDucking(when);
        }

        // Trigger Pitch Riser on bar 6 step 0
        if (barInSec === 6 && step === 0) {
          this.synth.playRiser(when, (60 / this.currentBpm) * 8, dInput);
        }
      } else {
        // Standard Groove / Drop: Full rhythm driving forward
        const hasHit = dPat.onset_grid.includes(step);
        const vel = dPat.velocities ? dPat.velocities[step] : 1.0;

        // Kick triggers: 0, 4, 8, 12 (or syncopations)
        const isKickStep = ([0, 4, 8, 12].includes(step % 16)) || ((step % 16) === 10 && dPat.groove_type.includes("syncopated"));
        if (isKickStep && hasHit) {
          const kickSlice = this.getBestSlice("drums:kick");
          if (kickSlice) this.engine.scheduleSlice("drums", kickSlice.buffer, when, { rate: kickSlice.rate, fadeMs: 2 });
          this.synth.playKick(when, vel, dInput);
          this.applySidechainDucking(when);
        }

        // Snare / Clap triggers: 4 and 12 (+ fill roll on bar 7)
        const isSnareStep = ((step % 16) === 4 || (step % 16) === 12) || (phraseBar === 7 && (step % 16) >= 8 && (step % 16) !== 15);
        if (isSnareStep) {
          const snareSlice = this.getBestSlice("drums:snare");
          if (snareSlice) this.engine.scheduleSlice("drums", snareSlice.buffer, when, { rate: snareSlice.rate, fadeMs: 2 });
          this.synth.playSnare(when, vel, dInput);
        }

        // Hi-Hats: Open hat on upbeats (2, 6, 10, 14) + closed 16th hats
        if ([2, 6, 10, 14].includes(step % 16)) {
          const hatSlice = this.getBestSlice("drums:hat_open");
          if (hatSlice) this.engine.scheduleSlice("drums", hatSlice.buffer, when, { rate: hatSlice.rate, fadeMs: 2 });
          else this.synth.playHat(when, 0.90, true, dInput);
        } else if (step % 2 === 1 && sec !== "INTRO") {
          const hatSlice = this.getBestSlice("drums:hat_closed");
          const hatVel = (step % 4 === 1 || step % 4 === 3) ? 0.72 : 0.58;
          if (hatSlice) this.engine.scheduleSlice("drums", hatSlice.buffer, when, { rate: hatSlice.rate, fadeMs: 2 });
          else this.synth.playHat(when, hatVel, false, dInput);
        }

        // Syncopated Perc on 3, 7, 11
        if ([3, 7, 11].includes(step % 16) && (sec === "DROP" || sec === "GROOVE_A")) {
          const percSlice = this.getBestSlice("drums:percussion");
          if (percSlice) this.engine.scheduleSlice("drums", percSlice.buffer, when, { rate: percSlice.rate, fadeMs: 2 });
          else this.synth.playPerc(when, 0.70, 420.0, dInput);
        }

        // Dramatic Turnaround Fill on Bar 7
        if (phraseBar === 7 && step === 14) {
          const fillSlice = this.getBestSlice("drums:fill");
          if (fillSlice) this.engine.scheduleSlice("drums", fillSlice.buffer, when, { rate: fillSlice.rate, fadeMs: 2 });
        }
      }
    }

    // -------------------------------------------------------------------------
    // 2. BASS (Harmonically Locked to Chord Progression Root)
    // -------------------------------------------------------------------------
    const bPat = this.activePatterns.bass;
    if (bPat && sec !== "BREAKDOWN" && !(sec === "BUILDUP" && barInSec >= 6)) {
      if (bPat.onset_grid.includes(step)) {
        let vel = bPat.velocities ? bPat.velocities[step] : 0.90;
        const dur = (bPat.groove_type === "rolling") ? 0.14 : 0.22;
        const subSlice = this.getBestSlice(bPat.groove_type === "rolling" ? "bass:rolling" : "bass:sub", chord.rootPitch);
        if (subSlice) {
          this.engine.scheduleSlice("bass", subSlice.buffer, when, { rate: subSlice.rate, fadeMs: 4 });
        }
        this.synth.playSub(when, vel, chord.bassHz, dur, bDest);
      }
    }

    // -------------------------------------------------------------------------
    // 3. SYNTHS / OTHER (Full Harmonic Triads & Arpeggios)
    // -------------------------------------------------------------------------
    const oPat = this.activePatterns.other;
    if (oPat) {
      if (oPat.onset_grid.includes(step)) {
        const vel = oPat.velocities ? oPat.velocities[step] : 0.85;

        if (oPat.groove_type === "riff" || sec === "DROP") {
          // Melodic phrase contour across the 8-bar musical arc
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
          this.synth.playArp(when, vel, arpFreq, oDest);

          if (step % 4 === 0) {
            const arpSlice = this.getBestSlice("other:arp", chord.rootPitch);
            if (arpSlice) this.engine.scheduleSlice("other", arpSlice.buffer, when, { rate: arpSlice.rate, fadeMs: 4 });
          }
        } else {
          // Polyphonic Synth Chord on upbeats (2, 6, 10, 14)
          const dur = (sec === "BREAKDOWN" || sec === "INTRO") ? 0.45 : 0.28;
          const chordSlice = this.getBestSlice("other:chord", chord.rootPitch);
          if (chordSlice) {
            this.engine.scheduleSlice("other", chordSlice.buffer, when, { rate: chordSlice.rate, fadeMs: 4 });
          }
          this.synth.playChord(when, vel, chord.chordHz, chord.isMinor, dur, oDest);
        }
      }
    }

    // -------------------------------------------------------------------------
    // 4. VOCALS (Singing Melodic Song Engine & Rhythmic Chops)
    // -------------------------------------------------------------------------
    const vPat = this.activePatterns.vocals;
    const isDropGap = (sec === "BUILDUP" && phraseBar === 7 && (step % 16) === 15);

    if (!isDropGap && vPat) {
      // 1. Check if the song engine has a lyrical singing syllable on this step
      const syllable = this.vocalSongArranger.getSyllableForStep(sec, phraseBar, step % 16);

      if (syllable || vPat.onset_grid.includes(step % 16)) {
        // Fallback syllable if triggered by pattern onset
        const syl: VocalSyllable = syllable || {
          text: (sec === "DROP") ? "FEEL" : "YEAH",
          vowel: "EE",
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

        // Store live vocal metadata for UI
        this.currentVocalMeta = {
          lyric: this.vocalSongArranger.getCurrentLyricLine(sec, phraseBar),
          syllable: syl.text,
          vowel: syl.vowel,
          note: noteName,
          songTitle: this.vocalSongArranger.getSongTitle(),
          sectionName: sec,
        };

        // Try acoustic slice sample first (phrase for long, chop for short)
        const isLong = syl.durationSteps >= 4 || sec === "BREAKDOWN" || sec === "INTRO";
        const slice = isLong
          ? (this.getBestSlice("vocals:phrase", melodyPitch) || this.getBestSlice("vocals:chop", melodyPitch))
          : (this.getBestSlice("vocals:chop", melodyPitch) || this.getBestSlice("vocals:phrase", melodyPitch));

        if (slice) {
          // Schedule real slice repitched to the singing melody note
          this.engine.scheduleSlice("vocals", slice.buffer, when, {
            rate: slice.rate,
            fadeMs: isLong ? 8 : 4,
          });
          // Also softly layer the singing formant body for authentic vocal formant clarity
          this.synth.playSingingVoice(when, noteHz, syl.vowel, Math.min(0.4, durSec), vel * 0.45, vInput);
        } else {
          // Autonomous singing formant synthesizer
          this.synth.playSingingVoice(when, noteHz, syl.vowel, durSec, vel, vInput);
        }
      }
    }
  }

  private composeBarPatterns(phraseBar: number): void {
    const sec = this.arranger.getSection();
    const isTurnaround = (phraseBar === 7);

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const currG = this.activePatterns[stem]?.groove_type || "four_on_floor_hats";
      const nextG = isTurnaround && stem === "drums"
        ? "turnaround_fill"
        : this.phraseModel.sampleNextGroove(phraseBar, stem, currG);

      this.activePatterns[stem] = this.synthesizePattern(stem, nextG);
    }
  }

  private synthesizePattern(stem: StemName, grooveType: string): SerializedPattern {
    // 1. Try to find matching pattern in standalone manifest
    if (this.manifest && this.manifest.patterns && this.manifest.patterns[stem]) {
      const match = this.manifest.patterns[stem].find((p) => p.groove_type === grooveType);
      if (match) return match;
    }

    // 2. Synthesize dynamic groove pattern
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
      } else if (grooveType === "breakbeat") {
        for (const s of [0, 2, 4, 6, 10, 12, 14, 15]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 4 || s === 12) ? 1.0 : 0.75;
        }
      } else {
        // four_on_floor_hats (standard driving club pattern)
        for (let s = 0; s < 16; s++) {
          onset_grid.push(s);
          if (s % 4 === 0) velocities[s] = 1.0;       // Kicks
          else if (s % 2 === 0) velocities[s] = 0.88; // Open Hats
          else velocities[s] = 0.62;                  // 16th Closed Hats
        }
      }
    } else if (stem === "bass") {
      if (grooveType === "rolling") {
        for (let s = 0; s < 16; s++) {
          onset_grid.push(s);
          velocities[s] = (s % 4 === 0) ? 0.70 : 0.90;
        }
      } else if (grooveType === "stabs") {
        for (const s of [0, 3, 6, 10, 12]) {
          onset_grid.push(s);
          velocities[s] = 0.90;
        }
      } else {
        // offbeat pump on 2, 6, 10, 14
        for (const s of [2, 6, 10, 14, 3, 7, 11, 15]) {
          onset_grid.push(s);
          velocities[s] = (s % 4 === 2) ? 1.0 : 0.65;
        }
      }
    } else if (stem === "other") {
      if (grooveType === "riff") {
        for (let s = 0; s < 16; s += 2) {
          onset_grid.push(s);
          velocities[s] = (s % 4 === 0) ? 0.95 : 0.75;
        }
      } else {
        // chords on upbeats 2, 6, 10, 14
        for (const s of [2, 6, 10, 14]) {
          onset_grid.push(s);
          velocities[s] = 0.90;
        }
      }
    } else {
      // vocals: multi-template dynamic patterns
      if (grooveType === "vocal_hook") {
        for (const s of [0, 3, 6, 8, 10, 12, 14]) {
          onset_grid.push(s);
          velocities[s] = (s === 0 || s === 14) ? 1.0 : 0.92;
        }
      } else if (grooveType === "vocal_call_response") {
        for (const s of [2, 4, 7, 10, 12, 14]) {
          onset_grid.push(s);
          velocities[s] = (s % 2 === 0) ? 0.92 : 0.85;
        }
      } else if (grooveType === "vocal_syncopated_chop") {
        for (const s of [1, 4, 7, 10, 13]) {
          onset_grid.push(s);
          velocities[s] = 0.92;
        }
      } else if (grooveType === "vocal_stutter") {
        for (let s = 0; s < 16; s += 2) {
          onset_grid.push(s);
          velocities[s] = 0.80 + (s / 16) * 0.20;
        }
      } else if (grooveType === "vocal_sustained" || grooveType === "phrase") {
        for (const s of [0, 8]) {
          onset_grid.push(s);
          velocities[s] = 0.95;
        }
      } else if (grooveType === "vocal_verse") {
        for (const s of [0, 3, 6, 11]) {
          onset_grid.push(s);
          velocities[s] = 0.90;
        }
      } else {
        // legacy chop
        for (const s of [2, 6, 14]) {
          onset_grid.push(s);
          velocities[s] = 0.92;
        }
      }
    }

    return {
      id: `pat_${stem}_${grooveType}`,
      name: `${stem}_${grooveType}`,
      groove_type: grooveType,
      energy: 0.85,
      density: onset_grid.length / 16,
      onset_grid,
      velocities,
      length: 16,
      key: "8A",
    };
  }

  private initDefaultPatterns(): void {
    this.activePatterns.drums = this.synthesizePattern("drums", "four_on_floor_hats");
    this.activePatterns.bass = this.synthesizePattern("bass", "offbeat");
    this.activePatterns.other = this.synthesizePattern("other", "chords");
    this.activePatterns.vocals = this.synthesizePattern("vocals", "vocal_verse");
  }

  private buildMeta(stepInPattern: number, patternInCycle: number, phraseBar: number): any {
    const sec = this.arranger.getSection();
    const isFill = (phraseBar === 7);
    const chord = this.activeProgression[phraseBar % this.activeProgression.length];

    const patterns: Record<string, string> = {};
    const patternGrooves: Record<string, string> = {};
    const patternGrids: Record<string, number[]> = {};
    const patternAudioScores: Record<string, number> = {};

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const p = this.activePatterns[stem];
      if (p) {
        patterns[stem] = `KG-505 [${chord.name}] (${p.groove_type})`;
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
      direct_ml_engine: true,
    };
  }
}
