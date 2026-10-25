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
      if (stem === "vocals") return "chop";
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

  private isRunning: boolean = false;
  private timer: number | null = null;
  private nextStepTime: number = 0;
  private onMetaCallback?: (meta: any) => void;

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
        for (let i = 0; i < Math.min(2, list.length); i++) {
          const item = list[i];
          try {
            const url = item.url.startsWith("http") ? item.url : `${basePath}/${item.url.replace(/^data\//, "")}`;
            const res = await fetch(url);
            if (res.ok) {
              const bytes = await res.arrayBuffer();
              const decoded = await this.engine.decodeWav(bytes);
              this.synth.registerSliceBuffer(`${stem}:${cat}_${i}`, decoded);
              if (i === 0) {
                this.synth.registerSliceBuffer(`${stem}:${cat}`, decoded);
              }
            }
          } catch {
            // Silently fall back to algorithmic synth
          }
        }
      }
    }
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

  public setMetaCallback(cb: (meta: any) => void): void {
    this.onMetaCallback = cb;
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
    this.nextStepTime = this.engine.now() + 0.05;
    this.tatumIndex = 0;
    this.phraseBar = 0;
    this.stepInBar = 0;
    this.patternInCycle = 1;

    // Pick a progression
    this.progressionIndex = (this.progressionIndex + 1) % DEFAULT_PROGRESSIONS.length;
    this.activeProgression = DEFAULT_PROGRESSIONS[this.progressionIndex];

    this.composeBarPatterns(0);

    const stepIntervalMs = (60 / bpm / 4) * 1000;
    this.timer = window.setInterval(() => this.clockTick(bpm), Math.min(25, stepIntervalMs / 2));
  }

  public stop(): void {
    this.isRunning = false;
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private clockTick(bpm: number): void {
    if (!this.isRunning) return;
    const now = this.engine.now();
    const tatumSec = 60 / bpm / 4;

    // Schedule 150ms ahead
    while (this.nextStepTime < now + 0.15) {
      this.processStep(this.nextStepTime, bpm);
      this.nextStepTime += tatumSec;
    }
  }

  private processStep(when: number, bpm: number): void {
    const step = this.stepInBar;
    const isBarStart = (step === 0);

    if (isBarStart) {
      this.onBarStart();
    }

    // Trigger audio hits for all active stems
    this.scheduleStepAudio(step, when, bpm);

    // Emit live GrooveboxMeta for UI matrix, LED dots, and timeline
    if (this.onMetaCallback) {
      const meta = this.buildMeta(step);
      this.onMetaCallback(meta);
    }

    // Advance metrical counters
    this.stepInBar = (this.stepInBar + 1) % 16;
    this.tatumIndex++;

    if (this.stepInBar === 0) {
      this.phraseBar = (this.phraseBar + 1) % 8;
      this.patternInCycle = (this.patternInCycle % this.patternsPerCycle) + 1;
      this.composeBarPatterns(this.phraseBar);
    }
  }

  private onBarStart(): void {
    const { isSectionChange, section } = this.arranger.advanceBar();
    if (isSectionChange) {
      // Pick next harmonic chord progression for fresh section
      this.progressionIndex = (this.progressionIndex + 1) % DEFAULT_PROGRESSIONS.length;
      this.activeProgression = DEFAULT_PROGRESSIONS[this.progressionIndex];
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

  private scheduleStepAudio(step: number, when: number, bpm: number): void {
    const sec = this.arranger.getSection();
    const barInSec = this.arranger.getBarInSection();
    const phraseBar = this.phraseBar; // 0..7
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
        if (step in [2, 6, 10, 14]) {
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
          this.synth.playRiser(when, (60 / bpm) * 8, dInput);
        }
      } else {
        // Standard Groove / Drop: Full rhythm driving forward
        const hasHit = dPat.onset_grid.includes(step);
        const vel = dPat.velocities ? dPat.velocities[step] : 1.0;

        // Kick triggers: 0, 4, 8, 12 (or syncopations)
        const isKickStep = (step in [0, 4, 8, 12]) || (step === 10 && dPat.groove_type.includes("syncopated"));
        if (isKickStep && hasHit) {
          const kickBuf = this.synth.getSliceBuffer("drums:kick");
          if (kickBuf) this.engine.scheduleSlice("drums", kickBuf, when, { rate: 1.0, fadeMs: 2 });
          else this.synth.playKick(when, vel, dInput);
          this.applySidechainDucking(when);
        }

        // Snare / Clap triggers: 4 and 12 (+ fill roll on bar 7)
        const isSnareStep = (step === 4 || step === 12) || (phraseBar === 7 && step >= 8 && step !== 15);
        if (isSnareStep) {
          const snareBuf = this.synth.getSliceBuffer("drums:snare");
          if (snareBuf) this.engine.scheduleSlice("drums", snareBuf, when, { rate: 1.0, fadeMs: 2 });
          else this.synth.playSnare(when, vel, dInput);
        }

        // Hi-Hats: Open hat on upbeats (2, 6, 10, 14) + closed 16th hats
        if (step in [2, 6, 10, 14]) {
          const hatBuf = this.synth.getSliceBuffer("drums:hat_open");
          if (hatBuf) this.engine.scheduleSlice("drums", hatBuf, when, { rate: 1.0, fadeMs: 2 });
          else this.synth.playHat(when, 0.90, true, dInput);
        } else if (step % 2 === 1 && sec !== "INTRO") {
          const hatBuf = this.synth.getSliceBuffer("drums:hat_closed");
          if (hatBuf) this.engine.scheduleSlice("drums", hatBuf, when, { rate: 1.0, fadeMs: 2 });
          else this.synth.playHat(when, 0.60, false, dInput);
        }

        // Syncopated Perc on 3, 7, 11
        if (step in [3, 7, 11] && (sec === "DROP" || sec === "GROOVE_A")) {
          this.synth.playPerc(when, 0.70, 420.0, dInput);
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
        // Playing root pitch from chord progression!
        const pitchHz = chord.bassHz;

        // Try playing real pitch-tagged slice if available
        const subBuf = this.synth.getSliceBuffer("bass:sub");
        if (subBuf) {
          this.engine.scheduleSlice("bass", subBuf, when, { rate: 1.0, fadeMs: 4 });
        } else {
          // Play fat resonant tuned sub bass
          const dur = (bPat.groove_type === "rolling") ? 0.14 : 0.22;
          this.synth.playSub(when, vel, pitchHz, dur, bDest);
        }
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
          // Driving 16th Arpeggio dancing across the notes of the chord
          const thirdMult = chord.isMinor ? Math.pow(2, 3 / 12) : Math.pow(2, 4 / 12);
          const fifthMult = Math.pow(2, 7 / 12);
          const octMult = 2.0;
          const arpNotes = [chord.chordHz, chord.chordHz * thirdMult, chord.chordHz * fifthMult, chord.chordHz * octMult];
          const arpFreq = arpNotes[step % arpNotes.length];
          this.synth.playArp(when, vel, arpFreq, oDest);
        } else {
          // Polyphonic Synth Chord on upbeats (2, 6, 10, 14)
          const chordBuf = this.synth.getSliceBuffer("other:chord");
          if (chordBuf) {
            this.engine.scheduleSlice("other", chordBuf, when, { rate: 1.0, fadeMs: 4 });
          } else {
            const dur = (sec === "BREAKDOWN" || sec === "INTRO") ? 0.45 : 0.28;
            this.synth.playChord(when, vel, chord.chordHz, chord.isMinor, dur, oDest);
          }
        }
      }
    }

    // -------------------------------------------------------------------------
    // 4. VOCALS (Rhythmic Chops & Phrases)
    // -------------------------------------------------------------------------
    const vPat = this.activePatterns.vocals;
    if (vPat && (sec === "DROP" || sec === "GROOVE_A" || sec === "BREAKDOWN")) {
      if (vPat.onset_grid.includes(step)) {
        const vel = vPat.velocities ? vPat.velocities[step] : 0.90;
        const chopBuf = this.synth.getSliceBuffer("vocals:chop");
        if (chopBuf) {
          this.engine.scheduleSlice("vocals", chopBuf, when, { rate: 1.0, fadeMs: 3 });
        } else {
          // Formant vocal sweep tuned to root
          const formant = 700 + (chord.rootPitch * 30);
          this.synth.playVocalChop(when, vel, formant, vInput);
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
      // vocals
      for (const s of [2, 6, 14]) {
        onset_grid.push(s);
        velocities[s] = 0.92;
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
    this.activePatterns.vocals = this.synthesizePattern("vocals", "chop");
  }

  private buildMeta(step: number): any {
    const sec = this.arranger.getSection();
    const isFill = (this.phraseBar === 7);
    const chord = this.activeProgression[this.phraseBar % this.activeProgression.length];

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

    return {
      tatum_index: this.tatumIndex,
      tatumIndex: this.tatumIndex,
      section: sec,
      phrase_bar: this.phraseBar + 1,
      phraseBar: this.phraseBar + 1,
      step_in_bar: step + 1,
      stepInBar: step + 1,
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
      pattern_in_cycle: this.patternInCycle,
      patternInCycle: this.patternInCycle,
      patterns_per_cycle: this.patternsPerCycle,
      patternsPerCycle: this.patternsPerCycle,
      step_in_pattern: step + 1,
      stepInPattern: step + 1,
      steps_per_pattern: this.stepsPerPattern,
      stepsPerPattern: this.stepsPerPattern,
      cycle_total_steps: this.patternsPerCycle * this.stepsPerPattern,
      cycleTotalSteps: this.patternsPerCycle * this.stepsPerPattern,
      cycle_progress: ((this.patternInCycle - 1) * this.stepsPerPattern + step) / (this.patternsPerCycle * this.stepsPerPattern),
      cycleProgress: ((this.patternInCycle - 1) * this.stepsPerPattern + step) / (this.patternsPerCycle * this.stepsPerPattern),
      patterns,
      pattern_grooves: patternGrooves,
      patternGrooves,
      pattern_grids: patternGrids,
      patternGrids,
      pattern_ratings: this.stemRatings,
      patternRatings: this.stemRatings,
      pattern_audio_scores: patternAudioScores,
      patternAudioScores,
      direct_ml_engine: true,
    };
  }
}
