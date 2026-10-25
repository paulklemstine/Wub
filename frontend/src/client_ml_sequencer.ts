// Client-Side ML Pattern Sequencer and Autonomous Groovebox Engine
//
// Enables 100% standalone, in-browser operation without requiring a live WebSocket server.
// Directly loads pre-trained Markov phrase models, song arrangers, and acoustic pattern banks,
// reconstructing the exact dance patterns and scheduling sample-accurate Web Audio playback.

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
  track_id: string;
  segment_id: string;
}

export interface StandaloneManifest {
  version: number;
  default_bpm: number;
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

// -----------------------------------------------------------------------------
// 1. Client-Side Web Audio Algorithmic Slice Synthesizer
// -----------------------------------------------------------------------------

export class WebAudioSliceSynthesizer {
  private ctx: AudioContext;
  private noiseBuffer: AudioBuffer | null = null;
  private loadedSliceBuffers: Map<string, AudioBuffer> = new Map();

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

  // Authentic 808/909 Kick transient and body
  public playKick(when: number, velocity: number = 1.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sine";
    // Exponential pitch drop from 160 Hz to 46 Hz
    osc.frequency.setValueAtTime(160, t);
    osc.frequency.exponentialRampToValueAtTime(46, t + 0.08);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 1.2, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.28);

    osc.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + 0.30);
  }

  // Crisp Snare / Clap transient
  public playSnare(when: number, velocity: number = 1.0, destNode?: AudioNode): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    // Body tone
    const osc = this.ctx.createOscillator();
    const oscGain = this.ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(185, t);
    osc.frequency.exponentialRampToValueAtTime(80, t + 0.05);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    oscGain.gain.setValueAtTime(v * 0.7, t);
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
      filter.frequency.value = 1800;
      filter.Q.value = 1.2;

      const noiseGain = this.ctx.createGain();
      noiseGain.gain.setValueAtTime(v * 0.9, t);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, t + 0.20);

      noise.connect(filter);
      filter.connect(noiseGain);
      noiseGain.connect(dest);

      noise.start(t);
      noise.stop(t + 0.22);
    }
  }

  // Sizzling Open / Closed Hi-Hat
  public playHat(when: number, velocity: number = 1.0, isOpen: boolean = false, destNode?: AudioNode): void {
    if (!this.noiseBuffer) return;
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const noise = this.ctx.createBufferSource();
    noise.buffer = this.noiseBuffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = "highpass";
    filter.frequency.value = 7200;

    const dur = isOpen ? 0.22 : 0.05;
    const v = Math.max(0.1, Math.min(1.0, velocity));

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(v * 0.65, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(dest);

    noise.start(t);
    noise.stop(t + dur + 0.02);
  }

  // Deep Sub / Rolling Bass
  public playSub(
    when: number,
    velocity: number = 1.0,
    pitchHz: number = 55.0,
    durationSec: number = 0.20,
    destNode?: AudioNode
  ): void {
    const t = Math.max(this.ctx.currentTime, when);
    const dest = destNode || this.ctx.destination;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(pitchHz, t);

    // Warm low-pass filter
    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(380, t);
    filter.frequency.exponentialRampToValueAtTime(140, t + durationSec);

    const v = Math.max(0.1, Math.min(1.0, velocity));
    gain.gain.setValueAtTime(v * 0.85, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + durationSec);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(dest);

    osc.start(t);
    osc.stop(t + durationSec + 0.02);
  }

  // Melodic Polyphonic Detuned Saw Synth Chords
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

    // Root, 3rd (minor 3 semitones or major 4), and 5th (7 semitones)
    const thirdMult = isMinor ? Math.pow(2, 3 / 12) : Math.pow(2, 4 / 12);
    const fifthMult = Math.pow(2, 7 / 12);
    const freqs = [rootHz, rootHz * thirdMult, rootHz * fifthMult];

    const chordGain = this.ctx.createGain();
    const v = Math.max(0.1, Math.min(1.0, velocity));
    chordGain.gain.setValueAtTime(v * 0.45, t);
    chordGain.gain.exponentialRampToValueAtTime(0.001, t + durationSec);

    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(2200, t);
    filter.frequency.exponentialRampToValueAtTime(700, t + durationSec);

    chordGain.connect(filter);
    filter.connect(dest);

    for (const f of freqs) {
      const osc = this.ctx.createOscillator();
      osc.type = "sawtooth";
      osc.frequency.value = f;
      osc.connect(chordGain);
      osc.start(t);
      osc.stop(t + durationSec + 0.02);
    }
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
    this.sectionDuration = (nextSec === "BUILDUP" || nextSec === "INTRO") ? 8 : 16;
  }

  public setSection(section: string): void {
    this.currentSection = section;
    this.barInSection = 0;
    this.sectionDuration = (section === "BUILDUP" || section === "INTRO") ? 8 : 16;
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
    this.synth = new WebAudioSliceSynthesizer(engine.ensureCtx());
    this.phraseModel = new ClientMarkovPhraseModel();
    this.arranger = new ClientSongArranger();
    this.initDefaultPatterns();
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
        if (list.length > 0) {
          const item = list[0];
          try {
            const url = item.url.startsWith("http") ? item.url : `${basePath}/${item.url.replace(/^data\//, "")}`;
            const res = await fetch(url);
            if (res.ok) {
              const bytes = await res.arrayBuffer();
              const decoded = await this.engine.decodeWav(bytes);
              this.synth.registerSliceBuffer(`${stem}:${cat}`, decoded);
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
      // Immediate smart response on thumbs down: reroll stem pattern immediately!
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

    // Initialize phrase patterns
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
    this.arranger.advanceBar();
  }

  private scheduleStepAudio(step: number, when: number, bpm: number): void {
    // 1. Drums
    const dPat = this.activePatterns.drums;
    const dInput = this.engine.getStemInput("drums");
    if (dPat && dPat.onset_grid.includes(step)) {
      const vel = dPat.velocities ? dPat.velocities[step] : 1.0;
      if (step === 0 || step === 4 || step === 8 || step === 12 || (step === 10 && dPat.groove_type.includes("syncopated"))) {
        // Kick hit
        const kickBuf = this.synth.getSliceBuffer("drums:kick");
        if (kickBuf) this.engine.scheduleSlice("drums", kickBuf, when, { rate: 1.0, fadeMs: 3 });
        else this.synth.playKick(when, vel, dInput);
      }
      if (step === 4 || step === 12 || (step >= 8 && dPat.groove_type === "turnaround_fill" && step !== 15)) {
        // Snare hit
        const snareBuf = this.synth.getSliceBuffer("drums:snare");
        if (snareBuf) this.engine.scheduleSlice("drums", snareBuf, when, { rate: 1.0, fadeMs: 3 });
        else this.synth.playSnare(when, vel, dInput);
      }
      if (step % 2 === 0) {
        // Open Hat on upbeats
        const hatBuf = this.synth.getSliceBuffer("drums:hat_open");
        if (hatBuf) this.engine.scheduleSlice("drums", hatBuf, when, { rate: 1.0, fadeMs: 3 });
        else this.synth.playHat(when, vel, true, dInput);
      } else {
        // 16th Closed Hat
        const hatBuf = this.synth.getSliceBuffer("drums:hat_closed");
        if (hatBuf) this.engine.scheduleSlice("drums", hatBuf, when, { rate: 1.0, fadeMs: 2 });
        else this.synth.playHat(when, vel * 0.7, false, dInput);
      }
    }

    // 2. Bass (with simulated sidechain ducking on kick steps)
    const bPat = this.activePatterns.bass;
    const bInput = this.engine.getStemInput("bass");
    if (bPat && bPat.onset_grid.includes(step)) {
      let vel = bPat.velocities ? bPat.velocities[step] : 1.0;
      // Duck bass slightly if kick hits simultaneously
      if (step === 0 || step === 4 || step === 8 || step === 12) {
        vel *= 0.65;
      }
      const subBuf = this.synth.getSliceBuffer("bass:sub");
      if (subBuf) this.engine.scheduleSlice("bass", subBuf, when, { rate: 1.0, fadeMs: 4 });
      else this.synth.playSub(when, vel, 55.0, 0.22, bInput);
    }

    // 3. Other (Synths / Chords / Arp)
    const oPat = this.activePatterns.other;
    const oInput = this.engine.getStemInput("other");
    if (oPat && oPat.onset_grid.includes(step)) {
      const vel = oPat.velocities ? oPat.velocities[step] : 0.85;
      const chordBuf = this.synth.getSliceBuffer("other:chord");
      if (chordBuf) this.engine.scheduleSlice("other", chordBuf, when, { rate: 1.0, fadeMs: 5 });
      else this.synth.playChord(when, vel, 130.81, true, 0.32, oInput);
    }

    // 4. Vocals
    const vPat = this.activePatterns.vocals;
    const vInput = this.engine.getStemInput("vocals");
    if (vPat && vPat.onset_grid.includes(step)) {
      const vel = vPat.velocities ? vPat.velocities[step] : 0.90;
      const chopBuf = this.synth.getSliceBuffer("vocals:chop");
      if (chopBuf) this.engine.scheduleSlice("vocals", chopBuf, when, { rate: 1.0, fadeMs: 4 });
      else this.synth.playVocalChop(when, vel, 800, vInput);
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
        // offbeat pump
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
        // chords
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

    const patterns: Record<string, string> = {};
    const patternGrooves: Record<string, string> = {};
    const patternGrids: Record<string, number[]> = {};
    const patternAudioScores: Record<string, number> = {};

    for (const stem of ["drums", "bass", "other", "vocals"] as StemName[]) {
      const p = this.activePatterns[stem];
      if (p) {
        patterns[stem] = `KG-505 Direct ML (${p.groove_type})`;
        patternGrooves[stem] = p.groove_type;
        const grid = new Array(this.stepsPerPattern).fill(0);
        for (const idx of p.onset_grid) {
          if (idx < this.stepsPerPattern) grid[idx] = 1;
        }
        patternGrids[stem] = grid;
        patternAudioScores[stem] = 0.82;
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
