// Wub KG-505 Autonomous Generative Electronic Groovebox
// 100% Standalone In-Browser ML Audio Engine & Sequencer
//
// Expected DOM ids (in sync with frontend/index.html):
//   Playback:        btn-play, btn-stop, status-line
//   Master:          bpm-slider (80-175), bpm-value, key-select
//   Canvases:        rings-canvas, master-canvas, nebula-canvas, psychedelic-bg
//   Per stem <s>:    <s>-mute, <s>-solo, <s>-volume, <s>-vol-val,
//                    <s>-chaos, <s>-chaos-val, <s>-synth, <s>-synth-val
//   where <s> is one of drums | bass | vocals | other.

import { AudioEngine, STEMS } from "./audio_engine";
import { NebulaRenderer, drawWaveformAnalysis, drawMasterAnalysis } from "./visualizer";
import type { SliceArcEvent, NebulaPoint, WaveformStemData } from "./visualizer";
import type { StemName } from "./types";
import { PsychedelicBackground } from "./psychedelic_bg";
import type { TripMode, TripPalette } from "./psychedelic_bg";
import { VERSION_INFO } from "./version";
import { ClientGrooveboxConductor } from "./client_ml_sequencer";
import { TamagotchiAvatar } from "./tamagotchi";

export interface StemState {
  stem: StemName;
  mute: boolean;
  solo: boolean;
  volume: number;
  chaos: number;
  synthBlend: number;
}

const BPM_MIN = 80;
const BPM_MAX = 175;
const DEFAULT_BPM = 96;
const RING_EVENT_CAP = 240;
const RING_WINDOW_SEC = 8;

export interface GrooveboxMeta {
  section: string;
  phraseBar: number;
  stepInBar: number;
  totalBars?: number;
  isFill: boolean;
  key?: string;
  camelotCode?: string;
  chordName?: string;
  energy?: number;
  sectionBarsElapsed?: number;
  sectionTotalBars?: number;
  nextSection?: string;
  sectionBarsLeft?: number;
  movementPhase?: string;
  isTransitionBar?: boolean;
  transitionType?: string;
  cycleMode?: string;
  patternInCycle?: number;
  patternsPerCycle?: number;
  stepInPattern?: number;
  stepsPerPattern?: number;
  cycleTotalSteps?: number;
  cycleProgress?: number;
  patterns: Record<string, string>;
  patternGrooves?: Record<string, string>;
  patternGrids?: Record<string, number[]>;
  patternTrackIds?: Record<string, string>;
  patternIds?: Record<string, string>;
  patternRatings?: Record<string, number>;
  patternAudioScores?: Record<string, number>;
  vocalLyric?: string;
  vocalSongTitle?: string;
  vocalSyllable?: string;
  vocalVowel?: string;
  vocalNote?: string;
  vocalPersona?: string;
  vocalTimbre?: string;
  activeSliceCoords?: Partial<Record<StemName, { x: number; y: number }>>;
  stemMixLevels?: Record<StemName, number>;
  musicScale?: {
    keyName: string;
    modeName: string;
    camelot: string;
    scaleRoot: number;
    scalePitches: number[];
    scaleNotes: string[];
    chordName: string;
    chordRoot: number;
    chordPitches: number[];
    chordNotes: string[];
    bassHz?: number;
    chordHz?: number;
    chordIndex?: number;
    totalChords?: number;
  };
  patternDetails?: Record<string, {
    id: string;
    name: string;
    grooveType: string;
    territory: string;
    energy: number;
    density: number;
    similarity: number;
    cohesion: number;
    onsetCount: number;
    velocities?: number[];
    goodnessScore?: number;
    isAccepted?: boolean;
    rejectionReasons?: string[];
  }>;
  adversarialGoodness?: Record<string, {
    score: number;
    isAccepted: boolean;
    threshold: number;
    spectralMatch: number;
    temporalMatch: number;
    crestMatch: number;
    harmonicMatch: number;
    structuralMatch: number;
    status: string;
    rejectionReasons: string[];
  }>;
}

interface UIEvent {
  atAudioSec: number;
  meta: GrooveboxMeta;
  step: number;
  totalSteps: number;
}

// ---------------------------------------------------------------------------
// DOM helpers.
// ---------------------------------------------------------------------------

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node === null) throw new Error(`Missing element #${id}`);
  return node as T;
}

function visualNow(): number {
  return performance.now() / 1000;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function asNumber(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

function asString(v: unknown, fallback: string): string {
  return typeof v === "string" ? v : fallback;
}

function asStem(v: unknown): StemName | null {
  return v === "drums" || v === "bass" || v === "vocals" || v === "other" ? v : null;
}

// Deterministic projection seeded PRNG for demo nebula points before static indices load
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function demoPoints(): NebulaPoint[] {
  const centers: Record<StemName, { x: number; y: number }> = {
    drums: { x: 0.35, y: 0.75 },
    bass: { x: 0.35, y: 0.75 },
    vocals: { x: 0.35, y: 0.72 },
    other: { x: 0.35, y: 0.72 },
  };
  const rand = mulberry32(1337);
  const pts: NebulaPoint[] = [];
  const gauss = (): number => (rand() + rand() + rand() - 1.5) * 0.22;
  for (const stem of STEMS) {
    const c = centers[stem];
    for (let i = 0; i < 120; i++) {
      pts.push({ stem, x: c.x + gauss(), y: c.y + gauss() });
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// App.
// ---------------------------------------------------------------------------

class WubApp {
  private engine = new AudioEngine();
  private nebula: NebulaRenderer | null = null;
  private ringsCtx: CanvasRenderingContext2D | null = null;
  private masterCtx: CanvasRenderingContext2D | null = null;
  private psychedelicBg: PsychedelicBackground | null = null;

  private states: Record<StemName, StemState> = {
    drums: { stem: "drums", mute: false, solo: false, volume: 1, chaos: 0.1, synthBlend: 0.0 },
    bass: { stem: "bass", mute: false, solo: false, volume: 1, chaos: 0.3, synthBlend: 0.0 },
    vocals: { stem: "vocals", mute: false, solo: false, volume: 1, chaos: 0.5, synthBlend: 0.0 },
    other: { stem: "other", mute: false, solo: false, volume: 1, chaos: 0.5, synthBlend: 0.0 },
  };

  private currentDisplayVolumes: Record<StemName, number> = {
    drums: 1.0,
    bass: 1.0,
    vocals: 1.0,
    other: 1.0,
  };

  private bpm: number = DEFAULT_BPM;
  private musicalKey: string = "C";
  private playing = false;

  private segmentPositions = new Map<string, { x: number; y: number }>();
  private trackStemPositions = new Map<string, { x: number; y: number }>();
  private timeScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private freqScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private masterTimeScratch: Uint8Array<ArrayBuffer> | null = null;
  private masterFreqScratch: Uint8Array<ArrayBuffer> | null = null;
  private ringEvents: SliceArcEvent[] = [];
  private analyserScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private statusEl!: HTMLElement;
  private lastGrooveboxMeta: GrooveboxMeta | null = null;
  private clientConductor: ClientGrooveboxConductor | null = null;
  private tamagotchi: TamagotchiAvatar | null = null;
  private uiEventQueue: UIEvent[] = [];
  private currentStepIndex: number = 0;
  private currentStepAudioTime: number = 0;
  private totalStepsInPattern: number = 16;

  init(): void {
    this.statusEl = el<HTMLElement>("status-line");

    // Display version info from version.ts
    const versionBadge = document.getElementById("mc-version-badge");
    if (versionBadge) {
      versionBadge.innerHTML = `v${VERSION_INFO.version} <span class="mc-build-num">#${VERSION_INFO.build}</span>`;
      versionBadge.title = `KG-505 Build #${VERSION_INFO.build} • Rev ${VERSION_INFO.commit} • Built ${VERSION_INFO.timestamp}`;
    }
    const footerVersion = document.getElementById("footer-version-tag");
    if (footerVersion) {
      footerVersion.textContent = `v${VERSION_INFO.version} (Build #${VERSION_INFO.build} • rev ${VERSION_INFO.commit})`;
    }
    console.info(`[KG-505] Initialized v${VERSION_INFO.version} (Build #${VERSION_INFO.build} • rev ${VERSION_INFO.commit})`);

    const ringsCanvas = el<HTMLCanvasElement>("rings-canvas");
    const rings = ringsCanvas.getContext("2d");
    if (rings === null) throw new Error("2D context unavailable for #rings-canvas");
    this.ringsCtx = rings;

    const masterCanvas = el<HTMLCanvasElement>("master-canvas");
    const master = masterCanvas.getContext("2d");
    if (master === null) throw new Error("2D context unavailable for #master-canvas");
    this.masterCtx = master;

    this.nebula = new NebulaRenderer(el<HTMLCanvasElement>("nebula-canvas"), {
      onReticleMove: (stem, x, y) => {
        this.clientConductor?.setReticleSteering(stem, x, y, true);
      },
    });
    this.nebula.setPoints(demoPoints());
    void this.loadUmapPoints();

    try {
      this.psychedelicBg = new PsychedelicBackground("psychedelic-bg");
      this.bindTripController();
    } catch (e) {
      console.warn("Psychedelic background init failed:", e);
    }

    try {
      this.tamagotchi = new TamagotchiAvatar();
      this.tamagotchi.initDOM();
    } catch (e) {
      console.warn("Tamagotchi init failed:", e);
    }

    this.bindTransport();
    this.bindMaster();
    for (const stem of STEMS) this.bindStem(stem);
    this.bindFeedbackControls();
    this.bindGrooveboxCycleControls();
    this.bindPlaybackModeControls();

    // Initialize Autonomous In-Browser ML Sequencer
    try {
      this.clientConductor = new ClientGrooveboxConductor(this.engine);
      if (this.segmentPositions.size > 0) {
        this.clientConductor.setSegmentPositions(this.segmentPositions);
      }
      for (const stem of STEMS) {
        const r = this.nebula.getReticle(stem);
        this.clientConductor.setReticleSteering(stem, r.x, r.y, false);
      }
      void this.clientConductor.loadStaticBundle("data").then(() => {
        if (this.segmentPositions.size > 0 && this.clientConductor) {
          this.clientConductor.setSegmentPositions(this.segmentPositions);
        }
      });
      this.clientConductor.setMetaCallback((meta, atAudioSec, step, totalSteps) => {
        this.enqueueUIEvent({
          atAudioSec,
          meta,
          step,
          totalSteps: totalSteps || 16,
        });
      });
    } catch (e) {
      console.warn("Client ML sequencer init failed:", e);
    }

    // Sticky BPM: load from localStorage if set, else default to 96 BPM
    const savedBpm = localStorage.getItem("wub_bpm");
    if (savedBpm) {
      const parsed = Number(savedBpm);
      if (Number.isFinite(parsed) && parsed >= BPM_MIN && parsed <= BPM_MAX) {
        this.bpm = parsed;
        const slider = document.getElementById("bpm-slider");
        if (slider instanceof HTMLInputElement) slider.value = String(this.bpm);
        const label = document.getElementById("bpm-value");
        if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
        this.engine.setTargetBpm(this.bpm);
      }
    } else {
      this.bpm = DEFAULT_BPM;
    }

    window.addEventListener("resize", () => {
      if (this.nebula !== null) this.nebula.resize();
    });
    window.addEventListener("beforeunload", () => this.teardown());

    this.setStatus(`KG-505 In-Browser ML Ready • 24,068 Patterns • Offline Standalone (${Math.round(this.bpm)} BPM)`);
    el<HTMLButtonElement>("btn-play").classList.remove("active");
    el<HTMLButtonElement>("btn-stop").classList.add("active");
    requestAnimationFrame((t) => this.frame(t));
  }

  private async loadUmapPoints(): Promise<void> {
    try {
      const fallbackPts: unknown[] = [];
      for (const s of STEMS) {
        try {
          const fres = await fetch(`./data/indices/umap_${s}.json`);
          if (fres.ok) {
            const rows = (await fres.json()) as unknown;
            if (Array.isArray(rows)) fallbackPts.push(...rows);
          }
        } catch {
          // ignore
        }
      }

      if (fallbackPts.length === 0) return;
      const pts: NebulaPoint[] = [];
      for (const item of fallbackPts) {
        if (!isRecord(item)) continue;
        const stem = asStem(item["stem"]);
        const x = asNumber(item["x"], -1);
        const y = asNumber(item["y"], -1);
        const segId = asString(item["segment_id"], "");
        const trackId = asString(item["track_id"], "");
        if (stem !== null && x >= 0 && y >= 0) {
          pts.push({ stem, x, y, trackId: trackId || undefined });
          if (segId) {
            this.segmentPositions.set(segId, { x, y });
            if (trackId) {
              this.segmentPositions.set(`${trackId}:${segId}`, { x, y });
            }
          }
          if (trackId) {
            this.trackStemPositions.set(`${trackId}:${stem}`, { x, y });
          }
        }
      }
      if (pts.length > 0 && this.nebula !== null) {
        this.nebula.setPoints(pts);
        if (this.clientConductor) {
          this.clientConductor.setSegmentPositions(this.segmentPositions);
          this.clientConductor.setTrackStemPositions(this.trackStemPositions);
        }
      }
    } catch {
      // Fallback stays as seeded demo points
    }
  }

  private getSliceCoordinates(stem: StemName, trackId: string, segmentId: string): { x: number; y: number } {
    const direct = this.segmentPositions.get(segmentId) ?? this.segmentPositions.get(`${trackId}:${segmentId}`);
    if (direct) return direct;

    const trackStem = this.trackStemPositions.get(`${trackId}:${stem}`);
    if (trackStem) return trackStem;

    const centers: Record<StemName, { x: number; y: number }> = {
      drums: { x: 0.35, y: 0.75 },
      bass: { x: 0.35, y: 0.75 },
      vocals: { x: 0.35, y: 0.72 },
      other: { x: 0.35, y: 0.72 },
    };
    const c = centers[stem];
    let h = 0;
    const key = `${trackId}:${segmentId}`;
    for (let i = 0; i < key.length; i++) {
      h = (h * 31 + key.charCodeAt(i)) >>> 0;
    }
    const angle = (h % 628) / 100;
    const dist = 0.02 + ((h % 60) / 1000);
    return {
      x: Math.min(0.95, Math.max(0.05, c.x + Math.cos(angle) * dist)),
      y: Math.min(0.95, Math.max(0.05, c.y + Math.sin(angle) * dist)),
    };
  }

  // -- transport -----------------------------------------------------------

  private bindTransport(): void {
    el<HTMLButtonElement>("btn-play").addEventListener("click", () => {
      void this.play();
    });
    el<HTMLButtonElement>("btn-stop").addEventListener("click", () => {
      void this.stop();
    });
  }

  private async play(): Promise<void> {
    if (this.playing) return;
    this.playing = true;
    el<HTMLButtonElement>("btn-play").classList.add("active");
    el<HTMLButtonElement>("btn-stop").classList.remove("active");
    try {
      await this.engine.resume();
    } catch (err) {
      this.playing = false;
      el<HTMLButtonElement>("btn-play").classList.remove("active");
      el<HTMLButtonElement>("btn-stop").classList.add("active");
      this.setStatus(`Audio failed: ${err instanceof Error ? err.message : "unknown error"}`);
      return;
    }
    this.uiEventQueue = [];
    this.currentStepIndex = 0;
    this.currentStepAudioTime = 0;

    this.applyAllMixerState();

    this.clientConductor?.start(this.bpm);
    if (this.clientConductor) {
      this.bpm = this.clientConductor.getBpm();
      const slider = document.getElementById("bpm-slider");
      if (slider instanceof HTMLInputElement) slider.value = String(this.bpm);
      const label = document.getElementById("bpm-value");
      if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
      this.engine.setTargetBpm(this.bpm);
    }
    this.setStatus(`Playing — 🧠 Direct In-Browser ML Engine (Sample-Accurate Sequencer @ ${Math.round(this.bpm)} BPM)`);
  }

  private async stop(): Promise<void> {
    this.playing = false;
    el<HTMLButtonElement>("btn-play").classList.remove("active");
    el<HTMLButtonElement>("btn-stop").classList.add("active");
    this.clientConductor?.stop();
    this.uiEventQueue = [];
    this.currentStepIndex = 0;
    this.currentStepAudioTime = 0;
    await this.engine.suspend();
    this.setStatus("Stopped.");
  }

  private teardown(): void {
    if (this.nebula !== null) this.nebula.dispose();
    this.engine.dispose();
  }

  // -- controls ------------------------------------------------------------

  private bindMaster(): void {
    const slider = el<HTMLInputElement>("bpm-slider");
    const value = el<HTMLElement>("bpm-value");
    slider.min = String(BPM_MIN);
    slider.max = String(BPM_MAX);
    slider.value = String(this.bpm);
    value.textContent = `${Math.round(this.bpm)} BPM`;
    slider.addEventListener("input", () => {
      const next = Math.min(BPM_MAX, Math.max(BPM_MIN, Number(slider.value)));
      if (!Number.isFinite(next)) return;
      this.bpm = next;
      localStorage.setItem("wub_bpm", String(next));
      value.textContent = `${Math.round(next)} BPM`;
      this.engine.setTargetBpm(next);
      if (this.playing && this.clientConductor) {
        this.clientConductor.setBpm(next);
      }
    });

    const key = el<HTMLSelectElement>("key-select");
    this.musicalKey = key.value !== "" ? key.value : "C";
    key.addEventListener("change", () => {
      this.musicalKey = key.value;
    });
  }

  private bindTripController(): void {
    if (this.psychedelicBg === null) return;
    const bg = this.psychedelicBg;

    // Master Toggle
    const toggleBtn = document.getElementById("trip-toggle-btn") as HTMLButtonElement | null;
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const cur = bg.getState().enabled;
        const next = !cur;
        bg.setState({ enabled: next });
        toggleBtn.classList.toggle("active", next);
        toggleBtn.setAttribute("aria-pressed", String(next));
        toggleBtn.textContent = next ? "LIGHT SHOW: ON" : "LIGHT SHOW: OFF";
      });
    }

    // Mode Buttons
    const modeBtns = document.querySelectorAll<HTMLButtonElement>(".trip-mode-btn");
    modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.mode as TripMode;
        if (!mode) return;
        bg.setState({ mode });
        modeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      });
    });

    // Palette Buttons
    const palBtns = document.querySelectorAll<HTMLButtonElement>(".trip-pal-btn");
    palBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const palette = btn.dataset.palette as TripPalette;
        if (!palette) return;
        bg.setState({ palette });
        palBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      });
    });

    // Intensity Slider
    const intensitySlider = document.getElementById("trip-intensity-slider") as HTMLInputElement | null;
    const intensityVal = document.getElementById("trip-intensity-val");
    if (intensitySlider && intensityVal) {
      intensitySlider.addEventListener("input", () => {
        const val = Number(intensitySlider.value);
        bg.setState({ intensity: val });
        intensityVal.textContent = `${Math.round(val * 100)}%`;
      });
    }

    // Speed Slider
    const speedSlider = document.getElementById("trip-speed-slider") as HTMLInputElement | null;
    const speedVal = document.getElementById("trip-speed-val");
    if (speedSlider && speedVal) {
      speedSlider.addEventListener("input", () => {
        const val = Number(speedSlider.value);
        bg.setState({ speed: val });
        speedVal.textContent = `${val.toFixed(1)}x`;
      });
    }

    // Audio Reactive Checkbox
    const reactiveCheck = document.getElementById("trip-reactive-check") as HTMLInputElement | null;
    if (reactiveCheck) {
      reactiveCheck.addEventListener("change", () => {
        bg.setState({ audioReactive: reactiveCheck.checked });
      });
    }
  }

  private bindFeedbackControls(): void {
    const btns = document.querySelectorAll<HTMLButtonElement>(".thumb-btn");
    btns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const stem = btn.dataset.stem as StemName;
        if (!stem) return;
        const isUp = btn.classList.contains("thumb-up");
        const rating = isUp ? 1 : -1;

        // Visual flash feedback
        btn.classList.add("active");
        setTimeout(() => btn.classList.remove("active"), 350);

        this.sendFeedback(stem, rating);
      });
    });
  }

  private sendFeedback(stem: StemName, rating: 1 | -1): void {
    if (this.tamagotchi) {
      this.tamagotchi.recordFeedback(stem, rating);
    }
    if (this.clientConductor) {
      this.clientConductor.recordFeedback(stem, rating);
      if (rating < 0) {
        this.clientConductor.rerollStem(stem);
      }
    }

    const stemName = stem.toUpperCase();
    if (rating > 0) {
      this.setStatus(`👍 ${stemName} approved! Reinforced in training preference model.`);
    } else {
      this.setStatus(`👎 ${stemName} downvoted! Rerolling stem immediately to a fresh compatible pattern...`);
    }
  }

  private bindGrooveboxCycleControls(): void {
    const cycleBtns = document.querySelectorAll<HTMLButtonElement>(".gb-cycle-btn");
    cycleBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.mode;
        if (!mode) return;
        if (this.clientConductor) {
          this.clientConductor.setCycleMode(mode);
        }
        cycleBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.setStatus(`Sequencer cycle set to ${mode.replace("x", " × ")} architecture`);
      });
    });
  }

  private bindPlaybackModeControls(): void {
    const modeBtns = document.querySelectorAll<HTMLButtonElement>("[data-playback]");
    modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.playback as "jukebox" | "slices" | "phrase_blocks" | undefined;
        if (!mode) return;
        if (this.clientConductor) {
          this.clientConductor.setPlaybackMode(mode);
        }
        modeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const label =
          mode === "jukebox"
            ? "Jukebox Groove (Studio Loops + Groovy Slices & Vocal Chops)"
            : mode === "phrase_blocks"
            ? "Master Phrases (4-Bar Cohesive Studio Blocks)"
            : "Live Slices (16th-Step Micro-Slice Re-sequencer)";
        this.setStatus(`Engine Mode: ${label}`);
      });
    });
  }

  private bindStem(stem: StemName): void {
    const st = this.states[stem];
    const muteBtn = el<HTMLButtonElement>(`${stem}-mute`);
    const soloBtn = el<HTMLButtonElement>(`${stem}-solo`);
    const vol = el<HTMLInputElement>(`${stem}-volume`);
    const volVal = el<HTMLElement>(`${stem}-vol-val`);
    const chaos = el<HTMLInputElement>(`${stem}-chaos`);
    const chaosVal = el<HTMLElement>(`${stem}-chaos-val`);
    const synth = el<HTMLInputElement>(`${stem}-synth`);
    const synthVal = el<HTMLElement>(`${stem}-synth-val`);

    vol.value = String(st.volume);
    volVal.textContent = st.volume.toFixed(2);
    chaos.value = String(st.chaos);
    chaosVal.textContent = st.chaos.toFixed(2);
    synth.value = String(st.synthBlend);
    synthVal.textContent = st.synthBlend.toFixed(2);

    muteBtn.addEventListener("click", () => {
      st.mute = !st.mute;
      this.engine.setMute(stem, st.mute);
      muteBtn.classList.toggle("active", st.mute);
      muteBtn.setAttribute("aria-pressed", String(st.mute));
    });
    soloBtn.addEventListener("click", () => {
      st.solo = !st.solo;
      this.engine.setSolo(stem, st.solo);
      soloBtn.classList.toggle("active", st.solo);
      soloBtn.setAttribute("aria-pressed", String(st.solo));
    });
    vol.addEventListener("input", () => {
      const v = Number(vol.value);
      if (!Number.isFinite(v)) return;
      st.volume = Math.min(1.5, Math.max(0, v));
      volVal.textContent = st.volume.toFixed(2);
      this.engine.setVolume(stem, st.volume);
    });
    chaos.addEventListener("input", () => {
      const c = Number(chaos.value);
      if (!Number.isFinite(c)) return;
      st.chaos = Math.min(1, Math.max(0, c));
      chaosVal.textContent = st.chaos.toFixed(2);
    });
    synth.addEventListener("input", () => {
      const s = Number(synth.value);
      if (!Number.isFinite(s)) return;
      st.synthBlend = Math.min(1, Math.max(0, s));
      synthVal.textContent = st.synthBlend.toFixed(2);
    });
  }

  private applyAllMixerState(): void {
    for (const stem of STEMS) {
      const st = this.states[stem];
      this.engine.setVolume(stem, st.volume);
      this.engine.setMute(stem, st.mute);
      this.engine.setSolo(stem, st.solo);
    }
    this.engine.setTargetBpm(this.bpm);
  }

  private enqueueUIEvent(ev: UIEvent): void {
    this.uiEventQueue.push(ev);
    if (this.uiEventQueue.length > 128) {
      this.uiEventQueue.splice(0, this.uiEventQueue.length - 128);
    }
  }

  private pushRingEvent(ev: SliceArcEvent): void {
    this.ringEvents.push(ev);
    if (this.ringEvents.length > RING_EVENT_CAP) {
      this.ringEvents.splice(0, this.ringEvents.length - RING_EVENT_CAP);
    }
  }

  private updateGrooveboxUI(gb: GrooveboxMeta): void {
    const curPattern = gb.patternInCycle ?? gb.phraseBar ?? 1;
    const totalPatterns = gb.patternsPerCycle ?? 8;
    const curStep = gb.stepInPattern ?? gb.stepInBar ?? 1;
    const totalSteps = gb.stepsPerPattern ?? 16;

    // 0. Active Cycle Mode Button
    if (gb.cycleMode) {
      const cycleBtns = document.querySelectorAll<HTMLButtonElement>(".gb-cycle-btn");
      cycleBtns.forEach((btn) => {
        if (btn.dataset.mode === gb.cycleMode) btn.classList.add("active");
        else btn.classList.remove("active");
      });
    }

    // 1. Step in Pattern Sequencer LED strip (16 or 32 steps)
    const stepStrip = document.getElementById("gb-step-strip");
    if (stepStrip) {
      if (totalSteps === 32) stepStrip.classList.add("mode-32");
      else stepStrip.classList.remove("mode-32");

      if (stepStrip.children.length !== totalSteps) {
        stepStrip.innerHTML = "";
        for (let s = 1; s <= totalSteps; s++) {
          const isBeat = (s - 1) % 4 === 0;
          const stepDiv = document.createElement("div");
          stepDiv.id = `gb-step-${s}`;
          stepDiv.className = isBeat ? "gb-step beat" : "gb-step";
          stepDiv.innerHTML = `<span class="gb-step-led"></span>${s}`;
          stepStrip.appendChild(stepDiv);
        }
      }
      for (let s = 1; s <= totalSteps; s++) {
        const stepEl = stepStrip.children[s - 1] as HTMLElement;
        if (stepEl) {
          if (s === curStep) stepEl.classList.add("active");
          else stepEl.classList.remove("active");
        }
      }
    }

    // 2. Pattern in Cycle Counter & LED dots
    const barNumEl = document.getElementById("gb-bar-num");
    if (barNumEl) barNumEl.textContent = String(curPattern);
    const patPerCycleEl = document.getElementById("gb-patterns-per-cycle");
    if (patPerCycleEl) patPerCycleEl.textContent = String(totalPatterns);

    const dotsContainer = document.getElementById("gb-bars-dots");
    if (dotsContainer) {
      if (dotsContainer.children.length !== totalPatterns) {
        dotsContainer.innerHTML = "";
        for (let b = 1; b <= totalPatterns; b++) {
          const dot = document.createElement("div");
          dot.id = `gb-dot-${b}`;
          dot.className = "gb-bar-dot";
          dotsContainer.appendChild(dot);
        }
      }
      for (let b = 1; b <= totalPatterns; b++) {
        const dot = dotsContainer.children[b - 1] as HTMLElement;
        if (dot) {
          if (b === curPattern) {
            dot.className = gb.isFill ? "gb-bar-dot fill" : "gb-bar-dot active";
          } else if (b < curPattern) {
            dot.className = "gb-bar-dot active";
          } else {
            dot.className = "gb-bar-dot";
          }
        }
      }
    }

    // 3. Section badge
    const secBadge = document.getElementById("gb-section-badge");
    if (secBadge) {
      secBadge.textContent = gb.section.replace("_", " ");
      const secColors: Record<string, string> = {
        DROP: "#ea580c",
        GROOVE_A: "#7c3aed",
        BUILDUP: "#ca8a04",
        BREAKDOWN: "#0891b2",
        INTRO: "#475569",
      };
      secBadge.style.background = secColors[gb.section] || "#6d28d9";
    }

    // 4. Fill badge
    const fillBadge = document.getElementById("gb-fill-badge");
    if (fillBadge) {
      fillBadge.style.display = gb.isFill ? "inline-block" : "none";
    }

    // 5. Patterns readout
    if (gb.patterns) {
      for (const stem of STEMS) {
        const patEl = document.getElementById(`gb-pat-${stem}`);
        if (patEl && gb.patterns[stem]) {
          patEl.textContent = gb.patterns[stem];
        }
      }
    }

    // 6. Song Structure & Arrangement Timeline
    const camelotKey = gb.camelotCode ?? gb.key;
    if (camelotKey) {
      const camelotEl = document.getElementById("struct-camelot");
      if (camelotEl) {
        const chord = gb.chordName ? ` (${gb.chordName})` : "";
        camelotEl.textContent = `${camelotKey}${chord}`;
      }
    }
    const barCounterEl = document.getElementById("struct-bar-counter");
    if (barCounterEl && gb.totalBars !== undefined) {
      barCounterEl.textContent = String(gb.totalBars);
    }
    const energyEl = document.getElementById("struct-energy");
    if (energyEl && gb.energy !== undefined) {
      energyEl.textContent = `${Math.round(gb.energy * 100)}%`;
    }
    const nextSecEl = document.getElementById("struct-next-sec");
    if (nextSecEl && gb.nextSection && gb.sectionBarsLeft !== undefined) {
      const movement = gb.movementPhase ? `[${gb.movementPhase}] ` : "";
      nextSecEl.textContent = `${movement}NEXT: ${gb.nextSection.replace("_", " ")} IN ${gb.sectionBarsLeft} BARS`;
    }
    const secProgressEl = document.getElementById("struct-sec-progress");
    if (secProgressEl && gb.sectionBarsElapsed !== undefined && gb.sectionTotalBars) {
      const pct = Math.min(100, Math.max(0, (gb.sectionBarsElapsed / gb.sectionTotalBars) * 100));
      secProgressEl.style.width = `${pct}%`;
    }
    const timelineBlocks = document.querySelectorAll(".struct-block");
    timelineBlocks.forEach((block) => {
      const secName = block.getAttribute("data-section");
      if (secName === gb.section) {
        block.classList.add("active");
      } else {
        block.classList.remove("active");
      }
    });

    // 7. Multi-Track Pattern Matrix (16 or 32 steps)
    for (const stem of STEMS) {
      const titleEl = document.getElementById(`matrix-title-${stem}`);
      if (titleEl && gb.patterns[stem]) {
        titleEl.textContent = gb.patterns[stem];
        const audioScore = gb.patternAudioScores?.[stem];
        if (audioScore !== undefined) {
          titleEl.title = `${gb.patterns[stem]} • Real Audio Similarity Match: ${Math.round(audioScore * 100)}%`;
        }
      }
      const grooveEl = document.getElementById(`matrix-groove-${stem}`);
      if (grooveEl && gb.patternGrooves?.[stem]) {
        grooveEl.textContent = gb.patternGrooves[stem].replace(/_/g, " ").toUpperCase();
      }
      const gridContainer = document.getElementById(`matrix-grid-${stem}`);
      if (gridContainer) {
        if (totalSteps === 32) gridContainer.classList.add("mode-32");
        else gridContainer.classList.remove("mode-32");

        if (gridContainer.children.length !== totalSteps) {
          gridContainer.innerHTML = "";
          for (let i = 0; i < totalSteps; i++) {
            const pip = document.createElement("div");
            pip.className = (i % 4 === 0) ? "matrix-pip beat" : "matrix-pip";
            gridContainer.appendChild(pip);
          }
        }
        const gridData = gb.patternGrids?.[stem];
        const pips = gridContainer.children;
        for (let i = 0; i < totalSteps; i++) {
          const pip = pips[i] as HTMLElement;
          const hasSlice = Boolean(gridData && gridData[i] === 1);
          if (hasSlice) pip.classList.add("has-slice");
          else pip.classList.remove("has-slice");

          if (i + 1 === curStep) pip.classList.add("playhead");
          else pip.classList.remove("playhead");
        }
      }

      // Update feedback scores
      const rating = gb.patternRatings?.[stem] ?? 0;
      const mScore = document.getElementById(`matrix-score-${stem}`);
      const dScore = document.getElementById(`deck-score-${stem}`);
      const fmt = rating > 0 ? `+${rating}` : String(rating);
      if (mScore) {
        mScore.textContent = fmt;
        mScore.className = rating > 0 ? "feedback-score positive" : (rating < 0 ? "feedback-score negative" : "feedback-score");
      }
      if (dScore) {
        dScore.textContent = fmt;
        dScore.className = rating > 0 ? "feedback-score positive" : (rating < 0 ? "feedback-score negative" : "feedback-score");
      }
    }

    // 8. Vocal Singing Status Readout
    const vocalLyricEl = document.getElementById("gb-vocal-lyric");
    if (vocalLyricEl && gb.vocalLyric) {
      vocalLyricEl.textContent = `"${gb.vocalLyric}"`;
    }
    const vocalSongTitleEl = document.getElementById("gb-vocal-song-title");
    if (vocalSongTitleEl && gb.vocalSongTitle) {
      vocalSongTitleEl.textContent = gb.vocalSongTitle;
    }
    const vocalNoteEl = document.getElementById("gb-vocal-note");
    if (vocalNoteEl && gb.vocalNote) {
      vocalNoteEl.textContent = `NOTE: ${gb.vocalNote}`;
    }
    const vocalVowelEl = document.getElementById("gb-vocal-vowel");
    if (vocalVowelEl && gb.vocalVowel) {
      vocalVowelEl.textContent = `[${gb.vocalVowel}]`;
    }
    const vocalSyllableEl = document.getElementById("gb-vocal-syllable");
    if (vocalSyllableEl && gb.vocalSyllable) {
      vocalSyllableEl.textContent = gb.vocalSyllable;
      vocalSyllableEl.classList.add("singing");
      window.setTimeout(() => vocalSyllableEl.classList.remove("singing"), 150);
    }

    // 9. Live Hardware Mixer Level Sliders Update
    if (gb.stemMixLevels) {
      for (const stem of STEMS) {
        const lvl = gb.stemMixLevels[stem];
        if (lvl !== undefined) {
          const targetLvl = lvl * (this.states[stem]?.volume ?? 1.0);
          const volSlider = document.getElementById(`${stem}-volume`) as HTMLInputElement | null;
          const volVal = document.getElementById(`${stem}-vol-val`);
          if (volSlider && document.activeElement !== volSlider) {
            volSlider.value = targetLvl.toFixed(2);
          }
          if (volVal && document.activeElement !== volSlider) {
            volVal.textContent = targetLvl.toFixed(2);
          }
        }
      }
    }

    // 10. Neon Trippy Modular Sequencer Rack & Harmonic Matrix
    if (gb.musicScale) {
      const ms = gb.musicScale;
      const camelotBadge = document.getElementById("rack-camelot-badge");
      if (camelotBadge && ms.camelot) {
        camelotBadge.textContent = ms.camelot;
      }
      const scaleNameEl = document.getElementById("rack-scale-name");
      if (scaleNameEl && ms.keyName) {
        scaleNameEl.textContent = ms.keyName.toUpperCase();
      }
      const scaleModeEl = document.getElementById("rack-scale-mode");
      if (scaleModeEl && ms.modeName) {
        scaleModeEl.textContent = ms.modeName;
      }
      const chordNameEl = document.getElementById("rack-chord-name");
      if (chordNameEl && ms.chordName) {
        chordNameEl.textContent = ms.chordName;
      }
      const chordNotesEl = document.getElementById("rack-chord-notes");
      if (chordNotesEl && ms.chordNotes) {
        chordNotesEl.textContent = ms.chordNotes.join(" • ");
      }
      const bassHzEl = document.getElementById("rack-bass-hz");
      if (bassHzEl && ms.bassHz !== undefined) {
        bassHzEl.textContent = `${ms.bassHz.toFixed(1)} Hz`;
      }
      const chordHzEl = document.getElementById("rack-chord-hz");
      if (chordHzEl && ms.chordHz !== undefined) {
        chordHzEl.textContent = `${ms.chordHz.toFixed(1)} Hz`;
      }

      // Update 7 Scale Degree Pads
      if (ms.scaleNotes && ms.scalePitches) {
        const chordPitches = ms.chordPitches || [];
        for (let i = 0; i < 7; i++) {
          const padEl = document.getElementById(`rack-scale-pad-${i}`);
          const noteEl = document.getElementById(`rack-scale-note-${i}`);
          const noteName = ms.scaleNotes[i] || "";
          const pitch = ms.scalePitches[i];
          if (noteEl) noteEl.textContent = noteName;
          if (padEl) {
            const isChordTone = chordPitches.includes(pitch);
            if (isChordTone) {
              padEl.classList.add("active-chord-tone");
            } else {
              padEl.classList.remove("active-chord-tone");
            }
          }
        }
      }
    }

    // Update 4-Stem Modular Rack Units
    const activeStepIdx = (curStep - 1) % 16;
    for (const stem of STEMS) {
      const details = gb.patternDetails?.[stem];
      const patNameEl = document.getElementById(`rack-stem-patname-${stem}`);
      if (patNameEl) {
        patNameEl.textContent = details?.name || gb.patterns?.[stem] || `PART ${stem.toUpperCase()}`;
      }

      const territoryEl = document.getElementById(`rack-stem-territory-${stem}`);
      if (territoryEl && details?.territory) {
        territoryEl.textContent = details.territory.toUpperCase();
      }

      const grooveEl = document.getElementById(`rack-stem-groove-${stem}`);
      if (grooveEl) {
        const gName = details?.grooveType || gb.patternGrooves?.[stem] || "PATTERN";
        grooveEl.textContent = gName.replace(/_/g, " ").toUpperCase();
      }

      // Gauges: Similarity Match, Adversarial Goodness, Energy, Density
      const gRes = gb.adversarialGoodness?.[stem];
      const gScore = gRes ? gRes.score : (details?.goodnessScore ?? details?.similarity ?? 0.94);
      const isAccepted = gRes ? gRes.isAccepted : (details?.isAccepted ?? true);
      const gPct = Math.round(gScore * 100);
      const goodBarEl = document.getElementById(`rack-stem-goodbar-${stem}`);
      const goodValEl = document.getElementById(`rack-stem-goodval-${stem}`);
      if (goodBarEl) {
        goodBarEl.style.width = `${gPct}%`;
        goodBarEl.style.background = isAccepted ? "" : "linear-gradient(90deg, #ef4444, #f97316)";
      }
      if (goodValEl) {
        goodValEl.textContent = isAccepted ? `${gPct}% [✓]` : `${gPct}% [MUTATED]`;
        goodValEl.style.color = isAccepted ? "" : "#ef4444";
      }

      const simVal = details?.similarity ?? gb.patternAudioScores?.[stem] ?? 0.94;
      const simPct = Math.round(simVal * 100);
      const simBarEl = document.getElementById(`rack-stem-simbar-${stem}`);
      const simValEl = document.getElementById(`rack-stem-simval-${stem}`);
      if (simBarEl) simBarEl.style.width = `${simPct}%`;
      if (simValEl) simValEl.textContent = `${simPct}%`;

      const energyVal = details?.energy ?? gb.energy ?? 0.85;
      const energyPct = Math.round(energyVal * 100);
      const enBarEl = document.getElementById(`rack-stem-enbar-${stem}`);
      const enValEl = document.getElementById(`rack-stem-enval-${stem}`);
      if (enBarEl) enBarEl.style.width = `${energyPct}%`;
      if (enValEl) enValEl.textContent = `${energyPct}%`;

      const densityVal = details?.density ?? 0.5;
      const onsetCount = details?.onsetCount ?? (gb.patternGrids?.[stem]?.filter((v) => v === 1).length ?? 8);
      const denBarEl = document.getElementById(`rack-stem-denbar-${stem}`);
      const denValEl = document.getElementById(`rack-stem-denval-${stem}`);
      if (denBarEl) denBarEl.style.width = `${Math.min(100, Math.round(densityVal * 100))}%`;
      if (denValEl) denValEl.textContent = `${onsetCount}/16`;

      // 16-Step Pads: Onset LEDs, Velocity Level, and Sweeping Playhead
      const gridData = gb.patternGrids?.[stem];
      const velocities = details?.velocities;
      for (let s = 0; s < 16; s++) {
        const stepPad = document.getElementById(`rack-step-pad-${stem}-${s}`);
        if (!stepPad) continue;
        const hasOnset = Boolean(gridData && gridData[s] === 1);
        if (hasOnset) {
          stepPad.classList.add("has-onset");
        } else {
          stepPad.classList.remove("has-onset");
        }

        const vel = (velocities && velocities[s] !== undefined) ? velocities[s] : (hasOnset ? 0.9 : 0);
        const velBar = stepPad.querySelector(".rack-vel-bar") as HTMLElement | null;
        if (velBar) {
          velBar.style.height = `${Math.round(vel * 100)}%`;
        }

        if (s === activeStepIdx) {
          stepPad.classList.add("playhead");
        } else {
          stepPad.classList.remove("playhead");
        }
      }
    }

    this.lastGrooveboxMeta = gb;
  }

  // -- render loop ---------------------------------------------------------

  private analyserLevel(stem: StemName): number {
    try {
      const analyser = this.engine.getAnalyser(stem);
      const n = analyser.fftSize;
      let scratch = this.analyserScratch.get(stem);
      if (scratch === undefined || scratch.length !== n) {
        scratch = new Uint8Array(n);
        this.analyserScratch.set(stem, scratch);
      }
      analyser.getByteTimeDomainData(scratch);
      let sum = 0;
      for (let i = 0; i < scratch.length; i++) {
        const v = (scratch[i] - 128) / 128;
        sum += v * v;
      }
      return Math.min(1, Math.sqrt(sum / scratch.length) * 2.5);
    } catch {
      return 0;
    }
  }

  private frame(_t: number): void {
    const t = visualNow();
    const audioNow = this.engine.now();

    // Consume queued UI events strictly when their scheduled audio playback time arrives
    let latestUIEvent: UIEvent | null = null;
    while (this.uiEventQueue.length > 0 && this.uiEventQueue[0].atAudioSec <= audioNow) {
      latestUIEvent = this.uiEventQueue.shift()!;
    }

    if (latestUIEvent !== null) {
      this.currentStepIndex = latestUIEvent.step;
      this.currentStepAudioTime = latestUIEvent.atAudioSec;
      this.totalStepsInPattern = latestUIEvent.totalSteps || 16;
      this.updateGrooveboxUI(latestUIEvent.meta);

      const curStep = latestUIEvent.step % 16;
      for (const stem of STEMS) {
        if (latestUIEvent.meta.patternGrids?.[stem]?.[curStep] === 1) {
          const trackId = latestUIEvent.meta.patternTrackIds?.[stem] || "kg505_ml";
          this.pushRingEvent({
            stem,
            trackId,
            atSec: t,
          });
          if (this.nebula !== null) {
            const pos = latestUIEvent.meta.activeSliceCoords?.[stem] ?? this.getSliceCoordinates(stem, trackId, "");
            this.nebula.setActiveSlice(stem, pos.x, pos.y, t);
          }
        }
      }
    }

    // Continuous sub-step playhead progression across pattern
    let playheadProgress: number | undefined = undefined;
    if (this.playing && this.totalStepsInPattern > 0) {
      const stepDur = 60 / this.bpm / 4;
      const stepElapsed = Math.max(0, audioNow - this.currentStepAudioTime);
      const frac = Math.min(1.0, stepElapsed / stepDur);
      playheadProgress = ((this.currentStepIndex + frac) % this.totalStepsInPattern) / this.totalStepsInPattern;
    }

    // Live Motorized Console Fader Tracking (60 FPS smooth interpolation)
    if (this.playing) {
      for (const stem of STEMS) {
        const autoLvl = this.lastGrooveboxMeta?.stemMixLevels?.[stem] ?? this.engine.getStemMixLevel(stem);
        const targetLvl = autoLvl * (this.states[stem]?.volume ?? 1.0);
        this.currentDisplayVolumes[stem] += (targetLvl - this.currentDisplayVolumes[stem]) * 0.18;
        if (Math.abs(targetLvl - this.currentDisplayVolumes[stem]) < 0.005) {
          this.currentDisplayVolumes[stem] = targetLvl;
        }

        const volSlider = document.getElementById(`${stem}-volume`) as HTMLInputElement | null;
        const volVal = document.getElementById(`${stem}-vol-val`);
        if (volSlider && document.activeElement !== volSlider) {
          volSlider.value = this.currentDisplayVolumes[stem].toFixed(2);
        }
        if (volVal && document.activeElement !== volSlider) {
          volVal.textContent = this.currentDisplayVolumes[stem].toFixed(2);
        }
      }
    }

    // Prune stale ring events.
    while (this.ringEvents.length > 0 && t - this.ringEvents[0].atSec > RING_WINDOW_SEC) {
      this.ringEvents.shift();
    }
    if (this.ringsCtx !== null) {
      const stemsData = {} as Record<StemName, WaveformStemData>;
      for (const stem of STEMS) {
        let time = this.timeScratch.get(stem);
        let freq = this.freqScratch.get(stem);
        let level = 0;
        try {
          const analyser = this.engine.getAnalyser(stem);
          if (!time || time.length !== analyser.fftSize) {
            time = new Uint8Array(analyser.fftSize);
            this.timeScratch.set(stem, time);
          }
          if (!freq || freq.length !== analyser.frequencyBinCount) {
            freq = new Uint8Array(analyser.frequencyBinCount);
            this.freqScratch.set(stem, freq);
          }
          analyser.getByteTimeDomainData(time);
          analyser.getByteFrequencyData(freq);
          level = this.analyserLevel(stem);
        } catch {
          // Analyser not initialised yet pre-gesture
        }
        const activeTrackId = this.lastGrooveboxMeta?.patternTrackIds?.[stem] || this.lastGrooveboxMeta?.patterns?.[stem];
        stemsData[stem] = {
          timeDomain: time,
          frequency: freq,
          level,
          activeTrackId: activeTrackId || undefined,
          isMuted: this.engine.isMuted(stem),
          isSolo: this.engine.isSolo(stem),
        };
      }
      drawWaveformAnalysis(this.ringsCtx, {
        stems: stemsData,
        bpm: this.bpm,
        timeSec: t,
        playheadProgress,
      });
    }
    if (this.masterCtx !== null) {
      let time = this.masterTimeScratch;
      let freq = this.masterFreqScratch;
      let level = 0;
      try {
        const analyser = this.engine.getMasterAnalyser();
        if (!time || time.length !== analyser.fftSize) {
          time = new Uint8Array(analyser.fftSize);
          this.masterTimeScratch = time;
        }
        if (!freq || freq.length !== analyser.frequencyBinCount) {
          freq = new Uint8Array(analyser.frequencyBinCount);
          this.masterFreqScratch = freq;
        }
        analyser.getByteTimeDomainData(time);
        analyser.getByteFrequencyData(freq);
        let sum = 0;
        for (let i = 0; i < time.length; i++) {
          const v = (time[i] - 128) / 128;
          sum += v * v;
        }
        level = Math.min(1, Math.sqrt(sum / time.length) * 2.2);
      } catch {
        // Analyser not initialised yet pre-gesture
      }

      drawMasterAnalysis(this.masterCtx, {
        timeDomain: time ?? undefined,
        frequency: freq ?? undefined,
        level,
        bpm: this.bpm,
        timeSec: t,
        playheadProgress,
      });
    }
    if (this.nebula !== null) {
      this.nebula.render({
        timeSec: t,
        bpm: this.bpm,
        levels: {
          drums: this.analyserLevel("drums"),
          bass: this.analyserLevel("bass"),
          vocals: this.analyserLevel("vocals"),
          other: this.analyserLevel("other"),
        },
      });
    }
    if (this.psychedelicBg !== null) {
      const drumsLvl = this.analyserLevel("drums");
      const bassLvl = this.analyserLevel("bass");
      const vocalsLvl = this.analyserLevel("vocals");
      const otherLvl = this.analyserLevel("other");
      const combinedBass = Math.max(drumsLvl * 1.15, bassLvl * 1.25);
      const combinedMid = Math.max(vocalsLvl, otherLvl);
      const combinedTreble = (drumsLvl * 0.5) + (otherLvl * 0.7);
      const masterLvl = Math.max(combinedBass, combinedMid, combinedTreble);

      this.psychedelicBg.render({
        bassLevel: combinedBass,
        midLevel: combinedMid,
        trebleLevel: combinedTreble,
        masterLevel: masterLvl,
      });
    }

    if (this.tamagotchi !== null) {
      const drumsLvl = this.analyserLevel("drums");
      const bassLvl = this.analyserLevel("bass");
      const vocalsLvl = this.analyserLevel("vocals");
      const otherLvl = this.analyserLevel("other");
      const masterLvl = Math.max(drumsLvl, bassLvl, vocalsLvl, otherLvl);
      // Voice instrument audio waveform
      const vocalWaveform = this.timeScratch.get("vocals") ?? null;
      const drumWaveform = this.timeScratch.get("drums") ?? null;
      const bassWaveform = this.timeScratch.get("bass") ?? null;
      const otherWaveform = this.timeScratch.get("other") ?? null;
      this.tamagotchi.update(
        this.lastGrooveboxMeta,
        audioNow,
        masterLvl,
        this.currentStepIndex,
        this.bpm,
        vocalWaveform,
        vocalsLvl,
        drumWaveform,
        bassWaveform,
        drumsLvl,
        bassLvl,
        otherWaveform,
        otherLvl
      );
    }

    requestAnimationFrame((next) => this.frame(next));
  }

  private setStatus(text: string): void {
    this.statusEl.textContent = text;
  }
}

function boot(): void {
  const app = new WubApp();
  try {
    app.init();
  } catch (err) {
    const status = document.getElementById("status-line");
    const detail = err instanceof Error ? err.message : "unknown initialisation error";
    if (status !== null) status.textContent = `Failed to start: ${detail}`;
    throw err;
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
