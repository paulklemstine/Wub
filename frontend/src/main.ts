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
import { SingingHeadPanel } from "./singing_head";
import { TransitionProbe } from "./transition_probe";
import { RaveLightshowPanel, type LightshowPresetId } from "./lightshow";
import {
  Fm1MidiInput,
  FM1_DEFAULT_BANK,
  FM1_KNOB_STEMS,
  buildSliceOrder,
  isFm1BankId,
  knobValueToSlot,
  scrollSlot,
} from "./fm1_midi";
import type { Fm1MidiDevice, Fm1StatusTone } from "./fm1_midi";

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
  // Galaxy-like fallback: one cluster per stem spread across the square.
  const centers: Record<StemName, { x: number; y: number }> = {
    drums: { x: 0.22, y: 0.68 },
    bass: { x: 0.72, y: 0.30 },
    vocals: { x: 0.68, y: 0.72 },
    other: { x: 0.30, y: 0.28 },
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
  private lightshow: RaveLightshowPanel | null = null;
  private lightshowPanelEl: HTMLElement | null | undefined = undefined;
  private lsBearOverlayCanvas: HTMLCanvasElement | null = null;
  private bearOverlayEnabled = true;
  private tapTimestamps: number[] = [];
  private macroBuildupTimer: ReturnType<typeof setInterval> | null = null;

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
  private umapLoadStarted = false;
  private umapPoints: NebulaPoint[] = [];
  private lastNebulaKitId: string | null = null;
  private nebulaKitBaseSet = false;
  private kitBarIndex = -1;
  private kitPoint: { x: number; y: number } | null = null;
  private kitBlockAt = 0;
  private remixPhraseIds: Partial<Record<StemName, string | null>> = {};
  private transitionProbe = new TransitionProbe();
  private lastRemixTracks: Partial<Record<StemName, string>> = {};
  private movementBlocks = 0;
  private movementTrackChanges = 0;
  private movementHoldBars = 0;
  private movementMaxHoldBars = 0;
  private movementLog: Array<{
    at: number;
    bars: number;
    changedTracks: StemName[];
    target: string;
    tracks: Partial<Record<StemName, string>>;
  }> = [];
  private timeScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private freqScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private masterTimeScratch: Uint8Array<ArrayBuffer> | null = null;
  private masterFreqScratch: Uint8Array<ArrayBuffer> | null = null;
  private ringEvents: SliceArcEvent[] = [];
  private analyserScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private statusEl!: HTMLElement;
  private lastGrooveboxMeta: GrooveboxMeta | null = null;
  private clientConductor: ClientGrooveboxConductor | null = null;
  private bpmSliderDragging = false;
  private tamagotchi: TamagotchiAvatar | null = null;
  private singingHead: SingingHeadPanel | null = null;
  private vocalFloatScratch: Float32Array<ArrayBuffer> | null = null;
  private uiEventQueue: UIEvent[] = [];
  private currentStepIndex: number = 0;
  private currentStepAudioTime: number = 0;
  private totalStepsInPattern: number = 16;

  // FM-1 / CTL-1 knob → phrase control.
  private fm1: Fm1MidiInput | null = null;
  private fm1StatusEl: HTMLElement | null = null;
  private fm1Sliders: Partial<Record<StemName, HTMLInputElement>> = {};
  private fm1Labels: Partial<Record<StemName, HTMLElement>> = {};
  private fm1Slot: Partial<Record<StemName, number>> = {};
  private fm1LastRaw: Partial<Record<StemName, number>> = {};
  private fm1Timers: Partial<Record<StemName, ReturnType<typeof setTimeout>>> = {};
  private fm1SliceCache: Partial<Record<StemName, { key: string; at: number; points: NebulaPoint[] }>> = {};

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
    this.installCrashReporter();

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
        // Live visual feedback only; the audio commits on drop.
        this.clientConductor?.setReticleSteering(stem, x, y, false);
      },
      onReticleDrop: (stem, x, y) => {
        this.clientConductor?.setReticleSteering(stem, x, y, true);
      },
      onReticleRelease: (stem) => {
        this.clientConductor?.clearReticleSteering(stem);
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

    try {
      this.singingHead = new SingingHeadPanel();
      this.singingHead.initDOM();
      this.singingHead.bindAudioEngine(this.engine);
    } catch (e) {
      console.warn("Singing head init failed:", e);
    }

    this.bindTransport();
    this.bindMaster();
    for (const stem of STEMS) this.bindStem(stem);
    this.bindFeedbackControls();
    this.bindGrooveboxCycleControls();
    this.bindPlaybackModeControls();
    this.bindKitControls();
    this.bindPanelDock();
    this.initLightshowAndDjControls();

    // Initialize Autonomous In-Browser ML Sequencer
    try {
      this.clientConductor = new ClientGrooveboxConductor(this.engine);
      this.bindMbGenreSelect();
      // Play-path diagnostics: __wubStats() in the console shows which brain
      // drove each bar (global planner vs legacy patterns vs studio loops).
      // __wubDebugBar() shows per-stem audio resolution for the current bar.
      try {
        (window as any).__wubStats = () => {
          const c = this.clientConductor as any;
          return c?.playStats
            ? {
              ...c.playStats,
              section: c.arranger?.getSection?.(),
              activeKit: c.activeKit?.track_id ?? null,
              blocksInSet: c.blocksInSet ?? null,
              kitSetBlocksLeft: c.kitSetBlocksLeft ?? null,
              nextKitTrack: c.nextKitTrack ?? null,
              kitSetChanges: c.kitSetChanges ?? null,
              tasteModel: c.tasteValAuc ?? null,
              tasteModelName: c.tasteModelName ?? null,
              lastTransitionTop3: c.lastTransitionTop3 ?? [],
              lastKitScore: c.lastKitScore ?? null,
              kitSteerTarget: c.kitSteerTarget ?? null,
              kitSteerDistance: c.kitSteerDistance ?? null,
              steeringStems: c.steeringStems,
            }
            : { error: "engine not ready" };
        };
        (window as any).__wubReticles = () => this.nebula?.reticles ?? null;
        (window as any).__wubTransitionSummary = () => this.transitionProbe.summary();
        (window as any).__wubGlueState = () => this.engine.glueState();
        (window as any).__wubMovementSummary = () => ({
          blocks: this.movementBlocks,
          trackChanges: this.movementTrackChanges,
          lastHoldBars: this.movementHoldBars,
          maxHoldBars: this.movementMaxHoldBars,
          log: this.movementLog,
          composition: this.clientConductor?.getCompositionTelemetry() ?? null,
        });
        (window as any).__wubDebugBar = () => {
          const c = this.clientConductor as any;
          return typeof c?.debugBar === "function" ? c.debugBar() : { error: "engine not ready" };
        };
        (window as unknown as { __wubConductor: ClientGrooveboxConductor }).__wubConductor =
          this.clientConductor;
        (window as unknown as { __wubRemixState: () => unknown }).__wubRemixState = () =>
          this.clientConductor?.debugStemRemix() ?? [];
        (window as unknown as { __wubAudioProbe: () => unknown }).__wubAudioProbe = () => {
          const analyser = this.engine.getMasterAnalyser();
          const data = new Float32Array(analyser.fftSize);
          analyser.getFloatTimeDomainData(data);
          let sum = 0;
          let peak = 0;
          for (let i = 0; i < data.length; i++) {
            const v = data[i];
            sum += v * v;
            peak = Math.max(peak, Math.abs(v));
          }
          const stems = ["drums", "bass", "other", "vocals"].map((stem) => {
            const an = this.engine.getAnalyser(stem as never);
            const buf = new Float32Array(an.fftSize);
            an.getFloatTimeDomainData(buf);
            let s2 = 0;
            for (let i = 0; i < buf.length; i++) s2 += buf[i] * buf[i];
            const input = this.engine.getStemInput(stem as never) as GainNode;
            return {
              stem,
              rms: Number(Math.sqrt(s2 / buf.length).toFixed(5)),
              vol: this.engine.getVolume(stem as never),
              busGain: Number((input.gain?.value ?? -1).toFixed(3)),
              muted: this.engine.isMuted(stem as never),
            };
          });
          return {
            now: Number(this.engine.now().toFixed(2)),
            rms: Number(Math.sqrt(sum / data.length).toFixed(5)),
            peak: Number(peak.toFixed(5)),
            playing: this.playing,
            stems,
          };
        };
        (window as unknown as { __wubNextKit: () => void }).__wubNextKit = () => {
          this.clientConductor?.forceNextKit();
        };
        (window as unknown as { __wubNebula: NebulaRenderer | null }).__wubNebula = this.nebula;
      } catch {
        // Non-browser context; skip
      }
      // A/B evaluation flags (?joint=0&harm=0): compare next-gen joint
      // composition + harmonic fit against the legacy independent engine.
      try {
        const params = new URLSearchParams(window.location.search);
        if (params.get("joint") === "0") this.clientConductor.jointEnabled = false;
        if (params.get("harm") === "0") this.clientConductor.harmonicFitEnabled = false;
        if (params.get("joint") === "0" || params.get("harm") === "0") {
          console.info(`[Wub] A/B mode: joint=${this.clientConductor.jointEnabled} harm=${this.clientConductor.harmonicFitEnabled}`);
        }
      } catch {
        // Non-browser context; keep defaults
      }
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
      this.clientConductor.setKitSetChangeCallback((info) => {
        const track = info.kit.title || info.kit.track_id;
        this.setStatus(`${track} • ${Math.round(info.kit.bpm)} BPM`);
      });
      this.clientConductor.onKitPhrase = (p) => {
        const t = visualNow();
        this.kitPoint = { x: p.x, y: p.y };
        this.kitBlockAt = p.at;
        if (this.nebula !== null) {
          for (const stem of STEMS) {
            this.nebula.setActiveSlice(stem, p.x, p.y, t);
          }
        }
      };
      this.clientConductor.onStemPhrases = (points) => {
        const t = visualNow();
        this.kitPoint = null;
        this.kitBlockAt = points[0]?.at ?? this.kitBlockAt;
        if (this.nebula === null) return;
        for (const point of points) {
          this.nebula.setActiveSlice(point.stem, point.x, point.y, t);
          this.nebula.setReticleTarget(point.stem, point.x, point.y, 0.28);
          this.remixPhraseIds[point.stem] = point.phraseId;
        }
        this.nebula.setHighlight(
          points.map((point) => ({
            x: point.x,
            y: point.y,
            stem: point.stem,
            phraseId: point.phraseId,
            active: true,
          }))
        );
      };
      // Hidden seam telemetry. No UI; logs only suspicious boundaries and
      // keeps the full capture in window.__wubTransitionLog.
      this.clientConductor.onKitTransition = (at, result) => {
        const tracks = result.stemTracks ?? {};
        const changedTracks: StemName[] = [];
        for (const stem of STEMS) {
          const track = tracks[stem];
          if (track && this.lastRemixTracks[stem] !== track) changedTracks.push(stem);
        }
        this.transitionProbe.markBoundary({
          at,
          blockBars: result.blockBars,
          changed: changedTracks,
          bridge: result.bridge,
          rotation: result.setChanged,
          clusterId: this.clientConductor?.activeClusterId ?? "",
          targetEnergy: result.targetEnergy,
          tracks,
        });
        this.movementBlocks += 1;
        this.movementTrackChanges += changedTracks.length;
        if (changedTracks.length > 0) {
          this.movementHoldBars = 0;
        } else {
          this.movementHoldBars += result.blockBars;
          this.movementMaxHoldBars = Math.max(this.movementMaxHoldBars, this.movementHoldBars);
        }
        this.lastRemixTracks = { ...tracks };
        this.movementLog.push({
          at,
          bars: result.blockBars,
          changedTracks,
          target: result.targetSection,
          tracks,
        });
        if (this.movementLog.length > 200) this.movementLog.splice(0, this.movementLog.length - 200);
      };
    } catch (e) {
      console.warn("Client ML sequencer init failed:", e);
    }

    // Hardware knob phrase control (M-VAVE FM-1 running CTL-1 firmware).
    this.initFm1();

    // Sticky BPM: load from localStorage if set, else default to 96 BPM
    const savedBpm = localStorage.getItem("wub_bpm");
    if (savedBpm) {
      const parsed = Number(savedBpm);
      if (Number.isFinite(parsed) && parsed >= BPM_MIN && parsed <= BPM_MAX) {
        this.bpm = parsed;
        const slider = document.getElementById("bpm-slider");
        if (slider instanceof HTMLInputElement) slider.value = String(Math.round(this.bpm));
        const label = document.getElementById("bpm-value");
        if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
        this.engine.setTargetBpm(this.bpm);
        if (this.clientConductor?.getPlaybackMode() !== "kit") {
          this.clientConductor?.setBpm(this.bpm);
        }
      }
    } else {
      this.bpm = DEFAULT_BPM;
    }

    window.addEventListener("resize", () => {
      if (this.nebula !== null) this.nebula.resize();
    });
    window.addEventListener("beforeunload", () => this.teardown());

    this.setStatus(`Ready • ${Math.round(this.bpm)} BPM`);
    el<HTMLButtonElement>("btn-play").classList.remove("active");
    el<HTMLButtonElement>("btn-stop").classList.add("active");
    requestAnimationFrame((t) => this.frame(t));
  }

  private async loadUmapPoints(): Promise<void> {
    if (this.umapLoadStarted) return;
    this.umapLoadStarted = true;
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
      // Galaxy spread: raw UMAP output clumps into part of the square
      // (x lives ~0.4..0.9). Percentile-stretch to fill the canvas while
      // preserving cluster structure. Applied BEFORE positions are stored,
      // so visuals, snapping, and sound-mapping share one space.
      const xs: number[] = [];
      const ys: number[] = [];
      for (const item of fallbackPts) {
        if (!isRecord(item)) continue;
        const x = asNumber(item["x"], -1);
        const y = asNumber(item["y"], -1);
        if (x >= 0 && y >= 0) {
          xs.push(x);
          ys.push(y);
        }
      }
      const percentile = (arr: number[], p: number): number => {
        if (arr.length === 0) return p === 0 ? 0 : 1;
        const sorted = [...arr].sort((a, b) => a - b);
        return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
      };
      const x0 = percentile(xs, 1);
      const x1 = percentile(xs, 99);
      const y0 = percentile(ys, 1);
      const y1 = percentile(ys, 99);
      const stretch = (v: number, lo: number, hi: number): number => {
        if (!(hi > lo)) return Math.min(1, Math.max(0, v));
        return Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
      };
      const pts: NebulaPoint[] = [];
      for (const item of fallbackPts) {
        if (!isRecord(item)) continue;
        const stem = asStem(item["stem"]);
        const x = stretch(asNumber(item["x"], -1), x0, x1);
        const y = stretch(asNumber(item["y"], -1), y0, y1);
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
        this.umapPoints = pts;
        const kitPts =
          this.clientConductor && this.clientConductor.getPlaybackMode() === "kit"
            ? this.clientConductor.getAllKitPhrasePoints()
            : [];
        this.nebula.setPoints(kitPts.length > 0 ? [...pts, ...kitPts] : pts);
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
    const songBtn = document.getElementById("btn-song-mode");
    if (songBtn instanceof HTMLButtonElement) {
      songBtn.addEventListener("click", () => {
        const nextMode = this.clientConductor?.getSongMode() === "morph" ? "song" : "morph";
        this.clientConductor?.setSongMode(nextMode);
        songBtn.textContent = nextMode === "song" ? "MODE: SONG" : "MODE: MORPH";
        songBtn.classList.toggle("active", nextMode === "song");
      });
    }
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
      if (slider instanceof HTMLInputElement) slider.value = String(Math.round(this.bpm));
      const label = document.getElementById("bpm-value");
      if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
      this.engine.setTargetBpm(this.bpm);
    }
    this.setStatus(`Playing • ${Math.round(this.bpm)} BPM`);
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
    slider.value = String(Math.round(this.bpm));
    value.textContent = `${Math.round(this.bpm)} BPM`;
    const startDrag = (): void => {
      this.bpmSliderDragging = true;
    };
    const endDrag = (): void => {
      this.bpmSliderDragging = false;
    };
    slider.addEventListener("pointerdown", startDrag);
    slider.addEventListener("mousedown", startDrag);
    slider.addEventListener("touchstart", startDrag, { passive: true });
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("mouseup", endDrag);
    window.addEventListener("touchend", endDrag, { passive: true });
    const applyBpmInput = (): void => {
      const next = Math.min(BPM_MAX, Math.max(BPM_MIN, Number(slider.value)));
      if (!Number.isFinite(next)) return;
      this.bpm = next;
      localStorage.setItem("wub_bpm", String(next));
      value.textContent = `${Math.round(next)} BPM`;
      this.engine.setTargetBpm(next);
      this.clientConductor?.setBpm(next);
    };
    slider.addEventListener("input", applyBpmInput);
    slider.addEventListener("change", () => {
      endDrag();
      applyBpmInput();
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
      // The light show is off by default (full-screen shader); reflect that in
      // the control instead of leaving a stale "ON" label.
      const initial = bg.getState().enabled;
      toggleBtn.classList.toggle("active", initial);
      toggleBtn.setAttribute("aria-pressed", String(initial));
      toggleBtn.textContent = initial ? "LIGHT SHOW: ON" : "LIGHT SHOW: OFF";

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
    document.getElementById("btn-save-listening")?.addEventListener("click", () => {
      const exporter = (window as unknown as { __wubSliceFlowExport?: () => void }).__wubSliceFlowExport;
      if (exporter) exporter();
      else this.setStatus("Start a listening session before saving it.");
    });
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
        const mode = btn.dataset.playback as "jukebox" | "slices" | "phrase_blocks" | "kit" | undefined;
        if (!mode) return;
        if (this.clientConductor) {
          this.clientConductor.setPlaybackMode(mode);
        }
        modeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        if (mode !== "kit") {
          void this.loadUmapPoints();
        }
        const label =
          mode === "jukebox"
            ? "Jukebox Groove (Studio Loops + Groovy Slices & Vocal Chops)"
            : mode === "phrase_blocks"
            ? "Master Phrases (4-Bar Cohesive Studio Blocks)"
            : mode === "kit"
            ? "Kit Player (Locked Studio Kit • Real 4-Bar Stem Phrases)"
            : "Live Slices (16th-Step Micro-Slice Re-sequencer)";
        this.setStatus(`Engine Mode: ${label}`);
      });
    });
  }

  private bindKitControls(): void {
    const nextKitBtn = document.getElementById("btn-next-kit");
    if (nextKitBtn !== null) {
      nextKitBtn.addEventListener("click", () => {
        this.clientConductor?.forceNextKit();
        this.setStatus("NEXT KIT queued — bridge starts on the next 4-bar boundary.");
      });
    }
  }

  private bindPanelDock(): void {
    const setPanelVisible = (panelId: string, visible: boolean) => {
      const panel = document.getElementById(panelId);
      if (!panel) return;
      panel.classList.toggle("panel-minimized", !visible);
      const dockBtn = document.querySelector<HTMLButtonElement>(`.panel-dock-btn[data-panel="${panelId}"]`);
      if (dockBtn) {
        dockBtn.classList.toggle("active", visible);
        dockBtn.setAttribute("aria-pressed", String(visible));
      }
      if (visible) {
        requestAnimationFrame(() => {
          this.nebula?.resize();
          window.dispatchEvent(new Event("resize"));
        });
      }
    };

    document.querySelectorAll<HTMLButtonElement>(".panel-dock-btn[data-panel]").forEach((btn) => {
      const id = btn.dataset.panel;
      if (!id) return;
      const panel = document.getElementById(id);
      if (panel) {
        const isVis = !panel.classList.contains("panel-minimized");
        btn.classList.toggle("active", isVis);
        btn.setAttribute("aria-pressed", String(isVis));
      }
      btn.addEventListener("click", () => {
        const target = document.getElementById(id);
        if (!target) return;
        const nextVis = target.classList.contains("panel-minimized");
        setPanelVisible(id, nextVis);
      });
    });

    document.querySelectorAll<HTMLButtonElement>("[data-panel-min]").forEach((btn) => {
      const id = btn.dataset.panelMin;
      if (!id) return;
      btn.addEventListener("click", () => setPanelVisible(id, false));
    });
  }

  private initLightshowAndDjControls(): void {
    // 1. Trippy Rave Lightshow (All 267 SuperAcid + SpaceTubes scenes)
    const lsCanvas = document.getElementById("lightshow-canvas") as HTMLCanvasElement | null;
    const presetBtns = document.querySelectorAll<HTMLButtonElement>("[data-ls-preset]");
    const presetSelect = document.getElementById("ls-preset-select") as HTMLSelectElement | null;
    const presetSearch = document.getElementById("ls-preset-search") as HTMLInputElement | null;
    const syncPresetBtns = (preset: LightshowPresetId) => {
      presetBtns.forEach((b) => b.classList.toggle("active", b.dataset.lsPreset === preset));
      if (presetSelect && presetSelect.value !== preset) {
        presetSelect.value = preset;
      }
    };
    const rebuildPresetSelect = (query: string = "") => {
      if (!presetSelect || !this.lightshow) return;
      const q = query.trim().toLowerCase();
      const all = this.lightshow.getPresets();
      const filtered = q ? all.filter((p) => p.label.toLowerCase().includes(q) || p.id.includes(q)) : all;
      const cur = this.lightshow.getState().preset;
      presetSelect.replaceChildren();
      for (const p of filtered) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = `${p.role === "fg" ? "★ " : ""}${p.label}`;
        if (p.id === cur) opt.selected = true;
        presetSelect.appendChild(opt);
      }
    };

    if (lsCanvas) {
      this.lightshow = new RaveLightshowPanel(lsCanvas, (_presets, activeId) => {
        if (presetSelect && presetSelect.options.length !== _presets.length && !(presetSearch?.value.trim())) {
          rebuildPresetSelect("");
        }
        syncPresetBtns(activeId);
      });
      rebuildPresetSelect("");
      presetBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
          const p = btn.dataset.lsPreset as LightshowPresetId | undefined;
          if (!p || !this.lightshow) return;
          this.lightshow.setState({ preset: p });
          syncPresetBtns(p);
        });
      });

      presetSelect?.addEventListener("change", () => {
        const val = presetSelect.value;
        if (val && this.lightshow) {
          this.lightshow.setState({ preset: val });
          syncPresetBtns(val);
        }
      });
      presetSearch?.addEventListener("input", () => rebuildPresetSelect(presetSearch.value));

      document.getElementById("btn-ls-prev")?.addEventListener("click", () => this.lightshow?.stepPreset(-1));
      document.getElementById("btn-ls-next")?.addEventListener("click", () => this.lightshow?.stepPreset(1));
      document.getElementById("btn-ls-rand")?.addEventListener("click", () => this.lightshow?.randomPreset());

      const dualBtn = document.getElementById("btn-ls-dual");
      if (dualBtn) {
        const dualOn = this.lightshow?.getState().dualLayer ?? false;
        dualBtn.classList.toggle("active", dualOn);
        dualBtn.textContent = `2-LAYER: ${dualOn ? "ON" : "OFF"}`;
        dualBtn.addEventListener("click", () => {
          if (!this.lightshow) return;
          const next = !this.lightshow.getState().dualLayer;
          this.lightshow.setState({ dualLayer: next });
          dualBtn.classList.toggle("active", next);
          dualBtn.textContent = `2-LAYER: ${next ? "ON" : "OFF"}`;
        });
      }

      const cortexBtn = document.getElementById("btn-ls-cortex");
      if (cortexBtn) {
        cortexBtn.addEventListener("click", () => {
          if (!this.lightshow) return;
          const next = !this.lightshow.getState().cortexAuto;
          this.lightshow.setState({ cortexAuto: next });
          cortexBtn.classList.toggle("active", next);
          cortexBtn.textContent = `CORTEX VJ: ${next ? "ON" : "OFF"}`;
        });
      }

      const wireLsBool = (
        id: string,
        key: "fxBloom" | "fxTrails" | "fxChroma" | "fxGlitch" | "fxKaleido" | "fxStrobe" | "blackout"
      ) => {
        const b = document.getElementById(id);
        if (!b) return;
        b.addEventListener("click", () => {
          if (!this.lightshow) return;
          const next = !this.lightshow.getState()[key];
          this.lightshow.setState({ [key]: next });
          b.classList.toggle("active", next);
        });
      };
      wireLsBool("btn-ls-bloom", "fxBloom");
      wireLsBool("btn-ls-trails", "fxTrails");
      wireLsBool("btn-ls-chroma", "fxChroma");
      wireLsBool("btn-ls-glitch", "fxGlitch");
      wireLsBool("btn-ls-kaleido", "fxKaleido");
      wireLsBool("btn-ls-strobe", "fxStrobe");
      wireLsBool("btn-ls-blackout", "blackout");

      document.getElementById("btn-ls-palette")?.addEventListener("click", () => {
        if (!this.lightshow) return;
        const cur = this.lightshow.getState().palette;
        this.lightshow.setState({ palette: (cur + 1) % 4 });
      });

      const lsGain = document.getElementById("ls-intensity-slider") as HTMLInputElement | null;
      lsGain?.addEventListener("input", () => {
        this.lightshow?.setState({ intensity: Number(lsGain.value) || 1.25 });
      });
      const lsSpd = document.getElementById("ls-speed-slider") as HTMLInputElement | null;
      lsSpd?.addEventListener("input", () => {
        this.lightshow?.setState({ speed: Number(lsSpd.value) || 1.05 });
      });
    }

    this.lsBearOverlayCanvas = document.getElementById("lightshow-bear-overlay") as HTMLCanvasElement | null;
    const bearOverlayBtn = document.getElementById("btn-ls-bear");
    const syncBearOverlay = () => {
      if (this.lsBearOverlayCanvas) {
        this.lsBearOverlayCanvas.style.display = this.bearOverlayEnabled ? "block" : "none";
      }
      if (bearOverlayBtn) {
        bearOverlayBtn.classList.toggle("active", this.bearOverlayEnabled);
        bearOverlayBtn.textContent = `BEAR OVERLAY: ${this.bearOverlayEnabled ? "ON" : "OFF"}`;
      }
    };
    const toggleBearOverlay = () => {
      this.bearOverlayEnabled = !this.bearOverlayEnabled;
      syncBearOverlay();
    };
    syncBearOverlay();
    bearOverlayBtn?.addEventListener("click", toggleBearOverlay);

    const lsPanel = document.getElementById("panel-lightshow");
    const fsBtn = document.getElementById("btn-lightshow-fullscreen");
    const toggleFullscreen = () => {
      if (!lsPanel) return;
      if (document.fullscreenElement === lsPanel) {
        void document.exitFullscreen?.();
      } else {
        void lsPanel.requestFullscreen?.();
      }
    };
    fsBtn?.addEventListener("click", toggleFullscreen);
    document.addEventListener("fullscreenchange", () => {
      if (fsBtn && lsPanel) {
        fsBtn.textContent = document.fullscreenElement === lsPanel ? "EXIT ✕" : "FULLSCREEN ⛶";
      }
    });

    // 2. DJ Control Surface Transport Helpers (Tap Tempo, 0.5x / 2x)
    const applyBpmImmediate = (newBpm: number) => {
      const clamped = Math.max(60, Math.min(200, Math.round(newBpm)));
      this.bpm = clamped;
      const s = document.getElementById("bpm-slider") as HTMLInputElement | null;
      const v = document.getElementById("bpm-value");
      if (s) s.value = String(Math.max(80, Math.min(175, clamped)));
      if (v) v.textContent = `${clamped} BPM`;
      localStorage.setItem("wub_bpm", String(clamped));
      this.engine.setTargetBpm(clamped);
      this.clientConductor?.setBpm(clamped);
    };

    document.getElementById("btn-tap-tempo")?.addEventListener("click", () => {
      const now = performance.now();
      if (this.tapTimestamps.length > 0 && now - this.tapTimestamps[this.tapTimestamps.length - 1] > 2200) {
        this.tapTimestamps = [];
      }
      this.tapTimestamps.push(now);
      if (this.tapTimestamps.length > 5) this.tapTimestamps.shift();
      if (this.tapTimestamps.length >= 2) {
        let sum = 0;
        for (let i = 1; i < this.tapTimestamps.length; i++) {
          sum += this.tapTimestamps[i] - this.tapTimestamps[i - 1];
        }
        const avgMs = sum / (this.tapTimestamps.length - 1);
        if (avgMs > 250 && avgMs < 1500) {
          applyBpmImmediate(60000 / avgMs);
        }
      }
    });
    document.getElementById("btn-bpm-half")?.addEventListener("click", () => applyBpmImmediate(this.bpm * 0.5));
    document.getElementById("btn-bpm-double")?.addEventListener("click", () => applyBpmImmediate(this.bpm * 2.0));

    // 3. Per-Stem DJ Strip (Kill / Echo Throw / Morph)
    const killBtns = document.querySelectorAll<HTMLButtonElement>("[data-dj-kill]");
    const syncKillBtns = () => {
      killBtns.forEach((b) => {
        const s = b.dataset.djKill as StemName | undefined;
        if (s) b.classList.toggle("active", this.engine.isStemKilled(s));
      });
    };
    killBtns.forEach((btn) => {
      const stem = btn.dataset.djKill as StemName | undefined;
      if (!stem) return;
      btn.addEventListener("click", () => {
        this.engine.setStemKill(stem, !this.engine.isStemKilled(stem));
        syncKillBtns();
      });
    });

    const echoThrowBtns = document.querySelectorAll<HTMLButtonElement>("[data-dj-echo]");
    const syncEchoBtns = () => {
      echoThrowBtns.forEach((b) => {
        const s = b.dataset.djEcho as StemName | undefined;
        if (s) b.classList.toggle("active", this.engine.isStemEchoThrow(s));
      });
    };
    echoThrowBtns.forEach((btn) => {
      const stem = btn.dataset.djEcho as StemName | undefined;
      if (!stem) return;
      btn.addEventListener("click", () => {
        this.engine.setStemEchoThrow(stem, !this.engine.isStemEchoThrow(stem));
        syncEchoBtns();
      });
    });

    document.querySelectorAll<HTMLButtonElement>("[data-dj-morph]").forEach((btn) => {
      const stem = btn.dataset.djMorph as StemName | undefined;
      if (!stem) return;
      btn.addEventListener("click", () => {
        btn.classList.add("active");
        setTimeout(() => btn.classList.remove("active"), 300);
        const pts = this.clientConductor?.getKitNebulaPoints().filter((p) => p.stem === stem) ?? [];
        if (pts.length > 0 && this.clientConductor) {
          const pick = pts[Math.floor(Math.random() * pts.length)];
          this.nebula?.setReticleTarget(stem, pick.x, pick.y, 0.25);
          this.clientConductor.setReticleSteering(stem, pick.x, pick.y, true);
        } else {
          this.clientConductor?.rerollStem(stem);
        }
        this.setStatus(`⚡ Morphing ${stem.toUpperCase()} onto a fresh compatible groove...`);
      });
    });

    // 4. Groovebox DJ Audio FX Knobs & Momentary Performance Pads
    const filterSlider = document.getElementById("dj-filter-slider") as HTMLInputElement | null;
    const filterVal = document.getElementById("dj-filter-val");
    const filterQSlider = document.getElementById("dj-filter-q-slider") as HTMLInputElement | null;
    const filterQVal = document.getElementById("dj-filter-q-val");
    const echoSlider = document.getElementById("dj-echo-slider") as HTMLInputElement | null;
    const echoVal = document.getElementById("dj-echo-val");
    const reverbSlider = document.getElementById("dj-reverb-slider") as HTMLInputElement | null;
    const reverbVal = document.getElementById("dj-reverb-val");
    const acidSlider = document.getElementById("dj-acid-slider") as HTMLInputElement | null;
    const acidVal = document.getElementById("dj-acid-val");
    const crushSlider = document.getElementById("dj-crush-slider") as HTMLInputElement | null;
    const crushVal = document.getElementById("dj-crush-val");

    const syncDjFxUI = () => {
      const st = this.engine.getDjFxState();
      if (filterSlider) filterSlider.value = String(st.filterSweep);
      if (filterVal) {
        filterVal.textContent =
          Math.abs(st.filterSweep) < 0.03
            ? "CENTER"
            : st.filterSweep < 0
              ? `LPF ${Math.round(-st.filterSweep * 100)}%`
              : `HPF ${Math.round(st.filterSweep * 100)}%`;
      }
      if (filterQSlider) filterQSlider.value = String(st.filterQ);
      if (filterQVal) filterQVal.textContent = st.filterQ.toFixed(1);
      if (echoSlider) echoSlider.value = String(st.dubEcho);
      if (echoVal) echoVal.textContent = `${Math.round(st.dubEcho * 100)}%`;
      if (reverbSlider) reverbSlider.value = String(st.reverbWash);
      if (reverbVal) reverbVal.textContent = `${Math.round(st.reverbWash * 100)}%`;
      if (acidSlider) acidSlider.value = String(st.acidDrive);
      if (acidVal) acidVal.textContent = `${Math.round(st.acidDrive * 100)}%`;
      if (crushSlider) crushSlider.value = String(st.crush);
      if (crushVal) crushVal.textContent = `${Math.round(st.crush * 100)}%`;
      syncKillBtns();
      syncEchoBtns();
    };

    filterSlider?.addEventListener("input", () => {
      this.engine.setDjFilter(Number(filterSlider.value));
      syncDjFxUI();
    });
    filterSlider?.addEventListener("dblclick", () => {
      this.engine.setDjFilter(0);
      syncDjFxUI();
    });
    filterQSlider?.addEventListener("input", () => {
      this.engine.setDjFilter(this.engine.getDjFxState().filterSweep, Number(filterQSlider.value));
      syncDjFxUI();
    });
    echoSlider?.addEventListener("input", () => {
      this.engine.setDjDubEcho(Number(echoSlider.value));
      syncDjFxUI();
    });
    reverbSlider?.addEventListener("input", () => {
      this.engine.setDjReverbWash(Number(reverbSlider.value));
      syncDjFxUI();
    });
    acidSlider?.addEventListener("input", () => {
      this.engine.setDjAcid(Number(acidSlider.value));
      syncDjFxUI();
    });
    crushSlider?.addEventListener("input", () => {
      this.engine.setDjCrush(Number(crushSlider.value));
      syncDjFxUI();
    });

    const rollBtns = document.querySelectorAll<HTMLButtonElement>("[data-dj-roll]");
    const engageRoll = (beats: number) => {
      this.engine.setDjRoll(beats);
      rollBtns.forEach((b) => b.classList.toggle("active", Number(b.dataset.djRoll) === beats && beats > 0));
    };
    rollBtns.forEach((btn) => {
      const beats = Number(btn.dataset.djRoll || 0);
      btn.addEventListener("pointerdown", () => engageRoll(beats));
      const stopRoll = () => {
        if (this.engine.getDjFxState().rollBeats === beats) engageRoll(0);
      };
      btn.addEventListener("pointerup", stopRoll);
      btn.addEventListener("pointerleave", stopRoll);
      btn.addEventListener("pointercancel", stopRoll);
    });

    const tapeBtn = document.getElementById("btn-dj-tapestop");
    const setTape = (on: boolean) => {
      this.engine.setDjTapeStop(on);
      tapeBtn?.classList.toggle("active", on);
    };
    tapeBtn?.addEventListener("pointerdown", () => setTape(true));
    tapeBtn?.addEventListener("pointerup", () => setTape(false));
    tapeBtn?.addEventListener("pointerleave", () => setTape(false));
    tapeBtn?.addEventListener("pointercancel", () => setTape(false));

    // Scene Performance Macros
    const clearBuildupTimer = () => {
      if (this.macroBuildupTimer !== null) {
        clearInterval(this.macroBuildupTimer);
        this.macroBuildupTimer = null;
      }
    };
    document.getElementById("btn-macro-buildup")?.addEventListener("click", () => {
      clearBuildupTimer();
      let p = 0;
      this.macroBuildupTimer = setInterval(() => {
        p = Math.min(1, p + 0.08);
        this.engine.setDjFilter(p * 0.72, 4.5);
        this.engine.setDjReverbWash(p * 0.65);
        this.engine.setDjDubEcho(p * 0.45);
        if (p > 0.72) engageRoll(0.5);
        syncDjFxUI();
        if (p >= 1) clearBuildupTimer();
      }, 90);
      this.setStatus("🔥 BUILDUP MACRO: HPF + Echo/Reverb rising...");
    });
    document.getElementById("btn-macro-drop")?.addEventListener("click", () => {
      clearBuildupTimer();
      this.engine.resetDjFx();
      engageRoll(0);
      setTape(false);
      this.lightshow?.pulseStrobeBriefly(360);
      syncDjFxUI();
      this.setStatus("⚡ DROP IMPACT: Full frequency slam!");
    });
    document.getElementById("btn-macro-break")?.addEventListener("click", () => {
      clearBuildupTimer();
      this.engine.setStemKill("drums", true);
      this.engine.setDjFilter(-0.22, 2.8);
      this.engine.setDjDubEcho(0.68);
      this.engine.setDjReverbWash(0.75);
      syncDjFxUI();
      this.setStatus("🌌 BREAKDOWN SPACE: Drums killed + Dub/Reverb wash");
    });
    document.getElementById("btn-macro-reset")?.addEventListener("click", () => {
      clearBuildupTimer();
      this.engine.resetDjFx();
      engageRoll(0);
      setTape(false);
      syncDjFxUI();
      this.setStatus("DJ FX reset cleanly.");
    });

    // Keyboard Hotkeys (Space=Play/Stop, Q/W/E=Rolls, T=TapeStop, 1-4=Stem Kills, F=Fullscreen)
    const isTypingTarget = (ev: KeyboardEvent): boolean => {
      const t = ev.target as HTMLElement | null;
      if (!t) return false;
      const tag = t.tagName;
      return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
    };
    window.addEventListener("keydown", (ev) => {
      if (isTypingTarget(ev) || ev.repeat) return;
      const k = ev.key.toLowerCase();
      if (ev.code === "Space") {
        ev.preventDefault();
        if (this.playing) void this.stop();
        else void this.play();
      } else if (k === "q") {
        engageRoll(1);
      } else if (k === "w") {
        engageRoll(0.5);
      } else if (k === "e") {
        engageRoll(0.25);
      } else if (k === "t") {
        setTape(true);
      } else if (k === "f") {
        toggleFullscreen();
      } else if (k === "o") {
        toggleBearOverlay();
      } else if (k === "1" || k === "2" || k === "3" || k === "4") {
        const stem = STEMS[Number(k) - 1];
        this.engine.setStemKill(stem, !this.engine.isStemKilled(stem));
        syncKillBtns();
      }
    });
    window.addEventListener("keyup", (ev) => {
      const k = ev.key.toLowerCase();
      if (k === "q" || k === "w" || k === "e") {
        engageRoll(0);
      } else if (k === "t") {
        setTape(false);
      }
    });
  }

  /** MusicBrainz genre multi-select next to the style select.  Options come
   *  from the manifest's per-kit genre buckets (dedup, all preselected);
   *  Ctrl/Cmd+click to narrow, empty to all. */
  private bindMbGenreSelect(): void {
    const el = document.getElementById("mb-genre-select") as HTMLSelectElement | null;
    if (el === null) return;
    const populate = (styles: readonly string[]) => {
      if (styles.length === 0) return;
      const prev = new Set([...el.selectedOptions].map((o) => o.value));
      el.replaceChildren();
      for (const s of styles) {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s.toUpperCase().replace(/_/g, " ");
        opt.selected = prev.size === 0 ? true : prev.has(s);
        el.appendChild(opt);
      }
    };
    if (this.clientConductor?.mbGenres?.length) populate(this.clientConductor.mbGenres);
    if (this.clientConductor) {
      this.clientConductor.onMbGenreStyles = populate;
    }
    el.addEventListener("change", () => {
      const selected = [...el.selectedOptions].map((o) => o.value);
      this.clientConductor?.applyMbGenreSelection(selected.length > 0 ? new Set(selected) : null);
      this.setStatus(
        selected.length === 0
          ? "Genre: all genres (full bank)."
          : selected.length === el.options.length
            ? "Genre: all genres (full bank)."
            : `Genre: ${selected.sort().join(" + ")} (restarts the set here).`,
      );
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

    // 0. Active Cycle Mode Button (only CYCLE buttons carry data-mode; engine
    // buttons use data-playback and must keep their own active state)
    if (gb.cycleMode) {
      const cycleBtns = document.querySelectorAll<HTMLButtonElement>(".gb-cycle-btn[data-mode]");
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

  private lastVisualFrame = -Infinity;
  private readonly frameLevels = new Map<StemName, number>();
  // Adaptive visual budget: audio/UI stay on every frame, but the heavy
  // 2D/3D painters alternate when the browser cannot keep up.
  private frameIntervalEma = 1000 / 60;
  private visualDegraded = false;
  private visualTick = 0;

  private analyserLevel(stem: StemName): number {
    const cached = this.frameLevels.get(stem);
    if (cached !== undefined) return cached;
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
      const level = Math.min(1, Math.sqrt(sum / scratch.length) * 2.5);
      this.frameLevels.set(stem, level);
      return level;
    } catch {
      return 0;
    }
  }

  /** Latest time-domain waveform for one stem (reused scratch buffer). */
  private stemWaveform(stem: StemName): Uint8Array | null {
    try {
      const analyser = this.engine.getAnalyser(stem);
      let time = this.timeScratch.get(stem);
      if (time === undefined || time.length !== analyser.fftSize) {
        time = new Uint8Array(analyser.fftSize);
        this.timeScratch.set(stem, time);
      }
      analyser.getByteTimeDomainData(time);
      return time;
    } catch {
      return null;
    }
  }

  /** Unquantized float32 waveform + sample rate + meter level for the Singing Head input stream. */
  private singingHeadInputFrame(mode: "vocals" | "mix"): {
    samples: Float32Array | null;
    sampleRate: number;
    level: number;
    muted: boolean;
  } {
    try {
      const analyser = mode === "mix"
        ? this.engine.getMixTrackAnalyser()
        : this.engine.getVocalTrackAnalyser();
      if (this.vocalFloatScratch === null || this.vocalFloatScratch.length !== analyser.fftSize) {
        this.vocalFloatScratch = new Float32Array(analyser.fftSize);
      }
      analyser.getFloatTimeDomainData(this.vocalFloatScratch);
      let sq = 0;
      for (let i = 0; i < this.vocalFloatScratch.length; i++) {
        const v = this.vocalFloatScratch[i];
        sq += v * v;
      }
      const rms = Math.sqrt(sq / Math.max(1, this.vocalFloatScratch.length));
      const level = mode === "mix"
        ? Math.min(1, rms * 2.2)
        : this.analyserLevel("vocals");
      return {
        samples: this.vocalFloatScratch,
        sampleRate: analyser.context.sampleRate,
        level,
        muted: this.engine.isSingingInputMuted(mode),
      };
    } catch {
      return {
        samples: null,
        sampleRate: 44100,
        level: 0,
        muted: this.engine.isSingingInputMuted(mode),
      };
    }
  }

  /** True post-glue/post-limiter full mix: level plus waveform for the bear. */
  private masterMixFrame(): { level: number; waveform: Uint8Array | null } {
    try {
      const analyser = this.engine.getMasterAnalyser();
      const n = analyser.fftSize;
      if (this.masterTimeScratch === null || this.masterTimeScratch.length !== n) {
        this.masterTimeScratch = new Uint8Array(n);
      }
      const time = this.masterTimeScratch;
      analyser.getByteTimeDomainData(time);
      let sum = 0;
      for (let i = 0; i < time.length; i++) {
        const v = (time[i] - 128) / 128;
        sum += v * v;
      }
      return {
        level: Math.min(1, Math.sqrt(sum / time.length) * 2.2),
        waveform: time,
      };
    } catch {
      return { level: 0, waveform: null };
    }
  }

  /**
   * Uncaught-error catcher: turns tab-crashes/rejections into a pasteable
   * `[Crash]` line instead of a frozen page. Console-only, no UI writes in
   * the lookahead path (registered once at init, outside the frame loop).
   */
  private installCrashReporter(): void {
    const report = (kind: string, detail: unknown): void => {
      console.info(`[Crash] ${kind}: ${detail instanceof Error ? detail.stack ?? detail.message : String(detail)}`);
      try {
        (window as any).__wubLastCrash = { kind, at: performance.now(), detail: String(detail) };
      } catch {
        // best-effort only
      }
    };
    window.onerror = (event, source, lineno, col, error) => {
      report("window.onerror", error ?? `${String(event)} @${source}:${lineno}:${col}`);
    };
    window.onunhandledrejection = (event) => {
      report("unhandledrejection", event.reason);
      event.preventDefault();
    };
  }

  /**
   * The lightshow is a multi-pass Three.js renderer (two shader passes plus a
   * trail ping-pong).  When its panel is collapsed there is nothing to show,
   * so skip the whole chain rather than paying for hidden render targets.
   */
  private lightshowPanelVisible(): boolean {
    if (this.lightshowPanelEl === undefined) {
      this.lightshowPanelEl = document.getElementById("panel-lightshow");
    }
    const panel = this.lightshowPanelEl;
    if (panel === null) return true;
    return !panel.classList.contains("panel-minimized");
  }

  private frame(_t: number): void {
    // Bound visual/FFT work on high-refresh displays and hidden tabs.
    if (document.hidden || _t - this.lastVisualFrame < 1000 / 60 - 1) {
      requestAnimationFrame((next) => this.frame(next));
      return;
    }
    // Preserve the remainder so 144 Hz does not collapse to a 48 FPS cadence.
    const elapsedRaw = _t - this.lastVisualFrame;
    const elapsed = Number.isFinite(elapsedRaw) ? elapsedRaw : 1000 / 60;
    this.lastVisualFrame = _t - (elapsed % (1000 / 60));
    this.frameLevels.clear();

    // If the compositor cannot sustain ~60 fps, alternate the heavy painters
    // instead of asking every one of them for a full frame at once.  Audio,
    // scheduling, the UI event queue and the glue bus still run every frame.
    this.frameIntervalEma = this.frameIntervalEma * 0.85 + Math.min(50, elapsed) * 0.15;
    if (!this.visualDegraded && this.frameIntervalEma > 21) this.visualDegraded = true;
    else if (this.visualDegraded && this.frameIntervalEma < 17) this.visualDegraded = false;
    const phase = this.visualTick++ & 1;
    const paintA = !this.visualDegraded || phase === 0;
    const paintB = !this.visualDegraded || phase === 1;

    const t = visualNow();
    const audioNow = this.engine.now();
    // Shared glue bus: vocal ducking and kick/bass sidechain, control-rate
    // but sample-scheduled by the AudioParam automation inside the engine.
    this.engine.updateGlue(audioNow);
    this.transitionProbe.update(audioNow, this.engine);

    // Silence watchdog: master bus at digital zero for >1.0 s mid-block
    // means the voices went mute (render-path stall). Rebuild + re-run
    // the current runs now instead of sitting dead until the next block.
    if (this.playing && this.transitionProbe.activeDropoutSecs(audioNow) > 1.0) {
      this.clientConductor?.recoverSilence(audioNow);
    }

    // Kit blocks have per-phrase slots; surface the resulting session BPM.
    if (this.playing && this.clientConductor && !this.bpmSliderDragging) {
      const liveBpm = this.clientConductor.getBpm();
      if (Number.isFinite(liveBpm) && Math.abs(liveBpm - this.bpm) > 0.05) {
        this.bpm = liveBpm;
        const slider = document.getElementById("bpm-slider");
        if (slider instanceof HTMLInputElement) slider.value = String(Math.round(liveBpm));
        const label = document.getElementById("bpm-value");
        if (label !== null) label.textContent = `${Math.round(liveBpm)} BPM`;
      }
    }

    // Kit mode: the base nebula combines the full UMAP slice space and all
    // 2,717 × 4 kit phrase points; the active kit is a bright highlight set.
    if (this.clientConductor && this.clientConductor.getPlaybackMode() === "kit") {
      if (!this.nebulaKitBaseSet && this.nebula !== null) {
        const basePoints = this.clientConductor.getAllKitPhrasePoints();
        if (basePoints.length > 0) {
          this.nebula.setPoints(this.umapPoints.length > 0 ? [...this.umapPoints, ...basePoints] : basePoints);
          this.nebulaKitBaseSet = true;
        }
      }
      const kitId = this.clientConductor.isStemRemix
        ? `cluster:${this.clientConductor.activeClusterId ?? "none"}`
        : (this.clientConductor.activeKit?.track_id ?? null);
      if (kitId !== this.lastNebulaKitId) {
        this.lastNebulaKitId = kitId;
        if (this.clientConductor.isStemRemix && this.nebula !== null) {
          // Rebuild the combined cloud when the cluster or filters change.
          const basePoints = this.clientConductor.getAllKitPhrasePoints();
          if (basePoints.length > 0) {
            this.nebula.setPoints(this.umapPoints.length > 0 ? [...this.umapPoints, ...basePoints] : basePoints);
          }
        }
        const highlight = this.clientConductor.getKitNebulaPoints();
        if (highlight.length > 0 && this.nebula !== null) {
          this.nebula.setHighlight(highlight);
        }
      }
    } else if (this.lastNebulaKitId !== null || this.nebulaKitBaseSet) {
      this.lastNebulaKitId = null;
      this.nebulaKitBaseSet = false;
      this.nebula?.setHighlight([]);
    }

    // Remix mode: show each stem's pinned style region as a ghost ring.
    if (this.clientConductor !== null && this.clientConductor.isStemRemix && this.nebula !== null) {
      for (const state of this.clientConductor.getStemRemixState()) {
        if (state.pin) this.nebula.setPinTarget(state.stem, state.pin.x, state.pin.y);
        else this.nebula.clearPinTarget(state.stem);
        // A long 8/16-bar run advances through several phrases; move the
        // reticle when the stem's actual playing phrase changes.
        if (state.phraseId && this.remixPhraseIds[state.stem] !== state.phraseId) {
          this.remixPhraseIds[state.stem] = state.phraseId;
          this.nebula.setActiveSlice(state.stem, state.x, state.y, visualNow());
          this.nebula.setReticleTarget(state.stem, state.x, state.y, 0.28);
        }
      }
    }

    // Every bar, glide the four reticles to the position representing that bar:
    // the current phrase point plus a small per-stem orbit so each bar reads as
    // a distinct spot in the kit constellation.
    if (
      this.clientConductor !== null &&
      this.clientConductor.getPlaybackMode() === "kit" &&
      !this.clientConductor.isStemRemix &&
      this.kitPoint !== null &&
      this.nebula !== null
    ) {
      const barSec = (60 / Math.max(40, this.bpm)) * 4;
      const barIdx = Math.floor(audioNow / barSec);
      if (barIdx !== this.kitBarIndex) {
        this.kitBarIndex = barIdx;
        const barInBlock = Math.max(0, Math.min(3, Math.floor((audioNow - this.kitBlockAt) / barSec)));
        const baseAngle = (barInBlock * Math.PI) / 2;
        STEMS.forEach((stem, i) => {
          const a = baseAngle + (i * Math.PI) / 2;
          this.nebula!.setReticleTarget(
            stem,
            this.kitPoint!.x + 0.034 * Math.cos(a),
            this.kitPoint!.y + 0.034 * Math.sin(a),
            0.34
          );
        });
      }
    }

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
          if (this.nebula !== null && this.clientConductor?.getPlaybackMode() !== "kit") {
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
    if (this.ringsCtx !== null && paintA) {
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
    if (this.masterCtx !== null && paintB) {
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
    if (this.nebula !== null && paintA) {
      this.nebula.render({
        timeSec: t,
        bpm: this.bpm,
        levels: {
          drums: this.analyserLevel("drums"),
          bass: this.analyserLevel("bass"),
          vocals: this.analyserLevel("vocals"),
          other: this.analyserLevel("other"),
        },
        activeTitle: (this.lastGrooveboxMeta as any)?.activeTrackTitle || this.lastGrooveboxMeta?.vocalSongTitle,
      });
    }
    if (this.psychedelicBg !== null && this.psychedelicBg.getState().enabled && paintA) {
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

    if (this.lightshow !== null && !document.hidden && this.lightshowPanelVisible() && paintA) {
      const drumsLvl = this.analyserLevel("drums");
      const bassLvl = this.analyserLevel("bass");
      const vocalsLvl = this.analyserLevel("vocals");
      const otherLvl = this.analyserLevel("other");
      const mixLvl = Math.max(drumsLvl, bassLvl, vocalsLvl, otherLvl);
      const glue = this.engine.glueState();
      this.lightshow.render({
        timeSec: t,
        bpm: this.bpm,
        section: this.lastGrooveboxMeta?.section ?? "GROOVE_A",
        drums: drumsLvl,
        bass: bassLvl,
        other: otherLvl,
        vocals: vocalsLvl,
        mix: mixLvl,
        drumPunch: glue.drumTransient ? 1.0 : drumsLvl * 0.6,
        freqBins: this.masterFreqScratch,
      });
    }

    if (this.singingHead !== null && paintB) {
      const shFrame = this.singingHeadInputFrame(this.singingHead.getInputMode());
      this.singingHead.update(
        this.lastGrooveboxMeta,
        audioNow,
        shFrame.samples,
        shFrame.sampleRate,
        shFrame.level,
        shFrame.muted,
        this.bpm
      );
    }

    if (this.tamagotchi !== null && paintB) {
      const drumsLvl = this.analyserLevel("drums");
      const bassLvl = this.analyserLevel("bass");
      const vocalsLvl = this.analyserLevel("vocals");
      const otherLvl = this.analyserLevel("other");
      // The true post-glue, post-limiter full mix, not a max of the stems.
      const mixFrame = this.masterMixFrame();
      // Per-instrument time-domain waveforms for the eyes/mouth/heart.
      const vocalWaveform = this.stemWaveform("vocals");
      const drumWaveform = this.stemWaveform("drums");
      const bassWaveform = this.stemWaveform("bass");
      const otherWaveform = this.stemWaveform("other");
      const liveMouth = this.singingHead ? this.singingHead.getLiveMouthState() : null;
      this.tamagotchi.update(
        this.lastGrooveboxMeta,
        audioNow,
        mixFrame.level,
        this.currentStepIndex,
        this.bpm,
        vocalWaveform,
        vocalsLvl,
        drumWaveform,
        bassWaveform,
        drumsLvl,
        bassLvl,
        otherWaveform,
        otherLvl,
        mixFrame.waveform,
        liveMouth,
        this.masterFreqScratch
      );
      if (this.bearOverlayEnabled && this.lsBearOverlayCanvas) {
        this.tamagotchi.drawOverlayInto(this.lsBearOverlayCanvas);
      }
    }

    requestAnimationFrame((next) => this.frame(next));
  }

  // -- FM-1 knob phrase control -------------------------------------------
  //
  // The stock FM-1 knobs never transmit CC. With the CTL-1 firmware, KNOB 1-4
  // send absolute CC in one of four banks; each knob is bound to the matching
  // stem in DJ-surface order (DRUMS, BASS, VOCALS, OTHER). A knob value maps
  // linearly onto that stem's deterministic slice catalogue, and the same
  // reticle-steering path as a mouse drag commits the phrase.

  private initFm1(): void {
    const strip = document.getElementById("fm1-strip");
    if (strip === null) return;
    this.fm1StatusEl = document.getElementById("fm1-status");

    const bankSelect = document.getElementById("fm1-bank-select");
    const selectedBank =
      bankSelect instanceof HTMLSelectElement && isFm1BankId(bankSelect.value)
        ? bankSelect.value
        : FM1_DEFAULT_BANK;
    if (bankSelect instanceof HTMLSelectElement) bankSelect.value = selectedBank;

    this.fm1 = new Fm1MidiInput({
      bank: selectedBank,
      onKnob: (stem, value) => this.handleFm1Knob(stem, value),
      onStatus: (message, tone) => this.setFm1Status(message, tone),
      onDevices: (devices, selectedId) => this.renderFm1Devices(devices, selectedId),
    });

    const connectBtn = document.getElementById("fm1-connect");
    if (connectBtn instanceof HTMLButtonElement) {
      connectBtn.addEventListener("click", () => {
        void this.connectFm1();
      });
    }
    const inputSelect = document.getElementById("fm1-input-select");
    if (inputSelect instanceof HTMLSelectElement) {
      inputSelect.addEventListener("change", () => {
        this.fm1?.selectInput(inputSelect.value === "" ? null : inputSelect.value);
      });
    }
    if (bankSelect instanceof HTMLSelectElement) {
      bankSelect.addEventListener("change", () => {
        if (isFm1BankId(bankSelect.value)) this.fm1?.setBank(bankSelect.value);
      });
    }

    FM1_KNOB_STEMS.forEach((stem, index) => {
      const slider = document.getElementById(`fm1-knob-${stem}`);
      if (slider instanceof HTMLInputElement) {
        this.fm1Sliders[stem] = slider;
        slider.addEventListener("input", () => {
          if (this.fm1 === null) return;
          this.handleFm1Absolute(stem, Number(slider.value));
        });
      }
      const label = document.getElementById(`fm1-val-${stem}`);
      if (label !== null) this.fm1Labels[stem] = label;
    });

    this.renderFm1Devices([], null);
    this.setFm1Status(
      "Stock FM-1 knobs send nothing — flash CTL-1, connect USB, then press CONNECT MIDI.",
    );

    (window as unknown as { __wubFM1: unknown }).__wubFM1 = {
      connect: () => this.connectFm1(),
      disconnect: () => this.fm1?.disconnect(),
      state: () => this.fm1?.state() ?? { supported: false, connected: false },
      bank: (bank: string) => {
        if (isFm1BankId(bank)) this.fm1?.setBank(bank);
      },
      injectCC: (cc: number, value: number) => this.fm1?.injectCC(cc, value) ?? null,
      knob: (index: number, value: number) => {
        const cc = this.fm1?.ccForKnob(index) ?? -1;
        if (FM1_KNOB_STEMS[index] === undefined || cc < 0) return null;
        return this.fm1?.injectCC(cc, value) ?? null;
      },
      slices: (stem: StemName) => this.fm1Slices(stem),
      select: (stem: StemName, slot: number) => this.selectFm1Slot(stem, slot),
      reticles: () => this.nebula?.reticles ?? null,
      remix: () => this.clientConductor?.debugStemRemix() ?? [],
    };
  }

  private async connectFm1(): Promise<boolean> {
    if (this.fm1 === null) return false;
    this.setFm1Status("Requesting Web MIDI access…");
    const ok = await this.fm1.connect();
    if (!ok) return false;
    if (this.fm1.connected) {
      this.setFm1Status(`Listening to ${this.fm1.deviceName ?? "MIDI input"}.`, "ok");
    } else {
      this.setFm1Status(
        `MIDI ready — choose a CTL-1 input (bank ${this.fm1.bankId}).`,
        "ok",
      );
    }
    return true;
  }

  private setFm1Status(message: string, tone: Fm1StatusTone = "info"): void {
    const elx = this.fm1StatusEl ?? document.getElementById("fm1-status");
    if (elx === null) return;
    elx.textContent = message;
    elx.dataset.tone = tone;
  }

  private renderFm1Devices(devices: Fm1MidiDevice[], selectedId: string | null): void {
    const select = document.getElementById("fm1-input-select");
    if (!(select instanceof HTMLSelectElement)) return;
    const previous = selectedId ?? select.value;
    select.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = devices.length === 0 ? "— no MIDI inputs —" : "— none —";
    select.appendChild(none);
    for (const device of devices) {
      const option = document.createElement("option");
      option.value = device.id;
      option.textContent =
        device.state === "connected" ? device.name : `${device.name} (${device.state})`;
      select.appendChild(option);
    }
    select.value =
      previous !== "" && devices.some((device) => device.id === previous) ? previous : "";
  }

  /** Deterministic per-stem slice order, cached per cluster/playback mode. */
  private fm1Slices(stem: StemName): NebulaPoint[] {
    const conductor = this.clientConductor;
    if (conductor === null) return [];
    const key = `${conductor.activeClusterId ?? "-"}|${conductor.getPlaybackMode()}|${
      conductor.isStemRemix ? "remix" : "kit"
    }`;
    const now = Date.now();
    const cached = this.fm1SliceCache[stem];
    if (cached !== undefined && cached.key === key && now - cached.at < 15000) {
      return cached.points;
    }
    const pool = conductor.stemSlices(stem);
    const points = buildSliceOrder(
      pool.length > 0
        ? pool
        : conductor.getAllKitPhrasePoints().filter((point) => point.stem === stem),
    );
    this.fm1SliceCache[stem] = { key, at: now, points };
    return points;
  }

  /**
   * Hardware knob turn: the CTL-1 encoder is endless and sends absolute 0..127
   * values with acceleration. The first event seeds a slot from that value;
   * after that each click scrolls by the (unwrapped) delta, so a slow turn
   * walks through every slice and a fast turn skips a few.
   */
  private handleFm1Knob(stem: StemName, value: number): void {
    const points = this.fm1Slices(stem);
    const count = points.length;
    if (count === 0) {
      this.fm1LastRaw[stem] = value;
      this.syncFm1Cell(stem, value, 0, 0);
      this.setFm1Status("No slices loaded yet — press PLAY, then turn the knob.", "warn");
      return;
    }
    const previous = this.fm1LastRaw[stem];
    const current = this.fm1Slot[stem];
    let slot: number;
    if (current === undefined || previous === undefined) {
      slot = knobValueToSlot(value, count);
    } else {
      slot = scrollSlot(current, value, previous, count);
    }
    this.fm1LastRaw[stem] = value;
    this.applyFm1Slot(stem, value, slot, points);
  }

  /** Virtual slider: a direct 0..127 jump to a slot (also re-anchors scroll). */
  private handleFm1Absolute(stem: StemName, value: number): void {
    const points = this.fm1Slices(stem);
    const count = points.length;
    const slot = knobValueToSlot(value, count);
    this.fm1LastRaw[stem] = value;
    this.applyFm1Slot(stem, value, slot, points);
  }

  private applyFm1Slot(
    stem: StemName,
    value: number,
    slot: number,
    points: NebulaPoint[],
  ): void {
    const count = points.length;
    if (count === 0) {
      this.syncFm1Cell(stem, value, 0, 0);
      this.setFm1Status("No slices loaded yet — press PLAY, then turn the knob.", "warn");
      return;
    }
    const index = Math.max(0, Math.min(count - 1, Math.round(slot)));
    this.syncFm1Cell(stem, value, index, count);
    const point = points[index];
    if (point === undefined) return;
    // Move the reticle immediately (same visual language as a drag) and commit
    // the pinned phrase once the knob settles.
    this.nebula?.setReticleTarget(stem, point.x, point.y, 0.18);
    if (this.fm1Slot[stem] === index) return;
    this.fm1Slot[stem] = index;
    const pending = this.fm1Timers[stem];
    if (pending !== undefined) clearTimeout(pending);
    this.fm1Timers[stem] = setTimeout(() => {
      delete this.fm1Timers[stem];
      this.commitFm1(stem, point);
    }, 180);
    this.setFm1Status(
      `K${FM1_KNOB_STEMS.indexOf(stem) + 1} ${stem.toUpperCase()} → slice ${index + 1}/${count}${
        point.trackId ? ` · ${point.trackId}` : ""
      }`,
    );
  }

  /** Debug/test jump to an exact slot in the stem's ordered slice list. */
  private selectFm1Slot(stem: StemName, slot: number): NebulaPoint | null {
    const points = this.fm1Slices(stem);
    if (points.length === 0) return null;
    const index = Math.max(0, Math.min(points.length - 1, Math.round(slot)));
    const value =
      points.length <= 1 ? 0 : Math.round((index * 127) / (points.length - 1));
    this.fm1LastRaw[stem] = value;
    this.applyFm1Slot(stem, value, index, points);
    return points[index] ?? null;
  }

  /** Commit a knob slice exactly like dropping the stem's reticle. */
  private commitFm1(stem: StemName, point: NebulaPoint): void {
    this.nebula?.setReticle(stem, point.x, point.y);
    this.clientConductor?.setReticleSteering(stem, point.x, point.y, true);
  }

  private syncFm1Cell(stem: StemName, value: number, slot: number, count: number): void {
    const slider = this.fm1Sliders[stem];
    if (slider !== undefined && Number(slider.value) !== value) slider.value = String(value);
    const label = this.fm1Labels[stem];
    if (label !== undefined) {
      label.textContent = count > 0 ? `${value} · ${slot + 1}/${count}` : `${value} · —`;
    }
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
