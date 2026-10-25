// Phase 3 entry: AudioEngine + NebulaRenderer + WebSocket client wiring.
//
// Expected DOM ids (must stay in sync with frontend/index.html):
//   Playback:        btn-play, btn-stop, status-line
//   Master:          bpm-slider (80-175), bpm-value, key-select
//   Canvases:        rings-canvas, nebula-canvas
//   Per stem <s>:    <s>-mute, <s>-solo, <s>-volume, <s>-vol-val,
//                    <s>-chaos, <s>-chaos-val
//   where <s> is one of drums | bass | vocals | other.
//
// Server protocol (JSON over /ws/stream):
//   inbound  {type:"hello", stems:[...], bpm, sample_rate}
//   inbound  {type:"tatum", tatum_index, start_time, bpm, slices:
//             {stem: {track_id, segment_id, start, duration, url,
//                     detune_cents, rate} | null}}
//   outbound {type:"set", stem, chaos}
//   outbound {type:"set_bpm", bpm}
//   outbound {type:"set_key", key}            (provisional, for key-select)
//   outbound {type:"set_gravity", stem, x, y} (provisional, for reticles)
//
// Timing assumption: `start_time` is an audio-clock-relative offset in
// seconds from message receipt, i.e. play at ctx.currentTime + start_time.
// Late (negative/small) values are clamped to now + 10 ms so slices still
// sound instead of throwing.
import { AudioEngine, STEMS } from "./audio_engine";
import { NebulaRenderer, drawRings, drawWaveformAnalysis, drawMasterAnalysis } from "./visualizer";
import type { SliceArcEvent, NebulaPoint, WaveformStemData } from "./visualizer";
import type { StemName } from "./types";
import { PsychedelicBackground } from "./psychedelic_bg";
import type { TripMode, TripPalette } from "./psychedelic_bg";
import { VERSION_INFO } from "./version";
import { ClientGrooveboxConductor } from "./client_ml_sequencer";

interface BackendConfig {
  httpBase: string;
  wsUrl: string;
  raw: string;
}

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
const CACHE_CAP = 300;
const WS_PATH = "/ws/stream";
const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8000;
const RING_EVENT_CAP = 240;
const RING_WINDOW_SEC = 8;

// ---------------------------------------------------------------------------
// Inbound message types (narrowed from unknown at the socket boundary).
// ---------------------------------------------------------------------------

interface HelloMessage {
  type: "hello";
  stems: StemName[];
  bpm: number;
  sampleRate: number;
}

interface SliceSpec {
  trackId: string;
  segmentId: string;
  start: number;
  duration: number;
  url: string;
  detuneCents: number;
  rate: number;
}

export interface GrooveboxMeta {
  section: string;
  phraseBar: number;
  stepInBar: number;
  totalBars?: number;
  isFill: boolean;
  key?: string;
  energy?: number;
  sectionBarsElapsed?: number;
  sectionTotalBars?: number;
  nextSection?: string;
  sectionBarsLeft?: number;
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
}

interface UIEvent {
  atAudioSec: number;
  meta: GrooveboxMeta;
  step: number;
  totalSteps: number;
}

interface TatumMessage {
  type: "tatum";
  tatumIndex: number;
  startTime: number;
  bpm: number;
  slices: Record<StemName, SliceSpec | null>;
  groovebox?: GrooveboxMeta;
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

function parseSliceSpec(v: unknown): SliceSpec | null {
  if (!isRecord(v)) return null;
  const trackId = asString(v["track_id"], "");
  const segmentId = asString(v["segment_id"], "");
  const url = asString(v["url"], "");
  if (trackId === "" || segmentId === "" || url === "") return null;
  return {
    trackId,
    segmentId,
    start: asNumber(v["start"], 0),
    duration: asNumber(v["duration"], 0),
    url,
    detuneCents: asNumber(v["detune_cents"], 0),
    rate: asNumber(v["rate"], 1),
  };
}

function parseServerMessage(raw: string): HelloMessage | TatumMessage | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const kind = parsed["type"];
  if (kind === "hello") {
    const seen: StemName[] = [];
    const stems = parsed["stems"];
    if (Array.isArray(stems)) {
      for (const s of stems) {
        const stem = asStem(s as unknown);
        if (stem !== null && !seen.includes(stem)) seen.push(stem);
      }
    }
    return {
      type: "hello",
      stems: seen,
      bpm: asNumber(parsed["bpm"], DEFAULT_BPM),
      sampleRate: asNumber(parsed["sample_rate"], 44100),
    };
  }
  if (kind === "tatum") {
    const slicesRaw: unknown = parsed["slices"];
    const slices = {} as Record<StemName, SliceSpec | null>;
    for (const stem of STEMS) {
      const spec = isRecord(slicesRaw) ? parseSliceSpec(slicesRaw[stem]) : null;
      slices[stem] = spec;
    }
    let gb: GrooveboxMeta | undefined = undefined;
    const gbRaw = parsed["groovebox"];
    if (isRecord(gbRaw)) {
      gb = {
        section: asString(gbRaw["section"], "GROOVE_A"),
        phraseBar: asNumber(gbRaw["phrase_bar"], 1),
        stepInBar: asNumber(gbRaw["step_in_bar"], 1),
        totalBars: asNumber(gbRaw["total_bars"], 0),
        isFill: Boolean(gbRaw["is_fill"]),
        key: asString(gbRaw["key"], "8B (C major)"),
        energy: asNumber(gbRaw["energy"], 0.6),
        sectionBarsElapsed: asNumber(gbRaw["section_bars_elapsed"], 1),
        sectionTotalBars: asNumber(gbRaw["section_total_bars"], 16),
        nextSection: asString(gbRaw["next_section"], "BUILDUP"),
        sectionBarsLeft: asNumber(gbRaw["section_bars_left"], 15),
        cycleMode: asString(gbRaw["cycle_mode"], "8x16"),
        patternInCycle: asNumber(gbRaw["pattern_in_cycle"], 1),
        patternsPerCycle: asNumber(gbRaw["patterns_per_cycle"], 8),
        stepInPattern: asNumber(gbRaw["step_in_pattern"], 1),
        stepsPerPattern: asNumber(gbRaw["steps_per_pattern"], 16),
        cycleTotalSteps: asNumber(gbRaw["cycle_total_steps"], 128),
        cycleProgress: asNumber(gbRaw["cycle_progress"], 0.0),
        patterns: isRecord(gbRaw["patterns"]) ? (gbRaw["patterns"] as Record<string, string>) : {},
        patternGrooves: isRecord(gbRaw["pattern_grooves"]) ? (gbRaw["pattern_grooves"] as Record<string, string>) : {},
        patternGrids: isRecord(gbRaw["pattern_grids"]) ? (gbRaw["pattern_grids"] as Record<string, number[]>) : {},
        patternTrackIds: isRecord(gbRaw["pattern_track_ids"]) ? (gbRaw["pattern_track_ids"] as Record<string, string>) : {},
        patternIds: isRecord(gbRaw["pattern_ids"]) ? (gbRaw["pattern_ids"] as Record<string, string>) : {},
        patternRatings: isRecord(gbRaw["pattern_ratings"]) ? (gbRaw["pattern_ratings"] as Record<string, number>) : {},
        patternAudioScores: isRecord(gbRaw["pattern_audio_scores"]) ? (gbRaw["pattern_audio_scores"] as Record<string, number>) : {},
        vocalLyric: asString(gbRaw["vocal_lyric"] || gbRaw["vocalLyric"], ""),
        vocalSongTitle: asString(gbRaw["vocal_song_title"] || gbRaw["vocalSongTitle"], ""),
        vocalSyllable: asString(gbRaw["vocal_syllable"] || gbRaw["vocalSyllable"], ""),
        vocalVowel: asString(gbRaw["vocal_vowel"] || gbRaw["vocalVowel"], ""),
        vocalNote: asString(gbRaw["vocal_note"] || gbRaw["vocalNote"], ""),
      };
    }
    return {
      type: "tatum",
      tatumIndex: Math.round(asNumber(parsed["tatum_index"], 0)),
      startTime: asNumber(parsed["start_time"], 0),
      bpm: asNumber(parsed["bpm"], DEFAULT_BPM),
      slices,
      groovebox: gb,
    };
  }
  return null;
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

// Deterministic demo projection so the nebula is populated before the
// server streams real UMAP coordinates (Phase 4). Seeded PRNG, one gaussian
// cluster per stem.
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
    drums: { x: 0.25, y: 0.3 },
    bass: { x: 0.7, y: 0.3 },
    vocals: { x: 0.3, y: 0.72 },
    other: { x: 0.72, y: 0.7 },
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

  private bpm: number = DEFAULT_BPM;
  private userSetBpm = false;
  private musicalKey: string = "C";
  private playing = false;
  private ws: WebSocket | null = null;
  private manualClose = false;
  private reconnectDelay = RECONNECT_BASE_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private tatumCount = 0;
  private localStepCount = 0;
  private nextTatumTime = 0;
  private tatumQueue: TatumMessage[] = [];
  private schedulerTimer: ReturnType<typeof setInterval> | null = null;

  // AudioBuffer cache keyed by track_id + segment_id, insertion-ordered LRU.
  private bufferCache = new Map<string, Promise<AudioBuffer>>();
  private segmentPositions = new Map<string, { x: number; y: number }>();
  private trackStemPositions = new Map<string, { x: number; y: number }>();
  private activeSlices: Partial<Record<StemName, SliceSpec>> = {};
  private timeScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private freqScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private masterTimeScratch: Uint8Array<ArrayBuffer> | null = null;
  private masterFreqScratch: Uint8Array<ArrayBuffer> | null = null;
  private ringEvents: SliceArcEvent[] = [];
  private analyserScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private statusEl!: HTMLElement;
  private lastGrooveboxMeta: GrooveboxMeta | null = null;
  private clientConductor: ClientGrooveboxConductor | null = null;
  private engineMode: "direct_ml" | "websocket_cloud" = "direct_ml";
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
        this.send({ type: "set_gravity", stem, x, y });
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

    this.bindTransport();
    this.bindBackendConfig();
    this.bindMaster();
    for (const stem of STEMS) this.bindStem(stem);
    this.bindFeedbackControls();
    this.bindGrooveboxCycleControls();

    // Initialize Autonomous In-Browser ML Sequencer
    try {
      this.clientConductor = new ClientGrooveboxConductor(this.engine);
      void this.clientConductor.loadStaticBundle("data");
      this.clientConductor.setMetaCallback((meta, atAudioSec, step, totalSteps) => {
        this.enqueueUIEvent({
          atAudioSec,
          meta,
          step,
          totalSteps: totalSteps || 16,
        });
      });
      this.bindEngineModeToggle();
    } catch (e) {
      console.warn("Client ML sequencer init failed:", e);
    }

    // Sticky BPM: load from localStorage if set, else default to 96 BPM
    const savedBpm = localStorage.getItem("wub_bpm");
    if (savedBpm) {
      const parsed = Number(savedBpm);
      if (Number.isFinite(parsed) && parsed >= BPM_MIN && parsed <= BPM_MAX) {
        this.bpm = parsed;
        this.userSetBpm = true;
        const slider = document.getElementById("bpm-slider");
        if (slider instanceof HTMLInputElement) slider.value = String(this.bpm);
        const label = document.getElementById("bpm-value");
        if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
        this.engine.setTargetBpm(this.bpm);
      }
    } else {
      this.bpm = DEFAULT_BPM;
      this.userSetBpm = true;
    }

    window.addEventListener("resize", () => {
      if (this.nebula !== null) this.nebula.resize();
    });
    window.addEventListener("beforeunload", () => this.teardown());

    this.setStatus("Ready — press Play (" + Math.round(this.bpm) + " BPM)");
    el<HTMLButtonElement>("btn-play").classList.remove("active");
    el<HTMLButtonElement>("btn-stop").classList.add("active");
    requestAnimationFrame((t) => this.frame(t));
  }

  private bindBackendConfig(): void {
    const input = document.getElementById("backend-host-input") as HTMLInputElement | null;
    const btn = document.getElementById("btn-backend-connect") as HTMLButtonElement | null;
    const helpBtn = document.getElementById("btn-backend-help") as HTMLButtonElement | null;
    const dialog = document.getElementById("backend-help-dialog") as HTMLDialogElement | null;
    const closeBtn = document.getElementById("btn-close-dialog") as HTMLButtonElement | null;

    const cfg = this.getBackendConfig();
    if (input) {
      input.value = cfg.raw;
      input.placeholder = window.location.hostname.endsWith("github.io") ? "192.168.1.136:8123" : window.location.host;
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") btn?.click();
      });
    }

    if (helpBtn && dialog) {
      helpBtn.addEventListener("click", () => dialog.showModal());
    }
    if (closeBtn && dialog) {
      closeBtn.addEventListener("click", () => dialog.close());
    }
    if (dialog) {
      dialog.addEventListener("click", (e) => {
        if (e.target === dialog) dialog.close();
      });
    }

    if (btn && input) {
      btn.addEventListener("click", () => {
        const val = input.value.trim();
        if (val) {
          localStorage.setItem("wub_backend", val);
        } else {
          localStorage.removeItem("wub_backend");
        }
        this.updateBackendStatus("connecting");
        if (this.playing) {
          if (this.ws) {
            this.manualClose = true;
            this.ws.close();
            this.ws = null;
          }
          this.connectWs();
        }
        void this.loadUmapPoints();
      });
    }
  }

  private bindEngineModeToggle(): void {
    const btn = document.getElementById("btn-engine-mode") as HTMLButtonElement | null;
    if (!btn) return;

    const updateBtn = () => {
      if (this.engineMode === "direct_ml") {
        btn.innerHTML = "🧠 In-Browser Direct ML";
        btn.classList.remove("cloud-mode");
        btn.title = "Mode: In-Browser Direct ML (100% standalone, zero server required). Click to switch to WebSocket Cloud Sync.";
      } else {
        btn.innerHTML = "⚡ WebSocket Cloud Sync";
        btn.classList.add("cloud-mode");
        btn.title = "Mode: WebSocket Cloud Sync (streaming from Python backend). Click to switch to In-Browser Direct ML.";
      }
    };
    updateBtn();

    btn.addEventListener("click", () => {
      const wasPlaying = this.playing;
      if (wasPlaying) {
        void this.stop();
      }
      this.engineMode = (this.engineMode === "direct_ml") ? "websocket_cloud" : "direct_ml";
      updateBtn();
      if (this.engineMode === "direct_ml") {
        this.setStatus("Mode: 🧠 Direct In-Browser ML Engine (Offline Ready — press Play)");
      } else {
        this.setStatus("Mode: ⚡ WebSocket Cloud Sync (requires Python server running)");
      }
      if (wasPlaying) {
        void this.play();
      }
    });
  }

  private updateBackendStatus(state: "connected" | "connecting" | "disconnected"): void {
    const dot = document.getElementById("backend-status-dot");
    if (!dot) return;
    if (state === "connected") dot.style.background = "#22c55e";
    else if (state === "connecting") dot.style.background = "#eab308";
    else dot.style.background = "#ef4444";
  }

  private getBackendConfig(): BackendConfig {
    const params = new URLSearchParams(window.location.search);
    let raw = params.get("backend") || localStorage.getItem("wub_backend") || "";
    raw = raw.trim();

    if (!raw) {
      if (window.location.hostname.endsWith("github.io")) {
        raw = "https://lamb-programmes-basis-api.trycloudflare.com";
      } else if (
        window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1" ||
        window.location.hostname.startsWith("192.168.") ||
        window.location.hostname.startsWith("10.")
      ) {
        raw = window.location.host;
      }
    }

    if (!raw) {
      return { httpBase: "", wsUrl: "", raw: "" };
    }

    let httpBase = "";
    let wsUrl = "";

    if (/^https?:\/\//i.test(raw)) {
      httpBase = raw.replace(/\/+$/, "");
      const wsProto = raw.toLowerCase().startsWith("https:") ? "wss:" : "ws:";
      wsUrl = httpBase.replace(/^https?:/i, wsProto) + WS_PATH;
    } else if (/^wss?:\/\//i.test(raw)) {
      wsUrl = raw.replace(/\/+$/, "");
      if (!wsUrl.endsWith(WS_PATH)) wsUrl += WS_PATH;
      const httpProto = raw.toLowerCase().startsWith("wss:") ? "https:" : "http:";
      httpBase = raw.replace(/^wss?:/i, httpProto).replace(new RegExp(WS_PATH + "$"), "").replace(/\/+$/, "");
    } else {
      const cleanHost = raw.replace(/\/+$/, "");
      const isCloudflareOrSecure = cleanHost.includes("trycloudflare.com") || cleanHost.includes("ngrok");
      const isLocal =
        cleanHost.startsWith("192.168.") ||
        cleanHost.startsWith("10.") ||
        cleanHost.startsWith("localhost") ||
        cleanHost.startsWith("127.");

      let httpProto = "http:";
      let wsProto = "ws:";

      if (isCloudflareOrSecure) {
        httpProto = "https:";
        wsProto = "wss:";
      } else if (window.location.protocol === "https:" && !isLocal) {
        httpProto = "https:";
        wsProto = "wss:";
      } else {
        httpProto = "http:";
        wsProto = "ws:";
      }

      httpBase = `${httpProto}//${cleanHost}`;
      wsUrl = `${wsProto}//${cleanHost}${WS_PATH}`;
    }

    return { httpBase, wsUrl, raw };
  }

  private async loadUmapPoints(): Promise<void> {
    try {
      let data: unknown = null;
      const cfg = this.getBackendConfig();
      if (cfg.httpBase) {
        try {
          const res = await fetch(`${cfg.httpBase}/api/umap`);
          if (res.ok) data = (await res.json()) as unknown;
        } catch {
          // Fall back to bundled static coordinates
        }
      }

      if (!Array.isArray(data) || data.length === 0) {
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
        data = fallbackPts;
      }

      if (!Array.isArray(data) || data.length === 0) return;
      const pts: NebulaPoint[] = [];
      for (const item of data) {
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
      drums: { x: 0.25, y: 0.3 },
      bass: { x: 0.7, y: 0.3 },
      vocals: { x: 0.3, y: 0.72 },
      other: { x: 0.72, y: 0.7 },
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
    this.manualClose = false;
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
    this.nextTatumTime = 0;
    this.tatumQueue = [];
    this.uiEventQueue = [];
    this.currentStepIndex = 0;
    this.currentStepAudioTime = 0;

    this.applyAllMixerState();

    if (this.engineMode === "direct_ml") {
      // Direct ML: Ensure any existing WebSocket / cloud scheduler is cleanly deactivated
      if (this.ws !== null) {
        try { this.ws.close(); } catch {}
        this.ws = null;
      }
      this.clearReconnect();
      if (this.schedulerTimer !== null) {
        clearInterval(this.schedulerTimer);
        this.schedulerTimer = null;
      }
      this.setStatus(`Playing — 🧠 Direct In-Browser ML Engine (Sample-Accurate Sequencer @ ${Math.round(this.bpm)} BPM)`);
      this.clientConductor?.start(this.bpm);
    } else {
      // WebSocket cloud mode: Stop client sequencer
      this.clientConductor?.stop();
      if (this.schedulerTimer !== null) clearInterval(this.schedulerTimer);
      this.schedulerTimer = setInterval(() => this.scheduleLoop(), 20);
      this.connectWs();
      this.setStatus("Playing — streaming synchronized stems via WebSocket…");
    }
  }

  private async stop(): Promise<void> {
    this.playing = false;
    this.manualClose = true;
    el<HTMLButtonElement>("btn-play").classList.remove("active");
    el<HTMLButtonElement>("btn-stop").classList.add("active");
    this.clearReconnect();
    if (this.schedulerTimer !== null) {
      clearInterval(this.schedulerTimer);
      this.schedulerTimer = null;
    }
    this.clientConductor?.stop();
    this.tatumQueue = [];
    this.uiEventQueue = [];
    this.currentStepIndex = 0;
    this.currentStepAudioTime = 0;
    this.nextTatumTime = 0;
    this.activeSlices = {};
    if (this.ws !== null) {
      try {
        this.ws.close();
      } catch {
        // Ignore close failures.
      }
      this.ws = null;
    }
    await this.engine.suspend();
    this.setStatus("Stopped.");
  }

  private teardown(): void {
    this.clearReconnect();
    if (this.schedulerTimer !== null) {
      clearInterval(this.schedulerTimer);
      this.schedulerTimer = null;
    }
    if (this.ws !== null) {
      try {
        this.ws.close();
      } catch {
        // Ignore.
      }
      this.ws = null;
    }
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
    value.textContent = `${this.bpm} BPM`;
    slider.addEventListener("input", () => {
      const next = Math.min(BPM_MAX, Math.max(BPM_MIN, Number(slider.value)));
      if (!Number.isFinite(next)) return;
      this.userSetBpm = true;
      this.bpm = next;
      localStorage.setItem("wub_bpm", String(next));
      value.textContent = `${Math.round(next)} BPM`;
      this.engine.setTargetBpm(next);
      this.send({ type: "set_bpm", bpm: next });
      if (this.playing && this.engineMode === "direct_ml" && this.clientConductor) {
        this.clientConductor.setBpm(next);
      }
    });

    const key = el<HTMLSelectElement>("key-select");
    this.musicalKey = key.value !== "" ? key.value : "C";
    key.addEventListener("change", () => {
      this.musicalKey = key.value;
      this.send({ type: "set_key", key: key.value });
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
    if (this.clientConductor) {
      this.clientConductor.recordFeedback(stem, rating);
    }

    const trackId = this.lastGrooveboxMeta?.patternTrackIds?.[stem] || this.activeSlices[stem]?.trackId || "";
    const patternId = this.lastGrooveboxMeta?.patternIds?.[stem] || "";

    // 1. Send via active WebSocket connection
    this.send({
      type: "feedback",
      stem,
      rating,
      track_id: trackId,
      pattern_id: patternId,
    });

    // 2. Fallback REST POST
    const cfg = this.getBackendConfig();
    fetch(`${cfg.httpBase}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stem,
        rating,
        track_id: trackId,
        pattern_id: patternId,
      }),
    }).catch(() => {});

    // 3. User feedback message
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
        this.send({ type: "set_cycle_mode", mode });

        const cfg = this.getBackendConfig();
        fetch(`${cfg.httpBase}/api/groovebox/cycle_mode`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        }).catch(() => {});

        cycleBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.setStatus(`Sequencer cycle set to ${mode.replace("x", " × ")} architecture`);
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
      this.send({ type: "set", stem, chaos: st.chaos });
    });
    synth.addEventListener("input", () => {
      const s = Number(synth.value);
      if (!Number.isFinite(s)) return;
      st.synthBlend = Math.min(1, Math.max(0, s));
      synthVal.textContent = st.synthBlend.toFixed(2);
      this.send({ type: "set_synth", stem, blend: st.synthBlend });
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

  // -- websocket -----------------------------------------------------------

  private connectWs(): void {
    if (!this.playing || this.ws !== null) return;
    const cfg = this.getBackendConfig();
    if (!cfg.wsUrl) {
      this.updateBackendStatus("disconnected");
      this.setStatus("Offline demo mode — enter backend address in Server box to stream.");
      return;
    }

    this.updateBackendStatus("connecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(cfg.wsUrl);
    } catch (err: unknown) {
      this.updateBackendStatus("disconnected");
      const isMixed = window.location.protocol === "https:" && cfg.wsUrl.startsWith("ws:");
      if (isMixed) {
        this.setStatus(
          "⚠️ Mixed Content Blocked: Browser refused unencrypted ws:// from HTTPS. " +
          "Fix: In Chrome, click 🔒 next to URL → Site Settings → Insecure content: Allow, " +
          "or connect directly at http://192.168.1.136:8123"
        );
      } else {
        this.setStatus(`Connection error: ${(err as Error)?.message || "Failed"}`);
      }
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    this.setStatus(`Connecting to ${cfg.wsUrl}…`);

    ws.onopen = () => {
      this.reconnectDelay = RECONNECT_BASE_MS;
      this.setStatus("Streaming.");
      this.updateBackendStatus("connected");
      // Re-sync server-side steering state after (re)connect.
      this.send({ type: "set_bpm", bpm: this.bpm });
      this.send({ type: "set_key", key: this.musicalKey });
      for (const stem of STEMS) {
        this.send({ type: "set", stem, chaos: this.states[stem].chaos });
      }
    };
    ws.onmessage = (ev: MessageEvent) => {
      const data: unknown = ev.data;
      if (typeof data !== "string") return;
      const msg = parseServerMessage(data);
      if (msg === null) return;
      if (msg.type === "hello") this.onHello(msg);
      else this.onTatum(msg);
    };
    ws.onerror = () => {
      this.updateBackendStatus("disconnected");
      const isMixed = window.location.protocol === "https:" && cfg.wsUrl.startsWith("ws:");
      if (isMixed) {
        this.setStatus(
          "⚠️ Mixed Content Blocked: Browser refused unencrypted connection from HTTPS. " +
          "Fix: Click 🔒 in address bar → Site Settings → Insecure content: Allow, " +
          "or open http://192.168.1.136:8123 directly."
        );
      } else {
        this.setStatus("Stream error — retrying…");
      }
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.updateBackendStatus("disconnected");
      if (this.playing && !this.manualClose) {
        this.setStatus("Stream closed — reconnecting…");
        this.scheduleReconnect();
      }
    };
  }

  private clearReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (!this.playing || this.manualClose || this.reconnectTimer !== null) return;
    const delay = this.reconnectDelay;
    this.reconnectDelay = Math.min(RECONNECT_MAX_MS, this.reconnectDelay * 2);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connectWs();
    }, delay);
  }

  private send(payload: Record<string, string | number>): void {
    if (this.ws === null || this.ws.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify(payload));
    } catch {
      // Send failures surface via onclose/onerror; ignore here.
    }
  }

  private onHello(msg: HelloMessage): void {
    if (!this.userSetBpm && Number.isFinite(msg.bpm) && msg.bpm > 0) {
      this.bpm = Math.min(BPM_MAX, Math.max(BPM_MIN, msg.bpm));
      const slider = document.getElementById("bpm-slider");
      if (slider instanceof HTMLInputElement) slider.value = String(this.bpm);
      const label = document.getElementById("bpm-value");
      if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
      this.engine.setTargetBpm(this.bpm);
    }
    const stems = msg.stems.length > 0 ? msg.stems.join(", ") : "none";
    this.setStatus(`Streaming — stems: ${stems}.`);
  }

  private onTatum(msg: TatumMessage): void {
    this.tatumCount += 1;
    if (!this.userSetBpm && Number.isFinite(msg.bpm) && msg.bpm > 0) {
      this.bpm = msg.bpm;
      this.engine.setTargetBpm(msg.bpm);
      const slider = document.getElementById("bpm-slider");
      if (slider instanceof HTMLInputElement) slider.value = String(this.bpm);
      const label = document.getElementById("bpm-value");
      if (label !== null) label.textContent = `${Math.round(this.bpm)} BPM`;
    }
    for (const stem of STEMS) {
      const spec = msg.slices[stem];
      if (spec !== null) {
        this.prefetchSlice(spec);
      }
    }
    this.tatumQueue.push(msg);
    if (this.tatumQueue.length > 64) {
      this.tatumQueue.splice(0, this.tatumQueue.length - 64);
    }
  }

  private prefetchSlice(spec: SliceSpec): void {
    const key = `${spec.trackId}:${spec.segmentId}`;
    if (this.cacheGet(key) === undefined) {
      const pending = this.fetchAndDecode(spec.url);
      this.cacheSet(key, pending);
    }
  }

  private scheduleLoop(): void {
    if (!this.playing) return;
    const now = this.engine.now();
    if (now <= 0) return;

    const lookAheadSec = 0.25;
    const tatumDur = 60.0 / this.bpm / 4.0;

    if (this.nextTatumTime < now) {
      this.nextTatumTime = now + 0.12;
    }

    while (this.nextTatumTime < now + lookAheadSec && this.tatumQueue.length > 0) {
      const msg = this.tatumQueue.shift()!;
      const when = this.nextTatumTime;

      if (msg.groovebox) {
        const step = (msg.groovebox.stepInPattern ?? 1) - 1;
        const totalSteps = msg.groovebox.stepsPerPattern ?? 16;
        this.enqueueUIEvent({
          atAudioSec: when,
          meta: msg.groovebox,
          step,
          totalSteps,
        });
      }

      for (const stem of STEMS) {
        const spec = msg.slices[stem];
        if (spec !== null) {
          this.pushRingEvent({ stem, trackId: spec.trackId, atSec: when });
          void this.loadAndScheduleAt(stem, spec, when, msg.bpm);
        }
      }
      this.nextTatumTime += tatumDur;
    }
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
    if (gb.key) {
      const camelotEl = document.getElementById("struct-camelot");
      if (camelotEl) camelotEl.textContent = gb.key;
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
      nextSecEl.textContent = `NEXT: ${gb.nextSection.replace("_", " ")} IN ${gb.sectionBarsLeft} BARS`;
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

    this.lastGrooveboxMeta = gb;
  }

  private cacheGet(key: string): Promise<AudioBuffer> | undefined {
    const hit = this.bufferCache.get(key);
    if (hit !== undefined) {
      // Refresh LRU position.
      this.bufferCache.delete(key);
      this.bufferCache.set(key, hit);
    }
    return hit;
  }

  private cacheSet(key: string, pending: Promise<AudioBuffer>): void {
    this.bufferCache.set(key, pending);
    while (this.bufferCache.size > CACHE_CAP) {
      const oldest = this.bufferCache.keys().next();
      if (oldest.done === true) break;
      this.bufferCache.delete(oldest.value);
    }
  }

  private async loadAndScheduleAt(
    stem: StemName,
    spec: SliceSpec,
    when: number,
    serverBpm?: number
  ): Promise<void> {
    const key = `${spec.trackId}:${spec.segmentId}`;
    let pending = this.cacheGet(key);
    if (pending === undefined) {
      pending = this.fetchAndDecode(spec.url);
      this.cacheSet(key, pending);
    }
    let buffer: AudioBuffer;
    try {
      buffer = await pending;
    } catch {
      // Evict poisoned entries so a retry refetches.
      this.bufferCache.delete(key);
      return;
    }
    if (!this.playing) return;
    const pos = this.getSliceCoordinates(stem, spec.trackId, spec.segmentId);
    const delayMs = Math.max(0, (when - this.engine.now()) * 1000);
    if (delayMs <= 10) {
      this.activeSlices[stem] = spec;
      if (this.nebula !== null) {
        this.nebula.setActiveSlice(stem, pos.x, pos.y, visualNow());
      }
    } else {
      setTimeout(() => {
        if (!this.playing) return;
        this.activeSlices[stem] = spec;
        if (this.nebula !== null) {
          this.nebula.setActiveSlice(stem, pos.x, pos.y, visualNow());
        }
      }, delayMs);
    }
    try {
      const origRate = spec.rate ?? 1.0;
      const sBpm = serverBpm && serverBpm > 0 ? serverBpm : 126.0;
      const tempoRatio = this.bpm / sBpm;
      const finalRate = origRate * tempoRatio;

      this.engine.scheduleSlice(stem, buffer, when, {
        rate: finalRate,
        detuneCents: spec.detuneCents,
        fadeMs: 4,
      });
    } catch {
      // Scheduling races (e.g. dispose during stop); safe to drop.
    }
  }

  private async fetchAndDecode(url: string): Promise<AudioBuffer> {
    const cfg = this.getBackendConfig();
    const fullUrl = url.startsWith("http") ? url : `${cfg.httpBase}${url}`;
    try {
      const res = await fetch(fullUrl);
      if (!res.ok) throw new Error(`Slice fetch failed: ${res.status}`);
      const bytes = await res.arrayBuffer();
      return await this.engine.decodeWav(bytes);
    } catch (err) {
      if (window.location.protocol === "https:" && fullUrl.startsWith("http:")) {
        this.setStatus(
          "⚠️ Mixed Content Blocked: Browser blocked slice fetch. In Chrome: Site Settings → Insecure Content: Allow."
        );
      }
      throw err;
    }
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
          this.pushRingEvent({
            stem,
            trackId: "kg505_ml",
            atSec: t,
          });
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
        const active = this.activeSlices[stem];
        stemsData[stem] = {
          timeDomain: time,
          frequency: freq,
          level,
          activeTrackId: active?.trackId,
          activeSegmentId: active?.segmentId,
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
