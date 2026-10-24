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
const DEFAULT_BPM = 126;
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
  isFill: boolean;
  patterns: Record<string, string>;
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
        isFill: Boolean(gbRaw["is_fill"]),
        patterns: isRecord(gbRaw["patterns"]) ? (gbRaw["patterns"] as Record<string, string>) : {},
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

  init(): void {
    this.statusEl = el<HTMLElement>("status-line");

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

    this.bindTransport();
    this.bindBackendConfig();
    this.bindMaster();
    for (const stem of STEMS) this.bindStem(stem);

    window.addEventListener("resize", () => {
      if (this.nebula !== null) this.nebula.resize();
    });
    window.addEventListener("beforeunload", () => this.teardown());

    this.setStatus("Ready — press Play.");
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
    try {
      await this.engine.resume();
    } catch (err) {
      this.playing = false;
      this.setStatus(`Audio failed: ${err instanceof Error ? err.message : "unknown error"}`);
      return;
    }
    this.nextTatumTime = 0;
    this.tatumQueue = [];
    if (this.schedulerTimer !== null) clearInterval(this.schedulerTimer);
    this.schedulerTimer = setInterval(() => this.scheduleLoop(), 20);

    this.applyAllMixerState();
    this.connectWs();
    this.setStatus("Playing — streaming synchronized stems…");
  }

  private async stop(): Promise<void> {
    this.playing = false;
    this.manualClose = true;
    this.clearReconnect();
    if (this.schedulerTimer !== null) {
      clearInterval(this.schedulerTimer);
      this.schedulerTimer = null;
    }
    this.tatumQueue = [];
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
      value.textContent = `${Math.round(next)} BPM`;
      this.engine.setTargetBpm(next);
      this.send({ type: "set_bpm", bpm: next });
    });

    const key = el<HTMLSelectElement>("key-select");
    this.musicalKey = key.value !== "" ? key.value : "C";
    key.addEventListener("change", () => {
      this.musicalKey = key.value;
      this.send({ type: "set_key", key: key.value });
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
        const delayMs = Math.max(0, (when - now) * 1000);
        setTimeout(() => {
          if (!this.playing) return;
          this.updateGrooveboxUI(msg.groovebox!);
        }, delayMs);
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

  private pushRingEvent(ev: SliceArcEvent): void {
    this.ringEvents.push(ev);
    if (this.ringEvents.length > RING_EVENT_CAP) {
      this.ringEvents.splice(0, this.ringEvents.length - RING_EVENT_CAP);
    }
  }

  private updateGrooveboxUI(gb: GrooveboxMeta): void {
    // 1. Step in Bar (1..16)
    for (let s = 1; s <= 16; s++) {
      const stepEl = document.getElementById(`gb-step-${s}`);
      if (stepEl) {
        if (s === gb.stepInBar) stepEl.classList.add("active");
        else stepEl.classList.remove("active");
      }
    }

    // 2. Bar in Phrase (1..8)
    const barNumEl = document.getElementById("gb-bar-num");
    if (barNumEl) barNumEl.textContent = String(gb.phraseBar);

    for (let b = 1; b <= 8; b++) {
      const dot = document.getElementById(`gb-dot-${b}`);
      if (dot) {
        if (b === gb.phraseBar) {
          if (gb.isFill) dot.className = "gb-bar-dot fill";
          else dot.className = "gb-bar-dot active";
        } else if (b < gb.phraseBar) {
          dot.className = "gb-bar-dot active";
        } else {
          dot.className = "gb-bar-dot";
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
