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
import { NebulaRenderer, drawRings } from "./visualizer";
import type { SliceArcEvent, NebulaPoint } from "./visualizer";
import type { StemName } from "./types";

export interface StemState {
  stem: StemName;
  mute: boolean;
  solo: boolean;
  volume: number;
  chaos: number;
}

const BPM_MIN = 80;
const BPM_MAX = 175;
const DEFAULT_BPM = 128;
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

interface TatumMessage {
  type: "tatum";
  tatumIndex: number;
  startTime: number;
  bpm: number;
  slices: Record<StemName, SliceSpec | null>;
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
    return {
      type: "tatum",
      tatumIndex: Math.round(asNumber(parsed["tatum_index"], 0)),
      startTime: asNumber(parsed["start_time"], 0),
      bpm: asNumber(parsed["bpm"], DEFAULT_BPM),
      slices,
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

  private states: Record<StemName, StemState> = {
    drums: { stem: "drums", mute: false, solo: false, volume: 1, chaos: 0.5 },
    bass: { stem: "bass", mute: false, solo: false, volume: 1, chaos: 0.5 },
    vocals: { stem: "vocals", mute: false, solo: false, volume: 1, chaos: 0.5 },
    other: { stem: "other", mute: false, solo: false, volume: 1, chaos: 0.5 },
  };

  private bpm: number = DEFAULT_BPM;
  private musicalKey: string = "C";
  private playing = false;
  private ws: WebSocket | null = null;
  private manualClose = false;
  private reconnectDelay = RECONNECT_BASE_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private tatumCount = 0;

  // AudioBuffer cache keyed by track_id + segment_id, insertion-ordered LRU.
  private bufferCache = new Map<string, Promise<AudioBuffer>>();
  private ringEvents: SliceArcEvent[] = [];
  private analyserScratch = new Map<StemName, Uint8Array<ArrayBuffer>>();
  private statusEl!: HTMLElement;

  init(): void {
    this.statusEl = el<HTMLElement>("status-line");

    const ringsCanvas = el<HTMLCanvasElement>("rings-canvas");
    const rings = ringsCanvas.getContext("2d");
    if (rings === null) throw new Error("2D context unavailable for #rings-canvas");
    this.ringsCtx = rings;

    this.nebula = new NebulaRenderer(el<HTMLCanvasElement>("nebula-canvas"), {
      onReticleMove: (stem, x, y) => {
        this.send({ type: "set_gravity", stem, x, y });
      },
    });
    this.nebula.setPoints(demoPoints());

    this.bindTransport();
    this.bindMaster();
    for (const stem of STEMS) this.bindStem(stem);

    window.addEventListener("resize", () => {
      if (this.nebula !== null) this.nebula.resize();
    });
    window.addEventListener("beforeunload", () => this.teardown());

    this.setStatus("Ready — press Play.");
    requestAnimationFrame((t) => this.frame(t));
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
    this.applyAllMixerState();
    this.connectWs();
    this.setStatus("Playing — connecting to stream…");
  }

  private async stop(): Promise<void> {
    this.playing = false;
    this.manualClose = true;
    this.clearReconnect();
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

    vol.value = String(st.volume);
    volVal.textContent = st.volume.toFixed(2);
    chaos.value = String(st.chaos);
    chaosVal.textContent = st.chaos.toFixed(2);

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

  private wsUrl(): string {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}${WS_PATH}`;
  }

  private connectWs(): void {
    if (!this.playing || this.ws !== null) return;
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.wsUrl());
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    this.setStatus("Connecting to /ws/stream…");

    ws.onopen = () => {
      this.reconnectDelay = RECONNECT_BASE_MS;
      this.setStatus("Streaming.");
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
      this.setStatus("Stream error — retrying…");
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
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
    if (Number.isFinite(msg.bpm) && msg.bpm > 0) {
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
    if (Number.isFinite(msg.bpm) && msg.bpm > 0) {
      this.bpm = msg.bpm;
      this.engine.setTargetBpm(msg.bpm);
    }
    for (const stem of STEMS) {
      const spec = msg.slices[stem];
      if (spec === null) continue;
      this.pushRingEvent({ stem, trackId: spec.trackId, atSec: visualNow() });
      void this.loadAndSchedule(stem, spec, msg.startTime);
    }
  }

  private pushRingEvent(ev: SliceArcEvent): void {
    this.ringEvents.push(ev);
    if (this.ringEvents.length > RING_EVENT_CAP) {
      this.ringEvents.splice(0, this.ringEvents.length - RING_EVENT_CAP);
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

  private async loadAndSchedule(stem: StemName, spec: SliceSpec, startOffset: number): Promise<void> {
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
    const offset = Number.isFinite(startOffset) ? startOffset : 0;
    const when = this.engine.now() + Math.max(0.01, offset);
    try {
      this.engine.scheduleSlice(stem, buffer, when, {
        rate: spec.rate,
        detuneCents: spec.detuneCents,
        fadeMs: 3,
      });
    } catch {
      // Scheduling races (e.g. dispose during stop); safe to drop.
    }
  }

  private async fetchAndDecode(url: string): Promise<AudioBuffer> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Slice fetch failed: ${res.status}`);
    const bytes = await res.arrayBuffer();
    return await this.engine.decodeWav(bytes);
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
      drawRings(this.ringsCtx, { events: this.ringEvents, bpm: this.bpm, timeSec: t });
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
