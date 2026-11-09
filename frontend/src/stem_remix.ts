// Per-stem remix engine: four independent 4-bar phrase streams sharing one
// bar grid.  Every stem owns a current phrase, a lazy audio cache, a pinned
// style region, and its own Signalsmith Stretch voice.  The engine keeps the
// session key/tempo fixed within a set, transposes any phrase key to the
// session key (up to a tritone), time-stretches any phrase tempo to the
// session grid, and rotates to the next BPM-ladder cluster every 104 bars.
// The candidate pool is the whole shipped bank (all 2,565 phrases) filtered
// only by the stretch/transpose hard limits below; near-tempo/near-key
// phrases are strongly preferred by scoring, far ones surface as the near
// pool is played out, so a long session traverses the full bank.
import { ENGINE_SAMPLE_RATE, type AudioEngine } from "./audio_engine";
import { concatRun, mixToMono } from "./kit_pcm";
import type { StemName } from "./types";
import type { Kit, KitCluster, KitManifest, KitPhrase, TasteScoresManifest } from "./kit_player";
import type { NebulaPoint } from "./visualizer";
import { StretchVoice } from "./stretch_voice";
import {
  ARC_SECTIONS, arcPredictNext, isArcModelArtifact,
  type ArcModelArtifact, type ArcPrediction,
} from "./arc_inference";
import { JointRanker } from "./joint_ranker";
import {
  rhythmMixScore,
  type RhythmEntry, type RhythmMeshModel,
} from "./rhythm_inference";
import {
  progressionScoreByPhraseId,
  type ProgressionContextArtifact, type ProgressionModelArtifact,
} from "./progression_inference";
import {
  targetAt, movementBudgetAt, harmonyPenalty, vocalChangeAllowed, scoreTransition,
  pickSignatures, signatureBonus, timbreBonus, selectTransitionTechnique,
  FLOW_SIGNATURES_ENABLED, FLOW_SIGNATURES_PER_STEM,
  FLOW_SIGNATURE_REPETITION_DISCOUNT, FLOW_TIMBRE_WINDOW_BARS,
  type TransitionTechnique,
} from "./arc_planner";

// Flow policy (audio-side): transition technique is only ever scheduled when
// FLOW_TRANSITIONS_ENABLED is on (arc_planner gate) — the pure selector
// already returns the plain hand-off otherwise.  These constants size the
// *scheduled* automation; all gating lives in arc_planner.selectTransitionTechnique.
const TRANSITION_VOCAL_REVERB_TAIL = 0.22;
const TRANSITION_VOCAL_REVERB_TAIL_SEC = 0.35;

const STEMS: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const PITCHED: readonly StemName[] = ["bass", "other", "vocals"];
// Any pitch class is reachable within a tritone, so every phrase key can be
// transposed to the session key.  Drums are unpitched (no shift).
const MAX_SHIFT: Record<StemName, number> = { drums: 0, bass: 6, other: 6, vocals: 6 };
// 48 bars of 4-bar blocks, 32 bars of 8-bar blocks, one 16-bar block, then an
// 8-bar drumless bridge: 104 bars / 19 blocks, then rotate the cluster.
const ARC_8_BAR_START = 48;
const ARC_16_BAR_START = 80;
const BRIDGE_START_BAR = 96;
const SET_BARS = 104;
const BLOCKS_PER_SET = 19;
const PIN_BARS = 32;
// A pin is a soft style region, not one frozen loop.  Candidate run starts
// inside this radius are preferred; the score then varies phrases within it.
const PIN_RADIUS = 0.2;
const PIN_SIGMA = 0.15;
const PIN_TRACK_PENALTY = 0.75;
const RECENT_BLOCKS = 6;
// Hard stretch limits: the phase vocoder smears transients and dulls highs
// as |rate - 1| grows; past ~±6% the "stretched" quality is clearly audible
// (listening report: rate 0.92 blocks read as filtered). [0.94, 1.06] keeps
// every run inside the transparent band; the 104-bar BPM ladder still brings
// every tempo group in near-tempo across sets, so full-bank traversal does
// not depend on extreme stretches.
const RATE_MIN = 0.94;
const RATE_MAX = 1.06;
const TEMPO_COST_WEIGHT = 2.0;
const KEY_COST_WEIGHT = 0.12;
const CACHE_LIMIT = 24;
// Phrases whose stem is effectively absent are not useful as independent
// streams; keep a floor per stem so a "bass" phrase actually has bass.
const MIN_RMS_DB: Record<StemName, number> = {
  drums: -28, bass: -28, other: -34, vocals: -42,
};
// Boundary fade profiles: snap drums, settle bass quickly, let pads/vocals
// overlap so the cross-song hand-off is felt rather than heard.
const STEM_FADE_IN: Record<StemName, number> = {
  drums: 0.004, bass: 0.014, other: 0.05, vocals: 0.08,
};
const STEM_FADE_OUT: Record<StemName, number> = {
  drums: 0.008, bass: 0.025, other: 0.07, vocals: 0.11,
};

export interface StemPin {
  x: number;
  y: number;
  /** Session bar at which the 32-bar pin releases. */
  untilBar: number;
}

export interface StemRemixState {
  stem: StemName;
  phraseId: string | null;
  trackId: string | null;
  section: string;
  energy: number;
  x: number;
  y: number;
  pin: StemPin | null;
  buffering: boolean;
}

export interface StemBlockResult {
  at: number;
  section: string;
  changed: StemName[];
  blocksPlayed: number;
  bridge: boolean;
  rotation: boolean;
  /** Bars in this block: 4, 8 or 16. */
  blockBars: number;
  /** Duration of this block in seconds. */
  blockDurationSec: number;
  /** One 4-bar slot for the session BPM display (960 / slotSec). */
  slotSec: number;
  /** Planned arc energy for this block, 0..1. */
  targetEnergy: number;
  phrases: Partial<Record<StemName, KitPhrase>>;
  kits: Partial<Record<StemName, Kit>>;
  /** Source track per stem for hidden transition telemetry. */
  stemTracks: Partial<Record<StemName, string>>;
  clusterId: string;
}

interface StemPoolEntry {
  kit: Kit;
  phrase: KitPhrase;
  stem: StemName;
  keyShift: number;
  rate: number;
  /** |rate - 1|: tempo distance from the session grid, for tiered scoring. */
  tempoCost: number;
  /** |keyShift|: transpose distance from the session key, for tiered scoring. */
  keyCost: number;
  x: number;
  y: number;
  energy: number;
  section: string;
  chroma: number[] | null;
  bassRoot: number | null;
  rmsDb: number;
  gain: number;
  /** Ears quality 0..1 (0.5 neutral when no ears sidecar is shipped). */
  earQ: number;
}

interface LoadedStemAudio {
  main: AudioBuffer;
  pre: AudioBuffer | null;
  preDur: number;
}

/** A decoded run of consecutive 4-bar phrases for one stem, as mono PCM.
 *  Produced off-main-thread by the kit audio worker (or by the main-thread
 *  fallback when the worker is unavailable); the voice only ever sees this. */
interface LoadedStemRun {
  /** Pre-roll + concatenated mains, ready for StretchVoice.addBuffers. */
  samples: Float32Array;
  /** Seconds of pre-roll at the head of `samples`. */
  preDur: number;
  /** Decoded seconds per main phrase (input timeline). */
  mainDurations: number[];
  sampleRate: number;
}

/** One phrase x stem the kit audio worker must decode for a run. */
interface WorkerRunItem {
  key: string;
  /** Absolute URL of the phrase audio (the worker resolves its own base). */
  url: string;
  /** Absolute URL of the pre-roll sidecar, when present. */
  preUrl: string | null;
}

/** A watchdog-resume schedule: re-enter a run at the block clock's position. */
interface ResumeHint {
  /** The run's original downbeat (audio time). */
  startedAt: number;
  /** The run's original end (audio time). */
  runEndAt: number;
}

interface CurrentStem {
  entry: StemPoolEntry;
  voice: StretchVoice;
  startedAt: number;
  blockIndex: number;
  /** Consecutive phrases scheduled as one voice. */
  run: StemPoolEntry[];
  /** Output time of each phrase's downbeat, relative to `startedAt`. */
  offsets: number[];
  runEndAt: number;
}

/** A voice that has been scheduled but whose downbeat may still be ahead. */
interface ScheduledRun {
  stem: StemName;
  run: StemPoolEntry[];
  voice: StretchVoice;
  startedAt: number;
  offsets: number[];
  runEndAt: number;
  blockIndex: number;
  manual: boolean;
}

/** The partial manual run that covers a drag until the next block boundary. */
interface ManualPin {
  entry: StemPoolEntry;
  run: StemPoolEntry[];
  voice: StretchVoice;
  startedAt: number;
  runEndAt: number;
  untilBar: number;
}

interface PendingCommit {
  entry: StemPoolEntry;
  at: number;
  partialBar: number;
  requestedAt: number;
}

interface ClusterTaste {
  phraseOrder: string[];
  transition: number[][];
}

function parseKey(name: string): { root: number; minor: boolean } | null {
  const match = /^([A-Ga-g])([#b♯♭]?)\s*(major|minor|maj|min)$/.exec(name.trim());
  if (!match) return null;
  const accidental = match[2] === "♯" ? "#" : match[2] === "♭" ? "B" : match[2].toUpperCase();
  const root = PITCH_CLASS[`${match[1].toUpperCase()}${accidental}`];
  if (root === undefined) return null;
  return { root, minor: match[3].toLowerCase().startsWith("min") };
}

const PITCH_CLASS: Record<string, number> = {
  C: 0, "C#": 1, DB: 1, D: 2, "D#": 3, EB: 3, E: 4, F: 5,
  "F#": 6, GB: 6, G: 7, "G#": 8, AB: 8, A: 9, "A#": 10, BB: 10, B: 11,
};
const CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
const CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10];

function camelot(key: string): { number: number; letter: "A" | "B" } | null {
  const parsed = parseKey(key);
  if (!parsed) return null;
  const table = parsed.minor ? CAMELOT_MINOR : CAMELOT_MAJOR;
  return { number: table[parsed.root], letter: parsed.minor ? "A" : "B" };
}

/**
 * Semitone shift from `phraseKey` to `sessionKey`, in -6..+6.  Any key is
 * reachable (major/minor included); relative major/minor and exact matches
 * naturally give 0.  Mode mismatches are allowed — the harmony scoring
 * (root intervals + chroma overlap) deprioritises clashes instead of
 * excluding the phrase outright, so the whole bank stays eligible.
 */
export function keyShift(sessionKey: string, phraseKey: string): number | null {
  const a = parseKey(sessionKey);
  const b = parseKey(phraseKey);
  if (!a || !b) return null;
  let shift = ((b.root - a.root + 12) % 12);
  if (shift > 6) shift -= 12;
  return shift;
}

function rotateChroma(chroma: readonly number[] | undefined, shift: number): number[] | null {
  if (!Array.isArray(chroma) || chroma.length !== 12) return null;
  const out = new Array<number>(12).fill(0);
  for (let i = 0; i < 12; i++) out[(i + shift + 12) % 12] = chroma[i] ?? 0;
  return out;
}

function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < 12; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

function rootPenalty(root: number | null, reference: number | null): number {
  if (root === null || reference === null) return 0;
  const interval = Math.abs(root - reference) % 12;
  if (interval === 0) return 0;
  if (interval === 5 || interval === 7) return 0.05;
  if (interval === 3 || interval === 4 || interval === 2) return 0.15;
  if (interval === 1 || interval === 11) return 0.35;
  if (interval === 6) return 0.8;
  return 0.25;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** Let the renderer breathe between the heavy synchronous tails of stem
 *  loads (mixToMono + concat + worklet buffer clone, 100-300 ms for 8/16-bar
 *  runs).  Four stems scheduled back-to-back in one macrotask measured
 *  600-1100 ms of main-thread blocking, enough to starve the audio render
 *  thread into a 639 ms mid-block dropout. */
function yieldToMain(): Promise<void> {
  const scheduler = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
  if (scheduler?.yield) return scheduler.yield();
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function hashRoll(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  let t = (hash + 0x6d2b79f5) >>> 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** Candidates presented to the joint ranker per stem (one per track). */
const JOINT_CANDIDATES_PER_STEM = 8;
/** Reserve slots for never-played material inside the shortlist. */
const JOINT_RESERVE = 2;
/** Per-stem phrases excluded after they play, so a winner cannot repeat at once. */
const STEM_HISTORY_AVOID = 16;
/** Policy-score window for near-best exploration in selectJoint. */
const EXPLORE_WINDOW = 0.06;
/** How many near-best joint candidates exploration may choose among. */
const EXPLORE_TOP_K = 5;
/** Pair combinations evaluated per stem pair (top of each sorted list). */
const PAIR_TOP_N = 4;
/** Ears soft-prior weight in jointOptions (good phrases win ties only). */
const EAR_W_JOINT = 0.20;
/** Ears soft-prior weight in heuristic fallbacks (morph/pin). */
const EAR_W_FALLBACK = 0.30;

/**
 * Kit audio decode+concat on a dedicated thread.  Main-thread PCM work (the
 * fetch->decode->mixToMono->concat chain) during 8/16-bar prefetches starved
 * the audio render thread into all-stem dropouts on a loaded box; the worker
 * carries it instead, returning transferable PCM the voice can addBuffers
 * directly.  If the worker is unavailable or hangs, the main-thread fallback
 * loader keeps the session running.
 */
class KitAudioWorkerLoader {
  /** False once construction, the worker error handler, or a timeout fails —
   *  or after repeated error responses (e.g. OfflineAudioContext missing in
   *  workers, which disables decode entirely in that browser). */
  ok = false;
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, {
    resolve: (run: LoadedStemRun | null) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();
  private consecutiveErrors = 0;
  private lastErrorMessage = "";

  start(sampleRate: number): void {
    if (this.ok && this.worker) return;
    // A previous worker disabled by repeated errors is terminated here so a
    // fresh session starts with a clean worker (not one orphan).
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch {
        // Already terminated.
      }
      this.worker = null;
    }
    this.consecutiveErrors = 0;
    try {
      const worker = new Worker(new URL("./kit_audio_worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (ev: MessageEvent) => this.onMessage(ev);
      worker.onerror = () => {
        console.warn("[StemRemix] kit audio worker error; main-thread decode from here");
        this.ok = false;
        this.failAll();
      };
      worker.postMessage({ type: "init", sampleRate });
      this.worker = worker;
      this.ok = true;
      console.info("[StemRemix] kit audio worker ready");
    } catch (err) {
      this.ok = false;
      this.worker = null;
      console.warn("[StemRemix] kit audio worker unavailable; main-thread decode", err);
    }
  }

  dispose(): void {
    try {
      this.worker?.terminate();
    } catch {
      // Already terminated.
    }
    this.worker = null;
    this.ok = false;
    this.failAll();
  }

  private onMessage(ev: MessageEvent): void {
    const msg = ev.data as {
      type?: string;
      id?: number;
      samples?: Float32Array;
      preDur?: number;
      mainDurations?: number[];
      sampleRate?: number;
    } | null | undefined;
    if (!msg || typeof msg.id !== "number") return;
    const pending = this.pending.get(msg.id);
    if (!pending) return;
    this.pending.delete(msg.id);
    clearTimeout(pending.timer);
    if (msg.type === "run" && ArrayBuffer.isView(msg.samples) && Number.isFinite(msg.preDur)) {
      this.consecutiveErrors = 0;
      pending.resolve({
        samples: msg.samples as Float32Array,
        preDur: msg.preDur ?? 0,
        mainDurations: Array.isArray(msg.mainDurations) ? msg.mainDurations : [],
        sampleRate: typeof msg.sampleRate === "number" ? msg.sampleRate : ENGINE_SAMPLE_RATE,
      });
    } else {
      // A real per-file failure (or error message): fail the run so the
      // main-thread loader can try; log the cause so decode bugs surface.
      const message = (msg as { message?: unknown }).message;
      if (typeof message === "string" && message.length > 0) {
        if (message !== this.lastErrorMessage) {
          console.warn(`[StemRemix] kit audio worker error: ${message}`);
        }
        this.lastErrorMessage = message;
      }
      // Repeated error responses mean the worker can never decode in this
      // browser (e.g. no OfflineAudioContext in workers) — disable it so
      // loads go straight to the main thread instead of paying a wasted
      // round-trip and a warning per run.
      this.consecutiveErrors++;
      if (this.consecutiveErrors >= 2 && this.ok) {
        this.ok = false;
        console.warn("[StemRemix] kit audio worker disabled (repeated errors); main-thread decode");
      }
      pending.resolve(null);
    }
  }

  private failAll(): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.resolve(null);
    }
    this.pending.clear();
  }

  load(items: WorkerRunItem[]): Promise<LoadedStemRun | null> {
    if (!this.ok || !this.worker) return Promise.resolve(null);
    return new Promise<LoadedStemRun | null>((resolve) => {
      const id = this.nextId++;
      const timer = setTimeout(() => {
        if (this.pending.delete(id)) {
          console.warn("[StemRemix] kit audio worker load timed out; main-thread decode from here");
          this.ok = false;
          resolve(null);
        }
      }, 5000);
      this.pending.set(id, { resolve, timer });
      try {
        this.worker!.postMessage({ type: "load", id, items });
      } catch {
        this.pending.delete(id);
        clearTimeout(timer);
        this.ok = false;
        resolve(null);
      }
    });
  }
}

export class StemRemixPlayer {
  private readonly engine: AudioEngine;
  private manifest: KitManifest | null = null;
  private baseUrl = "";
  private tasteScores: TasteScoresManifest | null = null;
  private clusters: KitCluster[] = [];
  /** Learned arc model artifacts (wub-arc-model-v1); null until loaded. */
  private arcArtifact: ArcModelArtifact | null = null;
  /** phraseId -> raw 48-d window features for arc context (null until loaded). */
  private arcFeatures: Map<string, number[]> | null = null;
  /** Last <=6 played block phrase ids (drums entry), most recent last. */
  private arcHistory: string[] = [];
  private arcLoggedUnavailable = false;
  private kitById = new Map<string, Kit>();
  private entryById = new Map<string, StemPoolEntry>();

  private cluster: KitCluster | null = null;
  private anchorKit: Kit | null = null;
  private currentAnchorPhrase: KitPhrase | null = null;
  private readonly anchorOpeningBlocks = 2;
  private readonly morphOrder: readonly StemName[] = ["drums", "bass", "other", "vocals"];
  private morphIndex = 0;
  private sessionKey = "C minor";
  private sessionBpm = 128;
  private blockSec = 960 / 128;
  private barSec = this.blockSec / 4;
  /** Duration/bars of the block currently being scheduled or played. */
  private currentBlockSec = this.blockSec;
  private currentBlockBars = 4;
  /** Session bar at the start of the block currently on the grid. */
  private blockStartBar = 0;
  /** Session bar at which the next block begins (0..104). */
  private sessionBar = 0;

  private pools: Record<StemName, StemPoolEntry[]> = {
    drums: [], bass: [], other: [], vocals: [],
  };
  private current: Partial<Record<StemName, CurrentStem>> = {};
  private voicePool: Partial<Record<StemName, StretchVoice[]>> = {};
  /** Decode+concat thread; loads fall back to the main thread without it. */
  private readonly audioWorker = new KitAudioWorkerLoader();
  private voiceIndex: Record<StemName, number> = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private jointRanker: JointRanker | null = null;
  private lastBlockEntries: Partial<Record<StemName, StemPoolEntry>> = {};
  private previousEntries: Partial<Record<StemName, StemPoolEntry>> = {};
  /** Run starts of the last scheduled block, for hold/repeat. */
  private lastBlockRuns: Partial<Record<StemName, StemPoolEntry[]>> = {};
  /** Voices scheduled ahead of their downbeat, promoted by tick(). */
  private scheduledRuns: Partial<Record<StemName, ScheduledRun[]>> = {};
  /** Stems with a partial manual commit awaiting the next block boundary. */
  private manualPins: Partial<Record<StemName, ManualPin>> = {};
  /** Stems whose pin audio is still being fetched; do not double-schedule. */
  private pinning = new Set<StemName>();
  private pinGeneration: Partial<Record<StemName, number>> = {};
  private pending: Partial<Record<StemName, PendingCommit>> = {};
  private pins: Partial<Record<StemName, StemPin>> = {};
  private blocksSinceChange: Record<StemName, number> = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private recent: string[] = [];
  /** Per-stem phrase history (last 32) so each stem's memory is independent. */
  private stemHistory: Record<StemName, string[]> = { drums: [], bass: [], other: [], vocals: [] };
  /** How many times each phrase+stem has been scheduled this session. */
  private playCount = new Map<string, number>();
  /** BPM-ladder walk direction for cluster rotation (+1 up, -1 down). */
  private ladderDirection = 1;
  private anchorRecent: string[] = [];
  private playedOnce = new Set<string>();
  private failedKits = new Set<string>();
  private cache = new Map<string, LoadedStemAudio>();
  private stemNext = new Map<string, number>();
  /** Ears quality keyed `${phraseId}:${stem}` (fallback: phrase q). */
  private earQuality = new Map<string, number>();
  /** Taste quality keyed phraseId, from kits[].phrase_order/quality rows. */
  private phraseQuality = new Map<string, number>();
  /** Signature return phrases per stem (1-2 ids), reselected per cluster. */
  private signatureIds: Record<StemName, Set<string>> = {
    drums: new Set(), bass: new Set(), other: new Set(), vocals: new Set(),
  };
  /** Session bar of each stem's last signature return (-1 = never). */
  private lastSignatureBar: Record<StemName, number> = {
    drums: -1, bass: -1, other: -1, vocals: -1,
  };
  /** Recent kit styles per stem for the timbre-diversity budget.
   *  Entries outside FLOW_TIMBRE_WINDOW_BARS are pruned at selection time. */
  private styleRecent: Array<{ stem: StemName; style: string; bar: number }> = [];
  /** Technique chosen for the block currently being prepared (stem -> plan),
   *  handed from beginBlock to finishBlock -> scheduleRun. */
  private pendingTechniques: Partial<Record<StemName, TransitionTechnique>> = {};

  private sessionStartAt = 0;
  private nextBlockAt = 0;
  private blockIndex = 0;
  /** Session bar of the last real stem-track change, vocal and drum change. */
  private lastMoveBar = 0;
  /** Session bar of the last track change per stem (8-bar cooldown is per stem). */
  private lastChangeBar: Record<StemName, number> = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private lastVocalChangeBar = -1;
  private lastDrumChangeBar = 0;
  /** Scheduled blocks, so a mid-block drag can find the block it is inside. */
  private blockGrid: Array<{ startAt: number; endAt: number; startBar: number; bars: number }> = [];
  private bridgeStep = 0;
  private nextCluster: KitCluster | null = null;
  private lastSection = "";
  private started = false;
  private preparing = false;
  private anchorPrepared: Array<{ stem: StemName; entry: StemPoolEntry; audio: LoadedStemRun }> = [];
  private lastResult: StemBlockResult | null = null;

  public onBlock?: (result: StemBlockResult) => void;
  public onStatus?: (message: string) => void;

  constructor(engine: AudioEngine) {
    this.engine = engine;
  }

  get ready(): boolean {
    return this.cluster !== null;
  }

  get currentCluster(): KitCluster | null {
    return this.cluster;
  }

  get bpm(): number {
    return this.sessionBpm;
  }

  get key(): string {
    return this.sessionKey;
  }

  get blockDurationSec(): number {
    return this.currentBlockSec;
  }

  get blockBars(): number {
    return this.currentBlockBars;
  }

  get sessionBarCount(): number {
    return this.blockStartBar;
  }

  get blocksInSet(): number {
    return this.blockIndex;
  }

  get blocksLeft(): number {
    return Math.max(0, BLOCKS_PER_SET - this.blockIndex);
  }

  get nextBlockTime(): number {
    return this.nextBlockAt;
  }

  /** Attach taste scores after the manifest has loaded. */
  setTasteScores(scores: TasteScoresManifest | null): void {
    this.tasteScores = scores;
    this.stemNext.clear();
    this.earQuality.clear();
    this.phraseQuality.clear();
    if (scores?.kits) {
      for (const kitScores of Object.values(scores.kits)) {
        const order = Array.isArray(kitScores?.phrase_order) ? kitScores.phrase_order : [];
        const quality = Array.isArray(kitScores?.quality) ? kitScores.quality : [];
        for (let i = 0; i < Math.min(order.length, quality.length); i++) {
          const phraseId = order[i];
          const q = Number(quality[i]);
          if (typeof phraseId === "string" && phraseId !== "" && Number.isFinite(q)) {
            this.phraseQuality.set(phraseId, Math.max(0, Math.min(1, q)));
          }
        }
      }
      if (this.phraseQuality.size > 0) {
        console.info(`[StemRemix] Taste quality loaded: ${this.phraseQuality.size} phrases`);
        this.reselectSignatures();
      }
    }
    if (scores?.ears) {
      for (const [phraseId, entry] of Object.entries(scores.ears)) {
        if (entry === null || typeof entry !== "object") continue;
        const q = Number(entry.q);
        if (Number.isFinite(q)) {
          this.earQuality.set(`${phraseId}:*`, Math.max(0, Math.min(1, q)));
          for (const stem of STEMS) {
            const perStemQ = Number(entry.perStem?.[stem]?.q);
            if (Number.isFinite(perStemQ)) {
              this.earQuality.set(`${phraseId}:${stem}`, Math.max(0, Math.min(1, perStemQ)));
            }
          }
        }
      }
      if (this.earQuality.size > 0) {
        console.info(`[StemRemix] Ears quality loaded: ${this.earQuality.size} entries`);
      }
    }
    if (!scores?.stem_next) return;
    for (const [phraseId, byStem] of Object.entries(scores.stem_next)) {
      for (const [stem, list] of Object.entries(byStem)) {
        for (const item of list ?? []) {
          this.stemNext.set(`${phraseId}:${stem}:${item.id}`, item.score);
        }
      }
    }
    if (this.stemNext.size > 0) {
      console.info(`[StemRemix] Per-stem judge scores loaded: ${this.stemNext.size} next entries`);
    }
  }

  /** Load the cluster bank + optional taste scores. */
  async load(basePath: string, taste?: TasteScoresManifest | null): Promise<void> {
    const base = basePath.replace(/\/+$/, "");
    const response = await fetch(`${base}/kit_manifest.json`);
    if (!response.ok) {
      throw new Error(`Kit manifest request failed (${response.status})`);
    }
    const manifest = (await response.json()) as KitManifest;
    if (!Array.isArray(manifest.kits) || manifest.kits.length === 0) {
      throw new Error("Kit manifest has no kits");
    }
    this.manifest = manifest;
    this.baseUrl = base;
    this.tasteScores = taste ?? null;
    this.clusters = Array.isArray(manifest.clusters) ? manifest.clusters : [];
    if (this.clusters.length === 0) {
      throw new Error("Kit manifest has no clusters; per-stem mode needs a cluster bank");
    }
    this.kitById.clear();
    for (const kit of manifest.kits) this.kitById.set(kit.track_id, kit);
    try {
      const ranker = new JointRanker();
      const ok = await ranker.load(base);
      this.jointRanker = ok ? ranker : null;
    } catch {
      this.jointRanker = null;
    }
    // Learned arc artifacts load off the first-audio path; a missing or
    // invalid artifact leaves the planned wave as the only behaviour.
    void this.loadArcArtifacts(base);
    // Learned phrase-syntax successors (gated bonus, never a filter); a
    // missing table keeps the syntax term exactly neutral.
    void this.loadSyntaxSuccessors(base);
    // Rhythm-mesh (onset interlock), chord-progression (harmonic grammar)
    // and vocal-edge (clean swap boundaries) sidecars: all optional, all
    // exactly neutral when absent.  They only re-order near-best mixes.
    void this.loadRhythmArtifacts(base);
    void this.loadProgressionArtifacts(base);
    void this.loadVocalMix(base);
  }

  /** Load the learned arc model + context sidecar (log once, never throw). */
  private async loadArcArtifacts(base: string): Promise<void> {
    try {
      const [modelRes, sideRes] = await Promise.all([
        fetch(`${base}/arc_model.json`),
        fetch(`${base}/arc_context_features.json`),
      ]);
      if (!modelRes.ok || !sideRes.ok) {
        throw new Error(`arc artifact ${modelRes.status}/${sideRes.status}`);
      }
      const model = (await modelRes.json()) as unknown;
      const side = (await sideRes.json()) as {
        feature_dim?: unknown;
        byPhraseId?: unknown;
        version?: unknown;
      };
      if (!isArcModelArtifact(model) || model.config.feature_dim !== 48) {
        throw new Error("invalid arc model artifact");
      }
      const featStd = model.feat_std;
      if (!Array.isArray(featStd) || featStd.length !== 48
        || featStd.some((v) => !Number.isFinite(v) || v <= 0)) {
        throw new Error("invalid arc model feat_std");
      }
      const by = side.byPhraseId;
      if (!by || typeof by !== "object") throw new Error("invalid arc sidecar");
      const map = new Map<string, number[]>();
      for (const [phraseId, row] of Object.entries(by as Record<string, unknown>)) {
        if (Array.isArray(row) && row.length === 48
          && row.every((v) => Number.isFinite(Number(v)))) {
          map.set(phraseId, row.map(Number));
        }
      }
      if (map.size === 0) throw new Error("empty arc sidecar");
      this.arcArtifact = model;
      this.arcFeatures = map;
      console.info(`[StemRemix] Arc model loaded (${map.size} context features)`);
    } catch (err) {
      if (!this.arcLoggedUnavailable) {
        this.arcLoggedUnavailable = true;
        console.warn("[StemRemix] arc model unavailable; planned wave only", err);
      }
    }
  }

  /** Learned phrase-syntax successor table (gated bonus source).
   *  Null when the table is absent or invalid — the syntax term then scores
   *  exactly neutral at 0.5 so playback never depends on the table. */
  private syntaxSuccessors: Map<string, Map<string, number>> | null = null;
  private syntaxLoggedUnavailable = false;

  /**
   * Load the learned phrase-syntax successor table
   * (`syntax_successors.json`, `{ anchor phraseId: { same_section_top,
   * overall_top } }` with per-successor scores from the intra-track syntax
   * model).  Logs once and stays null on any failure.
   */
  private async loadSyntaxSuccessors(base: string): Promise<void> {
    try {
      const res = await fetch(`${base}/syntax_successors.json`);
      if (!res.ok) throw new Error(`syntax successors ${res.status}`);
      const doc = (await res.json()) as {
        successors?: Record<string, {
          same_section_top?: Array<{ to?: unknown; score?: unknown }>;
          overall_top?: Array<{ to?: unknown; score?: unknown }>;
        }>;
      };
      const table = doc.successors;
      if (!table || typeof table !== "object") throw new Error("invalid syntax successors");
      const map = new Map<string, Map<string, number>>();
      for (const [fromId, entry] of Object.entries(table)) {
        const inner = new Map<string, number>();
        for (const list of [entry?.same_section_top, entry?.overall_top]) {
          if (!Array.isArray(list)) continue;
          for (const item of list) {
            const to = typeof item?.to === "string" ? item.to : "";
            const score = Number(item?.score);
            if (!to || !Number.isFinite(score)) continue;
            const prev = inner.get(to);
            if (prev === undefined || score > prev) inner.set(to, score);
          }
        }
        if (inner.size > 0) map.set(fromId, inner);
      }
      if (map.size === 0) throw new Error("empty syntax successors");
      this.syntaxSuccessors = map;
      console.info(`[StemRemix] Syntax successors loaded: ${map.size} anchors`);
    } catch (err) {
      if (!this.syntaxLoggedUnavailable) {
        this.syntaxLoggedUnavailable = true;
        console.warn("[StemRemix] syntax successors unavailable; syntax bonus neutral", err);
      }
    }
  }

  /** Rhythm-mesh pair model (wub-rhythm-mesh-model-v1); null until loaded. */
  private rhythmModel: RhythmMeshModel | null = null;
  /** phraseId -> 4-stem onset grids/fluxes for the mix-fitness bonus. */
  private rhythmEntries: Map<string, Record<StemName, RhythmEntry>> | null = null;
  private rhythmLoggedUnavailable = false;

  /**
   * Load the rhythm-mesh artifacts (`rhythm_scores.json` 64-step onset
   * grids, `rhythm_mesh_model.json` logistic pair models, and
   * `rhythm_flux.f32` step fluxes in phrase order).  Builds the per-phrase
   * entry map the candidate mix-fit bonus reads; logs once and stays null on
   * any failure (the term is then exactly neutral).
   */
  private async loadRhythmArtifacts(base: string): Promise<void> {
    try {
      const [scoresRes, modelRes, fluxRes] = await Promise.all([
        fetch(`${base}/rhythm_scores.json`),
        fetch(`${base}/rhythm_mesh_model.json`),
        fetch(`${base}/rhythm_flux.f32`),
      ]);
      if (!scoresRes.ok || !modelRes.ok || !fluxRes.ok) {
        throw new Error(`rhythm artifact ${scoresRes.status}/${modelRes.status}/${fluxRes.status}`);
      }
      const scores = (await scoresRes.json()) as {
        format?: unknown;
        phrases?: Array<Record<string, unknown>>;
      };
      const model = (await modelRes.json()) as RhythmMeshModel;
      if (scores.format !== "wub-rhythm-scores-v1" || !Array.isArray(scores.phrases)) {
        throw new Error("invalid rhythm scores");
      }
      if (!model || typeof model !== "object" || !model.pairs
        || typeof model.pairs !== "object" || Object.keys(model.pairs).length === 0) {
        throw new Error("invalid rhythm mesh model");
      }
      const fluxBuf = await fluxRes.arrayBuffer();
      const flux = new Float32Array(fluxBuf);
      if (flux.length !== scores.phrases.length * 4 * 64) {
        throw new Error("rhythm flux length mismatch");
      }
      const map = new Map<string, Record<StemName, RhythmEntry>>();
      for (let i = 0; i < scores.phrases.length; i++) {
        const phrase = scores.phrases[i];
        const id = typeof phrase?.id === "string" ? phrase.id : "";
        if (!id) continue;
        const entry = {} as Record<StemName, RhythmEntry>;
        let malformed = false;
        for (let s = 0; s < 4; s++) {
          const stem = STEMS[s];
          const gridStr = typeof phrase[stem] === "string" ? (phrase[stem] as string) : "";
          if (gridStr.length !== 64) { malformed = true; break; }
          const grid = new Uint8Array(64);
          for (let k = 0; k < 64; k++) grid[k] = gridStr.charCodeAt(k) === 49 ? 1 : 0;
          entry[stem] = {
            id,
            grid,
            flux: flux.slice((i * 4 + s) * 64, (i * 4 + s) * 64 + 64),
          };
        }
        if (!malformed && Object.keys(entry).length === 4) map.set(id, entry);
      }
      if (map.size === 0) throw new Error("empty rhythm entries");
      this.rhythmModel = model;
      this.rhythmEntries = map;
      console.info(`[StemRemix] Rhythm mesh loaded (${map.size} phrases)`);
    } catch (err) {
      if (!this.rhythmLoggedUnavailable) {
        this.rhythmLoggedUnavailable = true;
        console.warn("[StemRemix] rhythm mesh unavailable; rhythm bonus neutral", err);
      }
    }
  }

  /** Learned chord-progression model + context; null until loaded. */
  private progressionModel: ProgressionModelArtifact | null = null;
  private progressionContext: ProgressionContextArtifact | null = null;
  private progressionLoggedUnavailable = false;

  /**
   * Load the chord-progression artifacts (`progression_model.json` MLP +
   * Markov tables, `progression_context.json` per-phrase rel_chord/chroma).
   * Logs once and stays null on any failure (the term is then exactly
   * neutral).
   */
  private async loadProgressionArtifacts(base: string): Promise<void> {
    try {
      const [modelRes, ctxRes] = await Promise.all([
        fetch(`${base}/progression_model.json`),
        fetch(`${base}/progression_context.json`),
      ]);
      if (!modelRes.ok || !ctxRes.ok) {
        throw new Error(`progression artifact ${modelRes.status}/${ctxRes.status}`);
      }
      const model = (await modelRes.json()) as ProgressionModelArtifact;
      const ctx = (await ctxRes.json()) as ProgressionContextArtifact;
      if (!model || model.format !== "wub-progression-v1" || !model.mlp || !model.markov) {
        throw new Error("invalid progression model");
      }
      if (!ctx || ctx.format !== "wub-progression-context-v1" || !ctx.byPhraseId
        || typeof ctx.byPhraseId !== "object") {
        throw new Error("invalid progression context");
      }
      if (Object.keys(ctx.byPhraseId).length === 0) throw new Error("empty progression context");
      this.progressionModel = model;
      this.progressionContext = ctx;
      console.info(`[StemRemix] Chord progression loaded (${Object.keys(ctx.byPhraseId).length} phrases)`);
    } catch (err) {
      if (!this.progressionLoggedUnavailable) {
        this.progressionLoggedUnavailable = true;
        console.warn("[StemRemix] chord progression unavailable; progression bonus neutral", err);
      }
    }
  }

  /** phraseId -> per-phrase vocal edge strengths; null until loaded. */
  private vocalEdgeStrength: Map<string, number[]> | null = null;
  /** phraseId -> { starts, ends }: vocal activity at step 0 / step 63 of
   *  the phrase's 64-step grid (from vocal_activity16). */
  private vocalActivity: Map<string, { starts: boolean; ends: boolean }> | null = null;
  /** phraseId -> vocal segment membership { pos, lenBars } where pos is the
   *  0-based phrase index inside its (same-track, boundary-chained) segment
   *  and lenBars its total length in bars. */
  private vocalSegments: Map<string, { pos: number; lenBars: number }> | null = null;
  private vocalLoggedUnavailable = false;

  /**
   * Chain vocal segments from the per-phrase activity grids: consecutive
   * same-track phrases link when the first ends with activity and the next
   * starts with activity — one continuous performance.  Records each
   * phrase's position and total segment length so policy can prefer
   * entering at a segment start and play the whole segment.
   */
  private buildVocalSegments(activity: Map<string, { starts: boolean; ends: boolean }>): void {
    const manifest = this.manifest;
    if (!manifest) return;
    const segments = new Map<string, { pos: number; lenBars: number }>();
    for (const kit of manifest.kits) {
      const ordered = [...kit.phrases].sort(
        (a, b) => (Number(a.start_bar) || 0) - (Number(b.start_bar) || 0),
      );
      let i = 0;
      while (i < ordered.length) {
        // Start of a segment: a phrase with activity that is not a
        // continuation of the previous phrase.
        const start = activity.get(ordered[i].id);
        if (!start || (!start.starts && !start.ends)) { i++; continue; }
        let j = i;
        while (j + 1 < ordered.length) {
          const cur = activity.get(ordered[j].id);
          const next = activity.get(ordered[j + 1].id);
          if (!cur || !next || !cur.ends || !next.starts) break;
          const curBar = Number(ordered[j].start_bar) || 0;
          const nextBar = Number(ordered[j + 1].start_bar) || 0;
          if (nextBar !== curBar + 4) break;
          j++;
        }
        const lenBars = (j - i + 1) * 4;
        for (let k = i; k <= j; k++) {
          segments.set(ordered[k].id, { pos: k - i, lenBars });
        }
        i = j + 1;
      }
    }
    if (segments.size > 0) this.vocalSegments = segments;
  }

  /** Vocal segment membership of one phrase, or null when unknown. */
  private vocalSegmentOf(phraseId: string): { pos: number; lenBars: number } | null {
    return this.vocalSegments?.get(phraseId) ?? null;
  }

  /**
   * Load `vocal_mix.json` (per-phrase mix in/out zones + vocal edge
   * strengths + 64-step activity grids).  Keeps the activity endpoints
   * (active at step 0 / step 63) alongside the edge strengths so vocal
   * segments can be chained, and builds those segments immediately.
   * Logs once and stays null on any failure (the terms are then exactly
   * neutral).
   */
  private async loadVocalMix(base: string): Promise<void> {
    try {
      const res = await fetch(`${base}/vocal_mix.json`);
      if (!res.ok) throw new Error(`vocal mix ${res.status}`);
      const doc = (await res.json()) as Record<string, {
        vocal_edge_strength?: unknown;
        vocal_activity16?: unknown;
      }>;
      if (!doc || typeof doc !== "object") throw new Error("invalid vocal mix");
      const map = new Map<string, number[]>();
      const activity = new Map<string, { starts: boolean; ends: boolean }>();
      for (const [phraseId, entry] of Object.entries(doc)) {
        const edges = entry?.vocal_edge_strength;
        if (Array.isArray(edges) && edges.length > 0
          && edges.every((v) => Number.isFinite(Number(v)))) {
          map.set(phraseId, edges.map(Number));
        }
        const act = entry?.vocal_activity16;
        if (typeof act === "string" && act.length === 64) {
          activity.set(phraseId, { starts: act[0] === "1", ends: act[63] === "1" });
        }
      }
      if (map.size === 0) throw new Error("empty vocal mix");
      this.vocalEdgeStrength = map;
      this.vocalActivity = activity;
      this.buildVocalSegments(activity);
      const longCount = [...(this.vocalSegments?.values() ?? [])]
        .filter((s) => s.lenBars > 4).length;
      console.info(
        `[StemRemix] Vocal edges loaded (${map.size} phrases, ` +
        `${longCount} phrases inside >4-bar vocal segments)`,
      );
    } catch (err) {
      if (!this.vocalLoggedUnavailable) {
        this.vocalLoggedUnavailable = true;
        console.warn("[StemRemix] vocal mix unavailable; vocal-edge bonus neutral", err);
      }
    }
  }

  /**
   * Rhythm-mesh mean pair fit (0..1) of a candidate 4-stem mix, or
   * undefined when the artifacts/entries are missing (exactly neutral).
   */
  private rhythmFitFor(candidate: Partial<Record<StemName, StemPoolEntry>>): number | undefined {
    const model = this.rhythmModel;
    const entries = this.rhythmEntries;
    if (!model || !entries) return undefined;
    const stems = {} as Record<StemName, RhythmEntry>;
    for (const stem of STEMS) {
      const phraseEntries = entries.get(candidate[stem]?.phrase.id ?? "");
      if (!phraseEntries) return undefined;
      stems[stem] = phraseEntries[stem];
    }
    const fit = rhythmMixScore(model, stems);
    return Number.isFinite(fit.mean) ? clamp01(fit.mean) : undefined;
  }

  /**
   * Chord-progression fit (0..1) of the candidate next chord after the
   * recent harmonic context, or undefined when unavailable.  Anchored on
   * the drums phrase (the envelope the arc history already tracks); logprobs
   * are mapped so the uniform 12-class logprob scores exactly 0.5.
   */
  private progressionFor(candidate: Partial<Record<StemName, StemPoolEntry>>): number | undefined {
    const model = this.progressionModel;
    const ctx = this.progressionContext;
    if (!model || !ctx) return undefined;
    const prevId = this.arcHistory[this.arcHistory.length - 1] ?? "";
    const prev2Id = this.arcHistory.length >= 2 ? this.arcHistory[this.arcHistory.length - 2] : null;
    const nextId = candidate.drums?.phrase.id ?? "";
    if (!prevId || !nextId) return undefined;
    const { score, source } = progressionScoreByPhraseId(model, ctx, prevId, prev2Id, nextId);
    if (source === null || !Number.isFinite(score)) return undefined;
    return clamp01(0.5 + (score + Math.log(12)) * 0.22);
  }

  /**
   * Vocal-edge cleanliness (0..1) of a vocal swap: mean of the outgoing
   * end-edge and incoming start-edge strengths, or undefined when the
   * sidecar is absent.  When segment data is available the edges follow
   * real phrase segments — an outgoing phrase that is mid-segment has no
   * clean end, and an incoming phrase that is mid-segment has no clean
   * start — so swaps only land on true segment boundaries.
   */
  private vocalEdgeFor(
    current: Partial<Record<StemName, StemPoolEntry>>,
    candidate: Partial<Record<StemName, StemPoolEntry>>,
  ): number | undefined {
    const edges = this.vocalEdgeStrength;
    if (!edges) return undefined;
    const outId = current.vocals?.phrase.id ?? "";
    const inId = candidate.vocals?.phrase.id ?? "";
    const outEdges = edges.get(outId);
    const inEdges = edges.get(inId);
    if (!outEdges || !inEdges || outEdges.length === 0 || inEdges.length === 0) return undefined;
    const outSeg = this.vocalSegmentOf(outId);
    const inSeg = this.vocalSegmentOf(inId);
    // Segment end = last phrase of its segment; segment start = position 0.
    const outEnd = outSeg !== null
      ? (outSeg.pos === outSeg.lenBars / 4 - 1 ? (outEdges[outEdges.length - 1] ?? 0.5) : 0)
      : (outEdges[outEdges.length - 1] ?? 0.5);
    const inStart = inSeg !== null
      ? (inSeg.pos === 0 ? (inEdges[0] ?? 0.5) : 0)
      : (inEdges[0] ?? 0.5);
    return clamp01((outEnd + inStart) / 2);
  }

  /**
   * Length in bars of the incoming vocal segment when a vocal swap begins
   * a segment (whole performances score more); undefined (neutral) for
   * mid-segment entries or unavailable data.
   */
  private vocalSegmentBarsFor(
    candidate: Partial<Record<StemName, StemPoolEntry>>,
  ): number | undefined {
    const seg = this.vocalSegmentOf(candidate.vocals?.phrase.id ?? "");
    if (!seg || seg.pos !== 0) return undefined;
    return seg.lenBars;
  }

  /** Cosine between two 90-d stem chunks, skipping the keyShift/rate stamp
   *  slots (50/51) the ranker itself writes per entry. */
  private static stemCosine(a: Float32Array, b: Float32Array, offset: number): number {
    let dot = 0, na = 0, nb = 0;
    for (let d = 0; d < 90; d++) {
      if (d === 50 || d === 51) continue;
      const x = a[offset + d];
      const y = b[offset + d];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    return na > 1e-9 && nb > 1e-9 ? dot / Math.sqrt(na * nb) : 0;
  }

  /**
   * Learned next-block arc prediction from the last <=6 heard windows.
   * Null when the artifact (or usable context) is missing — the caller keeps
   * the planned wave.  Energy is the primary channel; the section is adopted
   * upstream only at confident phase boundaries.
   */
  private arcPredict(): ArcPrediction | null {
    const model = this.arcArtifact;
    const features = this.arcFeatures;
    if (!model || !features || this.arcHistory.length === 0) return null;
    const rows: number[][] = [];
    for (const phraseId of this.arcHistory) {
      const row = features.get(phraseId);
      if (row && row.length === 48) rows.push(row);
    }
    if (rows.length === 0) return null;
    const lastEnergy = clamp01(rows[rows.length - 1][32] ?? 0.5);
    return arcPredictNext(model, rows, lastEnergy);
  }

  /** Pick a cluster (optionally by style/id) and preload the opening phrase. */
  async startSession(clusterId?: string, style?: string): Promise<KitCluster> {
    if (this.manifest === null) throw new Error("StemRemixPlayer.load() first");
    const candidates = this.clusters.filter((c) => {
      if (clusterId) return c.id === clusterId;
      if (style) return c.style === style;
      return true;
    });
    const pool = candidates.length > 0 ? candidates : this.clusters;
    // Weight by phrase count so dense clusters are more likely, but keep the
    // smaller wave/house sessions in rotation.
    const total = pool.reduce((sum, c) => sum + Math.max(1, c.pool_phrases), 0);
    let roll = Math.random() * total;
    let chosen = pool[0];
    for (const cluster of pool) {
      roll -= Math.max(1, cluster.pool_phrases);
      if (roll <= 0) {
        chosen = cluster;
        break;
      }
    }
    this.applyCluster(chosen);
    // Fresh session: reset the cross-set traversal memory (applyCluster
    // preserves it across rotations so the full bank keeps rotating).
    this.recent = [];
    this.stemHistory = { drums: [], bass: [], other: [], vocals: [] };
    this.playCount.clear();
    this.playedOnce.clear();
    this.ladderDirection = 1;
    this.nextBlockAt = 0;
    this.blockIndex = 0;
    this.sessionBar = 0;
    this.blockStartBar = 0;
    this.currentBlockBars = 4;
    this.currentBlockSec = this.blockSec;
    this.nextCluster = null;
    this.started = false;
    this.preparing = false;
    this.lastResult = null;
    this.anchorPrepared = [];
    this.lastBlockRuns = {};
    this.scheduledRuns = {};
    this.manualPins = {};
    this.arcHistory = [];
    this.blockGrid = [];
    this.lastMoveBar = 0;
    this.lastChangeBar = { drums: 0, bass: 0, other: 0, vocals: 0 };
    this.lastVocalChangeBar = -1;
    this.lastDrumChangeBar = 0;
    this.styleRecent = [];
    this.lastSignatureBar = { drums: -1, bass: -1, other: -1, vocals: -1 };
    this.pendingTechniques = {};

    // Preload one real song: all four stems from the same kit so the first
    // 4 bars sound like a record before the remix starts. Scheduling waits
    // for the transport's first schedule() call.
    // The kit audio worker carries decode+concat off the main thread; the
    // anchor preload below is its first real load.
    this.audioWorker.start(ENGINE_SAMPLE_RATE);
    const anchor = this.anchorKit ?? this.chooseAnchor();
    if (anchor) {
      const allEntries = this.entriesForKit(anchor);
      const grooveEntries = allEntries.filter((entry) => entry.section === "GROOVE_A");
      const fullEntry = allEntries.find((entry) => STEMS.every((stem) =>
        allEntries.some((candidate) => candidate.stem === stem && candidate.phrase.id === entry.phrase.id)
      ));
      const fullAnySection = anchor.phrases.find((candidate) =>
        STEMS.every((stem) => this.entryById.has(`${candidate.id}:${stem}`))
      );
      // Prefer a real intro phrase when the anchor has one (planned arc).  If
      // the kit has no intro, start near the arc's 0.30 intro energy instead of
      // opening on a drop.
      const introPhrase = anchor.phrases.find((candidate) =>
        candidate.is_intro === true
        && STEMS.every((stem) => this.entryById.has(`${candidate.id}:${stem}`))
      );
      const fullPhrases = anchor.phrases.filter((candidate) =>
        STEMS.every((stem) => this.entryById.has(`${candidate.id}:${stem}`))
      );
      const introEnergy = targetAt(0).energy;
      const phraseEnergy = (candidate: KitPhrase): number => {
        const value = Number(candidate.energy);
        return Number.isFinite(value) ? value : 0.5;
      };
      const lowEnergyPhrase = [...fullPhrases].sort((a, b) =>
        Math.abs(phraseEnergy(a) - introEnergy) - Math.abs(phraseEnergy(b) - introEnergy)
      )[0];
      const run = this.longestRunPhrases(anchor, true);
      const phrase = introPhrase
        ?? lowEnergyPhrase
        ?? (run.length >= 2
          ? run[0]
          : (fullEntry?.phrase ?? fullAnySection ?? grooveEntries[0]?.phrase
            ?? allEntries[0]?.phrase ?? anchor.phrases[0]));
      const entries = allEntries.filter((entry) => entry.phrase.id === phrase.id);
      const byStem = new Map<StemName, StemPoolEntry>();
      for (const entry of entries) {
        if (!byStem.has(entry.stem)) byStem.set(entry.stem, entry);
      }
      const prepared = await Promise.all(STEMS.map(async (stem) => {
        const entry = byStem.get(stem) ?? this.anchorEntry(stem, phrase);
        if (!entry) return null;
        const loaded = await this.prepareEntry(stem, entry);
        return loaded ? { stem, entry: loaded.entry, audio: loaded.audio } : null;
      }));
      this.anchorPrepared = prepared.filter((entry): entry is { stem: StemName; entry: StemPoolEntry; audio: LoadedStemRun } => entry !== null);
      this.lastBlockEntries = {};
      this.lastBlockRuns = {};
      for (const preparedEntry of this.anchorPrepared) {
        this.lastBlockEntries[preparedEntry.stem] = preparedEntry.entry;
        this.lastBlockRuns[preparedEntry.stem] = [preparedEntry.entry];
      }
      this.previousEntries = { ...this.lastBlockEntries };
      this.lastMoveBar = 0;
      this.lastChangeBar = { drums: 0, bass: 0, other: 0, vocals: 0 };
      this.lastVocalChangeBar = this.anchorPrepared.some((entry) => entry.stem === "vocals") ? 0 : -1;
      this.lastDrumChangeBar = 0;
      for (const stem of STEMS) {
        if (!this.anchorPrepared.some((entry) => entry.stem === stem)) {
          console.warn(`[StemRemix] anchor missing ${stem}`);
        }
      }
    }
    console.info(
      `[StemRemix] Session ${chosen.id} • ${this.sessionBpm.toFixed(2)} BPM • ${this.sessionKey} • ` +
      `${this.pools.drums.length} drums / ${this.pools.bass.length} bass / ` +
      `${this.pools.other.length} other / ${this.pools.vocals.length} vocal phrases`
    );
    return chosen;
  }

  private applyCluster(cluster: KitCluster): void {
    this.cluster = cluster;
    this.sessionKey = cluster.key;
    this.sessionBpm = cluster.bpm;
    this.blockSec = 960 / cluster.bpm;
    this.barSec = this.blockSec / 4;
    this.pools = { drums: [], bass: [], other: [], vocals: [] };
    this.entryById.clear();
    // The pool is the whole shipped bank, not just the session cluster's
    // kits: the cluster only fixes the session key/tempo and the anchor song.
    // Every phrase whose stretch rate falls in [RATE_MIN, RATE_MAX] and whose
    // key can be transposed within MAX_SHIFT is eligible, so all 2,565
    // phrases are reachable; tiered scoring prefers near matches (see
    // jointOptions / selectMorph).
    for (const kit of this.kitById.values()) {
      if (this.failedKits.has(kit.track_id)) continue;
      for (const phrase of kit.phrases) {
        const duration = Number(phrase.duration_sec);
        if (!(duration > 0)) continue;
        const rate = duration / this.blockSec;
        if (rate < RATE_MIN || rate > RATE_MAX) continue;
        const tempoCost = Math.abs(rate - 1);
        for (const stem of STEMS) {
          const asset = phrase.stems?.[stem];
          if (!asset || !asset.file) continue;
          let shift = 0;
          if (stem !== "drums") {
            const value = keyShift(this.sessionKey, kit.key);
            if (value === null || Math.abs(value) > MAX_SHIFT[stem]) continue;
            shift = value;
          }
          const rmsDb = Number(asset.rms_db) || -20;
          if (rmsDb < MIN_RMS_DB[stem]) continue;
          const xy = phrase.stem_xy?.[stem];
          // Keep the bass root in session-key space when transposed, so the
          // harmony scoring can reason about shifted phrases instead of
          // treating every shifted root as unknown.
          const rawRoot = phrase.bass_root ?? null;
          const bassRoot = rawRoot === null
            ? null
            : (((rawRoot + shift) % 12) + 12) % 12;
          const entry: StemPoolEntry = {
            kit,
            phrase,
            stem,
            keyShift: shift,
            rate,
            tempoCost,
            keyCost: Math.abs(shift),
            x: xy?.x ?? phrase.x ?? 0.5,
            y: xy?.y ?? phrase.y ?? 0.5,
            energy: Number(phrase.energy) || 0.5,
            section: String(phrase.section ?? "GROOVE_A").toUpperCase(),
            chroma: rotateChroma(phrase.chroma, shift),
            bassRoot,
            rmsDb,
            gain: stemGain(stem, rmsDb),
            earQ: this.earQOf(stem, phrase.id),
          };
          this.pools[stem].push(entry);
          this.entryById.set(`${phrase.id}:${stem}`, entry);
        }
      }
    }
    // Play history is session-scoped, not set-scoped: it survives cluster
    // rotation so the full bank keeps rotating across sets instead of
    // resetting to the same near matches every 104 bars.  startSession()
    // clears it for a fresh session.
    this.anchorRecent = [];
    this.lastBlockEntries = {};
    this.previousEntries = {};
    this.lastBlockRuns = {};
    this.scheduledRuns = {};
    this.manualPins = {};
    this.anchorKit = this.chooseAnchor();
    this.currentAnchorPhrase = null;
    this.morphIndex = 0;
    this.lastMoveBar = 0;
    this.lastChangeBar = { drums: 0, bass: 0, other: 0, vocals: 0 };
    this.lastVocalChangeBar = -1;
    this.lastDrumChangeBar = 0;
    this.pendingTechniques = {};
    // Signatures are per-cluster (session key/tempo): reselect against the
    // fresh pool so the new set anchors on phrases that are legal *here*.
    // Rotation anchors may return immediately; the clocks clear here.
    this.lastSignatureBar = { drums: -1, bass: -1, other: -1, vocals: -1 };
    this.reselectSignatures();
  }

  /** Longest chain of consecutive 4-bar phrases, preferably with all stems. */
  private longestRunPhrases(kit: Kit, requireFull: boolean = false): KitPhrase[] {
    const usable = requireFull
      ? kit.phrases.filter((phrase) =>
          STEMS.every((stem) => this.entryById.has(`${phrase.id}:${stem}`)))
      : kit.phrases;
    const byStart = new Map<number, KitPhrase>();
    for (const phrase of usable) {
      if (typeof phrase.start_bar === "number") byStart.set(phrase.start_bar, phrase);
    }
    let best: KitPhrase[] = [];
    for (const phrase of usable) {
      if (typeof phrase.start_bar !== "number") continue;
      const run = [phrase];
      let bar = phrase.start_bar;
      while (byStart.has(bar + 4)) {
        bar += 4;
        run.push(byStart.get(bar)!);
      }
      if (run.length > best.length) best = run;
    }
    if (best.length === 0 && requireFull) return this.longestRunPhrases(kit, false);
    return best;
  }

  /** Bars in the arc block that starts at this session bar. */
  private blockBarsAt(sessionBar: number): number {
    if (sessionBar < ARC_8_BAR_START) return 4;
    if (sessionBar < ARC_16_BAR_START) return 8;
    if (sessionBar < BRIDGE_START_BAR) return 16;
    return 4; // 8-bar drumless bridge as two 4-bar blocks
  }

  private isBridgeBar(sessionBar: number): boolean {
    return sessionBar >= BRIDGE_START_BAR && sessionBar < SET_BARS;
  }

  /** The session bar at the end of the arc block containing `bar`. */
  private blockEndBar(bar: number): number {
    if (bar < ARC_8_BAR_START) return Math.min(ARC_8_BAR_START, (Math.floor(bar / 4) + 1) * 4);
    if (bar < ARC_16_BAR_START) {
      return Math.min(
        ARC_16_BAR_START,
        ARC_8_BAR_START + (Math.floor((bar - ARC_8_BAR_START) / 8) + 1) * 8,
      );
    }
    if (bar < BRIDGE_START_BAR) return BRIDGE_START_BAR;
    if (bar < SET_BARS) {
      return Math.min(
        SET_BARS,
        BRIDGE_START_BAR + (Math.floor((bar - BRIDGE_START_BAR) / 4) + 1) * 4,
      );
    }
    return SET_BARS;
  }

  /** The scheduled block that contains `now`, if the grid has started. */
  private blockAt(now: number): { startAt: number; endAt: number; startBar: number; bars: number } | null {
    let found: { startAt: number; endAt: number; startBar: number; bars: number } | null = null;
    for (const block of this.blockGrid) {
      if (block.startAt <= now + 0.01) found = block;
      else break;
    }
    return found;
  }

  /**
   * Consecutive phrases from `entry` in its own kit, one per 4 bars.  A short
   * source run is looped instead of freezing the last phrase, so an 8- or
   * 16-bar block still has real musical movement.
   */
  private runFor(entry: StemPoolEntry, phrases: number, fill: boolean = true): StemPoolEntry[] {
    const base: StemPoolEntry[] = [entry];
    const firstDuration = Number(entry.phrase.duration_sec) || 0;
    let bar = entry.phrase.start_bar;
    let current = entry;
    while (base.length < phrases && typeof bar === "number") {
      const nextPhrase = current.kit.phrases.find((candidate) =>
        candidate.start_bar === bar + 4
        && this.entryById.has(`${candidate.id}:${entry.stem}`),
      );
      if (!nextPhrase) break;
      const nextDuration = Number(nextPhrase.duration_sec) || 0;
      if (firstDuration > 0 && nextDuration > 0
        && Math.abs(nextDuration - firstDuration) / firstDuration > 0.01) {
        break; // different tempo; keep the run phase-locked to one 4-bar grid
      }
      bar += 4;
      current = this.entryById.get(`${nextPhrase.id}:${entry.stem}`)!;
      base.push(current);
    }
    if (!fill || base.length >= phrases) return base.slice(0, phrases);
    const loop = [...base];
    let index = 0;
    while (base.length < phrases) {
      base.push(loop[index % loop.length]);
      index += 1;
    }
    return base.slice(0, phrases);
  }

  private runAvailable(entry: StemPoolEntry, phrases: number): boolean {
    return this.runFor(entry, phrases, false).length >= phrases;
  }

  private runDurationSec(run: readonly StemPoolEntry[]): number {
    let total = 0;
    for (const entry of run) total += Number(entry.phrase.duration_sec) || 0;
    return total;
  }

  private playCountOf(stem: StemName, phraseId: string): number {
    return this.playCount.get(`${phraseId}:${stem}`) ?? 0;
  }

  /** Ears quality in 0..1; 0.5 neutral when the sidecar is absent. */
  private earQOf(stem: StemName, phraseId: string): number {
    return this.earQuality.get(`${phraseId}:${stem}`)
      ?? this.earQuality.get(`${phraseId}:*`)
      ?? 0.5;
  }

  private recordPlayed(stem: StemName, entry: StemPoolEntry): void {
    const key = `${entry.phrase.id}:${stem}`;
    this.playCount.set(key, (this.playCount.get(key) ?? 0) + 1);
    const history = this.stemHistory[stem];
    history.push(entry.phrase.id);
    if (history.length > 32) history.splice(0, history.length - 32);
    if (!this.recent.includes(entry.phrase.id)) {
      this.recent.push(entry.phrase.id);
    }
    if (entry.phrase.once) this.playedOnce.add(entry.phrase.id);
  }

  /** (A) True when this phrase+stem is a selected signature return. */
  private isSignature(stem: StemName, phraseId: string): boolean {
    return FLOW_SIGNATURES_ENABLED && this.signatureIds[stem].has(phraseId);
  }

  /** (A) Session bars since this stem's signature last returned (-1 if never). */
  private signatureGapBars(stem: StemName): number {
    const last = this.lastSignatureBar[stem];
    if (last === undefined || last < 0) return -1;
    // sessionBar is the bar of the block currently being selected; at run
    // time blockStartBar lags by the previous block, so the gap must read
    // the selection horizon, not the playing block.
    return Math.max(0, this.sessionBar - last);
  }

  /** (A) Energy-gated bonus for one candidate (0 when pinned/inside the gap). */
  private signatureBonusFor(stem: StemName, entry: StemPoolEntry, targetEnergy: number): number {
    return signatureBonus({
      isSignature: this.isSignature(stem, entry.phrase.id),
      targetEnergy,
      gapBars: this.signatureGapBars(stem),
      pinned: this.pins[stem] !== undefined || this.manualPins[stem] !== undefined,
    });
  }

  /** (C) Timbre-budget term for one candidate (soft bonus, never a ban). */
  private timbreBonusFor(stem: StemName, entry: StemPoolEntry, nowBar: number): number {
    const cutoff = nowBar - FLOW_TIMBRE_WINDOW_BARS;
    let count = 0;
    let blocks = 0;
    for (const item of this.styleRecent) {
      if (item.stem !== stem || item.bar <= cutoff) continue;
      blocks += 1;
      if (item.style === entry.kit.style) count += 1;
    }
    if (blocks <= 0) return 0;
    return timbreBonus(count, blocks);
  }

  /** (C) Record one scheduled block for the 32-bar timbre window. */
  private recordStyle(stem: StemName, style: string, bar: number): void {
    this.styleRecent.push({ stem, style, bar });
    // Bound the log: keep ~24 blocks of headroom including stale entries.
    if (this.styleRecent.length > 128) this.styleRecent.splice(0, this.styleRecent.length - 128);
  }

  /** (A) Reselect 1-2 signatures per stem from shipped signals only.
   *  Pool-relative (entries already rate/key-legal), so a cluster rotation
   *  instantly re-anchors on the new key/tempo.  Missing taste quality falls
   *  back inside pickSignatures (0.5), so this is safe without the sidecar. */
  private reselectSignatures(): void {
    if (!FLOW_SIGNATURES_ENABLED || this.pools.drums.length === 0) return;
    for (const stem of STEMS) {
      const candidates = this.pools[stem].map((entry) => ({
        phraseId: entry.phrase.id,
        trackId: entry.kit.track_id,
        quality: this.phraseQuality.get(entry.phrase.id) ?? null,
        earQ: entry.earQ,
        energy: entry.energy,
        section: entry.section,
        isIntro: entry.phrase.is_intro === true,
        isOutro: entry.phrase.is_outro === true,
        once: entry.phrase.once === true,
      }));
      const picked = pickSignatures(candidates, FLOW_SIGNATURES_PER_STEM);
      if (picked.size > 0) {
        this.signatureIds[stem] = picked;
        if (picked.size === 1) {
          console.info(`[StemRemix] signature ${stem}: ${[...picked].join(",")}`);
        }
      }
    }
    // No clock reset here: applyCluster/startSession clear the clocks for
    // rotation/fresh anchors, while a late setTasteScores() must not erase a
    // mid-set rest gap (a premature return would loop the motif).
  }

  private chooseAnchor(): Kit | null {
    const cluster = this.cluster;
    if (!cluster) return null;
    let best: { kit: Kit; score: number } | null = null;
    for (const kitId of cluster.kits) {
      const kit = this.kitById.get(kitId);
      if (!kit || this.failedKits.has(kitId)) continue;
      const sections = new Set(kit.phrases.map((p) => String(p.section).toUpperCase()));
      let score = 0;
      // The anchor song is transposed to the session key for the opening
      // blocks; a whole-song |shift| >= 4 is the classic "muddy cold open"
      // (v0.2.233 opened at shift 5).  Prefer anchors already near the key.
      const shift = keyShift(this.sessionKey, kit.key);
      if (shift === null) score -= 6;
      else if (Math.abs(shift) <= 2) score += 6;
      else if (Math.abs(shift) <= 3) score += 3;
      else score -= (Math.abs(shift) - 3) * 1.25;
      if (sections.has("GROOVE_A")) score += 2;
      if (sections.has("DROP")) score += 1.5;
      if (sections.has("BUILDUP")) score += 0.5;
      const fullPhraseList = kit.phrases.filter((phrase) =>
        STEMS.every((stem) => this.entryById.has(`${phrase.id}:${stem}`))
      );
      const fullPhrase = fullPhraseList.length > 0;
      if (fullPhrase) score += 2;
      // The planned arc opens on a 0.30 INTRO.  A kit whose quietest
      // full-stem phrase is near that target makes a much better cold open
      // than a kit that only contains drops.
      const energyOf = (phrase: KitPhrase): number => {
        const value = Number(phrase.energy);
        return Number.isFinite(value) ? value : 0.5;
      };
      const lowEnergy = fullPhraseList.length > 0
        ? Math.min(...fullPhraseList.map(energyOf))
        : 0.65;
      if (lowEnergy <= 0.40) score += 2.5;
      else if (lowEnergy <= 0.55) score += 1.0;
      const fullGroove = kit.phrases.some((phrase) =>
        String(phrase.section).toUpperCase() === "GROOVE_A"
        && STEMS.every((stem) => this.entryById.has(`${phrase.id}:${stem}`))
      );
      if (fullGroove) score += 2.5;
      // The planned arc opens on INTRO; prefer an anchor that actually has a
      // full-stem intro phrase instead of starting at a peak-energy drop.
      const fullIntro = kit.phrases.some((phrase) =>
        phrase.is_intro === true
        && STEMS.every((stem) => this.entryById.has(`${phrase.id}:${stem}`))
      );
      if (fullIntro) score += 6.0;
      // Prefer a kit that actually contains the next 4 bars, so the anchor
      // song can advance every 4 bars instead of looping an isolated window.
      const run = this.longestRunPhrases(kit, true);
      if (run.length > 1) score += Math.min(6, run.length) * 0.9;
      score += Math.min(1, kit.phrases.length / 6);
      score += hashRoll(kit.track_id) * 0.3;
      if (best === null || score > best.score) best = { kit, score };
    }
    return best?.kit ?? null;
  }

  private entriesForKit(kit: Kit, section?: string): StemPoolEntry[] {
    const entries: StemPoolEntry[] = [];
    for (const phrase of kit.phrases) {
      if (section && String(phrase.section).toUpperCase() !== section) continue;
      for (const stem of STEMS) {
        const entry = this.entryById.get(`${phrase.id}:${stem}`);
        if (entry) entries.push(entry);
      }
    }
    return entries;
  }

  private bestEntryForPhrase(phrase: KitPhrase, stem: StemName): StemPoolEntry | null {
    return this.entryById.get(`${phrase.id}:${stem}`) ?? null;
  }

  /** Called by the transport's block clock. Prepares at most one block at a
   *  time so async voice creation cannot reorder the bar grid. */
  private lastRecoverySec_ = -60;

  /**
   * Silence watchdog: the master bus sat at digital zero for seconds while
   * scheduling kept logging success — the voices themselves went mute
   * (render-path stall) without the scheduler noticing.  Rebuild the voice
   * pool and re-enter each playing run at the block clock's current
   * position, so the block continues instead of restarting — and never
   * replay the prefetched NEXT block (lastBlockRuns already points at it
   * during prefetch; replaying it jumped the set ahead and cascaded new
   * dropouts in v0.2.235).  Cooldown 5 s, one recovery per dropout.
   */
  recoverSilence(at: number): void {
    if (!this.started || this.preparing || this.cluster === null) return;
    if (at - this.lastRecoverySec_ < 5) return;
    const tasks: Array<{ stem: StemName; run: StemPoolEntry[]; transition?: TransitionTechnique }> = [];
    const resumes = new Map<StemName, ResumeHint>();
    for (const stem of STEMS) {
      const current = this.current[stem];
      if (current && current.run.length > 0 && current.runEndAt > at + 0.2) {
        tasks.push({ stem, run: [...current.run] });
        resumes.set(stem, { startedAt: current.startedAt, runEndAt: current.runEndAt });
      }
    }
    if (tasks.length === 0) return;
    this.lastRecoverySec_ = at;
    for (const stem of STEMS) {
      for (const voice of this.voicePool[stem] ?? []) voice.dispose();
      this.voicePool[stem] = [];
      this.voiceIndex[stem] = 0;
    }
    this.current = {};
    console.info(`[StemRemix] silence watchdog: resuming ${tasks.length} runs @ ${at.toFixed(2)}s`);
    // Guard the rebuild so schedule() cannot start a rival finishBlock while
    // voices are being recreated (voice-pool races -> stem dropouts).
    this.preparing = true;
    void this.finishBlock(tasks, at + 0.15, resumes).finally(() => {
      this.preparing = false;
    });
  }

  /** Called by the transport's block clock. Prepares at most one block at a
   *  time so async voice creation cannot reorder the bar grid. */
  schedule(at: number, section: string): StemBlockResult | null {
    if (this.cluster === null) return null;
    if (!this.started) {
      this.started = true;
      this.sessionStartAt = Math.max(at, this.engine.now() + 0.9);
      this.blockStartBar = 0;
      this.sessionBar = 4;
      this.currentBlockBars = 4;
      this.currentBlockSec = 4 * this.barSec;
      this.nextBlockAt = this.sessionStartAt + this.currentBlockSec;
      this.blockIndex = 1;
      this.blockGrid.push({
        startAt: this.sessionStartAt,
        endAt: this.nextBlockAt,
        startBar: 0,
        bars: 4,
      });
      const result = this.anchorResult(this.sessionStartAt, targetAt(0).phase);
      this.lastResult = result;
      void this.scheduleAnchor();
      return result;
    }
    // Begin the next block early enough to fetch 1-4 phrases per stem, but
    // late enough that a reticle drag has already set its pin.  The lead is a
    // fraction of the upcoming block, capped so the reticle never jumps too
    // far ahead of the audio.  A longer lead keeps a cache-miss fetch+decode
    // from finishing after the downbeat, which would otherwise leave a brief
    // all-stem silence while the already-faded old voices cannot be revived.
    const upcomingBars = this.sessionBar >= SET_BARS
      ? 4
      : this.blockBarsAt(this.sessionBar);
    const upcomingSec = upcomingBars * this.barSec;
    const lead = Math.min(8, Math.max(2.0, upcomingSec * 0.6));
    if (
      !this.preparing &&
      this.nextBlockAt > 0 &&
      this.nextBlockAt < this.engine.now() + lead
    ) {
      const when = this.nextBlockAt;
      const { result, tasks } = this.beginBlock(when, section);
      this.lastResult = result;
      this.preparing = true;
      void this.finishBlock(tasks, when).finally(() => {
        this.preparing = false;
      });
      return result;
    }
    return null;
  }

  private anchorResult(at: number, section: string): StemBlockResult {
    const phrases: Partial<Record<StemName, KitPhrase>> = {};
    const kits: Partial<Record<StemName, Kit>> = {};
    const stemTracks: Partial<Record<StemName, string>> = {};
    for (const prepared of this.anchorPrepared) {
      phrases[prepared.stem] = prepared.entry.phrase;
      kits[prepared.stem] = prepared.entry.kit;
      stemTracks[prepared.stem] = prepared.entry.kit.track_id;
    }
    const normalized = String(section || "GROOVE_A").toUpperCase();
    this.lastSection = normalized;
    this.currentAnchorPhrase = this.anchorPrepared.find((prepared) => prepared.stem === "drums")?.entry.phrase
      ?? this.anchorPrepared[0]?.entry.phrase
      ?? null;
    return {
      at,
      section: normalized,
      changed: [...STEMS],
      blocksPlayed: 0,
      bridge: false,
      rotation: false,
      blockBars: 4,
      blockDurationSec: this.currentBlockSec,
      slotSec: this.blockSec,
      targetEnergy: targetAt(0).energy,
      phrases,
      kits,
      stemTracks,
      clusterId: this.cluster?.id ?? "",
    };
  }

  private async scheduleAnchor(): Promise<void> {
    const at = this.sessionStartAt;
    await Promise.all(this.anchorPrepared.map(async (prepared) => {
      console.info(`[StemRemix] anchor ${prepared.stem} ${prepared.entry.phrase.id}`);
      await this.scheduleRun(prepared.stem, [prepared.entry], prepared.audio, at, 4, false);
    }));
  }

  private beginBlock(at: number, section: string): {
    result: StemBlockResult;
    tasks: Array<{ stem: StemName; run: StemPoolEntry[]; transition?: TransitionTechnique }>;
  } {
    const blocksPlayed = this.blockIndex;

    // End of the 104-bar arc: rotate to a Camelot/tempo neighbour.  The new
    // set starts with its own anchor record, exactly like a fresh session.
    let rotation = false;
    if (this.sessionBar >= SET_BARS) {
      // The new cluster has a different key/tempo; a pinned stem from the old
      // set must not keep playing across the boundary.
      for (const stem of STEMS) {
        const manual = this.manualPins[stem];
        if (manual) manual.voice.fadeOut(at, 0.05);
        delete this.manualPins[stem];
        delete this.pins[stem];
        this.pinning.delete(stem);
        this.pinGeneration[stem] = (this.pinGeneration[stem] ?? 0) + 1;
      }
      const next = this.nextCluster;
      this.nextCluster = null;
      if (next) {
        this.applyCluster(next);
        console.info(`[StemRemix] Cluster → ${next.id} • ${this.sessionBpm.toFixed(2)} BPM • ${this.sessionKey}`);
      } else {
        this.lastBlockEntries = {};
        this.previousEntries = {};
        this.lastBlockRuns = {};
        this.scheduledRuns = {};
        this.manualPins = {};
      }
      this.blockIndex = 0;
      this.sessionStartAt = at;
      this.nextBlockAt = at;
      this.sessionBar = 0;
      rotation = true;
    }

    const startBar = this.sessionBar;
    // Planned 104-bar arc.  The arranger's live section is no longer the
    // target in remix mode; the arc plans the wave, the ranker fills it in.
    // A learned arc model (when present) blends in at gated confidence:
    // energy nudges the plan, section only overrules at confident boundaries.
    const arc = targetAt(startBar);
    let normalized = arc.section;
    const displaySection = arc.phase;
    const bridge = !rotation && this.isBridgeBar(startBar);
    let targetEnergy = arc.energy;
    {
      const learned = this.arcPredict();
      if (learned) {
        if (Math.abs(learned.energy - targetEnergy) <= 0.25) {
          targetEnergy = clamp01(0.5 * targetEnergy + 0.5 * learned.energy);
        }
        if (!bridge && arc.barInPhase === 0 && learned.confidence >= 0.55
          && (ARC_SECTIONS as readonly string[]).includes(learned.section)) {
          normalized = learned.section;
        }
      }
    }
    const sectionChanged = normalized !== this.lastSection;
    this.lastSection = normalized;
    if (bridge && this.nextCluster === null) {
      this.nextCluster = this.chooseNeighbourCluster();
      if (this.nextCluster) {
        console.info(`[StemRemix] Bridge → ${this.nextCluster.id} in 8 bars`);
      }
    }
    const bars = rotation ? 4 : this.blockBarsAt(startBar);
    const blockSec = bars * this.barSec;
    const phrasesPerStem = Math.max(1, Math.round(bars / 4));
    const budget = movementBudgetAt(startBar, bars);

    // Pins are measured in session bars.  Release at the first block boundary
    // at or after 32 bars; the manual hold voice fades at the same boundary.
    for (const stem of STEMS) {
      const pin = this.pins[stem];
      if (pin && startBar >= pin.untilBar) {
        delete this.pins[stem];
        delete this.manualPins[stem];
        // The manual run is 32 bars long; resume from what is actually
        // playing, not from the end of that long run.
        const playing = this.current[stem]?.entry;
        if (playing) {
          this.lastBlockEntries[stem] = playing;
          this.lastBlockRuns[stem] = [playing];
        }
        this.onStatus?.(`${stem} released back to auto-mix`);
      }
    }

    const changed: StemName[] = [];
    const trackChanged: StemName[] = [];
    const phrases: Partial<Record<StemName, KitPhrase>> = {};
    const kits: Partial<Record<StemName, Kit>> = {};
    const stemTracks: Partial<Record<StemName, string>> = {};
    const tasks: Array<{ stem: StemName; run: StemPoolEntry[]; transition?: TransitionTechnique }> = [];

    if (bridge && !rotation) {
      for (const stem of ["drums", "bass"] as const) {
        this.current[stem]?.voice.fadeOut(at, 0.25);
      }
    }

    // Rotation is a fresh anchor; otherwise two opening blocks establish the
    // anchor record.  After that the joint ranker chooses 1-2 stems to move to
    // a different track and the others hold.  If no candidate clears the
    // threshold, the whole mix holds for a block.
    const opening = rotation || (!bridge && this.blockIndex < this.anchorOpeningBlocks);
    const holdAnchor = opening && !rotation && !sectionChanged
      && this.blockIndex % 2 === 1 && this.currentAnchorPhrase !== null
      && !this.hasConsecutiveNext();
    const anchorPhrase = opening
      ? (holdAnchor ? this.currentAnchorPhrase : this.chooseAnchorPhrase(normalized))
      : null;

    const priorEntries = { ...this.lastBlockEntries };
    const forceMove = !opening && !bridge && !rotation
      && startBar - this.lastMoveBar >= budget.maxHoldBars;
    // A stem that has not changed track for maxHoldBars is stale and must be
    // reconsidered even when the global cooldown would otherwise hold it.
    const staleStems = new Set<StemName>();
    if (!opening && !bridge && !rotation) {
      for (const stem of STEMS) {
        if (this.pins[stem] !== undefined || this.manualPins[stem]) continue;
        if (startBar - (this.lastChangeBar[stem] ?? 0) >= budget.maxHoldBars) {
          staleStems.add(stem);
        }
      }
    }
    const rankerReady = this.jointRanker?.ready === true;
    const jointT0 = performance.now();
    const joint = !opening && !bridge && !rotation && rankerReady
      ? this.selectJoint(
        normalized, targetEnergy, phrasesPerStem, startBar, sectionChanged, forceMove,
      )
      : null;
    const jointMs = performance.now() - jointT0;
    if (jointMs > 100) {
      console.info(`[StemRemix] joint select took ${jointMs.toFixed(0)}ms @ bar ${startBar}`);
    }
    // Fallback heuristic when the learned ranker is unavailable, or when a
    // stale stem still has no joint move (below the forced threshold).  Without
    // this, a missing ranker makes the engine advance/loop the anchor run
    // forever, which is exactly the "same slices over and over" failure.
    const useFallback = !opening && !bridge && !rotation
      && (!rankerReady || (joint === null && staleStems.size > 0));

    // Stale stems get morph priority in fallback mode (they've waited
    // longest); the budget cap below keeps simultaneous splices to at most
    // budget.maxChanges stems, matching the joint path's discipline.
    const order: StemName[] = bridge && !rotation
      ? ["other", "vocals"]
      : [...STEMS].sort((a, b) =>
        (staleStems.has(b) ? 1 : 0) - (staleStems.has(a) ? 1 : 0));
    const chosenRuns: Partial<Record<StemName, StemPoolEntry[]>> = {};
    let fallbackMorphs = 0;
    for (const stem of order) {
      let manual = this.manualPins[stem];
      if (manual && manual.runEndAt <= at + 0.05) {
        // The partial manual commit has handed off at this boundary.  Keep
        // pins[stem] and let the normal block scheduler choose a fresh run in
        // the pinned region.
        delete this.manualPins[stem];
        manual = undefined;
      }
      if (manual || this.pinning.has(stem)) {
        // A manual voice or its initial load already covers this block; do not
        // schedule a competing voice while it loads.
        const heldRun = manual?.run ?? this.lastBlockRuns[stem] ?? [];
        if (heldRun.length > 0) chosenRuns[stem] = heldRun;
        const first = manual?.entry ?? heldRun[0];
        if (first) {
          phrases[stem] = first.phrase;
          kits[stem] = first.kit;
          stemTracks[stem] = first.kit.track_id;
        }
        continue;
      }
      let entry: StemPoolEntry | null = null;
      if (this.pins[stem]) {
        entry = this.selectPinnedEntry(stem, this.pins[stem]!, phrasesPerStem)
          ?? this.holdEntry(stem);
      } else if (bridge) {
        const base = this.holdEntry(stem);
        if (base) {
          const pair = this.runFor(base, 2);
          entry = pair[Math.min(this.bridgeStep, pair.length - 1)] ?? pair[0] ?? base;
        }
      } else if (opening) {
        entry = this.anchorEntry(stem, anchorPhrase);
      } else if (joint) {
        entry = joint.entries[stem] ?? this.holdEntry(stem);
      } else if (useFallback) {
        const cooldownBars = phrasesPerStem * 4 <= 4 ? 8 : 0;
        const vocalOk = stem !== "vocals"
          || vocalChangeAllowed(startBar, this.lastVocalChangeBar, sectionChanged);
        const cooledDown = cooldownBars === 0 || sectionChanged
          || staleStems.has(stem)
          || startBar - (this.lastChangeBar[stem] ?? 0) >= cooldownBars;
        const morph = (vocalOk && cooledDown && fallbackMorphs < budget.maxChanges)
          ? this.selectMorph(stem, at, normalized, phrasesPerStem)
          : null;
        if (morph) fallbackMorphs++;
        entry = morph ?? this.holdEntry(stem);
      } else {
        entry = this.holdEntry(stem);
      }
      if (!entry) entry = this.holdEntry(stem);
      if (!entry) continue;

      // A held entry whose source run cannot fill an 8/16-bar block would loop
      // the same 4 bars two to four times inside one block.  Prefer a fresh
      // cross-song run with a real consecutive length instead (at most two
      // such conversions per block, so the movement budget stays close).
      if (phrasesPerStem > 1 && !this.pins[stem] && !joint?.changed.includes(stem)
        && !opening && !bridge && !rotation) {
        const holdLen = this.runFor(entry, phrasesPerStem, false).length;
        const conversions = trackChanged.length + changed.length;
        if (holdLen < phrasesPerStem && conversions < budget.maxChanges * 2) {
          const alt = this.selectMorph(stem, at, normalized, phrasesPerStem);
          if (alt && alt.kit.track_id !== entry.kit.track_id) entry = alt;
        }
      }

      const run = this.runFor(entry, phrasesPerStem);
      chosenRuns[stem] = run;
      const current = this.current[stem];
      const samePhrase = current
        && current.entry.phrase.id === run[0].phrase.id
        && current.entry.kit.track_id === run[0].kit.track_id;
      const sameTrack = current
        && current.entry.kit.track_id === run[0].kit.track_id;
      phrases[stem] = run[0].phrase;
      kits[stem] = run[0].kit;
      stemTracks[stem] = run[0].kit.track_id;
      // (A) + (C) + (B) bookkeeping for this boundary:
      // signature return clock, timbre window, and the DJ technique plan
      // (pure selector; all scheduling happens in scheduleRun at audio time).
      const outgoing = priorEntries[stem] ?? current?.entry;
      const vocalChangedForTech = !sameTrack && stem === "vocals";
      const technique = selectTransitionTechnique({
        stem,
        outgoingEnergy: outgoing?.energy ?? run[0].energy,
        incomingEnergy: run[0].energy,
        incomingRmsDb: run[0].rmsDb,
        targetEnergy,
        isBridge: bridge || rotation,
        vocalChanged: stem === "vocals" ? vocalChangedForTech : undefined,
      });
      if (technique.eqOut || technique.vocalReverbTail || technique.vocalFadeInSec !== null) {
        this.pendingTechniques[stem] = technique;
      } else {
        delete this.pendingTechniques[stem];
      }
      if (!sameTrack && this.isSignature(stem, run[0].phrase.id)) {
        this.lastSignatureBar[stem] = startBar;
        if (targetEnergy >= 0.5) {
          this.onStatus?.(`signature ${stem} returns at bar ${startBar}`);
        }
      }
      this.recordStyle(stem, run[0].kit.style, startBar);
      tasks.push({ stem, run, transition: technique });
      if (!samePhrase) changed.push(stem);
      if (!sameTrack) trackChanged.push(stem);
      for (const runEntry of run) {
        this.recordPlayed(stem, runEntry);
      }
    }
    this.previousEntries = priorEntries;
    this.lastBlockRuns = chosenRuns;
    const boundaryEntries: Partial<Record<StemName, StemPoolEntry>> = {};
    for (const stem of STEMS) {
      const run = chosenRuns[stem];
      if (run && run.length > 0) boundaryEntries[stem] = run[run.length - 1];
      else if (priorEntries[stem]) boundaryEntries[stem] = priorEntries[stem];
    }
    this.lastBlockEntries = boundaryEntries;
    const autoTrackChanged = trackChanged.filter((stem) => this.pins[stem] === undefined);
    if (autoTrackChanged.length > 0) {
      this.lastMoveBar = startBar;
      for (const stem of autoTrackChanged) this.lastChangeBar[stem] = startBar;
      if (autoTrackChanged.includes("vocals")) this.lastVocalChangeBar = startBar;
      if (autoTrackChanged.includes("drums")) this.lastDrumChangeBar = startBar;
    }
    if (opening) {
      this.currentAnchorPhrase = anchorPhrase;
      if (anchorPhrase) {
        this.anchorRecent.push(anchorPhrase.id);
        if (this.anchorRecent.length > 24) this.anchorRecent = this.anchorRecent.slice(-24);
      }
    }
    if (this.recent.length > 48) {
      this.recent = this.recent.slice(-48);
    }
    if (bridge) this.bridgeStep += 1;
    else this.bridgeStep = 0;

    const result: StemBlockResult = {
      at,
      section: displaySection,
      changed,
      blocksPlayed,
      bridge,
      rotation,
      blockBars: bars,
      blockDurationSec: blockSec,
      slotSec: this.blockSec,
      targetEnergy,
      phrases,
      kits,
      stemTracks,
      clusterId: this.cluster?.id ?? "",
    };

    this.blockStartBar = startBar;
    this.currentBlockBars = bars;
    this.currentBlockSec = blockSec;
    this.sessionBar = startBar + bars;
    this.nextBlockAt = at + blockSec;
    this.blockIndex += 1;
    this.blockGrid.push({ startAt: at, endAt: at + blockSec, startBar, bars });
    this.onBlock?.(result);
    return { result, tasks };
  }

  /** True when the kit contains the real next 4 bars (start_bar + 4). */
  private hasConsecutiveNext(): boolean {
    const kit = this.anchorKit;
    const current = this.currentAnchorPhrase;
    if (!kit || !current || typeof current.start_bar !== "number") return false;
    return kit.phrases.some((phrase) =>
      phrase.start_bar === current.start_bar + 4
      && this.entryById.has(`${phrase.id}:drums`),
    );
  }

  /** Assemble the joint state (424-d for v3, 400-d for the old v2 model). */
  private jointState(entries: Partial<Record<StemName, StemPoolEntry>>): Float32Array | null {
    const ranker = this.jointRanker;
    if (!ranker?.ready) return null;
    const stemDim = ranker.stemDim;
    const embOffset = 4 * stemDim;
    const out = new Float32Array(ranker.stateDim);
    const emb = new Float32Array(64);
    for (let i = 0; i < 4; i++) {
      const stem = STEMS[i];
      const entry = entries[stem];
      if (!entry) return null;
      const shift = keyShift(this.sessionKey, entry.kit.key) ?? 0;
      const state = ranker.stateFor(entry.phrase.id, shift, entry.rate);
      if (!state) return null;
      out.set(state.subarray(i * stemDim, (i + 1) * stemDim), i * stemDim);
      for (let k = 0; k < 64; k++) emb[k] += state[embOffset + k] / 4;
    }
    out.set(emb, embOffset);
    return out;
  }

  /** Top cross-song options for one stem, ranked by the per-stem judge. */
  private jointOptions(
    stem: StemName,
    current: StemPoolEntry,
    usedTracks: ReadonlySet<string>,
    phrases: number,
    section: string,
    targetEnergy: number,
  ): StemPoolEntry[] {
    // Per-stem memory, not a short global window: a phrase this stem played
    // in the last STEM_HISTORY_AVOID slots is excluded, so the previous
    // winner cannot simply win again two blocks later.  For 8/16-bar runs we
    // also exclude the exact run just heard.
    const avoid = new Set<string>();
    for (const id of this.stemHistory[stem].slice(-STEM_HISTORY_AVOID)) avoid.add(id);
    if (phrases > 1) {
      for (const runEntry of this.lastBlockRuns[stem] ?? []) avoid.add(runEntry.phrase.id);
    }
    const outroAllowed = this.blockIndex >= BLOCKS_PER_SET - 2;
    const candidates = this.pools[stem].filter((entry) => {
      if (entry.kit.track_id === current.kit.track_id) return false;
      if (this.failedKits.has(entry.kit.track_id)) return false;
      if (usedTracks.has(entry.kit.track_id)) return false;
      if (avoid.has(entry.phrase.id)) return false;
      if (entry.phrase.once && this.playedOnce.has(entry.phrase.id)) return false;
      if (entry.phrase.is_intro && this.blockIndex > 0) return false;
      if (entry.phrase.is_outro && !outroAllowed) return false;
      return true;
    });
    const scored = candidates.map((entry) => {
      const run = this.runFor(entry, phrases, false);
      let legal = run.length >= phrases;
      if (legal) {
        for (const runEntry of run) {
          if (runEntry.phrase.once && this.playedOnce.has(runEntry.phrase.id)) {
            legal = false;
            break;
          }
        }
      }
      const judge = this.judgeScore(stem, current.phrase.id, entry.phrase.id);
      // The per-stem judge only stores top-K links; when those are exhausted
      // (current track/recent filtered) fall back to a musical heuristic so
      // the joint ranker still sees plausible cross-song candidates.
      const style = entry.kit.style === this.cluster?.style ? 0.15 : 0;
      const energy = 0.35 - Math.abs(entry.energy - targetEnergy) * 0.5;
      const sectionScore = entry.section === section ? 0.25 : 0;
      const fallback = style + energy + sectionScore;
      // The judge scores are nearly flat at the top (all ~0.97+ for drums),
      // so a deterministic sort would surface the same handful of tracks
      // every block.  Play-count novelty plus a block-seeded hash breaks
      // those ties toward material that has not played yet, rotating the
      // whole pool through over a set instead of cycling a top-8.
      // Tempo/key tiers keep far stretches rare: near matches win while they
      // are fresh, but their play counts grow until a far, never-played
      // phrase's novelty outweighs its stretch cost and it surfaces.
      // Signature returns pay a discounted novelty decay so they are not
      // scored out for having played before, and under-used timbres earn a
      // small nudge (see arc_planner timbreBonus).
      const plays = this.playCountOf(stem, entry.phrase.id);
      const signatureBonus = this.signatureBonusFor(stem, entry, targetEnergy);
      const isSig = this.isSignature(stem, entry.phrase.id);
      const perceptionPlays = isSig ? plays * (1 - FLOW_SIGNATURE_REPETITION_DISCOUNT) : plays;
      const novelty = (perceptionPlays === 0 ? 0.35 : Math.max(-0.30, -0.12 * perceptionPlays))
        + hashRoll(`${entry.phrase.id}:${this.blockIndex}:${stem}`) * 0.20;
      const tierCost = entry.tempoCost * TEMPO_COST_WEIGHT
        + entry.keyCost * KEY_COST_WEIGHT;
      const earBonus = EAR_W_JOINT * (entry.earQ - 0.5);
      const timbre = this.timbreBonusFor(stem, entry, this.sessionBar);
      return {
        entry,
        full: legal,
        judge,
        score: (judge > 0 ? judge : fallback) + novelty - tierCost + earBonus + signatureBonus + timbre,
      };
    });
    const full = scored.filter((item) => item.full);
    // Prefer a real consecutive run.  Only loop a shorter run when the
    // cluster genuinely has no full-length run for this stem.
    const pool = full.length > 0 ? full : scored;
    pool.sort((a, b) => {
      if (a.full !== b.full) return a.full ? -1 : 1;
      return b.score - a.score;
    });
    // One candidate per track, so the joint ranker sees genuinely different
    // records instead of many windows from the same song.
    const seen = new Set<string>();
    const out: StemPoolEntry[] = [];
    for (const item of pool) {
      const trackId = item.entry.kit.track_id;
      if (seen.has(trackId)) continue;
      seen.add(trackId);
      out.push(item.entry);
      if (out.length >= JOINT_CANDIDATES_PER_STEM) break;
    }
    // Reserve a few shortlist slots for never-played material (least-played,
    // then nearest tempo/key) so far stretches are always evaluated by the
    // joint ranker instead of being scored out of the shortlist entirely.
    // This is what guarantees the full bank traverses over a long session.
    const reserve = [...pool]
      .filter((item) => !out.includes(item.entry))
      .sort((a, b) => {
        const pa = this.playCountOf(stem, a.entry.phrase.id);
        const pb = this.playCountOf(stem, b.entry.phrase.id);
        if (pa !== pb) return pa - pb;
        return a.entry.tempoCost + a.entry.keyCost * 0.05
          - (b.entry.tempoCost + b.entry.keyCost * 0.05);
      });
    let added = 0;
    for (const item of reserve) {
      if (added >= JOINT_RESERVE) break;
      const trackId = item.entry.kit.track_id;
      if (seen.has(trackId)) continue;
      if (this.playCountOf(stem, item.entry.phrase.id) !== 0) continue;
      seen.add(trackId);
      out.push(item.entry);
      added += 1;
    }
    // One guided slot for a rested signature return: without it the top-8
    // shortlist (elite-dense by design) would never contain the motif the
    // brief asks for, and the energy-gated bonus would have nothing to act
    // on.  One slot, only when the signature is legal + rested, so the
    // never-played reserve and the ranker keep their room.
    if (FLOW_SIGNATURES_ENABLED && !out.some((e) => this.isSignature(stem, e.phrase.id))) {
      const signature = pool.find((item) => {
        if (!this.isSignature(stem, item.entry.phrase.id)) return false;
        if (this.pins[stem] !== undefined || this.manualPins[stem]) return false;
        if (this.signatureBonusFor(stem, item.entry, targetEnergy) <= 0) return false;
        if (seen.has(item.entry.kit.track_id) && !out.some((e) => e.kit.track_id === item.entry.kit.track_id)) {
          return false;
        }
        return true;
      });
      if (signature) {
        if (out.length >= JOINT_CANDIDATES_PER_STEM) out.pop(); // keep one free
        if (!out.includes(signature.entry)) out.push(signature.entry);
      }
    }
    return out;
  }

  /**
   * Choose the best joint transition over 1- and 2-stem replacements.
   * Returns null when the best candidate is below the calibrated threshold.
   */
  private selectJoint(
    section: string,
    targetEnergy: number,
    phrases: number,
    startBar: number,
    sectionChanged: boolean,
    forceMove: boolean,
  ): {
    entries: Partial<Record<StemName, StemPoolEntry>>;
    changed: StemName[];
  } | null {
    const ranker = this.jointRanker;
    if (!ranker?.ready) return null;
    const nextBlockBars = phrases * 4;
    const threshold = ranker.thresholdFor(nextBlockBars);
    const budget = movementBudgetAt(startBar, nextBlockBars);
    const allowVocals = vocalChangeAllowed(startBar, this.lastVocalChangeBar, sectionChanged);
    // A drum track can stay for a long time while bass/pads carry the mix.
    // After 24 bars, prefer a drum-inclusive candidate when it is nearly as
    // strong as the best policy move; this breaks "same groove" plateaus.
    const drumStale = startBar - this.lastDrumChangeBar >= 24;
    // 4-bar blocks are the busiest part of the arc; each stem holds for at
    // least two blocks between track changes unless the arc changes section
    // or the max-hold budget forces a move.  The cooldown is per stem, not
    // global: one stem moving no longer blocks the other three for 8 bars,
    // which is what flattened the mix into one long hold.  8/16-bar blocks
    // are already 8+ bars, so they have no cooldown.
    const cooldownBars = nextBlockBars <= 4 ? 8 : 0;
    const cooledDown = (stem: StemName): boolean => {
      if (cooldownBars === 0 || sectionChanged || forceMove) return true;
      if (startBar - (this.lastChangeBar[stem] ?? 0) >= budget.maxHoldBars) return true;
      return startBar - (this.lastChangeBar[stem] ?? 0) >= cooldownBars;
    };
    const current: Partial<Record<StemName, StemPoolEntry>> = {};
    for (const stem of STEMS) {
      const manual = this.manualPins[stem];
      // Unpinned stems are scored at the block boundary (the last phrase of
      // the run just scheduled).  A manual pin is a long run, so use the
      // phrase that is actually playing at the boundary.
      current[stem] = manual
        ? (this.current[stem]?.entry ?? manual.entry)
        : (this.lastBlockEntries[stem] ?? this.current[stem]?.entry);
      if (!current[stem]) {
        this.onStatus?.(`Joint mix: no current ${stem}; holding`);
        return null;
      }
    }
    const previous: Partial<Record<StemName, StemPoolEntry>> = {};
    for (const stem of STEMS) previous[stem] = this.previousEntries[stem] ?? current[stem];
    const currentState = this.jointState(current);
    const previousState = this.jointState(previous);
    if (!currentState || !previousState) {
      this.onStatus?.("Joint mix: missing ranker state; holding");
      return null;
    }

    const unpinned = STEMS.filter((stem) =>
      this.pins[stem] === undefined && (stem !== "vocals" || allowVocals) && cooledDown(stem)
    );
    if (unpinned.length === 0) return null;
    const usedTracks = new Set<string>();
    for (const stem of STEMS) {
      const entry = current[stem];
      if (entry) usedTracks.add(entry.kit.track_id);
    }
    const options = new Map<StemName, StemPoolEntry[]>();
    for (const stem of unpinned) {
      options.set(
        stem,
        this.jointOptions(stem, current[stem]!, usedTracks, phrases, section, targetEnergy),
      );
    }
    const subsets: StemName[][] = [];
    for (const stem of unpinned) subsets.push([stem]);
    for (let i = 0; i < unpinned.length; i++) {
      for (let j = i + 1; j < unpinned.length; j++) {
        subsets.push([unpinned[i], unpinned[j]]);
      }
    }
    const heldBase: Partial<Record<StemName, StemPoolEntry>> = {};
    for (const stem of STEMS) {
      heldBase[stem] = this.manualPins[stem]
        ? current[stem]
        : (this.holdEntry(stem) ?? current[stem]);
    }
    const bestRef: {
      value: {
        entries: Partial<Record<StemName, StemPoolEntry>>;
        changed: StemName[];
        score: number;
        rawScore: number;
        reasons: string[];
      } | null;
    } = { value: null };
    const bestRawRef: {
      value: {
        entries: Partial<Record<StemName, StemPoolEntry>>;
        changed: StemName[];
        score: number;
        rawScore: number;
      } | null;
    } = { value: null };
    const bestDrumRef: {
      value: {
        entries: Partial<Record<StemName, StemPoolEntry>>;
        changed: StemName[];
        score: number;
        rawScore: number;
      } | null;
    } = { value: null };

    const scoredCandidates: Array<{
      entries: Partial<Record<StemName, StemPoolEntry>>;
      changed: StemName[];
      score: number;
      rawScore: number;
      reasons: string[];
    }> = [];

    const consider = (subset: StemName[], picks: StemPoolEntry[]) => {
      const changed = new Set(subset);
      const candidate: Partial<Record<StemName, StemPoolEntry>> = { ...heldBase };
      for (let i = 0; i < subset.length; i++) candidate[subset[i]] = picks[i];
      const candidateState = this.jointState(candidate);
      if (!candidateState) return;
      const rawScore = ranker.score(
        previousState, currentState, candidateState, changed, section, targetEnergy,
        this.currentBlockBars, nextBlockBars,
      );
      let candidateEnergy = 0;
      let harmony = 0;
      let repetition = 0;
      let novelty = 0;
      let signatureTerm = 0;
      let timbreTerm = 0;
      // Coherence (stem-matched cosine, joint space) and syntax (learned
      // phrase grammar) accumulate over the changed stems only; both are
      // undefined (exactly neutral) when their artifact/sidecar is absent.
      let coherenceSum = 0;
      let coherenceN = 0;
      let syntaxSum = 0;
      let syntaxN = 0;
      const stemDim = ranker.stemDim;
      for (const stem of STEMS) {
        const entry = candidate[stem];
        if (!entry) return;
        candidateEnergy += entry.energy;
        if (changed.has(stem)) {
          const stemIdx = STEMS.indexOf(stem);
          const stemOff = stemIdx * stemDim;
          coherenceSum += StemRemixPlayer.stemCosine(candidateState, currentState, stemOff);
          coherenceN++;
          const fromId = current[stem]?.phrase.id ?? "";
          const toId = entry.phrase.id;
          const table = this.syntaxSuccessors?.get(fromId);
          const syntaxHit = table?.get(toId);
          if (typeof syntaxHit === "number" && Number.isFinite(syntaxHit)) {
            syntaxSum += syntaxHit;
            syntaxN++;
          }
          const from = current[stem];
          if (from) {
            harmony += harmonyPenalty({
              stem,
              fromRoot: from.bassRoot,
              toRoot: entry.bassRoot,
              fromChroma: from.chroma,
              toChroma: entry.chroma,
              keyShift: entry.keyShift,
            });
          }
          const plays = this.playCountOf(stem, entry.phrase.id);
          // Signature returns cost discounted repetition so the motif can come
          // back without the novelty loop spinning it every block.  The 16-bar
          // rest (signatureBonus) is what keeps the discount honest.
          const isSig = this.isSignature(stem, entry.phrase.id);
          const discountedPlays = isSig ? plays * (1 - FLOW_SIGNATURE_REPETITION_DISCOUNT) : plays;
          repetition += plays === 0 ? 0 : Math.min(1, 0.3 + 0.2 * discountedPlays);
          novelty += plays === 0 ? 1 : 1 / (1 + discountedPlays);
          signatureTerm += this.signatureBonusFor(stem, entry, targetEnergy);
          timbreTerm += this.timbreBonusFor(stem, entry, startBar);
        }
      }
      candidateEnergy /= 4;
      harmony /= Math.max(1, subset.length);
      repetition = Math.min(1, repetition / Math.max(1, subset.length));
      novelty = subset.length > 0 ? novelty / subset.length : 0;
      // Mean soft terms so a 2-stem candidate is not worth twice a 1-stem one.
      const meanSignature = subset.length > 0 ? signatureTerm / subset.length : 0;
      const meanTimbre = subset.length > 0 ? timbreTerm / subset.length : 0;
      const meanCoherence = coherenceN > 0 ? coherenceSum / coherenceN : undefined;
      const meanSyntax = syntaxN > 0 ? clamp01(syntaxSum / syntaxN) : undefined;
      // Rhythm-mesh (candidate mix interlock), chord progression (grammar
      // fit after the recent harmonic context) and vocal-edge (clean swap
      // boundary) terms: all undefined (exactly neutral) when their sidecar
      // is missing, so a partial deploy never changes the ranking.
      const rhythmFit = this.rhythmFitFor(candidate);
      const progression = this.progressionFor(candidate);
      const vocalEdge = subset.includes("vocals") ? this.vocalEdgeFor(current, candidate) : undefined;
      const vocalSegmentBars = subset.includes("vocals") ? this.vocalSegmentBarsFor(candidate) : undefined;
      let earSum = 0;
      for (let i = 0; i < subset.length; i++) {
        earSum += candidate[subset[i]]!.earQ;
      }
      const earQuality = subset.length > 0 ? earSum / subset.length : 0.5;
      const policy = scoreTransition({
        jointScore: rawScore,
        targetEnergy,
        candidateEnergy,
        novelty,
        repetition,
        harmony,
        changedCount: subset.length,
        budget,
        includesVocals: subset.includes("vocals"),
        vocalAllowed: allowVocals,
        earQuality,
        signatureBonus: meanSignature,
        timbreBonus: meanTimbre,
        coherence: meanCoherence,
        syntax: meanSyntax,
        rhythmFit,
        progression,
        vocalEdge,
        vocalSegmentBars,
      });
      if (!bestRawRef.value || rawScore > bestRawRef.value.rawScore) {
        bestRawRef.value = { entries: candidate, changed: subset, score: policy.score, rawScore };
      }
      if (subset.includes("drums")
        && (!bestDrumRef.value || rawScore > bestDrumRef.value.rawScore)) {
        bestDrumRef.value = { entries: candidate, changed: subset, score: policy.score, rawScore };
      }
      if (!bestRef.value || policy.score > bestRef.value.score) {
        bestRef.value = {
          entries: candidate,
          changed: subset,
          score: policy.score,
          rawScore,
          reasons: policy.reasons,
        };
      }
      scoredCandidates.push({
        entries: candidate,
        changed: subset,
        score: policy.score,
        rawScore,
        reasons: policy.reasons,
      });
    };

    for (const subset of subsets) {
      if (subset.length > budget.maxChanges) continue;
      const lists = subset.map((stem) => options.get(stem) ?? []);
      if (lists.some((list) => list.length === 0)) continue;
      if (subset.length === 1) {
        for (const entry of lists[0]) consider(subset, [entry]);
      } else {
        // Pair scoring is the expensive part (a 2k-d MLP per combo).  Only
        // combine the top-ranked entries from each stem; jointOptions has
        // already rotated fresh material into that top set via play counts.
        const firsts = lists[0].slice(0, PAIR_TOP_N);
        const seconds = lists[1].slice(0, PAIR_TOP_N);
        for (const first of firsts) {
          for (const second of seconds) {
            if (first.kit.track_id === second.kit.track_id) continue;
            if (usedTracks.has(first.kit.track_id) || usedTracks.has(second.kit.track_id)) continue;
            consider(subset, [first, second]);
          }
        }
      }
    }

    const best = bestRef.value;
    if (!best) {
      const counts = unpinned.map((stem) => `${stem}:${options.get(stem)?.length ?? 0}`).join(" ");
      this.onStatus?.(`Joint mix: no candidates (${counts}); holding`);
      return null;
    }
    const bestRaw = bestRawRef.value;
    const bestDrum = bestDrumRef.value;
    if (drumStale && bestDrum && !best.changed.includes("drums")
      && bestDrum.rawScore >= threshold * 0.8
      && bestDrum.rawScore >= best.rawScore * 0.9) {
      this.onStatus?.(
        `Joint mix: drum refresh ${bestDrum.changed.join("+")} → ` +
        `${bestDrum.score.toFixed(3)} (raw ${bestDrum.rawScore.toFixed(3)})`,
      );
      return { entries: bestDrum.entries, changed: bestDrum.changed };
    }
    if (best.rawScore < threshold) {
      // The policy may rank a candidate below another raw candidate; fall
      // back to the best raw candidate if it still clears the learned gate.
      if (bestRaw && bestRaw.rawScore >= threshold) {
        this.onStatus?.(
          `Joint mix: policy pick ${best.score.toFixed(3)} below gate; ` +
          `using best raw ${bestRaw.changed.join("+")} → ${bestRaw.rawScore.toFixed(3)}`,
        );
        return { entries: bestRaw.entries, changed: bestRaw.changed };
      }
      if (forceMove && bestRaw && bestRaw.rawScore >= threshold * 0.5) {
        this.onStatus?.(
          `Joint mix: forced ${bestRaw.changed.join("+")} after ` +
          `${startBar} bars (raw ${bestRaw.rawScore.toFixed(3)})`,
        );
        return { entries: bestRaw.entries, changed: bestRaw.changed };
      }
      this.onStatus?.(
        `Joint mix: best raw ${best.rawScore.toFixed(3)} < ${threshold.toFixed(3)}; holding`,
      );
      return null;
    }
    // Near-best exploration: the ranker scores are clustered, so always
    // taking the argmax replays the same winning tracks.  When several
    // candidates sit within EXPLORE_WINDOW of the best policy score (and
    // still clear the learned gate), choose among them with a block-seeded
    // hash so the mix rotates through equally-good material.
    const gated = scoredCandidates
      .filter((item) => item.rawScore >= threshold)
      .sort((a, b) => b.score - a.score);
    if (gated.length > 1 && gated[0]) {
      const topScore = gated[0].score;
      const nearBest = gated
        .filter((item) => topScore - item.score <= EXPLORE_WINDOW)
        .slice(0, EXPLORE_TOP_K);
      if (nearBest.length > 1) {
        const ids = nearBest
          .map((item) => item.changed.map((stem) => item.entries[stem]?.phrase.id ?? stem).join("+"))
          .join("|");
        const pickIndex = Math.min(
          nearBest.length - 1,
          Math.floor(hashRoll(`${this.blockIndex}:${ids}`) * nearBest.length),
        );
        const picked = nearBest[pickIndex];
        if (picked && picked !== best) {
          const detail = picked.reasons.length > 0 ? ` [${picked.reasons.join(",")}]` : "";
          this.onStatus?.(
            `Joint mix: explore ${picked.changed.join("+")} → ${picked.score.toFixed(3)} ` +
            `(raw ${picked.rawScore.toFixed(3)})${detail}`,
          );
          return { entries: picked.entries, changed: picked.changed };
        }
      }
    }
    const detail = best.reasons.length > 0 ? ` [${best.reasons.join(",")}]` : "";
    this.onStatus?.(
      `Joint mix: ${best.changed.join("+")} → ${best.score.toFixed(3)} ` +
      `(raw ${best.rawScore.toFixed(3)})${detail}`,
    );
    return { entries: best.entries, changed: best.changed };
  }

  /** Continue the stem's run where it left off, or loop when it has ended. */
  private holdEntry(stem: StemName): StemPoolEntry | null {
    const run = this.lastBlockRuns[stem] ?? this.current[stem]?.run;
    if (run && run.length > 0) {
      const next = this.nextRunEntry(run[run.length - 1], stem);
      return next ?? run[0];
    }
    return this.current[stem]?.entry ?? this.lastBlockEntries[stem] ?? null;
  }

  /** The real +4-bar successor in the same kit, if it is legal for this run. */
  private nextRunEntry(entry: StemPoolEntry, stem: StemName): StemPoolEntry | null {
    const start = entry.phrase.start_bar;
    if (typeof start !== "number") return null;
    const nextPhrase = entry.kit.phrases.find((candidate) =>
      candidate.start_bar === start + 4
      && this.entryById.has(`${candidate.id}:${stem}`)
      && !(candidate.once && this.playedOnce.has(candidate.id)),
    );
    if (!nextPhrase) return null;
    const before = Number(entry.phrase.duration_sec) || 0;
    const after = Number(nextPhrase.duration_sec) || 0;
    if (before > 0 && after > 0 && Math.abs(after - before) / before > 0.01) {
      return null;
    }
    return this.entryById.get(`${nextPhrase.id}:${stem}`) ?? null;
  }

  /**
   * Pick a fresh phrase/run start inside a pinned style region.  The pin is a
   * region, not a frozen loop: candidates near the dropped point are scored by
   * distance, arc energy/section, the per-stem judge, recency and style.
   */
  private selectPinnedEntry(
    stem: StemName,
    pin: StemPin,
    phrases: number,
  ): StemPoolEntry | null {
    const pool = this.pools[stem];
    if (pool.length === 0) return null;
    const current = this.current[stem]?.entry ?? this.lastBlockEntries[stem] ?? null;
    const arc = targetAt(this.sessionBar);
    const targetSection = this.lastSection || arc.section;
    let targetEnergy = arc.energy;
    // Learned-arc energy blend (gated; section comes from the planned arc).
    {
      const learned = this.arcPredict();
      if (learned && Math.abs(learned.energy - targetEnergy) <= 0.25) {
        targetEnergy = clamp01(0.5 * targetEnergy + 0.5 * learned.energy);
      }
    }
    const outroAllowed = this.blockIndex >= BLOCKS_PER_SET - 2;
    const recent = new Set(this.stemHistory[stem].slice(-STEM_HISTORY_AVOID));

    const eligible = pool.filter((entry) => {
      if (this.failedKits.has(entry.kit.track_id)) return false;
      if (entry.phrase.once && this.playedOnce.has(entry.phrase.id)) return false;
      if (entry.phrase.is_intro && this.blockIndex > 0) return false;
      if (entry.phrase.is_outro && !outroAllowed) return false;
      return true;
    });
    if (eligible.length === 0) return null;

    // Consecutive phrases are preferable, but a pin must still move when the
    // cluster only has isolated windows near the region.
    const runLength = (entry: StemPoolEntry): number => {
      const run = this.runFor(entry, phrases, false);
      let length = 0;
      for (const item of run) {
        if (item.phrase.once && this.playedOnce.has(item.phrase.id)) break;
        length += 1;
      }
      return length;
    };

    const near: Array<{ entry: StemPoolEntry; distance: number }> = [];
    let nearest: { entry: StemPoolEntry; distance: number } | null = null;
    for (const entry of eligible) {
      const distance = Math.hypot(entry.x - pin.x, entry.y - pin.y);
      if (!nearest || distance < nearest.distance) nearest = { entry, distance };
      if (distance <= PIN_RADIUS) near.push({ entry, distance });
    }
    if (near.length === 0) {
      // Nothing survives the legal filters inside the region: fall back to the
      // nearest legal phrase, which is normally the exact phrase dropped on.
      if (nearest) {
        this.onStatus?.(`${stem}: pin region empty; using the nearest legal phrase`);
        return nearest.entry;
      }
      return null;
    }

    const fullNear = near.filter((item) => runLength(item.entry) >= phrases);
    const candidates = fullNear.length > 0 ? fullNear : near;

    const lastRun = new Set<string>();
    for (const entry of this.lastBlockRuns[stem] ?? []) lastRun.add(entry.phrase.id);

    const usedTracks = new Set<string>();
    for (const other of STEMS) {
      if (other === stem) continue;
      const otherEntry = this.current[other]?.entry ?? this.lastBlockEntries[other];
      if (otherEntry) usedTracks.add(otherEntry.kit.track_id);
    }

    let best: { entry: StemPoolEntry; score: number } | null = null;
    for (const { entry, distance } of candidates) {
      const proximity = Math.exp(-(distance * distance) / (2 * PIN_SIGMA * PIN_SIGMA));
      const energyFit = 1 - Math.min(1, Math.abs(entry.energy - targetEnergy));
      const sectionBonus = entry.section === targetSection ? 0.35 : 0;
      const sectionEnergy = SECTION_ENERGY[entry.section];
      const sectionFit = sectionEnergy === undefined
        ? 0
        : 0.12 * (1 - Math.abs(entry.energy - sectionEnergy));
      const judge = this.judgeScore(stem, current?.phrase.id ?? null, entry.phrase.id);
      const judgeBonus = judge > 0 ? 0.8 * judge : 0;
      const recentPenalty = recent.has(entry.phrase.id) ? 0.8 : 0;
      const lastRunPenalty = lastRun.has(entry.phrase.id) ? 1.4 : 0;
      const samePenalty = current
        && current.phrase.id === entry.phrase.id
        && current.kit.track_id === entry.kit.track_id ? 0.5 : 0;
      const styleBonus = entry.kit.style === this.cluster?.style ? 0.22 : 0;
      const trackPenalty = usedTracks.has(entry.kit.track_id) ? PIN_TRACK_PENALTY : 0;
      const plays = this.playCountOf(stem, entry.phrase.id);
      const novelty = hashRoll(`${entry.phrase.id}:${this.blockIndex}:${stem}`) * 0.15
        + (plays === 0 ? 0.3 : Math.max(-0.3, -0.12 * plays));
      const tierCost = entry.tempoCost * TEMPO_COST_WEIGHT * 0.5
        + entry.keyCost * KEY_COST_WEIGHT * 0.5;
      const earBonus = EAR_W_FALLBACK * (entry.earQ - 0.5);
      const score =
        2.4 * proximity +
        0.9 * energyFit +
        sectionBonus +
        sectionFit +
        judgeBonus +
        styleBonus +
        novelty -
        recentPenalty -
        lastRunPenalty -
        samePenalty -
        trackPenalty -
        tierCost +
        earBonus;
      if (!best || score > best.score) best = { entry, score };
    }
    return best?.entry ?? nearest?.entry ?? null;
  }

  /** Pick a cross-song phrase for one stem, excluding its current track. */
  private selectMorph(stem: StemName, at: number, section: string, phrases: number = 1): StemPoolEntry | null {
    const pool = this.pools[stem];
    if (pool.length === 0) return null;
    const current = this.current[stem]?.entry ?? null;
    if (!current) return null;
    const currentKit = current.kit.track_id;
    const target = SECTION_ENERGY[section] ?? 0.65;
    const chord = this.current.bass?.entry.chroma ?? null;
    const anchorRoot = this.current.bass?.entry.bassRoot ?? null;
    const recentSet = new Set(this.stemHistory[stem].slice(-STEM_HISTORY_AVOID));
    const usedTracks = new Set<string>();
    for (const other of STEMS) {
      if (other === stem) continue;
      const tid = this.current[other]?.entry.kit.track_id;
      if (tid) usedTracks.add(tid);
    }
    let best: { entry: StemPoolEntry; score: number } | null = null;

    // Prefer a track no other stem is using, so after four morphs all four
    // instruments come from different records; fall back if none exists.
    // Prefer candidates with a real consecutive run long enough for the
    // block; a short run would loop the same bars inside an 8/16-bar block.
    for (const strict of [true, false]) {
    for (const entry of pool) {
      if (entry.kit.track_id === currentKit) continue;   // force a new track
      if (strict && usedTracks.has(entry.kit.track_id)) continue;
      if (this.failedKits.has(entry.kit.track_id)) continue;
      if (entry.phrase.id === current.phrase.id) continue;
      if (recentSet.has(entry.phrase.id)) continue;
      if (entry.phrase.once && this.playedOnce.has(entry.phrase.id)) continue;
      if (entry.phrase.is_intro && this.blockIndex > 0) continue;
      if (entry.phrase.is_outro && this.blockIndex < BLOCKS_PER_SET - 2) continue;
      const runLen = phrases > 1 ? this.runFor(entry, phrases, false).length : phrases;
      const shortRunPenalty = runLen < phrases ? 2.0 : 0;

      const energy = Number(entry.energy) || 0.5;
      const sectionScore = (entry.section === section ? 0.6 : 0)
        - Math.abs(energy - target);
      const jump = Math.abs(energy - current.energy);
      const jumpPenalty = jump > 0.25 ? (jump - 0.25) * 0.8 : 0;

      let harmonic = 0;
      if (stem === "bass") {
        harmonic = -rootPenalty(entry.bassRoot ?? null, anchorRoot);
        if (chord && entry.chroma) harmonic += 0.4 * cosine(entry.chroma, chord);
      } else if (stem !== "drums" && chord && entry.chroma) {
        harmonic = 0.7 * cosine(entry.chroma, chord);
      }

      const judge = this.judgeScore(stem, current.phrase.id, entry.phrase.id);
      const judgeBonus = judge > 0 ? 1.5 + 2.5 * judge : -0.2;
      const style = entry.kit.style === this.cluster?.style ? 0.25 : 0;
      const presence = clamp01((entry.rmsDb - MIN_RMS_DB[stem] + 8) / 16) *
        (stem === "vocals" ? 0.25 : 0.6);
      const plays = this.playCountOf(stem, entry.phrase.id);
      const isSigMorph = this.isSignature(stem, entry.phrase.id);
      const morphPlays = isSigMorph ? plays * (1 - FLOW_SIGNATURE_REPETITION_DISCOUNT) : plays;
      const novelty = hashRoll(`${entry.phrase.id}:${this.blockIndex}:${stem}`) * 0.35
        + (morphPlays === 0 ? 0.4 : Math.max(-0.4, -0.15 * morphPlays));
      const tierCost = entry.tempoCost * TEMPO_COST_WEIGHT
        + entry.keyCost * KEY_COST_WEIGHT;
      const earBonus = EAR_W_FALLBACK * (entry.earQ - 0.5);
      // Fallback path honours the same soft policy as the joint path: a rested
      // signature earns its energy-gated bonus and an under-used timbre earns
      // its window bonus.  sectionEnergy is the fallback's target proxy since
      // selectMorph has no arc targetEnergy parameter.
      const sigBonus = this.signatureBonusFor(stem, entry, target);
      const timbre = this.timbreBonusFor(stem, entry, this.sessionBar);
      // Vocal segments: entering mid-segment means splicing into a vocal
      // that is already underway (a fragment); starting a long segment
      // means the whole performance can play.  Same policy as the joint
      // vocalEdge/vocalSegmentBars terms, as hard-ish heuristics here.
      let vocalSegmentPenalty = 0;
      let vocalSegmentBonus = 0;
      if (stem === "vocals") {
        const seg = this.vocalSegmentOf(entry.phrase.id);
        if (seg !== null) {
          if (seg.pos > 0) vocalSegmentPenalty = 1.0;
          else vocalSegmentBonus = 0.5 * Math.min(seg.lenBars, 12) / 12;
        }
      }
      const score =
        1.0 * sectionScore +
        0.9 * harmonic +
        0.25 * style +
        presence +
        novelty +
        judgeBonus -
        jumpPenalty -
        shortRunPenalty -
        tierCost +
        earBonus +
        sigBonus +
        timbre -
        vocalSegmentPenalty +
        vocalSegmentBonus;
      if (!best || score > best.score) best = { entry, score };
    }
    if (best) break;
    }
    if (!best) {
      this.onStatus?.(`${stem}: no compatible cross-song phrase; keeping current`);
    }
    return best?.entry ?? null;
  }

  /** Pick the next phrase from the anchor song, following the section arc. */
  private chooseAnchorPhrase(section: string): KitPhrase | null {
    const kit = this.anchorKit;
    if (!kit || kit.phrases.length === 0) return null;
    const target = SECTION_ENERGY[section] ?? 0.65;
    // Only avoid the last two phrases, so a six-phrase kit can keep cycling.
    const recent = new Set(this.anchorRecent.slice(-2));
    let best: { phrase: KitPhrase; score: number } | null = null;
    for (const phrase of kit.phrases) {
      if (recent.has(phrase.id) && kit.phrases.length > 2) continue;
      if (phrase.once && this.playedOnce.has(phrase.id)) continue;
      if (phrase.is_intro && this.blockIndex > 1) continue;
      if (phrase.is_outro && this.blockIndex < BLOCKS_PER_SET - 2) continue;
      const energy = Number(phrase.energy) || 0.5;
      const fullStem = STEMS.every((stem) => this.entryById.has(`${phrase.id}:${stem}`));
      const continuity = this.currentAnchorPhrase
        ? Math.abs(energy - (Number(this.currentAnchorPhrase.energy) || 0.5))
        : 0;
      let score = (String(phrase.section).toUpperCase() === section ? 0.9 : 0)
        - Math.abs(energy - target)
        - continuity * 0.6
        + (fullStem ? 0.5 : 0);
      if (continuity > 0.3 && String(phrase.section).toUpperCase() !== section) score -= 0.4;
      const currentStart = this.currentAnchorPhrase?.start_bar;
      if (typeof currentStart === "number" && typeof phrase.start_bar === "number") {
        const delta = phrase.start_bar - currentStart;
        if (delta === 4 && fullStem) score += 2.8;   // the real next 4 bars wins
        else if (delta === 4) score += 1.2;          // forward, but a stem is absent
        else if (delta === 8) score += 1.0;          // one window skipped
        else if (delta > 0 && delta < 16) score += 0.3;
        else if (delta >= 16) score -= 0.5;
        else if (delta < 0) score -= 0.2;    // prefer forward motion
      }
      if (phrase.once) score -= 0.2;
      score += hashRoll(`${phrase.id}:${this.blockIndex}`) * 0.12;
      if (!best || score > best.score) best = { phrase, score };
    }
    if (best) return best.phrase;
    let fallback = kit.phrases[0];
    let bestDelta = Number.POSITIVE_INFINITY;
    for (const phrase of kit.phrases) {
      const delta = Math.abs((Number(phrase.energy) || 0.5) - target);
      if (delta < bestDelta) {
        bestDelta = delta;
        fallback = phrase;
      }
    }
    return fallback;
  }

  /** The anchor phrase's entry for a stem, or the best same-song fallback. */
  private anchorEntry(stem: StemName, phrase: KitPhrase | null): StemPoolEntry | null {
    const kit = this.anchorKit;
    if (!kit) return null;
    if (phrase) {
      const exact = this.entryById.get(`${phrase.id}:${stem}`);
      if (exact) return exact;
    }
    const candidates = this.pools[stem].filter((entry) => entry.kit.track_id === kit.track_id);
    if (candidates.length === 0) return null;
    const targetSection = String(phrase?.section ?? "GROOVE_A").toUpperCase();
    const target = SECTION_ENERGY[targetSection] ?? 0.65;
    candidates.sort((a, b) => Math.abs(a.energy - target) - Math.abs(b.energy - target));
    return candidates[0];
  }

  private async finishBlock(
    tasks: Array<{ stem: StemName; run: StemPoolEntry[]; transition?: TransitionTechnique }>,
    at: number,
    resumes?: Map<StemName, ResumeHint>,
  ): Promise<void> {
    const callMs = performance.now();
    let maxLoadMs = 0;
    // Loads stay concurrent (fetch + decode are off-thread), but the
    // synchronous tail of each stem (mixToMono + concat + worklet clone,
    // 100-300 ms for 8/16-bar runs) is scheduled one stem at a time with a
    // main-thread yield between.  Four stems fired back-to-back in one
    // macrotask measured 600-1100 ms of blocking (v0.2.233 dropout).
    const audios = await Promise.all(tasks.map(async (task) => {
      const loadT0 = performance.now();
      const audio = await this.loadRunAudio(task.run);
      maxLoadMs = Math.max(maxLoadMs, performance.now() - loadT0);
      return { task, audio };
    }));
    for (const { task, audio } of audios) {
      const resume = resumes?.get(task.stem);
      if (!audio) {
        const current = this.current[task.stem];
        const fallbackRun = current?.run ?? (current ? [current.entry] : null);
        if (fallbackRun) {
          const fallback = await this.loadRunAudio(fallbackRun);
          if (fallback) {
            await this.scheduleRun(task.stem, fallbackRun, fallback, at, this.currentBlockBars, false, resume, task.transition);
          }
        }
      } else {
        await this.scheduleRun(task.stem, task.run, audio, at, this.currentBlockBars, false, resume, task.transition);
      }
      await yieldToMain();
    }
    // Scheduling diagnostics: a slow fetch/decode that lands near the bar
    // start explains 1 s dropout reports (schedule() clamps late starts).
    const latenessMs = (this.engine.now() - at) * 1000;
    if (maxLoadMs > 600 || latenessMs > -1200) {
      console.info(
        `[StemRemix] block load max=${maxLoadMs.toFixed(0)}ms ` +
        `call=${(performance.now() - callMs).toFixed(0)}ms lateness=${latenessMs.toFixed(0)}ms @ ${at.toFixed(2)}s`,
      );
    }
  }

  private select(stem: StemName, at: number, section: string, sectionChanged: boolean): StemPoolEntry | null {
    const pool = this.pools[stem];
    if (pool.length === 0) return null;
    const current = this.current[stem]?.entry ?? null;
    const pin = this.pins[stem] ?? null;
    const anchor = this.current.bass?.entry ?? null;
    const chord = anchor?.chroma ?? null;
    const recentSet = new Set(this.stemHistory[stem].slice(-STEM_HISTORY_AVOID));
    let best: { entry: StemPoolEntry; score: number } | null = null;

    for (const entry of pool) {
      if (entry.kit.track_id && this.failedKits.has(entry.kit.track_id)) continue;
      if (current && entry.phrase.id === current.phrase.id && entry.stem === stem) continue;
      if (recentSet.has(entry.phrase.id)) continue;
      if (entry.phrase.once && this.playedOnce.has(entry.phrase.id)) continue;
      if (entry.phrase.is_intro && this.blockIndex > 0) continue;
      if (entry.phrase.is_outro && this.blockIndex < BLOCKS_PER_SET - 2) continue;

      // Direction-aware section / energy rules.
      const target = SECTION_ENERGY[section] ?? 0.65;
      const energyDelta = Math.abs(entry.energy - target);
      let sectionScore = 0.5 - energyDelta;
      if (entry.section === section) sectionScore += 0.6;
      if (section === "BUILDUP" && current && entry.energy < current.energy - 0.05) sectionScore -= 0.8;
      if (section === "BREAKDOWN" && current && entry.energy > current.energy + 0.1) sectionScore -= 0.8;
      if (section === "DROP" && entry.energy < 0.6) sectionScore -= 0.5;

      // Harmonic fit: bass root motion + chroma overlap with the current bass.
      let harmonic = 0;
      if (stem === "bass" && anchor) {
        harmonic = -rootPenalty(entry.bassRoot ?? null, anchor.bassRoot ?? null);
        if (chord && entry.chroma) harmonic += 0.4 * cosine(entry.chroma, chord);
      } else if (stem !== "drums" && chord && entry.chroma) {
        harmonic = 0.7 * cosine(entry.chroma, chord);
        if (stem === "vocals") harmonic += 0.2;
      }

      // Pin: the style region is a soft target; auto-advance still picks
      // different phrases inside it.
      let pinScore = 0;
      let pinDist = 0;
      if (pin) {
        pinDist = Math.hypot(entry.x - pin.x, entry.y - pin.y);
        pinScore = Math.exp(-(pinDist * pinDist) / (2 * 0.18 * 0.18));
        if (pinDist > 0.42) pinScore *= 0.15;
      }

      const sameKit = current && entry.kit.track_id === current.kit.track_id ? 0.3 : 0;
      const style = entry.kit.style === this.cluster?.style ? 0.25 : 0;
      const novelty = hashRoll(`${entry.phrase.id}:${this.blockIndex}:${stem}`) * 0.25;
      const judge = this.judgeScore(stem, current?.phrase.id ?? null, entry.phrase.id);
      const judgeBonus = judge > 0 ? 1.5 + 2.5 * judge : -0.2;
      const presence = clamp01((entry.rmsDb - MIN_RMS_DB[stem] + 8) / 16) *
        (stem === "vocals" ? 0.25 : 0.6);
      const score =
        1.0 * sectionScore +
        0.9 * harmonic +
        (pin ? 2.2 * pinScore : 0) +
        0.55 * sameKit +
        0.25 * style +
        novelty +
        presence +
        judgeBonus;

      if (best === null || score > best.score) best = { entry, score };
    }

    if (best === null && pin) {
      // No legal phrase in the region: snap to the nearest legal phrase.
      let nearest: StemPoolEntry | null = null;
      let dist = Number.POSITIVE_INFINITY;
      for (const entry of pool) {
        const d = Math.hypot(entry.x - pin.x, entry.y - pin.y);
        if (d < dist) {
          dist = d;
          nearest = entry;
        }
      }
      if (nearest) {
        this.onStatus?.(`${stem}: no phrase at the target, snapped to the nearest legal phrase`);
        return nearest;
      }
    }
    return best?.entry ?? null;
  }

  private judgeScore(stem: StemName, fromId: string | null, toId: string): number {
    if (!fromId || this.stemNext.size === 0) return 0;
    return this.stemNext.get(`${fromId}:${stem}:${toId}`) ?? 0;
  }

  private async prepareEntry(
    stem: StemName,
    entry: StemPoolEntry,
  ): Promise<{ entry: StemPoolEntry; audio: LoadedStemRun } | null> {
    try {
      const audio = await this.loadRunAudio([entry]);
      return audio ? { entry, audio } : null;
    } catch (err) {
      console.warn(`[StemRemix] ${stem} load failed for ${entry.phrase.id}`, err);
      return null;
    }
  }

  private cacheKey(entry: StemPoolEntry): string {
    return `${entry.phrase.id}:${entry.stem}`;
  }

  private async loadAudio(entry: StemPoolEntry): Promise<LoadedStemAudio> {
    const key = this.cacheKey(entry);
    const cached = this.cache.get(key);
    if (cached) {
      this.cache.delete(key);
      this.cache.set(key, cached);
      return cached;
    }
    const asset = entry.phrase.stems?.[entry.stem];
    if (!asset) throw new Error("missing stem asset");
    const mainRes = await fetch(`${this.baseUrl}/${asset.file}`);
    if (!mainRes.ok) throw new Error(`stem ${mainRes.status} ${asset.file}`);
    const main = await this.engine.decodeAudio(await mainRes.arrayBuffer());
    let pre: AudioBuffer | null = null;
    if (asset.pre_roll?.file) {
      try {
        const preRes = await fetch(`${this.baseUrl}/${asset.pre_roll.file}`);
        if (preRes.ok) pre = await this.engine.decodeAudio(await preRes.arrayBuffer());
      } catch {
        // A missing tail must not block the phrase.
      }
    }
    const loaded: LoadedStemAudio = {
      main,
      pre,
      preDur: pre ? pre.duration : 0,
    };
    this.cache.set(key, loaded);
    if (this.cache.size > CACHE_LIMIT) {
      const oldest = this.cache.keys().next().value;
      if (oldest) this.cache.delete(oldest);
    }
    return loaded;
  }

  /** Decode every phrase in a run: worker first, main-thread fallback. */
  private async loadRunAudio(run: readonly StemPoolEntry[]): Promise<LoadedStemRun | null> {
    if (this.audioWorker.ok) {
      const items: WorkerRunItem[] = [];
      for (const entry of run) {
        const asset = entry.phrase.stems?.[entry.stem];
        if (!asset?.file) {
          // Same semantics as the old throw path: a missing asset fails the
          // run before any worker round-trip.
          this.failedKits.add(entry.kit.track_id);
          return null;
        }
        items.push({
          key: this.cacheKey(entry),
          url: new URL(`${this.baseUrl}/${asset.file}`, document.baseURI).toString(),
          preUrl: asset.pre_roll?.file
            ? new URL(`${this.baseUrl}/${asset.pre_roll.file}`, document.baseURI).toString()
            : null,
        });
      }
      const loaded = await this.audioWorker.load(items);
      if (loaded) return loaded;
      // The worker reported a failed run (or glitched silently): the
      // main-thread loader keeps the session alive from here.  It records
      // per-phrase failures (failedKits) the same way, so a genuinely
      // missing file still fails the run.
      return this.loadRunAudioMain(run);
    }
    return this.loadRunAudioMain(run);
  }

  /** Main-thread fallback: fetch + decode through the audio cache, then
   *  concat into one PCM run (the pre-worker path, kept for coverage). */
  private async loadRunAudioMain(run: readonly StemPoolEntry[]): Promise<LoadedStemRun | null> {
    const audios = await Promise.all(run.map(async (entry) => {
      try {
        return await this.loadAudio(entry);
      } catch (err) {
        this.failedKits.add(entry.kit.track_id);
        console.warn(`[StemRemix] run load failed for ${entry.phrase.id}`, err);
        return null;
      }
    }));
    if (audios.some((audio) => audio === null)) return null;
    const loaded = audios as LoadedStemAudio[];
    const sampleRate = loaded[0]?.main.sampleRate ?? ENGINE_SAMPLE_RATE;
    return concatRun(
      loaded.map((audio) => mixToMono(audio.main)),
      loaded[0]?.pre ? mixToMono(loaded[0].pre) : null,
      sampleRate,
    );
  }

  /** Two reusable voices per stem: one playing, one being prepared. */
  private async getVoice(stem: StemName, avoid?: StretchVoice): Promise<StretchVoice> {
    let pool = this.voicePool[stem];
    // The silence watchdog empties the pool ([]) when it rebuilds; an empty
    // array is truthy, so a plain `!pool` check never recreates it and the
    // index math below reads pool[NaN] = undefined, throwing on reset().
    // Treat an empty pool as missing so recovery actually rebuilds voices.
    if (!pool || pool.length === 0) {
      const ctx = this.engine.ensureCtx();
      const destination = this.engine.getStemInput(stem);
      const pan = stemPan(stem, stem);
      pool = await Promise.all([
        StretchVoice.create(ctx, destination, { pan }),
        StretchVoice.create(ctx, destination, { pan }),
      ]);
      this.voicePool[stem] = pool;
      this.voiceIndex[stem] = 0;
    }
    let index = (this.voiceIndex[stem] | 0) % pool.length;
    if (pool.length > 1 && pool[index] === avoid) {
      index = (index + 1) % pool.length;
    }
    this.voiceIndex[stem] = (index + 1) % pool.length;
    const voice = pool[index];
    await voice.reset();
    return voice;
  }

  private async scheduleRun(
    stem: StemName,
    run: readonly StemPoolEntry[],
    audio: LoadedStemRun,
    at: number,
    bars: number,
    manual: boolean,
    resume?: ResumeHint,
    transition?: TransitionTechnique,
  ): Promise<void> {
    const first = run[0];
    if (!first) return;
    try {
      const previous = this.current[stem];
      const voice = await this.getVoice(stem, previous?.voice);
      await voice.loadSamples(audio.samples);
      const preDur = audio.preDur;
      const rate = Math.max(RATE_MIN, Math.min(RATE_MAX, first.rate));
      // Normal scheduling: the pre-roll starts preDur/rate before the
      // downbeat.  A watchdog resume instead re-enters the run at the input
      // position the block clock has reached, so the block continues where it
      // stopped instead of restarting (or, as before, jumping to the next
      // block), and keeps the run's original end.
      let startAt = at - preDur / rate;
      let inputSec = 0;
      let runEndAt = at + bars * this.barSec;
      let startedAt = at;
      if (resume) {
        const elapsed = Math.max(0, at - resume.startedAt);
        if (resume.runEndAt - at < 0.08) return; // block already over
        const totalInputDur = audio.mainDurations.reduce((sum, d) => sum + d, 0);
        startAt = at;
        inputSec = Math.min(preDur + elapsed * rate, preDur + Math.max(0, totalInputDur - 0.05 / rate));
        runEndAt = resume.runEndAt;
        startedAt = resume.startedAt;
      }
      // (B) Incoming-vocal fade: a loud incoming vocal fades from silence over
      // ~150 ms instead of STEM_FADE_IN (80 ms) to avoid mid-vowel splats.
      // Only the *fade length* changes; gain stays the equal-gain hand-off.
      const vocalFade = transition?.vocalFadeInSec;
      const crossfadeSec = resume
        ? STEM_FADE_IN[stem]
        : Math.max(
          preDur > 0 ? preDur / rate : 0,
          stem === "vocals" && vocalFade !== null && vocalFade !== undefined ? vocalFade : 0,
          STEM_FADE_IN[stem],
        );
      voice.schedule({
        when: startAt,
        inputSec,
        rate,
        semitones: first.keyShift,
        formantCompensation: stem === "vocals",
        fadeInSec: crossfadeSec,
        gain: first.gain,
        // Equal-gain hand-off: fade in at the outgoing voice's level so a
        // quiet/loud phrase splice can't dip, then settle to this gain.
        handoffGain: previous && previous.voice !== voice
          ? previous.voice.outputGainTarget
          : undefined,
      });
      voice.fadeOut(runEndAt, STEM_FADE_OUT[stem]);

      const offsets: number[] = [];
      let offset = 0;
      for (const dur of audio.mainDurations) {
        offsets.push(offset);
        offset += dur / rate;
      }

      if (previous && previous.voice !== voice) {
        previous.voice.fadeOut(startAt, crossfadeSec);
        // (B) EQ-OUT tail: sweep the *outgoing* voice down to 300 Hz across
        // the same crossfade, so its mids do not mask the hotter incoming
        // stem.  Per-voice automation (the incoming voice stays bright),
        // scheduled sample-accurately at the boundary, restored on reuse.
        if (transition?.eqOut && !resume) {
          previous.voice.scheduleEqOut(startAt, crossfadeSec, transition.eqOutFloorHz);
        }
        // (B) Vocal reverb tail: extend the stem's send over the outgoing
        // fade so the tail dissolves into the new vocal.  Absolute-time ramp
        // on the glue bus; falls back to the plain hand-off when the bus
        // is unreachable.
        if (transition?.vocalReverbTail && stem === "vocals" && !resume) {
          this.engine.scheduleReverbTail(stem, startAt, TRANSITION_VOCAL_REVERB_TAIL, TRANSITION_VOCAL_REVERB_TAIL_SEC);
        }
      } else if (transition?.vocalReverbTail && stem === "vocals" && !resume) {
        // Same-voice hand-off (rare): the tail still belongs on the bus.
        this.engine.scheduleReverbTail(stem, startAt, TRANSITION_VOCAL_REVERB_TAIL, TRANSITION_VOCAL_REVERB_TAIL_SEC);
      }
      const scheduled = this.scheduledRuns[stem] ?? [];
      scheduled.push({
        stem,
        run: [...run],
        voice,
        startedAt,
        offsets,
        runEndAt,
        blockIndex: this.blockIndex,
        manual,
      });
      this.scheduledRuns[stem] = scheduled;
      console.info(
        `[StemRemix] ${stem} ← ${run.map((entry) => entry.phrase.id).join("+")} @ ${at.toFixed(2)}s ` +
        `(${bars} bars, start ${startAt.toFixed(2)}s, rate ${rate.toFixed(4)}, ` +
        `shift ${first.keyShift}, pre ${preDur.toFixed(3)}s)${resume ? " (resumed)" : ""}`
      );
    } catch (err) {
      console.warn(`[StemRemix] ${stem} schedule failed`, err);
      this.failedKits.add(first.kit.track_id);
    }
  }

  /** User dragged a reticle: pin the region and commit at the next bar. */
  setTarget(stem: StemName, x: number, y: number): StemPin | null {
    if (this.cluster === null) return null;
    const cx = clamp01(x);
    const cy = clamp01(y);
    const current = this.current[stem]?.entry ?? null;
    const nearest = this.nearest(stem, cx, cy);
    // Pin the region at the nearest real phrase, not at an impossible point
    // outside the stem's cloud; the dashed ring must always sit on the map.
    const pinX = nearest?.x ?? cx;
    const pinY = nearest?.y ?? cy;

    const now = this.engine.now();
    const grid = this.blockAt(now);
    const blockStartAt = grid?.startAt ?? (this.sessionStartAt > 0 ? this.sessionStartAt : now);
    const blockStartBar = grid?.startBar ?? this.blockStartBar;
    const barsIn = Math.max(0, (now - blockStartAt) / this.barSec);
    const offsetBars = Math.ceil(barsIn + 0.02);
    const at = Math.max(now + 0.01, blockStartAt + offsetBars * this.barSec);
    const partialBar = ((offsetBars % 4) + 4) % 4;
    const releaseBar = this.blockEndBar(blockStartBar + offsetBars + PIN_BARS);
    const pin: StemPin = {
      x: pinX,
      y: pinY,
      untilBar: releaseBar,
    };
    this.pins[stem] = pin;
    if (!nearest) return pin;
    if (Math.hypot(nearest.x - cx, nearest.y - cy) > 0.15) {
      this.onStatus?.(`${stem}: target outside the ${stem} cloud, snapped to the nearest legal phrase`);
    }
    const distance = current ? Math.hypot(nearest.x - current.x, nearest.y - current.y) : 1;
    const generation = (this.pinGeneration[stem] ?? 0) + 1;
    this.pinGeneration[stem] = generation;
    this.pinning.add(stem);
    void this.applyPin(stem, nearest, pin, at, partialBar, generation);
    if (distance > 0.06) {
      this.onStatus?.(`${stem} → ${nearest.phrase.section} (${nearest.kit.style}) at next bar`);
    } else {
      const holdBars = Math.max(1, releaseBar - (blockStartBar + offsetBars));
      this.onStatus?.(`${stem} pinned in place for ${holdBars} bars`);
    }
    return pin;
  }

  /**
   * Prefer a nearby phrase from the same kit with a real consecutive run.
   * The pin's centre stays where the user dropped it; only the starting window
   * moves, and never to a different kit.
   */
  private pinRunEntry(entry: StemPoolEntry, phrases: number, x: number, y: number): StemPoolEntry {
    const direct = this.runFor(entry, phrases, false);
    if (direct.length >= Math.min(2, phrases)) return entry;
    let best = entry;
    let bestLength = direct.length;
    let bestDistance = 0;
    for (const candidate of this.pools[entry.stem]) {
      if (candidate.kit.track_id !== entry.kit.track_id) continue;
      const distance = Math.hypot(candidate.x - x, candidate.y - y);
      if (distance > 0.2) continue;
      const length = this.runFor(candidate, phrases, false).length;
      if (length > bestLength || (length === bestLength && distance < bestDistance)) {
        best = candidate;
        bestLength = length;
        bestDistance = distance;
      }
    }
    return best;
  }

  /**
   * Commit a pinned style region at the next bar.  The voice only covers the
   * partial run up to the next block boundary (`handoffAt`); from there the
   * normal block scheduler keeps the pin and selects fresh phrases inside it.
   */
  private async applyPin(
    stem: StemName,
    entry: StemPoolEntry,
    pin: StemPin,
    at: number,
    partialBar: number,
    generation: number,
  ): Promise<void> {
    const handoffAt = this.nextBlockAt;
    if (at >= handoffAt - 0.05) {
      // The drag landed on/after the block boundary: no partial manual voice.
      // The block scheduler will pick the pinned phrase at the boundary.
      delete this.manualPins[stem];
      this.pinning.delete(stem);
      return;
    }
    const blockSpan = 4 * this.barSec;
    const phrases = Math.max(1, Math.ceil((handoffAt - at) / blockSpan) + 1);
    // If the snapped phrase is an isolated window, prefer a nearby phrase from
    // the same kit with a real consecutive run; the partial commit should not
    // loop one 4-bar window if the kit can move.
    const runEntry = this.pinRunEntry(entry, phrases, pin.x, pin.y);
    const run = this.runFor(runEntry, phrases);
    const audio = await this.loadRunAudio(run);
    if (!audio || this.pins[stem] !== pin || this.pinGeneration[stem] !== generation) {
      if (this.pinGeneration[stem] === generation) {
        delete this.pins[stem];
        this.pinning.delete(stem);
        this.onStatus?.(`${stem}: target phrase failed to load; keeping current`);
      }
      return;
    }
    try {
      const previous = this.current[stem];
      const voice = await this.getVoice(stem, previous?.voice);
      const rate = Math.max(RATE_MIN, Math.min(RATE_MAX, runEntry.rate));
      await voice.loadSamples(audio.samples);
      const preDur = audio.preDur;
      const crossfadeSec = Math.max(0.02, STEM_FADE_IN[stem], preDur / rate);
      let startAt = at;
      let inputBars = partialBar;
      const now = this.engine.now();
      if (now > at - 0.05) {
        const lateBars = Math.min(6, Math.ceil((now - at) / this.barSec + 0.1));
        startAt += Math.max(1, lateBars) * this.barSec;
        inputBars += Math.max(1, lateBars);
      }
      if (startAt >= handoffAt - 0.05) {
        // The load arrived after the partial window; hand straight over to the
        // block scheduler at the boundary instead of scheduling a dead voice.
        delete this.manualPins[stem];
        this.pinning.delete(stem);
        this.onStatus?.(`${stem}: pin commit late; starting at the next block`);
        return;
      }
      let sourceSec = 0;
      for (const dur of audio.mainDurations) sourceSec += dur;
      const firstDur = audio.mainDurations[0] ?? 0;
      const inputSec = Math.min(
        preDur + inputBars * (firstDur / 4),
        preDur + sourceSec - 0.05,
      );
      const endAt = handoffAt;
      voice.schedule({
        when: startAt,
        inputSec: Math.max(preDur, inputSec),
        rate,
        semitones: runEntry.keyShift,
        formantCompensation: stem === "vocals",
        fadeInSec: crossfadeSec,
        gain: runEntry.gain,
        handoffGain: previous && previous.voice !== voice
          ? previous.voice.outputGainTarget
          : undefined,
      });
      voice.fadeOut(endAt, STEM_FADE_OUT[stem]);

      // Silence the current voice and any block voice already queued inside
      // the partial commit window; this stem is manual until the handoff.
      if (previous && previous.voice !== voice) previous.voice.fadeOut(startAt, crossfadeSec);
      const kept: ScheduledRun[] = [];
      for (const scheduled of this.scheduledRuns[stem] ?? []) {
        if (scheduled.startedAt >= startAt - 0.02 && scheduled.startedAt < endAt) {
          if (scheduled.voice !== voice) scheduled.voice.fadeOut(scheduled.startedAt, 0.01);
        } else {
          kept.push(scheduled);
        }
      }
      const offsets: number[] = [];
      let offset = 0;
      for (const dur of audio.mainDurations) {
        offsets.push(offset);
        offset += dur / rate;
      }
      if (this.pinGeneration[stem] !== generation) {
        voice.fadeOut(startAt, 0.02);
        this.pinning.delete(stem);
        return;
      }
      kept.push({
        stem,
        run,
        voice,
        startedAt: startAt,
        offsets,
        runEndAt: endAt,
        blockIndex: this.blockIndex,
        manual: true,
      });
      this.scheduledRuns[stem] = kept;
      this.manualPins[stem] = {
        entry: runEntry,
        run,
        voice,
        startedAt: startAt,
        runEndAt: endAt,
        untilBar: pin.untilBar,
      };
      for (const phraseEntry of run) {
        this.recordPlayed(stem, phraseEntry);
      }
      this.pinning.delete(stem);
    } catch (err) {
      console.warn(`[StemRemix] ${stem} pinned run failed`, err);
      this.failedKits.add(entry.kit.track_id);
      delete this.pins[stem];
      this.pinning.delete(stem);
      this.onStatus?.(`${stem}: pinned run failed; keeping current`);
    }
  }

  release(stem: StemName): void {
    delete this.pins[stem];
    delete this.pending[stem];
    this.pinGeneration[stem] = (this.pinGeneration[stem] ?? 0) + 1;
    this.pinning.delete(stem);
    const manual = this.manualPins[stem];
    if (manual) {
      try {
        manual.voice.fadeOut(this.engine.now() + 0.02, 0.05);
      } catch {
        // The context may be closing.
      }
      delete this.manualPins[stem];
      this.scheduledRuns[stem] = (this.scheduledRuns[stem] ?? [])
        .filter((scheduled) => !(scheduled.manual && scheduled.voice === manual.voice));
    }
    this.onStatus?.(`${stem} released back to auto-mix`);
  }

  private nearest(stem: StemName, x: number, y: number): StemPoolEntry | null {
    let best: StemPoolEntry | null = null;
    let dist = Number.POSITIVE_INFINITY;
    for (const entry of this.pools[stem]) {
      if (this.failedKits.has(entry.kit.track_id)) continue;
      const d = Math.hypot(entry.x - x, entry.y - y);
      if (d < dist) {
        dist = d;
        best = entry;
      }
    }
    return best;
  }

  /** Called every scheduler tick; promote scheduled runs, release pins. */
  tick(now: number): void {
    if (this.blockGrid.length > 24) {
      this.blockGrid = this.blockGrid.filter((block) => block.endAt > now - 8);
    }
    for (const stem of STEMS) {
      const list = (this.scheduledRuns[stem] ?? [])
        .filter((scheduled) => scheduled.runEndAt > now - 0.5);
      let active: ScheduledRun | null = null;
      for (const scheduled of list) {
        if (scheduled.startedAt <= now + 0.01
          && (active === null || scheduled.startedAt >= active.startedAt)) {
          active = scheduled;
        }
      }
      if (active) {
        let index = 0;
        for (let i = 0; i < active.offsets.length; i++) {
          if (now >= active.startedAt + active.offsets[i] - 0.02) index = i;
        }
        const entry = active.run[Math.min(index, active.run.length - 1)];
        const current = this.current[stem];
        if (!current || current.voice !== active.voice) {
          this.current[stem] = {
            entry,
            voice: active.voice,
            startedAt: active.startedAt,
            blockIndex: active.blockIndex,
            run: active.run,
            offsets: active.offsets,
            runEndAt: active.runEndAt,
          };
        } else {
          current.entry = entry;
        }
      }
      this.scheduledRuns[stem] = list;

      const manual = this.manualPins[stem];
      if (manual && now >= manual.runEndAt - 0.02) {
        // The partial manual commit has handed off.  Keep pins[stem] so the
        // block scheduler can keep varying phrases inside the region until the
        // 32-bar boundary.
        delete this.manualPins[stem];
        const playing = this.current[stem]?.entry;
        if (playing) {
          this.lastBlockEntries[stem] = playing;
          this.lastBlockRuns[stem] = [playing];
        }
      }
      const pin = this.pins[stem];
      if (pin && this.blockStartBar >= pin.untilBar) {
        // Safety net; beginBlock normally releases at the same boundary.
        delete this.pins[stem];
        delete this.manualPins[stem];
        this.pinning.delete(stem);
        const playing = this.current[stem]?.entry;
        if (playing) {
          this.lastBlockEntries[stem] = playing;
          this.lastBlockRuns[stem] = [playing];
        }
        this.onStatus?.(`${stem} released back to auto-mix`);
      }
    }

    // Learned-arc context: record each newly-playing drums phrase once, so
    // beginBlock can offer the last <=6 windows to the arc model.
    const drumsNow = this.current.drums;
    if (drumsNow) {
      const phraseId = drumsNow.entry.phrase.id;
      if (this.arcHistory[this.arcHistory.length - 1] !== phraseId) {
        this.arcHistory.push(phraseId);
        if (this.arcHistory.length > 6) this.arcHistory.shift();
      }
    }
  }

  state(): StemRemixState[] {
    return STEMS.map((stem) => {
      const current = this.current[stem];
      const pin = this.pins[stem] ?? null;
      return {
        stem,
        phraseId: current?.entry.phrase.id ?? null,
        trackId: current?.entry.kit.track_id ?? null,
        section: current?.entry.section ?? "",
        energy: current?.entry.energy ?? 0,
        x: current?.entry.x ?? 0.5,
        y: current?.entry.y ?? 0.5,
        pin,
        buffering: this.pinning.has(stem),
      };
    });
  }

  points(): NebulaPoint[] {
    const points: NebulaPoint[] = [];
    for (const stem of STEMS) {
      for (const entry of this.pools[stem]) {
        points.push({
          x: entry.x,
          y: entry.y,
          stem,
          trackId: entry.kit.track_id,
          phraseId: entry.phrase.id,
          kitId: entry.kit.track_id,
        });
      }
    }
    return points;
  }

  activePoints(): NebulaPoint[] {
    const points: NebulaPoint[] = [];
    for (const stem of STEMS) {
      const current = this.current[stem];
      if (!current) continue;
      points.push({
        x: current.entry.x,
        y: current.entry.y,
        stem,
        trackId: current.entry.kit.track_id,
        phraseId: current.entry.phrase.id,
        kitId: current.entry.kit.track_id,
        active: true,
      });
    }
    return points;
  }

  private chooseNeighbourCluster(): KitCluster | null {
    if (!this.cluster) return null;
    // BPM-ladder walk: clusters sorted by tempo (then key), step one rung
    // per set, bouncing at the ends.  The old ±3 BPM / Camelot-neighbour rule
    // trapped the session in a single tempo band forever (e.g. 129 BPM could
    // only ever reach 126/129), so whole BPM groups of the bank were never
    // in session.  Walking the ladder systematically brings every tempo
    // group into range over successive sets; any-key transposition makes the
    // key jump at the set boundary safe (fresh anchor record).
    const ladder = [...this.clusters].sort((a, b) => {
      if (a.bpm !== b.bpm) return a.bpm - b.bpm;
      return a.key.localeCompare(b.key) || a.id.localeCompare(b.id);
    });
    const index = ladder.findIndex((c) => c.id === this.cluster!.id);
    if (index < 0 || ladder.length < 2) return null;
    let next = index + this.ladderDirection;
    if (next < 0 || next >= ladder.length) {
      this.ladderDirection = -this.ladderDirection;
      next = index + this.ladderDirection;
    }
    if (next < 0 || next >= ladder.length) return null;
    return ladder[next] ?? null;
  }

  stop(): void {
    for (const stem of STEMS) {
      const current = this.current[stem];
      if (current) current.voice.fadeOut(this.engine.now(), 0.05);
    }
    for (const stem of STEMS) {
      for (const voice of this.voicePool[stem] ?? []) voice.dispose();
    }
    this.voicePool = {};
    this.current = {};
    this.pending = {};
    this.pins = {};
    this.manualPins = {};
    this.scheduledRuns = {};
    this.lastBlockRuns = {};
    this.pinning.clear();
    this.pinGeneration = {};
    this.audioWorker.dispose();
  }
}

const SECTION_ENERGY: Record<string, number> = {
  INTRO: 0.25,
  BUILDUP: 0.65,
  DROP: 0.90,
  GROOVE_A: 0.83,
  BREAKDOWN: 0.20,
};

const TARGET_RMS_DB: Record<StemName, number> = {
  drums: -17, bass: -18, other: -20, vocals: -23,
};

/** Runtime level match for quiet phrases, capped at ~+7 dB, never baked. */
function stemGain(stem: StemName, rmsDb: number): number {
  const delta = TARGET_RMS_DB[stem] - rmsDb;
  return Math.max(0.45, Math.min(2.2, Math.pow(10, delta / 20)));
}

function stemPan(stem: StemName, key: string): number {
  if (stem === "bass" || stem === "drums") return 0;
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) | 0;
  const side = (hash & 1) === 0 ? -1 : 1;
  return stem === "vocals" ? side * 0.18 : side * 0.3;
}
