// M1 Kit Player: plays locked, real 4-bar stem phrases straight from the
// export kit -- no resampling, no slice resynthesis, no pitch shifting.
//
// Each kit contains up to 8 phrases x 4 stems. A phrase is scheduled as four
// independent AudioBufferSources at rate 1.0, so Opus's native decode rate is
// never reinterpreted. Block timing comes from the fitted kit tempo
// (`960 / median phrase duration`), not from sample counts.
import type { StemName } from "./types";
import type { AudioEngine } from "./audio_engine";
import type { NebulaPoint } from "./visualizer";

export interface KitStemAsset {
  file: string;
  peak: number;
  rms_db: number;
}

export interface KitPhrase {
  id: string;
  section: string;
  start_bar: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  energy: number;
  energy_db: number;
  embedding: number[];
  stems: Record<StemName, KitStemAsset>;
  x?: number;
  y?: number;
  /** 12-d harmonic chroma for transition compatibility (optional). */
  chroma?: number[];
  /** Pitch class of the bass root, 0..11 (optional). */
  bass_root?: number;
  /** First bars of the source song. */
  is_intro?: boolean;
  /** Last bars of the source song. */
  is_outro?: boolean;
  /** Intro/outro phrases are played at most once per set. */
  once?: boolean;
}

export type KitSteering = Record<StemName, { x: number; y: number }>;
export type KitSteeringMask = ReadonlySet<StemName>;

export interface Kit {
  track_id: string;
  title: string;
  bpm: number;
  key: string;
  style: string;
  source?: string;
  phrases: KitPhrase[];
}

export interface KitManifest {
  version: number;
  bars: number;
  sample_rate: number;
  bitrate: string;
  kits: Kit[];
}

export interface TasteNextKit {
  track_id: string;
  score: number;
  style: string;
  bpm: number;
  key: string;
}

export interface TasteKitScores {
  phrase_order: string[];
  transition: number[][];
  quality: number[];
  next: TasteNextKit[];
}

export interface TasteScoresManifest {
  version: number;
  model: string;
  val_auc: number;
  kits: Record<string, TasteKitScores>;
}

export interface TasteTransitionTop {
  id: string;
  weight: number;
}

/** One scheduled 4-bar block, including rotation/bridge metadata. */
export interface KitBlockResult {
  kit: Kit;
  phrase: KitPhrase;
  stems: StemName[];
  bridge: boolean;
  setChanged: boolean;
  blocksPlayedInSet: number;
  blockDurationSec: number;
  fitBpm: number | null;
  targetSection: string;
  energyDelta: number;
  selectionBand: string;
}

interface LoadedKitPhrase {
  meta: KitPhrase;
  buffers: Partial<Record<StemName, AudioBuffer>>;
}

interface LoadedKitData {
  kit: Kit;
  phrases: LoadedKitPhrase[];
  medianDurationSec: number;
  fitBpm: number | null;
}

interface CamelotPosition {
  number: number;
  letter: "A" | "B";
}

/** Number of 4-bar blocks in one DJ set before the bridge/rotation. */
export const KIT_SET_BLOCKS = 12;

/** Reject only truly tiny phrases; per-phrase slots handle normal variation. */
const MIN_PHRASE_DURATION_SEC = 3.0;

const STEM_ORDER: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const BRIDGE_STEMS: readonly StemName[] = ["other", "vocals"];
const ALL_STEMS: readonly StemName[] = STEM_ORDER;

// Per-stem crossfade profile: drums snap, bass settles quickly, while pads and
// vocals overlap for a musical hand-off.
const KIT_FADE_MS: Record<StemName, number> = {
  drums: 18,
  bass: 60,
  other: 250,
  vocals: 250,
};

// The 18/60/250 ms values above govern the choke crossfade. At an exact
// per-phrase boundary there is no overlap to crossfade, so keep the source's
// own seam fades short enough not to notch the 20 ms RMS envelope.
const KIT_SEAM_FADE_IN_MS: Record<StemName, number> = {
  drums: 6,
  bass: 12,
  other: 20,
  vocals: 20,
};
const KIT_SEAM_FADE_OUT_MS: Record<StemName, number> = {
  drums: 8,
  bass: 16,
  other: 20,
  vocals: 20,
};

// Section energy targets mirror ClientSongArranger so phrase choice follows the
// same arrangement arc the sequencer is currently in.
const SECTION_ENERGY: Record<string, number> = {
  INTRO: 0.25,
  BUILDUP: 0.65,
  DROP: 0.90,
  GROOVE_A: 0.83,
  BREAKDOWN: 0.20,
};

// Arrangement order used for adjacent-section fallback. DROP/GROOVE_A blocks
// must not borrow INTRO/BREAKDOWN phrases unless the arranger itself is there.
const SECTION_CHAIN: readonly string[] = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"];

const PITCH_CLASS: Record<string, number> = {
  C: 0, "C#": 1, DB: 1, D: 2, "D#": 3, EB: 3, E: 4, F: 5,
  "F#": 6, GB: 6, G: 7, "G#": 8, AB: 8, A: 9, "A#": 10, BB: 10, B: 11,
};

// Camelot numbers indexed by pitch class (C..B). A minor = 8A, C major = 8B.
const CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
const CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10];

function normaliseSection(section: string): string {
  return section.trim().toUpperCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseKey(keyName: string): { root: number; minor: boolean } | null {
  const match = /^([A-Ga-g])([#b♯♭]?)\s*(major|minor|maj|min)$/.exec(keyName.trim());
  if (!match) return null;
  const accidental = match[2] === "♯" ? "#" : match[2] === "♭" ? "B" : match[2].toUpperCase();
  const root = PITCH_CLASS[`${match[1].toUpperCase()}${accidental}`];
  if (root === undefined) return null;
  return { root, minor: match[3].toLowerCase().startsWith("min") };
}

function camelotForKey(keyName: string): CamelotPosition | null {
  const parsed = parseKey(keyName);
  if (parsed === null) return null;
  const table = parsed.minor ? CAMELOT_MINOR : CAMELOT_MAJOR;
  const number = table[parsed.root];
  return { number, letter: parsed.minor ? "A" : "B" };
}

function camelotDistance(a: CamelotPosition, b: CamelotPosition): number {
  if (a.number === b.number) return a.letter === b.letter ? 0 : 1;
  if (a.letter !== b.letter) return 99;
  const diff = Math.abs(a.number - b.number);
  return Math.min(diff, 12 - diff);
}

function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const length = Math.min(a.length, b.length);
  if (length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < length; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA <= 0 || normB <= 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Deterministic FNV-1a hash -> mulberry32 roll. Selection is reproducible for
// the same inputs without touching Math.random.
function deterministicRoll(input: string): number {
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

function joinUrl(base: string, file: string): string {
  return `${base.replace(/\/+$/, "")}/${file.replace(/^\/+/, "")}`;
}

export class KitPlayer {
  public static readonly KIT_SET_BLOCKS = KIT_SET_BLOCKS;

  /** Blocks scheduled in the current kit set (0-based while under way). */
  public blocksInSet: number = 0;
  /** The next kit selected for the upcoming set change (exposed for UI). */
  public nextKit: Kit | null = null;
  /** Number of successful set changes in this session. */
  public setChanges: number = 0;

  private readonly engine: AudioEngine;
  private baseUrl: string = "";
  private manifest: KitManifest | null = null;
  private kit: Kit | null = null;
  private phrases: LoadedKitPhrase[] = [];
  private recentPhraseIds: string[] = [];
  private playedPhraseIds: Set<string> = new Set();
  /** Kits whose audio failed to load this session; never retry them. */
  private failedKitIds: Set<string> = new Set();
  private sources: AudioBufferSourceNode[] = [];
  private lastBlockStems: Set<StemName> = new Set();
  private lastStemEndAt: Partial<Record<StemName, number>> = {};
  private readyFlag: boolean = false;
  private medianDurationSec: number = 0;
  private fittedBpm: number | null = null;
  private lastPhrase: KitPhrase | null = null;
  private lastSection: string | null = null;
  private kitHistory: string[] = [];
  private compatibility: Kit[] = [];
  private pendingLoadedKit: LoadedKitData | null = null;
  private pendingLoadPromise: Promise<boolean> | null = null;
  private bridgeBlocksForNext: number = 1;
  private forcedBridgeRemaining: number = 0;
  private waitingForNextLogged: boolean = false;
  private tasteScores: TasteScoresManifest | null = null;
  private tasteFailureLogged: boolean = false;
  private tasteRngState: number = 0x9e3779b9;
  private lastTransition: TasteTransitionTop[] = [];
  private lastKitScoreValue: number | null = null;
  private lastSelectionTarget: string = "";
  private lastSelectionBand: string = "none";
  private lastSelectionDelta: number = 0;
  private lastSteerTargetPoint: { x: number; y: number } | null = null;
  private lastSteerDistanceValue: number | null = null;

  constructor(engine: AudioEngine) {
    this.engine = engine;
  }

  get ready(): boolean {
    return this.readyFlag;
  }

  get currentKit(): Kit | null {
    return this.kit;
  }

  /** Alias used by the UI/set-rotation state. */
  get activeKit(): Kit | null {
    return this.kit;
  }

  get loadedPhraseCount(): number {
    return this.phrases.length;
  }

  /** Manifest BPM (metadata; may differ from the audio cut lengths). */
  get bpm(): number | null {
    return this.kit?.bpm ?? null;
  }

  /** Tempo fitted to the median phrase duration: 960 / median seconds. */
  get fitBpm(): number | null {
    return this.fittedBpm;
  }

  get medianPhraseDuration(): number {
    return this.medianDurationSec;
  }

  /** The phrase chosen by the most recent schedule() call. */
  get lastScheduledPhrase(): KitPhrase | null {
    return this.lastPhrase;
  }

  get key(): string | null {
    return this.kit?.key ?? null;
  }

  get kitSetBlocksLeft(): number {
    return Math.max(0, KIT_SET_BLOCKS - this.blocksInSet);
  }

  get nextKitTrack(): string | null {
    return this.nextKit?.track_id ?? null;
  }

  get kitSteerTarget(): { x: number; y: number } | null {
    return this.lastSteerTargetPoint === null ? null : { ...this.lastSteerTargetPoint };
  }

  get kitSteerDistance(): number | null {
    return this.lastSteerDistanceValue;
  }

  get compatibleKits(): Kit[] {
    return [...this.compatibility];
  }

  /** Fired when a phrase is scheduled, with its nebula coordinate. */
  public onPhraseScheduled?: (p: {
    x: number;
    y: number;
    at: number;
    stem: StemName;
    phraseId: string;
  }) => void;

  /** One nebula point per phrase in the active kit, coloured by loudest stem. */
  public nebulaPoints(): NebulaPoint[] {
    const kit = this.kit;
    if (kit === null) return [];
    return this.phrases.map((entry) => ({
      x: entry.meta.x ?? 0.5,
      y: entry.meta.y ?? 0.5,
      stem: this.loudestStem(entry.meta),
      trackId: kit.track_id,
    }));
  }

  /** The full 355-kit / 2,466-phrase space, independent of the active kit. */
  public allPhrasePoints(): NebulaPoint[] {
    const manifest = this.manifest;
    if (manifest === null) return [];
    const points: NebulaPoint[] = [];
    for (const kit of manifest.kits) {
      for (const phrase of kit.phrases) {
        points.push({
          x: phrase.x ?? 0.5,
          y: phrase.y ?? 0.5,
          stem: this.loudestStem(phrase),
          trackId: kit.track_id,
        });
      }
    }
    return points;
  }

  private loudestStem(phrase: KitPhrase): StemName {
    let loudest: StemName = "drums";
    let bestDb = Number.NEGATIVE_INFINITY;
    for (const stem of STEM_ORDER) {
      const db = phrase.stems?.[stem]?.rms_db;
      if (typeof db === "number" && Number.isFinite(db) && db > bestDb) {
        bestDb = db;
        loudest = stem;
      }
    }
    return loudest;
  }

  /** Stems that own the steering target: only stems the user actually dragged.
   *  With no drag, steering is fully disabled so the taste matrix alone decides
   *  (the visual auto-follow must never bias the music). */
  private steeringStems(userSteered?: KitSteeringMask): readonly StemName[] {
    if (userSteered !== undefined && userSteered.size > 0) {
      return STEM_ORDER.filter((stem) => userSteered.has(stem));
    }
    return [];
  }

  /** Mean reticle position used for kit-distance and stats. */
  private steeringTarget(
    steering?: KitSteering,
    userSteered?: KitSteeringMask
  ): { x: number; y: number } | null {
    if (!steering) return null;
    let sx = 0;
    let sy = 0;
    let count = 0;
    for (const stem of this.steeringStems(userSteered)) {
      const point = steering[stem];
      if (!point) continue;
      sx += point.x;
      sy += point.y;
      count++;
    }
    return count > 0 ? { x: sx / count, y: sy / count } : null;
  }

  /** Distance from a steering target to the nearest phrase in the active kit. */
  private closestPhraseDistance(target: { x: number; y: number }): number | null {
    if (this.phrases.length === 0) return null;
    let nearest = Number.POSITIVE_INFINITY;
    for (const entry of this.phrases) {
      const dx = (entry.meta.x ?? 0.5) - target.x;
      const dy = (entry.meta.y ?? 0.5) - target.y;
      nearest = Math.min(nearest, Math.hypot(dx, dy));
    }
    return Number.isFinite(nearest) ? nearest : null;
  }

  /** Reticle bias: 0.15 when far from the user-steered stems, 2.55 at centre. */
  private steeringMultiplier(
    phrase: KitPhrase,
    steering?: KitSteering,
    userSteered?: KitSteeringMask
  ): number {
    if (!steering) return 1;
    const stems = this.steeringStems(userSteered);
    const px = phrase.x ?? 0.5;
    const py = phrase.y ?? 0.5;
    let total = 0;
    let count = 0;
    for (const stem of stems) {
      const point = steering[stem];
      if (!point) continue;
      const dx = px - point.x;
      const dy = py - point.y;
      total += Math.exp(-(dx * dx + dy * dy) / (2 * 0.16 * 0.16));
      count++;
    }
    if (count === 0) return 1;
    const mean = total / count;
    return 0.15 + 2.4 * mean;
  }

  get tasteModelName(): string | null {
    return this.tasteScores?.model ?? null;
  }

  get tasteValAuc(): number | null {
    return this.tasteScores?.val_auc ?? null;
  }

  get lastTransitionTop3(): TasteTransitionTop[] {
    return [...this.lastTransition];
  }

  get lastKitScore(): number | null {
    return this.lastKitScoreValue;
  }

  get tasteReady(): boolean {
    return this.tasteScores !== null;
  }

  /** Fetch and validate the kit manifest. `basePath` is the manifest directory. */
  async loadManifest(basePath: string): Promise<KitManifest> {
    const base = basePath.replace(/\/+$/, "");
    const url = joinUrl(base, "kit_manifest.json");
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Kit manifest request failed (${response.status}) for ${url}`);
    }

    const raw: unknown = await response.json();
    if (!isRecord(raw) || !Array.isArray(raw["kits"]) || raw["kits"].length === 0) {
      throw new Error(`Kit manifest at ${url} has no kits`);
    }
    for (const candidate of raw["kits"]) {
      if (!isRecord(candidate) || typeof candidate["track_id"] !== "string" || !Array.isArray(candidate["phrases"])) {
        throw new Error(`Kit manifest at ${url} contains an invalid kit entry`);
      }
    }

    const manifest = raw as unknown as KitManifest;
    this.baseUrl = base;
    this.manifest = manifest;
    console.info(`[KitPlayer] Manifest loaded: ${manifest.kits.length} kits from ${base}`);
    return manifest;
  }

  /**
   * Load the trained taste model. A missing/broken file is non-fatal: log once
   * and keep the rule-based selector/rotation.
   */
  public async loadTasteScores(basePath: string): Promise<boolean> {
    const url = joinUrl(basePath, "taste_scores.json");
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Taste scores request failed (${response.status}) for ${url}`);
      }
      const raw: unknown = await response.json();
      if (
        !isRecord(raw) ||
        !isRecord(raw["kits"]) ||
        typeof raw["val_auc"] !== "number" ||
        typeof raw["model"] !== "string"
      ) {
        throw new Error(`Taste scores at ${url} are invalid`);
      }
      const scores = raw as unknown as TasteScoresManifest;
      this.tasteScores = scores;
      console.info(
        `[KitPlayer] Taste scores loaded: ${scores.model} (val_auc ${scores.val_auc.toFixed(4)}, ${Object.keys(scores.kits).length} kits)`
      );
      return true;
    } catch (err) {
      if (!this.tasteFailureLogged) {
        this.tasteFailureLogged = true;
        console.warn("[KitPlayer] Taste scores unavailable — using rule-based selection.", err);
      }
      this.tasteScores = null;
      return false;
    }
  }

  private async loadKitData(kit: Kit): Promise<LoadedKitData> {
    if (this.manifest === null) {
      throw new Error("KitPlayer.loadKitData called before loadManifest");
    }
    if (!Array.isArray(kit.phrases) || kit.phrases.length === 0) {
      throw new Error(`Kit ${kit.track_id} has no phrases`);
    }

    const phrases = await Promise.all(
      kit.phrases.map(async (phrase): Promise<LoadedKitPhrase> => {
        const decoded = await Promise.all(
          STEM_ORDER.map(async (stem): Promise<readonly [StemName, AudioBuffer | null]> => {
            const asset = phrase.stems?.[stem];
            if (!asset || typeof asset.file !== "string" || asset.file.length === 0) {
              return [stem, null] as const;
            }
            const response = await fetch(joinUrl(this.baseUrl, asset.file));
            if (!response.ok) {
              throw new Error(`Stem request failed (${response.status}) for ${asset.file}`);
            }
            const bytes = await response.arrayBuffer();
            const buffer = await this.engine.decodeAudio(bytes);
            return [stem, buffer] as const;
          })
        );

        const buffers: Partial<Record<StemName, AudioBuffer>> = {};
        for (const [stem, buffer] of decoded) {
          if (buffer !== null) buffers[stem] = buffer;
        }
        return { meta: phrase, buffers };
      })
    );

    const durations = phrases
      .map((phrase) => phrase.meta.duration_sec)
      .filter((duration) => Number.isFinite(duration) && duration > 0)
      .sort((a, b) => a - b);
    const midpoint = Math.floor(durations.length / 2);
    const median = durations.length === 0
      ? 0
      : (durations.length % 2 === 0
        ? (durations[midpoint - 1] + durations[midpoint]) / 2
        : durations[midpoint]);

    return {
      kit,
      phrases,
      medianDurationSec: median,
      fitBpm: median > 0 ? 960 / median : null,
    };
  }

  /** Decode all phrases x 4 stems for one kit and make it active. */
  async loadKit(kit: Kit): Promise<void> {
    if (this.manifest === null) {
      throw new Error("KitPlayer.loadKit called before loadManifest");
    }
    if (this.readyFlag && this.kit?.track_id === kit.track_id) return;

    const data = await this.loadKitData(kit);
    this.setActiveLoaded(data);
    console.info(
      `[KitPlayer] Kit ready: ${kit.track_id} • ${kit.bpm.toFixed(2)} BPM • ${kit.key} • ${data.phrases.length} phrases`
    );
    console.info(
      `[KitPlayer] Compatibility list: ${this.compatibility.length} kits (${kit.style}, ${kit.key}, ${kit.bpm.toFixed(2)} BPM)`
    );
  }

  /**
   * Swap in a loaded kit and reset the set counter. The caller is responsible
   * for the session tempo/key lock (the conductor does that on set change).
   */
  private setActiveLoaded(data: LoadedKitData): void {
    this.kit = data.kit;
    this.phrases = data.phrases;
    this.medianDurationSec = data.medianDurationSec;
    this.fittedBpm = data.fitBpm;
    this.readyFlag = true;
    this.blocksInSet = 0;
    this.bridgeBlocksForNext = 1;
    this.forcedBridgeRemaining = 0;
    this.lastPhrase = null;
    this.lastSection = null;
    this.recentPhraseIds = [];
    this.playedPhraseIds.clear();
    this.lastBlockStems.clear();
    this.lastStemEndAt = {};
    this.pendingLoadedKit = null;
    this.pendingLoadPromise = null;
    this.nextKit = null;
    this.waitingForNextLogged = false;
    this.lastTransition = [];
    this.seedTasteRng(data.kit.track_id);
    this.recordKitHistory(data.kit.track_id);
    this.compatibility = this.computeCompatibility(data.kit);
    if (this.fittedBpm !== null) {
      console.info(
        `[KitPlayer] Tempo fit: manifest ${data.kit.bpm.toFixed(2)} → ${this.fittedBpm.toFixed(2)} BPM (median phrase ${data.medianDurationSec.toFixed(2)}s)`
      );
    }
  }

  private recordKitHistory(trackId: string): void {
    this.kitHistory = [trackId, ...this.kitHistory.filter((id) => id !== trackId)].slice(0, 5);
  }

  private seedTasteRng(trackId: string): void {
    let hash = 2166136261;
    for (let i = 0; i < trackId.length; i++) {
      hash ^= trackId.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    this.tasteRngState = hash >>> 0;
  }

  private nextTasteRandom(): number {
    let t = (this.tasteRngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * 1.0 when all four stems are present, 0.25 when any is effectively empty,
   * 0 when all four are empty. Mix fullness without hard-excluding quiet
   * breakdown phrases.
   */
  private stemPresenceMultiplier(phrase: KitPhrase): number {
    let missing = 0;
    for (const stem of STEM_ORDER) {
      const peak = phrase.stems?.[stem]?.peak;
      if (typeof peak !== "number" || !Number.isFinite(peak) || peak < 0.05) {
        missing++;
      }
    }
    if (missing === 0) return 1;
    if (missing === STEM_ORDER.length) return 0;
    return 0.25;
  }

  /**
   * Compatibility pool for the next set: same style, Camelot distance <= 1,
   * BPM within 4 then 10 then closest available, excluding the last 5 kits.
   */
  private computeCompatibility(current: Kit): Kit[] {
    const kits = this.manifest?.kits ?? [];
    const history = new Set(this.kitHistory);
    const base = kits.filter(
      (kit) => kit.track_id !== current.track_id && !history.has(kit.track_id) && kit.style === current.style
    );
    const currentCamelot = camelotForKey(current.key);
    const camelotOk = currentCamelot === null
      ? base
      : base.filter((kit) => {
        const candidate = camelotForKey(kit.key);
        return candidate !== null && camelotDistance(currentCamelot, candidate) <= 1;
      });

    const preferBpm = (list: Kit[]): Kit[] => {
      const close4 = list.filter((kit) => Math.abs(kit.bpm - current.bpm) <= 4);
      if (close4.length > 0) return close4;
      const close10 = list.filter((kit) => Math.abs(kit.bpm - current.bpm) <= 10);
      if (close10.length > 0) return close10;
      // No same-style kit near the tempo: keep the closest available so the set
      // can still rotate (the bridge extends to two blocks in this case).
      return [...list]
        .sort((a, b) => Math.abs(a.bpm - current.bpm) - Math.abs(b.bpm - current.bpm))
        .slice(0, 8);
    };

    let pool = preferBpm(camelotOk);
    if (pool.length === 0) pool = preferBpm(base);
    if (pool.length === 0) {
      const anyStyle = kits.filter(
        (kit) => kit.track_id !== current.track_id && !history.has(kit.track_id)
      );
      pool = preferBpm(anyStyle);
    }
    return pool;
  }

  private chooseNextKit(): Kit | null {
    const current = this.kit;
    if (current === null) return null;

    // Trained rotation: current kit's next list is already style/BPM filtered.
    const tasteKit = this.tasteScores?.kits[current.track_id];
    if (tasteKit && Array.isArray(tasteKit.next) && tasteKit.next.length > 0) {
      const kits = this.manifest?.kits ?? [];
      const eligible = tasteKit.next
        .filter(
          (entry) =>
            entry !== null &&
            typeof entry === "object" &&
            typeof entry.track_id === "string" &&
            !this.kitHistory.includes(entry.track_id) &&
            !this.failedKitIds.has(entry.track_id)
        )
        .map((entry) => ({
          entry,
          kit: kits.find((kit) => kit.track_id === entry.track_id),
        }))
        .filter((row): row is { entry: TasteNextKit; kit: Kit } => row.kit !== undefined);
      if (eligible.length > 0) {
        const strong = eligible.filter(
          (row) =>
            row.kit.phrases.some((ph) => ph.section === "DROP" && ph.energy >= 0.8) &&
            row.kit.phrases.some((ph) => ph.section === "GROOVE_A")
        );
        const ranked = strong.length > 0 ? strong : eligible;
        ranked.sort(
          (a, b) => b.entry.score - a.entry.score || a.entry.track_id.localeCompare(b.entry.track_id)
        );
        const best = ranked[0];
        this.lastKitScoreValue = best.entry.score;
        console.info(
          `[KitPlayer] Taste next kit: ${best.entry.track_id} (score ${best.entry.score.toFixed(4)}, ${best.entry.style}, ${best.entry.bpm.toFixed(2)} BPM)`
        );
        return best.kit;
      }
    }

    // Fallback: existing same-style/Camelot/BPM compatibility pool.
    this.lastKitScoreValue = null;
    const pool = this.compatibility.filter(
      (kit) => !this.kitHistory.includes(kit.track_id) && !this.failedKitIds.has(kit.track_id)
    );
    if (pool.length === 0) return null;

    const sorted = [...pool].sort(
      (a, b) => Math.abs(a.bpm - current.bpm) - Math.abs(b.bpm - current.bpm)
    );
    const closestDelta = Math.abs((sorted[0]?.bpm ?? current.bpm) - current.bpm);
    const top = sorted.filter((kit) => Math.abs(kit.bpm - current.bpm) <= closestDelta + 1.5);
    const roll = deterministicRoll(
      `${current.track_id}|${this.kitHistory.join(",")}|${top.map((kit) => kit.track_id).join(",")}`
    );
    return top[Math.min(top.length - 1, Math.floor(roll * top.length))] ?? sorted[0] ?? null;
  }

  /**
   * Select and start decoding the next kit. Safe to call repeatedly; the
   * in-flight promise is shared and the audio thread is never blocked.
   */
  public prepareNextKit(): Promise<boolean> {
    if (this.pendingLoadedKit !== null) return Promise.resolve(true);
    if (this.pendingLoadPromise !== null) return this.pendingLoadPromise;
    const current = this.kit;
    if (current === null) return Promise.resolve(false);

    if (this.nextKit === null) {
      this.nextKit = this.chooseNextKit();
      if (this.nextKit === null) {
        console.warn("[KitPlayer] No compatible next kit found");
        return Promise.resolve(false);
      }
      const bpmDelta = Math.abs(this.nextKit.bpm - current.bpm);
      this.bridgeBlocksForNext = bpmDelta > 10 ? 2 : 1;
      if (this.bridgeBlocksForNext === 2) {
        console.info(
          `[KitPlayer] Next kit BPM differs by ${bpmDelta.toFixed(2)} — extended 2-block bridge`
        );
      }
    }

    const target = this.nextKit;
    this.pendingLoadPromise = (async () => {
      try {
        const loaded = await this.loadKitData(target);
        if (this.nextKit?.track_id !== target.track_id) return false;
        this.pendingLoadedKit = loaded;
        const scoreSuffix = this.lastKitScoreValue !== null
          ? `, taste ${this.lastKitScoreValue.toFixed(4)}`
          : "";
        console.info(
          `[KitPlayer] Next kit preloaded: ${target.track_id} (${target.style}, ${(loaded.fitBpm ?? target.bpm).toFixed(2)} BPM fit, median ${loaded.medianDurationSec.toFixed(2)}s${scoreSuffix})`
        );
        return true;
      } catch (err) {
        console.warn("[KitPlayer] Next kit preload failed; keeping current kit.", err);
        this.failedKitIds.add(target.track_id);
        this.nextKit = null;
        this.pendingLoadedKit = null;
        // Immediately try a different compatible kit so the bridge does not
        // extend forever behind a broken asset.
        setTimeout(() => { void this.prepareNextKit(); }, 0);
        return false;
      } finally {
        this.pendingLoadPromise = null;
      }
    })();
    return this.pendingLoadPromise;
  }

  /** Force a bridge on the next boundary, then switch kits. */
  public forceNextKit(): void {
    if (!this.readyFlag) return;
    this.blocksInSet = KIT_SET_BLOCKS;
    const preparation = this.prepareNextKit();
    this.forcedBridgeRemaining = Math.max(1, this.nextKit ? this.bridgeBlocksForNext : 1);
    console.info(
      `[KitPlayer] NEXT KIT queued — bridge on next boundary (${this.forcedBridgeRemaining} block(s))`
    );
    void preparation;
  }

  /**
   * If a user-steered reticle target is >0.30 from every phrase in the active
   * kit, switch to the closest same-style / ±10 BPM kit using the existing
   * drumless bridge. Called at the next block schedule after a drag.
   */
  public requestSteeringKitChange(
    steering?: KitSteering,
    userSteered?: KitSteeringMask
  ): boolean {
    if (!this.readyFlag || this.kit === null) return false;
    if (this.nextKit !== null || this.forcedBridgeRemaining > 0) return false;

    const target = this.steeringTarget(steering, userSteered);
    if (target === null) return false;
    const currentDistance = this.closestPhraseDistance(target);
    if (currentDistance === null || currentDistance <= 0.3) return false;

    this.lastSteerTargetPoint = target;
    this.lastSteerDistanceValue = currentDistance;

    const current = this.kit;
    const kits = this.manifest?.kits ?? [];
    let bestKit: Kit | null = null;
    let bestDistance = currentDistance;
    for (const kit of kits) {
      if (kit.track_id === current.track_id) continue;
      if (this.kitHistory.includes(kit.track_id)) continue;
      if (kit.style !== current.style) continue;
      if (Math.abs(kit.bpm - current.bpm) > 10) continue;
      let distance = Number.POSITIVE_INFINITY;
      for (const phrase of kit.phrases) {
        const dx = (phrase.x ?? 0.5) - target.x;
        const dy = (phrase.y ?? 0.5) - target.y;
        distance = Math.min(distance, Math.hypot(dx, dy));
      }
      if (distance < bestDistance - 1e-9) {
        bestDistance = distance;
        bestKit = kit;
      }
    }
    if (bestKit === null) return false;

    this.nextKit = bestKit;
    this.bridgeBlocksForNext = 1; // same style and |Δbpm| <= 10
    this.blocksInSet = KIT_SET_BLOCKS;
    this.forcedBridgeRemaining = 1;
    this.pendingLoadedKit = null;
    this.pendingLoadPromise = null;
    this.lastKitScoreValue = null;
    console.info(
      `[KitPlayer] Steering kit change → ${bestKit.track_id} (${bestKit.style}, ${bestKit.bpm.toFixed(2)} BPM) target distance ${currentDistance.toFixed(3)} → ${bestDistance.toFixed(3)}`
    );
    void this.prepareNextKit();
    return true;
  }

  /**
   * Schedule one block, handling set rotation and the drumless bridge.
   * The caller logs the result when it becomes due.
   */
  public scheduleBlock(
    engine: AudioEngine,
    when: number,
    section: string,
    steering?: KitSteering,
    userSteered?: KitSteeringMask
  ): KitBlockResult | null {
    if (!this.readyFlag || this.kit === null || this.phrases.length === 0) return null;
    const wanted = normaliseSection(section);
    const sectionChanged = this.lastSection !== null && this.lastSection !== wanted;
    this.lastSection = wanted;

    // 10th block of the set: start preloading the next kit in the background.
    if (
      this.blocksInSet === KIT_SET_BLOCKS - 3 &&
      this.nextKit === null &&
      this.pendingLoadedKit === null
    ) {
      void this.prepareNextKit();
    }

    // Forced bridge from NEXT KIT (blocksInSet stays at KIT_SET_BLOCKS).
    if (this.forcedBridgeRemaining > 0) {
      const result = this.scheduleBlockInternal(engine, when, wanted, sectionChanged, BRIDGE_STEMS, true, steering, userSteered);
      if (result !== null) this.forcedBridgeRemaining--;
      return result;
    }

    // End of set: switch when the next kit is ready, else extend the bridge.
    if (this.blocksInSet >= KIT_SET_BLOCKS) {
      if (this.pendingLoadedKit !== null) {
        const blocksPlayed = this.blocksInSet;
        const pending = this.pendingLoadedKit;
        this.setActiveLoaded(pending);
        const result = this.scheduleBlockInternal(engine, when, wanted, false, ALL_STEMS, false, steering, userSteered);
        if (result !== null) {
          this.blocksInSet = 1;
          result.setChanged = true;
          result.blocksPlayedInSet = blocksPlayed;
          this.setChanges++;
        }
        return result;
      }
      if (!this.waitingForNextLogged) {
        this.waitingForNextLogged = true;
        console.info("[KitPlayer] Next kit not ready — extending bridge");
      }
      void this.prepareNextKit();
      return this.scheduleBlockInternal(engine, when, wanted, sectionChanged, BRIDGE_STEMS, true, steering, userSteered);
    }

    // Normal drumless bridge window at the end of the set.
    const bridgeThreshold = KIT_SET_BLOCKS - this.bridgeBlocksForNext;
    if (this.blocksInSet >= bridgeThreshold) {
      const result = this.scheduleBlockInternal(engine, when, wanted, sectionChanged, BRIDGE_STEMS, true, steering, userSteered);
      if (result !== null) this.blocksInSet++;
      return result;
    }

    const result = this.scheduleBlockInternal(engine, when, wanted, sectionChanged, ALL_STEMS, false, steering, userSteered);
    if (result !== null) this.blocksInSet++;
    return result;
  }

  private scheduleBlockInternal(
    engine: AudioEngine,
    when: number,
    section: string,
    sectionChanged: boolean,
    stems: readonly StemName[],
    bridge: boolean,
    steering?: KitSteering,
    userSteered?: KitSteeringMask
  ): KitBlockResult | null {
    const kit = this.kit;
    if (kit === null) return null;

    const current = this.lastPhrase;
    const targetEnergy = SECTION_ENERGY[normaliseSection(section)] ?? current?.energy ?? 0.65;
    const phrase = this.selectPhrase(
      section,
      targetEnergy,
      this.recentPhraseIds.slice(-6),
      current,
      sectionChanged,
      steering,
      userSteered
    );
    if (phrase === null) return null;

    const loaded = this.phrases.find((entry) => entry.meta.id === phrase.id);
    if (!loaded) return null;

    const targetSection = this.lastSelectionTarget || normaliseSection(section);
    const energyDelta = current ? Math.abs(phrase.energy - current.energy) : 0;
    const selectionBand = this.lastSelectionBand;

    // The phrase's own cut defines this block's slot. The next block starts
    // exactly where this one ends (no median-length guessing, no resampling).
    const blockDuration = phrase.duration_sec > 0
      ? phrase.duration_sec
      : (64 * 60) / (kit.bpm * 4);
    const joinGaps: string[] = [];
    for (const stem of stems) {
      const voiceKey = `kit_${stem}`;
      if (this.lastBlockStems.has(stem)) {
        const prevEnd = engine.getVoiceEndAt(voiceKey) ?? this.lastStemEndAt[stem] ?? null;
        if (prevEnd !== null) {
          const delta = prevEnd - when;
          if (delta < -0.005) {
            joinGaps.push(`${stem} ${(delta * 1000).toFixed(1)}ms`);
          }
        }
      }

      const buffer = loaded.buffers[stem];
      if (!buffer) continue;
      const source = engine.scheduleSlice(stem, buffer, when, {
        rate: 1.0,
        duration: blockDuration,
        fadeMs: KIT_FADE_MS[stem],
        fadeInMs: KIT_SEAM_FADE_IN_MS[stem],
        fadeOutMs: KIT_SEAM_FADE_OUT_MS[stem],
        gain: 1.0,
        voiceKey,
      });
      this.sources.push(source);
      this.lastStemEndAt[stem] = engine.getVoiceEndAt(voiceKey) ?? when + blockDuration;
    }
    this.lastBlockStems = new Set(stems);
    if (joinGaps.length > 0) {
      console.warn(
        `[KitPlayer] Join gap: ${joinGaps.join(", ")} (prevEnd - nextStart, negative = dead air)`
      );
    }

    this.recentPhraseIds.push(phrase.id);
    if (this.recentPhraseIds.length > 6) {
      this.recentPhraseIds = this.recentPhraseIds.slice(-6);
    }
    this.playedPhraseIds.add(phrase.id);
    this.lastPhrase = phrase;
    this.onPhraseScheduled?.({
      x: phrase.x ?? 0.5,
      y: phrase.y ?? 0.5,
      at: when,
      stem: this.loudestStem(phrase),
      phraseId: phrase.id,
    });

    return {
      kit,
      phrase,
      stems: [...stems],
      bridge,
      setChanged: false,
      blocksPlayedInSet: this.blocksInSet,
      blockDurationSec: blockDuration,
      fitBpm: this.fittedBpm,
      targetSection,
      energyDelta,
      selectionBand,
    };
  }

  /**
   * Arranger-anchored phrase pick:
   * - target section is the section passed in (the arranger's live section);
   * - if the kit has no target-section phrases, use the nearest-energy phrase
   *   from the nearest allowed adjacent section in SECTION_CHAIN;
   * - staged energy band against the previous phrase: 0.15, 0.25, then (on a
   *   section change) 0.35/0.40; unchanged sections repeat rather than jump;
   * - taste matrix weights + quality + stem-presence penalty + proportional
   *   sampling when scores exist, otherwise the rule scorer.
   */
  selectPhrase(
    section: string,
    energy: number,
    recentIds: readonly string[],
    currentPhrase?: KitPhrase | null,
    sectionChanged: boolean = false,
    steering?: KitSteering,
    userSteered?: KitSteeringMask
  ): KitPhrase | null {
    if (this.phrases.length === 0) return null;
    const current = currentPhrase ?? this.lastPhrase;
    const targetSection = normaliseSection(section);
    const target = Number.isFinite(energy) ? energy : (current?.energy ?? 0.65);
    const steerTarget = this.steeringTarget(steering, userSteered);
    this.lastSteerTargetPoint = steerTarget;
    this.lastSteerDistanceValue = steerTarget === null
      ? null
      : this.closestPhraseDistance(steerTarget);
    const recent = (recentIds.length > 0 ? recentIds : this.recentPhraseIds).slice(-6);
    const all = this.phrases.map((phrase) => phrase.meta);
    const durationPool = all.filter((phrase) => phrase.duration_sec >= MIN_PHRASE_DURATION_SEC);
    let basePool = durationPool.length > 0 ? durationPool : all;

    // Intro/outro phrases are unique moments: play them once, in the right
    // part of the set, and never recycle them as a groove loop.
    const curated = basePool.filter((phrase) => {
      if (phrase.once === true && this.playedPhraseIds.has(phrase.id)) return false;
      if (phrase.is_intro === true && this.blocksInSet > 1 && targetSection !== "INTRO") return false;
      if (phrase.is_outro === true && this.blocksInSet < KIT_SET_BLOCKS - 2) return false;
      return true;
    });
    if (curated.length >= 2) basePool = curated;

    // Length coherence: kit phrases are all 4 bars, but beat-grid jitter can
    // make one 2-4% shorter. Playing that outlier changes the block tempo and
    // is heard as a gap/jerk at the join. Keep the tightest cluster around the
    // median when enough phrases remain.
    if (basePool.length > 3) {
      const sortedDur = [...basePool].map((p) => p.duration_sec).sort((a, b) => a - b);
      const medianDur = sortedDur[Math.floor(sortedDur.length / 2)] ?? 0;
      if (medianDur > 1e-6) {
        const tight = basePool.filter(
          (p) => Math.abs(p.duration_sec - medianDur) / medianDur <= 0.015
        );
        if (tight.length >= 3) {
          basePool = tight;
        } else {
          const wide = basePool.filter(
            (p) => Math.abs(p.duration_sec - medianDur) / medianDur <= 0.03
          );
          if (wide.length >= 3) basePool = wide;
        }
      }
    }

    let candidatePool: KitPhrase[];
    let bandLabel: string;
    if (targetSection === "DROP") {
      // A drop is defined by energy, not by the source song's section label.
      // Kits often label a mid-energy phrase "DROP" while their groove phrases
      // are hotter, which produced a build that repeated instead of dropping.
      const dropPool = this.dropPool(basePool, current, recent);
      candidatePool = dropPool.length > 0 ? dropPool : basePool;
      bandLabel = "drop";
    } else {
      const sectionPool = this.sectionPoolForSelection(basePool, targetSection, current);
      const banded = this.energyBandPool(sectionPool, current, sectionChanged, recent);
      candidatePool = banded.pool;
      bandLabel = banded.band >= 0 ? banded.band.toFixed(2) : (current ? "repeat" : "none");
    }

    if (candidatePool.length === 0) {
      if (current) {
        this.logPick("Rule", targetSection, current, current, "repeat", ` • ${current.id} (repeat)`);
        return current;
      }
      candidatePool = basePool;
      bandLabel = "none";
    }

    // Trained taste weights, restricted to the section + energy band.
    const activeKitId = this.kit?.track_id;
    const taste = activeKitId ? this.tasteScores?.kits[activeKitId] : undefined;
    if (taste && current) {
      const currentIndex = taste.phrase_order.indexOf(current.id);
      if (currentIndex >= 0) {
        const row = taste.transition[currentIndex] ?? [];
        const buildWeights = (applyPresence: boolean) => {
          const weighted: Array<{ phrase: KitPhrase; weight: number }> = [];
          let total = 0;
          let unpenalised = 0;
          for (const phrase of candidatePool) {
            const index = taste.phrase_order.indexOf(phrase.id);
            if (index < 0) continue;
            const transition = row[index] ?? 0;
            const quality = taste.quality[index] ?? 0;
            const presence = applyPresence ? this.stemPresenceMultiplier(phrase) : 1;
            const steeringBias = this.steeringMultiplier(phrase, steering, userSteered);
            const harmonic = this.harmonicMultiplier(phrase, current);
            const weight = transition * (0.5 + quality) * presence * steeringBias * harmonic;
            if (!Number.isFinite(weight) || weight <= 0) continue;
            weighted.push({ phrase, weight });
            total += weight;
            if (presence >= 1) unpenalised++;
          }
          return { weighted, total, unpenalised, presenceApplied: applyPresence };
        };

        let filtered = buildWeights(true);
        // If every positive-weight candidate is stem-penalised, use the
        // unpenalised weights so the mix can still choose something.
        if (filtered.unpenalised === 0) {
          filtered = buildWeights(false);
        }
        if (filtered.total > 0) {
          this.lastTransition = taste.phrase_order
            .map((id, index) => ({ id, weight: row[index] ?? 0 }))
            .filter((entry) => Number.isFinite(entry.weight) && entry.weight > 0)
            .sort((a, b) => b.weight - a.weight || a.id.localeCompare(b.id))
            .slice(0, 3);

          let roll = this.nextTasteRandom() * filtered.total;
          let chosen = filtered.weighted[filtered.weighted.length - 1];
          for (const entry of filtered.weighted) {
            roll -= entry.weight;
            if (roll <= 0) {
              chosen = entry;
              break;
            }
          }
          const top3 = this.lastTransition
            .map((entry) => `${entry.id}=${entry.weight.toFixed(4)}`)
            .join(", ");
          const presence = this.stemPresenceMultiplier(chosen.phrase);
          const presenceText = filtered.presenceApplied && presence < 1
            ? ` [stems x${presence}]`
            : "";
          this.logPick(
            "Taste",
            targetSection,
            chosen.phrase,
            current,
            bandLabel,
            ` • ${chosen.phrase.id} weight ${chosen.weight.toFixed(4)} • row top3: ${top3}${presenceText}`
          );
          return chosen.phrase;
        }
      }
    }

    // Rule fallback on the same section + energy-band pool.
    if (current) {
      const realNext = candidatePool.find(
        (phrase) => phrase.start_bar === current.start_bar + 4 && !recent.includes(phrase.id)
      );
      if (realNext) {
        const roll = deterministicRoll(`next|${current.id}|${recent.join(",")}|${targetSection}`);
        if (roll < 0.8) {
          this.logPick(
            "Rule",
            targetSection,
            realNext,
            current,
            bandLabel,
            ` • ${realNext.id} (real next window)`
          );
          return realNext;
        }
      }
    }

    let best: KitPhrase | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const candidate of candidatePool) {
      const sectionScore = normaliseSection(candidate.section) === targetSection ? 2.0 : 0.0;
      const energyDelta = current ? Math.abs(candidate.energy - current.energy) : 0;
      const energyScore = current ? 1.5 * Math.max(0, 1 - energyDelta / 0.3) : 0.0;
      const embeddingScore = current ? cosineSimilarity(candidate.embedding, current.embedding) : 0.0;
      const recencyPenalty = recent.includes(candidate.id) ? -1.0 : 0.0;
      const steeringBias = (this.steeringMultiplier(candidate, steering, userSteered) - 1) * 0.5;
      const tieBreak = deterministicRoll(
        `score|${targetSection}|${current?.id ?? ""}|${recent.join(",")}|${candidate.id}|${target.toFixed(4)}`
      ) * 0.05;
      const score = sectionScore + energyScore + embeddingScore + recencyPenalty + steeringBias + tieBreak;
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }

    if (best !== null) {
      this.logPick("Rule", targetSection, best, current, bandLabel, ` • ${best.id}`);
      return best;
    }
    if (current) {
      this.logPick("Rule", targetSection, current, current, "repeat", ` • ${current.id} (repeat)`);
      return current;
    }
    return candidatePool[0] ?? null;
  }

  /** Prefer the target section; otherwise borrow a stable adjacent section. */
  private sectionPoolForSelection(
    pool: KitPhrase[],
    targetSection: string,
    current: KitPhrase | null
  ): KitPhrase[] {
    const targetPool = pool.filter(
      (phrase) => normaliseSection(phrase.section) === targetSection
    );
    if (targetPool.length > 0) return targetPool;

    const forbidden = new Set<string>();
    if (targetSection === "DROP" || targetSection === "GROOVE_A") {
      forbidden.add("INTRO");
      forbidden.add("BREAKDOWN");
    }

    const targetIndex = SECTION_CHAIN.indexOf(targetSection);
    const ordered: string[] = [];
    if (targetIndex >= 0) {
      for (let distance = 1; distance < SECTION_CHAIN.length; distance++) {
        for (const index of [targetIndex - distance, targetIndex + distance]) {
          const section = SECTION_CHAIN[index];
          if (section !== undefined && !ordered.includes(section)) ordered.push(section);
        }
      }
    }
    for (const section of SECTION_CHAIN) {
      if (section !== targetSection && !ordered.includes(section)) ordered.push(section);
    }
    const allowed = ordered.filter((section) => !forbidden.has(section));

    // Keep the currently selected adjacent section stable while the arranger
    // target itself has no phrases.
    const previousSection = current ? normaliseSection(current.section) : null;
    if (previousSection !== null && allowed.includes(previousSection)) {
      const previousPool = pool.filter(
        (phrase) => normaliseSection(phrase.section) === previousSection
      );
      if (previousPool.length > 0) return previousPool;
    }

    // Nearest-energy phrase from the nearest allowed section tier.
    if (targetIndex >= 0) {
      for (let distance = 1; distance < SECTION_CHAIN.length; distance++) {
        const sections: string[] = [];
        for (const index of [targetIndex - distance, targetIndex + distance]) {
          const section = SECTION_CHAIN[index];
          if (section !== undefined && !forbidden.has(section)) sections.push(section);
        }
        const tier = pool.filter((phrase) => sections.includes(normaliseSection(phrase.section)));
        if (tier.length > 0) {
          if (current) {
            let nearest = tier[0];
            for (const candidate of tier) {
              if (
                Math.abs(candidate.energy - current.energy) <
                Math.abs(nearest.energy - current.energy)
              ) {
                nearest = candidate;
              }
            }
            return pool.filter(
              (phrase) => normaliseSection(phrase.section) === normaliseSection(nearest.section)
            );
          }
          return tier;
        }
      }
    }

    for (const section of allowed) {
      const sectionPool = pool.filter(
        (phrase) => normaliseSection(phrase.section) === section
      );
      if (sectionPool.length > 0) return sectionPool;
    }

    const fallback = pool.filter(
      (phrase) => !forbidden.has(normaliseSection(phrase.section))
    );
    return fallback.length > 0 ? fallback : pool;
  }

  /**
   * Staged energy bands vs the previous phrase. Unchanged sections stop at
   * 0.25 so they never produce an audible energy flap; a section change gets
   * one wider 0.35/0.40 allowance.
   */
  /**
   * Drop candidates: the hottest phrases in the kit, regardless of their
   * source section. A phrase only qualifies if it lifts the energy over the
   * current block; if the whole kit is cooler, the top-energy phrases are
   * used so the arrangement still lands with maximum impact.
   */
  private dropPool(
    pool: KitPhrase[],
    current: KitPhrase | null,
    recent: readonly string[]
  ): KitPhrase[] {
    if (pool.length === 0) return [];
    const currentEnergy = current?.energy ?? 0.6;
    const lastTwo = recent.slice(-2);

    // Already in a hot drop: hold the energy instead of alternating into a
    // breakdown. Repeating the current phrase is better than falling 0.3.
    if (currentEnergy >= 0.80) {
      const hot = pool.filter((phrase) => phrase.energy >= currentEnergy - 0.10);
      const freshHot = hot.filter((phrase) => !lastTwo.includes(phrase.id));
      if (freshHot.length > 0) return freshHot;
      return current ? [current] : hot;
    }

    // Coming out of a build: lift into the hottest phrases the kit has.
    const floor = Math.max(0.70, currentEnergy + 0.03);
    let candidates = pool.filter((phrase) => phrase.energy >= floor);
    if (candidates.length < 2) {
      candidates = [...pool]
        .sort((a, b) => b.energy - a.energy)
        .slice(0, Math.min(4, pool.length));
    }
    const fresh = candidates.filter((phrase) => !lastTwo.includes(phrase.id));
    return fresh.length >= 2 ? fresh : candidates;
  }

  private energyBandPool(
    pool: KitPhrase[],
    current: KitPhrase | null,
    sectionChanged: boolean,
    recent: readonly string[]
  ): { pool: KitPhrase[]; band: number } {
    if (current === null || pool.length === 0) return { pool, band: -1 };
    const bands = sectionChanged ? [0.15, 0.25, 0.35, 0.4] : [0.15, 0.25];
    const lastTwo = recent.slice(-2);
    let fallbackWithin: KitPhrase[] = [];
    let fallbackBand = -1;
    for (const band of bands) {
      const within = pool.filter(
        (phrase) => Math.abs(phrase.energy - current.energy) <= band + 1e-9
      );
      if (within.length === 0) continue;
      const nonRecent = within.filter((phrase) => !lastTwo.includes(phrase.id));
      if (nonRecent.length > 0) {
        return { pool: nonRecent, band };
      }
      if (fallbackWithin.length === 0) {
        fallbackWithin = within;
        fallbackBand = band;
      }
    }
    // No non-recent candidate in any band: repeat a recent phrase rather than
    // jump outside the target section.
    return { pool: fallbackWithin, band: fallbackBand };
  }

  /** Music-theory bias: chroma overlap and bass-root interval. */
  private harmonicMultiplier(phrase: KitPhrase, current: KitPhrase | null): number {
    if (current === null) return 1;
    let mult = 1;
    const a = current.chroma;
    const b = phrase.chroma;
    if (Array.isArray(a) && Array.isArray(b) && a.length === 12 && b.length === 12) {
      let dot = 0;
      let na = 0;
      let nb = 0;
      for (let i = 0; i < 12; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      const sim = dot / Math.sqrt(Math.max(1e-9, na) * Math.max(1e-9, nb));
      if (sim < 0.4) mult *= 0.25;
      else if (sim < 0.6) mult *= 0.6;
      else if (sim > 0.85) mult *= 1.15;
    }
    const ra = current.bass_root;
    const rb = phrase.bass_root;
    if (typeof ra === "number" && typeof rb === "number") {
      const interval = ((rb - ra) % 12 + 12) % 12;
      if (interval === 1 || interval === 2 || interval === 6) mult *= 0.35;
      else if (interval === 0 || interval === 5 || interval === 7) mult *= 1.1;
    }
    return mult;
  }

  private logPick(
    kind: "Taste" | "Rule",
    targetSection: string,
    phrase: KitPhrase,
    current: KitPhrase | null,
    band: string,
    extra: string = ""
  ): void {
    const delta = current ? Math.abs(phrase.energy - current.energy) : 0;
    this.lastSelectionTarget = targetSection;
    this.lastSelectionBand = band;
    this.lastSelectionDelta = delta;
    console.info(
      `[KitPlayer] [${kind} pick] target=${targetSection} got=${normaliseSection(phrase.section)} ΔE=${delta.toFixed(2)} band=${band}${extra}`
    );
  }

  /**
   * Schedule a single phrase using the continuity selector. Kept for callers
   * that do not need the set-rotation result object.
   */
  public schedule(engine: AudioEngine, when: number, section: string, log: boolean = true): boolean {
    const result = this.scheduleBlock(engine, when, section);
    if (result === null) return false;
    if (log) {
      this.logBlock(
        when,
        result.phrase,
        result.stems,
        result.bridge,
        result.targetSection,
        result.energyDelta,
        result.selectionBand
      );
    }
    return true;
  }

  /** Emit the standard one-line block log (the conductor calls this when due). */
  public logBlock(
    when: number,
    phrase: KitPhrase,
    stems: readonly StemName[] = STEM_ORDER,
    bridge: boolean = false,
    targetSection: string = "",
    energyDelta: number = 0,
    selectionBand: string = ""
  ): void {
    const selection = targetSection.length > 0
      ? ` target=${targetSection} ΔE=${energyDelta.toFixed(2)} band=${selectionBand}`
      : "";
    const suffix = bridge ? ` [bridge: ${stems.join("+")}]` : "";
    console.info(
      `[KitPlayer] 4-bar block @ ${when.toFixed(2)}s → ${phrase.id} (${phrase.section}, energy ${phrase.energy.toFixed(2)})${selection}${suffix}`
    );
  }

  /** Emit one set-change line per kit rotation. */
  public logSetChange(kit: Kit, blocksPlayed: number): void {
    console.info(
      `[KitPlayer] Set change → ${kit.title || kit.track_id} (${kit.style}, ${kit.bpm.toFixed(2)} BPM) after ${blocksPlayed} blocks`
    );
  }

  /** Stop everything this player scheduled (Play -> Stop). */
  stop(): void {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Already ended or stopped; safe to ignore.
      }
    }
    this.sources = [];
    this.blocksInSet = 0;
    this.lastPhrase = null;
    this.lastSection = null;
    this.recentPhraseIds = [];
    this.playedPhraseIds.clear();
    this.lastBlockStems.clear();
    this.lastStemEndAt = {};
    this.forcedBridgeRemaining = 0;
    this.waitingForNextLogged = false;
  }
}
