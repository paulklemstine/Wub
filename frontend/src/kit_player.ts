// M1 Kit Player: plays locked, real 4-bar stem phrases straight from the
// export kit -- no resampling, no slice resynthesis, no pitch shifting.
//
// Each kit contains up to 8 phrases x 4 stems. A phrase is scheduled as four
// independent AudioBufferSources at rate 1.0, so Opus's native decode rate is
// never reinterpreted. Block timing comes from the fitted kit tempo
// (`960 / median phrase duration`), not from sample counts.
import type { StemName } from "./types";
import type { AudioEngine } from "./audio_engine";

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
}

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

// Section energy targets mirror ClientSongArranger so phrase choice follows the
// same arrangement arc the sequencer is currently in.
const SECTION_ENERGY: Record<string, number> = {
  INTRO: 0.25,
  BUILDUP: 0.65,
  DROP: 0.90,
  GROOVE_A: 0.83,
  BREAKDOWN: 0.20,
};

// Adjacent sections come from ClientSongArranger's transition weights.
const ADJACENT_SECTIONS: Record<string, readonly string[]> = {
  INTRO: ["GROOVE_A", "BUILDUP"],
  GROOVE_A: ["BUILDUP", "BREAKDOWN", "DROP"],
  BUILDUP: ["DROP"],
  DROP: ["BREAKDOWN", "GROOVE_A", "BUILDUP"],
  BREAKDOWN: ["BUILDUP", "GROOVE_A"],
};

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
  private sources: AudioBufferSourceNode[] = [];
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

  get compatibleKits(): Kit[] {
    return [...this.compatibility];
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
            !this.kitHistory.includes(entry.track_id)
        )
        .map((entry) => ({
          entry,
          kit: kits.find((kit) => kit.track_id === entry.track_id),
        }))
        .filter((row): row is { entry: TasteNextKit; kit: Kit } => row.kit !== undefined);
      if (eligible.length > 0) {
        eligible.sort(
          (a, b) => b.entry.score - a.entry.score || a.entry.track_id.localeCompare(b.entry.track_id)
        );
        const best = eligible[0];
        this.lastKitScoreValue = best.entry.score;
        console.info(
          `[KitPlayer] Taste next kit: ${best.entry.track_id} (score ${best.entry.score.toFixed(4)}, ${best.entry.style}, ${best.entry.bpm.toFixed(2)} BPM)`
        );
        return best.kit;
      }
    }

    // Fallback: existing same-style/Camelot/BPM compatibility pool.
    this.lastKitScoreValue = null;
    const pool = this.compatibility.filter((kit) => !this.kitHistory.includes(kit.track_id));
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
        this.nextKit = null;
        this.pendingLoadedKit = null;
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
   * Schedule one block, handling set rotation and the drumless bridge.
   * The caller logs the result when it becomes due.
   */
  public scheduleBlock(engine: AudioEngine, when: number, section: string): KitBlockResult | null {
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
      const result = this.scheduleBlockInternal(engine, when, wanted, sectionChanged, BRIDGE_STEMS, true);
      if (result !== null) this.forcedBridgeRemaining--;
      return result;
    }

    // End of set: switch when the next kit is ready, else extend the bridge.
    if (this.blocksInSet >= KIT_SET_BLOCKS) {
      if (this.pendingLoadedKit !== null) {
        const blocksPlayed = this.blocksInSet;
        const pending = this.pendingLoadedKit;
        this.setActiveLoaded(pending);
        const result = this.scheduleBlockInternal(engine, when, wanted, false, ALL_STEMS, false);
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
      return this.scheduleBlockInternal(engine, when, wanted, sectionChanged, BRIDGE_STEMS, true);
    }

    // Normal drumless bridge window at the end of the set.
    const bridgeThreshold = KIT_SET_BLOCKS - this.bridgeBlocksForNext;
    if (this.blocksInSet >= bridgeThreshold) {
      const result = this.scheduleBlockInternal(engine, when, wanted, sectionChanged, BRIDGE_STEMS, true);
      if (result !== null) this.blocksInSet++;
      return result;
    }

    const result = this.scheduleBlockInternal(engine, when, wanted, sectionChanged, ALL_STEMS, false);
    if (result !== null) this.blocksInSet++;
    return result;
  }

  private scheduleBlockInternal(
    engine: AudioEngine,
    when: number,
    section: string,
    sectionChanged: boolean,
    stems: readonly StemName[],
    bridge: boolean
  ): KitBlockResult | null {
    const kit = this.kit;
    if (kit === null) return null;

    const current = this.lastPhrase;
    const targetEnergy = SECTION_ENERGY[normaliseSection(section)] ?? current?.energy ?? 0.65;
    const phrase = this.selectPhrase(
      section,
      targetEnergy,
      this.recentPhraseIds.slice(-3),
      current,
      sectionChanged
    );
    if (phrase === null) return null;

    const loaded = this.phrases.find((entry) => entry.meta.id === phrase.id);
    if (!loaded) return null;

    const blockDuration = this.medianDurationSec > 0
      ? this.medianDurationSec
      : (64 * 60) / (kit.bpm * 4);
    for (const stem of stems) {
      const buffer = loaded.buffers[stem];
      if (!buffer) continue;
      const source = engine.scheduleSlice(stem, buffer, when, {
        rate: 1.0,
        duration: blockDuration,
        fadeMs: KIT_FADE_MS[stem],
        gain: 1.0,
        voiceKey: `kit_${stem}`,
      });
      this.sources.push(source);
    }

    this.recentPhraseIds.push(phrase.id);
    if (this.recentPhraseIds.length > 3) {
      this.recentPhraseIds = this.recentPhraseIds.slice(-3);
    }
    this.lastPhrase = phrase;

    return {
      kit,
      phrase,
      stems: [...stems],
      bridge,
      setChanged: false,
      blocksPlayedInSet: this.blocksInSet,
      blockDurationSec: blockDuration,
      fitBpm: this.fittedBpm,
    };
  }

  /**
   * Continuity-first phrase pick:
   * - hard energy band vs the current phrase (0.15, then 0.3)
   * - stay in the current section unless the arranger section changed
   * - 80% chance to use the real `start_bar + 4` next window when available
   * - section/energy/embedding score with a last-3 recency penalty
   */
  selectPhrase(
    section: string,
    energy: number,
    recentIds: readonly string[],
    currentPhrase?: KitPhrase | null,
    sectionChanged: boolean = false
  ): KitPhrase | null {
    if (this.phrases.length === 0) return null;
    const current = currentPhrase ?? this.lastPhrase;
    const target = Number.isFinite(energy) ? energy : (current?.energy ?? 0.65);
    const recent = (recentIds.length > 0 ? recentIds : this.recentPhraseIds).slice(-3);
    const all = this.phrases.map((phrase) => phrase.meta);

    // Holes: never pick a phrase materially shorter than the fitted block.
    const minDuration = this.medianDurationSec > 0 ? this.medianDurationSec * 0.9 : 0;
    const activeKitId = this.kit?.track_id;
    const taste = activeKitId ? this.tasteScores?.kits[activeKitId] : undefined;
    if (taste && current) {
      const currentIndex = taste.phrase_order.indexOf(current.id);
      if (currentIndex >= 0) {
        const lastTwo = recent.slice(-2);
        const durationPool = all.filter(
          (phrase) =>
            (minDuration <= 0 || phrase.duration_sec >= minDuration) &&
            !lastTwo.includes(phrase.id)
        );
        const row = taste.transition[currentIndex] ?? [];
        const buildWeights = (applyEnergyGuard: boolean) => {
          const weighted: Array<{ phrase: KitPhrase; weight: number }> = [];
          let total = 0;
          for (const phrase of durationPool) {
            if (applyEnergyGuard && Math.abs(phrase.energy - current.energy) > 0.35) continue;
            const index = taste.phrase_order.indexOf(phrase.id);
            if (index < 0) continue;
            const transition = row[index] ?? 0;
            const quality = taste.quality[index] ?? 0;
            const weight = transition * (0.5 + quality);
            if (!Number.isFinite(weight) || weight <= 0) continue;
            weighted.push({ phrase, weight });
            total += weight;
          }
          return { weighted, total };
        };

        let filtered = buildWeights(true);
        if (filtered.total <= 0) {
          // Wrong-style / sparse-row guard: relax the 0.35 energy guard once.
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
          console.info(
            `[KitPlayer] Taste pick: ${chosen.phrase.id} weight ${chosen.weight.toFixed(4)} • row top3: ${top3}`
          );
          return chosen.phrase;
        }
      }
    }

    let pool = minDuration > 0
      ? all.filter((phrase) => phrase.duration_sec >= minDuration)
      : all;
    if (pool.length === 0) pool = all;

    const wanted = normaliseSection(section);
    const preferred = current ? normaliseSection(current.section) : wanted;
    if (!sectionChanged) {
      const sameSection = pool.filter((phrase) => normaliseSection(phrase.section) === preferred);
      if (sameSection.length > 0) pool = sameSection;
    } else {
      let nextSection = pool.filter((phrase) => normaliseSection(phrase.section) === wanted);
      if (nextSection.length === 0) {
        const adjacent = new Set(ADJACENT_SECTIONS[wanted] ?? []);
        nextSection = pool.filter((phrase) => adjacent.has(normaliseSection(phrase.section)));
      }
      if (nextSection.length > 0) pool = nextSection;
    }

    // Hard continuity band: 0.15 normally, 0.3 only when 0.15 has no match.
    if (current) {
      const within015 = pool.filter(
        (phrase) => Math.abs(phrase.energy - current.energy) <= 0.15
      );
      if (within015.length > 0) {
        pool = within015;
      } else {
        const within03 = pool.filter(
          (phrase) => Math.abs(phrase.energy - current.energy) <= 0.3
        );
        if (within03.length > 0) pool = within03;
      }
    }

    // Real song order: if the actual next 4-bar window exists, use it often.
    if (current) {
      const realNext = pool.find(
        (phrase) => phrase.start_bar === current.start_bar + 4 && !recent.includes(phrase.id)
      );
      if (realNext) {
        const roll = deterministicRoll(`next|${current.id}|${recent.join(",")}|${wanted}`);
        if (roll < 0.8) return realNext;
      }
    }

    let best: KitPhrase | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const candidate of pool) {
      const sectionScore = normaliseSection(candidate.section) === wanted ? 2.0 : 0.0;
      const energyDelta = current ? Math.abs(candidate.energy - current.energy) : 0;
      const energyScore = current ? 1.5 * Math.max(0, 1 - energyDelta / 0.3) : 0.0;
      const embeddingScore = current ? cosineSimilarity(candidate.embedding, current.embedding) : 0.0;
      const recencyPenalty = recent.includes(candidate.id) ? -1.0 : 0.0;
      const tieBreak = deterministicRoll(
        `score|${wanted}|${current?.id ?? ""}|${recent.join(",")}|${candidate.id}|${target.toFixed(4)}`
      ) * 0.05;
      const score = sectionScore + energyScore + embeddingScore + recencyPenalty + tieBreak;
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    return best;
  }

  /**
   * Schedule a single phrase using the continuity selector. Kept for callers
   * that do not need the set-rotation result object.
   */
  public schedule(engine: AudioEngine, when: number, section: string, log: boolean = true): boolean {
    const result = this.scheduleBlock(engine, when, section);
    if (result === null) return false;
    if (log) this.logBlock(when, result.phrase, result.stems, result.bridge);
    return true;
  }

  /** Emit the standard one-line block log (the conductor calls this when due). */
  public logBlock(
    when: number,
    phrase: KitPhrase,
    stems: readonly StemName[] = STEM_ORDER,
    bridge: boolean = false
  ): void {
    const suffix = bridge ? ` [bridge: ${stems.join("+")}]` : "";
    console.info(
      `[KitPlayer] 4-bar block @ ${when.toFixed(2)}s → ${phrase.id} (${phrase.section}, energy ${phrase.energy.toFixed(2)})${suffix}`
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
    this.forcedBridgeRemaining = 0;
    this.waitingForNextLogged = false;
  }
}
