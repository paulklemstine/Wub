// Per-stem remix engine: four independent 4-bar phrase streams sharing one
// bar grid.  Every stem owns a current phrase, a lazy audio cache, a pinned
// style region, and its own Signalsmith Stretch voice.  The engine keeps the
// session key/tempo fixed, transposes Camelot-compatible neighbours, and
// rotates to a new cluster every 12 blocks.
import type { AudioEngine } from "./audio_engine";
import type { StemName } from "./types";
import type { Kit, KitCluster, KitManifest, KitPhrase, TasteScoresManifest } from "./kit_player";
import type { NebulaPoint } from "./visualizer";
import { StretchVoice } from "./stretch_voice";

const STEMS: readonly StemName[] = ["drums", "bass", "other", "vocals"];
const PITCHED: readonly StemName[] = ["bass", "other", "vocals"];
const MAX_SHIFT: Record<StemName, number> = { drums: 0, bass: 5, other: 5, vocals: 3 };
const BLOCKS_PER_SET = 12;
const PIN_BLOCKS = 8; // 32 bars
const RECENT_BLOCKS = 6;
const TEMPO_FIT_LIMIT = 0.02;
const CACHE_LIMIT = 64;
// Phrases whose stem is effectively absent are not useful as independent
// streams; keep a floor per stem so a "bass" phrase actually has bass.
const MIN_RMS_DB: Record<StemName, number> = {
  drums: -28, bass: -28, other: -34, vocals: -42,
};

export interface StemPin {
  x: number;
  y: number;
  untilBlock: number;
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
  phrases: Partial<Record<StemName, KitPhrase>>;
  kits: Partial<Record<StemName, Kit>>;
  clusterId: string;
}

interface StemPoolEntry {
  kit: Kit;
  phrase: KitPhrase;
  stem: StemName;
  keyShift: number;
  rate: number;
  x: number;
  y: number;
  energy: number;
  section: string;
  chroma: number[] | null;
  bassRoot: number | null;
  rmsDb: number;
  gain: number;
}

interface LoadedStemAudio {
  main: AudioBuffer;
  pre: AudioBuffer | null;
  preDur: number;
}

interface CurrentStem {
  entry: StemPoolEntry;
  voice: StretchVoice;
  startedAt: number;
  blockIndex: number;
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

/** Semitone shift from `phraseKey` to `sessionKey`; 0 for relative major/minor. */
export function keyShift(sessionKey: string, phraseKey: string): number | null {
  const a = parseKey(sessionKey);
  const b = parseKey(phraseKey);
  const ca = camelot(sessionKey);
  const cb = camelot(phraseKey);
  if (!a || !b || !ca || !cb) return null;
  // Exact key and relative major/minor share the same key signature.
  if (ca.number === cb.number) return 0;
  const sameMode = ca.letter === cb.letter;
  const wheel = Math.abs(ca.number - cb.number);
  const distance = Math.min(wheel, 12 - wheel);
  if (!sameMode || distance !== 1) return null;
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

export class StemRemixPlayer {
  private readonly engine: AudioEngine;
  private manifest: KitManifest | null = null;
  private baseUrl = "";
  private tasteScores: TasteScoresManifest | null = null;
  private clusters: KitCluster[] = [];
  private kitById = new Map<string, Kit>();
  private entryById = new Map<string, StemPoolEntry>();

  private cluster: KitCluster | null = null;
  private sessionKey = "C minor";
  private sessionBpm = 128;
  private blockSec = 960 / 128;
  private barSec = this.blockSec / 4;

  private pools: Record<StemName, StemPoolEntry[]> = {
    drums: [], bass: [], other: [], vocals: [],
  };
  private current: Partial<Record<StemName, CurrentStem>> = {};
  private pending: Partial<Record<StemName, PendingCommit>> = {};
  private pins: Partial<Record<StemName, StemPin>> = {};
  private blocksSinceChange: Record<StemName, number> = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private recent: string[] = [];
  private playedOnce = new Set<string>();
  private failedKits = new Set<string>();
  private cache = new Map<string, LoadedStemAudio>();
  private stemNext = new Map<string, number>();

  private sessionStartAt = 0;
  private nextBlockAt = 0;
  private blockIndex = 0;
  private bridgeLeft = 0;
  private nextCluster: KitCluster | null = null;
  private lastSection = "";
  private started = false;
  private preparing = false;
  private anchorPrepared: Array<{ stem: StemName; entry: StemPoolEntry; audio: LoadedStemAudio }> = [];
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
    return this.blockSec;
  }

  get blocksInSet(): number {
    return this.blockIndex;
  }

  get nextBlockTime(): number {
    return this.nextBlockAt;
  }

  /** Attach taste scores after the manifest has loaded. */
  setTasteScores(scores: TasteScoresManifest | null): void {
    this.tasteScores = scores;
    this.stemNext.clear();
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
    this.nextBlockAt = 0;
    this.blockIndex = 0;
    this.bridgeLeft = 0;
    this.nextCluster = null;
    this.started = false;
    this.preparing = false;
    this.lastResult = null;
    this.anchorPrepared = [];

    // Preload one real song: all four stems from the same kit so the first
    // 4 bars sound like a record before the remix starts. Scheduling waits
    // for the transport's first schedule() call.
    const anchor = this.chooseAnchor();
    if (anchor) {
      const entries = this.entriesForKit(anchor, "GROOVE_A").length > 0
        ? this.entriesForKit(anchor, "GROOVE_A")
        : this.entriesForKit(anchor);
      const byStem = new Map<StemName, StemPoolEntry>();
      for (const entry of entries) {
        if (!byStem.has(entry.stem)) byStem.set(entry.stem, entry);
      }
      const phrase = entries.find((entry) => STEMS.every((stem) =>
        entries.some((candidate) => candidate.stem === stem && candidate.phrase.id === entry.phrase.id)
      ))?.phrase ?? entries[0]?.phrase ?? anchor.phrases[0];
      const prepared = await Promise.all(STEMS.map(async (stem) => {
        const entry = byStem.get(stem) ?? this.bestEntryForPhrase(phrase, stem);
        if (!entry) return null;
        const loaded = await this.prepareEntry(stem, entry);
        return loaded ? { stem, entry: loaded.entry, audio: loaded.audio } : null;
      }));
      this.anchorPrepared = prepared.filter((entry): entry is { stem: StemName; entry: StemPoolEntry; audio: LoadedStemAudio } => entry !== null);
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
    const kitIds = new Set(cluster.kits);
    for (const kitId of kitIds) {
      const kit = this.kitById.get(kitId);
      if (!kit || this.failedKits.has(kitId)) continue;
      for (const phrase of kit.phrases) {
        const duration = Number(phrase.duration_sec);
        if (!(duration > 0)) continue;
        const rate = duration / this.blockSec;
        if (Math.abs(rate - 1) > TEMPO_FIT_LIMIT) continue;
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
          const entry: StemPoolEntry = {
            kit,
            phrase,
            stem,
            keyShift: shift,
            rate,
            x: xy?.x ?? phrase.x ?? 0.5,
            y: xy?.y ?? phrase.y ?? 0.5,
            energy: Number(phrase.energy) || 0.5,
            section: String(phrase.section ?? "GROOVE_A").toUpperCase(),
            chroma: rotateChroma(phrase.chroma, shift),
            bassRoot: shift === 0 ? (phrase.bass_root ?? null) : null,
            rmsDb,
            gain: stemGain(stem, rmsDb),
          };
          this.pools[stem].push(entry);
          this.entryById.set(`${phrase.id}:${stem}`, entry);
        }
      }
    }
    this.recent = [];
    this.playedOnce.clear();
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
      if (kit.key === this.sessionKey) score += 4;
      if (sections.has("GROOVE_A")) score += 2;
      if (sections.has("DROP")) score += 1.5;
      if (sections.has("BUILDUP")) score += 0.5;
      const fullPhrase = kit.phrases.some((phrase) =>
        STEMS.every((stem) => this.entryById.has(`${phrase.id}:${stem}`))
      );
      if (fullPhrase) score += 2;
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
  schedule(at: number, section: string): StemBlockResult | null {
    if (this.cluster === null) return null;
    if (!this.started) {
      this.started = true;
      this.sessionStartAt = Math.max(at, this.engine.now() + 0.9);
      this.nextBlockAt = this.sessionStartAt + this.blockSec;
      this.blockIndex = 1;
      const result = this.anchorResult(this.sessionStartAt, section);
      this.lastResult = result;
      void this.scheduleAnchor();
      return result;
    }
    if (
      !this.preparing &&
      this.nextBlockAt > 0 &&
      this.nextBlockAt < this.engine.now() + this.blockSec * 0.9
    ) {
      const when = this.nextBlockAt;
      const { result, tasks } = this.beginBlock(when, section);
      this.lastResult = result;
      this.nextBlockAt += this.blockSec;
      this.blockIndex += 1;
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
    for (const prepared of this.anchorPrepared) {
      phrases[prepared.stem] = prepared.entry.phrase;
      kits[prepared.stem] = prepared.entry.kit;
    }
    return {
      at,
      section: String(section || "GROOVE_A").toUpperCase(),
      changed: [...STEMS],
      blocksPlayed: 0,
      bridge: false,
      rotation: false,
      phrases,
      kits,
      clusterId: this.cluster?.id ?? "",
    };
  }

  private async scheduleAnchor(): Promise<void> {
    const at = this.sessionStartAt;
    await Promise.all(this.anchorPrepared.map(async (prepared) => {
      console.info(`[StemRemix] anchor ${prepared.stem} ${prepared.entry.phrase.id}`);
      await this.scheduleFull(prepared.stem, prepared.entry, prepared.audio, at);
    }));
  }

  private beginBlock(at: number, section: string): {
    result: StemBlockResult;
    tasks: Array<{ stem: StemName; entry: StemPoolEntry; audio?: LoadedStemAudio }>;
  } {
    const normalized = String(section || "GROOVE_A").toUpperCase();
    const sectionChanged = normalized !== this.lastSection;
    this.lastSection = normalized;

    // Cluster rotation: 2-block drumless bridge at blocks 10-11, switch at 12.
    let bridge = false;
    let rotation = false;
    const blocksPlayed = this.blockIndex;
    if (this.blockIndex === BLOCKS_PER_SET - 2 && this.nextCluster === null) {
      this.nextCluster = this.chooseNeighbourCluster();
      if (this.nextCluster) {
        this.bridgeLeft = 2;
        console.info(`[StemRemix] Bridge → ${this.nextCluster.id} in 2 blocks`);
      }
    }
    if (this.blockIndex >= BLOCKS_PER_SET - 2 && this.bridgeLeft > 0) {
      bridge = true;
      this.bridgeLeft -= 1;
    }
    if (this.blockIndex >= BLOCKS_PER_SET && this.nextCluster) {
      const next = this.nextCluster;
      this.nextCluster = null;
      this.applyCluster(next);
      this.blockIndex = 0;
      this.sessionStartAt = at;
      this.nextBlockAt = at;
      rotation = true;
      bridge = false;
      console.info(`[StemRemix] Cluster → ${next.id} • ${this.sessionBpm.toFixed(2)} BPM • ${this.sessionKey}`);
    }

    const changed: StemName[] = [];
    const phrases: Partial<Record<StemName, KitPhrase>> = {};
    const kits: Partial<Record<StemName, Kit>> = {};
    const tasks: Array<{ stem: StemName; entry: StemPoolEntry; audio?: LoadedStemAudio }> = [];

    if (bridge && !rotation) {
      for (const stem of ["drums", "bass"] as const) {
        this.current[stem]?.voice.fadeOut(at, 0.25);
      }
    }

    const advance = new Set<StemName>();
    if (this.blockIndex === 0 || rotation || sectionChanged) {
      for (const stem of STEMS) advance.add(stem);
    } else {
      for (const stem of STEMS) {
        if (this.pins[stem]) advance.add(stem);
      }
      const auto = STEMS.filter((stem) => !advance.has(stem))
        .sort((a, b) => this.blocksSinceChange[b] - this.blocksSinceChange[a]);
      const need = Math.max(1, Math.min(2, auto.length));
      for (const stem of auto.slice(0, need)) advance.add(stem);
    }

    const order: StemName[] = bridge && !rotation
      ? ["other", "vocals"]
      : ["drums", "bass", "other", "vocals"];
    for (const stem of order) {
      if (!advance.has(stem)) {
        const current = this.current[stem];
        if (current) {
          phrases[stem] = current.entry.phrase;
          kits[stem] = current.entry.kit;
          const audio = this.cache.get(this.cacheKey(current.entry));
          if (audio) tasks.push({ stem, entry: current.entry, audio });
        }
        this.blocksSinceChange[stem] += 1;
        continue;
      }
      const entry = this.select(stem, at, normalized, sectionChanged);
      if (!entry) {
        const current = this.current[stem];
        if (current) {
          phrases[stem] = current.entry.phrase;
          kits[stem] = current.entry.kit;
          const audio = this.cache.get(this.cacheKey(current.entry));
          if (audio) tasks.push({ stem, entry: current.entry, audio });
        }
        continue;
      }
      phrases[stem] = entry.phrase;
      kits[stem] = entry.kit;
      tasks.push({ stem, entry });
      changed.push(stem);
      this.blocksSinceChange[stem] = 0;
      this.recent.push(entry.phrase.id);
      if (entry.phrase.once) this.playedOnce.add(entry.phrase.id);
    }
    if (this.recent.length > RECENT_BLOCKS * 5) {
      this.recent = this.recent.slice(-RECENT_BLOCKS * 5);
    }
    for (const stem of STEMS) {
      const pin = this.pins[stem];
      if (pin && this.blockIndex >= pin.untilBlock) {
        delete this.pins[stem];
        this.onStatus?.(`${stem} released back to auto-mix`);
      }
    }

    const result: StemBlockResult = {
      at,
      section: normalized,
      changed,
      blocksPlayed,
      bridge,
      rotation,
      phrases,
      kits,
      clusterId: this.cluster?.id ?? "",
    };
    this.onBlock?.(result);
    return { result, tasks };
  }

  private async finishBlock(
    tasks: Array<{ stem: StemName; entry: StemPoolEntry; audio?: LoadedStemAudio }>,
    at: number,
  ): Promise<void> {
    await Promise.all(tasks.map(async (task) => {
      let audio = task.audio;
      if (!audio) {
        const prepared = await this.prepareEntry(task.stem, task.entry);
        if (!prepared) {
          this.failedKits.add(task.entry.kit.track_id);
          const current = this.current[task.stem];
          if (current) {
            const fallback = this.cache.get(this.cacheKey(current.entry));
            if (fallback) await this.scheduleFull(task.stem, current.entry, fallback, at);
          }
          return;
        }
        audio = prepared.audio;
      }
      await this.scheduleFull(task.stem, task.entry, audio, at);
    }));
  }

  private select(stem: StemName, at: number, section: string, sectionChanged: boolean): StemPoolEntry | null {
    const pool = this.pools[stem];
    if (pool.length === 0) return null;
    const current = this.current[stem]?.entry ?? null;
    const pin = this.pins[stem] ?? null;
    const anchor = this.current.bass?.entry ?? null;
    const chord = anchor?.chroma ?? null;
    const recentSet = new Set(this.recent);
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
  ): Promise<{ entry: StemPoolEntry; audio: LoadedStemAudio } | null> {
    try {
      const audio = await this.loadAudio(entry);
      return { entry, audio };
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

  private async scheduleFull(
    stem: StemName,
    entry: StemPoolEntry,
    audio: LoadedStemAudio,
    at: number,
  ): Promise<void> {
    try {
      const ctx = this.engine.ensureCtx();
      const voice = await StretchVoice.create(ctx, this.engine.getStemInput(stem), {
        pan: stemPan(stem, entry.phrase.id),
      });
      const preDur = await voice.load(audio.main, audio.pre);
      const startAt = at - preDur / Math.max(0.5, entry.rate);
      voice.schedule({
        when: startAt,
        inputSec: 0,
        rate: entry.rate,
        semitones: entry.keyShift,
        formantCompensation: stem === "vocals",
        fadeInSec: 0.006,
        gain: entry.gain,
      });
      voice.fadeOut(at + this.blockSec, 0.09);
      const previous = this.current[stem];
      if (previous) previous.voice.fadeOut(startAt, preDur > 0 ? preDur / entry.rate : 0.06);
      this.current[stem] = {
        entry,
        voice,
        startedAt: at,
        blockIndex: this.blockIndex,
      };
      console.info(
        `[StemRemix] ${stem} ← ${entry.phrase.id} @ ${at.toFixed(2)}s ` +
        `(start ${startAt.toFixed(2)}s, rate ${entry.rate.toFixed(4)}, shift ${entry.keyShift}, pre ${preDur.toFixed(3)}s)`
      );
    } catch (err) {
      console.warn(`[StemRemix] ${stem} schedule failed`, err);
      this.failedKits.add(entry.kit.track_id);
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
    const pin: StemPin = { x: pinX, y: pinY, untilBlock: this.blockIndex + PIN_BLOCKS };
    this.pins[stem] = pin;
    if (!nearest) return pin;
    if (Math.hypot(nearest.x - cx, nearest.y - cy) > 0.15) {
      this.onStatus?.(`${stem}: target outside the ${stem} cloud, snapped to the nearest legal phrase`);
    }
    const distance = current ? Math.hypot(nearest.x - current.x, nearest.y - current.y) : 1;
    if (distance > 0.06) {
      const now = this.engine.now();
      const start = this.sessionStartAt > 0 ? this.sessionStartAt : now + 0.5;
      const bar = Math.ceil((now - start) / this.barSec + 0.15);
      const at = start + Math.max(0, bar) * this.barSec;
      const partialBar = ((Math.round((at - start) / this.barSec) % 4) + 4) % 4;
      this.pending[stem] = { entry: nearest, at, partialBar, requestedAt: now };
      void this.prepareCommit(stem, this.pending[stem]!);
      this.onStatus?.(`${stem} → ${nearest.phrase.section} (${nearest.kit.style}) at next bar`);
    } else {
      this.onStatus?.(`${stem} pinned in place for 32 bars`);
    }
    return pin;
  }

  release(stem: StemName): void {
    delete this.pins[stem];
    delete this.pending[stem];
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

  private async prepareCommit(stem: StemName, pending: PendingCommit): Promise<void> {
    const prepared = await this.prepareEntry(stem, pending.entry);
    if (!prepared) {
      this.failedKits.add(pending.entry.kit.track_id);
      delete this.pending[stem];
      this.onStatus?.(`${stem}: target phrase failed to load; keeping current`);
      return;
    }
    // If the fetch missed the bar, wait for the following bar; the tick loop
    // will commit as soon as the audio is ready.
    const now = this.engine.now();
    if (now > pending.at - 0.08) {
      const barsLate = Math.ceil((now - pending.at) / this.barSec + 0.1);
      pending.at += Math.max(1, barsLate) * this.barSec;
      pending.partialBar = ((pending.partialBar + Math.max(1, barsLate)) % 4 + 4) % 4;
    }
    pending.entry = prepared.entry;
    this.pending[stem] = pending;
    const ctx = this.engine.ensureCtx();
    const voice = await StretchVoice.create(ctx, this.engine.getStemInput(stem), {
      pan: stemPan(stem, pending.entry.phrase.id),
    });
    const preDur = await voice.load(prepared.audio.main, prepared.audio.pre);
    const inputSec = preDur + pending.partialBar * (prepared.audio.main.duration / 4);
    voice.schedule({
      when: pending.at,
      inputSec: Math.min(inputSec, prepared.audio.main.duration - 0.05),
      rate: pending.entry.rate,
      semitones: pending.entry.keyShift,
      formantCompensation: stem === "vocals",
      fadeInSec: 0.02,
      gain: pending.entry.gain,
    });
    const nextBlock = this.sessionStartAt + (Math.floor((pending.at - this.sessionStartAt) / this.blockSec) + 1) * this.blockSec;
    voice.fadeOut(nextBlock, 0.05);
    const previous = this.current[stem];
    if (previous) previous.voice.fadeOut(pending.at, 0.05);
    this.current[stem] = {
      entry: pending.entry,
      voice,
      startedAt: pending.at - pending.partialBar * this.barSec,
      blockIndex: this.blockIndex,
    };
    delete this.pending[stem];
    this.recent.push(pending.entry.phrase.id);
  }

  /** Called every scheduler tick; release pins / finish a late commit. */
  tick(now: number): void {
    for (const stem of STEMS) {
      const pin = this.pins[stem];
      if (pin && this.blockIndex >= pin.untilBlock) {
        delete this.pins[stem];
        this.onStatus?.(`${stem} released back to auto-mix`);
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
        buffering: this.pending[stem] !== undefined,
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
    const current = camelot(this.cluster.key);
    const candidates = this.clusters
      .filter((c) => c.id !== this.cluster!.id)
      .filter((c) => Math.abs(c.bpm - this.cluster!.bpm) <= 3)
      .map((c) => {
        const other = camelot(c.key);
        if (!current || !other) return { cluster: c, distance: 99 };
        const wheel = Math.abs(current.number - other.number);
        const distance = Math.min(wheel, 12 - wheel) + (current.letter === other.letter ? 0 : 0.5);
        return { cluster: c, distance };
      })
      .filter((c) => c.distance <= 1.5)
      .sort((a, b) => a.distance - b.distance);
    return candidates[0]?.cluster ?? null;
  }

  stop(): void {
    for (const stem of STEMS) {
      const current = this.current[stem];
      if (current) current.voice.fadeOut(this.engine.now(), 0.05);
    }
    this.current = {};
    this.pending = {};
    this.pins = {};
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
