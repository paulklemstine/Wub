// M1 Kit Player: plays locked, real 4-bar stem phrases straight from the
// export kit -- no resampling, no slice resynthesis, no pitch shifting.
//
// Each kit contains 8 phrases x 4 stems. A phrase is scheduled as four
// independent AudioBufferSources at rate 1.0, so Opus's native 48 kHz decode
// rate is never reinterpreted as 44.1 kHz. Block timing comes from the kit BPM
// (`64 * 60 / (bpm * 4)` seconds per 4 bars), not from sample counts.
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

interface LoadedKitPhrase {
  meta: KitPhrase;
  buffers: Partial<Record<StemName, AudioBuffer>>;
}

const STEM_ORDER: readonly StemName[] = ["drums", "bass", "other", "vocals"];

// Section energy targets mirror ClientSongArranger so phrase choice follows the
// same arrangement arc the sequencer is currently in.
const SECTION_ENERGY: Record<string, number> = {
  INTRO: 0.25,
  BUILDUP: 0.65,
  DROP: 0.90,
  GROOVE_A: 0.83,
  BREAKDOWN: 0.20,
};

// Adjacent sections come from ClientSongArranger's transition weights: if the
// current section has no phrase in a kit, these are the musically safe
// neighbours to borrow a phrase from.
const ADJACENT_SECTIONS: Record<string, readonly string[]> = {
  INTRO: ["GROOVE_A", "BUILDUP"],
  GROOVE_A: ["BUILDUP", "BREAKDOWN", "DROP"],
  BUILDUP: ["DROP"],
  DROP: ["BREAKDOWN", "GROOVE_A", "BUILDUP"],
  BREAKDOWN: ["BUILDUP", "GROOVE_A"],
};

function normaliseSection(section: string): string {
  return section.trim().toUpperCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Deterministic FNV-1a hash -> mulberry32 roll. Selection is reproducible for
// the same (section, energy target, recent ids) without touching Math.random.
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

  constructor(engine: AudioEngine) {
    this.engine = engine;
  }

  get ready(): boolean {
    return this.readyFlag;
  }

  get currentKit(): Kit | null {
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

  /** Decode all 8 phrases x 4 stems for one kit (32 AudioBuffers). */
  async loadKit(kit: Kit): Promise<void> {
    if (this.manifest === null) {
      throw new Error("KitPlayer.loadKit called before loadManifest");
    }
    if (this.readyFlag && this.kit?.track_id === kit.track_id) return;
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

    this.kit = kit;
    this.phrases = phrases;

    // The exported cut lengths, not the manifest BPM, are the real 4-bar
    // length. Fit the session tempo to the median phrase so block joins close
    // without resampling: 4 bars of 16ths = 64 * 60 / (bpm * 4) = 960 / bpm.
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
    this.medianDurationSec = median;
    this.fittedBpm = median > 0 ? 960 / median : null;
    if (this.fittedBpm !== null) {
      console.info(
        `[KitPlayer] Tempo fit: manifest ${kit.bpm.toFixed(2)} → ${this.fittedBpm.toFixed(2)} BPM (median phrase ${median.toFixed(2)}s)`
      );
    }

    this.recentPhraseIds = [];
    this.readyFlag = true;
    console.info(
      `[KitPlayer] Kit ready: ${kit.track_id} • ${kit.bpm.toFixed(2)} BPM • ${kit.key} • ${phrases.length} phrases`
    );
  }

  /**
   * Rule-based phrase pick: current section first, then adjacent sections,
   * then any section. Within that pool the energy nearest the target wins;
   * deterministic RNG only breaks exact energy ties. The last two scheduled ids
   * are excluded so a phrase never repeats back-to-back.
   */
  selectPhrase(section: string, energy: number, recentIds: readonly string[]): KitPhrase | null {
    if (this.phrases.length === 0) return null;
    const target = Number.isFinite(energy) ? energy : 0.65;
    const blocked = new Set(recentIds.slice(-2));
    const all = this.phrases.map((phrase) => phrase.meta);

    let pool = all.filter((phrase) => !blocked.has(phrase.id));
    if (pool.length === 0) pool = all;

    const wanted = normaliseSection(section);
    let sectionPool = pool.filter((phrase) => normaliseSection(phrase.section) === wanted);
    if (sectionPool.length === 0) {
      const adjacent = new Set(ADJACENT_SECTIONS[wanted] ?? []);
      sectionPool = pool.filter((phrase) => adjacent.has(normaliseSection(phrase.section)));
    }
    if (sectionPool.length === 0) sectionPool = pool;

    // Keep the 4-bar joins tight: prefer phrases whose cut length is within
    // ±15 ms of the median, then choose the energy closest to the target.
    if (this.medianDurationSec > 0) {
      const nearMedian = sectionPool.filter(
        (phrase) => Math.abs(phrase.duration_sec - this.medianDurationSec) <= 0.015
      );
      if (nearMedian.length > 0) sectionPool = nearMedian;
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    for (const phrase of sectionPool) {
      const distance = Math.abs(phrase.energy - target);
      if (distance < bestDistance) bestDistance = distance;
    }
    const tied = sectionPool.filter((phrase) => Math.abs(Math.abs(phrase.energy - target) - bestDistance) < 1e-9);
    const roll = deterministicRoll(`${wanted}|${target.toFixed(4)}|${recentIds.slice(-2).join(",")}|${tied.length}`);
    const index = Math.min(tied.length - 1, Math.floor(roll * tied.length));
    return tied[index] ?? null;
  }

  /**
   * Schedule the four stems of one 4-bar phrase at `when`. Every call creates a
   * fresh AudioBufferSource starting at offset 0 (a rewind), so the phrase can
   * never drift or resume mid-buffer. Playback rate is always 1.0 and detune is
   * left at 0: the kit is already at the locked session tempo.
   */
  schedule(engine: AudioEngine, when: number, section: string, log: boolean = true): boolean {
    if (!this.readyFlag || this.kit === null || this.phrases.length === 0) return false;

    const targetEnergy = SECTION_ENERGY[normaliseSection(section)] ?? 0.65;
    const chosen = this.selectPhrase(section, targetEnergy, this.recentPhraseIds);
    if (chosen === null) return false;
    const loaded = this.phrases.find((phrase) => phrase.meta.id === chosen.id);
    if (!loaded) return false;

    // Use the exact median cut length as the block duration; scheduleSlice
    // clamps and fades the tiny tail difference instead of resampling.
    const blockDuration = this.medianDurationSec > 0
      ? this.medianDurationSec
      : (64 * 60) / (this.kit.bpm * 4);
    for (const stem of STEM_ORDER) {
      const buffer = loaded.buffers[stem];
      if (!buffer) continue;
      const source = engine.scheduleSlice(stem, buffer, when, {
        rate: 1.0,
        duration: blockDuration,
        fadeMs: 12,
        gain: 1.0,
        voiceKey: `kit_${stem}`,
      });
      this.sources.push(source);
    }

    this.recentPhraseIds.push(loaded.meta.id);
    if (this.recentPhraseIds.length > 8) {
      this.recentPhraseIds = this.recentPhraseIds.slice(-8);
    }
    this.lastPhrase = loaded.meta;
    if (log) this.logBlock(when, loaded.meta);
    return true;
  }

  /** Emit the standard one-line block log (the conductor calls this when due). */
  public logBlock(when: number, phrase: KitPhrase): void {
    console.info(
      `[KitPlayer] 4-bar block @ ${when.toFixed(2)}s → ${phrase.id} (${phrase.section}, energy ${phrase.energy.toFixed(2)})`
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
  }
}
