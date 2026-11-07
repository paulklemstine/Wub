/** Kit audio decode + concat worker: the preload thread for the kit engine.
 *
 * Why this exists: every 8/16-bar prefetch used to do multi-MB synchronous
 * PCM work on the main thread (mixToMono + concat + structured clones for
 * four stems at once, plus the synchronous joint select beside it).  On a
 * loaded machine that burst starves the audio render thread into
 * all-stem digital-zero dropouts (the silence watchdog kept firing during
 * 8/16-bar prefetches in v0.2.235).  Decoding and concatenating here keeps
 * that cost off the main thread entirely; the main thread only pays for the
 * worklet's addBuffers clone and the scheduling calls.
 *
 * The worker owns a 24-entry LRU of decoded mono PCM (same key space as the
 * main-thread fallback cache: phraseId:stem).  Cache misses are fetched and
 * decoded with an OfflineAudioContext at the engine's sample rate, so the
 * concatenated run is a single transferable postMessage back to the main
 * thread (no copy on return).
 *
 * Protocol (main thread):
 *   -> { type: "init", sampleRate }                              (once)
 *   -> { type: "load", id, items: [{ key, url, preUrl }] }      (one run)
 *   <- { type: "run", id, samples, preDur, mainDurations, sampleRate }
 *   <- { type: "error", id, message }
 * Any item failure fails the whole run (mirrors the main-thread loader's
 * "a missing phrase fails the run" semantics).
 */
import { concatRun, mixToMono } from "./kit_pcm";

interface PoolItem {
  key: string;
  url: string;
  preUrl: string | null;
}

interface CacheEntry {
  mono: Float32Array;
  pre: Float32Array | null;
}

const CACHE_LIMIT = 24;
const cache = new Map<string, CacheEntry>();
let sampleRate = 44100;

interface Host {
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
}

function host(): Host {
  return self as unknown as Host;
}

function cacheGet(key: string): CacheEntry | undefined {
  const entry = cache.get(key);
  if (entry) {
    cache.delete(key);
    cache.set(key, entry);
  }
  return entry;
}

function cacheSet(key: string, entry: CacheEntry): void {
  cache.set(key, entry);
  if (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
}

async function decodeMono(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`stem ${res.status} ${url}`);
  const bytes = await res.arrayBuffer();
  // Length is ignored by decodeAudioData; it decodes to this context's rate.
  const ctx = new OfflineAudioContext(2, Math.max(1, sampleRate), sampleRate);
  const audio = await ctx.decodeAudioData(bytes);
  return mixToMono(audio);
}

async function loadItem(item: PoolItem): Promise<{ mono: Float32Array; pre: Float32Array | null }> {
  const cached = cacheGet(item.key);
  if (cached) return cached;
  const mono = await decodeMono(item.url);
  let pre: Float32Array | null = null;
  if (item.preUrl) {
    try {
      pre = await decodeMono(item.preUrl);
    } catch {
      // A missing tail must not block the phrase (same rule as the main path).
      pre = null;
    }
  }
  cacheSet(item.key, { mono, pre });
  return { mono, pre };
}

async function handleLoad(id: number, items: PoolItem[]): Promise<void> {
  // Decode this run's phrases concurrently (fetch + decode are async), then
  // concatenate; only the first phrase's pre-roll starts the run.
  const loaded = await Promise.all(items.map((item) => loadItem(item)));
  const { samples, preDur, mainDurations } = concatRun(
    loaded.map((entry) => entry.mono),
    loaded[0]?.pre ?? null,
    sampleRate,
  );
  host().postMessage(
    { type: "run", id, samples, preDur, mainDurations, sampleRate },
    [samples.buffer],
  );
}

(self as unknown as {
  onmessage: ((ev: MessageEvent) => void) | null;
}).onmessage = (ev: MessageEvent): void => {
  const msg = ev.data as
    | { type?: string; sampleRate?: number; id?: number; items?: PoolItem[] }
    | null
    | undefined;
  if (!msg || typeof msg.type !== "string") return;
  if (msg.type === "init") {
    if (typeof msg.sampleRate === "number" && msg.sampleRate > 0) {
      sampleRate = msg.sampleRate;
    }
    return;
  }
  if (msg.type === "load" && typeof msg.id === "number" && Array.isArray(msg.items)) {
    const id = msg.id;
    void handleLoad(id, msg.items).catch((err: unknown) => {
      host().postMessage({
        type: "error",
        id,
        message: err instanceof Error ? err.message : String(err),
      });
    });
  }
};
