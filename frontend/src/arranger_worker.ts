/** Sequence-model inference worker (onnxruntime-web, wasm CPU).
 *
 * Model-agnostic: serves the bar arranger (6 tokens/bar) and the tatum
 * filler (64 steps/bar) from one file. Two worker instances are spawned,
 * one per model, each init()ed with its own config name.
 *
 * Protocol (main thread):
 *   {type:'init', basePath, config:'arranger_gpt.json'} -> {type:'ready', vocab, ctx}
 *   {type:'plan', history:number[], nSteps, temp, topK, seed}
 *     -> {type:'plan', steps:number[], seed}  (flat token ids)
 *   Any failure -> {type:'error', message} (caller falls back to Markov).
 *
 * History is chronological flat tokens. The worker pads/truncates to ctx
 * and runs nSteps autoregressive steps with temperature + top-k sampling
 * (mulberry32, seeded).
 */
import * as ort from "onnxruntime-web/wasm";

let session: ort.InferenceSession | null = null;
let vocab = 1222;
let ctx = 192;

function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function init(basePath: string, configName: string): Promise<void> {
  const configUrl = `${basePath}/${configName}`;
  const cfgRes = await fetch(configUrl);
  if (!cfgRes.ok) throw new Error(`config fetch failed: ${configUrl}`);
  const cfg = (await cfgRes.json()) as { vocab?: number; ctx?: number; model_int8?: string; model?: string };
  vocab = cfg.vocab || 1222;
  ctx = cfg.ctx || 192;
  ort.env.wasm.wasmPaths = `${basePath}/ort/`;
  // Prefer int8 (smaller download); fall back to fp32.
  const urls = [cfg.model_int8 ? `${basePath}/${cfg.model_int8}` : "", cfg.model ? `${basePath}/${cfg.model}` : ""].filter(
    (u) => u.length > 0
  );
  let lastErr: unknown = null;
  for (const u of urls) {
    try {
      session = await ort.InferenceSession.create(u, { executionProviders: ["wasm"] });
      return;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("ORT session failed");
}

async function plan(history: number[], nSteps: number, temp: number, topK: number, seed: number): Promise<number[]> {
  if (!session) throw new Error("model not initialized");
  const rnd = mulberry32(seed);
  const out: number[] = [];
  let toks = history.slice(-ctx);
  for (let s = 0; s < nSteps; s++) {
    const win = toks.slice(-ctx);
    const pad = new Array(Math.max(0, ctx - win.length)).fill(0);
    const ids = pad.concat(win);
    const feeds: Record<string, ort.Tensor> = {
      tokens: new ort.Tensor("int64", BigInt64Array.from(ids.map((v) => BigInt(v))), [1, ctx]),
    };
    const res = await session.run(feeds);
    const logits = res["logits"] as ort.Tensor;
    // last position logits
    const data = logits.data as Float32Array;
    const off = (ctx - 1) * vocab;
    const row = data.slice(off, off + vocab);
    const t = Math.max(1e-6, temp);
    // temperature + top-k
    const indexed = Array.from(row, (v, i) => i);
    indexed.sort((a, b2) => row[b2] / t - row[a] / t);
    const cand = indexed.slice(0, Math.max(1, Math.min(topK, vocab)));
    const weights = cand.map((i) => Math.exp(row[i] / t - row[cand[0]] / t));
    let r = rnd() * weights.reduce((a, b2) => a + b2, 0);
    let nxt = cand[cand.length - 1];
    for (let i = 0; i < cand.length; i++) {
      r -= weights[i];
      if (r <= 0) {
        nxt = cand[i];
        break;
      }
    }
    out.push(nxt);
    toks.push(nxt);
  }
  return out;
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as { type?: string } & Record<string, unknown>;
  try {
    if (msg.type === "init") {
      await init((msg.basePath as string) || "data", (msg.config as string) || "arranger_gpt.json");
      (self as unknown as { postMessage: (m: unknown) => void }).postMessage({ type: "ready", vocab, ctx });
    } else if (msg.type === "plan") {
      const steps = await plan(
        (msg.history as number[]) || [],
        (msg.nSteps as number) || 6,
        (msg.temp as number) || 1.0,
        (msg.topK as number) || 20,
        (msg.seed as number) || 0
      );
      (self as unknown as { postMessage: (m: unknown) => void }).postMessage({ type: "plan", steps, seed: msg.seed });
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    (self as unknown as { postMessage: (m: unknown) => void }).postMessage({
      type: "error",
      message,
      // Init failures are fatal (no model); plan failures are transient.
      // One bad draw must never brick the worker.
      fatal: message.includes("config fetch failed") || message.includes("ORT session"),
    });
  }
};
