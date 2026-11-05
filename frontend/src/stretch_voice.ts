// One-shot Signalsmith Stretch voice for a real 4-bar stem phrase.
//
// The kit phrase is decoded at the AudioContext rate, so no resampling is
// introduced.  A phrase is prepared (pre-roll + main body concatenated) once,
// then scheduled with independent tempo (`rate`) and pitch (`semitones`).
// Signalsmith compensates for its own processing latency when `output` is
// scheduled ahead of time.
import type { StretchNode } from "signalsmith-stretch";

type StretchFactory = (
  ctx: BaseAudioContext,
  options?: AudioWorkletNodeOptions,
) => Promise<StretchNode>;

let factoryPromise: Promise<StretchFactory> | null = null;

/**
 * Load the prebuilt Signalsmith module as a native ESM asset instead of
 * letting Vite bundle it. The library stringifies its AudioWorkletProcessor
 * and WASM loader into a Blob; bundler transforms break that source, which
 * makes the worklet silence the node in production.
 */
function loadFactory(): Promise<StretchFactory> {
  if (factoryPromise === null) {
    factoryPromise = (async () => {
      // Resolve relative to this module without `new URL(...)`, which Vite
      // rewrites as a build-time asset path even inside a dynamic import.
      const here = import.meta.url;
      const base = here.slice(0, here.lastIndexOf("/") + 1);
      const url = `${base}../vendor/signalsmith-stretch.mjs`;
      const module = await import(/* @vite-ignore */ url);
      const factory = (module.default ?? module) as unknown;
      if (typeof factory !== "function") {
        throw new Error("Signalsmith Stretch vendor module did not export a factory");
      }
      return factory as StretchFactory;
    })();
  }
  return factoryPromise;
}

export interface StretchVoiceOptions {
  /** Output gain before the stem bus (0..1.5). */
  gain?: number;
  /** Stereo pan for this voice (-1..1). */
  pan?: number;
  /** Block size in ms; smaller reacts faster, larger is cheaper. */
  blockMs?: number;
}

export interface StretchScheduleOptions {
  /** Absolute AudioContext time of the first sample. */
  when: number;
  /** Input-buffer position in seconds (0 = start of the pre-roll). */
  inputSec?: number;
  /** Output/input tempo ratio. 1.0 keeps the source tempo. */
  rate?: number;
  /** Pitch shift in semitones. */
  semitones?: number;
  /** Preserve vocal formants while shifting. */
  formantCompensation?: boolean;
  /** Fade-in length in seconds. */
  fadeInSec?: number;
  /** Optional gain override for this phrase. */
  gain?: number;
}

export class StretchVoice {
  public readonly node: StretchNode;
  private readonly gain: GainNode;
  private readonly panner: StereoPannerNode;
  private disposed = false;
  private scheduled = false;
  private targetGain = 1;

  private constructor(node: StretchNode, gain: GainNode, panner: StereoPannerNode) {
    this.node = node;
    this.gain = gain;
    this.panner = panner;
  }

  static async create(
    ctx: AudioContext,
    destination: AudioNode,
    options: StretchVoiceOptions = {},
  ): Promise<StretchVoice> {
    const factory = await loadFactory();
    // Keep the library's default input count. `numberOfInputs: 0` makes
    // Chrome skip the first process() call in some builds, which permanently
    // silences the node. Nothing is connected to the input, so buffer
    // playback still takes the sample-buffer path.
    const node = await factory(ctx, {
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
    if (typeof options.blockMs === "number" && options.blockMs > 0) {
      node.configure({ blockMs: options.blockMs });
    }
    const gain = ctx.createGain();
    gain.gain.value = 0;
    const panner = ctx.createStereoPanner();
    panner.pan.value = Math.max(-1, Math.min(1, options.pan ?? 0));
    node.connect(gain);
    gain.connect(panner);
    panner.connect(destination);
    return new StretchVoice(node, gain, panner);
  }

  /**
   * Concatenate the optional pre-bar tail with the main phrase.  The caller
   * schedules `inputSec = preRollSec` at the bar time so the pre-roll plays
   * just before the bar and the phrase's downbeat lands exactly on it.
   */
  async load(buffer: AudioBuffer, preRoll: AudioBuffer | null): Promise<number> {
    const main = mixToMono(buffer);
    const pre = preRoll ? mixToMono(preRoll) : null;
    const samples = new Float32Array((pre?.length ?? 0) + main.length);
    if (pre) samples.set(pre, 0);
    samples.set(main, pre?.length ?? 0);
    await this.node.addBuffers([samples]);
    return (pre?.length ?? 0) / buffer.sampleRate;
  }

  /** Schedule the prepared phrase. Must be called after `load`. */
  schedule(options: StretchScheduleOptions): void {
    if (this.disposed) return;
    const ctx = this.gain.context;
    const when = Math.max(options.when, ctx.currentTime + 0.005);
    const fadeIn = Math.max(0.002, options.fadeInSec ?? 0.003);
    const target = Math.max(0, Math.min(1.5, options.gain ?? 1));
    this.targetGain = target;
    const gain = this.gain.gain;
    gain.cancelScheduledValues(when);
    gain.setValueAtTime(0.0001, when);
    gain.linearRampToValueAtTime(target, when + fadeIn);
    this.node.schedule({
      outputTime: when,
      active: true,
      input: Math.max(0, options.inputSec ?? 0),
      rate: options.rate ?? 1,
      semitones: options.semitones ?? 0,
      formantCompensation: options.formantCompensation ?? false,
    });
    this.scheduled = true;
  }

  /** Ramp this voice out and stop it. */
  fadeOut(at: number, durationSec: number = 0.08): void {
    if (this.disposed) return;
    const ctx = this.gain.context;
    const stopAt = Math.max(at, ctx.currentTime + 0.005);
    const end = stopAt + Math.max(0.005, durationSec);
    const gain = this.gain.gain;
    try {
      // Do not cancel earlier automation here: a phrase is scheduled and its
      // end fade is queued in the same call, and cancelAndHoldAtTime would
      // cancel the fade-in and leave the voice silent.
      gain.setValueAtTime(this.targetGain, stopAt);
      gain.linearRampToValueAtTime(0.0001, end);
    } catch {
      // Parameter automation can fail if the context is closing.
    }
    try {
      // The library's stop(when) drops its time argument in 1.3.2 (it reads
      // `output` while the scheduler reads `outputTime`), which stops the
      // voice immediately. Schedule the deactivate segment directly.
      // The library's schedule() drops the previous active segment when a
      // deactivate segment is queued at the same future time, so let the
      // buffer run out and dispose the node after the gain fade instead.
      const delayMs = Math.max(0, (end - ctx.currentTime + 0.15) * 1000);
      globalThis.setTimeout(() => this.dispose(), delayMs);
    } catch {
      // Already stopped.
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    try {
      this.node.stop();
    } catch {
      // Already stopped.
    }
    try {
      this.node.disconnect();
      this.gain.disconnect();
      this.panner.disconnect();
    } catch {
      // Already disconnected.
    }
  }

  get isScheduled(): boolean {
    return this.scheduled;
  }

  /** Post-gain/panner output, for chaining into a stem bus or analyser. */
  get output(): AudioNode {
    return this.panner;
  }

  /** Current gain parameter value (debug). */
  get gainValue(): number {
    return this.gain.gain.value;
  }
}

function mixToMono(buffer: AudioBuffer): Float32Array {
  const channels = buffer.numberOfChannels;
  if (channels <= 1) return buffer.getChannelData(0);
  const length = buffer.length;
  const out = new Float32Array(length);
  for (let c = 0; c < channels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i++) out[i] += data[i];
  }
  const scale = 1 / channels;
  for (let i = 0; i < length; i++) out[i] *= scale;
  return out;
}
