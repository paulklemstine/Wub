// Phase 3: sample-accurate 4-bus Web Audio engine for the infinite jukebox.
//
// Signal path per stem:
//   slice SourceNode -> per-slice GainNode (equal-power fades)
//     -> bus GainNode (volume/mute/solo) -> BiquadFilterNode -> AnalyserNode
//     -> master DynamicsCompressorNode -> destination.
//
// Scheduling model: callers pass an absolute AudioContext timestamp (`when`)
// and this engine calls `source.start(when)` directly, which is already
// sample-accurate. A 25 ms watchdog interval (SCHEDULER_INTERVAL_MS) owns the
// 150 ms schedule-ahead bookkeeping window (SCHEDULE_AHEAD_TIME_S) and sweeps
// finished sources so disconnected nodes can be GC'd. If a future backend
// needs deferred dispatch, this is the poll point for it.
import type { StemName } from "./types";

export const STEMS: readonly StemName[] = ["drums", "bass", "vocals", "other"];
export const ENGINE_SAMPLE_RATE = 44100;
export const SCHEDULER_INTERVAL_MS = 25;
export const SCHEDULE_AHEAD_TIME_S = 0.15;
export const DEFAULT_FADE_MS = 3;
export const MAX_VOLUME = 1.5;

export interface SliceScheduleOptions {
  /** Playback rate multiplier (time-stretch factor from the server). */
  rate?: number;
  /** Detune in cents (pitch correction from the server). */
  detuneCents?: number;
  /** Equal-power fade length in ms (default 3). */
  fadeMs?: number;
  /** Optional playback duration clamp in seconds (fades out at end of duration). */
  duration?: number;
}

interface StemFilterSpec {
  type: BiquadFilterType;
  frequency: number;
  q: number;
}

// Sensible per-stem defaults: drums wide open, bass tightly low-passed so
// sub content stays clean, other/vocals low-passed high enough to be neutral.
const STEM_FILTERS: Record<StemName, StemFilterSpec> = {
  drums: { type: "lowpass", frequency: 18000, q: 0.7 },
  bass: { type: "lowpass", frequency: 800, q: 0.7 },
  other: { type: "lowpass", frequency: 16000, q: 0.7 },
  vocals: { type: "lowpass", frequency: 12000, q: 0.7 },
};

interface StemBus {
  input: GainNode;
  filter: BiquadFilterNode;
  analyser: AnalyserNode;
}

function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.min(MAX_VOLUME, Math.max(0, v));
}

function resolveAudioContextCtor(): typeof AudioContext | null {
  if (typeof window !== "undefined" && typeof window.AudioContext !== "undefined") {
    return window.AudioContext;
  }
  if (typeof window !== "undefined") {
    const candidate: unknown = (window as unknown as Record<string, unknown>)["webkitAudioContext"];
    if (typeof candidate === "function") {
      return candidate as typeof AudioContext;
    }
  }
  return null;
}

export class AudioEngine {
  readonly scheduleAheadTime = SCHEDULE_AHEAD_TIME_S;
  readonly schedulerIntervalMs = SCHEDULER_INTERVAL_MS;

  private ctx_: AudioContext | null = null;
  private master_: DynamicsCompressorNode | null = null;
  private masterAnalyser_: AnalyserNode | null = null;
  private buses_: Record<StemName, StemBus> | null = null;
  private timer_: ReturnType<typeof setInterval> | null = null;
  private liveSources_: Map<AudioBufferSourceNode, number> = new Map();
  private activeVoices_: Partial<Record<StemName, { source: AudioBufferSourceNode; gain: GainNode; endAt: number }>> = {};
  private targetBpm_: number = 128;

  // Desired mixer state, applied lazily so UI can be touched before the
  // AudioContext exists (creating one pre-gesture would start suspended).
  private volumes_: Record<StemName, number> = { drums: 1, bass: 1, vocals: 1, other: 1 };
  private mutes_: Record<StemName, boolean> = { drums: false, bass: false, vocals: false, other: false };
  private solos_: Record<StemName, boolean> = { drums: false, bass: false, vocals: false, other: false };

  get ready(): boolean {
    return this.ctx_ !== null && this.buses_ !== null;
  }

  get targetBpm(): number {
    return this.targetBpm_;
  }

  ensureCtx(): AudioContext {
    if (this.ctx_ !== null && this.buses_ !== null) return this.ctx_;
    const Ctor = resolveAudioContextCtor();
    if (Ctor === null) throw new Error("Web Audio is not supported in this browser");
    const ctx = new Ctor({ sampleRate: ENGINE_SAMPLE_RATE });

    const master = ctx.createDynamicsCompressor();
    master.threshold.value = -12;
    master.knee.value = 20;
    master.ratio.value = 4;
    master.attack.value = 0.003;
    master.release.value = 0.2;

    const masterAnalyser = ctx.createAnalyser();
    masterAnalyser.fftSize = 2048;
    masterAnalyser.smoothingTimeConstant = 0.8;

    master.connect(masterAnalyser);
    masterAnalyser.connect(ctx.destination);

    const buses = {} as Record<StemName, StemBus>;
    for (const stem of STEMS) {
      const input = ctx.createGain();
      const spec = STEM_FILTERS[stem];
      const filter = ctx.createBiquadFilter();
      filter.type = spec.type;
      filter.frequency.value = spec.frequency;
      filter.Q.value = spec.q;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      input.connect(filter);
      filter.connect(analyser);
      analyser.connect(master);
      buses[stem] = { input, filter, analyser };
    }

    this.ctx_ = ctx;
    this.master_ = master;
    this.masterAnalyser_ = masterAnalyser;
    this.buses_ = buses;
    this.applyGains();
    this.startWatchdog();
    return ctx;
  }

  private startWatchdog(): void {
    if (this.timer_ !== null) return;
    this.timer_ = setInterval(() => {
      const ctx = this.ctx_;
      if (ctx === null) return;
      const horizon = ctx.currentTime - 2;
      for (const [node, endAt] of this.liveSources_) {
        if (endAt < horizon) {
          try {
            node.disconnect();
          } catch {
            // Already disconnected; safe to ignore.
          }
          this.liveSources_.delete(node);
        }
      }
    }, SCHEDULER_INTERVAL_MS);
  }

  private applyGains(): void {
    if (this.ctx_ === null || this.buses_ === null) return;
    const t = this.ctx_.currentTime;
    const anySolo = STEMS.some((s) => this.solos_[s]);
    for (const stem of STEMS) {
      const bus = this.buses_[stem];
      const audible = !this.mutes_[stem] && (!anySolo || this.solos_[stem]);
      bus.input.gain.setTargetAtTime(audible ? this.volumes_[stem] : 0, t, 0.01);
    }
  }

  /** Create (lazily) and resume the context. Must be called from a gesture. */
  async resume(): Promise<void> {
    const ctx = this.ensureCtx();
    if (ctx.state === "suspended") await ctx.resume();
  }

  /** Suspend output (Stop button). Keeps mixer state for the next Play. */
  async suspend(): Promise<void> {
    if (this.ctx_ !== null && this.ctx_.state === "running") {
      await this.ctx_.suspend();
    }
  }

  setTargetBpm(bpm: number): void {
    if (!Number.isFinite(bpm)) return;
    this.targetBpm_ = Math.min(220, Math.max(40, bpm));
  }

  setVolume(stem: StemName, v: number): void {
    this.volumes_[stem] = clampVolume(v);
    this.applyGains();
  }

  getVolume(stem: StemName): number {
    return this.volumes_[stem];
  }

  setMute(stem: StemName, muted: boolean): void {
    this.mutes_[stem] = muted;
    this.applyGains();
  }

  isMuted(stem: StemName): boolean {
    return this.mutes_[stem];
  }

  setSolo(stem: StemName, solo: boolean): void {
    this.solos_[stem] = solo;
    this.applyGains();
  }

  isSolo(stem: StemName): boolean {
    return this.solos_[stem];
  }

  getAnalyser(stem: StemName): AnalyserNode {
    this.ensureCtx();
    const buses = this.buses_;
    if (buses === null) throw new Error("AudioEngine buses not initialised");
    return buses[stem].analyser;
  }

  getStemInput(stem: StemName): AudioNode {
    this.ensureCtx();
    const buses = this.buses_;
    if (buses === null) throw new Error("AudioEngine buses not initialised");
    return buses[stem].input;
  }

  /** Dynamic sidechain ducking on a stem bus (e.g. ducking bass on kick transients). */
  duckStem(stem: StemName, when: number, duckGain: number = 0.35, releaseSec: number = 0.075): void {
    if (this.ctx_ === null || this.buses_ === null) return;
    const bus = this.buses_[stem];
    const targetVol = this.volumes_[stem];
    const t = Math.max(this.ctx_.currentTime, when);
    try {
      bus.input.gain.cancelScheduledValues(t);
      bus.input.gain.setValueAtTime(targetVol * duckGain, t);
      bus.input.gain.exponentialRampToValueAtTime(Math.max(0.001, targetVol), t + releaseSec);
    } catch {
      // Safe fallback
    }
  }

  getMasterAnalyser(): AnalyserNode {
    this.ensureCtx();
    if (this.masterAnalyser_ === null) throw new Error("AudioEngine master analyser not initialised");
    return this.masterAnalyser_;
  }

  /** Current audio-clock time, or 0 when the context does not exist yet. */
  now(): number {
    return this.ctx_ !== null ? this.ctx_.currentTime : 0;
  }

  async decodeWav(bytes: ArrayBuffer): Promise<AudioBuffer> {
    const ctx = this.ensureCtx();
    // decodeAudioData detaches its input; decode a copy instead.
    const copy = bytes.slice(0);
    return await ctx.decodeAudioData(copy);
  }

  /**
   * Schedule one slice at absolute context time `when`. Late timestamps are
   * clamped to just after now so late-arriving slices still sound. Returns
   * the live source node.
   */
  scheduleSlice(
    stem: StemName,
    buffer: AudioBuffer,
    when: number,
    opts: SliceScheduleOptions = {},
  ): AudioBufferSourceNode {
    const ctx = this.ensureCtx();
    const buses = this.buses_;
    if (buses === null) throw new Error("AudioEngine buses not initialised");

    const rate = typeof opts.rate === "number" && Number.isFinite(opts.rate) && opts.rate > 0 ? opts.rate : 1;
    const detune =
      typeof opts.detuneCents === "number" && Number.isFinite(opts.detuneCents)
        ? Math.min(2400, Math.max(-2400, opts.detuneCents))
        : 0;
    const requestedFadeMs =
      typeof opts.fadeMs === "number" && Number.isFinite(opts.fadeMs) ? Math.max(0, opts.fadeMs) : DEFAULT_FADE_MS;

    const startAt = Math.max(when, ctx.currentTime + 0.005);
    const naturalDurSec = buffer.duration / rate;
    const durSec =
      typeof opts.duration === "number" && Number.isFinite(opts.duration) && opts.duration > 0 && opts.duration < naturalDurSec
        ? opts.duration
        : naturalDurSec;

    // Crossfade out previous voice on this stem so voices never pile up into noise
    const prev = this.activeVoices_[stem];
    if (prev && prev.endAt > startAt) {
      const crossfadeSec = Math.min(requestedFadeMs / 1000, 0.008);
      const stopAt = startAt + crossfadeSec;
      try {
        prev.gain.gain.cancelScheduledValues(startAt);
        prev.gain.gain.setValueAtTime(prev.gain.gain.value, startAt);
        prev.gain.gain.linearRampToValueAtTime(0, stopAt);
        prev.source.stop(stopAt + 0.01);
      } catch {
        // Safe if already stopped
      }
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = rate;
    source.detune.value = detune;

    const sliceGain = ctx.createGain();
    source.connect(sliceGain);
    sliceGain.connect(buses[stem].input);

    if (requestedFadeMs > 0 && durSec > 0) {
      const fadeSec = Math.min(requestedFadeMs / 1000, durSec / 2);
      if (fadeSec > 0) {
        const n = Math.max(2, Math.round(fadeSec * ctx.sampleRate));
        const fadeIn = new Float32Array(n);
        const fadeOut = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          const t = i / (n - 1);
          // Equal-power (cosine) curves: sin^2 + cos^2 = 1 across the seam.
          fadeIn[i] = Math.sin(0.5 * Math.PI * t);
          fadeOut[i] = Math.cos(0.5 * Math.PI * t);
        }
        const endAt = startAt + durSec;
        sliceGain.gain.setValueAtTime(0, startAt);
        sliceGain.gain.setValueCurveAtTime(fadeIn, startAt, fadeSec);
        sliceGain.gain.setValueAtTime(1, startAt + fadeSec);
        sliceGain.gain.setValueCurveAtTime(fadeOut, endAt - fadeSec, fadeSec);
      } else {
        sliceGain.gain.value = 1;
      }
    } else {
      sliceGain.gain.value = 1;
    }

    const endAt = startAt + durSec;
    this.activeVoices_[stem] = { source, gain: sliceGain, endAt };
    source.onended = () => {
      this.liveSources_.delete(source);
      if (this.activeVoices_[stem]?.source === source) {
        delete this.activeVoices_[stem];
      }
      try {
        source.disconnect();
        sliceGain.disconnect();
      } catch {
        // Already disconnected; safe to ignore.
      }
    };
    this.liveSources_.set(source, endAt);
    source.start(startAt);
    try {
      source.stop(endAt + 0.05);
    } catch {
      // stop() after the natural end throws in some browsers; ignore.
    }
    return source;
  }

  /**
   * Synthesize real punchy drum, bass, or melodic stabs in Web Audio.
   * Feeds directly into stem buses, respecting mutes, volume, and visualizers.
   */
  triggerSynth(
    stem: StemName,
    when: number,
    opts: { type?: string; freq?: number; gain?: number; decay?: number } = {},
  ): void {
    const ctx = this.ensureCtx();
    const buses = this.buses_;
    if (buses === null) return;

    const startAt = Math.max(when, ctx.currentTime);
    const busInput = buses[stem].input;

    if (stem === "drums") {
      const type = opts.type || "kick";
      if (type === "kick") {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(busInput);

        const decay = opts.decay || 0.28;
        osc.frequency.setValueAtTime(155, startAt);
        osc.frequency.exponentialRampToValueAtTime(36, startAt + decay * 0.4);

        gain.gain.setValueAtTime(1.1 * (opts.gain ?? 1), startAt);
        gain.gain.exponentialRampToValueAtTime(0.001, startAt + decay);

        osc.start(startAt);
        osc.stop(startAt + decay + 0.05);
      } else if (type === "snare" || type === "clap") {
        const osc = ctx.createOscillator();
        const oscGain = ctx.createGain();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(180, startAt);
        osc.frequency.exponentialRampToValueAtTime(80, startAt + 0.1);
        oscGain.gain.setValueAtTime(0.7 * (opts.gain ?? 1), startAt);
        oscGain.gain.exponentialRampToValueAtTime(0.001, startAt + 0.15);
        osc.connect(oscGain);
        oscGain.connect(busInput);
        osc.start(startAt);
        osc.stop(startAt + 0.16);

        const noiseLen = Math.floor(ctx.sampleRate * 0.15);
        const noiseBuf = ctx.createBuffer(1, noiseLen, ctx.sampleRate);
        const data = noiseBuf.getChannelData(0);
        for (let i = 0; i < noiseLen; i++) data[i] = Math.random() * 2 - 1;
        const noiseNode = ctx.createBufferSource();
        noiseNode.buffer = noiseBuf;

        const noiseFilter = ctx.createBiquadFilter();
        noiseFilter.type = "bandpass";
        noiseFilter.frequency.value = 1600;
        noiseFilter.Q.value = 1.2;

        const noiseGain = ctx.createGain();
        noiseGain.gain.setValueAtTime(0.8 * (opts.gain ?? 1), startAt);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, startAt + 0.15);

        noiseNode.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(busInput);
        noiseNode.start(startAt);
        noiseNode.stop(startAt + 0.16);
      } else {
        const isClosed = type !== "open_hat";
        const decay = isClosed ? 0.05 : 0.22;
        const noiseLen = Math.floor(ctx.sampleRate * decay);
        const noiseBuf = ctx.createBuffer(1, noiseLen, ctx.sampleRate);
        const data = noiseBuf.getChannelData(0);
        for (let i = 0; i < noiseLen; i++) data[i] = Math.random() * 2 - 1;
        const noiseNode = ctx.createBufferSource();
        noiseNode.buffer = noiseBuf;

        const noiseFilter = ctx.createBiquadFilter();
        noiseFilter.type = "highpass";
        noiseFilter.frequency.value = 7500;

        const noiseGain = ctx.createGain();
        noiseGain.gain.setValueAtTime(0.4 * (opts.gain ?? 1), startAt);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, startAt + decay);

        noiseNode.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(busInput);
        noiseNode.start(startAt);
        noiseNode.stop(startAt + decay + 0.02);
      }
    } else if (stem === "bass") {
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const gain = ctx.createGain();

      osc.type = "sawtooth";
      const freq = opts.freq || 65.4;
      osc.frequency.setValueAtTime(freq, startAt);

      filter.type = "lowpass";
      filter.frequency.setValueAtTime(700, startAt);
      filter.frequency.exponentialRampToValueAtTime(180, startAt + 0.2);
      filter.Q.value = 3.0;

      const decay = opts.decay || 0.22;
      gain.gain.setValueAtTime(0.7 * (opts.gain ?? 1), startAt);
      gain.gain.exponentialRampToValueAtTime(0.001, startAt + decay);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(busInput);

      osc.start(startAt);
      osc.stop(startAt + decay + 0.05);
    } else {
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const gain = ctx.createGain();

      osc1.type = stem === "vocals" ? "triangle" : "sawtooth";
      osc2.type = "square";

      const freq = opts.freq || (stem === "vocals" ? 330 : 261.6);
      osc1.frequency.setValueAtTime(freq, startAt);
      osc2.frequency.setValueAtTime(freq * 1.004, startAt);

      filter.type = "bandpass";
      filter.frequency.setValueAtTime(1400, startAt);
      filter.frequency.exponentialRampToValueAtTime(600, startAt + 0.25);
      filter.Q.value = 2.0;

      const decay = opts.decay || 0.25;
      gain.gain.setValueAtTime(0.35 * (opts.gain ?? 1), startAt);
      gain.gain.exponentialRampToValueAtTime(0.001, startAt + decay);

      osc1.connect(filter);
      osc2.connect(filter);
      filter.connect(gain);
      gain.connect(busInput);

      osc1.start(startAt);
      osc2.start(startAt);
      osc1.stop(startAt + decay + 0.05);
      osc2.stop(startAt + decay + 0.05);
    }
  }

  dispose(): void {
    if (this.timer_ !== null) {
      clearInterval(this.timer_);
      this.timer_ = null;
    }
    for (const [node] of this.liveSources_) {
      try {
        node.stop();
      } catch {
        // Already stopped; safe to ignore.
      }
      try {
        node.disconnect();
      } catch {
        // Already disconnected; safe to ignore.
      }
    }
    this.liveSources_.clear();
    this.activeVoices_ = {};
    const ctx = this.ctx_;
    this.ctx_ = null;
    this.master_ = null;
    this.masterAnalyser_ = null;
    this.buses_ = null;
    if (ctx !== null) {
      void ctx.close().catch(() => undefined);
    }
  }
}
