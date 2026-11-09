// Phase 3: sample-accurate 4-bus Web Audio engine for the infinite jukebox.
//
// Signal path per stem:
//   slice SourceNode -> per-slice GainNode (equal-power fades)
//     -> bus GainNode (volume/mute/solo) -> BiquadFilterNode -> AnalyserNode
//     -> glue GainNode -> master DynamicsCompressorNode -> destination.
//
// Shared glue reverb:
//   stem AnalyserNode -> reverb send GainNode -> ConvolverNode
//     -> high-pass BiquadFilterNode (~300 Hz) -> return GainNode (-20 dB)
//     -> master.
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
export const DEFAULT_FADE_MS = 6;
export const MAX_VOLUME = 1.5;

// Glue bus / shared reverb tuning. Deliberately conservative: the glue stage is
// a mix layer, never a replacement for setStemMixLevel's input gain.
const GLUE_ANALYSER_FFT = 2048;
const GLUE_MIN = 0.5;
const GLUE_MAX = 1.0;
const GLUE_RAMP_SEC = 0.05;
const VOCAL_RMS_FLOOR = 0.01;
const VOCAL_RMS_FULL = 0.06;
const VOCAL_DUCK_DB = 1.5;
const KICK_DUCK_DB = 2.5;
const VOCAL_ATTACK_SEC = 0.02;
const VOCAL_RELEASE_SEC = 0.25;
const KICK_ATTACK_SEC = 0.008;
const KICK_RELEASE_SEC = 0.12;
const KICK_TRANSIENT_HOLD_SEC = 0.06;
const KICK_SHORT_WINDOW = 512;
const REVERB_SECONDS = 0.55;
const REVERB_PRE_DELAY_SEC = 0.012;
const REVERB_RETURN_DB = -20;
const REVERB_HIGHPASS_HZ = 300;
// DJ transition technique (B): vocal reverb tail while the outgoing vocal
// fades.  Boosted send briefly, then back to the user's send.  Conservative
// on purpose: the tail must read as a hand-off, never as a hall.
const TRANSITION_REVERB_TAIL = 0.22;
const TRANSITION_REVERB_TAIL_SEC = 0.35;

const DEFAULT_REVERB_SENDS: Record<StemName, number> = {
  drums: 0,
  bass: 0,
  other: 0.08,
  vocals: 0.10,
};

export interface GlueState {
  enabled: boolean;
  gains: Record<StemName, number>;
  base: Record<StemName, number>;
  duck: Record<StemName, number>;
  vocalRms: number;
  vocalPresence: number;
  drumShortRms: number;
  drumTransient: boolean;
}

export interface SliceScheduleOptions {
  /** Playback rate multiplier (time-stretch factor from the server). */
  rate?: number;
  /** Detune in cents (pitch correction from the server). */
  detuneCents?: number;
  /** Equal-power fade length in ms (default 3). */
  fadeMs?: number;
  /** Optional shorter source fade-in for block seams (ms). */
  fadeInMs?: number;
  /** Optional shorter source fade-out for block seams (ms). */
  fadeOutMs?: number;
  /** Optional playback duration clamp in seconds (fades out at end of duration). */
  duration?: number;
  /** Gain / velocity scaling (0.0..1.5). */
  gain?: number;
  /** Dedicated voice track key (e.g. "drums:kick", "drums:snare", "drums:hat_closed", "drums:hat_open", "bass"). */
  voiceKey?: string;
  /** Microtiming nudge in ms (+ = late/laid-back). Bounded ±15ms by scheduler. */
  whenOffsetMs?: number;  /** Voice keys to choke (fade out over crossfadeSec). E.g. ["drums:hat_open"] choked by hat_closed. */
  chokeKeys?: string[];
}

interface StemFilterSpec {
  type: BiquadFilterType;
  frequency: number;
  q: number;
}

// Sensible per-stem defaults: drums wide open, bass tightly low-passed so
// sub content stays clean, other/vocals low-passed high enough to be neutral.
const STEM_FILTERS: Record<StemName, StemFilterSpec> = {
  drums: { type: "lowpass", frequency: 20000, q: 0.7 },
  bass: { type: "lowpass", frequency: 20000, q: 0.7 },
  other: { type: "lowpass", frequency: 20000, q: 0.7 },
  vocals: { type: "lowpass", frequency: 20000, q: 0.7 },
};

interface StemBus {
  input: GainNode;
  filter: BiquadFilterNode;
  analyser: AnalyserNode;
  glue: GainNode;
  reverbSend: GainNode;
}

function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.min(MAX_VOLUME, Math.max(0, v));
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

function clampGlue(v: number): number {
  if (!Number.isFinite(v)) return GLUE_MAX;
  return Math.min(GLUE_MAX, Math.max(GLUE_MIN, v));
}

function rmsOf(data: Float32Array, start = 0, end = data.length): number {
  const from = Math.max(0, Math.min(data.length, start));
  const to = Math.max(from, Math.min(data.length, end));
  if (to <= from) return 0;
  let sum = 0;
  for (let i = from; i < to; i++) {
    const v = data[i];
    sum += v * v;
  }
  return Math.sqrt(sum / (to - from));
}

function peakOf(data: Float32Array, start = 0, end = data.length): number {
  const from = Math.max(0, Math.min(data.length, start));
  const to = Math.max(from, Math.min(data.length, end));
  let peak = 0;
  for (let i = from; i < to; i++) {
    const a = Math.abs(data[i]);
    if (a > peak) peak = a;
  }
  return peak;
}

/**
 * Short stereo plate impulse, generated in JS so the kit bank never needs an
 * extra asset. Independent L/R noise and a gentle one-pole low-pass keep the
 * tail dark; ConvolverNode's built-in normalisation handles absolute level.
 */
function createReverbImpulse(ctx: AudioContext): AudioBuffer {
  const length = Math.max(1, Math.floor(ctx.sampleRate * REVERB_SECONDS));
  const preDelay = Math.min(length - 1, Math.floor(ctx.sampleRate * REVERB_PRE_DELAY_SEC));
  const impulse = ctx.createBuffer(2, length, ctx.sampleRate);
  for (let ch = 0; ch < impulse.numberOfChannels; ch++) {
    const data = impulse.getChannelData(ch);
    let smooth = 0;
    for (let i = 0; i < length; i++) {
      if (i < preDelay) {
        data[i] = 0;
        continue;
      }
      const tail = (i - preDelay) / Math.max(1, length - preDelay);
      smooth += ((Math.random() * 2 - 1) - smooth) * 0.35;
      const fadeOut = Math.min(1, (1 - tail) * 20);
      data[i] = smooth * Math.exp(-5.5 * tail) * fadeOut;
    }
  }
  return impulse;
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
  private probeAnalyser_: AnalyserNode | null = null;
  private buses_: Record<StemName, StemBus> | null = null;
  private timer_: ReturnType<typeof setInterval> | null = null;
  private liveSources_: Map<AudioBufferSourceNode, number> = new Map();
  private activeVoices_: Partial<Record<string, { source: AudioBufferSourceNode; gain: GainNode; endAt: number }>> = {};
  private targetBpm_: number = 128;

  // Desired mixer state, applied lazily so UI can be touched before the
  // AudioContext exists (creating one pre-gesture would start suspended).
  private volumes_: Record<StemName, number> = { drums: 1, bass: 1, vocals: 1, other: 1 };
  private autoVolumes_: Record<StemName, number> = { drums: 1, bass: 1, vocals: 1, other: 1 };
  private mutes_: Record<StemName, boolean> = { drums: false, bass: false, vocals: false, other: false };
  private solos_: Record<StemName, boolean> = { drums: false, bass: false, vocals: false, other: false };

  // Glue bus state. glueBase_ is the manual target set by setStemGlue();
  // glueDuck_ is the automatic sidechain factor applied by updateGlue().
  private glueEnabled_ = true;
  private glueBase_: Record<StemName, number> = { drums: 1, bass: 1, vocals: 1, other: 1 };
  private glueDuck_: Record<StemName, number> = { drums: 1, bass: 1, vocals: 1, other: 1 };
  private glueTargets_: Record<StemName, number> = { drums: 1, bass: 1, vocals: 1, other: 1 };
  private reverbSends_: Record<StemName, number> = { ...DEFAULT_REVERB_SENDS };
  private readonly vocalTimeData_ = new Float32Array(GLUE_ANALYSER_FFT);
  private readonly drumTimeData_ = new Float32Array(GLUE_ANALYSER_FFT);
  private vocalRms_ = 0;
  private vocalPresence_ = 0;
  private drumShortRms_ = 0;
  private drumTransient_ = false;
  private drumTransientUntil_ = 0;

  get ready(): boolean {
    return this.ctx_ !== null && this.buses_ !== null;
  }

  /** AudioContext state for diagnostics ("running" | "suspended" | ...). */
  contextState(): string {
    return this.ctx_ === null ? "no-context" : this.ctx_.state;
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

    // Gain staging: four stems summing at unity hit +4 dBFS on drum
    // transients (measured peak 1.68 in the built app). Trim before the
    // compressor, then a limiter and a soft clipper so the destination never
    // sees more than ~0.98. Clipping was audible on every hit.
    const masterTrim = ctx.createGain();
    masterTrim.gain.value = 0.62;

    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -3;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.001;
    limiter.release.value = 0.08;

    // Transparent below 0.8, tanh-limited above; asymptote 1.0 so a hot
    // transient can never square off at the DAC.
    const shaper = ctx.createWaveShaper();
    const curve = new Float32Array(4096);
    const knee = 0.8;
    for (let i = 0; i < curve.length; i++) {
      const x = (i / (curve.length - 1)) * 2 - 1;
      const a = Math.abs(x);
      const y = a <= knee ? a : knee + (1 - knee) * Math.tanh((a - knee) / (1 - knee));
      curve[i] = Math.sign(x) * Math.min(1, y);
    }
    shaper.curve = curve;
    shaper.oversample = "4x";

    const masterAnalyser = ctx.createAnalyser();
    masterAnalyser.fftSize = 2048;
    masterAnalyser.smoothingTimeConstant = 0.8;

    // Hidden seam probe tap: a long time-domain window so a 10-20 fps main
    // thread still contains the block boundary. It is not connected onward.
    const probeAnalyser = ctx.createAnalyser();
    probeAnalyser.fftSize = 16384;
    probeAnalyser.smoothingTimeConstant = 0;

    master.connect(masterTrim);
    masterTrim.connect(limiter);
    limiter.connect(shaper);
    shaper.connect(masterAnalyser);
    shaper.connect(probeAnalyser);
    masterAnalyser.connect(ctx.destination);

    // One shared, short reverb for the whole kit. The impulse is generated in
    // JS (no asset) and the return is high-passed + trimmed so it reads as
    // glue, not as a wash.
    const convolver = ctx.createConvolver();
    convolver.normalize = true;
    convolver.buffer = createReverbImpulse(ctx);

    const reverbHighpass = ctx.createBiquadFilter();
    reverbHighpass.type = "highpass";
    reverbHighpass.frequency.value = REVERB_HIGHPASS_HZ;
    reverbHighpass.Q.value = 0.707;

    const reverbReturn = ctx.createGain();
    reverbReturn.gain.value = dbToGain(REVERB_RETURN_DB);

    convolver.connect(reverbHighpass);
    reverbHighpass.connect(reverbReturn);
    reverbReturn.connect(master);

    const buses = {} as Record<StemName, StemBus>;
    for (const stem of STEMS) {
      const input = ctx.createGain();
      const spec = STEM_FILTERS[stem];
      const filter = ctx.createBiquadFilter();
      filter.type = spec.type;
      filter.frequency.value = spec.frequency;
      filter.Q.value = spec.q;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = GLUE_ANALYSER_FFT;
      analyser.smoothingTimeConstant = 0.8;
      const glue = ctx.createGain();
      glue.gain.value = this.glueEnabled_ ? this.glueBase_[stem] : GLUE_MAX;
      this.glueDuck_[stem] = 1;
      this.glueTargets_[stem] = glue.gain.value;
      const reverbSend = ctx.createGain();
      reverbSend.gain.value = this.reverbSends_[stem];
      input.connect(filter);
      filter.connect(analyser);
      analyser.connect(glue);
      glue.connect(master);
      analyser.connect(reverbSend);
      reverbSend.connect(convolver);
      buses[stem] = { input, filter, analyser, glue, reverbSend };
    }

    this.ctx_ = ctx;
    this.master_ = master;
    this.masterAnalyser_ = masterAnalyser;
    this.probeAnalyser_ = probeAnalyser;
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
      const eff = audible ? (this.volumes_[stem] * this.autoVolumes_[stem]) : 0;
      bus.input.gain.setTargetAtTime(eff, t, 0.02);
    }
  }

  /**
   * Automated gentle mix level for transcendent section and phrase crossfades.
   * Modulates bus input gain smoothly with exponential asymptotic interpolation.
   */
  setStemMixLevel(stem: StemName, level: number, rampTimeSec: number = 0.08): void {
    this.autoVolumes_[stem] = Math.max(0, Math.min(1.5, level));
    if (this.ctx_ === null || this.buses_ === null) return;
    const bus = this.buses_[stem];
    const anySolo = STEMS.some((s) => this.solos_[s]);
    const audible = !this.mutes_[stem] && (!anySolo || this.solos_[stem]);
    const targetGain = audible ? (this.volumes_[stem] * this.autoVolumes_[stem]) : 0;
    const t = this.ctx_.currentTime;
    try {
      bus.input.gain.setTargetAtTime(targetGain, t, Math.max(0.01, rampTimeSec));
    } catch {
      // Safe fallback
    }
  }

  getStemMixLevel(stem: StemName): number {
    return this.autoVolumes_[stem];
  }

  // ---- glue bus ---------------------------------------------------------

  /**
   * Manual glue-base gain for one stem. updateGlue() may duck below this base
   * while enabled, but the stem input gain (setStemMixLevel) is never touched.
   */
  setStemGlue(stem: StemName, gain: number, rampSec: number = GLUE_RAMP_SEC): void {
    this.glueBase_[stem] = clampGlue(gain);
    const ctx = this.ctx_;
    if (!this.glueEnabled_ || ctx === null || this.buses_ === null) return;
    const target = clampGlue(this.glueBase_[stem] * this.glueDuck_[stem]);
    this.scheduleGlue(stem, target, ctx.currentTime, rampSec);
  }

  setGlueEnabled(enabled: boolean): void {
    this.glueEnabled_ = enabled;
    const ctx = this.ctx_;
    const buses = this.buses_;
    if (ctx === null || buses === null) return;
    const t = ctx.currentTime;
    for (const stem of STEMS) {
      if (!enabled) this.glueDuck_[stem] = 1;
      const target = enabled
        ? clampGlue(this.glueBase_[stem] * this.glueDuck_[stem])
        : GLUE_MAX;
      this.scheduleGlue(stem, target, t, GLUE_RAMP_SEC);
    }
    if (!enabled) {
      this.vocalPresence_ = 0;
      this.drumTransient_ = false;
      this.drumTransientUntil_ = 0;
    }
  }

  /**
   * Per-frame automatic glue. Reads the vocal analyser for presence and the
   * drum analyser for transients, then ducks only the glue gains:
   *   vocals -> other/bass  (up to 1.5 dB, 20 ms attack / 250 ms release)
   *   drums  -> bass        (up to 2.5 dB, 8 ms attack / 120 ms release)
   * Never touches input.gain, so section mixing remains authoritative.
   */
  updateGlue(now: number): void {
    const ctx = this.ctx_;
    const buses = this.buses_;
    if (ctx === null || buses === null || !this.glueEnabled_) return;

    const t = Number.isFinite(now) ? Math.max(now, ctx.currentTime) : ctx.currentTime;

    buses.vocals.analyser.getFloatTimeDomainData(this.vocalTimeData_);
    const vocalRms = rmsOf(this.vocalTimeData_);
    this.vocalRms_ = vocalRms;
    const vocalPresence = clamp01(
      (vocalRms - VOCAL_RMS_FLOOR) / (VOCAL_RMS_FULL - VOCAL_RMS_FLOOR),
    );
    this.vocalPresence_ = vocalPresence;

    buses.drums.analyser.getFloatTimeDomainData(this.drumTimeData_);
    const drumFullRms = rmsOf(this.drumTimeData_);
    const shortStart = this.drumTimeData_.length - KICK_SHORT_WINDOW;
    const drumShortRms = rmsOf(this.drumTimeData_, shortStart);
    const drumShortPeak = peakOf(this.drumTimeData_, shortStart);
    this.drumShortRms_ = drumShortRms;
    const transientDetected =
      drumShortRms > 0.012 &&
      (drumShortRms > drumFullRms * 1.6 || drumShortPeak > Math.max(0.35, drumFullRms * 4));
    if (transientDetected) {
      this.drumTransientUntil_ = Math.max(
        this.drumTransientUntil_,
        t + KICK_TRANSIENT_HOLD_SEC,
      );
    }
    const drumTransient = t < this.drumTransientUntil_;
    this.drumTransient_ = drumTransient;

    const vocalDuckDb = VOCAL_DUCK_DB * vocalPresence;
    const bassDuckDb = vocalDuckDb + (drumTransient ? KICK_DUCK_DB : 0);
    this.glueDuck_.other = dbToGain(-vocalDuckDb);
    this.glueDuck_.bass = dbToGain(-bassDuckDb);

    const otherTarget = clampGlue(this.glueBase_.other * this.glueDuck_.other);
    const bassTarget = clampGlue(this.glueBase_.bass * this.glueDuck_.bass);

    const otherTau =
      otherTarget < this.glueTargets_.other ? VOCAL_ATTACK_SEC : VOCAL_RELEASE_SEC;
    const kickRecent = t < this.drumTransientUntil_ + KICK_RELEASE_SEC;
    const bassTau =
      bassTarget < this.glueTargets_.bass
        ? drumTransient
          ? KICK_ATTACK_SEC
          : VOCAL_ATTACK_SEC
        : kickRecent
          ? KICK_RELEASE_SEC
          : VOCAL_RELEASE_SEC;
    this.scheduleGlue("other", otherTarget, t, otherTau);
    this.scheduleGlue("bass", bassTarget, t, bassTau);
  }

  private scheduleGlue(stem: StemName, target: number, t: number, tau: number): void {
    const buses = this.buses_;
    if (buses === null) return;
    // setTargetAtTime already approaches asymptotically; avoid re-scheduling
    // an unchanged target every animation frame.
    if (Math.abs(target - this.glueTargets_[stem]) < 1e-4) return;
    try {
      buses[stem].glue.gain.setTargetAtTime(target, t, Math.max(0.005, tau));
    } catch {
      // Automation is best-effort; never let glue stop the transport.
    }
    this.glueTargets_[stem] = target;
  }

  glueState(): GlueState {
    const gains = {} as Record<StemName, number>;
    const base = {} as Record<StemName, number>;
    const duck = {} as Record<StemName, number>;
    const buses = this.buses_;
    for (const stem of STEMS) {
      base[stem] = this.glueBase_[stem];
      duck[stem] = this.glueDuck_[stem];
      gains[stem] = buses !== null
        ? buses[stem].glue.gain.value
        : this.glueEnabled_
          ? this.glueBase_[stem]
          : GLUE_MAX;
    }
    return {
      enabled: this.glueEnabled_,
      gains,
      base,
      duck,
      vocalRms: this.vocalRms_,
      vocalPresence: this.vocalPresence_,
      drumShortRms: this.drumShortRms_,
      drumTransient: this.drumTransient_,
    };
  }

  // ---- shared reverb ----------------------------------------------------

  /** Set one stem's send into the shared short reverb. Defaults: other 0.08,
   * vocals 0.10, drums/bass 0. */
  setReverbSend(stem: StemName, amount: number, rampSec: number = GLUE_RAMP_SEC): void {
    const clamped = clamp01(amount);
    this.reverbSends_[stem] = clamped;
    const ctx = this.ctx_;
    const buses = this.buses_;
    if (ctx === null || buses === null) return;
    try {
      buses[stem].reverbSend.gain.setTargetAtTime(
        clamped,
        ctx.currentTime,
        Math.max(0.005, rampSec),
      );
    } catch {
      // Safe fallback
    }
  }

  getReverbSend(stem: StemName): number {
    return this.reverbSends_[stem];
  }

  /** Vocal reverb tail at a block boundary: bump the send for ~0.35 s so the
   *  outgoing vocal dissolves while the incoming one attacks.  Scheduled with
   *  sample-accurate ramps at absolute audio time `at` (no setTimeout, no
   *  DOM); the stored user send never changes.  No-op before the ctx exists. */
  scheduleReverbTail(
    stem: StemName,
    at: number,
    tailAmount: number = TRANSITION_REVERB_TAIL,
    tailSec: number = TRANSITION_REVERB_TAIL_SEC,
  ): void {
    const ctx = this.ctx_;
    const buses = this.buses_;
    if (ctx === null || buses === null) return;
    const startAt = Math.max(at, ctx.currentTime + 0.005);
    const endAt = startAt + Math.max(0.05, Math.min(3, tailSec));
    const baseline = clamp01(this.reverbSends_[stem]);
    const tail = Math.max(baseline, clamp01(tailAmount));
    if (tail - baseline < 0.02) return;
    try {
      const param = buses[stem].reverbSend.gain;
      param.cancelScheduledValues(startAt);
      param.setValueAtTime(baseline, startAt);
      param.linearRampToValueAtTime(tail, startAt + 0.03);
      param.setValueAtTime(tail, Math.max(startAt + 0.03, endAt - 0.10));
      param.linearRampToValueAtTime(baseline, endAt);
    } catch {
      // Automation is best-effort; the gain hand-off stands alone.
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

  /** Real-time reticule timbre filter steering for an individual stem bus. */
  setStemFilter(stem: StemName, freqHz: number, q?: number): void {
    if (this.ctx_ === null || this.buses_ === null) return;
    const bus = this.buses_[stem];
    if (!bus) return;
    const clampedFreq = Math.max(20, Math.min(22000, freqHz));
    const t = this.ctx_.currentTime;
    bus.filter.frequency.setTargetAtTime(clampedFreq, t, 0.04);
    if (q !== undefined && Number.isFinite(q)) {
      const clampedQ = Math.max(0.1, Math.min(20, q));
      bus.filter.Q.setTargetAtTime(clampedQ, t, 0.04);
    }
  }

  /** Retrieve the live stem filter parameters. */
  getStemFilter(stem: StemName): { frequency: number; q: number } | null {
    if (this.ctx_ === null || this.buses_ === null) return null;
    const bus = this.buses_[stem];
    if (!bus) return null;
    return {
      frequency: bus.filter.frequency.value,
      q: bus.filter.Q.value,
    };
  }

  getMasterAnalyser(): AnalyserNode {
    this.ensureCtx();
    if (this.masterAnalyser_ === null) throw new Error("AudioEngine master analyser not initialised");
    return this.masterAnalyser_;
  }

  /** Long-window analyser for the hidden transition probe. */
  getProbeAnalyser(): AnalyserNode {
    this.ensureCtx();
    if (this.probeAnalyser_ === null) throw new Error("AudioEngine probe analyser not initialised");
    return this.probeAnalyser_;
  }

  /** Current audio-clock time, or 0 when the context does not exist yet. */
  now(): number {
    return this.ctx_ !== null ? this.ctx_.currentTime : 0;
  }

  /** End time of the currently tracked voice, if one is still active. */
  getVoiceEndAt(voiceKey: string): number | null {
    const voice = this.activeVoices_[voiceKey];
    return voice ? voice.endAt : null;
  }

  /** Decode fetched audio bytes (WAV legacy or Ogg Opus) into an AudioBuffer.
   * decodeAudioData sniffs the container; anything it rejects falls back to
   * the shared-slice synth path at the call site. */
  async decodeAudio(bytes: ArrayBuffer): Promise<AudioBuffer> {
    const ctx = this.ensureCtx();
    // decodeAudioData detaches its input; decode a copy instead.
    const copy = bytes.slice(0);
    return await ctx.decodeAudioData(copy);
  }

  /** Stable stereo position for a voice (mono slices need explicit width). */
  private static panFor(stem: StemName, key: string): number {
    if (stem === "bass") return 0;
    let h = 0;
    for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
    const side = (h & 1) === 0 ? -1 : 1;
    if (stem === "drums") {
      if (/kick|snare/i.test(key)) return 0;
      return side * 0.18;
    }
    if (stem === "vocals") return side * 0.22;
    return side * 0.38;
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
    const requestedFadeInMs =
      typeof opts.fadeInMs === "number" && Number.isFinite(opts.fadeInMs)
        ? Math.max(0, opts.fadeInMs)
        : requestedFadeMs;
    const requestedFadeOutMs =
      typeof opts.fadeOutMs === "number" && Number.isFinite(opts.fadeOutMs)
        ? Math.max(0, opts.fadeOutMs)
        : requestedFadeMs;

    const startAt = Math.max(when + (typeof opts.whenOffsetMs === "number" && Number.isFinite(opts.whenOffsetMs) ? Math.max(-15, Math.min(15, opts.whenOffsetMs)) / 1000 : 0), ctx.currentTime + 0.005);
    const naturalDurSec = buffer.duration / rate;
    const durSec =
      typeof opts.duration === "number" && Number.isFinite(opts.duration) && opts.duration > 0 && opts.duration < naturalDurSec
        ? opts.duration
        : naturalDurSec;

    // Choke voice tracking: allows independent drum tracks (kick, snare, hat) while choking open hats or previous bass
    const vKey = opts.voiceKey || stem;
    const chokes = [vKey, ...(opts.chokeKeys || [])];

    // Dynamic, click-free crossfade release duration:
    // Drums choke quickly (12-18ms for one-shots, up to 120ms for phrase loops) to keep transients tight.
    // Tonal stems (bass, other, vocals) crossfade gracefully (25-250ms) to prevent sub-bass phase tears and pops.
    const isLongLoop = durSec >= 1.5;
    const crossfadeSec = stem === "drums"
      ? Math.max(0.010, Math.min(requestedFadeMs / 1000, isLongLoop ? 0.120 : 0.018))
      : Math.max(0.025, Math.min(requestedFadeMs / 1000, isLongLoop ? 0.250 : 0.048));

    for (const cKey of chokes) {
      const prev = this.activeVoices_[cKey];
      if (prev && prev.endAt > startAt) {
        const stopAt = startAt + crossfadeSec;
        try {
          prev.gain.gain.cancelScheduledValues(startAt);
          if (typeof (prev.gain.gain as any).cancelAndHoldAtTime === "function") {
            (prev.gain.gain as any).cancelAndHoldAtTime(startAt);
          }
          prev.gain.gain.setValueAtTime(prev.gain.gain.value, startAt);
          prev.gain.gain.linearRampToValueAtTime(0.0001, stopAt);
          prev.source.stop(stopAt + 0.02);
        } catch {
          // Safe if already stopped
        }
      }
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = rate;
    source.detune.value = detune;

    const vGain = typeof opts.gain === "number" && Number.isFinite(opts.gain) ? Math.max(0, Math.min(1.5, opts.gain)) : 1.0;
    const sliceGain = ctx.createGain();
    sliceGain.gain.value = 0.0;
    // Every shipped slice is mono, so the old chain produced a perfectly
    // correlated (dead centre) mix -- measured L/R correlation 1.000. Place
    // each voice in the field instead of hard-panning the file: kick/snare
    // and bass stay centred, hats/pads/vocals get a stable offset keyed by
    // the slice id so a given voice never jumps around.
    const panner = ctx.createStereoPanner();
    panner.pan.value = AudioEngine.panFor(stem, vKey);
    source.connect(sliceGain);
    sliceGain.connect(panner);
    panner.connect(buses[stem].input);

    const endAt = startAt + durSec;
    if (requestedFadeMs > 0 && durSec > 0.01) {
      // De-clicking micro fade-in prevents DC offset pop on slice attack.
      // A separate shorter seam fade keeps block joins from notching.
      const fadeInSec = Math.max(0.002, Math.min(requestedFadeInMs / 1000, durSec / 4));
      // Musical fade-out guarantees click-free release decay.
      const fadeOutSec = stem === "drums"
        ? Math.max(0.005, Math.min(requestedFadeOutMs / 1000, 0.020, durSec / 3))
        : Math.max(0.015, Math.min(requestedFadeOutMs / 1000, 0.045, durSec / 3));

      sliceGain.gain.setValueAtTime(0.0001, startAt);
      sliceGain.gain.linearRampToValueAtTime(vGain, startAt + fadeInSec);
      if (endAt - fadeOutSec > startAt + fadeInSec) {
        sliceGain.gain.setValueAtTime(vGain, endAt - fadeOutSec);
      }
      sliceGain.gain.linearRampToValueAtTime(0.0001, endAt);
    } else {
      sliceGain.gain.setValueAtTime(vGain, startAt);
    }
    this.activeVoices_[vKey] = { source, gain: sliceGain, endAt };
    source.onended = () => {
      this.liveSources_.delete(source);
      if (this.activeVoices_[vKey]?.source === source) {
        delete this.activeVoices_[vKey];
      }
      try {
        source.disconnect();
        sliceGain.disconnect();
        panner.disconnect();
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
    this.probeAnalyser_ = null;
    this.buses_ = null;
    this.drumTransient_ = false;
    this.drumTransientUntil_ = 0;
    this.vocalPresence_ = 0;
    if (ctx !== null) {
      void ctx.close().catch(() => undefined);
    }
  }
}
