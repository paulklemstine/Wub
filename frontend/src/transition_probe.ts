// Hidden transition-seam probe.
//
// main.ts marks scheduled kit-block boundaries, then calls update() from the
// existing animation loop. The probe takes one analyser snapshot just after
// each boundary (plus a pre-boundary spectrum for flux), computes a few cheap
// seam metrics, and keeps them in a capped global log for offline inspection.
// There is deliberately no UI, no ScriptProcessor and no per-frame capture.
import { ENGINE_SAMPLE_RATE, type AudioEngine } from "./audio_engine";
import type { StemName } from "./types";

const STEM_ORDER: readonly StemName[] = ["drums", "bass", "vocals", "other"];

const LOG_KEY = "__wubTransitionLog";
const LOG_CAP = 500;
const MAX_PENDING = 16;
const MAX_JUMP_SAMPLES = 2000;

const RMS_WINDOW_SEC = 0.02;
const BEFORE_CAPTURE_LEAD_SEC = 0.04;
const POST_CAPTURE_DELAY_SEC = 0.02;
const MAX_CAPTURE_LATE_SEC = 0.3;
const SEAM_JUMP_THRESHOLD = 0.85;
const DIP_DB_THRESHOLD = 1.5;

export interface TransitionBoundary {
  at: number;
  blockBars: number;
  changed: readonly StemName[];
  bridge: boolean;
  rotation: boolean;
  clusterId: string;
  /** Planned arc energy for this block, 0..1. */
  targetEnergy?: number;
  tracks?: Partial<Record<StemName, string | null>>;
  pin?: boolean;
}

export interface TransitionCapture {
  /** Boundary timestamp on the AudioContext clock. */
  at: number;
  /** When the analyser snapshot was actually taken. */
  capturedAt: number;
  blockBars: number;
  changed: StemName[];
  bridge: boolean;
  rotation: boolean;
  clusterId: string;
  targetEnergy?: number;
  tracks?: Partial<Record<StemName, string | null>>;
  pin?: boolean;
  /** Max |x[n] - x[n-1]| in the 2048-sample master window. */
  maxJump: number;
  preRms: number;
  postRms: number;
  /** Positive when the 20 ms after the boundary is quieter than before. */
  rmsDipDb: number;
  /** Raw L1 distance between pre/post analyser dB spectra (all bins). */
  spectralFlux: number;
  seam: boolean;
  dip: boolean;
  suspicious: boolean;
}

export interface TransitionProbeSummary {
  marked: number;
  captured: number;
  missed: number;
  pending: number;
  seamEvents: number;
  dipEvents: number;
  suspiciousEvents: number;
  p50MaxJump: number;
  p95MaxJump: number;
}

interface PendingBoundary {
  info: TransitionBoundary;
  beforeFreq: Float32Array | null;
}

interface TransitionLogGlobal {
  __wubTransitionLog?: TransitionCapture[];
}

function getTransitionLog(): TransitionCapture[] {
  const host = globalThis as unknown as TransitionLogGlobal;
  const existing = host[LOG_KEY];
  if (Array.isArray(existing)) return existing;
  const log: TransitionCapture[] = [];
  host[LOG_KEY] = log;
  return log;
}

function finiteDb(value: number): number {
  if (!Number.isFinite(value)) return -160;
  return Math.max(-200, Math.min(0, value));
}

function maxSampleJump(data: Float32Array, start: number = 0, end: number = data.length): number {
  const from = Math.max(1, Math.min(data.length, start));
  const to = Math.max(from, Math.min(data.length, end));
  let max = 0;
  for (let i = from; i < to; i++) {
    const delta = Math.abs(data[i] - data[i - 1]);
    if (delta > max) max = delta;
  }
  return max;
}

function rmsOf(data: Float32Array, start: number, end: number): number {
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

function spectralFlux(before: Float32Array | null, after: Float32Array): number {
  if (before === null) return 0;
  const n = Math.min(before.length, after.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.abs(finiteDb(before[i]) - finiteDb(after[i]));
  }
  return sum;
}

function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const rank = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, rank))];
}

function formatTracks(tracks: Partial<Record<StemName, string | null>>): string {
  const parts: string[] = [];
  for (const stem of STEM_ORDER) {
    const id = tracks[stem];
    if (id !== undefined) parts.push(`${stem}:${id ?? "-"}`);
  }
  return parts.length > 0 ? parts.join(",") : "n/a";
}

export class TransitionProbe {
  private pending_: PendingBoundary[] = [];
  private markedCount_ = 0;
  private capturedCount_ = 0;
  private missedCount_ = 0;
  private seamCount_ = 0;
  private dipCount_ = 0;
  private suspiciousCount_ = 0;
  private maxJumps_: number[] = [];
  private lastCapturedAt_ = Number.NEGATIVE_INFINITY;

  /** Queue an analyser capture around `info.at` (AudioContext seconds). */
  markBoundary(info: TransitionBoundary): void {
    if (!Number.isFinite(info.at)) return;
    if (Math.abs(info.at - this.lastCapturedAt_) < 0.02) return;
    if (this.pending_.some((pending) => Math.abs(pending.info.at - info.at) < 0.02)) {
      return;
    }
    this.markedCount_++;
    this.pending_.push({ info, beforeFreq: null });
    if (this.pending_.length > MAX_PENDING) {
      this.pending_.shift();
      this.missedCount_++;
    }
  }

  /**
   * Called once per animation frame with the AudioContext clock. Only queued
   * boundaries do any work; the common case is a few array-length checks.
   */
  update(now: number, engine: AudioEngine): void {
    if (this.pending_.length === 0 || !Number.isFinite(now) || !engine.ready) return;

    const analyser = engine.getProbeAnalyser();
    const fftSize = analyser.fftSize;
    const binCount = analyser.frequencyBinCount;
    const sampleRate = ENGINE_SAMPLE_RATE;

    for (let i = this.pending_.length - 1; i >= 0; i--) {
      const pending = this.pending_[i];
      const info = pending.info;
      const delta = now - info.at;

      // Still too early to care.
      if (delta < -BEFORE_CAPTURE_LEAD_SEC) continue;

      // Grab the pre-boundary spectrum once. If the frame skipped just past
      // the boundary, still take it; the post capture follows next frame.
      if (pending.beforeFreq === null && delta < POST_CAPTURE_DELAY_SEC) {
        const before = new Float32Array(binCount);
        analyser.getFloatFrequencyData(before);
        pending.beforeFreq = before;
      }

      // Wait until the 20 ms post-boundary window can be fully inside the
      // analyser's 2048-sample time buffer.
      if (delta < POST_CAPTURE_DELAY_SEC) continue;

      // A stalled tab can push the boundary out of the analyser window.
      if (delta > MAX_CAPTURE_LATE_SEC) {
        this.pending_.splice(i, 1);
        this.missedCount_++;
        continue;
      }

      const timeData = new Float32Array(fftSize);
      analyser.getFloatTimeDomainData(timeData);
      const afterFreq = new Float32Array(binCount);
      analyser.getFloatFrequencyData(afterFreq);

      const boundaryIndex = Math.max(
        0,
        Math.min(timeData.length, Math.round(timeData.length - delta * sampleRate)),
      );
      const maxJump = maxSampleJump(
        timeData,
        boundaryIndex - Math.round(0.03 * sampleRate),
        boundaryIndex + Math.round(0.03 * sampleRate),
      );
      const windowSamples = Math.max(1, Math.round(RMS_WINDOW_SEC * sampleRate));
      const preSamples = Math.min(windowSamples, boundaryIndex);
      const postSamples = Math.min(windowSamples, timeData.length - boundaryIndex);
      const preRms =
        preSamples > 0 ? rmsOf(timeData, boundaryIndex - preSamples, boundaryIndex) : 0;
      const postRms =
        postSamples > 0
          ? rmsOf(timeData, boundaryIndex, boundaryIndex + postSamples)
          : 0;
      const rmsDipDb =
        preRms > 1e-9 && postRms > 1e-9
          ? 20 * Math.log10(preRms / postRms)
          : preRms > 1e-9
            ? 60
            : 0;

      const flux = spectralFlux(pending.beforeFreq, afterFreq);
      const targetEnergy = info.targetEnergy ?? 0.65;
      const seam = maxJump > SEAM_JUMP_THRESHOLD;
      // A planned drop to an intro/breakdown is an intentional level change,
      // not a seam; only flag a dip when the arc is not asking for one.
      const intentionalDrop = targetEnergy < 0.45 && rmsDipDb < 6;
      const dip = rmsDipDb > DIP_DB_THRESHOLD && !intentionalDrop;
      const suspicious = seam || dip;

      const entry: TransitionCapture = {
        at: info.at,
        capturedAt: now,
        blockBars: info.blockBars,
        changed: Array.from(info.changed),
        bridge: info.bridge,
        rotation: info.rotation,
        clusterId: info.clusterId,
        targetEnergy: info.targetEnergy,
        tracks: info.tracks ? { ...info.tracks } : undefined,
        pin: info.pin,
        maxJump,
        preRms,
        postRms,
        rmsDipDb,
        spectralFlux: flux,
        seam,
        dip,
        suspicious,
      };

      const log = getTransitionLog();
      log.push(entry);
      if (log.length > LOG_CAP) log.splice(0, log.length - LOG_CAP);

      this.capturedCount_++;
      this.lastCapturedAt_ = info.at;
      this.maxJumps_.push(maxJump);
      if (this.maxJumps_.length > MAX_JUMP_SAMPLES) {
        this.maxJumps_.splice(0, this.maxJumps_.length - MAX_JUMP_SAMPLES);
      }
      if (seam) this.seamCount_++;
      if (dip) this.dipCount_++;
      if (suspicious) this.suspiciousCount_++;

      if (suspicious) {
        console.info(
          `[TransitionProbe] suspicious seam at ${info.at.toFixed(3)}s ` +
            `block=${info.blockBars}bars ` +
            `changed=${info.changed.length > 0 ? info.changed.join("+") : "none"} ` +
            `target=${targetEnergy.toFixed(2)} ` +
            `bridge=${info.bridge} rotation=${info.rotation} ` +
            `tracks=${info.tracks ? formatTracks(info.tracks) : "n/a"} ` +
            `maxJump=${maxJump.toFixed(3)} dip=${rmsDipDb.toFixed(2)}dB ` +
            `flux=${flux.toFixed(1)}`,
        );
      }

      this.pending_.splice(i, 1);
    }
  }

  summary(): TransitionProbeSummary {
    return {
      marked: this.markedCount_,
      captured: this.capturedCount_,
      missed: this.missedCount_,
      pending: this.pending_.length,
      seamEvents: this.seamCount_,
      dipEvents: this.dipCount_,
      suspiciousEvents: this.suspiciousCount_,
      p50MaxJump: percentile(this.maxJumps_, 50),
      p95MaxJump: percentile(this.maxJumps_, 95),
    };
  }
}
