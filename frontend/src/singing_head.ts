// -----------------------------------------------------------------------------
// Singing Head: Voice Workshop Mid-Sagittal Side View & Interactive 3D View
// Placed at the bottom of the KG-505 interface and driven in real time by the
// reverse-trained acoustic-to-articulatory model on the vocal instrument stem.
// -----------------------------------------------------------------------------

import type { AudioEngine } from "./audio_engine";
import type { GrooveboxMeta } from "./main";
import {
  SINGING_FFT_SIZE,
  SINGING_VOWELS,
  SingingHeadInference,
  isValidSingingHeadModel,
  type SingingHeadArticulation,
  type SingingVowel,
} from "./singing_head_model";
import {
  VocalFieldState,
  articulationToTract,
  createVocalFieldState,
  defaultAnatomy,
  defaultTeeth,
  updateVocalFieldState,
} from "./vocal_sim/articulation_bridge";
import { type FieldMode } from "./vocal_sim/canvas_colors";
import { SideHeadView } from "./vocal_sim/side_head_view";
import { type CameraPreset3D, Tract3DView } from "./vocal_sim/tract_3d_view";
import { VocalTractSynthesizer } from "./vocal_sim/tract_synth";
import { computeGeometry, type TractGeometry } from "./vocal_sim/tract/geometry";
import {
  type Articulators,
  ARTICULATOR_KEYS,
  ARTICULATOR_RANGES,
  DEFAULT_ARTICULATORS,
} from "./vocal_sim/tract/types";

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function clampRange(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

interface SpringChannel {
  pos: number;
  vel: number;
}

function makeSpring(initial: number): SpringChannel {
  return { pos: initial, vel: 0 };
}

/**
 * Exact critically-damped 2nd-order spring-damper (zeta = 1.0) with asymmetric
 * natural frequencies. Attenuates high-frequency frame jitter at O(1/w^2) and
 * produces smooth, C1-continuous human articulation without overshoot.
 */
function stepSpringAsym(
  ch: SpringChannel,
  target: number,
  dt: number,
  attackOmega: number,
  releaseOmega: number
): number {
  const omega = target > ch.pos ? attackOmega : releaseOmega;
  const e = Math.exp(-omega * dt);
  const d = ch.pos - target;
  const c = ch.vel + omega * d;
  ch.pos = target + (d + c * dt) * e;
  ch.vel = (ch.vel - omega * c * dt) * e;
  return ch.pos;
}

function smoothAsym(cur: number, target: number, dt: number, attackTau: number, releaseTau: number): number {
  const tau = target > cur ? attackTau : releaseTau;
  const alpha = 1 - Math.exp(-dt / Math.max(0.001, tau));
  return cur + (target - cur) * alpha;
}

function hzToNoteName(hz: number): string {
  if (hz <= 30) return "--";
  const midi = Math.round(69 + 12 * Math.log2(hz / 440));
  const name = NOTE_NAMES[((midi % 12) + 12) % 12];
  const octave = Math.floor(midi / 12) - 1;
  return `${name}${octave}`;
}

export type SingingHeadViewMode = "side" | "3d";
export type SingingHeadAudioMode = "track" | "synth";
export type SingingHeadInputMode = "vocals" | "mix";

export class SingingHeadPanel {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private statusBadgeEl: HTMLElement | null = null;
  private modelBadgeEl: HTMLElement | null = null;
  private host3dEl: HTMLElement | null = null;
  private controls3dEl: HTMLElement | null = null;

  private audioEngine: AudioEngine | null = null;
  private audioSourceMode: SingingHeadAudioMode = "track";
  private inputMode: SingingHeadInputMode = "vocals";
  private readonly tractSynth = new VocalTractSynthesizer();

  private readonly inference = new SingingHeadInference();
  private readonly silentFrame = new Float32Array(SINGING_FFT_SIZE);

  // Voice Workshop mid-sagittal & 3D tract state
  private viewMode: SingingHeadViewMode = "side";
  private fieldMode: FieldMode = "pressure";
  private showLabels = true;
  private readonly anatomy = defaultAnatomy();
  private readonly teeth = defaultTeeth();
  private readonly tractArt: Articulators = { ...DEFAULT_ARTICULATORS };
  private tractGeom: TractGeometry = computeGeometry(this.tractArt, this.anatomy, this.teeth);
  private readonly fieldState: VocalFieldState = createVocalFieldState();
  private readonly sideView = new SideHeadView(`${import.meta.env.BASE_URL}anatomy/head.jpg`);
  private tract3dView: Tract3DView | null = null;

  // Smoothed articulatory state (reverse-trained from vocal stem audio)
  private aperture = 0;
  private width = 0.5;
  private roundness = 0;
  private jawDrop = 0;
  private tongueHeight = 0.55;
  private pitchNorm = 0.42;
  private vibrato = 0;
  private tension = 0;
  private breath = 0;
  private vowelProbs: number[] = [0.2, 0.2, 0.2, 0.2, 0.2];
  private topVowel: SingingVowel = "AH";

  private lastF0Hz = 0;
  private smoothedRms = 0;
  private slowRms = 0;
  private smoothedVoicing = 0;
  private lastAudioSec = 0;
  private modelLoadStarted = false;

  // Reusable smoothed articulation snapshot for tract bridge
  private readonly smoothedTract: Articulators = { ...DEFAULT_ARTICULATORS };
  private readonly scalarSprings = {
    aperture: makeSpring(0),
    width: makeSpring(0.5),
    roundness: makeSpring(0),
    jawDrop: makeSpring(0),
    tongueHeight: makeSpring(0.55),
    pitchNorm: makeSpring(0.42),
    vibrato: makeSpring(0),
    tension: makeSpring(0),
    breath: makeSpring(0),
  };
  private readonly vowelSprings: SpringChannel[] = [
    makeSpring(0.2),
    makeSpring(0.2),
    makeSpring(0.2),
    makeSpring(0.2),
    makeSpring(0.2),
  ];
  private readonly tractSprings: Record<keyof Articulators, SpringChannel> = {
    jaw: makeSpring(DEFAULT_ARTICULATORS.jaw),
    tongueX: makeSpring(DEFAULT_ARTICULATORS.tongueX),
    tongueY: makeSpring(DEFAULT_ARTICULATORS.tongueY),
    tongueRoot: makeSpring(DEFAULT_ARTICULATORS.tongueRoot),
    tipX: makeSpring(DEFAULT_ARTICULATORS.tipX),
    tipY: makeSpring(DEFAULT_ARTICULATORS.tipY),
    groove: makeSpring(DEFAULT_ARTICULATORS.groove),
    tongueSides: makeSpring(DEFAULT_ARTICULATORS.tongueSides),
    lipOpen: makeSpring(DEFAULT_ARTICULATORS.lipOpen),
    lipRound: makeSpring(DEFAULT_ARTICULATORS.lipRound),
    lipProtrude: makeSpring(DEFAULT_ARTICULATORS.lipProtrude),
    lipTuck: makeSpring(DEFAULT_ARTICULATORS.lipTuck),
    velum: makeSpring(DEFAULT_ARTICULATORS.velum),
    larynx: makeSpring(DEFAULT_ARTICULATORS.larynx),
  };
  private readonly smoothedArt: SingingHeadArticulation = {
    aperture: 0,
    width: 0.5,
    roundness: 0,
    jawDrop: 0,
    tongueHeight: 0.55,
    pitchNorm: 0.42,
    vibrato: 0,
    tension: 0,
    breath: 0,
    vowelProbs: this.vowelProbs,
    topVowel: "AH",
    tract: undefined,
    rms: 0,
    f0Hz: 0,
    voicing: 0,
    centroidHz: 0,
    trained: false,
  };

  public initDOM(
    canvasId: string = "singing-head-canvas",
    statusBadgeId: string = "singing-head-status",
    modelBadgeId: string = "singing-head-model-badge"
  ): void {
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
    if (this.canvas) {
      this.ctx = this.canvas.getContext("2d");
    }
    this.statusBadgeEl = document.getElementById(statusBadgeId);
    this.modelBadgeEl = document.getElementById(modelBadgeId);
    this.host3dEl = document.getElementById("singing-head-3d-host");
    this.controls3dEl = document.getElementById("sh-3d-controls");
    if (this.host3dEl) {
      this.tract3dView = new Tract3DView(this.host3dEl);
    }

    this.bindToolbarControls();
    void this.loadModel();
    this.render(null, this.silentFrame, 0, false, 126, 0.016);
  }

  public bindAudioEngine(engine: AudioEngine): void {
    this.audioEngine = engine;
    engine.setVocalSynthMode(this.audioSourceMode === "synth");
    if (this.audioSourceMode === "synth" && engine.ready) {
      const target = engine.getVocalSynthTarget();
      this.tractSynth.attach(target.ctx, target.destination);
      this.tractSynth.setEnabled(true);
    }
  }

  public setAudioSourceMode(mode: SingingHeadAudioMode): void {
    this.audioSourceMode = mode;
    const isSynth = mode === "synth";
    document.getElementById("sh-audio-track")?.classList.toggle("active", !isSynth);
    document.getElementById("sh-audio-synth")?.classList.toggle("active", isSynth);
    this.tractSynth.setEnabled(isSynth);
    if (this.audioEngine) {
      this.audioEngine.setVocalSynthMode(isSynth);
      if (isSynth && this.audioEngine.ready) {
        const target = this.audioEngine.getVocalSynthTarget();
        this.tractSynth.attach(target.ctx, target.destination);
      }
    }
  }

  public getAudioSourceMode(): SingingHeadAudioMode {
    return this.audioSourceMode;
  }

  public setInputMode(mode: SingingHeadInputMode): void {
    this.inputMode = mode;
    const isMix = mode === "mix";
    document.getElementById("sh-input-vocals")?.classList.toggle("active", !isMix);
    document.getElementById("sh-input-mix")?.classList.toggle("active", isMix);
  }

  public getInputMode(): SingingHeadInputMode {
    return this.inputMode;
  }

  private bindToolbarControls(): void {
    const audioTrackBtn = document.getElementById("sh-audio-track");
    const audioSynthBtn = document.getElementById("sh-audio-synth");
    audioTrackBtn?.addEventListener("click", () => this.setAudioSourceMode("track"));
    audioSynthBtn?.addEventListener("click", () => this.setAudioSourceMode("synth"));

    const inputVocalsBtn = document.getElementById("sh-input-vocals");
    const inputMixBtn = document.getElementById("sh-input-mix");
    inputVocalsBtn?.addEventListener("click", () => this.setInputMode("vocals"));
    inputMixBtn?.addEventListener("click", () => this.setInputMode("mix"));

    const sideBtn = document.getElementById("sh-mode-side");
    const view3dBtn = document.getElementById("sh-mode-3d");
    sideBtn?.addEventListener("click", () => this.setViewMode("side"));
    view3dBtn?.addEventListener("click", () => this.setViewMode("3d"));

    const fieldBtns = document.querySelectorAll<HTMLButtonElement>("[data-sh-field]");
    fieldBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.shField as FieldMode | undefined;
        if (!mode) return;
        this.fieldMode = mode;
        if (this.tract3dView) this.tract3dView.mode = mode;
        fieldBtns.forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    const labelsBtn = document.getElementById("sh-toggle-labels");
    labelsBtn?.addEventListener("click", () => {
      this.showLabels = !this.showLabels;
      if (this.tract3dView) this.tract3dView.showLabels = this.showLabels;
      labelsBtn.classList.toggle("active", this.showLabels);
    });

    const presetBtns = document.querySelectorAll<HTMLButtonElement>("[data-sh-preset]");
    presetBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const preset = btn.dataset.shPreset as CameraPreset3D | undefined;
        if (!preset || !this.tract3dView) return;
        this.tract3dView.setPreset(preset);
        presetBtns.forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    const cutBtn = document.getElementById("sh-toggle-cut");
    cutBtn?.addEventListener("click", () => {
      if (!this.tract3dView) return;
      this.tract3dView.cut = !this.tract3dView.cut;
      cutBtn.classList.toggle("active", this.tract3dView.cut);
    });

    const tissuesBtn = document.getElementById("sh-toggle-tissues");
    tissuesBtn?.addEventListener("click", () => {
      if (!this.tract3dView) return;
      this.tract3dView.tissues = !this.tract3dView.tissues;
      tissuesBtn.classList.toggle("active", this.tract3dView.tissues);
    });
  }

  public setViewMode(mode: SingingHeadViewMode): void {
    this.viewMode = mode;
    document.getElementById("sh-mode-side")?.classList.toggle("active", mode === "side");
    document.getElementById("sh-mode-3d")?.classList.toggle("active", mode === "3d");
    if (this.controls3dEl) {
      this.controls3dEl.style.display = mode === "3d" ? "inline-flex" : "none";
    }
    this.tract3dView?.setActive(mode === "3d", this.tractGeom);
  }

  public async loadModel(url: string = `${import.meta.env.BASE_URL}data/singing_head_model.json`): Promise<boolean> {
    if (this.modelLoadStarted) return this.inference.isTrained();
    this.modelLoadStarted = true;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: unknown = await res.json();
      if (!isValidSingingHeadModel(json)) {
        throw new Error("Invalid singing_head_model.json schema");
      }
      this.inference.setModel(json);
      const stubs = json.corpus_stats?.stubs_used ?? 0;
      const stems = json.corpus_stats?.stems_used ?? 0;
      const acc = json.corpus_stats?.val_vowel_acc;
      console.info(
        `[SingingHead] loaded ${json.format} • ${stubs} stubs + ${stems} stems` +
        `${json.weights.b3.length >= 28 ? " • 14D tract" : ""}` +
        `${typeof acc === "number" ? ` • vowel acc ${(acc * 100).toFixed(1)}%` : ""}`
      );
      return true;
    } catch (err) {
      console.warn("SingingHeadPanel: falling back to analytical model:", err);
      return false;
    }
  }

  public update(
    meta: GrooveboxMeta | null,
    audioNow: number,
    vocalSamples: Float32Array | null,
    sampleRate: number,
    vocalLevel: number,
    isMuted: boolean,
    bpm: number
  ): void {
    const dt =
      this.lastAudioSec > 0
        ? Math.min(0.08, Math.max(0.001, audioNow - this.lastAudioSec))
        : 0.016;
    this.lastAudioSec = audioNow;

    const samples = !isMuted && vocalSamples && vocalSamples.length > 0 ? vocalSamples : this.silentFrame;
    const pred: SingingHeadArticulation = this.inference.infer(samples, sampleRate > 0 ? sampleRate : 44100);

    // Track fast and slow syllable RMS envelopes so inter-syllable dips close the
    // mouth cleanly while voiced syllables track the reconstructed vocal simulator parameters.
    this.smoothedRms = smoothAsym(this.smoothedRms, pred.rms, dt, 0.020, 0.090);
    this.slowRms = smoothAsym(this.slowRms, pred.rms, dt, 0.14, 0.30);
    this.smoothedVoicing = smoothAsym(this.smoothedVoicing, pred.voicing, dt, 0.03, 0.14);

    const sylGate = clamp01((pred.rms - 0.0022) / 0.012);

    const targetAperture = clamp01(pred.aperture * sylGate);
    const targetWidth = clamp01(0.5 + (pred.width - 0.5) * sylGate);
    const targetRoundness = clamp01(pred.roundness * sylGate);
    const targetJawDrop = clamp01(pred.jawDrop * sylGate);
    const targetTongueHeight = clamp01(0.55 + (pred.tongueHeight - 0.55) * sylGate);

    // Critically-damped 2nd-order spring-damper articulators: natural syllable
    // articulation (omega_attack 28-32 rad/s) and smooth co-articulation
    // (omega_release 18-22 rad/s), rejecting frame-to-frame jitter without overshoot.
    this.aperture = stepSpringAsym(this.scalarSprings.aperture, targetAperture, dt, 32, 20);
    this.width = stepSpringAsym(this.scalarSprings.width, targetWidth, dt, 28, 18);
    this.roundness = stepSpringAsym(this.scalarSprings.roundness, targetRoundness, dt, 28, 18);
    this.jawDrop = stepSpringAsym(this.scalarSprings.jawDrop, targetJawDrop, dt, 30, 19);
    this.tongueHeight = stepSpringAsym(this.scalarSprings.tongueHeight, targetTongueHeight, dt, 28, 18);
    this.pitchNorm = stepSpringAsym(this.scalarSprings.pitchNorm, pred.pitchNorm, dt, 24, 15);
    this.vibrato = stepSpringAsym(this.scalarSprings.vibrato, pred.vibrato, dt, 18, 12);
    this.tension = stepSpringAsym(this.scalarSprings.tension, pred.tension, dt, 26, 16);
    this.breath = stepSpringAsym(this.scalarSprings.breath, pred.breath, dt, 28, 18);

    for (let k = 0; k < 5; k++) {
      this.vowelProbs[k] = stepSpringAsym(this.vowelSprings[k], pred.vowelProbs[k], dt, 28, 18);
    }
    let bestK = 0;
    for (let k = 1; k < 5; k++) {
      if (this.vowelProbs[k] > this.vowelProbs[bestK]) bestK = k;
    }
    this.topVowel = SINGING_VOWELS[bestK];

    if (pred.f0Hz > 0) {
      this.lastF0Hz = this.lastF0Hz > 0 ? smoothAsym(this.lastF0Hz, pred.f0Hz, dt, 0.04, 0.04) : pred.f0Hz;
    } else if (this.smoothedRms < 0.004) {
      this.lastF0Hz = 0;
    }

    this.smoothedArt.aperture = this.aperture;
    this.smoothedArt.width = this.width;
    this.smoothedArt.roundness = this.roundness;
    this.smoothedArt.jawDrop = this.jawDrop;
    this.smoothedArt.tongueHeight = this.tongueHeight;
    this.smoothedArt.pitchNorm = this.pitchNorm;
    this.smoothedArt.vibrato = this.vibrato;
    this.smoothedArt.tension = this.tension;
    this.smoothedArt.breath = this.breath;
    this.smoothedArt.topVowel = this.topVowel;
    this.smoothedArt.rms = this.smoothedRms;
    this.smoothedArt.f0Hz = this.lastF0Hz;
    this.smoothedArt.voicing = this.smoothedVoicing;
    this.smoothedArt.centroidHz = pred.centroidHz;
    this.smoothedArt.trained = pred.trained;
    if (pred.tract) {
      for (let i = 0; i < ARTICULATOR_KEYS.length; i++) {
        const k = ARTICULATOR_KEYS[i];
        const [lo, hi] = ARTICULATOR_RANGES[k];
        const restVal = DEFAULT_ARTICULATORS[k];
        const rawVal = pred.tract[k];
        const targetVal = clampRange(restVal + (rawVal - restVal) * sylGate, lo, hi);
        this.smoothedTract[k] = stepSpringAsym(this.tractSprings[k], targetVal, dt, 30, 19);
      }
      this.smoothedArt.tract = this.smoothedTract;
    } else {
      this.smoothedArt.tract = undefined;
    }

    const timeSec = performance.now() / 1000;
    articulationToTract(this.smoothedArt, timeSec, this.tractArt);
    this.tractGeom = computeGeometry(this.tractArt, this.anatomy, this.teeth);
    updateVocalFieldState(this.smoothedArt, this.tractArt, this.tractGeom, timeSec, this.fieldState);

    if (this.audioSourceMode === "synth" && this.audioEngine && this.audioEngine.ready) {
      const target = this.audioEngine.getVocalSynthTarget();
      this.tractSynth.attach(target.ctx, target.destination);
      // Drive HEAD SYNTH directly from low-latency inferred controls (avoid 40-140ms visual spring smear)
      this.tractSynth.update(pred, pred.tract ?? this.tractArt, this.tractGeom, isMuted);
    }

    // Header live telemetry readout
    if (this.statusBadgeEl) {
      const singing = !isMuted && (this.aperture > 0.07 || vocalLevel > 0.02);
      if (isMuted) {
        this.statusBadgeEl.textContent = "MUTED";
      } else if (singing) {
        this.statusBadgeEl.textContent = `${this.topVowel}${this.lastF0Hz > 0 ? ` • ${hzToNoteName(this.lastF0Hz)}` : ""}`;
      } else {
        this.statusBadgeEl.textContent = "STANDBY";
      }
    }

    this.render(meta, samples, vocalLevel, isMuted, bpm, dt);
  }

  private render(
    meta: GrooveboxMeta | null,
    samples: Float32Array,
    vocalLevel: number,
    isMuted: boolean,
    _bpm: number,
    dt: number
  ): void {
    if (!this.canvas) return;
    if (!this.ctx) {
      this.ctx = this.canvas.getContext("2d");
      if (!this.ctx) return;
    }
    const ctx = this.ctx;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const logicalW = 960;
    const logicalH = 380;
    const targetW = Math.round(logicalW * dpr);
    const targetH = Math.round(logicalH * dpr);
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    const t = performance.now() / 1000;
    const active = !isMuted && (this.aperture > 0.06 || this.smoothedRms > 0.004 || vocalLevel > 0.015);

    // -------------------------------------------------------------------------
    // 1. Background Chassis & Cyber Grid
    // -------------------------------------------------------------------------
    const bgGrad = ctx.createRadialGradient(logicalW * 0.5, logicalH * 0.48, 20, logicalW * 0.5, logicalH * 0.48, 520);
    bgGrad.addColorStop(0, active ? "rgba(26, 14, 42, 0.98)" : "rgba(11, 15, 24, 0.98)");
    bgGrad.addColorStop(0.55, "rgba(8, 10, 17, 0.99)");
    bgGrad.addColorStop(1, "#05070c");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, logicalW, logicalH);

    ctx.strokeStyle = "rgba(0, 229, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 40; x < logicalW; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, logicalH);
      ctx.stroke();
    }
    for (let y = 34; y < logicalH; y += 34) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(logicalW, y);
      ctx.stroke();
    }

    // -------------------------------------------------------------------------
    // 2. Left Panel: Vocal Waveform & Inverse Vowel Posterior Pentagon
    // -------------------------------------------------------------------------
    this.drawLeftTelemetry(ctx, samples, active, isMuted, t);

    // -------------------------------------------------------------------------
    // 3. Center Stage: Voice Workshop Exact Side View or 3D Vocal Tract View
    // -------------------------------------------------------------------------
    const vx = 266;
    const vy = 12;
    const vw = 428;
    const vh = 356;

    if (this.viewMode === "side") {
      this.sideView.renderRegion(
        ctx,
        dpr,
        vx,
        vy,
        vw,
        vh,
        this.tractGeom,
        this.anatomy,
        this.tractArt,
        this.teeth,
        active ? this.fieldState : null,
        this.fieldMode,
        this.showLabels,
        t,
        dt
      );
      ctx.strokeStyle = active ? "rgba(0, 229, 255, 0.36)" : "rgba(100, 116, 139, 0.28)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.roundRect(vx, vy, vw, vh, 8);
      ctx.stroke();
    } else if (this.tract3dView) {
      this.tract3dView.renderFrame(this.tractGeom, active ? this.fieldState : null);
    }

    // -------------------------------------------------------------------------
    // 4. Right Panel: Reverse-Trained Articulator Meters & Lyric Status
    // -------------------------------------------------------------------------
    this.drawRightTelemetry(ctx, meta, active, isMuted);

    ctx.restore();
  }

  private drawLeftTelemetry(
    ctx: CanvasRenderingContext2D,
    samples: Float32Array,
    active: boolean,
    isMuted: boolean,
    t: number
  ): void {
    const px = 22;
    const py = 40;
    const pw = 232;

    // Box 1: Live Vocal Stem Waveform
    ctx.save();
    ctx.fillStyle = "rgba(10, 14, 24, 0.78)";
    ctx.strokeStyle = active ? "rgba(255, 183, 0, 0.45)" : "rgba(100, 116, 139, 0.25)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(px, py, pw, 96, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = "bold 10px monospace";
    ctx.fillStyle = "#ffb700";
    ctx.fillText("WAVEFORM", px + 10, py + 16);

    const waveMidY = py + 56;
    ctx.strokeStyle = "rgba(255, 183, 0, 0.18)";
    ctx.beginPath();
    ctx.moveTo(px + 10, waveMidY);
    ctx.lineTo(px + pw - 10, waveMidY);
    ctx.stroke();

    ctx.beginPath();
    const waveW = pw - 20;
    const steps = 96;
    for (let i = 0; i <= steps; i++) {
      const u = i / steps;
      const idx = Math.min(samples.length - 1, Math.floor(u * (samples.length - 1)));
      const amp = isMuted ? 0 : samples[idx] ?? 0;
      const x = px + 10 + u * waveW;
      const y = waveMidY + Math.max(-30, Math.min(30, amp * 82));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = active ? "#ffb700" : "#475569";
    ctx.shadowBlur = active ? 8 : 0;
    ctx.shadowColor = "#ffb700";
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.restore();

    // Box 2: Vowel Formant Space Pentagon (AH, EE, OH, OO, AY)
    const rTop = py + 110;
    const rH = 190;
    ctx.save();
    ctx.fillStyle = "rgba(10, 14, 24, 0.78)";
    ctx.strokeStyle = active ? "rgba(0, 229, 255, 0.4)" : "rgba(100, 116, 139, 0.25)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(px, rTop, pw, rH, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = "bold 10px monospace";
    ctx.fillStyle = "#00e5ff";
    ctx.fillText("VOWELS", px + 10, rTop + 16);

    const rcx = px + pw * 0.5;
    const rcy = rTop + 108;
    const rad = 58;

    // Guide rings
    ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
    ctx.lineWidth = 1;
    for (const scale of [0.35, 0.7, 1.0]) {
      ctx.beginPath();
      for (let k = 0; k < 5; k++) {
        const a = (k / 5) * Math.PI * 2 - Math.PI / 2;
        const vx = rcx + Math.cos(a) * rad * scale;
        const vy = rcy + Math.sin(a) * rad * scale;
        if (k === 0) ctx.moveTo(vx, vy);
        else ctx.lineTo(vx, vy);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Axes & labels
    for (let k = 0; k < 5; k++) {
      const a = (k / 5) * Math.PI * 2 - Math.PI / 2;
      const vx = rcx + Math.cos(a) * rad;
      const vy = rcy + Math.sin(a) * rad;
      ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
      ctx.beginPath();
      ctx.moveTo(rcx, rcy);
      ctx.lineTo(vx, vy);
      ctx.stroke();

      const lx = rcx + Math.cos(a) * (rad + 16);
      const ly = rcy + Math.sin(a) * (rad + 14) + 3;
      const isTop = SINGING_VOWELS[k] === this.topVowel && active;
      ctx.font = isTop ? "bold 11px monospace" : "10px monospace";
      ctx.fillStyle = isTop ? "#ffeb3b" : "#94a3b8";
      ctx.textAlign = "center";
      ctx.fillText(SINGING_VOWELS[k], lx, ly);
    }
    ctx.textAlign = "left";

    // Probability polygon
    ctx.beginPath();
    for (let k = 0; k < 5; k++) {
      const a = (k / 5) * Math.PI * 2 - Math.PI / 2;
      const p = clamp01(this.vowelProbs[k]);
      const dist = (0.16 + 0.84 * p) * rad;
      const vx = rcx + Math.cos(a) * dist;
      const vy = rcy + Math.sin(a) * dist;
      if (k === 0) ctx.moveTo(vx, vy);
      else ctx.lineTo(vx, vy);
    }
    ctx.closePath();
    ctx.fillStyle = active ? "rgba(255, 0, 170, 0.26)" : "rgba(100, 116, 139, 0.14)";
    ctx.fill();
    ctx.strokeStyle = active ? "#ff33cc" : "#64748b";
    ctx.shadowBlur = active ? 10 : 0;
    ctx.shadowColor = "#ff00aa";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Pulsing center dot when voicing
    const pulse = 2.5 + (active ? this.smoothedVoicing * 3.5 + Math.sin(t * 10) * 0.5 : 0);
    ctx.beginPath();
    ctx.arc(rcx, rcy, pulse, 0, Math.PI * 2);
    ctx.fillStyle = active ? "#00f0ff" : "#475569";
    ctx.fill();
    ctx.restore();
  }

  private drawRightTelemetry(
    ctx: CanvasRenderingContext2D,
    meta: GrooveboxMeta | null,
    active: boolean,
    isMuted: boolean
  ): void {
    const pw = 236;
    const px = 960 - 22 - pw;
    const py = 40;
    const ph = 300;

    ctx.save();
    ctx.fillStyle = "rgba(10, 14, 24, 0.78)";
    ctx.strokeStyle = active ? "rgba(255, 0, 170, 0.42)" : "rgba(100, 116, 139, 0.25)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(px, py, pw, ph, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = "bold 10px monospace";
    ctx.fillStyle = "#ff77e9";
    ctx.fillText("ARTICULATION", px + 12, py + 18);

    const bars: Array<{ label: string; val: number; color: string }> = [
      { label: "APERTURE", val: this.aperture, color: "#ffb700" },
      { label: "WIDTH", val: this.width, color: "#00f0ff" },
      { label: "ROUND", val: this.roundness, color: "#ff33cc" },
      { label: "JAW DROP", val: this.jawDrop, color: "#ff8800" },
      { label: "TONGUE", val: this.tongueHeight, color: "#00ffa6" },
      { label: "PITCH", val: this.pitchNorm, color: "#a855f7" },
      { label: "VIBRATO", val: this.vibrato, color: "#facc15" },
      { label: "TENSION", val: this.tension, color: "#f43f5e" },
      { label: "BREATH", val: this.breath, color: "#38bdf8" },
    ];

    const startY = py + 36;
    const rowH = 22;
    const barX = px + 76;
    const barW = pw - 122;

    for (let i = 0; i < bars.length; i++) {
      const b = bars[i];
      const y = startY + i * rowH;
      const v = clamp01(b.val);

      ctx.font = "10px monospace";
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(b.label, px + 12, y + 8);

      ctx.fillStyle = "rgba(30, 41, 59, 0.85)";
      ctx.beginPath();
      ctx.roundRect(barX, y, barW, 9, 4);
      ctx.fill();

      ctx.fillStyle = b.color;
      ctx.shadowBlur = active && v > 0.25 ? 6 : 0;
      ctx.shadowColor = b.color;
      ctx.beginPath();
      ctx.roundRect(barX, y, Math.max(2, barW * v), 9, 4);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "right";
      ctx.fillText(`${Math.round(v * 100)}%`, px + pw - 10, y + 8);
      ctx.textAlign = "left";
    }

    // Bottom readout inside right box
    const footY = py + ph - 48;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
    ctx.beginPath();
    ctx.moveTo(px + 12, footY);
    ctx.lineTo(px + pw - 12, footY);
    ctx.stroke();

    const noteTxt = this.lastF0Hz > 0 ? `${hzToNoteName(this.lastF0Hz)} • ${Math.round(this.lastF0Hz)} Hz` : "-- Hz";
    const statusTxt = isMuted ? "MUTED" : active ? `SINGING [${this.topVowel}]` : "IDLE";
    ctx.font = "bold 11px monospace";
    ctx.fillStyle = active ? "#ffeb3b" : "#94a3b8";
    ctx.fillText(`${statusTxt}  ${noteTxt}`, px + 12, footY + 18);

    const trackLabel =
      (meta as { activeTrackTitle?: string } | null)?.activeTrackTitle ||
      meta?.vocalSongTitle ||
      meta?.patterns?.vocals ||
      "";
    ctx.font = "10px monospace";
    ctx.fillStyle = "#67e8f9";
    const clipped = trackLabel.length > 29 ? `${trackLabel.slice(0, 28)}…` : trackLabel;
    ctx.fillText(clipped, px + 12, footY + 35);

    ctx.restore();
  }
}
