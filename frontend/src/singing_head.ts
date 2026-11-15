// -----------------------------------------------------------------------------
// Singing Head: Stylized Neon/Cyberpunk 2D Canvas Vocal Articulator
// Placed at the bottom of the KG-505 interface and driven in real time by the
// reverse-trained acoustic-to-articulatory model on the vocal instrument stem.
// -----------------------------------------------------------------------------

import type { GrooveboxMeta } from "./main";
import {
  SINGING_FFT_SIZE,
  SINGING_VOWELS,
  SingingHeadInference,
  isValidSingingHeadModel,
  type SingingHeadArticulation,
  type SingingVowel,
} from "./singing_head_model";

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
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

interface FloatingSpark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  color: string;
  size: number;
}

export class SingingHeadPanel {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private statusBadgeEl: HTMLElement | null = null;
  private modelBadgeEl: HTMLElement | null = null;

  private readonly inference = new SingingHeadInference();
  private readonly silentFrame = new Float32Array(SINGING_FFT_SIZE);

  // Smoothed articulatory state
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
  private smoothedVoicing = 0;
  private lastAudioSec = 0;

  // Animation state
  private blinkTimer = 0;
  private nextBlinkAt = 3.6;
  private blinkAmount = 0; // 0 = open, 1 = closed
  private sparks: FloatingSpark[] = [];
  private sparkCooldown = 0;
  private modelLoadStarted = false;

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

    void this.loadModel();
    this.render(null, this.silentFrame, 0, false, 126, 0.016);
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
      if (this.modelBadgeEl) {
        const accStr = typeof acc === "number" ? ` • ${(acc * 100).toFixed(0)}% VOWEL ACC` : "";
        this.modelBadgeEl.textContent = `REVERSE-TRAINED • ${stubs} STUBS + ${stems} STEMS${accStr}`;
      }
      return true;
    } catch (err) {
      console.warn("SingingHeadPanel: falling back to analytical model:", err);
      if (this.modelBadgeEl) {
        this.modelBadgeEl.textContent = "ANALYTICAL FALLBACK";
      }
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

    // Smooth articulators: ultra-fast attack (16-24 ms) for consonant/syllable snap,
    // musical release (90-140 ms) so held notes & vibrato glide cleanly.
    this.aperture = smoothAsym(this.aperture, pred.aperture, dt, 0.016, 0.095);
    this.width = smoothAsym(this.width, pred.width, dt, 0.025, 0.11);
    this.roundness = smoothAsym(this.roundness, pred.roundness, dt, 0.025, 0.11);
    this.jawDrop = smoothAsym(this.jawDrop, pred.jawDrop, dt, 0.018, 0.10);
    this.tongueHeight = smoothAsym(this.tongueHeight, pred.tongueHeight, dt, 0.028, 0.12);
    this.pitchNorm = smoothAsym(this.pitchNorm, pred.pitchNorm, dt, 0.032, 0.16);
    this.vibrato = smoothAsym(this.vibrato, pred.vibrato, dt, 0.045, 0.18);
    this.tension = smoothAsym(this.tension, pred.tension, dt, 0.025, 0.14);
    this.breath = smoothAsym(this.breath, pred.breath, dt, 0.020, 0.09);

    for (let k = 0; k < 5; k++) {
      this.vowelProbs[k] = smoothAsym(this.vowelProbs[k], pred.vowelProbs[k], dt, 0.024, 0.11);
    }
    let bestK = 0;
    for (let k = 1; k < 5; k++) {
      if (this.vowelProbs[k] > this.vowelProbs[bestK]) bestK = k;
    }
    this.topVowel = SINGING_VOWELS[bestK];

    this.smoothedRms = smoothAsym(this.smoothedRms, pred.rms, dt, 0.02, 0.12);
    this.smoothedVoicing = smoothAsym(this.smoothedVoicing, pred.voicing, dt, 0.03, 0.14);
    if (pred.f0Hz > 0) {
      this.lastF0Hz = this.lastF0Hz > 0 ? this.lastF0Hz * 0.6 + pred.f0Hz * 0.4 : pred.f0Hz;
    } else if (this.smoothedRms < 0.004) {
      this.lastF0Hz = 0;
    }

    // Natural eye blink timer
    this.blinkTimer += dt;
    if (this.blinkTimer >= this.nextBlinkAt) {
      const blinkProgress = (this.blinkTimer - this.nextBlinkAt) / 0.16;
      if (blinkProgress >= 1) {
        this.blinkAmount = 0;
        this.blinkTimer = 0;
        this.nextBlinkAt = 2.8 + Math.random() * 2.6;
      } else {
        this.blinkAmount = Math.sin(blinkProgress * Math.PI);
      }
    } else {
      this.blinkAmount = 0;
    }

    // Floating vocal sparks on energetic voiced singing
    this.sparkCooldown = Math.max(0, this.sparkCooldown - dt);
    if (!isMuted && this.aperture > 0.28 && this.smoothedVoicing > 0.42 && this.sparkCooldown <= 0) {
      this.sparkCooldown = 0.09;
      const side = Math.random() < 0.5 ? -1 : 1;
      this.sparks.push({
        x: side * (26 + Math.random() * 44),
        y: 28 + (Math.random() - 0.5) * 26,
        vx: side * (12 + Math.random() * 20),
        vy: -(24 + Math.random() * 34),
        life: 1.0,
        color: this.tension > 0.55 ? "#ffb700" : "#00f0ff",
        size: 1.5 + Math.random() * 2.2,
      });
      if (this.sparks.length > 28) this.sparks.shift();
    }
    for (let i = this.sparks.length - 1; i >= 0; i--) {
      const s = this.sparks[i];
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.life -= dt * 1.15;
      if (s.life <= 0) this.sparks.splice(i, 1);
    }

    // Header live telemetry readout
    if (this.statusBadgeEl) {
      const singing = !isMuted && (this.aperture > 0.07 || vocalLevel > 0.02);
      if (isMuted) {
        this.statusBadgeEl.textContent = "VOCALS MUTED • STANDBY";
      } else if (singing) {
        const note = this.lastF0Hz > 0 ? `${hzToNoteName(this.lastF0Hz)} (${Math.round(this.lastF0Hz)}Hz)` : "CHOP/BREATH";
        this.statusBadgeEl.textContent = `[${this.topVowel}] • ${note} • APERTURE ${Math.round(this.aperture * 100)}%`;
      } else {
        this.statusBadgeEl.textContent = "LISTENING TO VOCAL INSTRUMENT…";
      }
    }

    this.render(meta, samples, vocalLevel, isMuted, bpm, dt);
  }

  private render(
    meta: GrooveboxMeta | null,
    samples: Float32Array,
    vocalLevel: number,
    isMuted: boolean,
    bpm: number,
    _dt: number
  ): void {
    if (!this.canvas) return;
    if (!this.ctx) {
      this.ctx = this.canvas.getContext("2d");
      if (!this.ctx) return;
    }
    const ctx = this.ctx;

    // Keep backing resolution crisp on HiDPI screens while preserving 960x340 logical space
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const logicalW = 960;
    const logicalH = 340;
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
    const beatPhase = (t * (bpm / 60)) % 1;
    const beatBounce = active ? Math.sin(beatPhase * Math.PI * 2) * 1.8 : Math.sin(t * 1.4) * 1.2;

    // -------------------------------------------------------------------------
    // 1. Background Chassis & Cyber Grid
    // -------------------------------------------------------------------------
    const bgGrad = ctx.createRadialGradient(logicalW * 0.5, logicalH * 0.48, 20, logicalW * 0.5, logicalH * 0.48, 520);
    bgGrad.addColorStop(0, active ? "rgba(26, 14, 42, 0.98)" : "rgba(11, 15, 24, 0.98)");
    bgGrad.addColorStop(0.55, "rgba(8, 10, 17, 0.99)");
    bgGrad.addColorStop(1, "#05070c");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, logicalW, logicalH);

    // Subtle perspective scanline grid
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
    // 2. Left Panel: Vocal Waveform & Formant Vowel Pentagon Radar
    // -------------------------------------------------------------------------
    this.drawLeftTelemetry(ctx, samples, active, isMuted, t);

    // -------------------------------------------------------------------------
    // 3. Center Portrait: Cyberpunk Neon Singing Head
    // -------------------------------------------------------------------------
    const headX = logicalW * 0.5;
    const pitchLift = (this.pitchNorm - 0.42) * -8;
    const vibX = active ? Math.sin(t * 32) * this.vibrato * 2.2 : 0;
    const headY = logicalH * 0.47 + beatBounce + pitchLift;
    const tilt = active ? (this.pitchNorm - 0.45) * 0.05 + Math.sin(t * 1.7) * 0.015 : Math.sin(t * 0.9) * 0.01;

    this.drawSingingHead(ctx, headX + vibX, headY, tilt, samples, active, isMuted, meta, t);

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
    const py = 20;
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
    ctx.fillText("VOCAL INSTRUMENT WAVEFORM", px + 10, py + 16);

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
    ctx.fillText("INVERSE VOWEL POSTERIOR", px + 10, rTop + 16);

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
    const py = 20;
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
    ctx.fillText("ARTICULATOR TELEMETRY", px + 12, py + 18);

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
      "VOCAL INSTRUMENT";
    ctx.font = "10px monospace";
    ctx.fillStyle = "#67e8f9";
    const clipped = trackLabel.length > 29 ? `${trackLabel.slice(0, 28)}…` : trackLabel;
    ctx.fillText(clipped, px + 12, footY + 35);

    ctx.restore();
  }

  private drawSingingHead(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    tilt: number,
    samples: Float32Array,
    active: boolean,
    isMuted: boolean,
    meta: GrooveboxMeta | null,
    t: number
  ): void {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(tilt);

    const isDrop = meta?.section === "DROP";
    const primaryCyan = active ? "#00f0ff" : "#38bdf8";
    const hotMagenta = active ? (this.tension > 0.55 || isDrop ? "#ff2e93" : "#ff4de3") : "#7e22ce";
    const amberGlow = "#ffb700";

    // -------------------------------------------------------------------------
    // A. Concentric Vocal Aura Halo
    // -------------------------------------------------------------------------
    const auraR = 128 + this.aperture * 26 + this.tension * 18;
    const halo = ctx.createRadialGradient(0, 0, 24, 0, 0, auraR);
    halo.addColorStop(0, active ? "rgba(255, 0, 170, 0.26)" : "rgba(0, 229, 255, 0.08)");
    halo.addColorStop(0.55, active ? "rgba(0, 229, 255, 0.14)" : "rgba(15, 23, 42, 0.05)");
    halo.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(0, 0, auraR, 0, Math.PI * 2);
    ctx.fill();

    // Outer resonance ring reacting to vocal waveform
    ctx.save();
    ctx.beginPath();
    const ringSegs = 72;
    for (let i = 0; i <= ringSegs; i++) {
      const a = (i / ringSegs) * Math.PI * 2;
      const idx = Math.floor((i / ringSegs) * (samples.length - 1));
      const s = isMuted ? 0 : samples[idx] ?? 0;
      const r = 124 + this.aperture * 10 + s * (active ? 22 : 3);
      const rx = Math.cos(a) * r;
      const ry = Math.sin(a) * r * 0.96;
      if (i === 0) ctx.moveTo(rx, ry);
      else ctx.lineTo(rx, ry);
    }
    ctx.closePath();
    ctx.strokeStyle = active ? "rgba(0, 240, 255, 0.42)" : "rgba(56, 189, 248, 0.16)";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.restore();

    // -------------------------------------------------------------------------
    // B. Cyber Neck & Pulsing Vocal Cords
    // -------------------------------------------------------------------------
    const jawOffset = this.jawDrop * 20 + this.aperture * 7;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(-34, 72 + jawOffset * 0.35);
    ctx.lineTo(-42, 146);
    ctx.lineTo(42, 146);
    ctx.lineTo(34, 72 + jawOffset * 0.35);
    ctx.closePath();
    const neckGrad = ctx.createLinearGradient(0, 70, 0, 146);
    neckGrad.addColorStop(0, "rgba(14, 21, 38, 0.96)");
    neckGrad.addColorStop(1, "rgba(6, 10, 20, 0.98)");
    ctx.fillStyle = neckGrad;
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 229, 255, 0.38)";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // Vertical vocal cord waveform ribbons inside the throat
    for (const side of [-1, 1]) {
      ctx.beginPath();
      for (let k = 0; k <= 24; k++) {
        const u = k / 24;
        const sIdx = Math.min(samples.length - 1, Math.floor(u * 256));
        const amp = isMuted ? 0 : samples[sIdx] ?? 0;
        const x = side * (9 + u * 6) + amp * (active ? 14 : 1) * side;
        const y = 78 + jawOffset * 0.35 + u * 64;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = active ? amberGlow : "rgba(100, 116, 139, 0.35)";
      ctx.shadowBlur = active ? 8 : 0;
      ctx.shadowColor = amberGlow;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // C. Cyber Headphone / Ear Nodes
    // -------------------------------------------------------------------------
    for (const side of [-1, 1]) {
      ctx.save();
      const ex = side * 80;
      const ey = -4;
      ctx.fillStyle = "#0e1526";
      ctx.strokeStyle = active ? hotMagenta : "#475569";
      ctx.shadowBlur = active ? 12 : 4;
      ctx.shadowColor = hotMagenta;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.roundRect(ex - 11, ey - 26, 22, 52, 9);
      ctx.fill();
      ctx.stroke();

      // Inner LED bar on ear cup
      ctx.fillStyle = active ? primaryCyan : "#334155";
      ctx.fillRect(ex - 3, ey - 14 * (0.35 + 0.65 * this.aperture), 6, 28 * (0.35 + 0.65 * this.aperture));
      ctx.restore();
    }

    // -------------------------------------------------------------------------
    // D. Sculpted Head & Hinged Jaw Silhouette
    // -------------------------------------------------------------------------
    const cheekRound = this.roundness * 4;
    const chinY = 92 + jawOffset;
    const jawCornerY = 44 + jawOffset * 0.55;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, -104);
    // Right cranium -> temple -> cheekbone -> hinged jaw -> chin
    ctx.bezierCurveTo(54, -104, 78, -68, 76, -18);
    ctx.bezierCurveTo(75, 16, 66 - cheekRound, 32, 54 - cheekRound, jawCornerY);
    ctx.quadraticCurveTo(32, chinY - 4, 15, chinY);
    ctx.lineTo(-15, chinY);
    ctx.quadraticCurveTo(-32, chinY - 4, -54 + cheekRound, jawCornerY);
    ctx.bezierCurveTo(-66 + cheekRound, 32, -75, 16, -76, -18);
    ctx.bezierCurveTo(-78, -68, -54, -104, 0, -104);
    ctx.closePath();

    const faceGrad = ctx.createRadialGradient(0, -24, 14, 0, 10, 120);
    faceGrad.addColorStop(0, active ? "rgba(32, 24, 62, 0.96)" : "rgba(18, 24, 38, 0.96)");
    faceGrad.addColorStop(0.65, "rgba(13, 18, 32, 0.98)");
    faceGrad.addColorStop(1, "#070b14");
    ctx.fillStyle = faceGrad;
    ctx.fill();

    // Neon rim outline
    ctx.strokeStyle = primaryCyan;
    ctx.shadowBlur = active ? 18 + this.tension * 14 : 6;
    ctx.shadowColor = primaryCyan;
    ctx.lineWidth = 2.6;
    ctx.stroke();

    // Inner cyber contour lines (cheekbones & forehead crown)
    ctx.shadowBlur = 0;
    ctx.strokeStyle = active ? "rgba(255, 77, 227, 0.45)" : "rgba(100, 116, 139, 0.24)";
    ctx.lineWidth = 1.2;
    // Crown arc
    ctx.beginPath();
    ctx.arc(0, -34, 58, 1.18 * Math.PI, 1.82 * Math.PI);
    ctx.stroke();
    // Cheek cyber panels
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(side * 66, -2);
      ctx.lineTo(side * 44, 18);
      ctx.lineTo(side * 36, 42 + jawOffset * 0.35);
      ctx.stroke();
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // E. Expressive Eyebrows & Cyber Eyes
    // -------------------------------------------------------------------------
    const eyeY = -24;
    const eyeXOff = 29;
    const browLift = (this.pitchNorm - 0.4) * 9 + this.aperture * 3;
    const browFurrow = this.tension * 5;

    for (const side of [-1, 1]) {
      const ex = side * eyeXOff;
      // Eyebrow
      ctx.save();
      ctx.strokeStyle = active ? primaryCyan : "#64748b";
      ctx.shadowBlur = active ? 8 : 0;
      ctx.shadowColor = primaryCyan;
      ctx.lineWidth = 2.8;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(ex - side * 14, eyeY - 19 - browLift + browFurrow);
      ctx.quadraticCurveTo(
        ex,
        eyeY - 24 - browLift,
        ex + side * 15,
        eyeY - 17 - browLift - browFurrow * 0.4
      );
      ctx.stroke();
      ctx.restore();

      // Eye socket & iris
      ctx.save();
      const openH = Math.max(1.2, 10.5 * (1 - this.blinkAmount * 0.92));
      ctx.beginPath();
      ctx.ellipse(ex, eyeY, 15, openH, 0, 0, Math.PI * 2);
      ctx.fillStyle = "#04070e";
      ctx.fill();
      ctx.strokeStyle = active ? primaryCyan : "#475569";
      ctx.lineWidth = 1.8;
      ctx.stroke();

      if (this.blinkAmount < 0.78) {
        ctx.save();
        ctx.clip();
        // Gaze rises subtly with vocal pitch
        const gazeY = eyeY - (this.pitchNorm - 0.45) * 4;
        const irisR = 6.8 + this.tension * 1.4;
        const irisColor = this.tension > 0.58 ? amberGlow : active ? primaryCyan : "#38bdf8";

        ctx.beginPath();
        ctx.arc(ex, gazeY, irisR, 0, Math.PI * 2);
        ctx.fillStyle = irisColor;
        ctx.shadowBlur = active ? 12 : 3;
        ctx.shadowColor = irisColor;
        ctx.fill();

        // Dark cyber pupil
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(ex, gazeY, 3.0 - this.tension * 0.7, 0, Math.PI * 2);
        ctx.fillStyle = "#05070e";
        ctx.fill();

        // Specular catchlight
        ctx.beginPath();
        ctx.arc(ex - 2.4, gazeY - 2.2, 1.6, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.restore();
      }
      ctx.restore();
    }

    // Minimalist cyber nose bridge
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.55)";
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(0, -16);
    ctx.lineTo(-3.5, 11);
    ctx.lineTo(3.5, 13);
    ctx.stroke();
    ctx.restore();

    // -------------------------------------------------------------------------
    // F. Reverse-Trained Articulated Singing Mouth (Lips, Cavity, Teeth, Tongue)
    // -------------------------------------------------------------------------
    const mouthY = 44 + jawOffset * 0.48;
    const halfW = 14 + this.width * 24 - this.roundness * 12;
    const halfH = 1.8 + this.aperture * 23 + this.jawDrop * 5;
    const cornerLift = (this.width - 0.5) * -6 + this.roundness * 3.5;
    const bowDepth = 2.6 + (1 - this.roundness) * 2.2;
    const vibRip = active ? Math.sin(t * 36) * this.vibrato * 1.6 : 0;

    const leftX = -halfW;
    const rightX = halfW;
    const cornerY = mouthY + cornerLift;
    const topY = mouthY - halfH * 0.72 + vibRip;
    const botY = mouthY + halfH * 1.08 + vibRip;

    // Helper path for inner mouth cavity
    const traceMouthCavity = (): void => {
      ctx.beginPath();
      ctx.moveTo(leftX, cornerY);
      ctx.bezierCurveTo(
        leftX * 0.48,
        topY - bowDepth * 0.45,
        -halfW * 0.18,
        topY,
        0,
        topY + bowDepth * 0.35
      );
      ctx.bezierCurveTo(
        halfW * 0.18,
        topY,
        rightX * 0.48,
        topY - bowDepth * 0.45,
        rightX,
        cornerY
      );
      ctx.bezierCurveTo(
        rightX * 0.62,
        botY,
        -rightX * 0.62,
        botY,
        leftX,
        cornerY
      );
      ctx.closePath();
    };

    if (this.aperture > 0.045) {
      // 1) Oral cavity interior + vocal resonance throat glow
      ctx.save();
      traceMouthCavity();
      ctx.fillStyle = "#080312";
      ctx.fill();
      ctx.clip();

      // Throat resonance glow when voicing/belting
      const throatGlow = ctx.createRadialGradient(0, mouthY + halfH * 0.1, 2, 0, mouthY + halfH * 0.1, Math.max(12, halfW));
      throatGlow.addColorStop(0, `rgba(255, 60, 150, ${(0.18 + 0.55 * this.tension) * this.smoothedVoicing})`);
      throatGlow.addColorStop(1, "rgba(8, 3, 18, 0)");
      ctx.fillStyle = throatGlow;
      ctx.fillRect(-halfW - 4, topY - 8, halfW * 2 + 8, halfH * 2 + 16);

      // 2) Articulated Tongue (rises with tongueHeight, fronts with width)
      const tongueTop = botY - (0.28 + 0.46 * this.tongueHeight) * halfH * 1.15;
      const tongueW = halfW * (0.76 - 0.18 * this.roundness);
      ctx.beginPath();
      ctx.ellipse(0, tongueTop + halfH * 0.55, tongueW, Math.max(5, halfH * 0.68), 0, Math.PI, Math.PI * 2);
      const tongueGrad = ctx.createLinearGradient(0, tongueTop, 0, botY + 6);
      tongueGrad.addColorStop(0, "#ff5c9e");
      tongueGrad.addColorStop(1, "#9d174d");
      ctx.fillStyle = tongueGrad;
      ctx.fill();

      // 3) Upper Teeth Row (revealed on open vowels, wide EE/AY, and fricative breath)
      const teethExpose = clamp01((this.aperture * 1.2 + this.width * 0.45 + this.breath * 0.6 - this.roundness * 0.75) * 0.9);
      if (teethExpose > 0.18) {
        const teethH = Math.min(halfH * 0.46, 3.5 + teethExpose * 5.2);
        const teethW = halfW * 0.78;
        ctx.fillStyle = "rgba(235, 250, 255, 0.94)";
        ctx.beginPath();
        ctx.roundRect(-teethW, topY - 1, teethW * 2, teethH, 3);
        ctx.fill();

        // Subtle tooth dividers
        ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
        ctx.lineWidth = 0.8;
        for (let tx = -2; tx <= 2; tx++) {
          const xPos = (tx / 3) * teethW;
          ctx.beginPath();
          ctx.moveTo(xPos, topY);
          ctx.lineTo(xPos, topY + teethH - 1);
          ctx.stroke();
        }
      }

      // Lower teeth hint on wide open belts
      if (this.aperture > 0.48 && this.jawDrop > 0.42 && this.roundness < 0.55) {
        const ltW = halfW * 0.65;
        ctx.fillStyle = "rgba(220, 240, 255, 0.78)";
        ctx.beginPath();
        ctx.roundRect(-ltW, botY - 4.2, ltW * 2, 5, 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // 4) Outer Neon Lips (Upper & Lower)
    ctx.save();
    const lipColor = active ? hotMagenta : "#db2777";
    ctx.strokeStyle = lipColor;
    ctx.shadowBlur = active ? 14 + this.aperture * 12 : 5;
    ctx.shadowColor = lipColor;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Upper lip line (with subtle waveform shimmer when singing)
    ctx.lineWidth = 2.6 + this.roundness * 1.1;
    ctx.beginPath();
    ctx.moveTo(leftX - 2, cornerY);
    ctx.bezierCurveTo(
      leftX * 0.5,
      topY - bowDepth * 0.85,
      -halfW * 0.16,
      topY - bowDepth * 0.35,
      0,
      topY + bowDepth * 0.25
    );
    ctx.bezierCurveTo(
      halfW * 0.16,
      topY - bowDepth * 0.35,
      rightX * 0.5,
      topY - bowDepth * 0.85,
      rightX + 2,
      cornerY
    );
    ctx.stroke();

    // Lower lip line
    ctx.lineWidth = 3.0 + this.roundness * 1.2;
    ctx.beginPath();
    ctx.moveTo(leftX - 1.5, cornerY);
    ctx.bezierCurveTo(
      leftX * 0.62,
      botY + 2.5,
      rightX * 0.62,
      botY + 2.5,
      rightX + 1.5,
      cornerY
    );
    ctx.stroke();

    // Breath / sibilance mist wisps when breath > 0.30
    if (active && this.breath > 0.30) {
      ctx.shadowBlur = 6;
      ctx.shadowColor = "#38bdf8";
      ctx.strokeStyle = `rgba(125, 211, 252, ${Math.min(0.65, this.breath * 0.7)})`;
      ctx.lineWidth = 1.2;
      for (let w = -1; w <= 1; w++) {
        const phase = (t * 6 + w * 0.7) % 1;
        const wx = w * 10;
        const wy = mouthY + 4 + phase * 18;
        ctx.beginPath();
        ctx.arc(wx, wy, 2 + phase * 5, 0.2 * Math.PI, 0.8 * Math.PI);
        ctx.stroke();
      }
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // G. Floating Vocal Sparks
    // -------------------------------------------------------------------------
    ctx.save();
    for (const sp of this.sparks) {
      ctx.globalAlpha = clamp01(sp.life);
      ctx.fillStyle = sp.color;
      ctx.shadowBlur = 8;
      ctx.shadowColor = sp.color;
      ctx.beginPath();
      ctx.arc(sp.x, sp.y, sp.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    ctx.restore();
  }
}
