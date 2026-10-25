// Phase 3: Canvas 2D visualizers — concentric stem rings + UMAP nebula.
// Both renderers are devicePixelRatio-aware and dependency-free.
import type { StemName } from "./types";

export const STEM_COLORS: Record<StemName, string> = {
  drums: "#ff5470",
  bass: "#ffb020",
  other: "#38bdf8",
  vocals: "#c084fc",
};

/** Inner -> outer ring order. */
export const RING_ORDER: readonly StemName[] = ["drums", "bass", "other", "vocals"];

export interface WaveformStemData {
  timeDomain?: Uint8Array<ArrayBuffer>;
  frequency?: Uint8Array<ArrayBuffer>;
  level: number;
  activeTrackId?: string;
  activeSegmentId?: string;
  isMuted?: boolean;
  isSolo?: boolean;
}

export interface WaveformAnalysisState {
  stems: Record<StemName, WaveformStemData>;
  bpm: number;
  /** Visual-clock seconds (performance.now()/1000). */
  timeSec: number;
  /** Normalized position of active sequencer playhead across the pattern (0..1). */
  playheadProgress?: number;
}

/**
 * 4-Stem Real-Time Waveform Analysis Station.
 * Renders dedicated oscilloscope waveforms, FFT spectral density,
 * beat grid lines, live playhead laser, and peak/RMS meters per stem.
 */
export function drawWaveformAnalysis(
  ctx2d: CanvasRenderingContext2D,
  state: WaveformAnalysisState,
): void {
  const canvas = ctx2d.canvas;
  const dpr = typeof window !== "undefined" && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
  const cssW = canvas.clientWidth > 0 ? canvas.clientWidth : canvas.width;
  const cssH = canvas.clientHeight > 0 ? canvas.clientHeight : canvas.height;
  const wantW = Math.max(1, Math.round(cssW * dpr));
  const wantH = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== wantW || canvas.height !== wantH) {
    canvas.width = wantW;
    canvas.height = wantH;
  }
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssW;
  const h = cssH;
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.fillStyle = "#090910";
  ctx2d.fillRect(0, 0, w, h);

  const stems: StemName[] = ["drums", "bass", "other", "vocals"];
  const laneH = h / stems.length;
  const bpm = Number.isFinite(state.bpm) && state.bpm > 0 ? state.bpm : 126;
  const beatPhase = (state.timeSec * bpm) / 60.0;
  const playheadX = typeof state.playheadProgress === "number"
    ? Math.max(0, Math.min(1, state.playheadProgress)) * w
    : ((beatPhase % 4) / 4) * w;

  for (let i = 0; i < stems.length; i++) {
    const stem = stems[i];
    const laneY = i * laneH;
    const midY = laneY + laneH / 2;
    const color = STEM_COLORS[stem];
    const data = state.stems[stem];
    const level = data?.level ?? 0;

    // Lane background with subtle gradient
    const bgGrad = ctx2d.createLinearGradient(0, laneY, 0, laneY + laneH);
    bgGrad.addColorStop(0, "rgba(20, 20, 32, 0.45)");
    bgGrad.addColorStop(0.5, "rgba(10, 10, 18, 0.2)");
    bgGrad.addColorStop(1, "rgba(20, 20, 32, 0.45)");
    ctx2d.fillStyle = bgGrad;
    ctx2d.fillRect(0, laneY, w, laneH);

    // Lane top divider line
    if (i > 0) {
      ctx2d.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      ctx2d.moveTo(0, laneY);
      ctx2d.lineTo(w, laneY);
      ctx2d.stroke();
    }

    // Horizontal center zero-axis
    ctx2d.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx2d.lineWidth = 1;
    ctx2d.beginPath();
    ctx2d.moveTo(0, midY);
    ctx2d.lineTo(w, midY);
    ctx2d.stroke();

    // Vertical musical beat grid lines (4 beats in view)
    const beatsInView = 4;
    for (let b = 1; b < beatsInView; b++) {
      const bx = (b / beatsInView) * w;
      ctx2d.strokeStyle = b === 2 ? "rgba(255, 255, 255, 0.08)" : "rgba(255, 255, 255, 0.04)";
      ctx2d.lineWidth = 1;
      ctx2d.beginPath();
      ctx2d.moveTo(bx, laneY);
      ctx2d.lineTo(bx, laneY + laneH);
      ctx2d.stroke();
    }

    // FFT Background Spectral Density
    const freq = data?.frequency;
    if (freq && freq.length > 0 && !data.isMuted) {
      const numBars = 48;
      const barW = (w - 180) / numBars;
      const startX = 110;
      ctx2d.fillStyle = color;
      ctx2d.globalAlpha = 0.12;
      for (let b = 0; b < numBars; b++) {
        const bin = Math.floor((b / numBars) * (freq.length * 0.6));
        const val = freq[bin] / 255.0;
        const barH = val * (laneH * 0.72);
        ctx2d.fillRect(startX + b * barW, laneY + laneH - barH, barW - 1, barH);
      }
      ctx2d.globalAlpha = 1;
    }

    // Real-Time Oscilloscope Waveform (Mirrored Glowing Envelope)
    const time = data?.timeDomain;
    if (time && time.length > 0 && !data.isMuted) {
      const step = Math.max(1, Math.floor(time.length / 200));
      const points: { x: number; topY: number; botY: number }[] = [];
      const waveStartX = 110;
      const waveEndX = w - 70;
      const waveWidth = waveEndX - waveStartX;
      const maxAmp = laneH * 0.42;

      let pIdx = 0;
      for (let tIdx = 0; tIdx < time.length; tIdx += step) {
        const norm = (time[tIdx] - 128) / 128.0;
        const amp = Math.abs(norm);
        const x = waveStartX + (pIdx / (time.length / step)) * waveWidth;
        const yOff = Math.max(1, amp * maxAmp);
        points.push({ x, topY: midY - yOff, botY: midY + yOff });
        pIdx++;
      }

      if (points.length > 1) {
        // Mirrored gradient core
        const grad = ctx2d.createLinearGradient(0, midY - maxAmp, 0, midY + maxAmp);
        grad.addColorStop(0, color + "66");
        grad.addColorStop(0.5, color + "1a");
        grad.addColorStop(1, color + "66");

        ctx2d.fillStyle = grad;
        ctx2d.beginPath();
        ctx2d.moveTo(points[0].x, points[0].topY);
        for (let p = 1; p < points.length; p++) {
          ctx2d.lineTo(points[p].x, points[p].topY);
        }
        for (let p = points.length - 1; p >= 0; p--) {
          ctx2d.lineTo(points[p].x, points[p].botY);
        }
        ctx2d.closePath();
        ctx2d.fill();

        // Glowing neon top/bottom traces
        ctx2d.strokeStyle = color;
        ctx2d.lineWidth = 1.6;
        ctx2d.shadowColor = color;
        ctx2d.shadowBlur = 6;
        ctx2d.beginPath();
        ctx2d.moveTo(points[0].x, points[0].topY);
        for (let p = 1; p < points.length; p++) {
          ctx2d.lineTo(points[p].x, points[p].topY);
        }
        ctx2d.stroke();

        ctx2d.beginPath();
        ctx2d.moveTo(points[0].x, points[0].botY);
        for (let p = 1; p < points.length; p++) {
          ctx2d.lineTo(points[p].x, points[p].botY);
        }
        ctx2d.stroke();
        ctx2d.shadowBlur = 0;
      }
    } else {
      // Idle baseline
      ctx2d.strokeStyle = data?.isMuted ? "rgba(239, 68, 68, 0.3)" : color + "33";
      ctx2d.lineWidth = 1.2;
      ctx2d.beginPath();
      ctx2d.moveTo(110, midY);
      ctx2d.lineTo(w - 70, midY);
      ctx2d.stroke();
    }

    // Left HUD: Stem Badge
    const badgeW = 92;
    const badgeH = 22;
    const badgeX = 8;
    const badgeY = laneY + (laneH - badgeH) / 2;

    ctx2d.fillStyle = "rgba(16, 16, 26, 0.85)";
    ctx2d.strokeStyle = color + "66";
    ctx2d.lineWidth = 1.5;
    ctx2d.beginPath();
    ctx2d.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
    ctx2d.fill();
    ctx2d.stroke();

    // Stem dot
    ctx2d.fillStyle = color;
    ctx2d.beginPath();
    ctx2d.arc(badgeX + 11, badgeY + badgeH / 2, 4, 0, 2 * Math.PI);
    ctx2d.fill();

    // Stem text
    ctx2d.fillStyle = "#ffffff";
    ctx2d.font = "bold 11px system-ui, sans-serif";
    ctx2d.textAlign = "left";
    ctx2d.fillText(stem.toUpperCase(), badgeX + 22, badgeY + 15);

    // Mute/Solo flag or Slice ID
    if (data?.isMuted) {
      ctx2d.fillStyle = "#ef4444";
      ctx2d.font = "bold 9px system-ui, sans-serif";
      ctx2d.fillText("MUTED", badgeX + 2, laneY + 12);
    } else if (data?.isSolo) {
      ctx2d.fillStyle = "#a855f7";
      ctx2d.font = "bold 9px system-ui, sans-serif";
      ctx2d.fillText("SOLO", badgeX + 2, laneY + 12);
    } else if (data?.activeSegmentId) {
      ctx2d.fillStyle = "rgba(255, 255, 255, 0.45)";
      ctx2d.font = "9px monospace";
      const shortId = data.activeSegmentId.replace("seg_", "");
      ctx2d.fillText(shortId, badgeX + 2, laneY + 12);
    }

    // Right HUD: VU Peak/RMS Meter
    const meterX = w - 55;
    const meterW = 42;
    const meterH = 14;
    const meterY = laneY + (laneH - meterH) / 2;

    ctx2d.fillStyle = "rgba(16, 16, 26, 0.7)";
    ctx2d.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx2d.lineWidth = 1;
    ctx2d.fillRect(meterX, meterY, meterW, meterH);
    ctx2d.strokeRect(meterX, meterY, meterW, meterH);

    const filledW = Math.max(0, Math.min(meterW, level * meterW));
    if (filledW > 0 && !data?.isMuted) {
      const meterGrad = ctx2d.createLinearGradient(meterX, 0, meterX + meterW, 0);
      meterGrad.addColorStop(0, "#22c55e");
      meterGrad.addColorStop(0.7, "#eab308");
      meterGrad.addColorStop(1, "#ef4444");
      ctx2d.fillStyle = meterGrad;
      ctx2d.fillRect(meterX + 1, meterY + 1, filledW - 2, meterH - 2);
    }

    // dB numerical readout
    const db = level > 0.001 ? Math.round(20 * Math.log10(level)) : -60;
    ctx2d.fillStyle = "rgba(255, 255, 255, 0.6)";
    ctx2d.font = "9px monospace";
    ctx2d.textAlign = "right";
    ctx2d.fillText(db > -60 ? `${db}dB` : "-inf", w - 8, laneY + laneH - 6);
  }

  // Master Global Playhead Laser Line
  ctx2d.save();
  ctx2d.strokeStyle = "#ffffff";
  ctx2d.lineWidth = 1.8;
  ctx2d.shadowColor = "#38bdf8";
  ctx2d.shadowBlur = 8;
  ctx2d.globalAlpha = 0.85;
  ctx2d.beginPath();
  ctx2d.moveTo(playheadX, 0);
  ctx2d.lineTo(playheadX, h);
  ctx2d.stroke();
  ctx2d.restore();
}

export interface MasterAnalysisState {
  timeDomain?: Uint8Array<ArrayBuffer>;
  frequency?: Uint8Array<ArrayBuffer>;
  level: number;
  bpm: number;
  /** Visual-clock seconds (performance.now()/1000). */
  timeSec: number;
  /** Normalized position of active sequencer playhead across the pattern (0..1). */
  playheadProgress?: number;
}

/**
 * Combined Master Output Audio Analysis.
 * Shows the full-mix stereo oscilloscope waveform, 64-band FFT spectrum,
 * vertical beat grid lines, and master peak/RMS meter.
 */
export function drawMasterAnalysis(
  ctx2d: CanvasRenderingContext2D,
  state: MasterAnalysisState,
): void {
  const canvas = ctx2d.canvas;
  const dpr = typeof window !== "undefined" && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
  const cssW = canvas.clientWidth > 0 ? canvas.clientWidth : canvas.width;
  const cssH = canvas.clientHeight > 0 ? canvas.clientHeight : canvas.height;
  const wantW = Math.max(1, Math.round(cssW * dpr));
  const wantH = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== wantW || canvas.height !== wantH) {
    canvas.width = wantW;
    canvas.height = wantH;
  }
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssW;
  const h = cssH;
  ctx2d.clearRect(0, 0, w, h);

  // Studio background
  const bg = ctx2d.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, "#080812");
  bg.addColorStop(0.5, "#0d0d1a");
  bg.addColorStop(1, "#080812");
  ctx2d.fillStyle = bg;
  ctx2d.fillRect(0, 0, w, h);

  // Center zero-axis
  const midY = h / 2;
  ctx2d.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(0, midY);
  ctx2d.lineTo(w, midY);
  ctx2d.stroke();

  // Vertical musical beat grid lines
  const bpm = Number.isFinite(state.bpm) && state.bpm > 0 ? state.bpm : 126;
  const beatsInView = 4;
  for (let b = 1; b < beatsInView; b++) {
    const bx = (b / beatsInView) * w;
    ctx2d.strokeStyle = b === 2 ? "rgba(255, 255, 255, 0.12)" : "rgba(255, 255, 255, 0.05)";
    ctx2d.beginPath();
    ctx2d.moveTo(bx, 0);
    ctx2d.lineTo(bx, h);
    ctx2d.stroke();
  }

  // 64-band FFT Spectrum Analyzer (Background)
  const freq = state.frequency;
  if (freq && freq.length > 0) {
    const numBars = 64;
    const barPad = 2;
    const barW = Math.max(2, (w - 180) / numBars);
    const startX = 110;

    for (let b = 0; b < numBars; b++) {
      const bin = Math.floor((b / numBars) * (freq.length * 0.7));
      const val = freq[bin] / 255.0;
      const barH = val * (h * 0.82);
      const bx = startX + b * barW;

      const barGrad = ctx2d.createLinearGradient(0, h, 0, h - barH);
      barGrad.addColorStop(0, "rgba(34, 197, 94, 0.2)");
      barGrad.addColorStop(0.6, "rgba(56, 189, 248, 0.25)");
      barGrad.addColorStop(1, "rgba(168, 85, 247, 0.35)");

      ctx2d.fillStyle = barGrad;
      ctx2d.fillRect(bx, h - barH, barW - barPad, barH);
    }
  }

  // Mirrored Master Oscilloscope Waveform (Glowing Emerald-Cyan)
  const time = state.timeDomain;
  if (time && time.length > 0) {
    const step = Math.max(1, Math.floor(time.length / 320));
    const waveStartX = 110;
    const waveEndX = w - 70;
    const waveWidth = waveEndX - waveStartX;
    const maxAmp = h * 0.44;
    const points: { x: number; topY: number; botY: number }[] = [];

    let pIdx = 0;
    for (let tIdx = 0; tIdx < time.length; tIdx += step) {
      const norm = (time[tIdx] - 128) / 128.0;
      const amp = Math.abs(norm);
      const x = waveStartX + (pIdx / (time.length / step)) * waveWidth;
      const yOff = Math.max(1, amp * maxAmp);
      points.push({ x, topY: midY - yOff, botY: midY + yOff });
      pIdx++;
    }

    if (points.length > 1) {
      // Mirrored gradient envelope
      const grad = ctx2d.createLinearGradient(0, midY - maxAmp, 0, midY + maxAmp);
      grad.addColorStop(0, "rgba(34, 197, 94, 0.45)");
      grad.addColorStop(0.5, "rgba(6, 182, 212, 0.15)");
      grad.addColorStop(1, "rgba(34, 197, 94, 0.45)");

      ctx2d.fillStyle = grad;
      ctx2d.beginPath();
      ctx2d.moveTo(points[0].x, points[0].topY);
      for (let p = 1; p < points.length; p++) ctx2d.lineTo(points[p].x, points[p].topY);
      for (let p = points.length - 1; p >= 0; p--) ctx2d.lineTo(points[p].x, points[p].botY);
      ctx2d.closePath();
      ctx2d.fill();

      // Top glowing neon trace
      ctx2d.save();
      ctx2d.strokeStyle = "#4ade80";
      ctx2d.lineWidth = 1.8;
      ctx2d.shadowColor = "#22d3ee";
      ctx2d.shadowBlur = 9;
      ctx2d.beginPath();
      ctx2d.moveTo(points[0].x, points[0].topY);
      for (let p = 1; p < points.length; p++) ctx2d.lineTo(points[p].x, points[p].topY);
      ctx2d.stroke();

      // Bottom glowing neon trace
      ctx2d.beginPath();
      ctx2d.moveTo(points[0].x, points[0].botY);
      for (let p = 1; p < points.length; p++) ctx2d.lineTo(points[p].x, points[p].botY);
      ctx2d.stroke();
      ctx2d.restore();
    }
  }

  // Master Badge & Labels
  ctx2d.fillStyle = "#ffffff";
  ctx2d.font = "bold 13px system-ui, sans-serif";
  ctx2d.textAlign = "left";
  ctx2d.fillText("MASTER MIX", 16, 26);

  ctx2d.fillStyle = "rgba(255, 255, 255, 0.55)";
  ctx2d.font = "10px monospace";
  ctx2d.fillText("COMBINED OUTPUT", 16, 42);

  // Status LED dot
  const level = state.level;
  const isPlaying = level > 0.005;
  ctx2d.beginPath();
  ctx2d.arc(104, 22, 4, 0, 2 * Math.PI);
  ctx2d.fillStyle = isPlaying ? "#22c55e" : "rgba(255, 255, 255, 0.2)";
  if (isPlaying) {
    ctx2d.shadowColor = "#4ade80";
    ctx2d.shadowBlur = 8;
  }
  ctx2d.fill();
  ctx2d.shadowBlur = 0;

  // Master Stereo VU Meter (Right Side)
  const meterX = w - 55;
  const meterW = 42;
  const meterH = h - 36;
  const meterY = 18;

  ctx2d.fillStyle = "rgba(14, 14, 24, 0.8)";
  ctx2d.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx2d.lineWidth = 1;
  ctx2d.fillRect(meterX, meterY, meterW, meterH);
  ctx2d.strokeRect(meterX, meterY, meterW, meterH);

  const fillH = Math.max(0, Math.min(meterH, level * meterH));
  if (fillH > 0) {
    const meterGrad = ctx2d.createLinearGradient(0, meterY + meterH, 0, meterY);
    meterGrad.addColorStop(0, "#22c55e");
    meterGrad.addColorStop(0.7, "#eab308");
    meterGrad.addColorStop(1, "#ef4444");
    ctx2d.fillStyle = meterGrad;
    ctx2d.fillRect(meterX + 2, meterY + meterH - fillH + 2, meterW - 4, fillH - 4);
  }

  // Numerical dB readout
  const db = level > 0.001 ? Math.round(20 * Math.log10(level)) : -60;
  ctx2d.fillStyle = db > -3 ? "#ef4444" : "rgba(255, 255, 255, 0.85)";
  ctx2d.font = "bold 10px monospace";
  ctx2d.textAlign = "center";
  ctx2d.fillText(db > -60 ? `${db}dB` : "-inf", meterX + meterW / 2, h - 8);

  // Master Global Playhead Laser Line
  const beatPhase = (state.timeSec * bpm) / 60.0;
  const playheadX = typeof state.playheadProgress === "number"
    ? Math.max(0, Math.min(1, state.playheadProgress)) * w
    : ((beatPhase % 4) / 4) * w;
  ctx2d.save();
  ctx2d.strokeStyle = "#ffffff";
  ctx2d.lineWidth = 2;
  ctx2d.shadowColor = "#4ade80";
  ctx2d.shadowBlur = 10;
  ctx2d.globalAlpha = 0.9;
  ctx2d.beginPath();
  ctx2d.moveTo(playheadX, 0);
  ctx2d.lineTo(playheadX, h);
  ctx2d.stroke();
  ctx2d.restore();
}

/** One recently scheduled slice, pushed by the app when it schedules audio. */
export interface SliceArcEvent {
  stem: StemName;
  trackId: string;
  /** Visual-clock seconds (performance.now()/1000), not audio time. */
  atSec: number;
}

export interface RingsState {
  events: SliceArcEvent[];
  bpm: number;
  /** Visual-clock seconds (performance.now()/1000). */
  timeSec: number;
}

export interface RingsOptions {
  /** How long a slice arc stays visible, in seconds. Default: 2 beats. */
  windowSec?: number;
}

function hashTrack(trackId: string): number {
  let h = 0;
  for (let i = 0; i < trackId.length; i++) {
    h = (h * 31 + trackId.charCodeAt(i)) >>> 0;
  }
  return h;
}

/**
 * Draw 4 concentric rings (inner=drums, then bass, other, vocals outer).
 * Recent slice arcs are placed around each ring by track hash; a playhead
 * dot rotates once per beat at the current bpm.
 */
export function drawRings(
  ctx2d: CanvasRenderingContext2D,
  state: RingsState,
  opts: RingsOptions = {},
): void {
  const canvas = ctx2d.canvas;
  const dpr = typeof window !== "undefined" && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
  const cssW = canvas.clientWidth > 0 ? canvas.clientWidth : canvas.width;
  const cssH = canvas.clientHeight > 0 ? canvas.clientHeight : canvas.height;
  const wantW = Math.max(1, Math.round(cssW * dpr));
  const wantH = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== wantW || canvas.height !== wantH) {
    canvas.width = wantW;
    canvas.height = wantH;
  }
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssW;
  const h = cssH;
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.fillStyle = "#0b0b14";
  ctx2d.fillRect(0, 0, w, h);

  const bpm = Number.isFinite(state.bpm) && state.bpm > 0 ? state.bpm : 120;
  const windowSec = typeof opts.windowSec === "number" && opts.windowSec > 0 ? opts.windowSec : 120 / bpm;
  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.max(10, Math.min(w, h) / 2 - 14);
  const band = maxR / (RING_ORDER.length + 0.5);

  // Playhead: one revolution per beat.
  const phase = (state.timeSec * bpm) / 60;
  const playAngle = 2 * Math.PI * (phase - Math.floor(phase)) - Math.PI / 2;

  for (let ring = 0; ring < RING_ORDER.length; ring++) {
    const stem = RING_ORDER[ring];
    const r = band * (ring + 1);
    const color = STEM_COLORS[stem];

    ctx2d.beginPath();
    ctx2d.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx2d.strokeStyle = color + "2e";
    ctx2d.lineWidth = Math.max(4, band * 0.42);
    ctx2d.stroke();

    // Recent slice arcs for this stem.
    for (const ev of state.events) {
      if (ev.stem !== stem) continue;
      const age = state.timeSec - ev.atSec;
      if (age < 0 || age > windowSec) continue;
      const alpha = 1 - age / windowSec;
      const center = ((hashTrack(ev.trackId) % 628) / 100) - Math.PI / 2;
      const span = 0.6;
      ctx2d.beginPath();
      ctx2d.arc(cx, cy, r, center - span / 2, center + span / 2);
      ctx2d.strokeStyle = color;
      ctx2d.globalAlpha = 0.25 + 0.75 * alpha;
      ctx2d.lineWidth = Math.max(5, band * 0.5);
      ctx2d.lineCap = "round";
      ctx2d.stroke();
      ctx2d.globalAlpha = 1;
    }

    // Playhead dot on this ring.
    const px = cx + Math.cos(playAngle) * r;
    const py = cy + Math.sin(playAngle) * r;
    ctx2d.beginPath();
    ctx2d.arc(px, py, Math.max(3, band * 0.16), 0, 2 * Math.PI);
    ctx2d.fillStyle = "#ffffff";
    ctx2d.globalAlpha = 0.9;
    ctx2d.fill();
    ctx2d.globalAlpha = 1;

    // Stem label at the bottom of the ring.
    ctx2d.fillStyle = color;
    ctx2d.globalAlpha = 0.85;
    ctx2d.font = "11px system-ui, sans-serif";
    ctx2d.textAlign = "center";
    ctx2d.fillText(stem, cx, cy + r + 11);
    ctx2d.globalAlpha = 1;
  }
}

export interface NebulaPoint {
  /** Normalized 0..1. */
  x: number;
  /** Normalized 0..1. */
  y: number;
  stem: StemName;
  trackId?: string;
}

export interface NebulaFrame {
  /** Visual-clock seconds (performance.now()/1000). */
  timeSec: number;
  bpm: number;
  /** Optional per-stem RMS levels 0..1 for reticle pulse. */
  levels?: Partial<Record<StemName, number>>;
}

export type ReticleMoveHandler = (stem: StemName, x: number, y: number) => void;

export interface NebulaRendererOptions {
  onReticleMove?: ReticleMoveHandler;
}

const RETICLE_STEMS: readonly StemName[] = ["drums", "bass", "vocals", "other"];
const DEFAULT_RETICLES: Record<StemName, { x: number; y: number }> = {
  drums: { x: 0.25, y: 0.3 },
  bass: { x: 0.7, y: 0.3 },
  vocals: { x: 0.3, y: 0.72 },
  other: { x: 0.72, y: 0.7 },
};
const GRID_N = 24;

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

export class NebulaRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private points: NebulaPoint[] = [];
  private alphas: number[] = [];
  private reticleMap: Record<StemName, { x: number; y: number }> = {
    drums: { ...DEFAULT_RETICLES.drums },
    bass: { ...DEFAULT_RETICLES.bass },
    vocals: { ...DEFAULT_RETICLES.vocals },
    other: { ...DEFAULT_RETICLES.other },
  };
  private basePositions: Record<StemName, { x: number; y: number }> = {
    drums: { ...DEFAULT_RETICLES.drums },
    bass: { ...DEFAULT_RETICLES.bass },
    vocals: { ...DEFAULT_RETICLES.vocals },
    other: { ...DEFAULT_RETICLES.other },
  };
  private lastDriftNotifySec: number = 0;

  // Autonomous Lissajous drift profiles with unique incommensurate frequencies
  private static readonly DRIFT_PROFILES: Record<StemName, { sx: number; sy: number; px: number; py: number; r: number }> = {
    drums:  { sx: 0.048, sy: 0.035, px: 0.0, py: 1.4, r: 0.075 },
    bass:   { sx: 0.032, sy: 0.046, px: 2.2, py: 0.7, r: 0.080 },
    vocals: { sx: 0.039, sy: 0.028, px: 4.1, py: 3.3, r: 0.070 },
    other:  { sx: 0.044, sy: 0.039, px: 1.6, py: 5.2, r: 0.075 },
  };

  private onReticleMove?: ReticleMoveHandler;
  private dragStem: StemName | null = null;
  private activeSliceMarkers: Partial<Record<StemName, { x: number; y: number; timeSec: number }>> = {};

  setActiveSlice(stem: StemName, x: number, y: number, timeSec: number): void {
    this.activeSliceMarkers[stem] = { x: clamp01(x), y: clamp01(y), timeSec };
  }

  private handlePointerDown: (e: PointerEvent) => void;
  private handlePointerMove: (e: PointerEvent) => void;
  private handlePointerUp: (e: PointerEvent) => void;

  constructor(canvas: HTMLCanvasElement, opts: NebulaRendererOptions = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (ctx === null) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.onReticleMove = opts.onReticleMove;
    canvas.style.touchAction = "none";

    this.handlePointerDown = (e: PointerEvent) => {
      const hit = this.hitTest(e);
      if (hit !== null) {
        this.dragStem = hit;
        try {
          canvas.setPointerCapture(e.pointerId);
        } catch {
          // setPointerCapture may throw for mouse; dragging still works.
        }
        this.moveReticleFromEvent(hit, e);
        e.preventDefault();
      }
    };
    this.handlePointerMove = (e: PointerEvent) => {
      if (this.dragStem !== null) {
        this.moveReticleFromEvent(this.dragStem, e);
        e.preventDefault();
      }
    };
    this.handlePointerUp = (e: PointerEvent) => {
      if (this.dragStem !== null) {
        this.dragStem = null;
        try {
          canvas.releasePointerCapture(e.pointerId);
        } catch {
          // Ignore release failures.
        }
      }
    };
    canvas.addEventListener("pointerdown", this.handlePointerDown);
    canvas.addEventListener("pointermove", this.handlePointerMove);
    canvas.addEventListener("pointerup", this.handlePointerUp);
    canvas.addEventListener("pointercancel", this.handlePointerUp);
    this.resize();
  }

  /** Sync backing store to CSS size x devicePixelRatio. Call on layout. */
  resize(): void {
    const canvas = this.canvas;
    const dpr = typeof window !== "undefined" && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
    const cssW = canvas.clientWidth > 0 ? canvas.clientWidth : canvas.width;
    const cssH = canvas.clientHeight > 0 ? canvas.clientHeight : canvas.height;
    const wantW = Math.max(1, Math.round(cssW * dpr));
    const wantH = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== wantW || canvas.height !== wantH) {
      canvas.width = wantW;
      canvas.height = wantH;
    }
  }

  /** Replace the scatter set; coordinates are clamped to 0..1. */
  setPoints(points: NebulaPoint[]): void {
    this.points = points.map((p) => ({ stem: p.stem, trackId: p.trackId, x: clamp01(p.x), y: clamp01(p.y) }));
    // Density estimate on a coarse grid; denser cells render more opaque.
    const counts = new Float32Array(GRID_N * GRID_N);
    const cellOf = (x: number, y: number): number => {
      const gx = Math.min(GRID_N - 1, Math.floor(x * GRID_N));
      const gy = Math.min(GRID_N - 1, Math.floor(y * GRID_N));
      return gy * GRID_N + gx;
    };
    for (const p of this.points) counts[cellOf(p.x, p.y)] += 1;
    let max = 1;
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] > max) max = counts[i];
    }
    this.alphas = this.points.map((p) => 0.25 + 0.65 * (counts[cellOf(p.x, p.y)] / max));
  }

  getReticle(stem: StemName): { x: number; y: number } {
    const r = this.reticleMap[stem];
    return { x: r.x, y: r.y };
  }

  get reticles(): Record<StemName, { x: number; y: number }> {
    return {
      drums: this.getReticle("drums"),
      bass: this.getReticle("bass"),
      vocals: this.getReticle("vocals"),
      other: this.getReticle("other"),
    };
  }

  /** Programmatic move (no callback, so server echoes cannot loop). */
  setReticle(stem: StemName, x: number, y: number): void {
    const p = { x: clamp01(x), y: clamp01(y) };
    this.reticleMap[stem] = p;
    this.basePositions[stem] = { ...p };
  }

  render(frame: NebulaFrame): void {
    const canvas = this.canvas;
    const ctx = this.ctx;
    this.resize();
    const dpr = typeof window !== "undefined" && window.devicePixelRatio > 0 ? window.devicePixelRatio : 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const rect = canvas.getBoundingClientRect();
    const w = rect.width > 0 ? rect.width : canvas.width / dpr;
    const h = rect.height > 0 ? rect.height : canvas.height / dpr;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b0b14";
    ctx.fillRect(0, 0, w, h);

    // Apply autonomous organic Lissajous drift when stems are not manually dragged
    const t = frame.timeSec;
    const shouldNotify = (t - this.lastDriftNotifySec) > 0.08;
    if (shouldNotify) {
      this.lastDriftNotifySec = t;
    }

    for (const stem of RETICLE_STEMS) {
      if (this.dragStem !== stem) {
        const prof = NebulaRenderer.DRIFT_PROFILES[stem];
        const base = this.basePositions[stem];
        const dx = Math.sin(t * prof.sx * 2 * Math.PI + prof.px) * prof.r;
        const dy = Math.cos(t * prof.sy * 2 * Math.PI + prof.py) * prof.r;
        const driftedX = clamp01(base.x + dx);
        const driftedY = clamp01(base.y + dy);
        this.reticleMap[stem] = { x: driftedX, y: driftedY };
        if (shouldNotify && this.onReticleMove) {
          this.onReticleMove(stem, driftedX, driftedY);
        }
      }
    }

    // Faint grid.
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
      ctx.beginPath();
      ctx.moveTo((w * i) / 6, 0);
      ctx.lineTo((w * i) / 6, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, (h * i) / 6);
      ctx.lineTo(w, (h * i) / 6);
      ctx.stroke();
    }

    // Scatter points.
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i];
      ctx.globalAlpha = this.alphas[i] ?? 0.5;
      ctx.fillStyle = STEM_COLORS[p.stem];
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Gravity reticles with beat-synced bouncing pulse (draggable to steer sound)
    const bpm = Number.isFinite(frame.bpm) && frame.bpm > 0 ? frame.bpm : 120;
    const beatPhase = (frame.timeSec * bpm) / 60;
    const pulse = 0.5 + 0.5 * Math.sin(2 * Math.PI * (beatPhase - Math.floor(beatPhase)));
    for (const stem of RETICLE_STEMS) {
      const r = this.reticleMap[stem];
      const px = r.x * w;
      const py = r.y * h;
      const level = frame.levels !== undefined ? (frame.levels[stem] ?? 0) : 0;
      const radius = 12 + 4 * pulse + 12 * Math.min(1, Math.max(0, level));
      const color = STEM_COLORS[stem];

      // Subtle drift orbit guide ring
      const prof = NebulaRenderer.DRIFT_PROFILES[stem];
      const base = this.basePositions[stem];
      ctx.beginPath();
      ctx.ellipse(base.x * w, base.y * h, prof.r * w, prof.r * h, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.globalAlpha = 0.18;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // Outer glow pulse
      ctx.beginPath();
      ctx.arc(px, py, radius + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3 * (0.5 + 0.5 * pulse);
      ctx.stroke();

      // Main pulsing circle
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.globalAlpha = 0.95;
      ctx.stroke();

      // Center crosshairs
      ctx.beginPath();
      ctx.moveTo(px - radius - 6, py);
      ctx.lineTo(px + radius + 6, py);
      ctx.moveTo(px, py - radius - 6);
      ctx.lineTo(px, py + radius + 6);
      ctx.globalAlpha = 0.65;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Stem label and real-time parameter feedback
      ctx.fillStyle = color;
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.textAlign = "center";
      const stemLabels: Record<string, string> = {
        drums: "Drum",
        bass: "Bass",
        other: "Other",
        vocals: "Vocal",
      };
      ctx.fillText(stemLabels[stem] || stem, px, py - radius - 8);

      // Timbre & Groove parameter readout
      ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
      ctx.font = "9px monospace";
      ctx.fillText(`X:${Math.round(r.x * 100)}% Y:${Math.round(r.y * 100)}%`, px, py + radius + 13);
    }

    // Active playing slice highlights with white pixel and expanding ping ring
    for (const stem of RETICLE_STEMS) {
      const marker = this.activeSliceMarkers[stem];
      if (!marker) continue;
      const age = frame.timeSec - marker.timeSec;
      if (age < 0 || age > 1.2) continue;
      const progress = age / 1.2;
      const alpha = Math.max(0, 1 - progress);
      const pingR = 4 + 18 * progress;
      const px = marker.x * w;
      const py = marker.y * h;
      const color = STEM_COLORS[stem];

      ctx.save();
      // Expanding ping ring
      ctx.beginPath();
      ctx.arc(px, py, pingR, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * alpha;
      ctx.globalAlpha = 0.8 * alpha;
      ctx.stroke();

      // White pixel center highlight
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.restore();
    }
  }

  private toNormalized(e: PointerEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width > 0 ? rect.width : 1;
    const h = rect.height > 0 ? rect.height : 1;
    return {
      x: clamp01((e.clientX - rect.left) / w),
      y: clamp01((e.clientY - rect.top) / h),
    };
  }

  private hitTest(e: PointerEvent): StemName | null {
    const rect = this.canvas.getBoundingClientRect();
    // Hit radius: 20 CSS px expressed in normalized units.
    const rx = 20 / (rect.width > 0 ? rect.width : 1);
    const ry = 20 / (rect.height > 0 ? rect.height : 1);
    const p = this.toNormalized(e);
    let best: StemName | null = null;
    let bestDist = Infinity;
    for (const stem of RETICLE_STEMS) {
      const r = this.reticleMap[stem];
      const d = Math.hypot((p.x - r.x) / Math.max(rx, 1e-6), (p.y - r.y) / Math.max(ry, 1e-6));
      if (d <= 1 && d < bestDist) {
        best = stem;
        bestDist = d;
      }
    }
    return best;
  }

  private moveReticleFromEvent(stem: StemName, e: PointerEvent): void {
    const p = this.toNormalized(e);
    this.reticleMap[stem] = p;
    this.basePositions[stem] = { ...p };
    if (this.onReticleMove !== undefined) {
      this.onReticleMove(stem, p.x, p.y);
    }
  }

  dispose(): void {
    this.canvas.removeEventListener("pointerdown", this.handlePointerDown);
    this.canvas.removeEventListener("pointermove", this.handlePointerMove);
    this.canvas.removeEventListener("pointerup", this.handlePointerUp);
    this.canvas.removeEventListener("pointercancel", this.handlePointerUp);
    this.dragStem = null;
  }
}
