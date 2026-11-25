// Real-time Web Audio Vocal Tract Synthesizer driven by the Singing Head's
// reconstructed 14-D Voice Workshop articulators, 44-section area function,
// Webster transmission-line transfer function H_vocal(f), and glottal excitation.

import { SingingHeadArticulation } from "../singing_head_model";
import { AreaFunction, N_SECTIONS, sectionsFromGeometry } from "./tract/areaFunction";
import { TractGeometry } from "./tract/geometry";
import { Articulators } from "./tract/types";

const N_FREQ_GRID = 80;
const FREQ_MIN_HZ = 75.0;
const FREQ_MAX_HZ = 9200.0;

const FREQ_GRID = new Float32Array(N_FREQ_GRID);
for (let i = 0; i < N_FREQ_GRID; i++) {
  const u = i / (N_FREQ_GRID - 1);
  FREQ_GRID[i] = FREQ_MIN_HZ * Math.pow(FREQ_MAX_HZ / FREQ_MIN_HZ, u);
}

export interface TractAcousticProfile {
  formantHz: [number, number, number, number];
  formantQ: [number, number, number, number];
  formantGain: [number, number, number, number];
  nasalGain: number;
  sibCenterHz: number;
  fricGain: number;
  openDrive: number;
  minArea: number;
  minIdxNorm: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Evaluate the 44-section Webster transmission-line transfer function H_vocal(f)
 * and constriction turbulence H_fric(f) matching scripts/generate_tract_soundbank.py,
 * and extract the 4 resonant formant poles, nasal coupling, and sibilance peak.
 */
export function evaluateTractAcoustics(
  tractArt: Articulators,
  geom: TractGeometry,
  hVocalScratch?: Float32Array,
  sectionsScratch?: AreaFunction
): TractAcousticProfile {
  const sec = sectionsFromGeometry(geom, N_SECTIONS, sectionsScratch);
  const areas = sec.areas;
  const dx = Math.max(0.2, sec.dx);
  const effLip = Math.max(0.04, geom.lipArea);
  const velum = clamp(tractArt.velum, 0, 1);
  const lipTuck = clamp(tractArt.lipTuck, 0, 1);

  let minArea = 999.0;
  let minIdx = 0;
  for (let s = 0; s < N_SECTIONS; s++) {
    if (areas[s] < minArea) {
      minArea = areas[s];
      minIdx = s;
    }
  }
  const minIdxNorm = minIdx / (N_SECTIONS - 1);

  const rho = 1.14e-3;
  const cSnd = 35000.0;
  const alpha = 0.0028;
  const rLip = Math.sqrt(Math.max(0.01, effLip / Math.PI));
  const lRad = (8.0 * rho) / (3.0 * Math.PI * Math.PI * rLip);
  const coshA = Math.cosh(alpha * dx);
  const sinhA = Math.sinh(alpha * dx);

  const hVocal = hVocalScratch && hVocalScratch.length >= N_FREQ_GRID
    ? hVocalScratch
    : new Float32Array(N_FREQ_GRID);

  let hOralSum = 0;
  for (let fi = 0; fi < N_FREQ_GRID; fi++) {
    const f = FREQ_GRID[fi];
    const w = 2.0 * Math.PI * f;
    const kdx = (w / cSnd) * dx;
    const cosK = Math.cos(kdx);
    const sinK = Math.sin(kdx);

    const coshRe = coshA * cosK;
    const coshIm = sinhA * sinK;
    const sinhRe = sinhA * cosK;
    const sinhIm = coshA * sinK;

    let pRe = 0.015 * ((rho * cSnd) / effLip) * Math.pow(w / 6000.0, 2);
    let pIm = w * lRad;
    let uRe = 1.0;
    let uIm = 0.0;

    for (let s = N_SECTIONS - 1; s >= 0; s--) {
      const a = Math.max(0.025, areas[s]);
      const z = (rho * cSnd) / a;
      const invZ = 1.0 / z;

      const suRe = coshRe * uRe - coshIm * uIm;
      const suIm = coshRe * uIm + coshIm * uRe;
      const spRe = (sinhRe * pRe - sinhIm * pIm) * invZ;
      const spIm = (sinhRe * pIm + sinhIm * pRe) * invZ;
      const uNextRe = suRe + spRe;
      const uNextIm = suIm + spIm;

      const zsuRe = z * (sinhRe * uRe - sinhIm * uIm);
      const zsuIm = z * (sinhRe * uIm + sinhIm * uRe);
      const cpRe = coshRe * pRe - coshIm * pIm;
      const cpIm = coshRe * pIm + coshIm * pRe;

      uRe = uNextRe;
      uIm = uNextIm;
      pRe = zsuRe + cpRe;
      pIm = zsuIm + cpIm;
    }

    const uMag = Math.sqrt(Math.max(1e-18, uRe * uRe + uIm * uIm));
    const hOral = Math.min(40.0, 1.0 / uMag);
    hOralSum += hOral;
    hVocal[fi] = hOral;
  }

  // Pick the 4 oral formant peaks in canonical physiological bands with minimum spacing.
  // Use the pure oral transfer function hOral (before adding the parallel nasal branch,
  // which has its own dedicated resonator in VocalTractSynthesizer).
  const bands: readonly [number, number, number][] = [
    [190.0, 920.0, 520.0],    // F1
    [680.0, 2750.0, 1480.0],  // F2
    [1850.0, 3850.0, 2650.0], // F3
    [2950.0, 5100.0, 3650.0], // F4
  ];

  const formantHz: [number, number, number, number] = [520, 1480, 2650, 3650];
  const formantQ: [number, number, number, number] = [5.5, 7.5, 9.0, 10.5];
  const formantGain: [number, number, number, number] = [1.0, 0.75, 0.45, 0.28];

  let prevHz = 0.0;
  for (let k = 0; k < 4; k++) {
    const [bLo, bHi, fallbackHz] = bands[k];
    const minSpacing = k === 0 ? 0 : k === 1 ? 260 : 320;
    const loHz = Math.max(bLo, prevHz + minSpacing);
    let bestIdx = -1;
    let bestVal = -1;

    // First look for a local maximum inside [loHz, bHi]
    for (let fi = 1; fi < N_FREQ_GRID - 1; fi++) {
      const f = FREQ_GRID[fi];
      if (f < loHz || f > bHi) continue;
      const v = hVocal[fi];
      if (v >= hVocal[fi - 1] && v >= hVocal[fi + 1] && v > bestVal) {
        bestVal = v;
        bestIdx = fi;
      }
    }
    // Fallback to global maximum inside [loHz, bHi]
    if (bestIdx < 0) {
      for (let fi = 0; fi < N_FREQ_GRID; fi++) {
        const f = FREQ_GRID[fi];
        if (f < loHz || f > bHi) continue;
        if (hVocal[fi] > bestVal) {
          bestVal = hVocal[fi];
          bestIdx = fi;
        }
      }
    }

    if (bestIdx >= 0) {
      // Parabolic interpolation in log-frequency space for smooth continuous formant tracking
      let fPeak = FREQ_GRID[bestIdx];
      if (bestIdx > 0 && bestIdx < N_FREQ_GRID - 1) {
        const y0 = Math.log(Math.max(1e-6, hVocal[bestIdx - 1]));
        const y1 = Math.log(Math.max(1e-6, hVocal[bestIdx]));
        const y2 = Math.log(Math.max(1e-6, hVocal[bestIdx + 1]));
        const denom = 2 * (2 * y1 - y0 - y2);
        if (denom > 1e-6) {
          const delta = clamp((y2 - y0) / denom, -0.5, 0.5);
          const f0 = FREQ_GRID[bestIdx - 1];
          const f2 = FREQ_GRID[bestIdx + 1];
          fPeak = clamp(FREQ_GRID[bestIdx] + delta * 0.5 * (f2 - f0), loHz, bHi);
        }
      }
      formantHz[k] = fPeak;
      // Estimate Q from local sharpness of peak relative to neighbors
      const neighborMean =
        0.5 *
        (hVocal[Math.max(0, bestIdx - 2)] +
          hVocal[Math.min(N_FREQ_GRID - 1, bestIdx + 2)]);
      const sharpness = clamp(bestVal / Math.max(0.05, neighborMean), 1.0, 5.0);
      const baseQ = 4.2 + k * 1.6;
      formantQ[k] = clamp(baseQ + (sharpness - 1.0) * 2.2, 3.5, 16.0);
      formantGain[k] = clamp(bestVal / 10.0, 0.12, 1.65);
    } else {
      formantHz[k] = Math.max(loHz, fallbackHz);
    }
    prevHz = formantHz[k];
  }

  const hOralMean = clamp(hOralSum / N_FREQ_GRID, 0.1, 4.0);
  for (let fi = 0; fi < N_FREQ_GRID; fi++) {
    const f = FREQ_GRID[fi];
    const nasalMurmur =
      2.6 * Math.exp(-0.5 * Math.pow((f - 245.0) / 115.0, 2)) +
      0.85 * Math.exp(-0.5 * Math.pow((f - 1050.0) / 260.0, 2));
    const oralAntires = 1.0 - 0.55 * velum * Math.exp(-0.5 * Math.pow((f - 960.0) / 280.0, 2));
    hVocal[fi] =
      hVocal[fi] * oralAntires * (1.0 - 0.52 * velum) +
      velum * nasalMurmur * (0.5 + 0.5 * hOralMean);
  }

  const jawNorm = clamp(tractArt.jaw, 0, 1);
  const lipOpenNorm = clamp((tractArt.lipOpen + 0.4) / 1.6, 0, 1);
  const lipAreaNorm = clamp(effLip / 4.0, 0, 1);
  const openDrive = clamp(0.50 * jawNorm + 0.35 * lipOpenNorm + 0.15 * lipAreaNorm, 0.02, 1.0);

  const sibCenterHz = 2400.0 + 4900.0 * clamp((minIdxNorm - 0.45) / 0.50, 0, 1);
  const tightness = clamp((0.36 - minArea) / 0.32, 0, 1);
  const tuckAmt = clamp((lipTuck - 0.20) / 0.70, 0, 1);
  const fricGain = clamp(tightness + 0.65 * tuckAmt, 0, 1.2);

  return {
    formantHz,
    formantQ,
    formantGain,
    nasalGain: velum,
    sibCenterHz,
    fricGain,
    openDrive,
    minArea,
    minIdxNorm,
  };
}

import {
  CHIMERA_ARTIC_LEN,
  CHIMERA_SOURCE_LEN,
  STUDIO_MIC_EQ,
  mapChimeraParamsToArrays,
  mapWubToChimeraArrays,
} from "./chimera_mapping";

let cachedChimeraWasmPromise: Promise<ArrayBuffer> | null = null;
const registeredContexts = new WeakSet<BaseAudioContext>();

function fetchChimeraWasm(): Promise<ArrayBuffer> {
  if (cachedChimeraWasmPromise === null) {
    cachedChimeraWasmPromise = (async () => {
      const resp = await fetch("./chimera.wasm");
      if (!resp.ok) {
        throw new Error(`Failed to fetch chimera.wasm (${resp.status})`);
      }
      return await resp.arrayBuffer();
    })();
  }
  return cachedChimeraWasmPromise;
}

/**
 * Real-time Web Audio synthesizer producing the acoustic output of the Singing Head's
 * reconstructed vocal tract geometry and glottal parameters via the CHIMERA branched
 * physical-modeling airway AudioWorklet + Studio-Mic EQ + RMS level lock.
 */
export class VocalTractSynthesizer {
  private ctx: AudioContext | null = null;
  private outputNode: GainNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private workletGain: GainNode | null = null;
  private eqHp: BiquadFilterNode | null = null;
  private eqPeak: BiquadFilterNode | null = null;
  private eqShelf: BiquadFilterNode | null = null;
  private rawAnalyser: AnalyserNode | null = null;
  private readonly rawScratch = new Float32Array(256);

  private workletReady = false;
  private workletInitializing = false;
  private initEpoch = 0;

  private readonly hVocalScratch = new Float32Array(N_FREQ_GRID);
  private sectionsScratch: AreaFunction | undefined = undefined;
  private readonly artic17 = new Float64Array(CHIMERA_ARTIC_LEN);
  private readonly source12 = new Float64Array(CHIMERA_SOURCE_LEN);
  private readonly articMsg = new Float32Array(CHIMERA_ARTIC_LEN);
  private readonly sourceMsg = new Float32Array(CHIMERA_SOURCE_LEN);

  private smoothedF0 = 165.0;
  private smoothedRms = 0.0;
  private smoothedRawRms = 0.14;
  private active = false;

  /**
   * Lazily build and connect the CHIMERA vocal tract synthesizer graph to `destination`.
   */
  public attach(ctx: AudioContext, destination: AudioNode): void {
    if (this.ctx === ctx && this.outputNode !== null) {
      return;
    }
    this.dispose();
    this.ctx = ctx;

    const out = ctx.createGain();
    out.gain.value = 0.0;
    out.connect(destination);
    this.outputNode = out;

    // Studio-Mic EQ chain matching offline soundbank feature extraction
    const eqHp = ctx.createBiquadFilter();
    eqHp.type = "highpass";
    eqHp.frequency.value = STUDIO_MIC_EQ.highpassHz;
    eqHp.Q.value = STUDIO_MIC_EQ.highpassQ;
    this.eqHp = eqHp;

    const eqPeak = ctx.createBiquadFilter();
    eqPeak.type = "peaking";
    eqPeak.frequency.value = STUDIO_MIC_EQ.peakHz;
    eqPeak.Q.value = STUDIO_MIC_EQ.peakQ;
    eqPeak.gain.value = STUDIO_MIC_EQ.peakGainDb;
    this.eqPeak = eqPeak;

    const eqShelf = ctx.createBiquadFilter();
    eqShelf.type = "highshelf";
    eqShelf.frequency.value = STUDIO_MIC_EQ.highshelfHz;
    eqShelf.Q.value = STUDIO_MIC_EQ.highshelfQ;
    eqShelf.gain.value = STUDIO_MIC_EQ.highshelfGainDb;
    this.eqShelf = eqShelf;

    const rawAnalyser = ctx.createAnalyser();
    rawAnalyser.fftSize = 256;
    rawAnalyser.smoothingTimeConstant = 0.5;
    this.rawAnalyser = rawAnalyser;

    const wGain = ctx.createGain();
    wGain.gain.value = 1.0;
    this.workletGain = wGain;

    wGain.connect(eqHp);
    eqHp.connect(eqPeak);
    eqPeak.connect(eqShelf);
    eqShelf.connect(rawAnalyser);
    rawAnalyser.connect(out);

    void this.initWorklet(ctx, wGain, ++this.initEpoch);
  }

  private async initWorklet(
    ctx: AudioContext,
    targetNode: AudioNode,
    epoch: number
  ): Promise<void> {
    if (this.workletInitializing || typeof AudioWorkletNode === "undefined" || !ctx.audioWorklet) {
      return;
    }
    this.workletInitializing = true;
    try {
      const wasmPromise = fetchChimeraWasm();
      if (!registeredContexts.has(ctx)) {
        await ctx.audioWorklet.addModule("./chimera_worklet.js");
        registeredContexts.add(ctx);
      }
      const wasmBytes = await wasmPromise;
      if (epoch !== this.initEpoch || this.ctx !== ctx) {
        return;
      }

      const node = new AudioWorkletNode(ctx, "wub-chimera-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      node.port.onmessage = (ev: MessageEvent<{ type?: string; message?: string }>) => {
        if (epoch !== this.initEpoch) return;
        const msg = ev.data;
        if (msg?.type === "ready") {
          this.workletReady = true;
        } else if (msg?.type === "error") {
          console.warn("[Chimera Worklet]", msg.message);
        }
      };
      node.connect(targetNode);
      this.workletNode = node;
      // Copy wasmBytes so the cached ArrayBuffer stays intact across context resets
      node.port.postMessage({ type: "wasm", bytes: wasmBytes.slice(0) });
    } catch (err) {
      console.warn("[Chimera Worklet] failed to initialize:", err);
    } finally {
      this.workletInitializing = false;
    }
  }

  public setEnabled(enabled: boolean): void {
    this.active = enabled;
    if (!enabled && this.ctx && this.outputNode) {
      this.outputNode.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.015);
      this.workletNode?.port.postMessage({ type: "gate", level: 0.0 });
    }
  }

  public isEnabled(): boolean {
    return this.active;
  }

  /**
   * Update the CHIMERA vocal tract synthesizer parameters at 60 FPS from the
   * Singing Head's live reconstructed vocal tract articulators and glottal state.
   */
  public update(
    art: SingingHeadArticulation,
    tractArt: Articulators,
    geom: TractGeometry,
    isMuted: boolean
  ): TractAcousticProfile {
    const profile = evaluateTractAcoustics(
      tractArt,
      geom,
      this.hVocalScratch,
      this.sectionsScratch
    );

    if (!this.ctx || !this.outputNode || !this.active) {
      return profile;
    }

    const t = this.ctx.currentTime;
    const quiet = isMuted || art.rms < 0.0025;
    if (quiet) {
      this.smoothedRms *= 0.78;
      this.outputNode.gain.setTargetAtTime(0.0, t, 0.018);
      if (this.workletNode && this.workletReady) {
        this.workletNode.port.postMessage({ type: "gate", level: 0.0 });
      }
      return profile;
    }

    // Track fundamental pitch matching training regime (use measured pred.f0Hz when voiced, else neural f0HzPred)
    const ch = art.chimera;
    if (art.f0Hz > 65 && art.voicing > 0.30) {
      const targetF0 = clamp(art.f0Hz, 70, 760);
      const ratio = targetF0 / Math.max(40, this.smoothedF0);
      const snap = this.smoothedRms < 0.008 || ratio > 1.09 || ratio < 0.917 ? 1.0 : 0.85;
      this.smoothedF0 += (targetF0 - this.smoothedF0) * snap;
    } else if (ch && ch.f0HzPred > 65) {
      this.smoothedF0 += (clamp(ch.f0HzPred, 70, 760) - this.smoothedF0) * 0.65;
    } else {
      const larynxShift = Math.pow(2, (tractArt.larynx * 2.2) / 12.0);
      const fallbackF0 = clamp((130 + art.pitchNorm * 190) * larynxShift, 85, 520);
      this.smoothedF0 += (fallbackF0 - this.smoothedF0) * 0.25;
    }

    const centroidDrive = clamp(((art.centroidHz || 1400) - 500) / 3000, 0, 1);

    const mapped = ch
      ? mapChimeraParamsToArrays(ch, art, isMuted, this.smoothedF0, this.artic17, this.source12)
      : mapWubToChimeraArrays(
          tractArt,
          art,
          isMuted,
          this.smoothedF0,
          centroidDrive,
          this.artic17,
          this.source12
        );

    if (this.workletNode && this.workletReady) {
      for (let i = 0; i < CHIMERA_ARTIC_LEN; i++) this.articMsg[i] = this.artic17[i];
      for (let i = 0; i < CHIMERA_SOURCE_LEN; i++) this.sourceMsg[i] = this.source12[i];
      this.workletNode.port.postMessage({
        type: "params",
        artic: this.articMsg,
        source: this.sourceMsg,
        gate: mapped.gate,
        femaleMix: mapped.femaleMix,
      });
    }

    // Dynamic RMS matching so CHIMERA's waveguide output tracks the track audio's exact envelope
    if (this.rawAnalyser) {
      this.rawAnalyser.getFloatTimeDomainData(this.rawScratch);
      let sq = 0;
      for (let i = 0; i < this.rawScratch.length; i++) {
        const s = this.rawScratch[i];
        sq += s * s;
      }
      const instRawRms = Math.sqrt(sq / this.rawScratch.length);
      if (instRawRms > 0.004) {
        this.smoothedRawRms += (instRawRms - this.smoothedRawRms) * 0.35;
      }
    }

    const rmsAlpha = art.rms > this.smoothedRms ? 0.90 : 0.65;
    this.smoothedRms += (art.rms - this.smoothedRms) * rmsAlpha;
    const matchedGain = clamp(
      this.smoothedRms / Math.max(0.02, this.smoothedRawRms),
      0.0,
      3.0
    );
    this.outputNode.gain.setTargetAtTime(matchedGain, t, 0.007);

    return profile;
  }

  public dispose(): void {
    this.initEpoch++;
    this.workletReady = false;
    this.workletInitializing = false;
    try {
      this.workletNode?.port.postMessage({ type: "gate", level: 0.0 });
      this.workletNode?.disconnect();
      this.outputNode?.disconnect();
    } catch {
      // Ignore already disconnected nodes
    }
    this.workletNode = null;
    this.workletGain = null;
    this.eqHp = null;
    this.eqPeak = null;
    this.eqShelf = null;
    this.rawAnalyser = null;
    this.outputNode = null;
    this.ctx = null;
  }
}
