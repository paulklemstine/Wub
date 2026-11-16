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

/**
 * Real-time Web Audio synthesizer producing the acoustic output of the Singing Head's
 * reconstructed vocal tract geometry and glottal parameters.
 */
export class VocalTractSynthesizer {
  private ctx: AudioContext | null = null;
  private outputNode: GainNode | null = null;

  private glottalOsc: OscillatorNode | null = null;
  private subOsc: OscillatorNode | null = null;
  private subGainNode: GainNode | null = null;
  private vibratoOsc: OscillatorNode | null = null;
  private vibratoGain: GainNode | null = null;

  private voicedGain: GainNode | null = null;
  private tiltFilter: BiquadFilterNode | null = null;
  private bodyFilter: BiquadFilterNode | null = null;
  private bodyGainNode: GainNode | null = null;

  // Cascade formant resonators (F1 -> F2 -> F3 -> F4)
  private cascadeFilters: BiquadFilterNode[] = [];
  private cascadeOutGain: GainNode | null = null;

  // Parallel formant resonators (F1..F4 + Nasal murmur)
  private parallelFilters: BiquadFilterNode[] = [];
  private parallelGains: GainNode[] = [];
  private nasalFilter: BiquadFilterNode | null = null;
  private nasalGainNode: GainNode | null = null;
  private oralNotchFilter: BiquadFilterNode | null = null;
  private radiationFilter: BiquadFilterNode | null = null;
  private hpFilter: BiquadFilterNode | null = null;

  // Aspiration + Sibilant frication noise generator
  private noiseSource: AudioBufferSourceNode | null = null;
  private aspirationGain: GainNode | null = null;
  private fricFilter: BiquadFilterNode | null = null;
  private fricGainNode: GainNode | null = null;

  private readonly hVocalScratch = new Float32Array(N_FREQ_GRID);
  private sectionsScratch: AreaFunction | undefined = undefined;
  private smoothedF0 = 165.0;
  private smoothedRms = 0.0;
  private active = false;

  /**
   * Lazily build and connect the vocal tract synthesizer graph to `destination`.
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

    // Build custom Rosenberg / LF glottal flow derivative PeriodicWave (64 harmonics)
    const nHarm = 64;
    const real = new Float32Array(nHarm);
    const imag = new Float32Array(nHarm);
    for (let k = 1; k < nHarm; k++) {
      // Bright harmonic glottal flow derivative spectrum balanced by vocal-tract tilt + radiation
      const amp = 1.0 / Math.pow(k, 0.72);
      // Alternating phase curvature avoids harsh sawtooth buzz crest factor
      const phase = (k * 0.38) + ((k & 1) ? 0.15 : -0.15);
      real[k] = amp * Math.cos(phase);
      imag[k] = amp * Math.sin(phase);
    }
    const glottalWave = ctx.createPeriodicWave(real, imag, { disableNormalization: false });

    const glottalOsc = ctx.createOscillator();
    glottalOsc.setPeriodicWave(glottalWave);
    glottalOsc.frequency.value = this.smoothedF0;
    this.glottalOsc = glottalOsc;

    // Gentle chest fundamental reinforcement oscillator (dynamically attenuated on bright vowels)
    const subOsc = ctx.createOscillator();
    subOsc.type = "sine";
    subOsc.frequency.value = this.smoothedF0;
    const subGain = ctx.createGain();
    subGain.gain.value = 0.10;
    subOsc.connect(subGain);
    this.subOsc = subOsc;
    this.subGainNode = subGain;

    // Vibrato LFO (5.6 Hz natural singing vibrato)
    const vibratoOsc = ctx.createOscillator();
    vibratoOsc.type = "sine";
    vibratoOsc.frequency.value = 5.6;
    const vibratoGain = ctx.createGain();
    vibratoGain.gain.value = 0.0;
    vibratoOsc.connect(vibratoGain);
    vibratoGain.connect(glottalOsc.frequency);
    vibratoGain.connect(subOsc.frequency);
    this.vibratoOsc = vibratoOsc;
    this.vibratoGain = vibratoGain;

    // Glottal open-quotient spectral tilt filter (brighter when belting/open, softer when closed)
    const tiltFilter = ctx.createBiquadFilter();
    tiltFilter.type = "lowpass";
    tiltFilter.frequency.value = 2800;
    tiltFilter.Q.value = 0.7;
    this.tiltFilter = tiltFilter;

    const voicedGain = ctx.createGain();
    voicedGain.gain.value = 0.0;
    this.voicedGain = voicedGain;

    glottalOsc.connect(tiltFilter);
    subGain.connect(tiltFilter);
    tiltFilter.connect(voicedGain);

    // Looping 2-second white noise buffer for breath aspiration & consonant frication
    const noiseLen = Math.floor(ctx.sampleRate * 2.0);
    const noiseBuf = ctx.createBuffer(1, noiseLen, ctx.sampleRate);
    const noiseData = noiseBuf.getChannelData(0);
    let b0 = 0;
    let b1 = 0;
    for (let i = 0; i < noiseLen; i++) {
      const w = Math.random() * 2 - 1;
      // Gently pink-weighted turbulence source
      b0 = 0.96 * b0 + 0.04 * w;
      b1 = 0.82 * b1 + 0.18 * w;
      noiseData[i] = 0.65 * w + 0.35 * (b0 + b1);
    }
    const noiseSource = ctx.createBufferSource();
    noiseSource.buffer = noiseBuf;
    noiseSource.loop = true;
    this.noiseSource = noiseSource;

    const aspirationGain = ctx.createGain();
    aspirationGain.gain.value = 0.0;
    noiseSource.connect(aspirationGain);
    this.aspirationGain = aspirationGain;

    // Sum voiced glottal pulse + glottal aspiration into the vocal tract entry
    const tractEntry = ctx.createGain();
    tractEntry.gain.value = 1.0;
    voicedGain.connect(tractEntry);
    aspirationGain.connect(tractEntry);

    // 1. Cascade 4-formant resonator chain (F1 -> F2 -> F3 -> F4)
    const initHz = [520, 1480, 2650, 3650];
    const initQ = [5.5, 7.5, 9.0, 10.5];
    let prevCascade: AudioNode = tractEntry;
    this.cascadeFilters = [];
    for (let k = 0; k < 4; k++) {
      const bp = ctx.createBiquadFilter();
      bp.type = "peaking";
      bp.frequency.value = initHz[k];
      bp.Q.value = initQ[k] * 0.65;
      bp.gain.value = 11.5 - k * 1.5;
      prevCascade.connect(bp);
      prevCascade = bp;
      this.cascadeFilters.push(bp);
    }
    const cascadeOutGain = ctx.createGain();
    cascadeOutGain.gain.value = 0.55;
    prevCascade.connect(cascadeOutGain);
    this.cascadeOutGain = cascadeOutGain;

    // 2. Parallel 4-formant bandpass bank (weighted by 44-section Webster transfer amplitudes)
    const parallelSum = ctx.createGain();
    parallelSum.gain.value = 0.85;
    this.parallelFilters = [];
    this.parallelGains = [];
    for (let k = 0; k < 4; k++) {
      const bp = ctx.createBiquadFilter();
      bp.type = "bandpass";
      bp.frequency.value = initHz[k];
      bp.Q.value = initQ[k];
      const g = ctx.createGain();
      g.gain.value = 0.5 / (k + 1);
      tractEntry.connect(bp);
      bp.connect(g);
      g.connect(parallelSum);
      this.parallelFilters.push(bp);
      this.parallelGains.push(g);
    }

    // Chest resonance warmth filter (280 Hz fundamental body, trimmed on bright belted vocals)
    const bodyFilter = ctx.createBiquadFilter();
    bodyFilter.type = "lowpass";
    bodyFilter.frequency.value = 280;
    bodyFilter.Q.value = 0.9;
    const bodyGain = ctx.createGain();
    bodyGain.gain.value = 0.06;
    tractEntry.connect(bodyFilter);
    bodyFilter.connect(bodyGain);
    this.bodyFilter = bodyFilter;
    this.bodyGainNode = bodyGain;

    // Nasal velum parallel resonator (245 Hz nasal murmur)
    const nasalFilter = ctx.createBiquadFilter();
    nasalFilter.type = "bandpass";
    nasalFilter.frequency.value = 245;
    nasalFilter.Q.value = 3.2;
    const nasalGainNode = ctx.createGain();
    nasalGainNode.gain.value = 0.0;
    tractEntry.connect(nasalFilter);
    nasalFilter.connect(nasalGainNode);
    this.nasalFilter = nasalFilter;
    this.nasalGainNode = nasalGainNode;

    // Oral anti-resonance notch when velum is open (~960 Hz)
    const oralNotchFilter = ctx.createBiquadFilter();
    oralNotchFilter.type = "peaking";
    oralNotchFilter.frequency.value = 960;
    oralNotchFilter.Q.value = 2.2;
    oralNotchFilter.gain.value = 0.0;
    this.oralNotchFilter = oralNotchFilter;

    // Lip radiation high-shelf impedance projection (boosts 650+ Hz proportionally to mouth opening & brightness)
    const radiationFilter = ctx.createBiquadFilter();
    radiationFilter.type = "highshelf";
    radiationFilter.frequency.value = 650;
    radiationFilter.gain.value = 4.0;
    this.radiationFilter = radiationFilter;

    // Sub-formant high-pass shaping (keeps warm fundamental on dark vowels, clears sub-F1 boom on bright vocals)
    const hpFilter = ctx.createBiquadFilter();
    hpFilter.type = "highpass";
    hpFilter.frequency.value = 180;
    hpFilter.Q.value = 0.7;
    this.hpFilter = hpFilter;

    cascadeOutGain.connect(oralNotchFilter);
    parallelSum.connect(oralNotchFilter);
    bodyGain.connect(oralNotchFilter);
    oralNotchFilter.connect(radiationFilter);
    nasalGainNode.connect(radiationFilter);
    radiationFilter.connect(hpFilter);
    hpFilter.connect(out);

    // Front-cavity constriction sibilance / frication path (/s/, /sh/, /f/)
    const fricFilter = ctx.createBiquadFilter();
    fricFilter.type = "bandpass";
    fricFilter.frequency.value = 4800;
    fricFilter.Q.value = 2.4;
    const fricGainNode = ctx.createGain();
    fricGainNode.gain.value = 0.0;
    noiseSource.connect(fricFilter);
    fricFilter.connect(fricGainNode);
    fricGainNode.connect(out);
    this.fricFilter = fricFilter;
    this.fricGainNode = fricGainNode;

    const now = ctx.currentTime;
    glottalOsc.start(now);
    subOsc.start(now);
    vibratoOsc.start(now);
    noiseSource.start(now);
  }

  public setEnabled(enabled: boolean): void {
    this.active = enabled;
    if (!enabled && this.ctx && this.outputNode) {
      this.outputNode.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.015);
    }
  }

  public isEnabled(): boolean {
    return this.active;
  }

  /**
   * Update the vocal tract synthesizer parameters at 60 FPS from the Singing Head's
   * live reconstructed vocal tract geometry and glottal state.
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
    const tau = 0.016; // 16 ms smoothing time constant for zipper-free audio updates

    if (isMuted || art.rms < 0.0025) {
      this.smoothedRms *= 0.78;
      this.outputNode.gain.setTargetAtTime(0.0, t, 0.018);
      return profile;
    }

    // Track fundamental pitch f0Hz 1:1 from song vocals on voiced frames (snap quickly on note onset)
    const larynxShift = Math.pow(2, (tractArt.larynx * 2.2) / 12.0);
    if (art.f0Hz > 65 && art.voicing > 0.20) {
      const targetF0 = clamp(art.f0Hz, 70, 820);
      const snap = this.smoothedRms < 0.008 ? 1.0 : 0.82;
      this.smoothedF0 += (targetF0 - this.smoothedF0) * snap;
    } else {
      const fallbackF0 = clamp((130 + art.pitchNorm * 190) * larynxShift, 85, 520);
      this.smoothedF0 += (fallbackF0 - this.smoothedF0) * 0.20;
    }

    if (this.glottalOsc && this.subOsc) {
      this.glottalOsc.frequency.setTargetAtTime(this.smoothedF0, t, 0.010);
      this.subOsc.frequency.setTargetAtTime(this.smoothedF0, t, 0.010);
    }

    // Vibrato depth in Hz (up to ~0.45 semitones when voiced and sustaining)
    if (this.vibratoGain && this.vibratoOsc) {
      const vibGate = clamp(art.voicing * Math.max(0, (art.rms - 0.008) * 28), 0, 1);
      const vibHz = this.smoothedF0 * 0.024 * clamp(art.vibrato, 0, 1) * vibGate;
      this.vibratoGain.gain.setTargetAtTime(vibHz, t, 0.025);
    }

    const centroidDrive = clamp(((art.centroidHz || 1400) - 500) / 3000, 0, 1);

    // Dynamic chest/sub balance: rich fundamental for pitch lock on warm vocals, trimmed on bright singing
    if (this.subGainNode) {
      const subLvl = 0.14 * clamp(1.0 - 0.85 * centroidDrive, 0.05, 1.0);
      this.subGainNode.gain.setTargetAtTime(subLvl, t, tau);
    }
    if (this.bodyGainNode) {
      const bodyLvl = 0.06 * clamp(1.0 - centroidDrive, 0.0, 1.0);
      this.bodyGainNode.gain.setTargetAtTime(bodyLvl, t, tau);
    }

    // Glottal spectral tilt (tracked cleanly from sung spectral centroid + mouth opening + tension)
    if (this.tiltFilter) {
      const cutoffHz = clamp(
        1150 + 2100 * profile.openDrive + 3600 * centroidDrive + 1100 * clamp(art.tension, 0, 1),
        850,
        9200
      );
      this.tiltFilter.frequency.setTargetAtTime(cutoffHz, t, tau);
    }

    if (this.hpFilter) {
      const hpHz = clamp(130 + 400 * centroidDrive, 120, 560);
      this.hpFilter.frequency.setTargetAtTime(hpHz, t, tau);
    }

    // Voiced vs. aspirated glottal excitation
    const voicing = clamp(art.voicing, 0.04, 1.0);
    const breath = clamp(art.breath, 0.0, 1.0);
    if (this.voicedGain) {
      const vLvl = clamp(0.10 + 0.90 * voicing, 0.08, 1.05);
      this.voicedGain.gain.setTargetAtTime(vLvl, t, tau);
    }
    if (this.aspirationGain) {
      const aspLvl = clamp((1.0 - voicing) * 0.34 + breath * 0.22, 0.01, 0.45);
      this.aspirationGain.gain.setTargetAtTime(aspLvl, t, tau);
    }

    // Update 4 cascade & parallel formant filters from the 44-section Webster solution
    for (let k = 0; k < 4; k++) {
      const fHz = profile.formantHz[k];
      const qVal = profile.formantQ[k];
      const gVal = profile.formantGain[k];

      const cas = this.cascadeFilters[k];
      if (cas) {
        cas.frequency.setTargetAtTime(fHz, t, tau);
        cas.Q.setTargetAtTime(clamp(qVal * 0.65, 2.5, 11.0), t, tau);
        cas.gain.setTargetAtTime(clamp(6.0 + gVal * 9.5 - k * 1.0, 2.0, 18.0), t, tau);
      }

      const par = this.parallelFilters[k];
      const parG = this.parallelGains[k];
      if (par && parG) {
        par.frequency.setTargetAtTime(fHz, t, tau);
        par.Q.setTargetAtTime(qVal, t, tau);
        const sign = (k & 1) === 1 ? -1 : 1; // Alternating signs reduce inter-formant phase cancellation
        parG.gain.setTargetAtTime(sign * gVal * (0.65 / (1 + k * 0.35)), t, tau);
      }
    }

    // Nasal coupling & oral anti-resonance
    if (this.nasalGainNode && this.oralNotchFilter) {
      this.nasalGainNode.gain.setTargetAtTime(profile.nasalGain * 0.35, t, tau);
      this.oralNotchFilter.gain.setTargetAtTime(-8.5 * profile.nasalGain, t, tau);
    }

    // Lip radiation high-shelf impedance boost (+6 dB/oct oral projection)
    if (this.radiationFilter) {
      const radDb = clamp(-3.0 + 7.5 * profile.openDrive + 12.0 * centroidDrive, -4.0, 15.5);
      this.radiationFilter.gain.setTargetAtTime(radDb, t, tau);
    }

    // Sibilance / frication turbulence (/s/, /sh/, /f/)
    if (this.fricFilter && this.fricGainNode) {
      this.fricFilter.frequency.setTargetAtTime(profile.sibCenterHz, t, tau);
      const fricAmt = clamp(
        profile.fricGain * (0.22 + 0.65 * breath + 0.45 * (1.0 - voicing)),
        0.0,
        0.65
      );
      this.fricGainNode.gain.setTargetAtTime(fricAmt, t, tau);
    }

    // Overall vocal output gain matched 1:1 to the song's vocal track RMS + mouth opening radiation
    this.smoothedRms += (art.rms - this.smoothedRms) * 0.65;
    const radScale = 0.65 + 0.55 * Math.sqrt(profile.openDrive);
    const outGain = clamp(this.smoothedRms * 3.3 * radScale, 0.0, 1.8);
    this.outputNode.gain.setTargetAtTime(outGain, t, 0.012);

    return profile;
  }

  public dispose(): void {
    try {
      this.glottalOsc?.stop();
      this.subOsc?.stop();
      this.vibratoOsc?.stop();
      this.noiseSource?.stop();
    } catch {
      // Ignore if already stopped
    }
    try {
      this.outputNode?.disconnect();
    } catch {
      // Ignore
    }
    this.glottalOsc = null;
    this.subOsc = null;
    this.subGainNode = null;
    this.vibratoOsc = null;
    this.vibratoGain = null;
    this.voicedGain = null;
    this.tiltFilter = null;
    this.bodyFilter = null;
    this.bodyGainNode = null;
    this.cascadeFilters = [];
    this.cascadeOutGain = null;
    this.parallelFilters = [];
    this.parallelGains = [];
    this.nasalFilter = null;
    this.nasalGainNode = null;
    this.oralNotchFilter = null;
    this.radiationFilter = null;
    this.hpFilter = null;
    this.noiseSource = null;
    this.aspirationGain = null;
    this.fricFilter = null;
    this.fricGainNode = null;
    this.outputNode = null;
    this.ctx = null;
  }
}
