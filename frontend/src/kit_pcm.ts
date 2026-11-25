// Pure PCM helpers shared by the kit audio worker and the main-thread
// fallback (stretch_voice): stereo->mono mix, plus pre-roll + run concat.
// No DOM, no engine imports, so both entry points can bundle it safely.

/** Average the channels of a decoded buffer into mono PCM. */
export function mixToMono(buffer: AudioBuffer): Float32Array {
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

/** Split a decoded buffer into two channel arrays (left, right).
 *  Mono buffers duplicate their single channel; stereo passes through. */
export function splitChannels(buffer: AudioBuffer): [Float32Array, Float32Array] {
  const left = buffer.getChannelData(0);
  const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
  return [left.slice(), right.slice()];
}

export interface ConcatRunResult {
  /** Pre-roll + concatenated mains per channel, ready for StretchVoice.addBuffers. */
  samples: [Float32Array, Float32Array];
  /** Seconds of pre-roll at the front of `samples`. */
  preDur: number;
  /** Decoded seconds per main phrase (input timeline, mono). */
  mainDurations: number[];
  sampleRate: number;
}

export interface SliceSpec {
  /** Index into the loaded items array for this run. */
  itemIndex: number;
  /** Normalized start fraction [0..1) within the 4-bar phrase. */
  startFrac: number;
  /** Normalized end fraction (0..1] within the 4-bar phrase. */
  endFrac: number;
  /** Target duration in bars (defaults to (endFrac - startFrac) * 4). */
  targetBars?: number;
  /** Linear gain multiplier at the start of this slice (0 = silence gap). */
  gain?: number;
  /** Linear gain multiplier at the end of this slice (defaults to `gain`). */
  endGain?: number;
}

/**
 * Find a nearby zero crossing (or minimum-energy sample) within `windowFrames`
 * around `targetFrame` to avoid click transients at sub-bar slice boundaries.
 */
export function snapZeroCrossing(
  channel: Float32Array,
  targetFrame: number,
  windowFrames: number = 144,
): number {
  const n = channel.length;
  if (n <= 2) return 0;
  const clamped = Math.max(0, Math.min(n - 1, Math.round(targetFrame)));
  if (clamped <= 2 || clamped >= n - 2 || windowFrames <= 0) return clamped;

  const lo = Math.max(1, clamped - windowFrames);
  const hi = Math.min(n - 2, clamped + windowFrames);
  let bestIdx = clamped;
  let bestAbs = Math.abs(channel[clamped]);

  // Search outward from `clamped` for the nearest true sign flip.
  for (let d = 0; d <= windowFrames; d++) {
    const r = clamped + d;
    if (r <= hi) {
      const a = channel[r - 1];
      const b = channel[r];
      if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) {
        return Math.abs(b) <= Math.abs(a) ? r : r - 1;
      }
      if (Math.abs(b) < bestAbs) {
        bestAbs = Math.abs(b);
        bestIdx = r;
      }
    }
    const l = clamped - d;
    if (l >= lo) {
      const a = channel[l - 1];
      const b = channel[l];
      if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) {
        return Math.abs(b) <= Math.abs(a) ? l : l - 1;
      }
      if (Math.abs(b) < bestAbs) {
        bestAbs = Math.abs(b);
        bestIdx = l;
      }
    }
  }
  return bestIdx;
}

export interface CoalescedSliceSpec extends SliceSpec {
  barSpan: number;
}

/** True when `slices` represents a single uncut 4-bar phrase from `itemIndex === 0` at unit gain. */
export function isContiguousFullPhrase(slices: readonly SliceSpec[] | undefined): boolean {
  if (!slices || slices.length === 0) return false;
  let expectedStart = 0;
  for (const s of slices) {
    if (s.itemIndex !== 0) return false;
    const g0 = s.gain ?? 1;
    const g1 = s.endGain ?? g0;
    if (Math.abs(g0 - 1) > 0.02 || Math.abs(g1 - 1) > 0.02) return false;
    if (Math.abs(s.startFrac - expectedStart) > 1e-4) return false;
    const rawBars = (s.endFrac - s.startFrac) * 4;
    if (s.targetBars !== undefined && Math.abs(s.targetBars - rawBars) > 1e-3) return false;
    expectedStart = s.endFrac;
  }
  return Math.abs(expectedStart - 1) < 1e-4;
}

/** Merge adjacent contiguous bars from the same phrase and flat gain into one uncut waveform span. */
export function coalesceContiguousSlices(slices: readonly SliceSpec[]): CoalescedSliceSpec[] {
  const out: CoalescedSliceSpec[] = [];
  for (const raw of slices) {
    const rawBars = (raw.endFrac - raw.startFrac) * 4;
    const tgtBars = raw.targetBars !== undefined && raw.targetBars > 0 ? raw.targetBars : rawBars;
    const item: CoalescedSliceSpec = { ...raw, targetBars: tgtBars, barSpan: tgtBars };
    const prev = out[out.length - 1];
    if (prev) {
      const pg0 = prev.gain ?? 1;
      const pg1 = prev.endGain ?? pg0;
      const cg0 = item.gain ?? 1;
      const cg1 = item.endGain ?? cg0;
      const prevRawBars = (prev.endFrac - prev.startFrac) * 4;
      const sameSource = prev.itemIndex === item.itemIndex;
      const contiguous = Math.abs(item.startFrac - prev.endFrac) < 1e-4;
      const flatGain =
        Math.abs(pg0 - pg1) < 0.05 &&
        Math.abs(cg0 - cg1) < 0.05 &&
        Math.abs(pg1 - cg0) < 0.05;
      const unstretched =
        Math.abs((prev.targetBars ?? prevRawBars) - prevRawBars) < 1e-3 &&
        Math.abs(tgtBars - rawBars) < 1e-3;
      if (sameSource && contiguous && flatGain && unstretched) {
        prev.endFrac = item.endFrac;
        prev.targetBars = (prev.targetBars ?? prevRawBars) + tgtBars;
        prev.barSpan += tgtBars;
        prev.endGain = item.endGain ?? cg1;
        continue;
      }
    }
    out.push(item);
  }
  return out;
}

/** Resample `src[s0 .. s0 + srcLen)` to exact `targetLen` frames using 4-point Hermite cubic interpolation. */
export function resampleChannelHermite(
  src: Float32Array,
  s0: number,
  srcLen: number,
  targetLen: number,
): Float32Array {
  if (srcLen === targetLen && s0 >= 0 && s0 + targetLen <= src.length) {
    return src.subarray(s0, s0 + targetLen);
  }
  const out = new Float32Array(targetLen);
  if (targetLen <= 0 || src.length === 0) return out;
  if (targetLen === 1) {
    const idx = Math.max(0, Math.min(src.length - 1, s0));
    out[0] = src[idx];
    return out;
  }
  const step = Math.max(1, srcLen - 1) / (targetLen - 1);
  const maxIdx = src.length - 1;
  for (let i = 0; i < targetLen; i++) {
    const pos = s0 + i * step;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const xm1 = src[Math.max(0, Math.min(maxIdx, idx - 1))];
    const x0 = src[Math.max(0, Math.min(maxIdx, idx))];
    const x1 = src[Math.max(0, Math.min(maxIdx, idx + 1))];
    const x2 = src[Math.max(0, Math.min(maxIdx, idx + 2))];
    const c0 = x0;
    const c1 = 0.5 * (x1 - xm1);
    const c2 = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2;
    const c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
    out[i] = ((c3 * frac + c2) * frac + c1) * frac + c0;
  }
  return out;
}

/** Match secondary donor RMS loudness to `primary`, clamped to ±4.5 dB (`[0.595, 1.679]`). */
export function matchDonorRmsGain(
  primary: readonly [Float32Array, Float32Array],
  donorSliceL: Float32Array,
  donorSliceR: Float32Array,
): number {
  const [pL, pR] = primary;
  const pLen = Math.min(pL.length, pR.length);
  const dLen = Math.min(donorSliceL.length, donorSliceR.length);
  if (pLen <= 0 || dLen <= 0) return 1.0;

  let pSumSq = 0;
  const pStride = Math.max(1, Math.floor(pLen / 8192));
  let pCount = 0;
  for (let i = 0; i < pLen; i += pStride) {
    const l = pL[i];
    const r = pR[i];
    pSumSq += 0.5 * (l * l + r * r);
    pCount++;
  }
  let dSumSq = 0;
  const dStride = Math.max(1, Math.floor(dLen / 4096));
  let dCount = 0;
  for (let i = 0; i < dLen; i += dStride) {
    const l = donorSliceL[i];
    const r = donorSliceR[i];
    dSumSq += 0.5 * (l * l + r * r);
    dCount++;
  }
  const pRms = Math.sqrt(pSumSq / Math.max(1, pCount));
  const dRms = Math.sqrt(dSumSq / Math.max(1, dCount));
  if (pRms < 5e-4 || dRms < 5e-4) return 1.0;
  const ratio = pRms / dRms;
  return Math.max(0.595, Math.min(1.679, ratio));
}

/**
 * Assemble a stereo run from sub-phrase slices (1-bar, 2-beat, 1-beat, or
 * 1/16th stutter cells) across one or more decoded 4-bar donor phrases.
 * Coalesces contiguous same-phrase bars with zero internal cuts, cubic-resamples
 * tempo-mismatched donor bars to exact target frames, RMS-matches secondary
 * donors within ±4.5 dB, and applies equal-power overlap crossfades at joins.
 */
export function assembleSlicedStereoRun(
  phrases: readonly [Float32Array, Float32Array][],
  pre: readonly [Float32Array, Float32Array] | null,
  slices: readonly SliceSpec[],
  sampleRate: number,
  crossfadeMs: number = 28,
): ConcatRunResult {
  if (phrases.length === 0 || slices.length === 0) {
    return concatStereoRun(phrases, pre, sampleRate);
  }
  if (phrases.length === 1 && isContiguousFullPhrase(slices)) {
    return concatStereoRun(phrases, pre, sampleRate);
  }

  const mergedSlices = coalesceContiguousSlices(slices);
  const refLen = phrases[0][0].length;
  const refBarFrames = Math.max(1, refLen / 4);
  const xfadeFrames = Math.max(0, Math.min(4096, Math.round((crossfadeMs / 1000) * sampleRate)));
  const snapFrames = Math.max(0, Math.round(0.003 * sampleRate));

  const segments: Array<{ l: Float32Array; r: Float32Array; tailL: Float32Array; tailR: Float32Array; targetFrames: number }> = [];
  for (const spec of mergedSlices) {
    const srcIdx = Math.max(0, Math.min(phrases.length - 1, Math.floor(spec.itemIndex)));
    const [srcL, srcR] = phrases[srcIdx];
    const srcLen = srcL.length;
    const sFrac = Math.max(0, Math.min(0.999, spec.startFrac));
    const eFrac = Math.max(sFrac + 0.001, Math.min(1, spec.endFrac));
    const rawBars = (eFrac - sFrac) * 4;
    const targetBars = spec.targetBars !== undefined && spec.targetBars > 0 ? spec.targetBars : rawBars;
    const targetFrames = Math.max(1, Math.round(targetBars * refBarFrames));

    const g0 = spec.gain ?? 1.0;
    const g1 = spec.endGain ?? g0;
    if (g0 <= 1e-5 && g1 <= 1e-5) {
      segments.push({
        l: new Float32Array(targetFrames),
        r: new Float32Array(targetFrames),
        tailL: new Float32Array(0),
        tailR: new Float32Array(0),
        targetFrames,
      });
      continue;
    }

    const startRaw = Math.round(sFrac * srcLen);
    const endRaw = Math.round(eFrac * srcLen);
    const startIdx = sFrac <= 1e-4 ? 0 : snapZeroCrossing(srcL, startRaw, snapFrames);
    const endIdx = eFrac >= 0.9999
      ? srcLen
      : Math.max(startIdx + 1, snapZeroCrossing(srcL, endRaw, snapFrames));
    const cutLen = Math.max(1, endIdx - startIdx);

    const resL = resampleChannelHermite(srcL, startIdx, cutLen, targetFrames);
    const resR = resampleChannelHermite(srcR, startIdx, cutLen, targetFrames);
    const donorGain = srcIdx > 0 ? matchDonorRmsGain(phrases[0], resL, resR) : 1.0;

    const segL = new Float32Array(targetFrames);
    const segR = new Float32Array(targetFrames);
    for (let i = 0; i < targetFrames; i++) {
      const t = targetFrames > 1 ? i / (targetFrames - 1) : 0;
      const env = (g0 + (g1 - g0) * t) * donorGain;
      segL[i] = resL[i] * env;
      segR[i] = resR[i] * env;
    }

    // Grab up to xfadeFrames past endIdx for equal-power overlap-add crossfading
    // without altering the cumulative bar-grid duration.
    const extraAvail = Math.min(xfadeFrames, Math.max(0, srcLen - endIdx));
    const tailLen = targetFrames > xfadeFrames ? Math.min(xfadeFrames, extraAvail > 0 ? extraAvail : Math.min(xfadeFrames, 128)) : 0;
    const tailL = new Float32Array(tailLen);
    const tailR = new Float32Array(tailLen);
    for (let k = 0; k < tailLen; k++) {
      if (k < extraAvail) {
        tailL[k] = srcL[endIdx + k] * g1 * donorGain;
        tailR[k] = srcR[endIdx + k] * g1 * donorGain;
      } else if (endIdx - tailLen + k >= startIdx) {
        const decay = 1 - k / Math.max(1, tailLen);
        const tailIdx = Math.max(0, Math.min(targetFrames - 1, targetFrames - tailLen + k));
        tailL[k] = segL[tailIdx] * decay * 0.45;
        tailR[k] = segR[tailIdx] * decay * 0.45;
      }
    }
    segments.push({ l: segL, r: segR, tailL, tailR, targetFrames });
  }

  // Place each segment at its exact grid position with equal-power overlap tails so
  // every join is click-free and cumulative duration equals the exact sum of targetFrames.
  const preLen = pre ? pre[0].length : 0;
  const totalMainFrames = segments.reduce((sum, s) => sum + s.targetFrames, 0);
  const outL = new Float32Array(preLen + totalMainFrames);
  const outR = new Float32Array(preLen + totalMainFrames);
  if (pre) {
    outL.set(pre[0], 0);
    outR.set(pre[1], 0);
  }

  let cursor = preLen;
  for (let idx = 0; idx < segments.length; idx++) {
    const seg = segments[idx];
    const nextSeg = idx + 1 < segments.length ? segments[idx + 1] : null;
    const prevSeg = idx > 0 ? segments[idx - 1] : null;
    const ovIn = prevSeg
      ? Math.min(xfadeFrames, prevSeg.tailL.length, Math.floor(seg.targetFrames / 3))
      : 0;
    const ovOut = nextSeg
      ? Math.min(xfadeFrames, seg.tailL.length, Math.floor(nextSeg.targetFrames / 3))
      : 0;

    // Head of this segment (fade in over ovIn if previous segment extended its tail here)
    for (let k = 0; k < seg.targetFrames; k++) {
      const wIn = k < ovIn && ovIn > 1 ? Math.sin((0.5 * Math.PI * k) / (ovIn - 1)) : 1;
      outL[cursor + k] += seg.l[k] * wIn;
      outR[cursor + k] += seg.r[k] * wIn;
    }

    // Write outgoing crossfade tail onto the beginning of the next slot
    if (ovOut > 1 && cursor + seg.targetFrames + ovOut <= outL.length) {
      for (let k = 0; k < ovOut; k++) {
        const wOut = Math.cos((0.5 * Math.PI * k) / (ovOut - 1));
        outL[cursor + seg.targetFrames + k] += seg.tailL[k] * wOut;
        outR[cursor + seg.targetFrames + k] += seg.tailR[k] * wOut;
      }
    }

    cursor += seg.targetFrames;
  }

  return {
    samples: [outL, outR],
    preDur: preLen / sampleRate,
    mainDurations: [totalMainFrames / sampleRate],
    sampleRate,
  };
}

/**
 * Concatenate the optional pre-bar tail with the main phrases of one run.
 * The mains are consecutive 4-bar windows of one source, so the joins are
 * real bar lines and need no crossfade.  Channels are concatenated in
 * lockstep so stereo content survives end to end.
 */
export function concatStereoRun(
  mains: readonly [Float32Array, Float32Array][],
  pre: readonly [Float32Array, Float32Array] | null,
  sampleRate: number,
): ConcatRunResult {
  const preLen = pre ? pre[0].length : 0;
  const channelTotal = preLen + mains.reduce((sum, pair) => sum + pair[0].length, 0);
  const channels: [Float32Array, Float32Array] = [
    new Float32Array(channelTotal),
    new Float32Array(channelTotal),
  ];
  let offset = 0;
  if (pre) {
    channels[0].set(pre[0], 0);
    channels[1].set(pre[1], 0);
    offset = preLen;
  }
  const mainDurations: number[] = [];
  for (const pair of mains) {
    channels[0].set(pair[0], offset);
    channels[1].set(pair[1], offset);
    mainDurations.push(pair[0].length / sampleRate);
    offset += pair[0].length;
  }
  return { samples: [channels[0], channels[1]], preDur: preLen / sampleRate, mainDurations, sampleRate };
}

/**
 * Concatenate the optional pre-bar tail with the main phrases of one run.
 * The mains are consecutive 4-bar windows of one source, so the joins are
 * real bar lines and need no crossfade.
 */
export function concatRun(
  mainMonos: readonly Float32Array[],
  preMono: Float32Array | null,
  sampleRate: number,
): ConcatRunResult {
  // Mono input is duplicated into both channels so the downstream path
  // always hands the voice a well-formed stereo run.
  const mains = mainMonos.map((m) => [m, m] as [Float32Array, Float32Array]);
  const pre = preMono ? ([preMono, preMono] as [Float32Array, Float32Array]) : null;
  return concatStereoRun(mains, pre, sampleRate);
}
