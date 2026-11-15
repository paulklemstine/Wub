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

/**
 * Assemble a stereo run from sub-phrase slices (1-bar, 2-beat, 1-beat, or
 * 1/16th stutter cells) across one or more decoded 4-bar donor phrases.
 * Applies zero-crossing snapping, per-slice gain envelopes (including
 * pre-drop silence gaps), and equal-power micro-crossfades at joins.
 */
export function assembleSlicedStereoRun(
  phrases: readonly [Float32Array, Float32Array][],
  pre: readonly [Float32Array, Float32Array] | null,
  slices: readonly SliceSpec[],
  sampleRate: number,
  crossfadeMs: number = 8,
): ConcatRunResult {
  if (phrases.length === 0 || slices.length === 0) {
    return concatStereoRun(phrases, pre, sampleRate);
  }
  const refLen = phrases[0][0].length;
  const refBarFrames = Math.max(1, refLen / 4);
  const xfadeFrames = Math.max(0, Math.min(2048, Math.round((crossfadeMs / 1000) * sampleRate)));
  const snapFrames = Math.max(0, Math.round(0.003 * sampleRate));

  const segments: Array<{ l: Float32Array; r: Float32Array; tailL: Float32Array; tailR: Float32Array; targetFrames: number }> = [];
  for (const spec of slices) {
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

    const segL = new Float32Array(targetFrames);
    const segR = new Float32Array(targetFrames);
    const copyLen = Math.min(cutLen, targetFrames);
    const fadeOutTail = cutLen > targetFrames ? Math.min(xfadeFrames, Math.floor(copyLen / 4)) : 0;

    for (let i = 0; i < copyLen; i++) {
      const t = copyLen > 1 ? i / (copyLen - 1) : 0;
      let env = g0 + (g1 - g0) * t;
      if (fadeOutTail > 0 && i >= copyLen - fadeOutTail) {
        const ft = (copyLen - 1 - i) / Math.max(1, fadeOutTail);
        env *= ft;
      }
      segL[i] = srcL[startIdx + i] * env;
      segR[i] = srcR[startIdx + i] * env;
    }

    // Grab up to xfadeFrames past copyLen (or damp the tail) for overlap-add crossfading
    // without shortening the cumulative grid duration.
    const extraAvail = Math.min(xfadeFrames, Math.max(0, srcLen - (startIdx + copyLen)));
    const tailLen = copyLen >= targetFrames - xfadeFrames ? Math.min(xfadeFrames, extraAvail > 0 ? extraAvail : 64) : 0;
    const tailL = new Float32Array(tailLen);
    const tailR = new Float32Array(tailLen);
    for (let k = 0; k < tailLen; k++) {
      if (k < extraAvail) {
        tailL[k] = srcL[startIdx + copyLen + k] * g1;
        tailR[k] = srcR[startIdx + copyLen + k] * g1;
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
