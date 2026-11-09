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
