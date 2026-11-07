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

export interface ConcatRunResult {
  /** Pre-roll + concatenated mains, ready for StretchVoice.addBuffers. */
  samples: Float32Array;
  /** Seconds of pre-roll at the front of `samples`. */
  preDur: number;
  /** Decoded seconds per main phrase (input timeline, mono). */
  mainDurations: number[];
  sampleRate: number;
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
  let total = 0;
  for (const main of mainMonos) total += main.length;
  const preLen = preMono?.length ?? 0;
  const samples = new Float32Array(preLen + total);
  let offset = 0;
  if (preMono) {
    samples.set(preMono, 0);
    offset = preLen;
  }
  const mainDurations: number[] = [];
  for (const main of mainMonos) {
    samples.set(main, offset);
    offset += main.length;
    mainDurations.push(main.length / sampleRate);
  }
  return { samples, preDur: preLen / sampleRate, mainDurations, sampleRate };
}
