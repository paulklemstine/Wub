// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/**
 * The frame the anatomical backdrop is painted in: canonical centimetres
 * (glottis at the origin, before the speaker's size and the pharynx stretch)
 * mapped to the pixels of the image.
 */
export const BACKDROP_FRAME = { x0: -7, x1: 13, y0: -7, y1: 23, width: 1024, height: 1536 };

/** Pixel of a canonical point in the backdrop image. */
export function framePixel(x: number, y: number): [number, number] {
  const f = BACKDROP_FRAME, k = f.width / (f.x1 - f.x0);
  return [(x - f.x0) * k, (f.y1 - y) * k];
}
