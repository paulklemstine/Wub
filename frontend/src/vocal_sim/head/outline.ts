// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/**
 * The parts of the head that never touch the air, in canonical drawing
 * coordinates (cm, glottis at the origin, before the speaker's size and the
 * pharynx stretch). `canonToWorld` maps them like the airway, so they follow a
 * child's head, a longer pharynx or a lower larynx.
 */

/** Back of the head, from the neck to the crown. */
export const HEAD_BACK = [-4.6, -3.6, -4.3, 1.0, -4.7, 5.0, -5.5, 8.6, -6.1, 12.0, -5.8, 15.6, -4.2, 18.6, -1.0, 20.6, 3.0, 21.2];
/** Forehead, nose and upper lip, from the crown down. */
export const HEAD_FRONT_UP = [3.0, 21.2, 6.6, 20.3, 8.4, 18.9, 9.0, 17.2, 8.4, 15.8, 9.3, 14.2, 10.2, 12.8, 10.7, 12.0, 10.5, 11.35, 9.6, 11.05, 9.05, 10.9, 9.0, 10.3, 9.15, 9.5];
/** Chin and the front of the neck (they turn with the jaw by `chinWeight`). */
export const CHIN = [8.95, 7.6, 8.55, 7.0, 8.25, 6.55, 8.6, 5.7, 8.55, 4.7, 7.7, 3.8, 5.6, 3.25, 3.7, 2.7, 2.75, 0.6, 2.65, -3.6];
/** Centre line of the nasal cavity, from the nasopharynx to the nostril. */
export const NASAL_CENTRE = [-0.25, 10.75, 0.1, 11.6, 0.9, 12.3, 2.2, 12.75, 4.0, 12.95, 5.8, 12.85, 7.3, 12.5, 8.6, 11.95, 9.55, 11.25];
/** Half height of the nasal cavity at each point of the centre line. */
export const NASAL_HALF = [0.42, 0.5, 0.62, 0.8, 0.88, 0.85, 0.7, 0.45, 0.22];
/** How much a point of the chin turns with the jaw (0 at the neck, 1 at the chin). */
export const chinWeight = (x: number) => Math.max(0, Math.min(1, (x - 3.2) / 3.0));

/**
 * A smooth curve through the points of a polyline (interleaved x, y): a
 * centripetal Catmull-Rom spline, `steps` points per segment, through every
 * given point, without overshoot at sharp corners.
 */
export function smoothOutline(input: number[], steps = 8): number[] {
  // repeated points would make a segment of zero length
  const pts: number[] = [];
  for (let i = 0; i < input.length; i += 2) {
    const m = pts.length;
    if (m && Math.hypot(input[i] - pts[m - 2], input[i + 1] - pts[m - 1]) < 1e-6) continue;
    pts.push(input[i], input[i + 1]);
  }
  const n = pts.length / 2;
  if (n < 3) return pts;
  const P = (i: number): [number, number] => {
    const j = Math.max(0, Math.min(n - 1, i));
    return [pts[2 * j], pts[2 * j + 1]];
  };
  const out: number[] = [pts[0], pts[1]];
  for (let i = 0; i < n - 1; i++) {
    const p0 = P(i - 1), p1 = P(i), p2 = P(i + 1), p3 = P(i + 2);
    const d = (a: [number, number], b: [number, number]) => Math.max(1e-6, Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1])));
    const t1 = d(p0, p1), t2 = t1 + d(p1, p2), t3 = t2 + d(p2, p3);
    for (let k = 1; k <= steps; k++) {
      const t = t1 + ((t2 - t1) * k) / steps;
      const q: number[] = [];
      for (let c = 0; c < 2; c++) {
        const a1 = ((t1 - t) * p0[c] + t * p1[c]) / t1;
        const a2 = ((t2 - t) * p1[c] + (t - t1) * p2[c]) / (t2 - t1);
        const a3 = ((t3 - t) * p2[c] + (t - t2) * p3[c]) / (t3 - t2);
        const b1 = ((t2 - t) * a1 + t * a2) / t2;
        const b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1);
        q.push(((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1));
      }
      out.push(q[0], q[1]);
    }
  }
  return out;
}
