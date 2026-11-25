// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/**
 * Cross-sections of the vocal tract in three dimensions.
 *
 * The mid-sagittal model gives, on each grid line, the distance d between the
 * lower and the upper wall on the midline. Away from the midline, at the
 * lateral coordinate z, the two walls come together: the palate falls from the
 * top of its vault toward the teeth while the tongue rises to its margins,
 * braced against the upper molars (Stone 1990; Gick et al. 2017); in the
 * pharynx the tongue root and the posterior wall curve toward the lateral
 * walls. The height of the airway is then
 *
 *   h(z) = d - K |z|^p,
 *
 * a lens between two walls that meet at |z| = (d / K)^(1/p), closed at the
 * sides by the teeth or the lateral pharyngeal walls at |z| = W / 2. A lens
 * has the area A = 2p / (p + 1) d^(1 + 1/p) K^(-1/p): this is the power law
 * A = alpha d^beta that X-ray and cast studies fitted region by region (Heinz &
 * Stevens 1965; Sundberg et al. 1987), with beta = 1 + 1/p. Here the lens is
 * the geometry itself, so that each section also has
 *
 *   - its true perimeter, which sets the viscous and thermal losses and the
 *     hydraulic diameter of a jet (a flat slit has far more wall than a round
 *     duct of the same area);
 *   - the share of that wall which is soft tissue and yields to the sound
 *     pressure (tongue, velum, pharynx, cheeks, lips) and the share which is
 *     bone and does not (hard palate, teeth);
 *   - the central groove of the tongue blade, a channel about 5 mm wide with
 *     raised sides (Narayanan, Alwan & Haker 1995);
 *   - the lateral channels of a lateral approximant: the tongue tip closes on
 *     the midline while the margins are drawn in and lowered, and the air flows
 *     past both sides of the tongue, 0.1-0.5 cm^2 a side beside the contact,
 *     joining the central channel behind it in crescent-shaped sections
 *     (Narayanan, Alwan & Haker 1997). Where the two run apart, the central
 *     channel above the tongue ends at the contact: a blind side branch, the
 *     supralingual cavity, which puts an antiresonance into the 2-5 kHz
 *     range (Zhang & Espy-Wilson 2004).
 *
 * All lengths here are canonical (the drawing frame of an adult man divided by
 * the speaker's scale): areas scale with s^2 and perimeters with s.
 */

/** Slope of the raised sides of a grooved tongue blade (cm of rise per cm). */
export const GROOVE_WALL = 6;
/** Lateral channels of a lateral approximant (canonical cm): how far inside the
 *  teeth the drawn-in margin of the tongue starts to fall, how far it falls at
 *  the teeth, and the depth of seal it must clear before air passes. */
export const SIDE_WIDTH = 0.7;
export const SIDE_DROP = 1.1;
export const SIDE_SEAL = 0.1;

/** Shape of one cross-section. */
export interface Lens {
  /** approach of the walls, h(z) = d - K |z|^p (canonical cm) */
  K: number;
  p: number;
  /** half width between the lateral walls (canonical cm) */
  half: number;
  /** central groove of the tongue (0..1) and its half width (canonical cm) */
  groove: number;
  grooveHalf: number;
  /** margins of the tongue drawn in and lowered, 0 = braced against the teeth */
  sides: number;
  /** yielding share (0..1) of the upper wall, the lower wall and the lateral walls */
  softUp: number;
  softLow: number;
  softSide: number;
}

export interface Section {
  /** open area (canonical cm^2) */
  area: number;
  /** wetted perimeter (canonical cm) */
  perimeter: number;
  /** part of the perimeter that yields (canonical cm) */
  soft: number;
  /** area of the lateral channels beside the tongue (canonical cm^2) */
  lateral: number;
  /** lateral extent of the central channel (canonical cm) */
  width: number;
  /** the central channel when the lateral channels run apart from it, ending
   *  at the contact (the supralingual cavity): its area, perimeter and
   *  yielding perimeter (canonical units; 0 when the section is one lumen) */
  supra: number;
  supraPerimeter: number;
  supraSoft: number;
}

export function newLens(): Lens {
  return { K: 1, p: 2, half: 2, groove: 0, grooveHalf: 0.25, sides: 0, softUp: 1, softLow: 1, softSide: 1 };
}

export function newSection(): Section {
  return { area: 0, perimeter: 0, soft: 0, lateral: 0, width: 0, supra: 0, supraPerimeter: 0, supraSoft: 0 };
}

/** Lens exponent p and curvature K equivalent to the power law alpha d^beta. */
export function lensFromLaw(alpha: number, beta: number, out: Lens): Lens {
  const p = 1 / Math.max(beta - 1, 0.05);
  const c = (2 * p) / (p + 1);
  out.p = p;
  out.K = Math.pow(c / alpha, p);
  return out;
}

// 6-point Gauss-Legendre rule on [0, 1]
const GX = [0.033765242898424, 0.169395306766868, 0.380690406958402, 0.619309593041598, 0.830604693233132, 0.966234757101576];
const GW = [0.085662246189585, 0.180380786524069, 0.233956967286346, 0.233956967286346, 0.180380786524069, 0.085662246189585];

/** Height of the central channel at |z| (canonical cm; negative where the walls meet). */
export function centralHeight(d: number, L: Lens, z: number): number {
  const a = (1 - L.groove) * L.K;
  const b = L.groove * GROOVE_WALL;
  return d - a * Math.pow(z, L.p) - (z > L.grooveHalf ? b * (z - L.grooveHalf) : 0);
}

/** Height of a lateral channel at |z| (canonical cm; negative where it is sealed). */
export function sideHeight(L: Lens, z: number): number {
  if (L.sides <= 0) return -1;
  const z0 = Math.max(0, L.half - SIDE_WIDTH);
  if (z <= z0) return -SIDE_SEAL;
  return (L.sides * SIDE_DROP * (z - z0)) / (L.half - z0) - SIDE_SEAL;
}

/**
 * Area, perimeter and yielding wall of the cross-section with mid-sagittal
 * height d (canonical cm; 0 or less = the walls touch on the midline).
 */
export function lensSection(d: number, L: Lens, out: Section): Section {
  out.area = 0; out.perimeter = 0; out.soft = 0; out.lateral = 0; out.width = 0;
  out.supra = 0; out.supraPerimeter = 0; out.supraSoft = 0;
  const p = L.p, half = L.half, zg = L.grooveHalf;
  const a = (1 - L.groove) * L.K;
  const b = L.groove * GROOVE_WALL;
  let ze = 0;
  let capped = false;
  // area of the central channel out to |z| = z1 (z1 <= ze)
  const areaTo = (z1: number) => {
    let a1 = d * z1 - (a * Math.pow(z1, p + 1)) / (p + 1);
    if (z1 > zg) a1 -= (b * (z1 - zg) * (z1 - zg)) / 2;
    return 2 * Math.max(0, a1);
  };
  // arc length of its walls out to |z| = z1: each carries half of the slope of h
  const arcTo = (z1: number) => {
    let arc = 0;
    const seg = (z0: number, z2: number, grooved: boolean) => {
      const w = z2 - z0;
      if (w <= 0) return;
      for (let q = 0; q < 6; q++) {
        const z = z0 + GX[q] * w;
        const sl = 0.5 * (a * p * Math.pow(z, p - 1) + (grooved ? b : 0));
        arc += GW[q] * w * Math.sqrt(1 + sl * sl);
      }
    };
    seg(0, Math.min(z1, zg), false);
    if (z1 > zg) seg(zg, z1, true);
    return arc;
  };
  if (d > 0) {
    // where the walls of the central channel meet
    let zw: number;
    const hg = d - a * Math.pow(zg, p);
    if (b <= 0 || hg <= 0) {
      zw = a > 0 ? Math.pow(d / a, 1 / p) : Infinity;
    } else {
      // beyond the groove: d - a z^p - b (z - zg) = 0. The function is concave and
      // negative at both the edge without the groove and the edge without the
      // curvature, so Newton from the nearer of the two converges from the right.
      let z = zg + hg / b;
      if (a > 0) z = Math.min(z, Math.pow(d / a, 1 / p));
      for (let it = 0; it < 30; it++) {
        const f = d - a * Math.pow(z, p) - b * (z - zg);
        const fp = -(a * p * Math.pow(z, p - 1) + b);
        const dz = f / fp;
        z -= dz;
        if (Math.abs(dz) < 1e-9) break;
      }
      zw = z;
    }
    ze = Math.min(zw, half);
    capped = zw > half;
    out.width = 2 * ze;
    out.area = areaTo(ze);
    const arc = arcTo(ze);
    out.perimeter = 4 * arc;
    out.soft = 2 * (L.softUp + L.softLow) * arc;
    if (capped) {
      const hw = Math.max(0, centralHeight(d, L, half));
      out.perimeter += 2 * hw;
      out.soft += 2 * L.softSide * hw;
    }
  }
  // lateral channels: air the central channel does not already carry
  if (L.sides > 0) {
    const z0 = Math.max(0, half - SIDE_WIDTH);
    const drop = L.sides * SIDE_DROP;
    const top = drop - SIDE_SEAL;
    if (top > 0 && half > z0) {
      const zo = z0 + (SIDE_SEAL * (half - z0)) / drop;
      let extra = 0;
      const seg = (u0: number, u1: number) => {
        const w = u1 - u0;
        if (w <= 0) return;
        for (let q = 0; q < 6; q++) {
          const z = u0 + GX[q] * w;
          const hc = d > 0 && z <= ze ? Math.max(0, centralHeight(d, L, z)) : 0;
          const hs = sideHeight(L, z);
          if (hs > hc) extra += GW[q] * w * (hs - hc);
        }
      };
      if (ze > zo && ze < half) { seg(zo, ze); seg(ze, half); } else seg(zo, half);
      const alone = 0.5 * (half - zo) * top;
      // the lumen above the top of the tongue, inside the drawn-in margins,
      // is the supralingual cavity where the flow has split into the lateral
      // channels (Zhang & Espy-Wilson 2004 divide the sections the same way)
      if (out.area > 0) {
        const zc = Math.min(ze, zo);
        const arc = arcTo(zc);
        out.supra = areaTo(zc); out.supraPerimeter = 4 * arc; out.supraSoft = 2 * (L.softUp + L.softLow) * arc;
      }
      // a channel that merges with the central one shares its walls with it
      const frac = alone > 0 ? Math.min(1, extra / alone) : 0;
      const run = half - zo;
      const margin = Math.hypot(run, top);
      out.area += 2 * extra;
      out.lateral = 2 * extra;
      out.perimeter += 2 * frac * (margin + run + top);
      out.soft += 2 * frac * (L.softLow * margin + L.softUp * run + L.softSide * top);
    }
  }
  return out;
}

/** Perimeter of an ellipse with axes w and h (Ramanujan). */
export function ellipsePerimeter(w: number, h: number): number {
  const a = Math.max(0, w) / 2, b = Math.max(0, h) / 2;
  return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
}

/**
 * Half width (canonical cm) of the oral cavity between the lingual surfaces of
 * the upper teeth, where the tongue braces, at a distance D (canonical cm)
 * behind the incisal edge. Upper arch of adult men: about 3.5 cm between the
 * canine tips and 4.8 cm between the first molars (dental cast studies), less
 * between the lingual surfaces; the arch narrows to about 1.6 cm just behind
 * the incisors.
 */
const ARCH_D = [0.33, 1.11, 2.22, 3.33, 4.44, 5.56];
const ARCH_HALF = [0.89, 1.44, 1.72, 1.94, 2.06, 2.0];

export function archHalfWidth(D: number): number {
  if (D <= ARCH_D[0]) return ARCH_HALF[0];
  for (let i = 1; i < ARCH_D.length; i++) {
    if (D <= ARCH_D[i]) {
      const t = (D - ARCH_D[i - 1]) / (ARCH_D[i] - ARCH_D[i - 1]);
      return ARCH_HALF[i - 1] + t * (ARCH_HALF[i] - ARCH_HALF[i - 1]);
    }
  }
  return ARCH_HALF[ARCH_HALF.length - 1];
}

/** Half width of the lumen of the pharynx and of the epilaryngeal tube (canonical cm, adult man). */
export const PHARYNX_HALF = 2.0;
export const LARYNX_HALF = 0.9;

/**
 * The piriform fossae: two blind pouches on either side of the laryngeal tube,
 * opening into the hypopharynx near the top of the tube (about 2.5 cm above
 * the vocal folds) and ending just above the entrance of the oesophagus. In
 * adult men each is about 1.8 cm deep with 1.0-1.4 cm^3 of air; as side
 * branches they put a deep antiresonance at 4-5 kHz into every vowel and lower
 * the formants below it (Dang & Honda 1997). Filling them with water raises the
 * antiresonance until it vanishes.
 */
export const PIRIFORM = {
  /** canonical spine coordinate of the opening (cm) */
  sc: 2.4,
  /** effective length of each pouch for an adult man (cm, life size) */
  length: 2.35,
  /** area of both pouches together at the opening and at the bottom (cm^2, adult man) */
  areaOpen: 1.4,
  areaBottom: 0.5,
};

/** Sections allocated for the fossae (the shortest tract has the shortest sections). */
export const PIRIFORM_SECTIONS = 8;
/** Sections allocated for the supralingual cavity of a lateral (up to about 5 cm). */
export const SUPRA_SECTIONS = 12;
