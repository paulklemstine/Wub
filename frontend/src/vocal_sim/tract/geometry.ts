// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
import { Anatomy, Articulators, Teeth } from './types';
import {
  newLens, newSection, lensSection, lensFromLaw, ellipsePerimeter, archHalfWidth, PHARYNX_HALF, LARYNX_HALF, PIRIFORM,
} from './section3d';

/**
 * Mid-sagittal articulatory model of the vocal tract.
 *
 * In the spirit of Maeda (1990) and Birkholz's VocalTractLab, the tract is
 * described in the mid-sagittal plane by rigid and deformable structures
 * (posterior pharyngeal wall, velum, hard palate, incisors, lips, jaw, tongue).
 * A grid of lines perpendicular to a reference spine (horizontal in the
 * pharynx, radial around the tongue body, near vertical at the front - the
 * classical "semi-polar" grid) is intersected with those structures to get the
 * mid-sagittal width d(s) of the airway. Each grid line then cuts the tract
 * in three dimensions (section3d.ts): a lens between the tongue and the
 * palate or the pharyngeal wall, closed at the sides by the teeth or the
 * lateral walls of the pharynx, with the tongue's groove and lateral channels;
 * an elliptic opening of width w_lip at the lips; an explicit slit between the
 * incisor edges at the teeth. Each section has its area, its perimeter and the
 * part of the perimeter that is soft tissue. The centreline length is measured
 * through the midpoints of the grid lines, so lip protrusion and larynx
 * lowering really lengthen the tube.
 *
 * All coordinates are in cm. The canonical frame (adult male) has the glottis
 * at (0, 0), x towards the lips, y upwards.
 */

export type SolidId =
  | 'wall' | 'palate' | 'upperIncisor' | 'velum' | 'upperLip'
  | 'larynx' | 'tongue' | 'lowerIncisor' | 'lowerLip' | 'floor';

export const UPPER_SOLIDS: SolidId[] = ['wall', 'palate', 'upperIncisor', 'velum', 'upperLip'];
export const LOWER_SOLIDS: SolidId[] = ['larynx', 'tongue', 'lowerIncisor', 'lowerLip', 'floor'];

export interface Solid {
  id: SolidId;
  upper: boolean;
  /** interleaved x, y */
  pts: number[];
  minX: number; maxX: number; minY: number; maxY: number;
}

export interface Ray {
  /** spine point and unit normal (world coordinates) */
  sx: number; sy: number; nx: number; ny: number;
  /** start offset along the normal (negative) and total length */
  t0: number; t1: number;
  /** canonical spine coordinate (cm) used for region classification */
  sc: number;
}

export interface RayHit {
  /** lower and upper boundary positions along the normal (world) */
  tLow: number;
  tUp: number;
  /** mid-sagittal width (cm), 0 = contact */
  d: number;
  /** cross-sectional area (cm^2) */
  area: number;
  /** wetted perimeter of the section (cm) and the part of it that is soft tissue (cm) */
  perimeter: number;
  soft: number;
  /** area of the lateral channels beside the tongue (cm^2) */
  lateral: number;
  /** lateral extent of the channel (cm): the width of the jet a narrow passage makes */
  width: number;
  /** the part of the area, perimeter and yielding perimeter that is the
   *  supralingual cavity of a lateral: the central channel the lateral
   *  channels have left, ending at the tip contact (0 = none) */
  supra: number;
  supraPerimeter: number;
  supraSoft: number;
  /** the cross-section lens of a tongue, pharynx or larynx section (canonical
   *  units, see section3d.ts); lensP = 0 for the lips, teeth and slits */
  lensK: number;
  lensP: number;
  lensHalf: number;
  lensGroove: number;
  lensSides: number;
  /** air pocket below the tongue blade (cm), counted for the front rays */
  sub: number;
  /** the last air pocket below the channel: where it starts along the
   *  normal, the solid below it, and the thickness of the solid that
   *  separates it from the channel (-1 when there is none) */
  pocketT: number;
  pocketId: SolidId | null;
  chord: number;
  lowId: SolidId | null;
  upId: SolidId | null;
  /** true when the grid line is beyond the lips (outside the tract) */
  outside: boolean;
  /** midpoint of the channel */
  mx: number; my: number;
  /** kind of area rule used */
  rule: 'larynx' | 'pharynx' | 'velar' | 'oral' | 'teeth' | 'lips' | 'labiodental' | 'closed';
}

export interface Slit {
  /** vertical opening between the incisor edges (cm, negative = overlap) */
  vertical: number;
  /** horizontal clearance between the incisors (cm) */
  horizontal: number;
  /** effective slit height (cm) */
  height: number;
  /** lateral slit width (cm) */
  width: number;
  /** total slit area including the diastema (cm^2) */
  area: number;
  /** hydraulic diameter of the slit (cm) */
  hydraulic: number;
  /** wetted perimeter of the slit (cm) */
  perimeter: number;
  /** world coordinates of the upper and lower incisor edges */
  ux: number; uy: number; lx: number; ly: number;
}

export interface TractGeometry {
  solids: Solid[];
  rays: Ray[];
  hits: RayHit[];
  /** index of the last ray inside the tract */
  lastRay: number;
  /** cumulative centreline length at each ray (cm) */
  s: Float64Array;
  /** total tract length glottis to lips (cm) */
  length: number;
  slit: Slit;
  /** centreline position of the teeth (cm from the glottis) */
  teethPos: number;
  /** centreline position of the lip opening (start of the lips) */
  lipsPos: number;
  /** lip opening area (cm^2) */
  lipArea: number;
  /** world points of interest for the views */
  landmarks: Record<string, [number, number]>;
  /** tongue body circle (world) for the handles */
  tongueCenter: [number, number];
  tipPoint: [number, number];
  /** jaw rotation in radians */
  jawAngle: number;
  /** spine polyline (world) */
  spine: number[];
  /** size of the drawing: world cm per canonical cm */
  scale: number;
  /** the piriform fossae: where they open along the centreline, how deep the
   *  air in them reaches and how deep they are (cm), and the area of both
   *  together at the opening and at the bottom (cm^2) */
  piriform: { at: number; length: number; full: number; areaOpen: number; areaBottom: number };
}

// ---------------------------------------------------------------- spine ---

const SPINE_X = -0.45;
const BEND_Y = 7.0;
const ARC_C: [number, number] = [2.45, 7.0];
const ARC_R = 2.9;
const ARC_END = (73 * Math.PI) / 180;
const ARC_LEN = ARC_R * (Math.PI - ARC_END);
const FRONT_LEN = 7.0;
export const SPINE_LENGTH = BEND_Y + ARC_LEN + FRONT_LEN;
/** canonical spine coordinate where the pharynx ends / the front segment starts */
export const SC_LARYNX = 2.1;
export const SC_PHARYNX = BEND_Y;
export const SC_FRONT = BEND_Y + ARC_LEN;

function spinePoint(sc: number): [number, number, number, number] {
  // returns x, y, nx, ny with n pointing to the upper/posterior wall
  if (sc <= BEND_Y) return [SPINE_X, sc, -1, 0];
  if (sc <= SC_FRONT) {
    const phi = Math.PI - (sc - BEND_Y) / ARC_R;
    const c = Math.cos(phi), s = Math.sin(phi);
    return [ARC_C[0] + ARC_R * c, ARC_C[1] + ARC_R * s, c, s];
  }
  const ex = ARC_C[0] + ARC_R * Math.cos(ARC_END);
  const ey = ARC_C[1] + ARC_R * Math.sin(ARC_END);
  const tx = Math.sin(ARC_END), ty = -Math.cos(ARC_END);
  const u = sc - SC_FRONT;
  return [ex + tx * u, ey + ty * u, Math.cos(ARC_END), Math.sin(ARC_END)];
}

/** Canonical spine coordinates of the grid lines (denser at the front). */
const GRID_SC: number[] = (() => {
  const out: number[] = [];
  let s = 0;
  while (s < SPINE_LENGTH) {
    out.push(s);
    s += s < SC_FRONT + 1.5 ? 0.25 : 0.14;
  }
  out.push(SPINE_LENGTH);
  return out;
})();

// --------------------------------------------------------- canonical anatomy

const J: [number, number] = [-0.6, 11.0]; // jaw pivot (temporomandibular joint, projected)
const JAW_MAX = (14 * Math.PI) / 180;
/** jaw opening of the held labiodental fricatives /f v/ (see PHONEMES) */
const LABIODENTAL_JAW = 0.12;
/** lip opening (articulator) at which a tucked lower lip reaches the incisal edge */
const LABIODENTAL_EDGE_OPEN = 0.5;

const WALL: number[] = [
  -0.55, -2.5, -0.55, 0.0, -0.62, 0.8, -0.75, 1.8, -0.9, 2.8, -1.02, 4.0, -1.12, 5.2, -1.15, 6.4,
  -1.12, 7.4, -1.02, 8.3, -0.85, 9.1, -0.62, 9.8, -0.45, 10.5, -0.5, 11.3, -0.1, 11.9, 0.6, 12.2,
  0.6, 13.5, -4, 13.5, -4, -2.5,
];

const PALATE: number[] = [
  0.75, 10.15, 1.5, 10.5, 2.5, 10.72, 3.5, 10.78, 4.3, 10.65, 5.0, 10.35, 5.55, 9.95, 5.95, 9.55,
  6.25, 9.25, 6.5, 9.05, 6.85, 9.55, 7.25, 9.85, 7.1, 10.45, 6.5, 10.95, 4.0, 11.3, 1.5, 11.05, 0.7, 10.65,
];

const UPPER_INCISOR: number[] = [
  6.42, 9.15, 6.6, 8.75, 6.78, 8.4, 6.93, 8.12, 7.0, 8.03, 7.1, 8.02, 7.18, 8.2, 7.27, 8.7, 7.32, 9.3,
  7.2, 9.75, 6.7, 9.7,
];

const LOWER_INCISOR: number[] = [
  6.38, 6.95, 6.55, 7.55, 6.72, 7.98, 6.8, 8.18, 6.86, 8.24, 6.94, 8.2, 6.98, 7.8, 7.02, 7.1, 6.85, 6.7,
];

const LARYNX: number[] = [
  0.42, -2.5, 0.42, 0.0, 0.46, 0.8, 0.56, 1.8, 0.72, 2.8, 0.9, 3.55, 1.0, 3.6, 1.12, 3.2, 1.2, 2.5,
  1.55, 2.3, 4.5, 2.0, 4.5, -2.5,
];

const FLOOR: number[] = [
  5.85, 6.55, 6.38, 6.95, 6.85, 6.7, 7.4, 6.0, 7.6, 4.5, 6.0, 3.0, 4.0, 3.0, 4.5, 4.5, 5.3, 5.9,
];

const LIP_CONTACT_Y = 8.28;
const TONGUE_C0: [number, number] = [2.85, 6.95];
const TONGUE_R = 2.45;
const TIP0: [number, number] = [6.15, 7.9];
const FLOOR_ATTACH: [number, number] = [5.75, 6.6];
/** tongue root position of the drawing's rest shape, cm: the lower pharynx of
 *  most vowels measured by MRI (Story 2008) is about 1 cm^2, narrower than the
 *  circle body alone gives */
const ROOT_REST = -1.2;

function rot(p: [number, number], c: [number, number], a: number): [number, number] {
  const ca = Math.cos(a), sa = Math.sin(a);
  const dx = p[0] - c[0], dy = p[1] - c[1];
  return [c[0] + ca * dx - sa * dy, c[1] + sa * dx + ca * dy];
}

function rotPts(pts: number[], c: [number, number], a: number): number[] {
  const out = new Array<number>(pts.length);
  const ca = Math.cos(a), sa = Math.sin(a);
  for (let i = 0; i < pts.length; i += 2) {
    const dx = pts[i] - c[0], dy = pts[i + 1] - c[1];
    out[i] = c[0] + ca * dx - sa * dy;
    out[i + 1] = c[1] + sa * dx + ca * dy;
  }
  return out;
}

/** Tangent angle on circle (c, r) from external point p; side +1 = counterclockwise. */
function tangentAngle(p: [number, number], c: [number, number], r: number, side: 1 | -1): number {
  const dx = p[0] - c[0], dy = p[1] - c[1];
  const D = Math.hypot(dx, dy);
  const phi = Math.atan2(dy, dx);
  if (D <= r * 1.0001) return phi;
  return phi + side * Math.acos(r / D);
}

function buildTongue(a: Articulators, jawAngle: number): { pts: number[]; c: [number, number]; tip: [number, number] } {
  const cLocal: [number, number] = [TONGUE_C0[0] + a.tongueX, TONGUE_C0[1] + a.tongueY];
  const c = rot(cLocal, J, jawAngle * 0.75);
  const r = TONGUE_R;
  const h: [number, number] = [1.4 + 0.6 * a.tongueX, 3.2 + 0.25 * a.tongueY];
  let tip = rot([TIP0[0] + a.tipX, TIP0[1] + a.tipY], J, jawAngle);
  // tongue length constraint: the tip stays within reach of the body
  {
    const dx = tip[0] - c[0], dy = tip[1] - c[1];
    const D = Math.hypot(dx, dy);
    const dMin = r + 0.35, dMax = r + 3.2;
    if (D < dMin) tip = [c[0] + (dx / D) * dMin, c[1] + (dy / D) * dMin];
    else if (D > dMax) tip = [c[0] + (dx / D) * dMax, c[1] + (dy / D) * dMax];
  }
  const fl = rot(FLOOR_ATTACH, J, jawAngle);

  const pts: number[] = [];
  pts.push(h[0], h[1]);
  const phiRoot = tangentAngle(h, c, r, -1);
  let phiBlade = tangentAngle(tip, c, r, 1);
  let start = phiRoot;
  while (start < phiBlade) start += 2 * Math.PI;
  while (start - phiBlade > 2 * Math.PI) start -= 2 * Math.PI;
  const span = start - phiBlade;
  const nArc = Math.max(4, Math.ceil(span / (5 * Math.PI / 180)));
  for (let i = 0; i <= nArc; i++) {
    const phi = start - (span * i) / nArc;
    pts.push(c[0] + r * Math.cos(phi), c[1] + r * Math.sin(phi));
  }
  // blade (straight tangent) to the tip, with a small rounded tip
  const bx = c[0] + r * Math.cos(phiBlade), by = c[1] + r * Math.sin(phiBlade);
  const dx = tip[0] - bx, dy = tip[1] - by;
  const L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L; // blade direction
  const rt = 0.16;
  const tcx = tip[0] - ux * rt - uy * -rt * 0, tcy = tip[1] - uy * rt;
  // tip arc from the upper side (normal (-uy, ux)) around the front to the underside
  const a0 = Math.atan2(ux, -uy); // angle of normal (-uy, ux)
  for (let i = 0; i <= 5; i++) {
    const ang = a0 - (Math.PI * i) / 5;
    pts.push(tcx + rt * Math.cos(ang), tcy + rt * Math.sin(ang));
  }
  // underside: quadratic Bezier from below the tip to the floor attachment
  const u0x = tcx + rt * Math.cos(a0 - Math.PI), u0y = tcy + rt * Math.sin(a0 - Math.PI);
  const cx = Math.min(u0x, fl[0] + 0.5) - 0.25, cy = fl[1] + 0.25;
  for (let i = 1; i <= 8; i++) {
    const t = i / 8;
    const x = (1 - t) * (1 - t) * u0x + 2 * (1 - t) * t * cx + t * t * fl[0];
    const y = (1 - t) * (1 - t) * u0y + 2 * (1 - t) * t * cy + t * t * fl[1];
    pts.push(x, y);
  }
  const deep = rotPts([5.6, 5.4], J, jawAngle * 0.5);
  pts.push(deep[0], deep[1], 4.6, 3.1, 2.2, 2.3, 1.45, 2.6);
  // tongue root: the back of the tongue below the body centre moves forward or
  // back on its own (as in Maeda's tongue-shape parameter), fully at the hyoid
  // and fading out toward the dorsum, so the lower pharynx widens or narrows
  // while the dorsum stays where it is
  const root = a.tongueRoot + ROOT_REST;
  if (root) {
    const top = c[1] + 0.3, bottom = c[1] - r;
    for (let i = 0; i < pts.length; i += 2) {
      if (pts[i] >= c[0] || pts[i + 1] >= top) continue;
      const u = Math.min(1, (top - pts[i + 1]) / (top - bottom));
      pts[i] += root * u * u * (3 - 2 * u);
    }
  }
  return { pts, c, tip };
}

function buildVelum(a: Articulators): number[] {
  const base: [number, number] = [0.78, 10.38];
  // lowering swings the velum down and forward, away from the pharyngeal wall;
  // the uvula stays clear of the tongue dorsum so the oral passage remains open
  const ang = ((207 + 30 * a.velum) * Math.PI) / 180;
  const len = 2.1 + 0.4 * a.velum;
  const n = 8;
  const upper: number[] = [];
  const lower: number[] = [];
  for (let i = 0; i <= n; i++) {
    const s = (len * i) / n;
    const bend = 0.12 * (i / n) * (i / n);
    const dir = ang + bend;
    // centreline by integrating direction approximately
    const cx = base[0] + Math.cos(ang) * s + Math.cos(dir + Math.PI / 2) * bend * 0.6;
    const cy = base[1] + Math.sin(ang) * s + Math.sin(dir + Math.PI / 2) * bend * 0.6;
    const th = 0.46 - 0.22 * (i / n);
    const nx = Math.cos(dir - Math.PI / 2), ny = Math.sin(dir - Math.PI / 2);
    upper.push(cx - nx * th, cy - ny * th);
    lower.push(cx + nx * th, cy + ny * th);
  }
  const pts: number[] = [];
  for (let i = 0; i < upper.length; i += 2) pts.push(upper[i], upper[i + 1]);
  for (let i = lower.length - 2; i >= 0; i -= 2) pts.push(lower[i], lower[i + 1]);
  return pts;
}

function lipProtrusion(a: Articulators): number {
  return Math.min(0.9, 0.65 * a.lipProtrude + 0.25 * a.lipRound);
}

function buildUpperLip(a: Articulators): number[] {
  const p = lipProtrusion(a);
  const y = LIP_CONTACT_Y + 0.3 * a.lipOpen;
  return [
    7.38 + 0.3 * p, 9.65, 7.42 + 0.5 * p, 8.8, 7.5 + 0.8 * p, y + 0.06, 7.8 + p, y, 8.22 + p, y + 0.02,
    8.42 + p, y + 0.25, 8.55 + 0.9 * p, 8.9, 8.42 + 0.5 * p, 9.6, 8.1, 10.15, 7.45, 10.2,
  ];
}

function buildLowerLip(a: Articulators, jawAngle: number): number[] {
  const p = lipProtrusion(a);
  const t = a.lipTuck;
  // Lip closure is a motor goal: for a bilabial (lipOpen well below zero) the
  // lower lip rises relative to the jaw to meet the upper lip whatever the jaw
  // opening (motor equivalence; Abbs & Gracco 1984). The jaw rotation lowers
  // the lip contact point by about 8.8 cm * sin(angle).
  const drive = Math.min(1, Math.max(0, (-a.lipOpen - 0.1) / 0.2));
  const jawDrop = 8.8 * Math.sin(-jawAngle);
  // The same for the labiodentals: a tucked lower lip keeps its slit under
  // the upper incisors whatever the jaw, rising when the jaw is more open than
  // for a held /f v/ (a short /v/ after an open vowel) and giving way when it
  // is closer (a /v/ before a close /u/)
  const tuckDrop = t * (jawDrop - 8.8 * Math.sin(JAW_MAX * LABIODENTAL_JAW));
  const y = LIP_CONTACT_Y - 0.7 * a.lipOpen + 0.25 * t + drive * jawDrop + (1 - drive) * tuckDrop;
  // A lip tucks under the incisors only below their edge, reached near the
  // lip opening of a held /f v/. Rising above it (on the way from /v/ to a
  // rounded /w/, as in "vuole") it slides out in front of them instead of
  // closing against them.
  const rise = 0.7 * (LABIODENTAL_EDGE_OPEN - a.lipOpen) - 0.25 * (1 - t);
  const tx = t * Math.min(1, Math.max(0, 1 - rise / 0.06));
  const pts = [
    7.05 - 0.5 * tx, 7.0, 7.1 - 0.5 * tx, 7.6, 7.25 - 0.55 * tx, y - 0.12, 7.5 - 0.55 * tx, y,
    8.15 + p - 0.3 * tx, y - 0.02, 8.38 + p - 0.2 * tx, y - 0.3, 8.4 + 0.8 * p, 7.4, 8.2, 6.6, 7.7, 6.0, 7.0, 6.2,
  ];
  return rotPts(pts, J, jawAngle);
}

function buildUpperIncisor(t: Teeth): number[] {
  const out = UPPER_INCISOR.slice();
  for (let i = 0; i < out.length; i += 2) {
    const y = out[i + 1];
    const crown = Math.min(1, Math.max(0, (9.75 - y) / 1.7));
    out[i] += (t.advance + t.overjet) * crown;
    if (y < 8.6) out[i + 1] -= t.upperLength * Math.min(1, (8.6 - y) / 0.55);
  }
  return out;
}

function buildLowerIncisor(t: Teeth, jawAngle: number): number[] {
  const out = LOWER_INCISOR.slice();
  for (let i = 0; i < out.length; i += 2) {
    const y = out[i + 1];
    const crown = Math.min(1, Math.max(0, (y - 6.7) / 1.5));
    out[i] += t.advance * crown;
    if (y > 7.7) out[i + 1] += (t.lowerLength - t.slitOpen) * Math.min(1, (y - 7.7) / 0.5);
  }
  return rotPts(out, J, jawAngle);
}

function buildLarynx(a: Articulators): number[] {
  const out = LARYNX.slice();
  for (let i = 0; i < out.length; i += 2) {
    const y = out[i + 1];
    if (y > 1.8 && out[i] < 2) out[i] += (0.5 * a.tongueX + 0.8 * (a.tongueRoot + ROOT_REST)) * Math.min(1, (y - 1.8) / 1.6);
  }
  return out;
}

// ------------------------------------------------------------- transform

interface Xf { k: number; s: number }

/** The canonical drawing is slightly larger than life; this brings the neutral
 *  adult-male tract to about 17.5 cm. */
export const CANON_SCALE = 0.9;

function makeXf(an: Anatomy, a: Articulators): Xf {
  const lp = Math.max(3.0, BEND_Y * an.pharynx - a.larynx / CANON_SCALE);
  return { k: lp / BEND_Y, s: an.scale * CANON_SCALE };
}

/** A speaker's size (world cm per canonical cm) and the vertical stretch of
 *  the canonical drawing below BEND_Y (the pharynx length and the larynx height). */
export function canonTransform(an: Anatomy, a: Articulators): { s: number; k: number } {
  const xf = makeXf(an, a);
  return { s: xf.s, k: xf.k };
}

function xfPts(pts: number[], xf: Xf): number[] {
  const out = new Array<number>(pts.length);
  for (let i = 0; i < pts.length; i += 2) {
    const y = pts[i + 1];
    const yy = y < BEND_Y ? BEND_Y - (BEND_Y - y) * xf.k : y;
    out[i] = pts[i] * xf.s;
    out[i + 1] = yy * xf.s;
  }
  return out;
}

function xfPoint(p: [number, number], xf: Xf): [number, number] {
  const y = p[1] < BEND_Y ? BEND_Y - (BEND_Y - p[1]) * xf.k : p[1];
  return [p[0] * xf.s, y * xf.s];
}

function makeSolid(id: SolidId, pts: number[]): Solid {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < pts.length; i += 2) {
    if (pts[i] < minX) minX = pts[i];
    if (pts[i] > maxX) maxX = pts[i];
    if (pts[i + 1] < minY) minY = pts[i + 1];
    if (pts[i + 1] > maxY) maxY = pts[i + 1];
  }
  return { id, upper: UPPER_SOLIDS.includes(id), pts, minX, maxX, minY, maxY };
}

// ------------------------------------------------------------ ray casting

function pointInPolygon(x: number, y: number, pts: number[]): boolean {
  let inside = false;
  const n = pts.length;
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const xi = pts[i], yi = pts[i + 1], xj = pts[j], yj = pts[j + 1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

interface Ev { t: number; solid: number }
const evBuf: Ev[] = [];
for (let i = 0; i < 128; i++) evBuf.push({ t: 0, solid: 0 });

/** Thickness (cm, scaled) of lower lip over which a split gap stays connected. */
const LIP_SPLIT = 0.3;
/** The lateral channels of a lateral are fully open up to SIDES_FULL behind the
 *  incisal edge and close again over SIDES_FADE as the margins of the tongue
 *  brace against the molars (canonical cm). */
const SIDES_FULL = 4.3;
const SIDES_FADE = 1.0;

function castRay(ray: Ray, solids: Solid[], hit: RayHit): void {
  const px = ray.sx + ray.nx * ray.t0, py = ray.sy + ray.ny * ray.t0;
  const dx = ray.nx, dy = ray.ny;
  const T = ray.t1;
  const ex = px + dx * T, ey = py + dy * T;
  const rminX = Math.min(px, ex), rmaxX = Math.max(px, ex), rminY = Math.min(py, ey), rmaxY = Math.max(py, ey);
  let nev = 0;
  let lowerIn = 0, upperIn = 0;
  const inside: boolean[] = [];
  for (let si = 0; si < solids.length; si++) {
    const sol = solids[si];
    inside.push(false);
    if (sol.maxX < rminX || sol.minX > rmaxX || sol.maxY < rminY || sol.minY > rmaxY) continue;
    const p = sol.pts;
    if (px >= sol.minX && px <= sol.maxX && py >= sol.minY && py <= sol.maxY && pointInPolygon(px, py, p)) {
      inside[si] = true;
      if (sol.upper) upperIn++; else lowerIn++;
    }
    const n = p.length;
    for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
      const ax = p[j], ay = p[j + 1], bx = p[i], by = p[i + 1];
      const ux = bx - ax, uy = by - ay;
      const den = dx * uy - dy * ux;
      if (Math.abs(den) < 1e-12) continue;
      const wx = ax - px, wy = ay - py;
      const t = (wx * uy - wy * ux) / den;
      const s = (wx * dy - wy * dx) / den;
      if (s < 0 || s >= 1 || t <= 0 || t > T) continue;
      if (nev < evBuf.length) { evBuf[nev].t = t; evBuf[nev].solid = si; nev++; }
    }
  }
  // insertion sort by t
  for (let i = 1; i < nev; i++) {
    const t = evBuf[i].t, s = evBuf[i].solid;
    let j = i - 1;
    while (j >= 0 && evBuf[j].t > t) { evBuf[j + 1].t = evBuf[j].t; evBuf[j + 1].solid = evBuf[j].solid; j--; }
    evBuf[j + 1].t = t; evBuf[j + 1].solid = s;
  }
  let airStart = lowerIn === 0 && upperIn === 0 ? 0 : -1;
  let airStartId: SolidId | null = null;
  let lastLowerExit = 0;
  let lastLowerId: SolidId | null = null;
  // the lower solid the ray is in (the one an upper solid touches at a contact)
  let inLowerId: SolidId | null = null;
  for (let si = 0; si < solids.length; si++) if (inside[si] && !solids[si].upper) inLowerId = solids[si].id;
  let sub = 0;
  let pocketT = -1, pocketId: SolidId | null = null, chordStart = 0;
  hit.outside = true;
  hit.sub = 0;
  hit.pocketT = -1; hit.pocketId = null; hit.chord = 0;
  for (let e = 0; e < nev; e++) {
    const ev = evBuf[e];
    const sol = solids[ev.solid];
    const wasIn = inside[ev.solid];
    inside[ev.solid] = !wasIn;
    if (sol.upper) upperIn += wasIn ? -1 : 1; else lowerIn += wasIn ? -1 : 1;
    if (!sol.upper && !wasIn) inLowerId = sol.id;
    if (!sol.upper && wasIn && lowerIn === 0) { lastLowerExit = ev.t; lastLowerId = sol.id; }
    if (lowerIn === 0 && upperIn === 0) {
      if (pocketT >= 0 && airStart < 0) hit.chord = ev.t - chordStart;
      airStart = ev.t; airStartId = sol.id;
    } else if (airStart >= 0 && !wasIn) {
      // entering a solid from air
      if (sol.upper) {
        hit.tLow = airStart; hit.tUp = ev.t; hit.d = ev.t - airStart;
        hit.lowId = airStartId; hit.upId = sol.id; hit.outside = false; hit.sub = sub;
        hit.pocketT = pocketT; hit.pocketId = pocketId;
        if (pocketT < 0) hit.chord = 0;
        return;
      }
      sub += ev.t - airStart; // air pocket between lower solids
      pocketT = airStart; pocketId = airStartId; chordStart = ev.t;
      airStart = -1;
    } else if (!wasIn && sol.upper && lowerIn > 0) {
      // upper solid entered while still inside a lower solid: contact
      hit.tLow = ev.t; hit.tUp = ev.t; hit.d = 0;
      hit.lowId = inLowerId ?? lastLowerId ?? sol.id; hit.upId = sol.id; hit.outside = false; hit.sub = sub;
      return;
    }
  }
  hit.tLow = lastLowerExit; hit.tUp = lastLowerExit; hit.d = 0; hit.lowId = lastLowerId; hit.upId = null;
}

// ------------------------------------------------------- area conversion

/**
 * Region coefficients of the cross-section lenses, written as the power law
 * A = alpha d^beta (cm) of the uncapped lens they are equivalent to (see
 * section3d.ts).
 */
export const AREA_LAW = {
  // the epilaryngeal tube above the glottis is about 0.3-0.6 cm^2 (Story 2008)
  larynx: { alpha: 0.55, beta: 1.3 },
  pharynx: { alpha: 1.9, beta: 1.45 },
  oral: { alpha: 1.6, beta: 1.5 },
  /** lateral width of a grooved blade channel (cm) */
  grooveWidth: 0.5,
  /** effective width of the labiodental slit (cm) */
  labiodentalWidth: 2.0,
  /** shape factor of the incisal slit (lens-shaped opening) */
  slitShape: 0.85,
  /** interdental leakage area when the teeth are closed (cm^2) */
  leak: 0.015,
};

export function lipWidth(a: Articulators, scale: number): number {
  return scale * (4.6 - 3.5 * Math.min(1, a.lipRound));
}

function computeSlit(upper: number[], lower: number[], t: Teeth, scale: number, groove: number): Slit {
  // incisal edges: lowest point of the upper incisor, highest point of the lower one
  let ux = 0, uy = Infinity, lx = 0, ly = -Infinity;
  for (let i = 0; i < upper.length; i += 2) if (upper[i + 1] < uy) { uy = upper[i + 1]; ux = upper[i]; }
  for (let i = 0; i < lower.length; i += 2) if (lower[i + 1] > ly) { ly = lower[i + 1]; lx = lower[i]; }
  const vertical = uy - ly;
  // horizontal clearance: upper incisor lingual surface vs lower labial surface
  const horizontal = Math.max(0.02 * scale, ux - lx - 0.12 * scale);
  let height: number;
  if (vertical > 0) height = Math.hypot(vertical, Math.max(0, ux - lx - 0.12 * scale) * 0.5);
  else height = Math.max(0.02 * scale, horizontal * 0.6 + vertical * 0.15);
  height = Math.max(0, height);
  // with a flat (ungrooved) tongue the air also escapes between the lateral
  // teeth; a grooved blade seals the sides and leaves only the incisal slit
  const width = (Math.max(0.05, t.slitWidth) + 2.2 * (1 - Math.min(1, Math.max(0, groove)))) * scale;
  const dia = Math.max(0, t.diastema) * scale;
  const area = AREA_LAW.slitShape * height * width + dia * Math.min(0.7 * scale, height + 0.5 * scale) + AREA_LAW.leak * scale * scale;
  const hydraulic = (2 * height * width) / (height + width + 1e-9);
  const perimeter = 2 * (height + width) + (dia > 0 ? 2 * Math.min(0.7 * scale, height + 0.5 * scale) : 0);
  return { vertical, horizontal, height, width, area, hydraulic: Math.max(hydraulic, 0.02), perimeter, ux, uy, lx, ly };
}

// ---------------------------------------------------------------- main

export interface GeometryOptions {
  /** reuse storage of a previous result */
  reuse?: TractGeometry;
}

/** Builds the mid-sagittal geometry and the fine (grid-line) area function. */
export function computeGeometry(a: Articulators, an: Anatomy, teeth: Teeth, opts: GeometryOptions = {}): TractGeometry {
  // opening the jaw is a clockwise rotation (y up) about the condyle
  const jawAngle = -JAW_MAX * Math.max(-0.1, Math.min(1.2, a.jaw));
  const xf = makeXf(an, a);
  const s = xf.s;

  const tongue = buildTongue(a, jawAngle);
  const upperInc = buildUpperIncisor(teeth);
  const lowerInc = buildLowerIncisor(teeth, jawAngle);
  const solidsCanon: [SolidId, number[]][] = [
    ['wall', WALL],
    ['palate', PALATE],
    ['upperIncisor', upperInc],
    ['velum', buildVelum(a)],
    ['upperLip', buildUpperLip(a)],
    ['larynx', buildLarynx(a)],
    ['tongue', tongue.pts],
    ['lowerIncisor', lowerInc],
    ['lowerLip', buildLowerLip(a, jawAngle)],
    ['floor', rotPts(FLOOR, J, jawAngle)],
  ];
  const solids = solidsCanon.map(([id, pts]) => makeSolid(id, xfPts(pts, xf)));

  // grid lines from the transformed spine
  const prev = opts.reuse;
  const rays: Ray[] = prev?.rays ?? [];
  const hits: RayHit[] = prev?.hits ?? [];
  const spine: number[] = [];
  for (let i = 0; i < GRID_SC.length; i++) {
    const sc = GRID_SC[i];
    const [x, y, nx, ny] = spinePoint(sc);
    const [wx, wy] = xfPoint([x, y], xf);
    // normals: horizontal in the pharynx are unchanged by the vertical stretch
    const t0 = sc <= BEND_Y ? -3.4 : sc <= SC_FRONT ? -2.75 : -2.2;
    const t1 = 6.0;
    if (!rays[i]) rays[i] = { sx: 0, sy: 0, nx: 0, ny: 0, t0: 0, t1: 0, sc: 0 };
    const r = rays[i];
    r.sx = wx; r.sy = wy; r.nx = nx; r.ny = ny; r.t0 = t0 * s; r.t1 = t1 * s; r.sc = sc;
    spine.push(wx, wy);
    if (!hits[i]) hits[i] = { tLow: 0, tUp: 0, d: 0, area: 0, perimeter: 0, soft: 0, lateral: 0, width: 0, supra: 0, supraPerimeter: 0, supraSoft: 0, lensK: 0, lensP: 0, lensHalf: 0, lensGroove: 0, lensSides: 0, sub: 0, pocketT: -1, pocketId: null, chord: 0, lowId: null, upId: null, outside: true, mx: 0, my: 0, rule: 'oral' };
  }
  rays.length = GRID_SC.length;
  hits.length = GRID_SC.length;

  const slit = computeSlit(xfPts(upperInc, xf), xfPts(lowerInc, xf), teeth, s, a.groove);
  const wLip = lipWidth(a, s);
  const law = AREA_LAW;
  const scaleArea = (alpha: number, beta: number) => alpha * Math.pow(s, 2 - beta);
  const pw = an.pharynxWidth;
  const lens = newLens(), pocket = newLens();
  const sec = newSection(), sec2 = newSection();
  lensFromLaw(law.oral.alpha, law.oral.beta, pocket);
  const tongueSides = Math.min(1, Math.max(0, a.tongueSides));

  let lastRay = rays.length - 1;
  let firstOutside = -1;
  for (let i = 0; i < rays.length; i++) {
    const ray = rays[i];
    const h = hits[i];
    castRay(ray, solids, h);
    if (h.outside) {
      if (i > 10 && firstOutside < 0) firstOutside = i;
      h.area = 0; h.perimeter = 0; h.soft = 0; h.lateral = 0; h.width = 0; h.rule = 'oral';
      h.supra = 0; h.supraPerimeter = 0; h.supraSoft = 0;
      continue;
    }
    let tLowW = ray.t0 + h.tLow;
    const tUpW = ray.t0 + h.tUp;
    const d = Math.max(0, h.d);
    const sc = ray.sc;
    let area = 0, perim = 0, soft = 0, lateral = 0, width = 0;
    let supra = 0, supraP = 0, supraS = 0;
    let rule: RayHit['rule'];
    h.lensP = 0;
    const isLipU = h.upId === 'upperLip', isLipL = h.lowId === 'lowerLip';
    const isTeethU = h.upId === 'upperIncisor', isTeethL = h.lowId === 'lowerIncisor';
    if (isTeethU && isLipL) {
      rule = 'labiodental';
      const w = law.labiodentalWidth * s * 0.8;
      area = w * d; width = w;
      perim = 2 * (w + d); soft = 0.5 * perim; // a soft lip under hard teeth
    } else if (isTeethL && isLipU) {
      // from the lower incisors up to the upper lip, across the space behind
      // the lips: the lips, not the slit between the incisors, bound it at
      // the sides (the rule keeps the name of the teeth region)
      rule = 'teeth';
      area = (Math.PI / 4) * d * wLip; width = wLip;
      perim = ellipsePerimeter(wLip, d); soft = 0.5 * perim;
    } else if (isTeethU || isTeethL) {
      rule = 'teeth';
      if (isTeethU && isTeethL) {
        area = slit.area; perim = slit.perimeter; soft = 0; width = slit.width;
      } else {
        const oral = scaleArea(law.oral.alpha, law.oral.beta) * Math.pow(d, law.oral.beta);
        area = Math.min(oral, Math.max(slit.area, law.slitShape * d * slit.width));
        // a slit between the teeth and the tongue or a lip
        const w = Math.min(slit.width, area / Math.max(d, 0.01 * s));
        perim = 2 * (w + area / Math.max(w, 0.01 * s)); soft = 0.5 * perim; width = w;
      }
    } else if (isLipU || isLipL) {
      rule = 'lips';
      area = (Math.PI / 4) * d * wLip; width = wLip;
      perim = ellipsePerimeter(wLip, d); soft = perim;
    } else {
      // a lens between the tongue (or the larynx) and the pharyngeal wall,
      // the velum or the palate, closed at the sides
      const mxc = (ray.sx + ray.nx * 0.5 * (tLowW + tUpW)) / s;
      const behind = slit.ux / s - mxc; // canonical distance behind the incisal edge
      let f = 0;
      if (sc < SC_LARYNX) {
        rule = 'larynx';
        lensFromLaw(law.larynx.alpha, law.larynx.beta, lens);
        lens.half = LARYNX_HALF;
      } else if (sc <= SC_PHARYNX) {
        rule = 'pharynx';
        lensFromLaw(law.pharynx.alpha * pw, law.pharynx.beta, lens);
        lens.half = PHARYNX_HALF * pw;
      } else {
        rule = sc <= SC_FRONT ? 'velar' : 'oral';
        f = sc <= SC_FRONT ? (sc - SC_PHARYNX) / (SC_FRONT - SC_PHARYNX) : 1;
        lensFromLaw(law.pharynx.alpha * pw * (1 - f) + law.oral.alpha * f, law.pharynx.beta * (1 - f) + law.oral.beta * f, lens);
        lens.half = PHARYNX_HALF * pw * (1 - f) + archHalfWidth(behind) * f;
      }
      lens.groove = rule === 'oral' ? Math.min(1, Math.max(0, a.groove)) : 0;
      lens.grooveHalf = 0.5 * law.grooveWidth;
      // the lateral channels of a lateral run from the tip contact back to
      // where the margins of the tongue brace against the molars again
      const brace = Math.min(1, Math.max(0, (behind - SIDES_FULL) / SIDES_FADE));
      lens.sides = h.lowId === 'tongue' && f > 0 ? tongueSides * (1 - brace) : 0;
      // the hard palate does not yield; the teeth and cheeks close the oral cavity at the sides
      lens.softUp = h.upId === 'palate' ? 0 : 1;
      lens.softLow = 1;
      lens.softSide = 1 - 0.5 * f;
      lensSection(d / s, lens, sec);
      h.lensK = lens.K; h.lensP = lens.p; h.lensHalf = lens.half; h.lensGroove = lens.groove; h.lensSides = lens.sides;
      area = sec.area * s * s; perim = sec.perimeter * s; soft = sec.soft * s; lateral = sec.lateral * s * s; width = sec.width * s;
      // the flow splits where the channels are fully open: in front of that
      // the central channel is a blind cavity
      if (brace <= 0 && sec.supra > 0) { supra = sec.supra * s * s; supraP = sec.supraPerimeter * s; supraS = sec.supraSoft * s; }
      // the air pocket under a raised blade is laterally connected to the channel
      if (rule === 'oral' && d > 0 && h.sub > 0.05 * s) {
        pocket.half = lens.half;
        lensSection(h.sub / s, pocket, sec2);
        area += 0.7 * sec2.area * s * s; perim += 0.7 * sec2.perimeter * s; soft += 0.7 * sec2.soft * s;
      }
    }
    if (d <= 0 && lateral <= 0) {
      if (rule === 'teeth' && isTeethU && isTeethL) { area = slit.area; perim = slit.perimeter; soft = 0; width = slit.width; }
      else { area = 0; perim = 0; soft = 0; width = 0; rule = 'closed'; }
    }
    // the tip of the lower lip entering the gap above the lower incisors
    // splits it in two; while the part of the lip on the line is thin the
    // air still flows around it, so the area and the midpoint pass smoothly
    // from the undivided gap to the one between the lips
    if (isLipL && h.pocketId === 'lowerIncisor' && h.chord < LIP_SPLIT * s) {
      const w = (1 - h.chord / (LIP_SPLIT * s)) ** 2;
      const dFull = h.tUp - h.pocketT;
      const full = isTeethU ? slit.area : (Math.PI / 4) * dFull * wLip;
      const fullP = isTeethU ? slit.perimeter : ellipsePerimeter(wLip, dFull);
      area = w * full + (1 - w) * area;
      soft = w * (isTeethU ? 0 : 0.5 * fullP) + (1 - w) * soft;
      perim = w * fullP + (1 - w) * perim;
      width = w * (isTeethU ? slit.width : wLip) + (1 - w) * width;
      tLowW = ray.t0 + (w * h.pocketT + (1 - w) * h.tLow);
    }
    h.mx = ray.sx + ray.nx * 0.5 * (tLowW + tUpW);
    h.my = ray.sy + ray.ny * 0.5 * (tLowW + tUpW);
    h.area = area;
    h.perimeter = perim;
    h.soft = soft;
    h.lateral = lateral;
    h.width = width;
    h.supra = supra; h.supraPerimeter = supraP; h.supraSoft = supraS;
    h.rule = rule;
  }
  if (firstOutside > 0) lastRay = firstOutside - 1;
  // a central channel apart from the lateral channels is a blind cavity only
  // when the tip closes it in front; otherwise the channels join again
  for (let i = 0; i <= lastRay; i++) {
    if (hits[i].supra <= 0) continue;
    let j = i;
    while (j < lastRay && hits[j + 1].supra > 0) j++;
    const front = hits[j + 1];
    if (!(j < lastRay && front.d <= 0 && front.lateral > 0)) {
      for (let q = i; q <= j; q++) { hits[q].supra = 0; hits[q].supraPerimeter = 0; hits[q].supraSoft = 0; }
    }
    i = j;
  }

  // centreline length through the channel midpoints
  const sArr = prev?.s && prev.s.length === rays.length ? prev.s : new Float64Array(rays.length);
  sArr[0] = 0;
  for (let i = 1; i <= lastRay; i++) {
    const h0 = hits[i - 1], h1 = hits[i];
    sArr[i] = sArr[i - 1] + Math.hypot(h1.mx - h0.mx, h1.my - h0.my);
  }
  for (let i = lastRay + 1; i < rays.length; i++) sArr[i] = sArr[lastRay];
  const length = sArr[lastRay];

  // teeth and lips positions along the centreline
  let teethPos = -1, lipsPos = -1, lipArea = 0;
  // the obstacle sits at the incisal edges: the grid line passing closest to them
  const ex = 0.5 * (slit.ux + slit.lx), ey = 0.5 * (slit.uy + slit.ly);
  let bestD = Infinity;
  for (let i = 0; i <= lastRay; i++) {
    const h = hits[i];
    const r = rays[i];
    const ax = r.sx + r.nx * (r.t0 + h.tLow), ay = r.sy + r.ny * (r.t0 + h.tLow);
    const bx = r.sx + r.nx * (r.t0 + h.tUp), by = r.sy + r.ny * (r.t0 + h.tUp);
    const vx = bx - ax, vy = by - ay;
    const L2 = vx * vx + vy * vy;
    const t = L2 > 1e-9 ? Math.max(0, Math.min(1, ((ex - ax) * vx + (ey - ay) * vy) / L2)) : 0;
    const dd = Math.hypot(ax + vx * t - ex, ay + vy * t - ey);
    if (dd < bestD) { bestD = dd; teethPos = sArr[i]; }
    if (h.rule === 'lips' || h.rule === 'labiodental') { if (lipsPos < 0) lipsPos = sArr[i]; lipArea = h.area; }
  }
  if (teethPos < 0) teethPos = length - 1.0 * s;
  if (lipsPos < 0) lipsPos = length - 0.5 * s;
  if (lipArea === 0) lipArea = hits[lastRay].area;

  const lm = (p: [number, number]) => xfPoint(p, xf);
  const landmarks: Record<string, [number, number]> = {
    glottis: lm([0, 0]),
    jawPivot: lm(J),
    pns: lm([0.75, 10.15]),
    upperEdge: [slit.ux, slit.uy],
    lowerEdge: [slit.lx, slit.ly],
    lipFront: lm([8.22 + lipProtrusion(a), LIP_CONTACT_Y + 0.3 * a.lipOpen]),
    nostril: lm([8.0, 10.6]),
  };

  // the piriform fossae open where the epilaryngeal tube ends
  let pirAt = 0;
  for (let i = 0; i <= lastRay; i++) if (rays[i].sc >= PIRIFORM.sc) { pirAt = sArr[i]; break; }
  const pirFull = PIRIFORM.length * an.scale;
  const piriform = {
    at: pirAt, full: pirFull, length: pirFull * Math.min(1, Math.max(0, an.piriform)),
    areaOpen: PIRIFORM.areaOpen * an.scale * an.scale, areaBottom: PIRIFORM.areaBottom * an.scale * an.scale,
  };

  return {
    solids, rays, hits, lastRay, s: sArr, length, slit, teethPos, lipsPos, lipArea, landmarks,
    tongueCenter: lm(tongue.c), tipPoint: lm(tongue.tip), jawAngle, spine, scale: s, piriform,
  };
}

/** Converts world coordinates of a tongue-centre drag into articulator offsets. */
export function tongueOffsetFromWorld(x: number, y: number, a: Articulators, an: Anatomy): { tongueX: number; tongueY: number } {
  const jawAngle = -JAW_MAX * a.jaw;
  const xf = makeXf(an, a);
  const yc = y / xf.s;
  const canon: [number, number] = [x / xf.s, yc < BEND_Y ? BEND_Y - (BEND_Y - yc) / xf.k : yc];
  const local = rot(canon, J, -jawAngle * 0.75);
  return { tongueX: local[0] - TONGUE_C0[0], tongueY: local[1] - TONGUE_C0[1] };
}

export function tipOffsetFromWorld(x: number, y: number, a: Articulators, an: Anatomy): { tipX: number; tipY: number } {
  const jawAngle = -JAW_MAX * a.jaw;
  const xf = makeXf(an, a);
  const yc = y / xf.s;
  const canon: [number, number] = [x / xf.s, yc < BEND_Y ? BEND_Y - (BEND_Y - yc) / xf.k : yc];
  const local = rot(canon, J, -jawAngle);
  return { tipX: local[0] - TIP0[0], tipY: local[1] - TIP0[1] };
}

/** Jaw opening value whose lower incisor edge is closest to a dragged point. */
export function jawFromWorld(x: number, y: number, an: Anatomy): number {
  const p: [number, number] = [x / (an.scale * CANON_SCALE), y / (an.scale * CANON_SCALE)];
  const a0 = Math.atan2(8.24 - J[1], 6.86 - J[0]);
  const a1 = Math.atan2(p[1] - J[1], p[0] - J[0]);
  let d = a0 - a1;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return Math.max(0, Math.min(1, d / JAW_MAX));
}

export const GEOMETRY_CONST = { J, JAW_MAX, TONGUE_R, LIP_CONTACT_Y, BEND_Y };

/**
 * Maps canonical drawing coordinates (interleaved x, y) to world coordinates
 * with the speaker's size and pharynx stretch. `jawWeight` optionally rotates
 * each point with the jaw by a weight 0..1 (decorative skin of the chin).
 */
export function canonToWorld(pts: number[], an: Anatomy, a: Articulators, jawWeight?: (x: number, y: number) => number): number[] {
  let p = pts;
  if (jawWeight) {
    const jawAngle = -JAW_MAX * Math.max(-0.1, Math.min(1.2, a.jaw));
    p = new Array<number>(pts.length);
    for (let i = 0; i < pts.length; i += 2) {
      const w = jawWeight(pts[i], pts[i + 1]);
      const r = rot([pts[i], pts[i + 1]], J, jawAngle * w);
      p[i] = r[0]; p[i + 1] = r[1];
    }
  }
  return xfPts(p, makeXf(an, a));
}

/** Velum opening (0..1) for a dragged world point near the velum tip. */
export function velumFromWorld(x: number, y: number, an: Anatomy): number {
  const s = an.scale * CANON_SCALE;
  const dx = x / s - 0.78, dy = y / s - 10.38;
  let ang = (Math.atan2(dy, dx) * 180) / Math.PI;
  if (ang < 0) ang += 360;
  return Math.max(0, Math.min(1, (ang - 207) / 30));
}

/** Tip of the velum (world), for the handle. */
export function velumTip(a: Articulators, an: Anatomy): [number, number] {
  const ang = ((207 + 30 * a.velum) * Math.PI) / 180;
  const len = 2.1 + 0.4 * a.velum - 0.25;
  const s = an.scale * CANON_SCALE;
  return [(0.78 + Math.cos(ang) * len) * s, (10.38 + Math.sin(ang) * len) * s];
}
