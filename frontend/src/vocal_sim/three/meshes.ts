// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/**
 * Three-dimensional meshes of the vocal tract, built from the same
 * cross-sections the acoustics uses (src/dsp/tract/section3d.ts): on every
 * grid line of the mid-sagittal model the section is a lens between the
 * upper wall and the tongue (or an ellipse between the lips, a slit between
 * the teeth), and here each wall gets its place across the tract. Pure data:
 * flat arrays for the WebGL view, no three.js and no DOM, so that it can be
 * tested.
 *
 * World frame: x forward (toward the lips), y up, z to the speaker's left,
 * in cm; the mid-sagittal plane is z = 0.
 */
import { TractGeometry, RayHit, SolidId, AREA_LAW } from '../tract/geometry';
import { centralHeight, sideHeight, newLens, Lens, LARYNX_HALF } from '../tract/section3d';

/** Points across one side of a section (the ring of the airway has twice as many). */
export const RING_SIDE = 15;
/** Points across a tissue sheet. */
export const SHEET = 21;

/** Share of the approach of the walls taken by the upper wall, by structure. */
function upperShare(id: SolidId | null): number {
  switch (id) {
    case 'wall': return 0.3;
    case 'velum': return 0.4;
    case 'palate': return 0.55;
    default: return 0.5;
  }
}

/** Tissue thickness (canonical cm) drawn behind each wall; 0 = drawn otherwise (teeth). */
const THICK: Record<SolidId, number> = {
  wall: 0.9, palate: 0.45, upperIncisor: 0, velum: 0.7, upperLip: 0.9,
  larynx: 0.6, tongue: 1.8, lowerIncisor: 0, lowerLip: 0.9, floor: 0.8,
};

/** Colour (linear-ish RGB 0..1) of each structure. */
export const TISSUE_RGB: Record<SolidId | 'none', [number, number, number]> = {
  wall: [0.64, 0.31, 0.4], palate: [0.8, 0.75, 0.65], upperIncisor: [0.96, 0.93, 0.87], velum: [0.77, 0.41, 0.49],
  upperLip: [0.88, 0.48, 0.55], larynx: [0.64, 0.31, 0.4], tongue: [0.87, 0.44, 0.53], lowerIncisor: [0.96, 0.93, 0.87],
  lowerLip: [0.88, 0.48, 0.55], floor: [0.58, 0.28, 0.36], none: [0.22, 0.18, 0.23],
};

/** The walls of one section at lateral position z, as offsets along its grid line. */
export interface Walls {
  /** upper and lower wall (cm from the start of the grid line) */
  roof: number;
  floor: number;
}

const lensScratch = newLens();

function lensOf(h: RayHit): Lens {
  const L = lensScratch;
  L.K = h.lensK; L.p = h.lensP; L.half = h.lensHalf; L.groove = h.lensGroove;
  L.grooveHalf = 0.5 * AREA_LAW.grooveWidth; L.sides = h.lensSides;
  return L;
}

/**
 * Upper and lower wall of section i at lateral position z (world cm). Past
 * the edge of the airway both walls coincide: the tongue touches the palate
 * or the teeth, the lips close at the corners.
 */
export function wallsAt(g: TractGeometry, i: number, z: number, out: Walls): Walls {
  const h = g.hits[i];
  const s = g.scale;
  const d = Math.max(0, h.d);
  const az = Math.abs(z);
  if (h.lensP > 0) {
    const L = lensOf(h);
    const zc = az / s;
    // the central channel: its walls keep approaching past its edge (the
    // vault of the palate, the margins of the tongue below it), by at most
    // a centimetre more
    const hc = Math.max(-s, s * centralHeight(d / s, L, Math.min(zc, L.half)));
    const hs = L.sides > 0 ? s * sideHeight(L, Math.min(zc, L.half)) : -1;
    const lam = upperShare(h.upId);
    out.roof = h.tUp - lam * (d - Math.min(hc, d));
    out.floor = out.roof - Math.max(0, hc, hs);
    return out;
  }
  if (h.rule === 'lips' || (h.rule === 'teeth' && h.lowId === 'lowerIncisor' && h.upId === 'upperLip')) {
    // an ellipse between the lips (or from the lower incisors to the upper lip)
    const a = 0.5 * h.width;
    const gap = a > 0 && az < a ? d * Math.sqrt(1 - (az / a) * (az / a)) : 0;
    out.roof = h.tUp - 0.5 * (d - gap);
    out.floor = out.roof - gap;
    return out;
  }
  // slits: between the incisors, a lip under the teeth, the teeth and the tongue
  const a = 0.5 * h.width;
  const gap = az <= a ? d : 0;
  out.roof = h.tUp - 0.5 * (d - gap);
  out.floor = out.roof - gap;
  return out;
}

/** Lateral half extent (world cm) of the airway of section i. */
export function airHalfWidth(g: TractGeometry, i: number): number {
  const h = g.hits[i];
  if (h.rule === 'closed') return 0;
  if (h.lensP > 0) return h.lensSides > 0 ? h.lensHalf * g.scale : 0.5 * h.width;
  return 0.5 * h.width;
}

/** Lateral half extent (world cm) of the tissues around section i. */
function tissueHalfWidth(g: TractGeometry, i: number): number {
  const h = g.hits[i];
  const s = g.scale;
  if (h.lensP > 0) return h.lensHalf * s + 0.25 * s;
  if (h.rule === 'lips') return 0.5 * h.width + 0.5 * s;
  return Math.max(0.5 * h.width, 0.6 * s);
}

export interface Sheet {
  /** xyz per vertex: per grid line SHEET points on the wall, then SHEET behind it */
  positions: Float32Array;
  colors: Float32Array;
  index: Uint32Array;
}

export interface Tooth {
  x: number; y: number; z: number;
  /** mesio-distal width, crown height, bucco-lingual depth (cm) */
  w: number; h: number; d: number;
  /** turn about the vertical axis (along the arch) and tilt of the occlusal plane (rad) */
  yaw: number; tilt: number;
  lower: boolean;
}

export interface Pouch {
  /** opening (world x, y), direction of the pouch (unit, x y), lateral offset of each pouch */
  x: number; y: number; dx: number; dy: number; z: number;
  /** full depth and depth of air (cm); radius at the opening and at the bottom (cm) */
  length: number; air: number; rOpen: number; rBottom: number;
}

export interface TractMeshes {
  /** the airway: per grid line a ring of 2 RING_SIDE points (roof left to
   *  right, floor right to left) */
  air: { positions: Float32Array; ray: Uint16Array; index: Uint32Array };
  upper: Sheet;
  lower: Sheet;
  teeth: Tooth[];
  pouch: Pouch;
  /** centre and size of the tract, for the camera */
  center: [number, number, number];
  radius: number;
  /** points to label (null = not present) */
  anchors: Record<'tongue' | 'palate' | 'velum' | 'teeth' | 'lips' | 'larynx' | 'pharynx' | 'piriform' | 'channels' | 'supra', [number, number, number] | null>;
}

const wTmp: Walls = { roof: 0, floor: 0 };

function point(g: TractGeometry, i: number, t: number, z: number, out: Float32Array, o: number): void {
  const r = g.rays[i];
  out[o] = r.sx + r.nx * (r.t0 + t);
  out[o + 1] = r.sy + r.ny * (r.t0 + t);
  out[o + 2] = z;
}

/** Triangles of a strip of rings: ring i point j to ring i+1, closed or open across. */
function stripIndex(rings: number, per: number, closed: boolean): Uint32Array {
  const across = closed ? per : per - 1;
  const idx = new Uint32Array((rings - 1) * across * 6);
  let q = 0;
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < across; j++) {
      const a = i * per + j, b = i * per + ((j + 1) % per), c = a + per, d = b + per;
      idx[q++] = a; idx[q++] = c; idx[q++] = b;
      idx[q++] = b; idx[q++] = c; idx[q++] = d;
    }
  }
  return idx;
}

/** Index of a slab: the wall surface, the surface behind it and the two edges. */
function slabIndex(rings: number): Uint32Array {
  const per = 2 * SHEET;
  const out: number[] = [];
  for (let i = 0; i < rings - 1; i++) {
    const r0 = i * per, r1 = (i + 1) * per;
    for (let j = 0; j < SHEET - 1; j++) {
      // wall surface
      out.push(r0 + j, r1 + j, r0 + j + 1, r0 + j + 1, r1 + j, r1 + j + 1);
      // behind
      const b0 = r0 + SHEET + j, b1 = r1 + SHEET + j;
      out.push(b0, b0 + 1, b1, b0 + 1, b1 + 1, b1);
    }
    // the two lateral edges
    for (const [w, b] of [[0, SHEET], [SHEET - 1, 2 * SHEET - 1]]) {
      out.push(r0 + w, r0 + b, r1 + w, r0 + b, r1 + b, r1 + w);
    }
  }
  return Uint32Array.from(out);
}

/** Lateral sample positions in -1..1, denser toward the edges. */
const ACROSS = Array.from({ length: RING_SIDE }, (_, j) => Math.sin((Math.PI / 2) * (-1 + (2 * j) / (RING_SIDE - 1))));
const SHEET_ACROSS = Array.from({ length: SHEET }, (_, j) => -1 + (2 * j) / (SHEET - 1));

/**
 * Builds (or refills, with `out`) the meshes of a geometry. The topology
 * depends only on the number of grid lines, so the arrays are reused.
 */
export function buildTractMeshes(g: TractGeometry, out?: TractMeshes): TractMeshes {
  const R = g.rays.length;
  const last = g.lastRay;
  const s = g.scale;
  const ringN = 2 * RING_SIDE;
  const m: TractMeshes = out ?? {
    air: { positions: new Float32Array(R * ringN * 3), ray: new Uint16Array(R * ringN), index: stripIndex(R, ringN, true) },
    upper: { positions: new Float32Array(R * 2 * SHEET * 3), colors: new Float32Array(R * 2 * SHEET * 3), index: slabIndex(R) },
    lower: { positions: new Float32Array(R * 2 * SHEET * 3), colors: new Float32Array(R * 2 * SHEET * 3), index: slabIndex(R) },
    teeth: [],
    pouch: { x: 0, y: 0, dx: 0, dy: -1, z: 0, length: 0, air: 0, rOpen: 0, rBottom: 0 },
    center: [0, 0, 0], radius: 10,
    anchors: { tongue: null, palate: null, velum: null, teeth: null, lips: null, larynx: null, pharynx: null, piriform: null, channels: null, supra: null },
  };
  const ap = m.air.positions, ar = m.air.ray;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < R; i++) {
    const src = Math.min(i, last);
    const zmax = airHalfWidth(g, src);
    for (let j = 0; j < RING_SIDE; j++) {
      const z = zmax * ACROSS[j];
      wallsAt(g, src, z, wTmp);
      point(g, src, wTmp.roof, z, ap, (i * ringN + j) * 3);
      point(g, src, wTmp.floor, z, ap, (i * ringN + ringN - 1 - j) * 3);
      ar[i * ringN + j] = src; ar[i * ringN + ringN - 1 - j] = src;
    }
    const o = (i * ringN) * 3;
    minX = Math.min(minX, ap[o]); maxX = Math.max(maxX, ap[o]); minY = Math.min(minY, ap[o + 1]); maxY = Math.max(maxY, ap[o + 1]);
  }
  // tissues behind the walls
  for (const [sheet, upper] of [[m.upper, true], [m.lower, false]] as [Sheet, boolean][]) {
    const P = sheet.positions, C = sheet.colors;
    for (let i = 0; i < R; i++) {
      const src = Math.min(i, last);
      const h = g.hits[src];
      const id = upper ? h.upId : h.lowId;
      const thick = (id ? THICK[id] : 0) * s;
      const W = tissueHalfWidth(g, src);
      const col = TISSUE_RGB[id ?? 'none'];
      for (let j = 0; j < SHEET; j++) {
        const z = W * SHEET_ACROSS[j];
        wallsAt(g, src, z, wTmp);
        const t = upper ? wTmp.roof : wTmp.floor;
        // the tissue thins toward its lateral edge
        const behind = thick * (0.35 + 0.65 * Math.sqrt(Math.max(0, 1 - SHEET_ACROSS[j] * SHEET_ACROSS[j])));
        const a = (i * 2 * SHEET + j) * 3, b = (i * 2 * SHEET + SHEET + j) * 3;
        point(g, src, t, z, P, a);
        point(g, src, upper ? t + behind : t - behind, z, P, b);
        for (let c = 0; c < 3; c++) { C[a + c] = col[c]; C[b + c] = col[c] * 0.7; }
      }
    }
  }
  m.center = [0.5 * (minX + maxX), 0.5 * (minY + maxY), 0];
  m.radius = 0.5 * Math.hypot(maxX - minX, maxY - minY) + 2 * s;
  m.teeth = buildTeeth(g, m.teeth);
  buildPouch(g, m.pouch);
  setAnchors(g, m);
  return m;
}

/**
 * Centre line of the upper dental arch, (distance behind the incisal edge,
 * half width) in canonical cm: about 3.5 cm between the canine tips and 5 cm
 * between the centres of the first molars in adult men (dental cast studies).
 */
const ARCH: [number, number][] = [[0, 0], [0.15, 0.45], [0.45, 1.0], [0.9, 1.45], [1.5, 1.75], [2.5, 2.05], [3.5, 2.3], [4.5, 2.45], [5.5, 2.5]];
/** Upper teeth from the midline back: mesio-distal width, crown height, depth (canonical cm). */
const TEETH: [number, number, number][] = [[0.85, 1.05, 0.7], [0.66, 0.9, 0.6], [0.76, 1.0, 0.8], [0.7, 0.8, 0.9], [0.66, 0.75, 0.9], [1.0, 0.7, 1.1], [0.9, 0.65, 1.05]];

function archAt(arc: number): [number, number, number] {
  // walk the arch by arc length: (behind, half width, direction of the arch)
  let acc = 0;
  for (let k = 1; k < ARCH.length; k++) {
    const [d0, z0] = ARCH[k - 1], [d1, z1] = ARCH[k];
    const seg = Math.hypot(d1 - d0, z1 - z0);
    if (acc + seg >= arc || k === ARCH.length - 1) {
      const t = Math.min(1, (arc - acc) / seg);
      return [d0 + (d1 - d0) * t, z0 + (z1 - z0) * t, Math.atan2(z1 - z0, d1 - d0)];
    }
    acc += seg;
  }
  return [ARCH[ARCH.length - 1][0], ARCH[ARCH.length - 1][1], 0];
}

function buildTeeth(g: TractGeometry, out: Tooth[]): Tooth[] {
  const s = g.scale;
  const lm = g.landmarks;
  const [ux, uy] = lm.upperEdge, [lx, ly] = lm.lowerEdge, [jx, jy] = lm.jawPivot;
  // the occlusal plane rises a little toward the molars
  const tilt = 0.12;
  const jaw = g.jawAngle;
  const res: Tooth[] = out;
  res.length = 0;
  for (const lower of [false, true]) {
    let arc = 0;
    for (const [w0, h0, d0] of TEETH) {
      const w = w0 * s * (lower ? 0.9 : 1), hh = h0 * s, dd = d0 * s;
      const [back, half, dir] = archAt((arc + 0.5 * w0) / 1);
      arc += w0;
      for (const side of [-1, 1]) {
        let x: number, y: number;
        if (!lower) {
          x = ux - back * s; y = uy + tilt * back * s + 0.5 * hh;
        } else {
          // the lower arch closes under the upper one and turns with the jaw
          const x0 = lx - back * s + 0.1 * s, y0 = ly + tilt * back * s - 0.5 * hh;
          const c = Math.cos(jaw), sn = Math.sin(jaw);
          // lx, ly already follow the jaw: turn the run along the arch only
          const ddx = x0 - lx, ddy = y0 - ly;
          x = lx + ddx * c - ddy * sn; y = ly + ddx * sn + ddy * c;
          void jx; void jy;
        }
        res.push({ x, y, z: side * (half + 0.05) * s, w, h: hh, d: dd, yaw: side * dir, tilt: tilt + (lower ? jaw : 0), lower });
      }
    }
  }
  return res;
}

function buildPouch(g: TractGeometry, p: Pouch): void {
  const pf = g.piriform;
  const s = g.scale;
  // the grid line where the fossae open, and the direction down the larynx
  let i = 0;
  while (i < g.lastRay && g.s[i] < pf.at) i++;
  const h = g.hits[i], r = g.rays[i];
  const tm = 0.5 * (h.tLow + h.tUp);
  p.x = r.sx + r.nx * (r.t0 + tm);
  p.y = r.sy + r.ny * (r.t0 + tm);
  const j = Math.max(0, i - 3);
  const hj = g.hits[j], rj = g.rays[j];
  const tj = 0.5 * (hj.tLow + hj.tUp);
  const vx = rj.sx + rj.nx * (rj.t0 + tj) - p.x, vy = rj.sy + rj.ny * (rj.t0 + tj) - p.y;
  const n = Math.hypot(vx, vy) || 1;
  p.dx = vx / n; p.dy = vy / n;
  p.length = pf.full;
  p.air = pf.length;
  // one pouch: half of the area of both
  p.rOpen = Math.sqrt(pf.areaOpen / 2 / Math.PI);
  p.rBottom = Math.sqrt(pf.areaBottom / 2 / Math.PI);
  p.z = LARYNX_HALF * s + p.rOpen;
}

function setAnchors(g: TractGeometry, m: TractMeshes): void {
  const a = m.anchors;
  const s = g.scale;
  const mid = (i: number): [number, number, number] => {
    const h = g.hits[i], r = g.rays[i];
    const t = 0.5 * (h.tLow + h.tUp);
    return [r.sx + r.nx * (r.t0 + t), r.sy + r.ny * (r.t0 + t), 0];
  };
  const findRay = (pred: (h: RayHit) => boolean): number => {
    for (let i = 0; i <= g.lastRay; i++) if (pred(g.hits[i])) return i;
    return -1;
  };
  a.tongue = [g.tongueCenter[0], g.tongueCenter[1], 0];
  const pal = findRay((h) => h.upId === 'palate');
  a.palate = pal >= 0 ? (() => { const p = mid(pal + 4 <= g.lastRay ? pal + 4 : pal); const r = g.rays[pal]; return [p[0] + r.nx * 0.8 * s, p[1] + r.ny * 0.8 * s, 0] as [number, number, number]; })() : null;
  const vel = findRay((h) => h.upId === 'velum');
  a.velum = vel >= 0 ? mid(vel) : null;
  a.teeth = [g.landmarks.upperEdge[0], g.landmarks.upperEdge[1], 1.2 * s];
  a.lips = [g.landmarks.lipFront[0], g.landmarks.lipFront[1], 0];
  a.larynx = mid(Math.min(2, g.lastRay));
  const ph = findRay((h) => h.rule === 'pharynx');
  a.pharynx = ph >= 0 ? mid(Math.min(g.lastRay, ph + 3)) : null;
  const p = m.pouch;
  a.piriform = p.air > 0.05 ? [p.x + p.dx * 0.5 * p.length, p.y + p.dy * 0.5 * p.length, p.z] : null;
  // a lateral: the channels beside the contact and the cavity behind it
  let contact = -1, supra = -1;
  for (let i = 0; i <= g.lastRay; i++) {
    const h = g.hits[i];
    if (h.d <= 0 && h.lateral > 0 && contact < 0) contact = i;
    if (h.supra > 0 && supra < 0) supra = i;
  }
  if (contact >= 0) {
    const c = mid(contact);
    const h = g.hits[contact];
    a.channels = [c[0], c[1], (h.lensHalf - 0.3) * s];
  } else a.channels = null;
  a.supra = supra >= 0 && contact > supra ? mid(Math.round(0.5 * (supra + contact))) : null;
}
