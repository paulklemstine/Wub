// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
import { AREA_CLOSED } from '../constants';
import { TractGeometry } from './geometry';
import { PIRIFORM_SECTIONS, SUPRA_SECTIONS } from './section3d';

/** Number of equal-length sections of the oral tract (glottis to lips). */
export const N_SECTIONS = 44;
/** Largest perimeter relative to a round duct (a slit about 50 times wider than high). */
const SHAPE_MAX = 8;

/**
 * A tube description shared by the frequency-domain reference model and the
 * time-domain waveguide: N equal sections from the glottis (index 0) to the
 * lips (index N-1).
 */
export interface AreaFunction {
  areas: Float64Array;
  length: number;
  /** length of one section, cm */
  dx: number;
  /** section containing the incisors (obstacle position) */
  teethIndex: number;
  /** section where the lips start */
  lipsIndex: number;
  /** perimeter of each section relative to a circle of the same area, for the
   *  viscous and thermal losses (default 1: a round duct) */
  shape?: Float64Array;
  /** yielding (soft-tissue) perimeter of each section relative to the same
   *  circle (default 1: all soft; 0 = bony walls) */
  soft?: Float64Array;
  /** the piriform fossae, a blind side branch at a junction */
  piriform?: BlindBranch;
  /** the supralingual cavity of a lateral, a blind side branch (all sections
   *  closed when there is none) */
  supralingual?: BlindBranch;
}

/**
 * A blind side branch joining the tract at junction `at` (between sections
 * at and at + 1), in sections of the tract's length from the opening to the
 * closed end; the sections past the end are closed. Both piriform fossae are
 * one such branch (they are alike in most speakers); so is the supralingual
 * cavity of a lateral, from where the flow splits into the lateral channels
 * to the tip contact.
 */
export interface BlindBranch {
  at: number;
  areas: Float64Array;
}

export function uniformTube(length = 17.5, area = 3.0, n = N_SECTIONS): AreaFunction {
  const areas = new Float64Array(n).fill(area);
  return { areas, length, dx: length / n, teethIndex: n - 3, lipsIndex: n - 2 };
}

/**
 * Resamples the fine grid-line area function of the geometry onto N equal
 * sections. Within a section the acoustic mass is preserved (harmonic mean of
 * the area), and any closure inside a section closes the whole section, so
 * that narrow constrictions keep their acoustic effect.
 */
export function sectionsFromGeometry(g: TractGeometry, n = N_SECTIONS, out?: AreaFunction): AreaFunction {
  const L = Math.max(g.length, 4);
  const dx = L / n;
  const areas = out?.areas && out.areas.length === n ? out.areas : new Float64Array(n);
  const s = g.s;
  const last = g.lastRay;
  const shape = out?.shape && out.shape.length === n ? out.shape : new Float64Array(n);
  const soft = out?.soft && out.soft.length === n ? out.soft : new Float64Array(n);
  // the path of the air: without the supralingual cavity, a side branch
  const A = (i: number) => g.hits[i].area - g.hits[i].supra;
  const P = (i: number) => g.hits[i].perimeter - g.hits[i].supraPerimeter;
  const W = (i: number) => g.hits[i].soft - g.hits[i].supraSoft;
  const circle = (a: number) => 2 * Math.sqrt(Math.PI * a);
  let j = 0; // fine index pointer
  const SUB = 6;
  for (let k = 0; k < n; k++) {
    const a = k * dx, b = (k + 1) * dx;
    let inv = 0, visc = 0, wet = 0, yielding = 0;
    let minA = Infinity, minI = 0;
    for (let q = 0; q < SUB; q++) {
      const x = a + ((q + 0.5) / SUB) * (b - a);
      while (j < last - 1 && s[j + 1] < x) j++;
      const s0 = s[j], s1 = s[j + 1];
      const t = s1 > s0 ? Math.min(1, Math.max(0, (x - s0) / (s1 - s0))) : 0;
      const area = Math.max(A(j) + (A(j + 1) - A(j)) * t, AREA_CLOSED);
      const per = P(j) + (P(j + 1) - P(j)) * t;
      inv += 1 / area;
      // the viscous resistance of the pieces adds up: perimeter / area^2
      visc += (per > 0 ? per : circle(area)) / (area * area);
      wet += per;
      yielding += W(j) + (W(j + 1) - W(j)) * t;
    }
    // closures of single grid lines inside the section
    for (let i = 0; i <= last; i++) {
      if (s[i] >= a - 1e-9 && s[i] <= b + 1e-9 && A(i) < minA) { minA = A(i); minI = i; }
    }
    let area = SUB / inv;
    let ratio = Math.max(1, ((visc / SUB) * area * area) / circle(area));
    // share of the wall that yields
    let yields = wet > 0 ? Math.min(1, yielding / wet) : 1;
    if (minA < 0.004) {
      // a closure: its losses are those of a closed section, whatever its shape
      area = Math.max(AREA_CLOSED, minA);
      ratio = 1;
      yields = P(minI) > 0 ? Math.min(1, W(minI) / P(minI)) : 1;
    }
    areas[k] = area;
    shape[k] = Math.min(SHAPE_MAX, ratio);
    soft[k] = shape[k] * yields;
  }
  const teethIndex = Math.min(n - 1, Math.max(0, Math.floor(g.teethPos / dx)));
  const lipsIndex = Math.min(n - 1, Math.max(0, Math.floor(g.lipsPos / dx)));
  const res = out ?? { areas, length: L, dx, teethIndex, lipsIndex };
  res.areas = areas; res.length = L; res.dx = dx; res.teethIndex = teethIndex; res.lipsIndex = lipsIndex;
  res.shape = shape; res.soft = soft;
  res.piriform = piriformBranch(g, dx, res.piriform);
  res.supralingual = supralingualBranch(g, dx, res.supralingual);
  return res;
}

/** Narrowest supralingual cavity counted as a branch (cm^2). */
const SUPRA_MIN = 0.02;

/**
 * The supralingual cavity as sections of length dx from where the flow
 * splits into the lateral channels (its opening) to the tip contact (its
 * closed end), the last section partly closed so that the depth is not
 * rounded to whole sections.
 */
export function supralingualBranch(g: TractGeometry, dx: number, out?: BlindBranch): BlindBranch {
  const areas = out?.areas.length === SUPRA_SECTIONS ? out.areas : new Float64Array(SUPRA_SECTIONS);
  const h = g.hits, s = g.s, last = g.lastRay;
  // the longest run of grid lines with a cavity
  let i0 = -1, i1 = -1;
  for (let i = 0; i <= last; i++) {
    if (h[i].supra <= SUPRA_MIN) continue;
    let j = i;
    while (j < last && h[j + 1].supra > SUPRA_MIN) j++;
    if (i0 < 0 || s[j] - s[i] > s[i1] - s[i0]) { i0 = i; i1 = j; }
    i = j;
  }
  areas.fill(AREA_CLOSED);
  if (i0 < 0) {
    const res = out ?? { at: -1, areas };
    res.areas = areas;
    return res;
  }
  // the split and the contact lie halfway to the neighbouring grid lines
  const open = i0 > 0 ? 0.5 * (s[i0 - 1] + s[i0]) : s[i0];
  const end = i1 < last ? 0.5 * (s[i1] + s[i1 + 1]) : s[i1];
  const e = (end - open) / dx;
  let j = i0;
  for (let k = 0; k < SUPRA_SECTIONS; k++) {
    const fill = Math.min(1, Math.max(0, e - k));
    if (fill <= 0) break;
    const x = Math.min(end, open + (k + 0.5) * dx);
    while (j < i1 && s[j + 1] < x) j++;
    const t = j < i1 && s[j + 1] > s[j] ? Math.min(1, Math.max(0, (x - s[j]) / (s[j + 1] - s[j]))) : 0;
    const a = j < i1 ? h[j].supra + (h[j + 1].supra - h[j].supra) * t : h[j].supra;
    areas[k] = Math.max(AREA_CLOSED, a * fill);
  }
  // the opening moves to another junction only when it is well past it
  const want = open / dx - 1;
  const at = Math.max(1, out && out.at >= 0 && Math.abs(want - out.at) < 0.75 ? out.at : Math.round(want));
  const res = out ?? { at, areas };
  res.at = at; res.areas = areas;
  return res;
}

/**
 * The fossae as sections of length dx: a funnel from the opening to the
 * bottom, the last section partly closed so that the depth is not rounded to
 * whole sections, and the sections below the air closed.
 */
export function piriformBranch(g: TractGeometry, dx: number, out?: BlindBranch): BlindBranch {
  const pf = g.piriform;
  const areas = out?.areas.length === PIRIFORM_SECTIONS ? out.areas : new Float64Array(PIRIFORM_SECTIONS);
  const e = pf.length / dx;
  for (let k = 0; k < PIRIFORM_SECTIONS; k++) {
    const x = Math.min(1, ((k + 0.5) * dx) / Math.max(pf.full, 1e-6));
    const a = pf.areaOpen + (pf.areaBottom - pf.areaOpen) * x;
    const fill = Math.min(1, Math.max(0, e - k));
    areas[k] = Math.max(AREA_CLOSED, a * fill);
  }
  // the opening moves to another junction only when it is well past it
  const want = pf.at / dx - 1;
  const at = Math.max(1, out && Math.abs(want - out.at) < 0.75 ? out.at : Math.round(want));
  const res = out ?? { at, areas };
  res.at = at; res.areas = areas;
  return res;
}

/**
 * Resamples a drawn or measured tube profile onto n sections. Each source
 * value is the area at the centre of its own cell, so a profile that already
 * has n values comes through unchanged.
 */
export function resampleTube(src: ArrayLike<number>, n: number, out: Float64Array, floor: number): Float64Array {
  const m = src.length;
  for (let i = 0; i < n; i++) {
    const x = Math.min(m - 1, Math.max(0, ((i + 0.5) / n) * m - 0.5));
    const j = Math.min(m - 2, Math.floor(x)), t = x - j;
    out[i] = Math.max(floor, src[j] * (1 - t) + src[j + 1] * t);
  }
  return out;
}

/** Index and value of the minimum area between two sections (inclusive). */
export function minArea(af: AreaFunction, from: number, to: number): { index: number; area: number } {
  let idx = from, v = Infinity;
  for (let i = from; i <= to; i++) if (af.areas[i] < v) { v = af.areas[i]; idx = i; }
  return { index: idx, area: v };
}
