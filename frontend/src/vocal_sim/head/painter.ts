// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).
/**
 * Paints the illustrated head on a canvas from a `HeadScene`: the backdrop in
 * its bands, then the soft parts drawn from the acoustic geometry. The air
 * itself (coloured by the sound field) is drawn afterwards by the head view,
 * so nothing here can hide it.
 */
import { HeadScene, Mat, Mesh, mul, PIXELS } from './scene';
import { framePixel } from './frame';
import { fieldColor } from '../canvas_colors';

/** Silent air, as the head view draws it. */
const AIR = '#05070c';
const AIR_TINT = fieldColor(0.12, 0.28);

export interface Backdrop {
  img: HTMLImageElement;
  /** colours picked from the painting, so the drawn parts match it */
  tone: { tissue: string; skin: string; mucosa: string; tongue: string };
  /** a tile of mucosa in the painting's colour, made when first needed */
  mucosaTile?: HTMLCanvasElement;
  /** the same flecks on nothing, to lay over a colour of its own */
  flecksTile?: HTMLCanvasElement;
}

let backdrop: Backdrop | null = null;
let failed = false;
let started = false;

/** The painted backdrop, once loaded (null until then, or if it cannot load). */
export function getBackdrop(url = 'anatomy/head.jpg'): Backdrop | null {
  if (backdrop || failed) return backdrop;
  if (!started && typeof Image !== 'undefined') {
    started = true;
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => { backdrop = { img, tone: sampleTones(img) }; };
    img.onerror = () => { failed = true; };
    img.src = url;
  }
  return null;
}

/** Average colour of a few pixels around canonical points of the painting. */
function sampleTones(img: HTMLImageElement): Backdrop['tone'] {
  const fallback = { tissue: '#b9606b', skin: '#e3a98f', mucosa: '#c9606e', tongue: '#d9727f' };
  try {
    const cv = document.createElement('canvas');
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    const c = cv.getContext('2d', { willReadFrequently: true });
    if (!c) return fallback;
    c.drawImage(img, 0, 0);
    const avg = (x: number, y: number) => {
      const [px, py] = framePixel(x, y);
      const d = c.getImageData(Math.round(px) - 4, Math.round(py) - 4, 9, 9).data;
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
      const n = d.length / 4;
      return `rgb(${(r / n) | 0},${(g / n) | 0},${(b / n) | 0})`;
    };
    return { tissue: avg(3.2, 4.0), skin: avg(8.6, 10.78), mucosa: avg(7.75, 9.2), tongue: avg(3.5, 7.6) };
  } catch {
    return fallback;
  }
}

const setMat = (ctx: CanvasRenderingContext2D, m: Mat) => ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);

/** A tile of wet mucosa, seamless: soft lighter and darker flecks on the painting's colour,
 *  or on nothing without one. */
function mucosaTile(base?: string): HTMLCanvasElement {
  const N = 128;
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const g = c.getContext('2d')!;
  if (base) {
    g.fillStyle = base;
    g.fillRect(0, 0, N, N);
  }
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 260; i++) {
    const x = rnd() * N, y = rnd() * N, r = 1.5 + rnd() * 4.5;
    g.fillStyle = rnd() < 0.5 ? `rgba(255,225,228,${(0.05 + rnd() * 0.08).toFixed(3)})` : `rgba(110,30,40,${(0.04 + rnd() * 0.07).toFixed(3)})`;
    for (const dx of [-N, 0, N]) for (const dy of [-N, 0, N]) {
      g.beginPath();
      g.arc(x + dx, y + dy, r, 0, 2 * Math.PI);
      g.fill();
    }
  }
  return c;
}

const mucosaPatterns = new WeakMap<CanvasRenderingContext2D, { mucosa?: CanvasPattern | string; flecks?: CanvasPattern | string }>();

/** The mucosa as a fill for this context, a tile every 2.5 cm or so (world units);
 *  with `flecks`, only its flecks. */
function mucosa(ctx: CanvasRenderingContext2D, bd: Backdrop, flecks = false): CanvasPattern | string {
  let all = mucosaPatterns.get(ctx);
  if (!all) mucosaPatterns.set(ctx, (all = {}));
  const key = flecks ? 'flecks' : 'mucosa';
  let p = all[key];
  if (!p) {
    const tile = flecks ? (bd.flecksTile ??= mucosaTile()) : (bd.mucosaTile ??= mucosaTile(bd.tone.mucosa));
    const pat = ctx.createPattern(tile, 'repeat');
    if (pat) pat.setTransform(new DOMMatrix().scale(0.02));
    p = all[key] = pat ?? (flecks ? 'rgba(0,0,0,0)' : bd.tone.mucosa);
  }
  return p;
}

function poly(ctx: CanvasRenderingContext2D, p: number[]): void {
  ctx.moveTo(p[0], p[1]);
  for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
  ctx.closePath();
}

/** A closed outline with its corners rounded: curves from the middle of each edge to the
 *  middle of the next, bent by the point between them. */
function smoothPoly(ctx: CanvasRenderingContext2D, p: number[]): void {
  const n = p.length / 2;
  const mid = (i: number) => {
    const j = (i + 1) % n;
    return [(p[2 * i] + p[2 * j]) / 2, (p[2 * i + 1] + p[2 * j + 1]) / 2];
  };
  const m0 = mid(n - 1);
  ctx.moveTo(m0[0], m0[1]);
  for (let i = 0; i < n; i++) {
    const m = mid(i);
    ctx.quadraticCurveTo(p[2 * i], p[2 * i + 1], m[0], m[1]);
  }
  ctx.closePath();
}

/** Clip to everything outside the polygons (coordinates of the current transform). */
function clipOut(ctx: CanvasRenderingContext2D, polys: number[][], R = 200): void {
  for (const p of polys) {
    ctx.beginPath();
    ctx.rect(-R, -R, 2 * R, 2 * R);
    poly(ctx, p);
    ctx.clip('evenodd');
  }
}

/**
 * The backdrop: a base of tissue under the whole head, then each band of the
 * painting in its place, and the bend under the jaw triangle by triangle.
 * `world` maps world centimetres to canvas pixels.
 */
export function paintBackdrop(ctx: CanvasRenderingContext2D, sc: HeadScene, world: Mat, bd: Backdrop): void {
  ctx.save();
  setMat(ctx, world);
  clipOut(ctx, sc.hole);
  ctx.beginPath();
  poly(ctx, sc.outline);
  ctx.fillStyle = bd.tone.tissue;
  ctx.fill();
  // flecked like the mucosa, where a wide jaw opens a gap between the bands
  ctx.fillStyle = mucosa(ctx, bd, true);
  ctx.fill();
  ctx.restore();
  ctx.imageSmoothingQuality = 'high';
  for (const b of sc.bands) {
    ctx.save();
    const m = mul(world, b.mat);
    setMat(ctx, m);
    for (const c of b.clip) { ctx.beginPath(); poly(ctx, c); ctx.clip(); }
    clipOut(ctx, b.exclude);
    setMat(ctx, world);
    clipOut(ctx, sc.hole);
    if (b.world) { ctx.beginPath(); poly(ctx, b.world); ctx.clip(); }
    setMat(ctx, mul(m, PIXELS));
    ctx.drawImage(bd.img, 0, 0);
    ctx.restore();
  }
  ctx.save();
  setMat(ctx, world);
  ctx.beginPath();
  poly(ctx, sc.outline);
  ctx.clip();
  clipOut(ctx, sc.hole);
  for (const t of sc.bend) paintTriangle(ctx, world, t.from, t.to, bd.img);
  ctx.restore();
}

/**
 * Fades the bottom of the painted neck out, so the head does not end on a cut.
 * Use on a layer of its own: it erases what is under it.
 */
export function fadeNeck(ctx: CanvasRenderingContext2D, sc: HeadScene, world: Mat): void {
  ctx.save();
  setMat(ctx, world);
  const y0 = sc.neckEnd, y1 = sc.neckEnd + 1.6 * sc.s;
  const g = ctx.createLinearGradient(0, y0, 0, y1);
  g.addColorStop(0, 'rgba(0,0,0,1)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.globalCompositeOperation = 'destination-out';
  ctx.fillStyle = g;
  ctx.fillRect(-100, y0 - 100, 200, y1 - y0 + 100);
  ctx.restore();
}

/** The air of the nose and above the velum, lit (0..1) when the velum is open and it carries sound. */
export function paintGlow(ctx: CanvasRenderingContext2D, sc: HeadScene, world: Mat, glow: number): void {
  if (glow <= 0.01) return;
  ctx.save();
  setMat(ctx, world);
  ctx.beginPath();
  poly(ctx, sc.nasal);
  ctx.fillStyle = `rgba(42,212,189,${(0.28 * glow).toFixed(3)})`;
  ctx.fill();
  ctx.beginPath();
  poly(ctx, sc.port);
  ctx.fillStyle = `rgba(42,212,189,${(0.4 * glow).toFixed(3)})`;
  ctx.fill();
  ctx.restore();
}

/**
 * What the geometry draws over the backdrop, before the air: the space behind
 * the lips, the air above the velum, the patches where the painting had air,
 * and the moving solids carrying their own painting.
 */
export function paintSoftParts(ctx: CanvasRenderingContext2D, sc: HeadScene, world: Mat, bd: Backdrop): void {
  const s = sc.s;
  ctx.save();
  setMat(ctx, world);

  // mucosa behind the lips, under the tongue's tip, and wherever the painting's air is tissue now
  ctx.fillStyle = mucosa(ctx, bd);
  for (const q of [...sc.inner, sc.vestibule, ...sc.patch]) {
    ctx.beginPath();
    poly(ctx, q);
    ctx.fill();
  }

  // the air above the velum: air like the rest, with a wet rim along its vault and wall
  ctx.beginPath();
  poly(ctx, sc.port);
  ctx.fillStyle = AIR;
  ctx.fill();
  ctx.fillStyle = AIR_TINT;
  ctx.fill();
  ctx.save();
  clipOut(ctx, [sc.port]);
  ctx.beginPath();
  const from = 2 * sc.portVelum - 2;
  ctx.moveTo(sc.port[from], sc.port[from + 1]);
  for (let i = from + 2; i < sc.port.length; i += 2) ctx.lineTo(sc.port[i], sc.port[i + 1]);
  ctx.lineWidth = 0.07 * s;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(255,205,212,0.32)';
  ctx.stroke();
  ctx.restore();
  ctx.restore();

  // the velum before the palate, which hides its root as the bone hides where it starts
  paintVelum(ctx, sc.velum, world, s, mucosa(ctx, bd, true));

  for (const sp of sc.sprites) {
    const tooth = sp.id === 'upperIncisor' || sp.id === 'lowerIncisor';
    const texture = () => {
      setMat(ctx, mul(mul(world, sp.tex), PIXELS));
      ctx.drawImage(bd.img, 0, 0);
    };
    if (sp.area) {
      // a lip with the face in front of it: skin under the painting, wherever it does not reach;
      // the rest of the lip lies where the jaw's own painting already has it
      ctx.save();
      setMat(ctx, world);
      ctx.beginPath();
      poly(ctx, sp.area);
      ctx.fillStyle = bd.tone.skin;
      ctx.fill();
      ctx.clip();
      if (sp.areaMesh) paintMesh(ctx, world, sp.areaMesh, bd.img);
      else texture();
      ctx.restore();
      continue;
    }
    ctx.save();
    setMat(ctx, world);
    ctx.beginPath();
    poly(ctx, sp.poly);
    ctx.fillStyle = tooth ? '#efe7d6' : sp.id === 'tongue' ? bd.tone.tongue : sp.id === 'palate' ? '#e9dcc0' : bd.tone.mucosa;
    ctx.fill();
    ctx.clip();
    if (sp.mesh) {
      paintMesh(ctx, world, sp.mesh, bd.img);
    } else {
      ctx.beginPath();
      poly(ctx, sp.painted);
      ctx.clip();
      texture();
    }
    ctx.restore();
    if (tooth) {
      ctx.save();
      setMat(ctx, world);
      ctx.beginPath();
      poly(ctx, sp.poly);
      ctx.lineWidth = 0.025 * s;
      ctx.strokeStyle = 'rgba(160,140,110,0.7)';
      ctx.stroke();
      ctx.restore();
    }
  }
}

/** The affine map taking the triangle p0 p1 p2 onto q0 q1 q2, or null if p is degenerate. */
function triangleMap(p: number[], q: number[]): Mat | null {
  const a = p[2] - p[0], b = p[3] - p[1], c = p[4] - p[0], d = p[5] - p[1];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-9) return null;
  const A = q[2] - q[0], B = q[3] - q[1], C = q[4] - q[0], D = q[5] - q[1];
  // [m0 m2; m1 m3] = [A C; B D] * inverse([a c; b d])
  const m0 = (A * d - C * b) / det, m2 = (C * a - A * c) / det;
  const m1 = (B * d - D * b) / det, m3 = (D * a - B * c) / det;
  return [m0, m1, m2, m3, q[0] - m0 * p[0] - m2 * p[1], q[1] - m1 * p[0] - m3 * p[1]];
}

/**
 * A solid that changes shape, its painting stretched onto it: a fan of triangles from
 * its centre, each mapping the painted triangle at rest onto the same triangle now.
 * Each triangle is clipped a little larger than itself so no seam shows between them.
 */
function paintMesh(ctx: CanvasRenderingContext2D, world: Mat, mesh: Mesh, img: HTMLImageElement): void {
  const { from, to, cFrom, cTo } = mesh;
  const n = from.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const p = [cFrom[0], cFrom[1], from[2 * i], from[2 * i + 1], from[2 * j], from[2 * j + 1]];
    const q = [cTo[0], cTo[1], to[2 * i], to[2 * i + 1], to[2 * j], to[2 * j + 1]];
    paintTriangle(ctx, world, p, q, img);
  }
}

/** The painting's triangle p (canonical) drawn onto the triangle q (world), clipped a
 *  pixel larger than q so that no seam shows between neighbours. */
function paintTriangle(ctx: CanvasRenderingContext2D, world: Mat, p: number[], q: number[], img: HTMLImageElement): void {
  const m = triangleMap(p, q);
  if (!m) return;
  const grow = 0.9 / Math.hypot(world[0], world[1]); // about a pixel, in world units
  const gx = (q[0] + q[2] + q[4]) / 3, gy = (q[1] + q[3] + q[5]) / 3;
  ctx.save();
  setMat(ctx, world);
  ctx.beginPath();
  for (let k = 0; k < 6; k += 2) {
    const dx = q[k] - gx, dy = q[k + 1] - gy, l = Math.hypot(dx, dy) || 1;
    const x = q[k] + (dx / l) * grow, y = q[k + 1] + (dy / l) * grow;
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.clip();
  setMat(ctx, mul(mul(world, m), PIXELS));
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

/** The velum: soft palate muscle under a wet mucosa, lit from the mouth. Its outline has
 *  the oral side first (base to tip), then the nasal side back (tip to base). */
function paintVelum(ctx: CanvasRenderingContext2D, v: number[], world: Mat, s: number, flecks: CanvasPattern | string): void {
  const half = v.length / 4; // points per side
  const oral = (i: number): [number, number] => [v[2 * i], v[2 * i + 1]];
  const nasal = (i: number): [number, number] => [v[2 * (2 * half - 1 - i)], v[2 * (2 * half - 1 - i) + 1]];
  const line = (f: (i: number) => [number, number], w = 0) => {
    ctx.beginPath();
    for (let i = 0; i < half; i++) {
      const a = f(i), b = nasal(i);
      const x = a[0] + w * (b[0] - a[0]), y = a[1] + w * (b[1] - a[1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
  };
  ctx.save();
  setMat(ctx, world);
  // lighter on the side of the mouth, darker towards the nose
  const m = half >> 1;
  const g = ctx.createLinearGradient(oral(m)[0], oral(m)[1], nasal(m)[0], nasal(m)[1]);
  g.addColorStop(0, '#e48e9b');
  g.addColorStop(0.5, '#c45d6d');
  g.addColorStop(1, '#8b3646');
  // the body, and the uvula's rounded end
  let turn = 0;
  for (let i = 0, n = v.length / 2; i < n; i++) {
    const j = (i + 1) % n;
    turn += v[2 * i] * v[2 * j + 1] - v[2 * j] * v[2 * i + 1];
  }
  const body = () => {
    ctx.beginPath();
    smoothPoly(ctx, v);
    const t0 = oral(half - 1), t1 = nasal(half - 1);
    ctx.moveTo((t0[0] + t1[0]) / 2 + Math.hypot(t1[0] - t0[0], t1[1] - t0[1]) / 2, (t0[1] + t1[1]) / 2);
    // the same way round as the outline, so the two make one shape
    ctx.arc((t0[0] + t1[0]) / 2, (t0[1] + t1[1]) / 2, Math.hypot(t1[0] - t0[0], t1[1] - t0[1]) / 2, 0, 2 * Math.PI, turn < 0);
  };
  ctx.fillStyle = g;
  body();
  ctx.fill('nonzero');
  ctx.fillStyle = flecks;
  ctx.fill('nonzero');
  // muscle fibres along it
  ctx.save();
  body();
  ctx.clip('nonzero');
  ctx.lineWidth = 0.02 * s;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(255,212,220,0.22)';
  for (const w of [0.28, 0.5, 0.72]) {
    line(oral, w);
    ctx.stroke();
  }
  ctx.restore();
  // a darker edge towards the nose, a wet light on the side of the mouth
  line(nasal);
  ctx.lineWidth = 0.03 * s;
  ctx.strokeStyle = 'rgba(80,20,30,0.5)';
  ctx.stroke();
  line(oral);
  ctx.lineWidth = 0.05 * s;
  ctx.strokeStyle = 'rgba(255,214,220,0.45)';
  ctx.stroke();
  ctx.restore();
}

/** A thin wet rim along the air's walls, on the tissue side only. */
export function paintRim(ctx: CanvasRenderingContext2D, sc: HeadScene, world: Mat): void {
  ctx.save();
  setMat(ctx, world);
  // not across the opening between the lips, where the grid's steps cut through the air,
  // nor along the air above the velum
  clipOut(ctx, sc.mouth.length ? [sc.lumen, sc.mouth, sc.port] : [sc.lumen, sc.port]);
  ctx.lineWidth = 0.07 * sc.s;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(255,205,212,0.32)';
  // where the two walls touch (closed lips, a stop) there is no air to have a rim
  const shut = (i: number) => Math.hypot(sc.upper[i] - sc.lower[i], sc.upper[i + 1] - sc.lower[i + 1]) < 0.03 * sc.s;
  for (const e of [sc.lower, sc.upper]) {
    ctx.beginPath();
    for (let i = 0; i < e.length; i += 2) {
      if (i === 0 || (shut(i - 2) && shut(i))) ctx.moveTo(e[i], e[i + 1]);
      else ctx.lineTo(e[i], e[i + 1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}
