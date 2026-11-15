// Exact Side-View Mid-Sagittal Head Renderer ported from MithrilMan/voice-workshop
// (Apache-2.0, Copyright 2026 Fabio Angela) to vanilla TypeScript for Wub KG-505.

import { VocalFieldState } from "./articulation_bridge";
import { C, fieldColor, FieldMode } from "./canvas_colors";
import { CHIN, chinWeight, HEAD_BACK, HEAD_FRONT_UP, NASAL_CENTRE, NASAL_HALF } from "./head/outline";
import { fadeNeck, getBackdrop, paintBackdrop, paintGlow, paintRim, paintSoftParts } from "./head/painter";
import { headScene, HeadScene, Mat, mul } from "./head/scene";
import { N_SECTIONS } from "./tract/areaFunction";
import { canonToWorld, Solid, TractGeometry } from "./tract/geometry";
import { Anatomy, Articulators, Teeth } from "./tract/types";

interface Particle {
  s: number;
  off: number;
  nose: boolean;
  age: number;
}

function drawLabelPill(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  bg = "rgba(11,14,20,0.75)"
): void {
  ctx.font = "600 11px system-ui, sans-serif";
  const m = ctx.measureText(text);
  const w = m.width + 10;
  const h = 17;
  const x0 = x - w / 2;
  ctx.fillStyle = bg;
  ctx.beginPath();
  ctx.roundRect(x0, y - h / 2, w, h, 5);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y + 0.5);
}

function rayPoints(geom: TractGeometry, i: number): [number, number, number, number] {
  const r = geom.rays[i];
  const h = geom.hits[i];
  return [
    r.sx + r.nx * (r.t0 + h.tLow),
    r.sy + r.ny * (r.t0 + h.tLow),
    r.sx + r.nx * (r.t0 + h.tUp),
    r.sy + r.ny * (r.t0 + h.tUp),
  ];
}

function drawChannel(
  ctx: CanvasRenderingContext2D,
  geom: TractGeometry,
  X: (x: number) => number,
  Y: (y: number) => number,
  mode: FieldMode,
  tel: VocalFieldState | null
): void {
  const n = geom.lastRay;
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const q = rayPoints(geom, i);
    if (i === 0) ctx.moveTo(X(q[0]), Y(q[1]));
    else ctx.lineTo(X(q[0]), Y(q[1]));
  }
  for (let i = n; i >= 0; i--) {
    const q = rayPoints(geom, i);
    ctx.lineTo(X(q[2]), Y(q[3]));
  }
  ctx.closePath();
  ctx.fillStyle = "#05070c";
  ctx.fill();

  const L = geom.length;
  for (let i = 0; i < n; i++) {
    const h = geom.hits[i];
    const h2 = geom.hits[i + 1];
    if (h.d <= 0 && h2.d <= 0) continue;
    const k = Math.max(0, Math.min(N_SECTIONS - 1, Math.floor((geom.s[i] / L) * N_SECTIONS)));
    let v = 0.12;
    let alpha = 0.28;
    if (tel && tel.ug > 2) {
      if (mode === "pressure") v = tel.pRms[k];
      else if (mode === "velocity") v = tel.uRms[k];
      else {
        const u = k <= tel.constrictionIndex ? tel.ug : tel.uMouth;
        const vel = u / Math.max(h.area, 0.02);
        v = Math.min(1, Math.log10(1 + vel / 30) / 2.3);
      }
      alpha = 0.85;
    }
    const a = rayPoints(geom, i);
    const b = rayPoints(geom, i + 1);
    ctx.beginPath();
    ctx.moveTo(X(a[0]), Y(a[1]));
    ctx.lineTo(X(a[2]), Y(a[3]));
    ctx.lineTo(X(b[2]), Y(b[3]));
    ctx.lineTo(X(b[0]), Y(b[1]));
    ctx.closePath();
    ctx.fillStyle = fieldColor(v, alpha);
    ctx.fill();
  }
}

function drawSolid(
  ctx: CanvasRenderingContext2D,
  sol: Solid,
  path: (pts: number[], close?: boolean) => void,
  X: (x: number) => number,
  s: number
): void {
  path(sol.pts);
  switch (sol.id) {
    case "tongue": {
      const g = ctx.createLinearGradient(X(sol.minX), 0, X(sol.maxX), 0);
      g.addColorStop(0, "#f08aa0");
      g.addColorStop(1, "#a8445e");
      ctx.fillStyle = g;
      break;
    }
    case "upperIncisor":
    case "lowerIncisor":
      ctx.fillStyle = C.tooth;
      break;
    case "palate":
      ctx.fillStyle = "#cdbfa6";
      break;
    case "velum":
      ctx.fillStyle = "#c4687d";
      break;
    case "upperLip":
    case "lowerLip":
      ctx.fillStyle = C.lip;
      break;
    case "wall": {
      const g = ctx.createLinearGradient(X(-2.4 * s), 0, X(-0.45 * s), 0);
      g.addColorStop(0, "rgba(120,60,80,0)");
      g.addColorStop(0.55, "rgba(150,72,94,0.35)");
      g.addColorStop(1, "#a24f66");
      ctx.fillStyle = g;
      break;
    }
    case "larynx": {
      const g = ctx.createLinearGradient(X(0.4 * s), 0, X(2.6 * s), 0);
      g.addColorStop(0, "#a24f66");
      g.addColorStop(1, "rgba(120,60,80,0)");
      ctx.fillStyle = g;
      break;
    }
    case "floor":
      ctx.fillStyle = "rgba(147,72,92,0.75)";
      break;
  }
  ctx.fill();
  if (sol.id === "upperIncisor" || sol.id === "lowerIncisor") {
    ctx.strokeStyle = "#b9b09b";
    ctx.lineWidth = 1;
    ctx.stroke();
  } else if (
    sol.id === "tongue" ||
    sol.id === "upperLip" ||
    sol.id === "lowerLip" ||
    sol.id === "velum"
  ) {
    ctx.strokeStyle = "rgba(255,220,225,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function paintFlat(
  ctx: CanvasRenderingContext2D,
  geom: TractGeometry,
  an: Anatomy,
  a: Articulators,
  X: (x: number) => number,
  Y: (y: number) => number,
  path: (pts: number[], close?: boolean) => void,
  k: number,
  s: number,
  tel: VocalFieldState | null,
  nc: number[],
  port: number,
  nasalFlow: number,
  mode: FieldMode
): void {
  const back = canonToWorld(HEAD_BACK, { ...an, pharynx: 1 }, { ...a, larynx: 0 });
  const frontUp = canonToWorld(HEAD_FRONT_UP, an, a);
  const chin = canonToWorld(CHIN, { ...an, pharynx: 1 }, { ...a, larynx: 0 }, chinWeight);
  const silhouette = [...back, ...frontUp, ...chin];

  const sk = ctx.createLinearGradient(X(-5), 0, X(10), 0);
  sk.addColorStop(0, "#1a1622");
  sk.addColorStop(0.6, "#2a2130");
  sk.addColorStop(1, "#3a2836");
  path(silhouette);
  ctx.fillStyle = sk;
  ctx.fill();
  ctx.strokeStyle = "rgba(247,112,138,0.35)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  const eye = canonToWorld([7.4, 15.9], an, a);
  ctx.strokeStyle = "rgba(231,235,243,0.25)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.ellipse(X(eye[0]), Y(eye[1]), 0.55 * k * s, 0.22 * k * s, 0, Math.PI * 0.05, Math.PI * 0.95);
  ctx.stroke();

  const ear = canonToWorld([-2.6, 13.4], an, a);
  ctx.beginPath();
  ctx.ellipse(X(ear[0]), Y(ear[1]), 0.9 * k * s, 1.5 * k * s, -0.15, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(216,210,194,0.07)";
  for (let i = 0; i < 6; i++) {
    const v = canonToWorld([-2.9, -2.6 + i * 1.9, -1.7, -1.1 + i * 1.9], an, a);
    ctx.beginPath();
    ctx.roundRect(X(v[0]), Y(v[3]), (v[2] - v[0]) * k, (v[3] - v[1]) * k, 4);
    ctx.fill();
  }

  const up: number[] = [];
  const lo: number[] = [];
  for (let i = 0; i < NASAL_HALF.length; i++) {
    const j = Math.min(NASAL_HALF.length - 2, i);
    const dx = nc[2 * j + 2] - nc[2 * j];
    const dy = nc[2 * j + 3] - nc[2 * j + 1];
    const L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L;
    const ny = dx / L;
    const hw = NASAL_HALF[i] * s;
    up.push(nc[2 * i] + nx * hw, nc[2 * i + 1] + ny * hw);
    lo.unshift(nc[2 * i + 1] - ny * hw);
    lo.unshift(nc[2 * i] - nx * hw);
  }
  path([...up, ...lo]);
  ctx.fillStyle = "#06080d";
  ctx.fill();
  if (port > 0.02) {
    ctx.fillStyle = `rgba(42,212,189,${0.1 + 0.45 * Math.min(1, port / 1.2) * (0.35 + 0.65 * nasalFlow)})`;
    ctx.fill();
  }
  ctx.strokeStyle = "rgba(200,107,124,0.5)";
  ctx.lineWidth = 1;
  ctx.stroke();

  drawChannel(ctx, geom, X, Y, mode, tel);

  for (const sol of geom.solids) {
    const soft = sol.id === "wall" || sol.id === "larynx" || sol.id === "floor";
    if (soft) {
      ctx.save();
      path(silhouette);
      ctx.clip();
    }
    drawSolid(ctx, sol, path, X, s);
    if (soft) ctx.restore();
  }
}

function drawParticles(
  ctx: CanvasRenderingContext2D,
  geom: TractGeometry,
  X: (x: number) => number,
  Y: (y: number) => number,
  s: number,
  tel: VocalFieldState | null,
  parts: Particle[],
  dt: number,
  nc: number[]
): void {
  const n = geom.lastRay;
  const L = geom.length;
  const ug = tel ? tel.ug : 0;
  const um = tel ? tel.uMouth : 0;
  const un = tel ? tel.uNose : 0;
  const flowing = ug > 5;
  const target = flowing ? 110 : 0;
  while (parts.length < target) {
    parts.push({ s: Math.random() * L, off: Math.random() - 0.5, nose: false, age: Math.random() });
  }
  if (!flowing) {
    parts.length = Math.max(0, parts.length - 4);
  }
  const ci = tel ? tel.constrictionIndex : 0;
  const rayAt = (sv: number) => {
    let lo2 = 0;
    let hi = n;
    while (hi - lo2 > 1) {
      const m = (lo2 + hi) >> 1;
      if (geom.s[m] < sv) lo2 = m;
      else hi = m;
    }
    return lo2;
  };
  ctx.fillStyle = "rgba(231,235,243,0.75)";
  for (const q of parts) {
    q.age += dt;
    if (q.nose) {
      const sp = 0.25 + 0.6 * Math.min(1, Math.abs(un) / 200);
      q.s += sp * dt;
      if (q.s >= 1 || Math.abs(un) < 3) {
        q.nose = false;
        q.s = 0;
        continue;
      }
      const m = NASAL_CENTRE.length / 2 - 1;
      const f = q.s * m;
      const j = Math.min(m - 1, Math.floor(f));
      const u = f - j;
      const x = nc[2 * j] + (nc[2 * j + 2] - nc[2 * j]) * u;
      const y = nc[2 * j + 1] + (nc[2 * j + 3] - nc[2 * j + 1]) * u;
      ctx.beginPath();
      ctx.arc(X(x), Y(y + q.off * 0.6 * s), 1.4, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }
    if (q.s > L) {
      const last = geom.hits[n];
      const prev = geom.hits[Math.max(0, n - 3)];
      const dx = last.mx - prev.mx;
      const dy = last.my - prev.my;
      const dl = Math.hypot(dx, dy) || 1;
      const ex = q.s - L;
      if (ex > 3.5 * s || um < 2) {
        q.s = 0;
        q.off = Math.random() - 0.5;
        continue;
      }
      const sp = 2 + 5 * Math.log10(1 + Math.abs(um) / Math.max(geom.lipArea, 0.05) / 50);
      q.s += sp * dt;
      const x = last.mx + (dx / dl) * ex - (dy / dl) * q.off * (0.3 + ex * 0.5);
      const y = last.my + (dy / dl) * ex + (dx / dl) * q.off * (0.3 + ex * 0.5);
      ctx.globalAlpha = Math.max(0, 1 - ex / (3.5 * s));
      ctx.beginPath();
      ctx.arc(X(x), Y(y), 1.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      continue;
    }
    const r = rayAt(q.s);
    const h = geom.hits[r];
    const sec = Math.floor((q.s / L) * N_SECTIONS);
    const u = sec <= ci ? ug : um;
    const vel = u / Math.max(h.area, 0.03);
    const sp = 1.5 + 5 * Math.log10(1 + vel / 50);
    q.s += sp * dt * s;
    if (
      tel &&
      tel.nasalPort > 0.05 &&
      Math.abs(un) > 3 &&
      Math.abs(q.s - geom.s[Math.round(0.43 * n)]) < 0.15 &&
      Math.random() < Math.abs(un) / (Math.abs(un) + Math.abs(um) + 1)
    ) {
      q.nose = true;
      q.s = 0;
      continue;
    }
    if (h.d <= 0.001) {
      q.s = 0;
      continue;
    }
    const pts = rayPoints(geom, r);
    const t = 0.5 + q.off * 0.8;
    const x = pts[0] + (pts[2] - pts[0]) * t;
    const y = pts[1] + (pts[3] - pts[1]) * t;
    ctx.beginPath();
    ctx.arc(X(x), Y(y), 1.3, 0, Math.PI * 2);
    ctx.fill();
  }
}

export class SideHeadView {
  private readonly layerCv: HTMLCanvasElement;
  private layerKey = "";
  private sceneCache: HeadScene | null = null;
  private readonly parts: Particle[] = [];
  private readonly backdropUrl: string;

  constructor(backdropUrl: string) {
    this.backdropUrl = backdropUrl;
    this.layerCv = document.createElement("canvas");
  }

  public renderRegion(
    ctx: CanvasRenderingContext2D,
    dpr: number,
    rx: number,
    ry: number,
    rw: number,
    rh: number,
    geom: TractGeometry,
    an: Anatomy,
    a: Articulators,
    teeth: Teeth,
    tel: VocalFieldState | null,
    mode: FieldMode,
    showLabels: boolean,
    timeSec: number,
    dt: number
  ): void {
    ctx.save();
    ctx.beginPath();
    ctx.rect(rx, ry, rw, rh);
    ctx.clip();

    const s = an.scale * 0.9;
    const X0 = -5.4;
    const X1 = 11.8;
    const Y0 = -3.4;
    let Y1 = 16.8;
    if (rh / rw > (Y1 - Y0) / (X1 - X0)) {
      Y1 = Math.min(19.9, Y0 + (rh / rw) * (X1 - X0));
    }
    const k = Math.min(rw / (X1 - X0), rh / (Y1 - Y0));
    const ox = rx + (rw - (X1 - X0) * k) / 2 - X0 * k;
    const oy = ry + rh - (rh - (Y1 - Y0) * k) / 2 + Y0 * k;
    const X = (x: number) => ox + x * k;
    const Y = (y: number) => oy - y * k;
    const path = (pts: number[], close = true) => {
      ctx.beginPath();
      ctx.moveTo(X(pts[0]), Y(pts[1]));
      for (let i = 2; i < pts.length; i += 2) ctx.lineTo(X(pts[i]), Y(pts[i + 1]));
      if (close) ctx.closePath();
    };

    // Panel radial background matching Voice Workshop HeadView
    const bg = ctx.createRadialGradient(rx + rw * 0.55, ry + rh * 0.45, 10, rx + rw * 0.55, ry + rh * 0.45, Math.max(rw, rh) * 0.75);
    bg.addColorStop(0, "#141a26");
    bg.addColorStop(1, C.bg);
    ctx.fillStyle = bg;
    ctx.fillRect(rx, ry, rw, rh);

    const nc = canonToWorld(NASAL_CENTRE, an, a);
    const nasalFlow = tel ? Math.min(1, Math.abs(tel.uNose) / 250) : 0;
    const port = tel ? tel.nasalPort : 0;
    const bd = getBackdrop(this.backdropUrl);

    if (bd) {
      const localWorld: Mat = [k, 0, 0, -k, ox - rx, oy - ry];
      const dprLocalWorld: Mat = mul([dpr, 0, 0, dpr, 0, 0], localWorld);
      const cw = Math.max(1, Math.round(rw * dpr));
      const ch = Math.max(1, Math.round(rh * dpr));
      const key = `${cw}x${ch}:${dpr.toFixed(2)}:${a.jaw.toFixed(3)},${a.tongueX.toFixed(3)},${a.tongueY.toFixed(3)},${a.tongueRoot.toFixed(3)},${a.tipX.toFixed(3)},${a.tipY.toFixed(3)},${a.lipOpen.toFixed(3)},${a.lipRound.toFixed(3)},${a.lipProtrude.toFixed(3)},${a.velum.toFixed(3)},${a.larynx.toFixed(3)},${teeth.slitOpen.toFixed(3)}`;
      if (this.layerKey !== key || !this.sceneCache) {
        if (this.layerCv.width !== cw || this.layerCv.height !== ch) {
          this.layerCv.width = cw;
          this.layerCv.height = ch;
        }
        const lc = this.layerCv.getContext("2d");
        if (lc) {
          lc.setTransform(1, 0, 0, 1, 0, 0);
          lc.clearRect(0, 0, cw, ch);
          this.sceneCache = headScene(geom, an, a);
          paintBackdrop(lc, this.sceneCache, dprLocalWorld, bd);
          paintSoftParts(lc, this.sceneCache, dprLocalWorld, bd);
          fadeNeck(lc, this.sceneCache, dprLocalWorld);
          const top = lc.createLinearGradient(0, 0, 0, 28 * dpr);
          top.addColorStop(0, "rgba(0,0,0,1)");
          top.addColorStop(1, "rgba(0,0,0,0)");
          lc.globalCompositeOperation = "destination-out";
          lc.fillStyle = top;
          lc.fillRect(0, 0, cw, 28 * dpr);
          lc.globalCompositeOperation = "source-over";
          this.layerKey = key;
        }
      }

      if (this.sceneCache) {
        const sc = this.sceneCache;
        const pageWorld: Mat = mul([dpr, 0, 0, dpr, 0, 0], [k, 0, 0, -k, ox, oy]);
        ctx.drawImage(this.layerCv, 0, 0, cw, ch, rx, ry, rw, rh);
        ctx.save();
        paintGlow(ctx, sc, pageWorld, port > 0.02 ? Math.min(1, port / 1.2) * (0.35 + 0.65 * nasalFlow) : 0);
        ctx.restore();
        if (sc.mouth.length) {
          path(sc.mouth);
          ctx.fillStyle = "#05070c";
          ctx.fill();
          ctx.fillStyle = fieldColor(0.12, 0.28);
          ctx.fill();
        }
        drawChannel(ctx, geom, X, Y, mode, tel);
        ctx.save();
        paintRim(ctx, sc, pageWorld);
        ctx.restore();
      }
    } else {
      paintFlat(ctx, geom, an, a, X, Y, path, k, s, tel, nc, port, nasalFlow, mode);
    }

    // Glottis pulse & vocal folds
    const gl = geom.landmarks.glottis;
    const voicing = tel ? tel.voicingAmp : 0;
    if (voicing > 0.02) {
      const f0 = tel?.f0 || 120;
      const pulse = 0.5 + 0.5 * Math.sin(timeSec * 2 * Math.PI * Math.min(6, f0 / 30));
      const rg = ctx.createRadialGradient(X(gl[0]), Y(gl[1]), 1, X(gl[0]), Y(gl[1]), (0.9 + 0.4 * pulse) * k);
      rg.addColorStop(0, `rgba(246,168,35,${0.75 * voicing})`);
      rg.addColorStop(1, "rgba(246,168,35,0)");
      ctx.fillStyle = rg;
      ctx.beginPath();
      ctx.arc(X(gl[0]), Y(gl[1]), (0.9 + 0.4 * pulse) * k, 0, Math.PI * 2);
      ctx.fill();
    }
    {
      const open = tel ? Math.min(1, tel.glottalArea / 0.25) : 0.3;
      ctx.fillStyle = C.amber;
      const gx = X(gl[0]);
      const gy = Y(gl[1]);
      const gw = 0.42 * k * s;
      ctx.beginPath();
      ctx.moveTo(gx - gw, gy - 2);
      ctx.lineTo(gx - 1 - open * 2.5, gy);
      ctx.lineTo(gx - gw, gy + 2);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(gx + gw, gy - 2);
      ctx.lineTo(gx + 1 + open * 2.5, gy);
      ctx.lineTo(gx + gw, gy + 2);
      ctx.closePath();
      ctx.fill();
    }

    // Live airflow particles along vocal tract & nasal cavity
    drawParticles(ctx, geom, X, Y, s, tel, this.parts, Math.min(0.05, Math.max(0.005, dt)), nc);

    // Optional anatomical landmark labels (Aa toggle)
    if (showLabels) {
      const lipFront = geom.landmarks.lipFront;
      const ue = geom.landmarks.upperEdge;
      const le = geom.landmarks.lowerEdge;
      const landmarks: [string, number, number, string][] = [
        ["tongue", geom.tongueCenter[0], geom.tongueCenter[1] + 0.7, C.teal],
        ["lips", lipFront[0] + 1.15 * s, lipFront[1] - 0.1 * s, C.teal],
        ["teeth", 0.5 * (ue[0] + le[0]) + 0.1, 0.5 * (ue[1] + le[1]) + 0.85, C.teal],
        ["larynx", gl[0] + 2.0 * s, gl[1] - 0.15 * s, C.amber],
      ];
      for (const [txt, wx, wy, col] of landmarks) {
        drawLabelPill(ctx, txt, X(wx), Y(wy), col);
      }
    }

    if (tel?.phone) {
      drawLabelPill(ctx, `/${tel.phone}/`, rx + rw - 38, ry + 20, C.amber);
    }

    ctx.restore();
  }
}
