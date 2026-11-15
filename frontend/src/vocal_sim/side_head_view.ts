// Cartoon Rave Wub-Bear Side-View Mid-Sagittal Head Renderer
// Powered by the 14-parameter Voice Workshop anatomical geometry.

import { VocalFieldState } from "./articulation_bridge";
import { C, fieldColor, FieldMode } from "./canvas_colors";
import { CHIN, chinWeight, HEAD_BACK, HEAD_FRONT_UP, NASAL_CENTRE, NASAL_HALF, smoothOutline } from "./head/outline";
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
  ctx.fillStyle = "#060212";
  ctx.fill();

  const L = geom.length;
  for (let i = 0; i < n; i++) {
    const h = geom.hits[i];
    const h2 = geom.hits[i + 1];
    if (h.d <= 0 && h2.d <= 0) continue;
    const k = Math.max(0, Math.min(N_SECTIONS - 1, Math.floor((geom.s[i] / L) * N_SECTIONS)));
    let v = 0.14;
    let alpha = 0.34;
    if (tel && tel.ug > 2) {
      if (mode === "pressure") v = tel.pRms[k];
      else if (mode === "velocity") v = tel.uRms[k];
      else {
        const u = k <= tel.constrictionIndex ? tel.ug : tel.uMouth;
        const vel = u / Math.max(h.area, 0.02);
        v = Math.min(1, Math.log10(1 + vel / 30) / 2.3);
      }
      alpha = 0.92;
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
      g.addColorStop(0, "#d43394");
      g.addColorStop(0.55, "#ff5ec9");
      g.addColorStop(1, "#ff8ce3");
      ctx.fillStyle = g;
      break;
    }
    case "upperIncisor":
    case "lowerIncisor":
      ctx.fillStyle = "#ffffff";
      break;
    case "palate":
      ctx.fillStyle = "#9d7ae8";
      break;
    case "velum":
      ctx.fillStyle = "#eb5ec0";
      break;
    case "upperLip":
    case "lowerLip":
      ctx.fillStyle = "#ff70dc";
      break;
    case "wall": {
      const g = ctx.createLinearGradient(X(-2.4 * s), 0, X(-0.45 * s), 0);
      g.addColorStop(0, "rgba(75,24,120,0)");
      g.addColorStop(0.55, "rgba(130,44,185,0.45)");
      g.addColorStop(1, "#a642dc");
      ctx.fillStyle = g;
      break;
    }
    case "larynx": {
      const g = ctx.createLinearGradient(X(0.4 * s), 0, X(2.6 * s), 0);
      g.addColorStop(0, "#a642dc");
      g.addColorStop(1, "rgba(75,24,120,0)");
      ctx.fillStyle = g;
      break;
    }
    case "floor":
      ctx.fillStyle = "rgba(128,46,168,0.82)";
      break;
  }
  ctx.fill();

  if (sol.id === "upperIncisor" || sol.id === "lowerIncisor") {
    ctx.strokeStyle = "#5f3ea8";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  } else if (
    sol.id === "tongue" ||
    sol.id === "upperLip" ||
    sol.id === "lowerLip" ||
    sol.id === "velum"
  ) {
    ctx.strokeStyle = "#19072e";
    ctx.lineWidth = 2.0;
    ctx.lineJoin = "round";
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,228,248,0.65)";
    ctx.lineWidth = 1.0;
    ctx.stroke();
  } else if (sol.id === "palate") {
    ctx.strokeStyle = "#2b124d";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function drawBearEar(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  rxPx: number,
  ryPx: number,
  rotRad: number,
  hue: number,
  isFar: boolean
): void {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(rotRad);

  if (!isFar) {
    ctx.save();
    ctx.shadowColor = "rgba(255, 174, 0, 0.85)";
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.ellipse(0, 0, rxPx, ryPx, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffcf2e";
    ctx.lineWidth = 4.0;
    ctx.stroke();
    ctx.restore();
  }

  const earGrad = ctx.createRadialGradient(-rxPx * 0.2, -ryPx * 0.25, rxPx * 0.1, 0, 0, Math.max(rxPx, ryPx));
  if (isFar) {
    earGrad.addColorStop(0, `hsla(${(hue + 280) % 360}, 78%, 48%, 0.72)`);
    earGrad.addColorStop(1, `hsla(${(hue + 230) % 360}, 72%, 22%, 0.82)`);
  } else {
    earGrad.addColorStop(0, `hsla(${(hue + 30) % 360}, 92%, 68%, 0.78)`);
    earGrad.addColorStop(0.55, `hsla(${(hue + 315) % 360}, 86%, 44%, 0.78)`);
    earGrad.addColorStop(1, `hsla(${(hue + 260) % 360}, 76%, 26%, 0.88)`);
  }
  ctx.beginPath();
  ctx.ellipse(0, 0, rxPx, ryPx, 0, 0, Math.PI * 2);
  ctx.fillStyle = earGrad;
  ctx.fill();

  ctx.strokeStyle = isFar ? "rgba(255, 180, 90, 0.55)" : "#ffe45c";
  ctx.lineWidth = isFar ? 1.8 : 2.3;
  ctx.stroke();

  // Glowing pink gummy bear inner ear cushion matching WubGotchi
  ctx.save();
  if (!isFar) {
    ctx.shadowColor = "#ff77e9";
    ctx.shadowBlur = 10;
  }
  ctx.beginPath();
  ctx.ellipse(rxPx * 0.08, ryPx * 0.08, rxPx * 0.56, ryPx * 0.56, 0, 0, Math.PI * 2);
  ctx.fillStyle = isFar ? "rgba(255, 120, 210, 0.34)" : "rgba(255, 145, 218, 0.65)";
  ctx.fill();
  ctx.restore();

  if (!isFar) {
    ctx.beginPath();
    ctx.ellipse(-rxPx * 0.28, -ryPx * 0.38, rxPx * 0.24, ryPx * 0.14, -0.45, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
    ctx.fill();
  }
  ctx.restore();
}

function drawAnimeEye(
  ctx: CanvasRenderingContext2D,
  ex: number,
  ey: number,
  rPx: number,
  timeSec: number,
  voicing: number,
  mouthOpen: number
): void {
  const blinkPhase = (timeSec * 0.28) % 1;
  const isBlink = blinkPhase > 0.94;
  const happyBelt = voicing > 0.72 && mouthOpen > 0.45;

  ctx.save();
  ctx.translate(ex, ey);

  // Cute raised neon gold eyebrow
  ctx.strokeStyle = "#ffe45c";
  ctx.lineWidth = Math.max(1.8, rPx * 0.19);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-rPx * 0.95, -rPx * (0.95 + 0.18 * voicing));
  ctx.quadraticCurveTo(0, -rPx * (1.28 + 0.24 * voicing), rPx * 0.95, -rPx * (0.85 + 0.15 * voicing));
  ctx.stroke();

  if (isBlink || happyBelt) {
    ctx.strokeStyle = "#19072e";
    ctx.lineWidth = Math.max(2.6, rPx * 0.28);
    ctx.beginPath();
    if (happyBelt) {
      ctx.arc(0, rPx * 0.2, rPx * 0.85, Math.PI * 1.12, Math.PI * 1.88);
    } else {
      ctx.moveTo(-rPx * 0.9, 0);
      ctx.quadraticCurveTo(0, rPx * 0.32, rPx * 0.9, 0);
    }
    ctx.stroke();

    ctx.strokeStyle = "#ffe45c";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.restore();
    return;
  }

  // Almond sclera
  ctx.beginPath();
  ctx.ellipse(0, 0, rPx * 1.05, rPx * 0.78, -0.06, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.strokeStyle = "#19072e";
  ctx.lineWidth = Math.max(2, rPx * 0.18);
  ctx.stroke();

  // Neon cyan-to-violet iris
  const irisGrad = ctx.createLinearGradient(0, -rPx * 0.6, 0, rPx * 0.65);
  irisGrad.addColorStop(0, "#38f3e3");
  irisGrad.addColorStop(0.55, "#6b4dff");
  irisGrad.addColorStop(1, "#ff4cd8");
  ctx.beginPath();
  ctx.ellipse(rPx * 0.12, 0, rPx * 0.6, rPx * 0.64, 0, 0, Math.PI * 2);
  ctx.fillStyle = irisGrad;
  ctx.fill();

  // Pupil + kawaii double sparkle catchlights
  ctx.beginPath();
  ctx.arc(rPx * 0.16, rPx * 0.03, rPx * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = "#0c041c";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(-rPx * 0.06, -rPx * 0.24, rPx * 0.22, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(rPx * 0.34, rPx * 0.24, rPx * 0.11, 0, Math.PI * 2);
  ctx.fillStyle = "#fff59e";
  ctx.fill();

  // Upper lash curve + cute cat/bear outer wing
  ctx.strokeStyle = "#19072e";
  ctx.lineWidth = Math.max(2.3, rPx * 0.24);
  ctx.beginPath();
  ctx.ellipse(0, -rPx * 0.04, rPx * 1.08, rPx * 0.76, -0.06, Math.PI * 1.08, Math.PI * 1.94);
  ctx.stroke();
  ctx.restore();
}

function paintRaveBearHead(
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
  mode: FieldMode,
  timeSec: number
): void {
  const voicing = tel ? tel.voicingAmp : 0;
  const hue = (timeSec * 42) % 360;
  const beatScale = 1 + voicing * 0.08;
  const earWiggle = Math.sin(timeSec * 12) * (0.04 + 0.14 * voicing);

  const back = canonToWorld(HEAD_BACK, { ...an, pharynx: 1 }, { ...a, larynx: 0 });
  const frontUp = canonToWorld(HEAD_FRONT_UP, an, a);
  const chin = canonToWorld(CHIN, { ...an, pharynx: 1 }, { ...a, larynx: 0 }, chinWeight);
  const rawSilhouette = [...back, ...frontUp, ...chin];
  const silhouette = smoothOutline(rawSilhouette, 5);

  // Far bear ear peeking behind the top forehead crown
  const farEarWorld = canonToWorld([3.3, 21.8], { ...an, pharynx: 1 }, { ...a, larynx: 0 });
  drawBearEar(
    ctx,
    X(farEarWorld[0]),
    Y(farEarWorld[1]),
    1.85 * k * s * beatScale,
    2.05 * k * s * beatScale,
    0.26 - earWiggle * 0.7,
    hue,
    true
  );

  // Translucent gummi bear body gradient (WubGotchi aesthetic)
  const jellyGrad = ctx.createLinearGradient(X(-6.0 * s), Y(21.5 * s), X(10.5 * s), Y(2.0 * s));
  jellyGrad.addColorStop(0, `hsla(${(hue + 28) % 360}, 92%, 64%, 0.78)`);
  jellyGrad.addColorStop(0.45, `hsla(${(hue + 318) % 360}, 86%, 52%, 0.72)`);
  jellyGrad.addColorStop(0.85, `hsla(${(hue + 215) % 360}, 84%, 40%, 0.80)`);
  jellyGrad.addColorStop(1, "rgba(18, 6, 40, 0.92)");

  // Outer golden neon bloom aura on bear silhouette
  ctx.save();
  ctx.shadowColor = "#ffae00";
  ctx.shadowBlur = 18 + voicing * 10;
  path(silhouette);
  ctx.fillStyle = jellyGrad;
  ctx.fill();
  ctx.strokeStyle = "#ffcf2e";
  ctx.lineWidth = 4.6;
  ctx.lineJoin = "round";
  ctx.stroke();
  ctx.restore();

  // Crisp inner laser gold stroke
  path(silhouette);
  ctx.strokeStyle = "#ffe45c";
  ctx.lineWidth = 2.0;
  ctx.lineJoin = "round";
  ctx.stroke();

  // Clipped interior cartoon details: glowing blush, glossy shine, spine pills
  ctx.save();
  path(silhouette);
  ctx.clip();

  // Rosy kawaii rave cheek blush (brightens when singing)
  const cheek = canonToWorld([5.2, 12.9], an, a);
  const cheekR = 2.05 * k * s;
  const blushGrad = ctx.createRadialGradient(X(cheek[0]), Y(cheek[1]), 0, X(cheek[0]), Y(cheek[1]), cheekR);
  blushGrad.addColorStop(0, `rgba(255, 88, 200, ${0.38 + 0.36 * voicing})`);
  blushGrad.addColorStop(1, "rgba(255, 88, 200, 0)");
  ctx.fillStyle = blushGrad;
  ctx.beginPath();
  ctx.arc(X(cheek[0]), Y(cheek[1]), cheekR, 0, Math.PI * 2);
  ctx.fill();

  // Glossy gummi bear specular crown shine
  const crown = canonToWorld([1.2, 19.6], { ...an, pharynx: 1 }, { ...a, larynx: 0 });
  ctx.save();
  ctx.translate(X(crown[0]), Y(crown[1]));
  ctx.rotate(-0.35);
  ctx.beginPath();
  ctx.ellipse(0, 0, 2.9 * k * s, 0.72 * k * s, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255, 255, 255, 0.24)";
  ctx.fill();
  ctx.restore();

  // Glowing cyber-vertebrae pills down the neck
  for (let i = 0; i < 6; i++) {
    const v = canonToWorld([-2.9, -2.6 + i * 1.9, -1.7, -1.1 + i * 1.9], an, a);
    ctx.fillStyle = i % 2 === 0 ? "rgba(42, 232, 212, 0.22)" : "rgba(255, 92, 214, 0.20)";
    ctx.strokeStyle = "rgba(255, 228, 92, 0.28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(X(v[0]), Y(v[3]), (v[2] - v[0]) * k, (v[3] - v[1]) * k, 5);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  // Foreground round Wub-Bear ear overlapping top-back crown
  const nearEarWorld = canonToWorld([-1.5, 21.1], { ...an, pharynx: 1 }, { ...a, larynx: 0 });
  drawBearEar(
    ctx,
    X(nearEarWorld[0]),
    Y(nearEarWorld[1]),
    2.15 * k * s * beatScale,
    2.3 * k * s * beatScale,
    -0.28 + earWiggle,
    hue,
    false
  );

  // Cute round shiny button bear nose right at the profile nose tip
  const nose = canonToWorld([10.55, 11.85], an, a);
  ctx.save();
  ctx.translate(X(nose[0]), Y(nose[1]));
  ctx.rotate(0.2);
  ctx.beginPath();
  ctx.ellipse(0, 0, 0.72 * k * s, 0.56 * k * s, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#19072e";
  ctx.fill();
  ctx.strokeStyle = "#ffe45c";
  ctx.lineWidth = 1.6;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(-0.22 * k * s, -0.18 * k * s, 0.18 * k * s, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.restore();

  // Sparkly anime eye
  const eye = canonToWorld([7.05, 16.0], an, a);
  drawAnimeEye(ctx, X(eye[0]), Y(eye[1]), 0.85 * k * s, timeSec, voicing, a.lipOpen);

  // Nasal cavity + neon cyan/magenta shimmer when velum port is open
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
  ctx.fillStyle = "#060212";
  ctx.fill();
  if (port > 0.02) {
    const nAlpha = Math.min(0.75, 0.16 + 0.55 * Math.min(1, port / 1.2) * (0.35 + 0.65 * nasalFlow));
    ctx.fillStyle = `rgba(42, 232, 212, ${nAlpha})`;
    ctx.fill();
  }
  ctx.strokeStyle = "rgba(255, 119, 233, 0.65)";
  ctx.lineWidth = 1.4;
  ctx.stroke();

  // Standing-wave sound field inside the vocal channel
  drawChannel(ctx, geom, X, Y, mode, tel);

  // Cel-shaded tongue, palate, velum, lips & bright bear teeth
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
  ctx.fillStyle = "rgba(88, 245, 232, 0.9)";
  for (let i = 0; i < parts.length; i++) {
    const q = parts[i];
    q.age += dt;
    ctx.fillStyle = i % 3 === 0 ? "rgba(255, 224, 96, 0.92)" : i % 3 === 1 ? "rgba(255, 110, 220, 0.88)" : "rgba(72, 246, 232, 0.92)";
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
      ctx.arc(X(x), Y(y + q.off * 0.6 * s), 1.7, 0, Math.PI * 2);
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
      ctx.arc(X(x), Y(y), 1.8, 0, Math.PI * 2);
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
    ctx.arc(X(x), Y(y), 1.65, 0, Math.PI * 2);
    ctx.fill();
  }
}

export class SideHeadView {
  private readonly parts: Particle[] = [];

  constructor(_backdropUrl: string) {}

  public renderRegion(
    ctx: CanvasRenderingContext2D,
    _dpr: number,
    rx: number,
    ry: number,
    rw: number,
    rh: number,
    geom: TractGeometry,
    an: Anatomy,
    a: Articulators,
    _teeth: Teeth,
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
    const X0 = -5.8;
    const X1 = 12.2;
    const Y0 = -3.4;
    let Y1 = 20.0;
    if (rh / rw > (Y1 - Y0) / (X1 - X0)) {
      Y1 = Math.min(22.2, Y0 + (rh / rw) * (X1 - X0));
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

    const voicing = tel ? tel.voicingAmp : 0;

    // Cosmic rave nebula backdrop + festival laser rays + audio halo
    const cxBg = rx + rw * 0.52;
    const cyBg = ry + rh * 0.48;
    const bg = ctx.createRadialGradient(cxBg, cyBg, 8, cxBg, cyBg, Math.max(rw, rh) * 0.78);
    bg.addColorStop(0, "#1c093d");
    bg.addColorStop(0.55, "#0d0522");
    bg.addColorStop(1, C.bg);
    ctx.fillStyle = bg;
    ctx.fillRect(rx, ry, rw, rh);

    ctx.save();
    for (let i = 0; i < 6; i++) {
      const ang = (i / 6) * Math.PI * 2 + timeSec * 0.25;
      const beamLen = Math.max(rw, rh) * 0.9;
      ctx.beginPath();
      ctx.moveTo(cxBg, cyBg);
      ctx.lineTo(cxBg + Math.cos(ang - 0.08) * beamLen, cyBg + Math.sin(ang - 0.08) * beamLen);
      ctx.lineTo(cxBg + Math.cos(ang + 0.08) * beamLen, cyBg + Math.sin(ang + 0.08) * beamLen);
      ctx.closePath();
      ctx.fillStyle = i % 2 === 0
        ? `rgba(255, 76, 216, ${0.035 + voicing * 0.055})`
        : `rgba(42, 232, 212, ${0.035 + voicing * 0.055})`;
      ctx.fill();
    }
    const aura = ctx.createRadialGradient(cxBg, cyBg, 6, cxBg, cyBg, Math.min(rw, rh) * (0.44 + voicing * 0.14));
    aura.addColorStop(0, `rgba(255, 198, 54, ${0.14 + voicing * 0.2})`);
    aura.addColorStop(0.6, `rgba(255, 76, 216, ${0.08 + voicing * 0.14})`);
    aura.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = aura;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.restore();

    const nc = canonToWorld(NASAL_CENTRE, an, a);
    const nasalFlow = tel ? Math.min(1, Math.abs(tel.uNose) / 250) : 0;
    const port = tel ? tel.nasalPort : 0;

    paintRaveBearHead(ctx, geom, an, a, X, Y, path, k, s, tel, nc, port, nasalFlow, mode, timeSec);

    // Glottis neon starburst & vocal folds
    const gl = geom.landmarks.glottis;
    if (voicing > 0.02) {
      const f0 = tel?.f0 || 120;
      const pulse = 0.5 + 0.5 * Math.sin(timeSec * 2 * Math.PI * Math.min(6, f0 / 30));
      const rg = ctx.createRadialGradient(X(gl[0]), Y(gl[1]), 1, X(gl[0]), Y(gl[1]), (1.1 + 0.55 * pulse) * k);
      rg.addColorStop(0, `rgba(255, 224, 96, ${0.9 * voicing})`);
      rg.addColorStop(0.45, `rgba(255, 76, 216, ${0.5 * voicing})`);
      rg.addColorStop(1, "rgba(255, 76, 216, 0)");
      ctx.fillStyle = rg;
      ctx.beginPath();
      ctx.arc(X(gl[0]), Y(gl[1]), (1.1 + 0.55 * pulse) * k, 0, Math.PI * 2);
      ctx.fill();
    }
    {
      const open = tel ? Math.min(1, tel.glottalArea / 0.25) : 0.3;
      ctx.fillStyle = C.amber;
      const gx = X(gl[0]);
      const gy = Y(gl[1]);
      const gw = 0.44 * k * s;
      ctx.beginPath();
      ctx.moveTo(gx - gw, gy - 2.2);
      ctx.lineTo(gx - 1 - open * 2.5, gy);
      ctx.lineTo(gx - gw, gy + 2.2);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(gx + gw, gy - 2.2);
      ctx.lineTo(gx + 1 + open * 2.5, gy);
      ctx.lineTo(gx + gw, gy + 2.2);
      ctx.closePath();
      ctx.fill();
    }

    // Live neon airflow particles along vocal tract & nasal cavity
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
