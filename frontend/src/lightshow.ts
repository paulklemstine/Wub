// Trippy Rave Lightshow Output Engine
// Powered by shaders & post-FX ported from github.com/paulklemstine/SuperAcid
// (acid-warp, breathing-mandala, burning-julia, cosmic-serpent, color-organ,
// apollonian-gasket, acid-smiley) + raver1975 SpaceTubes hyperspace laser tunnel.
// Runs a two-pass WebGL pipeline (Scene Shader -> Ping-Pong Feedback Trails +
// SuperAcid Bloom / Chromatic Aberration / Glitch / Kaleidoscope / Safe Strobe)
// inside #lightshow-canvas with side VJ controls & Fullscreen support.

export type LightshowPresetId =
  | "acid-warp"
  | "breathing-mandala"
  | "burning-julia"
  | "cosmic-serpent"
  | "color-organ"
  | "apollonian"
  | "acid-smiley"
  | "space-tubes";

export const LIGHTSHOW_PRESETS: readonly { id: LightshowPresetId; label: string }[] = [
  { id: "acid-warp", label: "ACID WARP" },
  { id: "breathing-mandala", label: "DMT MANDALA" },
  { id: "space-tubes", label: "SPACETUBES" },
  { id: "burning-julia", label: "BURNING JULIA" },
  { id: "cosmic-serpent", label: "COSMIC SERPENT" },
  { id: "color-organ", label: "COLOR ORGAN" },
  { id: "apollonian", label: "APOLLONIAN" },
  { id: "acid-smiley", label: "ACID SMILEY" },
];

export interface LightshowState {
  enabled: boolean;
  preset: LightshowPresetId;
  cortexAuto: boolean;
  intensity: number; // 0.3 .. 2.2
  speed: number;     // 0.2 .. 2.5
  palette: number;   // 0 .. 3
  fxBloom: boolean;
  fxTrails: boolean;
  fxChroma: boolean;
  fxGlitch: boolean;
  fxStrobe: boolean;
  fxKaleido: boolean;
  blackout: boolean;
}

export interface LightshowAudioFrame {
  timeSec: number;
  bpm: number;
  section: string;
  drums: number;
  bass: number;
  other: number;
  vocals: number;
  mix: number;
  drumPunch: number;
  freqBins?: Uint8Array | null;
}

const VERT_SRC = `
attribute vec2 a_pos;
varying vec2 vUv;
void main() {
  vUv = (a_pos + 1.0) * 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`;

const SCENE_HEADER_GLSL = `
precision highp float;
varying vec2 vUv;

uniform vec2 uRes;
uniform float uTime;
uniform float uBpm;
uniform float uBeat;
uniform float uBpmSin;
uniform float uKick;
uniform float uOnset;
uniform float uEnergy;
uniform float uIntensity;
uniform float uSpeed;
uniform float uPalette;
uniform float uChaos;
uniform float uSpectrum0;
uniform float uSpectrum1;
uniform float uSpectrum2;
uniform float uSpectrum3;
uniform float uSpectrum4;
uniform float uSpectrum5;
uniform float uSpectrum6;
uniform float uSpectrum7;

#define PI 3.14159265359
#define TAU 6.28318530718

float bass()   { return uSpectrum0 * 0.35 + uSpectrum1 * 0.45 + uSpectrum2 * 0.20; }
float mids()   { return uSpectrum2 * 0.30 + uSpectrum3 * 0.40 + uSpectrum4 * 0.30; }
float treble() { return uSpectrum5 * 0.25 + uSpectrum6 * 0.45 + uSpectrum7 * 0.30; }

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
    mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}

float fbm4(vec2 p) {
  float v = 0.0, a = 0.5;
  mat2 r = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 4; i++) {
    v += a * noise2(p);
    p = r * p * 2.02;
    a *= 0.5;
  }
  return v;
}

vec3 hsv2rgb(float h, float s, float v) {
  vec3 c = abs(fract(h + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0;
  return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}

vec2 psychedelicUV(vec2 uv) {
  float m = 0.035 + uEnergy * 0.025;
  uv += m * vec2(sin(uTime * 0.7 + uv.y * 4.0), cos(uTime * 0.6 + uv.x * 4.0));
  return uv;
}

vec3 psychedelicColor(vec3 c) {
  float luma = dot(c, vec3(0.299, 0.587, 0.114));
  vec3 r1 = 0.5 + 0.5 * cos(TAU * (luma * 2.6 + uTime * 0.14 + uPalette * 0.25 + vec3(0.0, 0.33, 0.67)));
  vec3 r2 = 0.5 + 0.5 * cos(TAU * (luma * 1.4 - uTime * 0.08 + uPalette * 0.15 + vec3(0.1, 0.5, 0.8)));
  vec3 blendR = mix(r1, r2, 0.3 + uEnergy * 0.2);
  c = mix(c, blendR * max(length(c), 0.22) * 1.55, 0.32 + uEnergy * 0.25);
  float grey = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(vec3(grey), c, 1.55 + uEnergy * 0.35);
  return pow(max(c, 0.0), vec3(0.86));
}

vec2 aspectUv() {
  vec2 p = vUv - 0.5;
  p.x *= uRes.x / max(1.0, uRes.y);
  return p;
}
`;

// SuperAcid shader bodies ported cleanly to WebGL1 GLSL
const PRESET_BODIES: Record<LightshowPresetId, string> = {
  "acid-warp": `
void main() {
  vec2 p = aspectUv() * 3.6;
  p = psychedelicUV(p);
  float time = uTime * uSpeed * 1.65;
  float bWarp = bass() * 1.8;
  float mHue = mids() * 0.65;
  float spike = uOnset * 2.4;

  float s1 = 1.25 + bWarp * 0.5 + spike * 0.25;
  vec2 p1 = p + vec2(sin(p.y * 2.5 + time * 0.7 + uChaos * sin(time * 0.2)), cos(p.x * 2.3 + time * 0.6)) * s1;
  float s2 = 0.95 + bWarp * 0.35;
  vec2 p2 = p1 + vec2(sin(p1.y * 3.7 + time * 0.5 + p1.x * 1.3), cos(p1.x * 3.1 - time * 0.4 + p1.y * 0.9)) * s2;
  float s3 = 0.75 + bWarp * 0.25;
  vec2 p3 = p2 + vec2(sin(p2.y * 5.1 + time * 0.3 + p2.x * 2.1), cos(p2.x * 4.7 - time * 0.35 + p2.y * 1.7)) * s3;
  vec2 pf = p3 + vec2(sin(p3.y * 7.3 + time * 0.2), cos(p3.x * 6.9 - time * 0.25)) * (0.55 + spike * 0.3);

  float pat1 = sin(pf.x * 5.0 + pf.y * 4.0 + time * 0.3) * 0.5 + 0.5;
  float pat2 = sin(pf.x * 6.5 - pf.y * 5.5 + time * 0.4) * 0.5 + 0.5;
  float pat3 = sin((pf.x + pf.y) * 3.5 + time * 0.5) * 0.5 + 0.5;
  float comb = mix(mix(pat1, pat2, 0.55), pat3, 0.35);

  float hue = fract(comb * 0.8 + uPalette * 0.24 + time * 0.05 + mHue);
  vec3 col = hsv2rgb(hue, clamp(0.78 + comb * 0.2, 0.0, 1.0), clamp(uIntensity * (comb * 0.7 + 0.32 + uEnergy * 0.3), 0.0, 1.4));
  col += vec3(uKick * 0.25);
  col = psychedelicColor(smoothstep(vec3(0.0), vec3(0.92), pow(col, vec3(0.85))));
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`,

  "breathing-mandala": `
vec2 kaleido(vec2 p, float folds) {
  float a = atan(p.y, p.x);
  float sec = TAU / folds;
  a = mod(a + TAU, sec);
  a = abs(a - sec * 0.5);
  return vec2(cos(a), sin(a)) * length(p);
}
void main() {
  vec2 p = aspectUv() * 2.4;
  p = psychedelicUV(p);
  float time = uTime * uSpeed * 0.75;
  float bassVal = bass();
  float midsVal = mids();
  float trebleVal = treble();
  float breathCycle = uBpmSin * 0.5 + 0.5;
  p /= (1.0 + breathCycle * 0.38 + bassVal * 0.28);

  float folds = floor(8.0 + uChaos * 6.0);
  vec2 kp = kaleido(p, folds);
  float r = length(p);
  float kr = length(kp);

  float pat = 0.0, amp = 1.0;
  for (int i = 0; i < 6; i++) {
    float fi = float(i);
    float freq = 3.2 + fi * 4.0;
    float g = abs(sin(kp.x * freq + time * (0.5 + fi * 0.2)) * cos(kp.y * freq * 0.7 - time * 0.3));
    float ring = pow(abs(sin(kr * (5.0 + fi * 3.0) - time * 0.5)), 2.0);
    float dm = abs(sin((abs(kp.x + kp.y) + abs(kp.x - kp.y)) * (4.0 + fi * 2.0) + time * 0.3));
    pat += (g * 0.4 + ring * 0.3 + dm * 0.3) * amp;
    amp *= 0.6;
  }
  pat *= 0.5;
  float hue = pat * 0.6 + time * 0.08 + r * 0.3 + uPalette * 0.25;
  vec3 j1 = vec3(0.95, 0.22, 0.58), j2 = vec3(0.18, 0.78, 1.0), j3 = vec3(0.32, 0.95, 0.55), j4 = vec3(1.0, 0.85, 0.22);
  float h4 = fract(hue) * 4.0;
  vec3 col = h4 < 1.0 ? mix(j1, j2, fract(h4)) : h4 < 2.0 ? mix(j2, j3, fract(h4)) : h4 < 3.0 ? mix(j3, j4, fract(h4)) : mix(j4, j1, fract(h4));
  col *= pat;
  col += pow(abs(sin(kr * 20.0 - time)), 8.0) * 0.18 * vec3(1.0, 0.82, 0.65);
  col += exp(-r * r * 7.0) * mix(j1, j4, sin(time * 0.5) * 0.5 + 0.5) * (0.4 + uKick * 0.6);
  col *= (0.85 + breathCycle * 0.3 + midsVal * 0.2 + trebleVal * 0.15) * uIntensity;
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,

  "space-tubes": `
void main() {
  vec2 p = aspectUv() * 2.0;
  float time = uTime * uSpeed;
  float b = bass();
  float tr = treble();
  p += vec2(sin(time * 0.9), cos(time * 0.7)) * (0.12 + 0.16 * b);
  float r = length(p) + 0.001;
  float ang = atan(p.y, p.x);

  // Hyperspace wireframe + hexagonal neon tunnel ribs inspired by SpaceTubes
  float z = 1.6 / r + time * (2.6 + b * 2.2);
  float rib = pow(abs(sin(z * 2.4)), 10.0);
  float sector = abs(sin(ang * 6.0 + sin(z * 0.45) * 1.8));
  float rail = smoothstep(0.92, 1.0, sector);

  float star = smoothstep(0.985, 1.0, hash21(floor(vec2(ang * 12.0, z * 3.0)))) * (0.6 + tr * 1.4);
  float hue = fract(z * 0.08 + uPalette * 0.25 + time * 0.05);
  vec3 neon1 = hsv2rgb(hue, 0.95, 1.0);
  vec3 neon2 = hsv2rgb(fract(hue + 0.42), 0.92, 1.0);

  vec3 col = mix(vec3(0.02, 0.01, 0.08), neon1 * 0.85, rib);
  col += neon2 * rail * (0.45 + uKick * 0.75);
  col += vec3(0.85, 0.95, 1.0) * star;
  col *= smoothstep(0.06, 0.35, r); // deep dark center vortex
  col += exp(-r * 6.0) * neon2 * (0.5 + b * 0.9);
  col *= uIntensity * (0.85 + 0.45 * uEnergy);
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,

  "burning-julia": `
vec3 burnPal(float t) {
  return 0.5 + 0.5 * cos(TAU * (vec3(1.0, 0.7, 0.4) * t + vec3(uPalette * 0.2, 0.15, 0.3)));
}
void main() {
  vec2 uv = aspectUv() * 2.8;
  uv = psychedelicUV(uv);
  float time = uTime * uSpeed * 0.55;
  float rotA = time * 0.15 + uChaos * sin(time * 0.1) * 0.25;
  float cr = cos(rotA), sr = sin(rotA);
  uv = vec2(uv.x * cr - uv.y * sr, uv.x * sr + uv.y * cr);

  float angle = time * 0.22;
  vec2 c = vec2(0.72 * cos(angle), 0.72 * sin(angle)) + vec2(bass() * 0.14, mids() * 0.1);
  vec2 z = uv;
  float iter = 0.0, trap = 1e6, eTrap = 1e6;
  for (int i = 0; i < 64; i++) {
    if (dot(z, z) > 128.0) break;
    z = vec2(abs(z.x), abs(z.y));
    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
    trap = min(trap, length(z));
    eTrap = min(eTrap, abs(z.x * z.y));
    iter += 1.0;
  }
  vec3 col;
  if (dot(z, z) > 128.0) {
    float ct = fract(iter * 0.03 + time * 0.05 + bass() * 0.1);
    col = burnPal(ct) * (0.25 + 0.8 * (1.0 - iter / 64.0));
    col += burnPal(fract(ct + 0.3)) * exp(-eTrap * 18.0) * (0.35 + treble() * 0.8);
  } else {
    col = burnPal(fract(trap * 2.5 + 0.5)) * 0.25 + burnPal(fract(trap * 4.0)) * exp(-eTrap * 10.0) * 0.45;
  }
  col *= uIntensity * (1.25 + uKick * 0.4);
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,

  "cosmic-serpent": `
void main() {
  vec2 p = aspectUv() * 2.4;
  p = psychedelicUV(p);
  float time = uTime * uSpeed * 0.6;
  float bVal = bass(), mVal = mids(), tVal = treble();
  float radius = 0.36 + bVal * 0.22;
  float rotSpd = time * (1.2 + mVal * 0.8);
  float glow1 = 0.0, glow2 = 0.0, rungGlow = 0.0, bestT1 = 0.0, bestT2 = 0.0, bd1 = 99.0, bd2 = 99.0;

  for (int i = 0; i < 36; i++) {
    float fi = float(i);
    float t = (fi / 36.0 - 0.5) * 2.8;
    float axisX = sin(t * 2.2 + time * 0.9) * 0.42 * (1.0 + bVal * 0.9);
    float hA = t * 5.2 + rotSpd;
    vec2 s1 = vec2(axisX + cos(hA) * radius, t);
    vec2 s2 = vec2(axisX + cos(hA + PI) * radius, t);
    float d1 = length(p - s1);
    float d2 = length(p - s2);
    if (d1 < bd1) { bd1 = d1; bestT1 = t; }
    if (d2 < bd2) { bd2 = d2; bestT2 = t; }
    float thick = 0.065 + uBeat * 0.02;
    glow1 += exp(-d1 * d1 / (thick * thick)) * (0.7 + 0.3 * sin(hA)) * 0.16;
    glow2 += exp(-d2 * d2 / (thick * thick)) * (0.7 + 0.3 * cos(hA)) * 0.16;
    if (mod(fi, 4.0) < 1.0) {
      vec2 dir = s2 - s1;
      float len = length(dir);
      vec2 pt = s1 + dir * clamp(dot(p - s1, dir) / max(len * len, 0.001), 0.0, 1.0);
      float rd = length(p - pt);
      rungGlow += exp(-rd * rd * 280.0) * 0.45;
    }
  }
  vec3 c1 = hsv2rgb(fract(bestT1 * 0.25 + uPalette * 0.2 + 0.32), 0.9, 1.0);
  vec3 c2 = hsv2rgb(fract(bestT2 * 0.25 + uPalette * 0.2 + 0.72), 0.9, 1.0);
  vec3 col = c1 * glow1 + c2 * glow2 + mix(c1, c2, 0.5) * rungGlow * (0.6 + tVal);
  col += (c1 * glow1 + c2 * glow2) * uKick * 2.2;
  col *= uIntensity * 1.9;
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,

  "color-organ": `
void main() {
  vec2 p = aspectUv() * 3.0;
  p = psychedelicUV(p);
  float time = uTime * uSpeed;
  vec2 fp = p + vec2(fbm4(p * 1.4 + time * 0.2) - 0.5, fbm4(p * 1.4 - time * 0.16 + 7.0) - 0.5) * (0.25 + uChaos * 0.3);
  float r = length(fp);
  float a = atan(fp.y, fp.x);
  float bArr[8];
  bArr[0] = uSpectrum0; bArr[1] = uSpectrum1; bArr[2] = uSpectrum2; bArr[3] = uSpectrum3;
  bArr[4] = uSpectrum4; bArr[5] = uSpectrum5; bArr[6] = uSpectrum6; bArr[7] = uSpectrum7;

  vec3 col = vec3(0.0);
  float wSum = 0.0;
  for (int i = 0; i < 8; i++) {
    float fi = float(i);
    float zoneR = (fi + 0.5) / 8.0 * 1.7;
    float bNoise = noise2(vec2(a * 3.0 + fi * 5.0, time * 0.3 + fi)) * 0.14 * uChaos;
    float d = abs(r - zoneR + bNoise);
    float w = exp(-d * d / 0.06);
    float hue = fi / 8.0 + uPalette * 0.22 + time * 0.03;
    float amp = bArr[i];
    vec3 zCol = hsv2rgb(fract(hue), 0.5 + amp * 0.5, (0.22 + amp * 0.95) * (1.0 + 0.35 * sin(time * (2.0 + fi * 0.4))));
    col += zCol * w;
    wSum += w;
  }
  col /= max(wSum, 0.001);
  col += hsv2rgb(fract(a / TAU + uPalette * 0.25), 0.8, 0.4) * (sin(a * 6.0 + time * 0.5 + r * 3.0) * 0.5 + 0.5) * 0.25;
  col *= (0.75 + uEnergy * 0.65 + uKick * 0.35) * uIntensity;
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,

  apollonian: `
void main() {
  vec2 uv = aspectUv() * 2.8;
  uv = psychedelicUV(uv);
  float time = uTime * uSpeed * 0.85;
  float b = bass() * 1.6, tr = treble() * 1.6;
  float rotA = time * 0.16 + uBeat * 0.18;
  float cs = cos(rotA), sn = sin(rotA);
  vec2 p = vec2(uv.x * cs - uv.y * sn, uv.x * sn + uv.y * cs);
  float scale = 1.0, trap = 0.0, edgeAcc = 0.0;
  for (int i = 0; i < 12; i++) {
    p.x = abs(p.x);
    float fA = PI / 3.0 + uChaos * sin(time * 0.15 + float(i) * 0.5) * 0.22;
    vec2 fn = vec2(sin(fA), -cos(fA));
    float fd = dot(p, fn);
    if (fd < 0.0) p -= 2.0 * fd * fn;
    float rad = (1.0 + 0.16 * sin(time * 0.3 + float(i))) * (1.0 + b * 0.1 * sin(float(i)));
    vec2 diff = p - vec2(1.0, 0.0);
    float invS = rad * rad / max(dot(diff, diff), 0.0005);
    p = vec2(1.0, 0.0) + diff * invS;
    scale *= invS;
    float dist = length(diff) - rad * 0.5;
    edgeAcc += smoothstep(0.025 * scale, 0.0, abs(dist) / scale) / float(i + 1);
    trap += length(p) / float(i + 1);
  }
  float ct = fract(trap * 0.15 + time * 0.08 + uPalette * 0.25);
  vec3 col = hsv2rgb(ct, 0.9, 0.4 + edgeAcc * 1.6);
  col += hsv2rgb(fract(ct + 0.35), 0.95, 1.0) * edgeAcc * (0.6 + tr * 0.8);
  col *= (1.1 + uKick * 0.35) * uIntensity;
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,

  "acid-smiley": `
void main() {
  vec2 p = aspectUv() * 1.85;
  p = psychedelicUV(p);
  float time = uTime * uSpeed * 1.3;
  float b = bass(), m = mids(), tr = treble();
  float r2 = dot(p, p) + 0.0001;
  float ang = atan(p.y, p.x);
  float sp = ang * 3.0 + r2 * 18.0 - time * 1.4 + b * 2.2;
  vec3 bgCol = hsv2rgb(fract(sp * 0.12 + uPalette * 0.25 + time * 0.05), 0.95, 0.85 * uIntensity);

  float faceR = 0.42 * (1.0 + 0.04 * sin(time * 2.0) + uKick * 0.06);
  vec2 fp = p + vec2(sin(p.y * 10.0 + time * 2.0), cos(p.x * 10.0 + time * 1.7)) * 0.02;
  float fMask = smoothstep(faceR * faceR + 0.004, faceR * faceR - 0.004, dot(fp, fp));
  vec3 fCol = vec3(1.0, 0.9, 0.08);

  // Eyes + audio-reactive singing smiley mouth & pink tongue
  float eL = dot((fp - vec2(-faceR * 0.32, faceR * 0.22)) / vec2(faceR * 0.1, faceR * 0.15), (fp - vec2(-faceR * 0.32, faceR * 0.22)) / vec2(faceR * 0.1, faceR * 0.15));
  float eR = dot((fp - vec2(faceR * 0.32, faceR * 0.22)) / vec2(faceR * 0.1, faceR * 0.15), (fp - vec2(faceR * 0.32, faceR * 0.22)) / vec2(faceR * 0.1, faceR * 0.15));
  float eMask = max(smoothstep(1.2, 0.7, eL), smoothstep(1.2, 0.7, eR));

  float mOpen = smoothstep(0.05, 0.55, b * 0.6 + m * 0.4 + uEnergy * 0.35);
  vec2 mp = (fp - vec2(0.0, -faceR * 0.28)) / vec2(faceR * (0.28 + 0.1 * m), faceR * (0.04 + 0.22 * mOpen));
  float mFill = smoothstep(1.15, 0.8, dot(mp, mp));
  float teeth = step(abs(fp.y - (-faceR * 0.28 + faceR * 0.14 * mOpen)), faceR * 0.03 * mOpen) * mFill;
  float tongue = smoothstep(1.1, 0.7, dot((fp - vec2(0.0, -faceR * 0.36)) / vec2(faceR * 0.12, faceR * 0.07 * mOpen + 0.001), (fp - vec2(0.0, -faceR * 0.36)) / vec2(faceR * 0.12, faceR * 0.07 * mOpen + 0.001))) * mFill;

  fCol = mix(fCol, vec3(0.05), max(eMask, mFill));
  fCol = mix(fCol, vec3(0.96, 0.25, 0.45), tongue);
  fCol = mix(fCol, vec3(1.0), teeth);
  vec3 col = mix(bgCol, fCol, fMask);
  col = mix(col, 1.0 - col, uKick * 0.45 * tr);
  gl_FragColor = vec4(clamp(psychedelicColor(col), 0.0, 1.0), 1.0);
}`,
};

// SuperAcid Post-Process Compositor: Trails feedback + Kaleido fold + Chromatic Aberration + Glitch + Bloom + Safe Strobe
const POST_FRAG_GLSL = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uSceneTex;
uniform sampler2D uPrevTex;
uniform float uTime;
uniform float uEnergy;
uniform float uKick;
uniform float uFxBloom;
uniform float uFxTrails;
uniform float uFxChroma;
uniform float uFxGlitch;
uniform float uFxStrobe;
uniform float uFxKaleido;
uniform float uBlackout;
uniform float uBpm;

float hash1(float n) { return fract(sin(n) * 43758.5453123); }

void main() {
  if (uBlackout > 0.5) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = vUv;
  // Kaleidoscope mirror post-fold
  if (uFxKaleido > 0.5) {
    vec2 p = uv - 0.5;
    float r = length(p);
    float a = atan(p.y, p.x);
    float sec = 6.2831853 / 8.0;
    a = abs(mod(a + 6.2831853, sec) - sec * 0.5);
    uv = 0.5 + vec2(cos(a), sin(a)) * r;
  }

  // Glitch block horizontal tear
  if (uFxGlitch > 0.5) {
    float band = floor(uv.y * 24.0 + floor(uTime * 16.0));
    float rnd = hash1(band);
    float gate = step(0.82 - uKick * 0.28, rnd);
    uv.x += (hash1(band + 13.0) - 0.5) * 0.08 * gate;
  }

  // Chromatic Aberration (radial RGB split)
  vec3 col;
  if (uFxChroma > 0.5) {
    vec2 dir = uv - 0.5;
    float off = (0.008 + uKick * 0.014 + uEnergy * 0.006);
    col.r = texture2D(uSceneTex, clamp(uv + dir * off, 0.001, 0.999)).r;
    col.g = texture2D(uSceneTex, clamp(uv, 0.001, 0.999)).g;
    col.b = texture2D(uSceneTex, clamp(uv - dir * off, 0.001, 0.999)).b;
  } else {
    col = texture2D(uSceneTex, clamp(uv, 0.001, 0.999)).rgb;
  }

  // Fast 4-tap Bloom glow
  if (uFxBloom > 0.5) {
    float d = 0.006 + uEnergy * 0.005;
    vec3 bSum =
      texture2D(uSceneTex, clamp(uv + vec2(d, 0.0), 0.001, 0.999)).rgb +
      texture2D(uSceneTex, clamp(uv - vec2(d, 0.0), 0.001, 0.999)).rgb +
      texture2D(uSceneTex, clamp(uv + vec2(0.0, d), 0.001, 0.999)).rgb +
      texture2D(uSceneTex, clamp(uv - vec2(0.0, d), 0.001, 0.999)).rgb;
    vec3 bright = max(bSum * 0.25 - 0.35, 0.0);
    col += bright * (0.75 + uKick * 0.5);
  }

  // Ping-pong Zoom Feedback Trails
  if (uFxTrails > 0.5) {
    vec2 prevUv = (vUv - 0.5) * (0.975 - uKick * 0.012) + 0.5;
    vec3 prevCol = texture2D(uPrevTex, clamp(prevUv, 0.001, 0.999)).rgb;
    col = max(col, prevCol * 0.72);
  }

  // Glitch subtle scanlines
  if (uFxGlitch > 0.5) {
    float scan = 0.85 + 0.15 * sin(vUv.y * 420.0 + uTime * 25.0);
    col *= scan;
  }

  // WCAG-safe Strobe (strictly capped <= 2.6 Hz)
  if (uFxStrobe > 0.5) {
    float safeHz = min(2.6, max(1.0, uBpm / 60.0));
    float ph = fract(uTime * safeHz);
    float flash = step(ph, 0.12) * (0.45 + 0.35 * uKick);
    col = mix(col, vec3(1.0), flash);
  }

  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

const DISPLAY_FRAG_GLSL = `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTex;
void main() {
  gl_FragColor = texture2D(uTex, vUv);
}
`;

function compileShader(gl: WebGLRenderingContext, type: number, src: string): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.warn("[Lightshow] Shader compile warning:", gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function linkProgram(gl: WebGLRenderingContext, vertSrc: string, fragSrc: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.warn("[Lightshow] Program link warning:", gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

export class RaveLightshowPanel {
  private readonly canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | null = null;
  private quadBuf: WebGLBuffer | null = null;
  private scenePrograms = new Map<LightshowPresetId, WebGLProgram>();
  private postProgram: WebGLProgram | null = null;
  private displayProgram: WebGLProgram | null = null;
  private sceneFbo: WebGLFramebuffer | null = null;
  private sceneTex: WebGLTexture | null = null;
  private trailFbos: [WebGLFramebuffer | null, WebGLFramebuffer | null] = [null, null];
  private trailTexs: [WebGLTexture | null, WebGLTexture | null] = [null, null];
  private pingIdx = 0;
  private fboW = 0;
  private fboH = 0;

  private state: LightshowState = {
    enabled: true,
    preset: "acid-warp",
    cortexAuto: true,
    intensity: 1.25,
    speed: 1.05,
    palette: 0,
    fxBloom: true,
    fxTrails: true,
    fxChroma: true,
    fxGlitch: false,
    fxStrobe: false,
    fxKaleido: false,
    blackout: false,
  };

  private lastSection = "";
  private startMs = performance.now();
  private onAutoPresetChange: ((preset: LightshowPresetId) => void) | null = null;

  constructor(canvas: HTMLCanvasElement, onAutoPresetChange?: (preset: LightshowPresetId) => void) {
    this.canvas = canvas;
    this.onAutoPresetChange = onAutoPresetChange ?? null;
    this.initGL();
  }

  public getState(): LightshowState {
    return { ...this.state };
  }

  public setState(patch: Partial<LightshowState>): void {
    this.state = { ...this.state, ...patch };
  }

  public pulseStrobeBriefly(ms: number = 320): void {
    const prev = this.state.fxStrobe;
    this.state.fxStrobe = true;
    setTimeout(() => {
      this.state.fxStrobe = prev;
    }, ms);
  }

  private initGL(): void {
    const gl = this.canvas.getContext("webgl", { alpha: false, antialias: false, preserveDrawingBuffer: false });
    if (!gl) return;
    this.gl = gl;

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.quadBuf = buf;

    for (const { id } of LIGHTSHOW_PRESETS) {
      const prog = linkProgram(gl, VERT_SRC, `${SCENE_HEADER_GLSL}\n${PRESET_BODIES[id]}`);
      if (prog) this.scenePrograms.set(id, prog);
    }
    this.postProgram = linkProgram(gl, VERT_SRC, POST_FRAG_GLSL);
    this.displayProgram = linkProgram(gl, VERT_SRC, DISPLAY_FRAG_GLSL);
  }

  private createTarget(gl: WebGLRenderingContext, w: number, h: number): { fbo: WebGLFramebuffer; tex: WebGLTexture } | null {
    const tex = gl.createTexture();
    const fbo = gl.createFramebuffer();
    if (!tex || !fbo) return null;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex };
  }

  private ensureTargets(): boolean {
    const gl = this.gl;
    if (!gl) return false;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(160, Math.round((rect.width || this.canvas.width || 480) * dpr));
    const h = Math.max(120, Math.round((rect.height || this.canvas.height || 300) * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    if (this.fboW === w && this.fboH === h && this.sceneFbo && this.trailFbos[0] && this.trailFbos[1]) {
      return true;
    }
    for (const fb of [this.sceneFbo, this.trailFbos[0], this.trailFbos[1]]) {
      if (fb) gl.deleteFramebuffer(fb);
    }
    for (const tx of [this.sceneTex, this.trailTexs[0], this.trailTexs[1]]) {
      if (tx) gl.deleteTexture(tx);
    }
    const s = this.createTarget(gl, w, h);
    const t0 = this.createTarget(gl, w, h);
    const t1 = this.createTarget(gl, w, h);
    if (!s || !t0 || !t1) return false;
    this.sceneFbo = s.fbo;
    this.sceneTex = s.tex;
    this.trailFbos = [t0.fbo, t1.fbo];
    this.trailTexs = [t0.tex, t1.tex];
    this.fboW = w;
    this.fboH = h;
    return true;
  }

  private drawQuad(gl: WebGLRenderingContext, prog: WebGLProgram): void {
    const loc = gl.getAttribLocation(prog, "a_pos");
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  public render(frame: LightshowAudioFrame): void {
    if (!this.state.enabled) return;
    const gl = this.gl;
    if (!gl || !this.ensureTargets()) return;

    // CORTEX Auto-VJ Director: rotates SuperAcid preset smoothly on musical section shifts
    if (this.state.cortexAuto && frame.section && frame.section !== this.lastSection) {
      this.lastSection = frame.section;
      const idx = LIGHTSHOW_PRESETS.findIndex((p) => p.id === this.state.preset);
      const next = LIGHTSHOW_PRESETS[(Math.max(0, idx) + 1) % LIGHTSHOW_PRESETS.length].id;
      this.state.preset = next;
      this.onAutoPresetChange?.(next);
    }

    const prog = this.scenePrograms.get(this.state.preset) ?? this.scenePrograms.values().next().value;
    if (!prog || !this.postProgram || !this.displayProgram) return;

    const uTime = (performance.now() - this.startMs) * 0.001;
    const beatFrac = ((frame.timeSec * Math.max(60, frame.bpm)) / 60) % 1;
    const bpmSin = Math.sin(frame.timeSec * (Math.max(60, frame.bpm) / 60) * Math.PI * 2);

    // 8 spectral bands from master FFT (or synthesized from stem levels)
    const spec = new Float32Array(8);
    if (frame.freqBins && frame.freqBins.length >= 64) {
      const bins = frame.freqBins;
      const ranges = [
        [0, 3], [3, 8], [8, 16], [16, 32],
        [32, 64], [64, 128], [128, 256], [256, Math.min(512, bins.length)],
      ];
      for (let b = 0; b < 8; b++) {
        let sum = 0;
        const [lo, hi] = ranges[b];
        for (let k = lo; k < hi; k++) sum += bins[k];
        spec[b] = sum / Math.max(1, (hi - lo) * 255);
      }
    } else {
      spec[0] = frame.bass;
      spec[1] = frame.drums * 0.8 + frame.bass * 0.4;
      spec[2] = frame.other * 0.7;
      spec[3] = frame.vocals;
      spec[4] = frame.vocals * 0.8 + frame.other * 0.5;
      spec[5] = frame.other;
      spec[6] = frame.drums * 0.9;
      spec[7] = frame.mix;
    }

    // Pass 1: Render SuperAcid Scene -> sceneFbo
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFbo);
    gl.viewport(0, 0, this.fboW, this.fboH);
    gl.useProgram(prog);
    gl.uniform2f(gl.getUniformLocation(prog, "uRes"), this.fboW, this.fboH);
    gl.uniform1f(gl.getUniformLocation(prog, "uTime"), uTime);
    gl.uniform1f(gl.getUniformLocation(prog, "uBpm"), frame.bpm);
    gl.uniform1f(gl.getUniformLocation(prog, "uBeat"), beatFrac);
    gl.uniform1f(gl.getUniformLocation(prog, "uBpmSin"), bpmSin);
    gl.uniform1f(gl.getUniformLocation(prog, "uKick"), frame.drumPunch);
    gl.uniform1f(gl.getUniformLocation(prog, "uOnset"), frame.drumPunch);
    gl.uniform1f(gl.getUniformLocation(prog, "uEnergy"), frame.mix);
    gl.uniform1f(gl.getUniformLocation(prog, "uIntensity"), this.state.intensity);
    gl.uniform1f(gl.getUniformLocation(prog, "uSpeed"), this.state.speed);
    gl.uniform1f(gl.getUniformLocation(prog, "uPalette"), this.state.palette);
    gl.uniform1f(gl.getUniformLocation(prog, "uChaos"), 0.55);
    for (let i = 0; i < 8; i++) {
      gl.uniform1f(gl.getUniformLocation(prog, `uSpectrum${i}`), spec[i]);
    }
    this.drawQuad(gl, prog);

    // Pass 2: Post-Process (Bloom / Trails / Chroma / Glitch / Kaleido / Strobe) -> next trailFbo
    const prevIdx = this.pingIdx;
    const nextIdx = 1 - prevIdx;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.trailFbos[nextIdx]);
    gl.viewport(0, 0, this.fboW, this.fboH);
    gl.useProgram(this.postProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.postProgram, "uSceneTex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.trailTexs[prevIdx]);
    gl.uniform1i(gl.getUniformLocation(this.postProgram, "uPrevTex"), 1);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uTime"), uTime);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uEnergy"), frame.mix);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uKick"), frame.drumPunch);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uFxBloom"), this.state.fxBloom ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uFxTrails"), this.state.fxTrails ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uFxChroma"), this.state.fxChroma ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uFxGlitch"), this.state.fxGlitch ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uFxStrobe"), this.state.fxStrobe ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uFxKaleido"), this.state.fxKaleido ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uBlackout"), this.state.blackout ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.postProgram, "uBpm"), frame.bpm);
    this.drawQuad(gl, this.postProgram);

    // Pass 3: Blit to screen canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.fboW, this.fboH);
    gl.useProgram(this.displayProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.trailTexs[nextIdx]);
    gl.uniform1i(gl.getUniformLocation(this.displayProgram, "uTex"), 0);
    this.drawQuad(gl, this.displayProgram);

    this.pingIdx = nextIdx;
  }

  public dispose(): void {
    const gl = this.gl;
    if (!gl) return;
    for (const p of this.scenePrograms.values()) gl.deleteProgram(p);
    if (this.postProgram) gl.deleteProgram(this.postProgram);
    if (this.displayProgram) gl.deleteProgram(this.displayProgram);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    for (const fb of [this.sceneFbo, this.trailFbos[0], this.trailFbos[1]]) {
      if (fb) gl.deleteFramebuffer(fb);
    }
    for (const tx of [this.sceneTex, this.trailTexs[0], this.trailTexs[1]]) {
      if (tx) gl.deleteTexture(tx);
    }
  }
}
