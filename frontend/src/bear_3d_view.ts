// Electro-Animatronic Wub Techno Dance Bear 2.0 — Interactive Three.js 3D Stage
// Renders the full dancing cyber-gummi bear with DMT/LSD lighting, interactive
// OrbitControls camera presets, and realistic visible singing mouth anatomy
// (sculpted upper/lower lips, pearl teeth rows, articulated tongue, dark oral
// cavity, and glowing back throat/uvula) driven live by the Singing Head model.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import type { LiveMouthState } from "./singing_head";

export type BearCameraPreset = "three" | "front" | "mouth" | "side" | "orbit";

export const BEAR_CAMERA_PRESETS: readonly BearCameraPreset[] = [
  "three",
  "mouth",
  "front",
  "side",
  "orbit",
];

export const BEAR_CAMERA_LABELS: Record<BearCameraPreset, string> = {
  three: "3/4 HERO",
  mouth: "MOUTH ZOOM",
  front: "FRONT",
  side: "PROFILE",
  orbit: "AUTO ORBIT",
};

export type BearMouthStyle = "real" | "acid" | "cyber";

/**
 * Dual offscreen framebuffer texture pass: two 512x512 psychedelic shader
 * renders per frame (sky sphere + head sphere). It is the most expensive part
 * of the 3D view and pegs a software-GL / low-end machine, so it ships off.
 * With it off the spheres keep the animated gummi palette and the scene
 * lighting; flip this to true to bring the shader-wrapped textures back.
 */
export const SPHERE_TEXTURE_ENABLED = false;

export interface BearVisualConfig {
  showCage: boolean;
  showVisor: boolean;
  showHalo: boolean;
  showBikini: boolean;
  mouthStyle: BearMouthStyle;
  danceGain: number; // 0.3 .. 2.0
  mouthGain: number; // 0.6 .. 1.8
}

export const DEFAULT_BEAR_VISUAL_CONFIG: BearVisualConfig = {
  showCage: true,
  showVisor: true,
  showHalo: true,
  showBikini: true,
  mouthStyle: "real",
  danceGain: 1.0,
  mouthGain: 1.0,
};

export interface Bear3DTelemetry {
  stage: 1 | 2 | 3 | 4;
  bpm: number;
  drums: number;
  bass: number;
  other: number;
  vocals: number;
  mix: number;
  drumPunch: number;
  bassPunch: number;
  otherPunch: number;
  mixPunch: number;
  isBlinking: boolean;
  isDrop: boolean;
  action: string | null;
  mouth: LiveMouthState | null;
  fallbackAperture: number;
  fallbackVowel: string;
  freqBins?: Uint8Array | null;
  visual?: BearVisualConfig;
}

interface DualVizPreset {
  id: string;
  label: string;
  role: "bg" | "fg";
  defaults: Record<string, number>;
  frag: string;
}

const DUAL_OFF_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const DUAL_BUILTIN_PRESETS: DualVizPreset[] = [
  {
    id: "acid-warp-sky",
    label: "ACID WARP",
    role: "bg",
    defaults: {},
    frag: `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform float uKick;
uniform float uEnergy;
uniform float uSpeed;
uniform float uPalette;
uniform float uSpectrum0;
uniform float uSpectrum3;
vec3 hsv2rgb(float h, float s, float v) {
  vec3 c = abs(fract(h + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0;
  return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}
void main() {
  vec2 p = (vUv - 0.5) * 4.0;
  float time = uTime * uSpeed * 1.4;
  float b = (uSpectrum0 + uSpectrum3) * 0.85;
  vec2 p1 = p + vec2(sin(p.y * 2.5 + time * 0.7), cos(p.x * 2.3 + time * 0.6)) * (1.1 + b);
  vec2 p2 = p1 + vec2(sin(p1.y * 3.7 + time * 0.5 + p1.x * 1.3), cos(p1.x * 3.1 - time * 0.4)) * 0.85;
  float pat = sin(p2.x * 4.8 + p2.y * 4.2 + time * 0.35) * 0.5 + 0.5;
  float hue = fract(pat * 0.85 + uPalette * 0.25 + time * 0.05);
  vec3 col = hsv2rgb(hue, 0.92, clamp(pat * 0.8 + 0.28 + uEnergy * 0.35 + uKick * 0.2, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}`,
  },
  {
    id: "chroma-mandala-head",
    label: "CHROMA MANDALA",
    role: "bg",
    defaults: {},
    frag: `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform float uKick;
uniform float uEnergy;
uniform float uSpeed;
uniform float uPalette;
uniform float uSpectrum1;
uniform float uSpectrum5;
vec3 hsv2rgb(float h, float s, float v) {
  vec3 c = abs(fract(h + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0;
  return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}
void main() {
  vec2 p = (vUv - 0.5) * 2.0;
  float r = length(p);
  float a = atan(p.y, p.x);
  float t = uTime * uSpeed;
  float k = abs(sin(a * 8.0 + t * 1.6)) * (0.45 + 0.45 * uSpectrum5);
  float rings = sin(r * 18.0 - t * 3.2 + sin(a * 6.0) * 1.5) * 0.5 + 0.5;
  float grid = smoothstep(0.35, 0.9, rings * (0.6 + k)) + uKick * 0.25;
  float hue = fract(0.55 + r * 0.45 - t * 0.08 + uPalette * 0.25 + uSpectrum1 * 0.2);
  vec3 col = hsv2rgb(hue, 0.95, clamp(grid * (0.85 + 0.45 * uEnergy), 0.18, 1.0));
  gl_FragColor = vec4(col, 1.0);
}`,
  },
  {
    id: "space-tubes-sky",
    label: "SPACE TUBES",
    role: "bg",
    defaults: {},
    frag: `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform float uKick;
uniform float uEnergy;
uniform float uSpeed;
uniform float uPalette;
uniform float uSpectrum0;
uniform float uSpectrum6;
vec3 hsv2rgb(float h, float s, float v) {
  vec3 c = abs(fract(h + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0;
  return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}
void main() {
  vec2 p = (vUv - 0.5) * 2.0;
  float t = uTime * uSpeed;
  float r = length(p) + 0.01;
  float ang = atan(p.y, p.x);
  float z = 1.4 / r + t * (2.2 + uSpectrum0 * 1.8);
  float rib = pow(abs(sin(z * 2.6)), 8.0);
  float rail = smoothstep(0.9, 1.0, abs(sin(ang * 7.0 + sin(z * 0.4) * 2.0)));
  float hue = fract(z * 0.09 + uPalette * 0.25 + t * 0.04);
  vec3 col = mix(vec3(0.03, 0.01, 0.1), hsv2rgb(hue, 0.95, 1.0) * 0.9, rib);
  col += hsv2rgb(fract(hue + 0.42), 0.92, 1.0) * rail * (0.5 + uKick * 0.7 + uSpectrum6 * 0.4);
  gl_FragColor = vec4(clamp(col * (0.85 + 0.45 * uEnergy), 0.0, 1.0), 1.0);
}`,
  },
];

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Smoothly maps vowel label to default mouth shape when SingingHead isn't loaded yet. */
function fallbackMouthParams(vowel: string, ap: number) {
  const v = vowel.toUpperCase();
  let width = 0.55;
  let round = 0.25;
  let jaw = ap * 0.72;
  let tongueX = 0.0;
  let tongueY = 0.0;
  let tipY = 0.0;
  if (v === "EE") {
    width = 0.92;
    round = 0.08;
    jaw = ap * 0.28;
    tongueX = 0.8;
    tongueY = 0.75;
    tipY = 0.45;
  } else if (v === "OH") {
    width = 0.32;
    round = 0.9;
    jaw = ap * 0.56;
    tongueX = -0.5;
    tongueY = 0.35;
  } else if (v === "OO") {
    width = 0.2;
    round = 0.98;
    jaw = ap * 0.2;
    tongueX = 0.15;
    tongueY = 0.85;
  } else if (v === "AY") {
    width = 0.8;
    round = 0.15;
    jaw = ap * 0.52;
    tongueX = 0.45;
    tongueY = 0.35;
  } else if (v === "AH") {
    width = 0.62;
    round = 0.12;
    jaw = ap * 0.9;
    tongueX = -0.1;
    tongueY = -0.45;
  }
  return { width, round, jaw, tongueX, tongueY, tipY };
}

interface LipParams {
  halfW: number;
  upperH: number;
  lowerH: number;
  protrude: number;
  roundness: number;
  tuck: number;
  thick: number;
}

/** Updates a tubular lip-arc mesh in-place (zero allocations per frame). */
function updateLipTubeGeometry(
  geo: THREE.BufferGeometry,
  isUpper: boolean,
  p: LipParams,
  uSegs: number,
  vSegs: number
): void {
  const posAttr = geo.getAttribute("position") as THREE.BufferAttribute;
  const arr = posAttr.array as Float32Array;
  const cxArr = new Float32Array(uSegs + 1);
  const cyArr = new Float32Array(uSegs + 1);
  const czArr = new Float32Array(uSegs + 1);
  const radArr = new Float32Array(uSegs + 1);

  for (let i = 0; i <= uSegs; i++) {
    const t = i / uSegs; // 0..1 left corner to right corner
    const sx = (t - 0.5) * 2; // -1..1
    const arcEnv = Math.pow(Math.max(0, Math.cos(sx * Math.PI * 0.5)), 0.92); // 0 at corners, 1 center
    const x = sx * p.halfW;
    // Slight wrap back around the snout curvature
    const zBase = p.protrude * Math.pow(arcEnv, 1.35) - 0.16 * (sx * sx);

    let y: number;
    let r: number;
    if (isUpper) {
      // Sculpted M-shaped Cupid's bow: center tubercle dip + twin philtrum peaks
      const cleft = Math.exp(-Math.pow(sx * 4.4, 2));
      const peaks = Math.exp(-Math.pow((Math.abs(sx) - 0.34) * 4.2, 2));
      const bow = (peaks * 0.052 - cleft * 0.032) * (1 - 0.55 * p.roundness) * arcEnv;
      y = p.upperH * Math.pow(arcEnv, 0.85) + bow;
      r = p.thick * (0.12 + 0.96 * Math.pow(arcEnv, 0.85));
    } else {
      const tuckDrop = -p.tuck * 0.14 * arcEnv;
      y = -p.lowerH * Math.pow(arcEnv, 0.82) - 0.015 * arcEnv;
      r = p.thick * 1.16 * (0.12 + 1.02 * Math.pow(arcEnv, 0.85));
      czArr[i] = zBase + tuckDrop;
      cxArr[i] = x;
      cyArr[i] = y;
      radArr[i] = r;
      continue;
    }
    cxArr[i] = x;
    cyArr[i] = y;
    czArr[i] = zBase;
    radArr[i] = r;
  }

  let ptr = 0;
  for (let i = 0; i <= uSegs; i++) {
    const iPrev = Math.max(0, i - 1);
    const iNext = Math.min(uSegs, i + 1);
    let tx = cxArr[iNext] - cxArr[iPrev];
    let ty = cyArr[iNext] - cyArr[iPrev];
    let tz = czArr[iNext] - czArr[iPrev];
    const tLen = Math.hypot(tx, ty, tz) || 1;
    tx /= tLen;
    ty /= tLen;
    tz /= tLen;

    // Binormal points mainly +Z (outward from mouth), normal ~ perpendicular in YZ
    let nx = -ty * 0.0 - tz * 0.0;
    let ny = tz * tx - tx * tz;
    // Construct clean stable frame using front vector (0, 0, 1) crossed with tangent
    const fx = 0;
    const fy = 0;
    const fz = 1;
    // N = normalize(F x T)
    nx = fy * tz - fz * ty;
    ny = fz * tx - fx * tz;
    let nz = fx * ty - fy * tx;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;
    // B = T x N
    const bx = ty * nz - tz * ny;
    const by = tz * nx - tx * nz;
    const bz = tx * ny - ty * nx;

    const r = radArr[i];
    for (let j = 0; j <= vSegs; j++) {
      const phi = (j / vSegs) * Math.PI * 2;
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      arr[ptr++] = cxArr[i] + r * (c * nx + s * bx);
      arr[ptr++] = cyArr[i] + r * (c * ny + s * by);
      arr[ptr++] = czArr[i] + r * (c * nz + s * bz);
    }
  }
  posAttr.needsUpdate = true;
  geo.computeVertexNormals();
}

function createLipTubeGeometry(uSegs: number, vSegs: number): THREE.BufferGeometry {
  const vertCount = (uSegs + 1) * (vSegs + 1);
  const pos = new Float32Array(vertCount * 3);
  const idx: number[] = [];
  for (let i = 0; i < uSegs; i++) {
    for (let j = 0; j < vSegs; j++) {
      const a = i * (vSegs + 1) + j;
      const b = (i + 1) * (vSegs + 1) + j;
      const c = (i + 1) * (vSegs + 1) + j + 1;
      const d = i * (vSegs + 1) + j + 1;
      idx.push(a, b, d, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(idx);
  return geo;
}

/** Articulated curved 3D tongue grid updated in-place from SingingHead articulators. */
function createTongueGeometry(nx: number, nz: number): THREE.BufferGeometry {
  const pos = new Float32Array((nx + 1) * (nz + 1) * 3);
  const idx: number[] = [];
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      const a = i * (nz + 1) + j;
      const b = (i + 1) * (nz + 1) + j;
      const c = (i + 1) * (nz + 1) + j + 1;
      const d = i * (nz + 1) + j + 1;
      idx.push(a, b, d, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(idx);
  return geo;
}

function updateTongueGeometry(
  geo: THREE.BufferGeometry,
  nx: number,
  nz: number,
  tongueX: number,
  tongueY: number,
  tipX: number,
  tipY: number,
  tongueSides: number,
  groove: number,
  jawOpen: number
): void {
  const posAttr = geo.getAttribute("position") as THREE.BufferAttribute;
  const arr = posAttr.array as Float32Array;
  let ptr = 0;
  for (let iz = 0; iz <= nz; iz++) {
    const v = iz / nz; // 0 = back root in pharynx, 1 = front tongue tip
    const taper = v < 0.72 ? 1.0 : Math.cos(((v - 0.72) / 0.28) * Math.PI * 0.44);
    const halfSpan = (0.44 - 0.08 * clamp(tongueSides, 0, 1)) * Math.max(0.28, taper);
    // Smooth S-arch from root -> dorsum -> tip
    const bodyArch = Math.sin(v * Math.PI * 0.92);
    const tipMask = Math.pow(clamp((v - 0.55) / 0.45, 0, 1), 1.6);
    const zBase = -0.68 + v * 0.98 + tongueX * 0.14 * (1 - 0.3 * tipMask) + tipX * 0.12 * tipMask;
    const baseY =
      -0.28 -
      jawOpen * 0.16 +
      tongueY * 0.22 * bodyArch +
      tipY * 0.22 * tipMask -
      0.08 * v * v;

    for (let ix = 0; ix <= nx; ix++) {
      const u = (ix / nx - 0.5) * 2; // -1..1 across width
      const x = u * halfSpan;
      const edgeDrop = -0.11 * u * u - tongueSides * 0.07 * u * u;
      const sulcus = -(0.045 + 0.075 * groove) * Math.exp(-u * u * 10) * Math.sin(v * Math.PI);
      const y = baseY + edgeDrop + sulcus;
      arr[ptr++] = x;
      arr[ptr++] = y;
      arr[ptr++] = zBase - 0.04 * (u * u);
    }
  }
  posAttr.needsUpdate = true;
  geo.computeVertexNormals();
}

export class Bear3DView {
  private readonly host: HTMLElement;
  private readonly disposables: Array<{ dispose: () => void }> = [];
  private renderer: THREE.WebGLRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;
  private ro: ResizeObserver | null = null;
  private active = false;
  private initialized = false;
  private webglFailed = false;
  private preset: BearCameraPreset = "three";
  private onPresetChange: ((preset: BearCameraPreset) => void) | null = null;

  private readonly hudEl: HTMLDivElement;
  private readonly presetBtns = new Map<BearCameraPreset, HTMLButtonElement>();
  private readonly statusBanner: HTMLDivElement;

  // Rig groups & reactive meshes (floating circle/sphere Wub Bear head only)
  private stageRoot = new THREE.Group();
  private bodyGroup = new THREE.Group();
  private headGroup = new THREE.Group();
  private headMesh: THREE.Mesh | null = null;
  private skyMesh: THREE.Mesh | null = null;
  private leftEar = new THREE.Group();
  private rightEar = new THREE.Group();
  private leftEye = new THREE.Group();
  private rightEye = new THREE.Group();
  private leftIris: THREE.Mesh | null = null;
  private rightIris: THREE.Mesh | null = null;
  private leftPupil: THREE.Mesh | null = null;
  private rightPupil: THREE.Mesh | null = null;
  private visorMesh: THREE.Mesh | null = null;
  private haloMesh: THREE.Mesh | null = null;
  private earBowGroup: THREE.Group | null = null;
  private bikiniMat: THREE.MeshPhysicalMaterial | null = null;
  private mandalaFloor: THREE.Group | null = null;
  private mandalaBack: THREE.Group | null = null;
  private moteRing: THREE.Group | null = null;

  // Dual offscreen framebuffers: contrasting sky animated texture + sphere head texture
  private skyTarget: THREE.WebGLRenderTarget | null = null;
  private headTarget: THREE.WebGLRenderTarget | null = null;
  private offScene = new THREE.Scene();
  private offCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private offQuadGeo = new THREE.PlaneGeometry(2, 2);
  private offQuadMesh: THREE.Mesh | null = null;
  private vizPresets: DualVizPreset[] = [...DUAL_BUILTIN_PRESETS];
  private vizHeaders: Record<string, string> = {};
  private vizMatCache = new Map<string, THREE.ShaderMaterial>();
  private skyPresetId = "space-tubes-sky";
  private headPresetId = "chroma-mandala-head";
  private lastTexSwapSec = 0;
  private dualStartMs = performance.now();
  private readonly dualUniforms: Record<string, THREE.IUniform<number>> = {
    uTime: { value: 0 },
    uAudioTime: { value: 0 },
    uBpm: { value: 126 },
    uBeat: { value: 0 },
    uBpmSin: { value: 0 },
    uBpmSin2: { value: 0 },
    uBpmSin4: { value: 0 },
    uKick: { value: 0 },
    uOnset: { value: 0 },
    uEnergy: { value: 0 },
    uIntensity: { value: 1.15 },
    uSpeed: { value: 1.0 },
    uPalette: { value: 0 },
    uWarmth: { value: 0 },
    uBrightness: { value: 0 },
    uMusicDensity: { value: 0.4 },
    uForwardFlight: { value: 0.3 },
    uSpectrum0: { value: 0 },
    uSpectrum1: { value: 0 },
    uSpectrum2: { value: 0 },
    uSpectrum3: { value: 0 },
    uSpectrum4: { value: 0 },
    uSpectrum5: { value: 0 },
    uSpectrum6: { value: 0 },
    uSpectrum7: { value: 0 },
    uBandFlux0: { value: 0 },
    uBandFlux1: { value: 0 },
    uBandFlux2: { value: 0 },
    uBandFlux3: { value: 0 },
    uBandFlux4: { value: 0 },
    uBandFlux5: { value: 0 },
    uBandFlux6: { value: 0 },
    uBandFlux7: { value: 0 },
  };

  // Realistic mouth components
  private mouthAnchor = new THREE.Group();
  private lowerJawGroup = new THREE.Group();
  private upperLipGeo: THREE.BufferGeometry | null = null;
  private lowerLipGeo: THREE.BufferGeometry | null = null;
  private tongueGeo: THREE.BufferGeometry | null = null;
  private readonly lipUSegs = 28;
  private readonly lipVSegs = 10;
  private readonly tongueNx = 14;
  private readonly tongueNz = 18;
  private upperTeeth: THREE.InstancedMesh | null = null;
  private lowerTeeth: THREE.InstancedMesh | null = null;
  private uvulaMesh: THREE.Mesh | null = null;
  private velumMesh: THREE.Mesh | null = null;
  private throatMesh: THREE.Mesh | null = null;
  private throatLight: THREE.PointLight | null = null;
  private readonly dummy = new THREE.Object3D();

  // Dynamic materials for psychedelic DMT/LSD spectrum shifting
  private gummiMat: THREE.MeshPhysicalMaterial | null = null;
  private cageMat: THREE.MeshBasicMaterial | null = null;
  private muzzleMat: THREE.MeshStandardMaterial | null = null;
  private lipMat: THREE.MeshPhysicalMaterial | null = null;
  private tongueMat: THREE.MeshPhysicalMaterial | null = null;
  private throatMat: THREE.MeshStandardMaterial | null = null;
  private visorMat: THREE.MeshStandardMaterial | null = null;
  private earPadMat: THREE.MeshStandardMaterial | null = null;
  private eyeIrisMatL: THREE.MeshStandardMaterial | null = null;
  private eyeIrisMatR: THREE.MeshStandardMaterial | null = null;
  private discoKeyLight: THREE.PointLight | null = null;
  private discoRimLight: THREE.PointLight | null = null;

  private phase = 0;
  private lastMs = 0;
  private readonly tmpColor = new THREE.Color();

  constructor(host: HTMLElement, onPresetChange?: (preset: BearCameraPreset) => void) {
    this.host = host;
    this.onPresetChange = onPresetChange ?? null;

    this.hudEl = document.createElement("div");
    this.hudEl.className = "bear3d-hud";
    for (const p of BEAR_CAMERA_PRESETS) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "bear3d-cam-btn";
      btn.textContent = p === "three" ? "¾" : p === "mouth" ? "MOUTH" : p.toUpperCase();
      btn.title = `3D Camera: ${BEAR_CAMERA_LABELS[p]}`;
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.setCameraPreset(p);
      });
      this.presetBtns.set(p, btn);
      this.hudEl.appendChild(btn);
    }
    this.host.appendChild(this.hudEl);

    this.statusBanner = document.createElement("div");
    this.statusBanner.className = "bear3d-status";
    this.statusBanner.style.display = "none";
    this.host.appendChild(this.statusBanner);
    this.syncPresetButtons();
  }

  public isActive(): boolean {
    return this.active;
  }

  public getCameraPreset(): BearCameraPreset {
    return this.preset;
  }

  public cycleCameraPreset(): BearCameraPreset {
    const idx = BEAR_CAMERA_PRESETS.indexOf(this.preset);
    const next = BEAR_CAMERA_PRESETS[(idx + 1) % BEAR_CAMERA_PRESETS.length];
    this.setCameraPreset(next);
    return next;
  }

  public setCameraPreset(preset: BearCameraPreset): void {
    this.preset = preset;
    this.syncPresetButtons();
    this.onPresetChange?.(preset);
    if (!this.camera || !this.controls) return;
    const d = 4.15;
    if (preset === "mouth") {
      this.camera.position.set(0.12, -0.1, 2.38);
      this.controls.target.set(0, -0.17, 1.05);
    } else if (preset === "front") {
      this.camera.position.set(0, 0.08, d);
      this.controls.target.set(0, 0, 0);
    } else if (preset === "side") {
      this.camera.position.set(3.95, 0.14, 1.15);
      this.controls.target.set(0, 0, 0.15);
    } else if (preset === "orbit") {
      this.camera.position.set(2.15, 0.55, 3.65);
      this.controls.target.set(0, 0, 0);
    } else {
      // "three" (hero 3/4 floating sphere head zoomed to fill screen)
      this.camera.position.set(1.38, 0.42, 3.85);
      this.controls.target.set(0, 0, 0.05);
    }
    this.controls.update();
  }

  public randomizeTextures(): void {
    const bgList = this.vizPresets.filter((p) => p.role === "bg");
    const all = bgList.length >= 2 ? bgList : this.vizPresets;
    if (all.length === 0) return;
    const skyIdx = Math.floor(Math.random() * all.length);
    let headIdx = Math.floor(Math.random() * all.length);
    if (all.length > 1 && headIdx === skyIdx) {
      headIdx = (skyIdx + 1 + Math.floor(Math.random() * (all.length - 1))) % all.length;
    }
    this.skyPresetId = all[skyIdx].id;
    this.headPresetId = all[headIdx].id;
    this.lastTexSwapSec = (performance.now() - this.dualStartMs) * 0.001;
  }

  private loadPresetBundle(): void {
    void fetch("/data/superacid_presets.json")
      .then((res) => (res.ok ? res.json() : null))
      .then((doc: { headers?: Record<string, string>; presets?: DualVizPreset[] } | null) => {
        if (!doc || !Array.isArray(doc.presets) || doc.presets.length === 0) return;
        this.vizHeaders = doc.headers ?? {};
        this.vizPresets = [...DUAL_BUILTIN_PRESETS, ...doc.presets];
        this.randomizeTextures();
      })
      .catch(() => {
        // Keep built-in shaders if bundle unavailable
      });
  }

  private getDualShaderMaterial(presetId: string): THREE.ShaderMaterial | null {
    const cached = this.vizMatCache.get(presetId);
    if (cached) return cached;
    const meta = this.vizPresets.find((p) => p.id === presetId) ?? DUAL_BUILTIN_PRESETS[0];
    if (!meta) return null;
    let frag = meta.frag;
    for (const [token, hdr] of Object.entries(this.vizHeaders)) {
      frag = frag.replace(token, hdr);
    }
    if (!frag.includes("precision ")) {
      frag = `precision highp float;\n${frag}`;
    }
    const uniforms: Record<string, THREE.IUniform<number>> = { ...this.dualUniforms };
    for (const [k, v] of Object.entries(meta.defaults || {})) {
      if (!(k in uniforms)) {
        uniforms[k] = { value: Number.isFinite(v) ? v : 0.6 };
      }
    }
    const mat = new THREE.ShaderMaterial({
      vertexShader: DUAL_OFF_VERT,
      fragmentShader: frag,
      uniforms,
      depthTest: false,
      depthWrite: false,
    });
    if (this.vizMatCache.size >= 16) {
      const oldest = this.vizMatCache.keys().next().value;
      if (oldest) {
        this.vizMatCache.get(oldest)?.dispose();
        this.vizMatCache.delete(oldest);
      }
    }
    this.vizMatCache.set(presetId, mat);
    return mat;
  }

  private syncPresetButtons(): void {
    for (const [p, btn] of this.presetBtns) {
      btn.classList.toggle("active", p === this.preset);
    }
  }

  public setActive(active: boolean): void {
    this.active = active;
    this.host.style.display = active ? "block" : "none";
    if (active && !this.initialized && !this.webglFailed) {
      this.initWebGL();
    }
    if (active) {
      this.syncSize();
    }
  }

  private track<T extends { dispose: () => void }>(res: T): T {
    this.disposables.push(res);
    return res;
  }

  private initWebGL(): void {
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      this.webglFailed = true;
      this.statusBanner.textContent = "WebGL unavailable — switch to 2D Bear.";
      this.statusBanner.style.display = "block";
      return;
    }
    this.initialized = true;
    this.renderer = renderer;
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(0x050112, 1);

    const dom = renderer.domElement;
    dom.className = "bear3d-canvas";
    dom.setAttribute("aria-label", "Electro-Animatronic Wub Bear 3D Camera View");
    this.host.prepend(dom);

    dom.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.statusBanner.textContent = "WebGL context lost — toggle 2D/3D to restore.";
      this.statusBanner.style.display = "block";
    });

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1.4, 0.1, 120);
    this.scene = scene;
    this.camera = camera;

    // Dual offscreen framebuffers (512x512): sky shader texture vs contrasting sphere shader texture.
    // Skipped entirely while SPHERE_TEXTURE_ENABLED is false — no render
    // targets, no fullscreen shader quad, no preset bundle fetch.
    if (SPHERE_TEXTURE_ENABLED) {
      const fbOpts: THREE.RenderTargetOptions = {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
      };
      this.skyTarget = new THREE.WebGLRenderTarget(512, 512, fbOpts);
      this.headTarget = new THREE.WebGLRenderTarget(512, 512, fbOpts);
      this.skyTarget.texture.wrapS = THREE.RepeatWrapping;
      this.skyTarget.texture.wrapT = THREE.ClampToEdgeWrapping;
      this.headTarget.texture.wrapS = THREE.RepeatWrapping;
      this.headTarget.texture.wrapT = THREE.ClampToEdgeWrapping;
      this.offQuadMesh = new THREE.Mesh(this.offQuadGeo, this.getDualShaderMaterial(this.skyPresetId)!);
      this.offScene.add(this.offQuadMesh);
      this.loadPresetBundle();
    }

    // Animated Psychedelic Sky Sphere (shader-textured, or flat night sky)
    const skyGeo = this.track(new THREE.SphereGeometry(28, 40, 28));
    const skyMat = this.track(
      SPHERE_TEXTURE_ENABLED
        ? new THREE.MeshBasicMaterial({
            map: this.skyTarget!.texture,
            side: THREE.BackSide,
            depthWrite: false,
          })
        : new THREE.MeshBasicMaterial({
            color: 0x0a0420,
            side: THREE.BackSide,
            depthWrite: false,
          })
    );
    this.skyMesh = new THREE.Mesh(skyGeo, skyMat);
    scene.add(this.skyMesh);

    // DMT / Techno neon 3-point + point-disco lighting
    scene.add(new THREE.HemisphereLight(0xd8b4fe, 0x1e0638, 1.35));
    const key = new THREE.DirectionalLight(0xa5f3fc, 1.35);
    key.position.set(2.5, 4.0, 5.2);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xff4fd8, 1.25);
    rim.position.set(-4.0, 2.2, -3.5);
    scene.add(rim);

    this.discoKeyLight = new THREE.PointLight(0x00ffff, 2.2, 14);
    this.discoKeyLight.position.set(-2.4, 2.5, 3.2);
    scene.add(this.discoKeyLight);

    this.discoRimLight = new THREE.PointLight(0xff00e6, 2.4, 14);
    this.discoRimLight.position.set(2.6, 1.8, -2.5);
    scene.add(this.discoRimLight);

    this.buildMandalaStage(scene);
    this.buildCyberBearRig(scene);

    const controls = new OrbitControls(camera, dom);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.minDistance = 2.0;
    controls.maxDistance = 12.0;
    this.controls = controls;

    this.setCameraPreset(this.preset);

    this.ro = new ResizeObserver(() => this.syncSize());
    this.ro.observe(this.host);
    this.syncSize();
  }

  private buildMandalaStage(scene: THREE.Scene): void {
    const floor = new THREE.Group();
    floor.position.set(0, -1.72, 0);

    const ringSpecs: Array<[number, number, number, number]> = [
      [2.65, 0.025, 48, 0x00ffff],
      [2.05, 0.022, 6, 0xff00e6],
      [1.45, 0.02, 8, 0xffe600],
      [0.95, 0.018, 32, 0x00ffaa],
    ];
    for (const [rad, tube, radialSegs, col] of ringSpecs) {
      const geo = this.track(new THREE.TorusGeometry(rad, tube, 8, radialSegs));
      const mat = this.track(
        new THREE.MeshBasicMaterial({
          color: col,
          transparent: true,
          opacity: 0.72,
        })
      );
      const m = new THREE.Mesh(geo, mat);
      m.rotation.x = Math.PI / 2;
      floor.add(m);
    }
    scene.add(floor);
    this.mandalaFloor = floor;

    // Sacred geometry DMT prism frame rotating behind the floating head
    const back = new THREE.Group();
    back.position.set(0, 0, -2.6);
    for (let k = 0; k < 2; k++) {
      const geo = this.track(new THREE.TorusGeometry(2.15 + k * 0.55, 0.028, 8, k === 0 ? 6 : 12));
      const mat = this.track(
        new THREE.MeshBasicMaterial({
          color: k === 0 ? 0xff4fd8 : 0x00f0ff,
          transparent: true,
          opacity: 0.48,
        })
      );
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.z = k * 0.4;
      back.add(ring);
    }
    scene.add(back);
    this.mandalaBack = back;

    // Orbiting LSD neon motes
    const motes = new THREE.Group();
    motes.position.set(0, 0, 0);
    const moteGeo = this.track(new THREE.OctahedronGeometry(0.08, 0));
    const moteColors = [0x00ffff, 0xff00e6, 0xffe600, 0x00ffaa, 0xa855f7, 0xff3b9a];
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const mat = this.track(
        new THREE.MeshBasicMaterial({
          color: moteColors[i % moteColors.length],
          transparent: true,
          opacity: 0.85,
        })
      );
      const mote = new THREE.Mesh(moteGeo, mat);
      const r = 2.15 + (i % 3) * 0.42;
      mote.position.set(Math.cos(a) * r, ((i % 4) - 1.5) * 0.65, Math.sin(a) * r);
      motes.add(mote);
    }
    scene.add(motes);
    this.moteRing = motes;
  }

  private buildCyberBearRig(scene: THREE.Scene): void {
    this.stageRoot = new THREE.Group();
    scene.add(this.stageRoot);

    this.gummiMat = this.track(
      new THREE.MeshPhysicalMaterial({
        color: 0xffffff,
        map: SPHERE_TEXTURE_ENABLED ? this.headTarget?.texture ?? null : null,
        emissive: 0xffffff,
        emissiveMap: SPHERE_TEXTURE_ENABLED ? this.headTarget?.texture ?? null : null,
        emissiveIntensity: 0.72,
        roughness: 0.14,
        metalness: 0.08,
        clearcoat: 1.0,
        clearcoatRoughness: 0.08,
      })
    );

    this.cageMat = this.track(
      new THREE.MeshBasicMaterial({
        color: 0x00f0ff,
        wireframe: true,
        transparent: true,
        opacity: 0.28,
      })
    );

    this.muzzleMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xffdaf4,
        emissive: 0x5c1854,
        emissiveIntensity: 0.42,
        roughness: 0.24,
        metalness: 0.08,
      })
    );

    this.bikiniMat = this.track(
      new THREE.MeshPhysicalMaterial({
        color: 0xff1a8c,
        emissive: 0x99004d,
        emissiveIntensity: 0.65,
        roughness: 0.16,
        metalness: 0.25,
        clearcoat: 1.0,
        clearcoatRoughness: 0.1,
        side: THREE.DoubleSide,
      })
    );

    // --- Floating Circle/Sphere Wub Bear Head ONLY (body removed) ---
    this.bodyGroup = new THREE.Group();
    this.stageRoot.add(this.bodyGroup);

    this.headGroup = new THREE.Group();
    this.headGroup.position.set(0, 0, 0);
    this.bodyGroup.add(this.headGroup);

    const headGeo = this.track(new THREE.SphereGeometry(1.16, 48, 36));
    const headMesh = new THREE.Mesh(headGeo, this.gummiMat);
    headMesh.scale.set(1.06, 1.0, 0.96);
    this.headGroup.add(headMesh);
    this.headMesh = headMesh;

    const headCage = new THREE.Mesh(headGeo, this.cageMat);
    headCage.scale.set(1.1, 1.04, 1.0);
    this.headGroup.add(headCage);

    // Ears
    const earGeo = this.track(new THREE.SphereGeometry(0.36, 18, 14));
    const earPadGeo = this.track(new THREE.SphereGeometry(0.22, 14, 12));
    this.earPadMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xff77e9,
        emissive: 0xff00c8,
        emissiveIntensity: 0.9,
        roughness: 0.2,
      })
    );
    const goldMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xffd700,
        emissive: 0xffaa00,
        emissiveIntensity: 0.85,
        metalness: 0.5,
        roughness: 0.2,
      })
    );
    const earRimGeo = this.track(new THREE.TorusGeometry(0.34, 0.032, 8, 24));

    const makeEar = (sign: number): THREE.Group => {
      const ear = new THREE.Group();
      ear.position.set(sign * 0.88, 0.86, -0.04);
      ear.rotation.z = sign * -0.42;
      const outer = new THREE.Mesh(earGeo, this.gummiMat!);
      outer.scale.set(1.0, 1.0, 0.52);
      ear.add(outer);
      const pad = new THREE.Mesh(earPadGeo, this.earPadMat!);
      pad.position.set(0, 0, 0.13);
      pad.scale.set(0.95, 0.95, 0.35);
      ear.add(pad);
      const rim = new THREE.Mesh(earRimGeo, goldMat);
      ear.add(rim);
      this.headGroup.add(ear);
      return ear;
    };
    this.leftEar = makeEar(-1);
    this.rightEar = makeEar(1);

    // Kawaii ear ribbon bow on right ear
    if (this.bikiniMat) {
      const earBow = new THREE.Group();
      earBow.position.set(0.18, 0.3, 0.12);
      earBow.rotation.z = -0.35;
      const bowLoopGeo = this.track(new THREE.ConeGeometry(0.09, 0.2, 8));
      for (const ls of [-1, 1]) {
        const loop = new THREE.Mesh(bowLoopGeo, this.bikiniMat);
        loop.rotation.z = ls * 1.1;
        loop.position.x = ls * 0.09;
        earBow.add(loop);
      }
      this.rightEar.add(earBow);
      this.earBowGroup = earBow;
    }

    // Floating sacred crown halo
    const haloGeo = this.track(new THREE.TorusGeometry(0.84, 0.036, 10, 36));
    this.haloMesh = new THREE.Mesh(haloGeo, goldMat);
    this.haloMesh.rotation.x = Math.PI / 2 - 0.14;
    this.haloMesh.position.set(0, 1.44, 0);
    this.headGroup.add(this.haloMesh);

    // Eyes
    const scleraGeo = this.track(new THREE.SphereGeometry(0.18, 16, 14));
    const scleraMat = this.track(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.15 }));
    const irisGeo = this.track(new THREE.SphereGeometry(0.125, 14, 12));
    this.eyeIrisMatL = this.track(
      new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x00b8ff,
        emissiveIntensity: 0.95,
        roughness: 0.2,
      })
    );
    this.eyeIrisMatR = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xffe600,
        emissive: 0xffaa00,
        emissiveIntensity: 0.95,
        roughness: 0.2,
      })
    );
    const pupilGeo = this.track(new THREE.SphereGeometry(0.072, 12, 10));
    const darkMat = this.track(new THREE.MeshStandardMaterial({ color: 0x100222, roughness: 0.12 }));
    const sparkGeo = this.track(new THREE.SphereGeometry(0.042, 8, 8));
    const sparkMat = this.track(new THREE.MeshBasicMaterial({ color: 0xffffff }));

    const makeEye = (sign: number, irisMat: THREE.MeshStandardMaterial): THREE.Group => {
      const eye = new THREE.Group();
      eye.position.set(sign * 0.44, 0.24, 0.99);
      const sclera = new THREE.Mesh(scleraGeo, scleraMat);
      sclera.scale.set(1.0, 1.15, 0.65);
      eye.add(sclera);
      const iris = new THREE.Mesh(irisGeo, irisMat);
      iris.position.set(0, 0, 0.08);
      iris.scale.set(1.0, 1.05, 0.55);
      eye.add(iris);
      const pupil = new THREE.Mesh(pupilGeo, darkMat);
      pupil.position.set(0, 0, 0.125);
      pupil.scale.set(1.0, 1.05, 0.5);
      eye.add(pupil);
      const sp = new THREE.Mesh(sparkGeo, sparkMat);
      sp.position.set(-0.045, 0.055, 0.155);
      eye.add(sp);
      if (sign < 0) {
        this.leftIris = iris;
        this.leftPupil = pupil;
      } else {
        this.rightIris = iris;
        this.rightPupil = pupil;
      }
      this.headGroup.add(eye);
      return eye;
    };
    this.leftEye = makeEye(-1, this.eyeIrisMatL);
    this.rightEye = makeEye(1, this.eyeIrisMatR);

    // Translucent cyberpunk electro visor arc framing forehead/eyes
    this.visorMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x00d4ff,
        emissiveIntensity: 0.7,
        roughness: 0.1,
        metalness: 0.5,
        transparent: true,
        opacity: 0.26,
        side: THREE.DoubleSide,
      })
    );
    const visorGeo = this.track(
      new THREE.CylinderGeometry(1.25, 1.25, 0.3, 24, 1, true, -0.78, 1.56)
    );
    this.visorMesh = new THREE.Mesh(visorGeo, this.visorMat);
    this.visorMesh.position.set(0, 0.25, -0.02);
    this.headGroup.add(this.visorMesh);

    // Rosy cyber cheek glow LEDs
    const cheekGeo = this.track(new THREE.SphereGeometry(0.14, 12, 10));
    const cheekMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xff3b9a,
        emissive: 0xff0080,
        emissiveIntensity: 0.9,
        transparent: true,
        opacity: 0.72,
      })
    );
    for (const s of [-1, 1]) {
      const cheek = new THREE.Mesh(cheekGeo, cheekMat);
      cheek.position.set(s * 0.68, -0.04, 0.89);
      cheek.scale.set(1.2, 0.78, 0.45);
      this.headGroup.add(cheek);
    }

    // Upper snout/muzzle dome + glossy nose + cute whisker puffs cleanly above upper lip
    const snoutGeo = this.track(new THREE.SphereGeometry(0.52, 22, 18));
    const upperSnout = new THREE.Mesh(snoutGeo, this.muzzleMat);
    upperSnout.position.set(0, 0.01, 0.84);
    upperSnout.scale.set(1.12, 0.64, 0.72);
    this.headGroup.add(upperSnout);

    const puffGeo = this.track(new THREE.SphereGeometry(0.16, 14, 12));
    for (const s of [-1, 1]) {
      const puff = new THREE.Mesh(puffGeo, this.muzzleMat);
      puff.position.set(s * 0.16, 0.03, 1.2);
      puff.scale.set(1.05, 0.82, 0.68);
      this.headGroup.add(puff);
    }

    const noseGeo = this.track(new THREE.SphereGeometry(0.14, 16, 12));
    const nose = new THREE.Mesh(noseGeo, darkMat);
    nose.position.set(0, 0.12, 1.3);
    nose.scale.set(1.18, 0.8, 0.88);
    this.headGroup.add(nose);
    const noseSpark = new THREE.Mesh(sparkGeo, sparkMat);
    noseSpark.position.set(-0.04, 0.16, 1.4);
    this.headGroup.add(noseSpark);

    // -------------------------------------------------------------------------
    // REALISTIC VISIBLE SINGING MOUTH ANATOMY ON THE BEAR'S FACE
    // (Lips, teeth rows, articulated 3D tongue, oral cavity & glowing back throat)
    // -------------------------------------------------------------------------
    this.mouthAnchor = new THREE.Group();
    this.mouthAnchor.position.set(0, -0.17, 1.24);
    this.headGroup.add(this.mouthAnchor);

    // Hinged lower chin/jaw cup (set cleanly below lip rim so zero occlusion)
    this.lowerJawGroup = new THREE.Group();
    this.lowerJawGroup.position.set(0, 0.01, -0.16);
    this.mouthAnchor.add(this.lowerJawGroup);

    const chinGeo = this.track(new THREE.SphereGeometry(0.42, 20, 14));
    const chinMesh = new THREE.Mesh(chinGeo, this.muzzleMat);
    chinMesh.position.set(0, -0.25, -0.12);
    chinMesh.scale.set(1.1, 0.5, 0.76);
    this.lowerJawGroup.add(chinMesh);

    // Dark oral cavity interior sphere + luminous back pharynx/throat
    const cavityMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0x260516,
        emissive: 0x180210,
        emissiveIntensity: 0.6,
        roughness: 0.65,
        side: THREE.BackSide,
      })
    );
    const cavityGeo = this.track(new THREE.SphereGeometry(0.42, 18, 14));
    const cavityMesh = new THREE.Mesh(cavityGeo, cavityMat);
    cavityMesh.position.set(0, -0.02, -0.22);
    cavityMesh.scale.set(1.05, 0.88, 1.35);
    this.mouthAnchor.add(cavityMesh);

    // Glowing back throat / pharynx arch & dangling uvula
    this.throatMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xd93878,
        emissive: 0xff2266,
        emissiveIntensity: 0.55,
        roughness: 0.4,
      })
    );
    const throatGeo = this.track(new THREE.SphereGeometry(0.24, 16, 12));
    this.throatMesh = new THREE.Mesh(throatGeo, this.throatMat);
    this.throatMesh.position.set(0, -0.01, -0.56);
    this.throatMesh.scale.set(1.1, 1.0, 0.45);
    this.mouthAnchor.add(this.throatMesh);

    const velumGeo = this.track(new THREE.TorusGeometry(0.22, 0.045, 8, 20, Math.PI));
    this.velumMesh = new THREE.Mesh(velumGeo, this.throatMat);
    this.velumMesh.position.set(0, 0.06, -0.4);
    this.mouthAnchor.add(this.velumMesh);

    const uvulaGeo = this.track(new THREE.CapsuleGeometry(0.032, 0.07, 6, 8));
    this.uvulaMesh = new THREE.Mesh(uvulaGeo, this.throatMat);
    this.uvulaMesh.position.set(0, 0.03, -0.44);
    this.mouthAnchor.add(this.uvulaMesh);

    this.throatLight = new THREE.PointLight(0xff3388, 0.4, 2.2);
    this.throatLight.position.set(0, 0.0, -0.35);
    this.mouthAnchor.add(this.throatLight);

    // Upper & Lower Pearl Teeth Rows (10 curved teeth per arch)
    const toothGeo = this.track(new RoundedBoxGeometry(0.068, 0.088, 0.045, 2, 0.015));
    const toothMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xfffaf6,
        emissive: 0x3b2248,
        emissiveIntensity: 0.25,
        roughness: 0.16,
        metalness: 0.05,
      })
    );
    this.upperTeeth = new THREE.InstancedMesh(toothGeo, toothMat, 10);
    this.lowerTeeth = new THREE.InstancedMesh(toothGeo, toothMat, 10);
    for (let i = 0; i < 10; i++) {
      const ang = ((i - 4.5) / 4.5) * 0.85;
      const x = Math.sin(ang) * 0.28;
      const z = Math.cos(ang) * 0.18 - 0.18;
      this.dummy.position.set(x, 0.035, z);
      this.dummy.rotation.set(0.08, ang, 0);
      this.dummy.scale.set(1, 1, 1);
      this.dummy.updateMatrix();
      this.upperTeeth.setMatrixAt(i, this.dummy.matrix);

      this.dummy.position.set(x * 0.94, -0.055, z + 0.14);
      this.dummy.rotation.set(-0.08, ang, 0);
      this.dummy.scale.set(0.92, 0.85, 0.92);
      this.dummy.updateMatrix();
      this.lowerTeeth.setMatrixAt(i, this.dummy.matrix);
    }
    this.upperTeeth.instanceMatrix.needsUpdate = true;
    this.lowerTeeth.instanceMatrix.needsUpdate = true;
    this.mouthAnchor.add(this.upperTeeth);
    this.lowerJawGroup.add(this.lowerTeeth);

    // Articulated 3D Tongue inside the mouth cavity
    this.tongueGeo = this.track(createTongueGeometry(this.tongueNx, this.tongueNz));
    this.tongueMat = this.track(
      new THREE.MeshPhysicalMaterial({
        color: 0xff5599,
        emissive: 0x6a093c,
        emissiveIntensity: 0.45,
        roughness: 0.28,
        clearcoat: 0.65,
        clearcoatRoughness: 0.2,
        side: THREE.DoubleSide,
      })
    );
    const tongueMesh = new THREE.Mesh(this.tongueGeo, this.tongueMat);
    tongueMesh.position.set(0, 0.05, -0.06);
    this.mouthAnchor.add(tongueMesh);

    // Sculpted 3D Upper & Lower Lips
    this.upperLipGeo = this.track(createLipTubeGeometry(this.lipUSegs, this.lipVSegs));
    this.lowerLipGeo = this.track(createLipTubeGeometry(this.lipUSegs, this.lipVSegs));
    this.lipMat = this.track(
      new THREE.MeshPhysicalMaterial({
        color: 0xff4fa8,
        emissive: 0x700c45,
        emissiveIntensity: 0.45,
        roughness: 0.22,
        clearcoat: 0.85,
        clearcoatRoughness: 0.14,
      })
    );
    const upperLip = new THREE.Mesh(this.upperLipGeo, this.lipMat);
    upperLip.position.set(0, 0, 0.14);
    const lowerLip = new THREE.Mesh(this.lowerLipGeo, this.lipMat);
    lowerLip.position.set(0, 0, 0.14);
    this.mouthAnchor.add(upperLip);
    this.mouthAnchor.add(lowerLip);
  }

  private syncSize(): void {
    if (!this.renderer || !this.camera) return;
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (w <= 0 || h <= 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  public renderFrame(tel: Bear3DTelemetry): void {
    if (!this.active || this.webglFailed) return;
    if (!this.initialized) {
      this.initWebGL();
    }
    if (!this.renderer || !this.scene || !this.camera || !this.controls) return;

    const nowMs = performance.now();
    const dt = this.lastMs > 0 ? Math.min(0.05, (nowMs - this.lastMs) / 1000) : 0.016;
    this.lastMs = nowMs;
    const beatHz = Math.max(60, tel.bpm) / 60;
    this.phase += dt * beatHz * Math.PI;
    const t = nowMs / 1000;
    const uTime = (nowMs - this.dualStartMs) * 0.001;

    const vis = tel.visual ?? DEFAULT_BEAR_VISUAL_CONFIG;

    // 0. Dual Offscreen Framebuffer Pass (disabled unless SPHERE_TEXTURE_ENABLED):
    //    Pass A -> skyTarget (animated psychedelic sky sphere texture)
    //    Pass B -> headTarget (contrasting random visualization texture wrapped onto the 3D sphere head)
    if (SPHERE_TEXTURE_ENABLED) {
      if (uTime - this.lastTexSwapSec > 28 && tel.mix > 0.08) {
        this.randomizeTextures();
      }
      const du = this.dualUniforms;
      const beatAngle = t * beatHz * Math.PI * 2;
      du.uTime.value = uTime;
      du.uAudioTime.value = t;
      du.uBpm.value = Math.max(60, tel.bpm || 126);
      du.uBeat.value = (t * beatHz) % 1;
      du.uBpmSin.value = Math.sin(beatAngle);
      du.uBpmSin2.value = Math.sin(beatAngle * 2);
      du.uBpmSin4.value = Math.sin(beatAngle * 4);
      du.uKick.value = tel.drumPunch;
      du.uOnset.value = tel.drumPunch;
      du.uEnergy.value = tel.mix;
      du.uSpeed.value = 0.85 + 0.35 * vis.danceGain;
      du.uPalette.value = (tel.stage - 1) % 4;
      du.uWarmth.value = tel.bass;
      du.uBrightness.value = tel.other * 0.6 + tel.drums * 0.4;
      du.uMusicDensity.value = Math.min(1, (tel.drums + tel.bass + tel.other + tel.vocals) * 0.45);
      du.uForwardFlight.value = tel.mix * 1.2;
      if (tel.freqBins && tel.freqBins.length >= 64) {
        const bins = tel.freqBins;
        const ranges = [
          [0, 3], [3, 8], [8, 16], [16, 32],
          [32, 64], [64, 128], [128, 256], [256, Math.min(512, bins.length)],
        ];
        for (let b = 0; b < 8; b++) {
          let sum = 0;
          const [lo, hi] = ranges[b];
          for (let k = lo; k < hi; k++) sum += bins[k];
          const amp = sum / Math.max(1, (hi - lo) * 255);
          du[`uSpectrum${b}`].value = amp;
          du[`uBandFlux${b}`].value = amp * tel.drumPunch;
        }
      } else {
        const vals = [
          tel.bass,
          tel.drums * 0.8 + tel.bass * 0.4,
          tel.other * 0.7,
          tel.vocals,
          tel.vocals * 0.8 + tel.other * 0.5,
          tel.other,
          tel.drums * 0.9,
          tel.mix,
        ];
        for (let b = 0; b < 8; b++) {
          du[`uSpectrum${b}`].value = vals[b];
          du[`uBandFlux${b}`].value = vals[b] * tel.drumPunch;
        }
      }
      if (this.offQuadMesh && this.skyTarget && this.headTarget) {
        const skyMat = this.getDualShaderMaterial(this.skyPresetId);
        if (skyMat) {
          this.offQuadMesh.material = skyMat;
          this.renderer.setRenderTarget(this.skyTarget);
          this.renderer.render(this.offScene, this.offCam);
        }
        const headTexMat = this.getDualShaderMaterial(this.headPresetId);
        if (headTexMat) {
          this.offQuadMesh.material = headTexMat;
          this.renderer.setRenderTarget(this.headTarget);
          this.renderer.render(this.offScene, this.offCam);
        }
        this.renderer.setRenderTarget(null);
      }
    }

    // 1. DMT / LSD psychedelic stage palette sweep + dual-FBO sphere texture illumination
    const baseHue = ((tel.stage - 1) * 0.2 + t * 0.07 + tel.mix * 0.12) % 1;
    if (this.gummiMat) {
      if (SPHERE_TEXTURE_ENABLED) {
        this.gummiMat.color.setHex(0xffffff);
        this.gummiMat.emissive.setHex(0xffffff);
        this.gummiMat.emissiveIntensity = 0.62 + tel.mix * 0.68 + tel.drumPunch * 0.32;
      } else {
        // No shader texture: drive the gummi skin from the stage palette so the
        // head still breathes with the mix without the per-frame FBO passes.
        this.tmpColor.setHSL((baseHue + 0.62) % 1, 0.85, 0.54 + 0.1 * tel.mix);
        this.gummiMat.color.copy(this.tmpColor);
        this.gummiMat.emissive.copy(this.tmpColor);
        this.gummiMat.emissiveIntensity = 0.36 + tel.mix * 0.45 + tel.drumPunch * 0.22;
      }
    }
    if (this.skyMesh) {
      this.skyMesh.rotation.y = t * 0.04;
    }
    if (this.cageMat) {
      this.cageMat.visible = vis.showCage;
      this.tmpColor.setHSL((baseHue + 0.5) % 1, 1.0, 0.58);
      this.cageMat.color.copy(this.tmpColor);
      this.cageMat.opacity = vis.showCage ? 0.2 + 0.32 * tel.mix + 0.25 * tel.drumPunch : 0;
    }
    if (this.lipMat && this.tongueMat && this.throatMat) {
      if (vis.mouthStyle === "acid") {
        this.lipMat.color.setHex(0xb8ff00);
        this.lipMat.emissive.setHex(0x00ffaa);
        this.tongueMat.color.setHex(0x00ffff);
        this.tongueMat.emissive.setHex(0x0066ff);
        this.throatMat.color.setHex(0x9d00ff);
        this.throatMat.emissive.setHex(0xd900ff);
      } else if (vis.mouthStyle === "cyber") {
        this.lipMat.color.setHex(0x00f0ff);
        this.lipMat.emissive.setHex(0x0088ff);
        this.tongueMat.color.setHex(0xff00c8);
        this.tongueMat.emissive.setHex(0x6600aa);
        this.throatMat.color.setHex(0x00ffff);
        this.throatMat.emissive.setHex(0x00aaff);
      } else {
        this.lipMat.color.setHex(0xe00751);
        this.lipMat.emissive.setHex(0x610022);
        this.lipMat.roughness = 0.12;
        this.lipMat.clearcoat = 1.0;
        this.lipMat.clearcoatRoughness = 0.05;
        this.tongueMat.color.setHex(0xff5599);
        this.tongueMat.emissive.setHex(0x6a093c);
        this.throatMat.color.setHex(0xd93878);
        this.throatMat.emissive.setHex(0xff2266);
      }
    }
    if (this.discoKeyLight && this.discoRimLight) {
      this.tmpColor.setHSL((t * 0.18) % 1, 1.0, 0.55);
      this.discoKeyLight.color.copy(this.tmpColor);
      this.discoKeyLight.intensity = 1.6 + tel.drums * 2.2;
      this.tmpColor.setHSL((t * 0.18 + 0.48) % 1, 1.0, 0.55);
      this.discoRimLight.color.copy(this.tmpColor);
      this.discoRimLight.intensity = 1.6 + tel.bass * 2.4;
    }

    // 2. Floating Circle Head Techno Bounce (squash & stretch + hype/eat/pet animations)
    const hypeBoost = (tel.action === "HYPE" ? 1.75 : 1.0) * vis.danceGain;
    const danceGroove = Math.sin(this.phase) * (0.08 + 0.12 * tel.drums) * hypeBoost;
    const bounceY = Math.abs(Math.sin(this.phase)) * (0.1 + 0.15 * tel.drumPunch + 0.08 * tel.bass) * vis.danceGain;
    const squash = clamp((tel.drumPunch * 0.12 + tel.drums * 0.07) * vis.danceGain, 0, 0.22);
    this.bodyGroup.position.set(Math.sin(this.phase * 0.5) * 0.12 * hypeBoost, bounceY, 0);
    this.bodyGroup.scale.set(1 + squash * 0.65, 1 - squash * 0.5, 1 + squash * 0.65);
    this.bodyGroup.rotation.y = Math.sin(this.phase * 0.5) * 0.2 * hypeBoost;
    this.bodyGroup.rotation.z = Math.cos(this.phase * 0.5) * 0.08 * hypeBoost;

    // Head bob & rhythmic tilt
    this.headGroup.rotation.x = Math.sin(this.phase * 2) * (0.05 + 0.08 * tel.drumPunch) - danceGroove * 0.25;
    this.headGroup.rotation.z = Math.sin(this.phase * 0.5) * 0.1;

    // Ear wiggle & floating DMT halo
    const earWig = Math.sin(t * 9) * (0.05 + 0.18 * tel.vocals + 0.1 * tel.drumPunch);
    this.leftEar.rotation.z = 0.42 + earWig;
    this.rightEar.rotation.z = -0.42 - earWig;
    if (this.haloMesh) {
      this.haloMesh.visible = vis.showHalo;
      this.haloMesh.position.y = 1.44 + Math.sin(t * 3.4) * 0.06;
      this.haloMesh.rotation.z = t * 1.1;
    }
    if (this.visorMesh) {
      this.visorMesh.visible = vis.showVisor;
    }

    // Eyes react to drums (left) + bass (right) + blink squash
    const blinkScale = tel.isBlinking ? 0.14 : 1.0;
    this.leftEye.scale.set(1, blinkScale, 1);
    this.rightEye.scale.set(1, blinkScale, 1);
    if (this.leftIris && this.leftPupil) {
      const ls = 1 + tel.drums * 0.28 + tel.drumPunch * 0.18;
      this.leftIris.scale.set(ls, ls * 1.05, 0.55);
      this.leftPupil.scale.set(1 / Math.sqrt(ls), 1.05 / Math.sqrt(ls), 0.5);
    }
    if (this.rightIris && this.rightPupil) {
      const rs = 1 + tel.bass * 0.28 + tel.bassPunch * 0.18;
      this.rightIris.scale.set(rs, rs * 1.05, 0.55);
      this.rightPupil.scale.set(1 / Math.sqrt(rs), 1.05 / Math.sqrt(rs), 0.5);
    }
    if (this.visorMat) {
      this.visorMat.opacity = vis.showVisor ? 0.16 + 0.3 * tel.mix + (tel.isDrop ? 0.2 : 0) : 0;
    }
    if (this.earBowGroup) {
      this.earBowGroup.visible = vis.showBikini;
    }
    if (this.bikiniMat) {
      this.bikiniMat.emissiveIntensity = 0.55 + tel.mix * 0.95 + tel.drumPunch * 0.5;
    }

    // Spin mandala rings & orbiting LSD motes
    if (this.mandalaFloor) {
      this.mandalaFloor.visible = vis.showHalo;
      this.mandalaFloor.rotation.y = t * 0.35;
      const ms = 1 + tel.bass * 0.12;
      this.mandalaFloor.scale.set(ms, 1, ms);
    }
    if (this.mandalaBack) {
      this.mandalaBack.visible = vis.showHalo;
      this.mandalaBack.rotation.z = -t * 0.48;
      const bs = 1 + tel.mix * 0.18;
      this.mandalaBack.scale.set(bs, bs, 1);
    }
    if (this.moteRing) {
      this.moteRing.visible = vis.showHalo;
      this.moteRing.rotation.y = t * 0.65;
      this.moteRing.position.y = Math.sin(t * 2.1) * 0.15;
    }

    // 3. REALISTIC SINGING MOUTH ARTICULATION (Zero-latency Head Synth synced!)
    const m = tel.mouth;
    const art = m?.rawArt ?? m?.art;
    const tract = m?.rawTract ?? m?.tract;
    const fb = fallbackMouthParams(tel.fallbackVowel, tel.fallbackAperture);

    const ap = clamp((m?.active ? art!.aperture : tel.fallbackAperture) * vis.mouthGain, 0, 1);
    const widthNorm = m?.active ? clamp(art!.width, 0, 1) : fb.width;
    const roundNorm = m?.active ? clamp(art!.roundness, 0, 1) : fb.round;
    const jawNorm = m?.active ? clamp(art!.jawDrop, 0, 1) : fb.jaw;
    const lipOpenExtra = tract ? clamp(tract.lipOpen, -0.25, 0.95) : 0;
    const lipRound = tract ? clamp(tract.lipRound, 0, 1) : roundNorm;
    const lipProtrude = tract ? clamp(tract.lipProtrude, -0.35, 1.0) : roundNorm * 0.55;
    const lipTuck = tract ? clamp(tract.lipTuck, 0, 1) : 0;
    const velum = tract ? clamp(tract.velum, 0, 1) : 0.2;
    const tongueX = tract ? clamp(tract.tongueX, -1.5, 1.5) : fb.tongueX;
    const tongueY = tract ? clamp(tract.tongueY, -1.2, 1.4) : fb.tongueY;
    const tipX = tract ? clamp(tract.tipX, -1.0, 1.2) : 0;
    const tipY = tract ? clamp(tract.tipY, -1.0, 1.2) : fb.tipY;
    const tongueSides = tract ? clamp(tract.tongueSides, 0, 1) : 0;
    const groove = tract ? clamp(tract.groove, 0, 1) : 0.15;

    const singingGlow = clamp(tel.vocals * 1.3 + ap * 0.75, 0, 1);

    // Lip dimensions (sculpted lipstick lips with lively jaw & lip excursion)
    const eatDrop = tel.action === "EAT" ? 0.22 * (0.5 + 0.5 * Math.sin(t * 14)) : 0;
    const halfW = clamp((0.28 + 0.19 * widthNorm - 0.14 * lipRound) * (0.85 + 0.15 * vis.mouthGain), 0.15, 0.48);
    const openTotal = clamp(0.012 + (ap * 0.26 + jawNorm * 0.18 + lipOpenExtra * 0.14 + eatDrop), 0.01, 0.42);
    const upperH = 0.036 + openTotal * 0.22;
    const lowerH = 0.02 + openTotal * 0.9;
    const protrude = 0.04 + lipProtrude * 0.15 + lipRound * 0.1;

    // Lower jaw drops smoothly with mouth opening
    this.lowerJawGroup.position.y = 0.01 - openTotal * 0.76;
    this.lowerJawGroup.rotation.x = 0.05 + openTotal * 1.22;

    if (this.upperLipGeo && this.lowerLipGeo) {
      const lp: LipParams = {
        halfW,
        upperH,
        lowerH,
        protrude,
        roundness: lipRound,
        tuck: lipTuck,
        thick: 0.046 + lipRound * 0.016,
      };
      updateLipTubeGeometry(this.upperLipGeo, true, lp, this.lipUSegs, this.lipVSegs);
      updateLipTubeGeometry(this.lowerLipGeo, false, lp, this.lipUSegs, this.lipVSegs);
    }

    if (this.tongueGeo) {
      updateTongueGeometry(
        this.tongueGeo,
        this.tongueNx,
        this.tongueNz,
        tongueX,
        tongueY,
        tipX,
        tipY,
        tongueSides,
        groove,
        openTotal * 1.8
      );
    }

    // Back throat glow, velum soft-palate lift & uvula bobbing when singing
    if (this.throatMat && this.throatLight && this.uvulaMesh && this.velumMesh) {
      this.throatMat.emissiveIntensity = 0.35 + singingGlow * 2.1 + tel.vocals * 1.2;
      this.throatLight.intensity = 0.25 + singingGlow * 2.8;
      this.velumMesh.position.y = 0.07 + velum * 0.05;
      this.uvulaMesh.position.y = 0.03 + velum * 0.04;
      this.uvulaMesh.rotation.z = Math.sin(t * 24) * 0.22 * singingGlow;
    }

    if (this.preset === "orbit") {
      const orbitA = t * 0.55;
      this.camera.position.set(Math.sin(orbitA) * 4.05, 0.48 + Math.sin(t * 0.9) * 0.35, Math.cos(orbitA) * 4.05);
      this.controls.target.set(0, 0, 0);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  public dispose(): void {
    this.ro?.disconnect();
    this.controls?.dispose();
    this.skyTarget?.dispose();
    this.headTarget?.dispose();
    this.offQuadGeo.dispose();
    this.vizMatCache.forEach((m) => m.dispose());
    this.vizMatCache.clear();
    for (const d of this.disposables) d.dispose();
    this.disposables.length = 0;
    this.renderer?.dispose();
  }
}
