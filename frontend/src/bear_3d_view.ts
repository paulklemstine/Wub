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
  visual?: BearVisualConfig;
}

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
    const arcEnv = Math.cos(sx * Math.PI * 0.5); // 0 at corners, 1 center
    const x = sx * p.halfW;
    // Slight wrap back around the snout curvature
    const zBase = p.protrude * Math.pow(arcEnv, 1.35) - 0.18 * (sx * sx);

    let y: number;
    let r: number;
    if (isUpper) {
      // Cupid's bow dip at center + raised philtrum peaks
      const bow = (Math.cos(sx * Math.PI * 2.1) * 0.045 - 0.03) * (1 - 0.6 * p.roundness) * arcEnv;
      y = p.upperH * Math.pow(arcEnv, 0.85) + bow;
      r = p.thick * (0.45 + 0.65 * Math.pow(arcEnv, 0.7));
    } else {
      const tuckDrop = -p.tuck * 0.14 * arcEnv;
      y = -p.lowerH * Math.pow(arcEnv, 0.88) - 0.02 * arcEnv;
      r = p.thick * (0.45 + 0.72 * Math.pow(arcEnv, 0.7));
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

  // Rig groups & reactive meshes
  private stageRoot = new THREE.Group();
  private bodyGroup = new THREE.Group();
  private headGroup = new THREE.Group();
  private leftArm = new THREE.Group();
  private rightArm = new THREE.Group();
  private leftLeg = new THREE.Group();
  private rightLeg = new THREE.Group();
  private leftEar = new THREE.Group();
  private rightEar = new THREE.Group();
  private leftEye = new THREE.Group();
  private rightEye = new THREE.Group();
  private leftIris: THREE.Mesh | null = null;
  private rightIris: THREE.Mesh | null = null;
  private leftPupil: THREE.Mesh | null = null;
  private rightPupil: THREE.Mesh | null = null;
  private visorMesh: THREE.Mesh | null = null;
  private heartMesh: THREE.Mesh | null = null;
  private haloMesh: THREE.Mesh | null = null;
  private bikiniGroup: THREE.Group | null = null;
  private earBowGroup: THREE.Group | null = null;
  private bikiniMat: THREE.MeshPhysicalMaterial | null = null;
  private mandalaFloor: THREE.Group | null = null;
  private mandalaBack: THREE.Group | null = null;
  private moteRing: THREE.Group | null = null;

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
  private heartMat: THREE.MeshStandardMaterial | null = null;
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
    const d = 7.4;
    if (preset === "mouth") {
      this.camera.position.set(0.18, 2.02, 3.05);
      this.controls.target.set(0, 1.94, 1.02);
    } else if (preset === "front") {
      this.camera.position.set(0, 1.55, d);
      this.controls.target.set(0, 1.05, 0);
    } else if (preset === "side") {
      this.camera.position.set(6.4, 1.55, 2.1);
      this.controls.target.set(0, 1.2, 0.2);
    } else if (preset === "orbit") {
      this.camera.position.set(3.2, 2.1, 6.3);
      this.controls.target.set(0, 1.15, 0);
    } else {
      // "three" (hero 3/4)
      this.camera.position.set(2.55, 2.05, 6.25);
      this.controls.target.set(0, 1.18, 0.15);
    }
    this.controls.update();
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
    scene.fog = new THREE.FogExp2(0x070118, 0.036);
    const camera = new THREE.PerspectiveCamera(36, 1.4, 0.1, 100);
    this.scene = scene;
    this.camera = camera;

    // DMT / Techno neon 3-point + point-disco lighting
    scene.add(new THREE.HemisphereLight(0xc084fc, 0x16052c, 1.25));
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
    floor.position.set(0, -1.22, 0);

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

    // Sacred geometry DMT prism frame rotating behind the bear
    const back = new THREE.Group();
    back.position.set(0, 1.05, -2.4);
    for (let k = 0; k < 2; k++) {
      const geo = this.track(new THREE.TorusGeometry(2.15 + k * 0.55, 0.028, 8, k === 0 ? 6 : 12));
      const mat = this.track(
        new THREE.MeshBasicMaterial({
          color: k === 0 ? 0xff4fd8 : 0x00f0ff,
          transparent: true,
          opacity: 0.42,
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
    motes.position.set(0, 1.0, 0);
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
      const r = 2.3 + (i % 3) * 0.45;
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
        color: 0xff66e0,
        emissive: 0x3b0764,
        emissiveIntensity: 0.55,
        roughness: 0.18,
        metalness: 0.12,
        transmission: 0.24,
        thickness: 0.8,
        clearcoat: 0.9,
        clearcoatRoughness: 0.12,
        transparent: true,
        opacity: 0.9,
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
        emissiveIntensity: 0.4,
        roughness: 0.28,
        metalness: 0.08,
      })
    );

    this.heartMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0xff22aa,
        emissive: 0xff0088,
        emissiveIntensity: 1.4,
        roughness: 0.15,
      })
    );

    // --- Torso ---
    this.bodyGroup = new THREE.Group();
    this.stageRoot.add(this.bodyGroup);

    const torsoGeo = this.track(new THREE.SphereGeometry(0.92, 26, 20));
    const torsoMesh = new THREE.Mesh(torsoGeo, this.gummiMat);
    torsoMesh.scale.set(1.06, 1.15, 0.88);
    torsoMesh.position.set(0, 0.02, 0);
    this.bodyGroup.add(torsoMesh);

    const torsoCage = new THREE.Mesh(torsoGeo, this.cageMat);
    torsoCage.scale.set(1.1, 1.18, 0.92);
    torsoCage.position.set(0, 0.02, 0);
    this.bodyGroup.add(torsoCage);

    // Glowing inner cyber heart crystal
    const heartGeo = this.track(new THREE.OctahedronGeometry(0.32, 1));
    this.heartMesh = new THREE.Mesh(heartGeo, this.heartMat);
    this.heartMesh.position.set(0, 0.1, 0.62);
    this.heartMesh.scale.set(1.05, 1.12, 0.6);
    this.bodyGroup.add(this.heartMesh);

    // Techno collar ring
    const collarGeo = this.track(new THREE.TorusGeometry(0.68, 0.065, 10, 28));
    const collarMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x00d8ff,
        emissiveIntensity: 0.95,
        metalness: 0.6,
        roughness: 0.2,
      })
    );
    const collar = new THREE.Mesh(collarGeo, collarMat);
    collar.rotation.x = Math.PI / 2;
    collar.position.set(0, 0.95, 0.04);
    this.bodyGroup.add(collar);

    // --- Sexy Kawaii Neon Rave Bikini ---
    const bikini = new THREE.Group();
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
    const trimMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x00d8ff,
        emissiveIntensity: 0.9,
        roughness: 0.2,
      })
    );

    // Triangle bikini top cups + halter straps + center clasp jewel
    const cupGeo = this.track(new THREE.ConeGeometry(0.28, 0.34, 4, 1));
    for (const s of [-1, 1]) {
      const cup = new THREE.Mesh(cupGeo, this.bikiniMat);
      cup.position.set(s * 0.33, 0.34, 0.72);
      cup.rotation.set(-0.22, s * 0.36, s * -0.14);
      cup.scale.set(1.05, 1.0, 0.48);
      bikini.add(cup);

      const strapGeo = this.track(new THREE.CylinderGeometry(0.02, 0.02, 0.62, 8));
      const strap = new THREE.Mesh(strapGeo, trimMat);
      strap.position.set(s * 0.27, 0.68, 0.52);
      strap.rotation.set(-0.62, 0, s * 0.22);
      bikini.add(strap);
    }
    const bustBandGeo = this.track(new THREE.TorusGeometry(0.84, 0.024, 8, 28));
    const bustBand = new THREE.Mesh(bustBandGeo, trimMat);
    bustBand.rotation.x = Math.PI / 2;
    bustBand.position.set(0, 0.22, 0.02);
    bikini.add(bustBand);

    const gemGeo = this.track(new THREE.OctahedronGeometry(0.075, 0));
    const centerGem = new THREE.Mesh(gemGeo, trimMat);
    centerGem.position.set(0, 0.3, 0.85);
    bikini.add(centerGem);

    // Low-rise V-cut bikini bottoms + hip waistband + side tie bows
    const waistBandGeo = this.track(new THREE.TorusGeometry(0.88, 0.028, 8, 30));
    const waistBand = new THREE.Mesh(waistBandGeo, trimMat);
    waistBand.rotation.x = Math.PI / 2;
    waistBand.position.set(0, -0.34, 0.02);
    bikini.add(waistBand);

    const briefGeo = this.track(new THREE.ConeGeometry(0.46, 0.44, 4, 1));
    const frontBrief = new THREE.Mesh(briefGeo, this.bikiniMat);
    frontBrief.position.set(0, -0.52, 0.66);
    frontBrief.rotation.set(0.42, Math.PI / 4, Math.PI);
    frontBrief.scale.set(1.25, 1.0, 0.45);
    bikini.add(frontBrief);

    const backBrief = new THREE.Mesh(briefGeo, this.bikiniMat);
    backBrief.position.set(0, -0.52, -0.66);
    backBrief.rotation.set(-0.42, Math.PI / 4, Math.PI);
    backBrief.scale.set(1.3, 1.0, 0.45);
    bikini.add(backBrief);

    const bowLoopGeo = this.track(new THREE.ConeGeometry(0.075, 0.16, 8));
    for (const s of [-1, 1]) {
      const hipBow = new THREE.Group();
      hipBow.position.set(s * 0.89, -0.34, 0.08);
      for (const ls of [-1, 1]) {
        const loop = new THREE.Mesh(bowLoopGeo, this.bikiniMat);
        loop.rotation.z = ls * 0.9;
        loop.position.y = ls * 0.06;
        hipBow.add(loop);
      }
      const knot = new THREE.Mesh(gemGeo, trimMat);
      knot.scale.set(0.65, 0.65, 0.65);
      hipBow.add(knot);
      bikini.add(hipBow);
    }
    this.bodyGroup.add(bikini);
    this.bikiniGroup = bikini;

    // --- Limbs (arms + paws, legs + feet) ---
    const limbGeo = this.track(new THREE.CapsuleGeometry(0.24, 0.52, 8, 14));
    const pawPadMat = this.track(
      new THREE.MeshStandardMaterial({
        color: 0x00ffff,
        emissive: 0x0088ff,
        emissiveIntensity: 0.9,
        roughness: 0.25,
      })
    );
    const padGeo = this.track(new THREE.SphereGeometry(0.14, 12, 10));

    const makeArm = (sign: number): THREE.Group => {
      const g = new THREE.Group();
      g.position.set(sign * 0.92, 0.56, 0.05);
      const armMesh = new THREE.Mesh(limbGeo, this.gummiMat!);
      armMesh.position.set(sign * 0.14, -0.26, 0.12);
      armMesh.rotation.z = sign * 0.55;
      armMesh.rotation.x = -0.35;
      g.add(armMesh);
      const paw = new THREE.Mesh(padGeo, pawPadMat);
      paw.position.set(sign * 0.34, -0.52, 0.28);
      g.add(paw);
      this.bodyGroup.add(g);
      return g;
    };
    this.leftArm = makeArm(-1);
    this.rightArm = makeArm(1);

    const legGeo = this.track(new THREE.CapsuleGeometry(0.27, 0.44, 8, 14));
    const makeLeg = (sign: number): THREE.Group => {
      const g = new THREE.Group();
      g.position.set(sign * 0.52, -0.78, 0.06);
      const legMesh = new THREE.Mesh(legGeo, this.gummiMat!);
      legMesh.position.set(0, -0.2, 0.08);
      g.add(legMesh);
      const sole = new THREE.Mesh(padGeo, pawPadMat);
      sole.scale.set(1.25, 0.75, 1.5);
      sole.position.set(0, -0.44, 0.26);
      g.add(sole);
      this.bodyGroup.add(g);
      return g;
    };
    this.leftLeg = makeLeg(-1);
    this.rightLeg = makeLeg(1);

    // --- Head Group ---
    this.headGroup = new THREE.Group();
    this.headGroup.position.set(0, 1.92, 0.04);
    this.bodyGroup.add(this.headGroup);

    const headGeo = this.track(new THREE.SphereGeometry(1.02, 28, 22));
    const headMesh = new THREE.Mesh(headGeo, this.gummiMat);
    headMesh.scale.set(1.16, 0.98, 0.96);
    this.headGroup.add(headMesh);

    const headCage = new THREE.Mesh(headGeo, this.cageMat);
    headCage.scale.set(1.2, 1.02, 1.0);
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
      ear.position.set(sign * 0.84, 0.78, -0.05);
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
    const haloGeo = this.track(new THREE.TorusGeometry(0.76, 0.036, 10, 36));
    this.haloMesh = new THREE.Mesh(haloGeo, goldMat);
    this.haloMesh.rotation.x = Math.PI / 2 - 0.14;
    this.haloMesh.position.set(0, 1.32, 0);
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
      eye.position.set(sign * 0.44, 0.24, 0.86);
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
      new THREE.CylinderGeometry(1.12, 1.12, 0.28, 24, 1, true, -0.78, 1.56)
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
      cheek.position.set(s * 0.68, -0.04, 0.76);
      cheek.scale.set(1.2, 0.78, 0.45);
      this.headGroup.add(cheek);
    }

    // Upper snout/muzzle dome + glossy nose + cute whisker puffs above upper lip
    const snoutGeo = this.track(new THREE.SphereGeometry(0.56, 22, 18));
    const upperSnout = new THREE.Mesh(snoutGeo, this.muzzleMat);
    upperSnout.position.set(0, -0.06, 0.76);
    upperSnout.scale.set(1.16, 0.74, 0.78);
    this.headGroup.add(upperSnout);

    const puffGeo = this.track(new THREE.SphereGeometry(0.17, 14, 12));
    for (const s of [-1, 1]) {
      const puff = new THREE.Mesh(puffGeo, this.muzzleMat);
      puff.position.set(s * 0.16, -0.02, 1.16);
      puff.scale.set(1.05, 0.88, 0.75);
      this.headGroup.add(puff);
    }

    const noseGeo = this.track(new THREE.SphereGeometry(0.145, 16, 12));
    const nose = new THREE.Mesh(noseGeo, darkMat);
    nose.position.set(0, 0.08, 1.24);
    nose.scale.set(1.2, 0.82, 0.9);
    this.headGroup.add(nose);
    const noseSpark = new THREE.Mesh(sparkGeo, sparkMat);
    noseSpark.position.set(-0.04, 0.13, 1.35);
    this.headGroup.add(noseSpark);

    // -------------------------------------------------------------------------
    // REALISTIC VISIBLE SINGING MOUTH ANATOMY ON THE BEAR'S FACE
    // (Lips, teeth rows, articulated 3D tongue, oral cavity & glowing back throat)
    // -------------------------------------------------------------------------
    this.mouthAnchor = new THREE.Group();
    this.mouthAnchor.position.set(0, -0.18, 1.06);
    this.headGroup.add(this.mouthAnchor);

    // Hinged lower chin/jaw cup
    this.lowerJawGroup = new THREE.Group();
    this.lowerJawGroup.position.set(0, 0.02, -0.16);
    this.mouthAnchor.add(this.lowerJawGroup);

    const chinGeo = this.track(new THREE.SphereGeometry(0.46, 20, 14));
    const chinMesh = new THREE.Mesh(chinGeo, this.muzzleMat);
    chinMesh.position.set(0, -0.22, -0.02);
    chinMesh.scale.set(1.18, 0.56, 0.85);
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

    const vis = tel.visual ?? DEFAULT_BEAR_VISUAL_CONFIG;

    // 1. DMT / LSD psychedelic stage palette sweep per creature stage
    const baseHue = ((tel.stage - 1) * 0.2 + t * 0.07 + tel.mix * 0.12) % 1;
    if (this.gummiMat) {
      this.tmpColor.setHSL(baseHue, 0.88, 0.62);
      this.gummiMat.color.copy(this.tmpColor);
      this.tmpColor.setHSL((baseHue + 0.12) % 1, 0.9, 0.24 + tel.mix * 0.18);
      this.gummiMat.emissive.copy(this.tmpColor);
      this.gummiMat.emissiveIntensity = 0.45 + tel.mix * 0.75 + tel.drumPunch * 0.35;
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
        this.lipMat.color.setHex(0xff4fa8);
        this.lipMat.emissive.setHex(0x700c45);
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

    // 2. Techno Dance Bounce (squash & stretch + hype/eat/pet animations)
    const hypeBoost = (tel.action === "HYPE" ? 1.75 : 1.0) * vis.danceGain;
    const danceGroove = Math.sin(this.phase) * (0.08 + 0.12 * tel.drums) * hypeBoost;
    const bounceY = Math.abs(Math.sin(this.phase)) * (0.09 + 0.14 * tel.drumPunch + 0.08 * tel.bass) * vis.danceGain;
    const squash = clamp((tel.drumPunch * 0.12 + tel.drums * 0.07) * vis.danceGain, 0, 0.22);
    this.bodyGroup.position.set(Math.sin(this.phase * 0.5) * 0.09 * hypeBoost, bounceY, 0);
    this.bodyGroup.scale.set(1 + squash * 0.65, 1 - squash * 0.5, 1 + squash * 0.65);
    this.bodyGroup.rotation.y = Math.sin(this.phase * 0.5) * 0.18 * hypeBoost;
    this.bodyGroup.rotation.z = Math.cos(this.phase * 0.5) * 0.07 * hypeBoost;

    // Pump arms & swing legs to BPM
    const armPump = Math.sin(this.phase) * (0.35 + 0.45 * tel.drums) * hypeBoost;
    this.leftArm.rotation.x = -0.35 + armPump;
    this.rightArm.rotation.x = -0.35 - armPump;
    this.leftArm.rotation.z = 0.18 + Math.abs(armPump) * 0.25;
    this.rightArm.rotation.z = -0.18 - Math.abs(armPump) * 0.25;

    const legKick = Math.sin(this.phase) * (0.14 + 0.24 * tel.bass);
    this.leftLeg.rotation.x = legKick;
    this.rightLeg.rotation.x = -legKick;

    // Head bob & rhythmic tilt
    this.headGroup.rotation.x = Math.sin(this.phase * 2) * (0.04 + 0.08 * tel.drumPunch) - danceGroove * 0.25;
    this.headGroup.rotation.z = Math.sin(this.phase * 0.5) * 0.1;

    // Heart crystal swells with 'other' pads/synths
    if (this.heartMesh && this.heartMat) {
      const hs = 1.0 + tel.other * 0.45 + tel.otherPunch * 0.25;
      this.heartMesh.scale.set(1.05 * hs, 1.12 * hs, 0.6 * hs);
      this.heartMesh.rotation.y = t * 1.4;
      this.heartMat.emissiveIntensity = 0.8 + tel.other * 2.2;
    }

    // Ear wiggle & floating DMT halo
    const earWig = Math.sin(t * 9) * (0.05 + 0.18 * tel.vocals + 0.1 * tel.drumPunch);
    this.leftEar.rotation.z = 0.42 + earWig;
    this.rightEar.rotation.z = -0.42 - earWig;
    if (this.haloMesh) {
      this.haloMesh.visible = vis.showHalo;
      this.haloMesh.position.y = 1.32 + Math.sin(t * 3.4) * 0.06;
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
    if (this.bikiniGroup) {
      this.bikiniGroup.visible = vis.showBikini;
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
      this.moteRing.position.y = 1.0 + Math.sin(t * 2.1) * 0.15;
    }

    // 3. REALISTIC SINGING MOUTH ARTICULATION (Lips, Teeth, Tongue, Throat)
    const m = tel.mouth;
    const art = m?.art;
    const tract = m?.tract;
    const fb = fallbackMouthParams(tel.fallbackVowel, tel.fallbackAperture);

    const ap = clamp((m?.active ? art!.aperture : tel.fallbackAperture) * vis.mouthGain, 0, 1);
    const widthNorm = m?.active ? clamp(art!.width, 0, 1) : fb.width;
    const roundNorm = m?.active ? clamp(art!.roundness, 0, 1) : fb.round;
    const jawNorm = m?.active ? clamp(art!.jawDrop, 0, 1) : fb.jaw;
    const lipOpenExtra = tract ? clamp(tract.lipOpen, -0.25, 0.9) : 0;
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

    const singingGlow = clamp(tel.vocals * 1.3 + ap * 0.65, 0, 1);

    // Lip dimensions (bear mouth width ~0.68, open height scales with jaw + lipOpen)
    const eatDrop = tel.action === "EAT" ? 0.18 * (0.5 + 0.5 * Math.sin(t * 14)) : 0;
    const halfW = clamp(0.29 + 0.17 * widthNorm - 0.13 * lipRound, 0.14, 0.44);
    const openTotal = clamp(0.014 + (ap * 0.22 + jawNorm * 0.14 + lipOpenExtra * 0.12 + eatDrop), 0.01, 0.34);
    const upperH = 0.032 + openTotal * 0.24;
    const lowerH = 0.018 + openTotal * 0.86;
    const protrude = 0.02 + lipProtrude * 0.13 + lipRound * 0.08;

    // Lower jaw drops smoothly with mouth opening
    this.lowerJawGroup.position.y = 0.02 - openTotal * 0.72;
    this.lowerJawGroup.rotation.x = 0.06 + openTotal * 1.15;

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
      this.camera.position.set(Math.sin(orbitA) * 6.4, 1.85 + Math.sin(t * 0.9) * 0.45, Math.cos(orbitA) * 6.4);
      this.controls.target.set(0, 1.15, 0);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  public dispose(): void {
    this.ro?.disconnect();
    this.controls?.dispose();
    for (const d of this.disposables) d.dispose();
    this.disposables.length = 0;
    this.renderer?.dispose();
  }
}
