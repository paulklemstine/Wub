// Exact 3D Three.js Vocal Tract View ported from MithrilMan/voice-workshop
// (Apache-2.0, Copyright 2026 Fabio Angela) to vanilla TypeScript for Wub KG-505.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { VocalFieldState } from "./articulation_bridge";
import { fieldRgb, FieldMode } from "./canvas_colors";
import { buildTractMeshes, Pouch, TractMeshes } from "./three/meshes";
import { N_SECTIONS } from "./tract/areaFunction";
import { TractGeometry } from "./tract/geometry";

export type CameraPreset3D = "side" | "front" | "top" | "three";
type AnchorKey = keyof TractMeshes["anchors"];

const ANCHOR_LABELS: Record<AnchorKey, string> = {
  tongue: "tongue",
  palate: "hard palate",
  velum: "velum",
  teeth: "teeth",
  lips: "lips",
  larynx: "larynx",
  pharynx: "pharynx",
  piriform: "piriform fossae",
  channels: "lateral channels",
  supra: "supralingual cavity",
};

const ANCHORS = Object.keys(ANCHOR_LABELS) as AnchorKey[];
const WATER = new THREE.Color(0.27, 0.62, 0.92);

function fieldAt(
  g: TractGeometry,
  i: number,
  mode: FieldMode,
  tel: VocalFieldState | null
): number {
  if (!tel || tel.ug <= 2) return 0.15;
  const k = Math.max(0, Math.min(N_SECTIONS - 1, Math.floor((g.s[i] / g.length) * N_SECTIONS)));
  if (mode === "pressure") return tel.pRms[k];
  if (mode === "velocity") return tel.uRms[k];
  const u = k <= tel.constrictionIndex ? tel.ug : tel.uMouth;
  const vel = u / Math.max(g.hits[i].area, 0.02);
  return Math.min(1, Math.log10(1 + vel / 30) / 2.3);
}

export class Tract3DView {
  private readonly host: HTMLElement;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;
  private airGeo: THREE.BufferGeometry | null = null;
  private airCol: Float32Array | null = null;
  private airMat: THREE.MeshStandardMaterial | null = null;
  private upperMesh: THREE.Mesh | null = null;
  private lowerMesh: THREE.Mesh | null = null;
  private toothGeo: THREE.BufferGeometry | null = null;
  private toothMat: THREE.MeshStandardMaterial | null = null;
  private teethMesh: THREE.InstancedMesh | null = null;
  private pouchMat: THREE.MeshStandardMaterial | null = null;
  private waterMat: THREE.MeshStandardMaterial | null = null;
  private readonly pouchParts: THREE.Mesh[] = [];
  private readonly clipOn: THREE.Plane[] = [new THREE.Plane(new THREE.Vector3(0, 0, -1), 0)];
  private readonly dummy = new THREE.Object3D();
  private readonly projVec = new THREE.Vector3();
  private readonly placed: [number, number][] = [];
  private meshes: TractMeshes | null = null;
  private lastGeom: TractGeometry | null = null;
  private pouchKey = "";
  private initialized = false;
  private webglFailed = false;
  private active = false;
  private ro: ResizeObserver | null = null;

  public cut = true;
  public tissues = true;
  public showLabels = true;
  public mode: FieldMode = "pressure";

  private readonly labelsLayer: HTMLDivElement;
  private readonly labelSpans: Partial<Record<AnchorKey, HTMLSpanElement>> = {};
  private readonly statusBanner: HTMLDivElement;

  constructor(host: HTMLElement) {
    this.host = host;

    this.labelsLayer = document.createElement("div");
    this.labelsLayer.className = "tract3d-labels";
    this.labelsLayer.setAttribute("aria-hidden", "true");
    for (const k of ANCHORS) {
      const span = document.createElement("span");
      span.className = "tract3d-label";
      span.textContent = ANCHOR_LABELS[k];
      span.style.display = "none";
      this.labelsLayer.appendChild(span);
      this.labelSpans[k] = span;
    }
    this.host.appendChild(this.labelsLayer);

    this.statusBanner = document.createElement("div");
    this.statusBanner.className = "tract3d-hint";
    this.statusBanner.textContent =
      "Drag to orbit • Scroll to zoom • Air coloured by sound pressure, tissues by anatomy";
    this.host.appendChild(this.statusBanner);
  }

  public setActive(visible: boolean, initialGeom?: TractGeometry): void {
    this.active = visible;
    this.host.style.display = visible ? "block" : "none";
    if (visible && !this.initialized && !this.webglFailed && initialGeom) {
      this.initWebGL(initialGeom);
    }
    if (visible && this.initialized) {
      this.resize();
    }
  }

  public isActive(): boolean {
    return this.active;
  }

  public setPreset(preset: CameraPreset3D): void {
    if (!this.meshes || !this.camera || !this.controls) return;
    const [cx, cy, cz] = this.meshes.center;
    const r = this.meshes.radius;
    const pos: Record<CameraPreset3D, [number, number, number]> = {
      side: [cx, cy, cz + 2.7 * r],
      front: [cx + 2.7 * r, cy + 0.15 * r, cz],
      top: [cx, cy + 2.7 * r, cz + 0.02 * r],
      three: [cx + 1.3 * r, cy + 0.6 * r, cz + 2.2 * r],
    };
    const lift = preset === "top" ? 0 : 0.12 * r;
    this.camera.position.set(pos[preset][0], pos[preset][1] + lift, pos[preset][2]);
    this.controls.target.set(cx, cy + lift, cz);
    this.controls.update();
  }

  private initWebGL(geom0: TractGeometry): void {
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: "low-power",
      });
    } catch {
      this.webglFailed = true;
      this.statusBanner.textContent = "WebGL unavailable in this browser — 3D view disabled.";
      return;
    }
    this.initialized = true;
    this.renderer = renderer;
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.localClippingEnabled = true;
    renderer.setClearColor(0x0b0e14, 1);

    const canvas = renderer.domElement;
    canvas.className = "tract3d-canvas";
    canvas.setAttribute("aria-label", "vocal tract 3D");
    this.host.prepend(canvas);

    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.statusBanner.textContent = "WebGL context lost — toggle Side/3D to restore.";
    });

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 400);
    scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x2a1d26, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.6, 1, 1.2);
    camera.add(key);
    const rim = new THREE.DirectionalLight(0x9fd8ff, 0.65);
    rim.position.set(-1, 0.3, -0.8);
    camera.add(rim);
    scene.add(camera);
    this.scene = scene;
    this.camera = camera;

    const meshes = buildTractMeshes(geom0);
    this.meshes = meshes;
    this.lastGeom = geom0;

    // Air cavity mesh
    const airGeo = new THREE.BufferGeometry();
    airGeo.setAttribute("position", new THREE.BufferAttribute(meshes.air.positions, 3));
    const airCol = new Float32Array(meshes.air.positions.length);
    airGeo.setAttribute("color", new THREE.BufferAttribute(airCol, 3));
    airGeo.setIndex(new THREE.BufferAttribute(meshes.air.index, 1));
    const airMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.5,
      metalness: 0,
      side: THREE.DoubleSide,
      emissive: 0x0a1a20,
    });
    scene.add(new THREE.Mesh(airGeo, airMat));
    this.airGeo = airGeo;
    this.airCol = airCol;
    this.airMat = airMat;

    // Tissue sheets
    const makeSheet = (sh: TractMeshes["upper"]) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(sh.positions, 3));
      g.setAttribute("color", new THREE.BufferAttribute(sh.colors, 3));
      g.setIndex(new THREE.BufferAttribute(sh.index, 1));
      const mat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.75,
        metalness: 0,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9,
      });
      const mesh = new THREE.Mesh(g, mat);
      scene.add(mesh);
      return mesh;
    };
    this.upperMesh = makeSheet(meshes.upper);
    this.lowerMesh = makeSheet(meshes.lower);

    // Teeth instances
    const toothGeo = new RoundedBoxGeometry(1, 1, 1, 2, 0.18);
    const toothMat = new THREE.MeshStandardMaterial({
      color: 0xf4eedf,
      roughness: 0.35,
      metalness: 0,
    });
    const teethMesh = new THREE.InstancedMesh(toothGeo, toothMat, 64);
    scene.add(teethMesh);
    this.toothGeo = toothGeo;
    this.toothMat = toothMat;
    this.teethMesh = teethMesh;

    // Piriform fossae pouches
    const pouchMat = new THREE.MeshStandardMaterial({
      vertexColors: false,
      roughness: 0.5,
      side: THREE.DoubleSide,
      color: 0x2ad4bd,
    });
    const waterMat = new THREE.MeshStandardMaterial({
      color: WATER,
      roughness: 0.2,
      transparent: true,
      opacity: 0.75,
    });
    this.pouchMat = pouchMat;
    this.waterMat = waterMat;
    for (let q = 0; q < 4; q++) {
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), q < 2 ? pouchMat : waterMat);
      scene.add(mesh);
      this.pouchParts.push(mesh);
    }

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.minDistance = 4;
    controls.maxDistance = 120;
    this.controls = controls;

    this.refreshGeometry(geom0);
    this.setPreset("three");
    this.resize();

    if (typeof ResizeObserver !== "undefined") {
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.host);
    }
  }

  private resize(): void {
    if (!this.renderer || !this.camera) return;
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private placePouch(p: Pouch): void {
    const k = [p.length, p.air, p.rOpen, p.rBottom].map((x) => x.toFixed(3)).join(",");
    const rAt = (x: number) => p.rOpen + (p.rBottom - p.rOpen) * (p.length > 0 ? x / p.length : 0);
    if (k !== this.pouchKey) {
      this.pouchKey = k;
      const parts: [number, number][] = [
        [0, p.air],
        [p.air, p.length],
      ];
      for (let q = 0; q < 4; q++) {
        const [a, b] = parts[q < 2 ? 0 : 1];
        const m = this.pouchParts[q];
        m.geometry.dispose();
        m.geometry =
          b - a > 1e-3
            ? new THREE.CylinderGeometry(rAt(a), rAt(b), b - a, 18, 1, !(b >= p.length - 1e-6))
            : new THREE.BufferGeometry();
      }
    }
    const dir = new THREE.Vector3(p.dx, p.dy, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().negate());
    for (let q = 0; q < 4; q++) {
      const [a, b] = q < 2 ? [0, p.air] : [p.air, p.length];
      const m = this.pouchParts[q];
      const c = 0.5 * (a + b);
      m.position.set(p.x + p.dx * c, p.y + p.dy * c, (q % 2 === 0 ? 1 : -1) * p.z);
      m.quaternion.copy(quat);
      m.visible = b - a > 1e-3;
    }
  }

  private refreshGeometry(geom: TractGeometry): void {
    if (!this.meshes || !this.airGeo || !this.upperMesh || !this.lowerMesh || !this.teethMesh) {
      return;
    }
    this.meshes = buildTractMeshes(geom, this.meshes);
    this.airGeo.attributes.position.needsUpdate = true;
    this.airGeo.computeVertexNormals();
    for (const m of [this.upperMesh, this.lowerMesh]) {
      m.geometry.attributes.position.needsUpdate = true;
      m.geometry.attributes.color.needsUpdate = true;
      m.geometry.computeVertexNormals();
    }
    this.meshes.teeth.forEach((th, i) => {
      this.dummy.position.set(th.x, th.y, th.z);
      this.dummy.rotation.set(0, 0, 0);
      this.dummy.rotateZ(th.tilt);
      this.dummy.rotateY(-th.yaw);
      this.dummy.scale.set(th.d, th.h, th.w);
      this.dummy.updateMatrix();
      this.teethMesh!.setMatrixAt(i, this.dummy.matrix);
    });
    this.teethMesh.count = this.meshes.teeth.length;
    this.teethMesh.instanceMatrix.needsUpdate = true;
    this.placePouch(this.meshes.pouch);
  }

  public renderFrame(geom: TractGeometry, tel: VocalFieldState | null): void {
    if (!this.active || this.webglFailed) return;
    if (!this.initialized) {
      this.initWebGL(geom);
    }
    if (
      !this.renderer ||
      !this.scene ||
      !this.camera ||
      !this.controls ||
      !this.meshes ||
      !this.airGeo ||
      !this.airCol ||
      !this.upperMesh ||
      !this.lowerMesh ||
      !this.toothMat ||
      !this.teethMesh ||
      !this.pouchMat
    ) {
      return;
    }

    if (geom !== this.lastGeom) {
      this.lastGeom = geom;
      this.refreshGeometry(geom);
    }

    // Colour the air mesh by standing-wave sound field
    const ray = this.meshes.air.ray;
    let prev = -1;
    let r = 0;
    let gc = 0;
    let b = 0;
    for (let q = 0; q < ray.length; q++) {
      if (ray[q] !== prev) {
        prev = ray[q];
        [r, gc, b] = fieldRgb(fieldAt(geom, prev, this.mode, tel));
      }
      this.airCol[3 * q] = r / 255;
      this.airCol[3 * q + 1] = gc / 255;
      this.airCol[3 * q + 2] = b / 255;
    }
    this.airGeo.attributes.color.needsUpdate = true;

    const pf = geom.piriform;
    let pi = 0;
    while (pi < geom.lastRay && geom.s[pi] < pf.at) pi++;
    const [pr, pg, pb] = fieldRgb(fieldAt(geom, pi, this.mode, tel));
    this.pouchMat.color.setRGB(pr / 255, pg / 255, pb / 255);

    for (const m of [this.upperMesh, this.lowerMesh]) {
      m.visible = this.tissues;
      const mat = m.material as THREE.MeshStandardMaterial;
      mat.clippingPlanes = this.cut ? this.clipOn : null;
      mat.opacity = this.cut ? 0.95 : 0.35;
      mat.depthWrite = this.cut;
    }
    this.toothMat.clippingPlanes = this.cut ? this.clipOn : null;
    this.teethMesh.visible = this.tissues;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);

    // Project anatomical anchors to 2D label spans
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    this.placed.length = 0;
    for (const k of ANCHORS) {
      const span = this.labelSpans[k];
      if (!span) continue;
      const a = this.meshes.anchors[k];
      if (!this.showLabels || !a || w <= 0 || h <= 0) {
        span.style.display = "none";
        continue;
      }
      this.projVec.set(a[0], a[1], a[2]).project(this.camera);
      const x = ((this.projVec.x + 1) / 2) * w;
      const y = ((1 - this.projVec.y) / 2) * h;
      const covered =
        this.placed.some(([px, py]) => Math.abs(px - x) < 72 && Math.abs(py - y) < 16) ||
        y < 28 ||
        y > h - 24;
      if (this.projVec.z > 1 || this.projVec.z < -1 || covered) {
        span.style.display = "none";
        continue;
      }
      this.placed.push([x, y]);
      span.style.display = "";
      span.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -50%)`;
    }
  }

  public dispose(): void {
    this.ro?.disconnect();
    this.controls?.dispose();
    this.airGeo?.dispose();
    this.airMat?.dispose();
    if (this.upperMesh) {
      this.upperMesh.geometry.dispose();
      (this.upperMesh.material as THREE.Material).dispose();
    }
    if (this.lowerMesh) {
      this.lowerMesh.geometry.dispose();
      (this.lowerMesh.material as THREE.Material).dispose();
    }
    this.toothGeo?.dispose();
    this.toothMat?.dispose();
    this.teethMesh?.dispose();
    for (const m of this.pouchParts) m.geometry.dispose();
    this.pouchMat?.dispose();
    this.waterMat?.dispose();
    this.renderer?.dispose();
  }
}
