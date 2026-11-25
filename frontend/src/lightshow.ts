// Trippy Rave Lightshow Output Engine
// Powered by ALL 267 visual scenes ported from github.com/paulklemstine/SuperAcid
// + raver1975 SpaceTubes hyperspace tunnel (268 total scenes!).
// Runs Three.js ShaderMaterial scene rendering + optional 2-layer BG+FG compositing
// + Ping-Pong Feedback Trails + SuperAcid Bloom / Chromatic Aberration / Glitch /
// Kaleidoscope / WCAG-safe Strobe inside #lightshow-canvas.

import * as THREE from "three";

export type LightshowPresetId = string;

export interface LightshowPresetMeta {
  id: string;
  label: string;
  role: "bg" | "fg";
  defaults: Record<string, number>;
  frag: string;
}

interface SuperAcidBundleJson {
  version: number;
  count: number;
  headers: Record<string, string>;
  presets: LightshowPresetMeta[];
}

export interface LightshowState {
  enabled: boolean;
  preset: string;
  fgPreset: string;
  dualLayer: boolean;
  cortexAuto: boolean;
  intensity: number; // 0 .. 2.2 (0 = gain at zero on boot)
  speed: number;     // 0.2 .. 2.5 (0.2 = slider hard left)
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

const FULLSCREEN_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// Built-in instant presets so frame 0 renders before async superacid_presets.json completes
const SPACETUBES_BUILTIN: LightshowPresetMeta = {
  id: "space-tubes",
  label: "SPACE TUBES",
  role: "bg",
  defaults: {},
  frag: `
precision highp float;
#define PI 3.14159265359
#define TAU 6.28318530718
varying vec2 vUv;
uniform float uTime;
uniform float uKick;
uniform float uEnergy;
uniform float uIntensity;
uniform float uSpeed;
uniform float uPalette;
uniform float uSpectrum0;
uniform float uSpectrum1;
uniform float uSpectrum6;

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
vec3 hsv2rgb(float h, float s, float v) {
  vec3 c = abs(fract(h + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0;
  return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}
void main() {
  vec2 p = (vUv - 0.5) * 2.0;
  float time = uTime * uSpeed;
  float b = uSpectrum0 * 0.5 + uSpectrum1 * 0.5;
  p += vec2(sin(time * 0.9), cos(time * 0.7)) * (0.12 + 0.16 * b);
  float r = length(p) + 0.001;
  float ang = atan(p.y, p.x);
  float z = 1.6 / r + time * (2.6 + b * 2.2);
  float rib = pow(abs(sin(z * 2.4)), 10.0);
  float rail = smoothstep(0.92, 1.0, abs(sin(ang * 6.0 + sin(z * 0.45) * 1.8)));
  // Wrap the angular cell index so the x=-pi/x=+pi seam hashes the same cell.
  // Without this the star field jumps at the polar cut instead of tiling.
  float starCell = mod(floor(((ang + PI) / TAU) * 12.0), 12.0);
  float star = smoothstep(0.985, 1.0, hash21(vec2(starCell, floor(z * 3.0)))) * (0.6 + uSpectrum6 * 1.4);
  float hue = fract(z * 0.08 + uPalette * 0.25 + time * 0.05);
  vec3 neon1 = hsv2rgb(hue, 0.95, 1.0);
  vec3 neon2 = hsv2rgb(fract(hue + 0.42), 0.92, 1.0);
  vec3 col = mix(vec3(0.02, 0.01, 0.08), neon1 * 0.85, rib);
  col += neon2 * rail * (0.45 + uKick * 0.75);
  col += vec3(0.85, 0.95, 1.0) * star;
  col *= smoothstep(0.06, 0.35, r);
  col += exp(-r * 6.0) * neon2 * (0.5 + b * 0.9);
  col *= uIntensity * (0.85 + 0.45 * uEnergy);
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,
};

const ACID_WARP_BUILTIN: LightshowPresetMeta = {
  id: "acid-warp",
  label: "ACID WARP",
  role: "bg",
  defaults: {},
  frag: `
precision highp float;
varying vec2 vUv;
uniform float uTime;
uniform float uKick;
uniform float uEnergy;
uniform float uIntensity;
uniform float uSpeed;
uniform float uPalette;
uniform float uSpectrum0;
uniform float uSpectrum3;

vec3 hsv2rgb(float h, float s, float v) {
  vec3 c = abs(fract(h + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0;
  return v * mix(vec3(1.0), clamp(c, 0.0, 1.0), s);
}
void main() {
  vec2 p = (vUv - 0.5) * 3.6;
  float time = uTime * uSpeed * 1.6;
  float b = (uSpectrum0 + uSpectrum3) * 0.9;
  vec2 p1 = p + vec2(sin(p.y * 2.5 + time * 0.7), cos(p.x * 2.3 + time * 0.6)) * (1.2 + b);
  vec2 p2 = p1 + vec2(sin(p1.y * 3.7 + time * 0.5 + p1.x * 1.3), cos(p1.x * 3.1 - time * 0.4)) * 0.9;
  vec2 pf = p2 + vec2(sin(p2.y * 5.1 + time * 0.3), cos(p2.x * 4.7 - time * 0.35)) * 0.65;
  float pat = sin(pf.x * 5.0 + pf.y * 4.0 + time * 0.3) * 0.5 + 0.5;
  float hue = fract(pat * 0.8 + uPalette * 0.25 + time * 0.05);
  vec3 col = hsv2rgb(hue, 0.9, clamp(uIntensity * (pat * 0.75 + 0.3 + uEnergy * 0.35), 0.0, 1.4));
  col += vec3(uKick * 0.22);
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,
};

const COMPOSITOR_POST_FRAG = `
precision highp float;
varying vec2 vUv;

uniform sampler2D uBgTex;
uniform sampler2D uFgTex;
uniform sampler2D uPrevTex;
uniform float uDualLayer;
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

vec3 sampleLayer(vec2 uv) {
  vec3 c = texture2D(uBgTex, clamp(uv, 0.001, 0.999)).rgb;
  if (uDualLayer > 0.5) {
    vec3 fg = texture2D(uFgTex, clamp(uv, 0.001, 0.999)).rgb;
    float fgLuma = max(fg.r, max(fg.g, fg.b));
    c = mix(c, c + fg * 0.95, smoothstep(0.06, 0.45, fgLuma));
  }
  return c;
}

void main() {
  if (uBlackout > 0.5) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = vUv;
  if (uFxKaleido > 0.5) {
    vec2 p = uv - 0.5;
    float r = length(p);
    float a = atan(p.y, p.x);
    float sec = 6.2831853 / 8.0;
    a = abs(mod(a + 6.2831853, sec) - sec * 0.5);
    uv = 0.5 + vec2(cos(a), sin(a)) * r;
  }

  if (uFxGlitch > 0.5) {
    float band = floor(uv.y * 24.0 + floor(uTime * 16.0));
    float rnd = hash1(band);
    float gate = step(0.82 - uKick * 0.28, rnd);
    uv.x += (hash1(band + 13.0) - 0.5) * 0.08 * gate;
  }

  vec3 col;
  if (uFxChroma > 0.5) {
    vec2 dir = uv - 0.5;
    float off = 0.007 + uKick * 0.014 + uEnergy * 0.006;
    col.r = sampleLayer(uv + dir * off).r;
    col.g = sampleLayer(uv).g;
    col.b = sampleLayer(uv - dir * off).b;
  } else {
    col = sampleLayer(uv);
  }

  if (uFxBloom > 0.5) {
    float d = 0.006 + uEnergy * 0.005;
    vec3 bSum =
      sampleLayer(uv + vec2(d, 0.0)) +
      sampleLayer(uv - vec2(d, 0.0)) +
      sampleLayer(uv + vec2(0.0, d)) +
      sampleLayer(uv - vec2(0.0, d));
    vec3 bright = max(bSum * 0.25 - 0.35, 0.0);
    col += bright * (0.72 + uKick * 0.45);
  }

  if (uFxTrails > 0.5) {
    vec2 prevUv = (vUv - 0.5) * (0.976 - uKick * 0.012) + 0.5;
    vec3 prevCol = texture2D(uPrevTex, clamp(prevUv, 0.001, 0.999)).rgb;
    col = max(col, prevCol * 0.72);
  }

  if (uFxGlitch > 0.5) {
    float scan = 0.86 + 0.14 * sin(vUv.y * 420.0 + uTime * 25.0);
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

const BLIT_FRAG = `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTex;
void main() {
  gl_FragColor = texture2D(uTex, vUv);
}
`;

export const LIGHTSHOW_PRESETS: readonly { id: string; label: string }[] = [
  { id: "acid-warp", label: "ACID WARP" },
  { id: "breathing-mandala", label: "DMT MANDALA" },
  { id: "space-tubes", label: "SPACETUBES" },
  { id: "dmt-hyperspace", label: "DMT HYPERSPACE" },
  { id: "ego-death", label: "EGO DEATH" },
  { id: "cosmic-serpent", label: "COSMIC SERPENT" },
  { id: "apollonian-gasket", label: "APOLLONIAN" },
  { id: "acid-smiley", label: "ACID SMILEY" },
];

export class RaveLightshowPanel {
  private readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer | null = null;
  private rendererAttempted = false;
  private fallbackCtx: CanvasRenderingContext2D | null = null;
  private contextLost = false;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private quadGeo = new THREE.PlaneGeometry(2, 2);
  private quadMesh: THREE.Mesh;

  private bgTarget: THREE.WebGLRenderTarget | null = null;
  private fgTarget: THREE.WebGLRenderTarget | null = null;
  private trailTargets: [THREE.WebGLRenderTarget | null, THREE.WebGLRenderTarget | null] = [null, null];
  private pingIdx = 0;
  private rtW = 0;
  private rtH = 0;

  private postMat: THREE.ShaderMaterial | null = null;
  private blitMat: THREE.ShaderMaterial | null = null;

  private presets: LightshowPresetMeta[] = [ACID_WARP_BUILTIN, SPACETUBES_BUILTIN];
  private presetMap = new Map<string, LightshowPresetMeta>();
  private headers: Record<string, string> = {};
  private materialCache = new Map<string, THREE.ShaderMaterial>();
  private readonly maxCachedMaterials = 18;
  private sharedUniforms: Record<string, THREE.IUniform<number>> = {};

  private state: LightshowState = {
    enabled: true,
    preset: "acid-warp",
    fgPreset: "cosmic-serpent",
    // Two full shader scenes per frame doubled fill-rate for a layer most
    // sessions never notice. Opt in with the 2-LAYER button when wanted.
    dualLayer: true,
    cortexAuto: true,
    intensity: 0,
    speed: 0.2,
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
  private lastRotateSec = 0;
  private startMs = performance.now();
  private onStateSync: ((presets: readonly LightshowPresetMeta[], activeId: string, fgId: string) => void) | null = null;

  constructor(
    canvas: HTMLCanvasElement,
    onStateSync?: (presets: readonly LightshowPresetMeta[], activeId: string, fgId: string) => void
  ) {
    this.canvas = canvas;
    this.onStateSync = onStateSync ?? null;
    this.quadMesh = new THREE.Mesh(this.quadGeo);
    this.scene.add(this.quadMesh);

    for (const p of this.presets) this.presetMap.set(p.id, p);
    this.initSharedUniforms();
    this.canvas.addEventListener("webglcontextlost", () => {
      this.contextLost = true;
      this.canvas.style.background = "radial-gradient(ellipse, #28194a, #050509)";
      this.canvas.title = "Graphics context lost; audio playback continues. Reload to retry graphics.";
    });
    void this.loadSuperAcidBank();
  }

  private initSharedUniforms(): void {
    const names = [
      "uTime", "uAudioTime", "uEnergy", "uBeat", "uOnset", "uBpm", "uKick",
      "uSpectralCentroid", "uSpectralFlux", "uPitch", "uPitchConfidence",
      "uSpectrum0", "uSpectrum1", "uSpectrum2", "uSpectrum3",
      "uSpectrum4", "uSpectrum5", "uSpectrum6", "uSpectrum7",
      "uSpectralRolloff", "uSpectralFlatness", "uZeroCrossingRate",
      "uSpectralSpread", "uSpectralSlope", "uSpectralCrestFactor",
      "uSpectralBandwidth", "uSpectralEntropy", "uBrightness", "uWarmth",
      "uHarmonicity", "uBpmSin", "uBpmSin2", "uBpmSin4",
      "uChromaDominant",
      "uChroma0", "uChroma1", "uChroma2", "uChroma3", "uChroma4", "uChroma5",
      "uChroma6", "uChroma7", "uChroma8", "uChroma9", "uChroma10", "uChroma11",
      "uMfcc0", "uMfcc1", "uMfcc2", "uMfcc3", "uMfcc4", "uMfcc5", "uMfcc6", "uMfcc7",
      "uPerceptualSharpness", "uPerceptualSpread",
      "uBandFlux0", "uBandFlux1", "uBandFlux2", "uBandFlux3",
      "uBandFlux4", "uBandFlux5", "uBandFlux6", "uBandFlux7",
      "uMusicDensity", "uForwardFlight",
      "uSensitivity", "uPalette", "uSpeed", "uZoom", "uIntensity", "uChaos",
    ];
    for (const n of names) {
      this.sharedUniforms[n] = { value: 0 };
    }
    this.sharedUniforms.uSensitivity.value = 1.15;
    this.sharedUniforms.uZoom.value = 0.0;
    this.sharedUniforms.uIntensity.value = 1.25;
    this.sharedUniforms.uSpeed.value = 1.05;
    this.sharedUniforms.uChaos.value = 0.5;
    this.sharedUniforms.uPitchConfidence.value = 0.7;
    this.sharedUniforms.uHarmonicity.value = 0.6;
  }

  private initRenderer(): void {
    if (this.rendererAttempted) return;
    this.rendererAttempted = true;
    try {
      // Probe once before Three.js, which otherwise throws and logs retries.
      const context = this.canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false });
      if (!context) {
        this.initFallback();
        return;
      }
      const r = new THREE.WebGLRenderer({
        context,
        canvas: this.canvas,
        alpha: false,
        antialias: false,
        powerPreference: "high-performance",
      });
      r.setPixelRatio(1);
      r.autoClear = false;
      this.renderer = r;

      this.postMat = new THREE.ShaderMaterial({
        vertexShader: FULLSCREEN_VERT,
        fragmentShader: COMPOSITOR_POST_FRAG,
        uniforms: {
          uBgTex: { value: null },
          uFgTex: { value: null },
          uPrevTex: { value: null },
          uDualLayer: { value: 1 },
          uTime: { value: 0 },
          uEnergy: { value: 0 },
          uKick: { value: 0 },
          uFxBloom: { value: 1 },
          uFxTrails: { value: 1 },
          uFxChroma: { value: 1 },
          uFxGlitch: { value: 0 },
          uFxStrobe: { value: 0 },
          uFxKaleido: { value: 0 },
          uBlackout: { value: 0 },
          uBpm: { value: 126 },
        },
        depthTest: false,
        depthWrite: false,
      });

      this.blitMat = new THREE.ShaderMaterial({
        vertexShader: FULLSCREEN_VERT,
        fragmentShader: BLIT_FRAG,
        uniforms: { uTex: { value: null } },
        depthTest: false,
        depthWrite: false,
      });
    } catch (err) {
      console.warn("[Lightshow] Three.js WebGL unavailable:", err);
      this.renderer = null;
      this.initFallback();
    }
  }

  private initFallback(): void {
    try {
      this.fallbackCtx = this.canvas.getContext("2d");
    } catch {
      this.fallbackCtx = null;
    }
    this.canvas.style.background = "radial-gradient(ellipse, #28194a, #050509)";
    this.canvas.title = "WebGL unavailable; using the audio-reactive 2D lightshow.";
    console.info("[Lightshow] Using the 2D fallback (WebGL unavailable)");
  }

  private renderFallback(frame: LightshowAudioFrame): void {
    const ctx = this.fallbackCtx;
    if (!ctx) return;
    const w = Math.max(1, Math.round(this.canvas.clientWidth || 640));
    const h = Math.max(1, Math.round(this.canvas.clientHeight || 400));
    // Same bounded pixel budget as the GPU path; CSS supplies display size.
    const scale = Math.min(1, Math.sqrt(600_000 / (w * h)));
    const rw = Math.max(1, Math.round(w * scale));
    const rh = Math.max(1, Math.round(h * scale));
    if (this.canvas.width !== rw || this.canvas.height !== rh) {
      this.canvas.width = rw;
      this.canvas.height = rh;
    }
    ctx.fillStyle = "#050509";
    ctx.fillRect(0, 0, rw, rh);
    if (this.state.blackout) return;
    const energy = Math.max(0, Math.min(1, frame.mix));
    const gradient = ctx.createRadialGradient(rw / 2, rh / 2, 0, rw / 2, rh / 2, Math.max(rw, rh) * 0.7);
    gradient.addColorStop(0, `hsla(${(frame.timeSec * 12) % 360}, 85%, 45%, ${0.2 + energy * 0.5})`);
    gradient.addColorStop(1, "#050509");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, rw, rh);
    const bins = frame.freqBins;
    ctx.fillStyle = "#8cecff";
    for (let i = 0; i < 48; i++) {
      const magnitude = bins?.length ? bins[Math.floor(i * bins.length / 48)] / 255 : energy;
      const height = Math.max(1, magnitude * rh * 0.7);
      ctx.fillRect(i * rw / 48, (rh - height) / 2, Math.max(1, rw / 48 - 2), height);
    }
  }

  public async loadSuperAcidBank(url?: string): Promise<void> {
    const resolvedUrl = url ?? `${import.meta.env.BASE_URL}data/superacid_presets.json`;
    try {
      const res = await fetch(resolvedUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as SuperAcidBundleJson;
      if (!json || !Array.isArray(json.presets)) return;
      this.headers = json.headers ?? {};
      const merged: LightshowPresetMeta[] = [...json.presets, SPACETUBES_BUILTIN];
      this.presets = merged;
      this.presetMap.clear();
      for (const p of merged) this.presetMap.set(p.id, p);
      // Clear cached builtins cleanly if overridden by full SuperAcid bank
      this.materialCache.forEach((m) => m.dispose());
      this.materialCache.clear();
      this.onStateSync?.(this.presets, this.state.preset, this.state.fgPreset);
      console.info(`[Lightshow] Loaded ${merged.length} SuperAcid + SpaceTubes visual scenes.`);
    } catch (err) {
      console.warn("[Lightshow] Could not load superacid_presets.json; using builtin shaders:", err);
    }
  }

  public getPresets(): readonly LightshowPresetMeta[] {
    return this.presets;
  }

  public getState(): LightshowState {
    return { ...this.state };
  }

  public setState(patch: Partial<LightshowState>): void {
    this.state = { ...this.state, ...patch };
    this.onStateSync?.(this.presets, this.state.preset, this.state.fgPreset);
  }

  public stepPreset(delta: number): void {
    if (this.presets.length === 0) return;
    const idx = Math.max(0, this.presets.findIndex((p) => p.id === this.state.preset));
    const nextIdx = (idx + delta + this.presets.length) % this.presets.length;
    const picked = this.presets[nextIdx];
    this.setState({ preset: picked.id });
  }

  public randomPreset(): void {
    const bgs = this.presets.filter((p) => p.role === "bg");
    const fgs = this.presets.filter((p) => p.role === "fg");
    const bgPool = bgs.length > 0 ? bgs : this.presets;
    const nextBg = bgPool[Math.floor(Math.random() * bgPool.length)];
    const nextFg = fgs.length > 0 ? fgs[Math.floor(Math.random() * fgs.length)] : nextBg;
    this.setState({ preset: nextBg.id, fgPreset: nextFg.id });
  }

  public pulseStrobeBriefly(ms: number = 320): void {
    const prev = this.state.fxStrobe;
    this.state.fxStrobe = true;
    setTimeout(() => {
      this.state.fxStrobe = prev;
    }, ms);
  }

  private getMaterialForPreset(id: string): THREE.ShaderMaterial | null {
    const cached = this.materialCache.get(id);
    if (cached) return cached;

    const meta = this.presetMap.get(id) ?? this.presetMap.get("acid-warp") ?? this.presets[0];
    if (!meta) return null;

    let frag = meta.frag;
    for (const [token, hdr] of Object.entries(this.headers)) {
      frag = frag.replace(token, hdr);
    }
    if (!frag.includes("precision ")) {
      frag = `precision highp float;\n${frag}`;
    }

    const uniforms: Record<string, THREE.IUniform<number>> = { ...this.sharedUniforms };
    for (const [k, v] of Object.entries(meta.defaults || {})) {
      if (!(k in uniforms)) {
        uniforms[k] = { value: Number.isFinite(v) ? v : 0.6 };
      }
    }

    const mat = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: frag,
      uniforms,
      depthTest: false,
      depthWrite: false,
    });

    if (this.materialCache.size >= this.maxCachedMaterials) {
      const oldestKey = this.materialCache.keys().next().value;
      if (oldestKey) {
        this.materialCache.get(oldestKey)?.dispose();
        this.materialCache.delete(oldestKey);
      }
    }
    this.materialCache.set(id, mat);
    return mat;
  }

  private ensureTargets(): boolean {
    if (!this.renderer) return false;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(240, Math.round(rect.width || this.canvas.width || 640));
    const h = Math.max(160, Math.round(rect.height || this.canvas.height || 400));
    const scale = Math.min(1.5, window.devicePixelRatio || 1, Math.sqrt(600_000 / (w * h)));
    const rw = Math.max(1, Math.round(w * scale));
    const rh = Math.max(1, Math.round(h * scale));

    if (this.rtW !== rw || this.rtH !== rh) {
      this.renderer.setSize(rw, rh, false);
      this.bgTarget?.dispose();
      this.fgTarget?.dispose();
      this.trailTargets[0]?.dispose();
      this.trailTargets[1]?.dispose();
      const opts: THREE.RenderTargetOptions = {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        depthBuffer: false,
        stencilBuffer: false,
      };
      this.bgTarget = new THREE.WebGLRenderTarget(rw, rh, opts);
      this.fgTarget = new THREE.WebGLRenderTarget(rw, rh, opts);
      this.trailTargets = [
        new THREE.WebGLRenderTarget(rw, rh, opts),
        new THREE.WebGLRenderTarget(rw, rh, opts),
      ];
      this.rtW = rw;
      this.rtH = rh;
    }
    return Boolean(this.bgTarget && this.fgTarget && this.trailTargets[0] && this.trailTargets[1]);
  }

  public render(frame: LightshowAudioFrame): void {
    if (!this.state.enabled || document.hidden || this.contextLost) return;
    if (!this.rendererAttempted) this.initRenderer();
    if (!this.renderer) {
      this.renderFallback(frame);
      return;
    }
    if (!this.postMat || !this.blitMat) return;
    if (!this.ensureTargets()) return;

    const uTime = (performance.now() - this.startMs) * 0.001;

    // CORTEX Auto-VJ: change preset on section changes OR every ~24s of active music
    if (this.state.cortexAuto && frame.section) {
      const secChanged = frame.section !== this.lastSection;
      const timedRotate = uTime - this.lastRotateSec > 24 && frame.mix > 0.05;
      if ((secChanged && this.lastSection !== "") || timedRotate) {
        this.lastRotateSec = uTime;
        this.randomPreset();
      }
      this.lastSection = frame.section;
    }

    // Update audio uniforms shared across all 268 shaders
    const bpm = Math.max(60, frame.bpm || 126);
    const beatFrac = ((frame.timeSec * bpm) / 60) % 1;
    const beatAngle = frame.timeSec * (bpm / 60) * Math.PI * 2;
    const u = this.sharedUniforms;
    u.uTime.value = uTime;
    u.uAudioTime.value = frame.timeSec;
    u.uBpm.value = bpm;
    u.uBeat.value = beatFrac;
    u.uBpmSin.value = Math.sin(beatAngle);
    u.uBpmSin2.value = Math.sin(beatAngle * 2);
    u.uBpmSin4.value = Math.sin(beatAngle * 4);
    u.uKick.value = frame.drumPunch;
    u.uOnset.value = frame.drumPunch;
    u.uEnergy.value = frame.mix;
    u.uIntensity.value = this.state.intensity;
    u.uSpeed.value = this.state.speed;
    u.uPalette.value = this.state.palette;
    u.uWarmth.value = frame.bass;
    u.uBrightness.value = frame.other * 0.6 + frame.drums * 0.4;
    u.uMusicDensity.value = Math.min(1, (frame.drums + frame.bass + frame.other + frame.vocals) * 0.45);
    u.uForwardFlight.value = frame.mix * 1.2;

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
        const amp = sum / Math.max(1, (hi - lo) * 255);
        u[`uSpectrum${b}`].value = amp;
        u[`uBandFlux${b}`].value = amp * frame.drumPunch;
      }
    } else {
      const vals = [
        frame.bass,
        frame.drums * 0.8 + frame.bass * 0.4,
        frame.other * 0.7,
        frame.vocals,
        frame.vocals * 0.8 + frame.other * 0.5,
        frame.other,
        frame.drums * 0.9,
        frame.mix,
      ];
      for (let b = 0; b < 8; b++) {
        u[`uSpectrum${b}`].value = vals[b];
        u[`uBandFlux${b}`].value = vals[b] * frame.drumPunch;
      }
    }

    const bgMat = this.getMaterialForPreset(this.state.preset);
    if (!bgMat) return;

    // Pass 1: Render Base Scene
    this.quadMesh.material = bgMat;
    this.renderer.setRenderTarget(this.bgTarget);
    this.renderer.render(this.scene, this.camera);

    // Optional Pass 1B: Render Foreground Overlay Scene (BG+FG dual-layer SuperAcid compositing)
    const showDual = this.state.dualLayer && this.state.fgPreset && this.state.fgPreset !== this.state.preset;
    if (showDual) {
      const fgMat = this.getMaterialForPreset(this.state.fgPreset);
      if (fgMat) {
        this.quadMesh.material = fgMat;
        this.renderer.setRenderTarget(this.fgTarget);
        this.renderer.render(this.scene, this.camera);
      }
    }

    // Pass 2: Post-Process Compositor -> ping-pong trail target
    const prevIdx = this.pingIdx;
    const nextIdx = 1 - prevIdx;
    const pu = this.postMat.uniforms;
    pu.uBgTex.value = this.bgTarget!.texture;
    pu.uFgTex.value = this.fgTarget!.texture;
    pu.uPrevTex.value = this.trailTargets[prevIdx]!.texture;
    pu.uDualLayer.value = showDual ? 1 : 0;
    pu.uTime.value = uTime;
    pu.uEnergy.value = frame.mix;
    pu.uKick.value = frame.drumPunch;
    pu.uFxBloom.value = this.state.fxBloom ? 1 : 0;
    pu.uFxTrails.value = this.state.fxTrails ? 1 : 0;
    pu.uFxChroma.value = this.state.fxChroma ? 1 : 0;
    pu.uFxGlitch.value = this.state.fxGlitch ? 1 : 0;
    pu.uFxStrobe.value = this.state.fxStrobe ? 1 : 0;
    pu.uFxKaleido.value = this.state.fxKaleido ? 1 : 0;
    pu.uBlackout.value = this.state.blackout ? 1 : 0;
    pu.uBpm.value = bpm;

    this.quadMesh.material = this.postMat;
    this.renderer.setRenderTarget(this.trailTargets[nextIdx]);
    this.renderer.render(this.scene, this.camera);

    // Pass 3: Blit final result to #lightshow-canvas
    this.blitMat.uniforms.uTex.value = this.trailTargets[nextIdx]!.texture;
    this.quadMesh.material = this.blitMat;
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);

    this.pingIdx = nextIdx;
  }

  public dispose(): void {
    this.materialCache.forEach((m) => m.dispose());
    this.materialCache.clear();
    this.postMat?.dispose();
    this.blitMat?.dispose();
    this.quadGeo.dispose();
    this.bgTarget?.dispose();
    this.fgTarget?.dispose();
    this.trailTargets[0]?.dispose();
    this.trailTargets[1]?.dispose();
    this.renderer?.dispose();
  }
}
