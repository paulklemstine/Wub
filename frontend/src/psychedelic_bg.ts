// Psychedelic DMT / LSD / Mescaline Self-Reactive Light Show Engine
// WebGL full-screen real-time shader with organic melting, kaleidoscopic fractal folding,
// neon color cycling, and multi-band audio reactivity.

export type TripMode = "hybrid" | "dmt" | "lsd" | "mescaline";
export type TripPalette = "neon" | "acid" | "solar" | "cyber";

export interface TripControllerState {
  enabled: boolean;
  mode: TripMode;
  palette: TripPalette;
  intensity: number; // 0.0 - 1.5
  speed: number;     // 0.2 - 3.0
  audioReactive: boolean;
}

const VERT_SHADER_SRC = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = (a_position + 1.0) * 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAG_SHADER_SRC = `
precision highp float;

varying vec2 v_uv;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
uniform float u_level;
uniform float u_intensity;
uniform float u_speed;
uniform int u_mode;     // 0: Hybrid, 1: DMT, 2: LSD Melting, 3: Mescaline
uniform int u_palette;  // 0: Neon Hyper, 1: Acid Rainbow, 2: Solar Amber, 3: Cyber UV

#define PI 3.14159265359
#define TAU 6.28318530718

// 2D Rotation matrix
mat2 rot(float a) {
  float s = sin(a), c = cos(a);
  return mat2(c, -s, s, c);
}

// Pseudo-random hash
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// 2D Value Noise
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
        dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
    mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
        dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x),
    u.y
  );
}

// 4-Octave Fractal Brownian Motion for liquid melting
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 m = rot(0.42);
  for (int i = 0; i < 4; i++) {
    v += a * noise(p);
    p = m * p * 2.04;
    a *= 0.5;
  }
  return v;
}

// Inigo Quilez cosine neon palette generator
vec3 neonCosine(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(TAU * (c * t + d));
}

vec3 getPaletteColor(float t, int pal) {
  vec3 col;
  if (pal == 0) {
    // Hyper-Vivid Electric Neon DMT (Hot Magenta, Electric Cyan, Acid Lime, Solar Gold)
    col = neonCosine(t, vec3(0.68, 0.62, 0.72), vec3(0.58, 0.52, 0.62), vec3(1.2, 1.0, 1.0), vec3(0.00, 0.33, 0.67));
  } else if (pal == 1) {
    // Blazing Acid Rainbow LSD (Saturated Ultraviolet, Neon Emerald, Fiery Orange, Electric Azure)
    col = neonCosine(t, vec3(0.72, 0.66, 0.66), vec3(0.62, 0.58, 0.58), vec3(2.0, 1.2, 0.8), vec3(0.50, 0.20, 0.25));
  } else if (pal == 2) {
    // Solar Mescaline Fire & Electric Turquoise
    col = neonCosine(t, vec3(0.92, 0.65, 0.42), vec3(0.65, 0.58, 0.48), vec3(1.2, 1.0, 0.8), vec3(0.12, 0.28, 0.60));
  } else {
    // Cyberpunk Neon Ultraviolet & Toxic Lime
    col = neonCosine(t, vec3(0.62, 0.48, 0.85), vec3(0.72, 0.62, 0.68), vec3(1.6, 1.4, 0.9), vec3(0.75, 0.12, 0.45));
  }
  return col * 1.45;
}

// --- 1. DMT Hyperbolic Fractal Kaleidoscope ---
vec3 renderDMT(vec2 uv, float t) {
  // Self-reactive breathing + bass punch
  float bassPulse = u_bass * 0.45;
  vec2 p = uv * (2.0 - bassPulse);

  // Seamless borderless radial texture wrapping
  float r = length(p);
  float a = atan(p.y, p.x);
  float sectors = 6.0;
  float twist = t * 0.12 + u_mid * 0.3;
  float sectorAngle = TAU / sectors;
  float na = (a + twist) / sectorAngle;
  // Continuous triangle wave reflection ensures seamless borderless angular wrapping without discontinuities
  float wrappedA = (abs(fract(na) - 0.5) * 2.0 - 0.5) * sectorAngle;
  p = vec2(cos(wrappedA), sin(wrappedA)) * r;

  // Multi-tier DMT liquid melting domain warping
  vec2 meltWarp1 = vec2(
    fbm(p * 1.4 + vec2(0.0, t * 0.32)),
    fbm(p * 1.4 + vec2(4.6, -t * 0.28))
  );
  vec2 meltWarp2 = vec2(
    fbm(p * 2.2 + 2.8 * meltWarp1 + vec2(1.8, -t * 0.22) + vec2(2.3, 7.5)),
    fbm(p * 2.2 + 2.8 * meltWarp1 + vec2(8.1, 3.4))
  );
  p += meltWarp2 * (0.65 + 0.55 * u_bass);

  // Hyperbolic iterated folding (Mandelbox-style machine elf geometry)
  float glowAccum = 0.0;
  float foldScale = 1.0;
  for (int i = 0; i < 5; i++) {
    // Seamless periodic folding
    p = abs(p) - vec2(0.42 + 0.12 * sin(t * 0.35), 0.32 + 0.08 * cos(t * 0.28));
    p *= rot(0.55 + t * 0.06 + u_treble * 0.25);
    float d = dot(p, p);
    p = p / max(d, 0.14) - vec2(0.45, 0.35);
    foldScale *= 1.35;
    glowAccum += exp(-abs(d - 0.48) * 4.2) / float(i + 1);
  }

  // Smooth borderless color cycling
  float colorShift = t * 0.18 + length(p) * 0.08 + glowAccum * 0.28;
  vec3 baseCol = getPaletteColor(colorShift, u_palette);
  
  // Neon edge luminescence & psychedelic melt glow
  vec3 neonCol = baseCol * (glowAccum * 1.6 + 0.25);
  neonCol += vec3(0.15, 0.45, 0.7) * (u_bass * 0.85);
  return neonCol;
}

// --- 2. LSD Organic Melting Fluid ---
vec3 renderLSD(vec2 uv, float t) {
  vec2 p = uv * 2.4;

  // Multi-tier fluid domain warping
  float meltSpeed = t * 0.25;
  vec2 q = vec2(
    fbm(p + vec2(0.0, meltSpeed * 0.8)),
    fbm(p + vec2(5.2, -meltSpeed * 0.6))
  );

  // Downward liquid dripping & chromatic turbulence
  vec2 drip = vec2(0.0, -t * 0.2 - q.y * 0.4);
  vec2 r = vec2(
    fbm(p + 3.2 * q + drip + vec2(1.7, 9.2)),
    fbm(p + 3.2 * q + drip + vec2(8.3, 2.8))
  );

  vec2 meltedP = p + (r * (1.3 + u_bass * 0.7));

  // Chromatic dispersion (RGB split at fluid edges)
  float disp = 0.035 * (1.0 + u_treble * 1.5);
  float rCh = fbm(meltedP * 1.8 + vec2(disp, 0.0));
  float gCh = fbm(meltedP * 1.8);
  float bCh = fbm(meltedP * 1.8 - vec2(disp, 0.0));

  float colIndex = t * 0.2 + gCh * 1.4 + length(q) * 0.6;
  vec3 fluidCol = getPaletteColor(colIndex, u_palette);

  // Blend with chromatic dispersion
  fluidCol.r = mix(fluidCol.r, rCh * 1.4, 0.4);
  fluidCol.b = mix(fluidCol.b, bCh * 1.4, 0.4);

  // Melting liquid gloss / specularity
  float gloss = pow(max(0.0, gCh - 0.2), 3.0) * 2.2;
  fluidCol += vec3(gloss);

  return fluidCol;
}

// --- 3. Mescaline Sacred Aztec/Mayan Crystal Tapestry ---
vec3 renderMescaline(vec2 uv, float t) {
  vec2 p = uv * 3.6;
  p *= rot(t * 0.04);

  // Aztec stepped lattice tessellation
  vec2 grid = abs(fract(p - 0.5) - 0.5);
  float diamond = abs(grid.x - grid.y);
  float square = max(grid.x, grid.y);
  float lattice = min(diamond, square);

  // Fractal concentric diamond rings
  float rings = abs(sin(length(p) * 6.0 - t * 0.8 - u_bass * 4.0));
  float stepped = floor(rings * 4.0) / 4.0;

  // Pulsating laser glow along sacred geometries
  float glow = 0.038 / (lattice + 0.02 + 0.03 * (1.0 - min(1.0, u_bass * 1.5)));
  glow += 0.025 / (abs(diamond - 0.25) + 0.03);

  float colIndex = t * 0.12 + stepped * 0.5 + lattice * 0.8;
  vec3 tapestryCol = getPaletteColor(colIndex, u_palette);
  tapestryCol *= (glow * 1.4 + 0.25);

  return tapestryCol;
}

void main() {
  // Normalize screen coordinates centered at origin
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);

  // Self-reactive continuous time with speed scaling
  float t = u_time * u_speed;

  // Gentle autonomous breathing & vortex sway
  float breath = sin(t * 0.4) * 0.08 + cos(t * 0.25) * 0.05;
  uv *= (1.0 + breath - u_bass * 0.15);
  uv *= rot(sin(t * 0.15) * 0.1 + u_mid * 0.15);

  vec3 finalColor = vec3(0.0);

  if (u_mode == 1) {
    // Pure DMT Hyperbolic Fractal
    finalColor = renderDMT(uv, t);
  } else if (u_mode == 2) {
    // Pure LSD Melting Fluid
    finalColor = renderLSD(uv, t);
  } else if (u_mode == 3) {
    // Pure Mescaline Sacred Tapestry
    finalColor = renderMescaline(uv, t);
  } else {
    // Hybrid Mode: Seamless continuous psychedelic transformation
    float morphCycle = t * 0.08;
    float w1 = 0.5 + 0.5 * sin(morphCycle);
    float w2 = 0.5 + 0.5 * sin(morphCycle + 2.094);
    float w3 = 0.5 + 0.5 * sin(morphCycle + 4.188);
    float sumW = w1 + w2 + w3;
    w1 /= sumW; w2 /= sumW; w3 /= sumW;

    vec3 cDmt = renderDMT(uv, t);
    vec3 cLsd = renderLSD(uv, t);
    vec3 cMes = renderMescaline(uv, t);
    finalColor = cDmt * w1 + cLsd * w2 + cMes * w3;
  }

  // Hyper-saturated neon bloom & radiance curve to make colors intensely bright and pop
  finalColor = pow(finalColor, vec3(0.78)) * 1.85;

  // Vivid neon highlights & specular pop
  finalColor += pow(finalColor, vec3(2.0)) * 0.55;

  // Audio bass drop shockwave burst
  finalColor += vec3(0.35, 0.22, 0.45) * pow(u_bass, 1.5);

  // Master intensity scaling
  finalColor *= u_intensity;

  gl_FragColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);
}
`;

export class PsychedelicBackground {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;

  // Uniform locations
  private uResolutionLoc: WebGLUniformLocation | null = null;
  private uTimeLoc: WebGLUniformLocation | null = null;
  private uBassLoc: WebGLUniformLocation | null = null;
  private uMidLoc: WebGLUniformLocation | null = null;
  private uTrebleLoc: WebGLUniformLocation | null = null;
  private uLevelLoc: WebGLUniformLocation | null = null;
  private uIntensityLoc: WebGLUniformLocation | null = null;
  private uSpeedLoc: WebGLUniformLocation | null = null;
  private uModeLoc: WebGLUniformLocation | null = null;
  private uPaletteLoc: WebGLUniformLocation | null = null;

  // Audio smoothing filters
  private smoothBass = 0;
  private smoothMid = 0;
  private smoothTreble = 0;
  private smoothLevel = 0;

  // State
  private state: TripControllerState = {
    enabled: true,
    mode: "hybrid",
    palette: "neon",
    intensity: 1.4,
    speed: 1.1,
    audioReactive: true,
  };

  private startTime = performance.now();
  private lastTime = performance.now();
  private accumulatedTime = 0;

  constructor(canvasId = "psychedelic-bg") {
    let el = document.getElementById(canvasId) as HTMLCanvasElement | null;
    if (!el) {
      el = document.createElement("canvas");
      el.id = canvasId;
      document.body.insertBefore(el, document.body.firstChild);
    }
    this.canvas = el;
    this.initWebGL();
    this.initResize();
  }

  private initWebGL(): void {
    const gl =
      this.canvas.getContext("webgl", {
        alpha: false,
        depth: false,
        stencil: false,
        antialias: false,
        powerPreference: "high-performance",
      }) ||
      (this.canvas.getContext("experimental-webgl") as WebGLRenderingContext | null);

    if (!gl) {
      console.warn("[PsychedelicBG] WebGL not supported; psychedelic background disabled.");
      return;
    }
    this.gl = gl;

    // Create shaders
    const vertShader = this.compileShader(gl.VERTEX_SHADER, VERT_SHADER_SRC);
    const fragShader = this.compileShader(gl.FRAGMENT_SHADER, FRAG_SHADER_SRC);
    if (!vertShader || !fragShader) return;

    const prog = gl.createProgram();
    if (!prog) return;
    gl.attachShader(prog, vertShader);
    gl.attachShader(prog, fragShader);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("[PsychedelicBG] Link failed:", gl.getProgramInfoLog(prog));
      return;
    }
    this.program = prog;

    // Quad geometry (2 triangles covering clip space -1..1)
    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    const verts = new Float32Array([
      -1.0, -1.0,
       1.0, -1.0,
      -1.0,  1.0,
      -1.0,  1.0,
       1.0, -1.0,
       1.0,  1.0,
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

    const posAttr = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(posAttr);
    gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);

    // Uniform locations
    this.uResolutionLoc = gl.getUniformLocation(prog, "u_resolution");
    this.uTimeLoc = gl.getUniformLocation(prog, "u_time");
    this.uBassLoc = gl.getUniformLocation(prog, "u_bass");
    this.uMidLoc = gl.getUniformLocation(prog, "u_mid");
    this.uTrebleLoc = gl.getUniformLocation(prog, "u_treble");
    this.uLevelLoc = gl.getUniformLocation(prog, "u_level");
    this.uIntensityLoc = gl.getUniformLocation(prog, "u_intensity");
    this.uSpeedLoc = gl.getUniformLocation(prog, "u_speed");
    this.uModeLoc = gl.getUniformLocation(prog, "u_mode");
    this.uPaletteLoc = gl.getUniformLocation(prog, "u_palette");
  }

  private compileShader(type: number, src: string): WebGLShader | null {
    const gl = this.gl;
    if (!gl) return null;
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("[PsychedelicBG] Shader compilation failed:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  private initResize(): void {
    const resize = () => {
      // Downscale slightly on high-density / 4K displays to preserve 60fps
      const dpr = Math.min(1.25, window.devicePixelRatio || 1);
      const w = Math.round(window.innerWidth * dpr);
      const h = Math.round(window.innerHeight * dpr);
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
    };
    window.addEventListener("resize", resize);
    resize();
  }

  public setState(patch: Partial<TripControllerState>): void {
    this.state = { ...this.state, ...patch };
  }

  public getState(): TripControllerState {
    return { ...this.state };
  }

  /**
   * Render one frame of the psychedelic light show.
   * Self-reactive math guarantees continuous motion even when audio is silent.
   */
  public render(audioData?: {
    bassLevel?: number;
    midLevel?: number;
    trebleLevel?: number;
    masterLevel?: number;
  }): void {
    const gl = this.gl;
    const prog = this.program;
    if (!gl || !prog) return;

    if (!this.state.enabled) {
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.04, 0.05, 0.08, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }

    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastTime) / 1000);
    this.lastTime = now;
    this.accumulatedTime += dt;

    // Audio reactivity with exponential smoothing
    let targetBass = 0;
    let targetMid = 0;
    let targetTreble = 0;
    let targetLevel = 0;

    if (this.state.audioReactive && audioData) {
      targetBass = audioData.bassLevel ?? 0;
      targetMid = audioData.midLevel ?? 0;
      targetTreble = audioData.trebleLevel ?? 0;
      targetLevel = audioData.masterLevel ?? 0;
    }

    // Smooth audio pulses
    const smoothFactor = 0.18;
    this.smoothBass += (targetBass - this.smoothBass) * smoothFactor;
    this.smoothMid += (targetMid - this.smoothMid) * smoothFactor;
    this.smoothTreble += (targetTreble - this.smoothTreble) * smoothFactor;
    this.smoothLevel += (targetLevel - this.smoothLevel) * smoothFactor;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(prog);

    // Uniforms
    gl.uniform2f(this.uResolutionLoc, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.uTimeLoc, this.accumulatedTime);
    gl.uniform1f(this.uBassLoc, this.smoothBass);
    gl.uniform1f(this.uMidLoc, this.smoothMid);
    gl.uniform1f(this.uTrebleLoc, this.smoothTreble);
    gl.uniform1f(this.uLevelLoc, this.smoothLevel);
    gl.uniform1f(this.uIntensityLoc, this.state.intensity);
    gl.uniform1f(this.uSpeedLoc, this.state.speed);

    // Mode: 0: Hybrid, 1: DMT, 2: LSD, 3: Mescaline
    const modeInt =
      this.state.mode === "dmt"
        ? 1
        : this.state.mode === "lsd"
        ? 2
        : this.state.mode === "mescaline"
        ? 3
        : 0;
    gl.uniform1i(this.uModeLoc, modeInt);

    // Palette: 0: Neon, 1: Acid, 2: Solar, 3: Cyber
    const palInt =
      this.state.palette === "acid"
        ? 1
        : this.state.palette === "solar"
        ? 2
        : this.state.palette === "cyber"
        ? 3
        : 0;
    gl.uniform1i(this.uPaletteLoc, palInt);

    // Draw full-screen quad
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
