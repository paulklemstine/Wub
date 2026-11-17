// -----------------------------------------------------------------------------
// WubGotchi: Neural Music Tamagotchi Avatar
// Autonomous ASCII character that listens, dances, sings, and evolves with the music.
// -----------------------------------------------------------------------------

import type { GrooveboxMeta } from "./main";
import {
  Bear3DView,
  BEAR_CAMERA_LABELS,
  DEFAULT_BEAR_VISUAL_CONFIG,
  type BearCameraPreset,
  type BearMouthStyle,
  type BearVisualConfig,
} from "./bear_3d_view";
import type { LiveMouthState } from "./singing_head";

export type TamagotchiStage = 1 | 2 | 3 | 4;

export interface TamagotchiStats {
  xp: number;
  stage: TamagotchiStage;
  stageName: string;
  vibe: number;       // 0..100%
  mood: string;
  affection: number;  // 0..10
  favoriteStem: string;
  barsListened: number;
}

const STAGE_CONFIGS: Record<TamagotchiStage, { name: string; minXp: number }> = {
  1: { name: "GUMMI CUB (RAVE ORB)", minXp: 0 },
  2: { name: "NEON GUMMI (DISCO ORB)", minXp: 250 },
  3: { name: "CYBER GUMMI (CYBER ORB)", minXp: 750 },
  4: { name: "COSMIC GUMMI (ASTRAL ORB)", minXp: 1800 },
};

const MUSIC_NOTES = ["✨", "💖", "🐾", "🌈", "✦", "✧", "💫", "♬", "⭐", "♡", "🍬", "🍭"];

export interface GummiLoopVertex {
  x: number;
  y: number;
  nx: number;
  ny: number;
  u: number;
}

/**
 * Builds the smooth closed Catmull-Rom parametric loop for the floating kawaii
 * HEART head of the Wub Creature (two round top lobes, a soft central notch and
 * a tapered bottom point — no ears, no circular orb).
 * Outward unit normals are precomputed for seamless audio-reactive wave displacement.
 */
function buildGummiHeartPoints(): GummiLoopVertex[] {
  const halfPts: [number, number][] = [
    [0.0, -34.0],     // 0: soft central heart notch (top dip)
    [11.0, -40.0],    // 1: notch shoulder
    [21.0, -47.5],    // 2: inner lobe swell
    [31.0, -52.0],    // 3: left lobe crown
    [41.0, -49.5],    // 4: outer lobe round
    [48.0, -40.0],    // 5: upper heart side
    [51.0, -28.0],    // 6: widest cheek line
    [50.0, -14.0],    // 7: cheek
    [46.0, 0.0],      // 8: lower cheek
    [39.5, 14.0],     // 9: jaw taper
    [30.5, 27.5],     // 10: chin taper
    [19.5, 39.0],     // 11: lower point slope
    [9.5, 47.5],      // 12: tip shoulder
    [0.0, 52.0],      // 13: rounded bottom heart point
  ];

  const leftPts: [number, number][] = [];
  for (let i = halfPts.length - 2; i >= 1; i--) {
    leftPts.push([-halfPts[i][0], halfPts[i][1]]);
  }
  const allPts = halfPts.concat(leftPts);
  const pLen = allPts.length;

  const sampled: { x: number; y: number }[] = [];
  const stepsPerSeg = 5;
  for (let i = 0; i < pLen; i++) {
    const p0 = allPts[(i - 1 + pLen) % pLen];
    const p1 = allPts[i];
    const p2 = allPts[(i + 1) % pLen];
    const p3 = allPts[(i + 2) % pLen];
    for (let s = 0; s < stepsPerSeg; s++) {
      const t = s / stepsPerSeg;
      const t2 = t * t;
      const t3 = t2 * t;
      const x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
      const y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
      sampled.push({ x, y });
    }
  }

  const n = sampled.length;
  const result: GummiLoopVertex[] = [];
  for (let i = 0; i < n; i++) {
    const prev = sampled[(i - 1 + n) % n];
    const next = sampled[(i + 1) % n];
    const tx = next.x - prev.x;
    const ty = next.y - prev.y;
    const len = Math.hypot(tx, ty);
    const nx = len > 1e-6 ? ty / len : 0.0;
    const ny = len > 1e-6 ? -tx / len : -1.0;
    result.push({ x: sampled[i].x, y: sampled[i].y, nx, ny, u: i / n });
  }
  return result;
}

const GUMMI_HEART_LOOP_POINTS: GummiLoopVertex[] = buildGummiHeartPoints();

export type VocalTimbre =
  | "SOULFUL"   // Warm rich diva / house / chest singing
  | "CYBER"     // Vocoder / vocoloid / metallic formant
  | "ETHEREAL"  // Wave / whisper / breathy falsetto
  | "GRITTY"    // Breaks / rap MC / biting staccato
  | "BELT";     // Maximum power stadium belt

export class TamagotchiAvatar {
  private containerEl: HTMLElement | null = null;
  private asciiEl: HTMLElement | null = null;
  private canvasEl: HTMLCanvasElement | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private lyricBarEl: HTMLElement | null = null;
  private stageBadgeEl: HTMLElement | null = null;
  private vibeEl: HTMLElement | null = null;
  private moodEl: HTMLElement | null = null;
  private xpEl: HTMLElement | null = null;

  // Pet State
  private xp: number = 0;
  private stage: TamagotchiStage = 1;
  private vibe: number = 85;
  private mood: string = "CHILL";
  private affection: number = 5;
  private favoriteStem: string = "bass";
  private barsListened: number = 0;

  // Animation & Beat Timing
  private lastRenderSec: number = 0;
  private lastBlinkSec: number = 0;
  private isBlinking: boolean = false;
  private isHeadBobbed: boolean = false;
  private noteFloating: { char: string; x: number; y: number; life: number }[] = [];
  private speechBubble: string = "";
  private heartBeatPhase: number = 0;
  private speechTimer: number = 0;
  private lastStepIndex: number = -1;
  private lastBarNum: number = -1;
  private temporaryAction: string | null = null;
  private actionTimer: number = 0;

  // Vocal Waveform & Mathematical Singing Mouth State
  private smoothedVocalAmp: number = 0;
  private singingLoopPhase: number = 0;

  // Vocal acoustics for mouth articulation (pitch + formants, tracked at
  // 30 Hz from the vocal analyser waveform at the engine's fixed 44100 Hz).
  // These drive mouth shape/width/height so the bear forms the actual sound.
  private vocalF0Hz: number = 0;
  private vocalF0Confidence: number = 0;
  private vocalF0MedianHz: number = 220;
  private vocalVibratoSemis: number = 0;
  private vocalF1: number = 0;
  private vocalF2: number = 0;
  private vocalVowelFormant: string = "";
  private vocalFrame: number = 0;

  // Per-stem reactive envelopes. Every visual feature reads one instrument so
  // the bear reads like a four-piece band: eyes=drums/bass, mouth=vocals,
  // heart=other, body/halo/motes=the full mix.
  private stemEnv = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private stemPunch = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private mixEnv: number = 0;
  private mixPunch: number = 0;
  private otherTone: number = 0;

  // Live Singing Head oral articulators + 3D Camera stage
  private liveMouth: LiveMouthState | null = null;
  private host3dEl: HTMLElement | null = null;
  private bear3dView: Bear3DView | null = null;
  private view3dActive: boolean = false;
  private btn3dToggleEl: HTMLElement | null = null;
  private btn3dCamEl: HTMLElement | null = null;
  private visualConfig: BearVisualConfig = { ...DEFAULT_BEAR_VISUAL_CONFIG };

  constructor() {
    this.loadState();
  }

  public initDOM(
    asciiId: string = "tamagotchi-ascii",
    stageBadgeId: string = "tamagotchi-stage-badge",
    vibeId: string = "tamagotchi-vibe",
    moodId: string = "tamagotchi-mood",
    xpId: string = "tamagotchi-xp"
  ): void {
    this.asciiEl = document.getElementById(asciiId);
    this.stageBadgeEl = document.getElementById(stageBadgeId);
    this.vibeEl = document.getElementById(vibeId);
    this.moodEl = document.getElementById(moodId);
    this.xpEl = document.getElementById(xpId);

    this.canvasEl = document.getElementById("tamagotchi-canvas") as HTMLCanvasElement | null;
    if (this.canvasEl) {
      this.ctx2d = this.canvasEl.getContext("2d");
    }
    this.lyricBarEl = document.getElementById("tamagotchi-lyric-bar");

    // Bind action buttons
    const btnPet = document.getElementById("btn-tamagotchi-pet");
    if (btnPet) btnPet.addEventListener("click", () => this.pet());

    const btnFeed = document.getElementById("btn-tamagotchi-feed");
    if (btnFeed) btnFeed.addEventListener("click", () => this.feed());

    const btnHype = document.getElementById("btn-tamagotchi-hype");
    if (btnHype) btnHype.addEventListener("click", () => this.hype());

    this.host3dEl = document.getElementById("tamagotchi-3d-host");
    this.btn3dToggleEl = document.getElementById("btn-tamagotchi-3d");
    this.btn3dCamEl = document.getElementById("btn-tamagotchi-cam");
    if (this.btn3dToggleEl) {
      this.btn3dToggleEl.addEventListener("click", () => this.toggle3DMode());
    }
    if (this.btn3dCamEl) {
      this.btn3dCamEl.addEventListener("click", () => this.cycle3DCamera());
    }
    this.sync3DButtons(this.bear3dView?.getCameraPreset() ?? "three");
    this.bindVisualControlBar();

    this.render();
  }

  public getVisualConfig(): BearVisualConfig {
    return { ...this.visualConfig };
  }

  public setVisualConfig(patch: Partial<BearVisualConfig>): void {
    this.visualConfig = { ...this.visualConfig, ...patch };
  }

  private bindVisualControlBar(): void {
    const toggleVisualBool = (id: string, key: "showCage" | "showVisor" | "showHalo" | "showBikini", label: string) => {
      const btn = document.getElementById(id);
      if (!btn) return;
      const sync = () => {
        const on = this.visualConfig[key];
        btn.classList.toggle("active", on);
        btn.textContent = `${label}: ${on ? "ON" : "OFF"}`;
      };
      sync();
      btn.addEventListener("click", () => {
        this.visualConfig[key] = !this.visualConfig[key];
        sync();
      });
    };
    toggleVisualBool("btn-bear-cage", "showCage", "CAGE");
    toggleVisualBool("btn-bear-visor", "showVisor", "VISOR");
    toggleVisualBool("btn-bear-halo", "showHalo", "HALO");
    toggleVisualBool("btn-bear-bikini", "showBikini", "BOW");

    const texBtn = document.getElementById("btn-bear-tex");
    if (texBtn) {
      texBtn.addEventListener("click", () => {
        if (!this.view3dActive) {
          this.set3DMode(true);
        }
        this.randomize3DTextures();
      });
    }

    const mouthStyleBtn = document.getElementById("btn-bear-mouth-style");
    if (mouthStyleBtn) {
      const styles: BearMouthStyle[] = ["real", "acid", "cyber"];
      const syncMouth = () => {
        mouthStyleBtn.textContent = `MOUTH: ${this.visualConfig.mouthStyle.toUpperCase()}`;
      };
      syncMouth();
      mouthStyleBtn.addEventListener("click", () => {
        const idx = styles.indexOf(this.visualConfig.mouthStyle);
        this.visualConfig.mouthStyle = styles[(idx + 1) % styles.length];
        syncMouth();
      });
    }

    const danceSlider = document.getElementById("bear-dance-slider") as HTMLInputElement | null;
    const danceVal = document.getElementById("bear-dance-val");
    if (danceSlider) {
      danceSlider.addEventListener("input", () => {
        const v = Number(danceSlider.value);
        if (Number.isFinite(v)) {
          this.visualConfig.danceGain = Math.max(0.3, Math.min(2.0, v));
          if (danceVal) danceVal.textContent = `${Math.round(v * 100)}%`;
        }
      });
    }

    const mouthSlider = document.getElementById("bear-mouth-slider") as HTMLInputElement | null;
    const mouthVal = document.getElementById("bear-mouth-val");
    if (mouthSlider) {
      mouthSlider.addEventListener("input", () => {
        const v = Number(mouthSlider.value);
        if (Number.isFinite(v)) {
          this.visualConfig.mouthGain = Math.max(0.6, Math.min(1.8, v));
          if (mouthVal) mouthVal.textContent = `${Math.round(v * 100)}%`;
        }
      });
    }
  }

  public randomize3DTextures(): void {
    this.bear3dView?.randomizeTextures();
  }

  public set3DMode(active: boolean): void {
    // Build the WebGL scene on first activation only: constructing the Three.js
    // stage, geometries and shaders at page load burned CPU for a view most
    // sessions never open.
    this.view3dActive = active ? this.ensure3DView() : false;
    if (this.canvasEl && this.canvasEl.dataset.offscreen !== "true") {
      this.canvasEl.style.display = this.view3dActive ? "none" : "block";
    }
    this.bear3dView?.setActive(this.view3dActive);
    this.sync3DButtons(this.bear3dView?.getCameraPreset() ?? "three");
  }

  /** Lazily create the 3D Wub Bear scene. Returns false if it cannot start. */
  private ensure3DView(): boolean {
    if (this.bear3dView !== null) return true;
    if (this.host3dEl === null) return false;
    try {
      this.bear3dView = new Bear3DView(this.host3dEl, (preset) => {
        this.sync3DButtons(preset);
      });
      return true;
    } catch (err) {
      console.warn("[tamagotchi] 3D view unavailable:", err);
      this.bear3dView = null;
      return false;
    }
  }

  public toggle3DMode(): void {
    this.set3DMode(!this.view3dActive);
  }

  public cycle3DCamera(): void {
    if (!this.view3dActive) {
      this.set3DMode(true);
      return;
    }
    const next = this.bear3dView?.cycleCameraPreset() ?? "three";
    this.sync3DButtons(next);
  }

  private sync3DButtons(preset: BearCameraPreset): void {
    if (this.btn3dToggleEl) {
      this.btn3dToggleEl.textContent = this.view3dActive ? "2D VIEW" : "3D CAM";
      this.btn3dToggleEl.classList.toggle("active", this.view3dActive);
    }
    if (this.btn3dCamEl) {
      this.btn3dCamEl.style.display = this.view3dActive ? "inline-flex" : "none";
      this.btn3dCamEl.textContent = `CAM: ${BEAR_CAMERA_LABELS[preset]}`;
    }
  }

  // ---------------------------------------------------------------------------
  // Learning & Interaction Triggers
  // ---------------------------------------------------------------------------

  public pet(): void {
    this.addXp(15);
    this.vibe = Math.min(100, this.vibe + 12);
    this.affection = Math.min(10, this.affection + 1);
    this.temporaryAction = "PET";
    this.actionTimer = 2.0;
    this.speechBubble = "💖 SQUISH! LOVE THIS GROOVE ✨";
    this.speechTimer = 2.5;
    this.saveState();
  }

  public feed(): void {
    this.addXp(25);
    this.vibe = Math.min(100, this.vibe + 15);
    this.temporaryAction = "EAT";
    this.actionTimer = 2.2;
    this.speechBubble = "🐻 NOM NOM! JUICY GUMMI SLICE! 🍬";
    this.speechTimer = 2.5;
    this.saveState();
  }

  public hype(): void {
    this.addXp(20);
    this.vibe = 100;
    this.temporaryAction = "HYPE";
    this.actionTimer = 3.0;
    this.speechBubble = "⚡ PSYCHEDELIC GUMMI BOUNCE! 🌈";
    this.speechTimer = 3.0;
    this.saveState();
  }

  public recordFeedback(stem: string, rating: number): void {
    if (rating > 0) {
      this.addXp(40);
      this.vibe = Math.min(100, this.vibe + 10);
      this.affection = Math.min(10, this.affection + 1);
      this.favoriteStem = stem;
      this.speechBubble = `YES! MORE ${stem.toUpperCase()}! ♥`;
      this.speechTimer = 2.5;
    } else {
      // Learns from mistakes
      this.addXp(15);
      this.vibe = Math.max(30, this.vibe - 5);
      this.speechBubble = `LEARNING... TIGHTENING ${stem.toUpperCase()}!`;
      this.speechTimer = 2.5;
    }
    this.saveState();
  }

  public addXp(amount: number): void {
    this.xp += amount;
    this.checkEvolution();
    this.updateDashboard();
  }

  private checkEvolution(): void {
    let nextStage: TamagotchiStage = 1;
    if (this.xp >= STAGE_CONFIGS[4].minXp) nextStage = 4;
    else if (this.xp >= STAGE_CONFIGS[3].minXp) nextStage = 3;
    else if (this.xp >= STAGE_CONFIGS[2].minXp) nextStage = 2;

    if (nextStage > this.stage) {
      this.stage = nextStage;
      this.vibe = 100;
      this.temporaryAction = "EVOLVE";
      this.actionTimer = 3.5;
      this.speechBubble = `✨ EVOLVED TO ${STAGE_CONFIGS[this.stage].name}! 🐻🌈`;
      this.speechTimer = 3.5;
      this.saveState();
    }
  }

  // ---------------------------------------------------------------------------
  // Step & Audio Frame Update
  // ---------------------------------------------------------------------------

  private static clamp01(value: number): number {
    return Math.max(0, Math.min(1, value));
  }

  /** Asymmetric envelope follower: fast attack, musical release. */
  private smoothEnvelope(
    current: number,
    target: number,
    dt: number,
    attackSec: number,
    releaseSec: number
  ): number {
    const tau = target > current ? attackSec : releaseSec;
    return current + (target - current) * (1 - Math.exp(-dt / Math.max(0.001, tau)));
  }

  /** Zero-crossing brightness (0..1) of a live stem waveform. */
  private waveformBrightness(waveform?: Uint8Array | null): number {
    if (!waveform || waveform.length < 8) return 0;
    let crossings = 0;
    let prevSign = waveform[0] >= 128 ? 1 : -1;
    for (let i = 2; i < waveform.length; i += 2) {
      const sample = waveform[i] - 128;
      const sign = sample >= 0 ? 1 : -1;
      if (sign !== prevSign) crossings++;
      prevSign = sign;
    }
    const zcr = crossings / (waveform.length * 0.5);
    return TamagotchiAvatar.clamp01(zcr * 3.2);
  }

  /** Smooth every stem plus the true post-mix master into visual envelopes. */
  private updateAudioEnvelopes(
    dt: number,
    drumLevel: number,
    bassLevel: number,
    vocalsLevel: number,
    otherLevel: number,
    mixLevel: number,
    otherWaveform?: Uint8Array | null
  ): void {
    const prevDrums = this.stemEnv.drums;
    const prevBass = this.stemEnv.bass;
    const prevVocals = this.stemEnv.vocals;
    const prevOther = this.stemEnv.other;
    const prevMix = this.mixEnv;

    // The analyser levels are already RMS*2.5; lift the quieter stems with a
    // soft knee so a 0.1 RMS vocal still opens the mouth while peaks keep
    // their headroom instead of flat-topping at 1.0.
    const lift = (level: number, gain: number): number =>
      1 - Math.exp(-Math.max(0, level) * gain);
    const drumTarget = TamagotchiAvatar.clamp01(lift(drumLevel, 2.4));
    const bassTarget = TamagotchiAvatar.clamp01(lift(bassLevel, 1.8));
    const vocalTarget = TamagotchiAvatar.clamp01(lift(vocalsLevel, 3.2));
    const otherTarget = TamagotchiAvatar.clamp01(lift(otherLevel, 1.9));
    const mixTarget = TamagotchiAvatar.clamp01(lift(mixLevel, 1.7));

    this.stemEnv.drums = this.smoothEnvelope(prevDrums, drumTarget, dt, 0.022, 0.16);
    this.stemEnv.bass = this.smoothEnvelope(prevBass, bassTarget, dt, 0.030, 0.22);
    this.stemEnv.vocals = this.smoothEnvelope(prevVocals, vocalTarget, dt, 0.040, 0.26);
    this.stemEnv.other = this.smoothEnvelope(prevOther, otherTarget, dt, 0.080, 0.42);
    this.mixEnv = this.smoothEnvelope(prevMix, mixTarget, dt, 0.030, 0.24);

    // Attack transients: one-shot decaying pulses used for kicks, syllables,
    // bass plucks and pad swells. They make the bear hit, not just lean.
    const decay = (value: number, tau: number): number => value * Math.exp(-dt / tau);
    this.stemPunch.drums = Math.max(
      Math.max(0, this.stemEnv.drums - prevDrums) * 5.0,
      decay(this.stemPunch.drums, 0.12)
    );
    this.stemPunch.bass = Math.max(
      Math.max(0, this.stemEnv.bass - prevBass) * 4.0,
      decay(this.stemPunch.bass, 0.18)
    );
    this.stemPunch.vocals = Math.max(
      Math.max(0, this.stemEnv.vocals - prevVocals) * 4.5,
      decay(this.stemPunch.vocals, 0.16)
    );
    this.stemPunch.other = Math.max(
      Math.max(0, this.stemEnv.other - prevOther) * 3.0,
      decay(this.stemPunch.other, 0.30)
    );
    this.mixPunch = Math.max(
      Math.max(0, this.mixEnv - prevMix) * 4.0,
      decay(this.mixPunch, 0.18)
    );

    // 'other' timbre: pad/chord brightness shifts the heart from warm gold
    // toward cool magenta/violet without changing its candy identity.
    const otherBrightness = this.stemEnv.other > 0.035
      ? this.waveformBrightness(otherWaveform)
      : this.otherTone;
    this.otherTone = this.smoothEnvelope(this.otherTone, otherBrightness, dt, 0.18, 0.65);
  }

  public update(
    meta: GrooveboxMeta | null,
    audioNow: number,
    rmsLevel: number,
    stepIndex: number,
    bpm: number = 126,
    waveform?: Uint8Array | null,
    vocalsLevel?: number,
    drumWaveform?: Uint8Array | null,
    bassWaveform?: Uint8Array | null,
    drumLevel?: number,
    bassLevel?: number,
    otherWaveform?: Uint8Array | null,
    otherLevel?: number,
    masterWaveform?: Uint8Array | null,
    liveMouth?: LiveMouthState | null,
    freqBins?: Uint8Array | null
  ): void {
    this.liveMouth = liveMouth ?? null;
    const dt = this.lastRenderSec > 0 ? Math.max(0.001, audioNow - this.lastRenderSec) : 0.016;
    this.lastRenderSec = audioNow;

    this.updateAudioEnvelopes(
      dt,
      drumLevel ?? 0,
      bassLevel ?? 0,
      vocalsLevel ?? 0,
      otherLevel ?? 0,
      rmsLevel,
      otherWaveform
    );

    // Mouth articulation: track pitch/formants at 30 Hz from the vocal
    // waveform before either face renders (they share the result).
    this.vocalFrame += 1;
    if (this.vocalFrame % 2 === 0) {
      this.analyzeVocalAcoustics(waveform, dt);
    }

    // Real-time Audio and Vocal Reactivity for Electric Rave Glow
    if (this.asciiEl) {
      this.asciiEl.style.setProperty("--audio-lvl", this.mixEnv.toFixed(3));
      this.asciiEl.style.setProperty("--vocal-lvl", this.stemEnv.vocals.toFixed(3));
      this.asciiEl.style.setProperty("--drum-lvl", this.stemEnv.drums.toFixed(3));
      this.asciiEl.style.setProperty("--bass-lvl", this.stemEnv.bass.toFixed(3));
      this.asciiEl.style.setProperty("--other-lvl", this.stemEnv.other.toFixed(3));
      const screenEl = this.asciiEl.parentElement;
      if (screenEl) {
        screenEl.style.setProperty("--audio-lvl", this.mixEnv.toFixed(3));
        screenEl.style.setProperty("--vocal-lvl", this.stemEnv.vocals.toFixed(3));
        screenEl.style.setProperty("--drum-lvl", this.stemEnv.drums.toFixed(3));
        screenEl.style.setProperty("--bass-lvl", this.stemEnv.bass.toFixed(3));
        screenEl.style.setProperty("--other-lvl", this.stemEnv.other.toFixed(3));
      }
    }

    // Decay timers
    if (this.actionTimer > 0) {
      this.actionTimer -= dt;
      if (this.actionTimer <= 0) this.temporaryAction = null;
    }
    if (this.speechTimer > 0) {
      this.speechTimer -= dt;
      if (this.speechTimer <= 0) this.speechBubble = "";
    }

    // Blink every 3.5 - 5 seconds
    if (audioNow - this.lastBlinkSec > 4.0) {
      this.isBlinking = true;
      this.lastBlinkSec = audioNow;
      setTimeout(() => { this.isBlinking = false; }, 140);
    }

    // Step-triggered physical head bobbing & XP progression
    if (stepIndex !== this.lastStepIndex) {
      this.lastStepIndex = stepIndex;

      // Bob down on 4/4 beats: 0, 4, 8, 12
      const isBeat = (stepIndex % 4 === 0);
      this.isHeadBobbed = isBeat;

      // Emit floating notes when vocal is singing
      if (meta && meta.vocalSyllable && (stepIndex % 2 === 0)) {
        const char = MUSIC_NOTES[Math.floor(Math.random() * MUSIC_NOTES.length)];
        this.noteFloating.push({
          char,
          x: 140 + (Math.random() - 0.5) * 60,
          y: 125,
          life: 1.0,
        });
      }

      // Check bar progression
      const curBar = meta?.phraseBar ?? 0;
      if (curBar !== this.lastBarNum && stepIndex === 0) {
        this.lastBarNum = curBar;
        this.barsListened++;
        this.addXp(2); // +2 XP for every bar heard!
      }
    }

    // Update floating notes
    for (let i = this.noteFloating.length - 1; i >= 0; i--) {
      const n = this.noteFloating[i];
      n.life -= dt * 1.1;
      n.y -= dt * 26.0;
      if (n.life <= 0) this.noteFloating.splice(i, 1);
    }

    // Mood mapping from Section & Audio Dynamics
    const sec = meta?.section ?? "GROOVE_A";
    const isDropGap = (sec === "BUILDUP" && meta?.phraseBar === 8 && (stepIndex % 16) === 15);

    if (this.temporaryAction === "EVOLVE") {
      this.mood = "EVOLVING!";
    } else if (this.temporaryAction === "PET") {
      this.mood = "LOVED ♥";
    } else if (this.temporaryAction === "EAT") {
      this.mood = "NOMMING 🍕";
    } else if (this.temporaryAction === "HYPE") {
      this.mood = "SUPERCHARGED ⚡";
    } else if (isDropGap) {
      this.mood = "HOLDING BREATH!";
    } else if (sec === "DROP") {
      this.mood = "RAVE ECSTASY";
      this.vibe = Math.min(100, this.vibe + dt * 4);
    } else if (sec === "BUILDUP") {
      this.mood = "TENSION RISING!";
      this.vibe = Math.min(100, this.vibe + dt * 2);
    } else if (sec === "BREAKDOWN") {
      this.mood = "SOULFUL VIBE";
    } else if (sec === "INTRO") {
      this.mood = "WAKING UP";
    } else {
      this.mood = "GROOVING";
    }

    // Set speech bubble from active vocal lyrics if no temporary speech
    if (this.speechTimer <= 0 && meta?.vocalLyric) {
      this.speechBubble = `"${meta.vocalLyric}"`;
    }

    this.renderCanvas(
      meta,
      isDropGap,
      waveform,
      dt,
      bpm,
      drumWaveform,
      bassWaveform,
      otherWaveform,
      masterWaveform,
      drumLevel ?? 0,
      bassLevel ?? 0,
      vocalsLevel ?? 0,
      rmsLevel
    );
    if (this.view3dActive && this.bear3dView) {
      const mouthAperture = Math.max(0, Math.min(1, this.smoothedVocalAmp * 2.2));
      const vowel = this.vocalVowelFormant || meta?.vocalVowel || "AH";
      this.bear3dView.renderFrame({
        stage: this.stage,
        bpm,
        drums: this.stemEnv.drums,
        bass: this.stemEnv.bass,
        other: this.stemEnv.other,
        vocals: this.stemEnv.vocals,
        mix: this.mixEnv,
        drumPunch: this.stemPunch.drums,
        bassPunch: this.stemPunch.bass,
        otherPunch: this.stemPunch.other,
        mixPunch: this.mixPunch,
        isBlinking: this.isBlinking,
        isDrop: sec === "DROP",
        action: this.temporaryAction,
        mouth: this.liveMouth,
        fallbackAperture: mouthAperture,
        fallbackVowel: vowel,
        freqBins,
        visual: this.visualConfig,
      });
    }
    this.renderASCII(meta, isDropGap, waveform, dt, bpm, vocalsLevel, drumLevel ?? 0, bassLevel ?? 0, rmsLevel);
    this.updateDashboard();
  }

  /** Composite the transparent dancing/singing Wub Creature canvas cleanly onto the Lightshow overlay canvas. */
  public drawOverlayInto(targetCanvas: HTMLCanvasElement): void {
    const src = this.canvasEl;
    if (!src) return;
    const rect = targetCanvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const dw = Math.max(160, Math.round((rect.width || targetCanvas.width || 640) * dpr));
    const dh = Math.max(120, Math.round((rect.height || targetCanvas.height || 400) * dpr));
    if (targetCanvas.width !== dw || targetCanvas.height !== dh) {
      targetCanvas.width = dw;
      targetCanvas.height = dh;
    }
    const ctx = targetCanvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, dw, dh);
    const sw = src.width || 280;
    const sh = src.height || 200;
    const scale = Math.min(dw / sw, dh / sh) * (1.06 + 0.06 * this.mixEnv + 0.04 * this.mixPunch);
    const drawW = sw * scale;
    const drawH = sh * scale;
    const dx = (dw - drawW) * 0.5;
    const dy = (dh - drawH) * 0.5;
    ctx.save();
    ctx.globalAlpha = 0.92;
    ctx.drawImage(src, dx, dy, drawW, drawH);
    ctx.restore();
  }

  // ---------------------------------------------------------------------------
  // Vocal acoustics: pitch (F0) + formants (F1/F2) for mouth articulation
  // ---------------------------------------------------------------------------
  /** Small iterative radix-2 FFT (magnitude only, in place). */
  private static fftMagnitude(input: Float32Array): Float32Array {
    const n = input.length;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    re.set(input);
    // Bit-reversal permutation.
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
        tmp = im[i]; im[i] = im[j]; im[j] = tmp;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wLenRe = Math.cos(ang);
      const wLenIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let wRe = 1;
        let wIm = 0;
        const half = len >> 1;
        for (let k = 0; k < half; k++) {
          const uRe = re[i + k];
          const uIm = im[i + k];
          const vRe = re[i + k + half] * wRe - im[i + k + half] * wIm;
          const vIm = re[i + k + half] * wIm + im[i + k + half] * wRe;
          re[i + k] = uRe + vRe;
          im[i + k] = uIm + vIm;
          re[i + k + half] = uRe - vRe;
          im[i + k + half] = uIm - vIm;
          const nextWRe = wRe * wLenRe - wIm * wLenIm;
          wIm = wRe * wLenIm + wIm * wLenRe;
          wRe = nextWRe;
        }
      }
    }
    const mag = new Float32Array(n >> 1);
    for (let i = 0; i < mag.length; i++) {
      mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    return mag;
  }

  private static nextPow2(value: number): number {
    let n = 1024;
    while (n < value) n <<= 1;
    return n;
  }

  /**
   * Track vocal pitch, formants and vowel from the vocal time-domain
   * waveform (Uint8-ish 0..255, centred 128) at 44100 Hz.  Runs every other
   * frame (30 Hz) — plenty for articulation.  All state decays safely when
   * the voice is silent, so the mouth falls back to amplitude + vowel table.
   */
  private analyzeVocalAcoustics(
    waveform: Uint8Array | null | undefined,
    dt: number
  ): void {
    const sr = 44100;
    if (!waveform || waveform.length < 512) {
      this.vocalF0Confidence = Math.max(0, this.vocalF0Confidence - dt * 2);
      return;
    }
    // Overall level: unvoiced/silent frames hold formants, drop confidence.
    let energy = 0;
    for (let i = 0; i < waveform.length; i++) {
      const v = (waveform[i] - 128) / 128;
      energy += v * v;
    }
    const rms = Math.sqrt(energy / waveform.length);
    // ---- pitch via normalised autocorrelation (decimated x2) ----
    const dec = 2;
    const srDec = sr / dec;
    const m = Math.floor(waveform.length / dec);
    const centered = new Float32Array(m);
    let mean = 0;
    for (let i = 0; i < m; i++) mean += (waveform[i * dec] - 128) / 128;
    mean /= Math.max(1, m);
    let power = 0;
    for (let i = 0; i < m; i++) {
      const v = (waveform[i * dec] - 128) / 128 - mean;
      centered[i] = v;
      power += v * v;
    }
    let bestLag = 0;
    let bestVal = 0;
    if (power > 1e-6) {
      const minLag = Math.max(8, Math.floor(srDec / 800));
      const maxLag = Math.min(m - 1, Math.ceil(srDec / 80));
      for (let lag = minLag; lag <= maxLag; lag++) {
        const n = m - lag;
        let sum = 0;
        for (let i = 0; i < n; i++) sum += centered[i] * centered[i + lag];
        // Normalised by total power; lag-decay correction keeps long lags honest.
        const corr = sum / Math.max(1e-9, power);
        const score = corr / Math.max(0.01, 1 - lag / m);
        if (score > bestVal) {
          bestVal = score;
          bestLag = lag;
        }
      }
    }
    const confidence = Math.max(0, Math.min(1, bestVal));
    this.vocalF0Confidence = confidence;
    if (bestLag > 0 && confidence > 0.35 && rms > 0.015) {
      const f0 = srDec / bestLag;
      const prev = this.vocalF0Hz;
      this.vocalF0Hz = prev > 0 ? prev * 0.55 + f0 * 0.45 : f0;
      this.vocalF0MedianHz = this.vocalF0MedianHz * 0.995 + this.vocalF0Hz * 0.005;
      const semis = Math.abs(12 * Math.log2(Math.max(20, this.vocalF0Hz) / Math.max(20, this.vocalF0MedianHz)));
      this.vocalVibratoSemis = this.vocalVibratoSemis * 0.85 + semis * 0.15;
    } else {
      this.vocalF0Hz *= Math.max(0, 1 - dt * 3);
      this.vocalVibratoSemis *= Math.max(0, 1 - dt * 4);
    }
    // ---- formants via FFT magnitude (F1/F2 bands) ----
    // Only update on voiced frames so consonant noise doesn't drag vowels.
    if (confidence > 0.25 || rms > 0.06) {
      const n = TamagotchiAvatar.nextPow2(Math.min(4096, waveform.length));
      const re = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const idx = Math.min(waveform.length - 1, Math.floor((i / n) * waveform.length));
        const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1));
        re[i] = ((waveform[idx] - 128) / 128 - mean) * hann;
      }
      const mag = TamagotchiAvatar.fftMagnitude(re);
      const binHz = sr / n;
      const bandMax = (loHz: number, hiHz: number): number => {
        const lo = Math.max(1, Math.floor(loHz / binHz));
        const hi = Math.min(mag.length - 1, Math.ceil(hiHz / binHz));
        let peak = -Infinity;
        let peakBin = lo;
        for (let b = lo; b <= hi; b++) {
          // 3-bin smoothing rejects single-bin spikes.
          const sm = (mag[Math.max(0, b - 1)] + 2 * mag[b] + mag[Math.min(mag.length - 1, b + 1)]) / 4;
          if (sm > peak) {
            peak = sm;
            peakBin = b;
          }
        }
        return peakBin * binHz;
      };
      const f1 = bandMax(250, 1100);
      const f2 = bandMax(800, 2800);
      const k = Math.min(1, dt * 12);
      this.vocalF1 = this.vocalF1 > 0 ? this.vocalF1 * (1 - k) + f1 * k : f1;
      this.vocalF2 = this.vocalF2 > 0 ? this.vocalF2 * (1 - k) + f2 * k : f2;
      // Vowel = nearest reference in normalised (F1, F2) space.
      const refs: Array<[string, number, number]> = [
        ["AH", 700, 1200], ["EE", 300, 2300], ["OH", 600, 900],
        ["OO", 300, 870], ["AY", 650, 1700],
      ];
      let best = "";
      let bestD = Infinity;
      for (const [name, r1, r2] of refs) {
        const d = ((this.vocalF1 - r1) / 350) ** 2 + ((this.vocalF2 - r2) / 900) ** 2;
        if (d < bestD) {
          bestD = d;
          best = name;
        }
      }
      this.vocalVowelFormant = best;
    }
  }

  // ---------------------------------------------------------------------------
  // Live Waveform Animated Singing Mouth
  // ---------------------------------------------------------------------------

  private extractWaveformMouth(
    waveform: Uint8Array | null | undefined,
    isSinging: boolean,
    vowel: string,
    isDropGap: boolean,
    dt: number,
    bpm: number,
    vocalsLevel?: number
  ): string {
    if (isDropGap) return "  O  ";

    // 1. Physical Waveform Acoustic Feature Extraction
    let waveRms = 0;
    let peakDev = 0;
    let zeroCrossings = 0;

    if (waveform && waveform.length > 0) {
      const len = waveform.length;
      let sumSquares = 0;
      let prevSign = 0;
      for (let i = 0; i < len; i++) {
        const dev = (waveform[i] - 128) / 128; // Normalized -1.0 .. +1.0
        sumSquares += dev * dev;
        const absDev = Math.abs(dev);
        if (absDev > peakDev) peakDev = absDev;
        const sign = dev >= 0 ? 1 : -1;
        if (i > 0 && sign !== prevSign) zeroCrossings++;
        prevSign = sign;
      }
      waveRms = Math.sqrt(sumSquares / len);
    }

    const zcr = waveform && waveform.length > 0 ? zeroCrossings / waveform.length : 0.12;

    // 2. Instantaneous Vocal Audio Envelope Follower
    // Voice audio drives the primary mouth aperture (instant attack, musical vocal decay)
    const instantAmp = (waveform && waveform.length > 0)
      ? Math.max(waveRms * 2.6, peakDev * 0.95)
      : Math.max(vocalsLevel ?? 0, isSinging ? 0.40 : 0);

    const attackRate = 28.0;
    const releaseRate = 8.5;
    const rate = instantAmp > this.smoothedVocalAmp ? attackRate : releaseRate;
    this.smoothedVocalAmp += (instantAmp - this.smoothedVocalAmp) * Math.min(1.0, dt * rate);

    // Mouth animation strictly active for the voice instrument only
    const voiceActive = isSinging || this.smoothedVocalAmp >= 0.035 || (vocalsLevel ?? 0) >= 0.015;
    if (!voiceActive) {
      this.smoothedVocalAmp = Math.max(0, this.smoothedVocalAmp - dt * 2.5);
      return "  ‿  ";
    }

    // 3. Mathematically Derived Singing Loop
    // Vibrato cycle locked to musical tempo (~3.2 - 4.2 Hz)
    const vibratoHz = (bpm / 60) * 1.6;
    // Acoustic texture from waveform zero-crossing rate modulates continuous loop speed
    const loopAdvance = dt * vibratoHz * (1.0 + Math.min(0.5, zcr * 2.0)) * Math.PI * 2;
    this.singingLoopPhase = (this.singingLoopPhase + loopAdvance) % (Math.PI * 2);

    // Mathematical aperture modulated by real-time vocal audio envelope & singing loop vibrato
    const vibratoOsc = Math.sin(this.singingLoopPhase);
    const dynAperture = Math.max(0, Math.min(1, this.smoothedVocalAmp * (1.2 + 0.32 * vibratoOsc) * 1.8));
    const tier = Math.min(4, Math.floor(dynAperture * 5));

    // Determine Vowel Geometry: from slice metadata or inferred from waveform formant energy (ZCR)
    let activeVowel = (vowel || "").toUpperCase();
    if (!activeVowel || !["AH", "EE", "OH", "OO", "AY"].includes(activeVowel)) {
      if (zcr > 0.22) activeVowel = "EE";       // High formant brightness -> wide smile
      else if (zcr < 0.08) activeVowel = "OO";  // Deep fundamental -> tight rounded vowel
      else if (zcr < 0.15) activeVowel = "OH";  // Resonant mid vowel -> round trumpet mouth
      else activeVowel = "AH";                  // Standard open vocalization
    }

    // 4. Mathematical Mouth Shape Tables (Primary vs Resonant Vibrato Phase)
    // Every single glyph string is strictly 5 monospace characters
    const MOUTH_SHAPES: Record<string, { primary: string[]; resonant: string[] }> = {
      AH: {
        primary:  ["  -  ", " ⌢─⌣ ", " ╭─╮ ", " ╭▽╮ ", " ╭█╮ "],
        resonant: ["  -  ", " ⌢─⌣ ", " ╭◯╮ ", " ╭◇╮ ", " ❨█❩ "],
      },
      EE: {
        primary:  [" ─‿─ ", " ═‿═ ", " ═─═ ", " ◖═◗ ", " [≡] "],
        resonant: [" ─‿─ ", " ═‿═ ", " ◖─◗ ", " ◖■◗ ", " [█] "],
      },
      OH: {
        primary:  ["  ·  ", "  o  ", "  O  ", "  ◯  ", " ❨◯❩ "],
        resonant: ["  ·  ", "  o  ", "  0  ", "  ◎  ", " ❨◎❩ "],
      },
      OO: {
        primary:  ["  ·  ", "  ∘  ", "  ᴥ  ", "  o  ", " ◖·◗ "],
        resonant: ["  ·  ", "  ∘  ", "  o  ", "  ●  ", " ◖●◗ "],
      },
      AY: {
        primary:  ["  ‿  ", " ─⌣─ ", " ╭‿╮ ", " ❨⌢❩ ", " ❨▲❩ "],
        resonant: ["  ‿  ", " ─‿─ ", " ╭─╮ ", " ❨═❩ ", " ❨▼❩ "],
      },
      DEFAULT: {
        primary:  ["  ‿  ", " ⌢─⌣ ", " ╭─╮ ", " ╭◯╮ ", " ❨█❩ "],
        resonant: ["  ‿  ", " ─‿─ ", " ╭◇╮ ", " ╭◆╮ ", " ❨◆❩ "],
      },
    };

    const family = MOUTH_SHAPES[activeVowel] ?? MOUTH_SHAPES.DEFAULT;
    // Alternate between primary and resonant singing loop forms at peak of vibrato phase
    const isResonantPhase = Math.cos(this.singingLoopPhase) > 0.15;
    return (isResonantPhase && tier >= 2) ? family.resonant[tier] : family.primary[tier];
  }

  // ---------------------------------------------------------------------------
  // ASCII Rendering Engine
  // ---------------------------------------------------------------------------
  private renderASCII(
    meta: GrooveboxMeta | null,
    isDropGap: boolean,
    waveform?: Uint8Array | null,
    dt: number = 0.016,
    bpm: number = 126,
    vocalsLevel?: number,
    drumLevel: number = 0,
    bassLevel: number = 0,
    rmsLevel: number = 0
  ): void {
    if (!this.asciiEl) return;

    const sec = meta?.section ?? "GROOVE_A";
    const vowel = this.vocalVowelFormant || meta?.vocalVowel || "";
    const isSinging = Boolean(meta?.vocalSyllable);
    const bob = this.isHeadBobbed;

    // Toggle glowing psychedelic singing shimmer
    if (isSinging) {
      this.asciiEl.classList.add("singing");
    } else {
      this.asciiEl.classList.remove("singing");
    }

    // Determine Big Expressive Anime Eyes (v200: matched pair, both eyes share
    // the same glyph — mood states override both together).
    let eyes = "◕   ◕";
    if (this.isBlinking) {
      eyes = "◡   ◡";
    } else if (isDropGap) {
      eyes = "⊙   ⊙";
    } else if (this.temporaryAction === "PET") {
      eyes = "♥   ♥";
    } else if (sec === "DROP" || this.temporaryAction === "HYPE") {
      eyes = "★   ★";
    } else if (sec === "BUILDUP") {
      eyes = "✧   ✧";
    } else if (sec === "BREAKDOWN") {
      eyes = "˘   ˘";
    } else if (this.stage >= 3) {
      eyes = "◕   ◕";
    }

    // Audio reactive dynamic "X" shaped nose in the middle of the face
    let nose = "  ><  ";
    if (this.stemEnv.drums > 0.45 || this.mixEnv > 0.40) {
      nose = " ⚔X⚔ ";
    } else if (isSinging) {
      nose = " ✧X✧ ";
    } else if (sec === "DROP") {
      nose = " ★X★ ";
    } else if (this.isBlinking) {
      nose = "  x  ";
    } else {
      nose = (Math.sin(performance.now() / 280) > 0) ? "  ><  " : "  ╳  ";
    }

    // Determine Live Animated Waveform Singing Mouth (Moved Down)
    const mouth = this.extractWaveformMouth(waveform, isSinging, vowel, isDropGap, dt, bpm, vocalsLevel);

    // Render Round Orb Stages with big anime eyes, dynamic "X" nose, and pregenerated sound shape singing mouth
    let faceLines: string[] = [];
    const noteChar = this.noteFloating.length > 0 ? this.noteFloating[0].char : "✨";

    if (this.stage === 1) {
      // Stage 1: Gummi Cub
      faceLines = bob ? [
        `      <span class="rave-orb">.╭ʕ •ᴥ• ʔ╮.</span>       `,
        `     <span class="rave-orb">(</span>   <span class="rave-eyes">${eyes}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">${noteChar}</span>  `,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~✦</span> `,
        `     <span class="rave-orb">(</span>  <span class="rave-mouth">${mouth}</span>  <span class="rave-orb">)</span> <span class="rave-spark">~♡</span> `,
        `      <span class="rave-orb">·╰─(づ｡◕‿‿◕｡)づ─╯·</span>  `,
      ] : [
        `      <span class="rave-orb">.╭ʕ •ᴥ• ʔ╮.</span>       `,
        `     <span class="rave-orb">(</span>   <span class="rave-eyes">${eyes}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">${noteChar}</span>  `,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~~*</span> `,
        `     <span class="rave-orb">(</span>  <span class="rave-mouth">${mouth}</span>  <span class="rave-orb">)</span> <span class="rave-spark">~~*</span> `,
        `      <span class="rave-orb">·╰─(づ｡◕‿‿◕｡)づ─╯·</span>  `,
      ];
    } else if (this.stage === 2) {
      // Stage 2: Neon Gummi
      faceLines = bob ? [
        `     <span class="rave-wings">.·´</span><span class="rave-orb">╭ʕ •ᴥ• ʔ╮</span><span class="rave-wings">\`·.</span>     `,
        `   <span class="rave-wings">[o(</span>  <span class="rave-eyes">${eyes}</span>  <span class="rave-wings">)o]</span>  <span class="rave-spark">${noteChar}</span> `,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~✦</span> `,
        `     <span class="rave-orb">(</span>  <span class="rave-mouth">${mouth}</span>  <span class="rave-orb">)</span> <span class="rave-spark">~🌈</span> `,
        `     <span class="rave-wings">\`·.</span><span class="rave-orb">╰─| GUMMI |─╯</span><span class="rave-wings">.·´</span>   `,
      ] : [
        `     <span class="rave-wings">.·´</span><span class="rave-orb">╭ʕ •ᴥ• ʔ╮</span><span class="rave-wings">\`·.</span>     `,
        `   <span class="rave-wings">[o(</span>  <span class="rave-eyes">${eyes}</span>  <span class="rave-wings">)o]</span>  <span class="rave-spark">${noteChar}</span> `,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~~*</span> `,
        `     <span class="rave-orb">(</span>  <span class="rave-mouth">${mouth}</span>  <span class="rave-orb">)</span> <span class="rave-spark">~~✨</span> `,
        `     <span class="rave-wings">\`·.</span><span class="rave-orb">╰─| GUMMI |─╯</span><span class="rave-wings">.·´</span>   `,
      ];
    } else if (this.stage === 3) {
      // Stage 3: Cyber Gummi
      faceLines = bob ? [
        `     <span class="rave-wings">~/</span><span class="rave-orb">╭ʕ ⚡ᴥ⚡ ʔ╮</span><span class="rave-wings">\\~</span>     `,
        `   <span class="rave-wings">~/(\\</span> (<span class="rave-eyes">${eyes}</span>) <span class="rave-wings">/\\)~</span> <span class="rave-spark">${noteChar}</span>`,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~✦</span> `,
        `    <span class="rave-body">&lt;[~</span> <span class="rave-mouth">${mouth}</span> <span class="rave-body">~]&gt;</span>    `,
        `     <span class="rave-orb">·╰─| GUMMI |─╯·</span>    `,
      ] : [
        `     <span class="rave-wings">~/</span><span class="rave-orb">╭ʕ ⚡ᴥ⚡ ʔ╮</span><span class="rave-wings">\\~</span>     `,
        `   <span class="rave-wings">~/\\(</span> (<span class="rave-eyes">${eyes}</span>) <span class="rave-wings">/\\)~</span> <span class="rave-spark">${noteChar}</span>`,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~~*</span> `,
        `    <span class="rave-body">\\[~</span> <span class="rave-mouth">${mouth}</span> <span class="rave-body">~]/</span>    `,
        `     <span class="rave-orb">·╰─| GUMMI |─╯·</span>    `,
      ];
    } else {
      // Stage 4: Cosmic Gummi
      faceLines = bob ? [
        `    <span class="rave-spark">✧</span><span class="rave-wings">~/</span><span class="rave-orb">╭ʕ ✦ᴥ✦ ʔ╮</span><span class="rave-wings">\\~</span><span class="rave-spark">✧</span>    `,
        `  <span class="rave-wings">~/{\\(</span> (<span class="rave-eyes">${eyes}</span>) <span class="rave-wings">/\\}~</span> <span class="rave-spark">${noteChar}</span>`,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~✦</span> `,
        `   <span class="rave-body">&lt;[~</span> ✧<span class="rave-mouth">${mouth}</span>✧ <span class="rave-body">~]&gt;</span>    `,
        `    <span class="rave-orb">·╰─| 505 |─╯·</span>       `,
      ] : [
        `    <span class="rave-spark">✧</span><span class="rave-wings">~/</span><span class="rave-orb">╭ʕ ✦ᴥ✦ ʔ╮</span><span class="rave-wings">\\~</span><span class="rave-spark">✧</span>    `,
        `  <span class="rave-wings">~/{\\(</span> (<span class="rave-eyes">${eyes}</span>) <span class="rave-wings">/\\}~</span> <span class="rave-spark">${noteChar}</span>`,
        `     <span class="rave-orb">(</span>   <span class="rave-nose">${nose}</span>   <span class="rave-orb">)</span>  <span class="rave-spark">~~*</span> `,
        `   <span class="rave-body">\\[~</span> ✧<span class="rave-mouth">${mouth}</span>✧ <span class="rave-body">~]/</span>    `,
        `    <span class="rave-orb">·╰─| 505 |─╯·</span>       `,
      ];
    }

    // Build borderless psychedelic LCD output
    const songNote = meta?.vocalNote
      ? `<span class="rave-note">♪ SING: ${meta.vocalNote} [${vowel || "--"}] ♪</span>`
      : `<span class="rave-note">♫ BEAT: ${sec} ♫</span>`;

    const screenHtml = [
      ...faceLines,
      "",
      songNote,
    ].join("\n");

    this.asciiEl.innerHTML = screenHtml;
  }

  private updateDashboard(): void {
    if (this.stageBadgeEl) {
      this.stageBadgeEl.textContent = `LVL ${this.stage} • ${STAGE_CONFIGS[this.stage].name}`;
    }
    if (this.vibeEl) {
      this.vibeEl.textContent = `${Math.round(this.vibe)}%`;
    }
    if (this.moodEl) {
      this.moodEl.textContent = this.mood;
    }
    if (this.xpEl) {
      this.xpEl.textContent = String(this.xp);
    }
  }

  // ---------------------------------------------------------------------------
  // Audio Reactive Single Line Loop & Live Waveform Mouth Canvas Engine
  // ---------------------------------------------------------------------------

  private renderCanvas(
    meta: GrooveboxMeta | null,
    isDropGap: boolean,
    waveform?: Uint8Array | null,
    dt: number = 0.016,
    bpm: number = 126,
    drumWaveform?: Uint8Array | null,
    bassWaveform?: Uint8Array | null,
    otherWaveform?: Uint8Array | null,
    masterWaveform?: Uint8Array | null,
    drumLevel: number = 0,
    bassLevel: number = 0,
    vocalsLevel: number = 0,
    rmsLevel: number = 0
  ): void {
    if (!this.canvasEl) return;
    if (!this.ctx2d) {
      this.ctx2d = this.canvasEl.getContext("2d");
      if (!this.ctx2d) return;
    }
    const ctx = this.ctx2d;
    const w = this.canvasEl.width;
    const h = this.canvasEl.height;

    ctx.clearRect(0, 0, w, h);

    const sec = meta?.section ?? "GROOVE_A";
    const vowel = this.vocalVowelFormant || meta?.vocalVowel || "";
    const isSinging = Boolean(meta?.vocalSyllable) || (vocalsLevel >= 0.008);
    const bob = this.isHeadBobbed;
    const t = performance.now() / 1000;
    const env = this.stemEnv;
    const punch = this.stemPunch;
    const mixEnv = this.mixEnv;
    const mixPunch = this.mixPunch;

    // The heart belongs to 'other': pads and chords make it breathe and swell.
    this.heartBeatPhase = (this.heartBeatPhase + dt * (
      0.40 + env.other * 1.55 + punch.other * 2.0 + mixPunch * 0.35
    )) % 1;

    // Center coordinates with beat bobbing for the floating kawaii heart head
    const cx = w * 0.5;
    const bobOffset = (bob ? 3.5 : 0.0) + 2.4 * mixPunch;
    const cy = h * 0.54 + bobOffset;

    // Stage-based color palette
    let mainColor = "#ff77e9";
    let glowColor = "#ff00a0";
    let accentColor = "#00ffff";
    if (this.stage === 2) {
      mainColor = "#ff00c8";
      glowColor = "#ff0077";
      accentColor = "#00ffcc";
    } else if (this.stage === 3) {
      mainColor = "#00f0ff";
      glowColor = "#0088ff";
      accentColor = "#ffe600";
    } else if (this.stage === 4) {
      mainColor = "#ffe600";
      glowColor = "#ffaa00";
      accentColor = "#ff00e6";
    }

    if (isSinging) {
      this.canvasEl.classList.add("singing");
    } else {
      this.canvasEl.classList.remove("singing");
    }

    // -------------------------------------------------------------------------
    // 1. KAWAII HEART DYNAMICS (SQUISH, STRETCH & LUB-DUB BEAT THUMP)
    // -------------------------------------------------------------------------
    const vis = this.visualConfig;
    const drumSquish = Math.min(1.4, env.drums * 1.15 * vis.danceGain);
    const squishY = 1.0 - 0.06 * drumSquish - 0.025 * mixPunch * vis.danceGain + (bob ? -0.02 * vis.danceGain : 0.015);
    const squishX = 1.0 + 0.06 * drumSquish + 0.025 * mixPunch * vis.danceGain - (bob ? -0.02 * vis.danceGain : 0.015);
    // Real heartbeat thump: pads/chords drive heartBeatPhase, and the whole
    // heart swells on the systole then relaxes, so it reads as a living heart.
    const beatThump = Math.pow(Math.max(0, Math.sin(this.heartBeatPhase * Math.PI * 2)), 6);
    const baseScale = (1.34 + 0.02 * this.stage) * (1.0 + 0.025 * mixEnv) * (1.0 + 0.045 * beatThump);
    const scaleX = baseScale * squishX;
    const scaleY = baseScale * squishY;

    // Rave halo behind the floating head: stage-coloured bloom that breathes with
    // the true full mix, so all four stems add up to the glow.
    ctx.save();
    const haloR = 112 * (1 + 0.11 * mixEnv + 0.05 * mixPunch);
    const halo = ctx.createRadialGradient(cx, cy, 8, cx, cy, haloR);
    halo.addColorStop(0.0, glowColor);
    halo.addColorStop(0.55, glowColor);
    halo.addColorStop(1.0, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = halo;
    ctx.globalAlpha = 0.07 + 0.24 * mixEnv + 0.08 * mixPunch;
    ctx.beginPath();
    ctx.arc(cx, cy, haloR, 0, Math.PI * 2);
    ctx.fill();

    // DMT / LSD sacred-geometry mandala ring pulsing behind the floating head
    if (vis.showHalo) {
      const mandalaR = (68 + 12 * mixEnv + 6 * mixPunch) * baseScale;
      const hexSides = 6;
      ctx.translate(cx, cy);
      ctx.rotate(t * 0.35);
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 1.1;
      ctx.globalAlpha = 0.14 + 0.26 * mixEnv;
      for (let layer = 0; layer < 2; layer++) {
        ctx.save();
        ctx.rotate(layer * (Math.PI / hexSides) - t * layer * 0.55);
        const rLayer = mandalaR * (layer === 0 ? 1.0 : 0.82);
        ctx.beginPath();
        for (let k = 0; k <= hexSides; k++) {
          const a = (k / hexSides) * Math.PI * 2;
          const hx = Math.cos(a) * rLayer;
          const hy = Math.sin(a) * rLayer;
          if (k === 0) ctx.moveTo(hx, hy);
          else ctx.lineTo(hx, hy);
        }
        ctx.stroke();
        ctx.restore();
      }
    }
    ctx.restore();


    // -------------------------------------------------------------------------
    // 2. THE AUDIO REACTIVE KAWAII HEART LOOP (HEART OUTLINE, NO EARS)
    // -------------------------------------------------------------------------
    const N = GUMMI_HEART_LOOP_POINTS.length;
    const displacedPoints: { x: number; y: number }[] = new Array(N);

    for (let i = 0; i < N; i++) {
      const pt = GUMMI_HEART_LOOP_POINTS[i];

      // Full-mix body shaping: a short smoothed window of the master waveform
      // literally pushes the silhouette in and out with the summed four stems.
      let mixSample = 0;
      if (masterWaveform && masterWaveform.length > 8) {
        const center = Math.floor(pt.u * (masterWaveform.length - 1));
        let sum = 0;
        let count = 0;
        for (let k = -5; k <= 5; k++) {
          const idx = Math.max(0, Math.min(masterWaveform.length - 1, center + k));
          sum += (masterWaveform[idx] - 128) / 128;
          count++;
        }
        mixSample = sum / count;
      }

      const waveRipple = (
        Math.sin(6 * Math.PI * 2 * pt.u - t * 4.2) * 2.8 +
        Math.cos(9 * Math.PI * 2 * pt.u + t * 6.5) * 1.8 +
        Math.sin(3 * Math.PI * 2 * pt.u + t * 2.2) * (env.drums * 4.2) +
        Math.cos(12 * Math.PI * 2 * pt.u - t * 7.5) * (env.vocals * 3.0) +
        Math.sin(2 * Math.PI * 2 * pt.u + t * 3.0) * (env.bass * 3.8) +
        Math.cos(5 * Math.PI * 2 * pt.u + t * 1.7) * (env.other * 2.4) +
        Math.sin(8 * Math.PI * 2 * pt.u + t * 12.0) * (punch.drums * 2.4) +
        mixSample * (2.0 + 5.5 * mixEnv)
      ) * Math.min(1.25, 0.24 + mixEnv * 0.85 + mixPunch * 0.35);

      const px = cx + (pt.x * scaleX) + pt.nx * waveRipple;
      const py = cy + (pt.y * scaleY) + pt.ny * waveRipple;
      displacedPoints[i] = { x: px, y: py };
    }

    // 2A. Juicy Translucent Candy Gelatin Fill
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const p = displacedPoints[i];
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();

    const jellyGrad = ctx.createRadialGradient(
      cx, cy, 8,
      cx, cy, 58 * baseScale
    );
    if (this.stage === 1) {
      // Pink Strawberry Gummi
      jellyGrad.addColorStop(0.0, "rgba(255, 120, 195, 0.58)");
      jellyGrad.addColorStop(0.55, "rgba(255, 60, 160, 0.40)");
      jellyGrad.addColorStop(1.0, "rgba(255, 20, 140, 0.20)");
    } else if (this.stage === 2) {
      // Neon Wildberry Gummi
      jellyGrad.addColorStop(0.0, "rgba(235, 70, 255, 0.60)");
      jellyGrad.addColorStop(0.55, "rgba(180, 40, 255, 0.42)");
      jellyGrad.addColorStop(1.0, "rgba(100, 20, 230, 0.22)");
    } else if (this.stage === 3) {
      // Cyber Mint / Raspberry Gummi
      jellyGrad.addColorStop(0.0, "rgba(0, 255, 245, 0.60)");
      jellyGrad.addColorStop(0.55, "rgba(0, 180, 255, 0.44)");
      jellyGrad.addColorStop(1.0, "rgba(0, 80, 220, 0.22)");
    } else {
      // Cosmic Golden Honey Gummi
      jellyGrad.addColorStop(0.0, "rgba(255, 235, 80, 0.62)");
      jellyGrad.addColorStop(0.55, "rgba(255, 160, 20, 0.46)");
      jellyGrad.addColorStop(1.0, "rgba(255, 80, 120, 0.24)");
    }
    ctx.fillStyle = jellyGrad;
    ctx.fill();
    ctx.restore();

    // 2B. Outer Translucent Celestial Jelly Glow Aura Ring
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const pt = GUMMI_HEART_LOOP_POINTS[i];
      const p = displacedPoints[i];
      const auraDist = 3.2;
      const ax = p.x + pt.nx * auraDist;
      const ay = p.y + pt.ny * auraDist;
      if (i === 0) ctx.moveTo(ax, ay);
      else ctx.lineTo(ax, ay);
    }
    ctx.closePath();
    // Warm gold hairline aura around the whole creature.
    ctx.strokeStyle = "#ffd76a";
    ctx.shadowBlur = 6;
    ctx.shadowColor = "#ffb300";
    ctx.lineWidth = 1.8;
    ctx.globalAlpha = 0.30 + 0.45 * mixEnv + 0.10 * mixPunch;
    ctx.stroke();

    // 2C. Main Radiant Kawaii Heart Perimeter (Neon Bloom Pass)
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const p = displacedPoints[i];
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    // Golden neon bloom (the beloved yellow outline).
    ctx.shadowBlur = 20 + 30 * mixEnv + 12 * mixPunch;
    ctx.shadowColor = "#ffae00";
    ctx.strokeStyle = "#ffcf2e";
    ctx.lineWidth = 4.6;
    ctx.globalAlpha = 0.52;
    ctx.stroke();

    // 2D. Inner Crisp Laser Vector Pass (White/Neon core)
    ctx.shadowBlur = 9;
    ctx.shadowColor = "#fff3b0";
    ctx.strokeStyle = "#ffe45c";
    ctx.lineWidth = 2.4;
    ctx.globalAlpha = 1.0;
    ctx.stroke();
    ctx.restore();

    // -------------------------------------------------------------------------
    // 3. KAWAII HEART LOBE GEL CUSHIONS (AUDIO REACTIVE VOCAL SHIMMER)
    // -------------------------------------------------------------------------
    ctx.save();
    const vocalPulse = 1 + 0.20 * Math.min(1.4, env.vocals * 1.5) + 0.10 * punch.vocals;
    const lobeW = 14.0 * baseScale * vocalPulse;
    const lobeH = 11.5 * baseScale * vocalPulse;
    const lobeSway = Math.sin(t * 6.2) * (env.vocals * 1.4 + punch.vocals * 0.8);

    // Soft candy-gel shine sitting inside each round heart lobe.
    for (const side of [-1, 1]) {
      const lx = cx + side * 26 * scaleX;
      const ly = cy - 40 * scaleY + lobeSway * side;
      const lobeGrad = ctx.createRadialGradient(
        lx - lobeW * 0.25, ly - lobeH * 0.35, 1,
        lx, ly, lobeW
      );
      lobeGrad.addColorStop(0.0, `rgba(255, 245, 252, ${0.34 + 0.32 * mixEnv})`);
      lobeGrad.addColorStop(0.55, `rgba(255, 150, 220, ${0.16 + 0.24 * env.vocals})`);
      lobeGrad.addColorStop(1.0, "rgba(255, 90, 190, 0)");
      ctx.fillStyle = lobeGrad;
      ctx.beginPath();
      ctx.ellipse(lx, ly, lobeW, lobeH, side * -0.28, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 4. KAWAII LOBE RIBBON BOW (HEART ACCESSORY, NO BODY)
    // -------------------------------------------------------------------------
    if (vis.showBikini) {
      ctx.save();
      const bikTrim = "#00f6ff";
      const bx = cx + 35 * scaleX;
      const by = cy - 51 * scaleY;
      const bw = 7.4 * baseScale;
      const bh = 4.5 * baseScale;
      ctx.translate(bx, by);
      ctx.rotate(0.42);
      ctx.fillStyle = "#ff2ea6";
      ctx.strokeStyle = bikTrim;
      ctx.lineWidth = 1.2;
      ctx.shadowBlur = 8 + 8 * mixEnv;
      ctx.shadowColor = "#ff0088";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-bw, -bh);
      ctx.quadraticCurveTo(-bw * 1.15, 0, -bw, bh);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(bw, -bh);
      ctx.quadraticCurveTo(bw * 1.15, 0, bw, bh);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, Math.max(1.8, bw * 0.28), 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.restore();
    }

    // 5. Floating rave motes orbiting the kawaii heart head (full-mix shimmer).
    ctx.save();
    for (let k = 0; k < 7; k++) {
      const a = t * (0.36 + k * 0.04) + k * (Math.PI * 2 / 7);
      const rr = 0.96 + 0.10 * Math.sin(t * 0.7 + k * 1.3);
      const mx = cx + Math.cos(a) * 64 * scaleX * rr;
      const my = cy + Math.sin(a) * 60 * scaleY * rr;
      const tw = 0.35 + 0.65 * Math.pow(Math.abs(Math.sin(t * 2.4 + k * 1.7)), 2);
      const rad = (0.9 + 1.6 * tw) * (0.75 + 0.55 * mixEnv + 0.25 * mixPunch);
      ctx.beginPath();
      ctx.arc(mx, my, rad, 0, Math.PI * 2);
      ctx.fillStyle = k % 3 === 0 ? "#ffffff" : (k % 3 === 1 ? accentColor : glowColor);
      ctx.globalAlpha = tw * (0.30 + 0.62 * mixEnv);
      ctx.shadowBlur = 10 + 8 * mixPunch;
      ctx.shadowColor = accentColor;
      ctx.fill();
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 6. DELICIOUS 3D GUMMI CANDY SPECULAR HIGHLIGHTS
    // -------------------------------------------------------------------------
    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
    ctx.lineWidth = 2.2;
    ctx.lineCap = "round";

    // Twin heart-lobe shine arcs (the classic kawaii glossy heart highlight)
    ctx.beginPath();
    ctx.arc(cx - 27 * scaleX, cy - 40 * scaleY, 9 * scaleX, 1.12 * Math.PI, 1.68 * Math.PI);
    ctx.stroke();
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.arc(cx + 25 * scaleX, cy - 40 * scaleY, 6.5 * scaleX, 1.18 * Math.PI, 1.72 * Math.PI);
    ctx.stroke();
    ctx.globalAlpha = 1.0;
    ctx.lineWidth = 2.2;

    // Lobe crown gloss dots
    ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
    ctx.beginPath();
    ctx.arc(cx - 36 * scaleX, cy - 47 * scaleY, 2.2, 0, Math.PI * 2);
    ctx.arc(cx + 36 * scaleX, cy - 47 * scaleY, 2.2, 0, Math.PI * 2);
    ctx.fill();

    // Chubby cheek gloss pills
    ctx.beginPath();
    ctx.arc(cx - 34 * scaleX, cy - 6 * scaleY, 2.0, 0, Math.PI * 2);
    ctx.arc(cx + 34 * scaleX, cy - 6 * scaleY, 2.0, 0, Math.PI * 2);
    ctx.fill();

    // Soft inner cleft shimmer at the top notch of the heart
    const cleftGrad = ctx.createRadialGradient(
      cx, cy - 33 * scaleY, 0.5,
      cx, cy - 33 * scaleY, 9 * baseScale
    );
    cleftGrad.addColorStop(0.0, `rgba(255, 255, 255, ${0.22 + 0.34 * mixEnv})`);
    cleftGrad.addColorStop(1.0, "rgba(255, 255, 255, 0)");
    ctx.fillStyle = cleftGrad;
    ctx.beginPath();
    ctx.ellipse(cx, cy - 33 * scaleY, 9 * baseScale, 4.5 * baseScale, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // -------------------------------------------------------------------------
    // 8. ROSY CHEEKS & KAWAII BLUSH SLASHES (///)
    // -------------------------------------------------------------------------
    const cheekSpacing = 28 * scaleX;
    const cheekY = cy + 2 * scaleY;
    const lCheekX = cx - cheekSpacing;
    const rCheekX = cx + cheekSpacing;

    ctx.save();
    ctx.fillStyle = `rgba(255, 90, 180, ${0.34 + 0.38 * mixEnv + 0.10 * mixPunch})`;
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#ff0077";
    ctx.beginPath();
    ctx.ellipse(lCheekX, cheekY, 8.5 * baseScale, 5.5 * baseScale, 0, 0, Math.PI * 2);
    ctx.ellipse(rCheekX, cheekY, 8.5 * baseScale, 5.5 * baseScale, 0, 0, Math.PI * 2);
    ctx.fill();

    // Kawaii blush slash marks (///)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 1.4;
    ctx.lineCap = "round";
    for (let b = -1; b <= 1; b++) {
      // Left cheek slashes
      ctx.beginPath();
      ctx.moveTo(lCheekX + b * 3.5 - 1.5, cheekY + 2.5);
      ctx.lineTo(lCheekX + b * 3.5 + 1.5, cheekY - 2.5);
      ctx.stroke();

      // Right cheek slashes
      ctx.beginPath();
      ctx.moveTo(rCheekX + b * 3.5 - 1.5, cheekY + 2.5);
      ctx.lineTo(rCheekX + b * 3.5 + 1.5, cheekY - 2.5);
      ctx.stroke();
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 9. BIG EXPRESSIVE ANIME WAVEFORM EYES
    // -------------------------------------------------------------------------
    const eyeSpacing = 18 * scaleX;
    const eyeY = cy - 11 * scaleY;
    const leftEyeX = cx - eyeSpacing;
    const rightEyeX = cx + eyeSpacing;

    ctx.save();
    ctx.shadowBlur = 10;
    ctx.shadowColor = accentColor;
    ctx.strokeStyle = accentColor;
    ctx.fillStyle = accentColor;
    ctx.lineWidth = 2.2;

    if (this.isBlinking) {
      this.drawAnimeBlinkEye(ctx, leftEyeX, eyeY, accentColor, sec === "DROP");
      this.drawAnimeBlinkEye(ctx, rightEyeX, eyeY, accentColor, sec === "DROP");
    } else if (this.temporaryAction === "PET") {
      ctx.shadowBlur = 14;
      ctx.shadowColor = "#ff69b4";
      ctx.fillStyle = "#ff69b4";
      this.drawHeart(ctx, leftEyeX, eyeY, 11);
      this.drawHeart(ctx, rightEyeX, eyeY, 11);
    } else {
      this.drawWaveformEye(
        ctx, leftEyeX, eyeY, drumWaveform, bassWaveform, drumLevel, bassLevel,
        sec === "DROP", isDropGap, accentColor, cx
      );
      this.drawWaveformEye(
        ctx, rightEyeX, eyeY, drumWaveform, bassWaveform, drumLevel, bassLevel,
        sec === "DROP", isDropGap, accentColor, cx
      );
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 10. CUTE BEAR SNOUT / MUZZLE
    // -------------------------------------------------------------------------
    const muzzleY = cy + 6 * scaleY;
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, muzzleY, 13 * scaleX, 10 * scaleY, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 248, 252, 0.36)";
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 220, 245, 0.65)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();

    // -------------------------------------------------------------------------
    // 11. KAWAII BEAR BUTTON NOSE
    // -------------------------------------------------------------------------
    const noseY = cy + 2 * scaleY;
    this.drawKawaiiBearNose(
      ctx, cx, noseY,
      env.drums, env.bass, env.vocals, mixEnv,
      isSinging, waveform, accentColor, baseScale
    );

    // -------------------------------------------------------------------------
    // 12. THE LIVE VOCAL WAVEFORM AUDIO REACTIVE MOUTH
    //     Sexy real lipstick lips + visible teeth, tongue, oral cavity & throat
    //     driven instantaneously by the Singing Head / CHIMERA synth params.
    // -------------------------------------------------------------------------
    const lm = this.liveMouth;
    const lmActive = Boolean(lm?.active);
    const liveArt = lm?.rawArt ?? lm?.art;
    const liveTract = lm?.rawTract ?? lm?.tract;
    const mouthY = cy + 13.5 * scaleY
      // Pitch lift: higher F0 rides the mouth up a touch (subtle, ±2px).
      + (this.vocalF0Hz > 0
        ? Math.max(-0.35, Math.min(0.55, (this.vocalF0Hz - 220) / 700)) * 3.0 * scaleY
        : 0);
    // Responsive aperture: blend fast acoustic follower with low-latency Head Synth aperture.
    const mouthAperture = Math.max(
      0,
      Math.min(
        1,
        Math.max(
          this.smoothedVocalAmp * 2.5,
          lmActive && liveArt ? Math.max(liveArt.aperture, liveArt.jawDrop * 0.9) : 0
        )
      )
    );
    const activeSing = isSinging || lmActive || vocalsLevel >= 0.004 || mouthAperture > 0.04;

    const mouthSteps = 64;
    const mouthGlow = activeSing ? "#ff2e93" : "#ff4da6";

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const buildLipColors = (): { top: string; mid: string; shade: string; bot: string; liner: string } => {
      if (vis.mouthStyle === "acid") {
        return { top: "#eaff7b", mid: "#9dff00", shade: "#008f4c", bot: "#5cff9d", liner: "#005e36" };
      }
      if (vis.mouthStyle === "cyber") {
        return { top: "#a6f6ff", mid: "#00c8ff", shade: "#4c1d95", bot: "#ff66e5", liner: "#2e1065" };
      }
      // Classic glossy ruby-rose real lipstick
      return { top: "#ff6b9d", mid: "#e00751", shade: "#8f0030", bot: "#ff266e", liner: "#610021" };
    };
    const lc = buildLipColors();

    // Sexy Cupid's bow + pillowy bottom contour helper (parametric true ellipse)
    const cupidWave = (cosP: number, roundFrac: number): number => {
      // Twin M-arch peaks around |cosP| ~ 0.38 with subtle center cleft dip
      const centerCleft = Math.exp(-Math.pow(cosP * 4.2, 2));
      const sidePeaks = Math.exp(-Math.pow((Math.abs(cosP) - 0.36) * 4.4, 2));
      return (sidePeaks * 1.15 - centerCleft * 0.65) * (1 - 0.55 * roundFrac);
    };

    if (!activeSing) {
      // Sexy slightly-parted glossy resting lipstick pout (Cupid's bow + plush lower pillow)
      const breathe = Math.sin(t * 2.4) * 0.45;
      const rw = (11.2 + breathe * 0.5) * scaleX * (0.8 + 0.2 * vis.mouthGain);
      const rhUpper = (3.4 + breathe * 0.25) * scaleY * vis.mouthGain;
      const rhLower = (4.8 + breathe * 0.3) * scaleY * vis.mouthGain;

      ctx.beginPath();
      for (let j = 0; j <= mouthSteps; j++) {
        const phi = (j / mouthSteps) * Math.PI * 2;
        const cosP = Math.cos(phi);
        const sinP = Math.sin(phi);
        const x = cx + rw * cosP;
        let y: number;
        if (sinP <= 0) {
          // Taper cleanly to corners (sinP == 0 -> y = mouthY)
          const bow = cupidWave(cosP, 0.1) * 2.2 * scaleY;
          y = mouthY + rhUpper * sinP - bow * Math.pow(-sinP, 0.6);
        } else {
          y = mouthY + rhLower * Math.pow(sinP, 0.85);
        }
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();

      const lipGrad = ctx.createLinearGradient(cx, mouthY - rhUpper, cx, mouthY + rhLower);
      lipGrad.addColorStop(0.0, lc.top);
      lipGrad.addColorStop(0.42, lc.mid);
      lipGrad.addColorStop(0.55, lc.shade);
      lipGrad.addColorStop(1.0, lc.bot);
      ctx.fillStyle = lipGrad;
      ctx.shadowBlur = 9;
      ctx.shadowColor = mouthGlow;
      ctx.fill();
      ctx.strokeStyle = lc.liner;
      ctx.lineWidth = 1.05;
      ctx.stroke();

      // Natural slightly-curved sensual lip parting seam
      ctx.beginPath();
      ctx.moveTo(cx - rw * 0.94, mouthY);
      ctx.quadraticCurveTo(cx - rw * 0.35, mouthY + 1.0 * scaleY, cx, mouthY + 0.2 * scaleY);
      ctx.quadraticCurveTo(cx + rw * 0.35, mouthY + 1.0 * scaleY, cx + rw * 0.94, mouthY);
      ctx.strokeStyle = "#2b0010";
      ctx.lineWidth = 1.45;
      ctx.stroke();

      // Twin Cupid's bow glints + lush wet lower-lip specular gloss
      ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.ellipse(cx + s * rw * 0.26, mouthY - rhUpper * 0.58, 1.8 * baseScale, 0.65 * baseScale, s * -0.25, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.ellipse(cx, mouthY + rhLower * 0.52, rw * 0.36, 1.15 * scaleY, 0, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.78)";
      ctx.fill();
    } else {
      // Live responsive articulation from low-latency Head Synth / Singing Head params
      const upVowel = ((liveArt && lmActive ? liveArt.topVowel : vowel) || "AH").toUpperCase();

      let baseW = 13.5;
      let baseH = 9.5;
      if (upVowel === "EE") { baseW = 17.0; baseH = 5.2; }
      else if (upVowel === "OH") { baseW = 9.6; baseH = 12.2; }
      else if (upVowel === "OO") { baseW = 7.4; baseH = 6.8; }
      else if (upVowel === "AY") { baseW = 15.2; baseH = 8.4; }

      if (this.vocalF2 > 500) {
        baseW *= Math.max(0.68, Math.min(1.35, 0.84 + (this.vocalF2 - 1200) / 1500));
      }

      const artRound = liveTract ? Math.max(0, Math.min(1, liveTract.lipRound)) : (liveArt?.roundness ?? 0.2);
      const artWidth = liveArt ? Math.max(0, Math.min(1, liveArt.width)) : 0.55;
      const artOpen = liveArt
        ? Math.max(liveArt.aperture, liveArt.jawDrop * 0.92) + Math.max(0, liveTract?.lipOpen ?? 0) * 0.35
        : mouthAperture;

      const artW = lmActive && liveArt
        ? 9.2 + 9.4 * artWidth - 5.4 * artRound
        : baseW * (0.68 + 0.52 * mouthAperture);
      const artH = lmActive && liveArt
        ? 1.4 + 12.6 * Math.min(1.15, artOpen)
        : baseH * (0.35 + 0.95 * mouthAperture);

      const vibWobble = 1 + Math.sin(t * Math.PI * 2 * 5.0)
        * Math.min(0.14, this.vocalVibratoSemis * 0.14);
      const curW = Math.max(6.8, artW * vibWobble) * scaleX * (0.78 + 0.22 * vis.mouthGain);
      const curH = Math.max(1.8, artH * vibWobble) * scaleY * vis.mouthGain;

      const lipRound = artRound;
      const lipTuck = liveTract ? Math.max(0, Math.min(1, liveTract.lipTuck)) : 0;
      const velumOpen = liveTract ? Math.max(0, Math.min(1, liveTract.velum)) : 0.25;
      const tX = liveTract ? Math.max(-1.4, Math.min(1.4, liveTract.tongueX)) : (upVowel === "EE" ? 0.7 : upVowel === "OH" ? -0.5 : 0);
      const tY = liveTract ? Math.max(-1.2, Math.min(1.3, liveTract.tongueY)) : (upVowel === "EE" || upVowel === "OO" ? 0.75 : upVowel === "AH" ? -0.45 : 0.1);
      const tipLift = liveTract ? Math.max(-0.8, Math.min(1.2, liveTract.tipY)) : (upVowel === "EE" ? 0.4 : 0);

      // Compute true elliptical inner cavity & tapered outer sexy lipstick lips (64 steps)
      const innerPts: { x: number; y: number }[] = [];
      const outerPts: { x: number; y: number }[] = [];
      const upperThickMax = (3.1 + 1.1 * lipRound) * scaleY * (0.8 + 0.2 * vis.mouthGain);
      const lowerThickMax = (4.3 + 1.3 * lipRound) * (1 - 0.35 * lipTuck) * scaleY * (0.8 + 0.2 * vis.mouthGain);
      const waveLen = waveform ? waveform.length : 0;

      for (let j = 0; j <= mouthSteps; j++) {
        const phi = (j / mouthSteps) * Math.PI * 2;
        const cosP = Math.cos(phi);
        const sinP = Math.sin(phi);
        let sampleNorm = 0;
        if (waveform && waveLen > 0) {
          const wIdx = Math.floor((j / mouthSteps) * (waveLen - 1));
          sampleNorm = (waveform[wIdx] - 128) / 128.0;
        } else {
          sampleNorm = Math.sin(phi * 3.0 + t * 14.0) * 0.25;
        }

        // Exact ellipse point + subtle acoustic rim vibration (never distorts corners)
        const cornerWeight = Math.pow(Math.abs(sinP), 0.72);
        const waveDisp = sampleNorm * (1.5 * mouthAperture + 0.35) * cornerWeight;
        const innerBow = sinP < 0 ? cupidWave(cosP, lipRound) * 0.65 * scaleY * (-sinP) : 0;
        const ix = cx + curW * cosP;
        const iy = mouthY + (curH + waveDisp) * sinP - innerBow;
        innerPts.push({ x: ix, y: iy });

        // Outer lipstick contour tapers gracefully cleanly to the sharp corners!
        let oy: number;
        const ox = cx + (curW + 0.8 * lipRound * cornerWeight) * cosP;
        if (sinP <= 0) {
          const upperBow = cupidWave(cosP, lipRound) * 2.4 * scaleY * Math.pow(-sinP, 0.55);
          oy = iy - upperThickMax * Math.pow(-sinP, 0.72) - upperBow;
        } else {
          oy = iy + lowerThickMax * Math.pow(sinP, 0.78);
        }
        outerPts.push({ x: ox, y: oy });
      }

      const traceLoop = (pts: { x: number; y: number }[]) => {
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
          if (i === 0) ctx.moveTo(pts[i].x, pts[i].y);
          else ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();
      };

      // A) Deep Dark Oral Cavity + Glowing Back Throat (Pharynx / Soft Palate / Uvula)
      ctx.save();
      traceLoop(innerPts);
      ctx.clip();

      const throatGrad = ctx.createRadialGradient(
        cx, mouthY - curH * 0.1, curH * 0.08,
        cx, mouthY, Math.max(curW, curH) * 1.25
      );
      throatGrad.addColorStop(0.0, `rgba(255, 55, 130, ${0.38 + 0.55 * mouthAperture})`);
      throatGrad.addColorStop(0.45, "#590425");
      throatGrad.addColorStop(1.0, "#120108");
      ctx.fillStyle = throatGrad;
      ctx.fillRect(cx - curW * 1.3, mouthY - curH * 1.4, curW * 2.6, curH * 2.8);

      // Soft palate arches & dangling uvula at back of throat when open
      if (curH > 4.2) {
        const uvulaY = mouthY - curH * (0.28 - velumOpen * 0.14);
        const uvSway = Math.sin(t * 22) * 1.0 * mouthAperture;
        ctx.strokeStyle = "rgba(255, 115, 175, 0.58)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, uvulaY + curH * 0.14, curW * 0.42, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();

        ctx.fillStyle = `rgba(255, 125, 185, ${0.75 + 0.25 * mouthAperture})`;
        ctx.beginPath();
        ctx.ellipse(cx + uvSway, uvulaY, 1.8 * scaleX, 2.8 * scaleY, uvSway * 0.08, 0, Math.PI * 2);
        ctx.fill();
      }

      // B) Articulated Shaded Tongue Body & Tip
      const tongueCenterY = mouthY + curH * (0.52 - 0.36 * tY - 0.15 * tipLift);
      const tongueCenterX = cx + tX * curW * 0.14;
      const tongueW = curW * 0.72;
      const tongueH = Math.max(3.0, curH * 0.52);
      const tongueGrad = ctx.createRadialGradient(
        tongueCenterX, tongueCenterY - tongueH * 0.3, 1,
        tongueCenterX, tongueCenterY, tongueW
      );
      if (vis.mouthStyle === "acid") {
        tongueGrad.addColorStop(0.0, "#a5f3fc");
        tongueGrad.addColorStop(0.55, "#00f0ff");
        tongueGrad.addColorStop(1.0, "#0055ff");
      } else if (vis.mouthStyle === "cyber") {
        tongueGrad.addColorStop(0.0, "#f5d0fe");
        tongueGrad.addColorStop(0.55, "#ff00e6");
        tongueGrad.addColorStop(1.0, "#5b0099");
      } else {
        tongueGrad.addColorStop(0.0, "#ffa6cb");
        tongueGrad.addColorStop(0.55, "#f04286");
        tongueGrad.addColorStop(1.0, "#9e144b");
      }
      ctx.fillStyle = tongueGrad;
      ctx.beginPath();
      ctx.ellipse(tongueCenterX, tongueCenterY + tongueH * 0.2, tongueW, tongueH, 0, Math.PI, Math.PI * 2);
      ctx.ellipse(tongueCenterX, tongueCenterY + tongueH * 0.2, tongueW, tongueH * 0.65, 0, 0, Math.PI);
      ctx.fill();

      // Tongue dorsal groove + wet highlight
      ctx.strokeStyle = "rgba(120, 10, 50, 0.55)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(tongueCenterX, tongueCenterY - tongueH * 0.4);
      ctx.lineTo(tongueCenterX, tongueCenterY + tongueH * 0.15);
      ctx.stroke();

      // C) Curved Pearl Teeth Rows
      const toothVis = Math.max(0, Math.min(1, (curH - 2.6) / 4.5));
      if (toothVis > 0.05) {
        const toothCount = 6;
        const archW = curW * 1.38;
        const cellW = archW / toothCount;
        const startX = cx - archW * 0.5;
        const upperToothH = Math.max(1.8, Math.min(4.2, curH * 0.26)) * toothVis;
        const upperY = mouthY - curH * 0.98;

        ctx.fillStyle = "rgba(255, 252, 250, 0.96)";
        ctx.strokeStyle = "rgba(210, 195, 215, 0.6)";
        ctx.lineWidth = 0.65;
        for (let k = 0; k < toothCount; k++) {
          const distFromCtr = Math.abs((k + 0.5) / toothCount - 0.5) * 2;
          const curveDrop = (1 - distFromCtr * distFromCtr) * 0.9 * scaleY;
          const tx = startX + k * cellW + 0.45;
          const tw = Math.max(1.6, cellW - 0.9);
          ctx.beginPath();
          ctx.roundRect(tx, upperY + curveDrop, tw, upperToothH, [0, 0, 1.8, 1.8]);
          ctx.fill();
          ctx.stroke();
        }

        if (curH > 6.8 && lipTuck < 0.45) {
          const lowerToothH = Math.max(1.4, Math.min(3.2, curH * 0.18)) * toothVis;
          const lowerY = mouthY + curH * 0.98 - lowerToothH;
          for (let k = 0; k < toothCount; k++) {
            const tx = startX + k * cellW + 0.7;
            const tw = Math.max(1.4, cellW - 1.4);
            ctx.beginPath();
            ctx.roundRect(tx, lowerY, tw, lowerToothH, [1.5, 1.5, 0, 0]);
            ctx.fill();
            ctx.stroke();
          }
        }
      }
      ctx.restore();

      // D) Sculpted Sexy Lipstick Cushion Ring (Even-Odd fill between outerPts & innerPts)
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < outerPts.length; i++) {
        if (i === 0) ctx.moveTo(outerPts[i].x, outerPts[i].y);
        else ctx.lineTo(outerPts[i].x, outerPts[i].y);
      }
      ctx.closePath();
      for (let i = innerPts.length - 1; i >= 0; i--) {
        if (i === innerPts.length - 1) ctx.moveTo(innerPts[i].x, innerPts[i].y);
        else ctx.lineTo(innerPts[i].x, innerPts[i].y);
      }
      ctx.closePath();

      const lipGrad = ctx.createLinearGradient(cx, mouthY - curH - upperThickMax, cx, mouthY + curH + lowerThickMax);
      lipGrad.addColorStop(0.0, lc.top);
      lipGrad.addColorStop(0.35, lc.mid);
      lipGrad.addColorStop(0.58, lc.shade);
      lipGrad.addColorStop(1.0, lc.bot);
      ctx.fillStyle = lipGrad;
      ctx.shadowBlur = 10 + 14 * mouthAperture;
      ctx.shadowColor = mouthGlow;
      ctx.fill("evenodd");

      // Sensual darker lip-liner rim & inner shadow
      traceLoop(outerPts);
      ctx.strokeStyle = lc.liner;
      ctx.lineWidth = 1.05;
      ctx.stroke();

      traceLoop(innerPts);
      ctx.strokeStyle = "#360012";
      ctx.lineWidth = 0.95;
      ctx.stroke();

      // Glossy wet lipstick specular highlights (Cupid's bow peaks + plush bottom lip)
      ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
      for (const s of [-1, 1]) {
        ctx.beginPath();
        ctx.ellipse(
          cx + s * curW * 0.28,
          mouthY - curH - upperThickMax * 0.42,
          Math.max(1.4, curW * 0.14),
          0.65 * scaleY,
          s * -0.22,
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
      ctx.beginPath();
      ctx.ellipse(
        cx,
        mouthY + curH + lowerThickMax * 0.48,
        curW * 0.36,
        Math.max(0.85, lowerThickMax * 0.24),
        0,
        0,
        Math.PI * 2
      );
      ctx.fillStyle = "rgba(255, 255, 255, 0.78)";
      ctx.fill();
      ctx.restore();
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 4. FLOATING SPARKS & MUSICAL PARTICLES
    // -------------------------------------------------------------------------
    ctx.save();
    for (const note of this.noteFloating) {
      ctx.font = "bold 13px monospace";
      ctx.fillStyle = mainColor;
      ctx.shadowBlur = 10;
      ctx.shadowColor = accentColor;
      ctx.globalAlpha = Math.max(0, Math.min(1.0, note.life));
      ctx.fillText(note.char, note.x, note.y);
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 5. LYRICS & ACTIVE VOCAL METADATA DISPLAY
    // -------------------------------------------------------------------------
    if (this.lyricBarEl) {
      if (meta?.vocalNote) {
        this.lyricBarEl.innerHTML = `<span style="color: #ffff00;">♪ SING: ${meta.vocalNote} [${vowel || "--"}]</span> • <span style="color: #00ffff;">${this.speechBubble || meta.vocalLyric || sec}</span>`;
      } else {
        this.lyricBarEl.innerHTML = `<span style="color: #00ffaa;">♫ ${sec}</span> • <span style="color: #ff77e9;">${this.speechBubble || "WUB GROOVE"}</span>`;
      }
    }
  }

  private drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number, outerR: number, innerR: number): void {
    ctx.beginPath();
    for (let i = 0; i < 8; i++) {
      const r = i % 2 === 0 ? outerR : innerR;
      const a = (i / 8) * Math.PI * 2 - Math.PI / 2;
      const x = cx + r * Math.cos(a);
      const y = cy + r * Math.sin(a);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }

  private drawHeart(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number): void {
    ctx.beginPath();
    const topCurveHeight = size * 0.3;
    ctx.moveTo(cx, cy + size * 0.2);
    ctx.bezierCurveTo(cx, cy - topCurveHeight, cx - size, cy - topCurveHeight, cx - size, cy + size * 0.1);
    ctx.bezierCurveTo(cx - size, cy + size * 0.5, cx, cy + size * 0.8, cx, cy + size);
    ctx.bezierCurveTo(cx, cy + size * 0.8, cx + size, cy + size * 0.5, cx + size, cy + size * 0.1);
    ctx.bezierCurveTo(cx + size, cy - topCurveHeight, cx, cy - topCurveHeight, cx, cy + size * 0.2);
    ctx.closePath();
    ctx.fill();
  }

  private drawAnimeBlinkEye(ctx: CanvasRenderingContext2D, eyeX: number, eyeY: number, color: string, isDrop: boolean): void {
    ctx.save();
    ctx.strokeStyle = isDrop ? "#ffffff" : color;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.lineWidth = 3.6;
    ctx.lineCap = "round";

    // Cute curved anime blink arch: ( ◡ ) with outer lash flick
    const w = 13;
    ctx.beginPath();
    ctx.arc(eyeX, eyeY + 2, w, 1.15 * Math.PI, 1.85 * Math.PI, false);
    ctx.stroke();

    // Outer anime lash flick
    const canvasWidth = this.canvasEl ? this.canvasEl.width : 200;
    const flickDir = eyeX < canvasWidth * 0.5 ? -1 : 1;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(eyeX + flickDir * (w * 0.85), eyeY - 2);
    ctx.lineTo(eyeX + flickDir * (w + 4), eyeY - 6);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * One eye = one instrument. Left eye follows drums, right eye follows bass;
   * the iris ripple, pupil size and internal sparkle use only that stem's
   * waveform/envelope, while both eyes keep the original cyan/gold palette.
   */
  private drawWaveformEye(
    ctx: CanvasRenderingContext2D,
    eyeX: number,
    eyeY: number,
    drumWaveform?: Uint8Array | null,
    bassWaveform?: Uint8Array | null,
    drumLevel: number = 0,
    bassLevel: number = 0,
    isDrop: boolean = false,
    isDropGap: boolean = false,
    accentColor: string = "#00ffff",
    faceCenterX: number = 100
  ): void {
    const eyeSteps = 36;
    const t = performance.now() / 1000;

    // Anime Eye Dimensions: Big expressive vertical oval
    const baseIrisRx = (isDropGap ? 13.5 : (isDrop ? 12.5 : 11.0)) + Math.min(3.5, bassLevel * 3.8);
    const baseIrisRy = (isDropGap ? 19.5 : (isDrop ? 18.0 : 16.0)) + Math.min(4.5, bassLevel * 5.0);

    const bassWaveLen = bassWaveform ? bassWaveform.length : 0;
    const irisStroke = isDrop ? "#00ffff" : "#00ffcc";
    const irisGlow = isDrop ? "#00e5ff" : "#00ffaa";

    // 1. Sclera / Eye White backing for crisp anime pop
    ctx.save();
    ctx.fillStyle = "rgba(240, 248, 255, 0.94)";
    ctx.beginPath();
    ctx.ellipse(eyeX, eyeY, baseIrisRx + 2.5, baseIrisRy + 2.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 2. Iris: Rich Anime Multi-Color Gradient with Audio-Reactive Bass Waveform Ripple
    ctx.save();
    ctx.shadowBlur = isDrop ? 14 : 9;
    ctx.shadowColor = irisGlow;
    ctx.strokeStyle = irisStroke;
    ctx.lineWidth = 2.2;

    const grad = ctx.createLinearGradient(eyeX, eyeY - baseIrisRy, eyeX, eyeY + baseIrisRy);
    if (isDrop) {
      grad.addColorStop(0.0, "rgba(20, 0, 50, 0.98)");
      grad.addColorStop(0.35, "rgba(0, 180, 255, 0.88)");
      grad.addColorStop(0.80, "rgba(0, 255, 230, 0.95)");
      grad.addColorStop(1.0, "rgba(255, 255, 255, 0.9)");
    } else {
      grad.addColorStop(0.0, "rgba(10, 15, 40, 0.96)");
      grad.addColorStop(0.45, "rgba(0, 120, 215, 0.85)");
      grad.addColorStop(0.85, "rgba(0, 255, 170, 0.92)");
      grad.addColorStop(1.0, "rgba(180, 255, 235, 0.9)");
    }
    ctx.fillStyle = grad;

    ctx.beginPath();
    for (let j = 0; j <= eyeSteps; j++) {
      const phi = (j / eyeSteps) * Math.PI * 2;
      let bSample = 0;
      if (bassWaveform && bassWaveLen > 0) {
        const idx = Math.floor((j / eyeSteps) * (bassWaveLen - 1));
        bSample = (bassWaveform[idx] - 128) / 128.0;
      } else {
        bSample = Math.sin(phi * 2.0 + t * 7.0) * 0.22;
      }
      const disp = bSample * (3.0 * Math.min(1.5, bassLevel * 2.5) + (isDropGap ? 1.6 : 0.7));
      const rx = Math.max(2.0, baseIrisRx + disp * 0.7);
      const ry = Math.max(3.0, baseIrisRy + disp);
      const x = eyeX + rx * Math.cos(phi);
      const y = eyeY + ry * Math.sin(phi);
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // 3. Inner Pupil: Dark Anime Core pulsing to Drum Waveform
    const basePupilRx = (isDropGap ? 7.2 : (isDrop ? 6.5 : 5.5)) + Math.min(2.5, drumLevel * 3.5);
    const basePupilRy = (isDropGap ? 11.0 : (isDrop ? 9.8 : 8.5)) + Math.min(3.5, drumLevel * 4.5);
    const drumWaveLen = drumWaveform ? drumWaveform.length : 0;
    const pupilStroke = isDrop ? "#ffffff" : "#ffe600";
    const pupilGlow = isDrop ? "#ffffff" : "#ffbb00";

    ctx.save();
    ctx.shadowBlur = 8;
    ctx.shadowColor = pupilGlow;
    ctx.strokeStyle = pupilStroke;
    ctx.fillStyle = "rgba(10, 5, 24, 0.92)";
    ctx.lineWidth = 1.8;

    ctx.beginPath();
    for (let j = 0; j <= eyeSteps; j++) {
      const phi = (j / eyeSteps) * Math.PI * 2;
      let dSample = 0;
      if (drumWaveform && drumWaveLen > 0) {
        const idx = Math.floor((j / eyeSteps) * (drumWaveLen - 1));
        dSample = (drumWaveform[idx] - 128) / 128.0;
      } else {
        dSample = Math.sin(phi * 3.0 + t * 11.0) * 0.22;
      }
      const disp = dSample * (2.4 * Math.min(1.5, drumLevel * 2.5) + (isDropGap ? 1.4 : 0.5));
      const rx = Math.max(1.5, basePupilRx + disp * 0.6);
      const ry = Math.max(2.0, basePupilRy + disp);
      const x = eyeX + rx * Math.cos(phi);
      const y = eyeY + ry * Math.sin(phi);
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // 4. Sweeping Anime Upper Eyelash Line & Wing Flick
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = isDrop ? "#ffffff" : "#0d1b2a";
    ctx.lineWidth = 3.6;
    ctx.shadowBlur = 8;
    ctx.shadowColor = accentColor;

    const isLeft = eyeX < faceCenterX;
    const flickDir = isLeft ? -1 : 1;
    const lashRx = baseIrisRx * 1.25;

    ctx.beginPath();
    ctx.moveTo(eyeX - flickDir * (lashRx * 0.65), eyeY - baseIrisRy * 0.70);
    ctx.quadraticCurveTo(eyeX, eyeY - baseIrisRy * 1.18, eyeX + flickDir * lashRx, eyeY - baseIrisRy * 0.85);
    // Outer anime lash flick
    ctx.lineTo(eyeX + flickDir * (lashRx + 5), eyeY - baseIrisRy * 1.10);
    ctx.stroke();

    // Delicate lower anime lash accent
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(13, 27, 42, 0.65)";
    ctx.beginPath();
    ctx.arc(eyeX, eyeY + baseIrisRy * 0.92, baseIrisRx * 0.55, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();
    ctx.restore();

    // 5. Classic Anime Catchlights (Big sparkling glint + secondary glints)
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#ffffff";

    // Primary large anime catchlight (upper left)
    const glintPulse = 1.0 + 0.15 * Math.sin(t * 6.0);
    const glint1R = (isDrop ? 4.5 : 3.8) * glintPulse;
    ctx.beginPath();
    ctx.ellipse(eyeX - baseIrisRx * 0.36, eyeY - baseIrisRy * 0.38, glint1R, glint1R * 1.1, -0.3, 0, Math.PI * 2);
    ctx.fill();

    // Secondary soft anime catchlight (lower right)
    const glint2R = isDrop ? 2.2 : 1.8;
    ctx.beginPath();
    ctx.arc(eyeX + baseIrisRx * 0.38, eyeY + baseIrisRy * 0.36, glint2R, 0, Math.PI * 2);
    ctx.fill();

    // Tertiary micro sparkle (lower left iris reflection)
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.arc(eyeX - baseIrisRx * 0.25, eyeY + baseIrisRy * 0.45, 1.2, 0, Math.PI * 2);
    ctx.fill();

    // If DROP: Add a dazzling 4-point anime star in the center!
    if (isDrop) {
      ctx.globalAlpha = 0.9;
      this.drawStar(ctx, eyeX, eyeY, 5.0, 1.8);
    }
    ctx.restore();
  }

  private drawKawaiiBearNose(
    ctx: CanvasRenderingContext2D,
    noseX: number,
    noseY: number,
    drumLevel: number = 0,
    bassLevel: number = 0,
    vocalsLevel: number = 0,
    rmsLevel: number = 0,
    isSinging: boolean = false,
    waveform?: Uint8Array | null,
    accentColor: string = "#00ffff",
    baseScale: number = 0.88
  ): void {
    ctx.save();
    const t = performance.now() / 1000;
    const audioEnergy = Math.min(1.5, drumLevel * 1.8 + rmsLevel * 1.2 + (isSinging ? 0.4 : 0.0));
    const pulse = 1.0 + 0.22 * audioEnergy;
    const r = 3.6 * baseScale * pulse;

    ctx.translate(noseX, noseY);

    // Cute rounded kawaii heart / bear button nose
    ctx.beginPath();
    ctx.moveTo(0, r * 0.45);
    ctx.bezierCurveTo(-r * 1.3, -r * 0.75, -r * 1.3, -r * 1.25, 0, -r * 0.55);
    ctx.bezierCurveTo(r * 1.3, -r * 1.25, r * 1.3, -r * 0.75, 0, r * 0.45);
    ctx.closePath();

    const noseColor = isSinging ? "#ffff00" : (drumLevel > 0.45 ? "#ff0077" : "#ff3399");
    const noseGlow = isSinging ? "#ffea00" : "#ff00a0";

    ctx.shadowBlur = 8 + 12 * audioEnergy;
    ctx.shadowColor = noseGlow;
    ctx.fillStyle = noseColor;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Sparkling specular catchlight on nose
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(-r * 0.35, -r * 0.65, 1.1 * pulse, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  private drawDynamicNose(
    ctx: CanvasRenderingContext2D,
    noseX: number,
    noseY: number,
    drumLevel: number = 0,
    bassLevel: number = 0,
    vocalsLevel: number = 0,
    rmsLevel: number = 0,
    isSinging: boolean = false,
    waveform?: Uint8Array | null,
    accentColor: string = "#00ffff"
  ): void {
    this.drawKawaiiBearNose(ctx, noseX, noseY, drumLevel, bassLevel, vocalsLevel, rmsLevel, isSinging, waveform, accentColor);
  }

  public render(): void {
    this.renderCanvas(null, false);
    this.renderASCII(null, false);
    this.updateDashboard();
  }

  // ---------------------------------------------------------------------------
  // Persistence (LocalStorage)
  // ---------------------------------------------------------------------------

  private saveState(): void {
    try {
      const data: TamagotchiStats = {
        xp: this.xp,
        stage: this.stage,
        stageName: STAGE_CONFIGS[this.stage].name,
        vibe: this.vibe,
        mood: this.mood,
        affection: this.affection,
        favoriteStem: this.favoriteStem,
        barsListened: this.barsListened,
      };
      localStorage.setItem("wub_tamagotchi_save", JSON.stringify(data));
    } catch {
      // Storage unavailable
    }
  }

  private loadState(): void {
    try {
      const raw = localStorage.getItem("wub_tamagotchi_save");
      if (raw) {
        const d = JSON.parse(raw) as Partial<TamagotchiStats>;
        if (typeof d.xp === "number") this.xp = d.xp;
        if (typeof d.stage === "number" && (d.stage >= 1 && d.stage <= 4)) this.stage = d.stage as TamagotchiStage;
        if (typeof d.affection === "number") this.affection = d.affection;
        if (typeof d.favoriteStem === "string") this.favoriteStem = d.favoriteStem;
        if (typeof d.barsListened === "number") this.barsListened = d.barsListened;
        this.checkEvolution();
      }
    } catch {
      // Use defaults
    }
  }

  // ---------------------------------------------------------------------------
  // String Utilities
  // ---------------------------------------------------------------------------

  private padCenter(text: string, len: number): string {
    const clean = text.substring(0, len);
    const diff = len - clean.length;
    if (diff <= 0) return clean;
    const left = Math.floor(diff / 2);
    const right = diff - left;
    return " ".repeat(left) + clean + " ".repeat(right);
  }

  private truncate(text: string, max: number): string {
    return text.length > max ? text.substring(0, max - 2) + ".." : text;
  }
}
