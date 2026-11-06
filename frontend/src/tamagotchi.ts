// -----------------------------------------------------------------------------
// WubGotchi: Neural Music Tamagotchi Avatar
// Autonomous ASCII character that listens, dances, sings, and evolves with the music.
// -----------------------------------------------------------------------------

import type { GrooveboxMeta } from "./main";

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
 * Builds the smooth closed Catmull-Rom parametric loop for the Kawaii Gummi Bear body contour.
 * Outward unit normals are precomputed for seamless audio-reactive wave displacement.
 */
function buildGummiBearPoints(): GummiLoopVertex[] {
  const halfPts: [number, number][] = [
    [0.0, -48.0],     // 0: top head center dip
    [14.0, -50.0],    // 1: top head slope
    [23.0, -53.0],    // 2: inner ear base
    [30.0, -66.0],    // 3: inner ear curve
    [40.0, -68.0],    // 4: ear peak
    [48.0, -59.0],    // 5: ear outer round
    [44.0, -46.0],    // 6: outer ear base
    [46.0, -33.0],    // 7: upper cheek
    [50.0, -16.0],    // 8: chubby cheek max
    [45.0, -2.0],     // 9: lower cheek curve
    [35.0, 8.0],      // 10: cute neck pinch
    [40.0, 14.0],     // 11: shoulder
    [52.0, 19.0],     // 12: arm top
    [58.0, 27.0],     // 13: chubby paw tip
    [52.0, 35.0],     // 14: paw bottom round
    [42.0, 38.0],     // 15: arm under
    [36.0, 42.0],     // 16: waist indent
    [44.0, 52.0],     // 17: chubby tummy upper
    [48.0, 64.0],     // 18: chubby tummy max
    [42.0, 76.0],     // 19: hip
    [34.0, 84.0],     // 20: outer foot
    [22.0, 88.0],     // 21: foot bottom round
    [11.0, 84.0],     // 22: inner foot
    [0.0, 74.0],      // 23: crotch center
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

const GUMMI_BEAR_LOOP_POINTS: GummiLoopVertex[] = buildGummiBearPoints();

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

  // Per-stem reactive envelopes. Every visual feature reads one instrument so
  // the bear reads like a four-piece band: eyes=drums/bass, mouth=vocals,
  // heart=other, body/halo/motes=the full mix.
  private stemEnv = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private stemPunch = { drums: 0, bass: 0, other: 0, vocals: 0 };
  private mixEnv: number = 0;
  private mixPunch: number = 0;
  private otherTone: number = 0;

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

    this.render();
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
    masterWaveform?: Uint8Array | null
  ): void {
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
      bassLevel ?? 0
    );
    this.renderASCII(meta, isDropGap, waveform, dt, bpm);
    this.updateDashboard();
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
  /** Tiny ASCII eye glyph driven only by one stem's envelope/punch. */
  private stemEyeGlyph(level: number, punch: number): string {
    if (punch > 0.55 || level > 0.72) return "◉";
    if (level > 0.40) return "◕";
    if (punch > 0.18) return "✦";
    if (level < 0.08) return "◔";
    return "◕";
  }

  private renderASCII(
    meta: GrooveboxMeta | null,
    isDropGap: boolean,
    waveform?: Uint8Array | null,
    dt: number = 0.016,
    bpm: number = 126
  ): void {
    if (!this.asciiEl) return;

    const sec = meta?.section ?? "GROOVE_A";
    const vowel = meta?.vocalVowel ?? "";
    const isSinging = Boolean(meta?.vocalSyllable) || this.stemEnv.vocals >= 0.02;
    const bob = this.isHeadBobbed;

    // Toggle glowing psychedelic singing shimmer
    if (isSinging) {
      this.asciiEl.classList.add("singing");
    } else {
      this.asciiEl.classList.remove("singing");
    }

    // Determine Big Expressive Anime Eyes. Left=drums, right=bass; mood states
    // still override both, but the default pair breathes with its own stem.
    let leftEye = this.stemEyeGlyph(this.stemEnv.drums, this.stemPunch.drums);
    let rightEye = this.stemEyeGlyph(this.stemEnv.bass, this.stemPunch.bass);
    if (this.isBlinking) {
      leftEye = "◡";
      rightEye = "◡";
    } else if (isDropGap) {
      leftEye = "⊙";
      rightEye = "⊙";
    } else if (this.temporaryAction === "PET") {
      leftEye = "♥";
      rightEye = "♥";
    } else if (sec === "DROP" || this.temporaryAction === "HYPE") {
      leftEye = "★";
      rightEye = "★";
    } else if (sec === "BUILDUP") {
      leftEye = "✧";
      rightEye = "✧";
    } else if (sec === "BREAKDOWN") {
      leftEye = "˘";
      rightEye = "˘";
    }
    const eyes = `${leftEye}   ${rightEye}`;

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
    const mouth = this.extractWaveformMouth(waveform, isSinging, vowel, isDropGap, dt, bpm, this.stemEnv.vocals);

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
    bassLevel: number = 0
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
    const vowel = meta?.vocalVowel ?? "";
    const isSinging = Boolean(meta?.vocalSyllable) || (this.stemEnv.vocals >= 0.02);
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

    // Center coordinates with beat bobbing
    const cx = w * 0.5;
    const bobOffset = (bob ? 3.0 : 0.0) + 2.0 * mixPunch;
    const cy = h * 0.46 + bobOffset;

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
    // 1. KAWAII GUMMI BEAR DYNAMICS (SQUISH & STRETCH BOUNCE)
    // -------------------------------------------------------------------------
    const drumSquish = Math.min(1.2, env.drums * 1.15);
    const squishY = 1.0 - 0.06 * drumSquish - 0.025 * mixPunch + (bob ? -0.02 : 0.015);
    const squishX = 1.0 + 0.06 * drumSquish + 0.025 * mixPunch - (bob ? -0.02 : 0.015);
    const baseScale = (0.86 + 0.02 * this.stage) * (1.0 + 0.025 * mixEnv);
    const scaleX = baseScale * squishX;
    const scaleY = baseScale * squishY;

    // Rave halo behind the creature: stage-coloured bloom that breathes with
    // the true full mix, so all four stems add up to the body glow.
    ctx.save();
    const haloR = 104 * (1 + 0.11 * mixEnv + 0.05 * mixPunch);
    const halo = ctx.createRadialGradient(cx, cy + 4 * scaleY, 8, cx, cy + 4 * scaleY, haloR);
    halo.addColorStop(0.0, glowColor);
    halo.addColorStop(0.55, glowColor);
    halo.addColorStop(1.0, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = halo;
    ctx.globalAlpha = 0.07 + 0.24 * mixEnv + 0.08 * mixPunch;
    ctx.beginPath();
    ctx.arc(cx, cy + 4 * scaleY, haloR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();


    // -------------------------------------------------------------------------
    // 2. THE AUDIO REACTIVE RING LOOP (CREATURE ROUND ORB OUTLINE & WHOLE GUMMI BEAR BODY)
    // -------------------------------------------------------------------------
    const N = GUMMI_BEAR_LOOP_POINTS.length;
    const displacedPoints: { x: number; y: number }[] = new Array(N);

    for (let i = 0; i < N; i++) {
      const pt = GUMMI_BEAR_LOOP_POINTS[i];

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
      cx, cy + 20 * scaleY, 10,
      cx, cy + 20 * scaleY, 78 * baseScale
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
      const pt = GUMMI_BEAR_LOOP_POINTS[i];
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

    // 2C. Main Radiant Gummi Bear Perimeter (Neon Bloom Pass)
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
    // 3. CUTE INNER EAR CUSHIONS (AUDIO REACTIVE WIGGLE)
    // -------------------------------------------------------------------------
    ctx.save();
    const vocalPulse = 1 + 0.20 * Math.min(1.4, env.vocals * 1.5) + 0.10 * punch.vocals;
    const earPadW = 6.8 * baseScale * vocalPulse;
    const earPadH = 8.0 * baseScale * vocalPulse;
    const earWiggle = Math.sin(t * 14.0) * (env.vocals * 2.8 + punch.vocals * 1.5);

    // Left Inner Ear Pad
    ctx.save();
    ctx.translate(cx - 32 * scaleX, cy - 54 * scaleY + earWiggle);
    ctx.rotate(-0.25);
    ctx.beginPath();
    ctx.ellipse(0, 0, earPadW, earPadH, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 160, 210, 0.55)";
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#ff77e9";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 230, 245, 0.75)";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.restore();

    // Right Inner Ear Pad
    ctx.save();
    ctx.translate(cx + 32 * scaleX, cy - 54 * scaleY - earWiggle);
    ctx.rotate(0.25);
    ctx.beginPath();
    ctx.ellipse(0, 0, earPadW, earPadH, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 160, 210, 0.55)";
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#ff77e9";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 230, 245, 0.75)";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.restore();
    ctx.restore();

    // -------------------------------------------------------------------------
    // 4. RAVE HEART — the 'other' instrument (pads, chords, atmosphere)
    // -------------------------------------------------------------------------
    ctx.save();
    const bellyY = cy + 42 * scaleY - 10;
    const otherEnv = env.other;
    const otherPunch = punch.other;
    const beatPhase = this.heartBeatPhase;
    const beatPulse = 1.0 + 0.18 * otherEnv + 0.18 * otherPunch + 0.035 * Math.sin(beatPhase * Math.PI * 2);
    const heartW = 23 * scaleX * beatPulse;
    const heartH = 21 * scaleY * beatPulse;
    const heartWave = otherWaveform ?? null;
    // Warm gold pads -> pink -> cool violet as 'other' gets brighter.
    const otherHue = (45 - 135 * this.otherTone + 360) % 360;
    const heartGlow = `hsl(${otherHue.toFixed(0)}, 100%, 60%)`;
    const heartLine = `hsl(${otherHue.toFixed(0)}, 100%, 72%)`;
    const heartCore = `hsl(${otherHue.toFixed(0)}, 100%, 89%)`;

    const heartPathAt = (cw: number, ch: number): void => {
      const top = bellyY - ch * 0.55;
      const bottom = bellyY + ch * 0.62;
      const notch = bellyY - ch * 0.08;
      ctx.beginPath();
      ctx.moveTo(cx, bottom);
      ctx.bezierCurveTo(cx - cw * 1.14, bellyY + ch * 0.10, cx - cw * 0.72, top - ch * 0.22, cx, notch);
      ctx.bezierCurveTo(cx + cw * 0.72, top - ch * 0.22, cx + cw * 1.14, bellyY + ch * 0.10, cx, bottom);
      ctx.closePath();
    };

    // Expanding heartbeat ripples on every 'other' swell, not on the kick.
    for (let i = 2; i >= 0; i--) {
      const ph = (beatPhase + i / 3) % 1;
      const rippleScale = 1 + ph * 0.72;
      ctx.save();
      heartPathAt(heartW * rippleScale, heartH * rippleScale);
      ctx.strokeStyle = i === 0 ? heartCore : heartLine;
      ctx.globalAlpha = (1 - ph) * (0.12 + 0.58 * otherEnv + 0.35 * otherPunch);
      ctx.lineWidth = 1.1 + (1 - ph) * 1.8;
      ctx.shadowBlur = 10 + 16 * (1 - ph) * (0.5 + otherEnv);
      ctx.shadowColor = heartGlow;
      ctx.stroke();
      ctx.restore();
    }

    // Jelly heart body.
    heartPathAt(heartW, heartH);
    const bellyGrad = ctx.createRadialGradient(
      cx, bellyY - heartH * 0.10, 2,
      cx, bellyY, heartW * 1.05
    );
    if (this.stage === 4) {
      bellyGrad.addColorStop(0.0, "rgba(255, 252, 220, 0.62)");
      bellyGrad.addColorStop(0.55, "rgba(255, 210, 80, 0.42)");
      bellyGrad.addColorStop(1.0, "rgba(255, 120, 60, 0.20)");
    } else if (this.stage === 3) {
      bellyGrad.addColorStop(0.0, "rgba(235, 255, 255, 0.60)");
      bellyGrad.addColorStop(0.55, "rgba(120, 245, 255, 0.40)");
      bellyGrad.addColorStop(1.0, "rgba(60, 130, 255, 0.20)");
    } else if (this.stage === 2) {
      bellyGrad.addColorStop(0.0, "rgba(255, 245, 255, 0.60)");
      bellyGrad.addColorStop(0.55, "rgba(235, 130, 255, 0.40)");
      bellyGrad.addColorStop(1.0, "rgba(150, 40, 255, 0.20)");
    } else {
      bellyGrad.addColorStop(0.0, "rgba(255, 250, 254, 0.60)");
      bellyGrad.addColorStop(0.55, "rgba(255, 200, 233, 0.40)");
      bellyGrad.addColorStop(1.0, "rgba(255, 150, 215, 0.20)");
    }
    ctx.fillStyle = bellyGrad;
    ctx.shadowBlur = 14 + 20 * otherEnv + 12 * otherPunch;
    ctx.shadowColor = heartGlow;
    ctx.fill();

    // Inside the heart: live waveform + twinkling rave dust, clipped to the shape.
    ctx.save();
    heartPathAt(heartW, heartH);
    ctx.clip();
    if (heartWave && heartWave.length > 0) {
      ctx.beginPath();
      const steps = 56;
      for (let j = 0; j <= steps; j++) {
        const u = j / steps;
        const x = cx + (u - 0.5) * heartW * 2.0;
        const idx = Math.floor(u * (heartWave.length - 1));
        const amp = (heartWave[idx] - 128) / 128;
        const y = bellyY + 1.2 + amp * heartH * 0.62 * (0.45 + 0.85 * otherEnv);
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = heartCore;
      ctx.globalAlpha = 0.50 + 0.50 * otherEnv;
      ctx.lineWidth = 1.3;
      ctx.shadowBlur = 10 + 8 * otherPunch;
      ctx.shadowColor = heartGlow;
      ctx.stroke();
    }
    for (let k = 0; k < 9; k++) {
      const a = t * (0.6 + k * 0.11) + k * 2.1;
      const rr = 0.20 + 0.55 * (0.5 + 0.5 * Math.sin(t * 0.8 + k * 1.3));
      const sx = cx + Math.cos(a) * heartW * rr;
      const sy = bellyY + Math.sin(a * 1.3 + k) * heartH * rr;
      const tw = 0.3 + 0.7 * Math.pow(Math.abs(Math.sin(t * 2.7 + k * 1.9)), 2);
      ctx.beginPath();
      ctx.arc(sx, sy, (0.8 + 1.5 * tw) * (0.7 + 0.55 * otherEnv + 0.25 * otherPunch), 0, Math.PI * 2);
      ctx.fillStyle = k % 3 === 0 ? "#ffffff" : (k % 3 === 1 ? heartCore : heartLine);
      ctx.globalAlpha = tw * (0.30 + 0.70 * otherEnv);
      ctx.shadowBlur = 8;
      ctx.shadowColor = heartGlow;
      ctx.fill();
    }
    ctx.restore();

    // Glossy candy highlight on the upper-left lobe.
    ctx.save();
    ctx.translate(cx - heartW * 0.34, bellyY - heartH * 0.24);
    ctx.rotate(-0.5);
    ctx.beginPath();
    ctx.ellipse(0, 0, heartW * 0.20, heartH * 0.10, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.62)";
    ctx.shadowBlur = 6;
    ctx.shadowColor = "#ffffff";
    ctx.fill();
    ctx.restore();

    // Double neon outline: soft halo + crisp core.
    heartPathAt(heartW, heartH);
    ctx.strokeStyle = heartLine;
    ctx.lineWidth = 3.4;
    ctx.globalAlpha = 0.50 + 0.38 * otherEnv + 0.12 * otherPunch;
    ctx.shadowBlur = 18 + 24 * otherEnv + 14 * otherPunch;
    ctx.shadowColor = heartGlow;
    ctx.stroke();
    heartPathAt(heartW, heartH);
    ctx.strokeStyle = heartCore;
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.92;
    ctx.shadowBlur = 7;
    ctx.shadowColor = "#fff3b0";
    ctx.stroke();
    ctx.restore();

    // -------------------------------------------------------------------------
    // 5. KAWAII PAW BEANS (PINK JELLY PADS ON HANDS & FEET)
    // -------------------------------------------------------------------------
    ctx.save();
    const drawPaw = (pawX: number, pawY: number, angle: number, isHand: boolean) => {
      ctx.save();
      ctx.translate(pawX, pawY);
      ctx.rotate(angle);
      ctx.fillStyle = "rgba(255, 150, 210, 0.65)";
      ctx.shadowBlur = 6;
      ctx.shadowColor = "#ff77e9";

      // Main palm/foot pad
      const padR = (isHand ? 3.5 : 4.2) * baseScale * (1 + 0.18 * Math.min(1.4, env.bass * 1.5 + punch.bass));
      ctx.beginPath();
      ctx.arc(0, 0, padR, 0, Math.PI * 2);
      ctx.fill();

      // 3 Cute tiny toe beans
      const beanR = (isHand ? 1.6 : 2.0) * baseScale;
      const beanDist = padR + beanR + 1.2;
      for (let b = -1; b <= 1; b++) {
        const bAng = b * 0.45 - Math.PI / 2;
        const bx = beanDist * Math.cos(bAng);
        const by = beanDist * Math.sin(bAng);
        ctx.beginPath();
        ctx.arc(bx, by, beanR, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    };

    // Left & Right Hand Paws
    drawPaw(cx - 48 * scaleX, cy + 24 * scaleY, -0.65, true);
    drawPaw(cx + 48 * scaleX, cy + 24 * scaleY, 0.65, true);

    // Left & Right Foot Paws
    drawPaw(cx - 18 * scaleX, cy + 70 * scaleY, 0.15, false);
    drawPaw(cx + 18 * scaleX, cy + 70 * scaleY, -0.15, false);
    ctx.restore();

    // 5B. Floating rave motes orbiting the creature (full-mix shimmer).
    ctx.save();
    for (let k = 0; k < 7; k++) {
      const a = t * (0.32 + k * 0.035) + k * (Math.PI * 2 / 7);
      const rr = 0.92 + 0.10 * Math.sin(t * 0.7 + k * 1.3);
      const mx = cx + Math.cos(a) * 64 * scaleX * rr;
      const my = cy + Math.sin(a) * 78 * scaleY * rr;
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

    // Forehead shine arc
    ctx.beginPath();
    ctx.arc(cx - 12 * scaleX, cy - 35 * scaleY, 14 * scaleX, 1.15 * Math.PI, 1.65 * Math.PI);
    ctx.stroke();

    // Ear gloss dots
    ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
    ctx.beginPath();
    ctx.arc(cx - 36 * scaleX, cy - 60 * scaleY, 2.2, 0, Math.PI * 2);
    ctx.arc(cx + 36 * scaleX, cy - 60 * scaleY, 2.2, 0, Math.PI * 2);
    ctx.fill();

    // Chubby cheek gloss pills
    ctx.beginPath();
    ctx.arc(cx - 34 * scaleX, cy - 14 * scaleY, 2.0, 0, Math.PI * 2);
    ctx.arc(cx + 34 * scaleX, cy - 14 * scaleY, 2.0, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // -------------------------------------------------------------------------
    // 8. ROSY CHEEKS & KAWAII BLUSH SLASHES (///)
    // -------------------------------------------------------------------------
    const cheekSpacing = 28 * scaleX;
    const cheekY = cy - 6 * scaleY;
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
    const eyeY = cy - 18 * scaleY;
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
    const muzzleY = cy - 4 * scaleY;
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
    const noseY = cy - 7 * scaleY;
    this.drawKawaiiBearNose(
      ctx, cx, noseY,
      env.drums, env.bass, env.vocals, mixEnv,
      isSinging, waveform, accentColor, baseScale
    );

    // -------------------------------------------------------------------------
    // 12. THE LIVE VOCAL WAVEFORM AUDIO REACTIVE MOUTH
    // -------------------------------------------------------------------------
    const mouthY = cy + 1 * scaleY;
    const mouthAperture = Math.max(0, Math.min(1,
      Math.max(this.smoothedVocalAmp, this.stemEnv.vocals) * 2.1 + punch.vocals * 0.35));

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const mouthSteps = 64;
    const mouthColor = isSinging ? "#ffff00" : "#ff0077";
    const mouthGlow = isSinging ? "#ffea00" : "#ff77e9";

    ctx.shadowBlur = isSinging ? (14 + 18 * mouthAperture) : 8;
    ctx.shadowColor = mouthGlow;
    ctx.strokeStyle = mouthColor;
    ctx.fillStyle = isSinging ? "rgba(255, 234, 0, 0.22)" : "rgba(255, 105, 180, 0.15)";
    ctx.lineWidth = 2.4 + 1.4 * mouthAperture + 1.2 * punch.vocals;

    ctx.beginPath();
    if (!isSinging && this.stemEnv.vocals < 0.02) {
      // Resting cute kawaii bear smile ( ᵕ ‿ ᵕ ) / w-shape
      const breathe = Math.sin(t * 2.5) * 1.0;
      const rx = 6.5 + breathe;
      const ry = 3.2 + breathe * 0.4;
      for (let j = 0; j <= mouthSteps; j++) {
        const phi = (j / mouthSteps) * Math.PI * 2;
        const mx = cx + rx * Math.cos(phi);
        const my = mouthY + ry * Math.sin(phi) + 1.2 * Math.pow(Math.cos(phi), 2);
        if (j === 0) ctx.moveTo(mx, my);
        else ctx.lineTo(mx, my);
      }
    } else {
      // Live Audio Waveform Voice Instrument Singing Mini Loop!
      let baseW = 12;
      let baseH = 10;
      const upVowel = (vowel || "AH").toUpperCase();
      if (upVowel === "EE") { baseW = 15; baseH = 5; }
      else if (upVowel === "OH") { baseW = 9; baseH = 14; }
      else if (upVowel === "OO") { baseW = 7; baseH = 7; }
      else if (upVowel === "AY") { baseW = 14; baseH = 9; }

      const curW = baseW * (0.6 + 0.9 * mouthAperture);
      const curH = baseH * (0.6 + 1.1 * mouthAperture);

      const waveLen = waveform ? waveform.length : 0;
      for (let j = 0; j <= mouthSteps; j++) {
        const phi = (j / mouthSteps) * Math.PI * 2;
        let sampleNorm = 0;
        if (waveform && waveLen > 0) {
          const wIdx = Math.floor((j / mouthSteps) * (waveLen - 1));
          sampleNorm = (waveform[wIdx] - 128) / 128.0;
        } else {
          sampleNorm = Math.sin(phi * 3.0 + t * 14.0) * 0.4;
        }

        const rad = Math.sqrt(Math.pow(curW * Math.cos(phi), 2) + Math.pow(curH * Math.sin(phi), 2));
        const waveDisp = sampleNorm * (5.5 * mouthAperture + 1.2 + 2.5 * punch.vocals);
        const mx = cx + (rad + waveDisp * 0.7) * Math.cos(phi);
        const my = mouthY + (rad + waveDisp) * Math.sin(phi);

        if (j === 0) ctx.moveTo(mx, my);
        else ctx.lineTo(mx, my);
      }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
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
