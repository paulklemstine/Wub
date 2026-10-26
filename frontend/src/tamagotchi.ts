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
  1: { name: "RAVE KITTEN", minXp: 0 },
  2: { name: "DISCO CAT", minXp: 250 },
  3: { name: "CYBER NEKO", minXp: 750 },
  4: { name: "ASTRAL NEKO", minXp: 1800 },
};

const MUSIC_NOTES = ["✨", "💖", "🐾", "🌈", "✦", "✧", "💫", "♬", "⭐", "♡"];

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
  private speechTimer: number = 0;
  private lastStepIndex: number = -1;
  private lastBarNum: number = -1;
  private temporaryAction: string | null = null;
  private actionTimer: number = 0;

  // Vocal Waveform & Mathematical Singing Mouth State
  private smoothedVocalAmp: number = 0;
  private singingLoopPhase: number = 0;

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
    this.speechBubble = "💖 NEIGH! LOVE THIS GROOVE ✨";
    this.speechTimer = 2.5;
    this.saveState();
  }

  public feed(): void {
    this.addXp(25);
    this.vibe = Math.min(100, this.vibe + 15);
    this.temporaryAction = "EAT";
    this.actionTimer = 2.2;
    this.speechBubble = "🦄 NOM NOM! SPARKLE AUDIO SLICE!";
    this.speechTimer = 2.5;
    this.saveState();
  }

  public hype(): void {
    this.addXp(20);
    this.vibe = 100;
    this.temporaryAction = "HYPE";
    this.actionTimer = 3.0;
    this.speechBubble = "⚡ PSYCHEDELIC BASS DROP! 🌈";
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
      this.speechBubble = `✨ EVOLVED TO ${STAGE_CONFIGS[this.stage].name}! 🦄🌈`;
      this.speechTimer = 3.5;
      this.saveState();
    }
  }

  // ---------------------------------------------------------------------------
  // Step & Audio Frame Update
  // ---------------------------------------------------------------------------

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
    otherLevel?: number
  ): void {
    const dt = this.lastRenderSec > 0 ? Math.max(0.001, audioNow - this.lastRenderSec) : 0.016;
    this.lastRenderSec = audioNow;

    // Real-time Audio and Vocal Reactivity for Electric Rave Glow
    if (this.asciiEl) {
      this.asciiEl.style.setProperty("--audio-lvl", rmsLevel.toFixed(3));
      this.asciiEl.style.setProperty("--vocal-lvl", (vocalsLevel ?? 0).toFixed(3));
      const screenEl = this.asciiEl.parentElement;
      if (screenEl) {
        screenEl.style.setProperty("--audio-lvl", rmsLevel.toFixed(3));
        screenEl.style.setProperty("--vocal-lvl", (vocalsLevel ?? 0).toFixed(3));
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
      vocalsLevel ?? 0,
      rmsLevel,
      drumWaveform,
      bassWaveform,
      drumLevel ?? 0,
      bassLevel ?? 0,
      otherWaveform,
      otherLevel ?? 0
    );
    this.renderASCII(meta, isDropGap, waveform, dt, bpm, vocalsLevel);
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

  private renderASCII(
    meta: GrooveboxMeta | null,
    isDropGap: boolean,
    waveform?: Uint8Array | null,
    dt: number = 0.016,
    bpm: number = 126,
    vocalsLevel?: number
  ): void {
    if (!this.asciiEl) return;

    const sec = meta?.section ?? "GROOVE_A";
    const vowel = meta?.vocalVowel ?? "";
    const isSinging = Boolean(meta?.vocalSyllable);
    const bob = this.isHeadBobbed;

    // Toggle glowing psychedelic singing shimmer
    if (isSinging) {
      this.asciiEl.classList.add("singing");
    } else {
      this.asciiEl.classList.remove("singing");
    }

    // Determine Eyes (seductive starry unicorn expressions)
    let eyes = "✧   ✧";
    if (this.isBlinking) {
      eyes = "˘   ˘";
    } else if (isDropGap) {
      eyes = "O   O";
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

    // Determine Live Animated Waveform Singing Mouth!
    const mouth = this.extractWaveformMouth(waveform, isSinging, vowel, isDropGap, dt, bpm, vocalsLevel);

    // Render Neko / Cat Stages without shifting borders
    let faceLines: string[] = [];
    const noteChar = this.noteFloating.length > 0 ? this.noteFloating[0].char : "✨";

    if (this.stage === 1) {
      // Stage 1: Rave Kitten (cat ears, sparkly star gaze, sweet smile)
      faceLines = bob ? [
        `       <span class="rave-ears">/\\___/\\</span>       `,
        `      (  <span class="rave-eyes">${eyes}</span>  )   <span class="rave-spark">${noteChar}</span>  `,
        `     (  <span class="rave-mouth">${mouth}</span>  )  <span class="rave-spark">~♡</span> `,
        `      <span class="rave-body">/     \\</span>        `,
      ] : [
        `       <span class="rave-ears">/\\___/\\</span>       `,
        `      (  <span class="rave-eyes">${eyes}</span>  )   <span class="rave-spark">${noteChar}</span>  `,
        `     (  <span class="rave-mouth">${mouth}</span>  ) <span class="rave-spark">~~*</span> `,
        `      <span class="rave-body">|     |</span>        `,
      ];
    } else if (this.stage === 2) {
      // Stage 2: Disco Cat (rave shades/headphones, perky cat ears, rainbow vibes)
      faceLines = bob ? [
        `       <span class="rave-ears">/\\___/\\</span>       `,
        `    <span class="rave-wings">[o(</span> <span class="rave-eyes">⌐■_■</span> <span class="rave-wings">)o]</span>  <span class="rave-spark">${noteChar}</span> `,
        `     (  <span class="rave-mouth">${mouth}</span>  )  <span class="rave-spark">~🌈</span> `,
        `      <span class="rave-body">/| WUB |\\</span>      `,
      ] : [
        `       <span class="rave-ears">/\\___/\\</span>       `,
        `    <span class="rave-wings">[o(</span> <span class="rave-eyes">⌐■_■</span> <span class="rave-wings">)o]</span>  <span class="rave-spark">${noteChar}</span> `,
        `     (  <span class="rave-mouth">${mouth}</span>  ) <span class="rave-spark">~~✨</span> `,
        `      <span class="rave-body">\\| WUB |/</span>      `,
      ];
    } else if (this.stage === 3) {
      // Stage 3: Cyber Neko (sleek cyber lightning ears, electric wings, star gaze)
      faceLines = bob ? [
        `       <span class="rave-ears">/\\_⚡_/\\</span>       `,
        `    <span class="rave-wings">~/(\\</span> (<span class="rave-eyes">★   ★</span>) <span class="rave-wings">/\\)~</span> <span class="rave-spark">${noteChar}</span>`,
        `     <span class="rave-body">&lt;[~</span> <span class="rave-mouth">${mouth}</span> <span class="rave-body">~]&gt;</span>   `,
        `      <span class="rave-body">/  | WUB |  \\</span>    `,
        `     <span class="rave-body">(____/   \\____)</span>   `,
      ] : [
        `       <span class="rave-ears">/\\_⚡_/\\</span>       `,
        `    <span class="rave-wings">~/\\(</span> (<span class="rave-eyes">★   ★</span>) <span class="rave-wings">/\\)~</span> <span class="rave-spark">${noteChar}</span>`,
        `     <span class="rave-body">\\[~</span> <span class="rave-mouth">${mouth}</span> <span class="rave-body">~]/</span>   `,
        `      <span class="rave-body">|  | WUB |  |</span>    `,
        `     <span class="rave-body">(____/   \\____)</span>   `,
      ];
    } else {
      // Stage 4: Astral Neko (celestial glowing ears, nebula wings, divine sparkle)
      faceLines = bob ? [
        `      . <span class="rave-ears">/\\✦/\\</span> .       `,
        `   <span class="rave-wings">~/{\\(</span> (<span class="rave-eyes">♥   ♥</span>) <span class="rave-wings">/\\}~</span> <span class="rave-spark">${noteChar}</span>`,
        `    <span class="rave-body">&lt;[~</span> ✧<span class="rave-mouth">${mouth}</span>✧ <span class="rave-body">~]&gt;</span>   `,
        `     <span class="rave-body">/    | 505 |    \\</span>   `,
        `    <span class="rave-body">(____/       \\____)</span>  `,
      ] : [
        `      . <span class="rave-ears">/\\✦/\\</span> .       `,
        `   <span class="rave-wings">~/{\\(</span> (<span class="rave-eyes">♥   ♥</span>) <span class="rave-wings">/\\}~</span> <span class="rave-spark">${noteChar}</span>`,
        `    <span class="rave-body">\\[~</span> ✧<span class="rave-mouth">${mouth}</span>✧ <span class="rave-body">~]/</span>   `,
        `     <span class="rave-body">|    | 505 |    |</span>   `,
        `    <span class="rave-body">(____/       \\____)</span>  `,
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
    vocalsLevel: number = 0,
    rmsLevel: number = 0,
    drumWaveform?: Uint8Array | null,
    bassWaveform?: Uint8Array | null,
    drumLevel: number = 0,
    bassLevel: number = 0,
    otherWaveform?: Uint8Array | null,
    otherLevel: number = 0
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
    const isSinging = Boolean(meta?.vocalSyllable) || (vocalsLevel >= 0.008);
    const bob = this.isHeadBobbed;
    const t = performance.now() / 1000;

    // Center coordinates with beat bobbing
    const cx = w * 0.5;
    const bobOffset = bob ? 4.0 : 0.0;
    const cy = h * 0.48 + bobOffset;

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

    // Circular angular distance with full 2*PI wrap-around
    const angDist = (a: number, b: number): number => {
      let diff = Math.abs(a - b) % (Math.PI * 2);
      if (diff > Math.PI) diff = Math.PI * 2 - diff;
      return diff;
    };

    // -------------------------------------------------------------------------
    // 1. THE AUDIO REACTIVE SINGLE LINE LOOP (CREATURE HEAD & CAT EARS SILHOUETTE)
    // -------------------------------------------------------------------------
    const N = 180;
    const baseR = 44 + 4 * this.stage;
    const audioExpansion = 1.0 + 0.20 * Math.min(1.5, rmsLevel);

    // Mirrored Cat Ears across the vertical axis (-0.5 * Math.PI)
    // Symmetrical offsets wrapped with circular angular distance (2*PI)
    const earOffset = 0.20 * Math.PI; // ~36 degrees offset each side
    const leftEarAngle = -0.5 * Math.PI - earOffset;  // ~ -0.70 * PI (left cat ear)
    const rightEarAngle = -0.5 * Math.PI + earOffset; // ~ -0.30 * PI (right cat ear)

    // Audio-reactive ear twitching to 'other' instrument (synths/chords/leads)
    let otherTwitch = 0;
    if (otherWaveform && otherWaveform.length > 0) {
      const oDev = (otherWaveform[0] - 128) / 128.0;
      otherTwitch = Math.abs(oDev) * 0.22;
    }
    const earHeight = (0.85 + 0.20 * this.stage) * (1.0 + otherLevel * 0.40 + otherTwitch);
    const earWidth = 0.082;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const theta = (i / N) * Math.PI * 2 - Math.PI / 2; // -pi/2 is top center (vertical axis)

      // Cute pear/kawaii head outline
      const headProfile = 1.0 + 0.12 * Math.sin(theta);

      const dLeftEar = angDist(theta, leftEarAngle);
      const dRightEar = angDist(theta, rightEarAngle);

      // Sharp pointed cat ear peaks
      const leftEarPeak = earHeight * Math.exp(-Math.pow(dLeftEar / earWidth, 2));
      const rightEarPeak = earHeight * Math.exp(-Math.pow(dRightEar / earWidth, 2));
      const catEars = leftEarPeak + rightEarPeak;

      // Forehead dip between the ears at the vertical axis
      const dCenter = angDist(theta, -0.5 * Math.PI);
      const foreheadDip = -0.06 * Math.exp(-Math.pow(dCenter / 0.10, 2));

      // Audio reactive standing waves rippling around the perimeter
      const waveRipple = (
        Math.sin(5 * theta - t * 4.2) * 3.6 +
        Math.cos(9 * theta + t * 6.5) * 2.2
      ) * Math.min(1.0, rmsLevel * 2.0);

      const r = (baseR * (headProfile + catEars + foreheadDip)) * audioExpansion + waveRipple;
      const px = cx + r * Math.cos(theta);
      const py = cy + r * Math.sin(theta);

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();

    // Outer neon bloom pass
    ctx.shadowBlur = 14 + 18 * rmsLevel;
    ctx.shadowColor = glowColor;
    ctx.strokeStyle = glowColor;
    ctx.lineWidth = 4.2;
    ctx.globalAlpha = 0.40;
    ctx.stroke();

    // Inner crisp laser vector pass
    ctx.shadowBlur = 6;
    ctx.shadowColor = "#ffffff";
    ctx.strokeStyle = mainColor;
    ctx.lineWidth = 2.2;
    ctx.globalAlpha = 1.0;
    ctx.stroke();
    ctx.restore();

    // 1.5. HARMONIC SYNTH CROWN / AURA (Vibrating to Other instrument)
    if (otherLevel > 0.015 || (otherWaveform && otherWaveform.length > 0)) {
      ctx.save();
      ctx.strokeStyle = "#ff007f"; // Vibrant neon magenta for 'other' instrument
      ctx.shadowBlur = 10 + 12 * otherLevel;
      ctx.shadowColor = "#ff00a0";
      ctx.lineWidth = 1.8;
      ctx.globalAlpha = 0.55 + 0.45 * Math.min(1.0, otherLevel * 1.8);
      ctx.beginPath();
      const haloR = baseR * 0.62;
      const haloY = cy - baseR * 0.82;
      const oLen = otherWaveform ? otherWaveform.length : 0;
      for (let j = 0; j <= 32; j++) {
        const phi = Math.PI + (j / 32) * Math.PI; // semi-circle arc above forehead
        let oNorm = 0;
        if (oLen > 0 && otherWaveform) {
          const idx = Math.floor((j / 32) * (oLen - 1));
          oNorm = (otherWaveform[idx] - 128) / 128.0;
        }
        const hR = haloR + oNorm * 6.0;
        const hx = cx + hR * Math.cos(phi);
        const hy = haloY + hR * Math.sin(phi) * 0.45;
        if (j === 0) ctx.moveTo(hx, hy);
        else ctx.lineTo(hx, hy);
      }
      ctx.stroke();
      ctx.restore();
    }

    // -------------------------------------------------------------------------
    // 2. EXPRESSIVE FACIAL FEATURES
    // -------------------------------------------------------------------------
    const eyeSpacing = 22 + 2 * this.stage;
    const eyeY = cy - 4;
    const leftEyeX = cx - eyeSpacing;
    const rightEyeX = cx + eyeSpacing;

    // Cheeks subtle neon blush
    ctx.save();
    ctx.fillStyle = "rgba(255, 105, 180, 0.28)";
    ctx.beginPath();
    ctx.arc(leftEyeX - 10, eyeY + 14, 6.5, 0, Math.PI * 2);
    ctx.arc(rightEyeX + 10, eyeY + 14, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Glowing Cyber Whiskers (Audio Reactive Twitching Whisker Vectors)
    ctx.save();
    ctx.strokeStyle = accentColor;
    ctx.shadowBlur = 8;
    ctx.shadowColor = accentColor;
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.65 + 0.35 * Math.min(1.0, rmsLevel);

    const whiskerTwitch = Math.sin(t * 8.0) * (1.0 + drumLevel * 2.5);
    const wSpan = 18 + 4 * this.stage;

    // Left Whiskers (fanning outward left)
    const lCheekX = leftEyeX - 12;
    const lCheekY = eyeY + 14;
    ctx.beginPath();
    ctx.moveTo(lCheekX, lCheekY - 3);
    ctx.lineTo(lCheekX - wSpan, lCheekY - 7 + whiskerTwitch);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(lCheekX - 2, lCheekY + 1);
    ctx.lineTo(lCheekX - wSpan - 3, lCheekY + 1 + whiskerTwitch * 0.5);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(lCheekX, lCheekY + 5);
    ctx.lineTo(lCheekX - wSpan, lCheekY + 8 - whiskerTwitch * 0.6);
    ctx.stroke();

    // Right Whiskers (fanning outward right, mirrored)
    const rCheekX = rightEyeX + 12;
    const rCheekY = eyeY + 14;
    ctx.beginPath();
    ctx.moveTo(rCheekX, rCheekY - 3);
    ctx.lineTo(rCheekX + wSpan, rCheekY - 7 + whiskerTwitch);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(rCheekX + 2, rCheekY + 1);
    ctx.lineTo(rCheekX + wSpan + 3, rCheekY + 1 + whiskerTwitch * 0.5);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(rCheekX, rCheekY + 5);
    ctx.lineTo(rCheekX + wSpan, rCheekY + 8 - whiskerTwitch * 0.6);
    ctx.stroke();
    ctx.restore();

    // Eyes rendering
    ctx.save();
    ctx.shadowBlur = 10;
    ctx.shadowColor = accentColor;
    ctx.strokeStyle = accentColor;
    ctx.fillStyle = accentColor;
    ctx.lineWidth = 2.2;

    if (this.isBlinking) {
      ctx.shadowBlur = 8;
      ctx.shadowColor = accentColor;
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(leftEyeX, eyeY, 6, 0.2 * Math.PI, 0.8 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(rightEyeX, eyeY, 6, 0.2 * Math.PI, 0.8 * Math.PI);
      ctx.stroke();
    } else if (this.temporaryAction === "PET") {
      ctx.shadowBlur = 10;
      ctx.shadowColor = "#ff69b4";
      ctx.fillStyle = "#ff69b4";
      this.drawHeart(ctx, leftEyeX, eyeY, 6.5);
      this.drawHeart(ctx, rightEyeX, eyeY, 6.5);
    } else {
      // Audio Reactive Dual Concentric Waveform Eyes:
      // Inner Drum Pupil Ring vibrating inside Outer Bass Iris Ring!
      this.drawWaveformEye(ctx, leftEyeX, eyeY, drumWaveform, bassWaveform, drumLevel, bassLevel, sec === "DROP", isDropGap);
      this.drawWaveformEye(ctx, rightEyeX, eyeY, drumWaveform, bassWaveform, drumLevel, bassLevel, sec === "DROP", isDropGap);
    }
    ctx.restore();

    // -------------------------------------------------------------------------
    // 3. THE MINI LOOP AUDIO REACTIVE MOUTH (SINGS LIVE VOCAL WAVEFORM)
    // -------------------------------------------------------------------------
    const mouthY = cy + 22;
    const mouthAperture = Math.max(0, Math.min(1, this.smoothedVocalAmp * 2.2));

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const mouthSteps = 64;
    const mouthColor = isSinging ? "#ffff00" : "#00ff88";
    const mouthGlow = isSinging ? "#ffea00" : "#00ffcc";

    ctx.shadowBlur = isSinging ? (14 + 18 * mouthAperture) : 8;
    ctx.shadowColor = mouthGlow;
    ctx.strokeStyle = mouthColor;
    ctx.fillStyle = isSinging ? "rgba(255, 234, 0, 0.15)" : "transparent";
    ctx.lineWidth = 2.4;

    ctx.beginPath();
    if (!isSinging && vocalsLevel < 0.006) {
      // Resting / Breathing cute smiling mini loop
      const breathe = Math.sin(t * 2.5) * 1.2;
      const rx = 10 + breathe;
      const ry = 4 + breathe * 0.5;
      for (let j = 0; j <= mouthSteps; j++) {
        const phi = (j / mouthSteps) * Math.PI * 2;
        const mx = cx + rx * Math.cos(phi);
        const my = mouthY + ry * Math.sin(phi) + 1.8 * Math.pow(Math.cos(phi), 2);
        if (j === 0) ctx.moveTo(mx, my);
        else ctx.lineTo(mx, my);
      }
    } else {
      // Live Audio Waveform Voice Instrument Singing Mini Loop!
      let baseW = 14;
      let baseH = 12;
      const upVowel = (vowel || "AH").toUpperCase();
      if (upVowel === "EE") { baseW = 18; baseH = 6; }
      else if (upVowel === "OH") { baseW = 11; baseH = 16; }
      else if (upVowel === "OO") { baseW = 8; baseH = 8; }
      else if (upVowel === "AY") { baseW = 16; baseH = 10; }

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
        const waveDisp = sampleNorm * (6.0 * mouthAperture + 1.5);
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

  private drawSparkleEye(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number): void {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx - r * 0.3, cy - r * 0.3, r * 0.35, 0, Math.PI * 2);
    ctx.fill();
  }

  private drawWaveformEye(
    ctx: CanvasRenderingContext2D,
    eyeX: number,
    eyeY: number,
    drumWaveform?: Uint8Array | null,
    bassWaveform?: Uint8Array | null,
    drumLevel: number = 0,
    bassLevel: number = 0,
    isDrop: boolean = false,
    isDropGap: boolean = false
  ): void {
    const eyeSteps = 32;
    const t = performance.now() / 1000;

    // -------------------------------------------------------------------------
    // 1. OUTER RING: BASS IRIS WAVEFORM
    // -------------------------------------------------------------------------
    const baseIrisR = (isDropGap ? 11.5 : (isDrop ? 9.5 : 8.2)) + Math.min(3.2, bassLevel * 4.5);
    const bassWaveLen = bassWaveform ? bassWaveform.length : 0;
    const irisStroke = isDrop ? "#00ffff" : "#00ffcc";
    const irisGlow = isDrop ? "#00e5ff" : "#00ffaa";

    ctx.save();
    ctx.shadowBlur = isDrop ? 12 : 8;
    ctx.shadowColor = irisGlow;
    ctx.strokeStyle = irisStroke;
    ctx.fillStyle = "rgba(0, 255, 204, 0.14)";
    ctx.lineWidth = 1.9;

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
      const disp = bSample * (2.8 * Math.min(1.5, bassLevel * 2.5) + (isDropGap ? 1.5 : 0.7));
      const r = Math.max(2.0, baseIrisR + disp);
      const x = eyeX + r * Math.cos(phi);
      const y = eyeY + r * Math.sin(phi);
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // -------------------------------------------------------------------------
    // 2. INNER RING: DRUM PUPIL WAVEFORM OUTLINED INSIDE BASS IRIS
    // -------------------------------------------------------------------------
    const basePupilR = (isDropGap ? 6.5 : (isDrop ? 4.8 : 3.8)) + Math.min(2.5, drumLevel * 3.8);
    const drumWaveLen = drumWaveform ? drumWaveform.length : 0;
    const pupilStroke = isDrop ? "#ffffff" : "#ffe600";
    const pupilGlow = isDrop ? "#ffffff" : "#ffbb00";

    ctx.save();
    ctx.shadowBlur = 7;
    ctx.shadowColor = pupilGlow;
    ctx.strokeStyle = pupilStroke;
    ctx.fillStyle = "rgba(10, 10, 22, 0.88)";
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
      const disp = dSample * (2.2 * Math.min(1.5, drumLevel * 2.5) + (isDropGap ? 1.2 : 0.5));
      const r = Math.max(1.2, basePupilR + disp);
      const x = eyeX + r * Math.cos(phi);
      const y = eyeY + r * Math.sin(phi);
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Sharp pinpoint glint highlight
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(eyeX - basePupilR * 0.35, eyeY - basePupilR * 0.35, 1.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
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
