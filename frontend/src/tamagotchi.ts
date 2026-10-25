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
  1: { name: "BABY UNICORN", minXp: 0 },
  2: { name: "DISCO UNICORN", minXp: 250 },
  3: { name: "CYBER PEGASUS", minXp: 750 },
  4: { name: "ASTRAL GODDESS", minXp: 1800 },
};

const MUSIC_NOTES = ["✨", "💖", "🦄", "🌈", "✦", "✧", "💫", "♬", "⭐", "♡"];

export class TamagotchiAvatar {
  private containerEl: HTMLElement | null = null;
  private asciiEl: HTMLElement | null = null;
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
    waveform?: Uint8Array | null
  ): void {
    const dt = this.lastRenderSec > 0 ? Math.max(0.001, audioNow - this.lastRenderSec) : 0.016;
    this.lastRenderSec = audioNow;

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
          x: Math.floor(Math.random() * 4) + (Math.random() > 0.5 ? 1 : 24),
          y: 6,
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
      n.life -= dt * 1.2;
      n.y -= dt * 2.0;
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

    this.renderASCII(meta, isDropGap, waveform);
    this.updateDashboard();
  }

  // ---------------------------------------------------------------------------
  // Live Waveform Animated Singing Mouth
  // ---------------------------------------------------------------------------

  private extractWaveformMouth(
    waveform: Uint8Array | null | undefined,
    isSinging: boolean,
    vowel: string,
    isDropGap: boolean
  ): string {
    if (isDropGap) return " O ";

    const WAVE_CHARS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

    if (!waveform || waveform.length === 0) {
      if (isSinging) {
        if (vowel === "AH") return " ▃█▃ ";
        if (vowel === "EE") return " ▄▇▄ ";
        if (vowel === "OH") return " ▅█▅ ";
        if (vowel === "OO") return "  █  ";
        return " ▃▅▃ ";
      }
      return " ‿ ";
    }

    // Sample 5 points across the center of the waveform buffer
    const len = waveform.length;
    const startIdx = Math.floor(len * 0.25);
    const endIdx = Math.floor(len * 0.75);
    const step = Math.floor((endIdx - startIdx) / 4);

    let maxAmp = 0;
    const amps: number[] = [];
    for (let i = 0; i < 5; i++) {
      const idx = Math.min(len - 1, startIdx + i * step);
      const val = Math.abs(waveform[idx] - 128) / 128; // 0..1
      amps.push(val);
      if (val > maxAmp) maxAmp = val;
    }

    // If completely quiet / resting
    if (maxAmp < 0.030) {
      return isSinging ? " ‿ " : " · ";
    }

    // Boost sensitivity for animated singing visibility
    const boost = isSinging ? 3.4 : 2.6;
    const chars = amps.map((a, i) => {
      let mod = a * boost;
      if (vowel === "AH" || vowel === "OH") {
        if (i === 2) mod *= 1.35; // taller center mouth opening
      } else if (vowel === "EE") {
        mod *= 0.95; // wide teeth spread
      }
      const charIdx = Math.min(WAVE_CHARS.length - 1, Math.floor(mod * (WAVE_CHARS.length - 1)));
      return WAVE_CHARS[Math.max(1, charIdx)];
    });

    return chars.join("");
  }

  // ---------------------------------------------------------------------------
  // ASCII Rendering Engine
  // ---------------------------------------------------------------------------

  private renderASCII(meta: GrooveboxMeta | null, isDropGap: boolean, waveform?: Uint8Array | null): void {
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

    // Determine Live Animated Waveform Mouth!
    const mouth = this.extractWaveformMouth(waveform, isSinging, vowel, isDropGap);

    // Render Unicorn Stages without shifting borders
    let faceLines: string[] = [];
    const noteChar = this.noteFloating.length > 0 ? this.noteFloating[0].char : "✨";

    if (this.stage === 1) {
      // Stage 1: Baby Unicorn (spiral horn, sparkly star gaze, sweet smile)
      faceLines = bob ? [
        `       . \\▲/ .       `,
        `      (\\_/)   ${noteChar}    `,
        `     ( ${eyes} )     `,
        `     (  ${mouth}  ) ~♡  `,
        `      /     \\       `,
      ] : [
        `       . \\▲/ .       `,
        `      (\\_/)   ${noteChar}    `,
        `     ( ${eyes} )     `,
        `     (  ${mouth}  ) ~~* `,
        `      |     |       `,
      ];
    } else if (this.stage === 2) {
      // Stage 2: Disco Unicorn (rave shades/headphones, glitter horn, rainbow mane)
      faceLines = bob ? [
        `      . : \\▲/ : .     `,
        `     (\\\\___/)   ${noteChar}  `,
        `    [o( ⌐■_■ )o]    `,
        `     (  ${mouth}  ) ~🌈 `,
        `      /| WUB |\\     `,
      ] : [
        `      . : \\▲/ : .     `,
        `     (\\\\___/)   ${noteChar}  `,
        `    [o( ⌐■_■ )o]    `,
        `     (  ${mouth}  ) ~~✨`,
        `      \\| WUB |/     `,
      ];
    } else if (this.stage === 3) {
      // Stage 3: Cyber Pegasus (sleek cyber horn, electric wings, star gaze)
      faceLines = bob ? [
        `      *   \\⚡/   *     `,
        `    ~/(\\ (★   ★) /\\)~ ${noteChar}`,
        `     <[~  ${mouth}  ~]>   `,
        `      /  | WUB |  \\    `,
        `     (____/   \\____)   `,
      ] : [
        `      *   \\⚡/   *     `,
        `    ~/\\( (★   ★) /\\)~ ${noteChar}`,
        `     \\[~  ${mouth}  ~]/   `,
        `      |  | WUB |  |    `,
        `     (____/   \\____)   `,
      ];
    } else {
      // Stage 4: Astral Goddess (celestial glowing horn, nebula wings, divine sparkle)
      faceLines = bob ? [
        `     . : * \\✦/ * : .   `,
        `   ~/{\\( (♥   ♥) /\\}~ ${noteChar}`,
        `    <[~ ✧${mouth}✧ ~]>   `,
        `     /    | 505 |    \\   `,
        `    (____/       \\____)  `,
      ] : [
        `     . : * \\✦/ * : .   `,
        `   ~/{\\( (♥   ♥) /\\}~ ${noteChar}`,
        `    \\[~ ✧${mouth}✧ ~]/   `,
        `     |    | 505 |    |   `,
        `    (____/       \\____)  `,
      ];
    }

    // Build borderless psychedelic LCD output
    const speech = this.speechBubble ? `🗨 ${this.truncate(this.speechBubble, 30)}` : "✨ WUB UNICORN ✨";
    const songNote = meta?.vocalNote
      ? `♪ SING: ${meta.vocalNote} [${vowel || "--"}] ♪`
      : `♫ BEAT: ${sec} ♫`;

    const screenText = [
      speech,
      "",
      ...faceLines,
      "",
      songNote,
    ].join("\n");

    this.asciiEl.textContent = screenText;
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

  public render(): void {
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
