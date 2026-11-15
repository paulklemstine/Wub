/**
 * Music theory, functional harmony, chord-quality-safe pitch solving,
 * and rhythmic counterpoint scoring for real-time bar composition.
 */

export type StemName = "drums" | "bass" | "other" | "vocals";
export type TriadQuality = "minor" | "major" | "sus" | "mono" | "perc";
export type MotifSchema = "AAAB" | "ABAB" | "ABAC" | "AABB" | "ABCD";
export type VocalRole = "none" | "teaser" | "airy" | "hook_1bar" | "hook_2bar" | "chop" | "verse";

export interface ProgressionChord {
  relRoot: number;
  quality: "minor" | "major" | "sus";
}

export interface CellStemMeta {
  /** Self-similarity motif schema across the 4 source bars */
  s: MotifSchema;
  /** Per-bar acoustic purity score (0..1) */
  p: [number, number, number, number];
  /** Per-bar inferred root pitch class (0..11, or -1 for drums/silent) */
  r: [number, number, number, number];
  /** Per-bar triad quality */
  q: [TriadQuality, TriadQuality, TriadQuality, TriadQuality];
  /** Per-bar 16-bit 16th-note onset bitmasks */
  o: [number, number, number, number];
  /** Drums only: bar 3 acts as turnaround fill */
  fill?: boolean;
  /** Drums only: high-density bar 3 fill */
  dfill?: boolean;
  /** Vocals only: role category */
  vr?: VocalRole;
  /** Vocals only: suitable as chorus hook */
  hook?: boolean;
  /** Vocals only: hook cycle length in bars (1 or 2) */
  hb?: number;
  /** Vocals only: cleanest chop bar index (0..3) */
  cb?: number;
  /** Vocals only: vowel class */
  vow?: string;
}

export interface CellIndexData {
  version: number;
  format: string;
  count: number;
  stem_names: StemName[];
  style_progressions: Record<string, Record<string, Array<Array<[number, string] | (string | number)[]>>>>;
  section_motif_priors: Record<string, Record<string, Record<string, number>>>;
  phrases: Record<string, Record<string, CellStemMeta>>;
}

const PITCH_CLASS: Record<string, number> = {
  C: 0, "B#": 0,
  "C#": 1, DB: 1,
  D: 2,
  "D#": 3, EB: 3,
  E: 4, FB: 4,
  F: 5, "E#": 5,
  "F#": 6, GB: 6,
  G: 7,
  "G#": 8, AB: 8,
  A: 9,
  "A#": 10, BB: 10,
  B: 11, CB: 11,
};

export const STYLE_PROGRESSIONS: Record<string, Record<"minor" | "major", ProgressionChord[][]>> = {
  wave: {
    minor: [
      [{ relRoot: 0, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 3, quality: "major" }, { relRoot: 10, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 5, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 7, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 10, quality: "major" }, { relRoot: 8, quality: "major" }, { relRoot: 10, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 5, quality: "minor" }, { relRoot: 7, quality: "minor" }],
    ],
    major: [
      [{ relRoot: 0, quality: "major" }, { relRoot: 7, quality: "major" }, { relRoot: 9, quality: "minor" }, { relRoot: 5, quality: "major" }],
      [{ relRoot: 0, quality: "major" }, { relRoot: 9, quality: "minor" }, { relRoot: 5, quality: "major" }, { relRoot: 7, quality: "major" }],
    ],
  },
  cyber: {
    minor: [
      [{ relRoot: 0, quality: "minor" }, { relRoot: 0, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 10, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 1, quality: "major" }, { relRoot: 0, quality: "minor" }, { relRoot: 10, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 5, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 7, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 3, quality: "major" }, { relRoot: 10, quality: "major" }, { relRoot: 7, quality: "minor" }],
    ],
    major: [
      [{ relRoot: 0, quality: "major" }, { relRoot: 10, quality: "major" }, { relRoot: 5, quality: "major" }, { relRoot: 0, quality: "major" }],
      [{ relRoot: 0, quality: "major" }, { relRoot: 7, quality: "major" }, { relRoot: 9, quality: "minor" }, { relRoot: 5, quality: "major" }],
    ],
  },
  house: {
    minor: [
      [{ relRoot: 0, quality: "minor" }, { relRoot: 5, quality: "minor" }, { relRoot: 10, quality: "major" }, { relRoot: 3, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 0, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 10, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 7, quality: "minor" }, { relRoot: 5, quality: "minor" }, { relRoot: 0, quality: "minor" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 8, quality: "major" }, { relRoot: 3, quality: "major" }, { relRoot: 7, quality: "minor" }],
    ],
    major: [
      [{ relRoot: 0, quality: "major" }, { relRoot: 9, quality: "minor" }, { relRoot: 5, quality: "major" }, { relRoot: 7, quality: "major" }],
      [{ relRoot: 0, quality: "major" }, { relRoot: 7, quality: "major" }, { relRoot: 9, quality: "minor" }, { relRoot: 5, quality: "major" }],
      [{ relRoot: 0, quality: "major" }, { relRoot: 10, quality: "major" }, { relRoot: 5, quality: "major" }, { relRoot: 0, quality: "major" }],
    ],
  },
  breaks: {
    minor: [
      [{ relRoot: 0, quality: "minor" }, { relRoot: 0, quality: "minor" }, { relRoot: 10, quality: "major" }, { relRoot: 8, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 3, quality: "major" }, { relRoot: 5, quality: "minor" }, { relRoot: 10, quality: "major" }],
      [{ relRoot: 0, quality: "minor" }, { relRoot: 5, quality: "minor" }, { relRoot: 0, quality: "minor" }, { relRoot: 7, quality: "major" }],
    ],
    major: [
      [{ relRoot: 0, quality: "major" }, { relRoot: 10, quality: "major" }, { relRoot: 5, quality: "major" }, { relRoot: 0, quality: "major" }],
      [{ relRoot: 0, quality: "major" }, { relRoot: 5, quality: "major" }, { relRoot: 7, quality: "major" }, { relRoot: 5, quality: "major" }],
    ],
  },
};

const DOWNBEAT_MASK = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12);
const OFFBEAT_MASK = (1 << 2) | (1 << 6) | (1 << 10) | (1 << 14);
const EVEN_MASK = 0b0101010101010101;

export function clamp01(x: number): number {
  return x <= 0 ? 0 : x >= 1 ? 1 : x;
}

export function popcount16(mask: number): number {
  let m = mask & 0xffff;
  let c = 0;
  while (m !== 0) {
    m &= m - 1;
    c++;
  }
  return c;
}

export function wrapSemitones(shift: number, maxShift = 6): number {
  let s = (((Math.round(shift) + 6) % 12) + 12) % 12 - 6;
  if (maxShift < 6) {
    if (s > maxShift) s = maxShift;
    if (s < -maxShift) s = -maxShift;
  }
  return s;
}

export function parseKey(keyStr: string | undefined): { root: number; mode: "minor" | "major" } {
  if (!keyStr) return { root: 0, mode: "minor" };
  const m = keyStr.trim().match(/^([A-Ga-g])\s*([#b♯♭]?)\s*(major|minor|maj|min|m)?$/i);
  if (!m) return { root: 0, mode: "minor" };
  const letter = m[1].toUpperCase();
  let acc = m[2] || "";
  if (acc === "♯") acc = "#";
  else if (acc === "♭" || acc === "b" || acc === "B") acc = "B";
  const root = PITCH_CLASS[`${letter}${acc}`] ?? 0;
  const qual = (m[3] || "").toLowerCase();
  const mode: "minor" | "major" = qual.startsWith("m") && !qual.startsWith("maj") ? "minor" : "major";
  return { root, mode };
}

export function getProgression(
  style: string,
  mode: "minor" | "major",
  bars = 4,
  index = 0,
  customBank?: CellIndexData["style_progressions"] | null,
): ProgressionChord[] {
  const st = style.toLowerCase();
  const safeStyle = st in STYLE_PROGRESSIONS ? st : "house";
  let bank = STYLE_PROGRESSIONS[safeStyle][mode];

  if (customBank && customBank[safeStyle]?.[mode]?.length) {
    const rawList = customBank[safeStyle][mode];
    const parsed: ProgressionChord[][] = [];
    for (const row of rawList) {
      if (!Array.isArray(row) || row.length === 0) continue;
      const chords: ProgressionChord[] = [];
      for (const pair of row) {
        if (Array.isArray(pair) && pair.length >= 2) {
          const r = Number(pair[0]) || 0;
          const qRaw = String(pair[1] || mode).toLowerCase();
          const q: "minor" | "major" | "sus" =
            qRaw === "major" ? "major" : qRaw === "sus" ? "sus" : "minor";
          chords.push({ relRoot: ((r % 12) + 12) % 12, quality: q });
        }
      }
      if (chords.length >= 4) parsed.push(chords);
    }
    if (parsed.length > 0) bank = parsed;
  }

  const base = bank[((index % bank.length) + bank.length) % bank.length];
  const out: ProgressionChord[] = [];
  for (let i = 0; i < bars; i++) {
    out.push(base[i % base.length]);
  }
  return out;
}

export function thirdClashPenalty(cellQuality: TriadQuality, targetQuality: "minor" | "major" | "sus"): number {
  if (cellQuality === "mono" || cellQuality === "sus" || cellQuality === "perc" || targetQuality === "sus") {
    return 0;
  }
  if (
    (cellQuality === "minor" && targetQuality === "major") ||
    (cellQuality === "major" && targetQuality === "minor")
  ) {
    return 0.65;
  }
  return 0;
}

export function chordTriadPcs(rootPc: number, quality: "minor" | "major" | "sus"): [number, number, number] {
  const r = ((rootPc % 12) + 12) % 12;
  if (quality === "major") return [r, (r + 4) % 12, (r + 7) % 12];
  if (quality === "sus") return [r, (r + 5) % 12, (r + 7) % 12];
  return [r, (r + 3) % 12, (r + 7) % 12];
}

export function solveCellPitchShift(
  stem: StemName,
  cellRoot: number,
  cellQuality: TriadQuality,
  sessionRoot: number,
  targetRelRoot: number,
  targetQuality: "minor" | "major" | "sus",
  maxShift = 6,
  allowRelative = true,
): { semitones: number; penalty: number } {
  if (stem === "drums" || cellRoot < 0) {
    return { semitones: 0, penalty: 0 };
  }

  const targetAbsRoot = (((sessionRoot + targetRelRoot) % 12) + 12) % 12;
  const cRoot = ((cellRoot % 12) + 12) % 12;

  if (stem === "bass") {
    const raw = (((targetAbsRoot - cRoot + 6) % 12) + 12) % 12 - 6;
    const semitones = wrapSemitones(raw, maxShift);
    return { semitones, penalty: Math.abs(raw) > maxShift ? 0.25 : 0 };
  }

  if (stem === "vocals") {
    const triad = chordTriadPcs(targetAbsRoot, targetQuality);
    let bestShift = wrapSemitones(targetAbsRoot - cRoot, maxShift);
    let bestDist = 99;
    for (const tone of triad) {
      const s = (((tone - cRoot + 6) % 12) + 12) % 12 - 6;
      if (Math.abs(s) <= maxShift && Math.abs(s) < bestDist) {
        bestDist = Math.abs(s);
        bestShift = s;
      }
    }
    if (bestDist === 99) {
      return { semitones: wrapSemitones(targetAbsRoot - cRoot, maxShift), penalty: 0.35 };
    }
    const landed = ((cRoot + bestShift) % 12 + 12) % 12;
    return { semitones: bestShift, penalty: landed === targetAbsRoot ? 0 : 0.04 };
  }

  // stem === "other": chord-quality-locked harmony
  const clash = thirdClashPenalty(cellQuality, targetQuality);
  if (clash === 0) {
    const raw = (((targetAbsRoot - cRoot + 6) % 12) + 12) % 12 - 6;
    return {
      semitones: wrapSemitones(raw, maxShift),
      penalty: Math.abs(raw) > maxShift ? 0.25 : 0,
    };
  }

  if (allowRelative) {
    const relTarget =
      cellQuality === "minor" ? (targetAbsRoot + 9) % 12 : (targetAbsRoot + 3) % 12;
    const raw = (((relTarget - cRoot + 6) % 12) + 12) % 12 - 6;
    if (Math.abs(raw) <= maxShift) {
      return { semitones: raw, penalty: 0.15 };
    }
  }

  const raw = (((targetAbsRoot - cRoot + 6) % 12) + 12) % 12 - 6;
  return { semitones: wrapSemitones(raw, maxShift), penalty: clash };
}

export function scoreKickBassPocket(drumMask16: number, bassMask16: number): number {
  const d = drumMask16 & 0xffff;
  const b = bassMask16 & 0xffff;
  if (d === 0 || b === 0) return 0.5;

  const dDown = d & DOWNBEAT_MASK;
  const bDown = b & DOWNBEAT_MASK;
  const bOff = b & OFFBEAT_MASK;
  const dDownCnt = popcount16(dDown);
  const bDownCnt = popcount16(bDown);

  const unisonLock = dDownCnt > 0 ? popcount16(dDown & bDown) / dDownCnt : 0;
  const offbeatPocket = dDownCnt >= 2 && bDownCnt <= 1 ? popcount16(bOff) / 4 : 0;
  const pocket = Math.max(unisonLock, 0.92 * offbeatPocket);

  // Adjacent 16th-note rub immediately after a kick downbeat
  const rotatedDDown = ((dDown << 1) | (dDown >>> 15)) & 0xffff;
  const rubMask = rotatedDDown & b & (~(bDown << 1) & 0xffff);
  const rubPenalty = 0.14 * popcount16(rubMask);

  const bTotal = Math.max(1, popcount16(b));
  const evenRatio = popcount16(b & EVEN_MASK) / bTotal;
  return clamp01(0.35 + 0.55 * pocket + 0.1 * evenRatio - rubPenalty);
}

export function scoreSynthVocalUnmasking(
  otherMask16: number,
  vocalMask16: number,
): number {
  const o = otherMask16 & 0xffff;
  const v = vocalMask16 & 0xffff;
  const oCnt = popcount16(o);
  const vCnt = popcount16(v);
  const od = oCnt / 16;
  const vd = vCnt / 16;

  if (vCnt === 0 || vd < 0.08) return 0.85;
  if (od <= 0.2) return clamp01(0.75 + 0.25 * (1 - od / 0.2));

  const overlap = popcount16(o & v);
  const union = Math.max(1, popcount16(o | v));
  const jaccard = overlap / union;
  const callResp = oCnt > 0 ? popcount16(o & (~v & 0xffff)) / oCnt : 0;
  const denseClash = Math.max(0, od + vd - 0.85) * 0.6;
  return clamp01(0.55 + 0.4 * callResp - 0.45 * jaccard - denseClash);
}
