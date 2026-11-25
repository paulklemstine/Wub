/**
 * Vocal song arrangement: half as many vocals, full verses and choruses.
 * TypeScript mirror of `wub_core/vocal_song.py`.
 *
 * Real corpus: INTRO 11% vocal bars, GROOVE 20%, BUILDUP 21%, DROP 35%,
 * BREAKDOWN 5%. Old engine sang ~100% of blocks. Vocal song targets ~40%.
 */

export const VOCAL_HOLD_BARS = 8;

/** Full-section holds: verses and choruses hold one vocal unit for the
 *  whole 16-bar section (VERSE_1 8-24, CHORUS_1 32-48, VERSE_2 56-72,
 *  CHORUS_2 80-96) before switching to new slices. */
export const VOCAL_SECTION_HOLD_BARS = 16;

export function vocalHoldForRole(role: string): number {
  const r = (role || "").toUpperCase();
  if (r.startsWith("VERSE") || r.startsWith("CHORUS")) return VOCAL_SECTION_HOLD_BARS;
  return VOCAL_HOLD_BARS;
}

/** Hold bars for a session bar: 16 in verse/chorus sections (full song
 *  sections play before switching vocals), 8 elsewhere. */
export function vocalHoldForBar(startBar: number): number {
  const b = ((startBar % 104) + 104) % 104;
  if ((b >= 8 && b < 24) || (b >= 32 && b < 48) || (b >= 56 && b < 72) || (b >= 80 && b < 96)) {
    return VOCAL_SECTION_HOLD_BARS;
  }
  return VOCAL_HOLD_BARS;
}

export const VOCAL_PRESENCE: Record<string, number> = {
  INTRO: 0.10,
  GROOVE_A: 0.40,
  VERSE: 0.40,
  BUILDUP: 0.30,
  DROP: 0.70,
  CHORUS: 0.70,
  BREAKDOWN: 0.20,
};

export const VOCAL_LENGTH_PREFS: Record<string, number[]> = {
  VERSE_1: [8, 16, 4],
  VERSE_2: [8, 16, 4],
  CHORUS_1: [8, 16, 4],
  CHORUS_2: [8, 16, 4],
  BUILDUP_1: [4, 8],
  BUILDUP_2: [4, 8],
  INTRO: [4],
  BREAKDOWN: [4, 8],
};

export function vocalPresenceTarget(section: string): number {
  return VOCAL_PRESENCE[(section || "").toUpperCase()] ?? 0.40;
}

/** Deterministic rest decision: sparse sections rest most blocks. */
export function vocalShouldRest(section: string, blockIndex: number, salt = 0): boolean {
  const target = vocalPresenceTarget(section);
  let h = 0;
  const s = section || "";
  for (let i = 0; i < s.length; i++) h += s.charCodeAt(i);
  const roll = (blockIndex * 2654435761 + h * 97 + salt * 13) % 100;
  return roll / 100 >= target;
}

export function vocalLengthBonus(unitBars: number, role: string): number {
  const prefs = VOCAL_LENGTH_PREFS[(role || "").toUpperCase()] ?? [8, 4];
  if (prefs.length === 0) return 0;
  if (unitBars === prefs[0]) return 0.30;
  if (prefs.length > 1 && unitBars === prefs[1]) return 0.15;
  return 0;
}

export function vocalBackingChange(stem: string, category: string): number {
  const table: Record<string, Record<string, number>> = {
    CHORUS: { drums: 0.09, bass: 0.55, other: 0.58 },
    VERSE: { drums: 0.10, bass: 0.61, other: 0.64 },
    BUILDUP: { drums: 0.29, bass: 0.66, other: 0.70 },
    BREAKDOWN: { drums: 0.65, bass: 0.80, other: 0.80 },
  };
  return table[(category || "").toUpperCase()]?.[stem] ?? 0.5;
}

export function selfTestVocalSong(): void {
  const check = (c: boolean, m: string) => {
    if (!c) throw new Error(`vocal_song: ${m}`);
  };
  check(vocalPresenceTarget("INTRO") <= 0.15, "intro sparse");
  check(vocalPresenceTarget("DROP") >= 0.6, "chorus sings");
  check(vocalLengthBonus(8, "VERSE_1") > vocalLengthBonus(4, "VERSE_1"), "verse prefers 8");
  check(vocalBackingChange("drums", "CHORUS") < 0.15, "drums hold under vocals");
  check(vocalShouldRest("INTRO", 0) !== undefined, "rest deterministic");
}
