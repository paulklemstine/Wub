"""Vocal song arrangement: half as many vocals, full verses and choruses.

Real training songs (data/processed, 150 tracks):
- vocal bar rates: INTRO 11%, GROOVE_A 20%, BUILDUP 21%, DROP 35%, BREAKDOWN 5%.
- vocal run lengths: 4 bars 84%, 8 bars 10%, 12+ bars 6%.
- vocal change prob: 73% inside sections, 77% at section changes.
- chorus backing: drums hold 91% (change 9%), bass changes 55%, other 58%.
- verse backing: drums hold 90%, bass 61%, other 64%.

The old engine maximized vocals: chooseVocalPrimary filtered to
activeBars>0, frac>=0.18, avgP>=0.16, so vocals played every block (~100%).
vocal_cadence min_hold 4 bars changed vocals every block. Result: vocal
wallpaper, no verses/choruses, no rests.

Vocal song policy:
- Presence targets by section (match real songs, ~half of old engine):
  INTRO 10% (mostly rest/teaser), GROOVE/VERSE 40% (verse units, rests between),
  BUILDUP 30% (sparse chops), DROP/CHORUS 70% (full hook, still breathes),
  BREAKDOWN 20% (airy or rest).
  Overall ~40-50% of blocks have lead vocals vs old ~100%: half as much.
- Full units: prefer 8/16-bar mined units (vocal_phrases.json, 1620 units)
  for VERSE/CHORUS. Hold the unit for its full length, don't chop to 4 bars.
  CHORUS prefers hook_1bar/hook_2bar + repeat units; VERSE prefers verse/chop.
- Arrangement: VERSE_1 (unit A, 8 bars) -> CHORUS_1 (hook B, 8/16 bars) ->
  VERSE_2 (return A or variation) -> CHORUS_2 (return B). Backing follows
  vocal entrances: bass/other change 55-65% with new vocal, drums hold 90%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Presence targets: fraction of 4-bar blocks in this section that should
# carry a lead vocal. Matches real bar rates scaled to block level, roughly
# half the old engine's ~100% vocal wallpaper.
PRESENCE_BY_SECTION: Dict[str, float] = {
    "INTRO": 0.10,
    "GROOVE_A": 0.40,
    "VERSE": 0.40,
    "BUILDUP": 0.30,
    "DROP": 0.70,
    "CHORUS": 0.70,
    "BREAKDOWN": 0.20,
}

# Preferred vocal unit lengths (bars) by song role. Real lengths: CHORUS
# 4 bars 72% / 8 bars 12% / 16 bars 16%; VERSE similar. For a vocal song we
# bias to the long tail: full 8/16-bar performances, not 4-bar chops.
PREFERRED_LENGTHS: Dict[str, List[int]] = {
    "VERSE_1": [8, 16, 4],
    "VERSE_2": [8, 16, 4],
    "CHORUS_1": [8, 16, 4],
    "CHORUS_2": [8, 16, 4],
    "BUILDUP_1": [4, 8],
    "BUILDUP_2": [4, 8],
    "INTRO": [4],
    "BREAKDOWN": [4, 8],
}

VOCAL_HOLD_BARS = 8  # half as many changes as old 4-bar hold


@dataclass
class VocalUnit:
    """One mined multi-bar vocal performance."""

    track_id: str
    ids: List[str]  # consecutive 4-bar phrase ids
    bars: int
    score: float = 0.0
    repeat: bool = False
    hook: bool = False


def presence_target(section: str) -> float:
    """Fraction of blocks in this section that should have lead vocals."""
    return PRESENCE_BY_SECTION.get(section.upper(), 0.40)


def should_rest(section: str, block_index: int, salt: int = 0) -> bool:
    """Deterministic rest decision: sparse sections rest more blocks.

    Uses a hash of (section, block_index) so rests are stable across
    reloads but vary across the set. Presence target 0.4 -> rest 60%.
    """
    target = presence_target(section)
    # Simple deterministic hash: (block_index * 2654435761 + hash(section)) % 100
    h = (block_index * 2654435761 + sum(ord(c) for c in section) * 97 + salt * 13) % 100
    return (h / 100.0) >= target


def prefer_unit(unit_bars: int, role: str) -> float:
    """Length preference bonus for a vocal unit in this song role.

    First preference gets +0.30, second +0.15, else 0. Long units win in
    VERSE/CHORUS; short chops win in BUILDUP/INTRO.
    """
    prefs = PREFERRED_LENGTHS.get(role.upper(), [8, 4])
    if not prefs:
        return 0.0
    if unit_bars == prefs[0]:
        return 0.30
    if len(prefs) > 1 and unit_bars == prefs[1]:
        return 0.15
    return 0.0


def backing_should_change(stem: str, category: str) -> float:
    """Probability backing stem changes when a new vocal section starts.

    Mined from vocal_section_model.json backing_change:
    CHORUS drums 9% / bass 55% / other 58%; VERSE 10% / 61% / 64%.
    """
    table = {
        "CHORUS": {"drums": 0.09, "bass": 0.55, "other": 0.58},
        "VERSE": {"drums": 0.10, "bass": 0.61, "other": 0.64},
        "BUILDUP": {"drums": 0.29, "bass": 0.66, "other": 0.70},
        "BREAKDOWN": {"drums": 0.65, "bass": 0.80, "other": 0.80},
    }
    return table.get(category.upper(), {}).get(stem, 0.5)


def select_unit(
    units: List[VocalUnit],
    role: str,
    recalled_ids: Optional[List[str]] = None,
) -> Optional[VocalUnit]:
    """Pick a full vocal unit for this song role.

    - Theme recall wins: if the recalled hook/verse unit is available, return it.
    - Else prefer long units matching the role (8/16 bars for VERSE/CHORUS),
      weighted by mined score + repeat flag in choruses.
    """
    if not units:
        return None
    if recalled_ids:
        for u in units:
            if u.ids == recalled_ids:
                return u
    is_chorus = "CHORUS" in role.upper()
    scored = []
    for u in units:
        s = u.score + prefer_unit(u.bars, role)
        if is_chorus and u.repeat:
            s += 0.10
        if is_chorus and u.hook:
            s += 0.15
        scored.append((s, u))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]
