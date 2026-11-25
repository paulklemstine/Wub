"""Song foundation: stable 8-bar loops + 32-bar foundations that repeat like real songs.

Real training songs (data/processed, 156 tracks mined):
- within-section similarity stays high at lag 8 (coarse 0.425, motif 0.346)
  and lag 16 (0.307/0.212): grooves RETURN, they don't churn.
- section durations: INTRO 8 (4-16), GROOVE_A 12-16 (4-38), BUILDUP 5 (4-11),
  DROP 9 (4-30), BREAKDOWN 15 (5-50). Sections last 8-16 bars, not 4.
- motif return at 8 bars: drums 51%, bass 18%, other 15%, vocals 8%.
  novelty rates: drums 41%, bass 79%, other 80%, vocals 90%.
  -> drums MUST loop, vocals MUST stay fresh.

The old engine changed 1 stem per 4-bar block with novelty 0.35 + hash 0.20
dominating return bonus ~0.22, and STEM_HISTORY_AVOID 16 (64 bars) made any
8/16-bar return ineligible. Result: constant churn, no song.

This module implements the song-first policy:
- 8-bar loops: two consecutive 4-bar phrases from the SAME track, repeated.
  Bar 8 carries a fill (drums) when available.
- 32-bar foundations: drums+bass locked from one track for 32 bars.
  other/vocals may vary every 8 bars but must pass hard compatibility.
- Section changes every 8-16 bars allow 2-3 stem swaps; inside sections 0-1.
- Hard compatibility gates (not soft bonuses): tempo, key, harmony, rhythm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

STEMS = ("drums", "bass", "other", "vocals")

# Mined from data/processed (see scripts/eval_flow.py --build-reference).
# Section durations: p50 and p10-p90 band.
SECTION_DURATION_P50: Dict[str, int] = {
    "INTRO": 8,
    "GROOVE_A": 12,
    "BUILDUP": 5,
    "DROP": 9,
    "BREAKDOWN": 15,
}

# 8-bar motif return rate per stem (same 4-bar pattern returns after 8 bars).
RETURN_RATE_8: Dict[str, float] = {
    "drums": 0.51,
    "bass": 0.18,
    "other": 0.15,
    "vocals": 0.08,
}

# Novelty rate per stem (new material introduced per 4-bar boundary).
NOVELTY_RATE: Dict[str, float] = {
    "drums": 0.41,
    "bass": 0.79,
    "other": 0.80,
    "vocals": 0.90,
}

FOUNDATION_BARS = 32
LOOP_BARS = 8
BLOCK_BARS = 4

# Hard compatibility gates for foundation membership.
MAX_TEMPO_DEVIATION_FOUNDATION = 0.03  # |rate-1| <= 3% for drums/bass
MAX_TEMPO_DEVIATION_DECORATIVE = 0.06  # other/vocals may stretch a bit more
MAX_KEY_SHIFT_FOUNDATION = 3  # semitones for pitched foundation stems
MAX_KEY_SHIFT_DECORATIVE = 4
MIN_CHROMA_COSINE = 0.60  # quartet harmony floor (p10 in real songs is 0.655)
FORBIDDEN_ROOT_INTERVALS = {1, 6, 11}  # minor 2nd, tritone, major 7th


@dataclass
class LoopCandidate:
    """An 8-bar loop built from one track's phrases."""

    track_id: str
    phrase_ids: List[str]  # 2x 4-bar phrase ids, played A A B B or A B A B
    schema: str = "AABB"  # AABB or ABAB
    has_fill: bool = False
    energy: float = 0.5
    section: str = "GROOVE_A"


@dataclass
class FoundationState:
    """A 32-bar stable core: drums+bass locked, other/vocals vary slowly."""

    start_bar: int = 0
    bars: int = FOUNDATION_BARS
    drum_track: str = ""
    bass_track: str = ""
    drum_loop: Optional[LoopCandidate] = None
    bass_loop: Optional[LoopCandidate] = None
    changes_this_foundation: int = 0

    def bars_remaining(self, bar: int) -> int:
        return max(0, self.start_bar + self.bars - bar)

    def is_expired(self, bar: int) -> bool:
        return bar >= self.start_bar + self.bars


def section_for_bar(bar_in_set: int) -> Tuple[str, float]:
    """Song-like arrangement for a 104-bar set, mined durations.

    Returns (section, target_energy). Sections last 8-16 bars, not 4:
      0-8 INTRO (0.30), 8-24 GROOVE (0.68), 24-32 BUILDUP (0.84),
      32-48 CHORUS/DROP (0.94), 48-56 BREAKDOWN (0.44),
      56-72 VERSE/GROOVE (0.74), 72-80 BUILDUP (0.88),
      80-96 CHORUS/DROP (0.96), 96-104 OUTRO/BRIDGE (0.25).
    """
    b = bar_in_set % 104
    if b < 8:
        return "INTRO", 0.30
    if b < 24:
        return "GROOVE_A", 0.68
    if b < 32:
        return "BUILDUP", 0.84
    if b < 48:
        return "DROP", 0.94
    if b < 56:
        return "BREAKDOWN", 0.44
    if b < 72:
        return "GROOVE_A", 0.74
    if b < 80:
        return "BUILDUP", 0.88
    if b < 96:
        return "DROP", 0.96
    return "BREAKDOWN", 0.25


def movement_budget_song(bar: int, block_bars: int, section_changed: bool) -> Dict[str, int]:
    """Song-like movement budget: hold foundations, move at sections.

    - Inside a section: 4-bar blocks hold (0 changes), 8-bar blocks allow 1.
    - At a section change: allow up to 3 (one foundation stem + two decorative).
    - Foundations last 32 bars; max_hold is 32, not 16.
    """
    if block_bars <= 4:
        if section_changed:
            return {"minChanges": 1, "maxChanges": 2, "maxHoldBars": 32}
        return {"minChanges": 0, "maxChanges": 1, "maxHoldBars": 32}
    # 8/16-bar blocks
    if section_changed:
        return {"minChanges": 1, "maxChanges": 3, "maxHoldBars": 32}
    return {"minChanges": 0, "maxChanges": 1, "maxHoldBars": 32}


def is_compatible(
    stem: str,
    rate: float,
    key_shift: int,
    chroma_cosine: Optional[float] = None,
    root_interval: Optional[int] = None,
    is_foundation: bool = True,
) -> Tuple[bool, str]:
    """Hard compatibility gate for a candidate phrase in a foundation.

    Returns (ok, reason). Unlike soft bonuses, a False here rejects the
    candidate outright: no extreme stretches, no tritones, no harmonic mush.
    """
    if stem == "drums":
        # Drums are unpitched: only tempo matters.
        limit = MAX_TEMPO_DEVIATION_FOUNDATION if is_foundation else MAX_TEMPO_DEVIATION_DECORATIVE
        if abs(rate - 1.0) > limit + 1e-9:
            return False, f"tempo {rate:.3f} exceeds {limit:.2f}"
        return True, "ok"
    max_rate = MAX_TEMPO_DEVIATION_FOUNDATION if is_foundation else MAX_TEMPO_DEVIATION_DECORATIVE
    if abs(rate - 1.0) > max_rate + 1e-9:
        return False, f"tempo {rate:.3f} exceeds {max_rate:.2f}"
    max_shift = MAX_KEY_SHIFT_FOUNDATION if is_foundation else MAX_KEY_SHIFT_DECORATIVE
    if abs(key_shift) > max_shift:
        return False, f"key shift {key_shift} exceeds {max_shift}"
    if root_interval is not None and root_interval % 12 in FORBIDDEN_ROOT_INTERVALS:
        return False, f"forbidden root interval {root_interval % 12}"
    if chroma_cosine is not None and chroma_cosine < MIN_CHROMA_COSINE:
        return False, f"chroma {chroma_cosine:.2f} below floor {MIN_CHROMA_COSINE:.2f}"
    return True, "ok"


def should_return(stem: str, age_bars: int) -> float:
    """Probability that a stem should return its loop at this age.

    Matches corpus return rates: drums high at 8 bars, vocals almost never.
    Used as a ranking bonus, not a hard rule.
    """
    base = RETURN_RATE_8.get(stem, 0.15)
    if age_bars == 8:
        return base
    if age_bars == 16:
        return base * 0.8
    if age_bars == 32:
        return base * 0.7
    return 0.0


def repetition_bonus(stem: str, plays: int, age_bars: Optional[int]) -> float:
    """Song-like repetition bonus: reward returns, lightly penalize loops.

    - A phrase heard 8/16/32 bars ago gets a return bonus (drums large).
    - A phrase on its 2nd play gets a small penalty; 3rd+ gets larger.
      Drums are penalized least (they must loop), vocals most.
    """
    bonus = 0.0
    if age_bars is not None and age_bars in (8, 16, 32):
        bonus += should_return(stem, age_bars) * 0.45
    if plays == 0:
        # Fresh material: small bonus scaled by stem novelty rate.
        # Drums 0.41 -> +0.08, vocals 0.90 -> +0.18. Much smaller than old 0.35.
        bonus += NOVELTY_RATE.get(stem, 0.7) * 0.20
    elif plays == 1:
        bonus -= {"drums": 0.02, "bass": 0.06, "other": 0.06, "vocals": 0.10}.get(stem, 0.06)
    else:
        bonus -= {"drums": 0.08, "bass": 0.15, "other": 0.15, "vocals": 0.25}.get(stem, 0.15)
    return bonus


def build_loop_order(schema: str, block_bars: int) -> List[int]:
    """Bar-to-phrase mapping for an 8-bar loop repeated to fill block_bars.

    Schemas: AABB (bars 0,1->A, 2,3->B, repeat) or ABAB (alternate).
    Returns list of phrase indices (0/1) per 4-bar phrase slot.
    """
    slots = max(1, block_bars // 4)
    if schema == "ABAB":
        return [i % 2 for i in range(slots)]
    # AABB: pairs
    return [(i // 2) % 2 for i in range(slots)]
