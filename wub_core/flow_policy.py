"""DJ flow policy mirror for the browser engine (see frontend/src/arc_planner.ts).

This module is a pure, dependency-free mirror of the in-browser pure policy
helpers (A) signature phrases, (C) timbre-budget diversity and (B) DJ
transition techniques.  The TypeScript implementation in ``arc_planner.ts``
is the live path (the browser never imports Python); this file exists so
``pytest`` can cover the policy math without a JS test harness.  Keep the
constants and formulas here in sync with the TS source when retuning.

All helpers degrade to the pre-existing behaviour when a gate is closed:
0 bonus, no signatures, plain equal-gain hand-off.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set

# ---- Tunable knobs (mirror of arc_planner.ts FLOW_* constants) ----
FLOW_SIGNATURES_ENABLED = True
FLOW_SIGNATURES_PER_STEM = 2
FLOW_SIGNATURE_PEAK_ENERGY = 0.90
FLOW_SIGNATURE_BONUS = 0.14
FLOW_SIGNATURE_MIN_GAP_BARS = 16
FLOW_SIGNATURE_REPETITION_DISCOUNT = 0.6
FLOW_SIGNATURE_BONUS_FLOOR = 0.50

FLOW_TIMBRE_WINDOW_BARS = 32
FLOW_TIMBRE_BONUS = 0.08
FLOW_TIMBRE_OVERUSE_PENALTY = 0.04

FLOW_TRANSITIONS_ENABLED = True
FLOW_EQ_OUT_MIN_GAIN = 0.15
FLOW_EQ_OUT_FLOOR_HZ = 300
FLOW_VOCAL_LOUD_RMS_DB = -18.0
FLOW_VOCAL_FADE_LOUD_SEC = 0.15
FLOW_VOCAL_CROSSFADE_SEC = 0.22

STEMS = ("drums", "bass", "other", "vocals")


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def signature_energy_scale(target_energy: float) -> float:
    """0 at/below FLOOR, 1 at/above PEAK, linear between (monotonic)."""
    energy = _clamp01(target_energy if math.isfinite(target_energy) else 0.5)
    peak = max(0.51, FLOW_SIGNATURE_PEAK_ENERGY)
    denom = peak - FLOW_SIGNATURE_BONUS_FLOOR
    if denom <= 0:
        return 0.0
    return _clamp01((energy - FLOW_SIGNATURE_BONUS_FLOOR) / denom)


def pick_signatures(
    candidates: Sequence[dict],
    per_stem: int = FLOW_SIGNATURES_PER_STEM,
) -> Set[str]:
    """Pick 1-2 signature phrase ids from shipped signals only.

    Excludes intro/outro/once moments, prefers high taste quality + ears +
    energy with a small DROP-section bias, prefers distinct source tracks,
    and breaks ties deterministically by phrase id.  Missing quality falls
    back to 0.5 so taste-less banks still get anchors.  Empty input yields
    an empty set.
    """
    picked: Set[str] = set()
    if not FLOW_SIGNATURES_ENABLED:
        return picked
    count = max(0, int(per_stem))
    if count <= 0 or not candidates:
        return picked

    scored: List[tuple] = []  # (phrase_id, track_id, score)
    for c in candidates:
        phrase_id = c.get("phraseId", c.get("phrase_id", ""))
        if not phrase_id:
            continue
        if c.get("isIntro") or c.get("is_intro"):
            continue
        if c.get("isOutro") or c.get("is_outro"):
            continue
        if c.get("once"):
            continue
        quality_raw = c.get("quality", c.get("q"))
        try:
            quality = _clamp01(float(quality_raw)) if quality_raw is not None else 0.5
            if quality_raw is not None and not math.isfinite(float(quality_raw)):
                quality = 0.5
        except (TypeError, ValueError):
            quality = 0.5
        ear_q_raw = c.get("earQ", c.get("ear_q", 0.5))
        try:
            ear_q = _clamp01(float(ear_q_raw))
        except (TypeError, ValueError):
            ear_q = 0.5
        energy_raw = c.get("energy", 0.5)
        try:
            energy = _clamp01(float(energy_raw))
        except (TypeError, ValueError):
            energy = 0.5
        section = str(c.get("section", "")).upper()
        section_bonus = 0.10 if section == "DROP" else 0.05 if section == "GROOVE_A" else 0.0
        score = 0.55 * quality + 0.15 * ear_q + 0.20 * energy + section_bonus
        track_id = str(c.get("trackId", c.get("track_id", "")))
        scored.append((phrase_id, track_id, score))

    # Sort by descending score, then phrase id, then track (deterministic).
    scored.sort(key=lambda item: (-item[2], item[0], item[1]))

    used_tracks: Set[str] = set()
    for phrase_id, track_id, _score in scored:
        if len(picked) >= count:
            break
        if track_id and track_id in used_tracks:
            continue
        picked.add(phrase_id)
        if track_id:
            used_tracks.add(track_id)
    if len(picked) < count:
        for phrase_id, _track_id, _score in scored:
            if len(picked) >= count:
                break
            if phrase_id not in picked:
                picked.add(phrase_id)
    return picked


def signature_bonus(
    is_signature: bool,
    target_energy: float,
    gap_bars: float,
    pinned: bool = False,
) -> float:
    """Energy-gated bonus 0..FLOW_SIGNATURE_BONUS (0 when not rested/pinned)."""
    if not FLOW_SIGNATURES_ENABLED:
        return 0.0
    if not is_signature:
        return 0.0
    if pinned:
        return 0.0
    try:
        gap = float(gap_bars)
    except (TypeError, ValueError):
        gap = -1.0
    rested = (not math.isfinite(gap)) or gap < 0 or gap >= FLOW_SIGNATURE_MIN_GAP_BARS
    if not rested:
        return 0.0
    return FLOW_SIGNATURE_BONUS * signature_energy_scale(target_energy)


def timbre_bonus(recent_count: int, window_blocks: int) -> float:
    """Soft bonus for an under-used timbre (+0.08 unseen -> -0.04 x3, bounded)."""
    count = max(0, int(recent_count))
    blocks = max(0, int(window_blocks))
    if blocks <= 0:
        return 0.0
    step = (FLOW_TIMBRE_BONUS + FLOW_TIMBRE_OVERUSE_PENALTY) / 3.0
    bonus = FLOW_TIMBRE_BONUS - step * count
    floor = -FLOW_TIMBRE_OVERUSE_PENALTY * 2.0
    return max(floor, min(FLOW_TIMBRE_BONUS, bonus))


def select_transition_technique(
    stem: str,
    outgoing_energy: float,
    incoming_energy: float,
    incoming_rms_db: float,
    target_energy: float,
    is_bridge: bool,
    vocal_changed: bool = True,
) -> Dict[str, object]:
    """Pure DJ-technique selector (EQ-OUT tail + vocal treatment only).

    Drums never EQ to 300 Hz; bridges stay plain; everything degrades to
    the equal-gain hand-off when the master gate is closed.
    """
    plain: Dict[str, object] = {
        "eq_out": False,
        "eq_out_floor_hz": FLOW_EQ_OUT_FLOOR_HZ,
        "vocal_reverb_tail": False,
        "vocal_fade_in_sec": None,
        "vocal_crossfade_sec": None,
    }
    if not FLOW_TRANSITIONS_ENABLED:
        return dict(plain)
    if is_bridge:
        return dict(plain)
    outgoing = _clamp01(outgoing_energy if math.isfinite(outgoing_energy) else 0.5)
    incoming = _clamp01(incoming_energy if math.isfinite(incoming_energy) else 0.5)
    try:
        rms_db = float(incoming_rms_db)
    except (TypeError, ValueError):
        rms_db = float("nan")
    result: Dict[str, object] = dict(plain)
    if stem != "drums" and (incoming - outgoing) >= FLOW_EQ_OUT_MIN_GAIN:
        result["eq_out"] = True
    if stem == "vocals":
        if vocal_changed:
            result["vocal_reverb_tail"] = True
            result["vocal_crossfade_sec"] = FLOW_VOCAL_CROSSFADE_SEC
        if math.isfinite(rms_db) and rms_db >= FLOW_VOCAL_LOUD_RMS_DB and incoming >= 0.40:
            result["vocal_fade_in_sec"] = FLOW_VOCAL_FADE_LOUD_SEC
    return result


def repetition_discount_factor(is_signature: bool) -> float:
    """Multiplies play counts for signatures (1 - discount = 0.4)."""
    if not FLOW_SIGNATURES_ENABLED or not is_signature:
        return 1.0
    return 1.0 - FLOW_SIGNATURE_REPETITION_DISCOUNT
