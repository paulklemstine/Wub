"""Music theory, functional harmony, counterpoint, and acoustic purity engine.

Provides pure deterministic rules shared by corpus mining, cell indexing,
neural compatibility training, and runtime verification:
1. Key parsing & tonic-relative Roman-numeral chord progressions per style.
2. Triad quality classification (minor / major / sus / mono).
3. Chord-quality-safe semitone pitch shifting for bass, other, and vocals.
4. Rhythmic counterpoint: kick-bass pocket lock & synth-vocal unmasking.
5. Per-cell acoustic purity scoring.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

STEMS: Tuple[str, ...] = ("drums", "bass", "other", "vocals")

PITCH_CLASS: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

_KEY_RE = re.compile(r"^\s*([A-Ga-g])\s*([#b♯♭]?)\s*(major|minor|maj|min|m)?\s*$", re.IGNORECASE)

# Tonic-relative progressions: list of (rel_root_semitones, quality) per bar.
# Supports 4-bar and 8-bar blueprints per style and mode.
STYLE_PROGRESSIONS: Dict[str, Dict[str, List[List[Tuple[int, str]]]]] = {
    "wave": {
        "minor": [
            [(0, "minor"), (8, "major"), (3, "major"), (10, "major")],   # i - VI - III - VII
            [(0, "minor"), (5, "minor"), (8, "major"), (7, "major")],    # i - iv - VI - V
            [(0, "minor"), (10, "major"), (8, "major"), (10, "major")],  # i - VII - VI - VII
            [(0, "minor"), (8, "major"), (5, "minor"), (7, "minor")],    # i - VI - iv - v
        ],
        "major": [
            [(0, "major"), (7, "major"), (9, "minor"), (5, "major")],    # I - V - vi - IV
            [(0, "major"), (9, "minor"), (5, "major"), (7, "major")],    # I - vi - IV - V
        ],
    },
    "cyber": {
        "minor": [
            [(0, "minor"), (0, "minor"), (8, "major"), (10, "major")],   # i - i - VI - VII
            [(0, "minor"), (1, "major"), (0, "minor"), (10, "major")],   # i - bII - i - VII (Phrygian dark)
            [(0, "minor"), (5, "minor"), (8, "major"), (7, "major")],    # i - iv - VI - V
            [(0, "minor"), (3, "major"), (10, "major"), (7, "minor")],   # i - III - VII - v
        ],
        "major": [
            [(0, "major"), (10, "major"), (5, "major"), (0, "major")],   # I - bVII - IV - I
            [(0, "major"), (7, "major"), (9, "minor"), (5, "major")],
        ],
    },
    "house": {
        "minor": [
            [(0, "minor"), (5, "minor"), (10, "major"), (3, "major")],   # i - iv - VII - III
            [(0, "minor"), (0, "minor"), (8, "major"), (10, "major")],   # i - i - VI - VII
            [(0, "minor"), (7, "minor"), (5, "minor"), (0, "minor")],    # i - v - iv - i
            [(0, "minor"), (8, "major"), (3, "major"), (7, "minor")],    # i - VI - III - v
        ],
        "major": [
            [(0, "major"), (9, "minor"), (5, "major"), (7, "major")],    # I - vi - IV - V
            [(0, "major"), (7, "major"), (9, "minor"), (5, "major")],    # I - V - vi - IV
            [(0, "major"), (10, "major"), (5, "major"), (0, "major")],   # I - bVII - IV - I
        ],
    },
    "breaks": {
        "minor": [
            [(0, "minor"), (0, "minor"), (10, "major"), (8, "major")],   # i - i - VII - VI
            [(0, "minor"), (3, "major"), (5, "minor"), (10, "major")],   # i - III - iv - VII
            [(0, "minor"), (5, "minor"), (0, "minor"), (7, "major")],    # i - iv - i - V
        ],
        "major": [
            [(0, "major"), (10, "major"), (5, "major"), (0, "major")],   # I - bVII - IV - I
            [(0, "major"), (5, "major"), (7, "major"), (5, "major")],    # I - IV - V - IV
        ],
    },
}

MIN_RMS_DB: Dict[str, float] = {
    "drums": -28.0,
    "bass": -28.0,
    "other": -34.0,
    "vocals": -42.0,
}

DOWNBEAT_STEPS = {0, 4, 8, 12}
OFFBEAT_STEPS = {2, 6, 10, 14}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def wrap_semitones(shift: int, max_shift: int = 6) -> int:
    """Wrap a semitone delta into [-6, +6] (or clamp if max_shift < 6)."""
    s = ((int(shift) + 6) % 12) - 6
    if max_shift < 6:
        if s > max_shift:
            s = max(-max_shift, min(max_shift, s))
        elif s < -max_shift:
            s = max(-max_shift, min(max_shift, s))
    return s


def parse_key(key_str: str) -> Tuple[int, str]:
    """Parse 'F# minor' / 'C major' / 'Am' -> (root_pc 0..11, 'minor'|'major')."""
    if not key_str or not isinstance(key_str, str):
        return 0, "minor"
    m = _KEY_RE.match(key_str.strip())
    if not m:
        return 0, "minor"
    letter = m.group(1).upper()
    acc = m.group(2) or ""
    if acc == "♯":
        acc = "#"
    elif acc in ("♭", "b", "B"):
        acc = "B"
    root = PITCH_CLASS.get(f"{letter}{acc}", 0)
    qual = (m.group(3) or "").lower()
    mode = "minor" if qual.startswith("m") and not qual.startswith("maj") else "major"
    return root, mode


def get_progression(
    style: str,
    mode: str = "minor",
    bars: int = 4,
    index: int = 0,
) -> List[Tuple[int, str]]:
    """Return a style- and mode-conditioned Roman-numeral chord progression."""
    st = style.lower() if style and style.lower() in STYLE_PROGRESSIONS else "house"
    md = "minor" if str(mode).lower().startswith("min") else "major"
    bank = STYLE_PROGRESSIONS[st][md]
    base = bank[index % len(bank)]
    if bars <= len(base):
        return list(base[:bars])
    out: List[Tuple[int, str]] = []
    while len(out) < bars:
        out.extend(base)
    return out[:bars]


def classify_triad_quality(chroma_12: Sequence[float]) -> Tuple[int, str, float]:
    """Infer dominant root, triad quality ('minor'|'major'|'sus'|'mono'), and tonal clarity.

    Evaluates minor triad (0, 3, 7), major triad (0, 4, 7), and sus/power (0, 5/2, 7)
    across all 12 candidate roots. Monophonic inputs (strong single bin or root+fifth)
    return 'mono' so they remain harmonically safe over both minor and major chords.
    """
    if not chroma_12 or len(chroma_12) != 12:
        return 0, "mono", 0.0
    vec = [max(0.0, float(v)) for v in chroma_12]
    total = sum(vec)
    if total <= 1e-8:
        return 0, "mono", 0.0
    norm = [v / total for v in vec]

    best_root = max(range(12), key=lambda i: norm[i])
    top_weight = norm[best_root]
    sorted_w = sorted(norm, reverse=True)
    second_w = sorted_w[1] if len(sorted_w) > 1 else 0.0
    third_w = sorted_w[2] if len(sorted_w) > 2 else 0.0

    # Monophonic clarity: tonic dominates or only root + fifth are active
    fifth_w = norm[(best_root + 7) % 12]
    if top_weight >= 0.52 and third_w < 0.10:
        return best_root, "mono", clamp01((top_weight - 0.15) / 0.55)
    if top_weight + fifth_w >= 0.72 and third_w < 0.09:
        return best_root, "mono", clamp01((top_weight + fifth_w - 0.30) / 0.50)

    # Score candidate triad templates over all 12 roots
    best_score = -1.0
    best_qual = "minor"
    cand_root = best_root
    for r in range(12):
        w_root = norm[r]
        w_m3 = norm[(r + 3) % 12]
        w_M3 = norm[(r + 4) % 12]
        w_4 = norm[(r + 5) % 12]
        w_5 = norm[(r + 7) % 12]
        w_2 = norm[(r + 2) % 12]

        s_min = 1.2 * w_root + 1.0 * w_m3 + 0.9 * w_5 - 0.8 * w_M3
        s_maj = 1.2 * w_root + 1.0 * w_M3 + 0.9 * w_5 - 0.8 * w_m3
        s_sus = 1.15 * w_root + 0.85 * max(w_4, w_2) + 0.95 * w_5 - 0.5 * max(w_m3, w_M3)

        for qual, score in (("minor", s_min), ("major", s_maj), ("sus", s_sus)):
            if score > best_score:
                best_score = score
                best_qual = qual
                cand_root = r

    # Clarity reflects how strongly the triad energy stands out over uniform noise (1/12)
    clarity = clamp01((best_score - 0.25) / 0.55)
    if best_qual in ("minor", "major") and second_w < 0.12:
        best_qual = "mono"
    return cand_root, best_qual, clarity


def third_clash_penalty(cell_quality: str, target_quality: str) -> float:
    """Return 0.65 if minor vs major thirds collide, else 0.0."""
    cq = (cell_quality or "mono").lower()
    tq = (target_quality or "minor").lower()
    if cq in ("mono", "sus") or tq in ("sus", "mono"):
        return 0.0
    if (cq == "minor" and tq == "major") or (cq == "major" and tq == "minor"):
        return 0.65
    return 0.0


def chord_triad_pcs(root_pc: int, quality: str) -> Tuple[int, int, int]:
    """Return pitch classes (root, third/fourth, fifth) for a chord."""
    r = int(root_pc) % 12
    q = (quality or "minor").lower()
    if q == "major":
        return (r, (r + 4) % 12, (r + 7) % 12)
    if q == "sus":
        return (r, (r + 5) % 12, (r + 7) % 12)
    return (r, (r + 3) % 12, (r + 7) % 12)


def solve_cell_pitch_shift(
    stem: str,
    cell_root: Optional[int],
    cell_quality: str,
    session_root: int,
    target_rel_root: int,
    target_quality: str,
    max_shift: int = 6,
    allow_relative: bool = True,
) -> Tuple[int, float]:
    """Compute chord-quality-safe semitone shift and music-theory penalty.

    Returns ``(semitone_shift, theory_penalty)`` where ``theory_penalty`` is
    in ``0.0..1.0`` (0 = ideal harmonic fit, >=0.65 = third clash).
    """
    if stem == "drums" or cell_root is None or cell_root < 0:
        return 0, 0.0

    target_abs_root = (int(session_root) + int(target_rel_root)) % 12
    c_root = int(cell_root) % 12
    c_qual = (cell_quality or "mono").lower()
    t_qual = (target_quality or "minor").lower()

    if stem == "bass":
        shift = wrap_semitones(target_abs_root - c_root, max_shift)
        # Small preference penalty only if shift had to be clamped past max_shift
        raw_s = ((target_abs_root - c_root + 6) % 12) - 6
        penalty = 0.25 if abs(raw_s) > max_shift else 0.0
        return shift, penalty

    if stem == "vocals":
        # Lock vocals to the chord triad (or session tonic when target_rel_root == 0)
        triad = chord_triad_pcs(target_abs_root, t_qual)
        best_shift = wrap_semitones(target_abs_root - c_root, max_shift)
        best_dist = 99
        for tone in triad:
            s = ((tone - c_root + 6) % 12) - 6
            if abs(s) <= max_shift and abs(s) < best_dist:
                best_dist = abs(s)
                best_shift = s
        if best_dist == 99:
            return wrap_semitones(target_abs_root - c_root, max_shift), 0.35
        # Root match gets 0.0 penalty; 5th/3rd gets tiny color penalty (0.04)
        tone_penalty = 0.0 if (c_root + best_shift) % 12 == target_abs_root else 0.04
        return best_shift, tone_penalty

    # stem == "other": enforce triad-quality lock to prevent minor/major 3rd clashes
    clash = third_clash_penalty(c_qual, t_qual)
    if clash == 0.0:
        raw_s = ((target_abs_root - c_root + 6) % 12) - 6
        shift = wrap_semitones(raw_s, max_shift)
        penalty = 0.25 if abs(raw_s) > max_shift else 0.0
        return shift, penalty

    if allow_relative:
        # Relative minor of major chord is -3 semitones (e.g. Fm over Abmaj -> Abmaj6);
        # relative major of minor chord is +3 semitones (e.g. Ebmaj over Cm -> Cm7).
        rel_target = (target_abs_root - 3) % 12 if c_qual == "minor" else (target_abs_root + 3) % 12
        raw_s = ((rel_target - c_root + 6) % 12) - 6
        if abs(raw_s) <= max_shift:
            return raw_s, 0.15

    raw_s = ((target_abs_root - c_root + 6) % 12) - 6
    return wrap_semitones(raw_s, max_shift), clash


def _normalize_onset_steps(onset_16: Sequence[int]) -> set[int]:
    """Accept either a 16-element binary/float mask or a list of active step indices."""
    if len(onset_16) == 16 and all(v in (0, 1, False, True) or (isinstance(v, float) and 0.0 <= v <= 1.0) for v in onset_16):
        # Distinguish a 16-step binary vector from a list of indices only if values are 0/1 floats
        # Wait: if an onset index list happens to have length 16, it would contain 0..15.
        if any(v > 1 for v in onset_16):
            return {int(v) % 16 for v in onset_16}
        return {i for i, v in enumerate(onset_16) if float(v) > 0.25}
    return {int(v) % 16 for v in onset_16}


def onset_bits_to_set(mask16: int) -> set[int]:
    return {i for i in range(16) if (int(mask16) >> i) & 1}


def set_to_onset_bits(steps: Sequence[int]) -> int:
    mask = 0
    for s in steps:
        mask |= 1 << (int(s) % 16)
    return mask


def score_kick_bass_pocket(
    drum_onset_16: Sequence[int],
    bass_onset_16: Sequence[int],
) -> float:
    """Score rhythmic pocket between drums and bass in 0.0..1.0.

    Rewards:
    - Downbeat unison lock (steps 0, 8, and 4, 12)
    - Classic offbeat dance/house pocket (bass on 2, 6, 10, 14 while drums hit 0, 4, 8, 12)
    Penalizes:
    - Muddy 16th-note rub immediately adjacent (+/-1 step) to kick downbeats when unaligned
    """
    d_steps = _normalize_onset_steps(drum_onset_16)
    b_steps = _normalize_onset_steps(bass_onset_16)
    if not d_steps or not b_steps:
        return 0.5

    d_down = d_steps & DOWNBEAT_STEPS
    b_down = b_steps & DOWNBEAT_STEPS
    b_off = b_steps & OFFBEAT_STEPS

    unison_lock = len(d_down & b_down) / max(1, len(d_down)) if d_down else 0.0
    offbeat_pocket = len(b_off) / 4.0 if len(d_down) >= 2 and len(b_down) <= 1 else 0.0
    pocket = max(unison_lock, 0.92 * offbeat_pocket)

    # Penalize adjacent 16th-note rub right after downbeat kicks without offbeat groove
    rub = 0
    for k in d_down:
        if ((k + 1) % 16) in b_steps and k not in b_steps:
            rub += 1
    rub_penalty = 0.14 * rub

    # Subtle bonus for sharing beat-grid alignment (even subdivisions)
    even_b = sum(1 for s in b_steps if s % 2 == 0) / max(1, len(b_steps))
    score = 0.35 + 0.55 * pocket + 0.10 * even_b - rub_penalty
    return clamp01(score)


def score_synth_vocal_unmasking(
    other_onset_16: Sequence[int],
    vocal_onset_16: Sequence[int],
    other_density: float,
    vocal_density: float,
) -> float:
    """Score frequency & rhythmic unmasking between `other` and `vocals` in 0.0..1.0.

    Rewards call-and-response (synth stabs firing in vocal rest steps) or sparse
    sustained pad accompaniment when vocals are busy; penalizes simultaneous
    high-density rhythmic collisions.
    """
    o_steps = _normalize_onset_steps(other_onset_16)
    v_steps = _normalize_onset_steps(vocal_onset_16)
    od = clamp01(other_density)
    vd = clamp01(vocal_density)

    if vd < 0.08 or not v_steps:
        return 0.85  # No active vocal to mask

    # If `other` is sparse pad (low onset density), it leaves vocal space open
    if od <= 0.20:
        return clamp01(0.75 + 0.25 * (1.0 - od / 0.20))

    overlap = len(o_steps & v_steps)
    union = len(o_steps | v_steps)
    jaccard = overlap / max(1, union)
    call_resp = len(o_steps - v_steps) / max(1, len(o_steps))

    # Simultaneous dense rhythmic clashes hurt intelligibility
    dense_clash = max(0.0, (od + vd) - 0.85) * 0.6
    score = 0.55 + 0.40 * call_resp - 0.45 * jaccard - dense_clash
    return clamp01(score)


def score_cell_purity(
    stem: str,
    rms_db: float,
    peak: float,
    chroma_12: Sequence[float],
    onset_count: int,
) -> float:
    """Compute acoustic purity score (0..1) for a stem cell."""
    floor = MIN_RMS_DB.get(stem, -34.0)
    if rms_db < floor - 6.0 or peak <= 1e-4:
        return 0.0

    presence = clamp01((rms_db - floor + 6.0) / 18.0)
    peak_db = 20.0 * math.log10(max(1e-5, float(peak)))
    crest_db = max(0.0, peak_db - float(rms_db))

    if stem == "drums":
        # Crisp drum transients have healthy crest factor (6..18 dB) and rhythmic density
        transient = clamp01((crest_db - 3.0) / 12.0)
        density_fit = clamp01(onset_count / 6.0)
        return clamp01(0.40 * presence + 0.40 * transient + 0.20 * density_fit)

    if stem == "bass":
        # Clean bass cells have strong monophonic pitch concentration
        _, qual, clarity = classify_triad_quality(chroma_12)
        mono_bonus = 0.20 if qual == "mono" else 0.0
        return clamp01(0.40 * presence + 0.40 * clarity + mono_bonus)

    if stem == "vocals":
        # Clean vocal cells have audible formant presence and clear tonal focus
        _, _, clarity = classify_triad_quality(chroma_12)
        density_fit = 1.0 if 1 <= onset_count <= 10 else 0.65
        return clamp01(0.45 * presence + 0.35 * clarity + 0.20 * density_fit)

    # stem == "other": reward tonal harmonic clarity and presence
    _, _, clarity = classify_triad_quality(chroma_12)
    return clamp01(0.45 * presence + 0.55 * clarity)
