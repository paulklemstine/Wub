"""Prototype spec: learned fills/turnarounds, groove pocket, loudness normalization.

Pure functions only — no I/O, no side effects, no numpy dependency.
Intended as the reference semantics for a future frontend
(`client_ml_sequencer.ts` scheduler) + backend (`StatisticalGrooveModel`,
`PatternAudioFeedbackEngine`) implementation. The main agent owns those
integration points; this module must stay dependency-free so it can be
ported 1:1 to TypeScript later.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# 1. Fill / turnaround bar detection
# ---------------------------------------------------------------------------

#: Slice categories that count as fill material. Covers both backend
#: (`groovy_pattern_assembler.py` AcousticSliceClassifier: "fill") and
#: frontend (`SliceAsset` category strings like "drums:fill") conventions.
FILL_CATEGORIES = frozenset({"fill", "drums:fill", "vocals:stutter"})

#: Canonical turnaround landing steps (beat-4 roll into the next phrase).
TURNAROUND_STEPS = (12, 13, 14)

#: Minimum second-half (steps 8..15) onset count for a "dense" drum bar.
DENSE_SECOND_HALF_MIN = 5

#: Minimum overall density (onsets / 16) for a "dense" bar.
DENSE_DENSITY_MIN = 0.5


def detect_fill_bar(onset_grid: Sequence[int], slice_categories: Sequence[str]) -> bool:
    """Heuristic: is this 16-step drum bar a turnaround fill?

    Fires when the bar is dense AND carries fill-category slices, or when
    the canonical beat-4 turnaround run (steps 12, 13, 14) is fully present::

        wub_core/groove_miner.py DEFAULT_ONSET_PRIORS["drums"]["fill"]
        wub_core/groovy_pattern_assembler.py turnaround_fill template
        frontend composeBarPatterns bar-7 turnaround (steps 12..14)
    """
    onsets = {int(s) % 16 for s in onset_grid}
    if not onsets:
        return False
    cats = [str(c).lower() for c in slice_categories]
    fill_hits = sum(1 for c in cats if c in FILL_CATEGORIES)
    density = len(onsets) / 16.0
    second_half = sum(1 for s in onsets if s >= 8)

    # Canonical turnaround run always counts, even without labelled slices.
    if all(s in onsets for s in TURNAROUND_STEPS):
        return True
    # Dense bar with explicit fill-category slices.
    if fill_hits >= 1 and (density >= DENSE_DENSITY_MIN or second_half >= DENSE_SECOND_HALF_MIN):
        return True
    return False


# ---------------------------------------------------------------------------
# 2. Groove pocket: per-step microtiming nudges + velocity humanize
# ---------------------------------------------------------------------------

#: Hardcoded priors in the spirit of
#: `StatisticalGrooveModel.empirical_microtiming` (ms; + = laid-back/late).
#: Kick pushes (-2ms), snare sits back (+4ms), hats swing late (+8ms on
#: off-16ths), bass lays back (+6ms). Everything else plays straight.
KICK_NUDGE_MS = -2.0
SNARE_NUDGE_MS = 4.0
HAT_SWING_NUDGE_MS = 8.0
BASS_NUDGE_MS = 6.0

#: Hard safety bound for any scheduled microtiming offset.
POCKET_BOUND_MS = 15.0

#: Velocity humanize depth (±8% around the written velocity).
HUMANIZE_DEPTH = 0.08


def _clamp_ms(value: float) -> float:
    return max(-POCKET_BOUND_MS, min(POCKET_BOUND_MS, float(value)))


def _is_off_16th(step: int) -> bool:
    """Off-16th = odd 16th slot, plus the offbeat-8th slots (2, 6, 10, 14)."""
    s = step % 16
    return (s % 2 == 1) or (s % 4 == 2)


def pocket_nudge(stem: str, step: int) -> float:
    """Return the schedule-time microtiming offset in milliseconds.

    `stem` is an instrument role ("kick" | "snare" | "hat"/"hats" | "bass" |
    ...) or a mix stem ("drums" | "bass" | "other" | "vocals"). For "drums"
    the canonical dance role of the step decides: 0/8 kick, 4/12 snare,
    2/6/10/14 hats, remaining odd steps hat-ghost swing. Always bounded to
    ±15ms so a bad prior can never flam audibly.
    """
    s = step % 16
    key = str(stem).lower()
    if key == "kick":
        return _clamp_ms(KICK_NUDGE_MS)
    if key == "snare":
        return _clamp_ms(SNARE_NUDGE_MS)
    if key in ("hat", "hats", "hat_closed", "hat_open"):
        return _clamp_ms(HAT_SWING_NUDGE_MS if _is_off_16th(s) else 0.0)
    if key == "bass":
        return _clamp_ms(BASS_NUDGE_MS)
    if key == "drums":
        if s in (0, 8):
            return _clamp_ms(KICK_NUDGE_MS)
        if s in (4, 12):
            return _clamp_ms(SNARE_NUDGE_MS)
        if s in (2, 6, 10, 14) or s % 2 == 1:
            return _clamp_ms(HAT_SWING_NUDGE_MS)
        return 0.0
    return 0.0


def pocket_table(stem: str) -> Dict[int, float]:
    """Full 16-step nudge table for one stem (handy for porting to TS)."""
    return {s: pocket_nudge(stem, s) for s in range(16)}


def humanize_velocity(velocity: float, jitter: float) -> float:
    """Apply ±8% humanize around a written pattern velocity.

    `jitter` is a caller-supplied deterministic value in [-1, 1] (e.g. a
    hashed (bar, step) or an seeded RNG draw) — kept as an argument so this
    stays pure. Output clamped to [0, 1.5] to match the
    `AudioEngine.scheduleSlice` gain clamp.
    """
    j = max(-1.0, min(1.0, float(jitter)))
    return max(0.0, min(1.5, float(velocity) * (1.0 + HUMANIZE_DEPTH * j)))


# ---------------------------------------------------------------------------
# 3. Slice loudness normalization
# ---------------------------------------------------------------------------

#: Reference level for slice gains (LUFS-ish). -14 LUFS matches streaming
#: loudness norms and keeps hot acoustic slices from clipping the mix bus.
TARGET_LUFS = -14.0

#: Safety rails: never attenuate below -6dB (0.5x) or boost above +6dB (2x),
#: mirroring the frontend `scheduleSlice` gain clamp ceiling.
GAIN_MIN = 0.5
GAIN_MAX = 2.0


def loudness_normalize_gain(slice_loud_db: float, target_db: float = TARGET_LUFS) -> float:
    """Map a slice loudness reading (dB, LUFS-ish) to a linear schedule gain.

    gain = 10^((target - slice_loud) / 20), clamped to [0.5, 2.0].
    A slice already at target -> 1.0; a quiet -20 LUFS slice -> ~2.0;
    a hot -8 LUFS slice -> ~0.5. No measurement here — the caller supplies
    whatever loudness estimate it has (`loudness_max`, manifest `loudness`,
    or offline LUFS scan); this is only the mapping.
    """
    gain = 10.0 ** ((float(target_db) - float(slice_loud_db)) / 20.0)
    return max(GAIN_MIN, min(GAIN_MAX, gain))


def normalized_gains(loudness_by_step: Dict[int, float], target_db: float = TARGET_LUFS) -> List[float]:
    """16-step gain vector; steps without a slice entry default to 1.0."""
    return [loudness_normalize_gain(loudness_by_step[s], target_db) if s in loudness_by_step else 1.0 for s in range(16)]
