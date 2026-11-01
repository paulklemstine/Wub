"""Generation-evaluation harness for endless-DJ-mix generation.

Pure-Python (+ numpy, NO torch) metrics that score symbolic bar patterns
produced by a generative music app. The app synthesizes endless DJ mixes
from audio slices of ~1800 analyzed dance songs (4 stems:
drums/bass/other/vocals); these metrics judge a generated mix for musical
quality without rendering any audio.

Canonical symbolic bar (dict) schema
------------------------------------
Only a subset of keys is required per metric; missing keys degrade
gracefully (documented per function):

- ``"section"``: str, e.g. ``"DROP"`` (default ``"GROOVE_A"``).
- ``"onset_grids"``: dict stem -> list[int 0..15] of onset steps. A flat
  ``"onset_grid"`` list is accepted as the drums grid.
- ``"velocities"``: dict stem -> list[float 0..1] (16 steps), or a single
  16-step list, or a scalar. Fallbacks: ``"velocity_curve"``,
  ``"velocity"``, ``"energy"``/``"density"`` scalars, else a binary
  onset-derived curve.
- Pitched (sounding) pitch classes: ``"pitches"`` dict stem -> list[int]
  (only ``bass``/``other``/``vocals`` are read; drums are unpitched and
  ignored), or flat ``"<stem>_pitches"`` / ``"<stem>_pitch"`` keys, or a
  bare ``"root_pitch"`` scalar used as the single sounding pitch when no
  explicit pitched info exists.
- Bar chord: ``"chord"`` / ``"chord_tones"`` / ``"chord_pitches"``
  (list of pitch classes), else ``"chord_root"`` (+ ``"chord_quality"``,
  ``"major"``/``"minor"``) expanded to a triad, else ``"root_pitch"`` as
  chord root, else ``"key"`` (e.g. ``"C major"``) parsed to a triad.
- Identity: ``"bar_id"`` / ``"id"`` / ``"signature"``; when absent a
  canonical signature is derived from onset grids + sounding pitch
  classes (same function used for corpus bars, so mixed comparisons work).

Metric formulas
---------------
- ``harmonic_collision_rate``: fraction of chord-annotated bars where any
  simultaneously-sounding pitched pitch class is a semitone rub (minor
  2nd / major 7th, i.e. circular distance 1, octave-insensitive) away
  from any chord tone.
- ``groove_discontinuity``: mean over adjacent bar pairs of
  ``clip((1 - jaccard(drum grids)) + 0.25*(miss_i + miss_j), 0, 1)``,
  where ``miss`` is the fraction of downbeats ``{0, 4, 8, 12}`` absent
  from that bar's drum grid (so missing downbeats are penalized).
- ``energy_curve_distance``: per-section mean 16-step velocity curves of
  the generated bars vs corpus curves; section MSEs averaged weighted by
  generated bar count.
- ``originality_score``: ``1 - longest_run / len(generated)`` where
  ``longest_run`` is the longest run of consecutively generated bar ids
  each present in the corpus id set (verbatim-copy guard).
- ``evaluate_mix`` goodness (higher = better)::

      goodness = 0.25 * (1 - harmonic_collision_rate)
               + 0.20 * (1 - groove_discontinuity)
               + 0.15 * (1 / (1 + energy_curve_distance))
               + 0.20 * originality_score
               + 0.20 * interlock_score

  Harmony and groove carry the most weight (a clashing or rhythm-broken
  mix is unusable); energy fit matters less (velocity proxies are noisy);
  originality guards against verbatim copying.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "PITCHED_STEMS",
    "DOWNBEATS",
    "SECTIONS",
    "EVAL_WEIGHTS",
    "bar_signature",
    "harmonic_collision_rate",
    "groove_discontinuity",
    "stem_interlock",
    "energy_curve_distance",
    "originality_score",
    "build_corpus_stats",
    "evaluate_mix",
]

PITCHED_STEMS: Tuple[str, ...] = ("bass", "other", "vocals")
ALL_STEMS: Tuple[str, ...] = ("drums", "bass", "other", "vocals")
DOWNBEATS: Tuple[int, ...] = (0, 4, 8, 12)
SECTIONS: Tuple[str, ...] = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
CURVE_LEN = 16
DOWNBEAT_PENALTY = 0.25  # weight of missing-downbeat misses added to (1 - jaccard)

EVAL_WEIGHTS: Dict[str, float] = {
    "harmony": 0.25,
    "groove": 0.20,
    "energy": 0.15,
    "originality": 0.20,
    "interlock": 0.20,
}

_NOTE_TO_PC: Dict[str, int] = {
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


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _popcount(x: int) -> int:
    try:
        return int(x).bit_count()  # type: ignore[attr-defined]
    except AttributeError:  # Python < 3.10
        return bin(int(x)).count("1")


def _as_pc_list(value: Any) -> List[int]:
    """Coerce a scalar / sequence to a list of pitch classes (mod 12)."""
    if value is None:
        return []
    if isinstance(value, str):
        key = value.strip().upper().replace("♯", "#").replace("♭", "B")
        if key in _NOTE_TO_PC:
            return [_NOTE_TO_PC[key]]
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value) % 12]
    if isinstance(value, (float, np.floating)):
        return [int(value) % 12]
    if isinstance(value, (list, tuple, np.ndarray)):
        out: List[int] = []
        for v in list(value):
            try:
                if isinstance(v, str):
                    out.extend(_as_pc_list(v))
                elif v is not None:
                    out.append(int(v) % 12)
            except (TypeError, ValueError):
                continue
        return out
    return []


def _as_step_list(value: Any) -> List[int]:
    """Coerce to sorted unique step indices in 0..15."""
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return [v] if 0 <= v < 16 else []
    try:
        steps = sorted({int(s) % 16 for s in list(value) if s is not None})
    except TypeError:
        return []
    return [s for s in steps if 0 <= s < 16]


def get_onset_grid(bar: Mapping[str, Any], stem: str = "drums") -> List[int]:
    """Return the onset grid for a stem from a symbolic bar dict."""
    grids = bar.get("onset_grids")
    if isinstance(grids, Mapping) and stem in grids:
        return _as_step_list(grids[stem])
    if stem == "drums" and "onset_grid" in bar:
        return _as_step_list(bar.get("onset_grid"))
    for key in (f"{stem}_onset_grid", f"{stem}_grid", f"{stem}_onsets"):
        if key in bar:
            return _as_step_list(bar.get(key))
    return []


def get_sounding_pcs(bar: Mapping[str, Any]) -> List[int]:
    """Union of sounding pitched pitch classes (bass/other/vocals only)."""
    pcs: List[int] = []
    pitches = bar.get("pitches")
    if isinstance(pitches, Mapping):
        for stem in PITCHED_STEMS:
            if stem in pitches:
                pcs.extend(_as_pc_list(pitches[stem]))
    elif isinstance(pitches, (list, tuple, np.ndarray)):
        pcs.extend(_as_pc_list(pitches))
    for stem in PITCHED_STEMS:
        for key in (f"{stem}_pitches", f"{stem}_pitch", f"{stem}_pcs", f"{stem}_pc"):
            if key in bar:
                pcs.extend(_as_pc_list(bar.get(key)))
    for key in ("pitch_classes", "pitch_class", "pcs"):
        if key in bar:
            pcs.extend(_as_pc_list(bar.get(key)))
    if not pcs and "root_pitch" in bar:
        # Bare-root bars: the root pitch is the sounding note.
        pcs.extend(_as_pc_list(bar.get("root_pitch")))
    seen = set()
    uniq = [p for p in pcs if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]
    return uniq


def _triad(root: int, quality: str = "major") -> List[int]:
    root %= 12
    q = (quality or "major").strip().lower()
    third = 3 if q.startswith("min") else 4
    return [root, (root + third) % 12, (root + 7) % 12]


def parse_key_to_chord(key: Any) -> Optional[List[int]]:
    """Parse ``'C major'`` / ``'A# minor'`` style keys to triad tones."""
    if not isinstance(key, str) or not key.strip():
        return None
    parts = key.strip().split()
    note = parts[0].upper().replace("♯", "#").replace("♭", "B")
    if note not in _NOTE_TO_PC:
        return None
    rest = " ".join(parts[1:]).lower()
    quality = "minor" if ("minor" in rest or rest.strip() in ("m", "min")) else "major"
    return _triad(_NOTE_TO_PC[note], quality)


def get_chord_tones(bar: Mapping[str, Any]) -> Optional[List[int]]:
    """Return the bar chord tones (pitch classes) or None if unknown."""
    for key in ("chord", "chord_tones", "chord_pitches"):
        if bar.get(key) is not None:
            tones = _as_pc_list(bar.get(key))
            return tones or None
    if "chord_root" in bar and bar.get("chord_root") is not None:
        try:
            root = int(bar["chord_root"]) % 12
        except (TypeError, ValueError):
            root = None
        if root is not None:
            quality = str(bar.get("chord_quality", bar.get("quality", "major")))
            return _triad(root, quality)
    if "root_pitch" in bar and bar.get("root_pitch") is not None:
        try:
            root = int(bar["root_pitch"]) % 12
            quality = str(bar.get("chord_quality", bar.get("quality", "major")))
            return _triad(root, quality)
        except (TypeError, ValueError):
            pass
    if "root" in bar and isinstance(bar.get("root"), (int, np.integer, float)):
        try:
            return _triad(int(bar["root"]) % 12,
                          str(bar.get("chord_quality", bar.get("quality", "major"))))
        except (TypeError, ValueError):
            pass
    if "key" in bar:
        return parse_key_to_chord(bar.get("key"))
    return None


def bar_velocity_curve(bar: Mapping[str, Any], length: int = CURVE_LEN) -> np.ndarray:
    """Return a length-``length`` velocity curve (float 0..1) for a bar.

    Prefers explicit ``velocities`` (dict stem -> 16 steps, 16-step list,
    or scalar), then ``velocity_curve`` / ``velocity`` / ``energy`` /
    ``density`` scalars broadcast flat, else a binary curve with 1.0 at
    onset steps. Dict input averages across the stems present.
    """
    def _fit(vec: Any) -> Optional[np.ndarray]:
        arr = np.asarray(list(vec), dtype=np.float64).ravel()
        if arr.size == 0:
            return None
        if arr.size == length:
            return arr
        # Resample arbitrary-length curves onto `length` steps.
        old = np.linspace(0.0, 1.0, arr.size)
        new = np.linspace(0.0, 1.0, length)
        return np.interp(new, old, arr)

    v = bar.get("velocities", None)
    if isinstance(v, Mapping):
        curves = []
        for stem_vec in v.values():
            if isinstance(stem_vec, (list, tuple, np.ndarray)):
                c = _fit(stem_vec)
                if c is not None:
                    curves.append(c)
        if curves:
            return np.clip(np.mean(np.stack(curves), axis=0), 0.0, 1.0)
    elif isinstance(v, (list, tuple, np.ndarray)):
        c = _fit(v)
        if c is not None:
            return np.clip(c, 0.0, 1.0)
    elif isinstance(v, (int, float, np.integer, np.floating)):
        return np.clip(float(v), 0.0, 1.0) * np.ones(length)
    for key in ("velocity_curve", "velocity", "energy", "density"):
        if key in bar and isinstance(bar[key], (int, float, np.integer, np.floating)):
            return np.clip(float(bar[key]), 0.0, 1.0) * np.ones(length)
        if key in bar and isinstance(bar[key], (list, tuple, np.ndarray)):
            c = _fit(bar[key])
            if c is not None:
                return np.clip(c, 0.0, 1.0)
    curve = np.zeros(length)
    steps = set()
    grids = bar.get("onset_grids")
    if isinstance(grids, Mapping):
        for g in grids.values():
            steps.update(_as_step_list(g))
    else:
        steps.update(_as_step_list(bar.get("onset_grid", [])))
    for s in steps:
        if 0 <= s < length:
            curve[s] = 1.0
    return curve


def section_of(bar: Mapping[str, Any]) -> str:
    sec = str(bar.get("section", "GROOVE_A") or "GROOVE_A").upper()
    return sec


def bar_signature(bar: Mapping[str, Any]) -> str:
    """Stable bar identity string from onset grids + sounding pitch classes.

    Explicit ``bar_id`` / ``id`` / ``signature`` keys win; otherwise the
    signature is ``drums=..|bass=..|other=..|vocals=..|pcs=..`` with sorted
    steps and pitch classes, so corpus and generated bars share one id
    space for the verbatim-copy check.
    """
    for key in ("bar_id", "id", "signature"):
        if bar.get(key) is not None:
            return str(bar[key])
    parts = []
    for stem in ALL_STEMS:
        parts.append(f"{stem}=" + ",".join(map(str, get_onset_grid(bar, stem))))
    parts.append("pcs=" + ",".join(map(str, sorted(get_sounding_pcs(bar)))))
    return "|".join(parts)


def _grid_mask(steps: Sequence[int]) -> int:
    mask = 0
    for s in steps:
        try:
            s = int(s) % 16
        except (TypeError, ValueError):
            continue
        if 0 <= s < 16:
            mask |= (1 << s)
    return mask


# --------------------------------------------------------------------------
# 1. Harmonic collision rate
# --------------------------------------------------------------------------

def harmonic_collision_rate(bars: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of chord-annotated bars with a semitone rub vs the chord.

    A bar collides when any simultaneously-sounding pitched pitch class
    (union across bass/other/vocals; drums ignored) sits a minor 2nd
    (1 semitone) or major 7th (11 semitones) away from any chord tone,
    octave-insensitive, i.e. circular pitch-class distance == 1.
    Bars without chord info are excluded; bars with a chord but no
    pitched notes count as clean. Empty input (or no chord-annotated
    bars) returns 0.0.
    """
    bars = list(bars or [])
    denom = 0
    hits = 0
    for bar in bars:
        chord = get_chord_tones(bar)
        if not chord:
            continue
        denom += 1
        sounding = get_sounding_pcs(bar)
        if not sounding:
            continue
        ct = np.asarray(sorted({c % 12 for c in chord}), dtype=np.int64)
        sp = np.asarray(sorted({p % 12 for p in sounding}), dtype=np.int64)
        # Circular semitone distance == 1  <=>  (p - c) % 12 in {1, 11}.
        diff = (sp[None, :] - ct[:, None]) % 12
        if bool(np.any((diff == 1) | (diff == 11))):
            hits += 1
    return (hits / denom) if denom else 0.0


# --------------------------------------------------------------------------
# 2. Groove discontinuity
# --------------------------------------------------------------------------

def _bar_interlock(onset: Mapping[str, Sequence[int]]) -> Optional[float]:
    """Kick/bass rhythmic lock for one bar: fraction of bass onsets that
    coincide with drum downbeat (kick-proxy) steps.

    Returns None when the bar has no bass onsets (sparse sections are not
    punished for resting bass). Empty drum grids fall back to the four
    downbeats as the kick grid.
    """
    try:
        bass = sorted(set(int(x) for x in (onset.get("bass") or [])))
    except (TypeError, ValueError):
        return None
    if not bass:
        return None
    try:
        drums = sorted(set(int(x) for x in (onset.get("drums") or [])))
    except (TypeError, ValueError):
        drums = []
    kick = [d for d in drums if d in DOWNBEATS] or list(DOWNBEATS)
    kick_set = set(kick)
    hits = sum(1 for b in bass if b in kick_set)
    return hits / len(bass)


def stem_interlock(
    bars: Sequence[Mapping[str, Any]],
    corpus_stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Score bass/drum rhythmic lock across bars against the corpus norm.

    Returns ``{"rate": mean coincidence, "score": 1/(1+|z|)}`` where z is
    measured against ``corpus_stats["interlock_mean/std"]`` when present
    (otherwise score falls back to the raw rate). Bars without bass are
    skipped; no valid bars -> rate 0.0.
    """
    rates: List[float] = []
    for bar in list(bars or []):
        grids = bar.get("onset_grids") if isinstance(bar, Mapping) else None
        if not isinstance(grids, Mapping):
            continue
        r = _bar_interlock(grids)
        if r is not None:
            rates.append(r)
    if not rates:
        return {"rate": 0.0, "score": 0.0}
    rate = float(np.mean(rates))
    stats = dict(corpus_stats or {})
    mean = stats.get("interlock_mean")
    std = stats.get("interlock_std")
    try:
        if mean is None or std is None or float(std) <= 0:
            return {"rate": rate, "score": max(0.0, min(1.0, rate))}
        z = abs(rate - float(mean)) / float(std)
        return {"rate": rate, "score": float(1.0 / (1.0 + z))}
    except (TypeError, ValueError):
        return {"rate": rate, "score": max(0.0, min(1.0, rate))}


def groove_discontinuity(bars: Sequence[Mapping[str, Any]]) -> float:
    """Mean adjacent-bar drum discontinuity in [0, 1] (lower = steadier).

    Per adjacent pair: ``clip((1 - jaccard) + 0.25*(miss_i + miss_j),
    0, 1)`` with jaccard over the two drum onset grids (two empty grids
    count as identical) and ``miss`` the fraction of downbeats
    ``{0, 4, 8, 12}`` missing from each bar. Identical four-on-the-floor
    bars score 0; disjoint grids with missing downbeats score 1.
    Fewer than 2 bars returns 0.0.
    """
    bars = list(bars or [])
    if len(bars) < 2:
        return 0.0
    masks = [_grid_mask(get_onset_grid(b, "drums")) for b in bars]
    misses = []
    for m in masks:
        missing = sum(1 for d in DOWNBEATS if not (m & (1 << d)))
        misses.append(missing / 4.0)
    scores = []
    for i in range(1, len(masks)):
        a, b = masks[i - 1], masks[i]
        union = a | b
        if union == 0:
            jacc = 1.0
        else:
            jacc = _popcount(a & b) / _popcount(union)
        pair = (1.0 - jacc) + DOWNBEAT_PENALTY * (misses[i - 1] + misses[i])
        scores.append(min(1.0, max(0.0, pair)))
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------
# 3. Energy curve distance
# --------------------------------------------------------------------------

def _normalize_curves_arg(
    curves: Any,
) -> Dict[str, np.ndarray]:
    """Accept a full corpus-stats dict or a plain {section: curve} dict."""
    if isinstance(curves, Mapping) and "section_curves" in curves:
        curves = curves["section_curves"]
    out: Dict[str, np.ndarray] = {}
    if isinstance(curves, Mapping):
        for sec, curve in curves.items():
            try:
                arr = np.asarray(curve, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                continue
            if arr.size == 0:
                continue
            if arr.size == 1:
                arr = np.repeat(arr, CURVE_LEN)
            elif arr.size != CURVE_LEN:
                old = np.linspace(0.0, 1.0, arr.size)
                new = np.linspace(0.0, 1.0, CURVE_LEN)
                arr = np.interp(new, old, arr)
            out[str(sec).upper()] = arr
    return out


def _group_bars(
    generated: Any,
) -> Dict[str, List[Mapping[str, Any]]]:
    """Accept a bar list or a {section: bars} dict; return section groups."""
    if isinstance(generated, Mapping) and not any(
        k in generated for k in ("onset_grid", "onset_grids", "velocities", "chord")
    ):
        groups: Dict[str, List[Mapping[str, Any]]] = {}
        for sec, blist in generated.items():
            try:
                members = list(blist)  # type: ignore[arg-type]
            except TypeError:
                continue
            if members:
                groups[str(sec).upper()] = members
        return groups
    try:
        bars = list(generated or [])
    except TypeError:
        return {}
    groups = {}
    for b in bars:
        if not isinstance(b, Mapping):
            continue
        groups.setdefault(section_of(b), []).append(b)
    return groups


def energy_curve_distance(
    generated_section_bars: Any,
    corpus_curves: Any,
) -> float:
    """MSE between generated and corpus per-section mean velocity curves.

    Both args are flexible: generated is a bar list (each with
    ``"section"``) or a ``{section: bars}`` dict; corpus is a
    ``{section: 16-step curve}`` dict or a full :func:`build_corpus_stats`
    dict (``"section_curves"`` is unwrapped). Per section present in both,
    ``mse = mean((gen_curve - corpus_curve)^2)`` with ``gen_curve`` the
    mean :func:`bar_velocity_curve` over that section's bars; the return
    is the generated-bar-count-weighted mean over sections. No overlapping
    sections (or empty input) returns 0.0.
    """
    groups = _group_bars(generated_section_bars)
    corpus = _normalize_curves_arg(corpus_curves)
    if not groups or not corpus:
        return 0.0
    total_n = 0
    weighted = 0.0
    for sec, blist in groups.items():
        if sec not in corpus or not blist:
            continue
        mat = np.stack([bar_velocity_curve(b) for b in blist], axis=0)
        gen_curve = np.mean(mat, axis=0)
        mse = float(np.mean((gen_curve - corpus[sec]) ** 2))
        weighted += len(blist) * mse
        total_n += len(blist)
    return (weighted / total_n) if total_n else 0.0


# --------------------------------------------------------------------------
# 4. Originality
# --------------------------------------------------------------------------

def _longest_run_vs_set(
    generated_ids: Sequence[str],
    corpus_set: Set[str],
) -> int:
    longest = 0
    run = 0
    for bid in generated_ids:
        if bid in corpus_set:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _longest_run_vs_sequences(
    generated_ids: Sequence[str],
    sequences: Sequence[Sequence[str]],
) -> int:
    """Longest contiguous slice of generated appearing in any corpus song."""
    n = len(generated_ids)
    if n == 0 or not sequences:
        return 0
    best = 0
    for seq in sequences:
        seq = list(seq)
        m = len(seq)
        if m == 0:
            continue
        # Classic longest-common-substring DP with O(m) memory.
        prev = [0] * (m + 1)
        for i in range(1, n + 1):
            cur = [0] * (m + 1)
            gi = generated_ids[i - 1]
            for j in range(1, m + 1):
                if gi == seq[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
                        if best == n:
                            return best
            prev = cur
    return best


def longest_copy_run(
    generated_bar_ids: Sequence[str],
    corpus_bar_ids: Any,
) -> int:
    """Longest exact contiguous match run of generated ids vs the corpus.

    Corpus may be a set/list of single-bar ids (run = longest stretch of
    generated bars each individually present), a list of per-song id
    sequences (run = longest generated slice appearing contiguously in
    any song), or a full :func:`build_corpus_stats` dict (``"bar_ids"``
    is unwrapped).
    """
    gen = [str(x) for x in (generated_bar_ids or [])]
    if not gen:
        return 0
    if isinstance(corpus_bar_ids, Mapping) and "bar_ids" in corpus_bar_ids:
        corpus_bar_ids = corpus_bar_ids["bar_ids"]
    if corpus_bar_ids is None:
        return 0
    items = list(corpus_bar_ids) if not isinstance(corpus_bar_ids, (set, frozenset)) else list(corpus_bar_ids)
    if not items:
        return 0
    if any(isinstance(x, (list, tuple)) and not isinstance(x, str) for x in items):
        seqs: List[List[str]] = []
        singletons: Set[str] = set()
        for x in items:
            if isinstance(x, str):
                singletons.add(x)
            elif isinstance(x, (list, tuple)):
                s = [str(v) for v in x]
                if s:
                    # A 1-tuple is a singleton, not a sequence.
                    if len(s) == 1:
                        singletons.add(s[0])
                    else:
                        seqs.append(s)
            else:
                singletons.add(str(x))
        best = _longest_run_vs_set(gen, singletons) if singletons else 0
        if seqs:
            best = max(best, _longest_run_vs_sequences(gen, seqs))
        return best
    corpus_set = {str(x) for x in items}
    return _longest_run_vs_set(gen, corpus_set)


def originality_score(
    generated_bar_ids: Sequence[str],
    corpus_bar_ids: Any,
) -> float:
    """``1 - longest_run / len(generated)`` in [0, 1] (higher = newer).

    An exact copy of corpus bars scores ~0; fully novel ids score 1.
    Empty generated input or an empty corpus returns 1.0 (nothing to
    call a copy).
    """
    gen = list(generated_bar_ids or [])
    if not gen:
        return 1.0
    run = longest_copy_run(gen, corpus_bar_ids)
    return max(0.0, min(1.0, 1.0 - run / len(gen)))


# --------------------------------------------------------------------------
# 5. Corpus stats
# --------------------------------------------------------------------------

def _heuristic_section(energy: float, density: float) -> str:
    """Per-bar section guess from mean confidence (energy) and density.

    Mirrors the spirit of the PatternBank phrase classifier at bar level:
    loud + dense -> DROP; solid groove -> GROOVE_A; mid -> BUILDUP;
    sparse -> INTRO; near-silent -> BREAKDOWN.
    """
    if energy >= 0.55 and density >= 0.35:
        return "DROP"
    if energy >= 0.35 and density >= 0.20:
        return "GROOVE_A"
    if energy >= 0.25 or density >= 0.12:
        return "BUILDUP"
    if energy >= 0.12 or density >= 0.05:
        return "INTRO"
    return "BREAKDOWN"


def _accumulate_symbolic_bar(bar: Mapping[str, Any], acc: Dict[str, Any]) -> None:
    sec = section_of(bar)
    curve = bar_velocity_curve(bar)
    acc["sec_sums"][sec] = acc["sec_sums"].get(sec, np.zeros(CURVE_LEN)) + curve
    acc["sec_counts"][sec] = acc["sec_counts"].get(sec, 0) + 1
    acc["bar_ids"].add(bar_signature(bar))
    pcs = sorted({p % 12 for p in get_sounding_pcs(bar)})
    if pcs:
        acc["cooc"][np.ix_(pcs, pcs)] += 1
    grids = bar.get("onset_grids") if isinstance(bar, Mapping) else None
    if isinstance(grids, Mapping):
        ilock = acc.setdefault("interlock", {"sum": 0.0, "sumsq": 0.0, "n": 0})
        r = _bar_interlock(grids)
        if r is not None:
            ilock["sum"] += r
            ilock["sumsq"] += r * r
            ilock["n"] += 1
    acc["n_bars"] += 1


def _accumulate_track(track: Any, acc: Dict[str, Any]) -> int:
    """Fold one TrackAnalysis into the accumulators. Returns bars added."""
    tatums = list(getattr(track, "tatums", []) or [])
    segments = list(getattr(track, "segments", []) or [])
    n = len(tatums)
    n_bars = n // CURVE_LEN
    if n_bars <= 0:
        return 0
    tatums = tatums[: n_bars * CURVE_LEN]

    # Segment lookup tables; dominant pitch class via vectorized argmax.
    seg_silent: Dict[str, bool] = {}
    seg_pc: Dict[str, int] = {}
    if segments:
        try:
            pmat = np.asarray([list(s.pitches) for s in segments], dtype=np.float64)
            if pmat.ndim == 2 and pmat.shape[1] >= 12:
                pcs = np.argmax(pmat[:, :12], axis=1) % 12
                for s, pc in zip(segments, pcs.tolist()):
                    seg_pc[s.id] = int(pc)
        except (TypeError, ValueError):
            pass
        for s in segments:
            try:
                seg_silent[s.id] = bool(s.is_silent)
            except AttributeError:
                continue

    # Vectorized velocity proxy: tatum confidence reshaped to (bars, 16).
    try:
        conf = np.array([float(t.confidence) for t in tatums], dtype=np.float64)
    except (TypeError, ValueError, AttributeError):
        conf = np.zeros(n_bars * CURVE_LEN)
    conf = np.clip(conf.reshape(n_bars, CURVE_LEN), 0.0, 1.0)

    # Single Python pass for onsets (segment-change per stem, sounding on
    # the downbeat even across a barline) + pitch sets + active steps.
    onset: Dict[str, List[List[int]]] = {s: [[] for _ in range(n_bars)] for s in ALL_STEMS}
    bar_pcs: List[Set[int]] = [set() for _ in range(n_bars)]
    bar_active: List[Set[int]] = [set() for _ in range(n_bars)]
    prev: Dict[str, Any] = {}
    get_sil = seg_silent.get
    get_pc = seg_pc.get
    for idx, tat in enumerate(tatums):
        b = idx // CURVE_LEN
        step = idx - b * CURVE_LEN
        ss = getattr(tat, "stem_segments", None) or {}
        for stem in ALL_STEMS:
            ids = ss.get(stem)
            if not ids:
                continue
            sid = ids[-1]
            if get_sil(sid, False):
                prev[stem] = sid
                continue
            if step == 0 or sid != prev.get(stem):
                onset[stem][b].append(step)
            prev[stem] = sid
            pc = get_pc(sid)
            if pc is not None:
                bar_pcs[b].add(int(pc))
            bar_active[b].add(step)

    bar_energy = conf.mean(axis=1)
    cooc = acc["cooc"]
    ilock = acc.setdefault("interlock", {"sum": 0.0, "sumsq": 0.0, "n": 0})
    for b in range(n_bars):
        r = _bar_interlock({st: onset[st][b] for st in ALL_STEMS})
        if r is not None:
            ilock["sum"] += r
            ilock["sumsq"] += r * r
            ilock["n"] += 1
        density = len(bar_active[b]) / CURVE_LEN
        sec = _heuristic_section(float(bar_energy[b]), density)
        acc["sec_sums"][sec] = acc["sec_sums"].get(sec, np.zeros(CURVE_LEN)) + conf[b]
        acc["sec_counts"][sec] = acc["sec_counts"].get(sec, 0) + 1
        parts = []
        for stem in ALL_STEMS:
            parts.append(f"{stem}=" + ",".join(map(str, sorted(set(onset[stem][b])))))
        parts.append("pcs=" + ",".join(map(str, sorted(bar_pcs[b]))))
        acc["bar_ids"].add("|".join(parts))
        if bar_pcs[b]:
            idxs = sorted(bar_pcs[b])
            cooc[np.ix_(idxs, idxs)] += 1
    acc["n_bars"] += n_bars
    acc["n_tracks"] += 1
    return n_bars


def _finalize_interlock(acc: Mapping[str, Any]) -> Tuple[float, float]:
    ilock = acc.get("interlock", {}) if isinstance(acc, Mapping) else {}
    try:
        n = int(ilock.get("n", 0) or 0)
        if n <= 0:
            return 0.5, 0.25
        mean = float(ilock.get("sum", 0.0)) / n
        var = max(0.0, float(ilock.get("sumsq", 0.0)) / n - mean * mean)
        return mean, max(1e-3, float(np.sqrt(var)))
    except (TypeError, ValueError):
        return 0.5, 0.25


def build_corpus_stats(tracks: Any) -> Dict[str, Any]:
    """Aggregate per-section energy curves + bar ids + pitch co-occurrence.

    Accepts an iterable of ``TrackAnalysis`` (or anything with
    ``.tatums`` / ``.segments``), plain symbolic bar dicts, one-level
    nested song -> bars lists, or an object exposing ``.tracks``
    (e.g. a CorpusStore). Returns::

        {"section_curves": {section: [16 mean velocities]},
         "section_mean_density": {section: float},   # reserved, NaN-safe
         "section_bar_counts": {section: int},
         "bar_ids": set[str],
         "pitch_cooccurrence": [[int]] (12x12),
         "n_tracks": int, "n_bars": int}

    Performance: per track the confidence matrix is handled with numpy
    (reshape + mean over a (bars, 16) array; segment pitch argmax over a
    (segments, 12) matrix) plus a single linear pass over tatums for
    onsets — O(total tatums) with small constants, comfortably under
    10 minutes for 1800 tracks in a single process.
    """
    if hasattr(tracks, "tracks") and not hasattr(tracks, "tatums"):
        try:
            tracks = tracks.tracks  # CorpusStore-like
        except TypeError:
            pass
    try:
        items = list(tracks or [])
    except TypeError:
        items = [tracks]

    acc: Dict[str, Any] = {
        "sec_sums": {},
        "sec_counts": {},
        "bar_ids": set(),
        "cooc": np.zeros((12, 12), dtype=np.int64),
        "n_tracks": 0,
        "n_bars": 0,
    }
    for item in items:
        if item is None:
            continue
        if hasattr(item, "tatums"):
            _accumulate_track(item, acc)
        elif isinstance(item, Mapping):
            if "tatums" in item and isinstance(item.get("tatums"), (list, tuple)):
                # Serialized track-ish dict; not enough symbolic info to
                # do better than skipping (no segment loudness map).
                continue
            _accumulate_symbolic_bar(item, acc)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                if hasattr(sub, "tatums"):
                    _accumulate_track(sub, acc)
                elif isinstance(sub, Mapping):
                    _accumulate_symbolic_bar(sub, acc)
                # deeper nesting is not a documented input; ignore.
            acc["n_tracks"] += 1
        # unknown element types are ignored (documented leniency).

    curves: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    for sec, total in acc["sec_sums"].items():
        c = int(acc["sec_counts"].get(sec, 0))
        counts[sec] = c
        curves[sec] = (np.asarray(total, dtype=np.float64) / c).tolist() if c else [0.0] * CURVE_LEN
    return {
        "section_curves": curves,
        "section_mean_density": {sec: float("nan") for sec in curves},
        "section_bar_counts": counts,
        "bar_ids": set(acc["bar_ids"]),
        "pitch_cooccurrence": np.asarray(acc["cooc"], dtype=np.int64).tolist(),
        "n_tracks": int(acc["n_tracks"]),
        "n_bars": int(acc["n_bars"]),
        "interlock_mean": _finalize_interlock(acc)[0],
        "interlock_std": _finalize_interlock(acc)[1],
        "interlock_bars": int(acc.get("interlock", {}).get("n", 0)),
    }


# --------------------------------------------------------------------------
# 6. Full mix report
# --------------------------------------------------------------------------

def evaluate_mix(
    bars: Sequence[Mapping[str, Any]],
    corpus_stats: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Score a generated mix; returns a report dict plus ``goodness``.

    ``corpus_stats`` is a :func:`build_corpus_stats` dict (its
    ``section_curves`` feed the energy term and ``bar_ids`` the
    originality term); ``None``/empty is treated as "no corpus" (energy
    distance 0, originality 1). Report keys: ``n_bars``,
    ``harmonic_collision_rate``, ``groove_discontinuity``,
    ``energy_curve_distance``, ``energy_score`` (``1/(1+mse)``),
    ``originality_score``, ``longest_copy_run``, ``goodness``.
    """
    bars = list(bars or [])
    stats = dict(corpus_stats or {})
    hcr = harmonic_collision_rate(bars)
    gd = groove_discontinuity(bars)
    ecd = energy_curve_distance(bars, stats)
    energy_score = 1.0 / (1.0 + ecd)
    il = stem_interlock(bars, stats)
    gen_ids = [bar_signature(b) for b in bars]
    run = longest_copy_run(gen_ids, stats.get("bar_ids", set()))
    orig = 1.0 - (run / len(gen_ids)) if gen_ids else 1.0
    w = EVAL_WEIGHTS
    goodness = (
        w["harmony"] * (1.0 - hcr)
        + w["groove"] * (1.0 - gd)
        + w["energy"] * energy_score
        + w["originality"] * orig
        + w["interlock"] * il["score"]
    )
    return {
        "n_bars": len(bars),
        "harmonic_collision_rate": float(hcr),
        "groove_discontinuity": float(gd),
        "energy_curve_distance": float(ecd),
        "energy_score": float(energy_score),
        "originality_score": float(max(0.0, min(1.0, orig))),
        "longest_copy_run": int(run),
        "interlock_rate": float(il["rate"]),
        "interlock_score": float(il["score"]),
        "goodness": float(max(0.0, min(1.0, goodness))),
    }
