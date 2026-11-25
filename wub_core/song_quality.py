"""Song-quality scoring for generated arrangements (reference-free).

Scores a symbolic song (bars of 16-step onset grids + per-step pitch classes
+ velocities) on five musical axes, 0..100 total:

- groove (25): kick downbeats present, bass on offbeats, steady same-section flow
- harmony (25): no semitone rubs, chord-tone membership, stepwise key motion
- arrangement (20): section energy arc, fills at boundaries, section durations
- repetition (15): motif reprise inside a healthy band (neither random nor stuck)
- mix (15): per-stem level balance, transient pileup avoidance

This is a *proxy* objective, not truth: weights live in WEIGHTS and must be
calibrated against blind A/B votes (see docs/SONG_QUALITY.md). Until the
metric agrees with human winners >=70% of pairs, use it for measurement
only -- not for generation decisions.

Bar dict format (constructed by scripts/eval_song_quality.py from
global_patterns prototypes + global vocab metadata):
{
  "section": "DROP",
  "chord_root": 0,            # pitch class 0..11
  "chord_minor": True,
  "onset_grids": {"drums": [0, 4, ...], "bass": [...], "other": [...], "vocals": [...]},
  "step_pitches": {"bass": {step: pc}, "other": {step: pc}, "vocals": {step: pc}},
  "velocities": {"drums": [16 floats], ...},
}
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wub_core.generation_eval import groove_discontinuity

STEMS = ("drums", "bass", "other", "vocals")
DOWNBEATS = (0, 4, 8, 12)
OFFBEATS = (2, 6, 10, 14)

WEIGHTS = {"groove": 25.0, "harmony": 25.0, "arrangement": 20.0, "repetition": 15.0, "mix": 15.0}

SECTION_ENERGY = {"INTRO": 0.25, "GROOVE_A": 0.83, "BUILDUP": 0.65, "DROP": 0.90, "BREAKDOWN": 0.20}
SECTION_DURATION_MEAN = {"INTRO": 8.0, "GROOVE_A": 16.0, "BUILDUP": 8.0, "DROP": 16.0, "BREAKDOWN": 8.0}
STEM_LEVEL = {"drums": 1.0, "bass": 0.85, "other": 0.75, "vocals": 0.70}


def _circ(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % 12
    return min(d, 12 - d)


def _triad(root: int, minor: bool) -> List[int]:
    r = int(root) % 12
    return [r, (r + 3 if minor else r + 4) % 12, (r + 7) % 12]


def _scale(root: int, minor: bool) -> List[int]:
    r = int(root) % 12
    steps = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    return sorted({(r + s) % 12 for s in steps})


def _grids(bar: Mapping[str, Any]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    grids = bar.get("onset_grids") or {}
    for s in STEMS:
        try:
            out[s] = sorted({int(x) % 16 for x in (grids.get(s) or [])})
        except (TypeError, ValueError):
            out[s] = []
    return out


def _vel(bar: Mapping[str, Any], stem: str, step: int, default: float = 0.9) -> float:
    try:
        v = (bar.get("velocities") or {}).get(stem)
        if v is not None:
            return float(v[step % 16])
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    return default


def _sounding_pcs(bar: Mapping[str, Any], step: int, span_cap: int = 8) -> List[int]:
    """Pitch classes sounding on this step: onsets here + sustained tails.

    Sustain = an earlier onset whose span (distance to next onset, capped)
    still covers this step. Mirrors the tatum HOLD convention.
    """
    grids = _grids(bar)
    sp = bar.get("step_pitches") or {}
    pcs: List[int] = []
    for stem in ("bass", "other", "vocals"):
        onsets = grids[stem]
        if not onsets:
            continue
        pitches = sp.get(stem) or {}
        # onset exactly here
        if step in onsets:
            try:
                pcs.append(int(pitches[step]) % 12)
                continue
            except (TypeError, ValueError, KeyError):
                pass
        # sustained tail?
        prev = [o for o in onsets if o < step]
        if not prev:
            continue
        o = prev[-1]
        nxt = next((x for x in onsets if x > o), 16)
        if step - o < min(span_cap, nxt - o):
            try:
                pcs.append(int(pitches[o]) % 12)
            except (TypeError, ValueError, KeyError):
                pass
    seen: List[int] = []
    for p in pcs:
        if p not in seen:
            seen.append(p)
    return seen


def _bar_energy(bar: Mapping[str, Any]) -> float:
    """Event-rate x intensity: mean onset velocity scaled by total density.

    A sparse INTRO of loud hits still reads quiet; a dense DROP reads hot.
    Calibrated so typical sections land near SECTION_ENERGY targets."""
    grids = _grids(bar)
    vals: List[float] = []
    for s in STEMS:
        for st in grids[s]:
            vals.append(_vel(bar, s, st))
    if not vals:
        return 0.0
    total_onsets = len(vals)
    return float(np.mean(vals)) * (0.3 + 0.7 * min(1.0, total_onsets / 24.0))


def score_groove(bars: Sequence[Mapping[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Kick downbeats + offbeat bass + steady same-section flow."""
    bars = list(bars)
    if not bars:
        return 0.0, {"downbeats": 0.0, "offbeat_bass": 0.0, "continuity": 0.0}
    downs, offs = [], []
    for bar in bars:
        grids = _grids(bar)
        sec = str(bar.get("section", "GROOVE_A"))
        if sec in ("DROP", "GROOVE_A"):
            hit = sum(1 for d in DOWNBEATS if d in grids["drums"])
            downs.append(hit / 4.0)
        if grids["bass"]:
            on = sum(1 for b in grids["bass"] if b in OFFBEATS)
            offs.append(on / max(1, len(grids["bass"])))
    downbeats = float(np.mean(downs)) if downs else 0.9  # sparse sections don't punish
    offbeat = float(np.mean(offs)) if offs else 1.0  # resting bass is fine
    # continuity per same-section run (a section change legitimately breaks flow)
    conts: List[float] = []
    run: List[Mapping[str, Any]] = []
    for bar in bars + [{}]:
        sec = str(bar.get("section", "")) if bar else None
        if run and sec != str(run[-1].get("section", "")):
            conts.append(1.0 - groove_discontinuity(
                [{"onset_grids": {s: _grids(b)[s] for s in STEMS}} for b in run]))
            run = []
        if bar:
            run.append(bar)
    continuity = float(np.mean(conts)) if conts else 1.0
    score = 0.5 * downbeats + 0.3 * offbeat + 0.2 * continuity
    return score, {"downbeats": downbeats, "offbeat_bass": offbeat, "continuity": continuity}


def score_harmony(bars: Sequence[Mapping[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """No semitone rubs, chord-tone membership, stepwise key motion."""
    bars = list(bars)
    if not bars:
        return 0.0, {"no_rubs": 0.0, "chord_tones": 0.0, "key_drift": 0.0}
    rub_free, member = [], []
    for bar in bars:
        root = int(bar.get("chord_root", 0)) % 12
        minor = bool(bar.get("chord_minor", True))
        tones = set(_triad(root, minor))
        scale = set(_scale(root, minor))
        steps = 0
        clean = 0
        mem_hit = mem_tot = 0
        for step in range(16):
            pcs = _sounding_pcs(bar, step)
            if not pcs:
                continue
            steps += 1
            rub = any(_circ(p, t) == 1 for p in pcs for t in tones)
            if not rub:
                clean += 1
            for p in pcs:
                mem_tot += 1
                if p in tones or p in scale:
                    mem_hit += 1
        rub_free.append(clean / steps if steps else 1.0)
        member.append(mem_hit / mem_tot if mem_tot else 1.0)
    drifts = []
    for a, b in zip(bars, bars[1:]):
        try:
            d = _circ(int(a.get("chord_root", 0)), int(b.get("chord_root", 0)))
        except (TypeError, ValueError):
            d = 0
        same_sec = str(a.get("section")) == str(b.get("section"))
        if same_sec:
            drifts.append(1.0 if d <= 2 else max(0.0, 1.0 - (d - 2) / 4.0))
        else:
            drifts.append(1.0)  # section changes earn a fresh key
    no_rubs = float(np.mean(rub_free))
    chord_tones = float(np.mean(member))
    key_drift = float(np.mean(drifts)) if drifts else 1.0
    score = 0.5 * no_rubs + 0.3 * chord_tones + 0.2 * key_drift
    return score, {"no_rubs": no_rubs, "chord_tones": chord_tones, "key_drift": key_drift}


def _section_spans(bars: Sequence[Mapping[str, Any]]) -> List[Tuple[str, int, int]]:
    """(section, start_idx, length) runs."""
    spans: List[Tuple[str, int, int]] = []
    for i, bar in enumerate(bars):
        sec = str(bar.get("section", "GROOVE_A"))
        if spans and spans[-1][0] == sec:
            s, st, ln = spans[-1]
            spans[-1] = (s, st, ln + 1)
        else:
            spans.append((sec, i, 1))
    return spans


def score_arrangement(
    bars: Sequence[Mapping[str, Any]],
    duration_means: Optional[Mapping[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Energy arc match, fills at boundaries, section durations."""
    bars = list(bars)
    if not bars:
        return 0.0, {"energy_arc": 0.0, "boundary_fills": 0.0, "durations": 0.0}
    energies = [_bar_energy(b) for b in bars]
    # a1: section mean energy vs target
    by_sec: Dict[str, List[float]] = {}
    for b, e in zip(bars, energies):
        by_sec.setdefault(str(b.get("section", "GROOVE_A")), []).append(e)
    arc = [1.0 - min(1.0, abs(float(np.mean(v)) - SECTION_ENERGY.get(s, 0.65)) * 2.0) for s, v in by_sec.items()]
    energy_arc = float(np.mean(arc)) if arc else 0.0
    # a2: boundary fills = above-average drum density on section-final bars
    spans = _section_spans(bars)
    fills = []
    for sec, st, ln in spans:
        if st + ln >= len(bars):
            continue  # song end: no transition to prepare
        sec_mean = float(np.mean([_drum_density(bars[j]) for j in range(st, st + ln)])) if ln else 0.0
        last = _drum_density(bars[st + ln - 1])
        fills.append(1.0 if last >= sec_mean + 0.10 else 0.5)
    boundary = float(np.mean(fills)) if fills else 0.75
    # a3: durations vs mined means (gaussian falloff)
    durs = duration_means or SECTION_DURATION_MEAN
    dscores = []
    for sec, _st, ln in spans:
        mean = float(durs.get(sec, 8.0))
        std = max(1.0, mean / 2.0)
        z = abs(ln - mean) / std
        dscores.append(float(np.exp(-0.5 * z * z)))
    durations = float(np.mean(dscores)) if dscores else 1.0
    score = 0.45 * energy_arc + 0.30 * boundary + 0.25 * durations
    return score, {"energy_arc": energy_arc, "boundary_fills": boundary, "durations": durations}


def _drum_density(bar: Mapping[str, Any]) -> float:
    return len(_grids(bar)["drums"]) / 16.0


def _bar_sig(bar: Mapping[str, Any]) -> Tuple:
    grids = _grids(bar)
    return tuple(sorted(grids[s]) for s in STEMS)


def score_repetition(bars: Sequence[Mapping[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Motif reprise inside a healthy band: exact bar repeats at lag 4/8.

    Target match rate in [0.15, 0.6]: lower = random mush, higher = stuck loop.
    """
    bars = list(bars)
    sigs = [_bar_sig(b) for b in bars]
    rates = []
    for lag in (4, 8):
        if len(sigs) <= lag:
            continue
        hits = sum(1 for i in range(lag, len(sigs)) if sigs[i] == sigs[i - lag])
        rates.append(hits / (len(sigs) - lag))
    rate = float(np.mean(rates)) if rates else 0.0
    if 0.15 <= rate <= 0.6:
        score = 1.0
    elif rate < 0.15:
        score = max(0.0, rate / 0.15)
    else:
        score = max(0.0, (1.0 - rate) / 0.4)
    return score, {"reprise_rate": rate}


def score_mix(bars: Sequence[Mapping[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Per-stem level balance + transient pileup avoidance + harshness control."""
    bars = list(bars)
    if not bars:
        return 0.0, {"balance": 0.0, "no_pileup": 0.0, "smoothness": 0.0}
    levels: Dict[str, List[float]] = {s: [] for s in STEMS}
    for bar in bars:
        grids = _grids(bar)
        for s in STEMS:
            if grids[s]:
                levels[s].append(float(np.mean([_vel(bar, s, st) for st in grids[s]])))
    present = {s: (float(np.mean(v)) if v else None) for s, v in levels.items()}
    ref = present.get("drums") or 0.9
    parts = []
    for s in STEMS:
        if present[s] is None:
            continue  # silent stem: neutral, not punished
        tgt = STEM_LEVEL[s] * ref
        parts.append(1.0 - min(1.0, abs(present[s] - tgt)))
    balance = float(np.mean(parts)) if parts else 1.0
    # pileup: steps with >=3 simultaneous onsets at vel>=0.9
    pile, active = 0, 0
    centroids: List[float] = []
    for bar in bars:
        grids = _grids(bar)
        cents = bar.get("step_centroids") or {}
        for step in range(16):
            firing = [s for s in STEMS if step in grids[s] and _vel(bar, s, step) >= 0.9]
            if firing:
                active += 1
                if len(firing) >= 3:
                    pile += 1
            for s in STEMS:
                if step in grids[s]:
                    try:
                        centroids.append(float(cents.get(s, {}).get(step, float("nan"))))
                    except (TypeError, ValueError, AttributeError):
                        pass
    no_pileup = 1.0 - min(1.0, (pile / max(1, active)) * 2.0)
    import math as _math
    finite = [c for c in centroids if _math.isfinite(c)]
    if finite:
        mean_c = float(np.mean(finite))
        # Warm but present: full marks 600Hz..4.5kHz, linear falloff to
        # dull (<200Hz) or harsh (>9kHz).
        if mean_c < 600:
            smoothness = max(0.0, (mean_c - 200) / 400.0)
        elif mean_c <= 4500:
            smoothness = 1.0
        else:
            smoothness = max(0.0, (9000 - mean_c) / 4500.0)
    else:
        smoothness = 1.0  # no centroid info: neutral, not punished
    score = 0.55 * balance + 0.20 * no_pileup + 0.25 * smoothness
    return score, {"balance": balance, "no_pileup": no_pileup, "smoothness": smoothness}


def score_song(
    bars: Sequence[Mapping[str, Any]],
    duration_means: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Full 0..100 song score with per-axis breakdown."""
    bars = list(bars)
    g, gd = score_groove(bars)
    h, hd = score_harmony(bars)
    a, ad = score_arrangement(bars, duration_means)
    r, rd = score_repetition(bars)
    m, md = score_mix(bars)
    total = (
        WEIGHTS["groove"] * g + WEIGHTS["harmony"] * h + WEIGHTS["arrangement"] * a
        + WEIGHTS["repetition"] * r + WEIGHTS["mix"] * m
    )
    return {
        "score": round(total, 1),
        "groove": round(g * 100, 1), "groove_detail": {k: round(v, 3) for k, v in gd.items()},
        "harmony": round(h * 100, 1), "harmony_detail": {k: round(v, 3) for k, v in hd.items()},
        "arrangement": round(a * 100, 1), "arrangement_detail": {k: round(v, 3) for k, v in ad.items()},
        "repetition": round(r * 100, 1), "repetition_detail": {k: round(v, 3) for k, v in rd.items()},
        "mix": round(m * 100, 1), "mix_detail": {k: round(v, 3) for k, v in md.items()},
        "bars": len(bars),
    }


def score_bar(
    bar: Mapping[str, Any],
    prev: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Single-bar score for live reranking (groove + harmony + mix only).

    Pass the previous bar for continuity/drift context; otherwise the bar
    is judged standalone.
    """
    ctx = [prev, bar] if prev else [bar]
    g, gd = score_groove(ctx)
    h, hd = score_harmony(ctx)
    m, md = score_mix([bar])
    total = WEIGHTS["groove"] * g + WEIGHTS["harmony"] * h + WEIGHTS["mix"] * m
    denom = WEIGHTS["groove"] + WEIGHTS["harmony"] + WEIGHTS["mix"]
    return {
        "score": round(100.0 * total / denom, 1),
        "groove": round(g * 100, 1), "groove_detail": {k: round(v, 3) for k, v in gd.items()},
        "harmony": round(h * 100, 1), "harmony_detail": {k: round(v, 3) for k, v in hd.items()},
        "mix": round(m * 100, 1), "mix_detail": {k: round(v, 3) for k, v in md.items()},
    }
