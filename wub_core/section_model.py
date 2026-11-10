"""Expanded section learning and transition model for infinite navigation.

Derives a richer, feature-learned section vocabulary from corpus bar features
and mines real-song transition structure. Pure/data-facing logic shared by the
build script and tests.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BASE_SECTIONS = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"]

EXPANDED = [
    "INTRO",
    "WARMUP",
    "GROOVE_LOW",
    "GROOVE",
    "GROOVE_HIGH",
    "BUILDUP",
    "RISE",
    "DROP",
    "PEAK",
    "BREAKDOWN",
    "BRIDGE",
    "OUTRO",
]

# Map every expanded section to the 5-class base the ranker/arc models know.
BASE_OF = {
    "INTRO": "INTRO",
    "WARMUP": "GROOVE_A",
    "GROOVE_LOW": "GROOVE_A",
    "GROOVE": "GROOVE_A",
    "GROOVE_HIGH": "GROOVE_A",
    "BUILDUP": "BUILDUP",
    "RISE": "BUILDUP",
    "DROP": "DROP",
    "PEAK": "DROP",
    "BREAKDOWN": "BREAKDOWN",
    "BRIDGE": "BREAKDOWN",
    "OUTRO": "BREAKDOWN",
}

BASE_INDEX = {s: i for i, s in enumerate(BASE_SECTIONS)}

# Musical target energies (0..1) for navigation - the engine's intensity scale.
# These are priors blended with learned relative intensity from data.
TARGET_ENERGY_PRIOR = {
    "INTRO": 0.28,
    "WARMUP": 0.42,
    "GROOVE_LOW": 0.50,
    "GROOVE": 0.68,
    "GROOVE_HIGH": 0.78,
    "BUILDUP": 0.62,
    "RISE": 0.82,
    "DROP": 0.92,
    "PEAK": 0.96,
    "BREAKDOWN": 0.30,
    "BRIDGE": 0.45,
    "OUTRO": 0.22,
}


def _slope(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    y = np.asarray(values, dtype=np.float64)
    denom = float(np.dot(x, x))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def span_descriptor(span, feats: List[Dict[str, Any]], n_bars_total: int) -> Dict[str, float]:
    run = feats[span.start_bar:span.end_bar]
    if not run:
        return {
            "energy_db": -30.0, "density": 0.0, "drums": 0.0, "bass": 0.0,
            "vocals": 0.0, "other": 0.0, "harm": 0.0, "e_slope": 0.0,
            "d_slope": 0.0, "pos": 0.0, "end_frac": 0.0, "bars": float(getattr(span, 'n_bars', 4)),
            "base": str(getattr(span, 'label', 'GROOVE_A')),
        }
    energies = [float(f.get("energy", -30.0)) for f in run]
    dens = [float(f.get("total_density", 0.0)) for f in run]
    drums = [float(f.get("densities", {}).get("drums", 0.0)) for f in run]
    bass = [float(f.get("densities", {}).get("bass", 0.0)) for f in run]
    vocals = [float(f.get("densities", {}).get("vocals", 0.0)) for f in run]
    other = [float(f.get("densities", {}).get("other", 0.0)) for f in run]
    harm = [float(f.get("harmonic_change", 0.0)) for f in run]
    pos = span.start_bar / max(1, n_bars_total - 1)
    end_frac = span.end_bar / max(1, n_bars_total)
    return {
        "energy_db": float(np.mean(energies)),
        "density": float(np.mean(dens)),
        "drums": float(np.mean(drums)),
        "bass": float(np.mean(bass)),
        "vocals": float(np.mean(vocals)),
        "other": float(np.mean(other)),
        "harm": float(np.mean(harm)),
        "e_slope": _slope(energies),
        "d_slope": _slope(dens),
        "pos": float(pos),
        "end_frac": float(end_frac),
        "bars": float(span.n_bars),
        "base": str(span.label),
    }


def _rank_normalize(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return []
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
    return ranks.tolist()


def classify_expanded(desc: Dict[str, float], th: Dict[str, float]) -> str:
    """Classify a span into an expanded section using data-calibrated thresholds.

    Uses within-corpus-calibrated density/slope thresholds plus position and the
    base heuristic label. Energy is treated via relative-friendly rules because
    raw dB energy saturates; density/drums/slope/position carry the signal.
    """
    base = str(desc.get("base", "GROOVE_A"))
    d = float(desc.get("density", 0.05))
    drums = float(desc.get("drums", 0.0))
    e_slope = float(desc.get("e_slope", 0.0))
    d_slope = float(desc.get("d_slope", 0.0))
    pos = float(desc.get("pos", 0.5))
    end_frac = float(desc.get("end_frac", 0.5))
    vocals = float(desc.get("vocals", 0.0))
    erank = float(desc.get("energy_rank", 0.5))
    drank = float(desc.get("density_rank", 0.5))

    slope_up = th.get("slope_rise", 0.05)
    slope_down = th.get("slope_fall", -0.05)

    # Outro: very near the end, not a peak/drop moment.
    if end_frac >= 0.93 and base in ("BREAKDOWN", "GROOVE_A", "INTRO") and erank < 0.75:
        return "OUTRO"

    # Intro/warmup at start.
    if pos <= 0.12:
        if base == "INTRO" or (erank <= 0.35 and drank <= 0.45):
            return "INTRO"
        if erank <= 0.55 and drank <= 0.55:
            return "WARMUP"

    if base == "INTRO":
        return "INTRO" if erank <= 0.45 else "WARMUP"

    if base == "DROP":
        if erank >= 0.75 or drank >= th.get("d_p85", 0.18) or drums >= th.get("dr_p75", 0.08):
            return "PEAK"
        return "DROP"

    if base == "BUILDUP":
        rising = e_slope >= slope_up or d_slope >= th.get("d_slope_rise", 0.002)
        # Strong tension rise / late pre-drop
        if (rising and erank >= 0.45) or erank >= 0.75 or drank >= th.get("d_p75", 0.14):
            return "RISE"
        # Early or low buildup
        if pos <= 0.2 and erank <= 0.5:
            return "BUILDUP"
        return "RISE" if rising else "BUILDUP"

    if base == "BREAKDOWN":
        if end_frac >= 0.88 and erank < 0.6:
            return "OUTRO"
        # A contrasting mid-song bridge keeps some harmonic/body presence.
        if 0.2 <= pos <= 0.8 and (erank >= 0.35 or drank >= th.get("d_p33", 0.04)):
            if drums <= th.get("dr_p33", 0.02) and erank >= 0.3:
                return "BRIDGE"
        return "BREAKDOWN"

    # GROOVE_A refined.
    if erank >= 0.82 or (drank >= th.get("d_p85", 0.18) and erank >= 0.6):
        return "GROOVE_HIGH"
    if erank <= 0.28 and drank <= th.get("d_p33", 0.04):
        return "GROOVE_LOW"
    if e_slope >= slope_up and erank >= 0.55:
        return "GROOVE_HIGH"
    if e_slope <= slope_down and erank <= 0.45:
        return "GROOVE_LOW"
    # Vocal-forward mid sections read as groove/verse space.
    if vocals >= th.get("v_p66", 0.02) and 0.35 <= erank <= 0.75:
        return "GROOVE"
    if erank >= 0.55 or drank >= th.get("d_p66", 0.1):
        return "GROOVE"
    return "GROOVE_LOW"


def learn_thresholds(descriptors: List[Dict[str, float]]) -> Dict[str, float]:
    if not descriptors:
        return {
            "d_p33": 0.04, "d_p50": 0.07, "d_p66": 0.10, "d_p75": 0.14, "d_p85": 0.18,
            "dr_p33": 0.02, "dr_p66": 0.06, "dr_p75": 0.08,
            "v_p66": 0.02, "slope_rise": 0.05, "slope_fall": -0.05, "d_slope_rise": 0.002,
        }
    def pct(arr, p):
        a = np.asarray(arr, float)
        return float(np.percentile(a, p)) if len(a) else 0.0
    d = [x["density"] for x in descriptors]
    dr = [x["drums"] for x in descriptors]
    v = [x["vocals"] for x in descriptors]
    es = [x["e_slope"] for x in descriptors]
    ds = [x["d_slope"] for x in descriptors]
    es_pos = [x for x in es if x > 0]
    es_neg = [x for x in es if x < 0]
    ds_pos = [x for x in ds if x > 0]
    return {
        "d_p33": pct(d, 33), "d_p50": pct(d, 50), "d_p66": pct(d, 66),
        "d_p75": pct(d, 75), "d_p85": pct(d, 85),
        "dr_p33": pct(dr, 33), "dr_p66": pct(dr, 66), "dr_p75": pct(dr, 75),
        "v_p66": pct(v, 66),
        "slope_rise": max(0.02, pct(es_pos, 60) if es_pos else 0.05),
        "slope_fall": min(-0.02, pct(es_neg, 40) if es_neg else -0.05),
        "d_slope_rise": max(0.0005, pct(ds_pos, 60) if ds_pos else 0.002),
    }


def build_transition_model(sequences: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    trans: Dict[str, Counter] = defaultdict(Counter)
    starts: Counter = Counter()
    durations: Dict[str, List[int]] = defaultdict(list)
    rel_energy: Dict[str, List[float]] = defaultdict(list)
    section_counts: Counter = Counter()
    trans3: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for seq in sequences:
        if not seq:
            continue
        starts[seq[0]["section"]] += 1
        for i, item in enumerate(seq):
            s = item["section"]
            section_counts[s] += 1
            durations[s].append(int(item.get("bars", 4)))
            rel_energy[s].append(float(item.get("intensity", TARGET_ENERGY_PRIOR.get(s, 0.5))))
            if i + 1 < len(seq):
                n = seq[i + 1]["section"]
                trans[s][n] += 1
                ctx = seq[i - 1]["section"] if i - 1 >= 0 else "<START>"
                trans3[(ctx, s)][n] += 1

    def norm_counter(c: Counter, min_p: float = 0.01, max_items: int = 10):
        total = sum(c.values()) or 1
        items = [{"to": k, "p": v / total, "n": int(v)} for k, v in c.most_common()]
        filtered = [it for it in items if it["p"] >= min_p][:max_items]
        if not filtered and items:
            filtered = items[:3]
        s = sum(it["p"] for it in filtered) or 1.0
        out = []
        for it in filtered:
            out.append({"to": it["to"], "p": round(it["p"] / s, 5), "n": it["n"]})
        return out

    transitions = {s: norm_counter(c) for s, c in trans.items()}
    transitions3 = {f"{a}|{b}": norm_counter(c) for (a, b), c in trans3.items()}
    start_probs = norm_counter(starts, min_p=0.005)

    # Infinite-navigation adjustments: real songs end at OUTRO, but an infinite
    # set must continue. Learn continuations and also add musically valid
    # re-entries so the navigator never dead-ends.
    if "OUTRO" in transitions:
        # Keep learned short-tail but ensure strong re-entry options.
        reentries = [
            {"to": "INTRO", "p": 0.20, "n": 0},
            {"to": "WARMUP", "p": 0.15, "n": 0},
            {"to": "GROOVE_LOW", "p": 0.25, "n": 0},
            {"to": "BREAKDOWN", "p": 0.25, "n": 0},
            {"to": "GROOVE", "p": 0.15, "n": 0},
        ]
        # blend: 40% learned, 60% infinite re-entry (learned outro->... exists rarely)
        learned = transitions["OUTRO"]
        blended: Dict[str, float] = defaultdict(float)
        for it in learned:
            blended[it["to"]] += 0.4 * float(it["p"])
        for it in reentries:
            blended[it["to"]] += 0.6 * float(it["p"])
        transitions["OUTRO"] = norm_counter(Counter({k: v for k, v in blended.items()}), min_p=0.02)

    # Ensure every expanded section has at least one outgoing transition.
    fallbacks = {
        "INTRO": "WARMUP", "WARMUP": "GROOVE", "GROOVE_LOW": "GROOVE",
        "GROOVE": "GROOVE_HIGH", "GROOVE_HIGH": "RISE", "BUILDUP": "RISE",
        "RISE": "PEAK", "DROP": "PEAK", "PEAK": "BREAKDOWN",
        "BREAKDOWN": "GROOVE_LOW", "BRIDGE": "GROOVE", "OUTRO": "GROOVE_LOW",
    }
    for s in EXPANDED:
        if s not in transitions or not transitions[s]:
            fb = fallbacks.get(s, "GROOVE")
            transitions[s] = [{"to": fb, "p": 1.0, "n": 0}]

    dur_stats = {}
    learned_target = {}
    for s in EXPANDED:
        vals = durations.get(s, [])
        ev = rel_energy.get(s, [])
        if vals:
            arr = np.array(vals, float)
            # snap durations to 4-bar multiples for phrase alignment
            med = int(np.median(arr))
            med = max(4, min(16, int(round(med / 4.0) * 4)))
            dur_stats[s] = {
                "mean_bars": round(float(arr.mean()), 2),
                "std_bars": round(float(arr.std()), 2),
                "median_bars": med,
                "min_bars": max(4, int(arr.min())) if len(arr) else 4,
                "max_bars": min(32, int(arr.max())) if len(arr) else 16,
            }
        else:
            dur_stats[s] = {"mean_bars": 8, "std_bars": 4, "median_bars": 8, "min_bars": 4, "max_bars": 16}
        # Blend learned relative intensity with musical prior.
        prior = TARGET_ENERGY_PRIOR.get(s, 0.5)
        if ev:
            learned_e = float(np.mean(ev))
            learned_target[s] = round(_clamp01(0.6 * prior + 0.4 * learned_e), 3)
        else:
            learned_target[s] = prior

    # Top arc motifs for reference/debug and template seeding.
    arc_counter: Counter = Counter()
    for seq in sequences:
        labels = [x["section"] for x in seq]
        if 4 <= len(labels) <= 16:
            arc_counter[tuple(labels)] += 1
        for w in (4, 5, 6):
            for i in range(0, max(0, len(labels) - w + 1)):
                arc_counter[tuple(labels[i:i + w])] += 1
    top_arcs = [{"sections": list(k), "n": int(v)} for k, v in arc_counter.most_common(60)]

    expanded_for_base: Dict[str, List[str]] = defaultdict(list)
    for e, b in BASE_OF.items():
        expanded_for_base[b].append(e)

    return {
        "format": "wub-section-model-v1",
        "version": 1,
        "expanded_sections": EXPANDED,
        "base_sections": BASE_SECTIONS,
        "base_of": BASE_OF,
        "base_index": BASE_INDEX,
        "expanded_for_base": dict(expanded_for_base),
        "target_energy": learned_target,
        "durations": dur_stats,
        "transitions": transitions,
        "transitions3": transitions3,
        "start_probs": start_probs,
        "top_arcs": top_arcs,
        "stats": {
            "tracks": 0,  # filled by caller
            "total_spans": int(sum(section_counts.values())),
            "section_counts": {k: int(v) for k, v in section_counts.items()},
        },
    }
