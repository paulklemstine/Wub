#!/usr/bin/env python3
"""Phase 0 flow evaluation for Wub sets.

This is the missing instrument: it scores a played (or synthesised) set against
**real music statistics**, not against a pairwise AUC.  It is deliberately
model-free so it can be run before, during and after the BarLM / flow-critic
upgrade described in ``docs/FLOW_UPGRADE.md``.

Two artifacts:

``wub-flow-reference-v1``
    Aggregate statistics mined once from ``data/processed/*.json`` with
    ``wub_core.section_arcs``: section bigram/trigram counts, per-section
    duration percentiles, per-bar energy/density lag profile (structural
    self-similarity), energy envelopes and per-track metric samples.

``wub-set-trace-v1``
    One row per **bar** of a played set.  The browser recorder
    (``frontend/src/set_trace.ts``) emits this; ``--demo-trace`` writes a
    synthetic one so the scorer can be smoke-tested without a session.

Metrics (all reported with a percentile against the real-track sample):

* ``form``       mean trigram log-probability of the section sequence.
* ``durations``  fraction of section runs inside the corpus p10..p90 band.
* ``structure``  correlation / RMSE of the bar-to-bar similarity profile at
                 lags 1,2,4,8,16,32 versus real tracks.  This is the metric
                 that separates "a song with a groove" from "constant shuffle":
                 real tracks are self-similar at 8/16 bars and varied at 4.
* ``sources``    per-stem bars-per-source-change and phrase-use entropy.  The
                 live mode target is slow change (16-32 bars) with high entropy;
                 a 4-bar swap rate fails here.
* ``energy``     dynamic range, envelope RMSE and DTW distance to the nearest
                 real-track energy arc.
* ``seams``      p95 max sample jump and RMS-dip rate from the transition probe.

Usage::

    # 1. mine the reference once (laptop; 1.7k analyses live here)
    python3 scripts/eval_flow.py --build-reference \
        --processed data/processed --max-tracks 160 \
        --out /tmp/opencode/flow_reference.json

    # 2. score a trace exported from the browser
    python3 scripts/eval_flow.py --trace /tmp/opencode/set_trace.json \
        --reference /tmp/opencode/flow_reference.json \
        --out /tmp/opencode/set_report.json

    # 3. smoke test without any corpus
    python3 scripts/eval_flow.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TRACE_FORMAT = "wub-set-trace-v1"
REFERENCE_FORMAT = "wub-flow-reference-v1"
BASE_SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
STEMS = ("drums", "bass", "other", "vocals")
LAGS = (1, 2, 4, 8, 16, 32)
ARC_POINTS = 64

# Starting gates.  Phase 0 calibrates these: the target is "inside the real
# distribution", so most gates are percentiles rather than absolutes.
# The structure metric is DIAGNOSTIC in Phase 0: the 5-d bar vector measures
# arrangement stability (it is too coarse to prove motif return).  It becomes a
# hard gate once traces carry onset masks + timbre (Phase 1), so only a
# genuinely inverted profile fails here.
DEFAULT_GATES: Dict[str, float] = {
    "form_percentile": 20.0,
    "structure_corr": -0.25,
    "structure_rmse": 1.00,
    "duration_in_band": 0.70,
    "energy_percentile": 20.0,
    "min_bars_per_source_change": 8.0,
    "max_bars_per_source_change": 96.0,
    "seam_jump_p95": 0.45,
    "seam_dips_per_min": 1.0,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def base_section(label: Any) -> str:
    """Map any expanded / song-role / legacy label onto the 5 base sections."""
    text = str(label or "GROOVE_A").strip().upper().replace("-", "_")
    if text in BASE_SECTIONS:
        return text
    if not text:
        return "GROOVE_A"
    if text.startswith("GROOVE") or text.startswith("VERSE") or text.startswith("WARMUP"):
        return "INTRO" if text.startswith("WARMUP") else "GROOVE_A"
    if text.startswith("BUILD") or text.startswith("RISE"):
        return "BUILDUP"
    if text.startswith("DROP") or text.startswith("CHORUS") or text.startswith("PEAK"):
        return "DROP"
    if text.startswith("BREAK") or text.startswith("BRIDGE") or text.startswith("OUTRO"):
        return "BREAKDOWN"
    return "GROOVE_A"


def energy01(value: Any) -> float:
    """Accept 0..1 mix energy or a dB value; return 0..1."""
    e = _finite(value, -60.0)
    if e < -1.0:
        return clamp01((e + 60.0) / 60.0)
    return clamp01(e)


def _stem_payload(bar: Dict[str, Any], stem: str) -> Dict[str, Any]:
    stems = bar.get("stems")
    if not isinstance(stems, dict):
        return {}
    payload = stems.get(stem)
    return payload if isinstance(payload, dict) else {}


def bar_vector(bar: Dict[str, Any]) -> np.ndarray:
    """5-d comparable bar vector: energy + per-stem onset density.

    Density is onsets/16 in both worlds: real bars from
    ``section_arcs.bar_features`` and engine bars from the phrase's 16-step
    onset mask.  When a trace omits density the active flag is used as a coarse
    fallback so the scorer still runs.
    """
    vec = [energy01(bar.get("energy", bar.get("target_energy", 0.0)))]
    for stem in STEMS:
        payload = _stem_payload(bar, stem)
        density = payload.get("density")
        if density is None:
            active = payload.get("active")
            density = 0.5 if active else 0.0
        vec.append(clamp01(_finite(density, 0.0)))
    return np.asarray(vec, dtype=np.float64)


def trace_matrix(bars: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.stack([bar_vector(bar) for bar in bars]) if bars else np.zeros((0, 5))


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score columns so structure comparison is scale-invariant."""
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 0.0
    safe = np.where(std > 0.0, std, 1.0)
    out = (matrix - mean) / safe
    out[:, std <= 0.0] = 0.0
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def lag_profile(matrix: np.ndarray, lags: Sequence[int] = LAGS,
                sections: Optional[Sequence[str]] = None) -> Dict[int, float]:
    """Mean self-similarity at each lag.

    With ``sections``, only pairs inside the same section are counted: that is
    where a real track's motif actually returns.  Across a section boundary the
    profile is expected to drop.
    """
    std = standardize(matrix)
    n = std.shape[0]
    profile: Dict[int, float] = {}
    for lag in lags:
        if n - lag < 4:
            continue
        if sections is None:
            pairs = range(n - lag)
        else:
            pairs = [i for i in range(n - lag)
                     if sections[i] == sections[i + lag]]
            if len(pairs) < 4:
                continue
        vals = [cosine(std[i], std[i + lag]) for i in pairs]
        profile[int(lag)] = float(np.mean(vals)) if vals else 0.0
    return profile


def _profile_vector(profile: Dict[Any, float], lags: Sequence[int] = LAGS) -> np.ndarray:
    return np.asarray([_finite(profile.get(str(lag), profile.get(lag, 0.0))) for lag in lags],
                      dtype=np.float64)


def structure_metrics(profile: Dict[int, float],
                      ref_profile: Dict[str, float]) -> Tuple[float, float]:
    """Correlation and RMSE between the trace lag profile and the corpus mean."""
    a = _profile_vector(profile)
    b = _profile_vector(ref_profile)
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        corr = 1.0 if np.allclose(a, b) else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return corr, rmse


def section_runs(bars: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
    runs: List[Tuple[str, int]] = []
    for bar in bars:
        label = base_section(bar.get("section", bar.get("kind")))
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


def section_sequence(bars: Sequence[Dict[str, Any]]) -> List[str]:
    return [label for label, _ in section_runs(bars)]


def ngram_counts(sequences: Iterable[Sequence[str]], order: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for seq in sequences:
        for n in range(order, len(seq) + 1):
            key = ">".join(seq[n - order:n])
            counts[key] = counts.get(key, 0) + 1
    return counts


def ngram_logprob(seq: Sequence[str],
                  tables: Dict[int, Dict[str, int]],
                  vocab: Sequence[str],
                  order: int = 3,
                  alpha: float = 0.5) -> float:
    """Mean log-probability per transition with backoff to shorter orders."""
    if len(seq) < 2:
        return -math.log(max(len(vocab), 2))
    total = 0.0
    steps = 0
    for i in range(1, len(seq)):
        used = 0
        prob = 1.0 / max(len(vocab), 2)
        for size in range(min(order, i + 1), 1, -1):
            ctx = ">".join(seq[i - size + 1:i])
            key = ">".join(seq[i - size + 1:i + 1])
            table = tables.get(size)
            if not table:
                continue
            ctx_count = sum(c for k, c in table.items() if k.rsplit(">", 1)[0] == ctx)
            if ctx_count == 0:
                continue
            prob = (table.get(key, 0) + alpha) / (ctx_count + alpha * len(vocab))
            used = size
            break
        total += math.log(max(prob, 1e-9))
        steps += 1
    return total / max(1, steps)


def duration_fit(runs: Sequence[Tuple[str, int]],
                 durations: Dict[str, Dict[str, float]]) -> Tuple[float, int]:
    """Fraction of known section runs inside the corpus p10..p90 band."""
    checked = 0
    inside = 0
    for label, n_bars in runs:
        band = durations.get(label)
        if not band:
            continue
        checked += 1
        if band["p10"] <= n_bars <= band["p90"]:
            inside += 1
    if checked == 0:
        return 1.0, 0
    return inside / checked, checked


def source_stats(bars: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-stem source-change rate and phrase-use entropy."""
    out: Dict[str, Dict[str, float]] = {}
    for stem in STEMS:
        tracks: List[Optional[str]] = []
        phrases: List[str] = []
        for bar in bars:
            payload = _stem_payload(bar, stem)
            active = payload.get("active", bool(payload.get("phrase")))
            track = payload.get("track") if active else None
            tracks.append(str(track) if track else None)
            if active and payload.get("phrase"):
                phrases.append(str(payload["phrase"]))
        active_bars = sum(1 for item in tracks if item)
        changes = 0
        prev: Optional[str] = None
        for item in tracks:
            if item is None:
                continue
            if prev is not None and item != prev:
                changes += 1
            prev = item
        bars_per_change = (active_bars / changes) if changes else float(max(active_bars, 0))
        entropy = 0.0
        distinct = len(set(phrases))
        if phrases:
            counts: Dict[str, int] = {}
            for phrase in phrases:
                counts[phrase] = counts.get(phrase, 0) + 1
            total = float(len(phrases))
            entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
            if distinct > 1:
                entropy /= math.log(distinct)
        out[stem] = {
            "changes": float(changes),
            "bars_per_change": float(bars_per_change),
            "distinct": float(len(set(t for t in tracks if t))),
            "phrase_entropy": clamp01(entropy),
        }
    return out


def resample(seq: Sequence[float], points: int = ARC_POINTS) -> np.ndarray:
    arr = np.asarray([_finite(v) for v in seq], dtype=np.float64)
    if arr.size == 0:
        return np.zeros(points, dtype=np.float64)
    if arr.size == 1:
        return np.full(points, arr[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, arr.size)
    dst = np.linspace(0.0, 1.0, points)
    return np.interp(dst, src, arr)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised DTW on 1-d sequences (lower is closer)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0
    inf = float("inf")
    prev = np.full(m + 1, inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, inf, dtype=np.float64)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m] / float(n + m))


def energy_metrics(bars: Sequence[Dict[str, Any]],
                   ref: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    energy = [energy01(bar.get("energy", bar.get("target_energy", 0.0))) for bar in bars]
    if not energy:
        return {"range": 0.0, "nearest_dtw": None, "envelope_rmse": None}
    arr = np.asarray(energy, dtype=np.float64)
    out: Dict[str, Any] = {
        "mean": float(arr.mean()),
        "range": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
        "nearest_dtw": None,
        "envelope_rmse": None,
    }
    if not ref:
        return out
    arc = resample(energy)
    envelope = ref.get("energy_envelope") or {}
    mean_env = envelope.get("mean")
    if isinstance(mean_env, list) and len(mean_env) == ARC_POINTS:
        out["envelope_rmse"] = float(
            np.sqrt(np.mean((arc - np.asarray(mean_env, dtype=np.float64)) ** 2)))
    exemplars = ref.get("energy_arcs") or []
    if exemplars:
        out["nearest_dtw"] = float(min(
            dtw_distance(arc, np.asarray(ex, dtype=np.float64)) for ex in exemplars))
    return out


def seam_metrics(trace: Dict[str, Any], n_bars: int) -> Dict[str, float]:
    seams = trace.get("seams") or []
    jumps = [abs(_finite(s.get("max_jump"))) for s in seams if isinstance(s, dict)]
    dips = [abs(_finite(s.get("rms_dip_db"))) for s in seams if isinstance(s, dict)]
    minutes = max(1e-6, n_bars / 128.0)  # 128 bars/min at 128 BPM, 4/4
    return {
        "boundaries": float(len(seams)),
        "jump_p50": float(np.percentile(jumps, 50)) if jumps else 0.0,
        "jump_p95": float(np.percentile(jumps, 95)) if jumps else 0.0,
        "dip_events": float(sum(1 for d in dips if d > 1.5)),
        "dips_per_min": float(sum(1 for d in dips if d > 1.5) / minutes),
    }


def percentile(value: Optional[float], samples: Sequence[float],
               higher_is_better: bool = True) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray([s for s in samples if s is not None and np.isfinite(s)], dtype=np.float64)
    if arr.size == 0:
        return None
    if higher_is_better:
        return float(100.0 * np.mean(arr <= value))
    return float(100.0 * np.mean(arr >= value))


# ---------------------------------------------------------------------------
# Trace scoring
# ---------------------------------------------------------------------------

def score_trace(trace: Dict[str, Any],
                ref: Optional[Dict[str, Any]] = None,
                gates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    gates = {**DEFAULT_GATES, **(gates or {})}
    bars = trace.get("bars") or []
    if not bars:
        raise ValueError("trace has no bars")
    matrix = trace_matrix(bars)
    labels = [base_section(bar.get("section", bar.get("kind"))) for bar in bars]
    profile = lag_profile(matrix)
    within = lag_profile(matrix, sections=labels)
    runs = section_runs(bars)
    seq = [label for label, _ in runs]
    sources = source_stats(bars)
    energy = energy_metrics(bars, ref)
    seams = seam_metrics(trace, len(bars))

    report: Dict[str, Any] = {
        "format": TRACE_FORMAT,
        "mode": trace.get("mode", "unknown"),
        "bars": len(bars),
        "blocks": len({bar.get("block") for bar in bars if bar.get("block") is not None}),
        "section_runs": [{"section": label, "bars": n} for label, n in runs],
        "form": {"sequence": seq},
        "structure": {"profile": {str(k): v for k, v in profile.items()}},
        "structure_within": {"profile": {str(k): v for k, v in within.items()}},
        "sources": sources,
        "energy": energy,
        "seams": seams,
    }

    verdict_reasons: List[str] = []
    ok = True

    if ref:
        tables = {
            int(order): table
            for order, table in (ref.get("ngram") or {}).items()
            if str(order).isdigit()
        }
        vocab = ref.get("vocab") or list(BASE_SECTIONS)
        ll = ngram_logprob(seq, tables, vocab, order=3)
        report["form"]["ll"] = ll
        report["form"]["percentile"] = percentile(ll, ref.get("metrics", {}).get("form_ll", []))
        pct = report["form"]["percentile"]
        if pct is not None and pct < gates["form_percentile"]:
            ok = False
            verdict_reasons.append(f"form likelihood at p{pct:.0f} of real tracks")

        corr, rmse = structure_metrics(profile, ref.get("lag_profile") or {})
        report["structure"]["corr"] = corr
        report["structure"]["rmse"] = rmse
        report["structure"]["percentile"] = percentile(
            corr, ref.get("metrics", {}).get("structure_corr", []))
        within_ref = ref.get("lag_profile_within_section")
        if within_ref:
            wcorr, wrmse = structure_metrics(within, within_ref)
            report["structure_within"]["corr"] = wcorr
            report["structure_within"]["rmse"] = wrmse
            report["structure_within"]["percentile"] = percentile(
                wcorr, ref.get("metrics", {}).get("structure_within_corr", []))
        if corr < gates["structure_corr"]:
            ok = False
            verdict_reasons.append(f"structure correlation {corr:.2f} < {gates['structure_corr']:.2f}")
        if rmse > gates["structure_rmse"]:
            ok = False
            verdict_reasons.append(f"structure rmse {rmse:.2f} > {gates['structure_rmse']:.2f}")

        fit, checked = duration_fit(runs, ref.get("duration_bars") or {})
        report["durations"] = {"in_band": fit, "checked": checked}
        if checked >= 2 and fit < gates["duration_in_band"]:
            ok = False
            verdict_reasons.append(f"only {fit:.0%} of section durations inside real p10-p90")

        report["energy"]["percentile"] = percentile(
            energy.get("nearest_dtw"), ref.get("metrics", {}).get("energy_dtw", []),
            higher_is_better=False)
        pct = report["energy"]["percentile"]
        if pct is not None and pct < gates["energy_percentile"]:
            ok = False
            verdict_reasons.append(f"energy arc at p{pct:.0f} of real tracks")

        fast_stem = min(
            ((stem, stats["bars_per_change"]) for stem, stats in sources.items()
             if stats["changes"] > 0),
            key=lambda item: item[1], default=None)
        slow_stem = max(
            ((stem, stats["bars_per_change"]) for stem, stats in sources.items()
             if stats["changes"] > 0),
            key=lambda item: item[1], default=None)
        report["sources"]["fastest"] = fast_stem[0] if fast_stem else None
        if fast_stem and fast_stem[1] < gates["min_bars_per_source_change"]:
            ok = False
            verdict_reasons.append(
                f"{fast_stem[0]} source changes every {fast_stem[1]:.1f} bars "
                f"(< {gates['min_bars_per_source_change']:.0f})")
        if slow_stem and slow_stem[1] > gates["max_bars_per_source_change"]:
            verdict_reasons.append(
                f"{slow_stem[0]} holds {slow_stem[1]:.0f} bars (glacial, not a failure)")

    if seams["jump_p95"] > gates["seam_jump_p95"]:
        ok = False
        verdict_reasons.append(f"seam jump p95 {seams['jump_p95']:.2f} > {gates['seam_jump_p95']:.2f}")
    if seams["dips_per_min"] > gates["seam_dips_per_min"]:
        ok = False
        verdict_reasons.append(f"{seams['dips_per_min']:.1f} level dips/min > {gates['seam_dips_per_min']:.1f}")

    report["verdict"] = {"pass": ok, "reasons": verdict_reasons}
    return report


# ---------------------------------------------------------------------------
# Reference mining (real tracks)
# ---------------------------------------------------------------------------

def _track_from_processed(path: Path) -> Optional[Dict[str, Any]]:
    from wub_core.section_arcs import bar_features, segment_sections

    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    segments = data.get("segments") or []
    tatums = data.get("tatums") or []
    if len(tatums) < 16 * 48:
        return None
    feats = bar_features(segments, tatums)
    if len(feats) < 48:
        return None
    spans = segment_sections(feats)
    labels = ["GROOVE_A"] * len(feats)
    for span in spans:
        for bar in range(span.start_bar, min(len(labels), span.end_bar)):
            labels[bar] = base_section(span.label)
    matrix = np.asarray([
        [energy01(f["energy"])] + [clamp01(_finite(f["densities"].get(s, 0.0))) for s in STEMS]
        for f in feats
    ], dtype=np.float64)
    energy = matrix[:, 0]
    runs: List[Tuple[str, int]] = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return {
        "track_id": str(data.get("track_id") or path.stem),
        "bpm": _finite(data.get("bpm"), 0.0),
        "key": str(data.get("key") or ""),
        "bars": len(feats),
        "labels": labels,
        "runs": runs,
        "matrix": matrix,
        "energy": energy,
    }


def _pct(values: Sequence[float], points: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p)}": 0.0 for p in points}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in points}


def build_reference(processed_dir: Path,
                    max_tracks: int = 160,
                    seed: int = 505,
                    exemplars: int = 32) -> Dict[str, Any]:
    paths = sorted(processed_dir.glob("*.json"))
    random.Random(seed).shuffle(paths)
    limit = len(paths) if max_tracks <= 0 else min(len(paths), max_tracks)
    tracks: List[Dict[str, Any]] = []
    for path in paths[:limit]:
        track = _track_from_processed(path)
        if track is not None:
            tracks.append(track)
        if len(tracks) >= limit:
            break
    if len(tracks) < 3:
        raise SystemExit(f"not enough usable tracks under {processed_dir} ({len(tracks)})")

    sequences = [[label for label, _ in t["runs"]] for t in tracks]
    ngram = {2: ngram_counts(sequences, 2), 3: ngram_counts(sequences, 3)}

    duration_samples: Dict[str, List[float]] = {}
    for track in tracks:
        for label, n_bars in track["runs"]:
            duration_samples.setdefault(label, []).append(float(n_bars))
    durations = {
        label: {
            **_pct(values, (10, 50, 90)),
            "mean": float(np.mean(values)),
            "n": len(values),
        }
        for label, values in duration_samples.items()
    }

    # Aggregate lag profile: mean/std over tracks, globally and within sections.
    profiles = [lag_profile(t["matrix"]) for t in tracks]
    within_profiles = [lag_profile(t["matrix"], sections=t["labels"]) for t in tracks]
    lag_mean: Dict[str, float] = {}
    lag_std: Dict[str, float] = {}
    within_mean: Dict[str, float] = {}
    for lag in LAGS:
        vals = [p[lag] for p in profiles if lag in p]
        lag_mean[str(lag)] = float(np.mean(vals)) if vals else 0.0
        lag_std[str(lag)] = float(np.std(vals)) if vals else 0.0
        wvals = [p[lag] for p in within_profiles if lag in p]
        within_mean[str(lag)] = float(np.mean(wvals)) if wvals else 0.0

    arcs = [resample(t["energy"]) for t in tracks]
    envelope_mean = np.mean(np.stack(arcs), axis=0)
    envelope_std = np.std(np.stack(arcs), axis=0)

    # Per-track metric samples (the percentile population).
    form_ll: List[float] = []
    structure_corr: List[float] = []
    structure_rmse: List[float] = []
    structure_within_corr: List[float] = []
    energy_dtw: List[float] = []
    energy_range: List[float] = []
    for i, track in enumerate(tracks):
        ll = ngram_logprob([label for label, _ in track["runs"]],
                           ngram, list(BASE_SECTIONS), order=3)
        corr, rmse = structure_metrics(profiles[i], lag_mean)
        wcorr, _ = structure_metrics(within_profiles[i], within_mean)
        nearest = min(
            (dtw_distance(arcs[i], arcs[j]) for j in range(len(arcs)) if j != i),
            default=None)
        form_ll.append(ll)
        structure_corr.append(corr)
        structure_rmse.append(rmse)
        structure_within_corr.append(wcorr)
        if nearest is not None:
            energy_dtw.append(nearest)
        energy_range.append(float(np.percentile(track["energy"], 95)
                                  - np.percentile(track["energy"], 5)))

    rng = random.Random(seed + 1)
    chosen = list(range(len(arcs)))
    rng.shuffle(chosen)
    exemplar_arcs = [arcs[i].tolist() for i in chosen[:max(1, min(exemplars, len(arcs)))]]

    return {
        "format": REFERENCE_FORMAT,
        "version": 1,
        "tracks": len(tracks),
        "vocab": list(BASE_SECTIONS),
        "ngram": {str(order): table for order, table in ngram.items()},
        "duration_bars": durations,
        "lag_profile": lag_mean,
        "lag_profile_std": lag_std,
        "lag_profile_within_section": within_mean,
        "energy_envelope": {
            "mean": envelope_mean.tolist(),
            "std": envelope_std.tolist(),
            "points": ARC_POINTS,
        },
        "energy_arcs": exemplar_arcs,
        "metrics": {
            "form_ll": form_ll,
            "structure_corr": structure_corr,
            "structure_rmse": structure_rmse,
            "structure_within_corr": structure_within_corr,
            "energy_dtw": energy_dtw,
            "energy_range": energy_range,
        },
        "samples": {
            "bpm": _pct([t["bpm"] for t in tracks if t["bpm"] > 0], (10, 50, 90)),
            "bars": _pct([float(t["bars"]) for t in tracks], (10, 50, 90)),
        },
    }


# ---------------------------------------------------------------------------
# Demo trace (smoke test / baseline shape)
# ---------------------------------------------------------------------------

def demo_trace(mode: str = "arc", bars: int = 104, seed: int = 505) -> Dict[str, Any]:
    """Synthetic traces for the three shapes worth distinguishing.

    ``arc``    the *current* shipped 104-bar shape (4 -> 8 -> 16 + bridge).
               Expected to fail form/duration: that is the baseline finding.
    ``song``   a corpus-shaped form (several short GROOVE_A/DROP sections).
    ``churn``  a 4-bar source swap, the failure mode the source gate catches.
    """
    rng = random.Random(seed)
    plans = {
        "arc": [("INTRO", 16), ("GROOVE_A", 32), ("BUILDUP", 16),
                ("DROP", 32), ("BREAKDOWN", 8)],
        "song": [("INTRO", 8), ("GROOVE_A", 16), ("DROP", 16), ("BREAKDOWN", 8),
                 ("GROOVE_A", 24), ("DROP", 16), ("BREAKDOWN", 16)],
    }
    plan = plans["song"] if mode == "song" else plans["arc"]
    energy_target = {
        "INTRO": 0.33, "GROOVE_A": 0.72, "BUILDUP": 0.84,
        "DROP": 0.95, "BREAKDOWN": 0.30,
    }
    section_of_bar: List[str] = []
    for label, n_bars in plan:
        section_of_bar.extend([label] * n_bars)

    def section_for(bar: int) -> str:
        if not section_of_bar:
            return "GROOVE_A"
        return section_of_bar[min(bar, len(section_of_bar) - 1)]

    pattern_bars = 24 if mode != "churn" else 4
    # The engine swaps at 4-bar phrase boundaries, so density is piecewise
    # constant within a phrase.  Real bars fluctuate more; that gap is itself a
    # Phase 1 finding.
    base = {"drums": 0.62, "bass": 0.40, "other": 0.34, "vocals": 0.22}
    pattern_density = dict(base)

    pattern_id = 0
    pattern_start = 0
    source = {stem: f"t{rng.randrange(6)}_{stem}" for stem in STEMS}
    phrase = {stem: f"p{rng.randrange(8)}_{stem}" for stem in STEMS}
    rows: List[Dict[str, Any]] = []
    for bar in range(bars):
        if bar - pattern_start >= pattern_bars:
            pattern_id += 1
            pattern_start = bar
            if mode != "churn":
                moved = rng.sample(list(STEMS), k=rng.choice([1, 2]))
                for stem in moved:
                    source[stem] = f"t{rng.randrange(6)}_{stem}"
                    phrase[stem] = f"p{rng.randrange(8)}_{stem}"
            else:
                for stem in STEMS:
                    source[stem] = f"t{rng.randrange(40)}_{stem}"
                    phrase[stem] = f"p{rng.randrange(8)}_{stem}"
            pattern_density = {
                stem: clamp01(base[stem] + rng.uniform(-0.10, 0.10))
                for stem in STEMS
            }
        sec = section_for(bar)
        tgt = energy_target[sec]
        # Bar-level variation inside a stable pattern: fills, dropouts.
        fill = (bar + 1) % 8 == 0
        active = {stem: True for stem in STEMS}
        if sec == "BREAKDOWN" and bar % 16 == 0:
            active["drums"] = False
            active["bass"] = False
        stems = {}
        for stem in STEMS:
            density = 0.0
            if active[stem]:
                if mode == "churn":
                    density = clamp01(base[stem] + rng.uniform(-0.40, 0.40))
                else:
                    density = clamp01(pattern_density[stem] + rng.uniform(-0.01, 0.01)
                                      + (0.12 if (fill and stem == "drums") else 0.0))
            stems[stem] = {
                "phrase": phrase[stem],
                "track": source[stem],
                "active": active[stem],
                "density": density,
            }
        energy = clamp01(tgt + (rng.uniform(-0.01, 0.01) if mode != "churn"
                                else rng.uniform(-0.25, 0.25))
                         + (0.03 if fill else 0.0))
        rows.append({
            "bar": bar,
            "block": bar // 4,
            "pattern": pattern_id,
            "section": sec,
            "target_energy": tgt,
            "energy": energy,
            "stems": stems,
            "changed": [],
        })
    seams = []
    for bar in range(16, bars, 8):
        seams.append({
            "bar": bar,
            "max_jump": round(abs(rng.gauss(0.10, 0.06)), 4),
            "rms_dip_db": round(abs(rng.gauss(0.4, 0.4)), 3),
            "flux": round(abs(rng.gauss(0.12, 0.05)), 4),
        })
    return {
        "format": TRACE_FORMAT,
        "version": 1,
        "mode": mode,
        "bars": rows,
        "seams": seams,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> List[str]:
    checks: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    check(base_section("GROOVE_HIGH") == "GROOVE_A", "expanded groove maps to base")
    check(base_section("CHORUS_2") == "DROP", "song chorus maps to drop")
    check(base_section("OUTRO_BRIDGE") == "BREAKDOWN", "outro bridge maps to breakdown")
    check(abs(energy01(-60.0)) < 1e-6, "dB floor maps to 0")
    check(abs(energy01(0.0) - 0.0) < 1e-6, "0 dB maps to 0 (dB branch)")

    # A song-like pattern repeats every 8 bars; a churn trace does not.
    song = demo_trace("song", bars=104, seed=7)
    churn = demo_trace("churn", bars=104, seed=7)
    song_profile = lag_profile(trace_matrix(song["bars"]))
    churn_profile = lag_profile(trace_matrix(churn["bars"]))
    check(song_profile.get(8, 0.0) > churn_profile.get(8, 0.0),
          "song-like trace is more self-similar at 8 bars than churn")

    ref = {
        "vocab": list(BASE_SECTIONS),
        "ngram": {
            "2": ngram_counts([["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"]], 2),
            "3": ngram_counts([["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"]], 3),
        },
        "duration_bars": {label: {"p10": 8, "p50": 16, "p90": 32} for label in BASE_SECTIONS},
        "lag_profile": song_profile,
        "energy_arcs": [resample([energy01(b["energy"]) for b in song["bars"]]).tolist()],
        "metrics": {
            "form_ll": [-2.0, -1.5, -1.2],
            "structure_corr": [0.5, 0.7, 0.9],
            "structure_rmse": [0.05, 0.1, 0.2],
            "energy_dtw": [0.01, 0.05, 0.1],
        },
    }
    good = score_trace(song, ref)
    bad = score_trace(churn, ref)
    check(good["structure"]["corr"] > bad["structure"]["corr"],
          "scorer ranks the song-like trace higher on structure")
    check(bad["verdict"]["pass"] is False, "churn trace fails the source gate")
    check(good["form"]["percentile"] is not None, "form percentile is reported")

    # periodicity: an 8-bar loop should be strongly self-similar at lag 8.
    loop = [{"energy": 0.5 + 0.02 * (i % 8), "section": "DROP",
             "stems": {s: {"phrase": f"p{i % 8}", "track": "t0", "active": True,
                           "density": 0.2 + 0.05 * ((i + len(s)) % 8)} for s in STEMS}}
            for i in range(64)]
    profile = lag_profile(trace_matrix(loop))
    check(profile[8] > 0.99, "8-bar loop is self-similar at lag 8")

    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-reference", action="store_true",
                    help="mine wub-flow-reference-v1 from --processed")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--max-tracks", type=int, default=160)
    ap.add_argument("--trace", default="")
    ap.add_argument("--reference", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--demo-trace", default="", help="write a synthetic trace and exit")
    ap.add_argument("--demo-mode", default="arc", choices=("arc", "song", "churn"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        for message in self_test():
            print(f"[flow] ok: {message}")
        return

    if args.demo_trace:
        trace = demo_trace(args.demo_mode)
        Path(args.demo_trace).write_text(json.dumps(trace, separators=(",", ":")))
        print(f"[flow] wrote {args.demo_trace} ({len(trace['bars'])} bars)")
        return

    if args.build_reference:
        ref = build_reference(Path(args.processed), max_tracks=args.max_tracks)
        text = json.dumps(ref, separators=(",", ":"))
        if args.out:
            Path(args.out).write_text(text)
            print(f"[flow] reference: {ref['tracks']} tracks -> {args.out}")
        else:
            print(text)
        print("[flow] duration bands (bars):")
        for label, band in ref["duration_bars"].items():
            print(f"  {label:11s} p10 {band['p10']:.0f}  p50 {band['p50']:.0f}  "
                  f"p90 {band['p90']:.0f}  n={band['n']}")
        print("[flow] lag profile: " + "  ".join(
            f"{lag}:{value:+.2f}" for lag, value in ref["lag_profile"].items()))
        print("[flow] within-section: " + "  ".join(
            f"{lag}:{value:+.2f}"
            for lag, value in ref.get("lag_profile_within_section", {}).items()))
        return

    if not args.trace:
        ap.error("one of --build-reference, --demo-trace, --trace or --self-test is required")

    trace = json.loads(Path(args.trace).read_text())
    ref = json.loads(Path(args.reference).read_text()) if args.reference else None
    report = score_trace(trace, ref)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[flow] report -> {args.out}")
    else:
        print(text)
    print(f"[flow] {'PASS' if report['verdict']['pass'] else 'FAIL'}: "
          + ("; ".join(report["verdict"]["reasons"]) or "all gates"))
    print("[flow] sections: " + " > ".join(
        f"{run['section']}({run['bars']})" for run in report["section_runs"]))
    for stem in STEMS:
        stats = report["sources"][stem]
        print(f"[flow] {stem:6s} {stats['bars_per_change']:.1f} bars/source "
              f"({stats['changes']:.0f} changes, {stats['distinct']:.0f} sources, "
              f"entropy {stats['phrase_entropy']:.2f})")


if __name__ == "__main__":
    main()
