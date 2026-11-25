"""Measure how closely a generated session copies the training songs' style.

Compares per-block telemetry from the app (movement log: section, bars,
changed stems, source cohort) against distributions mined from the analyzed
songs. Returns a 0..1 style-fidelity score and per-dimension divergences.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from wub_core.section_arcs import bar_features, segment_sections

SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
STEMS = ("drums", "bass", "other", "vocals")


def corpus_stats(processed: Path, max_tracks: int = 0,
                 cohort_artifact: Path | None = None) -> Dict[str, Any]:
    paths = sorted(processed.glob("*.json"))
    if max_tracks > 0:
        paths = paths[::max(1, len(paths) // max_tracks)][:max_tracks]
    durations: Counter = Counter()
    transitions: Counter = Counter()
    changes: Counter = Counter()
    cohorts: Counter = Counter()
    slopes: List[float] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text())
            bars = bar_features(doc.get("segments") or [], doc.get("tatums") or [])
            spans = segment_sections(bars)
        except Exception:
            continue
        labels: List[str] = []
        for s in spans:
            lab = s.label if s.label in SECTIONS else "GROOVE_A"
            labels.extend([lab] * s.n_bars)
            durations[min(32, max(4, int(s.n_bars)))] += 1
            vals = [float(b.get("energy", -60.0)) for b in bars[s.start_bar:s.start_bar + s.n_bars]]
            if len(vals) >= 3:
                x = np.arange(len(vals), dtype=float)
                slopes.append(float(np.polyfit(x, np.asarray(vals), 1)[0]))
        for i in range(1, len(spans)):
            a = labels[(sum(s.n_bars for s in spans[:i]) - 1)]
            b = labels[sum(s.n_bars for s in spans[:i])]
            transitions[(a, b)] += 1
            if i > 0:
                changes[min(4, i)] += 0  # placeholder; per-boundary changes need onsets
    total = max(1, sum(durations.values()))
    cohort: Dict[str, float] = {}
    if cohort_artifact and cohort_artifact.exists():
        doc = json.loads(cohort_artifact.read_text())
        counts: Counter = Counter()
        for row in doc.get("cohorts", []):
            counts["+".join(sorted(row.get("stems", []))) or "none"] += int(row.get("count", 0))
        ctotal = max(1, sum(counts.values()))
        cohort = {k: v / ctotal for k, v in counts.items()}
    return {
        "cohort": cohort,
        "duration": {str(k): v / total for k, v in sorted(durations.items())},
        "transition": {f"{a}->{b}": v / max(1, sum(transitions.values()))
                       for (a, b), v in transitions.most_common()},
        "slope_p10": float(np.percentile(slopes, 10)) if slopes else 0.0,
        "slope_p50": float(np.percentile(slopes, 50)) if slopes else 0.0,
        "slope_p90": float(np.percentile(slopes, 90)) if slopes else 0.0,
        "tracks": len(paths),
    }


def session_stats(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations: Counter = Counter()
    transitions: Counter = Counter()
    changes: Counter = Counter()
    cohorts: Counter = Counter()
    for i, entry in enumerate(log):
        bars = int(entry.get("bars") or 4)
        section = str(entry.get("target") or "GROOVE_A")
        section = section if section in SECTIONS else "GROOVE_A"
        durations[min(32, max(4, bars))] += 1
        if i > 0:
            prev = str(log[i - 1].get("target") or section)
            prev = prev if prev in SECTIONS else "GROOVE_A"
            transitions[(prev, section)] += 1
            changes[min(4, len(entry.get("changedTracks") or []))] += 1
            cohorts["+".join(sorted(entry.get("changedTracks") or [])) or "none"] += 1
    total = max(1, sum(durations.values()))
    return {
        "duration": {str(k): v / total for k, v in sorted(durations.items())},
        "transition": {f"{a}->{b}": v / max(1, sum(transitions.values()))
                       for (a, b), v in transitions.items()},
        "change": {str(k): v / max(1, sum(changes.values())) for k, v in changes.items()},
        "cohort": {k: v / max(1, sum(cohorts.values())) for k, v in cohorts.items()},
        "blocks": len(log),
    }


def js(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    pa = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    pb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    pa = pa / max(1e-9, pa.sum()); pb = pb / max(1e-9, pb.sum())
    m = 0.5 * (pa + pb)
    def kl(x, y):
        mask = x > 1e-9
        return float(np.sum(x[mask] * np.log2(x[mask] / np.maximum(y[mask], 1e-9))))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def style_fidelity(log: List[Dict[str, Any]], corpus: Dict[str, Any]) -> Dict[str, Any]:
    sess = session_stats(log)
    d_trans = js(sess["transition"] or corpus["transition"], corpus["transition"])
    d_dur = js(sess["duration"], corpus["duration"])
    d_cohort = (js(sess["cohort"], corpus["cohort"]) if corpus.get("cohort") else 0.0)
    fidelity = float(np.clip(1 - 0.35 * d_trans - 0.25 * d_dur - 0.40 * d_cohort, 0, 1))
    return {
        "style_fidelity": round(fidelity, 4),
        "transition_divergence": round(d_trans, 4),
        "duration_divergence": round(d_dur, 4),
        "cohort_divergence": round(d_cohort, 4),
        "change_distribution": sess["change"],
        "blocks": sess["blocks"],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--log", type=Path, required=True, help="JSON array of movement-log entries")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    corpus = corpus_stats(args.processed)
    log = json.loads(args.log.read_text())
    result = style_fidelity(log, corpus)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
