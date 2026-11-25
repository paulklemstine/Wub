"""Learn section transitions, durations and energy slopes from real songs.

The browser ships a hand-written 104-bar arc. This mines `data/processed/*.json`
with the repo's own `wub_core.section_arcs` detector and exports an optional
`song_arc_model.json` that the browser can sample to build a per-set arc:

* section transition probabilities,
* section length distributions (in 4-bar cells),
* per-section mean energy and energy slope,
* common section archetypes across the corpus.

No audio is changed; the sidecar only supplies section/energy candidates to the
existing planner (see docs/SLICE_FLOW.md).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from wub_core.section_arcs import bar_features, segment_sections

FORMAT = "wub-song-arc-v1"


def song_sections(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tatums = doc.get("tatums") or []
    if len(tatums) < 64:
        return None
    feats = bar_features(segs, tatums)
    spans = segment_sections(feats)
    if len(spans) < 3:
        return None
    runs = []
    for span in spans:
        label = span.label
        bars = feats[span.start_bar:span.start_bar + span.n_bars]
        energies = [float(b.get("energy", -60.0)) for b in bars]
        # Least-squares slope in dB per bar (build/fade shape of the section).
        x = np.arange(len(energies), dtype=float)
        slope = float(np.polyfit(x, np.asarray(energies), 1)[0]) if len(energies) >= 3 else 0.0
        mean = float(np.mean(energies))
        runs.append({
            "section": label,
            "bars": int(span.n_bars),
            "mean_db": mean,
            "slope_db_per_bar": slope,
        })
    return runs


def dist(counter: Counter, total: int) -> Dict[str, float]:
    return {str(k): round(v / max(1, total), 4) for k, v in sorted(counter.items())}


def mine(processed: Path, max_tracks: int) -> Dict[str, Any]:
    paths = sorted(processed.glob("*.json"))
    step = max(1, len(paths) // max_tracks) if max_tracks > 0 else 1
    paths = paths[::step][:max_tracks] if max_tracks > 0 else paths

    transitions = defaultdict(Counter)
    lengths = defaultdict(Counter)
    energies = defaultdict(list)
    slopes = defaultdict(list)
    boundaries = defaultdict(list)
    archetypes = Counter()
    tracks = 0
    for path in paths:
        try:
            runs = song_sections(json.loads(path.read_text()))
        except Exception:
            continue
        if not runs:
            continue
        tracks += 1
        for i, run in enumerate(runs):
            sec = run["section"]
            lengths[sec][run["bars"]] += 1
            energies[sec].append(run["mean_db"])
            slopes[sec].append(run["slope_db_per_bar"])
            if i + 1 < len(runs):
                nxt = runs[i + 1]["section"]
                transitions[sec][nxt] += 1
                boundaries[(sec, nxt)].append(runs[i + 1]["mean_db"] - run["mean_db"])
        archetypes[tuple(r["section"] for r in runs)] += 1

    all_db = [v for values in energies.values() for v in values]
    lo = float(np.percentile(all_db, 5)) if all_db else -40.0
    hi = float(np.percentile(all_db, 95)) if all_db else 0.0
    span = max(1e-6, hi - lo)

    def scale(v: float) -> float:
        return round(float(np.clip((v - lo) / span, 0, 1)), 4)

    section_energy = {
        sec: {
            "mean_db": round(float(np.mean(values)), 3),
            "target": scale(float(np.mean(values))),
            "p25_db": round(float(np.percentile(values, 25)), 3),
            "p75_db": round(float(np.percentile(values, 75)), 3),
            "songs": len(values),
        } for sec, values in sorted(energies.items())
    }
    section_slope = {
        sec: {
            "mean_db_per_bar": round(float(np.mean(values)), 4),
            "p10": round(float(np.percentile(values, 10)), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
        } for sec, values in sorted(slopes.items())
    }
    length_rows = {}
    for sec, counter in sorted(lengths.items()):
        arr: List[int] = []
        for length, count in counter.items():
            arr.extend([int(length)] * count)
        length_rows[sec] = {
            "p10": int(np.percentile(arr, 10)),
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "mean": round(float(np.mean(arr)), 2),
            "min_bars": 4,
        }
    boundary_rows = {
        f"{a}->{b}": {
            "p10": round(float(np.percentile(v, 10)), 3),
            "p50": round(float(np.percentile(v, 50)), 3),
            "p90": round(float(np.percentile(v, 90)), 3),
        } for (a, b), v in sorted(boundaries.items()) if len(v) >= 20
    }
    arches = [
        {"sections": list(seq), "count": count, "p": round(count / max(1, tracks), 4)}
        for seq, count in archetypes.most_common(24)
    ]

    return {
        "format": FORMAT,
        "version": 1,
        "tracks": tracks,
        "transition": {sec: dist(counter, sum(counter.values()))
                       for sec, counter in sorted(transitions.items())},
        "lengths": length_rows,
        "section_energy": section_energy,
        "section_slope": section_slope,
        "boundary_delta_db": boundary_rows,
        "archetypes": arches,
        "report": {
            "source": "data/processed/*.json",
            "detector": "wub_core.section_arcs.segment_sections",
            "label_provenance": "acoustic change-point detection, not human annotation",
            "use": "sample a per-set section/energy arc; never changes audio assets",
            "energy_scale": {"p5_db": round(lo, 3), "p95_db": round(hi, 3)},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()
    doc = mine(args.processed, args.max_tracks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps({
        "tracks": doc["tracks"],
        "transition": doc["transition"],
        "lengths": doc["lengths"],
        "energy": {k: v["target"] for k, v in doc["section_energy"].items()},
        "slope": {k: v["mean_db_per_bar"] for k, v in doc["section_slope"].items()},
        "top_archetypes": doc["archetypes"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
