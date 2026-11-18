"""Mine motif return rates and vocal cadence from real songs.

From `data/processed/*.json` this computes, for every stem, how often a 4-bar
rhythmic/melodic pattern returns 8/16/32 bars later, how often it repeats at
all, and how long stable vocal pattern runs last. The two exported sidecars are
ranking/budget priors only: they never schedule a bar or change audio.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
LAGS = (2, 4, 8)  # 4-bar cells -> 8, 16, 32 bars


def song_cells(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tats = doc.get("tatums") or []
    if len(tats) < 64:
        return None
    feats = bar_features(segs, tats)
    spans = segment_sections(feats)
    labels = [s.label for s in spans for _ in range(s.n_bars)]
    n = len(tats) // 16
    labels = (labels + ["?"] * n)[:n]
    by_id = {str(s.get("id")): s for s in segs if s.get("id")}
    prev = {s: None for s in STEMS}
    cells = []
    for c in range(n // 4):
        union = {s: 0 for s in STEMS}
        presence = {s: 0 for s in STEMS}
        for b in range(c * 4, (c + 1) * 4):
            for step in range(16):
                ss = (tats[b * 16 + step].get("stem_segments") or {})
                for stem in STEMS:
                    ids = ss.get(stem, []) or []
                    cur = ids[-1] if ids else None
                    if cur is not None:
                        presence[stem] += 1
                    onset = (step == 0 and cur is not None) or (cur is not None and cur != prev[stem])
                    if onset:
                        union[stem] |= 1 << step
                    prev[stem] = cur
        cells.append({
            "union": union,
            "presence": {s: presence[s] / 64.0 for s in STEMS},
            "section": labels[c * 4],
        })
    return cells


def jaccard(a: int, b: int) -> float:
    if not (a or b):
        return 1.0
    return (a & b).bit_count() / (a | b).bit_count()


def mine(processed: Path, max_tracks: int):
    hit = defaultdict(int)
    seen = defaultdict(int)
    repeat = defaultdict(int)
    boundaries = defaultdict(int)
    vocal_change = defaultdict(lambda: [0, 0])
    vocal_runs = Counter()
    tracks = 0
    paths = sorted(processed.glob("*.json"))
    step = max(1, len(paths) // max_tracks) if max_tracks > 0 else 1
    paths = paths[::step][:max_tracks] if max_tracks > 0 else paths

    for path in paths:
        try:
            cells = song_cells(json.loads(path.read_text()))
        except Exception:
            continue
        if not cells or len(cells) < 12:
            continue
        tracks += 1
        for i, cell in enumerate(cells):
            for s in STEMS:
                if cell["presence"][s] < 0.05:
                    continue
                for lag in LAGS:
                    if i >= lag:
                        seen[(s, lag)] += 1
                        if jaccard(cells[i - lag]["union"][s], cell["union"][s]) >= 0.75:
                            hit[(s, lag)] += 1
                if i > 0:
                    boundaries[s] += 1
                    if jaccard(cells[i - 1]["union"][s], cell["union"][s]) >= 0.75:
                        repeat[s] += 1
            # Vocal pattern change at a 4-bar boundary.
            if i > 0:
                kind = "section" if cell["section"] != cells[i - 1]["section"] else "inside"
                sim = jaccard(cells[i - 1]["union"]["vocals"], cell["union"]["vocals"])
                vocal_change[kind][1] += 1
                if sim < 0.55:
                    vocal_change[kind][0] += 1
        # Stable vocal runs (consecutive cells whose pattern is close).
        run = 0
        for i, cell in enumerate(cells):
            if cell["presence"]["vocals"] < 0.05:
                if run:
                    vocal_runs[min(8, max(1, run))] += 1
                    run = 0
                continue
            if i == 0 or jaccard(cells[i - 1]["union"]["vocals"], cell["union"]["vocals"]) >= 0.55:
                run += 1
            else:
                if run:
                    vocal_runs[min(8, max(1, run))] += 1
                run = 1
        if run:
            vocal_runs[min(8, max(1, run))] += 1

    # `lag` is in 4-bar cells; expose the equivalent number of bars.
    return_rate = {
        s: {str(lag * 4): round(hit[(s, lag)] / max(1, seen[(s, lag)]), 4) for lag in LAGS}
        for s in STEMS
    }
    novelty = {s: round(1 - repeat[s] / max(1, boundaries[s]), 4) for s in STEMS}
    total_runs = sum(vocal_runs.values())
    run_dist = {str(k * 4): round(v / max(1, total_runs), 4) for k, v in sorted(vocal_runs.items())}

    motif_doc = {
        "format": "wub-motif-calendar-v1", "version": 1,
        "tracks": tracks, "return_rate": return_rate, "novelty_rate": novelty,
        "policy": {
            "return_bonus": {s: round(min(0.30, max(0.04, return_rate[s]["8"] * 0.45)), 4) for s in STEMS},
            "novelty_bonus": {s: round(min(0.18, novelty[s] * 0.10), 4) for s in STEMS},
        },
        "report": {
            "source": "data/processed/*.json",
            "return_rule": "4-bar union onset Jaccard >= 0.75 at lag 2/4/8 cells",
            "use": "soft ranking prior only; never schedules or resamples audio",
        },
    }
    vocal_doc = {
        "format": "wub-vocal-cadence-v1", "version": 2,
        "tracks": tracks,
        "run_lengths_bars": run_dist,
        "change_prob": {
            kind: round(v[0] / max(1, v[1]), 4) for kind, v in vocal_change.items()
        },
        "policy": {
            "min_hold_bars": 8,
            "section_change_bonus": round(
                max(0.04, (vocal_change["section"][0] / max(1, vocal_change["section"][1])
                           - vocal_change["inside"][0] / max(1, vocal_change["inside"][1])) * 0.6), 4),
        },
        "report": {
            "source": "data/processed/*.json",
            "run_rule": "consecutive 4-bar cells with vocal onset Jaccard >= 0.55",
            "use": "vocal release budget only; section changes always release",
        },
    }
    return motif_doc, vocal_doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()
    motif_doc, vocal_doc = mine(args.processed, args.max_tracks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "motif_calendar.json").write_text(json.dumps(motif_doc, separators=(",", ":")) + "\n")
    (args.out_dir / "vocal_cadence.json").write_text(json.dumps(vocal_doc, separators=(",", ":")) + "\n")
    print(json.dumps({"motif": motif_doc["policy"], "vocal": vocal_doc["policy"],
                      "return_rates": motif_doc["return_rate"]}, indent=2))


if __name__ == "__main__":
    main()
