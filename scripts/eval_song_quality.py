"""Score generated songs with the reference-free song-quality metric.

Samples deterministic songs (seeded journeys + bar transitions) and scores
each with wub_core.song_quality.score_song. Prints a build-comparable
summary and writes data/song_quality_eval.json for tracking across commits.

This is the leaderboard half of the quality loop; the other half is blind
A/B votes (docs/SONG_QUALITY.md). Until metric-vs-human agreement >= 70%,
use these numbers for measurement only.

Usage: PYTHONPATH=. python3 scripts/eval_song_quality.py [--songs 8] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from wub_core.song_quality import STEMS, score_song

SEC_VEL = {"INTRO": 0.8, "GROOVE_A": 0.9, "BUILDUP": 0.85, "DROP": 0.95, "BREAKDOWN": 0.8}


def _sample_songs(pats: dict, id_pitch: Dict[str, int], n_songs: int,
                  bars_per_section: int, seed: int) -> List[List[Dict[str, Any]]]:
    rng = random.Random(seed)
    protos = {s: pats["bar_prototypes"][s] for s in STEMS}
    trans = {s: pats.get("bar_transitions", {}).get(s, {}) for s in STEMS}
    arcs = pats.get("section_arcs", {}).get("arcs", [])
    songs = []
    for _ in range(n_songs):
        if arcs:
            weights = [a.get("count", 1) for a in arcs]
            arc = rng.choices(arcs, weights=weights, k=1)[0]["sections"]
        else:
            arc = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN", "GROOVE_A"]
        bars: List[Dict[str, Any]] = []
        prev = {s: None for s in STEMS}
        for sec in arc:
            for _ in range(bars_per_section):
                bar_toks: Dict[str, List] = {}
                for s in STEMS:
                    nxt = None
                    options = trans[s].get(prev[s] or "") if prev[s] else None
                    if options:
                        keys = list(options.keys())
                        probs = [options[k] for k in keys]
                        r, acc = rng.random() * sum(probs), 0.0
                        for k, p in zip(keys, probs):
                            acc += p
                            if r <= acc:
                                nxt = k
                                break
                    if nxt is None:
                        top = protos[s][0] if protos[s] else None
                        nxt = top["signature"] if top else ".".join(["."] * 16)
                    # resolve signature -> exemplar tokens
                    toks = None
                    for p in protos[s]:
                        if p["signature"] == nxt:
                            toks = list(p["tokens"])
                            break
                    if toks is None:
                        toks = [None] * 16
                    bar_toks[s] = toks
                    prev[s] = nxt
                onset = {s: [i for i, t in enumerate(bar_toks[s]) if t] for s in STEMS}
                step_pitches: Dict[str, Dict[int, int]] = {s: {} for s in ("bass", "other", "vocals")}
                for s in ("bass", "other", "vocals"):
                    for i, t in enumerate(bar_toks[s]):
                        if t and t in id_pitch:
                            step_pitches[s][i] = int(id_pitch[t]) % 12
                v = SEC_VEL.get(sec, 0.9)
                vels = {s: [v if i in onset[s] else 0.0 for i in range(16)] for s in STEMS}
                # chord root: bass first-onset pitch, else 0 (C), minor like the corpus
                broot = 0
                for i in onset["bass"]:
                    if i in step_pitches["bass"]:
                        broot = step_pitches["bass"][i]
                        break
                bars.append({
                    "section": sec, "chord_root": broot, "chord_minor": True,
                    "onset_grids": onset, "step_pitches": step_pitches, "velocities": vels,
                })
        songs.append(bars)
    return songs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="frontend/public/data/global_patterns.json")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--songs", type=int, default=8)
    ap.add_argument("--bars-per-section", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/song_quality_eval.json")
    args = ap.parse_args()

    pats = json.loads(Path(args.patterns).read_bytes())
    gvoc = json.loads(Path(args.vocab).read_bytes())
    id_pitch = {v["id"]: int(v.get("pitch", 0)) for s in STEMS for v in gvoc["vocab_by_stem"][s]}
    durations = pats.get("section_arcs", {}).get("durations", {})
    dur_means = {s: float(d.get("mean", 8.0)) for s, d in durations.items()} or None

    songs = _sample_songs(pats, id_pitch, args.songs, args.bars_per_section, args.seed)
    results = [score_song(bars, duration_means=dur_means) for bars in songs]
    scores = [r["score"] for r in results]
    summary = {
        "version": 1,
        "seed": args.seed,
        "songs": args.songs,
        "bars_per_section": args.bars_per_section,
        "mean": round(float(sum(scores) / max(1, len(scores))), 1),
        "min": round(min(scores), 1) if scores else 0.0,
        "max": round(max(scores), 1) if scores else 0.0,
        "axes": {},
        "songs_detail": results,
    }
    for ax in ("groove", "harmony", "arrangement", "repetition", "mix"):
        vals = [r[ax] for r in results]
        summary["axes"][ax] = round(float(sum(vals) / max(1, len(vals))), 1)
    Path(args.out).write_text(json.dumps(summary, separators=(",", ":")))
    print(f"song-quality: mean={summary['mean']} min={summary['min']} max={summary['max']} (n={len(scores)})", flush=True)
    print("  axes: " + " ".join(f"{k}={v}" for k, v in summary["axes"].items()), flush=True)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
