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
DENS_TGT = {"INTRO": {"drums": 0.15, "bass": 0.1, "other": 0.25, "vocals": 0.1},
            "GROOVE_A": {"drums": 0.45, "bass": 0.3, "other": 0.35, "vocals": 0.2},
            "BUILDUP": {"drums": 0.5, "bass": 0.3, "other": 0.4, "vocals": 0.3},
            "DROP": {"drums": 0.55, "bass": 0.35, "other": 0.5, "vocals": 0.3},
            "BREAKDOWN": {"drums": 0.05, "bass": 0.05, "other": 0.3, "vocals": 0.3}}


def _sample_songs(pats: dict, id_pitch: Dict[str, int], id_cent: Dict[str, float],
                  n_songs: int, bars_per_section: int, max_bars: int,
                  seed: int) -> List[List[Dict[str, Any]]]:
    rng = random.Random(seed)
    _jt = pats.get("joint_tuples", []) or []
    _pmi_vals = [t.get("pmi", 0) for t in _jt if isinstance(t, dict)]
    _pmi_lo = min(_pmi_vals) if _pmi_vals else 0.0
    _pmi_hi = max(_pmi_vals) if _pmi_vals else 1.0
    _pmi_map = {t["sig"]: t["pmi"] for t in _jt if isinstance(t, dict) and "sig" in t}
    _cr = pats.get("call_response", {}) or {}
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
        durs = pats.get("section_arcs", {}).get("durations", {})
        for sec in arc:
            if bars_per_section and bars_per_section > 0:
                n_sec = bars_per_section
            else:
                n_sec = max(4, min(max_bars, int(round(float(durs.get(sec, {}).get("mean", 8.0))))))
            for _bi in range(n_sec):
                bar_toks: Dict[str, List] = {}
                is_final = _bi == n_sec - 1
                for s in STEMS:
                    cands = []
                    for _ in range(3):
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
                        cands.append(nxt)
                    # mirror the shipped draw: normalized joint PMI + call-response + density band
                    scored = []
                    for cand in cands:
                        combo = "|".join([(cand if st == s else (prev[st] or ".")) for st in STEMS])
                        raw = _pmi_map.get(combo, 0.0)
                        pmi = 0.0 if raw <= 0 else max(0.0, min(1.0, (raw - _pmi_lo) / max(1e-9, _pmi_hi - _pmi_lo)))
                        cross = 0.0
                        for o in STEMS:
                            if o == s or not prev[o]:
                                continue
                            tab = _cr.get(f"{o}->{s}") or {}
                            v = (tab.get(prev[o]) or {}).get(cand, 0.0)
                            if isinstance(v, (int, float)):
                                cross += float(v)
                        dens = sum(1 for x in cand.split("-") if x != ".") / 16.0
                        tgt = float(DENS_TGT.get(sec, {}).get(s, 0.3))
                        if is_final and s == "drums":
                            tgt = min(0.9, tgt + 0.2)  # fill anticipation on section-final bars
                        fit = max(0.0, 1.0 - abs(dens - tgt) / 0.35)
                        down = 0.0
                        if s == "drums" and sec in ("DROP", "GROOVE_A"):
                            parts = cand.split('-')
                            down = sum(1 for q in (0, 4, 8, 12) if parts[q] != '.') / 4.0
                        scored.append((pmi * 2.0 + cross * 1.0 + fit * 1.5 + down * 2.0, cand))
                        if pmi > 0.3 and cross > 0:
                            break
                    scored.sort(key=lambda t: -t[0])
                    nxt = scored[0][1] if scored else cands[0]
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
                if sec in ("DROP", "GROOVE_A"):
                    for q in (0, 4, 8, 12):
                        if q not in onset["drums"]:
                            onset["drums"].append(q)
                    onset["drums"].sort()
                step_pitches: Dict[str, Dict[int, int]] = {s: {} for s in ("bass", "other", "vocals")}
                step_centroids: Dict[str, Dict[int, float]] = {s: {} for s in STEMS}
                for s in ("bass", "other", "vocals"):
                    for i, t in enumerate(bar_toks[s]):
                        if t and t in id_pitch:
                            step_pitches[s][i] = int(id_pitch[t]) % 12
                for s in STEMS:
                    for i, t in enumerate(bar_toks[s]):
                        if t and t in id_cent:
                            step_centroids[s][i] = float(id_cent[t])
                v = SEC_VEL.get(sec, 0.9)
                vels = {s: [v if i in onset[s] else 0.0 for i in range(16)] for s in STEMS}
                bars.append({
                    "section": sec, "chord_root": 0, "chord_minor": True,
                    "onset_grids": onset, "step_pitches": step_pitches,
                    "step_centroids": step_centroids, "velocities": vels,
                })
        # Tonic-relative progression: the song's own key (pitch histogram
        # mode) drives i-VI-III-VII roots, cycled every 4 bars like the
        # frontend scheduler. Fair across keys; no bass-line cheating.
        hist: dict = {}
        for b in bars:
            for st in ("bass", "other", "vocals"):
                for pc in (b["step_pitches"].get(st) or {}).values():
                    hist[int(pc) % 12] = hist.get(int(pc) % 12, 0) + 1
        tonic = max(hist, key=lambda k: hist[k]) if hist else 0
        rel = [0, -4, 3, -2]
        for idx, b in enumerate(bars):
            b["chord_root"] = (tonic + rel[(idx // 4) % len(rel)]) % 12
            b["chord_minor"] = True
        songs.append(bars)
    return songs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="frontend/public/data/global_patterns.json")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--songs", type=int, default=8)
    ap.add_argument("--bars-per-section", type=int, default=0,
                        help="bars per section; <=0 follows mined section durations")
    ap.add_argument("--max-bars", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/song_quality_eval.json")
    args = ap.parse_args()

    pats = json.loads(Path(args.patterns).read_bytes())
    gvoc = json.loads(Path(args.vocab).read_bytes())
    id_pitch = {v["id"]: int(v.get("pitch", 0)) for s in STEMS for v in gvoc["vocab_by_stem"][s]}
    id_cent = {v["id"]: float(v.get("centroid", 1200.0)) for s in STEMS for v in gvoc["vocab_by_stem"][s]}
    durations = pats.get("section_arcs", {}).get("durations", {})
    dur_means = {s: float(d.get("mean", 8.0)) for s, d in durations.items()} or None

    songs = _sample_songs(pats, id_pitch, id_cent, args.songs, args.bars_per_section, args.max_bars, args.seed)
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
