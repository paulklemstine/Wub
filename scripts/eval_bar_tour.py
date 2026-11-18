#!/usr/bin/env python3
"""Compare greedy vs beam bar-tour navigation on the same candidate pool.

This is the offline yardstick for the TSP-like planner: given an identical set
of bar cells, does a receding-horizon search with the learned value head find a
route with better flow, motif return and novelty than the one-step choice?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wub_core.bar_tour import Cell, plan_tour
from wub_core.slice_flow import DIM, STEMS
from scripts.train_slice_flow import partition


def load_models(base: Path) -> dict:
    flow = json.loads((base / "slice_flow.json").read_text())
    models = {"critic": flow["critic"], "mix": flow["mix"]}
    motif = base / "motif_return.json"
    if motif.exists():
        models["motif"] = json.loads(motif.read_text())
    value = base / "bar_tour_value.json"
    if value.exists():
        models["value"] = json.loads(value.read_text())["net"]
    return models


def route_metrics(cells, route, models):
    from wub_core.bar_tour import edge_scores, net_forward, sigmoid, VALUE_INPUT_DIM
    feats = np.stack([cells[i].features for i in route])
    edges = [edge_scores(feats[i], feats[i + 1:i + 2], cells[route[i]].stem, models)[0]
             for i in range(len(route) - 1)]
    return {
        "flow": float(np.mean(edges)) if edges else 0.0,
        "changes": sum(cells[route[i]].track != cells[route[i - 1]].track
                       for i in range(1, len(route))),
        "distinct": len({cells[i].phrase for i in route}),
        "bars": len(route),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kits", default="frontend/public/data/kits")
    ap.add_argument("--tracks", type=int, default=12)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--seed", type=int, default=505)
    args = ap.parse_args()

    base = Path(args.kits)
    manifest = json.loads((base / "kit_manifest.json").read_text())
    model = json.loads((base / "slice_flow.json").read_text())
    features = np.fromfile(base / model["feature_file"], dtype=np.float32).reshape(-1, 4, 4, DIM)
    rows = {pid: i for i, pid in enumerate(model["phrase_ids"])}
    models = load_models(base)
    rng = np.random.default_rng(args.seed)

    kits = [k for k in manifest["kits"] if partition(str(k["track_id"]))
            and sum(p["id"] in rows for p in k["phrases"]) >= 3]
    kits = [kits[int(i)] for i in rng.choice(len(kits), size=min(args.tracks, len(kits)), replace=False)]

    variants = {"greedy": {}, "beam": {}, "beam+value": {}}
    for kit in kits:
        phrases = sorted((p for p in kit["phrases"] if p["id"] in rows),
                         key=lambda p: int(p.get("start_bar", 0)))
        for stem_index, stem in enumerate(STEMS):
            pool_phrases = [p for k in manifest["kits"]
                            if k.get("key") == kit.get("key") and k["track_id"] != kit["track_id"]
                            for p in k["phrases"] if p["id"] in rows]
            chosen = phrases[:2] + [pool_phrases[int(i)] for i in
                                    rng.choice(len(pool_phrases), size=min(args.pool, len(pool_phrases)), replace=False)]
            track_by_phrase = {p["id"]: str(k["track_id"]) for k in manifest["kits"]
                               for p in k["phrases"]}
            selected = phrases[:1] + chosen[2:]
            cells = [Cell(p["id"], track_by_phrase[p["id"]], stem, bar,
                          np.asarray(features[rows[p["id"]], stem_index, bar], np.float32))
                     for p in selected for bar in range(4)]
            if len(cells) < 16:
                continue
            start = 0
            plain = {k: v for k, v in models.items() if k != "value"}
            greedy = plan_tour(cells, start, args.horizon, plain, beam=1, branch=1)
            beam = plan_tour(cells, start, args.horizon, plain, beam=10, branch=10)
            with_value = plan_tour(cells, start, args.horizon, models, beam=10, branch=10)
            for name, route in (("greedy", greedy), ("beam", beam), ("beam+value", with_value)):
                metrics = route_metrics(cells, route, models)
                for key, value in metrics.items():
                    variants[name].setdefault(key, []).append(value)

    for name, values in variants.items():
        if not values.get("flow"):
            continue
        summary = {k: float(np.mean(v)) for k, v in values.items()}
        print(f"[tour] {name:11s} flow {summary['flow']:.4f}  "
              f"changes {summary['changes']:.2f}  distinct {summary['distinct']:.1f}  "
              f"bars {summary['bars']:.1f}")


if __name__ == "__main__":
    main()
