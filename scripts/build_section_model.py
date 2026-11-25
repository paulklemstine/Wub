#!/usr/bin/env python3
"""Learn an expanded section vocabulary and transition model from the corpus.

Uses per-bar acoustic features (section_arcs.bar_features) and the existing
heuristic segmentation to derive richer, feature-learned section kinds, then
mines how real songs transition between them. Exports a browser-consumable
JSON model for infinite section navigation.

Usage:
  PYTHONPATH=. python3 scripts/build_section_model.py \
    --processed data/processed --out frontend/public/data/kits/section_model.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wub_core.section_arcs import bar_features, segment_sections
from wub_core.section_model import (
    BASE_OF,
    EXPANDED,
    TARGET_ENERGY_PRIOR,
    _clamp01,
    build_transition_model,
    classify_expanded,
    learn_thresholds,
    span_descriptor,
)


def _rank_normalize(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return []
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
    return ranks.tolist()


def mine_corpus(processed_dir: Path, max_tracks: Optional[int] = None, max_bars: int = 64):
    files = sorted(
        p for p in processed_dir.glob("*.json")
        if not p.name.startswith(("_", "meta_"))
    )
    if max_tracks:
        files = files[:max_tracks]

    parsed = []  # (descs_per_track)
    all_desc: List[Dict[str, float]] = []

    for jf in files:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        segs = raw.get("segments") or []
        tatums = raw.get("tatums") or []
        if len(tatums) < 64:
            continue
        tatums = tatums[: max_bars * 16]
        try:
            feats = bar_features(segs, tatums)
            spans = segment_sections(feats)
        except Exception:
            continue
        if not spans:
            continue
        n_bars = len(feats)
        descs = [span_descriptor(s, feats, n_bars) for s in spans]
        # Within-track relative ranks (stable across dB saturation).
        e_ranks = _rank_normalize([d["energy_db"] for d in descs])
        d_ranks = _rank_normalize([d["density"] for d in descs])
        for i, d in enumerate(descs):
            d["energy_rank"] = float(e_ranks[i]) if e_ranks else 0.5
            d["density_rank"] = float(d_ranks[i]) if d_ranks else 0.5
        parsed.append(descs)
        all_desc.extend(descs)

    thresholds = learn_thresholds(all_desc)

    sequences: List[List[Dict[str, Any]]] = []
    for descs in parsed:
        seq = []
        for desc in descs:
            kind = classify_expanded(desc, thresholds)
            # Intensity in engine 0..1: blend within-track energy rank with base prior.
            prior = TARGET_ENERGY_PRIOR.get(kind, 0.5)
            erank = float(desc.get("energy_rank", 0.5))
            drank = float(desc.get("density_rank", 0.5))
            intensity = _clamp01(0.55 * prior + 0.30 * erank + 0.15 * drank)
            seq.append({
                "section": kind,
                "base": BASE_OF.get(kind, "GROOVE_A"),
                "bars": int(max(4, round(desc.get("bars", 4)))),
                "intensity": round(float(intensity), 4),
            })
        if not seq:
            continue
        # Merge consecutive identical expanded sections.
        merged = [dict(seq[0])]
        for item in seq[1:]:
            if item["section"] == merged[-1]["section"]:
                merged[-1]["bars"] += item["bars"]
                merged[-1]["intensity"] = round(
                    (merged[-1]["intensity"] + item["intensity"]) / 2.0, 4
                )
            else:
                merged.append(dict(item))
        sequences.append(merged)

    return sequences, thresholds, len(all_desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="frontend/public/data/kits/section_model.json")
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--max-bars", type=int, default=64)
    args = ap.parse_args()

    processed = Path(args.processed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[section-model] mining {processed} ...", file=sys.stderr)
    sequences, thresholds, n_desc = mine_corpus(processed, args.max_tracks, args.max_bars)
    print(f"[section-model] tracks={len(sequences)} spans={n_desc}", file=sys.stderr)
    model = build_transition_model(sequences)
    model["thresholds"] = {k: round(float(v), 6) for k, v in thresholds.items()}
    model["stats"]["tracks"] = len(sequences)
    out.write_text(json.dumps(model, indent=2))
    print(f"[section-model] wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)
    print(json.dumps({
        "tracks": model["stats"]["tracks"],
        "sections": model["expanded_sections"],
        "section_counts": model["stats"]["section_counts"],
        "target_energy": model["target_energy"],
        "top_starts": model["start_probs"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
