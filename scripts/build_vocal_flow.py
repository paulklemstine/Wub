#!/usr/bin/env python3
"""Mine vocal phrase development from real song analyses.

Outputs a small prior used by BarComposer so a quiet vocal phrase is developed
across several *different* bars instead of copying its only active bar into
every output bar.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def vocal_bars(doc: dict) -> tuple[np.ndarray, np.ndarray]:
    seg = {str(s.get("id")): s for s in doc.get("segments", []) if isinstance(s, dict)}
    tat = doc.get("tatums", [])
    n = len(tat) // 16
    steps = np.zeros((n, 16), dtype=np.int8)
    chroma = np.zeros((n, 12), dtype=np.float32)
    for i, t in enumerate(tat[:n * 16]):
        ids = (t.get("stem_segments") or {}).get("vocals") or []
        if not ids:
            continue
        s = seg.get(str(ids[-1]))
        if s and not s.get("is_silent", False) and float(s.get("loudness_max", -60)) > -55:
            steps[i // 16, i % 16] = 1
            pitches = np.asarray(s.get("pitches") or [], dtype=np.float32)
            if pitches.size == 12:
                chroma[i // 16] += pitches / 16.0
    chroma /= np.maximum(chroma.max(axis=1, keepdims=True), 1e-6)
    return steps, chroma


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--max-tracks", type=int, default=600)
    ap.add_argument("--out", default="frontend/public/data/kits/vocal_flow.json")
    args = ap.parse_args()

    paths = sorted(Path(args.processed).glob("*.json"))
    if args.max_tracks > 0:
        paths = paths[:args.max_tracks]
    runs: Counter = Counter()
    distinct: Dict[int, List[int]] = defaultdict(list)
    changes = 0
    pairs = 0
    for path in paths:
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        steps, chroma = vocal_bars(doc)
        if steps.shape[0] < 24:
            continue
        active = (steps.sum(axis=1) >= 2)
        length = 0
        for value in list(active) + [False]:
            if value:
                length += 1
            elif length:
                runs[min(32, length)] += 1
                length = 0
        for start in range(0, len(active) - 3, 4):
            idx = [i for i in range(4) if active[start + i]]
            if not idx:
                continue
            signatures = [int(np.argmax(chroma[start + i])) for i in idx]
            distinct[len(idx)].append(len(set(signatures)))
            for a, b in zip(idx, idx[1:]):
                pairs += 1
                changes += int(signatures[idx.index(a)] != signatures[idx.index(b)])

    total = max(1, sum(runs.values()))
    run_prior = {str(k): round(v / total, 4) for k, v in sorted(runs.items()) if k <= 16}
    distinct_prior = {
        str(k): round(float(np.mean(v)), 3) for k, v in sorted(distinct.items()) if v
    }
    prior = {
        "format": "wub-vocal-flow-v1",
        "version": 1,
        "run_lengths": run_prior,
        "distinct_bars_per_active": distinct_prior,
        "variation_rate": round(changes / max(1, pairs), 4),
        "tracks": len(paths),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior, separators=(",", ":")) + "\n")
    print(f"[vocal] {len(paths)} tracks, {sum(runs.values())} vocal runs -> {out}")
    print(f"[vocal] run prior {run_prior}")
    print(f"[vocal] distinct bars {distinct_prior}, variation {prior['variation_rate']}")


if __name__ == "__main__":
    main()
