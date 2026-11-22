"""Cluster tracks into style ids from existing analysis JSONs (no audio decode).

Aggregates per-track mean timbre(12) + mean chroma(12) + bpm + mean loudness
from data/processed/*.json, StandardScales, KMeans into K styles, and writes
a {track_id: style_id} map with 1-based ids (0 reserved = unknown).

Usage (laptop):
  python3 scripts/cluster_styles.py --processed data/processed \\
      --out data/style_labels.json --k 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def track_vector(doc: dict):
    """Mean timbre/chroma + bpm + loudness from a TrackAnalysis JSON dict."""
    import numpy as np

    segs = doc.get("segments", []) or []
    if not segs:
        return None
    timbres, chromas, louds = [], [], []
    for sg in segs:
        t = sg.get("timbre")
        c = sg.get("chroma", sg.get("pitches"))
        if t is None or c is None or len(t) != 12 or len(c) != 12:
            continue
        timbres.append(t)
        chromas.append(c)
        try:
            louds.append(float(sg.get("loud_max", sg.get("loudness_max", -20.0))))
        except (TypeError, ValueError):
            pass
    if not timbres:
        return None
    timbres = np.asarray(timbres, dtype=np.float64)
    chromas = np.asarray(chromas, dtype=np.float64)
    try:
        bpm = float(doc.get("bpm", 0.0) or 0.0)
    except (TypeError, ValueError):
        bpm = 0.0
    mean_loud = float(np.mean(louds)) if louds else -20.0
    return (
        timbres.mean(axis=0).tolist()
        + chromas.mean(axis=0).tolist()
        + [bpm / 200.0, (mean_loud + 60.0) / 60.0]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster tracks into style ids")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/style_labels.json")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    proc = Path(args.processed)
    files = sorted(
        p for p in proc.glob("*.json") if not p.name.startswith(("_", "meta_"))
    )
    if not files:
        raise SystemExit(f"No analyses in {proc}")
    ids, vecs = [], []
    for p in files:
        try:
            doc = json.loads(p.read_text())
        except Exception as e:
            print(f"  WARN skipping {p.name}: {type(e).__name__}")
            continue
        tid = doc.get("track_id") or p.stem
        v = track_vector(doc)
        if v is None:
            continue
        ids.append(str(tid))
        vecs.append(v)
    X = np.asarray(vecs, dtype=np.float64)
    k = int(min(args.k, len(ids)))
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=int(args.seed))
    labels = km.fit_predict(scaler.transform(X))
    out = {tid: int(lab) + 1 for tid, lab in zip(ids, labels)}  # 1-based; 0 = unknown
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    from collections import Counter

    print(f"tracks clustered: {len(out)} (k={k}) -> {args.out}")
    for sid, n in sorted(Counter(out.values()).items()):
        print(f"  style {sid:3d}: {n:5d} tracks")
    print("done")


if __name__ == "__main__":
    main()
