"""Phase 3 CLI: Generate 2D UMAP projections for stem feature spaces.

Projects 28-D stem vectors from FAISS indices down to [0, 1] normalized 2D
coordinates for the interactive HTML5 Canvas Nebula timbre map.

Outputs:
  data/indices/umap_drums.json
  data/indices/umap_bass.json
  data/indices/umap_vocals.json
  data/indices/umap_other.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEMS = ("drums", "bass", "vocals", "other")


def normalize_coords(emb: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Scale 2D array to [margin, 1 - margin] range."""
    span = 1.0 - 2 * margin
    out = np.zeros_like(emb, dtype=np.float32)
    for dim in range(2):
        col = emb[:, dim]
        mn, mx = float(np.min(col)), float(np.max(col))
        rng = mx - mn if mx > mn else 1.0
        out[:, dim] = margin + ((col - mn) / rng) * span
    return out


def project_vectors(vecs: np.ndarray, method: str = "umap", n_neighbors: int = 15,
                    min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    """Project high-dim vectors to 2D using UMAP or PCA fallback."""
    if len(vecs) < 4:
        return np.random.default_rng(seed).uniform(0.1, 0.9, size=(len(vecs), 2)).astype(np.float32)

    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, len(vecs) - 1),
                min_dist=min_dist,
                random_state=seed,
                metric="euclidean",
            )
            return reducer.fit_transform(vecs).astype(np.float32)
        except Exception as e:
            print(f"  WARN: UMAP failed ({e}); falling back to PCA")

    # Fallback: PCA
    from sklearn.decomposition import PCA  # type: ignore

    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(vecs).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 2D UMAP coordinates for timbre nebula visualizer")
    ap.add_argument("--indices-dir", default="data/indices",
                    help="directory containing index_<stem>.faiss and meta_<stem>.json")
    ap.add_argument("--max-points", type=int, default=2000,
                    help="maximum number of slices to project per stem")
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="UMAP n_neighbors parameter")
    ap.add_argument("--min-dist", type=float, default=0.1,
                    help="UMAP min_dist parameter")
    ap.add_argument("--method", choices=["umap", "pca"], default="umap",
                    help="projection method: umap or pca")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed")
    args = ap.parse_args()

    import faiss  # type: ignore

    idx_dir = Path(args.indices_dir)
    if not idx_dir.exists():
        raise SystemExit(f"Indices directory {idx_dir} does not exist.")

    total_pts = 0
    for stem in STEMS:
        idx_path = idx_dir / f"index_{stem}.faiss"
        meta_path = idx_dir / f"meta_{stem}.json"
        out_path = idx_dir / f"umap_{stem}.json"

        if not idx_path.exists() or not meta_path.exists():
            print(f"[{stem}] skipping: missing {idx_path.name} or {meta_path.name}")
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"[{stem}] error reading meta: {e}")
            continue

        n_avail = len(meta)
        if n_avail == 0:
            print(f"[{stem}] empty metadata")
            continue

        n = min(n_avail, args.max_points)
        print(f"[{stem}] loading {n}/{n_avail} vectors from {idx_path.name}...")
        try:
            index = faiss.read_index(str(idx_path))
            vecs = index.reconstruct_n(0, n)
        except Exception as e:
            print(f"[{stem}] error reading FAISS index: {e}")
            continue

        print(f"[{stem}] projecting {len(vecs)} vectors via {args.method}...")
        coords_raw = project_vectors(
            vecs,
            method=args.method,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
        )
        coords = normalize_coords(coords_raw)

        out_rows: List[dict] = []
        for i in range(n):
            m = meta[i]
            out_rows.append({
                "segment_id": m.get("segment_id", f"{stem}_{i}"),
                "track_id": m.get("track_id", ""),
                "stem": stem,
                "x": float(np.round(coords[i, 0], 4)),
                "y": float(np.round(coords[i, 1], 4)),
            })

        out_path.write_text(json.dumps(out_rows))
        print(f"[{stem}] wrote {len(out_rows)} points -> {out_path.name}")
        total_pts += len(out_rows)

    print(f"\nDone: {total_pts} total 2D nebula points written across stems.")


if __name__ == "__main__":
    main()
