#!/usr/bin/env python3
"""Add deterministic 2D PCA coordinates to every kit phrase.

Reads frontend/public/data/kits/kit_manifest.json, projects the 64-d phrase
embeddings to two components, robust-scales each component with the 2nd/98th
percentiles, clips, and maps into the 0.08..0.92 nebula range.

A one-time .bak copy is kept next to the manifest.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frontend" / "public" / "data" / "kits" / "kit_manifest.json"
BACKUP = Path(str(MANIFEST) + ".bak")


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    phrases: list[dict] = []
    for kit in data["kits"]:
        phrases.extend(kit["phrases"])

    if not phrases:
        raise SystemExit("kit manifest contains no phrases")

    embeddings = np.asarray([p["embedding"] for p in phrases], dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise SystemExit(f"unexpected embedding matrix shape: {embeddings.shape}")

    pca = PCA(n_components=2, random_state=505, svd_solver="full")
    coords = pca.fit_transform(embeddings)

    # Robust-scale each component with the 2nd/98th percentiles, clip, map.
    scaled = np.empty_like(coords)
    for axis in range(2):
        lo, hi = np.percentile(coords[:, axis], [2.0, 98.0])
        if hi <= lo:
            norm = np.full(coords.shape[0], 0.5, dtype=np.float64)
        else:
            norm = np.clip((coords[:, axis] - lo) / (hi - lo), 0.0, 1.0)
        scaled[:, axis] = 0.08 + norm * 0.84

    if not BACKUP.exists():
        shutil.copy2(MANIFEST, BACKUP)

    for phrase, (x, y) in zip(phrases, scaled):
        phrase["x"] = round(float(x), 4)
        phrase["y"] = round(float(y), 4)

    with MANIFEST.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    ratios = pca.explained_variance_ratio_
    print(f"phrases: {len(phrases)}  embedding dimensions: {embeddings.shape[1]}")
    print(f"explained variance PC1: {ratios[0]:.6f}")
    print(f"explained variance PC2: {ratios[1]:.6f}")
    print(f"explained variance total: {float(ratios.sum()):.6f}")
    print(f"x range: {scaled[:, 0].min():.4f} .. {scaled[:, 0].max():.4f}")
    print(f"y range: {scaled[:, 1].min():.4f} .. {scaled[:, 1].max():.4f}")
    print(f"backup: {BACKUP}")


if __name__ == "__main__":
    main()
