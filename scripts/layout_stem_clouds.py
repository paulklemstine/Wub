"""Lay the four per-stem PCA clouds into separate quadrants of one map.

Independent per-stem PCAs each fill the whole 0..1 plane, so drawing them
together makes four overlapping clouds.  This keeps each stem's within-cloud
geometry but maps it into its own quadrant:

    drums  top-left      bass   top-right
    other  bottom-left   vocals bottom-right

Distances are only meaningful *within* a stem (as the per-stem map intends);
the quadrant placement makes the four spaces legible and gives each reticle a
clear region to drag in.

Usage:
    python3 scripts/layout_stem_clouds.py \
      --manifest frontend/public/data/kits/kit_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

QUADRANTS = {
    "drums": (0.06, 0.46, 0.54, 0.94),
    "bass": (0.54, 0.94, 0.54, 0.94),
    "other": (0.06, 0.46, 0.06, 0.46),
    "vocals": (0.54, 0.94, 0.06, 0.46),
}

SOURCE_LO = 0.08
SOURCE_SPAN = 0.84


def layout_point(stem: str, x: float, y: float) -> tuple[float, float]:
    """Map a 0.08..0.92 PCA point into its stem's quadrant."""
    x0, x1, y0, y1 = QUADRANTS[stem]
    nx = max(0.0, min(1.0, (float(x) - SOURCE_LO) / SOURCE_SPAN))
    ny = max(0.0, min(1.0, (float(y) - SOURCE_LO) / SOURCE_SPAN))
    return round(x0 + nx * (x1 - x0), 4), round(y0 + ny * (y1 - y0), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    args = ap.parse_args()

    path = Path(args.manifest)
    manifest = json.loads(path.read_text())
    moved = 0
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            stem_xy = phrase.get("stem_xy")
            if not isinstance(stem_xy, dict):
                continue
            for stem in QUADRANTS:
                point = stem_xy.get(stem)
                if not isinstance(point, dict):
                    continue
                point["x"], point["y"] = layout_point(
                    stem, point.get("x", 0.5), point.get("y", 0.5)
                )
                moved += 1
    path.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"laid out {moved} stem points in 4 quadrants; {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
