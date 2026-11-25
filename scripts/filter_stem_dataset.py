"""Filter a per-stem bar dataset to the tracks in a shipping manifest.

Usage:
    python3 scripts/filter_stem_dataset.py \
      --bars data/stem_bars --manifest kits_cluster/kit_manifest.json \
      --out data/stem_ship
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "other", "vocals")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/stem_bars")
    ap.add_argument("--manifest", default="kits_cluster/kit_manifest.json")
    ap.add_argument("--out", default="data/stem_ship")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    tracks = {kit["track_id"] for kit in manifest.get("kits", [])}
    print(f"shipping tracks: {len(tracks)}")

    for stem in STEMS:
        src = Path(args.bars) / stem
        idx = json.loads((src / "bar_index.json").read_text())
        arr = np.load(src / idx["mel_file"], mmap_mode="r")
        mask = np.array([tid in tracks for tid in idx["track_id"]], dtype=bool)
        n = int(mask.sum())
        dst = Path(args.out) / stem
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "mel.npy", np.asarray(arr[mask]))
        out = {k: v for k, v in idx.items() if k not in {"track_id", "phrase_id", "start_bar",
                                                        "bpm", "key", "style", "section",
                                                        "energy", "energy_db", "duration",
                                                        "is_intro", "is_outro", "n"}}
        for field in ("phrase_id", "track_id", "start_bar", "bpm", "key", "style", "section",
                      "energy", "energy_db", "duration", "is_intro", "is_outro"):
            values = np.asarray(idx[field])[mask]
            out[field] = values.tolist()
        out["n"] = n
        (dst / "bar_index.json").write_text(json.dumps(out, separators=(",", ":")))
        print(f"{stem}: {n} / {len(mask)} phrases -> {dst}")


if __name__ == "__main__":
    main()
