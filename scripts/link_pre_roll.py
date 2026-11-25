"""Attach existing ``*_pre.opus`` sidecars to a kit manifest.

``add_pre_roll.py`` extracts the pre-bar tails in worker processes; those
mutations do not propagate back to the parent manifest.  This pass reads the
files that actually landed on disk and writes the ``pre_roll`` fields, so the
player and the manifest never disagree.

Usage:
    python3 scripts/link_pre_roll.py --manifest kits_cluster/kit_manifest.json \
      --kits-dir kits_cluster
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kits-dir", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    kits_dir = Path(args.kits_dir)
    manifest = json.loads(manifest_path.read_text())

    linked = missing = 0
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            max_pre = 0.0
            for asset in (phrase.get("stems") or {}).values():
                rel = asset.get("file")
                if not rel:
                    continue
                pre_rel = str(Path(rel).with_name(Path(rel).stem + "_pre.opus"))
                pre_path = kits_dir / pre_rel
                if not pre_path.is_file():
                    missing += 1
                    continue
                duration = float(sf.info(str(pre_path)).duration)
                asset["pre_roll"] = {
                    "file": pre_rel,
                    "duration_sec": round(duration, 6),
                }
                max_pre = max(max_pre, duration)
                linked += 1
            phrase["pre_roll_sec"] = round(max_pre, 6)

    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    print(json.dumps({
        "linked": linked,
        "missing": missing,
        "manifest_mb": round(manifest_path.stat().st_size / 1e6, 2),
    }))


if __name__ == "__main__":
    main()
