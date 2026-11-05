#!/usr/bin/env python3
"""Merge window metadata from a taste dataset into the kit manifest.

``scripts/build_taste_dataset.py`` labels every real 4-bar window with the
player-facing metadata the phrase selector needs:

* ``chroma``    - 12-d harmonic chroma (same as feature dimensions 0..11),
* ``bass_root`` - integer argmax of the window's bass chroma (0..11),
* ``is_intro``  - the window starts at bar <= 4 or in an INTRO section,
* ``is_outro``  - the window ends within the last four bars of the track,
* ``once``      - ``is_intro or is_outro`` (play this phrase at most once).

This script copies those five fields onto every manifest phrase by joining on
``(track_id, start_bar)``.  It edits the manifest in place and keeps the
pre-merge manifest as ``kit_manifest.json.bak``.  Audio files, phrase ids,
embeddings, nebula coordinates and every other manifest field are left
untouched.

Usage::

    python3 scripts/merge_kit_metadata.py \\
        --dataset /tmp/taste_dataset_160.json \\
        --manifest frontend/public/data/kits/kit_manifest.json

The join must cover 100% of the manifest phrases; a missing window is a hard
error (exit code 1) rather than a silently unlabelled phrase.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

REQUIRED_FIELDS: Tuple[str, ...] = (
    "chroma", "bass_root", "is_intro", "is_outro", "once")
CHROMA_DIM = 12


def _window_index(dataset: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """``(track_id, start_bar)`` -> window row from the dataset's tracks."""
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for track in dataset.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("track_id") or "")
        if not track_id:
            continue
        for window in track.get("windows") or []:
            if not isinstance(window, dict):
                continue
            index[(track_id, int(window.get("start_bar", 0)))] = window
    return index


def _metadata(window: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and copy the five player fields out of one window."""
    chroma = window.get("chroma")
    if (not isinstance(chroma, list) or len(chroma) != CHROMA_DIM
            or not all(isinstance(v, (int, float)) for v in chroma)):
        raise ValueError(f"invalid chroma: {chroma!r}")
    root = window.get("bass_root")
    if not isinstance(root, int) or not 0 <= root < CHROMA_DIM:
        raise ValueError(f"invalid bass_root: {root!r}")
    flags = [window.get(key) for key in ("is_intro", "is_outro", "once")]
    if not all(isinstance(v, bool) for v in flags):
        raise ValueError(f"invalid flags: {flags!r}")
    if flags[2] != (flags[0] or flags[1]):
        raise ValueError(f"once != is_intro or is_outro: {flags!r}")
    return {
        "chroma": [float(v) for v in chroma],
        "bass_root": int(root),
        "is_intro": bool(flags[0]),
        "is_outro": bool(flags[1]),
        "once": bool(flags[2]),
    }


def merge(dataset: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Add the five metadata fields to every phrase; return join statistics."""
    index = _window_index(dataset)
    if not index:
        raise SystemExit("dataset has no track windows")

    total = matched = 0
    missing: List[str] = []
    missing_phrases = 0
    for kit in manifest.get("kits") or []:
        if not isinstance(kit, dict):
            continue
        track_id = str(kit.get("track_id") or "")
        for phrase in kit.get("phrases") or []:
            if not isinstance(phrase, dict):
                continue
            total += 1
            start_bar = int(phrase.get("start_bar", 0))
            window = index.get((track_id, start_bar))
            if window is None:
                missing.append(f"{track_id}@{start_bar}")
                continue
            meta = _metadata(window)
            phrase.update(meta)
            matched += 1
            if not all(field in phrase for field in REQUIRED_FIELDS):
                missing_phrases += 1

    return {
        "phrases": total,
        "matched": matched,
        "missing": missing[:20],
        "missing_count": len(missing),
        "phrases_missing_fields": missing_phrases,
        "match_rate": matched / max(1, total),
    }


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge taste-dataset window metadata into the kit manifest.")
    ap.add_argument("--dataset", default="/tmp/taste_dataset_160.json",
                    help="160-d taste dataset JSON")
    ap.add_argument("--manifest",
                    default="frontend/public/data/kits/kit_manifest.json",
                    help="kit manifest to update in place")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the join without writing the manifest")
    args = ap.parse_args()

    dataset_path = _resolve(args.dataset)
    manifest_path = _resolve(args.manifest)
    if not dataset_path.is_file():
        raise SystemExit(f"--dataset not a file: {dataset_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"--manifest not a file: {manifest_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("kits"), list):
        raise SystemExit(f"{manifest_path}: missing 'kits'")

    stats = merge(dataset, manifest)
    kits = len(manifest.get("kits") or [])
    phrase_counts = Counter(len(kit.get("phrases") or [])
                            for kit in manifest.get("kits") or [])

    print("=== Wub kit metadata merge ===", flush=True)
    print(f"  dataset  : {dataset_path} "
          f"(feature_dim={dataset.get('feature_dim')}, "
          f"tracks={len(dataset.get('tracks') or [])})", flush=True)
    print(f"  manifest : {manifest_path} ({kits} kits, "
          f"{stats['phrases']} phrases)", flush=True)
    print(f"  join     : {stats['matched']}/{stats['phrases']} "
          f"({stats['match_rate']:.2%}) by (track_id, start_bar)", flush=True)
    print(f"  phrases per kit: min={min(phrase_counts) if phrase_counts else 0}, "
          f"max={max(phrase_counts) if phrase_counts else 0}", flush=True)
    print(f"  fields   : {', '.join(REQUIRED_FIELDS)}", flush=True)

    if stats["missing_count"] or stats["phrases_missing_fields"]:
        print(f"  ! unmatched {stats['missing_count']} phrase(s), e.g. "
              f"{', '.join(stats['missing'])}", flush=True)
        print("  ! refusing to write a partially labelled manifest", flush=True)
        return 1

    if args.dry_run:
        print("  dry-run  : manifest not written", flush=True)
        return 0

    backup = Path(str(manifest_path) + ".bak")
    shutil.copy2(manifest_path, backup)
    print(f"  backup   : {backup} (pre-merge copy)", flush=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  wrote    : {manifest_path} "
          f"({manifest_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    print("  status   : PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
