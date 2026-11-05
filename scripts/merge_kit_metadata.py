"""Merge taste-dataset metadata into a kit manifest.

The cluster bank is selected from the full-corpus manifest, which predates the
taste feature pass.  This adds the per-phrase fields the player needs:
``chroma``, ``bass_root``, ``is_intro``, ``is_outro`` and ``once``.  It does
not add the 160-d feature vector (the player never reads it; it lives in
``taste_scores.json``).

The pure ``merge`` helper is used by the layout contract tests; the CLI wraps
it with a file-based integrity check.

Usage:
    PYTHONPATH=. python3 scripts/merge_kit_metadata.py \
      --manifest kits_cluster/kit_manifest.json \
      --dataset taste_dataset_cluster.json \
      --out kits_cluster/kit_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("chroma", "bass_root", "is_intro", "is_outro", "once")


def merge(dataset: dict, manifest: dict) -> dict:
    """Copy the available metadata fields into the manifest phrases in place."""
    rows = dataset.get("kit_phrases") or []
    meta = {row["phrase_id"]: row for row in rows if "phrase_id" in row}
    total = 0
    matched = 0
    missing: list[str] = []
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            total += 1
            row = meta.get(phrase.get("id"))
            if row is None:
                missing.append(str(phrase.get("id", "?")))
                continue
            for field in FIELDS:
                if field in row:
                    phrase[field] = row[field]
            # "once" is derived from the source position; keep a dataset flag
            # only when it is explicitly true.
            phrase["once"] = bool(
                row.get("is_intro") or row.get("is_outro") or row.get("once")
            )
            matched += 1
    return {
        "total": total,
        "matched": matched,
        "missing_count": len(missing),
        "missing": missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    dataset = json.loads(Path(args.dataset).read_text())
    stats = merge(dataset, manifest)

    if stats["missing"]:
        print(f"missing metadata for {stats['missing_count']} phrases: {stats['missing'][:5]}")
    if stats["total"] == 0:
        raise SystemExit("manifest has no phrases")
    ratio = stats["matched"] / stats["total"]
    print(f"merged {stats['matched']}/{stats['total']} phrases ({ratio:.1%})")
    if ratio < 0.99:
        raise SystemExit("metadata merge below 99%; refusing to write")

    manifest["metadata_merged"] = stats["matched"]
    Path(args.out).write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
