#!/usr/bin/env python3
"""Additive merge of ears scores into taste_scores.json + taste_dataset.json.

Preserves the dual-writer contract: `kits` (score_kits) and `stem_next`
(score_stem_kits) are never overwritten — `ears` is added as a new key, and
the taste dataset gets reserved dims 41..47 filled in a NEW file.

Usage:
    PYTHONPATH=. python3 scripts/ears_merge.py \
        --ears /tmp/wubtrain/ears/ears_scores.json \
        --scores frontend/public/data/kits/taste_scores.json \
        --out-scores /tmp/wubtrain/taste_scores_ears.json

    PYTHONPATH=. python3 scripts/ears_merge.py \
        --ears /tmp/wubtrain/ears/ears_scores.json \
        --dataset /tmp/wubtrain/taste_dataset.json \
        --out-dataset /tmp/wubtrain/taste_dataset_ears.json

Smoke test (shipped bank, copy — never mutate the shipped file in place):
    cp frontend/public/data/kits/taste_scores.json /tmp/opencode/taste_copy.json
    PYTHONPATH=. python3 scripts/ears_merge.py \
        --ears /tmp/opencode/ears_scores_smoke.json \
        --scores /tmp/opencode/taste_copy.json \
        --out-scores /tmp/opencode/taste_scores_ears.json
"""
import argparse
import json
import sys
from pathlib import Path


def merge_scores(scores_path, ears_path, out_path):
    scores = json.loads(Path(scores_path).read_text())
    ears_doc = json.loads(Path(ears_path).read_text())
    ears = ears_doc.get("ears", {})
    if not isinstance(ears, dict) or not ears:
        print("[ears_merge] ears file has no ears map; aborting", file=sys.stderr)
        return 1
    # Preserve both dual-writer keys explicitly.
    for key in ("kits", "stem_next"):
        if key not in scores:
            print(f"[ears_merge] WARNING: scores file missing {key!r} (dual-writer file?)")
    scores = dict(scores)
    scores["ears"] = ears
    scores["ears_model"] = ears_doc.get("model", "ears_dsp_v1")
    Path(out_path).write_text(json.dumps(scores))
    print(f"[ears_merge] scores: {len(scores.get('kits', {}))} kits, "
          f"{len(scores.get('stem_next', {}))} stem_next, +{len(ears)} ears -> {out_path}")


def merge_dataset(dataset_path, ears_path, manifest_path, out_path):
    dataset = json.loads(Path(dataset_path).read_text())
    ears_doc = json.loads(Path(ears_path).read_text())
    ears = ears_doc.get("ears", {})
    # phrase_id -> (track_id, start_bar) from the manifest.
    manifest = json.loads(Path(manifest_path).read_text())
    loc = {}
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            loc[phrase["id"]] = (kit.get("track_id"), phrase.get("start_bar"))
    filling = {}  # (track_id, start_bar) -> reserved list
    for pid, entry in ears.items():
        key = loc.get(pid)
        if key is not None and entry.get("reserved"):
            filling[key] = entry["reserved"]

    n_filled = 0
    n_total = 0
    for collection in (dataset.get("tracks", []),):
        for track in collection:
            tid = track.get("track_id")
            for window in track.get("windows", []):
                n_total += 1
                reserved = filling.get((tid, window.get("start_bar")))
                if reserved and isinstance(window.get("features"), list) and len(window["features"]) >= 48:
                    for i, v in enumerate(reserved[:7]):
                        try:
                            window["features"][41 + i] = float(v)
                        except (TypeError, ValueError):
                            pass
                    n_filled += 1
    for phrase in dataset.get("kit_phrases", []):
        # Locate via phrase_id directly (exact join, no (track,start) guess).
        n_total += 1
        reserved = ears.get(phrase.get("phrase_id", ""), {}).get("reserved")
        if reserved and isinstance(phrase.get("features"), list) and len(phrase["features"]) >= 48:
            for i, v in enumerate(reserved[:7]):
                try:
                    phrase["features"][41 + i] = float(v)
                except (TypeError, ValueError):
                    pass
            n_filled += 1
    Path(out_path).write_text(json.dumps(dataset))
    print(f"[ears_merge] dataset: filled {n_filled}/{n_total} windows "
          f"(dims 41..47) -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ears", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out-scores", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out-dataset", default=None)
    ap.add_argument("--manifest", default=None, help="kit_manifest.json (dataset join)")
    args = ap.parse_args(argv)
    rc = 0
    if args.scores:
        if not args.out_scores:
            print("[ears_merge] --scores needs --out-scores", file=sys.stderr)
            return 2
        rc = merge_scores(args.scores, args.ears, args.out_scores) or rc
    if args.dataset:
        if not args.out_dataset or not args.manifest:
            print("[ears_merge] --dataset needs --out-dataset and --manifest", file=sys.stderr)
            return 2
        merge_dataset(args.dataset, args.ears, args.manifest, args.out_dataset)
    if not (args.scores or args.dataset):
        print("[ears_merge] nothing to merge (pass --scores and/or --dataset)", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
