#!/usr/bin/env python3
"""Remove tracks that failed reconstruction QC from a staged kit bank.

The QC report is an audit of the full candidate bank.  A track with a real
waveform/alignment failure must not ship, but one bad track should not discard
the other 400.  This script removes the failed tracks from the manifest and
each cluster pool, optionally prunes their staging audio directories, and
writes a small audit file.

Usage:
    python3 scripts/filter_failed_kits.py \
      --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
      --report /tmp/wubtrain/arc8_stage/qc/kit_qc.json \
      --out /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
      --kits-dir /tmp/wubtrain/arc8_stage/kits_bank \
      --prune-audio
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kits-dir", default="")
    ap.add_argument("--prune-audio", action="store_true")
    ap.add_argument("--audit", default="")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    report_path = Path(args.report)
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())

    failed: dict[str, list[str]] = {}
    for failure in report.get("failures") or []:
        tid = str(failure.get("track_id") or "")
        if not tid:
            continue
        failed.setdefault(tid, []).append(
            f"{failure.get('phrase')}: {', '.join(failure.get('reasons') or [])}"
        )

    kits = list(manifest.get("kits") or [])
    keep = [kit for kit in kits if str(kit.get("track_id")) not in failed]
    removed = [kit for kit in kits if str(kit.get("track_id")) in failed]
    keep_ids = {str(kit.get("track_id")) for kit in keep}
    removed_ids = {str(kit.get("track_id")) for kit in removed}

    if not keep:
        raise SystemExit("refusing to remove every kit")
    if len(keep) < 300:
        raise SystemExit(f"only {len(keep)} kits remain; expected at least 300")

    clusters = []
    for cluster in manifest.get("clusters") or []:
        pool = [str(t) for t in cluster.get("kits") or [] if str(t) in keep_ids]
        if not pool:
            raise SystemExit(f"cluster {cluster.get('id')} has no kits after QC filtering")
        phrase_count = sum(
            len(kit.get("phrases") or []) for kit in keep
            if str(kit.get("track_id")) in set(pool)
        )
        updated = dict(cluster)
        updated["kits"] = pool
        updated["pool_phrases"] = phrase_count
        clusters.append(updated)

    manifest["kits"] = keep
    manifest["clusters"] = clusters
    out = Path(args.out)
    out.write_text(json.dumps(manifest, indent=2))

    pruned = 0
    if args.prune_audio and args.kits_dir:
        root = Path(args.kits_dir)
        for tid in removed_ids:
            target = root / "kits" / tid
            if target.is_dir():
                shutil.rmtree(target)
                pruned += 1

    audit = {
        "input_kits": len(kits),
        "kept_kits": len(keep),
        "removed_kits": sorted(removed_ids),
        "removed_phrases": sum(len(k.get("phrases") or []) for k in removed),
        "pruned_audio_dirs": pruned,
        "failures": {tid: failed[tid] for tid in sorted(removed_ids)},
        "clusters": len(clusters),
    }
    audit_path = Path(args.audit) if args.audit else out.parent / "qc" / "filtered_kits.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))

    print(f"input kits={len(kits)} kept={len(keep)} removed={len(removed_ids)}")
    print(f"removed: {', '.join(sorted(removed_ids)) or '(none)'}")
    print(f"pruned audio dirs={pruned}")
    print(f"audit -> {audit_path}")


if __name__ == "__main__":
    main()
