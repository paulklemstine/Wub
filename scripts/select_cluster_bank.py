"""Select a cluster-dense shipping bank for the per-stem remix engine.

Instead of spreading kits across every key (which leaves 0-12 phrases in a
same-key pool), this selects all kits that belong to 16 style-balanced
key/tempo clusters.  A cluster is a session key + tempo plus its Camelot
neighbour keys, all within 1% of the session tempo.  The same kit can appear
in more than one cluster's pool, but it is copied and listed once.

The output manifest adds a ``clusters`` array consumed by the player:

    {"id": "breaks_f_minor_129", "style": "breaks", "key": "F minor",
     "bpm": 129.0, "kits": ["<track_id>", ...],
     "neighbour_keys": ["Ab major", ...], "pool_phrases": 292}

Usage (on the laptop, where kits_all2 and the stems live):

    PYTHONPATH=. python3 scripts/select_cluster_bank.py \
      --manifest kits_all2/kit_manifest.json --source-dir kits_all2 \
      --out kits_cluster --workers 8
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.select_kit_bank import kit_score
from wub_core.composer import KEY_TO_CAMELOT

# 16 style-balanced clusters: four per style.  The tempo is the session
# tempo; the pool is every kit within 1% of it in the cluster key or a
# Camelot neighbour.
DEFAULT_CLUSTERS = [
    ("breaks", "F minor", 129),
    ("breaks", "E minor", 129),
    ("breaks", "E minor", 126),
    ("breaks", "F minor", 126),
    ("cyber", "F minor", 140),
    ("cyber", "D minor", 140),
    ("cyber", "A minor", 140),
    ("cyber", "E minor", 140),
    ("house", "F minor", 120),
    ("house", "D minor", 123),
    ("house", "C minor", 120),
    ("house", "D minor", 120),
    ("wave", "F minor", 88),
    ("wave", "F# minor", 110),
    ("wave", "D minor", 96),
    ("wave", "E minor", 99),
]

# Canonical spelling for each Camelot number/mode so Ab/G# and Bb/A# do not
# split one key into two pools.
CANONICAL = {
    (1, "A"): "G# minor",
    (1, "B"): "B major",
    (2, "A"): "D# minor",
    (2, "B"): "F# major",
    (3, "A"): "A# minor",
    (3, "B"): "C# major",
    (4, "A"): "F minor",
    (4, "B"): "G# major",
    (5, "A"): "C minor",
    (5, "B"): "D# major",
    (6, "A"): "G minor",
    (6, "B"): "A# major",
    (7, "A"): "D minor",
    (7, "B"): "F major",
    (8, "A"): "A minor",
    (8, "B"): "C major",
    (9, "A"): "E minor",
    (9, "B"): "G major",
    (10, "A"): "B minor",
    (10, "B"): "D major",
    (11, "A"): "F# minor",
    (11, "B"): "A major",
    (12, "A"): "C# minor",
    (12, "B"): "E major",
}

ALIASES = {
    "Ab minor": "G# minor", "G# minor": "G# minor",
    "Eb minor": "D# minor", "D# minor": "D# minor",
    "Bb minor": "A# minor", "A# minor": "A# minor",
    "Db major": "C# major", "C# major": "C# major",
    "Ab major": "G# major", "G# major": "G# major",
    "Eb major": "D# major", "D# major": "D# major",
    "Bb major": "A# major", "A# major": "A# major",
    "Db minor": "C# minor", "C# minor": "C# minor",
}


def canonical_key(key: str) -> str:
    if key in ALIASES:
        return ALIASES[key]
    camelot = KEY_TO_CAMELOT.get(key)
    if camelot and camelot in CANONICAL:
        return CANONICAL[camelot]
    return key


def neighbour_keys(key: str) -> list[str]:
    """The key itself plus its Camelot neighbours (same mode +/-1, relative)."""
    camelot = KEY_TO_CAMELOT.get(key)
    if not camelot:
        return [canonical_key(key)]
    number, mode = camelot
    wanted = {
        (number, mode),
        ((number - 2) % 12 + 1, mode),
        (number % 12 + 1, mode),
        (number, "B" if mode == "A" else "A"),
    }
    names = [CANONICAL[c] for c in wanted if c in CANONICAL]
    # Keep the session key first for readable manifests.
    own = canonical_key(key)
    names = [own] + sorted(n for n in names if n != own)
    return names


def copy_kit(kit: dict, src: Path, dst: Path) -> int:
    tid = kit["track_id"]
    out_dir = dst / "kits" / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for phrase in kit.get("phrases") or []:
        for asset in (phrase.get("stems") or {}).values():
            rel = asset.get("file")
            if not rel:
                continue
            source = src / tid / Path(rel).name
            if source.is_file():
                shutil.copy2(source, out_dir / source.name)
                copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="kits_all2/kit_manifest.json")
    ap.add_argument("--source-dir", default="kits_all2")
    ap.add_argument("--out", default="kits_cluster")
    ap.add_argument("--tempo-tolerance", type=float, default=0.01,
                    help="max |kit_bpm - session_bpm| / session_bpm (1%%)")
    ap.add_argument("--min-kits", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    all_kits = manifest.get("kits") or []
    by_key: dict[str, list[dict]] = {}
    for kit in all_kits:
        by_key.setdefault(canonical_key(kit["key"]), []).append(kit)

    clusters = []
    selected: dict[str, dict] = {}
    for style, key, bpm in DEFAULT_CLUSTERS:
        names = neighbour_keys(key)
        pool = []
        for name in names:
            for kit in by_key.get(name, []):
                if abs(float(kit["bpm"]) - bpm) / bpm <= args.tempo_tolerance:
                    pool.append(kit)
        # De-duplicate by track id (a kit can be reached through two keys).
        uniq = {kit["track_id"]: kit for kit in pool}
        if len(uniq) < args.min_kits:
            print(f"skip {style} {key} {bpm}: only {len(uniq)} kits", flush=True)
            continue
        # Keep every kit that still has at least four phrases after the
        # integrity pass; the pool density is the point of this bank.
        ordered = sorted(uniq.values(), key=kit_score, reverse=True)
        tids = [kit["track_id"] for kit in ordered]
        phrases = sum(len(kit.get("phrases") or []) for kit in ordered)
        cluster = {
            "id": f"{style}_{canonical_key(key).lower().replace(' ', '_').replace('#', 's')}_{bpm}",
            "style": style,
            "key": canonical_key(key),
            "bpm": float(bpm),
            "neighbour_keys": names,
            "kits": tids,
            "pool_phrases": phrases,
        }
        clusters.append(cluster)
        for kit in ordered:
            selected[kit["track_id"]] = kit
        print(f"{cluster['id']:28s} {len(tids):3d} kits {phrases:4d} phrases  "
              f"keys={names}", flush=True)

    print(f"union: {len(selected)} kits, "
          f"{sum(len(k.get('phrases') or []) for k in selected.values())} phrases",
          flush=True)

    if args.dry_run:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source_dir) / "kits"

    # Integrity: never ship a manifest entry whose audio is missing.
    validated: list[dict] = []
    dropped_phrases = dropped_kits = 0
    for kit in selected.values():
        phrases = []
        for phrase in kit.get("phrases") or []:
            ok = True
            for asset in (phrase.get("stems") or {}).values():
                rel = asset.get("file")
                if not rel or not (src / kit["track_id"] / Path(rel).name).is_file():
                    ok = False
                    break
            if ok:
                phrases.append(phrase)
            else:
                dropped_phrases += 1
        if len(phrases) >= 4:
            validated.append({**kit, "phrases": phrases})
        else:
            dropped_kits += 1
    valid_ids = {kit["track_id"] for kit in validated}
    for cluster in clusters:
        cluster["kits"] = [tid for tid in cluster["kits"] if tid in valid_ids]
        cluster["pool_phrases"] = sum(
            len(next(k for k in validated if k["track_id"] == tid)["phrases"])
            for tid in cluster["kits"]
        )
    clusters = [c for c in clusters if len(c["kits"]) >= args.min_kits]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        copied = sum(ex.map(lambda kit: copy_kit(kit, src, out), validated))

    out_manifest = {
        **{k: v for k, v in manifest.items() if k != "kits"},
        "version": 2,
        "bank": "cluster_v1",
        "clusters": clusters,
        "kits": validated,
    }
    (out / "kit_manifest.json").write_text(json.dumps(out_manifest, separators=(",", ":")))
    size = sum(f.stat().st_size for f in (out / "kits").rglob("*.opus"))
    print(f"selected {len(validated)} kits, "
          f"{sum(len(k['phrases']) for k in validated)} phrases, "
          f"{copied} files, {size / 1e6:.1f} MB", flush=True)
    if dropped_phrases or dropped_kits:
        print(f"integrity: dropped {dropped_phrases} phrases, {dropped_kits} kits", flush=True)


if __name__ == "__main__":
    main()
