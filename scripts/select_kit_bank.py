"""Select a style-balanced shipping bank from a full-corpus kit manifest.

Keeps the best kits per style by musical completeness rather than at random:
phrase count, section coverage, energy, stem presence and embedding novelty.
Copies only the selected kits' audio and writes a new manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def novelty(phrases: list[dict]) -> float:
    embs = [p.get("embedding") for p in phrases[:8]]
    embs = [e for e in embs if isinstance(e, list) and len(e) == 64]
    if len(embs) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            dot = sum(a * b for a, b in zip(embs[i], embs[j]))
            na = math.sqrt(sum(a * a for a in embs[i]))
            nb = math.sqrt(sum(b * b for b in embs[j]))
            if na > 1e-9 and nb > 1e-9:
                total += 1.0 - dot / (na * nb)
                n += 1
    return total / max(1, n)


def kit_score(kit: dict) -> float:
    phrases = kit.get("phrases") or []
    if not phrases:
        return -1.0
    count = len(phrases)
    sections = len({p.get("section") for p in phrases})
    energy = sum(float(p.get("energy", 0.0)) for p in phrases) / count
    peaks = [
        float(asset.get("peak", 0.0))
        for p in phrases
        for asset in (p.get("stems") or {}).values()
    ]
    mean_peak = sum(peaks) / len(peaks) if peaks else 0.0
    return (
        2.2 * (count / 8.0)
        + 1.2 * (sections / 5.0)
        + 1.0 * energy
        + 0.8 * min(1.0, mean_peak / 0.5)
        + 0.8 * novelty(phrases)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source-dir", required=True, help="dir containing kits/<tid>/")
    ap.add_argument("--out", required=True, help="output dir for kits/ + manifest")
    ap.add_argument("--per-style", type=int, default=150)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    by_style: dict[str, list[dict]] = {}
    for kit in manifest.get("kits", []):
        by_style.setdefault(str(kit.get("style", "house")), []).append(kit)

    selected: list[dict] = []
    for style, kits in sorted(by_style.items()):
        ranked = sorted(kits, key=kit_score, reverse=True)[: args.per_style]
        print(f"{style}: {len(kits)} candidates -> {len(ranked)} selected", flush=True)
        selected.extend(ranked)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source_dir) / "kits"
    copied = 0
    for kit in selected:
        tid = kit["track_id"]
        dst = out / "kits" / tid
        dst.mkdir(parents=True, exist_ok=True)
        for phrase in kit.get("phrases") or []:
            for asset in (phrase.get("stems") or {}).values():
                rel = asset.get("file")
                if not rel:
                    continue
                s = src / tid / Path(rel).name
                if s.exists():
                    shutil.copy2(s, dst / s.name)
                    copied += 1

    out_manifest = {**manifest, "kits": selected}
    # Keep only fields the client understands.
    (out / "kit_manifest.json").write_text(json.dumps(out_manifest, separators=(",", ":")))
    size = sum(f.stat().st_size for f in (out / "kits").rglob("*.opus"))
    print(f"selected {len(selected)} kits, {sum(len(k['phrases']) for k in selected)} phrases, "
          f"{copied} files, {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
