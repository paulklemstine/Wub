"""Corpus check: does loudness-bucket predict drum activity? What about density?

Answers whether the energy (loudness) conditioning signal CAN steer beats:
if P(drum-active | loud-bucket) is flat in the corpus, no model can learn
an energy->activity mapping from it, and the cond signal must change
(e.g. to onset-density buckets) instead of more epochs.
Usage: python3 scripts/corpus_energy_check.py --processed data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()

    from wub_core.model_transformer import energy_bucket

    files = sorted(
        p for p in Path(args.processed).glob("*.json") if not p.name.startswith(("_", "meta_"))
    )
    if args.max_tracks:
        files = files[: args.max_tracks]
    # per loudness-bucket: tatums, drum-active tatums
    loud_tots = [0] * 8
    loud_act = [0] * 8
    # per density-bucket (segments present across 4 stems, 0..8): same
    dens_tots = [0] * 9
    dens_act = [0] * 9
    n = 0
    for p in files:
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        segs = {s.get("id"): s for s in doc.get("segments", []) or []}
        for tat in doc.get("tatums", []) or []:
            segmap = tat.get("stem_segments", {}) or {}
            louds = []
            dens = 0
            drum_on = 0
            for st in ("drums", "bass", "other", "vocals"):
                ids = segmap.get(st) or []
                if not ids:
                    continue
                sg = segs.get(ids[0])
                if sg is None or sg.get("is_silent", False):
                    continue
                dens += 1
                if st == "drums":
                    drum_on = 1
                try:
                    louds.append(float(sg.get("loudness_max", float("nan"))))
                except (TypeError, ValueError):
                    pass
            lb = energy_bucket(max(louds) if louds else -90.0)
            loud_tots[lb] += 1
            loud_act[lb] += drum_on
            dens_tots[dens] += 1
            dens_act[dens] += drum_on
            n += 1
    print(f"tatums: {n}")
    print("loud-bucket : drum-active-rate (n)")
    for b in range(8):
        t = loud_tots[b]
        print(f"  {b}: {loud_act[b] / max(t, 1):.3f} ({t})")
    print("density-bucket (stems present) : drum-active-rate (n)")
    for d in range(9):
        t = dens_tots[d]
        print(f"  {d}: {dens_act[d] / max(t, 1):.3f} ({t})")


if __name__ == "__main__":
    main()
