"""Cross-song swap test: rebuild tracks using ONLY other songs' slices.

For each sampled stemmed track, remap its bars twice:
  self  = full global vocab (own slices allowed)
  swap  = global vocab with this track's slices removed
Reports per-stem fidelity (mean cosine similarity of matched slices),
the self-vs-swap gap, and vocab coverage. A reusable vocabulary keeps
the gap small: songs must be rebuildable without their own slices.

Usage: PYTHONPATH=. python3 scripts/eval_cross_song.py [--max-tracks 40 --bars 16]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List

from wub_core.global_vocabulary import GlobalVocabulary, GlobalVocabularyBuilder


def _load_track(path: Path, max_bars: int):
    raw = json.loads(path.read_bytes())
    tid = str(raw.get("track_id") or path.stem)
    tatums = (raw.get("tatums") or [])[: max(16, max_bars * 16)]
    return {
        "track_id": tid,
        "segments": raw.get("segments") or [],
        "tatums": tatums,
        "stems_available": list(raw.get("stems_available") or ["other"]),
        "is_stemmed": len(raw.get("stems_available") or []) > 1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--max-tracks", type=int, default=40)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/cross_song_eval.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    full = GlobalVocabulary.load(Path(args.vocab))
    full.build_index()
    builder = GlobalVocabularyBuilder()

    files = sorted(p for p in Path(args.processed).glob("*.json") if not p.name.startswith(("_", "meta_")))
    stemmed = []
    for jf in files:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        if len(raw.get("stems_available") or []) > 1 and (raw.get("tatums") or []):
            stemmed.append(jf)
    rng.shuffle(stemmed)
    chosen = stemmed[: args.max_tracks]

    per_stem_self: Dict[str, List[float]] = {}
    per_stem_swap: Dict[str, List[float]] = {}
    used_ids = set()
    donor_hits: Dict[str, int] = {}
    n_bars = 0
    for jf in chosen:
        t = _load_track(jf, args.bars)
        segs = t["segments"][: args.bars * 64]
        tats = t["tatums"][: args.bars * 16]
        if not segs or not tats:
            continue
        _, self_fid = builder.remap_bars_fast(segs, tats, full)
        filt = GlobalVocabulary(
            vocab_by_stem={s: [v for v in full.vocab_by_stem.get(s, []) if v.track_id != t["track_id"]]
                           for s in ("drums", "bass", "other", "vocals")},
            budget=dict(full.budget),
        )
        filt.build_index()
        swap_bars, swap_fid = builder.remap_bars_fast(segs, tats, filt)
        for s, v in self_fid.items():
            per_stem_self.setdefault(s, []).append(float(v))
        for s, v in swap_fid.items():
            per_stem_swap.setdefault(s, []).append(float(v))
        n_bars += len(swap_bars)
        for b in swap_bars:
            for s, steps in b.items():
                for gid in steps:
                    if gid:
                        used_ids.add(gid)

    def mean(xs):
        return round(sum(xs) / max(1, len(xs)), 4) if xs else 0.0

    out = {
        "tracks": len(chosen),
        "bars": n_bars,
        "per_stem": {
            s: {"self": mean(per_stem_self.get(s, [])), "swapped": mean(per_stem_swap.get(s, []))}
            for s in ("drums", "bass", "other", "vocals")
        },
        "coverage": round(len(used_ids) / max(1, sum(len(v) for v in full.vocab_by_stem.values())), 4),
        "n_used_slices": len(used_ids),
    }
    gaps = [out["per_stem"][s]["self"] - out["per_stem"][s]["swapped"] for s in ("drums", "bass", "other", "vocals")]
    out["mean_swap_gap"] = round(sum(gaps) / len(gaps), 4)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"cross-song swap: tracks={out['tracks']} bars={n_bars}", flush=True)
    for s, v in out["per_stem"].items():
        print(f"  {s}: self={v['self']} swapped={v['swapped']}", flush=True)
    print(f"  mean_swap_gap={out['mean_swap_gap']} coverage={out['coverage']} ({out['n_used_slices']} slices)", flush=True)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
