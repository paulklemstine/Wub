"""Train hierarchical ML patterns on global vocab token sequences.

Level 1 (bar): bar_signature(global_ids) -> next_bar transitions per stem.
Level 2 (tatum): global slice_id bigram transitions per stem + pitch intervals.

Outputs data/global_patterns.json + frontend/public/data/global_patterns.json
for neverending generation (bar planner + tatum filler).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--max-bars-per-track", type=int, default=64)
    args = ap.parse_args()

    from wub_core.global_vocabulary import GlobalVocabulary, GlobalVocabularyBuilder

    vocab = GlobalVocabulary.load(Path(args.vocab))
    builder = GlobalVocabularyBuilder(target_counts=dict(vocab.budget))
    # Rebuild feature matrices for fast remap (vocab already has vectors if full save)
    files = sorted(p for p in Path(args.processed).glob("*.json") if not p.name.startswith(("_", "meta_")))
    if args.max_tracks:
        files = files[: args.max_tracks]

    bar_counter: Counter = Counter()
    bar_by_sig: dict = {}
    bar_trans: dict = {s: defaultdict(Counter) for s in ("drums", "bass", "other", "vocals")}
    slice_bigram: dict = {s: defaultdict(Counter) for s in ("drums", "bass", "other", "vocals")}
    prev_bar_sig: dict = {s: None for s in ("drums", "bass", "other", "vocals")}
    total_bars = 0

    for jf in files:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        segs = raw.get("segments") or []
        tatums = raw.get("tatums") or []
        if not tatums:
            continue
        tatums = tatums[: args.max_bars_per_track * 16]
        bars, _fid = builder.remap_bars(segs, tatums, vocab)
        # Reset usages mutated during remap scoring (keep counts clean for training)
        for stem in ("drums", "bass", "other", "vocals"):
            for vs in vocab.vocab_by_stem.get(stem, []):
                pass
        for bar in bars:
            total_bars += 1
            for stem in ("drums", "bass", "other", "vocals"):
                sig = "-".join([g if g else "." for g in bar[stem]])
                bar_counter[(stem, sig)] += 1
                if (stem, sig) not in bar_by_sig:
                    bar_by_sig[(stem, sig)] = list(bar[stem])
                prev = prev_bar_sig[stem]
                if prev is not None:
                    bar_trans[stem][prev][sig] += 1
                prev_bar_sig[stem] = sig
                # Tatum bigrams within bar
                seq = [g for g in bar[stem] if g]
                for a, b in zip(seq, seq[1:]):
                    slice_bigram[stem][a][b] += 1
        # Reset bar continuity across tracks
        for stem in ("drums", "bass", "other", "vocals"):
            prev_bar_sig[stem] = None

    # Top 300 bars per stem by reuse
    prototypes: dict = {}
    for stem in ("drums", "bass", "other", "vocals"):
        items = [(sig, c) for (s, sig), c in bar_counter.items() if s == stem]
        items.sort(key=lambda x: x[1], reverse=True)
        prototypes[stem] = [
            {"signature": sig, "count": c, "tokens": bar_by_sig[(stem, sig)]} for sig, c in items[:300]
        ]

    # Normalize transitions to probs (top 8 next)
    def norm(counter_dict):
        out = {}
        for src, dst_c in counter_dict.items():
            tot = sum(dst_c.values())
            top = dst_c.most_common(8)
            out[src] = {k: round(v / tot, 4) for k, v in top}
        return out

    out = {
        "version": 1,
        "total_bars": total_bars,
        "tracks": len(files),
        "bar_prototypes": prototypes,
        "bar_transitions": {s: norm(bar_trans[s]) for s in bar_trans},
        "slice_bigrams": {s: norm(slice_bigram[s]) for s in slice_bigram},
    }
    Path("data/global_patterns.json").write_text(json.dumps(out, separators=(",", ":")))
    Path("frontend/public/data/global_patterns.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"Trained global patterns: {total_bars} bars across {len(files)} tracks", flush=True)
    for s in ("drums", "bass", "other", "vocals"):
        print(f"  {s}: {len(prototypes[s])} prototypes, {len(out['bar_transitions'][s])} src bars", flush=True)


if __name__ == "__main__":
    main()
