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
    section_seqs: list = []  # per-track List[SectionSpan] for full-arc mining
    try:
        from wub_core.section_arcs import bar_features as _bar_features
        from wub_core.section_arcs import segment_sections as _segment_sections
    except Exception:
        _bar_features = None  # type: ignore[assignment]
        _segment_sections = None  # type: ignore[assignment]
    all_bars_tracks: list = []

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
        bars, _fid = builder.remap_bars_fast(segs, tatums, vocab)
        # Reset usages mutated during remap scoring (keep counts clean for training)
        for stem in ("drums", "bass", "other", "vocals"):
            for vs in vocab.vocab_by_stem.get(stem, []):
                pass
        # Only stems actually present in the recording teach arrangement:
        # mix-only tracks carry no drum/bass/vocal signal, and their empty bars
        # would otherwise dominate prototypes (86% silence) and mute the kit.
        stems_avail = set(raw.get("stems_available") or [])
        use_stem = {s: (not stems_avail or s in stems_avail or s == "other") for s in ("drums", "bass", "other", "vocals")}
        # Joint mining only on full 4-stem bars (real pocket, never silence combos).
        for bar in bars:
            if all(any(bar[s]) for s in ("drums", "bass", "other", "vocals")):
                all_bars_tracks.append({s: list(bar[s]) for s in ("drums", "bass", "other", "vocals")})
        for bar in bars:
            recorded = False
            for stem in ("drums", "bass", "other", "vocals"):
                if not use_stem[stem]:
                    continue
                if not any(bar[stem]):
                    continue  # silence teaches nothing; never plan an empty bar
                recorded = True
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
            if recorded:
                total_bars += 1
        # Reset bar continuity across tracks
        for stem in ("drums", "bass", "other", "vocals"):
            prev_bar_sig[stem] = None

        # Real section detection from the same raw analysis (never breaks
        # training: one bad track is skipped, never the whole corpus).
        if _bar_features is not None and _segment_sections is not None:
            try:
                _feats = _bar_features(segs, tatums)
                _spans = _segment_sections(_feats)
                if _spans:
                    section_seqs.append(_spans)
            except Exception:
                pass

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
    # Additive section-arc model (top 50 full-song arcs + durations).
    # Existing keys above are untouched.
    try:
        from wub_core.section_arcs import mine_arcs as _mine_arcs
        out["section_arcs"] = _mine_arcs(section_seqs).to_dict()
    except Exception as e:
        print(f"  section_arcs skipped: {e}", flush=True)
    # Hierarchical joint stem-cohesion on GLOBAL vocab IDs (keeps existing keys).
    try:
        from wub_core.global_joint_cohesion import mine_call_response, mine_joint_tuples

        joint = mine_joint_tuples(all_bars_tracks, top_n=2000)
        out["joint_tuples"] = [
            {"sig": e["sig"], "count": e["count"], "pmi": e["pmi"]} for e in joint
        ]
        out["call_response"] = mine_call_response(all_bars_tracks, max_next=8)
    except Exception as exc:  # never break legacy training on mining failure
        print(f"  [warn] joint cohesion mining skipped: {exc}", flush=True)
        out.setdefault("joint_tuples", [])
        out.setdefault("call_response", {})
    Path("data/global_patterns.json").write_text(json.dumps(out, separators=(",", ":")))
    Path("frontend/public/data/global_patterns.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"Trained global patterns: {total_bars} bars across {len(files)} tracks", flush=True)
    for s in ("drums", "bass", "other", "vocals"):
        print(f"  {s}: {len(prototypes[s])} prototypes, {len(out['bar_transitions'][s])} src bars", flush=True)


if __name__ == "__main__":
    main()
