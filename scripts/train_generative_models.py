"""Train next-generation generative music models across the full corpus.

Produces small JSON artifacts for the static client bundle:
- data/generative/joint_bar_transitions.json (joint 4-stem bar tuples)
- data/generative/harmonic_compat.json      (harmonic co-occurrence tables)
- data/generative/transition_templates.json (learned section-boundary operators)
- data/generative/corpus_eval_stats.json    (eval harness reference stats)

Originality-first: the joint sampler refuses to extend exact corpus runs,
so generated mixes are hybrids, never verbatim copies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from wub_core.corpus import CorpusStore
from wub_core import generation_eval, harmonic_compat, joint_bar_model, transition_ops


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train generative music models")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--out", default="data/generative")
    ap.add_argument("--max-tracks", type=int, default=0, help="0 = all tracks")
    ap.add_argument("--max-bars", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    store = CorpusStore(
        Path(args.processed),
        stems_base=Path(args.stems),
        library_dirs=[],
    )
    n = store.refresh(max_tracks=args.max_tracks or None)
    log(f"Loaded {n} tracks in {time.time() - t0:.1f}s")
    tracks = store.tracks
    if not tracks:
        raise SystemExit("No tracks loaded; aborting.")

    # 1. Joint 4-stem bar tuples + transitions
    t0 = time.time()
    seqs = joint_bar_model.mine_bar_tuples(tracks, max_bars_per_song=args.max_bars)
    model = joint_bar_model.train_joint_transitions(seqs)
    compact = joint_bar_model.to_top_compact(model)
    (out_dir / "joint_bar_transitions.json").write_text(joint_bar_model.model_to_json(compact))
    log(f"Joint bar model: {len(seqs)} songs in {time.time() - t0:.1f}s")

    # 2. Harmonic compatibility tables
    t0 = time.time()
    harm = harmonic_compat.build_harmonic_tables(tracks)
    (out_dir / "harmonic_compat.json").write_text(harmonic_compat.dumps(harmonic_compat.to_compact(harm)))
    log(f"Harmonic compat tables in {time.time() - t0:.1f}s")

    # 3. Transition operators
    t0 = time.time()
    trans = transition_ops.build_transition_tables(tracks)
    (out_dir / "transition_templates.json").write_text(transition_ops.dumps(transition_ops.to_compact(trans)))
    log(f"Transition templates in {time.time() - t0:.1f}s")

    # 4. Eval reference stats
    t0 = time.time()
    stats = generation_eval.build_corpus_stats(tracks)
    (out_dir / "corpus_eval_stats.json").write_text(json.dumps(stats, separators=(",", ":"), default=str))
    log(f"Eval stats in {time.time() - t0:.1f}s")

    total = sum(p.stat().st_size for p in out_dir.glob("*.json"))
    log(f"Done. Generative bundle: {total / 1024:.0f} KB in {out_dir}")


if __name__ == "__main__":
    main()
