#!/usr/bin/env python3
"""CLI utility to train pattern audio feedback loop on the Wub corpus.

Assembles slices into pattern audio waveforms, scores against authentic audio from
the training songs, and optimizes generic good-sounding patterns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from wub_core.corpus import CorpusStore
from wub_core.pattern_audio_trainer import PatternAudioFeedbackEngine
from wub_core.patterns import PatternBank


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pattern audio waveform feedback loop against real songs.")
    parser.add_argument("--processed", default="data/processed", help="Path to processed JSON analyses")
    parser.add_argument("--stems", default="data/stems", help="Path to stems directory")
    parser.add_argument("--optimize", type=int, default=12, help="Number of generic patterns to optimize")
    parser.add_argument("--max-tracks", type=int, default=40, help="Max tracks to load into pattern bank")
    parser.add_argument("--iterations", type=int, default=3, help="Training iterations")
    args = parser.parse_args()

    proc_dir = Path(args.processed)
    stems_dir = Path(args.stems)

    store = CorpusStore(proc_dir, stems_base=stems_dir, library_dirs=[Path("pandora/good"), Path("pandora")])
    store.refresh()

    tracks = store.tracks
    if not tracks:
        print(f"Error: No analyses found in {proc_dir}")
        sys.exit(1)

    print(f"Loaded {len(tracks)} track analyses from corpus. Initializing PatternBank (max {args.max_tracks} tracks)...", flush=True)
    bank = PatternBank()
    bank.load_from_tracks(tracks[:args.max_tracks])
    print(f"PatternBank populated with {sum(len(p) for p in bank.patterns_by_stem.values())} patterns across stems.", flush=True)

    engine = PatternAudioFeedbackEngine()
    print("Starting Pattern Audio Waveform Feedback Training Loop...", flush=True)

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        res = engine.run_training_step(bank=bank, corpus_store=store, n_optimize=args.optimize)
        dur = time.time() - t0
        print(
            f"[Iteration {it}/{args.iterations}] Evaluated: {res['evaluated']} patterns | "
            f"Optimized: {res['optimized_generic_patterns']} generic patterns | "
            f"Mean real-audio match score: {res['mean_audio_similarity']} | "
            f"Elapsed: {dur:.2f}s",
            flush=True,
        )

    # Print summary of top-scoring grooves
    grooves = engine.feedback_data.get("grooves", {})
    if grooves:
        print("\n=== Learned Groove Acoustic Similarity to Real Songs ===", flush=True)
        for stem, stem_grooves in grooves.items():
            print(f" Stem [{stem}]:", flush=True)
            for gt, score in sorted(stem_grooves.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {gt:20s}: {score:.3f}", flush=True)

    print(f"\nFeedback database updated at {engine.feedback_path} ({len(engine.feedback_data.get('patterns', {}))} patterns recorded).", flush=True)


if __name__ == "__main__":
    main()
