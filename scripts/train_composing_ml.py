#!/usr/bin/env python3
"""Unified Composing Machine Learning Trainer for Wub.

Trains the Tier 1 Statistical Groove Model, Tier 2 Phrase Composer Model,
and Tier 3 Song Arranger on authentic dance music track structures, with
acoustic similarity feedback from real song audio wave matching.

Saves learned model weights to data/composing_ml_weights.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from wub_core.composer import COMPOSING_WEIGHTS_FILE, PhraseComposerModel, SongArranger
from wub_core.corpus import CorpusStore
from wub_core.pattern_audio_trainer import PatternAudioFeedbackEngine
from wub_core.patterns import PatternBank


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pattern and song composing ML models.")
    parser.add_argument("--processed", default="data/processed", help="Path to processed JSON analyses")
    parser.add_argument("--stems", default="data/stems", help="Path to stems directory")
    parser.add_argument("--max-tracks", type=int, default=100, help="Max tracks to train on (0 = all)")
    parser.add_argument("--audio-eval", type=int, default=16, help="Generic patterns to optimize against real audio")
    parser.add_argument("--out", default=str(COMPOSING_WEIGHTS_FILE), help="Output weights JSON")
    args = parser.parse_args()

    proc_dir = Path(args.processed)
    stems_dir = Path(args.stems)
    out_file = Path(args.out)

    t0 = time.time()
    print("=== Training Pattern & Song Composing ML Models ===", flush=True)

    store = CorpusStore(proc_dir, stems_base=stems_dir, library_dirs=[Path("pandora/good"), Path("pandora")])
    n_loaded = store.refresh(max_tracks=args.max_tracks if args.max_tracks > 0 else None)
    tracks = store.tracks
    print(f"Loaded {len(tracks)} tracks from corpus ({n_loaded} new).", flush=True)

    if not tracks:
        print("Error: No tracks available to train on.", flush=True)
        sys.exit(1)

    # 1. Populate PatternBank & extract multi-stem patterns and authentic phrases
    print("Extracting multi-stem phrases and sequencer patterns into PatternBank...", flush=True)
    bank = PatternBank()
    n_patterns = bank.load_from_tracks(tracks)
    print(
        f"PatternBank extracted {n_patterns} patterns and {len(bank.phrases)} authentic 8-bar phrases.",
        flush=True,
    )

    # 2. Run real-audio waveform assembly & acoustic similarity scoring
    if args.audio_eval > 0:
        print("Assembling pattern audio waveforms & scoring similarity against real training songs...", flush=True)
        fb_engine = PatternAudioFeedbackEngine()
        fb_res = fb_engine.run_training_step(bank=bank, corpus_store=store, n_optimize=args.audio_eval)
        print(
            f"Acoustic feedback loop evaluated {fb_res.get('evaluated')} patterns | "
            f"Optimized {fb_res.get('optimized_generic_patterns')} generic patterns | "
            f"Mean real-audio match score: {fb_res.get('mean_audio_similarity')}",
            flush=True,
        )

    # 3. Train Tier 2 Phrase Composer Model (weighted by audio similarity)
    print("Training Tier 2 Phrase Composer Markov Model across all 4 stems...", flush=True)
    phrase_model = PhraseComposerModel(weights_path=out_file)
    n_phrase_trans = phrase_model.train_on_tracks(tracks, bank)
    print(f"Learned {n_phrase_trans} bar-to-bar groove transitions across phrase bars.", flush=True)

    # 4. Train Tier 3 Song Arranger (macro-sections & harmonic phrase mixing)
    print("Training Tier 3 Song Arranger on macro song structures & energy curves...", flush=True)
    arranger = SongArranger(bank, weights_path=out_file)
    n_sec_trans = arranger.train_on_corpus(tracks)
    print(f"Learned {n_sec_trans} section transitions and energy targets:", flush=True)
    for sec, energy in sorted(arranger.section_energy_targets.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sec:12s}: target energy {energy:.3f}", flush=True)

    # 5. Persist unified weights
    phrase_model.save(out_file)
    arranger.save(out_file)

    dur = time.time() - t0
    print(f"\nSuccessfully trained composing ML models in {dur:.2f}s!")
    print(f"Model weights saved to {out_file} ({out_file.stat().st_size} bytes).", flush=True)


if __name__ == "__main__":
    main()
