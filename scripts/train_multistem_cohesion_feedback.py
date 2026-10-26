"""Train and Lock in Multi-Stem Acoustic Feature Feedback Loop for Wub KG-505.

Learns and locks in:
1. Authentic multi-stem audio mix similarity against real 4-stem song bars from the corpus.
2. Low-end separation & sidechain dynamics (kick-to-bass non-collision).
3. Vocal presence clarity & frequency carving (synths dipping -2.5dB to avoid masking vocals).
4. Inter-stem groove compatibility matrices (drums <-> bass, other <-> vocals, drums <-> vocals).
5. Slices & patterns acoustic scoring derived from real song feedback.

Exports locked weights directly to:
- data/composing_ml_weights.json
- frontend/public/data/composing_ml_weights.json
- data/pattern_audio_feedback.json
- frontend/public/data/standalone_patterns.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from wub_core.corpus import CorpusStore
from wub_core.patterns import PatternBank
from wub_core.pattern_audio_trainer import PatternAudioFeedbackEngine


def main() -> None:
    print("=== Training Multi-Stem Acoustic Feature Feedback Loop ===", flush=True)
    t0 = time.time()

    proc_dir = ROOT_DIR / "data" / "processed"
    stems_dir = ROOT_DIR / "data" / "stems"

    print(f"Loading corpus from {proc_dir}...", flush=True)
    store = CorpusStore(proc_dir, stems_base=stems_dir)
    store.refresh(max_tracks=160)

    multi_tracks = [t for t in store.tracks if len(t.stems_available) == 4]
    print(f"Found {len(multi_tracks)} full 4-stem tracks in training corpus.", flush=True)

    print("Extracting multi-stem phrases and patterns into memory bank...", flush=True)
    bank = PatternBank()
    n_pats = bank.load_from_tracks(multi_tracks[:60])
    print(f"Loaded {n_pats} patterns and {len(bank.phrases)} phrases across authentic tracks.", flush=True)

    print("Running multi-stem inter-stem acoustic cohesion training...", flush=True)
    fb_engine = PatternAudioFeedbackEngine()
    cohesion_res = fb_engine.train_inter_stem_cohesion(corpus_store=store, bank=bank, max_tracks=40)
    print(f"✓ Cohesion Training: {cohesion_res['authentic_bars_evaluated']} authentic multi-stem bars evaluated | "
          f"{cohesion_res['pattern_combos_evaluated']} phrase combos evaluated | "
          f"Mean cohesion: {cohesion_res['mean_multistem_cohesion']}", flush=True)

    print("Optimizing generic pattern archetypes against real song mixes...", flush=True)
    train_res = fb_engine.run_training_step(bank=bank, corpus_store=store, n_optimize=4)
    print(f"✓ Feedback Engine: {train_res['evaluated']} patterns evaluated | "
          f"{train_res['optimized_generic_patterns']} generic patterns optimized | "
          f"Mean audio similarity: {train_res['mean_audio_similarity']}", flush=True)

    print("Locking in feedback weights and exporting to client bundle...", flush=True)
    export_res = fb_engine.export_locked_feedback_weights(
        weights_path=ROOT_DIR / "data" / "composing_ml_weights.json",
        public_weights_path=ROOT_DIR / "frontend" / "public" / "data" / "composing_ml_weights.json",
        standalone_patterns_path=ROOT_DIR / "frontend" / "public" / "data" / "standalone_patterns.json",
    )
    print(f"✓ Locked weights exported to: {export_res['exported_to']}", flush=True)

    elapsed = time.time() - t0
    print(f"=== Multi-Stem Feature Feedback Training Complete in {elapsed:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
