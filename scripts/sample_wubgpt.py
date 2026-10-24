"""Phase 2B CLI: WubGPT generative concatenative assembly.

Loads a trained WubGPT checkpoint + VQ codebooks, autoregressively samples
a token sequence, maps tokens back to nearest real audio slices across the
corpus, and renders a pitch-locked, time-stretched 4-stem mashup.

Examples:
  python3 scripts/sample_wubgpt.py --checkpoint data/wubgpt/wubgpt.pt --steps 256 --out data/mashup_wubgpt.wav
  python3 scripts/sample_wubgpt.py --checkpoint data/wubgpt/wubgpt.pt --steps 128 --target-bpm 128 --pitch-lock
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wub_core.corpus import CorpusStore
from wub_core.model_markov import render_mashup
from wub_core.model_transformer import (
    STREAM_ORDER,
    build_dataset,
    load_checkpoint,
    sample,
    tokens_to_slices,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wub Phase 2B WubGPT sampler and mashup renderer")
    ap.add_argument("--checkpoint", default="data/wubgpt/wubgpt.pt",
                    help="path to trained WubGPT checkpoint")
    ap.add_argument("--processed", default="data/processed",
                    help="path to processed analysis JSONs")
    ap.add_argument("--stems", default="data/stems",
                    help="path to demucs stems directory")
    ap.add_argument("--input", default="pandora",
                    help="original audio library directory for rendering")
    ap.add_argument("--steps", type=int, default=256,
                    help="number of tatum steps to autoregressively sample")
    ap.add_argument("--temperature", type=float, default=0.75,
                    help="sampling temperature (>0.0; 0.75 locks rhythm while staying expressive)")
    ap.add_argument("--top-k", type=int, default=32,
                    help="top-k filtering per stem head")
    ap.add_argument("--top-p", type=float, default=0.85,
                    help="nucleus filtering top-p (0.85 removes long-tail noise)")
    ap.add_argument("--target-bpm", type=float, default=128.0,
                    help="target BPM to lock and time-stretch slices to (0 = no stretch)")
    ap.add_argument("--pitch-lock", action="store_true", default=True,
                    help="lock harmonic stems to master key")
    ap.add_argument("--no-pitch-lock", dest="pitch_lock", action="store_false",
                    help="disable harmonic pitch-lock")
    ap.add_argument("--primer-track", default=None,
                    help="optional track_id to use as prompt / primer")
    ap.add_argument("--device", default="auto",
                    help="computation device: cpu, cuda, or auto")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed")
    ap.add_argument("--out", default="data/mashup_wubgpt.wav",
                    help="output audio WAV path")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found at {ckpt_path}. Ensure model has finished training.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {ckpt_path} on {device}...")
    model, vqs, config = load_checkpoint(ckpt_path)
    model.to(torch.device(device))
    model.eval()

    store = CorpusStore(
        processed_dir=Path(args.processed),
        stems_base=Path(args.stems),
        library_dirs=[Path(args.input)],
    )
    n = store.refresh()
    print(f"Corpus: {len(store.analyses)} tracks ({n} newly loaded)")
    if not store.analyses:
        raise SystemExit(f"No analyses found in {args.processed}. Run scripts/analyze_corpus.py first.")

    pools = store.pools()
    print(f"Stem pools: { {k: len(v.slices) for k, v in pools.items()} }")

    # Construct prompt seed steps
    seed_steps: List[Dict[str, int]] = []
    primer = None
    if args.primer_track:
        primer = store.get(args.primer_track)
        if not primer:
            print(f"WARN: primer track {args.primer_track} not found in corpus; choosing first track")
    if primer is None and store.analyses:
        primer = store.analyses[args.seed % len(store.analyses)]

    if primer:
        seed_rows = build_dataset([primer], vqs, stride=1)
        if seed_rows:
            # Take up to 16 tatums (1 bar) as prompt
            seed_steps = seed_rows[: min(16, len(seed_rows))]

    if not seed_steps:
        # Fallback to silence tokens across 16 rhythmic subdivisions
        sil_row = {"r": 0}
        for st in STREAM_ORDER[1:]:
            sil_row[st] = vqs[st].silence_id if st in vqs else 0
        seed_steps = [dict(sil_row, r=i % 16) for i in range(16)]

    print(f"Sampling {args.steps} tatums with temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}...")
    sampled_rows = sample(
        model=model,
        vqs=vqs,
        seed_steps=seed_steps,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(f"Generated {len(sampled_rows)} token steps.")

    print("Mapping tokens to nearest audio slices in pool...")
    seq = tokens_to_slices(sampled_rows, pools, vqs)

    tids = {r.track_id for refs in seq.values() for r in refs if r is not None}
    print(f"Mapped to slices across {len(tids)} source tracks.")

    audio_for, fallback = store.audio_for(tids)
    missing = tids - set(fallback) - {t for (t, _s) in audio_for}
    if missing:
        print(f"WARN: {len(missing)} tracks have no resolvable audio; skipped in render")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering mashup to {out_path} (target_bpm={args.target_bpm}, pitch_lock={args.pitch_lock})...")
    out = render_mashup(
        seq=seq,
        audio_for=audio_for,
        fallback_audio=fallback,
        out_path=out_path,
        pitch_lock=args.pitch_lock,
        target_bpm=args.target_bpm if args.target_bpm > 0 else None,
    )
    total_slices = sum(len([r for r in refs if r is not None]) for refs in seq.values())
    print(f"Wrote {out} ({total_slices} active slices across {len(seq)} stem streams)")


if __name__ == "__main__":
    main()
