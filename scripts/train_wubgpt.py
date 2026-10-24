"""Phase 2B CLI: train the multi-stream VQ-Transformer (WubGPT).

Loads TrackAnalysis JSONs from --processed, fits per-stem StemVQs over
markov.build_pools slices, builds the tatum token dataset, trains WubGPT,
and saves a checkpoint (+ sibling VQ dir).

Example:
  python3 scripts/train_wubgpt.py --processed data/processed \\
      --model-out data/wubgpt/wubgpt.pt --epochs 8 --k 256 --batch 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_analyses(paths) -> list:
    """Parse TrackAnalysis JSONs, skipping corrupt/truncated files with a
    warning (killed writers and bad transfers must not abort training)."""
    from wub_core.schema import TrackAnalysis

    out = []
    for p in paths:
        try:
            out.append(TrackAnalysis(**json.loads(p.read_text())))
        except Exception as e:
            print(f"  WARN: skipping corrupt {p.name}: {type(e).__name__}: {str(e)[:80]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train WubGPT (Phase 2B)")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model-out", default="data/wubgpt/wubgpt.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=0,
                    help="use at most N tracks (0 = all)")
    ap.add_argument("--chunk-tracks", type=int, default=200,
                    help="parse N tracks at a time to bound RAM")
    ap.add_argument("--tatum-stride", type=int, default=1,
                    help="keep every Nth tatum row (full corpus on small RAM)")
    ap.add_argument("--vq-sample", type=int, default=150000,
                    help="cap slices per stem for KMeans (0 = all)")
    args = ap.parse_args()

    import gc
    import random

    from wub_core import model_markov as mm
    from wub_core import model_transformer as mt
    from wub_core.schema import STEMS, TrackAnalysis

    proc_dir = Path(args.processed)
    files = sorted(
        p for p in proc_dir.glob("*.json")
        if not p.name.startswith(("_", "meta_"))
    )
    if args.max_tracks and args.max_tracks > 0:
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, min(args.max_tracks, len(files))))
    if not files:
        raise SystemExit(f"No analyses in {proc_dir}. Run scripts/analyze_corpus.py first.")

    # Pass 1 (chunked): accumulate per-stem slices for VQ fitting. The full
    # 2.2 GB corpus must never sit in RAM as parsed objects at once.
    chunk = max(1, args.chunk_tracks)
    stem_slices: dict = {s: [] for s in STEMS}
    n_chunks = (len(files) + chunk - 1) // chunk
    for i in range(0, len(files), chunk):
        analyses = load_analyses(files[i:i + chunk])
        pools = mm.build_pools(analyses)
        for stem in STEMS:
            if stem in pools:
                stem_slices[stem].extend(pools[stem].slices)
        del analyses, pools
        gc.collect()
        print(f"pass1 chunk {i // chunk + 1}/{n_chunks}: slices "
              f"{ {k: len(v) for k, v in stem_slices.items()} }", flush=True)

    vq_rng = random.Random(args.seed)
    vqs: dict = {}
    for stem in STEMS:
        slices = stem_slices.pop(stem)
        if args.vq_sample and args.vq_sample > 0 and len(slices) > args.vq_sample:
            slices = vq_rng.sample(slices, args.vq_sample)
        vq = mt.StemVQ(stem, k=args.k)
        vq.fit(slices)
        vqs[stem] = vq
        print(f"vq[{stem}]: {len(slices)} slices -> {vq.n_clusters} clusters (+1 silence)",
              flush=True)
    del stem_slices
    gc.collect()

    # Pass 2 (chunked): token rows with stride; rows are small plain dicts.
    dataset: list = []
    stride = max(1, args.tatum_stride)
    for i in range(0, len(files), chunk):
        analyses = load_analyses(files[i:i + chunk])
        dataset.extend(mt.build_dataset(analyses, vqs, stride=stride))
        del analyses
        gc.collect()
    print(f"dataset: {len(dataset)} tatum rows (stride={stride})")
    if len(dataset) < 2:
        raise SystemExit("Need at least 2 tatum rows to train.")

    model = mt.WubGPT.from_vqs(vqs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params, vocabs={model.vocabs}")

    losses = mt.train_model(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    for i, loss in enumerate(losses):
        print(f"epoch {i + 1}/{args.epochs} loss={loss:.4f}")
    print(
        f"loss curve: first={losses[0]:.4f} last={losses[-1]:.4f} "
        f"min={min(losses):.4f} epochs={len(losses)}"
    )

    config = model.get_config()
    config.update(
        {
            "k": args.k,
            "lr": args.lr,
            "seed": args.seed,
            "n_rows": len(dataset),
            "n_tracks": len(files),
        }
    )
    out = mt.save_checkpoint(args.model_out, model, vqs, config)
    print(f"wrote {out} (+ sibling VQ dir {out.parent / (out.stem + '_vq')})")


if __name__ == "__main__":
    main()
