"""Phase 2B CLI: train the multi-stream VQ-Transformer (WubGPT).

Loads TrackAnalysis JSONs from --processed, fits per-stem StemVQs over
markov.build_pools slices, builds the tatum token dataset, trains WubGPT,
and saves a checkpoint (+ sibling VQ dir).

Conditioning (default ON): style (external cluster map), per-step energy
buckets, bar8 phrase position, and bound flags on cross-song transition
rows. Use --weights auto (default) for inverse-sqrt-frequency heads.

Example:
  python3 scripts/train_wubgpt.py --processed data/processed \\
      --model-out data/wubgpt/wubgpt.pt --epochs 8 --k 256 --batch 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

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


def load_style_map(path: str, n_styles: int) -> dict:
    """Load {track_id: style_id} JSON; clamp ids to 1..n_styles (0 = unknown)."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for tid, sid in raw.items():
        try:
            v = int(sid)
        except (TypeError, ValueError):
            continue
        out[str(tid)] = v if 1 <= v <= int(n_styles) else 0
    print(f"style map: {len(out)} tracks labeled (n_styles={n_styles})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train WubGPT (Phase 2B)")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model-out", default="data/wubgpt/wubgpt.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k", type=int, default=512,
                    help="VQ codebook resolution (clusters per stem, e.g. 512 or 1024)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128, help="Transformer embedding dimension")
    ap.add_argument("--n-layers", type=int, default=4, help="Transformer layer count")
    ap.add_argument("--n-heads", type=int, default=4, help="Transformer attention head count")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--multistem-only", action="store_true",
                    help="train exclusively on multi-stem (4-stem) separated tracks")
    ap.add_argument("--max-tracks", type=int, default=0,
                    help="use at most N tracks (0 = all)")
    ap.add_argument("--chunk-tracks", type=int, default=200,
                    help="parse N tracks at a time to bound RAM")
    ap.add_argument("--tatum-stride", type=int, default=1,
                    help="keep every Nth tatum row (full corpus on small RAM)")
    ap.add_argument("--vq-sample", type=int, default=150000,
                    help="cap slices per stem for KMeans (0 = all)")
    ap.add_argument("--style-map", default="",
                    help="JSON {track_id: style_id 1..N}; enables style conditioning")
    ap.add_argument("--n-styles", type=int, default=12,
                    help="style cluster count (style ids 1..N, 0 = unknown)")
    ap.add_argument("--no-cond", action="store_true",
                    help="disable style/energy/bar8/bound conditioning (legacy mode)")
    ap.add_argument("--transition-pairs", type=int, default=0,
                    help="append N cross-song transition row blocks (0 = off)")
    ap.add_argument("--transition-tail", type=int, default=32)
    ap.add_argument("--transition-head", type=int, default=64)
    ap.add_argument("--transition-bound", type=int, default=16)
    ap.add_argument("--weights", default="auto",
                    help="head loss weights: auto (inverse-sqrt freq) or none")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="save an epoch checkpoint every N epochs (0 = final only)")
    ap.add_argument("--T", type=int, default=32,
                    help="context window in steps (sliding windows, stride 1)")
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
    if args.multistem_only:
        stems_dir = Path("data/stems")
        if stems_dir.exists():
            ms_tids = set(p.name for p in stems_dir.iterdir() if p.is_dir() and (p / "drums.wav").exists())
            files = [p for p in files if p.stem in ms_tids]
            print(f"Filtered to {len(files)} multi-stem tracks for focused groove training.")
    if args.max_tracks and args.max_tracks > 0:
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, min(args.max_tracks, len(files))))
    if not files:
        raise SystemExit(f"No analyses in {proc_dir}. Run scripts/analyze_corpus.py first.")

    style_map = load_style_map(args.style_map, args.n_styles) if args.style_map else None
    use_cond = not args.no_cond
    cond_vocabs = None
    if use_cond:
        cond_vocabs = {
            "style": int(args.n_styles) + 1,
            "energy": mt.N_ENERGY,
            "bar8": mt.N_BAR8,
            "bound": mt.N_BOUND,
        }
        print(f"conditioning ON: {cond_vocabs} "
              f"(style_map={'yes' if style_map else 'no, style=0'})")
    else:
        print("conditioning OFF (legacy mode)")

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
        dataset.extend(mt.build_dataset(
            analyses, vqs, stride=stride,
            style_map=style_map, n_styles=args.n_styles,
        ))
        del analyses
        gc.collect()
    print(f"dataset: {len(dataset)} tatum rows (stride={stride})", flush=True)

    # Pass 2.5: cross-song transition rows (streaming: metadata first, then
    # load ONLY the paired tracks — holding all parsed analyses at once OOMs
    # small boxes, which is exactly what killed the first cond-training run).
    trans_rows: list = []
    if args.transition_pairs and args.transition_pairs > 0:
        metas: list = []
        for i in range(0, len(files), chunk):
            for tr in load_analyses(files[i:i + chunk]):
                m = mt.track_meta(tr)
                if m is None:
                    continue
                if style_map:
                    try:
                        m["style"] = int(style_map.get(str(m["id"]), 0))
                    except (TypeError, ValueError):
                        pass
                metas.append(m)
            gc.collect()
        print(f"transition metas: {len(metas)} tracks", flush=True)
        pairs = mt.select_transition_pairs_from_metas(
            metas, n_pairs=args.transition_pairs, seed=args.seed,
        )
        need = {a for p in pairs for a in p}
        by_id: dict = {}
        for i in range(0, len(files), chunk):
            for tr in load_analyses(files[i:i + chunk]):
                if tr.track_id in need:
                    by_id[tr.track_id] = tr
            gc.collect()
        trans_rows = mt.build_transition_rows(
            by_id, vqs, pairs,
            tail=args.transition_tail, head=args.transition_head,
            bound_width=args.transition_bound,
            style_map=style_map, n_styles=args.n_styles,
        )
        dataset.extend(trans_rows)
        print(f"transitions: {len(pairs)} pairs -> {len(trans_rows)} rows "
              f"(total {len(dataset)})", flush=True)
        del by_id, trans_rows, metas, pairs, need
        gc.collect()
    if len(dataset) < 2:
        raise SystemExit("Need at least 2 tatum rows to train.")

    model = mt.WubGPT.from_vqs(
        vqs,
        cond_vocabs=cond_vocabs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff=args.d_model * 4,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params, vocabs={model.vocabs}", flush=True)

    weights = "auto" if args.weights == "auto" else None
    n_rows = len(dataset)
    base_config = model.get_config()
    base_config.update(
        {
            "k": args.k,
            "lr": args.lr,
            "seed": args.seed,
            "n_rows": n_rows,
            "n_tracks": len(files),
            "n_styles": args.n_styles,
            "style_map": args.style_map,
            "transition_pairs": args.transition_pairs,
            "weights": args.weights,
            "T": args.T,
        }
    )

    def _epoch_ckpt(ep, _model, _losses):
        if int(args.checkpoint_every) > 0 and ep % int(args.checkpoint_every) == 0:
            p = Path(args.model_out)
            ep_path = p.parent / f"{p.stem}_ep{ep}{p.suffix}"
            mt.save_checkpoint(ep_path, _model, vqs, dict(base_config, epoch=ep, losses=list(_losses)))
            print(f"checkpoint: {ep_path} (epoch {ep})", flush=True)

    losses = mt.train_model(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        weights=weights,
        T=args.T,
        checkpoint_cb=_epoch_ckpt,
    )
    del dataset  # free ~1GB of row dicts; tensors already inside train_model
    gc.collect()
    for i, loss in enumerate(losses):
        print(f"epoch {i + 1}/{args.epochs} loss={loss:.4f}", flush=True)
    print(
        f"loss curve: first={losses[0]:.4f} last={losses[-1]:.4f} "
        f"min={min(losses):.4f} epochs={len(losses)}",
        flush=True,
    )

    config = dict(base_config, losses=list(losses))
    out = mt.save_checkpoint(args.model_out, model, vqs, config)
    print(f"wrote {out} (+ sibling VQ dir {out.parent / (out.stem + '_vq')})")


if __name__ == "__main__":
    main()
