"""Phase 4 CLI: Neural Latent Stem Slice Synthesis ("WubSynth" Demo).

Uses Meta EnCodec as a continuous latent audio autoencoder to demonstrate
the three Phase 4 generative states per stem:
  - alpha = 0.0: Nearest real recorded slice decoded through vocoder.
  - alpha = 0.5: Latent interpolation / acoustic morph between two distinct slices.
  - alpha = 1.0: Full continuous hallucination sampled from the learned stem prior.

Outputs:
  data/wubsynth_demo/synth_<stem>_000_real.wav
  data/wubsynth_demo/synth_<stem>_050_morph.wav
  data/wubsynth_demo/synth_<stem>_100_hallucination.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wub_core.corpus import CorpusStore
from wub_core.model_markov import _load_slice
from wub_core.synth_decoder import StemLatentSynth


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 WubSynth neural latent slice demo")
    ap.add_argument("--processed", default="data/processed",
                    help="path to processed analyses directory")
    ap.add_argument("--stems", default="data/stems",
                    help="path to separated stems directory")
    ap.add_argument("--input", default="pandora",
                    help="original audio library directory")
    ap.add_argument("--stem", default="bass", choices=["drums", "bass", "vocals", "other"],
                    help="stem type to synthesize")
    ap.add_argument("--n-slices", type=int, default=12,
                    help="number of real stem slices to embed to fit prior")
    ap.add_argument("--duration", type=float, default=1.5,
                    help="target slice duration in seconds")
    ap.add_argument("--device", default="cpu",
                    help="torch device (cpu or cuda)")
    ap.add_argument("--out-dir", default="data/wubsynth_demo",
                    help="output directory for demo WAVs")
    args = ap.parse_args()

    store = CorpusStore(
        processed_dir=Path(args.processed),
        stems_base=Path(args.stems),
        library_dirs=[Path(args.input)],
    )
    store.refresh()
    if not store.analyses:
        raise SystemExit(f"No analyses found in {args.processed}.")

    pools = store.pools()
    pool = pools.get(args.stem)
    if not pool or not pool.slices:
        raise SystemExit(f"No slices available in pool for stem {args.stem!r}.")

    print(f"Loading {args.n_slices} slices for stem {args.stem!r} (pool size: {len(pool.slices)})...")
    tids = {s.track_id for s in pool.slices}
    audio_for, fallback = store.audio_for(tids)

    valid_slices: List[Tuple[str, str, np.ndarray]] = []
    for s in pool.slices:
        if len(valid_slices) >= args.n_slices:
            break
        p = audio_for.get((s.track_id, args.stem), fallback.get(s.track_id))
        if p is None or not Path(p).exists():
            continue
        try:
            dur = min(max(s.duration, 0.4), 2.5)
            y = _load_slice(Path(p), s.start, dur, sr=44100)
            if len(y) > 2048 and np.max(np.abs(y)) > 1e-4:
                valid_slices.append((s.track_id, s.segment_id, y))
        except Exception:
            continue

    if len(valid_slices) < 2:
        raise SystemExit(f"Could not load at least 2 real slices for {args.stem!r}; need audio on disk.")

    print(f"Initializing StemLatentSynth with EnCodec backend (device={args.device})...")
    synth = StemLatentSynth.with_backend("encodec", device=args.device)

    print(f"Embedding {len(valid_slices)} slices into continuous latent space...")
    synth.embed_corpus({args.stem: valid_slices})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = synth.backend.sr

    z_a = synth.embedded[args.stem][0].z
    z_b = synth.embedded[args.stem][min(1, len(synth.embedded[args.stem]) - 1)].z

    # 1. Alpha = 0.0: Nearest real retrieval through vocoder
    print(f"Rendering alpha=0.0 (nearest real retrieval via vocoder)...")
    y_0 = synth.render_blend(args.stem, z_a, z_b, alpha=0.0, duration_sec=args.duration)
    path_0 = out_dir / f"synth_{args.stem}_000_real.wav"
    sf.write(str(path_0), y_0, sr)
    print(f"  -> wrote {path_0} ({len(y_0)} samples, peak={np.max(np.abs(y_0)):.3f})")

    # 2. Alpha = 0.5: Latent interpolation / acoustic morph
    print(f"Rendering alpha=0.5 (latent acoustic morph between 2 slices)...")
    y_50 = synth.render_blend(args.stem, z_a, z_b, alpha=0.5, duration_sec=args.duration)
    path_50 = out_dir / f"synth_{args.stem}_050_morph.wav"
    sf.write(str(path_50), y_50, sr)
    print(f"  -> wrote {path_50} ({len(y_50)} samples, peak={np.max(np.abs(y_50)):.3f})")

    # 3. Alpha = 1.0: Full continuous hallucination from prior
    print(f"Rendering alpha=1.0 (raw hallucination from fitted latent prior)...")
    z_hallucination = synth.sample_prior(args.stem)
    y_100 = synth.render_blend(args.stem, z_hallucination, None, alpha=1.0, duration_sec=args.duration)
    path_100 = out_dir / f"synth_{args.stem}_100_hallucination.wav"
    sf.write(str(path_100), y_100, sr)
    print(f"  -> wrote {path_100} ({len(y_100)} samples, peak={np.max(np.abs(y_100)):.3f})")

    print("\nPhase 4 WubSynth demo complete!")


if __name__ == "__main__":
    main()
