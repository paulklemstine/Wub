"""Phase 2A CLI: offline generative mashup renderer (WubBrain-lite).

Reads through CorpusStore, so it always renders from whatever is ingested.

Examples:
  python3 scripts/generate_mashup.py --list-tracks | head -20
  python3 scripts/generate_mashup.py --steps 200 --chaos other=0.85 --out data/mashup.wav
  python3 scripts/generate_mashup.py --steps 400 --force-tracks drums=<idA> \
      --force-tracks bass=<idB> --out data/test_stem_mashup.wav --pitch-lock
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_kv(groups: list) -> dict[str, str]:
    """Flatten repeatable --flag k=v groups (argparse append-of-lists)."""
    out: dict[str, str] = {}
    flat: list[str] = [p for g in groups for p in (g if isinstance(g, list) else [g])]
    for p in flat:
        if "=" not in p:
            raise SystemExit(f"expected stem=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Wub Phase 2A mashup renderer")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--input", default="pandora", help="original audio library for rendering")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--chaos", nargs="*", action="append", default=[],
                    help="repeatable stem=probability pairs, e.g. --chaos other=0.85 --chaos bass=0.6")
    ap.add_argument("--force-tracks", nargs="*", action="append", default=[],
                    help="repeatable stem=track_id pairs pinning a stem to one source track")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/mashup.wav")
    ap.add_argument("--pitch-lock", action="store_true")
    ap.add_argument("--target-bpm", type=float, default=0.0,
                    help="time-stretch all slices to this BPM (0 = native tempo)")
    ap.add_argument("--list-tracks", action="store_true")
    ap.add_argument("--w-timbre", type=float, default=1.0)
    ap.add_argument("--w-pitch", type=float, default=1.5)
    ap.add_argument("--w-harmony", type=float, default=1.0)
    args = ap.parse_args()

    from wub_core import model_markov as mm
    from wub_core.corpus import CorpusStore

    store = CorpusStore(processed_dir=Path(args.processed), stems_base=Path(args.stems),
                        library_dirs=[Path(args.input)])
    n = store.refresh()
    print(f"corpus: {len(store.analyses)} tracks ({n} loaded)")
    if not store.analyses:
        raise SystemExit(f"No analyses in {args.processed}. Run scripts/analyze_corpus.py first.")
    if args.list_tracks:
        for a in store.tracks:
            print(f"{a.track_id}  {a.bpm:6.1f}bpm  {a.key:12s}  "
                  f"{len(a.segments):4d}segs  {a.filename}")
        return

    pools = store.pools()
    print(f"pools: { {k: len(v.slices) for k, v in pools.items()} }")
    chaos = {k: float(v) for k, v in parse_kv(args.chaos).items()}
    force = parse_kv(args.force_tracks) or None
    seq = mm.generate(pools, steps=args.steps, chaos=chaos or None,
                      force_tracks=force, seed=args.seed,
                      w_timbre=args.w_timbre, w_pitch=args.w_pitch,
                      w_harmony=args.w_harmony)
    tids = {r.track_id for refs in seq.values() for r in refs}
    audio_for, fallback = store.audio_for(tids)
    missing = tids - set(fallback) - {t for (t, _s) in audio_for}
    if missing:
        print(f"WARN: {len(missing)} tracks have no resolvable audio; skipped in render")
    out = mm.render_mashup(seq, audio_for, fallback, Path(args.out),
                           pitch_lock=args.pitch_lock,
                           target_bpm=args.target_bpm or None)
    print(f"wrote {out} ({sum(len(v) for v in seq.values())} slices across "
          f"{len(seq)} stem streams, {len(tids)} source tracks)")


if __name__ == "__main__":
    main()
