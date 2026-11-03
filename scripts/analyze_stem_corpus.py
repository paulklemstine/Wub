"""Analyze every stem dir on disk into per-track analysis JSONs.

Usage (on the laptop, where the stems live):
  cd /tmp/wubtrain && PYTHONPATH=. python3 scripts/analyze_stem_corpus.py \
      --stems /mnt/e/Wub/data/stems --out data/processed --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def one(job):
    stem_dir, out_dir = job
    from wub_core.analyzer import analyze_track

    d = Path(stem_dir)
    out = Path(out_dir) / f"{d.name}.json"
    if out.exists() and out.stat().st_size > 1024:
        return (d.name, "skip", 0.0)
    t0 = time.time()
    try:
        analyze_track(d / "drums.wav", d, out, track_id=d.name, filename=d.name)
        return (d.name, "ok", time.time() - t0)
    except Exception as e:  # noqa: BLE001 - one bad track must not stop the run
        return (d.name, f"err:{type(e).__name__}:{e}", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default="/mnt/e/Wub/data/stems")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in Path(args.stems).iterdir() if p.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]
    jobs = [(str(d), str(out_dir)) for d in dirs]
    print(f"tracks: {len(jobs)} workers={args.workers}", flush=True)

    done = ok = err = skip = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            tid, status, dt = f.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  {tid}: {status}", flush=True)
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(1e-9, el)
                eta = (len(jobs) - done) / max(1e-9, rate)
                print(f"[{done}/{len(jobs)}] ok={ok} skip={skip} err={err} "
                      f"{rate*60:.1f}/min eta={eta/60:.0f}min last={dt:.1f}s", flush=True)
    print(json.dumps({"done": done, "ok": ok, "skip": skip, "err": err,
                      "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
