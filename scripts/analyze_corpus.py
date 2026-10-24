"""Phase 1 CLI: incremental, transfer-safe, parallel corpus analyzer.

Designed for a slowly-filling library (e.g. ~/Wub/pandora -> 5000 MP3s):
- Resume manifest (`_manifest.json`): skips unchanged files, retries failures.
- Transfer safety: skips files modified within --min-age-sec (still copying).
- Parallel workers for analysis; indices rebuilt from all JSONs at the end.
- `--watch` loop for continuous ingest while files arrive.
- `--no-separation` (default bulk path): analyze full mix; Demucs separation
  is opt-in per file because 5000 x Demucs on CPU is ~weeks. Run separation
  later on a curated subset via --separate-new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def iter_audio_files(in_dir: Path, recursive: bool) -> list[Path]:
    if not in_dir.exists():
        return []
    if recursive:
        candidates = [p for p in in_dir.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in in_dir.iterdir() if p.is_file()]
    return sorted(p for p in candidates if p.suffix.lower() in AUDIO_EXTS)


def load_manifest(proc_dir: Path) -> dict:
    mp = proc_dir / "_manifest.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            return {}
    return {}


def save_manifest(proc_dir: Path, manifest: dict) -> None:
    proc_dir.mkdir(parents=True, exist_ok=True)
    tmp = proc_dir / "_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(proc_dir / "_manifest.json")


def file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime": st.st_mtime}


def compute_track_id_fast(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def process_one(args_tuple) -> dict:
    """Worker entry: (path_str, proc_dir_str, stems_base_str, no_sep, model, fast_cpu, overwrite)."""
    path_str, proc_dir_str, stems_base_str, no_sep, model, fast_cpu = args_tuple
    # Local imports (worker-safe)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import warnings

    warnings.filterwarnings("ignore")
    from wub_core.analyzer import analyze_track
    from wub_core.schema import TrackAnalysis
    from wub_core.separator import compute_track_id, separate_track

    path = Path(path_str)
    proc_dir = Path(proc_dir_str)
    try:
        track_id = compute_track_id(path)
        out_json = proc_dir / f"{track_id}.json"
        if out_json.exists():
            tr = TrackAnalysis(**json.loads(out_json.read_text()))
            return {"path": path_str, "status": "cached", "track_id": track_id,
                    "bpm": tr.bpm, "tatums": len(tr.tatums), "segments": len(tr.segments)}
        if no_sep:
            tr = analyze_track(path, None, out_json, track_id=track_id, filename=path.name)
        else:
            _, sdir = separate_track(path, Path(stems_base_str),
                                     model=model, fast_cpu=fast_cpu)
            tr = analyze_track(path, sdir, out_json, track_id=track_id, filename=path.name)
        return {"path": path_str, "status": "ok", "track_id": track_id,
                "bpm": tr.bpm, "key": tr.key,
                "tatums": len(tr.tatums), "segments": len(tr.segments)}
    except Exception as e:
        return {"path": path_str, "status": "error", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3)}


def run_pass(in_dir: Path, proc_dir: Path, idx_dir: Path, stems_base: Path,
             no_sep: bool, model: str, fast_cpu: bool, workers: int,
             min_age_sec: float, limit: int, overwrite: bool) -> dict:
    from tqdm import tqdm

    from wub_core.indexer import build_indices
    from wub_core.schema import TrackAnalysis

    manifest = load_manifest(proc_dir)
    now = time.time()
    files = iter_audio_files(in_dir, recursive=True)
    # transfer safety: skip fresh files
    skipped_fresh = 0
    eligible: list[Path] = []
    for p in files:
        try:
            age = now - p.stat().st_mtime
        except OSError:
            continue
        if age < min_age_sec:
            skipped_fresh += 1
            continue
        fp = file_fingerprint(p)
        key = str(p)
        entry = manifest.get(key)
        if not overwrite and entry and entry.get("status") in ("ok", "cached") \
                and entry.get("size") == fp["size"] and entry.get("mtime") == fp["mtime"]:
            continue
        eligible.append(p)
    if limit and limit > 0:
        eligible = eligible[:limit]

    stats = {"found": len(files), "skipped_fresh": skipped_fresh,
             "eligible": len(eligible), "ok": 0, "cached": 0, "error": 0}
    if eligible:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        tasks = [(str(p), str(proc_dir), str(stems_base), no_sep, model, fast_cpu)
                 for p in eligible]
        # Unordered completion: one pathological file (e.g. a 90-min DJ set)
        # must not head-of-line-block the other 1500. Per-file timeout marks
        # stragglers as errors; the pool is cancelled at pass end.
        FILE_TIMEOUT = float(os.environ.get("WUB_FILE_TIMEOUT", "1200"))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # Rolling submission window: only workers*3 futures outstanding,
            # so queue wait stays ~minutes and the submit timestamp honestly
            # approximates execution start for the timeout below. (Submitting
            # all 2000 upfront made every queued task look timed-out.)
            queue = list(tasks)
            pending: dict = {}

            def top_up() -> None:
                while queue and len(pending) < max(16, workers * 3):
                    t = queue.pop(0)
                    pending[ex.submit(process_one, t)] = (t[0], time.time())

            top_up()
            with tqdm(total=len(tasks), desc="Analyzing") as bar:
                while pending or queue:
                    done, _ = wait(list(pending), timeout=30,
                                   return_when=FIRST_COMPLETED)
                    now2 = time.time()
                    for fut in list(done):
                        key, _ = pending.pop(fut)
                        try:
                            res = fut.result(timeout=5)
                        except Exception as e:
                            res = {"path": key, "status": "error",
                                   "error": f"{type(e).__name__}: {e}"}
                        try:
                            fp = file_fingerprint(Path(key))
                        except OSError:
                            fp = {}
                        manifest[key] = {**fp, **{k: v for k, v in res.items()
                                                  if k not in ("trace")},
                                         "updated": time.time()}
                        if res["status"] in ("ok", "cached"):
                            stats["ok" if res["status"] == "ok" else "cached"] += 1
                            if res["status"] == "ok":
                                print(f"  {Path(key).name}: bpm={res.get('bpm', 0):.1f} "
                                      f"tatums={res.get('tatums', 0)} segs={res.get('segments', 0)}")
                        else:
                            stats["error"] += 1
                            print(f"  FAIL {Path(key).name}: {res.get('error')}")
                            if os.environ.get("WUB_DEBUG"):
                                print(res.get("trace"))
                        if (stats["ok"] + stats["error"]) % 25 == 0:
                            save_manifest(proc_dir, manifest)
                        bar.update(1)
                    # Reap stragglers past the per-file budget so one hung
                    # file (or worker) can't stall the pass.
                    for fut, (key, t0) in list(pending.items()):
                        if not fut.done() and now2 - t0 > FILE_TIMEOUT:
                            pending.pop(fut)
                            fut.cancel()
                            manifest[key] = {"status": "error",
                                             "error": f"timeout>{FILE_TIMEOUT:.0f}s (long file? retry solo)",
                                             "updated": time.time()}
                            stats["error"] += 1
                            print(f"  TIMEOUT {Path(key).name} "
                                  f"({(now2 - t0) / 60:.0f}min, budgeted {FILE_TIMEOUT / 60:.0f}min)")
                            bar.update(1)
                    if not done:
                        bar.set_postfix_str(f"{len(pending)} active, {len(queue)} queued")
                    top_up()
        save_manifest(proc_dir, manifest)

    # Rebuild indices from all analyses (fast relative to DSP)
    analyses: list[TrackAnalysis] = []
    for jf in proc_dir.glob("*.json"):
        if jf.name.startswith("_") or jf.name.startswith("meta_"):
            continue
        try:
            analyses.append(TrackAnalysis(**json.loads(jf.read_text())))
        except Exception as e:
            print(f"  WARN: skipping corrupt {jf.name}: {e}")
    if analyses:
        paths = build_indices(analyses, idx_dir)
        print(f"Indices rebuilt from {len(analyses)} tracks: "
              + ", ".join(f"{k}={v.name}" for k, v in paths.items()))
        stats["indexed_tracks"] = len(analyses)
    else:
        print("No analyses yet; indices skipped.")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Wub incremental corpus analyzer")
    ap.add_argument("--input", default="pandora",
                    help='Input folder, e.g. "pandora" or "E:/pandora"')
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--indices", default="data/indices")
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--fast-cpu", action="store_true")
    ap.add_argument("--no-separation", dest="no_sep", action="store_true", default=True,
                    help="analyze full mix (default ON for bulk ingest)")
    ap.add_argument("--separate-new", dest="no_sep", action="store_false",
                    help="run Demucs separation (slow; use on curated subset)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--min-age-sec", type=float, default=60.0,
                    help="skip files modified this recently (still transferring)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--watch", action="store_true",
                    help="loop forever, ingesting new arrivals")
    ap.add_argument("--watch-interval", type=float, default=300.0)
    args = ap.parse_args()

    in_dir = Path(args.input.replace("\\", "/")).expanduser()
    proc_dir = Path(args.processed)
    idx_dir = Path(args.indices)
    stems_base = Path(args.stems)
    no_sep = args.no_sep

    print(f"Wub ingest: input={in_dir} workers={args.workers} "
          f"separation={'ON (slow)' if not no_sep else 'OFF (fast mix analysis)'}")
    while True:
        stats = run_pass(in_dir, proc_dir, idx_dir, stems_base, no_sep,
                         args.model, args.fast_cpu, args.workers,
                         args.min_age_sec, args.limit, args.overwrite)
        print(f"Pass done: {stats}")
        if not args.watch:
            break
        print(f"Watch: sleeping {args.watch_interval}s ...")
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
