"""Batch Demucs stem extraction for curated dance tracks.

Extracts 4 stems (drums, bass, vocals, other) and generates TrackAnalysis
JSONs for tracks in the optimal dance BPM window (default 122-132 BPM).

Usage:
  python3 scripts/batch_separate.py --limit 5
  python3 scripts/batch_separate.py --limit 20 --min-bpm 120 --max-bpm 135
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wub_core.analyzer import analyze_track
from wub_core.separator import separate_track, stems_exist


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Demucs Stem Extractor")
    ap.add_argument("--input", default="pandora", help="source audio directory")
    ap.add_argument("--processed", default="data/processed", help="processed JSON directory")
    ap.add_argument("--stems", default="data/stems", help="stems directory")
    ap.add_argument("--limit", type=int, default=5, help="number of tracks to separate")
    ap.add_argument("--min-bpm", type=float, default=122.0, help="min BPM filter")
    ap.add_argument("--max-bpm", type=float, default=132.0, help="max BPM filter")
    ap.add_argument("--fast-cpu", action="store_true", default=True, help="use fast CPU mode")
    args = ap.parse_args()

    proc_dir = Path(args.processed)
    stems_dir = Path(args.stems)

    manifest_file = proc_dir / "_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

    candidates = []
    for path_str, meta in manifest.items():
        tid = meta.get("track_id")
        bpm = meta.get("bpm", 0)
        p = Path(path_str)
        if not p.is_absolute():
            p = Path.cwd() / p
        if tid and p.exists() and args.min_bpm <= bpm <= args.max_bpm:
            if not stems_exist(stems_dir / tid):
                candidates.append((p, tid, bpm, meta.get("key", "unknown")))

    print(f"Found {len(candidates)} unstemmed tracks in BPM range [{args.min_bpm:.1f}, {args.max_bpm:.1f}]")
    if not candidates:
        print("No candidates to separate.")
        return

    selected = candidates[: args.limit]
    print(f"Starting Demucs separation on {len(selected)} track(s):\n")

    for idx, (path, tid, bpm, key) in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] Separating: {path.name}")
        print(f"       TID: {tid} | BPM: {bpm:.1f} | Key: {key}")
        t0 = time.time()
        try:
            _, sdir = separate_track(path, stems_dir, fast_cpu=args.fast_cpu)
            out_json = proc_dir / f"{tid}.json"
            tr = analyze_track(path, sdir, out_json, track_id=tid, filename=path.name)
            manifest[str(path)] = {
                "size": path.stat().st_size,
                "mtime": path.stat().st_mtime,
                "path": str(path),
                "status": "ok",
                "track_id": tid,
                "bpm": tr.bpm,
                "key": tr.key,
                "tatums": len(tr.tatums),
                "segments": len(tr.segments),
                "stems": list(tr.stems_available),
                "updated": time.time(),
            }
            manifest_file.write_text(json.dumps(manifest, indent=1))
            elapsed = time.time() - t0
            print(f"  --> Done in {elapsed:.1f}s! Segments: {len(tr.segments)}, Tatums: {len(tr.tatums)}\n")
        except Exception as e:
            print(f"  --> Failed: {e}\n")

    print(f"Batch completed. Stems directory now has {len([d for d in stems_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])} stemmed tracks.")


if __name__ == "__main__":
    main()
