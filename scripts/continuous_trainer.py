"""Continuous GPU Music Model Training Daemon for Wub.

Utilizes the GPU (CUDA) to continuously:
1. Separate dance music tracks from pandora/good and pandora/ into 4 high-fidelity stems (drums, bass, vocals, other).
2. Analyze tatums, segments, harmonic key, and 16-step patterns.
3. Mine and update statistical groove distributions across all stems.
4. Train WubGPT multi-stream transformer on newly acquired multi-stem data.
5. Keep the jukebox continuously learning, tightening its pocket, and expanding its repertoire.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from wub_core.analyzer import analyze_track
from wub_core.separator import separate_track, stems_exist


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [ContinuousTrainer] {msg}"
    print(line, flush=True)
    try:
        log_file = Path("data/training_log.txt")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def find_unstemmed_candidates(
    library_dirs: list[Path],
    manifest_path: Path,
    stems_dir: Path,
    min_bpm: float = 116.0,
    max_bpm: float = 140.0,
) -> list[tuple[Path, str, float, str]]:
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    candidates = []
    from wub_core.separator import compute_track_id

    # Prioritize pandora/good first, then general pandora
    for lib_dir in library_dirs:
        if not lib_dir.exists():
            continue
        for p in sorted(lib_dir.rglob("*.mp3")):
            # Check relative and absolute paths in manifest
            rel_str = str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p)
            meta = manifest.get(str(p)) or manifest.get(rel_str) or {}
            tid = meta.get("track_id")
            if not tid:
                tid = compute_track_id(p)
            bpm = meta.get("bpm", 126.0)
            key = meta.get("key", "unknown")
            sdir = stems_dir / tid
            if not stems_exist(sdir) and (min_bpm <= bpm <= max_bpm or bpm <= 0):
                candidates.append((p, tid, bpm, key))

    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous GPU Training Worker")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-bpm", type=float, default=116.0)
    ap.add_argument("--max-bpm", type=float, default=140.0)
    ap.add_argument("--max-tracks", type=int, default=100)
    ap.add_argument("--train-interval", type=int, default=5, help="train WubGPT after every N separated tracks")
    args = ap.parse_args()

    stems_dir = Path("data/stems")
    proc_dir = Path("data/processed")
    manifest_file = proc_dir / "_manifest.json"
    stems_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log(f"Continuous GPU Trainer active on {args.device.upper()} ({gpu_name})")

    library_dirs = [Path("pandora/good"), Path("pandora")]
    separated_this_session = 0

    while True:
        candidates = find_unstemmed_candidates(
            library_dirs=library_dirs,
            manifest_path=manifest_file,
            stems_dir=stems_dir,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
        )

        if not candidates:
            log("All candidate tracks in dance BPM range are already stemmed. Sleeping 60s...")
            time.sleep(60)
            continue

        log(f"Queue has {len(candidates)} unstemmed tracks. Starting next extraction...")
        p, tid, bpm, key = candidates[0]
        log(f"Separating [{tid}]: {p.name} | BPM: {bpm:.1f} | Key: {key}")

        t0 = time.time()
        try:
            _, sdir = separate_track(p, stems_dir, device=args.device, fast_cpu=False)
            out_json = proc_dir / f"{tid}.json"
            tr = analyze_track(p, sdir, out_json, track_id=tid, filename=p.name)

            # Update manifest
            manifest = {}
            if manifest_file.exists():
                try:
                    manifest = json.loads(manifest_file.read_text())
                except Exception:
                    pass
            manifest[str(p)] = {
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "path": str(p),
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
            separated_this_session += 1
            log(f"Successfully processed {p.name} in {elapsed:.1f}s (Total stemmed: {len([d for d in stems_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])})")

            # Pattern Audio Feedback Training Step:
            # Assembles slices of candidate patterns into entire audio waves,
            # scores acoustic similarity against real training songs, and optimizes generic patterns.
            try:
                from wub_core.corpus import CorpusStore
                from wub_core.patterns import PatternBank
                from wub_core.pattern_audio_trainer import PatternAudioFeedbackEngine

                c_store = CorpusStore(proc_dir, stems_base=stems_dir, library_dirs=library_dirs)
                c_store.refresh()
                p_bank = PatternBank()
                p_bank.load_from_tracks(c_store.tracks[:35])

                fb_engine = PatternAudioFeedbackEngine()
                fb_res = fb_engine.run_training_step(bank=p_bank, corpus_store=c_store, n_optimize=4)
                log(f"[PatternAudioFeedback] Evaluated {fb_res.get('evaluated')} patterns | Optimized {fb_res.get('optimized_generic_patterns')} generic patterns | Mean real-audio similarity: {fb_res.get('mean_audio_similarity')} (DB total: {fb_res.get('feedback_db_entries')})")
            except Exception as e:
                log(f"Pattern audio feedback training note: {e}")

            # Periodically train WubGPT on the newly expanded multi-stem dataset
            if separated_this_session % args.train_interval == 0:
                log(f"Triggering periodic GPU WubGPT model refinement ({separated_this_session} tracks added)...")
                cmd = [
                    sys.executable,
                    "scripts/train_wubgpt.py",
                    "--processed", str(proc_dir),
                    "--model-out", "data/wubgpt/wubgpt.pt",
                    "--device", args.device,
                    "--epochs", "4",
                    "--multistem-only",
                    "--batch", "32",
                ]
                subprocess.run(cmd, check=False)
                log("WubGPT checkpoint refined.")

                # Rebuild UMAP projections
                log("Updating FAISS indices and UMAP...")
                subprocess.run([sys.executable, "scripts/build_umap.py"], check=False)
                log("FAISS & UMAP updated.")

        except Exception as e:
            log(f"Error processing {p.name}: {e}")
            time.sleep(5)

        if 0 < args.max_tracks <= separated_this_session:
            log(f"Reached max tracks limit ({args.max_tracks}). Continuous trainer pausing.")
            break

        time.sleep(2)


if __name__ == "__main__":
    main()
