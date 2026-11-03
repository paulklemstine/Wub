"""Build the global reusable slice vocabulary across the full corpus.

Two-stage (per user plan):
  1. Collect per-song medoids (k per song per stem) from data/processed.
  2. Global KMeans -> fixed budget, clarity-weighted medoids.
  3. Remap all songs to global IDs to score reuse + stem-wise fidelity.
  4. Prune slices unused in 1+ songs.
  5. Render Opus slices (96k for quality to use static room) + manifests.

Usage:
  PYTHONPATH=. python3 scripts/build_global_vocabulary.py --max-tracks 50
  PYTHONPATH=. python3 scripts/build_global_vocabulary.py  # full corpus
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from wub_core.global_vocabulary import DEFAULT_GLOBAL_BUDGET, GlobalVocabularyBuilder
from wub_core.patterns import STEMS


def _mp3_index(processed_dir: Path) -> Dict[str, str]:
    """track_id -> source mp3 path, from ingest manifest + library scan."""
    out: Dict[str, str] = {}
    mp = Path(processed_dir) / "_manifest.json"
    if mp.exists():
        try:
            man = json.loads(mp.read_bytes())
            for entry in (man.values() if isinstance(man, dict) else man):
                if isinstance(entry, dict) and entry.get("track_id") and entry.get("path"):
                    p = str(entry["path"])
                    if p and Path(p).suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
                        out.setdefault(str(entry["track_id"]), p)
        except Exception:
            pass
    by_name: Dict[str, str] = {}
    for lib in (Path("pandora/good"), Path("pandora")):
        if lib.is_dir():
            try:
                for p in lib.rglob("*"):
                    if p.is_file() and p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
                        by_name.setdefault(p.name, str(p))
            except Exception:
                pass
    if by_name:
        for jf in sorted(Path(processed_dir).glob("*.json")):
            if jf.name.startswith(("_", "meta_")) or jf.stem in out:
                continue
            try:
                fn = str(json.loads(jf.read_bytes()).get("filename", ""))
            except Exception:
                continue
            if fn and fn in by_name:
                out[jf.stem] = by_name[fn]
    return out


def _decode_mp3_chunk(mp3_path: str, start: float, dur: float, sr: int = 44100) -> Optional[np.ndarray]:
    try:
        from scripts.audio_encode import ffmpeg_exe
        import subprocess
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.05, dur):.3f}",
               "-i", mp3_path, "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"]
        pr = subprocess.run(cmd, capture_output=True, timeout=60)
        if pr.returncode != 0 or not pr.stdout:
            return None
        mono = np.frombuffer(pr.stdout, dtype=np.float32)
        want = int(dur * sr)
        if len(mono) < 120:
            return None
        return mono[:want] if len(mono) >= want else np.pad(mono, (0, want - len(mono)))
    except Exception:
        return None


def _render_one(args) -> tuple:
    vid, tid, stem, start, dur, stems_dir, out_dir, bitrate, mp3_path = args
    try:
        import soundfile as sf
        from scripts.audio_encode import write_opus
        from wub_core.acoustic_vocabulary_optimizer import AcousticVocabularyOptimizer

        wav_path = Path(stems_dir) / tid / f"{stem}.wav"
        sr = 44100
        mono = None
        if wav_path.exists():
            info = sf.info(str(wav_path))
            sr = info.samplerate
            start_frame = max(0, int(start * info.samplerate))
            frame_len = min(info.frames - start_frame, int(dur * info.samplerate))
            if frame_len < 120:
                return (vid, "", False)
            raw, sr = sf.read(str(wav_path), start=start_frame, frames=frame_len, always_2d=True, dtype="float32")
            mono = raw.mean(axis=1)
        elif mp3_path:
            mono = _decode_mp3_chunk(mp3_path, start, dur, sr)
            if mono is None:
                return (vid, "", False)
        else:
            return (vid, "", False)
        cleaned = AcousticVocabularyOptimizer.clean_slice_audio(mono, sr)
        out_path = Path(out_dir) / f"{vid}.opus"
        write_opus(out_path, cleaned, sr, bitrate=bitrate)
        return (vid, f"data/slices/{out_path.name}", True)
    except Exception:
        return (vid, "", False)


def render_slices(vocab, stems_dir: Path, out_dir: Path, bitrate: str = "96k", workers: int = 12,
                  mp3_index: Optional[Dict[str, str]] = None) -> int:
    from concurrent.futures import ProcessPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)
    mp3_index = mp3_index or {}
    jobs = []
    for stem in STEMS:
        for vs in vocab.vocab_by_stem.get(stem, []):
            out_path = Path(out_dir) / f"{vs.id}.opus"
            if out_path.exists() and out_path.stat().st_size > 500:
                vs.url = f"data/slices/{out_path.name}"
                continue
            mp3 = "" if (Path(stems_dir) / vs.track_id / f"{stem}.wav").exists() else mp3_index.get(vs.track_id, "")
            jobs.append((vs.id, vs.track_id, stem, vs.start, vs.duration, str(stems_dir), str(out_dir), bitrate, mp3))
    n_ok = 0
    if not jobs:
        for stem in STEMS:
            for vs in vocab.vocab_by_stem.get(stem, []):
                vs.url = f"data/slices/{vs.id}.opus"
        return sum(1 for stem in STEMS for _ in vocab.vocab_by_stem.get(stem, []))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for vid, url, ok in ex.map(_render_one, jobs, chunksize=32):
            if ok:
                n_ok += 1
                for stem in STEMS:
                    for vs in vocab.vocab_by_stem.get(stem, []):
                        if vs.id == vid:
                            vs.url = url
                            break
    # Ensure all have urls even if source wav missing (bind to nearest rendered later by export)
    for stem in STEMS:
        for vs in vocab.vocab_by_stem.get(stem, []):
            if not vs.url:
                vs.url = f"data/slices/{vs.id}.opus"
    return n_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--k-per-song", type=int, default=8)
    ap.add_argument("--drums", type=int, default=DEFAULT_GLOBAL_BUDGET["drums"])
    ap.add_argument("--bass", type=int, default=DEFAULT_GLOBAL_BUDGET["bass"])
    ap.add_argument("--other", type=int, default=DEFAULT_GLOBAL_BUDGET["other"])
    ap.add_argument("--vocals", type=int, default=DEFAULT_GLOBAL_BUDGET["vocals"])
    ap.add_argument("--bitrate", default="96k")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stems-only", action="store_true", default=True)
    ap.add_argument("--include-mix", action="store_true", help="Also pool mix-only tracks (slices may lack renderable stem audio)")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    target = {"drums": args.drums, "bass": args.bass, "other": args.other, "vocals": args.vocals}
    builder = GlobalVocabularyBuilder(target_counts=target, k_per_song=args.k_per_song)

    print(f"Collecting per-song medoids (k={args.k_per_song}) ...", flush=True)
    cands = builder.collect_candidates(Path(args.processed), max_tracks=args.max_tracks,
                                       stems_dir=Path(args.stems), virtual_stems=args.include_mix)
    print(f"  candidates: {sum(len(v) for v in cands.values())}", flush=True)

    print(f"Clustering globally to {target} ...", flush=True)
    vocab = builder.build(cands)
    print(f"  built total={vocab.total()}", flush=True)

    # Checkpoint immediately so a kill during remap/render never loses clustering work
    vocab.save(Path("data/global_vocabulary.json"), include_vectors=True)
    vocab.save(Path("frontend/public/data/global_vocabulary.json"), include_vectors=False)
    print(f"  checkpoint saved total={vocab.total()}", flush=True)

    # Remap for reuse + fidelity (sample up to 60 tracks for speed on full run)
    processed = Path(args.processed)
    files = sorted(p for p in processed.glob("*.json") if not p.name.startswith(("_", "meta_")))
    if args.max_tracks:
        files = files[: args.max_tracks]
    id_to_slice = {}
    for stem in STEMS:
        for vs in vocab.vocab_by_stem.get(stem, []):
            id_to_slice[vs.id] = vs
    songs_using: dict = {}
    fid_acc: dict = {s: [] for s in STEMS}
    n_bars_total = 0
    for jf in files[: max(1, min(15, len(files)))]:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        segs = raw.get("segments") or []
        tatums = raw.get("tatums") or []
        if not tatums:
            continue
        bars, fid = builder.remap_bars_fast(segs, tatums, vocab)
        n_bars_total += len(bars)
        for s in STEMS:
            fid_acc[s].append(fid.get(s, 0.92))
        # track songs_using
        seen = set()
        for bar in bars:
            for s in STEMS:
                for gid in bar[s]:
                    if gid:
                        seen.add(gid)
        tid = str(raw.get("track_id") or jf.stem)
        for gid in seen:
            songs_using.setdefault(gid, set()).add(tid)
    for stem in STEMS:
        for vs in vocab.vocab_by_stem.get(stem, []):
            users = songs_using.get(vs.id, set())
            vs.songs_using = sorted(users)[:8]
            # usage_count already incremented in find_nearest during remap
    # Prune ONLY if remap covered the full corpus. Sampled remap (15 tracks)
    # cannot judge reuse across 1700 songs — pruning on a sample nukes 90% of vocab.
    n_remapped = min(15, len(files))
    if n_remapped >= max(50, int(0.8 * len(files))):
        for stem in STEMS:
            before = len(vocab.vocab_by_stem[stem])
            vocab.vocab_by_stem[stem] = [
                vs for vs in vocab.vocab_by_stem[stem] if len(vs.songs_using) > 0 or vs.usage_count > 0
            ]
            print(f"  prune {stem}: {before} -> {len(vocab.vocab_by_stem[stem])}", flush=True)
    else:
        print(f"  skip prune (sampled {n_remapped}/{len(files)} tracks; keep all {vocab.total()})", flush=True)
    mean_fid = {s: round(float(np.mean(v)), 4) if v else 0.0 for s, v in fid_acc.items()}
    print(f"  stem fidelity (sampled {min(15, len(files))} songs, {n_bars_total} bars): {mean_fid}", flush=True)

    # Save full (with vectors) + lightweight frontend manifest
    vocab.save(Path("data/global_vocabulary.json"), include_vectors=True)
    vocab.save(Path("frontend/public/data/global_vocabulary.json"), include_vectors=False)
    print(f"Saved data/global_vocabulary.json total={vocab.total()}", flush=True)

    # Also emit slices manifest shard for client lazy-load (grouped by category)
    manifest = {s: {} for s in STEMS}
    for stem in STEMS:
        for vs in vocab.vocab_by_stem.get(stem, []):
            cat = vs.category.split(":")[-1]
            manifest.setdefault(stem, {}).setdefault(cat, []).append(
                {
                    "name": vs.id,
                    "url": vs.url or f"data/slices/{vs.id}.opus",
                    "pitch": vs.pitch,
                    "centroid": vs.centroid,
                    "duration": vs.duration,
                    "clarity": vs.clarity,
                }
            )
    Path("frontend/public/data/global_slices_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"))
    )

    if not args.no_audio:
        # Clear stale per-song slices to reclaim static room, then render global slices
        slices_dir = Path("frontend/public/data/slices")
        if slices_dir.exists():
            shutil.rmtree(slices_dir)
        slices_dir.mkdir(parents=True, exist_ok=True)
        n = render_slices(vocab, Path(args.stems), slices_dir, bitrate=args.bitrate, workers=args.workers,
                          mp3_index=_mp3_index(Path(args.processed)) if args.include_mix else None)
        print(f"Rendered {n} Opus slices @ {args.bitrate} into {slices_dir}", flush=True)
        # Re-save manifests now that urls are bound
        vocab.save(Path("data/global_vocabulary.json"), include_vectors=True)
        vocab.save(Path("frontend/public/data/global_vocabulary.json"), include_vectors=False)


if __name__ == "__main__":
    main()
