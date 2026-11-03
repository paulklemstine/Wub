"""Vectorised stem analysis: same JSON shape, ~10x faster.

The stock analyzer calls librosa chroma_cqt / mfcc / spectral_centroid once
per slice (thousands of tiny FFTs per track). Here the same features are
computed once per stem and averaged over each slice's frame span, which is
numerically almost identical for everything but the shortest slices (those
fall back to the exact per-slice path).

Used to analyse the full E: stem corpus on the laptop without spending ten
hours of CPU.

Usage:
  PYTHONPATH=. python3 scripts/analyze_stem_corpus_vec.py \
      --stems /path/stems --out data/processed --workers 4
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from wub_core.analyzer import (
    MAX_SEGMENT_SEC,
    MIN_SEGMENT_SEC,
    SILENCE_THRESH_DB,
    SR,
    STEM_MAX_SEGMENT,
    STEM_MIN_SEGMENT,
    STEMS,
    Segment,
    TrackAnalysis,
    _peak_db,
    _rms_db,
    beats_to_tatums,
    detect_onsets,
    estimate_key,
    extract_beats,
    nearest_zero_crossing_offset,
)


def _frame_span(n_samples: int, sr: int, hop: int) -> np.ndarray:
    """Center time of each frame librosa would produce for this signal."""
    n_frames = 1 + n_samples // hop
    return (np.arange(n_frames) * hop) / float(sr)


def _slice_features_fast(y_stem: np.ndarray, sr: int, stem: str, track_id: str,
                        duration: float, segs: list[tuple[float, float]],
                        chroma: np.ndarray, chroma_t: np.ndarray,
                        mfcc: np.ndarray, mfcc_t: np.ndarray,
                        centroid: np.ndarray, centroid_t: np.ndarray) -> list[Segment]:
    import librosa

    out: list[Segment] = []
    for i, (s, e) in enumerate(segs):
        d = e - s
        if d <= 0:
            continue
        s_samp = int(round(s * sr))
        e_samp = int(round(e * sr))
        y_seg = y_stem[s_samp:e_samp]
        mask = (chroma_t >= s) & (chroma_t < e)
        if mask.sum() < 2 or len(y_seg) < 1024:
            # Slice too short for frame averaging: exact per-slice path.
            from wub_core.analyzer import segment_features

            feats = segment_features(y_seg, sr, y_stem, s_samp)
        else:
            pitches = chroma[:, mask].mean(axis=1)
            mx = float(np.max(pitches))
            if mx > 1e-9:
                pitches = pitches / mx
            pitches = np.clip(np.asarray(pitches, dtype=float), 0.0, 1.0)
            mmask = (mfcc_t >= s) & (mfcc_t < e)
            timbre = mfcc[:, mmask].mean(axis=1) if mmask.any() else mfcc.mean(axis=1)
            timbre = np.asarray(timbre, dtype=float).reshape(-1)[:12]
            if timbre.shape[0] < 12:
                timbre = np.pad(timbre, (0, 12 - timbre.shape[0]))
            cmask = (centroid_t >= s) & (centroid_t < e)
            spec_cent = float(centroid[cmask].mean()) if cmask.any() else float(centroid.mean())
            head = y_seg[: min(2048, len(y_seg))]
            feats = {
                "timbre": [float(x) for x in timbre],
                "pitches": [float(x) for x in pitches],
                "loudness_start": float(_rms_db(head)),
                "loudness_max": float(_peak_db(y_seg)),
                "loudness_max_time": float(int(np.argmax(np.abs(y_seg))) / sr),
                "spectral_centroid": spec_cent,
                "zero_crossing_start_offset": float(
                    nearest_zero_crossing_offset(y_stem, s_samp, sr)
                ),
                "is_silent": bool(_peak_db(y_seg) < SILENCE_THRESH_DB),
            }
        out.append(
            Segment(
                id=f"seg_{stem}_{i:04d}",
                track_id=track_id,
                stem=stem,  # type: ignore[arg-type]
                start=float(s),
                duration=float(d),
                is_silent=feats["is_silent"],
                loudness_start=feats["loudness_start"],
                loudness_max=feats["loudness_max"],
                loudness_max_time=feats["loudness_max_time"],
                pitches=feats["pitches"],
                timbre=feats["timbre"],
                spectral_centroid=feats["spectral_centroid"],
                zero_crossing_start_offset=feats["zero_crossing_start_offset"],
            )
        )
    return out


def analyze_track_fast(stem_dir: Path, out_json: Path, track_id: str) -> TrackAnalysis:
    import librosa
    import soundfile as sf

    from wub_core.separator import STEM_FILES

    stem_audio: dict[str, np.ndarray] = {}
    stems_available: list[str] = []
    duration = 0.0
    for stem in STEMS:
        p = stem_dir / STEM_FILES[stem]
        if not p.exists():
            continue
        y, sr = sf.read(str(p), always_2d=True)
        if sr != SR:
            y_mono = y.mean(axis=1)
            y_stem = librosa.resample(y_mono, orig_sr=sr, target_sr=SR).astype(np.float32)
        else:
            y_stem = y.mean(axis=1).astype(np.float32)
        stem_audio[stem] = y_stem
        stems_available.append(stem)
        duration = max(duration, len(y_stem) / float(SR))
    if not stem_audio:
        raise RuntimeError(f"no stems in {stem_dir}")

    beat_src = stem_audio.get("drums") if len(stem_audio.get("drums", [])) > SR else stem_audio.get("other")
    if beat_src is None:
        beat_src = next(iter(stem_audio.values()))
    beats, bpm = extract_beats(beat_src, SR)
    if len(beats) < 4 and "other" in stem_audio and beat_src is not stem_audio["other"]:
        beats, bpm = extract_beats(stem_audio["other"], SR)
    raw_tatums = beats_to_tatums(beats, duration)

    # One CQT per stem, reused for the key estimate and the slice features.
    # (The stock analyzer pays for a second CQT pass at hop 4096 just for the
    # key; averaging the hop-1024 chroma over the stem is equivalent to well
    # within the tuning estimate's resolution.)
    per_stem = {}
    for stem, y in stem_audio.items():
        chroma_full = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=1024)
        mfcc_full = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=512)
        centroid_full = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
        per_stem[stem] = (chroma_full, mfcc_full, centroid_full)
    chroma_acc = [per_stem[s][0].mean(axis=1) for s in ("bass", "other", "vocals") if s in per_stem]
    key = estimate_key(np.mean(chroma_acc, axis=0)) if chroma_acc else "C major"

    all_segments: list[Segment] = []
    seg_by_stem: dict[str, list[Segment]] = {}
    for stem in stems_available:
        y_stem = stem_audio[stem]
        onsets = detect_onsets(y_stem, SR, stem=stem, backtrack=True)
        bounds = [0.0] + [float(t) for t in onsets if 0.0 < float(t) < duration] + [duration]
        bounds = sorted(set(bounds))
        min_seg = STEM_MIN_SEGMENT.get(stem, MIN_SEGMENT_SEC)
        max_seg = STEM_MAX_SEGMENT.get(stem, MAX_SEGMENT_SEC)
        merged = [bounds[0]]
        for b in bounds[1:]:
            if b - merged[-1] < min_seg:
                continue
            merged.append(b)
        if merged[-1] < duration - 1e-6:
            merged.append(duration)
        capped = [merged[0]]
        for b in merged[1:]:
            while b - capped[-1] > max_seg:
                capped.append(capped[-1] + max_seg)
            capped.append(b)
        merged = capped
        segs = list(zip(merged[:-1], merged[1:]))
        if not segs:
            continue
        chroma, mfcc_full, centroid = per_stem[stem]
        mfcc = mfcc_full[1:13] if mfcc_full.shape[0] >= 13 else mfcc_full
        n = len(y_stem)
        segs_out = _slice_features_fast(
            y_stem, SR, stem, track_id, duration, segs,
            chroma, _frame_span(n, SR, 1024),
            mfcc, _frame_span(n, SR, 512),
            centroid, _frame_span(n, SR, 512),
        )
        seg_by_stem[stem] = segs_out
        all_segments.extend(segs_out)

    tatums = []
    from wub_core.schema import Tatum

    for idx, (s, d, conf) in enumerate(raw_tatums):
        e = s + d
        mapping = {st: [] for st in STEMS}
        for stem, segs in seg_by_stem.items():
            for sg in segs:
                if sg.start < e and (sg.start + sg.duration) > s:
                    mapping[stem].append(sg.id)
        tatums.append(
            Tatum(
                id=f"tatum_{idx:04d}",
                index=idx,
                bar_position=int(idx // 16),
                beat_position=int((idx % 16) // 4),
                tatum_position=int(idx % 4),
                start=float(s),
                duration=float(d),
                confidence=float(conf),
                stem_segments=mapping,  # type: ignore[arg-type]
            )
        )

    analysis = TrackAnalysis(
        track_id=track_id,
        filename=track_id,
        bpm=float(bpm),
        key=key,
        duration=float(duration),
        sample_rate=SR,
        stems_available=stems_available,  # type: ignore[arg-type]
        tatums=tatums,
        segments=all_segments,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(analysis.model_dump_json(indent=2))
    tmp.replace(out_json)
    return analysis


def one(job):
    stem_dir, out_dir = job
    d = Path(stem_dir)
    out = Path(out_dir) / f"{d.name}.json"
    if out.exists() and out.stat().st_size > 1024:
        return (d.name, "skip", 0.0)
    t0 = time.time()
    try:
        analyze_track_fast(d, out, d.name)
        return (d.name, "ok", time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return (d.name, f"err:{type(e).__name__}:{e}", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default="/mnt/e/Wub/data/stems")
    ap.add_argument("--out", default="data/processed_vec")
    ap.add_argument("--workers", type=int, default=4)
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
