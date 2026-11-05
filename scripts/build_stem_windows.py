"""Build per-stem training windows from the judge's aligned bar index.

The first per-stem dataset used only the shipping phrase cuts (6 per kit), so
``start_bar + 4`` positives were sparse.  This pass uses the mix judge's
aligned bar index (every bar of the 600 judge tracks), keeps every 4th bar so
the windows form real consecutive 4-bar transitions, and cuts the same window
from each source stem.

Each worker loads one track's four stems once and slices every window from
memory; reopening the source WAV per window is I/O-bound and ~8x slower.

Output matches the training script's format:

    data/stem_windows/drums/mel.npy          float16, N x 64 x 128
    data/stem_windows/drums/bar_index.json

Usage (on the laptop, where the source stems and analyses live):
    PYTHONPATH=. python3 scripts/build_stem_windows.py \
      --bar-index aj/bars/bar_index.json \
      --processed processed --stems /mnt/e/Wub/data/stems \
      --out data/stem_windows --every 4 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

STEMS = ("drums", "bass", "other", "vocals")
MEL = {
    "sr": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 64,
    "frames": 128,
}
FIELDS = (
    "phrase_id", "track_id", "start_bar", "bpm", "key", "style", "section",
    "energy", "energy_db", "duration", "is_intro", "is_outro",
)


def window_mel(y: np.ndarray, sr: int, start: float, duration: float) -> np.ndarray | None:
    lo = max(0, int(round(start * sr)))
    hi = min(len(y), int(round((start + duration) * sr)))
    if hi - lo < sr // 4:
        return None
    chunk = y[lo:hi]
    if sr != MEL["sr"]:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=MEL["sr"])
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=MEL["sr"],
        n_fft=MEL["n_fft"],
        hop_length=MEL["hop_length"],
        n_mels=MEL["n_mels"],
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=1.0)
    frames = db.shape[1]
    if frames == 0:
        return None
    edges = np.linspace(0, frames, MEL["frames"] + 1)
    pooled = np.empty((MEL["n_mels"], MEL["frames"]), dtype=np.float32)
    for i in range(MEL["frames"]):
        a = int(edges[i])
        b = max(a + 1, int(edges[i + 1]))
        pooled[:, i] = db[:, a:b].mean(axis=1)
    return pooled.astype(np.float16)


def one_track(job: tuple) -> list[tuple[str, dict, np.ndarray | None]]:
    track_id, windows, stems_root = job
    out: list[tuple[str, dict, np.ndarray | None]] = []
    loaded: dict[str, tuple[np.ndarray, int] | None] = {}
    for stem in STEMS:
        path = Path(stems_root) / track_id / f"{stem}.wav"
        if not path.is_file():
            loaded[stem] = None
            continue
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            y = data.mean(axis=1)
            if int(sr) != MEL["sr"]:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=MEL["sr"])
            loaded[stem] = (y, MEL["sr"])
        except Exception:
            loaded[stem] = None
            continue
    for meta, start, duration in windows:
        for stem in STEMS:
            item = loaded.get(stem)
            if item is None:
                out.append((stem, meta, None))
                continue
            y, sr = item
            out.append((stem, meta, window_mel(y, sr, start, duration)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar-index", required=True)
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--stems", default="/mnt/e/Wub/data/stems")
    ap.add_argument("--out", default="data/stem_windows")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    index = json.loads(Path(args.bar_index).read_text())
    n = len(index["track_id"])
    processed = Path(args.processed)
    tatum_cache: dict[str, list] = {}
    by_track: dict[str, list[tuple[dict, float, float]]] = {}

    for i in range(n):
        start_bar = int(index["start_bar"][i])
        if start_bar % args.every != 0:
            continue
        track_id = str(index["track_id"][i])
        analysis = tatum_cache.get(track_id)
        if analysis is None:
            path = processed / f"{track_id}.json"
            if not path.is_file():
                continue
            analysis = json.loads(path.read_text()).get("tatums") or []
            tatum_cache[track_id] = analysis
        tatum = start_bar * 4
        if tatum >= len(analysis):
            continue
        start = float(analysis[tatum].get("start", 0.0))
        duration = float(index["duration"][i])
        meta = {
            "phrase_id": f"{track_id}_{start_bar}",
            "track_id": track_id,
            "start_bar": start_bar,
            "bpm": float(index["bpm"][i]),
            "key": str(index["key"][i]),
            "style": str(index["style"][i]),
            "section": str(index["section"][i]),
            "energy": float(index["energy"][i]),
            "energy_db": float(index["energy_db"][i]),
            "duration": duration,
            "is_intro": bool(index["is_intro"][i]),
            "is_outro": bool(index["is_outro"][i]),
        }
        by_track.setdefault(track_id, []).append((meta, start, duration))

    jobs = [(tid, windows, args.stems) for tid, windows in by_track.items()]
    print(f"extracting {sum(len(v) for v in by_track.values()) * len(STEMS)} stem windows "
          f"from {len(jobs)} tracks", flush=True)

    columns: dict[str, dict[str, list]] = {stem: {f: [] for f in FIELDS} for stem in STEMS}
    arrays: dict[str, list[np.ndarray]] = {stem: [] for stem in STEMS}
    missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ti, rows in enumerate(ex.map(one_track, jobs, chunksize=1)):
            for stem, meta, vec in rows:
                if vec is None:
                    missing += 1
                    continue
                arrays[stem].append(vec)
                for field in FIELDS:
                    columns[stem][field].append(meta[field])
            if (ti + 1) % 50 == 0:
                print(f"  {ti + 1}/{len(jobs)} tracks (missing {missing})", flush=True)

    out_root = Path(args.out)
    for stem in STEMS:
        if not arrays[stem]:
            print(f"{stem}: no windows, skipping")
            continue
        out = out_root / stem
        out.mkdir(parents=True, exist_ok=True)
        arr = np.stack(arrays[stem]).astype(np.float16)
        np.save(out / "mel.npy", arr)
        index_out = {
            "mel_file": "mel.npy",
            "mel": MEL,
            "n": int(arr.shape[0]),
            **columns[stem],
        }
        (out / "bar_index.json").write_text(json.dumps(index_out, separators=(",", ":")))
        print(f"{stem}: {arr.shape[0]} windows, {arr.nbytes / 1e6:.0f} MB -> {out}", flush=True)
    print(f"done, missing {missing}", flush=True)


if __name__ == "__main__":
    main()
