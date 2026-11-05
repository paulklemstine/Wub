"""Build per-stem 4-bar log-mel datasets for the transition judge.

The mix-level judge proved the two-tower CNN can learn real bar-to-bar
transitions.  This pass produces the same 64x128 log-mel patches for each stem
separately, so four independent judges can be trained (drums, bass, other,
vocals) and each stem's embedding can drive its own nebula cloud.

Output layout (one dataset per stem):

    data/stem_bars/drums/mel.npy          float16, N x 64 x 128
    data/stem_bars/drums/bar_index.json   metadata + mel config

The format matches the mix judge's ``bar_index.json`` so the existing training
script can be pointed at each stem directory unchanged.

Usage (on the laptop, where kits_all2 and the stems live):
    PYTHONPATH=. python3 scripts/build_stem_bars.py \
      --manifest kits_all2/kit_manifest.json --kits-dir kits_all2 \
      --out data/stem_bars --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

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


def logmel(path: str) -> np.ndarray | None:
    try:
        y, sr = librosa.load(path, sr=MEL["sr"], mono=True)
    except Exception:
        return None
    if y.size < MEL["sr"] // 4:
        return None
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=MEL["n_fft"],
        hop_length=MEL["hop_length"],
        n_mels=MEL["n_mels"],
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=1.0)
    # Average-pool the time axis to exactly 128 frames (same as the mix judge).
    frames = db.shape[1]
    if frames == 0:
        return None
    edges = np.linspace(0, frames, MEL["frames"] + 1)
    pooled = np.empty((MEL["n_mels"], MEL["frames"]), dtype=np.float32)
    for i in range(MEL["frames"]):
        lo = int(edges[i])
        hi = max(lo + 1, int(edges[i + 1]))
        pooled[:, i] = db[:, lo:hi].mean(axis=1)
    return pooled.astype(np.float16)


def one(job: tuple) -> tuple[str, dict | None, np.ndarray | None, str]:
    stem, meta, path = job
    vec = logmel(path)
    if vec is None:
        return stem, None, None, path
    return stem, meta, vec, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="kits_all2/kit_manifest.json")
    ap.add_argument("--kits-dir", default="kits_all2")
    ap.add_argument("--out", default="data/stem_bars")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    kits_dir = Path(args.kits_dir)
    jobs: list[tuple] = []
    for kit in manifest.get("kits", []):
        track_id = kit["track_id"]
        for phrase in kit.get("phrases", []):
            for stem in STEMS:
                asset = (phrase.get("stems") or {}).get(stem)
                if not isinstance(asset, dict) or not asset.get("file"):
                    continue
                path = kits_dir / asset["file"]
                if not path.is_file():
                    continue
                meta = {
                    "phrase_id": phrase["id"],
                    "track_id": track_id,
                    "start_bar": int(phrase.get("start_bar", 0)),
                    "bpm": float(kit.get("bpm", 0.0)),
                    "key": str(kit.get("key", "")),
                    "style": str(kit.get("style", "")),
                    "section": str(phrase.get("section", "")),
                    "energy": float(phrase.get("energy", 0.0)),
                    "energy_db": float(phrase.get("energy_db", 0.0)),
                    "duration": float(phrase.get("duration_sec", 0.0)),
                    "is_intro": bool(phrase.get("is_intro", False)),
                    "is_outro": bool(phrase.get("is_outro", False)),
                }
                jobs.append((stem, meta, str(path)))

    print(f"extracting {len(jobs)} stem bars", flush=True)
    columns: dict[str, dict[str, list]] = {
        stem: {field: [] for field in FIELDS} for stem in STEMS
    }
    arrays: dict[str, list[np.ndarray]] = {stem: [] for stem in STEMS}
    missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (stem, meta, vec, err) in enumerate(ex.map(one, jobs, chunksize=8)):
            if vec is None:
                missing += 1
                continue
            arrays[stem].append(vec)
            for field in FIELDS:
                columns[stem][field].append(meta[field])
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)} (missing {missing})", flush=True)

    out_root = Path(args.out)
    for stem in STEMS:
        if not arrays[stem]:
            print(f"{stem}: no bars, skipping")
            continue
        out = out_root / stem
        out.mkdir(parents=True, exist_ok=True)
        arr = np.stack(arrays[stem]).astype(np.float16)
        np.save(out / "mel.npy", arr)
        index = {
            "mel_file": "mel.npy",
            "mel": MEL,
            "n": int(arr.shape[0]),
            **columns[stem],
        }
        (out / "bar_index.json").write_text(json.dumps(index, separators=(",", ":")))
        print(f"{stem}: {arr.shape[0]} bars, {arr.nbytes / 1e6:.0f} MB -> {out}", flush=True)
    print(f"done, missing {missing}", flush=True)


if __name__ == "__main__":
    main()
