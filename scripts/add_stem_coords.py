"""Add per-stem 2D nebula coordinates to the cluster bank.

The old manifest has one shared x/y per phrase (mixed-stem embedding).  The
per-stem remix UI needs four clouds, so this pass computes a compact timbre
descriptor for every phrase/stem, projects each stem's descriptors to two
components with PCA, and writes ``stem_xy``:

    "stem_xy": {"drums": {"x": .., "y": ..}, "bass": {...}, ...}

This is a real per-stem map (log-mel mean/std), not a cosmetic offset.  It can
be replaced later by the per-stem audio judge's 128-d embedding without any
frontend change.

Usage (on the laptop, where the kit audio lives):
    PYTHONPATH=. python3 scripts/add_stem_coords.py \
      --manifest kits_cluster/kit_manifest.json \
      --kits-dir kits_cluster --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from sklearn.decomposition import PCA

STEMS = ("drums", "bass", "other", "vocals")


def describe(path: str) -> np.ndarray | None:
    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
    except Exception:
        return None
    if y.size < sr // 4:
        return None
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # 64 bands x (mean, std, p10, p90) = 256-d timbre descriptor.
    stats = np.concatenate([
        logmel.mean(axis=1),
        logmel.std(axis=1),
        np.percentile(logmel, 10, axis=1),
        np.percentile(logmel, 90, axis=1),
    ]).astype(np.float32)
    return stats


def one(job: tuple) -> tuple[str, str, np.ndarray | None]:
    phrase_id, stem, path = job
    return phrase_id, stem, describe(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="kits_cluster/kit_manifest.json")
    ap.add_argument("--kits-dir", default="kits_cluster")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    kits_dir = Path(args.kits_dir)

    jobs = []
    by_id: dict[str, dict] = {}
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            by_id[phrase["id"]] = phrase
            for stem in STEMS:
                asset = (phrase.get("stems") or {}).get(stem)
                if not isinstance(asset, dict) or not asset.get("file"):
                    continue
                path = kits_dir / asset["file"]
                if path.is_file():
                    jobs.append((phrase["id"], stem, str(path)))

    features: dict[str, dict[str, np.ndarray]] = {stem: {} for stem in STEMS}
    missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for phrase_id, stem, vec in ex.map(one, jobs, chunksize=8):
            if vec is None:
                missing += 1
                continue
            features[stem][phrase_id] = vec
    print(f"described {sum(len(v) for v in features.values())} / {len(jobs)} files, missing {missing}")

    for stem in STEMS:
        ids = list(features[stem].keys())
        if len(ids) < 4:
            print(f"{stem}: too few descriptors ({len(ids)}), skipping")
            continue
        matrix = np.stack([features[stem][pid] for pid in ids])
        pca = PCA(n_components=2, random_state=505, svd_solver="full")
        coords = pca.fit_transform(matrix)
        scaled = np.empty_like(coords)
        for axis in range(2):
            lo, hi = np.percentile(coords[:, axis], [2.0, 98.0])
            if hi <= lo:
                norm = np.full(coords.shape[0], 0.5)
            else:
                norm = np.clip((coords[:, axis] - lo) / (hi - lo), 0.0, 1.0)
            scaled[:, axis] = 0.08 + norm * 0.84
        for pid, (x, y) in zip(ids, scaled):
            phrase = by_id[pid]
            stem_xy = phrase.setdefault("stem_xy", {})
            stem_xy[stem] = {"x": round(float(x), 4), "y": round(float(y), 4)}
        print(f"{stem}: {len(ids)} points, PC1 {pca.explained_variance_ratio_[0]:.3f}, "
              f"PC2 {pca.explained_variance_ratio_[1]:.3f}")

    out = Path(args.out) if args.out else manifest_path
    out.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
