"""Per-stem feature normalization + FAISS indices + UMAP coordinates.

Vector per segment (R28):
  x = [ zscore(timbre 12) || chroma 12 || zscore(loud_max 1) || metrical 3 ]
metrical = [bar_pos/8 (clipped 0..1), beat_pos/3, tatum_pos/3]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from wub_core.schema import TrackAnalysis, STEMS

DIM = 28


def metrical_vector(bar: int, beat: int, tatum: int) -> np.ndarray:
    return np.array(
        [min(bar / 8.0, 1.0), beat / 3.0, tatum / 3.0], dtype=np.float32
    )


def segment_to_raw(track: TrackAnalysis, seg_id: str) -> Tuple[np.ndarray, dict]:
    seg = next(s for s in track.segments if s.id == seg_id)
    # find metrical pos: first tatum containing this segment
    bar = beat = tat = 0
    for t in track.tatums:
        if seg_id in t.stem_segments.get(seg.stem, []):
            bar, beat, tat = t.bar_position, t.beat_position, t.tatum_position
            break
    timbre = np.asarray(seg.timbre, dtype=np.float32)
    chroma = np.asarray(seg.pitches, dtype=np.float32)
    loud = np.asarray([seg.loudness_max], dtype=np.float32)
    met = metrical_vector(bar, beat, tat)
    raw = np.concatenate([timbre, chroma, loud, met]).astype(np.float32)
    meta = {
        "track_id": track.track_id,
        "segment_id": seg.id,
        "stem": seg.stem,
        "start": seg.start,
        "duration": seg.duration,
        "bar": bar,
        "beat": beat,
        "tatum": tat,
    }
    return raw, meta


def build_indices(
    analyses: List[TrackAnalysis], out_dir: Path
) -> Dict[str, Path]:
    """Fit per-stem zscore stats, build 4 FAISS indices + unified. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import faiss  # type: ignore

        have_faiss = True
    except ImportError:
        have_faiss = False

    stats: Dict[str, dict] = {}
    paths: Dict[str, Path] = {}
    all_vecs: List[np.ndarray] = []
    all_meta: List[dict] = []

    for stem in STEMS:
        raws: List[np.ndarray] = []
        metas: List[dict] = []
        for tr in analyses:
            for sg in tr.segments:
                if sg.stem != stem or sg.is_silent:
                    continue
                raw, meta = segment_to_raw(tr, sg.id)
                raws.append(raw)
                metas.append(meta)
        if not raws:
            continue
        M = np.stack(raws)  # N x 28
        # zscore timbre dims 0..12 and loud dim 24 using per-stem stats
        mu = np.zeros(DIM, dtype=np.float64)
        sd = np.ones(DIM, dtype=np.float64)
        for cols in (list(range(12)), [24]):
            mu[cols] = M[:, cols].mean(axis=0)
            sd[cols] = M[:, cols].std(axis=0) + 1e-6
        stats[stem] = {"mean": mu.tolist(), "std": sd.tolist()}
        Z = (M - mu) / sd
        # chroma (12..24) + metrical (25..28) pass through (already 0..1)
        Z[:, 12:24] = M[:, 12:24]
        Z[:, 25:28] = M[:, 25:28]
        Z = Z.astype(np.float32)
        all_vecs.append(Z)
        all_meta.extend(metas)

        if have_faiss:
            import faiss

            idx = faiss.IndexFlatL2(DIM)
            idx.add(Z)
            p = out_dir / f"index_{stem}.faiss"
            faiss.write_index(idx, str(p))
            paths[stem] = p
        else:
            p = out_dir / f"index_{stem}.npy"
            np.save(p, Z)
            paths[stem] = p
        (out_dir / f"meta_{stem}.json").write_text(json.dumps(metas, indent=2))

    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    # unified index
    if all_vecs:
        U = np.concatenate(all_vecs, axis=0).astype(np.float32)
        if have_faiss:
            import faiss

            uidx = faiss.IndexFlatL2(DIM)
            uidx.add(U)
            up = out_dir / "index_unified.faiss"
            faiss.write_index(uidx, str(up))
            paths["unified"] = up
        else:
            up = out_dir / "index_unified.npy"
            np.save(up, U)
            paths["unified"] = up
        (out_dir / "meta_unified.json").write_text(json.dumps(all_meta, indent=2))

    # UMAP 2D coords per stem (optional, best-effort)
    try:
        import umap  # type: ignore

        for stem in STEMS:
            mp = out_dir / f"meta_{stem}.json"
            if stem not in paths or not mp.exists():
                continue
            vecs = (
                __import__("faiss").read_index(str(paths[stem])).reconstruct_n(0, 1000000)
                if have_faiss and str(paths[stem]).endswith(".faiss")
                else np.load(paths[stem])
            )
            n = min(len(vecs), 5000)
            emb = umap.UMAP(n_components=2, random_state=0).fit_transform(vecs[:n])
            (out_dir / f"umap_{stem}.json").write_text(
                json.dumps(
                    [
                        {"segment_id": m["segment_id"], "x": float(e[0]), "y": float(e[1])}
                        for m, e in zip(json.loads(mp.read_text())[:n], emb)
                    ]
                )
            )
    except Exception:
        pass
    return paths


def normalize_vector(raw: np.ndarray, stem: str, stats_path: Path) -> np.ndarray:
    stats = json.loads((stats_path).read_text())
    mu = np.asarray(stats[stem]["mean"], dtype=np.float32)
    sd = np.asarray(stats[stem]["std"], dtype=np.float32)
    z = (raw - mu) / sd
    z[12:24] = raw[12:24]
    z[25:28] = raw[25:28]
    return z.astype(np.float32)
