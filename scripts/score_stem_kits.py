"""Score the shipping bank with the four per-stem audio judges.

Each judge is a two-tower CNN trained on one stem's 64x128 log-mel bars.  This
pass:

1. encodes every shipping-bank phrase for each stem (128-d),
2. scores every ordered pair inside that stem's shipping pool,
3. keeps the top-K next phrases per phrase/stem for ``taste_scores.json``,
4. optionally replaces the timbre PCA ``stem_xy`` with the judge embedding
   PCA (the map then reflects what the transition model hears).

Output ``taste_scores.json`` has the existing per-kit fields (empty here) plus:

    "stem_next": {
      "<phrase_id>": {
        "drums": [{"id": "<next_phrase_id>", "score": 0.87}, ...],
        ...
      }
    }

Usage (on the laptop, after training the four checkpoints):
    PYTHONPATH=. python3 scripts/score_stem_kits.py \
      --bars data/stem_bars \
      --manifest frontend/public/data/kits/kit_manifest.json \
      --models data/stem_judge \
      --out frontend/public/data/kits/taste_scores.json \
      --write-coords --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

try:  # UMAP gives a legible 2-D cloud; PCA is the deterministic fallback.
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

from scripts.audio_judge_model import AudioJudge

STEMS = ("drums", "bass", "other", "vocals")


def load_bank(manifest_path: Path) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(manifest_path.read_text())
    phrase_ids = set()
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            phrase_ids.add(phrase["id"])
    return manifest, {pid: {} for pid in phrase_ids}


def stem_rows(bars_dir: Path, stem: str, wanted: set[str]) -> tuple[list[str], np.ndarray, dict]:
    idx = json.loads((bars_dir / stem / "bar_index.json").read_text())
    arr = np.load(bars_dir / stem / idx["mel_file"], mmap_mode="r")
    phrase_ids = idx["phrase_id"]
    rows = []
    ids = []
    for i, pid in enumerate(phrase_ids):
        if pid in wanted:
            ids.append(pid)
            rows.append(i)
    if not rows:
        raise SystemExit(f"no shipping phrases found in {bars_dir / stem}")
    return ids, np.asarray(rows, dtype=np.int64), idx


@torch.no_grad()
def encode(model: AudioJudge, arr: np.ndarray, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    out = []
    for start in range(0, len(rows), 256):
        batch = np.asarray(arr[rows[start:start + 256]]).astype(np.float32)
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        out.append(model.encode(x).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def top_next(
    model: AudioJudge,
    embeddings: torch.Tensor,
    ids: list[str],
    topk: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    emb = embeddings.to(device)
    n = emb.shape[0]
    result: dict[str, list[dict]] = {}
    for i in range(n):
        source = emb[i:i + 1].expand(n, -1)
        logits = model.head(torch.cat([source, emb], dim=1)).squeeze(1)
        logits[i] = -1e9
        k = min(topk, n - 1)
        values, indices = torch.topk(logits, k)
        probs = torch.sigmoid(values)
        result[ids[i]] = [
            {"id": ids[int(j)], "score": round(float(score), 4)}
            for j, score in zip(indices.cpu(), probs.cpu())
        ]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/stem_bars")
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--models", default="data/stem_judge")
    ap.add_argument("--out", default="frontend/public/data/kits/taste_scores.json")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write-coords", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    manifest_path = Path(args.manifest)
    manifest, phrase_map = load_bank(manifest_path)
    wanted = set(phrase_map)

    stem_next: dict[str, dict[str, list[dict]]] = {pid: {} for pid in wanted}
    aucs = []
    coords: dict[str, dict[str, dict[str, float]]] = {pid: {} for pid in wanted}

    for stem in STEMS:
        ckpt_path = Path(f"{args.models}_{stem}.pt")
        if not ckpt_path.is_file():
            raise SystemExit(f"missing checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = AudioJudge(emb_dim=int(ckpt["cfg"].get("emb_dim", 128)))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        aucs.append(float(ckpt.get("val_auc", 0.0)))

        ids, rows, _ = stem_rows(Path(args.bars), stem, wanted)
        embeddings = encode(model, np.load(Path(args.bars) / stem / "mel.npy", mmap_mode="r"), rows, device)
        print(f"{stem}: {len(ids)} phrases, emb {tuple(embeddings.shape)}, "
              f"val_auc {ckpt.get('val_auc', 0.0):.4f}", flush=True)

        nexts = top_next(model, embeddings, ids, args.topk, device)
        for pid, items in nexts.items():
            stem_next[pid][stem] = items

        if args.write_coords:
            if umap is not None:
                reducer = umap.UMAP(
                    n_components=2, random_state=505, n_neighbors=20,
                    min_dist=0.12, metric="euclidean", verbose=False,
                )
                xy = reducer.fit_transform(embeddings.numpy())
                method = "UMAP"
            else:
                pca = PCA(n_components=2, random_state=505, svd_solver="full")
                xy = pca.fit_transform(embeddings.numpy())
                method = "PCA"
            scaled = np.empty_like(xy)
            for axis in range(2):
                lo, hi = np.percentile(xy[:, axis], [2.0, 98.0])
                if hi <= lo:
                    norm = np.full(xy.shape[0], 0.5)
                else:
                    norm = np.clip((xy[:, axis] - lo) / (hi - lo), 0.0, 1.0)
                scaled[:, axis] = 0.08 + norm * 0.84
            for pid, (x, y) in zip(ids, scaled):
                coords[pid][stem] = {"x": round(float(x), 4), "y": round(float(y), 4)}
            print(f"{stem}: {method} map {scaled[:, 0].min():.2f}..{scaled[:, 0].max():.2f} / "
                  f"{scaled[:, 1].min():.2f}..{scaled[:, 1].max():.2f}", flush=True)

    out = {
        "version": 2,
        "model": "stem_audio_judge_v1",
        "val_auc": round(float(np.mean(aucs)), 6),
        "note": "four per-stem two-tower CNNs; scores are sigmoid probabilities",
        "kits": {},
        "stem_next": {pid: by_stem for pid, by_stem in stem_next.items() if by_stem},
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.write_coords:
        for kit in manifest.get("kits", []):
            for phrase in kit.get("phrases", []):
                phrase["stem_xy"] = coords.get(phrase["id"], {})
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
        print(f"updated stem_xy in {manifest_path} ({manifest_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
