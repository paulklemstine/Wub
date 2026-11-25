"""Fit per-stem VQ codebooks over the global slice vocabulary for tatum modeling.

Slices are too sparse (15k ids) for tatum transitions; codebooks compress
each stem to K acoustic codes. Tatum steps then predict codes (dense), and
each code maps back to one medoid slice id for rendering.

K: drums 384, bass 256, other 384, vocals 256 (+SIL=HOLD ids follow codes).
Inputs are the 26-D slice_feature_raw vectors stored on GlobalVocabSlice.
Outputs data/tatum_vq/{stem}/ (StemVQ scaler+centers) and
frontend/public/data/tatum_vq.json (medoid slice per code for the renderer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "other", "vocals")
K_PER_STEM = {"drums": 384, "bass": 256, "other": 384, "vocals": 256}


def _fake_slice(feat) -> dict:
    f = [float(v) for v in feat]
    return {
        "timbre": f[0:12],
        "pitches": f[12:24],
        "loudness_max": f[24],
        "duration": float(np.expm1(max(0.0, f[25]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--out", default="data/tatum_vq")
    ap.add_argument("--drums", type=int, default=K_PER_STEM["drums"])
    ap.add_argument("--bass", type=int, default=K_PER_STEM["bass"])
    ap.add_argument("--other", type=int, default=K_PER_STEM["other"])
    ap.add_argument("--vocals", type=int, default=K_PER_STEM["vocals"])
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wub_core.model_transformer import StemVQ, slice_feature_raw

    gvoc = json.loads(Path(args.vocab).read_bytes())
    manifest = {"version": 1, "stems": {}}
    for stem in STEMS:
        vs = gvoc["vocab_by_stem"][stem]
        feats = [np.asarray(v["feature_vector"], dtype=np.float32) for v in vs]
        fake = [_fake_slice(f) for f in feats]
        # sanity: reconstructed raw features match stored vectors
        X0 = np.stack([slice_feature_raw(s) for s in fake])
        assert X0.shape[1] == 26
        vq = StemVQ(stem, k=getattr(args, stem)).fit(fake)
        outdir = Path(args.out) / stem
        vq.save(outdir)
        # medoid slice per code (nearest member in standardized space)
        Xs = vq.transform(X0)
        labels = vq.kmeans.predict(Xs)
        medoids = []
        for c in range(vq.n_clusters):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                medoids.append(vs[0]["id"])
                continue
            d = np.sum((Xs[members] - vq.centers_[c]) ** 2, axis=1)
            medoids.append(vs[int(members[int(np.argmin(d))])]["id"])
        counts = [int((labels == c).sum()) for c in range(vq.n_clusters)]
        manifest["stems"][stem] = {
            "k": vq.n_clusters,
            "sil_id": vq.silence_id,
            "hold_id": vq.silence_id + 1,
            "medoid_slice": medoids,
            "counts": counts,
        }
        print(f"{stem}: k={vq.n_clusters} medoids ok", flush=True)
    Path("frontend/public/data/tatum_vq.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print("wrote frontend/public/data/tatum_vq.json", flush=True)


if __name__ == "__main__":
    main()
