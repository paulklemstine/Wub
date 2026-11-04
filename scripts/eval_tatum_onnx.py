"""Musical sanity check for the shipped ONNX tatum filler (int8 vs fp32).

Feeds the exact window the browser sends: ctx-6 committed tokens followed by
the 6-token header of the next bar. Reports the same onset/hold rates as
scripts/eval_tatum.py.

Usage: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/eval_tatum_onnx.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort

STEMS = ("drums", "bass", "other", "vocals")
OFF = {"drums": 0, "bass": 1026, "other": 2052, "vocals": 3078}
SIL = {"drums": 1024, "bass": 2050, "other": 3076, "vocals": 4102}
HOLD = {"drums": 1025, "bass": 2051, "other": 3077, "vocals": 4103}


def gen(sess, ctx, history, temp=1.0, top_k=20, seed=0):
    """Mirror arranger_worker.ts: right-pad to ctx, read the last real token."""
    rng = random.Random(seed)
    toks = list(history)
    out = []
    for _ in range(64):
        win = toks[-ctx:]
        ids = np.asarray([win + [0] * (ctx - len(win))], dtype=np.int64)
        logits = sess.run(None, {"tokens": ids})[0][0, len(win) - 1] / max(1e-6, temp)
        idx = np.argpartition(-logits, top_k)[:top_k]
        row = logits[idx]
        w = np.exp(row - row.max())
        p = w / w.sum()
        nxt = int(rng.choices(idx.tolist(), weights=p.tolist(), k=1)[0])
        toks.append(nxt)
        out.append(nxt)
    return out


def rates(steps):
    out = {}
    for k, s in enumerate(STEMS):
        codes = [steps[i * 4 + k] for i in range(16)]
        ons = sum(1 for c in codes if OFF[s] <= c < OFF[s] + 1024 and c != SIL[s])
        holds = sum(1 for c in codes if c == HOLD[s])
        out[s] = (ons / 16, holds / 16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/tatum_dataset_full.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--bar", type=int, default=4, help="bar index whose header we feed")
    args = ap.parse_args()
    meta = json.loads(Path("frontend/public/data/tatum_gpt.json").read_text())
    ctx = int(meta["ctx"])
    data = json.loads(Path(args.dataset).read_text())
    k = args.bar
    tracks = [t for t in data["val"] if t["bars"] > k][: args.n]
    print(f"tracks {len(tracks)} bar {k} ctx {ctx}")

    for name in ("tatum_gpt.onnx", "tatum_gpt_int8.onnx"):
        sess = ort.InferenceSession(f"frontend/public/data/{name}", providers=["CPUExecutionProvider"])
        acc = {s: [0.0, 0.0] for s in STEMS}
        for ti, t in enumerate(tracks):
            toks = t["tokens"]
            header = toks[k * 70: k * 70 + 6]
            # Three bars of history: header lands at position 210, exactly as
            # in training (headers at 0/70/140/210).
            hist = toks[max(0, k * 70 - 210): k * 70]
            steps = gen(sess, ctx, hist + header, seed=ti)
            r = rates(steps)
            for s in STEMS:
                acc[s][0] += r[s][0]
                acc[s][1] += r[s][1]
        n = len(tracks)
        print(name)
        for s in STEMS:
            print(f"  {s:6s} onset {acc[s][0]/n:.3f} hold {acc[s][1]/n:.3f}")


if __name__ == "__main__":
    main()
