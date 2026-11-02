"""Evaluate the tatum-filler GPT musically (CPU-safe).

For N val bars: feed the 6-token bar header, generate 16 steps x 4 stems
(top-k sampling), then report vs training-set reference rates:
- onset_rate per stem (want near training: drums~0.38 bass~0.15 other~0.18 vox~0.14)
- hold_frac per stem (want near training ~0.53-0.59; collapse to all-HOLD = fail)
- distinct code combos / generated bars (want high; guards mode collapse)

Usage: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/eval_tatum.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.train_bar_arranger import BarGPT

STEMS = ("drums", "bass", "other", "vocals")
OFF = {"drums": 0, "bass": 386, "other": 644, "vocals": 1030}
K = {"drums": 384, "bass": 256, "other": 384, "vocals": 256}
SIL = {s: OFF[s] + K[s] for s in STEMS}
HOLD = {s: OFF[s] + K[s] + 1 for s in STEMS}


@torch.no_grad()
def generate_bar(model, header, device, temp=1.0, top_k=20, seed=0):
    rng = random.Random(seed)
    toks = list(header)
    out = []
    for _ in range(64):
        x = torch.tensor([toks[-model.ctx:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(1e-6, temp)
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        nxt = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
        toks.append(nxt)
        out.append(nxt)
    return [out[i * 4:(i + 1) * 4] for i in range(16)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/tatum_gpt.pt")
    ap.add_argument("--dataset", default="data/tatum_dataset.json")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = BarGPT(vocab=cfg["vocab"], d=cfg["d"], layers=cfg["layers"],
                   heads=cfg["heads"], ctx=cfg["ctx"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = json.loads(Path(args.dataset).read_bytes())
    val = ds["val"]
    n = min(args.n, len(val))
    on = {s: 0 for s in STEMS}
    hold = {s: 0 for s in STEMS}
    tot = 0
    seen = set()
    for i in range(n):
        seq = val[i]["tokens"]
        nbars = len(seq) // 70
        if nbars < 2:
            continue
        b = (i * 7) % max(1, nbars - 1)
        header = seq[b * 70:b * 70 + 6]
        steps = generate_bar(model, header, device, seed=1000 + i)
        tot += 16
        seen.add(tuple(t for st in steps for t in st))
        for k, s in enumerate(STEMS):
            for st in steps:
                v = st[k]
                if v == SIL[s]:
                    continue
                if v == HOLD[s]:
                    hold[s] += 1
                else:
                    on[s] += 1
    print(f"bars: {n}", flush=True)
    print("  onset rate:", {s: round(on[s] / max(1, tot), 3) for s in STEMS}, "(train ~0.38/0.15/0.18/0.14)")
    print("  hold frac:", {s: round(hold[s] / max(1, tot), 3) for s in STEMS}, "(train ~0.53-0.59)")
    print(f"  distinct 16-step bars: {len(seen)}/{n} (want high)", flush=True)


if __name__ == "__main__":
    main()
