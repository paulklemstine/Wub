"""A/B the tatum filler's context: 6-token header vs 4 bars of history.

For val tracks, take bars 0-2 as history, feed the bar-3 header, generate the
64 steps, then score the generated bar against the real bar 3:
  - onset F1 per stem (does the model predict *where* hits land?)
  - HOLD/SIL/onset rate drift
  - distinct bars

Usage: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/eval_tatum_context.py \
         --ckpt data/tatum256_eval.pt --n 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.train_bar_arranger import BarGPT

STEMS = ("drums", "bass", "other", "vocals")
OFF = {"drums": 0, "bass": 1026, "other": 2052, "vocals": 3078}
SIL = {"drums": 1024, "bass": 2050, "other": 3076, "vocals": 4102}
HOLD = {"drums": 1025, "bass": 2051, "other": 3077, "vocals": 4103}
TOK_PER_BAR = 70


@torch.no_grad()
def gen(model, history, device, temp=1.0, top_k=20, seed=0):
    rng = random.Random(seed)
    toks = list(history)
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
    return out


def bar_stats(steps):
    """steps: 64 flat tokens -> per-stem onset/hold/sil rates."""
    out = {}
    for k, s in enumerate(STEMS):
        codes = [steps[i * 4 + k] for i in range(16)]
        ons = sum(1 for c in codes if OFF[s] <= c < OFF[s] + 1024 and c != SIL[s])
        holds = sum(1 for c in codes if c == HOLD[s])
        sils = sum(1 for c in codes if c == SIL[s])
        out[s] = (ons / 16, holds / 16, sils / 16)
    return out


def onset_f1(gen_steps, ref_steps):
    f1s = []
    for k, s in enumerate(STEMS):
        g = [1 if OFF[s] <= gen_steps[i * 4 + k] < OFF[s] + 1024 and gen_steps[i * 4 + k] != SIL[s] else 0 for i in range(16)]
        r = [1 if OFF[s] <= ref_steps[i * 4 + k] < OFF[s] + 1024 and ref_steps[i * 4 + k] != SIL[s] else 0 for i in range(16)]
        tp = sum(1 for a, b in zip(g, r) if a and b)
        fp = sum(1 for a, b in zip(g, r) if a and not b)
        fn = sum(1 for a, b in zip(g, r) if b and not a)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return f1s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/tatum256_eval.pt")
    ap.add_argument("--dataset", default="data/tatum_dataset.json")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    device = "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = BarGPT(vocab=cfg["vocab"], d=cfg["d"], layers=cfg["layers"],
                   heads=cfg["heads"], ctx=cfg["ctx"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = json.loads(Path(args.dataset).read_text())
    tracks = [t for t in data["val"] if t["bars"] >= 5][: args.n]
    print(f"tracks: {len(tracks)} ckpt={args.ckpt} ctx={cfg['ctx']}")

    f1_ctx, f1_no = [], []
    rates = {"ctx": [], "noctx": [], "ref": []}
    distinct = {"ctx": set(), "noctx": set()}
    for ti, t in enumerate(tracks):
        toks = t["tokens"]
        hist = toks[: 3 * TOK_PER_BAR]
        header = toks[3 * TOK_PER_BAR: 3 * TOK_PER_BAR + 6]
        ref = toks[3 * TOK_PER_BAR + 6: 4 * TOK_PER_BAR]
        g_ctx = gen(model, hist + header, device, seed=ti)
        g_no = gen(model, header, device, seed=ti)
        f1_ctx.append(onset_f1(g_ctx, ref))
        f1_no.append(onset_f1(g_no, ref))
        rates["ctx"].append(bar_stats(g_ctx))
        rates["noctx"].append(bar_stats(g_no))
        rates["ref"].append(bar_stats(ref))
        distinct["ctx"].add(tuple(g_ctx))
        distinct["noctx"].add(tuple(g_no))

    def avg(rows):
        return {s: tuple(sum(r[s][j] for r in rows) / len(rows) for j in range(3)) for s in STEMS}

    print("onset F1 vs real bar (mean over stems):")
    for k, s in enumerate(STEMS):
        print(f"  {s:6s} with-ctx {sum(x[k] for x in f1_ctx)/len(f1_ctx):.3f}   header-only {sum(x[k] for x in f1_no)/len(f1_no):.3f}")
    print("rates onset/hold/sil:")
    for tag in ("ctx", "noctx", "ref"):
        a = avg(rates[tag])
        print(f"  {tag:6s} " + " ".join(f"{s}={a[s][0]:.2f}/{a[s][1]:.2f}/{a[s][2]:.2f}" for s in STEMS))
    print(f"distinct bars: with-ctx {len(distinct['ctx'])}/{len(tracks)}  header-only {len(distinct['noctx'])}/{len(tracks)}")


if __name__ == "__main__":
    main()
