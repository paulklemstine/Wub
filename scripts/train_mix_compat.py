#!/usr/bin/env python3
"""Train a per-bar mix-compat critic and export browser-ready weights.

Input: mix_compat_dataset.json from scripts/build_mix_compat_dataset.py
  - examples[].features: (4 bars, FEATURE_DIM)
  - examples[].label: graded compatibility in [0,1] (rhythm-prioritized)
  - examples[].per_bar: per-bar heuristic scores (aux target)

Model: small MLP over [bar_features, pairwise rhythm summaries, phrase agg]
predicts (compat, rhythm_subscore) per bar; phrase score = weighted(min, mean).
Exports JSON weights + scaler consumable by frontend/src/mix_compat.ts.

  PYTHONPATH=. python3 scripts/train_mix_compat.py \
    --dataset /tmp/wubtrain/mix_compat_dataset/mix_compat_dataset.json \
    --out data/mix_compat --epochs 40 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MixCompatNet(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # [compat, rhythm]
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data/mix_compat")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data = json.loads(Path(args.dataset).read_text())
    examples = data.get("examples") or []
    if not examples:
        raise SystemExit("no examples")
    feat_dim = int(data.get("summary", {}).get("feature_dim") or len(examples[0]["features"][0]))
    bars = int(data.get("summary", {}).get("bars_per_phrase") or 4)

    # Per-bar samples: (N*bars, feat_dim), targets (compat, rhythm)
    X, Yc, Yr = [], [], []
    groups = []  # example index per bar sample for min/mean eval
    for ei, ex in enumerate(examples):
        f = np.asarray(ex["features"], dtype=np.float32)
        if f.shape != (bars, feat_dim):
            continue
        per = ex.get("per_bar") or [ex.get("label", 0.0)] * bars
        per_r = ex.get("rhythm_per_bar") or [ex.get("rhythm", ex.get("label", 0.0))] * bars
        for b in range(bars):
            X.append(f[b])
            Yc.append(float(per[b]))
            Yr.append(float(per_r[b]))
            groups.append(ei)
    X = np.stack(X)
    Y = np.stack([Yc, Yr], axis=1).astype(np.float32)
    groups = np.asarray(groups)
    n_ex = len(examples)
    print(f"[mix-compat] samples={len(X)} bars={bars} feat_dim={feat_dim} examples~{n_ex}")

    # example-level split
    perm = rng.permutation(n_ex)
    n_val = max(1, int(args.val_frac * n_ex))
    val_ex = set(perm[:n_val].tolist())
    train_mask = np.array([g not in val_ex for g in groups])
    val_mask = ~train_mask

    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = ((X - mean) / std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(Xn[train_mask]), torch.from_numpy(Y[train_mask]))
    val_ds = TensorDataset(torch.from_numpy(Xn[val_mask]), torch.from_numpy(Y[val_mask]))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1024)

    model = MixCompatNet(feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = nn.MSELoss()

    best = {"val": 1e9, "state": None}
    history = []
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb) + 0.5 * loss_fn(pred[:, 1], yb[:, 1])  # rhythm emphasis
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            vtot = 0.0; n = 0
            preds = []; ys = []
            for xb, yb in val_dl:
                p = model(xb.to(device)).cpu()
                vtot += float(loss_fn(p, yb).item()) * len(xb)
                n += len(xb)
                preds.append(p.numpy()); ys.append(yb.numpy())
        vloss = vtot / max(1, n)
        pv = np.concatenate(preds) if preds else np.zeros((0, 2), np.float32)
        yv = np.concatenate(ys) if ys else np.zeros((0, 2), np.float32)
        corr = float(np.corrcoef(pv[:, 0], yv[:, 0])[0, 1]) if len(pv) > 1 else 0.0
        history.append({"epoch": ep, "train_loss": tot / max(1, len(train_ds)), "val_loss": vloss, "val_corr": corr})
        print(f"[mix-compat] ep {ep} train={tot/max(1,len(train_ds)):.4f} val={vloss:.4f} corr={corr:.4f}")
        if vloss < best["val"]:
            best = {"val": vloss, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    if best["state"]:
        model.load_state_dict(best["state"])
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "history": history,
                "feature_dim": feat_dim, "bars": bars, "val_loss": best["val"]}, out_dir / "mix_compat.pt")

    sd = model.state_dict()
    def flat(name):
        return sd[name].detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()
    export = {
        "version": 1,
        "model": "mix_compat_per_bar_v1",
        "feature_dim": feat_dim,
        "bars": bars,
        "rhythm_priority": True,
        "source_cap": 2,
        "aggregate": "0.6*min + 0.4*mean (per-bar compat)",
        "scaler": {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()},
        "weights": {
            "w1": flat("net.0.weight"), "b1": flat("net.0.bias"),
            "w2": flat("net.3.weight"), "b2": flat("net.3.bias"),
            "w3": flat("net.5.weight"), "b3": flat("net.5.bias"),
        },
        "val_loss": best["val"],
        "history": history,
    }
    (out_dir / "mix_compat.json").write_text(json.dumps(export, separators=(",", ":")))
    print(f"[mix-compat] wrote {out_dir}/mix_compat.json val={best['val']:.4f}")


if __name__ == "__main__":
    main()
