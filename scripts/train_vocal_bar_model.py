#!/usr/bin/env python3
"""Train a *next vocal bar* model from real songs.

This is deliberately not another audio-to-articulator model.  It learns the
two things the shipped composer currently guesses with heuristics:

* what the next vocal bar's onset shape / pitch class looks like given the
  last two bars (so a hold can develop instead of copying one cell), and
* how many more bars the current vocal run is likely to last (so a phrase is
  held for a learned length instead of a fixed 4 bars).

Track-disjoint splits are assigned before any examples are emitted.  The
exported nets are compact enough to run in the browser at block preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from scripts.build_vocal_flow import vocal_bars

DIM = 1 + 1 + 12 + 16
HISTORY = 2
MAX_RUN = 16


def is_val(track: str) -> bool:
    return int(hashlib.sha256(track.encode()).hexdigest()[:8], 16) % 10 == 0


def bar_features(steps: np.ndarray, chroma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar input vector, next-bar target vector, and run-length target."""
    count = steps.shape[0]
    vecs = np.zeros((count, DIM), dtype=np.float32)
    for b in range(count):
        onset = steps[b]
        root = int(np.argmax(chroma[b])) if chroma[b].any() else 11
        vecs[b, 0] = 1.0 if onset.sum() >= 2 else 0.0
        vecs[b, 1] = min(1.0, float(onset.sum()) / 8.0)
        vecs[b, 2 + root] = 1.0
        vecs[b, 14:30] = onset
    targets = np.zeros((count, 29), dtype=np.float32)
    for b in range(count - 1):
        targets[b, :12] = vecs[b + 1, 2:14]
        targets[b, 12:28] = vecs[b + 1, 14:30]
        targets[b, 28] = vecs[b + 1, 0]
    lengths = np.zeros(count, dtype=np.float32)
    for b in range(count):
        remaining = 0
        while b + remaining < count and remaining < MAX_RUN and vecs[b + remaining, 0] > 0.5:
            remaining += 1
        lengths[b] = remaining / MAX_RUN
    return vecs, targets, lengths


def build_dataset(processed: Path, max_tracks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    lengths: List[float] = []
    val: List[bool] = []
    paths = sorted(processed.glob("*.json"))
    if max_tracks > 0:
        paths = paths[:max_tracks]
    for path in paths:
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        steps, chroma = vocal_bars(doc)
        if steps.shape[0] < 24:
            continue
        track = str(doc.get("track_id") or path.stem)
        vecs, tgts, runs = bar_features(steps, chroma)
        for b in range(HISTORY, len(vecs) - 1):
            xs.append(np.concatenate([vecs[b - HISTORY], vecs[b], vecs[b - HISTORY + 1], vecs[b]]))
            targets.append(tgts[b])
            lengths.append(runs[b])
            val.append(is_val(track))
    if len(xs) < 64:
        raise SystemExit("not enough vocal bar sequences in the corpus")
    return np.stack(xs), np.stack(targets), np.asarray(lengths, np.float32), np.asarray(val)


def export(net: torch.nn.Module, mean: np.ndarray, std: np.ndarray) -> dict:
    state = net.state_dict()
    return {
        "mean": mean.tolist(), "std": std.tolist(),
        "w1": state["0.weight"].cpu().tolist(), "b1": state["0.bias"].cpu().tolist(),
        "w2": state["2.weight"].cpu().tolist(), "b2": state["2.bias"].cpu().tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--max-tracks", type=int, default=900)
    ap.add_argument("--out", default="frontend/public/data/kits/vocal_bar_model.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=505)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x, y, run, val = build_dataset(Path(args.processed), args.max_tracks)
    val_mask = np.asarray(val, dtype=bool)
    if val_mask.sum() < 16 or (~val_mask).sum() < 16:
        raise SystemExit("vocal bar model partition too small")
    mean = x[~val_mask].mean(axis=0)
    std = np.maximum(x[~val_mask].std(axis=0), 1e-3)
    device = torch.device(args.device)
    xt = torch.tensor((x[~val_mask] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[~val_mask], dtype=torch.float32, device=device)
    rt = torch.tensor(run[~val_mask], dtype=torch.float32, device=device)
    xv = torch.tensor((x[val_mask] - mean) / std, dtype=torch.float32, device=device)

    next_net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 48), torch.nn.Tanh(),
        torch.nn.Linear(48, y.shape[1]),
    ).to(device)
    len_net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 24), torch.nn.Tanh(),
        torch.nn.Linear(24, 1),
    ).to(device)
    opt = torch.optim.Adam(list(next_net.parameters()) + list(len_net.parameters()), lr=0.002)
    next_loss = torch.nn.MSELoss()
    len_loss = torch.nn.MSELoss()
    for _ in range(args.epochs):
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(order), 1024):
            idx = order[start:start + 1024]
            opt.zero_grad()
            loss = next_loss(next_net(xt[idx]), yt[idx]) + len_loss(len_net(xt[idx]).squeeze(-1), rt[idx])
            loss.backward()
            opt.step()
    next_net.eval()
    len_net.eval()
    with torch.no_grad():
        pred_run = len_net(xv).squeeze(-1).cpu().numpy()
    baseline = float(mean_absolute_error(run[val_mask], np.full_like(pred_run, run[~val_mask].mean())))
    mae = float(mean_absolute_error(run[val_mask], pred_run))
    artifact = {
        "format": "wub-vocal-bar-v1",
        "version": 1,
        "input_dim": x.shape[1],
        "next_dim": y.shape[1],
        "next": export(next_net, mean, std),
        "length": export(len_net, mean, std),
        "report": {
            "samples": int(len(x)), "train": int((~val_mask).sum()),
            "validation": int(val_mask.sum()), "run_mae": mae,
            "run_baseline_mae": baseline,
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(f"[vocal-bar] {len(x)} bars, run MAE {mae:.4f} (baseline {baseline:.4f}) -> {out}")


if __name__ == "__main__":
    main()
