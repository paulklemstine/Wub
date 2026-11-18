#!/usr/bin/env python3
"""Train the value head that estimates future musical flow from a bar state.

The corpus supplies the trajectories: consecutive 4-bar phrases of an original
song give a chain of 32-d cells.  Immediate reward is the acoustic teacher's
join quality; the target is a discounted sum of the next 32 bars of that chain.
Track-disjoint splits prevent memorising a recording.  The exported 56->32->1
tanh net is loaded by the bar-tour beam search (and optionally by the browser
gate) as a soft estimate of where a route is heading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from wub_core.bar_tour import VALUE_INPUT_DIM, value_features
from wub_core.slice_flow import DIM, STEMS, acoustic_teacher


def is_val(track: str) -> bool:
    return int(hashlib.sha256(track.encode()).hexdigest()[:8], 16) % 10 == 0


def build_dataset(base: Path) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
    manifest = json.loads((base / "kit_manifest.json").read_text())
    model = json.loads((base / "slice_flow.json").read_text())
    features = np.fromfile(base / model["feature_file"], dtype=np.float32).reshape(-1, 4, 4, DIM)
    rows = {pid: i for i, pid in enumerate(model["phrase_ids"])}
    x: List[np.ndarray] = []
    y: List[float] = []
    val: List[bool] = []
    horizon, gamma = 8, 0.9
    for kit in manifest["kits"]:
        track = str(kit["track_id"])
        phrases = [p for p in kit["phrases"] if p["id"] in rows]
        phrases.sort(key=lambda p: int(p.get("start_bar", 0)))
        if len(phrases) < 2:
            continue
        for stem_index, stem in enumerate(STEMS):
            chain: List[np.ndarray] = []
            for phrase in phrases:
                for bar in range(4):
                    vec = features[rows[phrase["id"]], stem_index, bar]
                    if vec.any():
                        chain.append(np.asarray(vec, dtype=np.float64))
            if len(chain) < horizon + 2:
                continue
            rewards = [acoustic_teacher(chain[i], chain[i + 1], stem)
                       for i in range(len(chain) - 1)]
            for t in range(len(chain) - horizon - 1):
                history = chain[max(0, t - 8):t]
                repetitions = sum(1 for c in history if np.allclose(c, chain[t])) / max(1, len(history))
                state = value_features(chain[t], history, t * 0.5, repetitions)
                future = sum(gamma ** k * rewards[t + k] for k in range(horizon))
                x.append(state)
                y.append(future)
                val.append(is_val(track))
    return np.stack(x), np.asarray(y, np.float32), val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kits", default="frontend/public/data/kits")
    ap.add_argument("--out", default="frontend/public/data/kits/bar_tour_value.json")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=505)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x, y, val = build_dataset(Path(args.kits))
    val_mask = np.asarray(val)
    if val_mask.sum() < 16 or (~val_mask).sum() < 16:
        raise SystemExit("value dataset partition too small")
    mean = x[~val_mask].mean(axis=0)
    std = np.maximum(x[~val_mask].std(axis=0), 1e-3)
    device = torch.device(args.device)
    xt = torch.tensor((x[~val_mask] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[~val_mask], dtype=torch.float32, device=device)
    xv = torch.tensor((x[val_mask] - mean) / std, dtype=torch.float32, device=device)

    net = torch.nn.Sequential(
        torch.nn.Linear(VALUE_INPUT_DIM, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(order), 1024):
            idx = order[start:start + 1024]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(xv).squeeze(-1).cpu().numpy()
    mse = float(mean_squared_error(y[val_mask], pred))
    baseline = float(mean_squared_error(y[val_mask], np.full_like(pred, y[~val_mask].mean())))
    state = net.state_dict()
    artifact = {
        "format": "wub-bar-tour-value-v1",
        "version": 1,
        "input_dim": VALUE_INPUT_DIM,
        "net": {
            "mean": mean.tolist(), "std": std.tolist(),
            "w1": state["0.weight"].cpu().tolist(), "b1": state["0.bias"].cpu().tolist(),
            "w2": state["2.weight"].cpu().tolist(), "b2": state["2.bias"].cpu().tolist(),
        },
        "report": {
            "samples": int(len(x)), "train": int((~val_mask).sum()),
            "validation": int(val_mask.sum()), "val_mse": mse,
            "baseline_mse": baseline,
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(f"[value] {len(x)} states, val MSE {mse:.4f} (constant baseline {baseline:.4f}) -> {out}")


if __name__ == "__main__":
    main()
