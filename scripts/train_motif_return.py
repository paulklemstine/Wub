#!/usr/bin/env python3
"""Train the motif-return head: does a later cell bring an earlier idea back?

Supervision comes from the shipped acoustic cell bank, not source identity:
positives are cells 8/16/32 bars later in the SAME original song (where a
reprise or variation usually lives), negatives are cross-song pairs plus
same-song cells 64 bars away.  Track partitions are assigned before sampling,
so a held-out song never contributes to both sides of the split.

The exported 104->32->1 tanh network is a soft preference, not a rule.  It is
loaded optionally by frontend/src/slice_flow.ts and used to favour a candidate
that recalls the current foundation after ~60-120 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from wub_core.slice_flow import DIM, STEMS

LAGS: Tuple[int, ...] = (8, 16, 32)
HARD_LAG = 64
INPUT_DIM = DIM * 3 + len(STEMS) + len(LAGS) + 1
PID_RE = re.compile(r"^kit_(.+)_p\d+$")


def is_val(track: str) -> bool:
    return int(hashlib.sha256(track.encode()).hexdigest()[:8], 16) % 10 == 0


def load_cells(kits: Path):
    artifact = json.loads((kits / "slice_flow.json").read_text())
    if artifact.get("format") != "slice_flow_v1" or artifact.get("feature_dim") != DIM:
        raise ValueError("slice_flow feature schema mismatch")
    features = np.fromfile(kits / artifact["feature_file"], dtype="<f4")
    ids = artifact["phrase_ids"]
    if features.size != len(ids) * 16 * DIM or not np.isfinite(features).all():
        raise ValueError("invalid slice-flow features")
    rows = {pid: i for i, pid in enumerate(ids)}
    manifest = json.loads((kits / "kit_manifest.json").read_text())
    cells = {}
    for kit in manifest["kits"]:
        track = str(kit["track_id"])
        for phrase in sorted(kit["phrases"], key=lambda p: (int(p["start_bar"]), p["id"])):
            if phrase["id"] not in rows:
                continue
            for si, stem in enumerate(STEMS):
                for bar in range(4):
                    offset = ((rows[phrase["id"]] * 4 + si) * 4 + bar) * DIM
                    vec = features[offset:offset + DIM].astype(np.float64)
                    if vec.any():
                        # Dictionary positions and phrase ordinals are NOT time.
                        cells.setdefault((stem, track, int(phrase["start_bar"]) + bar), vec)
    return cells, sorted({key[1] for key in cells})


def make_features(a: np.ndarray, b: np.ndarray, stem: int, lag: int, age: float) -> np.ndarray:
    out = np.zeros(INPUT_DIM, dtype=np.float32)
    out[:DIM] = a
    out[DIM:2 * DIM] = b
    out[2 * DIM:3 * DIM] = np.abs(a - b)
    out[3 * DIM + stem] = 1.0
    out[3 * DIM + len(STEMS) + lag] = 1.0
    out[-1] = age
    return out


def build_pairs(cells, tracks: List[str], seed: int):
    rng = np.random.default_rng(seed)
    by_stem = {stem: {} for stem in STEMS}
    for stem, track, bar in cells:
        by_stem[stem].setdefault(track, []).append(bar)
    vectors, labels, split = [], [], []
    donors_used = {False: set(), True: set()}
    for si, stem in enumerate(STEMS):
        groups = by_stem[stem]
        for track, anchors in sorted(groups.items()):
            held = is_val(track)
            donors = sorted(t for t in groups if t != track and is_val(t) == held)
            if not donors:
                continue
            for bar in sorted(anchors):
                for lag_index, lag in enumerate(LAGS):
                    target = cells.get((stem, track, bar + lag))
                    if target is None:
                        continue
                    age = lag / 16  # normalized elapsed time; runtime supplies exact tempo
                    anchor = cells[(stem, track, bar)]
                    vectors.append(make_features(anchor, target, si, lag_index, age))
                    labels.append(1.); split.append(held)
                    other = donors[int(rng.integers(len(donors)))]
                    donor_bar = int(rng.choice(groups[other]))
                    vectors.append(make_features(anchor, cells[(stem, other, donor_bar)], si, lag_index, age))
                    labels.append(0.); split.append(held)
                    donors_used[held].add(other)
                    far = cells.get((stem, track, bar + HARD_LAG))
                    if far is not None:
                        # Keep context identical: lag/age must not reveal the label.
                        vectors.append(make_features(anchor, far, si, lag_index, age))
                        labels.append(0.); split.append(held)
    x = np.asarray(vectors, np.float32); y = np.asarray(labels, np.float32)
    val = np.asarray(split, bool)
    if min(int(val.sum()), int((~val).sum())) < 32:
        raise ValueError("not enough track-disjoint motif examples")
    return x, y, val, {"pairs": len(y), "tracks": len(tracks),
        "validation_tracks": sum(is_val(t) for t in tracks),
        "train_track_ids": [t for t in tracks if not is_val(t)],
        "validation_track_ids": [t for t in tracks if is_val(t)],
        "training_donor_tracks": sorted(donors_used[False]),
        "validation_donor_tracks": sorted(donors_used[True]),
        "split_method": "track_hash_before_pairing", "lag_unit": "source_bars", "age_unit": "16_bars"}


def export(net: torch.nn.Module, mean: np.ndarray, std: np.ndarray) -> dict:
    w1, b1 = net[0].weight.detach().cpu().numpy(), net[0].bias.detach().cpu().numpy()
    w2, b2 = net[2].weight.detach().cpu().numpy(), net[2].bias.detach().cpu().numpy()
    return {
        "mean": mean.tolist(), "std": std.tolist(),
        "w1": w1.tolist(), "b1": b1.tolist(),
        "w2": w2.tolist(), "b2": b2.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kits", default="frontend/public/data/kits")
    ap.add_argument("--out", default="frontend/public/data/kits/motif_return.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    kits = Path(args.kits)
    cells, tracks = load_cells(kits)
    x, y, val_mask, counts = build_pairs(cells, tracks, args.seed)
    torch.set_num_threads(4)
    train_idx = np.flatnonzero(~val_mask); val_idx = np.flatnonzero(val_mask)
    mean = x[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_std = x[train_idx].astype(np.float64).std(axis=0)
    std = np.maximum(raw_std, .03).astype(np.float32)
    xt = torch.tensor((x[train_idx] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    xv = torch.tensor((x[val_idx] - mean) / std, dtype=torch.float32, device=device)

    net = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), 2048):
            batch = order[start:start + 2048]
            opt.zero_grad()
            loss = loss_fn(net(xt[batch]).squeeze(-1), yt[batch])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        net[0].weight[:, torch.tensor(raw_std < 1e-5, device=device)] = 0
        logits = net(xv).squeeze(-1).cpu().numpy()
    auc = float(roc_auc_score(y[val_idx], logits))
    artifact = {
        "format": MOTIF_FORMAT,
        "version": 2,
        "feature_dim": DIM,
        "input_dim": INPUT_DIM,
        "lags": list(LAGS),
        "net": export(net, mean, std),
        "report": {
            **counts,
            "val_auc": auc,
            "val_positive_rate": float(y[val_idx].mean()),
            "input_dim": INPUT_DIM,
        },
    }
    if not np.isfinite(auc) or auc < .6:
        raise RuntimeError("motif validation did not pass; shipping model unchanged")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(f"[motif] {counts['pairs']} pairs from {counts['tracks']} tracks "
          f"({counts['validation_tracks']} held out) -> {out}")
    print(f"[motif] val AUC {auc:.3f} (positive rate {y[val_idx].mean():.3f})")


MOTIF_FORMAT = "wub-motif-return-v1"

if __name__ == "__main__":
    main()
