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


def load_cells(kits: Path) -> Tuple[Dict[Tuple[str, str, int, int], np.ndarray], List[str]]:
    artifact = json.loads((kits / "slice_flow.json").read_text())
    if artifact.get("format") != "slice_flow_v1" or artifact.get("feature_dim") != DIM:
        raise SystemExit("slice_flow.json has an unexpected format or feature dimension")
    feature_path = kits / artifact["feature_file"]
    features = np.fromfile(feature_path, dtype=np.float32)
    ids = artifact["phrase_ids"]
    if features.size != len(ids) * 16 * DIM:
        raise SystemExit("slice_flow feature file length does not match phrase_ids")
    cells: Dict[Tuple[str, str, int, int], np.ndarray] = {}
    tracks: set[str] = set()
    for phrase_index, pid in enumerate(ids):
        match = PID_RE.match(str(pid))
        if not match:
            continue
        track = match.group(1)
        tracks.add(track)
        for stem_index, stem in enumerate(STEMS):
            for bar in range(4):
                offset = ((phrase_index * 4 + stem_index) * 4 + bar) * DIM
                vec = features[offset:offset + DIM].astype(np.float64)
                if not np.all(np.isfinite(vec)) or not np.any(vec):
                    continue
                cells[(stem, track, phrase_index, bar)] = vec
    return cells, sorted(tracks)


def make_features(a: np.ndarray, b: np.ndarray, stem: int, lag: int, age: float) -> np.ndarray:
    out = np.zeros(INPUT_DIM, dtype=np.float32)
    out[:DIM] = a
    out[DIM:2 * DIM] = b
    out[2 * DIM:3 * DIM] = np.abs(a - b)
    out[3 * DIM + stem] = 1.0
    out[3 * DIM + len(STEMS) + lag] = 1.0
    out[-1] = age
    return out


def build_pairs(cells: Dict[Tuple[str, str, int, int], np.ndarray],
                tracks: List[str], seed: int):
    rng = np.random.default_rng(seed)
    by_stem: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        stem: {} for stem in STEMS
    }
    for (stem, track, phrase, bar) in cells:
        by_stem[stem].setdefault(track, []).append((phrase, bar))

    x_pos: List[np.ndarray] = []
    x_neg: List[np.ndarray] = []
    train_tracks = [t for t in tracks if not is_val(t)]
    for stem_index, stem in enumerate(STEMS):
        groups = by_stem[stem]
        sources = [t for t in groups if len(groups[t]) >= 3]
        for track in sources:
            anchors = groups[track]
            for phrase, bar in anchors:
                for lag_index, lag in enumerate(LAGS):
                    target = cells.get((stem, track, phrase + lag // 4, bar))
                    if target is None:
                        continue
                    age = float(rng.uniform(0.1, 1.4))
                    x_pos.append(make_features(cells[(stem, track, phrase, bar)],
                                               target, stem_index, lag_index, age))
                    # Cross-song distractor at the same acoustic distance.
                    other = track
                    for _ in range(8):
                        other = str(rng.choice(train_tracks))
                        if other != track and other in groups:
                            break
                    if other != track and other in groups:
                        donor = groups[other][int(rng.integers(len(groups[other])))]
                        donor_vec = cells.get((stem, other, donor[0], donor[1]))
                        if donor_vec is not None:
                            x_neg.append(make_features(cells[(stem, track, phrase, bar)],
                                                       donor_vec, stem_index, lag_index, age))
                    # Same-song but too far apart to be a reprise: blocks the
                    # shortcut of learning only "this is the same record".
                    far = cells.get((stem, track, phrase + HARD_LAG // 4, bar))
                    if far is not None:
                        x_neg.append(make_features(cells[(stem, track, phrase, bar)],
                                                   far, stem_index, lag_index, 2.0))
    if len(x_pos) < 64 or len(x_neg) < 64:
        raise SystemExit("not enough same-song reprise pairs in the shipped bank")
    n = min(len(x_pos), len(x_neg))
    order_pos = rng.permutation(len(x_pos))[:n]
    order_neg = rng.permutation(len(x_neg))[:n]
    x = np.stack([x_pos[i] for i in order_pos] + [x_neg[i] for i in order_neg])
    y = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)])
    split = np.asarray([is_val(t) for t in tracks])  # only for the report
    return x, y, {"pairs": int(2 * n), "tracks": len(tracks),
                  "validation_tracks": int(split.sum())}


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
    x, y, counts = build_pairs(cells, tracks, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(y))
    val_count = max(1, int(len(order) * 0.2))
    val_idx, train_idx = order[:val_count], order[val_count:]

    mean = x[train_idx].mean(axis=0)
    std = np.maximum(x[train_idx].std(axis=0), 1e-3)
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
        for start in range(0, len(xt), 2048):
            batch = slice(start, start + 2048)
            opt.zero_grad()
            loss = loss_fn(net(xt[batch]).squeeze(-1), yt[batch])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        logits = net(xv).squeeze(-1).cpu().numpy()
    auc = float(roc_auc_score(y[val_idx], logits))
    artifact = {
        "format": MOTIF_FORMAT,
        "version": 1,
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(f"[motif] {counts['pairs']} pairs from {counts['tracks']} tracks "
          f"({counts['validation_tracks']} held out) -> {out}")
    print(f"[motif] val AUC {auc:.3f} (positive rate {y[val_idx].mean():.3f})")


MOTIF_FORMAT = "wub-motif-return-v1"

if __name__ == "__main__":
    main()
