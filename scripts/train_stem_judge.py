"""Train one per-stem transition judge.

The four stem streams need their own notion of "this phrase belongs with that
one".  The mix-level judge already models real bar-to-bar transitions; this
model is trained per stem on **same-song** phrase pairs versus cross-song
pairs, so it learns the stem's timbre/role fit across songs.

Pipeline:
  1. build_stem_bars.py    -> data/stem_bars/<stem>   (full corpus)
  2. filter_stem_dataset.py -> data/stem_ship/<stem>  (390 shipping tracks)
  3. this script           -> data/stem_judge_<stem>.pt

Usage:
    PYTHONPATH=. python3 scripts/train_stem_judge.py \
      --bars data/stem_ship --stem drums \
      --out data/stem_judge_drums.pt --log aj/logs/stem_drums.json \
      --epochs 12 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_judge_model import AudioJudge  # noqa: E402

NEG_TYPES = ("cross_song_match", "cross_key", "cross_style")


class PairDataset(Dataset):
    def __init__(self, arr: np.ndarray, a: np.ndarray, b: np.ndarray, y: np.ndarray):
        self.arr = arr
        self.a = a
        self.b = b
        self.y = y

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(np.asarray(self.arr[self.a[i]])),
            torch.from_numpy(np.asarray(self.arr[self.b[i]])),
            torch.tensor(float(self.y[i])),
        )


def load(bars: Path):
    idx = json.loads((bars / "bar_index.json").read_text())
    arr = np.array(np.load(bars / idx["mel_file"], mmap_mode="r"))
    return idx, arr


def split_tracks(idx, seed: int = 505, val_frac: float = 0.10):
    tracks = list(dict.fromkeys(idx["track_id"]))
    style_of = {}
    for tid in tracks:
        style_of[tid] = str(idx["style"][idx["track_id"].index(tid)])
    rng = random.Random(seed)
    train, val = [], []
    for style in sorted(set(style_of.values())):
        ids = [t for t in tracks if style_of[t] == style]
        rng.shuffle(ids)
        n_val = max(2, int(round(val_frac * len(ids))))
        val.extend(ids[:n_val])
        train.extend(ids[n_val:])
    return train, val, style_of


def track_rows(idx):
    rows = defaultdict(list)
    for i, tid in enumerate(idx["track_id"]):
        rows[str(tid)].append(i)
    for tid in rows:
        rows[tid].sort(key=lambda i: int(idx["start_bar"][i]))
    return rows


def build_candidates(idx, rows):
    by_style_key = defaultdict(list)
    by_style = defaultdict(list)
    bpm = {}
    key = {}
    style = {}
    for tid, rs in rows.items():
        first = rs[0]
        style[tid] = str(idx["style"][first])
        key[tid] = str(idx["key"][first])
        bpm[tid] = float(idx["bpm"][first])
        by_style_key[(style[tid], key[tid])].append(tid)
        by_style[style[tid]].append(tid)
    same = defaultdict(list)
    diff_key = defaultdict(list)
    cross_style = defaultdict(list)
    for tid in rows:
        for other in by_style_key[(style[tid], key[tid])]:
            if other != tid and abs(bpm[other] - bpm[tid]) <= 4.0:
                same[tid].append(other)
        for other in by_style[style[tid]]:
            if other == tid or key[other] == key[tid]:
                continue
            if abs(bpm[other] - bpm[tid]) <= 10.0:
                diff_key[tid].append(other)
        for other in rows:
            if style[other] != style[tid] and abs(bpm[other] - bpm[tid]) <= 10.0:
                cross_style[tid].append(other)
    return same, diff_key, cross_style, bpm, key, style


def build_pairs(idx, rows, tracks, same, diff_key, cross_style, rng, max_per_track=60):
    pos = []
    for tid in tracks:
        rs = rows[tid]
        if len(rs) < 2:
            continue
        pairs = [(a, b) for a in rs for b in rs if a != b]
        if len(pairs) > max_per_track:
            pairs = rng.sample(pairs, max_per_track)
        pos.extend(pairs)
    pos = np.asarray(pos, dtype=np.int64)
    neg = {t: np.full((len(pos), 2), -1, dtype=np.int64) for t in NEG_TYPES}
    for i, (ra, rb) in enumerate(pos):
        tid = str(idx["track_id"][ra])
        candidates = [
            ("cross_song_match", same.get(tid) or []),
            ("cross_key", diff_key.get(tid) or []),
            ("cross_style", cross_style.get(tid) or []),
        ]
        for name, pool in candidates:
            if not pool:
                continue
            other = pool[int(rng.integers(len(pool)))]
            rc = rows[other][int(rng.integers(len(rows[other])))]
            neg[name][i] = (ra, rc)
    return pos, neg


def make_epoch(pos, neg, epoch, rng):
    a = [pos[:, 0]]
    b = [pos[:, 1]]
    y = [np.ones(len(pos), np.float32)]
    for ti, name in enumerate(NEG_TYPES):
        sel = np.where((np.arange(len(pos)) + epoch + ti) % len(NEG_TYPES) == ti)[0]
        sel = sel[neg[name][sel, 0] >= 0]
        if len(sel) == 0:
            continue
        a.append(neg[name][sel, 0])
        b.append(neg[name][sel, 1])
        y.append(np.zeros(len(sel), np.float32))
    a = np.concatenate(a)
    b = np.concatenate(b)
    y = np.concatenate(y)
    perm = rng.permutation(len(y))
    return a[perm], b[perm], y[perm]


def score(model, arr, pairs, device, batch=512):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            p = pairs[i:i + batch]
            a = torch.from_numpy(np.asarray(arr[p[:, 0]])).float().unsqueeze(1).to(device)
            b = torch.from_numpy(np.asarray(arr[p[:, 1]])).float().unsqueeze(1).to(device)
            out.append(model(a, b).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, arr, pos, neg, device):
    pl = score(model, arr, pos, device)
    metrics = {"n_pos": int(len(pos))}
    logs = [pl]
    for name in NEG_TYPES:
        valid = neg[name][:, 0] >= 0
        if not valid.any():
            metrics[f"auc_{name}"] = float("nan")
            continue
        nl = score(model, arr, neg[name][valid], device)
        metrics[f"auc_{name}"] = float(roc_auc_score(
            np.r_[np.ones(len(pl)), np.zeros(len(nl))], np.r_[pl, nl]
        ))
        metrics[f"n_{name}"] = int(valid.sum())
        logs.append(nl)
    y = np.r_[np.ones(len(pl)), np.zeros(sum(len(x) for x in logs[1:]))]
    s = np.concatenate(logs)
    metrics["auc_overall"] = float(roc_auc_score(y, s))
    pred = (s >= 0.0).astype(np.int64)
    metrics["acc"] = float(accuracy_score(y, pred))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/stem_ship")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    bars = Path(args.bars) / args.stem
    idx, arr = load(bars)
    print(f"[stem-judge:{args.stem}] {arr.shape} phrases", flush=True)
    train_tids, val_tids, style_of = split_tracks(idx, seed=args.seed)
    rows = track_rows(idx)
    same, diff_key, cross_style, *_ = build_candidates(idx, rows)
    pos_tr, neg_tr = build_pairs(idx, rows, train_tids, same, diff_key, cross_style, rng)
    pos_va, neg_va = build_pairs(idx, rows, val_tids, same, diff_key, cross_style, rng)
    print(f"[stem-judge:{args.stem}] train {len(pos_tr)} pos / val {len(pos_va)} pos", flush=True)

    model = AudioJudge(emb_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"auc": -1.0, "epoch": -1}
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        a, b, y = make_epoch(pos_tr, neg_tr, epoch, rng)
        ds = PairDataset(arr, a, b, y)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
        model.train()
        total = 0.0
        for xa, xb, yb in dl:
            xa = xa.unsqueeze(1).to(device)
            xb = xb.unsqueeze(1).to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xa, xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(yb)
        sched.step()
        metrics = evaluate(model, arr, pos_va, neg_va, device)
        metrics["loss"] = total / max(1, len(a))
        metrics["epoch"] = epoch
        metrics["sec"] = round(time.time() - t0, 1)
        history.append(metrics)
        print(f"[stem-judge:{args.stem}] ep {epoch} auc {metrics['auc_overall']:.4f} "
              f"loss {metrics['loss']:.3f} ({metrics['sec']:.0f}s)", flush=True)
        if metrics["auc_overall"] > best["auc"]:
            best = {"auc": metrics["auc_overall"], "epoch": epoch}
            out = Path(args.out) if args.out else Path(f"data/stem_judge_{args.stem}.pt")
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "version": 1,
                "model": f"stem_audio_judge_{args.stem}_v1",
                "arch": "two_tower_cnn_shared_emb128",
                "stem": args.stem,
                "cfg": {
                    "mel": idx.get("mel"),
                    "input_norm": "per_bar_mean_subtract/20",
                    "emb_dim": 128,
                    "bars": 4,
                    "pair_order": "first_phrase_then_second_phrase",
                },
                "state_dict": model.state_dict(),
                "val_auc": best["auc"],
                "val_metrics": metrics,
                "epoch": epoch,
                "seed": args.seed,
            }, out)
            if args.log:
                Path(args.log).write_text(json.dumps({"history": history, "best": best}, indent=1))
    print(f"[stem-judge:{args.stem}] best epoch {best['epoch']} auc {best['auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
