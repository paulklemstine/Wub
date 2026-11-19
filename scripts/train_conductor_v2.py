"""Train conductor v2: 6-cell history, larger MLP, balanced section loss.

Same real-song corpus as v1, but the state sees six consecutive 4-bar cells of
all four stems instead of two, plus per-cell section/energy and harmonic
alignment. The goal is a materially better section/change predictor before we
let the model drive the set with authority 1.0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
PITCHED = ("bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
PATTERN_DIM = 48
HISTORY = 6
PER_CELL = PATTERN_DIM * 4 + 5 + 1  # 192 patterns + 5 section + 1 energy = 198
CONTEXT = 5 + 1 + 1 + 1 + 6  # section, energy, delta, phase, 6 harmony features
STATE_DIM = HISTORY * PER_CELL + CONTEXT  # 1188 + 14 = 1202
TARGET_DIM = 202
HIDDEN = 256
FORMAT = "wub-conductor-v2"


def song_cells(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tats = doc.get("tatums") or []
    n = len(tats) // 16
    if n < 32:
        return None
    feats = bar_features(segs, tats)
    spans = segment_sections(feats)
    labels = [s.label for s in spans for _ in range(s.n_bars)]
    labels = (labels + ["?"] * n)[:n]
    by_id = {str(s.get("id")): s for s in segs if s.get("id")}
    prev = {s: None for s in STEMS}
    bars = []
    for b in range(n):
        union = {s: 0 for s in STEMS}
        present = {s: 0 for s in STEMS}
        chroma: Dict[str, List[np.ndarray]] = {s: [] for s in PITCHED}
        louds = []
        for step in range(16):
            ss = (tats[b * 16 + step].get("stem_segments") or {})
            for stem in STEMS:
                ids = ss.get(stem, []) or []
                cur = ids[-1] if ids else None
                if cur is not None:
                    present[stem] += 1
                on = (step == 0 and cur is not None) or (cur is not None and cur != prev[stem])
                if on:
                    union[stem] |= 1 << step
                    seg = by_id.get(str(cur))
                    if seg is not None:
                        louds.append(float(seg.get("loudness_max", -20.0)))
                        if stem in chroma:
                            p = np.asarray(seg.get("pitches") or [0.0] * 12, dtype=float)[:12]
                            if p.sum() > 1e-8:
                                chroma[stem].append(p / p.sum())
                prev[stem] = cur
        bars.append((union, present, chroma, float(np.mean(louds)) if louds else -60.0, labels[b]))
    return bars


def pattern_vec(union: int, present: int, chroma: List[np.ndarray]):
    v = np.zeros(PATTERN_DIM, dtype=np.float32)
    for step in range(16):
        v[step] = (union >> step) & 1
    count = int(v[:16].sum())
    v[16] = count / 16.0
    v[17] = sum((union >> s) & 1 for s in (0, 4, 8, 12)) / 4.0
    v[18] = sum((union >> s) & 1 for s in (1, 3, 5, 7, 9, 11, 13, 15)) / 8.0
    for q in range(4):
        v[19 + q] = sum((union >> s) & 1 for s in range(q * 4, q * 4 + 4)) / 4.0
    c = None
    if chroma:
        c = np.mean(np.stack(chroma), axis=0)
        c = c / max(1e-8, c.sum())
        v[23 + int(np.argmax(c))] = 1.0
        v[35:47] = c.astype(np.float32)
    v[47] = 1.0 if present > 0 else 0.0
    return v, c


def cell(bars, start: int):
    chunk = bars[start:start + 4]
    vecs, chromas, unions, presences = [], {}, {}, {}
    for stem in STEMS:
        u = 0
        for b in chunk:
            u |= b[0][stem]
        all_chroma = [cc for b in chunk for cc in b[2].get(stem, [])]
        v, c = pattern_vec(u, sum(b[1][stem] for b in chunk), all_chroma)
        vecs.append(v)
        chromas[stem] = c
        unions[stem] = u
        presences[stem] = sum(b[1][stem] for b in chunk) / 64.0
    return {
        "pattern": np.concatenate(vecs), "chroma": chromas,
        "union": unions, "presence": presences,
        "energy": float(np.mean([b[3] for b in chunk])),
        "section": chunk[0][4],
    }


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def changed(a, b, stem):
    x, y = a["union"][stem], b["union"][stem]
    if not (x or y):
        return 0.0
    sim = (x & y).bit_count() / (x | y).bit_count()
    pres = abs(a["presence"][stem] - b["presence"][stem])
    return 1.0 if (sim < 0.55 or pres > 0.18) else 0.0


def onehot(section):
    v = np.zeros(5, dtype=np.float32)
    if section in SECTIONS:
        v[SECTIONS.index(section)] = 1
    return v


def state_for(cells, i):
    s: List[float] = []
    for h in range(HISTORY):
        c = cells[i - HISTORY + h]
        s.extend(c["pattern"].tolist())
        s.extend(onehot(c["section"]).tolist())
        s.append(float(np.clip((c["energy"] + 60) / 60, 0, 1)))
    cur = cells[i - 1]
    s.extend(onehot(cur["section"]).tolist())
    s.append(float(np.clip((cur["energy"] + 60) / 60, 0, 1)))
    s.append(float(np.clip((cur["energy"] - cells[i - 2]["energy"] + 12) / 24, 0, 1)))
    s.append(float((i % 26) / 26))
    ch = [cur["chroma"][stem] for stem in PITCHED]
    for a in range(3):
        for b in range(a + 1, 3):
            s.append(cosine(ch[a], ch[b]))
    roots = [int(np.argmax(c)) for c in ch if c is not None]
    for a in range(len(roots)):
        for b in range(a + 1, len(roots)):
            d = abs(roots[a] - roots[b]) % 12
            s.append(1.0 - min(d, 12 - d) / 6.0)
    while len(s) < STATE_DIM:
        s.append(0.0)
    assert len(s) == STATE_DIM, len(s)
    return np.asarray(s[:STATE_DIM], dtype=np.float32)


def mine(processed: Path, max_tracks: int):
    xs, ys, tracks, energies = [], [], [], []
    paths = sorted(processed.glob("*.json"))
    step = max(1, len(paths) // max_tracks) if max_tracks > 0 else 1
    paths = paths[::step][:max_tracks] if max_tracks > 0 else paths
    for path in paths:
        try:
            bars = song_cells(json.loads(path.read_text()))
        except Exception:
            continue
        if not bars or len(bars) < 40:
            continue
        cells = [cell(bars, c * 4) for c in range(len(bars) // 4)]
        energies.extend(c["energy"] for c in cells)
        for i in range(HISTORY, len(cells)):
            # The state ends at cells[i-1]; the immediate next cell is the
            # correct 4-bar target (the v1 trainer accidentally used i+1).
            nxt = cells[i]
            y = list(onehot(nxt["section"]))
            y.append(float(np.clip((nxt["energy"] + 60) / 60, 0, 1)))
            y.extend(changed(cells[i - 1], nxt, stem) for stem in STEMS)
            y.extend(nxt["pattern"].tolist())
            assert len(y) == TARGET_DIM
            xs.append(state_for(cells, i))
            ys.append(np.asarray(y, dtype=np.float32))
            tracks.append(path.stem)
    return np.stack(xs), np.stack(ys), tracks, energies


class Conductor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, TARGET_DIM)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.0012)
    ap.add_argument("--no-balance", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(2311)
    np.random.seed(2311)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, tracks, energies = mine(args.processed, args.max_tracks)
    unique = sorted(set(tracks))
    rng = np.random.default_rng(2311)
    rng.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    val = np.asarray([t in val_tracks for t in tracks], dtype=bool)
    tr = ~val
    mean = x[tr].mean(axis=0).astype(np.float32)
    std = x[tr].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    xt = torch.tensor((x[tr] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
    xv = torch.tensor((x[val] - mean) / std, dtype=torch.float32, device=device)
    yv = torch.tensor(y[val], dtype=torch.float32, device=device)
    net = Conductor().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # Class-balanced section loss: rare breakdowns/drops must be learned, not
    # drowned out by the dominant GROOVE self-transition.
    counts = np.bincount(y[tr][:, :5].argmax(axis=1), minlength=5).astype(np.float32)
    weights = np.clip(counts.mean() / np.maximum(1, counts), 0.5, 4.0)
    if args.no_balance:
        weights[:] = 1.0
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 512):
            idx = perm[s:s + 512]
            pred = net(xt[idx])
            sec = torch.nn.functional.cross_entropy(pred[:, :5], yt[idx][:, :5].argmax(1), weight=weights)
            energy = torch.nn.functional.mse_loss(pred[:, 5], yt[idx][:, 5])
            change = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 6:10], yt[idx][:, 6:10])
            pattern = torch.nn.functional.mse_loss(pred[:, 10:], yt[idx][:, 10:])
            loss = 2.0 * sec + 1.0 * energy + 1.5 * change + 0.5 * pattern
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}/{args.epochs}", flush=True)

    net.eval()
    with torch.no_grad():
        pred = net(xv).cpu().numpy()
    truth = yv.cpu().numpy()
    sec_true = truth[:, :5].argmax(1)
    sec_pred = pred[:, :5].argmax(1)
    acc = float((sec_pred == sec_true).mean())
    majority = float(np.bincount(sec_true).max() / len(sec_true))
    f1 = float(f1_score(sec_true, sec_pred, average="macro"))
    emae = float(np.mean(np.abs(pred[:, 5] - truth[:, 5])))
    ebase = float(np.mean(np.abs(truth[:, 5].mean() - truth[:, 5])))
    auc = {}
    for i, stem in enumerate(STEMS):
        col = truth[:, 6 + i]
        if 0 < col.sum() < len(col):
            auc[stem] = round(float(roc_auc_score(col, pred[:, 6 + i])), 4)
    pmse = float(np.mean((pred[:, 10:] - truth[:, 10:]) ** 2))
    pbase = float(np.mean((truth[:, 10:].mean(axis=0) - truth[:, 10:]) ** 2))

    def layer(mod):
        return {"w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
                "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()]}

    doc = {
        "format": FORMAT, "version": 2,
        "state_dim": STATE_DIM, "target_dim": TARGET_DIM, "hidden": HIDDEN,
        "history_cells": HISTORY, "sections": list(SECTIONS), "stems": list(STEMS),
        "pattern_dim": PATTERN_DIM,
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [layer(net.fc1), layer(net.fc2), layer(net.fc3)],
        "authority": 1.0,
        "report": {
            "tracks": len(unique), "train_examples": int(tr.sum()), "val_examples": int(val.sum()),
            "train_tracks": len(unique) - len(val_tracks), "val_tracks": len(val_tracks),
            "device": str(device), "epochs": args.epochs,
            "section_accuracy": round(acc, 4), "section_majority": round(majority, 4),
            "section_macro_f1": round(f1, 4),
            "energy_mae": round(emae, 4), "energy_baseline_mae": round(ebase, 4),
            "change_auc": auc,
            "pattern_mse": round(pmse, 5), "pattern_baseline_mse": round(pbase, 5),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps(doc["report"], indent=2))


if __name__ == "__main__":
    main()
