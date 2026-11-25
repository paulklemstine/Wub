"""Train the first offline learned conductor from real songs.

Supervision at every 4-bar boundary of ~1,900 analyzed songs:
  state   = last two 4-bar cells of all four stems + section/energy context
  targets = next section, next energy, which stems change, and the next 4-bar
            onset/chroma pattern of every stem

The exported browser artifact is a compact 3-layer MLP. It is a decision
*source*, not a new audio renderer: the browser still schedules only shipped
phrases/bars. See docs/CONDUCTOR.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
PATTERN_DIM = 48
STATE_DIM = 392
TARGET_DIM = 202
HIDDEN = 128
FORMAT = "wub-conductor-v1"


def song_cells(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tats = doc.get("tatums") or []
    n = len(tats) // 16
    if n < 16:
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
        chroma: Dict[str, List[np.ndarray]] = {s: [] for s in STEMS if s != "drums"}
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
        bars.append((union, present, chroma, float(np.mean(louds)) if louds else -60.0,
                     labels[b]))
    return bars


def pattern_vec(union: int, present: int, chroma: List[np.ndarray]) -> np.ndarray:
    v = np.zeros(PATTERN_DIM, dtype=np.float32)
    for step in range(16):
        v[step] = (union >> step) & 1
    count = int(v[:16].sum())
    v[16] = count / 16.0
    v[17] = sum((union >> s) & 1 for s in (0, 4, 8, 12)) / 4.0
    v[18] = sum((union >> s) & 1 for s in (1, 3, 5, 7, 9, 11, 13, 15)) / 8.0
    for q in range(4):
        v[19 + q] = sum((union >> s) & 1 for s in range(q * 4, q * 4 + 4)) / 4.0
    if chroma:
        c = np.mean(np.stack(chroma), axis=0)
        c = c / max(1e-8, c.sum())
        v[23 + int(np.argmax(c))] = 1.0
        v[35:47] = c.astype(np.float32)
    v[47] = 1.0 if present > 0 else 0.0
    return v


def cell(bars, start: int):
    chunk = bars[start:start + 4]
    vecs, unions, presences = [], {}, {}
    for stem in STEMS:
        u = 0
        for b in chunk:
            u |= b[0][stem]
        ch = [c for b in chunk for c in b[2].get(stem, [])]
        vecs.append(pattern_vec(u, sum(b[1][stem] for b in chunk), ch))
        unions[stem] = u
        presences[stem] = sum(b[1][stem] for b in chunk) / 64.0
    energy = float(np.mean([b[3] for b in chunk]))
    section = chunk[0][4]
    return {
        "pattern": np.concatenate(vecs), "union": unions,
        "presence": presences, "energy": energy, "section": section,
    }


def changed(a, b, stem: str) -> float:
    x, y = a["union"][stem], b["union"][stem]
    if not (x or y):
        return 0.0
    sim = (x & y).bit_count() / (x | y).bit_count()
    pres = abs(a["presence"][stem] - b["presence"][stem])
    return 1.0 if (sim < 0.55 or pres > 0.18) else 0.0


def mine(processed: Path, max_tracks: int):
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    track_ids: List[str] = []
    all_energy: List[float] = []
    paths = sorted(processed.glob("*.json"))
    step = max(1, len(paths) // max_tracks) if max_tracks > 0 else 1
    paths = paths[::step][:max_tracks] if max_tracks > 0 else paths
    for path in paths:
        try:
            bars = song_cells(json.loads(path.read_text()))
        except Exception:
            continue
        if not bars or len(bars) < 24:
            continue
        cells = [cell(bars, c * 4) for c in range(len(bars) // 4)]
        all_energy.extend(c["energy"] for c in cells)
        for i in range(2, len(cells)):
            # State ends at cells[i-1]; predict the immediately following cell.
            prev2, prev1, nxt = cells[i - 2], cells[i - 1], cells[i]
            x = list(prev2["pattern"]) + list(prev1["pattern"])
            sec = np.zeros(5, dtype=np.float32)
            if prev1["section"] in SECTIONS:
                sec[SECTIONS.index(prev1["section"])] = 1
            x.extend(sec.tolist())
            x.extend([float(np.clip((prev1["energy"] + 60) / 60, 0, 1)),
                      float(np.clip((prev1["energy"] - prev2["energy"] + 12) / 24, 0, 1)),
                      float((i % 26) / 26)])
            assert len(x) == STATE_DIM, len(x)
            y_next = np.zeros(5, dtype=np.float32)
            if nxt["section"] in SECTIONS:
                y_next[SECTIONS.index(nxt["section"])] = 1
            y = list(y_next)
            y.append(float(np.clip((nxt["energy"] + 60) / 60, 0, 1)))
            y.extend(changed(prev1, nxt, s) for s in STEMS)
            y.extend(nxt["pattern"].tolist())
            assert len(y) == TARGET_DIM, len(y)
            xs.append(np.asarray(x, dtype=np.float32))
            ys.append(np.asarray(y, dtype=np.float32))
            track_ids.append(path.stem)
    return np.stack(xs), np.stack(ys), track_ids, all_energy


class Conductor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, TARGET_DIM)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=18)
    args = ap.parse_args()
    torch.manual_seed(1407)
    np.random.seed(1407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, tracks, energies = mine(args.processed, args.max_tracks)
    unique = sorted(set(tracks))
    rng = np.random.default_rng(1407)
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
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 1024):
            idx = perm[s:s + 1024]
            pred = net(xt[idx])
            sec = torch.nn.functional.cross_entropy(pred[:, :5], yt[idx][:, :5].argmax(dim=1))
            energy = torch.nn.functional.mse_loss(pred[:, 5], yt[idx][:, 5])
            change = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 6:10], yt[idx][:, 6:10])
            pattern = torch.nn.functional.mse_loss(pred[:, 10:], yt[idx][:, 10:])
            loss = sec + energy + change + 0.5 * pattern
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(xv).cpu().numpy()
    truth = yv.cpu().numpy()
    sec_acc = float((pred[:, :5].argmax(1) == truth[:, :5].argmax(1)).mean())
    majority = float(np.bincount(truth[:, :5].argmax(1)).max() / len(truth))
    energy_mae = float(np.mean(np.abs(pred[:, 5] - truth[:, 5])))
    energy_base = float(np.mean(np.abs(truth[:, 5].mean() - truth[:, 5])))
    auc = {}
    for i, stem in enumerate(STEMS):
        col = truth[:, 6 + i]
        if 0 < col.sum() < len(col):
            auc[stem] = round(float(roc_auc_score(col, pred[:, 6 + i])), 4)
    pattern_mse = float(np.mean((pred[:, 10:] - truth[:, 10:]) ** 2))
    pattern_base = float(np.mean((truth[:, 10:].mean(axis=0) - truth[:, 10:]) ** 2))

    def layer(mod: nn.Linear):
        return {
            "w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
            "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()],
        }

    doc = {
        "format": FORMAT, "version": 1,
        "state_dim": STATE_DIM, "target_dim": TARGET_DIM, "hidden": HIDDEN,
        "sections": list(SECTIONS), "stems": list(STEMS),
        "pattern_dim": PATTERN_DIM,
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [layer(net.fc1), layer(net.fc2), layer(net.fc3)],
        "authority": 1.0,
        "report": {
            "tracks": len(unique), "train_examples": int(tr.sum()), "val_examples": int(val.sum()),
            "train_tracks": len(unique) - len(val_tracks), "val_tracks": len(val_tracks),
            "device": str(device), "epochs": args.epochs,
            "section_accuracy": round(sec_acc, 4), "section_majority": round(majority, 4),
            "energy_mae": round(energy_mae, 4), "energy_baseline_mae": round(energy_base, 4),
            "change_auc": auc,
            "pattern_mse": round(pattern_mse, 5), "pattern_baseline_mse": round(pattern_base, 5),
            "energy_scale": {"p5_db": round(float(np.percentile(energies, 5)), 3),
                             "p95_db": round(float(np.percentile(energies, 95)), 3)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps(doc["report"], indent=2))


if __name__ == "__main__":
    main()
