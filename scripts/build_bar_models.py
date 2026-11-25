"""Mine next-bar pattern models and quartet harmony thresholds from real songs.

Run on the Wub training box (WSL on 192.168.1.136) with PYTHONPATH pointing at
a checkout containing wub_core/section_arcs.py. Produces two browser sidecars:

  pattern_bar_model.json   per-stem next-4/16-bar pattern heads (drums/bass/other)
  harmony_continuity.json  real-song per-bar quartet chroma alignment thresholds

All examples come from data/processed/*.json; splits are track-disjoint.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
PITCHED = ("bass", "other", "vocals")
PATTERN_STEMS = ("drums", "bass", "other")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
DIM = 48
CTX = 7
INPUT_DIM = DIM + CTX
HIDDEN = 64
ON = (0, 4, 8, 12)


def _loud(seg) -> float:
    return float(seg.get("loudness_max", -20.0)) if seg else -20.0


def _chroma(seg) -> Optional[np.ndarray]:
    if not seg:
        return None
    p = np.asarray(seg.get("pitches") or [0.0] * 12, dtype=np.float64)[:12]
    if p.sum() <= 1e-8:
        return None
    return p / p.sum()


def _triad(chroma: Optional[np.ndarray]):
    """Root + inferred major/minor/sus third/fifth, matching browser r/q."""
    if chroma is None:
        return None, "mono"
    root = int(np.argmax(chroma))
    third = (root + 4) % 12 if chroma[(root + 4) % 12] >= chroma[(root + 3) % 12] else (root + 3) % 12
    fifth = (root + 7) % 12
    if chroma[(root + 5) % 12] > max(chroma[third], chroma[fifth]) + 0.05:
        third, quality = (root + 5) % 12, "sus"
    else:
        quality = "major" if third == (root + 4) % 12 else "minor"
    vec = np.zeros(12, dtype=np.float64)
    vec[root] = 1.0
    vec[third] = 0.72
    vec[fifth] = 0.72
    return vec / np.linalg.norm(vec), quality


def _cos(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    if d < 1e-9:
        return None
    return float(np.clip(np.dot(a, b) / d, -1.0, 1.0))


def pattern_vec(mask: int, chroma: Optional[np.ndarray], presence: int, energy: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for step in range(16):
        v[step] = (mask >> step) & 1
    count = int(v[:16].sum())
    v[16] = count / 16.0
    v[17] = sum((mask >> s) & 1 for s in ON) / 4.0
    v[18] = sum((mask >> s) & 1 for s in range(1, 16, 2)) / 8.0
    for q in range(4):
        v[19 + q] = sum((mask >> s) & 1 for s in range(q * 4, q * 4 + 4)) / 4.0
    if chroma is not None:
        root = int(np.argmax(chroma))
        v[23 + root] = 1.0
        v[35:47] = chroma.astype(np.float32)
    v[47] = 1.0 if presence > 0 else 0.0
    return v


def song_bars(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tatums = doc.get("tatums") or []
    seg_by_id = {str(s.get("id")): s for s in segs if s.get("id")}
    n = len(tatums) // 16
    if n < 12:
        return None
    feats = bar_features(segs, tatums)
    sections = segment_sections(feats)
    labels = [s.label for s in sections for _ in range(s.n_bars)]
    labels = (labels + ["?"] * n)[:n]

    prev_id = {s: None for s in STEMS}
    bars = []
    for b in range(n):
        bar = {"mask": {}, "chroma": {}, "triad": {}, "presence": {}, "energy": [], "section": labels[b]}
        for step in range(16):
            ss = (tatums[b * 16 + step].get("stem_segments") or {})
            for stem in STEMS:
                ids = ss.get(stem, []) or []
                cur = ids[-1] if ids else None
                if cur is not None:
                    bar["presence"][stem] = bar["presence"].get(stem, 0) + 1
                onset = (step == 0 and cur is not None) or (cur is not None and cur != prev_id[stem])
                if onset:
                    bar["mask"][stem] = bar["mask"].get(stem, 0) | (1 << step)
                    seg = seg_by_id.get(str(cur))
                    if seg is not None:
                        bar["energy"].append(_loud(seg))
                        if stem in PITCHED:
                            ch = _chroma(seg)
                            if ch is not None:
                                bar["chroma"].setdefault(stem, []).append(ch)
                prev_id[stem] = cur
        for stem in STEMS:
            bar["mask"][stem] = bar["mask"].get(stem, 0)
            bar["presence"][stem] = bar["presence"].get(stem, 0)
        bar["energy"] = float(np.mean(bar["energy"])) if bar["energy"] else -60.0
        for stem in PITCHED:
            chs = bar["chroma"].get(stem, [])
            mean = np.mean(np.stack(chs), axis=0) if chs else None
            bar["chroma"][stem] = mean
            bar["triad"][stem] = _triad(mean)[0]
        bars.append(bar)
    return bars


def context_vec(section: str, energy: float, delta: float) -> np.ndarray:
    c = np.zeros(CTX, dtype=np.float32)
    if section in SECTIONS:
        c[SECTIONS.index(section)] = 1.0
    c[5] = float(np.clip((delta + 12.0) / 24.0, 0.0, 1.0))
    c[6] = float(np.clip((energy + 60.0) / 60.0, 0.0, 1.0))
    return c


def mine(processed: Path, max_tracks: int):
    xs = {s: [] for s in PATTERN_STEMS}
    ys = {s: [] for s in PATTERN_STEMS}
    tracks = {s: [] for s in PATTERN_STEMS}
    harmony_min_raw: List[float] = []
    harmony_min_triad: List[float] = []
    harmony_root_ok: List[float] = []
    boundary_cos: List[float] = []
    bar_count = 0
    paths = sorted(processed.glob("*.json"))
    step = max(1, len(paths) // max_tracks) if max_tracks > 0 else 1
    paths = paths[::step][:max_tracks] if max_tracks > 0 else paths

    for path in paths:
        try:
            doc = json.loads(path.read_text())
            bars = song_bars(doc)
        except Exception:
            continue
        if not bars:
            continue
        tid = str(doc.get("track_id") or path.stem)
        for i, bar in enumerate(bars):
            bar_count += 1
            # Quartet chroma alignment from pitched stems in the same bar.
            for key, out in (("chroma", harmony_min_raw), ("triad", harmony_min_triad)):
                vecs = [bar[key][s] for s in PITCHED if bar[key][s] is not None]
                if len(vecs) >= 2:
                    pair = [_cos(vecs[j], vecs[k]) for j in range(len(vecs)) for k in range(j + 1, len(vecs))]
                    pair = [p for p in pair if p is not None]
                    if pair:
                        out.append(min(pair))
            roots = [int(np.argmax(bar["chroma"][s])) for s in PITCHED if bar["chroma"][s] is not None]
            if len(roots) >= 2:
                distances = [min((roots[j] - roots[k]) % 12, (roots[k] - roots[j]) % 12)
                             for j in range(len(roots)) for k in range(j + 1, len(roots))]
                harmony_root_ok.append(1.0 if max(distances) <= 1 else 0.0)
            # Harmonic continuity at the 4-bar cell boundary (bar 3 -> bar 0).
            if i % 4 == 0 and i > 0:
                prev = bars[i - 1]
                for s in PITCHED:
                    cc = _cos(prev["chroma"][s], bar["chroma"][s])
                    if cc is not None:
                        boundary_cos.append(cc)

            if i == 0:
                continue
            prev = bars[i - 1]
            delta = bar["energy"] - prev["energy"]
            ctx = context_vec(bar["section"], bar["energy"], delta)
            for s in PATTERN_STEMS:
                if prev["presence"][s] <= 0 or bar["presence"][s] <= 0:
                    continue
                x = np.concatenate([
                    pattern_vec(prev["mask"][s], prev["chroma"].get(s), prev["presence"][s], prev["energy"]),
                    ctx,
                ])
                y = pattern_vec(bar["mask"][s], bar["chroma"].get(s), bar["presence"][s], bar["energy"])
                xs[s].append(x)
                ys[s].append(y)
                tracks[s].append(tid)
    return xs, ys, tracks, bar_count, {
        "harmony_min_raw": harmony_min_raw,
        "harmony_min_triad": harmony_min_triad,
        "harmony_root_ok": harmony_root_ok,
        "boundary_cos": boundary_cos,
    }


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, DIM)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def train_stem(x: np.ndarray, y: np.ndarray, track_ids: List[str], seed: int, device):
    unique = sorted(set(track_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    val_mask = np.asarray([t in val_tracks for t in track_ids], dtype=bool)
    train_mask = ~val_mask
    xt = torch.tensor(x[train_mask], dtype=torch.float32, device=device)
    yt = torch.tensor(y[train_mask], dtype=torch.float32, device=device)
    xv = torch.tensor(x[val_mask], dtype=torch.float32, device=device)
    yv = torch.tensor(y[val_mask], dtype=torch.float32, device=device)
    mean = x[train_mask].mean(axis=0).astype(np.float32)
    std = x[train_mask].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    m = torch.tensor(mean, device=device)
    s = torch.tensor(std, device=device)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    baseline = yt.mean(dim=0)
    bs = 1024
    for epoch in range(24):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for start in range(0, perm.numel(), bs):
            idx = perm[start:start + bs]
            loss = torch.mean((net((xt[idx] - m) / s) - yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        val_loss = float(torch.mean((net((xv - m) / s) - yv) ** 2))
        base_loss = float(torch.mean((baseline - yv) ** 2))
    return {
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "w1": [[round(float(v), 5) for v in row] for row in net.fc1.weight.detach().cpu().numpy()],
        "b1": [round(float(v), 5) for v in net.fc1.bias.detach().cpu().numpy()],
        "w2": [[round(float(v), 5) for v in row] for row in net.fc2.weight.detach().cpu().numpy()],
        "b2": [round(float(v), 5) for v in net.fc2.bias.detach().cpu().numpy()],
        "train_examples": int(train_mask.sum()),
        "val_examples": int(val_mask.sum()),
        "val_mse": round(val_loss, 5),
        "baseline_mse": round(base_loss, 5),
        "train_tracks": len(unique) - len(val_tracks),
        "val_tracks": len(val_tracks),
    }


def pct(values: List[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values), q)), 4) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(707)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys, tracks, bars, harm = mine(args.processed, args.max_tracks)

    stems = {}
    for i, stem in enumerate(PATTERN_STEMS):
        if len(xs[stem]) < 1000:
            continue
        x = np.stack(xs[stem]).astype(np.float32)
        y = np.stack(ys[stem]).astype(np.float32)
        stems[stem] = train_stem(x, y, tracks[stem], 707 + i, device)
        print(stem, "examples", len(x), "val_mse", stems[stem]["val_mse"],
              "baseline", stems[stem]["baseline_mse"], flush=True)

    pattern_doc = {
        "format": "wub-pattern-bars-v1",
        "version": 1,
        "input_dim": INPUT_DIM,
        "output_dim": DIM,
        "hidden": HIDDEN,
        "layout": "onset16,density,downbeat,offbeat,quarters4,root1hot12,chroma12,activity | section5,energy_delta,energy",
        "stems": stems,
        "report": {
            "source": "data/processed/*.json (real 4-stem songs)",
            "bars": bars,
            "split": "track-disjoint 90/10 by track_id",
            "train_device": str(device),
            "target": "next bar pattern: onset grid, density, harmonic chroma, activity",
            "use": "rank only; never retimes, repitches or replaces a scheduled bar",
        },
    }
    harmony_doc = {
        "format": "wub-harmony-continuity-v1",
        "version": 1,
        "bars": bars,
        "pair_chroma_cos": {
            "raw": {"p10": pct(harm["harmony_min_raw"], 10), "p50": pct(harm["harmony_min_raw"], 50)},
            "triad": {"p10": pct(harm["harmony_min_triad"], 10), "p50": pct(harm["harmony_min_triad"], 50)},
        },
        "boundary_chroma_cos": {
            "p10": pct(harm["boundary_cos"], 10), "p50": pct(harm["boundary_cos"], 50),
        },
        "root_agreement_rate": round(float(np.mean(harm["harmony_root_ok"])), 4) if harm["harmony_root_ok"] else 0.0,
        "policy": {
            "min_bar_pair_cos": pct(harm["harmony_min_raw"], 10),
            "min_boundary_cos": pct(harm["boundary_cos"], 10),
            "max_clash_penalty": 0.22,
        },
        "report": {
            "source": "data/processed/*.json (real 4-stem songs)",
            "pair_rule": "minimum cosine between any two pitched-stem chroma vectors in one bar",
            "triad": "root + inferred third/fifth (major/minor/sus), same representation as browser cell r/q",
            "label_provenance": "acoustic chroma, not human annotation",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pattern_bar_model.json").write_text(json.dumps(pattern_doc, separators=(",", ":")) + "\n")
    (args.out_dir / "harmony_continuity.json").write_text(json.dumps(harmony_doc, indent=2) + "\n")
    print(json.dumps({
        "bars": bars,
        "harmony": harmony_doc["pair_chroma_cos"],
        "boundary": harmony_doc["boundary_chroma_cos"],
        "root_agreement": harmony_doc["root_agreement_rate"],
        "policy": harmony_doc["policy"],
    }, indent=2))


if __name__ == "__main__":
    main()
