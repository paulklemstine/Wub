"""Train a phrase-space conductor on real song chronology mapped to the bank.

Unlike conductor v2 (raw song cells), this model sees exactly what the browser
sees: the current phrase's four stem cell-vectors, its embedding, section and
tempo/key. It predicts the embedding of the phrase that actually follows in the
real song, plus its section and energy. Browser inference scores candidate
phrases by embedding similarity, so every output maps to a shipped phrase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
PATTERN_DIM = 48
EMBED_DIM = 64
STATE_DIM = EMBED_DIM + 4 * PATTERN_DIM + 5 + 1 + 2  # 264 (one phrase embedding + four stem patterns)
TARGET_DIM = EMBED_DIM + 5 + 1
HIDDEN = 128
FORMAT = "wub-phrase-conductor-v1"


def load_bank(kits_dir: Path):
    manifest = json.loads((kits_dir / "kit_manifest.json").read_text())
    cell = json.loads((kits_dir / "cell_index.json").read_text())
    chroma_doc = json.loads((kits_dir / "cell_chroma.json").read_text())
    chroma = chroma_doc.get("phrases", {})
    by_track: Dict[str, Dict[str, Any]] = {}
    info: Dict[str, Dict[str, Any]] = {}
    for kit in manifest.get("kits", []):
        tid = kit["track_id"]
        phrases = sorted(kit.get("phrases", []), key=lambda p: p.get("start_bar", 0))
        by_track[tid] = {
            "phrases": phrases,
            "bar_to_id": {int(p["start_bar"]): p["id"] for p in phrases},
        }
        for p in phrases:
            info[p["id"]] = {
                "embedding": np.asarray(p.get("embedding") or [0.0] * EMBED_DIM, dtype=np.float32),
                "section": str(p.get("section") or "GROOVE_A").upper(),
                "energy": float(p.get("energy") or 0.5),
                "bpm": float(kit.get("bpm") or 120),
                "key": str(kit.get("key") or ""),
                "cells": cell.get("phrases", {}).get(p["id"], {}),
                "chroma": chroma.get(p["id"], {}),
                "track_id": tid,
            }
    return by_track, info


def pattern_vec(meta: Dict[str, Any], chroma_row: List[float]) -> np.ndarray:
    v = np.zeros(PATTERN_DIM, dtype=np.float32)
    o = meta.get("o") or [0, 0, 0, 0]
    mask = int(o[0]) if o else 0
    for step in range(16):
        v[step] = (mask >> step) & 1
    count = int(v[:16].sum())
    v[16] = count / 16.0
    v[17] = sum((mask >> s) & 1 for s in (0, 4, 8, 12)) / 4.0
    v[18] = sum((mask >> s) & 1 for s in (1, 3, 5, 7, 9, 11, 13, 15)) / 8.0
    for q in range(4):
        v[19 + q] = sum((mask >> s) & 1 for s in range(q * 4, q * 4 + 4)) / 4.0
    r = (meta.get("r") or [-1, -1, -1, -1])[0]
    if isinstance(r, (int, float)) and r >= 0:
        v[23 + int(r) % 12] = 1.0
    c = np.asarray(chroma_row or [0.0] * 12, dtype=np.float32)
    if c.sum() > 1e-6:
        v[35:47] = c / c.sum()
    v[47] = 1.0 if count > 0 else 0.0
    return v


def key_root(s: str) -> int:
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
           "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    head = s.split()[0] if s else "C"
    return pcs.get(head.replace("m", ""), 0)


def state_vec(pid: str, section: str, info: Dict[str, Dict[str, Any]]) -> np.ndarray:
    meta = info[pid]
    v: List[float] = list(meta["embedding"])
    for stem in STEMS:
        cell_meta = (meta["cells"].get(stem) or {})
        row = (meta["chroma"].get(stem) or [[]])[0]
        v.extend(pattern_vec(cell_meta, row).tolist())
    one = np.zeros(5, dtype=np.float32)
    if section in SECTIONS:
        one[SECTIONS.index(section)] = 1
    v.extend(one.tolist())
    v.append(float(np.clip(meta["energy"], 0, 1)))
    v.append(min(1.0, meta["bpm"] / 200.0))
    d = abs(key_root(meta["key"]) - 0) % 12
    v.append(min(d, 12 - d) / 6.0)
    assert len(v) == STATE_DIM, len(v)
    return np.asarray(v, dtype=np.float32)


def target_vec(pid: str, section: str, info: Dict[str, Dict[str, Any]]) -> np.ndarray:
    meta = info[pid]
    v = list(meta["embedding"])
    one = np.zeros(5, dtype=np.float32)
    if section in SECTIONS:
        one[SECTIONS.index(section)] = 1
    v.extend(one.tolist())
    v.append(float(np.clip(meta["energy"], 0, 1)))
    assert len(v) == TARGET_DIM
    return np.asarray(v, dtype=np.float32)


def mine(kits_dir: Path, processed: Path, info, by_track):
    xs, ys, tracks, pairs = [], [], [], []
    for tid, entry in by_track.items():
        path = processed / f"{tid}.json"
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
            segs, tats = doc.get("segments") or [], doc.get("tatums") or []
            n = len(tats) // 16
            if n < 8:
                continue
            feats = bar_features(segs, tats)
            spans = segment_sections(feats)
            labels = [s.label for s in spans for _ in range(s.n_bars)]
            labels = (labels + ["?"] * n)[:n]
        except Exception:
            continue
        for p in entry["phrases"]:
            b = int(p["start_bar"])
            section = labels[b] if b < len(labels) else p["section"]
            section = section if section in SECTIONS else "GROOVE_A"
            # Include a few real later windows, discounted by distance.
            for dist in (1, 2, 3, 4):
                nid = entry["bar_to_id"].get(b + 4 * dist)
                if not nid or nid == p["id"]:
                    continue
                nb = b + 4 * dist
                next_section = labels[nb] if nb < len(labels) else info[nid]["section"]
                next_section = next_section if next_section in SECTIONS else "GROOVE_A"
                xs.append(state_vec(p["id"], section, info))
                ys.append(target_vec(nid, next_section, info))
                tracks.append(tid)
                pairs.append((p["id"], nid, dist))
    return np.stack(xs), np.stack(ys), tracks, pairs


class PhraseConductor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, TARGET_DIM)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(617)
    np.random.seed(617)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_track, info = load_bank(args.kits_dir)
    x, y, tracks, pairs = mine(args.kits_dir, args.processed, info, by_track)
    unique = sorted(set(tracks))
    rng = np.random.default_rng(617)
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
    net = PhraseConductor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(args.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 512):
            idx = perm[s:s + 512]
            pred = net(xt[idx])
            emb = torch.nn.functional.mse_loss(pred[:, :EMBED_DIM], yt[idx][:, :EMBED_DIM])
            sec = torch.nn.functional.cross_entropy(pred[:, 64:69], yt[idx][:, 64:69].argmax(1))
            energy = torch.nn.functional.mse_loss(pred[:, 69], yt[idx][:, 69])
            loss = emb + 0.5 * sec + 0.5 * energy
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(xv).cpu().numpy()
    truth = yv.cpu().numpy()
    def cosine_row(a, b):
        d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        mask = d > 1e-8
        sim = np.zeros(len(a), dtype=np.float32)
        sim[mask] = np.sum(a[mask] * b[mask], axis=1) / d[mask]
        return sim
    cos = cosine_row(pred[:, :EMBED_DIM], truth[:, :EMBED_DIM])
    base = cosine_row(np.tile(truth[:, :EMBED_DIM].mean(axis=0), (len(truth), 1)), truth[:, :EMBED_DIM])
    sec_acc = float((pred[:, 64:69].argmax(1) == truth[:, 64:69].argmax(1)).mean())
    majority = float(np.bincount(truth[:, 64:69].argmax(1)).max() / len(truth))
    energy_mae = float(np.mean(np.abs(pred[:, 69] - truth[:, 69])))
    energy_base = float(np.mean(np.abs(truth[:, 69].mean() - truth[:, 69])))

    def layer(mod):
        return {"w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
                "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()]}

    doc = {
        "format": FORMAT, "version": 1,
        "state_dim": STATE_DIM, "target_dim": TARGET_DIM, "hidden": HIDDEN,
        "embed_dim": EMBED_DIM, "pattern_dim": PATTERN_DIM,
        "sections": list(SECTIONS), "stems": list(STEMS),
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [layer(net.fc1), layer(net.fc2), layer(net.fc3)],
        "report": {
            "source": "real phrase chronology mapped to shipped phrase ids",
            "phrases": len(info), "pairs": len(pairs),
            "tracks": len(unique), "train_tracks": len(unique) - len(val_tracks),
            "val_tracks": len(val_tracks),
            "train_examples": int(tr.sum()), "val_examples": int(val.sum()),
            "epochs": args.epochs, "device": str(device),
            "val_embedding_cosine": round(float(cos.mean()), 4),
            "baseline_cosine": round(float(base.mean()), 4),
            "section_accuracy": round(sec_acc, 4), "section_majority": round(majority, 4),
            "energy_mae": round(energy_mae, 4), "energy_baseline_mae": round(energy_base, 4),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps(doc["report"], indent=2))


if __name__ == "__main__":
    main()
