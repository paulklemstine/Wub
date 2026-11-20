"""Learn phrase embeddings from real-song phrase transitions.

The shipped phrase embedding comes from a taste model trained on analysis
windows. This model instead learns a 64-d phrase representation from the
shipped bank's per-bar descriptors, with a contrastive objective: phrases that
really follow one another in the training songs should score higher than
same-section phrases from other songs.

The exported embeddings replace the 64-d `embedding` field when retraining the
phrase ranker; the browser artifact format is unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from audio_critic_features import load_bank
from train_bank_transition import SECTIONS, key_distance
from fine_tune_policy import mine

STEMS = ("drums", "bass", "other", "vocals")
STYLES = ("breaks", "cyber", "house", "wave")
FEAT_DIM = 144
EMBED_DIM = 64
HIDDEN = 128


def phrase_features(pid: str, bank: Dict[str, Any]) -> np.ndarray:
    p = bank[pid]
    v: List[float] = []
    for stem in STEMS:
        meta = p["cells"].get(stem, {})
        chroma = np.asarray((p["chroma"].get(stem) or [[]])[0], dtype=np.float32)
        if chroma.shape != (12,):
            chroma = np.zeros(12, dtype=np.float32)
        purity = np.asarray(meta.get("p") or [0.5] * 4, dtype=np.float32)
        ess = np.asarray(meta.get("es") or purity, dtype=np.float32)
        fid = float(meta.get("fid") or 0.65)
        roots = np.zeros(12, dtype=np.float32)
        for r in (meta.get("r") or []):
            if isinstance(r, (int, float)) and r >= 0:
                roots[int(r) % 12] += 0.25
        v.extend([
            float(purity.mean()), float(ess.mean()), fid,
            float(np.mean((meta.get("o") or [0])[:4]) and 0.5),
        ])
        v.extend(roots.tolist())
        v.extend(chroma.tolist())
        v.extend([float(np.mean(meta.get("hr") or [0.5])), float(np.mean(meta.get("tc") or [0.5]))])
    section = p["section"] if p["section"] in SECTIONS else "GROOVE_A"
    for s in SECTIONS:
        v.append(1.0 if s == section else 0.0)
    for s in STYLES:
        v.append(1.0 if p.get("style", "").lower().startswith(s) else 0.0)
    v.extend([
        min(1.0, p["bpm"] / 200.0),
        float(p.get("energy", 0.5)),
        (key_root(p["key"]) / 11.0),
        float(p.get("is_intro", False)), float(p.get("is_outro", False)),
    ])
    while len(v) < FEAT_DIM:
        v.append(0.0)
    return np.asarray(v[:FEAT_DIM], dtype=np.float32)


def key_root(key: str) -> int:
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
           "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    return pcs.get((key or "C").split()[0].replace("m", ""), 0)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, EMBED_DIM),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(1905)
    np.random.seed(1905)
    random.seed(1905)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bank = load_bank(args.kits_dir)
    feats = {pid: phrase_features(pid, bank) for pid in bank}
    pos, _ = mine(args.kits_dir, args.processed)
    by_section: Dict[str, List[str]] = defaultdict(list)
    for pid, meta in bank.items():
        by_section[meta["section"] if meta["section"] in SECTIONS else "GROOVE_A"].append(pid)

    pairs: List[Tuple[str, str]] = []
    for cur, nxt, section, _, _ in pos:
        pool = by_section.get(section) or list(bank)
        neg = random.choice(pool)
        if neg not in (cur, nxt):
            pairs.append((nxt, neg))  # positive phrase vs same-section negative

    model = Encoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    F = torch.tensor(np.stack([feats[pid] for pid in bank]), dtype=torch.float32, device=device)
    ids = {pid: i for i, pid in enumerate(bank)}
    for _ in range(args.epochs):
        model.train()
        order = list(range(len(pairs)))
        random.shuffle(order)
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            pos_idx = [ids[pairs[i][0]] for i in batch]
            neg_idx = [ids[pairs[i][1]] for i in batch]
            ep, en = model(F[pos_idx]), model(F[neg_idx])
            score = (ep * en).sum(dim=1) / np.sqrt(EMBED_DIM)
            loss = torch.nn.functional.softplus(score).mean()  # push positive above negative
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        E = model(F).cpu().numpy()
    emb = {pid: [round(float(v), 5) for v in E[ids[pid]]] for pid in bank}
    doc = {
        "format": "wub-phrase-embeddings-v1", "version": 1,
        "dim": EMBED_DIM, "source": "contrastive on real phrase transitions",
        "pairs": len(pairs), "epochs": args.epochs, "device": str(device),
        "phrases": emb,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps({"phrases": len(emb), "pairs": len(pairs), "epochs": args.epochs}, indent=2))


if __name__ == "__main__":
    main()
