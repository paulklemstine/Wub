"""Train a phrase-sequence conductor on real song trajectories.

Unlike phrase_conductor (single current phrase), this model sees the last eight
phrases from the real training songs and predicts the next phrase embedding,
which stems change, section and energy. Browser inference scores candidate
phrases by embedding similarity, so every prediction maps to a shipped phrase.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from audio_critic_features import load_bank
from train_bank_transition import load_manifest
from train_phrase_conductor import pattern_vec

SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
STEMS = ("drums", "bass", "other", "vocals")
PITCHED = ("bass", "other", "vocals")
HISTORY = 6
STEP_DIM = 64 + 3 * 48 + 5 + 1  # embedding + bass/other/vocals patterns + section + energy
STATE_DIM = HISTORY * STEP_DIM  # 8 * 218 = 1744
TARGET_DIM = 64 + 4 + 5 + 1  # embedding + change mask + section + energy
HIDDEN = 256


def step_features(pid: str, bank: Dict[str, Any], section: str) -> np.ndarray:
    p = bank[pid]
    v: List[float] = list(p["embedding"])
    for stem in PITCHED:
        row = (p["chroma"].get(stem) or [[]])[0]
        v.extend(pattern_vec(p["cells"].get(stem, {}), row).tolist())
    sec = section if section in SECTIONS else "GROOVE_A"
    for s in SECTIONS:
        v.append(1.0 if s == sec else 0.0)
    v.append(float(np.clip(p["energy"], 0, 1)))
    assert len(v) == STEP_DIM, len(v)
    return np.asarray(v, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(2809)
    np.random.seed(2809)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bank = load_bank(args.kits_dir)
    if args.embeddings:
        emb = json.loads(args.embeddings.read_text()).get("phrases", {})
        for pid, vec in emb.items():
            if pid in bank:
                bank[pid]["embedding"] = np.asarray(vec, dtype=np.float32)
    _, by_track, _ = load_manifest(args.kits_dir)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    tracks: List[str] = []
    for tid, entry in by_track.items():
        phrases = entry["phrases"]
        valid = []
        for i, p in enumerate(phrases):
            sec = str(p.get("section") or "GROOVE_A").upper()
            if p["id"] in bank:
                valid.append((p["id"], sec))
        if len(valid) < HISTORY + 1:
            continue
        for start in range(len(valid) - HISTORY):
            seq = valid[start:start + HISTORY + 1]
            if not all(seq[i + 1][0].replace("_p", "_p") for i in range(HISTORY)):
                continue
            x = np.concatenate([step_features(pid, bank, sec) for pid, sec in seq[:HISTORY]])
            nxt_id, nxt_sec = seq[HISTORY]
            target = list(bank[nxt_id]["embedding"])
            # Stem change: phrase id changes together for all stems in a track.
            changes = [1.0] * 4
            target.extend(changes)
            secvec = np.zeros(5, dtype=np.float32)
            if nxt_sec in SECTIONS:
                secvec[SECTIONS.index(nxt_sec)] = 1
            target.extend(secvec.tolist())
            target.append(float(np.clip(bank[nxt_id]["energy"], 0, 1)))
            xs.append(x)
            ys.append(np.asarray(target, dtype=np.float32))
            tracks.append(tid)
    if len(xs) < 100:
        raise SystemExit(f"not enough sequences: {len(xs)}")

    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    unique = sorted(set(tracks))
    random.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    val = np.asarray([t in val_tracks for t in tracks], dtype=bool)
    tr = ~val
    mean = X[tr].mean(axis=0); std = X[tr].std(axis=0); std[std < 1e-4] = 1.0
    xt = torch.tensor((X[tr] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(Y[tr], dtype=torch.float32, device=device)
    net = nn.Sequential(
        nn.Linear(STATE_DIM, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, 128), nn.Tanh(),
        nn.Linear(128, TARGET_DIM),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0008)
    for _ in range(args.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 256):
            idx = perm[s:s + 256]
            pred = net(xt[idx])
            loss = (torch.nn.functional.mse_loss(pred[:, :64], yt[idx][:, :64])
                    + 0.3 * torch.nn.functional.binary_cross_entropy_with_logits(
                        pred[:, 64:68], yt[idx][:, 64:68])
                    + 0.5 * torch.nn.functional.cross_entropy(
                        pred[:, 68:73], yt[idx][:, 68:73].argmax(dim=1))
                    + 0.3 * torch.nn.functional.mse_loss(pred[:, 73], yt[idx][:, 73]))
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        xv = torch.tensor((X[val] - mean) / std, dtype=torch.float32, device=device)
        pv = net(xv).cpu().numpy()
    truth = Y[val]
    cos = np.sum(pv[:, :64] * truth[:, :64], axis=1) / (
        np.linalg.norm(pv[:, :64], axis=1) * np.linalg.norm(truth[:, :64], axis=1) + 1e-8)
    sec_acc = float((pv[:, 68:73].argmax(1) == truth[:, 68:73].argmax(1)).mean())

    def layer(mod):
        return {"w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
                "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()]}

    doc = {
        "format": "wub-sequence-conductor-v1", "version": 1,
        "state_dim": STATE_DIM, "target_dim": TARGET_DIM, "history": HISTORY,
        "hidden": HIDDEN, "embed_dim": 64, "sections": list(SECTIONS), "stems": list(STEMS),
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [layer(net[0]), layer(net[2]), layer(net[4])],
        "report": {
            "source": "real song phrase trajectories mapped to shipped phrases",
            "sequences": len(X), "tracks": len(unique),
            "train_tracks": len(unique) - len(val_tracks), "val_tracks": len(val_tracks),
            "val_embedding_cosine": round(float(cos.mean()), 4),
            "section_accuracy": round(sec_acc, 4),
            "epochs": args.epochs, "device": str(device),
            "embeddings": "learned" if args.embeddings else "taste",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps({k: doc["report"][k] for k in (
        "sequences", "tracks", "val_embedding_cosine", "section_accuracy", "embeddings")}, indent=2))


if __name__ == "__main__":
    main()
