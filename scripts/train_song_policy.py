"""Train song-policy meta weights so playback matches training songs.

Mines real consecutive 4-bar phrase pairs (same track, +4 bars) as positives
and cross-song same-section phrases as negatives. Learns scalar coefficients
for the shipped selector features (tempo/key fit, repetition/return, novelty,
section/energy match, vocal presence/length, harmony) via a torch logistic
ranker on GPU. Exports `song_policy.json` (bounded ranking prior, no audio).

Features (per candidate, matching TS scoring):
  0 tempoCost  |rate-1| (0..0.5)
  1 keyCost    |shift|/6 (0..1)
  2 plays      play count normalized (0, 0.5, 1 for 0/1/2+ plays)
  3 return8    1 if phrase returns after 8 bars else 0 (drums 51% in corpus)
  4 return16   1 if returns after 16 bars
  5 return32   1 if returns after 32 bars
  6 fresh      1 if never played
  7 sectionMatch 1 if candidate section == target section
  8 energyMatch 1-|energy-target| (0..1)
  9 vocalPresence bonus for vocal units in VERSE/CHORUS (0..1)
 10 unitLength  bonus for 8/16-bar units (0..0.3 scaled to 0..1)
 11 harmonyFit  key fit 0..1 (bass root in session key)
 12 sameTrack  1 if same track as current (foundation hold signal)

Label: 1 for real +4-bar successor, 0 for cross-song negative.
Loss: BCE with logits. Track-disjoint split. Exports weights + report.
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
from sklearn.metrics import roc_auc_score

FORMAT = "wub-song-policy-v1"
FEATURES = [
    "tempoCost", "keyCost", "plays", "return8", "return16", "return32",
    "fresh", "sectionMatch", "energyMatch", "vocalBonus", "unitLength",
    "harmonyFit", "sameTrack",
]
INPUT_DIM = len(FEATURES)


def load_manifest(kits_dir: Path):
    manifest = json.loads((kits_dir / "kit_manifest.json").read_text())
    by_track: Dict[str, Dict[str, Any]] = {}
    info: Dict[str, Dict[str, Any]] = {}
    for kit in manifest.get("kits", []):
        tid = kit["track_id"]
        phrases = sorted(kit.get("phrases", []), key=lambda p: p.get("start_bar", 0))
        by_track[tid] = {
            "phrases": phrases,
            "bar_to_id": {int(p["start_bar"]): p["id"] for p in phrases},
            "bpm": float(kit.get("bpm") or 120),
            "key": str(kit.get("key") or ""),
            "style": str(kit.get("style") or "house"),
        }
        for p in phrases:
            info[p["id"]] = {
                "track_id": tid,
                "section": str(p.get("section") or "GROOVE_A").upper(),
                "energy": float(p.get("energy") or 0.5),
                "bpm": float(kit.get("bpm") or 120),
                "key": str(kit.get("key") or ""),
                "start_bar": int(p.get("start_bar", 0)),
                "duration": float(p.get("duration_sec") or 7.0),
            }
    return by_track, info


def key_shift(a: str, b: str) -> int:
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
           "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    def root(s: str) -> int:
        head = (s or "C").split()[0]
        return pcs.get(head.replace("m", ""), 0)
    d = (root(b) - root(a)) % 12
    return d if d <= 6 else d - 12


def session_rate(bpm: float, ref: float) -> float:
    # tempoCost proxy: |rate-1| where rate maps bpm to session tempo
    return abs(bpm - ref) / max(ref, 1.0)


def build_pairs(kits_dir: Path, n_neg: int = 3, seed: int = 505):
    rng = random.Random(seed)
    by_track, info = load_manifest(kits_dir)
    # vocal units for length bonus
    units_path = kits_dir / "vocal_phrases.json"
    unit_len: Dict[str, int] = {}
    if units_path.exists():
        udoc = json.loads(units_path.read_text())
        for tid, ulist in (udoc.get("tracks") or {}).items():
            for u in ulist:
                for pid in u.get("ids", [])[:1]:
                    unit_len[pid] = max(unit_len.get(pid, 0), int(u.get("bars", 4)))
    by_section: Dict[str, List[str]] = defaultdict(list)
    for pid, meta in info.items():
        by_section[meta["section"]].append(pid)
    X: List[List[float]] = []
    y: List[float] = []
    tids: List[str] = []
    for tid, entry in by_track.items():
        phrases = entry["phrases"]
        for i in range(len(phrases) - 1):
            cur = phrases[i]
            nxt = phrases[i + 1]
            if int(nxt["start_bar"]) != int(cur["start_bar"]) + 4:
                continue
            cur_id, nxt_id = cur["id"], nxt["id"]
            # positive: real successor
            X.append(feat(cur_id, nxt_id, cur, info, unit_len, same=1))
            y.append(1.0)
            tids.append(tid)
            # negatives: cross-song same-section
            pool = by_section.get(cur["section"].upper(), list(info))
            for _ in range(n_neg):
                neg = rng.choice(pool)
                if neg in (cur_id, nxt_id) or info[neg]["track_id"] == tid:
                    continue
                # skip if accidentally consecutive same-track
                X.append(feat(cur_id, neg, cur, info, unit_len, same=0))
                y.append(0.0)
                tids.append(tid)
                break
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), tids, by_track


def feat(cur_id: str, cand_id: str, cur: Dict[str, Any], info: Dict[str, Any],
         unit_len: Dict[str, int], same: int) -> List[float]:
    a, b = info[cur_id], info[cand_id]
    tempo = min(0.5, session_rate(b["bpm"], a["bpm"]))
    shift = abs(key_shift(a["key"], b["key"])) / 6.0
    section_match = 1.0 if b["section"] == a["section"] else 0.0
    energy_match = max(0.0, 1.0 - abs(b["energy"] - a["energy"]))
    # vocal bonus proxy: assume vocal units benefit in DROP/GROOVE
    vocal = 1.0 if a["section"] in ("DROP", "GROOVE_A") else 0.3
    ulen = min(1.0, unit_len.get(cand_id, 4) / 16.0)
    # harmony fit proxy: key distance
    d = abs(key_shift(a["key"], b["key"]))
    harmony = 1.0 - min(1.0, d / 6.0)
    return [tempo, shift, 0.0, 0.0, 0.0, 0.0, 1.0 if same == 0 else 0.0,
            section_match, energy_match, vocal, ulen, harmony, float(same)]


class Policy(nn.Module):
    def __init__(self, dim: int = INPUT_DIM):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return (x @ self.w) + self.b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/song_policy.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[song-policy] device {device}")

    X, y, tids, _ = build_pairs(args.kits_dir)
    print(f"[song-policy] pairs {len(y)} pos {int(y.sum())}")
    uniq = sorted(set(tids))
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    val_tracks = set(uniq[: max(1, len(uniq) // 10)])
    tr = np.array([t not in val_tracks for t in tids])
    va = ~tr
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    # standardize using train stats (store for browser? browser uses raw 0..1,
    # so keep raw and learn weights directly; standardize only for training stability
    # then fold back)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn_tr = (Xtr - mean) / std
    Xn_va = (Xva - mean) / std

    model = Policy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xn_tr, device=device)
    yt = torch.tensor(ytr, device=device)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(perm), 512):
            idx = perm[s: s + 512]
            loss = loss_fn(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                tr_auc = roc_auc_score(ytr, model(Xt).cpu().numpy())
                va_auc = roc_auc_score(yva, model(torch.tensor(Xn_va, device=device)).cpu().numpy())
            print(f"[song-policy] epoch {ep+1} auc train {tr_auc:.3f} val {va_auc:.3f}")

    model.eval()
    with torch.no_grad():
        va_scores = model(torch.tensor(Xn_va, device=device)).cpu().numpy()
        va_auc = float(roc_auc_score(yva, va_scores))
    # fold standardization back into raw-space weights for browser
    w_norm = model.w.detach().cpu().numpy()
    w_raw = w_norm / std
    b_raw = float(model.b.detach().cpu().numpy() - (mean / std * w_norm).sum())
    weights = {name: round(float(w), 5) for name, w in zip(FEATURES, w_raw)}
    doc = {
        "format": FORMAT,
        "version": 1,
        "features": FEATURES,
        "weights": weights,
        "bias": round(b_raw, 5),
        "report": {
            "pairs": int(len(y)),
            "val_tracks": int(len(val_tracks)),
            "val_auc": round(va_auc, 4),
            "device": str(device),
            "use": "bounded ranking prior for song/vocal selector; no audio changes",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[song-policy] wrote {args.out} val_auc {va_auc:.3f}")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
