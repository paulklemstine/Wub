"""Train song-policy compatibility weights (v2, no trivial track flag).

Positives: real +4-bar successors (same track). Negatives: 50% same-section
cross-song, 50% random-section cross-song, so sectionMatch carries signal.
Features: tempoCost, keyCost, sectionMatch, energyMatch, harmonyFit, vocalBonus.
No sameTrack/fresh/plays/return/unit (policy-level, mined separately).
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from train_song_policy import load_manifest, key_shift, session_rate

FEATURES = ["tempoCost", "keyCost", "sectionMatch", "energyMatch", "harmonyFit", "vocalBonus"]

def feat2(cur_id, cand_id, cur, info):
    a, b = info[cur_id], info[cand_id]
    tempo = min(0.5, session_rate(b["bpm"], a["bpm"]))
    shift = abs(key_shift(a["key"], b["key"])) / 6.0
    sec = 1.0 if b["section"] == a["section"] else 0.0
    energy = max(0.0, 1.0 - abs(b["energy"] - a["energy"]))
    d = abs(key_shift(a["key"], b["key"]))
    harmony = 1.0 - min(1.0, d / 6.0)
    vocal = 1.0 if b["section"] in ("DROP", "GROOVE_A") else 0.3
    return [tempo, shift, sec, energy, harmony, vocal]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/song_policy.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=505)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_track, info = load_manifest(args.kits_dir)
    by_section = defaultdict(list)
    for pid, meta in info.items():
        by_section[meta["section"]].append(pid)
    all_ids = list(info)
    X, y, tids = [], [], []
    rng = random.Random(args.seed)
    for tid, entry in by_track.items():
        phrases = entry["phrases"]
        for i in range(len(phrases) - 1):
            cur, nxt = phrases[i], phrases[i + 1]
            if int(nxt["start_bar"]) != int(cur["start_bar"]) + 4:
                continue
            X.append(feat2(cur["id"], nxt["id"], cur, info)); y.append(1.0); tids.append(tid)
            # 1 same-section neg, 1 random neg
            pool_same = by_section.get(cur["section"].upper(), all_ids)
            for _ in range(2):
                neg = rng.choice(pool_same if _ == 0 else all_ids)
                if neg in (cur["id"], nxt["id"]) or info[neg]["track_id"] == tid:
                    continue
                X.append(feat2(cur["id"], neg, cur, info)); y.append(0.0); tids.append(tid)
                break
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    uniq = sorted(set(tids)); rng.shuffle(uniq)
    val_tracks = set(uniq[: max(1, len(uniq) // 10)])
    tr = np.array([t not in val_tracks for t in tids]); va = ~tr
    mean, std = X[tr].mean(0), X[tr].std(0); std[std < 1e-6] = 1.0
    Xt = torch.tensor((X[tr] - mean) / std, device=device)
    yt = torch.tensor(y[tr], device=device)
    Xv = torch.tensor((X[va] - mean) / std, device=device)
    w = torch.zeros(len(FEATURES), device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(perm), 512):
            idx = perm[s: s + 512]
            loss = loss_fn(Xt[idx] @ w + b, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        va_auc = float(roc_auc_score(y[va], (Xv @ w + b).cpu().numpy()))
    w_raw = (w.detach().cpu().numpy() / std)
    b_raw = float(b.detach().cpu().numpy() - (mean / std * w.detach().cpu().numpy()).sum())
    weights = {n: round(float(v), 5) for n, v in zip(FEATURES, w_raw)}
    doc = {"format": "wub-song-policy-v1", "version": 2, "features": FEATURES,
           "weights": weights, "bias": round(b_raw, 5),
           "report": {"pairs": int(len(y)), "val_tracks": int(len(val_tracks)),
                      "val_auc": round(va_auc, 4), "device": str(device),
                      "use": "bounded compatibility prior; tempo/key/harmony/energy/section"}}
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"val_auc {va_auc:.3f}")
    print(json.dumps(weights, indent=2))

if __name__ == "__main__":
    main()
