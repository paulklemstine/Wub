"""Fine-tune the shipped phrase policy with quartet-harmony reward.

Reward is the same per-bar four-stem harmony fit the browser uses at selection
time. Policy gradient raises the probability of real next phrases that produce
a coherent quartet and lowers cross-song candidates that clash, while a KL term
keeps the policy near the shipped supervised model.
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

from train_bank_transition import SECTIONS, features, load_manifest
from fine_tune_policy import Policy, mine
from audio_critic_features import load_bank

PITCHED = ("bass", "other", "vocals")
PROGRESSION = [(0, "minor"), (5, "minor"), (7, "minor"), (5, "minor")]
PCS = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
       "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def key_root(key: str) -> int:
    return PCS.get((key or "C").split()[0].replace("m", ""), 0)


def triad(root: int, quality: str) -> np.ndarray:
    v = np.zeros(12, dtype=np.float32)
    v[root % 12] = 1
    third = (root + (3 if quality == "minor" else 5 if quality == "sus" else 4)) % 12
    v[third] = 1
    v[(root + 7) % 12] = 1
    return v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def harmony_fit(mix: Dict[str, str], shifts: Dict[str, int], bank: Dict[str, Any],
                session_root: int) -> float:
    total = 0.0
    for bar in range(4):
        rel, quality = PROGRESSION[bar % len(PROGRESSION)]
        target_root = (session_root + rel) % 12
        target = triad(target_root, quality)
        vectors: List[np.ndarray] = []
        root_score = 1.0
        for stem in PITCHED:
            pid = mix.get(stem)
            if not pid or pid not in bank:
                continue
            row = (bank[pid]["chroma"].get(stem) or [[]])[0]
            vec = np.asarray(row, dtype=np.float32)
            if vec.shape != (12,) or vec.sum() < 1e-8:
                meta = bank[pid]["cells"].get(stem, {})
                raw = (meta.get("r") or [-1, -1, -1, -1])[bar]
                root = (int(raw) + shifts.get(stem, 0)) % 12 if raw is not None and raw >= 0 else target_root
                vec = triad(root, "minor")
            else:
                vec = np.roll(vec, int(shifts.get(stem, 0)) % 12)
            vectors.append(vec)
            best = int(np.argmax(vec))
            dist = min((best - target_root) % 12, (target_root - best) % 12)
            root_score = min(root_score, 1 - dist / 6)
        if len(vectors) < 2:
            return 0.5
        min_cos = min(cos(vectors[i], vectors[j]) for i in range(len(vectors))
                      for j in range(i + 1, len(vectors)))
        chord_fit = max(cos(v, target) for v in vectors)
        total += 0.35 * min_cos + 0.35 * root_score + 0.30 * chord_fit
    return float(np.clip(total / 4, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.0004)
    ap.add_argument("--kl", type=float, default=0.02)
    args = ap.parse_args()
    torch.manual_seed(1707)
    np.random.seed(1707)
    random.seed(1707)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_path = args.ref or args.kits_dir / "bank_transition.json"
    ref_doc = json.loads(ref_path.read_text())
    bank = load_bank(args.kits_dir)
    pos, info = mine(args.kits_dir, args.processed)
    by_section: Dict[str, List[str]] = defaultdict(list)
    for pid, meta in info.items():
        by_section[meta["section"] if meta["section"] in SECTIONS else "GROOVE_A"].append(pid)

    chosen_x, rejected_x, rewards_c, rewards_r, tracks = [], [], [], [], []
    for cur, nxt, section, delta, tid in pos:
        pool = by_section.get(section) or list(info)
        neg = random.choice(pool)
        if neg in (cur, nxt):
            continue
        root = key_root(info[cur]["key"])
        chosen_mix = {s: nxt for s in PITCHED}
        rejected_mix = {s: nxt for s in PITCHED}
        rejected_mix[random.choice(PITCHED)] = neg
        rc = harmony_fit(chosen_mix, {}, bank, root)
        rr = harmony_fit(rejected_mix, {}, bank, root)
        if rc <= rr + 0.01:
            continue  # only train where harmony actually prefers the real pair
        chosen_x.append(features(cur, nxt, section, delta, info))
        rejected_x.append(features(cur, neg, section, delta, info))
        rewards_c.append(rc)
        rewards_r.append(rr)
        tracks.append(tid)
    if len(chosen_x) < 100:
        raise SystemExit(f"not enough harmony-aligned pairs: {len(chosen_x)}")

    C = np.stack(chosen_x).astype(np.float32)
    R = np.stack(rejected_x).astype(np.float32)
    unique = sorted(set(tracks))
    random.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    tr = np.asarray([t not in val_tracks for t in tracks], dtype=bool)
    va = ~tr
    mean = C[tr].mean(axis=0); std = C[tr].std(axis=0); std[std < 1e-4] = 1.0
    def norm(a):
        return torch.tensor((np.asarray(a) - mean) / std, dtype=torch.float32, device=device)

    policy = Policy().to(device)
    policy.load(ref_doc)
    reference = Policy().to(device)
    reference.load(ref_doc)
    for p in reference.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    TC, TR = norm(C[tr]), norm(R[tr])
    RC = torch.tensor(np.asarray(rewards_c)[tr], dtype=torch.float32, device=device)
    RR = torch.tensor(np.asarray(rewards_r)[tr], dtype=torch.float32, device=device)
    for _ in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(TC), device=device)
        for s in range(0, perm.numel(), 256):
            idx = perm[s:s + 256]
            lc, lr_ = policy(TC[idx]), policy(TR[idx])
            logits = torch.stack([lc, lr_], dim=1)
            logp = torch.nn.functional.log_softmax(logits, dim=1)
            probs = logp.exp()
            rewards = torch.stack([RC[idx], RR[idx]], dim=1)
            objective = -(probs * rewards).sum(dim=1).mean()
            with torch.no_grad():
                refp = torch.nn.functional.softmax(
                    torch.stack([reference(TC[idx]), reference(TR[idx])], dim=1), dim=1)
            kl = (probs * (logp - torch.log(refp.clamp_min(1e-6)))).sum(dim=1).mean()
            loss = objective + args.kl * kl
            opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    with torch.no_grad():
        vpc, vpr = policy(norm(C[va])), policy(norm(R[va]))
        pref = float((vpc > vpr).float().mean())
        rpc, rpr = reference(norm(C[va])), reference(norm(R[va]))
        base = float((rpc > rpr).float().mean())
        reward_gap = float(np.mean(np.asarray(rewards_c)[va]
                                   - np.asarray(rewards_r)[va]))

    out = policy.dump(mean, std, ref_doc)
    inherited = dict(ref_doc.get("report", {}))
    out["report"] = {
        "source": "policy gradient with per-bar quartet harmony reward",
        "pairs": len(C), "tracks": len(unique), "train_tracks": len(unique) - len(val_tracks),
        "val_tracks": len(val_tracks), "epochs": args.epochs, "kl": args.kl,
        "val_preference_accuracy": round(pref, 4),
        "reference_preference_accuracy": round(base, 4),
        "mean_reward_gap": round(reward_gap, 4),
        "device": str(device),
    }
    for key in ("source", "real_transitions", "tracks", "train_tracks", "val_tracks",
                "train_examples", "val_examples", "val_auc"):
        if key in inherited:
            out["report"].setdefault(key, inherited[key])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out["report"], indent=2))


if __name__ == "__main__":
    main()
