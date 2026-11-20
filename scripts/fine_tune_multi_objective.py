"""Fine-tune the phrase policy with a multi-objective reward.

Reward = 0.50 * per-bar quartet harmony
       + 0.25 * source novelty (track != current track)
       + 0.25 * arc energy fit

An explicit "repeat the current phrase" action is included with a heavily
discounted reward, so maximizing expected reward cannot collapse the session
onto a single source. KL to the shipped policy limits drift.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from train_bank_transition import SECTIONS, features, load_manifest
from fine_tune_policy import Policy, mine
from audio_critic_features import load_bank
from fine_tune_harmony_policy import harmony_fit, PITCHED, key_root

SECTION_ENERGY = {
    "INTRO": 0.30, "GROOVE_A": 0.72, "BUILDUP": 0.82, "DROP": 0.95, "BREAKDOWN": 0.20,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0004)
    ap.add_argument("--kl", type=float, default=0.03)
    ap.add_argument("--w-harmony", type=float, default=0.50)
    ap.add_argument("--w-novelty", type=float, default=0.25)
    ap.add_argument("--w-arc", type=float, default=0.25)
    ap.add_argument("--repeat-discount", type=float, default=0.15)
    args = ap.parse_args()
    torch.manual_seed(2711)
    np.random.seed(2711)
    random.seed(2711)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_path = args.ref or args.kits_dir / "bank_transition.json"
    ref_doc = json.loads(ref_path.read_text())
    bank = load_bank(args.kits_dir)
    pos, info = mine(args.kits_dir, args.processed)
    by_section: Dict[str, List[str]] = defaultdict(list)
    track_freq: Counter = Counter()
    for pid, meta in info.items():
        by_section[meta["section"] if meta["section"] in SECTIONS else "GROOVE_A"].append(pid)
        track_freq[meta["track_id"]] += 1

    states: List[np.ndarray] = []
    action_features: List[List[np.ndarray]] = []
    action_rewards: List[List[float]] = []
    tracks: List[str] = []
    for cur, nxt, section, delta, tid in pos:
        pool = by_section.get(section) or list(info)
        neg = random.choice(pool)
        if neg in (cur, nxt):
            continue
        root = key_root(info[cur]["key"])
        cur_track = info[cur]["track_id"]
        target_energy = SECTION_ENERGY.get(section, 0.7)

        def reward_for(cand: str, repeat: bool = False) -> float:
            chosen_mix = {s: nxt for s in PITCHED}
            mixed = dict(chosen_mix)
            mixed[random.choice(list(PITCHED))] = cand
            h = harmony_fit(mixed, {}, bank, root)
            cand_track = info[cand]["track_id"]
            novelty = 1.0 if cand_track != cur_track else 0.0
            novelty *= 0.5 + 0.5 / (1 + track_freq[cand_track] / 10)
            energy = float(info[cand].get("energy", 0.5))
            arc = max(0.0, 1 - abs(energy - target_energy) / 0.5)
            r = args.w_harmony * h + args.w_novelty * novelty + args.w_arc * arc
            if repeat:
                r *= args.repeat_discount
            return float(np.clip(r, 0, 1))

        actions = [nxt, neg, cur]
        feats = [features(cur, cand, section, delta, info) for cand in actions]
        rewards = [reward_for(nxt), reward_for(neg), reward_for(cur, repeat=True)]
        states.append(features(cur, cur, section, delta, info))  # state context only
        action_features.append(feats)
        action_rewards.append(rewards)
        tracks.append(tid)
    if len(states) < 100:
        raise SystemExit(f"not enough pairs: {len(states)}")

    X = np.stack(states).astype(np.float32)
    unique = sorted(set(tracks))
    random.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    tr = np.asarray([t not in val_tracks for t in tracks], dtype=bool)
    va = ~tr
    mean = X[tr].mean(axis=0); std = X[tr].std(axis=0); std[std < 1e-4] = 1.0
    def norm(a):
        return torch.tensor((np.asarray(a) - mean) / std, dtype=torch.float32, device=device)

    policy = Policy().to(device)
    policy.load(ref_doc)
    reference = Policy().to(device)
    reference.load(ref_doc)
    for p in reference.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    tensors = []
    for i in range(len(states)):
        tensors.append((
            norm(action_features[i]),
            torch.tensor(action_rewards[i], dtype=torch.float32, device=device),
            bool(tr[i]),
        ))

    for _ in range(args.epochs):
        policy.train()
        order = list(np.where(tr)[0])
        np.random.shuffle(order)
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            lcs = torch.stack([policy(tensors[i][0]) for i in batch])
            logp = torch.nn.functional.log_softmax(lcs, dim=1)
            probs = logp.exp()
            rewards = torch.stack([tensors[i][1] for i in batch])
            objective = -(probs * rewards).sum(dim=1).mean()
            with torch.no_grad():
                refp = torch.stack([
                    torch.nn.functional.softmax(reference(tensors[i][0]), dim=0) for i in batch])
            kl = (probs * (logp - torch.log(refp.clamp_min(1e-6)))).sum(dim=1).mean()
            loss = objective + args.kl * kl
            opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    with torch.no_grad():
        alpha = float(np.mean([action_rewards[i][0] - action_rewards[i][2] for i in np.where(va)[0]]))
        val_pref = float(np.mean([
            float(policy(tensors[i][0])[0] > policy(tensors[i][0])[1]) for i in np.where(va)[0]]))
        ref_pref = float(np.mean([
            float(reference(tensors[i][0])[0] > reference(tensors[i][0])[1]) for i in np.where(va)[0]]))
    out = policy.dump(mean, std, ref_doc)
    inherited = dict(ref_doc.get("report", {}))
    out["report"] = {
        "source": "multi-objective PG (harmony + novelty + arc, repeat-constrained)",
        "pairs": len(states), "tracks": len(unique),
        "train_tracks": len(unique) - len(val_tracks), "val_tracks": len(val_tracks),
        "epochs": args.epochs, "kl": args.kl,
        "val_preference_accuracy": round(val_pref, 4),
        "reference_preference_accuracy": round(ref_pref, 4),
        "mean_reward_gap": round(alpha, 4), "device": str(device),
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
