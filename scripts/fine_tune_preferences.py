"""Fine-tune the phrase policy from implicit reticule/pin preferences.

Input is the JSON exported by window.__wubPreferenceExport():
  {version, source, events:[{kind, stem, from, to, section, ...}]}

A retarget means "from this phrase, move this stem to that one"; a hold/drop
near the current phrase means "keep it"; a release means "stop forcing this
phrase". Because a listening session produces few gestures, each user event is
augmented with several same-section alternatives and mixed with real-song
chronology pairs so the policy does not drift away from the corpus.
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
from sklearn.metrics import roc_auc_score

from train_bank_transition import SECTIONS, features, load_manifest
from fine_tune_policy import Policy
from audio_critic_features import critic_score, load_bank, load_critic, phrase_features


def build_pairs(events: List[Dict[str, Any]], info: Dict[str, Any],
                augment: int, seed: int):
    rng = random.Random(seed)
    by_section: Dict[str, List[str]] = defaultdict(list)
    for pid, meta in info.items():
        by_section[meta["section"] if meta["section"] in SECTIONS else "GROOVE_A"].append(pid)
    pairs: List[Tuple[str, str, str, str, float, str, float]] = []
    counts = Counter(e.get("to") or e.get("from") for e in events if e.get("to") or e.get("from"))
    for ev in events:
        kind = ev.get("kind")
        src = ev.get("from")
        dst = ev.get("to") or src
        if not src or src not in info:
            continue
        section = ev.get("section") or info[src]["section"]
        section = section if section in SECTIONS else "GROOVE_A"
        delta = float(info.get(dst, info[src]).get("energy", 0.5) - info[src].get("energy", 0.5))
        chosen = dst if dst in info else src
        strength = 1.0
        if kind == "release":
            chosen = src  # released phrase is the rejected one; handled below
        pool = by_section.get(section) or list(info)
        repeats = max(1, min(4, counts.get(dst, 1)))
        for _ in range(augment * repeats):
            neg = rng.choice(pool)
            if neg in (src, chosen):
                continue
            if kind == "release":
                pairs.append((src, neg, src, section, delta, "user-release", 0.6))
            else:
                pairs.append((src, chosen, neg, section, delta, "user-pref", strength))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preferences", type=Path, required=True)
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--critic", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["dpo", "pg"], default="pg")
    ap.add_argument("--augment", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.0003)
    args = ap.parse_args()
    torch.manual_seed(2207)
    np.random.seed(2207)
    random.seed(2207)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_path = args.ref or args.kits_dir / "bank_transition.json"
    ref_doc = json.loads(ref_path.read_text())
    doc = json.loads(args.preferences.read_text())
    raw_events = doc.get("events", []) if isinstance(doc, dict) else list(doc)
    _, _, info = load_manifest(args.kits_dir)
    pairs = build_pairs(raw_events, info, args.augment, 2207)
    if not pairs:
        raise SystemExit("no usable preference events")
    critic = load_critic(args.critic) if args.critic else None
    bank = load_bank(args.kits_dir) if critic else {}

    pc, pr, weights = [], [], []
    for cur, nxt, neg, section, delta, origin, strength in pairs:
        pc.append(features(cur, nxt, section, delta, info))
        pr.append(features(cur, neg, section, delta, info))
        w = strength
        if critic and nxt in bank and neg in bank:
            w *= 0.5 + critic_score(phrase_features(nxt, bank), critic)
        weights.append(w)
    C = np.stack(pc).astype(np.float32)
    R = np.stack(pr).astype(np.float32)
    W = np.asarray(weights, dtype=np.float32)

    mean = C.mean(axis=0); std = C.std(axis=0); std[std < 1e-4] = 1.0
    def norm(a):
        return torch.tensor((np.asarray(a) - mean) / std, dtype=torch.float32, device=device)

    policy = Policy().to(device)
    policy.load(ref_doc)
    reference = Policy().to(device)
    reference.load(ref_doc)
    for p in reference.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    TC, TR, TW = norm(C), norm(R), torch.tensor(W, dtype=torch.float32, device=device)
    for _ in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(TC), device=device)
        for s in range(0, perm.numel(), 256):
            idx = perm[s:s + 256]
            lc, lr_ = policy(TC[idx]), policy(TR[idx])
            if args.mode == "dpo":
                with torch.no_grad():
                    rc, rr = reference(TC[idx]), reference(TR[idx])
                loss = -torch.nn.functional.logsigmoid(0.6 * ((lc - rc) - (lr_ - rr))) * TW[idx]
                loss = loss.mean()
            else:
                logits = torch.stack([lc, lr_], dim=1)
                probs = torch.nn.functional.softmax(logits, dim=1)
                rewards = torch.stack([TW[idx], torch.zeros_like(TW[idx])], dim=1)
                loss = -(probs * rewards).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    with torch.no_grad():
        vc, vr = policy(norm(C)), policy(norm(R))
        pref = float((vc > vr).float().mean())
        rc0, rr0 = reference(norm(C)), reference(norm(R))
        base = float((rc0 > rr0).float().mean())
    out = policy.dump(mean, std, ref_doc)
    out["report"] = {
        "source": "implicit reticule/pin preferences",
        "mode": args.mode, "events": len(raw_events), "pairs": len(pairs),
        "augment": args.augment, "epochs": args.epochs,
        "val_preference_accuracy": round(pref, 4),
        "reference_preference_accuracy": round(base, 4),
        "device": str(device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps(out["report"], indent=2))


if __name__ == "__main__":
    main()
