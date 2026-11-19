"""Fine-tune the phrase-selection policy with critic-derived preferences.

Two objectives are available:
  * DPO preference loss against the shipped bank-transition reference policy;
  * expected-reward policy gradient with the audio-derived song critic as reward.

Both keep the exact browser artifact format of bank_transition.json, so the
shipped inference path is unchanged.
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

from wub_core.section_arcs import bar_features, segment_sections
from train_bank_transition import (FORMAT, HIDDEN, INPUT_DIM, SECTIONS, features,
                                   key_distance, load_manifest, song_sections)
from audio_critic_features import critic_score, load_bank, load_critic, phrase_features


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x))).squeeze(1)

    def load(self, doc: Dict[str, Any]):
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(doc["layers"][0]["w"], dtype=torch.float32))
            self.fc1.bias.copy_(torch.tensor(doc["layers"][0]["b"], dtype=torch.float32))
            self.fc2.weight.copy_(torch.tensor(doc["layers"][1]["w"], dtype=torch.float32))
            self.fc2.bias.copy_(torch.tensor(doc["layers"][1]["b"], dtype=torch.float32))

    def dump(self, mean, std, ref_doc: Dict[str, Any]):
        return {
            "format": FORMAT, "version": 2,
            "input_dim": INPUT_DIM, "hidden": HIDDEN,
            "embedding_dim": ref_doc.get("embedding_dim", 64),
            "sections": ref_doc.get("sections", list(SECTIONS)),
            "topk": ref_doc.get("topk", {}),
            "mean": [round(float(v), 5) for v in mean],
            "std": [round(float(v), 5) for v in std],
            "layers": [
                {"w": [[round(float(v), 5) for v in row] for row in self.fc1.weight.detach().cpu().numpy()],
                 "b": [round(float(v), 5) for v in self.fc1.bias.detach().cpu().numpy()]},
                {"w": [[round(float(v), 5) for v in row] for row in self.fc2.weight.detach().cpu().numpy()],
                 "b": [round(float(v), 5) for v in self.fc2.bias.detach().cpu().numpy()]},
            ],
        }


def mine(kits_dir: Path, processed: Path):
    _, by_track, info = load_manifest(kits_dir)
    pos: List[Tuple[str, str, str, float, str]] = []
    for tid, entry in by_track.items():
        path = processed / f"{tid}.json"
        if not path.exists():
            continue
        try:
            labels, energies = song_sections(json.loads(path.read_text()))
        except Exception:
            continue
        for p in entry["phrases"]:
            b = int(p["start_bar"])
            nxt = entry["bar_to_id"].get(b + 4)
            if not nxt or nxt == p["id"]:
                continue
            section = labels[b] if b < len(labels) else p["section"]
            delta = (energies[b + 4] - energies[b]) if b + 4 < len(energies) else 0.0
            pos.append((p["id"], nxt, section if section in SECTIONS else "GROOVE_A", float(delta), tid))
    return pos, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--critic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["dpo", "pg", "both"], default="dpo")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=0.0005)
    args = ap.parse_args()
    torch.manual_seed(1207)
    np.random.seed(1207)
    random.seed(1207)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_doc = json.loads(args.ref.read_text())
    critic = load_critic(args.critic)
    bank = load_bank(args.kits_dir)
    pos, info = mine(args.kits_dir, args.processed)
    by_section: Dict[str, List[str]] = defaultdict(list)
    for pid, meta in info.items():
        by_section[meta["section"] if meta["section"] in SECTIONS else "GROOVE_A"].append(pid)

    pairs: List[Tuple[np.ndarray, np.ndarray, float, str]] = []
    for cur, nxt, section, delta, tid in pos:
        pool = by_section.get(section) or list(info)
        neg = random.choice(pool)
        if neg in (cur, nxt):
            continue
        rc = critic_score(phrase_features(nxt, bank), critic)
        rn = critic_score(phrase_features(neg, bank), critic)
        pairs.append((
            features(cur, nxt, section, delta, info),
            features(cur, neg, section, delta, info),
            1.0 if rc > rn else 0.35, tid,
        ))
    if len(pairs) < 200:
        raise SystemExit(f"not enough preference pairs: {len(pairs)}")

    unique = sorted({t for *_, t in pairs})
    random.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    chosen = np.stack([p[0] for p in pairs if p[3] not in val_tracks]).astype(np.float32)
    rejected = np.stack([p[1] for p in pairs if p[3] not in val_tracks]).astype(np.float32)
    margins = np.asarray([p[2] for p in pairs if p[3] not in val_tracks], dtype=np.float32)
    v_chosen = np.stack([p[0] for p in pairs if p[3] in val_tracks]).astype(np.float32)
    v_rejected = np.stack([p[1] for p in pairs if p[3] in val_tracks]).astype(np.float32)
    v_margins = np.asarray([p[2] for p in pairs if p[3] in val_tracks], dtype=np.float32)

    mean = chosen.mean(axis=0); std = chosen.std(axis=0); std[std < 1e-4] = 1.0
    def norm(a):
        return torch.tensor((np.asarray(a) - mean) / std, dtype=torch.float32, device=device)

    policy = Policy().to(device)
    policy.load(ref_doc)
    reference = Policy().to(device)
    reference.load(ref_doc)
    for p in reference.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    tc, tj, tm = norm(chosen), norm(rejected), torch.tensor(margins, device=device)
    for _ in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(tc), device=device)
        for s in range(0, perm.numel(), 256):
            idx = perm[s:s + 256]
            pc, pr = policy(tc[idx]), policy(tj[idx])
            loss = torch.zeros((), device=device)
            if args.mode in ("dpo", "both"):
                with torch.no_grad():
                    rc0, rr0 = reference(tc[idx]), reference(tj[idx])
                logr = torch.nn.functional.logsigmoid(
                    args.beta * ((pc - rc0) - (pr - rr0)))
                loss = loss - (logr * tm[idx]).mean()
            if args.mode in ("pg", "both"):
                # Expected critic-aligned reward; real chronology + critic margin.
                logits = torch.stack([pc, pr], dim=1)
                probs = torch.nn.functional.softmax(logits, dim=1)
                rewards = torch.stack([tm[idx], torch.zeros_like(tm[idx])], dim=1)
                loss = loss - (probs * rewards).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    policy.eval()
    with torch.no_grad():
        vpc, vpr = policy(norm(v_chosen)), policy(norm(v_rejected))
        pref = float((vpc > vpr).float().mean())
        base_pc, base_pr = reference(norm(v_chosen)), reference(norm(v_rejected))
        base_pref = float((base_pc > base_pr).float().mean())
        labels = np.concatenate([np.ones(len(vpc)), np.zeros(len(vpr))])
        scores = np.concatenate([vpc.cpu().numpy(), vpr.cpu().numpy()])
        auc = float(roc_auc_score(labels, scores))
        ref_scores = np.concatenate([base_pc.cpu().numpy(), base_pr.cpu().numpy()])
        ref_auc = float(roc_auc_score(labels, ref_scores))

    doc = policy.dump(mean, std, ref_doc)
    inherited = dict(ref_doc.get("report", {}))
    doc["report"] = {
        "source": "DPO/PG fine-tune against audio-derived song critic",
        "mode": args.mode, "epochs": args.epochs, "beta": args.beta,
        "tracks": len(unique), "train_tracks": len(unique) - len(val_tracks),
        "val_tracks": len(val_tracks), "pairs": len(pairs),
        "val_preference_accuracy": round(pref, 4),
        "reference_preference_accuracy": round(base_pref, 4),
        "val_auc": round(auc, 4), "reference_val_auc": round(ref_auc, 4),
        "device": str(device),
        "use": "bounded ranking prior; no audio changes",
    }
    for key in ("source", "real_transitions", "tracks", "train_tracks", "val_tracks",
                "train_examples", "val_examples"):
        if key in inherited:
            doc["report"].setdefault(key, inherited[key])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps({k: doc["report"][k] for k in (
        "mode", "pairs", "val_preference_accuracy", "reference_preference_accuracy", "val_auc")}, indent=2))


if __name__ == "__main__":
    main()
