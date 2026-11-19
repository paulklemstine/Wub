"""Train a phrase-transition model in the shipped bank's own phrase space.

The conductor v2 was trained on raw song cells, but the browser can only choose
shipped phrases/bars — a distribution mismatch. This model instead learns from
real song chronology *mapped to shipped phrase ids*: for each real 4-bar window
that belongs to a kit phrase, the following window's phrase id is the positive
next phrase. Negatives come from other phrases in the same section/style.

The exported sidecar is a compact logistic scorer plus a real-chronology top-K
successor table. It is a bounded ranking prior: it never schedules audio.
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
from audio_critic_features import critic_score, load_bank, load_critic, phrase_features

SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
INPUT_DIM = 136  # 64 current + 64 next + 5 section + 1 energy delta + 1 bpm + 1 key
HIDDEN = 32
FORMAT = "wub-bank-transition-v1"


def load_manifest(kits_dir: Path):
    manifest = json.loads((kits_dir / "kit_manifest.json").read_text())
    by_track: Dict[str, Dict[str, Any]] = {}
    phrase_info: Dict[str, Dict[str, Any]] = {}
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
            phrase_info[p["id"]] = {
                "embedding": np.asarray(p.get("embedding") or [0.0] * 64, dtype=np.float32),
                "track_id": tid,
                "section": str(p.get("section") or "GROOVE_A").upper(),
                "style": str(kit.get("style") or "house"),
                "bpm": float(kit.get("bpm") or 120),
                "key": str(kit.get("key") or ""),
            }
    return manifest, by_track, phrase_info


def song_sections(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tats = doc.get("tatums") or []
    n = len(tats) // 16
    if n < 8:
        return [], []
    feats = bar_features(segs, tats)
    spans = segment_sections(feats)
    labels = [s.label for s in spans for _ in range(s.n_bars)]
    labels = (labels + ["?"] * n)[:n]
    energies = [float(f.get("energy", -60.0)) for f in feats]
    return labels, energies


def key_distance(a: str, b: str) -> float:
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
           "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    def root(s):
        head = s.split()[0] if s else "C"
        return pcs.get(head.replace("m", ""), 0)
    d = abs(root(a) - root(b)) % 12
    return min(d, 12 - d) / 6.0


def features(cur_id: str, next_id: str, section: str, energy_delta: float,
             info: Dict[str, Dict[str, Any]]) -> np.ndarray:
    a, b = info[cur_id], info[next_id]
    v = list(a["embedding"]) + list(b["embedding"])
    one = np.zeros(5, dtype=np.float32)
    if section in SECTIONS:
        one[SECTIONS.index(section)] = 1
    v.extend(one.tolist())
    v.append(float(np.clip(energy_delta / 20.0, -1, 1)))
    v.append(min(1.0, abs(a["bpm"] - b["bpm"]) / 12.0))
    v.append(key_distance(a["key"], b["key"]))
    assert len(v) == INPUT_DIM, len(v)
    return np.asarray(v, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--critic", type=Path, default=None,
                    help="song_critic.json used to weight examples by audio-derived song-ness")
    args = ap.parse_args()
    torch.manual_seed(809)
    np.random.seed(809)
    random.seed(809)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest, by_track, info = load_manifest(args.kits_dir)
    pos: List[Tuple[str, str, str, float, str]] = []
    successors: Dict[str, Counter] = defaultdict(Counter)
    for tid, entry in by_track.items():
        path = args.processed / f"{tid}.json"
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
            pos.append((p["id"], nxt, section if section in SECTIONS else "GROOVE_A",
                        float(delta), tid))
            successors[p["id"]][nxt] += 1
    if len(pos) < 200:
        raise SystemExit(f"not enough real phrase transitions: {len(pos)}")

    unique = sorted({t for *_, t in pos})
    random.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    by_section: Dict[str, List[str]] = defaultdict(list)
    for pid, meta in info.items():
        by_section[meta["section"] if meta["section"] in SECTIONS else "GROOVE_A"].append(pid)

    critic = load_critic(args.critic) if args.critic else None
    bank_phrases = load_bank(args.kits_dir) if critic else {}
    sample_weights: List[float] = []

    def critic_of(pid: str) -> float:
        if not critic or pid not in bank_phrases:
            return 0.5
        return float(critic_score(phrase_features(pid, bank_phrases), critic))

    xs, ys, binary_y, track_of = [], [], [], []
    for cur, nxt, section, delta, tid in pos:
        xs.append(features(cur, nxt, section, delta, info))
        ys.append(1.0)
        binary_y.append(1.0)
        track_of.append(tid)
        sample_weights.append(0.5 + critic_of(nxt))
        for _ in range(2):
            pool = by_section.get(section) or list(info)
            neg = random.choice(pool)
            if neg == nxt or neg == cur:
                continue
            xs.append(features(cur, neg, section, delta, info))
            ys.append(0.0)
            binary_y.append(0.0)
            track_of.append(tid)
            sample_weights.append(0.5 + critic_of(neg))
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    val = np.asarray([t in val_tracks for t in track_of], dtype=bool)
    tr = ~val
    mean = x[tr].mean(axis=0).astype(np.float32)
    std = x[tr].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    xt = torch.tensor((x[tr] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
    wt = torch.tensor(np.asarray(sample_weights, dtype=np.float32)[tr], dtype=torch.float32, device=device)
    xv = torch.tensor((x[val] - mean) / std, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    for _ in range(args.epochs):
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 512):
            idx = perm[s:s + 512]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                net(xt[idx]).squeeze(1), yt[idx], reduction="none") * wt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(xv).squeeze(1).cpu().numpy()
    auc = float(roc_auc_score(y[val], pv))

    topk = {}
    for pid, counter in successors.items():
        total = sum(counter.values())
        topk[pid] = [
            {"id": q, "p": round(c / total, 4)}
            for q, c in counter.most_common(8)
        ]

    doc = {
        "format": FORMAT, "version": 1,
        "input_dim": INPUT_DIM, "hidden": HIDDEN,
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [
            {"w": [[round(float(v), 5) for v in row] for row in net[0].weight.detach().cpu().numpy()],
             "b": [round(float(v), 5) for v in net[0].bias.detach().cpu().numpy()]},
            {"w": [[round(float(v), 5) for v in row] for row in net[2].weight.detach().cpu().numpy()],
             "b": [round(float(v), 5) for v in net[2].bias.detach().cpu().numpy()]},
        ],
        "embedding_dim": 64,
        "sections": list(SECTIONS),
        "topk": topk,
        "report": {
            "source": "kit_manifest phrase ids + data/processed/*.json real chronology",
            "real_transitions": len(pos), "tracks": len(unique),
            "train_tracks": len(unique) - len(val_tracks), "val_tracks": len(val_tracks),
            "train_examples": int(tr.sum()), "val_examples": int(val.sum()),
            "val_auc": round(auc, 4), "epochs": args.epochs, "device": str(device),
            "audio_critic_weighted": bool(critic),
            "use": "bounded ranking prior over shipped phrases; no audio changes",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps({
        "real_transitions": len(pos), "tracks": len(unique),
        "train_examples": int(tr.sum()), "val_examples": int(val.sum()),
        "val_auc": round(auc, 4), "phrases_with_topk": len(topk),
    }, indent=2))


if __name__ == "__main__":
    main()
