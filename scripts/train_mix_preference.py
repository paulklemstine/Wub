"""Train the mix-preference donor prior from synthetic two-song mixes.

The jukebox mixes stems from different songs, but the ranker only ever saw
real mixes as positives and donor swaps as negatives: it learned
"real = good, synthetic = bad" and can only pick the least-bad donor.
This script mines *good* synthetic positives so the selector learns what a
working two-song mix looks like.

For every phrase, donor candidates are other tracks' phrases in the
learned-similarity neighborhood (64-d phrase-embedding cosine, same
embeddings the browser already loads), intersected with the key/tempo
ladder (bpm window + key shift, the same hard limits playback enforces).

No-trainwreck gates (all browser-computable, mirrored in stem_remix.ts):
  harmonyOk  bass-root interval in {0, 5, 7} (unison/fourth/fifth)
  energyOk   |energy - current| <= 0.20 (no energy jump)
  sectionOk  same section (style continuity)

Labels: synthetic donors only (the browser only ever scores cross-track
candidates, so training matches inference exactly). A donor passing all
three gates -> 1 (good synthetic); failing any gate -> 0. No gatePass
feature: the head learns soft weights for the sub-components (similarity,
tempo/key fit, section/energy continuity, harmony) instead of
re-discovering the gate. Real +4-bar successors are excluded: same-track
similarity (~0.95) would swamp the cross-track effect we need.

Usage:
    PYTHONPATH=. python3 scripts/train_mix_preference.py \
      --kits-dir frontend/public/data/kits \
      --out frontend/public/data/kits/mix_preference.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FORMAT = "wub-mix-preference-v1"
FEATURES = [
    "sim", "tempoCost", "keyCost", "sectionMatch", "energyMatch",
    "harmonyFit",
]
NEIGHBORS = 64
MAX_SYNTH_PER_CUR = 6

PITCH = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
         "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
         "A#": 10, "BB": 10, "B": 11}


def is_val(track: str) -> bool:
    return int(hashlib.sha256(track.encode()).hexdigest()[:8], 16) % 8 == 0


def parse_key(name: str):
    import re
    m = re.match(r"^([A-Ga-g])([#b\xe2\x99\xaf\xe2\x99\xad]?)\s*(major|minor|maj|min)$",
                 (name or "").strip())
    if not m:
        return None
    acc = m.group(2)
    if acc in ("\xe2\x99\xaf", "#"):
        acc = "#"
    elif acc in ("\xe2\x99\xad", "b", "B"):
        acc = "B"
    else:
        acc = ""
    root = PITCH.get(f"{m.group(1).upper()}{acc}")
    if root is None:
        return None
    minor = m.group(3).lower().startswith("min")
    eff = (root + 3) % 12 if minor else root
    return eff


def key_shift(session_key: str, phrase_key: str):
    a = parse_key(session_key)
    b = parse_key(phrase_key)
    if a is None or b is None:
        return None
    d = (a - b) % 12
    return d if d <= 6 else d - 12


def load_all(kits_dir: Path):
    manifest = json.loads((kits_dir / "kit_manifest.json").read_text())
    emb_doc = json.loads((kits_dir / "phrase_embeddings.json").read_text())
    emb = emb_doc.get("phrases", {})
    by_track: Dict[str, Dict[str, Any]] = {}
    info: Dict[str, Dict[str, Any]] = {}
    for kit in manifest.get("kits", []):
        tid = str(kit.get("track_id"))
        phrases = sorted(kit.get("phrases", []),
                         key=lambda p: int(p.get("start_bar", 0)))
        by_track[tid] = {
            "phrases": phrases,
            "bar_to_id": {int(p.get("start_bar", 0)): p.get("id")
                          for p in phrases},
            "bpm": float(kit.get("bpm") or 120),
            "key": str(kit.get("key") or "C major"),
        }
        for p in phrases:
            pid = str(p.get("id"))
            info[pid] = {
                "track_id": tid,
                "section": str(p.get("section") or "GROOVE_A").upper(),
                "energy": float(p.get("energy") or 0.5),
                "bpm": float(kit.get("bpm") or 120),
                "key": str(kit.get("key") or "C major"),
                "start_bar": int(p.get("start_bar", 0)),
                "bass_root": p.get("bass_root"),
            }
    return by_track, info, emb


def cosine(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(va @ va)
    nb = float(vb @ vb)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.clip((va @ vb) / np.sqrt(na * nb), -1.0, 1.0))


def harmony_interval(cur_root, cand_root):
    if cur_root is None or cand_root is None:
        return None
    try:
        r = int(cand_root) - int(cur_root)
    except (TypeError, ValueError):
        return None
    return abs(r) % 12


def gates(cur: Dict[str, Any], cand: Dict[str, Any]) -> Tuple[bool, float, float]:
    interval = harmony_interval(cur.get("bass_root"), cand.get("bass_root"))
    harmony_ok = interval in (0, 5, 7) if interval is not None else False
    energy_ok = abs(float(cand["energy"]) - float(cur["energy"])) <= 0.20
    section_ok = cand["section"] == cur["section"]
    harmony_fit = 0.5 if interval is None else 1.0 - min(interval, 12 - interval) / 6.0
    shift = key_shift(cur["key"], cand["key"])
    key_cost = abs(shift) / 6.0 if shift is not None else 0.5
    return (harmony_ok and energy_ok and section_ok), harmony_fit, key_cost


def feat_row(sim: float, cur: Dict[str, Any], cand: Dict[str, Any],
             harmony_fit: float, key_cost: float) -> List[float]:
    tempo = min(0.5, abs(float(cand["bpm"]) - float(cur["bpm"]))
                / max(1.0, float(cur["bpm"])))
    section_match = 1.0 if cand["section"] == cur["section"] else 0.0
    energy_match = max(0.0, 1.0 - abs(float(cand["energy"]) - float(cur["energy"])))
    return [sim, tempo, key_cost, section_match, energy_match, harmony_fit]


def build_dataset(kits_dir: Path, neighbors: int, seed: int):
    rng = random.Random(seed)
    by_track, info, emb = load_all(kits_dir)
    ids = [pid for pid in info if pid in emb]
    mat = np.asarray([emb[pid] for pid in ids], dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    mat = mat / norms
    pos_of = {pid: i for i, pid in enumerate(ids)}
    track_of = {pid: info[pid]["track_id"] for pid in ids}

    X: List[List[float]] = []
    y: List[float] = []
    tids: List[str] = []
    n_pos_synth = n_neg = 0
    for tid, entry in by_track.items():
        phrases = entry["phrases"]
        for i, cur_p in enumerate(phrases):
            cur_id = str(cur_p.get("id"))
            if cur_id not in emb:
                continue
            cur = info[cur_id]
            # synthetic donors only: embedding neighbors, other tracks only.
            # Real successors are excluded on purpose: same-track similarity
            # (~0.95) would swamp the cross-track effect the browser needs.
            v = mat[pos_of[cur_id]]
            sims = mat @ v
            order = np.argsort(-sims)
            seen = 0
            for j in order:
                if seen >= neighbors:
                    break
                cand_id = ids[j]
                if track_of[cand_id] == tid:
                    continue
                seen += 1
                cand = info[cand_id]
                # ladder intersect: tempo window + transposable key
                if abs(float(cand["bpm"]) - float(cur["bpm"])) / max(1.0, float(cur["bpm"])) > 0.35:
                    continue
                shift = key_shift(cur["key"], cand["key"])
                if shift is None or abs(shift) > 6:
                    continue
                sim = float(sims[j])
                ok, hfit, kcost = gates(cur, cand)
                X.append(feat_row(sim, cur, cand, hfit, kcost))
                y.append(1.0 if ok else 0.0)
                tids.append(tid)
                if ok:
                    n_pos_synth += 1
                else:
                    n_neg += 1
                if seen >= MAX_SYNTH_PER_CUR:
                    break
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            tids, {"pos_synth": n_pos_synth, "neg": n_neg})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/mix_preference.json"))
    ap.add_argument("--neighbors", type=int, default=NEIGHBORS)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()

    X, y, tids, counts = build_dataset(args.kits_dir, args.neighbors, args.seed)
    print(f"[mix-pref] rows {len(y)} pos_synth {counts['pos_synth']} "
          f"neg {counts['neg']}")
    if len(set(y.tolist())) < 2 or len(X) < 64:
        raise SystemExit("not enough labeled pairs for a track-disjoint split")

    uniq = sorted(set(tids))
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    val_tracks = set(uniq[: max(1, len(uniq) // 10)])
    tr = np.array([t not in val_tracks for t in tids])
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[~tr], y[~tr]

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(class_weight="balanced", max_iter=600,
                               random_state=args.seed)
    model.fit(scaler.transform(Xtr), ytr.astype(int))
    prob_va = model.predict_proba(scaler.transform(Xva))[:, 1]
    auc = float(roc_auc_score(yva, prob_va)) if len(set(yva.tolist())) > 1 else 0.5

    # fold standardization back into raw-space weights for the browser
    w_norm = model.coef_[0].astype(float)
    w_raw = w_norm / scaler.scale_
    b_raw = float(model.intercept_[0] - (scaler.mean_ / scaler.scale_ * w_norm).sum())
    weights = {name: round(float(w), 5) for name, w in zip(FEATURES, w_raw)}
    doc = {
        "format": FORMAT,
        "version": 1,
        "features": FEATURES,
        "weights": weights,
        "bias": round(b_raw, 5),
        "report": {
            "rows": int(len(y)),
            "pos_synth": counts["pos_synth"],
            "neg": counts["neg"],
            "val_tracks": int(len(val_tracks)),
            "val_auc": round(auc, 4),
            "neighbors": args.neighbors,
            "gates": "bass-root {0,5,7} + |dEnergy|<=0.20 + same section",
            "use": "bounded donor prior for synthetic two-song mixes; no audio changes",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"[mix-pref] val_auc {auc:.3f} -> {args.out}")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
