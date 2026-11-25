"""Train a multi-scale 'song-ness' critic from real four-stem windows.

Positives: consecutive 4-bar windows where all four stems come from the same
real song. Negatives: cross-song recombinations of the same kind the browser
makes (swap one or two stems with a similar-key/tempo/section song, or reorder
bars). Track-disjoint split, so validation songs are never seen in training.

The shipped browser model scores the assembled quartet at the final gate and
can only add a bounded adjustment; it never renders or reschedules audio.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
PITCHED = ("bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
PATTERN_DIM = 48
INPUT_DIM = 282  # 218 analysis features + 64 learned phrase embedding
HIDDEN = 64
FORMAT = "wub-song-critic-v1"


def song_data(doc: Dict[str, Any]):
    segs, tats = doc.get("segments") or [], doc.get("tatums") or []
    n = len(tats) // 16
    if n < 12:
        return None
    feats = bar_features(segs, tats)
    spans = segment_sections(feats)
    labels = [s.label for s in spans for _ in range(s.n_bars)]
    labels = (labels + ["?"] * n)[:n]
    by_id = {str(s.get("id")): s for s in segs if s.get("id")}
    prev = {s: None for s in STEMS}
    bars = []
    for b in range(n):
        union = {s: 0 for s in STEMS}
        present = {s: 0 for s in STEMS}
        chroma: Dict[str, List[np.ndarray]] = {s: [] for s in PITCHED}
        louds = []
        for step in range(16):
            ss = (tats[b * 16 + step].get("stem_segments") or {})
            for stem in STEMS:
                ids = ss.get(stem, []) or []
                cur = ids[-1] if ids else None
                if cur is not None:
                    present[stem] += 1
                on = (step == 0 and cur is not None) or (cur is not None and cur != prev[stem])
                if on:
                    union[stem] |= 1 << step
                    seg = by_id.get(str(cur))
                    if seg is not None:
                        louds.append(float(seg.get("loudness_max", -20.0)))
                        if stem in chroma:
                            p = np.asarray(seg.get("pitches") or [0.0] * 12, dtype=float)[:12]
                            if p.sum() > 1e-8:
                                chroma[stem].append(p / p.sum())
                prev[stem] = cur
        bars.append({
            "mask": union, "present": present,
            "chroma": {s: (np.mean(np.stack(chroma[s]), axis=0) if chroma[s] else None)
                       for s in PITCHED},
            "energy": float(np.mean(louds)) if louds else -60.0,
            "section": labels[b],
        })
    return bars


def pattern_vec(mask: int, present: int) -> np.ndarray:
    v = np.zeros(PATTERN_DIM, dtype=np.float32)
    for step in range(16):
        v[step] = (mask >> step) & 1
    count = int(v[:16].sum())
    v[16] = count / 16.0
    v[17] = sum((mask >> s) & 1 for s in (0, 4, 8, 12)) / 4.0
    v[18] = sum((mask >> s) & 1 for s in (1, 3, 5, 7, 9, 11, 13, 15)) / 8.0
    for q in range(4):
        v[19 + q] = sum((mask >> s) & 1 for s in range(q * 4, q * 4 + 4)) / 4.0
    v[47] = 1.0 if present > 0 else 0.0
    return v


def window(chunk, bpm: float, key: str):
    if len(chunk) < 4:
        return None
    masks = {s: 0 for s in STEMS}
    presence = {s: 0 for s in STEMS}
    chroma = {s: [] for s in PITCHED}
    energy = []
    for bar in chunk:
        for stem in STEMS:
            masks[stem] |= bar["mask"][stem]
            presence[stem] += bar["present"][stem]
        for stem in PITCHED:
            if bar["chroma"][stem] is not None:
                chroma[stem].append(bar["chroma"][stem])
        energy.append(bar["energy"])
    cvecs = {s: (np.mean(np.stack(chroma[s]), axis=0) if chroma[s] else np.zeros(12))
             for s in PITCHED}
    return {
        "pattern": {s: pattern_vec(masks[s], presence[s]) for s in STEMS},
        "presence": {s: presence[s] / 64.0 for s in STEMS},
        "chroma": cvecs,
        "mask": masks,
        "energy": float(np.mean(energy)),
        "section": chunk[0]["section"],
        "bpm": bpm, "key": key,
    }


def cosine(a, b):
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


def features(w, prev_energy=None, embedding=None):
    v: List[float] = []
    for s in STEMS:
        v.extend(w["pattern"][s].tolist())
        v.append(w["presence"][s])
        v.append(float(np.clip((w["energy"] + 60) / 60, 0, 1)))
    for s in PITCHED:
        v.append(float((w["chroma"][s] > 0).any()))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        v.append(cosine(w["chroma"][PITCHED[i]], w["chroma"][PITCHED[j]]))
        a, b = w["mask"][PITCHED[i]], w["mask"][PITCHED[j]]
        v.append(((a & b).bit_count() / (a | b).bit_count()) if (a or b) else 0.0)
    sec = w["section"] if w["section"] in SECTIONS else "GROOVE_A"
    for name in SECTIONS:
        v.append(1.0 if name == sec else 0.0)
    v.append(float(np.clip((w["energy"] + 60) / 60, 0, 1)))
    delta = 0.0 if prev_energy is None else (w["energy"] - prev_energy) / 20.0
    v.append(float(np.clip(delta, -1, 1)))
    v.append(min(1.0, w["bpm"] / 200.0))
    v.append(0.0)  # key distance placeholder (packed in embedding at runtime)
    if embedding is None or len(embedding) == 0:
        embedding = [0.0] * 64
    v.extend(float(x) for x in embedding)
    assert len(v) == INPUT_DIM, len(v)
    return np.asarray(v, dtype=np.float32)


def key_root(s: str) -> int:
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
           "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    head = s.split()[0] if s else "C"
    return pcs.get(head.replace("m", ""), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="phrase_embeddings.json to append to the 218-d critic features")
    args = ap.parse_args()
    torch.manual_seed(911)
    np.random.seed(911)
    random.seed(911)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_map: Dict[str, List[float]] = {}
    if args.embeddings:
        emb_map = json.loads(args.embeddings.read_text()).get("phrases", {})
    phrase_ids: Dict[str, List[str]] = {}
    if args.embeddings:
        manifest = json.loads((args.kits_dir / "kit_manifest.json").read_text())
        for kit in manifest.get("kits", []):
            phrase_ids[kit["track_id"]] = [p["id"] for p in sorted(
                kit.get("phrases", []), key=lambda x: x.get("start_bar", 0))]

    def emb_for(tid: str, idx: int) -> List[float]:
        if not args.embeddings:
            return []
        ids = phrase_ids.get(tid, [])
        return emb_map.get(ids[idx], []) if idx < len(ids) else []

    paths = sorted(args.processed.glob("*.json"))
    if args.max_tracks > 0:
        paths = paths[::max(1, len(paths) // args.max_tracks)][:args.max_tracks]
    songs: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        try:
            doc = json.loads(path.read_text())
            bars = song_data(doc)
        except Exception:
            continue
        if not bars:
            continue
        bpm = float(doc.get("bpm") or 120)
        key = str(doc.get("key") or "")
        windows = [window(bars[c * 4:c * 4 + 4], bpm, key)
                   for c in range(len(bars) // 4)]
        songs[path.stem] = {"windows": [w for w in windows if w], "doc": doc}
    tracks = sorted(songs)
    rng = random.Random(911)
    rng.shuffle(tracks)
    val_tracks = set(tracks[:max(1, len(tracks) // 10)])

    # Group positive windows by section and key/tempo for plausible negatives.
    pool: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for tid, song in songs.items():
        for i, w in enumerate(song["windows"]):
            pool.setdefault((w["section"], key_root(song["doc"].get("key", "")) // 2), []).append((tid, i))

    xs, ys, track_of = [], [], []
    real_scores: List[float] = []
    for tid, song in songs.items():
        wins = song["windows"]
        for i, w in enumerate(wins):
            prev = wins[i - 1]["energy"] if i > 0 else None
            xs.append(features(w, prev, emb_for(tid, i))); ys.append(1.0); track_of.append(tid)
            # Cross-song recombination: replace 1-2 stems with a similar song.
            key = (w["section"], key_root(song["doc"].get("key", "")) // 2)
            donors = [d for d in pool.get(key, []) if d[0] != tid]
            for _ in range(2):
                if not donors:
                    break
                did, di = rng.choice(donors)
                donor = songs[did]["windows"][di]
                mixed = dict(w)
                mixed["pattern"] = dict(w["pattern"])
                mixed["chroma"] = dict(w["chroma"])
                mixed["mask"] = dict(w["mask"])
                for stem in rng.sample(STEMS, 2):
                    mixed["pattern"][stem] = donor["pattern"][stem]
                    mixed["mask"][stem] = donor["mask"][stem]
                    if stem in PITCHED:
                        mixed["chroma"][stem] = donor["chroma"][stem]
                xs.append(features(mixed, prev, emb_for(did, di))); ys.append(0.0); track_of.append(tid)
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    val = np.asarray([t in val_tracks for t in track_of], dtype=bool)
    tr = ~val
    mean = x[tr].mean(axis=0).astype(np.float32)
    std = x[tr].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    xt = torch.tensor((x[tr] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
    xv = torch.tensor((x[val] - mean) / std, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN), nn.Tanh(),
                        nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.0015)
    for _ in range(args.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 512):
            idx = perm[s:s + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(xv).squeeze(1).cpu().numpy()
        pa = net(xt).squeeze(1).cpu().numpy()
    auc = float(roc_auc_score(y[val], pv))
    real_pos = net(torch.tensor((x[y > 0.5] - mean) / std, dtype=torch.float32, device=device))
    with torch.no_grad():
        real_scores = torch.sigmoid(real_pos).squeeze(1).cpu().numpy().tolist()
    floor = float(np.percentile(real_scores, 10))

    def layer(mod):
        return {"w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
                "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()]}

    doc = {
        "format": FORMAT, "version": 1,
        "input_dim": INPUT_DIM, "hidden": HIDDEN,
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [layer(net[0]), layer(net[2]), layer(net[4])],
        "report": {
            "source": "data/processed real 4-stem windows vs cross-song recombinations",
            "songs": len(songs), "tracks": len(tracks),
            "train_tracks": len(tracks) - len(val_tracks), "val_tracks": len(val_tracks),
            "positive_examples": int((y > 0.5).sum()), "negative_examples": int((y < 0.5).sum()),
            "val_auc": round(auc, 4), "real_score_p10": round(floor, 4),
            "epochs": args.epochs, "device": str(device),
            "embeddings": "learned-contrastive" if args.embeddings else "none",
            "use": "bounded final-gate adjustment; never schedules or renders audio",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps(doc["report"], indent=2))


if __name__ == "__main__":
    main()
