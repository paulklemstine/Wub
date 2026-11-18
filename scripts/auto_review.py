#!/usr/bin/env python3
"""Decide A/B handoffs automatically, using trained judges instead of a human.

The review pack from ``review_slice_flow.py`` is rendered in Python from the
same shipped Opus cells (no Stretch for these short matched-tempo clips), then
scored with the models that were trained on original songs:

* the final acoustic boundary critic (108-d, same as the browser gate),
* the assembled-mix critic (16-d, four stems per bar),
* the motif-return head (does the later bar recall the earlier idea?),
* a seam RMS-dip check on the actual rendered waveform.

The better-scoring variant wins with no listening or clicking required.  The
decisions are written as a preference JSONL that ``train_slice_flow.py`` can
ingest, so the system can also retrain itself from its own judgements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf

from wub_core.slice_flow import STEMS, describe_audio, mix_features, transition_features

SR = 22050
DIM = 32


def decode(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    return mono


def render_variant(variant: dict, base: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Render per-stem tracks and their mono mix for one A/B variant."""
    bpm = max(40.0, float(variant["bpm"]))
    bar_sec = 240.0 / bpm
    total_bars = max(int(p["atBar"] + p["bars"]) for p in variant["parts"])
    length = int(total_bars * bar_sec * SR) + 1
    stems: Dict[str, np.ndarray] = {stem: np.zeros(length, np.float32) for stem in STEMS}
    for part in variant["parts"]:
        stem = part["stem"]
        decoded = [decode(base / f) for f in part["files"]]
        position = int(part["atBar"] * bar_sec * SR)
        for cut in part["slices"]:
            source = decoded[int(cut["itemIndex"])]
            start = int(float(cut["startFrac"]) * len(source))
            end = int(float(cut["endFrac"]) * len(source))
            segment = source[start:end]
            target = max(1, int(float(cut["targetBars"]) * bar_sec * SR))
            if segment.size != target:
                segment = np.interp(np.linspace(0, 1, target),
                                    np.linspace(0, 1, segment.size), segment).astype(np.float32)
            gain = float(cut.get("gain", 1.0))
            end_gain = float(cut.get("endGain", gain))
            segment = segment * np.linspace(gain, end_gain, target, dtype=np.float32)
            stop = min(length, position + target)
            stems[stem][position:stop] += segment[:stop - position]
            position += target
    mix = np.mean(np.stack([stems[s] for s in STEMS]), axis=0)
    return stems, mix


def net_score(net: dict, x: np.ndarray) -> float:
    """Same inference path as frontend/src/slice_flow.ts (tanh hidden)."""
    mean = np.asarray(net["mean"], dtype=np.float64)
    std = np.asarray(net["std"], dtype=np.float64)
    z = (np.asarray(x, dtype=np.float64) - mean) / std
    hidden = np.tanh(np.asarray(net["w1"], dtype=np.float64) @ z + np.asarray(net["b1"]))
    return float((np.asarray(net["w2"], dtype=np.float64) @ hidden + np.asarray(net["b2"]))[0])


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def seam_dip(mix: np.ndarray, seam: int) -> float:
    """Depth of the 20 ms RMS hole around the join, in 0..1."""
    window = max(8, int(0.02 * SR))
    pre = mix[max(0, seam - window):seam]
    post = mix[seam:seam + window]
    before = float(np.sqrt(np.mean(pre ** 2))) if pre.size else 0.0
    after = float(np.sqrt(np.mean(post ** 2))) if post.size else 0.0
    peak = max(before, after, 1e-6)
    return float(abs(before - after) / peak)


def score_variant(variant: dict, base: Path, models: dict) -> dict:
    stems, mix = render_variant(variant, base)
    bpm = max(40.0, float(variant["bpm"]))
    bar_sec = 240.0 / bpm
    seam = int(4 * bar_sec * SR)
    bar_len = int(bar_sec * SR)

    critic = 0.0
    motif = 0.0
    features: List[np.ndarray] = []
    for stem in STEMS:
        track = stems[stem]
        a = describe_audio(track[max(0, seam - bar_len):seam], SR, stem)
        b = describe_audio(track[seam:seam + bar_len], SR, stem)
        if not a.any() or not b.any():
            continue
        x = transition_features(a, b, stem,
                                [0, 1, 1, 1, 0.75, 0, 0.7, 0.25])
        features.append(x)
        critic += sigmoid(net_score(models["critic"], x))
        m = models.get("motif")
        if m:
            y = np.zeros(m["input_dim"], dtype=np.float64)
            y[:DIM] = a
            y[DIM:2 * DIM] = b
            y[2 * DIM:3 * DIM] = np.abs(a - b)
            y[3 * DIM + STEMS.index(stem)] = 1
            y[3 * DIM + len(STEMS) + 1] = 1  # 16-bar return
            y[-1] = 0.75
            motif += sigmoid(net_score(m["net"], y))

    mix_score = 0.0
    count = 0
    for bar in range(4, 8):
        row = np.zeros(16, dtype=np.float64)
        for si, stem in enumerate(STEMS):
            chunk = stems[stem][bar * bar_len:(bar + 1) * bar_len]
            desc = describe_audio(chunk, SR, stem)
            for index, target in enumerate((0, 7, 9, 11)):
                row[si * 4 + index] = desc[target]
        mix_score += sigmoid(net_score(models["mix"], row))
        count += 1
    mix_score /= max(1, count)
    critic /= max(1, len(features))
    motif /= max(1, len(features))
    dip = seam_dip(mix, seam)
    total = 0.55 * critic + 0.25 * mix_score + 0.12 * motif + 0.08 * (1 - min(1, dip * 2))
    return {
        "score": round(total, 5),
        "critic": round(critic, 5),
        "mix": round(mix_score, 5),
        "motif": round(motif, 5),
        "rms_dip": round(dip, 5),
        "features": [f.tolist() for f in features],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kits", default="frontend/public/data/kits")
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--out", default="/tmp/wub-slice-flow/auto_review.jsonl")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    base = Path(args.kits)
    flow = json.loads((base / "slice_flow.json").read_text())
    models = {"critic": flow["critic"], "mix": flow["mix"]}
    motif_file = base / "motif_return.json"
    if motif_file.exists():
        models["motif"] = json.loads(motif_file.read_text())

    from scripts.review_slice_flow import make_reviews
    cases = make_reviews(base, args.cases, args.seed)
    if not cases:
        raise SystemExit("no review cases could be built")

    rows: List[dict] = []
    report: List[dict] = []
    for case in cases:
        scores = {name: score_variant(case[name], base, models) for name in ("a", "b")}
        winner = "a" if scores["a"]["score"] >= scores["b"]["score"] else "b"
        rows.append({
            "id": case["id"],
            "winner": winner,
            "source_tracks": sorted(set(case["a"]["source_tracks"] + case["b"]["source_tracks"])),
            "a": scores["a"]["features"][0],
            "b": scores["b"]["features"][0],
            "origin": "auto-reward-v1",
        })
        report.append({"id": case["id"], "winner": winner,
                       "margin": round(abs(scores["a"]["score"] - scores["b"]["score"]), 4),
                       "a": {k: v for k, v in scores["a"].items() if k != "features"},
                       "b": {k: v for k, v in scores["b"].items() if k != "features"}})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"[auto] {len(rows)} automatic A/B decisions -> {out}")
    for item in report:
        print(f"[auto] {item['id']}: {item['winner'].upper()} "
              f"(margin {item['margin']:.3f}, "
              f"critic {item['a']['critic']:.3f}/{item['b']['critic']:.3f}, "
              f"mix {item['a']['mix']:.3f}/{item['b']['mix']:.3f})")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
