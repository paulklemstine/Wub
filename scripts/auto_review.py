#!/usr/bin/env python3
"""Decide A/B handoffs automatically, using trained judges instead of a human.

The review pack from ``review_slice_flow.py`` is rendered in Python from the
shipping PCM assembler and Signalsmith Stretch in isolated Chrome, then
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
import asyncio
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from wub_core.slice_flow import STEMS, describe_audio, mix_features, transition_features

SR = 44100
DIM = 32


def render_variant(variant: dict, base: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run the same assembler, pitch/time worklet and safety stage as review.

    Extra offline output channels tap the four stereo buses in the SAME render,
    avoiding a surrogate resampler or four additional WASM graphs.
    """
    from scripts.review_slice_flow import Browser, RENDER_JS, ROOT
    async def render():
        async with Browser(kits=base) as browser:
            job = {**variant, "returnStemPCM": True}
            return await asyncio.wait_for(browser.evaluate(RENDER_JS.replace("JOB", json.dumps(job))), 120)
    result = asyncio.run(render())
    pcm = np.frombuffer(base64.b64decode(result["pcm"]), dtype="<f4").reshape(-1, result["channels"])
    if result["sampleRate"] != SR or not np.isfinite(pcm).all():
        raise ValueError("invalid browser render")
    stems = {stem: pcm[:, 2 + i * 2:4 + i * 2] for i, stem in enumerate(STEMS)}
    return stems, pcm[:, :2]


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
    seam = int((.15 + variant.get("seamBar", 4) * bar_sec) * SR)
    bar_len = int(bar_sec * SR)

    critic = 0.0
    motif = 0.0
    features: Dict[str, np.ndarray] = {}
    for stem in STEMS:
        track = stems[stem]
        a = describe_audio(track[max(0, seam - bar_len):seam], SR, stem)
        b = describe_audio(track[seam:seam + bar_len], SR, stem)
        if not a.any() or not b.any():
            continue
        x = transition_features(a, b, stem,
                                [0, 1, 1, 1, 0.75, 0, 0.7, 0.25])
        features[stem] = x
        critic += sigmoid(net_score(models["critic"], x))
        m = models.get("motif")
        returns = []
        if m:
            for lag_index, lag in enumerate((8, 16, 32)):
                anchor_start = seam - lag * bar_len
                if anchor_start < int(.15 * SR):
                    continue
                anchor = describe_audio(track[anchor_start:anchor_start + bar_len], SR, stem)
                y = np.zeros(m["input_dim"], dtype=np.float64)
                y[:DIM] = anchor; y[DIM:2 * DIM] = b
                y[2 * DIM:3 * DIM] = np.abs(anchor - b)
                y[3 * DIM + STEMS.index(stem)] = 1
                y[3 * DIM + len(STEMS) + lag_index] = 1
                y[-1] = lag / 16
                returns.append(sigmoid(net_score(m["net"], y)))
        motif += max(returns) if returns else .5

    mix_score = 0.0
    count = 0
    for bar in range(4, 8):
        row = np.zeros(16, dtype=np.float64)
        for si, stem in enumerate(STEMS):
            start = int(.15 * SR) + bar * bar_len
            chunk = stems[stem][start:start + bar_len]
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
        "features": {stem: f.tolist() for stem, f in features.items()},
        "renderer": "shipping-pcm-signalsmith-chrome",
        "motif_context_available": variant.get("seamBar", 4) >= 8,
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
        margin = scores["a"]["score"] - scores["b"]["score"]
        winner = "tie" if abs(margin) < .005 else ("a" if margin > 0 else "b")
        changed = case["a"].get("changedStem", "other")
        if changed != case["b"].get("changedStem", "other"):
            raise ValueError("A/B attribution requires the same tested stem")
        rows.append({
            "id": case["id"],
            "winner": winner,
            "source_tracks": sorted(set(case["a"]["source_tracks"] + case["b"]["source_tracks"])),
            "a": scores["a"]["features"][changed],
            "b": scores["b"]["features"][changed],
            "origin": "auto-reward-v2", "tested_stem": changed,
            "renderer": "shipping-pcm-signalsmith-chrome",
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
