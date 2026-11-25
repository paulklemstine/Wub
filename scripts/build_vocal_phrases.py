#!/usr/bin/env python3
"""Mine multi-bar vocal phrase units from real songs and learn which loop.

The shipped bank stores 4-bar cells, but real vocals develop over 8-32 bars:
a verse line, a pre-chorus, a chorus hook.  Playing one 4-bar cell, or worse
one active bar of it, loses the performance.  This script:

* finds every contiguous chain of shipped phrases in a track (4/8/12/16/24
  bars of one continuous performance),
* scores each chain's internal activity (coverage, variety, silence gaps,
  loop seam) from the cell index,
* measures whether the same melodic window recurs elsewhere in the song
  (chorus/hook recurrence) from the full analysis JSON, and
* trains a small track-disjoint logistic model that predicts recurrence from
  features the browser can compute without the full-song audio.

The exported ``vocal_phrases.json`` lets the composer hold a whole learned
phrase unit and loop it, instead of stitching single bars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.build_vocal_flow import vocal_bars

MIN_BARS = 8
MAX_BARS = 32
FEATURE_NAMES = [
    "bars32", "coverage", "variety", "seam", "gap", "density",
    "mean_purity", "hook", "active_frac", "root_stability", "energy_span",
]


def is_val(track: str) -> bool:
    return int(hashlib.sha256(track.encode()).hexdigest()[:8], 16) % 8 == 0


def track_chains(phrases: Sequence[dict]) -> List[List[dict]]:
    ordered = sorted(phrases, key=lambda p: int(p.get("start_bar") or 0))
    chains: List[List[dict]] = []
    run: List[dict] = []
    for phrase in ordered:
        if run and int(phrase.get("start_bar") or 0) != int(run[-1].get("start_bar") or 0) + 4:
            if len(run) >= 2:
                chains.append(run)
            run = []
        run.append(phrase)
    if len(run) >= 2:
        chains.append(run)
    return chains


def bar_signature(mask: int, root: int) -> np.ndarray:
    vec = np.zeros(28, dtype=np.float32)
    for i in range(16):
        vec[i] = 1.0 if (mask >> i) & 1 else 0.0
    if root >= 0:
        vec[16 + (root % 12)] = 1.5
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def full_song_signatures(doc: dict) -> Tuple[np.ndarray, np.ndarray]:
    steps, chroma = vocal_bars(doc)
    count = steps.shape[0]
    sig = np.zeros((count, 28), dtype=np.float32)
    active = np.zeros(count, dtype=bool)
    for b in range(count):
        mask = 0
        for i in range(16):
            if steps[b, i]:
                mask |= 1 << i
        root = int(np.argmax(chroma[b])) if chroma[b].any() else -1
        sig[b] = bar_signature(mask, root)
        active[b] = mask.bit_count() >= 2
    return sig, active


def recurrence_of(sig: np.ndarray, start: int, length: int) -> float:
    """Best cosine similarity of this L-bar window against any other window."""
    count = sig.shape[0]
    if start < 0 or start + 2 > count:
        return 0.0
    length = int(min(length, count - start))
    unit = sig[start:start + length]
    if unit.shape[0] < 2:
        return 0.0
    best = 0.0
    for cand in range(0, count - length + 1):
        if abs(cand - start) < max(4, length // 2):
            continue
        other = sig[cand:cand + length]
        if other.shape[0] != unit.shape[0]:
            continue
        sim = float(np.mean(np.sum(unit * other, axis=1)))
        if sim > best:
            best = sim
    return best


def unit_from_chain(
    window: Sequence[dict],
    unit_start_bar: int,
    sig: np.ndarray,
    cell: Dict[str, dict],
) -> dict:
    bars = len(window) * 4
    masks: List[int] = []
    roots: List[int] = []
    purities: List[float] = []
    for phrase in window:
        meta = (cell.get(str(phrase.get("id"))) or {}).get("vocals") or {}
        onset = meta.get("o") or [0, 0, 0, 0]
        root = meta.get("r") or [-1, -1, -1, -1]
        purity = meta.get("p") or [0.0, 0.0, 0.0, 0.0]
        for b in range(4):
            masks.append(int(onset[b]))
            roots.append(int(root[b]))
            purities.append(float(purity[b]))
    active_bars = [i for i, mask in enumerate(masks) if int(mask).bit_count() >= 2]
    coverage = len(active_bars) / max(1, bars)
    signatures = [(masks[i], roots[i]) for i in active_bars]
    variety = len(set(signatures)) / max(1, len(active_bars))
    dense = [int(mask).bit_count() for mask in masks if int(mask).bit_count() >= 2]
    density = float(np.mean(dense)) / 8.0 if dense else 0.0
    # Longest silent stretch inside the unit, normalized.
    gap = 0
    streak = 0
    for mask in masks:
        if int(mask).bit_count() >= 2:
            streak = 0
        else:
            streak += 1
            gap = max(gap, streak)
    norm_gap = gap / max(1, bars)
    # Loop seam: average cosine of consecutive active bars including wrap.
    seam = 0.0
    if len(active_bars) >= 2:
        pair_sum = 0.0
        pair_count = 0
        for a, b in zip(active_bars, active_bars[1:] + active_bars[:1]):
            vec_a = bar_signature(masks[a], roots[a])
            vec_b = bar_signature(masks[b], roots[b])
            pair_sum += float(np.dot(vec_a, vec_b))
            pair_count += 1
        seam = pair_sum / max(1, pair_count)
    mean_purity = float(np.mean([purities[i] for i in active_bars])) if active_bars else 0.0
    root_stability = (
        max(np.bincount([roots[i] % 12 for i in active_bars]).tolist()) / len(active_bars)
        if active_bars else 0.0
    )
    hook = 1.0 if any(
        ((cell.get(str(phrase.get("id"))) or {}).get("vocals") or {}).get("hook")
        for phrase in window
    ) else 0.0
    energies = [float(phrase.get("energy") or 0.0) for phrase in window]
    energy_span = float(np.std(energies)) if energies else 0.0
    recurrence = recurrence_of(sig, unit_start_bar, bars)
    features = [
        bars / 32.0, coverage, variety, seam, norm_gap, density,
        mean_purity, hook, coverage, root_stability, energy_span,
    ]
    return {
        "ids": [str(p.get("id")) for p in window],
        "bars": bars,
        "start_bar": int(window[0].get("start_bar") or 0),
        "features": [round(float(v), 5) for v in features],
        "recurrence": round(float(recurrence), 4),
        "repeat": 1 if recurrence >= 0.80 and coverage >= 0.35 else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--cell-index", default="frontend/public/data/kits/cell_index.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="frontend/public/data/kits/vocal_phrases.json")
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=505)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    cell = json.loads(Path(args.cell_index).read_text()).get("phrases", {})
    processed = Path(args.processed)
    kits = manifest.get("kits", [])
    if args.max_tracks > 0:
        kits = kits[: args.max_tracks]

    units_by_track: Dict[str, List[dict]] = {}
    train_x: List[List[float]] = []
    train_y: List[int] = []
    val_x: List[List[float]] = []
    val_y: List[int] = []
    skipped = 0

    for kit in kits:
        track = str(kit.get("track_id"))
        phrases = kit.get("phrases") or []
        chains = track_chains(phrases)
        if not chains:
            continue
        doc_path = processed / f"{track}.json"
        if not doc_path.exists():
            skipped += 1
            continue
        try:
            doc = json.loads(doc_path.read_text())
        except (OSError, ValueError):
            skipped += 1
            continue
        sig, active = full_song_signatures(doc)
        units: List[dict] = []
        val = is_val(track)
        for chain in chains:
            max_len = min(len(chain), MAX_BARS // 4)
            min_len = max(2, MIN_BARS // 4)
            for span in range(min_len, max_len + 1):
                for offset in range(0, len(chain) - span + 1):
                    window = chain[offset:offset + span]
                    unit = unit_from_chain(
                        window, int(window[0].get("start_bar") or 0), sig, cell,
                    )
                    if unit["recurrence"] <= 0 and unit["bars"] < 8:
                        continue
                    units.append(unit)
                    (val_x if val else train_x).append(unit["features"])
                    (val_y if val else train_y).append(unit["repeat"])
        if units:
            units.sort(key=lambda u: (-u["recurrence"], -u["bars"]))
            units_by_track[track] = units

    if len(train_x) < 64 or len(val_x) < 16:
        raise SystemExit("not enough phrase units for a track-disjoint split")

    x_train = np.asarray(train_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.int32)
    x_val = np.asarray(val_x, dtype=np.float32)
    y_val = np.asarray(val_y, dtype=np.int32)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        class_weight="balanced", max_iter=600, random_state=args.seed,
    ).fit(scaler.transform(x_train), y_train)
    prob_val = model.predict_proba(scaler.transform(x_val))[:, 1]
    auc = float(roc_auc_score(y_val, prob_val)) if len(set(y_val.tolist())) > 1 else 0.5
    weights = model.coef_[0].astype(float).tolist()
    bias = float(model.intercept_[0])

    def score_unit(unit: dict) -> float:
        z = (np.asarray(unit["features"], dtype=np.float32) - scaler.mean_) / scaler.scale_
        logit = float(np.dot(weights, z) + bias)
        prob = 1.0 / (1.0 + np.exp(-logit))
        # The learned chorus/recurrence probability is the main term; the
        # measured recurrence and seam keep a plain but well-formed phrase
        # usable when the classifier is uncertain.
        return float(np.clip(0.62 * prob + 0.26 * unit["recurrence"] + 0.12 * unit["features"][3], 0.0, 1.0))

    for track, units in list(units_by_track.items()):
        for unit in units:
            unit["score"] = round(score_unit(unit), 4)
        units.sort(key=lambda u: (-u["score"], -u["bars"]))
        # Keep the artifact compact: one strong window per length band (up to
        # two per band), preserving both short choruses and long verses.
        kept: List[dict] = []
        per_band: Dict[int, int] = {}
        for unit in units:
            band = int(unit["bars"]) // 8
            if per_band.get(band, 0) >= 2:
                continue
            per_band[band] = per_band.get(band, 0) + 1
            kept.append(unit)
            if len(kept) >= 10:
                break
        units_by_track[track] = kept

    artifact = {
        "format": "wub-vocal-phrases-v1",
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "mean": [round(float(v), 6) for v in scaler.mean_.tolist()],
        "scale": [round(float(v), 6) for v in scaler.scale_.tolist()],
        "weights": [round(float(v), 6) for v in weights],
        "bias": round(bias, 6),
        "tracks": units_by_track,
        "report": {
            "tracks": len(units_by_track),
            "units": int(sum(len(v) for v in units_by_track.values())),
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "positive_rate": round(float(y_train.mean()), 4),
            "val_auc": round(auc, 4),
            "skipped_tracks": skipped,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(
        f"[vocal-phrases] {artifact['report']['units']} units / "
        f"{artifact['report']['tracks']} tracks, recurrence AUC {auc:.3f} -> {out}"
    )
    print(f"[vocal-phrases] positive rate {artifact['report']['positive_rate']}, skipped {skipped}")


if __name__ == "__main__":
    main()
