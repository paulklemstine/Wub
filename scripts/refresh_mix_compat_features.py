#!/usr/bin/env python3
"""Refresh per-bar mix-compat features using the shipped onset sidecar.

Recomputes `features` (and rhythm sub-scores used for aux labels) for every
example in an existing mix_compat_dataset.json using the true per-bar onset
grids from mix_compat_onsets.f32/json, while PRESERVING teacher labels
(label, teacher_manifold, label_heuristic). This aligns training features
with runtime (browser uses the onset sidecar) without re-running the
expensive render+embed/MERT grading.

Usage:
  PYTHONPATH=. python3 scripts/refresh_mix_compat_features.py \
    --dataset /tmp/wubtrain/mix_compat_dataset_40k_mert2/mix_compat_dataset.json \
    --manifest frontend/public/data/kits/kit_manifest.json \
    --onsets frontend/public/data/kits/mix_compat_onsets.json \
    --out /tmp/wubtrain/mix_compat_dataset_40k_mert2_refreshed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_mix_compat_dataset import (
    BARS_PER_PHRASE,
    STEMS,
    heuristic_mix_score,
    per_bar_feature_vector as _canonical_per_bar_feature_vector,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_manifest(path: Path):
    m = json.loads(path.read_text())
    kits = {k["track_id"]: k for k in m.get("kits", []) if isinstance(k.get("track_id"), str)}
    # phrase_id -> {phrase, kit}
    by_phrase = {}
    for kit in m.get("kits", []):
        tid = kit.get("track_id")
        if not isinstance(tid, str):
            continue
        for p in kit.get("phrases") or []:
            pid = p.get("id")
            if isinstance(pid, str):
                by_phrase[pid] = {"phrase": p, "kit": kit, "track_id": tid}
    return kits, by_phrase


def load_onsets(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    f32 = meta_path.parent / meta.get("feature_file", "mix_compat_onsets.f32")
    data = np.fromfile(f32, dtype=np.float32)
    shape = meta.get("shape") or [meta.get("count", 0), 4, meta.get("steps", 64)]
    n, ns, steps = int(shape[0]), int(shape[1]), int(shape[2])
    if data.size < n * ns * steps:
        raise ValueError(f"onset file too small: {data.size} < {n*ns*steps}")
    data = data[: n * ns * steps].reshape(n, ns, steps)
    index = {str(k): int(v) for k, v in (meta.get("index") or {}).items()}
    stems = list(meta.get("stems") or STEMS)
    return data, index, stems


def grid_for(data, index, stems, phrase_id: str, stem: str):
    row = index.get(phrase_id)
    if row is None:
        return None
    try:
        si = stems.index(stem)
    except ValueError:
        return None
    if row < 0 or row >= data.shape[0]:
        return None
    return np.asarray(data[row, si], dtype=np.float32)


# ---- feature math (mirrors build_mix_compat_dataset.py) ----

def _safe_vec(x, n, fill=0.0):
    if not isinstance(x, list):
        return [fill] * n
    out = []
    for i in range(n):
        v = x[i] if i < len(x) else fill
        try:
            v = float(v)
        except Exception:
            v = fill
        if not np.isfinite(v):
            v = fill
        out.append(v)
    return out


def bar_slice(grid64: np.ndarray, bar: int) -> np.ndarray:
    s = bar * 16
    return grid64[s:s + 16]


def rhythm_pair_features(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    eps = 1e-6
    aa = a - a.mean()
    bb = b - b.mean()
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb) + eps)
    corr = float(np.dot(aa, bb) / denom)
    down = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], np.float32)
    off = 1.0 - down
    down_agree = float(np.dot(a * down, b * down) / (np.linalg.norm(a * down) * np.linalg.norm(b * down) + eps))
    off_agree = float(np.dot(a * off, b * off) / (np.linalg.norm(a * off) * np.linalg.norm(b * off) + eps))
    da = float(a.mean())
    db = float(b.mean())
    density_gap = abs(da - db)
    joint_pulse = float(np.mean(a * b))
    sync_a = float(np.dot(a, off) / (np.dot(a, down) + eps))
    sync_b = float(np.dot(b, off) / (np.dot(b, down) + eps))
    sync_gap = abs(sync_a - sync_b)
    return {
        "corr": corr, "down_agree": down_agree, "off_agree": off_agree,
        "density_gap": density_gap, "joint_pulse": joint_pulse, "sync_gap": sync_gap,
    }


def harmonic_features(phrases_by_stem: dict) -> dict:
    def chroma(p):
        c = p.get("chroma")
        if isinstance(c, list) and len(c) == 12:
            v = np.array([float(x) for x in c], np.float32)
            n = np.linalg.norm(v) + 1e-6
            return v / n
        return None
    cb = chroma(phrases_by_stem.get("bass", {}))
    co = chroma(phrases_by_stem.get("other", {}))
    cv = chroma(phrases_by_stem.get("vocals", {}))
    def cos(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))
    rb = phrases_by_stem.get("bass", {}).get("bass_root")
    ro = phrases_by_stem.get("other", {}).get("bass_root")
    try:
        rb = int(rb) if rb is not None else None
        ro = int(ro) if ro is not None else None
    except Exception:
        rb = ro = None
    root_interval_penalty = 0.0
    if rb is not None and ro is not None:
        iv = abs((ro - rb) % 12)
        iv = min(iv, 12 - iv)
        root_interval_penalty = {0: 0.0, 5: 0.05, 7: 0.05, 3: 0.1, 4: 0.1, 2: 0.2, 1: 0.4, 6: 0.6}.get(iv, 0.25)
    return {
        "bass_other_chroma": cos(cb, co),
        "bass_vocal_chroma": cos(cb, cv),
        "other_vocal_chroma": cos(co, cv),
        "root_interval_penalty": root_interval_penalty,
    }


def spectral_balance_features(phrases_by_stem: dict) -> dict:
    rms = {}
    for s in STEMS:
        st = (phrases_by_stem.get(s, {}).get("stems") or {}).get(s, {})
        v = st.get("rms_db")
        try:
            rms[s] = float(v)
        except Exception:
            rms[s] = -24.0
    low_crowd = max(0.0, (rms["drums"] + 18.0) / 18.0) * max(0.0, (rms["bass"] + 18.0) / 18.0)
    vocal_buried = max(0.0, (rms["other"] - rms["vocals"]) / 18.0)
    spread = float(np.std(list(rms.values())))
    return {"low_crowd": low_crowd, "vocal_buried": vocal_buried, "rms_spread": spread, **{f"rms_{s}": rms[s] for s in STEMS}}


def energy_features(phrases_by_stem: dict) -> dict:
    e = []
    for s in STEMS:
        try:
            e.append(float(phrases_by_stem.get(s, {}).get("energy", 0.5)))
        except Exception:
            e.append(0.5)
    e = np.array(e, np.float32)
    return {"energy_mean": float(e.mean()), "energy_std": float(e.std()), "energy_min": float(e.min()), "energy_max": float(e.max())}


def per_bar_feature_vector(bar: int, phrases_by_stem: dict, grids: dict) -> np.ndarray:
    pairs = [("drums", "bass"), ("drums", "other"), ("bass", "other"), ("drums", "vocals"), ("bass", "vocals"), ("other", "vocals")]
    rfeats = []
    corr_vals = []
    for a, b in pairs:
        ga = grids.get(a)
        gb = grids.get(b)
        if ga is None or gb is None:
            f = {"corr": 0.0, "down_agree": 0.0, "off_agree": 0.0, "density_gap": 0.5, "joint_pulse": 0.0, "sync_gap": 0.5}
        else:
            f = rhythm_pair_features(bar_slice(ga, bar), bar_slice(gb, bar))
        corr_vals.append(f["corr"])
        rfeats.extend([f["corr"], f["down_agree"], f["off_agree"], f["density_gap"], f["joint_pulse"], f["sync_gap"]])
    rhythm_agg = [float(np.mean(corr_vals)), float(np.min(corr_vals)), float(np.std(corr_vals))]
    harm = harmonic_features(phrases_by_stem)
    spec = spectral_balance_features(phrases_by_stem)
    ener = energy_features(phrases_by_stem)
    vec = []
    vec.extend(rhythm_agg)
    vec.extend(rfeats)
    vec.extend([harm["bass_other_chroma"], harm["bass_vocal_chroma"], harm["other_vocal_chroma"], harm["root_interval_penalty"]])
    vec.extend([spec["low_crowd"], spec["vocal_buried"], spec["rms_spread"]])
    vec.extend([spec[f"rms_{s}"] for s in STEMS])
    vec.extend([ener["energy_mean"], ener["energy_std"], ener["energy_min"], ener["energy_max"]])
    return np.array(vec, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--onsets", required=True, help="mix_compat_onsets.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("[refresh] loading onset sidecar")
    onset_data, onset_index, onset_stems = load_onsets(Path(args.onsets))
    log(f"[refresh] onsets: {len(onset_index)} phrases, stems={onset_stems}")
    log("[refresh] loading manifest")
    kits, by_phrase = load_manifest(Path(args.manifest))
    log(f"[refresh] manifest phrases: {len(by_phrase)}")
    log("[refresh] loading dataset")
    data = json.loads(Path(args.dataset).read_text())
    examples = data.get("examples") or []
    log(f"[refresh] examples: {len(examples)}")

    refreshed = 0
    missing_grid = 0
    for i, ex in enumerate(examples):
        phr = ex.get("phrases") or {}
        phrases_by_stem = {}
        kit_by_stem = {}
        grids = {}
        ok = True
        for s in STEMS:
            pid = phr.get(s)
            if not pid or pid not in by_phrase:
                ok = False
                break
            phrases_by_stem[s] = by_phrase[pid]["phrase"]
            kit_by_stem[s] = by_phrase[pid]["kit"]
            g = grid_for(onset_data, onset_index, onset_stems, pid, s)
            grids[s] = g
            if g is None:
                missing_grid += 1
        if not ok:
            continue
        per_bar = np.stack([
            _canonical_per_bar_feature_vector(b, phrases_by_stem, kit_by_stem, grids)
            for b in range(BARS_PER_PHRASE)
        ])
        grade = heuristic_mix_score(per_bar)
        ex["features"] = per_bar.astype(np.float32).tolist()
        ex["score"] = grade["mean"] if ex.get("kind") != "real" else max(grade["mean"], 0.85)
        ex["rhythm"] = grade["rhythm_mean"] if ex.get("kind") != "real" else max(grade["rhythm_mean"], 0.85)
        ex["min_bar"] = grade["min"] if ex.get("kind") != "real" else max(grade["min"], 0.80)
        ex["rhythm_per_bar"] = grade["rhythm_per_bar"]
        base_pb = grade["per_bar"]
        if "label" in ex:
            target_mean = float(ex["label"])
            pb_mean = sum(base_pb) / max(1, len(base_pb))
            delta = target_mean - pb_mean
            ex["per_bar"] = [float(max(0.0, min(1.0, v + delta))) for v in base_pb]
        else:
            ex["per_bar"] = base_pb
        refreshed += 1
        if (i + 1) % 10000 == 0:
            log(f"[refresh] {i+1}/{len(examples)} refreshed={refreshed}")

    summary = dict(data.get("summary") or {})
    summary["features_refreshed"] = True
    summary["features_source"] = "shipped_mix_compat_onsets (spectral-flux 64-step, recomputed from kit audio)"
    summary["onset_phrase_count"] = len(onset_index)
    summary["refreshed_examples"] = refreshed
    summary["missing_grid_lookups"] = missing_grid
    out = {"summary": summary, "examples": examples}
    (out_dir / "mix_compat_dataset.json").write_text(json.dumps(out))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[refresh] wrote {out_dir} refreshed={refreshed} missing_grids={missing_grid}")


if __name__ == "__main__":
    main()
