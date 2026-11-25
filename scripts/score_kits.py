#!/usr/bin/env python3
"""Score every kit with the Wub taste models.

Two models are used:

* ``data/taste_model.pt`` - the 8-window real/fake arrangement classifier.
  It is kept for the overall arrangement score (real-vs-shuffle AUC) but is
  no longer used for phrase transitions: a CLS classifier barely reacts to
  swapping one candidate window.
* ``data/taste_ranker.pt`` - the pairwise transition ranker.  A
  Transformer context encoder (last 1..6 windows) and a candidate MLP are
  trained with InfoNCE so the real next window ranks above same-track and
  same-style distractors.  Cosine scores drive the transition matrices,
  phrase quality and next-kit compatibility.

For each kit (track) in ``frontend/public/data/kits/kit_manifest.json`` the
script writes:

* ``transition`` - row-stochastic n x n phrase-transition matrix from ranker
  cosine scores.  A single global softmax temperature is tuned so the median
  row max probability is ~0.52 (not uniform, not one-hot).
* ``quality``    - each phrase scored as the successor of its own real
  aligned-window context, then z-scored within its style and squashed with a
  sigmoid, so wave is not penalised by a global style offset.
* ``next``       - top-10 next kits, first filtered to the same style and
  ``|delta bpm| <= 10``; if fewer than 10 survive the BPM window is relaxed
  to 20 for that kit (recorded in the report).
* ``phrase_order`` - unchanged manifest order.

The report at ``/tmp/taste_report.json`` also contains before/after retrieval
metrics, within-kit top-1/top-3 versus the old classifier, matrix sharpness,
per-style quality stats, and sample rows.

Usage:
    python3 scripts/score_kits.py
    python3 scripts/score_kits.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_taste_model import (  # noqa: E402
    CTX_LEN, REPO_ROOT, SEQ_LEN, PairwiseRanker, RankerData,
    TasteTransformer, build_ranker_data, build_track_data,
    encode_ranker_candidates, encode_ranker_contexts, load_dataset, predict,
    set_seed, synthetic_dataset, train_model, train_ranker_model,
)

CLASSIFIER_TEMPERATURE = 0.7
CLASSIFIER_CTX = 6
TARGET_ROW_MAX = 0.52
MAX_JSON_BYTES = 5 * 1024 * 1024


# --------------------------------------------------------------------------
# model loading / features
# --------------------------------------------------------------------------
def load_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TasteTransformer(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def load_ranker(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PairwiseRanker(**ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def score_sequences(model: TasteTransformer, seqs: np.ndarray,
                    device: torch.device, batch_size: int = 1024
                    ) -> np.ndarray:
    probs, _ = predict(model, seqs, None, device, batch_size)
    return probs.astype(np.float64)


def phrase_feature_map(dataset: dict) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = {}
    for row in dataset.get("kit_phrases") or []:
        key = (str(row["track_id"]), int(round(float(row["start_bar"]))))
        feat = np.asarray(row.get("features") or [], dtype=np.float32).reshape(-1)
        out[key] = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def nearest_window_feature(track, bar: int) -> np.ndarray | None:
    if track is None or len(track.feats) == 0:
        return None
    k = int(np.argmin(np.abs(track.bars - int(bar))))
    return track.feats[k]


def build_kits(manifest: dict, dataset: dict, feature_dim: int,
               tracks_by_id: dict | None = None) -> tuple[list[dict], int]:
    """Resolve phrase features and metadata for every manifest kit."""
    if tracks_by_id is None:
        tracks, _ = build_track_data(dataset)
        tracks_by_id = {t.track_id: t for t in tracks}
    pf = phrase_feature_map(dataset)
    kits: list[dict] = []
    missing = 0
    for entry in manifest.get("kits") or []:
        tid = str(entry["track_id"])
        feats: list[np.ndarray] = []
        order: list[str] = []
        for p in entry.get("phrases") or []:
            bar = int(round(float(p.get("start_bar", 0))))
            order.append(str(p.get("id", f"{tid}_{bar}")))
            feat = pf.get((tid, bar))
            if feat is None:
                feat = nearest_window_feature(tracks_by_id.get(tid), bar)
                missing += 1
            if feat is None:
                feat = np.zeros(feature_dim, dtype=np.float32)
            elif feat.size != feature_dim:
                raise ValueError(
                    f"kit {tid} phrase {bar}: feature dim {feat.size} != "
                    f"{feature_dim}")
            feats.append(feat.astype(np.float32))
        if not feats:
            continue
        kits.append({
            "track_id": tid,
            "style": str(entry.get("style", "unknown")),
            "bpm": float(entry.get("bpm", 0.0)),
            "key": str(entry.get("key", "")),
            "phrase_order": order,
            "feats": np.stack(feats).astype(np.float32),
            "bars": [int(round(float(p.get("start_bar", 0))))
                     for p in entry.get("phrases") or []],
        })
    return kits, missing


def _context(feats: np.ndarray, end_idx: int,
             max_len: int = CTX_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Right-aligned context ending at ``end_idx`` (inclusive)."""
    length = min(max_len, end_idx + 1)
    ctx = np.zeros((max_len, feats.shape[1]), dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    ctx[max_len - length:] = feats[end_idx - length + 1:end_idx + 1]
    mask[max_len - length:] = True
    return ctx, mask


def _context_for_classifier(data: RankerData, row: int,
                            max_ctx: int = 6) -> np.ndarray:
    """Last up to 6 real context windows for the old 8-token classifier."""
    length = int(data.mask[row].sum())
    actual = data.ctx[row, data.ctx.shape[1] - length:]
    if length > max_ctx:
        actual = actual[-max_ctx:]
        length = max_ctx
    if length < max_ctx:
        pad = np.repeat(actual[:1], max_ctx - length, axis=0)
        actual = np.concatenate([pad, actual], axis=0)
    return actual


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------
def classifier_transition_matrices(model: TasteTransformer, kits: list[dict],
                                   device: torch.device, batch_size: int
                                   ) -> list[np.ndarray]:
    """Old 3-context + candidate + 4 repeats classifier matrix (baseline)."""
    seqs: list[np.ndarray] = []
    owners: list[int] = []
    for ki, kit in enumerate(kits):
        feats = kit["feats"]
        n = len(feats)
        ctx = np.empty((n, 3), dtype=np.int64)
        for i in range(n):
            for c in range(3):
                j = i - 2 + c
                ctx[i, c] = j if j >= 0 else i
        for i in range(n):
            for j in range(n):
                seq = np.empty((SEQ_LEN, feats.shape[1]), dtype=np.float32)
                seq[:3] = feats[ctx[i]]
                seq[3:] = feats[j]
                seqs.append(seq)
                owners.append(ki)
    if not seqs:
        return [np.zeros((0, 0), dtype=np.float32) for _ in kits]
    scores = score_sequences(model, np.stack(seqs), device, batch_size)
    owners_arr = np.asarray(owners)
    out: list[np.ndarray] = []
    for ki, kit in enumerate(kits):
        n = len(kit["feats"])
        mat = scores[owners_arr == ki].reshape(n, n)
        z = mat / CLASSIFIER_TEMPERATURE
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        out.append(p)
    return out


def ranker_transition_scores(model: PairwiseRanker, kits: list[dict],
                             device: torch.device) -> list[np.ndarray]:
    """Cosine score matrix for every kit: context ending at i vs phrase j."""
    out: list[np.ndarray] = []
    max_len = int(model.cfg.get("max_len", CTX_LEN))
    for kit in kits:
        feats = kit["feats"]
        n = len(feats)
        ctx = np.empty((n, max_len, feats.shape[1]), dtype=np.float32)
        mask = np.empty((n, max_len), dtype=bool)
        for i in range(n):
            ctx[i], mask[i] = _context(feats, i, max_len)
        h = encode_ranker_contexts(model, ctx, mask, device, max(n, 32))
        c = encode_ranker_candidates(model, feats, device, max(n, 32))
        scores = h @ c.T
        # A next-phrase matrix should not self-loop: without this the context
        # (which contains phrase i) makes phrase i the trivial top candidate.
        np.fill_diagonal(scores, -1e6)
        out.append(scores)
    return out


def _softmax_rows(row: np.ndarray, temperature: float) -> np.ndarray:
    z = row / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def tune_temperature(mats: list[np.ndarray],
                     target: float = TARGET_ROW_MAX) -> tuple[float, dict]:
    """Global softmax T so the median row max probability is ~target."""
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for mat in mats:
        n = len(mat)
        if n > 1:
            groups[n].append(mat)

    def median_max(temperature: float) -> float:
        vals = []
        for arrs in groups.values():
            s = np.stack(arrs) / temperature
            s = s - s.max(axis=2, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=2, keepdims=True)
            vals.append(p.max(axis=2).ravel())
        if not vals:
            return 1.0
        return float(np.median(np.concatenate(vals)))

    if not groups:
        return 1.0, {"median_row_max": 1.0, "target": target}
    lo, hi = 1e-4, 50.0
    # larger T -> softer rows -> lower max
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if median_max(mid) > target:
            lo = mid
        else:
            hi = mid
    temperature = math.sqrt(lo * hi)
    return temperature, {
        "median_row_max": round(median_max(temperature), 6),
        "target": target,
        "search_bounds": [1e-4, 50.0],
    }


def apply_temperature(mats: list[np.ndarray], temperature: float
                      ) -> list[np.ndarray]:
    return [np.stack([_softmax_rows(row, temperature) for row in mat])
            for mat in mats]


def evaluate_within_kit(mats: list[np.ndarray],
                        kits: list[dict], exclude_self: bool = False) -> dict:
    """Top-1/top-3 for rows whose +4-bar successor is another kit phrase."""
    top1 = top3 = total = 0
    for kit, mat in zip(kits, mats):
        bars = kit["bars"]
        n = len(bars)
        for i in range(n):
            succ = [j for j in range(n) if j != i and bars[j] == bars[i] + 4]
            if not succ:
                continue
            j = succ[0]
            row = mat[i]
            if exclude_self:
                row = row.copy()
                row[i] = -1e9
            order = np.argsort(-row)
            total += 1
            if int(order[0]) == j:
                top1 += 1
            if j in set(int(x) for x in order[:3]):
                top3 += 1
    return {
        "top1_count": int(top1),
        "top3_count": int(top3),
        "n": int(total),
        "top1": round(top1 / max(total, 1), 6),
        "top3": round(top3 / max(total, 1), 6),
    }


def matrix_sharpness(mats: list[np.ndarray]) -> dict:
    maxes: list[float] = []
    entropies: list[float] = []
    near_uniform = 0
    total = 0
    for mat in mats:
        n = len(mat)
        if n == 0:
            continue
        uniform = 1.0 / n
        for row in mat:
            total += 1
            m = float(row.max())
            maxes.append(m)
            ent = float(-(row * np.log(np.clip(row, 1e-12, 1.0))).sum())
            entropies.append(ent)
            if m <= uniform + 0.02:
                near_uniform += 1
    arr = np.asarray(maxes)
    ent = np.asarray(entropies)
    return {
        "rows": int(total),
        "row_max_median": round(float(np.median(arr)), 6) if total else float("nan"),
        "row_max_mean": round(float(arr.mean()), 6) if total else float("nan"),
        "row_max_p10": round(float(np.quantile(arr, 0.1)), 6) if total else float("nan"),
        "row_max_p90": round(float(np.quantile(arr, 0.9)), 6) if total else float("nan"),
        "row_max_min": round(float(arr.min()), 6) if total else float("nan"),
        "row_max_max": round(float(arr.max()), 6) if total else float("nan"),
        "entropy_mean": round(float(ent.mean()), 6) if total else float("nan"),
        "entropy_median": round(float(np.median(ent)), 6) if total else float("nan"),
        "near_uniform_rows": int(near_uniform),
        "near_uniform_fraction": round(near_uniform / max(total, 1), 6),
    }


# --------------------------------------------------------------------------
# quality (ranker + per-style calibration)
# --------------------------------------------------------------------------
def ranker_quality(model: PairwiseRanker, kits: list[dict],
                   tracks_by_id: dict, device: torch.device
                   ) -> tuple[list[list[float]], np.ndarray, list[str],
                              dict, int]:
    """Score every phrase as successor of its own real aligned context."""
    ctxs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    cands: list[np.ndarray] = []
    raw = np.full(sum(len(k["feats"]) for k in kits), np.nan, dtype=np.float64)
    styles: list[str] = []
    valid_idx: list[int] = []
    fallback = 0
    phrase_at = 0
    max_len = int(model.cfg.get("max_len", CTX_LEN))
    for kit in kits:
        feats = kit["feats"]
        track = tracks_by_id.get(kit["track_id"])
        for i, bar in enumerate(kit["bars"]):
            ctx = np.zeros((max_len, feats.shape[1]), dtype=np.float32)
            mask = np.zeros(max_len, dtype=bool)
            if track is not None and len(track.feats):
                k = int(np.searchsorted(track.bars, int(bar), side="left")) - 1
                if k >= 0:
                    idxs = [k]
                    while (len(idxs) < max_len and idxs[-1] - 1 >= 0
                           and int(track.bars[idxs[-1]]
                                   - track.bars[idxs[-1] - 1]) == 4):
                        idxs.append(idxs[-1] - 1)
                    idxs = idxs[::-1]
                    length = len(idxs)
                    ctx[max_len - length:] = track.feats[idxs]
                    mask[max_len - length:] = True
            if mask.any():
                ctxs.append(ctx)
                masks.append(mask)
                cands.append(feats[i])
                valid_idx.append(phrase_at)
            else:
                fallback += 1
            styles.append(kit["style"])
            phrase_at += 1

    if ctxs:
        h = encode_ranker_contexts(model, np.stack(ctxs), np.stack(masks),
                                   device, 1024)
        c = encode_ranker_candidates(model, np.stack(cands), device, 2048)
        cos = (h * c).sum(axis=1)
        raw[np.asarray(valid_idx, dtype=np.int64)] = cos

    calibrated = np.full(len(raw), 0.5, dtype=np.float64)
    raw_by_style: dict[str, dict] = {}
    for style in sorted(set(styles)):
        idxs = np.asarray([i for i, s in enumerate(styles) if s == style])
        vals = raw[idxs]
        finite = vals[np.isfinite(vals)]
        if len(finite) >= 2:
            mu = float(finite.mean())
            sigma = float(finite.std())
        elif len(finite) == 1:
            mu, sigma = float(finite[0]), 1.0
        else:
            mu, sigma = 0.0, 1.0
        if sigma < 1e-6:
            sigma = 1.0
        z = np.clip((vals - mu) / sigma, -30.0, 30.0)
        q = 1.0 / (1.0 + np.exp(-z))
        q[~np.isfinite(vals)] = 0.5
        calibrated[idxs] = q
        raw_by_style[style] = {
            "n_phrases": int(len(idxs)),
            "raw_mean": round(mu, 6),
            "raw_std": round(sigma, 6),
            "quality_mean": round(float(q.mean()), 6),
            "quality_std": round(float(q.std()), 6),
            "quality_min": round(float(q.min()), 6),
            "quality_max": round(float(q.max()), 6),
        }

    out: list[list[float]] = []
    at = 0
    for kit in kits:
        n = len(kit["feats"])
        out.append([round(float(x), 4) for x in calibrated[at:at + n]])
        at += n
    return out, raw, styles, raw_by_style, fallback


# --------------------------------------------------------------------------
# next kits
# --------------------------------------------------------------------------
def next_kit_scores_ranked(model: PairwiseRanker, kits: list[dict],
                           device: torch.device) -> tuple[list[list[dict]], dict]:
    """Top-10 same-style successors, BPM <=10 then <=20 if needed."""
    n = len(kits)
    max_len = int(model.cfg.get("max_len", CTX_LEN))
    ctx = np.zeros((n, max_len, kits[0]["feats"].shape[1]), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)
    first = np.stack([k["feats"][0] for k in kits])
    for i, kit in enumerate(kits):
        ctx[i], mask[i] = _context(kit["feats"], len(kit["feats"]) - 1, max_len)
    h = encode_ranker_contexts(model, ctx, mask, device, 1024)
    c = encode_ranker_candidates(model, first, device, 2048)
    scores = h @ c.T

    relaxed: list[str] = []
    below_after_relax: list[str] = []
    tolerance_counts = {10: 0, 20: 0}
    out: list[list[dict]] = []
    for ai, a in enumerate(kits):
        cand = [bi for bi in range(n)
                if bi != ai and kits[bi]["style"] == a["style"]
                and abs(kits[bi]["bpm"] - a["bpm"]) <= 10.0]
        tol = 10
        if len(cand) < 10:
            cand = [bi for bi in range(n)
                    if bi != ai and kits[bi]["style"] == a["style"]
                    and abs(kits[bi]["bpm"] - a["bpm"]) <= 20.0]
            tol = 20
            relaxed.append(a["track_id"])
        tolerance_counts[tol] += 1
        cand.sort(key=lambda bi: -float(scores[ai, bi]))
        if len(cand) < 10:
            below_after_relax.append(a["track_id"])
        row = []
        for bi in cand[:10]:
            b = kits[bi]
            row.append({
                "track_id": b["track_id"],
                "score": round(float(scores[ai, bi]), 4),
                "style": b["style"],
                "bpm": round(b["bpm"], 3),
                "key": b["key"],
            })
        out.append(row)
    return out, {
        "kits_with_10_at_bpm10": int(tolerance_counts[10]),
        "kits_relaxed_to_bpm20": int(tolerance_counts[20]),
        "kits_below_10_after_relax": int(len(below_after_relax)),
        "relaxed_track_ids": relaxed,
        "below_10_track_ids": below_after_relax,
    }


# --------------------------------------------------------------------------
# arrangement validation (old classifier)
# --------------------------------------------------------------------------
def _val_track_ids(dataset: dict, ckpt: dict, seed: int = 505,
                   tracks: list | None = None) -> set[str]:
    ids = ckpt.get("val_track_ids") or []
    if ids:
        return set(map(str, ids))
    if tracks is None:
        tracks, _ = build_track_data(dataset)
    unique = sorted({t.track_id for t in tracks if t.spans})
    rng = random.Random(seed + 1)
    rng.shuffle(unique)
    n_val = max(1, int(round(0.1 * len(unique)))) if len(unique) > 1 else 1
    return set(unique[:n_val])


def validate(model: TasteTransformer, dataset: dict, ckpt: dict,
             device: torch.device, batch_size: int, max_pairs: int = 200,
             seed: int = 505, tracks: list | None = None) -> dict:
    """Real vs shuffled arrangement AUC with the old classifier."""
    from sklearn.metrics import roc_auc_score

    if tracks is None:
        tracks, _ = build_track_data(dataset)
    val_ids = _val_track_ids(dataset, ckpt, seed, tracks)
    pool: list[tuple[Any, int, int]] = []
    for t in tracks:
        if t.track_id in val_ids:
            pool.extend((t, a, b) for a, b in t.spans)
    rng = random.Random(seed + 2)
    if len(pool) > max_pairs:
        pool = rng.sample(pool, max_pairs)
    if not pool:
        for t in tracks:
            pool.extend((t, a, b) for a, b in t.spans)
            if len(pool) >= max_pairs:
                break
        pool = pool[:max_pairs]
    if not pool:
        return {"real_vs_shuffle_auc": float("nan"), "n_pairs": 0,
                "mean_real_score": float("nan"),
                "mean_shuffled_score": float("nan"),
                "mean_cross_song_score": float("nan")}

    real = np.stack([t.feats[a:b] for t, a, b in pool]).astype(np.float32)
    shuffled = real.copy()
    for row in shuffled:
        order = rng.sample(range(SEQ_LEN), SEQ_LEN)
        row[:] = row[order]
    if not any(not np.array_equal(real[i], shuffled[i])
               for i in range(len(real))):
        shuffled = np.roll(real, 1, axis=1)

    donor_tracks = [t for t in tracks if len(t.spans) > 0]
    cross = []
    for _ in range(len(pool)):
        donor = donor_tracks[rng.randrange(len(donor_tracks))]
        a, b = donor.spans[rng.randrange(len(donor.spans))]
        cross.append(donor.feats[a:b])
    cross_arr = np.stack(cross).astype(np.float32)

    n = len(pool)
    scores = score_sequences(model, np.concatenate([real, shuffled, cross_arr]),
                             device, batch_size)
    s_real, s_shuf, s_cross = scores[:n], scores[n:2 * n], scores[2 * n:]
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(labels, np.concatenate([s_real, s_shuf])))
    return {
        "real_vs_shuffle_auc": round(auc, 6),
        "n_pairs": int(n),
        "mean_real_score": round(float(s_real.mean()), 6),
        "mean_shuffled_score": round(float(s_shuf.mean()), 6),
        "mean_cross_song_score": round(float(s_cross.mean()), 6),
        "n_cross_song": int(len(s_cross)),
        "_ok": bool(s_real.mean() > s_shuf.mean() and auc > 0.8),
    }


# --------------------------------------------------------------------------
# retrieval evaluation (before classifier, after ranker)
# --------------------------------------------------------------------------
def full_track_retrieval(ranker: PairwiseRanker,
                         classifier: TasteTransformer, data: RankerData,
                         tracks: list, device: torch.device,
                         batch_size: int) -> dict:
    """Rank all aligned windows of the held-out track for each context."""
    h = encode_ranker_contexts(ranker, data.ctx, data.mask, device, batch_size)
    r_top1 = r_top5 = c_top1 = c_top5 = total = 0
    for ti in np.unique(data.track_idx):
        rows = np.nonzero(data.track_idx == ti)[0]
        if len(rows) == 0:
            continue
        track = tracks[int(ti)]
        tgt = data.target_idx[rows]
        n = len(track.feats)
        # ranker
        c = encode_ranker_candidates(ranker, track.feats, device, batch_size)
        order = np.argsort(-(h[rows] @ c.T), axis=1)
        r_top1 += int((order[:, 0] == tgt).sum())
        r_top5 += int((order[:, :5] == tgt[:, None]).any(axis=1).sum())
        # classifier
        ctx6 = np.stack([_context_for_classifier(data, int(r)) for r in rows])
        seqs = np.empty((len(rows), n, SEQ_LEN, ctx6.shape[-1]),
                        dtype=np.float32)
        seqs[:, :, :CLASSIFIER_CTX] = ctx6[:, None, :, :]
        seqs[:, :, CLASSIFIER_CTX] = track.feats[None, :, :]
        seqs[:, :, CLASSIFIER_CTX + 1] = track.feats[None, :, :]
        cs = score_sequences(classifier, seqs.reshape(-1, SEQ_LEN,
                                                       ctx6.shape[-1]),
                             device, batch_size).reshape(len(rows), n)
        order = np.argsort(-cs, axis=1)
        c_top1 += int((order[:, 0] == tgt).sum())
        c_top5 += int((order[:, :5] == tgt[:, None]).any(axis=1).sum())
        total += len(rows)
    return {
        "n_contexts": int(total),
        "before": {
            "top1_count": int(c_top1), "top5_count": int(c_top5),
            "n": int(total),
            "top1": round(c_top1 / max(total, 1), 6),
            "top5": round(c_top5 / max(total, 1), 6),
        },
        "after": {
            "top1_count": int(r_top1), "top5_count": int(r_top5),
            "n": int(total),
            "top1": round(r_top1 / max(total, 1), 6),
            "top5": round(r_top5 / max(total, 1), 6),
        },
    }


def controlled_retrieval(ranker: PairwiseRanker,
                         classifier: TasteTransformer, data: RankerData,
                         tracks: list, device: torch.device,
                         batch_size: int, seed: int = 505) -> dict:
    """True +4-bar successor vs 7 random same-track non-successors."""
    rng = random.Random(seed + 3)
    n_ctx = len(data.ctx)
    cand_idx = np.empty((n_ctx, 8), dtype=np.int64)
    for r in range(n_ctx):
        ti = int(data.track_idx[r])
        t = int(data.target_idx[r])
        track = tracks[ti]
        others = [j for j in range(len(track.feats))
                  if j != t and int(track.bars[j]) != int(data.ctx_end_bar[r]) + 4]
        rng.shuffle(others)
        picks = others[:7]
        while len(picks) < 7:
            picks.append(others[rng.randrange(len(others))] if others else t)
        cand_idx[r, 0] = t
        cand_idx[r, 1:] = picks

    h = encode_ranker_contexts(ranker, data.ctx, data.mask, device, batch_size)
    cand_feats = np.stack([
        tracks[int(data.track_idx[r])].feats[cand_idx[r]]
        for r in range(n_ctx)
    ]).astype(np.float32)
    c = encode_ranker_candidates(ranker, cand_feats.reshape(-1, cand_feats.shape[-1]),
                                 device, batch_size * 4).reshape(n_ctx, 8, -1)
    rs = np.einsum("nd,nkd->nk", h, c)
    r_order = np.argsort(-rs, axis=1)
    r_top1 = int((r_order[:, 0] == 0).sum())
    r_top5 = int((r_order[:, :5] == 0).sum())

    ctx6 = np.stack([_context_for_classifier(data, r) for r in range(n_ctx)])
    seqs = np.empty((n_ctx, 8, SEQ_LEN, ctx6.shape[-1]), dtype=np.float32)
    seqs[:, :, :CLASSIFIER_CTX] = ctx6[:, None, :, :]
    seqs[:, :, CLASSIFIER_CTX] = cand_feats
    seqs[:, :, CLASSIFIER_CTX + 1] = cand_feats
    cs = score_sequences(classifier, seqs.reshape(-1, SEQ_LEN, ctx6.shape[-1]),
                         device, batch_size * 2).reshape(n_ctx, 8)
    c_order = np.argsort(-cs, axis=1)
    c_top1 = int((c_order[:, 0] == 0).sum())
    c_top5 = int((c_order[:, :5] == 0).sum())
    return {
        "n_contexts": int(n_ctx),
        "before": {
            "top1_count": int(c_top1), "top5_count": int(c_top5),
            "n": int(n_ctx),
            "top1": round(c_top1 / max(n_ctx, 1), 6),
            "top5": round(c_top5 / max(n_ctx, 1), 6),
        },
        "after": {
            "top1_count": int(r_top1), "top5_count": int(r_top5),
            "n": int(n_ctx),
            "top1": round(r_top1 / max(n_ctx, 1), 6),
            "top5": round(r_top5 / max(n_ctx, 1), 6),
        },
    }


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
def score_all(args: argparse.Namespace) -> dict:
    started = time.time()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device))
    set_seed(args.seed)

    dataset = load_dataset(args.dataset)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    classifier, ckpt = load_model(args.model, device)
    ranker, rckpt = load_ranker(args.ranker, device)
    feature_dim = int(rckpt["cfg"]["feature_dim"])
    print(f"[score] classifier={args.model} val_auc={ckpt.get('val_auc')} "
          f"ranker={args.ranker} ranks={rckpt.get('val_top1')}", flush=True)

    tracks, _ = build_track_data(dataset)
    tracks_by_id = {t.track_id: t for t in tracks}
    kits, missing = build_kits(manifest, dataset, feature_dim, tracks_by_id)
    print(f"[score] kits={len(kits)} phrases={sum(len(k['feats']) for k in kits)} "
          f"phrase_features_from_track_windows={missing}", flush=True)

    score_mats = ranker_transition_scores(ranker, kits, device)
    temperature, temp_info = tune_temperature(score_mats)
    transitions = apply_temperature(score_mats, temperature)
    print(f"[score] transition temperature={temperature:.5f} "
          f"median_row_max={temp_info['median_row_max']:.3f}", flush=True)
    old_transitions = classifier_transition_matrices(
        classifier, kits, device, args.batch_size)

    qualities, raw_quality, quality_styles, quality_by_style, q_fallback = (
        ranker_quality(ranker, kits, tracks_by_id, device))
    nexts, next_filter = next_kit_scores_ranked(ranker, kits, device)
    print(f"[score] transition/quality/next passes done "
          f"({time.time() - started:.1f}s)", flush=True)

    payload = {
        "version": 1,
        "model": "taste_v1",
        "val_auc": round(float(ckpt.get("val_auc", float("nan"))), 6),
        "kits": {},
    }
    for kit, mat, qual, nxt in zip(kits, transitions, qualities, nexts):
        payload["kits"][kit["track_id"]] = {
            "phrase_order": kit["phrase_order"],
            "transition": [[round(float(v), 4) for v in row] for row in mat],
            "quality": qual,
            "next": nxt,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")),
                        encoding="utf-8")
    size = out_path.stat().st_size

    # ---- validation -----------------------------------------------------
    val = validate(classifier, dataset, ckpt, device, args.batch_size,
                   max_pairs=args.max_validation_pairs, seed=args.seed,
                   tracks=tracks)
    val_ids = set(map(str, ckpt.get("val_track_ids") or []))
    if not val_ids:
        val_ids = _val_track_ids(dataset, ckpt, args.seed, tracks)
    val_data = build_ranker_data(tracks, val_ids, seed=args.seed,
                                 context_len=int(rckpt["cfg"].get("max_len",
                                                                 CTX_LEN)))
    retrieval_full = full_track_retrieval(
        ranker, classifier, val_data, tracks, device, args.batch_size)
    retrieval_controlled = controlled_retrieval(
        ranker, classifier, val_data, tracks, device, args.batch_size,
        seed=args.seed)
    within_before = evaluate_within_kit(old_transitions, kits)
    within_after = evaluate_within_kit(transitions, kits)
    within_before_nodiag = evaluate_within_kit(old_transitions, kits,
                                               exclude_self=True)
    within_after_nodiag = evaluate_within_kit(transitions, kits,
                                              exclude_self=True)
    sharpness = matrix_sharpness(transitions)
    print(f"[score] retrieval full before/after top1="
          f"{retrieval_full['before']['top1']:.3f}/"
          f"{retrieval_full['after']['top1']:.3f}", flush=True)
    print(f"[score] within-kit before/after top1="
          f"{within_before['top1_count']}/{within_before['n']} -> "
          f"{within_after['top1_count']}/{within_after['n']}", flush=True)

    checks: dict[str, Any] = {}
    checks["matrices_ok"] = all(
        len(payload["kits"][k["track_id"]]["transition"]) == len(k["feats"]) and
        all(len(row) == len(k["feats"])
            for row in payload["kits"][k["track_id"]]["transition"])
        for k in kits)
    checks["rows_stochastic"] = all(
        abs(sum(row) - 1.0) < 1e-3
        for kit in payload["kits"].values() for row in kit["transition"])
    checks["size_ok"] = size < MAX_JSON_BYTES
    checks["quality_ok"] = all(
        0.0 <= q <= 1.0 for kit in payload["kits"].values()
        for q in kit["quality"])
    checks["next_ok"] = all(
        len(row["next"]) <= 10 and
        all(x["style"] == row["style"] and x["track_id"] != row["track_id"]
            for x in row["next"])
        for row in [dict(track_id=k["track_id"], style=k["style"],
                         next=payload["kits"][k["track_id"]]["next"])
                    for k in kits])
    checks["temperature_ok"] = 0.45 <= sharpness["row_max_median"] <= 0.6
    checks["diagonal_zero"] = all(
        abs(float(mat[i, i])) < 1e-4
        for mat in transitions for i in range(len(mat)))
    checks["real_vs_shuffle_ok"] = bool(val.get("_ok", False))
    checks["passed"] = bool(
        checks["matrices_ok"] and checks["rows_stochastic"]
        and checks["size_ok"] and checks["quality_ok"] and checks["next_ok"]
        and checks["temperature_ok"] and checks["diagonal_zero"]
        and checks["real_vs_shuffle_ok"])

    means = []
    for kit in kits:
        q = payload["kits"][kit["track_id"]]["quality"]
        means.append((float(np.mean(q)) if q else 0.0, kit["track_id"],
                      kit["style"], len(kit["feats"])))
    means.sort(key=lambda x: -x[0])
    sample_kit = max(kits, key=lambda k: len(k["feats"]))
    sample_rows = payload["kits"][sample_kit["track_id"]]["transition"][:3]
    report = {
        "val_auc": round(float(ckpt.get("val_auc", float("nan"))), 6),
        "mean_real_score": val["mean_real_score"],
        "mean_shuffled_score": val["mean_shuffled_score"],
        "mean_cross_song_score": val["mean_cross_song_score"],
        "real_vs_shuffle_auc": val["real_vs_shuffle_auc"],
        "retrieval": {
            "full_track": retrieval_full,
            "valid_successor_vs_random": retrieval_controlled,
            "n_contexts": retrieval_full["n_contexts"],
            "method": {
                "before": ("old classifier scored [last up to 6 context "
                           "windows, candidate, candidate] as an 8-token "
                           "sequence"),
                "after": "ranker cosine(context, candidate)",
                "full_track": ("candidates = all aligned windows of the "
                               "held-out track"),
                "valid_successor_vs_random": ("1 true +4-bar successor + 7 "
                                              "random same-track "
                                              "non-successors"),
            },
        },
        "within_kit": {
            "before": within_before,
            "after": within_after,
            "before_excluding_self": within_before_nodiag,
            "after_excluding_self": within_after_nodiag,
            "note": ("the new matrix zeroes the diagonal, so 'after' is a "
                     "self-excluded next-phrase metric; 'before' keeps the "
                     "old classifier's diagonal for comparison"),
        },
        "matrix": {
            "temperature": round(temperature, 6),
            "target_median_row_max": TARGET_ROW_MAX,
            "diagonal": ("excluded (set to zero before softmax: a next-phrase "
                         "matrix has no self-loop)"),
            **sharpness,
        },
        "quality_by_style": quality_by_style,
        "next_filter": next_filter,
        "top_kits": [
            {"track_id": t, "style": s, "mean_quality": round(q, 4),
             "n_phrases": n} for q, t, s, n in means[:5]
        ],
        "bottom_kits": [
            {"track_id": t, "style": s, "mean_quality": round(q, 4),
             "n_phrases": n} for q, t, s, n in means[-5:][::-1]
        ],
        "sample_matrix": {
            "track_id": sample_kit["track_id"],
            "phrase_order": sample_kit["phrase_order"],
            "bars": sample_kit["bars"],
            "rows": [[round(float(v), 4) for v in row] for row in sample_rows],
        },
        "checks": checks,
        "counts": {
            "kits": len(kits),
            "phrases": int(sum(len(k["feats"]) for k in kits)),
            "phrase_features_from_track_windows": int(missing),
            "quality_context_fallbacks": int(q_fallback),
            "validation_pairs": int(val["n_pairs"]),
            "json_bytes": int(size),
        },
        "model": {
            "classifier": {"path": str(args.model),
                           "params": int(ckpt.get("params", 0)),
                           "val_auc": round(float(ckpt.get("val_auc", 0.0)), 6)},
            "ranker": {"path": str(args.ranker),
                       "params": int(rckpt.get("params", 0)),
                       "val_top1": rckpt.get("val_top1"),
                       "val_top5": rckpt.get("val_top5"),
                       "context_len": int(rckpt["cfg"].get("max_len", 0))},
            "feature_dim": feature_dim,
        },
        "temperature": round(temperature, 6),
        "duration_sec": round(time.time() - started, 2),
        "static_split_note": (
            "val_auc is from the classifier's training-time val tracks; the "
            "same tracks are used for the real-vs-shuffle sanity check, so it "
            "is optimistic."),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[score] wrote {out_path} ({size/1e6:.2f} MB) "
          f"report={report_path}", flush=True)
    print(f"[score] classifier real={val['mean_real_score']:.3f} "
          f"shuf={val['mean_shuffled_score']:.3f} AUC="
          f"{val['real_vs_shuffle_auc']:.3f}", flush=True)
    print(f"[score] row max median={sharpness['row_max_median']:.3f} "
          f"near_uniform={sharpness['near_uniform_rows']}/"
          f"{sharpness['rows']}", flush=True)
    print(f"[score] checks={checks}", flush=True)
    return {"payload": payload, "report": report, "checks": checks,
            "size": size, "val": val, "kits": kits,
            "temperature": temperature}


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------
def _synthetic_manifest(dataset: dict) -> dict:
    kits = []
    for ti, t in enumerate(dataset["tracks"]):
        n_phr = 3 if ti == 0 else 5  # exercise the <4 phrase path
        phrases = []
        for k in range(n_phr):
            w = t["windows"][2 + k]  # consecutive bars, so +4 successors exist
            phrases.append({
                "id": f"kit_{t['track_id']}_p{k + 1:02d}",
                "section": w["section"],
                "start_bar": w["start_bar"],
                "energy": w["energy"],
                "duration_sec": 7.5,
            })
        kits.append({
            "track_id": t["track_id"], "title": t["track_id"],
            "bpm": t["bpm"], "key": t["key"], "style": t["style"],
            "phrases": phrases,
        })
    assert kits
    return {"version": 1, "bars": 4, "sample_rate": 44100, "kits": kits}


def _self_test() -> int:
    set_seed(505)
    ds = synthetic_dataset(n_tracks=20, n_windows=24)
    manifest = _synthetic_manifest(ds)
    with tempfile.TemporaryDirectory(prefix="taste_score_self_test_") as td:
        tmp = Path(td)
        ds_path = tmp / "taste_dataset.json"
        ds_path.write_text(json.dumps(ds), encoding="utf-8")
        manifest_path = tmp / "kit_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        train_args = argparse.Namespace(
            seed=505, epochs=6, batch_size=64, lr=1e-3, weight_decay=1e-2,
            device="cpu", d_model=64, n_layers=2, n_heads=4,
            dim_feedforward=256, dropout=0.05, max_sequences=None,
            dataset_path=str(ds_path))
        model_path = tmp / "taste_model.pt"
        train_model(ds, train_args, model_path, tmp / "train_log.json")

        ranker_args = argparse.Namespace(
            seed=505, ranker_epochs=3, ranker_batch_size=32,
            ranker_lr=1e-3, weight_decay=1e-2, device="cpu", dropout=0.05,
            ranker_d_model=32, ranker_layers=1, ranker_heads=2,
            max_contexts=None, dataset_path=str(ds_path))
        ranker_path = tmp / "taste_ranker.pt"
        train_ranker_model(ds, ranker_args, ranker_path,
                           tmp / "ranker_log.json", model_path)

        args = argparse.Namespace(
            dataset=str(ds_path), manifest=str(manifest_path),
            model=str(model_path), ranker=str(ranker_path),
            out=str(tmp / "taste_scores.json"),
            report=str(tmp / "taste_report.json"), device="cpu",
            batch_size=256, max_validation_pairs=32, seed=505)
        res = score_all(args)
        payload = json.loads(Path(args.out).read_text())
        report = json.loads(Path(args.report).read_text())

        assert len(payload["kits"]) == len(manifest["kits"])
        for entry in manifest["kits"]:
            tid = entry["track_id"]
            n = len(entry["phrases"])
            row = payload["kits"][tid]
            assert len(row["phrase_order"]) == n
            assert len(row["transition"]) == n
            assert all(len(r) == n for r in row["transition"])
            assert all(abs(sum(r) - 1.0) < 1e-3 for r in row["transition"])
            assert len(row["quality"]) == n
            assert all(0.0 <= q <= 1.0 for q in row["quality"])
            assert 1 <= len(row["next"]) <= 10
            assert all(set(("track_id", "score", "style", "bpm", "key"))
                       <= set(x) for x in row["next"])
            assert tid not in [x["track_id"] for x in row["next"]]
            assert all(x["style"] == entry["style"] for x in row["next"])
        assert len(payload["kits"]["syn0000"]["transition"]) == 3
        assert res["checks"]["matrices_ok"]
        assert res["checks"]["size_ok"]
        assert res["checks"]["quality_ok"]
        assert res["checks"]["next_ok"]
        assert 0.45 <= report["matrix"]["row_max_median"] <= 0.6
        assert report["within_kit"]["after"]["n"] > 0
        assert report["retrieval"]["full_track"]["after"]["n"] > 0
        assert len(report["sample_matrix"]["rows"]) == 3
        assert len(report["top_kits"]) == 5
        assert len(report["bottom_kits"]) == 5
        assert "quality_by_style" in report
        assert report["counts"]["json_bytes"] == Path(args.out).stat().st_size
        print(f"[self-test] OK kits={len(payload['kits'])} "
              f"bytes={report['counts']['json_bytes']} "
              f"within {report['within_kit']['after']['top1_count']}/"
              f"{report['within_kit']['after']['n']} "
              f"rowmax={report['matrix']['row_max_median']:.3f}", flush=True)
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/tmp/taste_dataset.json")
    ap.add_argument(
        "--manifest",
        default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--model", default="data/taste_model.pt")
    ap.add_argument("--ranker", default="data/taste_ranker.pt")
    ap.add_argument(
        "--out", default="frontend/public/data/kits/taste_scores.json")
    ap.add_argument("--report", default="/tmp/taste_report.json")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-validation-pairs", type=int, default=200)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    for attr in ("dataset", "manifest", "model", "ranker", "out", "report"):
        p = Path(getattr(args, attr))
        if not p.is_absolute():
            setattr(args, attr, str(REPO_ROOT / p))
    result = score_all(args)
    return 0 if result["checks"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
