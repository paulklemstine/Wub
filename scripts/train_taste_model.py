#!/usr/bin/env python3
"""Train Wub's "taste" model: a sequence-level realness ranker.

A *sequence* is 8 consecutive analysis windows from one track whose
``start_bar`` increases by exactly 4 bars.  The model learns to separate real
sequences (label 1) from four corruptions (label 0):

1. ``shuffle``          - the same 8 windows in a random order,
2. ``same_style``       - one window replaced by a window of another track of
                          the same style,
3. ``cross_track``      - the whole sequence replaced by a sequence from a
                          different track,
4. ``energy_jump``      - one window replaced by a same-track window from a
                          different section with |delta energy| > 0.3.

Architecture: 48 -> 128 projection, learned CLS token + learned positional
embedding (9 positions), 4-layer / 4-head Transformer encoder (GELU, pre-LN,
dropout 0.1, FFN 896), CLS -> 2-class linear head (~1.2M parameters).

Training: BCE-with-logits, AdamW (lr 3e-4, cosine), batch 128, 15 epochs,
class balanced with ``pos_weight = n_neg / n_pos``.  CUDA is used when
available.  Seed 505 everywhere.

The same file also trains a pairwise transition ranker (``--train-ranker``):
a 3-layer/4-head context Transformer over the last 1..6 windows, a 48->128->
128 candidate MLP, cosine score with a learnable logit scale, and InfoNCE
against 1 positive + 7 same-track / same-style negatives.  It reuses the
classifier checkpoint's train-only ``cfg.mean`` / ``cfg.std`` and track split.

Artifacts:
    data/taste_model.pt             {model, cfg, val_auc, val_track_ids, ...}
    data/taste_train_log.json       per-epoch losses/metrics + final report
    data/taste_ranker.pt            ranker {model, cfg, val_top1, val_top5}
    data/taste_ranker_train_log.json ranker per-epoch log

Usage:
    python3 scripts/train_taste_model.py \\
        --dataset /tmp/taste_dataset.json \\
        --out data/taste_model.pt --log data/taste_train_log.json
    python3 scripts/train_taste_model.py \\
        --dataset /tmp/taste_dataset.json --train-ranker \\
        --classifier-ckpt data/taste_model.pt \\
        --ranker-out data/taste_ranker.pt \\
        --ranker-log data/taste_ranker_train_log.json
    python3 scripts/train_taste_model.py --self-test
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 505
FEATURE_DIM = 48
SEQ_LEN = 8
BAR_STRIDE = 4
CTX_LEN = 6
RANKER_NEGATIVES = 7
KINDS = ("shuffle", "same_style", "cross_track", "energy_jump")
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# determinism / io
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(path: str | Path) -> dict:
    """Load ``taste_dataset.json`` or its ``.gz`` sibling."""
    p = Path(path)
    if not p.exists():
        gz = Path(str(p) + ".gz")
        if gz.exists():
            p = gz
        else:
            raise FileNotFoundError(f"taste dataset not found: {path} (or .gz)")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    if "tracks" not in data:
        raise ValueError(f"{p}: missing 'tracks'")
    return data


# --------------------------------------------------------------------------
# data structures
# --------------------------------------------------------------------------
@dataclasses.dataclass
class TrackData:
    idx: int
    track_id: str
    style: str
    bpm: float
    key: str
    bars: np.ndarray          # [n] int
    feats: np.ndarray         # [n, feature_dim] float32
    sections: list[str]
    energies: np.ndarray      # [n] float32
    spans: list[tuple[int, int]]  # valid 8-window spans (start, end)


def valid_spans(bars: Iterable[float], seq_len: int = SEQ_LEN,
                stride: int = BAR_STRIDE) -> list[tuple[int, int]]:
    """All (start, end) index spans whose bars increase by exactly ``stride``."""
    b = np.asarray(list(bars), dtype=np.int64)
    if len(b) < seq_len:
        return []
    d = np.diff(b)
    windows = np.lib.stride_tricks.sliding_window_view(d, seq_len - 1)
    mask = np.all(windows == stride, axis=1)
    return [(int(i), int(i + seq_len)) for i in np.nonzero(mask)[0]]


def build_track_data(dataset: dict) -> tuple[list[TrackData], int]:
    """Sort/dedupe windows per track and cache feature matrices."""
    tracks: list[TrackData] = []
    feature_dim = int(dataset.get("feature_dim", FEATURE_DIM))
    for t in dataset.get("tracks", []):
        tid = str(t.get("track_id", ""))
        ws = sorted(t.get("windows") or [],
                    key=lambda w: float(w.get("start_bar", 0.0)))
        if not ws:
            continue
        feats: list[np.ndarray] = []
        bars: list[int] = []
        sections: list[str] = []
        energies: list[float] = []
        seen: set[int] = set()
        # The taste dataset interleaves off-grid kit-phrase windows
        # (``aligned: false``) with the regular 4-bar grid.  Sequences must be
        # formed on the strictly aligned grid; missing flags count as aligned.
        for w in ws:
            if not bool(w.get("aligned", True)):
                continue
            bar = int(round(float(w.get("start_bar", 0))))
            if bar in seen:
                continue
            seen.add(bar)
            f = np.asarray(w.get("features") or [], dtype=np.float32).reshape(-1)
            if f.size != feature_dim:
                raise ValueError(
                    f"track {tid} bar {bar}: expected {feature_dim} features, "
                    f"got {f.size}")
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(f)
            bars.append(bar)
            sections.append(str(w.get("section", "")))
            energies.append(float(w.get("energy", 0.0)))
        arr = np.stack(feats).astype(np.float32)
        bars_arr = np.asarray(bars, dtype=np.int64)
        tracks.append(TrackData(
            idx=len(tracks),
            track_id=tid,
            style=str(t.get("style", "unknown")),
            bpm=float(t.get("bpm", 0.0)),
            key=str(t.get("key", "")),
            bars=bars_arr,
            feats=arr,
            sections=sections,
            energies=np.nan_to_num(np.asarray(energies, dtype=np.float32),
                                   nan=0.0),
            spans=valid_spans(bars_arr),
        ))
    return tracks, feature_dim


@dataclasses.dataclass
class Prepared:
    X: np.ndarray             # [N, 8, F] float32
    y: np.ndarray             # [N] float32
    kinds: np.ndarray         # [N] <U16
    styles: np.ndarray        # [N] <U32
    track_ids: np.ndarray     # [N] <U64  anchor track (split unit)
    source_track_ids: np.ndarray  # [N] track the windows actually came from
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_track_ids: list[str]
    counts: dict
    feature_dim: int


def _shuffled(seq: np.ndarray, rng: random.Random) -> np.ndarray:
    """Permutation of ``seq`` that is guaranteed not to be identical."""
    out = seq.copy()
    for _ in range(8):
        rng.shuffle(out)
        if not np.array_equal(out, seq):
            return out
    return np.roll(seq, 1)


def _donor_sequence(track: TrackData, rng: random.Random) -> np.ndarray:
    if track.spans:
        a, b = track.spans[rng.randrange(len(track.spans))]
        return np.arange(a, b)
    n = len(track.feats)
    return np.asarray([rng.randrange(n) for _ in range(SEQ_LEN)], dtype=np.int64)


def _pick_other(pool: list[int], exclude: int,
                rng: random.Random) -> int | None:
    """A random pool member different from ``exclude`` (None if impossible)."""
    if not pool:
        return None
    if len(pool) == 1 and pool[0] == exclude:
        return None
    for _ in range(8):
        idx = pool[rng.randrange(len(pool))]
        if idx != exclude:
            return idx
    return None


def prepare(dataset: dict, seed: int = SEED,
            max_sequences: int | None = None) -> Prepared:
    """Build positives + four corruptions and a track-disjoint 90/10 split."""
    tracks, feature_dim = build_track_data(dataset)
    if not tracks:
        raise ValueError("dataset has no usable tracks/windows")

    # anchors: one per real sequence
    anchors: list[tuple[int, np.ndarray]] = []
    for t in tracks:
        for a, b in t.spans:
            anchors.append((t.idx, np.arange(a, b, dtype=np.int64)))
    if not anchors:
        raise ValueError("no valid 8-window sequences (bars +4) in dataset")

    # Track-disjoint 90/10 split, decided before negatives are sampled so a
    # validation track never contributes a donor window to training.
    unique_ids = sorted({t.track_id for t in tracks if t.spans})
    split_rng = random.Random(seed + 1)
    split_rng.shuffle(unique_ids)
    n_val = max(1, int(round(0.1 * len(unique_ids)))) if len(unique_ids) > 1 else 1
    val_ids = set(unique_ids[:n_val])

    # Pre-index donors per split (and per style) so negative sampling stays
    # O(1) per anchor.
    donor_ids: dict[bool, list[int]] = {False: [], True: []}
    style_ids: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for t in tracks:
        in_val = t.track_id in val_ids
        donor_ids[in_val].append(t.idx)
        style_ids[(t.style, in_val)].append(t.idx)

    rng = random.Random(seed)
    if max_sequences is not None and len(anchors) > max_sequences:
        idx = sorted(rng.sample(range(len(anchors)), max_sequences))
        anchors = [anchors[i] for i in idx]

    records: list[tuple[np.ndarray, float, str, str, str, str]] = []
    skipped = Counter()
    for ti, seq in anchors:
        track = tracks[ti]
        in_val = track.track_id in val_ids
        seq_feats = track.feats[seq]
        records.append((seq_feats, 1.0, "real", track.style,
                        track.track_id, track.track_id))

        # 1. order corruption
        records.append((track.feats[_shuffled(seq, rng)], 0.0, "shuffle",
                        track.style, track.track_id, track.track_id))

        # 2. same-style window swap (same split only)
        donor_idx = _pick_other(style_ids[(track.style, in_val)], ti, rng)
        if donor_idx is not None:
            donor = tracks[donor_idx]
            pos = rng.randrange(SEQ_LEN)
            out = seq_feats.copy()
            out[pos] = donor.feats[rng.randrange(len(donor.feats))]
            records.append((out, 0.0, "same_style", track.style,
                            track.track_id, donor.track_id))
        else:
            skipped["same_style"] += 1

        # 3. whole sequence from another track (same split only)
        donor_idx = _pick_other(donor_ids[in_val], ti, rng)
        if donor_idx is not None:
            donor = tracks[donor_idx]
            records.append((donor.feats[_donor_sequence(donor, rng)], 0.0,
                            "cross_track", donor.style, track.track_id,
                            donor.track_id))
        else:
            skipped["cross_track"] += 1

        # 4. same-track energy-jump / section-change swap (optional)
        pos = rng.randrange(SEQ_LEN)
        base_sec = track.sections[seq[pos]]
        base_energy = float(track.energies[seq[pos]])
        cand = [k for k in range(len(track.feats))
                if track.sections[k] != base_sec
                and abs(float(track.energies[k]) - base_energy) > 0.3]
        if cand:
            out = seq_feats.copy()
            out[pos] = track.feats[cand[rng.randrange(len(cand))]]
            records.append((out, 0.0, "energy_jump", track.style,
                            track.track_id, track.track_id))
        else:
            skipped["energy_jump"] += 1

    n = len(records)
    X = np.empty((n, SEQ_LEN, feature_dim), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    kinds = np.empty(n, dtype="<U16")
    styles = np.empty(n, dtype="<U32")
    track_ids = np.empty(n, dtype="<U64")
    source_track_ids = np.empty(n, dtype="<U64")
    for k, (feat8, label, kind, style, anchor_tid, src_tid) in enumerate(records):
        X[k] = feat8
        y[k] = label
        kinds[k] = kind
        styles[k] = style
        track_ids[k] = anchor_tid
        source_track_ids[k] = src_tid

    is_val = np.asarray([tid in val_ids for tid in track_ids])
    train_idx = np.nonzero(~is_val)[0]
    val_idx = np.nonzero(is_val)[0]
    if len(train_idx) == 0:  # degenerate tiny fixture
        train_idx = val_idx
    if len(val_idx) == 0:
        val_idx = train_idx

    counts = {
        "tracks": len(tracks),
        "tracks_with_sequences": len(unique_ids),
        "windows_total": int(sum(len(t.get("windows") or [])
                                 for t in dataset.get("tracks", []))),
        "windows_aligned_used": int(sum(len(t.bars) for t in tracks)),
        "sequences": len(anchors),
        "samples": int(n),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "by_kind": {k: int((kinds == k).sum()) for k in KINDS},
        "skipped": dict(skipped),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_tracks": len(unique_ids) - len(val_ids),
        "val_tracks": len(val_ids),
        "feature_dim": feature_dim,
    }
    return Prepared(X, y, kinds, styles, track_ids, source_track_ids,
                    train_idx, val_idx, sorted(val_ids), counts, feature_dim)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class TasteTransformer(nn.Module):
    """8-token Transformer encoder with a CLS real/fake head."""

    def __init__(self, feature_dim: int = FEATURE_DIM, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4,
                 dim_feedforward: int = 896, seq_len: int = SEQ_LEN,
                 dropout: float = 0.1, n_classes: int = 2,
                 mean: list[float] | None = None,
                 std: list[float] | None = None):
        super().__init__()
        self.cfg = {
            "feature_dim": int(feature_dim),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "dim_feedforward": int(dim_feedforward),
            "seq_len": int(seq_len),
            "dropout": float(dropout),
            "n_classes": int(n_classes),
            "mean": None if mean is None else [float(v) for v in mean],
            "std": None if std is None else [float(v) for v in std],
        }
        self.register_buffer(
            "feat_mean",
            None if mean is None else
            torch.as_tensor(mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer(
            "feat_std",
            None if std is None else
            torch.as_tensor(std, dtype=torch.float32).view(1, 1, -1))
        self.proj = nn.Linear(feature_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Embedding(seq_len + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> logits [B, 2]
        b, t, _ = x.shape
        if self.feat_mean is not None:
            x = (x - self.feat_mean) / self.feat_std
        h = self.proj(x)
        cls = self.cls.expand(b, 1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos(torch.arange(t + 1, device=x.device)).unsqueeze(0)
        h = self.encoder(h)
        return self.head(self.ln(h[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ContextEncoder(nn.Module):
    """Variable-length (1..6) window sequence -> mean-pooled vector."""

    def __init__(self, feature_dim: int = FEATURE_DIM, d_model: int = 128,
                 n_layers: int = 3, n_heads: int = 4,
                 max_len: int = CTX_LEN, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,L,F], mask [B,L] True for real context windows
        h = self.proj(x) + self.pos(torch.arange(x.size(1),
                                                device=x.device)).unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return F.normalize(self.ln(pooled), dim=-1)


class CandidateEncoder(nn.Module):
    """One 48-d window -> 128-d candidate vector."""

    def __init__(self, feature_dim: int = FEATURE_DIM, d_model: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PairwiseRanker(nn.Module):
    """Cosine next-window ranker: context encoder vs candidate encoder."""

    def __init__(self, feature_dim: int = FEATURE_DIM, d_model: int = 128,
                 n_layers: int = 3, n_heads: int = 4, max_len: int = CTX_LEN,
                 dropout: float = 0.1, init_scale: float = 10.0,
                 mean: list[float] | None = None,
                 std: list[float] | None = None):
        super().__init__()
        self.cfg = {
            "feature_dim": int(feature_dim),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "max_len": int(max_len),
            "dropout": float(dropout),
            "init_scale": float(init_scale),
            "mean": None if mean is None else [float(v) for v in mean],
            "std": None if std is None else [float(v) for v in std],
        }
        self.register_buffer(
            "feat_mean",
            None if mean is None else
            torch.as_tensor(mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer(
            "feat_std",
            None if std is None else
            torch.as_tensor(std, dtype=torch.float32).view(1, 1, -1))
        self.ctx = ContextEncoder(max_len=max_len, feature_dim=feature_dim,
                                  d_model=d_model, n_layers=n_layers,
                                  n_heads=n_heads, dropout=dropout)
        self.cand = CandidateEncoder(feature_dim=feature_dim, d_model=d_model,
                                     dropout=dropout)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat_mean is not None:
            mean = self.feat_mean.squeeze(0)  # [1, F], broadcasts to 2-D or 3-D
            std = self.feat_std.squeeze(0)
            x = (x - mean) / std
        return x

    def encode_context(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.ctx(self._norm(x), mask)

    def encode_candidate(self, x: torch.Tensor) -> torch.Tensor:
        return self.cand(self._norm(x))

    def forward(self, ctx: torch.Tensor, mask: torch.Tensor,
                cand: torch.Tensor) -> torch.Tensor:
        h = self.encode_context(ctx, mask)
        c = self.encode_candidate(cand)
        return (h * c).sum(dim=-1) * self.logit_scale.clamp(1.0, 100.0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def compute_metrics(y: np.ndarray, scores: np.ndarray, kinds: np.ndarray,
                    styles: np.ndarray) -> dict:
    """Overall + per-corruption AUC, accuracy, and within-style AUC."""
    y = np.asarray(y)
    pred = (scores >= 0.5).astype(np.float32)
    out: dict[str, Any] = {
        "auc": _auc(y, scores),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "auc_by_corruption": {},
        "n": int(len(y)),
    }
    for kind in KINDS:
        mask = (kinds == "real") | (kinds == kind)
        if mask.sum() == 0:
            continue
        out["auc_by_corruption"][kind] = {
            "auc": _auc((kinds[mask] == "real").astype(np.float32),
                        scores[mask]),
            "n": int(mask.sum()),
        }
    # style sanity check: restrict to the style with most val positives
    pos = kinds == "real"
    chosen_style = None
    if pos.any():
        chosen_style = Counter(styles[pos]).most_common(1)[0][0]
        mask = (pos & (styles == chosen_style)) | (
            (kinds != "real") & (styles == chosen_style))
        out["within_style"] = {
            "style": str(chosen_style),
            "auc": _auc((kinds[mask] == "real").astype(np.float32),
                        scores[mask]),
            "n": int(mask.sum()),
        }
    else:
        out["within_style"] = {"style": None, "auc": float("nan"), "n": 0}
    return out


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, y: np.ndarray | None = None,
            device: torch.device | str = "cpu", batch_size: int = 512
            ) -> tuple[np.ndarray, float]:
    """Return sigmoid P(real) and BCE loss (0 when labels are absent)."""
    model.eval()
    probs: list[np.ndarray] = []
    losses: list[float] = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits[:, 1]).float().cpu().numpy())
        if y is not None:
            yb = torch.from_numpy(y[start:start + batch_size]).to(device)
            loss = F.binary_cross_entropy_with_logits(logits[:, 1], yb)
            losses.append(float(loss.item()) * len(xb))
    p = np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
    loss = float(sum(losses) / max(len(X), 1)) if y is not None else 0.0
    return p, loss


def _split_kind_counts(y: np.ndarray, kinds: np.ndarray) -> dict:
    y = np.asarray(y)
    return {
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "by_kind": {k: int((kinds == k).sum()) for k in KINDS},
    }


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_model(dataset: dict, args: argparse.Namespace,
                out_path: Path, log_path: Path) -> dict:
    started = time.time()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device))
    set_seed(args.seed)

    prepared = prepare(dataset, seed=args.seed, max_sequences=args.max_sequences)
    counts = prepared.counts
    print(f"[taste] tracks={counts['tracks']} sequences={counts['sequences']} "
          f"samples={counts['samples']} (pos={counts['positives']} "
          f"neg={counts['negatives']})", flush=True)
    print(f"[taste] split train={counts['train_samples']} "
          f"val={counts['val_samples']} "
          f"({counts['train_tracks']}/{counts['val_tracks']} tracks)",
          flush=True)

    Xtr = prepared.X[prepared.train_idx]
    ytr = prepared.y[prepared.train_idx]
    Xva = prepared.X[prepared.val_idx]
    yva = prepared.y[prepared.val_idx]
    kinds_va = prepared.kinds[prepared.val_idx]
    styles_va = prepared.styles[prepared.val_idx]

    # Standardise inside the model.  Stats come from train windows only and
    # are stored in cfg.mean / cfg.std so score_kits.py applies the exact same
    # transform.  A 1e-3 floor keeps constant columns (section one-hot can be
    # constant on a small split, reserved zeros are always constant) at 0.
    flat = Xtr.reshape(-1, prepared.feature_dim)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    constant_dims = np.nonzero(feat_std < 1e-6)[0].tolist()
    feat_std = np.maximum(feat_std, 1e-3)

    model = TasteTransformer(
        feature_dim=prepared.feature_dim, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward, seq_len=SEQ_LEN,
        dropout=args.dropout, n_classes=2,
        mean=feat_mean, std=feat_std).to(device)
    n_params = model.n_params()
    print(f"[taste] params={n_params:,} device={device} "
          f"feature_dim={prepared.feature_dim}", flush=True)

    n_pos = float(ytr.sum())
    n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([max(n_neg, 1.0) / max(n_pos, 1.0)],
                              dtype=torch.float32, device=device)
    print(f"[taste] class balance pos_weight={float(pos_weight):.3f}", flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)

    history: list[dict] = []
    best = {"epoch": 0, "auc": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size].numpy()
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            logits = model(xb)
            loss = criterion(logits[:, 1], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)
        scheduler.step()

        train_loss = total_loss / max(seen, 1)
        probs, val_loss = predict(model, Xva, yva, device, args.batch_size)
        metrics = compute_metrics(yva, probs, kinds_va, styles_va)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc": round(metrics["auc"], 6),
            "val_accuracy": round(metrics["accuracy"], 6),
            "lr": round(float(scheduler.get_last_lr()[0]), 8),
        }
        history.append(row)
        if metrics["auc"] > best["auc"]:
            best = {"epoch": epoch, "auc": float(metrics["auc"])}
        print(f"[taste] epoch {epoch:02d}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"auc={metrics['auc']:.4f} acc={metrics['accuracy']:.4f}",
              flush=True)

    # final metrics (last epoch model, honest val_auc)
    probs, val_loss = predict(model, Xva, yva, device, args.batch_size)
    metrics = compute_metrics(yva, probs, kinds_va, styles_va)
    val_auc = float(metrics["auc"])

    cfg = dict(model.cfg)
    ckpt = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": cfg,
        "val_auc": val_auc,
        "val_track_ids": prepared.val_track_ids,
        "seed": int(args.seed),
        "feature_dim": int(prepared.feature_dim),
        "params": int(n_params),
        "trained_on": str(getattr(args, "dataset_path", "")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)

    log = {
        "version": 1,
        "model": "taste_v1",
        "seed": int(args.seed),
        "dataset": str(getattr(args, "dataset_path", "")),
        "counts": counts,
        "split_counts": {
            "train": _split_kind_counts(ytr, prepared.kinds[prepared.train_idx]),
            "val": _split_kind_counts(yva, kinds_va),
        },
        "classes": {
            "positives": counts["positives"],
            "negatives": counts["negatives"],
            "pos_weight": round(float(pos_weight), 6),
            "positive_fraction": round(
                counts["positives"] / max(counts["samples"], 1), 6),
        },
        "feature_norm": {
            "source": "train_split_only",
            "std_floor": 1e-3,
            "scope": "all 48 columns (section one-hot included; reserved "
                     "constant zeros stay zero after the 1e-3 floor)",
            "constant_dims": constant_dims,
            "mean": [round(float(v), 6) for v in feat_mean],
            "std": [round(float(v), 6) for v in feat_std],
        },
        "model_cfg": cfg,
        "params": int(n_params),
        "device": str(device),
        "epochs": history,
        "best_epoch": best,
        "metrics": metrics,
        "val_auc": val_auc,
        "duration_sec": round(time.time() - started, 2),
        "dataset_stats": dataset.get("stats", {}),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[taste] saved {out_path} ({out_path.stat().st_size/1e6:.2f} MB) "
          f"log={log_path}", flush=True)
    print(f"[taste] val AUC={val_auc:.4f} "
          f"within-style={metrics['within_style']['style']} "
          f"AUC={metrics['within_style']['auc']:.4f}", flush=True)
    return {"metrics": metrics, "counts": counts, "params": n_params,
            "val_auc": val_auc, "checkpoint": str(out_path),
            "log": str(log_path)}


# --------------------------------------------------------------------------
# pairwise transition ranker
# --------------------------------------------------------------------------
@dataclasses.dataclass
class RankerData:
    ctx: np.ndarray            # [N, 6, F] right-aligned context
    mask: np.ndarray           # [N, 6] bool
    target: np.ndarray         # [N, F]
    target_idx: np.ndarray     # [N] index into the track's aligned windows
    target_bar: np.ndarray     # [N]
    ctx_end_bar: np.ndarray    # [N] bar of the last context window
    track_idx: np.ndarray      # [N] index into the tracks list
    track_ids: np.ndarray      # [N] <U64
    styles: np.ndarray         # [N] <U32
    counts: dict


def build_ranker_data(tracks: list[TrackData], allowed_ids: set[str] | None = None,
                      max_contexts: int | None = None,
                      seed: int = SEED, context_len: int = CTX_LEN) -> RankerData:
    """Every real next-window pair with 1..``context_len`` preceding windows."""
    ctxs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_idx: list[int] = []
    target_bar: list[int] = []
    ctx_end_bar: list[int] = []
    track_idx: list[int] = []
    track_ids: list[str] = []
    styles: list[str] = []
    lengths: Counter = Counter()
    n_tracks = 0
    for t in tracks:
        if allowed_ids is not None and t.track_id not in allowed_ids:
            continue
        n = len(t.feats)
        if n < 2:
            continue
        n_tracks += 1
        runs: list[tuple[int, int]] = []
        start = 0
        for k in range(1, n):
            if int(t.bars[k] - t.bars[k - 1]) != BAR_STRIDE:
                runs.append((start, k))
                start = k
        runs.append((start, n))
        for a, b in runs:
            for ti in range(a + 1, b):
                length = min(context_len, ti - a)
                ctx = np.zeros((context_len, t.feats.shape[1]), dtype=np.float32)
                ctx[context_len - length:] = t.feats[ti - length:ti]
                mask = np.zeros(context_len, dtype=bool)
                mask[context_len - length:] = True
                ctxs.append(ctx)
                masks.append(mask)
                targets.append(t.feats[ti])
                target_idx.append(ti)
                target_bar.append(int(t.bars[ti]))
                ctx_end_bar.append(int(t.bars[ti - 1]))
                track_idx.append(t.idx)
                track_ids.append(t.track_id)
                styles.append(t.style)
                lengths[length] += 1
    if not ctxs:
        raise ValueError("no ranker contexts could be built")

    data = RankerData(
        ctx=np.stack(ctxs).astype(np.float32),
        mask=np.stack(masks),
        target=np.stack(targets).astype(np.float32),
        target_idx=np.asarray(target_idx, dtype=np.int64),
        target_bar=np.asarray(target_bar, dtype=np.int64),
        ctx_end_bar=np.asarray(ctx_end_bar, dtype=np.int64),
        track_idx=np.asarray(track_idx, dtype=np.int64),
        track_ids=np.asarray(track_ids, dtype="<U64"),
        styles=np.asarray(styles, dtype="<U32"),
        counts={},
    )
    if max_contexts is not None and len(data.ctx) > max_contexts:
        rng = random.Random(seed)
        keep = np.asarray(sorted(rng.sample(range(len(data.ctx)), max_contexts)),
                          dtype=np.int64)
        data = RankerData(
            ctx=data.ctx[keep], mask=data.mask[keep], target=data.target[keep],
            target_idx=data.target_idx[keep], target_bar=data.target_bar[keep],
            ctx_end_bar=data.ctx_end_bar[keep], track_idx=data.track_idx[keep],
            track_ids=data.track_ids[keep], styles=data.styles[keep], counts={})

    data.counts = {
        "tracks": n_tracks,
        "contexts": int(len(data.ctx)),
        "context_len": int(context_len),
        "context_lengths": {str(k): int(v) for k, v in sorted(lengths.items())},
        "context_tracks": int(len(set(data.track_ids.tolist()))),
    }
    return data


def _sample_negatives(data: RankerData, idx: np.ndarray, tracks: list[TrackData],
                      style_pool: dict[str, list[int]],
                      rng: random.Random, margin: int = 1) -> np.ndarray:
    """7 negatives/context: 50/50 same-track (not within +/-margin) and same-style."""
    b = len(idx)
    out = np.empty((b, RANKER_NEGATIVES, data.target.shape[1]), dtype=np.float32)
    for row, k in enumerate(idx):
        ti = int(data.track_idx[k])
        t = int(data.target_idx[k])
        feats = tracks[ti].feats
        n = len(feats)
        style = str(data.styles[k])
        for j in range(RANKER_NEGATIVES):
            use_same = rng.random() < 0.5 or not style_pool.get(style)
            if use_same:
                c = t
                for _ in range(12):
                    c = rng.randrange(n)
                    if abs(c - t) > margin:
                        break
                if abs(c - t) <= margin:  # degenerate tiny track
                    c = (t + margin + 1) % n
                out[row, j] = feats[c]
            else:
                pool = style_pool[style]
                di = ti
                for _ in range(12):
                    di = pool[rng.randrange(len(pool))]
                    if di != ti:
                        break
                if di == ti:
                    di = pool[(pool.index(ti) + 1) % len(pool)]
                donor = tracks[di].feats
                out[row, j] = donor[rng.randrange(len(donor))]
    return out


@torch.no_grad()
def encode_ranker_contexts(model: PairwiseRanker, ctx: np.ndarray,
                           mask: np.ndarray, device: torch.device,
                           batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(ctx), batch_size):
        xb = torch.from_numpy(ctx[start:start + batch_size]).to(device)
        mb = torch.from_numpy(mask[start:start + batch_size]).to(device)
        out.append(model.encode_context(xb, mb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.cfg["d_model"]),
                                                    dtype=np.float32)


@torch.no_grad()
def encode_ranker_candidates(model: PairwiseRanker, feats: np.ndarray,
                             device: torch.device,
                             batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(feats), batch_size):
        xb = torch.from_numpy(feats[start:start + batch_size]).to(device)
        out.append(model.encode_candidate(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.cfg["d_model"]),
                                                    dtype=np.float32)


def ranker_retrieval_metrics(model: PairwiseRanker, data: RankerData,
                             tracks: list[TrackData], device: torch.device,
                             batch_size: int = 1024) -> dict:
    """Rank all windows of each held-out track; count top-1/top-5 hits."""
    h = encode_ranker_contexts(model, data.ctx, data.mask, device, batch_size)
    top1 = top5 = total = 0
    for ti in np.unique(data.track_idx):
        rows = np.nonzero(data.track_idx == ti)[0]
        if len(rows) == 0:
            continue
        c = encode_ranker_candidates(model, tracks[int(ti)].feats, device,
                                     batch_size)
        scores = h[rows] @ c.T
        tgt = data.target_idx[rows]
        order = np.argsort(-scores, axis=1)
        top1 += int((order[:, 0] == tgt).sum())
        top5 += int((order[:, :5] == tgt[:, None]).any(axis=1).sum())
        total += len(rows)
    return {
        "top1": round(top1 / max(total, 1), 6),
        "top5": round(top5 / max(total, 1), 6),
        "top1_count": int(top1),
        "top5_count": int(top5),
        "n_contexts": int(total),
    }


def _ranker_val_loss(model: PairwiseRanker, data: RankerData,
                     negatives: np.ndarray, device: torch.device,
                     batch_size: int = 512) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, len(data.ctx), batch_size):
            sl = slice(start, start + batch_size)
            ctx = torch.from_numpy(data.ctx[sl]).to(device)
            mask = torch.from_numpy(data.mask[sl]).to(device)
            pos = torch.from_numpy(data.target[sl]).to(device)
            neg = torch.from_numpy(negatives[sl]).to(device)
            b = len(ctx)
            h = model.encode_context(ctx, mask)
            cp = model.encode_candidate(pos)
            cn = model.encode_candidate(neg.reshape(-1, neg.size(-1))).reshape(
                b, RANKER_NEGATIVES, -1)
            logits = torch.cat(
                [(h * cp).sum(-1, keepdim=True),
                 torch.einsum("bd,bnd->bn", h, cn)], dim=1)
            logits = logits * model.logit_scale.clamp(1.0, 100.0)
            labels = torch.zeros(b, dtype=torch.long, device=device)
            total += float(F.cross_entropy(logits, labels, reduction="sum"))
            n += b
    return total / max(n, 1)


def train_ranker_model(dataset: dict, args: argparse.Namespace,
                       out_path: Path, log_path: Path,
                       classifier_ckpt: str | Path) -> dict:
    started = time.time()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device))
    set_seed(args.seed)
    ck = torch.load(classifier_ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    mean, std = cfg.get("mean"), cfg.get("std")
    if mean is None or std is None:
        raise ValueError(
            f"{classifier_ckpt}: missing cfg.mean/cfg.std; train the classifier first")
    val_ids = set(map(str, ck.get("val_track_ids") or []))
    tracks, feature_dim = build_track_data(dataset)
    if not val_ids:
        raise ValueError(f"{classifier_ckpt}: missing val_track_ids")
    train_ids = {t.track_id for t in tracks} - val_ids
    context_len = int(getattr(args, "ranker_context_len", CTX_LEN))
    margin = int(getattr(args, "ranker_hard_margin", 1))

    train_data = build_ranker_data(tracks, train_ids,
                                   max_contexts=args.max_contexts,
                                   seed=args.seed, context_len=context_len)
    val_data = build_ranker_data(tracks, val_ids, seed=args.seed,
                                 context_len=context_len)
    print(f"[ranker] contexts train={len(train_data.ctx)} "
          f"val={len(val_data.ctx)} tracks={train_data.counts['tracks']}/"
          f"{val_data.counts['tracks']} len={context_len} margin={margin}",
          flush=True)
    print(f"[ranker] context lengths train="
          f"{train_data.counts['context_lengths']}", flush=True)

    style_pool_train: dict[str, list[int]] = defaultdict(list)
    style_pool_val: dict[str, list[int]] = defaultdict(list)
    for t in tracks:
        if t.track_id in train_ids:
            style_pool_train[t.style].append(t.idx)
        elif t.track_id in val_ids:
            style_pool_val[t.style].append(t.idx)

    model = PairwiseRanker(
        feature_dim=feature_dim, d_model=args.ranker_d_model,
        n_layers=args.ranker_layers, n_heads=args.ranker_heads,
        max_len=context_len, dropout=args.dropout, init_scale=10.0,
        mean=mean, std=std).to(device)
    print(f"[ranker] params={model.n_params():,} device={device} "
          f"feature_dim={feature_dim}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ranker_lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.ranker_epochs, 1),
        eta_min=args.ranker_lr * 0.01)
    val_rng = random.Random(args.seed + 7)
    val_negatives = _sample_negatives(
        val_data, np.arange(len(val_data.ctx)), tracks, style_pool_val,
        val_rng, margin=margin)

    history: list[dict] = []
    for epoch in range(1, args.ranker_epochs + 1):
        model.train()
        perm = torch.randperm(len(train_data.ctx)).numpy()
        total = 0.0
        seen = 0
        for start in range(0, len(perm), args.ranker_batch_size):
            idx = perm[start:start + args.ranker_batch_size]
            negatives = _sample_negatives(train_data, idx, tracks,
                                          style_pool_train,
                                          random.Random(args.seed + epoch * 100003
                                                        + start),
                                          margin=margin)
            ctx = torch.from_numpy(train_data.ctx[idx]).to(device)
            mask = torch.from_numpy(train_data.mask[idx]).to(device)
            pos = torch.from_numpy(train_data.target[idx]).to(device)
            neg = torch.from_numpy(negatives).to(device)
            b = len(ctx)
            h = model.encode_context(ctx, mask)
            cp = model.encode_candidate(pos)
            cn = model.encode_candidate(neg.reshape(-1, neg.size(-1))).reshape(
                b, RANKER_NEGATIVES, -1)
            logits = torch.cat(
                [(h * cp).sum(-1, keepdim=True),
                 torch.einsum("bd,bnd->bn", h, cn)], dim=1)
            logits = logits * model.logit_scale.clamp(1.0, 100.0)
            labels = torch.zeros(b, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * b
            seen += b
        scheduler.step()
        train_loss = total / max(seen, 1)
        val_loss = _ranker_val_loss(model, val_data, val_negatives, device,
                                    args.ranker_batch_size * 4)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "logit_scale": round(float(model.logit_scale.detach().cpu()), 6),
            "lr": round(float(scheduler.get_last_lr()[0]), 8),
        }
        history.append(row)
        print(f"[ranker] epoch {epoch:02d}/{args.ranker_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"scale={row['logit_scale']:.2f}", flush=True)

    retrieval = ranker_retrieval_metrics(model, val_data, tracks, device)
    cfg_out = dict(model.cfg)
    ckpt = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": cfg_out,
        "val_loss": float(history[-1]["val_loss"]) if history else float("nan"),
        "val_top1": retrieval["top1"],
        "val_top5": retrieval["top5"],
        "val_track_ids": sorted(val_ids),
        "seed": int(args.seed),
        "feature_dim": int(feature_dim),
        "params": int(model.n_params()),
        "trained_on": str(getattr(args, "dataset_path", "")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    log = {
        "version": 1,
        "model": "taste_ranker_v1",
        "seed": int(args.seed),
        "dataset": str(getattr(args, "dataset_path", "")),
        "classifier_ckpt": str(classifier_ckpt),
        "counts": {
            "train": train_data.counts,
            "val": val_data.counts,
            "negatives_per_context": RANKER_NEGATIVES,
            "negative_mix": "50/50 same-track (not within +/-%d) / same-style track"
                            % margin,
            "negative_margin": int(margin),
            "context_len": int(context_len),
        },
        "model_cfg": cfg_out,
        "params": int(model.n_params()),
        "device": str(device),
        "epochs": history,
        "retrieval": retrieval,
        "duration_sec": round(time.time() - started, 2),
        "dataset_stats": dataset.get("stats", {}),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[ranker] saved {out_path} ({out_path.stat().st_size/1e6:.2f} MB) "
          f"log={log_path}", flush=True)
    print(f"[ranker] val top1={retrieval['top1']:.4f} "
          f"top5={retrieval['top5']:.4f}", flush=True)
    return {"checkpoint": str(out_path), "log": str(log_path),
            "retrieval": retrieval, "params": model.n_params(),
            "counts": log["counts"]}


# --------------------------------------------------------------------------
# synthetic fixture / self test
# --------------------------------------------------------------------------
def synthetic_dataset(n_tracks: int = 40, n_windows: int = 24,
                      seed: int = SEED) -> dict:
    """Small deterministic dataset with a learnable temporal signature.

    Each track has a base timbre plus a shared smooth temporal arc; real
    sequences follow the arc, shuffled sequences do not, and cross-track
    sequences carry the donor's timbre.
    """
    rng = np.random.default_rng(seed)
    styles = ("house", "breaks", "wave", "cyber")
    sections = ("INTRO", "GROOVE_A", "DROP", "BREAKDOWN")
    bars = [4 * k for k in range(n_windows)]
    arc = np.stack([np.sin(np.arange(FEATURE_DIM) * 0.17 + k * 0.42)
                    for k in range(n_windows)]).astype(np.float32)
    tracks = []
    kit_phrases = []
    for ti in range(n_tracks):
        tid = f"syn{ti:04d}"
        style = styles[ti % len(styles)]
        base = rng.normal(0, 0.55, FEATURE_DIM).astype(np.float32)
        windows = []
        for k in range(n_windows):
            sec = sections[min(k * len(sections) // n_windows, 3)]
            energy = round(k / (n_windows - 1), 4)
            feat = (0.6 * base + arc[k] +
                    0.12 * rng.normal(size=FEATURE_DIM)).astype(np.float32)
            windows.append({
                "start_bar": bars[k],
                "section": sec,
                "energy": energy,
                "features": [round(float(v), 5) for v in feat],
            })
        tracks.append({
            "track_id": tid, "style": style,
            "bpm": 90.0 + (ti % 17) * 3.0,
            "key": ["A minor", "C major", "F# minor"][ti % 3],
            "windows": windows,
        })
        for k in range(2, n_windows, 4):
            kit_phrases.append({
                "phrase_id": f"kit_{tid}_p{k:02d}",
                "track_id": tid,
                "start_bar": bars[k],
                "features": windows[k]["features"],
            })
    return {"version": 1, "feature_dim": FEATURE_DIM, "tracks": tracks,
            "kit_phrases": kit_phrases, "stats": {"synthetic": True}}


def _self_test() -> int:
    print("[self-test] synthetic preparation ...", flush=True)
    ds = synthetic_dataset()
    prep = prepare(ds)
    c = prep.counts
    assert prep.X.shape[1:] == (SEQ_LEN, FEATURE_DIM), prep.X.shape
    assert len(prep.X) == len(prep.y) == len(prep.kinds)
    assert c["positives"] > 0 and c["negatives"] > 0
    for kind in KINDS:
        assert c["by_kind"][kind] > 0, f"no {kind} negatives"
    assert c["positives"] + c["negatives"] == c["samples"]
    # no track shared between splits
    train_ids = set(prep.track_ids[prep.train_idx].tolist())
    val_ids = set(prep.track_ids[prep.val_idx].tolist())
    assert not (train_ids & val_ids), "track leak across splits"
    # first record is a real sequence, immediately followed by its shuffle;
    # the cross-track negative must come from a different track
    assert prep.kinds[0] == "real" and prep.y[0] == 1.0
    assert prep.kinds[1] == "shuffle" and prep.y[1] == 0.0
    assert not np.array_equal(prep.X[0], prep.X[1])
    group = list(prep.kinds[:5])
    xt = group.index("cross_track") if "cross_track" in group else 3
    assert prep.source_track_ids[xt] != prep.source_track_ids[0], \
        "cross-track shares source track"

    # The real dataset interleaves off-grid kit windows (aligned=false) with
    # the 4-bar grid; those must not break the +4 stride.
    off_ds = {"feature_dim": FEATURE_DIM, "tracks": [{
        "track_id": "offgrid", "style": "house", "bpm": 120.0, "key": "",
        "windows": [
            {"start_bar": b, "section": "GROOVE_A", "energy": 0.5,
             "aligned": (b % 4 == 0), "features": [float(b)] * FEATURE_DIM}
            for b in (0, 2, 4, 8, 10, 12, 16, 20, 24, 28)
        ]}]}
    off_tracks, _ = build_track_data(off_ds)
    assert list(off_tracks[0].bars) == [0, 4, 8, 12, 16, 20, 24, 28], \
        off_tracks[0].bars
    assert off_tracks[0].spans == [(0, 8)], off_tracks[0].spans

    print("[self-test] tiny training run ...", flush=True)
    with tempfile.TemporaryDirectory(prefix="taste_self_test_") as td:
        tmp = Path(td)
        args = argparse.Namespace(
            seed=SEED, epochs=10, batch_size=64, lr=1e-3, weight_decay=1e-2,
            device="cpu", d_model=64, n_layers=2, n_heads=4,
            dim_feedforward=256, dropout=0.05, max_sequences=None,
            dataset_path="synthetic")
        res = train_model(ds, args, tmp / "taste_model.pt",
                          tmp / "taste_train_log.json")
        m = res["metrics"]
        assert 0.0 <= m["auc"] <= 1.0 and math.isfinite(m["auc"])
        assert m["auc"] > 0.7, f"synthetic AUC unexpectedly low: {m['auc']}"
        assert set(m["auc_by_corruption"]) == set(KINDS)
        assert m["within_style"]["style"] is not None
        assert res["params"] > 0
        log = json.loads(Path(res["log"]).read_text())
        assert len(log["epochs"]) == 10
        assert math.isfinite(log["epochs"][-1]["train_loss"])
        ckpt = torch.load(res["checkpoint"], map_location="cpu",
                          weights_only=False)
        assert set(("model", "cfg", "val_auc")) <= set(ckpt)

        print("[self-test] tiny ranker training run ...", flush=True)
        ranker_args = argparse.Namespace(
            seed=SEED, ranker_epochs=3, ranker_batch_size=16,
            ranker_lr=1e-3, weight_decay=1e-2, device="cpu", dropout=0.05,
            ranker_d_model=32, ranker_layers=1, ranker_heads=2,
            max_contexts=None, dataset_path="synthetic")
        rres = train_ranker_model(
            ds, ranker_args, tmp / "taste_ranker.pt",
            tmp / "taste_ranker_train_log.json", res["checkpoint"])
        rck = torch.load(rres["checkpoint"], map_location="cpu",
                         weights_only=False)
        assert set(("model", "cfg", "val_top1", "val_top5")) <= set(rck)
        assert len(rck["cfg"]["mean"]) == FEATURE_DIM
        assert len(rck["cfg"]["std"]) == FEATURE_DIM
        assert 0.0 <= rck["val_top1"] <= 1.0
        assert 0.0 <= rck["val_top5"] <= 1.0
        rlog = json.loads(Path(rres["log"]).read_text())
        assert len(rlog["epochs"]) == 3
        assert math.isfinite(rlog["epochs"][-1]["train_loss"])
    print(f"[self-test] OK params={res['params']} val_auc={m['auc']:.4f} "
          f"within-style={m['within_style']['auc']:.4f} "
          f"ranker top1={rck['val_top1']:.3f} top5={rck['val_top5']:.3f}",
          flush=True)
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/tmp/taste_dataset.json")
    ap.add_argument("--out", default="data/taste_model.pt")
    ap.add_argument("--log", default="data/taste_train_log.json")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=896)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-sequences", type=int, default=None,
                    help="debug: cap the number of anchor sequences")
    ap.add_argument("--train-ranker", action="store_true",
                    help="train the pairwise transition ranker instead of the classifier")
    ap.add_argument("--classifier-ckpt", default="data/taste_model.pt")
    ap.add_argument("--ranker-out", default="data/taste_ranker.pt")
    ap.add_argument("--ranker-log", default="data/taste_ranker_train_log.json")
    ap.add_argument("--ranker-epochs", type=int, default=20)
    ap.add_argument("--ranker-batch-size", type=int, default=128)
    ap.add_argument("--ranker-lr", type=float, default=3e-4)
    ap.add_argument("--ranker-d-model", type=int, default=128)
    ap.add_argument("--ranker-layers", type=int, default=3)
    ap.add_argument("--ranker-heads", type=int, default=4)
    ap.add_argument("--ranker-context-len", type=int, default=CTX_LEN)
    ap.add_argument("--ranker-hard-margin", type=int, default=1,
                    help="same-track negatives must be farther than this")
    ap.add_argument("--max-contexts", type=int, default=None,
                    help="debug: cap the number of ranker contexts")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    dataset_path = Path(args.dataset)
    print(f"[taste] loading {dataset_path}", flush=True)
    dataset = load_dataset(dataset_path)
    args.dataset_path = str(dataset_path)
    if args.train_ranker:
        train_ranker_model(dataset, args, _resolve(args.ranker_out),
                           _resolve(args.ranker_log),
                           _resolve(args.classifier_ckpt))
        return 0
    train_model(dataset, args, _resolve(args.out), _resolve(args.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
