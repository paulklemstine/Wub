"""Phase 2B: multi-stream VQ-Transformer ("WubGPT").

Tokens per tatum step: s_t = (r_t, c_drums, c_bass, c_other, c_vocals),
where r_t in 0..15 is a metrical position token and each c_stem is a
per-stem vector-quantized slice token (K clusters + 1 silence token).

Optional conditioning streams (input-only, never predicted):
  style  - per-track style-cluster id (0 = unknown), constant per track
  energy - per-step loudness bucket 0..7 (fixed dB edges, see ENERGY_EDGES)
  bar8   - bar_position % 8 (phrase position; teaches 8-bar form)
  bound  - 1 on rows straddling a cross-song splice, else 0

Pipeline: markov.build_pools -> StemVQ.fit per stem -> build_dataset ->
WubGPT + train_model -> sample -> tokens_to_slices (nearest SliceRef).
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # runtime duck-typing only; avoids any import cycle
    from wub_core.model_markov import SliceRef, StemPool

__all__ = [
    "StemVQ",
    "build_dataset",
    "build_transition_rows",
    "track_meta",
    "select_transition_pairs_from_metas",
    "dataset_token_histogram",
    "energy_bucket",
    "select_transition_pairs",
    "WubGPT",
    "train_model",
    "sample",
    "tokens_to_slices",
    "save_checkpoint",
    "load_checkpoint",
    "slice_feature_raw",
    "STREAM_ORDER",
    "STEM_ORDER",
    "COND_ORDER",
    "N_DENSITY",
    "FEAT_DIM",
    "N_ENERGY",
    "N_BAR8",
    "N_BOUND",
    "ENERGY_EDGES",
]

TIMBRE_DIM = 12
CHROMA_DIM = 12
FEAT_DIM = 26  # [timbre12 || chroma12 || loud_max1 || log_dur1]
# Dims standardized with a per-stem StandardScaler (timbre + loudness + log_dur;
# chroma passes through, matching the indexer convention).
_STD_IDX = list(range(12)) + [24, 25]
_R_VOCAB = 16

STREAM_ORDER = ["r", "drums", "bass", "other", "vocals"]
STEM_ORDER = ["drums", "bass", "other", "vocals"]

# Conditioning streams (input-only; WubGPT never predicts these).
COND_ORDER = ["style", "energy", "bar8", "bound", "density"]
N_ENERGY = 8
N_BAR8 = 8
N_BOUND = 2
N_DENSITY = 5  # count of audible (present + non-silent) stems, 0..4
# Fixed dBFS edges -> 8 buckets (index 0 = near-silence, 7 = full-scale).
ENERGY_EDGES = (-90.0, -48.0, -40.0, -34.0, -28.0, -22.0, -16.0, -10.0, 0.0)


def energy_bucket(loud_db: float) -> int:
    """Map a loudness value in dBFS to an energy bucket 0..N_ENERGY-1."""
    try:
        v = float(loud_db)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(v):
        return 0
    idx = bisect_right(ENERGY_EDGES, v) - 1
    return int(min(max(idx, 0), N_ENERGY - 1))


def slice_feature_raw(obj) -> np.ndarray:
    """26-D raw feature vector for a SliceRef / Segment / equivalent dict."""
    if isinstance(obj, dict):
        timbre = np.asarray(obj.get("timbre"), dtype=np.float32)
        chroma = np.asarray(obj.get("chroma", obj.get("pitches")), dtype=np.float32)
        loud = obj.get("loud_max", obj.get("loudness_max", -20.0))
        dur = obj.get("duration", 0.25)
    else:
        timbre = np.asarray(obj.timbre, dtype=np.float32)
        chroma = np.asarray(
            getattr(obj, "chroma", getattr(obj, "pitches", None)), dtype=np.float32
        )
        loud = getattr(obj, "loud_max", getattr(obj, "loudness_max", -20.0))
        dur = getattr(obj, "duration", 0.25)
    if timbre.shape != (12,) or chroma.shape != (12,):
        raise ValueError("slice features must be timbre(12) + chroma(12) + loud(1) + dur(1)")
    log_dur = float(np.log1p(max(0.01, float(dur) if dur is not None else 0.25)))
    return np.concatenate([timbre, chroma, [float(loud), log_dur]]).astype(np.float32)


class StemVQ:
    """Per-stem KMeans codebook over standardized 25-D slice features."""

    def __init__(self, stem: str, k: int = 256):
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.stem = stem
        self.k = int(k)
        self.n_clusters = 0  # effective k after capping at n_samples
        self.scaler = StandardScaler()
        self.mean_ = np.zeros(len(_STD_IDX), dtype=np.float64)
        self.scale_ = np.ones(len(_STD_IDX), dtype=np.float64)
        self.centers_ = np.zeros((0, FEAT_DIM), dtype=np.float32)
        self.kmeans: Optional[KMeans] = None
        self.fitted = False

    @property
    def silence_id(self) -> int:
        return self.n_clusters

    @property
    def vocab_size(self) -> int:
        return self.n_clusters + 1

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.size == 0:
            return X
        out = X.copy()
        out[:, _STD_IDX] = (
            (X[:, _STD_IDX].astype(np.float64) - self.mean_) / self.scale_
        ).astype(np.float32)
        return out

    def fit(self, slices) -> "StemVQ":
        slices = list(slices)
        if slices:
            X = np.stack([slice_feature_raw(s) for s in slices]).astype(np.float32)
        else:
            X = np.zeros((0, FEAT_DIM), dtype=np.float32)
        n = X.shape[0]
        self.n_clusters = int(min(self.k, n))
        if n > 0:
            self.scaler.fit(X[:, _STD_IDX].astype(np.float64))
            self.mean_ = np.asarray(self.scaler.mean_, dtype=np.float64)
            self.scale_ = np.asarray(self.scaler.scale_, dtype=np.float64)
        if self.n_clusters > 0:
            Xs = self.transform(X)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
            self.kmeans.fit(Xs)
            self.centers_ = np.asarray(self.kmeans.cluster_centers_, dtype=np.float32)
        else:
            self.kmeans = None
            self.centers_ = np.zeros((0, FEAT_DIM), dtype=np.float32)
        self.fitted = True
        return self

    def encode(self, sl) -> int:
        if self.n_clusters == 0 or self.centers_.shape[0] == 0:
            return self.silence_id
        x = self.transform(slice_feature_raw(sl)[None, :])
        d = ((self.centers_ - x) ** 2).sum(axis=-1)
        return int(np.argmin(d))

    def centroid(self, token: int) -> np.ndarray:
        """Raw-space (inverse-standardized) centroid; zeros for silence."""
        if (
            self.n_clusters == 0
            or int(token) == self.silence_id
            or int(token) < 0
            or int(token) >= self.n_clusters
            or self.centers_.shape[0] == 0
        ):
            return np.zeros(FEAT_DIM, dtype=np.float32)
        c = self.centers_[int(token)].astype(np.float64).copy()
        c[_STD_IDX] = c[_STD_IDX] * self.scale_ + self.mean_
        return c.astype(np.float32)

    def save(self, dir) -> Path:
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(
            d / "vq.npz",
            centers=self.centers_,
            scaler_mean=self.mean_,
            scaler_scale=self.scale_,
        )
        (d / "vq.json").write_text(
            json.dumps({"stem": self.stem, "k": self.k, "n_clusters": self.n_clusters})
        )
        return d

    @classmethod
    def load(cls, dir) -> "StemVQ":
        d = Path(dir)
        cfg = json.loads((d / "vq.json").read_text())
        z = np.load(d / "vq.npz")
        vq = cls(cfg["stem"], cfg["k"])
        vq.n_clusters = int(cfg["n_clusters"])
        vq.mean_ = np.asarray(z["scaler_mean"], dtype=np.float64)
        vq.scale_ = np.asarray(z["scaler_scale"], dtype=np.float64)
        vq.centers_ = np.asarray(z["centers"], dtype=np.float32)
        sc = StandardScaler()
        sc.mean_ = vq.mean_
        sc.scale_ = vq.scale_
        sc.var_ = vq.scale_ ** 2
        sc.n_features_in_ = len(_STD_IDX)
        sc.n_samples_seen_ = 1
        vq.scaler = sc
        if vq.n_clusters > 0:
            km = KMeans(n_clusters=vq.n_clusters, n_init=10, random_state=0)
            km.cluster_centers_ = np.asarray(vq.centers_)
            km.n_features_in_ = FEAT_DIM
            vq.kmeans = km
        vq.fitted = True
        return vq


def build_dataset(
    analyses,
    vqs: Dict[str, StemVQ],
    stride: int = 1,
    style_map: Optional[Dict[str, int]] = None,
    n_styles: int = 12,
) -> List[Dict[str, int]]:
    """One token row per tatum: {r, drums, bass, other, vocals} + cond streams.

    Cond streams (always emitted; harmless when the model ignores them):
      style  - style_map[track_id] clamped to 1..n_styles, else 0 (unknown)
      energy - bucketed max segment loudness for this tatum (0..7)
      bar8   - bar_position % 8 (phrase position)
      bound  - 0 (cross-song splices set 1 near the junction; see
               build_transition_rows)
      density - count of audible (present and non-silent) stems, 0..4.
               Corpus ladder for drum activity: d1~0%, d2~18%, d3~73%,
               d4=100%. This is the primary "beats move with energy" knob;
               loudness buckets (energy) are weaker and non-monotonic.

    r = beat_position * 4 + tatum_position (0..15). Per stem, the first
    segment id in tatum.stem_segments[stem] is VQ-encoded; missing segment
    ids, silent segments, or stems absent from the track yield silence.
    stride>1 keeps every Nth tatum (full-corpus training on small boxes).
    """
    rows: List[Dict[str, int]] = []
    for tr in analyses:
        seg_by_id = {s.id: s for s in tr.segments}
        available = set(tr.stems_available or [])
        track_id = getattr(tr, "track_id", "")
        style = 0
        if style_map:
            try:
                style = int(style_map.get(track_id, 0))
            except (TypeError, ValueError):
                style = 0
            style = style if 1 <= style <= int(n_styles) else 0
        for tat in tr.tatums[:: max(1, int(stride))]:
            beat_pos = getattr(tat, "beat_position", tat.get("beat_position", 0) if isinstance(tat, dict) else 0)
            tatum_pos = getattr(tat, "tatum_position", tat.get("tatum_position", 0) if isinstance(tat, dict) else 0)
            r = (int(beat_pos) * 4 + int(tatum_pos)) % _R_VOCAB
            row: Dict[str, int] = {"r": int(r)}
            segmap = tat.stem_segments or {}
            loud_peak = float("-inf")
            audible = 0
            for stem in STEM_ORDER:
                vq = vqs.get(stem)
                sil = int(vq.silence_id) if vq is not None else 0
                # Ritual: stems the track lacks stay silent (no lookup).
                if stem not in available:
                    row[stem] = sil
                    continue
                ids = segmap.get(stem) or []
                if not ids:
                    row[stem] = sil
                    continue
                sg = seg_by_id.get(ids[0])
                if sg is None or bool(getattr(sg, "is_silent", False)) or vq is None:
                    row[stem] = sil
                    continue
                row[stem] = int(vq.encode(sg))
                audible += 1
                try:
                    loud_peak = max(loud_peak, float(getattr(sg, "loudness_max", float("-inf"))))
                except (TypeError, ValueError):
                    pass
            if not np.isfinite(loud_peak):
                loud_peak = -90.0
            row["style"] = int(style)
            row["energy"] = int(energy_bucket(loud_peak))
            row["density"] = int(min(max(audible, 0), N_DENSITY - 1))
            try:
                bar_pos = int(getattr(tat, "bar_position", tat.get("bar_position", 0) if isinstance(tat, dict) else 0))
            except (TypeError, ValueError):
                bar_pos = 0
            row["bar8"] = int(bar_pos % N_BAR8)
            row["bound"] = 0
            rows.append(row)
    return rows


def dataset_token_histogram(
    dataset: List[Dict[str, int]], streams: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Raw count histogram per stream over a token-row dataset."""
    if streams is None:
        streams = list(STREAM_ORDER)
    hists: Dict[str, np.ndarray] = {}
    for s in streams:
        vals = [int(row.get(s, 0)) for row in dataset]
        hists[s] = np.bincount(vals, minlength=(max(vals) + 1) if vals else 1).astype(np.float64)
    return hists


def _parse_key_root(key: Optional[str]) -> str:
    """Normalize 'E minor' -> 'e'; unparseable -> '' (never blocks pairing)."""
    if not key or not isinstance(key, str):
        return ""
    return key.strip().split()[0].lower()


def track_meta(tr) -> Optional[Dict[str, object]]:
    """Tiny per-track record for pairing: id, bpm, key root, mean loudness.

    Lets callers scan a corpus in streaming chunks (parse, extract 4 numbers,
    discard) instead of holding every parsed track in RAM at once.
    """
    tid = getattr(tr, "track_id", "")
    if not tid:
        return None
    try:
        bpm = float(getattr(tr, "bpm", 0.0) or 0.0)
    except (TypeError, ValueError):
        bpm = 0.0
    if bpm <= 0:
        return None
    tot, n = 0.0, 0
    for sg in getattr(tr, "segments", []) or []:
        try:
            v = float(getattr(sg, "loudness_max", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            tot += v
            n += 1
    return {
        "id": str(tid),
        "bpm": bpm,
        "key": _parse_key_root(getattr(tr, "key", "")),
        "loud": (tot / n) if n else -90.0,
        "style": 0,
    }


def select_transition_pairs(
    analyses,
    style_map: Optional[Dict[str, int]] = None,
    n_pairs: int = 64,
    seed: int = 0,
    max_tempo_drift: float = 0.06,
    max_energy_gap_db: float = 6.0,
    require_same_key: bool = True,
) -> List[Tuple[str, str]]:
    """Pick (A_id, B_id) cross-song splice pairs behind DJ gates.

    Gates: different tracks; tempo ratio within max_tempo_drift; same key
    root (unless require_same_key=False); mean segment loudness within
    max_energy_gap_db. Deterministic in (sorted ids, seed). Style is NOT
    gated (cross-style splices are the point) but pairs are spread across
    style pairs when a style_map is given.
    """
    metas: List[Dict[str, object]] = []
    for tr in analyses:
        m = track_meta(tr)
        if m is None:
            continue
        if style_map:
            try:
                m["style"] = int(style_map.get(str(m["id"]), 0))
            except (TypeError, ValueError):
                pass
        metas.append(m)
    return select_transition_pairs_from_metas(
        metas,
        n_pairs=n_pairs,
        seed=seed,
        max_tempo_drift=max_tempo_drift,
        max_energy_gap_db=max_energy_gap_db,
        require_same_key=require_same_key,
    )


def select_transition_pairs_from_metas(
    metas: List[Dict[str, object]],
    n_pairs: int = 64,
    seed: int = 0,
    max_tempo_drift: float = 0.06,
    max_energy_gap_db: float = 6.0,
    require_same_key: bool = True,
) -> List[Tuple[str, str]]:
    """Pair selection over lightweight track_meta() records (streaming-safe).

    Metas need {id, bpm, key, loud} with optional {style}. Same gates and
    deterministic ordering as select_transition_pairs; use this when the
    corpus cannot fit parsed in RAM (scan chunks, keep metas, load only
    the paired tracks afterwards).
    """
    import random

    infos = [dict(m) for m in metas if m.get("id") and (m.get("bpm") or 0) > 0]
    infos.sort(key=lambda d: d["id"])
    rng = random.Random(int(seed))
    cands: List[Tuple[str, str]] = []
    for i, a in enumerate(infos):
        for b in infos[i + 1 :]:
            if require_same_key and a["key"] and b["key"] and a["key"] != b["key"]:
                continue
            if abs(a["bpm"] - b["bpm"]) / max(a["bpm"], b["bpm"]) > max_tempo_drift:
                continue
            if abs(a["loud"] - b["loud"]) > max_energy_gap_db:
                continue
            cands.append((a["id"], b["id"]) if rng.random() < 0.5 else (b["id"], a["id"]))
    # Spread across style pairs when labels exist.
    if any(int(m.get("style") or 0) != 0 for m in infos):
        by_pair: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}
        sm = {str(m["id"]): int(m.get("style") or 0) for m in infos}
        for p in cands:
            by_pair.setdefault((sm.get(p[0], 0), sm.get(p[1], 0)), []).append(p)
        ordered: List[Tuple[str, str]] = []
        groups = list(by_pair.values())
        rng.shuffle(groups)
        idx = 0
        while len(ordered) < len(cands):
            grew = False
            for g in groups:
                if idx < len(g):
                    ordered.append(g[idx])
                    grew = True
            if not grew:
                break
            idx += 1
        cands = ordered
    else:
        rng.shuffle(cands)
    return cands[: max(0, int(n_pairs))]


def build_transition_rows(
    analyses_by_id: Dict[str, object],
    vqs: Dict[str, StemVQ],
    pairs: List[Tuple[str, str]],
    tail: int = 32,
    head: int = 64,
    bound_width: int = 16,
    style_map: Optional[Dict[str, int]] = None,
    n_styles: int = 12,
) -> List[Dict[str, int]]:
    """Concatenate tail(A) + head(B) rows with a bound=1 junction window.

    The destination style (style of B) labels every row: the model learns
    "steer toward style B across a marked junction". bar8/energy keep their
    per-side values (the junction discontinuity is exactly what bound=1
    marks). Rows are plain build_dataset dicts; unknown ids are skipped.
    """
    rows: List[Dict[str, int]] = []
    half = max(1, int(bound_width) // 2)
    for a_id, b_id in pairs:
        a = analyses_by_id.get(a_id)
        b = analyses_by_id.get(b_id)
        if a is None or b is None:
            continue
        ra = build_dataset([a], vqs, stride=1, style_map=style_map, n_styles=n_styles)
        rb = build_dataset([b], vqs, stride=1, style_map=style_map, n_styles=n_styles)
        if len(ra) < max(1, int(tail)) or len(rb) < max(1, int(head)):
            continue
        seg = ra[-int(tail):] + rb[: int(head)]
        try:
            style_b = int((style_map or {}).get(b_id, 0))
        except (TypeError, ValueError):
            style_b = 0
        style_b = style_b if 1 <= style_b <= int(n_styles) else 0
        j = int(tail)
        for k, row in enumerate(seg):
            row = dict(row)
            row["style"] = style_b
            row["bound"] = 1 if (j - half) <= k < (j + half) else 0
            rows.append(row)
    return rows


class WubGPT(nn.Module):
    """5-stream causal Transformer: per-step tokens (r + 4 stems).

    Each step contributes 5 tokens (own embedding table each) plus learned
    stream-type and positional embeddings; the flattened (seq*5) sequence
    goes through a causal TransformerEncoder, and 5 heads predict the NEXT
    step's tokens from the step's final-stream hidden state.

    Optional conditioning (cond_vocabs, e.g. style/energy/bar8/bound):
    per-step cond embeddings are summed and added to every stream state of
    that step. Cond streams are INPUTS only -- no heads predict them.
    A model built without cond_vocabs behaves exactly like the legacy model
    and loads legacy checkpoints.
    """

    STREAMS = tuple(STREAM_ORDER)
    COND_STREAMS = tuple(COND_ORDER)

    def __init__(
        self,
        vocabs: Optional[Dict[str, int]] = None,
        k: int = 256,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        ff: int = 1024,
        dropout: float = 0.1,
        max_steps: int = 1024,
        cond_vocabs: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        if vocabs is None:
            vocabs = {"r": _R_VOCAB, **{s: int(k) + 1 for s in STEM_ORDER}}
        vocabs = {s: int(v) for s, v in vocabs.items()}
        for s in self.STREAMS:
            if s not in vocabs or vocabs[s] < 1:
                raise ValueError(f"vocabs must define a positive size for {s!r}")
        self.vocabs = vocabs
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ff = int(ff)
        self.dropout = float(dropout)
        self.max_steps = int(max_steps)
        self.max_len = self.max_steps * len(self.STREAMS)
        if cond_vocabs is not None:
            cond_vocabs = {c: int(v) for c, v in cond_vocabs.items() if int(v) >= 1}
            unknown = [c for c in cond_vocabs if c not in self.COND_STREAMS]
            if unknown:
                raise ValueError(f"unknown cond streams: {unknown!r}")
        self.cond_vocabs = cond_vocabs

        self.emb = nn.ModuleDict(
            {s: nn.Embedding(vocabs[s], d_model) for s in self.STREAMS}
        )
        self.cond_emb = nn.ModuleDict(
            {c: nn.Embedding(v, d_model) for c, v in (cond_vocabs or {}).items()}
        )
        self.stream_emb = nn.Embedding(len(self.STREAMS), d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {s: nn.Linear(d_model, vocabs[s]) for s in self.STREAMS}
        )

    def get_config(self) -> dict:
        return {
            "vocabs": dict(self.vocabs),
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ff": self.ff,
            "dropout": self.dropout,
            "max_steps": self.max_steps,
            "cond_vocabs": dict(self.cond_vocabs) if self.cond_vocabs else None,
        }

    @classmethod
    def from_config(cls, config: dict) -> "WubGPT":
        return cls(
            vocabs=config["vocabs"],
            d_model=config.get("d_model", 384),
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 8),
            ff=config.get("ff", 1024),
            dropout=config.get("dropout", 0.1),
            max_steps=config.get("max_steps", 1024),
            cond_vocabs=config.get("cond_vocabs"),
        )

    @classmethod
    def from_vqs(
        cls, vqs: Dict[str, StemVQ], cond_vocabs: Optional[Dict[str, int]] = None, **kw
    ) -> "WubGPT":
        vocabs: Dict[str, int] = {"r": _R_VOCAB}
        for s in STEM_ORDER:
            vocabs[s] = int(vqs[s].vocab_size) if s in vqs else 1
        return cls(vocabs=vocabs, cond_vocabs=cond_vocabs, **kw)

    def _cond_sum(self, conds: Dict[str, Optional[torch.Tensor]], dev) -> Optional[torch.Tensor]:
        if not self.cond_emb:
            return None
        parts = []
        B = T = None
        for c, emb in self.cond_emb.items():
            t = conds.get(c)
            if t is None:
                continue
            t = t.to(dev)
            if B is None:
                B, T = t.shape
            parts.append(emb(t))
        if not parts:
            return None
        return sum(parts)

    def forward(self, r, drums=None, bass=None, other=None, vocals=None,
                style=None, energy=None, bar8=None, bound=None, density=None):
        if isinstance(r, dict):
            d = dict(r)
            drums = d.get("drums")
            bass = d.get("bass")
            other = d.get("other")
            vocals = d.get("vocals")
            style = d.get("style")
            energy = d.get("energy")
            bar8 = d.get("bar8")
            bound = d.get("bound")
            density = d.get("density")
            r = d["r"]
        toks = [r, drums, bass, other, vocals]
        B, T = r.shape
        n_stream = len(self.STREAMS)
        L = T * n_stream
        if L > self.max_len:
            raise ValueError(f"sequence too long: {T} steps > max_steps {self.max_steps}")
        dev = r.device
        e = torch.stack(
            [self.emb[s](toks[i]) for i, s in enumerate(self.STREAMS)], dim=2
        )  # (B,T,5,d)
        sids = torch.arange(n_stream, device=dev).view(1, 1, n_stream).expand(B, T, n_stream)
        h = e + self.stream_emb(sids)
        ch = self._cond_sum(
            {"style": style, "energy": energy, "bar8": bar8,
             "bound": bound, "density": density}, dev
        )
        if ch is not None:
            if ch.shape != (B, T, self.d_model):
                raise ValueError(f"cond inputs must be (B,T); got {ch.shape}")
            h = h + ch.unsqueeze(2)
        pids = torch.arange(L, device=dev).view(1, L, 1).expand(B, L, 1)
        h = h + self.pos_emb(pids).view(B, T, n_stream, -1)
        h = self.drop(h.view(B, L, -1))
        mask = torch.zeros(L, L, device=dev)
        mask = mask.masked_fill(
            torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        z = self.encoder(h, mask=mask).view(B, T, n_stream, -1)
        last = self.norm(z[:, :, -1, :])  # final stream state predicts next step
        return {s: self.heads[s](last) for s in self.STREAMS}


def _auto_stream_weights(
    dataset: List[Dict[str, int]], vocabs: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """Inverse-sqrt-frequency weights per head, mean-normalized, clamped.

    Dominant slice clusters (the safe boring predictions) get down-weighted;
    rare clusters get up-weighted. Silence is NOT special-cased: it is a
    legitimate musical event, it just must not dominate by frequency.
    """
    weights: Dict[str, torch.Tensor] = {}
    for s in STREAM_ORDER:
        vals = [int(row.get(s, 0)) for row in dataset]
        v = int(vocabs.get(s, (max(vals) + 1) if vals else 1))
        hist = np.bincount(vals, minlength=max(v, 1)).astype(np.float64)
        w = 1.0 / np.sqrt(hist + 1.0)
        w = w / max(w.mean(), 1e-12)
        w = np.clip(w, 0.2, 5.0)
        weights[s] = torch.tensor(w, dtype=torch.float32)
    return weights


def train_model(
    model: WubGPT,
    dataset: List[Dict[str, int]],
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    device="cpu",
    seed: int = 0,
    weights=None,
    T: int = 32,
    checkpoint_cb=None,
) -> List[float]:
    """Next-step training: CE per head summed. Returns per-epoch mean losses.

    weights: None (uniform), "auto" (inverse-sqrt frequency per head), or
      {stream: weight-vector}. Cond streams present in dataset rows feed the
      model only when it was built with cond_vocabs; otherwise ignored.
    T: context window in steps (sliding windows, stride 1).
    checkpoint_cb: optional callable called as cb(epoch_1based, model, losses)
      after each epoch (e.g. to save per-epoch checkpoints on long runs).
    Throughput notes (measured 3 s/batch -> target <0.5 s on RTX 4050/WSL):
      windows are gathered with ONE advanced index per stream on pinned CPU
      memory and moved with a single non-blocking H2D copy each; the loss
      stays on-device and is synchronized once per epoch, not per batch.
    """
    if len(dataset) < 2:
        raise ValueError("need at least 2 token rows to train")
    torch.manual_seed(seed)
    dev = torch.device(device)
    model.to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    keys = list(STREAM_ORDER)
    use_cond = bool(getattr(model, "cond_vocabs", None))
    ckeys = [c for c in COND_ORDER if use_cond and c in (dataset[0] if dataset else {})]
    pin = dev.type == "cuda"
    arr = {
        kk: torch.tensor([row[kk] for row in dataset], dtype=torch.long).pin_memory()
        if pin else torch.tensor([row[kk] for row in dataset], dtype=torch.long)
        for kk in keys
    }
    cond_arr = {
        ck: (torch.tensor([int(row.get(ck, 0)) for row in dataset], dtype=torch.long).pin_memory()
             if pin else torch.tensor([int(row.get(ck, 0)) for row in dataset], dtype=torch.long))
        for ck in ckeys
    }
    if weights == "auto":
        wmap = _auto_stream_weights(dataset, model.vocabs)
    elif isinstance(weights, dict):
        wmap = {kk: torch.as_tensor(weights[kk], dtype=torch.float32) for kk in keys if kk in weights}
    else:
        wmap = {}
    wmap = {kk: w.to(dev, non_blocking=True) for kk, w in wmap.items()}
    N = len(dataset)
    T = int(min(max(1, T), N - 1))  # context window; T+1-row sliding windows, stride 1
    nwin = N - T
    steps = torch.arange(T, dtype=torch.long)
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    for _ep in range(int(epochs)):
        perm = torch.from_numpy(rng.permutation(nwin).astype(np.int64))
        tot = torch.zeros((), device=dev)
        nb = 0
        for bi in range(0, nwin, max(1, int(batch_size))):
            idx = perm[bi : bi + int(batch_size)]
            base = idx.unsqueeze(1) + steps.unsqueeze(0)  # (B,T) window starts
            inp = {kk: arr[kk][base].to(dev, non_blocking=True) for kk in keys}
            if ckeys:
                for ck in ckeys:
                    inp[ck] = cond_arr[ck][base].to(dev, non_blocking=True)
            tgt = {kk: arr[kk][base + 1].to(dev, non_blocking=True) for kk in keys}
            opt.zero_grad(set_to_none=True)
            logits = model(inp)
            loss = 0.0
            for kk in keys:
                loss = loss + F.cross_entropy(
                    logits[kk].reshape(-1, logits[kk].size(-1)),
                    tgt[kk].reshape(-1),
                    weight=wmap.get(kk),
                )
            loss.backward()
            opt.step()
            tot = tot + loss.detach()
            nb += 1
        epoch_loss = float(tot.item()) / max(nb, 1)
        losses.append(epoch_loss)
        if checkpoint_cb is not None:
            checkpoint_cb(_ep + 1, model, list(losses))
    return losses


def sample(
    model: WubGPT,
    vqs: Dict[str, StemVQ],
    seed_steps: List[Dict[str, int]],
    steps: int,
    temperature: float = 1.0,
    top_k: int = 32,
    seed: int = 0,
    top_p: Optional[float] = None,
    style_id: int = 0,
    energy: int = 5,
    energy_schedule: Optional[List[int]] = None,
    start_bar8: int = -1,
    bound_id: int = 0,
    density: int = 4,
    density_schedule: Optional[List[int]] = None,
) -> List[Dict[str, int]]:
    """Autoregressively generate `steps` new token rows after `seed_steps`.

    Conditioning (only when the model was built with cond_vocabs):
      style_id - constant style for the whole generation
      energy / energy_schedule - per-step energy bucket (schedule wins)
      start_bar8 - bar8 of the first generated step; -1 = continue the
        seed's last bar8 (explicit values always win over the seed)
      bound_id - constant bound flag (0 = inside one song)
      density / density_schedule - per-step audible-stem count 0..4
        (schedule wins). Corpus ladder: d1~0%, d2~18%, d3~73%, d4=100%
        drum activity. This is the primary beats-with-energy knob.
    Seed rows carrying cond keys keep them as context; missing keys fall
    back to the constants above.
    """
    if not seed_steps:
        raise ValueError("seed_steps must be non-empty")
    if steps < 1:
        return []
    _ = vqs  # vocab sizes live on the model; vqs kept for API symmetry
    was_training = model.training
    model.eval()
    use_cond = bool(getattr(model, "cond_vocabs", None))
    keys = list(STREAM_ORDER)
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
    else:
        gen = torch.Generator().manual_seed(int(seed))
    last_seed = seed_steps[-1]
    if int(start_bar8) >= 0:
        bar0 = int(start_bar8)
    else:
        try:
            bar0 = int(last_seed.get("bar8", 0))
        except (TypeError, ValueError):
            bar0 = 0
    hist = [dict(s) for s in seed_steps]
    for s in hist:
        s.setdefault("style", int(style_id))
        s.setdefault("energy", int(energy))
        s.setdefault("bar8", bar0)
        s.setdefault("bound", int(bound_id))
        s.setdefault("density", int(density))
    out: List[Dict[str, int]] = []
    with torch.no_grad():
        for i in range(int(steps)):
            ctx = hist[-model.max_steps :] if len(hist) > model.max_steps else hist
            inp = {
                kk: torch.tensor(
                    [[int(row[kk]) for row in ctx]], dtype=torch.long, device=dev
                )
                for kk in keys
            }
            if use_cond:
                nctx = len(ctx)
                e_i = (
                    int(energy_schedule[i])
                    if energy_schedule and i < len(energy_schedule)
                    else int(energy)
                )
                d_i = (
                    int(density_schedule[i])
                    if density_schedule and i < len(density_schedule)
                    else int(density)
                )
                inp["style"] = torch.full((1, nctx), int(style_id), dtype=torch.long, device=dev)
                inp["energy"] = torch.tensor(
                    [[int(row.get("energy", e_i)) for row in ctx]], dtype=torch.long, device=dev
                )
                inp["bar8"] = torch.tensor(
                    [[int(row.get("bar8", (bar0 + (j // 16)) % N_BAR8)) for j, row in enumerate(ctx)]],
                    dtype=torch.long, device=dev,
                )
                inp["bound"] = torch.full((1, nctx), int(bound_id), dtype=torch.long, device=dev)
                inp["density"] = torch.tensor(
                    [[int(row.get("density", d_i)) for row in ctx]], dtype=torch.long, device=dev
                )
            logits = model(inp)
            nxt: Dict[str, int] = {}
            for kk in keys:
                l = logits[kk][0, -1].float()
                V = int(l.numel())
                if temperature <= 0:
                    nxt[kk] = int(torch.argmax(l).item())
                    continue
                l = l / max(float(temperature), 1e-9)
                kk2 = min(int(top_k), V) if top_k and int(top_k) > 0 else V
                if kk2 < V:
                    thr = torch.topk(l, kk2).values[-1]
                    l = torch.where(
                        l >= thr,
                        l,
                        torch.tensor(float("-inf"), device=dev),
                    )
                    probs = torch.softmax(l, dim=-1)
                else:
                    probs = torch.softmax(l, dim=-1)
                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    cut = cum - sp > float(top_p)
                    sp = sp.masked_fill(cut, 0.0)
                    sp = sp / (sp.sum() + 1e-12)
                    nxt[kk] = int(si[int(torch.multinomial(sp, 1, generator=gen))].item())
                else:
                    nxt[kk] = int(torch.multinomial(probs, 1, generator=gen).item())
            if use_cond:
                e_i = (
                    int(energy_schedule[i])
                    if energy_schedule and i < len(energy_schedule)
                    else int(energy)
                )
                nxt["style"] = int(style_id)
                nxt["energy"] = e_i
                nxt["bar8"] = (bar0 + (i // 16)) % N_BAR8
                nxt["bound"] = int(bound_id)
                nxt["density"] = (
                    int(density_schedule[i])
                    if density_schedule and i < len(density_schedule)
                    else int(density)
                )
            hist.append(nxt)
            out.append(dict(nxt))
    if was_training:
        model.train()
    return out


def tokens_to_slices(
    rows: List[Dict[str, int]],
    pools: Dict[str, "StemPool"],
    vqs: Dict[str, StemVQ],
) -> Dict[str, List[Optional["SliceRef"]]]:
    """Map each row's stem tokens to nearest SliceRefs (silence -> None).

    Distance is euclidean in each VQ's standardized 25-D space, brute force
    over the whole pool (all tracks); returned objects are the pool's own
    SliceRef instances.
    """
    out: Dict[str, List[Optional["SliceRef"]]] = {}
    for stem, pool in pools.items():
        if stem not in vqs:
            raise KeyError(f"no StemVQ for pooled stem {stem!r}")
        vq = vqs[stem]
        if not pool.slices:
            out[stem] = [None] * len(rows)
            continue
        X = np.stack([slice_feature_raw(s) for s in pool.slices]).astype(np.float32)
        Xs = vq.transform(X)
        C = vq.centers_
        col: List[Optional["SliceRef"]] = []
        for row in rows:
            tok = int(row.get(stem, vq.silence_id))
            if tok == vq.silence_id or tok < 0 or tok >= vq.n_clusters or C.shape[0] == 0:
                col.append(None)
                continue
            d = ((Xs - C[tok]) ** 2).sum(axis=1)
            col.append(pool.slices[int(np.argmin(d))])
        out[stem] = col
    return out


def save_checkpoint(path, model: WubGPT, vqs: Dict[str, StemVQ], config: dict) -> Path:
    """torch.save(state_dict + config) plus a sibling <stem>_vq dir of StemVQs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    vq_dir = p.parent / (p.stem + "_vq")
    for stem, vq in vqs.items():
        vq.save(vq_dir / stem)
    payload = {
        "config": dict(config),
        "vocabs": dict(model.vocabs),
        "vq_dir": vq_dir.name,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, p)
    return p


def load_checkpoint(
    path, model: Optional[WubGPT] = None, vqs=None, config=None
) -> Tuple[WubGPT, Dict[str, StemVQ], dict]:
    """Inverse of save_checkpoint. Missing model/vqs/config are rebuilt."""
    p = Path(path)
    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    cfg = dict(config) if config is not None else dict(payload.get("config", {}))
    if "vocabs" not in cfg and "vocabs" in payload:
        cfg["vocabs"] = dict(payload["vocabs"])
    if model is None:
        model = WubGPT.from_config(cfg)
    model.load_state_dict(payload["state_dict"])
    if vqs is None:
        vq_dir = p.parent / payload.get("vq_dir", p.stem + "_vq")
        vqs = {d.name: StemVQ.load(d) for d in sorted(vq_dir.iterdir()) if d.is_dir()}
    return model, vqs, cfg
