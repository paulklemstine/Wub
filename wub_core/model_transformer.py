"""Phase 2B: multi-stream VQ-Transformer ("WubGPT").

Tokens per tatum step: s_t = (r_t, c_drums, c_bass, c_other, c_vocals),
where r_t in 0..15 is a metrical position token and each c_stem is a
per-stem vector-quantized slice token (K clusters + 1 silence token).

Pipeline: markov.build_pools -> StemVQ.fit per stem -> build_dataset ->
WubGPT + train_model -> sample -> tokens_to_slices (nearest SliceRef).
"""
from __future__ import annotations

import json
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
    "WubGPT",
    "train_model",
    "sample",
    "tokens_to_slices",
    "save_checkpoint",
    "load_checkpoint",
    "slice_feature_raw",
    "STREAM_ORDER",
    "STEM_ORDER",
    "FEAT_DIM",
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


def build_dataset(analyses, vqs: Dict[str, StemVQ], stride: int = 1) -> List[Dict[str, int]]:
    """One token row per tatum: {r, drums, bass, other, vocals}.

    r = beat_position * 4 + tatum_position (0..15). Per stem, the first
    segment id in tatum.stem_segments[stem] is VQ-encoded; missing segment
    ids, silent segments, or stems absent from the track yield silence.
    stride>1 keeps every Nth tatum (full-corpus training on small boxes).
    """
    rows: List[Dict[str, int]] = []
    for tr in analyses:
        seg_by_id = {s.id: s for s in tr.segments}
        available = set(tr.stems_available or [])
        for tat in tr.tatums[:: max(1, int(stride))]:
            beat_pos = getattr(tat, "beat_position", tat.get("beat_position", 0) if isinstance(tat, dict) else 0)
            tatum_pos = getattr(tat, "tatum_position", tat.get("tatum_position", 0) if isinstance(tat, dict) else 0)
            r = (int(beat_pos) * 4 + int(tatum_pos)) % _R_VOCAB
            row: Dict[str, int] = {"r": int(r)}
            segmap = tat.stem_segments or {}
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
            rows.append(row)
    return rows


class WubGPT(nn.Module):
    """5-stream causal Transformer: per-step tokens (r + 4 stems).

    Each step contributes 5 tokens (own embedding table each) plus learned
    stream-type and positional embeddings; the flattened (seq*5) sequence
    goes through a causal TransformerEncoder, and 5 heads predict the NEXT
    step's tokens from the step's final-stream hidden state.
    """

    STREAMS = tuple(STREAM_ORDER)

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

        self.emb = nn.ModuleDict(
            {s: nn.Embedding(vocabs[s], d_model) for s in self.STREAMS}
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
        )

    @classmethod
    def from_vqs(cls, vqs: Dict[str, StemVQ], **kw) -> "WubGPT":
        vocabs: Dict[str, int] = {"r": _R_VOCAB}
        for s in STEM_ORDER:
            vocabs[s] = int(vqs[s].vocab_size) if s in vqs else 1
        return cls(vocabs=vocabs, **kw)

    def forward(self, r, drums=None, bass=None, other=None, vocals=None):
        if isinstance(r, dict):
            drums = r["drums"]
            bass = r["bass"]
            other = r["other"]
            vocals = r["vocals"]
            r = r["r"]
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


def train_model(
    model: WubGPT,
    dataset: List[Dict[str, int]],
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    device="cpu",
    seed: int = 0,
) -> List[float]:
    """Next-step training: CE per head summed. Returns per-epoch mean losses."""
    if len(dataset) < 2:
        raise ValueError("need at least 2 token rows to train")
    torch.manual_seed(seed)
    dev = torch.device(device)
    model.to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    keys = list(STREAM_ORDER)
    arr = {
        kk: torch.tensor([row[kk] for row in dataset], dtype=torch.long)
        for kk in keys
    }
    N = len(dataset)
    T = min(32, N - 1)  # context window; T+1-row sliding windows, stride 1
    nwin = N - T
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    for _ep in range(int(epochs)):
        perm = rng.permutation(nwin)
        tot, nb = 0.0, 0
        for bi in range(0, nwin, max(1, int(batch_size))):
            idx = perm[bi : bi + int(batch_size)]
            inp = {
                kk: torch.stack([arr[kk][int(s) : int(s) + T] for s in idx]).to(dev)
                for kk in keys
            }
            tgt = {
                kk: torch.stack([arr[kk][int(s) + 1 : int(s) + T + 1] for s in idx]).to(dev)
                for kk in keys
            }
            opt.zero_grad()
            logits = model(inp)
            loss = sum(
                F.cross_entropy(
                    logits[kk].reshape(-1, logits[kk].size(-1)), tgt[kk].reshape(-1)
                )
                for kk in keys
            )
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        losses.append(tot / max(nb, 1))
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
) -> List[Dict[str, int]]:
    """Autoregressively generate `steps` new token rows after `seed_steps`."""
    if not seed_steps:
        raise ValueError("seed_steps must be non-empty")
    if steps < 1:
        return []
    _ = vqs  # vocab sizes live on the model; vqs kept for API symmetry
    was_training = model.training
    model.eval()
    keys = list(STREAM_ORDER)
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    gen = torch.Generator().manual_seed(int(seed))
    hist = [dict(s) for s in seed_steps]
    out: List[Dict[str, int]] = []
    with torch.no_grad():
        for _ in range(int(steps)):
            ctx = hist[-model.max_steps :] if len(hist) > model.max_steps else hist
            inp = {
                kk: torch.tensor(
                    [[int(row[kk]) for row in ctx]], dtype=torch.long, device=dev
                )
                for kk in keys
            }
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
                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    cut = cum - sp > float(top_p)
                    sp = sp.masked_fill(cut, 0.0)
                    sp = sp / (sp.sum() + 1e-12)
                    nxt[kk] = int(si[int(torch.multinomial(sp, 1, generator=gen))].item())
                else:
                    nxt[kk] = int(torch.multinomial(probs, 1, generator=gen).item())
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
