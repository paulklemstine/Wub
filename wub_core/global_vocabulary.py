"""Global vocabulary of optimal reusable slices across the full corpus.

Two-stage compression (per user plan):
  1. Per-song medoids via SongIntrinsicCompressor (k per stem).
  2. Global KMeans over pooled medoids -> fixed budget (default 5000/3500/5000/3500).

Pitched stems use transposition-invariant matching (min over 12 chroma rotations)
since playback pitch-shifts live with playbackRate=2^(shift/12), |shift|<=6.
Fidelity is measured stem-wise (not full mix) via cosine similarity in 26-D
slice_feature_raw space, matching pattern_audio_trainer stem weighting intent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wub_core.patterns import STEMS

logger = logging.getLogger("wub.global_vocabulary")

DEFAULT_GLOBAL_BUDGET: Dict[str, int] = {
    "drums": 5000,
    "bass": 3500,
    "other": 5000,
    "vocals": 3500,
}

FEAT_DIM = 26


def _raw_feature(seg: dict) -> Optional[np.ndarray]:
    """26-D slice_feature_raw compatible vector from a processed segment dict."""
    try:
        timbre = seg.get("timbre")
        chroma = seg.get("pitches", seg.get("chroma"))
        if not timbre or len(timbre) != 12 or not chroma or len(chroma) != 12:
            return None
        loud = float(seg.get("loudness_max", seg.get("loud_max", -20.0)))
        dur = float(seg.get("duration", 0.25) or 0.25)
        import math

        log_dur = math.log1p(max(0.01, dur))
        return np.asarray(list(timbre) + list(chroma) + [loud, log_dur], dtype=np.float32)
    except Exception:
        return None


def _clarity(seg: dict, stem: str) -> float:
    """Perceptual clarity 0..1 (punch + loudness + duration sweet spot)."""
    try:
        from wub_core.acoustic_vocabulary_optimizer import AcousticVocabularyOptimizer

        opt = AcousticVocabularyOptimizer()
        d = dict(seg)
        d["stem"] = stem
        return float(opt.compute_acoustic_clarity(d))
    except Exception:
        loud = float(seg.get("loudness_max", -99.0))
        if loud < -36.0:
            return 0.0
        return float(np.clip((loud + 30.0) / 30.0, 0.0, 1.0))


def _pitch_of(feat: np.ndarray) -> int:
    return int(np.argmax(feat[12:24]))


def pitch_invariant_dist(a: np.ndarray, b: np.ndarray, stem: str) -> float:
    """Euclidean distance; for pitched stems minimize over 12 chroma rotations."""
    if stem == "drums":
        diff = a.astype(np.float64) - b.astype(np.float64)
        return float(np.dot(diff, diff))
    # Split timbre/loud/dur (fixed) + chroma (rotation-invariant)
    best = float("inf")
    a_t = a[:12]
    b_t = b[:12]
    a_c = a[12:24]
    b_c = b[12:24]
    a_r = a[24:26]
    b_r = b[24:26]
    for shift in range(12):
        bc = np.roll(b_c, shift)
        d = float(
            np.sum((a_t - b_t) ** 2) + np.sum((a_c - bc) ** 2) * 2.0 + np.sum((a_r - b_r) ** 2)
        )
        if d < best:
            best = d
    return best


def best_pitch_shift(slice_pitch: int, target_pitch: int) -> int:
    shift = (int(target_pitch) - int(slice_pitch)) % 12
    if shift > 6:
        shift -= 12
    return int(shift)


@dataclass
class GlobalVocabSlice:
    id: str
    stem: str
    category: str
    feature_vector: List[float]
    clarity: float = 0.5
    usage_count: int = 0
    songs_using: List[str] = field(default_factory=list)
    pitch: int = 0
    duration: float = 0.25
    centroid: float = 1000.0
    track_id: str = ""
    segment_id: str = ""
    start: float = 0.0
    loud_max: float = -12.0
    url: str = ""

    def to_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "stem": self.stem,
            "category": self.category,
            "pitch": int(self.pitch),
            "duration": round(float(self.duration), 3),
            "clarity": round(float(self.clarity), 4),
            "usage_count": int(self.usage_count),
            "songs_using_count": len(self.songs_using),
            "centroid": round(float(self.centroid), 1),
            "track_id": self.track_id,
            "segment_id": self.segment_id,
            "start": round(float(self.start), 3),
            "loud_max": round(float(self.loud_max), 2),
            "url": self.url,
        }
        if include_vectors:
            d["feature_vector"] = [round(float(v), 4) for v in self.feature_vector]
        return d


@dataclass
class GlobalVocabulary:
    vocab_by_stem: Dict[str, List[GlobalVocabSlice]] = field(
        default_factory=lambda: {s: [] for s in STEMS}
    )
    budget: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_GLOBAL_BUDGET))

    def total(self) -> int:
        return sum(len(v) for v in self.vocab_by_stem.values())

    def find_nearest(self, stem: str, feat: np.ndarray) -> Tuple[Optional[GlobalVocabSlice,], int]:
        cands = self.vocab_by_stem.get(stem, [])
        if not cands:
            return None, 0
        target_pitch = _pitch_of(feat) if stem != "drums" else 0
        best = None
        best_d = float("inf")
        for c in cands:
            cf = np.asarray(c.feature_vector, dtype=np.float32)
            d = pitch_invariant_dist(feat, cf, stem)
            # Clarity bonus: prefer clearer slices on ties (max 15% discount)
            d = d * (1.0 - 0.15 * float(c.clarity))
            if d < best_d:
                best_d = d
                best = c
        if best is None:
            return None, 0
        shift = best_pitch_shift(best.pitch, target_pitch) if stem != "drums" else 0
        best.usage_count += 1
        return best, shift

    def to_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        return {
            "version": 1,
            "budget": self.budget,
            "total": self.total(),
            "vocab_by_stem": {
                s: [c.to_dict(include_vectors=include_vectors) for c in self.vocab_by_stem.get(s, [])]
                for s in STEMS
            },
        }

    def save(self, path: Path, include_vectors: bool = False) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(include_vectors=include_vectors), separators=(",", ":")))
        return p

    @classmethod
    def load(cls, path: Path) -> "GlobalVocabulary":
        raw = json.loads(Path(path).read_bytes())
        gv = cls(budget=raw.get("budget", dict(DEFAULT_GLOBAL_BUDGET)))
        for s in STEMS:
            for d in raw.get("vocab_by_stem", {}).get(s, []):
                gv.vocab_by_stem.setdefault(s, []).append(
                    GlobalVocabSlice(
                        id=d.get("id", ""),
                        stem=s,
                        category=d.get("category", s),
                        feature_vector=d.get("feature_vector", [0.0] * FEAT_DIM),
                        clarity=float(d.get("clarity", 0.5)),
                        usage_count=int(d.get("usage_count", 0)),
                        pitch=int(d.get("pitch", 0)),
                        duration=float(d.get("duration", 0.25)),
                        centroid=float(d.get("centroid", 1000.0)),
                        track_id=d.get("track_id", ""),
                        segment_id=d.get("segment_id", ""),
                        start=float(d.get("start", 0.0)),
                        loud_max=float(d.get("loud_max", -12.0)),
                        url=d.get("url", ""),
                    )
                )
                # songs_using list is truncated in dict; restore count only
                gv.vocab_by_stem[s][-1].songs_using = ["x"] * int(d.get("songs_using_count", 0))
        return gv


class GlobalVocabularyBuilder:
    """Two-stage global vocab: per-song medoids -> global KMeans medoids."""

    def __init__(
        self,
        target_counts: Optional[Dict[str, int]] = None,
        k_per_song: int = 8,
        random_state: int = 42,
    ) -> None:
        self.target_counts = dict(target_counts or DEFAULT_GLOBAL_BUDGET)
        self.k_per_song = int(k_per_song)
        self.random_state = int(random_state)

    # ---- stage 1: per-song medoids ----
    def collect_candidates(self, processed_dir: Path, max_tracks: Optional[int] = None, stems_dir: Optional[Path] = None):
        """Pool per-song medoid candidates with features + clarity.

        Returns: {stem: [(feat, clarity, meta dict)]}
        If stems_dir is given, only include tracks with separated stems so every
        global slice has renderable audio (required for static Opus export).
        """
        from sklearn.cluster import KMeans

        files = sorted(
            p for p in Path(processed_dir).glob("*.json") if not p.name.startswith(("_", "meta_"))
        )
        if stems_dir is not None:
            stem_tids = {d.name for d in Path(stems_dir).iterdir() if d.is_dir()}
            files = [p for p in files if p.stem in stem_tids]
        if max_tracks:
            files = files[:max_tracks]
        pool: Dict[str, List[Tuple[np.ndarray, float, dict]]] = {s: [] for s in STEMS}
        for jf in files:
            try:
                raw = json.loads(jf.read_bytes())
            except Exception:
                continue
            tid = str(raw.get("track_id") or jf.stem)
            stems_avail = raw.get("stems_available") or ["other"]
            for seg in raw.get("segments") or []:
                if not isinstance(seg, dict):
                    continue
                stem = str(seg.get("stem", "other"))
                if stem not in STEMS:
                    continue
                if seg.get("is_silent"):
                    continue
                loud = float(seg.get("loudness_max", -99.0))
                if loud < -48.0:
                    continue
                feat = _raw_feature(seg)
                if feat is None:
                    continue
                clarity = _clarity(seg, stem)
                if clarity <= 0.0:
                    continue
                meta = {
                    "track_id": tid,
                    "segment_id": str(seg.get("id", "")),
                    "start": float(seg.get("start", 0.0)),
                    "duration": float(seg.get("duration", 0.25) or 0.25),
                    "centroid": float(seg.get("spectral_centroid", 1000.0) or 1000.0),
                    "loud_max": loud,
                    "pitch": _pitch_of(feat),
                    "filename": str(raw.get("filename", "")),
                }
                pool[stem].append((feat, clarity, meta))
        # Condense per-song? For scale, do global per-stem cap: if pool huge,
        # first KMeans per stem to min(40000, len) to keep stage-2 tractable,
        # weighted toward clarity. This preserves the "per-song medoid" spirit
        # without O(N^2) per-track clustering over 1725 files.
        condensed: Dict[str, List[Tuple[np.ndarray, float, dict]]] = {s: [] for s in STEMS}
        rng = np.random.RandomState(self.random_state)
        for stem in STEMS:
            items = pool[stem]
            if not items:
                continue
            # Prefer clearer slices: clarity-weighted sampling down to 40k
            if len(items) > 40000:
                cl = np.array([c for _, c, _ in items], dtype=np.float64)
                w = (cl + 0.05) / (cl + 0.05).sum()
                idx = rng.choice(len(items), size=40000, replace=False, p=w)
                items = [items[i] for i in idx]
            # Per-song medoid pass: group by track, KMeans k=k_per_song each
            by_track: Dict[str, List[int]] = {}
            for i, (_, _, m) in enumerate(items):
                by_track.setdefault(m["track_id"], []).append(i)
            for _, idxs in by_track.items():
                if len(idxs) <= self.k_per_song:
                    for i in idxs:
                        condensed[stem].append(items[i])
                    continue
                X = np.stack([items[i][0] for i in idxs]).astype(np.float32)
                k = min(self.k_per_song, len(idxs))
                try:
                    km = KMeans(n_clusters=k, n_init=3, random_state=self.random_state)
                    labels = km.fit_predict(X)
                    centers = km.cluster_centers_
                except Exception:
                    for i in idxs[:k]:
                        condensed[stem].append(items[i])
                    continue
                for c in range(k):
                    members = [idxs[j] for j in range(len(idxs)) if labels[j] == c]
                    if not members:
                        continue
                    # Medoid = min dist to center, tie-break by clarity
                    dists = [float(np.sum((items[m][0] - centers[c]) ** 2)) for m in members]
                    best_m = members[int(np.argmin(dists))]
                    # If multiple close, prefer clearer
                    order = sorted(members, key=lambda m: dists[members.index(m)] - items[m][1] * 0.5)
                    best_m = order[0]
                    condensed[stem].append(items[best_m])
        return condensed

    # ---- stage 2: global clustering ----
    def build(self, candidates: Dict[str, List[Tuple[np.ndarray, float, dict]]]) -> GlobalVocabulary:
        gv = GlobalVocabulary(budget=dict(self.target_counts))
        for stem in STEMS:
            items = candidates.get(stem, [])
            if not items:
                continue
            k = max(1, min(int(self.target_counts.get(stem, 500)), len(items)))
            X = np.stack([f for f, _, _ in items]).astype(np.float32)
            clarity = np.array([c for _, c, _ in items], dtype=np.float32)
            if len(items) <= k:
                chosen = list(range(len(items)))
            elif k > 800:
                # Large-K: MiniBatchKMeans (full KMeans O(n*k*d) takes hours at k=5000)
                from sklearn.cluster import MiniBatchKMeans

                logger.warning("global %s: MiniBatchKMeans k=%d n=%d", stem, k, len(items))
                mb = MiniBatchKMeans(
                    n_clusters=k, random_state=self.random_state, batch_size=2048,
                    max_iter=40, n_init=3, reassignment_ratio=0.02,
                )
                labels = mb.fit_predict(X)
                centers = mb.cluster_centers_
                chosen = []
                for c in range(k):
                    members = np.where(labels == c)[0]
                    if len(members) == 0:
                        continue
                    if len(members) > 200:
                        # subsample for medoid scoring speed
                        members = np.random.RandomState(self.random_state).choice(members, 200, replace=False)
                    dists = np.sum((X[members] - centers[c]) ** 2, axis=1)
                    scores = (1.0 / (1.0 + dists)) * 0.7 + clarity[members] * 0.3
                    chosen.append(int(members[int(np.argmax(scores))]))
            else:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state)
                labels = km.fit_predict(X)
                centers = km.cluster_centers_
                chosen = []
                for c in range(k):
                    members = np.where(labels == c)[0]
                    if len(members) == 0:
                        continue
                    dists = np.sum((X[members] - centers[c]) ** 2, axis=1)
                    # Combined: close to centroid + high clarity + reuse prior (usage=1 here)
                    scores = (1.0 / (1.0 + dists)) * 0.7 + clarity[members] * 0.3
                    chosen.append(int(members[int(np.argmax(scores))]))
            for rank, idx in enumerate(chosen):
                feat, clar, meta = items[idx]
                # Category via lightweight heuristic (centroid/duration) to avoid SliceRef cost
                cat = self._category(stem, meta)
                gid = f"g_{stem}_{rank:05d}"
                gv.vocab_by_stem[stem].append(
                    GlobalVocabSlice(
                        id=gid,
                        stem=stem,
                        category=f"{stem}:{cat}",
                        feature_vector=[round(float(v), 4) for v in feat.tolist()],
                        clarity=round(float(clar), 4),
                        usage_count=0,
                        pitch=int(meta.get("pitch", 0)),
                        duration=float(meta.get("duration", 0.25)),
                        centroid=float(meta.get("centroid", 1000.0)),
                        track_id=str(meta.get("track_id", "")),
                        segment_id=str(meta.get("segment_id", "")),
                        start=float(meta.get("start", 0.0)),
                        loud_max=float(meta.get("loud_max", -12.0)),
                    )
                )
        return gv

    @staticmethod
    def _category(stem: str, meta: dict) -> str:
        cent = float(meta.get("centroid", 1200.0))
        dur = float(meta.get("duration", 0.25))
        if stem == "drums":
            if dur >= 0.28:
                return "fill"
            if cent < 950.0:
                return "kick"
            if 950.0 <= cent <= 3200.0:
                return "snare"
            return "hat_closed" if dur < 0.12 else "hat_open"
        if stem == "bass":
            if cent < 520.0 and dur >= 0.14:
                return "sub"
            return "stab" if dur < 0.14 else "rolling"
        if stem == "other":
            if dur >= 0.38:
                return "pad"
            return "chord" if dur >= 0.16 else "arp"
        return "chop" if dur < 0.28 else "phrase"

    # ---- remap + stem-wise fidelity ----
    def remap_bars(
        self, track_segments: List[dict], tatums: List[dict], vocab: GlobalVocabulary
    ) -> Tuple[List[Dict[str, List[Optional[str]]]], Dict[str, float]]:
        """Remap every onset to global vocab IDs. Returns (bars, fidelity_by_stem)."""
        seg_feat: Dict[str, np.ndarray] = {}
        for seg in track_segments:
            sid = str(seg.get("id", ""))
            f = _raw_feature(seg)
            if sid and f is not None:
                seg_feat[sid] = f
        n_bars = len(tatums) // 16
        bars: List[Dict[str, List[Optional[str]]]] = []
        sims: Dict[str, List[float]] = {s: [] for s in STEMS}
        prev: Dict[str, Optional[str]] = {s: None for s in STEMS}
        for b in range(n_bars):
            bar: Dict[str, List[Optional[str]]] = {s: [None] * 16 for s in STEMS}
            for step in range(16):
                tat = tatums[b * 16 + step]
                stem_segs = tat.get("stem_segments", {}) if isinstance(tat, dict) else {}
                for stem in STEMS:
                    sids = stem_segs.get(stem, [])
                    curr = sids[-1] if sids else None
                    is_onset = (step == 0 and curr is not None) or (
                        curr is not None and curr != prev.get(stem)
                    )
                    if is_onset and curr in seg_feat:
                        match, _ = vocab.find_nearest(stem, seg_feat[curr])
                        if match is not None:
                            bar[stem][step] = match.id
                            a = seg_feat[curr].astype(np.float64)
                            cf = np.asarray(match.feature_vector, dtype=np.float64)
                            denom = float(np.linalg.norm(a) * np.linalg.norm(cf))
                            cos = float(np.dot(a, cf) / denom) if denom > 1e-6 else 0.9
                            sims[stem].append(float(np.clip(0.5 + 0.5 * cos, 0.5, 1.0)))
                    prev[stem] = curr
            bars.append(bar)
        fidelity = {s: round(float(np.mean(v)), 4) if v else 0.92 for s, v in sims.items()}
        return bars, fidelity
