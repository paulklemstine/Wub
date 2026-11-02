"""Real section detection + full-song arc mining for Wub.

Replaces the faked bar-index section labels in
``SongIntrinsicCompressor.downconvert_song`` (INTRO for bars 0-8, GROOVE_A for
8-24, ...) with sections segmented from real acoustic evidence in
``data/processed/*.json`` (tatums + segments, stems_available, bpm, key), and
mines full-song section arcs (e.g. INTRO -> GROOVE_A -> BUILDUP -> DROP ->
BREAKDOWN -> ...) across the corpus.

Pipeline:
  1. :func:`bar_features` -- per-bar energy (mean onset ``loud_max``),
     density (onset count / 16 per stem + total), harmonic rhythm (pitched-
     stem mean chroma-vector change vs the previous bar via
     ``wub_core.global_vocabulary._raw_feature``), plus onset grids and
     mean velocity for the archetype rules.
  2. :func:`segment_sections` -- change-point split on the energy/density
     gradient (sliding-window mean-shift, min section length 4 bars, max 16
     bars), then one archetype label per section:

     - DROP: drums density >= 0.5 with onsets on 0, 4, 8, 12
     - BUILDUP: rising velocity/density slope > threshold
     - BREAKDOWN: drums density < 0.15
     - INTRO: first section + low density
     - else GROOVE_A
  3. :func:`mine_arcs` -- count full 6-10 section sequences across tracks +
     per-section duration (bars) distributions into a :class:`SectionArcModel`
     (top 50 arcs), serializable into ``global_patterns.json`` under the
     additive ``section_arcs`` key.

Single-stem tracks (most of the corpus only carries ``other``) have no drums
onsets at all; for those the ``other`` onset grid acts as the rhythmic proxy
so every track still yields meaningful sections.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

STEMS: Tuple[str, ...] = ("drums", "bass", "other", "vocals")

DOWNBEATS: Tuple[int, ...] = (0, 4, 8, 12)

MIN_SECTION_BARS = 4
MAX_SECTION_BARS = 16
MIN_ARC_SECTIONS = 6
MAX_ARC_SECTIONS = 10
MAX_ARCS = 50

# Archetype rule thresholds.
DROP_MIN_DRUMS_DENSITY = 0.5
DROP_MIN_DOWNBEAT_FRAC = 0.5
BREAKDOWN_MAX_DRUMS_DENSITY = 0.15
INTRO_MAX_TOTAL_DENSITY = 0.25
BUILDUP_MIN_DENSITY_SLOPE = 0.01  # density fraction gained per bar
BUILDUP_MIN_ENERGY_SLOPE = 0.5  # dB gained per bar

# Boundary-signal weights for [energy, total_density, harmonic_change].
SIGNAL_WEIGHTS = (1.0, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Duck-typed accessors (raw processed JSON dicts or schema objects).
# ---------------------------------------------------------------------------

def _tatum_stem_segments(tatum: Any) -> Dict[str, List[str]]:
    if isinstance(tatum, dict):
        return tatum.get("stem_segments", {}) or {}
    return getattr(tatum, "stem_segments", {}) or {}


def _seg_val(seg: Any, *names: str, default: Any = None) -> Any:
    if isinstance(seg, dict):
        for n in names:
            if n in seg and seg[n] is not None:
                return seg[n]
        return default
    for n in names:
        if hasattr(seg, n):
            v = getattr(seg, n)
            if v is not None:
                return v
    return default


def _velocity_from_loud(loud: float) -> float:
    """Same loudness -> velocity mapping used by the groove miner/compressor."""
    return float(min(1.0, max(0.40, (float(loud) + 40.0) / 40.0)))


def _raw_chroma(seg: Any) -> Optional[np.ndarray]:
    """12-D chroma vector of a segment, or None when unavailable."""
    try:
        from wub_core.global_vocabulary import _raw_feature
    except Exception:
        return None
    try:
        feat = _raw_feature(seg if isinstance(seg, dict) else {
            "timbre": list(_seg_val(seg, "timbre", default=[0.0] * 12)),
            "pitches": list(_seg_val(seg, "pitches", "chroma", default=[0.0] * 12)),
            "loudness_max": float(_seg_val(seg, "loudness_max", "loud_max", default=-20.0)),
            "duration": float(_seg_val(seg, "duration", default=0.25)),
        })
    except Exception:
        return None
    if feat is None:
        return None
    return np.asarray(feat[12:24], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Per-bar acoustic features.
# ---------------------------------------------------------------------------

def bar_features(
    track_segments: Sequence[Any],
    tatums: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Compute per-bar acoustic features from raw segments + tatums.

    Energy = mean ``loud_max`` over the bar's onset segments (``-60.0`` when
    the bar has no onsets). Density = onset count / 16 per stem plus the
    mean over stems (``total_density``). Harmonic rhythm = distance between
    the bar's mean pitched-stem onset chroma vector and the previous bar's
    (0.0 for the first bar / when chroma is unavailable; drums are excluded
    from the chroma mean so key changes are not washed out by percussive
    timbre).
    """
    seg_by_id: Dict[str, Any] = {}
    for seg in track_segments or []:
        sid = _seg_val(seg, "id", default=None)
        if sid is not None:
            seg_by_id[str(sid)] = seg

    tatums = list(tatums or [])
    n_bars = len(tatums) // 16
    feats: List[Dict[str, Any]] = []
    prev: Dict[str, Optional[str]] = {s: None for s in STEMS}
    prev_chroma: Optional[np.ndarray] = None

    for b in range(n_bars):
        onsets_by_stem: Dict[str, List[int]] = {s: [] for s in STEMS}
        onset_louds: List[float] = []
        chroma_acc: List[np.ndarray] = []

        for step in range(16):
            tat = tatums[b * 16 + step]
            stem_segs = _tatum_stem_segments(tat)
            for stem in STEMS:
                sids = stem_segs.get(stem, []) or []
                curr = sids[-1] if sids else None
                is_onset = (step == 0 and curr is not None) or (
                    curr is not None and curr != prev.get(stem)
                )
                if is_onset:
                    onsets_by_stem[stem].append(step)
                    seg = seg_by_id.get(str(curr))
                    if seg is not None:
                        onset_louds.append(float(
                            _seg_val(seg, "loudness_max", "loud_max", default=-20.0)))
                        if stem != "drums":
                            # Harmony lives in pitched stems; drum chroma is
                            # timbral noise that would wash out bar-to-bar
                            # harmonic rhythm, so it is excluded here.
                            ch = _raw_chroma(seg)
                            if ch is not None:
                                chroma_acc.append(ch)
                prev[stem] = curr

        densities = {s: len(onsets_by_stem[s]) / 16.0 for s in STEMS}
        total_density = float(sum(densities.values()) / len(STEMS))
        energy = float(np.mean(onset_louds)) if onset_louds else -60.0
        mean_vel = float(np.mean([_velocity_from_loud(l) for l in onset_louds])) \
            if onset_louds else 0.0

        if chroma_acc:
            mean_chroma = np.mean(np.stack(chroma_acc), axis=0)
        else:
            mean_chroma = None
        if mean_chroma is not None and prev_chroma is not None:
            denom = float(np.linalg.norm(mean_chroma) * np.linalg.norm(prev_chroma))
            if denom > 1e-9:
                cos_sim = float(np.dot(mean_chroma, prev_chroma) / denom)
                harmonic_change = float(np.clip(1.0 - cos_sim, 0.0, 2.0))
            else:
                harmonic_change = 0.0
        else:
            harmonic_change = 0.0
        if mean_chroma is not None:
            prev_chroma = mean_chroma

        drums_onsets = list(onsets_by_stem["drums"])
        kick_downbeats = all(s in drums_onsets for s in DOWNBEATS)

        feats.append({
            "bar": b,
            "energy": energy,
            "densities": densities,
            "total_density": total_density,
            "mean_vel": mean_vel,
            "harmonic_change": harmonic_change,
            "drums_onsets": drums_onsets,
            "other_onsets": list(onsets_by_stem["other"]),
            "kick_downbeats": kick_downbeats,
        })

    return feats


# ---------------------------------------------------------------------------
# 2. Change-point segmentation + archetype labeling.
# ---------------------------------------------------------------------------

@dataclass
class SectionSpan:
    """One labeled contiguous run of bars."""
    label: str
    start_bar: int
    n_bars: int
    mean_energy: float = 0.0
    mean_density: float = 0.0

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.n_bars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "section": self.label,
            "start_bar": int(self.start_bar),
            "n_bars": int(self.n_bars),
            "end_bar": int(self.end_bar),
            "mean_energy": round(float(self.mean_energy), 3),
            "mean_density": round(float(self.mean_density), 4),
        }


def _signal_matrix(feats: List[Dict[str, Any]]) -> np.ndarray:
    """Normalized weighted (energy, total_density, harmonic_change) signal.

    Weights are SIGNAL_WEIGHTS = (1.0, 0.5, 1.0): harmonic rhythm (pitched-
    stem mean-chroma change vs the previous bar) is a first-class section cue
    alongside energy, while density gets half weight because gradual
    buildup ramps should not self-split.
    """
    n = len(feats)
    energy = np.array([f["energy"] for f in feats], dtype=np.float64)
    dens = np.array([f["total_density"] for f in feats], dtype=np.float64)
    harm = np.array([f.get("harmonic_change", 0.0) for f in feats], dtype=np.float64)

    def _minmax(v: np.ndarray) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    harm_max = float(harm.max()) if n else 0.0
    harm_n = harm / harm_max if harm_max > 1e-9 else np.zeros_like(harm)
    w = np.asarray(SIGNAL_WEIGHTS, dtype=np.float64)
    return np.stack([_minmax(energy), _minmax(dens), harm_n], axis=1) * w


def _boundary_scores(sig: np.ndarray, window: int = 1) -> np.ndarray:
    """Sliding-window mean-shift score for a boundary before each bar.

    score[b] = distance between the mean of ``window`` bars before b and the
    mean of ``window`` bars starting at b. score[0] = 0 (no boundary at
    start). The default window of 1 scores adjacent-bar change; harmonic
    spikes (single-bar impulses at section starts) would otherwise smear and
    double-trigger with wider windows.
    """
    n = sig.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for b in range(1, n):
        lo0 = max(0, b - window)
        hi1 = min(n, b + window)
        if b - lo0 == 0 or hi1 - b == 0:
            continue
        left = sig[lo0:b].mean(axis=0)
        right = sig[b:hi1].mean(axis=0)
        scores[b] = float(np.linalg.norm(right - left))
    return scores


def _split_points(n_bars: int, scores: np.ndarray) -> List[int]:
    """Greedy boundaries honoring min 4 / max 16 bars per section."""
    if n_bars <= 0:
        return []
    nz = scores[1:]
    if nz.size:
        thresh = float(max(0.20, np.median(nz) + 0.5 * np.std(nz)))
    else:
        thresh = 0.20
    bounds = [0]
    s = 0
    while s < n_bars:
        nxt = None
        latest = min(s + MAX_SECTION_BARS, n_bars)
        for b in range(s + MIN_SECTION_BARS, latest):
            if scores[b] > thresh:
                nxt = b
                break
        if nxt is None:
            if latest < n_bars:
                nxt = latest  # forced cut at max section length
            else:
                break
        bounds.append(nxt)
        s = nxt
    if bounds[-1] != n_bars:
        # Trailing run shorter than MIN_SECTION_BARS merges into the previous
        # section instead of forming a stub.
        if len(bounds) >= 2 and n_bars - bounds[-1] < MIN_SECTION_BARS:
            bounds[-1] = n_bars
        else:
            bounds.append(n_bars)

    # Merge any interior run shorter than MIN_SECTION_BARS (e.g. the echo of
    # a single-bar harmonic spike tripping an adjacent trigger) into its
    # previous section; a short leading run merges into its next section.
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < MIN_SECTION_BARS and len(merged) > 1:
            merged[-1] = b  # absorb stub into the previous section
        elif b - merged[-1] < MIN_SECTION_BARS:
            continue  # leading stub: drop this boundary, merge into next
        else:
            merged.append(b)
    if merged[-1] != n_bars:
        merged.append(n_bars)
    return merged


def _slope(values: List[float]) -> float:
    """Least-squares slope per bar (0.0 for < 2 samples)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def _label_section(
    feats: List[Dict[str, Any]],
    is_first: bool,
    drums_proxy_other: bool,
) -> str:
    """Apply the archetype rules to one run of bar features."""
    n = len(feats)
    if n == 0:
        return "GROOVE_A"

    def _rhythm_onsets(f: Dict[str, Any]) -> List[int]:
        if drums_proxy_other:
            # Single-stem track: fall back to the mix ("other") onset grid.
            return list(f.get("other_onsets", []) or [])
        return f["drums_onsets"]

    drums_dens = []
    downbeat_frac = 0
    for f in feats:
        grid = set(_rhythm_onsets(f))
        drums_dens.append(len(grid) / 16.0)
        if all(s in grid for s in DOWNBEATS):
            downbeat_frac += 1
    downbeat_frac = downbeat_frac / n
    mean_drums = float(np.mean(drums_dens))
    mean_total = float(np.mean([f["total_density"] for f in feats]))
    dens_slope = _slope([f["total_density"] for f in feats])
    energy_slope = _slope([f["energy"] for f in feats])

    # INTRO must win over BREAKDOWN for a sparse opening section.
    if is_first and mean_total <= INTRO_MAX_TOTAL_DENSITY:
        return "INTRO"
    if mean_drums >= DROP_MIN_DRUMS_DENSITY and downbeat_frac >= DROP_MIN_DOWNBEAT_FRAC:
        return "DROP"
    if dens_slope > BUILDUP_MIN_DENSITY_SLOPE or energy_slope > BUILDUP_MIN_ENERGY_SLOPE:
        return "BUILDUP"
    if mean_drums < BREAKDOWN_MAX_DRUMS_DENSITY:
        return "BREAKDOWN"
    return "GROOVE_A"


def segment_sections(bar_feats: List[Dict[str, Any]]) -> List[SectionSpan]:
    """Split per-bar features into labeled sections (min 4 / max 16 bars)."""
    n = len(bar_feats)
    if n == 0:
        return []

    # Tracks with no drums onsets at all (e.g. single-stem "other" analyses)
    # use the "other" onset grid as the rhythmic proxy for labeling.
    drums_any = any(len(f.get("drums_onsets", [])) > 0 for f in bar_feats)
    proxy = not drums_any

    sig = _signal_matrix(bar_feats)
    scores = _boundary_scores(sig)
    bounds = _split_points(n, scores)

    spans: List[SectionSpan] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        run = bar_feats[s:e]
        label = _label_section(run, is_first=(i == 0), drums_proxy_other=proxy)
        spans.append(SectionSpan(
            label=label,
            start_bar=s,
            n_bars=e - s,
            mean_energy=float(np.mean([f["energy"] for f in run])),
            mean_density=float(np.mean([f["total_density"] for f in run])),
        ))
    return spans


# Backwards-tolerant alias: some callers think in (label, start, length).
def spans_to_labels(spans: Sequence[Union[SectionSpan, Dict[str, Any], Tuple]]) -> List[str]:
    out: List[str] = []
    for sp in spans:
        if isinstance(sp, SectionSpan):
            out.append(sp.label)
        elif isinstance(sp, dict):
            out.append(str(sp.get("label", sp.get("section", "GROOVE_A"))))
        elif isinstance(sp, (list, tuple)) and sp:
            out.append(str(sp[0]))
        else:
            out.append(str(sp))
    return out


# ---------------------------------------------------------------------------
# 3. Full-song arc mining.
# ---------------------------------------------------------------------------

@dataclass
class SectionArcModel:
    """Mined full-song section arcs + per-section duration distributions."""
    arcs: List[Dict[str, Any]] = field(default_factory=list)  # {"sections": [...], "count": n}
    durations: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {SEC: {mean, std, count}}
    total_tracks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arcs": [
                {"sections": list(a["sections"]), "count": int(a["count"])}
                for a in self.arcs
            ],
            "durations": {
                sec: {
                    "mean": round(float(v["mean"]), 3),
                    "std": round(float(v["std"]), 3),
                    "count": int(v["count"]),
                }
                for sec, v in self.durations.items()
            },
            "total_tracks": int(self.total_tracks),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SectionArcModel":
        return cls(
            arcs=[{"sections": list(a["sections"]), "count": int(a["count"])}
                  for a in d.get("arcs", [])],
            durations={sec: {"mean": float(v["mean"]), "std": float(v["std"]),
                              "count": int(v.get("count", 0))}
                        for sec, v in d.get("durations", {}).items()},
            total_tracks=int(d.get("total_tracks", 0)),
        )


def _normalize_seq(
    seq: Sequence[Union[SectionSpan, Dict[str, Any], Tuple, str]],
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Return (labels, [(label, n_bars), ...]) for flexible span inputs."""
    labels: List[str] = []
    spans: List[Tuple[str, int]] = []
    for sp in seq:
        if isinstance(sp, SectionSpan):
            labels.append(sp.label)
            spans.append((sp.label, sp.n_bars))
        elif isinstance(sp, dict):
            lab = str(sp.get("label", sp.get("section", "GROOVE_A")))
            labels.append(lab)
            spans.append((lab, int(sp.get("n_bars", sp.get("bars", 0)))))
        elif isinstance(sp, (list, tuple)):
            lab = str(sp[0])
            labels.append(lab)
            spans.append((lab, int(sp[1]) if len(sp) > 1 else 0))
        else:
            labels.append(str(sp))
            spans.append((str(sp), 0))
    return labels, spans


def mine_arcs(
    all_section_seqs: Sequence[Sequence[Union[SectionSpan, Dict[str, Any], Tuple, str]]],
    min_sections: int = MIN_ARC_SECTIONS,
    max_sections: int = MAX_ARC_SECTIONS,
    top_k: int = MAX_ARCS,
) -> SectionArcModel:
    """Count full 6-10 section sequences + per-section bar-duration stats.

    Only sequences with ``min_sections <= len <= max_sections`` contribute to
    the arc counts (full-song arcs); every span with ``n_bars > 0``
    contributes to the duration distributions.
    """
    arc_counter: Counter = Counter()
    dur_lists: Dict[str, List[float]] = defaultdict(list)

    n_tracks = 0
    for seq in all_section_seqs or []:
        seq = list(seq)
        if not seq:
            continue
        n_tracks += 1
        labels, spans = _normalize_seq(seq)
        for lab, nb in spans:
            if nb and nb > 0:
                dur_lists[lab].append(float(nb))
        if min_sections <= len(labels) <= max_sections:
            arc_counter[tuple(labels)] += 1

    ranked = arc_counter.most_common(top_k)
    arcs = [{"sections": list(sections), "count": int(c)} for sections, c in ranked]
    durations: Dict[str, Dict[str, float]] = {}
    for sec, vals in dur_lists.items():
        arr = np.asarray(vals, dtype=np.float64)
        durations[sec] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "count": int(len(vals)),
        }

    return SectionArcModel(arcs=arcs, durations=durations, total_tracks=n_tracks)


# ---------------------------------------------------------------------------
# Standalone corpus entry point (also wired into train_global_patterns.py).
# ---------------------------------------------------------------------------

def mine_section_arcs_from_processed(
    processed_dir: Union[str, Path] = "data/processed",
    max_tracks: Optional[int] = None,
    max_bars_per_track: int = 64,
    min_sections: int = MIN_ARC_SECTIONS,
    max_sections: int = MAX_ARC_SECTIONS,
    top_k: int = MAX_ARCS,
) -> SectionArcModel:
    """Read raw analyses and mine the corpus section-arc model."""
    files = sorted(
        p for p in Path(processed_dir).glob("*.json")
        if not p.name.startswith(("_", "meta_"))
    )
    if max_tracks:
        files = files[:max_tracks]

    seqs: List[List[SectionSpan]] = []
    for jf in files:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        segs = raw.get("segments") or []
        tatums = raw.get("tatums") or []
        if not tatums:
            continue
        tatums = tatums[:max_bars_per_track * 16]
        try:
            feats = bar_features(segs, tatums)
            spans = segment_sections(feats)
        except Exception:
            continue
        if spans:
            seqs.append(spans)

    return mine_arcs(seqs, min_sections=min_sections,
                     max_sections=max_sections, top_k=top_k)


def section_arcs_payload(
    processed_dir: Union[str, Path] = "data/processed",
    max_tracks: Optional[int] = None,
    max_bars_per_track: int = 64,
) -> Dict[str, Any]:
    """JSON-serializable payload for the additive ``section_arcs`` key."""
    return mine_section_arcs_from_processed(
        processed_dir, max_tracks=max_tracks,
        max_bars_per_track=max_bars_per_track).to_dict()
