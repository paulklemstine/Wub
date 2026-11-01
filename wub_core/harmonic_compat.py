"""Compact learned harmonic-compatibility model for the generative DJ mix.

Learns, from real song slices (``TrackAnalysis`` via ``wub_core.corpus``),
which *simultaneous* stem combinations sound harmonious -- not just
key-matched. For every harmonic context ``(section, bar chord root,
minor/major)`` we aggregate:

* pitch-class histograms (0..11, C=0) of bass / other / vocals segment
  chroma (``Segment.pitches``), quantized by argmax;
* co-occurrence counts of ``(bass_groove x other_groove)`` pairs per bar,
  using the groove classifiers from ``wub_core.patterns``.

``compat_score`` then rates a candidate (bass, other, vocal) combination in
a target harmonic context: high when the pitch classes were frequently
observed together in that context in real songs, with an explicit penalty
for semitone clashes against the active chord tones (and between the
simultaneous stems themselves).

Everything is plain dicts / lists of floats: ``to_compact`` prunes to the
top entries so the JSON stays far below 1MB; ``load`` restores it.
Pure Python + numpy, no torch.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VERSION = 1
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")
PITCH_STEMS = ("bass", "other", "vocals")

# ---------------------------------------------------------------- key utils

_PC_BY_NAME = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11,
}
_KEY_RE = re.compile(r"^\s*([A-G])\s*(#|B)?", re.IGNORECASE)


def parse_key(key: Any) -> Tuple[int, bool]:
    """Parse a key string like 'C major' / 'F# minor' -> (root 0..11, is_minor)."""
    if not key:
        return 0, False
    s = str(key).strip().upper().replace("♯", "#").replace("♭", "B")
    m = _KEY_RE.match(s)
    if not m:
        return 0, False
    root_tok = (m.group(1) + (m.group(2) or "")).upper()
    root = _PC_BY_NAME.get(root_tok, 0)
    is_minor = "MIN" in s
    return root, is_minor


def chord_tones(root: int, is_minor: bool) -> Tuple[int, int, int]:
    """Triad pitch classes for the active chord."""
    third = (root + 3) % 12 if is_minor else (root + 4) % 12
    return (root % 12, third, (root + 7) % 12)


def circ_dist(a: int, b: int) -> int:
    """Circular semitone distance on the pitch-class wheel."""
    d = abs(int(a) - int(b)) % 12
    return min(d, 12 - d)


def context_key(section: str, root: int, is_minor: bool) -> str:
    sec = str(section or "GROOVE_A").upper()
    if sec not in SECTIONS:
        sec = "GROOVE_A"
    return f"{sec}|{int(root) % 12}|{'min' if is_minor else 'maj'}"


def quantize_chroma(chroma: Sequence[float]) -> int:
    """Quantize a 12-bin chroma vector to one pitch class 0..11."""
    arr = np.asarray(list(chroma), dtype=float)
    if arr.shape != (12,):
        raise ValueError(f"chroma must have 12 bins, got {arr.shape}")
    return int(np.argmax(arr)) % 12


# ------------------------------------------------------- input normalization

def _as_weighted_pcs(spec: Any) -> Dict[int, float]:
    """Normalize a pitch spec to {pitch_class: weight}.

    Accepts a single int pc, a 12-length chroma vector, a dict {pc: w},
    or any iterable of pitch classes.
    """
    if spec is None:
        return {}
    if isinstance(spec, (bool, np.bool_)):
        return {}
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return {int(spec) % 12: 1.0}
    if isinstance(spec, Mapping):
        out: Dict[int, float] = {}
        for k, v in spec.items():
            try:
                out[int(k) % 12] = out.get(int(k) % 12, 0.0) + max(0.0, float(v))
            except (TypeError, ValueError):
                continue
        return out
    vals = list(spec)  # type: ignore[arg-type]
    if len(vals) == 12:
        # 12-length sequences are interpreted as chroma vectors.
        out = {}
        for i, v in enumerate(vals):
            try:
                w = max(0.0, float(v))
            except (TypeError, ValueError):
                continue
            if w > 0:
                out[i] = w
        if out:
            return out
    out = {}
    for v in vals:
        try:
            pc = int(v) % 12
        except (TypeError, ValueError):
            continue
        out[pc] = out.get(pc, 0.0) + 1.0
    return out


# ------------------------------------------------------------- bar analysis

def _bar_windows(tatums: Sequence[Any], bar_len: int = 16) -> List[List[int]]:
    downs = [i for i, t in enumerate(tatums)
             if getattr(t, "beat_position", 0) == 0 and getattr(t, "tatum_position", 0) == 0]
    n = len(tatums)
    if len(downs) >= 1:
        wins = []
        for d in downs:
            if d + 1 >= n:
                continue
            wins.append(list(range(d, min(d + bar_len, n))))
        wins = [w for w in wins if len(w) >= 4]
        if wins:
            return wins
    return [list(range(i, min(i + bar_len, n))) for i in range(0, n, bar_len) if n - i >= 4]


def _bar_onsets(win: List[int], tatums: Sequence[Any], stem: str,
                prev_seg: Optional[str]) -> Tuple[List[int], Optional[str]]:
    """Onset step indices within a bar from segment-id changes (patterns.py style)."""
    onsets: List[int] = []
    cur = prev_seg
    for step, ti in enumerate(win):
        ids = (tatums[ti].stem_segments or {}).get(stem, []) if ti < len(tatums) else []
        seg = ids[-1] if ids else None
        if seg is not None and (step == 0 or seg != cur):
            onsets.append(step)
        cur = seg
    return onsets, cur


def _loud_energy(loud_max: float) -> float:
    return float(min(1.0, max(0.0, (loud_max + 60.0) / 60.0)))


def _classify_bar_section(energy: float, drum_d: float, bass_d: float) -> str:
    if energy >= 0.6 and drum_d >= 0.2:
        return "DROP"
    if energy >= 0.3 or drum_d >= 0.12 or bass_d >= 0.12:
        return "GROOVE_A"
    if energy < 0.2 and drum_d < 0.08 and bass_d < 0.08:
        return "BREAKDOWN"
    return "INTRO"


def infer_bar_sections(bar_feats: List[Dict[str, float]]) -> List[str]:
    """Per-bar section labels from energy/density (+BUILDUP ramp override).

    ``bar_feats`` entries need ``energy``, ``drum_d``, ``bass_d``.
    """
    labels = [_classify_bar_section(f.get("energy", 0.5), f.get("drum_d", 0.0),
                                    f.get("bass_d", 0.0)) for f in bar_feats]
    n = len(labels)
    # Smooth isolated single-bar blips (differing from both neighbours).
    for i in range(1, n - 1):
        if labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
            labels[i] = labels[i - 1]
    # Rising ramps into a DROP are BUILDUPs.
    for i in range(n):
        if labels[i] != "DROP":
            continue
        j = i - 1
        ramp: List[int] = []
        while j >= 0 and len(ramp) < 4 and labels[j] in ("GROOVE_A", "INTRO"):
            ramp.append(j)
            j -= 1
        ramp.reverse()
        if len(ramp) >= 2:
            es = [bar_feats[k].get("energy", 0.0) for k in ramp]
            if es[-1] - es[0] >= 0.15 and all(b >= a - 0.02 for a, b in zip(es, es[1:])):
                for k in ramp:
                    labels[k] = "BUILDUP"
    return labels


def _groove_classifiers():  # patterns.py when available, else local mirrors
    try:
        from wub_core.patterns import (  # type: ignore
            classify_bass_groove as _b, classify_synth_groove as _s)
        return None, _b, _s
    except Exception:
        def _b(grid: List[int], length: int) -> str:  # noqa: ANN001,ANN202
            s = {x for x in grid if x < 16}
            d = len(s) / 16.0
            if d >= 0.6:
                return "rolling"
            if {2, 6, 10, 14}.intersection(s) and not {0, 4, 8, 12}.issubset(s):
                return "offbeat"
            if d <= 0.25:
                return "sustained"
            return "stabs"

        def _s(grid: List[int], density: float) -> str:  # noqa: ANN001,ANN202
            if density >= 0.5:
                return "riff"
            if density <= 0.2:
                return "pad"
            return "hook"
        return None, _b, _s


# ------------------------------------------------------------------ training

def build_harmonic_tables(tracks: Sequence[Any],
                          bar_sections: Optional[Mapping[str, Sequence[str]]] = None,
                          max_bars_per_track: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate pitch-class histograms + groove-pair counts per context.

    ``tracks``: ``TrackAnalysis`` objects (needs ``.track_id``, ``.key``,
    ``.tatums`` with ``stem_segments``, ``.segments`` with ``pitches``).
    ``bar_sections``: optional arranger-like labels ``{track_id: [sec per bar]}``.
    """
    _, bass_groove_fn, synth_groove_fn = _groove_classifiers()
    hist: Dict[str, Dict[str, Dict[int, float]]] = {}
    pairs: Dict[str, Dict[str, float]] = {}
    bars: Dict[str, int] = {}
    total_bars = 0

    for tr in tracks:
        tatums = list(getattr(tr, "tatums", []) or [])
        if len(tatums) < 4:
            continue
        seg_by_id = {s.id: s for s in (getattr(tr, "segments", []) or [])}
        track_root, track_minor = parse_key(getattr(tr, "key", "C major"))
        wins = _bar_windows(tatums)
        if max_bars_per_track:
            wins = wins[:max_bars_per_track]

        # Per-bar features for section inference.
        feats: List[Dict[str, float]] = []
        bar_grids: List[Dict[str, List[int]]] = []
        prev: Dict[str, Optional[str]] = {}
        for win in wins:
            grids: Dict[str, List[int]] = {}
            louds: List[float] = []
            for stem in ("drums", "bass", "other", "vocals"):
                on, prev[stem] = _bar_onsets(win, tatums, stem, prev.get(stem))
                grids[stem] = [o for o in on if o < 16]
                for ti in win:
                    ids = (tatums[ti].stem_segments or {}).get(stem, [])
                    if ids:
                        sg = seg_by_id.get(ids[-1])
                        if sg is not None and not getattr(sg, "is_silent", False):
                            louds.append(_loud_energy(float(getattr(sg, "loudness_max", -30.0))))
                            break
            e = float(sum(louds) / len(louds)) if louds else 0.0
            feats.append({"energy": e,
                          "drum_d": len(grids["drums"]) / 16.0,
                          "bass_d": len(grids["bass"]) / 16.0})
            bar_grids.append(grids)

        tid = str(getattr(tr, "track_id", "?"))
        given = list((bar_sections or {}).get(tid, []) or [])
        sections = ([str(s).upper() for s in given] + infer_bar_sections(feats))[:len(wins)]

        for bi, win in enumerate(wins):
            # Bar chord root: dominant bass pc, else other, else track key.
            bass_pcs: List[int] = []
            other_pcs: List[int] = []
            for stem, acc in (("bass", bass_pcs), ("other", other_pcs)):
                seen = set()
                for ti in win:
                    for sid in (tatums[ti].stem_segments or {}).get(stem, []):
                        if sid in seen:
                            continue
                        seen.add(sid)
                        sg = seg_by_id.get(sid)
                        if sg is None or getattr(sg, "is_silent", False):
                            continue
                        try:
                            acc.append(quantize_chroma(getattr(sg, "pitches")))
                        except (ValueError, TypeError):
                            continue
            if bass_pcs:
                root = max(set(bass_pcs), key=lambda p: (bass_pcs.count(p), -p))
            elif other_pcs:
                root = max(set(other_pcs), key=lambda p: (other_pcs.count(p), -p))
            else:
                root = track_root
            ctx = context_key(sections[bi] if bi < len(sections) else "GROOVE_A",
                              root, track_minor)
            h = hist.setdefault(ctx, {s: {} for s in PITCH_STEMS})
            # Pitch-class histograms: one vote per sounding segment.
            for stem in PITCH_STEMS:
                seen = set()
                for ti in win:
                    for sid in (tatums[ti].stem_segments or {}).get(stem, []):
                        if sid in seen:
                            continue
                        seen.add(sid)
                        sg = seg_by_id.get(sid)
                        if sg is None or getattr(sg, "is_silent", False):
                            continue
                        try:
                            pc = quantize_chroma(getattr(sg, "pitches"))
                        except (ValueError, TypeError):
                            continue
                        h[stem][pc] = h[stem].get(pc, 0.0) + 1.0
            # Groove co-occurrence for this bar.
            grids = bar_grids[bi]
            bg = bass_groove_fn(grids["bass"], 16)
            og = synth_groove_fn(grids["other"], len(grids["other"]) / 16.0)
            pk = f"{bg}|{og}"
            pairs.setdefault(ctx, {})[pk] = pairs.setdefault(ctx, {}).get(pk, 0.0) + 1.0
            bars[ctx] = bars.get(ctx, 0) + 1
            total_bars += 1

    return {"version": VERSION, "hist": hist, "groove_pairs": pairs,
            "bars": bars, "tracks": len(list(tracks)),
            "total_bars": total_bars,
            "meta": {"sections": list(SECTIONS), "stems": list(PITCH_STEMS)}}


# ------------------------------------------------------------ compact + load

def to_compact(tables: Mapping[str, Any], top_pcs: int = 8, top_pairs: int = 16,
               min_count: float = 1.0) -> Dict[str, Any]:
    """Prune to top entries only; returns a JSON-serializable dict (<1MB)."""
    hist: Dict[str, Any] = {}
    for ctx, stems in dict(getattr(tables, "get", lambda k, d=None: {})("hist", {})).items():
        hist[str(ctx)] = {}
        for stem, counts in dict(stems).items():
            items = sorted(counts.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            kept = [(str(int(k) % 12), round(float(v), 3))
                    for k, v in items if float(v) >= min_count][:max(1, top_pcs)]
            hist[str(ctx)][str(stem)] = dict(kept)
    pairs: Dict[str, Any] = {}
    for ctx, counts in dict(getattr(tables, "get", lambda k, d=None: {})("groove_pairs", {})).items():
        items = sorted(counts.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        pairs[str(ctx)] = {str(k): round(float(v), 3) for k, v in items
                           if float(v) >= min_count} 
        pairs[str(ctx)] = dict(list(pairs[str(ctx)].items())[:max(1, top_pairs)])
    bars = {str(k): int(v) for k, v in
            dict(getattr(tables, "get", lambda k, d=None: {})("bars", {})).items()}
    src_meta = dict(getattr(tables, "get", lambda k, d=None: {})("meta", {}) or {})
    return {"version": VERSION, "hist": hist, "groove_pairs": pairs, "bars": bars,
            "tracks": int(getattr(tables, "get", lambda k, d=None: {})("tracks", 0)),
            "total_bars": int(getattr(tables, "get", lambda k, d=None: {})("total_bars", 0)),
            "meta": {**src_meta, "top_pcs": top_pcs, "top_pairs": top_pairs,
                     "min_count": min_count}}


def load(obj: Any) -> Dict[str, Any]:
    """Restore tables from a compact dict, JSON string, or path to JSON."""
    if isinstance(obj, (str, Path)) and not (isinstance(obj, str) and obj.lstrip().startswith("{")):
        obj = json.loads(Path(obj).read_text())
    elif isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, Mapping):
        raise ValueError("harmonic tables must be a mapping")
    data = dict(obj)
    if int(data.get("version", VERSION)) != VERSION:
        raise ValueError(f"unsupported harmonic tables version {data.get('version')}")
    for key in ("hist", "groove_pairs", "bars"):
        if key not in data or not isinstance(data[key], Mapping):
            raise ValueError(f"harmonic tables missing '{key}'")
    data.setdefault("tracks", 0)
    data.setdefault("total_bars", 0)
    data.setdefault("meta", {})
    return data


def dumps(tables: Mapping[str, Any], **kw: Any) -> str:
    return json.dumps(to_compact(tables), **kw)


def loads(s: str) -> Dict[str, Any]:
    return load(s)


# -------------------------------------------------------------------- scoring

def _hist_array(counts: Mapping[Any, float], alpha: float = 0.5) -> np.ndarray:
    """Laplace-smoothed 12-bin probability vector from {pc: count}."""
    c = np.full(12, alpha, dtype=float)
    for k, v in dict(counts).items():
        try:
            c[int(k) % 12] += max(0.0, float(v))
        except (TypeError, ValueError):
            continue
    tot = c.sum()
    return c / tot if tot > 0 else np.full(12, 1.0 / 12)


def _stem_fit(cand: Dict[int, float], prob: np.ndarray) -> float:
    if not cand:
        return 0.5  # absent stem is neutral
    wsum = sum(cand.values())
    if wsum <= 0:
        return 0.5
    peak = float(prob.max())
    if peak <= 0:
        return 0.0
    return float(sum(w * float(prob[pc]) for pc, w in cand.items()) / wsum / peak)


def _lookup_context_hist(ctx: str, tables: Mapping[str, Any], stem: str) -> List[Tuple[float, np.ndarray]]:
    """(weight, prob-vector) pairs with fallback from exact context to global."""
    hist = tables.get("hist", {})
    try:
        sec, root_s, mode = ctx.split("|")
        root = int(root_s)
    except ValueError:
        sec, root, mode = "GROOVE_A", 0, "maj"
    other_mode = "maj" if mode == "min" else "min"
    cands = [
        (1.0, ctx),
        (0.5, f"{sec}|{root}|{other_mode}"),
        (0.6, None),   # same root+mode, any section (merged below)
        (0.3, None),   # same root, any section/mode (merged below)
        (0.15, None),  # global (merged below)
    ]
    out: List[Tuple[float, np.ndarray]] = []
    if ctx in hist and stem in hist[ctx]:
        out.append((cands[0][0], _hist_array(hist[ctx][stem])))
    alt = f"{sec}|{root}|{other_mode}"
    if alt in hist and stem in hist[alt]:
        out.append((cands[1][0], _hist_array(hist[alt][stem])))
    for w, pred in ((0.6, lambda k: k.endswith(f"|{root}|{mode}")),
                    (0.3, lambda k: k.split("|")[1] == str(root) if "|" in k else False),
                    (0.15, lambda k: True)):
        merged: Dict[int, float] = {}
        for k, stems in hist.items():
            try:
                if pred(k) and stem in stems:
                    for pc, v in stems[stem].items():
                        merged[int(pc) % 12] = merged.get(int(pc) % 12, 0.0) + float(v)
            except (TypeError, ValueError):
                continue
        if merged:
            out.append((w, _hist_array(merged)))
    if not out:
        out.append((1.0, np.full(12, 1.0 / 12)))
    return out


def _mixed_prob(parts: List[Tuple[float, np.ndarray]]) -> np.ndarray:
    wsum = sum(w for w, _ in parts)
    mix = sum((w * p for w, p in parts), np.zeros(12))
    return mix / wsum if wsum > 0 else np.full(12, 1.0 / 12)


def compat_score(bass_pitches: Any, other_pitches: Any, vocal_pitch: Any,
                 chord_root: int, is_minor: bool, section: str,
                 tables: Mapping[str, Any],
                 bass_groove: Optional[str] = None,
                 other_groove: Optional[str] = None,
                 clash_penalty: float = 0.6) -> float:
    """Rate a simultaneous (bass, other, vocal) combination 0..1.

    High when the pitch classes were frequently observed in this harmonic
    context in real songs; semitone clashes against the chord tones (and
    between the simultaneous stems) pull the score down.
    """
    ctx = context_key(section, chord_root, is_minor)
    cands = {"bass": _as_weighted_pcs(bass_pitches),
             "other": _as_weighted_pcs(other_pitches),
             "vocals": _as_weighted_pcs(vocal_pitch)}
    weights = {"bass": 0.40, "other": 0.35, "vocals": 0.25}
    present = {s: bool(c) for s, c in cands.items()}
    wsum = sum(w for s, w in weights.items() if present[s]) or 1.0
    combo = 0.0
    for stem, w in weights.items():
        if not present[stem]:
            continue
        prob = _mixed_prob(_lookup_context_hist(ctx, tables, stem))
        combo += (w / wsum) * _stem_fit(cands[stem], prob)

    # Clash penalties: semitones (dist 1) against chord tones ...
    tones = chord_tones(int(chord_root) % 12, bool(is_minor))
    tot_w = sum(sum(c.values()) for c in cands.values()) or 1.0
    clash_w = 0.0
    all_pcs: List[Tuple[str, int, float]] = []
    for stem, c in cands.items():
        for pc, w in c.items():
            all_pcs.append((stem, pc, w))
            if any(circ_dist(pc, t) == 1 for t in tones):
                clash_w += w
    # ... and semitone rubs between the simultaneous stems themselves.
    rub_w = 0.0
    for i in range(len(all_pcs)):
        for j in range(i + 1, len(all_pcs)):
            (si, pi, wi), (sj, pj, wj) = all_pcs[i], all_pcs[j]
            if si != sj and circ_dist(pi, pj) == 1:
                rub_w += min(wi, wj)
    penalty = math.exp(-clash_penalty * clash_w / tot_w) * math.exp(-0.4 * rub_w / tot_w)

    # Groove-pair bonus when the caller names the grooves in play.
    bonus = 0.0
    if bass_groove and other_groove:
        pk = f"{bass_groove}|{other_groove}"
        gp = dict(tables.get("groove_pairs", {}).get(ctx, {}) or {})
        tot = sum(float(v) for v in gp.values())
        if tot > 0 and pk in gp:
            bonus = 0.10 * float(gp[pk]) / tot

    return float(min(1.0, max(0.0, combo * penalty + bonus)))


class HarmonicCompat:
    """Convenience wrapper: build once, score many, compact to JSON."""

    def __init__(self, tables: Optional[Mapping[str, Any]] = None) -> None:
        self.tables: Dict[str, Any] = dict(tables) if tables else {
            "version": VERSION, "hist": {}, "groove_pairs": {}, "bars": {},
            "tracks": 0, "total_bars": 0, "meta": {}}

    def fit(self, tracks: Sequence[Any], **kw: Any) -> "HarmonicCompat":
        self.tables = build_harmonic_tables(tracks, **kw)
        return self

    def score(self, bass_pitches: Any, other_pitches: Any, vocal_pitch: Any,
              chord_root: int, is_minor: bool, section: str, **kw: Any) -> float:
        return compat_score(bass_pitches, other_pitches, vocal_pitch,
                            chord_root, is_minor, section, self.tables, **kw)

    def to_compact(self, **kw: Any) -> Dict[str, Any]:
        return to_compact(self.tables, **kw)

    def save(self, path: Any, **kw: Any) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_compact(**kw)))
        return p

    @classmethod
    def load(cls, obj: Any) -> "HarmonicCompat":  # noqa: F811
        return cls(load(obj))
