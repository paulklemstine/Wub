"""Compact learned section-transition operators for the generative DJ mix.

Mines every real section boundary in the corpus (``TrackAnalysis`` via
``wub_core.corpus``). Boundaries use arranger-like section labels when the
caller supplies them (``bar_sections``), otherwise sections are inferred
per bar from energy jumps (loudness + onset density), which places
boundaries exactly where the music audibly shifts.

For each boundary type (e.g. ``BUILDUP->DROP``) we aggregate:

* mean energy curve over the 2 bars before + 2 bars after
  (``pre_curve`` / ``post_curve``, loudness + drum density/velocity);
* drum onset template of the last bar before (``fill_grid`` -- captures
  fills!) and of the first bar after (``entry_grid``);
* the distribution of groove changes per stem (``groove_changes``), using
  the groove classifiers from ``wub_core.patterns``.

``transition_template(from_sec, to_sec, tables)`` returns those operators
for generative crossfades; ``energy_delta(from, to, tables)`` gives the
expected energy step. ``to_compact`` / ``load`` keep the JSON tiny (<1MB).
Pure Python + numpy, no torch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VERSION = 1
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")

#: Prior section energies (fallback when a boundary type was never observed).
SECTION_ENERGY = {
    "INTRO": 0.25,
    "GROOVE_A": 0.45,
    "BUILDUP": 0.65,
    "DROP": 0.85,
    "BREAKDOWN": 0.20,
}

PRE_LEN = 2   # bars before the boundary
POST_LEN = 2  # bars after the boundary


def _norm_section(sec: Any) -> str:
    s = str(sec or "GROOVE_A").upper()
    return s if s in SECTIONS else "GROOVE_A"


def boundary_key(from_sec: str, to_sec: str) -> str:
    return f"{_norm_section(from_sec)}->{_norm_section(to_sec)}"


# ------------------------------------------------------------- bar analysis

def _bar_windows(tatums: Sequence[Any], bar_len: int = 16) -> List[List[int]]:
    downs = [i for i, t in enumerate(tatums)
             if getattr(t, "beat_position", 0) == 0 and getattr(t, "tatum_position", 0) == 0]
    n = len(tatums)
    if len(downs) >= 1:
        wins = [list(range(d, min(d + bar_len, n))) for d in downs if d + 1 < n]
        wins = [w for w in wins if len(w) >= 4]
        if wins:
            return wins
    return [list(range(i, min(i + bar_len, n))) for i in range(0, n, bar_len) if n - i >= 4]


def _loud_energy(loud_max: float) -> float:
    return float(min(1.0, max(0.0, (loud_max + 60.0) / 60.0)))


def _groove_fns():  # patterns.py when available, else local mirrors
    try:
        from wub_core.patterns import (  # type: ignore
            classify_drum_groove as _d,
            classify_bass_groove as _b,
            classify_synth_groove as _s,
        )
        return _d, _b, _s
    except Exception:
        def _d(grid: List[int], length: int) -> str:  # noqa: ANN001,ANN202
            s = {x for x in grid if x < 16}
            d = len(s) / 16.0
            if d >= 0.8:
                return "fill"
            if {0, 4, 8, 12}.issubset(s):
                return "four_on_floor_hats" if s & {2, 6, 10, 14} else "four_on_floor"
            return "minimal" if d <= 0.25 else "breakbeat"

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
        return _d, _b, _s


def _bar_features(tatums: Sequence[Any], seg_by_id: Mapping[str, Any],
                  win: List[int], prev: Dict[str, Optional[str]],
                  groove_fns: Tuple[Any, Any, Any]) -> Dict[str, Any]:
    drum_fn, bass_fn, synth_fn = groove_fns
    grids: Dict[str, List[int]] = {}
    bar_louds: List[float] = []
    for stem in ("drums", "bass", "other", "vocals"):
        onsets: List[int] = []
        cur = prev.get(stem)
        for step, ti in enumerate(win):
            ids = (tatums[ti].stem_segments or {}).get(stem, []) if ti < len(tatums) else []
            seg = ids[-1] if ids else None
            if seg is not None and (step == 0 or seg != cur):
                onsets.append(step)
            cur = seg
        prev[stem] = cur
        grids[stem] = [o for o in onsets if o < 16]
    for ti in win:
        for stem in ("bass", "other", "vocals", "drums"):
            for sid in (tatums[ti].stem_segments or {}).get(stem, []):
                sg = seg_by_id.get(sid)
                if sg is not None and not getattr(sg, "is_silent", False):
                    bar_louds.append(_loud_energy(float(getattr(sg, "loudness_max", -30.0))))
                    break
            else:
                continue
            break
    loud_e = float(sum(bar_louds) / len(bar_louds)) if bar_louds else 0.0
    drum_d = len(grids["drums"]) / 16.0
    # Energy fuses loudness with rhythmic density/velocity so fills and
    # drop entries move the curve even at matched loudness.
    energy = 0.65 * loud_e + 0.35 * drum_d
    return {
        "energy": float(energy),
        "loud_e": float(loud_e),
        "drum_d": float(drum_d),
        "bass_d": len(grids["bass"]) / 16.0,
        "grids": grids,
        "grooves": {
            "drums": drum_fn(grids["drums"], 16),
            "bass": bass_fn(grids["bass"], 16),
            "other": synth_fn(grids["other"], len(grids["other"]) / 16.0),
            "vocals": synth_fn(grids["vocals"], len(grids["vocals"]) / 16.0),
        },
    }


def _classify_bar_section(energy: float, drum_d: float, bass_d: float) -> str:
    if energy >= 0.6 and drum_d >= 0.2:
        return "DROP"
    if energy >= 0.3 or drum_d >= 0.12 or bass_d >= 0.12:
        return "GROOVE_A"
    if energy < 0.2 and drum_d < 0.08 and bass_d < 0.08:
        return "BREAKDOWN"
    return "INTRO"


def infer_bar_sections(feats: Sequence[Mapping[str, float]]) -> List[str]:
    """Per-bar sections from energy jumps (+BUILDUP ramp override)."""
    labels = [_classify_bar_section(float(f.get("energy", 0.5)),
                                    float(f.get("drum_d", 0.0)),
                                    float(f.get("bass_d", 0.0))) for f in feats]
    n = len(labels)
    for i in range(1, n - 1):  # smooth isolated single-bar blips
        if labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
            labels[i] = labels[i - 1]
    for i in range(n):  # rising ramps into a DROP are BUILDUPs
        if labels[i] != "DROP":
            continue
        j = i - 1
        ramp: List[int] = []
        while j >= 0 and len(ramp) < 4 and labels[j] in ("GROOVE_A", "INTRO"):
            ramp.append(j)
            j -= 1
        ramp.reverse()
        if len(ramp) >= 2:
            es = [float(feats[k].get("energy", 0.0)) for k in ramp]
            if es[-1] - es[0] >= 0.15 and all(b >= a - 0.02 for a, b in zip(es, es[1:])):
                for k in ramp:
                    labels[k] = "BUILDUP"
    return labels


# ------------------------------------------------------------------ training

def build_transition_tables(tracks: Sequence[Any],
                            bar_sections: Optional[Mapping[str, Sequence[str]]] = None,
                            jump_thresh: float = 0.25) -> Dict[str, Any]:
    """Mine every section boundary in the corpus.

    ``bar_sections``: optional arranger-like labels ``{track_id: [sec/bar]}``;
    without them, sections come from energy-jump inference. ``jump_thresh``
    additionally splits same-label bars separated by a large energy step.
    """
    groove_fns = _groove_fns()
    acc: Dict[str, Dict[str, Any]] = {}
    total = 0

    for tr in tracks:
        tatums = list(getattr(tr, "tatums", []) or [])
        if len(tatums) < 8:
            continue
        seg_by_id = {s.id: s for s in (getattr(tr, "segments", []) or [])}
        wins = _bar_windows(tatums)
        if len(wins) < 2:
            continue
        prev: Dict[str, Optional[str]] = {}
        feats = [_bar_features(tatums, seg_by_id, w, prev, groove_fns) for w in wins]

        tid = str(getattr(tr, "track_id", "?"))
        given = list((bar_sections or {}).get(tid, []) or [])
        if given:
            sections = [_norm_section(s) for s in given[:len(wins)]]
            sections += infer_bar_sections(feats)[len(sections):len(wins)]
        else:
            sections = infer_bar_sections(feats)

        for b in range(1, len(wins)):
            frm, to = sections[b - 1], sections[b]
            jump = abs(feats[b]["energy"] - feats[b - 1]["energy"]) >= jump_thresh
            if frm == to and not (jump and feats[b]["energy"] != feats[b - 1]["energy"]):
                continue
            key = boundary_key(frm, to)
            e = acc.setdefault(key, {
                "count": 0,
                "pre_sum": [0.0] * PRE_LEN, "pre_n": [0] * PRE_LEN,
                "post_sum": [0.0] * POST_LEN, "post_n": [0] * POST_LEN,
                "fill_sum": [0.0] * 16, "entry_sum": [0.0] * 16,
                "groove_changes": {s: {} for s in ("drums", "bass", "other", "vocals")},
            })
            e["count"] += 1
            total += 1
            # 2 bars before + 2 bars after (missing edge bars stay uncounted).
            for k in range(PRE_LEN):
                src = b - PRE_LEN + k
                if 0 <= src < len(feats):
                    e["pre_sum"][k] += feats[src]["energy"]
                    e["pre_n"][k] += 1
            for k in range(POST_LEN):
                src = b + k
                if 0 <= src < len(feats):
                    e["post_sum"][k] += feats[src]["energy"]
                    e["post_n"][k] += 1
            # Drum templates: last bar before (fills!) / first bar after.
            for step in feats[b - 1]["grids"]["drums"]:
                if 0 <= step < 16:
                    e["fill_sum"][step] += 1.0
            for step in feats[b]["grids"]["drums"]:
                if 0 <= step < 16:
                    e["entry_sum"][step] += 1.0
            # Groove-change distribution per stem.
            for stem in ("drums", "bass", "other", "vocals"):
                g0 = feats[b - 1]["grooves"][stem]
                g1 = feats[b]["grooves"][stem]
                gk = f"{g0}|{g1}"
                d = e["groove_changes"][stem]
                d[gk] = d.get(gk, 0.0) + 1.0

    transitions: Dict[str, Any] = {}
    for key, e in acc.items():
        n = max(1, int(e["count"]))
        transitions[key] = {
            "count": int(e["count"]),
            "pre_energy": [round(e["pre_sum"][k] / max(1, e["pre_n"][k]), 4)
                           for k in range(PRE_LEN)],
            "post_energy": [round(e["post_sum"][k] / max(1, e["post_n"][k]), 4)
                            for k in range(POST_LEN)],
            "fill_prob": [round(v / n, 4) for v in e["fill_sum"]],
            "entry_prob": [round(v / n, 4) for v in e["entry_sum"]],
            "groove_changes": {s: dict(d) for s, d in e["groove_changes"].items()},
        }
    return {"version": VERSION, "transitions": transitions,
            "total_boundaries": total,
            "meta": {"sections": list(SECTIONS),
                     "pre_len": PRE_LEN, "post_len": POST_LEN,
                     "jump_thresh": jump_thresh}}


# ------------------------------------------------------------ compact + load

def to_compact(tables: Mapping[str, Any], top_groove_pairs: int = 8) -> Dict[str, Any]:
    """Prune groove-change tails + round; JSON-serializable dict (<1MB)."""
    src = dict(tables.get("transitions", {}) or {})
    out: Dict[str, Any] = {}
    for key, e in src.items():
        gc: Dict[str, Any] = {}
        for stem, dist in dict(e.get("groove_changes", {})).items():
            items = sorted(dict(dist).items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
            gc[str(stem)] = {str(k): int(v) for k, v in items[:max(1, top_groove_pairs)]}
        out[str(key)] = {
            "count": int(e.get("count", 0)),
            "pre_energy": [round(float(v), 3) for v in e.get("pre_energy", [0.0] * PRE_LEN)],
            "post_energy": [round(float(v), 3) for v in e.get("post_energy", [0.0] * POST_LEN)],
            "fill_prob": [round(float(v), 3) for v in e.get("fill_prob", [0.0] * 16)],
            "entry_prob": [round(float(v), 3) for v in e.get("entry_prob", [0.0] * 16)],
            "groove_changes": gc,
        }
    return {"version": VERSION, "transitions": out,
            "total_boundaries": int(tables.get("total_boundaries", 0)),
            "meta": dict(tables.get("meta", {}) or {})}


def load(obj: Any) -> Dict[str, Any]:
    """Restore tables from a compact dict, JSON string, or path to JSON."""
    if isinstance(obj, (str, Path)) and not (isinstance(obj, str) and obj.lstrip().startswith("{")):
        obj = json.loads(Path(obj).read_text())
    elif isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, Mapping):
        raise ValueError("transition tables must be a mapping")
    data = dict(obj)
    if int(data.get("version", VERSION)) != VERSION:
        raise ValueError(f"unsupported transition tables version {data.get('version')}")
    if "transitions" not in data or not isinstance(data["transitions"], Mapping):
        raise ValueError("transition tables missing 'transitions'")
    data = to_compact(data)  # normalize shapes / rounding
    return data


def dumps(tables: Mapping[str, Any], **kw: Any) -> str:
    return json.dumps(to_compact(tables), **kw)


def loads(s: str) -> Dict[str, Any]:
    return load(s)


# ------------------------------------------------------------------- queries

def _as_binary(probs: Sequence[float], thresh: float = 0.5) -> List[int]:
    return [1 if float(p) >= thresh else 0 for p in probs]


def transition_template(from_sec: str, to_sec: str, tables: Mapping[str, Any],
                        thresh: float = 0.5) -> Dict[str, Any]:
    """Operator dict with pre_curve, post_curve, fill_grid, entry_grid.

    Falls back to the global boundary average (or neutral defaults) when the
    exact boundary type was never observed, so generation never crashes.
    """
    key = boundary_key(from_sec, to_sec)
    src = dict(tables.get("transitions", {}) or {})
    e = src.get(key)
    fallback = False
    if e is None:
        same_from = [v for k, v in src.items() if k.split("->")[0] == _norm_section(from_sec)]
        pool = same_from or list(src.values())
        if pool:
            pre = np.mean([v.get("pre_energy", [0.0] * PRE_LEN) for v in pool], axis=0)
            post = np.mean([v.get("post_energy", [0.0] * POST_LEN) for v in pool], axis=0)
            fill = np.mean([v.get("fill_prob", [0.0] * 16) for v in pool], axis=0)
            entry = np.mean([v.get("entry_prob", [0.0] * 16) for v in pool], axis=0)
            e = {"count": 0, "pre_energy": [round(float(v), 3) for v in pre],
                 "post_energy": [round(float(v), 3) for v in post],
                 "fill_prob": [round(float(v), 3) for v in fill],
                 "entry_prob": [round(float(v), 3) for v in entry],
                 "groove_changes": {}}
        else:
            e = {"count": 0,
                 "pre_energy": [SECTION_ENERGY[_norm_section(from_sec)]] * PRE_LEN,
                 "post_energy": [SECTION_ENERGY[_norm_section(to_sec)]] * POST_LEN,
                 "fill_prob": [0.0] * 16, "entry_prob": [0.0] * 16,
                 "groove_changes": {}}
        fallback = True
    fill_prob = [float(v) for v in e.get("fill_prob", [0.0] * 16)][:16]
    entry_prob = [float(v) for v in e.get("entry_prob", [0.0] * 16)][:16]
    fill_prob += [0.0] * (16 - len(fill_prob))
    entry_prob += [0.0] * (16 - len(entry_prob))
    return {
        "from": _norm_section(from_sec),
        "to": _norm_section(to_sec),
        "key": key,
        "count": int(e.get("count", 0)),
        "fallback": fallback,
        "pre_curve": [float(v) for v in e.get("pre_energy", [0.0] * PRE_LEN)],
        "post_curve": [float(v) for v in e.get("post_energy", [0.0] * POST_LEN)],
        "fill_prob": fill_prob,
        "entry_prob": entry_prob,
        "fill_grid": _as_binary(fill_prob, thresh),
        "entry_grid": _as_binary(entry_prob, thresh),
        "groove_changes": {str(s): {str(k): int(v) for k, v in dict(d).items()}
                           for s, d in dict(e.get("groove_changes", {})).items()},
    }


def energy_delta(from_sec: str, to_sec: str, tables: Mapping[str, Any]) -> float:
    """Expected energy step across the boundary (post mean minus pre mean)."""
    tpl = transition_template(from_sec, to_sec, tables)
    if tpl["count"] > 0 or not tpl["fallback"]:
        pre = float(np.mean(tpl["pre_curve"])) if tpl["pre_curve"] else 0.0
        post = float(np.mean(tpl["post_curve"])) if tpl["post_curve"] else 0.0
        return post - pre
    return SECTION_ENERGY[_norm_section(to_sec)] - SECTION_ENERGY[_norm_section(from_sec)]


class TransitionOps:
    """Convenience wrapper: mine once, query many, compact to JSON."""

    def __init__(self, tables: Optional[Mapping[str, Any]] = None) -> None:
        self.tables: Dict[str, Any] = dict(tables) if tables else {
            "version": VERSION, "transitions": {}, "total_boundaries": 0, "meta": {}}

    def fit(self, tracks: Sequence[Any], **kw: Any) -> "TransitionOps":
        self.tables = build_transition_tables(tracks, **kw)
        return self

    def template(self, from_sec: str, to_sec: str, **kw: Any) -> Dict[str, Any]:
        return transition_template(from_sec, to_sec, self.tables, **kw)

    def delta(self, from_sec: str, to_sec: str) -> float:
        return energy_delta(from_sec, to_sec, self.tables)

    def to_compact(self, **kw: Any) -> Dict[str, Any]:
        return to_compact(self.tables, **kw)

    def save(self, path: Any, **kw: Any) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_compact(**kw)))
        return p

    @classmethod
    def load(cls, obj: Any) -> "TransitionOps":  # noqa: F811
        return cls(load(obj))
