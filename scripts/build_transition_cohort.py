"""Mine a learned stem-cohort policy from real 4-bar song boundaries.

The shipped browser path changes at most 1-2 stems per handoff. Real songs
change 3-4 stems together at about half of their labeled section boundaries,
and the pitched trio (bass + other + vocals) often moves as one. This script
mines that distribution from ``data/processed/*.json`` (the same analysis the
rest of the kit bank is built from) and exports a small optional sidecar that

* raises the per-boundary movement budget only at section changes, and
* biases which stem cohort is chosen toward the learned co-change statistics.

It never reslices audio or replaces the judge/ranker: it only supplies prior
probabilities to the existing browser policy (see docs/SLICE_FLOW.md).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from wub_core.section_arcs import bar_features, segment_sections

STEMS = ("drums", "bass", "other", "vocals")
FORMAT = "wub-transition-cohort-v1"


def _mask(onsets) -> int:
    m = 0
    for s in onsets:
        m |= 1 << int(s)
    return m


def _jaccard(a: int, b: int) -> float:
    if not (a or b):
        return 1.0
    return (a & b).bit_count() / (a | b).bit_count()


def song_cells(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tatums = doc.get("tatums") or []
    seg_by_id = {str(s.get("id")): s for s in segs if s.get("id")}
    n = len(tatums) // 16
    if n < 12:
        return None
    feats = bar_features(segs, tatums)
    sections = segment_sections(feats)
    labels = [s.label for s in sections for _ in range(s.n_bars)]
    labels = (labels + ["?"] * n)[:n]

    prev_id = {s: None for s in STEMS}
    cells = []
    for c in range(n // 4):
        union = {s: 0 for s in STEMS}
        presence = {s: 0 for s in STEMS}
        chroma = []
        louds = []
        for bar in range(c * 4, (c + 1) * 4):
            for step in range(16):
                tat = tatums[bar * 16 + step]
                ss = tat.get("stem_segments") or {}
                for stem in STEMS:
                    ids = ss.get(stem, []) or []
                    cur = ids[-1] if ids else None
                    if cur is not None:
                        presence[stem] += 1
                    onset = (step == 0 and cur is not None) or (
                        cur is not None and cur != prev_id[stem])
                    if onset:
                        union[stem] |= 1 << step
                        seg = seg_by_id.get(str(cur))
                        if seg is not None:
                            louds.append(float(seg.get("loudness_max", -20.0)))
                            if stem in ("bass", "other", "vocals"):
                                p = np.asarray(seg.get("pitches") or [0.0] * 12, dtype=float)[:12]
                                if p.sum() > 1e-8:
                                    chroma.append(p / p.sum())
                    prev_id[stem] = cur
        mean_chroma = np.mean(np.stack(chroma), axis=0) if chroma else None
        cells.append({
            "union": union,
            "presence": {s: presence[s] / 64.0 for s in STEMS},
            "chroma": mean_chroma,
            "energy": float(np.mean(louds)) if louds else -60.0,
            "section": labels[c * 4],
        })
    return cells


def _cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    if d < 1e-9:
        return None
    return float(np.dot(a, b) / d)


def mine(processed: Path, max_tracks: int) -> Dict[str, Any]:
    paths = sorted(processed.glob("*.json"))
    step = max(1, len(paths) // max_tracks) if max_tracks > 0 else 1
    paths = paths[::step][:max_tracks] if max_tracks > 0 else paths

    counts = defaultdict(Counter)
    cohorts = defaultdict(Counter)
    stem_change = defaultdict(Counter)
    energy = defaultdict(list)
    harmony = defaultdict(list)
    tracks = 0
    for path in paths:
        try:
            cells = song_cells(json.loads(path.read_text()))
        except Exception:
            continue
        if not cells or len(cells) < 8:
            continue
        tracks += 1
        for c in range(1, len(cells)):
            prev, cur = cells[c - 1], cells[c]
            kind = "section" if prev["section"] != cur["section"] else "inside"
            changed = {}
            for s in STEMS:
                sim = _jaccard(prev["union"][s], cur["union"][s])
                pres = abs(prev["presence"][s] - cur["presence"][s])
                changed[s] = sim < 0.55 or pres > 0.18
            k = sum(changed.values())
            counts[kind][k] += 1
            cohorts[kind][tuple(s for s in STEMS if changed[s])] += 1
            for s in STEMS:
                stem_change[kind][(s, "change" if changed[s] else "hold")] += 1
            energy[kind].append(cur["energy"] - prev["energy"])
            cc = _cos(prev["chroma"], cur["chroma"])
            if cc is not None:
                harmony[kind].append(cc)

    def dist(counter: Counter, total: int) -> Dict[str, float]:
        return {str(k): round(v / max(1, total), 4) for k, v in sorted(counter.items())}

    inside_n = sum(counts["inside"].values())
    section_n = sum(counts["section"].values())
    section_3plus = sum(v for k, v in counts["section"].items() if k >= 3) / max(1, section_n)
    inside_3plus = sum(v for k, v in counts["inside"].items() if k >= 3) / max(1, inside_n)

    cohort_rows = []
    for kind in ("inside", "section"):
        total = inside_n if kind == "inside" else section_n
        for subset, value in cohorts[kind].most_common(16):
            cohort_rows.append({
                "kind": kind,
                "stems": list(subset),
                "p": round(value / max(1, total), 4),
                "count": value,
            })

    return {
        "format": FORMAT,
        "version": 1,
        "tracks": tracks,
        "boundaries": inside_n + section_n,
        "change_count": {
            "inside": dist(counts["inside"], inside_n),
            "section": dist(counts["section"], section_n),
        },
        "p_3plus": {
            "inside": round(inside_3plus, 4),
            "section": round(section_3plus, 4),
        },
        "cohorts": cohort_rows,
        "stem_change": {
            kind: {
                stem: round(v / max(1, sum(v2 for (s2, _), v2 in stem_change[kind].items() if s2 == stem)), 4)
                for stem in STEMS
                for (s, what), v in stem_change[kind].items()
                if s == stem and what == "change"
            }
            for kind in ("inside", "section")
        },
        "energy_delta": {
            kind: {
                "p10": round(float(np.percentile(v, 10)), 3),
                "p50": round(float(np.percentile(v, 50)), 3),
                "p90": round(float(np.percentile(v, 90)), 3),
            } for kind, v in energy.items()
        },
        "chroma_cos": {
            kind: {
                "p10": round(float(np.percentile(v, 10)), 3),
                "p50": round(float(np.percentile(v, 50)), 3),
            } for kind, v in harmony.items()
        },
        "policy": {
            "max_changes_inside": 2,
            "max_changes_section": 3 if section_3plus >= 0.35 else 2,
            "min_changes_section": 1,
            "pitch_cohort": ["bass", "other", "vocals"],
        },
        "handoff": {
            "enabled": True,
            "probability": 0.10,
            "max_per_source": 4,
            "max_stems": 4,
            "note": "rare deterministic-random full-source takeover at a section change",
        },
        "report": {
            "source": "data/processed/*.json (real 4-stem songs)",
            "section_labels": "wub_core.section_arcs.segment_sections",
            "change_rule": "4-bar union onset Jaccard < 0.55 or presence delta > 0.18",
            "label_provenance": "acoustic section detection, not human annotation",
            "track_disjoint": True,
            "note": "cohort priors only; no audio is regenerated or repitched",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()
    artifact = mine(args.processed, args.max_tracks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "tracks": artifact["tracks"],
        "boundaries": artifact["boundaries"],
        "p_3plus": artifact["p_3plus"],
        "policy": artifact["policy"],
        "top_section_cohorts": [r for r in artifact["cohorts"] if r["kind"] == "section"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
