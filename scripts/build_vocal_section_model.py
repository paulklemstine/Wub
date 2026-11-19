"""Learn 4/8/16-bar vocal section lengths and their backing from real songs.

Uses the mined multi-bar units in `vocal_phrases.json` plus section labels and
per-stem onsets in `data/processed/*.json` to answer:
  * how long a sung chorus/verse section is (4, 8 or 16 bars),
  * which backing stems (drums/bass/other) change when the vocal section starts,
  * which stems are actually present under it.

The exported sidecar is a ranking prior only: it never schedules a bar.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from wub_core.section_arcs import bar_features, segment_sections

BACKING = ("drums", "bass", "other")
STEMS = ("drums", "bass", "other", "vocals")


def category(label: str) -> str:
    s = label.upper()
    if s in ("DROP", "PEAK"):
        return "CHORUS"
    if s.startswith("GROOVE"):
        return "VERSE"
    if s in ("BUILDUP", "RISE", "INTRO"):
        return "BUILDUP"
    if s in ("BREAKDOWN", "BRIDGE", "OUTRO"):
        return "BREAKDOWN"
    return "VERSE"


def bin_bars(bars: int) -> int:
    if bars <= 6:
        return 4
    if bars <= 12:
        return 8
    return 16


def song_data(doc: Dict[str, Any]):
    segs = doc.get("segments") or []
    tats = doc.get("tatums") or []
    n = len(tats) // 16
    if n < 16:
        return None
    feats = bar_features(segs, tats)
    spans = segment_sections(feats)
    labels = [s.label for s in spans for _ in range(s.n_bars)]
    labels = (labels + ["?"] * n)[:n]
    by_id = {str(s.get("id")): s for s in segs if s.get("id")}
    prev = {s: None for s in STEMS}
    masks = []
    for b in range(n):
        union = {s: 0 for s in STEMS}
        for step in range(16):
            ss = (tats[b * 16 + step].get("stem_segments") or {})
            for stem in STEMS:
                ids = ss.get(stem, []) or []
                cur = ids[-1] if ids else None
                on = (step == 0 and cur is not None) or (cur is not None and cur != prev[stem])
                if on:
                    union[stem] |= 1 << step
                prev[stem] = cur
        masks.append(union)
    return labels, masks


def jaccard(a: int, b: int) -> float:
    if not (a or b):
        return 1.0
    return (a & b).bit_count() / (a | b).bit_count()


def cell_mask(masks, start: int, stem: str) -> int:
    out = 0
    for b in range(max(0, start), min(len(masks), start + 4)):
        out |= masks[b][stem]
    return out


def cell_density(masks, start: int, stem: str) -> float:
    bits = cell_mask(masks, start, stem).bit_count()
    return bits / 64.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=0)
    args = ap.parse_args()

    units = json.loads(args.units.read_text())["tracks"]
    paths = sorted(args.processed.glob("*.json"))
    if args.max_tracks > 0:
        paths = paths[::max(1, len(paths) // args.max_tracks)][:args.max_tracks]

    lengths = defaultdict(Counter)
    backing_change = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    presence = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cochange = defaultdict(lambda: [0, 0])
    short_sections = Counter()
    tracks = 0
    for path in paths:
        tid = path.stem
        rows = units.get(tid) or []
        if not rows:
            continue
        try:
            got = song_data(json.loads(path.read_text()))
        except Exception:
            continue
        if not got:
            continue
        labels, masks = got
        tracks += 1
        for unit in rows:
            start = int(unit.get("start_bar", 0))
            if start < 4 or start >= len(masks):
                continue
            cat = category(labels[start])
            lengths[cat][bin_bars(int(unit["bars"]))] += 1
            changed = []
            for stem in BACKING:
                prev_sim = jaccard(cell_mask(masks, start - 4, stem), cell_mask(masks, start, stem))
                prev_den = cell_density(masks, start - 4, stem)
                now_den = cell_density(masks, start, stem)
                is_change = prev_sim < 0.55 or abs(now_den - prev_den) > 0.18
                backing_change[cat][stem][1] += 1
                if is_change:
                    backing_change[cat][stem][0] += 1
                changed.append(is_change)
                presence[cat][stem][1] += 1
                if now_den > 0.05:
                    presence[cat][stem][0] += 1
            cochange[cat][1] += 1
            if all(changed):
                cochange[cat][0] += 1

        # 4-bar sung sections: a 4-bar cell whose vocal pattern starts fresh
        # (and does not simply continue into an 8/16-bar unit). The unit miner
        # cannot emit these, so measure them directly from bar-line activity.
        for c in range(len(masks) // 4):
            start = c * 4
            cur = cell_density(masks, start, "vocals")
            if cur < 0.05 or start >= len(labels):
                continue
            same_prev = c > 0 and jaccard(
                cell_mask(masks, start - 4, "vocals"), cell_mask(masks, start, "vocals")) >= 0.55
            same_next = c + 1 < len(masks) // 4 and jaccard(
                cell_mask(masks, start, "vocals"), cell_mask(masks, start + 4, "vocals")) >= 0.55
            section_flip = c > 0 and labels[start] != labels[start - 4]
            if (not same_prev) and (section_flip or not same_next):
                short_sections[category(labels[start])] += 1

    cats = ("CHORUS", "VERSE", "BUILDUP", "BREAKDOWN")

    def length_dist(cat: str):
        unit_total = sum(lengths[cat].values())
        total = unit_total + short_sections[cat]
        if total == 0:
            return {"4": 0.0, "8": 0.0, "16": 0.0}
        return {
            "4": round(short_sections[cat] / total, 4),
            "8": round(lengths[cat][8] / total, 4),
            "16": round(lengths[cat][16] / total, 4),
        }

    out = {
        "format": "wub-vocal-section-v1", "version": 1,
        "tracks": tracks,
        "lengths": {c: length_dist(c) for c in cats},
        "backing_change": {
            c: {s: round(backing_change[c][s][0] / max(1, backing_change[c][s][1]), 4)
                for s in BACKING} for c in cats
        },
        "backing_presence": {
            c: {s: round(presence[c][s][0] / max(1, presence[c][s][1]), 4)
                for s in BACKING} for c in cats
        },
        "all_backing_change": {
            c: round(cochange[c][0] / max(1, cochange[c][1]), 4) for c in cats
        },
        "report": {
            "source": "vocal_phrases.json units + data/processed/*.json sections",
            "bins": "<=6 bars -> 4, 8-12 -> 8, >=16 -> 16",
            "section_map": "DROP/PEAK=CHORUS, GROOVE*=VERSE, BUILDUP/RISE/INTRO=BUILDUP, BREAKDOWN/BRIDGE=BREAKDOWN",
            "use": "soft ranking prior for vocal length and backing cohort; no audio changes",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({k: out[k] for k in ("tracks", "lengths", "backing_change", "all_backing_change")}, indent=2))


if __name__ == "__main__":
    main()
