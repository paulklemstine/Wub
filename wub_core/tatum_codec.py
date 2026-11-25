"""Shared helpers for tatum-level modeling over global vocab slices.

- rank_index(prototypes): signature -> rank per stem (exact match).
- nearest_rank(...): fuzzy fallback mapping a bar to the closest top-300
  prototype by onset-grid + slice-category distance (medoid noise makes
  exact slice-ID signatures brittle; grooves survive).
- onset_spans(steps): contiguous-tatum spans between onsets, capped at 8.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

STEMS = ("drums", "bass", "other", "vocals")


def rank_index(prototypes: Dict[str, list]) -> Dict[str, Dict[str, int]]:
    return {s: {p["signature"]: i for i, p in enumerate(prototypes[s])} for s in STEMS}


def nearest_rank(
    stem: str,
    sig: str,
    toks: List[Optional[str]],
    rank: Dict[str, int],
    proto_grids: List[tuple],
    proto_cats: List[tuple],
    id_cat: Dict[str, str],
) -> Tuple[int, bool]:
    if sig in rank:
        return rank[sig], True
    grid = tuple(i for i, t in enumerate(toks) if t)
    cats = tuple(id_cat.get(t, ".") if t else "." for t in toks)
    best, bestd = 300, 1e9
    for r, (pg, pc) in enumerate(zip(proto_grids, proto_cats)):
        common = len(set(grid) & set(pg))
        d = (len(grid) + len(pg) - 2 * common)
        for i in grid:
            if i < 16 and pc[i] != "." and cats[i] != pc[i]:
                d += 0.5
        if d < bestd:
            bestd, best = d, r
    return best, False


def onset_spans(steps: List[Optional[str]], cap: int = 8) -> Dict[int, int]:
    """Map onset step -> span length (tatums until next onset), capped."""
    onsets = [i for i, t in enumerate(steps) if t]
    spans: Dict[int, int] = {}
    for k, o in enumerate(onsets):
        nxt = onsets[k + 1] if k + 1 < len(onsets) else 16
        spans[o] = max(1, min(cap, nxt - o))
    return spans
