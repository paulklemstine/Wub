"""TSP-like navigation of the space of bars and transitions.

Cities are 4-bar cells (phrase x stem x bar) from the shipped acoustic bank.
The cost of driving from one cell to the next is learned: the final acoustic
critic, the motif-return head, a source-change price and a repetition penalty.
A value head estimates how much musical flow remains after the current cell,
which turns the greedy one-block choice into a receding-horizon search.

This is deliberately NOT a travelling-salesman solver in the exact sense: we
do not need to visit every cell once.  It is a prize-collecting tour with
cooldowns, so the search uses beam search over a candidate pool instead of
subtour elimination.  The name is about the problem shape, not the algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wub_core.slice_flow import DIM, STEMS, transition_features

VALUE_INPUT_DIM = DIM * 2 + 8


@dataclass(frozen=True)
class Cell:
    phrase: str
    track: str
    stem: str
    bar: int
    features: np.ndarray


def net_forward(net: dict, x: np.ndarray) -> np.ndarray:
    """Vectorised tanh MLP, identical to the browser's inference path."""
    mean = np.asarray(net["mean"], dtype=np.float64)
    std = np.asarray(net["std"], dtype=np.float64)
    z = (np.asarray(x, dtype=np.float64) - mean) / std
    hidden = np.tanh(z @ np.asarray(net["w1"], dtype=np.float64).T + np.asarray(net["b1"]))
    return hidden @ np.asarray(net["w2"], dtype=np.float64).T + np.asarray(net["b2"])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def edge_scores(prev: np.ndarray, candidates: np.ndarray, stem: str,
                models: dict, ctx: Optional[Sequence[float]] = None) -> np.ndarray:
    """Batch score of one previous cell against many candidate cells."""
    context = list(ctx or [0, 1, 1, 1, 0.75, 0, 0.7, 0.25])
    prev = np.asarray(prev, dtype=np.float32)
    x = np.concatenate([
        np.tile(prev, (len(candidates), 1)),
        np.asarray(candidates, dtype=np.float32),
        np.abs(np.asarray(candidates, dtype=np.float32) - prev),
    ], axis=1)
    identity = np.zeros((len(candidates), 4), dtype=np.float32)
    identity[:, STEMS.index(stem)] = 1
    full = np.concatenate([x, identity, np.tile(np.asarray(context, np.float32), (len(candidates), 1))], axis=1)
    score = sigmoid(net_forward(models["critic"], full)[:, 0])

    motif = np.full(len(candidates), 0.5)
    if "motif" in models:
        m = models["motif"]
        y = np.zeros((len(candidates), m["input_dim"]), dtype=np.float64)
        y[:, :DIM] = prev
        y[:, DIM:2 * DIM] = candidates
        y[:, 2 * DIM:3 * DIM] = np.abs(np.asarray(candidates) - prev)
        y[:, 3 * DIM + STEMS.index(stem)] = 1
        y[:, 3 * DIM + len(STEMS) + 1] = 1  # 16-bar return
        y[:, -1] = 0.75
        motif = sigmoid(net_forward(m["net"], y)[:, 0])
    return 0.7 * score + 0.3 * motif


def value_features(current: np.ndarray, history: Sequence[np.ndarray],
                   age: float, repetition: float) -> np.ndarray:
    """State descriptor for the value head: now, recent texture, time context."""
    recent = np.mean(np.stack(history[-4:]), axis=0) if history else np.asarray(current)
    return np.concatenate([
        np.asarray(current, dtype=np.float64),
        np.asarray(recent, dtype=np.float64),
        [min(2.0, age / 60.0), min(1.0, repetition), 0.5, 0.5, 0.7, 0.25, 0.0, 0.0],
    ])


def value_features_batch(feats: np.ndarray, recent: np.ndarray, age: float,
                         repetition: float) -> np.ndarray:
    """Vectorised counterpart of :func:`value_features` for a candidate pool."""
    tail = np.tile([min(2.0, age / 60.0), min(1.0, repetition), 0.5, 0.5, 0.7, 0.25, 0.0, 0.0],
                   (len(feats), 1))
    return np.concatenate([feats, np.tile(recent, (len(feats), 1)), tail], axis=1)


@dataclass
class TourState:
    path: Tuple[int, ...]
    score: float
    changes: int
    last_track: str

    def key(self) -> Tuple[int, ...]:
        return self.path[-3:]


def plan_tour(cells: Sequence[Cell], start: int, horizon: int,
              models: dict, beam: int = 12, branch: int = 12,
              max_changes: int = 3, cooldown: int = 6) -> List[int]:
    """Beam search for a high-flow route through the candidate cells.

    Returns indices into ``cells`` (including the fixed start).  The score is
    accumulated edge quality plus the discounted value estimate, with hard
    cooldown/change limits so the result stays musically legal.
    """
    if not cells or start < 0 or start >= len(cells):
        return []
    feats = np.stack([c.features for c in cells])
    models = dict(models)
    beam_states = [TourState((start,), 0.0, 0, cells[start].track)]
    for step in range(horizon):
        proposals: List[TourState] = []
        for state in beam_states:
            prev = state.path[-1]
            recent = [cells[i].features for i in state.path]
            reps = sum(1 for i in state.path if cells[i].phrase == cells[prev].phrase) / max(1, len(state.path))
            scores = edge_scores(feats[prev], feats, cells[prev].stem, models)
            names = np.array([c.track for c in cells])
            illegal = (names == state.last_track) & (state.changes >= max_changes)
            for seen in state.path[-cooldown:]:
                illegal |= np.arange(len(cells)) == seen
            scores[illegal] = -1e3
            if "value" in models:
                recent = np.mean(np.stack(recent), axis=0)
                vf = value_features_batch(feats, recent, step * 4, reps)
                scores += 0.15 * sigmoid(net_forward(models["value"], vf)[:, 0])
            top = np.argsort(-scores)[:branch]
            for j in top:
                if scores[j] <= -1e2:
                    continue
                gain = float(scores[j])
                proposals.append(TourState(
                    state.path + (int(j),),
                    state.score + gain * (0.94 ** step),
                    state.changes + int(cells[j].track != state.last_track),
                    cells[j].track,
                ))
        if not proposals:
            break
        proposals.sort(key=lambda s: s.score, reverse=True)
        # Keep distinct recent windows so the beam does not collapse to one song.
        kept: List[TourState] = []
        seen = set()
        for state in proposals:
            signature = state.key()
            if signature in seen and len(kept) >= beam // 2:
                continue
            seen.add(signature)
            kept.append(state)
            if len(kept) >= beam:
                break
        beam_states = kept
    return list(max(beam_states, key=lambda s: s.score).path) if beam_states else [start]
