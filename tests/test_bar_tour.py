"""Tests for the bar-space tour planner and its value head."""
import json
from pathlib import Path

import numpy as np

from wub_core.bar_tour import (
    VALUE_INPUT_DIM, Cell, edge_scores, plan_tour, value_features,
    value_features_batch,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "frontend/public/data/kits"


def _net(dim=8):
    w1 = np.zeros((32, dim))
    w1[:min(32, dim), :min(32, dim)] = np.eye(min(32, dim))
    if dim >= 64:
        w1[:32, 32:64] = np.eye(32)
    return {
        "mean": [0.0] * dim, "std": [1.0] * dim,
        "w1": w1.tolist(), "b1": [0.0] * 32,
        "w2": [[1.0] * 32], "b2": [0.0],
    }


def test_value_features_shape_and_batch_agree():
    cur = np.arange(32, dtype=np.float64) / 32
    recent = np.ones(32, dtype=np.float64) * .5
    single = value_features(cur, [recent, recent], 30, .2)
    batch = value_features_batch(np.stack([cur, cur]), recent, 30, .2)
    assert single.shape == (VALUE_INPUT_DIM,)
    assert batch.shape == (2, VALUE_INPUT_DIM)
    assert np.allclose(single, batch[0])


def test_edge_scores_rank_stronger_candidate_higher():
    # A surrogate critic that likes brighter descriptors.
    dim = 108
    net = _net(dim)
    net["w2"] = [[0.0] * 32]
    for i in range(32):
        net["w2"][0][i] = 1.0 / 32
    models = {"critic": net}
    prev = np.full(32, .5, np.float32)
    candidates = np.stack([np.full(32, .2, np.float32), np.full(32, .9, np.float32)])
    scores = edge_scores(prev, candidates, "bass", models)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_plan_tour_respects_cooldown_and_change_budget():
    def cell(pid, track, value):
        return Cell(pid, track, "other", 0, np.full(32, value, np.float32))
    cells = [cell(f"p{i}", "t0" if i < 4 else "t1", .3 + .1 * (i % 4)) for i in range(8)]
    route = plan_tour(cells, 0, 4, {"critic": _net(108)}, beam=4, branch=3,
                      max_changes=1, cooldown=3)
    assert route[0] == 0
    assert len(route) == 5
    # No immediate repeat, and the cooldown window stays disjoint.
    assert all(route[i] != route[i - 1] for i in range(1, len(route)))
    assert len(set(route[-3:])) == 3
    changes = sum(cells[i].track != cells[i - 1].track for i in route[1:])
    assert changes <= 1


def test_shipping_value_sidecar_and_browser_tensor_are_valid():
    artifact = json.loads((BASE / "bar_tour_value.json").read_text())
    assert artifact["format"] == "wub-bar-tour-value-v1"
    assert artifact["input_dim"] == VALUE_INPUT_DIM
    assert artifact["report"]["val_mse"] < artifact["report"]["baseline_mse"]
    net = artifact["net"]
    assert len(net["w1"]) == 32
    assert all(len(row) == VALUE_INPUT_DIM for row in net["w1"])
    assert all(value > 0 for value in net["std"])
    source = (ROOT / "frontend/src/slice_flow.ts").read_text(encoding="utf-8")
    assert "bar_tour_value.json" in source
    assert "valueGain" in source
    sidecars = json.loads((BASE / "sidecars.json").read_text())
    assert sidecars["bar_tour_value"] is True
