import numpy as np

from scripts.build_joint_ranker import (
    INPUT_DIM, STATE_DIM, STEMS, SUBSETS, build_runs, key_shift, state_for,
    transition_vector,
)


def test_key_shift_uses_camelot_neighbours():
    assert key_shift("A minor", "A minor") == 0
    assert key_shift("A minor", "C major") == 0       # relative major
    assert key_shift("A minor", "D minor") == 5       # subdominant
    assert key_shift("A minor", "E minor") == -5      # dominant, short direction


def test_joint_subsets_are_one_or_two_stems():
    assert len(SUBSETS) == 10
    assert all(1 <= len(subset) <= 2 for subset in SUBSETS)


def test_build_runs_uses_the_one_percent_tempo_rule():
    kit = {"phrases": [
        {"id": "a", "start_bar": 0, "duration_sec": 7.50},
        {"id": "b", "start_bar": 4, "duration_sec": 7.52},
        {"id": "c", "start_bar": 8, "duration_sec": 8.40},  # tempo break
        {"id": "d", "start_bar": 12, "duration_sec": 8.42},
    ]}
    runs = build_runs(kit)
    assert [[p["id"] for p in run] for run in runs] == [["a", "b"], ["c", "d"]]


def test_joint_state_and_transition_shapes():
    states = {}
    for pid in ("a", "b"):
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        for stem_index in range(4):
            vec[stem_index * 84 + 51] = 1.0
        vec[336:] = 0.5
        states[pid] = vec
    current = state_for(states, {stem: "a" for stem in STEMS})
    candidate = state_for(states, {stem: "b" for stem in STEMS})
    assert current.shape == (STATE_DIM,)
    x = transition_vector(
        current, current, candidate, ("drums",), "DROP", 0.9,
        current_bars=8, next_bars=4,
    )
    assert x.shape == (INPUT_DIM,)
    mask_offset = 5 * STATE_DIM
    assert x[mask_offset] == 1.0                    # drums changed
    assert x[mask_offset + 4] == 0.5                # one of two stems
    assert x[mask_offset + 5 + 3] == 1.0            # DROP target
    block_offset = mask_offset + 5 + 6
    assert x[block_offset + 1] == 1.0               # current 8-bar block
    assert x[block_offset + 3] == 1.0               # next 4-bar block
