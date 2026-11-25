import numpy as np
import pytest

from scripts.build_joint_ranker import (
    INPUT_DIM, MIX_OFFSET, STATE_DIM, STEMS, STEM_DIM, SUBSETS, build_phrase_states,
    build_runs, key_shift, state_for, stem_state_vector, transition_vector,
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


def test_joint_state_dimensions_are_interleaved():
    assert STEM_DIM == 90
    assert MIX_OFFSET == 360
    assert STATE_DIM == 424
    assert INPUT_DIM == 2137


def test_joint_state_and_transition_shapes():
    states = {}
    for pid in ("a", "b"):
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        for stem_index in range(4):
            vec[stem_index * STEM_DIM + 51] = 1.0
            vec[stem_index * STEM_DIM + 47] = stem_index + 1.0
        vec[MIX_OFFSET:] = 0.5
        states[pid] = vec
    current = state_for(states, {stem: "a" for stem in STEMS})
    candidate = state_for(states, {stem: "b" for stem in STEMS})
    assert current.shape == (STATE_DIM,)
    # state_for keeps the per-stem chunks in drums/bass/other/vocals order.
    for stem_index in range(4):
        assert current[stem_index * STEM_DIM + 47] == stem_index + 1.0
    assert current[MIX_OFFSET:].tolist() == [0.5] * 64
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


def _phrase_row(onsets, bass_chroma=None, other_chroma=None):
    feats = np.zeros(160, dtype=np.float32)
    for stem_index, stem in enumerate(STEMS):
        lo = 48 + stem_index * 16
        feats[lo:lo + 16] = onsets.get(stem, [0.0] * 16)
    feats[112:124] = bass_chroma if bass_chroma is not None else [0.0] * 12
    feats[124:136] = other_chroma if other_chroma is not None else [0.0] * 12
    feats[136 + 2] = 1.0   # bass root
    feats[148 + 5] = 1.0   # chord root
    return feats


def test_build_phrase_states_interleaves_90d_stem_chunks():
    manifest = {
        "clusters": [{"id": "c1", "key": "A minor", "bpm": 128.0, "kits": ["t1"]}],
        "kits": [{
            "track_id": "t1", "key": "A minor", "bpm": 128.0,
            "phrases": [{
                "id": "p1", "start_bar": 0, "duration_sec": 7.5,
                "section": "DROP", "energy": 0.9,
                "stems": {
                    stem: {"file": "x.opus", "peak": 0.5, "rms_db": -16.0}
                    for stem in STEMS
                },
                "embedding": [0.25] * 64,
            }],
        }],
    }
    onsets = {
        "drums": [1.0, 0.0, 0.0, 0.0] * 4,
        "bass": [0.0, 0.0, 1.0, 0.0] * 4,
    }
    bass_chroma = [0.1 * i for i in range(12)]
    other_chroma = [0.05 * i for i in range(12)]
    rows = {"p1": {"features": _phrase_row(onsets, bass_chroma, other_chroma).tolist()}}
    emb32 = {stem: {"p1": np.full(32, i + 1, dtype=np.float32)}
             for i, stem in enumerate(STEMS)}

    states, meta, _ = build_phrase_states(manifest, rows, {"p1": 0}, {}, emb32)
    state = states["p1"]
    assert state.shape == (STATE_DIM,)
    assert meta["p1"]["track_id"] == "t1"

    # Stem chunks are interleaved: each 90-d slice carries its own features.
    assert state[0:16].tolist() == onsets["drums"]
    assert state[STEM_DIM:STEM_DIM + 16].tolist() == onsets["bass"]

    drums, bass = state[0:STEM_DIM], state[STEM_DIM:2 * STEM_DIM]
    other, vocals = state[2 * STEM_DIM:3 * STEM_DIM], state[3 * STEM_DIM:4 * STEM_DIM]
    bass_root = [0.0] * 12
    bass_root[2] = 1.0
    chord_root = [0.0] * 12
    chord_root[5] = 1.0
    assert drums[16:28].tolist() == pytest.approx([0.0] * 12)
    assert bass[16:28].tolist() == pytest.approx(bass_chroma)
    assert bass[28:40].tolist() == pytest.approx(bass_root)
    assert other[16:28].tolist() == pytest.approx(other_chroma)
    assert other[28:40].tolist() == pytest.approx(chord_root)
    # Vocals use the chord root for both chroma and root (existing behaviour).
    assert vocals[16:28].tolist() == pytest.approx(chord_root)
    assert vocals[28:40].tolist() == pytest.approx(chord_root)

    # Per-stem timbre, not four timbre blocks, sits at 58:90 of each chunk.
    for index, stem in enumerate(STEMS):
        chunk = state[index * STEM_DIM:(index + 1) * STEM_DIM]
        assert chunk[58:90].tolist() == pytest.approx([index + 1.0] * 32)

    assert state[MIX_OFFSET:].tolist() == pytest.approx([0.25] * 64)
    # Shared phrase features are copied into every stem chunk.
    for index in range(4):
        chunk = state[index * STEM_DIM:(index + 1) * STEM_DIM]
        assert chunk[40] == pytest.approx(0.9)
        assert chunk[41 + 3] == 1.0                 # DROP
        assert chunk[47] == pytest.approx(0.0)
        assert chunk[48] == pytest.approx(1 / 6)


def _stem_vector(stem, onset_bits, *, rms_db=-20.0, peak=0.5):
    return stem_state_vector(
        stem=stem, onset_bits=onset_bits, chroma=[0.0] * 12, root=[0.0] * 12,
        energy=0.5, section=1, rms_db=rms_db, peak=peak, rpos=0.0, rlen=1 / 6,
        has_next=0.0, key_shift_norm=0.0, tempo_rate=0.0, timbre=[0.0] * 32,
    )


def test_stem_rhythm_features():
    drums = _stem_vector("drums", [1.0, 0.0, 0.0, 0.0] * 4)
    assert drums[52] == pytest.approx(0.25)        # onset density = 4/16
    assert drums[53] == pytest.approx(0.0)         # offbeats 2/6/10/14
    assert drums[54] == pytest.approx(0.5)         # 2 of 4 on 4/12
    assert drums[55] == pytest.approx(0.0)         # syncopation: all on 0/4/8/12

    bass = _stem_vector("bass", [0.0, 0.0, 1.0, 0.0] * 4)
    assert bass[52] == pytest.approx(0.25)
    assert bass[53] == pytest.approx(1.0)
    assert bass[54] == pytest.approx(0.0)
    assert bass[55] == pytest.approx(1.0)

    pattern = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    other = _stem_vector("other", pattern)
    assert other[52] == pytest.approx(6 / 16)
    assert other[53] == pytest.approx(0.0)
    assert other[54] == pytest.approx(2 / 6)
    assert other[55] == pytest.approx(1 - 4 / 6)

    silent = _stem_vector("drums", [0.0] * 16)
    assert all(silent[52:56] == 0.0)              # silence is not syncopation


def test_vocal_presence_and_crest_are_vocal_only():
    vocals = _stem_vector("vocals", [0.0] * 16, rms_db=-20.0, peak=0.5)
    assert vocals[56] == pytest.approx((20.0) / 25.0)
    crest = (20.0 * np.log10(0.5) + 20.0) / 24.0
    assert vocals[57] == pytest.approx(crest)

    drums = _stem_vector("drums", [0.0] * 16, rms_db=-20.0, peak=0.5)
    assert drums[56] == 0.0
    assert drums[57] == 0.0
    assert _stem_vector("vocals", [0.0] * 16, rms_db=-60.0, peak=1e-9)[56] == 0.0
    assert _stem_vector("vocals", [0.0] * 16, rms_db=-10.0, peak=1.0)[56] == 1.0
