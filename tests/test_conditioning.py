"""Tests for WubGPT conditioning (style/energy/bar8/bound) + weighted loss."""
from __future__ import annotations

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.model_transformer import (
    COND_ORDER,
    N_BAR8,
    N_BOUND,
    N_ENERGY,
    STEM_ORDER,
    StemVQ,
    WubGPT,
    build_dataset,
    build_transition_rows,
    dataset_token_histogram,
    energy_bucket,
    load_checkpoint,
    sample,
    save_checkpoint,
    select_transition_pairs,
    train_model,
)
from wub_core.schema import Segment, Tatum, TrackAnalysis


def _segs(track_id, louds, stems=STEM_ORDER):
    segs = []
    for st in stems:
        for i, loud in enumerate(louds):
            segs.append(
                Segment(
                    id=f"{st}_{i}",
                    track_id=track_id,
                    stem=st,
                    start=i * 0.25,
                    duration=0.25,
                    is_silent=False,
                    loudness_start=loud,
                    loudness_max=loud,
                    loudness_max_time=0.01,
                    pitches=[1.0] * 12,
                    timbre=[0.1 * i] * 12,
                    spectral_centroid=800.0,
                    zero_crossing_start_offset=0.0,
                )
            )
    return segs


def _track(track_id, n=32, louds=(-30.0, -20.0, -10.0, -6.0), bpm=120.0, key="C major",
           stems=STEM_ORDER):
    segs = _segs(track_id, louds, stems)
    tatums = []
    for i in range(n):
        tatums.append(
            Tatum(
                id=f"tat_{i}",
                index=i,
                bar_position=i // 16,
                beat_position=(i // 4) % 4,
                tatum_position=i % 4,
                start=i * 0.125,
                duration=0.125,
                confidence=0.9,
                stem_segments={st: [f"{st}_{i % len(louds)}"] for st in stems},
            )
        )
    return TrackAnalysis(
        track_id=track_id,
        filename=f"{track_id}.mp3",
        duration=n * 0.125,
        sample_rate=44100,
        bpm=bpm,
        key=key,
        tatums=tatums,
        segments=segs,
        stems_available=list(stems),
    )


def _vqs(k=4):
    vqs = {}
    for st in STEM_ORDER:
        sl = [
            SliceRef(stem=st, track_id="t", segment_id=f"{st}_{i}", start=i * 0.25,
                     duration=0.25, timbre=[0.1 * i] * 12, chroma=[0.0] * 12,
                     loud_max=-10.0, track_pos=i, track_bpm=120.0)
            for i in range(8)
        ]
        vqs[st] = StemVQ(stem=st, k=k).fit(sl)
    return vqs


def test_energy_bucket_edges():
    assert energy_bucket(-90.0) == 0
    assert energy_bucket(float("-inf")) == 0
    assert energy_bucket(0.0) == N_ENERGY - 1
    assert energy_bucket(-6.0) == N_ENERGY - 1
    assert energy_bucket(-48.0) == 1
    prev = -1
    for db in range(-90, 1):
        b = energy_bucket(float(db))
        assert 0 <= b < N_ENERGY
        assert b >= prev
        prev = b


def test_build_dataset_cond_streams():
    vqs = _vqs()
    tr = _track("song_a", n=32, louds=(-30.0, -6.0))
    rows = build_dataset([tr], vqs, style_map={"song_a": 3}, n_styles=12)
    assert len(rows) == 32
    assert all(set(COND_ORDER) <= set(r) for r in rows)
    assert {r["style"] for r in rows} == {3}
    assert {r["bound"] for r in rows} == {0}
    assert [r["bar8"] for r in rows[:17]] == [0] * 16 + [1]
    assert max(r["bar8"] for r in rows) < N_BAR8
    # loud rows alternate -30 (bucket 4) / -6 (bucket 7)
    assert rows[0]["energy"] == energy_bucket(-30.0)
    assert rows[1]["energy"] == energy_bucket(-6.0)
    # unknown style -> 0; out-of-range clamped -> 0
    rows2 = build_dataset([tr], vqs, style_map={"other": 2}, n_styles=12)
    assert {r["style"] for r in rows2} == {0}
    rows3 = build_dataset([tr], vqs, style_map={"song_a": 99}, n_styles=12)
    assert {r["style"] for r in rows3} == {0}
    rows4 = build_dataset([tr], vqs)  # no map at all
    assert {r["style"] for r in rows4} == {0}


def test_cond_model_trains_and_legacy_ignores_cond():
    vqs = _vqs()
    tr = _track("song_a")
    rows = build_dataset([tr], vqs, style_map={"song_a": 2}, n_styles=12)
    cond_vocabs = {"style": 13, "energy": N_ENERGY, "bar8": N_BAR8, "bound": N_BOUND}
    m = WubGPT.from_vqs(vqs, cond_vocabs=cond_vocabs, d_model=16, n_layers=1,
                        n_heads=2, max_steps=64)
    assert set(m.cond_emb.keys()) == set(COND_ORDER)
    losses = train_model(m, rows, epochs=2, batch_size=4, seed=0, weights="auto", T=16)
    assert len(losses) == 2 and all(np.isfinite(losses))
    # legacy model (no cond) trains on the same cond-carrying rows, ignoring them
    m0 = WubGPT.from_vqs(vqs, d_model=16, n_layers=1, n_heads=2, max_steps=64)
    assert m0.cond_vocabs is None and len(m0.cond_emb) == 0
    losses0 = train_model(m0, rows, epochs=1, batch_size=4, seed=0, T=16)
    assert np.isfinite(losses0[0])


def test_weighted_vs_uniform_runs():
    vqs = _vqs()
    rows = build_dataset([_track("song_a")], vqs, style_map={"song_a": 1})
    m = WubGPT.from_vqs(vqs, d_model=16, n_layers=1, n_heads=2, max_steps=64)
    la = train_model(m, rows, epochs=1, batch_size=8, seed=1, weights="auto", T=16)
    m2 = WubGPT.from_vqs(vqs, d_model=16, n_layers=1, n_heads=2, max_steps=64)
    ln = train_model(m2, rows, epochs=1, batch_size=8, seed=1, weights=None, T=16)
    assert np.isfinite(la[0]) and np.isfinite(ln[0]) and la[0] > 0 and ln[0] > 0


def test_sample_cond_and_legacy_checkpoint_roundtrip(tmp_path):
    vqs = _vqs()
    rows = build_dataset([_track("song_a")], vqs, style_map={"song_a": 5})
    cond_vocabs = {"style": 13, "energy": N_ENERGY, "bar8": N_BAR8, "bound": N_BOUND}
    m = WubGPT.from_vqs(vqs, cond_vocabs=cond_vocabs, d_model=16, n_layers=1,
                        n_heads=2, max_steps=64)
    train_model(m, rows, epochs=1, batch_size=4, seed=0, T=16)
    arc = [2, 3, 4, 5, 6, 7]
    gen = sample(m, vqs, rows[:4], steps=20, temperature=0.8, top_k=2, seed=0,
                 style_id=5, energy=4, energy_schedule=arc, start_bar8=2)
    assert len(gen) == 20
    assert [g["style"] for g in gen] == [5] * 20
    assert [g["energy"] for g in gen] == arc + [4] * 14
    assert [g["bar8"] for g in gen[:16]] == [2] * 16
    assert [g["bar8"] for g in gen[16:20]] == [3] * 4
    ckpt = tmp_path / "cond.pt"
    save_checkpoint(ckpt, m, vqs, m.get_config())
    lm, _, cfg = load_checkpoint(ckpt)
    assert cfg["cond_vocabs"] == cond_vocabs
    # legacy checkpoint (no cond) loads into the new code path
    m0 = WubGPT.from_vqs(vqs, d_model=16, n_layers=1, n_heads=2, max_steps=64)
    ckpt0 = tmp_path / "legacy.pt"
    save_checkpoint(ckpt0, m0, vqs, m0.get_config())
    lm0, _, cfg0 = load_checkpoint(ckpt0)
    assert cfg0["cond_vocabs"] is None
    g0 = sample(lm0, vqs, rows[:4], steps=4, seed=0)
    assert len(g0) == 4


def test_transition_pair_gates_and_rows():
    vqs = _vqs()
    a = _track("aaa", bpm=120.0, key="C major")
    b = _track("bbb", bpm=122.0, key="C major")
    c = _track("ccc", bpm=150.0, key="F# minor")
    pairs = select_transition_pairs([a, b, c], n_pairs=8, seed=0)
    assert ("aaa", "bbb") in pairs or ("bbb", "aaa") in pairs
    assert not any("ccc" in p for p in pairs)  # tempo + key gates reject c
    d = _track("ddd", bpm=121.0, key="G major")
    pairs2 = select_transition_pairs([a, d], n_pairs=8, seed=0)
    assert pairs2 == []  # key gate
    pairs3 = select_transition_pairs([a, d], n_pairs=8, seed=0, require_same_key=False)
    assert len(pairs3) == 1
    by_id = {"aaa": a, "bbb": b}
    rows = build_transition_rows(by_id, vqs, [("aaa", "bbb")], tail=8, head=16,
                                 bound_width=8, style_map={"bbb": 4}, n_styles=12)
    assert len(rows) == 24
    assert [r["bound"] for r in rows] == [0] * 4 + [1] * 8 + [0] * 12
    assert {r["style"] for r in rows} == {4}
    hists = dataset_token_histogram(rows)
    assert set(hists) == {"r", "drums", "bass", "other", "vocals"}
    assert all(h.sum() == len(rows) for h in hists.values())


def test_metas_pairing_matches_analyses_pairing():
    from wub_core.model_transformer import (
        select_transition_pairs_from_metas,
        track_meta,
    )

    tracks = [
        _track("aaa", bpm=120.0, key="C major"),
        _track("bbb", bpm=122.0, key="C major"),
        _track("ccc", bpm=150.0, key="F# minor"),
        _track("ddd", bpm=121.0, key="G major"),
    ]
    sm = {"aaa": 1, "bbb": 2, "ccc": 3, "ddd": 4}
    direct = select_transition_pairs(tracks, style_map=sm, n_pairs=8, seed=0)
    metas = []
    for tr in tracks:
        m = track_meta(tr)
        m["style"] = sm[tr.track_id]
        metas.append(m)
    assert all(set(m) >= {"id", "bpm", "key", "loud", "style"} for m in metas)
    assert [m["id"] for m in metas] == ["aaa", "bbb", "ccc", "ddd"]
    assert metas[0]["key"] == "c" and metas[2]["key"] == "f#"
    via_metas = select_transition_pairs_from_metas(metas, n_pairs=8, seed=0)
    assert via_metas == direct
    # streaming invariant: metas are tiny (no segments/tatums attached)
    import pickle

    assert len(pickle.dumps(metas)) < 10000
