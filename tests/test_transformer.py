"""Tests for Phase 2B WubGPT and transformer pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from wub_core.model_markov import SliceRef, StemPool
from wub_core.model_transformer import (
    StemVQ,
    WubGPT,
    build_dataset,
    train_model,
    sample,
    tokens_to_slices,
    save_checkpoint,
    load_checkpoint,
    STEM_ORDER,
)
from wub_core.schema import TrackAnalysis, Tatum, Segment


def _fake_multi_stem_track(track_id: str, n: int = 16) -> TrackAnalysis:
    segs = []
    for stem in STEM_ORDER:
        for i in range(n):
            segs.append(
                Segment(
                    id=f"seg_{stem}_{i:04d}",
                    track_id=track_id,
                    stem=stem,
                    start=i * 0.25,
                    duration=0.25,
                    is_silent=False,
                    loudness_start=-20.0,
                    loudness_max=-8.0,
                    loudness_max_time=0.01,
                    pitches=[1.0] * 12,
                    timbre=[0.1 * i] * 12,
                    spectral_centroid=800.0,
                    zero_crossing_start_offset=0.0,
                )
            )
    tatums = []
    for i in range(n):
        tatums.append(
            Tatum(
                id=f"tatum_{i:04d}",
                index=i,
                bar_position=i // 4,
                beat_position=i % 4,
                tatum_position=0,
                start=i * 0.25,
                duration=0.25,
                confidence=0.9,
                stem_segments={
                    st: [f"seg_{st}_{i:04d}"] for st in STEM_ORDER
                },
            )
        )
    return TrackAnalysis(
        track_id=track_id,
        filename=f"{track_id}.mp3",
        duration=n * 0.25,
        sample_rate=44100,
        bpm=120.0,
        bpm_confidence=0.9,
        key="C major",
        key_confidence=0.8,
        mode=1,
        mode_confidence=0.8,
        time_signature=4,
        time_signature_confidence=0.9,
        bars=[],
        beats=[],
        tatums=tatums,
        segments=segs,
        stems_available=list(STEM_ORDER),
    )


def test_wubgpt_end_to_end():
    # 1. Fit small VQs
    vqs = {}
    for st in STEM_ORDER:
        vq = StemVQ(stem=st, k=4)
        dummy_slices = [
            SliceRef(
                stem=st,
                track_id="trk_dummy",
                segment_id=f"{st}_{i}",
                start=i * 0.25,
                duration=0.25,
                timbre=np.full(12, 0.1 * i, dtype=np.float32),
                chroma=np.zeros(12, dtype=np.float32),
                loud_max=-10.0,
                track_pos=i,
                track_bpm=120.0,
            )
            for i in range(8)
        ]
        vq.fit(dummy_slices)
        vqs[st] = vq
        assert vq.vocab_size == 5  # 4 clusters + 1 silence

    # 2. Build dataset
    track = _fake_multi_stem_track("trk_dummy", n=16)
    dataset = build_dataset([track], vqs, stride=1)
    assert len(dataset) == 16
    assert all(k in dataset[0] for k in ["r", "drums", "bass", "other", "vocals"])

    # 3. Model init & train
    model = WubGPT.from_vqs(vqs, d_model=32, n_heads=2, n_layers=2, max_steps=16)
    assert sum(p.numel() for p in model.parameters()) > 0
    losses = train_model(model, dataset, epochs=2, batch_size=4, lr=1e-3, device="cpu", seed=42)
    assert len(losses) == 2

    # 4. Autoregressive sampling
    seed_steps = dataset[:4]
    gen_rows = sample(model, vqs, seed_steps, steps=8, temperature=0.8, top_k=2, seed=42)
    assert len(gen_rows) == 8

    # 5. Tokens to slices
    pools = {
        st: StemPool(
            stem=st,
            slices=[
                SliceRef(
                    stem=st,
                    track_id="trk_dummy",
                    segment_id=f"{st}_{i}",
                    start=i * 0.25,
                    duration=0.25,
                    timbre=np.full(12, 0.1 * i, dtype=np.float32),
                    chroma=np.zeros(12, dtype=np.float32),
                    loud_max=-10.0,
                    track_pos=i,
                    track_bpm=120.0,
                )
                for i in range(8)
            ],
        )
        for st in STEM_ORDER
    }
    mapped = tokens_to_slices(gen_rows, pools, vqs)
    assert len(mapped) == 4
    for st in STEM_ORDER:
        assert len(mapped[st]) == 8

    # 6. Save & load checkpoint
    with tempfile.TemporaryDirectory() as td:
        ckpt_file = Path(td) / "wubgpt.pt"
        save_checkpoint(ckpt_file, model, vqs, model.get_config())
        loaded_model, loaded_vqs, cfg = load_checkpoint(ckpt_file)
        assert len(loaded_vqs) == 4
        assert loaded_model.d_model == model.d_model
