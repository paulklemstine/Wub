"""Cross-song swap eval: songs rebuild from other songs' slices."""
import json
import sys


def _seg(sid, stem, start, dur, loud=-10.0, cent=800.0, pitch=0):
    chroma = [0.0] * 12
    chroma[pitch % 12] = 1.0
    return {
        "id": sid, "track_id": "t", "stem": stem, "start": start,
        "duration": dur, "is_silent": False, "loudness_start": loud,
        "loudness_max": loud, "loudness_max_time": 0.01,
        "pitches": chroma, "timbre": [1.0] * 12,
        "spectral_centroid": cent, "zero_crossing_start_offset": 0.0,
    }


def _tatum(i, mapping):
    return {
        "id": f"tat_{i:04d}", "index": i, "bar_position": i // 16,
        "beat_position": (i % 16) // 4, "tatum_position": i % 4,
        "start": i * 0.125, "duration": 0.125, "confidence": 0.95,
        "stem_segments": mapping,
    }


def _slice(gid, stem, tid, sid, start, cent=800.0, pitch=0):
    chroma = [0.0] * 12
    chroma[pitch % 12] = 1.0
    return {
        "id": gid, "stem": stem, "category": stem, "pitch": pitch,
        "duration": 0.25, "clarity_score": 0.9, "usage_count": 0,
        "centroid": cent, "track_id": tid, "segment_id": sid,
        "start": start, "loud_max": -10.0, "url": "",
        "feature_vector": [1.0] * 12 + chroma + [-10.0, 0.25],
    }


def test_cross_song_swap_small(tmp_path, monkeypatch):
    import scripts.eval_cross_song as ecs

    proc = tmp_path / "processed"
    proc.mkdir()
    out = tmp_path / "out.json"
    # two stemmed tracks with near-identical drum + bass vocab slices
    vocab = {"vocab_by_stem": {s: [] for s in ("drums", "bass", "other", "vocals")}, "budget": {}}
    for ti, tid in enumerate(["aaa", "bbb"]):
        segs = [
            _seg(f"seg_d_{ti}_0", "drums", 0.0, 0.25, cent=500.0),
            _seg(f"seg_d_{ti}_1", "drums", 0.5, 0.25, cent=510.0),
            _seg(f"seg_b_{ti}_0", "bass", 0.25, 0.5, cent=200.0),
        ]
        tatums = [
            _tatum(i, {"drums": [f"seg_d_{ti}_0" if i % 2 == 0 else f"seg_d_{ti}_1"]}) for i in range(8)
        ] + [
            _tatum(8 + i, {"bass": [f"seg_b_{ti}_0"]}) for i in range(8)
        ]
        (proc / f"{tid}.json").write_text(json.dumps({
            "track_id": tid, "filename": f"{tid}.mp3",
            "stems_available": ["drums", "bass"],
            "segments": segs, "tatums": tatums,
        }))
        vocab["vocab_by_stem"]["drums"].append(_slice(f"g_drums_{ti}a", "drums", tid, f"seg_d_{ti}_0", 0.0, 500.0))
        vocab["vocab_by_stem"]["drums"].append(_slice(f"g_drums_{ti}b", "drums", tid, f"seg_d_{ti}_1", 0.5, 510.0))
        vocab["vocab_by_stem"]["bass"].append(_slice(f"g_bass_{ti}", "bass", tid, f"seg_b_{ti}_0", 0.25, 200.0))
    vpath = tmp_path / "vocab.json"
    vpath.write_text(json.dumps(vocab))
    monkeypatch.setattr(sys, "argv", ["eval_cross_song",
                                      "--processed", str(proc),
                                      "--vocab", str(vpath),
                                      "--max-tracks", "2", "--bars", "1",
                                      "--out", str(out)])
    ecs.main()
    res = json.loads(out.read_text())
    assert res["tracks"] == 2
    assert set(res["per_stem"]) == {"drums", "bass", "other", "vocals"}
    assert 0.0 <= res["mean_swap_gap"] < 0.15
    assert res["n_used_slices"] > 0
