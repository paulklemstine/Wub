"""Tests for virtual-stem pooling of mix-only tracks."""
import json

from wub_core.global_vocabulary import GlobalVocabularyBuilder


def _seg(sid, stem, start, dur, loud, cent, pitch):
    chroma = [0.0] * 12
    chroma[pitch] = 1.0
    return {
        "id": sid, "track_id": "mix1", "stem": stem, "start": start,
        "duration": dur, "is_silent": False, "loudness_start": loud - 10.0,
        "loudness_max": loud, "pitches": chroma,
        "timbre": [5.0] + [0.0] * 11, "spectral_centroid": cent,
    }


def test_virtual_stems_route_mix_other(tmp_path):
    segs = [
        _seg("d0", "other", 0.0, 0.12, -8.0, 500.0, 0),    # short+low -> drums
        _seg("b0", "other", 0.5, 1.00, -8.0, 300.0, 0),    # low cent -> bass
        _seg("v0", "other", 1.5, 0.40, -8.0, 2000.0, 5),   # vocal band -> vocals
        _seg("o0", "other", 2.5, 2.00, -8.0, 1500.0, 0),   # pad -> other only
    ]
    (tmp_path / "mix1.json").write_text(json.dumps({
        "track_id": "mix1", "stems_available": ["other"], "segments": segs,
    }))
    stems_dir = tmp_path / "stems"
    (stems_dir / "othertrack").mkdir(parents=True)
    builder = GlobalVocabularyBuilder(k_per_song=8)
    cands = builder.collect_candidates(tmp_path, stems_dir=stems_dir, virtual_stems=True)
    by_stem = {s: [m["segment_id"] for _, _, m in cands[s]] for s in cands}
    assert "d0" in by_stem["drums"]
    assert "b0" in by_stem["bass"]
    assert "v0" in by_stem["vocals"]
    assert "o0" in by_stem["other"]
    assert any(m.get("virtual") for _, _, m in cands["drums"])


def test_stemmed_tracks_unchanged(tmp_path):
    segs = [_seg("d0", "drums", 0.0, 0.12, -8.0, 500.0, 0)]
    (tmp_path / "t1.json").write_text(json.dumps({
        "track_id": "t1", "stems_available": ["drums"], "segments": segs,
    }))
    builder = GlobalVocabularyBuilder(k_per_song=8)
    cands = builder.collect_candidates(tmp_path, stems_dir=None, virtual_stems=True)
    assert [m["segment_id"] for _, _, m in cands["drums"]] == ["d0"]
    assert all(not m.get("virtual") for _, _, m in cands["drums"])
