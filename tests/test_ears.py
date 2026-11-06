"""Ears fusion + merge: additive, quality-first, schema-preserving."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _manifest(tmp_path):
    manifest = {
        "version": 2,
        "kits": [
            {
                "track_id": "t1",
                "style": "breaks",
                "bpm": 129.0,
                "key": "F minor",
                "phrases": [
                    {"id": "p1", "start_bar": 0, "section": "GROOVE_A"},
                    {"id": "p2", "start_bar": 4, "section": "GROOVE_A"},
                ],
            }
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _dsp(tmp_path):
    scores = {}
    for pid in ("p1", "p2"):
        scores[pid] = {}
        for stem in ("drums", "bass", "other", "vocals"):
            scores[pid][stem] = {
                "quality_score": 0.9 if pid == "p1" else 0.3,
                "rms_db": -15.0,
                "peak": 0.5,
                "spectral_contrast_mean": 20.0,
                "onset_strength_mean": 2.0,
                "zcr_mean": 0.05,
                "loudness_lufs": -16.0,
                "essentia_available": False,
                "pitch_available": False,
                "loudness_lufs_source": "rms_db_fallback",
            }
    doc = {"version": "1.0.0", "scores": scores}
    p = tmp_path / "dsp.json"
    p.write_text(json.dumps(doc))
    return p


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (r.stdout, r.stderr)
    return r.stdout


def test_ears_score_ranking_and_reserved(tmp_path):
    manifest = _manifest(tmp_path)
    dsp = _dsp(tmp_path)
    out = tmp_path / "ears_scores.json"
    _run([
        sys.executable, str(ROOT / "scripts" / "ears_score.py"),
        "--ears-dsp", str(dsp), "--manifest", str(manifest), "--out", str(out),
    ])
    doc = json.loads(out.read_text())
    assert doc["version"] == 1
    assert set(doc["ears"].keys()) == {"p1", "p2"}
    # p1 (0.9 raw) must outrank p2 (0.3 raw) after per-style calibration.
    assert doc["ears"]["p1"]["q"] > doc["ears"]["p2"]["q"]
    for entry in doc["ears"].values():
        assert 0.0 <= entry["q"] <= 1.0
        assert len(entry["reserved"]) == 7
        assert all(0.0 <= v <= 1.0 for v in entry["reserved"])


def test_ears_merge_preserves_dual_writer(tmp_path):
    manifest = _manifest(tmp_path)
    dsp = _dsp(tmp_path)
    ears = tmp_path / "ears_scores.json"
    _run([
        sys.executable, str(ROOT / "scripts" / "ears_score.py"),
        "--ears-dsp", str(dsp), "--manifest", str(manifest), "--out", str(ears),
    ])
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({
        "version": 2, "kits": {"t1": {"quality": [0.5]}}, "stem_next": {"p1": {}},
    }))
    out_scores = tmp_path / "scores_ears.json"
    _run([
        sys.executable, str(ROOT / "scripts" / "ears_merge.py"),
        "--ears", str(ears), "--scores", str(scores), "--out-scores", str(out_scores),
    ])
    merged = json.loads(out_scores.read_text())
    assert merged["kits"] == {"t1": {"quality": [0.5]}}
    assert merged["stem_next"] == {"p1": {}}
    assert set(merged["ears"].keys()) == {"p1", "p2"}


def test_ears_merge_dataset_fills_reserved(tmp_path):
    manifest = _manifest(tmp_path)
    dsp = _dsp(tmp_path)
    ears = tmp_path / "ears_scores.json"
    _run([
        sys.executable, str(ROOT / "scripts" / "ears_score.py"),
        "--ears-dsp", str(dsp), "--manifest", str(manifest), "--out", str(ears),
    ])
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({
        "version": 2,
        "feature_dim": 160,
        "tracks": [{
            "track_id": "t1",
            "windows": [{"start_bar": 4, "features": [0.0] * 160, "section": "GROOVE_A"}],
        }],
        "kit_phrases": [{"phrase_id": "p1", "features": [0.0] * 160}],
    }))
    out_ds = tmp_path / "dataset_ears.json"
    _run([
        sys.executable, str(ROOT / "scripts" / "ears_merge.py"),
        "--ears", str(ears), "--dataset", str(dataset),
        "--manifest", str(manifest), "--out-dataset", str(out_ds),
    ])
    out = json.loads(out_ds.read_text())
    feats = out["tracks"][0]["windows"][0]["features"]
    assert len(feats) == 160
    # Reserved dims 41..47 must be filled (non-zero for at least LUFS).
    assert any(v != 0.0 for v in feats[41:48])
    assert all(0.0 <= v <= 1.0 for v in feats[41:48])
    # Original dataset untouched (zeros).
    orig = json.loads(dataset.read_text())
    assert orig["tracks"][0]["windows"][0]["features"][41:48] == [0.0] * 7
