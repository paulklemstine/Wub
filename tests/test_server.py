"""Backend + corpus tests (tiny synthetic fixture, no ingest dependency)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")


@pytest.fixture()
def store(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.test_phase1 import _fake_track
    from wub_core.corpus import CorpusStore

    proc = tmp_path / "processed"
    proc.mkdir()
    lib = tmp_path / "lib"
    lib.mkdir()
    sr = 44100
    for tid in ("aa", "bb"):
        tr = _fake_track(tid, n=8)
        (proc / f"{tid}.json").write_text(tr.model_dump_json())
        # 1.6s of tone so all 8x0.1s slices resolve (analysis is fake re: audio)
        t = np.arange(int(sr * 1.6)) / sr
        y = (0.3 * np.sin(2 * np.pi * (220 if tid == "aa" else 330) * t)).astype(np.float32)
        sf.write(str(lib / f"{tid}.mp3"), np.stack([y, y], axis=1), sr)
    # manifest maps track_id -> source path (filenames differ from _fake_track
    # on purpose to exercise manifest resolution)
    (proc / "_manifest.json").write_text(json.dumps({
        str(lib / "aa.mp3"): {"track_id": "aa", "path": str(lib / "aa.mp3"),
                              "size": 1, "mtime": 1.0, "status": "ok"},
        str(lib / "bb.mp3"): {"track_id": "bb", "path": str(lib / "bb.mp3"),
                              "size": 1, "mtime": 1.0, "status": "ok"},
    }))
    st = CorpusStore(processed_dir=proc, stems_base=tmp_path / "stems",
                     library_dirs=[lib])
    assert st.refresh() == 2
    return st


def test_corpus_refresh_and_audio(store):
    tids = {"aa", "bb"}
    audio_for, fallback = store.audio_for(tids)
    assert set(fallback) == tids  # manifest resolution
    s = store.summary()
    assert s["tracks"] == 2 and s["segments"] == 16
    assert store.refresh() == 0  # second scan is a no-op (mtime cache)


def test_stream_session_buffer(store):
    from server.stream_manager import StreamSession

    sess = StreamSession(store=store, chaos={"other": 0.9}, bpm=128.0, seed=1)
    ev = sess.next_event()
    assert ev["type"] == "tatum" and ev["slices"]["other"] is not None
    assert ev["start_time"] > 0 and ev["tatum_duration"] > 0
    sl = ev["slices"]["other"]
    assert sl["url"].startswith("/audio/") and sl["rate"] > 0


def test_slice_render(store):
    from server.stream_manager import render_slice_wav

    wav = render_slice_wav(store, "aa", "seg_other_0000")
    assert wav[:4] == b"RIFF" and len(wav) > 1000


def test_api_endpoints(store, monkeypatch):
    httpx = pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import server.main as main

    monkeypatch.setattr(main, "STORE", store)
    monkeypatch.setattr(main, "CACHE", main.SliceCache(cap=16))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["tracks"] == 2
        assert len(client.get("/api/tracks").json()) == 2
        r = client.get("/audio/aa/seg_other_0000.wav")
        assert r.status_code == 200 and r.content[:4] == b"RIFF"
        assert client.get("/audio/zz/seg_other_0000.wav").status_code == 404
        m = client.post("/api/mashup", json={"steps": 12, "seed": 2})
        assert m.status_code == 200 and m.content[:4] == b"RIFF"
        cm = client.post("/api/groovebox/cycle_mode", json={"mode": "4x16"})
        assert cm.status_code == 200 and cm.json()["cycle_mode"] == "4x16"


def test_synth_blend_math():
    from wub_core.synth_decoder import StemLatentSynth

    synth = StemLatentSynth(backend=None)
    a = np.ones((4, 8), dtype=np.float32)
    b = np.zeros((4, 8), dtype=np.float32)
    z = synth.blend("bass", a, b, 0.5)
    assert z.shape == (4, 8) and abs(float(np.abs(z).mean()) - 0.5) < 1e-6
