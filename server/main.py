"""Wub backend: REST (tracks/stats/umap/audio/mashup) + WS 4-stem streamer.

All reads go through CorpusStore, which refreshes from disk on every
request — the API always reflects whatever has been ingested so far.
Run:  uvicorn server.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
import soundfile as sf

from server.stream_manager import SliceCache, StreamSession, render_slice_wav
from wub_core.corpus import CorpusStore


def _resolve_corpus_path(env_key: str, remote_mount: str, local_rel: str) -> Path:
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val)
    mount_path = Path(remote_mount)
    if mount_path.exists():
        return mount_path
    return Path(local_rel)


_PROCESSED_DIR = _resolve_corpus_path("WUB_PROCESSED_DIR", "/mnt/e/Wub/data/processed", "data/processed")
_STEMS_DIR = _resolve_corpus_path("WUB_STEMS_DIR", "/mnt/e/Wub/data/stems", "data/stems")
_LIBRARY_DIR = _resolve_corpus_path("WUB_LIBRARY_DIR", "/mnt/e/pandora", "pandora")

STORE = CorpusStore(processed_dir=_PROCESSED_DIR,
                    stems_base=_STEMS_DIR,
                    library_dirs=[_LIBRARY_DIR])
CACHE = SliceCache()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wub Infinite Jukebox")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    n = STORE.refresh(max_tracks=300)
    print(f"Wub store: {len(STORE.analyses)} tracks ({n} loaded at startup)")


@app.get("/health")
def health() -> dict:
    STORE.refresh(min_interval=10.0, max_tracks=300)
    return {"status": "ok", **STORE.summary()}


@app.get("/api/stats")
def stats() -> dict:
    STORE.refresh(min_interval=10.0, max_tracks=300)
    return STORE.summary()


@app.get("/api/tracks")
def tracks() -> list[dict]:
    STORE.refresh()
    return [{"track_id": t.track_id, "filename": t.filename, "bpm": t.bpm,
             "key": t.key, "duration": t.duration,
             "segments": len(t.segments), "tatums": len(t.tatums),
             "stems": list(t.stems_available)} for t in STORE.tracks]


@app.get("/api/umap")
def umap() -> list[dict]:
    return STORE.umap_points()


@app.get("/audio/{track_id}/{segment_id}.wav")
def audio_slice(track_id: str, segment_id: str) -> Response:
    STORE.refresh()
    key = f"{track_id}/{segment_id}"
    hit = CACHE.get(key)
    if hit is None:
        try:
            hit = render_slice_wav(STORE, track_id, segment_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        CACHE.put(key, hit)
    return Response(content=hit, media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/mashup")
def mashup(body: dict) -> FileResponse:
    """Offline render: {steps, chaos{stem:p}, force_tracks, seed, pitch_lock}."""
    from wub_core import model_markov as mm

    STORE.refresh()
    pools = STORE.pools()
    if not pools:
        raise HTTPException(status_code=409, detail="corpus empty")
    seq = mm.generate(pools, steps=int(body.get("steps", 200)),
                      chaos=body.get("chaos"), force_tracks=body.get("force_tracks"),
                      seed=int(body.get("seed", 7)))
    tids = {r.track_id for refs in seq.values() for r in refs}
    audio_for, fallback = STORE.audio_for(tids)
    tmp = Path(tempfile.mkdtemp(prefix="wub_")) / "mashup.wav"
    mm.render_mashup(seq, audio_for, fallback, tmp,
                     pitch_lock=bool(body.get("pitch_lock", False)))
    return FileResponse(str(tmp), media_type="audio/wav", filename="wub_mashup.wav")


@app.post("/api/feedback")
def submit_feedback(body: dict) -> dict:
    stem = body.get("stem")
    rating = int(body.get("rating", 0))
    track_id = body.get("track_id")
    pattern_id = body.get("pattern_id")
    from server.stream_manager import get_cached_pattern_bank
    from wub_core.groovebox import GrooveboxConductor
    bank = get_cached_pattern_bank(STORE)
    cond = GrooveboxConductor(bank)
    return cond.record_feedback(stem=stem, rating=rating, track_id=track_id, pattern_id=pattern_id)


@app.post("/api/groovebox/cycle_mode")
def api_set_cycle_mode(body: dict) -> dict:
    mode = str(body.get("mode", "8x16"))
    from server.stream_manager import get_cached_pattern_bank
    from wub_core.groovebox import GrooveboxConductor
    bank = get_cached_pattern_bank(STORE)
    cond = GrooveboxConductor(bank)
    ok = cond.set_cycle_mode(mode)
    return {"status": "ok" if ok else "error", "cycle_mode": cond.cycle_mode, "patterns_per_cycle": cond.patterns_per_cycle, "steps_per_pattern": cond.steps_per_pattern}


@app.get("/api/pattern_audio_feedback")
def api_pattern_audio_feedback() -> dict:
    from wub_core.pattern_audio_trainer import FEEDBACK_FILE
    if FEEDBACK_FILE.exists():
        try:
            return json.loads(FEEDBACK_FILE.read_text())
        except Exception:
            pass
    return {"version": 1, "patterns": {}, "grooves": {}, "eval_count": 0}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    STORE.refresh()
    session = StreamSession(store=STORE)
    await ws.send_json({"type": "hello",
                        "stems": [s for s in ("drums", "bass", "vocals", "other")],
                        "bpm": session.bpm, "sample_rate": 44100})
    # Burst initial 32 tatums (2 full bars) to prime the client buffer immediately
    for _ in range(32):
        ev = session.next_event()
        if ev is not None:
            await ws.send_json(ev)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.02)
                if msg.get("type") == "set" and "stem" in msg:
                    session.configure(stem=msg["stem"], chaos=msg.get("chaos"))
                elif msg.get("type") == "set_synth" and "stem" in msg:
                    session.configure(stem=msg["stem"], synth_blend=msg.get("blend"))
                elif msg.get("type") == "set_bpm":
                    session.configure(bpm=msg.get("bpm"))
                elif msg.get("type") == "set_chaos":
                    for stem, val in (msg.get("chaos") or {}).items():
                        session.configure(stem=stem, chaos=val)
                elif msg.get("type") == "set_key" and msg.get("key"):
                    session.key = str(msg["key"])  # reserved for key-lock (Phase 3+)
                elif msg.get("type") == "set_gravity":
                    stem = msg.get("stem")
                    if stem in ("drums", "bass", "vocals", "other"):
                        session.gravity[stem] = [msg.get("x"), msg.get("y")]
                elif msg.get("type") == "feedback":
                    res = session.record_feedback(
                        stem=msg.get("stem"),
                        rating=int(msg.get("rating", 0)),
                        track_id=msg.get("track_id"),
                        pattern_id=msg.get("pattern_id"),
                    )
                    await ws.send_json({"type": "feedback_ack", **res})
                elif msg.get("type") == "set_cycle_mode":
                    res = session.set_cycle_mode(str(msg.get("mode", "8x16")))
                    await ws.send_json({"type": "cycle_mode_ack", **res})
            except asyncio.TimeoutError:
                pass
            ev = session.next_event()
            if ev is None:
                await asyncio.sleep(0.2)
                continue
            await ws.send_json(ev)
            await asyncio.sleep(max(0.02, ev["tatum_duration"] * 0.9))
    except WebSocketDisconnect:
        pass


@app.get("/api/cells/status")
def api_cells_status() -> dict:
    STORE.refresh(min_interval=10.0, max_tracks=300)
    kits_dir = Path(__file__).resolve().parents[1] / "frontend" / "public" / "data" / "kits"
    return {
        "status": "ok",
        "processed_dir": str(_PROCESSED_DIR),
        "stems_dir": str(_STEMS_DIR),
        "library_dir": str(_LIBRARY_DIR),
        "cell_index_available": (kits_dir / "cell_index.json").exists(),
        "cell_compat_available": (kits_dir / "cell_compat.json").exists(),
        **STORE.summary(),
    }


@app.get("/api/cells/slice/{track_id}/{stem}.wav")
def api_cell_slice(track_id: str, stem: str, start_bar: int = 0, bars: float = 1.0) -> Response:
    if stem not in ("drums", "bass", "other", "vocals"):
        raise HTTPException(status_code=400, detail=f"invalid stem '{stem}'")
    stem_dir = _STEMS_DIR / track_id
    stem_path = None
    for ext in (".wav", ".flac", ".opus", ".mp3"):
        cand = stem_dir / f"{stem}{ext}"
        if cand.exists():
            stem_path = cand
            break
    if stem_path is None:
        raise HTTPException(status_code=404, detail=f"stem audio not found for '{track_id}/{stem}'")

    proc_path = _PROCESSED_DIR / f"{track_id}.json"
    bpm = 128.0
    if proc_path.exists():
        try:
            with open(proc_path, "r") as f:
                bpm = float(json.load(f).get("bpm") or 128.0)
        except Exception:
            bpm = 128.0

    bar_sec = 240.0 / max(40.0, bpm)
    start_sec = max(0.0, float(start_bar) * bar_sec)
    dur_sec = max(0.05, min(32.0, float(bars) * bar_sec))
    info = sf.info(str(stem_path))
    sr = int(info.samplerate)
    start_frame = min(max(0, int(start_sec * sr)), max(0, info.frames - 1))
    stop_frame = min(info.frames, start_frame + max(1, int(dur_sec * sr)))
    audio, _ = sf.read(str(stem_path), start=start_frame, stop=stop_frame, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


# Serve the built frontend (frontend/dist) when present so the UI and the
# /ws/stream endpoint share one origin. Mounted LAST so /health, /api/*,
# /audio/* and /ws/* keep priority.
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")
