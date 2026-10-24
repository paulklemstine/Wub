"""Wub backend: REST (tracks/stats/umap/audio/mashup) + WS 4-stem streamer.

All reads go through CorpusStore, which refreshes from disk on every
request — the API always reflects whatever has been ingested so far.
Run:  uvicorn server.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from server.stream_manager import SliceCache, StreamSession, render_slice_wav
from wub_core.corpus import CorpusStore

STORE = CorpusStore(processed_dir=Path("data/processed"),
                    stems_base=Path("data/stems"),
                    library_dirs=[Path("pandora")])
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
    n = STORE.refresh()
    print(f"Wub store: {len(STORE.analyses)} tracks ({n} loaded at startup)")


@app.get("/health")
def health() -> dict:
    STORE.refresh()
    return {"status": "ok", **STORE.summary()}


@app.get("/api/stats")
def stats() -> dict:
    STORE.refresh()
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


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    STORE.refresh()
    session = StreamSession(store=STORE)
    await ws.send_json({"type": "hello",
                        "stems": [s for s in ("drums", "bass", "vocals", "other")],
                        "bpm": session.bpm, "sample_rate": 44100})
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


# Serve the built frontend (frontend/dist) when present so the UI and the
# /ws/stream endpoint share one origin. Mounted LAST so /health, /api/*,
# /audio/* and /ws/* keep priority.
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")
