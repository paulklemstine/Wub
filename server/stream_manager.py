"""Multi-stem lookahead streamer: chunked Markov generation -> WS tatum events.

Each session holds per-stem chaos settings and a target BPM. Audio is NOT
pushed; events reference GET /audio/{track_id}/{segment_id}.wav which the
browser fetches, decodes, and caches (LRU-300). The frontend schedules on
its own AudioContext clock, so server timing only needs ~50ms accuracy.
"""
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from wub_core.corpus import CorpusStore

STEMS = ("drums", "bass", "vocals", "other")
SLICE_CACHE_CAP = 2000


@dataclass
class StreamSession:
    store: CorpusStore
    chaos: Dict[str, float] = field(default_factory=lambda: {s: 0.6 for s in STEMS})
    synth_blend: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STEMS})
    bpm: float = 128.0
    seed: int = 0
    pitch_lock: bool = False
    key: str = "C major"
    gravity: Dict[str, list] = field(default_factory=dict)
    _buffer: List[dict] = field(default_factory=list)
    _tatum_index: int = 0
    _rng_seed: int = 0

    def configure(self, stem: Optional[str] = None, chaos: Optional[float] = None,
                  bpm: Optional[float] = None, synth_blend: Optional[float] = None) -> None:
        if stem and chaos is not None and stem in self.chaos:
            self.chaos[stem] = max(0.0, min(1.0, float(chaos)))
        if stem and synth_blend is not None and stem in self.synth_blend:
            self.synth_blend[stem] = max(0.0, min(1.0, float(synth_blend)))
        if bpm:
            self.bpm = max(80.0, min(175.0, float(bpm)))

    @property
    def tatum_duration(self) -> float:
        return 60.0 / self.bpm / 4.0

    def _refill(self) -> None:
        from wub_core import model_markov as mm

        self.store.refresh()
        multistem = [t for t in self.store.tracks if len(t.stems_available) == 4]
        pools = mm.build_pools(multistem if multistem else self.store.tracks)
        if not pools:
            return
        seq = mm.generate(pools, steps=32,
                          chaos={s: self.chaos.get(s, 0.6) for s in pools},
                          seed=self.seed + self._rng_seed)
        self._rng_seed += 1
        bpms = {t.track_id: t.bpm for t in self.store.tracks}
        for k in range(32):
            slices: Dict[str, Optional[dict]] = {}
            for stem in STEMS:
                refs = seq.get(stem)
                r = refs[k] if refs and k < len(refs) else None
                if r is None:
                    slices[stem] = None
                else:
                    src_bpm = bpms.get(r.track_id, self.bpm) or self.bpm
                    rate = float(self.bpm) / float(src_bpm) if src_bpm > 0 else 1.0
                    rate = max(0.5, min(2.0, rate))

                    detune_cents = 0
                    if stem in ("bass", "vocals", "other") and hasattr(r, "chroma") and r.chroma is not None:
                        c = np.asarray(r.chroma)
                        if np.max(c) > 1e-6:
                            c = c / np.max(c)
                            shift = -(((int(np.argmax(c)) + 6) % 12) - 6)
                            detune_cents = int(max(-6, min(6, shift)) * 100)

                    slices[stem] = {
                        "track_id": r.track_id, "segment_id": r.segment_id,
                        "start": r.start, "duration": r.duration,
                        "url": f"/audio/{r.track_id}/{r.segment_id}.wav",
                        "detune_cents": detune_cents,
                        "rate": rate,
                        "synth_blend": self.synth_blend.get(stem, 0.0),
                    }
            self._buffer.append({"type": "tatum", "tatum_index": self._tatum_index,
                                 "tatum_duration": self.tatum_duration,
                                 "start_time": 0.06,
                                 "bpm": self.bpm, "slices": slices})
            self._tatum_index += 1

    def next_event(self) -> Optional[dict]:
        if len(self._buffer) < 8:
            self._refill()
        if not self._buffer:
            return None
        return self._buffer.pop(0)


class SliceCache:
    """In-memory WAV bytes for recently served slices."""

    def __init__(self, cap: int = SLICE_CACHE_CAP):
        self.cap = cap
        self._d: Dict[str, bytes] = {}
        self._order: List[str] = []

    def get(self, key: str) -> Optional[bytes]:
        return self._d.get(key)

    def put(self, key: str, val: bytes) -> None:
        if key in self._d:
            return
        self._d[key] = val
        self._order.append(key)
        while len(self._order) > self.cap:
            self._d.pop(self._order.pop(0), None)


def render_slice_wav(store: CorpusStore, track_id: str, segment_id: str,
                     sr: int = 44100) -> bytes:
    """Slice [start, start+duration] from the resolved source audio as WAV bytes."""
    tr = store.analyses.get(track_id)
    if tr is None:
        raise KeyError(f"unknown track {track_id}")
    seg = next((s for s in tr.segments if s.id == segment_id), None)
    if seg is None:
        raise KeyError(f"unknown segment {segment_id}")
    audio_for, fallback = store.audio_for({track_id})
    src = audio_for.get((track_id, seg.stem), fallback.get(track_id))
    if src is None:
        raise KeyError(f"no audio for {track_id}")
    info = sf.info(str(src))
    frames = max(1, int(seg.duration * info.samplerate))
    data, _ = sf.read(str(src), start=int(seg.start * info.samplerate),
                      frames=frames, always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(np.float32)
    if info.samplerate != sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=info.samplerate, target_sr=sr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.stack([mono, mono], axis=1), sr, format="WAV")
    return buf.getvalue()
