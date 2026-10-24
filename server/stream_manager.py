"""Multi-stem lookahead streamer: chunked Markov generation -> WS tatum events.

Each session holds per-stem chaos settings and a target BPM. Audio is NOT
pushed; events reference GET /audio/{track_id}/{segment_id}.wav which the
browser fetches, decodes, and caches (LRU-300). The frontend schedules on
its own AudioContext clock, so server timing only needs ~50ms accuracy.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from wub_core.corpus import CorpusStore
from wub_core.schema import TrackAnalysis

STEMS = ("drums", "bass", "vocals", "other")
SLICE_CACHE_CAP = 2000

PITCH_NAMES = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
               'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
               'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def get_key_semitones(src_key: str, target_key: str = "C major") -> int:
    try:
        s = src_key.strip().split()[0]
        t = target_key.strip().split()[0]
        s_idx = PITCH_NAMES.get(s, 0)
        t_idx = PITCH_NAMES.get(t, 0)
        return ((t_idx - s_idx + 6) % 12) - 6
    except Exception:
        return 0


@dataclass
class StemHeadState:
    track_id: str
    tatum_offset: int = 0
    active_seg_id: Optional[str] = None
    dwell_tatums: int = 0
    phrase_tatums: int = 32
    silent_tatums: int = 0


def find_active_tatums(tr: TrackAnalysis, stem: str, bar_phase: Optional[int] = None) -> List[int]:
    """Return tatum indices where `stem` has audible (non-silent) acoustic slices."""
    if not tr.tatums:
        return [0]
    segs_by_id = {s.id: s for s in tr.segments if s.stem == stem}
    active: List[int] = []
    for tat in tr.tatums:
        if bar_phase is not None:
            if (tat.beat_position * 4 + tat.tatum_position) % 16 != bar_phase:
                continue
        sids = tat.stem_segments.get(stem, [])
        for sid in sids:
            seg = segs_by_id.get(sid)
            if seg is not None and not seg.is_silent and seg.loudness_max > -32.0:
                active.append(tat.index)
                break
    if not active:
        if bar_phase is not None:
            active = [tat.index for tat in tr.tatums
                      if (tat.beat_position * 4 + tat.tatum_position) % 16 == bar_phase]
        else:
            active = [tat.index for tat in tr.tatums]
    return active or [0]


_WUBGPT_CACHE: Dict[str, Any] = {}


def get_cached_wubgpt() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if "model" in _WUBGPT_CACHE:
        return _WUBGPT_CACHE["model"], _WUBGPT_CACHE.get("vqs")
    if _WUBGPT_CACHE.get("checked", False):
        return None, None

    _WUBGPT_CACHE["checked"] = True
    candidates = [
        Path("data/wubgpt/wubgpt.pt"),
        Path("data/wubgpt.pt"),
    ]
    for p in candidates:
        if p.exists():
            try:
                from wub_core.model_transformer import load_checkpoint

                m, v, _ = load_checkpoint(p)
                m.eval()
                _WUBGPT_CACHE["model"] = m
                _WUBGPT_CACHE["vqs"] = v
                print(f"[WubGPT] Active live transformer model loaded from {p}")
                return m, v
            except Exception as e:
                print(f"[WubGPT] Could not load checkpoint from {p}: {e}")
    return None, None


_GLOBAL_PATTERN_BANK: Optional[PatternBank] = None


def get_cached_pattern_bank(store: CorpusStore) -> PatternBank:
    global _GLOBAL_PATTERN_BANK
    if _GLOBAL_PATTERN_BANK is not None and len(_GLOBAL_PATTERN_BANK.patterns_by_stem.get("drums", [])) > 0:
        return _GLOBAL_PATTERN_BANK

    from wub_core.patterns import PatternBank

    bank = PatternBank()
    multistem = [t for t in store.tracks if len(t.stems_available) == 4]
    target_tracks = multistem if multistem else store.tracks
    count = bank.load_from_tracks(target_tracks)
    print(f"[Groovebox] Loaded {count} sequencer patterns into pattern memory bank.")
    _GLOBAL_PATTERN_BANK = bank
    return bank


@dataclass
class StreamSession:
    store: CorpusStore
    chaos: Dict[str, float] = field(default_factory=lambda: {
        "drums": 0.1,   # Rhythmic backbone: steady 2-4 bar loops
        "bass": 0.3,    # Harmonic foundation: 2-4 bar phrases
        "vocals": 0.5,  # Vocal phrases
        "other": 0.5,   # Synths and hooks
    })
    synth_blend: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STEMS})
    bpm: float = 126.0
    seed: int = 0
    pitch_lock: bool = True
    key: str = "C major"
    gravity: Dict[str, list] = field(default_factory=dict)
    temperature: float = 0.75
    top_p: float = 0.85
    top_k: int = 32
    use_wubgpt: bool = True
    use_groovebox: bool = True
    _conductor: Optional[Any] = None
    _track_centers: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    _buffer: List[dict] = field(default_factory=list)
    _tatum_index: int = 0
    _rng_seed: int = 0
    _heads: Dict[str, StemHeadState] = field(default_factory=dict)

    def get_conductor(self) -> Any:
        if self._conductor is None:
            from wub_core.groovebox import GrooveboxConductor
            bank = get_cached_pattern_bank(self.store)
            self._conductor = GrooveboxConductor(bank, seed=int(self.seed + self._rng_seed))
        return self._conductor

    def _get_track_center(self, stem: str, track_id: str) -> Optional[Tuple[float, float]]:
        if not self._track_centers:
            idx_dir = Path("data/indices")
            for s in STEMS:
                f = idx_dir / f"umap_{s}.json"
                if f.exists():
                    try:
                        pts = json.loads(f.read_text())
                        accum: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
                        for p in pts:
                            tid = p.get("track_id")
                            if tid and "x" in p and "y" in p:
                                accum[tid].append((float(p["x"]), float(p["y"])))
                        self._track_centers[s] = {
                            tid: (float(np.mean([pt[0] for pt in pts_list])),
                                  float(np.mean([pt[1] for pt in pts_list])))
                            for tid, pts_list in accum.items()
                        }
                    except Exception:
                        pass
        return self._track_centers.get(stem, {}).get(track_id)

    def _pick_candidate_track(self, stem: str, candidates: List[TrackAnalysis], rng: np.random.Generator) -> TrackAnalysis:
        if len(candidates) <= 1:
            return candidates[0]
        grav = self.gravity.get(stem)
        if not grav or len(grav) < 2:
            return candidates[int(rng.choice(len(candidates)))]
        try:
            gx, gy = float(grav[0]), float(grav[1])
        except (ValueError, TypeError):
            return candidates[int(rng.choice(len(candidates)))]
        weights = []
        for tr in candidates:
            c = self._get_track_center(stem, tr.track_id)
            dist = np.hypot(c[0] - gx, c[1] - gy) if c else 0.45
            weights.append(float(np.exp(-12.0 * (dist ** 2))))
        tot = sum(weights)
        if tot <= 1e-6:
            return candidates[int(rng.choice(len(candidates)))]
        probs = [w / tot for w in weights]
        return candidates[int(rng.choice(len(candidates), p=probs))]

    def configure(self, stem: Optional[str] = None, chaos: Optional[float] = None,
                  bpm: Optional[float] = None, synth_blend: Optional[float] = None) -> None:
        if stem and chaos is not None and stem in self.chaos:
            self.chaos[stem] = max(0.0, min(1.0, float(chaos)))
        if stem and synth_blend is not None and stem in self.synth_blend:
            self.synth_blend[stem] = max(0.0, min(1.0, float(synth_blend)))
        if bpm:
            new_bpm = max(60.0, min(180.0, float(bpm)))
            if abs(new_bpm - self.bpm) > 0.05:
                self.bpm = new_bpm
                self._buffer.clear()

    @property
    def tatum_duration(self) -> float:
        return 60.0 / self.bpm / 4.0

    def _refill(self) -> None:
        self.store.refresh()
        all_tracks = self.store.tracks
        if not all_tracks:
            return

        tracks_by_id: Dict[str, TrackAnalysis] = {t.track_id: t for t in all_tracks}
        multistem = [t for t in all_tracks if len(t.stems_available) == 4]
        pool_tracks = multistem if multistem else all_tracks

        rng = np.random.default_rng(self.seed + self._rng_seed)
        self._rng_seed += 1

        # 1. Groovebox Conductor (Macro Arrangement & 16-Step Pattern Sequencer)
        if self.use_groovebox:
            try:
                conductor = self.get_conductor()
                if conductor.bank.patterns_by_stem.get("drums"):
                    for _ in range(32):
                        master_tatum = self._tatum_index
                        raw_slices, meta = conductor.next_step()
                        slices: Dict[str, Optional[dict]] = {}

                        for stem in STEMS:
                            sref = raw_slices.get(stem)
                            if sref is not None:
                                tr = tracks_by_id.get(sref.track_id)
                                src_bpm = float(tr.bpm) if (tr and tr.bpm > 0) else self.bpm
                                rate = max(0.5, min(2.0, float(self.bpm) / float(src_bpm)))
                                rate_cents = 1200.0 * np.log2(rate) if rate > 0 else 0.0

                                detune_cents = 0
                                if stem in ("bass", "vocals", "other"):
                                    if self.pitch_lock and tr:
                                        shift = get_key_semitones(tr.key, self.key)
                                        detune_cents = int(shift * 100 - rate_cents)
                                    else:
                                        detune_cents = int(-rate_cents)
                                else:
                                    detune_cents = int(-rate_cents)

                                slices[stem] = {
                                    "track_id": sref.track_id,
                                    "segment_id": sref.segment_id,
                                    "start": sref.start,
                                    "duration": sref.duration,
                                    "url": f"/audio/{sref.track_id}/{sref.segment_id}.wav",
                                    "detune_cents": detune_cents,
                                    "rate": rate,
                                    "synth_blend": self.synth_blend.get(stem, 0.0),
                                }
                            else:
                                slices[stem] = None

                        self._buffer.append({
                            "type": "tatum",
                            "tatum_index": master_tatum,
                            "bpm": self.bpm,
                            "tatum_duration": self.tatum_duration,
                            "slices": slices,
                            "groovebox": meta,
                        })
                        self._tatum_index += 1
                    return
            except Exception as e:
                print(f"[Groovebox] Fallback on refill error: {e}")

        # 2. If WubGPT model checkpoint exists, generate using Top-P nucleus sampling
        if self.use_wubgpt:
            model, vqs = get_cached_wubgpt()
            if model is not None and vqs is not None:
                try:
                    from wub_core.model_transformer import sample as wub_sample, tokens_to_slices
                    pools = self.store.pools(exclude_silent=True)
                    if pools:
                        cur_phase = self._tatum_index % 16
                        seed_steps = [{"r": (cur_phase + i) % 16, **{s: 0 for s in STEMS}} for i in range(16)]
                        sampled = wub_sample(
                            model=model,
                            vqs=vqs,
                            seed_steps=seed_steps,
                            steps=32,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            seed=int(self.seed + self._rng_seed),
                        )
                        seq = tokens_to_slices(sampled, pools, vqs)
                        for step_idx in range(len(sampled)):
                            master_tatum = self._tatum_index
                            slices: Dict[str, Optional[dict]] = {}
                            for stem in STEMS:
                                sref_list = seq.get(stem, [])
                                sref = sref_list[step_idx] if step_idx < len(sref_list) else None
                                if sref is not None:
                                    tr = tracks_by_id.get(sref.track_id)
                                    src_bpm = float(tr.bpm) if (tr and tr.bpm > 0) else self.bpm
                                    rate = max(0.5, min(2.0, float(self.bpm) / float(src_bpm)))
                                    rate_cents = 1200.0 * np.log2(rate) if rate > 0 else 0.0
                                    detune_cents = 0
                                    if stem in ("bass", "vocals", "other") and self.pitch_lock and tr:
                                        shift = get_key_semitones(tr.key, self.key)
                                        detune_cents = int(shift * 100 - rate_cents)
                                    else:
                                        detune_cents = int(-rate_cents)

                                    slices[stem] = {
                                        "track_id": sref.track_id,
                                        "segment_id": sref.segment_id,
                                        "start": sref.start,
                                        "duration": sref.duration,
                                        "url": f"/audio/{sref.track_id}/{sref.segment_id}.wav",
                                        "detune_cents": detune_cents,
                                        "rate": rate,
                                        "synth_blend": self.synth_blend.get(stem, 0.0),
                                    }
                                else:
                                    slices[stem] = None

                            self._buffer.append({
                                "type": "tatum",
                                "tatum_index": master_tatum,
                                "bpm": self.bpm,
                                "tatum_duration": self.tatum_duration,
                                "slices": slices,
                            })
                            self._tatum_index += 1
                        return
                except Exception:
                    pass

        # 2. Phrase-locked metrical lookahead generator with gravity timbre steering
        # Initialize heads if empty — lock to active downbeats so all stems start sounding immediately
        for stem in STEMS:
            if stem not in self._heads:
                cands = [t for t in pool_tracks if stem in t.stems_available]
                if not cands:
                    cands = [t for t in all_tracks if stem in t.stems_available]
                if not cands:
                    continue
                if stem == "bass" and "drums" in self._heads and rng.random() < 0.7:
                    drums_tid = self._heads["drums"].track_id
                    chosen_t = tracks_by_id.get(drums_tid) or self._pick_candidate_track(stem, cands, rng)
                else:
                    chosen_t = self._pick_candidate_track(stem, cands, rng)
                active_dbs = find_active_tatums(chosen_t, stem, bar_phase=0)
                init_tat = int(rng.choice(active_dbs[:8])) if active_dbs else 0
                phrase_len = int(rng.choice([64, 128])) if stem in ("drums", "bass") else int(rng.choice([32, 64]))
                self._heads[stem] = StemHeadState(
                    track_id=chosen_t.track_id,
                    tatum_offset=init_tat,
                    active_seg_id=None,
                    dwell_tatums=0,
                    phrase_tatums=phrase_len,
                    silent_tatums=0,
                )

        # Generate next 32 tatums (2 full bars)
        for _ in range(32):
            master_tatum = self._tatum_index
            bar_phase = master_tatum % 16
            slices: Dict[str, Optional[dict]] = {}

            for stem in STEMS:
                if stem not in self._heads:
                    slices[stem] = None
                    continue

                head = self._heads[stem]
                tr = tracks_by_id.get(head.track_id)
                cands = [t for t in pool_tracks if stem in t.stems_available]
                if not cands:
                    cands = [t for t in all_tracks if stem in t.stems_available]
                if not cands:
                    slices[stem] = None
                    continue

                if tr is None or stem not in tr.stems_available:
                    tr = cands[int(rng.choice(len(cands)))]
                    head.track_id = tr.track_id
                    head.tatum_offset = 0
                    head.active_seg_id = None
                    head.dwell_tatums = 0

                ch = float(self.chaos.get(stem, 0.5))

                # Check transition condition (phrase & bar quantized)
                track_ended = bool(tr.tatums and head.tatum_offset >= len(tr.tatums))
                can_jump = False
                if track_ended:
                    can_jump = True
                elif ch > 0.0 and head.dwell_tatums >= head.phrase_tatums:
                    if stem == "drums" and bar_phase == 0:
                        can_jump = rng.random() < ch
                    elif stem == "bass" and bar_phase in (0, 8):
                        can_jump = rng.random() < ch
                    elif stem in ("vocals", "other") and bar_phase in (0, 4, 8, 12):
                        can_jump = rng.random() < ch

                if can_jump:
                    if ch == 0.0 or len(cands) == 1:
                        next_tr = tr
                    elif stem == "bass" and "drums" in self._heads and rng.random() < 0.6:
                        drums_tr = tracks_by_id.get(self._heads["drums"].track_id)
                        next_tr = drums_tr if (drums_tr and "bass" in drums_tr.stems_available) else self._pick_candidate_track(stem, cands, rng)
                    else:
                        next_tr = self._pick_candidate_track(stem, cands, rng)

                    head.track_id = next_tr.track_id
                    head.dwell_tatums = 0
                    head.phrase_tatums = int(rng.choice([64, 128])) if stem in ("drums", "bass") else int(rng.choice([32, 64]))
                    head.active_seg_id = None
                    tr = next_tr

                    # Match metrical bar phase in an active musical section
                    active_matches = find_active_tatums(tr, stem, bar_phase=bar_phase)
                    head.tatum_offset = int(rng.choice(active_matches[:8])) if active_matches else 0

                # Resolve slice at current tatum offset
                cur_seg_id: Optional[str] = None
                cur_seg = None
                if tr.tatums and head.tatum_offset < len(tr.tatums):
                    tat = tr.tatums[head.tatum_offset]
                    stem_segs = tat.stem_segments.get(stem, [])
                    if stem_segs:
                        cur_seg_id = stem_segs[-1]
                else:
                    stem_segs_all = [s for s in tr.segments if s.stem == stem]
                    if stem_segs_all:
                        cur_seg_id = stem_segs_all[head.tatum_offset % len(stem_segs_all)].id

                if cur_seg_id is not None:
                    cur_seg = next((s for s in tr.segments if s.id == cur_seg_id), None)

                # Silence recovery: if stem is silent for >8 tatums, jump to active section
                if cur_seg is None or cur_seg.is_silent or cur_seg.loudness_max < -32.0:
                    head.silent_tatums += 1
                    if head.silent_tatums >= 8:
                        active_matches = find_active_tatums(tr, stem, bar_phase=bar_phase)
                        head.tatum_offset = int(rng.choice(active_matches[:8])) if active_matches else 0
                        head.silent_tatums = 0
                else:
                    head.silent_tatums = 0

                # Onset vs sustain
                is_first_tatum = (self._tatum_index == 0)
                is_onset = (cur_seg_id is not None) and (cur_seg_id != head.active_seg_id or is_first_tatum)

                if is_onset and cur_seg is not None and not cur_seg.is_silent:
                    head.active_seg_id = cur_seg_id
                    src_bpm = float(tr.bpm) if tr.bpm > 0 else self.bpm
                    rate = float(self.bpm) / float(src_bpm)
                    rate = max(0.5, min(2.0, rate))
                    rate_cents = 1200.0 * np.log2(rate) if rate > 0 else 0.0

                    detune_cents = 0
                    if stem in ("bass", "vocals", "other"):
                        if self.pitch_lock:
                            shift = get_key_semitones(tr.key, self.key)
                            detune_cents = int(shift * 100 - rate_cents)
                        else:
                            detune_cents = int(-rate_cents)
                    else:
                        # Drums: preserve natural punch and kick tuning
                        detune_cents = int(-rate_cents)

                    slices[stem] = {
                        "track_id": tr.track_id,
                        "segment_id": cur_seg_id,
                        "start": cur_seg.start,
                        "duration": cur_seg.duration,
                        "url": f"/audio/{tr.track_id}/{cur_seg_id}.wav",
                        "detune_cents": detune_cents,
                        "rate": rate,
                        "synth_blend": self.synth_blend.get(stem, 0.0),
                    }
                else:
                    slices[stem] = None
                    if cur_seg_id is None:
                        head.active_seg_id = None

                head.tatum_offset += 1
                head.dwell_tatums += 1

            self._buffer.append({
                "type": "tatum",
                "tatum_index": self._tatum_index,
                "tatum_duration": self.tatum_duration,
                "start_time": 0.06,
                "bpm": self.bpm,
                "slices": slices,
            })
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
