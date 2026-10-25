"""Phase 2A: multi-stem metrical-harmonic graph generator ("WubBrain-lite").

4 parallel playback heads (one per available stem), synchronized to a master
tatum clock. Transition cost per spec:

  D_m(i,j) = w_timbre * ||timbre_{i+1} - timbre_j||
           + w_pitch  * ||chroma_{i+1}  - chroma_j||
           + w_harmony* ||chroma_j - chroma_active_bass||

- Horizontal continuity: candidate j should sound like the natural successor
  of the current head position i (timbre + chroma closeness).
- Vertical harmony: melodic candidates should agree with the currently active
  bass chroma (key lock across songs).
- `chaos[m]` per stem: P(jump to best-matching slice elsewhere) vs continuing
  chronologically in the same track. chaos=0 -> straight playback, 1 -> full mash.
- `force_tracks`: pin each stem's pool to a different source track for the
  Phase 2 verification mashup.

Works on mix-only corpora (single "other" stream) and full 4-stem corpora.
Rendering applies equal-power micro-crossfades (audio_utils) so output is
click-free per the Audio Continuity First guardrail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.audio_utils import equal_power_crossfade, stitch_with_crossfade
from wub_core.schema import STEMS, TrackAnalysis

SR = 44100


@dataclass
class SliceRef:
    stem: str
    track_id: str
    segment_id: str
    start: float
    duration: float
    timbre: np.ndarray  # 12
    chroma: np.ndarray  # 12
    loud_max: float
    track_pos: int  # chronological order within (track, stem)
    track_bpm: float = 120.0  # source tempo, for time-stretch ratios
    beat_pos: int = 0  # 0..3 within bar
    tatum_pos: int = 0  # 0..3 within beat
    tatum_in_bar: int = 0  # 0..15 within 4/4 bar
    spectral_centroid: float = 0.0  # physical brightness in Hz


@dataclass
class StemPool:
    stem: str
    slices: List[SliceRef] = field(default_factory=list)

    def array(self, attr: str) -> np.ndarray:
        return np.stack([getattr(s, attr) for s in self.slices]).astype(np.float32)

    def by_track(self, track_id: str) -> List[int]:
        return [k for k, s in enumerate(self.slices) if s.track_id == track_id]


def build_pools(analyses: List[TrackAnalysis],
                exclude_silent: bool = True,
                max_duration: float = 16.0) -> Dict[str, StemPool]:
    pools: Dict[str, StemPool] = {}
    # Deterministic order: track, then time within (track, stem).
    for tr in sorted(analyses, key=lambda t: t.track_id):
        # Build segment metrical lookup from tatums
        seg_metrical: Dict[str, Tuple[int, int, int]] = {}
        for tat in getattr(tr, "tatums", []):
            subdiv = int((tat.beat_position * 4 + tat.tatum_position) % 16)
            for s_name, s_ids in getattr(tat, "stem_segments", {}).items():
                for sid in s_ids:
                    if sid not in seg_metrical:
                        seg_metrical[sid] = (tat.beat_position, tat.tatum_position, subdiv)

        tatum_sec = (60.0 / tr.bpm / 4.0) if (getattr(tr, "bpm", 0) and tr.bpm > 0) else 0.125
        per_stem: Dict[str, List] = {}
        for sg in sorted(tr.segments, key=lambda s: s.start):
            if exclude_silent and sg.is_silent:
                continue
            per_stem.setdefault(sg.stem, []).append(sg)
        for stem, segs in per_stem.items():
            pool = pools.setdefault(stem, StemPool(stem=stem))
            for pos, sg in enumerate(segs):
                if sg.duration > max_duration:
                    continue  # pre-cap corpora: unusable as atoms, toxic to render
                beat_p, tat_p, subdiv = seg_metrical.get(
                    sg.id,
                    (
                        int(round(sg.start / (tatum_sec * 4.0))) % 4,
                        int(round(sg.start / tatum_sec)) % 4,
                        int(round(sg.start / tatum_sec)) % 16,
                    )
                )
                pool.slices.append(SliceRef(
                    stem=stem, track_id=tr.track_id, segment_id=sg.id,
                    start=sg.start, duration=sg.duration,
                    timbre=np.asarray(sg.timbre, dtype=np.float32),
                    chroma=np.asarray(sg.pitches, dtype=np.float32),
                    loud_max=sg.loudness_max, track_pos=pos,
                    track_bpm=float(tr.bpm),
                    beat_pos=beat_p, tatum_pos=tat_p, tatum_in_bar=subdiv,
                    spectral_centroid=float(getattr(sg, "spectral_centroid", 0.0))))
    return pools


def transition_cost(current: SliceRef, successor_hint: Optional[SliceRef],
                    cand_timbre: np.ndarray, cand_chroma: np.ndarray,
                    active_bass_chroma: Optional[np.ndarray],
                    w_timbre: float = 1.0, w_pitch: float = 1.5,
                    w_harmony: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized D_m over all candidates. Returns (total, horiz, harm)."""
    if successor_hint is not None:
        dt = cand_timbre - successor_hint.timbre[None, :]
        dp = cand_chroma - successor_hint.chroma[None, :]
    else:
        dt = cand_timbre - current.timbre[None, :]
        dp = cand_chroma - current.chroma[None, :]
    horiz = w_timbre * np.linalg.norm(dt, axis=1) + w_pitch * np.linalg.norm(dp, axis=1)
    if active_bass_chroma is not None:
        harm = w_harmony * np.linalg.norm(cand_chroma - active_bass_chroma[None, :], axis=1)
    else:
        harm = np.zeros(len(cand_timbre), dtype=np.float32)
    return (horiz + harm).astype(np.float32), horiz, harm


def chroma_match_shift(cand: np.ndarray, target: np.ndarray) -> int:
    """Best circular pitch shift in semitones (-6..+6) aligning cand to target."""
    best_k, best_s = 0, -np.inf
    for k in range(-6, 7):
        s = float(np.dot(np.roll(cand, k), target))
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def generate(pools: Dict[str, StemPool],
             steps: int = 256,
             chaos: Optional[Dict[str, float]] = None,
             force_tracks: Optional[Dict[str, str]] = None,
             candidate_sample: int = 512,
             w_timbre: float = 1.0, w_pitch: float = 1.5, w_harmony: float = 1.0,
             seed: int = 0,
             initial_heads: Optional[Dict[str, int]] = None,
             min_dwell: int = 8) -> Dict[str, List[SliceRef]]:
    """Generate `steps` slice choices per stem. Returns {stem: [SliceRef...]}."""
    rng = np.random.default_rng(seed)
    chaos = chaos or {}
    seq: Dict[str, List[SliceRef]] = {}
    # Candidate index arrays per stem (optionally restricted to forced track)
    cand_idx: Dict[str, np.ndarray] = {}
    for stem, pool in pools.items():
        if force_tracks and stem in force_tracks:
            idx = pool.by_track(force_tracks[stem])
            if not idx:
                raise ValueError(f"force_tracks: no slices for {stem} in {force_tracks[stem]}")
            cand_idx[stem] = np.asarray(idx)
        else:
            cand_idx[stem] = np.arange(len(pool.slices))
    # Init heads
    if initial_heads is not None:
        head: Dict[str, int] = {s: int(initial_heads[s]) for s in pools if s in initial_heads}
        for s in pools:
            if s not in head:
                head[s] = int(cand_idx[s][0])
    else:
        head = {s: int(cand_idx[s][0]) for s in pools}
    dwell: Dict[str, int] = {s: 0 for s in pools}
    # Pre-stack features
    feat: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        s: (pools[s].array("timbre"), pools[s].array("chroma")) for s in pools
    }
    for _ in range(steps):
        # Active bass chroma = chroma of bass head (or None for mix-only)
        bass_chroma = None
        if "bass" in pools:
            bass_chroma = pools["bass"].slices[head["bass"]].chroma
        for stem, pool in pools.items():
            cur = pool.slices[head[stem]]
            ch = float(chaos.get(stem, 0.5))
            # chronological successor within same track
            succ = None
            same = [k for k in range(len(pool.slices))
                    if pool.slices[k].track_id == cur.track_id
                    and pool.slices[k].track_pos == cur.track_pos + 1]
            if same:
                succ = pool.slices[same[0]]

            # Jumps:
            # - If chaos == 0.0: NEVER jump as long as track has slices; loop to start if it ends.
            # - Allow jumps when dwell >= min_dwell (or lower for very high chaos) or when track ends.
            eff_dwell = max(2, min_dwell if ch < 0.8 else min_dwell // 2)
            want_jump = (ch > 0.0) and ((dwell[stem] >= eff_dwell and rng.random() < ch) or succ is None)

            if not want_jump and succ is not None:
                head[stem] = same[0]
                dwell[stem] += 1
            elif not want_jump:
                # Track ran out but chaos says continue: loop to its start.
                # chaos=0 therefore means straight looping playback.
                loop = [k for k in range(len(pool.slices))
                        if pool.slices[k].track_id == cur.track_id
                        and pool.slices[k].track_pos == 0]
                head[stem] = loop[0] if loop else head[stem]
                dwell[stem] = 0
            else:
                dwell[stem] = 0
                idx = cand_idx[stem]
                if len(idx) > candidate_sample:
                    idx = rng.choice(idx, size=candidate_sample, replace=False)
                # No self-loops and no chronological successor
                if len(idx) > 1:
                    idx = idx[idx != head[stem]]
                if succ is not None:
                    succ_idx = same[0]
                    idx = idx[idx != succ_idx]
                if len(idx) == 0:
                    head[stem] = same[0] if succ is not None else head[stem]
                    seq.setdefault(stem, []).append(pool.slices[head[stem]])
                    continue

                # Target metrical position to preserve groove
                target_subdiv = (cur.tatum_in_bar + 1) % 16
                cand_subdivs = np.array([pool.slices[i].tatum_in_bar for i in idx])

                exact_metric = idx[cand_subdivs == target_subdiv]
                beat_metric = idx[cand_subdivs % 4 == target_subdiv % 4]
                if len(exact_metric) >= 4:
                    eval_idx = exact_metric
                elif len(beat_metric) >= 4:
                    eval_idx = beat_metric
                else:
                    eval_idx = idx

                T, C = feat[stem]
                ct, cc = T[eval_idx], C[eval_idx]
                harm_ref = bass_chroma if stem != "bass" else None
                total, _, _ = transition_cost(cur, succ, ct, cc, harm_ref,
                                              w_timbre, w_pitch, w_harmony)

                temp = max(0.25 * float(total.std()), 1e-4)
                w = np.exp(-(total - total.min()) / temp).astype(np.float64)
                w = w / w.sum()
                head[stem] = int(rng.choice(eval_idx, p=w))
            seq.setdefault(stem, []).append(pool.slices[head[stem]])
    return seq


def _load_slice(audio_path: Path, start: float, duration: float,
                sr: int = SR) -> np.ndarray:
    import soundfile as sf

    info = sf.info(str(audio_path))
    src_sr = info.samplerate
    frames = int(duration * src_sr)
    got, _ = sf.read(str(audio_path), start=int(start * src_sr), frames=frames,
                     always_2d=True, dtype="float32")
    y = got.mean(axis=1)
    if src_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=src_sr, target_sr=sr)
    return np.asarray(y, dtype=np.float32)


def render_mashup(seq: Dict[str, List[SliceRef]],
                  audio_for: Dict[Tuple[str, str], Path],
                  fallback_audio: Dict[str, Path],
                  out_path: Path,
                  sr: int = SR,
                  pitch_lock: bool = False,
                  fade_ms: float = 3.0,
                  target_bpm: Optional[float] = None) -> Path:
    """Stitch each stem stream with micro-crossfades, sum to mix, write WAV.

    target_bpm enables phase-vocoder time-stretch per slice so sources at
    different tempi lock to one grid (pitch preserved).
    """
    import soundfile as sf

    import librosa

    stem_mixes: List[np.ndarray] = []
    fade = int(sr * fade_ms / 1000.0)
    skipped_long = 0
    tatum_sec = (60.0 / target_bpm / 4.0) if (target_bpm and target_bpm > 0) else 0.25
    for stem, refs in seq.items():
        acc: np.ndarray | None = None
        for n, r in enumerate(refs):
            if r is None:
                y = np.zeros(int(sr * tatum_sec), dtype=np.float32)
                acc = y if acc is None else equal_power_crossfade(acc, y, fade)
                del y
                continue
            # Safety: never materialize pathological slices (older corpora
            # may predate the analyzer's MAX_SEGMENT_SEC cap).
            if r.duration > 16.0:
                skipped_long += 1
                continue
            p = audio_for.get((r.track_id, stem), fallback_audio.get(r.track_id))
            if p is None or not Path(p).exists():
                continue
            y = _load_slice(Path(p), r.start, r.duration, sr)
            # Tempo lock: phase-vocoder stretch to the target grid.
            if target_bpm and r.track_bpm > 0 and len(y) > 2048:
                rate = float(target_bpm) / float(r.track_bpm)
                if 0.5 <= rate <= 2.0 and abs(rate - 1.0) > 0.02:
                    try:
                        y = librosa.effects.time_stretch(y, rate=rate)
                        y = np.asarray(y, dtype=np.float32)
                    except Exception:
                        pass
            # Pitch-shifting sub-100ms grains is inaudible; multi-second
            # evolving slices shift badly and blow RAM in the phase vocoder.
            # Only lock the musical middle range.
            if (pitch_lock and stem in ("bass", "vocals", "other")
                    and 4096 < len(y) < 8 * sr):
                shift = 0
                try:
                    c = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
                    if c.max() > 1e-6:
                        c = c / c.max()
                        shift = -(((int(np.argmax(c)) + 6) % 12) - 6)
                        shift = max(-6, min(6, shift))
                except Exception:
                    shift = 0
                if shift:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(shift))
                    y = np.asarray(y, dtype=np.float32)
            # Fold incrementally: O(1) slice memory instead of holding all
            # chunks (OOM-killer on 5 GB boxes with 1200+ slices).
            acc = y if acc is None else equal_power_crossfade(acc, y, fade)
            del y
            if n % 200 == 0:
                import gc

                gc.collect()
        if acc is not None:
            stem_mixes.append(acc)
    if not stem_mixes:
        raise RuntimeError("render_mashup: no audio resolved; check audio_for paths")
    if skipped_long:
        print(f"render_mashup: skipped {skipped_long} over-long (>16s) slices")
    n = max(len(m) for m in stem_mixes)
    mix = np.zeros(n, dtype=np.float64)
    for m in stem_mixes:
        mix[: len(m)] += m / max(len(stem_mixes), 1)
    peak = np.max(np.abs(mix)) + 1e-9
    mix = (mix / peak * 0.89).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.stack([mix, mix], axis=1), sr)
    return out_path
