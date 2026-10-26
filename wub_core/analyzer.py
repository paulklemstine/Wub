"""Echo-Wub analyzer: beat/tatum grid + per-stem onset slicing + features.

DSP rules:
- Beat grid locked to isolated drums stem, fallback to full mix.
- Tatums: subdivide each beat interval into N=4 (16th-note grid).
- Per-stem onsets with librosa.onset.onset_detect(backtrack=True) so slice
  starts snap to preceding energy minima (click prevention).
- Features per segment: 12-D timbre (MFCC mean), 12-D chroma, loudness
  envelope, spectral centroid, zero-crossing offset, silence flag.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from wub_core.schema import Segment, Tatum, TrackAnalysis, STEMS

SR = 44100
TATUMS_PER_BEAT = 4
SILENCE_THRESH_DB = -60.0
MIN_SEGMENT_SEC = 0.020  # 20 ms floor to avoid pathological chatter
MAX_SEGMENT_SEC = 4.0  # force-split longer stretches: a minutes-long "slice"
# (silent verse, missing stem) is useless for concatenative lookup and blows
# up render memory (hours of audio per sequence).


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        return y.mean(axis=1)
    return y


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def _peak_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x)) + 1e-12)
    return 20.0 * np.log10(peak)


def estimate_key(chroma_mean: np.ndarray) -> str:
    """Krumhansl-Schmuckler key estimation from mean chroma (12-D)."""
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best_key, best_score = "C major", -np.inf
    for shift in range(12):
        for profile, mode in ((major_profile, "major"), (minor_profile, "minor")):
            rotated = np.roll(profile, shift)
            score = float(np.corrcoef(chroma_mean, rotated)[0, 1])
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_key = f"{names[shift]} {mode}"
    return best_key


def extract_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Return (beat_times_sec, bpm). Tries madmom RNN first, else librosa."""
    beats: np.ndarray | None = None
    bpm = 120.0
    # Try madmom if installed (better downbeat tracking on drums)
    try:
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor

        proc = RNNBeatProcessor()(y)
        # madmom returns activations; peak-pick via its own processor chain
        from madmom.features.beats import BeatTrackingProcessor

        beats = np.array(BeatTrackingProcessor(fps=100)(proc))
        tempo_proc = TempoEstimationProcessor(fps=100)
        tempi = tempo_proc(proc)
        if len(tempi):
            bpm = float(tempi[0][0])
    except Exception:
        beats = None
    if beats is None or len(beats) < 4:
        import librosa

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beats_arr = np.asarray(beat_frames, dtype=float)
        if isinstance(tempo, np.ndarray):
            bpm = float(np.mean(tempo)) if tempo.size else 120.0
        else:
            bpm = float(tempo)
        beats = beats_arr
    if beats is None or len(beats) < 2:
        # beatless ambient fallback: synthesize 120bpm grid over duration
        dur = len(y) / float(sr)
        step = 60.0 / 120.0
        beats = np.arange(0.0, dur, step)
        bpm = 120.0
    beats = np.sort(np.asarray(beats, dtype=float))
    if bpm <= 0 or not np.isfinite(bpm):
        # derive from median inter-beat interval
        ibi = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
        bpm = 60.0 / max(float(ibi), 1e-3)
    return beats, float(bpm)


def beats_to_tatums(
    beats: np.ndarray, duration: float, sr: int = SR
) -> List[Tuple[float, float, float]]:
    """Subdivide beats into 4 tatums each. Returns list of (start, duration, confidence)."""
    tatums: List[Tuple[float, float, float]] = []
    if len(beats) == 0:
        return tatums
    # Leading tatums before first beat (extrapolate backward)
    # Trailing tatums after last beat (extrapolate forward)
    ext = np.median(np.diff(beats)) if len(beats) > 1 else 0.5
    for k in range(len(beats)):
        b0 = float(beats[k])
        b1 = float(beats[k + 1]) if k + 1 < len(beats) else b0 + float(ext)
        if b1 <= b0:
            continue
        step = (b1 - b0) / TATUMS_PER_BEAT
        for m in range(TATUMS_PER_BEAT):
            s = b0 + m * step
            if s >= duration:
                break
            d = min(step, duration - s)
            if d <= 0:
                continue
            tatums.append((s, d, 0.94))
    # Ensure coverage from 0: prepend if first tatum starts late
    if tatums and tatums[0][0] > 1e-3:
        first = tatums[0][0]
        step = tatums[0][1]
        t = 0.0
        pre: List[Tuple[float, float, float]] = []
        while t < first - 1e-6:
            d = min(step, first - t)
            pre.append((t, d, 0.5))
            t += d
        tatums = pre + tatums
    return tatums


def nearest_zero_crossing_offset(y: np.ndarray, idx: int, sr: int, window_ms: float = 2.0) -> float:
    """Search ±window for nearest zero crossing; return offset in seconds."""
    w = int(sr * window_ms / 1000.0)
    lo = max(0, idx - w)
    hi = min(len(y), idx + w)
    if hi <= lo:
        return 0.0
    seg = y[lo:hi]
    signs = np.sign(seg)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) == 0:
        return 0.0
    center = idx - lo
    best = crossings[np.argmin(np.abs(crossings - center))]
    return float(abs((lo + int(best)) - idx)) / float(sr)


def detect_onsets(y: np.ndarray, sr: int, stem: str = "other", backtrack: bool = True) -> np.ndarray:
    import librosa

    # Stem-adaptive onset envelope and peak-picking
    if stem == "drums":
        # Full-bandwidth with tight windowing for crisp transient detection (kicks, snares, 16th hats)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=256,
            backtrack=backtrack,
            pre_max=3,
            post_max=3,
            pre_avg=8,
            post_avg=8,
            delta=0.20,
            wait=2,
            units="time",
        )
    elif stem == "bass":
        # Low-frequency band focus (30-500 Hz, n_mels=32) for sub bass and bassline note onsets
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=256, fmin=30, fmax=500, n_mels=32
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=256,
            backtrack=backtrack,
            pre_max=4,
            post_max=4,
            pre_avg=8,
            post_avg=8,
            delta=0.18,
            wait=3,
            units="time",
        )
    elif stem == "vocals":
        # Formant/vowel frequency focus (250-5000 Hz, n_mels=64) with sensitive thresholding for vocal attacks
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=256, fmin=250, fmax=5000, n_mels=64
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=256,
            backtrack=backtrack,
            pre_max=4,
            post_max=4,
            pre_avg=8,
            post_avg=8,
            delta=0.16,
            wait=3,
            units="time",
        )
    else:  # "other" / synth / master mix
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=256, fmin=100
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=256,
            backtrack=backtrack,
            pre_max=5,
            post_max=5,
            pre_avg=10,
            post_avg=10,
            delta=0.20,
            wait=3,
            units="time",
        )
    return np.asarray(onsets, dtype=float)


def segment_features(
    y_seg: np.ndarray,
    sr: int,
    y_full: np.ndarray,
    abs_start_sample: int,
) -> dict:
    import librosa

    if len(y_seg) < 64:
        pad = 64 - len(y_seg)
        y_seg = np.pad(y_seg, (0, pad))
    # Spectral features need sufficient context; pad short slices with
    # edge reflection so chroma/MFCC stay valid without biasing loudness
    # (loudness is computed on the true slice above/below).
    y_feat = y_seg
    if len(y_feat) < 4096:
        y_feat = np.pad(y_feat, (0, 4096 - len(y_feat)), mode="edge")
    # MFCC: n_mfcc=13 then drop 0th (energy) -> 12-D timbre
    mfcc = librosa.feature.mfcc(y=y_feat, sr=sr, n_mfcc=13, hop_length=512)
    timbre = mfcc[1:13].mean(axis=1) if mfcc.shape[0] >= 13 else mfcc.mean(axis=1)[:12]
    timbre = np.asarray(timbre, dtype=float).reshape(-1)[:12]
    if timbre.shape[0] < 12:
        timbre = np.pad(timbre, (0, 12 - timbre.shape[0]))
    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y_feat, sr=sr, hop_length=1024)
    pitches = np.asarray(chroma.mean(axis=1), dtype=float).reshape(-1)[:12]
    if pitches.shape[0] < 12:
        pitches = np.pad(pitches, (0, 12 - pitches.shape[0]))
    mx = float(np.max(pitches))
    if mx > 1e-9:
        pitches = pitches / mx
    pitches = np.clip(pitches, 0.0, 1.0)
    # Loudness
    head = y_seg[: min(2048, len(y_seg))]
    loud_start = _rms_db(head)
    loud_max = _peak_db(y_seg)
    amax = int(np.argmax(np.abs(y_seg)))
    loud_max_time = float(amax) / float(sr)
    centroid = librosa.feature.spectral_centroid(y=y_feat, sr=sr)
    spec_cent = float(np.mean(centroid))
    zc_off = nearest_zero_crossing_offset(y_full, abs_start_sample, sr)
    is_silent = bool(loud_max < SILENCE_THRESH_DB)
    return {
        "timbre": [float(x) for x in timbre],
        "pitches": [float(x) for x in pitches],
        "loudness_start": float(loud_start),
        "loudness_max": float(loud_max),
        "loudness_max_time": float(loud_max_time),
        "spectral_centroid": float(spec_cent),
        "zero_crossing_start_offset": float(zc_off),
        "is_silent": is_silent,
    }


def slice_stem(
    y_stem: np.ndarray,
    sr: int,
    stem: str,
    track_id: str,
    duration: float,
) -> List[Segment]:
    onsets = detect_onsets(y_stem, sr, stem=stem, backtrack=True)
    bounds = [0.0] + [float(t) for t in onsets if 0.0 < float(t) < duration] + [duration]
    # dedupe + enforce min segment length; merge tiny slivers forward
    bounds = sorted(set(bounds))
    merged: List[float] = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < MIN_SEGMENT_SEC:
            continue
        merged.append(b)
    if merged[-1] < duration - 1e-6:
        merged.append(duration)
    # cap pathological lengths (sparse stems with no onsets for minutes)
    capped: List[float] = [merged[0]]
    for b in merged[1:]:
        while b - capped[-1] > MAX_SEGMENT_SEC:
            capped.append(capped[-1] + MAX_SEGMENT_SEC)
        capped.append(b)
    merged = capped
    segments: List[Segment] = []
    for i in range(len(merged) - 1):
        s, e = merged[i], merged[i + 1]
        d = e - s
        if d <= 0:
            continue
        s_samp = int(round(s * sr))
        e_samp = int(round(e * sr))
        y_seg = y_stem[s_samp:e_samp]
        feats = segment_features(y_seg, sr, y_stem, s_samp)
        seg = Segment(
            id=f"seg_{stem}_{i:04d}",
            track_id=track_id,
            stem=stem,  # type: ignore[arg-type]
            start=float(s),
            duration=float(d),
            is_silent=feats["is_silent"],
            loudness_start=feats["loudness_start"],
            loudness_max=feats["loudness_max"],
            loudness_max_time=feats["loudness_max_time"],
            pitches=feats["pitches"],
            timbre=feats["timbre"],
            spectral_centroid=feats["spectral_centroid"],
            zero_crossing_start_offset=feats["zero_crossing_start_offset"],
        )
        segments.append(seg)
    return segments


def analyze_track(
    input_audio: Path,
    stems_dir: Path | None,
    output_json: Path,
    track_id: str | None = None,
    filename: str | None = None,
) -> TrackAnalysis:
    """Full analysis for one track. Reads stems if present else full mix."""
    import soundfile as sf
    import librosa

    if track_id is None:
        h = hashlib.sha256()
        with open(input_audio, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        track_id = h.hexdigest()[:16]
    if filename is None:
        filename = input_audio.name

    stem_audio: Dict[str, np.ndarray] = {}
    sr_out = SR
    duration = 0.0
    stems_available: List[str] = []
    if stems_dir is not None and stems_dir.exists():
        from wub_core.separator import STEM_FILES

        for stem in STEMS:
            p = stems_dir / STEM_FILES[stem]
            if p.exists():
                y, sr = sf.read(str(p), always_2d=True)
                if sr != SR:
                    y_mono = y.mean(axis=1)
                    y_rs = librosa.resample(y_mono, orig_sr=sr, target_sr=SR)
                    y_stem = y_rs.astype(np.float32)
                else:
                    y_stem = y.mean(axis=1).astype(np.float32)
                stem_audio[stem] = y_stem
                stems_available.append(stem)
                duration = max(duration, len(y_stem) / float(SR))
    if not stem_audio:
        y, sr = librosa.load(str(input_audio), sr=SR, mono=True)
        y = np.asarray(y, dtype=np.float32)
        duration = len(y) / float(SR)
        # fallback: duplicate mix into 'other' for analysis continuity
        stem_audio = {"other": y}
        stems_available = ["other"]

    # Beat grid: drums stem preferred, else mix/other
    if "drums" in stem_audio and len(stem_audio["drums"]) > SR:
        beat_src = stem_audio["drums"]
    elif "other" in stem_audio:
        beat_src = stem_audio["other"]
    else:
        beat_src = next(iter(stem_audio.values()))
    beats, bpm = extract_beats(beat_src, SR)
    # Fallback to mix if drum-locked grid is degenerate
    if len(beats) < 4 and "other" in stem_audio and beat_src is not stem_audio["other"]:
        beats, bpm = extract_beats(stem_audio["other"], SR)

    raw_tatums = beats_to_tatums(beats, duration)
    # Key from mean chroma of harmonic stems (bass+other+vocals)
    import librosa as _lib

    chroma_acc: List[np.ndarray] = []
    for stem in ("bass", "other", "vocals"):
        if stem in stem_audio:
            c = _lib.feature.chroma_cqt(y=stem_audio[stem], sr=SR, hop_length=4096)
            chroma_acc.append(c.mean(axis=1))
    if chroma_acc:
        key = estimate_key(np.mean(chroma_acc, axis=0))
    else:
        key = "C major"

    # Per-stem slicing
    all_segments: List[Segment] = []
    seg_by_stem: Dict[str, List[Segment]] = {}
    for stem in stems_available:
        segs = slice_stem(stem_audio[stem], SR, stem, track_id, duration)
        seg_by_stem[stem] = segs
        all_segments.extend(segs)

    # Link tatums -> overlapping segment ids
    tatums: List[Tatum] = []
    for idx, (s, d, conf) in enumerate(raw_tatums):
        e = s + d
        mapping: Dict[str, List[str]] = {st: [] for st in STEMS}
        for stem, segs in seg_by_stem.items():
            for sg in segs:
                # overlap test
                if sg.start < e and (sg.start + sg.duration) > s:
                    mapping[stem].append(sg.id)
        bar = idx // 16
        beat = (idx % 16) // 4
        tpos = idx % 4
        tatums.append(
            Tatum(
                id=f"tatum_{idx:04d}",
                index=idx,
                bar_position=int(bar),
                beat_position=int(beat),
                tatum_position=int(tpos),
                start=float(s),
                duration=float(d),
                confidence=float(conf),
                stem_segments=mapping,  # type: ignore[arg-type]
            )
        )

    analysis = TrackAnalysis(
        track_id=track_id,
        filename=filename,
        bpm=float(bpm),
        key=key,
        duration=float(duration),
        sample_rate=SR,
        stems_available=stems_available,  # type: ignore[arg-type]
        tatums=tatums,
        segments=all_segments,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_json.with_suffix(".json.tmp")
    tmp.write_text(analysis.model_dump_json(indent=2))
    tmp.replace(output_json)  # atomic: kills never leave truncated JSON
    return analysis
