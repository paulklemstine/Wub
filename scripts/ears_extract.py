#!/usr/bin/env python3
"""Ears: DSP quality + transition-readiness scorer for every shipped kit phrase x stem.

Purpose
-------
Score every 4-bar kit phrase x stem for quality + transition-readiness so
repetition and weak phrases can be fixed and DJ-grade transitions stay
smooth. This is the "ears" step of the Wub KG-505 kit workflow
(see AGENTS.md + docs/KIT_ENGINE.md): the shipped in-browser kit engine
plays real 4-bar phrases at playbackRate 1.0, so a blurred transient or an
off-grid downbeat is immediately audible at a block boundary. Ears
measures the assets themselves, before selection/judging.

Required metrics (librosa, always available here) per phrase x stem:
  RMS/loudness, spectral contrast, MFCC mean/std, spectral flux mean,
  onset strength stats, chroma entropy, zero-crossing rate. Audio is
  loaded with soundfile/libsndfile (Ogg Opus decodes at 48 kHz; the
  player must use the decoded sr, never assume 44100). If an Opus file
  cannot be decoded, the stem is recorded with an ``error`` string and
  skipped -- never a crash.

Optional components (try/except ImportError, null fields when missing):
  - Essentia: tonal key, rhythm beat/onset, loudness (EBU R128-ish),
    spectral stats.  Fields: ``essentia_*``.
  - Aubio or madmom: onset times + beat grid, first-onset alignment to
    bar start, onset density regularity.  Fields: ``groove_*``
    (librosa fallback when neither is installed).
  - Basic Pitch or CREPE: pitch confidence / tuning deviation.
    Fields: ``pitch_*``.
  - Pyloudnorm: integrated loudness LUFS.  Field: ``loudness_lufs``
    (falls back to ``rms_db`` when missing; see
    ``loudness_lufs_source``).

Rhythm/groove: phrase BPM is estimated from the beat grid and checked
against the manifest duration (16 beats / duration) and the kit BPM;
off-grid phrases are flagged (``rhythm_off_grid`` / ``groove_off_grid``).

Quality flags (per stem):
  - silence: ``rms_db`` below the shipping MIN_RMS_DB floors
    (drums -28, bass -28, other -34, vocals -42).
  - clipping: ``peak >= 0.99``.
  - transient-blurred: low spectral flux + low onset strength
    (flux_norm < 0.14 and onset_strength_mean < 0.85).

Usage (smoke test on the shipped bank):
  python3 scripts/ears_extract.py \
      --manifest frontend/public/data/kits/kit_manifest.json \
      --kits-dir frontend/public/data/kits \
      --out /tmp/opencode/ears_dsp_smoke.json --limit 20

Full bank (run on the laptop after the kit rebuild; stems live at
/mnt/e/Wub/data/stems, processed JSON at $PROCESSED, default
/tmp/wubtrain/processed -- both recorded in the output metadata;
decoding itself uses the manifest's kit files, whose decode sr is
48 kHz for Opus):
  PYTHONPATH=. python3 scripts/ears_extract.py \
      --manifest frontend/public/data/kits/kit_manifest.json \
      --kits-dir frontend/public/data/kits \
      --out /tmp/ears_scores.json --workers 4

Output JSON: {version, manifest, kits_dir, n_scored, n_phrases,
n_errors, elapsed_sec, scores: {phrase_id: {stem: {...metrics...}}},
summary: {median/percentiles + flag counts}}.

Optional-dependency install (laptop only; NOT required to run):
  pip install essentia aubio madmom basic-pitch pyloudnorm
  (exact per-package commands are in the script header / smoke report.)

Repo style: argparse, PYTHONPATH=. compatible, numpy/scipy/soundfile/
librosa/torch only as hard deps; no frontend deps. Set
PYTHONPATH=. so ``wub_core`` imports keep working if added later.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # hard dependency
    print(f"soundfile is required: {exc}", file=sys.stderr)
    raise

try:
    import librosa
except ImportError as exc:
    print(f"librosa is required for ears_extract: {exc}", file=sys.stderr)
    raise

STEMS = ("drums", "bass", "other", "vocals")
# Shipping MIN_RMS_DB floors (dBFS) -- silence flag.
MIN_RMS_DB = {"drums": -28.0, "bass": -28.0, "other": -34.0, "vocals": -42.0}
CLIP_PEAK = 0.99
# Transient-blurred thresholds (calibrated on shipped bank, Apr 2026 smoke).
FLUX_NORM_BLURRED_THRESH = 0.14
ONSET_MEAN_BLURRED_THRESH = 0.85
BPM_OFF_GRID_PCT = 0.06
FIRST_ONSET_TOL_MS = 120.0
OUT_VERSION = "1.0.0"
HOP = 512
N_FFT = 2048


def _f(x: Any) -> Optional[float]:
    """JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def _resolve_path(kits_dir: Path, file_field: str, track_id: str = "") -> Optional[Path]:
    """Resolve a manifest file field to an on-disk path (or None if missing)."""
    if not file_field:
        return None
    cands = [
        kits_dir / file_field,
        kits_dir.parent / file_field,  # kits-dir given as .../kits
        kits_dir / "kits" / file_field,  # file given without kits/ prefix
        kits_dir / Path(file_field).name,
    ]
    if track_id:
        cands.append(kits_dir / "kits" / track_id / Path(file_field).name)
    for c in cands:
        if c.exists():
            return c
    return None


def _rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-9))


def _error_record(file_field: str, stem: str, err: str) -> Dict[str, Any]:
    """Record for a stem that could not be decoded (never raises)."""
    return {
        "file": file_field,
        "stem": stem,
        "error": err,
        "duration_sec": None,
        "sample_rate": None,
        "rms": None, "rms_db": None, "peak": None, "peak_db": None,
        "spectral_contrast_mean": None, "spectral_contrast_std": None,
        "mfcc_mean": None, "mfcc_std": None,
        "spectral_flux_mean": None, "spectral_flux_norm": None,
        "onset_strength_mean": None, "onset_strength_std": None,
        "chroma_entropy": None, "zcr_mean": None,
        "onset_times": None, "onset_count": None,
        "loudness_lufs": None, "loudness_lufs_source": None,
        "essentia_available": False, "groove_source": None,
        "pitch_available": False, "rhythm_off_grid": False,
        "is_silence": False, "is_clipping": False,
        "is_transient_blurred": False, "flags": ["decode_error"],
        "transition_ready": False, "quality_score": 0.0,
    }


def _essentia_fields(y: np.ndarray, sr: int, path: Optional[Path]) -> Dict[str, Any]:
    """Best-effort Essentia block; nulls when essentia is absent."""
    out: Dict[str, Any] = {
        "essentia_available": False,
        "essentia_key": None, "essentia_key_strength": None,
        "essentia_pitch_histogram": None,
        "essentia_beat_bpm": None, "essentia_onset_count": None,
        "essentia_loudness_r128": None, "essentia_spectral_centroid": None,
        "essentia_spectral_flatness": None, "essentia_error": None,
    }
    try:
        import essentia  # type: ignore  # noqa: F401
        import essentia.standard as es  # type: ignore
    except ImportError:
        return out
    out["essentia_available"] = True
    try:
        # Loudness (EBU R128-ish) -- takes mono float array.
        try:
            _, _, integrated, _range = es.LoudnessRE128()(y)  # type: ignore
            out["essentia_loudness_r128"] = _f(integrated)
        except Exception:
            pass
        # Key/tonal.
        try:
            tonic, mode, strength = es.Key()(y)  # type: ignore
            out["essentia_key"] = f"{tonic} {mode}"
            out["essentia_key_strength"] = _f(strength)
        except Exception:
            pass
        try:
            hpcp = es.HPCP()(y)  # type: ignore
            out["essentia_pitch_histogram"] = [float(v) for v in np.asarray(hpcp).tolist()]
        except Exception:
            pass
        # Rhythm (beat + onsets).
        try:
            bpm, _ticks, _est, _conf = es.RhythmExtractor2013()(y)  # type: ignore
            out["essentia_beat_bpm"] = _f(bpm)
        except Exception:
            pass
        try:
            onset_times = es.Onset()(y)  # type: ignore
            out["essentia_onset_count"] = int(len(onset_times))
        except Exception:
            pass
        # Spectral.
        try:
            out["essentia_spectral_centroid"] = _f(es.SpectralCentroidTime()(y))  # type: ignore
        except Exception:
            pass
        try:
            out["essentia_spectral_flatness"] = _f(es.SpectralFlatnessDinn()(y))  # type: ignore
        except Exception:
            pass
    except Exception as exc:  # never crash the scorer
        out["essentia_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def _groove_fields(
    y: np.ndarray,
    sr: int,
    path: Optional[Path],
    librosa_onsets: List[float],
    librosa_beats: List[float],
    librosa_bpm: Optional[float],
    duration_sec: float,
    expected_bpm: Optional[float],
    stem: str = "",
) -> Dict[str, Any]:
    """Onset alignment / density regularity; prefers aubio/madmom, else librosa."""
    onset_times = list(librosa_onsets)
    beat_times = list(librosa_beats)
    beat_bpm = librosa_bpm
    source = "librosa"
    if path is not None and path.exists():
        try:
            import aubio  # type: ignore

            try:
                o = aubio.onset(method="default", hop_size=HOP, sr=sr)
                onset_times = [float(t) for t in np.asarray(o(y)).tolist()]
                source = "aubio"
            except Exception:
                pass
            if beat_bpm is None or source == "aubio":
                try:
                    t = aubio.tempo(method="default", hop_size=HOP, sr=sr)
                    bpms = np.asarray(t(y))
                    if len(bpms):
                        beat_bpm = float(bpms[0])
                        source = "aubio"
                except Exception:
                    pass
        except ImportError:
            try:
                import madmom  # type: ignore

                try:
                    from madmom.features.onsets import OnsetProcessor  # type: ignore

                    # Madmom expects a file path or float array at native sr;
                    # use the file when available.
                    o = OnsetProcessor(sr=sr)(str(path))
                    onset_times = [float(t) for t in np.asarray(o).tolist()]
                    source = "madmom"
                except Exception:
                    pass
                try:
                    from madmom.features.downbeats import DBNDownBeatProcessor  # type: ignore

                    dbnd = DBNDownBeatProcessor(beats_per_bar=4, fps=100)
                    beats = dbnd(str(path))  # type: ignore
                    beat_times = [float(t) for t, _b in np.asarray(beats).tolist()]
                    source = "madmom"
                except Exception:
                    pass
            except ImportError:
                pass
    onset_times = sorted([t for t in onset_times if math.isfinite(t) and 0.0 <= t <= duration_sec + 0.05])
    beat_times = sorted([t for t in beat_times if math.isfinite(t)])
    if beat_bpm is None and len(beat_times) >= 2:
        diffs = np.diff(np.asarray(beat_times, dtype=float))
        med = float(np.median(diffs)) if len(diffs) else 0.0
        beat_bpm = (60.0 / med) if med > 1e-6 else None
    first = float(onset_times[0]) if onset_times else None
    align_ms = (first * 1000.0) if first is not None else None
    ioi = np.diff(np.asarray(onset_times, dtype=float)) if len(onset_times) >= 2 else np.array([])
    ioi_mean = float(np.mean(ioi)) if len(ioi) else None
    ioi_std = float(np.std(ioi)) if len(ioi) else None
    ioi_cv = (ioi_std / ioi_mean) if (ioi_mean and ioi_mean > 1e-9) else None
    ioi_median = float(np.median(ioi)) if len(ioi) else None
    bpm_err = None
    snapped_bpm = beat_bpm
    off_grid = False
    if beat_bpm is not None and expected_bpm:
        # Octave-error snapping: beat_track often locks half/double time
        # (e.g. 62.5 / 255 vs 126 BPM). A harmonic-lock is still on-grid.
        cands = [beat_bpm, beat_bpm / 2.0, beat_bpm * 2.0]
        best = min(cands, key=lambda c: abs(c - expected_bpm))
        snapped_bpm = best
        snap_err = abs(best - expected_bpm) / expected_bpm
        bpm_err = snap_err if best != beat_bpm else abs(beat_bpm - expected_bpm) / expected_bpm
        # Musical roles: drums define the grid, so a snapped BPM miss
        # flags off-grid. For bass/other/vocals beat_track is unreliable
        # (sustained bass, chords/lyrics), so the BPM miss alone does NOT
        # flag -- only a misaligned first onset does (see below).
        if stem in ("drums", ""):
            off_grid = bool(snap_err > BPM_OFF_GRID_PCT)
    if align_ms is not None and align_ms > FIRST_ONSET_TOL_MS:
        off_grid = True
    return {
        "groove_source": source,
        "groove_onset_times": [round(t, 4) for t in onset_times],
        "groove_beat_times": [round(t, 4) for t in beat_times],
        "groove_beat_bpm": _f(beat_bpm),
        "groove_beat_bpm_snapped": _f(snapped_bpm),
        "groove_first_onset_sec": _f(first),
        "groove_first_onset_align_ms": _f(align_ms),
        "groove_first_onset_aligned": bool(align_ms is not None and align_ms <= FIRST_ONSET_TOL_MS),
        "groove_onset_count": int(len(onset_times)),
        "groove_onset_density_per_sec": _f(len(onset_times) / duration_sec) if duration_sec > 0 else None,
        "groove_onset_ioi_mean_sec": _f(ioi_mean),
        "groove_onset_ioi_median_sec": _f(ioi_median),
        "groove_onset_ioi_std_sec": _f(ioi_std),
        "groove_onset_ioi_cv": _f(ioi_cv),
        "groove_expected_bpm": _f(expected_bpm),
        "groove_bpm_error_pct": _f(bpm_err),
        "groove_off_grid": bool(off_grid),
        "rhythm_est_bpm": _f(beat_bpm),
        "rhythm_expected_bpm": _f(expected_bpm),
        "rhythm_bpm_error_pct": _f(bpm_err),
        "rhythm_off_grid": bool(off_grid),
    }


_BP_MODEL = None  # cached Basic Pitch Model per worker process
_BP_FAILED = False


def _pitch_fields(y: np.ndarray, sr: int, path: Optional[Path], stem: str = "other") -> Dict[str, Any]:
    """Pitch confidence / tuning deviation; nulls when pitch libs absent."""
    out: Dict[str, Any] = {
        "pitch_available": False, "pitch_source": None,
        "pitch_confidence": None, "pitch_tuning_cents": None,
        "pitch_f0_median_hz": None, "pitch_voiced_fraction": None,
        "pitch_error": None,
    }
    # Drums are unpitched: running Basic Pitch on them is pure waste.
    if stem == "drums":
        out["pitch_source"] = "skipped-drums"
        return out

    def _tuning_cents(freqs: np.ndarray) -> Optional[float]:
        f = np.asarray(freqs, dtype=float)
        f = f[f > 20.0]
        if len(f) == 0:
            return None
        midi = 69.0 + 12.0 * np.log2(f / 440.0)
        dev = (midi - np.round(midi)) * 100.0  # cents from nearest semitone
        return float(np.median(np.abs(dev)))

    # 1) Basic Pitch (file-based; model cached once per worker process —
    #    predict() with no model arg reloads the ONNX weights per call, which
    #    cost ~30 s/stem. Passing the loaded Model reuses it.)
    global _BP_MODEL, _BP_FAILED
    try:
        try:
            from basic_pitch.inference import predict as bp_predict  # type: ignore

            if path is not None and path.exists() and not _BP_FAILED:
                if _BP_MODEL is None:
                    from basic_pitch.inference import Model as _BPModel  # type: ignore
                    from basic_pitch import ICASSP_2022_MODEL_PATH as _BP_DEFAULT  # type: ignore
                    _BP_MODEL = _BPModel(_BP_DEFAULT)
                notes, _model = bp_predict(str(path), _BP_MODEL)
                out["pitch_available"] = True
                out["pitch_source"] = "basic-pitch"
                try:
                    arr = np.asarray(notes)
                    if arr.size:
                        out["pitch_confidence"] = float(np.mean(arr[:, 5])) if arr.ndim == 2 and arr.shape[1] >= 6 else None
                        out["pitch_voiced_fraction"] = float(np.clip((arr[:, 3] - arr[:, 2]).mean() / max(1e-9, float(np.mean(arr[:, 3]))), 0.0, 1.0)) if arr.ndim == 2 and arr.shape[1] >= 4 else None
                        # f0 proxy from note pitches.
                        f0 = 440.0 * 2 ** ((arr[:, 1] - 69.0) / 12.0) if arr.ndim == 2 and arr.shape[1] >= 2 else np.array([])
                        if len(f0):
                            out["pitch_f0_median_hz"] = float(np.median(f0))
                            out["pitch_tuning_cents"] = _tuning_cents(f0)
                except Exception:
                    pass
                return out
        except ImportError:
            pass

        # 2) CREPE (numpy-based).
        try:
            import crepe  # type: ignore

            _t, freq, conf, _act = crepe.predict(y, sr, verbose=False)
            out["pitch_available"] = True
            out["pitch_source"] = "crepe"
            confa = np.asarray(conf, dtype=float)
            fra = np.asarray(freq, dtype=float)
            voiced = confa > 0.5
            out["pitch_confidence"] = float(np.mean(confa)) if len(confa) else None
            out["pitch_voiced_fraction"] = float(np.mean(voiced)) if len(confa) else None
            out["pitch_f0_median_hz"] = float(np.median(fra[voiced])) if voiced.any() else None
            out["pitch_tuning_cents"] = _tuning_cents(fra[voiced]) if voiced.any() else None
        except ImportError:
            pass
    except Exception as exc:  # never crash the scorer
        out["pitch_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def _loudness_fields(y_stereo: np.ndarray, sr: int, rms_db_val: float) -> Dict[str, Any]:
    """Integrated LUFS via pyloudnorm; falls back to RMS dB."""
    try:
        try:
            import pyloudnorm  # type: ignore

            meter = pyloudnorm.Meter(sr)
            data = y_stereo if y_stereo.ndim == 2 else y_stereo[:, None]
            val = meter.integrated_loudness(data)
            if math.isfinite(float(val)):
                return {"loudness_lufs": float(val), "loudness_lufs_source": "pyloudnorm"}
        except ImportError:
            pass
    except Exception:
        pass
    return {"loudness_lufs": _f(rms_db_val), "loudness_lufs_source": "rms_db_fallback"}


def score_stem(file_path: Path, stem: str, file_field: str,
               duration_sec: float, kit_bpm: Optional[float]) -> Dict[str, Any]:
    """Score one phrase x stem audio file. Never raises (errors are recorded)."""
    try:
        data, sr = sf.read(str(file_path), always_2d=True)
    except Exception as exc:
        # Opus/libndfile decode failures are reported, not fatal.
        return _error_record(file_field, stem,
                             f"sf.read failed ({type(exc).__name__}: {exc}) path={file_path}")
    if data is None or len(data) == 0:
        return _error_record(file_field, stem, f"empty audio path={file_path}")
    data = np.asarray(data, dtype=np.float32)
    n_ch = int(data.shape[1]) if data.ndim == 2 else 1
    mono = data.mean(axis=1).astype(np.float32) if data.ndim == 2 else data.astype(np.float32)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    duration_actual = float(len(mono) / float(sr)) if sr else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12)) if len(mono) else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-9))
    peak_db = 20.0 * math.log10(max(peak, 1e-9))
    expected_bpm = kit_bpm if (kit_bpm and kit_bpm > 10) else (
        (60.0 * 16.0 / duration_sec) if duration_sec > 0 else None)

    rec: Dict[str, Any] = {
        "file": file_field, "stem": stem, "error": None,
        "duration_sec": duration_actual, "sample_rate": int(sr), "channels": n_ch,
        "rms": float(rms), "rms_db": float(rms_db),
        "peak": float(peak), "peak_db": float(peak_db),
        "manifest_duration_sec": _f(duration_sec),
        "kit_bpm": _f(kit_bpm),
    }

    # --- librosa block (each feature isolated; one failure != whole stem) ---
    S = None
    sc_mean = sc_std = None
    mfcc_mean = mfcc_std = None
    flux_mean = flux_std = flux_max = flux_norm = None
    o_mean = o_std = o_max = None
    chroma_ent = chroma_mean = None
    zcr_mean = zcr_std = None
    centroid_mean = None
    onset_times: List[float] = []
    beat_bpm = None
    beat_times_lib: List[float] = []
    try:
        S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP, center=True))
        if S.size and S.shape[1] > 1:
            d = np.diff(S, axis=1)
            pos = np.maximum(0.0, d)
            flux_mean = float(np.mean(pos))
            flux_std = float(np.std(pos))
            flux_max = float(np.max(pos))
            mean_mag = float(np.mean(S))
            flux_norm = float(flux_mean / mean_mag) if mean_mag > 1e-12 else 0.0
    except Exception:
        S = None
    try:
        sc = librosa.feature.spectral_contrast(
            S=S, sr=sr, n_bands=6) if S is not None else librosa.feature.spectral_contrast(
            y=mono, sr=sr, n_bands=6)
        sc = np.asarray(sc, dtype=float)
        sc_mean = float(np.mean(sc)); sc_std = float(np.std(sc))
    except Exception:
        pass
    try:
        mf = np.asarray(librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13, hop_length=HOP), dtype=float)
        mfcc_mean = [float(v) for v in mf.mean(axis=1).tolist()]
        mfcc_std = float(np.mean(np.std(mf, axis=1)))
    except Exception:
        pass
    try:
        env = np.asarray(librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP), dtype=float)
        o_mean = float(np.mean(env)); o_std = float(np.std(env)); o_max = float(np.max(env))
    except Exception:
        pass
    try:
        try:
            chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=1024)
        except Exception:
            chroma = librosa.feature.chroma_stft(S=S, sr=sr) if S is not None else None
        if chroma is not None:
            cm = np.asarray(chroma, dtype=float).mean(axis=1)
            tot = float(cm.sum())
            if tot > 1e-9:
                p = cm / tot
                chroma_ent = float(-(p * np.log2(p + 1e-12)).sum())
            chroma_mean = [float(v) for v in cm.tolist()]
    except Exception:
        pass
    try:
        z = np.asarray(librosa.feature.zero_crossing_rate(mono), dtype=float)
        zcr_mean = float(np.mean(z)); zcr_std = float(np.std(z))
    except Exception:
        pass
    try:
        cent = np.asarray(librosa.feature.spectral_centroid(y=mono, sr=sr)[0], dtype=float)
        centroid_mean = float(np.mean(cent))
    except Exception:
        pass
    try:
        onset_times = [float(t) for t in librosa.onset.onset_detect(
            y=mono, sr=sr, units="time", backtrack=True)]
    except Exception:
        onset_times = []
    try:
        tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, hop_length=HOP)
        beat_bpm = float(np.asarray(tempo).ravel()[0]) if len(np.asarray(tempo).ravel()) else None
        if beats is not None and len(beats):
            beat_times_lib = [float(b * HOP / float(sr)) for b in np.asarray(beats).ravel()]
    except Exception:
        pass

    rec.update({
        "spectral_contrast_mean": _f(sc_mean), "spectral_contrast_std": _f(sc_std),
        "mfcc_mean": mfcc_mean, "mfcc_std": _f(mfcc_std),
        "mfcc_mean_mean": _f(float(np.mean(mfcc_mean))) if mfcc_mean else None,
        "spectral_flux_mean": _f(flux_mean), "spectral_flux_std": _f(flux_std),
        "spectral_flux_max": _f(flux_max), "spectral_flux_norm": _f(flux_norm),
        "onset_strength_mean": _f(o_mean), "onset_strength_std": _f(o_std),
        "onset_strength_max": _f(o_max),
        "chroma_entropy": _f(chroma_ent), "chroma_mean": chroma_mean,
        "zcr_mean": _f(zcr_mean), "zcr_std": _f(zcr_std),
        "spectral_centroid_mean": _f(centroid_mean),
        "onset_times": [round(t, 4) for t in onset_times],
        "onset_count": int(len(onset_times)),
        "onset_density_per_sec": _f(len(onset_times) / duration_actual)
        if duration_actual > 0 else None,
    })

    # --- optional blocks (all null-safe) ---
    rec.update(_loudness_fields(data, int(sr), rms_db))
    rec.update(_groove_fields(mono, int(sr), file_path, onset_times,
                              beat_times_lib, beat_bpm, duration_actual,
                              expected_bpm, stem))
    rec.update(_pitch_fields(mono, int(sr), file_path, stem))
    rec.update(_essentia_fields(mono, int(sr), file_path))

    # --- quality flags ---
    is_silence = bool(rms_db < MIN_RMS_DB.get(stem, -40.0))
    is_clipping = bool(peak >= CLIP_PEAK)
    is_blurred = bool(
        (flux_norm is not None and flux_norm < FLUX_NORM_BLURRED_THRESH)
        and (o_mean is not None and o_mean < ONSET_MEAN_BLURRED_THRESH))
    off_grid = bool(rec.get("rhythm_off_grid", False))
    flags: List[str] = []
    if is_silence:
        flags.append("silence")
    if is_clipping:
        flags.append("clipping")
    if is_blurred:
        flags.append("transient_blurred")
    if off_grid:
        flags.append("off_grid")
    score = 1.0
    if is_silence:
        score -= 0.5
    if is_clipping:
        score -= 0.3
    if is_blurred:
        score -= 0.2
    if off_grid:
        score -= 0.1
    rec.update({
        "is_silence": is_silence, "is_clipping": is_clipping,
        "is_transient_blurred": is_blurred,
        "flags": flags,
        "transition_ready": bool(not (is_silence or is_clipping or is_blurred or off_grid)),
        "quality_score": float(max(0.0, min(1.0, score))),
        "min_rms_db_floor": MIN_RMS_DB.get(stem),
    })
    return rec


def _job(args: Tuple[str, str, str, str, float, Optional[float]]) -> Tuple[str, str, Dict[str, Any], float]:
    """Top-level worker (picklable): one phrase x stem."""
    phrase_id, stem, path_str, file_field, dur, bpm = args
    t0 = time.time()
    try:
        rec = score_stem(Path(path_str), stem, file_field, dur, bpm)
    except Exception as exc:  # belt-and-braces: workers never raise
        rec = _error_record(file_field, stem, f"{type(exc).__name__}: {exc}")
    return phrase_id, stem, rec, time.time() - t0


def _iter_jobs(manifest: Dict[str, Any], kits_dir: Path, limit: int) -> List[Tuple[str, str, str, str, float, Optional[float]]]:
    jobs: List[Tuple[str, str, str, str, float, Optional[float]]] = []
    n_phrases = 0
    for kit in manifest.get("kits", []):
        kit_bpm = kit.get("bpm")
        try:
            kit_bpm = float(kit_bpm) if kit_bpm is not None else None
        except (TypeError, ValueError):
            kit_bpm = None
        track_id = kit.get("track_id", "")
        for ph in kit.get("phrases", []):
            if limit and n_phrases >= limit:
                return jobs
            n_phrases += 1
            pid = ph.get("id", f"unknown_{n_phrases}")
            dur = float(ph.get("duration_sec", 0.0) or 0.0)
            for stem in STEMS:
                st = (ph.get("stems", {}) or {}).get(stem)
                if not st or not st.get("file"):
                    continue
                p = _resolve_path(kits_dir, st["file"], track_id)
                # If unresolvable, still emit a job with the expected path so
                # the error is recorded with a clear path.
                path_str = str(p) if p else str(kits_dir / st["file"])
                jobs.append((pid, stem, path_str, st["file"], dur, kit_bpm))
    return jobs


def _summarize(scores: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    keys = ["rms_db", "peak", "spectral_contrast_mean", "spectral_flux_mean",
            "spectral_flux_norm", "onset_strength_mean", "chroma_entropy",
            "zcr_mean", "spectral_centroid_mean", "loudness_lufs",
            "groove_beat_bpm", "quality_score"]
    per_stem: Dict[str, Any] = {}
    overall: Dict[str, Any] = {}
    for stem in STEMS:
        vals: Dict[str, List[float]] = {k: [] for k in keys}
        n = n_err = n_sil = n_clip = n_blur = n_off = n_ready = 0
        for ph in scores.values():
            if stem not in ph:
                continue
            r = ph[stem]
            n += 1
            if r.get("error"):
                n_err += 1
                continue
            for k in keys:
                v = r.get(k)
                if v is not None and math.isfinite(float(v)):
                    vals[k].append(float(v))
            if r.get("is_silence"):
                n_sil += 1
            if r.get("is_clipping"):
                n_clip += 1
            if r.get("is_transient_blurred"):
                n_blur += 1
            if r.get("rhythm_off_grid"):
                n_off += 1
            if r.get("transition_ready"):
                n_ready += 1
        s: Dict[str, Any] = {"n": n, "n_errors": n_err,
                             "n_silence": n_sil, "n_clipping": n_clip,
                             "n_transient_blurred": n_blur, "n_off_grid": n_off,
                             "n_transition_ready": n_ready}
        for k in keys:
            vv = vals[k]
            s[k] = {"n": len(vv), "median": _percentile(vv, 50),
                    "p10": _percentile(vv, 10), "p90": _percentile(vv, 90)}
        per_stem[stem] = s
    # Overall (all stems pooled) for the headline keys.
    all_vals: Dict[str, List[float]] = {k: [] for k in keys}
    for ph in scores.values():
        for r in ph.values():
            if r.get("error"):
                continue
            for k in keys:
                v = r.get(k)
                if v is not None and math.isfinite(float(v)):
                    all_vals[k].append(float(v))
    for k in keys:
        vv = all_vals[k]
        overall[k] = {"n": len(vv), "median": _percentile(vv, 50),
                      "p10": _percentile(vv, 10), "p90": _percentile(vv, 90)}
    return {"per_stem": per_stem, "overall": overall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--kits-dir", default="frontend/public/data/kits")
    ap.add_argument("--out", default="/tmp/opencode/ears_dsp_smoke.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap phrases scored (0 = all; e.g. 20 for smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stems", default="/mnt/e/Wub/data/stems",
                    help="laptop stem corpus root (metadata only; decode uses manifest files)")
    ap.add_argument("--processed", default=os.environ.get("PROCESSED", "/tmp/wubtrain/processed"),
                    help="processed JSON dir (metadata only)")
    args = ap.parse_args()

    t_start = time.time()
    manifest_path = Path(args.manifest)
    kits_dir = Path(args.kits_dir)
    manifest = json.loads(manifest_path.read_text())
    jobs = _iter_jobs(manifest, kits_dir, args.limit)
    print(f"jobs: {len(jobs)} stems from {len(set(j[0] for j in jobs))} phrases "
          f"(limit={args.limit or 'all'} workers={args.workers})", flush=True)
    scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
    n_errors = 0
    per_stem_times: Dict[str, List[float]] = {s: [] for s in STEMS}
    if args.workers and args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, j): j for j in jobs}
            for f in as_completed(futs):
                pid, stem, rec, dt = f.result()
                scores.setdefault(pid, {})[stem] = rec
                per_stem_times.setdefault(stem, []).append(dt)
                if rec.get("error"):
                    n_errors += 1
    else:
        for j in jobs:
            pid, stem, rec, dt = _job(j)
            scores.setdefault(pid, {})[stem] = rec
            per_stem_times.setdefault(stem, []).append(dt)
            if rec.get("error"):
                n_errors += 1
    # Stable phrase order (manifest order).
    ordered = dict(sorted(scores.items()))
    summary = _summarize(ordered)
    all_dt = [d for v in per_stem_times.values() for d in v]
    elapsed = time.time() - t_start
    n_ph = max(1, len(ordered))
    out = {
        "version": OUT_VERSION,
        "manifest": str(manifest_path),
        "kits_dir": str(kits_dir),
        "processed": str(args.processed),
        "stems": str(args.stems),
        "limit": int(args.limit),
        "n_scored": int(sum(len(v) for v in ordered.values())),
        "n_phrases": int(len(ordered)),
        "n_errors": int(n_errors),
        "elapsed_sec": round(elapsed, 2),
        "sec_per_stem_mean": _f(float(np.mean(all_dt))) if all_dt else None,
        "sec_per_phrase_mean": _f(sum(all_dt) / n_ph) if all_dt else None,
        "scores": ordered,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(json.dumps({"out": str(out_path), "n_scored": out["n_scored"],
                      "n_phrases": out["n_phrases"], "n_errors": n_errors,
                      "elapsed_sec": out["elapsed_sec"],
                      "sec_per_stem_mean": out["sec_per_stem_mean"]}))
    for stem in STEMS:
        s = summary["per_stem"].get(stem, {})
        if s:
            print(f"  {stem}: n={s['n']} err={s['n_errors']} silence={s['n_silence']} "
                  f"clip={s['n_clipping']} blurred={s['n_transient_blurred']} "
                  f"off_grid={s['n_off_grid']} ready={s['n_transition_ready']} "
                  f"rms_db_med={s.get('rms_db', {}).get('median')} "
                  f"flux_norm_med={s.get('spectral_flux_norm', {}).get('median')}")


if __name__ == "__main__":
    main()
