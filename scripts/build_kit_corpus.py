"""Build the M0 phrase-kit corpus: 4-bar, 4-stem Opus windows + features.

A "kit" is a set of real 4-bar windows cut from one analysed 4-stem track. Each
phrase keeps the exact same time range in every stem, so the mix balance of the
original production survives untouched (no normalisation, no per-stem gain).

Pipeline per track (``scripts/build_tatum_dataset.py`` sets the section-arc
precedent):

1. ``wub_core.section_arcs.bar_features`` + ``segment_sections`` -> per-bar
   energy and labelled section spans (INTRO / GROOVE_A / BUILDUP / DROP /
   BREAKDOWN).
2. Enumerate 4-bar windows on bar boundaries: every section start first, then
   the disjoint ``[0, 4, 8, ...]`` grid filling the gaps, so candidates never
   overlap and every section downbeat is a candidate.
   Window start = ``tatums[16*b].start``, end = ``tatums[16*(b+bars)].start``;
   windows outside 3.0-20.0 s or past the tatum list are dropped.
3. Per window: section label, mean bar energy, per-stem peak/RMS straight from
   the audio, and a 64-d log-mel embedding of the 4-stem sum (mean over time,
   spectral-tilt removed, L2-normalised) used for novelty.
4. Windows with any silent stem (DC-free peak < 0.01, or RMS < -50 dBFS) or a
   near-silent sum are rejected. Kits with fewer than ``MIN_PHRASES``
   qualifying windows are skipped entirely.
5. Selection is greedy: energy + section-start bonus + novelty (cosine distance
   to every already-chosen window must stay >= 0.15), max two windows per
   section span, never overlapping.
6. The chosen windows are cut from ``<stems>/<tid>/<stem>.wav`` with an 8 ms
   linear fade in/out and encoded as Ogg Opus (default 64k) via
   ``scripts.audio_encode.write_opus``.

Output::

    <out>/kits/<track_id>/kit_<track_id>_pNN_<stem>.opus
    <out>/kit_manifest.json

Idempotency: a kit is reused verbatim when its manifest entry and all four
phrase files still exist; individual phrase files are skipped when the manifest
already records the same window. Audio is never re-normalised or re-gained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# One BLAS/FFT thread per worker: the pool already provides the parallelism.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wub_core.section_arcs import bar_features, segment_sections

try:  # ``scripts`` is a namespace package when the repo root is on sys.path.
    from scripts.audio_encode import write_opus
except Exception:  # pragma: no cover - direct ``python scripts/foo.py`` runs
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audio_encode import write_opus  # type: ignore[no-redef]

STEMS: Tuple[str, ...] = ("drums", "bass", "other", "vocals")
DEFAULT_SR = 44100
MIN_PHRASES = 4
MIN_PEAK = 0.01
# A stem can also be inaudible while a DC/rumble-only tail keeps its peak above
# 0.01 (Opus then strips that content entirely, shipping a -70 dB file). An RMS
# floor catches those windows; sparse-but-real stems (e.g. a kick-only
# breakdown at -44 dB RMS) still pass.
MIN_STEM_RMS_DB = -50.0
MIN_DURATION = 3.0
MAX_DURATION = 20.0
FADE_MS = 8.0
MIN_NOVELTY = 0.15
MAX_PER_SECTION = 2
W_ENERGY = 0.40
W_SECTION_START = 0.35
W_NOVELTY = 0.35
# Phrase-length coherence: a kit whose phrases differ in length makes the
# player's tempo jump at every 4-bar join. Prefer windows near the kit's
# median length and drop outliers when enough phrases remain.
W_DURATION = 1.8
DURATION_TOL = 0.006  # 0.6% of the median window length
EMBED_MELS = 64
MANIFEST_NAME = "kit_manifest.json"
INDEX_NAME = "_track_index.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def classify_style(filename: str, bpm: float) -> str:
    """``scripts.harvest_style_kits.classify_track_style`` with a safe default."""
    try:
        try:
            from scripts.harvest_style_kits import classify_track_style
        except Exception:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from harvest_style_kits import classify_track_style  # type: ignore
        return str(classify_track_style(filename or "", float(bpm or 0.0)))
    except Exception:
        return "house"


def _title_of(filename: Optional[str], track_id: str) -> str:
    name = str(filename or "").strip() or track_id
    for ext in (".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _peak_rms(mono: np.ndarray) -> Tuple[float, float]:
    """Peak and RMS in dBFS (floor -120 dB, never NaN/inf)."""
    if mono.size == 0:
        return 0.0, -120.0
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    rms_db = 20.0 * math.log10(max(rms, 1e-6))
    return peak, max(rms_db, -120.0)


def _ac_peak(mono: np.ndarray) -> float:
    """Peak after removing the DC offset.

    Demucs occasionally leaves a stem window with nothing but DC drift (one
    corpus bass window was 70% of its energy below 5 Hz). A codec strips DC,
    so such a window encodes as silence however loud its raw peak looks; the
    audibility gate has to see through the offset.
    """
    if mono.size == 0:
        return 0.0
    return float(np.max(np.abs(mono - mono.mean(dtype=np.float64))))


def _apply_fades(mono: np.ndarray, sr: int, fade_ms: float = FADE_MS) -> np.ndarray:
    """Linear 8 ms fade in/out, in place. Never normalises."""
    if mono.size < 4:
        return mono
    n = min(int(round(fade_ms * 0.001 * sr)), mono.size // 2)
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        mono[:n] *= ramp
        mono[-n:] *= ramp[::-1]
    return mono


def _embedding(sum_mono: np.ndarray, sr: int) -> np.ndarray:
    """64-d log-mel embedding of the 4-stem sum, L2-normalised.

    ``librosa.feature.melspectrogram`` (n_mels=64) -> log (dB) -> mean over
    time. The resulting 64 band levels are then high-passed along the mel axis
    (subtract a sigma=1-band Gaussian smoothing) and mean-centred.

    Why the extra step: a raw log-mel envelope is nearly identical for every
    window of one track (the broad spectral tilt dominates the angle), so
    cosine distances between raw envelopes sit around 0.003 and the 0.15
    novelty gate would reject every window. Removing the shared tilt makes the
    distance measure what actually changes between windows (kick/sub weight,
    hat and vocal presence) while staying a pure function of the window, so an
    embedding is reproducible and comparable across tracks.
    """
    import librosa
    from scipy.ndimage import gaussian_filter1d

    mel = librosa.feature.melspectrogram(
        y=np.asarray(sum_mono, dtype=np.float32), sr=int(sr), n_fft=2048,
        hop_length=512, n_mels=EMBED_MELS, power=2.0, fmax=sr / 2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    emb = np.asarray(mel_db.mean(axis=1), dtype=np.float64)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    emb = emb - gaussian_filter1d(emb, 1.0)
    emb -= emb.mean()
    norm = float(np.linalg.norm(emb))
    if norm > 1e-9:
        emb /= norm
    else:
        emb[:] = 0.0
    return emb


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, min(2.0, 1.0 - float(np.dot(a, b)))))


def _overlaps(a: dict, b: dict) -> bool:
    return a["start_frame"] < b["end_frame"] and b["start_frame"] < a["end_frame"]


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def _best_consecutive_run(
    candidates: Sequence[dict],
    n_max: int,
) -> List[dict]:
    """Longest chain of 4-bar windows at start_bar + 4, best energy first."""
    by_start = {int(c["start_bar"]): c for c in candidates if not c.get("taken")}
    best: List[dict] = []
    best_key = (-1, -1e9)
    for c in candidates:
        if c.get("taken"):
            continue
        run = [c]
        bar = int(c["start_bar"])
        while (bar + 4) in by_start:
            bar += 4
            run.append(by_start[bar])
        if len(run) < MIN_PHRASES:
            continue
        energy = sum(float(x.get("energy_norm", 0.5)) for x in run) / len(run)
        sections = len({x["span_key"] for x in run})
        hot = 1.0 if any(str(x.get("section", "")).upper() in ("DROP", "BUILDUP")
                         for x in run) else 0.0
        score = energy + 0.15 * sections + 0.25 * hot
        key = (min(len(run), n_max), score)
        if key > best_key:
            best_key = key
            best = run[:n_max]
    return best


def _select_phrases(
    candidates: Sequence[dict],
    n_max: int,
    max_per_section: int = MAX_PER_SECTION,
    min_novelty: float = MIN_NOVELTY,
) -> List[dict]:
    """Greedy pick of energy/novelty/section-start windows, never overlapping."""
    if not candidates:
        return []
    energies = np.array([float(c["energy_db"]) for c in candidates], dtype=np.float64)
    lo, hi = float(energies.min()), float(energies.max())
    span = hi - lo
    for c, e in zip(candidates, energies):
        c["energy_norm"] = float((e - lo) / span) if span > 1e-6 else 0.5

    durations = np.array([float(c["duration"]) for c in candidates], dtype=np.float64)
    duration_med = float(np.median(durations)) if len(durations) else 0.0

    run = _best_consecutive_run(candidates, n_max)
    chosen: List[dict] = list(run)
    for c in chosen:
        c["taken"] = True
    per_section: Counter = Counter(c["span_key"] for c in chosen)
    while len(chosen) < n_max:
        best: Optional[dict] = None
        best_score = -1e9
        for c in candidates:
            if c.get("taken"):
                continue
            if per_section[c["span_key"]] >= max_per_section:
                continue
            if any(_overlaps(c, o) for o in chosen):
                continue
            if chosen:
                novelty = min(_cosine_distance(c["embedding"], o["embedding"])
                              for o in chosen)
                if novelty < min_novelty:
                    continue
            else:
                novelty = 1.0
            duration_fit = 1.0 - min(
                1.0, abs(float(c["duration"]) - duration_med) / max(1e-6, 0.02 * duration_med)
            )
            score = (W_ENERGY * c["energy_norm"]
                     + W_SECTION_START * (1.0 if c["is_section_start"] else 0.0)
                     + W_NOVELTY * novelty
                     + W_DURATION * duration_fit)
            if score > best_score:
                best_score, best = score, c
        if best is None:
            break
        best["taken"] = True
        per_section[best["span_key"]] += 1
        chosen.append(best)

    if not run and duration_med > 1e-6 and len(chosen) > MIN_PHRASES:
        coherent = [
            c for c in chosen
            if abs(float(c["duration"]) - duration_med) / duration_med <= DURATION_TOL
        ]
        if len(coherent) >= MIN_PHRASES:
            chosen = coherent
    return chosen


# ---------------------------------------------------------------------------
# Per-track worker
# ---------------------------------------------------------------------------

class _StemReader:
    """Four persistent ``SoundFile`` handles with seek/read.

    Re-opening the same WAV for every window costs ~6 ms per open on the
    external NTFS corpus drive; seeking a live handle does not.
    """

    def __init__(self, paths: Dict[str, Path]) -> None:
        self._handles: Dict[str, Any] = {}
        try:
            for s in STEMS:
                self._handles[s] = sf.SoundFile(str(paths[s]))
        except Exception:
            self.close()
            raise

    def read(self, stem: str, start: int, frames: int) -> np.ndarray:
        h = self._handles[stem]
        h.seek(int(start))
        return h.read(int(frames), dtype="float32", always_2d=True)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()


def _process_track(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    tid = str(job["track_id"])
    bars = int(job["bars"])
    n_want = int(job["phrases"])
    bitrate = str(job["bitrate"])
    out_root = Path(job["out"])
    result: Dict[str, Any] = {
        "track_id": tid, "status": "skipped", "reason": "", "kit": None,
        "stats": {"candidates": 0, "rejected": {}, "phrases": 0,
                  "files_written": 0, "bytes": 0, "max_align_ms": 0.0},
    }
    rejections: Counter = Counter()

    def _finish(status: str, reason: str = "") -> Dict[str, Any]:
        result["status"] = status
        result["reason"] = reason
        stats = result["stats"]
        stats["rejected"] = dict(sorted(rejections.items()))
        stats["max_align_ms"] = round(float(stats["max_align_ms"]), 4)
        stats["sec"] = round(time.time() - t0, 3)
        return result

    # --- fast path: reuse a complete kit from a previous run -----------------
    existing = job.get("existing")
    if isinstance(existing, dict):
        phrases = existing.get("phrases") or []
        files = [st.get("file") for p in phrases if isinstance(p, dict)
                 for st in (p.get("stems") or {}).values()]
        if (existing.get("track_id") == tid and len(phrases) >= MIN_PHRASES
                and len(files) == len(phrases) * len(STEMS)
                and all((out_root / str(f)).is_file() and (out_root / str(f)).stat().st_size > 0
                        for f in files)):
            result["kit"] = existing
            return _finish("cached")

    try:
        analysis = json.loads(Path(job["analysis"]).read_text(encoding="utf-8"))
    except Exception as exc:
        return _finish("skipped", f"analysis_error:{type(exc).__name__}")

    tatums = analysis.get("tatums") or []
    segments = analysis.get("segments") or []
    n_bars = len(tatums) // 16
    if n_bars < bars:
        return _finish("skipped", "too_short")

    stem_dir = Path(job["stem_dir"])
    stem_paths = {s: stem_dir / f"{s}.wav" for s in STEMS}
    missing = [s for s, p in stem_paths.items() if not p.is_file()]
    if missing:
        return _finish("skipped", "missing_stems:" + ",".join(missing))

    infos = {}
    for s in STEMS:
        try:
            infos[s] = sf.info(str(stem_paths[s]))
        except Exception as exc:
            return _finish("skipped", f"audio_info_error:{s}:{type(exc).__name__}")
    sr = int(infos["drums"].samplerate)
    if any(int(infos[s].samplerate) != sr for s in STEMS):
        return _finish("skipped", "sample_rate_mismatch")
    if sr != DEFAULT_SR:
        return _finish("skipped", f"unsupported_sample_rate:{sr}")

    # --- bar features + section spans ---------------------------------------
    try:
        feats = bar_features(segments, tatums)
        spans = segment_sections(feats)
    except Exception as exc:
        return _finish("skipped", f"section_error:{type(exc).__name__}")
    if not feats:
        return _finish("skipped", "no_bar_features")

    sec_of_bar = ["GROOVE_A"] * n_bars
    span_of_bar = [-1] * n_bars
    span_starts = set()
    for idx, sp in enumerate(spans):
        span_starts.add(int(sp.start_bar))
        for b in range(max(0, int(sp.start_bar)), min(n_bars, int(sp.end_bar))):
            sec_of_bar[b] = sp.label
            span_of_bar[b] = idx

    # --- candidate windows ---------------------------------------------------
    max_frames = min(int(infos[s].frames) for s in STEMS)
    # Section starts win the coverage contest, then the leftover bars are
    # filled with the disjoint [0, bars, 2*bars, ...] grid: the candidate set
    # never overlaps itself and always contains every section downbeat.
    wanted = sorted({int(b) for b in span_starts if 0 <= int(b) <= n_bars - bars})
    wanted += [b for b in range(0, n_bars - bars + 1, bars) if b not in wanted]
    bar_starts: List[int] = []
    for b in wanted:
        if any(b < o + bars and o < b + bars for o in bar_starts):
            continue
        bar_starts.append(b)

    candidates: List[dict] = []
    for b in sorted(bar_starts):
        end_idx = 16 * (b + bars)
        if end_idx >= len(tatums):
            rejections["past_tatum_end"] += 1
            continue
        tatum_start = float(tatums[16 * b]["start"])
        tatum_end = float(tatums[end_idx]["start"])
        duration = tatum_end - tatum_start
        if duration < MIN_DURATION or duration > MAX_DURATION:
            rejections["duration_out_of_range"] += 1
            continue
        start_frame = int(round(tatum_start * sr))
        end_frame = int(round(tatum_end * sr))
        if end_frame - start_frame < int(MIN_DURATION * sr):
            rejections["short_frames"] += 1
            continue
        if end_frame > max_frames:
            rejections["past_audio_end"] += 1
            continue
        candidates.append({
            "start_bar": int(b),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": end_frame - start_frame,
            "duration": float(duration),
            "tatum_start": tatum_start,
            "section": str(sec_of_bar[b]),
            "span_key": int(span_of_bar[b]),
            "is_section_start": bool(b in span_starts),
            "energy_db": float(np.mean([f["energy"] for f in feats[b:b + bars]])),
        })
    result["stats"]["candidates"] = len(candidates)
    if not candidates:
        return _finish("skipped", "no_candidates")

    # --- audio pass: per-stem loudness + embedding per window ----------------
    reader = _StemReader(stem_paths)
    try:
        survivors: List[dict] = []
        for c in candidates:
            stem_stats: Dict[str, dict] = {}
            sum_buf: Optional[np.ndarray] = None
            silent_reason = ""
            for s in STEMS:
                try:
                    raw = reader.read(s, c["start_frame"], c["n_frames"])
                except Exception as exc:
                    silent_reason = f"read_error:{s}:{type(exc).__name__}"
                    break
                if raw.size == 0:
                    silent_reason = f"silent_stem:{s}"
                    break
                mono = raw.mean(axis=1, dtype=np.float32)
                if mono.shape[0] < c["n_frames"]:
                    mono = np.pad(mono, (0, c["n_frames"] - mono.shape[0]))
                peak, rms_db = _peak_rms(mono)
                if _ac_peak(mono) < MIN_PEAK:
                    silent_reason = f"silent_stem:{s}"
                    break
                if rms_db < MIN_STEM_RMS_DB:
                    silent_reason = f"quiet_stem:{s}"
                    break
                stem_stats[s] = {"peak": peak, "rms_db": rms_db}
                sum_buf = mono if sum_buf is None else sum_buf + mono
            if silent_reason:
                rejections[silent_reason] += 1
                continue
            if sum_buf is None:
                rejections["silent_sum"] += 1
                continue
            sum_peak, sum_rms_db = _peak_rms(sum_buf)
            if _ac_peak(sum_buf) < MIN_PEAK or sum_rms_db < MIN_STEM_RMS_DB:
                rejections["silent_sum"] += 1
                continue
            try:
                c["embedding"] = _embedding(sum_buf, sr)
            except Exception as exc:
                rejections[f"embed_error:{type(exc).__name__}"] += 1
                continue
            c["stems"] = stem_stats
            c["sum_peak"] = sum_peak
            survivors.append(c)

        if len(survivors) < MIN_PHRASES:
            return _finish("skipped", "too_few_windows")

        chosen = _select_phrases(survivors, n_want)
        if len(chosen) < MIN_PHRASES:
            return _finish("skipped", "too_few_qualifying")
        chosen.sort(key=lambda c: c["start_bar"])

        # --- cut + encode ----------------------------------------------------
        kit_dir = out_root / "kits" / tid
        kit_dir.mkdir(parents=True, exist_ok=True)
        old_phrases: Dict[str, dict] = {}
        if isinstance(existing, dict):
            old_phrases = {str(p.get("id")): p for p in (existing.get("phrases") or [])
                           if isinstance(p, dict)}

        phrases: List[dict] = []
        written = 0
        nbytes = 0
        max_align_ms = 0.0
        for i, c in enumerate(chosen, start=1):
            pid = f"kit_{tid}_p{i:02d}"
            start_sec = round(c["start_frame"] / sr, 6)
            end_sec = round(c["end_frame"] / sr, 6)
            phrase: Dict[str, Any] = {
                "id": pid,
                "section": c["section"],
                "start_bar": int(c["start_bar"]),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(c["n_frames"] / sr, 6),
                "energy": round(float(c["energy_norm"]), 4),
                "energy_db": round(float(c["energy_db"]), 2),
                "embedding": [float(round(float(v), 4)) for v in c["embedding"]],
                "stems": {},
            }
            max_align_ms = max(max_align_ms,
                               abs(c["start_frame"] / sr - c["tatum_start"]) * 1000.0)
            reuse = old_phrases.get(pid)
            reuse_ok = (isinstance(reuse, dict)
                        and abs(float(reuse.get("start_sec", -1.0)) - start_sec) < 1e-6)
            for s in STEMS:
                rel = f"kits/{tid}/{pid}_{s}.opus"
                path = out_root / rel
                cached = ((reuse or {}).get("stems") or {}).get(s) if reuse_ok else None
                if (reuse_ok and isinstance(cached, dict) and path.is_file()
                        and path.stat().st_size > 0):
                    phrase["stems"][s] = {
                        "file": rel,
                        "peak": float(cached.get("peak", 0.0)),
                        "rms_db": float(cached.get("rms_db", -120.0)),
                    }
                    continue
                try:
                    raw = reader.read(s, c["start_frame"], c["n_frames"])
                except Exception as exc:
                    return _finish("error", f"cut_read_error:{s}:{type(exc).__name__}")
                if raw.size == 0:
                    return _finish("error", f"cut_read_empty:{s}")
                mono = raw.mean(axis=1, dtype=np.float32)
                if mono.shape[0] < c["n_frames"]:
                    mono = np.pad(mono, (0, c["n_frames"] - mono.shape[0]))
                mono = np.ascontiguousarray(_apply_fades(mono, sr))
                peak, rms_db = _peak_rms(mono)
                try:
                    write_opus(path, mono, sr, bitrate)
                except Exception as exc:
                    return _finish("error", f"encode_error:{s}:{type(exc).__name__}")
                try:
                    size = path.stat().st_size
                    magic = path.open("rb").read(36)
                except Exception as exc:
                    return _finish("error", f"verify_error:{s}:{type(exc).__name__}")
                if size <= 0 or b"OpusHead" not in magic:
                    return _finish("error", f"bad_opus:{s}")
                written += 1
                nbytes += size
                phrase["stems"][s] = {
                    "file": rel,
                    "peak": round(peak, 4),
                    "rms_db": round(rms_db, 2),
                }
            phrases.append(phrase)

        kit = {
            "track_id": tid,
            "title": _title_of(analysis.get("filename"), tid),
            "bpm": round(float(analysis.get("bpm") or 0.0), 3),
            "key": str(analysis.get("key") or ""),
            "style": str(job.get("style") or "house"),
            "source": str(job["analysis"]),
            "phrases": phrases,
        }
        result["kit"] = kit
        result["stats"]["phrases"] = len(phrases)
        result["stats"]["files_written"] = written
        result["stats"]["bytes"] = nbytes
        result["stats"]["max_align_ms"] = max_align_ms
        return _finish("ok")
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Track index / sampling
# ---------------------------------------------------------------------------

def _read_track_meta(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    entry: Dict[str, Any] = {"path": path_str, "track_id": p.stem, "filename": "",
                             "bpm": 0.0, "key": "", "duration": 0.0,
                             "n_tatums": 0, "stems_available": []}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}"
        return entry
    entry["track_id"] = str(raw.get("track_id") or p.stem)
    entry["filename"] = str(raw.get("filename") or "")
    entry["bpm"] = float(raw.get("bpm") or 0.0)
    entry["key"] = str(raw.get("key") or "")
    entry["duration"] = float(raw.get("duration") or 0.0)
    entry["n_tatums"] = len(raw.get("tatums") or [])
    entry["stems_available"] = list(raw.get("stems_available") or [])
    return entry


def _load_or_build_index(processed: Path, stems_dir: Path, out_root: Path,
                         workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Per-analysis metadata, cached in ``<out>/_track_index.json``."""
    files = sorted(p for p in processed.glob("*.json")
                   if not p.name.startswith(("_", "meta_")))
    cache_path = out_root / INDEX_NAME
    cache = _load_json(cache_path) or {}
    cached: Dict[str, dict] = {}
    if cache.get("version") == 1 and isinstance(cache.get("entries"), list):
        for e in cache["entries"]:
            if isinstance(e, dict) and e.get("path"):
                cached[str(e["path"])] = e

    with_stems = str(stems_dir)
    entries: List[Dict[str, Any]] = []
    todo: List[Tuple[Path, Tuple[int, float]]] = []
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        old = cached.get(str(p))
        if (old and old.get("mtime") == st.st_mtime and old.get("size") == st.st_size
                and old.get("has_stems_dir") == with_stems
                and "has_stems" in old and "style" in old):
            entries.append(old)
        else:
            todo.append((p, (st.st_size, st.st_mtime)))

    if todo:
        workers = max(1, min(len(todo), workers or _default_workers()))
        metas: List[dict] = []
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                metas = list(ex.map(_read_track_meta, [str(p) for p, _ in todo],
                                    chunksize=8))
        else:
            metas = [_read_track_meta(str(p)) for p, _ in todo]
        by_path = {m["path"]: m for m in metas}
        for p, (size, mtime) in todo:
            m = by_path.get(str(p))
            if not m:
                continue
            m["size"] = size
            m["mtime"] = mtime
            tid = str(m.get("track_id") or p.stem)
            m["has_stems"] = all((stems_dir / tid / f"{s}.wav").is_file() for s in STEMS)
            m["style"] = classify_style(str(m.get("filename") or ""), float(m.get("bpm") or 0.0))
            m["has_stems_dir"] = with_stems
            entries.append(m)

    entries.sort(key=lambda e: str(e.get("path")))
    _write_json_atomic(cache_path, {"version": 1, "stems": with_stems,
                                    "entries": entries})
    return entries


def _read_track_ids(spec: str) -> List[str]:
    p = Path(spec)
    if p.is_file():
        raw = p.read_text(encoding="utf-8")
        parts = [ln.strip() for ln in raw.splitlines()]
    else:
        parts = [t.strip() for t in spec.replace(",", " ").split()]
    return [t for t in parts if t]


def _pick_tracks(entries: Sequence[dict], max_tracks: Optional[int],
                 seed: int, track_ids: Optional[str]) -> List[dict]:
    """Deterministic style-diverse sample (or an explicit id list)."""
    if track_ids:
        want = _read_track_ids(track_ids)
        by_id = {str(e.get("track_id")): e for e in entries if e.get("track_id")}
        missing = [t for t in want if t not in by_id]
        if missing:
            print(f"  ! {len(missing)} requested track id(s) not in --processed: "
                  f"{', '.join(missing[:5])}", flush=True)
        picks = [by_id[t] for t in want if t in by_id]
        return picks[:max_tracks] if max_tracks else picks

    if not max_tracks:
        return [e for e in entries if e.get("has_stems") and e.get("track_id")]

    groups: Dict[str, List[dict]] = {}
    for e in entries:
        if not e.get("track_id"):
            continue
        groups.setdefault(str(e.get("style") or "house"), []).append(e)
    styles = sorted(groups)
    per_style = max(1, math.ceil(max_tracks / max(1, len(styles))))
    rng = random.Random(seed)
    picked: List[dict] = []
    for st in styles:
        grp = sorted(groups[st], key=lambda e: str(e["track_id"]))
        preferred = [e for e in grp if e.get("has_stems")]
        pool = preferred if len(preferred) >= min(per_style, len(grp)) else grp
        k = min(per_style, len(pool))
        if k:
            picked.extend(rng.sample(pool, k))

    if len(picked) < max_tracks:  # top up when a style bucket was short
        seen = {str(e["track_id"]) for e in picked}
        extra = sorted((e for e in entries
                        if e.get("has_stems") and str(e.get("track_id")) not in seen),
                       key=lambda e: str(e["track_id"]))
        picked.extend(extra[: max_tracks - len(picked)])

    picked.sort(key=lambda e: str(e["track_id"]))
    return picked[:max_tracks]


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build 4-bar, 4-stem Opus phrase kits from analysed tracks.")
    ap.add_argument("--processed", default="data/processed",
                    help="directory of per-track analysis JSON (default: data/processed)")
    ap.add_argument("--stems", default="data/stems",
                    help="directory of <track_id>/{drums,bass,other,vocals}.wav")
    ap.add_argument("--out", default="data/kits", help="output root for kits/ + manifest")
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="limit the corpus (style-diverse deterministic sample)")
    ap.add_argument("--phrases-per-kit", type=int, default=8)
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--bitrate", default="64k")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto")
    ap.add_argument("--track-ids", default=None,
                    help="comma-separated ids or a file with one id per line")
    ap.add_argument("--seed", type=int, default=505,
                    help="seed for the style-diverse sample (default: 505)")
    args = ap.parse_args()

    if args.phrases_per_kit < MIN_PHRASES:
        raise SystemExit(f"--phrases-per-kit must be >= {MIN_PHRASES}")
    if args.bars < 1:
        raise SystemExit("--bars must be >= 1")
    if args.max_tracks is not None and args.max_tracks < 1:
        raise SystemExit("--max-tracks must be >= 1")

    workers = args.workers if args.workers > 0 else _default_workers()
    processed = Path(args.processed)
    stems_dir = Path(args.stems)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "kits").mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"=== Wub kit corpus: bars={args.bars} phrases<={args.phrases_per_kit} "
          f"bitrate={args.bitrate} workers={workers} ===", flush=True)
    print(f"  processed: {processed}", flush=True)
    print(f"  stems:     {stems_dir}", flush=True)
    print(f"  out:       {out_root}", flush=True)

    manifest_path = out_root / MANIFEST_NAME
    prev = _load_json(manifest_path) or {}
    prev_kits = {str(k.get("track_id")): k for k in (prev.get("kits") or [])
                 if isinstance(k, dict) and k.get("track_id")}
    compatible = (prev.get("bars") == args.bars
                  and str(prev.get("bitrate")) == str(args.bitrate))
    if prev and not compatible:
        print(f"  ! existing manifest has bars={prev.get('bars')} "
              f"bitrate={prev.get('bitrate')}: rebuilding every kit", flush=True)

    entries = _load_or_build_index(processed, stems_dir, out_root, workers)
    total_4stem = sum(1 for e in entries if e.get("has_stems"))
    print(f"  analyses: {len(entries)} ({total_4stem} with all 4 stems on disk)", flush=True)

    picks = _pick_tracks(entries, args.max_tracks, args.seed, args.track_ids)
    if not picks:
        raise SystemExit("no tracks selected")
    style_counts = Counter(str(e.get("style") or "house") for e in picks)
    print(f"  selected: {len(picks)} tracks "
          f"({', '.join(f'{k}={v}' for k, v in sorted(style_counts.items()))})", flush=True)

    jobs: List[dict] = []
    for e in picks:
        tid = str(e["track_id"])
        jobs.append({
            "track_id": tid,
            "analysis": str(e["path"]),
            "stem_dir": str(stems_dir / tid),
            "out": str(out_root),
            "bars": args.bars,
            "phrases": args.phrases_per_kit,
            "bitrate": args.bitrate,
            "style": str(e.get("style") or "house"),
            "existing": prev_kits.get(tid) if compatible else None,
        })

    kits: Dict[str, dict] = dict(prev_kits)
    results: List[dict] = []
    done = 0
    counters: Counter = Counter()
    rejected: Counter = Counter()
    files_written = 0
    bytes_written = 0
    max_align_ms = 0.0

    def _absorb(res: dict) -> None:
        nonlocal files_written, bytes_written, max_align_ms
        results.append(res)
        counters[res.get("status", "error")] += 1
        stats = res.get("stats") or {}
        for k, v in (stats.get("rejected") or {}).items():
            rejected[k] += int(v)
        files_written += int(stats.get("files_written") or 0)
        bytes_written += int(stats.get("bytes") or 0)
        max_align_ms = max(max_align_ms, float(stats.get("max_align_ms") or 0.0))
        kit = res.get("kit")
        if kit:
            kits[str(kit["track_id"])] = kit
        else:
            kits.pop(str(res.get("track_id")), None)

    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_track, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"track_id": job["track_id"], "status": "error",
                           "reason": f"worker_crash:{type(exc).__name__}:{exc}",
                           "kit": None, "stats": {}}
                _absorb(res)
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  [{done}/{len(jobs)}] {el:6.1f}s  "
                          f"kits={counters.get('ok', 0) + counters.get('cached', 0)} "
                          f"skipped={counters.get('skipped', 0)} "
                          f"errors={counters.get('error', 0)}", flush=True)
                if done % 25 == 0:
                    _write_json_atomic(manifest_path, _manifest(args, kits))
    else:
        for job in jobs:
            try:
                res = _process_track(job)
            except Exception as exc:
                res = {"track_id": job["track_id"], "status": "error",
                       "reason": f"worker_crash:{type(exc).__name__}:{exc}",
                       "kit": None, "stats": {}}
            _absorb(res)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {el:6.1f}s  "
                      f"kits={counters.get('ok', 0) + counters.get('cached', 0)} "
                      f"skipped={counters.get('skipped', 0)} "
                      f"errors={counters.get('error', 0)}", flush=True)
            if done % 25 == 0:
                _write_json_atomic(manifest_path, _manifest(args, kits))

    manifest = _manifest(args, kits)
    _write_json_atomic(manifest_path, manifest)

    n_phrases = sum(len(k.get("phrases") or []) for k in kits.values())
    n_files = sum(len(p.get("stems") or {}) for k in kits.values()
                  for p in (k.get("phrases") or []))
    total_bytes = sum((out_root / st["file"]).stat().st_size
                      for k in kits.values() for p in (k.get("phrases") or [])
                      for st in (p.get("stems") or {}).values()
                      if (out_root / st["file"]).is_file())
    elapsed = time.time() - t_start
    print("\n=== Summary ===", flush=True)
    print(f"  processed tracks : {len(jobs)} in {elapsed:.1f}s "
          f"({elapsed / max(1, len(jobs)):.2f}s/track, {workers} workers)", flush=True)
    print(f"  ok / cached / skip / error : {counters.get('ok', 0)} / "
          f"{counters.get('cached', 0)} / {counters.get('skipped', 0)} / "
          f"{counters.get('error', 0)}", flush=True)
    print(f"  kits in manifest : {len(kits)}", flush=True)
    print(f"  phrases          : {n_phrases}", flush=True)
    print(f"  opus files       : {n_files} "
          f"({total_bytes / 1e6:.1f} MB total, "
          f"{total_bytes / max(1, n_files) / 1e3:.1f} kB/file)", flush=True)
    print(f"  new files written: {files_written} ({bytes_written / 1e6:.1f} MB)", flush=True)
    print(f"  max bar align err: {max_align_ms:.4f} ms (limit 5 ms)", flush=True)
    if rejected:
        print("  rejected windows : " + ", ".join(f"{k}={v}" for k, v in
                                                   sorted(rejected.items())), flush=True)
    skip_reasons = Counter(str(r.get("reason", "")) for r in results
                           if r.get("status") != "ok" and r.get("status") != "cached")
    if skip_reasons:
        print("  skipped tracks   : " + ", ".join(f"{k}={v}" for k, v in
                                                   sorted(skip_reasons.items())), flush=True)
    print(f"  manifest         : {manifest_path} "
          f"({manifest_path.stat().st_size / 1e6:.2f} MB)", flush=True)


def _manifest(args: argparse.Namespace, kits: Dict[str, dict]) -> dict:
    return {
        "version": 1,
        "bars": int(args.bars),
        "sample_rate": DEFAULT_SR,
        "bitrate": str(args.bitrate),
        "kits": [kits[tid] for tid in sorted(kits)],
    }


if __name__ == "__main__":
    main()
