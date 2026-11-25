#!/usr/bin/env python3
"""M0 reconstruction + QC for the Wub kit engine.

The kit corpus (see `scripts/build_kit_corpus.py`) stores 4-bar stem phrases as
mono Ogg Opus. This tool does two jobs:

``decode``
    Reassemble every selected kit back into a continuous mix
    (``<out>/<track_id>_recon.wav``) plus a 4-channel stem file
    (``<out>/<track_id>_stems.wav``) for inspection.  Phrases are concatenated
    in ``start_bar`` order with a de-click fade at the joins.

``compare``
    Decode every phrase's Opus stems and measure them against the *original*
    source WAVs in ``--stems``.  Reports per-stem/per-phrase Pearson
    correlation, SNR and RMS delta, the measured codec/resampler lag, window
    overlap and duration errors, and flags anything below the QC thresholds.

Facts discovered while building this (verified on the local corpus):

* Ogg Opus *always* decodes at 48 kHz.  ``sf.info()`` on these files reports
  ``samplerate: 48000`` even though the manifest says ``sample_rate: 44100``
  and the source stems are 44.1 kHz.  ``compare`` therefore measures at the
  decode-native rate and resamples the reference *up*, so only one resampling
  step sits between the original PCM and the reported number.
  A player must use ``buffer.sampleRate`` / ``sf.info().samplerate`` and never
  assume 44100 when converting frames to seconds.
* The 44.1 k <-> 48 k conversion here is exact rational 147:160, implemented
  as a polyphase Kaiser-windowed sinc (no scipy).  Cross-checked against
  ffmpeg's swresample: identical correlation/SNR to 4 decimal places.

Only numpy / soundfile / json are required.  ``ffmpeg`` is used only as a
fallback decoder (or explicitly with ``--decoder ffmpeg``); it is already a
project dependency through ``scripts/audio_encode.py``.

Example
-------
    python3 scripts/kit_reconstruct.py --mode decode
    python3 scripts/kit_reconstruct.py --mode compare --track-id <tid>
    python3 scripts/kit_reconstruct.py --mode compare --fail-on-error \
        --report /tmp/kits_test/kit_qc.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - hard dependency
    raise SystemExit(f"kit_reconstruct.py needs soundfile: {exc}")

# Allow `python3 scripts/kit_reconstruct.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STEM_ORDER: Tuple[str, ...] = ("drums", "bass", "other", "vocals")
DEFAULT_MANIFEST = "/tmp/kits_test/kit_manifest.json"
DEFAULT_KITS_DIR = "/tmp/kits_test"
DEFAULT_STEMS = "data/stems"
DEFAULT_PROCESSED = "data/processed"
DEFAULT_OUT = "/tmp/kit_recon"

QC_MIN_CORRELATION = 0.95
QC_MIN_SNR_DB = 20.0
QC_MAX_ONSET_MS = 5.0
QC_MAX_DURATION_MS = 10.0
QC_MAX_TIMEBASE_MS = 20.0
QC_QUIET_DBFS = -40.0
CLICK_RATIO_MAX = 4.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _db(x: float) -> float:
    """Amplitude ratio -> dB, -inf for <= 0."""
    if x <= 0.0 or not math.isfinite(x):
        return float("-inf")
    return 20.0 * math.log10(x)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _peak(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def _dc(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x, dtype=np.float64))


def _sanitize(obj: Any) -> Any:
    """Replace NaN/inf with None so the report is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _split_ids(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for raw in values or []:
        for part in str(raw).replace(";", ",").split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


# ---------------------------------------------------------------------------
# resampling (polyphase windowed sinc, no scipy)
# ---------------------------------------------------------------------------

def _kaiser(u: np.ndarray, beta: float = 8.6) -> np.ndarray:
    """Continuous Kaiser window evaluated at ``u`` in [-1, 1]."""
    arg = beta * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    return np.i0(arg) / np.i0(beta)


def resample_sinc(
    x: np.ndarray,
    sr_in: int,
    sr_out: int,
    half: int = 16,
    rolloff: float = 0.95,
) -> np.ndarray:
    """Rational polyphase resampler (exact for sr pairs with a small gcd).

    44.1 k <-> 48 k is 147:160, so the phase pattern is exact and no phase
    quantisation noise is introduced.  Edges are padded with the first/last
    sample so a phrase that starts on a transient is not ramped to zero.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    if x.size == 0 or int(sr_in) == int(sr_out):
        return x.copy()

    g = math.gcd(int(sr_in), int(sr_out))
    L = int(sr_out) // g  # upsample factor
    M = int(sr_in) // g  # downsample factor
    n_out = int(round(x.size * L / M))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)

    K = max(2, int(half))
    phases = np.arange(L)
    taps = np.arange(-K + 1, K + 1)
    # n = distance from the interpolation point, in upsampled samples
    n = phases[:, None] - taps[None, :] * L
    fc = 0.5 * min(1.0 / L, 1.0 / M) * rolloff  # cutoff, cycles / upsampled sample
    # An ideal lowpass of cutoff fc has h[n] = 2*fc*sinc(2*fc*n); the extra
    # factor of two is what puts the -6 dB point at fc instead of fc/2.
    h = 2.0 * fc * np.sinc(2.0 * fc * n) * _kaiser(n / float(K * L))
    # Per-phase DC normalisation kills passband ripple.
    dc = h.sum(axis=1, keepdims=True)
    h = np.divide(h, dc, out=np.zeros_like(h), where=np.abs(dc) > 1e-12)

    pos = np.arange(n_out, dtype=np.int64) * M
    base = pos // L
    phase = (pos % L).astype(np.int64)

    xp = np.concatenate(
        [np.full(K, x[0], dtype=np.float64), x, np.full(K, x[-1], dtype=np.float64)]
    )
    idx = base[:, None] + taps[None, :] + K
    np.clip(idx, 0, xp.size - 1, out=idx)
    y = np.einsum("nk,nk->n", h[phase], xp[idx])
    return y


# ---------------------------------------------------------------------------
# audio I/O
# ---------------------------------------------------------------------------


@dataclass
class Decoded:
    audio: np.ndarray
    sample_rate: int
    backend: str
    source_rate: int
    resampled: bool

    @property
    def duration(self) -> float:
        return self.audio.size / float(self.sample_rate) if self.sample_rate else 0.0


def _read_ffmpeg(path: Path, target_rate: Optional[int]) -> Tuple[np.ndarray, int]:
    try:
        from scripts.audio_encode import ffmpeg_exe  # type: ignore
    except Exception:  # pragma: no cover - script executed from inside scripts/
        try:
            from audio_encode import ffmpeg_exe  # type: ignore
        except Exception:
            def ffmpeg_exe() -> str:  # type: ignore
                return "ffmpeg"

    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(int(target_rate or 48000)),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path}: {proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    data = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
    return data, int(target_rate or 48000)


def decode_audio(
    path: Path,
    target_rate: Optional[int] = None,
    decoder: str = "auto",
) -> Decoded:
    """Decode any supported audio file to mono float64.

    ``target_rate=None`` keeps the decoder-native rate.  ``decoder`` is one of
    ``auto`` / ``soundfile`` / ``ffmpeg``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if decoder in ("auto", "soundfile"):
        try:
            with sf.SoundFile(str(path)) as handle:
                src_rate = int(handle.samplerate)
                data = handle.read(dtype="float64", always_2d=True)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            rate = src_rate
            resampled = False
            if target_rate and int(target_rate) != src_rate:
                mono = resample_sinc(mono, src_rate, int(target_rate))
                rate, resampled = int(target_rate), True
            return Decoded(mono, rate, "soundfile", src_rate, resampled)
        except Exception:
            if decoder == "soundfile":
                raise

    mono, rate = _read_ffmpeg(path, target_rate)
    return Decoded(mono, rate, "ffmpeg", rate, False)


def read_source_window(
    path: Path,
    start_sec: float,
    duration_sec: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Read ``[start_sec, start_sec+duration)`` from a (possibly stereo) WAV.

    Returns ``(left, right, sample_rate, frames_read)``.  Mono files give
    ``left == right``.
    """
    with sf.SoundFile(str(path)) as handle:
        sr = int(handle.samplerate)
        start = int(round(max(0.0, start_sec) * sr))
        want = int(round(max(0.0, duration_sec) * sr))
        start = min(start, len(handle))
        handle.seek(start)
        data = handle.read(frames=want, dtype="float64", always_2d=True)
    if data.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, sr, 0
    left = data[:, 0]
    right = data[:, 1] if data.shape[1] > 1 else data[:, 0]
    return left, right, sr, data.shape[0]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _norms(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    n = min(x.size, y.size)
    xc = x[:n] - x[:n].mean()
    yc = y[:n] - y[:n].mean()
    return (
        float(np.sqrt(np.dot(xc, xc))),
        float(np.sqrt(np.dot(yc, yc))),
        float(np.dot(xc, yc)),
    )


def best_lag(
    ref: np.ndarray,
    dec: np.ndarray,
    sample_rate: int,
    max_lag_ms: float = 50.0,
) -> Tuple[int, float]:
    """Lag (samples) that maximises normalised cross-correlation.

    Positive lag means the decoded signal is *later* than the reference.
    """
    if ref.size < 16 or dec.size < 16:
        return 0, 0.0
    max_lag = int(round(sample_rate * max_lag_ms / 1000.0))
    max_lag = max(1, min(max_lag, min(ref.size, dec.size) - 1))
    n = ref.size + dec.size + max_lag + 8
    nfft = 1 << int(math.ceil(math.log2(n)))
    a = np.fft.rfft(ref, nfft)
    b = np.fft.rfft(dec, nfft)
    cc = np.fft.irfft(np.conj(a) * b, nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = cc[lags % nfft]
    norm = 1.0
    nr, nd, _ = _norms(ref, dec)
    if nr > 0 and nd > 0:
        norm = nr * nd
    vals = vals / norm
    k = int(np.argmax(vals))
    return int(lags[k]), float(vals[k])


def _align(ref: np.ndarray, dec: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        n = min(ref.size, dec.size - lag)
        return ref[:n], dec[lag : lag + n]
    if lag < 0:
        n = min(ref.size + lag, dec.size)
        return ref[-lag : -lag + n], dec[:n]
    n = min(ref.size, dec.size)
    return ref[:n], dec[:n]


def _snr(ref: np.ndarray, dec: np.ndarray) -> float:
    r = ref - ref.mean()
    d = dec - dec.mean()
    p = float(np.dot(r, r))
    e = float(np.dot(d - r, d - r))
    if p <= 0:
        return 0.0
    return 10.0 * math.log10(p / e) if e > 0 else float("inf")


def quick_corr(ref: np.ndarray, dec: np.ndarray) -> float:
    """Zero-lag Pearson on the common prefix (no FFT, no alignment search)."""
    n = min(ref.size, dec.size)
    if n < 64:
        return 0.0
    r = ref[:n]
    d = dec[:n]
    r = r - r.mean()
    d = d - d.mean()
    a = float(np.dot(r, r))
    b = float(np.dot(d, d))
    if a <= 0 or b <= 0:
        return 0.0
    return float(np.dot(r, d) / math.sqrt(a * b))


def pair_metrics(
    ref: np.ndarray,
    dec: np.ndarray,
    sample_rate: int,
    max_lag_ms: float = 50.0,
) -> Dict[str, Any]:
    """Correlation / SNR / level metrics of ``dec`` against ``ref``."""
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    dec = np.asarray(dec, dtype=np.float64).reshape(-1)
    lag, _ = best_lag(ref, dec, sample_rate, max_lag_ms)
    r, d = _align(ref, dec, lag)

    if r.size < 8:
        return {"status": "too_short", "frames": int(min(ref.size, dec.size))}

    rc = r - r.mean()
    dc_ = d - d.mean()
    nr = float(np.sqrt(np.dot(rc, rc)))
    nd = float(np.sqrt(np.dot(dc_, dc_)))
    dot = float(np.dot(rc, dc_))
    corr = dot / (nr * nd) if nr > 0 and nd > 0 else 0.0

    resid = dc_ - rc
    p_ref = float(np.dot(rc, rc))
    p_res = float(np.dot(resid, resid))
    snr = 10.0 * math.log10(p_ref / p_res) if p_res > 0 and p_ref > 0 else float("inf")

    # Optimal least-squares gain, and the SNR once that gain is removed.
    gain = dot / float(np.dot(dc_, dc_)) if np.dot(dc_, dc_) > 0 else 1.0
    resid_g = gain * dc_ - rc
    p_res_g = float(np.dot(resid_g, resid_g))
    snr_g = (
        10.0 * math.log10(p_ref / p_res_g)
        if p_res_g > 0 and p_ref > 0
        else float("inf")
    )

    rms_ref = _rms(r)
    rms_dec = _rms(d)
    core = max(0, int(round(0.010 * sample_rate)))
    if r.size > 4 * core:
        snr_core = _snr(r[core:-core], d[core:-core])
    else:
        snr_core = None

    # One spectrum pair drives the band and 12 kHz-limited diagnostics.
    R = np.fft.rfft(rc)
    D = np.fft.rfft(dc_)
    freqs = np.fft.rfftfreq(rc.size, 1.0 / sample_rate)
    keep = (freqs >= 20.0) & (freqs <= 12000.0)
    if np.any(keep) and not np.all(keep):
        band_ref = np.fft.irfft(R * keep, rc.size)
        band_dec = np.fft.irfft(D * keep, rc.size)
        snr_12k = _snr(band_ref, band_dec)
    else:
        snr_12k = None
    bands: Dict[str, Any] = {}
    total = float(np.sum(np.abs(R) ** 2)) or 1.0
    for lo, hi in (
        (20, 100),
        (100, 300),
        (300, 1000),
        (1000, 4000),
        (4000, 12000),
        (12000, 20000),
    ):
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            continue
        p = float(np.sum(np.abs(R[m]) ** 2))
        e = float(np.sum(np.abs(D[m] - R[m]) ** 2))
        bands[f"{lo}-{hi}"] = {
            "snr_db": 10.0 * math.log10(p / e) if e > 0 and p > 0 else None,
            "energy_frac": p / total,
        }

    return {
        "status": "ok",
        "frames": int(r.size),
        "lag_samples": int(lag),
        "lag_ms": 1000.0 * lag / sample_rate,
        "correlation": corr,
        "correlation_zero_lag": quick_corr(ref, dec),
        "snr_db": snr,
        "snr_db_gain_matched": snr_g,
        "snr_db_core": snr_core,
        "snr_db_12k": snr_12k,
        "bands": bands,
        "gain_db": _db(abs(gain)) if gain else float("-inf"),
        "rms_ref_db": _db(rms_ref),
        "rms_dec_db": _db(rms_dec),
        "rms_delta_db": _db(rms_dec) - _db(rms_ref),
        "residual_rms_dbfs": _db(_rms(resid)),
        "peak_ref": _peak(r),
        "peak_dec": _peak(d),
        "dc_ref": _dc(r),
        "dc_dec": _dc(d),
    }


# ---------------------------------------------------------------------------
# manifest / analysis plumbing
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "kits" not in manifest or not isinstance(manifest["kits"], list):
        raise SystemExit(f"{path}: manifest has no 'kits' list")
    return manifest


def select_kits(
    manifest: Dict[str, Any], wanted: Sequence[str], default_limit: int = 3
) -> Tuple[List[Dict[str, Any]], List[str]]:
    kits = manifest["kits"]
    if not wanted:
        return list(kits[:default_limit]), []
    out: List[Dict[str, Any]] = []
    matched: List[str] = []
    for kit in kits:
        tid = str(kit.get("track_id", ""))
        title = str(kit.get("title", ""))
        for w in wanted:
            if tid == w or (len(w) >= 6 and tid.startswith(w)) or title == w:
                out.append(kit)
                matched.append(w)
                break
    missing = [w for w in wanted if w not in matched]
    return out, missing


def load_analysis(processed: Path, track_id: str) -> Optional[Dict[str, Any]]:
    for cand in (processed / f"{track_id}.json", processed / f"track_{track_id}.json"):
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return None
    return None


def sorted_phrases(kit: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(
        kit.get("phrases", []),
        key=lambda p: (float(p.get("start_bar", 0.0)), float(p.get("start_sec", 0.0))),
    )


def resolve_phrase_file(kits_dir: Path, phrase: Dict[str, Any], stem: str) -> Optional[Path]:
    entry = (phrase.get("stems") or {}).get(stem)
    if not entry:
        return None
    rel = entry.get("file") if isinstance(entry, dict) else str(entry)
    if not rel:
        return None
    path = Path(rel)
    return path if path.is_absolute() else kits_dir / path


# ---------------------------------------------------------------------------
# join handling
# ---------------------------------------------------------------------------


def concat_phrases(
    blocks: Sequence[np.ndarray],
    sample_rate: int,
    fade_ms: float = 5.0,
    join_fade: str = "auto",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Concatenate phrase audio, de-clicking joins at the boundaries.

    ``join_fade``:
      ``auto``   - only fade when the raw step is a real discontinuity
      ``always`` - always apply a raised-cosine notch of ``fade_ms`` per side
      ``never``  - raw concatenation (bit-exact for sample-accurate cuts)

    A join is flagged as a click when the sample step across the boundary is
    more than ``CLICK_RATIO_MAX`` times the 99.5th percentile step of the
    surrounding 40 ms (the join sample itself excluded, so a legitimate
    kick transient on the bar line is not mistaken for a click).
    """
    blocks = [np.asarray(b, dtype=np.float64).reshape(-1) for b in blocks]
    if not blocks:
        return np.zeros(0, dtype=np.float64), []

    n_fade = max(1, int(round(sample_rate * max(0.0, fade_ms) / 1000.0)))
    total = int(sum(b.size for b in blocks))
    y = np.zeros(total, dtype=np.float64)
    joins: List[Dict[str, Any]] = []
    offset = 0
    for i, block in enumerate(blocks):
        y[offset : offset + block.size] = block
        if i > 0:
            joins.append({"index": i, "sample": int(offset)})
        offset += block.size

    for join in joins:
        b = int(join["sample"])
        if b <= 0 or b >= y.size:
            join.update({"click_ratio_before": 0.0, "click_ratio_after": 0.0})
            continue
        lo20 = max(1, b - int(0.020 * sample_rate))
        hi20 = min(y.size, b + int(0.020 * sample_rate))
        local = np.abs(np.diff(y[lo20:hi20]))
        k = b - 1 - lo20
        if 0 <= k < local.size:
            local = np.delete(local, k)
        local_p = float(np.percentile(local, 99.5)) if local.size else 0.0
        step = abs(float(y[b]) - float(y[b - 1]))
        ratio = step / local_p if local_p > 0 else 0.0
        apply_fade = join_fade == "always" or (
            join_fade == "auto" and ratio > CLICK_RATIO_MAX and step > 3e-3
        )
        if apply_fade:
            lo = max(0, b - n_fade)
            hi = min(y.size, b + n_fade)
            d = np.abs(np.arange(lo, hi) - b).astype(np.float64) / max(1, n_fade)
            win = 0.5 - 0.5 * np.cos(np.pi * np.clip(d, 0.0, 1.0))
            y[lo:hi] *= win
        post_step = abs(float(y[b]) - float(y[b - 1]))
        join.update(
            {
                "step": step,
                "step_dbfs": _db(step),
                "local_step_p995": local_p,
                "click_ratio_before": ratio,
                "faded": bool(apply_fade),
                "click_ratio_after": post_step / local_p if local_p > 0 else 0.0,
            }
        )
    return y, joins


# ---------------------------------------------------------------------------
# decode mode
# ---------------------------------------------------------------------------


def _decode_phrase_stem(
    kits_dir: Path,
    phrase: Dict[str, Any],
    stem: str,
    target_rate: int,
    decoder: str,
    cache: Dict[Path, Decoded],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    path = resolve_phrase_file(kits_dir, phrase, stem)
    info: Dict[str, Any] = {"file": str(path) if path else None, "stem": stem}
    if path is None:
        info["status"] = "missing_reference"
        return None, info
    if not path.exists():
        info["status"] = "missing_file"
        return None, info
    decoded = cache.get(path)
    if decoded is None:
        decoded = decode_audio(path, target_rate=target_rate, decoder=decoder)
        cache[path] = decoded
    info.update(
        {
            "status": "ok",
            "decoded_rate": decoded.sample_rate,
            "backend": decoded.backend,
            "peak": _peak(decoded.audio),
            "rms_db": _db(_rms(decoded.audio)),
            "dc": _dc(decoded.audio),
            "seconds": decoded.duration,
        }
    )
    return decoded.audio, info


def run_decode(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    kits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_rate = int(manifest.get("sample_rate") or 44100)
    cache: Dict[Path, Decoded] = {}
    summary: Dict[str, Any] = {
        "mode": "decode",
        "manifest": str(Path(args.manifest)),
        "target_rate": target_rate,
        "fade_ms": float(args.fade_ms),
        "join_fade": args.join_fade,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kits": [],
    }

    for kit in kits:
        tid = str(kit.get("track_id"))
        phrases = sorted_phrases(kit)
        kit_row: Dict[str, Any] = {
            "track_id": tid,
            "title": kit.get("title"),
            "bpm": kit.get("bpm"),
            "key": kit.get("key"),
            "bars": manifest.get("bars", 4),
            "n_phrases": len(phrases),
            "phrases": [],
            "stems": {},
        }
        if not phrases:
            kit_row["status"] = "no_phrases"
            summary["kits"].append(kit_row)
            continue

        stem_audio: Dict[str, np.ndarray] = {}
        for stem in STEM_ORDER:
            blocks: List[np.ndarray] = []
            phrase_rows: List[Dict[str, Any]] = []
            for phrase in phrases:
                audio, info = _decode_phrase_stem(
                    Path(args.kits_dir), phrase, stem, target_rate, args.decoder, cache
                )
                info["phrase"] = phrase.get("id")
                dur = float(phrase.get("duration_sec", 0.0)) or (
                    float(phrase.get("end_sec", 0.0))
                    - float(phrase.get("start_sec", 0.0))
                )
                phrase_rows.append(info)
                if audio is not None:
                    # Opus decodes to 48 kHz, so a 44.1 kHz block comes back
                    # +-1 sample off.  Snap each phrase back to its declared
                    # length, otherwise the error accumulates across joins.
                    want = int(round(dur * target_rate))
                    if want > 0 and abs(audio.size - want) <= max(
                        4, int(0.002 * target_rate)
                    ):
                        delta = int(audio.size - want)
                        if delta > 0:
                            audio = audio[:want]
                        elif delta < 0:
                            audio = np.pad(audio, (0, -delta))
                        info["snapped_samples"] = delta
                    blocks.append(audio)
                else:
                    # Keep the timeline intact: a missing phrase becomes silence
                    # of the declared length instead of shifting the rest.
                    blocks.append(np.zeros(max(0, int(round(dur * target_rate)))))
                    info["spliced_silence"] = True
            if not blocks:
                kit_row["stems"][stem] = {"status": "missing", "phrases": phrase_rows}
                continue

            y, joins = concat_phrases(blocks, target_rate, args.fade_ms, args.join_fade)
            stem_audio[stem] = y
            kit_row["stems"][stem] = {
                "status": "ok",
                "seconds": y.size / float(target_rate),
                "frames": int(y.size),
                "peak": _peak(y),
                "rms_db": _db(_rms(y)),
                "dc": _dc(y),
                "p999_step": (
                    float(np.percentile(np.abs(np.diff(y)), 99.9)) if y.size > 1 else 0.0
                ),
                "joins": joins,
                "phrases": phrase_rows,
            }

        kit_row["phrases"] = [
            {
                "id": p.get("id"),
                "start_bar": p.get("start_bar"),
                "start_sec": p.get("start_sec"),
                "seconds": float(p.get("duration_sec", 0.0)),
            }
            for p in phrases
        ]

        # ---- mix down -----------------------------------------------------
        if not stem_audio:
            kit_row["status"] = "no_audio"
            summary["kits"].append(kit_row)
            continue

        lengths = {len(v) for v in stem_audio.values()}
        n = max(lengths)
        mix = np.zeros(n, dtype=np.float64)
        for stem in STEM_ORDER:
            if stem in stem_audio:
                mix[: len(stem_audio[stem])] += stem_audio[stem]
        peak = _peak(mix)
        gain = 1.0
        clipped = 0
        if args.peak_mode == "normalise" and peak > 0.999:
            gain = 0.999 / peak
            out = mix * gain
        elif args.peak_mode == "clip":
            out = np.clip(mix, -0.999, 0.999)
            clipped = int(np.count_nonzero(np.abs(mix) > 0.999))
        else:
            out = mix
        kit_row["master"] = {
            "peak_before_norm": peak,
            "peak_mode": args.peak_mode,
            "normalised": bool(gain != 1.0),
            "gain": gain,
            "gain_db": _db(gain),
            "clipped_samples": clipped,
            "peak_after_norm": _peak(out),
            "rms_db": _db(_rms(out)),
            "dc": _dc(out),
            "seconds": n / float(target_rate),
            "frames": int(n),
            "len_mismatch": len(lengths) > 1,
        }

        recon_path = out_dir / f"{tid}_recon.wav"
        sf.write(str(recon_path), out.astype(np.float32), target_rate, subtype="FLOAT")
        kit_row["recon_path"] = str(recon_path)

        stems_path = out_dir / f"{tid}_stems.wav"
        chans = [stem_audio.get(s, np.zeros(n, dtype=np.float64)) for s in STEM_ORDER]
        stacked = np.stack([np.pad(c, (0, n - c.size)) for c in chans], axis=1)
        sf.write(str(stems_path), stacked.astype(np.float32), target_rate, subtype="FLOAT")
        kit_row["stems_path"] = str(stems_path)
        kit_row["stems_channel_order"] = list(STEM_ORDER)

        if args.split_stems:
            for idx, stem in enumerate(STEM_ORDER):
                mono = stacked[:, idx]
                sp = out_dir / f"{tid}_{stem}.wav"
                sf.write(str(sp), mono.astype(np.float32), target_rate, subtype="FLOAT")
                kit_row.setdefault("split_stems", {})[stem] = str(sp)

        # ---- optional fidelity check against the original stems -----------
        if args.check_source:
            kit_row["fidelity"] = _fidelity_vs_source(
                args, kit, phrases, mix, target_rate, manifest
            )
            kit_row["fidelity"]["master_gain_db"] = _db(gain) if gain else None

        summary["kits"].append(kit_row)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(_sanitize(summary), handle, indent=2, sort_keys=True)
        summary["report_path"] = str(report_path)
    return summary


def _fidelity_vs_source(
    args: argparse.Namespace,
    kit: Dict[str, Any],
    phrases: List[Dict[str, Any]],
    mix: np.ndarray,
    sample_rate: int,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct the reference from the source WAVs and compare to the mix."""
    tid = str(kit.get("track_id"))
    ref_blocks: List[np.ndarray] = []
    per_stem: Dict[str, Any] = {}
    for stem in STEM_ORDER:
        src = Path(args.stems) / tid / f"{stem}.wav"
        if not src.exists():
            per_stem[stem] = {"status": "missing_source"}
            continue
        blocks = []
        for phrase in phrases:
            start = float(phrase.get("start_sec", 0.0))
            dur = float(phrase.get("duration_sec", 0.0)) or (
                float(phrase.get("end_sec", 0.0)) - start
            )
            left, right, sr, _ = read_source_window(src, start, dur)
            mono = 0.5 * (left + right)
            if sr != sample_rate and mono.size:
                mono = resample_sinc(mono, sr, sample_rate)
            blocks.append(mono)
        if not blocks:
            per_stem[stem] = {"status": "empty"}
            continue
        y, _ = concat_phrases(blocks, sample_rate, 0.0, "never")
        ref_blocks.append(y)
        per_stem[stem] = {"status": "ok", "seconds": y.size / float(sample_rate)}

    if not ref_blocks:
        return {"status": "no_source"}
    n = max(len(b) for b in ref_blocks)
    ref = np.zeros(n, dtype=np.float64)
    for b in ref_blocks:
        ref[: len(b)] += b
    peak = _peak(ref)
    if peak > 0.999:
        ref *= 0.999 / peak
    m = mix[:n] if mix.size >= n else np.pad(mix, (0, n - mix.size))
    metrics = pair_metrics(ref, m, sample_rate)
    return {
        "status": "ok",
        "reference_peak": peak,
        "reference_seconds": n / float(sample_rate),
        "stems": per_stem,
        **metrics,
    }


# ---------------------------------------------------------------------------
# compare mode
# ---------------------------------------------------------------------------


def _downmix_pair(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    target_rate: int,
) -> Dict[str, np.ndarray]:
    """Resample L/R once, then build the three candidate downmixes."""
    if sr != target_rate and left.size:
        left = resample_sinc(left, sr, target_rate)
        right = resample_sinc(right, sr, target_rate)
    return {
        "mean": 0.5 * (left + right),
        "left": left,
        "right": right,
    }


def compare_phrase(
    args: argparse.Namespace,
    kit: Dict[str, Any],
    phrase: Dict[str, Any],
    analysis: Optional[Dict[str, Any]],
    bars: int,
    cache: Dict[Path, Decoded],
) -> Dict[str, Any]:
    start_sec = float(phrase.get("start_sec", 0.0))
    end_sec = float(phrase.get("end_sec", 0.0))
    duration = float(phrase.get("duration_sec", 0.0)) or (end_sec - start_sec)
    start_bar = int(phrase.get("start_bar", 0))
    tid = str(kit.get("track_id"))

    row: Dict[str, Any] = {
        "id": phrase.get("id"),
        "section": phrase.get("section"),
        "start_bar": start_bar,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration,
        "manifest_duration_ms": duration * 1000.0,
        "stems": {},
    }

    # ---- manifest vs analysis tatum grid -------------------------------
    tatum_start = tatum_end = None
    if analysis and analysis.get("tatums"):
        tatums = analysis["tatums"]
        i0 = 16 * start_bar
        i1 = 16 * (start_bar + bars)
        if 0 <= i0 < len(tatums):
            tatum_start = float(tatums[i0]["start"])
        if 0 <= i1 < len(tatums):
            tatum_end = float(tatums[i1]["start"])
        row["tatum_start_sec"] = tatum_start
        row["tatum_end_sec"] = tatum_end
        row["tatum_index"] = i0
        row["tatum_count"] = len(tatums)
        if tatum_start is not None:
            row["manifest_vs_tatum_ms"] = (start_sec - tatum_start) * 1000.0
            row["onset_ok"] = (
                abs(start_sec - tatum_start) * 1000.0 <= args.max_onset_ms
            )
        else:
            row["status_note"] = "tatum index out of range"
    else:
        row["status_note"] = "analysis JSON missing"

    # ---- per-stem comparison -------------------------------------------
    stem_rows: Dict[str, Any] = {}
    for stem in STEM_ORDER:
        entry: Dict[str, Any] = {"stem": stem}
        opus_path = resolve_phrase_file(Path(args.kits_dir), phrase, stem)
        src_path = Path(args.stems) / tid / f"{stem}.wav"
        if opus_path is None or not opus_path.exists():
            entry["status"] = "missing_opus"
            stem_rows[stem] = entry
            continue
        if not src_path.exists():
            entry["status"] = "missing_source"
            stem_rows[stem] = entry
            continue

        decoded = cache.get(opus_path)
        if decoded is None:
            rate = None if str(args.compare_rate) == "native" else int(args.compare_rate)
            decoded = decode_audio(opus_path, target_rate=rate, decoder=args.decoder)
            cache[opus_path] = decoded
        entry["decoded_rate"] = decoded.sample_rate
        entry["opus_native_rate"] = decoded.source_rate
        entry["decoded_seconds"] = decoded.duration
        entry["backend"] = decoded.backend
        entry["opus_bytes"] = opus_path.stat().st_size
        entry["duration_error_ms"] = (decoded.duration - duration) * 1000.0
        entry["duration_ok"] = abs(entry["duration_error_ms"]) <= args.max_duration_ms

        left, right, src_rate, frames = read_source_window(src_path, start_sec, duration)
        entry["source_rate"] = src_rate
        entry["source_frames"] = int(frames)
        entry["source_seconds"] = frames / float(src_rate) if src_rate else 0.0
        if frames == 0:
            entry["status"] = "empty_source_window"
            stem_rows[stem] = entry
            continue

        variants = _downmix_pair(left, right, src_rate, decoded.sample_rate)
        chosen = args.downmix if args.downmix in variants else "mean"
        entry["downmix"] = chosen
        metrics = pair_metrics(
            variants[chosen], decoded.audio, decoded.sample_rate, args.max_lag_ms
        )
        # The other candidates only need a zero-lag Pearson: timing is already
        # established by the chosen variant, this just identifies the builder's
        # downmix convention (mean / left / right).
        entry["downmix_variants"] = {
            name: quick_corr(variants[name], decoded.audio)
            for name in ("mean", "left", "right")
        }
        entry["downmix_variant_best"] = max(
            entry["downmix_variants"], key=entry["downmix_variants"].get
        )
        entry.update(metrics)
        if metrics.get("status") == "ok":
            entry["onset_ok"] = abs(metrics["lag_ms"]) <= args.max_onset_ms
            corr = metrics["correlation"]
            snr = metrics["snr_db"]
            entry["qc_ok"] = bool(
                corr >= args.min_correlation and snr >= args.min_snr
            )
            # A window that is essentially silent (breakdown / drum dropout)
            # cannot reach 20 dB SNR through a lossy codec: the Opus noise
            # floor lands around -45 dBFS.  That is a property of the window,
            # not a builder defect.  A wrong-stem bug would raise rms_dec, so
            # requiring *both* levels to be quiet keeps this safe.
            loudest = max(
                metrics.get("rms_ref_db", 0.0), metrics.get("rms_dec_db", 0.0)
            )
            if loudest < QC_QUIET_DBFS:
                entry["qc_ok"] = True
                entry["exempt_reason"] = "quiet_window"
        else:
            entry["qc_ok"] = False
        stem_rows[stem] = entry

    row["stems"] = stem_rows
    ok = [e for e in stem_rows.values() if e.get("status") == "ok"]
    if ok:
        row["min_correlation"] = min(e["correlation"] for e in ok)
        row["min_snr_db"] = min(e["snr_db"] for e in ok)
        row["max_abs_lag_ms"] = max(abs(e["lag_ms"]) for e in ok)
    else:
        row["status"] = "no_comparable_stems"
    row["qc_ok"] = bool(stem_rows) and all(e.get("qc_ok") for e in stem_rows.values())
    return row


def run_compare(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    kits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = int(manifest.get("bars") or 4)
    cache: Dict[Path, Decoded] = {}
    report: Dict[str, Any] = {
        "mode": "compare",
        "manifest": str(Path(args.manifest)),
        "kits_dir": str(Path(args.kits_dir)),
        "stems_dir": str(Path(args.stems)),
        "processed_dir": str(Path(args.processed)),
        "bars": bars,
        "thresholds": {
            "min_correlation": args.min_correlation,
            "min_snr_db": args.min_snr,
            "max_onset_ms": args.max_onset_ms,
            "max_duration_ms": args.max_duration_ms,
            "max_timebase_ms": QC_MAX_TIMEBASE_MS,
        },
        "downmix": args.downmix,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kits": [],
        "failures": [],
        "warnings": [],
    }

    all_rows: List[Dict[str, Any]] = []
    for kit in kits:
        tid = str(kit.get("track_id"))
        phrases = sorted_phrases(kit)
        analysis = load_analysis(Path(args.processed), tid)
        kit_row: Dict[str, Any] = {
            "track_id": tid,
            "title": kit.get("title"),
            "bpm": kit.get("bpm"),
            "key": kit.get("key"),
            "style": kit.get("style"),
            "n_phrases": len(phrases),
            "analysis_found": bool(analysis),
            "phrases": [],
        }

        # ---- timebase: source WAV length vs analysis duration ------------
        # The analysis tatum grid and the demucs stems should share a timeline.
        # A mismatch means every window for this track is shifted.
        src_drums = Path(args.stems) / tid / "drums.wav"
        if analysis and src_drums.exists():
            try:
                info = sf.info(str(src_drums))
                src_dur = info.frames / float(info.samplerate)
                ana_dur = float(analysis.get("duration") or 0.0)
                delta_ms = (src_dur - ana_dur) * 1000.0 if ana_dur else None
                kit_row["timebase"] = {
                    "source_seconds": src_dur,
                    "analysis_seconds": ana_dur,
                    "delta_ms": delta_ms,
                }
                if delta_ms is not None and abs(delta_ms) > QC_MAX_TIMEBASE_MS:
                    report["warnings"].append(
                        {
                            "scope": "kit",
                            "kind": "timebase",
                            "track_id": tid,
                            "detail": (
                                f"source WAV is {delta_ms:+.1f} ms longer than the analysis "
                                "duration; windows cut from the source may be offset "
                                "relative to the tatum grid"
                            ),
                        }
                    )
            except Exception as exc:  # pragma: no cover
                kit_row["timebase"] = {"error": str(exc)}

        # ---- overlap / ordering check ----------------------------------
        overlaps: List[Dict[str, Any]] = []
        prev = None
        gaps: List[float] = []
        for phrase in phrases:
            if prev is not None:
                gap = float(phrase.get("start_sec", 0.0)) - float(prev.get("end_sec", 0.0))
                gaps.append(gap)
                if gap < -1e-6:
                    overlaps.append(
                        {
                            "prev": prev.get("id"),
                            "next": phrase.get("id"),
                            "overlap_ms": -gap * 1000.0,
                        }
                    )
            prev = phrase
        kit_row["window_overlaps"] = overlaps
        kit_row["window_gaps_ms"] = [g * 1000.0 for g in gaps]
        kit_row["has_overlaps"] = bool(overlaps)

        # ---- duration consistency -------------------------------------
        dur_errors: List[Dict[str, Any]] = []
        for phrase in phrases:
            d = float(phrase.get("duration_sec", 0.0))
            span = float(phrase.get("end_sec", 0.0)) - float(phrase.get("start_sec", 0.0))
            err = (d - span) * 1000.0
            if abs(err) > args.max_duration_ms:
                dur_errors.append(
                    {"id": phrase.get("id"), "error_ms": err, "declared_ms": d * 1000.0,
                     "span_ms": span * 1000.0}
                )
        kit_row["duration_errors"] = dur_errors

        for phrase in phrases:
            row = compare_phrase(args, kit, phrase, analysis, bars, cache)
            kit_row["phrases"].append(row)
            all_rows.append({"track_id": tid, **row})
            if not row.get("qc_ok"):
                reasons, failing = _failure_reasons(row, args)
                bad = [
                    e
                    for s, e in (row.get("stems") or {}).items()
                    if s in failing and e.get("status") == "ok"
                ]
                report["failures"].append(
                    {
                        "track_id": tid,
                        "phrase": row.get("id"),
                        "start_bar": row.get("start_bar"),
                        "min_correlation": (
                            min(e["correlation"] for e in bad) if bad else None
                        ),
                        "min_snr_db": min(e["snr_db"] for e in bad) if bad else None,
                        "failing_stems": failing,
                        "diagnosis": _diagnose(row, failing),
                        "reasons": reasons,
                    }
                )

        ok_rows = [r for r in kit_row["phrases"] if r.get("min_correlation") is not None]
        if ok_rows:
            kit_row["min_correlation"] = min(r["min_correlation"] for r in ok_rows)
            kit_row["min_snr_db"] = min(r["min_snr_db"] for r in ok_rows)
            kit_row["mean_correlation"] = float(
                np.mean([r["min_correlation"] for r in ok_rows])
            )
            kit_row["mean_snr_db"] = float(np.mean([r["min_snr_db"] for r in ok_rows]))

        # ---- tatum-grid agreement between manifest and analysis -----------
        tatum_deltas = [
            abs(r["manifest_vs_tatum_ms"])
            for r in kit_row["phrases"]
            if r.get("manifest_vs_tatum_ms") is not None
        ]
        if tatum_deltas:
            worst = max(tatum_deltas)
            kit_row["max_abs_tatum_delta_ms"] = worst
            kit_row["median_tatum_delta_ms"] = float(np.median(tatum_deltas))
            if worst > QC_MAX_ONSET_MS:
                report["warnings"].append(
                    {
                        "scope": "kit",
                        "kind": "tatum_grid",
                        "track_id": tid,
                        "detail": (
                            f"manifest start_bar/start_sec are off this analysis' tatum "
                            f"grid by up to {worst:.1f} ms "
                            f"(median {np.median(tatum_deltas):.1f} ms). The manifest was "
                            "probably built from a different analysis JSON; bar indices "
                            "are only valid against the analysis used at build time."
                        ),
                    }
                )
        if overlaps:
            report["failures"].append(
                {
                    "scope": "kit",
                    "track_id": tid,
                    "phrase": "<windows>",
                    "start_bar": None,
                    "failing_stems": [],
                    "reasons": [
                        f"{len(overlaps)} overlapping window(s), worst "
                        f"{max(o['overlap_ms'] for o in overlaps):.1f} ms"
                    ],
                    "overlaps": overlaps,
                }
            )
        if dur_errors:
            report["failures"].append(
                {
                    "scope": "kit",
                    "track_id": tid,
                    "phrase": "<durations>",
                    "start_bar": None,
                    "failing_stems": [],
                    "reasons": [
                        f"{len(dur_errors)} phrase(s) with duration_sec != end_sec-start_sec"
                    ],
                    "duration_errors": dur_errors,
                }
            )
        report["kits"].append(kit_row)
        # A full 400-kit bank has thousands of source stems; keeping every
        # decoded Opus buffer in this cache exhausts RAM.  The cache is only
        # useful within one kit (phrases reuse the same four source stems).
        cache.clear()

    report["aggregate"] = _aggregate(all_rows, args)
    report["n_failures"] = len(report["failures"])
    report_path = Path(args.report) if args.report else out_dir / "kit_compare.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize(report), handle, indent=2, sort_keys=True)
    report["report_path"] = str(report_path)
    return report


def _diagnose(row: Dict[str, Any], failing: List[str]) -> str:
    """Classify a failing phrase by likely cause (builder bug vs codec limit)."""
    stems = row.get("stems") or {}
    bad = [stems[s] for s in failing if (stems.get(s) or {}).get("status") == "ok"]
    if not bad:
        return "missing_audio"
    if row.get("onset_ok") is False:
        return "manifest_offset"
    lags = [abs(e.get("lag_ms", 0.0)) for e in bad]
    if lags and max(lags) > QC_MAX_ONSET_MS:
        return "decode_offset"
    deltas = [abs(e.get("rms_delta_db", 0.0)) for e in bad]
    if deltas and max(deltas) > 1.0:
        return "level_mismatch"
    corrs = [e.get("correlation", 0.0) for e in bad]
    if min(corrs) < 0.95:
        return "waveform_mismatch"
    # Correlation is high, the residual is just codec noise: report it as such.
    return "codec_limited"


def _failure_reasons(
    row: Dict[str, Any], args: argparse.Namespace
) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    failing: List[str] = []
    for stem, entry in (row.get("stems") or {}).items():
        status = entry.get("status")
        if status not in (None, "ok"):
            reasons.append(f"{stem}: {status}")
            failing.append(stem)
            continue
        if entry.get("qc_ok"):
            continue
        failing.append(stem)
        if entry.get("correlation", 1.0) < args.min_correlation:
            reasons.append(f"{stem} correlation {entry['correlation']:.4f}")
        if entry.get("snr_db", 1e9) < args.min_snr:
            resid = entry.get("residual_rms_dbfs")
            reasons.append(
                f"{stem} snr {entry['snr_db']:.1f} dB "
                f"(resid {resid:.0f} dBFS)" if resid is not None
                else f"{stem} snr {entry['snr_db']:.1f} dB"
            )
        if abs(entry.get("lag_ms", 0.0)) > args.max_onset_ms:
            reasons.append(f"{stem} decoded lag {entry['lag_ms']:.2f} ms")
    if row.get("onset_ok") is False:
        reasons.append("manifest start outside ±5 ms of tatum")
    if (
        row.get("max_abs_lag_ms") is not None
        and row["max_abs_lag_ms"] > args.max_onset_ms
    ):
        reasons.append(f"phrase lag {row['max_abs_lag_ms']:.2f} ms")
    if not reasons:
        reasons.append("qc flag")
    return reasons, failing


def _aggregate(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    per_stem: Dict[str, Any] = {}
    for stem in STEM_ORDER:
        corrs, snrs, deltas, lags, s12k = [], [], [], [], []
        for row in rows:
            entry = (row.get("stems") or {}).get(stem) or {}
            if entry.get("status") != "ok":
                continue
            corrs.append(entry["correlation"])
            snrs.append(entry["snr_db"])
            deltas.append(entry["rms_delta_db"])
            lags.append(entry["lag_ms"])
            if entry.get("snr_db_12k") is not None:
                s12k.append(entry["snr_db_12k"])
        if corrs:
            per_stem[stem] = {
                "n": len(corrs),
                "correlation_mean": float(np.mean(corrs)),
                "correlation_min": float(np.min(corrs)),
                "correlation_p05": float(np.percentile(corrs, 5)),
                "snr_db_mean": float(np.mean(snrs)),
                "snr_db_min": float(np.min(snrs)),
                "snr_db_12k_mean": float(np.mean(s12k)) if s12k else None,
                "rms_delta_db_mean": float(np.mean(deltas)),
                "rms_delta_db_max_abs": float(np.max(np.abs(deltas))),
                "lag_ms_mean": float(np.mean(lags)),
                "lag_ms_max_abs": float(np.max(np.abs(lags))),
            }
        else:
            per_stem[stem] = {"n": 0}
    corrs = [r["min_correlation"] for r in rows if r.get("min_correlation") is not None]
    snrs = [r["min_snr_db"] for r in rows if r.get("min_snr_db") is not None]
    return {
        "n_phrases": len(rows),
        "n_compared": len(corrs),
        "correlation_mean": float(np.mean(corrs)) if corrs else None,
        "correlation_min": float(np.min(corrs)) if corrs else None,
        "snr_db_mean": float(np.mean(snrs)) if snrs else None,
        "snr_db_min": float(np.min(snrs)) if snrs else None,
        "n_below_correlation": int(sum(c < args.min_correlation for c in corrs)),
        "n_below_snr": int(sum(s < args.min_snr for s in snrs)),
        "per_stem": per_stem,
    }


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------


def _print_decode(summary: Dict[str, Any]) -> None:
    print("")
    print(f"{'track_id':<18} {'phr':>4} {'sec':>7} {'peak':>6} {'rms dB':>7} "
          f"{'gain dB':>8} {'joins':>6} {'worst join':>12}  file")
    for kit in summary["kits"]:
        master = kit.get("master") or {}
        joins_all = []
        faded = 0
        for stem in STEM_ORDER:
            entry = kit.get("stems", {}).get(stem) or {}
            for join in entry.get("joins", []):
                joins_all.append(join)
                faded += 1 if join.get("faded") else 0
        worst = max(joins_all, key=lambda j: j.get("click_ratio_after", 0.0), default={})
        worst_txt = (
            f"{worst.get('click_ratio_after', 0.0):.1f}@{worst.get('step_dbfs', 0.0):.0f}"
            if worst
            else "-"
        )
        print(
            f"{kit['track_id']:<18} {kit['n_phrases']:>4} "
            f"{master.get('seconds', 0.0):>7.2f} {master.get('peak_after_norm', 0.0):>6.3f} "
            f"{master.get('rms_db', 0.0) or 0.0:>7.1f} "
            f"{master.get('gain_db', 0.0) or 0.0:>+8.2f} "
            f"{f'{faded}/{len(joins_all)}':>6} "
            f"{worst_txt:>12}  {kit.get('recon_path', '')}"
        )
        fid = kit.get("fidelity") or {}
        if fid.get("status") == "ok":
            print(
                f"{'':<18} fidelity vs source: r={fid['correlation']:.5f} "
                f"snr={fid['snr_db']:.1f} dB lag={fid['lag_ms']:+.2f} ms "
                f"rms_delta={fid['rms_delta_db']:+.2f} dB"
            )
    print("")


def _print_compare(report: Dict[str, Any]) -> None:
    print("")
    header = (
        f"{'track_id':<18} {'phrase':<26} {'bar':>4} {'stem':<7} "
        f"{'corr':>8} {'snr dB':>7} {'s12k':>6} {'lag ms':>7} {'dur ms':>7} {'rms dB':>7}"
    )
    print(header)
    print("-" * len(header))
    for kit in report["kits"]:
        for row in kit["phrases"]:
            for stem in STEM_ORDER:
                entry = (row.get("stems") or {}).get(stem) or {}
                if entry.get("status") != "ok":
                    print(
                        f"{kit['track_id']:<18} {str(row.get('id'))[:26]:<26} "
                        f"{row.get('start_bar', 0):>4} {stem:<7} {entry.get('status', '?'):>8}"
                    )
                    continue
                print(
                    f"{kit['track_id']:<18} {str(row.get('id'))[:26]:<26} "
                    f"{row.get('start_bar', 0):>4} {stem:<7} "
                    f"{entry['correlation']:>8.5f} {entry['snr_db']:>7.1f} "
                    f"{_fmt(entry.get('snr_db_12k'), 1):>6} "
                    f"{entry['lag_ms']:>+7.2f} {entry['duration_error_ms']:>+7.2f} "
                    f"{entry['rms_delta_db']:>+7.2f}"
                )
    agg = report["aggregate"]
    print("")
    print(
        f"phrases={agg['n_phrases']} compared={agg['n_compared']} "
        f"corr mean={_fmt(agg['correlation_mean'], 5)} min={_fmt(agg['correlation_min'], 5)} "
        f"snr mean={_fmt(agg['snr_db_mean'], 1)} min={_fmt(agg['snr_db_min'], 1)} dB"
    )
    for stem, st in agg["per_stem"].items():
        if not st.get("n"):
            print(f"  {stem:<7} n=0")
            continue
        print(
            f"  {stem:<7} n={st['n']:<3} corr min={st['correlation_min']:.5f} "
            f"mean={st['correlation_mean']:.5f} | snr min={st['snr_db_min']:>5.1f} "
            f"mean={st['snr_db_mean']:>5.1f} (12k {_fmt(st.get('snr_db_12k_mean'), 1)}) | "
            f"rms {st['rms_delta_db_mean']:+.2f} dB | lag max {st['lag_ms_max_abs']:.2f} ms"
        )
    if report["failures"]:
        print("")
        print(f"FAILURES ({len(report['failures'])}):")
        for fail in report["failures"]:
            reasons = fail["reasons"]
            shown = "; ".join(reasons[:4])
            if len(reasons) > 4:
                shown += f"; ... (+{len(reasons) - 4})"
            print(
                f"  {fail['track_id']} {fail['phrase']} bar={fail.get('start_bar')} "
                f"corr={_fmt(fail.get('min_correlation'), 5)} "
                f"snr={_fmt(fail.get('min_snr_db'), 1)} "
                f"[{fail.get('diagnosis', '?')}] "
                f"stems={','.join(fail.get('failing_stems', []))} :: {shown}"
            )
    if not report["failures"]:
        print("")
        print("No phrases below QC thresholds.")
    warnings = report.get("warnings") or []
    if warnings:
        print("")
        print(f"WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"  {warn['track_id']} [{warn['kind']}] {warn['detail']}")
    print("")
    print(f"report -> {report.get('report_path')}")


def _fmt(v: Any, nd: int) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{nd}f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct and QC Wub 4-bar stem kits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="kit manifest JSON")
    p.add_argument("--kits-dir", default=DEFAULT_KITS_DIR, help="root for phrase Opus files")
    p.add_argument("--stems", default=DEFAULT_STEMS, help="root of original per-stem WAVs")
    p.add_argument("--processed", default=DEFAULT_PROCESSED, help="analysis JSON directory")
    p.add_argument(
        "--track-id",
        action="append",
        metavar="ID[,ID...]",
        help="repeatable / comma separated; defaults to the first 3 kits",
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument("--mode", choices=("decode", "compare"), default="decode")
    p.add_argument("--report", default=None, help="JSON report path")
    p.add_argument("--fade-ms", type=float, default=5.0, help="de-click width per join side")
    p.add_argument("--join-fade", choices=("auto", "always", "never"), default="auto")
    p.add_argument(
        "--peak-mode",
        choices=("normalise", "clip", "none"),
        default="normalise",
        help="master peak handling in decode mode",
    )
    p.add_argument("--decoder", choices=("auto", "soundfile", "ffmpeg"), default="auto")
    p.add_argument("--downmix", choices=("mean", "left", "right"), default="mean")
    p.add_argument("--max-lag-ms", type=float, default=50.0, help="alignment search window")
    p.add_argument(
        "--min-correlation",
        type=float,
        default=QC_MIN_CORRELATION,
        help="QC gate: per-stem Pearson correlation",
    )
    p.add_argument(
        "--min-snr",
        type=float,
        default=QC_MIN_SNR_DB,
        help="QC gate: per-stem full-band SNR in dB",
    )
    p.add_argument(
        "--max-onset-ms",
        type=float,
        default=QC_MAX_ONSET_MS,
        help="QC gate: allowed grid/decoder offset in ms",
    )
    p.add_argument(
        "--max-duration-ms",
        type=float,
        default=QC_MAX_DURATION_MS,
        help="QC gate: allowed phrase duration error in ms",
    )
    p.add_argument("--compare-rate", default="native", help="'native' or an integer sample rate")
    p.add_argument("--split-stems", action="store_true", help="also write four mono stem WAVs")
    p.add_argument(
        "--check-source",
        action="store_true",
        default=True,
        help="in decode mode, compare the mix against the original stems",
    )
    p.add_argument("--no-check-source", dest="check_source", action="store_false")
    p.add_argument("--json-out", action="store_true", help="also dump the run summary JSON")
    p.add_argument(
        "--fail-on-error", action="store_true", help="exit 2 when QC failures are found"
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    wanted = _split_ids(args.track_id)
    kits, missing = select_kits(manifest, wanted)
    if missing:
        print(f"[warn] track ids not in manifest: {', '.join(missing)}", file=sys.stderr)
    if not kits:
        raise SystemExit("no kits selected")

    if args.mode == "compare":
        report = run_compare(args, manifest, kits)
        _print_compare(report)
        failed = report["n_failures"] > 0
        if args.json_out:
            path = Path(args.out) / "kit_compare_summary.json"
            path.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=True))
            print(f"summary -> {path}")
        return 2 if (failed and args.fail_on_error) else 0

    summary = run_decode(args, manifest, kits)
    _print_decode(summary)
    if args.report:
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
