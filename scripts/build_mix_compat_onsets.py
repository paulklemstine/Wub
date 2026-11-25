#!/usr/bin/env python3
"""Compute per-bar onset/energy grids from shipped kit audio for mix-compat.

Recomputes (not copied) from the shipped bank so rhythm features match exactly
what the browser plays. For each phrase x stem it produces a 64-step grid
(16 steps per bar x 4 bars) of onset/energy strength, aligned to the phrase's
nominal bar grid (4 bars over duration_sec). Output is a compact sidecar the
browser can fetch to score true inter-stem rhythmic lock per bar.

Output:
  <out>/mix_compat_onsets.json  - {version, bars, steps_per_bar, stems, index, dim}
  <out>/mix_compat_onsets.f32   - float32 [N, 4 stems, 64] in phrase index order
  (optional --uint8 also writes .u8 quantized sidecar)

Run (from repo, using shipped kit audio):
  PYTHONPATH=. python3 scripts/build_mix_compat_onsets.py \
    --manifest frontend/public/data/kits/kit_manifest.json \
    --kits-dir frontend/public/data/kits \
    --out frontend/public/data/kits

Dependencies: numpy, soundfile or librosa (for opus decode) + scipy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "other", "vocals")
BARS = 4
STEPS_PER_BAR = 16
STEPS = BARS * STEPS_PER_BAR  # 64


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_mono(path: Path, sr: int = 22050) -> np.ndarray:
    """Load audio as mono float32 at sr. Tries soundfile, then librosa, then ffmpeg."""
    # soundfile
    try:
        import soundfile as sf
        data, file_sr = sf.read(str(path), always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        x = np.asarray(data, dtype=np.float32)
        if file_sr != sr:
            x = resample(x, file_sr, sr)
        return x
    except Exception:
        pass
    # librosa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import librosa
            x, _ = librosa.load(str(path), sr=sr, mono=True)
        return np.asarray(x, dtype=np.float32)
    except Exception:
        pass
    # ffmpeg fallback
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"could not decode {path}: {proc.stderr[:200]!r}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        # linear fallback
        n_out = int(round(len(x) * sr_out / sr_in))
        if n_out <= 1:
            return np.zeros(1, np.float32)
        idx = np.linspace(0, len(x) - 1, n_out)
        xp = np.arange(len(x))
        return np.interp(idx, xp, x).astype(np.float32)


def onset_envelope(x: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    """Spectral-flux onset envelope (positive differences), robust and fast."""
    if len(x) < 512:
        return np.zeros(1, np.float32)
    # STFT magnitude via framed FFT
    n_fft = 1024
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames < 2:
        return np.zeros(1, np.float32)
    # frame
    mags = []
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec = np.fft.rfft(frame * win)
        mags.append(np.abs(spec))
    M = np.stack(mags, axis=0)  # [T, F]
    # log compression + half-wave rectified flux
    M = np.log1p(M)
    diff = np.diff(M, axis=0)
    diff[diff < 0] = 0
    env = diff.sum(axis=1)
    # smooth a touch
    if len(env) >= 3:
        k = np.array([0.25, 0.5, 0.25], np.float32)
        env = np.convolve(env, k, mode="same")
    return env.astype(np.float32)


def grid_for_phrase(x: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Return 64-step onset grid aligned to 4 bars spanning duration_sec."""
    env = onset_envelope(x, sr)
    if env.size == 0 or not (duration_sec > 0):
        return np.zeros(STEPS, np.float32)
    hop = 256
    times = (np.arange(len(env)) + 1) * (hop / sr)  # flux between frame i and i+1
    # sample env at step centers across [0, duration_sec)
    step_times = (np.arange(STEPS) + 0.5) * (duration_sec / STEPS)
    grid = np.interp(step_times, times, env, left=0.0, right=0.0).astype(np.float32)
    # per-grid normalize (shape matters, not absolute level)
    mx = float(grid.max())
    if mx > 1e-8:
        grid = grid / mx
    else:
        grid[:] = 0.0
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--kits-dir", default="frontend/public/data/kits")
    ap.add_argument("--out", default="frontend/public/data/kits")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--uint8", action="store_true", help="also write quantized u8 sidecar")
    ap.add_argument("--limit", type=int, default=0, help="process first N phrases (debug)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    kits_dir = Path(args.kits_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    kits = manifest.get("kits") or []
    # stable phrase order for the index
    phrase_ids = []
    phrase_meta = {}
    for kit in kits:
        for p in kit.get("phrases") or []:
            pid = p.get("id")
            stems = p.get("stems") or {}
            if not isinstance(pid, str):
                continue
            if not all(isinstance(stems.get(s), dict) and stems[s].get("file") for s in STEMS):
                continue
            phrase_ids.append(pid)
            phrase_meta[pid] = (kit, p)
    if args.limit:
        phrase_ids = phrase_ids[:args.limit]
    n = len(phrase_ids)
    log(f"[onsets] phrases={n} stems={len(STEMS)} steps={STEPS} sr={args.sr}")

    data = np.zeros((n, len(STEMS), STEPS), dtype=np.float32)
    for i, pid in enumerate(phrase_ids):
        kit, p = phrase_meta[pid]
        dur = float(p.get("duration_sec") or 0.0)
        if not (dur > 0):
            # derive from kit bpm: 4 bars
            bpm = float(kit.get("bpm") or 120.0)
            dur = 4.0 * 4.0 * 60.0 / bpm
        for si, stem in enumerate(STEMS):
            rel = p["stems"][stem]["file"]
            path = kits_dir / rel
            if not path.is_file():
                log(f"[onsets] missing {path}")
                continue
            try:
                x = load_mono(path, sr=args.sr)
                data[i, si] = grid_for_phrase(x, args.sr, dur)
            except Exception as e:
                log(f"[onsets] warn {pid}:{stem} {e}")
        if (i + 1) % 200 == 0:
            log(f"[onsets] {i+1}/{n}")

    f32_path = out_dir / "mix_compat_onsets.f32"
    data.tofile(f32_path)
    index = {pid: i for i, pid in enumerate(phrase_ids)}
    meta = {
        "version": 1,
        "format": "mix_compat_onsets_v1",
        "bars": BARS,
        "steps_per_bar": STEPS_PER_BAR,
        "steps": STEPS,
        "stems": list(STEMS),
        "count": n,
        "feature_file": "mix_compat_onsets.f32",
        "dtype": "float32",
        "shape": [n, len(STEMS), STEPS],
        "index": index,
    }
    (out_dir / "mix_compat_onsets.json").write_text(json.dumps(meta, separators=(",", ":")))
    if args.uint8:
        # quantize 0..1 to 0..255
        q = np.clip(np.rint(data * 255.0), 0, 255).astype(np.uint8)
        q.tofile(out_dir / "mix_compat_onsets.u8")
        log(f"[onsets] wrote u8 {out_dir / 'mix_compat_onsets.u8'}")
    log(f"[onsets] wrote {f32_path} ({f32_path.stat().st_size/1e6:.2f} MB) + index for {n} phrases")


if __name__ == "__main__":
    main()
