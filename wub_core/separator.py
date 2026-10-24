"""Neural stem extraction wrapper around Hybrid Transformer Demucs.

Each source MP3 -> 4 phase-aligned stereo WAV stems at 44.1 kHz:
drums.wav, bass.wav, vocals.wav, other.wav
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

STEM_FILES: Dict[str, str] = {
    "drums": "drums.wav",
    "bass": "bass.wav",
    "vocals": "vocals.wav",
    "other": "other.wav",
}


def compute_track_id(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def stems_dir_for(base: Path, track_id: str) -> Path:
    return base / track_id


def stems_exist(stems_dir: Path) -> bool:
    return all((stems_dir / f).exists() for f in STEM_FILES.values())


def separate_track(
    input_mp3: Path,
    stems_base: Path,
    model: str = "htdemucs",
    fast_cpu: bool = False,
    device: str | None = None,
    overwrite: bool = False,
) -> tuple[str, Path]:
    """Run Demucs on `input_mp3`. Returns (track_id, stems_dir).

    Results are cached in `stems_base/{track_id}/` so separation runs once
    per song. Raises RuntimeError with install hint if demucs is missing.
    """
    track_id = compute_track_id(input_mp3)
    out_dir = stems_dir_for(stems_base, track_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if stems_exist(out_dir) and not overwrite:
        return track_id, out_dir

    # Resolve device (require at least 1.8GB VRAM free on CUDA, else fallback to CPU)
    if device is None:
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes = torch.cuda.mem_get_info()[0] if hasattr(torch.cuda, "mem_get_info") else 0
                device = "cuda" if free_bytes > int(1.8 * (1 << 30)) else "cpu"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "--out",
        str(out_dir.parent / "_demucs_tmp"),
        "--filename",
        "{track}/{stem}.{ext}",
    ]
    if device == "cpu":
        cmd += ["-d", "cpu"]
    if fast_cpu:
        # Fewer overlap segments + mp3 output is faster but we need wav;
        # use shifts=1 and overlap=0.25 for speed, keep wav for fidelity.
        # htdemucs max segment is 7.8 seconds, so use 7.
        cmd += ["--shifts", "1", "--overlap", "0.25", "--segment", "7"]
    cmd += [str(input_mp3)]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "demucs is not installed. Install with: pip install demucs torch torchaudio"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"demucs separation failed for {input_mp3}: {e}") from e

    # Demucs writes to _demucs_tmp/{model}/{input_stem}/stem.wav. Scope the
    # lookup to THIS track's directory: an unconstrained rglob would grab a
    # previous track's output and silently poison the cache.
    tmpbase = out_dir.parent / "_demucs_tmp" / model / input_mp3.stem
    src_dir: Path | None = tmpbase if (tmpbase / "drums.wav").exists() else None
    if src_dir is None:
        # Fallback for demucs variants that keep the file extension in the
        # track dir name; still scoped under this model run.
        cands = sorted((out_dir.parent / "_demucs_tmp").rglob("drums.wav"))
        cands = [c for c in cands
                 if c.parent.name in (input_mp3.stem, input_mp3.name)]
        src_dir = cands[0].parent if cands else None
    if src_dir is None:
        raise RuntimeError(f"demucs output not found for {input_mp3} "
                           f"(looked in {tmpbase})")

    import soundfile as sf

    for stem, fname in STEM_FILES.items():
        src = src_dir / f"{stem}.wav"
        dst = out_dir / fname
        if not src.exists():
            raise RuntimeError(f"demucs output missing stem {stem}: {src}")
        # Ensure 44.1k stereo WAV. Demucs already outputs 44.1k; resample if needed.
        y, sr = sf.read(str(src), always_2d=True)
        if sr != 44100:
            import librosa

            y_mono = y.mean(axis=1)
            y_rs = librosa.resample(y_mono, orig_sr=sr, target_sr=44100)
            y = __import__("numpy").repeat(y_rs[:, None], 2, axis=1)
            sr = 44100
        sf.write(str(dst), y, 44100)

    # Free the per-track tmp output; the cache dir is the source of truth.
    import shutil

    shutil.rmtree(tmpbase, ignore_errors=True)

    return track_id, out_dir


def list_stem_paths(stems_dir: Path) -> Dict[str, Path]:
    return {stem: stems_dir / fname for stem, fname in STEM_FILES.items()}
