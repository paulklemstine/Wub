"""Opus encoding for the static client bundle.

PCM WAV slices cost ~66 KB median; Ogg Opus at 64k mono costs ~6 KB for the
same audio (~10x smaller). That is what lets the full 1,724-song vocabulary
fit inside GitHub Pages' 1 GB published-site limit.

Decoding needs no client change beyond a container-agnostic
`AudioContext.decodeAudioData` call (Chrome/Firefox/Edge decode Ogg Opus;
anything that fails falls back to the shared-slice synth path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union

import numpy as np


def ffmpeg_exe() -> str:
    """Resolve a ffmpeg binary with libopus (bundled static build preferred)."""
    try:
        import imageio_ffmpeg  # type: ignore

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return "ffmpeg"


def write_opus(
    path: Union[str, Path],
    mono: np.ndarray,
    sr: int,
    bitrate: str = "64k",
) -> Path:
    """Encode mono float audio to Ogg Opus via piped raw PCM (no temp files).

    Raises subprocess.CalledProcessError if the encoder fails.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(np.asarray(mono, dtype=np.float32)).reshape(-1)
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(int(sr)),
        "-ac",
        "1",
        "-i",
        "-",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-compression_level",
        "10",
        "-frame_duration",
        "20",
        str(out),
    ]
    subprocess.run(cmd, input=data.tobytes(), check=True)
    return out


def opus_byte_ratio_guard(opus_path: Union[str, Path], num_samples: int) -> bool:
    """Sanity check: opus file must exist, carry the OpusHead magic, and be
    smaller than 16-bit PCM bytes for the same sample count."""
    p = Path(opus_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p, "rb") as f:
        magic = f.read(36)
    if b"OpusHead" not in magic:
        return False
    return p.stat().st_size < num_samples * 2
