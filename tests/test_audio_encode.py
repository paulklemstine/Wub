"""Unit tests for scripts/audio_encode.py (Opus static-bundle encoding)."""

import numpy as np

from scripts.audio_encode import opus_byte_ratio_guard, write_opus


def _sine_sr440(n: int = 8800, sr: int = 44100) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / float(sr)
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def test_write_opus_creates_valid_smaller_file(tmp_path):
    out = tmp_path / "tone.opus"
    mono = _sine_sr440()
    write_opus(out, mono, 44100, bitrate="64k")
    assert out.exists()
    assert opus_byte_ratio_guard(out, len(mono))


def test_opus_guard_rejects_pcm_sized_garbage(tmp_path):
    fake = tmp_path / "fake.opus"
    fake.write_bytes(b"\x00" * 4096)
    assert not opus_byte_ratio_guard(fake, 100)
