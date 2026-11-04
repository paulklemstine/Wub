"""Render a spectrogram + waveform PNG from a raw interleaved float32 PCM file.

Usage: python3 scripts/viz_pcm.py /tmp/pptr/master.pcm 44100 /tmp/pptr/master.png
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    path, sr_s, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sr = int(sr_s)
    x = np.fromfile(path, dtype=np.float32).reshape(-1, 2).T
    mono = x.mean(axis=0)
    n = len(mono)
    win = 2048
    hop = 512
    frames = 1 + (n - win) // hop
    spec = np.empty((win // 2 + 1, frames), dtype=np.float32)
    w = np.hanning(win).astype(np.float32)
    for i in range(frames):
        seg = mono[i * hop: i * hop + win] * w
        spec[:, i] = np.abs(np.fft.rfft(seg))
    db = 20 * np.log10(spec + 1e-6)
    db = np.clip((db + 80) / 80, 0, 1)
    # log-frequency remap to 512 rows
    freqs = np.linspace(0, win // 2, spec.shape[0])
    rows = 512
    logf = np.geomspace(30, sr / 2, rows)
    idx = np.searchsorted(freqs * sr / win, logf).clip(0, spec.shape[0] - 1)
    img = (db[idx[::-1], :] * 255).astype(np.uint8)
    im = Image.fromarray(img).convert("RGB").resize((1200, rows))

    # waveform strip
    h = 220
    wave = Image.new("RGB", (1200, h), (12, 12, 16))
    d = ImageDraw.Draw(wave)
    d.line([(0, h // 2), (1200, h // 2)], fill=(40, 40, 50))
    step = max(1, n // 1200)
    for px in range(1200):
        seg = mono[px * step:(px + 1) * step]
        if len(seg) == 0:
            continue
        lo, hi = float(seg.min()), float(seg.max())
        d.line([(px, int(h / 2 - hi * h / 2)), (px, int(h / 2 - lo * h / 2))], fill=(90, 220, 255))
    canvas = Image.new("RGB", (1200, rows + h + 24), (8, 8, 10))
    canvas.paste(im, (0, 0))
    canvas.paste(wave, (0, rows + 24))
    canvas.save(out)
    print(f"wrote {out} ({frames} frames, {n/sr:.1f}s)")


if __name__ == "__main__":
    main()
