"""Learn a small per-stem seam gain correction from the shipped kit audio.

Each phrase is level matched by whole-phrase RMS, but a 20 ms window at a bar
line can still be quieter/louder than the preceding phrase's last 20 ms. The
runtime already fades in at the outgoing voice's gain (`handoffGain`); this
script measures the corpus-wide first/last-window ratio and exports a bounded
correction so that the fade starts at an acoustically equal level.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

STEMS = ("drums", "bass", "other", "vocals")
WINDOW_SEC = 0.020


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    kits = args.kits_dir
    manifest = json.loads((kits / "kit_manifest.json").read_text())
    first: Dict[str, List[float]] = {s: [] for s in STEMS}
    last: Dict[str, List[float]] = {s: [] for s in STEMS}
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            for stem, meta in (phrase.get("stems") or {}).items():
                if stem not in STEMS:
                    continue
                rel = meta.get("file")
                path = kits / rel if rel else None
                if not path or not path.exists():
                    continue
                try:
                    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
                except Exception:
                    continue
                if y.shape[0] < sr * 0.1:
                    continue
                pre = float(kit.get("pre_roll_sec") or phrase.get("pre_roll_sec") or 0.0)
                start = int(pre * sr)
                y = y[start:]
                n = max(1, int(WINDOW_SEC * sr))
                head = rms(y[:n])
                tail = rms(y[-n:])
                if head > 1e-5 and tail > 1e-5:
                    first[stem].append(head)
                    last[stem].append(tail)

    correction: Dict[str, float] = {}
    report: Dict[str, dict] = {}
    for stem in STEMS:
        a = np.asarray(first[stem])
        b = np.asarray(last[stem])
        if len(a) < 50:
            correction[stem] = 1.0
            continue
        # A pair-level view: typical incoming head vs typical outgoing tail.
        ratio = float(np.median(b) / np.median(a))
        correction[stem] = round(float(np.clip(ratio, 0.88, 1.12)), 4)
        report[stem] = {
            "phrases": len(a),
            "median_head": round(float(np.median(a)), 5),
            "median_tail": round(float(np.median(b)), 5),
            "raw_ratio": round(ratio, 4),
        }
    out = args.out or (kits / "level_seam.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "wub-level-seam-v1", "version": 1,
        "window_sec": WINDOW_SEC,
        "correction": correction,
        "report": report,
    }, indent=2) + "\n")
    print(json.dumps({"correction": correction, "report": report}, indent=2))


if __name__ == "__main__":
    main()
