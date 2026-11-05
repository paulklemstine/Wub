"""Extract a short pre-bar tail for every kit phrase/stem.

The kit cuts start exactly on the bar line, so a drum attack that lands a few
milliseconds before the bar (or a pick-up) is lost; 41% of drum phrases begin
more than 12 dB below their own RMS. The player schedules these 80 ms tails
just before the bar, restoring the attack without re-cutting or re-encoding
the main phrase.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.audio_encode import write_opus

STEMS = ("drums", "bass", "other", "vocals")


def one(job: tuple) -> tuple[str, int, str, dict]:
    phrase, stems_root, kits_dir, pre = job
    tid = phrase["track_id"] if "track_id" in phrase else phrase["_tid"]
    start = float(phrase["start_sec"])
    made = 0
    err = ""
    max_pre = 0.0
    pre_fields: dict[str, dict] = {}
    for stem in STEMS:
        asset = (phrase.get("stems") or {}).get(stem)
        if not isinstance(asset, dict) or not asset.get("file"):
            continue
        src = Path(stems_root) / tid / f"{stem}.wav"
        if not src.is_file():
            err = f"missing {src}"
            continue
        with sf.SoundFile(str(src)) as f:
            sr = f.samplerate
            start_frame = int(round(start * sr))
            pre_frames = max(0, min(int(round(pre * sr)), start_frame))
            end_frame = start_frame
            if end_frame - (start_frame - pre_frames) < 32:
                continue
            f.seek(start_frame - pre_frames)
            data = f.read(end_frame - (start_frame - pre_frames), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        rel = f"kits/{tid}/{phrase['id']}_{stem}_pre.opus"
        out = Path(kits_dir) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        write_opus(out, mono, sr)
        dur = round(len(mono) / sr, 6)
        asset["pre_roll"] = {"file": rel, "duration_sec": dur}
        pre_fields[stem] = {"file": rel, "duration_sec": dur}
        max_pre = max(max_pre, dur)
        made += 1
    return phrase["id"], made, err, {
        "pre_roll_sec": round(max_pre, 6),
        "stems": pre_fields,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--kits-dir", default="frontend/public/data/kits")
    ap.add_argument("--pre", type=float, default=0.08)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    shutil.copy2(manifest_path, str(manifest_path) + ".prebak")

    jobs = []
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            phrase["_tid"] = kit["track_id"]
            jobs.append((phrase, args.stems, args.kits_dir, args.pre))

    by_id: dict[str, dict] = {}
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            by_id[phrase["id"]] = phrase

    made = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for pid, n, err, fields in ex.map(one, jobs):
            made += n
            if err:
                errors += 1
                print(f"{pid}: {err}", flush=True)
            phrase = by_id.get(pid)
            if phrase is not None:
                phrase["pre_roll_sec"] = fields["pre_roll_sec"]
                for stem, pre in fields["stems"].items():
                    asset = (phrase.get("stems") or {}).get(stem)
                    if isinstance(asset, dict):
                        asset["pre_roll"] = pre
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            phrase.pop("_tid", None)

    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    print(json.dumps({"phrases": len(jobs), "pre_roll_files": made, "errors": errors,
                      "manifest_mb": round(manifest_path.stat().st_size / 1e6, 2)}))


if __name__ == "__main__":
    main()
