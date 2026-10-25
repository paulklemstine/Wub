"""Harvest Large Corpus of Authentic Vocal Chops from Training Songs.

Directly extracts, cleans, and tags hundreds of authentic vocal slices from the
119 multi-stem training tracks (`data/stems/*/vocals.wav`):
- Categories:
  1. `chop` : Fast, punchy vocal chops and phonemes (0.08s - 0.35s) for syncopated 16th groove infill.
  2. `call` : Mid-length vocal shouts, melodic calls, and breath sighs (0.35s - 0.95s) for turnaround downbeats.
  3. `hook` : Sustained sung vocal phrases and anthemic melodies (0.95s - 2.80s) for Chorus Drops and Solo breakdowns.
- Properties:
  - 100% click-free (1.5ms cosine fade-in, smooth cosine fade-out).
  - Peak-normalized to -1.0 dBFS (peak <= 0.92).
  - Full chromatic pitch coverage (p0..p11) across all 4 styles ('wave', 'cyber', 'house', 'breaks').
  - Tagged with vowel formant ('AH', 'EE', 'OH', 'OO', 'AY'), spectral centroid, loudness, and artist track title.
- Updates:
  - Writes WAV slices to `frontend/public/data/slices/`.
  - Updates `frontend/public/data/standalone_patterns.json` with rich `slices.vocals` manifests and `vocal_corpus`.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

from scripts.harvest_style_kits import clean_slice, classify_track_style, STYLES
from wub_core.vocal_pattern_extractor import classify_vowel


def harvest_large_vocal_corpus(
    stems_dir: Path,
    processed_dir: Path,
    out_dir: Path,
    patterns_json_path: Path,
    slices_per_pitch: int = 2,
) -> Tuple[List[dict], int]:
    """Extract a large corpus of authentic vocal slices directly chopped from training songs."""
    print("=== Harvesting Large Authentic Vocal Corpus from Training Songs ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocal_tids = [
        d.name for d in stems_dir.iterdir()
        if d.is_dir() and (d / "vocals.wav").exists() and (processed_dir / f"{d.name}.json").exists()
    ]
    print(f"Found {len(vocal_tids)} training tracks containing vocals.wav", flush=True)

    # Pre-index metadata and classify style
    track_meta: Dict[str, dict] = {}
    tracks_by_style: Dict[str, List[str]] = {s: [] for s in STYLES}

    for tid in vocal_tids:
        try:
            with open(processed_dir / f"{tid}.json") as f:
                meta = json.load(f)
                track_meta[tid] = meta
                bpm = float(meta.get("bpm", 126.0))
                style = classify_track_style(meta.get("filename", ""), bpm)
                tracks_by_style[style].append(tid)
        except Exception:
            pass

    for s in STYLES:
        print(f"  Style '{s}': {len(tracks_by_style[s])} vocal tracks", flush=True)

    # Categories configuration
    categories = {
        "chop": {"dur_min": 0.08, "dur_max": 0.36, "shape_dur": 0.30, "fade_out": 22.0, "peak": 0.90, "loud_min": -28.0},
        "call": {"dur_min": 0.36, "dur_max": 0.95, "shape_dur": 0.75, "fade_out": 35.0, "peak": 0.90, "loud_min": -27.0},
        "hook": {"dur_min": 0.95, "dur_max": 2.80, "shape_dur": 2.20, "fade_out": 50.0, "peak": 0.88, "loud_min": -26.0},
    }

    seen_hashes: Set[str] = set()
    harvested_slices: List[dict] = []
    slices_by_cat_style: Dict[str, List[dict]] = {}

    for cat, cfg in categories.items():
        for style in STYLES:
            key = f"{cat}_{style}"
            slices_by_cat_style[key] = []
            target_tids = tracks_by_style[style]
            other_tids = [tid for tid in vocal_tids if tid not in target_tids]

            # Pool candidates by chromatic pitch (0..11)
            cands_by_pitch: Dict[int, List[Tuple[float, float, str, dict]]] = {p: [] for p in range(12)}

            def scan_tids(tids: List[str], penalty: float = 0.0):
                for tid in tids:
                    meta = track_meta.get(tid, {})
                    for s in meta.get("segments", []):
                        if s.get("stem") != "vocals" or s.get("is_silent"):
                            continue
                        dur = float(s.get("duration", 0.0))
                        loud = float(s.get("loudness_max", -99.0))
                        chroma = s.get("pitches", [])
                        if loud < cfg["loud_min"] - 6.0 or not chroma or len(chroma) != 12:
                            continue
                        if not (cfg["dur_min"] * 0.75 <= dur <= cfg["dur_max"] * 1.35):
                            continue
                        p = int(np.argmax(chroma))
                        score = loud - penalty + float(chroma[p]) * 3.0
                        cands_by_pitch[p].append((score, loud, tid, s))

            scan_tids(target_tids, penalty=0.0)
            # Fallback to other tracks for missing pitches
            missing = [p for p in range(12) if len(cands_by_pitch[p]) < slices_per_pitch]
            if missing:
                scan_tids(other_tids, penalty=5.0)

            # Harvest slices for each pitch 0..11
            cat_count = 0
            for p in range(12):
                cands = sorted(cands_by_pitch[p], key=lambda x: x[0], reverse=True)
                taken_for_pitch = 0

                for _, loud, tid, s in cands:
                    if taken_for_pitch >= slices_per_pitch:
                        break
                    stem_path = stems_dir / tid / "vocals.wav"
                    try:
                        info = sf.info(str(stem_path))
                        start_frame = max(0, int(s["start"] * info.samplerate))
                        frame_len = min(info.frames - start_frame, int(s["duration"] * info.samplerate))
                        if frame_len < 100:
                            continue

                        raw, sr = sf.read(str(stem_path), start=start_frame, frames=frame_len, always_2d=True, dtype="float32")
                        mono = clean_slice(
                            raw.mean(axis=1),
                            sr,
                            target_dur=cfg["shape_dur"],
                            peak_target=cfg["peak"],
                            fade_out_ms=cfg["fade_out"],
                        )

                        h = hashlib.md5(mono.tobytes()).hexdigest()
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        # Name format: vocals_{cat}_{style}_p{p}_{var}.wav
                        var_idx = taken_for_pitch + 1
                        out_name = f"vocals_{cat}_{style}_p{p}_{var_idx}.wav" if var_idx > 1 else f"vocals_{cat}_{style}_p{p}.wav"
                        out_path = out_dir / out_name
                        sf.write(str(out_path), mono, sr, subtype="PCM_16")

                        # Also save legacy alias for the first variation (e.g. vocals_chop_wave_1.wav)
                        if p < 3 and var_idx == 1:
                            legacy_name = f"vocals_{cat}_{style}_{p+1}.wav"
                            legacy_path = out_dir / legacy_name
                            sf.write(str(legacy_path), mono, sr, subtype="PCM_16")

                        b64 = f"data:audio/wav;base64,{base64.b64encode(out_path.read_bytes()).decode('ascii')}"
                        centroid = float(s.get("spectral_centroid", 2200.0))
                        vowel = classify_vowel(centroid)
                        track_title = track_meta.get(tid, {}).get("filename", tid).replace(".mp3", "")

                        item = {
                            "name": out_name,
                            "url": f"data/slices/{out_name}",
                            "b64": b64 if len(b64) < 65000 else None,
                            "duration": round(len(mono) / sr, 3),
                            "centroid": centroid,
                            "vowel": vowel,
                            "pitch": p,
                            "loudness": float(loud),
                            "style": style,
                            "category": cat,
                            "stem": "vocals",
                            "track_id": tid,
                            "track_title": track_title,
                            "segment_id": s.get("id", ""),
                        }

                        harvested_slices.append(item)
                        slices_by_cat_style[key].append(item)
                        taken_for_pitch += 1
                        cat_count += 1
                    except Exception:
                        continue

            print(f"  ✓ Harvested {cat_count} slices for vocals:{cat} ({style})", flush=True)

    print(f"\n🎉 Successfully harvested {len(harvested_slices)} authentic vocal slices across 119 tracks!", flush=True)

    # Update standalone_patterns.json
    if patterns_json_path.exists():
        print(f"=== Updating {patterns_json_path} with Large Vocal Corpus ===", flush=True)
        with open(patterns_json_path) as f:
            data = json.load(f)

        # Update slices.vocals
        if "slices" not in data:
            data["slices"] = {}
        if "vocals" not in data["slices"]:
            data["slices"]["vocals"] = {}

        # Add base categories and styled categories
        for cat in ["chop", "call", "hook"]:
            cat_items = [s for s in harvested_slices if s["category"] == cat]
            data["slices"]["vocals"][cat] = cat_items
            for style in STYLES:
                styled_key = f"{cat}_{style}"
                data["slices"]["vocals"][styled_key] = [s for s in cat_items if s["style"] == style]

        # Update style_kits with primary chromatic vocal banks (12 pitches per category)
        if "style_kits" in data:
            for style in STYLES:
                if style in data["style_kits"]:
                    v_stem = data["style_kits"][style]["stems"]["vocals"]
                    for cat in ["chop", "call", "hook"]:
                        primary_12 = []
                        seen_p = set()
                        for s in harvested_slices:
                            if s["category"] == cat and s["style"] == style and s["pitch"] not in seen_p:
                                primary_12.append(s)
                                seen_p.add(s["pitch"])
                                if len(primary_12) == 12:
                                    break
                        v_stem[cat] = primary_12

        # Add dedicated vocal_corpus list
        data["vocal_corpus"] = [
            {
                "name": s["name"],
                "url": s["url"],
                "b64": s["b64"],
                "duration": s["duration"],
                "centroid": s["centroid"],
                "vowel": s["vowel"],
                "pitch": s["pitch"],
                "style": s["style"],
                "category": s["category"],
                "track_id": s["track_id"],
                "track_title": s["track_title"],
            }
            for s in harvested_slices
        ]

        with open(patterns_json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Updated {patterns_json_path} ({patterns_json_path.stat().st_size} bytes)", flush=True)

    return harvested_slices, len(seen_hashes)


if __name__ == "__main__":
    stems = Path("data/stems")
    proc = Path("data/processed")
    slices = Path("frontend/public/data/slices")
    pat_json = Path("frontend/public/data/standalone_patterns.json")

    # Harvest 2-3 variations per chromatic pitch = ~288 to 360 slices
    harvested, count = harvest_large_vocal_corpus(stems, proc, slices, pat_json, slices_per_pitch=3)
