"""Export Standalone Client-Side ML Models, Pattern Bank, and Audio Slices.

Packages:
1. data/composing_ml_weights.json -> frontend/public/data/composing_ml_weights.json
2. Curated standalone patterns & phrases -> frontend/public/data/standalone_patterns.json
3. Clean authentic audio slice WAVs -> frontend/public/data/slices/
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import soundfile as sf

from wub_core.corpus import CorpusStore
from wub_core.patterns import PatternBank, STEMS
from wub_core.groovy_pattern_assembler import AcousticSliceClassifier


def export_bundle() -> None:
    print("=== Exporting Standalone In-Browser ML & Pattern Assets ===", flush=True)

    public_dir = Path("frontend/public/data")
    slices_dir = public_dir / "slices"
    public_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Composing ML Weights
    src_weights = Path("data/composing_ml_weights.json")
    dst_weights = public_dir / "composing_ml_weights.json"
    if src_weights.exists():
        shutil.copy(src_weights, dst_weights)
        print(f"✓ Copied {src_weights} ({dst_weights.stat().st_size} bytes)", flush=True)
    else:
        print(f"Warning: {src_weights} not found, generating placeholder", flush=True)
        dst_weights.write_text(json.dumps({"version": 1, "phrase_transitions": {}, "section_transitions": {}}))

    # 2. Extract Multi-Stem Tracks & Patterns from Authentic Corpus
    store = CorpusStore(Path("data/processed"), stems_base=Path("data/stems"))
    store.refresh(max_tracks=60)
    multistem_tracks = [t for t in store.tracks if len(t.stems_available) == 4]
    if not multistem_tracks:
        multistem_tracks = store.tracks[:20]

    bank = PatternBank()
    n_pats = bank.load_from_tracks(multistem_tracks)
    print(f"✓ Extracted {n_pats} patterns and {len(bank.phrases)} phrases across {len(multistem_tracks)} tracks into memory bank", flush=True)

    # 3. Harvest Clean Audio Slices per Acoustic Category
    extracted_slices_manifest: Dict[str, Dict[str, List[dict]]] = {
        s: {} for s in STEMS
    }

    # Desired slice categories per stem
    targets: Dict[str, List[str]] = {
        "drums": ["kick", "snare", "hat_closed", "hat_open", "percussion", "fill"],
        "bass": ["sub", "rolling", "stab"],
        "other": ["chord", "arp", "pad", "stab"],
        "vocals": ["hook", "phrase", "call", "chop"],
    }

    target_sr = 44100
    slice_count = 0

    # Pre-gather high-energy vocal slices directly from multistem tracks to guarantee loud, authentic singing
    vocal_candidates_by_cat: Dict[str, List] = {
        "hook": [],
        "phrase": [],
        "call": [],
        "chop": [],
    }

    for tr in multistem_tracks:
        for s in getattr(tr, "segments", []):
            if getattr(s, "stem", "") == "vocals" and not s.is_silent:
                loud = float(getattr(s, "loudness_max", -99.0))
                dur = float(getattr(s, "duration", 0.0))
                if loud >= -24.0 and dur >= 0.12:
                    if dur < 0.35:
                        vocal_candidates_by_cat["chop"].append((s, tr))
                    elif dur < 0.85:
                        vocal_candidates_by_cat["call"].append((s, tr))
                    elif dur < 1.8:
                        vocal_candidates_by_cat["hook"].append((s, tr))
                    elif dur <= 4.0:
                        vocal_candidates_by_cat["phrase"].append((s, tr))

    # Sort vocal candidates by loudness (clearest & loudest singing first)
    for cat in vocal_candidates_by_cat:
        vocal_candidates_by_cat[cat].sort(key=lambda item: getattr(item[0], "loudness_max", -99.0), reverse=True)

    for stem, cat_list in targets.items():
        pal = bank.slice_palette.get(stem, {})
        for cat in cat_list:
            extracted_slices_manifest[stem][cat] = []

            # Determine candidates
            if stem == "vocals":
                raw_candidates = vocal_candidates_by_cat.get(cat, [])
                candidates = [s for s, _tr in raw_candidates]
            else:
                candidates = pal.get(cat, [])
                if not candidates and "hat" in cat:
                    candidates = pal.get("hat", [])
                if not candidates:
                    candidates = [s for bucket in pal.values() for s in bucket]
                # Filter out near-silent slices and sort by loudness
                candidates = [s for s in candidates if getattr(s, "loud_max", -99.0) > -32.0]
                candidates.sort(key=lambda s: getattr(s, "loud_max", -99.0), reverse=True)

            # Pick top distinct slices across different pitch classes & different tracks
            taken = 0
            seen_pitches: Set[int] = set()
            seen_tracks: Set[str] = set()
            max_take = 6 if stem == "vocals" else 4

            # First pass: try diverse pitches & tracks
            for s in candidates:
                if taken >= max_take:
                    break
                pitch = int(np.argmax(s.pitches)) if hasattr(s, "pitches") and s.pitches else (
                    int(np.argmax(s.chroma)) if hasattr(s, "chroma") and s.chroma is not None and len(s.chroma) == 12 else 0
                )
                tid = s.track_id
                if (pitch in seen_pitches and len(seen_pitches) < 10) and (tid in seen_tracks and len(seen_tracks) < 5):
                    continue

                stem_path = Path("data/stems") / s.track_id / f"{stem}.wav"
                if not stem_path.exists():
                    stem_path = Path("data/stems") / s.track_id / f"{stem}.mp3"
                if not stem_path.exists():
                    continue

                try:
                    info = sf.info(str(stem_path))
                    start_frame = max(0, int(s.start * info.samplerate))
                    frame_len = min(info.frames - start_frame, int(s.duration * info.samplerate))
                    if frame_len <= 100:
                        continue

                    data, sr = sf.read(str(stem_path), start=start_frame, frames=frame_len, always_2d=True, dtype="float32")
                    mono = data.mean(axis=1)

                    # Normalize peak amplitude to -1.0 dBFS (0.89) to eliminate quiet bleed
                    peak = float(np.max(np.abs(mono)))
                    if peak > 0.001:
                        mono = mono * (0.89 / peak)

                    out_name = f"{stem}_{cat}_{taken+1}.wav"
                    out_wav = slices_dir / out_name
                    sf.write(str(out_wav), mono, sr, subtype="PCM_16")

                    # Also encode small base64 data URI fallback
                    data_bytes = out_wav.read_bytes()
                    b64_uri = f"data:audio/wav;base64,{base64.b64encode(data_bytes).decode('ascii')}"

                    extracted_slices_manifest[stem][cat].append({
                        "name": out_name,
                        "url": f"data/slices/{out_name}",
                        "b64": b64_uri if len(b64_uri) < 65000 else None,
                        "duration": float(s.duration),
                        "centroid": float(getattr(s, "spectral_centroid", 0.0)),
                        "pitch": pitch,
                        "loudness": float(getattr(s, "loud_max", getattr(s, "loudness_max", -12.0))),
                        "track_id": s.track_id,
                        "segment_id": getattr(s, "segment_id", getattr(s, "id", "")),
                    })
                    seen_pitches.add(pitch)
                    seen_tracks.add(tid)
                    taken += 1
                    slice_count += 1
                except Exception:
                    continue

            # Second pass fallback if fewer than desired were taken
            if taken < max_take:
                for s in candidates:
                    if taken >= max_take:
                        break
                    out_name = f"{stem}_{cat}_{taken+1}.wav"
                    out_wav = slices_dir / out_name
                    if out_wav.exists():
                        continue

                    stem_path = Path("data/stems") / s.track_id / f"{stem}.wav"
                    if not stem_path.exists():
                        stem_path = Path("data/stems") / s.track_id / f"{stem}.mp3"
                    if not stem_path.exists():
                        continue

                    try:
                        info = sf.info(str(stem_path))
                        start_frame = max(0, int(s.start * info.samplerate))
                        frame_len = min(info.frames - start_frame, int(s.duration * info.samplerate))
                        if frame_len <= 100:
                            continue

                        data, sr = sf.read(str(stem_path), start=start_frame, frames=frame_len, always_2d=True, dtype="float32")
                        mono = data.mean(axis=1)
                        peak = float(np.max(np.abs(mono)))
                        if peak > 0.001:
                            mono = mono * (0.89 / peak)

                        sf.write(str(out_wav), mono, sr, subtype="PCM_16")
                        data_bytes = out_wav.read_bytes()
                        b64_uri = f"data:audio/wav;base64,{base64.b64encode(data_bytes).decode('ascii')}"

                        pitch = int(np.argmax(s.pitches)) if hasattr(s, "pitches") and s.pitches else (
                            int(np.argmax(s.chroma)) if hasattr(s, "chroma") and s.chroma is not None and len(s.chroma) == 12 else 0
                        )

                        extracted_slices_manifest[stem][cat].append({
                            "name": out_name,
                            "url": f"data/slices/{out_name}",
                            "b64": b64_uri if len(b64_uri) < 65000 else None,
                            "duration": float(s.duration),
                            "centroid": float(getattr(s, "spectral_centroid", 0.0)),
                            "pitch": pitch,
                            "loudness": float(getattr(s, "loud_max", getattr(s, "loudness_max", -12.0))),
                            "track_id": s.track_id,
                            "segment_id": getattr(s, "segment_id", getattr(s, "id", "")),
                        })
                        taken += 1
                        slice_count += 1
                    except Exception:
                        continue

    print(f"✓ Exported {slice_count} clean acoustic slice WAVs to {slices_dir}", flush=True)

    # 4. Extract authentic harmonic chord progressions from real tracks
    chord_progressions: List[dict] = []
    for tr in multistem_tracks[:35]:
        tatums = getattr(tr, "tatums", [])
        n_bars = len(tatums) // 16
        bar_roots: List[int] = []
        for b in range(min(8, n_bars)):
            bar_tats = tatums[b * 16:(b + 1) * 16]
            chroma_sum = np.zeros(12)
            for t in bar_tats:
                for st in ("bass", "other"):
                    sids = getattr(t, "stem_segments", {}).get(st, [])
                    for sid in sids:
                        for s in getattr(tr, "segments", []):
                            if s.id == sid and not s.is_silent:
                                chroma_sum += np.array(s.pitches)
            if np.sum(chroma_sum) > 0.05:
                bar_roots.append(int(np.argmax(chroma_sum)))
            else:
                bar_roots.append(bar_roots[-1] if bar_roots else 0)
        if len(bar_roots) >= 4:
            # Pad to 8 bars if needed
            while len(bar_roots) < 8:
                bar_roots.append(bar_roots[-1])
            chord_progressions.append({
                "track_id": tr.track_id,
                "key": getattr(tr, "key", "C major"),
                "roots": bar_roots[:8],
            })
    print(f"✓ Extracted {len(chord_progressions)} authentic chord progressions from corpus", flush=True)

    # 5. Serialize Curated Pattern Bank for In-Browser ML Sequencing
    serialized_patterns: Dict[str, List[dict]] = {s: [] for s in STEMS}

    for stem, pats in bank.patterns_by_stem.items():
        # Group by groove_type
        by_groove: Dict[str, List] = {}
        for p in pats:
            by_groove.setdefault(p.groove_type, []).append(p)

        # For each groove type, sort by similarity + energy and pick diverse tracks
        for gt, g_pats in by_groove.items():
            valid_pats = [p for p in g_pats if p.density > 0.0]
            if not valid_pats:
                continue

            valid_pats.sort(
                key=lambda p: (getattr(p, "audio_similarity_score", 0.5) * 0.6 + float(p.energy) * 0.4),
                reverse=True,
            )

            seen_tracks = set()
            curated_for_gt = []
            for p in valid_pats:
                # Encourage diverse tracks while ensuring top patterns are kept
                if p.track_id not in seen_tracks or len(curated_for_gt) < 4:
                    seen_tracks.add(p.track_id)
                    curated_for_gt.append(p)
                if len(curated_for_gt) >= 8:
                    break

            for p in curated_for_gt:
                serialized_patterns[stem].append({
                    "id": p.id,
                    "name": p.pattern_name,
                    "groove_type": p.groove_type,
                    "energy": round(float(p.energy), 3),
                    "density": round(float(p.density), 3),
                    "onset_grid": p.onset_grid,
                    "velocities": p.velocities if p.velocities else [1.0 if i in p.onset_grid else 0.0 for i in range(p.length)],
                    "length": p.length,
                    "key": p.key,
                    "audio_similarity_score": round(float(getattr(p, "audio_similarity_score", 0.95)), 3),
                })

    # Summary of curated patterns
    print("✓ Curated patterns across stems and groove types:", flush=True)
    for stem, pats in serialized_patterns.items():
        groove_counts: Dict[str, int] = {}
        for p in pats:
            groove_counts[p["groove_type"]] = groove_counts.get(p["groove_type"], 0) + 1
        print(f"   {stem.upper()}: {len(pats)} total patterns {groove_counts}", flush=True)

    # Curate authentic multi-stem phrases
    serialized_phrases: List[dict] = []
    for phr in bank.phrases[:25]:
        bar_list = []
        for b in phr.bars:
            bar_list.append({
                "bar_num": b.bar_num,
                "phrase_bar": b.phrase_bar,
                "section": b.section,
                "onset_grids": b.onset_grids,
                "groove_types": b.groove_types,
                "energy": round(float(b.energy), 3),
                "key": b.key,
            })
        serialized_phrases.append({
            "id": phr.id,
            "section": phr.section,
            "bpm": phr.bpm,
            "key": phr.key,
            "energy": round(float(phr.energy), 3),
            "bars": bar_list,
        })

    # Curate all physical acoustic groove templates
    from wub_core.groovy_pattern_assembler import GLOBAL_GROOVE_TEMPLATES
    groove_templates: Dict[str, Dict[str, dict]] = {s: {} for s in STEMS}
    for stem, tmpl_dict in GLOBAL_GROOVE_TEMPLATES.items():
        for g_name, tmpl in tmpl_dict.items():
            steps = [
                {"step": s.step_index, "category": s.category, "velocity": s.velocity, "duration_scale": s.duration_scale}
                for s in tmpl.steps
            ]
            groove_templates[stem][g_name] = {
                "name": tmpl.name,
                "stem": tmpl.stem,
                "length": tmpl.length,
                "energy": tmpl.energy,
                "steps": steps,
            }

    standalone_manifest = {
        "version": 1,
        "default_bpm": 126.0,
        "chord_progressions": chord_progressions,
        "groove_templates": groove_templates,
        "patterns": serialized_patterns,
        "phrases": serialized_phrases,
        "slices": extracted_slices_manifest,
    }

    dst_patterns = public_dir / "standalone_patterns.json"
    dst_patterns.write_text(json.dumps(standalone_manifest, indent=2))
    print(f"✓ Exported {dst_patterns} ({dst_patterns.stat().st_size} bytes)", flush=True)

    print("=== Standalone Client Bundle Export Complete ===", flush=True)


if __name__ == "__main__":
    export_bundle()
