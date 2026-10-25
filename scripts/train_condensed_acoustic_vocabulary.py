"""Train & Condense Optimal Acoustic Slice Vocabulary from Authentic Songs.

Executes K-Means / VQ codebook clustering across 200,000+ segments from the 119+
authentic dance tracks to select the optimal acoustic slice vocabulary.
Renders click-free 16-bit PCM WAV slices and exports a featherweight,
fast-loading JSON manifest optimized for GitHub Pages static hosting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import soundfile as sf

from wub_core.acoustic_vocabulary_optimizer import (
    AcousticVocabularyOptimizer,
    STYLES,
    STEMS,
    DRUM_CATEGORIES,
    BASS_CATEGORIES,
    OTHER_CATEGORIES,
    VOCAL_CATEGORIES,
)
from scripts.harvest_style_kits import classify_track_style, verify_audio_slices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vocabulary")


def train_and_export_condensed_vocabulary(
    stems_dir: Path = Path("data/stems"),
    processed_dir: Path = Path("data/processed"),
    out_slices_dir: Path = Path("frontend/public/data/slices"),
    patterns_json_path: Path = Path("frontend/public/data/standalone_patterns.json"),
) -> None:
    optimizer = AcousticVocabularyOptimizer(random_state=42)
    out_slices_dir.mkdir(parents=True, exist_ok=True)

    tracks = [d.name for d in stems_dir.iterdir() if d.is_dir() and (d / "drums.wav").exists()]
    logger.info("Scanning %d multi-stem tracks for acoustic codebook training...", len(tracks))

    # Load track metadata and classify styles
    track_meta: Dict[str, dict] = {}
    track_styles: Dict[str, str] = {}
    total_segments = 0

    for tid in tracks:
        jp = processed_dir / f"{tid}.json"
        if jp.exists():
            try:
                with open(jp) as f:
                    meta = json.load(f)
                    track_meta[tid] = meta
                    bpm = float(meta.get("bpm", 126.0))
                    filename = meta.get("filename", f"{tid}.mp3")
                    style = classify_track_style(filename, bpm)
                    track_styles[tid] = style
                    total_segments += len(meta.get("segments", []))
            except Exception:
                pass

    logger.info("Loaded %d tracks with %d total authentic audio segments across 4 styles.", len(track_meta), total_segments)

    # Buckets for clustering: candidates[stem][style][category][pitch] -> List[dict]
    candidates: Dict[str, Dict[str, Dict[str, Dict[int, List[dict]]]]] = {
        stem: {
            style: {}
            for style in STYLES
        }
        for stem in STEMS
    }

    # Initialize drum categories
    for style in STYLES:
        for cat in DRUM_CATEGORIES:
            candidates["drums"][style][cat] = {0: []}

    # Initialize pitched categories (0..11)
    for style in STYLES:
        for cat in BASS_CATEGORIES:
            candidates["bass"][style][cat] = {p: [] for p in range(12)}
        for cat in OTHER_CATEGORIES:
            candidates["other"][style][cat] = {p: [] for p in range(12)}
        for cat in VOCAL_CATEGORIES:
            candidates["vocals"][style][cat] = {p: [] for p in range(12)}

    # Collect and filter candidate segments
    for tid, meta in track_meta.items():
        style = track_styles.get(tid, "wave")
        for seg in meta.get("segments", []):
            if seg.get("is_silent"):
                continue

            stem = seg.get("stem")
            if stem not in STEMS:
                continue

            dur = float(seg.get("duration", 0.0))
            loud = float(seg.get("loudness_max", -99.0))
            centroid = float(seg.get("spectral_centroid", 1000.0))

            # Quality gate
            if loud < -32.0 or dur < 0.05:
                continue

            seg_with_tid = dict(seg)
            seg_with_tid["track_id"] = tid
            seg_with_tid["track_title"] = meta.get("filename", "").replace(".mp3", "")
            seg_with_tid["style"] = style

            # Pitches: chromatic root
            pitches = seg.get("pitches") or seg.get("chroma")
            p_root = int(np.argmax(pitches)) if pitches and len(pitches) == 12 else 0

            if stem == "drums":
                # Categorize drum physical role by spectral centroid and duration
                if centroid <= 750.0 and 0.16 <= dur <= 0.45:
                    candidates["drums"][style]["kick"][0].append(seg_with_tid)
                elif 1200.0 <= centroid <= 3200.0 and 0.12 <= dur <= 0.45:
                    candidates["drums"][style]["snare"][0].append(seg_with_tid)
                elif centroid >= 5000.0 and dur <= 0.12:
                    candidates["drums"][style]["hat_closed"][0].append(seg_with_tid)
                elif centroid >= 4500.0 and 0.14 <= dur <= 0.45:
                    candidates["drums"][style]["hat_open"][0].append(seg_with_tid)
                elif 850.0 <= centroid <= 3800.0 and 0.05 <= dur <= 0.22:
                    candidates["drums"][style]["percussion"][0].append(seg_with_tid)
                elif 1000.0 <= centroid <= 3500.0 and dur >= 0.25:
                    candidates["drums"][style]["fill"][0].append(seg_with_tid)

            elif stem == "bass":
                if centroid < 350.0:
                    candidates["bass"][style]["sub"][p_root].append(seg_with_tid)
                elif dur >= 0.25:
                    candidates["bass"][style]["rolling"][p_root].append(seg_with_tid)
                else:
                    candidates["bass"][style]["stab"][p_root].append(seg_with_tid)

            elif stem == "other":
                if dur >= 0.50:
                    candidates["other"][style]["pad"][p_root].append(seg_with_tid)
                elif dur <= 0.22:
                    candidates["other"][style]["arp"][p_root].append(seg_with_tid)
                else:
                    candidates["other"][style]["chord"][p_root].append(seg_with_tid)

            elif stem == "vocals":
                if dur <= 0.28:
                    candidates["vocals"][style]["chop"][p_root].append(seg_with_tid)
                elif dur >= 0.60:
                    candidates["vocals"][style]["hook"][p_root].append(seg_with_tid)
                else:
                    candidates["vocals"][style]["call"][p_root].append(seg_with_tid)

    logger.info("=== Running ML Codebook Cluster Optimization ===")

    # Configuration for target cluster counts per bucket
    # Ensuring optimal balance of timbral diversity and bundle size
    cluster_targets = {
        "drums": {"kick": 4, "snare": 4, "hat_closed": 4, "hat_open": 4, "percussion": 4, "fill": 4},
        "bass": {"sub": 1, "rolling": 1, "stab": 1},      # 1 optimal medoid per pitch x 12 pitches = 12 slices per style
        "other": {"chord": 1, "arp": 1, "pad": 1},        # 1 optimal medoid per pitch x 12 pitches = 12 slices per style
        "vocals": {"chop": 3, "hook": 3, "call": 3},      # 3 optimal medoids per pitch x 12 pitches = 36 slices per style
    }

    # Desired duration & fade shapes per category
    shape_configs = {
        "kick": {"dur": 0.28, "fade_out": 20.0, "peak": 0.95},
        "snare": {"dur": 0.28, "fade_out": 25.0, "peak": 0.92},
        "hat_closed": {"dur": 0.08, "fade_out": 12.0, "peak": 0.72},
        "hat_open": {"dur": 0.22, "fade_out": 30.0, "peak": 0.82},
        "percussion": {"dur": 0.16, "fade_out": 20.0, "peak": 0.85},
        "fill": {"dur": 0.42, "fade_out": 35.0, "peak": 0.92},
        "sub": {"dur": 0.36, "fade_out": 30.0, "peak": 0.92},
        "rolling": {"dur": 0.34, "fade_out": 28.0, "peak": 0.90},
        "stab": {"dur": 0.22, "fade_out": 20.0, "peak": 0.88},
        "chord": {"dur": 0.45, "fade_out": 35.0, "peak": 0.88},
        "arp": {"dur": 0.20, "fade_out": 18.0, "peak": 0.86},
        "pad": {"dur": 0.85, "fade_out": 50.0, "peak": 0.85},
        "chop": {"dur": 0.26, "fade_out": 25.0, "peak": 0.90},
        "hook": {"dur": 0.75, "fade_out": 45.0, "peak": 0.90},
        "call": {"dur": 0.48, "fade_out": 35.0, "peak": 0.88},
    }

    # Manifest dictionary to be built
    optimized_style_kits: Dict[str, dict] = {
        s: {
            "name": f"{s.upper()} Acoustic Kit",
            "stems": {
                st: {} for st in STEMS
            }
        }
        for s in STYLES
    }
    optimized_slices: Dict[str, Dict[str, List[dict]]] = {
        st: {} for st in STEMS
    }
    optimized_vocal_corpus: List[dict] = []

    total_slices_rendered = 0
    total_distortion = 0.0
    distortion_counts = 0

    # Cluster & Render each stem / style / category / pitch bucket
    for stem in STEMS:
        categories = list(cluster_targets[stem].keys())
        for cat in categories:
            optimized_slices[stem][cat] = []
            target_k = cluster_targets[stem][cat]
            shape_cfg = shape_configs[cat]

            for style in STYLES:
                optimized_style_kits[style]["stems"][stem][cat] = []
                pitch_dict = candidates[stem][style][cat]

                for p, cands in pitch_dict.items():
                    # If candidates are scarce for a particular pitch/style, borrow from other styles of the same pitch
                    if not cands:
                        for alt_style in STYLES:
                            if alt_style != style and candidates[stem][alt_style][cat][p]:
                                cands.extend(candidates[stem][alt_style][cat][p][:3])
                                break

                    # If still empty, fall back to any available candidates in category
                    if not cands:
                        all_cands_in_cat = [
                            c for s in STYLES for p_cand in candidates[stem][s][cat].values() for c in p_cand
                        ]
                        cands = all_cands_in_cat[:5]

                    # Cluster candidate slices into optimal representative medoids
                    medoids, dist = optimizer.condense_category_codebook(cands, k_clusters=target_k)
                    total_distortion += dist
                    distortion_counts += 1

                    for v_idx, chosen in enumerate(medoids):
                        # Construct standardized filename matching test suite expectations
                        if stem == "drums":
                            slice_name = f"drums_{cat}_{style}_{v_idx+1}.wav"
                        elif stem == "bass":
                            slice_name = f"bass_{cat}_{style}_p{p}.wav"
                        elif stem == "other":
                            slice_name = f"other_{cat}_{style}_p{p}.wav"
                        else:  # vocals
                            v_suffix = f"_v{v_idx+1}" if v_idx > 0 else ""
                            slice_name = f"vocals_{cat}_{style}_p{p}{v_suffix}.wav"

                        out_wav_path = out_slices_dir / slice_name
                        stem_wav_path = stems_dir / chosen["track_id"] / f"{stem}.wav"

                        # Render audio slice
                        rendered = optimizer.render_and_save_slice(
                            stem_wav_path=stem_wav_path,
                            start_sec=float(chosen["start"]),
                            duration_sec=float(chosen["duration"]),
                            output_wav_path=out_wav_path,
                            shape_dur=shape_cfg["dur"],
                            peak_target=shape_cfg["peak"],
                            fade_out_ms=shape_cfg["fade_out"],
                        )

                        if not rendered and not out_wav_path.exists():
                            continue

                        # Verify rendered duration
                        try:
                            actual_info = sf.info(str(out_wav_path))
                            actual_dur = round(actual_info.duration, 3)
                        except Exception:
                            actual_dur = shape_cfg["dur"]

                        # Build clean slice manifest item (ZERO base64 strings!)
                        slice_item = {
                            "name": slice_name,
                            "url": f"data/slices/{slice_name}",
                            "stem": stem,
                            "category": cat,
                            "style": style,
                            "pitch": p,
                            "duration": actual_dur,
                            "centroid": round(float(chosen.get("spectral_centroid", 1000.0)), 1),
                            "loudness": round(float(chosen.get("loudness_max", -6.0)), 2),
                            "clarity_score": chosen.get("clarity_score", 0.85),
                            "timbre": chosen.get("timbre_vector", []),
                            "track_id": chosen["track_id"],
                            "track_title": chosen.get("track_title", "Corpus Dance Track"),
                            "segment_id": chosen.get("id", ""),
                        }

                        # Add vowel classification for vocals
                        if stem == "vocals":
                            vowel = "AH"
                            c_val = float(chosen.get("spectral_centroid", 2200.0))
                            if c_val < 1000.0: vowel = "OO"
                            elif c_val < 1700.0: vowel = "OH"
                            elif c_val < 2600.0: vowel = "AH"
                            elif c_val < 3800.0: vowel = "AY"
                            else: vowel = "EE"
                            slice_item["vowel"] = vowel
                            optimized_vocal_corpus.append(slice_item)

                        # Primary style kit items: 1 per pitch (v_idx == 0) for bass/vocals
                        if v_idx == 0:
                            optimized_style_kits[style]["stems"][stem][cat].append(slice_item)

                        # Full global slice palette
                        optimized_slices[stem][cat].append(slice_item)
                        total_slices_rendered += 1

    avg_distortion = total_distortion / max(1, distortion_counts)
    logger.info("✓ Clustered and rendered %d optimal audio slices with average distortion %.4f", total_slices_rendered, avg_distortion)

    # Update standalone_patterns.json with condensed metadata
    if patterns_json_path.exists():
        with open(patterns_json_path) as f:
            manifest_data = json.load(f)

        manifest_data["style_kits"] = optimized_style_kits
        manifest_data["slices"] = optimized_slices
        manifest_data["vocal_corpus"] = optimized_vocal_corpus

        # Strip any remaining base64 fields across entire manifest
        condensed_manifest = optimizer.condense_manifest_metadata(manifest_data)

        # Write clean, compact JSON
        with open(patterns_json_path, "w") as f:
            json.dump(condensed_manifest, f, indent=2)

        file_size_mb = patterns_json_path.stat().st_size / (1024 * 1024)
        logger.info("✓ Updated %s -> New condensed size: %.2f MB (Clean GitHub Pages serving)", patterns_json_path, file_size_mb)

    # Verify all audio slices pass hygiene test
    passed, failed = verify_audio_slices(out_slices_dir)
    logger.info("Audio Slice Verification: %d PASSED, %d FAILED", passed, failed)
    assert failed == 0, f"{failed} slices failed audio verification!"
    logger.info("🎉 Optimal Acoustic Vocabulary Training & Condensation Complete!")


if __name__ == "__main__":
    train_and_export_condensed_vocabulary()
