"""Export Standalone Client-Side ML Models, Pattern Bank, Whole-Pattern Loops, and Audio Slices.

Packages:
1. data/composing_ml_weights.json -> frontend/public/data/composing_ml_weights.json
2. Curated standalone patterns & phrases -> frontend/public/data/standalone_patterns.json
3. 1-bar studio-grade whole pattern stem loops -> frontend/public/data/loops/
4. Clean authentic audio slice WAVs -> frontend/public/data/slices/
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


def harvest_whole_pattern_loops(
    multistem_tracks: List,
    loops_dir: Path,
) -> Dict[str, List[dict]]:
    """Extract continuous 1-bar studio-grade whole pattern stem loops across iconic tracks."""
    print("=== Harvesting 1-Bar Whole Pattern Stem Loops ===", flush=True)
    loops_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[dict]] = {s: [] for s in STEMS}

    # Desired loop roles per stem
    targets: Dict[str, List[str]] = {
        "drums": ["drop", "groove_a", "breakbeat", "turnaround", "buildup", "minimal"],
        "bass": ["rolling", "sub_pump", "stabs", "sustained"],
        "other": ["riff", "chords", "pad", "arp"],
        "vocals": ["hook", "verse", "call", "stutter", "ambient"],
    }

    pool: Dict[str, Dict[str, List[dict]]] = {s: {c: [] for c in targets[s]} for s in targets}

    for tr in multistem_tracks:
        tid = tr.track_id
        tatums = getattr(tr, "tatums", [])
        n_bars = len(tatums) // 16
        bpm = float(getattr(tr, "bpm", 126.0))
        key = str(getattr(tr, "key", "C major"))
        clean_title = getattr(tr, "filename", "").replace(".mp3", "")

        for b in range(min(n_bars - 1, 140)):
            t0 = tatums[b * 16]
            t1 = tatums[(b + 1) * 16]
            start = float(getattr(t0, "start", t0["start"] if isinstance(t0, dict) else 0.0))
            end = float(getattr(t1, "start", t1["start"] if isinstance(t1, dict) else 0.0))
            dur = end - start
            if dur < 1.4 or dur > 2.6:
                continue

            for stem in STEMS:
                stem_path = Path("data/stems") / tid / f"{stem}.wav"
                if not stem_path.exists():
                    continue

                segs = [
                    s for s in getattr(tr, "segments", [])
                    if getattr(s, "stem", s.get("stem", "") if isinstance(s, dict) else "") == stem
                    and not getattr(s, "is_silent", s.get("is_silent", False) if isinstance(s, dict) else False)
                    and float(getattr(s, "start", s["start"] if isinstance(s, dict) else 0.0)) >= start - 0.05
                    and float(getattr(s, "start", s["start"] if isinstance(s, dict) else 0.0)) < end
                ]
                if not segs:
                    continue

                louds = [
                    float(getattr(s, "loudness_max", getattr(s, "loud_max", s.get("loudness_max", -99.0) if isinstance(s, dict) else -99.0)))
                    for s in segs
                ]
                max_loud = max(louds)
                if max_loud < -28.0:
                    continue

                onsets: List[int] = []
                for s in segs:
                    s_start = float(getattr(s, "start", s["start"] if isinstance(s, dict) else 0.0))
                    step_idx = int(round((s_start - start) / (dur / 16.0)))
                    if 0 <= step_idx < 16 and step_idx not in onsets:
                        onsets.append(step_idx)
                onsets.sort()

                pitch_sum = np.zeros(12)
                for s in segs:
                    pitches = getattr(s, "pitches", None)
                    if pitches and len(pitches) == 12:
                        pitch_sum += np.array(pitches)
                dom_pitch = int(np.argmax(pitch_sum)) if np.sum(pitch_sum) > 0.01 else 0

                # Musical role classification
                cat = None
                if stem == "drums":
                    if any(s in onsets for s in (11, 12, 13, 14, 15)) and len(onsets) >= 10:
                        cat = "turnaround"
                    elif len(onsets) >= 8 and any(s in onsets for s in (0, 4, 8, 12)):
                        cat = "drop" if max_loud > -8.0 else "groove_a"
                    elif any(s in onsets for s in (2, 6, 10, 14)) and len(onsets) >= 6:
                        cat = "breakbeat"
                    elif len(onsets) in (4, 5, 6):
                        cat = "buildup"
                    elif len(onsets) in (1, 2, 3):
                        cat = "minimal"
                elif stem == "bass":
                    if len(onsets) >= 8:
                        cat = "rolling"
                    elif any(s in onsets for s in (2, 6, 10, 14)):
                        cat = "sub_pump"
                    elif len(onsets) in (3, 4, 5):
                        cat = "stabs"
                    else:
                        cat = "sustained"
                elif stem == "other":
                    if len(onsets) >= 8:
                        cat = "arp"
                    elif any(s in onsets for s in (2, 6, 10, 14)) and len(onsets) in (3, 4, 5, 6):
                        cat = "chords"
                    elif len(onsets) in (4, 5, 6, 7):
                        cat = "riff"
                    else:
                        cat = "pad"
                elif stem == "vocals":
                    if max_loud > -14.0 and len(onsets) >= 4:
                        cat = "hook"
                    elif len(onsets) >= 6:
                        cat = "stutter"
                    elif len(onsets) in (3, 4, 5):
                        cat = "verse"
                    elif any(s in onsets for s in (0, 2, 8, 10)):
                        cat = "call"
                    else:
                        cat = "ambient"

                if cat and cat in pool[stem]:
                    pool[stem][cat].append({
                        "tid": tid,
                        "title": clean_title,
                        "bar": b,
                        "start": start,
                        "dur": dur,
                        "bpm": bpm,
                        "key": key,
                        "pitch": dom_pitch,
                        "onsets": onsets,
                        "max_loud": max_loud,
                    })

    # Pick top 2-3 per category across diverse tracks and render 16-bit WAVs
    total_rendered = 0
    for stem, cat_dict in pool.items():
        for cat, cand_list in cat_dict.items():
            cand_list.sort(key=lambda x: x["max_loud"], reverse=True)
            seen_tids: Set[str] = set()
            curated = []
            for item in cand_list:
                if item["tid"] not in seen_tids or len(curated) < 2:
                    seen_tids.add(item["tid"])
                    curated.append(item)
                if len(curated) >= 3:
                    break

            for idx, item in enumerate(curated):
                stem_path = Path("data/stems") / item["tid"] / f"{stem}.wav"
                try:
                    info = sf.info(str(stem_path))
                    s_frame = max(0, int(item["start"] * info.samplerate))
                    n_frames = min(int(item["dur"] * info.samplerate), info.frames - s_frame)
                    if n_frames <= 1000:
                        continue
                    audio, sr = sf.read(str(stem_path), start=s_frame, frames=n_frames, always_2d=True)
                    mono = audio.mean(axis=1)

                    # Peak normalize to -1.0 dBFS (0.88 factor)
                    peak = float(np.max(np.abs(mono)))
                    if peak > 0.001:
                        mono = mono * (0.88 / peak)

                    # Apply 4ms cosine fade in/out to guarantee seamless pop-free looping
                    fade_len = int(sr * 0.004)
                    if len(mono) > fade_len * 2:
                        fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
                        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
                        mono[:fade_len] *= fade_in
                        mono[-fade_len:] *= fade_out

                    loop_fname = f"{stem}_{cat}_{idx+1}.wav"
                    sf.write(str(loops_dir / loop_fname), mono, sr, subtype="PCM_16")

                    grid = [1 if i in item["onsets"] else 0 for i in range(16)]

                    # Map suggested song sections
                    suggested: List[str] = []
                    if cat in ("drop", "rolling", "riff", "hook"):
                        suggested = ["DROP", "GROOVE_A"]
                    elif cat in ("groove_a", "breakbeat", "sub_pump", "chords", "verse", "call"):
                        suggested = ["GROOVE_A"]
                    elif cat in ("buildup", "turnaround", "stutter", "arp"):
                        suggested = ["BUILDUP"]
                    elif cat in ("minimal", "sustained", "pad", "ambient"):
                        suggested = ["INTRO", "BREAKDOWN"]

                    manifest[stem].append({
                        "id": f"loop_{stem}_{cat}_{idx+1}",
                        "name": f"{item['title']} Bar {item['bar']} ({cat})",
                        "stem": stem,
                        "category": cat,
                        "url": f"data/loops/{loop_fname}",
                        "source_bpm": round(item["bpm"], 2),
                        "key": item["key"],
                        "dominant_pitch": item["pitch"],
                        "duration": round(item["dur"], 3),
                        "onset_grid": grid,
                        "track_id": item["tid"],
                        "track_title": item["title"],
                        "suggested_sections": suggested,
                    })
                    total_rendered += 1
                except Exception:
                    continue

    print(f"✓ Exported {total_rendered} continuous 1-bar studio stem loops to {loops_dir}", flush=True)
    return manifest


def harvest_cohesive_phrase_blocks(
    multistem_tracks: List,
    phrases_dir: Path,
) -> List[dict]:
    """Extract continuous 4-bar studio-grade multi-stem phrase blocks across iconic tracks."""
    print("=== Harvesting 4-Bar Cohesive Multi-Stem Phrase Blocks ===", flush=True)
    phrases_dir.mkdir(parents=True, exist_ok=True)
    manifest_phrases: List[dict] = []

    targets = {"DROP": 4, "GROOVE_A": 4, "BUILDUP": 3, "BREAKDOWN": 3, "INTRO": 3}
    collected: Dict[str, List[dict]] = {sec: [] for sec in targets}

    fade_len = int(0.004 * 44100)  # 4ms cosine fade
    t_fade = np.linspace(0, 1, fade_len)
    fade_in = np.sin(0.5 * np.pi * t_fade)
    fade_out = np.cos(0.5 * np.pi * t_fade)

    for tr in multistem_tracks:
        tid = tr.track_id
        tatums = getattr(tr, "tatums", [])
        segments = getattr(tr, "segments", [])
        n_bars = len(tatums) // 16
        bpm = float(getattr(tr, "bpm", 126.0))
        key = str(getattr(tr, "key", "A minor"))
        clean_title = getattr(tr, "filename", "").replace(".mp3", "")

        if not (118.0 <= bpm <= 132.0):
            continue

        stem_paths = {s: Path("data/stems") / tid / f"{s}.wav" for s in STEMS}
        if not all(p.exists() for p in stem_paths.values()):
            continue

        for b in range(0, min(n_bars - 4, 120), 4):
            t0 = tatums[b * 16]
            t1 = tatums[(b + 4) * 16]
            start = float(getattr(t0, "start", t0["start"] if isinstance(t0, dict) else 0.0))
            end = float(getattr(t1, "start", t1["start"] if isinstance(t1, dict) else 0.0))
            dur = end - start
            if dur < 6.8 or dur > 8.4:
                continue

            bar_segs = [
                s for s in segments
                if float(getattr(s, "start", s.get("start", 0.0) if isinstance(s, dict) else 0.0)) >= start - 0.05
                and float(getattr(s, "start", s.get("start", 0.0) if isinstance(s, dict) else 0.0)) < end
                and not getattr(s, "is_silent", s.get("is_silent", False) if isinstance(s, dict) else False)
            ]
            loudness_by_stem = {s: -99.0 for s in STEMS}
            for s in bar_segs:
                st = getattr(s, "stem", s.get("stem", "") if isinstance(s, dict) else "")
                if st in loudness_by_stem:
                    loud = float(getattr(s, "loudness_max", getattr(s, "loud_max", s.get("loudness_max", -99.0) if isinstance(s, dict) else -99.0)))
                    loudness_by_stem[st] = max(loudness_by_stem[st], loud)

            d_loud = loudness_by_stem["drums"]
            b_loud = loudness_by_stem["bass"]
            o_loud = loudness_by_stem["other"]
            v_loud = loudness_by_stem["vocals"]

            sec = None
            if d_loud >= -12.0 and b_loud >= -14.0 and o_loud >= -18.0:
                sec = "DROP"
            elif d_loud >= -15.0 and b_loud >= -18.0:
                sec = "GROOVE_A"
            elif d_loud < -25.0 and b_loud < -25.0 and (o_loud >= -22.0 or v_loud >= -22.0):
                sec = "INTRO" if b <= 8 else "BREAKDOWN"
            elif -22.0 <= d_loud < -14.0 and o_loud >= -18.0:
                sec = "BUILDUP"

            if not sec or len(collected[sec]) >= targets[sec]:
                continue

            if sum(1 for p in collected[sec] if p["track_id"] == tid) >= 2:
                continue

            f0 = int(round(start * 44100))
            flen = int(round(dur * 44100))
            phrase_id = f"phrase_{sec.lower()}_{len(collected[sec]) + 1}_{tid[:6]}"
            stem_urls = {}
            onset_grids = {}

            valid_extraction = True
            for stem in STEMS:
                out_wav = phrases_dir / f"{phrase_id}_{stem}.wav"
                try:
                    data, sr = sf.read(str(stem_paths[stem]), start=f0, frames=flen, always_2d=True, dtype="float32")
                    mono = data.mean(axis=1)

                    peak = float(np.max(np.abs(mono)))
                    if peak > 0.001:
                        mono = mono * (0.89 / peak)

                    if len(mono) > fade_len * 2:
                        mono[:fade_len] *= fade_in
                        mono[-fade_len:] *= fade_out

                    sf.write(str(out_wav), mono, sr, subtype="PCM_16")
                    stem_urls[stem] = f"data/phrases/{out_wav.name}"

                    onsets = []
                    for s in bar_segs:
                        st_name = getattr(s, "stem", s.get("stem", "") if isinstance(s, dict) else "")
                        if st_name == stem:
                            s_start = float(getattr(s, "start", s.get("start", 0.0) if isinstance(s, dict) else 0.0))
                            step_idx = int(round((s_start - start) / (dur / 16.0)))
                            if 0 <= step_idx < 16 and step_idx not in onsets:
                                onsets.append(step_idx)
                    onsets.sort()
                    grid_16 = [1 if i in onsets else 0 for i in range(16)]
                    if sum(grid_16) == 0 and stem == "drums" and sec in ("DROP", "GROOVE_A"):
                        grid_16 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                    onset_grids[stem] = grid_16
                except Exception:
                    valid_extraction = False
                    break

            if not valid_extraction:
                continue

            phrase_item = {
                "id": phrase_id,
                "track_id": tid,
                "track_title": clean_title,
                "key": key,
                "source_bpm": bpm,
                "section": sec,
                "energy": 0.95 if sec == "DROP" else (0.82 if sec == "GROOVE_A" else (0.65 if sec == "BUILDUP" else 0.25)),
                "bar_length": 4,
                "duration": round(dur, 3),
                "stems": stem_urls,
                "onset_grids": onset_grids,
            }
            collected[sec].append(phrase_item)
            manifest_phrases.append(phrase_item)
            print(f"   ✓ Extracted {sec} phrase: {clean_title} (bar {b}-{b+4}, {bpm} BPM, {key})", flush=True)

    print(f"✓ Total harvested {len(manifest_phrases)} cohesive 4-bar multi-stem phrase blocks.", flush=True)
    return manifest_phrases


def export_bundle() -> None:
    print("=== Exporting Standalone In-Browser ML & Pattern Assets ===", flush=True)

    public_dir = Path("frontend/public/data")
    slices_dir = public_dir / "slices"
    loops_dir = public_dir / "loops"
    phrases_dir = public_dir / "phrases"
    public_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)
    loops_dir.mkdir(parents=True, exist_ok=True)
    phrases_dir.mkdir(parents=True, exist_ok=True)

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
    store.refresh(max_tracks=40)
    multistem_tracks = [t for t in store.tracks if len(t.stems_available) == 4]
    if not multistem_tracks:
        multistem_tracks = store.tracks[:20]

    bank = PatternBank()
    n_pats = bank.load_from_tracks(multistem_tracks)
    print(f"✓ Extracted {n_pats} patterns and {len(bank.phrases)} phrases across {len(multistem_tracks)} tracks into memory bank", flush=True)

    # 3. Harvest Cohesive 4-Bar Multi-Stem Phrase Blocks
    extracted_phrase_blocks = harvest_cohesive_phrase_blocks(multistem_tracks, phrases_dir)

    # 4. Harvest Whole 1-Bar Pattern Loops Across Stems
    extracted_loops_manifest = harvest_whole_pattern_loops(multistem_tracks, loops_dir)

    # 4. Harvest Pure Audio Slices (Zero Duplicates, Chromatic Bass/Other 0..11)
    from scripts.harvest_pure_slices import harvest_pure_slices
    extracted_slices_manifest = harvest_pure_slices(
        Path("data/stems"),
        Path("data/processed"),
        slices_dir,
    )

    # 5. Extract authentic harmonic chord progressions from real tracks
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
            while len(bar_roots) < 8:
                bar_roots.append(bar_roots[-1])
            chord_progressions.append({
                "track_id": tr.track_id,
                "key": getattr(tr, "key", "C major"),
                "roots": bar_roots[:8],
            })
    print(f"✓ Extracted {len(chord_progressions)} authentic chord progressions from corpus", flush=True)

    # 6. Serialize Curated Pattern Bank for In-Browser ML Sequencing
    serialized_patterns: Dict[str, List[dict]] = {s: [] for s in STEMS}

    for stem, pats in bank.patterns_by_stem.items():
        by_groove: Dict[str, List] = {}
        for p in pats:
            by_groove.setdefault(p.groove_type, []).append(p)

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
                    "cohesion_score": round(float(getattr(p, "cohesion_score", 0.92)), 3),
                })

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

    # Load stem_cohesion_weights from weights file if present
    stem_cohesion_weights = {}
    if dst_weights.exists():
        try:
            w_data = json.loads(dst_weights.read_text())
            stem_cohesion_weights = w_data.get("stem_cohesion_weights", {})
        except Exception:
            pass

    standalone_manifest = {
        "version": 3,
        "default_bpm": 126.0,
        "chord_progressions": chord_progressions,
        "groove_templates": groove_templates,
        "patterns": serialized_patterns,
        "phrases": serialized_phrases,
        "slices": extracted_slices_manifest,
        "loops": extracted_loops_manifest,
        "phrase_blocks": extracted_phrase_blocks,
        "stem_cohesion_weights": stem_cohesion_weights,
    }

    dst_patterns = public_dir / "standalone_patterns.json"
    dst_patterns.write_text(json.dumps(standalone_manifest, indent=2))
    print(f"✓ Exported {dst_patterns} ({dst_patterns.stat().st_size} bytes)", flush=True)

    print("=== Standalone Client Bundle Export Complete ===", flush=True)


if __name__ == "__main__":
    export_bundle()
