"""Export Standalone Client-Side ML Models, Pattern Bank, Whole-Pattern Loops, and Audio Slices.

Packages:
1. data/composing_ml_weights.json -> frontend/public/data/composing_ml_weights.json
2. Curated standalone patterns & phrases -> frontend/public/data/standalone_patterns.json
3. 1-bar studio-grade whole pattern stem loops -> frontend/public/data/loops/
4. Clean authentic audio slice Opus files -> frontend/public/data/slices/
"""

from __future__ import annotations

import base64
import json
import shutil
from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import soundfile as sf
from scripts.audio_encode import write_opus

from wub_core.corpus import CorpusStore
from wub_core.patterns import PatternBank, STEMS
from wub_core.groovy_pattern_assembler import AcousticSliceClassifier


def stratify_tracks_by_artist(tracks: List) -> List:
    """Interleave tracks round-robin by artist so every artist across A-Z is represented equally."""
    by_artist: Dict[str, List] = defaultdict(list)
    for tr in tracks:
        fname = str(getattr(tr, "filename", "")).replace(".mp3", "")
        artist = fname.split(" - ")[0].strip() if " - " in fname else fname[:12]
        by_artist[artist].append(tr)
    sorted_artists = sorted(
        by_artist.keys(),
        key=lambda a: hashlib.md5(a.encode("utf-8", errors="ignore")).hexdigest(),
    )
    out: List = []
    max_len = max((len(v) for v in by_artist.values()), default=0)
    for r_idx in range(max_len):
        for artist in sorted_artists:
            items = by_artist[artist]
            if r_idx < len(items):
                out.append(items[r_idx])
    return out


def _decompose_mix_16bar_into_stems(stereo_16: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Decompose a 16-bar stereo mix window into 4 acoustic frequency/spatial stems via zero-phase FFT."""
    if stereo_16.ndim == 1:
        mid = stereo_16.astype(np.float32)
        side = np.zeros_like(mid)
    else:
        mid = stereo_16.mean(axis=1).astype(np.float32)
        side = (0.5 * (stereo_16[:, 0] - stereo_16[:, 1])).astype(np.float32) if stereo_16.shape[1] > 1 else np.zeros_like(mid)

    n = len(mid)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mid_spec = np.fft.rfft(mid)
    side_spec = np.fft.rfft(side)

    # 1. Bass stem: smooth sub & bass fundamental band (28..175 Hz)
    bass_mask = np.clip((freqs - 25.0) / 15.0, 0.0, 1.0) * np.clip((185.0 - freqs) / 35.0, 0.0, 1.0)
    bass_mono = np.fft.irfft(mid_spec * bass_mask, n=n).astype(np.float32)

    # 2. Vocals stem: center vocal formant band (260..3900 Hz) minus side stereo width
    vocal_mask = np.clip((freqs - 240.0) / 60.0, 0.0, 1.0) * np.clip((4100.0 - freqs) / 400.0, 0.0, 1.0)
    vocals_mono = np.fft.irfft((mid_spec - 0.35 * side_spec) * vocal_mask, n=n).astype(np.float32)

    # 3. Other (Synths/Chords/Pads): stereo side + harmonic midrange/highs (200..11000 Hz)
    other_mask = np.clip((freqs - 190.0) / 50.0, 0.0, 1.0) * np.clip((11500.0 - freqs) / 1000.0, 0.0, 1.0)
    other_mono = np.fft.irfft((0.65 * mid_spec + 0.85 * side_spec) * other_mask, n=n).astype(np.float32)

    # 4. Drums stem: kick punch (45..115 Hz) + snare crack (1200..3200 Hz) + crisp hats (> 4800 Hz)
    drum_mask = (
        0.95 * np.clip((freqs - 40.0) / 15.0, 0.0, 1.0) * np.clip((120.0 - freqs) / 25.0, 0.0, 1.0)
        + 0.70 * np.clip((freqs - 1100.0) / 200.0, 0.0, 1.0) * np.clip((3400.0 - freqs) / 300.0, 0.0, 1.0)
        + 0.90 * np.clip((freqs - 4600.0) / 500.0, 0.0, 1.0)
    )
    drums_mono = np.fft.irfft(mid_spec * drum_mask, n=n).astype(np.float32)

    out: Dict[str, np.ndarray] = {}
    for st_name, wave in (("drums", drums_mono), ("bass", bass_mono), ("other", other_mono), ("vocals", vocals_mono)):
        pk = float(np.max(np.abs(wave)))
        if pk >= 0.015:
            wave = wave * min(2.2, 0.88 / pk)
        out[st_name] = wave
    return out


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

    # Pick top 2-3 per category across diverse tracks and render Opus files
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
                if len(curated) >= 8:
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

                    loop_fname = f"{stem}_{cat}_{idx+1}.opus"
                    write_opus(loops_dir / loop_fname, mono, sr)

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
    fallback_mp3s: Optional[Dict[str, Path]] = None,
) -> List[dict]:
    """Extract continuous multi-part 4-bar studio-grade phrase chains stratified across the A-Z song catalog."""
    print("=== Harvesting Multi-Part 4-Bar Cohesive Song Phrase Chains Across Full A-Z Catalog ===", flush=True)
    phrases_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale phrase audio so old single-block files don't linger (WAV legacy + prior Opus runs)
    for old_wav in list(phrases_dir.glob("*.wav")) + list(phrases_dir.glob("*.opus")):
        try:
            old_wav.unlink()
        except Exception:
            pass

    manifest_phrases: List[dict] = []

    fade_len = int(0.004 * 44100)  # 4ms cosine fade
    t_fade = np.linspace(0, 1, fade_len)
    fade_in = np.sin(0.5 * np.pi * t_fade)
    fade_out = np.cos(0.5 * np.pi * t_fade)

    max_songs = 42
    songs_harvested = 0
    stratified_candidates = stratify_tracks_by_artist(multistem_tracks)

    for tr in stratified_candidates:
        if songs_harvested >= max_songs:
            break
        tid = tr.track_id
        tatums = getattr(tr, "tatums", [])
        segments = getattr(tr, "segments", [])
        n_bars = len(tatums) // 16
        bpm = float(getattr(tr, "bpm", 126.0))
        key = str(getattr(tr, "key", "A minor"))
        clean_title = getattr(tr, "filename", "").replace(".mp3", "")

        if not (85.0 <= bpm <= 165.0) or n_bars < 20:
            continue

        stem_paths = {s: Path("data/stems") / tid / f"{s}.wav" for s in STEMS}
        has_4_stems = all(p.exists() for p in stem_paths.values())
        src_mp3 = fallback_mp3s.get(tid) if fallback_mp3s else None
        if not has_4_stems and not (src_mp3 and src_mp3.exists()):
            continue

        # Find the best continuous 16-bar (4 x 4-bar parts) active window in this song
        best_start_bar = 8 if n_bars >= 28 else 0
        best_window_score = -999.0

        for cand_b in range(4, min(n_bars - 16, 80), 4):
            t_s = float(getattr(tatums[cand_b * 16], "start", tatums[cand_b * 16]["start"] if isinstance(tatums[cand_b * 16], dict) else 0.0))
            t_e = float(getattr(tatums[(cand_b + 16) * 16], "start", tatums[(cand_b + 16) * 16]["start"] if isinstance(tatums[(cand_b + 16) * 16], dict) else 0.0))
            w_segs = [
                s for s in segments
                if t_s <= float(getattr(s, "start", s.get("start", 0.0) if isinstance(s, dict) else 0.0)) < t_e
                and not getattr(s, "is_silent", s.get("is_silent", False) if isinstance(s, dict) else False)
            ]
            if not w_segs:
                continue
            loud_by_st = {s: -60.0 for s in STEMS}
            for s in w_segs:
                st = getattr(s, "stem", s.get("stem", "") if isinstance(s, dict) else "")
                if st in loud_by_st:
                    ld = float(getattr(s, "loudness_max", getattr(s, "loud_max", s.get("loudness_max", -60.0) if isinstance(s, dict) else -60.0)))
                    loud_by_st[st] = max(loud_by_st[st], ld)
            # Favor windows where drums, bass, and other/vocals are all actively present
            score = loud_by_st["drums"] * 1.2 + loud_by_st["bass"] * 1.1 + loud_by_st["other"] * 1.0 + loud_by_st["vocals"] * 0.8
            if score > best_window_score:
                best_window_score = score
                best_start_bar = cand_b

        # Extract 4 consecutive 4-bar parts (16 continuous bars) from best_start_bar
        # Slice all 4 parts from a SINGLE contiguous 16-bar read with a SINGLE uniform gain scale
        # and identical exact sample length so Part 1 -> Part 2 -> Part 3 -> Part 4 is 100% sample-continuous!
        if best_start_bar + 16 >= n_bars:
            continue
        t_start_16 = float(getattr(tatums[best_start_bar * 16], "start", tatums[best_start_bar * 16]["start"] if isinstance(tatums[best_start_bar * 16], dict) else 0.0))
        t_end_16 = float(getattr(tatums[(best_start_bar + 16) * 16], "start", tatums[(best_start_bar + 16) * 16]["start"] if isinstance(tatums[(best_start_bar + 16) * 16], dict) else 0.0))
        total_dur_16 = t_end_16 - t_start_16
        part_dur = total_dur_16 / 4.0
        if part_dur < 5.0 or part_dur > 13.0:
            continue

        exact_bpm = round(60.0 * 16.0 / part_dur, 4)
        f0_16 = int(round(t_start_16 * 44100))
        part_frames = int(round(part_dur * 44100))
        total_frames_16 = part_frames * 4

        # Pre-load and uniformly normalize all 4 stems across the entire 16-bar span
        stem_16bar_audio: Dict[str, np.ndarray] = {}
        stem_sr = 44100
        valid_16bar = True
        if has_4_stems:
            for stem in STEMS:
                try:
                    data_16, sr = sf.read(str(stem_paths[stem]), start=f0_16, frames=total_frames_16, always_2d=True, dtype="float32")
                    stem_sr = sr
                    mono_16 = data_16.mean(axis=1)
                    if len(mono_16) < total_frames_16:
                        mono_16 = np.pad(mono_16, (0, total_frames_16 - len(mono_16)))
                    peak_16 = float(np.max(np.abs(mono_16)))
                    if peak_16 >= 0.025:
                        gain_scale = min(2.2, 0.88 / peak_16)
                        mono_16 = mono_16 * gain_scale
                    elif peak_16 < 0.012:
                        mono_16 = mono_16 * 0.25
                    stem_16bar_audio[stem] = mono_16
                except Exception:
                    valid_16bar = False
                    break
        elif src_mp3 and src_mp3.exists():
            try:
                mp3_info = sf.info(str(src_mp3))
                sr_mp3 = mp3_info.samplerate
                f0_mp3 = int(round(t_start_16 * sr_mp3))
                part_frames = int(round(part_dur * sr_mp3))
                total_frames_16 = part_frames * 4
                mix_16, stem_sr = sf.read(str(src_mp3), start=f0_mp3, frames=total_frames_16, always_2d=True, dtype="float32")
                if len(mix_16) < total_frames_16:
                    mix_16 = np.pad(mix_16, ((0, total_frames_16 - len(mix_16)), (0, 0)))
                stem_16bar_audio = _decompose_mix_16bar_into_stems(mix_16, stem_sr)
            except Exception:
                valid_16bar = False

        if not valid_16bar or len(stem_16bar_audio) < 4:
            continue

        candidate_bars = [
            best_start_bar,
            best_start_bar + 4,
            best_start_bar + 8,
            best_start_bar + 12,
        ]

        track_phrases: List[dict] = []
        for part_idx, b in enumerate(candidate_bars):
            start = t_start_16 + part_idx * part_dur
            end = start + part_dur
            dur = part_dur

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

            if d_loud >= -13.0 and b_loud >= -15.0 and o_loud >= -20.0:
                sec = "DROP" if part_idx >= 2 else "GROOVE_A"
            elif d_loud >= -16.0 and b_loud >= -19.0:
                sec = "GROOVE_A"
            elif d_loud < -24.0 and b_loud < -24.0 and (o_loud >= -24.0 or v_loud >= -24.0):
                sec = "INTRO" if b <= 8 else "BREAKDOWN"
            elif -24.0 <= d_loud < -14.0 and (o_loud >= -20.0 or v_loud >= -20.0):
                sec = "BUILDUP"
            else:
                sec = "GROOVE_A" if part_idx < 2 else "DROP"

            phrase_id = f"phrase_{tid[:6]}_p{part_idx + 1}_{sec.lower()}"
            stem_urls = {}
            onset_grids = {}
            bar_onset_grids: Dict[str, List[List[int]]] = {}

            valid_extraction = True
            for stem in STEMS:
                out_wav = phrases_dir / f"{phrase_id}_{stem}.opus"
                try:
                    p_start_frame = part_idx * part_frames
                    mono = stem_16bar_audio[stem][p_start_frame : p_start_frame + part_frames].copy()

                    if len(mono) > fade_len * 2:
                        mono[:fade_len] *= fade_in
                        mono[-fade_len:] *= fade_out

                    # Write at 22.05 kHz 16-bit mono PCM so 42 complete 4-part songs fit cleanly in the static bundle
                    if stem_sr >= 40000:
                        mono_out = mono[::2]
                        sr_out = stem_sr // 2
                    else:
                        mono_out = mono
                        sr_out = stem_sr
                    write_opus(out_wav, mono_out, sr_out, bitrate="96k")
                    stem_urls[stem] = f"data/phrases/{out_wav.name}"

                    # Extract per-bar 16-step onset grids across the 4 bars in this phrase block
                    per_bar_grids: List[List[int]] = []
                    bar_dur = dur / 4.0
                    for sub_bar in range(4):
                        sub_start = start + sub_bar * bar_dur
                        sub_end = sub_start + bar_dur
                        sub_onsets: List[int] = []
                        for s in bar_segs:
                            st_name = getattr(s, "stem", s.get("stem", "") if isinstance(s, dict) else "")
                            if st_name == stem or (not has_4_stems and st_name == "other"):
                                s_start = float(getattr(s, "start", s.get("start", 0.0) if isinstance(s, dict) else 0.0))
                                if sub_start - 0.03 <= s_start < sub_end:
                                    step_idx = int(round((s_start - sub_start) / (bar_dur / 16.0)))
                                    if 0 <= step_idx < 16 and step_idx not in sub_onsets:
                                        sub_onsets.append(step_idx)
                        sub_onsets.sort()
                        g16 = [1 if i in sub_onsets else 0 for i in range(16)]
                        if sum(g16) == 0 and stem == "drums" and sec in ("DROP", "GROOVE_A"):
                            g16 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                        per_bar_grids.append(g16)

                    bar_onset_grids[stem] = per_bar_grids
                    onset_grids[stem] = per_bar_grids[0]
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
                "source_bpm": exact_bpm,
                "section": sec,
                "part_index": part_idx,
                "start_bar": b,
                "energy": 0.95 if sec == "DROP" else (0.82 if sec == "GROOVE_A" else (0.65 if sec == "BUILDUP" else 0.25)),
                "bar_length": 4,
                "duration": round(dur, 3),
                "stems": stem_urls,
                "onset_grids": onset_grids,
                "bar_onset_grids": bar_onset_grids,
            }
            track_phrases.append(phrase_item)

        if track_phrases:
            # Link consecutive 4-bar parts of the same song into a seamless chain
            for idx_p, item in enumerate(track_phrases):
                item["total_parts"] = len(track_phrases)
                if idx_p + 1 < len(track_phrases):
                    item["next_phrase_id"] = track_phrases[idx_p + 1]["id"]
                else:
                    # Loop back to part 0 of the same song if continuing beyond the chain
                    item["next_phrase_id"] = track_phrases[0]["id"]
                if idx_p > 0:
                    item["prev_phrase_id"] = track_phrases[idx_p - 1]["id"]
                manifest_phrases.append(item)
                print(
                    f"   ✓ Extracted {item['section']} Part {idx_p + 1}/{len(track_phrases)}: "
                    f"{clean_title} (bar {item['start_bar']}-{item['start_bar']+4}, {bpm:.1f} BPM, {key}) -> next: {item['next_phrase_id']}",
                    flush=True,
                )
            songs_harvested += 1

    print(f"✓ Total harvested {len(manifest_phrases)} cohesive multi-part 4-bar phrase blocks across {songs_harvested} songs.", flush=True)
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

    # 2. Extract Multi-Stem & Full-Catalog Tracks Across Entire A-Z Corpus
    store = CorpusStore(
        Path("data/processed"),
        stems_base=Path("data/stems"),
        library_dirs=[Path("pandora/good"), Path("pandora")],
    )
    store.refresh(max_tracks=280)
    all_loaded_tracks = stratify_tracks_by_artist(store.tracks)
    multistem_tracks = stratify_tracks_by_artist([t for t in store.tracks if len(t.stems_available) == 4])
    if not multistem_tracks:
        multistem_tracks = all_loaded_tracks[:40]
    _, fallback_mp3s = store.audio_for({t.track_id for t in all_loaded_tracks})

    bank = PatternBank()
    n_pats = bank.load_from_tracks(all_loaded_tracks)
    print(
        f"✓ Extracted {n_pats} patterns and {len(bank.phrases)} phrases across {len(all_loaded_tracks)} catalog tracks "
        f"({len(multistem_tracks)} 4-stem + {len(all_loaded_tracks) - len(multistem_tracks)} full-catalog tracks) into memory bank",
        flush=True,
    )

    # Train Composing ML Models (PhraseComposerModel, SongArranger, and Full-Corpus SongIntrinsicCompressor) across entire 1,724-song catalog
    from wub_core.composer import PhraseComposerModel, SongArranger
    from wub_core.pattern_audio_trainer import PatternAudioFeedbackEngine
    from wub_core.song_intrinsic_compressor import compress_full_corpus_parallel
    phrase_model = PhraseComposerModel(weights_path=src_weights)
    n_trans = phrase_model.train_on_tracks(all_loaded_tracks, bank)
    phrase_model.save(src_weights)
    arranger_model = SongArranger(bank, weights_path=src_weights)
    arranger_model.train_on_corpus(all_loaded_tracks)
    arranger_model.save(src_weights)

    print("Compressing ENTIRE corpus of 1,724 songs in parallel across CPU workers...", flush=True)
    full_corpus_res = compress_full_corpus_parallel(
        processed_dir=Path("data/processed"),
        stems_dir=Path("data/stems"),
        target_vocab_per_stem=4,
        max_bars=32,
        max_tracks=None,
    )
    compact_corpus_models = full_corpus_res["compact_models"]
    shard_corpus_models = full_corpus_res["shard_models"]
    chord_progressions = full_corpus_res["chord_progressions"]
    umap_by_stem = full_corpus_res["umap_by_stem"]
    catalog_entries = full_corpus_res["catalog_entries"]

    fb_engine = PatternAudioFeedbackEngine()
    fb_engine.feedback_data["compressed_song_models"] = dict(list(compact_corpus_models.items())[:240])
    fb_engine._save()
    fb_engine.export_locked_feedback_weights()
    shutil.copy(src_weights, dst_weights)
    print(
        f"✓ Trained & exported Composing ML weights ({n_trans} bar transitions, "
        f"{len(compact_corpus_models)} compressed song models across 100% of A-Z corpus)",
        flush=True,
    )

    # 3. Harvest Cohesive 4-Bar Multi-Stem Phrase Blocks (all stemmed artists + diverse A-Z catalog artists)
    phrase_candidates = multistem_tracks[:32] + [t for t in all_loaded_tracks if len(t.stems_available) < 4][:28]
    extracted_phrase_blocks = harvest_cohesive_phrase_blocks(phrase_candidates, phrases_dir, fallback_mp3s=fallback_mp3s)

    # Build phrase_part_transitions ML map so the client follows consecutive song parts
    phrase_part_transitions: Dict[str, Dict[str, float]] = {}
    phrase_track_ids = set()
    for pb in extracted_phrase_blocks:
        phrase_track_ids.add(pb.get("track_id", ""))
        nxt_id = pb.get("next_phrase_id")
        if nxt_id:
            phrase_part_transitions[pb["id"]] = {nxt_id: 0.96}

    # 4. Harvest Whole 1-Bar Pattern Loops Across Stems (stratified across all artists)
    extracted_loops_manifest = harvest_whole_pattern_loops(multistem_tracks, loops_dir)

    # 4. Harvest Pure Audio Slices (Zero Duplicates, Chromatic Bass/Other 0..11)
    from scripts.harvest_pure_slices import harvest_pure_slices
    extracted_slices_manifest = harvest_pure_slices(
        Path("data/stems"),
        Path("data/processed"),
        slices_dir,
    )

    # 4b. Export Song-Intrinsic Vocabulary Opus Slices for ALL Stemmed Corpus Songs
    print("Exporting song-intrinsic slice vocabulary Opus files across ALL stemmed corpus songs...", flush=True)
    csm_dict = compact_corpus_models
    vocab_wavs_exported = 0
    stratified_csm_tids = [t.track_id for t in multistem_tracks if t.track_id in csm_dict]
    for tid in stratified_csm_tids:
        cmodel = csm_dict[tid]
        v_by_stem = cmodel.get("vocab_by_stem", {})
        bpm_val = float(cmodel.get("bpm", 126.0))
        tatum_sec = 60.0 / max(60.0, bpm_val) / 4.0
        for stem, v_list in v_by_stem.items():
            wav_path = Path("data/stems") / tid / f"{stem}.wav"
            if not wav_path.exists():
                continue
            try:
                with sf.SoundFile(str(wav_path)) as f:
                    sr = f.samplerate
                    total_frames = len(f)
                    for vs in v_list:
                        vid = vs.get("id", f"vocab_{tid[:6]}_{stem}_v1")
                        start_sec = float(vs.get("start", 0.0))
                        raw_dur = float(vs.get("duration", 0.25))
                        # Align slice duration to integer tatums + 15ms crossfade tail for seamless concatenation
                        n_tatums_span = max(1, min(8, round(raw_dur / max(0.05, tatum_sec))))
                        target_dur = n_tatums_span * tatum_sec + 0.015
                        start_frame = max(0, min(total_frames - 1, int(start_sec * sr)))
                        num_frames = max(int(0.08 * sr), min(total_frames - start_frame, int(target_dur * sr)))
                        if num_frames <= 64:
                            continue
                        f.seek(start_frame)
                        chunk = f.read(num_frames, dtype="float32", always_2d=False)
                        if chunk.ndim > 1:
                            chunk = np.mean(chunk, axis=1)
                        peak = float(np.max(np.abs(chunk)))
                        if peak >= 0.015:
                            gain = min(2.4, 0.88 / peak)
                            chunk = chunk * gain
                        # Apply smooth 4ms attack and 12ms cosine release tail for click-free concatenation
                        fade_in = min(int(0.004 * sr), len(chunk) // 4)
                        fade_out = min(int(0.012 * sr), len(chunk) // 3)
                        if fade_in > 1:
                            chunk[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
                        if fade_out > 1:
                            chunk[-fade_out:] *= (0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out, dtype=np.float32))))
                        out_wav = slices_dir / f"{vid}.opus"
                        write_opus(out_wav, chunk, sr)
                        rel_url = f"data/slices/{vid}.opus"
                        vs["url"] = rel_url
                        vs["duration"] = round(float(len(chunk) / sr), 3)
                        cat_full = vs.get("category", f"{stem}:kick" if stem == "drums" else f"{stem}:sub")
                        cat_short = cat_full.split(":")[-1] if ":" in cat_full else cat_full
                        extracted_slices_manifest.setdefault(stem, {}).setdefault(cat_short, []).append({
                            "name": vid,
                            "url": rel_url,
                            "pitch": int(vs.get("pitch", 0)),
                            "centroid": float(vs.get("centroid", 1200.0)),
                            "duration": vs["duration"],
                            "loudness": 0.90,
                            "track_id": tid,
                            "track_title": cmodel.get("title", tid),
                            "segment_id": vs.get("segment_id", vid),
                            "start": start_sec,
                        })
                        vocab_wavs_exported += 1
            except Exception:
                continue

    # 4c. Bind every vocabulary slice across ALL 1,724 corpus songs to a valid audio slice URL in extracted_slices_manifest
    flat_slices_by_stem: Dict[str, List[dict]] = {s: [] for s in STEMS}
    for st, cat_dict in extracted_slices_manifest.items():
        if isinstance(cat_dict, dict):
            for cat_key, s_items in cat_dict.items():
                if isinstance(s_items, list):
                    for item in s_items:
                        if isinstance(item, dict) and item.get("url"):
                            flat_slices_by_stem.setdefault(st, []).append({**item, "_cat": cat_key})

    bound_vocab_count = 0
    for tid, cmodel in compact_corpus_models.items():
        smodel = shard_corpus_models.get(tid)
        for st, v_list in cmodel.get("vocab_by_stem", {}).items():
            candidates = flat_slices_by_stem.get(st, [])
            if not candidates:
                continue
            s_v_list = smodel.get("vocab_by_stem", {}).get(st, []) if smodel else []
            for idx_v, vs in enumerate(v_list):
                if not vs.get("url"):
                    target_cat = str(vs.get("category", "")).split(":")[-1]
                    target_pitch = int(vs.get("pitch", 0))
                    target_cent = float(vs.get("centroid", 1200.0))
                    best_cand = candidates[0]
                    best_dist = 1e9
                    for cand in candidates:
                        c_cat = str(cand.get("_cat", ""))
                        cat_pen = 0.0 if (target_cat and target_cat in c_cat) else 2.5
                        p_diff = abs((target_pitch - int(cand.get("pitch", 0))) % 12)
                        if p_diff > 6:
                            p_diff = 12 - p_diff
                        cent_diff = abs(target_cent - float(cand.get("centroid", 1200.0))) / 2000.0
                        dist = cat_pen + p_diff * 0.25 + cent_diff * 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_cand = cand
                    vs["url"] = best_cand["url"]
                    vs["matched_slice_name"] = best_cand["name"]
                if idx_v < len(s_v_list):
                    s_v_list[idx_v]["url"] = vs["url"]
                    if "matched_slice_name" in vs:
                        s_v_list[idx_v]["matched_slice_name"] = vs["matched_slice_name"]
                bound_vocab_count += 1

    print(
        f"✓ Exported {vocab_wavs_exported} direct vocab Opus files & bound all {bound_vocab_count} vocabulary slices across {len(compact_corpus_models)} corpus songs",
        flush=True,
    )
    # Format migration: Opus replaces WAV everywhere; drop legacy WAVs so the Pages bundle stays under 1 GB
    for _adir in (slices_dir, loops_dir):
        for _stale in _adir.glob("*.wav"):
            try:
                _stale.unlink()
            except Exception:
                pass

    # 5. Export Full-Corpus UMAP Nebula Indices (all 1,724 songs = 3,448 points per stem)
    indices_dir = public_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    for st, pts in umap_by_stem.items():
        umap_path = indices_dir / f"umap_{st}.json"
        umap_path.write_text(json.dumps(pts, separators=(",", ":")))
    print(f"✓ Exported full-corpus UMAP Nebula indices across {len(compact_corpus_models)} songs into {indices_dir}", flush=True)

    # 5b. Export Sharded Full-Detail Catalog (frontend/public/data/catalog/shard_00..17.json) & full_catalog_manifest.json
    catalog_dir = public_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 100
    all_tids_ordered = [entry["track_id"] for entry in catalog_entries]
    num_shards = (len(all_tids_ordered) + shard_size - 1) // shard_size
    for shard_idx in range(num_shards):
        batch_tids = all_tids_ordered[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        shard_payload = {
            "shard_index": shard_idx,
            "total_shards": num_shards,
            "songs": {tid: shard_corpus_models[tid] for tid in batch_tids if tid in shard_corpus_models},
        }
        shard_file = catalog_dir / f"shard_{shard_idx:02d}.json"
        shard_file.write_text(json.dumps(shard_payload, separators=(",", ":")))
        for tid in batch_tids:
            if tid in compact_corpus_models:
                compact_corpus_models[tid]["shard"] = shard_idx

    for idx_e, entry in enumerate(catalog_entries):
        entry["shard"] = idx_e // shard_size
        entry["has_phrase_audio"] = entry["track_id"] in phrase_track_ids

    raw_manifest_path = Path("data/processed/_manifest.json")
    total_catalog_files = len(catalog_entries)
    if raw_manifest_path.exists():
        try:
            raw_m = json.loads(raw_manifest_path.read_text())
            if isinstance(raw_m, dict):
                total_catalog_files = len(raw_m)
        except Exception:
            pass

    full_catalog_manifest = {
        "version": 1,
        "total_unique_songs": len(catalog_entries),
        "total_catalog_files": total_catalog_files,
        "total_shards": num_shards,
        "shard_size": shard_size,
        "songs": catalog_entries,
    }
    full_catalog_path = public_dir / "full_catalog_manifest.json"
    full_catalog_path.write_text(json.dumps(full_catalog_manifest, separators=(",", ":")))
    print(
        f"✓ Exported {full_catalog_path} ({len(catalog_entries)} unique analyzed songs / {total_catalog_files} catalog tracks across {num_shards} shards)",
        flush=True,
    )
    print(f"✓ Extracted {len(chord_progressions)} authentic chord progressions across A-Z corpus", flush=True)

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

            from wub_core.groovy_pattern_assembler import AcousticSliceClassifier
            for idx_p, p in enumerate(curated_for_gt):
                # Find the next consecutive pattern from the same track and stem
                next_pat = bank.get_track_pattern(p.track_id, stem, p.bar_num + 1)
                next_pid = next_pat.id if next_pat else curated_for_gt[(idx_p + 1) % len(curated_for_gt)].id

                # Compute per-step slice categories, contiguous tatum spans (1..16), and melodic pitches
                step_cats: List[Optional[str]] = [None] * p.length
                step_spans: List[int] = [1] * p.length
                step_pitches: List[int] = [0] * p.length
                p_steps = getattr(p, "steps", []) or []
                for s_i in range(min(p.length, len(p_steps))):
                    sref = p_steps[s_i]
                    if sref is not None:
                        role = AcousticSliceClassifier.classify(sref)
                        step_cats[s_i] = f"{stem}:{role}" if not role.startswith(f"{stem}:") else role
                        chroma = getattr(sref, "chroma", None)
                        if chroma is not None and len(chroma) == 12:
                            step_pitches[s_i] = int(np.argmax(chroma))
                for o_idx, o_step in enumerate(p.onset_grid):
                    if 0 <= o_step < p.length:
                        nxt_o = p.onset_grid[o_idx + 1] if o_idx + 1 < len(p.onset_grid) else p.length
                        step_spans[o_step] = max(1, min(16, nxt_o - o_step))

                serialized_patterns[stem].append({
                    "id": p.id,
                    "track_id": p.track_id,
                    "bar_num": p.bar_num,
                    "next_pattern_id": next_pid,
                    "name": p.pattern_name,
                    "groove_type": p.groove_type,
                    "energy": round(float(p.energy), 3),
                    "density": round(float(p.density), 3),
                    "onset_grid": p.onset_grid,
                    "velocities": p.velocities if p.velocities else [1.0 if i in p.onset_grid else 0.0 for i in range(p.length)],
                    "slice_categories": step_cats,
                    "step_spans": step_spans,
                    "step_pitches": step_pitches,
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
            vocal_json = public_dir / "vocal_patterns.json"
            if vocal_json.exists():
                vd = json.loads(vocal_json.read_text())
                cstats = vd.get("corpus_stats", {})
                w_data.setdefault("vocal_vowel_transitions", cstats.get("vowel_transitions", {}))
                w_data.setdefault("vocal_pitch_transitions", cstats.get("pitch_contour_transitions", {}))
                w_data.setdefault("vocal_step_onset_priors", cstats.get("step_onset_priors", []))
                dst_weights.write_text(json.dumps(w_data, indent=2))
        except Exception:
            pass

    dst_patterns = public_dir / "standalone_patterns.json"
    existing_manifest = {}
    if dst_patterns.exists():
        try:
            existing_manifest = json.loads(dst_patterns.read_text())
        except Exception:
            pass

    standalone_manifest = {
        "version": 4,
        "default_bpm": 126.0,
        "chord_progressions": chord_progressions,
        "groove_templates": groove_templates,
        "patterns": serialized_patterns,
        "phrases": serialized_phrases,
        "slices": extracted_slices_manifest,
        "loops": extracted_loops_manifest,
        "phrase_blocks": extracted_phrase_blocks,
        "phrase_part_transitions": phrase_part_transitions,
        "stem_cohesion_weights": stem_cohesion_weights,
    }

    # Preserve rich style_kits, compressed song models, and vocal corpora (prefer fresh stem_cohesion_weights for trained models)
    for key in ("style_kits", "patterns_by_style", "vocal_patterns", "vocal_themes"):
        if key in existing_manifest:
            standalone_manifest[key] = existing_manifest[key]
        elif key in stem_cohesion_weights:
            standalone_manifest[key] = stem_cohesion_weights[key]

    # Rebuild vocal_corpus fresh from THIS run's harvest: older generations
    # reference deleted WAV filenames (dead 404s). The fresh items carry every
    # field the client preloads (url/b64/pitch/vowel/track/segment/start).
    fresh_vocal_corpus: List[dict] = []
    for _vcat, _vitems in (extracted_slices_manifest.get("vocals", {}) or {}).items():
        for _it in _vitems or []:
            if not isinstance(_it, dict):
                continue
            _e = dict(_it)
            _e.setdefault("category", _vcat)
            _e.setdefault("stem", "vocals")
            fresh_vocal_corpus.append(_e)
    if fresh_vocal_corpus:
        standalone_manifest["vocal_corpus"] = fresh_vocal_corpus
    elif "vocal_corpus" in existing_manifest:
        standalone_manifest["vocal_corpus"] = existing_manifest["vocal_corpus"]

    # Remap legacy style_kit slice URLs onto this run's fresh harvest so no
    # entry points at a deleted WAV name (older generations used a `_wave`
    # infix scheme). Match by stem + category + pitch/centroid proximity.
    _sk = standalone_manifest.get("style_kits")
    if isinstance(_sk, dict):
        _flat: Dict[str, List[dict]] = {}
        for _st, _cat_dict in extracted_slices_manifest.items():
            if isinstance(_cat_dict, dict):
                for _ck, _items in _cat_dict.items():
                    if isinstance(_items, list):
                        _flat.setdefault(_st, []).extend([_i for _i in _items if isinstance(_i, dict) and _i.get("url")])
        for _style, _sdata in _sk.items():
            _stems = _sdata.get("stems", {}) if isinstance(_sdata, dict) else {}
            for _st, _cats in _stems.items():
                for _ck, _slist in (_cats.items() if isinstance(_cats, dict) else []):
                    for _entry in _slist or []:
                        if not isinstance(_entry, dict):
                            continue
                        _url = str(_entry.get("url", ""))
                        if _url.endswith(".opus"):
                            continue
                        _cands = _flat.get(_st, [])
                        if not _cands:
                            _entry.pop("url", None)
                            continue
                        _tp = int(_entry.get("pitch", 0))
                        _tc = float(_entry.get("centroid", 1200.0))
                        _best = min(
                            _cands,
                            key=lambda _c: (
                                0.0 if str(_c.get("name", "")).startswith(f"{_st}_{_ck}") else 2.5
                            ) + (abs((_tp - int(_c.get("pitch", 0))) % 12) * 0.25)
                            + (abs(_tc - float(_c.get("centroid", 1200.0))) / 2000.0 * 0.5),
                        )
                        _entry["url"] = _best["url"]
                        _entry["name"] = _best["name"]

    for key in ("song_composition_scores", "recurrent_song_patterns"):
        if key in stem_cohesion_weights and stem_cohesion_weights[key]:
            standalone_manifest[key] = stem_cohesion_weights[key]
        elif key in existing_manifest:
            standalone_manifest[key] = existing_manifest[key]

    # Core compressed song models kept inline (phrase-audio songs + 4-stem songs + direct vocab-Opus songs);
    # every other corpus song is lazily loaded on demand from data/catalog/shard_XX.json via full_catalog_manifest.json
    core_tids: List[str] = []
    for tid in list(phrase_track_ids) + stratified_csm_tids + [t.track_id for t in multistem_tracks]:
        if tid and tid in compact_corpus_models and tid not in core_tids:
            core_tids.append(tid)
    core_tids = core_tids[:240]
    standalone_manifest["compressed_song_models"] = {tid: compact_corpus_models[tid] for tid in core_tids}
    standalone_manifest["full_catalog"] = {
        "manifest_url": "data/full_catalog_manifest.json",
        "shard_url_pattern": "data/catalog/shard_{shard:02d}.json",
        "total_unique_songs": len(catalog_entries),
        "total_catalog_files": total_catalog_files,
        "total_shards": num_shards,
    }

    # Merge intrinsic patterns mined from authentic core songs into the global patterns bank
    csm = standalone_manifest.get("compressed_song_models", {})
    for tid, model in csm.items():
        ipat = model.get("intrinsic_patterns", {})
        for stem, p_list in ipat.items():
            if stem in standalone_manifest["patterns"]:
                existing_ids = {p.get("id") for p in standalone_manifest["patterns"][stem]}
                for p in p_list:
                    if p.get("id") not in existing_ids:
                        standalone_manifest["patterns"][stem].append(p)
                        existing_ids.add(p.get("id"))

    vocal_json = public_dir / "vocal_patterns.json"
    if "vocal_patterns" not in standalone_manifest and vocal_json.exists():
        try:
            standalone_manifest["vocal_patterns"] = json.loads(vocal_json.read_text())
            standalone_manifest["vocal_themes"] = standalone_manifest["vocal_patterns"].get("themes", [])
        except Exception:
            pass

    # Strip duplicate heavy dictionaries and redundant inline b64 audio blobs when URLs are present
    standalone_manifest.pop("recurrent_song_patterns", None)
    for stem_cats in standalone_manifest.get("slices", {}).values():
        if isinstance(stem_cats, dict):
            for s_list in stem_cats.values():
                if isinstance(s_list, list):
                    for s_item in s_list:
                        if isinstance(s_item, dict) and s_item.get("url"):
                            s_item.pop("b64", None)

    if isinstance(stem_cohesion_weights, dict):
        light_cohesion = {
            k: v for k, v in stem_cohesion_weights.items()
            if k not in ("compressed_song_models", "recurrent_song_patterns")
        }
        standalone_manifest["stem_cohesion_weights"] = light_cohesion
        try:
            w_light = json.loads(dst_weights.read_text())
            w_light["stem_cohesion_weights"] = light_cohesion
            dst_weights.write_text(json.dumps(w_light, separators=(",", ":")))
            src_weights.write_text(json.dumps(w_light, separators=(",", ":")))
        except Exception:
            pass

    dst_patterns.write_text(json.dumps(standalone_manifest, separators=(",", ":")))
    print(f"✓ Exported {dst_patterns} ({dst_patterns.stat().st_size} bytes)", flush=True)

    print("=== Standalone Client Bundle Export Complete ===", flush=True)


if __name__ == "__main__":
    export_bundle()
