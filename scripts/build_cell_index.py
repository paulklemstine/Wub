#!/usr/bin/env python3
"""Build multi-scale compositional cell index (`cell_index.json`) for Wub KG-505.

Analyzes all phrases in `frontend/public/data/kits/kit_manifest.json` across the
4 stems (`drums`, `bass`, `other`, `vocals`), using `mix_compat_onsets.f32`,
available `data/processed/<track_id>.json` tatum segments, and the shipped `.opus`
stem files to compute per-bar:
  - motif schemas (`AAAB`, `ABAB`, `ABAC`, `AABB`, `ABCD`)
  - tonal roots & triad qualities (`minor`, `major`, `sus`, `mono`, `perc`)
  - 16-bit 16th-note onset bitmasks (`bar_onsets`)
  - acoustic cell purity scores (`0..1`)
  - drum bar-3 turnaround fill detection
  - vocal compositional roles (`hook_1bar`, `hook_2bar`, `chop`, `airy`, `verse`, `teaser`)
  - empirical style progressions and section-conditioned motif schema priors

Outputs:
  - `frontend/public/data/kits/cell_index.json`
  - Updates `"cell_index": true` in `frontend/public/data/kits/sidecars.json`
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wub_core.essential_slices import (  # noqa: E402
    ESSENTIAL_ROLES,
    mine_processed_track_essential_bars,
    score_bar_seam_continuity,
    select_essential_slices_for_song,
)
from wub_core.motif_miner import (  # noqa: E402
    classify_motif_schema,
    classify_vocal_role,
    classify_vowel_from_centroid,
    detect_drum_turnaround_fill,
    extract_phrase_bar_features,
)
from wub_core.theory_engine import (  # noqa: E402
    MIN_RMS_DB,
    STEMS,
    STYLE_PROGRESSIONS,
    classify_triad_quality,
    parse_key,
    score_cell_purity,
    set_to_onset_bits,
)

# Worker globals for fast multi-process STFT chroma lookup tables
_G_TABLES: Dict[str, Any] = {}


def _init_worker(kits_root: str) -> None:
    sr = 48000
    tables: Dict[str, Any] = {"kits_root": kits_root, "sr": sr}

    for name, n_fft, fmin, fmax, decay_pow in (
        ("bass", 16384, 32.0, 320.0, 0.75),
        ("tonal", 8192, 85.0, 4200.0, 0.15),
    ):
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        valid = (freqs >= fmin) & (freqs <= fmax)
        f_v = freqs[valid]
        midi = 69.0 + 12.0 * np.log2(np.maximum(f_v, 1.0) / 440.0)
        pc = np.round(midi).astype(np.int32) % 12
        dist = np.abs(midi - np.round(midi))
        weight_bin = np.exp(-(dist ** 2) / (2.0 * 0.28 ** 2)) / np.maximum(f_v / 60.0, 1.0) ** decay_pow
        bin_to_pc = np.zeros((f_v.shape[0], 12), dtype=np.float32)
        bin_to_pc[np.arange(f_v.shape[0]), pc] = weight_bin.astype(np.float32)
        win = np.hanning(n_fft).astype(np.float32)
        tables[name] = {
            "n_fft": n_fft,
            "valid": valid,
            "f_v": f_v.astype(np.float32),
            "bin_to_pc": bin_to_pc,
            "win": win,
        }
    _G_TABLES.update(tables)


def _onsets_from_row(row_64: Optional[Sequence[float]]) -> Tuple[List[int], List[int], List[float]]:
    """Extract per-bar 16-bit onset masks, counts, and energies from a 64-step onset strength row."""
    if row_64 is None:
        return [0, 0, 0, 0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]

    arr = np.asarray(row_64, dtype=np.float32)
    bar_masks: List[int] = []
    bar_counts: List[int] = []
    bar_energies: List[float] = []
    max_val = float(arr.max()) if arr.size else 0.0
    thresh = max(0.16, 0.28 * max_val)

    for b in range(4):
        seg = arr[b * 16 : (b + 1) * 16]
        steps: List[int] = []
        for s in range(16):
            v = float(seg[s])
            prev_v = float(seg[s - 1]) if s > 0 else 0.0
            next_v = float(seg[s + 1]) if s < 15 else 0.0
            if v >= thresh and (v >= prev_v * 0.92 or v >= max_val * 0.45) and (v >= next_v * 0.85 or s == 0):
                steps.append(s)
        if not steps and seg.size and float(seg.max()) > 0.12:
            steps.append(int(np.argmax(seg)))
        bar_masks.append(set_to_onset_bits(steps))
        bar_counts.append(len(steps))
        bar_energies.append(float(seg.mean()) if seg.size else 0.0)

    return bar_masks, bar_counts, bar_energies


def _onsets_from_audio(bars: np.ndarray) -> Tuple[List[int], List[int], List[float]]:
    """Fallback 16-step per-bar onset estimator directly from decoded audio."""
    bar_masks: List[int] = []
    bar_counts: List[int] = []
    bar_energies: List[float] = []
    bar_rms = np.sqrt(np.mean(bars ** 2, axis=1) + 1e-12)
    ref_rms = max(float(bar_rms.max()), 1e-4)

    for b in range(4):
        seg = bars[b]
        n16 = (len(seg) // 16) * 16
        if n16 < 16:
            bar_masks.append(0)
            bar_counts.append(0)
            bar_energies.append(0.0)
            continue
        sub = seg[:n16].reshape(16, -1)
        env = np.sqrt(np.mean(sub ** 2, axis=1) + 1e-12)
        norm_env = env / max(float(env.max()), ref_rms * 0.25, 1e-4)
        steps: List[int] = []
        for s in range(16):
            prev_e = float(norm_env[s - 1]) if s > 0 else 0.0
            curr_e = float(norm_env[s])
            if curr_e >= 0.35 and (curr_e > prev_e * 1.15 or s == 0):
                steps.append(s)
        bar_masks.append(set_to_onset_bits(steps))
        bar_counts.append(len(steps))
        bar_energies.append(float(bar_rms[b] / ref_rms))
    return bar_masks, bar_counts, bar_energies


def _compute_bar_chroma_and_centroid(seg: np.ndarray, mode: str) -> Tuple[List[float], float, float]:
    """Compute 12-bin chroma, spectral centroid (Hz), and RMS (linear) for a 1-bar segment."""
    rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    if rms < 1e-4 or len(seg) < 512:
        return [0.0] * 12, 0.0, rms

    tbl = _G_TABLES["bass" if mode == "bass" else "tonal"]
    n_fft: int = tbl["n_fft"]
    valid: np.ndarray = tbl["valid"]
    f_v: np.ndarray = tbl["f_v"]
    bin_to_pc: np.ndarray = tbl["bin_to_pc"]
    win: np.ndarray = tbl["win"]

    if len(seg) < n_fft:
        pad = np.zeros(n_fft, dtype=np.float32)
        pad[: len(seg)] = seg
        frames = (pad * win)[None, :]
    else:
        hop = n_fft // 4
        starts = range(0, len(seg) - n_fft + 1, hop)
        frames = np.stack([seg[i : i + n_fft] * win for i in starts], axis=0)

    mag = np.abs(np.fft.rfft(frames, axis=1))[:, valid]
    spec = mag.mean(axis=0)
    if mode == "bass":
        # Square low-frequency magnitudes so fundamental pitch outweighs overtones
        chroma = ((spec ** 1.5) @ bin_to_pc).tolist()
    else:
        chroma = (spec @ bin_to_pc).tolist()

    spec_sum = float(spec.sum())
    centroid = float((spec * f_v).sum() / spec_sum) if spec_sum > 1e-8 else 0.0
    return chroma, centroid, rms


def _estimate_mean_seg_dur(bars: np.ndarray, sr: int, onset_counts: Sequence[int]) -> float:
    total_onsets = sum(onset_counts)
    total_dur = (bars.size / max(sr, 1))
    if total_onsets <= 0:
        return 0.45
    return round(min(2.0, max(0.08, total_dur / max(1, total_onsets))), 3)


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def _segment_centroid_norm(seg: np.ndarray, sr: int) -> float:
    if seg.size < 256:
        return 0.45
    n_fft = 2048
    if len(seg) < n_fft:
        pad = np.zeros(n_fft, dtype=np.float32)
        pad[: len(seg)] = seg
        chunk = pad
    else:
        mid = (len(seg) - n_fft) // 2
        chunk = seg[mid : mid + n_fft]
    win = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(chunk * win))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / max(1, sr))
    valid = (freqs >= 30.0) & (freqs <= 8500.0)
    s_v = spec[valid]
    tot = float(s_v.sum())
    if tot <= 1e-7:
        return 0.45
    cent = float((s_v * freqs[valid]).sum() / tot)
    return _clamp01(cent / 4500.0)


def _compute_bar_head_tail_seam(
    bars_audio: Optional[np.ndarray],
    sr: int,
    bar_energies: Sequence[float],
    bar_purities: Sequence[float],
    bar_masks: Sequence[int],
    bar_roots: Sequence[int],
    bar_qualities: Sequence[str],
    stem: str,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    hr: List[float] = []
    tr: List[float] = []
    hc: List[float] = []
    tc: List[float] = []

    if bars_audio is not None and bars_audio.shape[0] >= 4 and bars_audio.shape[1] >= 64:
        q_len = max(32, bars_audio.shape[1] // 4)
        for b in range(4):
            seg = bars_audio[b]
            h_seg = seg[:q_len]
            t_seg = seg[-q_len:]
            h_rms = float(np.sqrt(np.mean(h_seg ** 2) + 1e-12))
            t_rms = float(np.sqrt(np.mean(t_seg ** 2) + 1e-12))
            h_db = 20.0 * math.log10(max(1e-5, h_rms))
            t_db = 20.0 * math.log10(max(1e-5, t_rms))
            hr.append(round(_clamp01((h_db + 54.0) / 48.0), 2))
            tr.append(round(_clamp01((t_db + 54.0) / 48.0), 2))
            hc.append(round(_segment_centroid_norm(h_seg, sr) if h_rms >= 1e-4 else 0.45, 2))
            tc.append(round(_segment_centroid_norm(t_seg, sr) if t_rms >= 1e-4 else 0.45, 2))
    else:
        for b in range(4):
            base_r = round(_clamp01(0.55 * float(bar_purities[b]) + 0.45 * float(bar_energies[b])), 2)
            hr.append(base_r)
            tr.append(base_r)
            hc.append(0.45)
            tc.append(0.45)

    sm: List[float] = []
    for b in range(4):
        nb = (b + 1) % 4
        b0 = {
            "purity": float(bar_purities[b]),
            "mask": int(bar_masks[b]),
            "root": int(bar_roots[b]),
            "qual": str(bar_qualities[b]),
            "hr": hr[b],
            "tr": tr[b],
            "hc": hc[b],
            "tc": tc[b],
        }
        b1 = {
            "purity": float(bar_purities[nb]),
            "mask": int(bar_masks[nb]),
            "root": int(bar_roots[nb]),
            "qual": str(bar_qualities[nb]),
            "hr": hr[nb],
            "tr": tr[nb],
            "hc": hc[nb],
            "tc": tc[nb],
        }
        s_val = score_bar_seam_continuity(
            b0, b1, stem, is_contiguous=(b < 3), is_section_boundary=False
        )
        if bars_audio is not None and b < 3:
            w40 = min(bars_audio.shape[1] // 4, max(64, int(0.04 * sr)))
            tail_e = float(np.sqrt(np.mean(bars_audio[b][-w40:] ** 2) + 1e-12))
            head_e = float(np.sqrt(np.mean(bars_audio[nb][:w40] ** 2) + 1e-12))
            env_cont = _clamp01(1.0 - abs(tail_e - head_e) / max(tail_e, head_e, 0.05))
            s_val = 0.65 * s_val + 0.35 * env_cont
        sm.append(round(_clamp01(s_val), 2))
    return hr, tr, hc, tc, sm


def _process_kit(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process all phrases in one kit (runs in worker process)."""
    kit = task["kit"]
    tid = kit["track_id"]
    style = (kit.get("style") or "house").lower()
    key_str = kit.get("key") or "C minor"
    key_pc, key_mode = parse_key(key_str)
    onset_rows: Optional[Dict[str, np.ndarray]] = task.get("onset_rows")
    proc_path: Optional[str] = task.get("proc_path")
    kits_root = Path(_G_TABLES["kits_root"])

    proc_data: Optional[Dict[str, Any]] = None
    avail_stems: set[str] = set()
    if proc_path and os.path.exists(proc_path):
        try:
            with open(proc_path, "r") as f:
                proc_data = json.load(f)
            avail_stems = set(proc_data.get("stems_available") or [])
        except Exception:
            proc_data = None

    kit_results: Dict[str, Dict[str, Any]] = {}
    phrase_chroma_out: Dict[str, Dict[str, List[List[float]]]] = {}
    section_schema_samples: List[Tuple[str, str, str, float]] = []
    chord_seqs: List[Tuple[List[Tuple[int, str]], float]] = []
    kit_bars_by_stem: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STEMS}

    sorted_phrases = sorted(
        kit.get("phrases", []),
        key=lambda p: (int(p.get("start_bar", 0)), str(p.get("id", ""))),
    )

    for p_idx, phrase in enumerate(sorted_phrases):
        pid = phrase["id"]
        section = (phrase.get("section") or "GROOVE_A").upper()
        start_bar = int(phrase.get("start_bar", 0))
        stems_meta = phrase.get("stems") or {}
        onsets_4 = onset_rows.get(pid) if onset_rows is not None else None

        phrase_out: Dict[str, Any] = {}
        phrase_chroma: Dict[str, List[List[float]]] = {}
        bar_chord_votes: List[Tuple[int, str, float]] = [(-1, "mono", 0.0) for _ in range(4)]

        for s_idx, stem in enumerate(STEMS):
            sm = stems_meta.get(stem) or {}
            rms_db = float(sm.get("rms_db", -60.0))
            peak = float(sm.get("peak", 0.0))
            rel_file = sm.get("file")

            row_64 = onsets_4[s_idx] if onsets_4 is not None else None
            bar_masks, bar_counts, bar_energies = _onsets_from_row(row_64)

            proc_feat: Optional[Dict[str, Any]] = None
            if proc_data is not None and stem in avail_stems:
                proc_feat = extract_phrase_bar_features(proc_data, start_bar, stem, rms_db, peak)

            audio_path = (kits_root / rel_file) if rel_file else None
            bars_audio: Optional[np.ndarray] = None
            sr = 48000
            if audio_path and audio_path.exists():
                try:
                    y, sr = sf.read(str(audio_path), dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    n4 = (len(y) // 4) * 4
                    if n4 >= 4:
                        bars_audio = y[:n4].reshape(4, -1)
                except Exception:
                    bars_audio = None

            if row_64 is None and bars_audio is not None:
                bar_masks, bar_counts, bar_energies = _onsets_from_audio(bars_audio)

            bar_roots: List[int] = []
            bar_qualities: List[str] = []
            bar_purities: List[float] = []
            bar_vectors: List[List[float]] = []
            bar_chromas: List[List[float]] = []
            centroids: List[float] = []

            ref_rms = max(float(np.sqrt(np.mean(bars_audio ** 2) + 1e-12)) if bars_audio is not None else 10 ** (rms_db / 20.0), 1e-5)
            for b in range(4):
                bits = bar_masks[b]
                step_vec = [float((bits >> k) & 1) for k in range(16)]
                if bars_audio is not None:
                    b_seg = bars_audio[b]
                    b_rms = float(np.sqrt(np.mean(b_seg ** 2) + 1e-12))
                    b_peak = float(np.max(np.abs(b_seg))) if b_seg.size else peak
                    b_rms_db = 20.0 * math.log10(max(1e-5, b_rms))
                else:
                    b_rms_db = rms_db
                    b_peak = peak

                if stem == "drums":
                    bar_roots.append(-1)
                    bar_qualities.append("perc")
                    purity = score_cell_purity(stem, b_rms_db, b_peak, [0.0] * 12, bar_counts[b])
                    bar_purities.append(round(purity, 2))
                    bar_vectors.append(step_vec + [bar_energies[b], bar_counts[b] / 16.0])
                    bar_chromas.append([0.0] * 12)
                else:
                    if bars_audio is not None:
                        mode = "bass" if stem == "bass" else "tonal"
                        chroma, centroid, _ = _compute_bar_chroma_and_centroid(bars_audio[b], mode)
                        if centroid > 0:
                            centroids.append(centroid)
                    elif proc_feat is not None:
                        # Synthetic chroma around proc_feat root
                        chroma = [0.0] * 12
                        pr = proc_feat["bar_roots"][b]
                        if pr >= 0:
                            chroma[pr] = 1.0
                    else:
                        chroma = list(phrase.get("chroma") or [0.0] * 12)

                    if proc_feat is not None and proc_feat["bar_roots"][b] >= 0 and b_rms_db < MIN_RMS_DB[stem] - 4.0:
                        r_pc = proc_feat["bar_roots"][b]
                        qual = proc_feat["bar_qualities"][b]
                        clarity = 0.25
                    else:
                        r_pc, qual, clarity = classify_triad_quality(chroma)
                        if sum(chroma) <= 1e-6:
                            r_pc, qual = -1, "mono"

                    purity = score_cell_purity(stem, b_rms_db, b_peak, chroma, bar_counts[b])
                    if proc_feat is not None and stem == "bass" and r_pc < 0 and proc_feat["bar_roots"][b] >= 0:
                        r_pc = proc_feat["bar_roots"][b]
                        qual = proc_feat["bar_qualities"][b]

                    bar_roots.append(int(r_pc))
                    bar_qualities.append(str(qual))
                    bar_purities.append(round(purity, 2))

                    c_tot = sum(chroma) or 1.0
                    norm_c = [float(x) / c_tot for x in chroma]
                    bar_vectors.append(step_vec + norm_c)
                    bar_chromas.append(norm_c)

                    # Vote toward bar chord progression (`other` weighted more for harmony quality, `bass` for root)
                    if r_pc >= 0 and purity >= 0.30:
                        weight = purity * (1.25 if stem == "other" else 0.9 if stem == "bass" else 0.3)
                        if weight > bar_chord_votes[b][2]:
                            voted_q = qual if qual in ("minor", "major", "sus") else key_mode
                            bar_chord_votes[b] = (r_pc, voted_q, weight)

            schema = classify_motif_schema(bar_vectors)
            hr, tr, hc, tc, sm_arr = _compute_bar_head_tail_seam(
                bars_audio,
                sr,
                bar_energies,
                bar_purities,
                bar_masks,
                bar_roots,
                bar_qualities,
                stem,
            )
            entry: Dict[str, Any] = {
                "s": schema,
                "p": bar_purities,
                "r": bar_roots,
                "q": bar_qualities,
                "o": bar_masks,
                "hr": hr,
                "tr": tr,
                "hc": hc,
                "tc": tc,
                "sm": sm_arr,
            }

            if stem == "drums":
                is_fill, is_dfill = detect_drum_turnaround_fill(bar_counts, bar_energies)
                entry["fill"] = bool(is_fill)
                entry["dfill"] = bool(is_dfill)
            elif stem == "vocals":
                mean_seg = (
                    proc_feat["mean_seg_dur"]
                    if proc_feat is not None
                    else _estimate_mean_seg_dur(bars_audio, sr, bar_counts)
                    if bars_audio is not None
                    else 0.35
                )
                mean_pur = sum(bar_purities) / 4.0
                vrole = classify_vocal_role(section, rms_db, mean_pur, bar_counts, schema, mean_seg)
                vowel = (
                    classify_vowel_from_centroid(sum(centroids) / len(centroids))
                    if centroids
                    else (proc_feat["vowel"] if proc_feat else "AH")
                )
                entry["vr"] = vrole["vocal_role"]
                entry["hook"] = bool(vrole["is_chorus_hook"])
                entry["hb"] = int(vrole["hook_bars"])
                entry["cb"] = int(vrole["best_chop_bar"])
                entry["vow"] = vowel

            # Record chronological bars for whole-kit submodular essential-slice reconstitution
            for b in range(4):
                kit_bars_by_stem[stem].append({
                    "bar_in_phrase": b,
                    "section": section,
                    "purity": float(bar_purities[b]),
                    "mask": int(bar_masks[b]),
                    "root": int(bar_roots[b]),
                    "qual": str(bar_qualities[b]),
                    "chroma": bar_chromas[b],
                    "hr": hr[b],
                    "tr": tr[b],
                    "hc": hc[b],
                    "tc": tc[b],
                    "sm": sm_arr[b],
                    "is_fill": bool(stem == "drums" and entry.get("fill") and b == 3),
                    "is_hook": bool(stem == "vocals" and entry.get("hook") and bar_purities[b] >= 0.38),
                })

            phrase_out[stem] = entry
            if stem != "drums":
                phrase_chroma[stem] = [
                    [round(float(x), 4) for x in bar_chromas[b]] for b in range(4)
                ]
            mean_pur = sum(bar_purities) / 4.0
            if mean_pur >= 0.25:
                section_schema_samples.append((section, stem, schema, mean_pur))

        kit_results[pid] = phrase_out
        phrase_chroma_out[pid] = phrase_chroma

        # Record full-bar tonal progressions for style progression mining
        if all(v[0] >= 0 for v in bar_chord_votes):
            rel_prog = [((v[0] - key_pc) % 12, v[1]) for v in bar_chord_votes]
            avg_w = sum(v[2] for v in bar_chord_votes) / 4.0
            chord_seqs.append((rel_prog, avg_w))

    # Run whole-kit essential slice selection so every phrase/bar gets `es`, `er`, `fid`
    essential_info = select_essential_slices_for_song(kit_bars_by_stem, max_codebook_per_stem=8)
    for p_idx, phrase in enumerate(sorted_phrases):
        pid = phrase["id"]
        b0 = p_idx * 4
        b1 = b0 + 4
        for stem in STEMS:
            st_ess = essential_info["stems"][stem]
            entry = kit_results[pid][stem]
            entry["es"] = st_ess["essential_scores"][b0:b1]
            entry["er"] = st_ess["roles"][b0:b1]
            entry["fid"] = round(float(st_ess["fidelity"]), 3)

    return {
        "style": style,
        "mode": key_mode,
        "phrases": kit_results,
        "chroma": phrase_chroma_out,
        "schema_samples": section_schema_samples,
        "chord_seqs": chord_seqs,
        "mean_fidelity": float(essential_info["mean_fidelity"]),
        "role_transitions": essential_info["role_transitions"],
    }


def _mine_proc_grammar_worker(pfile_str: str) -> Optional[Tuple[bool, float, Dict[str, Dict[str, int]]]]:
    try:
        with open(pfile_str, "r") as f:
            td = json.load(f)
        mined = mine_processed_track_essential_bars(td, max_bars=64)
        ess = mined.get("essential") or {}
        mf = float(ess.get("mean_fidelity", 0.0))
        rt = ess.get("role_transitions") or {}
        return bool(mined.get("is_4stem")), mf, rt
    except Exception:
        return None


def _aggregate_progressions(
    raw_by_style_mode: Dict[Tuple[str, str], List[Tuple[List[Tuple[int, str]], float]]]
) -> Dict[str, Dict[str, List[List[Any]]]]:
    out: Dict[str, Dict[str, List[List[Any]]]] = {}
    for style, modes in STYLE_PROGRESSIONS.items():
        out[style] = {}
        for mode, prior_progs in modes.items():
            counts: Counter[Tuple[Tuple[int, str], ...]] = Counter()
            for seq, w in raw_by_style_mode.get((style, mode), []):
                counts[tuple(seq)] += w
            merged: List[List[List[Any]]] = []
            seen: set[Tuple[Tuple[int, str], ...]] = set()
            # Keep curated theory progressions first, then add top empirical mined progressions
            for p in prior_progs:
                tp = tuple(p)
                if tp not in seen:
                    seen.add(tp)
                    merged.append([[int(r), str(q)] for r, q in p])
            for seq_t, _ in counts.most_common(6):
                if seq_t not in seen:
                    seen.add(seq_t)
                    merged.append([[int(r), str(q)] for r, q in seq_t])
                    if len(merged) >= 8:
                        break
            out[style][mode] = merged
    return out


def _aggregate_motif_priors(
    samples: List[Tuple[str, str, str, float]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    buckets: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    for sec, stem, schema, w in samples:
        buckets[(sec, stem)][schema] += w

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (sec, stem), cnt in buckets.items():
        tot = sum(cnt.values()) or 1.0
        out.setdefault(sec, {})[stem] = {
            k: round(float(v) / tot, 3) for k, v in sorted(cnt.items())
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Wub multi-scale cell index")
    parser.add_argument("--kits-dir", type=Path, default=Path("frontend/public/data/kits"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-chroma", action="store_true",
                        help="also write cell_chroma.json (per-bar chroma for the quartet harmony check)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    args = parser.parse_args()

    t0 = time.time()
    kits_dir = args.kits_dir.resolve()
    manifest_path = kits_dir / "kit_manifest.json"
    out_path = (args.out or (kits_dir / "cell_index.json")).resolve()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load mix_compat_onsets if present
    onsets_arr: Optional[np.ndarray] = None
    onset_index: Dict[str, int] = {}
    onsets_meta_path = kits_dir / "mix_compat_onsets.json"
    onsets_bin_path = kits_dir / "mix_compat_onsets.f32"
    if onsets_meta_path.exists() and onsets_bin_path.exists():
        with open(onsets_meta_path, "r") as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        onsets_arr = np.fromfile(onsets_bin_path, dtype=np.float32).reshape(shape)
        onset_index = {str(k): int(v) for k, v in (meta.get("index") or {}).items()}
        print(f"Loaded mix_compat_onsets shape={shape}, mapped={len(onset_index)}")

    tasks: List[Dict[str, Any]] = []
    for kit in manifest.get("kits", []):
        tid = kit["track_id"]
        proc_file = args.processed_dir / f"{tid}.json"
        kit_onsets: Dict[str, np.ndarray] = {}
        if onsets_arr is not None:
            for p in kit.get("phrases", []):
                pid = p["id"]
                idx = onset_index.get(pid)
                if idx is not None and 0 <= idx < onsets_arr.shape[0]:
                    kit_onsets[pid] = onsets_arr[idx].copy()
        tasks.append(
            {
                "kit": kit,
                "proc_path": str(proc_file) if proc_file.exists() else None,
                "onset_rows": kit_onsets if kit_onsets else None,
            }
        )

    phrases_out: Dict[str, Dict[str, Any]] = {}
    chroma_out: Dict[str, Dict[str, List[List[float]]]] = {}
    all_schema_samples: List[Tuple[str, str, str, float]] = []
    raw_progs: Dict[Tuple[str, str], List[Tuple[List[Tuple[int, str]], float]]] = defaultdict(list)
    kit_fidelities: List[float] = []
    role_trans_counts: Dict[str, Dict[str, float]] = {
        r: {r2: 0.0 for r2 in ESSENTIAL_ROLES} for r in ESSENTIAL_ROLES
    }

    workers = max(1, int(args.workers))
    print(f"Indexing {len(tasks)} kits across {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(kits_dir),)) as pool:
        futures = [pool.submit(_process_kit, t) for t in tasks]
        done_cnt = 0
        for fut in as_completed(futures):
            res = fut.result()
            phrases_out.update(res["phrases"])
            if args.emit_chroma:
                chroma_out.update(res.get("chroma") or {})
            all_schema_samples.extend(res["schema_samples"])
            raw_progs[(res["style"], res["mode"])].extend(res["chord_seqs"])
            kit_fidelities.append(float(res.get("mean_fidelity", 0.0)))
            rt = res.get("role_transitions") or {}
            for r1 in ESSENTIAL_ROLES:
                for r2 in ESSENTIAL_ROLES:
                    role_trans_counts[r1][r2] += float((rt.get(r1) or {}).get(r2, 0.0))
            done_cnt += 1
            if done_cnt % 60 == 0 or done_cnt == len(tasks):
                print(f"  [{done_cnt}/{len(tasks)}] phrases indexed={len(phrases_out)} ({time.time()-t0:.1f}s)")

    # Also mine processed corpus tracks in parallel to enrich the compressed role grammar if present
    proc_fidelities: List[float] = []
    if args.processed_dir.exists():
        proc_files = sorted(args.processed_dir.glob("*.json"))
        if proc_files:
            print(f"Mining essential slice grammar across {len(proc_files)} processed tracks ({workers} workers)...")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_mine_proc_grammar_worker, str(pf)) for pf in proc_files]
                p_done = 0
                for fut in as_completed(futs):
                    p_done += 1
                    res_p = fut.result()
                    if res_p is not None:
                        is_4s, mf, rt = res_p
                        if mf > 0.0:
                            proc_fidelities.append(mf)
                        w = 1.5 if is_4s else 0.75
                        for r1 in ESSENTIAL_ROLES:
                            row_rt = rt.get(r1) or {}
                            for r2 in ESSENTIAL_ROLES:
                                role_trans_counts[r1][r2] += w * float(row_rt.get(r2, 0.0))
                    if p_done % 400 == 0 or p_done == len(proc_files):
                        print(f"  processed grammar [{p_done}/{len(proc_files)}] ({time.time()-t0:.1f}s)")

    role_trans_probs: Dict[str, Dict[str, float]] = {}
    for r1 in ESSENTIAL_ROLES:
        tot_r1 = sum(role_trans_counts[r1].values())
        if tot_r1 > 0.0:
            role_trans_probs[r1] = {
                r2: round(role_trans_counts[r1][r2] / tot_r1, 4) for r2 in ESSENTIAL_ROLES
            }
        else:
            role_trans_probs[r1] = {r2: round(1.0 / len(ESSENTIAL_ROLES), 4) for r2 in ESSENTIAL_ROLES}

    mean_kit_fid = sum(kit_fidelities) / len(kit_fidelities) if kit_fidelities else 0.0
    mean_proc_fid = sum(proc_fidelities) / len(proc_fidelities) if proc_fidelities else mean_kit_fid

    seam_vals: List[float] = []
    for p_map in phrases_out.values():
        for st_entry in p_map.values():
            sm_list = st_entry.get("sm")
            if sm_list:
                seam_vals.extend(float(x) for x in sm_list[:3])
    mean_seam_cont = (sum(seam_vals) / len(seam_vals)) if seam_vals else 0.0

    payload = {
        "version": 2,
        "format": "cell_index_v2",
        "count": len(phrases_out),
        "stem_names": list(STEMS),
        "mean_reconstitution_fidelity": round(mean_kit_fid, 4),
        "corpus_reconstitution_fidelity": round(mean_proc_fid, 4),
        "mean_seam_continuity": round(mean_seam_cont, 4),
        "essential_role_transitions": role_trans_probs,
        "style_progressions": _aggregate_progressions(raw_progs),
        "section_motif_priors": _aggregate_motif_priors(all_schema_samples),
        "phrases": phrases_out,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    if args.emit_chroma:
        chroma_path = kits_dir / "cell_chroma.json"
        with open(chroma_path, "w") as f:
            json.dump({
                "format": "cell_chroma_v1", "version": 1, "phrases": chroma_out,
            }, f, separators=(",", ":"))
        print(f"Wrote per-bar chroma for {len(chroma_out)} phrases to {chroma_path}")

    sidecars_path = kits_dir / "sidecars.json"
    if sidecars_path.exists():
        with open(sidecars_path, "r") as f:
            sc = json.load(f)
        sc["cell_index"] = True
        with open(sidecars_path, "w") as f:
            json.dump(sc, f, indent=2)
            f.write("\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({len(phrases_out)} phrases, {size_mb:.2f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
