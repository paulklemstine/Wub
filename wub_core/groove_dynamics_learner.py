"""Groove Dynamics Learner for Wub KG-505.

Analyzes corpus audio stems and processed metadata across 4 stylistic territories
('wave', 'cyber', 'house', 'breaks') to extract and serialize:
1. Micro-Timing & Swing Deviation Profiles: segment onset offsets Delta t(step, style)
   against quantized 16th-note grids (step 0..15), jitter, velocity dynamics, and MPC swing ratios.
2. Inter-Stem Call-and-Response Coupling: 4-stem co-occurrence matrices (drums, bass,
   other/synths, vocals) revealing shared rhythmic space, vocal pauses triggering synth stabs,
   and kick-to-bass syncopation.
3. Acoustic Sidechain Ducking Profiles: kick-to-bass gain attenuation depth (dB) and
   recovery release envelopes (ms) across genres.

Learned matrices are serialized to `frontend/public/data/groove_dynamics.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt
import soundfile as sf

logger = logging.getLogger("groove_dynamics_learner")

STYLES = ["wave", "cyber", "house", "breaks"]
STEMS = ["drums", "bass", "other", "vocals"]

STYLE_METADATA = {
    "wave": {
        "keywords": [
            "ambyion", "backwhen", "bitwvlf", "chvrn", "cafe royale", "chill children",
            "watermane", "sirens lullaby", "butterfly", "underwater", "timestretch"
        ],
    },
    "cyber": {
        "keywords": [
            "boris brejcha", "ann clue", "cushy", "culttastic", "apoptygma", "and one",
            "wumpscut", "zatox", "fear", "acid attack", "game over", "strap", "chief",
            "bmf", "face off", "immortality", "rock that shit"
        ],
    },
    "house": {
        "keywords": [
            "booka shade", "amtrac", "davi", "avoure", "bob moses", "kasablanca",
            "blaudiss", "body language", "definition", "gorgon city", "in deep"
        ],
    },
    "breaks": {
        "keywords": [
            "alvaro", "aaren san", "royksopp", "annie", "aero chord", "bassnectar",
            "white town", "zeds dead", "zomboy", "beatdown", "chasing heaven", "take you down"
        ],
    },
}


def classify_track_style(filename: str, bpm: float) -> str:
    """Classify track into one of 4 stylistic territories using title keywords and tempo."""
    fname_lower = filename.lower()
    for style, meta in STYLE_METADATA.items():
        if any(kw in fname_lower for kw in meta["keywords"]):
            return style

    # Acoustic / BPM fallbacks
    if bpm >= 135.0:
        return "cyber"
    elif 118.0 <= bpm <= 124.0:
        return "house"
    elif bpm < 118.0:
        return "wave"
    else:
        return "breaks"


class GrooveDynamicsLearner:
    """Learns micro-timing swing, inter-stem coupling, and sidechain ducking dynamics."""

    def __init__(
        self,
        processed_dir: Union[str, Path] = "data/processed",
        stems_dir: Union[str, Path] = "data/stems",
        sr: int = 44100,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.stems_dir = Path(stems_dir)
        self.sr = sr

    # =========================================================================
    # 1. Micro-Timing & Swing Deviation Profiles
    # =========================================================================

    def learn_microtiming(
        self,
        tracks_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract micro-timing deviations and velocity dynamics against quantized 16th-note grids."""
        json_files = sorted(list(self.processed_dir.glob("*.json")))
        if tracks_limit is not None and tracks_limit > 0:
            json_files = json_files[:tracks_limit]

        # Containers per style and per stem
        style_offsets: Dict[str, Dict[int, List[float]]] = {
            st: {s: [] for s in range(16)} for st in STYLES
        }
        style_velocities: Dict[str, Dict[int, List[float]]] = {
            st: {s: [] for s in range(16)} for st in STYLES
        }
        style_stem_offsets: Dict[str, Dict[str, Dict[int, List[float]]]] = {
            st: {stem: {s: [] for s in range(16)} for stem in STEMS} for st in STYLES
        }
        style_stem_velocities: Dict[str, Dict[str, Dict[int, List[float]]]] = {
            st: {stem: {s: [] for s in range(16)} for stem in STEMS} for st in STYLES
        }

        # Global aggregates
        global_offsets: Dict[int, List[float]] = {s: [] for s in range(16)}
        global_velocities: Dict[int, List[float]] = {s: [] for s in range(16)}

        tracks_processed = 0
        total_segments_analyzed = 0

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            bpm = float(meta.get("bpm", 126.0))
            if bpm <= 40.0 or bpm >= 240.0:
                bpm = 126.0

            style = classify_track_style(meta.get("filename", ""), bpm)
            if style not in STYLES:
                style = "house"

            t_step = 60.0 / bpm / 4.0
            t_bar = 16.0 * t_step

            segments = meta.get("segments", [])
            for seg in segments:
                if seg.get("is_silent"):
                    continue

                loud = float(seg.get("loudness_max", -99.0))
                if loud < -48.0:
                    continue

                start = float(seg.get("start", 0.0))
                stem = seg.get("stem", "other")
                if stem not in STEMS:
                    stem = "other"

                bar_idx = int(start // t_bar)
                t_in_bar = start - (bar_idx * t_bar)
                step = int(round(t_in_bar / t_step))
                quant = step * t_step
                if step == 16:
                    step = 0
                    quant = t_bar

                dev_sec = t_in_bar - quant
                # Limit deviation to half a step to avoid octave/step ambiguities
                if abs(dev_sec) > 0.5 * t_step:
                    continue

                offset_ms = float(dev_sec * 1000.0)
                norm_v = float(max(0.05, min(1.0, (loud + 48.0) / 48.0)))

                style_offsets[style][step].append(offset_ms)
                style_velocities[style][step].append(norm_v)
                style_stem_offsets[style][stem][step].append(offset_ms)
                style_stem_velocities[style][stem][step].append(norm_v)

                global_offsets[step].append(offset_ms)
                global_velocities[step].append(norm_v)
                total_segments_analyzed += 1

            tracks_processed += 1

        # Build output structure
        result_styles: Dict[str, Any] = {}
        for st in STYLES:
            mean_offsets = [
                float(np.mean(style_offsets[st][s])) if style_offsets[st][s] else 0.0
                for s in range(16)
            ]
            std_offsets = [
                float(np.std(style_offsets[st][s])) if style_offsets[st][s] else 0.0
                for s in range(16)
            ]
            mean_vels = [
                float(np.mean(style_velocities[st][s])) if style_velocities[st][s] else 0.8
                for s in range(16)
            ]

            even_offs = [mean_offsets[s] for s in [0, 2, 4, 6, 8, 10, 12, 14]]
            odd_offs = [mean_offsets[s] for s in [1, 3, 5, 7, 9, 11, 13, 15]]
            swing_delay_ms = float(np.mean(odd_offs) - np.mean(even_offs))

            # MPC Swing ratio calculation (50% = 0.50 straight, 54% = 0.54 swing)
            # Reference nominal step at 126 BPM is ~119.05 ms (quarter note is ~476.2 ms)
            ref_step_ms = (60.0 / 126.0 / 4.0) * 1000.0
            swing_ratio = float(max(0.45, min(0.70, 0.50 + (swing_delay_ms / (2.0 * ref_step_ms)))))

            # Per-stem breakdown
            stem_breakdown: Dict[str, Any] = {}
            for stem in STEMS:
                s_mean = [
                    float(np.mean(style_stem_offsets[st][stem][s]))
                    if style_stem_offsets[st][stem][s] else mean_offsets[s]
                    for s in range(16)
                ]
                s_jitter = [
                    float(np.std(style_stem_offsets[st][stem][s]))
                    if style_stem_offsets[st][stem][s] else std_offsets[s]
                    for s in range(16)
                ]
                s_vel = [
                    float(np.mean(style_stem_velocities[st][stem][s]))
                    if style_stem_velocities[st][stem][s] else mean_vels[s]
                    for s in range(16)
                ]
                stem_breakdown[stem] = {
                    "step_offsets_ms": [round(x, 2) for x in s_mean],
                    "step_jitter_ms": [round(x, 2) for x in s_jitter],
                    "step_velocity": [round(x, 3) for x in s_vel],
                }

            result_styles[st] = {
                "step_offsets_ms": [round(x, 2) for x in mean_offsets],
                "step_jitter_ms": [round(x, 2) for x in std_offsets],
                "step_velocity": [round(x, 3) for x in mean_vels],
                "swing_ratio": round(swing_ratio, 4),
                "swing_offset_ms": round(swing_delay_ms, 2),
                "stems": stem_breakdown,
            }

        global_mean_offsets = [
            float(np.mean(global_offsets[s])) if global_offsets[s] else 0.0
            for s in range(16)
        ]
        global_std_offsets = [
            float(np.std(global_offsets[s])) if global_offsets[s] else 0.0
            for s in range(16)
        ]
        global_mean_vels = [
            float(np.mean(global_velocities[s])) if global_velocities[s] else 0.8
            for s in range(16)
        ]

        return {
            "styles": result_styles,
            "global": {
                "step_offsets_ms": [round(x, 2) for x in global_mean_offsets],
                "step_jitter_ms": [round(x, 2) for x in global_std_offsets],
                "step_velocity": [round(x, 3) for x in global_mean_vels],
            },
            "tracks_analyzed": tracks_processed,
            "segments_analyzed": total_segments_analyzed,
        }

    # =========================================================================
    # 2. Inter-Stem Call-and-Response Coupling
    # =========================================================================

    def learn_inter_stem_coupling(
        self,
        tracks_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract 4-stem co-occurrence, call-and-response coupling, and step occupancy grids."""
        json_files = sorted(list(self.processed_dir.glob("*.json")))
        thresholds = {"drums": -22.0, "bass": -22.0, "other": -26.0, "vocals": -34.0}

        def make_accumulator() -> Dict[str, Any]:
            return {
                "co_occur": np.zeros((4, 4), dtype=np.float64),
                "call_resp": np.zeros((4, 4), dtype=np.float64),
                "active_counts": np.zeros(4, dtype=np.float64),
                "silent_counts": np.zeros(4, dtype=np.float64),
                "step_occupancy": np.zeros((16, 4), dtype=np.float64),
                "step_onsets": np.zeros((16, 4), dtype=np.float64),
                "total_tatums": 0,
                "total_bars": 0,
                "vocal_pause_synth_hits": 0,
                "vocal_pause_total": 0,
                "vocal_active_synth_hits": 0,
                "vocal_active_total": 0,
                "kick_step_bass_hits": 0,
                "kick_step_total": 0,
                "offbeat_bass_hits": 0,
                "offbeat_total": 0,
                "activity_series": [[] for _ in range(4)],
            }

        acc_by_style: Dict[str, Dict[str, Any]] = {st: make_accumulator() for st in STYLES}
        acc_global = make_accumulator()

        matched_tracks = 0

        for jf in json_files:
            if tracks_limit is not None and matched_tracks >= tracks_limit:
                break

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            available = set(meta.get("stems_available", []))
            if not set(STEMS).issubset(available):
                # Need separated stems for authentic 4-stem coupling
                continue

            bpm = float(meta.get("bpm", 126.0))
            style = classify_track_style(meta.get("filename", ""), bpm)
            if style not in STYLES:
                style = "house"

            t_step = 60.0 / bpm / 4.0
            t_bar = 16.0 * t_step
            duration = float(meta.get("duration", 0.0))
            bars_in_track = int(duration // t_bar) if t_bar > 0 else 0

            acc_s = acc_by_style[style]
            acc_s["total_bars"] += bars_in_track
            acc_global["total_bars"] += bars_in_track

            # Pre-filter active segment ids
            seg_active: Dict[str, bool] = {}
            for s in meta.get("segments", []):
                st_name = s.get("stem")
                if (
                    not s.get("is_silent")
                    and float(s.get("loudness_max", -99.0)) >= thresholds.get(st_name, -30.0)
                ):
                    seg_active[s.get("id", "")] = True
                    # Record onset density
                    start = float(s.get("start", 0.0))
                    bar_idx = int(start // t_bar) if t_bar > 0 else 0
                    t_in_bar = start - (bar_idx * t_bar)
                    step_idx = int(round(t_in_bar / t_step)) % 16
                    if st_name in STEMS:
                        stem_idx = STEMS.index(st_name)
                        acc_s["step_onsets"][step_idx, stem_idx] += 1
                        acc_global["step_onsets"][step_idx, stem_idx] += 1

            tatums = meta.get("tatums", [])
            for tat in tatums:
                step = int(tat.get("index", 0)) % 16
                is_act = [False] * 4

                for i, st_name in enumerate(STEMS):
                    seg_ids = tat.get("stem_segments", {}).get(st_name, [])
                    if any(seg_active.get(sid, False) for sid in seg_ids):
                        is_act[i] = True
                        acc_s["active_counts"][i] += 1
                        acc_global["active_counts"][i] += 1
                        acc_s["step_occupancy"][step, i] += 1
                        acc_global["step_occupancy"][step, i] += 1
                    else:
                        acc_s["silent_counts"][i] += 1
                        acc_global["silent_counts"][i] += 1

                    acc_s["activity_series"][i].append(1 if is_act[i] else 0)
                    acc_global["activity_series"][i].append(1 if is_act[i] else 0)

                acc_s["total_tatums"] += 1
                acc_global["total_tatums"] += 1

                # Co-occurrence & Call-and-Response pairs
                for i in range(4):
                    for j in range(4):
                        if is_act[i] and is_act[j]:
                            acc_s["co_occur"][i, j] += 1
                            acc_global["co_occur"][i, j] += 1
                        if (not is_act[i]) and is_act[j]:
                            acc_s["call_resp"][i, j] += 1
                            acc_global["call_resp"][i, j] += 1

                # Vocal pause vs synth stabs metric
                # index 2 is 'other', index 3 is 'vocals'
                if not is_act[3]:  # vocal pause
                    acc_s["vocal_pause_total"] += 1
                    acc_global["vocal_pause_total"] += 1
                    if is_act[2]:  # synth hit
                        acc_s["vocal_pause_synth_hits"] += 1
                        acc_global["vocal_pause_synth_hits"] += 1
                else:  # vocal active
                    acc_s["vocal_active_total"] += 1
                    acc_global["vocal_active_total"] += 1
                    if is_act[2]:
                        acc_s["vocal_active_synth_hits"] += 1
                        acc_global["vocal_active_synth_hits"] += 1

                # Kick syncopation vs bass metric
                # index 0 is 'drums', index 1 is 'bass'
                # Downbeat steps (0, 4, 8, 12) vs offbeats
                if step in (0, 4, 8, 12):
                    acc_s["kick_step_total"] += 1
                    acc_global["kick_step_total"] += 1
                    if is_act[1]:
                        acc_s["kick_step_bass_hits"] += 1
                        acc_global["kick_step_bass_hits"] += 1
                else:
                    acc_s["offbeat_total"] += 1
                    acc_global["offbeat_total"] += 1
                    if is_act[1]:
                        acc_s["offbeat_bass_hits"] += 1
                        acc_global["offbeat_bass_hits"] += 1

            matched_tracks += 1

        def format_accumulator(acc: Dict[str, Any]) -> Dict[str, Any]:
            n_tat = max(1.0, float(acc["total_tatums"]))
            n_bars = max(1.0, float(acc["total_bars"]))

            # Conditional Co-occurrence P(j | i active)
            co_mat = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                denom = max(1.0, acc["active_counts"][i])
                co_mat[i] = acc["co_occur"][i] / denom

            # Conditional Call-and-Response P(j | i silent)
            cr_mat = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                denom = max(1.0, acc["silent_counts"][i])
                cr_mat[i] = acc["call_resp"][i] / denom

            # Coupling Index: P(j | i silent) - P(j | i active)
            c_idx_mat = cr_mat - co_mat

            # Correlation matrix across activity series
            corr_mat = np.eye(4, dtype=np.float64)
            for i in range(4):
                s_i = np.array(acc["activity_series"][i], dtype=np.float64)
                if len(s_i) > 1 and np.std(s_i) > 1e-4:
                    for j in range(i + 1, 4):
                        s_j = np.array(acc["activity_series"][j], dtype=np.float64)
                        if len(s_j) > 1 and np.std(s_j) > 1e-4:
                            r = float(np.corrcoef(s_i, s_j)[0, 1])
                            corr_mat[i, j] = r
                            corr_mat[j, i] = r

            # Step occupancy (fraction of time stem is active on step 0..15)
            step_occ = np.zeros((16, 4), dtype=np.float64)
            steps_per_bar = n_tat / 16.0
            if steps_per_bar > 0:
                step_occ = acc["step_occupancy"] / steps_per_bar

            # Step onset density (onsets per bar on step 0..15)
            step_onsets_per_bar = acc["step_onsets"] / n_bars

            # Vocal-pause synth lift ratio (Laplace-smoothed ratio)
            p_synth_pause = (
                acc["vocal_pause_synth_hits"] / max(1.0, float(acc["vocal_pause_total"]))
            )
            p_synth_active = (
                acc["vocal_active_synth_hits"] / max(1.0, float(acc["vocal_active_total"]))
            )
            synth_lift = (p_synth_pause + 0.05) / (p_synth_active + 0.05)

            # Kick syncopation ratio: offbeat bass rate / downbeat bass rate (smoothed)
            p_bass_downbeat = acc["kick_step_bass_hits"] / max(1.0, float(acc["kick_step_total"]))
            p_bass_offbeat = acc["offbeat_bass_hits"] / max(1.0, float(acc["offbeat_total"]))
            kick_sync_ratio = (p_bass_offbeat + 0.05) / (p_bass_downbeat + 0.05)

            return {
                "co_occurrence_matrix": [[round(float(v), 3) for v in row] for row in co_mat],
                "call_and_response_matrix": [[round(float(v), 3) for v in row] for row in cr_mat],
                "coupling_index_matrix": [[round(float(v), 3) for v in row] for row in c_idx_mat],
                "correlation_matrix": [[round(float(v), 3) for v in row] for row in corr_mat],
                "step_occupancy": [
                    [round(float(step_occ[s, stem_i]), 3) for stem_i in range(4)]
                    for s in range(16)
                ],
                "step_onset_density": [
                    [round(float(step_onsets_per_bar[s, stem_i]), 3) for stem_i in range(4)]
                    for s in range(16)
                ],
                "vocal_pause_synth_lift": round(float(synth_lift), 3),
                "kick_bass_syncopation_ratio": round(float(kick_sync_ratio), 3),
            }

        result_styles = {st: format_accumulator(acc_by_style[st]) for st in STYLES}
        result_global = format_accumulator(acc_global)

        return {
            "styles": result_styles,
            "global": result_global,
            "multi_stem_tracks_analyzed": matched_tracks,
        }

    # =========================================================================
    # 3. Acoustic Sidechain Ducking Profiles
    # =========================================================================

    def learn_sidechain_ducking(
        self,
        tracks_limit: Optional[int] = None,
        duration_per_track: float = 40.0,
    ) -> Dict[str, Any]:
        """Measure kick-to-bass gain attenuation depth and recovery release envelopes from WAV stems."""
        stem_dirs = [
            p for p in sorted(list(self.stems_dir.glob("*")))
            if p.is_dir() and not p.name.startswith("_")
        ]
        if tracks_limit is not None and tracks_limit > 0:
            stem_dirs = stem_dirs[:tracks_limit]

        sos = butter(4, 120, btype="lowpass", fs=self.sr, output="sos")
        wb = int(0.035 * self.sr)  # 35ms pre-kick window
        wa = int(0.200 * self.sr)  # 200ms post-kick recovery window
        k_size = int(0.012 * self.sr)  # 12ms smoothing window
        boxcar = np.ones(k_size, dtype=np.float32) / float(k_size)

        style_ducking: Dict[str, Dict[str, List[Any]]] = {
            st: {"att": [], "rec": [], "rec90": [], "curves": []} for st in STYLES
        }
        global_ducking: Dict[str, List[Any]] = {
            "att": [], "rec": [], "rec90": [], "curves": []
        }

        tracks_measured = 0
        total_events = 0

        for sd in stem_dirs:
            dp = sd / "drums.wav"
            bp = sd / "bass.wav"
            if not (dp.exists() and bp.exists()):
                continue

            jpath = self.processed_dir / f"{sd.name}.json"
            bpm = 126.0
            filename = sd.name
            if jpath.exists():
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    bpm = float(meta.get("bpm", 126.0))
                    filename = meta.get("filename", sd.name)
                except Exception:
                    pass

            style = classify_track_style(filename, bpm)
            if style not in STYLES:
                style = "house"

            # Load audio up to duration_per_track
            max_samples = int(duration_per_track * self.sr)
            try:
                yd, sr_d = sf.read(str(dp), stop=max_samples, always_2d=False)
                yb, sr_b = sf.read(str(bp), stop=max_samples, always_2d=False)
            except Exception:
                continue

            if yd.ndim > 1:
                yd = yd.mean(axis=1)
            if yb.ndim > 1:
                yb = yb.mean(axis=1)

            # Resample if sample rate differs
            if sr_d != self.sr:
                step_idx = np.linspace(0, len(yd) - 1, int(len(yd) * self.sr / sr_d)).astype(int)
                yd = yd[step_idx]
            if sr_b != self.sr:
                step_idx_b = np.linspace(0, len(yb) - 1, int(len(yb) * self.sr / sr_b)).astype(int)
                yb = yb[step_idx_b]

            min_len = min(len(yd), len(yb))
            if min_len < self.sr * 2:
                continue

            yd = yd[:min_len].astype(np.float32)
            yb = yb[:min_len].astype(np.float32)

            # Low-pass filter drums to find kick transients
            sub = sosfilt(sos, yd)
            env_d = np.abs(sub)
            p90 = float(np.percentile(env_d, 90))
            thresh = p90 * 0.40
            if thresh < 1e-4:
                continue

            # Minimum distance between kicks (at least 200ms)
            min_dist = int(0.20 * self.sr)
            peaks, _ = find_peaks(env_d, distance=min_dist, height=thresh)

            # Smooth bass envelope
            smooth_b = np.convolve(np.abs(yb), boxcar, mode="same")

            for p in peaks:
                if p - wb < 0 or p + wa >= len(smooth_b):
                    continue

                pre_level = float(np.max(smooth_b[p - wb : p]))
                trough_off = int(np.argmin(smooth_b[p : p + int(0.045 * self.sr)]))
                p_tr = p + trough_off
                if p_tr + wa >= len(smooth_b):
                    continue

                post_min = float(smooth_b[p_tr])

                # Require active bass before kick and audible gain attenuation
                if pre_level > 0.012 and post_min < pre_level * 0.85:
                    att_db = float(20.0 * np.log10((post_min + 1e-4) / (pre_level + 1e-4)))
                    if -35.0 <= att_db <= -1.2:
                        style_ducking[style]["att"].append(att_db)
                        global_ducking["att"].append(att_db)

                        # Recovery time tau (63.2% recovery from trough)
                        targ_tau = post_min + 0.632 * (pre_level - post_min)
                        rec_idx = np.where(smooth_b[p_tr : p_tr + wa] >= targ_tau)[0]
                        if len(rec_idx) > 0:
                            tau_ms = float((rec_idx[0] / self.sr) * 1000.0)
                            style_ducking[style]["rec"].append(tau_ms)
                            global_ducking["rec"].append(tau_ms)

                        # 90% recovery
                        targ_90 = post_min + 0.90 * (pre_level - post_min)
                        rec90_idx = np.where(smooth_b[p_tr : p_tr + wa] >= targ_90)[0]
                        if len(rec90_idx) > 0:
                            t90_ms = float((rec90_idx[0] / self.sr) * 1000.0)
                            style_ducking[style]["rec90"].append(t90_ms)
                            global_ducking["rec90"].append(t90_ms)

                        # 16-point recovery envelope curve from 0 to 200ms
                        curve_idx = np.linspace(0, wa - 1, 16).astype(int)
                        curve = (smooth_b[p_tr + curve_idx] / (pre_level + 1e-4)).tolist()
                        style_ducking[style]["curves"].append(curve)
                        global_ducking["curves"].append(curve)

                        total_events += 1

            tracks_measured += 1

        def format_ducking_stats(bucket: Dict[str, List[Any]]) -> Dict[str, Any]:
            atts = bucket["att"]
            recs = bucket["rec"]
            rec90s = bucket["rec90"]
            curves = bucket["curves"]

            mean_att = float(np.mean(atts)) if atts else -6.0
            median_att = float(np.median(atts)) if atts else -6.0
            std_att = float(np.std(atts)) if atts else 2.0

            mean_rec = float(np.mean(recs)) if recs else 65.0
            median_rec = float(np.median(recs)) if recs else 50.0
            std_rec = float(np.std(recs)) if recs else 20.0

            mean_rec90 = float(np.mean(rec90s)) if rec90s else 80.0
            median_rec90 = float(np.median(rec90s)) if rec90s else 70.0

            if curves:
                mean_curve = [
                    round(float(x), 3) for x in np.mean(curves, axis=0)
                ]
            else:
                min_g = 10.0 ** (mean_att / 20.0)
                mean_curve = [
                    round(float(1.0 - (1.0 - min_g) * np.exp(-3.0 * t)), 3)
                    for t in np.linspace(0.0, 1.0, 16)
                ]

            return {
                "attenuation_depth_db": round(mean_att, 2),
                "median_attenuation_db": round(median_att, 2),
                "std_attenuation_db": round(std_att, 2),
                "recovery_time_ms": round(mean_rec, 1),
                "median_recovery_ms": round(median_rec, 1),
                "std_recovery_ms": round(std_rec, 1),
                "mean_recovery_90_ms": round(mean_rec90, 1),
                "median_recovery_90_ms": round(median_rec90, 1),
                "recovery_curve": mean_curve,
                "sample_count": len(atts),
            }

        result_styles = {st: format_ducking_stats(style_ducking[st]) for st in STYLES}
        result_global = format_ducking_stats(global_ducking)

        return {
            "styles": result_styles,
            "global": result_global,
            "tracks_measured": tracks_measured,
            "ducking_events_measured": total_events,
        }

    # =========================================================================
    # 4. Master Learning & Serialization
    # =========================================================================

    def fit(
        self,
        microtiming_limit: Optional[int] = None,
        coupling_limit: Optional[int] = None,
        sidechain_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit all 3 groove dynamics models across corpus and return master dictionary."""
        logger.info("Learning micro-timing swing deviation profiles...")
        microtiming = self.learn_microtiming(tracks_limit=microtiming_limit)

        logger.info("Learning inter-stem call-and-response coupling...")
        coupling = self.learn_inter_stem_coupling(tracks_limit=coupling_limit)

        logger.info("Learning acoustic sidechain ducking profiles...")
        sidechain = self.learn_sidechain_ducking(tracks_limit=sidechain_limit)

        return {
            "version": 1,
            "description": "Learned groove dynamics models across 4 stylistic territories (wave, cyber, house, breaks)",
            "styles": STYLES,
            "stems": STEMS,
            "microtiming": microtiming,
            "inter_stem_coupling": coupling,
            "sidechain_ducking": sidechain,
        }

    def save(
        self,
        model_data: Dict[str, Any],
        output_path: Union[str, Path] = "frontend/public/data/groove_dynamics.json",
    ) -> Path:
        """Serialize learned models to formatted JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)
        logger.info(f"Saved groove dynamics model to {out.resolve()}")
        return out

    @staticmethod
    def load(input_path: Union[str, Path] = "frontend/public/data/groove_dynamics.json") -> Dict[str, Any]:
        """Load learned groove dynamics model from JSON."""
        p = Path(input_path)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


# =============================================================================
# Helper Accessors for Sequencers & Pattern Assemblers
# =============================================================================

def get_style_sidechain_params(
    style: str,
    model_data: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Return (attenuation_depth_db, recovery_time_ms) for a style from model or fallbacks."""
    if model_data is not None and "sidechain_ducking" in model_data:
        st_data = model_data["sidechain_ducking"].get("styles", {}).get(style)
        if st_data:
            return float(st_data.get("attenuation_depth_db", -6.0)), float(st_data.get("recovery_time_ms", 65.0))

    defaults = {
        "wave": (-8.4, 25.0),
        "cyber": (-11.9, 68.0),
        "house": (-10.2, 38.0),
        "breaks": (-7.4, 15.0),
    }
    return defaults.get(style, (-6.0, 65.0))


def get_style_microtiming_offset(
    style: str,
    step: int,
    stem: Optional[str] = None,
    model_data: Optional[Dict[str, Any]] = None,
) -> float:
    """Return micro-timing offset in milliseconds for given style, step (0..15), and optional stem."""
    step = step % 16
    if model_data is not None and "microtiming" in model_data:
        st_data = model_data["microtiming"].get("styles", {}).get(style)
        if st_data:
            if stem and stem in st_data.get("stems", {}):
                stem_offsets = st_data["stems"][stem].get("step_offsets_ms", [])
                if len(stem_offsets) == 16:
                    return float(stem_offsets[step])
            step_offsets = st_data.get("step_offsets_ms", [])
            if len(step_offsets) == 16:
                return float(step_offsets[step])

    # Default microtiming swing offsets if model not loaded
    if step % 2 == 1:
        # Odd steps have positive swing lag
        return 4.0 if style in ("house", "wave") else 1.5
    return -1.5


def get_style_swing_ratio(
    style: str,
    model_data: Optional[Dict[str, Any]] = None,
) -> float:
    """Return MPC-style swing ratio (0.50 straight .. 0.66 heavy swing)."""
    if model_data is not None and "microtiming" in model_data:
        st_data = model_data["microtiming"].get("styles", {}).get(style)
        if st_data:
            return float(st_data.get("swing_ratio", 0.50))

    defaults = {"wave": 0.52, "cyber": 0.50, "house": 0.54, "breaks": 0.51}
    return defaults.get(style, 0.50)


# =============================================================================
# CLI Entrypoint
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Extract Groove Dynamics for Wub KG-505")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to data/processed")
    parser.add_argument("--stems-dir", default="data/stems", help="Path to data/stems")
    parser.add_argument(
        "--output",
        default="frontend/public/data/groove_dynamics.json",
        help="Path to output JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks analyzed")
    args = parser.parse_args()

    learner = GrooveDynamicsLearner(
        processed_dir=args.processed_dir,
        stems_dir=args.stems_dir,
    )
    print("🚀 Initiating Groove Dynamics extraction pipeline...")
    data = learner.fit(
        microtiming_limit=args.limit,
        coupling_limit=args.limit,
        sidechain_limit=args.limit,
    )
    out_path = learner.save(data, args.output)
    print(f"✅ Successfully extracted and saved groove dynamics to {out_path}!")
