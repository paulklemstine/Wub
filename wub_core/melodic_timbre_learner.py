"""Melodic & Timbre Feature Learner for Wub KG-505.

Analyzes the audio corpus (`data/processed/*.json` and `data/stems/*/*.wav`)
across 4 stylistic territories ('wave', 'cyber', 'house', 'breaks') to extract:
1. Filter Sweep & Energy Tension Ramps:
   - Spectral centroid trajectories and RMS across 4-bar and 8-bar sections
     (Buildups, Drops, Breakdowns).
   - Fits parametric Biquad cutoff frequency curves fc(t), resonance Q(t),
     and crescendo/tension ramps.
2. 3-to-5 Note Melodic Hook Motifs & Modal Scales:
   - N-gram chromatic pitch sequences relative to chord root (0..11).
   - Modal distribution projections (Dorian, Phrygian, Lydian, Aeolian) for
     synth leads and vocal melodies.
3. Acoustic Reverb Space Decay Profiles:
   - Schroeder backward integration RT60 estimation (ms).
   - Diffuse tail-to-direct wet/dry ratio estimation per style.

Outputs learned matrices to `frontend/public/data/melodic_timbre_models.json`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scripts.harvest_style_kits import STYLES, classify_track_style
except ImportError:
    STYLES = ["wave", "cyber", "house", "breaks"]

    STYLE_METADATA = {
        "wave": ["ambyion", "backwhen", "bitwvlf", "chvrn", "cafe royale", "chill children", "watermane", "sirens lullaby", "butterfly", "underwater", "timestretch"],
        "cyber": ["boris brejcha", "ann clue", "cushy", "culttastic", "apoptygma", "and one", "wumpscut", "zatox", "fear", "acid attack", "game over", "strap", "chief", "bmf", "face off", "immortality", "rock that shit"],
        "house": ["booka shade", "amtrac", "davi", "avoure", "bob moses", "kasablanca", "blaudiss", "body language", "definition", "gorgon city", "in deep"],
        "breaks": ["alvaro", "aaren san", "royksopp", "annie", "aero chord", "bassnectar", "white town", "zeds dead", "zomboy", "beatdown", "chasing heaven", "take you down"],
    }

    def classify_track_style(filename: str, bpm: float) -> str:
        fname_lower = filename.lower()
        for style, keywords in STYLE_METADATA.items():
            if any(kw in fname_lower for kw in keywords):
                return style
        if bpm >= 135.0:
            return "cyber"
        elif 118.0 <= bpm <= 124.0:
            return "house"
        elif bpm < 118.0:
            return "wave"
        else:
            return "breaks"

from wub_core.vocal_pattern_extractor import parse_key_root

logger = logging.getLogger(__name__)

# Modal Scale Templates (relative to root 0)
# Characteristic modal color tones receive enhanced weighting:
# - Dorian: natural 6th (9)
# - Phrygian: flat 2nd (1)
# - Lydian: sharp 4th (6)
# - Aeolian: flat 6th (8)
MODAL_TEMPLATES = {
    "dorian": np.array([1.0, 0.05, 0.7, 0.9, 0.05, 0.6, 0.05, 0.9, 0.05, 1.0, 0.8, 0.05], dtype=np.float32),
    "phrygian": np.array([1.0, 1.0, 0.05, 0.9, 0.05, 0.6, 0.05, 0.9, 0.8, 0.05, 0.8, 0.05], dtype=np.float32),
    "lydian": np.array([1.0, 0.05, 0.7, 0.05, 0.9, 0.05, 1.0, 0.9, 0.05, 0.8, 0.05, 0.8], dtype=np.float32),
    "aeolian": np.array([1.0, 0.05, 0.7, 0.9, 0.05, 0.6, 0.05, 0.9, 1.0, 0.05, 0.8, 0.05], dtype=np.float32),
}

SEMITONE_NAMES = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]

STYLE_ACOUSTIC_PROFILES = {
    "wave": {
        "space_name": "Ethereal Cathedral / Starlight Nebula",
        "description": "Lush, expansive diffuse space with high wet ratio and gentle high-frequency damping",
        "default_t60_ms": 2800.0,
        "default_wet_dry": 0.38,
        "default_predelay_ms": 32.0,
        "damping": 0.25,
        "room_size": 0.85,
    },
    "cyber": {
        "space_name": "Industrial Bunker / Steel Warehouse",
        "description": "Tight, gritty reflective space with aggressive early reflections and controlled tail",
        "default_t60_ms": 1450.0,
        "default_wet_dry": 0.26,
        "default_predelay_ms": 14.0,
        "damping": 0.65,
        "room_size": 0.52,
    },
    "house": {
        "space_name": "Velvet Club / Warm Wooden Dancefloor",
        "description": "Medium warm plate/hall with rich midrange bloom and balanced vocal prominence",
        "default_t60_ms": 1950.0,
        "default_wet_dry": 0.32,
        "default_predelay_ms": 22.0,
        "damping": 0.42,
        "room_size": 0.68,
    },
    "breaks": {
        "space_name": "Concrete Alley / Punchy Slapback Chamber",
        "description": "Short snappy decay preserving kick/snare punch with crisp stereo slapback",
        "default_t60_ms": 1350.0,
        "default_wet_dry": 0.22,
        "default_predelay_ms": 18.0,
        "damping": 0.55,
        "room_size": 0.48,
    },
}


def fit_power_curve(u: np.ndarray, y: np.ndarray, y_start: float, y_end: float) -> float:
    """Fit exponent alpha in y(u) = y_start + (y_end - y_start) * (u ** alpha) via least squares."""
    dy = y_end - y_start
    if abs(dy) < 1e-4 or len(u) < 2:
        return 1.0

    # Normalized target in [0, 1]
    norm_y = np.clip((y - y_start) / dy, 0.0, 1.0)
    # Mask out boundary endpoints 0 and 1 for log fitting
    mask = (u > 0.05) & (u < 0.95) & (norm_y > 0.01) & (norm_y < 0.99)
    if np.sum(mask) < 2:
        return 1.0

    log_u = np.log(u[mask])
    log_y = np.log(norm_y[mask])
    # log(norm_y) = alpha * log(u) -> alpha = sum(log_u * log_y) / sum(log_u^2)
    denom = np.sum(log_u**2)
    if denom <= 1e-6:
        return 1.0
    alpha = float(np.sum(log_u * log_y) / denom)
    return float(np.clip(alpha, 0.25, 4.0))


def fit_biquad_cutoff_curve(
    u_points: np.ndarray,
    fc_trajectory: np.ndarray,
    filter_type: str = "lowpass",
) -> Tuple[float, float, float, List[float]]:
    """Fit parametric cutoff curve fc(u) = f_start * (f_end / f_start) ^ (u ^ alpha).

    Returns:
        (f_start, f_end, alpha, trajectory_samples_16)
    """
    f_start = float(np.clip(fc_trajectory[0], 120.0, 20000.0))
    f_end = float(np.clip(fc_trajectory[-1], 120.0, 20000.0))

    if abs(f_end - f_start) < 20.0:
        alpha = 1.0
    else:
        log_start = np.log(f_start)
        log_end = np.log(f_end)
        log_fc = np.log(np.clip(fc_trajectory, 120.0, 20000.0))
        alpha = fit_power_curve(u_points, log_fc, log_start, log_end)

    # Sample at 16 equidistant points
    sample_u = np.linspace(0.0, 1.0, 16)
    sampled = [
        round(float(f_start * ((f_end / f_start) ** (float(u) ** alpha))), 2)
        for u in sample_u
    ]
    return f_start, f_end, round(alpha, 3), sampled


def fit_resonance_curve(
    u_points: np.ndarray,
    q_start: float,
    q_end: float,
    q_peak: float,
    u_peak: float = 0.88,
) -> Tuple[float, float, float, float, List[float]]:
    """Fit resonance curve Q(u) with peak right before section climax.

    Returns:
        (q_start, q_end, q_peak, u_peak, trajectory_samples_16)
    """
    sample_u = np.linspace(0.0, 1.0, 16)
    sampled = []
    for u in sample_u:
        if u <= u_peak:
            ratio = u / max(0.01, u_peak)
            q_val = q_start + (q_peak - q_start) * (ratio**1.8)
        else:
            ratio = (u - u_peak) / max(0.01, 1.0 - u_peak)
            q_val = q_peak - (q_peak - q_end) * (ratio**1.2)
        sampled.append(round(float(q_val), 3))
    return round(q_start, 2), round(q_end, 2), round(q_peak, 2), round(u_peak, 2), sampled


def fit_energy_crescendo_curve(
    u_points: np.ndarray,
    rms_trajectory: np.ndarray,
) -> Tuple[float, float, float, List[float]]:
    """Fit RMS tension crescendo curve RMS_dB(u) = L_start + (L_end - L_start) * u^gamma."""
    l_start = float(rms_trajectory[0])
    l_end = float(rms_trajectory[-1])
    gamma = fit_power_curve(u_points, rms_trajectory, l_start, l_end)

    sample_u = np.linspace(0.0, 1.0, 16)
    tension_curve = [round(float(u**gamma), 3) for u in sample_u]
    return round(l_start, 2), round(l_end, 2), round(gamma, 3), tension_curve


def classify_motif_affinity(ngram: Tuple[int, ...]) -> Tuple[str, str]:
    """Classify musical affinity and functional description of a relative pitch n-gram."""
    degrees = [SEMITONE_NAMES[p % 12] for p in ngram]
    deg_str = " - ".join(degrees)

    s = set(ngram)
    # Check defining intervals
    if 6 in s:
        return "Lydian Mystic Lift (#4)", deg_str
    elif 1 in s and 8 in s:
        return "Phrygian Tension Hook (b2, b6)", deg_str
    elif 1 in s:
        return "Phrygian Dark Step (b2)", deg_str
    elif 9 in s and 3 in s:
        return "Dorian Soulful 6th (b3, 6)", deg_str
    elif 8 in s and 3 in s:
        return "Aeolian Melancholy Hook (b3, b6)", deg_str
    elif s.issubset({0, 3, 7, 10}):
        return "Minor Pentatonic / Arp Hook", deg_str
    elif s.issubset({0, 4, 7, 11}):
        return "Major Seventh Shimmer", deg_str
    elif s.issubset({0, 2, 4, 7, 9}):
        return "Major Pentatonic Ascending", deg_str
    elif s.issubset({0, 5, 7}):
        return "Suspended Fourth Hook", deg_str
    elif len(ngram) >= 3 and ngram[0] == ngram[2] and ngram[0] != ngram[1]:
        return "Pedal Neighbor Bounce", deg_str
    else:
        return "Modal Linear Melodic Arc", deg_str


def estimate_t60_schroeder(signal: np.ndarray, sr: int) -> Optional[float]:
    """Estimate RT60 decay time (in seconds) via Schroeder backward integration."""
    if len(signal) < int(sr * 0.2):
        return None

    # Energy envelope
    squared = signal.astype(np.float64) ** 2
    # Reverse cumulative sum
    edc = np.cumsum(squared[::-1])[::-1]
    if edc[0] <= 1e-12:
        return None

    edc_db = 10.0 * np.log10(edc / (edc[0] + 1e-12) + 1e-12)

    # T20 evaluation (-5 dB to -25 dB)
    idx_5 = np.where(edc_db <= -5.0)[0]
    idx_25 = np.where(edc_db <= -25.0)[0]
    if len(idx_5) == 0 or len(idx_25) == 0:
        return None
    t5 = idx_5[0] / sr
    t25 = idx_25[0] / sr
    if t25 <= t5 + 0.01:
        return None

    # Extrapolate 20 dB drop to 60 dB
    t60 = 3.0 * (t25 - t5)
    return float(t60)


def estimate_wet_dry_ratio(signal: np.ndarray, sr: int) -> Optional[float]:
    """Estimate wet/dry diffuse reflection energy ratio from transient decay."""
    n_direct = int(0.045 * sr)  # First 45ms is direct sound
    n_tail = min(len(signal), int(0.75 * sr))  # 45ms to 750ms is reverberant tail
    if len(signal) < n_tail:
        return None

    squared = signal.astype(np.float64) ** 2
    e_direct = float(np.sum(squared[:n_direct]))
    e_tail = float(np.sum(squared[n_direct:n_tail]))

    if (e_direct + e_tail) <= 1e-12:
        return None

    # Calibration: mix wet ratio typically scales between 0.10 and 0.55
    raw_ratio = e_tail / (e_direct + e_tail)
    calibrated_wet = float(np.clip(raw_ratio * 0.45, 0.15, 0.65))
    return calibrated_wet


class MelodicTimbreLearner:
    """Extracts parametric filter sweeps, melodic hook motifs, modal distributions, and reverb profiles."""

    def __init__(
        self,
        processed_dir: Path | str = "data/processed",
        stems_dir: Path | str = "data/stems",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.stems_dir = Path(stems_dir)

    def analyze_filter_sweeps(self, max_tracks: Optional[int] = None) -> Dict[str, Any]:
        """Analyze spectral centroid and RMS trajectories across 4-bar and 8-bar sections."""
        logger.info("Analyzing filter sweep & energy tension ramps...")

        # Collectors for raw trajectories per style and section
        # sections: buildup_4bar, buildup_8bar, drop_4bar, drop_8bar, breakdown_4bar, breakdown_8bar
        raw_trajectories: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {
            s: {
                "buildup_4bar": [],
                "buildup_8bar": [],
                "drop_4bar": [],
                "drop_8bar": [],
                "breakdown_4bar": [],
                "breakdown_8bar": [],
            }
            for s in STYLES
        }

        json_files = sorted(list(self.processed_dir.glob("*.json")))
        if max_tracks is not None:
            json_files = json_files[:max_tracks]

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            bpm = float(d.get("bpm", 120.0))
            if bpm <= 0.0:
                bpm = 120.0
            bar_dur = 4.0 * (60.0 / bpm)
            duration = float(d.get("duration", 0.0))
            n_bars = int(duration // bar_dur)
            if n_bars < 12:
                continue

            style = classify_track_style(d.get("filename", ""), bpm)

            # Accumulate per-bar features
            bar_centroids: List[List[float]] = [[] for _ in range(n_bars)]
            bar_loudness: List[List[float]] = [[] for _ in range(n_bars)]
            bar_drum_loud: List[List[float]] = [[] for _ in range(n_bars)]

            for seg in d.get("segments", []):
                st = float(seg.get("start", 0.0))
                b = int(st // bar_dur)
                if 0 <= b < n_bars:
                    c = seg.get("spectral_centroid")
                    l = seg.get("loudness_max")
                    if c is not None:
                        bar_centroids[b].append(float(c))
                    if l is not None:
                        bar_loudness[b].append(float(l))
                    if seg.get("stem") == "drums" and l is not None:
                        bar_drum_loud[b].append(float(l))

            # Default baselines if missing
            c_mean = np.array([np.mean(x) if x else 2200.0 for x in bar_centroids], dtype=np.float32)
            l_mean = np.array([np.mean(x) if x else -35.0 for x in bar_loudness], dtype=np.float32)
            drm_mean = np.array([np.mean(x) if x else -50.0 for x in bar_drum_loud], dtype=np.float32)

            # Slide 8-bar and 4-bar windows
            for b in range(0, n_bars - 8, 4):
                c_win = c_mean[b : b + 8]
                l_win = l_mean[b : b + 8]
                drm_win = drm_mean[b : b + 8]

                c_slope = c_win[-1] - c_win[0]
                l_slope = l_win[-1] - l_win[0]
                avg_drm = float(np.mean(drm_win))
                avg_l = float(np.mean(l_win))

                # Buildup 8-bar: Rising centroid & loudness, culminating high
                if c_slope > 400.0 and l_slope > 3.0:
                    raw_trajectories[style]["buildup_8bar"].append((c_win, l_win))
                # Breakdown 8-bar: Minimal drums, quieter, warm space
                elif avg_drm < -24.0 and avg_l < -18.0:
                    raw_trajectories[style]["breakdown_8bar"].append((c_win, l_win))
                # Drop 8-bar: Driving drums and high energy
                elif avg_drm > -15.0 and avg_l > -14.0:
                    raw_trajectories[style]["drop_8bar"].append((c_win, l_win))

            for b in range(0, n_bars - 4, 2):
                c_win4 = c_mean[b : b + 4]
                l_win4 = l_mean[b : b + 4]
                drm_win4 = drm_mean[b : b + 4]

                c_slope = c_win4[-1] - c_win4[0]
                l_slope = l_win4[-1] - l_win4[0]
                avg_drm = float(np.mean(drm_win4))
                avg_l = float(np.mean(l_win4))

                if c_slope > 300.0 and l_slope > 2.5:
                    raw_trajectories[style]["buildup_4bar"].append((c_win4, l_win4))
                elif avg_drm < -24.0 and avg_l < -18.0:
                    raw_trajectories[style]["breakdown_4bar"].append((c_win4, l_win4))
                elif avg_drm > -15.0 and avg_l > -14.0:
                    raw_trajectories[style]["drop_4bar"].append((c_win4, l_win4))

        # Fit parametric models per style and section
        filter_models: Dict[str, Dict[str, Any]] = {}

        # Style baseline offsets
        style_q_peaking = {
            "wave": 2.2,
            "cyber": 5.4,
            "house": 3.0,
            "breaks": 4.2,
        }

        for style in STYLES:
            filter_models[style] = {}
            for sec_name, traj_list in raw_trajectories[style].items():
                is_8bar = "8bar" in sec_name
                is_buildup = "buildup" in sec_name
                is_drop = "drop" in sec_name
                is_breakdown = "breakdown" in sec_name

                n_steps = 8 if is_8bar else 4
                u_norm = np.linspace(0.0, 1.0, n_steps)

                if traj_list:
                    avg_c = np.mean([t[0] for t in traj_list], axis=0)
                    avg_l = np.mean([t[1] for t in traj_list], axis=0)
                else:
                    # Robust fallback defaults based on genre acoustics
                    if is_buildup:
                        avg_c = np.linspace(350.0, 16000.0, n_steps)
                        avg_l = np.linspace(-26.0, -4.0, n_steps)
                    elif is_drop:
                        avg_c = np.full(n_steps, 17500.0)
                        avg_l = np.full(n_steps, -4.5)
                    else:  # breakdown
                        avg_c = np.linspace(12000.0, 1800.0, n_steps)
                        avg_l = np.linspace(-12.0, -22.0, n_steps)

                # Map centroid to cutoff frequency range
                if is_buildup:
                    f_start = float(np.clip(avg_c[0] * 0.25, 200.0, 800.0))
                    f_end = float(np.clip(avg_c[-1] * 4.2, 12000.0, 19500.0))
                    fc_curve = np.linspace(f_start, f_end, n_steps)
                    f_st, f_en, alpha, fc_samples = fit_biquad_cutoff_curve(u_norm, fc_curve, "lowpass")
                    q_pk = style_q_peaking[style]
                    q_st, q_en, q_peak, u_peak, q_samples = fit_resonance_curve(
                        u_norm, q_start=1.0, q_end=1.1, q_peak=q_pk, u_peak=0.88
                    )
                elif is_drop:
                    f_st, f_en, alpha, fc_samples = 18000.0, 18000.0, 1.0, [18000.0] * 16
                    q_st, q_en, q_peak, u_peak, q_samples = 0.8, 0.8, 0.9, 0.5, [0.8] * 16
                else:  # breakdown
                    f_start = float(np.clip(avg_c[0] * 3.5, 9000.0, 15000.0))
                    f_end = float(np.clip(avg_c[-1] * 0.6, 900.0, 2400.0))
                    fc_curve = np.linspace(f_start, f_end, n_steps)
                    f_st, f_en, alpha, fc_samples = fit_biquad_cutoff_curve(u_norm, fc_curve, "lowpass")
                    q_st, q_en, q_peak, u_peak, q_samples = 1.2, 0.9, 1.4, 0.3, [1.1] * 16

                l_start, l_end, gamma, tension_curve = fit_energy_crescendo_curve(u_norm, avg_l)

                filter_models[style][sec_name] = {
                    "filter_type": "lowpass",
                    "cutoff": {
                        "f_start_hz": f_st,
                        "f_end_hz": f_en,
                        "curve": "exponential",
                        "exponent_alpha": alpha,
                        "trajectory_samples_16": fc_samples,
                    },
                    "resonance": {
                        "q_start": q_st,
                        "q_end": q_en,
                        "q_peak": q_peak,
                        "u_peak": u_peak,
                        "trajectory_samples_16": q_samples,
                    },
                    "energy_crescendo": {
                        "rms_start_db": l_start,
                        "rms_end_db": l_end,
                        "gamma": gamma,
                        "tension_curve_16": tension_curve,
                    },
                    "analyzed_instances": len(traj_list),
                }

        return filter_models

    def analyze_melodic_hook_motifs(self, max_tracks: Optional[int] = None) -> Dict[str, Any]:
        """Extract 3-to-5 note chromatic motifs relative to root and modal distributions."""
        logger.info("Extracting melodic hook motifs & modal scale distributions...")

        # Collectors
        pitch_histograms = {
            s: {"synth_lead": np.zeros(12, dtype=np.float64), "vocal_melody": np.zeros(12, dtype=np.float64)}
            for s in STYLES
        }

        motifs_by_style: Dict[str, Dict[str, Dict[int, Counter]]] = {
            s: {
                "synth_lead": {3: Counter(), 4: Counter(), 5: Counter()},
                "vocal_melody": {3: Counter(), 4: Counter(), 5: Counter()},
            }
            for s in STYLES
        }

        json_files = sorted(list(self.processed_dir.glob("*.json")))
        if max_tracks is not None:
            json_files = json_files[:max_tracks]

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            bpm = float(d.get("bpm", 120.0))
            style = classify_track_style(d.get("filename", ""), bpm)
            root = parse_key_root(d.get("key", "C major"))

            # Group segments by stem
            for stem, cat_key in [("other", "synth_lead"), ("vocals", "vocal_melody")]:
                segs = [
                    s
                    for s in d.get("segments", [])
                    if s.get("stem") == stem
                    and s.get("pitches")
                    and float(s.get("loudness_max", -99.0)) > -36.0
                    and float(s.get("duration", 0.0)) >= 0.06
                ]
                if not segs:
                    continue

                segs.sort(key=lambda x: float(x.get("start", 0.0)))

                # Build pitch contour sequence with rests / phrase breaks
                contour: List[Optional[int]] = []
                last_p: Optional[int] = None
                last_end = -1.0

                for seg in segs:
                    st = float(seg.get("start", 0.0))
                    dur = float(seg.get("duration", 0.1))
                    pitches = seg.get("pitches", [])
                    if len(pitches) != 12:
                        continue

                    p_abs = int(np.argmax(pitches))
                    p_rel = (p_abs - root) % 12

                    # Accumulate histogram
                    pitch_histograms[style][cat_key][p_rel] += 1.0

                    # Detect phrase boundary (> 2.5s gap)
                    if last_end >= 0.0 and (st - last_end) > 2.5:
                        contour.append(None)
                        last_p = None

                    # Melodic contour: record pitch change
                    if p_rel != last_p:
                        contour.append(p_rel)
                        last_p = p_rel

                    last_end = st + dur

                # Extract n-grams
                c_len = len(contour)
                for n in (3, 4, 5):
                    for i in range(c_len - n + 1):
                        window = contour[i : i + n]
                        if all(x is not None for x in window):
                            gram = tuple(int(x) for x in window)
                            motifs_by_style[style][cat_key][n][gram] += 1

        # Format output structure
        result: Dict[str, Dict[str, Any]] = {}

        for style in STYLES:
            result[style] = {}
            for stem_key in ("synth_lead", "vocal_melody"):
                hist = pitch_histograms[style][stem_key]
                tot_notes = float(hist.sum())
                if tot_notes > 0:
                    p_dist = hist / tot_notes
                else:
                    # fallback uniform
                    p_dist = np.full(12, 1.0 / 12.0)

                # Compute modal projection
                modal_sims = {}
                for mode_name, template in MODAL_TEMPLATES.items():
                    denom = float(np.linalg.norm(p_dist) * np.linalg.norm(template))
                    sim = float(np.dot(p_dist, template) / max(1e-8, denom))
                    modal_sims[mode_name] = sim

                # Softmax with temperature 0.08
                temp = 0.08
                exp_sims = {m: np.exp(s / temp) for m, s in modal_sims.items()}
                sum_exp = float(sum(exp_sims.values()))
                modal_dist = {m: round(float(v / sum_exp), 4) for m, v in exp_sims.items()}
                dominant_mode = max(modal_dist, key=modal_dist.get)

                # Format top motifs
                formatted_motifs = {}
                for n in (3, 4, 5):
                    c = motifs_by_style[style][stem_key][n]
                    tot_grams = max(1, sum(c.values()))
                    top_list = []
                    for gram, count in c.most_common(12):
                        affinity, degrees = classify_motif_affinity(gram)
                        top_list.append({
                            "ngram": list(gram),
                            "degree_names": degrees,
                            "affinity": affinity,
                            "count": count,
                            "probability": round(float(count / tot_grams), 4),
                        })
                    formatted_motifs[f"top_motifs_{n}note"] = top_list

                result[style][stem_key] = {
                    "modal_distribution": modal_dist,
                    "dominant_mode": dominant_mode,
                    "pitch_class_distribution": [round(float(x), 4) for x in p_dist],
                    "total_pitch_events": int(tot_notes),
                    **formatted_motifs,
                }

        return result

    def analyze_reverb_space_profiles(self, max_stems: Optional[int] = None) -> Dict[str, Any]:
        """Estimate acoustic decay profiles (T60 ms) and wet/dry ratios per style."""
        logger.info("Estimating acoustic reverb space decay profiles...")

        style_t60_values: Dict[str, List[float]] = {s: [] for s in STYLES}
        style_wet_values: Dict[str, List[float]] = {s: [] for s in STYLES}
        stem_specific_t60: Dict[str, Dict[str, List[float]]] = {
            s: {"vocals": [], "other": [], "drums": [], "bass": []} for s in STYLES
        }

        if sf is not None and self.stems_dir.exists():
            stem_dirs = sorted([p for p in self.stems_dir.iterdir() if p.is_dir()])
            if max_stems is not None:
                stem_dirs = stem_dirs[:max_stems]

            for sd in stem_dirs:
                # Resolve style
                jf = self.processed_dir / f"{sd.name}.json"
                bpm = 120.0
                fname = sd.name
                if jf.exists():
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            bpm = float(meta.get("bpm", 120.0))
                            fname = meta.get("filename", sd.name)
                    except Exception:
                        pass
                style = classify_track_style(fname, bpm)

                # Analyze stems
                for stem_name in ("vocals", "other", "drums", "bass"):
                    wav_path = sd / f"{stem_name}.wav"
                    if not wav_path.exists():
                        continue

                    try:
                        info = sf.info(wav_path)
                        sr = info.samplerate
                        dur_read = min(25.0, max(5.0, info.duration - 8.0))
                        data, _ = sf.read(wav_path, start=int(8.0 * sr), frames=int(dur_read * sr))
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)

                        # Transient peaks
                        hop = 256
                        rms = np.sqrt(np.convolve(data**2, np.ones(hop) / hop, mode="valid"))
                        thresh = float(np.mean(rms) * 2.2)
                        peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]) & (rms[1:-1] > thresh))[0]

                        for p_idx in peaks[:6]:
                            sample_idx = p_idx
                            seg = data[sample_idx : sample_idx + int(2.5 * sr)]
                            t60 = estimate_t60_schroeder(seg, sr)
                            if t60 is not None and 0.35 < t60 < 7.0:
                                t60_ms = t60 * 1000.0
                                style_t60_values[style].append(t60_ms)
                                stem_specific_t60[style][stem_name].append(t60_ms)

                            wet = estimate_wet_dry_ratio(seg, sr)
                            if wet is not None:
                                style_wet_values[style].append(wet)
                    except Exception:
                        continue

        # Build output profiles
        profiles: Dict[str, Any] = {}
        for style in STYLES:
            t60_list = style_t60_values[style]
            wet_list = style_wet_values[style]
            def_meta = STYLE_ACOUSTIC_PROFILES[style]

            t60_ms = float(np.median(t60_list)) if len(t60_list) >= 4 else def_meta["default_t60_ms"]
            wet_ratio = float(np.median(wet_list)) if len(wet_list) >= 4 else def_meta["default_wet_dry"]

            # Per-stem parameters
            stem_reverbs = {}
            for stem_name in ("vocals", "other", "drums", "bass"):
                st_list = stem_specific_t60[style][stem_name]
                st_t60 = float(np.median(st_list)) if len(st_list) >= 3 else t60_ms
                if stem_name == "vocals":
                    st_wet = round(min(0.55, wet_ratio * 1.15), 3)
                elif stem_name == "other":
                    st_wet = round(wet_ratio, 3)
                elif stem_name == "drums":
                    st_wet = round(max(0.12, wet_ratio * 0.55), 3)
                    st_t60 = min(st_t60, 1600.0)
                else:  # bass
                    st_wet = 0.05
                    st_t60 = min(st_t60, 800.0)

                stem_reverbs[stem_name] = {
                    "wet_ratio": st_wet,
                    "decay_ms": round(st_t60, 1),
                }

            profiles[style] = {
                "space_name": def_meta["space_name"],
                "description": def_meta["description"],
                "t60_decay_ms": round(t60_ms, 1),
                "t60_decay_s": round(t60_ms / 1000.0, 3),
                "wet_dry_ratio": round(wet_ratio, 3),
                "pre_delay_ms": def_meta["default_predelay_ms"],
                "high_frequency_damping": def_meta["damping"],
                "room_size": def_meta["room_size"],
                "measured_events": len(t60_list),
                "stem_reverbs": stem_reverbs,
            }

        return profiles

    def learn_and_export(
        self,
        output_path: Path | str = "frontend/public/data/melodic_timbre_models.json",
        max_tracks: Optional[int] = None,
        max_stems: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute extraction pipeline and write learned models to JSON."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Executing Melodic & Timbre learning pipeline...")
        filter_sweeps = self.analyze_filter_sweeps(max_tracks=max_tracks)
        melodic_motifs = self.analyze_melodic_hook_motifs(max_tracks=max_tracks)
        reverb_profiles = self.analyze_reverb_space_profiles(max_stems=max_stems)

        payload = {
            "version": "1.0.0",
            "generator": "wub_core.melodic_timbre_learner",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "styles": STYLES,
            "filter_sweep_models": filter_sweeps,
            "melodic_hook_motifs": melodic_motifs,
            "reverb_space_profiles": reverb_profiles,
            "metadata": {
                "processed_tracks_dir": str(self.processed_dir),
                "stems_dir": str(self.stems_dir),
            },
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Exported melodic & timbre models to %s", out_p)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn Melodic & Timbre Models across Corpus Styles.")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed JSON tracks.")
    parser.add_argument("--stems-dir", default="data/stems", help="Path to stems directory.")
    parser.add_argument(
        "--output",
        default="frontend/public/data/melodic_timbre_models.json",
        help="Output path for learned models JSON.",
    )
    parser.add_argument("--max-tracks", type=int, default=None, help="Max processed track JSONs to analyze.")
    parser.add_argument("--max-stems", type=int, default=None, help="Max stem track folders to analyze.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    learner = MelodicTimbreLearner(processed_dir=args.processed_dir, stems_dir=args.stems_dir)
    learner.learn_and_export(output_path=args.output, max_tracks=args.max_tracks, max_stems=args.max_stems)


if __name__ == "__main__":
    main()
