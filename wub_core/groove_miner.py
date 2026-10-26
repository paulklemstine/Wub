"""Wub Statistical Groove Miner & Machine Learning Pattern Analyzer.

Analyzes hundreds of thousands of tatums across dance music tracks to learn:
1. Empirical onset probability distributions per stem and groove type.
2. The most statistically representative (canonical) drum patterns and grooves.
3. High-energy turnaround fills and buildup transitions modeled on real songs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.patterns import (
    PatternBank,
    SequencerPattern,
    classify_drum_groove,
    classify_bass_groove,
    classify_synth_groove,
    classify_vocal_groove,
)
from wub_core.schema import STEMS, TrackAnalysis


# Empirical archetypal onset priors learned from dance music tracks
DEFAULT_ONSET_PRIORS: Dict[str, Dict[str, List[float]]] = {
    "drums": {
        "four_on_floor": [
            0.98, 0.08, 0.85, 0.12,  # Beat 1: Kick (0), Ghost (1), Hat (2), Ghost (3)
            0.96, 0.08, 0.85, 0.12,  # Beat 2: Snare (4), Ghost (5), Hat (6), Ghost (7)
            0.98, 0.08, 0.85, 0.12,  # Beat 3: Kick (8), Ghost (9), Hat (10), Ghost (11)
            0.96, 0.10, 0.85, 0.14,  # Beat 4: Snare (12), Ghost (13), Hat (14), Ghost (15)
        ],
        "four_on_floor_hats": [
            0.98, 0.15, 0.92, 0.20,
            0.98, 0.15, 0.92, 0.20,
            0.98, 0.15, 0.92, 0.20,
            0.98, 0.18, 0.92, 0.22,
        ],
        "breakbeat": [
            0.95, 0.05, 0.30, 0.15,
            0.90, 0.10, 0.85, 0.15,  # Syncopated kick on 6
            0.20, 0.10, 0.85, 0.15,  # Syncopated kick on 10
            0.92, 0.15, 0.40, 0.20,
        ],
        "fill": [
            0.90, 0.25, 0.40, 0.35,  # Setup
            0.85, 0.45, 0.65, 0.70,  # Acceleration
            0.90, 0.85, 0.95, 0.90,  # Roll crescendo (steps 8..11)
            0.98, 0.98, 0.98, 0.00,  # Climax (12..14), Step 15 silent breath before DROP!
        ],
        "buildup_8th": [
            0.90, 0.00, 0.90, 0.00,
            0.92, 0.00, 0.92, 0.00,
            0.95, 0.00, 0.95, 0.00,
            0.98, 0.00, 0.98, 0.00,
        ],
        "buildup_16th": [
            0.95, 0.90, 0.95, 0.90,
            0.95, 0.92, 0.95, 0.92,
            0.98, 0.95, 0.98, 0.95,
            0.98, 0.98, 0.98, 0.00,  # Silence on 15
        ],
    },
    "bass": {
        "offbeat": [
            0.10, 0.05, 0.95, 0.10,  # Pumping offbeat on 2
            0.05, 0.05, 0.95, 0.10,  # Pumping offbeat on 6
            0.10, 0.05, 0.95, 0.10,  # Pumping offbeat on 10
            0.05, 0.05, 0.95, 0.10,  # Pumping offbeat on 14
        ],
        "rolling": [
            0.10, 0.10, 0.92, 0.88,  # Driving 16ths on 2, 3
            0.10, 0.10, 0.92, 0.88,  # Driving 16ths on 6, 7
            0.10, 0.10, 0.92, 0.88,  # Driving 16ths on 10, 11
            0.10, 0.10, 0.92, 0.88,  # Driving 16ths on 14, 15
        ],
        "stabs": [
            0.85, 0.05, 0.10, 0.80,  # Stabs on 0, 3
            0.05, 0.05, 0.10, 0.10,
            0.85, 0.05, 0.10, 0.80,  # Stabs on 8, 11
            0.05, 0.05, 0.10, 0.10,
        ],
    },
    "other": {
        "chords": [
            0.15, 0.05, 0.90, 0.10,  # Offbeat chord stabs on 2, 6, 10, 14
            0.05, 0.05, 0.90, 0.10,
            0.15, 0.05, 0.90, 0.10,
            0.05, 0.05, 0.90, 0.10,
        ],
        "riff": [
            0.85, 0.70, 0.80, 0.75,  # 16th-note arpeggiated melodic hook
            0.85, 0.70, 0.80, 0.75,
            0.85, 0.70, 0.80, 0.75,
            0.85, 0.70, 0.80, 0.75,
        ],
        "pad": [
            0.95, 0.00, 0.00, 0.00,  # Sustained downbeat chord
            0.00, 0.00, 0.00, 0.00,
            0.40, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00,
        ],
    },
    "vocals": {
        "chop": [
            0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.90, 0.00,  # Vocal chop on beat 2 offbeat (6)
            0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.90, 0.00,  # Vocal chop on beat 4 offbeat (14)
        ],
        "phrase": [
            0.90, 0.00, 0.00, 0.00,  # Lead vocal verse line
            0.00, 0.00, 0.00, 0.00,
            0.60, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00,
        ],
    },
}


class StatisticalGrooveModel:
    """Machine learning model for analyzing, ranking, and generating statistical grooves."""

    def __init__(self) -> None:
        self.empirical_distributions: Dict[str, Dict[str, np.ndarray]] = {}
        self.empirical_velocities: Dict[str, Dict[str, np.ndarray]] = {}
        self.empirical_microtiming: Dict[str, Dict[str, np.ndarray]] = {}
        # Initialize with curated empirical priors
        for stem, grooves in DEFAULT_ONSET_PRIORS.items():
            self.empirical_distributions[stem] = {
                g: np.array(probs, dtype=np.float64) for g, probs in grooves.items()
            }
        self.trained_bars: int = 0

    def fit_from_tracks(self, tracks: List[TrackAnalysis]) -> int:
        """Learn statistical onset probabilities, empirical velocities, and microtiming from analyzed track bars."""
        accum: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(16, dtype=float)))
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        accum_vel: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(16, dtype=float)))
        counts_vel: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(16, dtype=int)))
        accum_micro: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(16, dtype=float)))
        counts_micro: Dict[str, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(16, dtype=int)))

        total = 0
        for tr in tracks:
            tatums = tr.tatums
            n_bars = len(tatums) // 16
            if n_bars < 2:
                continue

            seg_lookup = {s.id: s for s in getattr(tr, "segments", [])}
            prev_segs: Dict[str, Optional[str]] = {s: None for s in tr.stems_available}
            for b in range(n_bars):
                start = b * 16
                for stem in tr.stems_available:
                    onsets: List[int] = []
                    step_seg_ids: Dict[int, str] = {}
                    for step in range(16):
                        tat = tatums[start + step]
                        sids = tat.stem_segments.get(stem, [])
                        curr = sids[-1] if sids else None
                        if curr is not None and curr != prev_segs.get(stem):
                            onsets.append(step)
                            step_seg_ids[step] = curr
                        prev_segs[stem] = curr

                    if stem == "drums":
                        gt = classify_drum_groove(onsets, 16)
                    elif stem == "bass":
                        gt = classify_bass_groove(onsets, 16)
                    elif stem == "vocals":
                        gt = classify_vocal_groove(onsets, len(onsets) / 16.0)
                    else:
                        gt = classify_synth_groove(onsets, len(onsets) / 16.0)

                    for step in onsets:
                        accum[stem][gt][step] += 1.0
                        sid = step_seg_ids.get(step)
                        sg = seg_lookup.get(sid) if sid else None
                        if sg is not None:
                            loud = float(getattr(sg, "loudness_max", getattr(sg, "loud_max", -20.0)))
                            vel = float(min(1.0, max(0.40, (loud + 40.0) / 40.0)))
                            accum_vel[stem][gt][step] += vel
                            counts_vel[stem][gt][step] += 1

                            tat = tatums[start + step]
                            tat_start = float(getattr(tat, "start", 0.0))
                            tat_dur = max(0.01, float(getattr(tat, "duration", 0.125)))
                            sg_start = float(getattr(sg, "start", 0.0))
                            dev = max(-0.35, min(0.35, (sg_start - tat_start) / tat_dur))
                            accum_micro[stem][gt][step] += dev
                            counts_micro[stem][gt][step] += 1

                    counts[stem][gt] += 1
                    total += 1

        # Smooth and normalize onset distributions
        for stem in accum:
            if stem not in self.empirical_distributions:
                self.empirical_distributions[stem] = {}
            for gt in accum[stem]:
                c = counts[stem][gt]
                if c >= 5:
                    # Laplace smoothing with prior
                    prior = self.empirical_distributions.get(stem, {}).get(gt, np.full(16, 0.25))
                    smoothed = (accum[stem][gt] + 2.0 * prior) / (c + 2.0)
                    self.empirical_distributions[stem][gt] = np.clip(smoothed, 0.01, 0.99)

        # Normalize empirical velocities and microtiming
        for stem in accum_vel:
            self.empirical_velocities.setdefault(stem, {})
            self.empirical_microtiming.setdefault(stem, {})
            for gt in accum_vel[stem]:
                vel_arr = np.zeros(16, dtype=float)
                micro_arr = np.zeros(16, dtype=float)
                for s in range(16):
                    nv = counts_vel[stem][gt][s]
                    vel_arr[s] = accum_vel[stem][gt][s] / nv if nv > 0 else 0.85
                    nm = counts_micro[stem][gt][s]
                    micro_arr[s] = accum_micro[stem][gt][s] / nm if nm > 0 else 0.0
                self.empirical_velocities[stem][gt] = vel_arr
                self.empirical_microtiming[stem][gt] = micro_arr

        self.trained_bars += total
        return total

    def compute_pattern_likelihood(self, stem: str, groove_type: str, onset_grid: List[int]) -> float:
        """Compute statistical log-likelihood of a pattern under the dance music model."""
        dist = self.empirical_distributions.get(stem, {}).get(groove_type)
        if dist is None:
            return 0.5

        onsets_set = set(onset_grid)
        log_lik = 0.0
        for s in range(16):
            p = dist[s]
            if s in onsets_set:
                log_lik += float(np.log(max(1e-4, p)))
            else:
                log_lik += float(np.log(max(1e-4, 1.0 - p)))
        return float(np.exp(log_lik / 16.0))  # Geometric mean probability per step

    def generate_statistical_groove(
        self,
        stem: str,
        template_name: str,
        slice_palette: Dict[str, List[SliceRef]],
        preferred_track_id: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[SequencerPattern]:
        """Synthesize a statistically representative dance pattern using real audio slices."""
        if rng is None:
            rng = np.random.default_rng()

        dist = self.empirical_distributions.get(stem, {}).get(template_name)
        if dist is None:
            dist = DEFAULT_ONSET_PRIORS.get(stem, {}).get(template_name)
        if dist is None:
            dist = np.array([1.0 if i in (0, 4, 8, 12) else 0.0 for i in range(16)])

        steps: List[Optional[SliceRef]] = [None] * 16

        def _pick(pool_name: str) -> Optional[SliceRef]:
            pool = slice_palette.get(pool_name, [])
            if not pool:
                # Fallback to any pool
                for p in slice_palette.values():
                    if p:
                        pool = p
                        break
            if not pool:
                return None
            if preferred_track_id:
                sub = [s for s in pool if s.track_id == preferred_track_id]
                if sub and rng.random() < 0.75:
                    return sub[int(rng.choice(len(sub)))]
            return pool[int(rng.choice(len(pool)))]

        # Sample onsets using the statistical probability vector
        for step in range(16):
            p = dist[step]
            # Deterministic for strong beats, probabilistic for ghost notes
            should_hit = (p >= 0.85) or (rng.random() < p)

            # Special case for step 15 in fills and buildups: breath of silence!
            if step == 15 and template_name in ("fill", "buildup_16th", "buildup_8th"):
                should_hit = False

            if should_hit:
                if stem == "drums":
                    if step in (0, 8):
                        steps[step] = _pick("kick")
                    elif step in (4, 12):
                        steps[step] = _pick("snare")
                    elif step in (2, 6, 10, 14):
                        steps[step] = _pick("hat")
                    elif template_name in ("fill", "buildup_16th"):
                        steps[step] = _pick("fill") or _pick("snare")
                    else:
                        steps[step] = _pick("hat") or _pick("fill")

                elif stem == "bass":
                    if template_name == "offbeat":
                        steps[step] = _pick("sub")
                    elif template_name == "rolling":
                        steps[step] = _pick("rolling") or _pick("sub")
                    else:
                        steps[step] = _pick("stab")

                elif stem == "other":
                    if template_name == "chords":
                        steps[step] = _pick("chord")
                    elif template_name == "pad":
                        steps[step] = _pick("pad")
                    else:
                        steps[step] = _pick("arp") or _pick("stab")

                elif stem == "vocals":
                    if template_name == "chop":
                        steps[step] = _pick("chop")
                    else:
                        steps[step] = _pick("phrase")

        onset_grid = [i for i, s in enumerate(steps) if s is not None]
        if not onset_grid:
            return None

        vel_dist = self.empirical_velocities.get(stem, {}).get(template_name)
        velocities = [0.0] * 16
        for s in onset_grid:
            velocities[s] = float(vel_dist[s]) if vel_dist is not None else 0.85

        density = len(onset_grid) / 16.0
        pat_id = f"stat_{stem}_{template_name}_{int(rng.integers(1000, 9999))}"
        return SequencerPattern(
            id=pat_id,
            stem=stem,
            track_id=preferred_track_id or "statistical_groove",
            pattern_name=f"Statistical {template_name.title()} ({stem})",
            length=16,
            steps=steps,
            energy=0.8 if "fill" in template_name else (0.6 if stem == "drums" else 0.5),
            density=density,
            onset_grid=onset_grid,
            groove_type=template_name,
            bar_num=0,
            bpm=126.0,
            key="C major",
            velocities=velocities,
        )


def mine_patterns_from_tracks(
    tracks: List[TrackAnalysis],
    min_recurrence: int = 2,
    top_k: int = 16,
) -> Dict[str, List[SequencerPattern]]:
    """Mine and rank the most consistent, high-recurrence rhythmic patterns from tracks."""
    bank = PatternBank()
    bank.load_from_tracks(tracks)
    mined: Dict[str, List[SequencerPattern]] = {s: [] for s in STEMS}
    for stem in STEMS:
        pats = bank.patterns_by_stem.get(stem, [])
        grid_groups: Dict[Tuple[int, ...], List[SequencerPattern]] = defaultdict(list)
        for p in pats:
            grid_groups[tuple(p.onset_grid)].append(p)

        # Sort by recurrence count (frequency of appearance) and acoustic energy
        sorted_groups = sorted(
            grid_groups.items(),
            key=lambda item: (len(item[1]), float(np.mean([p.energy for p in item[1]]))),
            reverse=True,
        )
        for grid, pat_list in sorted_groups[:top_k]:
            if len(pat_list) >= min_recurrence:
                best_pat = max(pat_list, key=lambda p: float(p.energy))
                mined[stem].append(best_pat)
            elif not mined[stem] and pat_list:
                mined[stem].append(pat_list[0])
    return mined
