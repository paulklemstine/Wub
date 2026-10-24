"""Wub Groovebox: Sequencer Pattern Extraction and Memory Bank.

Extracts coherent 16-tatum (1 bar) and 32-tatum (2 bar) sequencer patterns
directly from multi-stem dance tracks. Patterns preserve the natural groove,
kick-bass pocket, and phrasing created by dance music producers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.model_markov import SliceRef, StemPool, build_pools
from wub_core.schema import STEMS, TrackAnalysis


@dataclass
class SequencerPattern:
    id: str
    stem: str
    track_id: str
    pattern_name: str
    length: int  # 16 (1 bar) or 32 (2 bars)
    steps: List[Optional[SliceRef]]  # Length 16 or 32
    energy: float  # Mean loudness/magnitude 0.0..1.0
    density: float  # Fraction of active (non-silent) steps
    onset_grid: List[int]  # Which step indices (0..15) have slice onsets
    groove_type: str  # e.g. "four_on_floor", "break", "fill", "rolling", "offbeat", "stabs", "riff", "pad"
    bpm: float = 126.0
    key: str = "C major"


def classify_drum_groove(onset_grid: List[int], length: int) -> str:
    """Classify 16-step drum pattern into dance groove categories."""
    first_16 = set(s for s in onset_grid if s < 16)
    has_four_on_floor = {0, 4, 8, 12}.issubset(first_16)
    density = len(first_16) / 16.0

    if density >= 0.75:
        return "fill"  # High density snare/drum roll or turnaround
    if has_four_on_floor:
        if 6 in first_16 or 14 in first_16 or 2 in first_16 or 10 in first_16:
            return "four_on_floor_hats"
        return "four_on_floor"
    if density <= 0.25:
        return "minimal"
    return "breakbeat"


def classify_bass_groove(onset_grid: List[int], length: int) -> str:
    """Classify 16-step bassline pattern into groove categories."""
    first_16 = set(s for s in onset_grid if s < 16)
    offbeats = {2, 6, 10, 14}
    density = len(first_16) / 16.0

    if density >= 0.6:
        return "rolling"  # Continuous 16th note rolling bassline
    if offbeats.intersection(first_16) and not {0, 4, 8, 12}.issubset(first_16):
        return "offbeat"  # Pumping offbeat bass avoiding kicks
    if density <= 0.25:
        return "sustained"
    return "stabs"


def classify_synth_groove(onset_grid: List[int], density: float) -> str:
    if density >= 0.5:
        return "riff"  # Arpeggio / driving synth hook
    if density <= 0.2:
        return "pad"  # Ambient pad / sustained chords
    return "hook"


class PatternBank:
    """In-memory bank of categorized 1-bar and 2-bar sequencer patterns."""

    def __init__(self) -> None:
        self.patterns_by_stem: Dict[str, List[SequencerPattern]] = {s: [] for s in STEMS}
        self.patterns_by_type: Dict[str, Dict[str, List[SequencerPattern]]] = {
            s: {} for s in STEMS
        }

    def load_from_tracks(self, tracks: List[TrackAnalysis], min_bpm: float = 118.0, max_bpm: float = 138.0) -> int:
        """Extract all valid 16-step and 32-step patterns from analyzed tracks."""
        pools = build_pools(tracks, exclude_silent=False)
        total_extracted = 0

        for tr in tracks:
            if not (min_bpm <= tr.bpm <= max_bpm):
                continue
            if len(tr.stems_available) < 2 and len(tracks) > 5:
                # In rich corpus, prioritize multi-stem tracks with true separated stems
                if len(tr.stems_available) == 1 and "other" in tr.stems_available:
                    pass

            n_tatums = len(tr.tatums)
            if n_tatums < 32:
                continue

            # Build quick map from (stem, tatum_idx) -> SliceRef
            tatum_slice_map: Dict[Tuple[str, int], SliceRef] = {}
            for stem in tr.stems_available:
                stem_pool = pools.get(stem)
                stem_slices = stem_pool.slices if stem_pool else []
                # Group by segment_id for instant lookup
                seg_lookup = {s.segment_id: s for s in stem_slices if s.track_id == tr.track_id}
                for t_idx, tat in enumerate(tr.tatums):
                    seg_ids = tat.stem_segments.get(stem, [])
                    if seg_ids and seg_ids[-1] in seg_lookup:
                        sref = seg_lookup[seg_ids[-1]]
                        tatum_slice_map[(stem, t_idx)] = sref

            # Find downbeat bars (every 16 tatums starting from a downbeat)
            # Find the first downbeat (beat 0, tatum 0)
            downbeat_indices = [
                i for i, tat in enumerate(tr.tatums)
                if tat.beat_position == 0 and tat.tatum_position == 0
            ]
            if not downbeat_indices:
                downbeat_indices = list(range(0, n_tatums - 16, 16))

            for start_idx in downbeat_indices:
                if start_idx + 16 > n_tatums:
                    continue

                bar_num = start_idx // 16

                for stem in tr.stems_available:
                    steps: List[Optional[SliceRef]] = []
                    onset_grid: List[int] = []
                    loudness_vals: List[float] = []

                    for step_idx in range(16):
                        t_pos = start_idx + step_idx
                        sref = tatum_slice_map.get((stem, t_pos))
                        if sref is not None and sref.loud_max > -48.0:
                            steps.append(sref)
                            onset_grid.append(step_idx)
                            loudness_vals.append(max(0.0, sref.loud_max + 60.0) / 60.0)
                        else:
                            steps.append(None)

                    density = len(onset_grid) / 16.0
                    # Skip completely silent bars for this stem
                    if density == 0.0:
                        continue

                    energy = float(np.mean(loudness_vals)) if loudness_vals else 0.0

                    # Classify groove type
                    if stem == "drums":
                        groove_type = classify_drum_groove(onset_grid, 16)
                    elif stem == "bass":
                        groove_type = classify_bass_groove(onset_grid, 16)
                    else:
                        groove_type = classify_synth_groove(onset_grid, density)

                    pattern_id = f"pat_{stem}_{tr.track_id[:6]}_{bar_num:03d}"
                    pat = SequencerPattern(
                        id=pattern_id,
                        stem=stem,
                        track_id=tr.track_id,
                        pattern_name=f"{tr.filename[:16]} Bar {bar_num} ({groove_type})",
                        length=16,
                        steps=steps,
                        energy=energy,
                        density=density,
                        onset_grid=onset_grid,
                        groove_type=groove_type,
                        bpm=tr.bpm,
                        key=tr.key,
                    )

                    self.patterns_by_stem[stem].append(pat)
                    self.patterns_by_type[stem].setdefault(groove_type, []).append(pat)
                    total_extracted += 1

        return total_extracted

    def get_patterns(
        self,
        stem: str,
        groove_type: Optional[str] = None,
        min_energy: float = 0.0,
        max_energy: float = 1.0,
    ) -> List[SequencerPattern]:
        """Query patterns matching stem, groove type, and energy."""
        candidates = self.patterns_by_type.get(stem, {}).get(groove_type, []) if groove_type else self.patterns_by_stem.get(stem, [])
        if not candidates:
            candidates = self.patterns_by_stem.get(stem, [])
        if not candidates:
            return []

        matched = [
            p for p in candidates
            if min_energy <= p.energy <= max_energy
        ]
        return matched if matched else candidates

    def pick_pattern(
        self,
        stem: str,
        groove_type: Optional[str] = None,
        min_energy: float = 0.0,
        max_energy: float = 1.0,
        preferred_track_id: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[SequencerPattern]:
        """Pick a musically compatible pattern using RNG."""
        if rng is None:
            rng = np.random.default_rng()

        candidates = self.get_patterns(stem, groove_type, min_energy, max_energy)
        if not candidates:
            return None

        if preferred_track_id:
            track_matches = [p for p in candidates if p.track_id == preferred_track_id]
            if track_matches and rng.random() < 0.65:
                return track_matches[int(rng.choice(len(track_matches)))]

        return candidates[int(rng.choice(len(candidates)))]
