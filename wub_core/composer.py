"""Wub Tier 2: Phrase Pattern Composer.

Learns and models bar-to-bar pattern progressions across 4-bar and 8-bar
musical phrases. Bridges the Tier 3 Macro Song Conductor (which plans sections
and energy envelopes) and the Tier 1 Micro Pattern Generator (which provides
16-step rhythmic grid slices).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.patterns import (
    PatternBank,
    SequencerPattern,
    classify_drum_groove,
    classify_bass_groove,
    classify_synth_groove,
)
from wub_core.schema import STEMS, TrackAnalysis


@dataclass
class BarState:
    """Acoustic and metrical state of a 1-bar pattern."""
    drum_type: str = "four_on_floor"
    bass_type: str = "offbeat"
    other_type: str = "hook"
    energy: float = 0.5
    phrase_bar: int = 0  # 0..7
    section: str = "GROOVE_A"


class PhraseComposerModel:
    """Trained Markov model of bar-to-bar musical phrase progressions."""

    def __init__(self) -> None:
        # Transitions: (phrase_bar, stem, current_groove) -> {next_groove: count}
        self.transitions: Dict[Tuple[int, str, str], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Empirical energy distribution per phrase bar: phrase_bar -> list of energies
        self.energy_by_bar: Dict[int, List[float]] = defaultdict(list)
        self.trained = False

    def train_on_tracks(self, tracks: List[TrackAnalysis], bank: PatternBank) -> int:
        """Train bar-to-bar transition probabilities from consecutive track bars."""
        n_transitions = 0

        for tr in tracks:
            if len(tr.stems_available) < 2:
                continue

            n_tatums = len(tr.tatums)
            n_bars = n_tatums // 16
            if n_bars < 4:
                continue

            for bar in range(n_bars - 1):
                p_bar_curr = bar % 8
                p_bar_next = (bar + 1) % 8

                for stem in ("drums", "bass", "other"):
                    pat_curr = bank.get_track_pattern(tr.track_id, stem, bar)
                    pat_next = bank.get_track_pattern(tr.track_id, stem, bar + 1)

                    if pat_curr and pat_next:
                        key = (p_bar_curr, stem, pat_curr.groove_type)
                        self.transitions[key][pat_next.groove_type] += 1.0
                        self.energy_by_bar[p_bar_curr].append(pat_curr.energy)
                        n_transitions += 1

        self.trained = True
        return n_transitions

    def sample_next_groove(
        self,
        phrase_bar: int,
        stem: str,
        current_groove: str,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Sample the next groove type for this stem given current phrase position."""
        if rng is None:
            rng = np.random.default_rng()

        # Turnaround Bar 8 rules
        if phrase_bar == 7 and stem == "drums":
            return "fill"
        if phrase_bar == 7 and stem == "bass":
            return "stabs"

        # Bar 5 energy lift
        if phrase_bar == 4 and stem == "drums":
            return "four_on_floor_hats"

        key = (phrase_bar, stem, current_groove)
        counts = self.transitions.get(key)
        if not counts:
            # Fallbacks based on stem defaults
            if stem == "drums":
                return "four_on_floor_hats" if phrase_bar in (4, 5, 6) else "four_on_floor"
            elif stem == "bass":
                return "rolling" if phrase_bar in (4, 5) else "offbeat"
            else:
                return "riff" if phrase_bar in (4, 5) else "hook"

        total = sum(counts.values())
        choices = list(counts.keys())
        probs = [c / total for c in counts.values()]
        return str(rng.choice(choices, p=probs))


class PatternComposer:
    """Orchestrates 4-bar and 8-bar musical compositions using Tier 1 patterns."""

    def __init__(self, bank: PatternBank, seed: int = 42) -> None:
        self.bank = bank
        self.rng = np.random.default_rng(seed)
        self.model = PhraseComposerModel()
        self.lead_track_id: Optional[str] = None
        self.lead_base_bar: int = 0
        self.current_bar_state = BarState()
        self.active_patterns: Dict[str, Optional[SequencerPattern]] = {s: None for s in STEMS}
        self.active_slice_ids: Dict[str, Optional[str]] = {s: None for s in STEMS}

    def train(self, tracks: List[TrackAnalysis]) -> int:
        return self.model.train_on_tracks(tracks, self.bank)

    def compose_bar(
        self,
        phrase_bar: int,
        section: str,
        target_energy: float = 0.5,
        lead_track_id: Optional[str] = None,
    ) -> Dict[str, Optional[SequencerPattern]]:
        """Compose the 16-step patterns for all stems for the upcoming bar."""
        if lead_track_id:
            self.lead_track_id = lead_track_id

        is_turnaround = (phrase_bar == 7)  # Bar 8 of 8-bar phrase

        # 1. Determine target groove types per stem using Tier 2 Phrase Model
        target_grooves: Dict[str, str] = {}
        for stem in ("drums", "bass", "other"):
            curr_g = getattr(self.current_bar_state, f"{stem}_type", "four_on_floor")
            target_grooves[stem] = self.model.sample_next_groove(
                phrase_bar, stem, curr_g, rng=self.rng
            )

        # 2. Select patterns from Tier 1 PatternBank
        new_patterns: Dict[str, Optional[SequencerPattern]] = {}

        if is_turnaround:
            # Bar 8: Guaranteed high-energy drum fill
            fill_pat = (
                self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
                or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
            )
            new_patterns["drums"] = fill_pat
            # Cut/stab bass on turnaround
            new_patterns["bass"] = self.bank.pick_pattern(
                "bass", groove_type="stabs", preferred_track_id=self.lead_track_id, rng=self.rng
            )
            new_patterns["other"] = self.bank.pick_pattern(
                "other", preferred_track_id=self.lead_track_id, rng=self.rng
            )
            new_patterns["vocals"] = self.bank.create_groove_pattern(
                "vocals", "chop", preferred_track_id=self.lead_track_id, rng=self.rng
            )

        elif self.lead_track_id and phrase_bar in (1, 2, 3, 5, 6):
            # Follow natural track progression for mid-phrase continuity
            track_bar = self.lead_base_bar + phrase_bar
            for stem in STEMS:
                tr_pat = self.bank.get_track_pattern(self.lead_track_id, stem, track_bar)
                if tr_pat:
                    new_patterns[stem] = tr_pat
                else:
                    new_patterns[stem] = self.bank.pick_pattern(
                        stem, groove_type=target_grooves.get(stem),
                        preferred_track_id=self.lead_track_id, rng=self.rng
                    )

        else:
            # Bar 1 (phrase_bar 0) or Bar 5 (phrase_bar 4): Pick solid thematic anchor
            for stem in ("drums", "bass", "other"):
                new_patterns[stem] = self.bank.pick_pattern(
                    stem,
                    groove_type=target_grooves.get(stem),
                    min_energy=target_energy * 0.7,
                    preferred_track_id=self.lead_track_id,
                    rng=self.rng,
                )
            new_patterns["vocals"] = self.bank.pick_pattern(
                "vocals", preferred_track_id=self.lead_track_id, rng=self.rng
            )
            if new_patterns.get("drums") and phrase_bar == 0:
                self.lead_track_id = new_patterns["drums"].track_id
                self.lead_base_bar = new_patterns["drums"].bar_num

        self.active_patterns = new_patterns
        self.current_bar_state = BarState(
            drum_type=target_grooves.get("drums", "four_on_floor"),
            bass_type=target_grooves.get("bass", "offbeat"),
            other_type=target_grooves.get("other", "hook"),
            energy=target_energy,
            phrase_bar=phrase_bar,
            section=section,
        )
        return new_patterns

    def get_step_slices(
        self,
        step_idx: int,
        phrase_bar: int,
        section: str,
    ) -> Dict[str, Optional[SliceRef]]:
        """Extract slices for step 0..15 with onset-vs-sustain awareness."""
        slices: Dict[str, Optional[SliceRef]] = {}

        for stem in STEMS:
            pat = self.active_patterns.get(stem)
            if pat is None or not pat.steps:
                slices[stem] = None
                self.active_slice_ids[stem] = None
                continue

            # Macro section muting
            if section == "BREAKDOWN":
                if stem == "drums":
                    slices[stem] = pat.steps[step_idx] if step_idx in (6, 14) and self.rng.random() < 0.4 else None
                    continue
                elif stem == "bass":
                    slices[stem] = None
                    continue

            if section == "BUILDUP" and phrase_bar in (6, 7) and stem == "bass":
                slices[stem] = None
                continue

            # Turnaround Bar 8: cut bass on steps 12..15
            if phrase_bar == 7 and stem == "bass" and step_idx >= 12:
                slices[stem] = None
                continue

            raw_sref = pat.steps[step_idx % len(pat.steps)]
            if raw_sref is None:
                slices[stem] = None
                self.active_slice_ids[stem] = None
                continue

            # Onset vs sustain: only trigger if segment changes
            seg_id = raw_sref.segment_id
            if seg_id != self.active_slice_ids[stem]:
                slices[stem] = raw_sref
                self.active_slice_ids[stem] = seg_id
            else:
                slices[stem] = None  # Sustain existing slice naturally!

        return slices
