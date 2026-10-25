"""Wub Groovebox Conductor: Clockwork Step Sequencer and Macro Arrangement Engine.

Controls song structure (Intro, Groove A, Buildup, Drop, Breakdown), selects
compatible 16-step sequencer patterns, handles 8-bar turnarounds and drum fills,
and maintains a rock-solid, jitter-free groove locked to the beat grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.composer import SongArranger
from wub_core.model_markov import SliceRef
from wub_core.patterns import MultiStemPattern, PatternBank, SequencerPattern, SongPhrase
from wub_core.schema import STEMS, TrackAnalysis


# Macro song arrangement section types
SECTIONS = ("GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN", "DROP")

# Standard section lengths in bars (4/4 time, 1 bar = 16 tatums)
SECTION_DURATIONS = {
    "GROOVE_A": 16,    # 16 bars = 256 tatums (main verse groove)
    "BUILDUP": 8,      # 8 bars = 128 tatums (snare rolls, rising tension)
    "DROP": 16,        # 16 bars = 256 tatums (peak energy drop, heavy drums & bass)
    "BREAKDOWN": 8,    # 8 bars = 128 tatums (melodic vocal/pad release)
    "INTRO": 8,        # 8 bars = 128 tatums (groove with synth hook)
}

# Next section transition map
NEXT_SECTION = {
    "INTRO": "GROOVE_A",
    "GROOVE_A": "BUILDUP",
    "BUILDUP": "DROP",
    "DROP": "BREAKDOWN",
    "BREAKDOWN": "DROP",  # From breakdown straight into the 2nd Drop
}


@dataclass
class GrooveboxState:
    section: str = "GROOVE_A"
    bar_in_section: int = 0
    phrase_bar: int = 0      # 0..7 (Bar 1 to Bar 8 of an 8-bar phrase)
    step_in_bar: int = 0     # 0..15 (16th note step within the bar)
    total_bars: int = 0
    total_steps: int = 0
    is_fill_bar: bool = False
    active_patterns: Dict[str, Optional[SequencerPattern]] = field(default_factory=dict)


class GrooveboxConductor:
    """Intelligent conductor that selects patterns and steps through the groovebox."""

    def __init__(self, pattern_bank: PatternBank, seed: int = 42) -> None:
        self.bank = pattern_bank
        self.rng = np.random.default_rng(seed)
        self.arranger = SongArranger(pattern_bank, seed=seed)
        self.state = GrooveboxState()
        self.state.section = "GROOVE_A"
        self.section_bars_remaining = SECTION_DURATIONS.get(self.state.section, 16)
        self.active_patterns: Dict[str, Optional[SequencerPattern]] = {s: None for s in STEMS}
        self.current_phrase: Optional[SongPhrase] = None
        self.current_pattern: Optional[MultiStemPattern] = None
        self.lead_track_id: Optional[str] = None
        self.lead_base_bar: int = 0
        self.active_slice_ids: Dict[str, Optional[str]] = {s: None for s in STEMS}
        self._init_section_patterns()

    def _init_section_patterns(self) -> None:
        """Select a cohesive 8-bar SongPhrase or fallback patterns matching the active section."""
        sec = self.state.section

        # Try to select an authentic multi-stem 8-bar phrase
        phrase = self.bank.pick_phrase(
            section=sec,
            preferred_track_id=self.lead_track_id,
            rng=self.rng,
        )
        if phrase and phrase.bars:
            self.current_phrase = phrase
            self.lead_track_id = phrase.track_id
            self.current_pattern = phrase.bars[0]
            self.lead_base_bar = self.current_pattern.bar_num
            for s in STEMS:
                self.active_patterns[s] = self.bank.get_track_pattern(
                    self.current_pattern.track_id, s, self.current_pattern.bar_num
                ) or self.bank.pick_pattern(s, preferred_track_id=self.lead_track_id, rng=self.rng)
            return

        # Fallback to single-stem patterns if no cohesive multi-stem phrase is available
        self.current_phrase = None
        self.current_pattern = None

        if sec == "DROP":
            drum_type = "four_on_floor_hats"
            bass_type = "rolling" if self.rng.random() < 0.6 else "offbeat"
            min_energy = 0.5
        elif sec == "BUILDUP":
            drum_type = "four_on_floor_hats"
            bass_type = "rolling"
            min_energy = 0.4
        elif sec == "BREAKDOWN":
            drum_type = "four_on_floor_hats"
            bass_type = "sustained"
            min_energy = 0.2
        elif sec == "INTRO":
            drum_type = "four_on_floor_hats"
            bass_type = "offbeat"
            min_energy = 0.3
        else:  # GROOVE_A
            drum_type = "four_on_floor_hats"
            bass_type = "offbeat"
            min_energy = 0.35

        self.active_patterns["drums"] = self.bank.pick_pattern(
            "drums", groove_type=drum_type, min_energy=min_energy, rng=self.rng
        ) or self.bank.create_groove_pattern("drums", "four_on_floor", rng=self.rng)

        if self.active_patterns["drums"]:
            self.lead_track_id = self.active_patterns["drums"].track_id
            self.lead_base_bar = self.active_patterns["drums"].bar_num

        self.active_patterns["bass"] = self.bank.pick_pattern(
            "bass", groove_type=bass_type, min_energy=min_energy,
            preferred_track_id=self.lead_track_id, rng=self.rng
        ) or self.bank.create_groove_pattern("bass", "offbeat", preferred_track_id=self.lead_track_id, rng=self.rng)

        synth_type = "riff" if sec == "DROP" else ("pad" if sec in ("BREAKDOWN", "INTRO") else "hook")
        self.active_patterns["other"] = self.bank.pick_pattern(
            "other", groove_type=synth_type, min_energy=min_energy,
            preferred_track_id=self.lead_track_id, rng=self.rng
        ) or self.bank.create_groove_pattern("other", "chords" if sec in ("DROP", "GROOVE_A") else "pad", preferred_track_id=self.lead_track_id, rng=self.rng)

        self.active_patterns["vocals"] = self.bank.pick_pattern(
            "vocals", min_energy=0.0,
            preferred_track_id=self.lead_track_id, rng=self.rng
        ) or self.bank.create_groove_pattern("vocals", "chop", preferred_track_id=self.lead_track_id, rng=self.rng)

    def _on_bar_start(self) -> None:
        """Triggered on Step 0 of each bar (beat 1)."""
        self.state.phrase_bar = self.state.total_bars % 8
        phrase_bar = self.state.phrase_bar  # 0..7
        self.state.bar_in_section += 1
        self.section_bars_remaining -= 1
        self.active_slice_ids = {s: None for s in STEMS}

        # Check section transition
        if self.section_bars_remaining <= 0:
            next_sec = NEXT_SECTION.get(self.state.section, "GROOVE_A")
            self.state.section = next_sec
            self.state.bar_in_section = 0
            self.section_bars_remaining = SECTION_DURATIONS.get(next_sec, 16)
            self._init_section_patterns()
            return

        is_turnaround = (phrase_bar == 7)
        self.state.is_fill_bar = is_turnaround

        # Cohesive Multi-Stem Phrase progression
        if self.current_phrase and phrase_bar < len(self.current_phrase.bars):
            if phrase_bar == 0 and self.state.total_bars > 0:
                # Plan the next phrase and macro section using SongArranger ML
                cur_key = self.current_phrase.key if self.current_phrase else "C major"
                cur_bpm = self.current_phrase.bpm if self.current_phrase else 126.0
                next_p, next_sec, is_sec_change = self.arranger.plan_next_phrase(
                    current_phrase=self.current_phrase,
                    current_section=self.state.section,
                    current_key=cur_key,
                    current_bpm=cur_bpm,
                    bars_in_section=self.state.bar_in_section,
                    total_bars=self.state.total_bars,
                )
                self.current_phrase = next_p
                if is_sec_change:
                    self.state.section = next_sec
                    self.state.bar_in_section = 0
                    self.section_bars_remaining = SECTION_DURATIONS.get(next_sec, 16)

            self.current_pattern = self.current_phrase.bars[phrase_bar]
            self.lead_track_id = self.current_phrase.track_id
            for s in STEMS:
                self.active_patterns[s] = self.bank.get_track_pattern(
                    self.current_pattern.track_id, s, self.current_pattern.bar_num
                ) or self.active_patterns.get(s)

            if is_turnaround:
                fill_drums = (
                    self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
                    or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
                )
                if fill_drums:
                    self.active_patterns["drums"] = fill_drums
                    if self.current_pattern:
                        self.current_pattern.stems["drums"] = list(fill_drums.steps)
            return

        # Fallback single-stem mode (used in tests with small synthetic tracks)
        if is_turnaround:
            fill_pat = (
                self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
                or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
            )
            if fill_pat:
                self.active_patterns["drums"] = fill_pat
        elif phrase_bar == 0:
            self._init_section_patterns()
        elif phrase_bar == 3 and self.rng.random() < 0.6:
            var_drums = self.bank.pick_pattern("drums", min_energy=0.4, preferred_track_id=self.lead_track_id, rng=self.rng)
            if var_drums:
                self.active_patterns["drums"] = var_drums
        elif phrase_bar == 4:
            lift_drums = self.bank.pick_pattern("drums", groove_type="four_on_floor_hats", preferred_track_id=self.lead_track_id, rng=self.rng)
            if lift_drums:
                self.active_patterns["drums"] = lift_drums
            lift_bass = self.bank.pick_pattern("bass", groove_type="rolling", preferred_track_id=self.lead_track_id, rng=self.rng)
            if lift_bass:
                self.active_patterns["bass"] = lift_bass
        else:
            if self.lead_track_id:
                track_bar = self.lead_base_bar + phrase_bar
                lead_drum_pat = self.bank.get_track_pattern(self.lead_track_id, "drums", track_bar)
                if lead_drum_pat and lead_drum_pat.energy >= 0.2:
                    self.active_patterns["drums"] = lead_drum_pat
                lead_bass_pat = self.bank.get_track_pattern(self.lead_track_id, "bass", track_bar)
                if lead_bass_pat:
                    self.active_patterns["bass"] = lead_bass_pat
                lead_other_pat = self.bank.get_track_pattern(self.lead_track_id, "other", track_bar)
                if lead_other_pat:
                    self.active_patterns["other"] = lead_other_pat

        if self.active_patterns.get("drums") is None:
            self.active_patterns["drums"] = self.bank.create_groove_pattern("drums", "four_on_floor_hats", rng=self.rng)
        if self.active_patterns.get("bass") is None:
            self.active_patterns["bass"] = self.bank.create_groove_pattern("bass", "offbeat", rng=self.rng)
        if self.active_patterns.get("other") is None:
            self.active_patterns["other"] = self.bank.create_groove_pattern("other", "chords", rng=self.rng)
        if self.active_patterns.get("vocals") is None:
            self.active_patterns["vocals"] = self.bank.create_groove_pattern("vocals", "chop", rng=self.rng)

    def next_step(self) -> Tuple[Dict[str, Optional[SliceRef]], dict]:
        """Advance the groovebox sequencer by 1 tatum step (1/16th note)."""
        step = self.state.step_in_bar

        if step == 0:
            self._on_bar_start()

        sec = self.state.section
        slices: Dict[str, Optional[SliceRef]] = {}

        for stem in STEMS:
            if self.current_pattern and stem in self.current_pattern.stems:
                stem_steps = self.current_pattern.stems[stem]
                candidate_sref = stem_steps[step] if step < len(stem_steps) else None
            else:
                pat = self.active_patterns.get(stem)
                if pat and pat.steps:
                    candidate_sref = pat.steps[step % len(pat.steps)]
                else:
                    candidate_sref = None

            # Onset vs sustain awareness: only re-trigger if segment changes or at start
            if candidate_sref is None:
                slices[stem] = None
                self.active_slice_ids[stem] = None
            else:
                seg_id = candidate_sref.segment_id
                if seg_id != self.active_slice_ids.get(stem) or self.state.total_steps == 0:
                    slices[stem] = candidate_sref
                    self.active_slice_ids[stem] = seg_id
                else:
                    slices[stem] = None

        meta = {
            "section": sec,
            "phrase_bar": self.state.phrase_bar + 1,
            "step_in_bar": step + 1,
            "total_bars": self.state.total_bars,
            "is_fill": self.state.is_fill_bar,
            "patterns": {
                s: (
                    f"{self.current_pattern.track_title[:16]} Bar {self.current_pattern.bar_num} ({self.current_pattern.groove_types.get(s, 'groove')})"
                    if (self.current_pattern and self.current_pattern.track_title)
                    else (self.active_patterns[s].pattern_name if self.active_patterns.get(s) else "empty")
                )
                for s in STEMS
            },
        }

        # Advance step counter
        self.state.step_in_bar = (step + 1) % 16
        self.state.total_steps += 1
        if self.state.step_in_bar == 0:
            self.state.total_bars += 1

        return slices, meta
