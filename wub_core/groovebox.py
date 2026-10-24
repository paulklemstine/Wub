"""Wub Groovebox Conductor: Clockwork Step Sequencer and Macro Arrangement Engine.

Controls song structure (Intro, Groove A, Buildup, Drop, Breakdown), selects
compatible 16-step sequencer patterns, handles 8-bar turnarounds and drum fills,
and maintains a rock-solid, jitter-free groove locked to the beat grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.patterns import PatternBank, SequencerPattern
from wub_core.schema import STEMS, TrackAnalysis


# Macro song arrangement section types
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")

# Standard section lengths in bars (4/4 time, 1 bar = 16 tatums)
SECTION_DURATIONS = {
    "INTRO": 8,        # 8 bars = 128 tatums (hats, subtle synth)
    "GROOVE_A": 16,    # 16 bars = 256 tatums (main verse groove)
    "BUILDUP": 8,      # 8 bars = 128 tatums (snare rolls, rising tension)
    "DROP": 16,        # 16 bars = 256 tatums (peak energy drop, heavy drums & bass)
    "BREAKDOWN": 8,    # 8 bars = 128 tatums (melodic vocal/pad release, kick drops out)
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
        self.state = GrooveboxState()
        self.state.section = "GROOVE_A"
        self.section_bars_remaining = SECTION_DURATIONS.get(self.state.section, 16)
        self.active_patterns: Dict[str, Optional[SequencerPattern]] = {s: None for s in STEMS}
        self.lead_track_id: Optional[str] = None
        self._init_section_patterns()

    def _init_section_patterns(self) -> None:
        """Select initial patterns matching the current section."""
        sec = self.state.section

        if sec == "DROP":
            drum_type = "four_on_floor_hats"
            bass_type = "rolling" if self.rng.random() < 0.6 else "offbeat"
            min_energy = 0.5
        elif sec == "BUILDUP":
            drum_type = "four_on_floor"
            bass_type = "stabs"
            min_energy = 0.4
        elif sec == "BREAKDOWN":
            drum_type = "minimal"
            bass_type = "sustained"
            min_energy = 0.1
        elif sec == "INTRO":
            drum_type = "minimal"
            bass_type = "stabs"
            min_energy = 0.2
        else:  # GROOVE_A
            drum_type = "four_on_floor"
            bass_type = "offbeat"
            min_energy = 0.35

        # Pick drums as the anchor
        self.active_patterns["drums"] = self.bank.pick_pattern(
            "drums", groove_type=drum_type, min_energy=min_energy, rng=self.rng
        )
        if self.active_patterns["drums"]:
            self.lead_track_id = self.active_patterns["drums"].track_id

        # Pick bass matching lead track or key
        self.active_patterns["bass"] = self.bank.pick_pattern(
            "bass", groove_type=bass_type, min_energy=min_energy,
            preferred_track_id=self.lead_track_id, rng=self.rng
        )

        # Pick synths / other
        synth_type = "riff" if sec == "DROP" else ("pad" if sec in ("BREAKDOWN", "INTRO") else "hook")
        self.active_patterns["other"] = self.bank.pick_pattern(
            "other", groove_type=synth_type, min_energy=min_energy,
            preferred_track_id=self.lead_track_id, rng=self.rng
        )

        # Pick vocals
        self.active_patterns["vocals"] = self.bank.pick_pattern(
            "vocals", min_energy=0.0,
            preferred_track_id=self.lead_track_id, rng=self.rng
        )

    def _on_bar_start(self) -> None:
        """Triggered on Step 0 of each bar (beat 1)."""
        self.state.phrase_bar = self.state.total_bars % 8
        self.state.bar_in_section += 1
        self.section_bars_remaining -= 1

        # Check section transition
        if self.section_bars_remaining <= 0:
            next_sec = NEXT_SECTION.get(self.state.section, "GROOVE_A")
            self.state.section = next_sec
            self.state.bar_in_section = 0
            self.section_bars_remaining = SECTION_DURATIONS.get(next_sec, 16)
            self._init_section_patterns()
            return

        # Check for 8-bar turnaround / fill bar
        is_turnaround = (self.state.phrase_bar == 7)  # Bar 8 of 8-bar phrase
        self.state.is_fill_bar = is_turnaround

        if is_turnaround:
            # Inject drum fill into the turnaround bar!
            fill_pat = self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
            if fill_pat:
                self.active_patterns["drums"] = fill_pat
        elif self.state.phrase_bar == 0:
            # Bar 1 of new phrase: restore solid groove
            self._init_section_patterns()
        elif self.state.phrase_bar == 4 and self.rng.random() < 0.5:
            # Bar 5 subtle variation (4-bar phrase switch)
            new_bass = self.bank.pick_pattern("bass", preferred_track_id=self.lead_track_id, rng=self.rng)
            if new_bass:
                self.active_patterns["bass"] = new_bass

    def next_step(self) -> Tuple[Dict[str, Optional[SliceRef]], dict]:
        """Advance the groovebox sequencer by 1 tatum step (1/16th note).
        
        Returns:
            slices: Dict mapping stem ("drums", "bass", "vocals", "other") to SliceRef or None.
            metadata: Dict with current section, bar (1..8), step (1..16), and fill status.
        """
        step = self.state.step_in_bar

        if step == 0:
            self._on_bar_start()

        sec = self.state.section
        slices: Dict[str, Optional[SliceRef]] = {}

        for stem in STEMS:
            pat = self.active_patterns.get(stem)
            if pat is None or not pat.steps:
                slices[stem] = None
                continue

            # In Breakdown, mute drums and bass to create the dramatic drop setup
            if sec == "BREAKDOWN":
                if stem == "drums":
                    # Only allow subtle hats/percs on offbeats or silence
                    slices[stem] = pat.steps[step] if step in (6, 14) and self.rng.random() < 0.4 else None
                    continue
                elif stem == "bass":
                    slices[stem] = None
                    continue

            # In Buildup last 2 bars (phrase_bar in 6, 7), drop the bass before the drop!
            if sec == "BUILDUP" and self.state.phrase_bar in (6, 7) and stem == "bass":
                slices[stem] = None
                continue

            step_idx = step % len(pat.steps)
            slices[stem] = pat.steps[step_idx]

        meta = {
            "section": sec,
            "phrase_bar": self.state.phrase_bar + 1,  # 1..8
            "step_in_bar": step + 1,                 # 1..16
            "total_bars": self.state.total_bars,
            "is_fill": self.state.is_fill_bar,
            "patterns": {
                s: (self.active_patterns[s].pattern_name if self.active_patterns.get(s) else "empty")
                for s in STEMS
            },
        }

        # Advance step counter
        self.state.step_in_bar = (step + 1) % 16
        self.state.total_steps += 1
        if self.state.step_in_bar == 0:
            self.state.total_bars += 1

        return slices, meta
