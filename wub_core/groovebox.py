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
        self.state = GrooveboxState()
        self.state.section = "GROOVE_A"
        self.section_bars_remaining = SECTION_DURATIONS.get(self.state.section, 16)
        self.active_patterns: Dict[str, Optional[SequencerPattern]] = {s: None for s in STEMS}
        self.lead_track_id: Optional[str] = None
        self.lead_base_bar: int = 0
        self.active_slice_ids: Dict[str, Optional[str]] = {s: None for s in STEMS}
        self._init_section_patterns()

    def _init_section_patterns(self) -> None:
        """Select initial patterns matching the current section."""
        sec = self.state.section

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

        # Pick drums as the anchor
        self.active_patterns["drums"] = self.bank.pick_pattern(
            "drums", groove_type=drum_type, min_energy=min_energy, rng=self.rng
        ) or self.bank.create_groove_pattern("drums", "four_on_floor", rng=self.rng)

        if self.active_patterns["drums"]:
            self.lead_track_id = self.active_patterns["drums"].track_id
            self.lead_base_bar = self.active_patterns["drums"].bar_num

        # Pick bass matching lead track or key
        self.active_patterns["bass"] = self.bank.pick_pattern(
            "bass", groove_type=bass_type, min_energy=min_energy,
            preferred_track_id=self.lead_track_id, rng=self.rng
        ) or self.bank.create_groove_pattern("bass", "offbeat", preferred_track_id=self.lead_track_id, rng=self.rng)

        # Pick synths / other
        synth_type = "riff" if sec == "DROP" else ("pad" if sec in ("BREAKDOWN", "INTRO") else "hook")
        self.active_patterns["other"] = self.bank.pick_pattern(
            "other", groove_type=synth_type, min_energy=min_energy,
            preferred_track_id=self.lead_track_id, rng=self.rng
        ) or self.bank.create_groove_pattern("other", "chords" if sec in ("DROP", "GROOVE_A") else "pad", preferred_track_id=self.lead_track_id, rng=self.rng)

        # Pick vocals
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

        # Check for 8-bar turnaround / fill bar
        is_turnaround = (phrase_bar == 7)  # Bar 8 of 8-bar phrase
        self.state.is_fill_bar = is_turnaround

        sec = self.state.section

        if is_turnaround:
            # Inject drum fill into the turnaround bar!
            fill_pat = (
                self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
                or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
            )
            if fill_pat:
                self.active_patterns["drums"] = fill_pat
        elif phrase_bar == 0:
            # Bar 1 of new phrase: restore solid groove
            self._init_section_patterns()
        elif phrase_bar == 3 and self.rng.random() < 0.6:
            # Bar 4 mini-turnaround (4-bar musical phrase boundary)
            var_drums = self.bank.pick_pattern("drums", min_energy=0.4, preferred_track_id=self.lead_track_id, rng=self.rng)
            if var_drums:
                self.active_patterns["drums"] = var_drums
        elif phrase_bar == 4:
            # Bar 5 energy lift (phrase 2 starts — open hats / rolling bass)
            lift_drums = self.bank.pick_pattern("drums", groove_type="four_on_floor_hats", preferred_track_id=self.lead_track_id, rng=self.rng)
            if lift_drums:
                self.active_patterns["drums"] = lift_drums
            lift_bass = self.bank.pick_pattern("bass", groove_type="rolling", preferred_track_id=self.lead_track_id, rng=self.rng)
            if lift_bass:
                self.active_patterns["bass"] = lift_bass
        else:
            # Bars 1, 2, 5, 6: Follow consecutive bars from the lead track for natural musical development!
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

        # Guarantee solid groove presence across all stems on every bar
        if self.active_patterns.get("drums") is None or len(self.active_patterns["drums"].onset_grid) < 4:
            self.active_patterns["drums"] = self.bank.pick_pattern(
                "drums", groove_type="four_on_floor_hats", preferred_track_id=self.lead_track_id, rng=self.rng
            ) or self.bank.create_groove_pattern("drums", "four_on_floor_hats", preferred_track_id=self.lead_track_id, rng=self.rng)

        if self.active_patterns.get("bass") is None or len(self.active_patterns["bass"].onset_grid) < 2:
            self.active_patterns["bass"] = self.bank.pick_pattern(
                "bass", groove_type="offbeat", preferred_track_id=self.lead_track_id, rng=self.rng
            ) or self.bank.create_groove_pattern("bass", "offbeat", preferred_track_id=self.lead_track_id, rng=self.rng)

        if self.active_patterns.get("other") is None or len(self.active_patterns["other"].onset_grid) < 2:
            self.active_patterns["other"] = self.bank.pick_pattern(
                "other", preferred_track_id=self.lead_track_id, rng=self.rng
            ) or self.bank.create_groove_pattern("other", "chords", preferred_track_id=self.lead_track_id, rng=self.rng)

        if self.active_patterns.get("vocals") is None:
            self.active_patterns["vocals"] = self.bank.pick_pattern(
                "vocals", preferred_track_id=self.lead_track_id, rng=self.rng
            ) or self.bank.create_groove_pattern("vocals", "chop", preferred_track_id=self.lead_track_id, rng=self.rng)

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
        p_bar = self.state.phrase_bar
        slices: Dict[str, Optional[SliceRef]] = {}

        for stem in STEMS:
            pat = self.active_patterns.get(stem)
            if pat is None or not pat.steps:
                slices[stem] = None
                continue

            # Section dynamics & arrangement sculpting:
            candidate_sref: Optional[SliceRef] = None
            if sec == "INTRO":
                # Intro groove: steady hats and synth hook with kicks on 0, 8
                if stem == "drums":
                    candidate_sref = pat.steps[step % len(pat.steps)] if step in (0, 4, 8, 12, 2, 6, 10, 14) else None
                elif stem == "bass":
                    candidate_sref = pat.steps[step % len(pat.steps)] if (p_bar >= 4 and step in (2, 6, 10, 14)) else None
                else:
                    candidate_sref = pat.steps[step % len(pat.steps)]

            elif sec == "BREAKDOWN":
                # Melodic breakdown: sustained pads/chords and vocals, with steady offbeat hats/snares
                if stem == "drums":
                    candidate_sref = pat.steps[step % len(pat.steps)] if step in (4, 12, 2, 6, 10, 14) else None
                elif stem == "bass":
                    candidate_sref = pat.steps[step % len(pat.steps)] if step in (0, 8) else None
                else:
                    candidate_sref = pat.steps[step % len(pat.steps)]

            elif sec == "BUILDUP":
                # Buildup snare rush & rising tension:
                if stem == "bass":
                    candidate_sref = pat.steps[step % len(pat.steps)] if (p_bar < 7 and step in (0, 4, 8, 12)) else None
                elif stem == "drums":
                    if p_bar in (0, 1, 2, 3):
                        candidate_sref = pat.steps[step % len(pat.steps)] if step % 4 == 0 else None
                    elif p_bar in (4, 5):
                        candidate_sref = pat.steps[step % len(pat.steps)] if step % 2 == 0 else None
                    elif p_bar == 6:
                        candidate_sref = pat.steps[step % len(pat.steps)] or pat.steps[4] or pat.steps[0]
                    elif p_bar == 7:
                        # Final bar: Roll on 0..14, Step 15 breath of silence before the DROP!
                        candidate_sref = None if step == 15 else (pat.steps[step % len(pat.steps)] or pat.steps[4] or pat.steps[0])
                    else:
                        candidate_sref = pat.steps[step % len(pat.steps)]
                else:
                    candidate_sref = pat.steps[step % len(pat.steps)]

            elif sec == "DROP":
                # Peak energy: full four-on-the-floor, rolling bass, main lead
                if self.state.is_fill_bar and stem == "bass" and step >= 12:
                    candidate_sref = None  # Brief turnaround cut
                else:
                    candidate_sref = pat.steps[step % len(pat.steps)]
            else:
                # Default GROOVE_A: full solid 16-step dance groove
                candidate_sref = pat.steps[step % len(pat.steps)]

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
                    slices[stem] = None  # Sustain existing slice naturally!

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
