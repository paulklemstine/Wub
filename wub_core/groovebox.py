"""Wub Groovebox Conductor: Clockwork Step Sequencer and Macro Arrangement Engine.

Controls song structure (Intro, Groove A, Buildup, Drop, Breakdown), selects
compatible 16-step sequencer patterns, handles 8-bar turnarounds and drum fills,
and maintains a rock-solid, jitter-free groove locked to the beat grid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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

# Cycle presets: dynamic multi-pattern cycle architectures
CYCLE_PRESETS = {
    "4x16": {"patterns": 4, "steps": 16, "label": "4 × 16"},
    "8x16": {"patterns": 8, "steps": 16, "label": "8 × 16"},
    "8x32": {"patterns": 8, "steps": 32, "label": "8 × 32"},
    "16x32": {"patterns": 16, "steps": 32, "label": "16 × 32"},
}
DEFAULT_CYCLE_MODE = "8x16"


@dataclass
class GrooveboxState:
    section: str = "GROOVE_A"
    bar_in_section: int = 0
    phrase_bar: int = 0      # 0..7 (Bar 1 to Bar 8 of an 8-bar phrase)
    step_in_bar: int = 0     # 0..15 (16th note step within the bar)
    pattern_in_cycle: int = 0  # 0..(patterns_per_cycle - 1)
    step_in_pattern: int = 0   # 0..(steps_per_pattern - 1)
    total_bars: int = 0
    total_steps: int = 0
    is_fill_bar: bool = False
    active_patterns: Dict[str, Optional[SequencerPattern]] = field(default_factory=dict)


class GrooveboxConductor:
    """Intelligent conductor that selects patterns and steps through the groovebox."""

    def __init__(self, pattern_bank: PatternBank, seed: int = 42, cross_stem_mashup: bool = True) -> None:
        self.bank = pattern_bank
        self.rng = np.random.default_rng(seed)
        self.arranger = SongArranger(pattern_bank, seed=seed)
        self.cycle_mode = DEFAULT_CYCLE_MODE
        self.patterns_per_cycle = CYCLE_PRESETS[self.cycle_mode]["patterns"]
        self.steps_per_pattern = CYCLE_PRESETS[self.cycle_mode]["steps"]
        self.cycle_total_steps = self.patterns_per_cycle * self.steps_per_pattern
        self.state = GrooveboxState()
        self.state.section = "GROOVE_A"
        self.section_bars_remaining = SECTION_DURATIONS.get(self.state.section, 16)
        self.active_patterns: Dict[str, Optional[SequencerPattern]] = {s: None for s in STEMS}
        self.current_phrase: Optional[SongPhrase] = None
        self.current_pattern: Optional[MultiStemPattern] = None
        self.lead_track_id: Optional[str] = None
        self.lead_base_bar: int = 0
        self.active_slice_ids: Dict[str, Optional[str]] = {s: None for s in STEMS}
        self.cross_stem_mashup = cross_stem_mashup
        self.feedback_data: Dict[str, Dict[str, int]] = {s: {} for s in STEMS}
        self.stem_weights: Dict[str, Dict[str, float]] = {s: {} for s in STEMS}
        self._load_feedback()
        self._init_section_patterns()
        if self.steps_per_pattern == 32:
            self._refresh_cycle_patterns()

    def set_cycle_mode(self, mode: str) -> bool:
        """Switch cycle preset dynamically (e.g. '4x16', '8x16', '8x32', '16x32')."""
        if mode not in CYCLE_PRESETS:
            return False
        self.cycle_mode = mode
        self.patterns_per_cycle = CYCLE_PRESETS[mode]["patterns"]
        self.steps_per_pattern = CYCLE_PRESETS[mode]["steps"]
        self.cycle_total_steps = self.patterns_per_cycle * self.steps_per_pattern
        self._refresh_cycle_patterns()
        return True

    def _refresh_cycle_patterns(self) -> None:
        """Ensure active stem patterns match the current step length (16 or 32)."""
        if self.steps_per_pattern == 32:
            for s in STEMS:
                cur = self.active_patterns.get(s)
                if cur is None or cur.length != 32:
                    p2 = self.bank.pick_pattern(s, preferred_track_id=self.lead_track_id, rng=self.rng)
                    cmp_pat = self.bank.create_compound_pattern(s, cur, p2, target_length=32, rng=self.rng)
                    if cmp_pat:
                        self.active_patterns[s] = cmp_pat
        elif self.steps_per_pattern == 16:
            for s in STEMS:
                cur = self.active_patterns.get(s)
                if cur is not None and cur.length == 32:
                    # Fall back to 16-step pattern
                    self.active_patterns[s] = self.bank.pick_pattern(s, preferred_track_id=self.lead_track_id, rng=self.rng)

        # Densify all active patterns to guarantee driving rhythm at all times
        for s in STEMS:
            if self.active_patterns.get(s):
                self.active_patterns[s] = self.bank.densify_pattern(self.active_patterns[s], rng=self.rng)

    def _load_feedback(self) -> None:
        p = Path("data/user_feedback.json")
        if p.exists():
            try:
                import json
                self.feedback_data = json.loads(p.read_text())
            except Exception:
                self.feedback_data = {s: {} for s in STEMS}
        else:
            self.feedback_data = {s: {} for s in STEMS}
        self._recompute_weights()

    def _save_feedback(self) -> None:
        p = Path("data/user_feedback.json")
        try:
            import json
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.feedback_data, indent=2))
        except Exception:
            pass

    def _recompute_weights(self) -> None:
        for s in STEMS:
            scores = self.feedback_data.get(s, {})
            weights = {}
            for tid, sc in scores.items():
                if sc > 0:
                    weights[tid] = min(6.0, 1.0 + 0.6 * sc)
                elif sc < 0:
                    weights[tid] = max(0.05, 1.0 / (1.0 + 0.9 * abs(sc)))
                else:
                    weights[tid] = 1.0
            self.stem_weights[s] = weights

    def record_feedback(
        self,
        stem: str,
        rating: int,
        track_id: Optional[str] = None,
        pattern_id: Optional[str] = None,
    ) -> dict:
        """Record user thumbs up/down, adapt selection weights, and trigger smart immediate response."""
        if stem not in STEMS:
            return {"status": "error", "message": f"invalid stem {stem}"}
        
        target_tid = track_id
        if not target_tid and self.active_patterns.get(stem):
            target_tid = self.active_patterns[stem].track_id

        if target_tid:
            cur = self.feedback_data.setdefault(stem, {}).get(target_tid, 0)
            self.feedback_data[stem][target_tid] = cur + int(rating)
            self._save_feedback()
            self._recompute_weights()

        action = "boosted" if rating > 0 else "rerolled"
        new_pat = None
        if rating < 0:
            # Smart immediate response on thumbs down: reroll this stem immediately!
            new_pat = self.reroll_stem(stem, exclude_track_id=target_tid)

        return {
            "status": "ok",
            "action": action,
            "stem": stem,
            "track_id": target_tid,
            "score": self.feedback_data.get(stem, {}).get(target_tid, 0) if target_tid else 0,
            "new_pattern": new_pat.pattern_name if new_pat else None,
        }

    def reroll_stem(self, stem: str, exclude_track_id: Optional[str] = None) -> Optional[SequencerPattern]:
        """Instantly switch a stem to a high-rated, musically compatible alternative across songs."""
        sec = self.state.section
        min_energy = 0.5 if sec == "DROP" else (0.35 if sec == "GROOVE_A" else 0.2)
        cur_key = self.current_phrase.key if self.current_phrase else (self.current_pattern.key if self.current_pattern else "C major")
        
        weights = dict(self.stem_weights.get(stem, {}))
        if exclude_track_id:
            weights[exclude_track_id] = 0.001

        if stem == "drums":
            g_type = "fill" if self.state.is_fill_bar else ("four_on_floor_hats" if sec in ("DROP", "BUILDUP") else "breakbeat")
            new_pat = self.bank.pick_pattern(
                "drums", groove_type=g_type, min_energy=min_energy,
                feedback_weights=weights, rng=self.rng
            ) or self.bank.create_groove_pattern("drums", "four_on_floor_hats", rng=self.rng)
        elif stem == "bass":
            g_type = "rolling" if sec == "DROP" else "offbeat"
            new_pat = self.bank.pick_pattern(
                "bass", groove_type=g_type, min_energy=min_energy,
                target_key=cur_key, feedback_weights=weights, rng=self.rng
            ) or self.bank.create_groove_pattern("bass", "offbeat", rng=self.rng)
        elif stem == "vocals":
            new_pat = self.bank.pick_pattern(
                "vocals", target_key=cur_key, min_energy=0.0,
                feedback_weights=weights, rng=self.rng
            ) or self.bank.create_groove_pattern("vocals", "chop", rng=self.rng)
        else: # other
            g_type = "riff" if sec == "DROP" else ("pad" if sec == "BREAKDOWN" else "hook")
            new_pat = self.bank.pick_pattern(
                "other", groove_type=g_type, min_energy=min_energy,
                target_key=cur_key, feedback_weights=weights, rng=self.rng
            ) or self.bank.create_groove_pattern("other", "chords", rng=self.rng)

        if new_pat:
            self.active_patterns[stem] = self.bank.densify_pattern(new_pat, rng=self.rng)
            self.active_slice_ids[stem] = None
        return self.active_patterns.get(stem)

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
                pat = self.bank.get_track_pattern(
                    self.current_pattern.track_id, s, self.current_pattern.bar_num
                ) or self.bank.pick_pattern(s, preferred_track_id=self.lead_track_id, rng=self.rng)
                if pat:
                    self.active_patterns[s] = self.bank.densify_pattern(pat, rng=self.rng)
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

        # Infill and densify all fallback section patterns
        for s in STEMS:
            if self.active_patterns.get(s):
                self.active_patterns[s] = self.bank.densify_pattern(self.active_patterns[s], rng=self.rng)

    def _on_pattern_start(self, pattern_in_cycle: int) -> None:
        """Triggered on Step 0 of each pattern in the cycle."""
        is_turnaround = (pattern_in_cycle == self.patterns_per_cycle - 1)
        self.state.is_fill_bar = is_turnaround
        self.state.pattern_in_cycle = pattern_in_cycle
        self.state.phrase_bar = pattern_in_cycle % 8

        # Section transition check on cycle restart
        if pattern_in_cycle == 0 and self.section_bars_remaining <= 0:
            next_sec = NEXT_SECTION.get(self.state.section, "GROOVE_A")
            self.state.section = next_sec
            self.state.bar_in_section = 0
            self.section_bars_remaining = SECTION_DURATIONS.get(next_sec, 16)
            self._init_section_patterns()
            if self.steps_per_pattern == 32:
                self._refresh_cycle_patterns()
            return

        # 32-step Compound Pattern Generation
        if self.steps_per_pattern == 32:
            if is_turnaround:
                base_pat = (
                    self.bank.pick_pattern("drums", groove_type="four_on_floor_hats", preferred_track_id=self.lead_track_id, rng=self.rng)
                    or self.bank.create_groove_pattern("drums", "four_on_floor_hats", rng=self.rng)
                )
                fill_pat = (
                    self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
                    or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
                )
                self.active_patterns["drums"] = self.bank.create_compound_pattern("drums", base_pat, fill_pat, target_length=32, rng=self.rng)
                for s in ("bass", "vocals", "other"):
                    p1 = self.active_patterns.get(s)
                    p2 = self.bank.pick_pattern(s, preferred_track_id=self.lead_track_id, rng=self.rng)
                    self.active_patterns[s] = self.bank.create_compound_pattern(s, p1, p2, target_length=32, rng=self.rng)
            else:
                bar_offset = (pattern_in_cycle * 2) % 8
                for s in STEMS:
                    p1 = self.bank.get_track_pattern(self.lead_track_id, s, self.lead_base_bar + bar_offset) if self.lead_track_id else None
                    p2 = self.bank.get_track_pattern(self.lead_track_id, s, self.lead_base_bar + bar_offset + 1) if self.lead_track_id else None
                    if not p1:
                        p1 = self.bank.pick_pattern(s, preferred_track_id=self.lead_track_id, rng=self.rng)
                    cmp_pat = self.bank.create_compound_pattern(s, p1, p2, target_length=32, rng=self.rng)
                    if cmp_pat:
                        self.active_patterns[s] = cmp_pat
            return

        # 16-step Standard Pattern Progression
        phrase_bar = pattern_in_cycle % 8

        # Cohesive Multi-Stem Phrase progression
        if self.current_phrase and phrase_bar < len(self.current_phrase.bars):
            if pattern_in_cycle == 0 and self.state.total_steps > 0:
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
                pat = self.bank.get_track_pattern(
                    self.current_pattern.track_id, s, self.current_pattern.bar_num
                ) or self.active_patterns.get(s)
                if pat:
                    self.active_patterns[s] = self.bank.densify_pattern(pat, rng=self.rng)

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

        # Fallback single-stem mode (used in tests with small synthetic tracks or dynamic mashups)
        if is_turnaround:
            fill_pat = (
                self.bank.pick_pattern("drums", groove_type="fill", min_energy=0.5, rng=self.rng)
                or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
            )
            if fill_pat:
                self.active_patterns["drums"] = fill_pat
        elif pattern_in_cycle == 0:
            self._init_section_patterns()
        elif pattern_in_cycle % 4 == 3 and self.rng.random() < 0.6:
            var_drums = self.bank.pick_pattern("drums", min_energy=0.4, preferred_track_id=self.lead_track_id, rng=self.rng)
            if var_drums:
                self.active_patterns["drums"] = var_drums
        elif pattern_in_cycle % 4 == 2:
            lift_drums = self.bank.pick_pattern("drums", groove_type="four_on_floor_hats", preferred_track_id=self.lead_track_id, rng=self.rng)
            if lift_drums:
                self.active_patterns["drums"] = lift_drums
            lift_bass = self.bank.pick_pattern("bass", groove_type="rolling", preferred_track_id=self.lead_track_id, rng=self.rng)
            if lift_bass:
                self.active_patterns["bass"] = lift_bass
        else:
            if self.lead_track_id:
                track_bar = self.lead_base_bar + (pattern_in_cycle % 8)
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

        # Infill and densify all active patterns to guarantee full driving rhythm
        for s in STEMS:
            if self.active_patterns.get(s):
                self.active_patterns[s] = self.bank.densify_pattern(self.active_patterns[s], rng=self.rng)

    def _on_bar_start(self, pattern_in_cycle: int, step_in_pattern: int) -> None:
        """Triggered on Step 0 of each bar (every 16 tatums)."""
        self.state.bar_in_section += 1
        self.section_bars_remaining -= 1
        self.active_slice_ids = {s: None for s in STEMS}

        if step_in_pattern == 0:
            self._on_pattern_start(pattern_in_cycle)

    def next_step(self) -> Tuple[Dict[str, Optional[SliceRef]], dict]:
        """Advance the groovebox sequencer by 1 tatum step (1/16th note)."""
        step_in_pattern = self.state.total_steps % self.steps_per_pattern
        pattern_in_cycle = (self.state.total_steps // self.steps_per_pattern) % self.patterns_per_cycle
        step_in_bar = self.state.total_steps % 16

        self.state.step_in_bar = step_in_bar
        self.state.step_in_pattern = step_in_pattern
        self.state.pattern_in_cycle = pattern_in_cycle
        self.state.phrase_bar = pattern_in_cycle % 8

        if step_in_bar == 0:
            self._on_bar_start(pattern_in_cycle, step_in_pattern)

        sec = self.state.section
        slices: Dict[str, Optional[SliceRef]] = {}

        for stem in STEMS:
            pat = self.active_patterns.get(stem)
            if pat and pat.steps:
                candidate_sref = pat.steps[step_in_pattern % len(pat.steps)]
            elif self.current_pattern and stem in self.current_pattern.stems:
                stem_steps = self.current_pattern.stems[stem]
                candidate_sref = stem_steps[step_in_bar % len(stem_steps)]
            else:
                candidate_sref = None

            # Step Sequencer trigger: every programmed slice step strikes cleanly
            slices[stem] = candidate_sref
            if candidate_sref is not None:
                self.active_slice_ids[stem] = candidate_sref.segment_id
            else:
                self.active_slice_ids[stem] = None

        sec_bars_elapsed = self.state.bar_in_section
        sec_total_bars = SECTION_DURATIONS.get(sec, 16)
        sec_bars_left = max(0, self.section_bars_remaining)

        cur_key = self.current_phrase.key if self.current_phrase else (self.current_pattern.key if self.current_pattern else "C major")
        from wub_core.composer import KEY_TO_CAMELOT
        camelot_tuple = KEY_TO_CAMELOT.get(cur_key, (8, "B"))
        camelot_str = f"{camelot_tuple[0]}{camelot_tuple[1]} ({cur_key})"

        sec_idx = SECTIONS.index(sec) if sec in SECTIONS else 0
        next_sec = SECTIONS[(sec_idx + 1) % len(SECTIONS)]
        target_energy = 0.9 if sec == "DROP" else (0.75 if sec == "BUILDUP" else (0.6 if sec == "GROOVE_A" else 0.4))

        pattern_titles = {}
        pattern_grooves = {}
        pattern_grids = {}
        pattern_track_ids = {}
        pattern_ids = {}
        pattern_ratings = {}
        pattern_audio_scores = {}

        for s in STEMS:
            pat = self.active_patterns.get(s)
            if pat:
                pattern_titles[s] = pat.pattern_name[:28]
                pattern_grooves[s] = pat.groove_type
                pattern_grids[s] = [
                    1 if step_i < len(pat.steps) and pat.steps[step_i] is not None else 0
                    for step_i in range(self.steps_per_pattern)
                ]
                pattern_track_ids[s] = pat.track_id
                pattern_ids[s] = pat.id
                pattern_ratings[s] = self.feedback_data.get(s, {}).get(pat.track_id, 0)
                pattern_audio_scores[s] = round(float(getattr(pat, "audio_similarity_score", 0.5)), 3)
            elif self.current_pattern and s in self.current_pattern.stems:
                pattern_titles[s] = f"{self.current_pattern.track_title[:24]} Bar {self.current_pattern.bar_num}"
                pattern_grooves[s] = self.current_pattern.groove_types.get(s, "groove")
                stem_steps = self.current_pattern.stems[s]
                pattern_grids[s] = [
                    1 if step_i < len(stem_steps) and stem_steps[step_i] is not None else 0
                    for step_i in range(self.steps_per_pattern)
                ]
                pattern_track_ids[s] = self.current_pattern.track_id
                pattern_ids[s] = f"pat_{s}_{self.current_pattern.track_id[:6]}_{self.current_pattern.bar_num:03d}"
                pattern_ratings[s] = self.feedback_data.get(s, {}).get(self.current_pattern.track_id, 0)
                pattern_audio_scores[s] = round(float(getattr(self.current_pattern, "audio_similarity_score", 0.5)), 3)
            else:
                pattern_titles[s] = "Rest"
                pattern_grooves[s] = "none"
                pattern_grids[s] = [0] * self.steps_per_pattern
                pattern_track_ids[s] = ""
                pattern_ids[s] = ""
                pattern_ratings[s] = 0
                pattern_audio_scores[s] = 0.5

        meta = {
            "section": sec,
            "phrase_bar": (pattern_in_cycle % 8) + 1,
            "step_in_bar": step_in_bar + 1,
            "total_bars": self.state.total_bars,
            "is_fill": self.state.is_fill_bar,
            "key": camelot_str,
            "energy": target_energy,
            "section_bars_elapsed": sec_bars_elapsed + 1,
            "section_total_bars": sec_total_bars,
            "next_section": next_sec,
            "section_bars_left": sec_bars_left,
            "cycle_mode": self.cycle_mode,
            "pattern_in_cycle": pattern_in_cycle + 1,
            "patterns_per_cycle": self.patterns_per_cycle,
            "step_in_pattern": step_in_pattern + 1,
            "steps_per_pattern": self.steps_per_pattern,
            "cycle_total_steps": self.cycle_total_steps,
            "cycle_progress": round((pattern_in_cycle * self.steps_per_pattern + step_in_pattern) / float(self.cycle_total_steps), 3),
            "patterns": pattern_titles,
            "pattern_grooves": pattern_grooves,
            "pattern_grids": pattern_grids,
            "pattern_track_ids": pattern_track_ids,
            "pattern_ids": pattern_ids,
            "pattern_ratings": pattern_ratings,
            "pattern_audio_scores": pattern_audio_scores,
        }

        # Advance step counter
        self.state.total_steps += 1
        if self.state.total_steps % 16 == 0:
            self.state.total_bars += 1

        return slices, meta
