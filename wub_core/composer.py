"""Wub Tier 2: Phrase Pattern Composer.

Learns and models bar-to-bar pattern progressions across 4-bar and 8-bar
musical phrases. Bridges the Tier 3 Macro Song Conductor (which plans sections
and energy envelopes) and the Tier 1 Micro Pattern Generator (which provides
16-step rhythmic grid slices).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.patterns import (
    MultiStemPattern,
    PatternBank,
    SequencerPattern,
    SongPhrase,
    classify_drum_groove,
    classify_bass_groove,
    classify_synth_groove,
    classify_vocal_groove,
)
from wub_core.schema import STEMS, TrackAnalysis

ROOT_DIR = Path(__file__).resolve().parents[1]
COMPOSING_WEIGHTS_FILE = ROOT_DIR / "data" / "composing_ml_weights.json"


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

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        # Transitions: (phrase_bar, stem, current_groove) -> {next_groove: count}
        self.transitions: Dict[Tuple[int, str, str], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Direct pattern-to-next-pattern continuation learned from consecutive bars of songs
        self.pattern_continuations: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Direct phrase-to-next-phrase continuation learned from consecutive 8-bar phrases of songs
        self.phrase_continuations: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Empirical energy distribution per phrase bar: phrase_bar -> list of energies
        self.energy_by_bar: Dict[int, List[float]] = defaultdict(list)
        self.trained = False
        self.weights_path = weights_path or COMPOSING_WEIGHTS_FILE
        self.load(self.weights_path)

    def train_on_tracks(self, tracks: List[TrackAnalysis], bank: PatternBank) -> int:
        """Train bar-to-bar transition probabilities and consecutive pattern/phrase continuations from songs."""
        n_transitions = 0

        for tr in tracks:
            if not tr.stems_available:
                continue

            n_tatums = len(tr.tatums)
            n_bars = n_tatums // 16
            if n_bars < 4:
                continue

            for bar in range(n_bars - 1):
                p_bar_curr = bar % 8

                for stem in STEMS:
                    pat_curr = bank.get_track_pattern(tr.track_id, stem, bar)
                    pat_next = bank.get_track_pattern(tr.track_id, stem, bar + 1)

                    if pat_curr and pat_next:
                        key = (p_bar_curr, stem, pat_curr.groove_type)
                        # Weight transitions by the acoustic similarity of the destination pattern
                        sim_score = getattr(pat_next, "audio_similarity_score", 0.5)
                        w = float(np.exp(2.0 * (sim_score - 0.5)))
                        self.transitions[key][pat_next.groove_type] += w
                        self.pattern_continuations[pat_curr.id][pat_next.id] += w * 10.0
                        self.energy_by_bar[p_bar_curr].append(pat_curr.energy)
                        n_transitions += 1
                    else:
                        # Fallback to multi-stem pattern bar index (covers full-catalog tracks with 1..4 stems)
                        msp_curr = bank.multistem_by_track_bar.get((tr.track_id, bar))
                        msp_next = bank.multistem_by_track_bar.get((tr.track_id, bar + 1))
                        if msp_curr and msp_next:
                            g_curr = msp_curr.groove_types.get(stem)
                            g_next = msp_next.groove_types.get(stem)
                            if g_curr and g_next and g_curr != "none" and g_next != "none":
                                key = (p_bar_curr, stem, g_curr)
                                self.transitions[key][g_next] += 1.0
                                self.energy_by_bar[p_bar_curr].append(msp_curr.energy)
                                n_transitions += 1

            # Also learn consecutive phrase-to-phrase continuations within the same song
            track_phrases = bank.phrases_by_track.get(tr.track_id, [])
            if len(track_phrases) >= 2:
                sorted_phrases = sorted(track_phrases, key=lambda p: p.phrase_index)
                for i in range(len(sorted_phrases) - 1):
                    p_curr = sorted_phrases[i]
                    p_next = sorted_phrases[i + 1]
                    self.phrase_continuations[p_curr.id][p_next.id] += 15.0

        self.trained = True
        return n_transitions

    def predict_next_pattern_id(
        self,
        current_pattern_id: str,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[str]:
        """Predict the next consecutive pattern ID learned from the song's arrangement."""
        cont = self.pattern_continuations.get(current_pattern_id)
        if not cont:
            return None
        if rng is None:
            return max(cont.items(), key=lambda kv: kv[1])[0]
        choices = list(cont.keys())
        total = sum(cont.values())
        if total <= 0:
            return choices[0]
        probs = [v / total for v in cont.values()]
        return str(rng.choice(choices, p=probs))

    def predict_next_phrase_id(
        self,
        current_phrase_id: str,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[str]:
        """Predict the next consecutive phrase ID learned from the song's macro structure."""
        cont = self.phrase_continuations.get(current_phrase_id)
        if not cont:
            return None
        if rng is None:
            return max(cont.items(), key=lambda kv: kv[1])[0]
        choices = list(cont.keys())
        total = sum(cont.values())
        if total <= 0:
            return choices[0]
        probs = [v / total for v in cont.values()]
        return str(rng.choice(choices, p=probs))

    def save(self, path: Optional[Path] = None) -> None:
        p = path or self.weights_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if p.exists():
                try:
                    existing = json.loads(p.read_text())
                except Exception:
                    pass
            serializable_transitions = {
                f"{k[0]}:{k[1]}:{k[2]}": dict(v) for k, v in self.transitions.items()
            }
            serializable_pat_cont = {
                k: {nk: round(float(nv), 3) for nk, nv in v.items()}
                for k, v in list(self.pattern_continuations.items())[:2500]
            }
            serializable_phr_cont = {
                k: {nk: round(float(nv), 3) for nk, nv in v.items()}
                for k, v in self.phrase_continuations.items()
            }
            avg_energy = {
                str(k): round(float(np.mean(v)), 3) for k, v in self.energy_by_bar.items() if v
            }
            existing["version"] = 2
            existing["phrase_transitions"] = serializable_transitions
            existing["pattern_continuations"] = serializable_pat_cont
            existing["phrase_continuations"] = serializable_phr_cont
            existing["energy_by_bar"] = avg_energy
            p.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    def load(self, path: Optional[Path] = None) -> bool:
        p = path or self.weights_path
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            raw_t = data.get("phrase_transitions") or data.get("transitions", {})
            for key_str, counts in raw_t.items():
                parts = key_str.split(":")
                if len(parts) == 3:
                    bar_idx = int(parts[0])
                    stem = parts[1]
                    groove = parts[2]
                    self.transitions[(bar_idx, stem, groove)].update({k: float(v) for k, v in counts.items()})
            raw_pc = data.get("pattern_continuations", {})
            for pid, nxt_map in raw_pc.items():
                if isinstance(nxt_map, dict):
                    self.pattern_continuations[pid].update({k: float(v) for k, v in nxt_map.items()})
            raw_phc = data.get("phrase_continuations", {})
            for phid, nxt_map in raw_phc.items():
                if isinstance(nxt_map, dict):
                    self.phrase_continuations[phid].update({k: float(v) for k, v in nxt_map.items()})
            self.trained = len(self.transitions) > 0
            return self.trained
        except Exception:
            return False

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
            elif stem == "vocals":
                return "chop" if phrase_bar in (1, 3, 5, 7) else "phrase"
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
        target_key: Optional[str] = None,
        feedback_weights: Optional[Dict[str, float]] = None,
        densify: bool = False,
    ) -> Dict[str, Optional[SequencerPattern]]:
        """Compose the 16-step patterns for all stems for the upcoming bar with harmonic and groove ML alignment."""
        if lead_track_id:
            self.lead_track_id = lead_track_id

        is_turnaround = (phrase_bar == 7)  # Bar 8 of 8-bar phrase

        # 1. Determine target groove types per stem using Tier 2 Phrase Model ML
        target_grooves: Dict[str, str] = {}
        for stem in ("drums", "bass", "other", "vocals"):
            curr_g = getattr(self.current_bar_state, f"{stem}_type", "four_on_floor" if stem == "drums" else "offbeat")
            target_grooves[stem] = self.model.sample_next_groove(
                phrase_bar, stem, curr_g, rng=self.rng
            )

        # 2. Select patterns from Tier 1 PatternBank with Song Continuation Priority
        new_patterns: Dict[str, Optional[SequencerPattern]] = {}

        if self.lead_track_id and phrase_bar == 0 and any(self.active_patterns.values()):
            # Advance lead_base_bar to the next consecutive 8-bar block of the current song!
            self.lead_base_bar += 8

        if is_turnaround:
            # Bar 8: Prefer the current song's consecutive bar 7 pattern or a same-track fill!
            track_bar = self.lead_base_bar + phrase_bar
            fill_pat = (
                (self.bank.get_track_pattern(self.lead_track_id, "drums", track_bar) if self.lead_track_id else None)
                or self.bank.pick_pattern(
                    "drums", groove_type="fill", min_energy=0.5,
                    preferred_track_id=self.lead_track_id,
                    feedback_weights=feedback_weights, densify=densify, rng=self.rng,
                )
                or self.bank.create_groove_pattern("drums", "fill", preferred_track_id=self.lead_track_id, rng=self.rng)
            )
            new_patterns["drums"] = self.bank.densify_pattern(fill_pat, rng=self.rng) if densify and fill_pat else fill_pat
            for s in ("bass", "other", "vocals"):
                tr_pat = self.bank.get_track_pattern(self.lead_track_id, s, track_bar) if self.lead_track_id else None
                if tr_pat:
                    new_patterns[s] = self.bank.densify_pattern(tr_pat, rng=self.rng) if densify else tr_pat
                elif s == "bass":
                    new_patterns["bass"] = self.bank.pick_pattern(
                        "bass", groove_type="stabs", preferred_track_id=self.lead_track_id,
                        target_key=target_key, feedback_weights=feedback_weights, densify=densify, rng=self.rng,
                    )
                elif s == "other":
                    new_patterns["other"] = self.bank.pick_pattern(
                        "other", preferred_track_id=self.lead_track_id,
                        target_key=target_key, feedback_weights=feedback_weights, densify=densify, rng=self.rng,
                    )
                else:
                    new_patterns["vocals"] = self.bank.create_groove_pattern(
                        "vocals", "chop", preferred_track_id=self.lead_track_id, rng=self.rng
                    )

        elif self.lead_track_id:
            # Follow natural consecutive bar progression of the current song across ALL bars (including bars 0 and 4)!
            track_bar = self.lead_base_bar + phrase_bar
            for stem in STEMS:
                tr_pat = self.bank.get_track_pattern(self.lead_track_id, stem, track_bar)
                if not tr_pat and self.active_patterns.get(stem):
                    # Try trained pattern_continuations for the active pattern
                    next_pid = self.model.predict_next_pattern_id(self.active_patterns[stem].id, rng=self.rng)
                    if next_pid:
                        for cand in self.bank.patterns_by_stem.get(stem, []):
                            if cand.id == next_pid:
                                tr_pat = cand
                                break
                if tr_pat:
                    new_patterns[stem] = self.bank.densify_pattern(tr_pat, rng=self.rng) if densify else tr_pat
                else:
                    k_target = target_key if stem in ("bass", "other", "vocals") else None
                    new_patterns[stem] = self.bank.pick_pattern(
                        stem, groove_type=target_grooves.get(stem),
                        preferred_track_id=self.lead_track_id,
                        target_key=k_target, feedback_weights=feedback_weights,
                        densify=densify, rng=self.rng,
                    )

        else:
            # Initial Bar 1 (when no lead_track_id is active yet): Pick solid thematic anchor
            for stem in ("drums", "bass", "other"):
                k_target = target_key if stem in ("bass", "other") else None
                new_patterns[stem] = self.bank.pick_pattern(
                    stem,
                    groove_type=target_grooves.get(stem),
                    min_energy=target_energy * 0.7,
                    preferred_track_id=self.lead_track_id,
                    target_key=k_target,
                    feedback_weights=feedback_weights,
                    densify=densify,
                    rng=self.rng,
                )
            new_patterns["vocals"] = self.bank.pick_pattern(
                "vocals", target_key=target_key, preferred_track_id=self.lead_track_id,
                feedback_weights=feedback_weights, densify=densify, rng=self.rng,
            )
            if new_patterns.get("drums"):
                self.lead_track_id = new_patterns["drums"].track_id
                self.lead_base_bar = new_patterns["drums"].bar_num

        # Infill any missing stems using physical acoustic groove phrase evolution
        missing_stems = [s for s in STEMS if new_patterns.get(s) is None]
        if missing_stems:
            try:
                from wub_core.groovy_pattern_assembler import assemble_evolving_phrase
                evolved = assemble_evolving_phrase(
                    phrase_bar=phrase_bar,
                    palette=self.bank.slice_palette,
                    target_key=target_key,
                    preferred_track_id=self.lead_track_id,
                    feedback_weights=feedback_weights,
                    section=section,
                    rng=self.rng,
                )
                for s in missing_stems:
                    if evolved.get(s):
                        new_patterns[s] = evolved[s]
            except Exception:
                pass

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


KEY_TO_CAMELOT: Dict[str, Tuple[int, str]] = {
    "Ab minor": (1, "A"), "G# minor": (1, "A"), "B major": (1, "B"),
    "Eb minor": (2, "A"), "D# minor": (2, "A"), "F# major": (2, "B"),
    "Bb minor": (3, "A"), "A# minor": (3, "A"), "Db major": (3, "B"), "C# major": (3, "B"),
    "F minor": (4, "A"), "Ab major": (4, "B"), "G# major": (4, "B"),
    "C minor": (5, "A"), "Eb major": (5, "B"), "D# major": (5, "B"),
    "G minor": (6, "A"), "Bb major": (6, "B"), "A# major": (6, "B"),
    "D minor": (7, "A"), "F major": (7, "B"),
    "A minor": (8, "A"), "C major": (8, "B"),
    "E minor": (9, "A"), "G major": (9, "B"),
    "B minor": (10, "A"), "D major": (10, "B"),
    "F# minor": (11, "A"), "A major": (11, "B"),
    "Db minor": (12, "A"), "C# minor": (12, "A"), "E major": (12, "B"),
}


def camelot_distance(k1: str, k2: str) -> float:
    """Calculate distance on the Camelot wheel (0.0 = exact match/relative, 1.0 = adjacent fifth)."""
    c1 = KEY_TO_CAMELOT.get(k1)
    c2 = KEY_TO_CAMELOT.get(k2)
    if not c1 or not c2:
        return 1.5
    num1, mode1 = c1
    num2, mode2 = c2
    diff = abs(num1 - num2)
    circle_dist = min(diff, 12 - diff)
    mode_penalty = 0.0 if mode1 == mode2 else 0.5
    return float(circle_dist + mode_penalty)


class SongArranger:
    """Tier 3 ML Song Arranger: Plans macro-sections and harmonic phrase assemblies.
    
    Trained on dance music song structures, energy envelopes, and harmonic mixing
    rules to assemble multi-phrase tracks that flow with natural tension and release.
    """

    def __init__(self, bank: PatternBank, seed: int = 42, weights_path: Optional[Path] = None) -> None:
        self.bank = bank
        self.rng = np.random.default_rng(seed)
        self.section_transition_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.section_energy_targets: Dict[str, float] = {
            "INTRO": 0.25,
            "GROOVE_A": 0.45,
            "BUILDUP": 0.65,
            "DROP": 0.85,
            "BREAKDOWN": 0.20,
        }
        self.trained = False
        self.weights_path = weights_path or COMPOSING_WEIGHTS_FILE
        self.load(self.weights_path)

    def train_on_corpus(self, tracks: List[TrackAnalysis]) -> int:
        """Learn section-to-section transitions, phrase durations, and energy profiles from corpus."""
        n_transitions = 0
        section_energies: Dict[str, List[float]] = defaultdict(list)
        self.section_transition_counts = defaultdict(lambda: defaultdict(float))

        # Seed authentic macro-section transition priors so every section has rich, variety-packed flow paths
        priors = {
            "INTRO": {"GROOVE_A": 6.5, "BUILDUP": 3.5},
            "GROOVE_A": {"BUILDUP": 5.0, "BREAKDOWN": 3.0, "DROP": 2.0},
            "BUILDUP": {"DROP": 10.0},
            "DROP": {"BREAKDOWN": 5.0, "GROOVE_A": 3.5, "BUILDUP": 1.5},
            "BREAKDOWN": {"BUILDUP": 7.0, "GROOVE_A": 3.0},
        }
        for sec, trans in priors.items():
            for target_sec, weight in trans.items():
                self.section_transition_counts[sec][target_sec] += weight

        for tr in tracks:
            phrases = self.bank.phrases_by_track.get(tr.track_id, [])
            if len(phrases) < 2:
                continue
            for i in range(len(phrases) - 1):
                s_curr = phrases[i].section
                s_next = phrases[i + 1].section
                section_energies[s_curr].append(phrases[i].energy)
                # Count inter-section transitions to learn authentic macro song progression
                if s_curr != s_next:
                    w = 1.0 + float(phrases[i + 1].energy)
                    self.section_transition_counts[s_curr][s_next] += w
                    n_transitions += 1

        for sec, energies in section_energies.items():
            if energies:
                self.section_energy_targets[sec] = float(np.mean(energies))

        self.trained = True
        return n_transitions

    def decide_next_section(
        self,
        current_section: str,
        bars_in_section: int,
        total_bars: int,
    ) -> str:
        """Determine whether to stay in current section or transition to next using trained transitions."""
        if bars_in_section < 8:
            return current_section

        # If trained transitions exist and section elapsed >= 8 bars, sample from learned distribution
        if self.trained and current_section in self.section_transition_counts:
            transition_prob = 0.50 if bars_in_section == 8 else (0.85 if bars_in_section == 16 else 1.0)
            if self.rng.random() < transition_prob:
                trans = self.section_transition_counts[current_section]
                total = sum(trans.values())
                if total > 0:
                    choices = list(trans.keys())
                    probs = [v / total for v in trans.values()]
                    sampled = str(self.rng.choice(choices, p=probs))
                    if sampled != current_section or bars_in_section >= 24:
                        return sampled

        if current_section == "INTRO":
            if bars_in_section >= 8:
                return "GROOVE_A"
            return "INTRO"

        elif current_section == "GROOVE_A":
            if bars_in_section >= 16:
                return "BUILDUP"
            elif bars_in_section >= 8:
                return "BUILDUP" if self.rng.random() < 0.45 else "GROOVE_A"
            return "GROOVE_A"

        elif current_section == "BUILDUP":
            if bars_in_section >= 8:
                return "DROP"
            return "BUILDUP"

        elif current_section == "DROP":
            if bars_in_section >= 16:
                return "BREAKDOWN"
            elif bars_in_section >= 8:
                return "BREAKDOWN" if self.rng.random() < 0.35 else "DROP"
            return "DROP"

        elif current_section == "BREAKDOWN":
            if bars_in_section >= 8:
                return "BUILDUP"
            return "BREAKDOWN"

        return "GROOVE_A"

    def plan_next_phrase(
        self,
        current_phrase: Optional[SongPhrase],
        current_section: str,
        current_key: str,
        current_bpm: float,
        bars_in_section: int,
        total_bars: int,
    ) -> Tuple[SongPhrase, str, bool]:
        """Select the next optimal 8-bar phrase and macro section using harmonic compatibility and real-audio quality scoring."""
        next_section = self.decide_next_section(current_section, bars_in_section, total_bars)
        is_sec_change = (next_section != current_section)

        candidates = list(self.bank.phrases)
        if not candidates:
            raise ValueError("No phrases available in PatternBank")

        target_energy = self.section_energy_targets.get(next_section, 0.5)

        # Compute weights for each candidate phrase
        scores = []
        for p in candidates:
            # 1. Section Compatibility
            if p.section == next_section:
                sec_score = 1.0
            elif next_section == "BUILDUP" and (p.has_fill or p.section == "GROOVE_A"):
                sec_score = 0.8
            elif next_section == "DROP" and p.energy >= 0.4:
                sec_score = 0.7
            elif next_section == "INTRO" and p.section in ("GROOVE_A", "BREAKDOWN"):
                sec_score = 0.6
            elif next_section == "BREAKDOWN" and p.energy < 0.3:
                sec_score = 0.7
            else:
                sec_score = 0.15

            # 2. Harmonic Key Compatibility
            k_dist = camelot_distance(current_key, p.key)
            key_score = float(np.exp(-1.4 * k_dist))

            # 3. BPM Compatibility
            bpm_diff = abs(p.bpm - current_bpm) / max(1.0, current_bpm)
            bpm_score = float(np.exp(-25.0 * (bpm_diff ** 2)))

            # 4. Energy Profile Match
            energy_diff = abs(p.energy - target_energy)
            energy_score = float(np.exp(-3.0 * (energy_diff ** 2)))

            # 5. Acoustic Similarity to Real Audio Training Data
            sim_scores = [getattr(b, "audio_similarity_score", 0.5) for b in p.bars] if p.bars else [0.5]
            avg_sim = float(np.mean(sim_scores))
            audio_score = float(np.exp(2.5 * (avg_sim - 0.5)))

            # 6. Track Narrative Continuity (Strongly prefer playing the next consecutive part of the active song!)
            narrative_score = 1.0
            if current_phrase is not None and p.track_id == current_phrase.track_id:
                if p.phrase_index == current_phrase.phrase_index + 1:
                    narrative_score = 250.0 if not is_sec_change else 120.0
                elif p.phrase_index > current_phrase.phrase_index:
                    narrative_score = 35.0
                elif p.phrase_index == current_phrase.phrase_index:
                    narrative_score = 2.0
                else:
                    narrative_score = 8.0

            total_score = sec_score * key_score * bpm_score * energy_score * audio_score * narrative_score
            scores.append(max(1e-6, total_score))

        scores_arr = np.array(scores, dtype=np.float64)
        probs = scores_arr / np.sum(scores_arr)

        chosen_idx = int(self.rng.choice(len(candidates), p=probs))
        chosen_phrase = candidates[chosen_idx]
        return chosen_phrase, next_section, is_sec_change

    def save(self, path: Optional[Path] = None) -> None:
        p = path or self.weights_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            serializable_sec = {
                k: dict(v) for k, v in self.section_transition_counts.items()
            }
            existing = {}
            if p.exists():
                try:
                    existing = json.loads(p.read_text())
                except Exception:
                    pass
            existing["section_transitions"] = serializable_sec
            existing["section_energy_targets"] = self.section_energy_targets
            p.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    def load(self, path: Optional[Path] = None) -> bool:
        p = path or self.weights_path
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            raw_sec = data.get("section_transitions", {})
            for sec, trans in raw_sec.items():
                self.section_transition_counts[sec].update({k: float(v) for k, v in trans.items() if k != sec})
            targets = data.get("section_energy_targets", {})
            for sec, target in targets.items():
                self.section_energy_targets[sec] = float(target)
            self.trained = len(self.section_transition_counts) > 0
            return self.trained
        except Exception:
            return False
