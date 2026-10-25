"""Groovy Pattern Assembler and Acoustic Slice Classifier for Wub KG-505.

Provides:
1. AcousticSliceClassifier: Physical classification of audio slices by spectral centroid,
   loudness, duration, and chroma polyphony.
2. Harmonic Consonance: Key alignment and dissonant-interval filtering against Camelot scales.
3. Multi-Bar Groove Templates: Rhythmic patterns with velocity curves and micro-accents.
4. Evolving Phrase Generator: 4-bar and 8-bar musical phrase arcs with phrase-level groove evolution.
5. Exact Right Audio Slice Selector: Multi-objective candidate ranking for pristine dance groove assembly.
6. Sidechain Ducking & Envelope Shaping: Eliminates bass mud and ensures crisp transient punch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.patterns import SequencerPattern, MultiStemPattern, STEMS
from wub_core.composer import KEY_TO_CAMELOT, camelot_distance

# ==============================================================================
# 1. Harmonic Pitch Profiles & Consonance Filtering
# ==============================================================================

# Semitone pitch class names: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
ROOT_TO_INDEX: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Camelot code to Root Note & Mode
CAMELOT_TO_KEY: Dict[str, Tuple[str, str]] = {
    "1A": ("Ab", "minor"), "1B": ("B", "major"),
    "2A": ("Eb", "minor"), "2B": ("F#", "major"),
    "3A": ("Bb", "minor"), "3B": ("Db", "major"),
    "4A": ("F", "minor"), "4B": ("Ab", "major"),
    "5A": ("C", "minor"), "5B": ("Eb", "major"),
    "6A": ("G", "minor"), "6B": ("Bb", "major"),
    "7A": ("D", "minor"), "7B": ("F", "major"),
    "8A": ("A", "minor"), "8B": ("C", "major"),
    "9A": ("E", "minor"), "9B": ("G", "major"),
    "10A": ("B", "minor"), "10B": ("D", "major"),
    "11A": ("F#", "minor"), "11B": ("A", "major"),
    "12A": ("Db", "minor"), "12B": ("E", "major"),
}


def build_scale_chroma_template(root_name: str, mode: str) -> np.ndarray:
    """Build a 12-dimensional chroma weight profile for a musical key."""
    root_idx = ROOT_TO_INDEX.get(root_name, 0)
    chroma = np.full(12, 0.05, dtype=np.float32)

    if "maj" in mode.lower() or mode == "B":
        # Major scale intervals: 0, 2, 4, 5, 7, 9, 11
        diatonic_weights = {
            0: 1.0,   # Root (tonic)
            7: 0.95,  # Perfect 5th
            4: 0.85,  # Major 3rd
            2: 0.65,  # Major 2nd
            5: 0.60,  # Perfect 4th
            9: 0.65,  # Major 6th
            11: 0.55, # Major 7th
        }
    else:
        # Natural minor scale intervals: 0, 2, 3, 5, 7, 8, 10
        diatonic_weights = {
            0: 1.0,   # Root (tonic)
            7: 0.95,  # Perfect 5th
            3: 0.85,  # Minor 3rd
            2: 0.65,  # Major 2nd
            5: 0.60,  # Perfect 4th
            8: 0.65,  # Minor 6th
            10: 0.60, # Minor 7th
        }

    for interval, weight in diatonic_weights.items():
        pitch = (root_idx + interval) % 12
        chroma[pitch] = weight

    norm = np.linalg.norm(chroma)
    return chroma / (norm + 1e-6)


# Cache templates for standard keys
_KEY_TEMPLATES: Dict[str, np.ndarray] = {}

def get_chroma_template_for_key(key_name: str) -> np.ndarray:
    """Return precomputed chroma template for any key (e.g. '8A', '8B', 'C major', 'A minor')."""
    if key_name in _KEY_TEMPLATES:
        return _KEY_TEMPLATES[key_name]

    clean = key_name.strip()
    if clean in CAMELOT_TO_KEY:
        root, mode = CAMELOT_TO_KEY[clean]
    elif " " in clean:
        parts = clean.split()
        root = parts[0]
        mode = parts[1] if len(parts) > 1 else "major"
    elif clean.endswith("m"):
        root = clean[:-1]
        mode = "minor"
    else:
        root = clean
        mode = "major"

    tmpl = build_scale_chroma_template(root, mode)
    _KEY_TEMPLATES[key_name] = tmpl
    return tmpl


def harmonic_consonance(slice_ref: SliceRef, target_key: Optional[str]) -> float:
    """Compute harmonic consonance (0.0..1.0) between slice chroma and target Camelot key."""
    if target_key is None or slice_ref.chroma is None or len(slice_ref.chroma) != 12:
        return 0.75

    s_norm = np.linalg.norm(slice_ref.chroma)
    if s_norm < 1e-4:
        return 0.70  # Unpitched / percussive

    tmpl = get_chroma_template_for_key(target_key)
    norm_slice = slice_ref.chroma / s_norm
    cosine = float(np.dot(norm_slice, tmpl))
    # Rescale [-1, 1] cosine similarity to [0.0, 1.0]
    return float(np.clip((cosine + 0.2) / 1.2, 0.0, 1.0))


# ==============================================================================
# 2. Physical Acoustic Slice Classifier
# ==============================================================================

class AcousticSliceClassifier:
    """Classifies audio slices into physical acoustic categories based on empirical acoustics."""

    @staticmethod
    def classify(slice_ref: SliceRef) -> str:
        """Classify a slice into a physical acoustic role."""
        stem = slice_ref.stem
        centroid = float(getattr(slice_ref, "spectral_centroid", 0.0))
        loudness = float(getattr(slice_ref, "loud_max", -30.0))
        duration = float(slice_ref.duration)
        chroma = slice_ref.chroma if slice_ref.chroma is not None else np.zeros(12)

        if stem == "drums":
            # Drums classification
            if duration >= 0.28:
                return "fill"
            if (centroid < 950.0 and loudness > -32.0) or (centroid == 0.0 and slice_ref.tatum_in_bar in (0, 8)):
                return "kick"
            if 950.0 <= centroid <= 3200.0 and loudness > -34.0:
                return "snare"
            if centroid > 2600.0 or slice_ref.tatum_in_bar in (2, 6, 10, 14):
                if duration >= 0.12:
                    return "hat_open"
                return "hat_closed"
            if loudness > -26.0 and duration >= 0.20:
                return "fill"
            return "percussion"

        elif stem == "bass":
            # Bass classification
            if (centroid < 520.0 and duration >= 0.14) or (centroid == 0.0 and duration >= 0.20):
                return "sub"
            if duration < 0.14:
                return "stab"
            return "rolling"

        elif stem == "other":
            # Synths / melodic stems
            if duration >= 0.38:
                return "pad"
            # Polyphony test: 2 or more prominent pitch classes
            polyphony = int(np.sum(chroma > 0.35))
            if polyphony >= 2:
                return "chord"
            if duration < 0.16:
                return "stab"
            return "arp"

        elif stem == "vocals":
            # Vocals
            if duration < 0.28:
                return "chop"
            return "phrase"

        return "default"

    @classmethod
    def categorize_palette(cls, slices: Sequence[SliceRef]) -> Dict[str, Dict[str, List[SliceRef]]]:
        """Organize a sequence of slices into a fully populated acoustic palette."""
        palette: Dict[str, Dict[str, List[SliceRef]]] = {
            "drums": {"kick": [], "snare": [], "hat_closed": [], "hat_open": [], "percussion": [], "fill": []},
            "bass": {"sub": [], "rolling": [], "stab": []},
            "other": {"chord": [], "arp": [], "pad": [], "stab": []},
            "vocals": {"chop": [], "phrase": []},
        }

        for s in slices:
            stem = s.stem
            if stem not in palette:
                continue
            cat = cls.classify(s)
            # Map into stem category keys
            if cat in palette[stem]:
                palette[stem][cat].append(s)
            else:
                # Fallback to closest canonical bucket
                if stem == "drums":
                    if "hat" in cat:
                        palette["drums"]["hat_closed"].append(s)
                    else:
                        palette["drums"]["percussion"].append(s)
                elif stem == "bass":
                    palette["bass"]["rolling"].append(s)
                elif stem == "other":
                    palette["other"]["chord"].append(s)
                elif stem == "vocals":
                    palette["vocals"]["chop"].append(s)

        return palette


# ==============================================================================
# 3. Dynamic Multi-Bar Groove Templates & Accent Curves
# ==============================================================================

@dataclass
class GrooveStep:
    step_index: int
    category: str
    velocity: float = 1.0        # 0.0..1.0
    duration_scale: float = 1.0  # Time envelope scaling


@dataclass
class GrooveTemplate:
    name: str
    stem: str
    length: int  # 16 or 32
    steps: List[GrooveStep]
    energy: float = 0.5


def build_drum_templates() -> Dict[str, GrooveTemplate]:
    """Predefined professional dance groove drum templates with velocity and accent curves."""
    templates = {}

    # 1. Four on the Floor with Driving Open Hats & 16th Closed Hats
    steps_fof = []
    # Kicks on 0, 4, 8, 12
    for s in (0, 4, 8, 12):
        steps_fof.append(GrooveStep(s, "kick", velocity=1.0))
    # Snares / claps on backbeats 4, 12
    for s in (4, 12):
        steps_fof.append(GrooveStep(s, "snare", velocity=0.96))
    # Open hats on upbeats 2, 6, 10, 14
    for s in (2, 6, 10, 14):
        steps_fof.append(GrooveStep(s, "hat_open", velocity=0.88))
    # Closed hats on 16th subdivisions
    for s in (1, 3, 5, 7, 9, 11, 13, 15):
        steps_fof.append(GrooveStep(s, "hat_closed", velocity=0.62, duration_scale=0.7))
    templates["four_on_floor_hats"] = GrooveTemplate("four_on_floor_hats", "drums", 16, steps_fof, energy=0.85)

    # 2. Syncopated Club Four on the Floor (Ghost Kick on Step 10 & 14)
    steps_syn = list(steps_fof)
    steps_syn.append(GrooveStep(10, "kick", velocity=0.72))
    steps_syn.append(GrooveStep(14, "percussion", velocity=0.68))
    templates["four_on_floor_syncopated"] = GrooveTemplate("four_on_floor_syncopated", "drums", 16, steps_syn, energy=0.90)

    # 3. Breakbeat Groove
    steps_bb = [
        GrooveStep(0, "kick", velocity=1.0),
        GrooveStep(4, "snare", velocity=0.95),
        GrooveStep(6, "kick", velocity=0.85),
        GrooveStep(10, "kick", velocity=0.92),
        GrooveStep(12, "snare", velocity=0.95),
        GrooveStep(15, "snare", velocity=0.55),  # ghost snare
    ]
    for s in (2, 6, 10, 14):
        steps_bb.append(GrooveStep(s, "hat_open", velocity=0.82))
    for s in (1, 3, 5, 7, 9, 11, 13):
        steps_bb.append(GrooveStep(s, "hat_closed", velocity=0.60))
    templates["breakbeat"] = GrooveTemplate("breakbeat", "drums", 16, steps_bb, energy=0.80)

    # 4. Turnaround Snare / Tom Fill with Breath on Step 15
    steps_fill = [
        GrooveStep(0, "kick", velocity=1.0),
        GrooveStep(4, "kick", velocity=0.95),
        GrooveStep(8, "kick", velocity=0.90),
        GrooveStep(8, "snare", velocity=0.70),
        GrooveStep(9, "snare", velocity=0.75),
        GrooveStep(10, "snare", velocity=0.82),
        GrooveStep(11, "snare", velocity=0.88),
        GrooveStep(12, "snare", velocity=0.95),
        GrooveStep(13, "snare", velocity=0.98),
        GrooveStep(14, "fill", velocity=1.0),
        # Notice step 15 is explicitly SILENT for the dramatic breath before the drop!
    ]
    templates["turnaround_fill"] = GrooveTemplate("turnaround_fill", "drums", 16, steps_fill, energy=0.95)

    # 5. Half-turnaround mini fill (for Bar 3 of 4-bar arc)
    steps_minifill = [
        GrooveStep(0, "kick", velocity=1.0),
        GrooveStep(4, "snare", velocity=0.95),
        GrooveStep(8, "kick", velocity=0.95),
        GrooveStep(12, "snare", velocity=0.95),
        GrooveStep(13, "snare", velocity=0.80),
        GrooveStep(14, "percussion", velocity=0.85),
    ]
    for s in (2, 6, 10):
        steps_minifill.append(GrooveStep(s, "hat_open", velocity=0.85))
    templates["mini_turnaround"] = GrooveTemplate("mini_turnaround", "drums", 16, steps_minifill, energy=0.82)

    return templates


def build_bass_templates() -> Dict[str, GrooveTemplate]:
    """Predefined professional bass groove templates."""
    templates = {}

    # 1. Pumping Offbeat Bass (ducked on kicks, hits on upbeats)
    steps_offbeat = [
        GrooveStep(2, "sub", velocity=1.0),
        GrooveStep(6, "sub", velocity=1.0),
        GrooveStep(10, "sub", velocity=1.0),
        GrooveStep(14, "sub", velocity=1.0),
        # Ghost 16th bounce
        GrooveStep(3, "stab", velocity=0.65),
        GrooveStep(7, "stab", velocity=0.65),
        GrooveStep(11, "stab", velocity=0.65),
        GrooveStep(15, "stab", velocity=0.65),
    ]
    templates["offbeat"] = GrooveTemplate("offbeat", "bass", 16, steps_offbeat, energy=0.85)

    # 2. Driving Rolling 16th Bassline (Psytrance / Techno driving groove)
    steps_rolling = []
    for s in range(16):
        if s in (0, 4, 8, 12):
            # Kick steps: sub tone or tight stab with sidechain clearance
            steps_rolling.append(GrooveStep(s, "sub", velocity=0.70))
        elif s in (2, 6, 10, 14):
            # Upbeats: accented rolling bass
            steps_rolling.append(GrooveStep(s, "rolling", velocity=1.0))
        else:
            # 16th subdivs: continuous driving roll
            steps_rolling.append(GrooveStep(s, "rolling", velocity=0.82))
    templates["rolling"] = GrooveTemplate("rolling", "bass", 16, steps_rolling, energy=0.92)

    # 3. Syncopated Funk / Electro Bass Stabs
    steps_stabs = [
        GrooveStep(0, "sub", velocity=1.0),
        GrooveStep(3, "stab", velocity=0.85),
        GrooveStep(6, "stab", velocity=0.90),
        GrooveStep(10, "stab", velocity=0.92),
        GrooveStep(12, "sub", velocity=0.88),
    ]
    templates["stabs"] = GrooveTemplate("stabs", "bass", 16, steps_stabs, energy=0.78)

    # 4. Sustained Sub Foundation (Deep root sub)
    steps_sustained = [
        GrooveStep(0, "sub", velocity=1.0, duration_scale=1.5),
        GrooveStep(8, "sub", velocity=0.90, duration_scale=1.5),
    ]
    templates["sustained"] = GrooveTemplate("sustained", "bass", 16, steps_sustained, energy=0.60)

    return templates


def build_other_templates() -> Dict[str, GrooveTemplate]:
    """Predefined synth / instruments templates."""
    templates = {}

    # 1. Offbeat Synth Chords
    steps_chords = [
        GrooveStep(2, "chord", velocity=0.90),
        GrooveStep(6, "chord", velocity=0.90),
        GrooveStep(10, "chord", velocity=0.95),
        GrooveStep(14, "chord", velocity=0.85),
    ]
    templates["chords"] = GrooveTemplate("chords", "other", 16, steps_chords, energy=0.75)

    # 2. Driving 16th Arpeggio Hook
    steps_arp = []
    for s in range(16):
        vel = 0.95 if s in (0, 4, 8, 12) else (0.80 if s in (2, 6, 10, 14) else 0.65)
        steps_arp.append(GrooveStep(s, "arp", velocity=vel))
    templates["riff"] = GrooveTemplate("riff", "other", 16, steps_arp, energy=0.88)

    # 3. Ambient Sustained Pad
    steps_pad = [
        GrooveStep(0, "pad", velocity=0.85, duration_scale=2.0),
        GrooveStep(8, "pad", velocity=0.80, duration_scale=2.0),
    ]
    templates["pad"] = GrooveTemplate("pad", "other", 16, steps_pad, energy=0.50)

    # 4. Syncopated Synth Stabs
    steps_stabs = [
        GrooveStep(3, "stab", velocity=0.88),
        GrooveStep(7, "stab", velocity=0.88),
        GrooveStep(11, "stab", velocity=0.92),
        GrooveStep(14, "stab", velocity=0.85),
    ]
    templates["hook"] = GrooveTemplate("hook", "other", 16, steps_stabs, energy=0.72)

    return templates


def build_vocal_templates() -> Dict[str, GrooveTemplate]:
    """Predefined vocal groove templates."""
    templates = {}

    # 1. Rhythmic Vocal Chops
    steps_chops = [
        GrooveStep(2, "chop", velocity=0.90),
        GrooveStep(6, "chop", velocity=0.85),
        GrooveStep(14, "chop", velocity=0.95),
    ]
    templates["chop"] = GrooveTemplate("chop", "vocals", 16, steps_chops, energy=0.70)

    # 2. Vocal Lead Phrase
    steps_phrase = [
        GrooveStep(0, "phrase", velocity=0.95, duration_scale=1.5),
        GrooveStep(8, "phrase", velocity=0.90, duration_scale=1.5),
    ]
    templates["phrase"] = GrooveTemplate("phrase", "vocals", 16, steps_phrase, energy=0.75)

    return templates


GLOBAL_GROOVE_TEMPLATES: Dict[str, Dict[str, GrooveTemplate]] = {
    "drums": build_drum_templates(),
    "bass": build_bass_templates(),
    "other": build_other_templates(),
    "vocals": build_vocal_templates(),
}


# ==============================================================================
# 4. Exact Right Audio Slice Selector
# ==============================================================================

class SliceSelector:
    """Selects the exact right audio slices to fill a groove template with multi-objective ranking."""

    def __init__(
        self,
        palette: Dict[str, Dict[str, List[SliceRef]]],
        feedback_weights: Optional[Dict[str, float]] = None,
        target_key: Optional[str] = None,
        preferred_track_id: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.palette = palette
        self.feedback_weights = feedback_weights or {}
        self.target_key = target_key
        self.preferred_track_id = preferred_track_id
        self.rng = rng or np.random.default_rng()

    def pick_slice(self, stem: str, required_category: str) -> Optional[SliceRef]:
        """Rank candidate slices by acoustic match, harmonic consonance, and feedback scores."""
        stem_pal = self.palette.get(stem, {})
        candidates = stem_pal.get(required_category, [])

        if not candidates:
            # Fallback across category buckets in this stem
            candidates = [s for bucket in stem_pal.values() for s in bucket]

        if not candidates:
            return None

        # If only 1 candidate, return it
        if len(candidates) == 1:
            return candidates[0]

        # Multi-objective candidate scoring
        # Score = 0.40 * AcousticCategoryMatch + 0.35 * HarmonicConsonance + 0.25 * FeedbackScore + TrackBonus
        scores = np.zeros(len(candidates), dtype=np.float32)

        for i, s in enumerate(candidates):
            # 1. Acoustic Category Match
            actual_cat = AcousticSliceClassifier.classify(s)
            c_score = 1.0 if actual_cat == required_category else 0.5

            # 2. Harmonic Consonance
            h_score = harmonic_consonance(s, self.target_key) if stem in ("bass", "other", "vocals") else 1.0

            # 3. Audio Similarity Feedback Score
            f_score = float(self.feedback_weights.get(s.track_id, 0.5))

            # 4. Preferred Track Bonus
            t_bonus = 0.20 if (self.preferred_track_id and s.track_id == self.preferred_track_id) else 0.0

            total_score = (0.35 * c_score) + (0.35 * h_score) + (0.30 * f_score) + t_bonus
            scores[i] = total_score

        # Softmax selection with temperature to introduce groovy musical variety
        temp = 0.25
        exp_scores = np.exp((scores - np.max(scores)) / temp)
        probs = exp_scores / np.sum(exp_scores)

        chosen_idx = int(self.rng.choice(len(candidates), p=probs))
        return candidates[chosen_idx]


# ==============================================================================
# 5. Groovy Pattern Assembler & Multi-Bar Evolution
# ==============================================================================

def assemble_groovy_pattern(
    stem: str,
    template_name: str,
    palette: Dict[str, Dict[str, List[SliceRef]]],
    target_key: Optional[str] = None,
    preferred_track_id: Optional[str] = None,
    feedback_weights: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SequencerPattern]:
    """Assemble a dance pattern using a rhythmic groove template and physical slice ranking."""
    stem_templates = GLOBAL_GROOVE_TEMPLATES.get(stem, {})
    tmpl = stem_templates.get(template_name)
    if tmpl is None:
        # Fallback to first available template
        tmpl = next(iter(stem_templates.values()), None)
    if tmpl is None:
        return None

    selector = SliceSelector(
        palette=palette,
        feedback_weights=feedback_weights,
        target_key=target_key,
        preferred_track_id=preferred_track_id,
        rng=rng,
    )

    steps: List[Optional[SliceRef]] = [None] * tmpl.length
    velocities: List[float] = [0.0] * tmpl.length
    onset_grid: List[int] = []

    for g_step in tmpl.steps:
        s_idx = g_step.step_index
        if 0 <= s_idx < tmpl.length:
            slice_ref = selector.pick_slice(stem, g_step.category)
            if slice_ref is not None:
                steps[s_idx] = slice_ref
                velocities[s_idx] = float(g_step.velocity)
                if s_idx not in onset_grid:
                    onset_grid.append(s_idx)

    onset_grid.sort()
    density = len(onset_grid) / float(tmpl.length)

    pat_id = f"pat_{stem}_{template_name}_{int(rng.integers(1000, 9999)) if rng else 1000}"
    return SequencerPattern(
        id=pat_id,
        stem=stem,
        track_id=preferred_track_id or (steps[onset_grid[0]].track_id if onset_grid and steps[onset_grid[0]] else "gen"),
        pattern_name=f"{stem}_{template_name}",
        length=tmpl.length,
        steps=steps,
        energy=tmpl.energy,
        density=density,
        onset_grid=onset_grid,
        groove_type=template_name,
        key=target_key or "C major",
        velocities=velocities,
    )


def assemble_evolving_phrase(
    phrase_bar: int,
    palette: Dict[str, Dict[str, List[SliceRef]]],
    target_key: Optional[str] = None,
    preferred_track_id: Optional[str] = None,
    feedback_weights: Optional[Dict[str, float]] = None,
    section: str = "GROOVE_A",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Optional[SequencerPattern]]:
    """Assemble an evolving multi-stem pattern for phrase bar 0..7 with phrase-level groove evolution."""
    if rng is None:
        rng = np.random.default_rng()

    # Musical phrase evolution roadmap across an 8-bar arc:
    # Bar 0 (A): Foundation (Kicks, backbeat snares, offbeat hats, offbeat bass, chords)
    # Bar 1 (A'): Added 16th ghost hats & rolling bounce
    # Bar 2 (B): Syncopated kick on step 10, vocal chop introduced
    # Bar 3 (Turnaround 1): Mini-turnaround fill on Bar 4 of 4-bar sub-phrase
    # Bar 4 (C): Energy peak / lift (syncopated 4-on-floor, rolling bass, driving arps)
    # Bar 5 (C'): Polyrhythmic hats and vocal hook
    # Bar 6 (D): Rising tension buildup
    # Bar 7 (Turnaround 2 / Climax): Climactic snare roll fill, bass cut, silence on 15 before drop!

    drum_template = "four_on_floor_hats"
    bass_template = "offbeat"
    other_template = "chords"
    vocal_template = "chop"

    if phrase_bar == 0:
        drum_template = "four_on_floor_hats"
        bass_template = "offbeat"
        other_template = "chords"
    elif phrase_bar == 1:
        drum_template = "four_on_floor_hats"
        bass_template = "rolling" if section in ("DROP", "GROOVE_A") else "offbeat"
        other_template = "chords"
    elif phrase_bar == 2:
        drum_template = "four_on_floor_syncopated"
        bass_template = "stabs"
        other_template = "hook"
        vocal_template = "chop"
    elif phrase_bar == 3:
        drum_template = "mini_turnaround"
        bass_template = "stabs"
        other_template = "chords"
    elif phrase_bar == 4:
        drum_template = "four_on_floor_syncopated"
        bass_template = "rolling"
        other_template = "riff"
        vocal_template = "chop"
    elif phrase_bar == 5:
        drum_template = "four_on_floor_hats"
        bass_template = "rolling"
        other_template = "riff"
    elif phrase_bar == 6:
        drum_template = "four_on_floor_syncopated"
        bass_template = "stabs"
        other_template = "hook"
    elif phrase_bar == 7:
        # Full climactic turnaround fill
        drum_template = "turnaround_fill"
        bass_template = "sustained"  # Deep sustained sub or cut
        other_template = "pad"
        vocal_template = "chop"

    patterns: Dict[str, Optional[SequencerPattern]] = {}
    patterns["drums"] = assemble_groovy_pattern(
        "drums", drum_template, palette,
        target_key=target_key, preferred_track_id=preferred_track_id,
        feedback_weights=feedback_weights, rng=rng,
    )
    patterns["bass"] = assemble_groovy_pattern(
        "bass", bass_template, palette,
        target_key=target_key, preferred_track_id=preferred_track_id,
        feedback_weights=feedback_weights, rng=rng,
    )
    patterns["other"] = assemble_groovy_pattern(
        "other", other_template, palette,
        target_key=target_key, preferred_track_id=preferred_track_id,
        feedback_weights=feedback_weights, rng=rng,
    )
    patterns["vocals"] = assemble_groovy_pattern(
        "vocals", vocal_template, palette,
        target_key=target_key, preferred_track_id=preferred_track_id,
        feedback_weights=feedback_weights, rng=rng,
    )

    return patterns


# ==============================================================================
# 6. Sidechain Ducking & Envelope Shaping
# ==============================================================================

def apply_sidechain_ducking(
    bass_waveform: np.ndarray,
    kick_steps: Sequence[int],
    bpm: float = 126.0,
    sr: int = 44100,
    duck_gain_db: float = -6.0,
    recovery_ms: float = 65.0,
) -> np.ndarray:
    """Apply simulated club sidechain ducking to bass audio during kick transients.
    
    Attenuates bass by duck_gain_db during the first few milliseconds of kick hits,
    with smooth exponential release to prevent low-end mud and phase cancellation.
    """
    if len(bass_waveform) == 0 or not kick_steps:
        return bass_waveform

    ducked = bass_waveform.copy()
    tatum_sec = (60.0 / bpm / 4.0) if bpm > 0 else 0.119
    step_samples = int(tatum_sec * sr)
    duck_samples = min(step_samples, int(sr * (recovery_ms / 1000.0)))

    min_gain = 10.0 ** (duck_gain_db / 20.0)  # e.g. -6 dB -> 0.501
    recovery_curve = 1.0 - (1.0 - min_gain) * np.exp(-3.0 * np.linspace(0.0, 1.0, duck_samples))

    for step in kick_steps:
        start_idx = step * step_samples
        if start_idx >= len(ducked):
            break
        end_idx = min(len(ducked), start_idx + duck_samples)
        applied_len = end_idx - start_idx
        if applied_len > 0:
            ducked[start_idx:end_idx] *= recovery_curve[:applied_len]

    return ducked


def apply_hihat_envelope(
    hat_waveform: np.ndarray,
    is_closed: bool,
    sr: int = 44100,
) -> np.ndarray:
    """Shape the decay envelope of hi-hat slices to eliminate trailing mud."""
    if len(hat_waveform) == 0:
        return hat_waveform

    shaped = hat_waveform.copy()
    if is_closed:
        # Closed hat: tight 80ms decay envelope
        decay_samples = min(len(shaped), int(sr * 0.08))
        if decay_samples > 0:
            env = np.exp(-4.0 * np.linspace(0.0, 1.0, len(shaped)))
            shaped *= env
    return shaped
