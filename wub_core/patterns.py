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
    bar_num: int = 0
    bpm: float = 126.0
    key: str = "C major"


@dataclass
class MultiStemPattern:
    """A cohesive 16-step (1 bar) pattern where all stems come from the SAME track and bar."""
    id: str
    track_id: str
    track_title: str
    bar_num: int
    phrase_bar: int  # 0..7
    section: str     # "INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"
    bpm: float = 126.0
    key: str = "C major"
    energy: float = 0.5
    is_turnaround: bool = False
    stems: Dict[str, List[Optional[SliceRef]]] = field(default_factory=dict)
    onset_grids: Dict[str, List[int]] = field(default_factory=dict)
    groove_types: Dict[str, str] = field(default_factory=dict)


@dataclass
class SongPhrase:
    """An authentic 8-bar musical phrase (128 tatums) extracted from a track."""
    id: str
    track_id: str
    track_title: str
    phrase_index: int
    section: str
    bpm: float
    key: str
    energy: float
    bars: List[MultiStemPattern]
    has_fill: bool = False


def classify_drum_groove(onset_grid: List[int], length: int) -> str:
    """Classify 16-step drum pattern into dance groove categories."""
    first_16 = set(s for s in onset_grid if s < 16)
    density = len(first_16) / 16.0

    if density >= 0.8:
        return "fill"  # High density snare/drum roll or turnaround
    has_four_on_floor = {0, 4, 8, 12}.issubset(first_16)
    if has_four_on_floor:
        if 2 in first_16 or 6 in first_16 or 10 in first_16 or 14 in first_16:
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


def classify_vocal_groove(onset_grid: List[int], density: float) -> str:
    if density >= 0.4:
        return "phrase"  # Full vocal phrase / verse line
    if density <= 0.15:
        return "chop"  # Vocal chop / stab
    return "hook"


class PatternBank:
    """In-memory bank of categorized 1-bar sequencer patterns and multi-stem song phrases."""

    def __init__(self) -> None:
        self.patterns_by_stem: Dict[str, List[SequencerPattern]] = {s: [] for s in STEMS}
        self.patterns_by_type: Dict[str, Dict[str, List[SequencerPattern]]] = {
            s: {} for s in STEMS
        }
        self.patterns_by_track: Dict[Tuple[str, str, int], SequencerPattern] = {}
        self.track_bar_counts: Dict[Tuple[str, str], int] = {}
        self.multistem_patterns: List[MultiStemPattern] = []
        self.multistem_by_track_bar: Dict[Tuple[str, int], MultiStemPattern] = {}
        self.phrases: List[SongPhrase] = []
        self.phrases_by_section: Dict[str, List[SongPhrase]] = {
            "GROOVE_A": [],
            "DROP": [],
            "BUILDUP": [],
            "BREAKDOWN": [],
            "INTRO": [],
        }
        self.phrases_by_track: Dict[str, List[SongPhrase]] = {}
        self.groove_model: Optional[Any] = None
        self.slice_palette: Dict[str, Dict[str, List[SliceRef]]] = {
            "drums": {"kick": [], "snare": [], "hat": [], "fill": []},
            "bass": {"sub": [], "stab": [], "rolling": []},
            "other": {"chord": [], "stab": [], "arp": [], "pad": []},
            "vocals": {"chop": [], "phrase": []},
        }

    def load_from_tracks(self, tracks: List[TrackAnalysis], min_bpm: float = 90.0, max_bpm: float = 160.0) -> int:
        """Extract cohesive MultiStemPatterns and SongPhrases alongside SequencerPatterns."""
        pools = build_pools(tracks, exclude_silent=False)
        total_extracted = 0

        for tr in tracks:
            if not (min_bpm <= tr.bpm <= max_bpm):
                continue

            n_tatums = len(tr.tatums)
            if n_tatums < 16:
                continue

            # Map from (stem, segment_id) -> SliceRef and pre-populate slice palette
            seg_lookup: Dict[Tuple[str, str], SliceRef] = {}
            for stem in tr.stems_available:
                stem_pool = pools.get(stem)
                if stem_pool:
                    for s in stem_pool.slices:
                        if s.track_id == tr.track_id:
                            seg_lookup[(stem, s.segment_id)] = s
                            if stem == "drums":
                                cent = getattr(s, "spectral_centroid", 1500.0)
                                if cent < 1200.0 and s.loud_max > -32.0:
                                    self.slice_palette["drums"]["kick"].append(s)
                                elif 1200.0 <= cent <= 3500.0 and s.loud_max > -35.0:
                                    self.slice_palette["drums"]["snare"].append(s)
                                elif cent > 2500.0:
                                    self.slice_palette["drums"]["hat"].append(s)
                                else:
                                    self.slice_palette["drums"]["fill"].append(s)
                            elif stem == "bass":
                                if s.duration > 0.15:
                                    self.slice_palette["bass"]["sub"].append(s)
                                else:
                                    self.slice_palette["bass"]["stab"].append(s)
                            elif stem == "other":
                                self.slice_palette["other"]["chord"].append(s)
                                self.slice_palette["other"]["pad"].append(s)
                            elif stem == "vocals":
                                self.slice_palette["vocals"]["chop"].append(s)
                                self.slice_palette["vocals"]["phrase"].append(s)

            # Find downbeat bars (every 16 tatums starting from a downbeat)
            downbeat_indices = [
                i for i, tat in enumerate(tr.tatums)
                if tat.beat_position == 0 and tat.tatum_position == 0
            ]
            if not downbeat_indices:
                downbeat_indices = list(range(0, n_tatums - 15, 16))

            prev_stem_seg: Dict[str, Optional[str]] = {s: None for s in tr.stems_available}
            track_multi_patterns: List[MultiStemPattern] = []

            for start_idx in downbeat_indices:
                if start_idx + 16 > n_tatums:
                    continue

                bar_num = start_idx // 16
                phrase_bar = bar_num % 8

                bar_stems: Dict[str, List[Optional[SliceRef]]] = {s: [None] * 16 for s in STEMS}
                bar_onsets: Dict[str, List[int]] = {s: [] for s in STEMS}
                bar_grooves: Dict[str, str] = {s: "none" for s in STEMS}
                bar_loudness: List[float] = []

                for step_idx in range(16):
                    t_pos = start_idx + step_idx
                    tat = tr.tatums[t_pos]

                    for stem in tr.stems_available:
                        seg_ids = tat.stem_segments.get(stem, [])
                        curr_seg = seg_ids[-1] if seg_ids else None

                        # TRUE ONSET: step 0 (downbeat) or a new segment starts on this tatum step
                        is_onset = (step_idx == 0 and curr_seg is not None) or (
                            curr_seg is not None and curr_seg != prev_stem_seg.get(stem)
                        )
                        if is_onset:
                            sref = seg_lookup.get((stem, curr_seg))
                            if sref is not None and sref.loud_max > -48.0:
                                bar_stems[stem][step_idx] = sref
                                bar_onsets[stem].append(step_idx)
                                bar_loudness.append(max(0.0, sref.loud_max + 60.0) / 60.0)
                        prev_stem_seg[stem] = curr_seg

                # --- PATTERN SNIPPETS FEEDBACK & RECONSTRUCTION LOOP ---
                # Feedback check 1: Drum rhythm integrity
                # Real dance drums must hold the rhythm without dropouts or awkward silence.
                drum_onsets = bar_onsets["drums"]
                has_downbeat = any(s in drum_onsets for s in (0, 8))
                has_backbeat = any(s in drum_onsets for s in (4, 12))
                is_rhythm_broken = (len(drum_onsets) < 4) or not (has_downbeat and has_backbeat)

                track_kicks = [s for s in self.slice_palette["drums"]["kick"] if s.track_id == tr.track_id]
                track_snares = [s for s in self.slice_palette["drums"]["snare"] if s.track_id == tr.track_id]
                track_hats = [s for s in self.slice_palette["drums"]["hat"] if s.track_id == tr.track_id]
                track_fills = [s for s in self.slice_palette["drums"]["fill"] if s.track_id == tr.track_id]

                k_ref = track_kicks[0] if track_kicks else (track_fills[0] if track_fills else (self.slice_palette["drums"]["kick"][0] if self.slice_palette["drums"]["kick"] else None))
                s_ref = track_snares[0] if track_snares else (track_fills[0] if track_fills else (self.slice_palette["drums"]["snare"][0] if self.slice_palette["drums"]["snare"] else k_ref))
                h_ref = track_hats[0] if track_hats else (self.slice_palette["drums"]["hat"][0] if self.slice_palette["drums"]["hat"] else s_ref)

                if is_rhythm_broken and k_ref is not None:
                    # Anchor downbeats 0 and 8
                    if bar_stems["drums"][0] is None:
                        bar_stems["drums"][0] = k_ref
                    if bar_stems["drums"][8] is None:
                        bar_stems["drums"][8] = k_ref
                    # Anchor backbeats 4 and 12
                    if s_ref is not None:
                        if bar_stems["drums"][4] is None:
                            bar_stems["drums"][4] = s_ref
                        if bar_stems["drums"][12] is None:
                            bar_stems["drums"][12] = s_ref
                    else:
                        if bar_stems["drums"][4] is None:
                            bar_stems["drums"][4] = k_ref
                        if bar_stems["drums"][12] is None:
                            bar_stems["drums"][12] = k_ref
                    # Anchor offbeat driving hi-hats 2, 6, 10, 14
                    if h_ref is not None:
                        for ob in (2, 6, 10, 14):
                            if bar_stems["drums"][ob] is None:
                                bar_stems["drums"][ob] = h_ref

                    bar_onsets["drums"] = [i for i, s in enumerate(bar_stems["drums"]) if s is not None]

                # Feedback check 2: Bass pocket integrity
                if len(bar_onsets["bass"]) == 0:
                    track_bass = [s for pool in self.slice_palette["bass"].values() for s in pool if s.track_id == tr.track_id]
                    if not track_bass:
                        track_bass = [s for pool in self.slice_palette["bass"].values() for s in pool]
                    if track_bass:
                        b_ref = track_bass[0]
                        for ob in (2, 6, 10, 14):
                            bar_stems["bass"][ob] = b_ref
                        bar_onsets["bass"] = [2, 6, 10, 14]

                # Feedback check 3: Synths/Other melodic groove integrity
                if len(bar_onsets["other"]) == 0:
                    track_other = [s for pool in self.slice_palette["other"].values() for s in pool if s.track_id == tr.track_id]
                    if not track_other:
                        track_other = [s for pool in self.slice_palette["other"].values() for s in pool]
                    if track_other:
                        o_ref = track_other[0]
                        for ob in (2, 10):
                            bar_stems["other"][ob] = o_ref
                        bar_onsets["other"] = [2, 10]

                # Classify groove types for each stem after reinforcement
                bar_grooves["drums"] = classify_drum_groove(bar_onsets["drums"], 16)
                bar_grooves["bass"] = classify_bass_groove(bar_onsets["bass"], 16)
                d_density = len(bar_onsets["other"]) / 16.0
                bar_grooves["other"] = classify_synth_groove(bar_onsets["other"], d_density)
                v_density = len(bar_onsets["vocals"]) / 16.0
                bar_grooves["vocals"] = classify_vocal_groove(bar_onsets["vocals"], v_density)

                energy = float(np.mean(bar_loudness)) if bar_loudness else 0.5
                is_turnaround = (phrase_bar == 7) or (bar_grooves["drums"] == "fill")

                # Build MultiStemPattern
                multi_id = f"msp_{tr.track_id[:6]}_{bar_num:03d}"
                msp = MultiStemPattern(
                    id=multi_id,
                    track_id=tr.track_id,
                    track_title=tr.filename.replace(".mp3", ""),
                    bar_num=bar_num,
                    phrase_bar=phrase_bar,
                    section="GROOVE_A",
                    bpm=tr.bpm,
                    key=tr.key,
                    energy=energy,
                    is_turnaround=is_turnaround,
                    stems=bar_stems,
                    onset_grids=bar_onsets,
                    groove_types=bar_grooves,
                )
                self.multistem_patterns.append(msp)
                self.multistem_by_track_bar[(tr.track_id, bar_num)] = msp
                track_multi_patterns.append(msp)

                # Populate single-stem SequencerPattern for backwards compatibility
                for stem in tr.stems_available:
                    steps = list(bar_stems[stem])
                    onset_grid = list(bar_onsets[stem])
                    density = len(onset_grid) / 16.0
                    groove_type = bar_grooves[stem]

                    # Harvest slices into palette
                    for s_idx in onset_grid:
                        sref = steps[s_idx]
                        if sref is None:
                            continue
                        if stem == "drums":
                            if s_idx in (0, 8) and sref.loud_max > -30.0:
                                self.slice_palette["drums"]["kick"].append(sref)
                            elif s_idx in (4, 12) and sref.loud_max > -30.0:
                                self.slice_palette["drums"]["snare"].append(sref)
                            elif s_idx in (2, 6, 10, 14):
                                self.slice_palette["drums"]["hat"].append(sref)
                            else:
                                self.slice_palette["drums"]["fill"].append(sref)
                        elif stem == "bass":
                            if s_idx in (2, 6, 10, 14):
                                self.slice_palette["bass"]["sub"].append(sref)
                            elif groove_type == "rolling" or s_idx in (1, 3, 5, 7, 9, 11, 13, 15):
                                self.slice_palette["bass"]["rolling"].append(sref)
                            else:
                                self.slice_palette["bass"]["stab"].append(sref)
                        elif stem == "other":
                            if sref.duration >= 0.25:
                                self.slice_palette["other"]["pad"].append(sref)
                                self.slice_palette["other"]["chord"].append(sref)
                            elif s_idx in (2, 6, 10, 14):
                                self.slice_palette["other"]["chord"].append(sref)
                                self.slice_palette["other"]["stab"].append(sref)
                            else:
                                self.slice_palette["other"]["arp"].append(sref)
                        elif stem == "vocals":
                            if sref.duration < 0.3:
                                self.slice_palette["vocals"]["chop"].append(sref)
                            else:
                                self.slice_palette["vocals"]["phrase"].append(sref)

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
                        bar_num=bar_num,
                        bpm=tr.bpm,
                        key=tr.key,
                    )

                    self.patterns_by_stem[stem].append(pat)
                    self.patterns_by_type[stem].setdefault(groove_type, []).append(pat)
                    self.patterns_by_track[(tr.track_id, stem, bar_num)] = pat
                    prev_max = self.track_bar_counts.get((tr.track_id, stem), 0)
                    self.track_bar_counts[(tr.track_id, stem)] = max(prev_max, bar_num + 1)
                    total_extracted += 1

            # Group consecutive 8 bars into authentic SongPhrases
            n_phrases = len(track_multi_patterns) // 8
            for p_idx in range(n_phrases):
                phrase_bars = track_multi_patterns[p_idx * 8 : (p_idx + 1) * 8]
                d_densities = [len(b.onset_grids.get("drums", [])) / 16.0 for b in phrase_bars]
                b_densities = [len(b.onset_grids.get("bass", [])) / 16.0 for b in phrase_bars]
                avg_d = float(np.mean(d_densities))
                avg_b = float(np.mean(b_densities))
                avg_energy = float(np.mean([b.energy for b in phrase_bars]))
                has_fill = any(b.is_turnaround for b in phrase_bars) or (phrase_bars[-1].groove_types.get("drums") == "fill")

                if avg_d >= 0.22 and avg_b >= 0.18 and avg_energy >= 0.35:
                    sec = "DROP"
                elif avg_d >= 0.12 or avg_b >= 0.12:
                    sec = "GROOVE_A"
                elif avg_d < 0.08 and avg_b < 0.08 and avg_energy < 0.2:
                    sec = "BREAKDOWN"
                else:
                    sec = "INTRO"

                for b in phrase_bars:
                    b.section = sec

                phrase = SongPhrase(
                    id=f"phr_{tr.track_id[:6]}_{p_idx:02d}",
                    track_id=tr.track_id,
                    track_title=tr.filename.replace(".mp3", ""),
                    phrase_index=p_idx,
                    section=sec,
                    bpm=tr.bpm,
                    key=tr.key,
                    energy=avg_energy,
                    bars=phrase_bars,
                    has_fill=has_fill,
                )
                self.phrases.append(phrase)
                self.phrases_by_section.setdefault(sec, []).append(phrase)
                self.phrases_by_track.setdefault(tr.track_id, []).append(phrase)

        # Train statistical groove model on extracted track bars
        try:
            from wub_core.groove_miner import StatisticalGrooveModel
            self.groove_model = StatisticalGrooveModel()
            self.groove_model.fit_from_tracks(tracks)
        except Exception:
            pass

        return total_extracted

    def get_multistem_pattern(self, track_id: str, bar_num: int) -> Optional[MultiStemPattern]:
        return self.multistem_by_track_bar.get((track_id, bar_num))

    def pick_phrase(
        self,
        section: Optional[str] = None,
        preferred_track_id: Optional[str] = None,
        target_key: Optional[str] = None,
        min_energy: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[SongPhrase]:
        if rng is None:
            rng = np.random.default_rng()

        candidates = list(self.phrases_by_section.get(section, [])) if section else list(self.phrases)
        if not candidates:
            candidates = list(self.phrases)
        if not candidates:
            return None

        if min_energy > 0:
            filtered = [p for p in candidates if p.energy >= min_energy]
            if filtered:
                candidates = filtered

        if preferred_track_id:
            track_matches = [p for p in candidates if p.track_id == preferred_track_id]
            if track_matches and rng.random() < 0.75:
                return track_matches[int(rng.choice(len(track_matches)))]

        return candidates[int(rng.choice(len(candidates)))]

    def get_track_pattern(self, track_id: str, stem: str, bar_num: int) -> Optional[SequencerPattern]:
        return self.patterns_by_track.get((track_id, stem, bar_num))

    def get_patterns(
        self,
        stem: str,
        groove_type: Optional[str] = None,
        min_energy: float = 0.0,
        max_energy: float = 1.0,
        strict_groove: bool = False,
    ) -> List[SequencerPattern]:
        """Query patterns matching stem, groove type, and energy."""
        if groove_type:
            candidates = list(self.patterns_by_type.get(stem, {}).get(groove_type, []))
            if not candidates and strict_groove:
                return []
        else:
            candidates = list(self.patterns_by_stem.get(stem, []))

        if not candidates:
            if strict_groove:
                return []
            candidates = list(self.patterns_by_stem.get(stem, []))
        if not candidates:
            return []

        matched = [
            p for p in candidates
            if min_energy <= p.energy <= max_energy
        ]
        return matched if matched else (candidates if not strict_groove else [])

    def pick_pattern(
        self,
        stem: str,
        groove_type: Optional[str] = None,
        min_energy: float = 0.0,
        max_energy: float = 1.0,
        preferred_track_id: Optional[str] = None,
        target_key: Optional[str] = None,
        feedback_weights: Optional[Dict[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
        strict_groove: bool = False,
    ) -> Optional[SequencerPattern]:
        """Pick a musically compatible pattern using RNG, Camelot key matching, and user feedback weighting."""
        if rng is None:
            rng = np.random.default_rng()

        is_strict = strict_groove or (groove_type == "fill")
        candidates = self.get_patterns(stem, groove_type, min_energy, max_energy, strict_groove=is_strict)
        if not candidates:
            synth_type = groove_type or ("four_on_floor_hats" if stem == "drums" else ("offbeat" if stem == "bass" else "chords"))
            return self.create_groove_pattern(stem, synth_type, preferred_track_id=preferred_track_id, rng=rng)

        # Camelot harmonic key matching for pitched stems (bass, vocals, other)
        if target_key and stem in ("bass", "vocals", "other"):
            try:
                from wub_core.composer import camelot_distance
                harmonic_candidates = [
                    p for p in candidates
                    if p.key and camelot_distance(target_key, p.key) <= 1.5
                ]
                if harmonic_candidates:
                    candidates = harmonic_candidates
            except Exception:
                pass

        if not candidates:
            return None

        # Feedback & preference weighting
        weights = np.ones(len(candidates), dtype=float)
        for i, p in enumerate(candidates):
            w = 1.0
            if feedback_weights and p.track_id in feedback_weights:
                w *= float(feedback_weights[p.track_id])
            if preferred_track_id and p.track_id == preferred_track_id:
                w *= 2.0
            weights[i] = max(0.01, w)

        tot = float(np.sum(weights))
        if tot > 0:
            probs = weights / tot
            idx = int(rng.choice(len(candidates), p=probs))
            return candidates[idx]

        return candidates[int(rng.choice(len(candidates)))]

    def create_groove_pattern(
        self,
        stem: str,
        template_name: str,
        preferred_track_id: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[SequencerPattern]:
        """Synthesize a dedicated 16-step dance groove pattern using statistical ML and real slices."""
        if rng is None:
            rng = np.random.default_rng()

        if self.groove_model is not None:
            stat_pat = self.groove_model.generate_statistical_groove(
                stem=stem,
                template_name=template_name,
                slice_palette=self.slice_palette.get(stem, {}),
                preferred_track_id=preferred_track_id,
                rng=rng,
            )
            if stat_pat is not None:
                return stat_pat

        pal = self.slice_palette.get(stem, {})
        steps: List[Optional[SliceRef]] = [None] * 16

        def _pick_slice(pool_name: str) -> Optional[SliceRef]:
            pool = pal.get(pool_name, [])
            if not pool:
                # Fallback to any slices for this stem
                pool = [s for p in self.patterns_by_stem.get(stem, []) for s in p.steps if s is not None]
            if not pool:
                return None
            if preferred_track_id:
                sub = [s for s in pool if s.track_id == preferred_track_id]
                if sub and rng.random() < 0.7:
                    return sub[int(rng.choice(len(sub)))]
            return pool[int(rng.choice(len(pool)))]

        if stem == "drums":
            if template_name == "fill":
                # Snare / tom rush fill into the next section
                for idx in (6, 8, 10, 12, 13, 14, 15):
                    steps[idx] = _pick_slice("fill") or _pick_slice("snare")
                steps[0] = _pick_slice("kick")
                steps[4] = _pick_slice("kick")
            elif template_name == "buildup_8th":
                for idx in (0, 2, 4, 6, 8, 10, 12, 14):
                    steps[idx] = _pick_slice("snare")
            elif template_name == "buildup_16th":
                for idx in range(16):
                    steps[idx] = _pick_slice("snare")
            elif template_name == "four_on_floor_hats":
                for idx in (0, 4, 8, 12):
                    steps[idx] = _pick_slice("kick")
                for idx in (4, 12):
                    steps[idx] = _pick_slice("snare")
                for idx in (2, 6, 10, 14):
                    steps[idx] = _pick_slice("hat")
            else:  # standard four_on_floor
                for idx in (0, 4, 8, 12):
                    steps[idx] = _pick_slice("kick")
                for idx in (4, 12):
                    steps[idx] = _pick_slice("snare")
        elif stem == "bass":
            if template_name == "offbeat":
                for idx in (2, 6, 10, 14):
                    steps[idx] = _pick_slice("sub")
            elif template_name == "rolling":
                for idx in (2, 3, 6, 7, 10, 11, 14, 15):
                    steps[idx] = _pick_slice("rolling") or _pick_slice("sub")
            else:
                for idx in (0, 3, 8, 11):
                    steps[idx] = _pick_slice("stab")
        elif stem == "other":
            if template_name == "chords":
                for idx in (2, 6, 10, 14):
                    steps[idx] = _pick_slice("chord")
            elif template_name == "pad":
                steps[0] = _pick_slice("pad")
            else:
                for idx in (0, 2, 4, 6, 8, 10, 12, 14):
                    steps[idx] = _pick_slice("arp")
        elif stem == "vocals":
            if template_name == "chop":
                steps[6] = _pick_slice("chop")
                steps[14] = _pick_slice("chop")
            else:
                steps[0] = _pick_slice("phrase")

        onset_grid = [i for i, s in enumerate(steps) if s is not None]
        if not onset_grid:
            return None

        density = len(onset_grid) / 16.0
        return SequencerPattern(
            id=f"synth_{stem}_{template_name}_{int(rng.integers(1000, 9999))}",
            stem=stem,
            track_id=preferred_track_id or "groovebox",
            pattern_name=f"Groovebox {template_name.title()} ({stem})",
            length=16,
            steps=steps,
            energy=0.7 if "fill" in template_name else 0.5,
            density=density,
            onset_grid=onset_grid,
            groove_type=template_name,
            bar_num=0,
            bpm=126.0,
            key="C major",
        )

    def create_compound_pattern(
        self,
        stem: str,
        pat1: Optional[SequencerPattern],
        pat2: Optional[SequencerPattern],
        target_length: int = 32,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[SequencerPattern]:
        """Combine or synthesize a 32-step compound pattern across two evolving bars."""
        if rng is None:
            rng = np.random.default_rng()

        p1 = pat1 or self.pick_pattern(stem, rng=rng)
        p2 = pat2 or self.pick_pattern(stem, preferred_track_id=p1.track_id if p1 else None, rng=rng)
        if not p1 and not p2:
            return None
        if not p1:
            p1 = p2
        if not p2:
            p2 = p1

        steps1 = list(p1.steps) if p1.steps else [None] * 16
        steps2 = list(p2.steps) if p2.steps else [None] * 16

        while len(steps1) < 16:
            steps1.append(None)
        steps1 = steps1[:16]

        while len(steps2) < 16:
            steps2.append(None)
        steps2 = steps2[:16]

        compound_steps = steps1 + steps2
        onset_grid = [i for i, s in enumerate(compound_steps) if s is not None]
        density = len(onset_grid) / 32.0

        return SequencerPattern(
            id=f"cmp_{stem}_{p1.track_id[:6]}_{p1.bar_num}_{p2.bar_num}",
            stem=stem,
            track_id=p1.track_id,
            pattern_name=f"{p1.pattern_name.split(' (')[0]} [32-Step Compound]",
            length=32,
            steps=compound_steps,
            energy=(p1.energy + p2.energy) / 2.0,
            density=density,
            onset_grid=onset_grid,
            groove_type=f"{p1.groove_type}_compound",
            bar_num=p1.bar_num,
            bpm=p1.bpm,
            key=p1.key,
        )
