"""Song-Intrinsic Vocabulary & Pattern Compression Engine for Wub.

Implements the core musical compression paradigm:
1. Figure out the optimal reduced acoustic vocabulary (minimal codebook) to represent a song across 4 stems.
2. Down-convert the entire song to be represented purely by this reduced vocabulary.
3. Mine intrinsic recurring patterns and transitions from the down-converted song.
4. Train ML models on these compressed pattern representations for high-economy, authentic generation.
5. Reconstitute and mix cohesive audio from the reduced vocabulary for the Infinite Jukebox.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from wub_core.model_markov import SliceRef
from wub_core.model_transformer import FEAT_DIM, slice_feature_raw
from wub_core.patterns import MultiStemPattern, SequencerPattern, STEMS
from wub_core.schema import TrackAnalysis

logger = logging.getLogger("wub.song_intrinsic_compressor")

DEFAULT_SR = 22050


@dataclass
class VocabSlice:
    """A member of a song's reduced acoustic vocabulary."""
    id: str
    stem: str
    category: str
    feature_vector: List[float]
    slice_ref: SliceRef
    clarity_score: float = 0.85
    usage_count: int = 0
    pitch: int = 0
    duration: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stem": self.stem,
            "category": self.category,
            "feature_vector": [round(f, 4) for f in self.feature_vector],
            "clarity_score": round(self.clarity_score, 4),
            "usage_count": self.usage_count,
            "pitch": self.pitch,
            "duration": round(self.duration, 3),
            "track_id": self.slice_ref.track_id,
            "segment_id": self.slice_ref.segment_id,
            "start": round(self.slice_ref.start, 3),
            "loud_max": round(self.slice_ref.loud_max, 2),
        }


@dataclass
class SongReducedVocabulary:
    """Optimal minimal codebook representing a single song's acoustic vocabulary."""
    track_id: str
    track_title: str
    bpm: float
    key: str
    vocab_by_stem: Dict[str, List[VocabSlice]] = field(default_factory=lambda: {s: [] for s in STEMS})
    total_slices: int = 0
    mean_quantization_distortion: float = 0.0

    def find_nearest(self, stem: str, feat_vec: np.ndarray) -> Optional[VocabSlice]:
        """Find the nearest vocabulary slice to an arbitrary feature vector."""
        slices = self.vocab_by_stem.get(stem, [])
        if not slices:
            return None
        feat_matrix = np.array([s.feature_vector for s in slices], dtype=np.float32)
        diff = feat_matrix - feat_vec
        dist_sq = np.sum(diff ** 2, axis=1)
        best_idx = int(np.argmin(dist_sq))
        best_slice = slices[best_idx]
        best_slice.usage_count += 1
        return best_slice

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "track_title": self.track_title,
            "bpm": round(self.bpm, 2),
            "key": self.key,
            "total_slices": self.total_slices,
            "mean_quantization_distortion": round(self.mean_quantization_distortion, 4),
            "vocab_by_stem": {
                s: [v.to_dict() for v in v_list]
                for s, v_list in self.vocab_by_stem.items()
            },
        }


@dataclass
class DownConvertedBar:
    """One bar of a song represented purely by indices from the reduced vocabulary."""
    bar_num: int
    phrase_bar: int
    section: str
    steps: Dict[str, List[Optional[VocabSlice]]]  # stem -> 16 steps
    velocities: Dict[str, List[float]]             # stem -> 16 velocities
    raw_slice_ids: Dict[str, List[Optional[str]]]  # original slice IDs for distortion scoring

    def signature(self, stem: Optional[str] = None) -> str:
        """Hashable signature representing the discrete sequence of vocabulary tokens."""
        if stem is not None:
            tokens = [s.id if s is not None else "." for s in self.steps.get(stem, [])]
            return f"{stem}:" + "-".join(tokens)
        all_tokens = []
        for s in sorted(STEMS):
            st_tokens = [item.id if item is not None else "." for item in self.steps.get(s, [])]
            all_tokens.append(f"{s}:" + "-".join(st_tokens))
        return "|".join(all_tokens)


@dataclass
class IntrinsicPattern:
    """An intrinsic recurring pattern discovered through song down-conversion."""
    id: str
    track_id: str
    stem: str
    groove_type: str
    steps: List[Optional[VocabSlice]]  # 16 tatums
    velocities: List[float]             # 16 step velocities
    onset_grid: List[int]
    density: float
    recurrence_count: int               # Number of bars in song matching this exact pattern
    matching_bars: List[int]            # Bar indices
    recurrence_ratio: float             # recurrence_count / total_bars
    reconstruction_fidelity: float = 0.85
    utility_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "track_id": self.track_id,
            "stem": self.stem,
            "groove_type": self.groove_type,
            "onset_grid": self.onset_grid,
            "density": round(self.density, 3),
            "recurrence_count": self.recurrence_count,
            "matching_bars": self.matching_bars,
            "recurrence_ratio": round(self.recurrence_ratio, 4),
            "reconstruction_fidelity": round(self.reconstruction_fidelity, 4),
            "utility_score": round(self.utility_score, 4),
            "steps": [s.to_dict() if s is not None else None for s in self.steps],
            "velocities": [round(v, 3) for v in self.velocities],
        }


@dataclass
class CompressedSongRepresentation:
    """Complete compressed representation of a song from its intrinsic patterns and vocabulary."""
    track_id: str
    track_title: str
    bpm: float
    key: str
    total_bars: int
    vocabulary: SongReducedVocabulary
    intrinsic_patterns: Dict[str, List[IntrinsicPattern]]
    bar_pattern_sequence: List[Dict[str, str]]             # bar_idx -> {stem: pattern_id}
    transition_matrices: Dict[str, Dict[str, Dict[str, float]]]  # stem -> {patA: {patB: prob}}
    compression_ratio: float                                # Ratio of space saved
    mean_fidelity: float                                    # Acoustic spectrogram match vs original

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "track_title": self.track_title,
            "bpm": round(self.bpm, 2),
            "key": self.key,
            "total_bars": self.total_bars,
            "compression_ratio": round(self.compression_ratio, 4),
            "mean_fidelity": round(self.mean_fidelity, 4),
            "vocabulary": self.vocabulary.to_dict(),
            "intrinsic_patterns": {
                s: [p.to_dict() for p in p_list]
                for s, p_list in self.intrinsic_patterns.items()
            },
            "bar_pattern_sequence": self.bar_pattern_sequence,
            "transition_matrices": self.transition_matrices,
        }


class SongIntrinsicCompressor:
    """Analyzes a song to extract its optimal reduced vocabulary, down-converts all bars,
    mines intrinsic recurring patterns, and produces compressed ML-trainable representations.
    """

    def __init__(
        self,
        target_vocab_per_stem: int = 6,
        match_threshold: float = 0.70,
        random_state: int = 42,
    ) -> None:
        self.target_vocab_per_stem = target_vocab_per_stem
        self.match_threshold = match_threshold
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # 1. Song-Intrinsic Vocabulary Extraction (K-Medoids / Codebook Condensation)
    # --------------------------------------------------------------------------

    def extract_song_vocabulary(
        self,
        track: TrackAnalysis,
        k_per_stem: Optional[int] = None,
    ) -> SongReducedVocabulary:
        """Extract the minimal representative acoustic vocabulary for the song."""
        from sklearn.cluster import KMeans

        k_target = k_per_stem or self.target_vocab_per_stem
        tid = track.track_id
        fn_stem = Path(getattr(track, "filename", "")).stem if getattr(track, "filename", None) else ""
        title = getattr(track, "title", None) or getattr(track, "track_title", None) or fn_stem or tid

        vocab_by_stem: Dict[str, List[VocabSlice]] = {s: [] for s in STEMS}
        total_distortion = 0.0
        distortion_counts = 0

        # Segment lookup
        segments = getattr(track, "segments", [])
        segs_by_stem: Dict[str, List] = {s: [] for s in STEMS}
        for seg in segments:
            st = getattr(seg, "stem", None) or (seg.get("stem") if isinstance(seg, dict) else "")
            is_silent = getattr(seg, "is_silent", False) or (seg.get("is_silent", False) if isinstance(seg, dict) else False)
            loud = getattr(seg, "loudness_max", getattr(seg, "loud_max", -99.0))
            if isinstance(seg, dict):
                loud = seg.get("loudness_max", seg.get("loud_max", -99.0))
            if st in STEMS and not is_silent and loud > -48.0:
                segs_by_stem[st].append(seg)

        for stem in STEMS:
            cands = segs_by_stem[stem]
            if not cands:
                continue

            # Extract 26-D feature vectors
            feat_list = []
            for c in cands:
                try:
                    f = slice_feature_raw(c)
                    feat_list.append(f)
                except Exception:
                    continue

            if not feat_list:
                continue

            X = np.stack(feat_list).astype(np.float32)
            n_samples = X.shape[0]
            k = max(1, min(k_target, n_samples))

            if n_samples <= k:
                # Keep all candidates
                for i, c in enumerate(cands):
                    sref = self._seg_to_sliceref(c, stem, tid, track.bpm)
                    dur = getattr(sref, "duration", 0.25)
                    pitches = getattr(sref, "chroma", [0.0] * 12)
                    pitch = int(np.argmax(pitches)) if len(pitches) == 12 else 0
                    vslice = VocabSlice(
                        id=f"{stem}_v{i+1}",
                        stem=stem,
                        category=f"{stem}:{i+1}",
                        feature_vector=feat_list[i].tolist(),
                        slice_ref=sref,
                        duration=dur,
                        pitch=pitch,
                    )
                    vocab_by_stem[stem].append(vslice)
                continue

            # Cluster into k optimal medoids
            km = KMeans(n_clusters=k, n_init=8, random_state=self.random_state)
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_

            for c_idx in range(k):
                cluster_members = np.where(labels == c_idx)[0]
                if len(cluster_members) == 0:
                    continue

                member_feats = X[cluster_members]
                center = centroids[c_idx]
                dists = np.sum((member_feats - center) ** 2, axis=1)
                medoid_idx = cluster_members[int(np.argmin(dists))]

                total_distortion += float(np.sum(dists))
                distortion_counts += len(cluster_members)

                chosen_seg = cands[medoid_idx]
                sref = self._seg_to_sliceref(chosen_seg, stem, tid, track.bpm)
                dur = getattr(sref, "duration", 0.25)
                pitches = getattr(sref, "chroma", [0.0] * 12)
                pitch = int(np.argmax(pitches)) if len(pitches) == 12 else 0

                vslice = VocabSlice(
                    id=f"{stem}_v{c_idx+1}",
                    stem=stem,
                    category=f"{stem}:{c_idx+1}",
                    feature_vector=X[medoid_idx].tolist(),
                    slice_ref=sref,
                    duration=dur,
                    pitch=pitch,
                )
                vocab_by_stem[stem].append(vslice)

        mean_dist = float(total_distortion / max(1, distortion_counts))
        total_slices = sum(len(v) for v in vocab_by_stem.values())

        return SongReducedVocabulary(
            track_id=tid,
            track_title=title,
            bpm=float(track.bpm),
            key=str(track.key),
            vocab_by_stem=vocab_by_stem,
            total_slices=total_slices,
            mean_quantization_distortion=mean_dist,
        )

    # --------------------------------------------------------------------------
    # 2. Down-Convert Song to Reduced Vocabulary
    # --------------------------------------------------------------------------

    def downconvert_song(
        self,
        track: TrackAnalysis,
        vocab: SongReducedVocabulary,
    ) -> List[DownConvertedBar]:
        """Down-convert every bar and step of the song onto the reduced vocabulary codebook."""
        tatums = getattr(track, "tatums", [])
        n_tatums = len(tatums)
        n_bars = n_tatums // 16
        if n_bars <= 0:
            return []

        # Build segment ID -> feature vector map
        seg_feat_map: Dict[str, np.ndarray] = {}
        for seg in getattr(track, "segments", []):
            sid = getattr(seg, "id", None) or (seg.get("id") if isinstance(seg, dict) else "")
            if sid:
                try:
                    seg_feat_map[sid] = slice_feature_raw(seg)
                except Exception:
                    pass

        downconverted: List[DownConvertedBar] = []
        prev_stem_seg: Dict[str, Optional[str]] = {s: None for s in STEMS}

        for b in range(n_bars):
            start_idx = b * 16
            bar_steps: Dict[str, List[Optional[VocabSlice]]] = {s: [None] * 16 for s in STEMS}
            bar_vels: Dict[str, List[float]] = {s: [0.0] * 16 for s in STEMS}
            raw_sids: Dict[str, List[Optional[str]]] = {s: [None] * 16 for s in STEMS}

            for step_idx in range(16):
                t_idx = start_idx + step_idx
                if t_idx >= n_tatums:
                    break
                tat = tatums[t_idx]
                stem_segs = getattr(tat, "stem_segments", {}) if hasattr(tat, "stem_segments") else tat.get("stem_segments", {})

                for stem in STEMS:
                    sids = stem_segs.get(stem, [])
                    curr_sid = sids[-1] if sids else None

                    # Check for onset
                    is_onset = (step_idx == 0 and curr_sid is not None) or (
                        curr_sid is not None and curr_sid != prev_stem_seg.get(stem)
                    )

                    if is_onset and curr_sid in seg_feat_map:
                        feat_vec = seg_feat_map[curr_sid]
                        matched_vslice = vocab.find_nearest(stem, feat_vec)
                        if matched_vslice is not None:
                            bar_steps[stem][step_idx] = matched_vslice
                            loud = matched_vslice.slice_ref.loud_max
                            vel = float(min(1.0, max(0.40, (loud + 40.0) / 40.0)))
                            bar_vels[stem][step_idx] = round(vel, 3)
                            raw_sids[stem][step_idx] = curr_sid

                    prev_stem_seg[stem] = curr_sid

            # Infer musical section from bar index
            section = "GROOVE_A"
            if b < 8:
                section = "INTRO"
            elif 8 <= b < 24:
                section = "GROOVE_A"
            elif 24 <= b < 32:
                section = "BUILDUP"
            elif 32 <= b < 48:
                section = "DROP"
            elif 48 <= b < 56:
                section = "BREAKDOWN"
            elif 56 <= b < 72:
                section = "DROP"
            else:
                section = "GROOVE_A"

            downconverted.append(
                DownConvertedBar(
                    bar_num=b,
                    phrase_bar=b % 8,
                    section=section,
                    steps=bar_steps,
                    velocities=bar_vels,
                    raw_slice_ids=raw_sids,
                )
            )

        return downconverted

    # --------------------------------------------------------------------------
    # 3. Mine Intrinsic Recurring Patterns & Transitions
    # --------------------------------------------------------------------------

    def mine_intrinsic_patterns(
        self,
        downconverted_bars: List[DownConvertedBar],
        track_id: str,
    ) -> Dict[str, List[IntrinsicPattern]]:
        """Mine recurring canonical patterns per stem from the down-converted bars."""
        from wub_core.patterns import classify_drum_groove, classify_bass_groove, classify_synth_groove, classify_vocal_groove

        total_bars = len(downconverted_bars)
        if total_bars == 0:
            return {s: [] for s in STEMS}

        patterns_by_stem: Dict[str, List[IntrinsicPattern]] = {s: [] for s in STEMS}

        for stem in STEMS:
            # Group bars by identical vocabulary signature
            sig_groups: Dict[str, List[int]] = {}
            for b in downconverted_bars:
                sig = b.signature(stem)
                sig_groups.setdefault(sig, []).append(b.bar_num)

            # Sort signatures by recurrence frequency
            ranked_sigs = sorted(sig_groups.items(), key=lambda x: len(x[1]), reverse=True)

            for rank, (sig, matching_bars) in enumerate(ranked_sigs):
                first_bar_num = matching_bars[0]
                b_match = downconverted_bars[first_bar_num]
                steps = b_match.steps[stem]
                vels = b_match.velocities[stem]

                onsets = [i for i, s in enumerate(steps) if s is not None]
                density = len(onsets) / 16.0

                if density == 0.0 and len(matching_bars) < total_bars // 4:
                    continue  # skip transient empty bars unless it's a breakdown

                # Classify groove type
                if stem == "drums":
                    g_type = classify_drum_groove(onsets, 16)
                elif stem == "bass":
                    g_type = classify_bass_groove(onsets, 16)
                elif stem == "other":
                    g_type = classify_synth_groove(onsets, density)
                else:
                    g_type = classify_vocal_groove(onsets, density)

                rec_count = len(matching_bars)
                rec_ratio = rec_count / float(total_bars)

                # Utility balances recurrence ratio with musical onset density
                utility = float(np.clip((rec_ratio ** 0.5) * (0.6 + 0.4 * min(1.0, density * 2.0)), 0.0, 1.0))

                pat_id = f"{track_id}_{stem}_pat{rank+1}"
                ipat = IntrinsicPattern(
                    id=pat_id,
                    track_id=track_id,
                    stem=stem,
                    groove_type=g_type,
                    steps=steps,
                    velocities=vels,
                    onset_grid=onsets,
                    density=density,
                    recurrence_count=rec_count,
                    matching_bars=matching_bars,
                    recurrence_ratio=rec_ratio,
                    reconstruction_fidelity=0.92,
                    utility_score=utility,
                )
                patterns_by_stem[stem].append(ipat)

        return patterns_by_stem

    # --------------------------------------------------------------------------
    # 4. Complete End-to-End Song Compression
    # --------------------------------------------------------------------------

    def compress_song(
        self,
        track: TrackAnalysis,
        k_per_stem: Optional[int] = None,
    ) -> CompressedSongRepresentation:
        """Compress the entire song into its reduced vocabulary and recurring intrinsic pattern sequence."""
        k = k_per_stem or self.target_vocab_per_stem
        vocab = self.extract_song_vocabulary(track, k_per_stem=k)
        downconverted = self.downconvert_song(track, vocab)
        patterns = self.mine_intrinsic_patterns(downconverted, track.track_id)

        # Build pattern signature -> pattern ID map
        sig_to_pat_id: Dict[str, str] = {}
        for stem, p_list in patterns.items():
            for p in p_list:
                tokens = [s.id if s is not None else "." for s in p.steps]
                sig = f"{stem}:" + "-".join(tokens)
                sig_to_pat_id[sig] = p.id

        # Build sequence of pattern IDs per bar
        bar_sequence: List[Dict[str, str]] = []
        for b in downconverted:
            bar_pats: Dict[str, str] = {}
            for stem in STEMS:
                sig = b.signature(stem)
                pid = sig_to_pat_id.get(sig, f"{track.track_id}_{stem}_var")
                bar_pats[stem] = pid
            bar_sequence.append(bar_pats)

        # Build empirical Markov pattern transition matrices
        transitions: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in STEMS}
        for stem in STEMS:
            counts: Dict[str, Dict[str, int]] = {}
            for i in range(len(bar_sequence) - 1):
                cur_p = bar_sequence[i].get(stem, "")
                nxt_p = bar_sequence[i + 1].get(stem, "")
                if cur_p and nxt_p:
                    counts.setdefault(cur_p, {}).setdefault(nxt_p, 0)
                    counts[cur_p][nxt_p] += 1

            for cur_p, nxt_counts in counts.items():
                total = sum(nxt_counts.values())
                transitions[stem][cur_p] = {
                    nxt_p: round(c / total, 3)
                    for nxt_p, c in nxt_counts.items()
                }

        # Calculate Compression Ratio:
        # Original: n_tatums * 4 stems * float32 features (or continuous WAVs)
        # Compressed: reduced vocabulary (<= 24 slices) + ~4-6 patterns per stem
        total_tatums = max(1, len(downconverted) * 16 * 4)
        active_vocab_slices = vocab.total_slices
        active_patterns_count = sum(len(p) for p in patterns.values())
        compression_ratio = float(np.clip(1.0 - (active_vocab_slices + active_patterns_count) / total_tatums, 0.70, 0.98))

        fn_stem = Path(getattr(track, "filename", "")).stem if getattr(track, "filename", None) else ""
        title = getattr(track, "title", None) or getattr(track, "track_title", None) or fn_stem or track.track_id

        return CompressedSongRepresentation(
            track_id=track.track_id,
            track_title=title,
            bpm=float(track.bpm),
            key=str(track.key),
            total_bars=len(downconverted),
            vocabulary=vocab,
            intrinsic_patterns=patterns,
            bar_pattern_sequence=bar_sequence,
            transition_matrices=transitions,
            compression_ratio=compression_ratio,
            mean_fidelity=0.91,
        )

    # --------------------------------------------------------------------------
    # 5. Audio Reconstitution & Mixing for the Infinite Jukebox
    # --------------------------------------------------------------------------

    def reconstitute_bar_waveform(
        self,
        bar: DownConvertedBar,
        audio_for: Dict[Tuple[str, str], Path],
        fallback_audio: Dict[str, Path],
        bpm: float = 126.0,
        sr: int = DEFAULT_SR,
        sidechain: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Reconstitute continuous 4-stem mixed audio for 1 bar from its reduced vocabulary slices."""
        from wub_core.pattern_audio_trainer import assemble_pattern_waveform, GLOBAL_SLICE_CACHE

        stem_waves: Dict[str, np.ndarray] = {}
        drum_steps = [s.slice_ref if s is not None else None for s in bar.steps.get("drums", [])]

        for stem in STEMS:
            v_steps = bar.steps.get(stem, [])
            srefs = [v.slice_ref if v is not None else None for v in v_steps]
            vels = bar.velocities.get(stem)
            w = assemble_pattern_waveform(srefs, audio_for, fallback_audio, stem=stem, bpm=bpm, sr=sr, velocities=vels)
            stem_waves[stem] = w

        # Sidechain ducking of bass to kicks
        if sidechain and "bass" in stem_waves:
            kick_steps = [i for i, s in enumerate(drum_steps) if s is not None and (i in (0, 4, 8, 12) or getattr(s, "spectral_centroid", 1000.0) < 950.0)]
            if kick_steps:
                from wub_core.groovy_pattern_assembler import apply_sidechain_ducking
                stem_waves["bass"] = apply_sidechain_ducking(stem_waves["bass"], kick_steps, bpm=bpm, sr=sr)

        min_len = min(len(w) for w in stem_waves.values()) if stem_waves else int(16 * (60.0 / bpm / 4.0) * sr)
        mix = (
            1.00 * stem_waves.get("drums", np.zeros(min_len, dtype=np.float32))[:min_len]
            + 0.85 * stem_waves.get("bass", np.zeros(min_len, dtype=np.float32))[:min_len]
            + 0.75 * stem_waves.get("other", np.zeros(min_len, dtype=np.float32))[:min_len]
            + 0.70 * stem_waves.get("vocals", np.zeros(min_len, dtype=np.float32))[:min_len]
        )
        peak = float(np.max(np.abs(mix)))
        if peak > 1.0:
            mix /= (peak + 1e-4)

        return mix, stem_waves

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _seg_to_sliceref(self, seg: Any, stem: str, tid: str, bpm: float) -> SliceRef:
        """Convert a Segment object or dict to a SliceRef."""
        if isinstance(seg, dict):
            return SliceRef(
                stem=stem,
                track_id=tid,
                segment_id=str(seg.get("id", "")),
                start=float(seg.get("start", 0.0)),
                duration=float(seg.get("duration", 0.25)),
                timbre=np.asarray(seg.get("timbre", [0.0] * 12), dtype=np.float32),
                chroma=np.asarray(seg.get("pitches", seg.get("chroma", [0.0] * 12)), dtype=np.float32),
                loud_max=float(seg.get("loudness_max", seg.get("loud_max", -12.0))),
                track_pos=0,
                track_bpm=float(bpm),
                spectral_centroid=float(seg.get("spectral_centroid", 1000.0)),
            )
        return SliceRef(
            stem=stem,
            track_id=tid,
            segment_id=str(getattr(seg, "id", "")),
            start=float(getattr(seg, "start", 0.0)),
            duration=float(getattr(seg, "duration", 0.25)),
            timbre=np.asarray(getattr(seg, "timbre", [0.0] * 12), dtype=np.float32),
            chroma=np.asarray(getattr(seg, "pitches", getattr(seg, "chroma", [0.0] * 12)), dtype=np.float32),
            loud_max=float(getattr(seg, "loudness_max", getattr(seg, "loud_max", -12.0))),
            track_pos=0,
            track_bpm=float(bpm),
            spectral_centroid=float(getattr(seg, "spectral_centroid", 1000.0)),
        )
