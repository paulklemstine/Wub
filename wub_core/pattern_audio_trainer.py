"""Pattern Audio Training & Real-Audio Acoustic Scoring Engine for Wub.

Increases feedback in training by:
1. Assembling slices of a candidate pattern into a continuous audio waveform (mono or 4-stem mix).
2. Extracting corresponding authentic reference audio bars from real training songs.
3. Scoring acoustic similarity using multi-scale spectral distance, rhythmic onset correlation,
   harmonic chroma matching, and dynamic range profiles.
4. Feeding audio similarity scores back into pattern ranking, slice selection probabilities,
   and generic good-sounding pattern generation.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from wub_core.audio_utils import equal_power_crossfade
from wub_core.model_markov import SliceRef
from wub_core.patterns import MultiStemPattern, PatternBank, SequencerPattern
from wub_core.schema import STEMS, TrackAnalysis

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SR = 22050
FEEDBACK_FILE = ROOT_DIR / "data" / "pattern_audio_feedback.json"


@dataclass
class AudioMatchMetrics:
    """Acoustic match metrics between synthesized pattern audio and real training song audio."""
    composite_score: float  # Normalized 0.0 .. 1.0 (higher = closer match to real music)
    spectral_similarity: float  # Cosine similarity of log-Mel spectrograms (0.0 .. 1.0)
    onset_correlation: float  # Pearson correlation of rhythmic onset envelope (0.0 .. 1.0)
    chroma_similarity: float  # Pitch-class harmonic similarity (0.0 .. 1.0)
    dynamic_match: float  # RMS dynamic envelope similarity (0.0 .. 1.0)
    stem: str = "master"
    groove_type: str = "generic"
    ref_track_id: Optional[str] = None
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 4),
            "spectral_similarity": round(self.spectral_similarity, 4),
            "onset_correlation": round(self.onset_correlation, 4),
            "chroma_similarity": round(self.chroma_similarity, 4),
            "dynamic_match": round(self.dynamic_match, 4),
            "stem": self.stem,
            "groove_type": self.groove_type,
            "ref_track_id": self.ref_track_id,
            "evaluated_at": self.evaluated_at,
        }


class SliceAudioCache:
    """Fast in-memory cache for loaded audio slices to accelerate synthesis and search."""

    def __init__(self, capacity: int = 1500) -> None:
        self.capacity = capacity
        self._cache: Dict[Tuple[str, float, float, int], np.ndarray] = {}
        self._order: List[Tuple[str, float, float, int]] = []

    def get_slice(self, audio_path: Path, start: float, duration: float, sr: int = DEFAULT_SR) -> Optional[np.ndarray]:
        key = (str(audio_path), round(start, 4), round(duration, 4), sr)
        if key in self._cache:
            return self._cache[key]

        if not audio_path.exists():
            return None

        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            src_sr = info.samplerate
            frames = max(1, int(duration * src_sr))
            start_frame = max(0, int(start * src_sr))
            if start_frame >= info.frames:
                return None

            frames = min(frames, info.frames - start_frame)
            data, _ = sf.read(str(audio_path), start=start_frame, frames=frames, always_2d=True, dtype="float32")
            mono = data.mean(axis=1)

            if src_sr != sr:
                import librosa
                mono = librosa.resample(mono, orig_sr=src_sr, target_sr=sr)

            y = np.asarray(mono, dtype=np.float32)

            # Insert into cache with eviction
            if len(self._cache) >= self.capacity:
                oldest = self._order.pop(0)
                self._cache.pop(oldest, None)

            self._cache[key] = y
            self._order.append(key)
            return y
        except Exception:
            return None


GLOBAL_SLICE_CACHE = SliceAudioCache()


def assemble_pattern_waveform(
    steps: List[Optional[SliceRef]],
    audio_for: Dict[Tuple[str, str], Path],
    fallback_audio: Dict[str, Path],
    stem: str,
    bpm: float = 126.0,
    sr: int = DEFAULT_SR,
    cache: Optional[SliceAudioCache] = None,
) -> np.ndarray:
    """Assemble individual slices of a pattern into a continuous audio waveform across 1 or 2 bars."""
    if cache is None:
        cache = GLOBAL_SLICE_CACHE

    n_steps = len(steps) if steps else 16
    tatum_sec = (60.0 / bpm / 4.0) if bpm > 0 else 0.119
    total_samples = int(n_steps * tatum_sec * sr)
    if total_samples <= 0:
        return np.zeros(int(16 * 0.119 * sr), dtype=np.float32)

    wave = np.zeros(total_samples, dtype=np.float32)
    step_samples = int(tatum_sec * sr)
    fade_samples = max(8, int(sr * 0.003))  # 3ms micro-crossfade

    for i, slice_ref in enumerate(steps):
        if slice_ref is None:
            continue

        p = audio_for.get((slice_ref.track_id, stem), fallback_audio.get(slice_ref.track_id))
        if p is None or not p.exists():
            continue

        slice_data = cache.get_slice(p, slice_ref.start, slice_ref.duration, sr=sr)
        if slice_data is None or len(slice_data) == 0:
            continue

        start_idx = i * step_samples
        if start_idx >= total_samples:
            break

        # Apply short fade in/out envelope to avoid boundary clicks
        y = slice_data.copy()
        if len(y) > fade_samples * 2:
            ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            y[:fade_samples] *= ramp
            y[-fade_samples:] *= ramp[::-1]

        end_idx = min(total_samples, start_idx + len(y))
        chunk_len = end_idx - start_idx
        if chunk_len > 0:
            wave[start_idx:end_idx] += y[:chunk_len]

    # Peak normalization with headroom
    peak = float(np.max(np.abs(wave)))
    if peak > 1.0:
        wave /= (peak + 1e-4)

    return wave


def assemble_multistem_waveform(
    multistem_steps: Dict[str, List[Optional[SliceRef]]],
    audio_for: Dict[Tuple[str, str], Path],
    fallback_audio: Dict[str, Path],
    bpm: float = 126.0,
    sr: int = DEFAULT_SR,
    stem_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Assemble all 4 stems into individual waveforms and a summed master audio wave."""
    if stem_weights is None:
        stem_weights = {"drums": 1.0, "bass": 0.85, "other": 0.75, "vocals": 0.70}

    stem_waves: Dict[str, np.ndarray] = {}
    master_wave: Optional[np.ndarray] = None

    for stem in STEMS:
        steps = multistem_steps.get(stem, [])
        if not steps:
            continue
        w = assemble_pattern_waveform(steps, audio_for, fallback_audio, stem=stem, bpm=bpm, sr=sr)
        stem_waves[stem] = w
        scaled = w * float(stem_weights.get(stem, 0.8))
        if master_wave is None:
            master_wave = np.zeros_like(scaled)
        if len(scaled) == len(master_wave):
            master_wave += scaled

    if master_wave is None:
        master_wave = np.zeros(int(16 * (60.0 / bpm / 4.0) * sr), dtype=np.float32)

    peak = float(np.max(np.abs(master_wave)))
    if peak > 0.95:
        master_wave *= (0.95 / (peak + 1e-4))

    return master_wave, stem_waves


def extract_reference_bar_audio(
    track_id: str,
    stem: str,
    bar_num: int,
    analysis: TrackAnalysis,
    audio_for: Dict[Tuple[str, str], Path],
    fallback_audio: Dict[str, Path],
    sr: int = DEFAULT_SR,
) -> Optional[np.ndarray]:
    """Extract an authentic 1-bar audio chunk from a real training song stem."""
    p = audio_for.get((track_id, stem), fallback_audio.get(track_id))
    if p is None or not p.exists():
        return None

    tatums = analysis.tatums
    start_step = bar_num * 16
    if start_step >= len(tatums):
        return None

    start_sec = tatums[start_step].start
    end_step = min(len(tatums) - 1, start_step + 15)
    end_sec = tatums[end_step].start + tatums[end_step].duration
    bar_duration = max(0.5, end_sec - start_sec)

    return GLOBAL_SLICE_CACHE.get_slice(p, start_sec, bar_duration, sr=sr)


def compute_audio_similarity(
    synth_wave: np.ndarray,
    real_wave: np.ndarray,
    sr: int = DEFAULT_SR,
    is_pitched: bool = False,
    stem: str = "drums",
    groove_type: str = "generic",
    ref_track_id: Optional[str] = None,
) -> AudioMatchMetrics:
    """Compute acoustic similarity between synthesized pattern audio and real training song audio.

    Measures:
    1. Multi-scale log-Mel spectrogram cosine similarity (spectral shape, EQ, timbral match).
    2. Rhythmic onset envelope Pearson correlation (pocket, groove, attack alignment).
    3. Harmonic chroma cosine similarity (pitch class alignment for bass, vocals, synths).
    4. Dynamic RMS envelope profile similarity across 16 steps.
    """
    import librosa

    # Align length to the shorter of the two
    min_len = min(len(synth_wave), len(real_wave))
    if min_len < int(0.25 * sr):
        return AudioMatchMetrics(
            composite_score=0.2,
            spectral_similarity=0.2,
            onset_correlation=0.0,
            chroma_similarity=0.2,
            dynamic_match=0.2,
            stem=stem,
            groove_type=groove_type,
            ref_track_id=ref_track_id,
        )

    s_synth = synth_wave[:min_len]
    s_real = real_wave[:min_len]

    # Silence check
    rms_synth = float(np.sqrt(np.mean(s_synth**2) + 1e-9))
    rms_real = float(np.sqrt(np.mean(s_real**2) + 1e-9))
    if rms_synth < 1e-4 or rms_real < 1e-4:
        return AudioMatchMetrics(
            composite_score=0.1,
            spectral_similarity=0.1,
            onset_correlation=0.0,
            chroma_similarity=0.1,
            dynamic_match=0.1,
            stem=stem,
            groove_type=groove_type,
            ref_track_id=ref_track_id,
        )

    # 1. Multi-Scale Log-Mel Spectrogram Similarity
    n_fft = 512
    hop_length = 128
    m_synth = librosa.feature.melspectrogram(y=s_synth, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=40)
    m_real = librosa.feature.melspectrogram(y=s_real, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=40)

    log_m_synth = np.log1p(np.maximum(0.0, m_synth))
    log_m_real = np.log1p(np.maximum(0.0, m_real))

    norm_synth = np.linalg.norm(log_m_synth)
    norm_real = np.linalg.norm(log_m_real)
    if norm_synth > 1e-6 and norm_real > 1e-6:
        spec_sim = float(np.sum(log_m_synth * log_m_real) / (norm_synth * norm_real))
        spec_sim = float(np.clip(spec_sim, 0.0, 1.0))
    else:
        spec_sim = 0.5

    # 2. Rhythmic Onset Envelope Pearson Correlation
    onset_synth = librosa.onset.onset_strength(y=s_synth, sr=sr, hop_length=hop_length)
    onset_real = librosa.onset.onset_strength(y=s_real, sr=sr, hop_length=hop_length)
    min_onset = min(len(onset_synth), len(onset_real))
    os1 = onset_synth[:min_onset]
    os2 = onset_real[:min_onset]

    std1 = float(np.std(os1))
    std2 = float(np.std(os2))
    if std1 > 1e-5 and std2 > 1e-5:
        corr = float(np.corrcoef(os1, os2)[0, 1])
        onset_corr = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))  # Map [-1, 1] to [0, 1]
    else:
        onset_corr = 0.5

    # 3. Harmonic Chroma Cosine Similarity (Pitched stems or master)
    chroma_sim = 0.5
    if is_pitched or stem in ("bass", "other", "vocals", "master"):
        try:
            fft_size = min(512, len(s_synth), len(s_real))
            if fft_size >= 128:
                c_synth = librosa.feature.chroma_stft(y=s_synth, sr=sr, n_fft=fft_size, hop_length=hop_length, n_chroma=12)
                c_real = librosa.feature.chroma_stft(y=s_real, sr=sr, n_fft=fft_size, hop_length=hop_length, n_chroma=12)
                min_c = min(c_synth.shape[1], c_real.shape[1])
                if min_c > 2:
                    cs1 = c_synth[:, :min_c]
                    cs2 = c_real[:, :min_c]
                    dots = np.sum(cs1 * cs2, axis=0)
                    norms = (np.linalg.norm(cs1, axis=0) * np.linalg.norm(cs2, axis=0)) + 1e-6
                    chroma_sim = float(np.clip(np.mean(dots / norms), 0.0, 1.0))
        except Exception:
            chroma_sim = 0.5

    # 4. Dynamic RMS Envelope Profile Match (16 bins)
    bin_size = max(1, min_len // 16)
    rms_bins_synth = np.array([np.sqrt(np.mean(s_synth[i * bin_size : (i + 1) * bin_size] ** 2) + 1e-8) for i in range(16)])
    rms_bins_real = np.array([np.sqrt(np.mean(s_real[i * bin_size : (i + 1) * bin_size] ** 2) + 1e-8) for i in range(16)])
    rms_bins_synth /= (np.max(rms_bins_synth) + 1e-6)
    rms_bins_real /= (np.max(rms_bins_real) + 1e-6)
    rms_diff = float(np.mean(np.abs(rms_bins_synth - rms_bins_real)))
    dynamic_match = float(np.clip(1.0 - rms_diff, 0.0, 1.0))

    # 5. Composite Match Score
    if stem == "drums":
        composite = 0.45 * spec_sim + 0.45 * onset_corr + 0.10 * dynamic_match
    elif stem in ("bass", "other", "vocals"):
        composite = 0.35 * spec_sim + 0.25 * onset_corr + 0.30 * chroma_sim + 0.10 * dynamic_match
    else:  # master or full mix
        composite = 0.40 * spec_sim + 0.30 * onset_corr + 0.20 * chroma_sim + 0.10 * dynamic_match

    composite = float(np.clip(composite, 0.0, 1.0))

    return AudioMatchMetrics(
        composite_score=composite,
        spectral_similarity=spec_sim,
        onset_correlation=onset_corr,
        chroma_similarity=chroma_sim,
        dynamic_match=dynamic_match,
        stem=stem,
        groove_type=groove_type,
        ref_track_id=ref_track_id,
    )


class GenericPatternOptimizer:
    """Finds generic good-sounding patterns by assembling slices and scoring against real song audio."""

    def __init__(self, sr: int = DEFAULT_SR, seed: int = 42) -> None:
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    def optimize_pattern(
        self,
        stem: str,
        groove_type: str,
        slice_palette: Dict[str, List[SliceRef]],
        reference_audios: List[np.ndarray],
        audio_for: Dict[Tuple[str, str], Path],
        fallback_audio: Dict[str, Path],
        bpm: float = 126.0,
        iterations: int = 15,
    ) -> Tuple[SequencerPattern, AudioMatchMetrics]:
        """Assemble candidate slice sequences, score against real song audio, and iteratively optimize."""
        from wub_core.patterns import classify_drum_groove

        # Initial default placement based on groove type
        steps: List[Optional[SliceRef]] = [None] * 16

        def _sample_pool(category: str) -> Optional[SliceRef]:
            pool = slice_palette.get(category, [])
            if pool:
                return pool[int(self.rng.choice(len(pool)))]
            # Fallback to any slice in palette
            for p in slice_palette.values():
                if p:
                    return p[int(self.rng.choice(len(p)))]
            return None

        # Seed initial onsets based on dance archetypes
        if stem == "drums":
            for k_step in (0, 8):
                steps[k_step] = _sample_pool("kick")
            for s_step in (4, 12):
                steps[s_step] = _sample_pool("snare")
            for h_step in (2, 6, 10, 14):
                steps[h_step] = _sample_pool("hat")
            for c_step in (1, 3, 5, 7, 9, 11, 13, 15):
                steps[c_step] = _sample_pool("hat") or _sample_pool("fill")
        elif stem == "bass":
            sub_s = _sample_pool("sub")
            roll_s = _sample_pool("rolling") or sub_s
            for ob in (2, 6, 10, 14):
                steps[ob] = sub_s
            for p_step in (3, 7, 11, 15):
                steps[p_step] = roll_s
        elif stem == "other":
            chord_s = _sample_pool("chord")
            arp_s = _sample_pool("arp") or chord_s
            for m_step in (0, 2, 4, 6, 8, 10, 12, 14):
                steps[m_step] = chord_s if (m_step % 4 == 0) else arp_s
        else:  # vocals
            chop_s = _sample_pool("chop")
            for v_step in (2, 6, 10, 14):
                steps[v_step] = chop_s

        # Score initial assembly
        best_steps = list(steps)
        synth_wave = assemble_pattern_waveform(best_steps, audio_for, fallback_audio, stem=stem, bpm=bpm, sr=self.sr)

        def _evaluate_wave(w: np.ndarray) -> AudioMatchMetrics:
            if not reference_audios:
                # Synthetic baseline reference
                return AudioMatchMetrics(0.70, 0.70, 0.70, 0.70, 0.70, stem=stem, groove_type=groove_type)
            scores = []
            for ref in reference_audios[:3]:
                m = compute_audio_similarity(w, ref, sr=self.sr, stem=stem, groove_type=groove_type)
                scores.append(m)
            # Pick median/best representation
            scores.sort(key=lambda x: x.composite_score, reverse=True)
            return scores[0]

        best_metrics = _evaluate_wave(synth_wave)

        # Iterative feedback-driven slice mutation
        pools_list = list(slice_palette.keys())
        for _ in range(iterations):
            cand_steps = list(best_steps)
            # Pick a random active step to mutate
            active_indices = [i for i, s in enumerate(cand_steps) if s is not None]
            if not active_indices:
                break

            target_idx = int(self.rng.choice(active_indices))
            # Choose pool corresponding to step or random pool
            pool_key = pools_list[int(self.rng.choice(len(pools_list)))] if pools_list else "kick"
            new_slice = _sample_pool(pool_key)
            if new_slice is None:
                continue

            cand_steps[target_idx] = new_slice
            cand_wave = assemble_pattern_waveform(cand_steps, audio_for, fallback_audio, stem=stem, bpm=bpm, sr=self.sr)
            cand_metrics = _evaluate_wave(cand_wave)

            # Accept if score improves or equal
            if cand_metrics.composite_score >= best_metrics.composite_score:
                best_steps = cand_steps
                best_metrics = cand_metrics

        onset_grid = [i for i, s in enumerate(best_steps) if s is not None]
        density = len(onset_grid) / float(len(best_steps))

        pat = SequencerPattern(
            id=f"optimized_{stem}_{groove_type}_{int(self.rng.integers(1000, 9999))}",
            stem=stem,
            track_id="trained_generic_model",
            pattern_name=f"Audio-Optimized {groove_type.title()} ({stem})",
            length=16,
            steps=best_steps,
            energy=0.7 if stem == "drums" else 0.6,
            density=density,
            onset_grid=onset_grid,
            groove_type=groove_type,
            bar_num=0,
            bpm=bpm,
            key="C major",
            audio_similarity_score=best_metrics.composite_score,
        )

        return pat, best_metrics


class PatternAudioFeedbackEngine:
    """Coordinates feedback between pattern audio synthesis and real song training data."""

    def __init__(self, feedback_path: Path = FEEDBACK_FILE) -> None:
        self.feedback_path = feedback_path
        self.feedback_data: Dict[str, Any] = self._load()
        self.optimizer = GenericPatternOptimizer()

    def _load(self) -> Dict[str, Any]:
        if self.feedback_path.exists():
            try:
                return json.loads(self.feedback_path.read_text())
            except Exception:
                pass
        return {
            "version": 1,
            "patterns": {},  # pat_id -> metrics dict
            "slices": {},    # slice_id -> score float
            "tracks": {},    # track_id -> score float
            "grooves": {},   # stem -> {groove_type: mean_score}
            "eval_count": 0,
            "last_updated": time.time(),
        }

    def _save(self) -> None:
        try:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            self.feedback_data["last_updated"] = time.time()
            self.feedback_path.write_text(json.dumps(self.feedback_data, indent=2))
        except Exception:
            pass

    def record_metrics(self, pattern_id: str, metrics: AudioMatchMetrics, steps: List[Optional[SliceRef]]) -> None:
        """Record evaluation feedback for a pattern and reward/penalize constituent slices."""
        self.feedback_data["patterns"][pattern_id] = metrics.to_dict()
        self.feedback_data["eval_count"] = self.feedback_data.get("eval_count", 0) + 1

        # Track score
        if metrics.ref_track_id:
            cur_t = self.feedback_data["tracks"].get(metrics.ref_track_id, 0.5)
            self.feedback_data["tracks"][metrics.ref_track_id] = round(0.85 * cur_t + 0.15 * metrics.composite_score, 4)

        # Slice scores: reward slices participating in high-scoring audio patterns
        for s in steps:
            if s is not None and s.segment_id:
                cur_s = self.feedback_data["slices"].get(s.segment_id, 0.5)
                # Exponential moving average update
                self.feedback_data["slices"][s.segment_id] = round(0.80 * cur_s + 0.20 * metrics.composite_score, 4)

        # Groove archetype benchmark
        stem_grooves = self.feedback_data["grooves"].setdefault(metrics.stem, {})
        cur_g = stem_grooves.get(metrics.groove_type, 0.5)
        stem_grooves[metrics.groove_type] = round(0.85 * cur_g + 0.15 * metrics.composite_score, 4)

    def evaluate_pattern(
        self,
        pat: SequencerPattern,
        corpus_analyses: Dict[str, TrackAnalysis],
        audio_for: Dict[Tuple[str, str], Path],
        fallback_audio: Dict[str, Path],
        sr: int = DEFAULT_SR,
    ) -> AudioMatchMetrics:
        """Synthesize pattern audio waveform and score against real audio from the source track."""
        synth_wave = assemble_pattern_waveform(pat.steps, audio_for, fallback_audio, stem=pat.stem, bpm=pat.bpm, sr=sr)

        # Look up real audio bar from source track if available
        real_wave: Optional[np.ndarray] = None
        if pat.track_id in corpus_analyses:
            an = corpus_analyses[pat.track_id]
            real_wave = extract_reference_bar_audio(
                track_id=pat.track_id,
                stem=pat.stem,
                bar_num=pat.bar_num,
                analysis=an,
                audio_for=audio_for,
                fallback_audio=fallback_audio,
                sr=sr,
            )

        if real_wave is None:
            # Fall back to any reference stem wave from the corpus
            for tid, an in corpus_analyses.items():
                if pat.stem in an.stems_available:
                    rw = extract_reference_bar_audio(
                        track_id=tid,
                        stem=pat.stem,
                        bar_num=1,
                        analysis=an,
                        audio_for=audio_for,
                        fallback_audio=fallback_audio,
                        sr=sr,
                    )
                    if rw is not None and np.max(np.abs(rw)) > 0.05:
                        real_wave = rw
                        break

        if real_wave is None:
            # Synthetic baseline match
            metrics = AudioMatchMetrics(
                composite_score=0.75,
                spectral_similarity=0.75,
                onset_correlation=0.75,
                chroma_similarity=0.75,
                dynamic_match=0.75,
                stem=pat.stem,
                groove_type=pat.groove_type,
                ref_track_id=pat.track_id,
            )
        else:
            metrics = compute_audio_similarity(
                synth_wave=synth_wave,
                real_wave=real_wave,
                sr=sr,
                stem=pat.stem,
                groove_type=pat.groove_type,
                ref_track_id=pat.track_id,
            )

        pat.audio_similarity_score = metrics.composite_score
        self.record_metrics(pat.id, metrics, pat.steps)
        return metrics

    def run_training_step(
        self,
        bank: PatternBank,
        corpus_store: Any,
        n_optimize: int = 8,
        sr: int = DEFAULT_SR,
    ) -> Dict[str, Any]:
        """Run a full audio feedback training iteration:

        1. Evaluates existing patterns against real audio tracks.
        2. Optimizes generic patterns using slice assembly + real audio matching.
        3. Updates feedback weights and saves state.
        """
        analyses = corpus_store.analyses
        if not analyses:
            return {"status": "empty_corpus", "evaluated": 0}

        audio_for, fallback = corpus_store.audio_for(set(analyses.keys()))

        # Collect reference audios per stem from real tracks
        stem_refs: Dict[str, List[np.ndarray]] = {s: [] for s in STEMS}
        for tid, an in list(analyses.items())[:10]:
            for stem in an.stems_available:
                if len(stem_refs[stem]) < 4:
                    ref = extract_reference_bar_audio(tid, stem, 1, an, audio_for, fallback, sr=sr)
                    if ref is not None and np.max(np.abs(ref)) > 0.05:
                        stem_refs[stem].append(ref)

        evaluated = 0
        total_sim = 0.0

        # 1. Evaluate patterns in bank
        for stem in STEMS:
            pats = bank.patterns_by_stem.get(stem, [])
            for pat in pats[:12]:
                m = self.evaluate_pattern(pat, analyses, audio_for, fallback, sr=sr)
                total_sim += m.composite_score
                evaluated += 1

        # 2. Find and optimize generic good-sounding patterns
        optimized_count = 0
        archetypes = {
            "drums": ["four_on_floor_hats", "breakbeat", "fill"],
            "bass": ["offbeat", "rolling", "stabs"],
            "other": ["chords", "riff", "pad"],
            "vocals": ["chop", "phrase"],
        }

        for stem, types in archetypes.items():
            refs = stem_refs.get(stem, [])
            pal = bank.slice_palette.get(stem, {})
            for gt in types:
                if optimized_count >= n_optimize:
                    break
                opt_pat, opt_metrics = self.optimizer.optimize_pattern(
                    stem=stem,
                    groove_type=gt,
                    slice_palette=pal,
                    reference_audios=refs,
                    audio_for=audio_for,
                    fallback_audio=fallback,
                    bpm=126.0,
                    iterations=12,
                )
                bank.add_pattern(opt_pat)
                self.record_metrics(opt_pat.id, opt_metrics, opt_pat.steps)
                total_sim += opt_metrics.composite_score
                evaluated += 1
                optimized_count += 1

        self._save()

        mean_sim = round(total_sim / max(1, evaluated), 4)
        return {
            "status": "ok",
            "evaluated": evaluated,
            "optimized_generic_patterns": optimized_count,
            "mean_audio_similarity": mean_sim,
            "feedback_db_entries": len(self.feedback_data.get("patterns", {})),
        }
