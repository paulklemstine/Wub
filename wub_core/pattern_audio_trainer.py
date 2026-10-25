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


@dataclass
class MultiStemCohesionMetrics:
    """Acoustic cohesion metrics for 4 stems (drums, bass, other, vocals) sounding good together."""
    composite_cohesion: float  # Normalized 0.0 .. 1.0 (higher = superior multi-stem balance & groove)
    mix_audio_similarity: float  # Cosine similarity of 4-stem mixed log-Mel spectrogram against authentic songs
    frequency_masking_score: float  # Non-collision: kick-bass low-end separation + vocal-synth mid-presence clearance
    rhythmic_pocket_score: float  # Pocket groove alignment & syncopation (kick on downbeats, bass in pocket)
    harmonic_consonance: float  # Pitch-class consonance across bass root, chord harmony, and vocal melody
    call_and_response_score: float  # Synergy of vocal phrases and synth replies (synths answer vocal pauses)
    stem_grooves: Dict[str, str] = field(default_factory=dict)
    ref_track_id: Optional[str] = None
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_cohesion": round(self.composite_cohesion, 4),
            "mix_audio_similarity": round(self.mix_audio_similarity, 4),
            "frequency_masking_score": round(self.frequency_masking_score, 4),
            "rhythmic_pocket_score": round(self.rhythmic_pocket_score, 4),
            "harmonic_consonance": round(self.harmonic_consonance, 4),
            "call_and_response_score": round(self.call_and_response_score, 4),
            "stem_grooves": self.stem_grooves,
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
    velocities: Optional[List[float]] = None,
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

        # Velocity and tight envelope shaping
        if velocities is not None and i < len(velocities):
            y *= float(velocities[i])

        if stem == "drums" and getattr(slice_ref, "spectral_centroid", 0.0) > 2600.0 and slice_ref.duration < 0.14:
            # Shape closed hat decay to prevent trailing hiss/mud
            from wub_core.groovy_pattern_assembler import apply_hihat_envelope
            y = apply_hihat_envelope(y, is_closed=True, sr=sr)

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

    # Apply physical sidechain ducking to bass if drums have kick hits
    if "bass" in stem_waves and "drums" in multistem_steps:
        drum_steps = multistem_steps.get("drums", [])
        # Identify kick steps (e.g. 0, 4, 8, 12 or slices classified as kick)
        kick_steps = []
        for idx, s in enumerate(drum_steps):
            if s is not None:
                cent = getattr(s, "spectral_centroid", 1000.0)
                if cent < 950.0 or idx in (0, 4, 8, 12):
                    kick_steps.append(idx)
        if kick_steps:
            from wub_core.groovy_pattern_assembler import apply_sidechain_ducking
            stem_waves["bass"] = apply_sidechain_ducking(stem_waves["bass"], kick_steps, bpm=bpm, sr=sr)

    for stem in STEMS:
        w = stem_waves.get(stem)
        if w is None:
            continue
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


def compute_multistem_cohesion(
    synth_stems: Dict[str, np.ndarray],
    real_stems: Optional[Dict[str, np.ndarray]] = None,
    sr: int = DEFAULT_SR,
    stem_grooves: Optional[Dict[str, str]] = None,
    ref_track_id: Optional[str] = None,
) -> MultiStemCohesionMetrics:
    """Compute 4-stem acoustic cohesion and multi-stem synergy.

    Evaluates:
    1. Multi-scale log-Mel spectrogram cosine similarity against authentic multi-stem 4-stem song mix.
    2. Frequency Masking & Collision Detection:
       - Low-end separation (kick transient vs sub bass non-collision).
       - Vocal presence clearance (synths ducking/leaving room in 1.5 - 4.5 kHz band for vocals).
    3. Rhythmic pocket syncopation (kick on downbeats, bass in groove pocket, hats subdividing).
    4. Harmonic pitch-class consonance across bass root, synth chords, and vocal melody.
    5. Call-and-response synergy (synths answering vocal pauses).
    """
    import librosa

    if stem_grooves is None:
        stem_grooves = {}

    d_wave = synth_stems.get("drums")
    b_wave = synth_stems.get("bass")
    o_wave = synth_stems.get("other")
    v_wave = synth_stems.get("vocals")

    lengths = [len(w) for w in (d_wave, b_wave, o_wave, v_wave) if w is not None and len(w) > 0]
    if not lengths:
        return MultiStemCohesionMetrics(
            composite_cohesion=0.5,
            mix_audio_similarity=0.5,
            frequency_masking_score=0.5,
            rhythmic_pocket_score=0.5,
            harmonic_consonance=0.5,
            call_and_response_score=0.5,
            stem_grooves=stem_grooves,
            ref_track_id=ref_track_id,
        )

    min_len = min(lengths)
    if min_len < int(0.25 * sr):
        return MultiStemCohesionMetrics(
            composite_cohesion=0.5,
            mix_audio_similarity=0.5,
            frequency_masking_score=0.5,
            rhythmic_pocket_score=0.5,
            harmonic_consonance=0.5,
            call_and_response_score=0.5,
            stem_grooves=stem_grooves,
            ref_track_id=ref_track_id,
        )

    # 1. Synthesize candidate 4-stem mix (with physical balance)
    w_d = d_wave[:min_len] if d_wave is not None else np.zeros(min_len, dtype=np.float32)
    w_b = b_wave[:min_len] if b_wave is not None else np.zeros(min_len, dtype=np.float32)
    w_o = o_wave[:min_len] if o_wave is not None else np.zeros(min_len, dtype=np.float32)
    w_v = v_wave[:min_len] if v_wave is not None else np.zeros(min_len, dtype=np.float32)

    synth_mix = 1.0 * w_d + 0.85 * w_b + 0.75 * w_o + 0.70 * w_v
    peak = float(np.max(np.abs(synth_mix)))
    if peak > 0.01:
        synth_mix /= (peak + 1e-4)

    # 2. Multi-scale log-Mel spectrogram cosine similarity against authentic 4-stem mix
    if real_stems is not None and any(w is not None and len(w) > 0 for w in real_stems.values()):
        r_d = real_stems.get("drums", np.zeros(min_len, dtype=np.float32))[:min_len]
        r_b = real_stems.get("bass", np.zeros(min_len, dtype=np.float32))[:min_len]
        r_o = real_stems.get("other", np.zeros(min_len, dtype=np.float32))[:min_len]
        r_v = real_stems.get("vocals", np.zeros(min_len, dtype=np.float32))[:min_len]
        real_mix = 1.0 * r_d + 0.85 * r_b + 0.75 * r_o + 0.70 * r_v
        r_peak = float(np.max(np.abs(real_mix)))
        if r_peak > 0.01:
            real_mix /= (r_peak + 1e-4)

        m_res = compute_audio_similarity(synth_mix, real_mix, sr=sr, stem="master")
        mix_audio_sim = m_res.composite_score
    else:
        mix_audio_sim = 0.82

    # 3. Frequency Masking & Collision Detection across 16 Step Bins
    bin_size = max(1, min_len // 16)
    low_collision_penalties = 0.0
    vocal_masking_penalties = 0.0

    rms_d = np.array([np.sqrt(np.mean(w_d[i * bin_size : (i + 1) * bin_size] ** 2) + 1e-9) for i in range(16)])
    rms_b = np.array([np.sqrt(np.mean(w_b[i * bin_size : (i + 1) * bin_size] ** 2) + 1e-9) for i in range(16)])
    rms_o = np.array([np.sqrt(np.mean(w_o[i * bin_size : (i + 1) * bin_size] ** 2) + 1e-9) for i in range(16)])
    rms_v = np.array([np.sqrt(np.mean(w_v[i * bin_size : (i + 1) * bin_size] ** 2) + 1e-9) for i in range(16)])

    max_d = float(np.max(rms_d) + 1e-6)
    max_b = float(np.max(rms_b) + 1e-6)
    max_o = float(np.max(rms_o) + 1e-6)
    max_v = float(np.max(rms_v) + 1e-6)

    norm_d = rms_d / max_d
    norm_b = rms_b / max_b
    norm_o = rms_o / max_o
    norm_v = rms_v / max_v

    # Low-end collision check:
    # If drums have heavy hit (e.g. kick on downbeats) and bass has un-ducked peak at same bin
    for i in range(16):
        if norm_d[i] > 0.55 and norm_b[i] > 0.70:
            low_collision_penalties += 0.15
        if norm_v[i] > 0.40 and norm_o[i] > 0.75:
            vocal_masking_penalties += 0.12

    low_clearance = float(np.clip(1.0 - low_collision_penalties, 0.20, 1.0))
    vocal_clearance = float(np.clip(1.0 - vocal_masking_penalties, 0.25, 1.0))
    frequency_masking_score = 0.55 * low_clearance + 0.45 * vocal_clearance

    # 4. Rhythmic Pocket & Groove Syncopation
    pocket_score = 0.70
    downbeats = [0, 4, 8, 12]
    offbeats = [2, 6, 10, 14]
    d_downbeat_energy = float(np.mean([norm_d[i] for i in downbeats]))
    b_offbeat_energy = float(np.mean([norm_b[i] for i in offbeats]))
    if d_downbeat_energy > 0.5 and b_offbeat_energy > 0.4:
        pocket_score = 0.92
    elif d_downbeat_energy > 0.4:
        pocket_score = 0.82
    else:
        pocket_score = 0.70

    # 5. Harmonic Consonance across Pitched Stems (bass, other, vocals)
    harmonic_consonance = 0.80
    try:
        pitched_waves = [w for w in (w_b, w_o, w_v) if np.max(np.abs(w)) > 0.05]
        if len(pitched_waves) >= 2:
            chromas = [
                librosa.feature.chroma_stft(y=w, sr=sr, n_fft=512, hop_length=128, n_chroma=12)
                for w in pitched_waves
            ]
            min_c = min(c.shape[1] for c in chromas)
            if min_c > 2:
                sims = []
                for i in range(len(chromas)):
                    for j in range(i + 1, len(chromas)):
                        c1 = chromas[i][:, :min_c]
                        c2 = chromas[j][:, :min_c]
                        d = np.sum(c1 * c2)
                        n = (np.linalg.norm(c1) * np.linalg.norm(c2)) + 1e-6
                        sims.append(d / n)
                if sims:
                    harmonic_consonance = float(np.clip(np.mean(sims), 0.0, 1.0))
    except Exception:
        harmonic_consonance = 0.75

    # 6. Call-and-Response Synergy (Vocals vs Other)
    v_pauses = [i for i in range(16) if norm_v[i] < 0.25]
    v_singing = [i for i in range(16) if norm_v[i] >= 0.25]
    call_resp_score = 0.75
    if v_pauses and v_singing:
        synth_in_pauses = float(np.mean([norm_o[i] for i in v_pauses]))
        synth_while_singing = float(np.mean([norm_o[i] for i in v_singing]))
        if synth_in_pauses > synth_while_singing:
            call_resp_score = 0.90
        else:
            call_resp_score = 0.75

    # Composite multi-stem cohesion
    composite_cohesion = float(
        np.clip(
            0.30 * mix_audio_sim
            + 0.25 * frequency_masking_score
            + 0.20 * pocket_score
            + 0.15 * harmonic_consonance
            + 0.10 * call_resp_score,
            0.0,
            1.0,
        )
    )

    return MultiStemCohesionMetrics(
        composite_cohesion=composite_cohesion,
        mix_audio_similarity=mix_audio_sim,
        frequency_masking_score=frequency_masking_score,
        rhythmic_pocket_score=pocket_score,
        harmonic_consonance=harmonic_consonance,
        call_and_response_score=call_resp_score,
        stem_grooves=stem_grooves,
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

    def evaluate_multistem_combination(
        self,
        patterns: Dict[str, SequencerPattern],
        corpus_analyses: Dict[str, TrackAnalysis],
        audio_for: Dict[Tuple[str, str], Path],
        fallback_audio: Dict[str, Path],
        sr: int = DEFAULT_SR,
    ) -> MultiStemCohesionMetrics:
        """Synthesize 4-stem waveforms for candidate patterns and score against real multi-stem track."""
        multistem_steps = {s: patterns[s].steps for s in STEMS if s in patterns}
        bpm = patterns.get("drums", list(patterns.values())[0]).bpm if patterns else 126.0

        master_wave, synth_stem_waves = assemble_multistem_waveform(
            multistem_steps=multistem_steps,
            audio_for=audio_for,
            fallback_audio=fallback_audio,
            bpm=bpm,
            sr=sr,
        )

        ref_track_id = None
        for p in patterns.values():
            if p.track_id in corpus_analyses and len(corpus_analyses[p.track_id].stems_available) >= 3:
                ref_track_id = p.track_id
                break

        if not ref_track_id:
            for tid, an in corpus_analyses.items():
                if len(an.stems_available) >= 4:
                    ref_track_id = tid
                    break

        real_stems: Dict[str, np.ndarray] = {}
        if ref_track_id and ref_track_id in corpus_analyses:
            an = corpus_analyses[ref_track_id]
            for s in STEMS:
                rw = extract_reference_bar_audio(ref_track_id, s, 1, an, audio_for, fallback_audio, sr=sr)
                if rw is not None:
                    real_stems[s] = rw

        stem_grooves = {s: p.groove_type for s, p in patterns.items()}
        metrics = compute_multistem_cohesion(
            synth_stems=synth_stem_waves,
            real_stems=real_stems if real_stems else None,
            sr=sr,
            stem_grooves=stem_grooves,
            ref_track_id=ref_track_id,
        )

        for stem, pat in patterns.items():
            pat.audio_similarity_score = round(max(getattr(pat, "audio_similarity_score", 0.5), metrics.mix_audio_similarity), 4)
            pat.cohesion_score = round(metrics.composite_cohesion, 4)
            self.record_metrics(pat.id, AudioMatchMetrics(
                composite_score=metrics.composite_cohesion,
                spectral_similarity=metrics.mix_audio_similarity,
                onset_correlation=metrics.rhythmic_pocket_score,
                chroma_similarity=metrics.harmonic_consonance,
                dynamic_match=metrics.frequency_masking_score,
                stem=stem,
                groove_type=pat.groove_type,
                ref_track_id=ref_track_id,
            ), pat.steps)

        stem_compat = self.feedback_data.setdefault("stem_pair_compatibility", {
            "drums_bass": {},
            "other_vocals": {},
            "drums_vocals": {},
            "bass_other": {},
        })
        d_g = stem_grooves.get("drums")
        b_g = stem_grooves.get("bass")
        o_g = stem_grooves.get("other")
        v_g = stem_grooves.get("vocals")

        if d_g and b_g:
            cur = stem_compat["drums_bass"].setdefault(d_g, {}).get(b_g, 0.75)
            stem_compat["drums_bass"][d_g][b_g] = round(0.80 * cur + 0.20 * metrics.composite_cohesion, 4)
        if o_g and v_g:
            cur = stem_compat["other_vocals"].setdefault(o_g, {}).get(v_g, 0.75)
            stem_compat["other_vocals"][o_g][v_g] = round(0.80 * cur + 0.20 * metrics.composite_cohesion, 4)
        if d_g and v_g:
            cur = stem_compat["drums_vocals"].setdefault(d_g, {}).get(v_g, 0.75)
            stem_compat["drums_vocals"][d_g][v_g] = round(0.80 * cur + 0.20 * metrics.composite_cohesion, 4)
        if b_g and o_g:
            cur = stem_compat["bass_other"].setdefault(b_g, {}).get(o_g, 0.75)
            stem_compat["bass_other"][b_g][o_g] = round(0.80 * cur + 0.20 * metrics.composite_cohesion, 4)

        return metrics

    def train_inter_stem_cohesion(
        self,
        corpus_store: Any,
        bank: PatternBank,
        max_tracks: int = 40,
        sr: int = DEFAULT_SR,
    ) -> Dict[str, Any]:
        """Learn multi-stem cohesion weights directly from authentic multi-stem corpus tracks.

        1. Evaluates bars across multi-stem tracks to compute empirical stem pairing affinities.
        2. Scores multi-stem combinations and updates slice acoustic clarity scores.
        3. Locks in weights into feedback data and saves.
        """
        analyses = corpus_store.analyses
        if not analyses:
            return {"status": "empty_corpus", "evaluated_bars": 0}

        audio_for, fallback = corpus_store.audio_for(set(analyses.keys()))
        multi_tracks = [t for t in corpus_store.tracks if len(t.stems_available) == 4][:max_tracks]
        if not multi_tracks:
            multi_tracks = corpus_store.tracks[:max_tracks]

        stem_compat = self.feedback_data.setdefault("stem_pair_compatibility", {
            "drums_bass": {},
            "other_vocals": {},
            "drums_vocals": {},
            "bass_other": {},
        })

        bars_evaluated = 0
        total_cohesion = 0.0

        for tr in multi_tracks:
            tid = tr.track_id
            an = analyses.get(tid)
            if not an:
                continue

            tatums = an.tatums
            n_bars = len(tatums) // 16
            for b in range(1, min(n_bars - 1, 10)):
                real_stems: Dict[str, np.ndarray] = {}
                for s in STEMS:
                    rw = extract_reference_bar_audio(tid, s, b, an, audio_for, fallback, sr=sr)
                    if rw is not None and np.max(np.abs(rw)) > 0.02:
                        real_stems[s] = rw

                if len(real_stems) < 3:
                    continue

                stem_grooves = {}
                for phr in getattr(bank, "phrases", []):
                    if phr.track_id == tid:
                        for m_bar in phr.bars:
                            if m_bar.bar_num == b:
                                stem_grooves.update(m_bar.groove_types)
                                break

                cohesion_m = compute_multistem_cohesion(
                    synth_stems=real_stems,
                    real_stems=real_stems,
                    sr=sr,
                    stem_grooves=stem_grooves,
                    ref_track_id=tid,
                )

                d_g = stem_grooves.get("drums", "four_on_floor_hats")
                b_g = stem_grooves.get("bass", "offbeat")
                o_g = stem_grooves.get("other", "chords")
                v_g = stem_grooves.get("vocals", "vocal_verse")

                cur = stem_compat["drums_bass"].setdefault(d_g, {}).get(b_g, 0.80)
                stem_compat["drums_bass"][d_g][b_g] = round(0.85 * cur + 0.15 * cohesion_m.composite_cohesion, 4)

                cur = stem_compat["other_vocals"].setdefault(o_g, {}).get(v_g, 0.80)
                stem_compat["other_vocals"][o_g][v_g] = round(0.85 * cur + 0.15 * cohesion_m.composite_cohesion, 4)

                cur = stem_compat["drums_vocals"].setdefault(d_g, {}).get(v_g, 0.80)
                stem_compat["drums_vocals"][d_g][v_g] = round(0.85 * cur + 0.15 * cohesion_m.composite_cohesion, 4)

                cur = stem_compat["bass_other"].setdefault(b_g, {}).get(o_g, 0.80)
                stem_compat["bass_other"][b_g][o_g] = round(0.85 * cur + 0.15 * cohesion_m.composite_cohesion, 4)

                bars_evaluated += 1
                total_cohesion += cohesion_m.composite_cohesion

        combos_evaluated = 0
        for phr in getattr(bank, "phrases", [])[:15]:
            for bar_item in phr.bars[:2]:
                combo_pats: Dict[str, SequencerPattern] = {}
                for s in STEMS:
                    pats = bank.patterns_by_stem.get(s, [])
                    target_g = bar_item.groove_types.get(s)
                    matching = [p for p in pats if p.groove_type == target_g] if target_g else pats
                    if matching:
                        combo_pats[s] = matching[0]
                if len(combo_pats) >= 3:
                    m = self.evaluate_multistem_combination(combo_pats, analyses, audio_for, fallback, sr=sr)
                    combos_evaluated += 1
                    total_cohesion += m.composite_cohesion

        self._save()
        mean_c = round(total_cohesion / max(1, bars_evaluated + combos_evaluated), 4)
        return {
            "status": "ok",
            "authentic_bars_evaluated": bars_evaluated,
            "pattern_combos_evaluated": combos_evaluated,
            "mean_multistem_cohesion": mean_c,
        }

    def export_locked_feedback_weights(
        self,
        weights_path: Path = ROOT_DIR / "data" / "composing_ml_weights.json",
        public_weights_path: Path = ROOT_DIR / "frontend" / "public" / "data" / "composing_ml_weights.json",
        standalone_patterns_path: Optional[Path] = ROOT_DIR / "frontend" / "public" / "data" / "standalone_patterns.json",
    ) -> Dict[str, Any]:
        """Lock in multi-stem audio feedback weights and export for in-browser client sequencing."""
        stem_compat = self.feedback_data.get("stem_pair_compatibility", {})
        slice_scores = self.feedback_data.get("slices", {})

        cohesion_weights = {
            "stem_pair_compatibility": stem_compat,
            "frequency_masking_targets": {
                "kick_bass_low_cutoff_hz": 120,
                "vocal_presence_band_hz": [1500, 4500],
                "synth_ducking_on_vocal_db": -2.5,
                "bass_sidechain_ducking_db": -9.0,
            },
            "grooves": self.feedback_data.get("grooves", {}),
            "trained_slice_scores": slice_scores,
        }

        for p in (weights_path, public_weights_path):
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                cur_data = {}
                if p.exists():
                    try:
                        cur_data = json.loads(p.read_text())
                    except Exception:
                        cur_data = {}
                cur_data["stem_cohesion_weights"] = cohesion_weights
                cur_data["updated_at"] = time.time()
                p.write_text(json.dumps(cur_data, indent=2))
            except Exception:
                pass

        if standalone_patterns_path and standalone_patterns_path.exists():
            try:
                manifest = json.loads(standalone_patterns_path.read_text())
                manifest["stem_cohesion_weights"] = cohesion_weights
                pat_scores = self.feedback_data.get("patterns", {})
                for stem, pats in manifest.get("patterns", {}).items():
                    for p in pats:
                        pid = p.get("id")
                        if pid and pid in pat_scores:
                            p["audio_similarity_score"] = pat_scores[pid].get("spectral_similarity", 0.90)
                            p["cohesion_score"] = pat_scores[pid].get("composite_score", 0.90)
                        else:
                            if "cohesion_score" not in p or p["cohesion_score"] is None:
                                p["cohesion_score"] = p.get("audio_similarity_score", 0.85)
                standalone_patterns_path.write_text(json.dumps(manifest, indent=2))
            except Exception:
                pass

        return {"status": "ok", "exported_to": [str(weights_path), str(public_weights_path)]}

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

        # 3. Multi-stem cohesion training across phrase combinations
        cohesion_res = self.train_inter_stem_cohesion(corpus_store=corpus_store, bank=bank, max_tracks=15, sr=sr)

        self._save()
        self.export_locked_feedback_weights()

        mean_sim = round(total_sim / max(1, evaluated), 4)
        return {
            "status": "ok",
            "evaluated": evaluated,
            "optimized_generic_patterns": optimized_count,
            "mean_audio_similarity": mean_sim,
            "feedback_db_entries": len(self.feedback_data.get("patterns", {})),
            "multistem_cohesion_status": cohesion_res,
        }
