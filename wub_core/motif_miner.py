"""Self-similarity motif schema miner, chorus vocal hook & drum fill detector.

Analyzes 4-bar phrases into:
1. Canonical bar-to-bar motif schemas: AAAB, ABAB, ABAC, AABB, ABCD.
2. Vocal roles: 1-bar/2-bar chorus hooks, buildup loop anchors, sub-bar chops,
   airy sustained phrases, verse phraselets, and sparse intro teasers.
3. Drum turnaround fill detection on bar 3 (the 4th bar of a 4-bar phrase).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wub_core.theory_engine import (
    MIN_RMS_DB,
    clamp01,
    classify_triad_quality,
    score_cell_purity,
    set_to_onset_bits,
)

VOWELS = ("AH", "EE", "OH", "OO", "AY")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / math.sqrt(na * nb)


def bar_similarity_matrix(bar_vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """Compute NxN cosine similarity matrix over bar feature vectors."""
    n = len(bar_vectors)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
        for j in range(i + 1, n):
            sim = _cosine(bar_vectors[i], bar_vectors[j])
            mat[i][j] = sim
            mat[j][i] = sim
    return mat


def classify_motif_schema(
    bar_vectors: Sequence[Sequence[float]],
    sim_threshold: float = 0.80,
) -> str:
    """Classify a 4-bar phrase into AAAB, ABAB, ABAC, AABB, or ABCD.

    Uses the 4x4 bar self-similarity matrix with deterministic tie-breaking so
    repeating bars (e.g. bars 0,1,2 matching while bar 3 fills/turns around)
    yield clean compositional templates.
    """
    if len(bar_vectors) < 4:
        return "ABCD"
    sim = bar_similarity_matrix(bar_vectors[:4])
    s01 = sim[0][1]
    s02 = sim[0][2]
    s03 = sim[0][3]
    s12 = sim[1][2]
    s13 = sim[1][3]
    s23 = sim[2][3]

    # AAAB: first 3 bars strongly self-similar; bar 4 diverges (or all 4 similar)
    if s01 >= sim_threshold and s02 >= sim_threshold and s12 >= sim_threshold:
        return "AAAB"

    # ABAB: 2-bar period (bar 0 ~ bar 2 and bar 1 ~ bar 3, while 0 != 1)
    if s02 >= sim_threshold and s13 >= sim_threshold and s01 < sim_threshold:
        return "ABAB"

    # AABB: two pairs (bar 0 ~ bar 1 and bar 2 ~ bar 3)
    if s01 >= sim_threshold and s23 >= sim_threshold and s02 < sim_threshold:
        return "AABB"

    # ABAC: bar 0 and bar 2 anchor, bar 1 and bar 3 vary
    if s02 >= sim_threshold:
        return "ABAC"

    # Relaxed secondary check with slightly lower threshold for noisy stems
    soft = max(0.68, sim_threshold - 0.10)
    if s01 >= soft and s02 >= soft:
        return "AAAB"
    if s02 >= soft and s13 >= soft:
        return "ABAB"
    if s01 >= soft and s23 >= soft:
        return "AABB"
    if s02 >= soft:
        return "ABAC"
    return "ABCD"


def classify_vowel_from_centroid(centroid_hz: float) -> str:
    """Map spectral centroid (Hz) to one of the 5 expressive vowel classes."""
    c = float(centroid_hz or 0.0)
    if c < 1100.0:
        return "OO"
    if c < 1900.0:
        return "OH"
    if c < 2700.0:
        return "AH"
    if c < 3800.0:
        return "AY"
    return "EE"


def detect_drum_turnaround_fill(
    bar_onset_counts: Sequence[int],
    bar_energies: Optional[Sequence[float]] = None,
) -> Tuple[bool, bool]:
    """Detect whether bar 3 (index 3) acts as a drum turnaround / fill.

    Returns ``(is_fill_bar, is_high_density_fill)``.
    """
    if len(bar_onset_counts) < 4:
        return False, False
    head = [int(c) for c in bar_onset_counts[:3]]
    b4 = int(bar_onset_counts[3])
    mean_head = sum(head) / 3.0
    density_spike = b4 >= max(6, int(round(mean_head * 1.25 + 1.0)))
    drop_spike = b4 <= max(2, int(round(mean_head * 0.45))) and mean_head >= 4.0
    energy_spike = False
    if bar_energies and len(bar_energies) >= 4:
        mean_e = sum(float(e) for e in bar_energies[:3]) / 3.0
        energy_spike = float(bar_energies[3]) > mean_e + 0.08
    is_fill = density_spike or drop_spike or energy_spike
    return bool(is_fill), bool(b4 >= 8 and density_spike)


def classify_vocal_role(
    section: str,
    rms_db: float,
    purity: float,
    bar_onset_counts: Sequence[int],
    schema: str,
    mean_seg_dur_sec: float = 0.35,
) -> Dict[str, Any]:
    """Classify a 4-bar vocal phrase into a compositional role and chop anchor."""
    if rms_db < MIN_RMS_DB["vocals"] - 2.0 or sum(bar_onset_counts) == 0:
        return {
            "vocal_role": "none",
            "is_chorus_hook": False,
            "hook_bars": 0,
            "best_chop_bar": 0,
            "best_chop_beat": 0,
        }

    sec = (section or "GROOVE_A").upper()
    counts = [int(c) for c in (list(bar_onset_counts) + [0, 0, 0, 0])[:4]]
    active_bars = [i for i, c in enumerate(counts) if c > 0]
    best_bar = max(range(4), key=lambda i: counts[i]) if active_bars else 0

    if len(active_bars) <= 1 or sum(counts) <= 3:
        return {
            "vocal_role": "teaser",
            "is_chorus_hook": False,
            "hook_bars": 1,
            "best_chop_bar": best_bar,
            "best_chop_beat": 0,
        }

    if mean_seg_dur_sec >= 0.65 or sec == "BREAKDOWN":
        role = "airy"
        is_hook = sec in ("DROP", "PEAK") and purity >= 0.45
        hook_bars = 2 if schema in ("ABAB", "AABB") else 1
    elif sec in ("DROP", "PEAK") or (schema in ("AAAB", "ABAB", "AABB") and purity >= 0.42):
        hook_bars = 1 if schema == "AAAB" else 2
        role = "hook_1bar" if hook_bars == 1 else "hook_2bar"
        is_hook = True
    elif mean_seg_dur_sec <= 0.24:
        role = "chop"
        is_hook = purity >= 0.50
        hook_bars = 1
    else:
        role = "verse"
        is_hook = False
        hook_bars = 2 if schema in ("ABAB", "AABB") else 1

    return {
        "vocal_role": role,
        "is_chorus_hook": is_hook,
        "hook_bars": hook_bars,
        "best_chop_bar": int(best_bar),
        "best_chop_beat": 0,
    }


def extract_phrase_bar_features(
    track_analysis: Dict[str, Any],
    start_bar: int,
    stem: str,
    stem_rms_db: float,
    stem_peak: float,
) -> Optional[Dict[str, Any]]:
    """Extract 4-bar per-bar roots, triad qualities, onset bitmasks, purity, and schema
    for one phrase of one stem directly from a processed track JSON."""
    tatums = track_analysis.get("tatums") or []
    segments = track_analysis.get("segments") or []
    start_tat = int(start_bar) * 16
    if start_tat + 64 > len(tatums):
        return None

    seg_by_id = {str(s.get("id")): s for s in segments if s.get("id") is not None}
    bar_vectors: List[List[float]] = []
    bar_roots: List[int] = []
    bar_qualities: List[str] = []
    bar_onsets: List[int] = []
    bar_purities: List[float] = []
    bar_counts: List[int] = []
    seg_durs: List[float] = []
    centroids: List[float] = []

    prev_sid: Optional[str] = None
    for b in range(4):
        chroma_sum = [0.0] * 12
        onset_steps: List[int] = []
        step_vec = [0.0] * 16
        loud_vals: List[float] = []

        for step in range(16):
            tat = tatums[start_tat + b * 16 + step]
            sids = (tat.get("stem_segments") or {}).get(stem) or []
            curr_sid = str(sids[-1]) if sids else None
            is_onset = curr_sid is not None and (curr_sid != prev_sid or step == 0)
            prev_sid = curr_sid

            if curr_sid and curr_sid in seg_by_id:
                sg = seg_by_id[curr_sid]
                if not sg.get("is_silent", False):
                    pitches = sg.get("pitches") or [0.0] * 12
                    if len(pitches) == 12:
                        for k in range(12):
                            chroma_sum[k] += float(pitches[k])
                    if is_onset:
                        onset_steps.append(step)
                        step_vec[step] = 1.0
                        loud_vals.append(float(sg.get("loudness_max", -20.0)))
                        seg_durs.append(float(sg.get("duration", 0.25)))
                        if sg.get("spectral_centroid") is not None:
                            centroids.append(float(sg.get("spectral_centroid")))

        bar_onsets.append(set_to_onset_bits(onset_steps))
        bar_counts.append(len(onset_steps))

        if stem == "drums":
            bar_roots.append(-1)
            bar_qualities.append("perc")
            purity = score_cell_purity(stem, stem_rms_db, stem_peak, chroma_sum, len(onset_steps))
            bar_vectors.append(step_vec + [len(onset_steps) / 16.0])
        else:
            root_pc, qual, _ = classify_triad_quality(chroma_sum)
            if sum(chroma_sum) <= 1e-6:
                root_pc = -1
                qual = "mono"
            bar_roots.append(root_pc)
            bar_qualities.append(qual)
            purity = score_cell_purity(stem, stem_rms_db, stem_peak, chroma_sum, len(onset_steps))
            c_tot = sum(chroma_sum) or 1.0
            norm_c = [v / c_tot for v in chroma_sum]
            bar_vectors.append(step_vec + norm_c)

        bar_purities.append(round(purity, 3))

    schema = classify_motif_schema(bar_vectors)
    is_fill, is_dense_fill = detect_drum_turnaround_fill(bar_counts) if stem == "drums" else (False, False)
    mean_dur = sum(seg_durs) / len(seg_durs) if seg_durs else 0.35
    vowel = classify_vowel_from_centroid(sum(centroids) / len(centroids)) if centroids else "AH"

    return {
        "schema": schema,
        "purity": bar_purities,
        "bar_roots": bar_roots,
        "bar_qualities": bar_qualities,
        "bar_onsets": bar_onsets,
        "bar_counts": bar_counts,
        "is_fill_bar": is_fill,
        "is_dense_fill": is_dense_fill,
        "mean_seg_dur": round(mean_dur, 3),
        "vowel": vowel,
    }
