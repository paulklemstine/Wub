"""Build the "taste" dataset: one 160-d feature vector per real 4-bar window.

A *taste* model scores synthetic arrangements by how much they resemble real
songs, so the training set is every real 4-bar window of the analysed corpus
described by a fixed-length, audio-free feature vector.  Only analysis JSON is
read (``tatums`` + ``segments`` + header fields); no stem WAV is opened, so a
full-corpus pass takes minutes instead of hours.

Per track
---------
1. ``wub_core.section_arcs.bar_features`` -> per-bar energy (mean onset
   ``loudness_max``, dB) and per-stem onset density.
2. ``wub_core.section_arcs.segment_sections`` -> labelled spans
   (INTRO / GROOVE_A / BUILDUP / DROP / BREAKDOWN).
3. Every non-overlapping 4-bar window on the bar grid, ``start_bar`` in
   ``0, 4, 8, ...`` (the *aligned* windows), plus one extra window per kit
   phrase whose ``start_bar`` is not a multiple of 4 (``aligned: false``; see
   "Kit join").  Windows that run past the last bar are skipped.

Feature layout
--------------
Dimensions ``0..47`` are the original 48-d block and stay byte-identical to
the v1 dataset.  Dimensions ``48..159`` add the causal evidence the ranker
needs to choose the *next* phrase rather than merely recognise a real
arrangement: which stem actually fired on which tatum, and what the bass and
harmony are doing.  The full layout is::

     0..11   12  chroma       mean ``pitches`` of the unique non-silent bass /
                              other / vocals segments referenced by the
                              window's 64 tatums (drums excluded: percussive
                              chroma is noise), L1-normalised; all-zero when
                              the window has no pitched material (an all-zero
                              chroma vector is itself evidence).
    12..23   12  timbre       mean ``timbre`` of the unique non-silent segments
                              of all four stems.
    24..27    4  onset dens.  drums / bass / other / vocals: fraction of the 64
                              tatums whose segment id differs from the previous
                              tatum's for that stem (the track's first tatum
                              counts when a segment is present).  This is the
                              literal "new onset" reading, computed on the whole
                              track, so a window's first tatum is judged
                              against the tatum before it.
    28..31    4  loudness     per-stem mean ``loudness_max`` over the unique
                              non-silent segments of that stem, clipped
                              -60..0 dB, scaled ``(x + 60) / 60`` -> 0..1
                              (0 = no material in that stem).
    32        1  energy       mean of the 4 ``bar_features`` bar energies (dB),
                              mapped ``(e + 60) / 60`` clipped 0..1.
    33        1  energy slope ``norm(last bar) - norm(first bar)``, -1..1.
    34..38    5  section      one-hot of the section label of the window's
                              *first* bar (INTRO / GROOVE_A / BUILDUP / DROP /
                              BREAKDOWN).
    39        1  since start  ``(start_bar - span.start_bar) / 16``, clipped
                              0..1.
    40        1  vocal pres.  fraction of the 64 tatums whose vocals segment is
                              *audible*: not silent and ``loudness_max``
                              >= -50 dB (the audibility floor
                              ``build_kit_corpus.MIN_STEM_RMS_DB`` already uses).
                              Two literal readings were rejected on evidence:
                              "fraction of tatums with a vocals onset" is
                              identical to feature 27 (vocals onset density),
                              and "fraction of tatums with a non-silent vocals
                              segment" is 1.0 in 97.6% of windows (mean 0.99)
                              because analyses almost never flag vocals
                              segments as silent.  The -50 dB reading keeps the
                              onset column and the presence column independent
                              and measures the thing that matters to a taste
                              model: vocals audible versus buried.  Flip
                              ``VOCAL_AUDIBLE_DB`` to ``-inf`` for the plain
                              non-silent reading.
    41..47    7  reserved     zeros, kept free for future features.
    48..111  64  onset bits   stems in order drums, bass, other, vocals; each
                              stem owns 16 bits, one per tatum of the bar grid.
                              The four bars of the window are folded onto that
                              16-step grid: bit ``k`` is 1 when a segment of
                              that stem starts on tatum ``k`` of *any* of the
                              four bars.  A segment is first assigned to the
                              tatum containing its ``start`` (its first
                              overlap), then that tatum contributes its modulo
                              16 position; a segment referenced by several
                              consecutive tatums therefore sets exactly one
                              position.  This is the standard 16-step rhythmic
                              signature (kick / bass / chord / vocal pattern)
                              and the only 64-bit reading that keeps
                              ``FEATURE_DIM = 160``; it is positional in the
                              bar, not in the 4-bar span.
   112..123  12  bass chroma  mean ``pitches`` of the unique non-silent bass
                              segments of the window, L1-normalised; zeros
                              when the window has no bass material.
   124..135  12  other chroma same for the ``other`` stem.
   136..147  12  bass root    one-hot of ``argmax(bass chroma)``; all-zero bass
                              chroma yields root 0 (C) by convention.
   148..159  12  chord root   one-hot of ``argmax(bass chroma + other chroma)``.
                              Bass and ``other`` are each L1-normalised, so
                              they contribute equally regardless of how much
                              material each stem carries.  All-zero input
                              yields root 0 (C) by convention.

That is 12+12+4+4+1+1+5+1+1 = 41 real features plus the reserved tail in
``0..47``, then 64 onset bits and 4 x 12 harmonic features:
48 + 64 + 48 = 160.  ``feature_dim: 160`` is a hard contract checked by the
validator, including a check that every value is finite and that every
one-hot block sums to 1.

Window metadata
---------------
Every window (and every joined kit phrase) also carries the evidence the
browser player uses to gate phrase choices::

    chroma      the existing 12-d harmonic chroma (equal to features 0..11)
    bass_root   int argmax of the bass chroma (0..11)
    is_intro    start_bar <= 4 or the window's section is INTRO
    is_outro    start_bar + 4 >= n_bars - 4
    once        is_intro or is_outro (a phrase that should play at most once)

``once`` is deliberately a derived copy rather than a third independent flag:
the player can trust either reading.

Kit join
--------
``scripts/build_kit_corpus.py`` seeds its 4-bar candidates with every section
start, so most kit phrases begin on a bar that is *not* a multiple of 4.
Joining them to the aligned grid alone would drop ~60% of the phrases, and all
kit phrases must be present.  Each off-grid kit start bar therefore gets its own
window (``aligned: false``) so ``kit_phrases`` joins exactly on
``(track_id, start_bar)`` and copies that window's vector verbatim.  Consumers
that want the strictly disjoint grid filter on ``aligned``.

Usage::

    python scripts/build_taste_dataset.py \\
        --processed processed --kits kits_full/kit_manifest.json \\
        --out /tmp/wubtrain/taste_dataset.json

Output: ``{"version": 2, "feature_dim": 160, "tracks": [...],
"kit_phrases": [...], "stats": {...}}``, every float rounded to 5 decimals.
The pass is deterministic: files, tracks, windows and phrases are all sorted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# One BLAS thread per worker: the pool already provides the parallelism.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wub_core.section_arcs import bar_features, segment_sections

STEMS: Tuple[str, ...] = ("drums", "bass", "other", "vocals")
PITCHED_STEMS: Tuple[str, ...] = ("bass", "other", "vocals")
SECTIONS: Tuple[str, ...] = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")

WINDOW_BARS = 4
TATUMS_PER_BAR = 16
TATUMS_PER_WINDOW = WINDOW_BARS * TATUMS_PER_BAR  # 64
BASE_FEATURE_DIM = 48          # v1 layout, byte-identical at indices 0..47
# 16 bits per stem = the standard 16-step (one-bar) onset signature; the four
# bars of the window are folded (OR) onto that grid.  4 stems x 16 = 64 bits,
# which is the only reading consistent with the FEATURE_DIM=160 contract.
ONSET_BITS_PER_STEM = TATUMS_PER_BAR
ONSET_DIM = ONSET_BITS_PER_STEM * len(STEMS)
CHROMA_DIM = 12
HARMONIC_DIM = 4 * CHROMA_DIM  # bass / other chroma + bass-root / chord-root
FEATURE_DIM = BASE_FEATURE_DIM + ONSET_DIM + HARMONIC_DIM  # 160
RESERVED_DIMS = 7  # 3 requested + 4 pad so the v1 block reaches 48
ONSET_EPS = 1e-6   # tatum/segment boundary tolerance (seconds)

LOUD_DB_MIN = -60.0
LOUD_DB_MAX = 0.0
# Vocals below this peak level are treated as not present (same -50 dB floor as
# build_kit_corpus.MIN_STEM_RMS_DB).  Set to -inf for "any non-silent segment".
VOCAL_AUDIBLE_DB = -50.0
SECTION_LEN_NORM = 16.0
DECIMALS = 5
DEFAULT_WORKERS = 8


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _r5(value: float) -> float:
    """Round to 5 decimals, folding -0.0 into 0.0 for a clean JSON payload."""
    out = round(float(value), DECIMALS)
    return 0.0 if out == 0.0 else out


def _round_vec(values: Sequence[float]) -> List[float]:
    return [_r5(v) for v in values]


def _norm_db(value: float) -> float:
    """Clip a dB value to -60..0 and scale it to 0..1."""
    v = float(value)
    if not math.isfinite(v):
        v = LOUD_DB_MIN
    return float(np.clip((v - LOUD_DB_MIN) / (LOUD_DB_MAX - LOUD_DB_MIN), 0.0, 1.0))


def _stats_vec(raw: Any, dim: int = 12) -> Optional[np.ndarray]:
    """A 12-d analysis vector, or None when missing/short/non-finite."""
    if not isinstance(raw, (list, tuple)) or len(raw) < dim:
        return None
    try:
        vec = np.asarray(raw[:dim], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vec.shape != (dim,) or not np.all(np.isfinite(vec)):
        return None
    return vec


def _classify_style(filename: str, bpm: float) -> str:
    """``scripts.harvest_style_kits.classify_track_style`` with a safe default."""
    try:
        try:
            from scripts.harvest_style_kits import classify_track_style
        except Exception:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from harvest_style_kits import classify_track_style  # type: ignore
        return str(classify_track_style(filename or "", float(bpm or 0.0)))
    except Exception:
        return "house"


def _prefix_sum(flags: np.ndarray) -> np.ndarray:
    """``prefix[i]`` = sum of ``flags[:i]`` so a range sum is O(1)."""
    if flags.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(flags, dtype=np.float64)))


def _root_onehot(chroma: Sequence[float]) -> Tuple[List[float], int]:
    """12-d one-hot of ``argmax(chroma)`` plus the integer root.

    An all-zero chroma vector has no musical root; ``argmax`` still returns 0,
    so the documented convention is C.  The caller keeps the raw chroma next
    to this flag, so a consumer can test ``sum(chroma) == 0`` if it needs to
    distinguish "C" from "unknown".
    """
    values = np.asarray(chroma, dtype=np.float64)
    root = int(np.argmax(values)) if values.size == CHROMA_DIM else 0
    root = max(0, min(CHROMA_DIM - 1, root))
    onehot = [0.0] * CHROMA_DIM
    onehot[root] = 1.0
    return onehot, root


# ---------------------------------------------------------------------------
# Per-track feature extraction
# ---------------------------------------------------------------------------

class _WindowFeaturizer:
    """All per-track precomputation shared by that track's 4-bar windows."""

    def __init__(self, analysis: Dict[str, Any], bar_feats: List[Dict[str, Any]],
                 spans: Sequence[Any], n_bars: int) -> None:
        self.tatums: List[dict] = list(analysis.get("tatums") or [])
        self.bar_feats = bar_feats
        self.n_bars = int(n_bars)

        # --- segment attribute lookup ---------------------------------------
        self.seg_pitch: Dict[str, Optional[np.ndarray]] = {}
        self.seg_timbre: Dict[str, Optional[np.ndarray]] = {}
        self.seg_loud: Dict[str, float] = {}
        self.seg_silent: Dict[str, bool] = {}
        self.seg_start: Dict[str, Optional[float]] = {}
        for seg in analysis.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            sid = str(seg.get("id") or "")
            if not sid:
                continue
            self.seg_silent[sid] = bool(seg.get("is_silent") or False)
            self.seg_pitch[sid] = _stats_vec(seg.get("pitches"))
            self.seg_timbre[sid] = _stats_vec(seg.get("timbre"))
            loud = seg.get("loudness_max")
            self.seg_loud[sid] = float(loud) if loud is not None else LOUD_DB_MIN
            start = seg.get("start")
            if start is None:
                self.seg_start[sid] = None
            else:
                try:
                    value = float(start)
                except (TypeError, ValueError):
                    value = float("nan")
                self.seg_start[sid] = value if math.isfinite(value) else None

        # --- per-bar segment references (union over the bar's 16 tatums) ----
        self.bar_refs: List[Dict[str, List[str]]] = []
        for b in range(self.n_bars):
            refs: Dict[str, List[str]] = {s: [] for s in STEMS}
            seen: Dict[str, set] = {s: set() for s in STEMS}
            for step in range(TATUMS_PER_BAR):
                tat = self.tatums[b * TATUMS_PER_BAR + step]
                stem_segments = tat.get("stem_segments") or {}
                for stem in STEMS:
                    for raw_id in stem_segments.get(stem) or []:
                        sid = str(raw_id)
                        if sid in seen[stem]:
                            continue
                        seen[stem].add(sid)
                        if sid not in self.seg_silent:
                            continue  # dangling reference: no features to use
                        if self.seg_silent[sid]:
                            continue  # silent segments carry no evidence
                        refs[stem].append(sid)
            self.bar_refs.append(refs)

        # --- onset flags + vocal presence per tatum --------------------------
        n_tatums = len(self.tatums)
        onsets = {s: np.zeros(n_tatums, dtype=np.float64) for s in STEMS}
        vocals_on = np.zeros(n_tatums, dtype=np.float64)
        prev: Dict[str, Optional[str]] = {s: None for s in STEMS}
        for i, tat in enumerate(self.tatums):
            stem_segments = tat.get("stem_segments") or {}
            for stem in STEMS:
                ids = stem_segments.get(stem) or []
                curr = str(ids[-1]) if ids else None
                if curr is not None and (i == 0 or curr != prev[stem]):
                    onsets[stem][i] = 1.0
                prev[stem] = curr
            for raw_id in stem_segments.get("vocals") or []:
                sid = str(raw_id)
                if sid not in self.seg_silent or self.seg_silent[sid]:
                    continue
                if self.seg_loud.get(sid, LOUD_DB_MIN) >= VOCAL_AUDIBLE_DB:
                    vocals_on[i] = 1.0
                    break
        self.onset_prefix = {s: _prefix_sum(onsets[s]) for s in STEMS}
        self.vocals_prefix = _prefix_sum(vocals_on)

        # --- positional onset bits (one bit per stem segment, first overlap) --
        # ``onsets`` above measures "the segment id changed"; the 64-d block
        # needs the tatum in which each segment actually begins.  Analyse every
        # tatum of the track in order so a segment referenced by several
        # consecutive tatums is assigned to the one containing its ``start``.
        self.onset_flags = {s: np.zeros(n_tatums, dtype=np.float64)
                            for s in STEMS}
        self.onset_bad = {"missing_segment": 0, "missing_start": 0,
                          "outside_tatums": 0, "not_referenced": 0}
        self.bad_tatums: set = set()
        tatum_starts = np.asarray(
            [float(tat.get("start") or 0.0) for tat in self.tatums],
            dtype=np.float64)
        tatum_ends = tatum_starts + np.asarray(
            [float(tat.get("duration") or 0.0) for tat in self.tatums],
            dtype=np.float64)
        first_ref: Dict[str, Dict[str, int]] = {s: {} for s in STEMS}
        tatum_ids: List[Dict[str, List[str]]] = []
        for i, tat in enumerate(self.tatums):
            stem_segments = tat.get("stem_segments") or {}
            ids_here: Dict[str, List[str]] = {}
            for stem in STEMS:
                ids = [str(raw_id) for raw_id in (stem_segments.get(stem) or [])]
                ids_here[stem] = ids
                for sid in ids:
                    first_ref[stem].setdefault(sid, i)
            tatum_ids.append(ids_here)
        for stem in STEMS:
            for sid, first_i in first_ref[stem].items():
                target = first_i
                if sid not in self.seg_start:
                    self.onset_bad["missing_segment"] += 1
                    self.bad_tatums.add(target)
                elif self.seg_start[sid] is None:
                    self.onset_bad["missing_start"] += 1
                    self.bad_tatums.add(target)
                else:
                    start = float(self.seg_start[sid])
                    idx = bisect_right(tatum_starts, start + ONSET_EPS) - 1
                    idx = max(0, min(n_tatums - 1, idx))
                    inside = (start >= tatum_starts[idx] - ONSET_EPS
                              and start <= tatum_ends[idx] + ONSET_EPS)
                    if not inside:
                        # Start outside the analysed tatum grid (usually a
                        # segment cut at the very end): keep the onset at its
                        # first reference instead of dropping it.
                        self.onset_bad["outside_tatums"] += 1
                        self.bad_tatums.add(target)
                    elif sid not in tatum_ids[idx].get(stem, []):
                        # The tatum covering the start does not reference the
                        # segment (analysis gap).  Fall back to the first
                        # referencing tatum so the onset is not lost.
                        self.onset_bad["not_referenced"] += 1
                        self.bad_tatums.add(target)
                    else:
                        target = idx
                self.onset_flags[stem][target] = 1.0
        bad_mask = np.zeros(n_tatums, dtype=bool)
        for tatum_idx in self.bad_tatums:
            bad_mask[tatum_idx] = True
        self.onset_bad_prefix = _prefix_sum(bad_mask)

        # --- section map ----------------------------------------------------
        self.sec_of_bar: List[str] = ["GROOVE_A"] * self.n_bars
        self.sec_start_of_bar: List[int] = [0] * self.n_bars
        for span in spans:
            label = str(getattr(span, "label", "GROOVE_A"))
            start = max(0, int(getattr(span, "start_bar", 0)))
            end = min(self.n_bars, int(getattr(span, "end_bar", self.n_bars)))
            for b in range(start, end):
                self.sec_of_bar[b] = label
                self.sec_start_of_bar[b] = start

    # -- feature blocks ------------------------------------------------------

    def _union(self, start_bar: int, stems: Sequence[str]) -> List[str]:
        """Unique non-silent segment ids referenced by the window, in order."""
        out: List[str] = []
        seen: set = set()
        for b in range(start_bar, start_bar + WINDOW_BARS):
            for stem in stems:
                for sid in self.bar_refs[b][stem]:
                    if sid in seen:
                        continue
                    seen.add(sid)
                    out.append(sid)
        return out

    def _chroma(self, start_bar: int) -> List[float]:
        ids = self._union(start_bar, PITCHED_STEMS)
        acc = [self.seg_pitch[sid] for sid in ids
               if self.seg_pitch.get(sid) is not None]
        if not acc:
            return [0.0] * 12
        mean = np.mean(np.stack(acc), axis=0)
        total = float(np.sum(np.abs(mean)))
        if total <= 1e-9:
            return [0.0] * 12
        return [float(v) for v in mean / total]

    def _stem_chroma(self, start_bar: int, stem: str) -> List[float]:
        """Mean ``pitches`` of one stem's unique non-silent window segments."""
        ids = self._union(start_bar, (stem,))
        acc = [self.seg_pitch[sid] for sid in ids
               if self.seg_pitch.get(sid) is not None]
        if not acc:
            return [0.0] * CHROMA_DIM
        mean = np.mean(np.stack(acc), axis=0)
        total = float(np.sum(np.abs(mean)))
        if total <= 1e-9:
            return [0.0] * CHROMA_DIM
        return [float(v) for v in mean / total]

    def _timbre(self, start_bar: int) -> List[float]:
        ids = self._union(start_bar, STEMS)
        acc = [self.seg_timbre[sid] for sid in ids
               if self.seg_timbre.get(sid) is not None]
        if not acc:
            return [0.0] * 12
        return [float(v) for v in np.mean(np.stack(acc), axis=0)]

    def _loudness(self, start_bar: int) -> List[float]:
        out: List[float] = []
        for stem in STEMS:
            ids = self._union(start_bar, (stem,))
            if not ids:
                out.append(0.0)  # no material -> -60 dB -> 0.0
                continue
            mean_db = float(np.mean([self.seg_loud.get(sid, LOUD_DB_MIN)
                                     for sid in ids]))
            out.append(_norm_db(mean_db))
        return out

    def energy(self, start_bar: int) -> Tuple[float, float]:
        """(mean window energy 0..1, last-minus-first bar energy -1..1)."""
        values = [_norm_db(self.bar_feats[b]["energy"])
                  for b in range(start_bar, start_bar + WINDOW_BARS)]
        mean = float(np.mean(values))
        slope = float(np.clip(values[-1] - values[0], -1.0, 1.0))
        return mean, slope

    def _density(self, start_bar: int) -> List[float]:
        lo = start_bar * TATUMS_PER_BAR
        hi = lo + TATUMS_PER_WINDOW
        return [float((self.onset_prefix[s][hi] - self.onset_prefix[s][lo])
                      / TATUMS_PER_WINDOW) for s in STEMS]

    def _vocal_presence(self, start_bar: int) -> float:
        lo = start_bar * TATUMS_PER_BAR
        hi = lo + TATUMS_PER_WINDOW
        return float((self.vocals_prefix[hi] - self.vocals_prefix[lo])
                     / TATUMS_PER_WINDOW)

    def _onset_bits(self, start_bar: int) -> List[float]:
        """64 onset bits: 4 stem 16-step signatures, 4 bars folded onto 16."""
        lo = start_bar * TATUMS_PER_BAR
        hi = lo + TATUMS_PER_WINDOW
        out: List[float] = []
        for stem in STEMS:
            folded = np.zeros(TATUMS_PER_BAR, dtype=np.float64)
            for step, value in enumerate(self.onset_flags[stem][lo:hi]):
                if value:
                    folded[step % TATUMS_PER_BAR] = 1.0
            out.extend(float(v) for v in folded)
        return out

    def window_has_bad_onset(self, start_bar: int) -> bool:
        """True when any tatum in the window needed an onset fallback."""
        lo = start_bar * TATUMS_PER_BAR
        hi = lo + TATUMS_PER_WINDOW
        return bool(self.onset_bad_prefix[hi] - self.onset_bad_prefix[lo] > 0.0)

    def describe(self, start_bar: int) -> Tuple[List[float], Dict[str, Any]]:
        """The 160-d vector and the player metadata of the window."""
        energy, slope = self.energy(start_bar)
        label = self.sec_of_bar[start_bar]
        onehot = [1.0 if label == sec else 0.0 for sec in SECTIONS]
        since = float(np.clip((start_bar - self.sec_start_of_bar[start_bar])
                              / SECTION_LEN_NORM, 0.0, 1.0))
        chroma = self._chroma(start_bar)
        bass_chroma = _round_vec(self._stem_chroma(start_bar, "bass"))
        other_chroma = _round_vec(self._stem_chroma(start_bar, "other"))
        # Derive the roots from the *serialised* (rounded) chroma so the
        # one-hot always agrees with the values a consumer reads back.
        bass_root_vec, bass_root = _root_onehot(bass_chroma)
        chord_root_vec, _chord_root = _root_onehot(
            [a + b for a, b in zip(bass_chroma, other_chroma)])
        vector = (
            chroma                                # 12  (v1, unchanged)
            + self._timbre(start_bar)             # 12  (v1, unchanged)
            + self._density(start_bar)            #  4  (v1, unchanged)
            + self._loudness(start_bar)           #  4  (v1, unchanged)
            + [energy]                            #  1
            + [slope]                             #  1
            + onehot                              #  5
            + [since]                             #  1
            + [self._vocal_presence(start_bar)]   #  1
            + [0.0] * RESERVED_DIMS               #  7
            + self._onset_bits(start_bar)         # 64
            + bass_chroma                         # 12
            + other_chroma                        # 12
            + bass_root_vec                       # 12
            + chord_root_vec                      # 12
        )
        if len(vector) != FEATURE_DIM:  # pragma: no cover - layout guard
            raise AssertionError(
                f"feature layout is {len(vector)}, expected {FEATURE_DIM}")
        is_intro = bool(start_bar <= WINDOW_BARS or label == "INTRO")
        is_outro = bool(start_bar + WINDOW_BARS >= self.n_bars - WINDOW_BARS)
        meta: Dict[str, Any] = {
            "chroma": _round_vec(chroma),
            "bass_root": int(bass_root),
            "is_intro": is_intro,
            "is_outro": is_outro,
            "once": bool(is_intro or is_outro),
        }
        return _round_vec(vector), meta

    def features(self, start_bar: int) -> List[float]:
        """The 160-d feature vector of the 4-bar window at ``start_bar``."""
        return self.describe(start_bar)[0]


# ---------------------------------------------------------------------------
# Per-track worker
# ---------------------------------------------------------------------------

def _process_track(job: Dict[str, Any]) -> Dict[str, Any]:
    """Read one analysis JSON and emit its window features."""
    path = Path(str(job["analysis"]))
    kit_starts = [int(b) for b in (job.get("kit_starts") or [])]
    result: Dict[str, Any] = {
        "track_id": path.stem, "status": "ok", "reason": "",
        "track": None, "windows": [],
        "anomaly": {"n_tatums": 0, "n_bars": 0, "tatum_remainder": 0},
    }

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"analysis_error:{type(exc).__name__}"
        return result

    tatums = raw.get("tatums") or []
    segments = raw.get("segments") or []
    track_id = str(raw.get("track_id") or path.stem)
    result["track_id"] = track_id
    n_bars = len(tatums) // TATUMS_PER_BAR
    result["anomaly"] = {"n_tatums": len(tatums), "n_bars": n_bars,
                         "tatum_remainder": len(tatums) % TATUMS_PER_BAR}

    if n_bars < WINDOW_BARS:
        result["status"] = "skipped"
        result["reason"] = "too_short"
        return result

    try:
        bar_feats = bar_features(segments, tatums)
        spans = segment_sections(bar_feats)
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"section_error:{type(exc).__name__}"
        return result
    n_bars = min(n_bars, len(bar_feats))

    try:
        featurizer = _WindowFeaturizer(raw, bar_feats, spans, n_bars)
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"feature_error:{type(exc).__name__}"
        return result

    max_start = n_bars - WINDOW_BARS
    grid = list(range(0, max_start + 1, WINDOW_BARS))
    on_grid = set(grid)
    # Off-grid kit phrase starts: join them exactly instead of dropping them.
    extra = sorted({b for b in kit_starts
                    if 0 <= b <= max_start and b not in on_grid})

    windows: List[Dict[str, Any]] = []
    bad_windows = 0
    for start_bar in grid + extra:
        try:
            vector, meta = featurizer.describe(start_bar)
            energy = featurizer.energy(start_bar)[0]
        except Exception as exc:
            result["status"] = "error"
            result["reason"] = f"window_error:{type(exc).__name__}@{start_bar}"
            return result
        if featurizer.window_has_bad_onset(start_bar):
            bad_windows += 1
        windows.append({
            "start_bar": int(start_bar),
            "section": str(featurizer.sec_of_bar[start_bar]),
            "energy": _r5(energy),
            "aligned": bool(start_bar % WINDOW_BARS == 0),
            **meta,
            "features": vector,
        })
    windows.sort(key=lambda w: w["start_bar"])

    onset = dict(featurizer.onset_bad)
    onset["windows_affected"] = int(bad_windows)
    onset["total"] = int(sum(featurizer.onset_bad.values()))
    result["onset"] = onset

    bpm = float(raw.get("bpm") or 0.0)
    result["track"] = {
        "track_id": track_id,
        "style": _classify_style(str(raw.get("filename") or ""), bpm),
        "bpm": _r5(bpm),
        "key": str(raw.get("key") or ""),
        "windows": windows,
    }
    return result


# ---------------------------------------------------------------------------
# Manifest handling + kit join
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("kits"), list):
        raise SystemExit(f"{path}: not a kit manifest (missing 'kits')")
    return payload


def _manifest_phrases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for kit in manifest.get("kits") or []:
        if not isinstance(kit, dict):
            continue
        for phrase in kit.get("phrases") or []:
            if not isinstance(phrase, dict):
                continue
            out.append({
                "phrase_id": str(phrase.get("id") or ""),
                "track_id": str(kit.get("track_id") or ""),
                "start_bar": int(phrase.get("start_bar") or 0),
            })
    out.sort(key=lambda p: (p["phrase_id"], p["track_id"], p["start_bar"]))
    return out


def _analysis_files(processed: Path) -> List[Path]:
    return sorted(p for p in processed.glob("*.json")
                  if not p.name.startswith(("_", "meta_")))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(dataset: Dict[str, Any], n_phrases: int, matched: int,
              min_windows: int, min_frac: float) -> List[str]:
    """Assert the delivery contract on the assembled payload."""
    problems: List[str] = []
    if int(dataset["feature_dim"]) != FEATURE_DIM:
        problems.append(f"feature_dim={dataset['feature_dim']} != {FEATURE_DIM}")

    bad_dim = 0
    non_finite = 0
    bad_meta = 0
    bad_bits = 0
    bad_onehot = 0
    meta_mismatch = 0
    vectors = [w["features"] for t in dataset["tracks"] for w in t["windows"]]
    vectors += [p["features"] for p in dataset["kit_phrases"]]
    for vec in vectors:
        if len(vec) != FEATURE_DIM:
            bad_dim += 1
            continue
        if not all(math.isfinite(v) for v in vec):
            non_finite += 1
            continue
        bits = vec[BASE_FEATURE_DIM:BASE_FEATURE_DIM + ONSET_DIM]
        if any(v not in (0.0, 1.0) for v in bits):
            bad_bits += 1
        # 48..159 must contain the 4 one-hot blocks (2 chroma, 2 one-hot).
        roots = (
            vec[BASE_FEATURE_DIM + ONSET_DIM + 2 * CHROMA_DIM:
                BASE_FEATURE_DIM + ONSET_DIM + 3 * CHROMA_DIM],
            vec[BASE_FEATURE_DIM + ONSET_DIM + 3 * CHROMA_DIM:],
        )
        if any(abs(sum(block) - 1.0) > 1e-4 for block in roots):
            bad_onehot += 1
    if bad_dim:
        problems.append(f"{bad_dim} vector(s) with dimension != {FEATURE_DIM}")
    if non_finite:
        problems.append(f"{non_finite} vector(s) with NaN/inf")
    if bad_bits:
        problems.append(f"{bad_bits} vector(s) with non-binary onset bits")
    if bad_onehot:
        problems.append(f"{bad_onehot} vector(s) with a broken root one-hot")

    def _check_meta(row: Dict[str, Any]) -> None:
        nonlocal bad_meta, meta_mismatch
        chroma = row.get("chroma")
        root = row.get("bass_root")
        flags = (row.get("is_intro"), row.get("is_outro"), row.get("once"))
        if (not isinstance(chroma, list) or len(chroma) != CHROMA_DIM
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in chroma)
                or not isinstance(root, int) or not 0 <= root < CHROMA_DIM
                or not all(isinstance(v, bool) for v in flags)
                or flags[2] != (flags[0] or flags[1])):
            bad_meta += 1
            return
        vec = row.get("features") or []
        if len(vec) != FEATURE_DIM:
            return
        mismatch = any(abs(float(a) - float(b)) > 1e-6
                       for a, b in zip(chroma, vec[:CHROMA_DIM]))
        bass = vec[BASE_FEATURE_DIM + ONSET_DIM:
                   BASE_FEATURE_DIM + ONSET_DIM + CHROMA_DIM]
        if int(np.argmax(np.asarray(bass, dtype=np.float64))) != root:
            mismatch = True
        if mismatch:
            meta_mismatch += 1

    for track in dataset["tracks"]:
        for window in track["windows"]:
            _check_meta(window)
    for phrase in dataset["kit_phrases"]:
        _check_meta(phrase)
    if bad_meta:
        problems.append(f"{bad_meta} row(s) with missing/invalid window metadata")
    if meta_mismatch:
        problems.append(f"{meta_mismatch} row(s) whose metadata disagrees with "
                        f"its feature vector")

    if matched != n_phrases:
        problems.append(f"kit join {matched}/{n_phrases} phrases")

    tracks = dataset["tracks"]
    ok_tracks = sum(1 for t in tracks if len(t["windows"]) >= min_windows)
    frac = ok_tracks / max(1, len(tracks))
    if frac < min_frac:
        problems.append(f"only {frac:.1%} of tracks have >= {min_windows} windows "
                        f"(need {min_frac:.0%})")
    return problems


# ---------------------------------------------------------------------------
# CLI + driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the 160-d, 4-bar-window taste dataset from analysis JSON.")
    ap.add_argument("--processed", default="processed",
                    help="directory of per-track analysis JSON (default: processed)")
    ap.add_argument("--kits", default="kits_full/kit_manifest.json",
                    help="kit manifest whose phrases must all join a window")
    ap.add_argument("--out", default="taste_dataset.json", help="output JSON path")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"process pool size (default: {DEFAULT_WORKERS})")
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="debug: only the first N analyses")
    ap.add_argument("--min-windows", type=int, default=8,
                    help="validation floor for windows per track (default: 8)")
    ap.add_argument("--min-track-frac", type=float, default=0.90,
                    help="fraction of tracks that must reach --min-windows "
                         "(default: 0.90)")
    args = ap.parse_args()

    processed = Path(args.processed)
    manifest_path = Path(args.kits)
    out_path = Path(args.out)
    workers = max(1, int(args.workers or DEFAULT_WORKERS))
    t_start = time.time()

    if not processed.is_dir():
        raise SystemExit(f"--processed not a directory: {processed}")
    if not manifest_path.is_file():
        raise SystemExit(f"--kits not a file: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    if int(manifest.get("bars") or WINDOW_BARS) != WINDOW_BARS:
        print(f"  ! manifest bars={manifest.get('bars')} but windows are "
              f"{WINDOW_BARS} bars; joining on start_bar only", flush=True)
    phrases = _manifest_phrases(manifest)
    kit_count = len(manifest.get("kits") or [])

    files = _analysis_files(processed)
    if args.max_tracks:
        files = files[: max(1, int(args.max_tracks))]
    if not files:
        raise SystemExit(f"no analysis JSON under {processed}")

    # Phrase start bars per track: each worker needs the off-grid ones it owns.
    starts_by_track: Dict[str, List[int]] = {}
    for phrase in phrases:
        starts_by_track.setdefault(phrase["track_id"], []).append(phrase["start_bar"])

    print("=== Wub taste dataset ===", flush=True)
    print(f"  processed : {processed} ({len(files)} analyses)", flush=True)
    print(f"  kits      : {manifest_path} ({kit_count} kits, {len(phrases)} phrases)",
          flush=True)
    print(f"  windows   : {WINDOW_BARS} bars, {FEATURE_DIM} features, "
          f"{workers} workers", flush=True)
    print(f"  out       : {out_path}", flush=True)

    jobs = [{"analysis": str(p),
             "kit_starts": sorted(set(starts_by_track.get(p.stem, [])))}
            for p in files]

    tracks: List[Dict[str, Any]] = []
    status: Counter = Counter()
    reasons: Counter = Counter()
    anomalies: List[Dict[str, Any]] = []
    onset_anomalies: List[Dict[str, Any]] = []
    onset_totals: Counter = Counter()
    n_windows = 0
    n_aligned = 0
    done = 0
    t0 = time.time()

    def _absorb(res: Dict[str, Any]) -> None:
        nonlocal n_windows, n_aligned
        status[res.get("status", "error")] += 1
        anomaly = res.get("anomaly") or {}
        if anomaly and (anomaly.get("tatum_remainder")
                        or res.get("status") != "ok"):
            anomalies.append({"track_id": res.get("track_id"),
                              "reason": res.get("reason") or "partial_bar",
                              **anomaly})
        onset = res.get("onset") or {}
        if onset.get("total") or onset.get("windows_affected"):
            onset_anomalies.append({"track_id": res.get("track_id"),
                                    **onset})
            onset_totals["tracks"] += 1
            onset_totals["windows"] += int(onset.get("windows_affected") or 0)
            for key, value in onset.items():
                if isinstance(value, int) and key != "windows_affected":
                    onset_totals[key] += value
        if res.get("status") != "ok" or not res.get("track"):
            if res.get("reason"):
                reasons[str(res["reason"])] += 1
            return
        track = res["track"]
        for window in track["windows"]:
            n_windows += 1
            if window.get("aligned"):
                n_aligned += 1
        tracks.append(track)

    def _progress() -> None:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"  [{done}/{len(jobs)}] {elapsed:6.1f}s "
              f"({rate:5.1f} tracks/s) windows={n_windows} "
              f"errors={status.get('error', 0)}", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_track, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # pragma: no cover - worker crash
                    res = {"track_id": Path(job["analysis"]).stem, "status": "error",
                           "reason": f"worker_crash:{type(exc).__name__}:{exc}",
                           "track": None, "windows": [], "anomaly": {}}
                _absorb(res)
                done += 1
                if done % 100 == 0 or done == len(jobs):
                    _progress()
    else:
        for job in jobs:
            try:
                res = _process_track(job)
            except Exception as exc:  # pragma: no cover - worker crash
                res = {"track_id": Path(job["analysis"]).stem, "status": "error",
                       "reason": f"worker_crash:{type(exc).__name__}:{exc}",
                       "track": None, "windows": [], "anomaly": {}}
            _absorb(res)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                _progress()

    tracks.sort(key=lambda t: t["track_id"])
    onset_anomalies.sort(key=lambda row: str(row.get("track_id")))
    window_index = {t["track_id"]: {w["start_bar"]: w for w in t["windows"]}
                    for t in tracks}

    # --- kit phrase join ----------------------------------------------------
    kit_phrases: List[Dict[str, Any]] = []
    unmatched: Counter = Counter()
    for phrase in phrases:
        windows = window_index.get(phrase["track_id"]) or {}
        window = windows.get(phrase["start_bar"])
        if window is None:
            unmatched[f"{phrase['track_id']}@{phrase['start_bar']}"] += 1
            continue
        kit_phrases.append({
            "phrase_id": phrase["phrase_id"],
            "track_id": phrase["track_id"],
            "start_bar": int(phrase["start_bar"]),
            "chroma": list(window["chroma"]),
            "bass_root": int(window["bass_root"]),
            "is_intro": bool(window["is_intro"]),
            "is_outro": bool(window["is_outro"]),
            "once": bool(window["once"]),
            "features": list(window["features"]),
        })
    kit_phrases.sort(key=lambda p: (p["phrase_id"], p["track_id"], p["start_bar"]))
    matched = len(kit_phrases)

    style_counts = Counter(t["style"] for t in tracks)
    dataset: Dict[str, Any] = {
        "version": 2,
        "feature_dim": FEATURE_DIM,
        "feature_layout": {
            "base": [0, BASE_FEATURE_DIM],
            "onset_bits": [BASE_FEATURE_DIM,
                           BASE_FEATURE_DIM + ONSET_DIM],
            "bass_chroma": [BASE_FEATURE_DIM + ONSET_DIM,
                            BASE_FEATURE_DIM + ONSET_DIM + CHROMA_DIM],
            "other_chroma": [BASE_FEATURE_DIM + ONSET_DIM + CHROMA_DIM,
                             BASE_FEATURE_DIM + ONSET_DIM + 2 * CHROMA_DIM],
            "bass_root": [BASE_FEATURE_DIM + ONSET_DIM + 2 * CHROMA_DIM,
                          BASE_FEATURE_DIM + ONSET_DIM + 3 * CHROMA_DIM],
            "chord_root": [BASE_FEATURE_DIM + ONSET_DIM + 3 * CHROMA_DIM,
                           FEATURE_DIM],
            "onset_stem_order": list(STEMS),
        },
        "tracks": tracks,
        "kit_phrases": kit_phrases,
        "stats": {
            "tracks": len(tracks),
            "windows": n_windows,
            "kits": kit_count,
            "phrases": len(phrases),
            "phrases_matched": matched,
            "aligned_windows": n_aligned,
            "kit_windows": n_windows - n_aligned,
            "feature_dim": FEATURE_DIM,
            "onset_bits": ONSET_DIM,
            "onset_anomalies": {
                "tracks_affected": int(onset_totals.get("tracks", 0)),
                "windows_affected": int(onset_totals.get("windows", 0)),
                "segments_with_fallback": int(onset_totals.get("total", 0)),
                "segments_missing_segment": int(
                    onset_totals.get("missing_segment", 0)),
                "segments_missing_start": int(
                    onset_totals.get("missing_start", 0)),
                "segments_outside_tatums": int(
                    onset_totals.get("outside_tatums", 0)),
                "segments_not_referenced": int(
                    onset_totals.get("not_referenced", 0)),
                "track_details": onset_anomalies,
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, separators=(",", ":"), allow_nan=False)
    tmp_path.replace(out_path)
    out_bytes = out_path.stat().st_size

    problems = _validate(dataset, len(phrases), matched, int(args.min_windows),
                         float(args.min_track_frac))

    # --- summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    window_counts = [len(t["windows"]) for t in tracks]
    short_tracks = [{"track_id": t["track_id"], "windows": len(t["windows"])}
                    for t in tracks if len(t["windows"]) < int(args.min_windows)]
    aligned_frac = n_aligned / max(1, n_windows)

    print("\n=== Summary ===", flush=True)
    print(f"  elapsed          : {elapsed:.1f}s ({workers} workers, "
          f"{len(jobs) / max(1e-9, elapsed):.1f} tracks/s)", flush=True)
    print(f"  tracks           : {len(tracks)} ok / "
          f"{status.get('skipped', 0)} skipped / {status.get('error', 0)} errors",
          flush=True)
    print(f"  windows          : {n_windows} ({n_aligned} aligned = {aligned_frac:.1%}, "
          f"{n_windows - n_aligned} kit)", flush=True)
    print(f"  features         : {FEATURE_DIM} = {BASE_FEATURE_DIM} base + "
          f"{ONSET_DIM} onset bits + {HARMONIC_DIM} harmonic", flush=True)
    print(f"  windows/track    : median {np.median(window_counts):.0f}, "
          f"min {min(window_counts)}, max {max(window_counts)}", flush=True)
    print(f"  tracks >= {args.min_windows} windows: "
          f"{len(tracks) - len(short_tracks)}/{len(tracks)} "
          f"({(len(tracks) - len(short_tracks)) / max(1, len(tracks)):.2%})",
          flush=True)
    print(f"  kit phrases      : {matched}/{len(phrases)} joined "
          f"({matched / max(1, len(phrases)):.2%}), {kit_count} kits", flush=True)
    print("  styles           : " + ", ".join(f"{k}={v}" for k, v
                                               in sorted(style_counts.items())), flush=True)
    print(f"  output           : {out_path} ({out_bytes / 1e6:.2f} MB, "
          f"{out_bytes / max(1, n_windows + matched):.0f} B/window)", flush=True)

    if short_tracks:
        print(f"  ! tracks with < {args.min_windows} windows: "
              + ", ".join(f"{t['track_id']}={t['windows']}" for t in short_tracks[:10]),
              flush=True)
    partial = [a for a in anomalies if a.get("tatum_remainder")]
    if partial:
        print(f"  ! {len(partial)} analysis(es) with tatums % 16 != 0 "
              f"(partial last bar ignored): "
              + ", ".join(f"{a['track_id']}({a['n_tatums']})" for a in partial[:10]),
              flush=True)
    if onset_totals.get("total") or onset_totals.get("windows"):
        print(f"  ! onset fallbacks : {onset_totals.get('total', 0)} segment(s) "
              f"in {onset_totals.get('tracks', 0)} track(s), "
              f"{onset_totals.get('windows', 0)} window(s) affected "
              f"(missing_segment={onset_totals.get('missing_segment', 0)}, "
              f"missing_start={onset_totals.get('missing_start', 0)}, "
              f"outside_tatums={onset_totals.get('outside_tatums', 0)}, "
              f"not_referenced={onset_totals.get('not_referenced', 0)})",
              flush=True)
    else:
        print("  onset bits       : 100% derived from segment starts "
              "(no fallbacks)", flush=True)
    if reasons:
        print("  skip reasons     : " + ", ".join(f"{k}={v}" for k, v
                                                   in sorted(reasons.items())), flush=True)
    if unmatched:
        print(f"  ! {sum(unmatched.values())} unmatched kit phrase(s), e.g. "
              + ", ".join(list(sorted(unmatched))[:5]), flush=True)

    print("  validation       : " + ("PASS" if not problems else
                                     "FAIL -> " + "; ".join(problems)), flush=True)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
