"""Hierarchical joint stem-cohesion on GLOBAL vocab IDs.

Mining operates on bars already remapped to global vocab IDs via
``GlobalVocabularyBuilder.remap_bars_fast``: each bar is a dict
``{stem: [gid-or-None x16]}`` with stems drums/bass/other/vocals.

Provides:
- :func:`mine_joint_tuples` — joint 4-stem signature counts + PMI compatibility
  ``p(d,b,o,v) / (p(d) p(b) p(o) p(v))`` with per-section breakdowns.
- :func:`mine_call_response` — 2-bar transitions, per-stem self + cross-stem
  (e.g. ``vocals->drums``), conditioned on section via ``SECTION|prev_sig`` keys.
- :func:`passes_theory` — theory gate (pitch-shift <=6, kick/sub collision).
- :func:`rerank_bars` / :func:`rank_candidates` — weighted rerank
  ``0.5*learned_pmi + 0.3*audio_sim + 0.2*theory``.

Pure Python + numpy, deterministic (RandomState(42) where randomness needed).
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wub_core.global_vocabulary import best_pitch_shift

logger = logging.getLogger("wub.global_joint_cohesion")

STEMS: Tuple[str, ...] = ("drums", "bass", "other", "vocals")
PITCHED_STEMS: Tuple[str, ...] = ("bass", "other", "vocals")
KICK_STEPS: Tuple[int, ...] = (0, 4, 8, 12)

RNG_SEED = 42


# ---------------------------------------------------------------------------
# signature helpers
# ---------------------------------------------------------------------------

def bar_stem_sig(tokens: Optional[Sequence[Optional[str]]]) -> str:
    """Per-stem bar signature: global IDs joined by '-', None -> '.'."""
    if not tokens:
        return "-".join(["."] * 16)
    parts: List[str] = []
    for g in list(tokens)[:16]:
        parts.append(g if g else ".")
    while len(parts) < 16:
        parts.append(".")
    return "-".join(parts)


def joint_sig(bar: Dict[str, Any]) -> str:
    """Joint 4-stem signature: per-stem sigs joined by '|' in STEMS order."""
    if not isinstance(bar, dict):
        return "|||"
    return "|".join(bar_stem_sig(bar.get(s)) for s in STEMS)


def split_joint_sig(sig: str) -> Dict[str, str]:
    """Split a joint sig back into {stem: per-stem sig}."""
    parts = str(sig).split("|")
    # per-stem sigs never contain '|', so exactly 4 parts expected.
    while len(parts) < 4:
        parts.append("")
    return {s: parts[i] for i, s in enumerate(STEMS)}


def _normalize_tracks(
    bars_per_track: Any,
) -> List[List[Dict[str, Any]]]:
    """Accept flat List[Bar] or nested List[List[Bar]] -> always nested."""
    if not bars_per_track:
        return []
    items = list(bars_per_track)
    if not items:
        return []
    first = items[0]
    # Flat = first element looks like a Bar dict (has stem keys).
    if isinstance(first, dict) and any(k in STEMS for k in first.keys()):
        return [items]
    # Otherwise assume nested.
    out: List[List[Dict[str, Any]]] = []
    for tr in items:
        if tr is None:
            out.append([])
        elif isinstance(tr, dict):
            out.append([tr])
        else:
            try:
                out.append(list(tr))
            except TypeError:
                out.append([])
    return out


def _normalize_sections(
    sections: Any, n_tracks: int
) -> List[List[Optional[str]]]:
    """Normalize section labels to parallel nested lists (may be all-None)."""
    if sections is None:
        return [[None] * 0 for _ in range(n_tracks)]
    secs = list(sections)
    if not secs:
        return [[None] * 0 for _ in range(n_tracks)]
    # Flat list of str/None -> single track.
    if all(s is None or isinstance(s, str) for s in secs):
        # Ambiguous when n_tracks > 1; treat as single-track only if n_tracks == 1.
        if n_tracks == 1:
            return [list(secs)]
        # Otherwise assume one label per track (not per bar) -> expand?
        # Fall through to nested handling.
        if len(secs) == n_tracks and all(s is None or isinstance(s, str) for s in secs):
            return [[s] for s in secs]
        return [list(secs)] + [[None] * 0 for _ in range(n_tracks - 1)]
    # Nested.
    out: List[List[Optional[str]]] = []
    for tr in secs:
        if tr is None:
            out.append([])
        elif isinstance(tr, str):
            out.append([tr])
        else:
            try:
                out.append(list(tr))
            except TypeError:
                out.append([])
    while len(out) < n_tracks:
        out.append([])
    return out[:n_tracks]


# ---------------------------------------------------------------------------
# joint tuple mining (PMI)
# ---------------------------------------------------------------------------

def mine_joint_tuples(
    bars_per_track: Any,
    sections_per_track: Any = None,
    top_n: int = 2000,
) -> List[Dict[str, Any]]:
    """Mine joint 4-stem tuple counts + PMI compatibility.

    ``bars_per_track``: flat ``List[Bar]`` or nested ``List[List[Bar]]`` where
    each Bar is ``{stem: [gid-or-None x16]}`` from ``remap_bars_fast``.
    ``sections_per_track``: optional parallel section labels (flat or nested).

    Returns top-``top_n`` entries sorted by ``count`` desc:
    ``{sig, count, pmi, pmi_ratio, sections, stems}`` where ``pmi`` is natural-log
    PMI ``log(p_joint / prod p_marginals)`` and ``pmi_ratio`` is the raw ratio.
    """
    tracks = _normalize_tracks(bars_per_track)
    n_tracks = len(tracks)
    if n_tracks == 0:
        return []
    if sections_per_track is None:
        sec_tracks: List[List[Optional[str]]] = [[None] * len(tr) for tr in tracks]
    else:
        sec_tracks = _normalize_sections(sections_per_track, n_tracks)
        # Pad/truncate to track lengths.
        padded: List[List[Optional[str]]] = []
        for ti, tr in enumerate(tracks):
            s = sec_tracks[ti] if ti < len(sec_tracks) else []
            s = list(s) + [None] * max(0, len(tr) - len(s))
            padded.append(s[: len(tr)])
        sec_tracks = padded

    joint_counts: Counter = Counter()
    stem_counts: Dict[str, Counter] = {s: Counter() for s in STEMS}
    section_joint: Dict[str, Counter] = defaultdict(Counter)
    joint_to_stems: Dict[str, Dict[str, str]] = {}
    total = 0

    for ti, track_bars in enumerate(tracks):
        secs = sec_tracks[ti] if ti < len(sec_tracks) else [None] * len(track_bars)
        for bi, bar in enumerate(track_bars):
            if not isinstance(bar, dict):
                continue
            per = {s: bar_stem_sig(bar.get(s)) for s in STEMS}
            jsig = "|".join(per[s] for s in STEMS)
            joint_counts[jsig] += 1
            for s in STEMS:
                stem_counts[s][per[s]] += 1
            sec = secs[bi] if bi < len(secs) else None
            sec_key = str(sec) if sec is not None else "ALL"
            section_joint[sec_key][jsig] += 1
            if jsig not in joint_to_stems:
                joint_to_stems[jsig] = dict(per)
            total += 1

    if total == 0:
        return []

    results: List[Dict[str, Any]] = []
    for jsig, c in joint_counts.items():
        p_joint = float(c) / float(total)
        prod = 1.0
        for s in STEMS:
            prod *= float(stem_counts[s][joint_to_stems[jsig][s]]) / float(total)
        prod = max(prod, 1e-12)
        ratio = p_joint / prod
        pmi = math.log(max(ratio, 1e-12))
        sec_counts = {
            sec: int(cnt[jsig]) for sec, cnt in section_joint.items() if jsig in cnt
        }
        results.append(
            {
                "sig": jsig,
                "count": int(c),
                "pmi": round(float(pmi), 4),
                "pmi_ratio": round(float(ratio), 4),
                "sections": sec_counts,
                "stems": dict(joint_to_stems[jsig]),
            }
        )
    # Deterministic: count desc, pmi desc, sig asc.
    results.sort(key=lambda d: (-int(d["count"]), -float(d["pmi"]), str(d["sig"])))
    cap = max(0, int(top_n))
    return results[:cap]


def build_pmi_lookup(
    joint_tuples: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Map joint sig -> normalized 0..1 compatibility score.

    Uses ``pmi_ratio / (1 + pmi_ratio)`` (= sigmoid of log-PMI); independent
    tuples map to 0.5, strongly associated to ~1.0.
    """
    out: Dict[str, float] = {}
    for entry in joint_tuples or []:
        if not isinstance(entry, dict) or "sig" not in entry:
            continue
        sig = str(entry["sig"])
        if "pmi_ratio" in entry:
            try:
                r = float(entry["pmi_ratio"])
            except (TypeError, ValueError):
                r = 1.0
            out[sig] = float(r / (1.0 + r)) if r >= 0 else 0.5
        elif "pmi" in entry:
            try:
                p = float(entry["pmi"])
            except (TypeError, ValueError):
                p = 0.0
            out[sig] = float(1.0 / (1.0 + math.exp(-p)))
        else:
            out[sig] = 0.5
    return out


def _normalize_pmi_value(v: float) -> float:
    """Best-effort normalization of a raw PMI-ish value to 0..1."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(f):
        return 0.5
    if f < 0.0:
        # log-PMI negative -> sigmoid
        return float(1.0 / (1.0 + math.exp(-f)))
    if f > 1.0:
        # raw ratio -> ratio/(1+ratio)
        return float(f / (1.0 + f))
    return float(f)


# ---------------------------------------------------------------------------
# call-response mining (2-bar pairs, self + cross-stem)
# ---------------------------------------------------------------------------

def mine_call_response(
    bar_seqs: Any,
    section_seqs: Any = None,
    max_next: int = 8,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mine 2-bar call-response transitions.

    For each consecutive bar pair (t -> t+1) within a track, records:
    - self transitions per stem: ``prev_sig(stem) -> next_sig(stem)`` under key ``stem``;
    - cross-stem transitions for every ordered pair ``ps != ns`` under key
      ``"{ps}->{ns}"`` (e.g. vocal bar t -> drum bar t+1 under ``vocals->drums``).

    When ``section_seqs`` is given, the previous key is prefixed
    ``"SECTION|prev_sig"`` so transitions stay conditioned on section.
    Probs are ``count / total`` per prev (top-``max_next`` kept, matching
    ``train_global_patterns`` convention).
    """
    tracks = _normalize_tracks(bar_seqs)
    n_tracks = len(tracks)
    if n_tracks == 0:
        return {s: {} for s in STEMS}
    if section_seqs is None:
        sec_tracks: List[List[Optional[str]]] = [[None] * len(tr) for tr in tracks]
    else:
        sec_tracks = _normalize_sections(section_seqs, n_tracks)
        padded: List[List[Optional[str]]] = []
        for ti, tr in enumerate(tracks):
            s = sec_tracks[ti] if ti < len(sec_tracks) else []
            s = list(s) + [None] * max(0, len(tr) - len(s))
            padded.append(s[: len(tr)])
        sec_tracks = padded

    counts: Dict[str, Dict[str, Counter]] = {}

    def _bump(key: str, prev: str, nxt: str) -> None:
        d = counts.setdefault(key, {})
        c = d.get(prev)
        if c is None:
            c = Counter()
            d[prev] = c
        c[nxt] += 1

    for ti, bars in enumerate(tracks):
        secs = sec_tracks[ti] if ti < len(sec_tracks) else [None] * len(bars)
        for t in range(len(bars) - 1):
            prev_bar = bars[t]
            next_bar = bars[t + 1]
            if not isinstance(prev_bar, dict) or not isinstance(next_bar, dict):
                continue
            sec = secs[t] if t < len(secs) else None
            prefix = f"{sec}|" if sec is not None else ""
            prev_sigs = {s: bar_stem_sig(prev_bar.get(s)) for s in STEMS}
            next_sigs = {s: bar_stem_sig(next_bar.get(s)) for s in STEMS}
            for s in STEMS:
                _bump(s, f"{prefix}{prev_sigs[s]}", next_sigs[s])
            for ps in STEMS:
                for ns in STEMS:
                    if ps == ns:
                        continue
                    _bump(f"{ps}->{ns}", f"{prefix}{prev_sigs[ps]}", next_sigs[ns])

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, prev_map in counts.items():
        out[key] = {}
        for prev, ctr in prev_map.items():
            tot = sum(ctr.values())
            if tot <= 0:
                continue
            top = ctr.most_common(max(0, int(max_next)))
            out[key][prev] = {nxt: round(float(c) / float(tot), 4) for nxt, c in top}
    return out


# ---------------------------------------------------------------------------
# theory gate
# ---------------------------------------------------------------------------

def _lookup_slice(
    gid: Optional[str], stem_hint: str, vocab: Any
) -> Tuple[int, str]:
    """Return (pitch 0..11, category str) for a global slice ID."""
    if gid is None:
        return 0, ""
    if vocab is not None:
        # GlobalVocabulary instance shape.
        try:
            vbs = getattr(vocab, "vocab_by_stem", None)
            if isinstance(vbs, dict):
                order = [stem_hint] + [s for s in STEMS if s != stem_hint]
                for s in order:
                    bucket = vbs.get(s) or []
                    for sl in bucket:
                        if isinstance(sl, dict):
                            sid = sl.get("id")
                            if sid == gid:
                                return int(sl.get("pitch", 0)), str(sl.get("category", ""))
                        else:
                            if getattr(sl, "id", None) == gid:
                                return int(getattr(sl, "pitch", 0)), str(
                                    getattr(sl, "category", "")
                                )
        except Exception:
            pass
        # Dict shapes.
        if isinstance(vocab, dict):
            try:
                if gid in vocab and isinstance(vocab[gid], dict):
                    info = vocab[gid]
                    return int(info.get("pitch", 0)), str(info.get("category", gid))
            except Exception:
                pass
            for s in STEMS:
                bucket = vocab.get(s)
                if isinstance(bucket, dict) and gid in bucket:
                    info = bucket[gid]
                    if isinstance(info, dict):
                        try:
                            return int(info.get("pitch", 0)), str(info.get("category", ""))
                        except Exception:
                            continue
                    try:
                        return int(getattr(info, "pitch", 0)), str(
                            getattr(info, "category", "")
                        )
                    except Exception:
                        continue
                if isinstance(bucket, list):
                    for sl in bucket:
                        sid = sl.get("id") if isinstance(sl, dict) else getattr(sl, "id", None)
                        if sid == gid:
                            if isinstance(sl, dict):
                                try:
                                    return int(sl.get("pitch", 0)), str(sl.get("category", ""))
                                except Exception:
                                    continue
                            try:
                                return int(getattr(sl, "pitch", 0)), str(
                                    getattr(sl, "category", "")
                                )
                            except Exception:
                                continue
    # Fallback: pitch unknown (0), category inferred from the ID string itself
    # so synthetic IDs containing 'kick'/'sub' still gate correctly.
    return 0, str(gid or "")


def _chord_root(chord: Any) -> Optional[int]:
    if chord is None:
        return None
    if isinstance(chord, (int, float, np.integer, np.floating)):
        return int(chord) % 12
    if isinstance(chord, (list, tuple)) and chord:
        try:
            return int(chord[0]) % 12
        except (TypeError, ValueError):
            return 0
    if isinstance(chord, dict):
        for k in ("root", "chord_root", "pitch", "target_pitch", "bass_pitch"):
            if k in chord:
                try:
                    return int(chord[k]) % 12
                except (TypeError, ValueError):
                    continue
        return 0
    try:
        return int(chord) % 12  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def passes_theory(
    bar: Dict[str, Any],
    chord: Any = None,
    vocab: Any = None,
) -> bool:
    """Theory gate for one joint bar.

    - Pitch: every pitched-stem slice must be transposable to ``chord`` root
      with ``abs(best_pitch_shift) <= 6`` (always true by construction — the
      playback engine never shifts more than half an octave — but enforced
      explicitly per spec).
    - Rhythm: kick steps (0,4,8,12) with a ``kick`` drum slice sounding against
      a ``sub`` bass slice on the same step -> reject (sidechain collision).
    - Vocal band: dense vocal bars are logged only, never rejected.
    """
    if not isinstance(bar, dict):
        return False
    target = _chord_root(chord)
    if target is not None:
        for stem in PITCHED_STEMS:
            toks = bar.get(stem) or []
            for gid in list(toks)[:16]:
                if gid is None:
                    continue
                pitch, _cat = _lookup_slice(gid, stem, vocab)
                try:
                    shift = int(best_pitch_shift(int(pitch), int(target)))
                except Exception:
                    continue
                if abs(shift) > 6:
                    logger.info(
                        "theory reject: pitch shift %d >6 stem=%s gid=%s target=%d",
                        shift, stem, gid, target,
                    )
                    return False
    drum_toks = list(bar.get("drums") or [])
    bass_toks = list(bar.get("bass") or [])
    for step in KICK_STEPS:
        if step >= len(drum_toks) or step >= len(bass_toks):
            continue
        d_gid = drum_toks[step]
        b_gid = bass_toks[step]
        if d_gid is None or b_gid is None:
            continue
        _, d_cat = _lookup_slice(d_gid, "drums", vocab)
        _, b_cat = _lookup_slice(b_gid, "bass", vocab)
        if "kick" in str(d_cat).lower() and "sub" in str(b_cat).lower():
            logger.info(
                "theory reject: kick/sub collision at step %d d=%s b=%s",
                step, d_gid, b_gid,
            )
            return False
    try:
        voc_toks = bar.get("vocals") or []
        n_voc = sum(1 for g in list(voc_toks)[:16] if g is not None)
        if n_voc > 12:
            logger.info("vocal band note: dense vocal bar (%d onsets), log only", n_voc)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# rerank helper: 0.5*learned_pmi + 0.3*audio_sim + 0.2*theory
# ---------------------------------------------------------------------------

def rerank_bars(
    candidate_bars: Sequence[Dict[str, Any]],
    section: Optional[str] = None,
    chord: Any = None,
    vocab: Any = None,
    pmi_lookup: Optional[Dict[str, float]] = None,
    joint_tuples: Optional[Sequence[Dict[str, Any]]] = None,
    audio_scores: Any = None,
    synth_waves: Any = None,
    ref_wave: Any = None,
    sr: int = 22050,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> List[Dict[str, Any]]:
    """Rank candidate joint bars by learned PMI + audio sim + theory.

    - ``pmi_lookup``: ``{joint_sig: score}`` (raw or normalized; normalized
      via :func:`_normalize_pmi_value`). When omitted but ``joint_tuples``
      is given, the lookup is built with :func:`build_pmi_lookup` plus a
      section-support blend when ``section`` matches an entry's sections.
    - ``audio_scores``: precomputed per-candidate scores (list or {idx: score}).
      When omitted but ``synth_waves`` + ``ref_wave`` are given, scores come
      from ``pattern_audio_trainer.compute_audio_similarity``; otherwise 0.5.
    - theory is 1.0 (pass) / 0.0 (fail) via :func:`passes_theory`.

    Returns entries sorted by score desc (deterministic tie-break on sig):
    ``{index, bar, sig, score, pmi, audio, theory}``.
    """
    _rng = np.random.RandomState(RNG_SEED)  # reserved for deterministic tie-breaks
    _ = _rng
    cands = list(candidate_bars or [])
    lookup: Dict[str, float] = dict(pmi_lookup) if pmi_lookup else {}
    section_blend: Dict[str, float] = {}
    if not lookup and joint_tuples:
        lookup = build_pmi_lookup(joint_tuples)
        if section is not None:
            for entry in joint_tuples:
                if not isinstance(entry, dict) or "sig" not in entry:
                    continue
                sig = str(entry["sig"])
                try:
                    cnt = max(1, int(entry.get("count", 1)))
                except (TypeError, ValueError):
                    cnt = 1
                secs = entry.get("sections") or {}
                frac = 0.0
                if isinstance(secs, dict):
                    try:
                        frac = float(secs.get(str(section), 0)) / float(cnt)
                    except (TypeError, ValueError, ZeroDivisionError):
                        frac = 0.0
                section_blend[sig] = float(max(0.0, min(1.0, frac)))
    w_pmi, w_audio, w_theory = (float(weights[0]), float(weights[1]), float(weights[2]))

    # Lazily resolve audio via compute_audio_similarity only when needed.
    computed_audio: Dict[int, float] = {}
    need_audio = audio_scores is None and synth_waves is not None and ref_wave is not None
    if need_audio:
        try:
            from wub_core.pattern_audio_trainer import compute_audio_similarity  # noqa: PLC0415

            ref_arr = np.asarray(ref_wave, dtype=np.float32)
            for i, _b in enumerate(cands):
                try:
                    if isinstance(synth_waves, dict):
                        sw = synth_waves.get(i)
                    else:
                        sw = synth_waves[i] if i < len(synth_waves) else None
                    if sw is None:
                        computed_audio[i] = 0.5
                        continue
                    m = compute_audio_similarity(
                        synth_wave=np.asarray(sw, dtype=np.float32),
                        real_wave=ref_arr,
                        sr=int(sr),
                    )
                    computed_audio[i] = float(np.clip(float(m.composite_score), 0.0, 1.0))
                except Exception:
                    computed_audio[i] = 0.5
        except Exception:
            computed_audio = {}

    scored: List[Dict[str, Any]] = []
    for i, bar in enumerate(cands):
        jsig = joint_sig(bar) if isinstance(bar, dict) else ""
        if jsig in lookup:
            pmi_norm = _normalize_pmi_value(lookup[jsig])
            if section is not None and jsig in section_blend:
                pmi_norm = float(0.8 * pmi_norm + 0.2 * section_blend[jsig])
        else:
            pmi_norm = 0.5
        if audio_scores is not None:
            try:
                if isinstance(audio_scores, dict):
                    audio = float(audio_scores.get(i, 0.5))
                else:
                    audio = float(audio_scores[i]) if i < len(audio_scores) else 0.5
            except (TypeError, ValueError, IndexError, KeyError):
                audio = 0.5
        elif i in computed_audio:
            audio = float(computed_audio[i])
        else:
            audio = 0.5
        audio = float(max(0.0, min(1.0, audio)))
        try:
            theory = 1.0 if passes_theory(bar, chord, vocab) else 0.0
        except Exception:
            theory = 0.5
        score = float(w_pmi * pmi_norm + w_audio * audio + w_theory * theory)
        scored.append(
            {
                "index": int(i),
                "bar": bar,
                "sig": jsig,
                "score": round(float(score), 4),
                "pmi": round(float(pmi_norm), 4),
                "audio": round(float(audio), 4),
                "theory": float(theory),
            }
        )
    scored.sort(key=lambda d: (-float(d["score"]), str(d["sig"]), int(d["index"])))
    return scored


def rank_candidates(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Alias of :func:`rerank_bars`."""
    return rerank_bars(*args, **kwargs)
