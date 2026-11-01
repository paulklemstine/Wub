"""Joint 4-stem bar-transition model.

One Markov model over JOINT 4-stem bar tuples so stems sound composed
together, instead of sampling drums/bass/other/vocals quasi-independently.

Pipeline
--------
1. :func:`mine_bar_tuples` chunks each track's tatums into 16-tatum bars and
   encodes every bar as a compact 4-stem signature
   ``(groove_type id, quantized density, downbeat mask)`` per stem.
2. :func:`train_joint_transitions` builds bigram counts of next-tuple given
   ``(current tuple, section)`` plus per-section tuple priors, with add-k
   smoothing so unseen transitions keep nonzero probability.
3. :func:`sample_next` draws the next tuple id with a temperature parameter
   and an optional originality guard (:func:`exceeds_copy_run`).
4. :func:`to_top_compact` prunes a model for small JSON export.
5. :func:`save_model` / :func:`load_model` persist the model as JSON.

Pure Python + numpy only (no torch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

STEMS: List[str] = ["drums", "bass", "other", "vocals"]

# All groove labels the per-stem classifiers in wub_core.patterns can emit,
# plus sentinels for absent stems. Sorted for deterministic ids.
GROOVE_NAMES: List[str] = sorted([
    "breakbeat",
    "fill",
    "four_on_floor",
    "four_on_floor_hats",
    "minimal",
    "hook",
    "pad",
    "riff",
    "offbeat",
    "rolling",
    "stabs",
    "sustained",
    "chop",
    "vocal_call_response",
    "vocal_hook",
    "vocal_stutter",
    "vocal_sustained",
    "vocal_syncopated_chop",
    "vocal_verse",
    "missing",
    "none",
])
GROOVE_TO_ID: Dict[str, int] = {g: i for i, g in enumerate(GROOVE_NAMES)}
MISSING_GROOVE_ID: int = GROOVE_TO_ID["missing"]

BAR_STEPS = 16
DENSITY_LEVELS = 4
MASK_LEVELS = 4

# Fixed section vocabulary (matches PatternBank phrase labels).
SECTIONS: List[str] = ["INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"]

# Per-stem signature packs as code = (groove_id * 4 + density_q) * 4 + db_mask.
CODES_PER_STEM = len(GROOVE_NAMES) * DENSITY_LEVELS * MASK_LEVELS

BarTuple = Tuple[int, int, int, int]
BarEvent = Tuple[str, BarTuple]
SongSequence = List[BarEvent]


# ---------------------------------------------------------------------------
# signature helpers
# ---------------------------------------------------------------------------

def quantize_density(n_onsets: int) -> int:
    """Quantize an onset count (0..16) to 2 bits: 0=silent,1=sparse,2=mid,3=dense."""
    n = int(n_onsets)
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    if n <= 6:
        return 2
    return 3


def downbeat_mask(onset_steps: Sequence[int]) -> int:
    """2-bit mask: bit0 = onset on step 0, bit1 = onset on step 8."""
    s = set(int(x) for x in onset_steps)
    return (1 if 0 in s else 0) | (2 if 8 in s else 0)


def encode_stem(groove: str, n_onsets: int, onset_steps: Sequence[int]) -> int:
    """Pack one stem's (groove, density, downbeat mask) into a small int."""
    gid = GROOVE_TO_ID.get(groove, GROOVE_TO_ID["none"])
    return (gid * DENSITY_LEVELS + quantize_density(n_onsets)) * MASK_LEVELS + downbeat_mask(onset_steps)


def decode_stem(code: int) -> Tuple[str, int, int]:
    """Unpack a stem code into (groove_name, density_q, downbeat_mask)."""
    code = int(code)
    mask = code % MASK_LEVELS
    code //= MASK_LEVELS
    dens = code % DENSITY_LEVELS
    gid = code // DENSITY_LEVELS
    groove = GROOVE_NAMES[gid] if 0 <= gid < len(GROOVE_NAMES) else "none"
    return groove, dens, mask


def tuple_key(tup: Sequence[int]) -> str:
    """Canonical JSON-safe key for a 4-stem BarTuple."""
    return "-".join(str(int(c)) for c in tup)


def parse_tuple_key(key: str) -> BarTuple:
    parts = key.split("-")
    if len(parts) != 4:
        raise ValueError(f"bad BarTuple key: {key!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def _classify(stem: str, onset_grid: List[int], density: float) -> str:
    """Classify one bar's onset grid with the shared wub_core.patterns vocabulary."""
    from wub_core.patterns import (
        classify_bass_groove,
        classify_drum_groove,
        classify_synth_groove,
        classify_vocal_groove,
    )

    if stem == "drums":
        return classify_drum_groove(onset_grid, BAR_STEPS)
    if stem == "bass":
        return classify_bass_groove(onset_grid, BAR_STEPS)
    if stem == "other":
        return classify_synth_groove(onset_grid, density)
    if stem == "vocals":
        return classify_vocal_groove(onset_grid, density)
    return "none"


def _label_phrase(avg_d: float, avg_b: float, avg_tot: float) -> str:
    """Map phrase-mean densities to a section label."""
    if avg_d >= 0.40 and avg_b >= 0.25:
        return "DROP"
    if avg_d >= 0.20 or avg_b >= 0.20:
        return "GROOVE_A"
    if avg_d >= 0.10 or avg_b >= 0.10 or avg_tot >= 0.10:
        return "BUILDUP"
    if avg_tot < 0.04:
        return "BREAKDOWN"
    return "INTRO"


# ---------------------------------------------------------------------------
# mining
# ---------------------------------------------------------------------------

def mine_bar_tuples(
    tracks: Sequence[Any],
    max_bars_per_song: Optional[int] = 64,
) -> List[SongSequence]:
    """Mine joint 4-stem bar tuples, one ``(section, tuple)`` sequence per song.

    Each track's tatums are chunked into consecutive 16-tatum bars (truncated
    to ``max_bars_per_song``).  A tatum counts as a stem onset when the stem
    has a segment on that tatum that differs from the previous tatum's
    segment (bar-start steps always count as onsets when present, so held
    notes re-articulate on the downbeat).  Presence/change detection is
    numpy-vectorized per track; only the cheap per-bar groove classifier runs
    in Python, so ~1800 tracks mine in seconds.

    Tracks with fewer than 16 tatums are skipped.  Stems absent from
    ``stems_available`` are encoded with the ``"missing"`` groove (distinct
    from merely silent bars), so missing-stem tracks are handled explicitly.
    """
    cap = max_bars_per_song if (max_bars_per_song and max_bars_per_song > 0) else None
    sequences: List[SongSequence] = []

    for tr in tracks:
        tatums = list(getattr(tr, "tatums", []) or [])
        n_all = len(tatums)
        if n_all < BAR_STEPS:
            continue
        n_bars = n_all // BAR_STEPS
        if cap is not None:
            n_bars = min(n_bars, cap)
        if n_bars <= 0:
            continue
        n_use = n_bars * BAR_STEPS
        tatums = tatums[:n_use]
        avail = set(getattr(tr, "stems_available", []) or [])

        # Per-stem onset grids, vectorized per track.
        grids: Dict[str, List[List[int]]] = {}
        counts: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        for stem in STEMS:
            if stem not in avail:
                grids[stem] = [[] for _ in range(n_bars)]
                counts[stem] = np.zeros(n_bars, dtype=np.int64)
                masks[stem] = np.zeros(n_bars, dtype=np.int64)
                continue
            seg_ids: List[Optional[str]] = []
            for t in tatums:
                try:
                    segs = (t.stem_segments or {}).get(stem) or []
                except Exception:
                    segs = []
                seg_ids.append(segs[-1] if segs else None)
            ids_arr = np.asarray(seg_ids, dtype=object)
            present = np.asarray([s is not None for s in seg_ids], dtype=bool)
            changed = np.ones(n_use, dtype=bool)
            if n_use > 1:
                # object-array != is elementwise; None compares fine.
                changed[1:] = ids_arr[1:] != ids_arr[:-1]
            bar_start = (np.arange(n_use) % BAR_STEPS) == 0
            onset = present & (changed | bar_start)
            mat = onset.reshape(n_bars, BAR_STEPS)
            grids[stem] = [np.flatnonzero(row).tolist() for row in mat]
            counts[stem] = mat.sum(axis=1).astype(np.int64)
            masks[stem] = (
                mat[:, 0].astype(np.int64) + 2 * mat[:, 8].astype(np.int64)
            )

        # Phrase-level (8-bar) section labels from mean stem densities.
        dens = np.stack(
            [counts[s].astype(float) / BAR_STEPS for s in STEMS], axis=1
        )  # (n_bars, 4)
        bar_sections: List[str] = [""] * n_bars
        for p0 in range(0, n_bars, 8):
            chunk = dens[p0:p0 + 8]
            avg_d = float(chunk[:, 0].mean())
            avg_b = float(chunk[:, 1].mean())
            avg_tot = float(chunk.mean())
            sec = _label_phrase(avg_d, avg_b, avg_tot)
            for b in range(p0, min(p0 + 8, n_bars)):
                bar_sections[b] = sec

        seq: SongSequence = []
        for b in range(n_bars):
            codes: List[int] = []
            for stem in STEMS:
                grid = grids[stem][b]
                n = int(counts[stem][b])
                if stem not in avail:
                    groove = "missing"
                else:
                    groove = _classify(stem, grid, n / BAR_STEPS)
                gid = GROOVE_TO_ID.get(groove, GROOVE_TO_ID["none"])
                code = (gid * DENSITY_LEVELS + quantize_density(n)) * MASK_LEVELS + int(masks[stem][b])
                codes.append(code)
            seq.append((bar_sections[b], (codes[0], codes[1], codes[2], codes[3])))
        sequences.append(seq)

    return sequences


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_joint_transitions(
    sequences: Sequence[SongSequence],
    k: float = 0.5,
) -> Dict[str, Any]:
    """Train bigram counts of next-tuple given ``(current tuple, section)``.

    ``sequences`` is one ``[(section, BarTuple), ...]`` list per song, as
    returned by :func:`mine_bar_tuples`.  The conditioning section is the
    *current* bar's section.  ``k`` is the add-k (Lidstone) smoothing mass:
    query-time probabilities are ``(count + k) / (total + k * V)`` so unseen
    transitions keep nonzero probability.  Per-section tuple priors (bar
    unigram counts) are stored alongside the bigram counts.
    """
    if k < 0:
        raise ValueError("smoothing k must be >= 0")
    seqs = [list(s) for s in sequences]

    vocab_sorted = sorted({tup for seq in seqs for (_, tup) in seq})
    tuple_to_id = {tuple_key(t): i for i, t in enumerate(vocab_sorted)}
    n_seqs = len(seqs)
    n_bars = sum(len(s) for s in seqs)

    tuple_counts = [0] * len(vocab_sorted)
    prior_counts: Dict[str, Dict[str, int]] = {}
    transitions: Dict[str, Dict[str, Dict[str, int]]] = {}
    sections = sorted({sec for seq in seqs for (sec, _) in seq})

    for seq in seqs:
        ids = [(sec, tuple_to_id[tuple_key(tup)]) for (sec, tup) in seq]
        for sec, tid in ids:
            tuple_counts[tid] += 1
            prior_counts.setdefault(sec, {}).setdefault(str(tid), 0)
            prior_counts[sec][str(tid)] += 1
        for (sec_cur, cur), (_, nxt) in zip(ids, ids[1:]):
            transitions.setdefault(sec_cur, {}).setdefault(str(cur), {}).setdefault(str(nxt), 0)
            transitions[sec_cur][str(cur)][str(nxt)] += 1

    return {
        "version": 1,
        "stems": list(STEMS),
        "grooves": list(GROOVE_NAMES),
        "tuples": [list(t) for t in vocab_sorted],
        "tuple_to_id": dict(tuple_to_id),
        "tuple_counts": list(tuple_counts),
        "sections": sections,
        "transitions": transitions,
        "priors": prior_counts,
        "k": float(k),
        "vocab_size": len(vocab_sorted),
        "n_sequences": n_seqs,
        "n_bars": n_bars,
    }


def _resolve_id(model: Dict[str, Any], tup: Union[int, Sequence[int], str, Any]) -> Optional[int]:
    """Resolve a tuple id, BarTuple, or key string to an int id (None if unknown)."""
    v = int(model.get("vocab_size", 0))
    if isinstance(tup, bool):
        return None
    if isinstance(tup, (int, np.integer)):
        i = int(tup)
        return i if 0 <= i < v else None
    if isinstance(tup, str):
        try:
            t = parse_tuple_key(tup)
        except ValueError:
            return None
        return model.get("tuple_to_id", {}).get(tuple_key(t))
    try:
        seq = tuple(int(c) for c in tup)  # type: ignore[union-attr]
    except Exception:
        return None
    if len(seq) != 4:
        return None
    return model.get("tuple_to_id", {}).get(tuple_key(seq))


def _bigram_counts(model: Dict[str, Any], section: str, cur_id: int) -> Dict[str, int]:
    sec_table = model.get("transitions", {}).get(section, {})
    nxt_table = sec_table.get(str(cur_id), {})
    return dict(nxt_table) if isinstance(nxt_table, dict) else {}


def transition_prob(
    model: Dict[str, Any],
    current: Union[int, Sequence[int], str],
    section: str,
    nxt: Union[int, Sequence[int], str],
) -> float:
    """Add-k smoothed P(next | current, section).  Unseen pairs get k/(total+kV) > 0."""
    v = int(model.get("vocab_size", 0))
    if v <= 0:
        return 0.0
    k = float(model.get("k", 0.5))
    cur_id = _resolve_id(model, current)
    nxt_id = _resolve_id(model, nxt)
    if cur_id is None or nxt_id is None:
        return 0.0
    counts = _bigram_counts(model, section, cur_id)
    total = sum(counts.values())
    return (counts.get(str(nxt_id), 0) + k) / (total + k * v)


def prior_prob(
    model: Dict[str, Any],
    section: str,
    tup: Union[int, Sequence[int], str],
) -> float:
    """Add-k smoothed per-section prior P(tuple | section)."""
    v = int(model.get("vocab_size", 0))
    if v <= 0:
        return 0.0
    k = float(model.get("k", 0.5))
    tid = _resolve_id(model, tup)
    if tid is None:
        return 0.0
    sec_table = model.get("priors", {}).get(section, {})
    total = sum(sec_table.values()) if isinstance(sec_table, dict) else 0
    c = sec_table.get(str(tid), 0) if isinstance(sec_table, dict) else 0
    return (c + k) / (total + k * v)


def _full_dist(
    model: Dict[str, Any],
    cur_id: Optional[int],
    section: str,
) -> np.ndarray:
    """Dense smoothed distribution over next ids for (section, cur_id).

    Falls back to the section prior (then the global unigram) when the
    current tuple/section bigram was never observed.
    """
    v = int(model.get("vocab_size", 0))
    k = float(model.get("k", 0.5))
    probs = np.zeros(v, dtype=float)
    if v <= 0:
        return probs
    if cur_id is not None:
        counts = _bigram_counts(model, section, cur_id)
        total = sum(counts.values())
        if total > 0 or counts:
            denom = total + k * v
            probs[:] = k / denom
            for ns, c in counts.items():
                probs[int(ns)] = (c + k) / denom
            return probs
    # fallback: section prior
    sec_table = model.get("priors", {}).get(section, {})
    if isinstance(sec_table, dict) and sec_table:
        total = sum(sec_table.values())
        denom = total + k * v
        probs[:] = k / denom
        for ns, c in sec_table.items():
            probs[int(ns)] = (c + k) / denom
        return probs
    # fallback: global unigram
    counts_all = np.asarray(model.get("tuple_counts", [0] * v)[:v], dtype=float)
    total = float(counts_all.sum())
    denom = total + k * v
    if denom <= 0:
        probs[:] = 1.0 / v
    else:
        probs[:] = (counts_all + k) / denom
    return probs


# ---------------------------------------------------------------------------
# originality guard + sampling
# ---------------------------------------------------------------------------

def exceeds_copy_run(history: Sequence[Any], candidate: Any, max_run: int) -> bool:
    """True if appending ``candidate`` would extend an exact repeat-run past ``max_run``.

    Counts the trailing run of ``candidate`` at the end of ``history``; if
    that run already has length ``max_run``, one more copy would make
    ``max_run + 1`` in a row, so the candidate is refused.  The caller tracks
    the recent generated tuple history and filters candidates with this
    helper to avoid looping or copying long verbatim runs from the corpus.
    """
    run = 0
    for past in reversed(list(history)):
        if past == candidate:
            run += 1
        else:
            break
    return (run + 1) > int(max_run)


def sample_next(
    model: Dict[str, Any],
    current_tuple: Union[int, Sequence[int], str],
    section: str,
    rng: Any = None,
    temperature: float = 1.0,
    history: Optional[Sequence[Any]] = None,
    max_copy_run: Optional[int] = None,
) -> int:
    """Sample the next joint-tuple id given ``(current_tuple, section)``.

    ``temperature`` sharpens (``< 1``) or flattens (``> 1``) the smoothed
    distribution; ``temperature <= 0`` returns the argmax deterministically.
    When ``history`` and ``max_copy_run`` are given, candidates refused by
    :func:`exceeds_copy_run` are removed and the distribution renormalized
    (if every candidate is refused, the guard is ignored rather than
    returning nothing).
    """
    v = int(model.get("vocab_size", 0))
    if v <= 0:
        raise ValueError("empty joint-bar model: no tuples to sample")
    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(None if rng is None else int(rng))

    cur_id = _resolve_id(model, current_tuple)
    probs = _full_dist(model, cur_id, section).astype(float)

    t = float(temperature)
    if t <= 0:
        order = np.argsort(-probs, kind="stable")
        best = int(order[0])
        if history is not None and max_copy_run is not None:
            for cand in order:
                if not exceeds_copy_run(history, int(cand), int(max_copy_run)):
                    return int(cand)
            return best
        return best
    if t != 1.0:
        # p^(1/T) renormalization; probs are strictly positive via smoothing.
        probs = np.power(probs, 1.0 / t)
        s = probs.sum()
        probs = probs / s if s > 0 else np.full(v, 1.0 / v)

    if history is not None and max_copy_run is not None:
        keep = np.array(
            [not exceeds_copy_run(history, i, int(max_copy_run)) for i in range(v)],
            dtype=bool,
        )
        if keep.any():
            probs = probs * keep.astype(float)
            s = probs.sum()
            probs = probs / s if s > 0 else keep.astype(float) / keep.sum()

    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.full(v, 1.0 / v)
    else:
        probs = probs / s
    return int(rng.choice(v, p=probs))


# ---------------------------------------------------------------------------
# compaction + persistence
# ---------------------------------------------------------------------------

def to_top_compact(
    model: Dict[str, Any],
    max_tuples: int = 4000,
    max_next: int = 8,
) -> Dict[str, Any]:
    """Prune a model to the top ``max_tuples`` tuples and top-``max_next``
    next-transitions per ``(section, current)`` key, renormalizing nothing
    (counts stay raw; smoothing still applies at query time).

    Keeps the JSON export small (<2MB for the default caps: a few thousand
    tuples plus a bounded fan-out per key).
    """
    v = int(model.get("vocab_size", 0))
    counts_all: List[int] = list(model.get("tuple_counts", []))
    keep_n = max(0, min(int(max_tuples), v))
    ranked = sorted(range(v), key=lambda i: (-(counts_all[i] if i < len(counts_all) else 0), i))
    kept = sorted(ranked[:keep_n])
    remap = {old: new for new, old in enumerate(kept)}

    old_tuples = model.get("tuples", [])
    new_model: Dict[str, Any] = {
        "version": int(model.get("version", 1)),
        "stems": list(model.get("stems", list(STEMS))),
        "grooves": list(model.get("grooves", list(GROOVE_NAMES))),
        "tuples": [list(old_tuples[o]) for o in kept],
        "tuple_counts": [int(counts_all[o]) if o < len(counts_all) else 0 for o in kept],
        "tuple_to_id": {},
        "sections": list(model.get("sections", [])),
        "transitions": {},
        "priors": {},
        "k": float(model.get("k", 0.5)),
        "vocab_size": len(kept),
        "n_sequences": int(model.get("n_sequences", 0)),
        "n_bars": int(model.get("n_bars", 0)),
    }
    for new, old in enumerate(kept):
        new_model["tuple_to_id"][tuple_key(new_model["tuples"][new])] = new

    old_priors = model.get("priors", {})
    for sec, table in old_priors.items():
        if not isinstance(table, dict):
            continue
        new_table: Dict[str, int] = {}
        for ns, c in table.items():
            try:
                o = int(ns)
            except (TypeError, ValueError):
                continue
            if o in remap:
                new_table[str(remap[o])] = int(c)
        if new_table:
            new_model["priors"][sec] = new_table

    old_trans = model.get("transitions", {})
    for sec, cur_table in old_trans.items():
        if not isinstance(cur_table, dict):
            continue
        new_cur: Dict[str, Dict[str, int]] = {}
        for cs, nxt_table in cur_table.items():
            try:
                o_cur = int(cs)
            except (TypeError, ValueError):
                continue
            if o_cur not in remap or not isinstance(nxt_table, dict):
                continue
            scored = []
            for ns, c in nxt_table.items():
                try:
                    o_nxt = int(ns)
                except (TypeError, ValueError):
                    continue
                if o_nxt in remap:
                    scored.append((int(c), o_nxt))
            scored.sort(key=lambda kv: (-kv[0], kv[1]))
            pruned = scored[: max(0, int(max_next))]
            if pruned:
                new_cur[str(remap[o_cur])] = {str(remap[o]): c for c, o in pruned}
        if new_cur:
            new_model["transitions"][sec] = new_cur

    return new_model


def model_to_json(model: Dict[str, Any]) -> str:
    """Serialize a model to a JSON string."""
    return json.dumps(model)


def model_from_json(payload: Union[str, bytes, Dict[str, Any]]) -> Dict[str, Any]:
    """Deserialize a model from a JSON string/bytes/dict."""
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def save_model(model: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save a model as JSON. Returns the path written."""
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model), encoding="utf-8")
    return p


def load_model(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a model saved with :func:`save_model`."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
