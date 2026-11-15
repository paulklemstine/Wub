"""Essential slice extraction via submodular song reconstitution.

Finds the optimal "essential" bar slices of each song -- the compact dictionary
of drums, bass, synth/other, and vocal bar cells that reconstitutes the maximum
share of the original song's rhythm, harmony, timbre, and structural feel when
recomposed into bars, patterns, phrases, and sections.

Also mines chronological slice-to-slice transition pairs and compressed pattern
role grammars (``A`` primary groove, ``B`` variation/contrast, ``T`` turnaround/fill,
``H`` chorus/hook climax) across both the curated kit bank and raw processed
track analyses (both 4-stem and single-stem virtual-routed tracks).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wub_core.theory_engine import (
    MIN_RMS_DB,
    STEMS,
    chord_triad_pcs,
    classify_triad_quality,
    parse_key,
    score_cell_purity,
)

ESSENTIAL_ROLES: Tuple[str, ...] = ("A", "B", "T", "H")


def _popcount16(x: int) -> int:
    return bin(int(x) & 0xFFFF).count("1")


def _smear_mask16(mask: int) -> Tuple[float, ...]:
    """Soft 16-step onset vector with +/-1 sixteenth-note swing tolerance."""
    m = int(mask) & 0xFFFF
    vec = [0.0] * 16
    for k in range(16):
        if (m >> k) & 1:
            vec[k] += 1.0
            vec[(k - 1) & 15] += 0.25
            vec[(k + 1) & 15] += 0.25
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-8:
        return (0.0,) * 16
    inv = 1.0 / norm
    return tuple(v * inv for v in vec)


def _fallback_chroma(root: int, qual: str) -> Tuple[float, ...]:
    """Synthesize a soft 12-bin chroma profile from (root, quality)."""
    if int(root) < 0:
        return (0.0,) * 12
    pcs = chord_triad_pcs(int(root), str(qual or "mono"))
    vec = [0.0] * 12
    for idx, pc in enumerate(pcs):
        w = 1.0 if idx == 0 else (0.78 if idx == len(pcs) - 1 else 0.65)
        vec[pc % 12] = max(vec[pc % 12], w)
    return tuple(vec)


def _normalize12(vec: Sequence[float]) -> Tuple[float, ...]:
    if not vec or len(vec) != 12:
        return (0.0,) * 12
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-8:
        return (0.0,) * 12
    inv = 1.0 / norm
    return tuple(float(x) * inv for x in vec)


def pitch_invariant_chroma_sim(c_a: Sequence[float], c_b: Sequence[float]) -> float:
    """Max circular-transposed chroma cosine similarity within +/-6 semitones.

    Slightly prefers small transpositions (0.025 decay per semitone) so exact or
    near-key harmonic matches outrank extreme repitching.
    """
    na = _normalize12(c_a)
    nb = _normalize12(c_b)
    if sum(na) <= 1e-8 or sum(nb) <= 1e-8:
        return 0.35
    best = 0.0
    for shift in range(-6, 7):
        dot = sum(na[i] * nb[(i - shift) % 12] for i in range(12))
        pen = 1.0 - 0.025 * abs(shift)
        score = max(0.0, dot * pen)
        if score > best:
            best = score
    return min(1.0, best)


def bar_reconstitution_similarity(
    bar_a: Dict[str, Any],
    bar_b: Dict[str, Any],
    stem: str,
) -> float:
    """Pairwise musical reconstitution similarity in [0, 1] between two bars.

    Measures how well candidate bar ``bar_a`` captures the rhythmic groove,
    pitch-invariant harmony/melodic contour, and loudness/density feel of ``bar_b``.
    """
    pa = float(bar_a.get("purity", 0.0))
    pb = float(bar_b.get("purity", 0.0))
    if pa < 0.08 and pb < 0.08:
        return 0.65
    if pa < 0.08 or pb < 0.08:
        return 0.05

    ra = bar_a.get("_rvec")
    if ra is None:
        ra = _smear_mask16(int(bar_a.get("mask", 0)))
    rb = bar_b.get("_rvec")
    if rb is None:
        rb = _smear_mask16(int(bar_b.get("mask", 0)))

    r_sum_a = sum(ra)
    r_sum_b = sum(rb)
    if r_sum_a > 1e-8 and r_sum_b > 1e-8:
        rhythm_sim = max(0.0, min(1.0, sum(x * y for x, y in zip(ra, rb))))
    elif r_sum_a <= 1e-8 and r_sum_b <= 1e-8:
        rhythm_sim = 0.6
    else:
        rhythm_sim = 0.22

    da = _popcount16(int(bar_a.get("mask", 0))) / 16.0
    db = _popcount16(int(bar_b.get("mask", 0))) / 16.0
    env_sim = max(0.0, 1.0 - 0.55 * abs(pa - pb) - 0.45 * abs(da - db))

    purity_gate = min(1.0, pa / 0.22)
    if stem == "drums":
        raw = 0.72 * rhythm_sim + 0.28 * env_sim
        return min(1.0, raw * purity_gate)

    ca = bar_a.get("_cvec")
    if ca is None:
        chroma_a = bar_a.get("chroma")
        ca = _normalize12(chroma_a) if chroma_a else _fallback_chroma(int(bar_a.get("root", -1)), str(bar_a.get("qual", "mono")))
    cb = bar_b.get("_cvec")
    if cb is None:
        chroma_b = bar_b.get("chroma")
        cb = _normalize12(chroma_b) if chroma_b else _fallback_chroma(int(bar_b.get("root", -1)), str(bar_b.get("qual", "mono")))

    harm_sim = pitch_invariant_chroma_sim(ca, cb)
    raw = 0.48 * rhythm_sim + 0.36 * harm_sim + 0.16 * env_sim
    return min(1.0, raw * purity_gate)


def select_essential_slices_for_song(
    bars_by_stem: Dict[str, List[Dict[str, Any]]],
    max_codebook_per_stem: int = 12,
) -> Dict[str, Any]:
    """Select essential slices per stem that reconstitute the full song.

    Uses greedy submodular facility location per stem to identify the minimal
    set of archetype slices covering the song's weighted bars, then assigns every
    bar its continuous ``essential_score`` (0..1), ``reconstitution_share`` (0..1),
    and compressed structural pattern role (``A``, ``B``, ``T``, or ``H``), plus
    whole-song reconstitution fidelity and chronological role transitions.
    """
    stem_results: Dict[str, Dict[str, Any]] = {}
    stem_fidelities: List[float] = []
    role_transitions: Dict[str, Dict[str, int]] = {
        r: {r2: 0 for r2 in ESSENTIAL_ROLES} for r in ESSENTIAL_ROLES
    }

    for stem in STEMS:
        raw_bars = bars_by_stem.get(stem) or []
        n = len(raw_bars)
        if n == 0:
            stem_results[stem] = {
                "essential_indices": [],
                "essential_scores": [],
                "reconstitution_shares": [],
                "roles": [],
                "fidelity": 0.0,
                "active_bars": 0,
            }
            continue

        # Precompute normalized vectors once per bar for O(N^2) efficiency
        prep: List[Dict[str, Any]] = []
        weights: List[float] = []
        for idx, b in enumerate(raw_bars):
            purity = max(0.0, min(1.0, float(b.get("purity", 0.0))))
            chroma = b.get("chroma")
            root = int(b.get("root", -1))
            qual = str(b.get("qual", "mono"))
            cvec = _normalize12(chroma) if chroma else _fallback_chroma(root, qual)
            sec = str(b.get("section", "GROOVE_A")).upper()
            sec_w = 1.25 if ("DROP" in sec or "CHORUS" in sec) else (1.1 if "GROOVE" in sec else 1.0)
            item = dict(b)
            item["_rvec"] = _smear_mask16(int(b.get("mask", 0)))
            item["_cvec"] = cvec
            item["_idx"] = idx
            prep.append(item)
            weights.append(max(0.04, purity * sec_w))

        tot_w = sum(weights) or 1.0
        active_indices = [i for i, b in enumerate(prep) if float(b.get("purity", 0.0)) >= 0.12]
        if not active_indices:
            stem_results[stem] = {
                "essential_indices": [],
                "essential_scores": [round(0.25 * float(b.get("purity", 0.0)), 3) for b in prep],
                "reconstitution_shares": [0.0] * n,
                "roles": ["A"] * n,
                "fidelity": 0.0,
                "active_bars": 0,
            }
            continue

        # Full NxN similarity matrix among song bars for this stem (fast dot/cosine inner loop)
        rvecs = [b["_rvec"] for b in prep]
        cvecs = [b["_cvec"] for b in prep]
        purities = [float(b.get("purity", 0.0)) for b in prep]
        densities = [_popcount16(int(b.get("mask", 0))) / 16.0 for b in prep]
        r_active = [sum(rv) > 1e-8 for rv in rvecs]
        c_active = [sum(cv) > 1e-8 for cv in cvecs]

        def _fast_pitch_sim(ci: Tuple[float, ...], cj: Tuple[float, ...]) -> float:
            best_c = 0.0
            for sh in range(-6, 7):
                d = sum(ci[k] * cj[(k - sh) % 12] for k in range(12)) * (1.0 - 0.025 * abs(sh))
                if d > best_c:
                    best_c = d
            return 1.0 if best_c > 1.0 else (0.0 if best_c < 0.0 else best_c)

        sim_mat: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            pi = purities[i]
            sim_mat[i][i] = 1.0 if pi >= 0.12 else 0.35
            ri = rvecs[i]
            ci = cvecs[i]
            di = densities[i]
            rai = r_active[i]
            cai = c_active[i]
            for j in range(i + 1, n):
                pj = purities[j]
                if pi < 0.08 and pj < 0.08:
                    sim_mat[i][j] = 0.65
                    sim_mat[j][i] = 0.65
                    continue
                if pi < 0.08 or pj < 0.08:
                    sim_mat[i][j] = 0.05
                    sim_mat[j][i] = 0.05
                    continue
                if rai and r_active[j]:
                    rs = sum(x * y for x, y in zip(ri, rvecs[j]))
                    if rs > 1.0:
                        rs = 1.0
                    elif rs < 0.0:
                        rs = 0.0
                elif not rai and not r_active[j]:
                    rs = 0.6
                else:
                    rs = 0.22
                esim = max(0.0, 1.0 - 0.55 * abs(pi - pj) - 0.45 * abs(di - densities[j]))
                if stem == "drums":
                    core = 0.72 * rs + 0.28 * esim
                else:
                    hsim = _fast_pitch_sim(ci, cvecs[j]) if (cai and c_active[j]) else 0.35
                    core = 0.48 * rs + 0.36 * hsim + 0.16 * esim
                sij = min(1.0, core * min(1.0, pi / 0.22))
                sji = min(1.0, core * min(1.0, pj / 0.22))
                sim_mat[i][j] = sij
                sim_mat[j][i] = sji

        k_budget = max(2, min(max_codebook_per_stem, len(active_indices), math.ceil(n / 6.0) + 1))
        covered = [0.0] * n
        selected: List[int] = []
        selected_set: set[int] = set()
        marginal_shares: Dict[int, float] = {}

        for step in range(k_budget):
            best_i = -1
            best_obj = -1e9
            best_gain = 0.0
            for i in active_indices:
                if i in selected_set:
                    continue
                gain_num = 0.0
                for j in range(n):
                    sij = sim_mat[i][j]
                    if sij > covered[j]:
                        gain_num += weights[j] * (sij - covered[j])
                norm_gain = gain_num / tot_w

                p_i = float(prep[i].get("purity", 0.0))
                hook_bonus = 0.03 if prep[i].get("is_hook") else 0.0
                fill_bonus = 0.025 if prep[i].get("is_fill") else 0.0
                sec_i = str(prep[i].get("section", "")).upper()
                peak_bonus = 0.02 if ("DROP" in sec_i or "GROOVE" in sec_i) else 0.0
                obj = norm_gain + 0.06 * p_i + hook_bonus + fill_bonus + peak_bonus
                if obj > best_obj:
                    best_obj = obj
                    best_i = i
                    best_gain = norm_gain

            if best_i < 0 or (step >= 2 and best_gain < 0.008):
                break
            selected.append(best_i)
            selected_set.add(best_i)
            marginal_shares[best_i] = best_gain
            for j in range(n):
                if sim_mat[best_i][j] > covered[j]:
                    covered[j] = sim_mat[best_i][j]

        active_w = sum(weights[j] for j in active_indices) or tot_w
        fidelity = sum(weights[j] * covered[j] for j in active_indices) / active_w
        fidelity = round(max(0.0, min(1.0, fidelity)), 4)
        stem_fidelities.append(fidelity)

        # Assign cluster archetype (index 0 in `selected` = primary motif A)
        primary_idx = selected[0] if selected else active_indices[0]
        bar_to_archetype: List[int] = [primary_idx] * n
        for j in range(n):
            best_u = primary_idx
            best_sim = -1.0
            for u in selected:
                if sim_mat[u][j] > best_sim:
                    best_sim = sim_mat[u][j]
                    best_u = u
            bar_to_archetype[j] = best_u

        # Raw reconstitution share of each selected slice (how many song bars depend on it)
        cluster_mass: Dict[int, float] = {u: 0.0 for u in selected}
        for j in active_indices:
            u = bar_to_archetype[j]
            cluster_mass[u] = cluster_mass.get(u, 0.0) + weights[j] * sim_mat[u][j]
        max_mass = max(cluster_mass.values()) if cluster_mass else 1.0
        if max_mass <= 1e-8:
            max_mass = 1.0

        scores: List[float] = []
        shares: List[float] = []
        roles: List[str] = []

        for j in range(n):
            pj = float(prep[j].get("purity", 0.0))
            u = bar_to_archetype[j]
            rep_sim = sim_mat[u][j]
            share_j = marginal_shares.get(j, 0.0)
            if j in selected_set:
                mass_norm = cluster_mass.get(j, 0.0) / max_mass
                es = 0.42 + 0.28 * pj + 0.18 * mass_norm + 0.12 * min(1.0, share_j * 3.5)
                if prep[j].get("is_hook") or prep[j].get("is_fill"):
                    es += 0.05
                es = max(0.0, min(0.99, es))
                shares.append(round(share_j, 4))
            else:
                es = 0.08 + 0.34 * pj + 0.16 * rep_sim
                if pj < 0.12:
                    es *= 0.45
                es = max(0.0, min(0.64, es))
                shares.append(round(0.25 * rep_sim * (cluster_mass.get(u, 0.0) / tot_w), 4))
            scores.append(round(es, 3))

            # Assign compressed structural role A / B / T / H
            sec_j = str(prep[j].get("section", "GROOVE_A")).upper()
            b_pos = int(prep[j].get("bar_in_phrase", j % 4)) % 4
            prev_sim = sim_mat[j - 1][j] if j > 0 else 1.0
            next_sec = str(prep[j + 1].get("section", sec_j)).upper() if j + 1 < n else sec_j
            is_fill_j = bool(prep[j].get("is_fill")) and (b_pos == 3)
            is_hook_j = bool(prep[j].get("is_hook"))

            if is_hook_j or (j in selected_set and ("DROP" in sec_j or "CHORUS" in sec_j) and j != primary_idx and pj >= 0.55 and u != primary_idx and selected.index(u) <= 2 and stem in ("vocals", "other")):
                role = "H"
            elif is_fill_j or (b_pos == 3 and (prev_sim < 0.68 or next_sec != sec_j) and pj >= 0.2):
                role = "T"
            elif u == primary_idx or rep_sim >= 0.80 * sim_mat[primary_idx][u]:
                # Check direct similarity to primary archetype so variants cluster cleanly
                role = "A" if sim_mat[primary_idx][j] >= 0.72 or u == primary_idx else "B"
            else:
                role = "B"
            roles.append(role)

        # Accumulate sequential role grammar transitions (weight >=0.12 active bars)
        for j in range(n - 1):
            if float(prep[j].get("purity", 0.0)) >= 0.12 and float(prep[j + 1].get("purity", 0.0)) >= 0.12:
                r1 = roles[j]
                r2 = roles[j + 1]
                role_transitions[r1][r2] += 1

        stem_results[stem] = {
            "essential_indices": selected,
            "essential_scores": scores,
            "reconstitution_shares": shares,
            "roles": roles,
            "fidelity": fidelity,
            "active_bars": len(active_indices),
        }

    mean_fidelity = round(sum(stem_fidelities) / len(stem_fidelities), 4) if stem_fidelities else 0.0
    return {
        "stems": stem_results,
        "mean_fidelity": mean_fidelity,
        "role_transitions": role_transitions,
    }


def _classify_virtual_stem(seg: Dict[str, Any], seg_dur: float) -> str:
    """Route an unseparated master segment into virtual stem role for 1-stem JSONs."""
    centroid = float(seg.get("spectral_centroid") or 1800.0)
    timbre = seg.get("timbre") or []
    t2 = float(timbre[1]) if len(timbre) > 1 else 0.0
    t4 = float(timbre[3]) if len(timbre) > 3 else 0.0
    if (seg_dur < 0.24 and t4 > 8.0) or (centroid > 4200.0 and seg_dur < 0.35):
        return "drums"
    if centroid < 380.0 and seg_dur >= 0.12:
        return "bass"
    if 550.0 <= centroid <= 3200.0 and seg_dur >= 0.28 and abs(t2) < 35.0:
        return "vocals"
    return "other"


def _infer_section_from_norm_pos(pos: float, density: float, energy_norm: float) -> str:
    if pos < 0.14 or energy_norm < 0.30:
        return "INTRO"
    if pos > 0.86:
        return "OUTRO" if energy_norm < 0.45 else "DROP"
    if energy_norm >= 0.78 and density >= 0.35:
        return "DROP"
    if 0.35 <= pos <= 0.78 and energy_norm < 0.48:
        return "BREAKDOWN"
    if energy_norm >= 0.56 and pos >= 0.25:
        return "BUILDUP" if density >= 0.45 else "GROOVE_A"
    return "GROOVE_A"


def mine_processed_track_essential_bars(
    track_doc: Dict[str, Any],
    max_bars: int = 96,
) -> Dict[str, Any]:
    """Extract chronological per-stem bar cells + essential slices from a processed JSON.

    Works on BOTH full 4-stem track analyses and single-stem (``['other']``)
    master analyses by virtually routing unseparated segments when needed, so
    all ~1,896 analyzed corpus tracks contribute essential slices and macro
    bar/phrase/section transitions.
    """
    tid = str(track_doc.get("track_id") or "unknown")
    bpm = float(track_doc.get("bpm") or 125.0)
    if not (55.0 <= bpm <= 200.0):
        bpm = 125.0
    bar_sec = 240.0 / bpm
    key_pc, key_mode = parse_key(str(track_doc.get("key") or "C minor"))

    avail = set(track_doc.get("stems_available") or [])
    is_4stem = len(avail.intersection(STEMS)) >= 3
    tatums = track_doc.get("tatums") or []
    segments = track_doc.get("segments") or []
    if not tatums and not segments:
        return {"track_id": tid, "is_4stem": is_4stem, "bar_quartets": [], "essential": {}}

    seg_by_id: Dict[str, Dict[str, Any]] = {}
    for s in segments:
        sid = s.get("id")
        if sid:
            seg_by_id[str(sid)] = s

    duration = float(track_doc.get("duration") or 0.0)
    if duration <= 0.0 and tatums:
        last_t = tatums[-1]
        duration = float(last_t.get("start", 0.0)) + float(last_t.get("duration", 0.0))
    total_bars = min(max_bars, max(4, int(duration / max(bar_sec, 0.5))))

    # Accumulate per-bar per-stem segment statistics
    bar_masks: Dict[str, List[int]] = {s: [0] * total_bars for s in STEMS}
    bar_chroma: Dict[str, List[List[float]]] = {s: [[0.0] * 12 for _ in range(total_bars)] for s in STEMS}
    bar_w_sum: Dict[str, List[float]] = {s: [0.0] * total_bars for s in STEMS}
    bar_loud_w: Dict[str, List[float]] = {s: [0.0] * total_bars for s in STEMS}
    bar_cent_w: Dict[str, List[float]] = {s: [0.0] * total_bars for s in STEMS}
    bar_seg_durs: Dict[str, List[List[float]]] = {s: [[] for _ in range(total_bars)] for s in STEMS}

    seen_seg_in_bar: Dict[Tuple[str, int, str], bool] = {}

    if tatums:
        for tat in tatums:
            t_start = float(tat.get("start", 0.0))
            b_idx = int(t_start / bar_sec)
            if b_idx < 0 or b_idx >= total_bars:
                continue
            rel = (t_start - b_idx * bar_sec) / bar_sec
            step16 = max(0, min(15, int(rel * 16.0)))
            stem_map = tat.get("stem_segments") or {}

            for s_name in (STEMS if is_4stem else ("other",)):
                for seg_id in (stem_map.get(s_name) or []):
                    seg = seg_by_id.get(str(seg_id))
                    if not seg or seg.get("is_silent"):
                        continue
                    l_max = float(seg.get("loudness_max", -60.0))
                    if l_max < -52.0:
                        continue
                    s_dur = float(seg.get("duration", 0.25))
                    target_stem = s_name if is_4stem else _classify_virtual_stem(seg, s_dur)
                    if target_stem not in bar_masks:
                        target_stem = "other"

                    # Trigger onset bit when segment begins near this tatum or shows strong attack
                    seg_start = float(seg.get("start", t_start))
                    l_start = float(seg.get("loudness_start", l_max))
                    if abs(seg_start - t_start) <= max(0.08, bar_sec / 14.0) or (l_max - l_start) >= 12.0:
                        bar_masks[target_stem][b_idx] |= (1 << step16)
                        if not is_4stem and target_stem == "bass" and (l_max - l_start) >= 14.0:
                            bar_masks["drums"][b_idx] |= (1 << step16)

                    # Accumulate chroma & energy once per (stem, bar, seg_id) or small slice weight
                    amp = 10.0 ** (max(-60.0, min(0.0, l_max)) / 20.0)
                    w = max(1e-4, amp * max(0.05, min(2.0, s_dur)))
                    dedup_key = (target_stem, b_idx, str(seg_id))
                    if dedup_key not in seen_seg_in_bar:
                        seen_seg_in_bar[dedup_key] = True
                        pitches = seg.get("pitches") or [0.0] * 12
                        if len(pitches) == 12:
                            cb = bar_chroma[target_stem][b_idx]
                            for k in range(12):
                                cb[k] += float(pitches[k]) * w
                        bar_w_sum[target_stem][b_idx] += w
                        bar_loud_w[target_stem][b_idx] += l_max * w
                        bar_cent_w[target_stem][b_idx] += float(seg.get("spectral_centroid") or 1200.0) * w
                        bar_seg_durs[target_stem][b_idx].append(s_dur)
    else:
        # Fallback if a JSON omits tatums and provides segments directly
        for seg in segments:
            if seg.get("is_silent"):
                continue
            l_max = float(seg.get("loudness_max", -60.0))
            if l_max < -52.0:
                continue
            s_start = float(seg.get("start", 0.0))
            b_idx = int(s_start / bar_sec)
            if b_idx < 0 or b_idx >= total_bars:
                continue
            rel = (s_start - b_idx * bar_sec) / bar_sec
            step16 = max(0, min(15, int(rel * 16.0)))
            s_dur = float(seg.get("duration", 0.25))
            raw_stem = str(seg.get("stem") or "other")
            target_stem = raw_stem if (is_4stem and raw_stem in bar_masks) else _classify_virtual_stem(seg, s_dur)
            bar_masks[target_stem][b_idx] |= (1 << step16)
            amp = 10.0 ** (max(-60.0, min(0.0, l_max)) / 20.0)
            w = max(1e-4, amp * max(0.05, min(2.0, s_dur)))
            pitches = seg.get("pitches") or [0.0] * 12
            if len(pitches) == 12:
                cb = bar_chroma[target_stem][b_idx]
                for k in range(12):
                    cb[k] += float(pitches[k]) * w
            bar_w_sum[target_stem][b_idx] += w
            bar_loud_w[target_stem][b_idx] += l_max * w
            bar_cent_w[target_stem][b_idx] += float(seg.get("spectral_centroid") or 1200.0) * w
            bar_seg_durs[target_stem][b_idx].append(s_dur)

    # First pass loudness normalization for section inference
    bar_energy_raw: List[float] = []
    for b_idx in range(total_bars):
        tot_e = 0.0
        for s_name in STEMS:
            ws = bar_w_sum[s_name][b_idx]
            if ws > 0.0:
                ldb = bar_loud_w[s_name][b_idx] / ws
                tot_e += 10.0 ** (max(-60.0, ldb) / 20.0)
        bar_energy_raw.append(tot_e)
    max_e = max(bar_energy_raw) if bar_energy_raw else 1.0
    if max_e <= 1e-6:
        max_e = 1.0

    # Assemble per-stem sequential bar lists
    bars_by_stem: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STEMS}
    bar_sections: List[str] = []
    for b_idx in range(total_bars):
        d_cnt = _popcount16(bar_masks["drums"][b_idx])
        sec = _infer_section_from_norm_pos(
            b_idx / max(1, total_bars - 1),
            d_cnt / 16.0,
            bar_energy_raw[b_idx] / max_e,
        )
        bar_sections.append(sec)

        for s_name in STEMS:
            ws = bar_w_sum[s_name][b_idx]
            rms_db = (bar_loud_w[s_name][b_idx] / ws) if ws > 1e-8 else -60.0
            peak_amp = min(1.0, 10.0 ** ((rms_db + 4.5) / 20.0)) if ws > 1e-8 else 0.0
            mask = bar_masks[s_name][b_idx]
            cnt = _popcount16(mask)
            # If tatums occurred densely without attack delta, ensure at least beat pulses when active
            if cnt == 0 and ws > 1e-8 and rms_db > MIN_RMS_DB[s_name]:
                mask = 0x1111
                cnt = 4
            chroma = bar_chroma[s_name][b_idx]
            if s_name == "drums":
                root, qual = -1, "perc"
                purity = score_cell_purity(s_name, rms_db, peak_amp, [0.0] * 12, cnt)
            else:
                root, qual, _ = classify_triad_quality(chroma)
                if sum(chroma) <= 1e-6:
                    root, qual = -1, "mono"
                purity = score_cell_purity(s_name, rms_db, peak_amp, chroma, cnt)

            is_fill = False
            if s_name == "drums" and (b_idx % 4) == 3 and cnt >= 6 and len(bars_by_stem["drums"]) >= 1:
                prev_cnt = _popcount16(bars_by_stem["drums"][-1]["mask"])
                is_fill = cnt >= prev_cnt + 2

            durs = bar_seg_durs[s_name][b_idx]
            mean_seg_dur = (sum(durs) / len(durs)) if durs else 0.35
            is_hook = bool(
                s_name == "vocals"
                and purity >= 0.42
                and 2 <= cnt <= 10
                and mean_seg_dur >= 0.25
                and sec in ("DROP", "GROOVE_A", "BUILDUP")
            )

            bars_by_stem[s_name].append({
                "stem": s_name,
                "bar_idx": b_idx,
                "bar_in_phrase": b_idx % 4,
                "section": sec,
                "purity": round(float(purity), 3),
                "mask": int(mask),
                "root": int(root),
                "qual": str(qual),
                "chroma": chroma,
                "hook": is_hook,
                "is_hook": is_hook,
                "fill": is_fill,
                "is_fill": is_fill,
                "schema": "AAAB" if is_fill else "ABAB",
                "session_root": key_pc,
                "target_quality": key_mode,
            })

    # For 4-bar phrase motif heuristics (ABAB vs AAAB vs ABCD)
    for s_name in STEMS:
        s_bars = bars_by_stem[s_name]
        for p_start in range(0, len(s_bars) - 3, 4):
            r0 = s_bars[p_start]["root"]
            r2 = s_bars[p_start + 2]["root"]
            m0 = s_bars[p_start]["mask"]
            m2 = s_bars[p_start + 2]["mask"]
            schema = "ABAB" if (r0 == r2 and m0 == m2) else ("AAAB" if s_bars[p_start + 3]["fill"] else "AABB")
            for k in range(4):
                s_bars[p_start + k]["schema"] = schema

    essential = select_essential_slices_for_song(bars_by_stem)
    for s_name in STEMS:
        es_list = essential["stems"][s_name]["essential_scores"]
        er_list = essential["stems"][s_name]["roles"]
        sh_list = essential["stems"][s_name]["reconstitution_shares"]
        fid_s = essential["stems"][s_name]["fidelity"]
        for b_idx, bar_cell in enumerate(bars_by_stem[s_name]):
            bar_cell["essential_score"] = es_list[b_idx] if b_idx < len(es_list) else 0.0
            bar_cell["essential_role"] = er_list[b_idx] if b_idx < len(er_list) else "A"
            bar_cell["reconstitution_share"] = sh_list[b_idx] if b_idx < len(sh_list) else 0.0
            bar_cell["song_fidelity"] = fid_s

    # Build bar_quartets (only include bars where at least 2 stems or overall mix is non-silent)
    bar_quartets: List[Dict[str, Any]] = []
    for b_idx in range(total_bars):
        cells_b = {s_name: bars_by_stem[s_name][b_idx] for s_name in STEMS}
        active_stems = sum(1 for s_name in STEMS if cells_b[s_name]["purity"] >= 0.14)
        mean_pur = sum(cells_b[s_name]["purity"] for s_name in STEMS) / 4.0
        if active_stems >= 2 or mean_pur >= 0.14:
            # Drop large chroma vector before storing quartet list to save RAM across 1,896 tracks
            compact_cells: Dict[str, Dict[str, Any]] = {}
            for s_name, c in cells_b.items():
                cc = {k: v for k, v in c.items() if k != "chroma"}
                compact_cells[s_name] = cc
            bar_quartets.append({
                "track_id": tid,
                "bar_idx": b_idx,
                "bar_pos": b_idx % 4,
                "section": bar_sections[b_idx],
                "session_root": key_pc,
                "target_quality": key_mode,
                "is_4stem": is_4stem,
                "mean_fidelity": float(essential["mean_fidelity"]),
                "cells": compact_cells,
            })

    return {
        "track_id": tid,
        "is_4stem": is_4stem,
        "total_bars": total_bars,
        "bar_quartets": bar_quartets,
        "essential": {
            "mean_fidelity": essential["mean_fidelity"],
            "role_transitions": essential["role_transitions"],
            "stem_fidelities": {s: essential["stems"][s]["fidelity"] for s in STEMS},
        },
    }
