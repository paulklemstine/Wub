#!/usr/bin/env python3
"""Train the neural 4-stem Bar/Cell Compatibility scorer (`cell_compat.json`).

Constructs positive (real co-occurring 4-stem bars + theory-clean cross-song
quartets) and hard-negative (3rd clashes, kick-bass 16th rubs, synth-vocal
masking, silent/muddy stems) quartets from `cell_index.json`, `kit_manifest.json`,
and optional 4-stem `processed/*.json` corpus files. Trains `CellCompatNet`
on GPU/CPU and exports lightweight JSON weights for `frontend/src/cell_compat.ts`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wub_core.essential_slices import (  # noqa: E402
    mine_processed_track_essential_bars,
    select_essential_slices_for_song,
)
from wub_core.motif_miner import extract_phrase_bar_features  # noqa: E402
from wub_core.theory_engine import (  # noqa: E402
    STEMS,
    clamp01,
    onset_bits_to_set,
    parse_key,
    score_kick_bass_pocket,
    score_synth_vocal_unmasking,
    solve_cell_pitch_shift,
    third_clash_penalty,
)

DOWNBEAT_MASK = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)
OFFBEAT_MASK = (1 << 2) | (1 << 6) | (1 << 10) | (1 << 14)
EVEN_MASK = 0b0101010101010101


def _popcount16(mask: int) -> int:
    return bin(int(mask) & 0xFFFF).count("1")


def build_cell_feature_vector(
    stem: str,
    purity: float,
    mask16: int,
    root: int,
    quality: str,
    semitones_shift: int,
    target_quality: str,
    is_hook: bool = False,
    is_fill: bool = False,
    schema: str = "AAAB",
) -> List[float]:
    """24-d descriptor matching `frontend/src/cell_compat.ts:buildCellFeatureVector`."""
    m = int(mask16) & 0xFFFF
    cnt = _popcount16(m)
    v = [0.0] * 24
    stem_idx = STEMS.index(stem) if stem in STEMS else 0
    v[stem_idx] = 1.0

    v[4] = float(purity)
    v[5] = cnt / 16.0
    v[6] = _popcount16(m & DOWNBEAT_MASK) / 4.0
    v[7] = _popcount16(m & OFFBEAT_MASK) / 4.0
    v[8] = (_popcount16(m & (~EVEN_MASK & 0xFFFF)) / cnt) if cnt > 0 else 0.0

    q = (quality or "mono").lower()
    if q == "minor":
        v[9] = 1.0
    elif q == "major":
        v[10] = 1.0
    elif q == "sus":
        v[11] = 1.0
    else:
        v[12] = 1.0

    v[13] = _popcount16(m & 0x000F) / 4.0
    v[14] = _popcount16((m >> 4) & 0x000F) / 4.0
    v[15] = _popcount16((m >> 8) & 0x000F) / 4.0
    v[16] = _popcount16((m >> 12) & 0x000F) / 4.0

    v[17] = min(1.0, abs(int(semitones_shift)) / 6.0)
    v[18] = float(third_clash_penalty(q, target_quality))
    v[19] = 1.0 if int(root) >= 0 else 0.0
    v[20] = 1.0 if is_hook else 0.0
    v[21] = 1.0 if is_fill else 0.0
    v[22] = 1.0 if schema in ("AAAB", "ABAB") else 0.0
    v[23] = 1.0 if schema in ("AABB", "ABAC") else 0.0
    return v


class CellCompatNet(nn.Module):
    """Joint vertical 4-stem mix compatibility + horizontal arrangement scorer."""

    def __init__(self, input_dim: int = 24, embed_dim: int = 32, arr_ctx_dim: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.arr_ctx_dim = arr_ctx_dim
        self.enc_fc1 = nn.Linear(input_dim, 48)
        self.enc_fc2 = nn.Linear(48, embed_dim)
        self.mix_fc1 = nn.Linear(embed_dim * 4, 64)
        self.mix_fc2 = nn.Linear(64, 1)
        self.arr_fc1 = nn.Linear(embed_dim * 2 + arr_ctx_dim, 48)
        self.arr_fc2 = nn.Linear(48, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.enc_fc1(x))
        e = self.enc_fc2(h)
        return F.normalize(e, dim=-1, eps=1e-8)

    def forward(self, quartet_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # quartet_x: [B, 4, 24]
        bsz = quartet_x.shape[0]
        flat = quartet_x.reshape(bsz * 4, self.input_dim)
        embs = self.encode(flat).reshape(bsz, 4, self.embed_dim)
        cat = embs.reshape(bsz, self.embed_dim * 4)
        logit = self.mix_fc2(F.gelu(self.mix_fc1(cat))).squeeze(-1)
        return logit, embs

    def forward_transition(
        self,
        prev_x: torch.Tensor,
        next_x: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        # prev_x: [B, 24], next_x: [B, 24], ctx: [B, 8]
        e_prev = self.encode(prev_x)
        e_next = self.encode(next_x)
        cat = torch.cat([e_prev, e_next, ctx], dim=-1)
        return self.arr_fc2(F.gelu(self.arr_fc1(cat))).squeeze(-1)

    def export_json_dict(
        self,
        val_auc: float,
        val_acc: float,
        sample_count: int,
        arr_val_auc: float = 0.0,
        arr_val_acc: float = 0.0,
        arr_sample_count: int = 0,
        essential_slices_mined: int = 0,
        tracks_mined: int = 0,
        mean_reconstitution_fidelity: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "version": 2,
            "format": "cell_compat_v2",
            "input_dim": int(self.input_dim),
            "embed_dim": int(self.embed_dim),
            "arr_ctx_dim": int(self.arr_ctx_dim),
            "samples": int(sample_count),
            "val_auc": round(float(val_auc), 4),
            "val_acc": round(float(val_acc), 4),
            "arr_samples": int(arr_sample_count),
            "arr_val_auc": round(float(arr_val_auc), 4),
            "arr_val_acc": round(float(arr_val_acc), 4),
            "essential_slices_mined": int(essential_slices_mined),
            "tracks_mined": int(tracks_mined),
            "mean_reconstitution_fidelity": round(float(mean_reconstitution_fidelity), 4),
            "enc_w1": [[round(float(x), 6) for x in row] for row in self.enc_fc1.weight.detach().cpu().tolist()],
            "enc_b1": [round(float(x), 6) for x in self.enc_fc1.bias.detach().cpu().tolist()],
            "enc_w2": [[round(float(x), 6) for x in row] for row in self.enc_fc2.weight.detach().cpu().tolist()],
            "enc_b2": [round(float(x), 6) for x in self.enc_fc2.bias.detach().cpu().tolist()],
            "mix_w1": [[round(float(x), 6) for x in row] for row in self.mix_fc1.weight.detach().cpu().tolist()],
            "mix_b1": [round(float(x), 6) for x in self.mix_fc1.bias.detach().cpu().tolist()],
            "mix_w2": [[round(float(x), 6) for x in row] for row in self.mix_fc2.weight.detach().cpu().tolist()],
            "mix_b2": [round(float(x), 6) for x in self.mix_fc2.bias.detach().cpu().tolist()],
            "arr_w1": [[round(float(x), 6) for x in row] for row in self.arr_fc1.weight.detach().cpu().tolist()],
            "arr_b1": [round(float(x), 6) for x in self.arr_fc1.bias.detach().cpu().tolist()],
            "arr_w2": [[round(float(x), 6) for x in row] for row in self.arr_fc2.weight.detach().cpu().tolist()],
            "arr_b2": [round(float(x), 6) for x in self.arr_fc2.bias.detach().cpu().tolist()],
        }


def _mine_proc_track_worker(pfile_str: str) -> Optional[Dict[str, Any]]:
    try:
        with open(pfile_str, "r") as f:
            doc = json.load(f)
        res = mine_processed_track_essential_bars(doc, max_bars=96)
        if not res.get("bar_quartets"):
            return None
        return res
    except Exception:
        return None


def _collect_corpus_units_and_transitions(
    cell_index_path: Path,
    manifest_path: Path,
    processed_dir: Optional[Path] = None,
    max_extra_proc: int = 3000,
    workers: int = 8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    with open(cell_index_path, "r") as f:
        ci = json.load(f)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    kit_by_phrase: Dict[str, Dict[str, Any]] = {}
    phrases_by_track: Dict[str, List[Dict[str, Any]]] = {}
    for kit in manifest.get("kits", []):
        kp, km = parse_key(kit.get("key", "C minor"))
        tid = kit["track_id"]
        ph_sorted = sorted(
            kit.get("phrases", []),
            key=lambda p: (int(p.get("start_bar", 0)), str(p.get("id", ""))),
        )
        phrases_by_track[tid] = ph_sorted
        for ph in ph_sorted:
            kit_by_phrase[ph["id"]] = {
                "key_pc": kp,
                "mode": km,
                "track_id": tid,
                "section": str(ph.get("section", "GROOVE_A")).upper(),
            }

    bar_quartets: List[Dict[str, Any]] = []
    transition_pairs: List[Dict[str, Any]] = []
    fidelities: List[float] = []
    seen_kit_tracks: set[str] = set()
    ess_slice_count = 0

    phrases_ci = ci.get("phrases") or {}

    # 1. Collect chronological bars and transitions from all shipped kits in `cell_index.json`
    for tid, ph_list in phrases_by_track.items():
        seen_kit_tracks.add(tid)
        track_seq: List[Dict[str, Any]] = []
        for ph in ph_list:
            pid = ph["id"]
            stems_map = phrases_ci.get(pid)
            if not stems_map:
                continue
            meta_kit = kit_by_phrase[pid]
            sec = meta_kit["section"]
            for b in range(4):
                cells_4: Dict[str, Dict[str, Any]] = {}
                for st in STEMS:
                    sm = stems_map[st]
                    es_list = sm.get("es") or [0.5, 0.5, 0.5, 0.5]
                    er_list = sm.get("er") or ["A", "A", "A", "B"]
                    es_val = float(es_list[b]) if b < len(es_list) else 0.5
                    er_val = str(er_list[b]) if b < len(er_list) else "A"
                    if es_val >= 0.50:
                        ess_slice_count += 1
                    cells_4[st] = {
                        "stem": st,
                        "purity": float(sm["p"][b]),
                        "mask": int(sm["o"][b]),
                        "root": int(sm["r"][b]),
                        "qual": str(sm["q"][b]),
                        "hook": bool(sm.get("hook", False)),
                        "fill": bool(sm.get("fill", False) and b == 3),
                        "schema": str(sm.get("s", "AAAB")),
                        "essential_score": es_val,
                        "essential_role": er_val,
                        "session_root": meta_kit["key_pc"],
                        "target_quality": meta_kit["mode"],
                    }
                bq = {
                    "track_id": tid,
                    "bar_pos": b,
                    "section": sec,
                    "session_root": meta_kit["key_pc"],
                    "target_quality": meta_kit["mode"],
                    "is_4stem": True,
                    "cells": cells_4,
                }
                bar_quartets.append(bq)
                track_seq.append(bq)

        # Consecutive phrase/bar transitions within this kit
        for i in range(len(track_seq) - 1):
            b0 = track_seq[i]
            b1 = track_seq[i + 1]
            is_bound = b0["section"] != b1["section"]
            for st in STEMS:
                c0 = b0["cells"][st]
                c1 = b1["cells"][st]
                if c0["purity"] < 0.14 and c1["purity"] < 0.14:
                    continue
                transition_pairs.append({
                    "track_id": tid,
                    "stem": st,
                    "prev": c0,
                    "next": c1,
                    "bar_pos": int(b1["bar_pos"]) % 4,
                    "is_section_boundary": is_bound,
                    "prev_es": float(c0["essential_score"]),
                    "next_es": float(c1["essential_score"]),
                    "session_root": b1["session_root"],
                    "target_quality": b1["target_quality"],
                    "is_4stem": True,
                })

    kit_fid = float(ci.get("mean_reconstitution_fidelity") or 0.0)
    if kit_fid > 0.0:
        fidelities.append(kit_fid)

    # 2. Mine up to all ~1,896 full-length tracks in `processed_dir` in parallel
    tracks_mined = len(seen_kit_tracks)
    if processed_dir and processed_dir.exists():
        proc_files = sorted(processed_dir.glob("*.json"))[:max_extra_proc]
        if proc_files:
            print(f"Mining essential slices & transitions across {len(proc_files)} processed tracks ({workers} workers)...")
            proc_quartets_added = 0
            proc_trans_added = 0
            proc_tracks_added = 0
            with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = [pool.submit(_mine_proc_track_worker, str(pf)) for pf in proc_files]
                for fut in as_completed(futures):
                    res = fut.result()
                    if not res:
                        continue
                    tid = res["track_id"]
                    bqs = res["bar_quartets"]
                    is_4s = bool(res.get("is_4stem", False))
                    ess = res.get("essential") or {}
                    mf = float(ess.get("mean_fidelity", 0.0))
                    if mf > 0.0:
                        fidelities.append(mf)
                    proc_tracks_added += 1

                    # Prioritize real 4-stem tracks & non-kit tracks into quartet mix pool,
                    # and contribute essential single-stem bars up to a clean cap per track
                    take_bqs = bqs if is_4s or (tid not in seen_kit_tracks) else []
                    for q in take_bqs:
                        for st in STEMS:
                            if q["cells"][st].get("essential_score", 0.0) >= 0.50:
                                ess_slice_count += 1
                        bar_quartets.append(q)
                        proc_quartets_added += 1

                    # Extract full-song chronological arrangement transitions from EVERY mined track
                    for i in range(len(bqs) - 1):
                        b0 = bqs[i]
                        b1 = bqs[i + 1]
                        if b1["bar_idx"] - b0["bar_idx"] > 2:
                            continue
                        is_bound = b0["section"] != b1["section"]
                        for st in STEMS:
                            c0 = b0["cells"][st]
                            c1 = b1["cells"][st]
                            if c0["purity"] < 0.14 and c1["purity"] < 0.14:
                                continue
                            # Include all high-value transitions + essential slices
                            if not is_4s and max(c0.get("essential_score", 0.0), c1.get("essential_score", 0.0)) < 0.42 and not is_bound:
                                continue
                            transition_pairs.append({
                                "track_id": tid,
                                "stem": st,
                                "prev": c0,
                                "next": c1,
                                "bar_pos": int(b1["bar_pos"]) % 4,
                                "is_section_boundary": is_bound,
                                "prev_es": float(c0.get("essential_score", 0.5)),
                                "next_es": float(c1.get("essential_score", 0.5)),
                                "session_root": b1["session_root"],
                                "target_quality": b1["target_quality"],
                                "is_4stem": is_4s,
                            })
                            proc_trans_added += 1

            tracks_mined = len(seen_kit_tracks) + proc_tracks_added
            print(
                f"  Mined {proc_tracks_added} processed tracks -> +{proc_quartets_added} bar quartets, "
                f"+{proc_trans_added} chronological transitions"
            )

    mean_fid = sum(fidelities) / len(fidelities) if fidelities else 0.0
    stats = {
        "tracks_mined": tracks_mined,
        "essential_slices_mined": ess_slice_count,
        "mean_reconstitution_fidelity": mean_fid,
    }
    return bar_quartets, transition_pairs, stats


def _score_and_featurize_quartet(
    cells_4: Dict[str, Dict[str, Any]],
    session_root: int,
    target_rel_root: int,
    target_quality: str,
    allow_relative: bool = True,
    override_shift: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    vec_4 = np.zeros((4, 24), dtype=np.float32)
    shifts: Dict[str, int] = {}
    penalties: Dict[str, float] = {}
    purities: List[float] = []

    for s_idx, st in enumerate(STEMS):
        c = cells_4[st]
        if override_shift is not None and st != "drums":
            sh = int(override_shift)
            pen = max(0.35, float(third_clash_penalty(c["qual"], target_quality)))
        else:
            sh, pen = solve_cell_pitch_shift(
                st, c["root"], c["qual"], session_root, target_rel_root, target_quality, 6, allow_relative
            )
        shifts[st] = sh
        penalties[st] = pen
        purities.append(float(c["purity"]))
        vec_4[s_idx] = build_cell_feature_vector(
            stem=st,
            purity=c["purity"],
            mask16=c["mask"],
            root=c["root"],
            quality=c["qual"],
            semitones_shift=sh,
            target_quality=target_quality,
            is_hook=c["hook"],
            is_fill=c["fill"],
            schema=c["schema"],
        )

    d_steps = list(onset_bits_to_set(cells_4["drums"]["mask"]))
    b_steps = list(onset_bits_to_set(cells_4["bass"]["mask"]))
    o_steps = list(onset_bits_to_set(cells_4["other"]["mask"]))
    v_steps = list(onset_bits_to_set(cells_4["vocals"]["mask"]))

    pocket = score_kick_bass_pocket(d_steps, b_steps)
    unmask = score_synth_vocal_unmasking(o_steps, v_steps, len(o_steps) / 16.0, len(v_steps) / 16.0)
    mean_pur = sum(purities) / 4.0
    harm = clamp01(1.0 - max(penalties["bass"], penalties["other"], penalties["vocals"] * 0.7))
    theory = clamp01(0.35 * mean_pur + 0.25 * pocket + 0.20 * unmask + 0.20 * harm)
    return vec_4, theory


def _featurize_single_cell(
    cell: Dict[str, Any],
    session_root: int,
    target_rel_root: int,
    target_quality: str,
) -> Tuple[np.ndarray, float, int]:
    sh, pen = solve_cell_pitch_shift(
        cell["stem"],
        int(cell["root"]),
        str(cell["qual"]),
        int(session_root),
        int(target_rel_root),
        str(target_quality),
        6,
        True,
    )
    v = build_cell_feature_vector(
        stem=str(cell["stem"]),
        purity=float(cell["purity"]),
        mask16=int(cell["mask"]),
        root=int(cell["root"]),
        quality=str(cell["qual"]),
        semitones_shift=sh,
        target_quality=str(target_quality),
        is_hook=bool(cell.get("hook", False)),
        is_fill=bool(cell.get("fill", False)),
        schema=str(cell.get("schema", "AAAB")),
    )
    return np.asarray(v, dtype=np.float32), float(pen), int(sh)


def build_training_dataset(
    bar_quartets: List[Dict[str, Any]],
    seed: int = 42,
    samples_target: int = 56000,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STEMS}
    ess_pools: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STEMS}
    for bq in bar_quartets:
        for st in STEMS:
            c = bq["cells"][st]
            if c["purity"] >= 0.22:
                pools[st].append(c)
                if float(c.get("essential_score", 0.5)) >= 0.48:
                    ess_pools[st].append(c)
    for st in STEMS:
        if not ess_pools[st]:
            ess_pools[st] = pools[st]

    xs: List[np.ndarray] = []
    ys: List[float] = []

    # 1. Ground-truth same-song co-occurring bars (prioritizing clean 4-stem and essential slices)
    gt_items: List[Tuple[np.ndarray, float]] = []
    for bq in bar_quartets:
        active_cnt = sum(1 for st in STEMS if bq["cells"][st]["purity"] >= 0.22)
        if active_cnt < 3:
            continue
        vec, theory = _score_and_featurize_quartet(
            bq["cells"], bq["session_root"], 0, bq["target_quality"]
        )
        if theory < 0.58:
            continue
        mean_es = sum(float(bq["cells"][st].get("essential_score", 0.5)) for st in STEMS) / 4.0
        ess_boost = 0.08 * max(0.0, mean_es - 0.35)
        is_4s = bool(bq.get("is_4stem", True))
        label = max(0.68, min(0.99, (0.48 if is_4s else 0.42) + 0.54 * theory + ess_boost))
        gt_items.append((vec, label))

    max_gt = max(100, int(samples_target * 0.30))
    if len(gt_items) > max_gt:
        rng.shuffle(gt_items)
        gt_items = gt_items[:max_gt]
    for vec, label in gt_items:
        xs.append(vec)
        ys.append(label)

    # 2. Balanced cross-song essential slice recompositions (~50% pos) + multi-clash hard negatives (~50% neg)
    while len(xs) < samples_target:
        tgt_q = rng.choice(("minor", "major"))
        tgt_root = rng.randrange(12)
        rel_root = rng.choice((0, 3, 5, 7, 8, 10))

        mode = rng.random()
        if mode < 0.28:
            # Draw from essential pools when building clean cross-song recomposed quartets
            q_cells = {st: dict(rng.choice(ess_pools[st])) for st in STEMS}
            q_cells["other"] = dict(q_cells["other"])
            q_cells["other"]["qual"] = tgt_q if rng.random() < 0.72 else "mono"
            q_cells["other"]["purity"] = max(0.45, q_cells["other"]["purity"])
            q_cells["drums"] = dict(q_cells["drums"])
            q_cells["drums"]["purity"] = max(0.50, q_cells["drums"]["purity"])
            q_cells["bass"] = dict(q_cells["bass"])
            q_cells["bass"]["purity"] = max(0.45, q_cells["bass"]["purity"])
            q_cells["bass"]["mask"] = q_cells["drums"]["mask"] | (1 << 0)
            vec, theory = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            if theory >= 0.62:
                xs.append(vec)
                ys.append(min(0.98, 0.46 + 0.57 * theory))
        elif mode < 0.48:
            # Hard negative A: major vs minor 3rd harmonic clash + large transpose
            q_cells = {st: dict(rng.choice(pools[st])) for st in STEMS}
            clash_q = "major" if tgt_q == "minor" else "minor"
            for st_clash in ("bass", "other", "vocals"):
                q_cells[st_clash] = dict(q_cells[st_clash])
                q_cells[st_clash]["qual"] = clash_q
                q_cells[st_clash]["root"] = (tgt_root + rel_root + 5) % 12
                q_cells[st_clash]["schema"] = "ABCD"
            vec, _ = _score_and_featurize_quartet(
                q_cells,
                tgt_root,
                rel_root,
                tgt_q,
                allow_relative=False,
                override_shift=rng.choice((5, 6)),
            )
            xs.append(vec)
            ys.append(0.04)
        elif mode < 0.66:
            # Hard negative B: kick-bass 16th-note rub collision
            q_cells = {st: dict(rng.choice(pools[st])) for st in STEMS}
            q_cells["drums"] = dict(q_cells["drums"])
            q_cells["bass"] = dict(q_cells["bass"])
            q_cells["drums"]["mask"] = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)
            q_cells["bass"]["mask"] = (1 << 1) | (1 << 5) | (1 << 9) | (1 << 13)
            q_cells["bass"]["purity"] = rng.uniform(0.32, 0.55)
            q_cells["bass"]["schema"] = "ABCD"
            vec, _ = _score_and_featurize_quartet(
                q_cells,
                tgt_root,
                rel_root,
                tgt_q,
                allow_relative=False,
                override_shift=rng.choice((4, 5, 6)),
            )
            xs.append(vec)
            ys.append(0.06)
        elif mode < 0.84:
            # Hard negative C: dense synth masking busy vocal across all 16th steps + lower purity
            q_cells = {st: dict(rng.choice(pools[st])) for st in STEMS}
            q_cells["other"] = dict(q_cells["other"])
            q_cells["vocals"] = dict(q_cells["vocals"])
            q_cells["other"]["mask"] = 0xFFFF
            q_cells["vocals"]["mask"] = 0xFFFF
            q_cells["other"]["purity"] = rng.uniform(0.28, 0.42)
            q_cells["vocals"]["purity"] = rng.uniform(0.25, 0.40)
            q_cells["other"]["schema"] = "ABCD"
            q_cells["vocals"]["schema"] = "ABCD"
            vec, _ = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            xs.append(vec)
            ys.append(0.05)
        else:
            # Hard negative D: combined multi-clash (rub + 0xFFFF masking + clash_q + large shift + ABCD + low purity)
            q_cells = {st: dict(rng.choice(pools[st])) for st in STEMS}
            clash_q = "major" if tgt_q == "minor" else "minor"
            q_cells["drums"] = dict(q_cells["drums"])
            q_cells["bass"] = dict(q_cells["bass"])
            q_cells["other"] = dict(q_cells["other"])
            q_cells["vocals"] = dict(q_cells["vocals"])
            q_cells["drums"]["mask"] = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)
            q_cells["drums"]["schema"] = "ABCD"
            q_cells["bass"]["mask"] = (1 << 1) | (1 << 5) | (1 << 9) | (1 << 13)
            q_cells["bass"]["purity"] = rng.uniform(0.05, 0.42)
            q_cells["bass"]["qual"] = clash_q
            q_cells["bass"]["schema"] = "ABCD"
            q_cells["other"]["mask"] = 0xFFFF
            q_cells["other"]["purity"] = rng.uniform(0.05, 0.38)
            q_cells["other"]["qual"] = clash_q
            q_cells["other"]["schema"] = "ABCD"
            q_cells["vocals"]["mask"] = 0xFFFF
            q_cells["vocals"]["purity"] = rng.uniform(0.05, 0.35)
            q_cells["vocals"]["qual"] = clash_q
            q_cells["vocals"]["schema"] = "ABCD"
            vec, _ = _score_and_featurize_quartet(
                q_cells,
                tgt_root,
                rel_root,
                tgt_q,
                allow_relative=False,
                override_shift=6,
            )
            xs.append(vec)
            ys.append(0.02)

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    return x_arr, y_arr


def _build_arr_ctx(
    bar_pos: int,
    is_section_boundary: bool,
    prev_es: float,
    next_es: float,
    prev_purity: float,
    next_purity: float,
) -> np.ndarray:
    ctx = np.zeros(8, dtype=np.float32)
    ctx[int(bar_pos) & 3] = 1.0
    ctx[4] = 1.0 if is_section_boundary else 0.0
    ctx[5] = clamp01(float(prev_es))
    ctx[6] = clamp01(float(next_es))
    delta = max(-1.0, min(1.0, float(next_purity) - float(prev_purity)))
    ctx[7] = 0.5 + 0.5 * delta
    return ctx


def build_transition_dataset(
    transition_pairs: List[Dict[str, Any]],
    seed: int = 101,
    samples_target: int = 64000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build horizontal bar/phrase/section arrangement transition dataset."""
    rng = random.Random(seed)
    ess_by_stem: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STEMS}
    all_by_stem: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STEMS}
    for tp in transition_pairs:
        st = tp["stem"]
        c_next = tp["next"]
        if c_next["purity"] >= 0.22:
            all_by_stem[st].append(c_next)
            if tp["next_es"] >= 0.50:
                ess_by_stem[st].append(c_next)
    for st in STEMS:
        if not ess_by_stem[st]:
            ess_by_stem[st] = all_by_stem[st]

    prev_xs: List[np.ndarray] = []
    next_xs: List[np.ndarray] = []
    ctxs: List[np.ndarray] = []
    ys: List[float] = []

    # 1. Real chronological corpus transitions (up to 42% of budget)
    real_shuffled = list(transition_pairs)
    rng.shuffle(real_shuffled)
    max_real = max(100, int(samples_target * 0.42))
    for tp in real_shuffled[:max_real]:
        s_root = int(tp["session_root"])
        t_qual = str(tp["target_quality"])
        px, p_pen, _ = _featurize_single_cell(tp["prev"], s_root, 0, t_qual)
        nx, n_pen, _ = _featurize_single_cell(tp["next"], s_root, 0, t_qual)
        ess_bonus = 0.12 * max(tp["prev_es"], tp["next_es"])
        pos_bonus = 0.05 if (tp["bar_pos"] == 3 and tp["next"].get("fill")) or tp["next"].get("hook") else 0.0
        harm_ok = 1.0 - 0.35 * max(p_pen, n_pen)
        base = 0.80 if tp.get("is_4stem", True) else 0.74
        label = max(0.68, min(0.98, base * harm_ok + ess_bonus + pos_bonus))
        prev_xs.append(px)
        next_xs.append(nx)
        ctxs.append(_build_arr_ctx(
            tp["bar_pos"],
            tp["is_section_boundary"],
            tp["prev_es"],
            tp["next_es"],
            tp["prev"]["purity"],
            tp["next"]["purity"],
        ))
        ys.append(label)

    # 2. Theory-clean cross-song essential continuations vs hard arrangement negatives
    while len(ys) < samples_target:
        st = rng.choice(STEMS)
        if not ess_by_stem[st] or not all_by_stem[st]:
            continue
        tgt_root = rng.randrange(12)
        tgt_qual = rng.choice(("minor", "major"))
        b_pos = rng.randrange(4)
        is_bound = rng.random() < 0.22

        roll = rng.random()
        if roll < 0.24:
            # Cross-song essential continuation: compatible pocket + smooth harmony
            c0 = dict(rng.choice(ess_by_stem[st]))
            c1 = dict(rng.choice(ess_by_stem[st]))
            if st != "drums":
                c0["root"] = tgt_root
                c0["qual"] = tgt_qual if rng.random() < 0.75 else "mono"
                # Smooth diatonic / fourth-fifth step
                c1["root"] = (tgt_root + rng.choice((0, 0, 2, 5, 7, 10))) % 12
                c1["qual"] = tgt_qual if rng.random() < 0.7 else "mono"
            cnt0 = _popcount16(c0["mask"])
            cnt1 = _popcount16(c1["mask"])
            if abs(cnt0 - cnt1) > 5 and not is_bound:
                c1["mask"] = c0["mask"] ^ (1 << rng.choice((2, 6, 10, 14)))
            es0 = max(0.55, float(c0.get("essential_score", 0.65)))
            es1 = max(0.55, float(c1.get("essential_score", 0.65)))
            px, _, _ = _featurize_single_cell(c0, tgt_root, 0, tgt_qual)
            nx, n_pen, _ = _featurize_single_cell(c1, tgt_root, 0, tgt_qual)
            if n_pen <= 0.35:
                prev_xs.append(px)
                next_xs.append(nx)
                ctxs.append(_build_arr_ctx(b_pos, is_bound, es0, es1, c0["purity"], c1["purity"]))
                ys.append(0.82)
        elif roll < 0.56:
            # Hard negative 1: clashing harmonic jump / mismatched non-essential cross-song cut
            c0 = dict(rng.choice(all_by_stem[st]))
            c1 = dict(rng.choice(all_by_stem[st]))
            clash_q = "major" if tgt_qual == "minor" else "minor"
            if st != "drums":
                c0["root"] = tgt_root
                c0["qual"] = tgt_qual
                c1["root"] = (tgt_root + rng.choice((1, 6, 11))) % 12
                c1["qual"] = clash_q
            else:
                # Wild rhythmic discontinuity mid-phrase without section boundary
                c0["mask"] = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)
                c1["mask"] = 0xAAAA
                is_bound = False
            px, _, _ = _featurize_single_cell(c0, tgt_root, 0, tgt_qual)
            nx, _, _ = _featurize_single_cell(c1, tgt_root, 0, tgt_qual)
            prev_xs.append(px)
            next_xs.append(nx)
            ctxs.append(_build_arr_ctx(b_pos, is_bound, 0.55, 0.18, c0["purity"], c1["purity"]))
            ys.append(0.08)
        elif roll < 0.78:
            # Hard negative 2: mid-phrase dead dropout / non-essential bleed interrupting groove
            c0 = dict(rng.choice(ess_by_stem[st]))
            c1 = dict(c0)
            c1["purity"] = 0.04
            c1["mask"] = 0
            c1["root"] = -1
            c1["qual"] = "mono"
            c1["hook"] = False
            c1["fill"] = False
            px, _, _ = _featurize_single_cell(c0, tgt_root, 0, tgt_qual)
            nx, _, _ = _featurize_single_cell(c1, tgt_root, 0, tgt_qual)
            prev_xs.append(px)
            next_xs.append(nx)
            ctxs.append(_build_arr_ctx(rng.choice((1, 2)), False, 0.72, 0.05, c0["purity"], 0.04))
            ys.append(0.05)
        else:
            # Hard negative 3: misplaced turnaround fill on bar 0 downbeat OR vocal hook chaos
            c0 = dict(rng.choice(all_by_stem[st]))
            c1 = dict(rng.choice(all_by_stem[st]))
            if st == "drums":
                c1["fill"] = True
                c1["mask"] = 0xFF00
                bad_pos = 0
            else:
                bad_pos = rng.choice((1, 2))
                c1["qual"] = "major" if tgt_qual == "minor" else "minor"
                c1["root"] = (tgt_root + 6) % 12
                c1["purity"] = max(0.12, c0["purity"] * 0.45)
            px, _, _ = _featurize_single_cell(c0, tgt_root, 0, tgt_qual)
            nx, _, _ = _featurize_single_cell(c1, tgt_root, 0, tgt_qual)
            prev_xs.append(px)
            next_xs.append(nx)
            ctxs.append(_build_arr_ctx(bad_pos, False, 0.60, 0.20, c0["purity"], c1["purity"]))
            ys.append(0.11)

    return (
        np.stack(prev_xs, axis=0).astype(np.float32),
        np.stack(next_xs, axis=0).astype(np.float32),
        np.stack(ctxs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true >= 0.5]
    neg = y_score[y_true < 0.5]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    auc = (ranks[y_true >= 0.5].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint Wub CellCompatNet + EssentialArrangementNet")
    parser.add_argument("--cell-index", type=Path, default=Path("frontend/public/data/kits/cell_index.json"))
    parser.add_argument("--manifest", type=Path, default=Path("frontend/public/data/kits/kit_manifest.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/cell_compat.json"))
    parser.add_argument("--sidecars", type=Path, default=Path("frontend/public/data/kits/sidecars.json"))
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--samples", type=int, default=64000)
    parser.add_argument("--arr-samples", type=int, default=64000)
    parser.add_argument("--max-proc-tracks", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu"
    )
    print(f"Training CellCompatNet (vertical mix + horizontal essential arrangement) on device={device}")

    quartets, trans_pairs, stats = _collect_corpus_units_and_transitions(
        args.cell_index,
        args.manifest,
        args.processed_dir,
        max_extra_proc=args.max_proc_tracks,
        workers=args.workers,
    )
    print(
        f"Collected {len(quartets)} bar quartets, {len(trans_pairs)} real transitions across "
        f"{stats['tracks_mined']} tracks ({stats['essential_slices_mined']} essential slices, "
        f"mean_fid={stats['mean_reconstitution_fidelity']:.4f})"
    )

    x_all, y_all = build_training_dataset(quartets, samples_target=args.samples)
    ap_all, an_all, ac_all, ay_all = build_transition_dataset(trans_pairs, samples_target=args.arr_samples)

    rng_np = np.random.default_rng(42)
    idx_m = np.arange(len(x_all))
    rng_np.shuffle(idx_m)
    x_all, y_all = x_all[idx_m], y_all[idx_m]

    idx_a = np.arange(len(ay_all))
    rng_np.shuffle(idx_a)
    ap_all, an_all, ac_all, ay_all = ap_all[idx_a], an_all[idx_a], ac_all[idx_a], ay_all[idx_a]

    n_val_m = max(512, int(0.15 * len(x_all)))
    x_train, y_train = torch.from_numpy(x_all[n_val_m:]), torch.from_numpy(y_all[n_val_m:])
    x_val, y_val = torch.from_numpy(x_all[:n_val_m]).to(device), y_all[:n_val_m]

    n_val_a = max(512, int(0.15 * len(ay_all)))
    ap_train, an_train, ac_train, ay_train = (
        torch.from_numpy(ap_all[n_val_a:]),
        torch.from_numpy(an_all[n_val_a:]),
        torch.from_numpy(ac_all[n_val_a:]),
        torch.from_numpy(ay_all[n_val_a:]),
    )
    ap_val = torch.from_numpy(ap_all[:n_val_a]).to(device)
    an_val = torch.from_numpy(an_all[:n_val_a]).to(device)
    ac_val = torch.from_numpy(ac_all[:n_val_a]).to(device)
    ay_val = ay_all[:n_val_a]

    mix_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    arr_loader = DataLoader(
        TensorDataset(ap_train, an_train, ac_train, ay_train), batch_size=args.batch_size, shuffle=True
    )

    model = CellCompatNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = 0.0
    best_auc = 0.0
    best_acc = 0.0
    best_arr_auc = 0.0
    best_arr_acc = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        arr_iter = iter(arr_loader)

        for bx, by in mix_loader:
            bx = bx.to(device)
            by = by.to(device)
            try:
                bp, bn, bc, bay = next(arr_iter)
            except StopIteration:
                arr_iter = iter(arr_loader)
                bp, bn, bc, bay = next(arr_iter)
            bp, bn, bc, bay = bp.to(device), bn.to(device), bc.to(device), bay.to(device)

            opt.zero_grad()
            logit, embs = model(bx)
            bce_mix = F.binary_cross_entropy_with_logits(logit, by)
            db_cos = (embs[:, 0] * embs[:, 1]).sum(dim=-1)
            ov_cos = (embs[:, 2] * embs[:, 3]).sum(dim=-1)
            pair_target = 2.0 * by - 1.0
            pair_loss = 0.15 * (F.mse_loss(db_cos, pair_target) + F.mse_loss(ov_cos, pair_target))

            arr_logit = model.forward_transition(bp, bn, bc)
            bce_arr = F.binary_cross_entropy_with_logits(arr_logit, bay)

            loss = bce_mix + pair_loss + bce_arr
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            n_batches += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            arr_val_logits = model.forward_transition(ap_val, an_val, ac_val)
            arr_val_probs = torch.sigmoid(arr_val_logits).cpu().numpy()

        val_auc = _roc_auc(y_val, val_probs)
        val_acc = float(((val_probs >= 0.5) == (y_val >= 0.5)).mean())
        arr_val_auc = _roc_auc(ay_val, arr_val_probs)
        arr_val_acc = float(((arr_val_probs >= 0.5) == (ay_val >= 0.5)).mean())
        comb_score = math.sqrt(max(1e-6, val_auc) * max(1e-6, arr_val_auc))
        if comb_score >= best_score:
            best_score = comb_score
            best_auc = val_auc
            best_acc = val_acc
            best_arr_auc = arr_val_auc
            best_arr_acc = arr_val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs:
            print(
                f"  epoch {ep:2d}/{args.epochs} loss={tot_loss/max(1, n_batches):.4f} "
                f"mix_auc={val_auc:.4f} mix_acc={val_acc:.4f} "
                f"arr_auc={arr_val_auc:.4f} arr_acc={arr_val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    out_doc = model.export_json_dict(
        val_auc=best_auc,
        val_acc=best_acc,
        sample_count=len(x_all),
        arr_val_auc=best_arr_auc,
        arr_val_acc=best_arr_acc,
        arr_sample_count=len(ay_all),
        essential_slices_mined=stats["essential_slices_mined"],
        tracks_mined=stats["tracks_mined"],
        mean_reconstitution_fidelity=stats["mean_reconstitution_fidelity"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_doc, f, separators=(",", ":"))

    if args.sidecars.exists():
        with open(args.sidecars, "r") as f:
            sc = json.load(f)
        sc["cell_compat"] = True
        with open(args.sidecars, "w") as f:
            json.dump(sc, f, indent=2)
            f.write("\n")

    size_kb = args.out.stat().st_size / 1024.0
    print(
        f"Exported {args.out} ({size_kb:.1f} KB, val_auc={best_auc:.4f}, val_acc={best_acc:.4f}, "
        f"arr_val_auc={best_arr_auc:.4f}, arr_val_acc={best_arr_acc:.4f}) in {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
