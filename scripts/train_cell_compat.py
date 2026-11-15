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
    """Lightweight 4-stem bar cell compatibility encoder + mix quality head."""

    def __init__(self, input_dim: int = 24, embed_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.enc_fc1 = nn.Linear(input_dim, 48)
        self.enc_fc2 = nn.Linear(48, embed_dim)
        self.mix_fc1 = nn.Linear(embed_dim * 4, 64)
        self.mix_fc2 = nn.Linear(64, 1)

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

    def export_json_dict(self, val_auc: float, val_acc: float, sample_count: int) -> Dict[str, Any]:
        return {
            "version": 1,
            "format": "cell_compat_v1",
            "input_dim": int(self.input_dim),
            "embed_dim": int(self.embed_dim),
            "samples": int(sample_count),
            "val_auc": round(float(val_auc), 4),
            "val_acc": round(float(val_acc), 4),
            "enc_w1": [[round(float(x), 6) for x in row] for row in self.enc_fc1.weight.detach().cpu().tolist()],
            "enc_b1": [round(float(x), 6) for x in self.enc_fc1.bias.detach().cpu().tolist()],
            "enc_w2": [[round(float(x), 6) for x in row] for row in self.enc_fc2.weight.detach().cpu().tolist()],
            "enc_b2": [round(float(x), 6) for x in self.enc_fc2.bias.detach().cpu().tolist()],
            "mix_w1": [[round(float(x), 6) for x in row] for row in self.mix_fc1.weight.detach().cpu().tolist()],
            "mix_b1": [round(float(x), 6) for x in self.mix_fc1.bias.detach().cpu().tolist()],
            "mix_w2": [[round(float(x), 6) for x in row] for row in self.mix_fc2.weight.detach().cpu().tolist()],
            "mix_b2": [round(float(x), 6) for x in self.mix_fc2.bias.detach().cpu().tolist()],
        }


def _collect_bar_units(
    cell_index_path: Path,
    manifest_path: Path,
    processed_dir: Optional[Path] = None,
    max_extra_proc: int = 400,
) -> List[Dict[str, Any]]:
    with open(cell_index_path, "r") as f:
        ci = json.load(f)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    kit_by_phrase: Dict[str, Dict[str, Any]] = {}
    for kit in manifest.get("kits", []):
        kp, km = parse_key(kit.get("key", "C minor"))
        for ph in kit.get("phrases", []):
            kit_by_phrase[ph["id"]] = {"key_pc": kp, "mode": km, "track_id": kit["track_id"]}

    bar_quartets: List[Dict[str, Any]] = []
    for pid, stems_map in (ci.get("phrases") or {}).items():
        meta_kit = kit_by_phrase.get(pid, {"key_pc": 0, "mode": "minor", "track_id": pid})
        for b in range(4):
            bar_entry: Dict[str, Any] = {
                "track_id": meta_kit["track_id"],
                "session_root": meta_kit["key_pc"],
                "target_quality": meta_kit["mode"],
                "cells": {},
            }
            for st in STEMS:
                sm = stems_map[st]
                bar_entry["cells"][st] = {
                    "stem": st,
                    "purity": float(sm["p"][b]),
                    "mask": int(sm["o"][b]),
                    "root": int(sm["r"][b]),
                    "qual": str(sm["q"][b]),
                    "hook": bool(sm.get("hook", False)),
                    "fill": bool(sm.get("fill", False)),
                    "schema": str(sm.get("s", "AAAB")),
                }
            bar_quartets.append(bar_entry)

    # Also mine additional 4-stem processed corpus tracks when available
    if processed_dir and processed_dir.exists():
        seen_tracks = {q["track_id"] for q in bar_quartets}
        added = 0
        for pfile in sorted(processed_dir.glob("*.json")):
            if added >= max_extra_proc:
                break
            tid = pfile.stem
            if tid in seen_tracks:
                continue
            try:
                with open(pfile, "r") as f:
                    doc = json.load(f)
            except Exception:
                continue
            if len(doc.get("stems_available") or []) < 4:
                continue
            n_bars = len(doc.get("tatums") or []) // 16
            if n_bars < 8:
                continue
            kp, km = parse_key(doc.get("key", "C minor"))
            # Sample up to 3 4-bar phrases evenly across the track
            sample_starts = [max(0, min(n_bars - 4, int(frac * n_bars))) for frac in (0.15, 0.45, 0.72)]
            for sb in sample_starts:
                feats = {
                    st: extract_phrase_bar_features(doc, sb, st, -18.0, 0.6)
                    for st in STEMS
                }
                if any(v is None for v in feats.values()):
                    continue
                for b in range(4):
                    q_cells = {}
                    for st in STEMS:
                        fv = feats[st]
                        assert fv is not None
                        q_cells[st] = {
                            "stem": st,
                            "purity": float(fv["purity"][b]),
                            "mask": int(fv["bar_onsets"][b]),
                            "root": int(fv["bar_roots"][b]),
                            "qual": str(fv["bar_qualities"][b]),
                            "hook": False,
                            "fill": bool(fv.get("is_fill_bar", False)),
                            "schema": str(fv.get("schema", "AAAB")),
                        }
                    bar_quartets.append(
                        {
                            "track_id": tid,
                            "session_root": kp,
                            "target_quality": km,
                            "cells": q_cells,
                        }
                    )
            added += 1
        if added > 0:
            print(f"Merged {added} extra 4-stem processed tracks from {processed_dir}")

    return bar_quartets


def _score_and_featurize_quartet(
    cells_4: Dict[str, Dict[str, Any]],
    session_root: int,
    target_rel_root: int,
    target_quality: str,
) -> Tuple[np.ndarray, float]:
    vec_4 = np.zeros((4, 24), dtype=np.float32)
    shifts: Dict[str, int] = {}
    penalties: Dict[str, float] = {}
    purities: List[float] = []

    for s_idx, st in enumerate(STEMS):
        c = cells_4[st]
        sh, pen = solve_cell_pitch_shift(
            st, c["root"], c["qual"], session_root, target_rel_root, target_quality, 6, True
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


def build_training_dataset(
    bar_quartets: List[Dict[str, Any]],
    seed: int = 42,
    samples_target: int = 36000,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    pools: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STEMS}
    for bq in bar_quartets:
        for st in STEMS:
            if bq["cells"][st]["purity"] >= 0.25:
                pools[st].append(bq["cells"][st])

    xs: List[np.ndarray] = []
    ys: List[float] = []

    # 1. Ground-truth same-song co-occurring bars (positive when non-silent)
    for bq in bar_quartets:
        active_cnt = sum(1 for st in STEMS if bq["cells"][st]["purity"] >= 0.22)
        if active_cnt < 3:
            continue
        vec, theory = _score_and_featurize_quartet(
            bq["cells"], bq["session_root"], 0, bq["target_quality"]
        )
        label = max(0.70, min(1.0, 0.50 + 0.55 * theory))
        xs.append(vec)
        ys.append(label)

    # 2. Cross-song positives + hard negatives for sharp contrast
    while len(xs) < samples_target:
        # Cross-song quartet with controlled target chord quality
        tgt_q = rng.choice(("minor", "major"))
        tgt_root = rng.randrange(12)
        rel_root = rng.choice((0, 3, 5, 7, 8, 10))
        q_cells = {st: dict(rng.choice(pools[st])) for st in STEMS}

        mode = rng.random()
        if mode < 0.38:
            # Synthesize a theory-clean cross-song quartet positive
            q_cells["other"] = dict(q_cells["other"])
            q_cells["other"]["qual"] = tgt_q if rng.random() < 0.7 else "mono"
            q_cells["other"]["purity"] = max(0.45, q_cells["other"]["purity"])
            q_cells["drums"] = dict(q_cells["drums"])
            q_cells["drums"]["purity"] = max(0.50, q_cells["drums"]["purity"])
            q_cells["bass"] = dict(q_cells["bass"])
            q_cells["bass"]["purity"] = max(0.45, q_cells["bass"]["purity"])
            q_cells["bass"]["mask"] = q_cells["drums"]["mask"] | (1 << 0)
            vec, theory = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            if theory >= 0.62:
                xs.append(vec)
                ys.append(min(0.98, 0.45 + 0.58 * theory))
        elif mode < 0.60:
            # Hard negative A: major vs minor 3rd harmonic clash + large transpose
            clash_q = "major" if tgt_q == "minor" else "minor"
            q_cells["other"] = dict(q_cells["other"])
            q_cells["other"]["qual"] = clash_q
            q_cells["other"]["root"] = (tgt_root + rel_root + 5) % 12
            vec, _ = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            xs.append(vec)
            ys.append(0.08)
        elif mode < 0.78:
            # Hard negative B: kick-bass 16th-note rub collision
            q_cells["drums"] = dict(q_cells["drums"])
            q_cells["bass"] = dict(q_cells["bass"])
            q_cells["drums"]["mask"] = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)
            q_cells["bass"]["mask"] = (1 << 1) | (1 << 5) | (1 << 9) | (1 << 13)
            vec, _ = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            xs.append(vec)
            ys.append(0.14)
        elif mode < 0.90:
            # Hard negative C: dense synth masking busy vocal across all 16th steps
            q_cells["other"] = dict(q_cells["other"])
            q_cells["vocals"] = dict(q_cells["vocals"])
            q_cells["other"]["mask"] = 0xFFFF
            q_cells["vocals"]["mask"] = 0xFFFF
            vec, _ = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            xs.append(vec)
            ys.append(0.12)
        else:
            # Hard negative D: dead / near-silent stems
            bad_stem = rng.choice(STEMS[:3])
            q_cells[bad_stem] = dict(q_cells[bad_stem])
            q_cells[bad_stem]["purity"] = 0.03
            q_cells[bad_stem]["mask"] = 0
            vec, _ = _score_and_featurize_quartet(q_cells, tgt_root, rel_root, tgt_q)
            xs.append(vec)
            ys.append(0.05)

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    return x_arr, y_arr


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
    parser = argparse.ArgumentParser(description="Train Wub CellCompatNet")
    parser.add_argument("--cell-index", type=Path, default=Path("frontend/public/data/kits/cell_index.json"))
    parser.add_argument("--manifest", type=Path, default=Path("frontend/public/data/kits/kit_manifest.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/cell_compat.json"))
    parser.add_argument("--sidecars", type=Path, default=Path("frontend/public/data/kits/sidecars.json"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu"
    )
    print(f"Training CellCompatNet on device={device}")

    quartets = _collect_bar_units(args.cell_index, args.manifest, args.processed_dir)
    print(f"Collected {len(quartets)} base bar quartets")
    x_all, y_all = build_training_dataset(quartets, samples_target=args.samples)

    idx = np.arange(len(x_all))
    np.random.default_rng(42).shuffle(idx)
    x_all, y_all = x_all[idx], y_all[idx]

    n_val = max(512, int(0.15 * len(x_all)))
    x_train, y_train = torch.from_numpy(x_all[n_val:]), torch.from_numpy(y_all[n_val:])
    x_val, y_val = torch.from_numpy(x_all[:n_val]).to(device), y_all[:n_val]

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    model = CellCompatNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc = 0.0
    best_acc = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad()
            logit, embs = model(bx)
            bce = F.binary_cross_entropy_with_logits(logit, by)
            # Pairwise cosine regularizer pulling positive drums<->bass & other<->vocals closer
            db_cos = (embs[:, 0] * embs[:, 1]).sum(dim=-1)
            ov_cos = (embs[:, 2] * embs[:, 3]).sum(dim=-1)
            pair_target = 2.0 * by - 1.0
            pair_loss = 0.15 * (F.mse_loss(db_cos, pair_target) + F.mse_loss(ov_cos, pair_target))
            loss = bce + pair_loss
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * bx.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = _roc_auc(y_val, val_probs)
        val_acc = float(((val_probs >= 0.5) == (y_val >= 0.5)).mean())
        if val_auc >= best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 6 == 0 or ep == args.epochs:
            print(f"  epoch {ep:2d}/{args.epochs} loss={tot_loss/len(x_train):.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    out_doc = model.export_json_dict(best_auc, best_acc, len(x_all))
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
    print(f"Exported {args.out} ({size_kb:.1f} KB, val_auc={best_auc:.4f}, val_acc={best_acc:.4f}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
