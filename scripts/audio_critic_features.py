"""Shared helpers: build the browser critic's 218-D features for a bank phrase.

Used to weight/re-label phrase-combination training with the audio-distilled
song_critic.json, without decoding audio at training time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

STEMS = ("drums", "bass", "other", "vocals")
PITCHED = ("bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")


def load_bank(kits_dir: Path):
    manifest = json.loads((kits_dir / "kit_manifest.json").read_text())
    cells = json.loads((kits_dir / "cell_index.json").read_text()).get("phrases", {})
    chroma = json.loads((kits_dir / "cell_chroma.json").read_text()).get("phrases", {})
    phrases: Dict[str, Dict[str, Any]] = {}
    for kit in manifest.get("kits", []):
        for p in kit.get("phrases", []):
            phrases[p["id"]] = {
                "id": p["id"],
                "section": str(p.get("section") or "GROOVE_A").upper(),
                "energy": float(p.get("energy") or 0.5),
                "bpm": float(kit.get("bpm") or 120),
                "key": str(kit.get("key") or ""),
                "track_id": kit["track_id"],
                "cells": cells.get(p["id"], {}),
                "chroma": chroma.get(p["id"], {}),
            }
    return phrases


def pattern_vec(meta: Dict[str, Any], row: List[float]) -> np.ndarray:
    v = np.zeros(48, dtype=np.float32)
    mask = int((meta.get("o") or [0, 0, 0, 0])[0])
    for step in range(16):
        v[step] = (mask >> step) & 1
    n = int(v[:16].sum())
    v[16] = n / 16
    v[17] = sum((mask >> s) & 1 for s in (0, 4, 8, 12)) / 4
    v[18] = sum((mask >> s) & 1 for s in (1, 3, 5, 7, 9, 11, 13, 15)) / 8
    for q in range(4):
        v[19 + q] = sum((mask >> s) & 1 for s in range(q * 4, q * 4 + 4)) / 4
    r = (meta.get("r") or [-1, -1, -1, -1])[0]
    if isinstance(r, (int, float)) and r >= 0:
        v[23 + int(r) % 12] = 1
    if row:
        c = np.asarray(row, dtype=np.float32)
        if c.sum() > 1e-6:
            v[35:47] = c / c.sum()
    v[47] = 1.0 if n > 0 else 0.0
    return v


def phrase_features(pid: str, phrases: Dict[str, Dict[str, Any]]) -> np.ndarray:
    p = phrases[pid]
    v: List[float] = []
    masks: Dict[str, int] = {}
    chromas: Dict[str, np.ndarray] = {}
    for stem in STEMS:
        meta = p["cells"].get(stem, {})
        row = (p["chroma"].get(stem) or [[]])[0]
        pv = pattern_vec(meta, row)
        masks[stem] = sum(int((meta.get("o") or [0])[b]) for b in range(4))
        v.extend(pv.tolist())
        v.append(min(1.0, masks[stem] / 16))
        v.append(float(np.clip(p["energy"], 0, 1)))
        if stem in PITCHED:
            c = np.asarray(row, dtype=np.float32)
            chromas[stem] = c / c.sum() if c.sum() > 1e-8 else np.zeros(12)
    for stem in PITCHED:
        v.append(1.0 if chromas[stem].sum() > 0 else 0.0)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = PITCHED[i], PITCHED[j]
            na, nb = np.linalg.norm(chromas[a]), np.linalg.norm(chromas[b])
            v.append(float(np.dot(chromas[a], chromas[b]) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0)
            ma, mb = masks[a], masks[b]
            v.append(((ma & mb).bit_count() / (ma | mb).bit_count()) if (ma or mb) else 0.0)
    section = p["section"] if p["section"] in SECTIONS else "GROOVE_A"
    for s in SECTIONS:
        v.append(1.0 if s == section else 0.0)
    v.extend([float(np.clip(p["energy"], 0, 1)), 0.0, min(1.0, p["bpm"] / 200), 0.0])
    assert len(v) == 218, len(v)
    return np.asarray(v, dtype=np.float32)


def load_critic(path: Path):
    doc = json.loads(path.read_text())
    return doc


def critic_score(features: np.ndarray, model: Dict[str, Any]) -> float:
    x = (features - np.asarray(model["mean"], dtype=np.float32)) / np.asarray(model["std"], dtype=np.float32)
    a = x
    layers = model["layers"]
    for i, layer in enumerate(layers[:2]):
        w = np.asarray(layer["w"], dtype=np.float32)
        b = np.asarray(layer["b"], dtype=np.float32)
        a = np.tanh(a @ w.T + b)
    out = np.asarray(layers[2]["w"], dtype=np.float32)
    b = np.asarray(layers[2]["b"], dtype=np.float32)
    logit = float((a @ out.T + b)[0])
    return 1 / (1 + np.exp(-max(-30, min(30, logit))))
