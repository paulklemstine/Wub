#!/usr/bin/env python3
"""DJ-grade stem transition scorer for Wub KG-505.

Rates a candidate same-stem cross-song phrase transition (old -> new at a
block boundary) on six DJ-ear axes, each 0..1, plus a weighted total and a
deterministic rank.  This is an offline analysis / training-feature tool: it
never touches playback and deliberately models Stretch behaviour analytically,
without rendering audio, so thousands of pairs score in seconds.

Axes (all higher = smoother splice)
------------------------------------
* ``energy``    phrase-energy delta plus distance to the planned arc target,
                plus stem-RMS trajectory.  A DJ hears the floor lift or dip
                when the arc target is missed, and the per-stem level step
                when RMS jumps.
* ``harmonic``  after transposing both phrases to the session key, chroma
                cosine plus bass-root interval consonance.  Intervals 0/5/7
                are free, 1/6/11 clash (mirrors ``harmonyPenalty`` in
                ``frontend/src/arc_planner.ts``).  Drums are unpitched: 1.0.
* ``groove``    Stretch rate cost |rate-1|, onset-density match, and
                downbeat alignment.  Without audio/ears, density and
                downbeat fall back to manifest proxies (crest match); with
                ears or decoded audio they use real onsets.
* ``spectral``  centroid / flatness / MFCC delta (how audible is the
                splice?).  Without ears/audio, falls back to the manifest
                64-d mix embedding cosine + chroma L1 + crest delta.
* ``loudness``  stem-RMS delta plus peak (dB) delta.
* ``presence``  both phrases actually have audio.  Smooth ramp above
                ``MIN_RMS_DB`` per stem (same floors as ``stem_remix.ts``);
                below-floor pairs are capped, not deleted.

Stretch simulation (no render)
------------------------------
``blockSec = 960 / session_bpm`` (4 bars x 4 beats).  Each phrase gets
``rate = duration_sec / blockSec`` and ``shift = session_root - phrase_root``
wrapped to -6..+6 semitones, exactly the ``stem_remix.ts`` gate
``[RATE_MIN=0.60, RATE_MAX=1.65]``.  This assumes Signalsmith Stretch costs
are monotonic in |rate-1| and |shift| and that formant/window artifacts at
extreme rates are captured by the penalties below.  It is a cheap proxy, not
a perceptual render; validate against decoded audio (see ``--validate-n``).

NOTE on shift sign: ``stem_remix.ts`` ``keyShift()`` returns
``phrase_root - session_root``; this scorer uses the musically correct
``session_root - phrase_root`` so rotated chroma/roots match what the ear
hears after transposition.  Pairwise interval magnitudes are preserved
under a global sign flip only if applied consistently, so use this
scorer's harmonic axis as the ear-truth; do not mix its shifts into the
browser engine without flipping the sign.

Ears inputs (optional, graceful degrade)
----------------------------------------
``--ears-dsp``   JSON mapping phrase_id -> per-phrase DSP features
                 (centroid_hz, flatness, mfcc_mean, onset_density,
                 rms_db, peak, chroma).  Several container shapes are
                 accepted: bare dict, {"phrases": {...}}, {"phrases": [...]},
                 {"data": {...}}.  Missing file / unknown shape -> ignored.
``--ears-embed`` JSON mapping phrase_id -> embedding vector (any dim).
                 Used for the spectral axis when present.
When neither is present, manifest proxies are used and Librosa (if
available) computes real metrics for up to ``--validate-n`` transitions.

Usage
-----
    python3 scripts/score_transitions.py \\
      --manifest frontend/public/data/kits/kit_manifest.json \\
      --kits-dir frontend/public/data/kits \\
      --session-bpm 129 --session-key "F minor" \\
      --stem drums --n 50 --seed 505 \\
      --out /tmp/opencode/transitions_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VERSION = 1
STEMS = ("drums", "bass", "other", "vocals")
MIN_RMS_DB = {"drums": -28.0, "bass": -28.0, "other": -34.0, "vocals": -42.0}
RATE_MIN, RATE_MAX = 0.60, 1.65
# Planned 104-bar arc (frontend/src/arc_planner.ts): (start, bars, energy)
ARC_SEGMENTS = ((0, 16, 0.30), (16, 32, 0.72), (48, 16, 0.82), (64, 32, 0.95), (96, 8, 0.20))
DEFAULT_WEIGHTS = {
    "groove": 0.28,
    "harmonic": 0.24,
    "energy": 0.18,
    "spectral": 0.15,
    "loudness": 0.10,
    "presence": 0.05,
}
BASS_INTERVAL_PENALTY = [0.00, 0.55, 0.12, 0.10, 0.10, 0.00, 0.85, 0.00, 0.15, 0.15, 0.20, 0.50]
VOCAL_INTERVAL_PENALTY = [0.00, 0.70, 0.16, 0.14, 0.12, 0.00, 0.95, 0.00, 0.18, 0.18, 0.24, 0.65]
PITCH = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
         "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def parse_key(name: str | None):
    if not name:
        return None
    import re
    m = re.match(r"^([A-Ga-g])([#b♯♭]?)\s*(major|minor|maj|min|M|m)$", str(name).strip())
    if not m:
        # also accept "Fm" / "F#m" shorthand
        m2 = re.match(r"^([A-Ga-g])([#b])?m$", str(name).strip())
        if not m2:
            return None
        m = m2
        acc = "#" if m.group(2) == "#" else ("B" if m.group(2) == "b" else "")
        root = PITCH.get(f"{m.group(1).upper()}{acc}")
        return (root, True) if root is not None else None
    acc = "#" if m.group(2) == "♯" else ("B" if m.group(2) == "♭" else m.group(2).upper())
    root = PITCH.get(f"{m.group(1).upper()}{acc}")
    if root is None:
        return None
    mode = m.group(3).lower()
    minor = mode.startswith("min") or mode == "m"
    return root, minor


def camelot(name: str | None):
    p = parse_key(name)
    if p is None:
        return None
    root, minor = p
    return (CAMELOT_MINOR if minor else CAMELOT_MAJOR)[root], ("A" if minor else "B")


def key_shift(session: str, phrase: str) -> tuple[float, bool]:
    """Musically correct shift to bring phrase to session: session - phrase, -6..+6."""
    a = parse_key(session)
    b = parse_key(phrase)
    if a is None or b is None:
        return 0.0, True
    s = (a[0] - b[0] + 12) % 12
    return (float(s - 12 if s > 6 else s)), False


def rotate_chroma(chroma, shift: float):
    if not chroma or len(chroma) != 12:
        return None
    k = int(round(shift))
    out = [0.0] * 12
    for i in range(12):
        out[(i + k + 12) % 12] = float(chroma[i] or 0.0)
    return out


def cosine(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    dot = na = nb = 0.0
    for i in range(n):
        x, y = float(a[i] or 0.0), float(b[i] or 0.0)
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return None
    return dot / math.sqrt(na * nb)


def target_at(bar: int):
    b = ((int(bar) % 104) + 104) % 104
    for start, bars, energy in ARC_SEGMENTS:
        if start <= b < start + bars:
            phase = {0: "INTRO", 16: "GROOVE_A", 48: "BUILDUP", 64: "DROP", 96: "BRIDGE"}[start]
            return {"bar": b, "energy": energy, "phase": phase}
    return {"bar": b, "energy": 0.72, "phase": "GROOVE_A"}


def peak_db(peak: float) -> float:
    return 20.0 * math.log10(max(float(peak or 0.0), 1e-6))


# ---------------- ears loading (flexible, graceful) ----------------
def _normalize_ears(path: Path):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        return {}, f"unreadable: {e}"
    for key in ("phrases", "data", "dsp", "embeddings", "items"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], (dict, list)):
            raw = raw[key]
            break
    if isinstance(raw, dict):
        # limit to entries that look like features or vectors
        return raw, None
    if isinstance(raw, list):
        out = {}
        for row in raw:
            if isinstance(row, dict) and ("phrase_id" in row or "id" in row):
                pid = str(row.get("phrase_id") or row.get("id"))
                out[pid] = row
        return out, None
    return {}, "unexpected top-level type"


def load_ears(dsp_path, embed_path):
    dsp, embed, notes = {}, {}, []
    if dsp_path:
        p = Path(str(dsp_path))
        if p.exists():
            dsp, err = _normalize_ears(p)
            notes.append(f"ears-dsp: {'loaded '+str(len(dsp))+' entries' if not err else 'ignored ('+str(err)+')'}")
        else:
            notes.append(f"ears-dsp: not found at {p}, using fallback")
    if embed_path:
        p = Path(str(embed_path))
        if p.exists():
            embed, err = _normalize_ears(p)
            notes.append(f"ears-embed: {'loaded '+str(len(embed))+' entries' if not err else 'ignored ('+str(err)+')'}")
        else:
            notes.append(f"ears-embed: not found at {p}, using fallback")
    return dsp, embed, notes


def ears_get(ears: dict, pid: str):
    return ears.get(pid) or ears.get(pid.split(":", 1)[0]) or None


def ears_float(row, *keys, default=None):
    if not isinstance(row, dict):
        return default
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return default


def ears_vec(row, *keys):
    if row is None:
        return None
    if isinstance(row, list):
        return [float(x) for x in row]
    if isinstance(row, dict):
        for k in keys:
            if isinstance(row.get(k), list):
                try:
                    return [float(x) for x in row[k]]
                except (TypeError, ValueError):
                    continue
    return None


# ---------------- librosa validation (graceful) ----------------
def audio_features(path: Path):
    """Decode one opus stem file. Returns dict or None on failure."""
    try:
        import librosa  # available per spec
        import numpy as np  # noqa
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        if len(y) < sr // 4:
            return None
        cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        flat = float(librosa.feature.spectral_flatness(y=y).mean())
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        onset = librosa.onset.onset_detect(y=y, sr=sr, units="time", hop_length=512)
        dur = len(y) / max(sr, 1)
        density = len(onset) / max(dur, 1e-6)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        peak_idx = int(0.15 * sr / 512)
        downbeat = float(env[: max(1, peak_idx)].max() / (env.max() + 1e-12)) if len(env) else 0.0
        rms = float(np.sqrt(np.mean(y**2) + 1e-12))
        return {"centroid": cent, "flatness": flat, "mfcc": [float(x) for x in mfcc],
                "onset_density": float(density), "downbeat_ratio": float(downbeat),
                "rms_db": float(20 * math.log10(rms + 1e-12)), "peak": float(np.abs(y).max())}
    except Exception:  # noqa: BLE001 - decode best-effort
        return None


# ---------------- axis scorers ----------------
def score_energy(pf, pt, rms_f, rms_t, target):
    e_f = float(pf.get("energy", 0.5) or 0.5)
    e_t = float(pt.get("energy", 0.5) or 0.5)
    de = abs(e_t - e_f)
    dt = abs(e_t - float(target))
    dr = abs(float(rms_t) - float(rms_f))
    s = 1.0 - clamp01(0.45 * (de / 0.5) + 0.30 * (dt / 0.5) + 0.25 * (dr / 12.0))
    return clamp01(s), {"energy_from": round(e_f, 4), "energy_to": round(e_t, 4),
                        "delta_energy": round(de, 4), "delta_target": round(dt, 4),
                        "delta_rms_db": round(dr, 3)}


def score_harmonic(stem, kit_f, kit_t, pf, pt, session_key):
    if stem == "drums":
        return 1.0, {"method": "unpitched_drums"}
    sh_f, miss_f = key_shift(session_key, str(kit_f.get("key") or ""))
    sh_t, miss_t = key_shift(session_key, str(kit_t.get("key") or ""))
    cf = rotate_chroma(pf.get("chroma"), sh_f)
    ct = rotate_chroma(pt.get("chroma"), sh_t)
    cos = cosine(cf, ct)
    chroma_comp = None if cos is None else clamp01(cos)
    # amplify saturated chroma cosine (typ. 0.83-0.99) into 0..1
    chroma_score = None if cos is None else clamp01(1.0 - (1.0 - cos) / 0.35)
    rf = pf.get("bass_root")
    rt = pt.get("bass_root")
    interval = None
    interval_score = None
    penalty = None
    if rf is not None and rt is not None:
        rfn = (int(rf) + int(round(sh_f))) % 12
        rtn = (int(rt) + int(round(sh_t))) % 12
        interval = abs(rtn - rfn) % 12
        table = VOCAL_INTERVAL_PENALTY if stem == "vocals" else BASS_INTERVAL_PENALTY
        penalty = table[interval]
        interval_score = 1.0 - penalty
    excess = max(0.0, abs(sh_t) - 3.0)
    shift_pen = min(0.16, excess * 0.04)
    # mode mismatch: parallel major/minor clash, relative major allowed
    ca = camelot(session_key)
    cb = camelot(str(kit_t.get("key") or ""))
    mode_mismatch = False
    relative = False
    if ca and cb:
        relative = (ca[0] == cb[0] and ca[1] != cb[1])
        mode_mismatch = (ca[1] != cb[1] and not relative)
    # combine
    if stem == "other":
        base = chroma_score if chroma_score is not None else 0.5
        s = base
    elif interval_score is not None and chroma_score is not None:
        s = 0.70 * interval_score + 0.30 * chroma_score
    elif interval_score is not None:
        s = interval_score
    elif chroma_score is not None:
        s = chroma_score
    else:
        s = 0.5
    s = s - shift_pen - (0.10 if mode_mismatch else 0.0)
    return clamp01(s), {
        "shift_from": round(sh_f, 2), "shift_to": round(sh_t, 2),
        "cosine": None if cos is None else round(float(cos), 4),
        "chroma_score": None if chroma_score is None else round(float(chroma_score), 4),
        "interval": interval, "interval_penalty": penalty,
        "key_unknown": bool(miss_f or miss_t),
        "mode_mismatch": bool(mode_mismatch), "relative_major_minor": bool(relative),
        "to_key": str(kit_t.get("key") or ""), "from_key": str(kit_f.get("key") or "")}


def score_groove(stem, rate_f, rate_t, rms_f, peak_f, rms_t, peak_t, ears_f, ears_t, aud_f, aud_t):
    dev = (abs(rate_f - 1.0) + abs(rate_t - 1.0)) / 2.0
    stretch = 1.0 - clamp01(dev / 0.30)
    od_f = ears_float(ears_f, "onset_density", "onset_per_sec") if ears_f else None
    od_t = ears_float(ears_t, "onset_density", "onset_per_sec") if ears_t else None
    if od_f is None and aud_f:
        od_f = aud_f.get("onset_density")
    if od_t is None and aud_t:
        od_t = aud_t.get("onset_density")
    if od_f is not None and od_t is not None:
        denom = max(0.5, (float(od_f) + float(od_t)) / 2.0)
        dens = 1.0 - clamp01(abs(float(od_f) - float(od_t)) / denom)
        method = "onset_density"
    else:
        crest_f = peak_db(peak_f) - float(rms_f)
        crest_t = peak_db(peak_t) - float(rms_t)
        dens = 1.0 - clamp01(abs(crest_f - crest_t) / 12.0)
        method = "crest_proxy"
    # downbeat alignment (audio/ears only; proxy = neutral 0.5-blend)
    db_f = ears_float(ears_f, "downbeat_ratio") if ears_f else None
    db_t = ears_float(ears_t, "downbeat_ratio") if ears_t else None
    if db_f is None and aud_f is not None:
        db_f = aud_f.get("downbeat_ratio")
    if db_t is None and aud_t is not None:
        db_t = aud_t.get("downbeat_ratio")
    if db_f is not None and db_t is not None:
        down = 1.0 - clamp01(abs(float(db_f) - float(db_t)) / 0.40)
    else:
        # planar: sections far from downbeat-groove phases lose a little
        down = 0.65
        method += "+downbeat_neutral"
    s = 0.45 * stretch + 0.35 * dens + 0.20 * down
    # "audible extreme" = |rate-1| > 0.30 (smearing, not just gate overflow);
    # gate overflow (<0.60 or >1.65) is the engine hard-exclude subset.
    extreme = bool(abs(rate_f - 1.0) > 0.30 or abs(rate_t - 1.0) > 0.30)
    gate_out = bool(rate_f < RATE_MIN or rate_f > RATE_MAX
                    or rate_t < RATE_MIN or rate_t > RATE_MAX)
    return clamp01(s), {"rate_from": round(rate_f, 4), "rate_to": round(rate_t, 4),
                        "stretch_score": round(stretch, 4), "density_score": round(dens, 4),
                        "downbeat_score": round(down, 4), "density_method": method,
                        "onset_density_from": None if od_f is None else round(float(od_f), 3),
                        "onset_density_to": None if od_t is None else round(float(od_t), 3),
                        "extreme_stretch": extreme, "outside_gate": gate_out}


def manifest_spectral_proxy(pf, pt, embed_f, embed_t):
    """Manifest-only spectral proxy (embedding + chroma), also used as validation baseline."""
    ef = embed_f if embed_f is not None else pf.get("embedding")
    et = embed_t if embed_t is not None else pt.get("embedding")
    emb_cos = cosine(ef, et)
    emb_score = None if emb_cos is None else clamp01((float(emb_cos) + 1.0) / 2.0)
    ch_f, ch_t = pf.get("chroma"), pt.get("chroma")
    if ch_f and ch_t and len(ch_f) == 12 and len(ch_t) == 12:
        l1 = sum(abs(float(a) - float(b)) for a, b in zip(ch_f, ch_t))
        chroma_s = 1.0 - clamp01(l1 / 0.60)
    else:
        l1, chroma_s = None, None
    if emb_score is not None and chroma_s is not None:
        s = 0.65 * emb_score + 0.35 * chroma_s
        method = "embedding_chroma_proxy"
    elif emb_score is not None:
        s, method = emb_score, "embedding_only"
    elif chroma_s is not None:
        s, method = chroma_s, "chroma_only"
    else:
        s, method = 0.5, "neutral"
    return clamp01(s), method, emb_cos, l1


def score_spectral(pf, pt, stem, ears_f, ears_t, embed_f, embed_t, aud_f, aud_t):
    proxy_s, proxy_method, emb_cos, l1 = manifest_spectral_proxy(pf, pt, embed_f, embed_t)
    # ears centroid/flatness/mfcc path
    cf = ears_float(ears_f, "centroid_hz", "centroid") if ears_f else None
    ct = ears_float(ears_t, "centroid_hz", "centroid") if ears_t else None
    ff = ears_float(ears_f, "flatness") if ears_f else None
    ft = ears_float(ears_t, "flatness") if ears_t else None
    mf = ears_vec(ears_f, "mfcc_mean", "mfcc")
    mt = ears_vec(ears_t, "mfcc_mean", "mfcc")
    if cf is None and aud_f:
        cf = aud_f.get("centroid")
    if ct is None and aud_t:
        ct = aud_t.get("centroid")
    if ff is None and aud_f:
        ff = aud_f.get("flatness")
    if ft is None and aud_t:
        ft = aud_t.get("flatness")
    if mf is None and aud_f:
        mf = aud_f.get("mfcc")
    if mt is None and aud_t:
        mt = aud_t.get("mfcc")
    if cf is not None and ct is not None:
        rel = abs(float(cf) - float(ct)) / max(500.0, (float(cf) + float(ct)) / 2.0)
        cent_s = 1.0 - clamp01(rel / 1.0)
    else:
        cent_s = None
    if ff is not None and ft is not None:
        flat_s = 1.0 - clamp01(abs(float(ff) - float(ft)) / 0.30)
    else:
        flat_s = None
    if mf and mt and len(mf) == len(mt):
        import math as _m
        d = _m.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(mf, mt)))
        mfcc_s = 1.0 - clamp01(d / 20.0)
    else:
        mfcc_s = None
    have_real = sum(x is not None for x in (cent_s, flat_s, mfcc_s)) >= 2
    if have_real:
        parts = [x for x in (cent_s, flat_s, mfcc_s) if x is not None]
        s = sum(parts) / len(parts)
        return clamp01(s), {"method": "ears_or_audio_dsp",
                            "manifest_proxy": round(float(proxy_s), 4),
                            "manifest_proxy_method": proxy_method,
                            "centroid_from": None if cf is None else round(float(cf), 1),
                            "centroid_to": None if ct is None else round(float(ct), 1),
                            "flatness_from": ff, "flatness_to": ft, "mfcc_dist": None}
    return clamp01(proxy_s), {"method": proxy_method, "manifest_proxy": round(float(proxy_s), 4),
                        "emb_cosine": None if emb_cos is None else round(float(emb_cos), 4),
                        "chroma_l1": None if l1 is None else round(float(l1), 4)}


def score_loudness(rms_f, rms_t, peak_f, peak_t):
    dr = abs(float(rms_t) - float(rms_f))
    dp = abs(peak_db(peak_t) - peak_db(peak_f))
    s = 1.0 - clamp01(0.6 * (dr / 12.0) + 0.4 * (dp / 12.0))
    return clamp01(s), {"delta_rms_db": round(dr, 3), "delta_peak_db": round(dp, 3),
                        "peak_db_from": round(peak_db(peak_f), 2), "peak_db_to": round(peak_db(peak_t), 2)}


def score_presence(stem, rms_f, rms_t):
    floor = MIN_RMS_DB[stem]
    rf = clamp01((float(rms_f) - floor + 8.0) / 16.0)
    rt = clamp01((float(rms_t) - floor + 8.0) / 16.0)
    s = min(rf, rt)
    below = bool(float(rms_f) < floor or float(rms_t) < floor)
    return clamp01(s), {"rms_from": round(float(rms_f), 2), "rms_to": round(float(rms_t), 2),
                        "floor": floor, "below_floor": below}


def quantile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    pos = (len(s) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def stats(vals):
    if not vals:
        return {"n": 0}
    import statistics
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
            "median": round(float(statistics.median(vals)), 4),
            "p10": round(float(quantile(vals, 0.1)), 4),
            "p90": round(float(quantile(vals, 0.9)), 4),
            "min": round(float(min(vals)), 4), "max": round(float(max(vals)), 4)}


def sample_pairs(pool, n, seed):
    """Deterministic same-stem cross-song sampling (pool sorted by phrase id)."""
    rng = random.Random(seed)
    out = []
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        a = pool[rng.randrange(len(pool))]
        b = pool[rng.randrange(len(pool))]
        if a["phrase_id"] == b["phrase_id"]:
            continue
        if a["track_id"] == b["track_id"]:
            continue
        # hash-deterministic tiebreak so re-runs match even if pool order shifts
        h = hashlib.md5(f"{seed}:{len(out)}:{a['phrase_id']}:{b['phrase_id']}".encode()).hexdigest()
        out.append((a, b, h))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json")
    ap.add_argument("--kits-dir", default="frontend/public/data/kits")
    ap.add_argument("--session-bpm", type=float, default=129.0)
    ap.add_argument("--session-key", default="F minor")
    ap.add_argument("--stem", choices=list(STEMS), default="drums")
    ap.add_argument("--session-bar", type=int, default=32, help="bar on the 104-bar arc for the energy target (default 32 = mid-GROOVE 0.72)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--ears-dsp", default=None)
    ap.add_argument("--ears-embed", default=None)
    ap.add_argument("--validate-n", type=int, default=20, help="decode up to N transitions for real spectral validation (0 disables)")
    ap.add_argument("--no-audio", action="store_true", help="never decode opus, manifest proxies only")
    ap.add_argument("--out", default="/tmp/opencode/transitions_smoke.json")
    args = ap.parse_args(argv)

    manifest_path = Path(str(args.manifest))
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    kits_dir = Path(str(args.kits_dir))
    if not kits_dir.is_absolute():
        kits_dir = REPO_ROOT / kits_dir
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stem = args.stem
    block_sec = 960.0 / float(args.session_bpm)
    tgt = target_at(args.session_bar)

    pools = {s: [] for s in STEMS}
    phrase_index = {}
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            pid = str(phrase.get("id"))
            phrase_index[pid] = (kit, phrase)
            for s in STEMS:
                asset = (phrase.get("stems") or {}).get(s)
                if asset and asset.get("file"):
                    pools[s].append({"phrase_id": pid, "track_id": str(kit.get("track_id")),
                                     "kit": kit, "phrase": phrase, "asset": asset})
    for s in STEMS:
        pools[s].sort(key=lambda e: e["phrase_id"])
    pool = pools[stem]
    if len(pool) < 2:
        print(f"[score-transitions] pool too small for stem {stem}: {len(pool)}", file=sys.stderr)
        return 1

    ears_dsp, ears_embed, ear_notes = load_ears(args.ears_dsp, args.ears_embed)
    pairs = sample_pairs(pool, int(args.n), int(args.seed))

    # ---- optional audio validation (bounded, cached) ----
    validate_n = 0 if args.no_audio else max(0, int(args.validate_n))
    audio_cache: dict[str, dict | None] = {}

    def get_audio(track_file: str, cache_key: str):
        if cache_key in audio_cache:
            return audio_cache[cache_key]
        p = kits_dir / str(track_file)
        feats = audio_features(p) if p.exists() else None
        audio_cache[cache_key] = feats
        return feats

    # choose validation subset deterministically: first validate_n sampled pairs
    validate_idx = set(range(min(validate_n, len(pairs))))

    transitions = []
    for i, (a, b, h) in enumerate(pairs):
        kit_f, pf = a["kit"], a["phrase"]
        kit_t, pt = b["kit"], b["phrase"]
        rate_f = float(pf.get("duration_sec", block_sec)) / block_sec
        rate_t = float(pt.get("duration_sec", block_sec)) / block_sec
        rms_f = float(a["asset"].get("rms_db", -20.0) or -20.0)
        rms_t = float(b["asset"].get("rms_db", -20.0) or -20.0)
        peak_f = float(a["asset"].get("peak", 0.0) or 0.0)
        peak_t = float(b["asset"].get("peak", 0.0) or 0.0)
        ears_f = ears_get(ears_dsp, a["phrase_id"]) if ears_dsp else None
        ears_t = ears_get(ears_dsp, b["phrase_id"]) if ears_dsp else None
        emb_f = ears_vec(ears_get(ears_embed, a["phrase_id"]) if ears_embed else None, "embedding", "vec")
        emb_t = ears_vec(ears_get(ears_embed, b["phrase_id"]) if ears_embed else None, "embedding", "vec")
        aud_f = aud_t = None
        if i in validate_idx:
            aud_f = get_audio(str(a["asset"]["file"]), a["phrase_id"])
            aud_t = get_audio(str(b["asset"]["file"]), b["phrase_id"])
        e_s, e_d = score_energy(pf, pt, rms_f, rms_t, tgt["energy"])
        h_s, h_d = score_harmonic(stem, kit_f, kit_t, pf, pt, str(args.session_key))
        g_s, g_d = score_groove(stem, rate_f, rate_t, rms_f, peak_f, rms_t, peak_t, ears_f, ears_t, aud_f, aud_t)
        s_s, s_d = score_spectral(pf, pt, stem, ears_f, ears_t, emb_f, emb_t, aud_f, aud_t)
        l_s, l_d = score_loudness(rms_f, rms_t, peak_f, peak_t)
        p_s, p_d = score_presence(stem, rms_f, rms_t)
        axes = {"energy": round(e_s, 4), "harmonic": round(h_s, 4), "groove": round(g_s, 4),
                "spectral": round(s_s, 4), "loudness": round(l_s, 4), "presence": round(p_s, 4)}
        weights = dict(DEFAULT_WEIGHTS)
        if stem == "drums":
            weights.pop("harmonic", None)
            wsum = sum(weights.values())
            weights = {k: v / wsum for k, v in weights.items()}
        total = sum(weights[k] * axes[k] for k in weights)
        capped = False
        if p_d.get("below_floor"):
            total = min(total, 0.35)
            capped = True
        rec = {"from_id": a["phrase_id"], "to_id": b["phrase_id"],
               "from_track": a["track_id"], "to_track": b["track_id"],
               "axes": axes, "total": round(float(total), 4), "capped_below_floor": capped,
               "details": {"energy": e_d, "harmonic": h_d, "groove": g_d, "spectral": s_d,
                           "loudness": l_d, "presence": p_d},
               "hash": h}
        if i in validate_idx:
            rec["validation"] = {
                "audio_from": None if aud_f is None else {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in aud_f.items() if k != "mfcc"},
                "audio_to": None if aud_t is None else {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in aud_t.items() if k != "mfcc"},
                "audio_available": bool(aud_f and aud_t),
            }
        transitions.append(rec)

    transitions.sort(key=lambda r: -r["total"])
    for rank, rec in enumerate(transitions, 1):
        rec["rank"] = rank

    axis_names = ["energy", "harmonic", "groove", "spectral", "loudness", "presence"]
    summary = {f"{ax}_stats": stats([float(r["axes"][ax]) for r in transitions]) for ax in axis_names}
    summary["total_stats"] = stats([float(r["total"]) for r in transitions])
    summary["weights"] = {"note": "drums renormalized without harmonic (unpitched)"} if stem == "drums" else {}
    summary["weights"]["default"] = DEFAULT_WEIGHTS
    # validation correlation: manifest proxy vs audio-derived real spectral.
    # axes["spectral"] equals the real score on validation rows, so the honest
    # comparison is details.spectral.manifest_proxy vs axes["spectral"].
    proxy, real = [], []
    for r in transitions:
        v = r.get("validation") or {}
        if not v.get("audio_available"):
            continue
        mp = (r.get("details", {}).get("spectral", {}) or {}).get("manifest_proxy")
        if mp is None:
            continue
        proxy.append(float(mp))
        real.append(float(r["axes"]["spectral"]))
    if len(proxy) >= 4:
        import statistics
        mx, my = statistics.fmean(proxy), statistics.fmean(real)
        num = sum((x - mx) * (y - my) for x, y in zip(proxy, real))
        den = math.sqrt(sum((x - mx) ** 2 for x in proxy) * sum((y - my) ** 2 for y in real)) or 1.0
        summary["validation"] = {"n": len(proxy), "pearson_proxy_vs_real": round(num / den, 4),
                                 "note": "positive = manifest proxy tracks decoded spectral distance"}
    else:
        summary["validation"] = {"n": len(proxy), "note": "insufficient decoded pairs for correlation"}
    summary.update({
        "n_below_floor": sum(1 for r in transitions if r["details"]["presence"]["below_floor"]),
        "n_extreme_stretch": sum(1 for r in transitions if r["details"]["groove"]["extreme_stretch"]),
        "n_mode_mismatch": sum(1 for r in transitions if r["details"]["harmonic"].get("mode_mismatch")),
        "n_clash_interval": sum(1 for r in transitions if r["details"]["harmonic"].get("interval") in (1, 6, 11)),
        "ear_notes": ear_notes,
        "assumptions": ("rate=duration_sec/(960/session_bpm); shift=session-phrase wrapped -6..+6 (musically correct, "
                        "opposite sign to stem_remix keyShift); no Stretch render; chroma/embedding proxy when ears/audio absent"),
    })

    payload = {"version": VERSION,
               "session": {"bpm": float(args.session_bpm), "key": str(args.session_key),
                           "bar": int(args.session_bar), "target_energy": tgt["energy"],
                           "phase": tgt["phase"], "block_sec": round(block_sec, 4)},
               "stem": stem, "n_scored": len(transitions), "seed": int(args.seed),
               "transitions": transitions, "summary": summary}
    out = Path(str(args.out))
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[score-transitions] stem={stem} n={len(transitions)} wrote {out} "
          f"total_mean={summary['total_stats'].get('mean')} min={summary['total_stats'].get('min')} "
          f"max={summary['total_stats'].get('max')} below_floor={summary['n_below_floor']} "
          f"extreme={summary['n_extreme_stretch']} clash={summary['n_clash_interval']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
