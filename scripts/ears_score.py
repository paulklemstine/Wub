#!/usr/bin/env python3
"""Fuse DSP + embedding ears into per-phrase quality scores.

Quality-first soft prior for the remix engine: good phrases win ties, weak
phrases stay eligible (breaks/drops still need quiet material). Output is a
small additive sidecar consumed by scripts/ears_merge.py:

    {version: 1, phrase_ids: [...], ears: {
        phrase_id: {q: 0..1, perStem: {stem: {q: 0..1, ...}}, reserved: [7 floats]}}}

`reserved` fills taste-dataset dims 41..47 (zero-drift slot, see
build_taste_dataset.FEATURE_LAYOUT): LUFS, crest, flatness, onset clarity,
rumble, vocal audibility, ear-confidence. Values are 0..1; missing ears
default to 0.5 (neutral) so old rows stay valid.

Usage (laptop, AGENTS workflow):
    PYTHONPATH=. python3 scripts/ears_score.py \
        --ears-dsp /tmp/wubtrain/ears/dsp.json \
        --manifest kits_ship/kit_manifest.json \
        --out /tmp/wubtrain/ears/ears_scores.json

    # with embeddings (optional, improves novelty/diversity term):
    PYTHONPATH=. python3 scripts/ears_score.py \
        --ears-dsp /tmp/wubtrain/ears/dsp.json \
        --ears-embed /tmp/wubtrain/ears/embed.json \
        --manifest kits_ship/kit_manifest.json \
        --out /tmp/wubtrain/ears/ears_scores.json

Smoke test on the shipped bank (no /mnt/e needed):
    PYTHONPATH=. python3 scripts/ears_score.py \
        --ears-dsp /tmp/opencode/ears_dsp_smoke.json \
        --manifest frontend/public/data/kits/kit_manifest.json \
        --out /tmp/opencode/ears_scores_smoke.json
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

STEMS = ("drums", "bass", "other", "vocals")


def _phrases_by_id(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    by_id = {}
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            by_id[phrase["id"]] = {
                "track_id": kit.get("track_id"),
                "style": kit.get("style", "unknown"),
                "section": str(phrase.get("section", "GROOVE_A")).upper(),
                "start_bar": phrase.get("start_bar"),
            }
    return by_id


def _z_sigmoid(values):
    """Per-style z-scored sigmoid (mirrors score_kits.ranker_quality)."""
    if not values:
        return {}
    vals = [v for v in values.values() if v is not None]
    if len(vals) < 2:
        return {k: 0.5 for k in values}
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) or 1.0
    out = {}
    for key, v in values.items():
        if v is None:
            out[key] = 0.5
        else:
            z = (v - mean) / (std if std > 1e-9 else 1.0)
            out[key] = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, z))))
    return out


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _reserved_for(stem_scores):
    """Best-effort reserved dims 41..47 from DSP fields (0..1 each)."""
    # Caller passes per-stem dicts; aggregate conservatively (mean of stems).
    def mean(key, default=0.5):
        vals = [s.get(key) for s in stem_scores.values() if s.get(key) is not None]
        return statistics.mean(vals) if vals else default

    lufs = mean("loudness_lufs", -18.0)
    lufs_n = _clamp01((lufs + 60.0) / 40.0)
    # crest = peak_linear / rms_linear; DSP gives peak + rms_db.
    crests = []
    for s in stem_scores.values():
        peak, rms_db = s.get("peak"), s.get("rms_db")
        if peak is not None and rms_db is not None:
            rms = 10 ** (rms_db / 20.0)
            crests.append(peak / rms if rms > 0 else 1.0)
    crest = statistics.mean(crests) if crests else 1.0
    crest_n = _clamp01(math.log10(max(1.0, crest)) / 2.0) if crest > 0 else 0.0
    # flatness proxy: low spectral contrast ~ flat (harsh/noisy).
    contrast = mean("spectral_contrast_mean", 18.0)
    flatness_n = _clamp01(1.0 - (contrast / 30.0))
    # onset clarity: normalised onset strength.
    clarity = mean("onset_strength_mean", 1.0)
    clarity_n = _clamp01(clarity / 4.0)
    # rumble proxy: very low ZCR with low centroid suggests mud/rumble.
    zcr = mean("zcr_mean", 0.05)
    rumble_n = _clamp01(1.0 - (zcr / 0.10)) if zcr < 0.10 else 0.0
    # vocal audibility: DSP has no PANNs here; proxy = vocals onset presence.
    vocal = stem_scores.get("vocals", {})
    vocal_n = _clamp01((vocal.get("onset_strength_mean") or 0.5) / 2.0)
    # confidence: fraction of non-null ears (DSP always, others optional).
    conf = 0.5
    try:
        sample = next(iter(stem_scores.values()))
        checks = [
            sample.get("essentia_available") is not None,
            sample.get("pitch_available") is not None,
            sample.get("loudness_lufs_source") != "rms_db_fallback",
        ]
        conf = 0.4 + 0.2 * sum(1 for c in checks if c)
    except StopIteration:
        pass
    return [lufs_n, crest_n, flatness_n, clarity_n, rumble_n, vocal_n, _clamp01(conf)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ears-dsp", required=True, help="ears_extract.py output JSON")
    ap.add_argument("--ears-embed", default=None, help="ears_embed.py output JSON (optional)")
    ap.add_argument("--manifest", required=True, help="kit_manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    dsp = json.loads(Path(args.ears_dsp).read_text())
    meta = _phrases_by_id(args.manifest)
    scores = dsp.get("scores", {})

    embed_q = {}
    if args.ears_embed:
        try:
            emb = json.loads(Path(args.ears_embed).read_text())
            # Novelty proxy: distance of each embedding to the centroid,
            # mapped so isolated (novel) phrases score slightly higher.
            vecs = emb.get("embeddings", {})
            if vecs:
                all_vecs = [v for stems in vecs.values() for v in stems.values() if v]
                if all_vecs:
                    dim = len(all_vecs[0])
                    centroid = [statistics.mean(col) for col in zip(*all_vecs)]
                    dists = {}
                    for pid, stems in vecs.items():
                        ds = []
                        for vec in stems.values():
                            if vec and len(vec) == dim:
                                ds.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, centroid))))
                        if ds:
                            dists[pid] = statistics.mean(ds)
                    # Novelty: above-median distance -> small bonus (0..1).
                    ranked = _z_sigmoid(dists)
                    embed_q = ranked
        except Exception as err:  # never block DSP scoring on embed issues
            print(f"[ears_score] embed merge skipped: {err}", file=sys.stderr)

    # Raw quality per phrase: DSP quality_score is per stem; phrase q = mean.
    raw_q = {}
    per_stem_raw = {}
    for pid, stems in scores.items():
        qs = {}
        for stem, s in stems.items():
            q = s.get("quality_score", 0.5)
            qs[stem] = q if q is not None else 0.5
        per_stem_raw[pid] = qs
        raw_q[pid] = statistics.mean(qs.values()) if qs else 0.5

    # Per-style z-scored calibration (like score_kits.ranker_quality).
    styles = {}
    for pid, q in raw_q.items():
        style = meta.get(pid, {}).get("style", "unknown")
        styles.setdefault(style, {})[pid] = q
    calibrated = {}
    for style, vals in styles.items():
        calibrated.update(_z_sigmoid(vals))

    ears = {}
    for pid in sorted(scores.keys()):
        qs = per_stem_raw.get(pid, {})
        # Per-stem calibrated q (same style map, per stem raw values).
        per_stem_q = {}
        for stem in STEMS:
            if stem in qs:
                svals = {p: per_stem_raw[p][stem] for p in per_stem_raw if stem in per_stem_raw[p]}
                style_map = {p: svals[p] for p in svals if meta.get(p, {}).get("style", "unknown") == meta.get(pid, {}).get("style", "unknown")}
                per_stem_q[stem] = _z_sigmoid(style_map).get(pid, 0.5) if style_map else calibrated.get(pid, 0.5)
        q = calibrated.get(pid, 0.5)
        # Embed novelty nudges quality up/down slightly (never dominates).
        if pid in embed_q:
            q = _clamp01(q + 0.10 * (embed_q[pid] - 0.5))
        stem_scores = scores[pid]
        ears[pid] = {
            "q": round(q, 4),
            "perStem": {k: {"q": round(float(v), 4)} for k, v in per_stem_q.items()},
            "reserved": [round(float(v), 4) for v in _reserved_for(stem_scores)],
            "style": meta.get(pid, {}).get("style", "unknown"),
            "section": meta.get(pid, {}).get("section", "GROOVE_A"),
        }

    out = {
        "version": 1,
        "model": "ears_dsp_v1" + ("+embed" if embed_q else ""),
        "phrase_ids": sorted(ears.keys()),
        "ears": ears,
        "note": "Quality-first soft prior; q is per-style z-scored DSP quality (embed novelty nudge if provided). reserved -> taste dims 41..47.",
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    qs = [v["q"] for v in ears.values()]
    print(f"[ears_score] phrases={len(ears)} q mean={statistics.mean(qs):.3f} "
          f"p10={sorted(qs)[len(qs)//10]:.3f} p90={sorted(qs)[9*len(qs)//10]:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
