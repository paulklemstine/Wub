"""Build the tatum-level dataset: every 16th-note slot as 4-stream VQ codes.

Bar frame (6 tokens): [SEC, CHORD, R_D, R_B, R_O, R_V] with prototype ranks
(fuzzy-mapped like the bar dataset). Steps (16 x 4 tokens): per-stem code id,
SIL when nothing sounds, HOLD while a previous onset's slice sustains.

Token layout (model-local ids):
  D codes 0..383, SIL 384, HOLD 385
  B codes 386..641, SIL 642, HOLD 643
  O codes 644..1027, SIL 1028, HOLD 1029
  V codes 1030..1285, SIL 1286, HOLD 1287
  SEC 1288..1292, CHORD 1293..1305 (0..11, 12=unknown)
  R_D 1306..1605, R_B 1606..1905, R_O 1906..2205, R_V 2206..2505
VOCAB = 2506.

Only tracks with all 4 stems (same lesson as bar training: missing stems
teach silence). Outputs data/tatum_dataset.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wub_core.tatum_codec import STEMS, nearest_rank, onset_spans, rank_index

N_RANK = 300
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")


def _code_layout(vq_manifest):
    """FIXED layout independent of K (max 1024 codes/stem), so codebooks can
    grow without rebuilding every consumer:
    D codes 0..1023, SIL 1024, HOLD 1025;
    B codes 1026..2049, SIL 2050, HOLD 2051;
    O codes 2052..3075, SIL 3076, HOLD 3077;
    V codes 3078..4101, SIL 4102, HOLD 4103;
    SEC 4104..4108, CHORD 4109..4121 (0..11, 12=unknown);
    R_D 4122..4421, R_B 4422..4721, R_O 4722..5021, R_V 5022..5321.
    VOCAB = 5322. Unused code ids (>K) never appear in training; the
    renderer treats unknown codes as rest."""
    K = {s: int(vq_manifest["stems"][s]["k"]) for s in STEMS}
    OFF = {"drums": 0, "bass": 1026, "other": 2052, "vocals": 3078}
    R_OFF = {"drums": 4122, "bass": 4422, "other": 4722, "vocals": 5022}
    return K, OFF, R_OFF, {"sec": 4104, "chord": 4109, "vocab": 5322}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--patterns", default="data/global_patterns.json")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--vq", default="data/tatum_vq")
    ap.add_argument("--max-bars-per-track", type=int, default=64)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--stems-list", default=None,
                    help="file of track ids known to have all 4 stems on disk "
                         "(e.g. a listing of a remote stems dir); these override "
                         "the stems_available field in the analysis json")
    ap.add_argument("--val-pct", type=int, default=8)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wub_core.global_vocabulary import GlobalVocabulary, GlobalVocabularyBuilder
    from wub_core.model_transformer import StemVQ, slice_feature_raw

    pats = json.loads(Path(args.patterns).read_bytes())
    gvoc = json.loads(Path(args.vocab).read_bytes())
    rank = rank_index(pats["bar_prototypes"])
    id_cat, id_feat = {}, {}
    for s in STEMS:
        for v in gvoc["vocab_by_stem"][s]:
            id_cat[v["id"]] = str(v.get("category", s)).split(":")[-1]
            id_feat[v["id"]] = v["feature_vector"]
    proto_grids, proto_cats = {}, {}
    for s in STEMS:
        for r, p in enumerate(pats["bar_prototypes"][s]):
            toks = p["tokens"]
            proto_grids[(s, r)] = tuple(i for i, t in enumerate(toks) if t)
            proto_cats[(s, r)] = tuple(id_cat.get(t, ".") if t else "." for t in toks)

    # per-stem VQ: slice -> code via standardized nearest centroid
    vqs = {}
    for s in STEMS:
        vqs[s] = StemVQ.load(Path(args.vq) / s)
    id_code = {}
    for s in STEMS:
        vq = vqs[s]
        for v in gvoc["vocab_by_stem"][s]:
            f = [float(x) for x in v["feature_vector"]]
            fake = {"timbre": f[0:12], "pitches": f[12:24],
                    "loudness_max": f[24], "duration": float(math.expm1(max(0.0, f[25])))}
            id_code[(s, v["id"])] = int(vq.encode(fake))

    def code_of(stem, gid):
        if gid is None:
            return None
        return id_code.get((stem, gid))

    builder = GlobalVocabularyBuilder()
    gvoc_obj = _vocab_obj(gvoc)
    K, OFF, R_OFF, BASE = _code_layout(json.loads(Path("frontend/public/data/tatum_vq.json").read_bytes()))
    SEC_OFF, CHORD_OFF, VOCAB = BASE["sec"], BASE["chord"], BASE["vocab"]
    print(f"  code layout: K={K} vocab={VOCAB}", flush=True)
    files = sorted(p for p in Path(args.processed).glob("*.json") if not p.name.startswith(("_", "meta_")))
    if args.max_tracks:
        files = files[: args.max_tracks]
    stems_have = None
    if args.stems_list:
        stems_have = {ln.strip() for ln in Path(args.stems_list).read_text().splitlines() if ln.strip()}
        print(f"  stems list: {len(stems_have)} complete track ids", flush=True)

    try:
        from wub_core.section_arcs import bar_features as _bar_features
        from wub_core.section_arcs import segment_sections as _segment_sections
    except Exception:
        _bar_features = None  # type: ignore[assignment]
        _segment_sections = None  # type: ignore[assignment]

    train, val = [], []
    n_tracks = n_bars = n_empty = n_fuzzy = 0
    for jf in files:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        if not _track_has_stems(raw, jf.stem, stems_have):
            continue
        segs = raw.get("segments") or []
        tatums = (raw.get("tatums") or [])[: args.max_bars_per_track * 16]
        if not tatums:
            continue
        tid = str(raw.get("track_id") or jf.stem)
        bars, _ = builder.remap_bars_fast(segs, tatums, gvoc_obj)
        # section per bar
        sec_of_bar = {}
        if _bar_features is not None and _segment_sections is not None:
            try:
                feats = _bar_features(segs, tatums)
                for sp in _segment_sections(feats):
                    for b in range(sp.start_bar, min(sp.end_bar, len(bars))):
                        sec_of_bar[b] = sp.label
            except Exception:
                pass
        seq = []
        for b, bar in enumerate(bars):
            if not any(any(bar[s]) for s in STEMS):
                n_empty += 1
                continue
            # header: SEC, CHORD, 4 ranks
            sec = sec_of_bar.get(b, "GROOVE_A")
            toks = [SEC_OFF + (SECTIONS.index(sec) if sec in SECTIONS else 1)]
            chord, pitch_src = 12, None
            for s in ("bass", "other"):
                for gid in bar[s]:
                    if gid and gid in id_feat:
                        f = id_feat[gid]
                        pitch_src = int(np.argmax(f[12:24]))
                        break
                if pitch_src is not None:
                    break
            toks.append(CHORD_OFF + (pitch_src if pitch_src is not None else 12))
            ranks = {}
            ok = True
            for s in STEMS:
                sig = "-".join([g if g else "." for g in bar[s]])
                protos = pats["bar_prototypes"][s]
                grids = [proto_grids[(s, r)] for r in range(len(protos))]
                cats = [proto_cats[(s, r)] for r in range(len(protos))]
                r, exact = nearest_rank(s, sig, bar[s], rank[s], grids, cats, id_cat)
                if not exact:
                    n_fuzzy += 1
                if r >= N_RANK:
                    ok = False
                    break
                ranks[s] = r
                toks.append(R_OFF[s] + r)
            if not ok:
                continue
            # 16 steps x 4 codes with HOLD fill from spans
            spans = {s: onset_spans(bar[s]) for s in STEMS}
            for step in range(16):
                for s in STEMS:
                    gid = bar[s][step]
                    if gid is not None:
                        toks.append(OFF[s] + code_of(s, gid))
                    else:
                        # sustaining slice from an earlier onset in this bar?
                        held = any(o < step < o + spans[s].get(o, 1) for o in spans[s])
                        toks.append(OFF[s] + 1025 if held else OFF[s] + 1024)
            seq.extend(toks)
        if len(seq) // 70 < 4:
            continue
        n_tracks += 1
        n_bars += len(seq) // 70
        h = int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 100
        (val if h < args.val_pct else train).append({"track_id": tid, "bars": len(seq) // 70, "tokens": seq})
    Path("data/tatum_dataset.json").write_text(json.dumps({
        "version": 1, "vocab": VOCAB, "tokens_per_bar": 70,
        "sections": list(SECTIONS),
        "train": train, "val": val,
        "tracks": n_tracks, "bars": n_bars, "empty_bars_dropped": n_empty,
    }, separators=(",", ":")))
    print(f"dataset: {n_tracks} tracks, {n_bars} bars ({n_empty} empty dropped)", flush=True)
    print(f"  train seqs: {len(train)}, val seqs: {len(val)}, fuzzy-mapped: {n_fuzzy}", flush=True)


def _track_has_stems(raw: dict, fallback_tid: str, stems_have: set[str] | None) -> bool:
    """True when the analysis carries all four stems.

    ``--stems-list`` covers the case where the stems live on another machine
    and the analysis's own ``stems_available`` field is stale: a track id in
    that list is trusted to have all four stems on disk.
    """
    if set(raw.get("stems_available") or []) >= set(STEMS):
        return True
    if not stems_have:
        return False
    return str(raw.get("track_id") or fallback_tid) in stems_have


def _vocab_obj(gvoc_raw):
    from wub_core.global_vocabulary import GlobalVocabulary
    gv = GlobalVocabulary(budget={s: 0 for s in STEMS})
    for s in STEMS:
        from wub_core.global_vocabulary import GlobalVocabSlice
        for d in gvoc_raw["vocab_by_stem"][s]:
            gv.vocab_by_stem[s].append(GlobalVocabSlice(
                id=d["id"], stem=s, category=d.get("category", s),
                feature_vector=d["feature_vector"], clarity=float(d.get("clarity", 0.5)),
                pitch=int(d.get("pitch", 0)), duration=float(d.get("duration", 0.25)),
                centroid=float(d.get("centroid", 1000.0)), track_id=d.get("track_id", ""),
                segment_id=d.get("segment_id", ""), start=float(d.get("start", 0.0)),
                loud_max=float(d.get("loud_max", -12.0)), url=d.get("url", "")))
    gv.build_index()
    return gv


if __name__ == "__main__":
    main()
