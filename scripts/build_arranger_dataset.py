"""Build the bar-arranger GPT dataset: ordered token bars per track.

Bar encoding (6 tokens): [SEC, CHORD, D, B, O, V]
- D/B/O/V: prototype rank 0..299 per stem (count-desc from global_patterns),
  UNKຸمامwhen the bar sig is outside top-300 (per-stem UNK ids).
- SEC: section label 0..4 from section_arcs segmentation of the same track.
- CHORD: bass first-onset slice pitch 0..11 (fallback: other first-onset),
  12 = unknown (no pitched onset).

Only stems present in stems_available teach; bars with all 4 stems empty
are dropped (same rule as pattern training). Deterministic 95/5
train/val split by track_id hash.

Outputs data/arranger_dataset.json + data/arranger_vocab.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STEMS = ("drums", "bass", "other", "vocals")
SECTIONS = ("INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN")

# Token id layout
N_PROTO = 300
OFF = {"drums": 0, "bass": 301, "other": 602, "vocals": 903}
UNK = {"drums": 300, "bass": 601, "other": 902, "vocals": 1203}
SEC_OFF = 1204
CHORD_OFF = 1209  # +0..11 pitch, +12 unknown
VOCAB = 1222


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--patterns", default="data/global_patterns.json")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--max-bars-per-track", type=int, default=64)
    ap.add_argument("--max-tracks", type=int, default=None)
    ap.add_argument("--val-pct", type=int, default=5)
    ap.add_argument("--stems-only", action="store_true",
                    help="only tracks with all 4 separated stems (clean drum/bass/vocal signal)")
    args = ap.parse_args()

    from wub_core.global_vocabulary import GlobalVocabulary, GlobalVocabularyBuilder

    pats = json.loads(Path(args.patterns).read_bytes())
    rank = {}
    for stem in STEMS:
        rank[stem] = {p["signature"]: i for i, p in enumerate(pats["bar_prototypes"][stem])}

    gvoc = GlobalVocabulary.load(Path(args.vocab))
    id_pitch = {}
    for stem in STEMS:
        for vs in gvoc.vocab_by_stem.get(stem, []):
            id_pitch[vs.id] = int(vs.pitch)

    try:
        from wub_core.section_arcs import bar_features as _bar_features
        from wub_core.section_arcs import segment_sections as _segment_sections
    except Exception:
        _bar_features = None  # type: ignore[assignment]
        _segment_sections = None  # type: ignore[assignment]

    builder = GlobalVocabularyBuilder()
    # Nearest-prototype map: exact slice-ID sigs are brittle (medoid noise
    # fragments identical grooves), so bars map to the closest top-300
    # prototype by onset-grid + slice-category distance. Coverage ~100%.
    proto_grid = {}
    proto_cats = {}
    id_cat = {}
    for stem in STEMS:
        for v in gvoc.vocab_by_stem.get(stem, []):
            id_cat[v.id] = str(v.category).split(":")[-1]
    for stem in STEMS:
        for r, p in enumerate(pats["bar_prototypes"][stem]):
            toks = p["tokens"]
            grid = tuple(i for i, t in enumerate(toks) if t)
            cats = tuple(id_cat.get(t, ".") if t else "." for t in toks)
            proto_grid[(stem, r)] = grid
            proto_cats[(stem, r)] = cats

    def nearest_rank(stem, sig, toks):
        if sig in rank[stem]:
            return rank[stem][sig], True
        grid = tuple(i for i, t in enumerate(toks) if t)
        cats = tuple(id_cat.get(t, ".") if t else "." for t in toks)
        best, bestd = 300, 1e9
        glen = len(grid)
        for r in range(len(pats["bar_prototypes"][stem])):
            pg = proto_grid[(stem, r)]
            pc = proto_cats[(stem, r)]
            common = len(set(grid) & set(pg))
            d = (glen + len(pg) - 2 * common)
            for i in grid:
                if i < 16 and pc[i] != "." and cats[i] != pc[i]:
                    d += 0.5
            if d < bestd:
                bestd, best = d, r
        return best, False
    files = sorted(p for p in Path(args.processed).glob("*.json") if not p.name.startswith(("_", "meta_")))
    if args.max_tracks:
        files = files[: args.max_tracks]

    train, val = [], []
    n_tracks = n_bars = n_empty = n_fuzzy = 0
    for jf in files:
        try:
            raw = json.loads(jf.read_bytes())
        except Exception:
            continue
        if args.stems_only and set(raw.get("stems_available") or []) < {"drums", "bass", "other", "vocals"}:
            continue
        segs = raw.get("segments") or []
        tatums = (raw.get("tatums") or [])[: args.max_bars_per_track * 16]
        if not tatums:
            continue
        tid = str(raw.get("track_id") or jf.stem)
        stems_avail = set(raw.get("stems_available") or [])
        use = {s: (not stems_avail or s in stems_avail or s == "other") for s in STEMS}
        bars, _ = builder.remap_bars_fast(segs, tatums, gvoc)

        # Section label per bar from audio-derived segmentation.
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
            if not any(any(bar[s]) for s in STEMS if use[s]):
                n_empty += 1
                continue
            toks = []
            # Section + chord come first so the model learns arrangement flow.
            sec = sec_of_bar.get(b, "GROOVE_A")
            toks.append(SEC_OFF + (SECTIONS.index(sec) if sec in SECTIONS else 1))
            chord = 12
            for stem in ("bass", "other"):
                if not use[stem]:
                    continue
                for gid in bar[stem]:
                    if gid and gid in id_pitch:
                        chord = id_pitch[gid] % 12
                        break
                if chord != 12:
                    break
            toks.append(CHORD_OFF + chord)
            for stem in STEMS:
                if not use[stem] or not any(bar[stem]):
                    toks.append(UNK[stem])
                    continue
                sig = "-".join([g if g else "." for g in bar[stem]])
                r, exact = nearest_rank(stem, sig, bar[stem])
                if not exact:
                    n_fuzzy += 1
                toks.append(OFF[stem] + r if r < 300 else UNK[stem])
            seq.extend(toks)
        if len(seq) // 6 < 8:
            continue
        n_tracks += 1
        n_bars += len(seq) // 6
        h = int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 100
        (val if h < args.val_pct else train).append({"track_id": tid, "bars": len(seq) // 6, "tokens": seq})

    Path("data/arranger_dataset.json").write_text(json.dumps({
        "version": 1, "vocab": VOCAB, "tokens_per_bar": 6,
        "sections": list(SECTIONS),
        "train": train, "val": val,
        "tracks": n_tracks, "bars": n_bars, "empty_bars_dropped": n_empty,
    }, separators=(",", ":")))
    Path("data/arranger_vocab.json").write_text(json.dumps({
        "version": 1, "offsets": OFF, "unk": UNK, "sec_off": SEC_OFF,
        "chord_off": CHORD_OFF, "vocab": VOCAB, "n_proto": N_PROTO,
        "sections": list(SECTIONS),
    }, separators=(",", ":")))
    print(f"dataset: {n_tracks} tracks, {n_bars} bars ({n_empty} empty dropped)", flush=True)
    print(f"  train seqs: {len(train)}, val seqs: {len(val)}, fuzzy-mapped bars: {n_fuzzy}", flush=True)


if __name__ == "__main__":
    main()
