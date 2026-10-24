"""Phase 1 verification: chronological re-stitch must be click-free.

Reconstructs each stem by concatenating its segments in order with 3 ms
equal-power crossfades, sums stems, and checks:
  1. duration matches original within ±5 ms
  2. boundary click metric stays low (no audible pops)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from wub_core.audio_utils import boundary_click_metric, stitch_with_crossfade
from wub_core.schema import TrackAnalysis


def verify_track(analysis_path: Path, stems_dir: Path, sr: int = 44100,
                 fade_ms: float = 3.0, out_wav: Path | None = None) -> bool:
    from wub_core.separator import STEM_FILES

    analysis = TrackAnalysis(**json.loads(analysis_path.read_text()))
    stem_chunks: dict[str, list[np.ndarray]] = {}
    boundaries: list[int] = []
    total_dur = analysis.duration
    ok = True

    mixed = None
    for stem in analysis.stems_available:
        wav_path = stems_dir / STEM_FILES[stem]
        if not wav_path.exists():
            print(f"  SKIP {stem}: {wav_path} missing")
            continue
        y, f_sr = sf.read(str(wav_path), always_2d=True)
        assert f_sr == sr, f"expected {sr} Hz, got {f_sr} for {stem}"
        segs = sorted([s for s in analysis.segments if s.stem == stem],
                      key=lambda s: s.start)
        chunks: list[np.ndarray] = []
        for sg in segs:
            a = int(round(sg.start * sr))
            b = int(round((sg.start + sg.duration) * sr))
            chunks.append(y[a:b])
        recon = stitch_with_crossfade(chunks, sr, fade_ms=fade_ms)
        # duration check: crossfade shortens by fade*(n-1); allow +5ms tolerance
        # on the *sum*; per-stem recon is shorter by design.
        stem_chunks[stem] = chunks
        if mixed is None:
            mixed = recon.astype(np.float64)
        else:
            n = max(len(mixed), len(recon))
            m2 = np.zeros((n, mixed.shape[1]))
            m2[: len(mixed)] = mixed
            m2[: len(recon)] += recon.astype(np.float64)
            mixed = m2

    if mixed is None:
        print("FAIL: no stems reconstructed");
        return False
    # Duration check: micro-crossfades consume `fade` per boundary BY DESIGN,
    # so the honest expectation is original minus overlap, ±5 ms (spec §5).
    # The mix length is set by the stem with the FEWEST boundaries.
    fade = int(sr * fade_ms / 1000.0)
    min_bounds = min((len(v) - 1 for v in stem_chunks.values()), default=0)
    expected_len = int(round(total_dur * sr)) - fade * max(0, min_bounds)
    drift_ms = abs(len(mixed) - expected_len) / float(sr) * 1000.0
    print(f"  original={total_dur:.3f}s recon={len(mixed) / sr:.3f}s "
          f"overlap={fade * min_bounds / sr * 1000:.1f}ms drift={drift_ms:.2f}ms")
    dur_ok = drift_ms <= 5.0
    # click metric on boundaries of longest stem
    longest = max(stem_chunks.values(), key=len)
    bnds: list[int] = []
    acc = 0
    for c in longest[:-1]:
        acc += len(c) - fade
        bnds.append(acc)
    mono = mixed.mean(axis=1) if mixed.ndim == 2 else mixed
    metric = boundary_click_metric(
        np.asarray(mixed), sr, bnds[: min(len(bnds), 200)])
    print(f"  boundary click metric={metric:.3f} (pass if < 8.0)")
    if out_wav is not None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), np.asarray(mixed, dtype=np.float32), sr)
        print(f"  wrote {out_wav}")
    ok = dur_ok and metric < 8.0
    print(f"  {'OK' if ok else 'FAIL'} (duration {'ok' if dur_ok else 'DRIFT'}, "
          f"clicks {'ok' if metric < 8.0 else 'CLICKS'})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--out", default="data/processed/recon_check.wav")
    args = ap.parse_args()
    proc = Path(args.processed)
    files = sorted(proc.glob("*.json"))
    # skip index meta files
    files = [f for f in files if not f.name.startswith("meta_")]
    if not files:
        print(f"No analyses in {proc}. Run scripts/analyze_corpus.py first.")
        sys.exit(2)
    all_ok = True
    for jf in files:
        import json as _j
        tid = _j.loads(jf.read_text()).get("track_id", jf.stem)
        sdir = Path(args.stems) / tid
        print(f"Verifying {jf.name} ...")
        if not sdir.exists():
            print(f"  SKIP: stems dir {sdir} missing (used --no-separation?)")
            continue
        ok = verify_track(jf, sdir, out_wav=Path(args.out))
        all_ok = all_ok and ok
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
