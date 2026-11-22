"""Evaluate a conditioned WubGPT checkpoint vs the legacy one.

Samples fixed-seed continuations from both checkpoints, then reports:
  * token stats: per-stem silence rate, unique clusters used, energy-bucket histogram
  * audio stats: rendered mashup onset density (librosa) vs corpus segment rate
    computed from data/processed (fast, no audio decode of the corpus)

Usage (laptop, after training finishes):
  python3 scripts/eval_cond_wubgpt.py \\
      --new-ckpt data/wubgpt/wubgpt_cond.pt --old-ckpt data/wubgpt/wubgpt.pt \\
      --out-dir data/eval_cond --steps 256 --n-samples 4

Exit code 0 = new model meets all gates; 1 = investigate (report still written).
Gates (see RUNBOOK reasoning): density_ratio >= 0.8 vs corpus drums-seg rate,
silence_rate <= legacy + 0.05, unique_clusters >= legacy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def corpus_drum_rate(proc_dir: Path) -> float:
    """Mean drum segments-per-second across processed tracks (onset proxy)."""
    total_seg, total_dur = 0, 0.0
    files = sorted(p for p in proc_dir.glob("*.json") if not p.name.startswith(("_", "meta_")))
    for p in files:
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        try:
            dur = float(doc.get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        n = sum(1 for sg in doc.get("segments", []) or [] if sg.get("stem") == "drums"
                and not sg.get("is_silent", False))
        total_seg += n
        total_dur += dur
    return total_seg / max(total_dur, 1e-9)


def render_onset_density(wav_path: Path) -> float:
    """Onsets per second of a rendered mix (librosa)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    if y.size == 0:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3,
                                   post_avg=5, delta=0.5, wait=10)
    return float(len(peaks)) / (len(y) / float(sr))


def sample_and_stats(model, vqs, seed_rows, steps, style_id, energy_arc, seed):
    from wub_core.model_transformer import sample

    gen = sample(
        model, vqs, seed_rows, steps=steps, temperature=0.75, top_k=32,
        top_p=0.85, seed=seed, style_id=style_id,
        energy_schedule=energy_arc, start_bar8=-1,
    )
    return gen


def token_stats(rows):
    import numpy as np

    stats = {}
    for stem in ("drums", "bass", "other", "vocals"):
        toks = [int(r.get(stem, 0)) for r in rows]
        if not toks:
            stats[stem] = {"silence_rate": 1.0, "unique": 0}
            continue
        # silence id unknown here; approximate: least-frequent? No - caller passes vqs.
        stats[stem] = {"toks": toks}
    energies = [int(r.get("energy", -1)) for r in rows]
    stats["energy_hist"] = [int(np.sum(np.asarray(energies) == b)) for b in range(8)] if energies else []
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate conditioned vs legacy WubGPT")
    ap.add_argument("--new-ckpt", default="data/wubgpt/wubgpt_cond.pt")
    ap.add_argument("--old-ckpt", default="data/wubgpt/wubgpt.pt")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stems", default="data/stems")
    ap.add_argument("--input", default="pandora")
    ap.add_argument("--out-dir", default="data/eval_cond")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--target-bpm", type=float, default=128.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch

    from wub_core.corpus import CorpusStore
    from wub_core.model_markov import render_mashup
    from wub_core.model_transformer import (
        STREAM_ORDER,
        build_dataset,
        load_checkpoint,
        tokens_to_slices,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Corpus drum-segment rate (onset proxy)...")
    corp_rate = corpus_drum_rate(Path(args.processed))
    print(f"  corpus drums segs/sec: {corp_rate:.3f}")

    store = CorpusStore(
        processed_dir=Path(args.processed),
        stems_base=Path(args.stems),
        library_dirs=[Path(args.input)],
    )
    store.refresh()
    pools = store.pools()
    primer = store.analyses[args.seed % len(store.analyses)] if store.analyses else None
    if primer is None:
        raise SystemExit("No analyses found.")

    report = {"corpus_drum_segs_per_sec": corp_rate, "models": {}}
    ok = True
    for tag, ckpt in (("new", args.new_ckpt), ("old", args.old_ckpt)):
        p = Path(ckpt)
        if not p.exists():
            print(f"SKIP {tag}: {p} missing")
            report["models"][tag] = {"missing": True}
            if tag == "new":
                ok = False
            continue
        model, vqs, config = load_checkpoint(p)
        model.to(device)
        use_cond = bool(getattr(model, "cond_vocabs", None))
        print(f"[{tag}] cond={use_cond} params={sum(x.numel() for x in model.parameters())/1e6:.1f}M")
        seed_rows_all = build_dataset([primer], vqs, stride=1)
        seed_rows = seed_rows_all[:16]
        all_gen, sil_rates, uniqs, ehist = [], {}, {}, np.zeros(8)
        for s in range(args.n_samples):
            style_id = (s * 5) % 12 + 1 if use_cond else 0
            arc = [3, 4, 5, 6, 7, 6, 5, 4][: min(8, args.steps)]
            gen = sample_and_stats(model, vqs, seed_rows, args.steps, style_id, arc, args.seed + s)
            all_gen.extend(gen)
            for stem in ("drums", "bass", "other", "vocals"):
                sil = vqs[stem].silence_id if stem in vqs else 0
                toks = [int(r.get(stem, sil)) for r in gen]
                sil_rates.setdefault(stem, []).append(sum(t == sil for t in toks) / max(len(toks), 1))
                uniqs.setdefault(stem, []).append(len(set(toks)))
            for b in range(8):
                ehist[b] += sum(1 for r in gen if int(r.get("energy", -1)) == b)
        entry = {
            "cond": use_cond,
            "mean_silence_rate": {k: float(np.mean(v)) for k, v in sil_rates.items()},
            "mean_unique_tokens": {k: float(np.mean(v)) for k, v in uniqs.items()},
            "energy_hist": [int(x) for x in ehist],
        }
        # Render first sample to audio for onset density.
        first_n = all_gen[: args.steps]
        seq = tokens_to_slices(first_n, pools, vqs)
        tids = {r.track_id for refs in seq.values() for r in refs if r is not None}
        audio_for, fallback = store.audio_for(tids)
        wav = out_dir / f"mashup_{tag}.wav"
        render_mashup(seq=seq, audio_for=audio_for, fallback_audio=fallback,
                      out_path=wav, pitch_lock=True,
                      target_bpm=args.target_bpm if args.target_bpm > 0 else None)
        dens = render_onset_density(wav)
        entry["rendered_onsets_per_sec"] = dens
        entry["density_ratio_vs_corpus"] = dens / max(corp_rate, 1e-9)
        report["models"][tag] = entry
        print(f"[{tag}] silence={entry['mean_silence_rate']} uniq={entry['mean_unique_tokens']}")
        print(f"[{tag}] onsets/sec={dens:.3f} ratio_vs_corpus={entry['density_ratio_vs_corpus']:.2f}")

    new, old = report["models"].get("new", {}), report["models"].get("old", {})
    gates = {}
    if not new.get("missing") and not old.get("missing"):
        gates["density_ratio>=0.8"] = new["density_ratio_vs_corpus"] >= 0.8
        gates["silence<=legacy+0.05"] = all(
            new["mean_silence_rate"][s] <= old["mean_silence_rate"][s] + 0.05
            for s in ("drums", "bass", "other", "vocals")
        )
        gates["unique>=legacy"] = all(
            new["mean_unique_tokens"][s] >= old["mean_unique_tokens"][s]
            for s in ("drums", "bass", "other", "vocals")
        )
        ok = all(gates.values())
    report["gates"] = gates
    report["pass"] = bool(ok)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=1))
    print("GATES:", gates, "-> PASS" if ok else "-> FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
