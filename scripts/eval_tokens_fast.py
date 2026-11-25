"""Fast token-level gates: no corpus parse, no rendering.

Loads two checkpoints, samples fixed seeds from a single primer track, and
reports per-stem silence rate, unique tokens, and energy-bucket histograms.
Decides density-adjacent gates that don't need audio:
  silence<=legacy+0.05, unique>=legacy, energy-spread (std of hist) >= legacy.
Usage (laptop):
  python3 scripts/eval_tokens_fast.py --new-ckpt data/wubgpt/wubgpt_cond.pt \\
      --old-ckpt data/wubgpt/wubgpt.pt --primer-json data/processed/<id>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast token-level model comparison")
    ap.add_argument("--new-ckpt", default="data/wubgpt/wubgpt_cond.pt")
    ap.add_argument("--old-ckpt", default="data/wubgpt/wubgpt.pt")
    ap.add_argument("--primer-json", default="")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/eval_cond/token_gates.json")
    ap.add_argument("--n-seeds", type=int, default=4,
                    help="independent rollouts per condition (averaged; autoregressive variance is large)")
    ap.add_argument("--steer", action="store_true",
                    help="energy-steering test: sample cond model at energy 2 vs 7 "
                         "(temp 1.0) and compare drum non-silence rate")
    args = ap.parse_args()

    import numpy as np
    import torch

    from wub_core.model_transformer import build_dataset, load_checkpoint, sample
    from wub_core.schema import TrackAnalysis

    if args.primer_json:
        primer_doc = json.loads(Path(args.primer_json).read_text())
    else:
        proc = Path(args.processed)
        files = sorted(p for p in proc.glob("*.json") if not p.name.startswith(("_", "meta_")))
        if not files:
            raise SystemExit(f"No analyses in {proc}")
        primer_doc = json.loads(files[args.seed % len(files)].read_text())
    primer = TrackAnalysis(**primer_doc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {}
    ok = True
    if args.steer:
        from wub_core.model_transformer import sample as _sample

        model, vqs, _ = load_checkpoint(args.new_ckpt)
        if not getattr(model, "cond_vocabs", None):
            raise SystemExit("steer test needs a conditioned checkpoint")
        model.to(device)
        seed_rows = build_dataset([primer], vqs, stride=1)[:16]
        steer = {}

        def _probe(energy, density, temp, tag):
            rates, uniqs, n = [], set(), max(1, int(args.n_seeds))
            for r in range(n):
                gen = _sample(model, vqs, seed_rows, steps=args.steps, temperature=temp,
                              top_k=0, top_p=0.0, seed=args.seed + r, style_id=1,
                              energy=energy, density=density, start_bar8=-1)
                dns, du = [], set()
                for row in gen:
                    t = int(row.get("drums", -1))
                    dns.append(t != vqs["drums"].silence_id)
                    du.add(t)
                rates.append(sum(dns) / max(len(dns), 1))
                uniqs.update(du)
            steer[tag] = {"drum_nonsilence_rate": sum(rates) / len(rates),
                          "drum_unique": len(uniqs)}
            print(f"[steer {tag} n={n}] {steer[tag]}", flush=True)

        for temp in (0.75, 1.0):
            _probe(2, 4, temp, f"energy2_temp{temp}")
            _probe(7, 4, temp, f"energy7_temp{temp}")
        for temp in (0.75, 1.0):
            _probe(5, 1, temp, f"density1_temp{temp}")
            _probe(5, 4, temp, f"density4_temp{temp}")
        report = {"steer": steer}
        spread_e = (steer["energy7_temp1.0"]["drum_nonsilence_rate"]
                    - steer["energy2_temp1.0"]["drum_nonsilence_rate"])
        spread_d = (steer["density4_temp1.0"]["drum_nonsilence_rate"]
                    - steer["density1_temp1.0"]["drum_nonsilence_rate"])
        report["energy_steers_drums"] = bool(spread_e > 0.05)
        report["density_steers_drums"] = bool(spread_d > 0.20)
        print("energy steers drums:", report["energy_steers_drums"], f"(delta={spread_e:.3f})", flush=True)
        print("density steers drums:", report["density_steers_drums"], f"(delta={spread_d:.3f})", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        raise SystemExit(0 if report["energy_steers_drums"] else 1)
    for tag, ckpt in (("new", args.new_ckpt), ("old", args.old_ckpt)):
        model, vqs, _ = load_checkpoint(ckpt)
        model.to(device)
        use_cond = bool(getattr(model, "cond_vocabs", None))
        seed_rows = build_dataset([primer], vqs, stride=1)[:16]
        sil, uniq, ehist = {}, {}, np.zeros(8)
        for s in range(args.n_samples):
            style_id = (s * 5) % 12 + 1 if use_cond else 0
            arc = [3, 4, 5, 6, 7, 6, 5, 4]
            gen = sample(model, vqs, seed_rows, steps=args.steps, temperature=0.75,
                         top_k=32, top_p=0.85, seed=args.seed + s, style_id=style_id,
                         energy_schedule=arc, start_bar8=-1)
            for stem in ("drums", "bass", "other", "vocals"):
                sid = vqs[stem].silence_id if stem in vqs else 0
                toks = [int(r.get(stem, sid)) for r in gen]
                sil.setdefault(stem, []).append(sum(t == sid for t in toks) / max(len(toks), 1))
                uniq.setdefault(stem, []).append(len(set(toks)))
            for r in gen:
                ehist[int(r.get("energy", 0))] += 1
        report[tag] = {
            "cond": use_cond,
            "mean_silence_rate": {k: float(np.mean(v)) for k, v in sil.items()},
            "mean_unique_tokens": {k: float(np.mean(v)) for k, v in uniq.items()},
            "energy_hist": [int(x) for x in ehist],
            "energy_spread_std": float(np.std(ehist)),
        }
        print(f"[{tag}] silence={report[tag]['mean_silence_rate']}", flush=True)
        print(f"[{tag}] unique={report[tag]['mean_unique_tokens']}", flush=True)
        print(f"[{tag}] energy_hist={report[tag]['energy_hist']}", flush=True)
    new, old = report["new"], report["old"]
    gates = {
        "silence<=legacy+0.05": all(
            new["mean_silence_rate"][s] <= old["mean_silence_rate"][s] + 0.05
            for s in ("drums", "bass", "other", "vocals")),
        "unique>=legacy": all(
            new["mean_unique_tokens"][s] >= old["mean_unique_tokens"][s]
            for s in ("drums", "bass", "other", "vocals")),
        "energy_spread>=legacy": new["energy_spread_std"] >= old["energy_spread_std"],
    }
    report["gates"] = gates
    report["pass"] = bool(all(gates.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("GATES:", gates, "-> PASS" if all(gates.values()) else "-> FAIL", flush=True)
    raise SystemExit(0 if all(gates.values()) else 1)


if __name__ == "__main__":
    main()
