"""Emit the optional variety-prior config (policy over measured selector bias)."""
import argparse, json
from pathlib import Path

CONFIG = {
    "format": "wub-variety-prior-v1", "version": 1,
    "track_history_bars": 16,
    "track_reuse_penalty": 0.18,
    "vocal_freshness_weight": 0.30,
    "joint_reserve": 4,
    "source_novelty_weight": 0.30,
    "max_recall_per_set": 1,
    "report": {
        "source": "measurement of the shipped bank and selector",
        "findings": {
            "per_stem_judge_top1_median": {"drums": 0.982, "bass": 0.984, "vocals": 0.961, "other": 0.740},
            "earq_nonbass_max": 0.578,
            "cluster_tracks": "9-48",
            "phrase_quality": "no extreme outliers",
        },
        "use": "bounded additive variety bias; no filtering, no audio changes",
    },
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/variety_prior.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(CONFIG, indent=2) + "\n")

if __name__ == "__main__":
    main()
