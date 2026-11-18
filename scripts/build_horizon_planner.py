"""Emit the optional short-horizon planner policy config.

This is a policy layer over signals already learned from the corpus (next-bar
pattern head, phrase taste quality, song arc and cohort priors), not a new
acoustic model. It is shipped as a sidecar so the whole planner can be removed
by one flag; the browser treats the weights as a bounded additive bias only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build() -> dict:
    return {
        "format": "wub-horizon-planner-v1",
        "version": 2,
        "horizon_blocks": 3,
        "weights": {"bar": 0.38, "phrase": 0.12, "sequence": 0.50},
        "max_score": 0.18,
        "arc_tolerance": 0.12,
        "report": {
            "source": "policy config over corpus-learned signals",
            "signals": [
                "pattern_bar_model (next-bar fit from 254,927 real bars)",
                "phrase earQ / taste quality blended with recency freshness",
                "song_arc_model future energy target + transition_cohort cohort prior",
            ],
            "use": "additive ranking bias only; never schedules, repitches, retimes or changes phrase boundaries",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("frontend/public/data/kits/horizon_planner.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(), indent=2) + "\n")


if __name__ == "__main__":
    main()
