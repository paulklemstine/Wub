# Learned conductor

The conductor is the first offline-trained model that decides all four musical
things at once instead of scoring one narrow seam:

- **next section** (5 classes),
- **next target energy**,
- **which stems change** at the 4-bar boundary (4 logits),
- **the next 4-bar onset/chroma pattern of each stem** (4 × 48).

## Training data

`scripts/train_conductor.py` reads the real-song analyses in
`data/processed/*.json`. For each 4-bar boundary it builds a 392-d state from
the previous two 4-bar cells of all four stems (onset grid, density,
downbeats, quarters, root, chroma, activity) plus section/energy context, and
trains a 392→128→128→202 tanh MLP against the *observed* next cell.

The first run used **1,894 songs / 57,347 examples**, a track-disjoint 90/10
split (1,705 / 189 tracks), 18 epochs on the WSL GPU:

| head | model | baseline |
|---|---|---|
| section accuracy | 0.549 | 0.515 (majority) |
| energy MAE | 0.0717 | 0.0730 (mean) |
| change AUC | drums 0.787 / bass 0.715 / other 0.738 / vocals 0.723 | 0.5 |
| pattern MSE | 0.0681 | 0.0880 (mean) |

Retrain with:

```bash
PYTHONPATH=. python3 scripts/train_conductor.py \
  --processed /mnt/e/Wub/data/processed \
  --out frontend/public/data/kits/conductor.json --epochs 18
```

## Browser use

`frontend/src/conductor.ts` validates the artifact, standardizes the state,
runs the three dense layers and decodes every head. `StemRemixPlayer` loads it
as the optional `conductor` sidecar and, at each block, calls `runConductor()`
from the current/previous phrase cells. With the shipped
`"authority": 1.0`, the conductor's section and energy **replace** the
hand-written arc; its change head biases which stems move, and its pattern
heads drive per-slot bar choice in `BarComposer`.

The hand-written rules are still the fallback when the sidecar is absent or
invalid, and all hard safety gates remain: ranker thresholds, seam gates,
pins, source cap, key/tempo limits, foundation protection and the final
`SliceFlow` cut. Lowering `authority` in the artifact blends the conductor
back toward the curated rules; changing only that field is the rollback knob.

Listener feedback is not yet part of training. The next iteration should add
A/B or like/skip pairs and fine-tune the policy head with preference loss.
