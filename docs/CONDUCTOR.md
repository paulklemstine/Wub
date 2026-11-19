# Learned conductor

The conductor is the first offline-trained model that decides all four musical
things at once instead of scoring one narrow seam:

- **next section** (5 classes),
- **next target energy**,
- **which stems change** at the 4-bar boundary (4 logits),
- **the next 4-bar onset/chroma pattern of each stem** (4 × 48).

## Training data

The shipped model is **v2**, trained by `scripts/train_conductor_v2.py`.
`scripts/train_conductor.py` is the original two-cell prototype and reads the real-song analyses in
`data/processed/*.json`. For each 4-bar boundary it builds a 392-d state from
the previous two 4-bar cells of all four stems (onset grid, density,
downbeats, quarters, root, chroma, activity) plus section/energy context, and
trains a 392→128→128→202 tanh MLP against the *observed* next cell.

v2 uses a six-cell (24-bar) history, a 1202→256→256→202 MLP and 25 epochs.
The v1 data prep accidentally predicted the cell **two** steps ahead; fixing
that and extending the context produced a large gain. Final track-disjoint run:
**1,881 songs / 46,323 examples**, 1,693 / 188 train/val tracks.

| head | v2 | v1 | baseline |
|---|---|---|---|
| section accuracy | **0.637** | 0.549 | 0.550 (majority) |
| section macro F1 | **0.549** | – | – |
| energy MAE | **0.050** | 0.072 | 0.064 (mean) |
| change AUC | drums 0.795 / bass 0.730 / other 0.737 / vocals 0.634 | 0.72–0.79 | 0.5 |
| pattern MSE | **0.0631** | 0.0681 | 0.0869 (mean) |

Class balancing and 40 epochs were also tried and were worse (0.618 section,
0.773 drums AUC), so the shipped recipe is the unweighted 25-epoch run.

Retrain with:

```bash
PYTHONPATH=. python3 scripts/train_conductor_v2.py \
  --processed /mnt/e/Wub/data/processed \
  --out frontend/public/data/kits/conductor.json \
  --epochs 25 --lr 0.0008 --no-balance
```

## Browser use

`frontend/src/conductor.ts` validates the artifact, standardizes the state,
runs the three dense layers and decodes every head. `StemRemixPlayer` loads it
as the optional `conductor` sidecar and, at each block, calls `runConductor()`
from the current/previous phrase cells. With the shipped
`"authority": 0.30`, the conductor is a **bias** on top of the curated arc,
not an arbiter that replaces it; its change head biases which stems move, and its pattern
heads drive per-slot bar choice in `BarComposer`.

The hand-written rules are still the fallback when the sidecar is absent or
invalid, and all hard safety gates remain: ranker thresholds, seam gates,
pins, source cap, key/tempo limits, foundation protection and the final
`SliceFlow` cut. Lowering `authority` in the artifact blends the conductor
back toward the curated rules; changing only that field is the rollback knob.

Listener feedback is not yet part of training. The next iteration should add
A/B or like/skip pairs and fine-tune the policy head with preference loss.
