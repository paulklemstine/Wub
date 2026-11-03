#!/bin/bash
# Wub continuous retraining orchestrator: rebuilds the global slice
# vocabulary, pattern bank, datasets, neural arrangers and static bundle.
# Safe to re-run: every stage checkpoints; stages after a failure can resume
# by re-running (idempotent renders skip existing opus files).
#
# Full run (~1-2h, mostly corpus remap + opus renders). Laptop GPU optional:
# set TRAIN_ON_LAPTOP=1 with LAPTOP=user@host to offload GPT training.
set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET_D=${BUDGET_D:-8000}
BUDGET_B=${BUDGET_B:-6000}
BUDGET_O=${BUDGET_O:-8000}
BUDGET_V=${BUDGET_V:-6000}
KPS=${K_PER_SONG:-24}

echo "=== [1/7] global vocabulary (virtual stems from full corpus) ==="
PYTHONPATH=. python3 -u scripts/build_global_vocabulary.py \
  --include-mix --k-per-song "$KPS" \
  --drums "$BUDGET_D" --bass "$BUDGET_B" --other "$BUDGET_O" --vocals "$BUDGET_V" \
  --bitrate 96k --workers 12

echo "=== [2/7] VQ codebooks ==="
PYTHONPATH=. python3 -u scripts/fit_tatum_vq.py --drums 768 --bass 512 --other 768 --vocals 512

echo "=== [3/7] pattern bank (bar transitions, joint tuples, arcs) ==="
PYTHONPATH=. python3 -u scripts/train_global_patterns.py

echo "=== [4/7] datasets (bar + tatum) ==="
PYTHONPATH=. python3 -u scripts/build_arranger_dataset.py --val-pct 5
PYTHONPATH=. python3 -u scripts/build_tatum_dataset.py

echo "=== [5/7] neural arrangers (bar d128/L2 + tatum d128/L2 + tatum d256/L4) ==="
if [ "${TRAIN_ON_LAPTOP:-0}" = "1" ]; then
  echo "offloading GPT training to $LAPTOP"
  scp data/arranger_dataset.json data/tatum_dataset.json scripts/train_bar_arranger.py "$LAPTOP:/tmp/wubtrain/data/" 2>/dev/null || true
  ssh "$LAPTOP" "cd /tmp/wubtrain && mv -f data/train_bar_arranger.py scripts/ 2>/dev/null; PYTHONPATH=/tmp/wubtrain nohup python3 -u scripts/train_bar_arranger.py --data data/arranger_dataset.json --epochs 25 --batch 64 --d 128 --layers 2 --heads 2 --ckpt data/arranger_gpt.pt --log data/arranger_train_log.json > train_bar.log 2>&1 &"
  ssh "$LAPTOP" "cd /tmp/wubtrain && PYTHONPATH=/tmp/wubtrain nohup python3 -u scripts/train_bar_arranger.py --data data/tatum_dataset.json --epochs 25 --batch 64 --d 128 --layers 2 --heads 2 --ckpt data/tatum128.pt --log data/tatum128_log.json > train_tatum128.log 2>&1 &"
  ssh "$LAPTOP" "cd /tmp/wubtrain && PYTHONPATH=/tmp/wubtrain nohup python3 -u scripts/train_bar_arranger.py --data data/tatum_dataset.json --epochs 25 --batch 64 --ctx 280 --stride 70 --d 256 --layers 4 --heads 4 --ckpt data/tatum256.pt --log data/tatum256_log.json > train_tatum256.log 2>&1 &"
  echo "launched on laptop; poll with: ssh $LAPTOP 'tail -n 3 /tmp/wubtrain/train_*.log'"
else
  PYTHONPATH=. python3 -u scripts/train_bar_arranger.py --data data/arranger_dataset.json --epochs 25 --batch 64 --ckpt data/arranger_gpt.pt --log data/arranger_train_log.json
  PYTHONPATH=. python3 -u scripts/train_bar_arranger.py --data data/tatum_dataset.json --epochs 25 --batch 64 --ctx 280 --stride 70 --ckpt data/tatum_gpt.pt --log data/tatum_train_log.json
  PYTHONPATH=. python3 -u scripts/train_bar_arranger.py --data data/tatum_dataset.json --epochs 25 --batch 64 --ctx 280 --stride 70 --d 256 --layers 4 --heads 4 --ckpt data/tatum256_gpt.pt --log data/tatum256_train_log.json
fi

echo "=== [6/7] musical eval gates ==="
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/eval_bar_arranger.py --ckpt data/arranger_gpt.pt --dataset data/arranger_dataset.json --n 20
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 scripts/eval_tatum.py --ckpt data/tatum_gpt.pt --dataset data/tatum_dataset.json --n 20

echo "=== [7/7] ONNX export + frontend build + tests ==="
PYTHONPATH=. python3 scripts/export_arranger_onnx.py --ckpt data/arranger_gpt.pt --out frontend/public/data/arranger_gpt.onnx --meta-out frontend/public/data/arranger_gpt.json --tokens-per-bar 6
PYTHONPATH=. python3 scripts/export_arranger_onnx.py --ckpt data/tatum_gpt.pt --out frontend/public/data/tatum_gpt.onnx --meta-out frontend/public/data/tatum_gpt.json --tokens-per-bar 70
(cd frontend && npm run build)
pytest -q
echo "ALL GREEN - commit with: git add -A && git commit -m 'feat: retrained ...' && git push origin main"
