#!/bin/bash
# Full song-policy training: mined stats + neural compatibility weights.
# Runs locally (GTX 1060, CUDA) with data/processed (1725 tracks).
# Logs to /tmp/wub-full-train.log.
set -u
cd /home/raver1975/Wub
LOG=/tmp/wub-full-train.log
echo "[full-train] starting $(date)" | tee -a "$LOG"

echo "[1/6] building taste dataset from analysis JSON" | tee -a "$LOG"
PYTHONPATH=. python3 scripts/build_taste_dataset.py \
  --processed data/processed \
  --kits frontend/public/data/kits/kit_manifest.json \
  --out /tmp/taste_dataset_full.json >>"$LOG" 2>&1
echo "taste_dataset exit: $?" | tee -a "$LOG"

echo "[2/6] training song-policy compatibility weights (GPU)" | tee -a "$LOG"
PYTHONPATH=. python3 scripts/train_song_policy_v2.py \
  --kits-dir frontend/public/data/kits \
  --out frontend/public/data/kits/song_policy.json \
  --epochs 40 >>"$LOG" 2>&1
echo "song_policy exit: $?" | tee -a "$LOG"

echo "[3/6] mining motif calendar + vocal cadence (CPU)" | tee -a "$LOG"
PYTHONPATH=. python3 scripts/build_motif_calendar.py \
  --processed data/processed \
  --out-dir /tmp/motif_out >>"$LOG" 2>&1
echo "motif_calendar exit: $? (outputs to /tmp/motif_out)" | tee -a "$LOG"
ls /tmp/motif_out 2>&1 | tee -a "$LOG"

echo "[4/6] mining transition cohorts (CPU)" | tee -a "$LOG"
PYTHONPATH=. python3 scripts/build_transition_cohort.py \
  --processed data/processed \
  --out /tmp/transition_cohort_full.json >>"$LOG" 2>&1
echo "cohort exit: $?" | tee -a "$LOG"

echo "[5/6] training taste classifier (GPU, 20 epochs)" | tee -a "$LOG"
PYTHONPATH=. python3 scripts/train_taste_model.py \
  --dataset /tmp/taste_dataset_full.json \
  --out /tmp/taste_model_full.pt \
  --log /tmp/taste_train_full_log.json --epochs 20 --device cuda >>"$LOG" 2>&1
echo "taste classifier exit: $?" | tee -a "$LOG"

echo "[6/6] training bank transition (GPU)" | tee -a "$LOG"
PYTHONPATH=. python3 scripts/train_bank_transition.py \
  --kits-dir frontend/public/data/kits \
  --processed data/processed \
  --out /tmp/bank_transition_full.json --epochs 20 >>"$LOG" 2>&1
echo "bank_transition exit: $?" | tee -a "$LOG"

echo "[full-train] done $(date)" | tee -a "$LOG"
