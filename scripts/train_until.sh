#!/usr/bin/env bash
# Self-healing training loop: trains toward --epochs, resuming from the
# latest epoch checkpoint after any crash (the Victus WSL box throws an
# opaque CUDA "unknown error" every ~5-8 GPU-hours and kills the run).
# Each iteration resumes from the newest ep checkpoint; per-epoch
# checkpoints (see --checkpoint-every) bound every loss to one epoch.
# Usage:
#   scripts/train_until.sh --model-out data/wubgpt/wubgpt_dens8.pt --epochs 8 \
#       [extra train_wubgpt.py flags...] 2>&1 | tee train_until.log
set -u
MODEL_OUT=""
EPOCHS=8
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model-out) MODEL_OUT="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        *) EXTRA+=("$1"); shift;;
    esac
done
if [ -z "$MODEL_OUT" ]; then echo "usage: $0 --model-out PATH --epochs N [train flags...]" >&2; exit 2; fi
DIR=$(dirname "$MODEL_OUT")
BASE=$(basename "$MODEL_OUT" .pt)
for i in $(seq 1 "$((EPOCHS + 2))"); do
    if [ -f "$MODEL_OUT" ]; then echo "DONE: $MODEL_OUT exists"; break; fi
    LATEST=$(ls -v "$DIR"/"$BASE"_ep*.pt 2>/dev/null | tail -n 1 || true)
    if [ -z "$LATEST" ]; then echo "no epoch checkpoint yet; cannot resume" >&2; exit 1; fi
    echo "=== attempt $i: resuming from $LATEST (target $EPOCHS) ==="
    python3 scripts/train_wubgpt.py --model-out "$MODEL_OUT" --epochs "$EPOCHS" \
        --resume-from "$LATEST" "${EXTRA[@]}" || echo "attempt $i exited $?; continuing"
done
if [ -f "$MODEL_OUT" ]; then echo "DONE: $MODEL_OUT exists"; exit 0; else echo "GAVE UP after $i attempts" >&2; exit 1; fi
