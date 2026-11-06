#!/usr/bin/env bash
# Re-cut the Wub 4-bar kit bank at 8 phrases per kit and rebuild every model
# artifact that ships with it -- without touching the live frontend bank.
#
# This is the reproducible laptop pipeline for the 4 -> 8 -> 16-bar arc bank:
#
#   corpus (kits_arc8) -> taste dataset -> taste classifier + ranker
#   -> 16-cluster shipping subset -> metadata + pre-rolls + nebula coords
#   -> per-stem log-mel bars -> four stem judges -> taste_scores.json
#   -> optional joint 4-stem ranker -> reconstruction QC -> run coverage
#
# Live-bank separation
# --------------------
#   * The corpus cutter writes only to OUT (default kits_arc8).
#   * Every selected/scored artifact is written below STAGE
#     (default /tmp/wubtrain/arc8_stage).
#   * frontend/public/data/kits is read only if the caller explicitly points a
#     command at it; this script refuses OUT/STAGE paths inside the live bank.
#   * Promotion to the live bank is a separate, reviewed step.  This script
#     never copies into frontend/public/data/kits.
#
# Safety / re-runs
# ----------------
#   * build_kit_corpus.py is idempotent (it reuses complete kit phrases).
#   * If the corpus log already ends in "DONE rc=0" the cutter is skipped.
#   * If another corpus build is running for OUT, the script waits unless
#     WAIT_FOR_CORPUS=0.
#   * TASTE/CLUSTER/BARS/JUDGE/SCORE stages are skipped when their output is
#     newer than its input; FORCE=1 rebuilds everything.
#
# Usage on the laptop (from the repo checkout, normally /tmp/wubtrain):
#
#   bash scripts/recut_arc_bank.sh
#
# Common overrides:
#   PHRASES_PER_KIT=8 KITS=390 OUT=kits_arc8 \
#   STAGE=/tmp/wubtrain/arc8_stage bash scripts/recut_arc_bank.sh
#
#   FORCE=1                 # redo every stage
#   SKIP_CORPUS=1           # use an existing OUT/kit_manifest.json
#   WAIT_FOR_CORPUS=0       # do not wait for an already-running cutter
#   RUN_JOINT_RANKER=0      # skip the optional joint-ranker rebuild
#   RUN_QC=0                # skip kit_reconstruct for a quick smoke run
#   RUN_ADD_STEM_COORDS=1   # use timbre PCA instead of judge stem_xy
#   VERIFY_LIVE=0           # skip the live-manifest hash check
#   ALLOW_KIT_MISMATCH=1    # do not fail when the cluster union != KITS
#   ALLOW_COVERAGE_FAIL=1   # do not fail when <80% of clusters have runs
#
set -euo pipefail

case "${1:-}" in
  -h|--help)
    sed -n '2,$p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  "")
    ;;
  *)
    echo "unknown argument: $1 (use --help; all configuration is via environment variables)" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ROOT="${WUB_ROOT:-/tmp/wubtrain}"
STEMS_DIR="${STEMS_DIR:-/mnt/e/Wub/data/stems}"
PROCESSED_DIR="${PROCESSED_DIR:-$ROOT/processed}"

PHRASES_PER_KIT="${PHRASES_PER_KIT:-8}"
KITS="${KITS:-390}"
OUT="${OUT:-kits_arc8}"
STAGE="${STAGE:-/tmp/wubtrain/arc8_stage}"

BARS="${BARS:-4}"
BITRATE="${BITRATE:-64k}"
WORKERS="${WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-2}"
DEVICE="${DEVICE:-auto}"

# The 16 clusters currently top out around 160 kits in one style.  A 200 cap
# therefore keeps the full 390-kit union while still exercising a real
# style-balanced selection.  Raise it if a future cut has a larger style.
PER_STYLE="${PER_STYLE:-200}"

TASTE_EPOCHS="${TASTE_EPOCHS:-20}"
RANKER_EPOCHS="${RANKER_EPOCHS:-60}"
JUDGE_EPOCHS="${JUDGE_EPOCHS:-12}"
JOINT_EPOCHS="${JOINT_EPOCHS:-30}"
# Opus residuals are codec-limited; 0.95/20 dB is too strict for a 15-20 dB
# codec noise floor. 0.90/10 dB still catches real waveform/alignment errors.
QC_MIN_CORRELATION="${QC_MIN_CORRELATION:-0.90}"
QC_MIN_SNR="${QC_MIN_SNR:-10}"

RUN_JOINT_RANKER="${RUN_JOINT_RANKER:-1}"
RUN_QC="${RUN_QC:-1}"
RUN_ADD_STEM_COORDS="${RUN_ADD_STEM_COORDS:-0}"
VERIFY_LIVE="${VERIFY_LIVE:-1}"
KEEP_CLUSTER="${KEEP_CLUSTER:-0}"
FORCE="${FORCE:-0}"
SKIP_CORPUS="${SKIP_CORPUS:-0}"
WAIT_FOR_CORPUS="${WAIT_FOR_CORPUS:-1}"
ALLOW_KIT_MISMATCH="${ALLOW_KIT_MISMATCH:-0}"
ALLOW_COVERAGE_FAIL="${ALLOW_COVERAGE_FAIL:-0}"

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Paths and guards
# ---------------------------------------------------------------------------
[[ -d "$ROOT" ]] || { echo "WUB_ROOT does not exist: $ROOT" >&2; exit 2; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s/%s\n' "$ROOT" "$1" ;;
  esac
}

OUT_ABS="$(readlink -m "$(abs_path "$OUT")")"
STAGE_ABS="$(readlink -m "$(abs_path "$STAGE")")"
LIVE_ABS="$(readlink -m "$ROOT/frontend/public/data/kits")"
FLOW_ABS="$(readlink -m "$ROOT/kits_flow")"

case "$OUT_ABS/" in
  "$LIVE_ABS"/*) echo "refusing: OUT points into the live frontend bank ($LIVE_ABS)" >&2; exit 2 ;;
  "$FLOW_ABS"/*) echo "refusing: OUT points into kits_flow ($FLOW_ABS)" >&2; exit 2 ;;
esac
case "$STAGE_ABS/" in
  "$LIVE_ABS"/*) echo "refusing: STAGE points into the live frontend bank ($LIVE_ABS)" >&2; exit 2 ;;
  "$FLOW_ABS"/*) echo "refusing: STAGE points into kits_flow ($FLOW_ABS)" >&2; exit 2 ;;
esac

LOG_DIR="${LOG_DIR:-$ROOT}"
LOG_DIR_ABS="$(abs_path "$LOG_DIR")"
mkdir -p "$LOG_DIR_ABS" "$STAGE_ABS"

MAIN_LOG="$LOG_DIR_ABS/arc8_pipeline.log"
CORPUS_LOG="$LOG_DIR_ABS/arc8_corpus.log"
TASTE_LOG="$LOG_DIR_ABS/arc8_taste_dataset.log"
TRAIN_LOG="$LOG_DIR_ABS/arc8_train_taste.log"
SELECT_LOG="$LOG_DIR_ABS/arc8_select.log"
META_LOG="$LOG_DIR_ABS/arc8_metadata.log"
PREROLL_LOG="$LOG_DIR_ABS/arc8_preroll.log"
COORD_LOG="$LOG_DIR_ABS/arc8_coords.log"
STEMBARS_LOG="$LOG_DIR_ABS/arc8_stem_bars.log"
JUDGE_LOG="$LOG_DIR_ABS/arc8_train_judge.log"
SCORE_LOG="$LOG_DIR_ABS/arc8_score.log"
JOINT_LOG="$LOG_DIR_ABS/arc8_joint_ranker.log"
QC_LOG="$LOG_DIR_ABS/arc8_qc.log"
COVERAGE_LOG="$LOG_DIR_ABS/arc8_coverage.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MAIN_LOG"
}

run_logged() {
  local log_path="$1"; shift
  log "RUN: $*"
  if ! "$@" >>"$log_path" 2>&1; then
    log "FAILED -- see $log_path"
    return 1
  fi
}

json_count() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(len(data.get("kits", [])))
PY
}

phrase_count() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(sum(len(k.get("phrases", [])) for k in data.get("kits", [])))
PY
}

bars_complete() {  # bars_complete BARS_DIR MANIFEST
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
with manifest_path.open("r", encoding="utf-8") as fh:
    manifest = json.load(fh)
expected_tracks = {str(k.get("track_id")) for k in manifest.get("kits", [])}
expected_phrases = {
    str(p.get("id")) for k in manifest.get("kits", []) for p in k.get("phrases", [])
}
for stem in ("drums", "bass", "other", "vocals"):
    index = root / stem / "bar_index.json"
    mel = root / stem / "mel.npy"
    if not index.is_file() or not mel.is_file():
        raise SystemExit(1)
    with index.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if len(data.get("phrase_id", [])) != len(expected_phrases):
        raise SystemExit(1)
    if {str(x) for x in data.get("phrase_id", [])} != expected_phrases:
        raise SystemExit(1)
    if {str(x) for x in data.get("track_id", [])} != expected_tracks:
        raise SystemExit(1)
PY
}

restore_mtime_if_same() {  # restore_mtime_if_same BEFORE TARGET
  local before="$1" target="$2"
  if cmp -s "$before" "$target"; then
    # Keep the old timestamp so downstream "is this newer?" checks do not
    # retrain/rebuild when a manifest rewrite was content-identical.
    touch -r "$before" "$target"
  fi
  rm -f "$before"
}

run_manifest_step() {  # run_manifest_step MANIFEST LOG COMMAND...
  local manifest="$1" log_path="$2"; shift 2
  local before="$STAGE_ABS/.manifest_before.json"
  cp -p "$manifest" "$before"
  if ! run_logged "$log_path" "$@"; then
    rm -f "$before"
    return 1
  fi
  restore_mtime_if_same "$before" "$manifest"
}

is_newer() {  # is_newer OUTPUT INPUT...
  local out="$1"; shift
  [[ -s "$out" ]] || return 1
  [[ "$FORCE" == "1" ]] && return 1
  local input
  for input in "$@"; do
    [[ -e "$input" ]] || continue
    if [[ "$input" -nt "$out" ]]; then
      return 1
    fi
  done
  return 0
}

same_track_set() {  # same_track_set MANIFEST_A MANIFEST_B
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
def ids(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k.get("track_id")) for k in data.get("kits", [])}
a, b = ids(sys.argv[1]), ids(sys.argv[2])
raise SystemExit(0 if a == b else 1)
PY
}

scores_match_bank() {  # scores_match_bank SCORES_JSON MANIFEST_JSON
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    scores = json.load(fh).get("kits", {})
with open(sys.argv[2], "r", encoding="utf-8") as fh:
    manifest = json.load(fh).get("kits", [])
expected = {str(k.get("track_id")): len(k.get("phrases", [])) for k in manifest}
actual = {str(tid): len(row.get("transition", [])) for tid, row in scores.items()}
raise SystemExit(0 if actual == expected else 1)
PY
}

# One orchestrator at a time.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOG_DIR_ABS/arc8_pipeline.lock"
  if ! flock -n 9; then
    echo "another recut_arc_bank.sh is already running (lock $LOG_DIR_ABS/arc8_pipeline.lock)" >&2
    exit 3
  fi
fi

log "=== Wub 8-phrase arc bank recut ==="
log "root=$ROOT out=$OUT stage=$STAGE kits=$KITS phrases/kit=$PHRASES_PER_KIT"

LIVE_MANIFEST="$LIVE_ABS/kit_manifest.json"
LIVE_HASH_BEFORE=""
if [[ "$VERIFY_LIVE" == "1" && -f "$LIVE_MANIFEST" ]]; then
  LIVE_HASH_BEFORE="$(sha256sum "$LIVE_MANIFEST" | awk '{print $1}')"
  log "live manifest baseline: $LIVE_HASH_BEFORE"
fi

for required in "$PROCESSED_DIR" "$STEMS_DIR" \
                "$ROOT/scripts/build_kit_corpus.py" \
                "$ROOT/scripts/build_taste_dataset.py" \
                "$ROOT/scripts/train_taste_model.py" \
                "$ROOT/scripts/select_cluster_bank.py" \
                "$ROOT/scripts/select_kit_bank.py" \
                "$ROOT/scripts/score_kits.py" \
                "$ROOT/scripts/score_stem_kits.py" \
                "$ROOT/scripts/build_stem_bars.py" \
                "$ROOT/scripts/train_stem_judge.py" \
                "$ROOT/scripts/add_pre_roll.py" \
                "$ROOT/scripts/link_pre_roll.py" \
                "$ROOT/scripts/merge_kit_metadata.py" \
                "$ROOT/scripts/add_kit_coords.py" \
                "$ROOT/scripts/add_stem_coords.py" \
                "$ROOT/scripts/kit_reconstruct.py"; do
  [[ -e "$required" ]] || { log "missing required path: $required"; exit 2; }
done

if [[ "$DEVICE" == "auto" ]]; then
  if "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi
log "device=$DEVICE workers=$WORKERS"

# ---------------------------------------------------------------------------
# 1. Corpus cutter -- fresh 8-phrase kits, never kits_flow/live
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ABS"

if [[ "$SKIP_CORPUS" != "1" ]]; then
  if [[ -f "$LOG_DIR_ABS/arc8_corpus.pid" ]]; then
    corpus_pid="$(cat "$LOG_DIR_ABS/arc8_corpus.pid" 2>/dev/null || true)"
    if [[ -n "$corpus_pid" ]] && kill -0 "$corpus_pid" 2>/dev/null; then
      if [[ "$WAIT_FOR_CORPUS" == "1" ]]; then
        log "corpus build already running (PID $corpus_pid); waiting..."
        while kill -0 "$corpus_pid" 2>/dev/null; do sleep 60; done
      else
        log "corpus build already running (PID $corpus_pid); WAIT_FOR_CORPUS=0 so stopping"
        exit 3
      fi
    fi
  fi

  if [[ -s "$OUT_ABS/kit_manifest.json" ]] && \
     grep -Eq '^DONE( rc=0)?$' "$CORPUS_LOG" 2>/dev/null; then
    log "corpus already complete: $OUT_ABS/kit_manifest.json (skip cutter)"
  else
    log "cutting corpus -> $OUT_ABS (log $CORPUS_LOG)"
    if ! run_logged "$CORPUS_LOG" "$PYTHON" -u scripts/build_kit_corpus.py \
        --processed "$PROCESSED_DIR" \
        --stems "$STEMS_DIR" \
        --out "$OUT_ABS" \
        --phrases-per-kit "$PHRASES_PER_KIT" \
        --bars "$BARS" \
        --bitrate "$BITRATE" \
        --workers "$WORKERS"; then
      printf 'DONE rc=1\n' >> "$CORPUS_LOG"
      exit 1
    fi
    printf 'DONE rc=0\n' >> "$CORPUS_LOG"
  fi
fi

[[ -s "$OUT_ABS/kit_manifest.json" ]] || { log "missing $OUT_ABS/kit_manifest.json"; exit 2; }
CORPUS_KITS="$(json_count "$OUT_ABS/kit_manifest.json")"
log "corpus manifest: $CORPUS_KITS kits"
[[ "$CORPUS_KITS" -gt 0 ]] || { log "corpus manifest has no kits"; exit 2; }

# ---------------------------------------------------------------------------
# 2. Taste dataset for the new manifest
# ---------------------------------------------------------------------------
TASTE_DATASET="$STAGE_ABS/taste_dataset.json"
if is_newer "$TASTE_DATASET" "$OUT_ABS/kit_manifest.json"; then
  log "taste dataset is current (skip)"
else
  log "building taste dataset (log $TASTE_LOG)"
  run_logged "$TASTE_LOG" "$PYTHON" -u scripts/build_taste_dataset.py \
    --processed "$PROCESSED_DIR" \
    --kits "$OUT_ABS/kit_manifest.json" \
    --out "$TASTE_DATASET" \
    --workers "$WORKERS"
fi

# ---------------------------------------------------------------------------
# 3. Taste classifier + pairwise ranker
# ---------------------------------------------------------------------------
TASTE_MODEL="$STAGE_ABS/taste_model.pt"
TASTE_RANKER="$STAGE_ABS/taste_ranker.pt"
if is_newer "$TASTE_MODEL" "$TASTE_DATASET"; then
  log "taste classifier is current (skip)"
else
  log "training taste classifier (log $TRAIN_LOG)"
  run_logged "$TRAIN_LOG" "$PYTHON" -u scripts/train_taste_model.py \
    --dataset "$TASTE_DATASET" \
    --out "$TASTE_MODEL" \
    --log "$STAGE_ABS/taste_train_log.json" \
    --epochs "$TASTE_EPOCHS" \
    --device "$DEVICE"
fi

if is_newer "$TASTE_RANKER" "$TASTE_DATASET" "$TASTE_MODEL"; then
  log "taste ranker is current (skip)"
else
  log "training pairwise taste ranker (log $TRAIN_LOG)"
  run_logged "$TRAIN_LOG" "$PYTHON" -u scripts/train_taste_model.py \
    --dataset "$TASTE_DATASET" \
    --train-ranker \
    --classifier-ckpt "$TASTE_MODEL" \
    --ranker-out "$TASTE_RANKER" \
    --ranker-log "$STAGE_ABS/taste_ranker_train_log.json" \
    --ranker-epochs "$RANKER_EPOCHS" \
    --ranker-context-len 6 \
    --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# 4. 16-cluster bank, then a style-balanced shipping subset
# ---------------------------------------------------------------------------
CLUSTER_DIR="$STAGE_ABS/kits_cluster"
BANK_DIR="$STAGE_ABS/kits_bank"
CLUSTER_MANIFEST="$CLUSTER_DIR/kit_manifest.json"
BANK_MANIFEST="$BANK_DIR/kit_manifest.json"

# KEEP_CLUSTER=0 deletes the intermediate copy after the final bank is made.
# On a later re-run, reuse the final bank as its own cluster source instead of
# trying to re-copy and re-select.
if [[ ! -s "$CLUSTER_MANIFEST" && -s "$BANK_MANIFEST" ]]; then
  log "cluster intermediate absent; validating the existing bank in place"
  CLUSTER_DIR="$BANK_DIR"
  CLUSTER_MANIFEST="$BANK_MANIFEST"
fi

if is_newer "$CLUSTER_MANIFEST" "$OUT_ABS/kit_manifest.json"; then
  log "cluster bank is current (skip)"
else
  log "selecting 16-cluster bank (log $SELECT_LOG)"
  rm -rf "$CLUSTER_DIR"
  run_logged "$SELECT_LOG" "$PYTHON" -u scripts/select_cluster_bank.py \
    --manifest "$OUT_ABS/kit_manifest.json" \
    --source-dir "$OUT_ABS" \
    --out "$CLUSTER_DIR" \
    --workers "$WORKERS"
fi

CLUSTER_KITS="$(json_count "$CLUSTER_MANIFEST")"
log "cluster union: $CLUSTER_KITS kits (target $KITS)"
if [[ "$CLUSTER_KITS" -ne "$KITS" && "$ALLOW_KIT_MISMATCH" != "1" ]]; then
  log "ERROR: cluster union is $CLUSTER_KITS, expected $KITS."
  log "       Set ALLOW_KIT_MISMATCH=1 only after inspecting the pool logs."
  exit 2
fi

if is_newer "$BANK_MANIFEST" "$CLUSTER_MANIFEST"; then
  log "shipping subset is current (skip)"
else
  log "style-balanced shipping subset (per-style=$PER_STYLE; log $SELECT_LOG)"
  if [[ "$BANK_DIR" != "$CLUSTER_DIR" ]]; then
    rm -rf "$BANK_DIR"
  fi
  run_logged "$SELECT_LOG" "$PYTHON" -u scripts/select_kit_bank.py \
    --manifest "$CLUSTER_MANIFEST" \
    --source-dir "$CLUSTER_DIR" \
    --out "$BANK_DIR" \
    --per-style "$PER_STYLE"
fi

BANK_KITS="$(json_count "$BANK_MANIFEST")"
log "shipping subset: $BANK_KITS kits"
if [[ "$BANK_KITS" -ne "$CLUSTER_KITS" ]]; then
  log "ERROR: select_kit_bank dropped kits ($BANK_KITS < $CLUSTER_KITS)."
  log "       The cluster manifest would reference missing kits; raise PER_STYLE."
  exit 2
fi
if ! same_track_set "$BANK_MANIFEST" "$CLUSTER_MANIFEST"; then
  log "ERROR: shipping subset track set differs from the cluster union."
  exit 2
fi

if [[ "$KEEP_CLUSTER" != "1" && "$CLUSTER_DIR" != "$BANK_DIR" ]]; then
  # The final bank is a complete copy; the cluster intermediate is no longer
  # needed and would double the staging footprint.
  rm -rf "$CLUSTER_DIR"
  log "removed intermediate cluster copy (KEEP_CLUSTER=0)"
fi

# ---------------------------------------------------------------------------
# 5. Player metadata, pre-bar tails, and nebula coordinates
# ---------------------------------------------------------------------------
log "merging chroma/bass/once metadata (log $META_LOG)"
run_manifest_step "$BANK_MANIFEST" "$META_LOG" \
  "$PYTHON" -u scripts/merge_kit_metadata.py \
  --manifest "$BANK_MANIFEST" \
  --dataset "$TASTE_DATASET" \
  --out "$BANK_MANIFEST"

log "extracting 80 ms pre-bar tails (log $PREROLL_LOG)"
run_manifest_step "$BANK_MANIFEST" "$PREROLL_LOG" \
  "$PYTHON" -u scripts/add_pre_roll.py \
  --manifest "$BANK_MANIFEST" \
  --stems "$STEMS_DIR" \
  --kits-dir "$BANK_DIR" \
  --workers "$WORKERS"
run_manifest_step "$BANK_MANIFEST" "$PREROLL_LOG" \
  "$PYTHON" -u scripts/link_pre_roll.py \
  --manifest "$BANK_MANIFEST" \
  --kits-dir "$BANK_DIR"

# ---------------------------------------------------------------------------
# 6. Per-stem bars and four transition judges
# ---------------------------------------------------------------------------
STEM_BARS="$STAGE_ABS/stem_bars"
BANK_PHRASES="$(phrase_count "$BANK_MANIFEST")"
if [[ "$FORCE" != "1" ]] && bars_complete "$STEM_BARS" "$BANK_MANIFEST"; then
  log "per-stem bars are current ($BANK_PHRASES phrases/stem; skip)"
else
  log "building per-stem log-mel bars (log $STEMBARS_LOG)"
  run_logged "$STEMBARS_LOG" "$PYTHON" -u scripts/build_stem_bars.py \
    --manifest "$BANK_MANIFEST" \
    --kits-dir "$BANK_DIR" \
    --out "$STEM_BARS" \
    --workers "$WORKERS"
fi

for stem in drums bass other vocals; do
  ckpt="$STAGE_ABS/stem_judge_${stem}.pt"
  if is_newer "$ckpt" "$STEM_BARS/$stem/mel.npy"; then
    log "stem judge $stem is current (skip)"
  else
    log "training stem judge $stem (log $JUDGE_LOG)"
    run_logged "$JUDGE_LOG" "$PYTHON" -u scripts/train_stem_judge.py \
      --bars "$STEM_BARS" \
      --stem "$stem" \
      --out "$ckpt" \
      --log "$STAGE_ABS/stem_judge_${stem}_log.json" \
      --epochs "$JUDGE_EPOCHS" \
      --workers "$JUDGE_WORKERS" \
      --device "$DEVICE"
  fi
done

# ---------------------------------------------------------------------------
# 7. Kit-level and per-stem scores
# ---------------------------------------------------------------------------
KIT_SCORES="$STAGE_ABS/taste_scores_kits.json"
STEM_SCORES="$STAGE_ABS/taste_scores_stems.json"
FINAL_SCORES="$BANK_DIR/taste_scores.json"

if is_newer "$KIT_SCORES" "$STEM_BARS/drums/mel.npy" "$TASTE_DATASET" \
      "$TASTE_MODEL" "$TASTE_RANKER" && \
   scores_match_bank "$KIT_SCORES" "$BANK_MANIFEST"; then
  log "kit-level taste scores are current (skip)"
else
  log "scoring kit-level transitions (log $SCORE_LOG)"
  run_logged "$SCORE_LOG" "$PYTHON" -u scripts/score_kits.py \
    --dataset "$TASTE_DATASET" \
    --manifest "$BANK_MANIFEST" \
    --model "$TASTE_MODEL" \
    --ranker "$TASTE_RANKER" \
    --out "$KIT_SCORES" \
    --report "$STAGE_ABS/taste_report.json" \
    --device "$DEVICE"
fi

stem_ready=1
stem_inputs=("$STEM_BARS/drums/mel.npy")
for stem in drums bass other vocals; do
  [[ -s "$STAGE_ABS/stem_judge_${stem}.pt" ]] || stem_ready=0
  stem_inputs+=("$STAGE_ABS/stem_judge_${stem}.pt")
done
if [[ "$stem_ready" == "1" ]] && is_newer "$STEM_SCORES" "${stem_inputs[@]}"; then
  log "per-stem scores are current (skip)"
else
  log "scoring per-stem transitions and stem_xy (log $SCORE_LOG)"
  # --write-coords replaces the timbre PCA with the judge-embedding UMAP/PCA
  # directly in the staged manifest.
  run_manifest_step "$BANK_MANIFEST" "$SCORE_LOG" \
    "$PYTHON" -u scripts/score_stem_kits.py \
    --bars "$STEM_BARS" \
    --manifest "$BANK_MANIFEST" \
    --models "$STAGE_ABS/stem_judge" \
    --out "$STEM_SCORES" \
    --topk 24 \
    --write-coords \
    --device "$DEVICE"
fi

if [[ "$RUN_ADD_STEM_COORDS" == "1" ]]; then
  log "replacing judge stem_xy with timbre PCA via add_stem_coords (log $COORD_LOG)"
  run_manifest_step "$BANK_MANIFEST" "$COORD_LOG" \
    "$PYTHON" -u scripts/add_stem_coords.py \
    --manifest "$BANK_MANIFEST" \
    --kits-dir "$BANK_DIR" \
    --workers "$WORKERS" \
    --out "$BANK_MANIFEST"
fi

log "merging kit and stem scores -> $FINAL_SCORES"
"$PYTHON" - "$KIT_SCORES" "$STEM_SCORES" "$FINAL_SCORES" <<'PY'
import json
import sys

kit_path, stem_path, out_path = sys.argv[1:4]
with open(kit_path, "r", encoding="utf-8") as fh:
    kit_scores = json.load(fh)
with open(stem_path, "r", encoding="utf-8") as fh:
    stem_scores = json.load(fh)

merged = dict(kit_scores)
merged["version"] = 2
merged["stem_next"] = stem_scores.get("stem_next", {})
merged["stem_model"] = stem_scores.get("model", "stem_audio_judge_v1")
merged["stem_val_auc"] = stem_scores.get("val_auc")
merged["note"] = stem_scores.get(
    "note", "four per-stem two-tower CNNs; scores are sigmoid probabilities"
)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(merged, fh, separators=(",", ":"))
print(f"wrote {out_path}: kits={len(merged.get('kits', {}))} "
      f"stem_next={len(merged.get('stem_next', {}))}")
PY

# add_kit_coords.py has no CLI: it always writes
# <repo>/frontend/public/data/kits/kit_manifest.json.  Run an unmodified copy
# inside a throw-away root so the live manifest is never touched.  Keep this
# after the score pass: score_stem_kits writes a compact manifest, and running
# the same pretty-print twice makes the second pass byte-identical.
COORD_TMP="$STAGE_ABS/_add_kit_coords"
COORD_BEFORE="$STAGE_ABS/.manifest_before_coords.json"
rm -rf "$COORD_TMP"
mkdir -p "$COORD_TMP/scripts" "$COORD_TMP/frontend/public/data/kits"
cp "$ROOT/scripts/add_kit_coords.py" "$COORD_TMP/scripts/add_kit_coords.py"
cp "$BANK_MANIFEST" "$COORD_TMP/frontend/public/data/kits/kit_manifest.json"
cp -p "$BANK_MANIFEST" "$COORD_BEFORE"
log "projecting kit nebula coordinates in a staging sandbox (log $COORD_LOG)"
run_logged "$COORD_LOG" "$PYTHON" -u "$COORD_TMP/scripts/add_kit_coords.py"
cp "$COORD_TMP/frontend/public/data/kits/kit_manifest.json" "$BANK_MANIFEST"
restore_mtime_if_same "$COORD_BEFORE" "$BANK_MANIFEST"
rm -rf "$COORD_TMP"

# ---------------------------------------------------------------------------
# 8. Optional joint four-stem ranker (requires the mix judge)
# ---------------------------------------------------------------------------
if [[ "$RUN_JOINT_RANKER" == "1" ]]; then
  MIX_JUDGE="$ROOT/data/audio_judge.pt"
  JOINT_DIR="$STAGE_ABS/joint_ranker"
  if [[ -s "$MIX_JUDGE" ]]; then
    if is_newer "$JOINT_DIR/joint_ranker.json" "$BANK_MANIFEST" "$STEM_BARS/drums/mel.npy"; then
      log "joint ranker is current (skip)"
    else
      log "training joint 4-stem ranker (log $JOINT_LOG)"
      run_logged "$JOINT_LOG" "$PYTHON" -u scripts/build_joint_ranker.py \
        --manifest "$BANK_MANIFEST" \
        --dataset "$TASTE_DATASET" \
        --bars "$STEM_BARS" \
        --mix-judge "$MIX_JUDGE" \
        --encoders "$STAGE_ABS/stem_judge" \
        --out "$JOINT_DIR" \
        --epochs "$JOINT_EPOCHS" \
        --device "$DEVICE"
    fi
  else
    log "joint ranker requested but $MIX_JUDGE is missing; skipping"
  fi
else
  log "joint ranker disabled (RUN_JOINT_RANKER=0)"
fi

# ---------------------------------------------------------------------------
# 9. QC: reconstruction compare + run coverage
# ---------------------------------------------------------------------------
QC_DIR="$STAGE_ABS/qc"
mkdir -p "$QC_DIR"
QC_REPORT="$QC_DIR/kit_qc.json"

if [[ "$RUN_QC" == "1" ]]; then
  if is_newer "$QC_REPORT" "$BANK_MANIFEST"; then
    log "reconstruction QC is current (skip)"
  else
    QC_IDS="$("$PYTHON" - "$BANK_MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(",".join(str(k["track_id"]) for k in data.get("kits", [])))
PY
)"
    log "running reconstruction QC on all bank kits (log $QC_LOG)"
    run_logged "$QC_LOG" "$PYTHON" -u scripts/kit_reconstruct.py \
      --mode compare \
      --manifest "$BANK_MANIFEST" \
      --kits-dir "$BANK_DIR" \
      --stems "$STEMS_DIR" \
      --processed "$PROCESSED_DIR" \
      --track-id "$QC_IDS" \
      --out "$QC_DIR/recon" \
      --report "$QC_REPORT" \
      --min-correlation "$QC_MIN_CORRELATION" \
      --min-snr "$QC_MIN_SNR" \
      --fail-on-error
  fi
else
  log "reconstruction QC disabled (RUN_QC=0)"
fi

COVERAGE_JSON="$QC_DIR/run_coverage.json"
log "computing 4/8/16/32-bar run coverage (log $COVERAGE_LOG)"
"$PYTHON" - "$BANK_MANIFEST" "$COVERAGE_JSON" >"$COVERAGE_LOG" 2>&1 <<'PY'
import collections
import json
import sys

manifest_path, out_path = sys.argv[1:3]
with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)
kits = manifest.get("kits", [])
clusters = manifest.get("clusters", [])
stems = ("drums", "bass", "other", "vocals")
thresholds = (1, 2, 4, 8)  # phrases -> 4, 8, 16, 32 bars


def longest_run(phrases, stem):
    bars = sorted({
        int(p["start_bar"])
        for p in phrases
        if isinstance((p.get("stems") or {}).get(stem), dict)
        and (p["stems"][stem].get("file"))
    })
    if not bars:
        return 0
    best = current = 1
    for a, b in zip(bars, bars[1:]):
        current = current + 1 if b == a + 4 else 1
        best = max(best, current)
    return best


per_stem = {stem: {n: 0 for n in thresholds} for stem in stems}
kit_runs = {}
for kit in kits:
    phrases = kit.get("phrases", [])
    row = {}
    for stem in stems:
        run = longest_run(phrases, stem)
        row[stem] = run
        for n in thresholds:
            if run >= n:
                per_stem[stem][n] += 1
    kit_runs[str(kit["track_id"])] = row

cluster_rows = []
for cluster in clusters:
    ids = [str(t) for t in cluster.get("kits", [])]
    row = {"id": cluster.get("id"), "kits": len(ids), "stems": {}}
    for stem in stems:
        best = max((kit_runs.get(t, {}).get(stem, 0) for t in ids), default=0)
        full = sum(1 for t in ids if kit_runs.get(t, {}).get(stem, 0) >= 4)
        row["stems"][stem] = {"max_run_phrases": best,
                              "kits_with_full_4": full}
    cluster_rows.append(row)

cluster_full = {
    stem: sum(1 for row in cluster_rows if row["stems"][stem]["max_run_phrases"] >= 4)
    for stem in stems
}
cluster_frac = {
    stem: (cluster_full[stem] / len(cluster_rows)) if cluster_rows else 0.0
    for stem in stems
}

summary = {
    "kits": len(kits),
    "phrases": sum(len(k.get("phrases", [])) for k in kits),
    "avg_phrases_per_kit": round(
        sum(len(k.get("phrases", [])) for k in kits) / max(1, len(kits)), 3
    ),
    "stems": {stem: {"kits_with_run": per_stem[stem]} for stem in stems},
    "clusters": len(cluster_rows),
    "clusters_with_full_4_phrase_run": cluster_full,
    "cluster_coverage": {stem: round(cluster_frac[stem], 4) for stem in stems},
    "cluster_detail": cluster_rows,
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print(f"kits={summary['kits']} phrases={summary['phrases']} "
      f"avg={summary['avg_phrases_per_kit']}")
for stem in stems:
    counts = per_stem[stem]
    print(f"{stem:6s} kits with >=1/2/4/8 phrase runs "
          f"(4/8/16/32 bars): {counts[1]}/{counts[2]}/{counts[4]}/{counts[8]}")
print(f"clusters with a full 4-phrase (16-bar) run per stem:")
for stem in stems:
    print(f"  {stem:6s} {cluster_full[stem]}/{len(cluster_rows)} "
          f"({cluster_frac[stem]:.1%})")
PY
cat "$COVERAGE_LOG" | tee -a "$MAIN_LOG"

COVERAGE_OK="$("$PYTHON" - "$COVERAGE_JSON" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)
fracs = data.get("cluster_coverage", {})
print("1" if fracs and all(v >= 0.80 for v in fracs.values()) else "0")
PY
)"
if [[ "$COVERAGE_OK" != "1" && "$ALLOW_COVERAGE_FAIL" != "1" ]]; then
  log "ERROR: run-coverage gate failed (<80% of clusters have a full 4-phrase run for every stem)."
  log "       Inspect $COVERAGE_JSON; set ALLOW_COVERAGE_FAIL=1 only for a deliberate exception."
  exit 4
fi

# ---------------------------------------------------------------------------
# 10. Size report
# ---------------------------------------------------------------------------
log "=== size ==="
du -sh "$BANK_DIR" | tee -a "$MAIN_LOG"
du -sh "$BANK_DIR/kits" | tee -a "$MAIN_LOG"
du -sh "$STAGE_ABS" | tee -a "$MAIN_LOG"
BANK_MB="$(du -sm "$BANK_DIR" | awk '{print $1}')"
if [[ "$BANK_MB" -ge 1024 ]]; then
  log "WARNING: staging bank is ${BANK_MB} MiB, over the 1 GiB Pages budget"
else
  log "staging bank is ${BANK_MB} MiB (under the 1 GiB Pages budget)"
fi

if [[ -n "$LIVE_HASH_BEFORE" ]]; then
  LIVE_HASH_AFTER="$(sha256sum "$LIVE_MANIFEST" | awk '{print $1}')"
  if [[ "$LIVE_HASH_AFTER" != "$LIVE_HASH_BEFORE" ]]; then
    log "ERROR: the live frontend manifest changed during the run:"
    log "       before=$LIVE_HASH_BEFORE after=$LIVE_HASH_AFTER"
    log "       Inspect $LIVE_MANIFEST; staging output is still intact."
    exit 5
  fi
  log "live manifest unchanged ($LIVE_HASH_AFTER)"
fi

log "=== DONE ==="
log "bank:    $BANK_DIR"
log "scores:  $FINAL_SCORES"
log "qc:      $QC_REPORT"
log "coverage:$COVERAGE_JSON"
log "logs:    $LOG_DIR_ABS/arc8_*.log"
