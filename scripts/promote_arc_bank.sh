#!/usr/bin/env bash
# Promote the staged 8-phrase arc bank into the live frontend bank.
#
# The recut pipeline stops at $STAGE.  This helper validates the staging
# artifacts and, only with APPLY=1, copies them into frontend/public/data/kits.
# It never commits; review `git diff` and listen before committing.
#
#   # dry run: validate and print sizes only
#   bash scripts/promote_arc_bank.sh
#
#   # apply after reviewing the staging QC
#   APPLY=1 bash scripts/promote_arc_bank.sh
#
# Expected staging layout (see docs/ARC_BANK_RECUT.md):
#   $STAGE/kits_bank/kit_manifest.json
#   $STAGE/kits_bank/kits/
#   $STAGE/kits_bank/taste_scores.json
#   $STAGE/joint_ranker/joint_ranker.json
#   $STAGE/joint_ranker/joint_features.f32
#   $STAGE/qc/kit_qc.json
#   $STAGE/qc/run_coverage.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
STAGE="${STAGE:-/tmp/wubtrain/arc8_stage}"
APPLY="${APPLY:-0}"
KITS="${KITS:-0}"  # 0 = accept the filtered manifest count
MIN_CLUSTER_COVERAGE="${MIN_CLUSTER_COVERAGE:-0.80}"
MAX_BANK_MIB="${MAX_BANK_MIB:-860}"

BANK="$STAGE/kits_bank"
JOINT="$STAGE/joint_ranker"
QC="$STAGE/qc"
LIVE="$ROOT/frontend/public/data/kits"
PYTHON="${PYTHON:-python3}"

fail() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[promote] $*"; }

for path in \
  "$BANK/kit_manifest.json" \
  "$BANK/taste_scores.json" \
  "$BANK/kits" \
  "$JOINT/joint_ranker.json" \
  "$JOINT/joint_features.f32" \
  "$QC/kit_qc.json" \
  "$QC/run_coverage.json"; do
  [[ -e "$path" ]] || fail "missing staging artifact: $path"
done

log "stage=$STAGE"
log "live=$LIVE"
log "mode=$([[ "$APPLY" == "1" ]] && echo APPLY || echo DRY-RUN)"

# 1. Manifest / taste / joint / QC validation.
"$PYTHON" - "$BANK" "$JOINT" "$QC" "$KITS" "$MIN_CLUSTER_COVERAGE" <<'PY'
import json
import os
import sys

bank, joint, qc, kits_target, min_coverage = sys.argv[1:6]
kits_target = int(kits_target)
min_coverage = float(min_coverage)

def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

manifest = load(os.path.join(bank, "kit_manifest.json"))
clusters = manifest.get("clusters") or []
kits = manifest.get("kits") or []
phrases = sum(len(k.get("phrases") or []) for k in kits)
if kits_target > 0 and len(kits) != kits_target:
    raise SystemExit(f"manifest has {len(kits)} kits, expected {kits_target}")
if len(kits) < 300:
    raise SystemExit(f"manifest has only {len(kits)} kits")
if len(clusters) != 16:
    raise SystemExit(f"manifest has {len(clusters)} clusters, expected 16")
if phrases < 2500:
    raise SystemExit(f"manifest has only {phrases} phrases")
for kit in kits:
    for phrase in kit.get("phrases") or []:
        for stem in ("drums", "bass", "other", "vocals"):
            asset = (phrase.get("stems") or {}).get(stem) or {}
            if not asset.get("file"):
                raise SystemExit(f"missing {stem} file on {phrase.get('id')}")

taste = load(os.path.join(bank, "taste_scores.json"))
if not isinstance(taste.get("stem_next"), dict) or not taste["stem_next"]:
    raise SystemExit("taste_scores.json is missing stem_next")
if not isinstance(taste.get("kits"), dict) or not taste["kits"]:
    raise SystemExit("taste_scores.json is missing kit scores")

model = load(os.path.join(joint, "joint_ranker.json"))
if model.get("input_dim") != 2017 or model.get("state_dim") != 400:
    raise SystemExit(f"unexpected joint model dims: {model.get('input_dim')}/{model.get('state_dim')}")
if model.get("feature_dim") != 400:
    raise SystemExit("joint feature_dim is not 400")
if not isinstance(model.get("thresholds"), dict):
    raise SystemExit("joint model has no per-block thresholds")
feature_path = os.path.join(joint, "joint_features.f32")
expected = len(model.get("phrase_ids") or []) * int(model["feature_dim"]) * 4
actual = os.path.getsize(feature_path)
if actual != expected:
    raise SystemExit(f"joint_features.f32 is {actual} bytes, expected {expected}")

qc_report = load(os.path.join(qc, "kit_qc.json"))
qc_kits = qc_report.get("kits") or []
if len(qc_kits) < 300:
    raise SystemExit(f"QC report covers only {len(qc_kits)} kits")
failed_tracks = {str(f.get("track_id")) for f in (qc_report.get("failures") or []) if f.get("track_id")}
manifest_tracks = {str(k.get("track_id")) for k in kits}
remaining_failures = failed_tracks & manifest_tracks
if remaining_failures:
    raise SystemExit(
        "failed tracks are still in the manifest: " + ", ".join(sorted(remaining_failures))
    )
print(f"[promote] QC failures excluded: {len(failed_tracks)} track(s)")
coverage = load(os.path.join(qc, "run_coverage.json"))
fracs = coverage.get("cluster_coverage") or {}
for stem in ("drums", "bass", "other", "vocals"):
    if float(fracs.get(stem, 0.0)) < min_coverage:
        raise SystemExit(f"cluster coverage for {stem} is {fracs.get(stem)} < {min_coverage}")
print(f"[promote] validated: kits={len(kits)} clusters={len(clusters)} phrases={phrases}")
print(f"[promote] joint val_auc={model.get('val_auc')} thresholds={model.get('thresholds')}")
print(f"[promote] cluster coverage={fracs}")
PY

# 2. Size budget.
BANK_MIB="$(du -sm "$BANK" | awk '{print $1}')"
LIVE_MIB="$(du -sm "$LIVE" | awk '{print $1}')"
log "staging bank: ${BANK_MIB} MiB; live bank: ${LIVE_MIB} MiB (max ${MAX_BANK_MIB} MiB)"
(( BANK_MIB <= MAX_BANK_MIB )) || fail "staging bank exceeds ${MAX_BANK_MIB} MiB"

if [[ "$APPLY" != "1" ]]; then
  log "dry-run complete; re-run with APPLY=1 to copy"
  exit 0
fi

# 3. Safety: the live bank must be committed before replacement.
if [[ -n "$(git -C "$ROOT" status --porcelain -- frontend/public/data/kits)" ]]; then
  fail "frontend/public/data/kits has uncommitted changes; commit or restore it first"
fi

log "copying kit manifest, taste scores and joint model"
cp "$BANK/kit_manifest.json" "$LIVE/kit_manifest.json"
cp "$BANK/taste_scores.json" "$LIVE/taste_scores.json"
cp "$JOINT/joint_ranker.json" "$LIVE/joint_ranker.json"
cp "$JOINT/joint_features.f32" "$LIVE/joint_features.f32"

log "syncing kit audio (delete stale kit directories)"
rsync -a --delete "$BANK/kits/" "$LIVE/kits/"

log "building frontend"
( cd "$ROOT/frontend" && npm run build )

log "sizes:"
du -sh "$ROOT/frontend/dist" "$LIVE" "$LIVE/kits"
log "live bank promoted. Review git diff and listen locally before committing."
