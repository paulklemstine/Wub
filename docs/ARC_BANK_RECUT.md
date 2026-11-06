# ARC Bank Re-cut — 8 Phrases per Kit

This document describes the reproducible re-cut of the Wub 4-bar kit bank from
the current `kits_flow` shape (390 kits, 2,203 phrases, 5.65 phrases/kit) to an
8-phrase-per-kit bank for the 4 → 8 → 16-bar arc. The new bank is built and
validated in **staging**; the live frontend bank is promoted separately after
review.

Orchestration script: [`scripts/recut_arc_bank.sh`](../scripts/recut_arc_bank.sh).

---

## 1. Inputs, outputs, and the live-bank boundary

Run on the laptop where the source stems and analysis JSON live:

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `WUB_ROOT` | `/tmp/wubtrain` | repo checkout / working root |
| `STEMS_DIR` | `/mnt/e/Wub/data/stems` | original 4-stem WAVs |
| `PROCESSED_DIR` | `$WUB_ROOT/processed` | per-track analysis JSON |
| `OUT` | `kits_arc8` | fresh full-corpus kit output |
| `STAGE` | `/tmp/wubtrain/arc8_stage` | all selected/scored artifacts |
| `PHRASES_PER_KIT` | `8` | cutter phrase budget |
| `KITS` | `402` | expected 16-cluster union (390 was the initial target) |
| `PER_STYLE` | `200` | style cap for the final subset |

The script refuses any `OUT` or `STAGE` path inside:

* `frontend/public/data/kits` — the live browser bank;
* `kits_flow` — the current 390-kit reference bank.

The corpus cutter writes only to `OUT`. Selection, models, scores, QC, and
coverage all write below `STAGE`. **Nothing in this pipeline copies into the
live frontend.** Promotion is a separate reviewed step (§8).

The script also hashes `frontend/public/data/kits/kit_manifest.json` before and
after the run and fails if it changed. Set `VERIFY_LIVE=0` only when another
process is intentionally updating the live bank during the build.

Logs are deliberately flat and predictable:

```
/tmp/wubtrain/arc8_pipeline.log
/tmp/wubtrain/arc8_corpus.log
/tmp/wubtrain/arc8_taste_dataset.log
/tmp/wubtrain/arc8_train_taste.log
/tmp/wubtrain/arc8_select.log
/tmp/wubtrain/arc8_metadata.log
/tmp/wubtrain/arc8_preroll.log
/tmp/wubtrain/arc8_stem_bars.log
/tmp/wubtrain/arc8_train_judge.log
/tmp/wubtrain/arc8_score.log
/tmp/wubtrain/arc8_joint_ranker.log
/tmp/wubtrain/arc8_qc.log
/tmp/wubtrain/arc8_coverage.log
```

---

## 2. One-command pipeline

```bash
cd /tmp/wubtrain
PHRASES_PER_KIT=8 \
KITS=402 \
OUT=kits_arc8 \
STAGE=/tmp/wubtrain/arc8_stage \
bash scripts/recut_arc_bank.sh
```

Useful overrides:

```bash
FORCE=1               # rebuild every stage
SKIP_CORPUS=1         # reuse an existing OUT/kit_manifest.json
WAIT_FOR_CORPUS=0     # fail instead of waiting for a running cutter
RUN_JOINT_RANKER=0    # skip the optional joint 4-stem ranker
RUN_QC=0              # skip reconstruction QC for a smoke run
RUN_ADD_STEM_COORDS=1 # timbre PCA stem_xy instead of judge embeddings
KEEP_CLUSTER=1        # keep the intermediate cluster copy for inspection
ALLOW_KIT_MISMATCH=1  # do not fail if the cluster union is not exactly 402
ALLOW_COVERAGE_FAIL=1 # do not fail the run-coverage gate
```

The script holds `flock` on `/tmp/wubtrain/arc8_pipeline.lock`, so two
orchestrator runs cannot fight over the same checkpoints.

---

## 3. Exact pipeline

### 3.1 Corpus cutter — fresh 8-phrase kits

```bash
PYTHONPATH=. python3 -u scripts/build_kit_corpus.py \
  --processed /tmp/wubtrain/processed \
  --stems /mnt/e/Wub/data/stems \
  --out kits_arc8 \
  --phrases-per-kit 8 \
  --bars 4 \
  --bitrate 64k \
  --workers 8
```

* Output: `kits_arc8/kit_manifest.json` + `kits_arc8/kits/<track_id>/*.opus`.
* `kits_flow` and the live frontend are never passed as `--out`.
* The cutter is idempotent: complete kits and phrase files are reused on a
  re-run; incomplete work resumes.
* The script skips the cutter if `arc8_corpus.log` already ends in
  `DONE rc=0` and the manifest exists.

Observed on the laptop: 1,842 analysed tracks, about 87 s per 10 kits with
8 workers at 8 phrases/kit, i.e. a multi-hour full-corpus run. This is expected.

### 3.2 Taste dataset

```bash
PYTHONPATH=. python3 -u scripts/build_taste_dataset.py \
  --processed /tmp/wubtrain/processed \
  --kits kits_arc8/kit_manifest.json \
  --out /tmp/wubtrain/arc8_stage/taste_dataset.json \
  --workers 8
```

The dataset validates that every phrase in `kits_arc8` joins a real 4-bar
window. A validation failure is fatal.

### 3.3 Taste classifier and transition ranker

```bash
PYTHONPATH=. python3 -u scripts/train_taste_model.py \
  --dataset /tmp/wubtrain/arc8_stage/taste_dataset.json \
  --out /tmp/wubtrain/arc8_stage/taste_model.pt \
  --log /tmp/wubtrain/arc8_stage/taste_train_log.json \
  --epochs 20 --device cuda

PYTHONPATH=. python3 -u scripts/train_taste_model.py \
  --dataset /tmp/wubtrain/arc8_stage/taste_dataset.json \
  --train-ranker \
  --classifier-ckpt /tmp/wubtrain/arc8_stage/taste_model.pt \
  --ranker-out /tmp/wubtrain/arc8_stage/taste_ranker.pt \
  --ranker-log /tmp/wubtrain/arc8_stage/taste_ranker_train_log.json \
  --ranker-epochs 60 --ranker-context-len 6 --device cuda
```

`--device auto` is resolved once by the orchestrator to `cuda` or `cpu`.

### 3.4 16-cluster bank and the 402-kit shipping subset

```bash
PYTHONPATH=. python3 -u scripts/select_cluster_bank.py \
  --manifest kits_arc8/kit_manifest.json \
  --source-dir kits_arc8 \
  --out /tmp/wubtrain/arc8_stage/kits_cluster \
  --workers 8

PYTHONPATH=. python3 -u scripts/select_kit_bank.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_cluster/kit_manifest.json \
  --source-dir /tmp/wubtrain/arc8_stage/kits_cluster \
  --out /tmp/wubtrain/arc8_stage/kits_bank \
  --per-style 200
```

`select_cluster_bank` is the authoritative 16-cluster selection. The current
16 pools union to **402 kits**. `select_kit_bank` then applies the style cap
and copies the final shipping subset.

`PER_STYLE=200` is intentionally non-binding for the current pools
(breaks 160, cyber 107, house 77, wave 46). It is a safety cap, not a hidden
way to shrink the bank. The script verifies two things:

1. the cluster union equals `KITS` (402 by default);
2. `select_kit_bank` kept exactly the same track set.

If a style cap drops a kit, the cluster metadata would reference a missing
kit, so the pipeline fails instead of shipping an inconsistent manifest. Raise
`PER_STYLE` or change the cluster selection deliberately.

### 3.5 Player metadata, pre-bar tails, and nebula coordinates

```bash
PYTHONPATH=. python3 -u scripts/merge_kit_metadata.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --dataset /tmp/wubtrain/arc8_stage/taste_dataset.json \
  --out /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json

PYTHONPATH=. python3 -u scripts/add_pre_roll.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --stems /mnt/e/Wub/data/stems \
  --kits-dir /tmp/wubtrain/arc8_stage/kits_bank \
  --workers 8

PYTHONPATH=. python3 -u scripts/link_pre_roll.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --kits-dir /tmp/wubtrain/arc8_stage/kits_bank
```

`merge_kit_metadata.py` adds `chroma`, `bass_root`, `is_intro`, `is_outro`, and
`once`, and refuses to write if fewer than 99% of phrases join. With a dataset
built from the same corpus manifest the join is 100%.

#### `add_kit_coords.py` has no CLI

The checked-in script hardcodes:

```
<repo>/frontend/public/data/kits/kit_manifest.json
```

Running it directly would rewrite the live laptop manifest. The orchestrator
therefore copies the unmodified script **and** the staged manifest into a
throw-away root:

```
$STAGE/_add_kit_coords/scripts/add_kit_coords.py
$STAGE/_add_kit_coords/frontend/public/data/kits/kit_manifest.json
```

runs it there, copies the resulting manifest back to `$STAGE/kits_bank`, and
deletes the sandbox. **Do not invoke `scripts/add_kit_coords.py` directly on a
machine that has a live frontend manifest.**

`add_stem_coords.py` is the timbre-PCA alternative for `stem_xy`:

```bash
PYTHONPATH=. python3 scripts/add_stem_coords.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --kits-dir /tmp/wubtrain/arc8_stage/kits_bank \
  --workers 8 \
  --out /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json
```

It is not used by default because `score_stem_kits.py --write-coords` below
writes the stronger judge-embedding UMAP/PCA coordinates directly into the
staged manifest.

### 3.6 Per-stem bars and four transition judges

```bash
PYTHONPATH=. python3 -u scripts/build_stem_bars.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --kits-dir /tmp/wubtrain/arc8_stage/kits_bank \
  --out /tmp/wubtrain/arc8_stage/stem_bars \
  --workers 8

for stem in drums bass other vocals; do
  PYTHONPATH=. python3 -u scripts/train_stem_judge.py \
    --bars /tmp/wubtrain/arc8_stage/stem_bars \
    --stem "$stem" \
    --out "/tmp/wubtrain/arc8_stage/stem_judge_${stem}.pt" \
    --log "/tmp/wubtrain/arc8_stage/stem_judge_${stem}_log.json" \
    --epochs 12 --workers 2 --device cuda
done
```

`build_stem_bars` produces 64×128 log-mel patches, four per phrase (one per
stem). Training on the shipping bank directly is intentional; the older
full-corpus `filter_stem_dataset.py` pass is unnecessary because the bars are
already scoped to the 402-kit bank.

### 3.7 Kit-level and per-stem scores

```bash
PYTHONPATH=. python3 -u scripts/score_kits.py \
  --dataset /tmp/wubtrain/arc8_stage/taste_dataset.json \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --model /tmp/wubtrain/arc8_stage/taste_model.pt \
  --ranker /tmp/wubtrain/arc8_stage/taste_ranker.pt \
  --out /tmp/wubtrain/arc8_stage/taste_scores_kits.json \
  --report /tmp/wubtrain/arc8_stage/taste_report.json \
  --device cuda

PYTHONPATH=. python3 -u scripts/score_stem_kits.py \
  --bars /tmp/wubtrain/arc8_stage/stem_bars \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --models /tmp/wubtrain/arc8_stage/stem_judge \
  --out /tmp/wubtrain/arc8_stage/taste_scores_stems.json \
  --topk 24 --write-coords --device cuda
```

`score_stem_kits.py --write-coords` writes the four `stem_xy` clouds into the
staged manifest. The orchestrator then merges the kit-level `transition`,
`quality`, and `next` fields with the per-stem `stem_next` map into the final:

```
/tmp/wubtrain/arc8_stage/kits_bank/taste_scores.json
```

### 3.8 Optional joint 4-stem ranker

The live arc player consumes the distilled joint ranker. Rebuild it after the
four stem judges exist:

```bash
PYTHONPATH=. python3 -u scripts/build_joint_ranker.py \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --dataset /tmp/wubtrain/arc8_stage/taste_dataset.json \
  --bars /tmp/wubtrain/arc8_stage/stem_bars \
  --mix-judge /tmp/wubtrain/data/audio_judge.pt \
  --encoders /tmp/wubtrain/arc8_stage/stem_judge \
  --out /tmp/wubtrain/arc8_stage/joint_ranker \
  --epochs 30 --device cuda
```

Outputs: `joint_features.f32`, `joint_ranker.json`, `joint_ranker.pt` under
`$STAGE/joint_ranker`. Set `RUN_JOINT_RANKER=0` to skip. If
`data/audio_judge.pt` is absent, the stage logs a warning and is skipped rather
than failing the bank.

### 3.9 Reconstruction QC

```bash
PYTHONPATH=. python3 -u scripts/kit_reconstruct.py \
  --mode compare \
  --manifest /tmp/wubtrain/arc8_stage/kits_bank/kit_manifest.json \
  --kits-dir /tmp/wubtrain/arc8_stage/kits_bank \
  --stems /mnt/e/Wub/data/stems \
  --processed /tmp/wubtrain/processed \
  --track-id "<all 402 comma-separated track ids>" \
  --out /tmp/wubtrain/arc8_stage/qc/recon \
  --report /tmp/wubtrain/arc8_stage/qc/kit_qc.json \
  --min-correlation 0.90 --min-snr 10 \
  --fail-on-error
```

The orchestrator expands `--track-id` to every kit in the bank. QC thresholds
come from `kit_reconstruct.py`:

| Metric | Gate |
| :--- | :--- |
| Pearson correlation | ≥ 0.90 |
| SNR | ≥ 10 dB |
| Onset/codec lag | ≤ 5 ms |
| Duration error | ≤ 10 ms |
| Timebase error | ≤ 20 ms |
| Silent/quiet phrase | rejected |

The 0.90 / 10 dB gates are **codec-aware**: Opus residuals on this bank sit
around 15-20 dB, so the old 0.95 / 20 dB defaults flag almost every phrase as
`codec_limited`. The lower gate still catches real waveform/alignment errors
(correlation <0.90 or SNR <10 dB).

`--fail-on-error` stops the pipeline for inspection. If the failures are
isolated tracks, run `scripts/filter_failed_kits.py` to remove them from the
manifest and cluster pools, prune their staging audio, and promote the
remaining bank. The promotion helper verifies that every failed track is
absent from the final manifest.

### 3.10 Run-coverage summary

The final stage reads the staged manifest and computes, per stem:

* the longest chain of consecutive phrases with `start_bar` increasing by
  exactly 4 bars;
* how many kits have runs of **4, 8, 16, and 32 bars** (1, 2, 4, and 8
  consecutive 4-bar phrases);
* per cluster, whether any kit provides a full **4-phrase = 16-bar** run.

Output: `$STAGE/qc/run_coverage.json` plus a printed table:

```
kits=402 phrases=... avg=...
drums  kits with >=1/2/4/8 phrase runs (4/8/16/32 bars): ...
...
clusters with a full 4-phrase (16-bar) run per stem:
  drums  16/16 (100.0%)
  ...
```

---

## 4. Size budget

Final shipping result for this pass:

```
358 kits / 2,487 phrases / 16 clusters / 659 MiB
```

The 402-kit candidate union produced 2,798 phrases. Full reconstruction QC
(0.90 correlation / 10 dB SNR) found 73 failures across 44 tracks; those
tracks were removed, leaving 358 kits. Every cluster still has a full
4-phrase (16-bar) run for all four stems (100% cluster coverage), and the new
joint ranker scores 0.801 / 0.784 / 0.747 AUC on 4/8/16-bar validation.

Measured older reference bank:

```
kits_flow: 390 kits, 2,203 phrases, 5.65 phrases/kit, 570 MiB on disk
```

Projected 8-phrase bank at the 390-kit target:

| Scenario | Main audio | + pre-rolls + metadata | Total |
| :--- | ---: | ---: | ---: |
| selected avg 7.0 phrases/kit | ≈ 681 MB | ≈ 720–735 MB | ≈ 687–701 MiB |
| selected avg 8.0 phrases/kit | ≈ 779 MB | ≈ 820–835 MB | ≈ 782–796 MiB |

The estimate uses the observed 62.4 kB average per 4-bar drum phrase from the
in-progress 8-phrase cutter and the 1.15 kB average pre-roll size from
`kits_flow`. The old 5.65-phrase bank measures 570 MiB; the full-corpus cutter
is averaging ~6.9 phrases/kit (59% of kits reach the full 8), so the selected
bank should land around **700–800 MiB**. That is the target, not a hard
requirement: the hard gate is under 1 GiB.

The budget applies to the whole published site, not only the kit bank. Before
promotion, check:

```bash
cd frontend
npm run build
du -sh dist public/data/kits
```

The orchestrator prints `du -sh` for `$STAGE/kits_bank` and
`$STAGE/kits_bank/kits`, and warns if the staging bank alone reaches 1 GiB.

---

## 5. QC gates

A run is shippable only when all of these pass:

1. **Manifest integrity** — every phrase has four Opus files; the cluster
   selection drops nothing silently.
2. **Kit count** — exactly `KITS` (402 candidate; 358 after QC filtering) kits in the 16-cluster union.
3. **Metadata join** — ≥99% of phrases receive `chroma`/`bass_root`/`once`;
   expected 100% because the dataset is built from the same manifest.
4. **Reconstruction** — `kit_reconstruct.py --mode compare --fail-on-error`
   passes all correlation/SNR/lag/duration gates for all candidate kits; isolated failed tracks are removed by `filter_failed_kits.py`.
5. **Run coverage** — **at least 80% of the 16 clusters must have a full
   4-phrase (16-bar) run for each of drums, bass, other, and vocals.**
   The reference `kits_flow` bank is already at 16/16 (100%) for all four
   stems; the 8-phrase cut should be at least as good.
6. **Size** — staged bank under 1 GiB, and the whole published site under
   1 GiB.

Current reference coverage for context:

```
kits_flow: 390 kits, 2,203 phrases
kits with >=4-phrase (16-bar) run: 218 / 390 per stem
clusters with a full 4-phrase run: 16 / 16 per stem (100%)
```

---

## 6. Run-coverage target details

A phrase is a real 4-bar window. A run is a sequence of phrases from one track
whose `start_bar` values are consecutive multiples of 4:

```
phrase p01 start_bar=13, p02=17, p03=21, p04=25  -> 4-phrase = 16-bar run
```

The UI arc uses 4-, 8-, and 16-bar blocks. A 16-bar block needs a 4-phrase run;
an 8-bar block needs 2 phrases. Runs longer than the block are truncated/looped
by the player, so the important coverage is:

* **4 bars** — 1 phrase (always present if the kit has audio);
* **8 bars** — 2 consecutive phrases;
* **16 bars** — 4 consecutive phrases;
* **32 bars** — 8 consecutive phrases (a bonus, useful for pins).

The gate is cluster-level rather than bank-level because a cluster with only
single phrases cannot fill its 16-bar blocks musically, even if the rest of the
bank is dense.

---

## 7. Re-runs and recovery

The script is safe to re-run:

* the cutter resumes and reuses complete phrases;
* a completed corpus is skipped when `arc8_corpus.log` ends in `DONE rc=0`;
* a running cutter is waited for (`WAIT_FOR_CORPUS=0` disables the wait);
* manifests that are rewritten with identical content keep their old mtime, so
  downstream bars/scores/QC are not rebuilt spuriously;
* `FORCE=1` is the explicit “redo everything” switch.

If the cluster union is not `KITS`:

```bash
grep -E "skip|union|selected" /tmp/wubtrain/arc8_select.log
```

Inspect the skipped cluster/tempo pool before using
`ALLOW_KIT_MISMATCH=1`. Do not lower `KITS` silently: the target is a property
of the bank, not a display preference.

If QC fails, the report is at `$STAGE/qc/kit_qc.json`; the per-phrase rows show
which stem/track failed. If coverage fails, inspect `$STAGE/qc/run_coverage.json`
and the per-cluster `max_run_phrases`.

---

## 8. Promotion (separate, reviewed step)

The recut pipeline stops at staging. Promotion is deliberately a reviewed
step, but the copying/validation can be done by
`scripts/promote_arc_bank.sh`:

* dry run (default): validates the manifest (402 kits, 16 clusters, all stems),
  taste scores, joint model dimensions/feature file, QC failures, cluster
  coverage, and the staging size; prints the result.
* `APPLY=1`: refuses if `frontend/public/data/kits` has uncommitted changes,
  then copies the manifest, taste scores, joint model, and kit audio, runs the
  frontend build, and prints `du -sh`. It never commits.

Review `$STAGE/qc/kit_qc.json`, `$STAGE/qc/run_coverage.json`, and the dry run
before applying. The manual checklist remains:

1. Review `$STAGE/qc/kit_qc.json` and `$STAGE/qc/run_coverage.json`.
2. Confirm `$STAGE/kits_bank/taste_scores.json` has both kit-level and
   `stem_next` fields.
3. Build the frontend and check the 1 GiB budget.
4. If QC found isolated failing tracks, run
   `python3 scripts/filter_failed_kits.py --manifest ... --report ... --out ...
   --kits-dir ... --prune-audio` before promotion.
5. Promote, then review `git diff` and listen locally before committing.
6. Remove stale kit directories from the live bank only as part of that
   reviewed promotion; never delete `kits_flow` or the staging outputs.

Neither `scripts/recut_arc_bank.sh` nor this document modifies
`frontend/public/data/kits`, `AGENTS.md`, or `docs/KIT_ENGINE.md`.
