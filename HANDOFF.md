# WUB HANDOFF — current state (2026-11-04)

## What this is

Wub KG-505 is now a **song-kit phrase jukebox**, not a VQ/medoid slice
synthesizer. A kit is one real four-stem track; a phrase is a real 4-bar window
cut from that track. The browser plays phrases at rate 1.0 (no resampling, no
pitch shift), locks tempo/key to the kit, plays 12-block sets, and rotates whole
kits through a drumless bridge. Full design: `docs/KIT_ENGINE.md`.

Repo root: `/home/raver1975/Wub` (git repo). The deployed site is the static
`frontend/dist` build; the FastAPI server and legacy slice paths are not needed
for the shipped engine.

## Machine layout

- **Local box** (Linux, 16 cores, 31 GB): repo, analysis, tests, frontend build,
  deploy. The **GTX 1060 is reserved for L.O.V.E** — do not schedule Wub
  training on it.
- **Laptop "Victus"** (Win11 + WSL Ubuntu, RTX 4050 6 GB, 6 cores, **5 GB RAM —
  the binding constraint**): Demucs separation and taste-model training.
- **Corpus on the laptop**: `/mnt/e/Wub/data/stems`, **1,842 stem directories,
  342 GB**. Each directory is one track's `drums.wav`, `bass.wav`, `other.wav`,
  `vocals.wav`.
- **Analysis on the laptop**: `/tmp/wubtrain/processed/*.json` — 1,699
  four-stem analyses are complete; a full rebuild to 1,842 is running.
- **Kit work on the laptop**: `/tmp/wubtrain/kits_all` (full bank),
  `/tmp/wubtrain/kits_ship` (style-balanced shipping subset), and
  `/tmp/wubtrain/taste_dataset.json`.
- **Access**: reverse tunnel from WSL on the laptop:
  ```bash
  ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
      -R 2223:localhost:22 raver1975@192.168.1.118
  ```
  Then from the local box:
  ```bash
  export WUB_LAPTOP_PASS='<laptop-wsl-password>'
  printf '%s' 'hostname' | tools/laptop.sh exec
  tools/laptop.sh put <local> <remote>
  tools/laptop.sh get <remote> <local>
  ```
  `tools/laptop.sh exec` has a 240 s per-call cap: launch long jobs with
  `setsid nohup … & disown` and poll logs. Direct Windows `ssh -p 2222` is
  broken (RST); use the reverse tunnel.

## Transfer budget

- Effective transfer rate is about **4 MB/s** over the tunnel. Plan for it; do
  not pull the 342 GB stem corpus across.
- Transfer **analysis JSON, model checkpoints, kit manifests, and the selected
  Opus bank** — not stems. The shipped kit bank is currently ~602 MB for 355
  kits; a full 600-kit bank is ~0.9 GB with the app.
- Small-file copies are slower than the raw rate suggests. Tar/zip a kit bank
  or processed subset before `tools/laptop.sh get`, and verify file counts and
  sizes after transfer.
- Long transfers should be chunked or run in the background; `laptop.sh` waits
  up to 1200 s for `put`/`get`, but the tunnel can drop.

## GPU split

| Machine | GPU | Work |
| :--- | :--- | :--- |
| Local | GTX 1060 | **L.O.V.E only** — Wub training is not scheduled here. |
| Laptop | RTX 4050 6 GB | Demucs separation, taste classifier/ranker training. |
| Local CPU | — | `analyze_stem_corpus_vec.py`, dataset builds, tests, Vite build. |
| Laptop CPU/WSL | — | Kit cutting/Opus encode, `score_kits.py`, QC. |

## Current pipeline (kit path)

| Step | Script | Output |
| :--- | :--- | :--- |
| 1. Analyse stems | `scripts/analyze_stem_corpus_vec.py` | `/tmp/wubtrain/processed/*.json` |
| 2. Cut 4-bar kits | `scripts/build_kit_corpus.py` | `kits_all/kits/<tid>/*.opus`, `kit_manifest.json` |
| 3. Build taste data | `scripts/build_taste_dataset.py` | `/tmp/wubtrain/taste_dataset.json` (48-d windows) |
| 4. Train classifier | `scripts/train_taste_model.py` | `taste_model.pt` |
| 5. Train ranker | `scripts/train_taste_model.py --train-ranker` | `taste_ranker.pt` |
| 6. Select bank | `scripts/select_kit_bank.py` | `kits_ship/` (style-balanced, ≤600 kits) |
| 7. Score bank | `scripts/score_kits.py` | `frontend/public/data/kits/taste_scores.json` |
| 8. Nebula coords | `scripts/add_kit_coords.py` | PCA `x`/`y` in `kit_manifest.json` |
| QC | `scripts/kit_reconstruct.py` | decode/compare report against source stems |

`scripts/retrain_all.sh` and `scripts/continuous_trainer.py` still describe the
old slice/vocabulary/arranger pipeline. Do not use them for a kit rebuild; the
kit path is the sequence above.

### Commands (laptop, repo checkout at `/tmp/wubtrain`)

```bash
cd /tmp/wubtrain
PROCESSED=/tmp/wubtrain/processed
STEMS=/mnt/e/Wub/data/stems

PYTHONPATH=. python3 scripts/analyze_stem_corpus_vec.py \
  --stems "$STEMS" --out "$PROCESSED" --workers 4

PYTHONPATH=. python3 scripts/build_kit_corpus.py \
  --processed "$PROCESSED" --stems "$STEMS" --out kits_all \
  --phrases-per-kit 6 --workers 8

PYTHONPATH=. python3 scripts/build_taste_dataset.py \
  --processed "$PROCESSED" --kits kits_all/kit_manifest.json \
  --out /tmp/wubtrain/taste_dataset.json

PYTHONPATH=. python3 scripts/train_taste_model.py \
  --dataset /tmp/wubtrain/taste_dataset.json \
  --out /tmp/wubtrain/taste_model.pt \
  --log /tmp/wubtrain/taste_train_log.json --epochs 20 --device cuda

PYTHONPATH=. python3 scripts/train_taste_model.py \
  --dataset /tmp/wubtrain/taste_dataset.json --train-ranker \
  --classifier-ckpt /tmp/wubtrain/taste_model.pt \
  --ranker-out /tmp/wubtrain/taste_ranker.pt \
  --ranker-log /tmp/wubtrain/taste_ranker_train_log.json \
  --ranker-epochs 60 --ranker-context-len 6 --device cuda

python3 scripts/select_kit_bank.py \
  --manifest kits_all/kit_manifest.json --source-dir kits_all \
  --out kits_ship --per-style 150

PYTHONPATH=. python3 scripts/score_kits.py \
  --dataset /tmp/wubtrain/taste_dataset.json \
  --manifest kits_ship/kit_manifest.json \
  --model /tmp/wubtrain/taste_model.pt \
  --ranker /tmp/wubtrain/taste_ranker.pt \
  --out frontend/public/data/kits/taste_scores.json

cp kits_ship/kit_manifest.json frontend/public/data/kits/kit_manifest.json
# cp does not delete stale kit dirs; remove the old kits/ tree when shrinking.
cp -r kits_ship/kits/. frontend/public/data/kits/kits/
python3 scripts/add_kit_coords.py
```

## Current status

- **Shipped**: 355 kits / 2,466 phrases / ~602 MB, four styles (breaks, cyber,
  house, wave), 60-epoch ranker, `val_auc` 0.845 (classifier), matrix median row
  max 0.52.
- **Ranker results** (held-out tracks, context 6):
  - 20 epochs: within-kit top-1 51.5% (171/332), full-track top-1 19.3%.
  - 60 epochs: within-kit top-1 59.0% (196/332), full-track top-1 23.8%,
    full-track top-5 74.3%.
  - 60 epochs + hard negatives: 22.2% full-track top-1 (worse).
  - 60 epochs + context 8: 23.0% full-track top-1 (worse).
- **In progress**: full corpus rebuild to 1,842 tracks and a larger kit bank; a
  d192/L4 120-epoch ranker run on the full corpus. Those weights are not
  shipped yet.
- **Pages budget**: published site must stay under 1 GB. `frontend/dist` was
  ~715 MB at 355 kits. Cap the bank at ~600 kits / 6 phrases / 64 kbps mono;
  `select_kit_bank.py --per-style 150` enforces the count ceiling.

## Known gotchas

- **Laptop 5 GB RAM, no swap**: OOM is the main failure mode. Use `--workers`
  conservatively (4 for analysis, 8 for kit cutting), and `nohup` everything
  long.
- **Ogg Opus always decodes at 48 kHz**, even though source stems and the
  manifest say 44.1 kHz. Read `AudioBuffer.sampleRate`; never assume 44.1 kHz
  when converting frames to seconds.
- **Kit tempo is fitted**, `960 / median phrase duration`, not the manifest
  BPM. Beat-grid errors change the slot length; see `docs/KIT_ENGINE.md`
  limitations.
- **Idempotency**: `build_kit_corpus.py` reuses a kit with at least 4 valid
  phrases and all four stem files; `analyze_stem_corpus_vec.py` resumes by
  track. Re-running is safe, but changing `--phrases-per-kit` requires a fresh
  `--out` or removal of the old manifest entry.
- **Kit manifest validation**: `tests/test_style_kits.py` requires ≥300 kits,
  valid BPM/key, 4–8 phrases per kit, and all Opus files present.
- **Demo/legacy UI modes are disabled**; `kit` is the shipped mode. A kit load
  failure logs once and falls back to `jukebox` rather than crashing.
- **Background shells get reaped**: use `setsid nohup … & disown` for long
  laptop jobs, then poll the log over the tunnel.

## Next steps

1. Finish the 1,842-track analysis and rebuild `kits_all`.
2. Let the d192/L4 120-epoch ranker finish; compare full-track top-1/top-5
   against the shipped 60-epoch/context-6 checkpoint before replacing
   `taste_scores.json`.
3. Re-select `kits_ship` under the 1 GB Pages budget, re-score, add coords, run
   `kit_reconstruct.py --mode compare` with the shipped `--manifest` /
   `--kits-dir`, `--stems /mnt/e/Wub/data/stems`,
   `--processed /tmp/wubtrain/processed`, and `--fail-on-error`.
4. Run `pytest`, `cd frontend && npm run build`, check `du -sh frontend/dist`,
   then commit/push (the pre-commit hook bumps the build number).
