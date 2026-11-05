# 🤖 AGENTS.md — Master Developer & AI Agent Handbook

Welcome to the **Wub KG-505** codebase. This document serves as the primary technical specification, architectural guide, and operational manual for AI agents and human engineers modifying, debugging, or extending this repository.

---

## 🧭 Project Identity & Core Philosophy

- **Name**: Wub KG-505 (Klemstine Group KG-505 Production).
- **Domain**: Real-time generative electronic music, audio stem analysis, real 4-bar phrase performance, and web-based audio engineering.
- **Mission**: Ingest an authentic audio training corpus (1,700+ dance tracks), separate into stems (drums, bass, other, vocals), mine real 4-bar phrases into song kits, learn a taste model for phrase and kit transitions, and play a neverending, highly musical DJ set directly in the browser.
- **Dual-Engine Execution**:
  1. **🧠 In-Browser Kit Jukebox (shipped)**: 100% standalone, client-side Web Audio engine. The cluster bank runs four independent stem streams that share one bar grid; each stem can be dragged to a new style region, pinned for 32 bars, and the session follows a 4→8→16-bar arc before rotating key/tempo every 104 bars. Real 4-bar phrases, no slice resynthesis or `playbackRate` resampling; tempo fit (<=2%) and Camelot transposition (<=5 semitones, vocals <=3) use Signalsmith Stretch. Full design: `docs/KIT_ENGINE.md`.
  2. **⚡ WebSocket Cloud Sync (legacy/hybrid)**: Client/server mode streaming tatums and slice references from the Python FastAPI server backed by FAISS indices. The deployed site does not need it; the kit bank is the shipped path.

---

## 🚨 Invariant Architectural Rules for Agents

When making changes to this codebase, you must adhere strictly to these non-negotiable engineering rules:

### 1. Test Suite Hygiene (100% Green Requirement)
- Every single test in `pytest` must pass at all times.
- Run `pytest` before committing any changes.
- Do not disable, `@pytest.mark.skip`, or comment out existing tests unless explicitly directed.
- If you introduce a new feature or modify an algorithm, add corresponding unit tests in `tests/`.

### 2. Frontend TypeScript Build Integrity
- The frontend must compile cleanly with zero TypeScript errors:
  ```bash
  cd frontend && npm run build
  ```
  (`tsc && vite build`).
- Do not use `@ts-ignore` or `any` unless interfacing with dynamic JSON schema payloads at boundaries.

### 3. Automated Version Control Protocol
- The project version is formatted as `v0.2.<build_number> #<build_number>` (e.g. `v0.2.184 #184`).
- A Git pre-commit hook (`.git/hooks/pre-commit`, also copied from `.githooks/pre-commit`) runs `scripts/bump_version.py` and increments the build number on every commit.
- Files kept in sync:
  - `version.json`
  - `frontend/src/version.ts`
  - `frontend/index.html` (version badge and footer)
- Whenever you commit with `git commit`, the hook triggers automatically. Verify the output displays `⚙️ [git hook] Incrementing Wub version control number...`.

### 4. Sample-Accurate Web Audio Timing & `UIEventQueue` Pattern
- **Lookahead Scheduling**: Web Audio requires scheduling audio notes 100–150ms into the future (`when = nextStepTime`).
- **NEVER Mutate DOM in Lookahead**: Calling `updateGrooveboxUI()` inside the audio scheduling loop causes the UI to jump ahead before the sound plays.
- **NEVER Use Uncoordinated `setTimeout`**: Clamped browser timer resolution (4ms) and main-thread load cause `setTimeout` delays to fire out of order, making the UI jump back and forth in time.
- **ALWAYS Use the `UIEventQueue` Pattern**:
  1. The audio scheduler pushes `{ atAudioSec, meta, step, totalSteps }` into `this.uiEventQueue`.
  2. In `frame()` (running at 60 FPS via `requestAnimationFrame`), compare `this.engine.now() >= event.atAudioSec`.
  3. When the audio clock reaches the event's playback time, shift the event and call `updateGrooveboxUI(latestUIEvent.meta)`.
  4. If multiple events expired during a tab pause, fast-forward to the latest event so the UI never plays catch-up or displays out-of-order steps.

### 5. JavaScript Array Inclusion vs. Object Key Check
- In Python: `if step in [0, 4, 8, 12]:` checks array elements.
- In JavaScript / TypeScript: `if (step in [0, 4, 8, 12])` checks **indices/keys** (`0, 1, 2, 3`), NOT values!
- **RULE**: In TypeScript, **ALWAYS** use:
  ```ts
  if ([0, 4, 8, 12].includes(step % 16))
  ```

### 6. Harmonic Pitch-Shifting Math (legacy slice modes only)
- **Kit mode must not repitch anything.** Kit phrases play at
  `playbackRate = 1.0` with no detune. The kit's key is the session key.
- This rule applies only to the legacy `jukebox`/`slices`/`phrase_blocks`
  fallback paths.
- All pitched legacy slices (bass, synths/chords, vocals) are tagged with chroma pitch $0 \dots 11$ (C=0, C#=1, ..., B=11).
- When repitching a slice with root $P_{\text{slice}}$ to match active chord root $P_{\text{target}}$:
  ```ts
  let shift = (targetPitch - slicePitch) % 12;
  if (shift > 6) shift -= 12;
  if (shift < -6) shift += 12;
  const playbackRate = Math.pow(2, shift / 12);
  ```
  This guarantees that slices never shift by more than 6 semitones (half an octave), preserving natural acoustic timbre while maintaining perfect musical harmony.

### 7. Kit Playback Invariants
**Legacy single-kit mode** (older manifests without `clusters`):
- **Never resample, time-stretch, or repitch a kit phrase.** `scheduleSlice`
  must use `rate: 1.0`; the kit's key is the session key.
- **Block duration is the phrase's own duration.** `blockDurationSec =
  phrase.duration_sec`; the next block starts exactly where the previous ends.
- **12-block set, 12th block bridge.** The final block plays `other` +
  `vocals`; preload the next kit on block 10 and switch only at a block
  boundary when ready; otherwise extend the bridge.

**Per-stem remix mode** (manifests with `clusters`; the shipped path):
- **Four stem streams, one bar grid.** Each stem has its own current phrase,
  voice, pin and lazy cache entry. Never splice two songs inside one stem's
  phrase; cross-song recombination happens only at block boundaries.
- **No `playbackRate` resampling.** Tempo fit uses Signalsmith Stretch `rate`
  (cap 2%); key fit uses `semitones` (drums 0, vocals ±3, bass/other ±5).
- **Anchor opening, then joint morph.** The first two 4-bar blocks stay on the
  session's anchor kit. Then a **joint transition ranker** evaluates 1- and
  2-stem candidate mixes from different tracks against the current mix (last
  two states + deltas) and picks the best. The mix judge is the teacher; the
  student is distilled to per-phrase features and runs in the browser. If the
  best candidate is below the calibrated threshold, the whole mix holds for a
  block. A held stem advances to its real +4-bar successor when its run has
  one, otherwise it loops that run; pinned stems hold while the others morph.
- **Drag commits at the next bar; pins last 32 bars.** The dragged stem schedules
  one long run from the next bar to the pin's block boundary, so a following
  8/16-bar block cannot take it back. The reticle follows the playing phrase; a
  dashed ghost ring marks the pinned region. Double-click releases the pin. If
  the target audio is not decoded by the bar, start on the next bar and show a
  buffering status.
- **4 → 8 → 16-bar arc.** A set is 104 bars / 19 blocks: 48 bars of 4-bar
  blocks, 32 bars of 8-bar blocks, one 16-bar block, then an 8-bar drumless
  `other + vocals` bridge. A block is a run of consecutive 4-bar phrases from
  one source track, concatenated into one Stretch voice; short runs loop.
  `blockBars`/`blockDurationSec` drive the audio grid while `slotSec` keeps the
  displayed BPM at the 4-bar value.
- **Cluster rotation every 104 bars.** The bridge starts at bar 96; the next
  Camelot-neighbour cluster (within ±3 BPM) takes over at bar 104 and releases
  all pins.
- **Lazy audio only.** 64-entry LRU, one block of prefetch lead, two alternates
  for drags. A failed load repeats the current phrase and blacklists the kit;
  it must never stop the transport.
- **Never create a StretchVoice per phrase.** Use the fixed two-voice-per-stem
  pool (8 nodes total), reset and reuse them. A run of 1-4 phrases is
  concatenated into one buffer and one voice; per-phrase nodes leak WASM memory
  and crash the tab after ~4 minutes.
- **Never normalise or re-gain assets.** A kit phrase keeps the source's
  per-stem balance; per-stem `peak`/`rms_db` are metadata. The small runtime
  level match in `stem_remix.ts` is a mixer layer, not a baked gain.
- **Taste model is optional.** A missing or invalid `taste_scores.json` logs
  once and falls back to the rule selector; it must never stop playback.

---

## 🗺️ Codebase Map & Responsibilities

### Frontend (`frontend/src/`)
| File | Responsibility |
| :--- | :--- |
| [`kit_player.ts`](file:///home/raver1975/Wub/frontend/src/kit_player.ts) | Kit facade: manifest/taste loading, legacy 12-block single-kit player, and the remix `scheduleBlock` adapter. |
| [`stem_remix.ts`](file:///home/raver1975/Wub/frontend/src/stem_remix.ts) | Per-stem remix engine: cluster pools, 4→8→16-bar arc, consecutive-phrase runs, joint ranker selection, 32-bar pins, next-bar drag commit, cluster rotation, lazy cache, and per-stem nebula points. |
| [`joint_ranker.ts`](file:///home/raver1975/Wub/frontend/src/joint_ranker.ts) | Browser inference for the joint four-stem transition ranker: 400-d states, last-two-state context, 1/2-stem scoring, calibrated hold threshold. |
| [`stretch_voice.ts`](file:///home/raver1975/Wub/frontend/src/stretch_voice.ts) | Signalsmith Stretch voice wrapper: pre-roll + 1-4 phrase run buffer, tempo/key scheduling, gain fades, pooled reuse. |
| [`client_ml_sequencer.ts`](file:///home/raver1975/Wub/frontend/src/client_ml_sequencer.ts) | In-browser conductor. Owns the kit block clock and kit tempo/key lock; retains the legacy jukebox/slice/step sequencer as fallback only. Also holds the chord/arranger state that drives kit section selection. |
| [`audio_engine.ts`](file:///home/raver1975/Wub/frontend/src/audio_engine.ts) | Core Web Audio context wrapper, 4-stem gain buses, analysers, `scheduleSlice` (rate/fades), and AudioBuffer decode for Opus/WAV kit assets. |
| [`main.ts`](file:///home/raver1975/Wub/frontend/src/main.ts) | Main application coordinator, UI event queue, transport controls, kit set-change status, reticle-to-phrase steering, and WebSocket listener. |
| [`visualizer.ts`](file:///home/raver1975/Wub/frontend/src/visualizer.ts) | 4-stem oscilloscope waveforms, FFT spectral density, peak/RMS meters, concentric radar rings, and the kit phrase nebula (PCA cloud, active-kit highlight, reticles). |
| [`tamagotchi.ts`](file:///home/raver1975/Wub/frontend/src/tamagotchi.ts) | WubGotchi neural music pet avatar, learning ASCII animated face expressions, vowel mouth shapes, beat head-bobbing, and XP evolution. |
| [`psychedelic_bg.ts`](file:///home/raver1975/Wub/frontend/src/psychedelic_bg.ts) | WebGL full-screen real-time raymarching shader with DMT / LSD kaleidoscopic folding, borderless polar wrapping, and RMS reactivity. |
| [`version.ts`](file:///home/raver1975/Wub/frontend/src/version.ts) | Git-hook synchronized build numbers and release timestamps. |

### Backend & Core Algorithms (`wub_core/` & `server/`)
| File | Responsibility |
| :--- | :--- |
| [`wub_core/section_arcs.py`](file:///home/raver1975/Wub/wub_core/section_arcs.py) | Per-bar energy/onset features and section labelling (INTRO / GROOVE_A / BUILDUP / DROP / BREAKDOWN) used by the kit corpus and taste dataset. |
| [`wub_core/groovy_pattern_assembler.py`](file:///home/raver1975/Wub/wub_core/groovy_pattern_assembler.py) | Legacy slice classifier, Camelot consonance profiles, multi-bar groove templates, and sidechain simulation. Not used by kit playback. |
| [`wub_core/pattern_audio_trainer.py`](file:///home/raver1975/Wub/wub_core/pattern_audio_trainer.py) | Legacy multi-scale log-Mel spectrogram similarity, onset correlation, and reinforcement feedback. |
| [`wub_core/patterns.py`](file:///home/raver1975/Wub/wub_core/patterns.py) | Legacy track-indexed slice palette, multi-stem patterns, and densification. |
| [`wub_core/groovebox.py`](file:///home/raver1975/Wub/wub_core/groovebox.py) | Server-side legacy groovebox conductor and 16/32-step sequencer. |
| [`wub_core/model_transformer.py`](file:///home/raver1975/Wub/wub_core/model_transformer.py) | WubGPT causal transformer for legacy multi-stem token sampling. |
| [`server/main.py`](file:///home/raver1975/Wub/server/main.py) | FastAPI application, REST endpoints (`/health`, `/api/tracks`, `/audio/...`), and WebSocket `/ws/stream`. |
| [`server/stream_manager.py`](file:///home/raver1975/Wub/server/stream_manager.py) | Stream session manager, lookahead buffer refill, and slice WAV rendering. |

### Scripts (`scripts/`)
| File | Responsibility |
| :--- | :--- |
| [`build_kit_corpus.py`](file:///home/raver1975/Wub/scripts/build_kit_corpus.py) | Cuts real 4-bar, 4-stem phrase kits from analysed tracks; writes `kits/<track_id>/*.opus` + `kit_manifest.json`. Never normalises or re-gains. |
| [`build_taste_dataset.py`](file:///home/raver1975/Wub/scripts/build_taste_dataset.py) | Builds the 48-d window feature dataset from analysis JSON only (chroma, timbre, density, loudness, section, energy, vocal presence) plus kit-phrase joins. |
| [`train_taste_model.py`](file:///home/raver1975/Wub/scripts/train_taste_model.py) | Trains the binary arrangement classifier and the pairwise transition ranker (context Transformer + candidate MLP, InfoNCE). |
| [`score_kits.py`](file:///home/raver1975/Wub/scripts/score_kits.py) | Scores the shipping bank into `taste_scores.json`: transition matrix, phrase quality, top-10 next kits; writes `/tmp/taste_report.json`. |
| [`select_kit_bank.py`](file:///home/raver1975/Wub/scripts/select_kit_bank.py) | Selects a style-balanced shipping bank by phrase count, section coverage, energy, stem presence, and embedding novelty; copies only selected Opus files. |
| [`add_kit_coords.py`](file:///home/raver1975/Wub/scripts/add_kit_coords.py) | Projects 64-d phrase embeddings to deterministic PCA `x`/`y` in `kit_manifest.json` for the legacy nebula. |
| [`select_cluster_bank.py`](file:///home/raver1975/Wub/scripts/select_cluster_bank.py) | Selects the 16 style-balanced key/tempo clusters and writes the `clusters` pool map. |
| [`add_pre_roll.py`](file:///home/raver1975/Wub/scripts/add_pre_roll.py) | Extracts the 80 ms pre-bar attack tails and writes `*_pre.opus` sidecars. |
| [`link_pre_roll.py`](file:///home/raver1975/Wub/scripts/link_pre_roll.py) | Reattaches on-disk pre-roll sidecars to a manifest (worker mutations are not propagated). |
| [`merge_kit_metadata.py`](file:///home/raver1975/Wub/scripts/merge_kit_metadata.py) | Merges chroma, bass root and once-only flags from the taste dataset into the selected bank. |
| [`add_stem_coords.py`](file:///home/raver1975/Wub/scripts/add_stem_coords.py) | Computes per-stem log-mel descriptors and PCA `stem_xy` for the four nebula clouds. |
| [`layout_stem_clouds.py`](file:///home/raver1975/Wub/scripts/layout_stem_clouds.py) | Places the four per-stem clouds in quadrants of the shared nebula. |
| [`build_stem_bars.py`](file:///home/raver1975/Wub/scripts/build_stem_bars.py) | Builds per-stem 64x128 log-mel bar datasets for the four transition judges. |
| [`filter_stem_dataset.py`](file:///home/raver1975/Wub/scripts/filter_stem_dataset.py) | Filters a per-stem bar dataset to the 390 shipping tracks. |
| [`train_stem_judge.py`](file:///home/raver1975/Wub/scripts/train_stem_judge.py) | Trains one same-song-vs-cross-song CNN per stem and writes its 128-d embedding. |
| [`score_stem_kits.py`](file:///home/raver1975/Wub/scripts/score_stem_kits.py) | Scores the shipping bank per stem, writes top-K `stem_next` and judge-PCA `stem_xy`. |
| [`build_joint_ranker.py`](file:///home/raver1975/Wub/scripts/build_joint_ranker.py) | Builds the joint transition dataset, distils the mix judge into the 400-d student ranker, and exports browser features/weights. |
| [`kit_reconstruct.py`](file:///home/raver1975/Wub/scripts/kit_reconstruct.py) | Decodes and QC-checks kit audio against source stems (correlation, SNR, RMS delta, codec lag, overlap/duration errors). |
| [`analyze_stem_corpus_vec.py`](file:///home/raver1975/Wub/scripts/analyze_stem_corpus_vec.py) | Vectorised stem analysis (whole-stem CQT/MFCC averaged per slice; much faster than the stock path). Run it where the stems live (`/mnt/e/Wub/data/stems` on the laptop) to build `processed/*.json`. |
| [`export_standalone_client_bundle.py`](file:///home/raver1975/Wub/scripts/export_standalone_client_bundle.py) | Legacy bundle builder for the retired slice/VQ path; not part of the kit rebuild. |
| [`bump_version.py`](file:///home/raver1975/Wub/scripts/bump_version.py) | Git pre-commit hook script that increments/syncs the build number in `version.json`, `frontend/src/version.ts`, and `frontend/index.html`. |

---

## 🎛️ Essential Workflows & Commands

### Running Tests
```bash
# Run entire test suite
pytest -v

# Run a specific test module
pytest tests/test_groovy_pattern_assembler.py -v
```

### Building Frontend
```bash
cd frontend
npm run build

# Published GitHub Pages site must stay under 1 GB.
du -sh dist public/data/kits
```

### Rebuilding / Retraining the Kit Bank
Run from the repo checkout on the laptop (RTX 4050; corpus on `/mnt/e`). Full
design and evaluation details: `docs/KIT_ENGINE.md`.
```bash
cd /tmp/wubtrain
PROCESSED=/tmp/wubtrain/processed
STEMS=/mnt/e/Wub/data/stems

# 1. Analyse every 4-stem directory (idempotent).
PYTHONPATH=. python3 scripts/analyze_stem_corpus_vec.py \
  --stems "$STEMS" --out "$PROCESSED" --workers 4

# 2. Cut real 4-bar phrase kits (6 phrases/kit for the Pages budget).
# To change the phrase count, use a fresh --out (existing kits are reused).
PYTHONPATH=. python3 scripts/build_kit_corpus.py \
  --processed "$PROCESSED" --stems "$STEMS" --out kits_all \
  --phrases-per-kit 6 --workers 8

# 3. Build the 48-d taste dataset from analysis JSON.
PYTHONPATH=. python3 scripts/build_taste_dataset.py \
  --processed "$PROCESSED" --kits kits_all/kit_manifest.json \
  --out /tmp/wubtrain/taste_dataset.json

# 4. Train classifier, then the 6-window transition ranker.
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

# 5. Select the shipping bank and score it.
python3 scripts/select_kit_bank.py \
  --manifest kits_all/kit_manifest.json --source-dir kits_all \
  --out kits_ship --per-style 150
PYTHONPATH=. python3 scripts/score_kits.py \
  --dataset /tmp/wubtrain/taste_dataset.json \
  --manifest kits_ship/kit_manifest.json \
  --model /tmp/wubtrain/taste_model.pt \
  --ranker /tmp/wubtrain/taste_ranker.pt \
  --out frontend/public/data/kits/taste_scores.json

# 6. Replace the shipped bank and add nebula PCA coordinates.
# cp does not delete stale kit dirs; remove the old kits/ tree when shrinking.
cp kits_ship/kit_manifest.json frontend/public/data/kits/kit_manifest.json
cp -r kits_ship/kits/. frontend/public/data/kits/kits/
python3 scripts/add_kit_coords.py

# 7. QC (optional but recommended before shipping).
python3 scripts/kit_reconstruct.py --mode compare \
  --manifest frontend/public/data/kits/kit_manifest.json \
  --kits-dir frontend/public/data/kits \
  --stems "$STEMS" --processed "$PROCESSED" \
  --fail-on-error --report /tmp/kits_qc/kit_qc.json
```

### Legacy Standalone Client Bundle (retired)
The old slice/VQ bundle is no longer on the shipped path. Do not rebuild it
unless explicitly testing a fallback:
```bash
PYTHONPATH=. python3 scripts/export_standalone_client_bundle.py
```

`scripts/retrain_all.sh` and `scripts/continuous_trainer.py` still rebuild the
retired slice/vocabulary/arranger pipeline. Use the kit sequence above instead.

### Launching Local Servers
```bash
# Start FastAPI backend server on port 8123
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8123

# Start Vite dev server for frontend
cd frontend && npm run dev
```

### Git Commit & Deployment
```bash
# Stage files
git add -A

# Commit (triggers pre-commit hook auto-incrementing build number)
git commit -m "feat: Description of changes"

# Push to origin (triggers GitHub Actions deployment to GitHub Pages)
git push origin main
```

---

## 🎧 Musical Arrangement Cheat Sheet

> [!NOTE]
> The shipped kit engine does not synthesise these step patterns. It uses the
> section names and energy targets to choose real 4-bar phrases. The per-step
> table below documents the **legacy symbolic arranger** (fallback only).

When adjusting legacy groove templates or generating symbolic patterns:

| Section | Duration | Energy | Kick Pattern | Bass Behavior | Synths & Leads | Vocals |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`INTRO`** | 8 bars | 0.25 | Downbeats only (0, 4, 8, 12), filtered | Completely muted | Ethereal floating chord pads | Ambient vocal calls ("Ooh..."), sustained pads on bars 0 & 4 |
| **`GROOVE_A`** | 16 bars | 0.83 | 4-on-the-floor + ghost on step 10 | Pumping offbeat sub bass | Call-and-response chord stabs | Lyrical narrative verses with Call & Response phrasing |
| **`BUILDUP`** | 8 bars | 0.65 | Pulse on 0, 4, 8, 12 (bars 1–5 only) | Cuts out on bar 7 | Rising saw riser on bar 6 | Rising vocal chants ("Higher..."), stutter rolls, silent breath on step 15 |
| **`DROP`** | 16 bars | 0.90 | Full driving 4/4 with sidechain ducking | Rolling 16th sub bassline | 8-bar melodic arpeggio arc | Anthemic vocal chorus hook singing full melody over driving beats |
| **`BREAKDOWN`**| 8 bars | 0.20 | Completely muted (0 kicks) | Completely muted | Lush polyphonic chord triads | Soulful solo singing melody with rich vowel formants & vibrato |

> [!IMPORTANT]
> On **Bar 7, Step 15** of a `BUILDUP`, step 15 **MUST BE SILENT**. This 1/16th-note breath (the drop gap) creates the critical acoustic contrast that makes the drop hit with maximum impact.
