# Mix Compatibility (Per-bar) — Plan & Shipped Status

Scores how well 4 stems (drums/bass/other/vocals) sound together **per bar**, so the engine prefers coherent cross-song recombinations and avoids "terrible combinations" — while creating new mixes (never the original 4 from one source).

**Status:** Phases 0–3 shipped (v0.2.260–v0.2.262). Mix-compat model + onset sidecar deployed; retrained on a 40k MERT render+embed teacher with features refreshed from shipped-audio onset grids.

## Goals (user-selected)
1. **Per-bar** scoring (4 bars/phrase), aggregate **0.6·min + 0.4·mean** so one bad bar pulls the mix down.
2. **Render+embed** offline teacher labels (closeness to real-song mix manifold).
3. **Source cap ≤2** stems per `track_id` in any selected mix (forbid all-4-same) — mixes are recombinations, not near-copies.
4. **Rhythm mismatch** is the top priority (kick-bass/hat/groove lock), with harmony/spectral-balance/energy as supporting axes.
5. **Extra model in browser** at runtime (small MLP, same load/export style as joint ranker).

## What makes stems clash (measured)
- **Rhythm/groove (priority):** per-bar onset-grid correlation/lock, downbeat/offbeat agreement, density & syncopation mismatch (pairwise, esp. drums↔bass).
- **Harmony/tonality:** chroma consonance (bass↔other↔vocals), bass-root interval penalty.
- **Spectral balance/masking:** low-end crowding (drums+bass), vocal buried by other, loudness spread.
- **Energy/arrangement coherence:** per-stem energy mean/std/min/max agreement.

## Shipped components

### Engine guardrail (Phase 0) — `frontend/src/stem_remix.ts`
- Candidate mixes rejected when any source appears in **>2 stems** (never 4-of-a-kind). Repair-aware: if the current mix is over-cap (e.g. anchor opening) only moves that strictly reduce worst overlap are allowed.
- Enforced in joint candidate evaluation/generation and the morph fallback.

### Per-bar onset sidecar (recomputed from shipped kit audio) — `scripts/build_mix_compat_onsets.py`, `frontend/public/data/kits/mix_compat_onsets.{json,f32}`
- 64-step onset/energy grids per phrase×stem (16 steps/bar × 4 bars) via spectral-flux envelopes aligned to each phrase’s 4-bar duration, normalized per grid. **~2.95 MB** for 2884 phrases.
- Recomputed from shipped stems (`frontend/public/data/kits/kits/**.opus`), not copied. Browser loader attaches grids for true inter-stem rhythmic lock at scoring time (falls back to coarse proxies if absent).

### Dataset builder + render+embed teacher (Phase 1) — `scripts/build_mix_compat_dataset.py`
- Samples **real** co-occurring 4-stem phrase windows (positives) and **random cross-song** recombinations (negatives, `source_cap <= 2`), tempo/key-normalized like the engine.
- Per-bar 54-d features (rhythm-first: pairwise onset corr/downbeat/offbeat/density/joint-pulse/sync-gap + aggregates; harmony; spectral balance; energy).
- **Render+embed teacher (`--render --embed`, MERT via `--mert --device cuda` on 4050, spectral fallback):** renders summed per-bar mixes from shipped stems (bar-sliced, tempo-normalized), embeds mixes, builds a **real-mix manifold bank**, grades by closeness to k-nearest real mixes (`teacher_manifold`, cosine mapped to 0..1). Label = **0.55·heuristic_rhythm/min-bar + 0.45·manifold** (reals floored ≥0.90). Rhythm stays primary; manifold = “closeness to real songs”.
- Shipped run: **40k graded** (2884 positives + 37116 negatives), **MERT encoder (cuda)**, bank 2884, k=15.

### Feature refresh (runtime alignment) — `scripts/refresh_mix_compat_features.py`
- Recomputes `features` using the **true shipped-audio onset sidecar** while preserving MERT teacher labels (`label`, `teacher_manifold`, `label_heuristic`), aligning training features with runtime onset grids (phrases carry no onset fields; sidecar is the ground truth). Retrained on the refreshed MERT-graded dataset.

### Critic model + browser integration (Phases 2–3) — `scripts/train_mix_compat.py`, `frontend/src/mix_compat.ts`
- Compact per-bar MLP (54→128→64→2: compat, rhythm), rhythm-emphasized loss; exports browser-ready `frontend/public/data/kits/mix_compat.json` (scaler + flat weights). Aggregates **0.6·min + 0.4·mean** per phrase.
- `stem_remix.ts` loads model + onset sidecar (optional), scores candidate mixes per bar and blends rhythm-first compat into selection with a worst-bar penalty (`bar-clash` / `rhythm-mismatch`). Neutral when unavailable.

## Reproduce / retrain
```bash
# 1) onset sidecar from shipped kit audio
PYTHONPATH=. python3 scripts/build_mix_compat_onsets.py \
  --manifest frontend/public/data/kits/kit_manifest.json \
  --kits-dir frontend/public/data/kits --out frontend/public/data/kits

# 2) 40k dataset with MERT render+embed teacher (4050)
PYTHONPATH=. python3 scripts/build_mix_compat_dataset.py \
  --manifest frontend/public/data/kits/kit_manifest.json --kits-dir frontend/public/data/kits \
  --out /tmp/wubtrain/mix_compat_dataset_40k_mert \
  --positives 2884 --negatives 37116 --render --embed --mert --device cuda \
  --teacher-real-bank 2884 --teacher-limit 40000 --teacher-k 15

# 3) refresh features with true onset grids (align to runtime), then train+export
PYTHONPATH=. python3 scripts/refresh_mix_compat_features.py \
  --dataset /tmp/wubtrain/mix_compat_dataset_40k_mert/mix_compat_dataset.json \
  --manifest frontend/public/data/kits/kit_manifest.json \
  --onsets frontend/public/data/kits/mix_compat_onsets.json \
  --out /tmp/wubtrain/mix_compat_dataset_40k_mert_refreshed
python3 scripts/train_mix_compat.py \
  --dataset /tmp/wubtrain/mix_compat_dataset_40k_mert_refreshed/mix_compat_dataset.json \
  --out /tmp/wubtrain/mix_compat_40k_mert --device cuda
cp /tmp/wubtrain/mix_compat_40k_mert/mix_compat.json frontend/public/data/kits/
```

## Next (Phase 4)
- **Hard-negative mining** from engine/session logs (`bar-clash`/`rhythm-mismatch`, low `mixMinBar`) fed back into the dataset.
- **A/B listening/eval:** compat term on/off, clash rate, diversity (cross-source) held high.
- Optionally reweight rhythm vs manifold, expand teacher set, or add a per-bar onset sidecar derived from alternative sources if needed.
