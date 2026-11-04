# Song Quality Metric — definitions, weights, calibration

> [!NOTE]
> **Scope:** this document describes the reference-free metric for the **legacy
> symbolic 16-step scheduler** (`wub_core/song_quality.py`). The shipped kit
> engine does not synthesise step grids and does not call this scorer; its
> quality signal is the learned taste model documented below and in
> [`KIT_ENGINE.md`](KIT_ENGINE.md). Keep this metric for legacy/offline
> comparisons, not as a gate on kit playback.

`wub_core/song_quality.py` scores symbolic arrangements without listening:
**groove 25 / harmony 25 / arrangement 20 / repetition 15 / mix 15 → 0–100.**

## Sub-metrics

- **Groove**: kick downbeats `{0,4,8,12}` present in DROP/GROOVE_A (50%);
  bass onsets on offbeats `{2,6,10,14}` (30%, resting bass scores 1.0);
  same-section drum-grid continuity via `generation_eval.groove_discontinuity` (20%).
- **Harmony**: per-step semitone rubs between simultaneously-sounding pitched
  slices and chord tones (50%); chord-tone + scale membership (30%);
  stepwise key motion within a section (20%).
- **Arrangement**: section mean energy vs targets
  (INTRO .25 / GROOVE_A .83 / BUILDUP .65 / DROP .90 / BREAKDOWN .20) (45%);
  above-average drum density on section-final bars = fills (30%);
  section durations vs mined means, gaussian falloff (25%).
- **Repetition**: exact bar repeats at lag 4/8 inside `[0.15, 0.6]`
  (below = random mush, above = stuck loop).
- **Mix**: per-stem onset-velocity balance vs drums 1.0 / bass .85 /
  other .75 / vocals .70 targets (55%); transient pileup
  (≥3 stems at vel≥0.9 on one step) penalty (20%); spectral smoothness —
  mean sounding-centroid inside 600Hz–4.5kHz, falling off toward dull
  (<200Hz) or harsh (>9kHz) (25%).

`score_bar()` (groove+harmony+mix only) was the live-rerank entry point for the
legacy symbolic scheduler. The kit engine does not call it.
`scripts/eval_song_quality.py` samples seeded songs and prints one number per
build into `data/song_quality_eval.json`.

## Known limitations (v1)

- Chord roots come from a real mined minor progression cycled every 4
  bars (like the frontend), with the song's tonic detected from its
  pitch histogram — fair across keys, no bass-line cheating.
- The eval mirrors the legacy symbolic scheduler: section density bands and
  kick-floor downbeats in the draw, section-based velocities.
  Anything the scheduler does that the eval doesn't mirror is a
  blind spot; keep the two in sync.
- Prototypes carry no exemplar velocities, so eval uses section-based
  velocities mirroring the legacy frontend scheduler (`sec DROP→0.95` etc.).
- No audio is rendered: timbre subtleties, reverb tails, and mastering
  glue are invisible to this metric. Human ears remain the ground truth.

## Calibration protocol (required before using the score for generation)

1. Curate a blind listening set: render pairs of continuations from the
   same seed (old renderer vs candidate), loudness-matched, order shuffled.
2. For each pair, score both variants with `score_song` on the exact
   played bars.
3. Agreement = fraction where the metric prefers the human winner.
   Target: **≥70%**. Below that, adjust `WEIGHTS`/sub-metrics and repeat —
   do NOT wire the metric into generation (rerank, hill-climb) until then.
4. Re-run calibration after any change to the legacy scheduler or vocabulary
   or the weights: the metric must track ears, not replace them.

## Cross-song swap test (legacy vocabulary diagnostic)

> [!NOTE]
> This test targets the retired global slice vocabulary. It is not a gate for
> the shipped kit bank, which never swaps slices across songs.

`scripts/eval_cross_song.py` rebuilds each stemmed track using ONLY slices
from *other* tracks (leave-one-song-out remap) and reports per-stem
fidelity vs the self-vocab rebuild. A vocabulary that can re-voice songs
it never contributed slices to is reusable by construction; one that can
only rebuild its donors is a filing cabinet. Track this gap per commit.

---

## Kit engine / taste metrics (shipped)

The shipped signal is **not** an audio score; it is a learned ranking of real
analysis windows. `scripts/score_kits.py` writes these fields to
`frontend/public/data/kits/taste_scores.json` and a full report to
`/tmp/taste_report.json`.

- **Window features:** 48-d vectors from analysis JSON only — chroma, timbre,
  per-stem onset density and loudness, energy and energy slope, section one-hot,
  time since section start, and vocal presence. See `KIT_ENGINE.md` for the
  exact layout.
- **Arrangement classifier:** real 8-window sequences vs shuffle, same-style,
  cross-track, and energy-jump corruptions. Report `val_auc` is this model's
  held-out real-vs-corrupt AUC (0.845 shipped); real-vs-shuffle AUC is 0.956.
- **Pairwise transition ranker:** context Transformer over 1–6 windows plus a
  candidate MLP, trained with InfoNCE against same-track/same-style negatives.
  It produces the per-kit `transition` matrix, `quality` vector, and `next`
  list used by `KitPlayer`.
- **Within-kit top-1/top-3:** for rows whose true `+4-bar` successor is another
  phrase in the same kit, does the matrix rank it first / top 3? Shipped:
  **59.0% / 87.0%** (196/332) with the 60-epoch ranker; 20 epochs was 51.5%.
- **Full-track top-1/top-5:** rank every aligned window of the held-out track
  for each context. Shipped: **23.8% / 74.3%** with 60 epochs/context 6; 20
  epochs was 19.3% top-1. Hard negatives (22.2%) and context 8 (23.0%) were
  worse.
- **Matrix sharpness:** median row max is tuned to **≈0.52** (0.52 shipped)
  with diagonal zeroed; this is the difference between a selector and a
  uniform or one-hot matrix.
- **Quality:** each phrase is scored as the successor of its own real
  aligned-window context, z-scored within style, then sigmoid-squashed. Compare
  quality within a style only; it is a ranking signal, not a 0–100 review.

Do not use `score_song` to accept or reject kit phrases: it evaluates symbolic
step grids the kit engine never plays. Use the ranker metrics plus
`scripts/kit_reconstruct.py --mode compare` (audio QC vs source stems) and
human listening.
