# Song Quality Metric — definitions, weights, calibration

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
  other .75 / vocals .70 targets (70%); transient pileup
  (≥3 stems at vel≥0.9 on one step) penalty (30%).

`score_bar()` (groove+harmony+mix only) is the live-rerank entry point.
`scripts/eval_song_quality.py` samples seeded songs and prints one number
per build into `data/song_quality_eval.json`.

## Known limitations (v1)

- Chord roots in eval are derived from the bass line itself, so harmony
  scores skew high for everything; relative deltas still carry signal.
- Prototypes carry no exemplar velocities, so eval uses section-based
  velocities mirroring the frontend scheduler (`sec DROP→0.95` etc.).
- No audio is rendered: timbre, reverb tails, and mastering glue are
  invisible to this metric. Human ears remain the ground truth.

## Calibration protocol (required before using the score for generation)

1. Generate blind A/B pairs (transport rig), collect ≥20 non-skipped votes.
2. For each pair, score both variants with `score_song` on the exact
   played bars (log `step_slice_ids` + sections per pair for this).
3. Agreement = fraction where the metric prefers the human winner.
   Target: **≥70%**. Below that, adjust `WEIGHTS`/sub-metrics and repeat —
   do NOT wire the metric into generation (rerank, hill-climb) until then.
4. Re-run calibration after any change to the scheduler, the vocab, or
   the weights: the metric must track ears, not replace them.

## From votes to training

Since votes now carry the played 6-token bars per variant
(`bars: {new: [...], old: [...]}` in `data/ab_results.json`),
`scripts/finetune_from_ab.py` performs winner-weighted supervised
fine-tuning (filtered behavior cloning: train on winners, ignore losers,
early-stop on winner-val perplexity). This is deliberately NOT full
RLHF/DPO: live play has no shared prompt prefix to pair over, so the
honest objective is likelihood of human-approved continuations.
Run it only after a voting round with >=4 usable winner sequences;
it refuses to train on fewer (avoids overfitting to a single session).
