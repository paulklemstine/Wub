# Quality telemetry

Run `window.__wubQualitySummary()` in the browser after a few minutes of play.
It returns five measured dimensions plus an overall score and the weakest one:

```js
{
  harmony:    { p10, p50, clashRate, meanRootSpread },
  arrangement:{ sectionChanges, uniqueSections, arcDeviation, meanEnergySlope },
  churn:      { uniqueTracks, uniquePhrases, changes, repeatRate },
  seams:      { dipEvents, seamEvents, p95Jump },
  mix:        { gainSpread, duckSpread, dominantStem },
  score:      0..1,
  weakest:    "harmony" | "arrangement" | "churn" | "seams" | "mix"
}
```

- **harmony** — worst pairwise chroma cosine between bass/other/vocals per bar
  and root spread; the floor comes from `harmony_continuity.json`.
- **arrangement** — section changes, unique sections, and mean deviation of the
  played energy from the (sampled or fixed) arc target.
- **churn** — unique tracks/phrases in recent per-stem history and repeat rate.
- **seams** — RMS dips and max sample jump at block boundaries from
  `TransitionProbe`.
- **mix** — spread of glue gains and which stem dominates.

The score is diagnostic, not a controller: no scheduling decision reads it.
Use it to decide which knob to tune. If `weakest` is `harmony`, strengthen the
quartet key/chord fit; if `churn`, adjust variety/motif weights; if `seams`,
work on fades and level matching; if `mix`, adjust glue/ducking; if
`arrangement`, adjust the arc energy blend and section policy.

This is the first step of rebuilding quality systematically: measure before
changing another weight.

### Harmony-reward policy gradient (evaluated, not shipped)

`scripts/fine_tune_harmony_policy.py` fine-tunes `bank_transition.json` with
policy gradient where the reward is the browser's per-bar quartet harmony fit
(1,193 harmony-aligned pairs, 400 tracks, KL 0.02). Offline validation stayed
at 1.000 preference accuracy and AUC 0.9898, but a headless 60 s run after
shipping it measured worse than the pre-PG policy: harmony p10 0.28–0.34 vs
0.30–0.48, and unique tracks collapsed to 1 vs 2→7. The reward over-favored a
single coherent source. The script remains available as an opt-in experiment;
the shipped artifact was reverted to the pre-PG policy.

### Multi-objective reward (evaluated, not shipped)

`scripts/fine_tune_multi_objective.py` adds harmony + novelty + arc reward with
an explicit low-reward "repeat current phrase" action and KL to the reference.
Conservative settings (harmony 0.85 / novelty 0.08 / arc 0.07, KL 0.08, 3
epochs) improved held-out preference accuracy 0.9858 → 0.9905, but a 60 s
headless run measured harmony p10 0.18–0.34 and clash 0.25–1.0, versus
0.30–0.48 and 0–0.5 for the shipped hand-weighted policy. The offline metric
does not transfer to the live mix; the script stays opt-in and the shipped
artifact was reverted.

### In-the-loop policy search

`scripts/headless_quality_search.py` serves candidate `bank_transition.json`
files to the built app in headless Chrome, plays a set for each, samples
`window.__wubQualitySummary()`, and ranks candidates by a live objective:

```
0.45*harmony_p10/0.6 - 0.25*clashRate - 0.25*arcDeviation/0.5
+ 0.25*uniqueTracks/10 - 0.15*dipEvents/5
```

It restores the original artifact when done, so the checkout is unchanged.
A five-way blend search over a multi-objective PG variant initially favored a
50% blend, but two repeated runs per variant showed that result was noise:
baseline averaged 0.337 vs 0.234 for the blend, with large anchor-dependent
variance. The shipped policy was therefore left unchanged. The harness remains
the right tool for evaluating future policies against the real mix.

### Learned phrase features

`scripts/train_phrase_embeddings.py` trains a contrastive 64-d phrase encoder
from the shipped per-bar descriptors (purity, essentiality, roots, chroma,
head/tail, section, style, tempo, key, energy). Positive pairs are real
consecutive phrases from the training songs; negatives are same-section
phrases from other songs. `scripts/train_bank_transition.py --embeddings`
retrains the browser ranker on these learned features instead of the older
taste embedding.

Offline AUC improved 0.9898 → 0.9932. Two repeated headless A/B runs each
favored the learned-feature policy: mean live objective 0.147 vs 0.006,
harmony p10 0.39–0.42 vs 0.33–0.39. The artifact was shipped with the exact
same browser format; only the weights and top-K successor table changed.

### Learned embeddings in the remaining heads

`phrase_conductor` was retrained on the contrastive embeddings: headless A/B
mean objective 0.225 vs 0.081 for the taste-embedding version, so it was
shipped. `song_critic` was also retrained with a 282-d input (218 analysis +
64 phrase embedding), but two headless runs each showed a slight live
regression (mean objective −0.023 vs +0.049), so its artifact was reverted.
The browser keeps 282-d support and the `phrase_embeddings` sidecar for future
experiments, but the shipped critic remains the 218-d audio-derived model.

### Tighter style-copy feedback loop (Nov 20, 2026)

`scripts/style_copy_metrics.py` compares a generated session's section,
duration, and stem-change cohort distributions against the analyzed songs and
the 61k-boundary `transition_cohort.json`. The headless harness now reports
`style_fidelity` alongside the live quality objective and ranks candidates by
their combination. Adding cohort divergence made the metric discriminate
between policies (0.35–0.47 instead of a constant ~0.48).

A four-repeat search over the shipped learned-embedding policy and the older
pre-PG policy ended in a statistical tie (mean combined 0.073 vs 0.072), with
large anchor-dependent variance. No policy change was shipped; the loop itself
is now the reliable artifact for future training runs.

### History-aware sequence conductor (evaluated, opt-in)

`scripts/train_sequence_conductor.py` trains on 799 real phrase trajectories
from 413 tracks, using the last six phrases to predict the next phrase
embedding, stem-change mask, section and energy. Browser inference is shipped
as `sequence_conductor.json` but disabled by default (`sidecars.json` flag
`sequence_conductor: false`). Four paired headless runs showed no reliable
gain: mean combined objective 0.172 with the model vs 0.167 without. This
confirms that phrase-level history alone does not improve the assembled mix
beyond the current rankers; the next step must change the objective or use
real listener preferences, not more history.
