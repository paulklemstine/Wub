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
