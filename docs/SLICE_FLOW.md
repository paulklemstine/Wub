# Slice development and transition training

The source library remains slices. Original songs provide evidence about
continuity and development; this does not turn playback into full-song streaming.
The target is a recognizable foundation for roughly 60–120 seconds, small musical
developments within it, and earned handoffs between assembled segments.

## Shipped path

`train_slice_flow.py` publishes `slice_flow.json` and `slice_flow_features.f32`.
The bank's 3,670 phrases have 58,720 acoustic stem/bar descriptors (32 floats each).
Features include actual head/tail level and brightness, four spectral bands,
attack/decay/sustain, rhythmic density/accent and fine onset offsets, stereo width,
and chroma. Missing/silent material is represented explicitly. Analysis never
changes or normalizes the source assets. The current 58,720 shipping cells are
mono, so stereo-width columns carry no learned information and are masked rather
than allowed to activate random weights.

Three 32-unit models run at block preparation, not per frame:

* A 108-input acoustic boundary critic scores adjacent final cells, after source
  selection, gain edits and pitch planning. Sub-bar features are approximations
  from their containing source bar; they are not a new live audio neural judge.
* A 16-input quartet critic measures spectral crowding, vocal masking and density
  balance throughout the final arrangement, rather than only its first bar.
* An 8-input hold/vary/handoff classifier learns from chronological full-song
  analyses and contributes a soft change budget. Patient foundation protection
  follows the requested pacing. It does not force exact loops for a minute.

The final gate is **after** `BarComposer`, so a downstream rewrite cannot bypass
it. It compares the composed plan with legal same-source continuations, protects
pinned stems, and limits new primary sources to one per 4-bar block (two on longer
blocks). It permits near-compatible novelty rather than always selecting the
acoustically safest repeat. Small fills/edits remain possible within a foundation.
Accepted cuts become the next boundary and thematic context. Speculative rejected
proposals do not update the composer's tail memory.

With the sidecar present, the 104-bar timing/tempo ladder remains, but the last
8 bars are a learned handoff window rather than mandatory drum/bass removal.
Cluster changes pass through the same final gate. Existing anchors are retained
where the new tempo/key pool permits them. When a tempo rung makes an outgoing
source illegal, the player evaluates a bounded shortlist of legal incoming runs
against the outgoing audio instead of bypassing the critic. It does not stretch
an incompatible old anchor beyond the supported rate to satisfy a change budget. Without a valid sidecar the original arc/bridge behavior
remains. The existing stretch limits, voice pool, pins, lazy cache, and audio-clock
UI queue remain in force.

## Training and evaluation

The complete staged loop trains the acoustic/mix/development heads, motif-return
head and future-flow value head, validates each, and promotes the model bundle
only after every stage passes. Source audio is symlinked read-only into staging;
a failed stage leaves the current shipping bundle untouched.

```bash
PYTHONPATH=. python3 scripts/train_slice_pipeline.py --device cpu
# Optional human feedback and explicitly labelled automatic judgements:
PYTHONPATH=. python3 scripts/train_slice_pipeline.py --device cpu \
  --feedback /path/to/slice-flow-preferences.jsonl --auto-cases 6
```

For just the three base heads:

```bash
PYTHONPATH=. python3 scripts/train_slice_flow.py \
  --kits frontend/public/data/kits --processed data/processed \
  --work /tmp/wub-slice-flow --epochs 35 --device cpu
```

Decoded acoustic extraction is cached. CUDA can be selected explicitly; default
selection uses CPU when GPU free memory is insufficient. Original analyses with
fewer than four separated stems are excluded from the development labels rather
than being treated as four independent instruments.

Track hashes assign train/validation partitions **before donor sampling**. All
donors belong to the same partition as their context. Natural chronology provides
positive evidence; cross-song transitions receive graded acoustic targets and
can be positive. Same-source targeted corruptions prevent a source-identity
shortcut. Constant input columns are masked at export so unobserved runtime
values cannot activate randomly initialized weights.

These are **weak acoustic/development labels**, not human taste ground truth.
High AUC means agreement with those targets. Compare the development classifier
with its majority baseline and require listening comparisons before claiming a
perceptual improvement. Publication rejects excessive acoustic validation error
or a development model below the held-out majority baseline.

## Listening feedback and captured output

The app's **Save session** button exports a bounded `wub-slice-trace-v1` record of
successfully scheduled runs: source IDs, cut ranges, slice and voice-mixer gains, actual common voice rate,
pitch automation, output bar
lengths, timing and boundary feature vectors. Future prepared runs are omitted.
Manual pins cancel superseded trace entries; tempo edits retime them. Failed
loads are traced as the fallback that was actually scheduled. Silence recovery
preserves the original sliced run instead of concatenating its donor list.
No microphone, user audio, or network feedback collection is added.

Create a blind listening pack (Chrome/Chromium and Python `websockets` required):

```bash
PYTHONPATH=. python3 scripts/review_slice_flow.py --cases 6 \
  --out /tmp/wub-slice-flow/review
```

Open `index.html`, compare A/B, and download the JSONL preferences. The clips use
the shipping PCM assembler, Signalsmith worklet, and a master safety stage in
isolated Chrome offline audio contexts. They hold the other stems constant to
make the tested accompaniment handoff attributable. The clips are not a bitwise
capture of the live glue/DJ-FX buses. Fresh isolated rendering contexts avoid WASM
accumulation across a batch. The page stores choices locally; nothing is sent.

Replay the first two blocks of an actual exported session:

```bash
PYTHONPATH=. python3 scripts/review_slice_flow.py \
  --trace /path/to/wub-slice-trace.json --out /tmp/wub-slice-flow/replay
```

Continue training with preferences and/or captured assemblies:

```bash
PYTHONPATH=. python3 scripts/train_slice_flow.py --device cpu \
  --feedback /path/to/slice-flow-preferences.jsonl --traces /path/to/traces
```

Human preferences use a pairwise loss. Both source-track lists and vectors are
validated; held-out tracks cannot enter feedback training. Trace examples use
only training-partition material and reset context across excluded material.
Human preference counts remain zero until real listening choices are imported;
automatic judgements are never reported as human feedback. Inspect the exported
model's report for actual corpus counts, split IDs, proxy errors and feedback
counts. The general sidecar builder preserves the new sidecar flag when present.

## Motif return and the Phase 1 evaluator

`scripts/eval_flow.py` now mines a **motif lag profile** from each real track:
energy, four 16-step onset masks and a 12-bin chroma per bar, compared only
within the same section at lags 1, 2, 4, 8, 16 and 32.  Against 78 real tracks
the return correlation is about +0.34 at 8 bars, +0.22 at 16 and +0.14 at 32,
so the metric can distinguish a groove that returns from constant novelty.  A
trace without onset masks degrades to the coarse Phase 0 shape instead of
fake-materialising structure.

`scripts/train_motif_return.py` trains a 104→32→1 tanh network on the
shipped 32-d cells. Positive pairs use **actual `start_bar` positions** 8/16/32
bars apart. Phrase-list indices cannot establish elapsed musical time. Cross-song
and distant same-song distractors share the same lag/age context, so their labels
cannot be inferred from an invented age. Every context and donor is partitioned
by source track before pairing. The shipped corrected model uses 83,332 pairs
from 500 tracks, with 57 held-out tracks, and obtains approximately 0.851 AUC on
these weak chronology labels. This is not a human listening score.

At runtime the model compares a proposed cell with actually recorded cells at
8/16/32-bar lookbacks. Missing history stays neutral. The most recent foundation
tail is not substituted for a theme supposedly heard a minute ago. Motif/value
bonuses are soft ranking terms and cannot relax the acoustic rejection guard.

## Autonomous listening judgement

`scripts/auto_review.py` can produce additional **model-generated** A/B labels.
It renders through the same shipping PCM assembler, Signalsmith pitch/time
worklet and master safety stage as the listening pack, in isolated Chrome. Four
additional stereo output taps provide the actual rendered stem buses in that
same pass; a surrogate interpolator is not used to invent what the worklet sounds
like. Boundary timing includes the render lead-in.

Feedback vectors belong to the stem that differs between A and B, not the first
unchanged stem. Near-ties are labelled as ties. Short clips with insufficient
8/16/32-bar history have a neutral motif score rather than a fabricated reprise.
Rows carry `origin: "auto-reward-v2"`, tested stem and renderer provenance.
Automatic labels use a weaker training weight than human preferences, and the
model report counts both separately. The staged pipeline rejects automatic
refinement if held-out acoustic MAE regresses beyond `max(0.001, 5%)`; its report
retains both candidate and baseline results instead of declaring self-judgement
a listening win. They are not independent listening
validation and do not establish a listener's taste.

```bash
PYTHONPATH=. python3 scripts/auto_review.py --cases 12 \
  --out /tmp/wub-slice-flow/auto_preferences.jsonl
PYTHONPATH=. python3 scripts/train_slice_pipeline.py --device cpu --auto-cases 12
```

Human listening remains the independent check. Packs have content-derived IDs,
so changing the audio comparison does not silently reuse an older A/B choice
stored in the browser. Training/validation source partitions also apply to the
review-pack donors and any imported preferences.

## Navigating the bar space (the TSP-shaped problem)

A set is a route through a graph whose cities are 4-bar cells and whose edge
costs are learned flow scores.  It is a prize-collecting tour, not an exact
TSP: the planner does not need to visit every cell, but it must avoid recent
cells, limit source changes and hold a foundation for 60–120 s.

`wub_core/bar_tour.py` implements an **offline** beam search with cooldowns,
a source-change budget and an optional value head. The browser final gate uses
the learned value as a soft term; it does not run this beam search on the audio
thread. Cooldowns apply to actual route positions and source changes are counted
between consecutive chosen nodes. Coverage includes proposed visits without
committing them as listening history.

`train_bar_tour_value.py` now uses only consecutive original source bars, never
chains across unshipped gaps or silently removed silent bars. The target is the
discounted reward over the next eight bars, elapsed age is measured in seconds,
and the exported value scale is applied linearly rather than sigmoid-saturating
a regression output. The corrected model uses 25,760 states; held-out MSE is
about 0.350 versus 0.532 for a constant predictor. Constant input columns are
masked in both optional heads, as in the base critic.

```bash
PYTHONPATH=. python3 scripts/train_bar_tour_value.py --device cpu
PYTHONPATH=. python3 scripts/eval_bar_tour.py --tracks 10 --pool 300 --horizon 8
```

The corrected six-track benchmark (`--tracks 6 --pool 80 --horizon 8`) uses
held-out donor tracks throughout: mean proxy flow is about 0.716 greedy versus
0.739 beam. These are comparable routes under one critic, not an independent
perceptual score. A motif bonus is evaluated only when the proposed route has
enough true lookback history. See `docs/reports/slice-flow-validation.json` for
the executed validation summary.

## Whole-spectrum coverage

A tour that only picks locally safe cells eventually mines one style/tempo/
key corner and stops moving the music forward.  `CoverageTracker` counts visits
across five spectrum dimensions — tempo decade, key, style, energy quartile and
source track — and gives under-visited cells an inverse-frequency bonus.  The
offline planner adds `0.22 x novelty` on top of the learned edge score; the
browser gate uses a lighter source-only version (`sourceNovelty`) so a recently
heard record is less likely to be selected again.

On that corrected benchmark, the coverage-aware route reaches about 2.5 source
styles / 3.7 source keys / 16 BPM of source-tempo span versus 1.9 / 2.6 / 11 for
plain beam, at a proxy-flow cost of roughly 0.016. These describe **source-bank
coverage**, not necessarily audible session-key or tempo changes: playback still
fits the source slices to the shared grid. Explicit style selections also constrain
the BPM-ladder handoff; an unrestricted selection still traverses all styles. They are a tradeoff to assess in
listening comparisons, not a reason to force an incompatible transition.

## Live validation

```bash
PYTHONPATH=. python3 scripts/smoke_slice_flow.py --seconds 330
# Force an early handoff for a shorter targeted integration check:
PYTHONPATH=. python3 scripts/smoke_slice_flow.py --seconds 95 --force-rotation
```

These checks run real AudioContext time in an isolated muted Chrome, sample the
actual master analyser, record scheduled slices, and verify exactly two voices
per stem. They capture errors, source-cluster changes, long tasks and sustained
digital-zero intervals. They do not accelerate the audio clock and are not a
substitute for human evaluation of musical phrasing or a full-device FPS test.
