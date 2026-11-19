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

## Vocal AI as a ranking layer

The shipped vocal path is the pre-development one: silent bars inside the primary
phrase are repaired by cycling the phrase's own active bars, and the section hook
loops over two bars. The two learned vocal artifacts only *rank* inside that
path.

- `vocal_bar_model.json` (`wub-vocal-bar-v1`, 120-d window → 29-d next-bar
  activity/chroma/onset head, plus a run-length head) scores the phrase's
  existing active bars. The repair keeps the same active-bar set and the same
  `b % ordered.length` cadence; only the order of those bars changes. If the
  model is missing or malformed the order is the historical
  `activeSrcBars[b % activeSrcBars.length]`.
- `vocal_phrases.json` (`wub-vocal-phrases-v1`) mines contiguous multi-bar vocal
  performances per track and scores recurrence/repeat. `chooseVocalPrimary` adds
  a small bonus to phrases that belong to such a unit; it never remaps bars,
  never forces a wrap, and never overrides the pinned/recall paths.

Both sidecars are optional (`sidecars.json` flags `vocal_bar_model` and
`vocal_phrases`); a failed fetch logs nothing and leaves the known-good
scheduling intact.

## Learned stem cohorts at section boundaries

`transition_cohort.json` is mined by `scripts/build_transition_cohort.py` from
the real 4-stem songs in `data/processed` (1,891 tracks / 61,118 four-bar
boundaries), using `wub_core.section_arcs` for section labels and 4-bar onset
grids for stem change detection. It replaces the old "1-2 stems at every
boundary" assumption with the measured distribution:

- inside a section: 2 stems change at 26% of boundaries, 3+ at 39%;
- at a section change: 3+ stems change at **50%** of boundaries, with
  `drums + bass + other + vocals` (21%) and `bass + other + vocals` (20%) the
  two most common cohorts; drums hold at 67%, vocals change at 77%;
- harmony barely moves at these seams (chroma cosine p10 = 0.895), so a wider
  handoff must keep key/chroma fit — it is not a license for a key jump.

The artifact is optional (`sidecars.json` flag `transition_cohort`). When it is
present, `movementBudgetAt(sessionBar, bars, sectionChanged, cap)` and the
final `SliceFlow.select()` gate permit up to 3 changed stems **only** at a
section change; inside sections and on the bridge the historical 1-2 / 0 cap
is unchanged. `selectJoint()` additionally enumerates 3-stem cohorts at those
boundaries under the same ≤2-stems-per-source recombination cap, and gives a
small centered tie-break toward the mined cohort probabilities. No audio is
regenerated, repitched or resliced; the judge/ranker still decides which
phrase each stem plays. With the sidecar missing or disabled, behavior is
byte-for-byte the previous 1-2 stem budget.

## Next-bar pattern heads and quartet harmony floors

Two more sidecars are learned from the real corpus (`scripts/build_bar_models.py`,
track-disjoint 90/10, 254,927 bars from `data/processed/*.json`):

- `pattern_bar_model.json` — three 55→64→48 tanh heads (drums, bass, other)
  predict the *next bar's* onset grid, density, downbeats, quarters, root and
  chroma from the current bar plus section/energy context. Validation MSE beats
  the mean baseline for every stem (drums 0.034 vs 0.082, bass 0.041 vs 0.061,
  other 0.039 vs 0.059). `BarComposer.pickBestCellForBar()` adds a small cosine
  fit term (**≤ ±0.20**) to its existing bar score; it never changes the donor
  phrase, key shift or timing.
- `harmony_continuity.json` — real-song chroma floors: the worst pairwise
  chroma cosine between pitched stems in one bar has p10 = 0.655, and the
  per-stem chroma cosine across a 4-bar boundary has p10 = 0.898. Candidate
  mixes below those floors receive a bounded ranking penalty (≤ 0.17 from the
  per-bar floor and ≤ 0.11 from a broken tail) in `selectJoint()`. The judged
  phrase, stretch rate and pitch shifts are untouched.

The browser needs true per-bar chroma for these checks, which `cell_index.json`
never shipped: `scripts/build_cell_index.py --emit-chroma` now writes
`cell_chroma.json` (3,670 phrases × 3 pitched stems × 4 bars × 12 bins) from
the same audio and DSP used for the index; the rebuilt `cell_index.json` is
byte-identical to the shipped one. All three sidecars are optional
(`sidecars.json` flags `cell_chroma`, `pattern_bar_model`,
`harmony_continuity`); without them the learned terms are exactly neutral.

## Learned set arc and rare full-source takeovers

`song_arc_model.json` (mined by `scripts/build_song_arc.py` from 1,895 real
songs) records section transition probabilities, duration percentiles, section
energy and energy slope. `learned_arc.ts` samples a full 104-bar arc from those
tables with a deterministic per-set seed (cluster + BPM). Crucially, the
sampled energy is blended **after** `SongArchitect.describeBlock()` — the
architect's fixed blueprint was overwriting the learned navigator/arc-model
energy and was the main source of "the drop feels like the previous section".
Real slopes are used as intended: intros and buildups rise, breakdowns fade.
The fixed `ARC_SEGMENTS` table and `targetAt()` are untouched; the sidecar
(`song_arc_model` flag) is optional.

`transition_cohort.json` now also carries a `handoff` policy: at a section
change, a deterministic hash gives a **10% chance** of a rare all-in moment
where up to four stems move to one source track. The old ≤2-stems-per-source
recombination cap remains everywhere else. During a takeover the section
budget is raised to four and the final `SliceFlow` gate is told to accept four;
the joint ranker's raw score is discounted (0.94) because it was trained only
on 1–2 stem moves. This directly targets the reported "mashup feel" without
turning the set into a cover of one record.

## Motif calendar and vocal release cadence

`scripts/build_motif_calendar.py` mines how often a 4-bar pattern returns after
8/16/32 bars and how often each stem introduces fresh material, from the same
1,895 real songs. The shipped `motif_calendar.json` says drums return at 8 bars
**51%** of the time (vs. bass 18%, other 15%, vocals 8%), while novelty rates run
from drums 41% to vocals 90%. A small learned prior now rewards a candidate
that brings a phrase back at its natural lag **or** introduces fresh material
when that stem has not done so lately — addressing both halves of "the groove
goes nowhere".

`vocal_cadence.json` mines stable vocal pattern runs (84% last only 4 bars) and
boundary change rates (73% inside a section, 77% at section changes).
`vocalChangeAllowed()` gained an optional learned 8-bar release budget instead
of the fixed 16-bar hold; section changes still always release, and 3-argument
callers keep the old 16-bar behavior. Both sidecars are optional and neutral
when absent.

## Seam level polish

`scripts/build_level_seam.py` measures the RMS of the first and last 20 ms of
every shipped phrase/stem (3,670 phrases each) and exports a bounded per-stem
ratio (`level_seam.json`). The engine already starts an incoming voice at the
outgoing voice's gain; this sidecar nudges only that *start* gain (drums 1.12,
bass 0.995, other 1.021, vocals 0.936) so the equal-power crossfade begins at
an acoustically equal level, then settles to the phrase's own gain. It is a
small polish term, not a new limiter, and is fully neutral when the sidecar is
absent.

## Vocal tuning pass (listener feedback: "vocal still stuck")

The corpus says stable vocal *patterns* last only 4 bars in 84% of runs and
change at 73% of boundaries. The first pass still held a phrase too long, so
the learned release budget is now 4 bars (not 8/16), the section-change vocal
bonus is 0.116, and three rank biases were softened: the current-phrase bonus
in `chooseVocalPrimary` 0.20 → 0.05, the mined-unit prior 0.15+0.20·score →
0.06+0.10·score, and the long-segment bonus in `scoreTransition` 0.15 → 0.08.
Theme recall still wins outright when a hook is deliberately recalled.

## Unified short-horizon planner

`horizon_planner.json` is a small policy config (not a new acoustic model) that
turns three already-learned signals into one **additive ranking bias**:

1. **bar** — `pattern_bar_model` fits a candidate phrase's best bar to the
   current block tail. When enabled, `buildContiguous4BarCells()` ranks the
   phrase's *own* four bars per output slot (each used once) instead of
   blindly taking `b % 4`. No new bar is introduced, no cross-song splice.
2. **phrase** — the candidate's learned `earQ` / taste quality.
3. **sequence** — fit to the corpus-mined arc energy 2–3 blocks ahead, blended
   with the learned section-cohort prior.

The combined bias is clamped to ±0.18 and added to the existing policy score;
the joint ranker, movement budget, pins, source cap and final SliceFlow gate
still make every hard decision. With the sidecar absent, bar selection returns
to contiguous `b % 4` and the score is exactly zero.

### Phrase-freshness tuning

Listener feedback: high-`earQ` phrases were being replayed too often. The phrase
horizon term now blends taste quality with recency (`0.6·earQ + 0.4·freshness`,
where a phrase unseen in this stem's 32-entry history is fully fresh), and its
weight dropped from 0.25 to 0.12 with the budget moved to bar (0.38) and
sequence (0.50). `scripts/build_horizon_planner.py` reproduces the config.

## Variety prior (track and vocal repetition)

The selector was collapsing variety for three measurable reasons: per-stem judge
top-1 scores are saturated (drums 0.982, bass 0.984, vocals 0.961), `earQ` is
nearly constant outside bass, and `SongArchitect` latched one chorus/verse hook
per set and replayed it in every sub-block. `variety_prior.json` adds bounded,
optional biases without changing any hard rule:

- recently used **source tracks** are penalized in the joint shortlist, the
  final joint score and the fallback `selectMorph` path; the never-played
  reserve grows from 2 to 4 slots;
- the next-bar phrase selector gets a session **freshness** callback
  (`0.6/(1+plays·0.5) + 0.4·age/12`), weighted 0.30;
- theme recall is budgeted to **one chorus and one verse recall per set**, at
  the first sub-block only (previously repeated through every sub-block);
- the final SliceFlow gate's `sourceNovelty` weight rises from 0.15 to 0.30.

Everything is neutral when the sidecar is absent; no phrase, bar, stretch rate
or slice boundary is changed by this prior.

### Variety tuning pass (listener feedback: "too scattered")

The first variety weights broke the sense of one song. They are pulled back to
half strength: track-reuse penalty 0.18 → 0.08, vocal freshness 0.30 → 0.12,
fallback morph multiplier 4 → 2, reserve 4 → 3, SliceFlow source novelty
0.30 → 0.20. Theme recall budget is 2 → both the first CHORUS_1 block and the
CHORUS_2 block may recall the hook, but sub-block replay stays disabled. The
two real fixes (no repeated hook in every sub-block, some vocal freshness)
remain.

## 4/8/16-bar sung sections and their backing

`scripts/build_vocal_section_model.py` combines the 1,620 mined multi-bar vocal
units with direct 4-bar activity detection and the corpus section labels.
It learns, per section category (CHORUS / VERSE / BUILDUP / BREAKDOWN):

- sung-section length distribution in **4 / 8 / 16 bars** (4-bar sections now
  measured directly because the unit miner only emits 8+ bar chains),
- probability that drums / bass / other change when the sung section starts,
- which backing stems are present under it.

In the corpus, a new chorus vocal changes bass 55% and other 58% of the time but
drums only 9%; a verse changes bass 61% / other 64% / drums 10%. The shipped
`vocal_section_model.json` feeds two bounded biases: `chooseVocalPrimary()` gets
a bonus when a candidate unit's length fits the section (short sections win in
breakdowns, 16-bar units slightly favoured in choruses), and `selectJoint()`
nudges bass/other to move with a new vocal section while drums hold. The
sidecar is optional and changes no bars, pitches, rates or slice boundaries.

## Learned conductor

`conductor.json` (a 392→128→128→202 MLP trained on 57,347 real-song boundaries,
track-disjoint) now supplies the next section, energy, stem-change mask and
per-stem next-bar patterns at every block. With `authority: 1.0` it replaces the
hand-written section/energy arbitration and drives bar choice and stem-change
ranking; the curated rules remain the fallback and all hard gates stay in
force. See docs/CONDUCTOR.md for the training recipe, metrics and rollback knob.

## Learning from the #300-era pipeline (bank-space transitions)

Comparing build #300 (`73a164e140`) to HEAD showed the older engine used the
same learned models (joint ranker, arc model, section navigator, cell/mix
compat) as **bounded scorers inside the selector**. Newer work added higher-layer
arbiters — most importantly the conductor with `authority: 1.0`, which replaced
the curated arc with a 5-class model that is wrong ~36% of the time. Nothing was
rolled back: the conductor is now shipped with `authority: 0.30`, so it biases
sections/energies/changes/bar choice but no longer overrides the hand-written
arc outright.

The bigger training fix is representational: conductor v2 learned from raw song
cells, while the browser can only choose shipped phrases. `scripts/train_bank_transition.py`
now maps real 4-bar windows to their **actual shipped phrase ids** (2,199 real
consecutive pairs from 428 tracks) and trains a 136→32→1 scorer on phrase
embeddings, section, energy delta and bpm/key distance, plus a mined top-K
successor table. Validation AUC is 0.984. `bank_transition.json` adds a bounded
`±0.075` ranking term to `jointOptions()`, so combinations that actually follow
one another in the training songs are preferred — without changing any audio,
phrase boundary or hard scheduling rule.

## Song-ness critic: sounding like the training songs

`scripts/train_song_critic.py` trains a 218→64→32→1 critic on **58,097 real
four-stem 4-bar windows** from the analyzed songs versus **116,194 cross-song
recombinations** (two stems swapped with a similar key/tempo/section song),
with a 155/17 track-disjoint split. Held-out AUC is 0.643 — a weak but real
signal — so it is deliberately used only as a **bounded adjustment (±0.08)** at
the final `SliceFlow` gate: it lowers the acceptance threshold for a proposed
cut when the assembled quartet looks like real songs, and tightens it when it
does not. It never renders, repitches, retimes or reschedules audio.

This is the first learned term that judges the *whole assembled quartet* rather
than one seam or one head, and it is the foundation for later preference
fine-tuning: real-song versus cross-song labels can be replaced by A/B or
like/skip labels without changing the browser path.

## Bounded final-gate beam search

`beam_search.json` enables a small search at the final cut instead of accepting
the first locally-good proposal. For each stem the gate keeps the candidate cut
plus its held fallback (pins and foundation-protected stems only have one legal
option), enumerates the resulting whole-quartet plans, and ranks them by
existing per-stem priority + quartet `mixScore` + a one-block-ahead viability
estimate from distinct sources and foundation stability, plus the bounded
song-critic adjustment. Depth 3 / width 4 / lookahead 0.25 keeps this tiny — at
most 16 plans per boundary — and every hard gate (pins, foundation, change cap,
critic-bounded veto) still applies. With the sidecar absent the original greedy
final gate is unchanged.

## Audio-level teacher and distilled song critic

`scripts/train_audio_judge.py` is the first model trained on the **actual
waveforms** of the training corpus (WSL box, RTX 4050): it renders 3,976 real
four-stem 4-bar windows from 200 songs at 16 kHz, builds log-mel + per-stem
RMS/centroid features, and trains a 1164→128→64→1 judge against cross-song
recombinations. Held-out AUC **0.774**. The teacher lives at
`data/audio_judge.json` (training-only, not shipped to the browser).

Its labels then train the 218-d browser critic on the *analysis* features the
engine can actually compute (`cell_index` + `cell_chroma`), with real negatives
built from donor song features. Distilled validation AUC **0.705**, up from
0.643 for the hand-labeled analysis critic. The browser artifact
`song_critic.json` has the same format/dims as before (v2), so `song_critic.ts`,
the ±0.08 final-gate adjustment and the bounded beam search are unchanged.

This is the first playback signal learned from audio rather than metadata —
the foundation for training the remaining heads (phrase, bar, cohort) against
the same teacher instead of hand-derived labels.

### Audio-derived reweighting of phrase heads

`scripts/audio_critic_features.py` lets the offline trainers compute the
distilled critic's 218-D features for any bank phrase without decoding audio.
`train_bank_transition.py` and `train_phrase_conductor.py` can now be rerun with
`--critic frontend/public/data/kits/song_critic.json`: real pairs that sound
like songs, and cross-song pairs that fool the critic, get extra training
weight. Both shipped artifacts were retrained this way. Measured validation
was essentially unchanged (bank AUC 0.9838, phrase cosine 0.6426), and an
experiment using critic scores as *soft targets* collapsed bank AUC to 0.53 and
was reverted — the critic is a gate/teacher, not a good direct regression
target for these heads. The audio-derived gain remains concentrated in
`song_critic.json` (AUC 0.705) at the final gate.
