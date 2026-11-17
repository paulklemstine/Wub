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
changes or normalizes the source assets.

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
Cluster changes pass through the same final gate, preserving old anchors while
new material arrives. Without a valid sidecar the original arc/bridge behavior
remains. The existing stretch limits, voice pool, pins, lazy cache, and audio-clock
UI queue remain in force.

## Training and evaluation

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
successfully scheduled runs: source IDs, cut ranges, gains, pitch/rate, output bar
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
No preference is invented when there is no human feedback. Inspect the exported
model's report for actual corpus counts, split IDs, proxy errors and feedback
counts. The general sidecar builder preserves the new sidecar flag when present.
