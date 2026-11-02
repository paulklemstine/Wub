# Blind A/B Rig Spec — Learned Fills / Groove Pocket (1 page)

## Goal
Decide by ear whether the new fill/pocket rendering beats the current
renderer, without bias. Same seed, same section, loudness-matched, hidden
labels; results feed back into `data/pattern_audio_feedback.json`.

## Trial generation (offline script, not implemented here)
1. Pick N pairs: fixed `{seed, section, bpm}` per pair (e.g. 8 pairs across
   INTRO / GROOVE_A / BUILDUP / DROP / BREAKDOWN).
2. Render each pair twice from the same seed: `A = old renderer`
   (current `composeBarPatterns` + schedule path, no nudges, no loudness
   mapping), `B = new renderer` (`wub_core/fill_pocket_spec.py` semantics:
   `detect_fill_bar` routing, `pocket_nudge` offsets, `humanize_velocity`
   ±8%, `loudness_normalize_gain` @ −14 LUFS target).
3. **Loudness-match** both files to −14 LUFS integrated (measure, then apply
   a single static gain per file) so "louder" can't win.
4. Randomize order per pair, store the secret mapping server-side only.

## UI (add to existing frontend; no implementation here)
- Transport: play X / play Y (loopable 4–8 bar excerpt), seamless toggle.
- Buttons: `Prefer X` / `Prefer Y` / `Skip`. Labels X/Y reshuffled per pair;
  never show "old"/"new", commit hashes, or render settings.
- Progress: `Pair k of N`. Optional comment box per pair.

## Result JSON schema (one record per judged pair)
```json
{
  "pair_id": "ab-2026-001",
  "winner": "X",
  "skipped": false,
  "seed": 42,
  "section": "DROP",
  "comment": "tighter hats",
  "created_at": 1730419200.0
}
```
- `pair_id`: string, unique per pair. `winner`: `"X"` | `"Y"`.
- `skipped`: bool (true ⇒ `winner` ignored). `seed`: int used for both renders.
- `section`: one of INTRO | GROOVE_A | BUILDUP | DROP | BREAKDOWN.
- `comment`: optional free text. `created_at`: unix epoch float.

## Retrain hook (into `data/pattern_audio_feedback.json`)
- After reveal, map X/Y → old/new via the secret mapping.
- On a "new" win: append `{"ab_wins": +1}` credit to the winning pair's
  `(stem, groove_type)` entries under `grooves`, and +0.05 (clamped ≤ 1.0)
  to participating `slices` scores — same EMA spirit as
  `PatternAudioFeedbackEngine.record_metrics`, never overwriting acoustic
  similarity scores, only adding an `ab` provenance field.
- On an "old" win or skip: record the pair outcome only
  (`ab_losses` / `ab_skips` counters); do not penalize slice scores.
- Aggregate stop rule: ship the new renderer when it wins ≥ 60% of
  non-skipped pairs with N ≥ 20; otherwise keep iterating on priors.

## Open questions
- Who judges (one producer vs. panel) and do we weight judges?
- Exact excerpt length (4 vs. 8 bars) and headphone vs. monitor policy?
- Store secret mapping + audio where (gitignored `data/ab/`?)?
