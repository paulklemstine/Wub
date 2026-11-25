# Training from reticule preferences

The app now logs implicit preferences from the nebula gestures:

- `onReticleDrop` → `StemRemixPlayer.setTarget()` logs a **`retarget`** when you
  move a stem to different territory (`from` = outgoing phrase, `to` = snapped
  target phrase), or a **`hold`** when you drop on the current phrase.
- Double-click / `release()` logs a **`release`** (negative for the released
  phrase).
- A long pin (32 bars) is a strong "keep this" signal; repeated drags to the
  same phrase/region increase that phrase's weight.

Each event captures both phrase ids, tracks, section, style, the raw drop
coordinates, whether it snapped, and the commit bar. The log is capped at 400
events and exposed as:

```js
window.__wubPreferenceLog()      // inspect
window.__wubPreferenceExport()   // download wub-preferences.json
```

## Fine-tune

After a few sessions, copy the downloaded file to
`data/user_preferences.json` and run:

```bash
PYTHONPATH=scripts:. python3 scripts/fine_tune_preferences.py \
  --preferences data/user_preferences.json \
  --ref frontend/public/data/kits/bank_transition.json \
  --critic frontend/public/data/kits/song_critic.json \
  --out frontend/public/data/kits/bank_transition.json \
  --mode pg --augment 3 --epochs 6
```

Because a session has few gestures, each event is augmented with same-section
alternatives and the audio-derived critic weights the pairs. Policy gradient
is the default because DPO regressed the shipped supervised policy in corpus
tests. The output keeps the exact `bank_transition.json` browser format and
mined top-K table, so no frontend code changes are needed.

A `release` event has no known replacement phrase, so the adapter treats the
released phrase as the rejected action and samples an in-section alternative
as the chosen one; retarget events use the dropped phrase as chosen and a
same-section alternative as rejected. When you have explicit "that sounded
terrible" moments, a separate negative log or button would give cleaner labels.
