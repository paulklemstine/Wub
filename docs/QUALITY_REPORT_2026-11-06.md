# Wub quality report — 2026-11-06

Status of the shipped 8-phrase bank and the next quality work.

## Shipped bank

| Item | Value |
| --- | ---: |
| Candidate kits cut | 402 |
| Full-QC failures | 73 phrases / 44 tracks |
| Shipping kits | **370** |
| Shipping phrases | **2,565** |
| Clusters | **16** |
| Bank size | **675 MiB** |
| Avg phrases/kit | 6.93 |
| Clusters with a full 4-phrase run, all stems | **16/16** |

The 44 removed tracks are audited in
`arc8_stage/qc/filtered_kits.json`. **12 strict trim-only recoveries** (78
phrases) were merged after passing the `0.90` correlation / `10 dB` SNR gate;
their audio is now in the shipping bank. Three further strict re-cuts are
staged but held until their new `rNN` phrases have joint features. Three
quiet-window-conditional tracks were not merged because they only pass with a
sub-`-40 dBFS` exemption.

## Models

| Model | Result |
| --- | --- |
| Per-stem judges (drums / bass / other / vocals) | 0.891 / 0.841 / 0.720 / 0.744 AUC |
| Joint ranker, overall | 0.8109 held-out AUC |
| Joint 4-bar / 8-bar / 16-bar | 0.8114 / 0.8187 / 0.7860 |
| Joint thresholds 4 / 8 / 16 | 0.178 / 0.179 / 0.126 |
| Phrases / state / input | 2,565 / 424-d / 2,137-d |

The shipped v3 joint state interleaves four 90-d stem chunks + 64-d mix:
52 phrase features + 4 groove + 2 vocal + 32 timbre per stem. This fixes the
old grouped-layout misalignment and adds onset density, offbeat/backbeat
ratio, syncopation, vocal presence and vocal crest.

The shipped scorer is a **weight-space soup** (alpha 0.5) of two 60-epoch
students: one trained on the 68k-example baseline set and one on a
121k-example deep set with more negative subsets. On a common
track-held-out set, the soup improves 4/8/16-bar AUC from
**0.796 / 0.800 / 0.782** to **0.811 / 0.819 / 0.786**; it also beats either
single run overall (0.811 vs 0.798 and 0.803).

## Runtime quality

- 300 s new-bank soak: **0 errors**, four distinct tracks, master RMS
  0.115–0.279, rotation covered.
- 70 s 370-kit/v4 soak: **0 errors**, **11/11** transitions captured,
  **0 seams**, **0 dips**; the joint ranker loaded **2,565** phrases.
- 300 s 370-kit/soup soak: **0 errors**, **34/35** transitions captured,
  **0 seams**, **4 level dips**, 33 track changes. The dips are the known
  phrase-boundary level-match issue, not clicks.
- Glue bus active: vocal duck on other/bass to **0.841** (−1.5 dB).
- 240 s reference-bank baseline: 24/24 captured, 0 seams, 3 dips,
  p50/p95 0.171/0.401.

The remaining audible issue is an occasional 1.5–2 dB level dip at a
phrase/run boundary, usually when a multi-phrase run advances or a bass/vocal
phrase changes. It is a level-match/phrase-energy issue, not a click.

## WubGotchi reactivity

The bear is now a four-stem meter, not a single master bounce:

- **Eyes = drums + bass** (both eyes share the same bass-waveform iris and
  drum-waveform pupil, exactly as in the earlier builds, plus the kick-driven
  squish and pupil punch).
- **Mouth = vocals** (waveform aperture, syllable punch, vowel geometry).
- **Heart = other** (pad/chord envelope, heartbeat rate, ripples, brightness
  hue shift; no longer driven by the kick).
- **Body, halo, cheeks and ambient motes = full mix** (true post-glue,
  post-limiter master level plus a smoothed master-waveform body contour).

All sources are smoothed with fast attack / musical release envelopes; the
full mix is read from the master analyser rather than a max of the four stems.

## Shipped in this pass

1. **Joint-state layout + groove/vocal features.** Interleaved 90-d per-stem
   chunks, 424-d state, 2,137-d input; retrained on the 370-kit bank.
2. **Pin-region playlist.** The drag commits a partial run to the next block
   boundary, then the scheduler selects a fresh phrase inside the pin region
   each block until the 32-bar release. Probe: 7 distinct phrases / 4 tracks
   across bars 16–40, release exactly at bar 48.
3. **WubGotchi four-stem reactivity.** Eyes=drums/bass, mouth=vocals,
   heart=other, body/halo=full mix.
4. **Recovery audit of the 44 QC-failed tracks.** 15 strict passes were found;
   12 trim-only kits are merged. The three re-cut kits and three quiet-window
   conditionals remain staging-only.
5. **Per-stem judge retrain (40 epochs each, best checkpoint per stem).**
   Vocals 0.706 → **0.744**, bass 0.835 → **0.841**; drums/other were already
   at their ceiling (kept the old checkpoints). The `stem_next` table was
   rescored for all 2,565 shipping phrases; kit-level taste scores are
   unchanged. A joint rebuild with the new timbre embeddings scored
   0.807 vs the shipped soup's 0.811 on the common held-out set — within
   run-to-run soup variance — so the deployed joint ranker/features were
   kept.
