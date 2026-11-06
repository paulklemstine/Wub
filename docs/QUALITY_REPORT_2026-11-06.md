# Wub quality report — 2026-11-06

Status of the shipped 8-phrase bank and the next quality work.

## Shipped bank

| Item | Value |
| --- | ---: |
| Candidate kits cut | 402 |
| Full-QC failures | 73 phrases / 44 tracks |
| Shipping kits | **358** |
| Shipping phrases | **2,487** |
| Clusters | **16** |
| Bank size | **659 MiB** |
| Avg phrases/kit | 6.95 |
| Clusters with a full 4-phrase run, all stems | **16/16** |

The 44 removed tracks are audited in
`arc8_stage/qc/filtered_kits.json`; their audio was pruned from the shipping
bank. The QC gate is `0.90` correlation / `10 dB` SNR (Opus codec-aware).

## Models

| Model | Result |
| --- | --- |
| Per-stem judges (drums / bass / other / vocals) | 0.891 / 0.835 / 0.720 / 0.706 AUC |
| Joint ranker, overall | 0.7854 held-out AUC |
| Joint 4-bar / 8-bar / 16-bar | 0.801 / 0.784 / 0.747 |
| Joint thresholds 4 / 8 / 16 | 0.082 / 0.067 / 0.038 |
| Phrases / state / input | 2,487 / 424-d / 2,137-d |

The shipped v3 joint state interleaves four 90-d stem chunks + 64-d mix:
52 phrase features + 4 groove + 2 vocal + 32 timbre per stem. This fixes the
old grouped-layout misalignment and adds onset density, offbeat/backbeat
ratio, syncopation, vocal presence and vocal crest.

## Runtime quality

- 300 s new-bank soak: **0 errors**, four distinct tracks, master RMS
  0.115–0.279, rotation covered.
- Live smoke: **8/8** transitions captured, **0 sample-jump seams**, 1 RMS dip;
  p50/p95 max jump **0.054 / 0.096**.
- Glue bus active: vocal duck on other/bass to **0.841** (−1.5 dB).
- 240 s reference-bank baseline: 24/24 captured, 0 seams, 3 dips,
  p50/p95 0.171/0.401.

The remaining audible issue is an occasional 1.5–2 dB level dip at a
phrase/run boundary, usually when a multi-phrase run advances or a bass/vocal
phrase changes. It is a level-match/phrase-energy issue, not a click.

## WubGotchi reactivity

The bear is now a four-stem meter, not a single master bounce:

- **Left eye = drums** (gold iris, kick-driven squish and pupil punch).
- **Right eye = bass** (cyan iris, bass-waveform ripple and pupil size).
- **Mouth = vocals** (waveform aperture, syllable punch, vowel geometry).
- **Heart = other** (pad/chord envelope, heartbeat rate, ripples, brightness
  hue shift; no longer driven by the kick).
- **Body, halo, cheeks and ambient motes = full mix** (true post-glue,
  post-limiter master level plus a smoothed master-waveform body contour).
- Four orbiting instrument motes give each stem its own colour and pulse:
  drums gold, bass cyan, vocals white, other magenta.

All sources are smoothed with fast attack / musical release envelopes; the
full mix is read from the master analyser rather than a max of the four stems.

## Shipped in this pass

1. **Joint-state layout + groove/vocal features.** Interleaved 90-d per-stem
   chunks, 424-d state, 2,137-d input; retrained on the 358-kit bank.
2. **Pin-region playlist.** The drag commits a partial run to the next block
   boundary, then the scheduler selects a fresh phrase inside the pin region
   each block until the 32-bar release. Probe: 7 distinct phrases / 4 tracks
   across bars 16–40, release exactly at bar 48.
3. **WubGotchi four-stem reactivity.** Eyes=drums/bass, mouth=vocals,
   heart=other, body/halo=full mix.
4. **Recovery audit of the 44 QC-failed tracks.** Still staging-only; only
   tracks that pass the 0.90/10 gate will be merged.
