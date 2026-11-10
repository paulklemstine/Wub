# Mix Compatibility (Per-bar) - Working Plan

## User choices
1. Per-bar scoring (aggregate over 4 bars of a phrase; min-bar matters)
2. Render+embed offline for teacher labels
3. Cap <=2 stems from same source track in any mix (forbid 4-same)
4. Rhythm mismatch is top priority
5. Extra model in browser at runtime

## Pipeline
- **Phase 0 (engine, now):** enforce `maxSameSource(mix) <= 2` in joint/morph selection; never allow 4-of-a-kind. Add compat hook points.
- **Phase 1 (offline, laptop 4050):** `scripts/build_mix_compat_dataset.py`
  - Sample mixes: positives = real co-occurring 4-stem phrases (per-bar windows aligned); negatives = random cross-song combos (tempo/key normalized as engine), stratified, excluding full-same-source.
  - Render per-bar summed audio (apply rate/shift like engine via rubberband/ffmpeg or resample+pitch approximation) for a sampled set; compute per-bar features:
    - Rhythm (priority): per-stem onset envelopes per bar (from stem bars/mel or onset detect), pairwise onset correlation/syncopation alignment (kick-bass, kick-other, bass-other, drums-vocals), groove consistency, downbeat lock.
    - Harmonic: chroma consonance bass/other/vocals, bass-root intervals.
    - Spectral/mask: band energy overlap (sub/low/mid/hi), loudness balance.
    - Energy/section coherence.
  - Optional mix embedding (MERT/PANNs via ears pipeline) for distance-to-real-manifold teacher.
  - Labels: graded compatibility score (real=1, random scored by heuristic+embed-distance teacher emphasizing rhythm), per-bar + min/mean aggregates.
- **Phase 2:** `scripts/train_mix_compat.py` trains compact pairwise/interaction model on per-bar features -> predicts compat (and rhythm subscore); exports `mix_compat.json` + weights for browser.
- **Phase 3 (engine):** `frontend/src/mix_compat.ts` loads model, scores candidate mixes per-bar from cheap features derivable in-browser (manifest chroma/energy/rms + precomputed per-bar rhythm descriptors shipped in sidecar). Blend `mix_compat` (min-bar weighted, rhythm-heavy) into joint/morph scoring. Keep <=2 cap.
- **Phase 4:** hard-negative mining from logs, A/B listening.

## Runtime feature feasibility
Full audio render in browser is too expensive per candidate; ship precomputed per-bar rhythm/onset descriptors per phrase×stem (small sidecar) so browser can compute cross-stem rhythm sync cheaply.
