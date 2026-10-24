# WUB HANDOFF — for the next AI agent (written 2026-10-24 ~11:00 UTC, tokens critical)

## What this is
Project Wub: "Infinite Jukebox" — MP3 library → Demucs stems → Echo-Nest-style
analysis (bars/beats/tatums + per-stem onset segments, Pydantic JSON) → per-stem
FAISS indices → generative mashups (Markov graph + VQ-Transformer) → FastAPI +
WebSocket 4-stem web stream → neural latent resynthesis. Repo root:
`/home/raver1975/Wub` (NOT a git repo). Full spec: `wub_agent_plan.md`. Ops guide: `README.md`.

## Machine layout
- **This box** (Linux, 16 cores, 31 GB): `~/Wub` repo, `pandora/` library (2236 MP3s,
  incl. `pandora/good/` 935 files), ingest + analysis + backend + tests.
- **Laptop "Victus"** (Win11 + WSL Ubuntu, RTX 4050 6 GB, 6 cores, **5 GB RAM — the
  binding constraint**): GPU separation + WubGPT training. MP3s at `/mnt/e/pandora`
  (1305 files, still filling) and `/mnt/e/pandora/good`.
- **Access**: reverse tunnel. In WSL laptop run:
  `ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -R 2223:localhost:22 raver1975@192.168.1.118`
  then use `export WUB_LAPTOP_PASS='<laptop-wsl-password>'` + `tools/laptop.sh`:
  `printf '%s' '<bash>' | tools/laptop.sh exec`, `tools/laptop.sh put/get`. Test with
  `printf '%s' 'hostname' | tools/laptop.sh exec`. Direct `ssh -p 2222` via Windows
  portproxy is BROKEN (RST after handshake) — don't waste time on it; the tunnel works.
  If "remote port forwarding failed", old tunnel is stale: `ps aux|grep 'sshd: raver'`,
  `kill` the unprivileged PIDs, ask user to re-run the tunnel command.

## Verified milestones (all green, evidence in logs)
1. **Phase 1**: 12/12 real Demucs-stem tracks re-stitch click-free
   (`scripts/verify_slicing.py` → PASS). Analyzer quirk fixed: short slices edge-padded
   to 4096 for CQT/MFCC; segments capped at 4 s (`MAX_SEGMENT_SEC`); verify math accounts
   for crossfade overlap honestly (see code comments).
2. **Phase 2A**: forced-diverse 4-stem mashups on laptop: `~/wub/verify60.wav` (77 s,
   4 source tracks, pitch-locked) + 5-min stress render. Local copy: `/tmp/opencode/verify60.wav`.
3. **Phase 2B**: WubGPT (StemVQ KMeans + 6L/8H/d384 causal Transformer) trains; smoke test
   passed. Full-corpus run: **IN PROGRESS on laptop** (see below).
4. **Phase 3**: backend e2e vs 1724-track corpus — `/health`, `/api/tracks`,
   `/audio/{tid}/{seg}.wav` (200), `POST /api/mashup` (200, 24 s artifact), live WS
   `/ws/stream` (hello + 6 tatums, all 4 stems fed, set/set_bpm accepted). Frontend:
   `tsc --noEmit` clean, `vite build` OK, served by FastAPI from `frontend/dist`.
5. **Suite**: 14/14 `pytest tests/` (schema, crossfade, indexer dim, analyzer e2e,
   markov chaos semantics, server API w/ fixture, synth blend math, hygiene incl.
   `--help` on all CLIs + repeatable-flag regression).

## Live jobs at handoff
- **Laptop training** (`~/wub_full/train.log`): 1724-track corpus (1,888,040 rows,
  stride 2), VQ 256×4 done, 11.1M-param model, `--epochs 3 --batch 128 --device cuda`.
  At handoff: in epoch 1, no checkpoint yet. Check:
  `printf '%s' 'grep -aE "^epoch|loss curve|^wrote" ~/wub_full/train.log | tail' | tools/laptop.sh exec`
  When done: `tools/laptop.sh get ~/wub_full/wubgpt.pt data/wubgpt/` (+ `wubgpt_vq/` dir).
  Next: WubGPT-driven Concatenative assembly (`sample` → `tokens_to_slices` → render;
  functions exist in `wub_core/model_transformer.py`, no CLI yet — build one).
- **Local ingest**: COMPLETE (1724 analyses, 0 errors final pass, 5 FAISS indices in
  `data/indices/`). Server on :8123 was a manual test (probably dead; restart with
  `setsid nohup python3 -m uvicorn server.main:app --host 127.0.0.1 --port 8123`).

## Critical gotchas (learned the hard way)
- **numba**: must be ≥0.68 with numpy 2.x or `librosa.beat.beat_track` SEGFAULTS
  (pinned in `requirements.txt`; diagnosed via `faulthandler`).
- **separator cache poisoning (FIXED)**: Demucs output lookup MUST be scoped to
  `_demucs_tmp/{model}/{input_stem}/` — unconstrained rglob copied wrong tracks' stems.
  Any stems separated before the fix are untrustworthy (laptop cache was purged).
- **argparse repeatable flags need `action="append"`** or only the last `--flag` applies
  (caused the "9/12 source tracks" false alarm; regression test in `tests/test_hygiene.py`).
- **`default([])` after kwargs is a SyntaxError** (looks like `default=[]`); hygiene test
  compiles everything + runs all `--help`s — run it after every edit.
- **Laptop 5 GB RAM**: no swap; OOM-killer is the #1 failure mode. Rules: render folds
  incrementally (already), pitch-shift only 4096 < len < 8·sr, skip slices >16 s,
  pools drop >16 s slices, KMeans via `--vq-sample`, training via `--chunk-tracks` +
  `--tatum-stride`, subsets via sequential `phase_subset.py` (NOT the pool runner).
- **`scripts/analyze_corpus.py`**: rolling submission window (workers×3) + 1200 s
  per-file timeout (`WUB_FILE_TIMEOUT`); manifest resume is fingerprint-based.
- **Long-tail segments**: sparse stems made 164 s "slices" → 2 h renders → OOM. Fixed at
  slicer (4 s cap) + pools + render guards. Old local JSONs predate the cap; pools filter them.
- **Background shells get reaped**: launch long jobs with `setsid nohup … & disown`.
- `tools/laptop.sh` passes the remote script via temp file (NOT stdin-heredoc into python —
  stdin collision); remote commands via base64 (quote-safe); 240 s per-call cap → nohup
  everything long on the laptop and poll logs.
- WSL IP changes on reboot (was 172.29.248.125); tunnel is immune to this (outbound).

## Next steps (ordered)
1. Collect finished WubGPT checkpoint → build `scripts/sample_wubgpt.py`
   (sample → tokens_to_slices → render_mashup with target_bpm/pitch-lock) → A/B vs Markov.
2. Expand stem subset on laptop (`phase_subset.py` takes sorted glob slice; edit range for
   files 12+ of `good/`) → copy JSONs back (track_ids match byte-identical files; verify
   with sha check as done before) → richer 4-stem pools.
3. Add per-slice time-stretch to `stream_manager` rate path test (server sends `rate`;
   frontend applies `playbackRate` — untested live) + frontend `/api/umap` fetch (currently
   seeded demo points).
4. Phase 4 demo: `pip install encodec` on laptop, `StemLatentSynth.with_backend('encodec')`,
   embed a stem subset, render the 0/50/100 % slider positions.
5. Optional: re-analyze local corpus with the 4 s cap (pools already filter; ~4 h).
