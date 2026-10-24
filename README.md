# Project Wub — Infinite Jukebox (all phases implemented)

AI-driven polyphonic concatenative synth: Demucs stems → Echo-Wub analyzer
(bars/beats/tatums + per-stem onset segments) → per-stem FAISS indices →
generative mashups (Markov graph + VQ-Transformer) → live 4-stem web stream →
neural latent resynthesis. Every layer reads through `CorpusStore`, so all of
it stays correct while the library grows.

## Quickstart

```bash
pip install -r requirements.txt  # torch+demucs only needed for separation
npm --prefix frontend install && npm --prefix frontend run build

# 1. Ingest (continuous while files arrive; resumes via manifest):
python3 scripts/analyze_corpus.py --input pandora --workers 8 --watch

# 2. Offline mashup (Markov graph, Phase 2A):
python3 scripts/generate_mashup.py --steps 200 --chaos other=0.85 --out data/mashup.wav

# 3. Train WubGPT (Phase 2B) once enough tracks are ingested:
python3 scripts/train_wubgpt.py --processed data/processed --model-out data/wubgpt.pt --epochs 8

# 4. Serve the infinite stream + UI (Phase 3):
uvicorn server.main:app --host 0.0.0.0 --port 8000  # open http://localhost:8000/

# Verification:
python3 scripts/verify_slicing.py   # click-free re-stitch, ±5 ms
pytest tests/ -v
```

Windows library variant: `--input "E:/pandora"` (backslashes normalized).

Notes:
- Default ingest is mix analysis (no Demucs). 5000 tracks x Demucs on CPU is
  weeks and ~800 GB of stems; separate a curated subset later instead.
- At ~10–18 s/track on 8 workers, 5000 tracks take ~2–3 h per pass.
- `data/processed/_manifest.json` tracks state; re-runs only process new/changed files.
- Live progress: `tail -f /tmp/opencode/wub_ingest.log`,
  `python3 scripts/generate_mashup.py --list-tracks | wc -l`.

## Layout

- `wub_core/schema.py` — Pydantic `TrackAnalysis`/`Tatum`/`Segment` (spec §4.2)
- `wub_core/separator.py` — Demucs wrapper, `data/stems/{track_id}/` cache
- `wub_core/analyzer.py` — drum-locked beats → 4× tatums → per-stem backtracked onsets + MFCC/chroma/loudness
- `wub_core/indexer.py` — 28-D vectors → `index_{drums,bass,vocals,other}.faiss` + unified + UMAP
- `wub_core/audio_utils.py` — 3 ms equal-power micro-crossfades
- `wub_core/corpus.py` — live registry: mtime-cached refresh, audio resolution, pools, index/UMAP views
- `wub_core/model_markov.py` — Phase 2A multi-stem metrical-harmonic graph + offline renderer
- `wub_core/model_transformer.py` — Phase 2B VQ + 6L/8H/d384 causal Transformer (WubGPT)
- `wub_core/synth_decoder.py` — Phase 4 latent backend (EnCodec/RAVE) + Synthetic/Real blend
- `server/main.py` — REST + `/ws/stream` 4-stem streamer, serves `frontend/dist`
- `server/stream_manager.py` — 32-tatum lookahead sessions, slice WAV cache
- `frontend/` — Vite+TS: 4-bus Web Audio scheduler, concentric-ring + UMAP-nebula canvas, stem deck UI
- `scripts/` — `analyze_corpus.py` (incremental ingest), `generate_mashup.py`, `train_wubgpt.py`, `verify_slicing.py`
