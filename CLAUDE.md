# 🤖 AGENTS.md — Master Developer & AI Agent Handbook

Welcome to the **Wub KG-505** codebase. This document serves as the primary technical specification, architectural guide, and operational manual for AI agents and human engineers modifying, debugging, or extending this repository.

---

## 🧭 Project Identity & Core Philosophy

- **Name**: Wub KG-505 (Klemstine Group KG-505 Production).
- **Domain**: Real-time generative electronic music, audio stem analysis, sample-accurate concatenative synthesis, and web-based audio engineering.
- **Mission**: Ingest an authentic audio training corpus (1,700+ dance tracks), separate into stems (drums, bass, other, vocals), extract rhythmic onsets and chroma pitch features, score acoustic similarity, and compose a neverending, highly musical electronic song directly in the browser.
- **Dual-Engine Execution**:
  1. **🧠 In-Browser Direct ML**: 100% standalone, client-side Web Audio engine using serialized Markov phrase models, chord progression engines, and acoustic WAV slice banks. Runs everywhere (including static hosting on GitHub Pages) with zero backend dependency.
  2. **⚡ WebSocket Cloud Sync**: Client/server mode streaming real-time tatums and slice references from the Python FastAPI server backed by FAISS vector indices and continuous GPU-trained WubGPT models.

---

## 🚨 Invariant Architectural Rules for Agents

When making changes to this codebase, you must adhere strictly to these non-negotiable engineering rules:

### 1. Test Suite Hygiene (100% Green Requirement)
- Every single test in `pytest` must pass at all times.
- Run `pytest` before committing any changes.
- Do not disable, `@pytest.mark.skip`, or comment out existing tests unless explicitly directed.
- If you introduce a new feature or modify an algorithm, add corresponding unit tests in `tests/`.

### 2. Frontend TypeScript Build Integrity
- The frontend must compile cleanly with zero TypeScript errors:
  ```bash
  cd frontend && npm run build
  ```
  (`tsc && vite build`).
- Do not use `@ts-ignore` or `any` unless interfacing with dynamic JSON schema payloads at boundaries.

### 3. Automated Version Control Protocol
- The project version is formatted as `v0.2.<build_number> #<build_number>` (e.g. `v0.2.45 #45`).
- A Git pre-commit hook (`scripts/pre_commit_version_bump.py` via `.git/hooks/pre-commit`) automatically increments the build number on every commit.
- Files kept in sync:
  - `VERSION`
  - `version.json`
  - `frontend/src/version.ts`
  - `server/version.py`
- Whenever you commit with `git commit`, the hook triggers automatically. Verify the output displays `⚙️ [git hook] Incrementing Wub version control number...`.

### 4. Sample-Accurate Web Audio Timing & `UIEventQueue` Pattern
- **Lookahead Scheduling**: Web Audio requires scheduling audio notes 100–150ms into the future (`when = nextStepTime`).
- **NEVER Mutate DOM in Lookahead**: Calling `updateGrooveboxUI()` inside the audio scheduling loop causes the UI to jump ahead before the sound plays.
- **NEVER Use Uncoordinated `setTimeout`**: Clamped browser timer resolution (4ms) and main-thread load cause `setTimeout` delays to fire out of order, making the UI jump back and forth in time.
- **ALWAYS Use the `UIEventQueue` Pattern**:
  1. The audio scheduler pushes `{ atAudioSec, meta, step, totalSteps }` into `this.uiEventQueue`.
  2. In `frame()` (running at 60 FPS via `requestAnimationFrame`), compare `this.engine.now() >= event.atAudioSec`.
  3. When the audio clock reaches the event's playback time, shift the event and call `updateGrooveboxUI(latestUIEvent.meta)`.
  4. If multiple events expired during a tab pause, fast-forward to the latest event so the UI never plays catch-up or displays out-of-order steps.

### 5. JavaScript Array Inclusion vs. Object Key Check
- In Python: `if step in [0, 4, 8, 12]:` checks array elements.
- In JavaScript / TypeScript: `if (step in [0, 4, 8, 12])` checks **indices/keys** (`0, 1, 2, 3`), NOT values!
- **RULE**: In TypeScript, **ALWAYS** use:
  ```ts
  if ([0, 4, 8, 12].includes(step % 16))
  ```

### 6. Harmonic Pitch-Shifting Math
- All pitched slices (bass, synths/chords, vocals) are tagged with chroma pitch $0 \dots 11$ (C=0, C#=1, ..., B=11).
- When repitching a slice with root $P_{\text{slice}}$ to match active chord root $P_{\text{target}}$:
  ```ts
  let shift = (targetPitch - slicePitch) % 12;
  if (shift > 6) shift -= 12;
  if (shift < -6) shift += 12;
  const playbackRate = Math.pow(2, shift / 12);
  ```
  This guarantees that slices never shift by more than 6 semitones (half an octave), preserving natural acoustic timbre while maintaining perfect musical harmony.

---

## 🗺️ Codebase Map & Responsibilities

### Frontend (`frontend/src/`)
| File | Responsibility |
| :--- | :--- |
| [`client_ml_sequencer.ts`](file:///home/raver1975/Wub/frontend/src/client_ml_sequencer.ts) | Autonomous in-browser ML conductor, Web Audio slice synthesizer (click kick, noise snare, metallic hats, sub bass, detuned saw chord, arpeggiator, vocal chop), harmonic progression engine, and section arranger. |
| [`audio_engine.ts`](file:///home/raver1975/Wub/frontend/src/audio_engine.ts) | Core Web Audio context wrapper, 4-stem gain buses, analysers, slice scheduling, and WAV decoding. |
| [`main.ts`](file:///home/raver1975/Wub/frontend/src/main.ts) | Main application coordinator, UI event queue, transport controls, feedback handling, and WebSocket listener. |
| [`visualizer.ts`](file:///home/raver1975/Wub/frontend/src/visualizer.ts) | 4-stem oscilloscope waveforms, FFT spectral density, peak/RMS meters, concentric radar rings, and UMAP nebula. |
| [`tamagotchi.ts`](file:///home/raver1975/Wub/frontend/src/tamagotchi.ts) | WubGotchi neural music pet avatar, learning ASCII animated face expressions, vowel mouth shapes, beat head-bobbing, and XP evolution. |
| [`psychedelic_bg.ts`](file:///home/raver1975/Wub/frontend/src/psychedelic_bg.ts) | WebGL full-screen real-time raymarching shader with DMT / LSD kaleidoscopic folding, borderless polar wrapping, and RMS reactivity. |
| [`version.ts`](file:///home/raver1975/Wub/frontend/src/version.ts) | Git-hook synchronized build numbers and release timestamps. |

### Backend & Core Algorithms (`wub_core/` & `server/`)
| File | Responsibility |
| :--- | :--- |
| [`wub_core/groovy_pattern_assembler.py`](file:///home/raver1975/Wub/wub_core/groovy_pattern_assembler.py) | Physical acoustic slice classifier, Camelot scale consonance profiles, multi-bar groove templates, and sidechain simulation. |
| [`wub_core/pattern_audio_trainer.py`](file:///home/raver1975/Wub/wub_core/pattern_audio_trainer.py) | Multi-scale log-Mel spectrogram cosine similarity, onset envelope correlation, and reinforcement feedback. |
| [`wub_core/patterns.py`](file:///home/raver1975/Wub/wub_core/patterns.py) | Fast $O(1)$ track-indexed slice palette, multi-stem patterns, pattern bank, and densification. |
| [`wub_core/groovebox.py`](file:///home/raver1975/Wub/wub_core/groovebox.py) | Server-side groovebox conductor, section state machine, and 16/32-step sequencer. |
| [`wub_core/model_transformer.py`](file:///home/raver1975/Wub/wub_core/model_transformer.py) | WubGPT causal transformer for multi-stem token sampling. |
| [`server/main.py`](file:///home/raver1975/Wub/server/main.py) | FastAPI application, REST endpoints (`/health`, `/api/tracks`, `/audio/...`), and WebSocket `/ws/stream`. |
| [`server/stream_manager.py`](file:///home/raver1975/Wub/server/stream_manager.py) | Stream session manager, lookahead buffer refill, and slice WAV rendering. |

### Scripts (`scripts/`)
| File | Responsibility |
| :--- | :--- |
| [`continuous_trainer.py`](file:///home/raver1975/Wub/scripts/continuous_trainer.py) | Long-running GPU daemon separating stems with Demucs and periodically retraining WubGPT. |
| [`export_standalone_client_bundle.py`](file:///home/raver1975/Wub/scripts/export_standalone_client_bundle.py) | Extracts top-scoring acoustic slice WAVs, chord progressions, and patterns into `frontend/public/data/`. |
| [`pre_commit_version_bump.py`](file:///home/raver1975/Wub/scripts/pre_commit_version_bump.py) | Git pre-commit hook that increments the build number across all version files. |

---

## 🎛️ Essential Workflows & Commands

### Running Tests
```bash
# Run entire test suite
pytest -v

# Run a specific test module
pytest tests/test_groovy_pattern_assembler.py -v
```

### Building Frontend
```bash
cd frontend
npm run build
```

### Re-exporting Standalone Client Bundle
When you extract new slices, train new weights, or update chord progressions:
```bash
PYTHONPATH=. python3 scripts/export_standalone_client_bundle.py
```

### Launching Local Servers
```bash
# Start FastAPI backend server on port 8123
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8123

# Start Vite dev server for frontend
cd frontend && npm run dev
```

### Git Commit & Deployment
```bash
# Stage files
git add -A

# Commit (triggers pre-commit hook auto-incrementing build number)
git commit -m "feat: Description of changes"

# Push to origin (triggers GitHub Actions deployment to GitHub Pages)
git push origin main
```

---

## 🎧 Musical Arrangement Cheat Sheet

When adjusting groove templates or generating patterns:

| Section | Duration | Energy | Kick Pattern | Bass Behavior | Synths & Leads | Vocals |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`INTRO`** | 8 bars | 0.25 | Downbeats only (0, 4, 8, 12), filtered | Completely muted | Ethereal floating chord pads | Ambient vocal calls ("Ooh..."), sustained pads on bars 0 & 4 |
| **`GROOVE_A`** | 16 bars | 0.83 | 4-on-the-floor + ghost on step 10 | Pumping offbeat sub bass | Call-and-response chord stabs | Lyrical narrative verses with Call & Response phrasing |
| **`BUILDUP`** | 8 bars | 0.65 | Pulse on 0, 4, 8, 12 (bars 1–5 only) | Cuts out on bar 7 | Rising saw riser on bar 6 | Rising vocal chants ("Higher..."), stutter rolls, silent breath on step 15 |
| **`DROP`** | 16 bars | 0.90 | Full driving 4/4 with sidechain ducking | Rolling 16th sub bassline | 8-bar melodic arpeggio arc | Anthemic vocal chorus hook singing full melody over driving beats |
| **`BREAKDOWN`**| 8 bars | 0.20 | Completely muted (0 kicks) | Completely muted | Lush polyphonic chord triads | Soulful solo singing melody with rich vowel formants & vibrato |

> [!IMPORTANT]
> On **Bar 7, Step 15** of a `BUILDUP`, step 15 **MUST BE SILENT**. This 1/16th-note breath (the drop gap) creates the critical acoustic contrast that makes the drop hit with maximum impact.
