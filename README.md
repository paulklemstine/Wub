# 🎛️ Wub KG-505 — The Infinite Electronic Groovebox

[![Deploy GitHub Pages](https://github.com/paulklemstine/Wub/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/paulklemstine/Wub/actions/workflows/deploy-pages.yml)
[![Tests](https://img.shields.io/badge/pytest-44%20passed%20(100%25)-brightgreen.svg)](#testing--quality-assurance)
[![Version](https://img.shields.io/badge/KG--505-v0.2.45%20%2345-7c3aed.svg)](#automated-versioning)
[![Live Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blue.svg)](https://paulklemstine.github.io/Wub/)

> **Klemstine Group KG-505 Production**  
> An autonomous AI music station and sample-accurate groovebox that trains on authentic electronic music corpus audio, learns multi-bar phrases and chord progressions, and composes an endless, evolving electronic song directly in your browser.

🌐 **Live Web Application**: [https://paulklemstine.github.io/Wub/](https://paulklemstine.github.io/Wub/)

---

## 🌟 Highlights & Capabilities

### 🧠 1. Dual-Engine Architecture
- **In-Browser Direct ML (100% Standalone)**:
  - Runs entirely client-side inside the browser using Web Audio API and lightweight serialized Markov transition graphs, acoustic slice registries, and harmonic progression banks.
  - Zero server, WebSocket, or backend required to play club-grade electronic music anywhere, including offline on static web hosts like GitHub Pages.
- **WebSocket Cloud Sync (Hybrid Server Stream)**:
  - Streams synchronized multi-stem audio slices from the FastAPI / Python server backed by a library of 1,700+ analyzed dance tracks, 740,000+ segments, 4-stem Demucs separation, and continuous GPU-trained **WubGPT** models.

---

### 🎹 2. Intelligent Musical Composition & Arrangement
- **Harmonic Chord Progression Engine**:
  - Automatically cycles through authentic 8-bar chord progression arcs learned from the corpus (e.g. `Am -> F -> C -> G`, `Dm -> Bb -> F -> C`, `Em -> C -> G -> D`).
  - Slices from real songs are tagged with their dominant chroma pitch ($0 \dots 11$) and repitched in real time using sample-accurate playback rate transposition:
    $$\Delta\text{semitones} = ((\text{targetPitch} - \text{slicePitch} + 6) \pmod{12}) - 6$$
    $$\text{playbackRate} = 2^{\frac{\Delta\text{semitones}}{12}}$$
- **Macro Tension & Release Arrangement**:
  - **`INTRO` (8 bars)**: Filtered downbeat kicks, warm ambient chord pads, with heavy bass muted to build expectation.
  - **`GROOVE_A` (16 bars)**: Hypnotic, rolling pocket rhythm with syncopated ghost percussions.
  - **`BUILDUP` (8 bars)**:
    - Snare rolls dynamically crescendo and double in frequency: quarter notes (bars 1–4) $\rightarrow$ eighth notes (bars 5–6) $\rightarrow$ sixteenth-note double-time rolls with rising exponential pitch risers.
    - Bass cuts out on bar 7.
    - Step 15 of bar 8 features the **silent drop gap breath** immediately before the drop.
  - **`DROP` (16 bars)**: Explosive maximum energy: punchy 4/4 kicks, driving rolling sub-bass, sizzling open hi-hats, polyphonic chord stabs, vocal chops, and full 16th arpeggios.
  - **`BREAKDOWN` (8 bars)**: Kicks and heavy bass cut out completely; ethereal floating pads and vocal phrases create spacious emotional relief.

---

### 🔊 3. Studio DSP & Acoustic Engineering
- **Dynamic Kick Sidechain Ducking**:
  - Every downbeat kick transient immediately ducks the bass bus by **-9 dB** (gain 0.35) and the synth bus by **-4 dB** (gain 0.63) with a 75ms exponential recovery curve, completely eliminating low-end phase mud.
- **Acoustic Drum Voicing & Choke**:
  - **Kick**: Tuned 909-style pitch sweeps (160 Hz $\rightarrow$ 44 Hz) with crisp click transient snap.
  - **Snare**: Tuned 195 Hz body layered with high-pass filtered white noise wire snap (1.9 kHz).
  - **Hi-Hats**: 6-oscillator metallic cluster with open-hat choking (closed hats instantly cancel ringing open hats).
  - **Sub Bass**: Resonant 24 dB low-pass filter tracking chord roots.

---

### 🌈 4. Trippy Self-Reactive Visualizers
- **Psychedelic WebGL Shader Canvas**:
  - Real-time Raymarching shader simulating DMT / LSD / mescaline kaleidoscopic fractal folding, neon color cycling, and liquid melting.
  - Polar coordinates seamlessly wrap at the borders (`mod(p, 2.0)` / smooth borderless textures) with dynamic reactivity to master RMS energy and tempo.
- **4-Stem Real-Time Waveform Station**:
  - Dedicated oscilloscopes, FFT spectral density bars, peak/RMS meters, and a sample-accurate laser playhead line smoothly sweeping across the 16/32 step grid.
- **Concentric Audio Rings**:
  - 4-ring radar display showing recent slice arcs per stem with exponential fade windows.
- **UMAP Slice Nebula**:
  - Interactive 2D projection of 740,000+ audio slices colored by stem. Click or drag stem gravity reticles to guide slice selection in real time.

---

### 🔄 5. Multi-Track Sequencer & Feedback Machine
- **Sequencer Architecture**:
  - Switchable cycle modes: **`4×16`** (64 steps), **`8×16`** (128 steps), or **`16×32`** (512 steps per master cycle).
  - Multi-track LED strip and 16/32 pip grid showing real-time playhead movement.
- **Reinforcement Feedback**:
  - 👍 / 👎 buttons next to each stem (Drums, Bass, Other, Vocals).
  - Downvoting a stem immediately triggers an on-the-fly reroll to a fresh compatible pattern and penalizes that slice archetype in the training preference model.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Audio Corpus Ingestion
        A[Raw MP3/WAV Corpus\n1,700+ Tracks] --> B[Demucs v4 Stem Separation\ndrums / bass / other / vocals]
        B --> C[Echo-Wub Analyzer\nBars / Beats / Tatums / Onsets]
        C --> D[28-D Feature Extraction\nMFCC, Chroma, Loudness, Centroid]
        D --> E[FAISS Indices &\nUMAP Dimensionality Reduction]
    end

    subgraph Machine Learning Training
        C --> F[Markov Phrase & Song Arranger\nBar Transitions & Section Models]
        D --> G[Pattern Audio Waveform Trainer\nLog-Mel Spectrogram Cosine Match]
        E --> H[Continuous GPU Trainer\nWubGPT Causal Transformer]
        F & G --> I[Export Standalone Bundle\nJSON weights, WAV Slices, Chords]
    end

    subgraph Dual Execution Runtime
        I --> J[🧠 In-Browser Direct ML\nClientGrooveboxConductor]
        H & E --> K[⚡ Python FastAPI Streamer\nUvicorn /ws/stream]
    end

    subgraph Client Audio & Visuals
        J & K --> L[Sample-Accurate UIEventQueue\nLocked to Web Audio Clock]
        L --> M[WebAudio DSP Engine\nSidechain Ducking + Voices]
        L --> N[Multi-Track Matrix & Step LEDs]
        M --> O[WebGL Psychedelic Background]
        M --> P[Waveform & FFT Analysis Station]
```

---

## 🚀 Quickstart Guide

### 1. Run in Your Browser (No Installation)
Visit **[https://paulklemstine.github.io/Wub/](https://paulklemstine.github.io/Wub/)** and click **▶ PLAY**.

### 2. Local Frontend Development
```bash
git clone https://github.com/paulklemstine/Wub.git
cd Wub/frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 3. Full Local Stack with Backend Server
```bash
# Set up Python environment
pip install -r requirements.txt

# Build frontend static assets
cd frontend && npm install && npm run build && cd ..

# Launch FastAPI uvicorn server on port 8123
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8123
# Open http://localhost:8123
```

### 4. Continuous GPU Training Daemon
If you have an NVIDIA GPU (e.g. GTX 1060+ / RTX):
```bash
python3 scripts/continuous_trainer.py --device cuda --min-bpm 116 --max-bpm 140
```
This daemon continuously:
1. Discovers unstemmed tracks in the library.
2. Runs Demucs stem separation on GPU.
3. Extracts backtracked tatums, segments, and chroma pitch profiles.
4. Updates vector quantizer vocabularies and trains WubGPT causal transformer weights.

---

## 🧪 Testing & Quality Assurance

Wub maintains a **100% green test suite** covering analyzers, groovebox conductors, audio synthesis, and server APIs:

```bash
# Run the full test suite
pytest -v
```

```
====================== 44 passed, 3 warnings in 27.73s ======================
tests/test_composer.py .................................. [ 11%]
tests/test_groovebox.py ................................. [ 29%]
tests/test_groovy_pattern_assembler.py .................. [ 45%]
tests/test_hygiene.py ................................... [ 52%]
tests/test_pattern_audio_trainer.py ..................... [ 63%]
tests/test_phase1.py .................................... [ 77%]
tests/test_server.py .................................... [ 88%]
tests/test_transformer.py ............................... [ 90%]
tests/test_version.py ................................... [100%]
```

---

## 🏷️ Automated Versioning

Every Git commit automatically increments the build number via a Git pre-commit hook:
```
VERSION = "0.2.45"
BUILD_NUMBER = 45
Format: v0.2.45 #45
```
Files synchronized on every commit:
- `VERSION`
- `version.json`
- `frontend/src/version.ts`
- `server/version.py`

When pushed to `main`, GitHub Actions automatically compiles TypeScript, bundles static assets, and deploys to GitHub Pages.

---

## 📂 Repository Layout

```
Wub/
├── .github/workflows/deploy-pages.yml   # Automatic GitHub Pages CI/CD workflow
├── data/
│   ├── composing_ml_weights.json       # Markov transition matrices & section weights
│   ├── processed/                      # Extracted track JSONs (tatums, segments, features)
│   ├── stems/                          # Demucs multi-stem audio files
│   └── standalone_patterns.json        # Curated patterns, phrases & chord progressions
├── frontend/
│   ├── public/data/slices/             # Clean 44.1kHz WAV slices for in-browser playback
│   ├── src/
│   │   ├── client_ml_sequencer.ts      # Autonomous client-side ML groovebox & sound engine
│   │   ├── audio_engine.ts             # Web Audio context, buses, analysers, and scheduling
│   │   ├── visualizer.ts               # Waveform oscilloscopes, FFT spectrum, and rings
│   │   ├── psychedelic_bg.ts           # WebGL Raymarching fractal shader
│   │   ├── main.ts                     # Main UI orchestrator & UIEventQueue
│   │   └── version.ts                  # Auto-incrementing version constants
│   └── index.html                      # KG-505 hardware studio interface
├── scripts/
│   ├── continuous_trainer.py           # Background GPU Demucs + WubGPT training daemon
│   ├── export_standalone_client_bundle.py # Bundle builder for standalone client
│   ├── analyze_corpus.py               # Multiprocessing feature extractor
│   └── train_wubgpt.py                 # PyTorch causal transformer trainer
├── server/
│   ├── main.py                         # FastAPI REST & WebSocket streaming server
│   ├── stream_manager.py               # Multi-client stream session coordinator
│   └── version.py                      # Server-side versioning
├── wub_core/
│   ├── groovy_pattern_assembler.py     # Physical acoustic classifier & groove templates
│   ├── groovebox.py                    # Server-side groovebox conductor
│   ├── pattern_audio_trainer.py        # Real-audio spectrogram similarity evaluator
│   ├── model_transformer.py            # WubGPT Transformer architecture
│   ├── model_markov.py                 # Multi-stem metrical graph
│   └── corpus.py                       # High-performance track & slice registry
└── tests/                              # Comprehensive test suite (44 passing tests)
```

---

## 📜 License
MIT License. Created by Paul Klemstine and the Wub KG-505 Production Team.
