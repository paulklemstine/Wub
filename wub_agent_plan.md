Project Wub: Autonomous Agent Execution Plan & Technical Specification1. System Overview & VisionProject Wub is an AI-driven "Infinite Jukebox" and polyphonic concatenative synthesizer for electronic music. Inspired by the legacy Echo Nest analyzer, Wub ingests a library of MP3 files, first splits each track into isolated instrument stems (drums, bass, vocals, other/synths), and then decomposes them into hierarchical rhythmic units (Bars, Beats, Tatums) and micro-acoustic onset slices (Segments). By extracting deep feature vectors per stem slice, Wub trains a multi-stream sequence model capable of generating infinite, cross-track electronic music mashups in real-time within a web browser (e.g., locking the drum groove from Track A with chopped bass wubs from Track B and vocal slices from Track C). Phase 4 introduces latent neural audio synthesis to hallucinate entirely new synthetic micro-slices per stem.2. Agent Instructions & Operational GuardrailsATTENTION AI CODING AGENT: Read these directives carefully before writing any code.Modular Incrementalism: Execute this project strictly phase-by-phase. Do not build the frontend or ML training loops until Phase 1 (Stem Separation, Audio Analysis & Slicing) passes its verification tests.Audio Continuity First: Concatenative synthesis is prone to clicks and pops. Every audio transition across every stem stream in the playback engine must implement a micro-crossfade ($2\text{ ms} - 5\text{ ms}$ equal-power curve) or snap to zero-crossings.Strict Typing & Schemas: Use Pydantic for all Python data structures and TypeScript interfaces for the frontend. The JSON feature database connecting the analyzer, the ML model, and the web audio engine must strictly adhere to the schema defined in Section 4.No Placeholder DSP: Implement actual neural stem extraction (demucs) and feature extraction using librosa and madmom/essentia. Do not mock audio features with random numbers.3. Recommended Tech StackLayerTechnologyPurposeStem Separationdemucs (htdemucs / htdemucs_ft), torch, torchaudioSplitting full mixes into isolated drums, bass, vocals, and other (synths/FX) stems.Audio DSP & SlicingPython 3.11+, librosa, scipy, soundfileBeat/tatum tracking, stem-specific onset detection, MFCC/Chroma extraction.Vector IndexingFAISS (Facebook AI Similarity Search)Dedicated $k$-NN indices per stem type for fast multi-stream slice matching.Machine LearningPyTorch, einops, scikit-learnTraining the multi-stem sequence generation model (Transformer/Markov) & RAVE/VAE synth.Backend APIFastAPI, Uvicorn, WebSocketsStreaming multi-stem slice metadata and binary audio chunks to the frontend.Frontend UI & AudioTypeScript, Vite, Web Audio API, HTML5 CanvasSample-accurate 4-bus stem scheduling, time-stretching, and infinite graph visualization.4. Data Architecture & Echo-Nest-Style SchemaEvery analyzed track is decomposed into a master Rhythmic Grid (bars, beats, tatums) derived from the full mix and drum stem, and Stem-Separated Acoustic Events (segments tagged by drums, bass, vocals, other).4.1 Feature Extraction Mathematical DefinitionsFor each segment $s_i$ on stem $m \in \{\text{drums}, \text{bass}, \text{vocals}, \text{other}\}$ bounded by time $[t_{\text{start}}, t_{\text{end}}]$:Timbre Vector ($\mathbf{v}_{\text{timbre}} \in \mathbb{R}^{12}$): The mean of the first 12 Mel-Frequency Cepstral Coefficients (MFCCs) across the segment frames.Pitch/Chroma Vector ($\mathbf{v}_{\text{chroma}} \in \mathbb{R}^{12}$): Normalized energy across the 12 semitone pitch classes ($\text{C}, \text{C}\#, \dots, \text{B}$). Particularly critical for harmonic alignment between bass, vocals, and other (synth) stems.Loudness Envelope & Stem Activity:
$$L_{\text{max}} = 20 \log_{10}(\max(\vert{}x_m(t)\vert{})), \quad L_{\text{start}} = 20 \log_{10}(\text{rms}(x_m(t_{\text{start}})))$$
Segments with $L_{\text{max}} < -60\text{ dB}$ are flagged as is_silent: true so the generator knows when a stem drops out.4.2 Unified JSON Schema (track_analysis.json){
  "track_id": "sha256_hash",
  "filename": "artist_track.mp3",
  "bpm": 128.0,
  "key": "F# minor",
  "duration": 214.5,
  "sample_rate": 44100,
  "stems_available": ["drums", "bass", "vocals", "other"],
  "tatums": [
    {
      "id": "tatum_0001",
      "index": 0,
      "bar_position": 0,
      "beat_position": 0,
      "tatum_position": 0,
      "start": 0.452,
      "duration": 0.117,
      "confidence": 0.94,
      "stem_segments": {
        "drums": ["seg_drums_0001"],
        "bass": ["seg_bass_0001", "seg_bass_0002"],
        "vocals": [],
        "other": ["seg_other_0001"]
      }
    }
  ],
  "segments": [
    {
      "id": "seg_bass_0001",
      "track_id": "sha256_hash",
      "stem": "bass",
      "start": 0.452,
      "duration": 0.065,
      "is_silent": false,
      "loudness_start": -24.5,
      "loudness_max": -6.2,
      "loudness_max_time": 0.012,
      "pitches": [0.95, 0.12, 0.04, 0.01, 0.22, 0.05, 0.02, 0.45, 0.01, 0.03, 0.11, 0.08],
      "timbre": [42.1, -12.4, 8.3, 1.2, -5.6, 3.1, -0.4, 2.2, -1.1, 0.8, -0.2, 0.5],
      "spectral_centroid": 450.5,
      "zero_crossing_start_offset": 0.0002
    }
  ]
}
5. Phased Implementation Plan for the AgentPhase 1: Stem Separation & The "Echo-Wub" Analyzer PipelineObjective: Build a CLI tool (wub_analyze.py) that ingests a folder of MP3s, separates each track into 4 stems (drums, bass, vocals, other), extracts the rhythmic grid and per-stem acoustic features, and builds stem-specific FAISS vector indices.Step 1.0: Neural Stem Extraction (separator.py)Run Hybrid Transformer Demucs (htdemucs) on each source MP3 to output 4 phase-aligned stereo WAV stems at $44.1\text{ kHz}$: drums.wav, bass.wav, vocals.wav, and other.wav (which captures lead synths, chords, pads, and FX).Provide a --fast-cpu flag to optionally use quantized ONNX models or cache separated stems in data/stems/{track_id}/ so separation only runs once per song.Step 1.1: Beat & Tatum Grid Extraction (Locked to Drum Stem)Run librosa.beat.beat_track (or madmom.features.beats.RNNBeatProcessor) directly on the isolated drums.wav stem (falling back to the full mix on beatless ambient tracks) for ultra-tight transient grid locking.Subdivide each beat interval $[b_k, b_{k+1}]$ into $N$ tatums ($N=4$ for 16th-note electronic grids):
$$t_{k, m} = b_k + \frac{m}{4}(b_{k+1} - b_k), \quad m \in \{0, 1, 2, 3\}$$Step 1.2: Per-Stem Onset-Driven Segment SlicingIndependently detect micro-transients on each of the 4 stems using librosa.onset.onset_detect(backtrack=True). Backtracking snaps the slice start point to the nearest preceding local energy minimum/zero-crossing on that specific stem to prevent clicks.Compute the 12-D pitches (librosa.feature.chroma_cqt) and 12-D timbre (librosa.feature.mfcc(n_mfcc=13)) for every non-silent stem segment.Step 1.3: Feature Normalization & Multi-Index FAISS BuildingConcatenate features into a single standardized vector per stem segment:
$$\mathbf{x}_{i, m} = [\text{zscore}(\mathbf{v}_{\text{timbre}}) \,\Vert{}\, \mathbf{v}_{\text{chroma}} \,\Vert{}\, \text{zscore}(L_{\text{max}}) \,\Vert{}\, \text{pos}_{\text{metrical}}] \in \mathbb{R}^{28}$$Build 4 separate faiss.IndexFlatL2 indices (index_drums.faiss, index_bass.faiss, index_vocals.faiss, index_other.faiss) plus a unified index for sub-millisecond similarity queries per instrument layer.Phase 1 Verification Criteria:Run a test script that reconstructs an original MP3 purely by re-stitching its 4 separated stems' segments in chronological order and summing them together. The output WAV must have zero audible clicks and match the original duration within $\pm 5\text{ ms}$.Phase 2: The Multi-Stem Generative Sequence Model ("WubBrain")Objective: Train a model that learns both horizontal transitions (how each stem evolves over time) and vertical harmonic/rhythmic compatibility (how bass, drums, vocals, and other lock together across different songs).Step 2.1: Multi-Stem Hybrid Graph (Fast Prototype)Maintain 4 parallel playback heads (one per stem: $m \in \{\text{drums}, \text{bass}, \text{vocals}, \text{other}\}$), synchronized to the master tatum clock.The transition cost $D_m(i, j)$ for stem $m$ jumping from Tatum $i$ to Tatum $j$ combines horizontal continuity with vertical harmonic compatibility against the currently active bass/synth stems:
$$D_m(i, j) = w_{\text{timbre}} \Vert{}\mathbf{v}_{\text{timbre}, i+1}^{(m)} - \mathbf{v}_{\text{timbre}, j}^{(m)}\Vert{}_2 + w_{\text{pitch}} \Vert{}\mathbf{v}_{\text{chroma}, i+1}^{(m)} - \mathbf{v}_{\text{chroma}, j}^{(m)}\Vert{}_2 + w_{\text{harmony}} \Vert{}\mathbf{v}_{\text{chroma}, j}^{(m)} - \mathbf{v}_{\text{chroma, active}}^{\text{(bass)}}\Vert{}_{2}$$Allow pitch-shifting of melodic stems (bass, vocals, other) by $\pm 6$ semitones during FAISS lookup so slices from tracks in different musical keys can be harmonically locked in real time.Step 2.2: Multi-Track Vector-Quantized Transformer (WubGPT)Train separate Vector Quantizers (RVQ or KMeans with $K=512$ centroids) for each of the 4 stem types.At each tatum step $t$, represent the musical state as a tuple of 5 discrete tokens:
$$\mathbf{s}_t = (r_t, c_t^{\text{drums}}, c_t^{\text{bass}}, c_t^{\text{other}}, c_t^{\text{vocals}})$$
where $r_t \in \{0, \dots, 15\}$ is the metrical tatum position in the bar (plus a special SILENCE token per stem when a stem drops out).Train a causal Transformer (6 layers, 8 heads, $d_{\text{model}}=384$) over flattened or parallel multi-stem token sequences so it learns both single-stem phrasing (e.g., drum fills on bar 4) and cross-stem interaction (e.g., sidechain bass wubs answering kick hits).Inference / Concatenative Assembly:Autoregressively predict the next tatum's stem tokens $(\hat{c}_{t+1}^{\text{drums}}, \hat{c}_{t+1}^{\text{bass}}, \hat{c}_{t+1}^{\text{other}}, \hat{c}_{t+1}^{\text{vocals}})$.Query the respective stem FAISS indices (index_drums, index_bass, etc.) to pull the best-matching real audio slices from across the entire MP3 library.Phase 2 Verification Criteria:Generate a 60-second offline WAV file (test_stem_mashup.wav) using WubGPT where each stem (drums, bass, vocals, other) is forced to pull slices from different source MP3s simultaneously, time-stretched to $128\text{ BPM}$ and pitch-locked to a common key.Phase 3: The "Infinite Wub" Web Application & 4-Stem Audio EngineObjective: Build the interactive HTML5/TypeScript frontend and FastAPI backend capable of endless, glitch-free 4-stem streaming and real-time user steering.Step 3.1: Backend Multi-Stem Streaming Service (server.py)Expose /ws/stream WebSocket endpoint.Maintain a lookahead buffer of 32 tatums (~4–8 seconds of audio) across all 4 stem channels.Stream JSON metadata packets detailing which source track and segment ID is playing on each stem (drums, bass, vocals, other), alongside pre-packed stem audio sprite sheets or binary chunks.Step 3.2: Sample-Accurate 4-Bus Web Audio Scheduler (AudioEngine.ts)CRITICAL: Use the Web Audio API hardware clock (AudioContext.currentTime) with a lookahead scheduler pattern (polling every $25\text{ ms}$ to schedule slices $150\text{ ms}$ into the future).Create 4 dedicated Web Audio sub-mix buses (GainNode + BiquadFilterNode + AnalyserNode for each of drums, bass, vocals, other) feeding into a master compressor/limiter node.For each scheduled stem segment:Create an AudioBufferSourceNode routed to its respective stem bus.Apply time-stretching to match targetBpm and pitch transposition (detune) to match the master harmonic key.Apply an AudioParam gain envelope: $3\text{ ms}$ equal-power fade-in at startTime and $3\text{ ms}$ fade-out at endTime.Step 3.3: Interactive Canvas UI ("The Wub Nebula & Stem Deck")Render two synchronized views on an HTML5 Canvas:The Multi-Concentric Infinite Jukebox Rings: 4 concentric rings representing Drums (inner), Bass, Synths/Other, and Vocals (outer), with glowing arcs showing independent cross-song jumps per stem.The 2D UMAP Timbre Map: A scatter plot of all micro-slices filterable by stem type. Users can drag 4 stem-specific "Gravity Reticles" across the map to steer textures independently (e.g., pin Drums to punchy breakbeats while dragging Bass into heavy neurofunk growls).Include live macro & per-stem controls:Per-Stem Mute / Solo / Volume & Chaos Sliders (e.g., lock Drums to $0\%$ chaos so a single track's beat plays steady, while setting Bass and Synths to $80\%$ chaos for rapid glitch-hop wub slicing).Master BPM Slider ($80 - 175\text{ BPM}$) & Master Key Lock.Slice Granularity per Stem (e.g., slice Drums at 16th-note Tatums, Bass at Micro-Segments, and Vocals at full Beats/Bars).Phase 3 Verification Criteria:Open the web UI, press Play, and run the 4-stem stream continuously for 10 minutes while toggling stem mutes and chaos sliders. Memory usage in Chrome DevTools must remain flat, and all 4 stems must stay phase-locked to the master beat grid with zero drift.Phase 4: Neural Synthetic Stem Slice Generation ("WubSynth" Extension)Objective: Allow the model to hallucinate brand-new audio slices per stem instead of only retrieving existing MP3 stem chunks.Step 4.1: Stem-Specific Latent Audio AutoencodersPass extracted stem slices (drums, bass, other) through a pre-trained RAVE (Realtime Audio Variational autoEncoder) or EnCodec latent encoder. Because stems are monophonic/isolated (especially bass and drums), latent reconstruction quality is dramatically cleaner than encoding full mixes.Step 4.2: Continuous Latent Prediction & DecodingModify WubGPT to output continuous latent vectors $\mathbf{z}_{t}^{(m)} \in \mathbb{R}^{d}$ per stem via a flow-matching head conditioned on the Transformer hidden state.Add a "Synthetic / Real Blend" slider per stem in the UI:At $0\%$ Synthetic: Use FAISS to snap $\mathbf{z}_{t}^{(m)}$ to the nearest real stem slice.At $50\%$ Synthetic: Interpolate in latent space between two real stem slices (e.g., morphing a dubstep bass growl with an acid 303 bassline: $\mathbf{z}_{\text{blend}} = 0.5\mathbf{z}_A + 0.5\mathbf{z}_B$) and decode through the neural vocoder.At $100\%$ Synthetic: Decode the raw model-predicted latent $\mathbf{z}_{t}^{(m)}$ directly into a brand-new synthetic drum hit, bass wub, or synth stab.6. Target Directory StructureInstruct the agent to scaffold the repository using this exact layout:wub/
├── data/
│   ├── raw_mp3s/               # Input electronic music MP3 collection
│   ├── stems/                  # Demucs separated WAVs (drums, bass, vocals, other)
│   ├── processed/              # Extracted JSON analyses, per-stem audio sprite sheets
│   └── indices/                # Per-stem FAISS indices & UMAP 2D coordinates
├── wub_core/
│   ├── __init__.py
│   ├── schema.py               # Pydantic models for Stems, Tatums, Segments, Tracks
│   ├── separator.py            # Demucs / audio-separator neural stem extraction wrapper
│   ├── analyzer.py             # Librosa/Madmom beat, tatum, and per-stem onset extractor
│   ├── indexer.py              # Per-stem feature normalization, FAISS, and UMAP builder
│   ├── model_markov.py         # Phase 2A: Multi-stem Metrical-Harmonic Graph generator
│   ├── model_transformer.py    # Phase 2B: Multi-stream VQ-Transformer (WubGPT)
│   └── synth_decoder.py        # Phase 4: Neural latent stem slice synthesizer
├── server/
│   ├── main.py                 # FastAPI app, REST endpoints & 4-stem WebSocket streamer
│   └── stream_manager.py       # Multi-stem lookahead buffer & state machine per client
├── frontend/
│   ├── index.html              # Single-page dark-mode electronic UI & 4-Stem Deck
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── main.ts             # UI bindings & per-stem state management
│       ├── audio_engine.ts     # Web Audio API 4-bus lookahead scheduler & crossfader
│       ├── visualizer.ts       # Canvas 2D Concentric Stem Rings + UMAP Nebula renderer
│       └── types.ts            # Mirrors wub_core/schema.py
├── scripts/
│   ├── analyze_corpus.py       # CLI to run Phase 1 stem separation & analysis pipeline
│   ├── verify_slicing.py       # Phase 1 zero-click 4-stem reconstruction test
│   └── train_wubgpt.py         # CLI to train the multi-stem sequence model
└── requirements.txt
