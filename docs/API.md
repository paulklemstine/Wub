# 🔌 Wub KG-505 API & Protocol Reference

This document documents the HTTP REST API and real-time WebSocket protocol exposed by the FastAPI server in `server/main.py`.

---

## 1. REST Endpoints

### `GET /health`
Returns system status, active corpus track counts, slice metrics, and available indices.

**Response `200 OK`**:
```json
{
  "status": "ok",
  "tracks": 1724,
  "segments": 744110,
  "tatums": 3787838,
  "stems": {
    "drums": 56,
    "bass": 56,
    "vocals": 56,
    "other": 1724
  },
  "last_scan": 1792904152.03,
  "indices": ["bass", "drums", "other", "unified", "vocals"]
}
```

---

### `GET /api/tracks`
Returns a list of all analyzed tracks with duration, BPM, Camelot musical key, and available stems.

**Response `200 OK`**:
```json
[
  {
    "track_id": "7a8b9c0d",
    "filename": "Booka Shade - Mandarine Girl.mp3",
    "duration": 342.15,
    "bpm": 124.0,
    "key": "A minor",
    "camelot": "8A",
    "stems_available": ["drums", "bass", "other", "vocals"]
  }
]
```

---

### `GET /audio/{track_id}/{slice_id}.wav`
Renders and streams a specific audio slice as a 16-bit 44.1kHz WAV with 3ms micro-fade smoothing.

- **Path Parameters**:
  - `track_id` (string): Track identifier.
  - `slice_id` (string): Slice identifier (e.g. `seg_drums_0042`).
- **Response**: `200 OK` with binary audio payload (`Content-Type: audio/wav`). If the slice is missing, returns `404 Not Found`.

---

### `GET /api/umap`
Returns 2D UMAP projection coordinates for all corpus slices to render the Slice Nebula.

**Response `200 OK`**:
```json
[
  {
    "track_id": "7a8b9c0d",
    "segment_id": "seg_drums_0042",
    "stem": "drums",
    "x": 0.428,
    "y": 0.612,
    "pitch": 0,
    "centroid": 245.0
  }
]
```

---

### `POST /api/feedback`
Records user reinforcement feedback (thumbs up / thumbs down) for a specific pattern or stem.

**Request Body**:
```json
{
  "stem": "bass",
  "rating": 1,
  "track_id": "7a8b9c0d",
  "pattern_id": "pat_bass_offbeat_001"
}
```

**Response `200 OK`**:
```json
{
  "status": "ok",
  "stem": "bass",
  "new_score": 3,
  "recorded": true
}
```

---

### `POST /api/groovebox/cycle_mode`
Switches master sequencer cycle architecture across patterns and steps.

**Request Body**:
```json
{
  "mode": "8x16"
}
```
*Valid modes*: `"4x16"`, `"8x16"`, `"16x32"`.

**Response `200 OK`**:
```json
{
  "status": "ok",
  "cycle_mode": "8x16",
  "patterns_per_cycle": 8,
  "steps_per_pattern": 16
}
```

---

## 2. WebSocket Protocol (`/ws/stream`)

The client connects to `ws://<host>:<port>/ws/stream` (or `wss://`) to receive synchronized streaming audio tatums and groovebox metadata.

### Inbound Messages (Server ➔ Client)

#### 1. `hello` (Connection handshake)
Sent immediately upon client connection.
```json
{
  "type": "hello",
  "stems": ["drums", "bass", "vocals", "other"],
  "bpm": 124.0,
  "sample_rate": 44100
}
```

#### 2. `tatum` (Scheduled metrical step event)
Sent with lookahead for sample-accurate scheduling.
```json
{
  "type": "tatum",
  "tatum_index": 48,
  "bpm": 124.0,
  "tatum_duration": 0.1209,
  "start_time": 0.06,
  "slices": {
    "drums": {
      "track_id": "7a8b9c0d",
      "segment_id": "seg_drums_0042",
      "start": 12.45,
      "duration": 0.24,
      "url": "/audio/7a8b9c0d/seg_drums_0042.wav",
      "detune_cents": 0,
      "rate": 1.0,
      "synth_blend": 0.0
    },
    "bass": null,
    "other": null,
    "vocals": null
  },
  "groovebox": {
    "section": "DROP",
    "phrase_bar": 4,
    "step_in_bar": 1,
    "total_bars": 12,
    "is_fill": false,
    "key": "8A (A minor)",
    "energy": 0.90,
    "section_bars_elapsed": 4,
    "section_total_bars": 16,
    "next_section": "GROOVE_A",
    "section_bars_left": 12,
    "cycle_mode": "8x16",
    "pattern_in_cycle": 4,
    "patterns_per_cycle": 8,
    "step_in_pattern": 1,
    "steps_per_pattern": 16,
    "cycle_total_steps": 128,
    "cycle_progress": 0.25,
    "patterns": {
      "drums": "KG-505 [Am] (four_on_floor_hats)",
      "bass": "KG-505 [Am] (rolling)",
      "other": "KG-505 [Am] (riff)",
      "vocals": "KG-505 [Am] (chop)"
    },
    "pattern_grooves": {
      "drums": "four_on_floor_hats",
      "bass": "rolling",
      "other": "riff",
      "vocals": "chop"
    },
    "pattern_grids": {
      "drums": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
      "bass": [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
      "other": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
      "vocals": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    }
  }
}
```

---

### Outbound Messages (Client ➔ Server)

| Message Type | Payload Parameters | Description |
| :--- | :--- | :--- |
| `set_bpm` | `{"type": "set_bpm", "bpm": 128.0}` | Updates tempo on the fly. |
| `set_cycle_mode` | `{"type": "set_cycle_mode", "mode": "16x32"}` | Changes master sequencer cycle architecture. |
| `set_gravity` | `{"type": "set_gravity", "stem": "bass", "x": 0.42, "y": 0.81}` | Guides slice search toward UMAP coordinates. |
| `feedback` | `{"type": "feedback", "stem": "drums", "rating": 1, "track_id": "...", "pattern_id": "..."}` | Submits 👍 (+1) or 👎 (-1) reinforcement feedback. |
| `set` | `{"type": "set", "stem": "other", "chaos": 0.35}` | Modulates temperature / stochastic variation for a stem. |
| `set_synth` | `{"type": "set_synth", "stem": "bass", "blend": 0.50}` | Blends between raw audio slices ($0.0$) and neural resynthesis ($1.0$). |
