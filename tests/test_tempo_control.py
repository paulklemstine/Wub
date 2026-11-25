"""Verify frontend tempo control propagation and anti-jump invariants."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"
ESBUILD = ROOT / "frontend" / "node_modules" / ".bin" / "esbuild"
NODE = shutil.which("node")


def test_tempo_control_contracts_in_ts() -> None:
    main_ts = (FRONTEND_SRC / "main.ts").read_text(encoding="utf-8")
    seq_ts = (FRONTEND_SRC / "client_ml_sequencer.ts").read_text(encoding="utf-8")
    kit_ts = (FRONTEND_SRC / "kit_player.ts").read_text(encoding="utf-8")
    remix_ts = (FRONTEND_SRC / "stem_remix.ts").read_text(encoding="utf-8")
    voice_ts = (FRONTEND_SRC / "stretch_voice.ts").read_text(encoding="utf-8")

    # main.ts must forward slider adjustments regardless of play state and sync slider + label without jumping
    assert "applyBpmInput" in main_ts
    assert "this.clientConductor?.setBpm(next)" in main_ts
    assert "!this.bpmSliderDragging" in main_ts
    assert "slider.value = String(Math.round(liveBpm))" in main_ts

    # client_ml_sequencer.ts must forward BPM to KitPlayer and never clobber user BPM override
    assert "this.userBpmOverride = clamped;" in seq_ts
    assert "this.kitPlayer.setBpm(clamped, true)" in seq_ts
    assert "if (this.userBpmOverride !== null) return;" in seq_ts
    assert "syncKitClockAfterTempoChange" in seq_ts

    # kit_player.ts & stem_remix.ts & stretch_voice.ts implement end-to-end live tempo changes
    assert "public setBpm(bpm: number, lockUser: boolean = true): boolean" in kit_ts
    assert "setBpm(bpm: number, lockUser: boolean = true): boolean" in remix_ts
    assert "updateTempo(changeAt: number, newRate: number" in voice_ts
    assert "rescheduleWindow(newStartAt: number, newRate: number" in voice_ts


def test_tempo_control_runtime_behavior_via_node() -> None:
    if NODE is None or not ESBUILD.exists():
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        remix_bundle = tmp / "stem_remix.mjs"
        seq_bundle = tmp / "client_ml_sequencer.mjs"

        for src, out in [
            (FRONTEND_SRC / "stem_remix.ts", remix_bundle),
            (FRONTEND_SRC / "client_ml_sequencer.ts", seq_bundle),
        ]:
            res = subprocess.run(
                [
                    str(ESBUILD),
                    str(src),
                    "--bundle",
                    "--format=esm",
                    "--platform=browser",
                    "--define:import.meta.env={}",
                    f"--outfile={out}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"esbuild failed for {src.name}: {res.stderr}"

        harness = tmp / "test_runner.mjs"
        harness.write_text(
            f"""
import assert from "node:assert/strict";
import {{ StemRemixPlayer }} from {json.dumps(remix_bundle.as_uri())};
import {{ ClientGrooveboxConductor }} from {json.dumps(seq_bundle.as_uri())};

// Minimal global browser/WebAudio shims for headless unit verification
globalThis.window = globalThis;
globalThis.localStorage = {{ getItem: () => null, setItem: () => {{}} }};
globalThis.performance = globalThis.performance ?? {{ now: () => Date.now() }};
globalThis.fetch = async () => ({{ ok: false, status: 404, json: async () => ({{}}) }});

function makeGainParam() {{
  return {{
    value: 1,
    cancelScheduledValues: () => {{}},
    setValueAtTime: () => {{}},
    linearRampToValueAtTime: () => {{}},
    exponentialRampToValueAtTime: () => {{}},
    setValueCurveAtTime: () => {{}},
    setTargetAtTime: () => {{}},
  }};
}}

let mockNow = 10.0;
const mockCtx = {{
  get currentTime() {{ return mockNow; }},
  sampleRate: 48000,
  createGain: () => ({{ gain: makeGainParam(), connect: () => {{}}, disconnect: () => {{}} }}),
  createBiquadFilter: () => ({{ type: "lowpass", frequency: makeGainParam(), Q: {{ value: 0.7 }}, connect: () => {{}}, disconnect: () => {{}} }}),
  createStereoPanner: () => ({{ pan: {{ value: 0 }}, connect: () => {{}}, disconnect: () => {{}} }}),
  createDynamicsCompressor: () => ({{ threshold: {{ value: 0 }}, knee: {{ value: 0 }}, ratio: {{ value: 1 }}, attack: {{ value: 0 }}, release: {{ value: 0 }}, connect: () => {{}} }}),
  createDelay: () => ({{ delayTime: {{ value: 0 }}, connect: () => {{}} }}),
  createBuffer: (ch, len, sr) => ({{ numberOfChannels: ch, length: len, sampleRate: sr, getChannelData: () => new Float32Array(len) }}),
}};

const mockEngine = {{
  ready: true,
  context: mockCtx,
  ensureCtx: () => mockCtx,
  sampleRate: 48000,
  now: () => mockNow,
  getStemBus: () => ({{ connect: () => {{}} }}),
  getStemOutputNode: () => ({{ connect: () => {{}} }}),
  setTargetBpmCalls: [],
  setTargetBpm(bpm) {{ this.setTargetBpmCalls.push(bpm); }},
  setStemFilter: () => {{}},
  scheduleReverbTail: () => {{}},
  scheduleSlice: () => ({{ stop: () => {{}} }}),
  decodeAudio: async () => mockCtx.createBuffer(1, 48000, 48000),
}};

// 1) Test StemRemixPlayer tempo update: pool rebuilding across extreme slider range & active voice retiming
const remix = new StemRemixPlayer(mockEngine);
const makeKit = (track_id, style, bpm, key, durationSec) => ({{
  track_id,
  style,
  bpm,
  key,
  phrases: [1, 2, 3, 4].map((idx) => ({{
    id: `${{track_id}}_p${{idx}}`,
    section: "GROOVE_A",
    energy: 0.7,
    start_bar: (idx - 1) * 4,
    duration_sec: durationSec,
    bass_root: 0,
    chroma: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    stems: {{
      drums: {{ file: `${{track_id}}/p${{idx}}_drums.opus`, rms_db: -12 }},
      bass: {{ file: `${{track_id}}/p${{idx}}_bass.opus`, rms_db: -12 }},
      other: {{ file: `${{track_id}}/p${{idx}}_other.opus`, rms_db: -14 }},
      vocals: {{ file: `${{track_id}}/p${{idx}}_vocals.opus`, rms_db: -16 }},
    }},
  }})),
}});

const cluster128 = {{ id: "c_128", bpm: 128, key: "A minor", style: "house", kits: ["k128"], pool_phrases: 4 }};
const cluster96 = {{ id: "c_96", bpm: 96, key: "A minor", style: "wave", kits: ["k96"], pool_phrases: 4 }};
remix.clusters = [cluster96, cluster128];
remix.kitById.set("k128", makeKit("k128", "house", 128, "A minor", 960 / 128));
remix.kitById.set("k96", makeKit("k96", "wave", 96, "A minor", 960 / 96));
remix.applyCluster(cluster128);
assert.equal(remix.bpm, 128);
assert.ok(remix.pools.drums.length === 4, "natural 128 BPM pool only picks 128 BPM phrases");

// Simulate live playing session with active voice + pre-scheduled next voice
const activeCalls = [];
const nextCalls = [];
const activeVoice = {{
  outputGainTarget: 1,
  updateTempo: (when, rate, endAt) => activeCalls.push({{ when, rate, endAt }}),
  fadeOut: () => {{}},
}};
const futureVoice = {{
  outputGainTarget: 1,
  rescheduleWindow: (startAt, rate, endAt) => nextCalls.push({{ startAt, rate, endAt }}),
}};

remix.started = true;
remix.sessionStartAt = 10;
remix.nextBlockAt = 17.5; // 7.5s = 4 bars at 128 BPM
remix.blockGrid = [{{ startAt: 10, endAt: 17.5, startBar: 0, bars: 4 }}];
const entry128 = remix.pools.drums[0];
remix.current.drums = {{
  entry: entry128,
  voice: activeVoice,
  startedAt: 10,
  blockIndex: 1,
  run: [entry128],
  offsets: [0],
  runEndAt: 17.5,
  preDur: 0,
  mainDurations: [7.5],
}};
remix.scheduledRuns.drums = [
  remix.current.drums,
  {{
    stem: "drums",
    run: [entry128],
    voice: futureVoice,
    startedAt: 17.5,
    offsets: [0],
    runEndAt: 25.0,
    blockIndex: 2,
    manual: false,
    preDur: 0.08,
    mainDurations: [7.5],
    crossfadeSec: 0.08,
  }},
];

// Change tempo from 128 -> 160 BPM at now=10
const changed = remix.setBpm(160, true);
assert.equal(changed, true);
assert.equal(remix.bpm, 160);
assert.equal(activeCalls.length, 1);
assert.ok(Math.abs(activeCalls[0].rate - 1.25) < 1e-6, `Expected live Stretch rate 1.25, got ${{activeCalls[0].rate}}`);
// Verify time-stretch algebra: elapsed up to changeAt=10.025 (0.025s at 128 BPM), remaining 7.475s scaled by 128/160 = 5.98s -> nextBlockAt = 16.005s
assert.ok(Math.abs(remix.nextBlockTime - 16.005) < 1e-3, `Expected nextBlockAt ~16.005, got ${{remix.nextBlockTime}}`);
assert.equal(nextCalls.length, 1);
assert.ok(Math.abs(nextCalls[0].rate - 1.25) < 1e-6);

// Widening to 175 BPM or 80 BPM must still populate every stem pool cleanly
remix.setBpm(175, true);
assert.ok(remix.pools.drums.length >= 4 && remix.pools.bass.length >= 4 && remix.pools.vocals.length >= 4);
remix.setBpm(80, true);
assert.ok(remix.pools.drums.length >= 4 && remix.pools.bass.length >= 4 && remix.pools.vocals.length >= 4);

// Cluster rotation must keep user's BPM override rather than snapping back to cluster.bpm
remix.applyCluster(cluster96);
assert.equal(remix.bpm, 80, "Cluster rotation must preserve user BPM override");

// 2) Test ClientGrooveboxConductor anti-jump guarantees (applySlotBpm, applyKitLock, start)
const conductor = new ClientGrooveboxConductor(mockEngine);
await new Promise((r) => setTimeout(r, 10));
conductor.playbackMode = "kit";
conductor.kitPlayer = {{
  currentKit: {{ track_id: "k128", key: "A minor", bpm: 128 }},
  fitBpm: 128,
  setBpmCalls: [],
  nextRemixBlockTime: 16.0,
  currentRemixBlockDurationSec: 6.0,
  get currentRemixSlotSec() {{ return 960 / this.fitBpm; }},
  setBpm(bpm, lockUser) {{
    this.setBpmCalls.push({{ bpm, lockUser }});
    this.fitBpm = bpm;
    return true;
  }},
}};

// Pre-start BPM set must survive start() & applyKitLock()
conductor.setBpm(152);
assert.equal(conductor.getBpm(), 152);
conductor.kitBlockLogQueue.push({{
  at: 18.0,
  result: {{ blockBars: 4, blockDurationSec: 7.5, slotSec: 7.5, fitBpm: 128 }},
}});
conductor.applySlotBpm(7.5);
assert.equal(conductor.getBpm(), 152, "applySlotBpm must NOT snap user BPM back to 128");
conductor.applyKitLock();
assert.equal(conductor.getBpm(), 152, "applyKitLock must NOT snap user BPM back to 128");
conductor.setBpm(140);
assert.equal(conductor.getBpm(), 140);
assert.ok(Math.abs(conductor.kitBlockLogQueue[0].result.slotSec - 960 / 140) < 1e-6);

console.log("TEMPO_NODE_TESTS_OK");
""",
            encoding="utf-8",
        )

        res = subprocess.run(
            [NODE, str(harness)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"Node test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "TEMPO_NODE_TESTS_OK" in res.stdout
