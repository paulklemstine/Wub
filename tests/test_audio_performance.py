"""Exercise the browser's cached neural layer against the original dense math."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_cached_dense_matches_reference_and_stays_bounded(tmp_path):
    module = tmp_path / "cached_dense.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/cached_dense.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={module}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert = require('node:assert/strict');
const { CachedDense } = require(process.argv[1]);
let seed = 71;
const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5);
for (const stemDim of [84, 90]) {
  const stateDim = 4 * stemDim + 64, n = stateDim * 5 + 17, width = 256;
  const weights = Float32Array.from({ length: n * width }, () => rand() * 0.1);
  const bias = Float32Array.from({ length: width }, rand);
  const input = Float32Array.from({ length: n }, rand);
  const layer = new CachedDense(n, stateDim, stemDim);
  const out = new Float32Array(width);
  const reference = () => {
    const result = new Float32Array(width);
    for (let j = 0; j < width; j++) {
      let sum = bias[j];
      for (let i = 0; i < n; i++) sum += weights[j * n + i] * input[i];
      result[j] = Math.max(0, sum);
    }
    return result;
  };
  for (let iteration = 0; iteration < 40; iteration++) {
    // Shared context, changing candidates and their deltas, then context change.
    if (iteration > 0) {
      const start = iteration === 30 ? 0 : 2 * stateDim + (iteration % 4) * stemDim;
      for (let i = start; i < start + stemDim; i++) input[i] = rand();
      for (let i = n - 17; i < n; i++) input[i] = rand();
    }
    layer.evaluate(input, weights, bias, out);
    const expected = reference();
    for (let j = 0; j < width; j++) assert.ok(Math.abs(out[j] - expected[j]) < 1e-6);
    // Repeated input must hit every chunk and retain the same output.
    const before = out.slice();
    layer.evaluate(input, weights, bias, out);
    assert.deepEqual(out, before);
  }
  for (const chunk of layer.chunks) assert.ok(chunk.entries.length <= 16);
  layer.clear();
  for (const chunk of layer.chunks) assert.equal(chunk.entries.length, 0);
  layer.evaluate(input, weights, bias, out);
  assert.deepEqual(out, reference());
}
"""
    subprocess.run(["node", "-e", script, str(module)], check=True, capture_output=True, text=True)


def test_worker_shares_inflight_decodes_and_warms_without_pcm_transfer(tmp_path):
    module = tmp_path / "kit_worker.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/kit_audio_worker.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={module}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert = require('node:assert/strict');
const messages = [];
global.self = { postMessage: (message, transfer) => messages.push({message, transfer}) };
let fetches = 0, decodes = 0, contexts = 0;
global.fetch = async (_url, options) => {
  assert.ok(options.signal);
  fetches++;
  return { ok: true, arrayBuffer: async () => new ArrayBuffer(8) };
};
global.OfflineAudioContext = class {
  constructor() { contexts++; }
  async decodeAudioData() {
    decodes++;
    await new Promise(resolve => setImmediate(resolve));
    return { numberOfChannels: 2, length: 100, getChannelData: () => new Float32Array(100) };
  }
};
require(process.argv[1]);
const send = data => self.onmessage({data});
const flush = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  send({type:'init', sampleRate: 44100});
  const item = {key:'phrase:drums', url:'/phrase.opus', preUrl:null};
  send({type:'warm', items:[item]});
  send({type:'load', id:1, items:[item]});
  send({type:'load', id:2, items:[item]});
  for (let i = 0; i < 8; i++) await flush();
  assert.equal(fetches, 1);
  assert.equal(decodes, 1);
  assert.equal(messages.length, 2); // warm must not allocate/transfer a run
  for (const {message, transfer} of messages) {
    assert.equal(message.type, 'run');
    assert.equal(message.samples.length, 2);
    assert.equal(transfer.length, 2);
  }
  send({type:'load', id:3, items:[{...item, key:'other', url:'/other.opus'}]});
  for (let i = 0; i < 8; i++) await flush();
  assert.equal(fetches, 2);
  assert.equal(contexts, 1); // decode context is reused
  send({type:'load', id:4, items:[item]});
  for (let i = 0; i < 4; i++) await flush();
  assert.equal(fetches, 2); // warmed PCM survives transferable run returns
  assert.equal(messages.at(-1).message.type, 'run');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run(["node", "-e", script, str(module)], check=True, capture_output=True, text=True)


def test_worklet_pcm_transfer_and_fallback_assembly(tmp_path):
    import json

    for name in ("stretch_voice", "stem_remix"):
        subprocess.run([
            str(ROOT / "frontend/node_modules/.bin/esbuild"),
            str(ROOT / f"frontend/src/{name}.ts"), "--bundle", "--platform=browser",
            "--format=esm", "--define:import.meta.env={}", f"--outfile={tmp_path / (name + '.mjs')}",
        ], check=True, capture_output=True, text=True)
    script = "import { StretchVoice } from " + json.dumps((tmp_path / "stretch_voice.mjs").as_uri()) + ";\n"
    script += "import { StemRemixPlayer } from " + json.dumps((tmp_path / "stem_remix.mjs").as_uri()) + ";\n"
    script += r"""
import assert from 'node:assert/strict';
const received = [];
const node = { addBuffers: async (channels, transfer) => {
  received.push(structuredClone(channels, {transfer}));
}};
const voice = new StretchVoice(node, {}, {}, {});
const left = new Float32Array([0.1, 0.2]), right = new Float32Array([0.3, 0.4]);
await voice.loadSamples([left, right], true);
assert.equal(left.byteLength, 0);
assert.equal(right.byteLength, 0);
assert.deepEqual(received[0][0], new Float32Array([0.1, 0.2]));
const retained = new Float32Array([0.5]);
await voice.loadSamples(retained);
assert.equal(retained.byteLength, 4); // default preserves callers' PCM
const shared = new Float32Array([0.7, 0.8]);
await voice.loadSamples([shared, shared], true); // deduplicated transfer list
assert.equal(shared.byteLength, 0);

// Fallback run buffers must own PCM: transferring them must not detach the
// cached AudioBuffer's channels. Concurrent runs yield separately.
const player = Object.create(StemRemixPlayer.prototype);
player.pcmAssembly = Promise.resolve();
player.failedKits = new Set();
const source = new Float32Array([0.2, 0.3, 0.4]);
player.loadAudio = async () => ({main: {
  numberOfChannels: 1, sampleRate: 44100, length: source.length,
  getChannelData: () => source,
}, pre: null});
let yields = 0;
globalThis.scheduler = {yield: async () => { yields++; }};
const entry = {kit:{track_id:'track'}, phrase:{id:'a'}};
const runs = await Promise.all([
  player.loadRunAudioMain([entry]), player.loadRunAudioMain([entry]),
]);
assert.equal(yields, 2);
for (const run of runs) {
  assert.deepEqual(run.samples[0], source);
  await voice.loadSamples(run.samples, true);
}
assert.equal(source.byteLength, 12);
assert.deepEqual(source, new Float32Array([0.2, 0.3, 0.4]));
"""
    subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True)


def test_nebula_raster_cache_invalidates_on_resize_and_data_change(tmp_path):
    module = tmp_path / "visualizer.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/visualizer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={module}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert = require('node:assert/strict');
const { NebulaRenderer } = require(process.argv[1]);
function canvas() {
  const result = {width:400, height:300, clientWidth:400, clientHeight:300,
    style:{}, addEventListener(){},
    getBoundingClientRect(){return {width:this.clientWidth, height:this.clientHeight};}};
  result.fills = 0;
  const ctx = new Proxy({canvas:result}, {get(target, key) {
    if (key in target) return target[key];
    if (key === 'fillRect') return () => result.fills++;
    if (String(key).startsWith('create')) return () => ({addColorStop(){}});
    if (key === 'measureText') return () => ({width:20});
    return () => {};
  }});
  result.getContext = () => ctx;
  return result;
}
global.window = {devicePixelRatio:3};
global.document = {createElement:canvas};
const surface = canvas();
const nebula = new NebulaRenderer(surface);
const points = Array.from({length:10000}, (_,i) => ({stem:'drums',trackId:'x',x:(i%100)/100,y:Math.floor(i/100)/100}));
nebula.setPoints(points);
const frame = {timeSec:1,bpm:120};
nebula.render(frame);
assert.equal(nebula.pointLayer.fills, 10000);
assert.equal(surface.width, 600); // high-DPI budget
nebula.render({...frame,timeSec:2});
assert.equal(nebula.pointLayer.fills, 10000); // no per-point work next frame
surface.clientWidth = 500;
nebula.render({...frame,timeSec:3});
assert.equal(nebula.pointLayer.fills, 20000);
nebula.setPoints(points.slice(0,10));
nebula.render({...frame,timeSec:4});
assert.equal(nebula.pointLayer.fills, 20010);
"""
    subprocess.run(["node", "-e", script, str(module)], check=True, capture_output=True, text=True)


def test_visual_budget_staggers_heavy_painters():
    """Audio stays on every frame; heavy 2D/3D painters alternate under load."""
    source = (ROOT / "frontend/src/main.ts").read_text(encoding="utf-8")
    assert "frameIntervalEma" in source
    assert "visualDegraded" in source
    assert "const paintA" in source and "const paintB" in source
    assert source.count("&& paintA") >= 4
    assert source.count("&& paintB") >= 3
    # The audio clock and glue bus must never be staggered.
    assert "this.engine.updateGlue(audioNow);" in source
    assert "this.transitionProbe.update(audioNow, this.engine);" in source


def test_lightshow_dual_layer_starts_off():
    source = (ROOT / "frontend/src/lightshow.ts").read_text(encoding="utf-8")
    assert "dualLayer: true" in source
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert 'id="btn-ls-dual" class="lightshow-btn active"' in html
    assert '>2-LAYER: ON</button>' in html


def test_lightshow_boot_gain_zero_speed_left():
    source = (ROOT / "frontend/src/lightshow.ts").read_text(encoding="utf-8")
    assert "intensity: 0" in source
    assert "speed: 0.2" in source
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert 'id="ls-intensity-slider"' in html
    assert 'value="0"' in html
    assert 'id="ls-speed-slider"' in html
    assert 'value="0.2"' in html
