// CHIMERA Physical-Modeling Vocal Tract AudioWorkletProcessor.
// Zero-dependency ES module loaded from ./chimera_worklet.js via AudioWorklet.addModule.

const ARTIC_LEN = 17;
const SOURCE_LEN = 12;
const BODY_SWITCH_HOLD_SEC = 0.75;
const BODY_FADE_QUANTUMS = 4; // ~11 ms dip at 44.1 kHz / 128

function quantizeFemaleMix(v) {
  if (v < 0.25) return 0.0;
  if (v < 0.75) return 0.5;
  return 1.0;
}

class WubChimeraProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.x = null;
    this.h = 0;
    this.ready = false;
    this.failed = false;

    this.targetArtic = new Float64Array(ARTIC_LEN);
    this.targetSource = new Float64Array(SOURCE_LEN);
    this.hasParams = false;
    this.gate = 0.0;

    // 3-bucket body resonance (0.0 male, 0.5 blended, 1.0 female) with hysteresis
    // so engine_set_body (which allocates a new airway) fires rarely and glitch-free.
    this.activeBody = 0.5;
    this.pendingBody = 0.5;
    this.pendingBodySince = 0.0;
    this.clockSec = 0.0;
    this.fadeQuantaLeft = 0;

    this.port.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg || this.failed) return;
      if (msg.type === "wasm" && msg.bytes) {
        this.initWasm(msg.bytes);
      } else if (msg.type === "params") {
        const a = msg.artic;
        const s = msg.source;
        if (a && a.length >= ARTIC_LEN) {
          for (let i = 0; i < ARTIC_LEN; i++) this.targetArtic[i] = a[i];
        }
        if (s && s.length >= SOURCE_LEN) {
          for (let i = 0; i < SOURCE_LEN; i++) this.targetSource[i] = s[i];
        }
        if (typeof msg.gate === "number") this.gate = msg.gate;
        if (typeof msg.femaleMix === "number") {
          const b = quantizeFemaleMix(msg.femaleMix);
          if (b !== this.pendingBody) {
            this.pendingBody = b;
            this.pendingBodySince = this.clockSec;
          }
        }
        this.hasParams = true;
      } else if (msg.type === "gate") {
        this.gate = typeof msg.level === "number" ? msg.level : 0.0;
      }
    };
  }

  async initWasm(wasmBytes) {
    try {
      const result = await WebAssembly.instantiate(wasmBytes, {});
      this.x = result.instance.exports;
      this.h = this.x.engine_new(sampleRate);
      // Start at neutral mid-body (0.5 female, 0.0 child).
      this.x.engine_set_body(this.h, 0.5, 0.0);
      this.activeBody = 0.5;
      this.x.engine_set_gate(this.h, this.gate);
      this.ready = true;
      this.port.postMessage({ type: "ready" });
    } catch (err) {
      this.failed = true;
      this.port.postMessage({ type: "error", message: String(err) });
    }
  }

  valuesView() {
    return new Float64Array(this.x.memory.buffer, this.x.engine_values(this.h), 32);
  }

  applyTargetsToEngine() {
    if (!this.hasParams) return;
    const x = this.x;
    const h = this.h;

    const aVals = this.valuesView();
    aVals.fill(NaN);
    for (let i = 0; i < ARTIC_LEN; i++) {
      aVals[i] = this.targetArtic[i];
    }
    x.engine_set_artic(h, 2);

    const sVals = this.valuesView();
    sVals.fill(NaN);
    for (let i = 0; i < SOURCE_LEN; i++) {
      sVals[i] = this.targetSource[i];
    }
    x.engine_set_source(h);
    x.engine_set_gate(h, this.gate);
  }

  process(_inputs, outputs) {
    const out = outputs[0]?.[0];
    if (!out) return true;
    const n = out.length;
    this.clockSec += n / sampleRate;

    if (!this.ready || this.failed || !this.hasParams) {
      out.fill(0);
      return true;
    }

    const x = this.x;
    const h = this.h;

    try {
      // Optional slow body-bucket transition with short mute fade
      if (
        this.pendingBody !== this.activeBody &&
        (this.clockSec - this.pendingBodySince >= BODY_SWITCH_HOLD_SEC || this.gate <= 0.01)
      ) {
        x.engine_set_body(h, this.pendingBody, 0.0);
        this.activeBody = this.pendingBody;
        this.fadeQuantaLeft = BODY_FADE_QUANTUMS;
      }

      this.applyTargetsToEngine();

      // engine_render expects <= 4096 samples (typically 128)
      const ptr = x.engine_render(h, n);
      const rendered = new Float32Array(x.memory.buffer, ptr, n);
      const envGain =
        this.fadeQuantaLeft > 0
          ? 1.0 - (this.fadeQuantaLeft-- / (BODY_FADE_QUANTUMS + 1))
          : 1.0;

      for (let i = 0; i < n; i++) {
        const s = rendered[i];
        out[i] = Number.isFinite(s) ? s * envGain : 0.0;
      }

      // Duplicate mono output to right channel if stereo output exists
      if (outputs[0].length > 1 && outputs[0][1]) {
        outputs[0][1].set(out);
      }
    } catch (err) {
      this.failed = true;
      out.fill(0);
      this.port.postMessage({ type: "error", message: String(err) });
    }

    return true;
  }
}

registerProcessor("wub-chimera-processor", WubChimeraProcessor);
