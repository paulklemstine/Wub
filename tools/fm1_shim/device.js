// Wub device shim — a firmware-shaped runtime that runs *in the emulator*.
//
// This is the "what if the FM-1 could play the kit bank" spike, without
// pretending the hardware can decode Opus or stretch audio. It fetches the
// real kit manifest and real Opus phrase files over HTTPS (GitHub raw, or a
// local bank served by serve.py), decodes them with the browser's own codec,
// and plays four stem streams on one 4-bar grid. Every CTL-1 message is wired:
// keys, the four knob banks, ENV, LFO, MASTER, PLAY/REC, PRESETS, MIDI clock,
// program change and aftertouch. Nothing here is shipped to the web player; it
// is a bench for the firmware idea.
//
// The on-screen panel is the emulator version of the FM-1. Press CONNECT MIDI
// to drive the exact same handler from a real CTL-1 over USB.

const STEMS = ["drums", "bass", "vocals", "other"];
const STEM_LABELS = { drums: "DRUMS", bass: "BASS", vocals: "VOCALS", other: "OTHER" };
const CC_BANKS = {
  A: [74, 71, 1, 7],
  B: [73, 72, 75, 76],
  C: [16, 17, 18, 19],
  D: [20, 21, 22, 23],
};
const DEFAULT_BASE =
  "https://raw.githubusercontent.com/paulklemstine/Wub/main/frontend/public/data/kits";
const KEY_BASE = 53; // CTL-1 lowest key: F3
const KEY_COUNT = 27;

const $ = (id) => document.getElementById(id);

const state = {
  base: new URLSearchParams(location.search).get("base") || DEFAULT_BASE,
  bank: "C",
  focus: "drums",
  octave: 0,
  bpm: 126,
  playing: false,
  loading: true,
  error: null,
  manifest: null,
  kits: [],
  pools: { drums: [], bass: [], vocals: [], other: [] },
  slot: { drums: 0, bass: 0, vocals: 0, other: 0 },
  current: { drums: null, bass: null, vocals: null, other: null },
  playing_phrase: { drums: null, bass: null, vocals: null, other: null },
  pending: {},
  buffering: {},
  midi: { supported: false, connected: false, deviceName: null, devices: [], last: null },
  sustain: false,
  mod: 0,
  program: 0,
  clockTicks: 0,
  lastTickMs: 0,
  keyHeld: new Set(),
};

// ---------------------------------------------------------------------------
// audio graph
// ---------------------------------------------------------------------------
let ctx = null;
let master = null;
let limiter = null;
let analyser = null;
let meterData = null;
const stemGain = {};
const stemFilter = {};
const activeSource = {}; // stem -> { src, gain }
const bufferCache = new Map(); // url -> AudioBuffer
const decoding = new Map(); // url -> Promise<AudioBuffer>

let startAt = 0;
let schedulerId = null;
let lastBar = -1;
let tickTimer = null;

const BAR_BEATS = 4;

function barSec() {
  return (BAR_BEATS * 60) / Math.max(40, state.bpm);
}

function ensureAudio() {
  if (ctx !== null) return ctx;
  const AC = window.AudioContext || window.webkitAudioContext;
  ctx = new AC({ latencyHint: "interactive" });
  master = ctx.createGain();
  master.gain.value = 0.9;
  limiter = ctx.createDynamicsCompressor();
  limiter.threshold.value = -8;
  limiter.knee.value = 6;
  limiter.ratio.value = 12;
  limiter.attack.value = 0.003;
  limiter.release.value = 0.25;
  analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  meterData = new Float32Array(analyser.fftSize);
  master.connect(limiter).connect(analyser).connect(ctx.destination);
  for (const stem of STEMS) {
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 20000;
    filter.Q.value = 0.6;
    const gain = ctx.createGain();
    gain.gain.value = 0.85;
    filter.connect(gain).connect(master);
    stemFilter[stem] = filter;
    stemGain[stem] = gain;
  }
  return ctx;
}

function uiMeter() {
  if (analyser === null) return 0;
  analyser.getFloatTimeDomainData(meterData);
  let peak = 0;
  for (let i = 0; i < meterData.length; i += 4) {
    peak = Math.max(peak, Math.abs(meterData[i]));
  }
  return peak;
}

// ---------------------------------------------------------------------------
// bank loading
// ---------------------------------------------------------------------------
function naturalCompare(a, b) {
  const ax = String(a).match(/(\d+|\D+)/g) ?? [String(a)];
  const bx = String(b).match(/(\d+|\D+)/g) ?? [String(b)];
  const n = Math.min(ax.length, bx.length);
  for (let i = 0; i < n; i++) {
    const an = /^\d/.test(ax[i]);
    const bn = /^\d/.test(bx[i]);
    if (an && bn) {
      const diff = Number(ax[i]) - Number(bx[i]);
      if (diff !== 0) return diff;
    } else if (ax[i] !== bx[i]) {
      return ax[i] < bx[i] ? -1 : 1;
    }
  }
  return ax.length - bx.length;
}

function joinUrl(base, file) {
  if (/^https?:/.test(file)) return file;
  return `${base.replace(/\/+$/, "")}/${file.replace(/^\/+/, "")}`;
}

async function loadManifest() {
  const url = joinUrl(state.base, "kit_manifest.json");
  setStatus(`fetching ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest ${res.status}`);
  const doc = await res.json();
  state.manifest = doc;
  state.kits = Array.isArray(doc.kits) ? doc.kits : [];
  if (state.kits.length === 0) throw new Error("manifest has no kits");

  for (const stem of STEMS) state.pools[stem] = [];
  for (const kit of state.kits) {
    for (const phrase of kit.phrases ?? []) {
      for (const stem of STEMS) {
        const part = phrase.stems?.[stem];
        if (!part || !part.file) continue;
        state.pools[stem].push({
          kitId: kit.track_id,
          title: kit.title,
          trackId: kit.track_id,
          phraseId: phrase.id,
          stem,
          section: phrase.section,
          energy: phrase.energy,
          bpm: kit.bpm,
          key: kit.key,
          style: kit.style,
          duration: phrase.duration_sec,
          file: part.file,
          url: joinUrl(state.base, part.file),
        });
      }
    }
  }
  for (const stem of STEMS) {
    state.pools[stem].sort((a, b) => {
      const track = naturalCompare(a.trackId, b.trackId);
      return track !== 0 ? track : naturalCompare(a.phraseId, b.phraseId);
    });
  }

  // Start on a kit that has all four stems so the first bar is a real mix.
  const firstKit = state.kits.find((kit) =>
    (kit.phrases ?? []).some((phrase) => STEMS.every((stem) => phrase.stems?.[stem]?.file)),
  ) ?? state.kits[0];
  state.current = { drums: null, bass: null, vocals: null, other: null };
  for (const stem of STEMS) {
    const index = state.pools[stem].findIndex(
      (point) => point.phraseId === firstKit.phrases?.[0]?.id,
    );
    state.slot[stem] = index >= 0 ? index : 0;
    state.current[stem] = state.pools[stem][state.slot[stem]] ?? null;
  }
  state.bpm = Math.round(firstKit.bpm ?? 126);
  state.loading = false;
  setStatus(
    `${state.kits.length} kits · ${state.pools.drums.length}/${state.pools.bass.length}/` +
      `${state.pools.vocals.length}/${state.pools.other.length} phrases`,
  );
  render();
  for (const stem of STEMS) prefetch(stem, state.slot[stem]);
}

function phraseFor(stem, slot) {
  const pool = state.pools[stem];
  if (pool.length === 0) return null;
  const index = ((Math.round(slot) % pool.length) + pool.length) % pool.length;
  return pool[index];
}

function prefetch(stem, slot) {
  const phrase = phraseFor(stem, slot);
  if (phrase === null) return;
  void bufferFor(phrase).catch(() => {});
  const next = phraseFor(stem, slot + 1);
  if (next !== null) void bufferFor(next).catch(() => {});
}

async function bufferFor(phrase) {
  const cached = bufferCache.get(phrase.url);
  if (cached) return cached;
  const inflight = decoding.get(phrase.url);
  if (inflight) return inflight;
  ensureAudio();
  const job = (async () => {
    const res = await fetch(phrase.url);
    if (!res.ok) throw new Error(`${res.status} ${phrase.url}`);
    const bytes = await res.arrayBuffer();
    const buffer = await ctx.decodeAudioData(bytes);
    bufferCache.set(phrase.url, buffer);
    if (bufferCache.size > 48) {
      const oldest = bufferCache.keys().next().value;
      bufferCache.delete(oldest);
    }
    return buffer;
  })();
  decoding.set(phrase.url, job);
  try {
    return await job;
  } finally {
    decoding.delete(phrase.url);
  }
}

// ---------------------------------------------------------------------------
// playback: four stems, one 4-bar grid
// ---------------------------------------------------------------------------
function rateFor(buffer) {
  // Fit the phrase's own 4 bars onto the grid. The shim only resamples tempo
  // (no stretch, no key shift) — that is the honest limit of this spike.
  const grid = BAR_BEATS * barSec();
  return Math.min(1.25, Math.max(0.75, buffer.duration / grid));
}

function startSource(stem, phrase, buffer, at) {
  const previous = activeSource[stem];
  if (previous) {
    previous.gain.gain.cancelScheduledValues(at);
    previous.gain.gain.setValueAtTime(previous.gain.gain.value, at);
    previous.gain.gain.linearRampToValueAtTime(0.0001, at + 0.012);
    try {
      previous.src.stop(at + 0.02);
    } catch (err) {
      /* already stopped */
    }
  }
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.loop = true;
  src.playbackRate.value = rateFor(buffer);
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.0001, at);
  gain.gain.linearRampToValueAtTime(1, at + 0.01);
  src.connect(gain).connect(stemFilter[stem]);
  src.start(at);
  activeSource[stem] = { src, gain };
  state.playing_phrase[stem] = phrase;
}

async function start() {
  ensureAudio();
  if (ctx.state === "suspended") await ctx.resume();
  if (state.playing) return;
  state.playing = true;
  startAt = ctx.currentTime + 0.12;
  lastBar = 0;
  state.pending = {};
  // Warm all four streams before the first bar so the grid starts together.
  const buffers = await Promise.all(
    STEMS.map(async (stem) => {
      const phrase = state.current[stem];
      if (phrase === null) return null;
      try {
        state.buffering[stem] = true;
        return { stem, phrase, buffer: await bufferFor(phrase) };
      } catch (err) {
        return null;
      } finally {
        state.buffering[stem] = false;
      }
    }),
  );
  for (const entry of buffers) {
    if (entry) startSource(entry.stem, entry.phrase, entry.buffer, startAt);
  }
  if (!state.playing) return;
  schedulerId = setInterval(schedule, 50);
  schedule();
  render();
  setStatus("playing");
}

function schedule() {
  if (!state.playing || ctx === null) return;
  const now = ctx.currentTime;
  const bar = Math.floor((now - startAt) / barSec());
  if (bar <= lastBar) return;
  // A commit lands exactly on the first bar boundary we have not played yet.
  const at = startAt + (lastBar + 1) * barSec();
  lastBar = Math.floor((now - startAt) / barSec());
  for (const stem of STEMS) {
    const pending = state.pending[stem];
    if (!pending) continue;
    delete state.pending[stem];
    const scheduledAt = Math.max(at, now + 0.02);
    void (async () => {
      try {
        const buffer = await bufferFor(pending);
        startSource(stem, pending, buffer, scheduledAt);
        render();
      } catch (err) {
        setStatus(`load failed: ${pending.url.split("/").pop()}`);
      }
    })();
  }
  render();
}

function stop() {
  if (ctx === null) return;
  state.playing = false;
  if (schedulerId !== null) clearInterval(schedulerId);
  schedulerId = null;
  const now = ctx.currentTime;
  for (const stem of STEMS) {
    const active = activeSource[stem];
    if (!active) continue;
    active.gain.gain.cancelScheduledValues(now);
    active.gain.gain.setValueAtTime(active.gain.gain.value, now);
    active.gain.gain.linearRampToValueAtTime(0.0001, now + 0.04);
    try {
      active.src.stop(now + 0.05);
    } catch (err) {
      /* already stopped */
    }
    delete activeSource[stem];
    state.playing_phrase[stem] = null;
  }
  render();
  setStatus("stopped");
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------
function commit(stem, phrase) {
  if (phrase === null) return;
  state.current[stem] = phrase;
  state.slot[stem] = state.pools[stem].indexOf(phrase);
  prefetch(stem, state.slot[stem]);
  if (!state.playing) {
    render();
    return;
  }
  state.pending[stem] = phrase;
  render();
}

function scrollStem(stem, delta) {
  const pool = state.pools[stem];
  if (pool.length === 0) return;
  const next = (((state.slot[stem] + delta) % pool.length) + pool.length) % pool.length;
  commit(stem, pool[next]);
  state.focus = stem;
}

function jumpStem(stem, slot) {
  const pool = state.pools[stem];
  if (pool.length === 0) return;
  const index = ((slot % pool.length) + pool.length) % pool.length;
  commit(stem, pool[index]);
  state.focus = stem;
}

function jumpKit(index) {
  const kit = state.kits[((index % state.kits.length) + state.kits.length) % state.kits.length];
  if (!kit) return;
  for (const stem of STEMS) {
    const phrase = (kit.phrases ?? []).find((p) => p.stems?.[stem]?.file);
    if (!phrase) continue;
    const pool = state.pools[stem];
    const slot = pool.findIndex((point) => point.phraseId === phrase.id);
    if (slot >= 0) jumpStem(stem, slot);
  }
  state.bpm = Math.round(kit.bpm ?? state.bpm);
}

function knobToStem(cc) {
  const index = (CC_BANKS[state.bank] ?? CC_BANKS.C).indexOf(cc);
  return index >= 0 ? STEMS[index] : null;
}

/** Virtual slider: absolute 0..127 jump (the UI is not an endless encoder). */
function handleSlider(stem, value) {
  const pool = state.pools[stem];
  if (pool.length === 0) return;
  const slot = Math.round((value * (pool.length - 1)) / 127);
  jumpStem(stem, slot);
  lastCc.set((CC_BANKS[state.bank] ?? CC_BANKS.C)[STEMS.indexOf(stem)], value);
}

// Endless encoders: keep the last raw value per CC and scroll by its delta.
const lastCc = new Map();

function handleKnob(stem, cc, value) {
  const previous = lastCc.get(cc);
  lastCc.set(cc, value);
  let delta;
  if (previous === undefined) {
    // First touch after a bank/page change seeds from the absolute position.
    const pool = state.pools[stem];
    const slot = Math.round((value * (pool.length - 1)) / 127);
    jumpStem(stem, slot);
    return;
  }
  delta = value - previous;
  if (delta > 64) delta -= 128;
  else if (delta < -64) delta += 128;
  if (delta === 0) return;
  scrollStem(stem, delta);
}

function setSustain(on) {
  state.sustain = on;
  if (ctx === null) return;
  const now = ctx.currentTime;
  for (const stem of STEMS) {
    const gain = stemGain[stem];
    const duck = on && (stem === "drums" || stem === "bass") ? 0.28 : 0.85;
    gain.gain.cancelScheduledValues(now);
    gain.gain.setTargetAtTime(duck, now, 0.05);
  }
  render();
}

function setMod(value) {
  state.mod = value;
  if (ctx === null) return;
  const hz = 200 * Math.pow(90, value / 127); // 200 Hz .. 18 kHz
  const now = ctx.currentTime;
  for (const stem of STEMS) stemFilter[stem].frequency.setTargetAtTime(hz, now, 0.02);
  render();
}

function setMaster(value) {
  if (ctx === null) return;
  master.gain.setTargetAtTime(0.15 + (value / 127) * 0.85, ctx.currentTime, 0.03);
}

function handleMessage(data) {
  if (!data || data.length === 0) return;
  const status = data[0];
  const type = status & 0xf0;
  state.midi.last = [...data];
  if (status === 0xf8) {
    state.clockTicks += 1;
    const now = performance.now();
    if (state.lastTickMs > 0) {
      const delta = now - state.lastTickMs;
      if (delta > 4 && delta < 400) {
        const bpm = 60000 / (24 * delta);
        state.bpm = Math.round(Math.max(40, Math.min(240, bpm)));
      }
    }
    state.lastTickMs = now;
    return;
  }
  if (status === 0xfa) {
    void start();
    return;
  }
  if (status === 0xfc) {
    stop();
    return;
  }
  if (type === 0x90 && data.length >= 3) {
    const note = data[1] & 0x7f;
    const velocity = data[2] & 0x7f;
    if (velocity === 0) {
      state.keyHeld.delete(note);
    } else {
      state.keyHeld.add(note);
      const index = note - KEY_BASE + state.octave * KEY_COUNT;
      jumpStem(state.focus, index);
    }
    render();
    return;
  }
  if (type === 0x80) {
    state.keyHeld.delete(data[1] & 0x7f);
    render();
    return;
  }
  if (type === 0xb0 && data.length >= 3) {
    const cc = data[1] & 0x7f;
    const value = data[2] & 0x7f;
    if (cc === 64) {
      setSustain(value >= 64);
      return;
    }
    if (cc === 1) {
      setMod(value);
      return;
    }
    if (cc === 123) {
      stop();
      return;
    }
    if (cc === 7) {
      setMaster(value);
      return;
    }
    const stem = knobToStem(cc);
    if (stem !== null) {
      handleKnob(stem, cc, value);
      return;
    }
    // MASTER on the Sound page can be any controller: treat unknown CC as volume.
    setMaster(value);
    return;
  }
  if (type === 0xc0) {
    state.program = data[1] & 0x7f;
    jumpKit(state.program);
    render();
    return;
  }
  if (type === 0xd0) {
    setMod(Math.max(state.mod, data[1] & 0x7f));
    return;
  }
  if (type === 0xa0 && data.length >= 3) {
    const stem = state.focus;
    if (ctx !== null) {
      const gain = stemGain[stem];
      gain.gain.setTargetAtTime(0.85 + ((data[2] & 0x7f) / 127) * 0.3, ctx.currentTime, 0.05);
    }
    return;
  }
  render();
}

// ---------------------------------------------------------------------------
// real MIDI
// ---------------------------------------------------------------------------
let midiAccess = null;
let midiInput = null;
let autoPick = true;

async function connectMidi() {
  if (typeof navigator.requestMIDIAccess !== "function") {
    setStatus("Web MIDI unavailable (Chrome/Edge over https or localhost)");
    return false;
  }
  try {
    midiAccess = await navigator.requestMIDIAccess({ sysex: false });
  } catch (err) {
    setStatus(`MIDI permission denied: ${err.name ?? err}`);
    return false;
  }
  state.midi.supported = true;
  midiAccess.onstatechange = () => refreshDevices(true);
  refreshDevices();
  if (midiInput === null && autoPick) autoSelect();
  return true;
}

function autoSelect() {
  if (midiAccess === null) return;
  const ports = [...midiAccess.inputs.values()];
  if (ports.length === 0) {
    setStatus("no MIDI inputs — plug the FM-1 in or use the on-screen panel");
    return;
  }
  const preferred = ports.find((p) => /ctl.?1|fm.?1/i.test(p.name ?? "")) ?? ports[0];
  attach(preferred);
}

function attach(port) {
  if (midiInput !== null) midiInput.onmidimessage = null;
  midiInput = port;
  port.onmidimessage = (event) => {
    if (event.data) handleMessage([...event.data]);
    render();
  };
  state.midi.connected = true;
  state.midi.deviceName = port.name ?? port.id;
  render();
  setStatus(`MIDI in: ${state.midi.deviceName}`);
}

function refreshDevices(reselect = false) {
  if (midiAccess === null) return;
  const devices = [...midiAccess.inputs.values()].map((port) => ({
    id: port.id,
    name: port.name ?? port.id,
  }));
  state.midi.devices = devices;
  const present = midiInput !== null && devices.some((d) => d.id === midiInput.id);
  if (!present) {
    if (midiInput !== null) midiInput.onmidimessage = null;
    midiInput = null;
    state.midi.connected = false;
    state.midi.deviceName = null;
  }
  if (reselect && midiInput === null && autoPick) autoSelect();
  render();
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function setStatus(text) {
  const el = $("status");
  if (el) el.textContent = text;
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function render() {
  const body = $("bank-info");
  if (body) {
    body.textContent = state.loading
      ? "loading manifest…"
      : `${state.base.includes("raw.githubusercontent") ? "github" : "local"} · ${state.kits.length} kits`;
  }
  const midi = $("midi-state");
  if (midi) {
    midi.textContent = state.midi.connected
      ? `MIDI: ${state.midi.deviceName}`
      : state.midi.supported
        ? "MIDI: no device"
        : "MIDI: on-screen panel";
  }
  const bank = $("bank-select");
  if (bank) bank.value = state.bank;
  const oct = $("octave-value");
  if (oct) oct.textContent = state.octave > 0 ? `+${state.octave}` : String(state.octave);
  const bpm = $("bpm-value");
  if (bpm) bpm.textContent = `${state.bpm} BPM`;
  const focus = $("focus-value");
  if (focus) focus.textContent = STEM_LABELS[state.focus];

  for (const stem of STEMS) {
    const row = $(`stem-${stem}`);
    if (row) {
      const current = state.current[stem];
      const playing = state.playing_phrase[stem];
      const live = playing ?? current;
      row.classList.toggle("is-focus", state.focus === stem);
      row.classList.toggle("is-playing", state.playing && playing !== null);
      row.classList.toggle("is-buffering", Boolean(state.buffering[stem]));
      const title = row.querySelector(".stem-title");
      if (title) title.textContent = live ? live.title ?? live.trackId : "—";
      const phrase = row.querySelector(".stem-phrase");
      if (phrase) {
        phrase.textContent = live
          ? `${live.section ?? "?"} · ${live.phraseId?.split("_").pop()} · ${pct(live.energy ?? 0)}`
          : "no phrase";
      }
      const slot = row.querySelector(".stem-slot");
      if (slot) slot.textContent = `${state.slot[stem] + 1}/${state.pools[stem].length}`;
      const bar = row.querySelector(".stem-bar");
      if (bar) bar.style.width = `${((state.slot[stem] + 1) / Math.max(1, state.pools[stem].length)) * 100}%`;
    }
    const knob = $(`knob-${stem}`);
    if (knob) knob.value = knobValueFor(stem);
    const knobLabel = $(`knob-value-${stem}`);
    if (knobLabel) knobLabel.textContent = String(Math.round(knobValueFor(stem)));
  }

  const transport = $("btn-play");
  if (transport) {
    transport.textContent = state.playing ? "STOP" : "PLAY";
    transport.classList.toggle("active", state.playing);
  }
  const meter = $("meter-fill");
  if (meter) meter.style.width = `${Math.min(100, uiMeter() * 140)}%`;
  const sustainBtn = $("btn-env");
  if (sustainBtn) sustainBtn.classList.toggle("active", state.sustain);
  const modFill = $("mod-fill");
  if (modFill) modFill.style.width = `${(state.mod / 127) * 100}%`;
  const last = $("last-msg");
  if (last) {
    last.textContent = state.midi.last
      ? state.midi.last.map((b) => b.toString(16).padStart(2, "0").toUpperCase()).join(" ")
      : "—";
  }
}

function knobValueFor(stem) {
  const pool = state.pools[stem];
  const slot = state.slot[stem];
  if (pool.length <= 1) return 0;
  return Math.round((slot * 127) / (pool.length - 1));
}

function bindUi() {
  const play = $("btn-play");
  if (play) {
    play.addEventListener("click", () => {
      if (state.playing) stop();
      else void start();
    });
  }
  const rec = $("btn-rec");
  if (rec) {
    rec.addEventListener("click", () => {
      stop();
      setSustain(false);
      handleMessage([0xb0, 64, 0]);
      handleMessage([0xb0, 123, 0]);
    });
  }
  const env = $("btn-env");
  if (env) {
    env.addEventListener("pointerdown", () => handleMessage([0xb0, 64, 127]));
    env.addEventListener("pointerup", () => handleMessage([0xb0, 64, 0]));
    env.addEventListener("pointerleave", () => handleMessage([0xb0, 64, 0]));
  }
  const lfo = $("btn-lfo");
  if (lfo) {
    lfo.addEventListener("pointerdown", () => handleMessage([0xb0, 1, 127]));
    lfo.addEventListener("pointerup", () => handleMessage([0xb0, 1, 0]));
    lfo.addEventListener("pointerleave", () => handleMessage([0xb0, 1, 0]));
  }
  const presets = $("btn-presets");
  if (presets) {
    presets.addEventListener("click", () => {
      const next = (state.program + 1) % 128;
      handleMessage([0xc0, next]);
    });
  }
  const bank = $("bank-select");
  if (bank) {
    bank.addEventListener("change", () => {
      state.bank = bank.value;
      render();
      setStatus(`knob bank ${state.bank}`);
    });
  }
  const octDown = $("btn-oct-down");
  const octUp = $("btn-oct-up");
  if (octDown) octDown.addEventListener("click", () => { state.octave = Math.max(-3, state.octave - 1); render(); });
  if (octUp) octUp.addEventListener("click", () => { state.octave = Math.min(3, state.octave + 1); render(); });

  const connect = $("btn-connect");
  if (connect) connect.addEventListener("click", () => void connectMidi());

  for (let i = 0; i < STEMS.length; i++) {
    const stem = STEMS[i];
    const knob = $(`knob-${stem}`);
    if (knob) {
      knob.addEventListener("input", () => {
        handleSlider(stem, Number(knob.value));
        if (!state.playing) void start();
      });
    }
  }

  // 27-key strip (F3..G5) — each key jumps the focused stem one phrase on.
  const keys = $("keys");
  if (keys) {
    for (let i = 0; i < KEY_COUNT; i++) {
      const key = document.createElement("button");
      key.type = "button";
      key.className = "key";
      key.textContent = String(i + 1);
      key.title = `Key ${i + 1} → phrase ${i + 1} of ${STEM_LABELS[state.focus]}`;
      key.addEventListener("pointerdown", () => {
        handleMessage([0x90, KEY_BASE + i, 100]);
      });
      key.addEventListener("pointerup", () => {
        handleMessage([0x80, KEY_BASE + i, 0]);
      });
      keys.appendChild(key);
    }
  }
}

// ---------------------------------------------------------------------------
// debug surface
// ---------------------------------------------------------------------------
window.__wubShim = {
  state: () => ({
    base: state.base,
    loading: state.loading,
    playing: state.playing,
    bpm: state.bpm,
    bank: state.bank,
    focus: state.focus,
    octave: state.octave,
    pools: Object.fromEntries(STEMS.map((stem) => [stem, state.pools[stem].length])),
    slot: { ...state.slot },
    current: Object.fromEntries(
      STEMS.map((stem) => [stem, state.current[stem] ? {
        phraseId: state.current[stem].phraseId,
        trackId: state.current[stem].trackId,
        title: state.current[stem].title,
        url: state.current[stem].url,
      } : null]),
    ),
    playingPhrase: Object.fromEntries(
      STEMS.map((stem) => [stem, state.playing_phrase[stem]?.phraseId ?? null]),
    ),
    midi: { ...state.midi },
    sustain: state.sustain,
    mod: state.mod,
    cached: bufferCache.size,
    peak: uiMeter(),
  }),
  inject: (bytes) => {
    handleMessage(bytes);
    return true;
  },
  knob: (index, value) => handleMessage([0xb0, (CC_BANKS[state.bank] ?? CC_BANKS.C)[index], value]),
  abs: (index, value) => handleSlider(STEMS[index], value),
  key: (semitone) => handleMessage([0x90, KEY_BASE + semitone, 100]),
  program: (value) => handleMessage([0xc0, value & 0x7f]),
  bank: (id) => {
    if (CC_BANKS[id]) {
      state.bank = id;
      render();
    }
  },
  start: () => start(),
  stop: () => stop(),
  _state: state,
};

bindUi();
loadManifest().catch((err) => {
  state.loading = false;
  state.error = String(err);
  setStatus(`bank load failed: ${err}`);
  console.error(err);
});
render();
