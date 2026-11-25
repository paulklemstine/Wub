// FM-1 / CTL-1 emulator shim, injected before the app boots.
//
// There is no real MIDI loopback in headless Chrome on a machine where
// /dev/snd/seq is not readable, so this shim stands in for the *operating
// system* MIDI layer: it replaces navigator.requestMIDIAccess with one virtual
// "CTL-1 FM-1" input port and draws a small device panel whose four endless
// encoders emit real MIDIMessageEvents (status 0xB0) exactly like the CTL-1
// firmware: absolute 0..127 values, 1 click per detent, up to 6 clicks when
// spun fast, wrapping 127 -> 0.
//
// Everything above the OS boundary is the real app code: Web MIDI port
// enumeration, hot-plug statechange, CC parsing, bank matching, slice
// scrolling, debounce, reticle steering and the pinned-phrase commit.
//
// Test hooks (window.__fm1Emu):
//   turn(knob, clicks)  spin knob 0..3 by +/- clicks (same path as the wheel)
//   abs(knob, value)    slam the encoder to an absolute 0..127 value
//   setBank('A'..'D')   change the emulated device bank
//   plug() / unplug()   hot-plug the device
//   log                 every message the emulated device sent
(() => {
  const CC_BANKS = {
    A: [74, 71, 1, 7],
    B: [73, 72, 75, 76],
    C: [16, 17, 18, 19],
    D: [20, 21, 22, 23],
  };
  const STEMS = ["DRUMS", "BASS", "VOCALS", "OTHER"];

  const state = {
    bank: "C",
    values: [0, 0, 0, 0],
    connected: true,
  };
  const log = [];
  const ports = new Map();
  let port = null;

  const access = {
    inputs: ports,
    outputs: new Map(),
    sysexEnabled: false,
    onstatechange: null,
    addEventListener(type, fn) {
      if (type === "statechange") this.onstatechange = fn;
    },
    removeEventListener() {},
  };

  function makePort() {
    const p = {
      id: "ctl1-emu-1",
      name: "CTL-1 FM-1",
      manufacturer: "M-VAVE",
      type: "input",
      state: "connected",
      connection: "open",
      onmidimessage: null,
      onstatechange: null,
      open: async () => p,
      close: async () => p,
      addEventListener() {},
      removeEventListener() {},
    };
    return p;
  }

  function plug() {
    if (port !== null) return;
    port = makePort();
    ports.set(port.id, port);
    if (typeof access.onstatechange === "function") {
      access.onstatechange({ port });
    }
  }

  function unplug() {
    if (port === null) return;
    const gone = port;
    gone.state = "disconnected";
    gone.onmidimessage = null;
    ports.delete(gone.id);
    port = null;
    if (typeof access.onstatechange === "function") {
      access.onstatechange({ port: gone });
    }
  }

  function emit(cc, value) {
    const data = new Uint8Array([0xb0, cc & 0x7f, value & 0x7f]);
    let event;
    try {
      event = new MIDIMessageEvent("midimessage", { data });
    } catch (err) {
      event = new Event("midimessage");
      event.data = data;
    }
    log.push({ cc: data[1], value: data[2], at: performance.now() });
    if (log.length > 512) log.shift();
    if (port !== null && typeof port.onmidimessage === "function") {
      port.onmidimessage(event);
    }
  }

  function turn(knob, clicks) {
    if (knob < 0 || knob > 3) return null;
    const raw = Math.max(-32, Math.min(32, Math.round(clicks)));
    if (raw === 0) return state.values[knob];
    const cc = CC_BANKS[state.bank][knob];
    state.values[knob] = ((state.values[knob] + raw) % 128 + 128) % 128;
    emit(cc, state.values[knob]);
    paint();
    return state.values[knob];
  }

  function abs(knob, value) {
    if (knob < 0 || knob > 3) return null;
    const v = Math.max(0, Math.min(127, Math.round(value)));
    state.values[knob] = v;
    emit(CC_BANKS[state.bank][knob], v);
    paint();
    return v;
  }

  function setBank(bank) {
    if (!(bank in CC_BANKS)) return null;
    state.bank = bank;
    paint();
    return bank;
  }

  // ---- device panel ------------------------------------------------------
  const CSS = `
    #fm1-emu { position: fixed; top: 10px; right: 10px; z-index: 2147483000;
      width: 250px; padding: 10px 12px 12px; border-radius: 12px;
      background: rgba(9, 12, 20, 0.94); border: 1px solid #2b3648;
      box-shadow: 0 10px 30px rgba(0,0,0,0.55); color: #cbd5e1;
      font: 11px/1.35 ui-sans-serif, system-ui, dejavu, sans-serif; }
    #fm1-emu.is-off { opacity: 0.45; filter: grayscale(0.8); }
    #fm1-emu .emu-head { display: flex; align-items: center; gap: 6px;
      justify-content: space-between; margin-bottom: 8px; }
    #fm1-emu .emu-title { font-weight: 700; letter-spacing: 0.08em; color: #93c5fd; }
    #fm1-emu select, #fm1-emu button { background: #131a26; color: #e2e8f0;
      border: 1px solid #334155; border-radius: 6px; font: inherit; padding: 2px 4px; }
    #fm1-emu .emu-knobs { display: flex; gap: 6px; }
    #fm1-emu .emu-knob { flex: 1; text-align: center; user-select: none; }
    #fm1-emu .emu-dial { position: relative; width: 46px; height: 46px; margin: 0 auto 4px;
      border-radius: 50%; background: radial-gradient(circle at 32% 28%, #3b4a60, #10161f 70%);
      border: 2px solid #3f4d63; cursor: ns-resize; }
    #fm1-emu .emu-ptr { position: absolute; left: 50%; top: 50%; width: 2px; height: 17px;
      margin-left: -1px; background: #7dd3fc; transform-origin: 50% 100%;
      transform: translateY(-100%) rotate(0deg); border-radius: 2px; }
    #fm1-emu .emu-stem { color: #94a3b8; letter-spacing: 0.05em; }
    #fm1-emu .emu-val { color: #e2e8f0; font-variant-numeric: tabular-nums; }
    #fm1-emu .emu-foot { margin-top: 8px; color: #64748b; }
    #fm1-emu .emu-foot b { color: #7dd3fc; font-weight: 600; }
  `;

  function paint() {
    const root = document.getElementById("fm1-emu");
    if (!root) return;
    root.classList.toggle("is-off", !state.connected);
    const bankSel = root.querySelector("#emu-bank");
    if (bankSel) bankSel.value = state.bank;
    const portState = root.querySelector("#emu-port");
    if (portState) portState.textContent = state.connected ? "PLUGGED" : "UNPLUGGED";
    const bankBtn = root.querySelector("#emu-plug");
    if (bankBtn) bankBtn.textContent = state.connected ? "UNPLUG" : "PLUG IN";
    const ccs = CC_BANKS[state.bank];
    for (let i = 0; i < 4; i++) {
      const dial = root.querySelector(`#emu-dial-${i}`);
      const ptr = root.querySelector(`#emu-ptr-${i}`);
      const val = root.querySelector(`#emu-val-${i}`);
      const cc = root.querySelector(`#emu-cc-${i}`);
      const angle = (state.values[i] / 127) * 270 - 135;
      if (ptr) ptr.style.transform = `translateY(-100%) rotate(${angle}deg)`;
      if (val) val.textContent = String(state.values[i]).padStart(3, " ");
      if (cc) cc.textContent = `CC ${ccs[i]}`;
      if (dial) dial.dataset.value = String(state.values[i]);
    }
  }

  function buildPanel() {
    if (document.getElementById("fm1-emu")) return;
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const root = document.createElement("div");
    root.id = "fm1-emu";
    root.innerHTML = `
      <div class="emu-head">
        <span class="emu-title">FM-1 / CTL-1</span>
        <select id="emu-bank" aria-label="Emulated device bank">
          <option>A</option><option>B</option><option>C</option><option>D</option>
        </select>
        <button type="button" id="emu-plug">UNPLUG</button>
      </div>
      <div class="emu-knobs">
        ${STEMS.map(
          (stem, i) => `
          <div class="emu-knob">
            <div class="emu-dial" id="emu-dial-${i}" data-knob="${i}"><div class="emu-ptr" id="emu-ptr-${i}"></div></div>
            <div class="emu-stem">K${i + 1} ${stem.slice(0, 4)}</div>
            <div class="emu-val" id="emu-val-${i}">  0</div>
            <div id="emu-cc-${i}">CC 0</div>
          </div>`,
        ).join("")}
      </div>
      <div class="emu-foot">wheel a knob to turn it · <b id="emu-port">PLUGGED</b></div>
    `;
    document.body.appendChild(root);

    root.querySelector("#emu-bank").addEventListener("change", (event) => {
      setBank(event.target.value);
    });
    root.querySelector("#emu-plug").addEventListener("click", () => {
      if (state.connected) {
        state.connected = false;
        unplug();
      } else {
        state.connected = true;
        plug();
      }
      paint();
    });

    for (let i = 0; i < 4; i++) {
      const dial = root.querySelector(`#emu-dial-${i}`);
      dial.addEventListener(
        "wheel",
        (event) => {
          if (!state.connected) return;
          event.preventDefault();
          const fast = event.shiftKey ? 6 : 1;
          turn(i, event.deltaY < 0 ? fast : -fast);
        },
        { passive: false },
      );
    }
    paint();
  }

  // ---- OS boundary -------------------------------------------------------
  Object.defineProperty(navigator, "requestMIDIAccess", {
    configurable: true,
    writable: true,
    value: async () => access,
  });

  window.__fm1Emu = {
    turn,
    abs,
    setBank,
    plug() {
      state.connected = true;
      plug();
      paint();
    },
    unplug() {
      state.connected = false;
      unplug();
      paint();
    },
    get log() {
      return log.slice();
    },
    get state() {
      return { bank: state.bank, values: state.values.slice(), connected: state.connected };
    },
  };

  // The app boots on DOMContentLoaded, so the panel can be built immediately
  // when the document exists and otherwise waits for it.
  if (document.body) buildPanel();
  else document.addEventListener("DOMContentLoaded", buildPanel);

  plug();
})();
