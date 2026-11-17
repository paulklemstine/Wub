# FM-1 / CTL-1 Knob → Phrase Controller

This document covers the hardware knob control added to the browser player at
<https://paulklemstine.github.io/Wub/>. Four knobs on the M-VAVE FM-1 select
and scroll the instrument phrases for the four stems, exactly like dragging a
stem's reticle across the DJ control surface nebula.

---

## 1. Firmware requirement

**The stock M-VAVE FM-1 knobs do not transmit MIDI CC.** The factory firmware
is MIDI-IN only: the four upper-right knobs edit the synth patch. Nothing
reaches the browser, and no browser-side code can change that.

Knob control therefore requires the community **CTL-1** firmware:

- Source and manual: <https://github.com/charlesvestal/fm1-ctl>
- Web installer / docs: <https://charlesvestal.github.io/fm1-ctl/>
- Emulator: <https://charlesvestal.github.io/fm1-ctl/emu/>

CTL-1 makes the FM-1 appear as a MIDI controller named `CTL-1 FM-1`. KNOB 1–4
send absolute control changes in four banks. The Wub player identifies the
device automatically, but you can always pick it from the **INPUT** list.

> If the app says "stock FM-1 knobs send nothing", that is expected on factory
> firmware. Flash CTL-1 first.

---

## 2. Knob → stem mapping

Knob order follows the DJ control surface (left to right):

| Knob | Stem   | Colour   |
| :--- | :----- | :------- |
| K1   | DRUMS  | `#ff5470` |
| K2   | BASS   | `#ffb020` |
| K3   | VOCALS | `#c084fc` |
| K4   | OTHER  | `#38bdf8` |

### CC banks

| Bank | K1 | K2 | K3 | K4 |
| :--- | :- | :- | :- | :- |
| A | 74 | 71 | 1  | 7  |
| B | 73 | 72 | 75 | 76 |
| C | 16 | 17 | 18 | 19 |
| D | 20 | 21 | 22 | 23 |

**Bank C is the default.** Bank A reuses CC 1 (mod wheel) and CC 7 (channel
volume), which instruments and DAWs may also react to; C and D are clean
controller-only banks. Change banks on the device and with the **BANK** select
in the `FM-1 KNOB → PHRASE` strip so the two stay in sync.

---

## 3. What a knob turn does

1. The stem's live phrase pool — every phrase a pin can actually select at the
   current tempo, key and style — is ordered deterministically (track id, then
   phrase id, numeric-aware) and duplicate coordinates are collapsed. Before
   the kit bank loads it falls back to the full catalogue.
2. The CTL-1 knob is an endless encoder: it sends absolute 0–127 values that
   wrap at the ends and accelerate when spun fast. The first turn after the
   player loads seeds the slot from that value (`round(value * (count - 1) /
   127)`); after that every click scrolls one slice (up to six fast), wrapping
   from the last slice back to the first, so every slice is reachable.
   The virtual sliders are absolute: they jump to the slot for their value
   (`round(value * (count - 1) / 127)`) and re-anchor the scroll.
3. The stem's reticle glides to the slice's nebula coordinate immediately.
4. After ~180 ms of knob settling, the slice is committed through the same
   path as a reticle drop: the stem is pinned to the nearest legal phrase in
   that style region and the change lands at the next bar.
5. The pin holds for ~32 bars; double-click the reticle (or turn the knob to a
   new slice) to move on.

Turning a knob while the transport is stopped moves the reticle and stores the
steering position; playback starts from there.

The catalogue is cached for 15 seconds and rebuilt when the cluster rotates or
the playback mode changes, so the slice order is stable while you scroll but
follows the live bank.

---

## 4. Using the app

1. Flash CTL-1 and connect the FM-1 over USB (or Bluetooth MIDI).
2. Open <https://paulklemstine.github.io/Wub/> in Chrome or Edge (Web MIDI
   requires a secure context; `localhost` is also accepted).
3. In the DJ control surface, press **CONNECT MIDI** and allow the permission
   prompt.
4. Pick the input (`CTL-1 FM-1`) and the bank you selected on the device.
5. Turn K1–K4. Each click scrolls one slice (fast turns skip up to six); the
   strip shows `value · slice/total` for each stem.

The four **virtual sliders** in the strip are the no-hardware test harness:
dragging one performs an absolute 0–127 jump across that stem's slice list and
re-anchors the knob's scroll position.

---

## 5. Emulator in the testing cycle

There are two different emulators, and only one can drive the browser.

### Official FM-1 emulator — receive only

<https://joakimeriksson.github.io/fm1-emu/?fw=felucca&mhz=100> emulates the
FM-1 CPU, screen and audio in WebAssembly. It **accepts** MIDI input (so you
can play the emulated synth), but it does **not** send MIDI out. It cannot
drive a DAW or this browser player, so it is useful for checking firmware
behaviour and sounds, not for testing knob → phrase mapping.

### CTL-1 emulator — can send CC

<https://charlesvestal.github.io/fm1-ctl/emu/> emulates the controller
firmware and can route its knobs to a real MIDI output port. To test the whole
chain on one machine:

1. Create a virtual MIDI loopback:
   - Linux: `sudo modprobe snd-virmidi` (or use the `Midi Through` port), then
     connect the emulator output to the player input with `aconnect`.
   - macOS: enable **IAC Driver** in Audio MIDI Setup.
   - Windows: install **loopMIDI**.
2. In the CTL-1 emulator, choose the loopback as its MIDI output.
3. In the Wub player, press **CONNECT MIDI** and select the loopback input.
4. Turn the emulated knobs and watch the reticles move.

### Automated / headless testing

Without a loopback, use the debug hook exposed by the app:

```js
__wubFM1.state()                 // supported, connected, bank, devices, last CC
__wubFM1.slices("bass").length   // ordered slice count for a stem
__wubFM1.knob(1, 127)            // virtual K2 turn → value 127 (bass)
__wubFM1.injectCC(17, 64)        // raw CC in the current bank
__wubFM1.select("bass", 42)      // jump to slice 42 of the ordered list
__wubFM1.bank("D")               // switch receiver bank
__wubFM1.reticles()              // current reticle coordinates
__wubFM1.remix()                 // per-stem phrase, pin and buffering state
```

`tests/test_fm1_midi.py` runs the mapping logic under Node's type stripping and
checks the HTML/wiring contract. The in-page virtual sliders cover the UI path.

Playwright/Chrome DevTools Protocol note: Web MIDI permission must be granted
explicitly, e.g. `Browser.grantPermissions` with `permissions: ["midi"]`; a
headless profile without it returns `NotAllowedError`.

### Local device shim — full path without an OS loopback

Some machines have no ALSA sequencer access (`/dev/snd/seq` is root-only) and
cannot create a virtual MIDI cable. For those, `tools/fm1_emu_shim.js` stubs
**only** `navigator.requestMIDIAccess` and exposes one virtual `CTL-1 FM-1`
input plus a small on-page device panel whose four endless encoders emit real
`MIDIMessageEvent`s (status `0xB0`, absolute 0..127, 1 detent per wheel click,
6 when spun fast, wrapping 127 → 0).

Everything above the OS boundary is production code — port enumeration,
`statechange` hot-plug, CC parsing, bank matching, scroll accumulation,
debounce, reticle steering and the pinned-phrase commit.

```bash
cd frontend && npm run build && cd ..
python3 tools/fm1_emu_test.py --screenshot /tmp/opencode/fm1-emu.png
```

The harness serves `frontend/dist` on a free port, launches headless Chrome
with the shim injected before any app script, presses PLAY, connects the MIDI
port, then drives the emulated encoders (including a real `mouseWheel` on the
panel) and asserts 50+ invariants: CC bank routing, one-detent and accelerated
scroll, wrap-around, reticle/pin placement per stem, device-bank vs app-bank
mismatch, unplug/replug recovery, and a live master-bus audio probe. Exit code
is non-zero on any failure, so it can gate a release.

The shim's own interface is available as `window.__fm1Emu` (`turn`, `abs`,
`setBank`, `plug`, `unplug`, `state`, `log`).

> This is a *test* shim. On a real machine with a loopback, the CTL-1 emulator
> and the physical FM-1 use the same code path with no shim at all.

---

## 6. USB audio and running Wub on the device

Three different "USB" ideas get mixed together; only the first two are real
today.

| Idea | Status | Notes |
| :--- | :--- | :--- |
| FM-1 knobs → Wub over USB MIDI | ✅ shipped | Requires CTL-1 firmware; see sections 1–3. |
| Wub mix → FM-1 audio out (USB DAC) | ✅ with **fm1-base / Felucca** firmware | Community firmware adds the USB Audio **input** path, so the browser mix plays through the FM-1's DAC/headphone jack. Stock firmware's USB audio is not a general-purpose DAC. |
| Wub engine running **on** the FM-1 | ❌ not viable | See below. |

The FM-1 is a JieLi AC791N: 360 MHz dual-core pi32v2, **578 KB SRAM**, **1 MiB
flash with a ~581 KB app slot**, 5.8 ms audio DMA deadline, no PSRAM, no SD
slot. The shipped kit bank is ~448 MB of Opus plus a Signalsmith Stretch
worklet; neither the decoder nor the stretcher fits, and WiFi and BLE MIDI are
mutually exclusive on fm1-base. Porting the playback engine would be a new
firmware project, and the browser would still do it better.

If wireless device playback ever matters, the credible spike is a packed-bank
HTTP **Range** player that streams *pre-rendered* PCM runs — no Opus decode, no
stretch, no key shift. That is a "Wub phrase player", not the Wub engine.

### The device shim (emulator-first bench)

`tools/fm1_shim/` is that idea, running in the browser where the codec and the
memory already exist. It fetches the shipped `kit_manifest.json` and real Opus
phrases from GitHub raw (or a local bank), plays four stem streams on one 4-bar
grid with next-bar commits, and handles every CTL-1 message: keys, the four
knob banks, ENV, LFO, MASTER, PLAY/REC, PRESETS, MIDI clock and aftertouch.
The on-screen panel is the emulator; **CONNECT MIDI** drives the same handler
from real hardware.

```bash
python3 tools/fm1_shim/serve.py --port 8123 --bank frontend/public/data/kits
# open http://127.0.0.1:8123/?base=/bank
python3 tools/fm1_shim_test.py        # 29 headless checks
```

It is a bench for the firmware contract and the bank format, not firmware: it
resamples tempo with `playbackRate` (no stretch, no key shift) and relies on
the browser's Opus decoder. See `tools/fm1_shim/README.md` for the full control
map.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `CONNECT MIDI` says Web MIDI unavailable | Use Chrome/Edge on HTTPS or localhost. Safari/Firefox do not expose Web MIDI by default. |
| Permission denied (`NotAllowedError`) | Re-allow MIDI for the site, or grant the `midi` permission in an automated/headless profile. |
| No inputs listed | The FM-1 is not in controller mode. Flash CTL-1; reconnect USB; the port should read `CTL-1 FM-1`. |
| Knobs do nothing | Wrong bank: the device's bank and the strip's **BANK** select must match. |
| Knob moves the wrong stem | Bank A maps CC 1/7 to the last two knobs; switch to bank C or D. |
| Reticle moves but audio holds | The phrase commits at the next bar, and the stem holds its pin for ~32 bars. Double-click the reticle to release. |
| Slice list is empty | The kit bank has not loaded yet. Press PLAY once, then turn a knob. |

---

## 8. Implementation map

| File | Role |
| :--- | :--- |
| `frontend/src/fm1_midi.ts` | Pure mapping (`FM1_KNOB_STEMS`, `FM1_CC_BANKS`, `knobForCc`, `knobValueToSlot`, `scrollSlot`, `buildSliceOrder`) and the Web MIDI receiver class. |
| `frontend/src/main.ts` | `initFm1`, device/bank selects, virtual sliders, debounce + reticle commit, `window.__wubFM1`. |
| `frontend/index.html` | `#fm1-strip` under the nebula canvas and its CSS. |
| `tests/test_fm1_midi.py` | Node mapping assertions plus static HTML/wiring contract. |
| `tools/fm1_emu_shim.js` | Headless emulator: virtual CTL-1 MIDI port + encoder panel injected before app scripts. |
| `tools/fm1_emu_test.py` | Chrome DevTools Protocol harness that drives the shipped build and asserts the knob → phrase path end to end. |
| `tools/fm1_shim/` | Device-side shim: plays the shipped bank from HTTPS and maps every CTL-1 message (emulator-first firmware bench). |
| `tools/fm1_shim_test.py` | Headless checks for the device shim (bank loading, four streams, full control map, real MIDI round-trip). |
