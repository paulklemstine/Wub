# Wub device shim — the FM-1 that could

A browser page that behaves like a *device runtime*: it pulls the real kit
manifest and real Opus phrases over HTTPS, decodes them with the browser's own
codec, and plays four stem streams on one 4-bar grid while every CTL-1 control
drives it.

This is the "could the FM-1 play the Wub bank" spike, tested in the emulator
first. It is **not** shipped to the web player and it does **not** run on the
FM-1: the AC791N has no Opus decoder, no Signalsmith Stretch and no room for a
448 MB bank (see `docs/FM1_CONTROLLER.md`, section 6). What it *does* prove is
the control contract and the bank format.

## Run it

```bash
# default bank: GitHub raw (main branch), no checkout needed
python3 tools/fm1_shim/serve.py --port 8123
# → http://127.0.0.1:8123/

# offline / deterministic: serve a local kit directory at /bank
python3 tools/fm1_shim/serve.py --port 8123 --bank frontend/public/data/kits
# → http://127.0.0.1:8123/?base=/bank
```

`?base=` accepts any URL prefix that serves `kit_manifest.json` and `kits/`.
The default is
`https://raw.githubusercontent.com/paulklemstine/Wub/main/frontend/public/data/kits`,
which sends `access-control-allow-origin: *` and supports range requests.

Press **PLAY** (needs a click so the audio context can start). Press
**CONNECT MIDI** to drive the page from a real CTL-1 over USB; the on-screen
panel works without any hardware.

## Control map

Every message CTL-1 documents in its manual is handled.

| Control | MIDI | What the shim does |
| :--- | :--- | :--- |
| Keys (27, F3–G5) | note on/off | Jump the **focused** stem to that phrase slot; `OCT−`/`OCT+` page by 27 |
| KNOB 1–4 | CC per bank | Scroll that stem's phrase list by the encoder delta, commit on the next bar |
| KNOB 1–4 first touch | CC per bank | Seeds the slot from the absolute 0..127 position |
| FX / SELECT | — | Bank A–D on screen (`BANK` select); the CC numbers follow the manual |
| ENV | CC 64 | Breakdown hold: drums and bass duck to 28 % |
| LFO | CC 1 | Master low-pass sweep, 200 Hz – 18 kHz |
| MASTER | CC 7 or any unmapped CC | Master volume |
| PLAY | MIDI start / stop (`0xFA` / `0xFC`) | Start / stop the grid |
| REC | CC 123 (+ note offs, CC 64 = 0) | Panic: stop and release |
| PRESETS | program change (`0xC0`) | Jump all four stems to kit *n* |
| Clock | MIDI clock (`0xF8`, 24 ppqn) | Follow the tempo, 40–240 BPM |
| Touch | channel / poly aftertouch | Lift the focused stem a little |

Knob banks are the firmware's: A `74/71/1/7`, B `73/72/75/76`, C `16/17/18/19`
(default), D `20/21/22/23`. A bank mismatch between the device and the page's
`BANK` select looks exactly like dead knobs — that is the same trap as in the
web player.

## Playback model

- Four stem streams, one shared bar clock. A change is committed on the **next
  bar** (`4 * 60 / bpm`), never mid-phrase.
- Each phrase loops on the grid; tempo is fitted with `playbackRate` only.
  There is **no time stretch and no key shift** here — that is the honest limit
  of the spike, and the reason it is not the shipped engine.
- Decoded buffers live in a 48-entry LRU; the current phrase and its successor
  are prefetched on every knob turn.
- A failed decode logs and leaves the current phrase playing; it never stops
  the transport.

## Test it

```bash
python3 tools/fm1_shim_test.py --screenshot /tmp/opencode/fm1-shim.png
```

The test serves the shim with the local kit bank, runs it in headless Chrome,
and checks 29 invariants: pools built from the manifest, four streams audible,
slider seeding, encoder scroll, key jumps, bank routing, ENV/LFO, program
change kit jumps, MIDI start/stop, 24-ppqn tempo following, panic, and a real
Web MIDI round-trip through the emulated CTL-1 port
(`tools/fm1_emu_shim.js`).

Debug surface: `window.__wubShim` — `state()`, `inject(bytes)`, `knob(i, v)`,
`abs(i, v)`, `key(n)`, `program(p)`, `bank(id)`, `start()`, `stop()`.

## Files

| File | Role |
| :--- | :--- |
| `index.html` | Device panel (stem screen, knobs, transport, 27-key strip). |
| `device.js` | Manifest/bank loading, Opus decode cache, bar clock, MIDI router, UI. |
| `serve.py` | Static server plus optional `/bank` mount for a local kit dir. |
