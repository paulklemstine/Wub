#!/usr/bin/env python3
"""Drive the Wub device shim in headless Chrome.

Serves `tools/fm1_shim/` plus a local kit bank, opens the page, and exercises
every CTL-1 message the shim understands: knob banks, keys, ENV, LFO,
PRESETS, PLAY/REC, MIDI clock and aftertouch — plus a real Web MIDI path
through the emulated CTL-1 port from `tools/fm1_emu_shim.js`.

    python3 tools/fm1_shim_test.py --screenshot /tmp/opencode/fm1-shim.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm1_emu_test import CDP, Checks, page_ws, start_chrome  # noqa: E402

SHIM_DIR = ROOT / "tools" / "fm1_shim"
EMU_SHIM = ROOT / "tools" / "fm1_emu_shim.js"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, bank: Path | None) -> subprocess.Popen:
    cmd = [sys.executable, str(SHIM_DIR / "serve.py"), "--port", str(port)]
    if bank is not None:
        cmd += ["--bank", str(bank)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=0.5) as r:
                if r.status == 200:
                    return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("shim server did not start")


async def run(args) -> int:
    import websockets

    checks = Checks()
    port = free_port()
    bank = None if args.remote else Path(args.bank).resolve()
    server = start_server(port, bank)
    chrome, debug_port = start_chrome(args.chrome, Path(args.profile))
    ws = None
    try:
        ws = await websockets.connect(page_ws(debug_port), max_size=64 * 1024 * 1024)
        cdp = CDP(ws)
        await cdp.start()
        for method in ("Page.enable", "Runtime.enable", "Log.enable"):
            await cdp.send(method)
        await cdp.send(
            "Emulation.setDeviceMetricsOverride", width=1400, height=1000, deviceScaleFactor=1, mobile=False
        )
        await cdp.send("Page.addScriptToEvaluateOnNewDocument", source=EMU_SHIM.read_text())
        url = f"http://127.0.0.1:{port}/" + ("" if bank is None else "?base=/bank")
        print(f"→ {url}")
        await cdp.send("Page.navigate", url=url)
        await cdp.wait_for("!!window.__wubShim", timeout=30, label="shim boot")
        await cdp.wait_for("window.__wubShim.state().loading === false", timeout=120, label="manifest")

        print("\n[1] kit bank over HTTP")
        state = await cdp.js("window.__wubShim.state()")
        counts = state["pools"]
        checks.check(all(n > 500 for n in counts.values()), "four stem pools built from the manifest", str(counts))
        checks.check(all(state["current"][s] is not None for s in counts), "all four stems start on one kit")
        checks.check(
            len({state["current"][s]["trackId"] for s in counts}) == 1,
            "opening kit is shared by the four stems",
        )
        base = await cdp.js("document.getElementById('bank-info').textContent")
        checks.check("kits" in base, "header shows the loaded bank", base)

        print("\n[2] transport + audio")
        await cdp.click("#btn-play")
        await cdp.wait_for("window.__wubShim.state().playing", timeout=30, label="playing")
        probe = None
        for _ in range(80):
            probe = await cdp.js("window.__wubShim.state()")
            if probe and probe["peak"] > 0.001:
                break
            await asyncio.sleep(0.25)
        checks.check(probe["peak"] > 0.001, "master bus is making sound", f"peak={probe['peak']:.4f}")
        checks.check(
            all(probe["playingPhrase"][s] for s in counts),
            "all four stem streams are live",
            str(probe["playingPhrase"]),
        )
        checks.check(probe["cached"] >= 4, "decoded phrases cached", str(probe["cached"]))

        print("\n[3] absolute slider (K2 → last bass phrase)")
        await cdp.js("window.__wubShim.abs(1, 127)")
        await asyncio.sleep(0.2)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["slot"]["bass"], counts["bass"] - 1, "slider jumps to the last slot")
        await asyncio.sleep(2.6)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(
            state["playingPhrase"]["bass"],
            state["current"]["bass"]["phraseId"],
            "commit lands on the next bar",
        )

        print("\n[4] endless encoder + keys")
        await cdp.js("window.__wubShim.abs(0, 0)")
        await cdp.js("window.__wubShim.knob(0, 40)")
        await asyncio.sleep(0.1)
        mid = (await cdp.js("window.__wubShim.state()"))["slot"]["drums"]
        checks.eq(mid, 40, "first touch after a slider seeds from the absolute value")
        await cdp.js("window.__wubShim.knob(0, 43)")
        await asyncio.sleep(0.1)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["slot"]["drums"], (mid + 3) % counts["drums"], "three detents scroll three phrases")
        checks.eq(state["focus"], "drums", "the turned knob takes key focus")
        await cdp.js("window.__wubShim.key(6)")
        await asyncio.sleep(0.2)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["slot"]["drums"], 6, "key 7 jumps the focused stem to phrase 7")
        await cdp.js("window.__wubShim.bank('D')")
        await cdp.js("window.__wubShim.knob(1, 127)")
        await asyncio.sleep(0.1)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["slot"]["bass"], counts["bass"] - 1, "bank D seeds from the encoder position")
        await cdp.js("window.__wubShim.bank('C')")

        print("\n[5] knob banks")
        await cdp.js("window.__wubShim.bank('B')")
        await cdp.js("window.__wubShim.knob(1, 20)")
        await cdp.js("window.__wubShim.knob(1, 24)")
        await asyncio.sleep(0.2)
        last = await cdp.js("document.getElementById('last-msg').textContent")
        checks.eq(last, "B0 48 18", "bank B K2 sends CC 72 (0x48)")
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["bank"], "B", "shim bank follows the device")
        await cdp.js("window.__wubShim.bank('C')")

        print("\n[6] ENV / LFO / aftertouch")
        await cdp.js("window.__wubShim.inject([0xB0,64,127])")
        await asyncio.sleep(0.15)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["sustain"], True, "ENV (CC 64) holds the breakdown")
        await cdp.js("window.__wubShim.inject([0xB0,64,0])")
        await asyncio.sleep(0.15)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["sustain"], False, "releasing ENV returns the mix")
        await cdp.js("window.__wubShim.inject([0xB0,1,127])")
        await asyncio.sleep(0.15)
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["mod"], 127, "LFO (CC 1) opens the filter")
        await cdp.js("window.__wubShim.inject([0xB0,1,0])")

        print("\n[7] PRESETS → kit jump")
        await cdp.js("window.__wubShim.program(7)")
        await asyncio.sleep(0.4)
        state = await cdp.js("window.__wubShim.state()")
        tracks = {state["current"][s]["trackId"] for s in counts}
        checks.check(len(tracks) == 1, "program change moves all four stems to one kit", str(tracks))
        checks.eq(
            await cdp.js("document.getElementById('last-msg').textContent"),
            "C0 07",
            "program change is the last message",
        )

        print("\n[8] transport, clock, panic")
        await cdp.js("window.__wubShim.inject([0xFC])")
        await asyncio.sleep(0.3)
        checks.eq((await cdp.js("window.__wubShim.state()"))["playing"], False, "MIDI stop stops the grid")
        await cdp.js("window.__wubShim.inject([0xFA])")
        await cdp.wait_for("window.__wubShim.state().playing", timeout=20, label="midi start")
        checks.check(True, "MIDI start starts the grid")
        await cdp.ev(
            "new Promise((res) => { let n = 0; const id = setInterval(() => {"
            " window.__wubShim.inject([0xF8]); if (++n >= 48) { clearInterval(id); res(1); }"
            " }, 20.833); })",
            await_promise=True,
        )
        bpm = (await cdp.js("window.__wubShim.state()"))["bpm"]
        checks.check(112 <= bpm <= 128, "24-ppqn clock sets the tempo", str(bpm))
        await cdp.js("window.__wubShim.inject([0xB0,123,0])")
        await asyncio.sleep(0.3)
        checks.eq((await cdp.js("window.__wubShim.state()"))["playing"], False, "REC (CC 123) is panic")

        print("\n[9] real Web MIDI through the emulated CTL-1")
        await cdp.click("#btn-connect")
        await cdp.wait_for("window.__wubShim.state().midi.connected", timeout=15, label="midi")
        state = await cdp.js("window.__wubShim.state()")
        checks.eq(state["midi"]["deviceName"], "CTL-1 FM-1", "CTL-1 port attached")
        await cdp.click("#btn-play")
        await cdp.wait_for("window.__wubShim.state().playing", timeout=20, label="playing again")
        await cdp.js("window.__wubShim.abs(2, 0)")
        await cdp.js("window.__fm1Emu.turn(3, 2)")
        await asyncio.sleep(0.3)
        state = await cdp.js("window.__wubShim.state()")
        last = await cdp.js("document.getElementById('last-msg').textContent")
        checks.eq(last, "B0 13 02", "hardware-style CC 19 reaches the shim")
        checks.eq(state["focus"], "other", "K4 takes focus")

        errors = [
            e
            for e in cdp.events
            if e.get("method") == "Runtime.exceptionThrown"
            or (
                e.get("method") == "Log.entryAdded"
                and e.get("params", {}).get("entry", {}).get("level") == "error"
            )
        ]
        checks.check(not errors, "no console errors", str(errors[:2]))

        if args.screenshot:
            shot = await cdp.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(shot["data"]))
            print(f"\n📸 {out}")

        print()
        total = len(checks.results)
        if checks.failed:
            print(f"\033[31m{checks.failed}/{total} checks failed\033[0m")
            return 1
        print(f"\033[32mall {total} checks passed\033[0m")
        return 0
    finally:
        if ws is not None:
            await ws.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
        server.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=str(ROOT / "frontend/public/data/kits"))
    parser.add_argument(
        "--remote",
        action="store_true",
        help="use the default GitHub raw bank instead of a local one (needs network)",
    )
    parser.add_argument("--chrome", default="google-chrome")
    parser.add_argument("--profile", default="/tmp/opencode/chrome-fm1-shim")
    parser.add_argument("--screenshot", default=None)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
