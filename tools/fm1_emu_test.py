#!/usr/bin/env python3
"""Headless-Chrome harness for the FM-1 / CTL-1 knob control path.

This runs the *shipped* frontend (built `frontend/dist`) and swaps only the
operating-system MIDI layer for `tools/fm1_emu_shim.js`: one virtual
"CTL-1 FM-1" input port plus a small device panel whose four endless encoders
emit real `MIDIMessageEvent`s. Everything above that boundary is production
code — port enumeration, hot-plug, CC parsing, bank matching, scroll
accumulation, reticle steering, debounce and the pinned-phrase commit.

Usage:
    cd frontend && npm run build && cd ..
    python3 tools/fm1_emu_test.py --screenshot /tmp/opencode/fm1-emu.png

Requires Chrome (default `google-chrome`) and the `websockets` package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM = Path(__file__).resolve().parent / "fm1_emu_shim.js"
DEFAULT_PROFILE = Path("/tmp/opencode/chrome-fm1-emu")


# --------------------------------------------------------------------------
# minimal Chrome DevTools Protocol client
# --------------------------------------------------------------------------
class CDP:
    def __init__(self, ws) -> None:
        self.ws = ws
        self._next = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.events: list[dict] = []
        self.reader_task: asyncio.Task | None = None

    async def start(self) -> None:
        self.reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            mid = msg.get("id")
            if mid is not None:
                fut = self._pending.pop(mid, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
            else:
                self.events.append(msg)

    async def send(self, method: str, **params):
        self._next += 1
        mid = self._next
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        msg = await asyncio.wait_for(fut, 60)
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    async def ev(self, expression: str, await_promise: bool = False):
        result = await self.send(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=await_promise,
            userGesture=True,
        )
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            raise RuntimeError(f"JS error: {detail.get('text')} {detail.get('exception', {})}")
        return result.get("result", {}).get("value")

    async def js(self, expression: str):
        """Evaluate a JSON-producing expression and parse it."""
        value = await self.ev(f"JSON.stringify((() => ({expression}))())")
        return json.loads(value) if value not in (None, "") else None

    async def wait_for(self, expression: str, timeout: float = 60.0, label: str = ""):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                last = await self.ev(f"(() => ({expression}))()")
            except RuntimeError as err:  # page mid-navigation
                last = str(err)
            if last:
                return last
            await asyncio.sleep(0.15)
        raise TimeoutError(f"timed out waiting for {label or expression} (last={last!r})")

    async def rect(self, selector: str):
        return await self.js(
            "(() => { const el = document.querySelector(%s);"
            " if (!el) return null; el.scrollIntoView({block:'center'});"
            " const r = el.getBoundingClientRect();"
            " return {x: r.x + r.width/2, y: r.y + r.height/2, w: r.width, h: r.height}; })()"
            % json.dumps(selector)
        )

    async def click(self, selector: str) -> None:
        r = await self.rect(selector)
        if r is None:
            raise RuntimeError(f"no element {selector}")
        for kind in ("mousePressed", "mouseReleased"):
            await self.send(
                "Input.dispatchMouseEvent",
                type=kind,
                x=r["x"],
                y=r["y"],
                button="left",
                buttons=1,
                clickCount=1,
            )
        await asyncio.sleep(0.05)

    async def wheel(self, selector: str, delta: int) -> None:
        r = await self.rect(selector)
        if r is None:
            raise RuntimeError(f"no element {selector}")
        await self.send(
            "Input.dispatchMouseEvent",
            type="mouseWheel",
            x=r["x"],
            y=r["y"],
            deltaX=0,
            deltaY=delta,
            pointerType="mouse",
        )
        await asyncio.sleep(0.05)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.results.append((bool(ok), label, detail))
        mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"  {mark} {label}" + (f"  [{detail}]" if detail else ""))
        return bool(ok)

    def eq(self, actual, expected, label: str) -> bool:
        return self.check(actual == expected, label, f"{actual!r} == {expected!r}")

    @property
    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.results if not ok)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_preview(dist: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(dist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=0.5) as resp:
                if resp.status == 200:
                    return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("static server did not start")


def start_chrome(binary: str, profile: Path) -> tuple[subprocess.Popen, int]:
    if profile.exists():
        shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True, exist_ok=True)
    args = [
        binary,
        "--headless=new",
        "--remote-debugging-port=0",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--autoplay-policy=no-user-gesture-required",
        "--mute-audio",
        "--window-size=1500,1000",
        "--hide-scrollbars",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--disable-features=Translate,MediaRouter",
        "about:blank",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file = profile / "DevToolsActivePort"
    for _ in range(150):
        if port_file.exists():
            port = int(port_file.read_text().splitlines()[0])
            return proc, port
        if proc.poll() is not None:
            raise RuntimeError("chrome exited before exposing a debug port")
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("chrome did not open a debug port")


def page_ws(port: int) -> str:
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as resp:
                targets = json.loads(resp.read())
            for target in targets:
                if target.get("type") == "page":
                    return target["webSocketDebuggerUrl"]
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("no page target")


# --------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------
async def run(args) -> int:
    import websockets

    checks = Checks()
    dist = Path(args.dist).resolve()
    if not (dist / "index.html").exists():
        print(f"missing build at {dist}; run `cd frontend && npm run build`", file=sys.stderr)
        return 2

    port = free_port()
    server = start_preview(dist, port)
    chrome, debug_port = start_chrome(args.chrome, Path(args.profile))
    ws = None
    try:
        ws = await websockets.connect(page_ws(debug_port), max_size=64 * 1024 * 1024)
        cdp = CDP(ws)
        await cdp.start()
        await cdp.send("Page.enable")
        await cdp.send("Runtime.enable")
        await cdp.send("Log.enable")
        await cdp.send(
            "Emulation.setDeviceMetricsOverride",
            width=1500,
            height=1000,
            deviceScaleFactor=1,
            mobile=False,
        )
        source = SHIM.read_text()
        await cdp.send("Page.addScriptToEvaluateOnNewDocument", source=source)

        url = args.url or f"http://127.0.0.1:{port}/index.html"
        print(f"→ {url}")
        await cdp.send("Page.navigate", url=url)
        await cdp.wait_for("window.__wubFM1 && window.__fm1Emu", label="app + emulator")
        await cdp.wait_for("!!document.getElementById('btn-play')", label="transport")

        # -- keep the browser honest: no real MIDI device may exist ----------
        print("\n[1] emulator shim is the only MIDI source")
        state = await cdp.js("window.__wubFM1.state()")
        checks.check(state["supported"], "Web MIDI API visible to the app")
        checks.eq(state["connected"], False, "no input attached before CONNECT")
        checks.eq(state["devices"], [], "device list empty before CONNECT")
        strip = await cdp.js("document.getElementById('fm1-strip') !== null")
        checks.check(strip, "FM-1 strip rendered under the nebula")

        # -- start the transport --------------------------------------------
        print("\n[2] transport + kit bank")
        await cdp.click("#btn-play")
        await cdp.wait_for(
            "document.getElementById('btn-play')?.classList.contains('active')",
            timeout=30,
            label="playing",
        )
        checks.check(True, "transport reports playing")
        counts = {}
        for stem in ("drums", "bass", "vocals", "other"):
            n = await cdp.wait_for(
                f"window.__wubFM1.slices('{stem}').length", timeout=90, label=f"{stem} slices"
            )
            counts[stem] = int(n)
        checks.check(all(n > 50 for n in counts.values()), "all four stems expose a phrase pool", str(counts))

        # -- connect the emulated controller --------------------------------
        print("\n[3] CONNECT MIDI + auto-pick")
        await cdp.click("#fm1-connect")
        await cdp.wait_for("window.__wubFM1.state().connected", timeout=15, label="midi connected")
        state = await cdp.js("window.__wubFM1.state()")
        checks.eq(state["deviceName"], "CTL-1 FM-1", "auto-picked the CTL-1 port")
        checks.eq(state["bank"], "C", "default bank C")
        checks.eq(list(state["ccs"]), [16, 17, 18, 19], "bank C controller numbers")
        tone = await cdp.js("document.getElementById('fm1-status').dataset.tone")
        checks.eq(tone, "ok", "status strip turns green")
        options = await cdp.js(
            "[...document.querySelectorAll('#fm1-input-select option')].map(o => o.textContent)"
        )
        checks.check("CTL-1 FM-1" in options, "input select lists the device", str(options))

        # -- wheel on the emulated knob 1 (real DOM input event) ------------
        print("\n[4] wheel the emulated K1 encoder")
        before = await cdp.js("window.__wubFM1.state().lastKnob")
        await cdp.wheel("#emu-dial-0", -100)
        await asyncio.sleep(0.25)
        knob = await cdp.js("window.__wubFM1.state().lastKnob")
        checks.eq(before, None, "no knob event before the first turn")
        checks.eq(knob["stem"] if knob else None, "drums", "K1 routed to DRUMS")
        checks.eq(knob["cc"] if knob else None, 16, "K1 sent CC 16")
        label = await cdp.js("document.getElementById('fm1-val-drums').textContent")
        checks.check("/" in label, "virtual cell shows slot/count", label)
        emu = await cdp.js("window.__fm1Emu.state")
        checks.eq(emu["values"][0], 1, "encoder advanced one detent")

        # -- scrolling and acceleration -------------------------------------
        print("\n[5] endless scroll + acceleration")
        n = counts["drums"]
        # Align device and app on the same raw value + slot before scrolling.
        await cdp.js("window.__fm1Emu.abs(0, 0)")
        await cdp.js("window.__wubFM1.select('drums', 0)")
        await asyncio.sleep(0.05)
        await cdp.js("window.__fm1Emu.turn(0, 1)")
        await asyncio.sleep(0.25)
        label = await cdp.js("document.getElementById('fm1-val-drums').textContent")
        checks.check(label.endswith("/%d" % n), "one detent from slot 0 lands on slot 1", label)
        await cdp.js("window.__fm1Emu.turn(0, 6)")
        await asyncio.sleep(0.25)
        label = await cdp.js("document.getElementById('fm1-val-drums').textContent")
        checks.check("· 8/%d" % n in label, "fast spin moves six detents", label)

        # wrap backwards past the start
        await cdp.js("window.__fm1Emu.abs(0, 0)")
        await cdp.js("window.__wubFM1.select('drums', 0)")
        await cdp.js("window.__fm1Emu.turn(0, -1)")
        await asyncio.sleep(0.25)
        label = await cdp.js("document.getElementById('fm1-val-drums').textContent")
        checks.check("· %d/%d" % (n, n) in label, "turning back wraps to the last slice", label)

        # -- all four knobs -> four stems -----------------------------------
        print("\n[6] K1..K4 map to DRUMS/BASS/VOCALS/OTHER")
        reticles = await cdp.js("window.__wubFM1.reticles()")
        for index, stem in enumerate(("drums", "bass", "vocals", "other")):
            await cdp.js(f"window.__wubFM1.select('{stem}', 3)")
            # Align the emulated encoder with the raw value select() just used.
            await cdp.js(
                f"(() => {{ const n = window.__wubFM1.slices('{stem}').length;"
                f" window.__fm1Emu.abs({index}, Math.round(3*127/Math.max(1, n-1))); }})()"
            )
            await cdp.js(f"window.__fm1Emu.turn({index}, -2)")
            # 180 ms debounce + a reticle glide: give the commit time to land.
            await asyncio.sleep(0.7)
            knob = await cdp.js("window.__wubFM1.state().lastKnob")
            ccs = await cdp.js("window.__wubFM1.state().ccs")
            checks.eq(knob["stem"] if knob else None, stem, f"K{index + 1} routes to {stem.upper()}")
            checks.eq(knob["cc"] if knob else None, ccs[index], f"K{index + 1} uses CC {ccs[index]}")
            point = await cdp.js(f"window.__wubFM1.slices('{stem}')[1]")
            target = await cdp.js(f"window.__wubFM1.reticles()['{stem}']")
            moved = abs(target["x"] - reticles[stem]["x"]) > 1e-4 or abs(target["y"] - reticles[stem]["y"]) > 1e-4
            checks.check(moved, f"{stem} reticle moved", f"{target['x']:.3f},{target['y']:.3f}")
            checks.check(
                abs(target["x"] - point["x"]) < 6e-3 and abs(target["y"] - point["y"]) < 6e-3,
                f"{stem} reticle sits on the selected slice",
                f"{target['x']:.3f},{target['y']:.3f} vs {point['x']:.3f},{point['y']:.3f}",
            )
            remix = {s["stem"]: s for s in await cdp.js("window.__wubFM1.remix()")}
            pin = remix.get(stem, {}).get("pin")
            checks.check(pin is not None, f"{stem} phrase pinned after the debounce", str(pin))
            if pin is not None:
                checks.check(
                    abs(pin["x"] - point["x"]) < 2e-3 and abs(pin["y"] - point["y"]) < 2e-3,
                    f"{stem} pin targets the knob slice",
                )

        # -- device bank vs app bank ---------------------------------------
        print("\n[7] bank matching")
        await cdp.js("window.__wubFM1.bank('C')")
        await cdp.js("window.__fm1Emu.setBank('B')")
        slot_before = await cdp.js("document.getElementById('fm1-val-bass').textContent")
        knob_before = await cdp.js("window.__wubFM1.state().lastKnob")
        await cdp.js("window.__fm1Emu.turn(1, 3)")
        await asyncio.sleep(0.3)
        msg = await cdp.js("window.__wubFM1.state().lastMessage")
        knob = await cdp.js("window.__wubFM1.state().lastKnob")
        checks.eq(msg["cc"] if msg else None, 72, "device in bank B sends CC 72 on K2")
        checks.eq(knob, knob_before, "app in bank C ignores that CC")
        checks.eq(
            await cdp.js("document.getElementById('fm1-val-bass').textContent"),
            slot_before,
            "unmapped bank does not move a stem",
        )
        await cdp.js("window.__wubFM1.bank('B')")
        await cdp.js("window.__fm1Emu.turn(1, 3)")
        await asyncio.sleep(0.3)
        knob = await cdp.js("window.__wubFM1.state().lastKnob")
        checks.eq(knob["cc"] if knob else None, 72, "matching bank maps CC 72 to K2")
        checks.check(
            await cdp.js("document.getElementById('fm1-val-bass').textContent") != slot_before,
            "matching bank moves the stem again",
        )
        await cdp.js("window.__wubFM1.bank('C')")
        await cdp.js("window.__fm1Emu.setBank('C')")

        # -- hot-plug -------------------------------------------------------
        print("\n[8] unplug / replug")
        await cdp.js("window.__fm1Emu.unplug()")
        await asyncio.sleep(0.3)
        state = await cdp.js("window.__wubFM1.state()")
        checks.eq(state["connected"], False, "unplug detaches the input")
        checks.eq(state["devices"], [], "unplug empties the device list")
        options = await cdp.js(
            "[...document.querySelectorAll('#fm1-input-select option')].map(o => o.textContent)"
        )
        checks.check("— no MIDI inputs —" in options, "select shows the empty state", str(options))
        await cdp.js("window.__fm1Emu.plug()")
        await cdp.wait_for("window.__wubFM1.state().connected", timeout=10, label="replug")
        checks.check(True, "replug re-attaches the CTL-1 automatically")
        status = await cdp.js(
            "{ text: document.getElementById('fm1-status').textContent,"
            "  tone: document.getElementById('fm1-status').dataset.tone }"
        )
        checks.eq(status["tone"], "ok", "status recovers to green after replug")
        checks.check("CTL-1" in status["text"], "status names the live device", status["text"])

        # -- audio is still alive after all that ---------------------------
        print("\n[9] audio survives knob churn")
        probe = None
        for _ in range(20):
            probe = await cdp.js("window.__wubAudioProbe()")
            if probe and probe.get("peak", 0) > 0:
                break
            await asyncio.sleep(0.25)
        checks.check(bool(probe and probe.get("peak", 0) > 0), "master bus is making sound", f"peak={probe.get('peak') if probe else None}")
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
            shot = await cdp.send(
                "Page.captureScreenshot",
                format="png",
                captureBeyondViewport=True,
            )
            out = Path(args.screenshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(__import__("base64").b64decode(shot["data"]))
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
    parser.add_argument("--dist", default=str(ROOT / "frontend" / "dist"))
    parser.add_argument("--chrome", default=os.environ.get("WUB_CHROME", "google-chrome"))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--url", default=None, help="override page URL (otherwise serve --dist)")
    parser.add_argument("--screenshot", default=None)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
