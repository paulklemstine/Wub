"""In-the-loop policy selection using the live app's quality telemetry.

For each candidate bank_transition.json the harness serves it to the built app
in headless Chrome, plays a short set, samples window.__wubQualitySummary(),
and scores the mix. This evaluates policies on the assembled audio mix rather
than on offline AUC, which repeatedly failed to predict live quality.

Requires: google-chrome, python websockets/requests (already used by the repo's
diagnostics). Start with:

    python3 scripts/headless_quality_search.py --variants variants.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import websockets

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_copy_metrics import corpus_stats, style_fidelity

CDP_PORT = 9222
SERVER_PORT = 4173
DEFAULT_ARTIFACT = "data/kits/bank_transition.json"
DIST_POLICY = Path("frontend/dist") / DEFAULT_ARTIFACT
URL = f"http://127.0.0.1:{SERVER_PORT}/"


async def cdp(ws, method, params=None, counter=None):
    cid = counter[0]
    counter[0] += 1
    await ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if msg.get("id") == cid:
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg.get("result", {})


async def evaluate(ws, expr, counter):
    r = await cdp(ws, "Runtime.evaluate",
                  {"expression": expr, "returnByValue": True, "awaitPromise": True}, counter)
    return r.get("result", {}).get("value")


def objective(q: Dict[str, Any]) -> float:
    h = min(1.0, q["harmony"]["p10"] / 0.6)
    clash = q["harmony"]["clashRate"]
    arc = min(1.0, q["arrangement"]["arcDeviation"] / 0.5)
    tracks = min(1.0, q["churn"]["uniqueTracks"] / 10.0)
    dips = min(1.0, q["seams"]["dipEvents"] / 5.0)
    return round(0.45 * h - 0.25 * clash - 0.25 * arc + 0.25 * tracks - 0.15 * dips, 4)


async def run_variant(ws, counter, name: str, seconds: int, interval: int):
    await cdp(ws, "Network.enable", {"maxTotalBufferSize": 10_000_000}, counter)
    await cdp(ws, "Network.setCacheDisabled", {"cacheDisabled": True}, counter)
    await cdp(ws, "Page.navigate", {"url": URL + "?variant=" + name}, counter)
    for _ in range(30):
        await asyncio.sleep(1)
        if await evaluate(ws, "typeof window.__wubQualitySummary === 'function'", counter) is True:
            break
    else:
        return {"name": name, "error": "app did not initialize"}
    await evaluate(ws, "document.getElementById('btn-play')?.click(); 'clicked'", counter)
    samples = []
    elapsed = 0
    while elapsed < seconds:
        await asyncio.sleep(interval)
        elapsed += interval
        raw = await evaluate(
            ws,
            "(() => { const q = window.__wubQualitySummary && window.__wubQualitySummary();"
            " return q ? JSON.stringify(q) : null; })()", counter,
        )
        if raw:
            q = json.loads(raw)
            samples.append(q)
    usable = samples[1:] or samples
    if not usable:
        return {"name": name, "error": "no samples"}
    avg_obj = sum(objective(q) for q in usable) / len(usable)
    movement = await evaluate(
        ws, "JSON.stringify(window.__wubMovementSummary?.()?.log ?? [])", counter)
    return {
        "name": name,
        "objective": round(avg_obj, 4),
        "movement": json.loads(movement) if movement else [],
        "harmony_p10": round(sum(q["harmony"]["p10"] for q in usable) / len(usable), 4),
        "clash_rate": round(sum(q["harmony"]["clashRate"] for q in usable) / len(usable), 4),
        "arc_deviation": round(sum(q["arrangement"]["arcDeviation"] for q in usable) / len(usable), 4),
        "unique_tracks": usable[-1]["churn"]["uniqueTracks"],
        "dip_events": usable[-1]["seams"]["dipEvents"],
        "samples": len(samples),
    }


def ensure_services():
    try:
        requests.get(URL, timeout=2)
    except Exception:
        subprocess.Popen(
            ["python3", "-m", "http.server", str(SERVER_PORT), "--directory", "dist"],
            cwd=Path("frontend").resolve(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
    try:
        requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=2)
    except Exception:
        subprocess.Popen(
            ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--mute-audio",
             "--autoplay-policy=no-user-gesture-required", f"--remote-debugging-port={CDP_PORT}",
             "--user-data-dir=/tmp/opencode/chrome-quality-search", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        time.sleep(3)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=Path, required=True,
                    help="JSON object mapping variant name -> bank_transition.json path")
    ap.add_argument("--seconds", type=int, default=45)
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT,
                    help="artifact path relative to frontend/dist, e.g. data/kits/phrase_conductor.json")
    args = ap.parse_args()
    ensure_services()
    variants = json.loads(args.variants.read_text())
    global DIST_POLICY
    DIST_POLICY = Path("frontend/dist") / args.artifact
    pages = requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=5).json()
    page = next(p for p in pages if p.get("type") == "page")
    counter = [1]
    corpus = corpus_stats(
        Path("data/processed"),
        cohort_artifact=Path("frontend/public/data/kits/transition_cohort.json"),
    )
    results = []
    original = DIST_POLICY.read_bytes()
    try:
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=10_000_000) as ws:
            for name, path in variants.items():
                shutil.copyfile(path, DIST_POLICY)
                result = await run_variant(ws, counter, name, args.seconds, args.interval)
                fid = style_fidelity(result.pop("movement", []), corpus)
                result["style_fidelity"] = fid["style_fidelity"]
                result["combined"] = round(
                    result.get("objective", -999) + 0.2 * (fid["style_fidelity"] - 0.5), 4)
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        DIST_POLICY.write_bytes(original)
    results.sort(key=lambda r: r.get("combined", r.get("objective", -999)), reverse=True)
    report = {"results": results, "winner": results[0]["name"] if results else None}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
