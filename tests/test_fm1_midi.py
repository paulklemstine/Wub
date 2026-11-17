"""Static and logic tests for the M-VAVE FM-1 / CTL-1 knob phrase controller.

The mapping module is dependency-free TypeScript so it can be executed by
Node's built-in type stripping; the browser UI is covered by static contract
checks against `frontend/index.html` and `frontend/src/main.ts`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "frontend/src/fm1_midi.ts"
MAIN = ROOT / "frontend/src/main.ts"
INDEX = ROOT / "frontend/index.html"
DOC = ROOT / "docs/FM1_CONTROLLER.md"


def test_fm1_midi_html_static_contract() -> None:
    assert INDEX.exists(), "frontend/index.html must exist"
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="fm1-strip"' in html
    assert 'id="fm1-connect"' in html
    assert 'id="fm1-input-select"' in html
    assert 'id="fm1-bank-select"' in html
    assert 'id="fm1-status"' in html

    # One virtual slider per stem, in DJ-surface order (DRUMS, BASS, VOCALS, OTHER).
    positions = []
    for stem in ("drums", "bass", "vocals", "other"):
        marker = f'id="fm1-knob-{stem}"'
        assert marker in html, f"missing virtual knob for {stem}"
        positions.append(html.index(marker))
    assert positions == sorted(positions), "FM-1 knob sliders must follow DJ stem order"

    # The four DJ-surface kill buttons share that same order.
    dj_order = [html.index(f'data-dj-kill="{stem}"') for stem in ("drums", "bass", "vocals", "other")]
    assert dj_order == sorted(dj_order)


def test_fm1_main_wiring_static_contract() -> None:
    assert MAIN.exists()
    source = MAIN.read_text(encoding="utf-8")

    assert "Fm1MidiInput" in source
    assert "buildSliceOrder" in source
    assert "knobValueToSlot" in source
    assert "scrollSlot" in source
    assert "__wubFM1" in source
    # Knob commits must reuse the reticle-drop path (pin + next-bar commit).
    assert "setReticleSteering(stem, point.x, point.y, true)" in source
    assert "setReticleTarget(stem, point.x, point.y" in source
    # The knob maps onto the live legal pool exposed by the remix engine.
    assert "conductor.stemSlices(stem)" in source
    remix = (ROOT / "frontend/src/stem_remix.ts").read_text(encoding="utf-8")
    assert "stemPoolPoints(stem: StemName)" in remix
    conductor = (ROOT / "frontend/src/client_ml_sequencer.ts").read_text(encoding="utf-8")
    assert "public stemSlices(stem: StemName)" in conductor


def test_fm1_docs_exist_and_cover_firmware_requirement() -> None:
    assert DOC.exists(), "docs/FM1_CONTROLLER.md must document the controller"
    text = DOC.read_text(encoding="utf-8")
    assert "CTL-1" in text
    assert "charlesvestal/fm1-ctl" in text
    assert "midi-in only" in text.lower()
    assert "factory firmware" in text.lower()
    assert "__wubFM1" in text
    assert "emulator" in text.lower()


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    assert node is not None, "Node is required to test the TypeScript mapping module"
    return subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


def test_fm1_knob_mapping_with_node() -> None:
    assert MODULE.exists(), "frontend/src/fm1_midi.ts must exist"
    script = r'''
import assert from "node:assert/strict";
const m = await import("__MODULE__");

assert.deepEqual([...m.FM1_KNOB_STEMS], ["drums", "bass", "vocals", "other"]);
assert.deepEqual([...m.FM1_CC_BANKS.A], [74, 71, 1, 7]);
assert.deepEqual([...m.FM1_CC_BANKS.B], [73, 72, 75, 76]);
assert.deepEqual([...m.FM1_CC_BANKS.C], [16, 17, 18, 19]);
assert.deepEqual([...m.FM1_CC_BANKS.D], [20, 21, 22, 23]);
assert.equal(m.FM1_DEFAULT_BANK, "C");

assert.equal(m.knobForCc(16, "C"), 0);
assert.equal(m.knobForCc(19, "C"), 3);
assert.equal(m.knobForCc(74, "A"), 0);
assert.equal(m.knobForCc(7, "A"), 3);
assert.equal(m.knobForCc(16, "D"), -1);
assert.equal(m.isFm1BankId("C"), true);
assert.equal(m.isFm1BankId("E"), false);

// Linear slot map: first detent = first slice, last detent = last slice.
for (const count of [1, 2, 127, 128, 129, 2565]) {
  assert.equal(m.knobValueToSlot(0, count), 0);
  assert.equal(m.knobValueToSlot(127, count), count - 1);
  let previous = -1;
  for (let v = 0; v <= 127; v++) {
    const slot = m.knobValueToSlot(v, count);
    assert.ok(slot >= 0 && slot < count);
    assert.ok(slot >= previous, "knob sweep must never move backwards");
    previous = slot;
  }
}
assert.equal(m.knobValueToSlot(0, 0), 0);
assert.equal(m.knobValueToSlot(64, 100), 50);

// Endless-encoder scrolling: one click = one slice, wrap both ways,
// acceleration preserved, and no movement when the value repeats.
assert.equal(m.scrollSlot(10, 11, 10, 100), 11);
assert.equal(m.scrollSlot(10, 16, 10, 100), 16);
assert.equal(m.scrollSlot(10, 10, 10, 100), 10);
assert.equal(m.scrollSlot(0, 127, 0, 100), 99);
assert.equal(m.scrollSlot(99, 0, 127, 100), 0);
assert.equal(m.scrollSlot(3, 2, 3, 100), 2);
assert.equal(m.scrollSlot(Number.NaN, 64, 0, 100), 50);
assert.equal(m.scrollSlot(5, 9, 8, 1), 0);

// Slice order is catalogue order (track id then phrase id, numeric aware).
const points = [
  { stem: "drums", trackId: "track10", phraseId: "p2", x: 0.5, y: 0.5 },
  { stem: "drums", trackId: "track2", phraseId: "p10", x: 0.1, y: 0.2 },
  { stem: "drums", trackId: "track2", phraseId: "p9", x: 0.3, y: 0.4 },
  { stem: "drums", trackId: "track2", phraseId: "p9", x: 0.3, y: 0.4 },
];
const ordered = m.buildSliceOrder(points);
assert.deepEqual(
  ordered.map((p) => `${p.trackId}/${p.phraseId}`),
  ["track2/p9", "track2/p10", "track10/p2"],
);

// CC receiver: mapped CCs call onKnob, everything else is ignored.
const seen = [];
const fm1 = new m.Fm1MidiInput({ bank: "C", onKnob: (stem, value) => seen.push([stem, value]) });
assert.deepEqual(fm1.injectCC(17, 64), { stem: "bass", value: 64, cc: 17 });
assert.deepEqual(seen, [["bass", 64]]);
assert.equal(fm1.injectCC(99, 64), null);
assert.equal(fm1.handleRaw([0x90, 17, 64]), null);
assert.equal(fm1.handleRaw([0xb0, 17]), null);
fm1.setBank("D");
assert.deepEqual(fm1.injectCC(21, 127), { stem: "bass", value: 127, cc: 21 });
assert.equal(fm1.ccForKnob(0), 20);
assert.equal(fm1.ccForKnob(9), -1);
const state = fm1.state();
assert.equal(state.bank, "D");
assert.equal(state.connected, false);
assert.deepEqual([...state.ccs], [20, 21, 22, 23]);
console.log("ok");
'''.replace("__MODULE__", MODULE.as_uri())
    result = _run_node(script)
    assert result.returncode == 0, f"node mapping test failed:\n{result.stdout}\n{result.stderr}"
    assert "ok" in result.stdout


def test_fm1_banks_are_unique() -> None:
    """No controller number may appear twice across banks at the same knob."""
    script = r'''
import assert from "node:assert/strict";
const m = await import("__MODULE__");
const seen = new Set();
for (const bank of m.FM1_BANK_ORDER) {
  for (const cc of m.FM1_CC_BANKS[bank]) {
    assert.ok(cc >= 0 && cc <= 127, "CC out of range");
    assert.ok(!seen.has(cc), `CC ${cc} duplicated across banks`);
    seen.add(cc);
  }
}
console.log("ok");
'''
    result = _run_node(script.replace("__MODULE__", MODULE.as_uri()))
    assert result.returncode == 0, result.stderr
