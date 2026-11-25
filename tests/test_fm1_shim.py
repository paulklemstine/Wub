"""Static contract for the device shim (tools/fm1_shim).

The shim is a bench, not firmware: it fetches the shipped bank over HTTPS and
wires every CTL-1 message. These checks keep the control map and the bank URL
from drifting without needing Chrome or the network.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "tools" / "fm1_shim"
DEVICE = SHIM / "device.js"
PAGE = SHIM / "index.html"
SERVE = SHIM / "serve.py"
README = SHIM / "README.md"
DOC = ROOT / "docs/FM1_CONTROLLER.md"


def test_device_shim_files_exist() -> None:
    for path in (DEVICE, PAGE, SERVE, README):
        assert path.exists(), f"missing {path}"


def test_shim_defaults_to_the_shipped_bank_on_github() -> None:
    source = DEVICE.read_text(encoding="utf-8")
    assert "raw.githubusercontent.com/paulklemstine/Wub/main/frontend/public/data/kits" in source
    assert "kit_manifest.json" in source
    # Local override for offline work and tests.
    assert "base=/bank" in README.read_text(encoding="utf-8")


def test_shim_reads_the_real_manifest_shape() -> None:
    source = DEVICE.read_text(encoding="utf-8")
    for token in ("doc.kits", "kit.phrases", "phrase.stems", "part.file", "duration_sec"):
        assert token in source, f"manifest field {token} not used"


def test_shim_handles_every_ctl1_message() -> None:
    source = DEVICE.read_text(encoding="utf-8")
    # Transport, clock, program change, panic, sustain, mod wheel.
    for token in ("0xf8", "0xfa", "0xfc", "0xc0", "cc === 123", "cc === 64", "cc === 1", "cc === 7"):
        assert token in source, f"missing handler for {token}"
    # Knob banks must match the CTL-1 firmware table.
    assert 'C: [16, 17, 18, 19]' in source
    assert '"drums", "bass", "vocals", "other"' in source
    assert "KEY_BASE = 53" in source  # F3
    assert "KEY_COUNT = 27" in source


def test_shim_commits_on_the_next_bar() -> None:
    source = DEVICE.read_text(encoding="utf-8")
    assert "state.pending[stem] = phrase" in source
    assert "startAt + (lastBar + 1) * barSec()" in source
    # No stretching: tempo fit is playbackRate only.
    assert "playbackRate.value = rateFor(buffer)" in source
    assert "detune" not in source


def test_docs_point_at_the_shim() -> None:
    doc = DOC.read_text(encoding="utf-8")
    assert "tools/fm1_shim/" in doc
    assert "tools/fm1_shim_test.py" in doc
