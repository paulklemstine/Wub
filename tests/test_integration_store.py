"""Unified user storage for all integrations (Twitch + MIDI)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundle(name: str, source: str) -> Path:
    out = Path("/tmp/opencode") / f"{name}.cjs"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src" / source), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    return out


def test_integration_store_roundtrip_and_migration():
    out = bundle("integration_store", "integration_store.ts")
    script = r"""
const assert = require('node:assert/strict');
const m = require(process.argv[1]);
assert.equal(typeof m.loadIntegrations, 'function', 'loader exported');
assert.equal(typeof m.saveIntegrations, 'function', 'saver exported');
assert.equal(typeof m.clearIntegrations, 'function', 'clearer exported');
assert.equal(m.INTEGRATION_STORAGE_KEY, 'wub_integrations_v1', 'single key');
m.selfTestIntegrationStore();
"""
    result = subprocess.run(["node", "-e", script, str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_integrations_wired_in_app():
    src = (ROOT / "frontend/src/main.ts").read_text(encoding="utf-8")
    assert "integration_store" in src, "main uses the unified store"
    assert "wub_tw_channel" not in src, "no legacy Twitch keys left in app"
    assert "saveIntegrations({ midi:" in src, "MIDI input/bank persist"
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert 'id="tw-username"' in html and 'id="tw-token"' in html
