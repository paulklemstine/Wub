"""Tests for incremental version control and Git commit hook integration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VERSION_JSON = ROOT_DIR / "version.json"
VERSION_TS = ROOT_DIR / "frontend" / "src" / "version.ts"
INDEX_HTML = ROOT_DIR / "frontend" / "index.html"
PRE_COMMIT_HOOK = ROOT_DIR / ".githooks" / "pre-commit"
BUMP_SCRIPT = ROOT_DIR / "scripts" / "bump_version.py"


def test_version_files_exist_and_consistent():
    assert VERSION_JSON.exists(), "version.json must exist"
    data = json.loads(VERSION_JSON.read_text())

    for key in ("major", "minor", "patch", "build", "version", "commit", "timestamp"):
        assert key in data, f"Key '{key}' missing from version.json"

    assert data["build"] >= 36
    assert data["version"] == f"{data['major']}.{data['minor']}.{data['build']}"

    assert VERSION_TS.exists(), "frontend/src/version.ts must exist"
    ts_content = VERSION_TS.read_text()
    assert f'version: "{data["version"]}"' in ts_content
    assert f'build: {data["build"]}' in ts_content


def test_index_html_has_version_elements():
    assert INDEX_HTML.exists()
    html = INDEX_HTML.read_text()
    assert 'id="mc-version-badge"' in html
    assert 'class="mc-version-badge"' in html
    assert 'id="footer-version-tag"' in html


def test_pre_commit_hook_executable():
    assert PRE_COMMIT_HOOK.exists()
    assert os.access(PRE_COMMIT_HOOK, os.X_OK), ".githooks/pre-commit must be executable"
    content = PRE_COMMIT_HOOK.read_text()
    assert "bump_version.py" in content


def test_bump_version_dry_run():
    res = subprocess.run(
        [sys.executable, str(BUMP_SCRIPT), "--dry-run"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[dry-run] Version:" in res.stdout
