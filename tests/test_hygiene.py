"""Hygiene: every shipped file must compile; every CLI must parse --help."""
from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_python_compiles():
    bad = []
    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in p.parts or ".venv" in p.parts:
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError:
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"uncompilable files: {bad}"


def test_cli_help_parses():
    for argv in (["scripts/analyze_corpus.py", "--help"],
                 ["scripts/build_umap.py", "--help"],
                 ["scripts/demo_wubsynth.py", "--help"],
                 ["scripts/generate_mashup.py", "--help"],
                 ["scripts/sample_wubgpt.py", "--help"],
                 ["scripts/train_wubgpt.py", "--help"],
                 ["scripts/verify_slicing.py", "--help"]):
        r = subprocess.run([sys.executable, str(ROOT / argv[0]), argv[1]],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{argv[0]} --help failed: {r.stderr[-500:]}"


def test_repeatable_kv_flags_accumulate():
    """Regression: repeated --force-tracks/--chaos must all apply (append)."""
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    from scripts.generate_mashup import parse_kv  # noqa: E402

    fake_argv = [["drums=aaa"], ["bass=bbb"], ["vocals=ccc", "other=ddd"]]
    assert parse_kv(fake_argv) == {"drums": "aaa", "bass": "bbb",
                                   "vocals": "ccc", "other": "ddd"}
