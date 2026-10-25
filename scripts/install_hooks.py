#!/usr/bin/env python3
"""Installer for git commit hooks."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
HOOKS_SRC = ROOT_DIR / ".githooks" / "pre-commit"
GIT_HOOK_DEST = ROOT_DIR / ".git" / "hooks" / "pre-commit"


def install() -> None:
    if not HOOKS_SRC.exists():
        print(f"Hook source not found: {HOOKS_SRC}")
        return

    # Set git core.hooksPath
    try:
        subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT_DIR, check=True)
        print("Configured git core.hooksPath = .githooks")
    except Exception as e:
        print(f"Notice: git config core.hooksPath: {e}")

    # Copy to .git/hooks/pre-commit
    if GIT_HOOK_DEST.parent.exists():
        shutil.copy(HOOKS_SRC, GIT_HOOK_DEST)
        GIT_HOOK_DEST.chmod(0o755)
        print("Copied hook to .git/hooks/pre-commit")

    HOOKS_SRC.chmod(0o755)
    print("Git pre-commit version control hook installed successfully.")


if __name__ == "__main__":
    install()
