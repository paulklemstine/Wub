#!/usr/bin/env python3
"""Serve the Wub device shim and, optionally, a local kit bank.

The page defaults to the kit bank on GitHub raw (`?base=` overrides it). For
offline work or tests, point `--bank` at a kit directory that contains
`kit_manifest.json` and `kits/`; it is exposed at `/bank/` so the page can be
opened with `?base=/bank`.

    python3 tools/fm1_shim/serve.py --port 8123
    python3 tools/fm1_shim/serve.py --port 8123 --bank frontend/public/data/kits
    # open http://127.0.0.1:8123/?base=/bank
"""

from __future__ import annotations

import argparse
import functools
import http.server
import mimetypes
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIM = Path(__file__).resolve().parent

mimetypes.add_type("audio/ogg", ".opus")
mimetypes.add_type("application/json", ".json")


class Handler(http.server.SimpleHTTPRequestHandler):
    bank: Path | None = None
    shim: Path = SHIM

    def translate_path(self, path: str) -> str:  # noqa: D102
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if self.bank is not None and clean.startswith("/bank/"):
            rel = clean[len("/bank/") :]
            return str((self.bank / rel).resolve())
        return str((self.shim / clean.lstrip("/")).resolve())

    def end_headers(self) -> None:  # noqa: D102
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102
        print(f"[shim] {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--bank",
        default=None,
        help="kit directory to expose at /bank (kit_manifest.json + kits/)",
    )
    args = parser.parse_args()

    Handler.bank = Path(args.bank).resolve() if args.bank else None
    handler = functools.partial(Handler, directory=str(SHIM))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"Wub device shim → http://{args.host}:{args.port}/")
        if Handler.bank:
            print(f"local bank       → http://{args.host}:{args.port}/?base=/bank")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
