#!/usr/bin/env python3
"""Serve the results/ dashboard locally.

Bootstraps results/ from analyzer/dashboard.html and fixtures/seed_report.json
when missing, so a fresh clone can run `python3 server.py` and immediately see
a working dashboard.
"""

import argparse
import functools
import http.server
import shutil
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DASHBOARD = ROOT / "analyzer" / "dashboard.html"
SEED = ROOT / "fixtures" / "seed_report.json"


def bootstrap() -> None:
    RESULTS.mkdir(exist_ok=True)
    target = RESULTS / "dashboard.html"
    if not target.exists() or target.read_bytes() != DASHBOARD.read_bytes():
        shutil.copy(DASHBOARD, target)
        shutil.copy(DASHBOARD, RESULTS / "index.html")
    if not (RESULTS / "report.json").exists():
        if not SEED.exists():
            raise SystemExit(f"No report.json and no seed at {SEED}")
        shutil.copy(SEED, RESULTS / "report.json")
        print(f"Seeded results/report.json from {SEED.relative_to(ROOT)}")


class _Server(socketserver.TCPServer):
    allow_reuse_address = True


def serve(port: int, open_browser: bool) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(RESULTS))
    try:
        httpd = _Server(("127.0.0.1", port), handler)
    except OSError as e:
        if e.errno in (48, 98):  # EADDRINUSE on macOS / Linux
            raise SystemExit(f"Port {port} is in use. Pass --port <n> to pick another.") from e
        raise

    url = f"http://localhost:{port}/dashboard.html"
    print(f"Serving {RESULTS} at {url}")
    print("Press Ctrl+C to stop.\n")
    if open_browser:
        webbrowser.open(url)
    try:
        with httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = parser.parse_args()
    bootstrap()
    serve(args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
