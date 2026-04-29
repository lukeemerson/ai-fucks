#!/usr/bin/env python3
"""Serve the results/ dashboard on localhost:8080."""
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT    = 8080
RESULTS = Path(__file__).parent / "results"

if not RESULTS.exists():
    raise SystemExit("results/ not found — run the analyzer first")

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(RESULTS))

class _Server(socketserver.TCPServer):
    allow_reuse_address = True

with _Server(("127.0.0.1", PORT), handler) as httpd:
    print(f"Serving {RESULTS} at http://localhost:{PORT}/dashboard.html")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(f"http://localhost:{PORT}/dashboard.html")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
