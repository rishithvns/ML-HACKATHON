"""Serve the evaluator UI (runs on Laptop 2).

    python ui/serve.py --port 8501

Then open http://localhost:8501/?api=http://<laptop-1-ip>:8000
(the API address can also be typed into the page; it is remembered).
"""
import argparse
import functools
import http.server
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--host", default="0.0.0.0")
args = ap.parse_args()

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
    print(f"PathSnap UI on http://localhost:{args.port}/  (Ctrl+C to stop)")
    httpd.serve_forever()
