"""Static server for the web app. Hosts code only — no inference.

Sets COOP/COEP so SharedArrayBuffer is available (needed by the threaded
WASM fallback when WebGPU is unavailable). Run:

    python serve.py

Then open http://127.0.0.1:8000
"""
import os
import http.server
import socketserver

PORT = 8000
DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Serving http://127.0.0.1:{PORT}")
        httpd.serve_forever()
