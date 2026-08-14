"""Small local server for the future-GitHub-Pages-compatible dashboard."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DashboardHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        if self.path.endswith(".json"):
            self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


def serve_dashboard(web_root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    root = web_root.resolve()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"dashboard entrypoint not found: {root / 'index.html'}")
    handler = partial(DashboardHandler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Regime dashboard: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
