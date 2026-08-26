"""Small local server for the future-GitHub-Pages-compatible dashboard."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class DashboardHandler(SimpleHTTPRequestHandler):
    payload_bytes: bytes | None = None
    comparison_bytes: bytes | None = None
    json_overrides_enabled = False

    def _override_json_bytes(self) -> bytes | None:
        request_path = urlsplit(self.path).path
        if request_path == "/data/regime-results.json":
            return self.payload_bytes
        if request_path == "/data/v5-vs-v4-comparison.json":
            return self.comparison_bytes
        return None

    def _serve_override_json(self, *, include_body: bool) -> bool:
        request_path = urlsplit(self.path).path
        if request_path not in {
            "/data/regime-results.json",
            "/data/v5-vs-v4-comparison.json",
        }:
            return False
        if not self.json_overrides_enabled:
            return False
        if request_path == "/data/regime-results.json" and self.payload_bytes is None:
            return False
        selected = self._override_json_bytes()
        if selected is None:
            self.send_error(404, "dashboard JSON is unavailable")
            return True
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(selected)))
        self.end_headers()
        if include_body:
            self.wfile.write(selected)
        return True

    def do_GET(self) -> None:
        if not self._serve_override_json(include_body=True):
            super().do_GET()

    def do_HEAD(self) -> None:
        if not self._serve_override_json(include_body=False):
            super().do_HEAD()

    def end_headers(self) -> None:
        if urlsplit(self.path).path.endswith(".json"):
            self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


def _regular_json_path(path: Path | None, *, label: str) -> Path | None:
    if path is None:
        return None
    selected = path.resolve()
    if path.is_symlink() or not selected.is_file():
        raise FileNotFoundError(f"{label} not found or unsafe: {path}")
    return selected


def _frozen_json_bytes(
    path: Path | None,
    value: bytes | None,
    *,
    label: str,
) -> bytes | None:
    """Resolve one override once so later file replacement cannot change it."""

    if path is not None and value is not None:
        raise ValueError(f"{label} path and bytes are mutually exclusive")
    if value is not None:
        if not isinstance(value, bytes):
            raise TypeError(f"{label} bytes must be immutable bytes")
        return value
    selected = _regular_json_path(path, label=label)
    return selected.read_bytes() if selected is not None else None


def serve_dashboard(
    web_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    payload: Path | None = None,
    comparison: Path | None = None,
    payload_bytes: bytes | None = None,
    comparison_bytes: bytes | None = None,
) -> None:
    root = web_root.resolve()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"dashboard entrypoint not found: {root / 'index.html'}")
    selected_payload = _frozen_json_bytes(
        payload,
        payload_bytes,
        label="dashboard payload",
    )
    selected_comparison = _frozen_json_bytes(
        comparison,
        comparison_bytes,
        label="dashboard comparison",
    )
    configured_handler = type(
        "ConfiguredDashboardHandler",
        (DashboardHandler,),
        {
            "payload_bytes": selected_payload,
            "comparison_bytes": selected_comparison,
            "json_overrides_enabled": any(
                value is not None
                for value in (
                    payload,
                    comparison,
                    payload_bytes,
                    comparison_bytes,
                )
            ),
        },
    )
    handler = partial(configured_handler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Regime dashboard: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
