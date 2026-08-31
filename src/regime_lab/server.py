"""Small local server for the future-GitHub-Pages-compatible dashboard."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from regime_lab.dashboard_split import build_dashboard_split
from regime_lab.publication_contract import PublicContractError


class DashboardHandler(SimpleHTTPRequestHandler):
    payload_bytes: bytes | None = None
    core_bytes: bytes | None = None
    research_bytes: bytes | None = None
    comparison_bytes: bytes | None = None
    selection_family_bytes: bytes | None = None
    json_overrides_enabled = False

    def _disable_conditional_cache(self) -> None:
        for name in ("If-Modified-Since", "If-None-Match"):
            if name in self.headers:
                del self.headers[name]

    def _override_json_bytes(self) -> bytes | None:
        request_path = urlsplit(self.path).path
        if request_path == "/data/regime-core.json":
            return self.core_bytes
        if request_path == "/data/regime-research.json":
            return self.research_bytes
        if request_path == "/data/regime-results.json":
            return self.payload_bytes
        if request_path == "/data/v5-vs-v4-comparison.json":
            return self.comparison_bytes
        if request_path == "/data/selection-family-audit.json":
            return self.selection_family_bytes
        return None

    def _serve_override_json(self, *, include_body: bool) -> bool:
        request_path = urlsplit(self.path).path
        if request_path not in {
            "/data/regime-core.json",
            "/data/regime-research.json",
            "/data/regime-results.json",
            "/data/v5-vs-v4-comparison.json",
            "/data/selection-family-audit.json",
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
        self._disable_conditional_cache()
        if not self._serve_override_json(include_body=True):
            super().do_GET()

    def do_HEAD(self) -> None:
        self._disable_conditional_cache()
        if not self._serve_override_json(include_body=False):
            super().do_HEAD()

    def end_headers(self) -> None:
        self.send_header(
            "Cache-Control",
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
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


def _dashboard_split_bytes(payload: bytes | None) -> tuple[bytes | None, bytes | None]:
    """Project a selected V5 payload so core-first loading cannot bypass it."""

    if payload is None:
        return None, None
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(document, dict):
        return None, None
    meta = document.get("meta")
    research = document.get("research")
    if (
        not isinstance(meta, dict)
        or not isinstance(meta.get("generation_id"), str)
        or not isinstance(research, dict)
    ):
        return None, None
    try:
        return build_dashboard_split(document, payload_raw=payload)
    except PublicContractError:
        return None, None


def serve_dashboard(
    web_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    payload: Path | None = None,
    comparison: Path | None = None,
    selection_family: Path | None = None,
    payload_bytes: bytes | None = None,
    comparison_bytes: bytes | None = None,
    selection_family_bytes: bytes | None = None,
) -> None:
    root = web_root.resolve()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"dashboard entrypoint not found: {root / 'index.html'}")
    selected_payload = _frozen_json_bytes(
        payload,
        payload_bytes,
        label="dashboard payload",
    )
    selected_core, selected_research = _dashboard_split_bytes(selected_payload)
    selected_comparison = _frozen_json_bytes(
        comparison,
        comparison_bytes,
        label="dashboard comparison",
    )
    selected_selection_family = _frozen_json_bytes(
        selection_family,
        selection_family_bytes,
        label="dashboard selection-family audit",
    )
    configured_handler = type(
        "ConfiguredDashboardHandler",
        (DashboardHandler,),
        {
            "payload_bytes": selected_payload,
            "core_bytes": selected_core,
            "research_bytes": selected_research,
            "comparison_bytes": selected_comparison,
            "selection_family_bytes": selected_selection_family,
            "json_overrides_enabled": any(
                value is not None
                for value in (
                    payload,
                    selected_core,
                    selected_research,
                    comparison,
                    selection_family,
                    payload_bytes,
                    comparison_bytes,
                    selection_family_bytes,
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
