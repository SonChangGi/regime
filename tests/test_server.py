from __future__ import annotations

from email.message import Message
from http.server import SimpleHTTPRequestHandler
from io import BytesIO
from pathlib import Path

import pytest

from regime_lab import server


@pytest.mark.parametrize(
    "path",
    ["/", "/index.html", "/styles.css?v=abc", "/app.js?v=abc", "/data/regime-results.json"],
)
def test_dashboard_responses_disable_browser_cache(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str]] = []
    instance = object.__new__(server.DashboardHandler)
    instance.path = path
    instance.send_header = lambda name, value: captured.append((name, value))
    monkeypatch.setattr(SimpleHTTPRequestHandler, "end_headers", lambda _self: None)

    instance.end_headers()

    assert ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0") in captured
    assert ("Pragma", "no-cache") in captured
    assert ("Expires", "0") in captured


def test_dashboard_ignores_conditional_cache_headers() -> None:
    instance = object.__new__(server.DashboardHandler)
    instance.headers = Message()
    instance.headers["If-Modified-Since"] = "Fri, 21 Aug 2026 20:00:00 GMT"
    instance.headers["If-None-Match"] = '"stale-preview"'

    instance._disable_conditional_cache()

    assert "If-Modified-Since" not in instance.headers
    assert "If-None-Match" not in instance.headers


def test_regular_json_path_accepts_only_regular_non_symlink(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    link = tmp_path / "payload-link.json"
    link.symlink_to(payload)

    assert server._regular_json_path(payload, label="payload") == payload.resolve()
    assert server._regular_json_path(None, label="payload") is None
    with pytest.raises(FileNotFoundError, match="not found or unsafe"):
        server._regular_json_path(link, label="payload")
    with pytest.raises(FileNotFoundError, match="not found or unsafe"):
        server._regular_json_path(tmp_path / "missing.json", label="payload")


def test_serve_dashboard_freezes_selected_payloads_before_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    payload = tmp_path / "regime-results.json"
    comparison = tmp_path / "v5-vs-v4-comparison.json"
    selection_family = tmp_path / "selection-family-audit.json"
    original_payload = b'{"generation":"validated"}'
    original_comparison = b'{"comparison":"validated"}'
    original_selection_family = b'{"selection":"validated"}'
    payload.write_bytes(original_payload)
    comparison.write_bytes(original_comparison)
    selection_family.write_bytes(original_selection_family)
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, address: tuple[str, int], handler: object) -> None:
            captured["address"] = address
            captured["handler"] = handler

        def serve_forever(self) -> None:
            captured["served"] = True

        def server_close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeServer)

    server.serve_dashboard(
        web_root,
        host="127.0.0.1",
        port=9988,
        payload=payload,
        comparison=comparison,
        selection_family=selection_family,
    )

    handler = captured["handler"]
    assert captured["address"] == ("127.0.0.1", 9988)
    assert captured["served"] is True
    assert captured["closed"] is True
    payload.write_bytes(b'{"generation":"replaced"}')
    comparison.write_bytes(b'{"comparison":"replaced"}')
    selection_family.write_bytes(b'{"selection":"replaced"}')

    instance = object.__new__(handler.func)
    instance.send_response = lambda _status: None
    instance.send_header = lambda _name, _value: None
    instance.end_headers = lambda: None
    instance.path = "/data/regime-results.json"
    instance.wfile = BytesIO()
    assert instance._serve_override_json(include_body=True) is True
    assert instance.wfile.getvalue() == original_payload
    instance.path = "/data/v5-vs-v4-comparison.json"
    instance.wfile = BytesIO()
    assert instance._serve_override_json(include_body=True) is True
    assert instance.wfile.getvalue() == original_comparison
    instance.path = "/data/selection-family-audit.json"
    instance.wfile = BytesIO()
    assert instance._serve_override_json(include_body=True) is True
    assert instance.wfile.getvalue() == original_selection_family


def test_serve_dashboard_accepts_prevalidated_frozen_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, _address: tuple[str, int], handler: object) -> None:
            captured["handler"] = handler

        def serve_forever(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeServer)
    payload = b'{"generation":"validated"}'
    comparison = b'{"comparison":"validated"}'

    server.serve_dashboard(
        web_root,
        payload_bytes=payload,
        comparison_bytes=comparison,
    )

    handler = captured["handler"]
    assert handler.func.payload_bytes is payload
    assert handler.func.comparison_bytes is comparison
    assert handler.func.json_overrides_enabled is True


def test_serve_dashboard_projects_selected_v5_payload_for_core_first_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, _address: tuple[str, int], handler: object) -> None:
            captured["handler"] = handler

        def serve_forever(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(
        server,
        "build_dashboard_split",
        lambda _document, *, payload_raw: (b'{"core":true}', b'{"research":true}'),
    )
    payload = (
        b'{"meta":{"generation_id":"candidate"},"research":{},'
        b'"padding":"selected-local-preview"}'
    )

    server.serve_dashboard(web_root, payload_bytes=payload)

    handler = captured["handler"]
    instance = object.__new__(handler.func)
    instance.send_response = lambda _status: None
    instance.send_header = lambda _name, _value: None
    instance.end_headers = lambda: None
    instance.wfile = BytesIO()
    instance.path = "/data/regime-core.json"
    assert instance._serve_override_json(include_body=True) is True
    assert instance.wfile.getvalue() == b'{"core":true}'
    instance.wfile = BytesIO()
    instance.path = "/data/regime-research.json"
    assert instance._serve_override_json(include_body=True) is True
    assert instance.wfile.getvalue() == b'{"research":true}'
