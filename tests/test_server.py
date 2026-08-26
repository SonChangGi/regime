from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from regime_lab import server


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
    original_payload = b'{"generation":"validated"}'
    original_comparison = b'{"comparison":"validated"}'
    payload.write_bytes(original_payload)
    comparison.write_bytes(original_comparison)
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
    )

    handler = captured["handler"]
    assert captured["address"] == ("127.0.0.1", 9988)
    assert captured["served"] is True
    assert captured["closed"] is True
    payload.write_bytes(b'{"generation":"replaced"}')
    comparison.write_bytes(b'{"comparison":"replaced"}')

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
