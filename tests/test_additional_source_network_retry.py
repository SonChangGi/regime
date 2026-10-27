from __future__ import annotations

from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import time

import pytest
import requests
from urllib3.exceptions import MaxRetryError

from regime_lab.research import additional_sources as sources


SOURCE_ID = "vix"
SOURCE_URL = "https://example.invalid/research.csv"
RAW = b"DATE,CLOSE\n2026-10-23,19.5\n"
SENSITIVE = "private-transport-detail-must-not-become-the-error-message"


def response(status: int = 200, raw: object = RAW) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result.url = SOURCE_URL
    result._content = raw
    return result


@pytest.fixture(autouse=True)
def retry_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    monkeypatch.setattr(sources, "SOURCES", {SOURCE_ID: ("Synthetic public source", SOURCE_URL)})

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("test must provide a transport; real network access is forbidden")

    monkeypatch.setattr(sources.requests, "get", unexpected)
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


def transport(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> list[str]:
    pending = list(outcomes)
    calls: list[str] = []

    def get(url: str, *, timeout: tuple[int, int], headers: dict[str, str]) -> requests.Response:
        assert url == SOURCE_URL
        assert timeout == (15, 90)
        assert headers == {"User-Agent": "RegimePrivateResearch/1.0"}
        calls.append(url)
        assert pending, "unexpected request after the supplied transport outcomes"
        outcome = pending.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, requests.Response)
        return outcome

    monkeypatch.setattr(sources.requests, "get", get)
    return calls


def identity(path: Path) -> tuple[bytes, int, int]:
    stat = path.stat()
    return path.read_bytes(), stat.st_mtime_ns, stat.st_ino


def seed_cache(directory: Path, raw: bytes = RAW) -> tuple[Path, Path, dict[str, object]]:
    directory.mkdir(parents=True)
    digest = hashlib.sha256(raw).hexdigest()
    cached = directory / f"{SOURCE_ID}-{digest}.csv"
    cached.write_bytes(raw)
    record: dict[str, object] = {
        "name": "Synthetic public source", "url": SOURCE_URL,
        "sha256": digest, "bytes": len(raw), "file": cached.name,
        "fetched_at": "2026-10-09T01:00:00+00:00",
        "first_seen_at": "2026-10-02T01:00:00+00:00",
        "cost": "public_download_no_account", "availability_basis": "first_seen_snapshot",
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "regime-additional-sources/1",
                                   "sources": {SOURCE_ID: record}}, indent=2) + "\n")
    return manifest, cached, record


def transient(kind: str) -> object:
    if kind == "connection":
        return requests.exceptions.ConnectionError(SENSITIVE)
    if kind == "timeout":
        return requests.exceptions.Timeout(SENSITIVE)
    if kind == "connect_timeout":
        return requests.exceptions.ConnectTimeout(SENSITIVE)
    if kind == "read_timeout":
        return requests.exceptions.ReadTimeout(SENSITIVE)
    return response(int(kind))


@pytest.mark.parametrize("kind", ["connection", "timeout", "connect_timeout", "read_timeout",
                                  "408", "429", "500", "503", "599"])
def test_transient_failure_exhaustion_is_bounded_and_does_not_publish_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float], kind: str,
) -> None:
    directory = tmp_path / "sources"
    calls = transport(monkeypatch, [transient(kind) for _ in range(3)])

    with pytest.raises(sources.AdditionalSourceUnavailable) as caught:
        sources.collect_sources(directory, refresh=True)

    assert caught.value.source_id == SOURCE_ID
    assert caught.value.error_code == "additional_source_unavailable"
    assert SENSITIVE not in str(caught.value) and SOURCE_URL not in str(caught.value)
    assert calls == [SOURCE_URL] * 3
    assert retry_delays == [1, 2]
    assert list(directory.iterdir()) == []


def test_fresh_success_after_retries_publishes_only_the_successful_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float],
) -> None:
    directory = tmp_path / "sources"
    calls = transport(monkeypatch, [transient("connection"), response(429), response()])
    started = datetime.now(timezone.utc)

    manifest = sources.collect_sources(directory, refresh=True)

    ended = datetime.now(timezone.utc)
    record = manifest["sources"][SOURCE_ID]
    assert calls == [SOURCE_URL] * 3 and retry_delays == [1, 2]
    assert record["sha256"] == hashlib.sha256(RAW).hexdigest()
    assert record["bytes"] == len(RAW)
    assert record["first_seen_at"] == record["fetched_at"]
    assert started <= datetime.fromisoformat(record["first_seen_at"]) <= ended
    assert sources.read_source(directory, SOURCE_ID) == RAW
    assert json.loads((directory / "manifest.json").read_text()) == manifest
    assert {path.name for path in directory.iterdir()} == {"manifest.json", record["file"]}


def test_failed_refresh_preserves_old_manifest_and_cache_without_stale_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float],
) -> None:
    directory = tmp_path / "sources"
    manifest, cached, _ = seed_cache(directory)
    before = identity(manifest), identity(cached)
    calls = transport(monkeypatch, [transient("timeout") for _ in range(3)])

    with pytest.raises(sources.AdditionalSourceUnavailable):
        sources.collect_sources(directory, refresh=True)

    assert calls == [SOURCE_URL] * 3 and retry_delays == [1, 2]
    assert (identity(manifest), identity(cached)) == before
    assert {path.name for path in directory.iterdir()} == {manifest.name, cached.name}


def test_same_snapshot_after_retry_retains_first_seen_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float],
) -> None:
    directory = tmp_path / "sources"
    _, cached, previous = seed_cache(directory)
    before = identity(cached)
    calls = transport(monkeypatch, [response(503), response()])

    manifest = sources.collect_sources(directory, refresh=True)

    record = manifest["sources"][SOURCE_ID]
    assert calls == [SOURCE_URL] * 2 and retry_delays == [1]
    assert record["sha256"] == previous["sha256"]
    assert record["first_seen_at"] == previous["first_seen_at"]
    assert record["fetched_at"] != previous["fetched_at"]
    assert identity(cached) == before


def test_new_snapshot_after_retry_gets_new_first_seen_and_keeps_prior_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float],
) -> None:
    directory = tmp_path / "sources"
    _, cached, previous = seed_cache(directory)
    before = identity(cached)
    newer = RAW + b"2026-10-26,20.1\n"
    calls = transport(monkeypatch, [transient("connection"), response(raw=newer)])

    manifest = sources.collect_sources(directory, refresh=True)

    record = manifest["sources"][SOURCE_ID]
    assert calls == [SOURCE_URL] * 2 and retry_delays == [1]
    assert record["sha256"] == hashlib.sha256(newer).hexdigest()
    assert record["first_seen_at"] == record["fetched_at"]
    assert record["first_seen_at"] != previous["first_seen_at"]
    assert sources.read_source(directory, SOURCE_ID) == newer
    assert identity(cached) == before


@pytest.mark.parametrize("status", [400, 401, 403, 404, 407, 422, 499])
def test_nonretryable_http_errors_preserve_the_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float], status: int,
) -> None:
    directory = tmp_path / "sources"
    manifest, cached, _ = seed_cache(directory)
    before = identity(manifest), identity(cached)
    calls = transport(monkeypatch, [response(status), response()])

    with pytest.raises(requests.exceptions.HTTPError):
        sources.collect_sources(directory, refresh=True)

    assert calls == [SOURCE_URL] and retry_delays == []
    assert (identity(manifest), identity(cached)) == before


@pytest.mark.parametrize("error", [requests.exceptions.SSLError,
                                   requests.exceptions.InvalidURL, requests.exceptions.RequestException])
def test_ssl_and_other_nonretryable_transport_errors_are_not_reclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float],
    error: type[requests.exceptions.RequestException],
) -> None:
    directory = tmp_path / "sources"
    failure = error(SENSITIVE)
    calls = transport(monkeypatch, [failure, response()])

    with pytest.raises(error) as caught:
        sources.collect_sources(directory, refresh=True)

    assert caught.value is failure
    assert calls == [SOURCE_URL] and retry_delays == []
    assert list(directory.iterdir()) == []


def test_proxy_denial_is_not_retried_and_preserves_the_existing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float],
) -> None:
    directory = tmp_path / "sources"
    manifest, cached, _ = seed_cache(directory)
    before = identity(manifest), identity(cached)
    failure = requests.exceptions.ProxyError(SENSITIVE)
    calls = transport(monkeypatch, [failure, response()])

    with pytest.raises(requests.exceptions.ProxyError) as caught:
        sources.collect_sources(directory, refresh=True)

    assert caught.value is failure
    assert calls == [SOURCE_URL] and retry_delays == []
    assert (identity(manifest), identity(cached)) == before
    assert {path.name for path in directory.iterdir()} == {manifest.name, cached.name}


class AccessDeniedOS(OSError):
    """Exercise an errno-bearing OSError independently of PermissionError."""


def denial(kind: str) -> BaseException:
    if kind == "permission":
        return PermissionError(SENSITIVE)
    if kind in {"eacces", "eperm"}:
        return AccessDeniedOS(errno.EACCES if kind == "eacces" else errno.EPERM, SENSITIVE)
    if kind == "ssl":
        return requests.exceptions.SSLError(SENSITIVE)
    return requests.exceptions.ProxyError(SENSITIVE)


@pytest.mark.parametrize("kind", ["permission", "eacces", "eperm", "ssl", "proxy"])
@pytest.mark.parametrize("link", ["cause", "context", "urllib3_reason", "args"])
def test_wrapped_transport_denials_are_not_retried_or_reclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float], kind: str, link: str,
) -> None:
    directory = tmp_path / "sources"
    manifest, cached, _ = seed_cache(directory)
    before = identity(manifest), identity(cached)
    blocked = denial(kind)
    failure = requests.exceptions.ConnectionError(SENSITIVE)
    if link == "cause":
        failure.__cause__ = blocked
    elif link == "context":
        failure.__context__ = blocked
    elif link == "urllib3_reason":
        failure = requests.exceptions.ConnectionError(MaxRetryError(None, SOURCE_URL, reason=blocked))
    else:
        failure.args = (SENSITIVE, [("nested transport denial", blocked)])
    calls = transport(monkeypatch, [failure, response()])

    with pytest.raises(requests.exceptions.ConnectionError) as caught:
        sources.collect_sources(directory, refresh=True)

    assert caught.value is failure
    assert calls == [SOURCE_URL] and retry_delays == []
    assert (identity(manifest), identity(cached)) == before
    assert {path.name for path in directory.iterdir()} == {manifest.name, cached.name}


class OversizedBytes(bytes):
    def __len__(self) -> int:
        return 200_000_001


@pytest.mark.parametrize("raw", [b"", OversizedBytes(b"x"), "not bytes"])
def test_invalid_snapshot_bytes_do_not_trigger_network_retries_or_manifest_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_delays: list[float], raw: object,
) -> None:
    directory = tmp_path / "sources"
    manifest, cached, _ = seed_cache(directory)
    before = identity(manifest), identity(cached)
    calls = transport(monkeypatch, [response(raw=raw), response()])

    with pytest.raises((ValueError, TypeError)):
        sources.collect_sources(directory, refresh=True)

    assert calls == [SOURCE_URL] and retry_delays == []
    assert (identity(manifest), identity(cached)) == before
    assert {path.name for path in directory.iterdir()} == {manifest.name, cached.name}


def test_corrupt_cached_snapshot_is_not_retried_or_replaced_from_the_network(
    tmp_path: Path, retry_delays: list[float],
) -> None:
    directory = tmp_path / "sources"
    manifest, cached, _ = seed_cache(directory)
    cached.write_bytes(b"corrupt cache bytes")
    before = identity(manifest), identity(cached)

    with pytest.raises(ValueError, match="cached snapshot checksum mismatch"):
        sources.collect_sources(directory)

    assert retry_delays == []
    assert (identity(manifest), identity(cached)) == before
    assert {path.name for path in directory.iterdir()} == {manifest.name, cached.name}
