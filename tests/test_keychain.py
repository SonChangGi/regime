from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from regime_lab.keychain import (
    KeychainError,
    load_provider_secrets,
    provider_environment_from_keychain,
    provider_secrets_from_keychain,
    verify_provider_keychain_access,
)


def test_keychain_environment_is_restored() -> None:
    os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
    os.environ["FRED_API_KEY"] = "old"
    with patch(
        "regime_lab.keychain.load_provider_secrets",
        return_value={"FRED_API_KEY": "secret-a", "ALPHA_VANTAGE_API_KEY": "secret-b"},
    ):
        with provider_environment_from_keychain(rights_acknowledged=True):
            assert os.environ["FRED_API_KEY"] == "secret-a"
            assert os.environ["ALPHA_VANTAGE_API_KEY"] == "secret-b"
            assert os.environ["ALFRED_ML_RIGHTS_ACK"] == "1"
    assert os.environ["FRED_API_KEY"] == "old"
    assert "ALPHA_VANTAGE_API_KEY" not in os.environ
    assert "ALFRED_ML_RIGHTS_ACK" not in os.environ


def test_keychain_preflight_discards_secrets_and_supports_service_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        service = command[command.index("-s") + 1]
        has_account = "-a" in command
        returncode = 51 if service == "regime-fred-api-key" and has_account else 0
        return SimpleNamespace(returncode=returncode, stdout="secret\n")

    monkeypatch.setattr("regime_lab.keychain.subprocess.run", run)

    verify_provider_keychain_access(account="login-account")

    assert len(calls) == 3
    assert all(command[-1] == "-w" for command, _kwargs in calls)
    assert all(kwargs["stdin"] is subprocess.DEVNULL for _command, kwargs in calls)
    assert all(kwargs["capture_output"] is True for _command, kwargs in calls)
    assert all(kwargs["timeout"] == 10 for _command, kwargs in calls)


def test_keychain_preflight_reports_unreadable_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "regime_lab.keychain.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=51, stdout=""),
    )

    with pytest.raises(KeychainError, match="regime-fred-api-key"):
        verify_provider_keychain_access(account="login-account")


def test_keychain_preflight_rejects_empty_value_and_hides_reader_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "regime_lab.keychain.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="\n"),
    )
    with pytest.raises(KeychainError, match="empty"):
        verify_provider_keychain_access(account="")

    monkeypatch.setattr(
        "regime_lab.keychain.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["security"], 10)
        ),
    )
    with pytest.raises(KeychainError, match="cannot be read"):
        verify_provider_keychain_access(account="")


def test_loaded_keychain_secret_references_are_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = {"FRED_API_KEY": "secret-a", "ALPHA_VANTAGE_API_KEY": "secret-b"}
    monkeypatch.setattr(
        "regime_lab.keychain.load_provider_secrets",
        lambda **_kwargs: loaded,
    )

    with provider_secrets_from_keychain() as secrets:
        assert secrets is loaded
        assert all(secrets.values())

    assert loaded == {"FRED_API_KEY": "", "ALPHA_VANTAGE_API_KEY": ""}


def test_partial_keychain_load_clears_already_read_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[dict[str, str]] = []

    def read(service: str, **_kwargs: object) -> str:
        if service == "regime-fred-api-key":
            return "secret-a"
        raise KeychainError("second provider is unavailable")

    def clear(values: dict[str, str]) -> None:
        cleared.append(dict(values))
        for name in values:
            values[name] = ""

    monkeypatch.setattr("regime_lab.keychain.read_keychain_secret", read)
    monkeypatch.setattr("regime_lab.keychain._clear_secret_mapping", clear)

    with pytest.raises(KeychainError, match="second provider"):
        load_provider_secrets(account="login-account")

    assert cleared == [{"FRED_API_KEY": "secret-a"}]
