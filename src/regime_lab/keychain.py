"""macOS Keychain bridge that never prints or persists provider secrets."""

from __future__ import annotations

from contextlib import contextmanager
import getpass
import os
import subprocess
from typing import Iterator, Mapping


KEYCHAIN_SERVICES: Mapping[str, str] = {
    "FRED_API_KEY": "regime-fred-api-key",
    "ALPHA_VANTAGE_API_KEY": "regime-alpha-vantage-api-key",
}


class KeychainError(RuntimeError):
    pass


def read_keychain_secret(service: str, *, account: str | None = None) -> str:
    command = ["security", "find-generic-password"]
    if account:
        command.extend(["-a", account])
    command.extend(["-s", service, "-w"])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise KeychainError(f"Keychain item is unavailable: service={service}")
    secret = completed.stdout.rstrip("\r\n")
    if not secret:
        raise KeychainError(f"Keychain item is empty: service={service}")
    return secret


def load_provider_secrets(*, account: str | None = None) -> dict[str, str]:
    # Try the explicit account first, then service-only lookup because Keychain
    # Access can assign a different account label without changing the secret.
    selected_account = getpass.getuser() if account is None else account
    values: dict[str, str] = {}
    for env_name, service in KEYCHAIN_SERVICES.items():
        try:
            values[env_name] = read_keychain_secret(service, account=selected_account)
        except KeychainError:
            values[env_name] = read_keychain_secret(service)
    return values


@contextmanager
def provider_environment_from_keychain(
    *,
    rights_acknowledged: bool,
    account: str | None = None,
) -> Iterator[None]:
    """Temporarily expose keys to provider constructors within this process."""

    secrets = load_provider_secrets(account=account)
    updates = dict(secrets)
    updates["ALFRED_ML_RIGHTS_ACK"] = "1" if rights_acknowledged else "0"
    previous = {name: os.environ.get(name) for name in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
        # Do not leave extra live references longer than the execution boundary.
        for name in list(secrets):
            secrets[name] = ""
