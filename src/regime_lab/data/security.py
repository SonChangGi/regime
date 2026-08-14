"""Small secret-safety helpers used at logging and persistence boundaries."""

from __future__ import annotations

import re
from typing import Any, Mapping


REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}
_QUERY_SECRET = re.compile(
    r"(?i)(api[_-]?key|apikey|token|password|secret)=([^&\s]+)"
)


def is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith("_token")


def sanitize_mapping(value: Any) -> Any:
    """Recursively replace credential-like values before persistence/logging."""

    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_sensitive_key(key) else sanitize_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(sanitize_mapping(item) for item in value)
    if isinstance(value, list):
        return [sanitize_mapping(item) for item in value]
    if isinstance(value, str):
        return _QUERY_SECRET.sub(
            lambda match: f"{match.group(1)}={REDACTED}",
            value,
        )
    return value


def redact_text(text: object, *, secrets: tuple[str, ...] = ()) -> str:
    """Redact known secrets and credential-looking query parameters."""

    safe = _QUERY_SECRET.sub(lambda match: f"{match.group(1)}={REDACTED}", str(text))
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, REDACTED)
    return safe
