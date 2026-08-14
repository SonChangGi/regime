"""Injectable JSON transport and bounded retry machinery."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .security import redact_text


class JsonTransport(Protocol):
    def get_json(
        self,
        url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]: ...


class HttpStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str = "HTTP request failed") -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class ProviderRequestError(RuntimeError):
    """Secret-safe terminal provider request error."""

    def __init__(self, message: str, *, attempts: int, status_code: int | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.25
    max_backoff_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff values must be non-negative")


class UrllibJsonTransport:
    """Minimal stdlib transport; no credentials are retained on the instance."""

    def __init__(self, *, user_agent: str = "regime-lab/1") -> None:
        self.user_agent = user_agent

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        query = urlencode(params, doseq=True)
        request = Request(f"{url}?{query}", headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = json.load(response)
        except HTTPError as exc:
            raise HttpStatusError(exc.code, f"HTTP {exc.code}") from None
        except URLError as exc:
            raise OSError("provider connection failed") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("provider response root must be a JSON object")
        return payload


def request_json_with_retry(
    transport: JsonTransport,
    url: str,
    params: Mapping[str, Any],
    *,
    timeout: float,
    retry: RetryPolicy,
    sleeper: Callable[[float], None] = time.sleep,
    secrets: tuple[str, ...] = (),
) -> tuple[Mapping[str, Any], int]:
    """Fetch JSON, retrying only transient failures and returning attempt count."""

    for attempt in range(1, retry.max_attempts + 1):
        try:
            return transport.get_json(url, params, timeout=timeout), attempt
        except HttpStatusError as exc:
            transient = exc.status_code == 429 or 500 <= exc.status_code <= 599
            if not transient or attempt == retry.max_attempts:
                message = redact_text(exc, secrets=secrets)
                raise ProviderRequestError(
                    message,
                    attempts=attempt,
                    status_code=exc.status_code,
                ) from None
        except (TimeoutError, OSError) as exc:
            if attempt == retry.max_attempts:
                message = redact_text(exc, secrets=secrets)
                raise ProviderRequestError(message, attempts=attempt) from None
        except (TypeError, ValueError) as exc:
            message = redact_text(exc, secrets=secrets)
            raise ProviderRequestError(message, attempts=attempt) from None

        delay = min(
            retry.max_backoff_seconds,
            retry.backoff_seconds * (2 ** (attempt - 1)),
        )
        if delay:
            sleeper(delay)

    raise AssertionError("unreachable")
