"""Bounded provider smoke checks used before a full collection run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import sys
from zoneinfo import ZoneInfo

from regime_lab.data import (
    AlfredClient,
    AlfredConfig,
    AlphaVantageClient,
    AlphaVantageConfig,
    DailyRequestBudget,
)
from regime_lab.keychain import provider_environment_from_keychain
from regime_lab.config import project_root
from regime_lab.provider_rights import ProviderRightsError, verify_provider_rights


@dataclass(frozen=True)
class SmokeResult:
    provider: str
    health: str
    records: int
    requests: int
    issues: tuple[str, ...]


def run_alfred_smoke() -> SmokeResult:
    now = datetime.now(timezone.utc)
    realtime_end = now.astimezone(ZoneInfo("America/New_York")).date()
    realtime_start = realtime_end - timedelta(days=45)
    alfred = AlfredClient(AlfredConfig.from_env())
    alfred_result = alfred.fetch_realtime_observations(
        ["DGS10"],
        realtime_start=realtime_start,
        realtime_end=realtime_end,
        cutoff=now,
        observation_start=realtime_start,
        observation_end=realtime_end,
    )
    return SmokeResult(
        provider="alfred",
        health=alfred_result.health.value,
        records=len(alfred_result.records),
        requests=alfred_result.requests_made,
        issues=alfred_result.issues,
    )


def probe_alfred_schema() -> dict[str, object]:
    """Return response shape only; request credentials are never included."""

    config = AlfredConfig.from_env(vintage_batch_size=1)
    if not config.api_key:
        raise RuntimeError("FRED_API_KEY is unavailable")
    client = AlfredClient(config)
    payload = client.transport.get_json(
        config.base_url,
        {
            "series_id": "DGS10",
            "api_key": config.api_key,
            "file_type": "json",
            "output_type": 2,
            "vintage_dates": "2026-08-07",
            "observation_start": "2026-07-01",
            "observation_end": "2026-08-07",
            "limit": 100,
            "offset": 0,
        },
        timeout=config.timeout_seconds,
    )
    observations = payload.get("observations")
    sample = observations[0] if isinstance(observations, list) and observations else {}
    return {
        "top_level_keys": sorted(str(key) for key in payload),
        "count": payload.get("count"),
        "observations_type": type(observations).__name__,
        "observations_length": len(observations) if isinstance(observations, list) else None,
        "first_row": dict(sample) if isinstance(sample, dict) else type(sample).__name__,
    }


def run_alpha_smoke(*, budget_database: str | None = None) -> SmokeResult:
    now = datetime.now(timezone.utc)
    alpha = AlphaVantageClient(
        AlphaVantageConfig.from_env(),
        budget=DailyRequestBudget(limit=25, database_path=budget_database),
    )
    alpha_result = alpha.fetch_weekly_adjusted(
        ["SPY"],
        cutoff=now,
        fields=("adjusted_close", "volume"),
        observation_start=date(2026, 7, 1),
    )
    return SmokeResult(
        provider="alpha_vantage",
        health=alpha_result.health.value,
        records=len(alpha_result.records),
        requests=alpha_result.requests_made,
        issues=alpha_result.issues,
    )


def run_provider_smoke(*, budget_database: str | None = None) -> tuple[SmokeResult, ...]:
    return run_alfred_smoke(), run_alpha_smoke(budget_database=budget_database)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m regime_lab.smoke")
    parser.add_argument(
        "provider",
        choices=("all", "alfred", "alfred_schema", "alpha_vantage"),
        default="all",
        nargs="?",
    )
    parser.add_argument("--budget-database")
    parser.add_argument("--alfred-rights-confirmed", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    selected = args.provider
    if selected in {"all", "alfred", "alfred_schema"} and not args.alfred_rights_confirmed:
        raise SystemExit(
            "ALFRED smoke requires --alfred-rights-confirmed after verifying usage rights"
        )
    provider_ids = {
        "alfred": ("fred_alfred",),
        "alfred_schema": ("fred_alfred",),
        "alpha_vantage": ("alpha_vantage",),
        "all": ("fred_alfred", "alpha_vantage"),
    }[selected]
    try:
        verify_provider_rights(
            provider_ids,
            policy_path=project_root() / "config/provider_rights.json",
            capabilities=("collection",),
        )
    except ProviderRightsError as exc:
        raise SystemExit(str(exc)) from exc
    with provider_environment_from_keychain(
        rights_acknowledged=args.alfred_rights_confirmed
    ):
        if selected == "alfred_schema":
            print(json.dumps(probe_alfred_schema(), ensure_ascii=False, indent=2))
            return 0
        if selected == "alfred":
            results = (run_alfred_smoke(),)
        elif selected == "alpha_vantage":
            results = (run_alpha_smoke(budget_database=args.budget_database),)
        else:
            results = run_provider_smoke(budget_database=args.budget_database)
    print(
        json.dumps(
            [result.__dict__ for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(result.health == "ok" and result.records for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
