"""Typed, secret-free project configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config" / "series.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_config_path()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON config: {config_path}: {exc}") from exc

    for section in ("cutoff", "alfred", "alpha_vantage", "model"):
        if section not in config or not isinstance(config[section], dict):
            raise ConfigurationError(f"config section is required: {section}")
    if config["model"].get("horizon_weeks") != 1:
        raise ConfigurationError(
            "the primary next-state forecast must remain a one-week horizon"
        )
    transition_horizons = config["model"].get(
        "transition_horizons_weeks", [1]
    )
    if transition_horizons != [1, 4, 13]:
        raise ConfigurationError(
            "transition_horizons_weeks must be exactly [1, 4, 13]"
        )
    if config["model"].get("state_order") != [
        "risk_on",
        "transition",
        "risk_off",
    ]:
        raise ConfigurationError(
            "model.state_order must remain risk_on, transition, risk_off"
        )
    if len(config["alpha_vantage"].get("symbols", [])) > int(
        config["alpha_vantage"].get("daily_request_cap", 0)
    ):
        raise ConfigurationError("Alpha Vantage symbols exceed the configured daily cap")
    if not config["alfred"].get("series"):
        raise ConfigurationError("at least one ALFRED series is required")
    return config
