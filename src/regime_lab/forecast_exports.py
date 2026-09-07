"""Deterministic, payload-bound downloads for the forecast research panels."""

from __future__ import annotations

import json
from typing import Any

from regime_lab.publication_contract import (
    PublicContractError,
    reject_raw_provider_material,
)


FORECAST_EXPORT_BLOCKS = (
    "forecast_research",
    "calibration_audit",
    "forecast_information",
)


def build_forecast_exports(payload: dict[str, Any]) -> dict[str, bytes]:
    """Export derived blocks only; never follow arbitrary artifact paths."""
    research = payload.get("research", {})
    files = {}
    for name in FORECAST_EXPORT_BLOCKS:
        if name not in research:
            continue
        block = research[name]
        if not isinstance(block, dict):
            raise PublicContractError(f"{name} export must be an object")
        destination = f"data/{name}.json"
        artifacts = block.get("artifacts", [])
        if artifacts != [{"label": "전체 결과 JSON", "url": f"./{destination}"}]:
            raise PublicContractError(f"{name} requires its payload-bound download link")
        reject_raw_provider_material(block)
        files[destination] = (
            json.dumps(block, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    return files


def validate_forecast_exports(
    files: dict[str, bytes], *, payload: dict[str, Any]
) -> None:
    if files != build_forecast_exports(payload):
        raise PublicContractError("forecast downloads differ from the reviewed payload")
