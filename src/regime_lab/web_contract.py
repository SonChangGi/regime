"""Deterministic browser contract generated from the operating contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from regime_lab.config import project_root
from regime_lab.operating_contract import canonical_sha256, load_operating_contract


BROWSER_CONTRACT_SCHEMA = "regime-browser-contract/1"
GENERATED_BROWSER_CONTRACT_PATH = Path("web/operating-contract.generated.js")


class BrowserContractError(RuntimeError):
    """Raised when the generated browser contract is missing or stale."""


def browser_contract_document(
    operating_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = (
        dict(operating_document)
        if operating_document is not None
        else dict(load_operating_contract().document)
    )
    return {
        "schema_version": BROWSER_CONTRACT_SCHEMA,
        "operating_contract_schema": source["schema_version"],
        "operating_contract_version": source["contract_version"],
        "operating_contract_canonical_sha256": canonical_sha256(source),
        "state_order": source["state_order"],
        "state_meta": source["state_meta"],
        "label": source["label"],
        "forecast": {
            key: source["forecast"][key]
            for key in (
                "horizon_weeks",
                "official_gap_weeks",
                "transition_horizons_weeks",
                "decision_timezone",
                "control_decision_time",
            )
        },
        "models": source["models"],
        "selection_policy": source["selection_policy"],
        "lifecycle": source["lifecycle"],
    }


def render_browser_contract_javascript(
    operating_document: Mapping[str, Any] | None = None,
) -> bytes:
    document = browser_contract_document(operating_document)
    serialized = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    source = (
        "/* Generated from config/operating-contract.json. Do not edit. */\n"
        "(function (root) {\n"
        "  \"use strict\";\n"
        "  function deepFreeze(value) {\n"
        "    if (!value || typeof value !== \"object\" || Object.isFrozen(value)) return value;\n"
        "    Object.values(value).forEach(deepFreeze);\n"
        "    return Object.freeze(value);\n"
        "  }\n"
        f"  const contract = deepFreeze({serialized});\n"
        "  Object.defineProperty(root, \"REGIME_OPERATING_CONTRACT\", {\n"
        "    configurable: false, enumerable: true, value: contract, writable: false\n"
        "  });\n"
        "})(typeof globalThis === \"object\" ? globalThis : window);\n"
    )
    return source.encode("utf-8")


def generated_browser_contract_sha256(
    operating_document: Mapping[str, Any] | None = None,
) -> str:
    return hashlib.sha256(
        render_browser_contract_javascript(operating_document)
    ).hexdigest()


def validate_generated_browser_contract(
    path: str | Path | None = None,
    *,
    operating_document: Mapping[str, Any] | None = None,
) -> bytes:
    target = (
        Path(path)
        if path is not None
        else project_root() / GENERATED_BROWSER_CONTRACT_PATH
    )
    if target.is_symlink() or not target.is_file():
        raise BrowserContractError(f"generated browser contract is missing: {target}")
    actual = target.read_bytes()
    expected = render_browser_contract_javascript(operating_document)
    if actual != expected:
        raise BrowserContractError(
            "generated browser contract differs from config/operating-contract.json"
        )
    return actual


__all__ = [
    "BROWSER_CONTRACT_SCHEMA",
    "BrowserContractError",
    "GENERATED_BROWSER_CONTRACT_PATH",
    "browser_contract_document",
    "generated_browser_contract_sha256",
    "render_browser_contract_javascript",
    "validate_generated_browser_contract",
]
