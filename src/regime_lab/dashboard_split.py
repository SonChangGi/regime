"""Hash-bound core/research projection for the public dashboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from regime_lab.publication_contract import PublicContractError


CORE_PAYLOAD_DESTINATION = "data/regime-core.json"
RESEARCH_SIDECAR_DESTINATION = "data/regime-research.json"
CORE_PAYLOAD_SCHEMA_VERSION = "regime-dashboard-core/1"
RESEARCH_SIDECAR_SCHEMA_VERSION = "regime-dashboard-research/1"
MAX_CORE_PAYLOAD_BYTES = 3_500_000
MAX_CORE_TO_SOURCE_RATIO = 0.85


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_public_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def build_dashboard_split(
    payload: dict[str, Any],
    *,
    payload_raw: bytes,
) -> tuple[bytes, bytes]:
    """Build a core-first envelope and exact-generation research sidecar."""

    meta = payload.get("meta")
    generation_id = meta.get("generation_id") if isinstance(meta, dict) else None
    if not isinstance(generation_id, str) or not generation_id:
        raise PublicContractError(
            "V5 core/research split requires a non-empty meta.generation_id"
        )
    research = payload.get("research")
    if not isinstance(research, dict):
        raise PublicContractError("V5 core/research split requires a research object")

    source_payload_sha256 = _sha256(payload_raw)
    research_document = {
        "schema_version": RESEARCH_SIDECAR_SCHEMA_VERSION,
        "generation_id": generation_id,
        "source_payload_sha256": source_payload_sha256,
        "research": research,
    }
    research_raw = _canonical_public_json(research_document)
    core_document = {
        "schema_version": CORE_PAYLOAD_SCHEMA_VERSION,
        "generation_id": generation_id,
        "source_payload_sha256": source_payload_sha256,
        "payload": {key: value for key, value in payload.items() if key != "research"},
        "research_sidecar": {
            "path": Path(RESEARCH_SIDECAR_DESTINATION).name,
            "sha256": _sha256(research_raw),
        },
    }
    core_raw = _canonical_public_json(core_document)
    if len(core_raw) >= len(payload_raw):
        raise PublicContractError(
            "dashboard core projection must be smaller than the full payload"
        )
    if len(core_raw) > MAX_CORE_PAYLOAD_BYTES:
        raise PublicContractError("dashboard core projection exceeds its byte budget")
    if len(core_raw) / len(payload_raw) > MAX_CORE_TO_SOURCE_RATIO:
        raise PublicContractError(
            "dashboard core projection does not materially reduce initial bytes"
        )
    validate_dashboard_split(
        core_document,
        research_document,
        payload=payload,
        payload_raw=payload_raw,
        research_raw=research_raw,
    )
    return core_raw, research_raw


def validate_dashboard_split(
    core_document: dict[str, Any],
    research_document: dict[str, Any],
    *,
    payload: dict[str, Any],
    payload_raw: bytes,
    research_raw: bytes,
) -> None:
    """Validate projection identity, source binding, and sidecar integrity."""

    expected_core_keys = {
        "schema_version",
        "generation_id",
        "source_payload_sha256",
        "payload",
        "research_sidecar",
    }
    expected_research_keys = {
        "schema_version",
        "generation_id",
        "source_payload_sha256",
        "research",
    }
    if set(core_document) != expected_core_keys:
        raise PublicContractError("dashboard core envelope keys are not exact")
    if set(research_document) != expected_research_keys:
        raise PublicContractError("dashboard research sidecar keys are not exact")
    generation_id = payload.get("meta", {}).get("generation_id")
    source_payload_sha256 = _sha256(payload_raw)
    if core_document.get("schema_version") != CORE_PAYLOAD_SCHEMA_VERSION:
        raise PublicContractError("dashboard core envelope schema is invalid")
    if research_document.get("schema_version") != RESEARCH_SIDECAR_SCHEMA_VERSION:
        raise PublicContractError("dashboard research sidecar schema is invalid")
    if (
        core_document.get("generation_id") != generation_id
        or research_document.get("generation_id") != generation_id
    ):
        raise PublicContractError("dashboard split generation_id mismatch")
    if (
        core_document.get("source_payload_sha256") != source_payload_sha256
        or research_document.get("source_payload_sha256") != source_payload_sha256
    ):
        raise PublicContractError("dashboard split source payload hash mismatch")
    expected_core_payload = {
        key: value for key, value in payload.items() if key != "research"
    }
    if core_document.get("payload") != expected_core_payload:
        raise PublicContractError(
            "dashboard core payload differs from source projection"
        )
    if research_document.get("research") != payload.get("research"):
        raise PublicContractError(
            "dashboard research sidecar differs from source payload"
        )
    binding = core_document.get("research_sidecar")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise PublicContractError("dashboard research sidecar binding is invalid")
    if binding.get("path") != Path(RESEARCH_SIDECAR_DESTINATION).name:
        raise PublicContractError("dashboard research sidecar path is invalid")
    if binding.get("sha256") != _sha256(research_raw):
        raise PublicContractError("dashboard research sidecar hash mismatch")


__all__ = [
    "CORE_PAYLOAD_DESTINATION",
    "CORE_PAYLOAD_SCHEMA_VERSION",
    "MAX_CORE_PAYLOAD_BYTES",
    "MAX_CORE_TO_SOURCE_RATIO",
    "RESEARCH_SIDECAR_DESTINATION",
    "RESEARCH_SIDECAR_SCHEMA_VERSION",
    "build_dashboard_split",
    "validate_dashboard_split",
]
