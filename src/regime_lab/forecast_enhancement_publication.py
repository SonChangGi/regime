"""Generation-bound optional forecast comparison sidecars for every packager."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1, reviewed_candidate_payload
from regime_lab.publication_contract import PublicContractError, reject_raw_provider_material

FILENAME = "forecast-enhancements.json"
STATES_FILENAME = "forecast-enhancement-states.pkl"
STATES_MANIFEST_FILENAME = "forecast-enhancement-input-manifest.json"
DESTINATION = "data/" + FILENAME
DECLARATION = "forecast_enhancement_publication"
BINDING_SCHEMA = "regime-forecast-enhancement-binding/1"
POLICY_MODES = ("auto", "required", "omit")


def encode(document: Mapping) -> bytes:
    return (json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def source_contract_sha256(payload: Mapping) -> str:
    """Normalize review lifecycle and exclude the declaration's hash cycle."""
    source = reviewed_candidate_payload(payload)
    source.get("research", {}).pop(DECLARATION, None)
    return canonical_json_sha256_v1(source)


def content_sha256(document: Mapping) -> str:
    return canonical_json_sha256_v1({k: v for k, v in document.items() if k != "publication_binding"})


def validate_document(document: Mapping, payload: Mapping) -> None:
    from regime_lab.research.forecast_enhancements import validate_enhancements
    try:
        validate_enhancements(document)
        if not document.get("history") or not document.get("latest") or not isinstance(document.get("model_metrics"), list):
            raise ValueError("forecast enhancement prediction tables are missing")
        identity = lambda row: (row["model"], row["origin_date"], row["horizon_weeks"])
        history = {identity(row): row for row in document["history"]}
        if len({identity(row) for row in document["latest"]}) != len(document["latest"]):
            raise ValueError("forecast enhancement latest rows are duplicated")
        if any(history.get(identity(row)) != row or pd.Timestamp(row["origin_date"]) != pd.Timestamp(document["data_as_of"]) for row in document["latest"]):
            raise ValueError("forecast enhancement latest differs from its frozen history/cutoff")
        if document["source_generation_id"] != payload["meta"]["generation_id"]:
            raise ValueError("forecast enhancement generation differs from payload")
        if pd.Timestamp(document["data_as_of"]) != pd.Timestamp(payload["meta"]["data_as_of"]):
            raise ValueError("forecast enhancement cutoff differs from payload")
        reject_raw_provider_material(dict(document))
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicContractError(f"forecast enhancement contract: {exc}") from exc


def validate_declaration(record: Mapping, *, data_as_of: str | None = None) -> None:
    import re
    expected = {"schema_version", "status", "filename", "source_generation_id", "data_as_of", "sha256", "bytes"}
    if (not isinstance(record, Mapping) or set(record) != expected
            or record.get("schema_version") != BINDING_SCHEMA or record.get("status") != "required"
            or record.get("filename") != FILENAME or not isinstance(record.get("source_generation_id"), str)
            or not record["source_generation_id"] or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            or type(record.get("bytes")) is not int or record["bytes"] <= 0):
        raise ValueError("forecast enhancement required declaration is invalid")
    if data_as_of is not None and pd.Timestamp(record["data_as_of"]) != pd.Timestamp(data_as_of):
        raise ValueError("forecast enhancement declaration cutoff differs")


def bind_document(document: Mapping, payload: Mapping, *, source_raw: bytes | None = None) -> dict:
    """Bind a freshly generated result or a verified original research receipt.

    Importing an existing unbound file requires its original source-payload hash;
    merely replacing its generation/date fields can never make it current.
    """
    result = deepcopy(dict(document))
    validate_document(result, payload)
    if "publication_binding" in result:
        validate_binding(result, payload)
        return result
    if source_raw is not None:
        recorded = result.get("provenance", {}).get("input_sha256", {}).get("payload")
        if recorded != hashlib.sha256(source_raw).hexdigest():
            raise PublicContractError("unbound forecast enhancement lacks the exact source payload receipt")
    result["publication_binding"] = {
        "schema_version": BINDING_SCHEMA,
        "source_generation_id": payload["meta"]["generation_id"],
        "data_as_of": payload["meta"]["data_as_of"],
        "source_payload_contract_sha256": source_contract_sha256(payload),
        "content_sha256": content_sha256(result),
    }
    validate_binding(result, payload)
    return result


def declaration(document: Mapping) -> dict:
    raw = encode(document)
    return {"schema_version": BINDING_SCHEMA, "status": "required", "filename": FILENAME,
            "source_generation_id": document["source_generation_id"], "data_as_of": document["data_as_of"],
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def validate_binding(document: Mapping, payload: Mapping) -> None:
    validate_document(document, payload)
    binding = document.get("publication_binding")
    expected = {"schema_version": BINDING_SCHEMA,
                "source_generation_id": payload["meta"]["generation_id"],
                "data_as_of": payload["meta"]["data_as_of"],
                "source_payload_contract_sha256": source_contract_sha256(payload),
                "content_sha256": content_sha256(document)}
    if binding != expected:
        raise PublicContractError("forecast enhancement payload/content binding differs")
    required = payload.get("research", {}).get(DECLARATION)
    if required is not None:
        try:
            validate_declaration(required, data_as_of=payload["meta"]["data_as_of"])
        except ValueError as exc:
            raise PublicContractError(str(exc)) from exc
    if required is not None and required != declaration(document):
        raise PublicContractError("forecast enhancement differs from the generation's required declaration")


def package_sidecar(payload: Mapping, *, payload_raw: bytes, payload_path: Path,
                    path: Path | None = None, mode: str = "auto") -> tuple[dict[str, bytes], dict]:
    if mode not in POLICY_MODES:
        raise PublicContractError("unknown forecast enhancement packaging mode")
    required = payload.get("research", {}).get(DECLARATION) is not None
    if mode == "omit":
        if path is not None or required:
            raise PublicContractError("cannot omit explicitly supplied or generation-required forecast enhancements")
        return {}, {"mode": mode, "status": "omitted", "reason": "explicitly_disabled"}
    selected = path if path is not None else payload_path.with_name(FILENAME)
    if selected.is_symlink():
        raise PublicContractError("forecast enhancement must be a regular file, not a symbolic link")
    if not selected.exists():
        if path is not None or mode == "required" or required:
            raise PublicContractError("required forecast enhancement sidecar is missing")
        return {}, {"mode": mode, "status": "omitted", "reason": "not_configured_for_this_generation"}
    if not selected.is_file():
        raise PublicContractError("forecast enhancement sidecar is not a regular file")
    original = selected.read_bytes()
    try:
        document = json.loads(original)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicContractError("forecast enhancement must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise PublicContractError("forecast enhancement must be an object")
    # Preview CSV downloads are private companion artifacts. Package the complete
    # JSON evidence and record that projection instead of emitting broken links.
    if "publication_binding" not in document and document.get("provenance", {}).get("artifacts"):
        original_content_hash = content_sha256(document)
        links = document["provenance"].pop("artifacts")
        document["provenance"]["package_projection"] = {
            "source_content_sha256": original_content_hash,
            "omitted_companion_download_links": len(links),
            "evidence_download": "./" + DESTINATION,
        }
    bound = bind_document(document, payload, source_raw=payload_raw)
    raw = encode(bound)
    if selected.read_bytes() != original:
        raise PublicContractError("forecast enhancement source changed while packaging")
    return {DESTINATION: raw}, {"mode": mode, "status": "included", "destination": DESTINATION,
                                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                "binding": bound["publication_binding"]}


def validate_packaged_sidecar(files: Mapping[str, bytes], metadata: Mapping | None,
                             payload: Mapping) -> None:
    required = payload.get("research", {}).get(DECLARATION) is not None
    if metadata is None:
        if files or required:
            raise PublicContractError("forecast enhancement packaging policy is missing")
        return  # Legacy packages remain readable.
    if metadata.get("mode") not in POLICY_MODES:
        raise PublicContractError("forecast enhancement packaging policy is invalid")
    if metadata.get("status") == "omitted":
        reason = "explicitly_disabled" if metadata["mode"] == "omit" else "not_configured_for_this_generation"
        if files or required or metadata["mode"] == "required" or dict(metadata) != {"mode": metadata["mode"], "status": "omitted", "reason": reason}:
            raise PublicContractError("required forecast enhancements were omitted")
        return
    if set(files) != {DESTINATION} or metadata.get("status") != "included" or metadata["mode"] == "omit":
        raise PublicContractError("forecast enhancement package inventory differs")
    raw = files[DESTINATION]
    try:
        document = json.loads(raw)
        validate_binding(document, payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PublicContractError(f"packaged forecast enhancement is invalid: {exc}") from exc
    expected = {"mode": metadata["mode"], "status": "included", "destination": DESTINATION,
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "binding": document["publication_binding"]}
    if dict(metadata) != expected or encode(document) != raw:
        raise PublicContractError("forecast enhancement package hash or encoding differs")
