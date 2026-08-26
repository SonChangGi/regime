"""Generic, hash-bound selection-family audit sidecar (v2)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from regime_lab.analysis.labels import STATE_ORDER
from regime_lab.analysis.selection_evaluation import (
    build_selection_evaluation,
    validate_selection_evaluation,
)
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operating_contract import load_operating_contract


SELECTION_FAMILY_AUDIT_SCHEMA_VERSION = "selection-family-audit/v2"
_ORIGIN_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "target_date",
    "evaluation_split",
    "current_state",
    "actual",
    "train_size",
    "gap",
)
_REQUIRED_DIAGNOSTICS = frozenset(
    {
        "model",
        "is_reference",
        "selected",
        "gate_passed",
        "gate_reason",
        "log_loss",
        "brier",
        "fallback_count",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "generation_id",
        "evidence_track",
        "evidence_status",
        "candidate_manifest_sha256",
        "selection_period",
        "source_artifacts",
        "candidate_count",
        "candidate_set",
        "champion",
        "runner_up",
        "selection_reason",
        "policy_sha256",
        "complexity_registry",
        "complexity_registry_sha256",
        "fallback",
        "common_origin_contract",
        "candidates",
        "supplemental_evaluation",
        "sha256",
    }
)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_string(value: object, *, context: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return result


def _nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _artifact_record(
    path: Path,
    *,
    relative_path: str,
    row_count: int,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selection-family source is missing: {relative_path}")
    if type(row_count) is not int or row_count < 1:
        raise ValueError(f"selection-family source row count is invalid: {relative_path}")
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": row_count,
    }


def _boolean_series(value: pd.Series, *, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(value.dtype):
        return value.astype(bool)
    normalized = value.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{context} must contain only true/false")
    return normalized.eq("true")


def _selection_end_timestamp(value: object) -> pd.Timestamp:
    """Normalize the payload's exclusive selection end to a zoned instant.

    Historical V5 payloads preregistered the boundary as an ISO date
    (``2023-01-01``), while newer payloads may carry a full timestamp.  A date
    is unambiguously treated as midnight UTC; a timestamp must be zoned.
    """

    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("selection_end_at must be an ISO-8601 date/timestamp") from exc
    if result.tzinfo is None:
        text = str(value)
        if len(text) == 10 and text[4:5] == "-" and text[7:8] == "-":
            result = result.tz_localize("UTC")
        else:
            raise ValueError("selection_end_at timestamp must include a timezone")
    return result


def _supplemental_evidence_status(
    *,
    evidence_track: str,
    payload_mode: str | None,
) -> str:
    if payload_mode == "demo":
        return "synthetic_fixture"
    if evidence_track == "operational_oos":
        return "operational_oos"
    if evidence_track == "reconstructed_oos":
        return "historical_reconstructed_oos"
    raise ValueError("selection-family evidence_track is invalid")


def _validate_supplemental(document: Mapping[str, Any]) -> None:
    try:
        validate_selection_evaluation(document)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"selection-family supplemental evaluation is invalid: {exc}") from exc


def _json_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _validated_origins(
    predictions: pd.DataFrame,
    candidates: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    required = {"model", *_ORIGIN_COLUMNS}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"selection predictions missing columns: {missing}")
    frame = predictions.copy()
    frame["model"] = frame["model"].astype(str)
    if set(frame["model"]) != set(candidates):
        raise ValueError("selection diagnostics and prediction candidate sets differ")
    if not frame["evaluation_split"].astype(str).eq("selection").all():
        raise ValueError("selection-family audit accepts selection rows only")
    frame["origin_date"] = pd.to_datetime(
        frame["origin_date"], errors="raise", utc=True
    )
    frame["target_date"] = pd.to_datetime(
        frame["target_date"], errors="raise", utc=True
    )
    if frame.empty or not (frame["origin_date"] < frame["target_date"]).all():
        raise ValueError("selection predictions need valid origin-target rows")
    if frame.duplicated(["model", "origin_date", "target_date"]).any():
        raise ValueError("selection predictions contain duplicate candidate origins")
    invalid = sorted(set(frame["actual"].astype(str)).difference(STATE_ORDER))
    if invalid:
        raise ValueError(
            f"selection predictions contain unsupported actuals: {invalid}"
        )
    invalid_current = sorted(
        set(frame["current_state"].astype(str)).difference(STATE_ORDER)
    )
    if invalid_current:
        raise ValueError(
            "selection predictions contain unsupported current states: "
            f"{invalid_current}"
        )

    reference_model = str(candidates[0])
    reference = frame.loc[
        frame["model"].eq(reference_model), list(_ORIGIN_COLUMNS)
    ].sort_values(["origin_date", "target_date"], ignore_index=True)
    if reference.empty:
        raise ValueError("selection predictions must not be empty")
    for model in candidates[1:]:
        candidate = frame.loc[
            frame["model"].eq(model), list(_ORIGIN_COLUMNS)
        ].sort_values(["origin_date", "target_date"], ignore_index=True)
        try:
            pd.testing.assert_frame_equal(
                reference, candidate, check_dtype=False, check_like=False
            )
        except AssertionError as exc:
            raise ValueError(
                f"candidate {model} does not share exact selection origins and actuals"
            ) from exc

    records = [
        {
            "origin_at": pd.Timestamp(row.origin_date).isoformat(),
            "target_at": pd.Timestamp(row.target_date).isoformat(),
            "evaluation_split": str(row.evaluation_split),
            "current_state": str(row.current_state),
            "actual": str(row.actual),
            "train_size": int(row.train_size),
            "gap": int(row.gap),
        }
        for row in reference.itertuples(index=False)
    ]
    return records, _canonical_json_sha256(records)


def build_selection_family_audit(
    selection_diagnostics: pd.DataFrame,
    selection_predictions: pd.DataFrame,
    *,
    champion: str,
    selection_reason: str,
    policy_sha256: str,
    complexity_registry: Mapping[str, int],
    evidence_track: str,
    generation_id: str,
    candidate_manifest_sha256: str,
    declared_selection_period: str,
    selection_end_at: str,
    source_artifacts: Mapping[str, Mapping[str, Any]],
    expected_candidate_set: Sequence[str],
    runner_up: str | None = None,
    infer_runner_up: bool = True,
    fallback: Mapping[str, Any] | None = None,
    supplemental_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an all-candidate audit and reject post-selection/mismatched input."""

    if not isinstance(selection_diagnostics, pd.DataFrame):
        raise TypeError("selection_diagnostics must be a DataFrame")
    missing = sorted(_REQUIRED_DIAGNOSTICS.difference(selection_diagnostics.columns))
    if missing:
        raise ValueError(f"selection diagnostics missing columns: {missing}")
    diagnostics = selection_diagnostics.copy()
    diagnostics["model"] = diagnostics["model"].astype(str)
    if diagnostics.empty or diagnostics["model"].duplicated().any():
        raise ValueError("selection diagnostics need unique candidates")
    for field in ("is_reference", "selected", "gate_passed"):
        diagnostics[field] = _boolean_series(
            diagnostics[field],
            context=f"selection diagnostics {field}",
        )
    for metric in ("log_loss", "brier"):
        numeric = pd.to_numeric(diagnostics[metric], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric < 0).any():
            raise ValueError(f"selection diagnostics {metric} must be finite")
        diagnostics[metric] = numeric.astype(float)
    fallback_count = pd.to_numeric(diagnostics["fallback_count"], errors="coerce")
    if (
        fallback_count.isna().any()
        or (fallback_count < 0).any()
        or not np.equal(fallback_count, np.floor(fallback_count)).all()
    ):
        raise ValueError("selection diagnostics fallback_count must be integral")
    diagnostics["fallback_count"] = fallback_count.astype(int)
    candidates = diagnostics["model"].tolist()
    expected_candidates = [str(value) for value in expected_candidate_set]
    if (
        not expected_candidates
        or len(expected_candidates) != len(set(expected_candidates))
        or candidates != expected_candidates
    ):
        raise ValueError(
            "selection diagnostics must exactly follow the contracted candidate set"
        )
    champion = str(champion)
    if champion not in candidates:
        raise ValueError("champion is absent from evaluated candidates")
    selected = diagnostics.loc[diagnostics["selected"].astype(bool), "model"].tolist()
    if selected != [champion]:
        raise ValueError("selection diagnostics must select exactly the champion")
    champion_row = diagnostics.loc[diagnostics["model"].eq(champion)].iloc[0]
    if not bool(champion_row["gate_passed"]):
        raise ValueError("champion must pass the recorded selection gate")
    if not str(selection_reason).strip():
        raise ValueError("selection_reason must not be empty")
    policy_hash = _sha256_string(policy_sha256, context="policy_sha256")
    if evidence_track not in {"operational_oos", "reconstructed_oos"}:
        raise ValueError("evidence_track is invalid")
    generation = _nonempty_string(generation_id, context="generation_id")
    manifest_hash = _sha256_string(
        candidate_manifest_sha256,
        context="candidate_manifest_sha256",
    )

    raw_complexity = {str(key): value for key, value in complexity_registry.items()}
    if len(raw_complexity) != len(complexity_registry):
        raise ValueError("complexity registry model ids collide after normalization")
    registry_keys = set(raw_complexity)
    if registry_keys != set(candidates):
        raise ValueError("complexity registry must contain every candidate exactly")
    normalized_complexity: dict[str, int] = {}
    for model in candidates:
        value = raw_complexity[model]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError("complexity ranks must be non-negative integers")
        normalized_complexity[model] = int(value)

    origins, origins_hash = _validated_origins(selection_predictions, candidates)
    selection_end = _selection_end_timestamp(selection_end_at)
    origin_targets = pd.to_datetime(
        [row["target_at"] for row in origins], utc=True, errors="raise"
    )
    if not (origin_targets < selection_end.tz_convert("UTC")).all():
        raise ValueError(
            "selection-family input contains post-selection targets"
        )
    declared_period = _nonempty_string(
        declared_selection_period,
        context="declared_selection_period",
    )

    expected_source_names = {"selection_diagnostics", "oos_predictions"}
    if set(source_artifacts) != expected_source_names:
        raise ValueError(
            "source_artifacts must bind selection diagnostics and OOS predictions"
        )
    normalized_sources: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_source_names):
        raw_source = source_artifacts[name]
        if not isinstance(raw_source, Mapping) or set(raw_source) != {
            "path",
            "sha256",
            "row_count",
        }:
            raise ValueError(f"source_artifacts.{name} fields are invalid")
        row_count = raw_source.get("row_count")
        if type(row_count) is not int or row_count < 1:
            raise ValueError(f"source_artifacts.{name}.row_count is invalid")
        normalized_sources[name] = {
            "path": _nonempty_string(
                raw_source.get("path"),
                context=f"source_artifacts.{name}.path",
            ),
            "sha256": _sha256_string(
                raw_source.get("sha256"),
                context=f"source_artifacts.{name}.sha256",
            ),
            "row_count": row_count,
        }
    if normalized_sources["selection_diagnostics"]["row_count"] != len(candidates):
        raise ValueError(
            "selection diagnostics artifact row count differs from candidates"
        )
    if normalized_sources["oos_predictions"]["row_count"] < len(origins) * len(candidates):
        raise ValueError("OOS artifact cannot contain all matched selection rows")
    eligible_runner_ups = diagnostics.loc[
        diagnostics["gate_passed"].astype(bool) & ~diagnostics["model"].eq(champion)
    ].copy()
    if infer_runner_up and runner_up is None and not eligible_runner_ups.empty:
        eligible_runner_ups["complexity_rank"] = eligible_runner_ups["model"].map(
            normalized_complexity
        )
        eligible_runner_ups = eligible_runner_ups.sort_values(
            ["log_loss", "complexity_rank", "model"], kind="stable"
        )
        runner_up = str(eligible_runner_ups.iloc[0]["model"])
    if runner_up is not None:
        runner_up = str(runner_up)
        if runner_up == champion or runner_up not in candidates:
            raise ValueError("runner_up must be a different evaluated candidate")
        runner_row = diagnostics.loc[diagnostics["model"].eq(runner_up)].iloc[0]
        if not bool(runner_row["gate_passed"]):
            raise ValueError("runner_up must pass the recorded gate")

    fallback_document: dict[str, Any] | None = None
    if fallback is not None:
        fallback_model = str(fallback.get("model", ""))
        trigger = str(fallback.get("trigger", "")).strip()
        reason = str(fallback.get("reason", "")).strip()
        if fallback_model not in candidates or not trigger or not reason:
            raise ValueError(
                "fallback requires an evaluated model, trigger, and reason"
            )
        fallback_document = {
            "model": fallback_model,
            "trigger": trigger,
            "reason": reason,
        }

    candidate_rows: list[dict[str, Any]] = []
    for position, raw in enumerate(diagnostics.to_dict(orient="records"), start=1):
        model = str(raw["model"])
        gate_reason = str(raw["gate_reason"])
        candidate_rows.append(
            {
                "candidate_order": position,
                "model": model,
                "selected": model == champion,
                "runner_up": model == runner_up,
                "is_reference": bool(raw["is_reference"]),
                "complexity_rank": normalized_complexity[model],
                "gate": {
                    "passed_all": bool(raw["gate_passed"]),
                    "reason": gate_reason,
                    "failed_checks": []
                    if gate_reason == "passed"
                    else gate_reason.split(";"),
                    "fallback_count": int(raw["fallback_count"]),
                    "raw_p_value": _json_scalar(raw.get("raw_p_value")),
                    "holm_adjusted_p_value": _json_scalar(
                        raw.get("holm_adjusted_p_value")
                    ),
                },
                "metrics": {
                    "log_loss": float(raw["log_loss"]),
                    "brier": float(raw["brier"]),
                    "calibration_error": _json_scalar(
                        raw.get("calibration_error")
                    ),
                    "n_predictions": _json_scalar(raw.get("n_predictions")),
                },
            }
        )

    _validate_supplemental(supplemental_evaluation)
    if supplemental_evaluation.get("candidate_set") != candidates:
        raise ValueError("supplemental evaluation candidate set differs")
    if supplemental_evaluation.get("selected_champion_unchanged") != champion:
        raise ValueError("supplemental evaluation champion differs")
    evidence_status = _nonempty_string(
        supplemental_evaluation.get("evidence_status"),
        context="supplemental evaluation evidence_status",
    )
    supplemental_origins = supplemental_evaluation.get("common_origin_contract")
    if not isinstance(supplemental_origins, Mapping) or (
        supplemental_origins.get("origin_count") != len(origins)
        or supplemental_origins.get("first_origin_at") != origins[0]["origin_at"]
        or supplemental_origins.get("last_origin_at") != origins[-1]["origin_at"]
    ):
        raise ValueError("supplemental evaluation matched origins differ")

    body: dict[str, Any] = {
        "schema_version": SELECTION_FAMILY_AUDIT_SCHEMA_VERSION,
        "status": "completed",
        "generation_id": generation,
        "evidence_track": evidence_track,
        "evidence_status": evidence_status,
        "candidate_manifest_sha256": manifest_hash,
        "selection_period": {
            "role": "predeployment_selection_only",
            "declared": declared_period,
            "selection_end_at": selection_end.isoformat(),
            "first_origin_at": origins[0]["origin_at"],
            "last_origin_at": origins[-1]["origin_at"],
            "first_target_at": origins[0]["target_at"],
            "last_target_at": origins[-1]["target_at"],
        },
        "source_artifacts": normalized_sources,
        "candidate_count": len(candidate_rows),
        "candidate_set": candidates,
        "champion": champion,
        "runner_up": runner_up,
        "selection_reason": str(selection_reason),
        "policy_sha256": policy_hash,
        "complexity_registry": normalized_complexity,
        "complexity_registry_sha256": _canonical_json_sha256(
            normalized_complexity
        ),
        "fallback": fallback_document,
        "common_origin_contract": {
            "status": "matched",
            "columns": list(_ORIGIN_COLUMNS),
            "origin_count": len(origins),
            "first_origin_at": origins[0]["origin_at"],
            "last_origin_at": origins[-1]["origin_at"],
            "origins_sha256": origins_hash,
        },
        "candidates": candidate_rows,
        "supplemental_evaluation": dict(supplemental_evaluation),
    }
    return {**body, "sha256": _canonical_json_sha256(body)}


def validate_selection_family_audit(
    document: Mapping[str, Any],
    *,
    expected_generation_id: str | None = None,
) -> None:
    """Validate the canonical self-hash and principal cross-field invariants."""

    if not isinstance(document, Mapping):
        raise TypeError("selection-family audit must be a mapping")
    if set(document) != _AUDIT_FIELDS:
        raise ValueError("selection-family audit fields are not exact")
    if document.get("schema_version") != SELECTION_FAMILY_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported selection-family audit schema")
    body = dict(document)
    published_hash = _sha256_string(body.pop("sha256", ""), context="sha256")
    if _canonical_json_sha256(body) != published_hash:
        raise ValueError("selection-family audit canonical hash mismatch")
    if body.get("status") != "completed":
        raise ValueError("selection-family audit status must be completed")
    generation_id = _nonempty_string(
        body.get("generation_id"),
        context="selection-family audit generation_id",
    )
    if expected_generation_id is not None and generation_id != expected_generation_id:
        raise ValueError("selection-family audit generation_id mismatch")
    _sha256_string(
        body.get("candidate_manifest_sha256"),
        context="selection-family audit candidate_manifest_sha256",
    )
    _sha256_string(body.get("policy_sha256"), context="selection policy_sha256")
    _nonempty_string(body.get("selection_reason"), context="selection_reason")
    if body.get("evidence_status") not in {
        "historical_reconstructed_oos",
        "operational_oos",
        "synthetic_fixture",
    }:
        raise ValueError("selection-family audit evidence_status is invalid")
    allowed_evidence = {
        "reconstructed_oos": {
            "historical_reconstructed_oos",
            "synthetic_fixture",
        },
        "operational_oos": {"operational_oos"},
    }
    if body.get("evidence_status") not in allowed_evidence.get(
        body.get("evidence_track"), set()
    ):
        raise ValueError("selection-family audit evidence track/status mismatch")
    period = body.get("selection_period")
    if not isinstance(period, Mapping) or set(period) != {
        "role",
        "declared",
        "selection_end_at",
        "first_origin_at",
        "last_origin_at",
        "first_target_at",
        "last_target_at",
    }:
        raise ValueError("selection-family audit selection_period is invalid")
    if period.get("role") != "predeployment_selection_only":
        raise ValueError("selection-family audit selection_period role is invalid")
    end = _selection_end_timestamp(period.get("selection_end_at"))
    last_target = pd.Timestamp(period.get("last_target_at"))
    if last_target.tzinfo is None or last_target >= end:
        raise ValueError("selection-family audit contains post-selection evidence")
    for field in ("first_origin_at", "last_origin_at", "first_target_at"):
        timestamp = pd.Timestamp(period.get(field))
        if timestamp.tzinfo is None:
            raise ValueError(f"selection-family audit {field} must be zoned")
    sources = body.get("source_artifacts")
    if not isinstance(sources, Mapping) or set(sources) != {
        "selection_diagnostics",
        "oos_predictions",
    }:
        raise ValueError("selection-family audit source_artifacts are invalid")
    for name, raw_source in sources.items():
        if not isinstance(raw_source, Mapping) or set(raw_source) != {
            "path",
            "sha256",
            "row_count",
        }:
            raise ValueError(
                f"selection-family audit source_artifacts.{name} is invalid"
            )
        _nonempty_string(
            raw_source.get("path"),
            context=f"selection-family audit source_artifacts.{name}.path",
        )
        _sha256_string(
            raw_source.get("sha256"),
            context=f"selection-family audit source_artifacts.{name}.sha256",
        )
        if type(raw_source.get("row_count")) is not int or raw_source["row_count"] < 1:
            raise ValueError(
                f"selection-family audit source_artifacts.{name}.row_count is invalid"
            )
    candidates = body.get("candidates")
    candidate_set = body.get("candidate_set")
    if not isinstance(candidates, list) or not isinstance(candidate_set, list):
        raise ValueError("selection-family audit candidates are invalid")
    if (
        not candidates
        or not all(isinstance(value, str) and value for value in candidate_set)
        or len(candidate_set) != len(set(candidate_set))
        or not all(isinstance(row, Mapping) for row in candidates)
    ):
        raise ValueError("selection-family audit candidate rows are invalid")
    if [str(row.get("model")) for row in candidates] != [str(v) for v in candidate_set]:
        raise ValueError("selection-family audit candidate order mismatch")
    if int(body.get("candidate_count", -1)) != len(candidates):
        raise ValueError("selection-family audit candidate count mismatch")
    registry = body.get("complexity_registry")
    if not isinstance(registry, dict) or set(registry) != set(candidate_set):
        raise ValueError("selection-family audit complexity registry is invalid")
    if any(type(value) is not int or value < 0 for value in registry.values()):
        raise ValueError("selection-family audit complexity ranks are invalid")
    published_registry_hash = _sha256_string(
        body.get("complexity_registry_sha256", ""),
        context="complexity_registry_sha256",
    )
    if _canonical_json_sha256(registry) != published_registry_hash:
        raise ValueError("selection-family audit complexity registry hash mismatch")
    for position, row in enumerate(candidates, start=1):
        model = row.get("model")
        if (
            row.get("candidate_order") != position
            or row.get("complexity_rank") != registry.get(model)
            or type(row.get("selected")) is not bool
            or type(row.get("runner_up")) is not bool
            or type(row.get("is_reference")) is not bool
            or not isinstance(row.get("gate"), Mapping)
            or not isinstance(row.get("metrics"), Mapping)
        ):
            raise ValueError("selection-family audit candidate row is invalid")
    selected = [row for row in candidates if row.get("selected") is True]
    if len(selected) != 1 or selected[0].get("model") != body.get("champion"):
        raise ValueError("selection-family audit champion mismatch")
    runner_up = body.get("runner_up")
    runner_rows = [row for row in candidates if row.get("runner_up") is True]
    if (runner_up is None and runner_rows) or (
        runner_up is not None
        and (len(runner_rows) != 1 or runner_rows[0].get("model") != runner_up)
    ):
        raise ValueError("selection-family audit runner-up mismatch")
    common_origins = body.get("common_origin_contract")
    if not isinstance(common_origins, Mapping) or set(common_origins) != {
        "status",
        "columns",
        "origin_count",
        "first_origin_at",
        "last_origin_at",
        "origins_sha256",
    }:
        raise ValueError("selection-family common-origin contract is invalid")
    if (
        common_origins.get("status") != "matched"
        or common_origins.get("columns") != list(_ORIGIN_COLUMNS)
        or type(common_origins.get("origin_count")) is not int
        or common_origins["origin_count"] < 1
    ):
        raise ValueError("selection-family common origins are invalid")
    _sha256_string(
        common_origins.get("origins_sha256"),
        context="selection-family origins_sha256",
    )
    if sources["selection_diagnostics"]["row_count"] != len(candidates) or (
        sources["oos_predictions"]["row_count"]
        < len(candidates) * common_origins["origin_count"]
    ):
        raise ValueError("selection-family source row coverage is incomplete")
    supplemental = body.get("supplemental_evaluation")
    if not isinstance(supplemental, Mapping):
        raise ValueError("selection-family supplemental evaluation is required")
    _validate_supplemental(supplemental)
    if supplemental.get("evidence_status") != body.get("evidence_status"):
        raise ValueError("selection-family supplemental evidence status mismatch")
    if supplemental.get("candidate_set") != candidate_set:
        raise ValueError("selection-family supplemental candidate set mismatch")
    if supplemental.get("selected_champion_unchanged") != body.get("champion"):
        raise ValueError("selection-family supplemental champion mismatch")
    supplemental_origins = supplemental.get("common_origin_contract")
    if not isinstance(supplemental_origins, Mapping) or not isinstance(
        common_origins, Mapping
    ) or any(
        supplemental_origins.get(field) != common_origins.get(field)
        for field in ("origin_count", "first_origin_at", "last_origin_at")
    ):
        raise ValueError("selection-family supplemental origins mismatch")


def validate_selection_family_payload_binding(
    document: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Validate every sidecar field whose source of truth is the V5 payload."""

    meta = payload.get("meta")
    model = payload.get("model")
    selection = payload.get("selection")
    forecast = payload.get("forecast")
    if not all(isinstance(value, Mapping) for value in (meta, model, selection, forecast)):
        raise ValueError("selection-family payload contracts are missing")
    generation_id = _nonempty_string(
        meta.get("generation_id"), context="payload.meta.generation_id"
    )
    validate_selection_family_audit(
        document,
        expected_generation_id=generation_id,
    )
    expected_status = _supplemental_evidence_status(
        evidence_track=str(forecast.get("evidence_track")),
        payload_mode=str(meta.get("mode")) if meta.get("mode") is not None else None,
    )
    exact_pairs = (
        ("candidate_manifest_sha256", model.get("candidate_manifest_sha256")),
        ("champion", model.get("champion")),
        ("runner_up", selection.get("runner_up")),
        ("selection_reason", selection.get("selection_reason")),
        ("policy_sha256", selection.get("policy_sha256")),
        ("candidate_set", selection.get("candidate_set")),
        ("evidence_track", forecast.get("evidence_track")),
        ("evidence_status", expected_status),
    )
    for field, expected in exact_pairs:
        if document.get(field) != expected:
            raise ValueError(f"selection-family {field} differs from payload")
    period = document.get("selection_period")
    if not isinstance(period, Mapping) or period.get("declared") != model.get(
        "selection_period"
    ):
        raise ValueError("selection-family declared period differs from payload")
    if _selection_end_timestamp(period.get("selection_end_at")) != (
        _selection_end_timestamp(model.get("selection_end"))
    ):
        raise ValueError("selection-family selection end differs from payload")


def build_selection_family_audit_from_artifacts(
    payload: Mapping[str, Any],
    artifact_directory: str | Path,
    *,
    supplemental_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a generation sidecar from its exact private selection sources."""

    artifacts = Path(artifact_directory)
    diagnostics_path = artifacts / "selection-diagnostics.csv"
    predictions_path = artifacts / "oos-predictions.csv"
    candidate_manifest_path = artifacts / "candidate-manifest.json"
    for path, label in (
        (diagnostics_path, "selection-diagnostics.csv"),
        (predictions_path, "oos-predictions.csv"),
        (candidate_manifest_path, "candidate-manifest.json"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"selection-family source is missing: {label}")
    try:
        diagnostics = pd.read_csv(diagnostics_path)
        predictions = pd.read_csv(predictions_path)
        artifact_manifest = json.loads(
            candidate_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("selection-family source artifacts cannot be read") from exc
    for field in ("is_reference", "selected", "gate_passed"):
        if field not in diagnostics:
            raise ValueError(f"selection diagnostics missing {field}")
        diagnostics[field] = _boolean_series(
            diagnostics[field], context=f"selection diagnostics {field}"
        )

    meta = payload.get("meta")
    model = payload.get("model")
    selection = payload.get("selection")
    forecast = payload.get("forecast")
    if not all(isinstance(value, Mapping) for value in (meta, model, selection, forecast)):
        raise ValueError("selection-family payload contracts are missing")
    generation_id = _nonempty_string(
        meta.get("generation_id"), context="payload.meta.generation_id"
    )
    candidate_manifest_sha256 = _sha256_string(
        model.get("candidate_manifest_sha256"),
        context="payload.model.candidate_manifest_sha256",
    )
    if not isinstance(artifact_manifest, Mapping):
        raise ValueError("candidate-manifest.json must be an object")
    artifact_manifest_body = dict(artifact_manifest)
    artifact_manifest_hash = artifact_manifest_body.pop("sha256", None)
    if (
        artifact_manifest_hash != candidate_manifest_sha256
        or canonical_json_sha256_v1(artifact_manifest_body)
        != candidate_manifest_sha256
        or artifact_manifest_body != model.get("candidate_manifest")
    ):
        raise ValueError("candidate manifest differs from payload")

    candidate_names = diagnostics["model"].astype(str).tolist()
    expected_candidates = [str(value) for value in selection.get("candidate_set", ())]
    if candidate_names != expected_candidates:
        raise ValueError("selection candidate set differs from payload")
    operating = load_operating_contract()
    historical = operating.historical_reviewed_roster_by_manifest_sha256(
        candidate_manifest_sha256
    )
    manifest_models = artifact_manifest_body.get("models")
    if not isinstance(manifest_models, list):
        raise ValueError("candidate manifest model registry is invalid")
    manifest_complexity = {
        str(row.get("name")): row.get("complexity_rank")
        for row in manifest_models
        if isinstance(row, Mapping)
    }
    if historical is not None:
        if set(candidate_names) != set(historical["candidate_models"]):
            raise ValueError(
                "selection candidate set differs from the historical operating roster"
            )
        registry_source = manifest_complexity
    else:
        registry_source = operating.selection_policy["complexity_registry"]
    missing = sorted(set(candidate_names).difference(registry_source))
    if missing:
        raise ValueError(f"selection complexity registry is incomplete: {missing}")
    complexity: dict[str, int] = {}
    for name in candidate_names:
        value = registry_source[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"selection complexity rank is invalid for {name}")
        complexity[name] = int(value)

    selection_predictions = predictions.loc[
        predictions["evaluation_split"].astype(str).eq("selection")
    ].copy()
    if len(selection_predictions) == len(predictions):
        # Full OOS source files normally contain both selection and holdout.
        # Synthetic fixtures may intentionally contain only selection rows.
        pass
    elif set(predictions["evaluation_split"].astype(str)) != {"selection", "holdout"}:
        raise ValueError("OOS predictions contain an unsupported evaluation split")
    if "fallback" in selection_predictions:
        selection_predictions["fallback"] = _boolean_series(
            selection_predictions["fallback"],
            context="selection predictions fallback",
        )
    expected_supplemental = build_selection_evaluation(
        selection_predictions,
        diagnostics,
        evidence_status=_supplemental_evidence_status(
            evidence_track=str(forecast.get("evidence_track")),
            payload_mode=str(meta.get("mode")) if meta.get("mode") is not None else None,
        ),
    )
    if supplemental_evaluation is not None:
        validate_selection_evaluation(supplemental_evaluation)
        if dict(supplemental_evaluation) != expected_supplemental:
            raise ValueError(
                "supplemental evaluation differs from selection-only source evidence"
            )
    supplemental_evaluation = expected_supplemental
    return build_selection_family_audit(
        diagnostics,
        selection_predictions,
        champion=str(model.get("champion")),
        selection_reason=str(selection.get("selection_reason")),
        policy_sha256=str(selection.get("policy_sha256")),
        complexity_registry=complexity,
        evidence_track=str(forecast.get("evidence_track")),
        generation_id=generation_id,
        candidate_manifest_sha256=candidate_manifest_sha256,
        declared_selection_period=str(model.get("selection_period")),
        selection_end_at=str(model.get("selection_end")),
        source_artifacts={
            "selection_diagnostics": _artifact_record(
                diagnostics_path,
                relative_path="selection-diagnostics.csv",
                row_count=len(diagnostics),
            ),
            "oos_predictions": _artifact_record(
                predictions_path,
                relative_path="oos-predictions.csv",
                row_count=len(predictions),
            ),
        },
        expected_candidate_set=expected_candidates,
        runner_up=selection.get("runner_up"),
        infer_runner_up=False,
        fallback={
            "model": "markov",
            "trigger": "no_challenger_passes_gate",
            "reason": "official_probability_reference_fallback",
        },
        supplemental_evaluation=supplemental_evaluation,
    )


__all__ = [
    "SELECTION_FAMILY_AUDIT_SCHEMA_VERSION",
    "build_selection_family_audit",
    "build_selection_family_audit_from_artifacts",
    "validate_selection_family_audit",
    "validate_selection_family_payload_binding",
]
