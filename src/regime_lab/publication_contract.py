"""Installable validation contract for the public V5 comparison sidecar."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from regime_lab.contract_v5 import V5_LEGACY_REVIEWED_005_SNAPSHOT_SHA256
from regime_lab.frozen_v4 import (
    FROZEN_V4_BASELINE,
    FROZEN_V4_INVENTORY_FILE_COUNT,
    FROZEN_V4_OOS_PREDICTIONS,
)


V5_RESULT_VERSION = "weekly-regime-result-v5"
V5_COMPARISON_SCHEMA_VERSION = "regime-v5-v4-matched-comparison/1"
FORBIDDEN_LIVE_DERIVED_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "provider_response",
        "provider_responses",
        "raw_observation",
        "raw_observations",
        "request_params",
        "revision_seq",
        "secret",
        "snapshot_id",
        "token",
        "vintage_date",
    }
)
FORBIDDEN_COMPARISON_DATA_KEYS = frozenset(
    {
        "actual",
        "actuals",
        "feature_value",
        "feature_values",
        "observation",
        "observations",
        "provider_value",
        "provider_values",
        "raw_value",
        "raw_values",
        "record",
        "records",
        "row",
        "rows",
        "series_value",
        "series_values",
    }
)
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
V5_COMPARISON_TOP_LEVEL_KEYS = frozenset(
    {
        "comparison_contract",
        "fx_ablation",
        "inputs",
        "promotion_interpretation",
        "provider_or_raw_feature_values_included",
        "report_role",
        "schema_version",
        "v5_causal_multiscale_ensemble_vs_v5_markov",
        "v5_markov_vs_frozen_v4_markov",
    }
)
PARITY_METRICS = frozenset(
    {"balanced_accuracy", "brier", "fallback_rate", "log_loss"}
)


class PublicContractError(RuntimeError):
    """The public derived comparison does not satisfy its release contract."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    try:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicContractError("V5 payload cannot be canonicalized") from exc
    return _sha256(raw)


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _reject_raw_provider_material(value: object, *, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            is_provider_observation = (
                path.startswith("payload.sources[")
                and normalized in {"observation", "observations"}
            )
            if normalized in FORBIDDEN_LIVE_DERIVED_KEYS or is_provider_observation:
                raise PublicContractError(
                    f"live-derived payload contains forbidden provider material at {path}.{key}"
                )
            _reject_raw_provider_material(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_raw_provider_material(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and (
        value.startswith(("/Users/", "/private/", "file://")) or "\\Users\\" in value
    ):
        raise PublicContractError(
            f"live-derived payload contains a machine-local path at {path}"
        )


def _reject_comparison_raw_material(
    value: object,
    *,
    path: str = "comparison",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_COMPARISON_DATA_KEYS:
                raise PublicContractError(
                    f"V5 comparison contains row-level or raw material at {path}.{key}"
                )
            _reject_comparison_raw_material(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_comparison_raw_material(child, path=f"{path}[{index}]")


def _require_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicContractError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str] | frozenset[str],
    *,
    context: str,
) -> None:
    if set(value) != set(expected):
        raise PublicContractError(f"{context} fields are not exact")


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        raise PublicContractError(f"{context} must be a lowercase SHA-256")
    return value


def _require_positive_integer(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise PublicContractError(f"{context} must be a positive integer")
    return value


def _require_zero_integer(value: object, *, context: str) -> None:
    if type(value) is not int or value != 0:
        raise PublicContractError(f"{context} must be integer zero")


def _require_zero_number(value: object, *, context: str) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value != 0:
        raise PublicContractError(f"{context} must be exactly zero")


def _require_finite_number(value: object, *, context: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise PublicContractError(f"{context} must be a finite number")
    return float(value)


def _require_close_number(
    value: object,
    expected: object,
    *,
    context: str,
    tolerance: float = 5e-8,
) -> None:
    supplied = _require_finite_number(value, context=context)
    target = _require_finite_number(expected, context=f"{context} payload target")
    if not math.isclose(supplied, target, abs_tol=tolerance, rel_tol=0.0):
        raise PublicContractError(f"{context} does not match the reviewed payload")


def _validate_artifact_record(
    value: object,
    *,
    context: str,
    expected_path: str,
    require_row_count: bool,
) -> dict[str, Any]:
    record = _require_object(value, context=context)
    expected_keys = {"path", "sha256"}
    if require_row_count:
        expected_keys.add("row_count")
    _require_exact_keys(record, expected_keys, context=context)
    if record["path"] != expected_path:
        raise PublicContractError(f"{context}.path is invalid")
    _require_sha256(record["sha256"], context=f"{context}.sha256")
    if require_row_count:
        _require_positive_integer(record["row_count"], context=f"{context}.row_count")
    return record


def _validate_v5_comparison_inputs(
    report: dict[str, Any],
    payload: dict[str, Any],
    payload_raw: bytes,
) -> None:
    inputs = _require_object(report.get("inputs"), context="V5 comparison.inputs")
    _require_exact_keys(inputs, {"frozen_v4", "v5"}, context="V5 comparison.inputs")

    model = _require_object(payload.get("model"), context="V5 payload.model")
    payload_baseline = _require_object(
        model.get("baseline_v4"),
        context="V5 payload.model.baseline_v4",
    )
    if payload_baseline != dict(FROZEN_V4_BASELINE):
        raise PublicContractError("V5 payload frozen V4 baseline is not the reviewed contract")

    frozen = _require_object(
        inputs.get("frozen_v4"),
        context="V5 comparison.inputs.frozen_v4",
    )
    _require_exact_keys(
        frozen,
        {"baseline_id", "oos_predictions", "sha256sums", "verified_file_count"},
        context="V5 comparison.inputs.frozen_v4",
    )
    expected_baseline_id = Path(str(FROZEN_V4_BASELINE["artifacts_path"])).name
    if frozen["baseline_id"] != expected_baseline_id:
        raise PublicContractError("V5 comparison frozen baseline id is invalid")
    if (
        type(frozen["verified_file_count"]) is not int
        or frozen["verified_file_count"] != FROZEN_V4_INVENTORY_FILE_COUNT
    ):
        raise PublicContractError("V5 comparison frozen baseline file count is invalid")
    frozen_inventory = _validate_artifact_record(
        frozen["sha256sums"],
        context="V5 comparison.inputs.frozen_v4.sha256sums",
        expected_path="SHA256SUMS",
        require_row_count=False,
    )
    if frozen_inventory["sha256"] != payload_baseline["artifacts_inventory_sha256"]:
        raise PublicContractError("V5 comparison frozen baseline inventory hash mismatch")
    frozen_oos = _validate_artifact_record(
        frozen["oos_predictions"],
        context="V5 comparison.inputs.frozen_v4.oos_predictions",
        expected_path="oos-predictions.csv",
        require_row_count=True,
    )
    if frozen_oos != dict(FROZEN_V4_OOS_PREDICTIONS):
        raise PublicContractError("V5 comparison frozen OOS record is invalid")

    v5 = _require_object(inputs.get("v5"), context="V5 comparison.inputs.v5")
    _require_exact_keys(
        v5,
        {
            "fx_ablation_oos",
            "oos_predictions",
            "regime_results",
            "selection_diagnostics",
        },
        context="V5 comparison.inputs.v5",
    )
    regime_results = _validate_artifact_record(
        v5["regime_results"],
        context="V5 comparison.inputs.v5.regime_results",
        expected_path="regime-results.json",
        require_row_count=False,
    )
    if regime_results["sha256"] != _sha256(payload_raw):
        raise PublicContractError("V5 comparison payload hash mismatch")
    core_artifacts = _require_object(
        model.get("core_artifacts"),
        context="V5 payload.model.core_artifacts",
    )
    research_artifacts = _require_object(
        model.get("research_artifacts"),
        context="V5 payload.model.research_artifacts",
    )
    bindings = (
        (
            "fx_ablation_oos",
            "fx-ablation-oos.csv",
            research_artifacts.get("fx_ablation_oos"),
        ),
        (
            "oos_predictions",
            "oos-predictions.csv",
            core_artifacts.get("oos_predictions"),
        ),
        (
            "selection_diagnostics",
            "selection-diagnostics.csv",
            core_artifacts.get("selection_diagnostics"),
        ),
    )
    for field, expected_path, payload_record in bindings:
        record = _validate_artifact_record(
            v5[field],
            context=f"V5 comparison.inputs.v5.{field}",
            expected_path=expected_path,
            require_row_count=True,
        )
        if not isinstance(payload_record, dict) or record != payload_record:
            raise PublicContractError(
                f"V5 comparison {field} record does not match the payload manifest"
            )


def _validate_exact_markov_parity_split(value: object, *, context: str) -> int:
    split = _require_object(value, context=context)
    _require_exact_keys(
        split,
        {"common_keys", "delta_left_minus_right", "metrics", "probability_parity"},
        context=context,
    )
    common_keys = _require_object(split["common_keys"], context=f"{context}.common_keys")
    _require_exact_keys(common_keys, {"count", "sha256"}, context=f"{context}.common_keys")
    count = _require_positive_integer(common_keys["count"], context=f"{context}.common_keys.count")
    _require_sha256(common_keys["sha256"], context=f"{context}.common_keys.sha256")

    deltas = _require_object(
        split["delta_left_minus_right"],
        context=f"{context}.delta_left_minus_right",
    )
    _require_exact_keys(deltas, PARITY_METRICS, context=f"{context}.delta_left_minus_right")
    for metric, value in deltas.items():
        _require_zero_number(value, context=f"{context}.delta_left_minus_right.{metric}")

    metrics = _require_object(split["metrics"], context=f"{context}.metrics")
    _require_exact_keys(metrics, {"frozen_v4_markov", "v5_markov"}, context=f"{context}.metrics")
    frozen_metrics = _require_object(
        metrics["frozen_v4_markov"],
        context=f"{context}.metrics.frozen_v4_markov",
    )
    v5_metrics = _require_object(
        metrics["v5_markov"],
        context=f"{context}.metrics.v5_markov",
    )
    expected_metric_keys = {
        "balanced_accuracy",
        "brier",
        "fallback_count",
        "fallback_rate",
        "log_loss",
        "n",
    }
    _require_exact_keys(
        frozen_metrics,
        expected_metric_keys,
        context=f"{context}.metrics.frozen_v4_markov",
    )
    _require_exact_keys(
        v5_metrics,
        expected_metric_keys,
        context=f"{context}.metrics.v5_markov",
    )
    if frozen_metrics != v5_metrics:
        raise PublicContractError(f"{context} Markov metrics are not exactly equal")
    if type(frozen_metrics["n"]) is not int or frozen_metrics["n"] != count:
        raise PublicContractError(f"{context} Markov metric count does not match common keys")
    _require_zero_integer(
        frozen_metrics["fallback_count"],
        context=f"{context}.fallback_count",
    )
    _require_zero_number(frozen_metrics["fallback_rate"], context=f"{context}.fallback_rate")
    for metric in ("balanced_accuracy", "brier", "log_loss"):
        metric_value = frozen_metrics[metric]
        if type(metric_value) not in {int, float} or not math.isfinite(metric_value):
            raise PublicContractError(f"{context}.{metric} must be finite")

    parity = _require_object(split["probability_parity"], context=f"{context}.probability_parity")
    _require_exact_keys(
        parity,
        {"probability_numeric", "probability_token_bytes"},
        context=f"{context}.probability_parity",
    )
    numeric = _require_object(
        parity["probability_numeric"],
        context=f"{context}.probability_parity.probability_numeric",
    )
    _require_exact_keys(
        numeric,
        {"exact_float_parity", "maximum_absolute_difference", "mismatch_rows", "mismatch_values"},
        context=f"{context}.probability_parity.probability_numeric",
    )
    if numeric["exact_float_parity"] is not True:
        raise PublicContractError(f"{context} numeric probability parity is not exact")
    _require_zero_number(
        numeric["maximum_absolute_difference"],
        context=f"{context}.maximum_absolute_difference",
    )
    _require_zero_integer(
        numeric["mismatch_rows"],
        context=f"{context}.numeric_mismatch_rows",
    )
    _require_zero_integer(
        numeric["mismatch_values"],
        context=f"{context}.numeric_mismatch_values",
    )

    tokens = _require_object(
        parity["probability_token_bytes"],
        context=f"{context}.probability_parity.probability_token_bytes",
    )
    _require_exact_keys(
        tokens,
        {"exact_parity", "left_sha256", "mismatch_rows", "mismatch_values", "right_sha256"},
        context=f"{context}.probability_parity.probability_token_bytes",
    )
    if tokens["exact_parity"] is not True:
        raise PublicContractError(f"{context} token probability parity is not exact")
    left_sha = _require_sha256(tokens["left_sha256"], context=f"{context}.left_sha256")
    right_sha = _require_sha256(tokens["right_sha256"], context=f"{context}.right_sha256")
    _require_zero_integer(
        tokens["mismatch_rows"],
        context=f"{context}.token_mismatch_rows",
    )
    _require_zero_integer(
        tokens["mismatch_values"],
        context=f"{context}.token_mismatch_values",
    )
    if left_sha != right_sha:
        raise PublicContractError(f"{context} token probability hashes or mismatches differ")
    return count


def _validate_exact_markov_parity(report: dict[str, Any]) -> None:
    comparison = _require_object(
        report.get("v5_markov_vs_frozen_v4_markov"),
        context="V5 comparison Markov parity",
    )
    _require_exact_keys(
        comparison,
        {"common_keys", "join", "post_selection_holdout", "primary_selection"},
        context="V5 comparison Markov parity",
    )
    common_keys = _require_object(
        comparison["common_keys"],
        context="V5 comparison Markov parity.common_keys",
    )
    _require_exact_keys(
        common_keys,
        {"count", "sha256"},
        context="V5 comparison Markov parity.common_keys",
    )
    common_count = _require_positive_integer(
        common_keys["count"],
        context="V5 comparison Markov parity.common_keys.count",
    )
    _require_sha256(
        common_keys["sha256"],
        context="V5 comparison Markov parity.common_keys.sha256",
    )

    join = _require_object(comparison["join"], context="V5 comparison Markov parity.join")
    _require_exact_keys(
        join,
        {
            "common_key_count",
            "left_key_count",
            "left_model",
            "model_equivalence",
            "right_key_count",
            "right_model",
        },
        context="V5 comparison Markov parity.join",
    )
    if (
        type(join["common_key_count"]) is not int
        or join["common_key_count"] != common_count
        or type(join["left_key_count"]) is not int
        or join["left_key_count"] < common_count
        or type(join["right_key_count"]) is not int
        or join["right_key_count"] != common_count
        or join["left_model"] != "markov"
        or join["right_model"] != "markov"
        or join["model_equivalence"] != "exact_name"
    ):
        raise PublicContractError("V5 comparison Markov join contract is invalid")

    selection_count = _validate_exact_markov_parity_split(
        comparison["primary_selection"],
        context="V5 comparison primary selection parity",
    )
    holdout_count = _validate_exact_markov_parity_split(
        comparison["post_selection_holdout"],
        context="V5 comparison post-selection holdout parity",
    )
    if selection_count + holdout_count != common_count:
        raise PublicContractError("V5 comparison split counts do not equal common keys")


def _index_named_rows(
    value: object,
    *,
    name_field: str,
    context: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise PublicContractError(f"{context} must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = _require_object(raw, context=f"{context}[{index}]")
        name = row.get(name_field)
        if not isinstance(name, str) or not name or name in rows:
            raise PublicContractError(f"{context} has an invalid or duplicate {name_field}")
        rows[name] = row
    return rows


def _validate_fx_comparison(
    value: object,
    *,
    payload: dict[str, Any],
) -> None:
    context = "V5 comparison.fx_ablation"
    summary = _require_object(value, context=context)
    _require_exact_keys(
        summary,
        {
            "aggregate_crosscheck",
            "common_origins",
            "comparison_status",
            "interpretation",
            "payload_gate_metric_crosscheck",
            "source_status",
            "variants",
        },
        context=context,
    )
    model = _require_object(payload.get("model"), context="V5 payload.model")
    ablation = _require_object(
        model.get("fx_ablation"),
        context="V5 payload.model.fx_ablation",
    )
    if (
        summary.get("comparison_status") != "evaluated"
        or summary.get("source_status") != ablation.get("status")
        or ablation.get("status") != "evaluated"
        or summary.get("aggregate_crosscheck") is not True
        or summary.get("payload_gate_metric_crosscheck") is not True
        or summary.get("interpretation")
        != "diagnostic_only_not_a_promotion_decision"
    ):
        raise PublicContractError("V5 comparison FX publication identity is invalid")
    if (
        ablation.get("promotion_allowed") is not False
        or ablation.get("core_champion_promoted") is not False
    ):
        raise PublicContractError("V5 comparison FX promotion must remain disabled")

    payload_origins = _require_object(
        ablation.get("common_evaluation_origins"),
        context="V5 payload FX common origins",
    )
    common_origins = _require_object(
        summary.get("common_origins"),
        context=f"{context}.common_origins",
    )
    _require_exact_keys(
        common_origins,
        {"count", "sha256"},
        context=f"{context}.common_origins",
    )
    if common_origins != {
        "count": payload_origins.get("count"),
        "sha256": payload_origins.get("sha256"),
    }:
        raise PublicContractError("V5 comparison FX common origins do not match payload")
    _require_positive_integer(
        common_origins["count"],
        context=f"{context}.common_origins.count",
    )
    _require_sha256(
        common_origins["sha256"],
        context=f"{context}.common_origins.sha256",
    )

    payload_metrics = _index_named_rows(
        ablation.get("variant_metrics"),
        name_field="variant",
        context="V5 payload FX variant metrics",
    )
    variants = summary.get("variants")
    if not isinstance(variants, list) or [
        row.get("variant") if isinstance(row, dict) else None for row in variants
    ] != list(payload_metrics):
        raise PublicContractError("V5 comparison FX variant order does not match payload")
    control = payload_metrics.get("v4_control")
    if control is None:
        raise PublicContractError("V5 comparison FX control is missing")
    gate = _require_object(ablation.get("gate"), context="V5 payload FX gate")
    if gate.get("passed_variants") != []:
        raise PublicContractError("V5 comparison FX gate must not pass a variant")
    gate_rows = _index_named_rows(
        gate.get("comparisons"),
        name_field="variant",
        context="V5 payload FX gate comparisons",
    )
    exact_variant_fields = {
        "accuracy",
        "balanced_accuracy",
        "brier",
        "delta_vs_v4_control",
        "fallback_count",
        "feature_columns_sha256",
        "feature_count",
        "fx_feature_count",
        "log_loss",
        "n_predictions",
        "variant",
    }
    copied_fields = exact_variant_fields - {"delta_vs_v4_control"}
    for index, raw in enumerate(variants):
        row_context = f"{context}.variants[{index}]"
        row = _require_object(raw, context=row_context)
        _require_exact_keys(row, exact_variant_fields, context=row_context)
        name = str(row["variant"])
        payload_row = payload_metrics[name]
        for field in copied_fields:
            if field in {"accuracy", "balanced_accuracy", "brier", "log_loss"}:
                _require_close_number(
                    row[field],
                    payload_row.get(field),
                    context=f"{row_context}.{field}",
                )
            elif row[field] != payload_row.get(field):
                raise PublicContractError(
                    f"{row_context}.{field} does not match the reviewed payload"
                )
        delta = _require_object(
            row["delta_vs_v4_control"],
            context=f"{row_context}.delta_vs_v4_control",
        )
        _require_exact_keys(
            delta,
            {"brier", "log_loss"},
            context=f"{row_context}.delta_vs_v4_control",
        )
        expected_log_loss = float(payload_row["log_loss"]) - float(control["log_loss"])
        expected_brier = float(payload_row["brier"]) - float(control["brier"])
        _require_close_number(
            delta["log_loss"],
            expected_log_loss,
            context=f"{row_context}.delta_vs_v4_control.log_loss",
        )
        _require_close_number(
            delta["brier"],
            expected_brier,
            context=f"{row_context}.delta_vs_v4_control.brier",
        )
        if name != "v4_control":
            gate_row = gate_rows.get(name)
            if gate_row is None:
                raise PublicContractError(f"{row_context} has no payload gate comparison")
            _require_close_number(
                delta["log_loss"],
                -float(gate_row["mean_log_loss_improvement"]),
                context=f"{row_context}.payload_gate_log_loss",
            )
            _require_close_number(
                delta["brier"],
                gate_row["brier_difference"],
                context=f"{row_context}.payload_gate_brier",
            )


def _validate_multiscale_metric_row(
    value: object,
    *,
    context: str,
) -> dict[str, Any]:
    row = _require_object(value, context=context)
    expected = {
        "balanced_accuracy",
        "brier",
        "fallback_count",
        "fallback_rate",
        "log_loss",
        "n",
    }
    _require_exact_keys(row, expected, context=context)
    count = _require_positive_integer(row["n"], context=f"{context}.n")
    if type(row["fallback_count"]) is not int or not 0 <= row["fallback_count"] <= count:
        raise PublicContractError(f"{context}.fallback_count is invalid")
    for field in ("balanced_accuracy", "brier", "fallback_rate", "log_loss"):
        _require_finite_number(row[field], context=f"{context}.{field}")
    expected_fallback_rate = row["fallback_count"] / count
    _require_close_number(
        row["fallback_rate"],
        expected_fallback_rate,
        context=f"{context}.fallback_rate",
        tolerance=1e-12,
    )
    return row


def _validate_multiscale_split(
    value: object,
    *,
    context: str,
    payload_rows: dict[str, dict[str, Any]],
    payload_name_field: str,
    bind_balanced_accuracy: bool,
) -> int:
    split = _require_object(value, context=context)
    _require_exact_keys(
        split,
        {"common_keys", "delta_left_minus_right", "metrics"},
        context=context,
    )
    common_keys = _require_object(split["common_keys"], context=f"{context}.common_keys")
    _require_exact_keys(common_keys, {"count", "sha256"}, context=f"{context}.common_keys")
    count = _require_positive_integer(common_keys["count"], context=f"{context}.common_keys.count")
    _require_sha256(common_keys["sha256"], context=f"{context}.common_keys.sha256")
    metrics = _require_object(split["metrics"], context=f"{context}.metrics")
    model_names = {"causal_multiscale_ensemble", "v5_markov"}
    _require_exact_keys(metrics, model_names, context=f"{context}.metrics")
    left = _validate_multiscale_metric_row(
        metrics["causal_multiscale_ensemble"],
        context=f"{context}.metrics.causal_multiscale_ensemble",
    )
    right = _validate_multiscale_metric_row(
        metrics["v5_markov"],
        context=f"{context}.metrics.v5_markov",
    )
    if left["n"] != count or right["n"] != count:
        raise PublicContractError(f"{context} metric counts do not match common keys")

    deltas = _require_object(
        split["delta_left_minus_right"],
        context=f"{context}.delta_left_minus_right",
    )
    _require_exact_keys(deltas, PARITY_METRICS, context=f"{context}.delta_left_minus_right")
    for metric in PARITY_METRICS:
        _require_close_number(
            deltas[metric],
            float(left[metric]) - float(right[metric]),
            context=f"{context}.delta_left_minus_right.{metric}",
            tolerance=1e-12,
        )

    for sidecar_name, payload_name, metric_row in (
        ("causal_multiscale_ensemble", "causal_multiscale_ensemble", left),
        ("v5_markov", "markov", right),
    ):
        payload_row = payload_rows.get(payload_name)
        if payload_row is None or payload_row.get(payload_name_field) != payload_name:
            raise PublicContractError(f"{context} payload model row is missing: {payload_name}")
        if payload_row.get("n_predictions") != count:
            raise PublicContractError(f"{context} payload count mismatch: {payload_name}")
        if payload_row.get("fallback_count") != metric_row["fallback_count"]:
            raise PublicContractError(f"{context} payload fallback mismatch: {payload_name}")
        for field in ("log_loss", "brier"):
            _require_close_number(
                metric_row[field],
                payload_row.get(field),
                context=f"{context}.metrics.{sidecar_name}.{field}",
            )
        if bind_balanced_accuracy:
            _require_close_number(
                metric_row["balanced_accuracy"],
                payload_row.get("balanced_accuracy"),
                context=f"{context}.metrics.{sidecar_name}.balanced_accuracy",
            )
    return count


def _validate_multiscale_comparison(
    value: object,
    *,
    payload: dict[str, Any],
) -> None:
    context = "V5 comparison Multiscale diagnostic"
    comparison = _require_object(value, context=context)
    _require_exact_keys(
        comparison,
        {
            "common_keys",
            "join",
            "post_selection_holdout",
            "primary_selection",
            "selection_gate_crosscheck",
        },
        context=context,
    )
    common_keys = _require_object(comparison["common_keys"], context=f"{context}.common_keys")
    _require_exact_keys(common_keys, {"count", "sha256"}, context=f"{context}.common_keys")
    common_count = _require_positive_integer(common_keys["count"], context=f"{context}.common_keys.count")
    _require_sha256(common_keys["sha256"], context=f"{context}.common_keys.sha256")
    join = _require_object(comparison["join"], context=f"{context}.join")
    _require_exact_keys(
        join,
        {
            "common_key_count",
            "left_key_count",
            "left_model",
            "model_equivalence",
            "right_key_count",
            "right_model",
        },
        context=f"{context}.join",
    )
    if join != {
        "common_key_count": common_count,
        "left_key_count": common_count,
        "left_model": "causal_multiscale_ensemble",
        "model_equivalence": "fixed_pair_same_origin_and_target",
        "right_key_count": common_count,
        "right_model": "markov",
    }:
        raise PublicContractError("V5 comparison Multiscale join contract is invalid")

    model = _require_object(payload.get("model"), context="V5 payload.model")
    diagnostics = _index_named_rows(
        model.get("selection_diagnostics"),
        name_field="model",
        context="V5 payload selection diagnostics",
    )
    leaderboard = _index_named_rows(
        model.get("leaderboard"),
        name_field="name",
        context="V5 payload leaderboard",
    )
    selection_count = _validate_multiscale_split(
        comparison["primary_selection"],
        context=f"{context}.primary_selection",
        payload_rows=diagnostics,
        payload_name_field="model",
        bind_balanced_accuracy=False,
    )
    holdout_count = _validate_multiscale_split(
        comparison["post_selection_holdout"],
        context=f"{context}.post_selection_holdout",
        payload_rows=leaderboard,
        payload_name_field="name",
        bind_balanced_accuracy=True,
    )
    if selection_count + holdout_count != common_count:
        raise PublicContractError("V5 comparison Multiscale split counts are inconsistent")

    gate = _require_object(
        comparison["selection_gate_crosscheck"],
        context=f"{context}.selection_gate_crosscheck",
    )
    artifact_role = gate.get("artifact_role")
    if artifact_role == "selection_only_existing_champion_gate":
        if (
            _canonical_payload_sha256(payload)
            != V5_LEGACY_REVIEWED_005_SNAPSHOT_SHA256
        ):
            raise PublicContractError(
                "legacy V5 comparison gate is restricted to the exact reviewed "
                "0.05 snapshot"
            )
        gate_field = "pairwise_gate_against_markov"
    elif artifact_role == "selection_family_independently_recomputed":
        gate_field = "multiscale_gate_against_selection_reference"
    else:
        raise PublicContractError("V5 comparison Multiscale gate identity is invalid")
    _require_exact_keys(
        gate,
        {"artifact_role", "models", gate_field},
        context=f"{context}.selection_gate_crosscheck",
    )
    if type(gate.get(gate_field)) is not bool:
        raise PublicContractError("V5 comparison Multiscale gate identity is invalid")
    gate_models = _require_object(
        gate.get("models"),
        context=f"{context}.selection_gate_crosscheck.models",
    )
    champion = model.get("champion")
    if not isinstance(champion, str) or champion not in diagnostics:
        raise PublicContractError("V5 payload champion selection diagnostics are missing")
    champion_reference = diagnostics[champion].get("reference_model")
    if (
        not isinstance(champion_reference, str)
        or champion_reference not in diagnostics
    ):
        raise PublicContractError("V5 payload champion reference diagnostics are missing")
    required_gate_models = {
        "causal_multiscale_ensemble",
        "markov",
        champion,
        champion_reference,
    }
    _require_exact_keys(
        gate_models,
        required_gate_models,
        context=f"{context}.selection_gate_crosscheck.models",
    )
    gate_fields = {
        "brier",
        "fallback_count",
        "gate_passed",
        "gate_reason",
        "log_loss",
        "matched_metric_crosscheck",
        "n_predictions",
        "reference_model",
        "selected",
    }
    for name in sorted(required_gate_models):
        row_context = f"{context}.selection_gate_crosscheck.models.{name}"
        row = _require_object(gate_models[name], context=row_context)
        _require_exact_keys(row, gate_fields, context=row_context)
        payload_row = diagnostics.get(name)
        if payload_row is None or row.get("matched_metric_crosscheck") is not True:
            raise PublicContractError(f"{row_context} is not bound to selection diagnostics")
        for field in (
            "fallback_count",
            "gate_passed",
            "gate_reason",
            "n_predictions",
            "reference_model",
            "selected",
        ):
            if row.get(field) != payload_row.get(field):
                raise PublicContractError(f"{row_context}.{field} does not match payload")
        for field in ("brier", "log_loss"):
            _require_close_number(
                row[field],
                payload_row.get(field),
                context=f"{row_context}.{field}",
            )
    if (
        gate_models[champion].get("selected") is not True
        or gate_models[champion].get("gate_passed") is not True
        or gate_models[champion].get("reference_model") != champion_reference
    ):
        raise PublicContractError(
            "V5 comparison gate does not bind the selected champion to its reference"
        )
    if (
        artifact_role == "selection_only_existing_champion_gate"
        and diagnostics["causal_multiscale_ensemble"].get("reference_model")
        != "markov"
    ):
        raise PublicContractError(
            "legacy V5 comparison Multiscale gate is not a Markov comparison"
        )
    if (
        gate[gate_field]
        is not gate_models["causal_multiscale_ensemble"]["gate_passed"]
    ):
        raise PublicContractError(
            "V5 comparison Multiscale selection-reference gate does not match selection diagnostics"
        )


def _validate_champion_selection_evidence(payload: dict[str, Any]) -> None:
    """Bind the public sidecar to the payload's data-selected champion."""

    model = _require_object(payload.get("model"), context="V5 payload.model")
    champion = model.get("champion")
    if not isinstance(champion, str) or not champion:
        raise PublicContractError("V5 payload champion is invalid")

    leaderboard = _index_named_rows(
        model.get("leaderboard"),
        name_field="name",
        context="V5 payload leaderboard",
    )
    selected_leaderboard = [
        name
        for name, row in leaderboard.items()
        if row.get("selected") is True
    ]
    champion_leaderboard = [
        name for name, row in leaderboard.items() if row.get("is_champion") is True
    ]
    if selected_leaderboard != [champion] or champion_leaderboard != [champion]:
        raise PublicContractError(
            "V5 payload leaderboard does not select exactly its champion"
        )

    diagnostics = _index_named_rows(
        model.get("selection_diagnostics"),
        name_field="model",
        context="V5 payload selection diagnostics",
    )
    selected_diagnostics = [
        name for name, row in diagnostics.items() if row.get("selected") is True
    ]
    if selected_diagnostics != [champion]:
        raise PublicContractError(
            "V5 payload selection diagnostics do not select exactly its champion"
        )
    if diagnostics[champion].get("gate_passed") is not True:
        raise PublicContractError("V5 payload champion did not pass its selection gate")


def validate_v5_comparison_sidecar(
    report: dict[str, Any],
    *,
    payload: dict[str, Any],
    payload_raw: bytes,
) -> None:
    """Validate the only derived comparison permitted in a public V5 package."""

    _require_exact_keys(report, V5_COMPARISON_TOP_LEVEL_KEYS, context="V5 comparison")
    _reject_raw_provider_material(report, path="comparison")
    _reject_comparison_raw_material(report)
    if (
        report.get("schema_version") != V5_COMPARISON_SCHEMA_VERSION
        or report.get("report_role") != "derived_only_diagnostic_comparison"
        or report.get("promotion_interpretation") != "prohibited"
        or report.get("provider_or_raw_feature_values_included") is not False
    ):
        raise PublicContractError("V5 comparison publication identity is invalid")

    expected_contract = {
        "actual_must_match": True,
        "evaluation_split_must_match": True,
        "exact_key_fields": ["origin_date", "target_date", "model_or_equivalent_model"],
        "metric_definitions": {
            "balanced_accuracy": "mean_recall_over_actual_classes_present",
            "brier": "mean_three_state_sum_squared_error",
            "log_loss": "mean_negative_log_actual_probability_clip_1e-9_then_renormalize",
        },
        "probability_columns": ["p_risk_on", "p_transition", "p_risk_off"],
        "splits_are_never_pooled": ["selection", "holdout"],
        "unmatched_keys_excluded": True,
    }
    if report.get("comparison_contract") != expected_contract:
        raise PublicContractError("V5 comparison matching contract is invalid")
    _validate_v5_comparison_inputs(report, payload, payload_raw)
    _validate_champion_selection_evidence(payload)
    _validate_fx_comparison(report.get("fx_ablation"), payload=payload)
    _validate_multiscale_comparison(
        report.get("v5_causal_multiscale_ensemble_vs_v5_markov"),
        payload=payload,
    )
    _validate_exact_markov_parity(report)


reject_raw_provider_material = _reject_raw_provider_material
require_object = _require_object
require_sha256 = _require_sha256


__all__ = [
    "PublicContractError",
    "V5_COMPARISON_SCHEMA_VERSION",
    "V5_RESULT_VERSION",
    "reject_raw_provider_material",
    "require_object",
    "require_sha256",
    "validate_v5_comparison_sidecar",
]
