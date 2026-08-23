#!/usr/bin/env python3
"""Create a fail-closed static dashboard package.

Only the three static application assets, one explicitly selected dashboard
payload, and the required derived-only V5 comparison sidecar are copied. Live
publication is limited to the user's personal,
non-commercial *derived result* snapshot: raw observations, databases, model
artifacts, credentials, and arbitrary files are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from regime_lab.schema import ContractError, validate_dashboard_payload
from regime_lab.config import project_root
from regime_lab.frozen_v4 import (
    FROZEN_V4_BASELINE,
    FROZEN_V4_INVENTORY_FILE_COUNT,
    FROZEN_V4_OOS_PREDICTIONS,
)
from regime_lab.path_safety import UnsafeMutablePath, confined_mutable_path
from regime_lab.contract_v5 import V5_PUBLICATION_STATUS


STATIC_ALLOWLIST = ("index.html", "styles.css", "app.js")
PAYLOAD_DESTINATION = "data/regime-results.json"
V5_COMPARISON_FILENAME = "v5-vs-v4-comparison.json"
V5_COMPARISON_DESTINATION = f"data/{V5_COMPARISON_FILENAME}"
MANIFEST_DESTINATION = "publication-manifest.json"
PUBLICATION_MODE_DEMO = "demo"
PUBLICATION_MODE_LIVE_DERIVED = "live-derived"
V4_RESULT_VERSION = "weekly-regime-result-v4"
V5_RESULT_VERSION = "weekly-regime-result-v5"
V5_COMPARISON_SCHEMA_VERSION = "regime-v5-v4-matched-comparison/1"
LIVE_SOURCE_LICENSES_BY_RESULT_VERSION = {
    V4_RESULT_VERSION: {
        "alpha_vantage": "private_noncommercial",
        "alfred": "user_confirmed_ml_storage_derived",
    },
    V5_RESULT_VERSION: {
        "alpha_vantage": "private_noncommercial",
        "alfred": "user_confirmed_ml_storage_derived",
        "frb_h10": "federal_reserve_board_public_domain_citation_requested",
    },
}
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


class PackagingError(RuntimeError):
    """Raised when an input is not safe to package for public demo use."""


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink():
        raise PackagingError(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise PackagingError(f"{label} is missing or is not a regular file: {path}")
    return path.read_bytes()


def _decode_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError("payload must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PackagingError("payload root must be a JSON object")
    return payload


def _decode_v5_comparison(raw: bytes) -> dict[str, Any]:
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError("V5 comparison must be valid UTF-8 JSON") from exc
    if not isinstance(report, dict):
        raise PackagingError("V5 comparison root must be a JSON object")
    return report


def validate_public_demo_payload(payload: dict[str, Any]) -> None:
    """Fail closed unless the payload is an explicitly synthetic demo."""

    # Rights metadata is necessary but not sufficient for publication.  The
    # packaged document must also satisfy the exact dashboard contract; this
    # prevents a synthetic-looking but malformed payload from becoming a
    # public page that only fails after browser load.
    try:
        validate_dashboard_payload(payload)
    except ContractError as exc:
        raise PackagingError(f"dashboard payload contract is invalid: {exc}") from exc

    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("mode") != "demo":
        mode = meta.get("mode") if isinstance(meta, dict) else None
        raise PackagingError(f"only meta.mode=demo may be packaged (got {mode!r})")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PackagingError("demo payload must contain a non-empty sources array")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise PackagingError(f"sources[{index}] must be an object")
        source_id = source.get("id")
        license_class = source.get("license_class")
        if not isinstance(source_id, str) or not source_id.startswith("synthetic_"):
            raise PackagingError(
                f"sources[{index}].id must identify a synthetic fixture"
            )
        if license_class != "synthetic_fixture":
            raise PackagingError(
                f"sources[{index}].license_class must be synthetic_fixture"
            )


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
                raise PackagingError(
                    f"live-derived payload contains forbidden provider material at {path}.{key}"
                )
            _reject_raw_provider_material(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_raw_provider_material(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and (
        value.startswith(("/Users/", "/private/", "file://")) or "\\Users\\" in value
    ):
        raise PackagingError(
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
                raise PackagingError(
                    f"V5 comparison contains row-level or raw material at {path}.{key}"
                )
            _reject_comparison_raw_material(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_comparison_raw_material(child, path=f"{path}[{index}]")


def _require_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackagingError(f"{context} must be an object")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str] | frozenset[str],
    *,
    context: str,
) -> None:
    if set(value) != set(expected):
        raise PackagingError(f"{context} fields are not exact")


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        raise PackagingError(f"{context} must be a lowercase SHA-256")
    return value


def _require_positive_integer(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise PackagingError(f"{context} must be a positive integer")
    return value


def _require_zero_integer(value: object, *, context: str) -> None:
    if type(value) is not int or value != 0:
        raise PackagingError(f"{context} must be integer zero")


def _require_zero_number(value: object, *, context: str) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value != 0:
        raise PackagingError(f"{context} must be exactly zero")


def _require_finite_number(value: object, *, context: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise PackagingError(f"{context} must be a finite number")
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
        raise PackagingError(f"{context} does not match the reviewed payload")


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
        raise PackagingError(f"{context}.path is invalid")
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
        raise PackagingError("V5 payload frozen V4 baseline is not the reviewed contract")

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
        raise PackagingError("V5 comparison frozen baseline id is invalid")
    if (
        type(frozen["verified_file_count"]) is not int
        or frozen["verified_file_count"] != FROZEN_V4_INVENTORY_FILE_COUNT
    ):
        raise PackagingError("V5 comparison frozen baseline file count is invalid")
    frozen_inventory = _validate_artifact_record(
        frozen["sha256sums"],
        context="V5 comparison.inputs.frozen_v4.sha256sums",
        expected_path="SHA256SUMS",
        require_row_count=False,
    )
    if frozen_inventory["sha256"] != payload_baseline["artifacts_inventory_sha256"]:
        raise PackagingError("V5 comparison frozen baseline inventory hash mismatch")
    frozen_oos = _validate_artifact_record(
        frozen["oos_predictions"],
        context="V5 comparison.inputs.frozen_v4.oos_predictions",
        expected_path="oos-predictions.csv",
        require_row_count=True,
    )
    if frozen_oos != dict(FROZEN_V4_OOS_PREDICTIONS):
        raise PackagingError("V5 comparison frozen OOS record is invalid")

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
        raise PackagingError("V5 comparison payload hash mismatch")
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
            raise PackagingError(
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
        raise PackagingError(f"{context} Markov metrics are not exactly equal")
    if type(frozen_metrics["n"]) is not int or frozen_metrics["n"] != count:
        raise PackagingError(f"{context} Markov metric count does not match common keys")
    _require_zero_integer(
        frozen_metrics["fallback_count"],
        context=f"{context}.fallback_count",
    )
    _require_zero_number(frozen_metrics["fallback_rate"], context=f"{context}.fallback_rate")
    for metric in ("balanced_accuracy", "brier", "log_loss"):
        metric_value = frozen_metrics[metric]
        if type(metric_value) not in {int, float} or not math.isfinite(metric_value):
            raise PackagingError(f"{context}.{metric} must be finite")

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
        raise PackagingError(f"{context} numeric probability parity is not exact")
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
        raise PackagingError(f"{context} token probability parity is not exact")
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
        raise PackagingError(f"{context} token probability hashes or mismatches differ")
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
        raise PackagingError("V5 comparison Markov join contract is invalid")

    selection_count = _validate_exact_markov_parity_split(
        comparison["primary_selection"],
        context="V5 comparison primary selection parity",
    )
    holdout_count = _validate_exact_markov_parity_split(
        comparison["post_selection_holdout"],
        context="V5 comparison post-selection holdout parity",
    )
    if selection_count + holdout_count != common_count:
        raise PackagingError("V5 comparison split counts do not equal common keys")


def _index_named_rows(
    value: object,
    *,
    name_field: str,
    context: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise PackagingError(f"{context} must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = _require_object(raw, context=f"{context}[{index}]")
        name = row.get(name_field)
        if not isinstance(name, str) or not name or name in rows:
            raise PackagingError(f"{context} has an invalid or duplicate {name_field}")
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
        raise PackagingError("V5 comparison FX publication identity is invalid")
    if (
        ablation.get("promotion_allowed") is not False
        or ablation.get("core_champion_promoted") is not False
    ):
        raise PackagingError("V5 comparison FX promotion must remain disabled")

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
        raise PackagingError("V5 comparison FX common origins do not match payload")
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
        raise PackagingError("V5 comparison FX variant order does not match payload")
    control = payload_metrics.get("v4_control")
    if control is None:
        raise PackagingError("V5 comparison FX control is missing")
    gate = _require_object(ablation.get("gate"), context="V5 payload FX gate")
    if gate.get("passed_variants") != []:
        raise PackagingError("V5 comparison FX gate must not pass a variant")
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
                raise PackagingError(
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
                raise PackagingError(f"{row_context} has no payload gate comparison")
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
        raise PackagingError(f"{context}.fallback_count is invalid")
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
        raise PackagingError(f"{context} metric counts do not match common keys")

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
            raise PackagingError(f"{context} payload model row is missing: {payload_name}")
        if payload_row.get("n_predictions") != count:
            raise PackagingError(f"{context} payload count mismatch: {payload_name}")
        if payload_row.get("fallback_count") != metric_row["fallback_count"]:
            raise PackagingError(f"{context} payload fallback mismatch: {payload_name}")
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
        raise PackagingError("V5 comparison Multiscale join contract is invalid")

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
        raise PackagingError("V5 comparison Multiscale split counts are inconsistent")

    gate = _require_object(
        comparison["selection_gate_crosscheck"],
        context=f"{context}.selection_gate_crosscheck",
    )
    _require_exact_keys(
        gate,
        {"artifact_role", "models", "pairwise_gate_against_markov"},
        context=f"{context}.selection_gate_crosscheck",
    )
    if (
        gate.get("artifact_role") != "selection_only_existing_champion_gate"
        or gate.get("pairwise_gate_against_markov") is not False
    ):
        raise PackagingError("V5 comparison Multiscale gate identity is invalid")
    gate_models = _require_object(
        gate.get("models"),
        context=f"{context}.selection_gate_crosscheck.models",
    )
    _require_exact_keys(
        gate_models,
        {"causal_multiscale_ensemble", "markov"},
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
    for name in ("causal_multiscale_ensemble", "markov"):
        row_context = f"{context}.selection_gate_crosscheck.models.{name}"
        row = _require_object(gate_models[name], context=row_context)
        _require_exact_keys(row, gate_fields, context=row_context)
        payload_row = diagnostics.get(name)
        if payload_row is None or row.get("matched_metric_crosscheck") is not True:
            raise PackagingError(f"{row_context} is not bound to selection diagnostics")
        for field in (
            "fallback_count",
            "gate_passed",
            "gate_reason",
            "n_predictions",
            "reference_model",
            "selected",
        ):
            if row.get(field) != payload_row.get(field):
                raise PackagingError(f"{row_context}.{field} does not match payload")
        for field in ("brier", "log_loss"):
            _require_close_number(
                row[field],
                payload_row.get(field),
                context=f"{row_context}.{field}",
            )


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
        raise PackagingError("V5 comparison publication identity is invalid")

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
        raise PackagingError("V5 comparison matching contract is invalid")
    _validate_v5_comparison_inputs(report, payload, payload_raw)
    _validate_fx_comparison(report.get("fx_ablation"), payload=payload)
    _validate_multiscale_comparison(
        report.get("v5_causal_multiscale_ensemble_vs_v5_markov"),
        payload=payload,
    )
    _validate_exact_markov_parity(report)


def validate_public_live_derived_payload(payload: dict[str, Any]) -> None:
    """Validate a personal, non-commercial snapshot containing derived results only."""

    _reject_raw_provider_material(payload)
    try:
        validate_dashboard_payload(payload)
    except ContractError as exc:
        raise PackagingError(f"dashboard payload contract is invalid: {exc}") from exc

    meta = payload.get("meta")
    if not isinstance(meta, dict) or meta.get("mode") != "live":
        mode = meta.get("mode") if isinstance(meta, dict) else None
        raise PackagingError(f"live-derived publication requires meta.mode=live (got {mode!r})")
    result_version = meta.get("result_version")
    source_licenses = LIVE_SOURCE_LICENSES_BY_RESULT_VERSION.get(result_version)
    if source_licenses is None:
        raise PackagingError(
            "live-derived publication requires a reviewed V4 or V5 result"
        )
    if (
        result_version == V5_RESULT_VERSION
        and meta.get("publication_status") != V5_PUBLICATION_STATUS
    ):
        raise PackagingError(
            "live-derived V5 publication requires publication_status=reviewed_publication"
        )
    if result_version == V5_RESULT_VERSION:
        _validate_reviewed_candidate_hash(payload)

    weekly = payload.get("weekly")
    if not isinstance(weekly, list) or len(weekly) < 52:
        count = len(weekly) if isinstance(weekly, list) else 0
        raise PackagingError(
            f"live-derived publication requires at least 52 weekly results (got {count})"
        )

    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise PackagingError("live-derived payload must contain sources")
    by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise PackagingError(f"sources[{index}] must contain an id")
        source_id = source["id"]
        if source_id in by_id:
            raise PackagingError(f"duplicate live-derived source: {source_id}")
        by_id[source_id] = source
    if set(by_id) != set(source_licenses):
        raise PackagingError(
            f"live-derived sources do not match the reviewed {result_version} contract"
        )
    for source_id, expected_license in source_licenses.items():
        actual = by_id[source_id].get("license_class")
        if actual != expected_license:
            raise PackagingError(
                f"{source_id}.license_class must be {expected_license}"
            )


def validate_public_payload(
    payload: dict[str, Any],
    *,
    publication_mode: str,
    rights_acknowledged: bool,
) -> None:
    if publication_mode == PUBLICATION_MODE_DEMO:
        validate_public_demo_payload(payload)
        return
    if publication_mode == PUBLICATION_MODE_LIVE_DERIVED:
        if rights_acknowledged is not True:
            raise PackagingError(
                "live-derived publication requires explicit personal/non-commercial rights acknowledgement"
            )
        validate_public_live_derived_payload(payload)
        return
    raise PackagingError(f"unsupported publication mode: {publication_mode!r}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_reviewed_candidate_hash(payload: dict[str, Any]) -> None:
    meta = _require_object(payload.get("meta"), context="V5 payload.meta")
    review = _require_object(
        meta.get("publication_review"),
        context="V5 payload.meta.publication_review",
    )
    expected = _require_sha256(
        review.get("reviewed_candidate_sha256"),
        context="V5 payload.meta.publication_review.reviewed_candidate_sha256",
    )
    candidate_meta = dict(meta)
    candidate_meta.pop("publication_status", None)
    candidate_meta.pop("publication_review", None)
    candidate = dict(payload)
    candidate["meta"] = candidate_meta
    candidate_raw = (
        json.dumps(
            candidate,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")
    if _sha256(candidate_raw) != expected:
        raise PackagingError(
            "V5 publication review candidate hash does not match reconstructed bytes"
        )


def _write_public_file(root: Path, relative_path: str, value: bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
    target.chmod(0o644)


def package_public_dashboard(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
    publication_mode: str = PUBLICATION_MODE_DEMO,
    rights_acknowledged: bool = False,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a new static directory from an explicit, minimal allowlist."""

    web_root = Path(web_root)
    payload_path = Path(payload_path)
    output_directory = confined_mutable_path(
        output_directory,
        project_directory=project_root(),
        label="public dashboard output",
    )

    if output_directory.exists() or output_directory.is_symlink():
        raise PackagingError(
            f"output already exists; refusing to overwrite it: {output_directory}"
        )

    files: dict[str, bytes] = {}
    for relative_path in STATIC_ALLOWLIST:
        files[relative_path] = _read_regular_file(
            web_root / relative_path,
            label=f"allowlisted web asset {relative_path}",
        )

    payload_raw = _read_regular_file(payload_path, label="dashboard payload")
    payload = _decode_payload(payload_raw)
    validate_public_payload(
        payload,
        publication_mode=publication_mode,
        rights_acknowledged=rights_acknowledged,
    )
    files[PAYLOAD_DESTINATION] = payload_raw

    result_version = payload.get("meta", {}).get("result_version")
    if result_version == V5_RESULT_VERSION:
        resolved_comparison_path = (
            Path(comparison_path)
            if comparison_path is not None
            else payload_path.with_name(V5_COMPARISON_FILENAME)
        )
        comparison_raw = _read_regular_file(
            resolved_comparison_path,
            label="reviewed V5/V4 comparison sidecar",
        )
        comparison = _decode_v5_comparison(comparison_raw)
        validate_v5_comparison_sidecar(
            comparison,
            payload=payload,
            payload_raw=payload_raw,
        )
        files[V5_COMPARISON_DESTINATION] = comparison_raw

    is_live_derived = publication_mode == PUBLICATION_MODE_LIVE_DERIVED
    source_ids = sorted(
        source["id"]
        for source in payload.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "package_kind": (
            "personal_noncommercial_live_derived"
            if is_live_derived
            else "synthetic_demo_only"
        ),
        "payload_mode": "live" if is_live_derived else "demo",
        "publication_scope": (
            "personal_noncommercial_derived_results"
            if is_live_derived
            else "synthetic_fixture"
        ),
        "contains_raw_observations": False,
        "source_ids": source_ids,
        "payload_data_as_of": payload["meta"].get("data_as_of"),
        "files": {
            path: {"bytes": len(value), "sha256": _sha256(value)}
            for path, value in sorted(files.items())
        },
    }
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    staging.chmod(0o755)
    try:
        for relative_path, value in files.items():
            _write_public_file(staging, relative_path, value)
        _write_public_file(staging, MANIFEST_DESTINATION, manifest_raw)
        if output_directory.exists() or output_directory.is_symlink():
            raise PackagingError(
                f"output appeared during packaging; refusing overwrite: {output_directory}"
            )
        os.replace(staging, output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def package_public_demo(
    *,
    web_root: str | Path,
    payload_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Backward-compatible synthetic demo wrapper."""

    return package_public_dashboard(
        web_root=web_root,
        payload_path=payload_path,
        output_directory=output_directory,
        publication_mode=PUBLICATION_MODE_DEMO,
        rights_acknowledged=False,
        comparison_path=None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an allowlisted Regime dashboard for static publication"
    )
    parser.add_argument("--web-root", type=Path, default=Path("web"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/public-dashboard")
    )
    parser.add_argument(
        "--publication-mode",
        choices=(PUBLICATION_MODE_DEMO, PUBLICATION_MODE_LIVE_DERIVED),
        default=PUBLICATION_MODE_DEMO,
    )
    parser.add_argument(
        "--acknowledge-personal-noncommercial-publication",
        action="store_true",
        help="Required for live-derived publication; never expands provider rights",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help=(
            "Required reviewed derived-only V5/V4 comparison sidecar for a V5 "
            "payload; defaults to the payload directory"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = package_public_dashboard(
            web_root=args.web_root,
            payload_path=args.payload,
            output_directory=args.output,
            publication_mode=args.publication_mode,
            rights_acknowledged=args.acknowledge_personal_noncommercial_publication,
            comparison_path=args.comparison,
        )
    except (PackagingError, UnsafeMutablePath, OSError) as exc:
        print(f"public dashboard package refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "package_kind": manifest["package_kind"],
                "files": sorted(
                    [*manifest["files"], MANIFEST_DESTINATION]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
