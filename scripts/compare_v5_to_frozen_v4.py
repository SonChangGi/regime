#!/usr/bin/env python3
"""Strict, derived-only comparison of V5 with the reviewed frozen V4 baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from regime_lab.analysis.models import model_complexity_rank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V4_ARTIFACTS = PROJECT_ROOT / "artifacts/baselines/v4-20260821"
FROZEN_V4_INVENTORY_SHA256 = (
    "3b0ffe79dea816b2a47c22ecba7eebb9b8fa8f4e9e2bb4ccba30f982d69c7613"
)
FROZEN_V4_INVENTORY_FILE_COUNT = 23
STATE_ORDER = ("risk_on", "transition", "risk_off")
PROBABILITY_COLUMNS = tuple(f"p_{state}" for state in STATE_ORDER)
SPLIT_ORDER = ("selection", "holdout")
SPLIT_OUTPUT_KEYS = {
    "selection": "primary_selection",
    "holdout": "post_selection_holdout",
}
FX_VARIANTS = (
    "v4_control",
    "v4_plus_broad_index",
    "v4_plus_bilateral_panel",
    "v4_plus_all_fx",
)
SELECTION_BASELINE_MODELS = ("majority", "persistence", "markov")
SELECTION_BOOTSTRAP_BLOCK_WEEKS = 13
SELECTION_BOOTSTRAP_RESAMPLES = 1_999
SELECTION_BOOTSTRAP_SEED = 17
SELECTION_ALPHA = 0.05
SELECTION_BRIER_TOLERANCE = 0.01
SELECTION_SIMPLICITY_TOLERANCE = 0.01
CORE_ARTIFACT_PATHS = {
    "oos_predictions": "oos-predictions.csv",
    "selection_diagnostics": "selection-diagnostics.csv",
}
FX_ABLATION_OOS_PATH = "fx-ablation-oos.csv"
FX_ABLATION_OOS_COLUMNS = (
    "origin_date",
    "target_date",
    "variant",
    "evaluation_split",
    "current_state",
    "actual",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "train_size",
    "gap",
    "last_train_target",
    "purged_origin_count",
    "fallback",
    "fallback_reason",
    "common_origins_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INVENTORY_LINE = re.compile(
    r"(?P<sha256>[0-9a-f]{64})  (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
)


class ComparisonError(RuntimeError):
    """The requested comparison is not exactly matched and must not be used."""


@dataclass(frozen=True, slots=True)
class PredictionRow:
    origin_date: str
    target_date: str
    model: str
    evaluation_split: str
    actual: str
    probabilities: tuple[float, float, float]
    probability_tokens: tuple[str, str, str]
    fallback: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.origin_date, self.target_date, self.model)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _file_sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"missing/non-regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(path: Path, *, context: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing/non-regular file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{context} is not valid UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def _resolve_v5_payload(
    artifacts: Path,
    supplied: Path | None,
) -> Path:
    if supplied is not None:
        resolved = supplied.resolve()
        _require(
            resolved.is_file() and not resolved.is_symlink(),
            "supplied V5 payload is missing or invalid",
        )
        return resolved
    candidates = [artifacts / "regime-results.json"]
    if artifacts.name.endswith("-artifacts"):
        candidates.append(
            artifacts.parent
            / artifacts.name.removesuffix("-artifacts")
            / "regime-results.json"
        )
    if artifacts.name == "artifacts":
        candidates.append(artifacts.parent / "regime-results.json")
    existing = [
        candidate.resolve()
        for candidate in candidates
        if candidate.is_file() and not candidate.is_symlink()
    ]
    _require(existing, "V5 payload could not be resolved from the artifacts directory")
    hashes = {_file_sha256(path) for path in existing}
    _require(len(hashes) == 1, "multiple conflicting V5 payload candidates were found")
    return existing[0]


def _parse_bool(value: str, *, context: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ComparisonError(f"{context} must be exactly True or False")


def _finite_float(value: Any, *, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{context} must be numeric") from exc
    _require(math.isfinite(parsed), f"{context} must be finite")
    return parsed


def _optional_finite_float(value: Any, *, context: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _finite_float(value, context=context)


def _aware_timestamp(value: str, *, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ComparisonError(f"{context} must be an ISO-8601 timestamp") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        f"{context} must include a UTC offset",
    )
    return parsed


def _iso_date(value: str, *, context: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ComparisonError(f"{context} must be an ISO date") from exc
    _require(parsed.isoformat() == value, f"{context} must use canonical YYYY-MM-DD")
    return parsed


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    _require(not isinstance(value, bool), f"{context} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{context} must be an integer") from exc
    _require(str(parsed) == str(value) or isinstance(value, int), f"{context} must be an integer")
    _require(parsed >= minimum, f"{context} must be at least {minimum}")
    return parsed


def _read_predictions(path: Path, *, context: str) -> tuple[list[PredictionRow], int]:
    required = {
        "origin_date",
        "target_date",
        "model",
        "evaluation_split",
        "actual",
        "fallback",
        *PROBABILITY_COLUMNS,
    }
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ComparisonError(f"cannot read {context}") from exc
    with stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None, f"{context} has no header")
        _require(required.issubset(reader.fieldnames), f"{context} schema is incomplete")
        rows: list[PredictionRow] = []
        seen: set[tuple[str, str, str]] = set()
        for line_number, raw in enumerate(reader, start=2):
            row_context = f"{context} line {line_number}"
            origin = str(raw["origin_date"])
            target = str(raw["target_date"])
            model = str(raw["model"])
            split = str(raw["evaluation_split"])
            actual = str(raw["actual"])
            _require(origin != "" and target != "" and model != "", f"{row_context} key is empty")
            _require(
                _aware_timestamp(origin, context=f"{row_context}.origin_date")
                < _aware_timestamp(target, context=f"{row_context}.target_date"),
                f"{row_context} origin/target order is invalid",
            )
            _require(split in SPLIT_ORDER, f"{row_context} evaluation_split is invalid")
            _require(actual in STATE_ORDER, f"{row_context} actual state is invalid")
            tokens = tuple(str(raw[column]) for column in PROBABILITY_COLUMNS)
            probabilities = tuple(
                _finite_float(token, context=f"{row_context}.{column}")
                for token, column in zip(tokens, PROBABILITY_COLUMNS, strict=True)
            )
            _require(all(0.0 <= value <= 1.0 for value in probabilities), f"{row_context} probability is outside [0,1]")
            _require(
                math.isclose(sum(probabilities), 1.0, abs_tol=1e-9, rel_tol=0.0),
                f"{row_context} probabilities do not sum to one",
            )
            row = PredictionRow(
                origin_date=origin,
                target_date=target,
                model=model,
                evaluation_split=split,
                actual=actual,
                probabilities=probabilities,  # type: ignore[arg-type]
                probability_tokens=tokens,  # type: ignore[arg-type]
                fallback=_parse_bool(raw["fallback"], context=f"{row_context}.fallback"),
            )
            _require(row.key not in seen, f"{context} has duplicate key {row.key}")
            seen.add(row.key)
            rows.append(row)
    _require(rows, f"{context} is empty")
    return rows, len(rows)


def _verify_frozen_v4(directory: Path) -> dict[str, Any]:
    _require(directory.is_dir() and not directory.is_symlink(), "frozen V4 directory is missing or invalid")
    inventory_path = directory / "SHA256SUMS"
    inventory_bytes = inventory_path.read_bytes() if inventory_path.is_file() else b""
    _require(inventory_bytes, "frozen V4 SHA256SUMS is missing")
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    _require(
        inventory_sha256 == FROZEN_V4_INVENTORY_SHA256,
        "frozen V4 SHA256SUMS does not match the reviewed inventory",
    )
    try:
        inventory_text = inventory_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ComparisonError("frozen V4 SHA256SUMS must be ASCII") from exc
    _require(inventory_text.endswith("\n"), "frozen V4 SHA256SUMS must end with a newline")
    entries: dict[str, str] = {}
    for line in inventory_text.splitlines():
        match = _INVENTORY_LINE.fullmatch(line)
        _require(match is not None, "frozen V4 SHA256SUMS has an invalid row")
        assert match is not None
        name = match.group("name")
        _require(name not in entries and name != "SHA256SUMS", "frozen V4 SHA256SUMS has duplicate entries")
        entries[name] = match.group("sha256")
    canonical = "".join(f"{entries[name]}  {name}\n" for name in sorted(entries)).encode("ascii")
    _require(canonical == inventory_bytes, "frozen V4 SHA256SUMS is not canonical")
    _require(
        len(entries) == FROZEN_V4_INVENTORY_FILE_COUNT,
        "frozen V4 inventory file count does not match",
    )
    actual_names = {path.name for path in directory.iterdir()}
    _require(actual_names == {*entries, "SHA256SUMS"}, "frozen V4 file set does not match the inventory")
    for name, expected in entries.items():
        _require(_file_sha256(directory / name) == expected, f"frozen V4 artifact hash mismatch: {name}")
    _require("oos-predictions.csv" in entries, "frozen V4 OOS predictions are not inventoried")
    return {
        "inventory_sha256": inventory_sha256,
        "verified_file_count": len(entries),
        "oos_predictions_sha256": entries["oos-predictions.csv"],
    }


def _validate_v5_core_binding(
    payload: Mapping[str, Any],
    artifacts: Path,
    row_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    model = payload.get("model")
    _require(isinstance(model, Mapping), "V5 payload.model is missing")
    manifest = model.get("core_artifacts")
    _require(isinstance(manifest, Mapping), "V5 payload.model.core_artifacts is missing")
    output: dict[str, dict[str, Any]] = {}
    for key, expected_path in CORE_ARTIFACT_PATHS.items():
        raw = manifest.get(key)
        _require(isinstance(raw, Mapping), f"V5 core artifact manifest is missing {key}")
        _require(raw.get("path") == expected_path, f"V5 core artifact path mismatch: {key}")
        supplied_hash = raw.get("sha256")
        _require(isinstance(supplied_hash, str) and _SHA256.fullmatch(supplied_hash) is not None, f"V5 core artifact SHA-256 is invalid: {key}")
        supplied_rows = _integer(raw.get("row_count"), context=f"V5 core artifact {key}.row_count", minimum=1)
        _require(supplied_rows == row_counts[key], f"V5 core artifact row count mismatch: {key}")
        actual_hash = _file_sha256(artifacts / expected_path)
        _require(actual_hash == supplied_hash, f"V5 core artifact hash mismatch: {key}")
        output[key] = {
            "path": expected_path,
            "row_count": supplied_rows,
            "sha256": actual_hash,
        }
    return output


def _metrics(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    _require(bool(rows), "cannot compute metrics without matched rows")
    log_losses: list[float] = []
    brier_rows: list[float] = []
    recalls: dict[str, list[bool]] = {state: [] for state in STATE_ORDER}
    fallback_count = 0
    for row in rows:
        clipped = [min(1.0, max(1e-9, value)) for value in row.probabilities]
        total = sum(clipped)
        probability = [value / total for value in clipped]
        actual_position = STATE_ORDER.index(row.actual)
        log_losses.append(-math.log(probability[actual_position]))
        brier_rows.append(
            sum(
                (value - (1.0 if index == actual_position else 0.0)) ** 2
                for index, value in enumerate(probability)
            )
        )
        predicted = STATE_ORDER[max(range(len(STATE_ORDER)), key=probability.__getitem__)]
        recalls[row.actual].append(predicted == row.actual)
        fallback_count += int(row.fallback)
    present_recalls = [
        sum(values) / len(values) for values in recalls.values() if values
    ]
    return {
        "n": len(rows),
        "log_loss": sum(log_losses) / len(log_losses),
        "brier": sum(brier_rows) / len(brier_rows),
        "balanced_accuracy": sum(present_recalls) / len(present_recalls),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(rows),
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        field: float(left[field]) - float(right[field])
        for field in ("log_loss", "brier", "balanced_accuracy", "fallback_rate")
    }


def _key_summary(keys: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    canonical = [list(key) for key in sorted(keys)]
    return {"count": len(canonical), "sha256": _canonical_sha256(canonical)}


def _parity(
    left: Sequence[PredictionRow],
    right: Sequence[PredictionRow],
) -> dict[str, Any]:
    _require(len(left) == len(right), "parity inputs are not paired")
    token_mismatch_rows = 0
    token_mismatch_values = 0
    numeric_mismatch_rows = 0
    numeric_mismatch_values = 0
    maximum_difference = 0.0
    left_stream: list[list[str]] = []
    right_stream: list[list[str]] = []
    for left_row, right_row in zip(left, right, strict=True):
        left_stream.append([*left_row.key, *left_row.probability_tokens])
        right_stream.append([*right_row.key, *right_row.probability_tokens])
        token_differences = [a.encode("utf-8") != b.encode("utf-8") for a, b in zip(left_row.probability_tokens, right_row.probability_tokens, strict=True)]
        numeric_differences = [a != b for a, b in zip(left_row.probabilities, right_row.probabilities, strict=True)]
        token_mismatch_rows += int(any(token_differences))
        token_mismatch_values += sum(token_differences)
        numeric_mismatch_rows += int(any(numeric_differences))
        numeric_mismatch_values += sum(numeric_differences)
        maximum_difference = max(
            maximum_difference,
            *(abs(a - b) for a, b in zip(left_row.probabilities, right_row.probabilities, strict=True)),
        )
    return {
        "probability_token_bytes": {
            "exact_parity": token_mismatch_rows == 0,
            "mismatch_rows": token_mismatch_rows,
            "mismatch_values": token_mismatch_values,
            "left_sha256": _canonical_sha256(left_stream),
            "right_sha256": _canonical_sha256(right_stream),
        },
        "probability_numeric": {
            "exact_float_parity": numeric_mismatch_rows == 0,
            "mismatch_rows": numeric_mismatch_rows,
            "mismatch_values": numeric_mismatch_values,
            "maximum_absolute_difference": maximum_difference,
        },
    }


def _same_model_pairs(
    left_rows: Sequence[PredictionRow],
    right_rows: Sequence[PredictionRow],
    *,
    model: str,
    context: str,
) -> list[tuple[PredictionRow, PredictionRow]]:
    left = {row.key: row for row in left_rows if row.model == model}
    right = {row.key: row for row in right_rows if row.model == model}
    _require(left and right, f"{context} requires {model} in both inputs")
    common = sorted(set(left).intersection(right))
    _require(common, f"{context} has no exact common keys")
    pairs = [(left[key], right[key]) for key in common]
    for left_row, right_row in pairs:
        _require(left_row.actual == right_row.actual, f"{context} actual mismatch at {left_row.key}")
        _require(
            left_row.evaluation_split == right_row.evaluation_split,
            f"{context} evaluation_split mismatch at {left_row.key}",
        )
    return pairs


def _equivalent_model_pairs(
    rows: Sequence[PredictionRow],
    *,
    left_model: str,
    right_model: str,
) -> list[tuple[PredictionRow, PredictionRow]]:
    left = {(row.origin_date, row.target_date): row for row in rows if row.model == left_model}
    right = {(row.origin_date, row.target_date): row for row in rows if row.model == right_model}
    _require(left and right, f"V5 pair requires {left_model} and {right_model}")
    common = sorted(set(left).intersection(right))
    _require(common, "V5 multiscale-versus-markov pair has no exact common origins")
    pairs = [(left[key], right[key]) for key in common]
    for left_row, right_row in pairs:
        key = (left_row.origin_date, left_row.target_date)
        _require(left_row.actual == right_row.actual, f"V5 model-pair actual mismatch at {key}")
        _require(left_row.evaluation_split == right_row.evaluation_split, f"V5 model-pair evaluation_split mismatch at {key}")
    return pairs


def _split_comparison(
    pairs: Sequence[tuple[PredictionRow, PredictionRow]],
    *,
    left_label: str,
    right_label: str,
    key_model: str,
    include_parity: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        selected = [pair for pair in pairs if pair[0].evaluation_split == split]
        _require(selected, f"comparison has no exact common {split} keys")
        left = [pair[0] for pair in selected]
        right = [pair[1] for pair in selected]
        keys = [
            (row.origin_date, row.target_date, key_model)
            for row in left
        ]
        left_metrics = _metrics(left)
        right_metrics = _metrics(right)
        split_output: dict[str, Any] = {
            "common_keys": _key_summary(keys),
            "metrics": {left_label: left_metrics, right_label: right_metrics},
            "delta_left_minus_right": _metric_delta(left_metrics, right_metrics),
        }
        if include_parity:
            split_output["probability_parity"] = _parity(left, right)
        output[SPLIT_OUTPUT_KEYS[split]] = split_output
    return output


def _read_selection_diagnostics(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    required = {
        "model",
        "reference_model",
        "is_reference",
        "selected",
        "gate_passed",
        "gate_reason",
        "log_loss",
        "reference_log_loss",
        "absolute_log_loss_improvement",
        "brier",
        "reference_brier",
        "brier_difference",
        "fallback_count",
        "raw_p_value",
        "holm_adjusted_p_value",
        "n_predictions",
        "bootstrap_block_weeks",
        "bootstrap_effective_block_weeks",
        "bootstrap_resamples",
        "bootstrap_seed",
        "alpha",
        "minimum_log_loss_improvement",
        "brier_tolerance",
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "V5 selection diagnostics schema is incomplete")
        output: dict[str, dict[str, Any]] = {}
        row_count = 0
        for line_number, raw in enumerate(reader, start=2):
            row_count += 1
            model = str(raw["model"])
            _require(model and model not in output, f"V5 selection diagnostics has duplicate model at line {line_number}")
            row = {
                "model": model,
                "reference_model": str(raw["reference_model"]),
                "is_reference": _parse_bool(raw["is_reference"], context=f"selection diagnostics line {line_number}.is_reference"),
                "selected": _parse_bool(raw["selected"], context=f"selection diagnostics line {line_number}.selected"),
                "gate_passed": _parse_bool(raw["gate_passed"], context=f"selection diagnostics line {line_number}.gate_passed"),
                "gate_reason": str(raw["gate_reason"]),
                "log_loss": _finite_float(raw["log_loss"], context=f"selection diagnostics line {line_number}.log_loss"),
                "reference_log_loss": _finite_float(raw["reference_log_loss"], context=f"selection diagnostics line {line_number}.reference_log_loss"),
                "absolute_log_loss_improvement": _finite_float(raw["absolute_log_loss_improvement"], context=f"selection diagnostics line {line_number}.absolute_log_loss_improvement"),
                "brier": _finite_float(raw["brier"], context=f"selection diagnostics line {line_number}.brier"),
                "reference_brier": _finite_float(raw["reference_brier"], context=f"selection diagnostics line {line_number}.reference_brier"),
                "brier_difference": _finite_float(raw["brier_difference"], context=f"selection diagnostics line {line_number}.brier_difference"),
                "fallback_count": _integer(raw["fallback_count"], context=f"selection diagnostics line {line_number}.fallback_count"),
                "raw_p_value": _optional_finite_float(raw["raw_p_value"], context=f"selection diagnostics line {line_number}.raw_p_value"),
                "holm_adjusted_p_value": _optional_finite_float(raw["holm_adjusted_p_value"], context=f"selection diagnostics line {line_number}.holm_adjusted_p_value"),
                "n_predictions": _integer(raw["n_predictions"], context=f"selection diagnostics line {line_number}.n_predictions", minimum=1),
                "bootstrap_block_weeks": _integer(raw["bootstrap_block_weeks"], context=f"selection diagnostics line {line_number}.bootstrap_block_weeks", minimum=1),
                "bootstrap_effective_block_weeks": _integer(raw["bootstrap_effective_block_weeks"], context=f"selection diagnostics line {line_number}.bootstrap_effective_block_weeks", minimum=1),
                "bootstrap_resamples": _integer(raw["bootstrap_resamples"], context=f"selection diagnostics line {line_number}.bootstrap_resamples", minimum=1),
                "bootstrap_seed": _integer(raw["bootstrap_seed"], context=f"selection diagnostics line {line_number}.bootstrap_seed"),
                "alpha": _finite_float(raw["alpha"], context=f"selection diagnostics line {line_number}.alpha"),
                "minimum_log_loss_improvement": _finite_float(raw["minimum_log_loss_improvement"], context=f"selection diagnostics line {line_number}.minimum_log_loss_improvement"),
                "brier_tolerance": _finite_float(raw["brier_tolerance"], context=f"selection diagnostics line {line_number}.brier_tolerance"),
            }
            _require(
                math.isclose(row["absolute_log_loss_improvement"], row["reference_log_loss"] - row["log_loss"], abs_tol=1e-12, rel_tol=0.0),
                f"V5 selection diagnostics improvement is inconsistent for {model}",
            )
            _require(
                math.isclose(row["brier_difference"], row["brier"] - row["reference_brier"], abs_tol=1e-12, rel_tol=0.0),
                f"V5 selection diagnostics Brier difference is inconsistent for {model}",
            )
            output[model] = row
    _require(row_count > 0, "V5 selection diagnostics is empty")
    return output, row_count


def _selection_calibration_error(rows: Sequence[PredictionRow]) -> float:
    _require(bool(rows), "selection calibration requires predictions")
    correct_by_bin: list[list[float]] = [[] for _ in range(10)]
    confidence_by_bin: list[list[float]] = [[] for _ in range(10)]
    for row in rows:
        probability = np.asarray(row.probabilities, dtype=float)
        probability = np.clip(probability, 1e-9, 1.0)
        probability /= probability.sum()
        predicted_position = int(np.argmax(probability))
        confidence = float(probability[predicted_position])
        # Match the production top-label ECE bins: [0,.1], then (a,b].
        bin_position = next(
            index
            for index in range(10)
            if (confidence >= index / 10.0 if index == 0 else confidence > index / 10.0)
            and confidence <= (index + 1) / 10.0
        )
        correct_by_bin[bin_position].append(
            float(STATE_ORDER[predicted_position] == row.actual)
        )
        confidence_by_bin[bin_position].append(confidence)
    total = len(rows)
    return sum(
        (len(correct) / total)
        * abs(sum(correct) / len(correct) - sum(confidence) / len(confidence))
        for correct, confidence in zip(
            correct_by_bin,
            confidence_by_bin,
            strict=True,
        )
        if correct
    )


def _holm_adjusted_pvalues(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    for rank, (model, pvalue) in enumerate(ordered):
        candidate = min(1.0, float(pvalue) * (len(ordered) - rank))
        running_maximum = max(running_maximum, candidate)
        adjusted[model] = running_maximum
    return adjusted


def _selection_bootstrap_pvalues(
    improvements: Mapping[str, np.ndarray],
    *,
    block_length: int,
    resamples: int,
    random_state: int,
) -> tuple[dict[str, float], int]:
    _require(bool(improvements), "selection family has no learned challengers")
    lengths = {len(values) for values in improvements.values()}
    _require(len(lengths) == 1, "selection paired loss lengths differ")
    observation_count = lengths.pop()
    _require(observation_count > 0, "selection family is empty")
    effective_block = min(block_length, max(1, observation_count // 2))
    blocks_per_sample = int(np.ceil(observation_count / effective_block))
    generator = np.random.default_rng(random_state)
    starts = generator.integers(
        0,
        observation_count,
        size=(resamples, blocks_per_sample),
    )
    offsets = np.arange(effective_block)
    indices = (starts[..., np.newaxis] + offsets) % observation_count
    indices = indices.reshape(resamples, -1)[:, :observation_count]
    output: dict[str, float] = {}
    for model, differential in improvements.items():
        _require(
            np.isfinite(differential).all(),
            f"selection paired losses are non-finite for {model}",
        )
        observed = float(differential.mean())
        null_means = (differential - observed)[indices].mean(axis=1)
        output[model] = float(
            (1 + np.count_nonzero(null_means >= observed)) / (resamples + 1)
        )
    return output, effective_block


def _validate_payload_selection_diagnostics(
    payload: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> None:
    model = payload.get("model")
    _require(isinstance(model, Mapping), "V5 payload.model is missing")
    supplied = model.get("selection_diagnostics")
    _require(isinstance(supplied, list) and supplied, "V5 payload selection diagnostics are missing")
    supplied_by_model: dict[str, Mapping[str, Any]] = {}
    for position, raw in enumerate(supplied):
        _require(isinstance(raw, Mapping), f"V5 payload selection diagnostics row {position} is invalid")
        name = raw.get("model")
        _require(isinstance(name, str) and name and name not in supplied_by_model, "V5 payload selection diagnostics model names are invalid")
        supplied_by_model[name] = raw
    _require(set(supplied_by_model) == set(diagnostics), "V5 payload and artifact selection families differ")
    exact_fields = {
        "reference_model",
        "selected",
        "gate_passed",
        "gate_reason",
        "fallback_count",
        "n_predictions",
        "bootstrap_block_weeks",
        "bootstrap_effective_block_weeks",
        "bootstrap_resamples",
        "bootstrap_seed",
    }
    numeric_fields = {
        "log_loss",
        "reference_log_loss",
        "absolute_log_loss_improvement",
        "brier",
        "reference_brier",
        "brier_difference",
        "raw_p_value",
        "holm_adjusted_p_value",
        "alpha",
        "minimum_log_loss_improvement",
        "brier_tolerance",
    }
    for name, artifact_row in diagnostics.items():
        payload_row = supplied_by_model[name]
        for field in exact_fields:
            _require(
                payload_row.get(field) == artifact_row[field],
                f"V5 payload selection diagnostics {field} mismatch for {name}",
            )
        for field in numeric_fields:
            expected = artifact_row[field]
            actual = payload_row.get(field)
            if expected is None:
                _require(actual is None, f"V5 payload selection diagnostics {field} mismatch for {name}")
                continue
            _require(
                type(actual) in {int, float}
                and math.isfinite(float(actual))
                and math.isclose(float(actual), float(expected), abs_tol=1e-8, rel_tol=0.0),
                f"V5 payload selection diagnostics {field} mismatch for {name}",
            )


def _validate_selection_family(
    diagnostics: Mapping[str, Mapping[str, Any]],
    rows: Sequence[PredictionRow],
    *,
    champion: str,
) -> None:
    selection_rows = [row for row in rows if row.evaluation_split == "selection"]
    models = set(diagnostics)
    _require(models == {row.model for row in selection_rows}, "selection diagnostics and OOS model families differ")
    references = {str(row["reference_model"]) for row in diagnostics.values()}
    _require(len(references) == 1, "selection diagnostics reference models differ")
    reference_model = next(iter(references))
    _require(reference_model in models and reference_model in SELECTION_BASELINE_MODELS, "selection reference model is invalid")

    ordered_by_model: dict[str, list[PredictionRow]] = {}
    reference_keys: list[tuple[str, str, str]] | None = None
    for model in sorted(models):
        ordered = sorted(
            (row for row in selection_rows if row.model == model),
            key=lambda row: row.target_date,
        )
        keys = [(row.origin_date, row.target_date, row.actual) for row in ordered]
        _require(len({row.target_date for row in ordered}) == len(ordered), f"selection target dates are duplicated for {model}")
        if reference_keys is None:
            reference_keys = keys
        else:
            _require(keys == reference_keys, f"selection origins/outcomes differ for {model}")
        ordered_by_model[model] = ordered
    assert reference_keys is not None

    metrics = {model: _metrics(model_rows) for model, model_rows in ordered_by_model.items()}
    baseline_models = [model for model in SELECTION_BASELINE_MODELS if model in models]
    _require(bool(baseline_models), "selection family has no probability baseline")
    expected_reference = min(
        baseline_models,
        key=lambda name: (
            float(metrics[name]["log_loss"]),
            _selection_calibration_error(ordered_by_model[name]),
            name,
        ),
    )
    _require(reference_model == expected_reference, "selection reference model was not independently reproduced")

    common_configuration = {
        field: {row[field] for row in diagnostics.values()}
        for field in (
            "bootstrap_block_weeks",
            "bootstrap_effective_block_weeks",
            "bootstrap_resamples",
            "bootstrap_seed",
            "alpha",
            "minimum_log_loss_improvement",
            "brier_tolerance",
        )
    }
    _require(all(len(values) == 1 for values in common_configuration.values()), "selection diagnostic configuration differs across models")
    block_length = int(next(iter(common_configuration["bootstrap_block_weeks"])))
    resamples = int(next(iter(common_configuration["bootstrap_resamples"])))
    seed = int(next(iter(common_configuration["bootstrap_seed"])))
    alpha = float(next(iter(common_configuration["alpha"])))
    minimum_improvement = float(next(iter(common_configuration["minimum_log_loss_improvement"])))
    brier_tolerance = float(next(iter(common_configuration["brier_tolerance"])))
    _require(
        (block_length, resamples, seed) == (
            SELECTION_BOOTSTRAP_BLOCK_WEEKS,
            SELECTION_BOOTSTRAP_RESAMPLES,
            SELECTION_BOOTSTRAP_SEED,
        ),
        "selection moving-block bootstrap configuration is invalid",
    )
    _require(math.isclose(alpha, SELECTION_ALPHA, abs_tol=1e-12, rel_tol=0.0), "selection alpha is invalid")
    _require(math.isclose(brier_tolerance, SELECTION_BRIER_TOLERANCE, abs_tol=1e-12, rel_tol=0.0), "selection Brier tolerance is invalid")
    _require(
        any(math.isclose(minimum_improvement, allowed, abs_tol=1e-12, rel_tol=0.0) for allowed in (0.01, 0.05)),
        "selection minimum log-loss improvement is invalid",
    )

    loss_by_model: dict[str, np.ndarray] = {}
    positions = {state: index for index, state in enumerate(STATE_ORDER)}
    for model, model_rows in ordered_by_model.items():
        actual_probability = np.asarray(
            [row.probabilities[positions[row.actual]] for row in model_rows],
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            loss_by_model[model] = -np.log(actual_probability)
    challengers = [model for model in diagnostics if model not in SELECTION_BASELINE_MODELS]
    improvements = {
        model: loss_by_model[reference_model] - loss_by_model[model]
        for model in challengers
    }
    raw_pvalues, effective_block = _selection_bootstrap_pvalues(
        improvements,
        block_length=block_length,
        resamples=resamples,
        random_state=seed,
    )
    adjusted_pvalues = _holm_adjusted_pvalues(raw_pvalues)
    _require(
        common_configuration["bootstrap_effective_block_weeks"] == {effective_block},
        "selection effective bootstrap block was not independently reproduced",
    )

    reference_metrics = metrics[reference_model]
    passing: list[str] = []
    for model, diagnostic in diagnostics.items():
        model_metrics = metrics[model]
        improvement = float(reference_metrics["log_loss"]) - float(model_metrics["log_loss"])
        brier_difference = float(model_metrics["brier"]) - float(reference_metrics["brier"])
        for field, expected in (
            ("log_loss", model_metrics["log_loss"]),
            ("reference_log_loss", reference_metrics["log_loss"]),
            ("absolute_log_loss_improvement", improvement),
            ("brier", model_metrics["brier"]),
            ("reference_brier", reference_metrics["brier"]),
            ("brier_difference", brier_difference),
        ):
            _require(
                math.isclose(float(diagnostic[field]), float(expected), abs_tol=1e-12, rel_tol=0.0),
                f"selection diagnostics {field} mismatch for {model}",
            )
        _require(diagnostic["n_predictions"] == model_metrics["n"], f"selection diagnostics n_predictions mismatch for {model}")
        _require(diagnostic["fallback_count"] == model_metrics["fallback_count"], f"selection diagnostics fallback mismatch for {model}")
        _require(diagnostic["is_reference"] is (model == reference_model), f"selection diagnostics is_reference mismatch for {model}")

        if model in SELECTION_BASELINE_MODELS:
            _require(diagnostic["raw_p_value"] is None and diagnostic["holm_adjusted_p_value"] is None, f"selection baseline p-values must be empty for {model}")
        else:
            raw_p_value = diagnostic["raw_p_value"]
            adjusted_p_value = diagnostic["holm_adjusted_p_value"]
            _require(
                type(raw_p_value) in {int, float},
                f"selection diagnostics raw_p_value mismatch for {model}",
            )
            _require(
                type(adjusted_p_value) in {int, float},
                f"selection diagnostics holm_adjusted_p_value mismatch for {model}",
            )
            _require(
                math.isclose(float(raw_p_value), raw_pvalues[model], abs_tol=1e-12, rel_tol=0.0),
                f"selection diagnostics raw_p_value mismatch for {model}",
            )
            _require(
                math.isclose(float(adjusted_p_value), adjusted_pvalues[model], abs_tol=1e-12, rel_tol=0.0),
                f"selection diagnostics holm_adjusted_p_value mismatch for {model}",
            )

        failures: list[str] = []
        if model not in SELECTION_BASELINE_MODELS:
            if int(model_metrics["fallback_count"]) != 0:
                failures.append("fallback_present")
            if improvement + 1e-12 < minimum_improvement:
                failures.append("insufficient_log_loss_improvement")
            if adjusted_pvalues[model] > alpha:
                failures.append("holm_not_significant")
            if brier_difference > brier_tolerance + 1e-12:
                failures.append("brier_degradation")
            if not failures:
                passing.append(model)
        elif model != reference_model:
            failures.append("non_reference_baseline")
        expected_gate = model == reference_model or not failures
        expected_reason = "passed" if not failures else ";".join(failures)
        _require(diagnostic["gate_passed"] is expected_gate, f"selection diagnostics gate_passed mismatch for {model}")
        _require(diagnostic["gate_reason"] == expected_reason, f"selection diagnostics gate_reason mismatch for {model}")

    expected_champion = reference_model
    if passing:
        best_loss = min(float(metrics[model]["log_loss"]) for model in passing)
        near_best = [
            model
            for model in passing
            if float(metrics[model]["log_loss"])
            <= best_loss + SELECTION_SIMPLICITY_TOLERANCE
        ]
        expected_champion = min(
            near_best,
            key=lambda name: (
                model_complexity_rank(name),
                _selection_calibration_error(ordered_by_model[name]),
                float(metrics[name]["log_loss"]),
                name,
            ),
        )
    _require(champion == expected_champion, "declared champion was not independently reproduced from the selection family")
    _require(
        [name for name, row in diagnostics.items() if row["selected"] is True]
        == [expected_champion],
        "selection diagnostics selected flag was not independently reproduced",
    )


def _selection_gate_crosscheck(
    diagnostics: Mapping[str, Mapping[str, Any]],
    rows: Sequence[PredictionRow],
    *,
    champion: str,
) -> dict[str, Any]:
    champion_row = diagnostics.get(champion)
    _require(
        isinstance(champion_row, Mapping),
        "selection diagnostics is missing the declared champion",
    )
    champion_reference = champion_row.get("reference_model")
    _require(
        isinstance(champion_reference, str) and champion_reference,
        "declared champion reference model is invalid",
    )
    selected_models = [
        name for name, row in diagnostics.items() if row.get("selected") is True
    ]
    _require(
        selected_models == [champion],
        "selection diagnostics does not select exactly the declared champion",
    )

    model_names = sorted(
        {
            "causal_multiscale_ensemble",
            "markov",
            champion,
            champion_reference,
        }
    )
    output: dict[str, Any] = {}
    for model in model_names:
        row = diagnostics.get(model)
        _require(isinstance(row, Mapping), f"selection diagnostics is missing {model}")
        reference_model = row.get("reference_model")
        _require(
            isinstance(reference_model, str) and reference_model,
            f"selection diagnostics reference model is invalid for {model}",
        )
        reference_row = diagnostics.get(reference_model)
        _require(
            isinstance(reference_row, Mapping),
            f"selection diagnostics is missing reference {reference_model} for {model}",
        )
        pairs = _equivalent_model_pairs(
            rows,
            left_model=model,
            right_model=reference_model,
        )
        selected_pairs = [
            pair for pair in pairs if pair[0].evaluation_split == "selection"
        ]
        _require(
            selected_pairs,
            f"selection gate cross-check has no common selection rows for {model}",
        )
        computed = _metrics([pair[0] for pair in selected_pairs])
        computed_reference = _metrics([pair[1] for pair in selected_pairs])
        _require(row["n_predictions"] == computed["n"], f"selection diagnostics n_predictions mismatch for {model}")
        _require(row["fallback_count"] == computed["fallback_count"], f"selection diagnostics fallback mismatch for {model}")
        _require(
            reference_row["n_predictions"] == computed_reference["n"],
            f"selection diagnostics reference n_predictions mismatch for {model}",
        )
        _require(
            reference_row["fallback_count"] == computed_reference["fallback_count"],
            f"selection diagnostics reference fallback mismatch for {model}",
        )
        for metric in ("log_loss", "brier"):
            _require(
                math.isclose(float(row[metric]), float(computed[metric]), abs_tol=1e-12, rel_tol=0.0),
                f"selection diagnostics {metric} mismatch for {model}",
            )
            _require(
                math.isclose(
                    float(row[f"reference_{metric}"]),
                    float(computed_reference[metric]),
                    abs_tol=1e-12,
                    rel_tol=0.0,
                ),
                f"selection diagnostics reference {metric} mismatch for {model}",
            )
        output[model] = {
            "reference_model": reference_model,
            "selected": row["selected"],
            "gate_passed": row["gate_passed"],
            "gate_reason": row["gate_reason"],
            "n_predictions": row["n_predictions"],
            "fallback_count": row["fallback_count"],
            "log_loss": row["log_loss"],
            "brier": row["brier"],
            "matched_metric_crosscheck": True,
        }
    multiscale = diagnostics["causal_multiscale_ensemble"]
    return {
        "artifact_role": "selection_family_independently_recomputed",
        "multiscale_gate_against_selection_reference": bool(
            multiscale["gate_passed"]
        ),
        "models": output,
    }


def _read_fx_ablation_oos(
    path: Path,
) -> tuple[list[PredictionRow], int, str | None]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            tuple(reader.fieldnames or ()) == FX_ABLATION_OOS_COLUMNS,
            "FX ablation OOS sidecar columns/order are invalid",
        )
        rows: list[PredictionRow] = []
        metadata: dict[tuple[str, str, str], tuple[str, str, int, int, str, int]] = {}
        seen: set[tuple[str, str, str]] = set()
        origin_hashes: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            context = f"FX ablation OOS line {line_number}"
            origin = str(raw["origin_date"])
            target = str(raw["target_date"])
            variant = str(raw["variant"])
            current_state = str(raw["current_state"])
            actual = str(raw["actual"])
            _require(variant in FX_VARIANTS, f"{context}.variant is invalid")
            _require(raw["evaluation_split"] == "prospective_shadow", f"{context}.evaluation_split is invalid")
            _require(current_state in STATE_ORDER and actual in STATE_ORDER, f"{context} state is invalid")
            origin_date = _iso_date(origin, context=f"{context}.origin_date")
            target_date = _iso_date(target, context=f"{context}.target_date")
            _require(origin_date < target_date, f"{context} origin/target order is invalid")
            probabilities = tuple(
                _finite_float(raw[column], context=f"{context}.{column}")
                for column in PROBABILITY_COLUMNS
            )
            _require(all(0.0 <= value <= 1.0 for value in probabilities), f"{context} probability is outside [0,1]")
            _require(math.isclose(sum(probabilities), 1.0, abs_tol=1e-9, rel_tol=0.0), f"{context} probabilities do not sum to one")
            train_size = _integer(raw["train_size"], context=f"{context}.train_size", minimum=1)
            gap = _integer(raw["gap"], context=f"{context}.gap", minimum=1)
            purged = _integer(raw["purged_origin_count"], context=f"{context}.purged_origin_count", minimum=1)
            last_train_target = str(raw["last_train_target"])
            last_train_date = _iso_date(
                last_train_target,
                context=f"{context}.last_train_target",
            )
            _require(
                gap == 1 and last_train_date < origin_date,
                f"{context} purge contract is invalid",
            )
            fallback = _parse_bool(raw["fallback"], context=f"{context}.fallback")
            fallback_reason = str(raw["fallback_reason"])
            _require(fallback == bool(fallback_reason), f"{context} fallback reason is inconsistent")
            origin_hash = str(raw["common_origins_sha256"])
            _require(_SHA256.fullmatch(origin_hash) is not None, f"{context}.common_origins_sha256 is invalid")
            key = (origin, target, variant)
            _require(key not in seen, f"FX ablation OOS has duplicate key {key}")
            seen.add(key)
            origin_hashes.add(origin_hash)
            metadata[key] = (
                current_state,
                actual,
                train_size,
                gap,
                last_train_target,
                purged,
            )
            rows.append(
                PredictionRow(
                    origin_date=origin,
                    target_date=target,
                    model=variant,
                    evaluation_split="prospective_shadow",
                    actual=actual,
                    probabilities=probabilities,  # type: ignore[arg-type]
                    probability_tokens=tuple(str(raw[column]) for column in PROBABILITY_COLUMNS),  # type: ignore[arg-type]
                    fallback=fallback,
                )
            )
        if not rows:
            return [], 0, None
        _require(len(origin_hashes) == 1, "FX ablation OOS origin SHA-256 is inconsistent")

    pair_sets = {
        variant: {
            (row.origin_date, row.target_date)
            for row in rows
            if row.model == variant
        }
        for variant in FX_VARIANTS
    }
    first_pairs = pair_sets[FX_VARIANTS[0]]
    _require(first_pairs, "FX ablation OOS control rows are missing")
    _require(all(pairs == first_pairs for pairs in pair_sets.values()), "FX ablation variants do not share exact common origins")
    ordered_pairs = sorted(first_pairs)
    origin_sha256 = _canonical_sha256([list(pair) for pair in ordered_pairs])
    _require(origin_hashes == {origin_sha256}, "FX ablation OOS origin SHA-256 does not recompute")
    for origin, target in ordered_pairs:
        reference = metadata[(origin, target, FX_VARIANTS[0])]
        for variant in FX_VARIANTS[1:]:
            _require(metadata[(origin, target, variant)] == reference, f"FX ablation fold metadata mismatch at {(origin, target)}")
    return rows, len(rows), origin_sha256


def _fx_metrics(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    _require(bool(rows), "cannot compute FX metrics without rows")
    losses: list[float] = []
    brier_rows: list[float] = []
    correct = 0
    recalls: dict[str, list[bool]] = {state: [] for state in STATE_ORDER}
    fallback_count = 0
    for row in rows:
        actual_position = STATE_ORDER.index(row.actual)
        losses.append(-math.log(min(1.0, max(1e-9, row.probabilities[actual_position]))))
        brier_rows.append(
            sum(
                (value - (1.0 if index == actual_position else 0.0)) ** 2
                for index, value in enumerate(row.probabilities)
            )
        )
        predicted = STATE_ORDER[
            max(range(len(STATE_ORDER)), key=row.probabilities.__getitem__)
        ]
        is_correct = predicted == row.actual
        correct += int(is_correct)
        recalls[row.actual].append(is_correct)
        fallback_count += int(row.fallback)
    present_recalls = [sum(values) / len(values) for values in recalls.values() if values]
    return {
        "n_predictions": len(rows),
        "log_loss": sum(losses) / len(losses),
        "brier": sum(brier_rows) / len(brier_rows),
        "accuracy": correct / len(rows),
        "balanced_accuracy": sum(present_recalls) / len(present_recalls),
        "fallback_count": fallback_count,
    }


def _fx_summary(
    payload: Mapping[str, Any],
    artifacts: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model = payload.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("fx_ablation"), Mapping):
        return (
            {"comparison_status": "unavailable", "source_status": "missing", "reason": "fx_ablation_not_present"},
            None,
        )
    ablation = model["fx_ablation"]
    assert isinstance(ablation, Mapping)
    status = str(ablation.get("status", "missing"))
    sidecar_path = artifacts / FX_ABLATION_OOS_PATH
    research_manifest = model.get("research_artifacts")
    manifest_entry = (
        research_manifest.get("fx_ablation_oos")
        if isinstance(research_manifest, Mapping)
        else None
    )
    sidecar_exists = sidecar_path.is_file() and not sidecar_path.is_symlink()
    _require(
        sidecar_exists == isinstance(manifest_entry, Mapping),
        "FX ablation OOS sidecar/manifest presence mismatch",
    )
    if not sidecar_exists:
        return (
            {
                "comparison_status": "unavailable",
                "source_status": status,
                "reason": "fx_ablation_oos_not_present",
            },
            None,
        )
    _require(
        status in {"evaluated", "unavailable", "insufficient_history"},
        "FX ablation status is invalid",
    )
    assert isinstance(manifest_entry, Mapping)
    _require(manifest_entry.get("path") == FX_ABLATION_OOS_PATH, "FX ablation OOS manifest path is invalid")
    supplied_hash = manifest_entry.get("sha256")
    _require(isinstance(supplied_hash, str) and _SHA256.fullmatch(supplied_hash) is not None, "FX ablation OOS manifest SHA-256 is invalid")
    actual_hash = _file_sha256(sidecar_path)
    _require(actual_hash == supplied_hash, "FX ablation OOS manifest hash mismatch")
    sidecar_rows, row_count, recomputed_origin_hash = _read_fx_ablation_oos(sidecar_path)
    _require(
        _integer(
            manifest_entry.get("row_count"),
            context="FX ablation OOS manifest row_count",
            minimum=0 if status != "evaluated" else 1,
        )
        == row_count,
        "FX ablation OOS manifest row count mismatch",
    )
    input_manifest = {
        "path": FX_ABLATION_OOS_PATH,
        "row_count": row_count,
        "sha256": actual_hash,
    }
    if status != "evaluated":
        _require(
            status in {"unavailable", "insufficient_history"},
            "FX ablation status is invalid",
        )
        _require(row_count == 0, "non-evaluated FX ablation OOS sidecar must be empty")
        metrics = ablation.get("variant_metrics")
        _require(
            metrics in (None, []),
            "non-evaluated FX ablation must not contain variant metrics",
        )
        return (
            {
                "comparison_status": "unavailable",
                "source_status": status,
                "reason": ablation.get("status_reason") or f"fx_{status}",
            },
            input_manifest,
        )
    _require(row_count > 0, "evaluated FX ablation OOS sidecar is empty")
    assert recomputed_origin_hash is not None
    manifest = ablation.get("manifest")
    metrics = ablation.get("variant_metrics")
    origins = ablation.get("common_evaluation_origins")
    _require(isinstance(manifest, list) and isinstance(metrics, list), "evaluated FX manifest/leaderboard is missing")
    _require(isinstance(origins, Mapping), "evaluated FX common-origin manifest is missing")
    _require(len(manifest) == len(FX_VARIANTS) and len(metrics) == len(FX_VARIANTS), "evaluated FX variants are incomplete")
    origin_count = _integer(origins.get("count"), context="FX common origin count", minimum=1)
    origin_hash = origins.get("sha256")
    _require(isinstance(origin_hash, str) and _SHA256.fullmatch(origin_hash) is not None, "FX common origin SHA-256 is invalid")
    _require(origin_hash == recomputed_origin_hash, "FX payload common-origin SHA-256 does not match the OOS sidecar")
    _require(origin_count * len(FX_VARIANTS) == row_count, "FX payload common-origin count does not match the OOS sidecar")
    summary_rows: list[dict[str, Any]] = []
    control: dict[str, Any] | None = None
    for index, variant in enumerate(FX_VARIANTS):
        manifest_row = manifest[index]
        metric_row = metrics[index]
        _require(isinstance(manifest_row, Mapping) and isinstance(metric_row, Mapping), f"FX {variant} schema is invalid")
        _require(manifest_row.get("variant") == variant and metric_row.get("variant") == variant, f"FX variant order mismatch: {variant}")
        manifest_feature_count = _integer(
            manifest_row.get("feature_count"),
            context=f"FX {variant} manifest.feature_count",
        )
        manifest_feature_hash = manifest_row.get("feature_columns_sha256")
        _require(
            isinstance(manifest_feature_hash, str)
            and _SHA256.fullmatch(manifest_feature_hash) is not None,
            f"FX {variant} manifest feature hash is invalid",
        )
        feature_count = _integer(metric_row.get("feature_count"), context=f"FX {variant}.feature_count", minimum=1)
        fx_feature_count = _integer(metric_row.get("fx_feature_count"), context=f"FX {variant}.fx_feature_count")
        _require(feature_count >= fx_feature_count, f"FX {variant} feature counts are inconsistent")
        _require(
            manifest_feature_count == fx_feature_count,
            f"FX {variant} manifest/leaderboard feature count mismatch",
        )
        _require((variant == "v4_control") == (fx_feature_count == 0), f"FX {variant} control feature identity is invalid")
        feature_hash = metric_row.get("feature_columns_sha256")
        _require(isinstance(feature_hash, str) and _SHA256.fullmatch(feature_hash) is not None, f"FX {variant} feature hash is invalid")
        n_predictions = _integer(metric_row.get("n_predictions"), context=f"FX {variant}.n_predictions", minimum=1)
        _require(n_predictions == origin_count and metric_row.get("origin_sha256") == origin_hash, f"FX {variant} common-origin binding mismatch")
        recomputed = _fx_metrics(
            [row for row in sidecar_rows if row.model == variant]
        )
        _require(recomputed["n_predictions"] == n_predictions, f"FX {variant} independent row count mismatch")
        for metric in ("log_loss", "brier", "accuracy", "balanced_accuracy"):
            supplied = _finite_float(metric_row.get(metric), context=f"FX {variant}.{metric}")
            _require(
                math.isclose(supplied, float(recomputed[metric]), abs_tol=1e-12, rel_tol=0.0),
                f"FX {variant} independent {metric} mismatch",
            )
        supplied_fallback = _integer(metric_row.get("fallback_count"), context=f"FX {variant}.fallback_count")
        _require(supplied_fallback == recomputed["fallback_count"], f"FX {variant} independent fallback mismatch")
        row = {
            "variant": variant,
            "feature_count": feature_count,
            "fx_feature_count": fx_feature_count,
            "feature_columns_sha256": feature_hash,
            "n_predictions": n_predictions,
            "log_loss": recomputed["log_loss"],
            "brier": recomputed["brier"],
            "accuracy": recomputed["accuracy"],
            "balanced_accuracy": recomputed["balanced_accuracy"],
            "fallback_count": recomputed["fallback_count"],
        }
        if control is None:
            control = row
            row["delta_vs_v4_control"] = {"log_loss": 0.0, "brier": 0.0}
        else:
            row["delta_vs_v4_control"] = {
                "log_loss": row["log_loss"] - control["log_loss"],
                "brier": row["brier"] - control["brier"],
            }
        summary_rows.append(row)
    gate = ablation.get("gate")
    _require(isinstance(gate, Mapping), "evaluated FX gate is missing")
    comparisons = gate.get("comparisons")
    _require(
        isinstance(comparisons, list) and len(comparisons) == len(FX_VARIANTS) - 1,
        "evaluated FX gate comparisons are incomplete",
    )
    metric_index = {row["variant"]: row for row in summary_rows}
    for index, variant in enumerate(FX_VARIANTS[1:]):
        gate_row = comparisons[index]
        _require(isinstance(gate_row, Mapping), f"FX gate comparison is invalid: {variant}")
        _require(
            gate_row.get("variant") == variant
            and gate_row.get("reference_variant") == "v4_control",
            f"FX gate comparison identity mismatch: {variant}",
        )
        expected_improvement = (
            metric_index["v4_control"]["log_loss"] - metric_index[variant]["log_loss"]
        )
        expected_brier_difference = (
            metric_index[variant]["brier"] - metric_index["v4_control"]["brier"]
        )
        _require(
            math.isclose(
                _finite_float(
                    gate_row.get("mean_log_loss_improvement"),
                    context=f"FX gate {variant}.mean_log_loss_improvement",
                ),
                expected_improvement,
                abs_tol=1e-12,
                rel_tol=0.0,
            ),
            f"FX gate log-loss delta mismatch: {variant}",
        )
        _require(
            math.isclose(
                _finite_float(
                    gate_row.get("brier_difference"),
                    context=f"FX gate {variant}.brier_difference",
                ),
                expected_brier_difference,
                abs_tol=1e-12,
                rel_tol=0.0,
            ),
            f"FX gate Brier delta mismatch: {variant}",
        )
        _require(
            _integer(
                gate_row.get("control_fallback_count"),
                context=f"FX gate {variant}.control_fallback_count",
            )
            == metric_index["v4_control"]["fallback_count"]
            and _integer(
                gate_row.get("fallback_count"),
                context=f"FX gate {variant}.fallback_count",
            )
            == metric_index[variant]["fallback_count"],
            f"FX gate fallback count mismatch: {variant}",
        )
    return (
        {
            "comparison_status": "evaluated",
            "source_status": status,
            "common_origins": {"count": origin_count, "sha256": origin_hash},
            "variants": summary_rows,
            "aggregate_crosscheck": True,
            "payload_gate_metric_crosscheck": True,
            "interpretation": "diagnostic_only_not_a_promotion_decision",
        },
        input_manifest,
    )


def build_comparison(
    v5_artifacts: Path,
    v4_artifacts: Path = DEFAULT_V4_ARTIFACTS,
    *,
    v5_payload: Path | None = None,
) -> dict[str, Any]:
    """Build one deterministic report without exposing observations or features."""

    v5_directory = v5_artifacts.resolve()
    v4_directory = v4_artifacts.resolve()
    _require(v5_directory.is_dir() and not v5_directory.is_symlink(), "V5 artifacts directory is missing or invalid")
    frozen = _verify_frozen_v4(v4_directory)

    v4_oos_path = v4_directory / "oos-predictions.csv"
    v5_oos_path = v5_directory / "oos-predictions.csv"
    selection_path = v5_directory / "selection-diagnostics.csv"
    payload_path = _resolve_v5_payload(v5_directory, v5_payload)
    v4_rows, v4_row_count = _read_predictions(v4_oos_path, context="frozen V4 OOS predictions")
    v5_rows, v5_row_count = _read_predictions(v5_oos_path, context="V5 OOS predictions")
    diagnostics, diagnostic_row_count = _read_selection_diagnostics(selection_path)
    payload = _json_object(payload_path, context="V5 payload")
    core_manifest = _validate_v5_core_binding(
        payload,
        v5_directory,
        {
            "oos_predictions": v5_row_count,
            "selection_diagnostics": diagnostic_row_count,
        },
    )
    champion = str(payload.get("model", {}).get("champion", ""))
    _validate_payload_selection_diagnostics(payload, diagnostics)
    _validate_selection_family(diagnostics, v5_rows, champion=champion)

    markov_pairs = _same_model_pairs(
        v5_rows,
        v4_rows,
        model="markov",
        context="V5 versus frozen V4 markov",
    )
    multiscale_pairs = _equivalent_model_pairs(
        v5_rows,
        left_model="causal_multiscale_ensemble",
        right_model="markov",
    )
    markov_comparison = _split_comparison(
        markov_pairs,
        left_label="v5_markov",
        right_label="frozen_v4_markov",
        key_model="markov",
        include_parity=True,
    )
    markov_comparison["common_keys"] = _key_summary(
        [
            (left.origin_date, left.target_date, "markov")
            for left, _ in markov_pairs
        ]
    )
    multiscale_comparison = _split_comparison(
        multiscale_pairs,
        left_label="causal_multiscale_ensemble",
        right_label="v5_markov",
        key_model="causal_multiscale_ensemble~markov",
        include_parity=False,
    )
    multiscale_comparison["common_keys"] = _key_summary(
        [
            (
                left.origin_date,
                left.target_date,
                "causal_multiscale_ensemble~markov",
            )
            for left, _ in multiscale_pairs
        ]
    )
    multiscale_comparison["selection_gate_crosscheck"] = _selection_gate_crosscheck(
        diagnostics,
        v5_rows,
        champion=champion,
    )
    fx_summary, fx_input = _fx_summary(payload, v5_directory)

    return {
        "schema_version": "regime-v5-v4-matched-comparison/1",
        "report_role": "derived_only_diagnostic_comparison",
        "promotion_interpretation": "prohibited",
        "comparison_contract": {
            "exact_key_fields": ["origin_date", "target_date", "model_or_equivalent_model"],
            "actual_must_match": True,
            "evaluation_split_must_match": True,
            "unmatched_keys_excluded": True,
            "splits_are_never_pooled": list(SPLIT_ORDER),
            "probability_columns": list(PROBABILITY_COLUMNS),
            "metric_definitions": {
                "log_loss": "mean_negative_log_actual_probability_clip_1e-9_then_renormalize",
                "brier": "mean_three_state_sum_squared_error",
                "balanced_accuracy": "mean_recall_over_actual_classes_present",
            },
        },
        "inputs": {
            "v5": {
                "regime_results": {"path": "regime-results.json", "sha256": _file_sha256(payload_path)},
                **core_manifest,
                **({"fx_ablation_oos": fx_input} if fx_input is not None else {}),
            },
            "frozen_v4": {
                "baseline_id": "v4-20260821",
                "sha256sums": {"path": "SHA256SUMS", "sha256": frozen["inventory_sha256"]},
                "verified_file_count": frozen["verified_file_count"],
                "oos_predictions": {
                    "path": "oos-predictions.csv",
                    "row_count": v4_row_count,
                    "sha256": frozen["oos_predictions_sha256"],
                },
            },
        },
        "v5_markov_vs_frozen_v4_markov": {
            "join": {
                "left_model": "markov",
                "right_model": "markov",
                "model_equivalence": "exact_name",
                "left_key_count": sum(row.model == "markov" for row in v5_rows),
                "right_key_count": sum(row.model == "markov" for row in v4_rows),
                "common_key_count": len(markov_pairs),
            },
            **markov_comparison,
        },
        "v5_causal_multiscale_ensemble_vs_v5_markov": {
            "join": {
                "left_model": "causal_multiscale_ensemble",
                "right_model": "markov",
                "model_equivalence": "fixed_pair_same_origin_and_target",
                "left_key_count": sum(
                    row.model == "causal_multiscale_ensemble" for row in v5_rows
                ),
                "right_key_count": sum(row.model == "markov" for row in v5_rows),
                "common_key_count": len(multiscale_pairs),
            },
            **multiscale_comparison,
        },
        "fx_ablation": fx_summary,
        "provider_or_raw_feature_values_included": False,
    }


def canonical_json_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic, derived-only, exact-common-origin V5 versus "
            "frozen-V4 diagnostic report. This command never makes a promotion decision."
        )
    )
    parser.add_argument("--v5-artifacts", required=True, type=Path, help="completed private V5 artifact directory")
    parser.add_argument(
        "--v5-payload",
        type=Path,
        help="optional V5 regime-results.json override; sibling build layout is auto-detected",
    )
    parser.add_argument(
        "--v4-artifacts",
        type=Path,
        default=DEFAULT_V4_ARTIFACTS,
        help="reviewed frozen V4 artifact directory (default: artifacts/baselines/v4-20260821)",
    )
    parser.add_argument("--output", type=Path, help="write canonical JSON atomically; stdout when omitted")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_comparison(
            args.v5_artifacts,
            args.v4_artifacts,
            v5_payload=args.v5_payload,
        )
        content = canonical_json_bytes(report)
        if args.output is None:
            print(content.decode("utf-8"), end="")
        else:
            _write_atomic(args.output, content)
    except (ComparisonError, OSError) as exc:
        parser.exit(2, f"comparison failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
