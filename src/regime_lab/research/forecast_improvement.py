"""Publish frozen boundary challengers with matched, traceable research history.

This module returns an optional research block. It never changes the official
champion, issued forecasts, canonical states, or any caller-owned object.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
import pandas as pd

from regime_lab.analysis import boundary_forecast
from regime_lab.analysis.boundary_forecast import (
    build_boundary_inputs,
    forecast_boundary_latest,
    prequential_temperature,
    run_boundary_walk_forward,
)
from regime_lab.analysis.label_spec import load_label_spec
from regime_lab.analysis.validation import evaluate_predictions
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256
from regime_lab.schema import STATE_ORDER

SCHEMA_VERSION = "regime-forecast-improvement/1"
SELECTED_MODEL = "boundary_filtered_history"
RESEARCH_MODELS = (SELECTED_MODEL, "boundary_student_t")
MODEL_LABELS = {
    SELECTED_MODEL: "경계 전환 · 과거 충격",
    "boundary_student_t": "경계 전환 · Student-t",
    "causal_dynamic_ensemble": "동적 앙상블",
    "xgboost": "XGBoost",
    "recency_weighted_xgboost_208w": "최근 가중 XGBoost",
}
BASELINE_MODELS = ("causal_dynamic_ensemble", "xgboost", "recency_weighted_xgboost_208w")
PROBABILITY_COLUMNS = [f"p_{state}" for state in STATE_ORDER]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _identity(canonical: pd.DataFrame, states: pd.Series, baseline: pd.DataFrame) -> dict:
    sources = {
        "boundary_forecast": Path(boundary_forecast.__file__),
        "forecast_improvement": Path(__file__),
        "labels": Path(boundary_forecast.__file__).with_name("labels.py"),
        "evaluation": Path(boundary_forecast.__file__).with_name("validation.py"),
    }
    code_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    input_hashes = {"canonical": frame_sha256(canonical), "states": frame_sha256(states),
                    "baseline_oos": frame_sha256(baseline)}
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": canonical_json_sha256_v1(input_hashes),
        "canonical_sha256": input_hashes["canonical"],
        "states_sha256": input_hashes["states"],
        "baseline_oos_sha256": input_hashes["baseline_oos"],
        "code_sha256": canonical_json_sha256_v1(code_hashes),
        "label_spec_sha256": load_label_spec().spec_sha256,
        "runtime": {package: version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn")},
        "selected_model": SELECTED_MODEL,
        "models": list(RESEARCH_MODELS),
        "selection_end": "2023-01-01",
    }


def _validate_baseline(canonical: pd.DataFrame, states: pd.Series, baseline: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if not canonical.index.equals(states.index):
        raise ValueError("forecast research canonical and states must align")
    if len(states) < 523 or not states.isin(STATE_ORDER).all():
        raise ValueError("forecast research requires complete official state history")
    required = {"origin_date", "target_date", "model", "evaluation_split", "current_state", "actual", "fallback", *PROBABILITY_COLUMNS}
    if not required.issubset(baseline.columns):
        raise ValueError("forecast research baseline OOS is incomplete")
    chosen = baseline.loc[baseline.model.isin(BASELINE_MODELS)].copy(deep=True)
    if not chosen.model.eq(BASELINE_MODELS[0]).any():
        raise ValueError("forecast research requires the official dynamic baseline")
    chosen["origin_date"] = pd.to_datetime(chosen.origin_date, utc=True)
    chosen["target_date"] = pd.to_datetime(chosen.target_date, utc=True)
    if chosen.duplicated(["model", "origin_date"]).any():
        raise ValueError("forecast research baseline contains duplicate origins")
    reference = chosen.loc[chosen.model.eq(BASELINE_MODELS[0])].sort_values("origin_date")
    if set(reference.evaluation_split) != {"selection", "holdout"}:
        raise ValueError("forecast research requires both selection and holdout evidence")
    positions = []
    for row in reference.itertuples():
        if row.origin_date not in states.index or row.target_date not in states.index:
            raise ValueError("baseline forecast date is outside official state history")
        position = int(states.index.get_loc(row.origin_date))
        if position < 521 or position + 1 >= len(states) or states.index[position + 1] != row.target_date:
            raise ValueError("baseline OOS must target the next observed week")
        if row.current_state != states.iloc[position] or row.actual != states.iloc[position + 1]:
            raise ValueError("baseline OOS disagrees with official states")
        cutoff = pd.Timestamp("2023-01-01", tz="UTC")
        if row.evaluation_split == "selection" and row.target_date >= cutoff:
            raise ValueError("baseline selection crosses the frozen boundary")
        if row.evaluation_split == "holdout" and row.origin_date < cutoff:
            raise ValueError("baseline holdout crosses the frozen boundary")
        positions.append(position)
    expected_positions = [p for p in range(521, len(states) - 1)
                          if not (states.index[p].year < 2023 <= states.index[p + 1].year)]
    if positions != expected_positions:
        raise ValueError("forecast research requires every official OOS origin")
    reference_columns = ["origin_date", "target_date", "current_state", "actual", "evaluation_split"]
    expected = reference.loc[:, reference_columns].reset_index(drop=True)
    for model, frame in chosen.groupby("model"):
        actual = frame.sort_values("origin_date").loc[:, reference_columns].reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"baseline {model} does not share matched OOS history")
    probability = chosen[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("baseline probabilities are invalid")
    return chosen, positions


def _validate_predictions(predictions: pd.DataFrame, states: pd.Series, positions: list[int]) -> None:
    if set(predictions.model) != set(RESEARCH_MODELS) or len(predictions) != len(positions) * len(RESEARCH_MODELS):
        raise ValueError("boundary research OOS coverage differs")
    if predictions.duplicated(["model", "origin_date"]).any() or predictions.fallback.astype(bool).any():
        raise ValueError("boundary research contains duplicates or fallback predictions")
    for model in RESEARCH_MODELS:
        rows = predictions.loc[predictions.model.eq(model)].copy()
        rows["origin_date"] = pd.to_datetime(rows.origin_date, utc=True)
        rows = rows.sort_values("origin_date")
        if rows.origin_date.tolist() != states.index[positions].tolist():
            raise ValueError("boundary research origin set differs")
        history = []
        for position, row in zip(positions, rows.itertuples()):
            if pd.Timestamp(row.target_date) != states.index[position + 1]:
                raise ValueError("boundary research target is not next week")
            if row.current_state != states.iloc[position] or row.actual != states.iloc[position + 1]:
                raise ValueError("boundary research states differ")
            if pd.Timestamp(row.last_train_target) >= row.origin_date:
                raise ValueError("boundary research training is not purged")
            expected_split = "selection" if states.index[position + 1].year < 2023 else "holdout"
            if row.evaluation_split != expected_split:
                raise ValueError("boundary research split differs")
            raw = np.asarray([getattr(row, f"raw_p_{state}") for state in STATE_ORDER], dtype=float)
            probability = np.asarray([getattr(row, f"p_{state}") for state in STATE_ORDER], dtype=float)
            if not np.isfinite(raw).all() or (raw <= 0).any() or not np.isclose(raw.sum(), 1):
                raise ValueError("boundary raw probability is invalid")
            calibrated, temperature, count = prequential_temperature(raw, history, position)
            if not np.allclose(probability, calibrated, atol=1e-10, rtol=0) or not np.isclose(row.temperature, temperature, atol=1e-9, rtol=0) or int(row.calibration_rows) != count:
                raise ValueError("boundary calibration does not reproduce from completed OOS")
            history.append((position + 1, raw, STATE_ORDER.index(str(row.actual))))


def _metric_table(frame: pd.DataFrame) -> dict:
    output = {}
    for split in ("selection", "holdout"):
        group = frame.loc[frame.evaluation_split.eq(split)].copy()
        metric = evaluate_predictions(group).iloc[0].to_dict()
        metric.pop("model")
        actual = np.asarray([STATE_ORDER.index(value) for value in group.actual])
        current = np.asarray([STATE_ORDER.index(value) for value in group.current_state])
        predicted = group[PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
        worsening, recovery = actual > current, actual < current
        metric.update({
            "worsening_event_count": int(worsening.sum()),
            "on_time_worsening_count": int((worsening & (predicted > current)).sum()),
            "recovery_event_count": int(recovery.sum()),
            "on_time_recovery_count": int((recovery & (predicted < current)).sum()),
            "period_start": pd.Timestamp(group.origin_date.min()).isoformat(),
            "period_end": pd.Timestamp(group.target_date.max()).isoformat(),
        })
        output[split] = _json_safe(metric)
    return output


def _public_prediction(row: dict) -> dict:
    probabilities = {state: float(row[f"p_{state}"]) for state in STATE_ORDER}
    return {
        "origin_date": pd.Timestamp(row["origin_date"]).isoformat(),
        "target_date": pd.Timestamp(row["target_date"]).isoformat(),
        "current_state": str(row["current_state"]),
        "actual": None if row["actual"] is None else str(row["actual"]),
        "predicted": max(STATE_ORDER, key=lambda state: probabilities[state]),
        "probabilities": probabilities,
        "raw_probabilities": {state: float(row[f"raw_p_{state}"]) for state in STATE_ORDER},
        "evaluation_split": str(row["evaluation_split"]),
        "calibration": {"temperature": float(row["temperature"]), "rows": int(row["calibration_rows"]),
                        "last_train_target": pd.Timestamp(row["last_train_target"]).isoformat()},
    }


def _create_block(predictions: pd.DataFrame, latest: dict[str, dict], baseline: pd.DataFrame, identity: dict, states: pd.Series) -> dict:
    models = []
    for model in RESEARCH_MODELS:
        rows = predictions.loc[predictions.model.eq(model)].sort_values("origin_date")
        models.append({
            "id": model, "label": MODEL_LABELS[model], "metrics": _metric_table(rows),
            "latest": _public_prediction(latest[model]),
            "history": [_public_prediction(row) for row in rows.to_dict("records")],
        })
    baselines = [{"id": model, "label": MODEL_LABELS[model], "metrics": _metric_table(baseline.loc[baseline.model.eq(model)])}
                 for model in BASELINE_MODELS if baseline.model.eq(model).any()]
    return {
        "schema_version": SCHEMA_VERSION, "selected_model": SELECTED_MODEL,
        "data_as_of": states.index[-1].isoformat(), "evidence_track": "reconstructed_market",
        "selection": {"frozen_model": SELECTED_MODEL, "selection_end": "2023-01-01",
                      "rule": "frozen_after_2016_2022_probability_comparison", "weekly_reselection": False},
        "models": models, "baselines": baselines,
        "provenance": {**identity, "cache_key": canonical_json_sha256_v1(identity),
                       "oos_predictions_sha256": frame_sha256(predictions),
                       "matched_oos_origins": len(predictions) // len(RESEARCH_MODELS)},
    }


def _read_cache(directory: Path, identity: dict, baseline: pd.DataFrame, states: pd.Series, positions: list[int]) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("identity") != identity or set(manifest.get("files", {})) != {"block.json", "oos-predictions.csv", "latest.json"}:
        raise ValueError("forecast research cache identity differs")
    for name, record in manifest["files"].items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["bytes"]:
            raise ValueError("forecast research cache content hash differs")
    predictions = pd.read_csv(directory / "oos-predictions.csv", float_precision="round_trip")
    _validate_predictions(predictions, states, positions)
    latest = json.loads((directory / "latest.json").read_text())
    for model in RESEARCH_MODELS:
        row = latest[model]
        if row["actual"] is not None or row["evaluation_split"] != "unobserved" or pd.Timestamp(row["origin_date"]) != states.index[-1]:
            raise ValueError("forecast research cached latest observation differs")
        if pd.Timestamp(row["target_date"]) <= states.index[-1] or row["current_state"] != states.iloc[-1]:
            raise ValueError("forecast research cached latest state differs")
    expected = _create_block(predictions, latest, baseline, identity, states)
    block = json.loads((directory / "block.json").read_text())
    if block != expected:
        raise ValueError("forecast research cache metrics or history differ")
    return block


def build_forecast_improvement(canonical: pd.DataFrame, states: pd.Series, baseline_oos: pd.DataFrame, cache_directory: Path | None = None, *, progress: Callable[[str], None] | None = None) -> dict:
    """Return both frozen challengers with complete matched OOS and latest data."""
    canonical = canonical.copy(deep=True)
    states = states.copy(deep=True)
    baseline_oos = baseline_oos.copy(deep=True)
    baseline, positions = _validate_baseline(canonical, states, baseline_oos)
    identity = _identity(canonical, states, baseline_oos)
    key = canonical_json_sha256_v1(identity)
    destination = Path(cache_directory) / key if cache_directory is not None else None
    if destination is not None and destination.exists():
        return _read_cache(destination, identity, baseline, states, positions)
    if progress:
        progress("경계 전환 예측: 공식 원점과 과거 검증 정렬")
    inputs = build_boundary_inputs(canonical, states)
    predictions = run_boundary_walk_forward(inputs, origin_positions=positions, models=RESEARCH_MODELS, progress=progress)
    _validate_predictions(predictions, states, positions)
    latest = {model: forecast_boundary_latest(inputs, predictions, model=model) for model in RESEARCH_MODELS}
    block = _create_block(predictions, latest, baseline, identity, states)
    json.dumps(block, allow_nan=False)
    if destination is None:
        return block
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".forecast-research-", dir=destination.parent))
    try:
        predictions.to_csv(staging / "oos-predictions.csv", index=False)
        for name, value in (("block.json", block), ("latest.json", latest)):
            (staging / name).write_text(json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        files = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in staging.iterdir()}
        (staging / "manifest.json").write_text(json.dumps({"identity": identity, "files": files}, indent=2) + "\n")
        verified = _read_cache(staging, identity, baseline, states, positions)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            return _read_cache(destination, identity, baseline, states, positions)
        return verified
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def with_forecast_improvement(snapshot: dict, block: dict) -> dict:
    """Attach the optional block to a new snapshot without mutating its source."""
    if block.get("schema_version") != SCHEMA_VERSION or block.get("selected_model") != SELECTED_MODEL:
        raise ValueError("forecast improvement block contract differs")
    if pd.Timestamp(snapshot["meta"]["data_as_of"]) != pd.Timestamp(block["data_as_of"]):
        raise ValueError("forecast improvement cutoff differs from snapshot")
    result = deepcopy(snapshot)
    result.setdefault("research", {})["forecast_improvement"] = deepcopy(block)
    return result
