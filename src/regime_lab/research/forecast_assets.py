"""Model-specific asset outcomes for the frozen boundary forecast models."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from regime_lab import contract_v5, v5
from regime_lab.analysis import outcomes as outcome_analysis
from regime_lab.contract_v5 import validate_model_conditioned_statistics_values
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256
from regime_lab.research.forecast_contract import validate_forecast_improvement
from regime_lab.research.forecast_improvement import RESEARCH_MODELS
from regime_lab.schema import STATE_ORDER


def _weekly_predictions(canonical: pd.DataFrame, forecast_block: dict) -> tuple[list[dict], pd.DatetimeIndex]:
    validate_forecast_improvement(forecast_block, data_as_of=canonical.index[-1].isoformat())
    model_histories = {model["id"]: [row for row in model["history"] if row["evaluation_split"] == "holdout"]
                       for model in forecast_block["models"]}
    origins = [pd.Timestamp(row["origin_date"]) for row in model_histories[RESEARCH_MODELS[0]]]
    if not origins or any(origin not in canonical.index for origin in origins):
        raise ValueError("forecast asset origins must belong to canonical history")
    weekly = []
    for number, origin in enumerate(origins):
        position = canonical.index.get_loc(origin)
        forecasts = []
        for model in RESEARCH_MODELS:
            row = model_histories[model][number]
            if pd.Timestamp(row["origin_date"]) != origin or pd.Timestamp(row["target_date"]) != canonical.index[position + 1]:
                raise ValueError("forecast asset histories must share adjacent target weeks")
            probability = np.asarray([row["probabilities"][state] for state in STATE_ORDER])
            forecasts.append({"model": model, "state": STATE_ORDER[int(probability.argmax())]})
        weekly.append({"date": origin.date().isoformat(), "model_forecasts": forecasts})
    return weekly, pd.DatetimeIndex(origins)


def _identity(canonical: pd.DataFrame, forecast_block: dict, bootstrap_resamples: int) -> dict:
    sources = {"composer": Path(v5.__file__), "outcomes": Path(outcome_analysis.__file__),
               "contract": Path(contract_v5.__file__), "forecast_assets": Path(__file__)}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
    return {"schema_version": "regime-forecast-assets-cache/1",
            "canonical_sha256": frame_sha256(canonical),
            "forecast_block_sha256": canonical_json_sha256_v1(forecast_block),
            "code_sha256": canonical_json_sha256_v1(hashes),
            "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": 17,
            "models": list(RESEARCH_MODELS), "assets": list(outcome_analysis.ASSETS),
            "horizons_weeks": list(outcome_analysis.HORIZONS),
            "runtime": {package: version(package) for package in ("numpy", "pandas")}}


def _verify_outcome_execution(outcomes: pd.DataFrame, canonical: pd.DataFrame, weekly: list[dict], origins: pd.DatetimeIndex) -> None:
    if outcomes.duplicated(["conditioning_model", "origin_date", "asset", "horizon_weeks"]).any():
        raise ValueError("forecast asset outcomes contain duplicate observations")
    reference_frames = []
    for model in RESEARCH_MODELS:
        states = pd.Series([next(row["state"] for row in week["model_forecasts"] if row["model"] == model)
                            for week in weekly], index=origins, dtype=object)
        reference = outcome_analysis.build_forward_outcomes(canonical, states)
        reference.insert(0, "conditioning_model", model)
        reference_frames.append(reference)
    expected = pd.concat(reference_frames, ignore_index=True)
    actual = outcomes.copy()
    for column in ("origin_date", "entry_date", "exit_date"):
        actual[column] = pd.to_datetime(actual[column], utc=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-13, atol=1e-14)
    except AssertionError as exc:
        raise ValueError("forecast asset outcomes differ from next-open execution") from exc


def _read_cache(directory: Path, identity: dict, canonical: pd.DataFrame, weekly: list[dict], origins: pd.DatetimeIndex) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("identity") != identity or set(manifest.get("files", {})) != {"asset-statistics.json", "outcomes.csv", "statistics.csv"}:
        raise ValueError("forecast asset cache identity differs")
    for name, record in manifest["files"].items():
        data = (directory / name).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("forecast asset cache content hash differs")
    result = json.loads((directory / "asset-statistics.json").read_text())
    validate_model_conditioned_statistics_values(result, expected_models=RESEARCH_MODELS, expected_resamples=identity["bootstrap_resamples"])
    outcomes = pd.read_csv(directory / "outcomes.csv", float_precision="round_trip")
    _verify_outcome_execution(outcomes, canonical, weekly, origins)
    return result


def build_forecast_asset_statistics(canonical: pd.DataFrame, forecast_block: dict, *, bootstrap_resamples: int, cache_directory: Path | None = None) -> dict:
    """Use the established 1/4/13-week next-open outcome method for both models.

    Only completed, matched holdout origins classify outcomes. Latest unresolved
    predictions never enter historical performance. All six existing assets and
    all three forecast states retain the existing statistical contract.
    """
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, (int, np.integer)) or bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be a nonnegative integer")
    bootstrap_resamples = int(bootstrap_resamples)
    canonical = canonical.copy(deep=True)
    forecast_block = deepcopy(forecast_block)
    weekly, origins = _weekly_predictions(canonical, forecast_block)
    identity = _identity(canonical, forecast_block, bootstrap_resamples)
    destination = Path(cache_directory) / canonical_json_sha256_v1(identity) if cache_directory is not None else None
    if destination is not None and destination.exists():
        return _read_cache(destination, identity, canonical, weekly, origins)
    research, outcomes, statistics = v5._model_conditioned_research(
        canonical, weekly, RESEARCH_MODELS,
        bootstrap_resamples=bootstrap_resamples, matched_origins=origins,
    )
    result = research["model_conditioned_asset_stats"]
    validate_model_conditioned_statistics_values(result, expected_models=RESEARCH_MODELS, expected_resamples=bootstrap_resamples)
    _verify_outcome_execution(outcomes, canonical, weekly, origins)
    if destination is None:
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".forecast-assets-", dir=destination.parent))
    try:
        outcomes.to_csv(staging / "outcomes.csv", index=False)
        statistics.to_csv(staging / "statistics.csv", index=False)
        (staging / "asset-statistics.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        files = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in staging.iterdir()}
        (staging / "manifest.json").write_text(json.dumps({"identity": identity, "matched_holdout_origins": len(origins),
                                                        "outcome_rows": len(outcomes), "statistics_rows": len(statistics), "files": files}, indent=2) + "\n")
        verified = _read_cache(staging, identity, canonical, weekly, origins)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            return _read_cache(destination, identity, canonical, weekly, origins)
        return verified
    finally:
        if staging.exists():
            shutil.rmtree(staging)
