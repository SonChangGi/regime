"""Validate the displayed evidence for additive next-week forecasting models."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from regime_lab.schema import STATE_ORDER

MODEL_IDS = ("boundary_filtered_history", "boundary_student_t")


def _probabilities(value: object) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != set(STATE_ORDER):
        raise ValueError("forecast improvement needs an exact three-state vector")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value.values()):
        raise ValueError("forecast improvement probabilities must be numeric")
    p = np.array([value[state] for state in STATE_ORDER], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not math.isclose(float(p.sum()), 1, abs_tol=1e-8):
        raise ValueError("forecast improvement probabilities must form a finite simplex")
    return p


def _calibration(row: dict, origin: pd.Timestamp) -> None:
    value = row.get("calibration", {})
    temperature, count = value.get("temperature"), value.get("rows")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not .5 <= temperature <= 2:
        raise ValueError("forecast improvement calibration temperature invalid")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 156:
        raise ValueError("forecast improvement calibration sample invalid")
    last_target = pd.Timestamp(value.get("last_train_target"))
    if pd.isna(last_target) or last_target.tzinfo is None or last_target >= origin:
        raise ValueError("forecast improvement calibration includes an unresolved target")


def validate_forecast_improvement(block: dict, *, data_as_of: str | None = None,
                                 outcome_resamples: int | None = None) -> None:
    if block.get("schema_version") != "regime-forecast-improvement/1":
        raise ValueError("forecast improvement schema invalid")
    if block.get("selected_model") != MODEL_IDS[0] or block.get("evidence_track") != "reconstructed_market":
        raise ValueError("forecast improvement selection or evidence identity invalid")
    cutoff = pd.Timestamp(block["data_as_of"])
    if cutoff.tzinfo is None or (data_as_of is not None and cutoff != pd.Timestamp(data_as_of)):
        raise ValueError("forecast improvement cutoff differs from publication")
    models = block.get("models")
    if not isinstance(models, list) or [row.get("id") for row in models] != list(MODEL_IDS):
        raise ValueError("forecast improvement candidate roster invalid")
    provenance = block.get("provenance", {})
    for key in ("input_sha256", "code_sha256", "baseline_oos_sha256", "cache_key"):
        value = provenance.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"forecast improvement {key} invalid")
    reference = None
    for model in models:
        history = model.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError("forecast improvement history is empty")
        identities, scores = [], {"selection": [], "holdout": []}
        for row in history:
            origin, target = pd.Timestamp(row["origin_date"]), pd.Timestamp(row["target_date"])
            if origin.tzinfo is None or target.tzinfo is None or target > cutoff:
                raise ValueError("forecast improvement historical target is unresolved")
            if not 167 <= (target - origin).total_seconds() / 3600 <= 169:
                raise ValueError("forecast improvement must predict the adjacent week")
            split, current, actual = row["evaluation_split"], row["current_state"], row["actual"]
            if split not in scores or current not in STATE_ORDER or actual not in STATE_ORDER:
                raise ValueError("forecast improvement split or label invalid")
            p = _probabilities(row["probabilities"])
            _probabilities(row["raw_probabilities"])
            _calibration(row, origin)
            predicted = STATE_ORDER[int(p.argmax())]
            if row.get("predicted") != predicted:
                raise ValueError("forecast improvement predicted state differs from argmax")
            identities.append((origin, target, current, actual, split))
            event, alert = actual != current, predicted != current
            y = STATE_ORDER.index(actual)
            c, predicted_index = STATE_ORDER.index(current), STATE_ORDER.index(predicted)
            scores[split].append({"loss": -np.log(max(p[y], 1e-9)),
                                  "brier": float(((p - np.eye(3)[y]) ** 2).sum()),
                                  "event": event, "hit": event and alert, "false": not event and alert,
                                  "worsening": y > c, "worsening_hit": y > c and predicted_index > c,
                                  "recovery": y < c, "recovery_hit": y < c and predicted_index < c})
        if len({row[0] for row in identities}) != len(identities) or identities != sorted(identities):
            raise ValueError("forecast improvement history must have unique ordered origins")
        if reference is not None and identities != reference:
            raise ValueError("forecast improvement models use different evidence rows")
        reference = identities
        for split, rows in scores.items():
            if not rows:
                raise ValueError("forecast improvement is missing a comparison period")
            metric = model["metrics"][split]
            events, hits, false = (sum(r[key] for r in rows) for key in ["event", "hit", "false"])
            expected = {"n_predictions": len(rows), "log_loss": np.mean([r["loss"] for r in rows]),
                        "brier": np.mean([r["brier"] for r in rows]),
                        "transition_event_count": sum(r["event"] for r in rows),
                        "on_time_departure_count": sum(r["hit"] for r in rows),
                        "false_alarm_count": sum(r["false"] for r in rows),
                        "false_alarms_per_year": false / len(rows) * 52.1775,
                        "transition_recall": hits / events if events else 0,
                        "transition_precision": hits / (hits + false) if hits + false else 0,
                        "worsening_event_count": sum(r["worsening"] for r in rows),
                        "on_time_worsening_count": sum(r["worsening_hit"] for r in rows),
                        "recovery_event_count": sum(r["recovery"] for r in rows),
                        "on_time_recovery_count": sum(r["recovery_hit"] for r in rows)}
            for key, value in expected.items():
                if not isinstance(metric.get(key), (int, float)) or not math.isclose(metric[key], value, rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"forecast improvement {split}.{key} differs from history")
        latest = model["latest"]
        if pd.Timestamp(latest["origin_date"]) != cutoff or latest.get("actual") is not None:
            raise ValueError("forecast improvement latest origin or unresolved outcome invalid")
        if identities[-1][1] != cutoff or latest["current_state"] != identities[-1][3]:
            raise ValueError("forecast improvement latest state is not the last resolved target")
        target = pd.Timestamp(latest["target_date"])
        if target.tzinfo is None or not 167 <= (target - cutoff).total_seconds() / 3600 <= 169:
            raise ValueError("forecast improvement latest horizon invalid")
        if latest["current_state"] not in STATE_ORDER:
            raise ValueError("forecast improvement latest current state invalid")
        p = _probabilities(latest["probabilities"])
        _probabilities(latest["raw_probabilities"])
        if latest.get("predicted") != STATE_ORDER[int(p.argmax())]:
            raise ValueError("forecast improvement latest state differs from argmax")
        _calibration(latest, cutoff)
    if "asset_statistics" in block:
        from regime_lab.contract_v5 import validate_model_conditioned_statistics_values
        statistics = block["asset_statistics"]
        rows = statistics.get("rows", []) if isinstance(statistics, dict) else []
        if not rows:
            raise ValueError("forecast improvement asset statistics are empty")
        expected = outcome_resamples if outcome_resamples is not None else rows[0].get("bootstrap_resamples")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError("forecast improvement asset bootstrap count invalid")
        validate_model_conditioned_statistics_values(statistics, expected_models=MODEL_IDS, expected_resamples=expected)
