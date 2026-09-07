"""Reproduce a frozen real origin's calibration and ensemble, without model refits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.analysis.structural_models import forecast_structural_probabilities
from regime_lab.analysis.transitions import causal_state_durations
from regime_lab.analysis.validation import _calibrate_transition_probability
from regime_lab.schema import STATE_ORDER


def replay(*, payload_path: Path, artifacts: Path, states_path: Path) -> dict:
    paths = {"payload": payload_path, "states": states_path,
             "oos": artifacts / "oos-predictions.csv",
             "transition_oos": artifacts / "transition-oos-predictions.csv",
             "transition_candidates": artifacts / "transition-candidate-forecasts.csv"}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    payload = json.loads(payload_path.read_text())
    states = pd.read_pickle(states_path)
    history = pd.read_csv(paths["oos"])
    transition = pd.read_csv(paths["transition_oos"])
    candidates = pd.read_csv(paths["transition_candidates"])
    latest = payload["weekly"][-1]
    assert states.index[-1].date().isoformat() == latest["date"]
    model_rows = {row["model"]: row for row in latest["model_forecasts"]}
    rows = candidates.loc[candidates.model.eq("binary_xgboost") & candidates.horizon.eq(1)
                          & pd.to_datetime(candidates.origin_date, utc=True).eq(states.index[-1])]
    assert len(rows) == 1
    candidate = rows.iloc[0]
    calibration = transition.loc[transition.model.eq("binary_xgboost") & transition.horizon.eq(1)
                                 & transition.evaluation_split.eq("selection")].copy()
    assert "calibration_version" not in calibration  # this specifically checks archived v1
    results = {}
    for version in (None, "transition-calibration/1", "transition-calibration/2"):
        frame = calibration.copy()
        if version is not None:
            frame["calibration_version"] = version
        probability, method, fallback, reason = _calibrate_transition_probability(
            float(candidate.raw_p_change), frame, minimum_rows=12, random_state=17,
            selection_end=payload["model"]["selection_end"], origin=states.index[-1])

        def vector(name):
            value = np.array([model_rows[name]["probabilities"][s] for s in STATE_ORDER])
            return value / value.sum()  # remove publication's eight-decimal rounding only

        reconstructed = forecast_structural_probabilities(
            origin_date=states.index[-1], current_state=str(states.iloc[-1]),
            markov_probability=vector("markov"), xgboost_probability=vector("xgboost"),
            binary_xgboost_p_change=probability, historical_oos_predictions=history,
            expert_fallbacks={name: bool(model_rows[name].get("fallback", False))
                             for name in ("markov", "xgboost", "xgb_hazard_destination")},
            current_duration_weeks=int(causal_state_durations(states).iloc[-1]), include_multiscale=True)
        champion = payload["selection"]["operating_champion"]
        result = reconstructed.probabilities.loc[reconstructed.probabilities.model.eq(champion)].iloc[0]
        ensemble = {s: float(result[f"p_{s}"]) for s in STATE_ORDER}
        results[version or "unversioned_legacy"] = {"hazard": probability, "method": method,
            "fallback": fallback, "reason": reason, "ensemble": ensemble,
            "published_ensemble_max_difference": max(abs(ensemble[s] - model_rows[champion]["probabilities"][s]) for s in STATE_ORDER)}
    legacy = results["unversioned_legacy"]
    assert abs(legacy["hazard"] - float(candidate.p_change)) < 1e-12
    assert legacy == results["transition-calibration/1"]
    assert legacy["published_ensemble_max_difference"] < 1e-7
    assert abs(legacy["hazard"] - results["transition-calibration/2"]["hazard"]) > 1e-4
    assert hashes == {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    root = Path(__file__).resolve().parents[1]
    return {"status": "passed", "origin": states.index[-1].isoformat(),
            "scope": "real_origin_frozen_component_replay_no_model_refits_no_ledger_or_publication_writes",
            "input_sha256": hashes, "results": results,
            "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
                "src/regime_lab/analysis/validation.py", "src/regime_lab/operational_forecast.py",
                "scripts/replay_operational_calibration_compatibility.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(payload_path=args.payload, artifacts=args.artifacts, states_path=args.states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
