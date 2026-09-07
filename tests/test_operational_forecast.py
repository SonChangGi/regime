from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.analysis.models import BenchmarkProfile
from regime_lab.operational_forecast import (
    OperationalPreparationError,
    frame_sha256,
    prepare_operational_forecast,
    write_prepared_forecast,
)


def _bundle():
    dates = pd.date_range("2013-01-04", periods=600, freq="W-FRI", tz="UTC")
    features = pd.DataFrame({"x": np.sin(np.arange(600))}, index=dates)
    states = pd.Series(
        np.array(["risk_on", "transition", "risk_off"])[np.arange(600) // 10 % 3],
        index=dates,
    )
    rows = []
    for k in range(521, 599):
        for model in ("markov", "xgboost", "xgb_hazard_destination"):
            rows.append(
                {
                    "origin_date": dates[k],
                    "target_date": dates[k + 1],
                    "model": model,
                    "actual": states.iloc[k + 1],
                    "predicted": states.iloc[k],
                    "current_state": states.iloc[k],
                    "p_risk_on": 0.2,
                    "p_transition": 0.7,
                    "p_risk_off": 0.1,
                    "fallback": False,
                    "train_size": k - 1,
                    "gap": 1,
                    "evaluation_split": "holdout",
                }
            )
    oos = pd.DataFrame(rows)
    transition = pd.DataFrame(
        {
            "origin_date": dates[100:120],
            "target_end": dates[101:121],
            "model": "binary_xgboost",
            "horizon": 1,
            "evaluation_split": "selection",
            "actual_change": np.arange(20) % 2 == 0,
            "p_change": 0.2,
            "raw_p_change": 0.2,
        }
    )
    budget = asdict(BenchmarkProfile.quick().with_overrides(minimum_train_weeks=520))
    budget.pop("name")
    manifest = {"profile": "quick", "profile_budget": budget}
    features_manifest = {"feature_count": 1, "groups": [{"features": ["x"]}]}
    features_manifest["sha256"] = canonical_json_sha256_v1(features_manifest)
    locked = {
        "model": {
            "champion": "causal_dynamic_ensemble",
            "selection_status": "selected_by_gate",
            "candidate_manifest": manifest,
            "candidate_manifest_sha256": canonical_json_sha256_v1(manifest),
            "feature_manifest_sha256": features_manifest["sha256"],
            "selection_end": dates[130].date().isoformat(),
        },
        "selection": {"operating_champion": "causal_dynamic_ensemble"},
        "weekly": [
            {
                "date": dates[-1].date().isoformat(),
                "model_forecasts": [
                    {
                        "model": "causal_dynamic_ensemble",
                        "date": (dates[-1] + timedelta(days=7)).date().isoformat(),
                        "probabilities": {
                            "risk_on": 0.2,
                            "transition": 0.7,
                            "risk_off": 0.1,
                        },
                    }
                ],
            }
        ],
    }
    arguments = {
        "locked_payload": locked,
        "feature_manifest": features_manifest,
        "expected_input_hashes": {
            "features": frame_sha256(features),
            "states": frame_sha256(states),
            "oos_predictions": frame_sha256(oos),
            "transition_predictions": frame_sha256(transition),
        },
    }
    return features, states, oos, transition, arguments


def _stub_models(monkeypatch):
    import regime_lab.operational_forecast as module

    counts = {"base": 0, "hazard": 0}

    def base(*args, **kwargs):
        counts["base"] += 1
        return pd.Series([0.2, 0.7, 0.1], index=["risk_on", "transition", "risk_off"])

    def hazard(*args, **kwargs):
        counts["hazard"] += 1
        assert kwargs["train_stop"] == 598
        assert kwargs["test_position"] == 599
        return 0.2, False, "", None

    monkeypatch.setattr(module, "forecast_next_regime", base)
    monkeypatch.setattr(module, "_fit_transition_candidate", hazard)
    monkeypatch.setattr(
        module,
        "_calibrate_transition_probability",
        lambda *a, **kw: (0.2, "fixed_test", False, ""),
    )
    monkeypatch.setattr(
        module,
        "forecast_structural_probabilities",
        lambda **kw: SimpleNamespace(
            probabilities=pd.DataFrame(
                [
                    {
                        "model": "causal_dynamic_ensemble",
                        "predicted": "transition",
                        "fallback": False,
                        "p_risk_on": 0.2,
                        "p_transition": 0.7,
                        "p_risk_off": 0.1,
                    }
                ]
            )
        ),
    )
    return counts


def test_late_stale_current_input_is_refused_before_model_fit(monkeypatch):
    features, states, oos, transition, kwargs = _bundle()
    counts = _stub_models(monkeypatch)
    result = prepare_operational_forecast(features, states, oos, transition, **kwargs)
    assert result["status"] == "blocked"
    assert set(result["blocked_reasons"]) == {
        "scheduled_entry_missed",
        "stale_completed_week",
    }
    assert counts == {"base": 0, "hazard": 0}
    assert result["issued"] is False


def test_replay_fits_only_latest_and_preserves_reference_probabilities(
    monkeypatch, tmp_path
):
    features, states, oos, transition, kwargs = _bundle()
    counts = _stub_models(monkeypatch)
    result = prepare_operational_forecast(
        features,
        states,
        oos,
        transition,
        **kwargs,
        research_replay=True,
        decision_at=(features.index[-1] + timedelta(minutes=1)).to_pydatetime(),
    )
    assert counts == {"base": 2, "hazard": 1}
    assert result["reference_parity"]["passed"]
    assert result["research_replay"] and not result["operational_issue_eligible"]
    path = write_prepared_forecast(result, tmp_path)
    before = path.read_bytes()
    assert write_prepared_forecast(result, tmp_path) == path
    assert path.read_bytes() == before
    broken = deepcopy(result)
    broken["forecast"]["probabilities"]["risk_on"] += 0.01
    with pytest.raises(OperationalPreparationError, match="conflicting"):
        write_prepared_forecast(broken, tmp_path)


def test_cache_hash_and_stale_expert_history_are_rejected(monkeypatch):
    features, states, oos, transition, kwargs = _bundle()
    _stub_models(monkeypatch)
    changed = features.copy()
    changed.iloc[0, 0] += 1
    with pytest.raises(OperationalPreparationError, match="hashes differ"):
        prepare_operational_forecast(changed, states, oos, transition, **kwargs)
    missing = oos.iloc[:-3]
    kwargs["expected_input_hashes"]["oos_predictions"] = frame_sha256(missing)
    with pytest.raises(OperationalPreparationError, match="history is stale"):
        prepare_operational_forecast(
            features,
            states,
            missing,
            transition,
            **kwargs,
            research_replay=True,
            decision_at=(features.index[-1] + timedelta(minutes=1)).to_pydatetime(),
        )


def test_real_preparation_cannot_backdate_decision_clock():
    features, states, oos, transition, kwargs = _bundle()
    with pytest.raises(OperationalPreparationError, match="only for research replay"):
        prepare_operational_forecast(
            features,
            states,
            oos,
            transition,
            **kwargs,
            decision_at=features.index[-1].to_pydatetime(),
        )


@pytest.mark.parametrize("version", [None, "transition-calibration/1", "transition-calibration/2"])
def test_locked_calibration_version_reaches_real_transform(monkeypatch, version):
    import regime_lab.operational_forecast as module
    from regime_lab.analysis.validation import _calibrate_transition_probability
    features, states, oos, transition, kwargs = _bundle()
    _stub_models(monkeypatch)
    # Keep model fits cheap, but exercise the actual calibrator and its caller.
    monkeypatch.setattr(module, "_calibrate_transition_probability", _calibrate_transition_probability)
    if version is not None:
        transition["calibration_version"] = version
    kwargs["expected_input_hashes"]["transition_predictions"] = frame_sha256(transition)
    result = prepare_operational_forecast(features, states, oos, transition, **kwargs,
        research_replay=True, decision_at=(features.index[-1] + timedelta(minutes=1)).to_pydatetime())
    expected = version or "transition-calibration/1"
    assert result["key"]["calibration_version"] == result["calibration"]["version"] == expected
    if expected.endswith("/1"):
        assert result["calibration"]["method"] == "prequential_platt_logit"
        assert result["calibration"]["probability"] == pytest.approx(.5, abs=.001)
    else:
        assert result["calibration"]["method"] == "identity"
        assert result["calibration"]["probability"] == .2
        assert result["calibration"]["fallback"]


@pytest.mark.parametrize("damage", ["mixed", "missing_version", "unknown"])
def test_ambiguous_calibration_generation_is_rejected_before_fitting(monkeypatch, damage):
    features, states, oos, transition, kwargs = _bundle()
    counts = _stub_models(monkeypatch)
    if damage == "missing_version":
        transition["calibration_selection_as_of"] = "2023-01-01"
    else:
        transition["calibration_version"] = "transition-calibration/2"
        transition.loc[0, "calibration_version"] = "transition-calibration/1" if damage == "mixed" else "unknown"
    kwargs["expected_input_hashes"]["transition_predictions"] = frame_sha256(transition)
    with pytest.raises(OperationalPreparationError, match="calibration version"):
        prepare_operational_forecast(features, states, oos, transition, **kwargs,
            research_replay=True, decision_at=(features.index[-1] + timedelta(minutes=1)).to_pydatetime())
    assert counts == {"base": 0, "hazard": 0}


@pytest.mark.parametrize('damage', ['actual', 'current', 'internal_gap'])
def test_expert_agreement_cannot_override_official_history(monkeypatch, damage):
    features, states, oos, transition, kwargs = _bundle()
    _stub_models(monkeypatch)
    changed = oos.copy()
    selected = changed.origin_date.eq(features.index[580])
    if damage == 'internal_gap':
        changed = changed.loc[~selected].copy()
        pattern = 'internal weekly gap'
    else:
        field = 'actual' if damage == 'actual' else 'current_state'
        truth = str(changed.loc[selected, field].iloc[0])
        changed.loc[selected, field] = next(s for s in ['risk_on','transition','risk_off'] if s != truth)
        pattern = 'differs from official states'
    kwargs['expected_input_hashes']['oos_predictions'] = frame_sha256(changed)
    with pytest.raises(OperationalPreparationError, match=pattern):
        prepare_operational_forecast(features, states, changed, transition, **kwargs,
            research_replay=True, decision_at=(features.index[-1] + timedelta(minutes=1)).to_pydatetime())
