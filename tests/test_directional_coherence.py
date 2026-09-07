from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from regime_lab.analysis.directional_coherence import (
    reconcile_directional_path,
    upgrade_directional_payload,
)
from regime_lab.analysis.directional import run_directional_transition_benchmark
import regime_lab.analysis.directional as directional
from regime_lab.contract_v5 import (
    _validate_directional_coherence_evidence,
    V5ContractError,
)


def _rows(totals, first):
    return {
        f"{h}w": {
            "probability": p,
            "no_departure": 1 - p,
            "first_destination": {"risk_on": f, "transition": 0.0, "risk_off": p - f},
            "target_end": "2026-01-01",
            "model": "test",
            "method": "first_departure_state_within_h_or_no_departure",
        }
        for h, p, f in zip((1, 4, 13), totals, first, strict=True)
    }


def test_joint_projection_matches_independent_constrained_optimizer():
    rng = np.random.default_rng(315)
    for _ in range(50):
        totals = np.sort(rng.uniform(0.001, 0.999, 3))
        anchor = float(rng.uniform(0, totals[0]))
        original = _rows(totals, rng.uniform(size=3) * totals)
        copy = deepcopy(original)
        result = reconcile_directional_path(
            "transition",
            {
                "risk_on": anchor,
                "transition": 1 - totals[0],
                "risk_off": totals[0] - anchor,
            },
            original,
        )
        assert original == copy
        projected = np.array(
            [result[f"{h}w"]["first_destination"]["risk_on"] for h in (4, 13)]
        )
        target = np.array(
            [original[f"{h}w"]["first_destination"]["risk_on"] for h in (4, 13)]
        )
        reference = minimize(
            lambda x: np.square(x - target).sum(),
            [anchor, anchor],
            bounds=[(anchor, anchor + totals[1] - totals[0]), (0, 1)],
            constraints=[
                {"type": "ineq", "fun": lambda x: x[1] - x[0]},
                {
                    "type": "ineq",
                    "fun": lambda x: totals[2] - totals[1] - (x[1] - x[0]),
                },
            ],
            method="SLSQP",
            options={"ftol": 1e-12},
        )
        assert reference.success
        assert np.square(projected - target).sum() == pytest.approx(
            reference.fun, abs=2e-8
        )
        for state in ("risk_on", "risk_off"):
            values = [result[f"{h}w"]["first_destination"][state] for h in (1, 4, 13)]
            assert np.diff(values).min() >= -1e-8
        assert result["1w"]["first_destination"]["risk_on"] == anchor


def test_equal_totals_and_zero_exit_are_coherent():
    for p in (0.0, 0.4, 1.0):
        result = reconcile_directional_path(
            "transition",
            {"risk_on": p, "transition": 1 - p, "risk_off": 0},
            _rows([p] * 3, [p / 2] * 3),
        )
        assert [
            result[f"{h}w"]["first_destination"]["risk_on"] for h in (1, 4, 13)
        ] == [p] * 3


def test_directional_cache_reuses_unchanged_prefixes_when_future_appends(
    tmp_path, monkeypatch
):
    index = pd.date_range("2020-01-03", periods=90, freq="W-FRI")
    states = pd.Series(
        np.resize(["risk_on"] * 5 + ["transition"] * 3 + ["risk_off"] * 4, len(index)),
        index=index,
    )
    features = pd.DataFrame({"trend": np.arange(len(index), dtype=float)}, index=index)
    real = directional._fit_candidate
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["test_position"])
        return real(*args, **kwargs)

    monkeypatch.setattr(directional, "_fit_candidate", counted)
    kwargs = dict(
        horizons=(1,),
        models=("empirical_first_passage",),
        minimum_train_weeks=20,
        selection_end=index[55],
        cache_directory=tmp_path,
    )
    original = run_directional_transition_benchmark(
        features.iloc[:-1], states.iloc[:-1], **kwargs
    )
    assert calls
    calls.clear()
    extended = run_directional_transition_benchmark(features, states, **kwargs)
    assert calls == [89]
    old = original.predictions.set_index(["origin_date", "model"])
    new = extended.predictions.set_index(["origin_date", "model"]).loc[old.index]
    pd.testing.assert_frame_equal(old, new)


def test_projected_evaluation_does_not_score_unmatured_targets():
    weeks = []
    for d, s in zip(
        pd.date_range("2025-01-03", periods=6, freq="W-FRI"),
        ["transition"] * 4 + ["risk_on"] * 2,
    ):
        # Keep the synthetic current state consistent with zero origin mass.
        if s != "transition":
            break
        weeks.append(
            {
                "date": d.date().isoformat(),
                "current": {"state": s},
                "next_week": {
                    "probabilities": {
                        "risk_on": 0.15,
                        "transition": 0.8,
                        "risk_off": 0.05,
                    }
                },
                "directional_risk": _rows([0.2, 0.5, 0.8], [0.1, 0.2, 0.1]),
            }
        )
    payload = {
        "weekly": weeks,
        "model": {"selection_end": "2023-01-01", "directional_transition": {}},
    }
    copy = deepcopy(payload)
    result = upgrade_directional_payload(payload)
    assert payload == copy
    records = result["model"]["directional_transition"]["coherence_evidence"][
        "oos_predictions"
    ]
    assert len(records) == 6
    assert {r["horizon_weeks"] for r in records} == {1}
    _validate_directional_coherence_evidence(result)
    result["model"]["directional_transition"]["coherence_evidence"]["metrics"][0][
        "log_loss"
    ] += 0.1
    with pytest.raises(V5ContractError, match="must reproduce"):
        _validate_directional_coherence_evidence(result)
