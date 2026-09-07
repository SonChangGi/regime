"""Path targets, competing risks and causal origin tests (no operational writes)."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.boundary_forecast import (
    ASYMMETRIC_MODEL, BoundaryConfig, asymmetric_volatility, build_boundary_inputs,
    run_boundary_walk_forward,
)
from regime_lab.analysis.forecast_paths import (
    fit_direction_hazard, path_outcomes, project_multistate_paths,
)
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.schema import STATE_ORDER


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(453)
    returns = rng.standard_t(5, 558) * .025 + .001
    index = pd.date_range("2012-01-06", periods=len(returns), freq="W-FRI", tz="UTC")
    frame = pd.DataFrame({"spy_close": 100 * np.exp(returns.cumsum())}, index=index)
    labeler = CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(frame.iloc[:520])
    return frame, labeler.transform(frame)


def test_paths_distinguish_first_destination_endpoint_entry_and_occupancy():
    # on -> transition -> off -> on, including nonabsorbing recovery.
    matrix = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rows = project_multistate_paths(lambda s, a: matrix[STATE_ORDER.index(s)], "risk_on", 7, horizons=(1, 2, 3, 4))
    assert rows[0]["first_departure"]["transition"] == 1
    assert rows[1]["first_departure"]["risk_off"] == 0
    assert rows[1]["endpoint"]["risk_off"] == rows[1]["any_risk_off_entry"] == 1
    assert rows[2]["endpoint"]["risk_on"] == 1
    assert rows[2]["any_risk_off_entry"] == 1
    already_off = project_multistate_paths(lambda s, a: np.eye(3)[STATE_ORDER.index(s)], "risk_off", 4)
    assert all(r["any_risk_off_entry"] == 0 and r["any_risk_off_occupancy"] == 1 for r in already_off)


def test_paths_update_spell_age_and_reset_after_a_transition():
    calls = []
    def kernel(state, age):
        calls.append((state, age))
        if state == "risk_on":
            return np.asarray([1, 0, 0]) if age < 3 else np.asarray([0, 1, 0])
        return np.asarray([0, 1, 0]) if age < 2 else np.asarray([0, 0, 1])
    rows = project_multistate_paths(kernel, "risk_on", 2, horizons=(1, 2, 3, 4))
    assert rows[0]["endpoint"]["risk_on"] == 1
    assert rows[1]["endpoint"]["transition"] == 1
    assert ("risk_on", 3) in calls and ("transition", 1) in calls and ("transition", 2) in calls
    assert rows[-1]["any_risk_off_entry"] == 1


def test_matured_entry_for_risk_off_requires_leave_and_return():
    states = pd.Series(["risk_off", "risk_off", "transition", "risk_off", "risk_on"],
                       index=pd.date_range("2024-01-05", periods=5, freq="W-FRI", tz="UTC"))
    rows = path_outcomes(states, horizons=(1, 3)).set_index(["origin_date", "horizon_weeks"])
    one, three = rows.loc[(states.index[0], 1)], rows.loc[(states.index[0], 3)]
    assert not one.any_risk_off_entry and one.any_risk_off_occupancy
    assert three.any_risk_off_entry and three.first_departure == "transition"
    assert (states.index[-1], 1) not in rows.index


@pytest.mark.parametrize("probability", [[np.nan, 1, 0], [-.1, .6, .5], [.1, .2, .3]])
def test_path_engine_rejects_invalid_kernel_instead_of_normalizing(probability):
    with pytest.raises(ValueError):
        project_multistate_paths(lambda s, a: np.array(probability), "risk_on", 1)


def test_boundary_default_config_preserves_probabilities_and_opt_in_isolated(market):
    canonical, states = market
    default = build_boundary_inputs(canonical, states)
    configured = build_boundary_inputs(canonical, states, config=BoundaryConfig(), include_asymmetric=True)
    pd.testing.assert_frame_equal(default.features, configured.features)
    for name in default.mechanistic:
        pd.testing.assert_frame_equal(default.mechanistic[name], configured.mechanistic[name])
    assert ASYMMETRIC_MODEL not in default.mechanistic
    rows = run_boundary_walk_forward(configured, origin_positions=[535], models=(ASYMMETRIC_MODEL,))
    assert len(rows) == 1 and rows.p_risk_off.iloc[0] > 0


def test_asymmetric_volatility_responds_more_to_negative_equal_size_shock():
    up = pd.Series([.01] * 20 + [.05])
    down = pd.Series([.01] * 20 + [-.05])
    assert asymmetric_volatility(down).iloc[-1] > asymmetric_volatility(up).iloc[-1]
    np.testing.assert_array_equal(asymmetric_volatility(up).iloc[:-1], asymmetric_volatility(down).iloc[:-1])


def test_hazard_and_asymmetric_model_cannot_use_future_returns(market):
    canonical, states = market
    first = build_boundary_inputs(canonical, states, include_asymmetric=True)
    changed = canonical.copy()
    changed.iloc[536:, 0] *= 1.4
    labeler = first.labeler
    second = build_boundary_inputs(changed, labeler.transform(changed), include_asymmetric=True)
    for inputs in (first, second):
        assert pd.Timestamp(fit_direction_hazard(inputs, 535).audit["last_train_target"]) < states.index[535]
    p = fit_direction_hazard(first, 535).paths(str(states.iloc[535]), 5)
    q = fit_direction_hazard(second, 535).paths(str(states.iloc[535]), 5)
    assert p == q
    np.testing.assert_array_equal(first.mechanistic[ASYMMETRIC_MODEL].iloc[:536], second.mechanistic[ASYMMETRIC_MODEL].iloc[:536])


def test_hazard_competing_mass_and_impossible_directions(market):
    canonical, states = market
    model = fit_direction_hazard(build_boundary_inputs(canonical, states), 535)
    grid = model.probability_grid([1, 2, 8, 30])
    for (state, _), p in grid.items():
        assert np.isfinite(p).all() and p.sum() == pytest.approx(1)
        assert (p > 0).all()  # direct jumps retained by routing prior
    for current in STATE_ORDER:
        paths = model.paths(current, 8)
        assert all(paths[i]["any_risk_off_entry"] <= paths[i+1]["any_risk_off_entry"] + 1e-12 for i in range(2))
        assert all(sum(row["first_departure"].values()) == pytest.approx(1) for row in paths)
