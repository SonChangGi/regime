"""Fixed-recipe endpoint and evolving-price-path challengers, all origins causal.

Every fit uses targets strictly before its origin. Volatility parameters use
returns completed strictly before the origin; the observed origin return updates
the next variance. Simulated future prices recompute the official label score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regime_lab.analysis.boundary_forecast import build_boundary_inputs, next_states
from regime_lab.analysis.forecast_paths import fit_markov_path
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.schema import STATE_ORDER

PROB = [f"p_{s}" for s in STATE_ORDER]
CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")
MODEL_LABELS = {
    "direct_endpoint_ridge": "기간별 직접 국면 · Ridge",
    "direct_endpoint_xgboost": "기간별 직접 국면 · 얕은 XGBoost",
    "evolving_boundary_ewma": "가격·변동성 경로 · EWMA",
    "evolving_boundary_gjr_skewt": "가격·변동성 경로 · GJR-GARCH 비대칭 t",
    "markov_endpoint": "기간별 Markov 기준선",
    "directional_duration_hazard": "기존 방향·기간 경로",
    "boundary_filtered_history": "기존 경계 · 과거 잔차",
    "causal_dynamic_ensemble": "현재 운영 동적 앙상블",
}


@dataclass(frozen=True)
class EnhancementProtocol:
    version: str = "forecast-enhancements-v1"
    label_fit_weeks: int = 520
    train_window_weeks: int = 520
    horizons: tuple[int, ...] = (1, 4, 13)
    simulation_power: int = 9
    path_contamination: float = .01
    seed: int = 20260907
    direct_refit_every_weeks: int = 4
    bootstrap_resamples: int = 999
    calibration_min_fit: int = 104
    calibration_min_validation: int = 52
    calibration_window_weeks: int = 260
    calibration_material_improvement: float = .002
    alert_annual_budget: float = 4.
    alert_cooldown_weeks: int = 2
    alert_exit_fraction: float = .65

    def __post_init__(self):
        if self.horizons != (1, 4, 13):
            raise ValueError("v1 fixes the supported horizons")
        for field in ("label_fit_weeks", "train_window_weeks", "direct_refit_every_weeks", "bootstrap_resamples", "calibration_min_fit", "calibration_min_validation", "calibration_window_weeks"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not 4 <= self.simulation_power <= 16:
            raise ValueError("simulation_power must be between 4 and 16")

    def record(self):
        return {**asdict(self), "automatic_promotion": False,
                "evaluation_role": "2023_plus_previously_inspected_retrospective_diagnostic",
                "training_rule": "target_strictly_before_origin",
                "display_model": "direct_endpoint_ridge",
                "simulation_draws": 2 ** self.simulation_power,
                "simulation_method": "fixed_scrambled_sobol_common_random_numbers",
                "arch_reference": "https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html"}


def split(origin, target, observed=True):
    if not observed:
        return "prospective"
    if origin < CUTOFF <= target:
        return "cross_cutoff_excluded"
    return "selection" if target < CUTOFF else "retrospective_diagnostic"


def target_date(index, position, horizon):
    if position + horizon < len(index):
        return index[position + horizon]
    return (index[position].tz_convert("America/New_York") + pd.DateOffset(weeks=horizon)).tz_convert("UTC")


def row_for(states, position, horizon, model, probability, **extras):
    p = np.asarray(probability, float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-8):
        raise ValueError("invalid endpoint probability")
    origin = states.index[position]
    target = target_date(states.index, position, horizon)
    actual = str(states.iloc[position + horizon]) if position + horizon < len(states) else None
    current = str(states.iloc[position])
    return {"origin_date": origin, "target_date": target, "model": model,
            "horizon_weeks": horizon, "target": "endpoint", "current_state": current,
            "actual": actual, "evaluation_split": split(origin, target, actual is not None),
            **dict(zip(PROB, p)), "fallback": False, **extras}


def scores_from_prices(prices: np.ndarray, labeler: CausalRegimeLabeler) -> np.ndarray:
    """Score each simulated price history with the exact official transforms."""
    prices = np.asarray(prices, float)
    if prices.ndim != 2 or prices.shape[1] < 53 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("paths need at least 53 finite positive prices")
    logs = np.log(prices)
    returns = np.diff(logs, axis=1)
    raw = {}
    for window in (13, 26):
        scale = returns[:, -window:].std(axis=1, ddof=0) * np.sqrt(window)
        raw[f"trend_{window}w"] = np.divide(logs[:, -1] - logs[:, -window - 1], scale,
                                             out=np.zeros(len(prices)), where=scale > 0)
    for window in (4, 13):
        raw[f"vol_{window}w"] = returns[:, -window:].std(axis=1, ddof=0) * np.sqrt(52)
    for window in (13, 52):
        raw[f"drawdown_{window}w"] = 1 - prices[:, -1] / prices[:, -window:].max(axis=1)
    z = {key: (value - labeler.component_stats_[key][0]) / labeler.component_stats_[key][1] for key, value in raw.items()}
    trend = (z["trend_13w"] + z["trend_26w"]) / 2
    stress = sum(z[k] for k in ("vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w")) / 4
    tc, ts = labeler.composite_stats_["trend"]
    sc, ss = labeler.composite_stats_["stress"]
    return (trend - tc) / ts - (stress - sc) / ss


def evolve_paths(history, current_state, labeler, innovations, next_variance,
                 variance_parameters, horizons=(1, 4, 13)):
    """Simulate coherent endpoints/first departure/new Risk-off entry together."""
    noise = np.asarray(innovations, float)
    if noise.ndim != 2 or noise.shape[1] < max(horizons) or not np.isfinite(noise).all():
        raise ValueError("invalid path innovations")
    count = len(noise)
    prices = np.tile(np.asarray(history, float)[-53:], (count, 1))
    state = np.full(count, current_state, dtype=object)
    departed = np.zeros(count, dtype=bool)
    entered = np.zeros(count, dtype=bool)
    occupied = np.zeros(count, dtype=bool)
    variance = np.full(count, float(next_variance))
    contamination = .01
    no_entry_mass = np.eye(3)[STATE_ORDER.index(current_state)]
    no_entry_kernel = np.full((3, 3), 1 / 3)
    no_entry_kernel[:2, 2] = 0.
    omega, alpha, gamma, beta = variance_parameters
    rows = []
    for step in range(1, max(horizons) + 1):
        shock = np.sqrt(np.maximum(variance, 1e-12)) * noise[:, step - 1]
        # Numerical saturation is visible to validation, not a probability repair.
        if not np.isfinite(shock).all() or np.max(np.abs(shock)) > 10:
            raise ValueError("simulated return exceeds numerical support")
        prices = np.column_stack([prices[:, 1:], prices[:, -1] * np.exp(shock)])
        score = scores_from_prices(prices, labeler)
        following = state.copy()
        for name in STATE_ORDER:
            mask = state == name
            following[mask] = next_states(score[mask], name, labeler)
        departed |= following != current_state
        entered |= (state != "risk_off") & (following == "risk_off")
        occupied |= following == "risk_off"
        state = following
        no_entry_mass = no_entry_mass @ no_entry_kernel
        variance = omega + (alpha + gamma * (shock < 0)) * shock ** 2 + beta * variance
        if step in horizons:
            # Mix an entire uniform-state path law, rather than independently
            # smoothing marginals, preserving the one-week departure identity.
            p = (1-contamination) * np.array([(state == name).mean() for name in STATE_ORDER]) + contamination / 3
            rows.append({"horizon_weeks": step, "probability": p,
                         "first_departure_probability": float((1-contamination)*departed.mean()+contamination*(1-(1/3)**step)),
                         "risk_off_entry_probability": float((1-contamination)*entered.mean()+contamination*(1-no_entry_mass.sum())),
                         "risk_off_occupancy_probability": float((1-contamination)*occupied.mean()+contamination*(1-(2/3)**step)),
                         "simulation_standard_error_max": float(np.sqrt(.25 / count))})
    return rows


def fit_gjr(returns: np.ndarray, origin_position: int, window=520):
    from arch import arch_model
    from arch.univariate.distribution import SkewStudent
    start = max(1, origin_position - window)
    past = returns[start:origin_position] * 100
    with warnings.catch_warnings(record=True) as caught:
        fitted = arch_model(past, mean="Zero", vol="GARCH", p=1, o=1, q=1,
                            dist="skewt", rescale=False).fit(disp="off", show_warning=False,
                                                          options={"maxiter": 600, "ftol": 1e-8})
    if fitted.convergence_flag != 0 or not np.isfinite(fitted.params).all():
        raise ValueError(f"GJR optimizer did not converge: {fitted.convergence_flag}")
    params = fitted.params
    omega = float(params["omega"]) / 10000
    alpha, gamma, beta = [float(params[k]) for k in ("alpha[1]", "gamma[1]", "beta[1]")]
    # Fit excludes origin; first evolve to its variance, then assimilate origin.
    variance_origin = omega + (alpha + gamma * (returns[origin_position - 1] < 0)) * returns[origin_position - 1] ** 2 + beta * (float(fitted.conditional_volatility[-1]) / 100) ** 2
    next_variance = omega + (alpha + gamma * (returns[origin_position] < 0)) * returns[origin_position] ** 2 + beta * variance_origin
    distribution = SkewStudent()
    dist_params = [float(params["eta"]), float(params["lambda"])]
    audit = {"fit_start_position": start, "fit_last_position": origin_position - 1,
             "fit_rows": len(past), "convergence_flag": int(fitted.convergence_flag),
             "eta": dist_params[0], "skew_lambda": dist_params[1],
             "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
             "warning_count": len(caught)}
    return next_variance, (omega, alpha, gamma, beta), distribution, dist_params, audit


def _aligned(estimator, frame):
    p = np.full(3, 1e-6)
    for state, value in zip(estimator.classes_, estimator.predict_proba(frame)[0]):
        p[STATE_ORDER.index(str(state))] = value
    return p / p.sum()


def run_models(canonical, states, protocol=EnhancementProtocol(), progress=None):
    """All eligible weekly origins; monthly direct fits amortize computation."""
    if not canonical.index.equals(states.index):
        raise ValueError("state and canonical index mismatch")
    inputs = build_boundary_inputs(canonical, states, fit_weeks=protocol.label_fit_weeks)
    features = inputs.features
    returns = np.log(canonical.spy_close).diff().to_numpy(float)
    sigma = pd.Series(returns).ewm(span=13, adjust=False).std(bias=True).clip(lower=.003).to_numpy()
    residual = returns / np.r_[np.nan, sigma[:-1]]
    uniforms = qmc.Sobol(d=13, scramble=True, seed=protocol.seed).random_base2(protocol.simulation_power)
    rows, audits = [], []
    estimators = {}
    first = protocol.label_fit_weeks + 1
    for position in range(first, len(states)):
        matrix = fit_markov_path(states, position, train_window_weeks=protocol.train_window_weeks)
        current = str(states.iloc[position])
        for horizon in protocol.horizons:
            rows.append(row_for(states, position, horizon, "markov_endpoint", np.linalg.matrix_power(matrix, horizon)[STATE_ORDER.index(current)], last_train_target=states.index[position - 1]))
            if horizon not in (4, 13):
                continue
            refit = (position - first) % protocol.direct_refit_every_weeks == 0
            for model in ("direct_endpoint_ridge", "direct_endpoint_xgboost"):
                key = model, horizon
                if refit or key not in estimators:
                    train = np.arange(max(52, position - protocol.train_window_weeks), position - horizon)
                    x, y = features.iloc[train], states.iloc[train + horizon]
                    if model == "direct_endpoint_ridge":
                        estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=.1, max_iter=1200, tol=1e-6))
                    else:
                        from regime_lab.analysis.models import StateEncodedXGBoostClassifier
                        estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StateEncodedXGBoostClassifier(n_estimators=100, max_depth=2, min_child_weight=8, reg_lambda=12))
                    estimator.fit(x, y)
                    estimators[key] = estimator, states.index[train[-1] + horizon], states.index[position]
                estimator, last_target, fit_origin = estimators[key]
                rows.append(row_for(states, position, horizon, model, _aligned(estimator, features.iloc[[position]]), last_train_target=last_target, last_refit_origin=fit_origin))
        empirical = residual[max(2, position - protocol.train_window_weeks):position]
        empirical = np.clip(empirical[np.isfinite(empirical)], -8, 8)
        # Fixed zero-drift innovations distinguish volatility from drift fitting.
        empirical = empirical - empirical.mean()
        empirical = empirical / max(empirical.std(), 1e-6)
        ewma_noise = np.quantile(empirical, uniforms)
        candidates = [("evolving_boundary_ewma", sigma[position] ** 2, (0., 2 / 14, 0., 12 / 14), ewma_noise, False, "")]
        try:
            variance, params, dist, dparams, audit = fit_gjr(returns, position, protocol.train_window_weeks)
            innovations = dist.ppf(uniforms, dparams)
            candidates.append(("evolving_boundary_gjr_skewt", variance, params, innovations, False, ""))
            audits.append({"origin_date": states.index[position], **audit})
        except (ValueError, FloatingPointError) as error:
            candidates.append(("evolving_boundary_gjr_skewt", sigma[position] ** 2, (0., 2 / 14, 0., 12 / 14), ewma_noise, True, str(error)))
            audits.append({"origin_date": states.index[position], "fallback": True, "reason": str(error)})
        for model, variance, params, innovations, fallback, reason in candidates:
            simulated = evolve_paths(canonical.spy_close.to_numpy()[position - 52:position + 1], current, inputs.labeler, innovations, variance, params)
            for item in simulated:
                horizon, probability = item.pop("horizon_weeks"), item.pop("probability")
                rows.append(row_for(states, position, horizon, model, probability, last_train_target=states.index[position - 1], fallback=fallback, fallback_reason=reason, **item))
        if progress and ((position - first) % 26 == 0 or position == len(states) - 1):
            progress(f"All-origin models {position-first+1}/{len(states)-first}: {states.index[position].date()}")
    return pd.DataFrame(rows), pd.DataFrame(audits), inputs
