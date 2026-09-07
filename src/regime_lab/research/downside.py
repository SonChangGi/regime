"""Chronological, horizon-purged SPY downside distribution challengers."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss


@dataclass
class DownsideResult:
    summary: dict
    predictions: pd.DataFrame


def downside_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices.spy_close.astype(float)
    ret = close.pct_change(fill_method=None)
    result = pd.DataFrame(index=prices.index)
    result["return_4w"] = close.pct_change(4, fill_method=None)
    result["return_13w"] = close.pct_change(13, fill_method=None)
    result["volatility_13w"] = ret.rolling(13, min_periods=13).std()
    result["drawdown_26w"] = close / close.rolling(26, min_periods=26).max() - 1
    if "tlt_close" in prices:
        bond = prices.tlt_close.pct_change(fill_method=None)
        result["equity_duration_correlation_26w"] = ret.rolling(
            26, min_periods=26
        ).corr(bond)
    return result


def paired_block_interval(values, *, block=13, resamples=999, seed=17):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 * block:
        return [None, None]
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, len(values), size=(resamples, math.ceil(len(values) / block))
    )
    indices = (starts[:, :, None] + np.arange(block)) % len(values)
    means = values[indices.reshape(resamples, -1)[:, : len(values)]].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def run_downside_research(
    prices: pd.DataFrame,
    *,
    additional: pd.DataFrame | None = None,
    minimum_train_weeks: int = 520,
    refit_weeks: int = 13,
    selection_end: str = "2023-01-01",
) -> DownsideResult:
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices index must be ordered unique")
    if "spy_adjusted_open" not in prices:
        raise ValueError("adjusted open is required for executable horizon")
    x = downside_features(prices)
    variants = {"linear_quantile": x}
    if additional is not None:
        columns = [
            c
            for c in [
                "short_vol_ratio",
                "short_vol_ratio_change_1w",
                "vvix",
                "vvix_change_4w",
            ]
            if c in additional
        ]
        if columns:
            variants["linear_quantile_cboe"] = x.join(additional[columns])
    entry = prices.spy_adjusted_open.shift(-1)
    rows = []
    for horizon, loss_threshold in ((4, -0.05), (13, -0.10)):
        y = prices.spy_adjusted_open.shift(-(horizon + 1)) / entry - 1
        for name, features in variants.items():
            valid_x = (
                features.replace([np.inf, -np.inf], np.nan)
                .notna()
                .all(axis=1)
                .to_numpy()
            )
            fitted = None
            last_fit = -refit_weeks
            for i in range(minimum_train_weeks + horizon + 1, len(prices)):
                if not valid_x[i]:
                    continue
                # Labels exiting at or before this cutoff may enter training.
                eligible = np.flatnonzero(
                    valid_x
                    & y.notna().to_numpy()
                    & (np.arange(len(prices)) + horizon + 1 <= i)
                )
                if len(eligible) < minimum_train_weeks:
                    continue
                if fitted is None or i - last_fit >= refit_weeks:
                    scaler = StandardScaler().fit(features.iloc[eligible])
                    train = scaler.transform(features.iloc[eligible])
                    target = y.iloc[eligible].to_numpy()
                    quantiles = [
                        QuantileRegressor(quantile=q, alpha=0.001, solver="highs").fit(
                            train, target
                        )
                        for q in (0.1, 0.25)
                    ]
                    event = (target <= loss_threshold).astype(int)
                    classifier = (
                        LogisticRegression(C=1, max_iter=500).fit(train, event)
                        if len(np.unique(event)) == 2
                        else None
                    )
                    fitted = (
                        scaler,
                        quantiles,
                        classifier,
                        float(event.mean()),
                        eligible[-1],
                    )
                    last_fit = i
                scaler, models, classifier, prior, fit_train_end = fitted
                current = scaler.transform(features.iloc[[i]])
                # Rearrangement fixes quantile crossing without looking at returns.
                q10, q25 = sorted(float(m.predict(current)[0]) for m in models)
                probability = (
                    float(classifier.predict_proba(current)[0, 1])
                    if classifier is not None
                    else prior
                )
                base = {
                    "origin_date": prices.index[i].isoformat(),
                    "target_exit": (
                        prices.index[i + horizon + 1].isoformat()
                        if i + horizon + 1 < len(prices)
                        else None
                    ),
                    "horizon_weeks": horizon,
                    "model": name,
                    "q10": q10,
                    "q25": q25,
                    "loss_probability": probability,
                    "loss_threshold": loss_threshold,
                    "actual_return": float(y.iloc[i]) if pd.notna(y.iloc[i]) else None,
                    "train_end_origin": prices.index[fit_train_end].isoformat(),
                    "last_fit_origin": prices.index[last_fit].isoformat(),
                    "split": (
                        "diagnostic"
                        if prices.index[i].date().isoformat() >= selection_end
                        else "selection"
                        if i + horizon + 1 < len(prices)
                        and prices.index[i + horizon + 1].date().isoformat()
                        < selection_end
                        else "boundary_purged"
                    ),
                }
                rows.append(base)
                if name != "linear_quantile":
                    continue
                history = y.iloc[eligible].to_numpy()
                sigma = float(x.volatility_13w.iloc[i]) * np.sqrt(horizon)
                mu = float(history.mean())
                for baseline in ("historical_quantile", "volatility_baseline"):
                    if baseline == "historical_quantile":
                        a, b = np.quantile(history, [0.1, 0.25])
                        p = float((history <= loss_threshold).mean())
                    else:
                        a, b = mu - 1.281551565545 * sigma, mu - 0.674489750196 * sigma
                        p = 0.5 * (
                            1
                            + math.erf(
                                (loss_threshold - mu) / max(sigma, 1e-8) / np.sqrt(2)
                            )
                        )
                    rows.append(
                        {
                            **base,
                            "model": baseline,
                            "train_end_origin": prices.index[eligible[-1]].isoformat(),
                            "last_fit_origin": prices.index[i].isoformat(),
                            "q10": float(a),
                            "q25": float(b),
                            "loss_probability": p,
                        }
                    )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return DownsideResult(
            {
                "status": "insufficient_history",
                "target": "SPY next-open to open total return",
                "latest": [],
                "comparisons": [],
                "features": list(x.columns),
                "promotion": "research_only",
            },
            predictions,
        )
    comparisons = []
    for (horizon, split), part in predictions.dropna(subset=["actual_return"]).groupby(
        ["horizon_weeks", "split"]
    ):
        if split == "boundary_purged":
            continue
        # All variants share exactly the same origins, including Cboe availability.
        counts = part.groupby("origin_date").model.nunique()
        common = counts[counts == part.model.nunique()].index
        part = part[part.origin_date.isin(common)]
        baseline = part[part.model.eq("volatility_baseline")].set_index("origin_date")
        for name, candidate in part.groupby("model"):
            candidate = candidate.set_index("origin_date").sort_index()
            b = baseline.reindex(candidate.index)
            target = candidate.actual_return.to_numpy()
            q = candidate.q10.to_numpy()
            residual = target - q
            loss = np.maximum(0.1 * residual, -0.9 * residual)
            br = target - b.q10.to_numpy()
            bl = np.maximum(0.1 * br, -0.9 * br)
            comparisons.append(
                {
                    "horizon_weeks": int(horizon),
                    "split": split,
                    "model": name,
                    "weeks": len(candidate),
                    "pinball_q10": float(loss.mean()),
                    "pinball_q25": float(
                        mean_pinball_loss(target, candidate.q25, alpha=0.25)
                    ),
                    "coverage_q10": float((target <= q).mean()),
                    "coverage_q25": float((target <= candidate.q25).mean()),
                    "brier_loss": float(
                        np.mean(
                            (
                                candidate.loss_probability
                                - (target <= candidate.loss_threshold)
                            )
                            ** 2
                        )
                    ),
                    "delta_pinball_q10": float((loss - bl).mean()),
                    "delta_pinball_q10_ci": paired_block_interval(
                        loss - bl, block=max(13, int(horizon))
                    ),
                    "effective_nonoverlapping_windows": len(candidate) // int(horizon),
                }
            )
    latest = []
    if not predictions.empty:
        last = predictions[
            predictions.origin_date.eq(prices.index[-1].isoformat())
            & predictions.model.eq("linear_quantile")
        ]
        latest = last[
            [
                "origin_date",
                "horizon_weeks",
                "q10",
                "q25",
                "loss_probability",
                "loss_threshold",
            ]
        ].to_dict("records")
    return DownsideResult(
        {
            "status": "evaluated",
            "as_of": prices.index[-1].isoformat(),
            "target": "SPY next-open to open total return",
            "latest": latest,
            "comparisons": comparisons,
            "features": list(x.columns),
            "selection_end": selection_end,
            "promotion": "research_only",
            "refit_weeks": refit_weeks,
            "minimum_train_weeks": minimum_train_weeks,
            "uncertainty": "paired circular moving-block bootstrap; 13 weeks; 999 resamples",
            "availability_basis": "reconstructed_market; Cboe prior-day historical sensitivity",
        },
        predictions,
    )
