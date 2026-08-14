"""Deterministic synthetic data for offline end-to-end and UI verification."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from regime_lab.collection import weekly_cutoffs
from regime_lab.data import HealthStatus, Observation
from regime_lab.dataset import build_weekly_dataset
from regime_lab.pipeline import build_dashboard_result


def _simulate_inputs(config: Mapping[str, Any], *, seed: int = 20260811) -> tuple:
    rng = np.random.default_rng(seed)
    start = date(2009, 1, 2)
    end = datetime(2026, 8, 7, 20, tzinfo=timezone.utc)
    cutoffs = weekly_cutoffs(start, end)
    count = len(cutoffs)

    transition = np.array(
        [
            [0.965, 0.030, 0.005],
            [0.18, 0.64, 0.18],
            [0.01, 0.07, 0.92],
        ]
    )
    latent = np.zeros(count, dtype=int)
    for index in range(1, count):
        latent[index] = rng.choice(3, p=transition[latent[index - 1]])
    means = np.array([0.0025, 0.0001, -0.006])
    vols = np.array([0.014, 0.024, 0.043])
    spy_returns = means[latent] + vols[latent] * rng.standard_normal(count)

    symbols = tuple(config["alpha_vantage"]["symbols"])
    prices: dict[str, np.ndarray] = {}
    for position, symbol in enumerate(symbols):
        beta = 0.65 + (position % 8) * 0.09
        defensive = symbol in {"XLP", "XLU", "XLV", "SHY", "IEF", "TLT", "GLD"}
        credit = symbol in {"HYG", "LQD"}
        own = 0.006 + (position % 4) * 0.001
        returns = beta * spy_returns + own * rng.standard_normal(count)
        if defensive:
            returns -= 0.45 * spy_returns
        if credit:
            returns += np.where(latent == 2, -0.006, 0.001)
        if symbol == "UUP":
            returns = -0.25 * spy_returns + 0.006 * rng.standard_normal(count)
        prices[symbol] = 100.0 * np.exp(np.cumsum(returns))
    prices["SPY"] = 100.0 * np.exp(np.cumsum(spy_returns))

    growth = np.zeros(count)
    stress = np.zeros(count)
    for index in range(1, count):
        growth[index] = 0.94 * growth[index - 1] + (0.35 - 0.75 * latent[index]) + rng.normal(0, 0.35)
        stress[index] = 0.88 * stress[index - 1] + (latent[index] - 0.55) + rng.normal(0, 0.45)

    alfred_values: dict[str, np.ndarray] = {}
    for item in config["alfred"]["series"]:
        series_id = str(item["id"])
        noise = rng.normal(0, 0.15, count)
        if series_id in {"NFCI", "STLFSI4"}:
            values = stress / 4 + noise
        elif series_id in {"ICSA", "CCSA", "UNRATE"}:
            base = 220 if series_id == "ICSA" else 1800 if series_id == "CCSA" else 4.5
            scale = 35 if series_id == "ICSA" else 180 if series_id == "CCSA" else 0.45
            values = base + scale * stress + noise
        elif series_id in {"PAYEMS", "INDPRO", "RSAFS", "HOUST", "GDPC1"}:
            base = {"PAYEMS": 130000, "INDPRO": 95, "RSAFS": 300000, "HOUST": 1200, "GDPC1": 16000}[series_id]
            scale = base * 0.004
            values = base + scale * np.cumsum(np.tanh(growth / 5)) + noise
        elif series_id == "WALCL":
            values = 4000 + np.cumsum(2.5 + noise)
        elif series_id == "DTWEXBGS":
            values = 105 + 2.0 * stress + noise
        elif series_id in {"CPIAUCSL", "PCEPI"}:
            values = 200 + np.cumsum(0.4 + 0.04 * growth + noise / 10)
        elif series_id == "T10Y2Y":
            values = 1.0 + 0.25 * growth - 0.18 * stress + noise
        elif series_id in {"DGS3MO", "DGS2", "DGS10", "DFII10", "T10YIE", "FEDFUNDS"}:
            base = {"DGS3MO": 2.4, "DGS2": 2.8, "DGS10": 3.3, "DFII10": 1.0, "T10YIE": 2.2, "FEDFUNDS": 2.0}[series_id]
            values = base + 0.18 * growth + 0.14 * stress + noise
        else:
            values = growth + noise
        alfred_values[series_id] = np.asarray(values, dtype=float)

    retrieved_at = cutoffs[-1] + timedelta(days=30)
    records: list[Observation] = []
    for row_index, cutoff in enumerate(cutoffs):
        period = cutoff.date()
        for symbol in symbols:
            adjusted_close = float(prices[symbol][row_index])
            previous = (
                float(prices[symbol][row_index - 1])
                if row_index > 0
                else adjusted_close
            )
            raw_open = previous * float(np.exp(rng.normal(0.0, 0.0025)))
            raw_close = adjusted_close
            intraperiod_range = float(0.004 + abs(rng.normal(0.0, 0.006)))
            raw_high = max(raw_open, raw_close) * (1.0 + intraperiod_range)
            raw_low = min(raw_open, raw_close) * (1.0 - intraperiod_range)
            volume = float(
                (28_000_000 + 2_500_000 * (symbols.index(symbol) % 7))
                * np.exp(0.12 * stress[row_index] + rng.normal(0, 0.15))
            )
            field_values = {
                "open": raw_open,
                "high": raw_high,
                "low": raw_low,
                "close": raw_close,
                "adjusted_close": adjusted_close,
                "volume": volume,
            }
            for field in config["alpha_vantage"]["fields"]:
                records.append(
                    Observation(
                        source="alpha_vantage",
                        series_id=f"{symbol}.{field}",
                        observed_period_end=period,
                        value=float(field_values[str(field)]),
                        released_at=cutoff,
                        available_at=cutoff,
                        vintage_date=period,
                        retrieved_at=retrieved_at,
                        units="shares" if field == "volume" else "USD",
                        adjustment=(
                            "synthetic_adjusted"
                            if field == "adjusted_close"
                            else "synthetic"
                        ),
                        license_class="synthetic_fixture",
                        quality_status=HealthStatus.OK,
                        raw_sha256=hashlib.sha256(
                            f"{symbol}:{field}:{period}".encode()
                        ).hexdigest(),
                    )
                )
        macro_available = cutoff - timedelta(days=1)
        for series_id, values in alfred_values.items():
            records.append(
                Observation(
                    source="alfred",
                    series_id=series_id,
                    observed_period_end=macro_available.date(),
                    value=float(values[row_index]),
                    released_at=macro_available,
                    available_at=macro_available,
                    vintage_date=macro_available.date(),
                    retrieved_at=retrieved_at,
                    adjustment="synthetic",
                    license_class="synthetic_fixture",
                    quality_status=HealthStatus.OK,
                    raw_sha256=hashlib.sha256(f"{series_id}:{period}".encode()).hexdigest(),
                )
            )
    return cutoffs, tuple(records)


def generate_demo_payload(
    config: Mapping[str, Any],
    *,
    profile_name: str = "quick",
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], Any]:
    cutoffs, observations = _simulate_inputs(config)
    dataset = build_weekly_dataset(config, cutoffs, observations)
    sources = [
        {
            "id": "synthetic_market",
            "name": "Synthetic ETF fixture",
            "status": "degraded",
            "data_as_of": cutoffs[-1].isoformat(),
            "coverage": f"{cutoffs[0].date()}–{cutoffs[-1].date()}",
            "frequency": "weekly",
            "license_class": "synthetic_fixture",
        },
        {
            "id": "synthetic_macro",
            "name": "Synthetic macro vintage fixture",
            "status": "degraded",
            "data_as_of": cutoffs[-1].isoformat(),
            "coverage": f"{cutoffs[0].date()}–{cutoffs[-1].date()}",
            "frequency": "mixed → weekly",
            "license_class": "synthetic_fixture",
        },
    ]
    return build_dashboard_result(
        dataset,
        None,
        profile_name=profile_name,
        mode="demo",
        sources=sources,
        warnings=("고정 seed 모의자료이며 실제 미국 시장 판단이 아닙니다.",),
        progress=progress,
    )
