"""Convert revision-aware observations into a causal weekly model dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from regime_lab.analysis import FeatureConfig, build_weekly_features
from regime_lab.analysis.structural_features import (
    build_bank_credit_features,
    build_nelson_siegel_features,
    build_release_innovation_features,
    build_structural_feature_manifest,
)
from regime_lab.data import AsOfValue, HealthStatus, Observation, weekly_asof_join


_DEFAULT_ALPHA_FIELDS = ("adjusted_close", "volume")
_DEFAULT_OHLC_FEATURE_SYMBOLS = ("SPY", "IWM", "RSP", "HYG", "TLT")
_ANFCI_STRUCTURAL_FEATURES = frozenset(
    {
        "anfci__level",
        "anfci__change_1w",
        "anfci__change_4w",
        "anfci__z_52w",
    }
)


@dataclass(frozen=True)
class WeeklyDataset:
    canonical: pd.DataFrame
    features: pd.DataFrame
    availability: pd.DataFrame
    health: pd.Series
    feature_catalog: tuple[dict[str, Any], ...]
    feature_group_manifest: tuple[dict[str, Any], ...] = ()
    availability_basis: Literal[
        "source", "operational", "reconstructed_market"
    ] = "source"
    input_vintages: tuple[AsOfValue, ...] = ()
    latest_input_vintages: tuple[AsOfValue, ...] = ()


def _alpha_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in config["alpha_vantage"].get(
                "fields",
                _DEFAULT_ALPHA_FIELDS,
            )
            if str(item).strip()
        )
    )


def _alpha_research_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in config["alpha_vantage"].get("research_fields", ())
            if str(item).strip()
        )
    )


def _alpha_dataset_fields(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*_alpha_fields(config), *_alpha_research_fields(config))))


def _required_series(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    alpha = [
        ("alpha_vantage", f"{symbol}.{field}")
        for symbol in config["alpha_vantage"]["symbols"]
        for field in _alpha_dataset_fields(config)
    ]
    alfred = [
        ("alfred", str(item["id"])) for item in config["alfred"]["series"]
    ]
    return tuple(alpha + alfred)


def _max_ages(config: Mapping[str, Any]) -> dict[tuple[str, str], timedelta]:
    result: dict[tuple[str, str], timedelta] = {}
    for symbol in config["alpha_vantage"]["symbols"]:
        for field in _alpha_dataset_fields(config):
            result[("alpha_vantage", f"{symbol}.{field}")] = timedelta(days=10)
    by_frequency = {
        "daily": timedelta(days=14),
        "weekly": timedelta(days=21),
        "monthly": timedelta(days=75),
        "quarterly": timedelta(days=150),
    }
    for item in config["alfred"]["series"]:
        result[("alfred", str(item["id"]))] = by_frequency.get(
            str(item.get("frequency")), timedelta(days=90)
        )
    return result


def _column_name(source: str, series_id: str) -> str:
    if source == "alpha_vantage":
        symbol, field = series_id.split(".", 1)
        if field == "adjusted_close":
            return f"{symbol.lower()}_close"
        if field in {"open", "high", "low", "close"}:
            # Provider OHLC fields are unadjusted.  Keep that fact explicit so
            # raw close never collides with the model's adjusted close column.
            return f"{symbol.lower()}_raw_{field}"
        return f"{symbol.lower()}_{field}"
    return series_id.lower()


def _pivot_metric(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = rows.pivot(index="cutoff", columns=["source", "series_id"], values=metric)
    frame.columns = [_column_name(source, series_id) for source, series_id in frame.columns]
    return frame.sort_index()


def _status_priority(value: object) -> int:
    order = {
        "ok": 0,
        "stale": 1,
        "unavailable": 2,
        "degraded": 3,
        "revision_gap": 4,
        "quota_exhausted": 5,
        "rights_unconfirmed": 6,
        "schema_changed": 7,
        "license_blocked": 8,
    }
    raw = value.value if isinstance(value, HealthStatus) else str(value)
    return order.get(raw, 9)


def _weekly_health(rows: pd.DataFrame) -> pd.Series:
    def worst(group: pd.Series) -> str:
        values = [item.value if isinstance(item, HealthStatus) else str(item) for item in group]
        return max(values, key=_status_priority, default="unavailable")

    return rows.groupby("cutoff", sort=True)["quality_status"].apply(worst)


def _adjusted_ohlc(
    canonical: pd.DataFrame,
    symbols: Sequence[str],
    observed_periods: pd.DataFrame | None = None,
    *,
    feature_symbols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build split-consistent OHLC values and compact causal diagnostics.

    Alpha Vantage reports raw OHLC alongside an adjusted close.  Multiplying
    each row's raw OHLC by ``adjusted_close / raw_close`` puts all four prices
    on the same split/dividend-adjusted scale without interpolation or a
    future-row dependency.  A gap uses only the immediately preceding weekly
    adjusted close and therefore remains missing across a missing input week.

    Adjusted audit columns are materialized for every symbol.  Compact model
    diagnostics remain limited to ``feature_symbols`` so expanding outcome
    price coverage cannot silently change the forecasting feature set.
    """

    adjusted: dict[str, pd.Series] = {}
    features: dict[str, pd.Series] = {}
    feature_symbol_source = symbols if feature_symbols is None else feature_symbols
    resolved_feature_symbols = {
        str(configured_symbol).strip().lower()
        for configured_symbol in feature_symbol_source
        if str(configured_symbol).strip()
    }
    for configured_symbol in symbols:
        symbol = str(configured_symbol).strip().lower()
        if not symbol:
            continue
        columns = {
            field: f"{symbol}_raw_{field}"
            for field in ("open", "high", "low", "close")
        }
        adjusted_close_column = f"{symbol}_close"
        required = (*columns.values(), adjusted_close_column)
        if any(column not in canonical for column in required):
            missing = pd.Series(np.nan, index=canonical.index, dtype=float)
            for field in ("open", "high", "low"):
                adjusted[f"{symbol}_adjusted_{field}"] = missing.copy()
            adjusted[f"{symbol}_adjustment_factor"] = missing.copy()
            if symbol in resolved_feature_symbols:
                prefix = f"market_ohlc__{symbol}"
                for suffix in (
                    "log_high_low_range_1w",
                    "close_location_1w",
                    "log_gap_1w",
                ):
                    features[f"{prefix}__{suffix}"] = missing.copy()
            continue

        raw = {
            field: pd.to_numeric(canonical[column], errors="coerce").astype(float)
            for field, column in columns.items()
        }
        adjusted_close = pd.to_numeric(
            canonical[adjusted_close_column], errors="coerce"
        ).astype(float)
        same_period = pd.Series(True, index=canonical.index, dtype=bool)
        if observed_periods is not None:
            if any(column not in observed_periods for column in required):
                same_period = pd.Series(False, index=canonical.index, dtype=bool)
            else:
                reference_period = observed_periods[adjusted_close_column]
                same_period = reference_period.notna()
                for column in required[:-1]:
                    same_period &= observed_periods[column].eq(reference_period)
        factor = (adjusted_close / raw["close"]).where(
            same_period & (adjusted_close > 0.0) & (raw["close"] > 0.0)
        )
        adjusted_prices = {
            field: (raw[field] * factor).where(raw[field] > 0.0)
            for field in ("open", "high", "low")
        }
        adjusted_prices["close"] = adjusted_close.where(adjusted_close > 0.0)

        for field in ("open", "high", "low"):
            adjusted[f"{symbol}_adjusted_{field}"] = adjusted_prices[field]
        adjusted[f"{symbol}_adjustment_factor"] = factor

        if symbol not in resolved_feature_symbols:
            continue

        valid_range = (
            (adjusted_prices["high"] > 0.0)
            & (adjusted_prices["low"] > 0.0)
            & (adjusted_prices["high"] >= adjusted_prices["low"])
        )
        log_range = np.log(
            adjusted_prices["high"] / adjusted_prices["low"]
        ).where(valid_range)
        width = adjusted_prices["high"] - adjusted_prices["low"]
        close_location = (
            (adjusted_prices["close"] - adjusted_prices["low"])
            / width.replace(0.0, np.nan)
        ).where(valid_range).clip(0.0, 1.0)
        previous_close = adjusted_prices["close"].shift(1)
        log_gap = np.log(
            adjusted_prices["open"] / previous_close
        ).where((adjusted_prices["open"] > 0.0) & (previous_close > 0.0))

        prefix = f"market_ohlc__{symbol}"
        features[f"{prefix}__log_high_low_range_1w"] = log_range
        features[f"{prefix}__close_location_1w"] = close_location
        features[f"{prefix}__log_gap_1w"] = log_gap

    return (
        pd.DataFrame(adjusted, index=canonical.index, dtype=float),
        pd.DataFrame(features, index=canonical.index, dtype=float),
    )


def _select_features(engineered: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    selected: list[str] = []
    for column in engineered.columns:
        if column.startswith("spy_close__"):
            selected.append(column)
        elif "_vs_spy_close__relative_return_" in column:
            selected.append(column)
        elif column.startswith("spy_volume__"):
            selected.append(column)
        elif column.startswith(
            (
                "market_internal__",
                "market_spread__",
                "volume_internal__",
                "market_ohlc__",
                "market_group__",
                "treasury_curve__",
                "bank_credit__",
                "release_innovation__",
            )
        ):
            selected.append(column)
        elif column.startswith("anfci__"):
            # ANFCI is a preregistered structural block.  Keep its exact four
            # PIT transforms instead of inheriting the generic macro selector,
            # which would silently omit 1w change and add unregistered terms.
            if column in _ANFCI_STRUCTURAL_FEATURES:
                selected.append(column)
        elif column in {
            "iwm_close__realized_vol_13w",
            "hyg_close__realized_vol_13w",
            "lqd_close__realized_vol_13w",
            "tlt_close__realized_vol_13w",
        }:
            selected.append(column)
        elif column.endswith(
            (
                "__level",
                "__change_4w",
                "__z_52w",
                "__change_4w_z_52w",
                "__missing",
            )
        ):
            # Exclude non-SPY price/volume levels; those are represented as
            # scale-free relative returns above.
            base = column.split("__", 1)[0]
            if base.endswith("_volume") and base != "spy_volume":
                continue
            if not any(
                column.startswith(f"{prefix}_close__")
                for prefix in (
                    "qqq", "iwm", "dia", "rsp", "xlb", "xlc", "xly", "xlp",
                    "xlf", "xlk", "xli", "xle", "xlre", "xlv", "xlu", "shy",
                    "ief", "tlt", "hyg", "lqd", "gld", "uup",
                )
            ):
                selected.append(column)
    result = engineered.loc[:, list(dict.fromkeys(selected))].copy()
    result = pd.concat([result, availability], axis=1)
    # Columns that never exist in the requested history carry no information.
    return result.loc[:, result.notna().any(axis=0)]


def build_weekly_dataset(
    config: Mapping[str, Any],
    cutoffs: Sequence[object],
    observations: Sequence[Observation],
    *,
    availability_basis: Literal[
        "source", "operational", "reconstructed_market"
    ] = "source",
) -> WeeklyDataset:
    required = _required_series(config)
    selected = weekly_asof_join(
        cutoffs,
        observations,
        required_series=required,
        max_age_by_series=_max_ages(config),
        availability_basis=availability_basis,
    )
    rows = pd.DataFrame(asdict(row) for row in selected)
    if rows.empty:
        raise RuntimeError("as-of join produced no weekly rows")
    latest_cutoff = max(row.cutoff for row in selected)
    eligible_vintages = (
        row
        for row in selected
        if row.value is not None
        and row.observed_period_end is not None
        and row.provider_first_seen_at is not None
        and row.system_retrieved_at is not None
        and row.revision_seq is not None
        and row.raw_sha256 is not None
    )
    input_vintages = tuple(
        {
            (
                row.source,
                row.series_id,
                row.observed_period_end,
                row.source_released_at,
                row.provider_first_seen_at,
                row.system_retrieved_at,
                row.revision_seq,
                row.raw_sha256,
            ): row
            for row in eligible_vintages
        }.values()
    )
    latest_input_vintages = tuple(
        row
        for row in input_vintages
        if row.cutoff == latest_cutoff
    )
    canonical = _pivot_metric(rows, "value").astype(float)

    alpha_rows = rows.loc[rows["source"] == "alpha_vantage"].copy()
    alpha_observed_periods = _pivot_metric(alpha_rows, "observed_period_end")

    configured_symbols = tuple(
        str(item).strip().upper()
        for item in config["alpha_vantage"]["symbols"]
        if str(item).strip()
    )
    ohlc_feature_symbols = tuple(
        str(item).strip().upper()
        for item in config["alpha_vantage"].get(
            "ohlc_feature_symbols",
            _DEFAULT_OHLC_FEATURE_SYMBOLS,
        )
        if str(item).strip()
    )
    unknown_ohlc_symbols = set(ohlc_feature_symbols).difference(configured_symbols)
    if unknown_ohlc_symbols:
        raise ValueError(
            "alpha_vantage ohlc_feature_symbols are not configured symbols: "
            f"{sorted(unknown_ohlc_symbols)}"
        )
    alpha_fields = set(_alpha_fields(config))
    required_base_fields = {"adjusted_close", "volume"}
    if not required_base_fields.issubset(alpha_fields):
        raise ValueError(
            "alpha_vantage fields must include adjusted_close and volume"
        )
    required_ohlc_fields = {"open", "high", "low", "close"}
    if configured_symbols and not required_ohlc_fields.issubset(alpha_fields):
        raise ValueError(
            "configured Alpha Vantage symbols require open, high, low, and close "
            "for adjusted OHLC"
        )
    adjusted_ohlc, ohlc_features = _adjusted_ohlc(
        canonical,
        configured_symbols,
        observed_periods=alpha_observed_periods,
        feature_symbols=ohlc_feature_symbols,
    )
    canonical = pd.concat([canonical, adjusted_ohlc], axis=1)

    alfred_rows = rows.loc[rows["source"] == "alfred"].copy()
    alfred_observed_periods = _pivot_metric(alfred_rows, "observed_period_end")
    alfred_revision_sequences = _pivot_metric(alfred_rows, "revision_seq")
    availability_parts: list[pd.DataFrame] = []
    for metric in ("age_days", "release_lag_days", "is_filled"):
        part = _pivot_metric(alfred_rows, metric)
        part.columns = [f"{column}__{metric}" for column in part.columns]
        availability_parts.append(part.astype(float))
    availability = pd.concat(availability_parts, axis=1).reindex(canonical.index)

    price_columns = tuple(
        f"{symbol.lower()}_close" for symbol in configured_symbols
    )
    volume_columns = tuple(
        f"{symbol.lower()}_volume" for symbol in configured_symbols
    )
    alfred_columns = tuple(
        str(item["id"]).lower() for item in config["alfred"]["series"]
    )
    # Raw provider OHLC and the row-wise adjustment factor are audit columns,
    # not generic macro levels.  Only adjusted closes, all configured volumes,
    # and ALFRED values enter the shared feature builder.
    feature_input_columns = tuple(
        column
        for column in (*price_columns, *volume_columns, *alfred_columns)
        if column in canonical
    )
    feature_input = canonical.loc[:, feature_input_columns].copy()
    raw_symbol_groups = config["alpha_vantage"].get("symbol_groups", {})
    if not isinstance(raw_symbol_groups, Mapping):
        raise ValueError("alpha_vantage symbol_groups must be a mapping")
    price_groups = {
        str(name): tuple(str(member) for member in members)
        for name, members in raw_symbol_groups.items()
    }
    engineered = build_weekly_features(
        feature_input,
        FeatureConfig(
            price_columns=price_columns,
            price_groups=price_groups or None,
            benchmark_column="spy_close",
            volume_columns=volume_columns,
            volatility_windows=(4, 13),
            generic_z_windows=(52,),
            relative_lookbacks=(13, 26),
        ),
    )
    structural_parts: list[pd.DataFrame] = []
    feature_engineering = config.get("feature_engineering", {})
    if not isinstance(feature_engineering, Mapping):
        raise ValueError("feature_engineering must be a mapping")

    nelson_siegel = feature_engineering.get("nelson_siegel")
    if nelson_siegel is not None:
        if not isinstance(nelson_siegel, Mapping):
            raise ValueError("feature_engineering nelson_siegel must be a mapping")
        series_months = nelson_siegel.get("series_months", {})
        if not isinstance(series_months, Mapping):
            raise ValueError("nelson_siegel series_months must be a mapping")
        structural_parts.append(
            build_nelson_siegel_features(
                feature_input,
                {str(key): float(value) for key, value in series_months.items()},
                lambda_per_month=float(
                    nelson_siegel.get("lambda_per_month", 0.0609)
                ),
                minimum_maturities=int(
                    nelson_siegel.get("minimum_maturities", 4)
                ),
            )
        )

    bank_credit = feature_engineering.get("bank_credit")
    if bank_credit is not None:
        if not isinstance(bank_credit, Mapping):
            raise ValueError("feature_engineering bank_credit must be a mapping")
        structural_parts.append(
            build_bank_credit_features(
                feature_input,
                total_credit=str(bank_credit.get("total_credit", "TOTBKCR")),
                commercial_industrial=str(
                    bank_credit.get("commercial_industrial", "TOTCI")
                ),
                deposits=str(
                    bank_credit.get("deposits", "DPSACBW027SBOG")
                ),
                borrowings_millions=str(
                    bank_credit.get("borrowings_millions", "H8B3094NCBA")
                ),
            )
        )

    release_innovation = feature_engineering.get("release_innovation")
    if release_innovation is not None:
        if not isinstance(release_innovation, Mapping):
            raise ValueError(
                "feature_engineering release_innovation must be a mapping"
            )
        structural_parts.append(
            build_release_innovation_features(
                feature_input,
                alfred_observed_periods.reindex(feature_input.index),
                revision_sequences=alfred_revision_sequences.reindex(
                    feature_input.index
                ),
                series=tuple(
                    str(item) for item in release_innovation.get("series", ())
                ),
                prior_release_window=int(
                    release_innovation.get("prior_release_window", 12)
                ),
                minimum_prior_releases=int(
                    release_innovation.get("minimum_prior_releases", 4)
                ),
            )
        )

    engineered = pd.concat(
        [engineered, ohlc_features, *structural_parts], axis=1
    )
    if engineered.columns.has_duplicates:
        duplicates = engineered.columns[engineered.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate engineered features: {duplicates}")
    features = _select_features(engineered, availability)
    feature_group_manifest = build_structural_feature_manifest(features.columns)
    health = _weekly_health(rows).reindex(canonical.index).fillna("unavailable")

    catalog: list[dict[str, Any]] = []
    for symbol in config["alpha_vantage"]["symbols"]:
        catalog.append(
            {
                "id": f"{symbol}.adjusted_close",
                "label": f"{symbol} 주별 조정종가",
                "category": "시장·크로스에셋",
                "frequency": "weekly",
                "source": "Alpha Vantage",
            }
        )
    if "volume" in _alpha_fields(config):
        catalog.append(
            {
                "id": "alpha_weekly_volume_internals",
                "label": f"{len(config['alpha_vantage']['symbols'])}개 ETF 거래량 breadth·confirmation",
                "category": "시장 내부·유동성",
                "frequency": "weekly",
                "source": "Alpha Vantage · derived",
            }
        )
    if {"open", "high", "low", "close"}.issubset(_alpha_fields(config)):
        catalog.append(
            {
                "id": "alpha_adjusted_ohlc_internals",
                "label": f"{len(ohlc_feature_symbols)}개 핵심 ETF 조정 OHLC range·gap",
                "category": "시장 내부·유동성",
                "frequency": "weekly",
                "source": "Alpha Vantage · derived",
            }
        )
    for item in config["alfred"]["series"]:
        catalog.append(
            {
                "id": item["id"],
                "label": item["id"],
                "category": str(item.get("domain", "거시경제")),
                "frequency": item.get("frequency"),
                "source": "ALFRED",
            }
        )
    catalog.extend(
        [
            {
                "id": "release_age",
                "label": "발표 후 경과일·fill 여부",
                "category": "Point-in-time 품질",
                "frequency": "weekly as-of",
                "source": "derived",
            },
            {
                "id": "market_regime_label",
                "label": "13/26주 추세·4/13주 변동성·13/52주 낙폭",
                "category": "Reference label",
                "frequency": "weekly",
                "source": "derived from SPY",
            },
        ]
    )
    for group in feature_group_manifest:
        if group["id"] == "legacy_v3":
            continue
        catalog.append(
            {
                "id": f"structural_{group['id']}",
                "label": group["description"],
                "category": "구조 피처",
                "frequency": "weekly as-of",
                "source": "derived",
                "feature_count": group["feature_count"],
            }
        )
    return WeeklyDataset(
        canonical=canonical,
        features=features,
        availability=availability,
        health=health,
        feature_catalog=tuple(catalog),
        feature_group_manifest=feature_group_manifest,
        availability_basis=availability_basis,
        input_vintages=input_vintages,
        latest_input_vintages=latest_input_vintages,
    )


def factor_scores(
    features: pd.DataFrame,
    label_scores: pd.DataFrame,
) -> pd.DataFrame:
    def signed_mean(candidates: Sequence[tuple[str, float]]) -> pd.Series:
        values = [features[column] * sign for column, sign in candidates if column in features]
        if not values:
            return pd.Series(0.0, index=features.index)
        return pd.concat(values, axis=1).mean(axis=1, skipna=True).fillna(0.0)

    macro = signed_mean(
        [
            ("payems__z_52w", 1.0),
            ("indpro__z_52w", 1.0),
            ("rsafs__z_52w", 1.0),
            ("houst__z_52w", 1.0),
            ("gdpc1__z_52w", 1.0),
            ("unrate__z_52w", -1.0),
            ("icsa__z_52w", -1.0),
            ("ccsa__z_52w", -1.0),
        ]
    )
    financial = signed_mean(
        [
            ("nfci__z_52w", -1.0),
            ("nfcirisk__z_52w", -0.5),
            ("nfcicredit__z_52w", -0.5),
            ("nfcileverage__z_52w", -0.5),
            ("nfcinonfinleverage__z_52w", -0.5),
            ("stlfsi4__z_52w", -1.0),
            ("t10y2y__z_52w", 1.0),
            ("walcl__z_52w", 0.5),
        ]
    )
    aligned_scores = label_scores.reindex(features.index)
    return pd.DataFrame(
        {
            "trend": np.tanh(aligned_scores["trend_score"].fillna(0.0) / 2.0),
            "stress": -np.tanh(aligned_scores["stress_score"].fillna(0.0) / 2.0),
            "macro": np.tanh(macro / 2.0),
            "financial_conditions": np.tanh(financial / 2.0),
        },
        index=features.index,
    ).clip(-1.0, 1.0)


def evidence_drivers(features: pd.DataFrame, at: pd.Timestamp, limit: int = 8) -> list[dict[str, Any]]:
    if at not in features.index:
        return []
    row = features.loc[at]
    candidates = [column for column in features if column.endswith("__z_52w")]
    ranked = sorted(
        (
            (column, float(row[column]))
            for column in candidates
            if pd.notna(row[column]) and np.isfinite(float(row[column]))
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]
    labels = {
        "nfci": "Chicago Fed 금융여건",
        "nfcirisk": "Chicago Fed 금융위험 하위지수",
        "nfcicredit": "Chicago Fed 신용 하위지수",
        "nfcileverage": "Chicago Fed 레버리지 하위지수",
        "nfcinonfinleverage": "Chicago Fed 비금융 레버리지 하위지수",
        "stlfsi4": "St. Louis Fed 금융스트레스",
        "icsa": "신규 실업수당 청구",
        "ccsa": "계속 실업수당 청구",
        "unrate": "실업률",
        "payems": "비농업 고용",
        "indpro": "산업생산",
        "rsafs": "소매판매",
        "houst": "주택착공",
        "t10y2y": "10년-2년 금리차",
        "dgs10": "미 국채 10년물",
        "dtwexbgs": "광의 달러지수",
    }
    adverse_positive = {
        "nfci",
        "nfcirisk",
        "nfcicredit",
        "nfcileverage",
        "nfcinonfinleverage",
        "stlfsi4",
        "icsa",
        "ccsa",
        "unrate",
        "dtwexbgs",
    }
    output: list[dict[str, Any]] = []
    for column, value in ranked:
        base = column.split("__", 1)[0]
        impact = -value if base in adverse_positive else value
        output.append(
            {
                "feature": column,
                "label": labels.get(base, base.upper()),
                "value": round(value, 4),
                "impact": round(float(np.clip(impact, -4.0, 4.0)), 4),
                "direction": "risk_on" if impact >= 0 else "risk_off",
                "method": "rolling_z_evidence_proxy",
            }
        )
    return output
