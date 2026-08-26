from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from regime_lab.analysis.label_evaluation import (
    evaluate_label_definition,
    prefix_stability_report,
)
from regime_lab.analysis.label_research import (
    ResearchRegimeLabeler,
    _complete_equal_weight_share,
    raw_research_label_components,
)
from regime_lab.analysis.label_spec import (
    BROAD_EQUITY_BREADTH_SYMBOLS,
    FIXED_NINE_SECTORS,
    LABEL_SPEC_VERSION_LOCK,
    LabelSpecError,
    default_label_spec_path,
    label_spec_manifest_document,
    load_label_spec,
    load_label_spec_registry,
)
from regime_lab.analysis.labels import CausalRegimeLabeler
from regime_lab.analysis.pit_total_return import (
    CORPORATE_ACTION_CONTRACT,
    PITTotalReturnPanel,
    build_pit_total_return_panel,
    reconstruct_pit_total_return,
)
from regime_lab.schema import STATE_ORDER


PIT_SYMBOLS = (
    "spy",
    "rsp",
    "iwm",
    "xlb",
    "xle",
    "xlf",
    "xli",
    "xlk",
    "xlp",
    "xlu",
    "xlv",
    "xly",
)


def _pit_level_frame(rows: int = 620) -> pd.DataFrame:
    index = pd.date_range("2000-01-07", periods=rows, freq="W-FRI")
    generator = np.random.default_rng(20260826)
    common = generator.normal(0.001, 0.018, rows)
    values: dict[str, np.ndarray] = {}
    for position, symbol in enumerate(PIT_SYMBOLS):
        returns = (
            common * (1.0 + 0.03 * (position % 4))
            + generator.normal(0.0, 0.005 + 0.0002 * position, rows)
        )
        values[f"{symbol}_pit_total_return"] = 100.0 * np.exp(
            np.cumsum(returns)
        )
    return pd.DataFrame(values, index=index)


def _pit_panel_from_levels(levels: pd.DataFrame) -> PITTotalReturnPanel:
    decision_values = (
        pd.DatetimeIndex(levels.index).tz_localize("UTC")
        + timedelta(days=1, hours=12)
    )
    decisions = pd.Series(
        decision_values,
        index=levels.index,
    )
    source_release = pd.Series(
        decision_values - timedelta(hours=14), index=levels.index
    )
    first_seen = pd.Series(
        decision_values - timedelta(hours=13, minutes=55), index=levels.index
    )
    retrieved = pd.Series(
        decision_values - timedelta(hours=13, minutes=54), index=levels.index
    )
    results = {}
    for column in levels:
        symbol = column.removesuffix("_pit_total_return").upper()
        close = pd.to_numeric(levels[column], errors="raise").astype(float)
        frame = pd.DataFrame(
            {
                "raw_close": close,
                "dividend_amount": 0.0,
                "split_coefficient": 1.0,
                "corporate_action_contract": CORPORATE_ACTION_CONTRACT,
                "source_released_at": source_release,
                "provider_first_seen_at": first_seen,
                "system_retrieved_at": retrieved,
                "revision_seq": 0,
                "raw_sha256": [
                    hashlib.sha256(
                        f"{symbol}|{position}|{value:.17g}".encode("utf-8")
                    ).hexdigest()
                    for position, value in enumerate(close)
                ],
            },
            index=levels.index,
        )
        results[symbol] = reconstruct_pit_total_return(
            frame,
            decision_at=decisions,
            evidence_track="operational_oos",
        )
    return build_pit_total_return_panel(results)


def _pit_total_return_panel(rows: int = 620) -> PITTotalReturnPanel:
    return _pit_panel_from_levels(_pit_level_frame(rows))


def _legacy_v1_frame(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2019-01-04", periods=rows, freq="W-FRI")
    generator = np.random.default_rng(20260811)
    cycle = 0.012 * np.sin(np.arange(rows) / 9.0)
    shocks = generator.normal(0.001, 0.018, rows) + cycle
    return pd.DataFrame(
        {"spy_close": 100.0 * np.exp(np.cumsum(shocks))},
        index=index,
    )


def _legacy_v1_reference_outputs(
    frame: pd.DataFrame,
    *,
    train_rows: int,
) -> tuple[float, float, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Reproduce the pre-registry V1 implementation without shared helpers."""

    def raw_components(source: pd.DataFrame) -> pd.DataFrame:
        price = pd.to_numeric(source["spy_close"], errors="coerce").astype(float)
        log_price = np.log(price.where(price > 0.0))
        weekly_return = log_price.diff(1)
        output: dict[str, pd.Series] = {}
        for lookback in (13, 26):
            volatility = weekly_return.rolling(
                lookback,
                min_periods=max(4, lookback // 2),
            ).std(ddof=0)
            output[f"trend_{lookback}w"] = log_price.diff(lookback) / (
                volatility.replace(0.0, np.nan) * np.sqrt(float(lookback))
            )
        for window in (4, 13):
            output[f"vol_{window}w"] = weekly_return.rolling(
                window,
                min_periods=max(2, window // 2),
            ).std(ddof=0) * np.sqrt(52.0)
        for window in (13, 52):
            peak = price.rolling(
                window,
                min_periods=max(4, window // 2),
            ).max()
            output[f"drawdown_{window}w"] = -(price / peak - 1.0)
        return pd.DataFrame(output, index=source.index).replace(
            [np.inf, -np.inf], np.nan
        )

    def robust_location_scale(series: pd.Series) -> tuple[float, float]:
        finite = pd.to_numeric(series, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        centre = float(finite.median())
        q25, q75 = finite.quantile([0.25, 0.75]).to_numpy(dtype=float)
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-12:
            mad = float((finite - centre).abs().median() * 1.4826)
            scale = mad if np.isfinite(mad) and mad > 1e-12 else 1.0
        return centre, scale

    train_raw = raw_components(frame.iloc[:train_rows])
    component_stats = {
        column: robust_location_scale(train_raw[column])
        for column in train_raw
    }

    def standardized_components(raw: pd.DataFrame) -> pd.DataFrame:
        standardized = pd.DataFrame(index=raw.index)
        for column, (centre, scale) in component_stats.items():
            standardized[column] = (raw[column] - centre) / scale
        return standardized

    train_standardized = standardized_components(train_raw)
    train_trend = train_standardized[["trend_13w", "trend_26w"]].mean(axis=1)
    train_stress = train_standardized[
        ["vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w"]
    ].mean(axis=1)
    trend_stats = robust_location_scale(train_trend)
    stress_stats = robust_location_scale(train_stress)
    train_risk = (
        (train_trend - trend_stats[0]) / trend_stats[1]
        - (train_stress - stress_stats[0]) / stress_stats[1]
    ).dropna()
    lower = float(train_risk.quantile(0.30))
    upper = float(train_risk.quantile(0.70))

    standardized = standardized_components(raw_components(frame))
    trend_raw = standardized[["trend_13w", "trend_26w"]].mean(axis=1)
    stress_raw = standardized[
        ["vol_4w", "vol_13w", "drawdown_13w", "drawdown_52w"]
    ].mean(axis=1)
    trend_score = (trend_raw - trend_stats[0]) / trend_stats[1]
    stress_score = (stress_raw - stress_stats[0]) / stress_stats[1]
    scores = pd.DataFrame(
        {
            "trend_score": trend_score,
            "stress_score": stress_score,
            "risk_score": trend_score - stress_score,
        },
        index=frame.index,
    )

    state = "transition"
    margin = (upper - lower) * 0.15
    labels: list[str] = []
    for value in scores["risk_score"].to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append(state)
            continue
        if state == "transition":
            if value <= lower:
                state = "risk_off"
            elif value >= upper:
                state = "risk_on"
        elif state == "risk_on":
            if value <= lower - margin:
                state = "risk_off"
            elif value < upper - margin:
                state = "transition"
        else:
            if value >= upper + margin:
                state = "risk_on"
            elif value > lower + margin:
                state = "transition"
        labels.append(state)
    label_series = pd.Series(
        labels,
        index=frame.index,
        name="regime",
        dtype="object",
    )

    risk_values = scores["risk_score"].to_numpy(dtype=float)
    width = max(upper - lower, 1e-6)
    anchors = np.asarray(
        [upper + width / 2.0, (lower + upper) / 2.0, lower - width / 2.0]
    )
    scaled_distance = (risk_values[:, None] - anchors[None, :]) / width
    logits = -(scaled_distance**2) / 0.75
    logits[~np.isfinite(risk_values)] = np.asarray([-20.0, 0.0, -20.0])
    logits -= np.max(logits, axis=1, keepdims=True)
    memberships = np.exp(logits)
    memberships /= memberships.sum(axis=1, keepdims=True)
    membership_frame = pd.DataFrame(
        memberships,
        index=frame.index,
        columns=STATE_ORDER,
    )
    return lower, upper, label_series, scores, membership_frame


def test_typed_registry_freezes_v1_and_registers_only_unrun_shadows() -> None:
    registry = load_label_spec_registry()
    assert registry.default_spec == "v1_spy_hysteresis"
    assert tuple(registry.state_order) == STATE_ORDER
    assert set(registry.specs) == {
        "v1_spy_hysteresis",
        "v2_spy_pit_total_return",
        "v2_broad_equity",
    }
    frozen = registry.get()
    assert frozen.version == "market-causal-3state-v1"
    assert frozen.status == "official_frozen"
    assert frozen.membership.semantics == "distance_to_anchor_not_posterior"
    assert frozen.spec_sha256 == LABEL_SPEC_VERSION_LOCK[frozen.version]
    assert all(
        method.status.endswith("_unrun") and method.results is None
        for method in registry.shadow_methods.values()
    )

    manifest = label_spec_manifest_document()
    assert all(row["results"] is None for row in manifest["shadow_methods"])
    assert {
        row["membership_semantics"] for row in manifest["specs"]
    } == {"distance_to_anchor_not_posterior"}


def test_label_spec_hash_is_deterministic_and_mutation_needs_version_bump(
    tmp_path: Path,
) -> None:
    first = load_label_spec_registry()
    second = load_label_spec_registry()
    assert {
        name: spec.spec_sha256 for name, spec in first.specs.items()
    } == {
        name: spec.spec_sha256 for name, spec in second.specs.items()
    }

    document = json.loads(default_label_spec_path().read_text(encoding="utf-8"))
    document["specs"]["v1_spy_hysteresis"]["quantiles"]["lower"] = 0.29
    changed_path = tmp_path / "changed-label-spec.json"
    changed_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(LabelSpecError, match="changed without a version bump"):
        load_label_spec_registry(changed_path)

    document["specs"]["v1_spy_hysteresis"]["version"] = (
        "market-causal-3state-v1-unregistered"
    )
    bumped_path = tmp_path / "unregistered-label-spec.json"
    bumped_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(LabelSpecError, match="unregistered label spec version"):
        load_label_spec_registry(bumped_path)

    shadow_document = json.loads(
        default_label_spec_path().read_text(encoding="utf-8")
    )
    shadow_document["shadow_methods"]["pagan_sossounov_bull_bear"][
        "configuration"
    ]["window_months"] = 7
    shadow_path = tmp_path / "changed-shadow-label-spec.json"
    shadow_path.write_text(json.dumps(shadow_document), encoding="utf-8")
    with pytest.raises(LabelSpecError, match="changed without a version bump"):
        load_label_spec_registry(shadow_path)


def test_v1_default_output_matches_frozen_legacy_reference() -> None:
    frame = _legacy_v1_frame()
    labeler = CausalRegimeLabeler().fit(frame.iloc[:120])
    labels = labeler.transform(frame)
    assert hashlib.sha256(
        "|".join(labels.tolist()).encode("utf-8")
    ).hexdigest() == (
        "999b845d5caba658f939968668e4fbabc0d21b45f32e444be6ea8d2145cbc576"
    )
    (
        reference_lower,
        reference_upper,
        reference_labels,
        reference_scores,
        reference_memberships,
    ) = _legacy_v1_reference_outputs(frame, train_rows=120)
    assert labeler.lower_threshold_ == reference_lower
    assert labeler.upper_threshold_ == reference_upper
    assert_series_equal(labels, reference_labels, check_exact=True)
    assert_frame_equal(
        labeler.score_frame(frame),
        reference_scores,
        check_exact=True,
    )
    assert_frame_equal(
        labeler.state_probabilities(frame),
        reference_memberships,
        check_exact=True,
    )
    assert_frame_equal(
        labeler.state_probabilities(frame),
        labeler.state_memberships(frame),
    )
    assert labeler.config.membership_semantics == "distance_to_anchor_not_posterior"


def test_broad_equity_block_contract_and_fixed_breadth_denominator() -> None:
    specification = load_label_spec("v2_broad_equity")
    assert tuple(
        item.symbol for item in specification.series_for_block("direction")
    ) == ("SPY", "RSP", "IWM")
    assert tuple(
        item.symbol for item in specification.series_for_block("stress")
    ) == ("SPY",)
    assert tuple(
        item.symbol for item in specification.series_for_block("breadth")
    ) == BROAD_EQUITY_BREADTH_SYMBOLS
    assert BROAD_EQUITY_BREADTH_SYMBOLS == ("RSP", "IWM", *FIXED_NINE_SECTORS)

    index = pd.date_range("2025-01-03", periods=14, freq="W-FRI")
    levels = pd.DataFrame(
        {
            f"{symbol}_pit_total_return": np.full(len(index), 100.0)
            for symbol in PIT_SYMBOLS
        },
        index=index,
    )
    for symbol in ("rsp", "iwm", "xlb", "xle", "xlf", "xli", "xlk", "xlp", "xlu", "xlv"):
        levels.loc[index[-1], f"{symbol}_pit_total_return"] = 110.0
    levels.loc[index[-1], "xly_pit_total_return"] = 90.0
    panel = _pit_panel_from_levels(levels)
    raw = raw_research_label_components(panel, specification)

    assert set(raw["stress"]) == {
        "stress__spy__vol_4w",
        "stress__spy__vol_13w",
        "stress__spy__drawdown_13w",
        "stress__spy__drawdown_52w",
    }
    assert raw["breadth"].loc[
        index[-1], "breadth__positive_return_share_1w"
    ] == pytest.approx(10.0 / 11.0)
    assert raw["breadth"].loc[
        index[-1], "breadth__positive_trend_share_13w"
    ] == pytest.approx(10.0 / 11.0)

    changes = pd.DataFrame(
        [[1.0, 1.0, np.nan]],
        columns=["a", "b", "c"],
    )
    assert np.isnan(_complete_equal_weight_share(changes).iloc[0])


@pytest.mark.parametrize(
    "spec_id", ["v2_spy_pit_total_return", "v2_broad_equity"]
)
def test_challengers_freeze_train_only_prefix_and_are_prefix_stable(
    spec_id: str,
) -> None:
    levels = _pit_level_frame()
    changed_levels = levels.copy()
    changed_levels.iloc[520:] *= np.linspace(
        1.0, 5.0, len(changed_levels) - 520
    )[:, None]
    if spec_id == "v2_spy_pit_total_return":
        levels = levels[["spy_pit_total_return"]]
        changed_levels = changed_levels[["spy_pit_total_return"]]
    panel = _pit_panel_from_levels(levels)
    changed = _pit_panel_from_levels(changed_levels)

    first = ResearchRegimeLabeler(spec_id).fit(panel)
    second = ResearchRegimeLabeler(spec_id).fit(changed)
    assert first.fit_row_count_ == second.fit_row_count_ == 520
    assert first.train_end_ == second.train_end_ == panel.index[519]
    assert first.lower_threshold_ == second.lower_threshold_
    assert first.upper_threshold_ == second.upper_threshold_
    assert first.component_stats_ == second.component_stats_
    assert first.block_stats_ == second.block_stats_
    assert first.membership_semantics == "distance_to_anchor_not_posterior"

    assert_series_equal(
        first.transform(panel.slice_rows(520)),
        second.transform(changed.slice_rows(520)),
    )
    assert_frame_equal(
        first.score_frame(panel.slice_rows(520)),
        second.score_frame(changed.slice_rows(520)),
    )
    memberships = first.state_memberships(panel)
    assert tuple(memberships.columns) == STATE_ORDER
    np.testing.assert_allclose(memberships.sum(axis=1), 1.0, atol=1e-12)

    stability = prefix_stability_report(
        first,
        panel,
        prefix_lengths=(520, 560, 620),
    )
    assert stability["stable"].all()
    assert (stability["maximum_absolute_score_difference"] == 0.0).all()
    assert (stability["maximum_absolute_membership_difference"] == 0.0).all()


def test_v2_labeler_rejects_unbound_or_mutated_pit_inputs() -> None:
    panel = _pit_total_return_panel()
    with pytest.raises(TypeError, match="provenance-bound"):
        ResearchRegimeLabeler("v2_spy_pit_total_return").fit(
            panel.frame[["spy_pit_total_return"]]
        )

    spy_panel = build_pit_total_return_panel({"SPY": panel.results["SPY"]})
    spy_panel.frame.iloc[10, 0] *= 2.0
    with pytest.raises(ValueError, match="mutated"):
        ResearchRegimeLabeler("v2_spy_pit_total_return").fit(spy_panel)


def test_label_evaluation_covers_all_required_descriptive_dimensions() -> None:
    index = pd.date_range("2025-01-03", periods=18, freq="W-FRI")
    states = pd.Series(
        [
            "risk_on",
            "risk_on",
            "risk_on",
            "transition",
            "transition",
            "risk_off",
            "risk_off",
            "transition",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
            "risk_on",
        ],
        index=index,
        name="regime",
    )
    prices = pd.DataFrame(
        {
            "spy": [
                100.0,
                105.0,
                110.0,
                100.0,
                85.0,
                80.0,
                90.0,
                110.0,
                112.0,
                114.0,
                116.0,
                118.0,
                120.0,
                122.0,
                124.0,
                126.0,
                128.0,
                130.0,
            ]
        },
        index=index,
    )
    alternative = states.copy()
    alternative.iloc[3] = "risk_off"
    supplied_prefix = pd.DataFrame(
        [{"prefix_rows": 10, "stable": True}]
    )
    result = evaluate_label_definition(
        states,
        external_prices=prices,
        asset_columns={"SPY": "spy"},
        crash_asset="SPY",
        sensitivity_labels={"one_week_changed": alternative},
        prefix_stability=supplied_prefix,
    )

    assert result.occupancy["share"].sum() == pytest.approx(1.0)
    assert set(result.durations["state"]) == set(STATE_ORDER)
    assert int(result.flips.iloc[0]["flips"]) == 4
    assert set(result.external_outcomes["horizon_weeks"]) == {1, 4, 13}
    assert {
        "mean_return",
        "annualized_volatility",
        "mean_max_drawdown",
        "worst_max_drawdown",
    }.issubset(result.external_outcomes.columns)
    event = result.crash_recovery.iloc[0]
    assert event["peak_date"] == index[2]
    assert event["crash_date"] == index[4]
    assert event["recovery_date"] == index[7]
    assert event["crash_detection_lag_weeks"] == 1
    assert event["recovery_detection_lag_weeks"] == 1
    assert result.sensitivity.iloc[0]["changed_weeks"] == 1
    assert result.sensitivity.iloc[0]["agreement"] == pytest.approx(17.0 / 18.0)
    assert_frame_equal(result.prefix_stability, supplied_prefix)
