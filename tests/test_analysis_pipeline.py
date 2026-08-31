from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import regime_lab.pipeline as pipeline
from regime_lab import cli
from regime_lab.dataset import WeeklyDataset


class _BenchmarkIntercept(RuntimeError):
    pass


def _minimal_weekly_dataset(rows: int = 700) -> WeeklyDataset:
    index = pd.date_range("2012-01-06", periods=rows, freq="W-FRI")
    position = np.arange(rows, dtype=float)
    weekly_return = (
        0.001
        + 0.014 * np.sin(position / 11.0)
        + 0.008 * np.cos(position / 4.5)
    )
    spy = 100.0 * np.exp(np.cumsum(weekly_return))
    canonical = pd.DataFrame({"spy_close": spy}, index=index)
    features = pd.DataFrame(
        {
            "trend_proxy": np.sin(position / 11.0),
            "stress_proxy": np.cos(position / 4.5),
        },
        index=index,
    )
    return WeeklyDataset(
        canonical=canonical,
        features=features,
        availability=pd.DataFrame(index=index),
        health=pd.Series("ok", index=index, dtype="object"),
        feature_catalog=(),
        feature_group_manifest=(
            {
                "id": "legacy_v3",
                "description": "test legacy controls",
                "feature_count": len(features.columns),
                "features": tuple(features.columns),
            },
        ),
    )


def test_selection_diagnostics_derive_differences_from_serialized_metrics() -> None:
    table = pd.DataFrame(
        [
            {
                "model": "challenger",
                "reference_model": "markov",
                "selected": True,
                "gate_passed": True,
                "gate_reason": "passed",
                "log_loss": 0.234567894,
                "reference_log_loss": 0.345678905,
                "absolute_log_loss_improvement": 0.345678905 - 0.234567894,
                "brier": 0.123456784,
                "reference_brier": 0.234567895,
                "brier_difference": 0.123456784 - 0.234567895,
                "fallback_count": 0,
                "n_predictions": 52,
                "bootstrap_block_weeks": 4,
                "bootstrap_effective_block_weeks": 4,
                "bootstrap_resamples": 1_999,
                "bootstrap_seed": 17,
                "raw_p_value": 0.01,
                "holm_adjusted_p_value": 0.02,
                "alpha": 0.05,
                "minimum_log_loss_improvement": 0.01,
                "brier_tolerance": 0.0,
            }
        ]
    )

    row = pipeline._selection_diagnostic_rows(table)[0]

    np.testing.assert_allclose(
        row["absolute_log_loss_improvement"],
        row["reference_log_loss"] - row["log_loss"],
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        row["brier_difference"],
        row["brier"] - row["reference_brier"],
        atol=1e-10,
        rtol=0.0,
    )


def test_market_context_uses_scale_free_state_metrics_and_current_percentiles() -> None:
    index = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    position = np.arange(60, dtype=float)
    spy = 100.0 * np.power(1.01, position)
    canonical = pd.DataFrame({"spy_close": spy}, index=index)
    features = pd.DataFrame(
        {
            "market_group__gics_sector__positive_return_share_4w": np.linspace(
                0.2, 0.8, 60
            ),
            "market_spread__high_yield_investment_grade__relative_return_13w": (
                np.linspace(-0.04, 0.05, 60)
            ),
            "anfci__change_4w": np.linspace(-0.2, 0.3, 60),
        },
        index=index,
    )
    at = index[-1]

    context = pipeline._market_context(canonical, features, at)

    assert set(context) == {
        "spy_trend_26w",
        "spy_realized_vol_13w",
        "spy_drawdown_52w",
        "gics_sector_breadth_4w",
        "hyg_lqd_relative_13w",
        "anfci_change_4w",
    }
    np.testing.assert_allclose(
        context["spy_trend_26w"]["value"],
        np.power(1.01, 26) - 1.0,
    )
    assert context["spy_drawdown_52w"]["value"] == 0.0
    assert all("percentile_52w" in metric for metric in context.values())
    assert not any("price" in key or "close" in key for key in context)

    features.loc[at, "anfci__change_4w"] = np.nan
    missing = pipeline._market_context(canonical, features, at)
    assert missing["anfci_change_4w"]["value"] is None
    assert missing["anfci_change_4w"]["percentile_52w"] is None


@pytest.mark.parametrize(
    ("profile_name", "expected_minimum"),
    (("standard", 12), ("full", 12)),
)
def test_pipeline_passes_profile_specific_time_split_minimums(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    expected_minimum: int,
) -> None:
    captured: dict[str, object] = {}
    progress_messages: list[str] = []

    def intercept(*args, **kwargs):
        captured.update(kwargs)
        raise _BenchmarkIntercept

    monkeypatch.setattr(pipeline, "verify_frozen_v4_baseline", lambda: {})
    monkeypatch.setattr(pipeline, "run_benchmark", intercept)
    dataset = _minimal_weekly_dataset()

    with pytest.raises(_BenchmarkIntercept):
        pipeline.build_dashboard_result(
            dataset,
            None,
            profile_name=profile_name,
            mode="live",
            selection_end="2023-01-01",
            progress=progress_messages.append,
        )

    assert captured["selection_end"] == "2023-01-01"
    assert captured["minimum_selection_predictions"] == expected_minimum
    assert captured["minimum_holdout_predictions"] == expected_minimum
    assert captured["progress"] == progress_messages.append
    assert captured["checkpoint_directory"] is None
    assert captured["source_fingerprint_sha256"] is None
    assert captured["minimum_log_loss_improvement"] == 0.01


@pytest.mark.parametrize(
    ("contract_version", "override", "expected"),
    (("v4", 0.01, 0.05), ("v5", 0.05, 0.01)),
)
def test_pipeline_rejects_threshold_from_the_other_contract(
    contract_version: str,
    override: float,
    expected: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "verify_frozen_v4_baseline", lambda: {})

    with pytest.raises(
        ValueError,
        match=rf"{contract_version} minimum_log_loss_improvement must be {expected:.2f}",
    ):
        pipeline.build_dashboard_result(
            _minimal_weekly_dataset(),
            None,
            profile_name="standard",
            mode="live",
            selection_end="2023-01-01",
            contract_version=contract_version,
            minimum_log_loss_improvement=override,
        )


def test_pipeline_forwards_private_checkpoint_only_for_v5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def intercept(*args, **kwargs):
        captured.update(kwargs)
        raise _BenchmarkIntercept

    monkeypatch.setattr(pipeline, "verify_frozen_v4_baseline", lambda: {})
    monkeypatch.setattr(pipeline, "run_benchmark", intercept)
    checkpoint = tmp_path / "private-checkpoint"
    with pytest.raises(_BenchmarkIntercept):
        pipeline.build_dashboard_result(
            _minimal_weekly_dataset(),
            None,
            profile_name="standard",
            mode="live",
            selection_end="2023-01-01",
            contract_version="v5",
            checkpoint_directory=checkpoint,
            source_fingerprint_sha256="a" * 64,
        )

    assert captured["checkpoint_directory"] == checkpoint
    assert captured["source_fingerprint_sha256"] == "a" * 64
    assert captured["minimum_log_loss_improvement"] == 0.01

    with pytest.raises(ValueError, match="V5-only"):
        pipeline.build_dashboard_result(
            _minimal_weekly_dataset(),
            None,
            profile_name="standard",
            mode="live",
            selection_end="2023-01-01",
            contract_version="v4",
            checkpoint_directory=checkpoint,
        )


def _comparison_rows(origin: str, target: str) -> list[dict[str, object]]:
    rows = []
    for index, model in enumerate(pipeline.V5_FORECAST_COMPARISON_MODELS):
        transition = 0.6 + index * 0.02
        rows.append(
            {
                "origin_date": origin,
                "target_date": target,
                "model": model,
                "fallback": False,
                "fallback_reason": "",
                "p_risk_on": 0.3 - index * 0.01,
                "p_transition": transition,
                "p_risk_off": 0.1 - index * 0.01,
            }
        )
    return rows


def test_comparison_forecasts_combine_oos_and_latest_in_fixed_model_order() -> None:
    historical = pd.DataFrame(
        _comparison_rows("2026-08-07", "2026-08-14")
    )
    latest = pd.DataFrame(
        _comparison_rows("2026-08-14", "2026-08-21")
    )

    forecasts = pipeline._comparison_forecasts_by_origin(
        historical,
        latest,
        model_names=pipeline.V5_FORECAST_COMPARISON_MODELS,
    )

    assert list(forecasts) == [
        pd.Timestamp("2026-08-07", tz="UTC"),
        pd.Timestamp("2026-08-14", tz="UTC"),
    ]
    for origin, target in (
        (pd.Timestamp("2026-08-07", tz="UTC"), "2026-08-14"),
        (pd.Timestamp("2026-08-14", tz="UTC"), "2026-08-21"),
    ):
        rows = forecasts[origin]
        assert [row["model"] for row in rows] == list(
            pipeline.V5_FORECAST_COMPARISON_MODELS
        )
        assert {row["date"] for row in rows} == {target}
        assert {row["method"] for row in rows} == {
            "model_comparison_walk_forward_probability"
        }
        assert {row["state"] for row in rows} == {"transition"}


def test_comparison_forecasts_reject_incomplete_model_set() -> None:
    historical = pd.DataFrame(
        _comparison_rows("2026-08-07", "2026-08-14")[:-1]
    )
    latest = pd.DataFrame(
        _comparison_rows("2026-08-14", "2026-08-21")
    )

    with pytest.raises(RuntimeError, match="incomplete at"):
        pipeline._comparison_forecasts_by_origin(
            historical,
            latest,
            model_names=pipeline.V5_FORECAST_COMPARISON_MODELS,
        )


def test_cli_progress_printer_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture_print(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("builtins.print", capture_print)
    cli._flush_progress("walk-forward 1/10: 2026-01-02 → 2026-01-09")

    assert calls == [
        (
            ("walk-forward 1/10: 2026-01-02 → 2026-01-09",),
            {"flush": True},
        )
    ]


def test_live_pipeline_rejects_quick_smoke_profile() -> None:
    with pytest.raises(ValueError, match="quick smoke profile"):
        pipeline.build_dashboard_result(
            _minimal_weekly_dataset(),
            None,
            profile_name="quick",
            mode="live",
            selection_end="2023-01-01",
        )


def test_live_cli_exposes_only_claim_worthy_profiles() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["build", "--profile", "quick", "--alfred-rights-confirmed"]
        )
    for profile in ("standard", "full"):
        args = parser.parse_args(
            ["build", "--profile", profile, "--alfred-rights-confirmed"]
        )
        assert args.profile == profile


def test_holdout_diagnostic_flags_material_frozen_champion_regret() -> None:
    leaderboard = pd.DataFrame(
        {
            "model": [
                "markov",
                "persistence",
                "extra_trees",
                "random_forest",
                "hist_gradient_boosting",
                "elastic_net_logistic",
                "calibrated_linear_svm",
                "majority",
            ],
            "log_loss": [0.57794, 0.65, 0.75, 0.85, 0.95, 1.0421, 1.10, 1.20],
        }
    )

    diagnostic = pipeline.holdout_diagnostic(
        leaderboard,
        "elastic_net_logistic",
        selection_locked=True,
    )

    assert diagnostic == {
        "status": "weak_generalization",
        "applicable": True,
        "selection_locked": True,
        "metric": "multiclass_log_loss",
        "material_regret_threshold": 0.05,
        "champion_rank": 6,
        "model_count": 8,
        "champion_model": "elastic_net_logistic",
        "champion_log_loss": 1.0421,
        "best_model": "markov",
        "best_log_loss": 0.57794,
        "absolute_regret": 0.46416,
    }


def test_holdout_diagnostic_is_safe_for_legacy_demo() -> None:
    diagnostic = pipeline.holdout_diagnostic(
        pd.DataFrame(
            {"model": ["markov", "persistence"], "log_loss": [0.4, 0.5]}
        ),
        "markov",
        selection_locked=False,
    )

    assert diagnostic["status"] == "ok"
    assert diagnostic["applicable"] is False
    assert diagnostic["selection_locked"] is False
    assert diagnostic["champion_rank"] is None
    assert diagnostic["absolute_regret"] is None


def test_weak_holdout_degrades_only_global_live_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Global degradation is driven by the frozen-family diagnostic while
    # individual weekly data-health rows remain source/PIT health.  The full
    # v4 pipeline is exercised by the end-to-end demo/live tests; keep this
    # unit focused on the policy rather than recreating structural artifacts.
    diagnostic = pipeline.holdout_diagnostic(
        pd.DataFrame(
            {
                "model": ["markov", "xgboost"],
                "log_loss": [0.75, 0.40],
            }
        ),
        "markov",
        selection_locked=True,
    )
    assert diagnostic["status"] == "weak_generalization"
    assert diagnostic["absolute_regret"] == pytest.approx(0.35)
    warning = pipeline._holdout_generalization_warning(diagnostic)
    assert "사후 진단 결과를 이용해 champion을 교체하지 않았습니다" in warning
    assert "xgboost" in warning
