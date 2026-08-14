from __future__ import annotations

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
