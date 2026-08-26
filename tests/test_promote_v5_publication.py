from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_v5_publication.py"
SPEC = importlib.util.spec_from_file_location("promote_v5_publication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
promote_v5_publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote_v5_publication)


@pytest.mark.parametrize(
    "reasons",
    [
        ["weak_generalization"],
        ["calibration_drift"],
        ["weak_generalization", "calibration_drift"],
    ],
)
def test_publication_health_accepts_only_reviewed_model_warnings(
    reasons: list[str],
) -> None:
    promote_v5_publication._validate_publication_health(
        {"status": "degraded"},
        {"model_health": {"status": "review_due", "reasons": reasons}},
    )


def test_publication_health_accepts_fully_healthy_candidate() -> None:
    promote_v5_publication._validate_publication_health(
        {"status": "ok"},
        {"model_health": {"status": "ok", "reasons": []}},
    )


@pytest.mark.parametrize(
    ("meta_status", "model_status", "reasons"),
    [
        ("failed", "review_due", ["weak_generalization"]),
        ("degraded", "degraded", ["weak_generalization"]),
        ("degraded", "review_due", []),
        ("degraded", "review_due", ["provider_degraded"]),
        (
            "degraded",
            "review_due",
            ["weak_generalization", "provider_degraded"],
        ),
        (
            "degraded",
            "review_due",
            ["weak_generalization", "weak_generalization"],
        ),
        ("ok", "review_due", ["weak_generalization"]),
        ("ok", "ok", ["calibration_drift"]),
    ],
)
def test_publication_health_rejects_unapproved_or_inconsistent_health(
    meta_status: str,
    model_status: str,
    reasons: list[str],
) -> None:
    with pytest.raises(promote_v5_publication.PromotionError):
        promote_v5_publication._validate_publication_health(
            {"status": meta_status},
            {"model_health": {"status": model_status, "reasons": reasons}},
        )


@pytest.mark.parametrize("reasons", [None, "weak_generalization", [1]])
def test_publication_health_rejects_malformed_reason_list(reasons: object) -> None:
    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="reasons are invalid",
    ):
        promote_v5_publication._validate_publication_health(
            {"status": "degraded"},
            {"model_health": {"status": "review_due", "reasons": reasons}},
        )


def _gate_row(model: str, *, champion: str) -> dict[str, object]:
    selected = model == champion
    reference = model == "markov"
    if reference:
        log_loss = reference_log_loss = 0.36
        brier = reference_brier = 0.18
        raw_p_value = holm_adjusted_p_value = None
    elif selected:
        log_loss, reference_log_loss = 0.34, 0.36
        brier, reference_brier = 0.17, 0.18
        raw_p_value, holm_adjusted_p_value = 0.001, 0.01
    else:
        log_loss, reference_log_loss = 0.355, 0.36
        brier, reference_brier = 0.179, 0.18
        raw_p_value, holm_adjusted_p_value = 0.2, 0.4
    gate_passed = reference or selected
    return {
        "model": model,
        "reference_model": "markov",
        "selected": selected,
        "gate_passed": gate_passed,
        "gate_reason": "passed" if gate_passed else "insufficient_log_loss_improvement",
        "log_loss": log_loss,
        "reference_log_loss": reference_log_loss,
        "absolute_log_loss_improvement": reference_log_loss - log_loss,
        "brier": brier,
        "reference_brier": reference_brier,
        "brier_difference": brier - reference_brier,
        "fallback_count": 0,
        "n_predictions": 365,
        "bootstrap_block_weeks": 13,
        "bootstrap_effective_block_weeks": 13,
        "bootstrap_resamples": 1_999,
        "bootstrap_seed": 17,
        "raw_p_value": raw_p_value,
        "holm_adjusted_p_value": holm_adjusted_p_value,
        "alpha": 0.05,
        "minimum_log_loss_improvement": 0.01,
        "brier_tolerance": 0.01,
    }


def _candidate(champion: str) -> dict[str, object]:
    model_names = ("markov", "xgboost", "causal_multiscale_ensemble")
    return {
        "meta": {
            "mode": "live",
            "freshness": {"status": "current"},
            "status": "ok",
        },
        "model": {
            "profile": "standard",
            "champion": champion,
            "model_health": {"status": "ok", "reasons": []},
            "latest_forecast_fallback": False,
            "leaderboard": [
                {
                    "name": name,
                    "selected": name == champion,
                    "is_champion": name == champion,
                }
                for name in model_names
            ],
            "selection_diagnostics": [
                _gate_row(name, champion=champion) for name in model_names
            ],
            "fx_ablation": {
                "promotion_allowed": False,
                "core_champion_promoted": False,
                "gate": {"passed_variants": []},
            },
        },
        "weekly": [{"health": {"status": "ok"}}],
        "sources": [{"status": "ok", "issues": []}],
    }


def _parity_split() -> dict[str, object]:
    return {
        "probability_parity": {
            "probability_numeric": {
                "exact_float_parity": True,
                "maximum_absolute_difference": 0,
                "mismatch_rows": 0,
                "mismatch_values": 0,
            },
            "probability_token_bytes": {
                "exact_parity": True,
                "mismatch_rows": 0,
                "mismatch_values": 0,
            },
        },
        "delta_left_minus_right": {
            "log_loss": 0,
            "brier": 0,
            "balanced_accuracy": 0,
            "fallback_rate": 0,
        },
    }


def _comparison(
    candidate: dict[str, object],
    *,
    candidate_sha256: str,
) -> dict[str, object]:
    diagnostics = {
        row["model"]: row
        for row in candidate["model"]["selection_diagnostics"]
    }
    multiscale = diagnostics["causal_multiscale_ensemble"]
    champion = candidate["model"]["champion"]
    required_models = {
        "causal_multiscale_ensemble",
        "markov",
        champion,
        diagnostics[champion]["reference_model"],
    }
    comparison_models = {
        name: {
            field: diagnostics[name][field]
            for field in (
                "reference_model",
                "selected",
                "gate_passed",
                "gate_reason",
                "fallback_count",
                "n_predictions",
                "log_loss",
                "brier",
            )
        }
        | {"matched_metric_crosscheck": True}
        for name in required_models
    }
    return {
        "schema_version": promote_v5_publication.COMPARISON_SCHEMA,
        "report_role": "derived_only_diagnostic_comparison",
        "promotion_interpretation": "prohibited",
        "provider_or_raw_feature_values_included": False,
        "inputs": {
            "v5": {"regime_results": {"sha256": candidate_sha256}},
        },
        "v5_markov_vs_frozen_v4_markov": {
            "primary_selection": _parity_split(),
            "post_selection_holdout": _parity_split(),
        },
        "v5_causal_multiscale_ensemble_vs_v5_markov": {
            "selection_gate_crosscheck": {
                "artifact_role": "selection_family_independently_recomputed",
                "multiscale_gate_against_selection_reference": multiscale[
                    "gate_passed"
                ],
                "models": comparison_models,
            },
        },
    }


@pytest.mark.parametrize("champion", ("xgboost", "causal_multiscale_ensemble"))
def test_review_evidence_accepts_gate_selected_dynamic_champion(
    champion: str,
) -> None:
    candidate_sha256 = "a" * 64
    candidate = _candidate(champion)
    comparison = _comparison(candidate, candidate_sha256=candidate_sha256)
    comparison["v5_causal_multiscale_ensemble_vs_v5_markov"][
        "selection_gate_crosscheck"
    ]["models"]["causal_multiscale_ensemble"]["log_loss"] += 5e-9
    promote_v5_publication._validate_review_evidence(
        candidate,
        comparison,
        candidate_sha256=candidate_sha256,
    )


def test_review_evidence_rejects_champion_flag_drift() -> None:
    candidate_sha256 = "a" * 64
    candidate = _candidate("xgboost")
    candidate["model"]["leaderboard"][0]["selected"] = True
    candidate["model"]["leaderboard"][0]["is_champion"] = True
    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="candidate champion selection evidence failed",
    ):
        promote_v5_publication._validate_review_evidence(
            candidate,
            _comparison(candidate, candidate_sha256=candidate_sha256),
            candidate_sha256=candidate_sha256,
        )


def test_review_evidence_rejects_multiscale_selection_reference_gate_drift() -> None:
    candidate_sha256 = "a" * 64
    candidate = _candidate("causal_multiscale_ensemble")
    comparison = _comparison(candidate, candidate_sha256=candidate_sha256)
    comparison["v5_causal_multiscale_ensemble_vs_v5_markov"][
        "selection_gate_crosscheck"
    ]["multiscale_gate_against_selection_reference"] = False
    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="selection-reference gate differs from candidate evidence",
    ):
        promote_v5_publication._validate_review_evidence(
            candidate,
            comparison,
            candidate_sha256=candidate_sha256,
        )


def test_review_evidence_rejects_legacy_threshold_for_new_candidate() -> None:
    candidate_sha256 = "a" * 64
    candidate = _candidate("xgboost")
    for row in candidate["model"]["selection_diagnostics"]:
        row["minimum_log_loss_improvement"] = 0.05

    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="minimum_log_loss_improvement=0.01",
    ):
        promote_v5_publication._validate_review_evidence(
            candidate,
            _comparison(candidate, candidate_sha256=candidate_sha256),
            candidate_sha256=candidate_sha256,
        )


def test_review_evidence_rejects_missing_generic_champion_binding() -> None:
    candidate_sha256 = "a" * 64
    candidate = _candidate("xgboost")
    comparison = _comparison(candidate, candidate_sha256=candidate_sha256)
    del comparison["v5_causal_multiscale_ensemble_vs_v5_markov"][
        "selection_gate_crosscheck"
    ]["models"]["xgboost"]

    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="do not bind the champion and its reference",
    ):
        promote_v5_publication._validate_review_evidence(
            candidate,
            comparison,
            candidate_sha256=candidate_sha256,
        )


def test_review_evidence_rejects_generic_champion_metric_drift() -> None:
    candidate_sha256 = "a" * 64
    candidate = _candidate("xgboost")
    comparison = _comparison(candidate, candidate_sha256=candidate_sha256)
    comparison["v5_causal_multiscale_ensemble_vs_v5_markov"][
        "selection_gate_crosscheck"
    ]["models"]["xgboost"]["log_loss"] += 1e-4

    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="log_loss differs from candidate evidence for xgboost",
    ):
        promote_v5_publication._validate_review_evidence(
            candidate,
            comparison,
            candidate_sha256=candidate_sha256,
        )


def test_standalone_promotion_requires_exact_artifact_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    expected = {"schema_version": "test", "nested": {"value": 1}}
    monkeypatch.setattr(
        promote_v5_publication,
        "_build_expected_comparison",
        lambda **_kwargs: expected,
    )

    promote_v5_publication._validate_reproducible_comparison(
        {"nested": {"value": 1}, "schema_version": "test"},
        v5_artifacts=artifacts,
        candidate_path=candidate_path,
    )
    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="differs from the independently reproduced artifact comparison",
    ):
        promote_v5_publication._validate_reproducible_comparison(
            {"schema_version": "test", "nested": {"value": 2}},
            v5_artifacts=artifacts,
            candidate_path=candidate_path,
        )


def test_promotion_cli_requires_v5_artifacts() -> None:
    args = promote_v5_publication.parse_args(
        [
            "--candidate",
            "candidate.json",
            "--v5-artifacts",
            "artifacts",
            "--comparison",
            "comparison.json",
            "--output",
            "reviewed.json",
        ]
    )

    assert args.v5_artifacts == Path("artifacts")
