from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = ROOT / "scripts" / "package_public_demo.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_public_package.py"
LIVE_PUBLICATION = ROOT / "publication" / "live" / "regime-results.json"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_public_demo = _load_script("package_public_demo", PACKAGE_SCRIPT)
verify_public_package = _load_script("verify_public_package", VERIFY_SCRIPT)


def _web_root(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<main>demo</main>\n", encoding="utf-8")
    (web / "styles.css").write_text("main { color: black; }\n", encoding="utf-8")
    (web / "app.js").write_text("console.log('demo');\n", encoding="utf-8")
    return web


def _payload(tmp_path: Path) -> Path:
    estimate = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    payload = {
        "meta": {
            "schema_version": "1.0.0",
            "generated_at": "2026-08-14T00:00:00Z",
            "mode": "demo",
            "data_as_of": "2026-08-07T20:00:00+00:00",
            "timezone": "America/New_York",
        },
        "states": [
            {"id": "risk_on", "label": "Risk-on"},
            {"id": "transition", "label": "Transition"},
            {"id": "risk_off", "label": "Risk-off"},
        ],
        "model": {
            "champion": "markov",
            "selection_status": "provisional_predeployment",
            "leaderboard": [],
        },
        "sources": [
            {
                "id": "synthetic_market",
                "license_class": "synthetic_fixture",
                "status": "degraded",
            }
        ],
        "weekly": [
            {
                "date": "2026-08-07",
                "current": estimate,
                "next_week": estimate,
                "transition_probability": 0.2,
                "scores": {
                    "trend": 0.1,
                    "stress": -0.1,
                    "macro": 0.0,
                    "financial_conditions": 0.0,
                },
            }
        ],
        "feature_catalog": [
            {
                "id": "synthetic_feature",
                "category": "test",
                "frequency": "weekly",
                "source": "synthetic_market",
            }
        ],
    }
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _package(tmp_path: Path) -> Path:
    output = tmp_path / "public-demo"
    package_public_demo.package_public_demo(
        web_root=_web_root(tmp_path),
        payload_path=_payload(tmp_path),
        output_directory=output,
    )
    return output


def _minimal_v5_payload(tmp_path: Path) -> Path:
    payload = {
        "meta": {
            "mode": "live",
            "result_version": package_public_demo.V5_RESULT_VERSION,
            "data_as_of": "2026-08-21T20:00:00+00:00",
        },
        "model": {
            "champion": "markov",
            "baseline_v4": dict(package_public_demo.FROZEN_V4_BASELINE),
            "core_artifacts": {
                "oos_predictions": _record("oos-predictions.csv", "3" * 64, rows=5),
                "selection_diagnostics": _record(
                    "selection-diagnostics.csv", "4" * 64, rows=2
                ),
            },
            "research_artifacts": {
                "fx_ablation_oos": _record(
                    "fx-ablation-oos.csv", "2" * 64, rows=20
                )
            },
            "selection_diagnostics": [
                {
                    "model": "causal_multiscale_ensemble",
                    "reference_model": "markov",
                    "selected": False,
                    "gate_passed": False,
                    "gate_reason": "insufficient_log_loss_improvement",
                    "log_loss": 0.35,
                    "brier": 0.18,
                    "fallback_count": 0,
                    "n_predictions": 3,
                },
                {
                    "model": "markov",
                    "reference_model": "markov",
                    "selected": True,
                    "gate_passed": True,
                    "gate_reason": "passed",
                    "log_loss": 0.4,
                    "brier": 0.2,
                    "fallback_count": 0,
                    "n_predictions": 3,
                },
            ],
            "leaderboard": [
                {
                    "name": "causal_multiscale_ensemble",
                    "selected": False,
                    "is_champion": False,
                    "log_loss": 0.4,
                    "brier": 0.22,
                    "balanced_accuracy": 0.75,
                    "fallback_count": 0,
                    "n_predictions": 2,
                },
                {
                    "name": "markov",
                    "selected": True,
                    "is_champion": True,
                    "log_loss": 0.45,
                    "brier": 0.24,
                    "balanced_accuracy": 0.70,
                    "fallback_count": 0,
                    "n_predictions": 2,
                },
            ],
            "fx_ablation": _minimal_fx_ablation(),
        },
        "sources": [
            {"id": source_id, "license_class": license_class}
            for source_id, license_class in (
                package_public_demo.LIVE_SOURCE_LICENSES_BY_RESULT_VERSION[
                    package_public_demo.V5_RESULT_VERSION
                ].items()
            )
        ],
        "weekly": [{"date": f"week-{index:02d}"} for index in range(52)],
    }
    candidate_raw = _json_bytes(payload)
    payload["meta"]["publication_status"] = "reviewed_publication"
    payload["meta"]["publication_review"] = {
        "reviewed_candidate_sha256": hashlib.sha256(candidate_raw).hexdigest()
    }
    path = tmp_path / "v5.json"
    path.write_bytes(_json_bytes(payload))
    return path


def _record(path: str, sha256: str, *, rows: int | None = None) -> dict:
    value = {"path": path, "sha256": sha256}
    if rows is not None:
        value["row_count"] = rows
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=False)
        + "\n"
    ).encode("utf-8")


def _minimal_fx_ablation() -> dict:
    variants = []
    comparisons = []
    for index, name in enumerate(
        (
            "v4_control",
            "v4_plus_broad_index",
            "v4_plus_bilateral_panel",
            "v4_plus_all_fx",
        )
    ):
        log_loss = 0.5 + index * 0.01
        brier = 0.3 + index * 0.01
        variants.append(
            {
                "variant": name,
                "feature_count": 10 + index,
                "fx_feature_count": index,
                "feature_columns_sha256": f"{index + 1}" * 64,
                "log_loss": log_loss,
                "brier": brier,
                "accuracy": 0.7 - index * 0.01,
                "balanced_accuracy": 0.65 - index * 0.01,
                "n_predictions": 5,
                "fallback_count": 0,
            }
        )
        if index:
            comparisons.append(
                {
                    "variant": name,
                    "mean_log_loss_improvement": 0.5 - log_loss,
                    "brier_difference": brier - 0.3,
                }
            )
    return {
        "status": "evaluated",
        "common_evaluation_origins": {"count": 5, "sha256": "c" * 64},
        "variant_metrics": variants,
        "gate": {"comparisons": comparisons, "passed_variants": []},
        "promotion_allowed": False,
        "core_champion_promoted": False,
    }


def _parity_split(*, count: int, key_sha: str, token_sha: str) -> dict:
    metrics = {
        "balanced_accuracy": 0.8,
        "brier": 0.2,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "log_loss": 0.4,
        "n": count,
    }
    return {
        "common_keys": {"count": count, "sha256": key_sha},
        "delta_left_minus_right": {
            "balanced_accuracy": 0.0,
            "brier": 0.0,
            "fallback_rate": 0.0,
            "log_loss": 0.0,
        },
        "metrics": {
            "frozen_v4_markov": dict(metrics),
            "v5_markov": dict(metrics),
        },
        "probability_parity": {
            "probability_numeric": {
                "exact_float_parity": True,
                "maximum_absolute_difference": 0.0,
                "mismatch_rows": 0,
                "mismatch_values": 0,
            },
            "probability_token_bytes": {
                "exact_parity": True,
                "left_sha256": token_sha,
                "mismatch_rows": 0,
                "mismatch_values": 0,
                "right_sha256": token_sha,
            },
        },
    }


def _multiscale_split(
    *, count: int, key_sha: str, left: dict, right: dict
) -> dict:
    def metrics(row: dict) -> dict:
        return {
            "balanced_accuracy": row["balanced_accuracy"],
            "brier": row["brier"],
            "fallback_count": row["fallback_count"],
            "fallback_rate": row["fallback_count"] / count,
            "log_loss": row["log_loss"],
            "n": count,
        }

    left_metrics = metrics(left)
    right_metrics = metrics(right)
    return {
        "common_keys": {"count": count, "sha256": key_sha},
        "delta_left_minus_right": {
            metric: left_metrics[metric] - right_metrics[metric]
            for metric in ("balanced_accuracy", "brier", "fallback_rate", "log_loss")
        },
        "metrics": {
            "causal_multiscale_ensemble": left_metrics,
            "v5_markov": right_metrics,
        },
    }


def _v5_report(payload: Path) -> dict:
    baseline = package_public_demo.FROZEN_V4_BASELINE
    document = json.loads(payload.read_text(encoding="utf-8"))
    model = document["model"]
    fx = model["fx_ablation"]
    control = fx["variant_metrics"][0]
    fx_variants = [
        {
            **{
                field: row[field]
                for field in (
                    "variant",
                    "feature_count",
                    "fx_feature_count",
                    "feature_columns_sha256",
                    "n_predictions",
                    "log_loss",
                    "brier",
                    "accuracy",
                    "balanced_accuracy",
                    "fallback_count",
                )
            },
            "delta_vs_v4_control": {
                "log_loss": row["log_loss"] - control["log_loss"],
                "brier": row["brier"] - control["brier"],
            },
        }
        for row in fx["variant_metrics"]
    ]
    selection = {row["model"]: row for row in model["selection_diagnostics"]}
    holdout = {row["name"]: row for row in model["leaderboard"]}
    selection_left = {
        **selection["causal_multiscale_ensemble"],
        "balanced_accuracy": 0.76,
    }
    selection_right = {**selection["markov"], "balanced_accuracy": 0.71}
    return {
        "comparison_contract": {
            "actual_must_match": True,
            "evaluation_split_must_match": True,
            "exact_key_fields": [
                "origin_date",
                "target_date",
                "model_or_equivalent_model",
            ],
            "metric_definitions": {
                "balanced_accuracy": "mean_recall_over_actual_classes_present",
                "brier": "mean_three_state_sum_squared_error",
                "log_loss": "mean_negative_log_actual_probability_clip_1e-9_then_renormalize",
            },
            "probability_columns": ["p_risk_on", "p_transition", "p_risk_off"],
            "splits_are_never_pooled": ["selection", "holdout"],
            "unmatched_keys_excluded": True,
        },
        "fx_ablation": {
            "aggregate_crosscheck": True,
            "common_origins": dict(fx["common_evaluation_origins"]),
            "comparison_status": "evaluated",
            "interpretation": "diagnostic_only_not_a_promotion_decision",
            "payload_gate_metric_crosscheck": True,
            "source_status": "evaluated",
            "variants": fx_variants,
        },
        "inputs": {
            "frozen_v4": {
                "baseline_id": "v4-20260821",
                "oos_predictions": dict(
                    package_public_demo.FROZEN_V4_OOS_PREDICTIONS
                ),
                "sha256sums": _record(
                    "SHA256SUMS", baseline["artifacts_inventory_sha256"]
                ),
                "verified_file_count": package_public_demo.FROZEN_V4_INVENTORY_FILE_COUNT,
            },
            "v5": {
                "fx_ablation_oos": dict(
                    model["research_artifacts"]["fx_ablation_oos"]
                ),
                "oos_predictions": dict(model["core_artifacts"]["oos_predictions"]),
                "regime_results": _record(
                    "regime-results.json", hashlib.sha256(payload.read_bytes()).hexdigest()
                ),
                "selection_diagnostics": dict(
                    model["core_artifacts"]["selection_diagnostics"]
                ),
            },
        },
        "promotion_interpretation": "prohibited",
        "provider_or_raw_feature_values_included": False,
        "report_role": "derived_only_diagnostic_comparison",
        "schema_version": package_public_demo.V5_COMPARISON_SCHEMA_VERSION,
        "v5_causal_multiscale_ensemble_vs_v5_markov": {
            "common_keys": {"count": 5, "sha256": "a" * 64},
            "join": {
                "common_key_count": 5,
                "left_key_count": 5,
                "left_model": "causal_multiscale_ensemble",
                "model_equivalence": "fixed_pair_same_origin_and_target",
                "right_key_count": 5,
                "right_model": "markov",
            },
            "primary_selection": _multiscale_split(
                count=3,
                key_sha="b" * 64,
                left=selection_left,
                right=selection_right,
            ),
            "post_selection_holdout": _multiscale_split(
                count=2,
                key_sha="d" * 64,
                left=holdout["causal_multiscale_ensemble"],
                right=holdout["markov"],
            ),
            "selection_gate_crosscheck": {
                "artifact_role": "selection_only_existing_champion_gate",
                "models": {
                    name: {
                        field: row[field]
                        for field in (
                            "brier",
                            "fallback_count",
                            "gate_passed",
                            "gate_reason",
                            "log_loss",
                            "n_predictions",
                            "reference_model",
                            "selected",
                        )
                    }
                    | {"matched_metric_crosscheck": True}
                    for name, row in selection.items()
                },
                "pairwise_gate_against_markov": False,
            },
        },
        "v5_markov_vs_frozen_v4_markov": {
            "common_keys": {"count": 5, "sha256": "5" * 64},
            "join": {
                "common_key_count": 5,
                "left_key_count": 5,
                "left_model": "markov",
                "model_equivalence": "exact_name",
                "right_key_count": 5,
                "right_model": "markov",
            },
            "post_selection_holdout": _parity_split(
                count=2, key_sha="6" * 64, token_sha="7" * 64
            ),
            "primary_selection": _parity_split(
                count=3, key_sha="8" * 64, token_sha="9" * 64
            ),
        },
    }


def _package_v5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(package_public_demo, "validate_dashboard_payload", lambda _: None)
    monkeypatch.setattr(
        verify_public_package,
        "validate_public_payload",
        lambda *args, **kwargs: None,
    )
    payload = _minimal_v5_payload(tmp_path)
    comparison = tmp_path / package_public_demo.V5_COMPARISON_FILENAME
    comparison.write_text(json.dumps(_v5_report(payload), sort_keys=True), encoding="utf-8")
    output = tmp_path / "public-v5"
    package_public_demo.package_public_dashboard(
        web_root=_web_root(tmp_path),
        payload_path=payload,
        comparison_path=comparison,
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )
    return output


def _refresh_manifest_record(output: Path, relative_path: str) -> None:
    raw = (output / relative_path).read_bytes()
    manifest_path = output / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative_path] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_verifier_accepts_exact_synthetic_package(tmp_path: Path) -> None:
    result = verify_public_package.verify_public_package(_package(tmp_path))
    assert result["ok"] is True
    assert result["package_kind"] == "synthetic_demo_only"
    assert result["payload_mode"] == "demo"


def test_verifier_accepts_exact_live_derived_package(tmp_path: Path) -> None:
    output = tmp_path / "public-live"
    package_public_demo.package_public_dashboard(
        web_root=_web_root(tmp_path),
        payload_path=LIVE_PUBLICATION,
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )

    result = verify_public_package.verify_public_package(output)
    assert result["ok"] is True
    assert result["package_kind"] == "personal_noncommercial_live_derived"
    assert result["payload_mode"] == "live"


def test_verifier_accepts_exact_v5_live_package_with_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = verify_public_package.verify_public_package(
        _package_v5(tmp_path, monkeypatch)
    )
    assert result["ok"] is True
    assert result["comparison_included"] is True
    assert package_public_demo.V5_COMPARISON_DESTINATION in result["files"]


def test_verifier_refuses_v5_parity_tamper_even_with_matching_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _package_v5(tmp_path, monkeypatch)
    comparison_path = output / package_public_demo.V5_COMPARISON_DESTINATION
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    report["v5_markov_vs_frozen_v4_markov"]["primary_selection"][
        "delta_left_minus_right"
    ]["log_loss"] = 0.0001
    comparison_path.write_text(json.dumps(report), encoding="utf-8")
    _refresh_manifest_record(output, package_public_demo.V5_COMPARISON_DESTINATION)

    with pytest.raises(verify_public_package.VerificationError, match="exactly zero"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_v5_raw_rows_even_with_matching_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _package_v5(tmp_path, monkeypatch)
    comparison_path = output / package_public_demo.V5_COMPARISON_DESTINATION
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    report["fx_ablation"]["rows"] = [{"origin_date": "2026-08-21"}]
    comparison_path.write_text(json.dumps(report), encoding="utf-8")
    _refresh_manifest_record(output, package_public_demo.V5_COMPARISON_DESTINATION)

    with pytest.raises(
        verify_public_package.VerificationError,
        match="row-level or raw material",
    ):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_extra_file(tmp_path: Path) -> None:
    output = _package(tmp_path)
    (output / "live.sqlite3").write_bytes(b"provider data")
    with pytest.raises(verify_public_package.VerificationError, match="extra"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_tampered_allowlisted_asset(tmp_path: Path) -> None:
    output = _package(tmp_path)
    (output / "app.js").write_text("console.log('changed');\n", encoding="utf-8")
    with pytest.raises(verify_public_package.VerificationError, match="mismatch"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_manifest_data_as_of_drift(tmp_path: Path) -> None:
    output = _package(tmp_path)
    manifest_path = output / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_data_as_of"] = "1999-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(verify_public_package.VerificationError, match="data_as_of mismatch"):
        verify_public_package.verify_public_package(output)


def test_verifier_refuses_credential_like_material_even_with_matching_manifest(
    tmp_path: Path,
) -> None:
    output = _package(tmp_path)
    app = output / "app.js"
    raw = b"const FRED_API_KEY = 'must-not-publish';\n"
    app.write_bytes(raw)
    manifest_path = output / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["app.js"] = {
        "bytes": len(raw),
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        verify_public_package.VerificationError,
        match="credential-like",
    ):
        verify_public_package.verify_public_package(output)
