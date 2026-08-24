from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_public_demo.py"
PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
LIVE_PUBLICATION = ROOT / "publication" / "live" / "regime-results.json"
SPEC = importlib.util.spec_from_file_location("package_public_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
package_public_demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_public_demo)
REAL_VERIFY_PROVIDER_RIGHTS = package_public_demo.verify_provider_rights


@pytest.fixture(autouse=True)
def _grant_provider_rights_for_packaging_contract_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise package contracts independently of the live rights registry."""

    monkeypatch.setattr(
        package_public_demo,
        "verify_provider_rights",
        lambda *_args, **_kwargs: None,
    )


def test_current_live_publication_passes_reviewed_provider_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        package_public_demo,
        "verify_provider_rights",
        REAL_VERIFY_PROVIDER_RIGHTS,
    )
    manifest = package_public_demo.package_public_dashboard(
        web_root=ROOT / "web",
        payload_path=LIVE_PUBLICATION,
        output_directory=tmp_path / "approved-live",
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
        comparison_path=ROOT / "publication/live/v5-vs-v4-comparison.json",
    )
    assert manifest["publication_scope"] == (
        "personal_noncommercial_derived_results"
    )
    assert manifest["contains_raw_observations"] is False


def _web_root(tmp_path: Path) -> Path:
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text("<main>demo</main>\n", encoding="utf-8")
    (root / "styles.css").write_text("main { color: black; }\n", encoding="utf-8")
    (root / "app.js").write_text("console.log('demo');\n", encoding="utf-8")
    return root


def _payload(tmp_path: Path, *, mode: str = "demo", synthetic: bool = True) -> Path:
    source_id = "synthetic_market" if synthetic else "alpha_vantage"
    license_class = "synthetic_fixture" if synthetic else "private_noncommercial"
    estimate = {
        "state": "transition",
        "probabilities": {"risk_on": 0.25, "transition": 0.5, "risk_off": 0.25},
        "confidence": 0.5,
        "entropy": 0.95,
    }
    payload = {
        "meta": {
            "schema_version": "1.0.0",
            "generated_at": "2026-08-11T00:00:00Z",
            "mode": mode,
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
                "id": source_id,
                "license_class": license_class,
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
    path = tmp_path / f"{mode}-payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _minimal_live_payload(tmp_path: Path, *, result_version: str) -> Path:
    source_licenses = package_public_demo.LIVE_SOURCE_LICENSES_BY_RESULT_VERSION[
        result_version
    ]
    payload = {
        "meta": {
            "mode": "live",
            "result_version": result_version,
            "data_as_of": "2026-08-21T20:00:00+00:00",
        },
        "model": {"champion": "markov"},
        "sources": [
            {"id": source_id, "license_class": license_class}
            for source_id, license_class in source_licenses.items()
        ],
        "weekly": [{"date": f"week-{index:02d}"} for index in range(52)],
    }
    if result_version == package_public_demo.V5_RESULT_VERSION:
        payload["model"].update(
            {
                "baseline_v4": dict(package_public_demo.FROZEN_V4_BASELINE),
                "core_artifacts": {
                    "oos_predictions": _artifact_record(
                        "oos-predictions.csv", "7" * 64, rows=5
                    ),
                    "selection_diagnostics": _artifact_record(
                        "selection-diagnostics.csv", "8" * 64, rows=2
                    ),
                },
                "research_artifacts": {
                    "fx_ablation_oos": _artifact_record(
                        "fx-ablation-oos.csv", "6" * 64, rows=20
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
            }
        )
        candidate_raw = _json_bytes(payload)
        payload["meta"]["publication_status"] = "reviewed_publication"
        payload["meta"]["publication_review"] = {
            "reviewed_candidate_sha256": hashlib.sha256(candidate_raw).hexdigest()
        }
    path = tmp_path / f"{result_version}.json"
    path.write_bytes(_json_bytes(payload))
    return path


def _artifact_record(path: str, sha256: str, *, rows: int | None = None) -> dict:
    record = {"path": path, "sha256": sha256}
    if rows is not None:
        record["row_count"] = rows
    return record


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
        row = {
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
        variants.append(row)
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


def _parity_split(*, count: int, common_sha: str, token_sha: str) -> dict:
    metrics = {
        "balanced_accuracy": 0.8,
        "brier": 0.2,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "log_loss": 0.4,
        "n": count,
    }
    return {
        "common_keys": {"count": count, "sha256": common_sha},
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
    *,
    count: int,
    common_sha: str,
    left: dict,
    right: dict,
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
        "common_keys": {"count": count, "sha256": common_sha},
        "delta_left_minus_right": {
            metric: left_metrics[metric] - right_metrics[metric]
            for metric in ("balanced_accuracy", "brier", "fallback_rate", "log_loss")
        },
        "metrics": {
            "causal_multiscale_ensemble": left_metrics,
            "v5_markov": right_metrics,
        },
    }


def _v5_comparison(payload_path: Path) -> dict:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    baseline = package_public_demo.FROZEN_V4_BASELINE
    model = payload["model"]
    fx = model["fx_ablation"]
    fx_control = fx["variant_metrics"][0]
    fx_variants = []
    for row in fx["variant_metrics"]:
        fx_variants.append(
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
                    "log_loss": row["log_loss"] - fx_control["log_loss"],
                    "brier": row["brier"] - fx_control["brier"],
                },
            }
        )
    selection_rows = {row["model"]: row for row in model["selection_diagnostics"]}
    holdout_rows = {row["name"]: row for row in model["leaderboard"]}
    selection_left = {
        **selection_rows["causal_multiscale_ensemble"],
        "balanced_accuracy": 0.76,
    }
    selection_right = {**selection_rows["markov"], "balanced_accuracy": 0.71}
    selection = _parity_split(count=3, common_sha="1" * 64, token_sha="2" * 64)
    holdout = _parity_split(count=2, common_sha="3" * 64, token_sha="4" * 64)
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
            "common_origins": {
                "count": fx["common_evaluation_origins"]["count"],
                "sha256": fx["common_evaluation_origins"]["sha256"],
            },
            "comparison_status": "evaluated",
            "interpretation": "diagnostic_only_not_a_promotion_decision",
            "payload_gate_metric_crosscheck": True,
            "source_status": "evaluated",
            "variants": fx_variants,
        },
        "inputs": {
            "frozen_v4": {
                "baseline_id": "v4-20260821",
                "oos_predictions": _artifact_record(
                    package_public_demo.FROZEN_V4_OOS_PREDICTIONS["path"],
                    package_public_demo.FROZEN_V4_OOS_PREDICTIONS["sha256"],
                    rows=package_public_demo.FROZEN_V4_OOS_PREDICTIONS["row_count"],
                ),
                "sha256sums": _artifact_record(
                    "SHA256SUMS", baseline["artifacts_inventory_sha256"]
                ),
                "verified_file_count": package_public_demo.FROZEN_V4_INVENTORY_FILE_COUNT,
            },
            "v5": {
                "fx_ablation_oos": dict(
                    model["research_artifacts"]["fx_ablation_oos"]
                ),
                "oos_predictions": dict(model["core_artifacts"]["oos_predictions"]),
                "regime_results": _artifact_record(
                    "regime-results.json", payload_hash
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
                common_sha="b" * 64,
                left=selection_left,
                right=selection_right,
            ),
            "post_selection_holdout": _multiscale_split(
                count=2,
                common_sha="d" * 64,
                left=holdout_rows["causal_multiscale_ensemble"],
                right=holdout_rows["markov"],
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
                    for name, row in selection_rows.items()
                },
                "pairwise_gate_against_markov": False,
            },
        },
        "v5_markov_vs_frozen_v4_markov": {
            "common_keys": {"count": 5, "sha256": "9" * 64},
            "join": {
                "common_key_count": 5,
                "left_key_count": 5,
                "left_model": "markov",
                "model_equivalence": "exact_name",
                "right_key_count": 5,
                "right_model": "markov",
            },
            "post_selection_holdout": holdout,
            "primary_selection": selection,
        },
    }


def _write_v5_comparison(tmp_path: Path, payload_path: Path, report: dict | None = None) -> Path:
    path = tmp_path / package_public_demo.V5_COMPARISON_FILENAME
    path.write_text(
        json.dumps(report if report is not None else _v5_comparison(payload_path), sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_package_copies_only_allowlisted_assets_and_synthetic_payload(tmp_path: Path) -> None:
    web_root = _web_root(tmp_path)
    (web_root / ".env").write_text("SECRET=must-not-copy\n", encoding="utf-8")
    (web_root / "provider.sqlite3").write_bytes(b"private database")
    nested = web_root / "data"
    nested.mkdir()
    (nested / "live.json").write_text('{"mode":"live"}', encoding="utf-8")

    payload = _payload(tmp_path)
    output = tmp_path / "public-demo"
    manifest = package_public_demo.package_public_demo(
        web_root=web_root,
        payload_path=payload,
        output_directory=output,
    )

    packaged = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert packaged == {
        "index.html",
        "styles.css",
        "app.js",
        "data/regime-results.json",
        "publication-manifest.json",
    }
    assert manifest["package_kind"] == "synthetic_demo_only"
    assert manifest["payload_mode"] == "demo"
    assert output.stat().st_mode & 0o777 == 0o755
    copied_payload = (output / "data/regime-results.json").read_bytes()
    assert manifest["files"]["data/regime-results.json"]["sha256"] == hashlib.sha256(
        copied_payload
    ).hexdigest()


def test_package_refuses_live_payload_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "public-demo"
    with pytest.raises(package_public_demo.PackagingError, match="meta.mode=demo"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path, mode="live", synthetic=False),
            output_directory=output,
        )
    assert not output.exists()


def test_live_derived_package_requires_explicit_rights_acknowledgement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "public-live"
    with pytest.raises(package_public_demo.PackagingError, match="rights acknowledgement"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=LIVE_PUBLICATION,
            output_directory=output,
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=False,
        )
    assert not output.exists()


def test_package_copies_only_allowlisted_assets_and_live_derived_snapshot(
    tmp_path: Path,
) -> None:
    web_root = _web_root(tmp_path)
    (web_root / "provider.sqlite3").write_bytes(b"private database")
    output = tmp_path / "public-live"
    manifest = package_public_demo.package_public_dashboard(
        web_root=web_root,
        payload_path=LIVE_PUBLICATION,
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )

    assert manifest["package_kind"] == "personal_noncommercial_live_derived"
    assert manifest["payload_mode"] == "live"
    assert manifest["publication_scope"] == "personal_noncommercial_derived_results"
    assert manifest["contains_raw_observations"] is False
    result_version = json.loads(LIVE_PUBLICATION.read_text(encoding="utf-8"))["meta"][
        "result_version"
    ]
    assert manifest["source_ids"] == sorted(
        package_public_demo.LIVE_SOURCE_LICENSES_BY_RESULT_VERSION[result_version]
    )
    assert not (output / "provider.sqlite3").exists()
    payload = json.loads((output / "data/regime-results.json").read_text())
    assert payload["meta"]["mode"] == "live"
    assert len(payload["weekly"]) >= 52


def test_v4_live_package_remains_compatible_without_comparison_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_public_demo, "validate_dashboard_payload", lambda _: None)
    payload = _minimal_live_payload(
        tmp_path,
        result_version=package_public_demo.V4_RESULT_VERSION,
    )
    output = tmp_path / "public-v4"
    manifest = package_public_demo.package_public_dashboard(
        web_root=_web_root(tmp_path),
        payload_path=payload,
        comparison_path=tmp_path / "does-not-exist.json",
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )

    assert package_public_demo.V5_COMPARISON_DESTINATION not in manifest["files"]
    assert not (output / package_public_demo.V5_COMPARISON_DESTINATION).exists()


def test_v5_live_package_requires_and_includes_reviewed_comparison_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_public_demo, "validate_dashboard_payload", lambda _: None)
    payload = _minimal_live_payload(
        tmp_path,
        result_version=package_public_demo.V5_RESULT_VERSION,
    )
    with pytest.raises(package_public_demo.PackagingError, match="sidecar.*missing"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload,
            output_directory=tmp_path / "missing-sidecar-output",
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )

    sidecar = _write_v5_comparison(tmp_path, payload)
    output = tmp_path / "public-v5"
    second_root = tmp_path / "second"
    second_root.mkdir()
    manifest = package_public_demo.package_public_dashboard(
        web_root=_web_root(second_root),
        payload_path=payload,
        comparison_path=sidecar,
        output_directory=output,
        publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
        rights_acknowledged=True,
    )

    copied = (output / package_public_demo.V5_COMPARISON_DESTINATION).read_bytes()
    assert manifest["source_ids"] == ["alfred", "alpha_vantage", "frb_h10"]
    assert manifest["files"][package_public_demo.V5_COMPARISON_DESTINATION] == {
        "bytes": len(copied),
        "sha256": hashlib.sha256(copied).hexdigest(),
    }


def test_v5_live_package_refuses_unreviewed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_public_demo, "validate_dashboard_payload", lambda _: None)
    payload = _minimal_live_payload(
        tmp_path,
        result_version=package_public_demo.V5_RESULT_VERSION,
    )
    document = json.loads(payload.read_text(encoding="utf-8"))
    document["meta"].pop("publication_status")
    payload.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        package_public_demo.PackagingError,
        match="publication_status=reviewed_publication",
    ):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload,
            output_directory=tmp_path / "unreviewed-v5",
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )


@pytest.mark.parametrize(
    ("case", "error"),
    (
        ("payload_hash", "payload hash mismatch"),
        ("baseline_inventory", "baseline inventory hash mismatch"),
        ("frozen_oos", "frozen OOS record is invalid"),
        ("v5_oos", "oos_predictions record does not match"),
        ("fx_metric", "does not match the reviewed payload"),
        ("multiscale_metric", "does not match the reviewed payload"),
        ("raw_data_alias", "fields are not exact"),
        ("numeric_parity", "numeric probability parity is not exact"),
        ("token_parity", "token probability hashes or mismatches differ"),
        ("raw_rows", "row-level or raw material"),
    ),
)
def test_v5_live_package_refuses_unreviewed_or_raw_comparison_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error: str,
) -> None:
    monkeypatch.setattr(package_public_demo, "validate_dashboard_payload", lambda _: None)
    payload = _minimal_live_payload(
        tmp_path,
        result_version=package_public_demo.V5_RESULT_VERSION,
    )
    report = _v5_comparison(payload)
    if case == "payload_hash":
        report["inputs"]["v5"]["regime_results"]["sha256"] = "0" * 64
    elif case == "baseline_inventory":
        report["inputs"]["frozen_v4"]["sha256sums"]["sha256"] = "0" * 64
    elif case == "frozen_oos":
        report["inputs"]["frozen_v4"]["oos_predictions"]["sha256"] = "0" * 64
    elif case == "v5_oos":
        report["inputs"]["v5"]["oos_predictions"]["row_count"] = 1
    elif case == "fx_metric":
        report["fx_ablation"]["variants"][1]["log_loss"] += 0.1
    elif case == "multiscale_metric":
        report["v5_causal_multiscale_ensemble_vs_v5_markov"][
            "primary_selection"
        ]["metrics"]["causal_multiscale_ensemble"]["log_loss"] += 0.1
    elif case == "raw_data_alias":
        report["fx_ablation"]["data"] = [["2026-08-21", 1.2345]]
    elif case == "numeric_parity":
        report["v5_markov_vs_frozen_v4_markov"]["primary_selection"][
            "probability_parity"
        ]["probability_numeric"]["exact_float_parity"] = False
    elif case == "token_parity":
        report["v5_markov_vs_frozen_v4_markov"]["primary_selection"][
            "probability_parity"
        ]["probability_token_bytes"]["right_sha256"] = "0" * 64
    else:
        report["fx_ablation"]["rows"] = [{"date": "2026-08-21"}]
    sidecar = _write_v5_comparison(tmp_path, payload, report)

    with pytest.raises(package_public_demo.PackagingError, match=error):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload,
            comparison_path=sidecar,
            output_directory=tmp_path / "public-v5",
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )


def test_live_derived_package_refuses_raw_observation_material(tmp_path: Path) -> None:
    payload = json.loads(LIVE_PUBLICATION.read_text(encoding="utf-8"))
    payload["sources"][0]["observations"] = [{"date": "2026-08-07", "value": 1.0}]
    payload_path = tmp_path / "unsafe-live.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "public-live"

    with pytest.raises(package_public_demo.PackagingError, match="forbidden provider material"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload_path,
            output_directory=output,
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )
    assert not output.exists()


def test_v5_live_package_refuses_forged_reviewed_candidate_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_public_demo, "validate_dashboard_payload", lambda _: None)
    payload = _minimal_live_payload(
        tmp_path,
        result_version=package_public_demo.V5_RESULT_VERSION,
    )
    document = json.loads(payload.read_text(encoding="utf-8"))
    document["meta"]["publication_review"]["reviewed_candidate_sha256"] = "0" * 64
    payload.write_bytes(_json_bytes(document))

    with pytest.raises(package_public_demo.PackagingError, match="reconstructed bytes"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload,
            comparison_path=_write_v5_comparison(tmp_path, payload),
            output_directory=tmp_path / "forged-review",
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )


def test_live_derived_package_refuses_machine_local_paths(tmp_path: Path) -> None:
    payload = json.loads(LIVE_PUBLICATION.read_text(encoding="utf-8"))
    payload["meta"]["debug_path"] = "/Users/example/private-output.json"
    payload_path = tmp_path / "unsafe-path-live.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(package_public_demo.PackagingError, match="machine-local path"):
        package_public_demo.package_public_dashboard(
            web_root=_web_root(tmp_path),
            payload_path=payload_path,
            output_directory=tmp_path / "public-live",
            publication_mode=package_public_demo.PUBLICATION_MODE_LIVE_DERIVED,
            rights_acknowledged=True,
        )


def test_package_refuses_provider_data_mislabeled_as_demo(tmp_path: Path) -> None:
    output = tmp_path / "public-demo"
    with pytest.raises(package_public_demo.PackagingError, match="synthetic fixture"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path, mode="demo", synthetic=False),
            output_directory=output,
        )
    assert not output.exists()


def test_package_refuses_malformed_synthetic_payload(tmp_path: Path) -> None:
    payload_path = _payload(tmp_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["weekly"][0]["next_week"]["probabilities"] = {
        "risk_on": 0.8,
        "transition": 0.8,
        "risk_off": 0.1,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "public-demo"

    with pytest.raises(
        package_public_demo.PackagingError,
        match="dashboard payload contract is invalid",
    ):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=payload_path,
            output_directory=output,
        )

    assert not output.exists()


def test_package_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "public-demo"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(package_public_demo.PackagingError, match="refusing to overwrite"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path),
            output_directory=output,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_package_rejects_output_through_symlink_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic-link"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path),
            output_directory=linked_parent / "public-demo",
        )
    assert not (real_parent / "public-demo").exists()


def test_package_rejects_absolute_other_project_output_before_write(
    tmp_path: Path,
) -> None:
    other_project = ROOT.parent / "do-not-touch-regime-package" / "public-demo"
    with pytest.raises(ValueError, match="must stay below"):
        package_public_demo.package_public_demo(
            web_root=_web_root(tmp_path),
            payload_path=_payload(tmp_path),
            output_directory=other_project,
        )


def test_private_live_outputs_are_ignored_by_repository_contract() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/*.sqlite3*" in ignored
    assert "web/data/regime-results.json" in ignored
    assert "artifacts/" in ignored
    assert "artifacts/latest/" in ignored


def test_pages_workflow_uploads_only_verified_live_derived_package() -> None:
    workflow = PAGES_WORKFLOW.read_text(encoding="utf-8")
    assert "regime-lab demo" not in workflow
    assert "publication/live/regime-results.json" in workflow
    assert "--comparison publication/live/v5-vs-v4-comparison.json" in workflow
    assert "--publication-mode live-derived" in workflow
    assert "--acknowledge-personal-noncommercial-publication" in workflow
    assert "--output dist/public-dashboard" in workflow
    assert "verify_public_package.py dist/public-dashboard" in workflow
    assert (
        "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5.0.0"
        in workflow
    )
    assert "path: dist/public-dashboard" in workflow
    assert "web/data/regime-results.json" not in workflow
    assert "data/regime.sqlite3" not in workflow
    assert "artifacts/latest" not in workflow
    assert "secrets." not in workflow
