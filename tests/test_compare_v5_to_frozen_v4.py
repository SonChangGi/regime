from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from regime_lab.analysis.validation import (
    evaluate_predictions,
    select_champion_with_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_v5_to_frozen_v4.py"
SPEC = importlib.util.spec_from_file_location("compare_v5_to_frozen_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


FIELDS = (
    "origin_date",
    "target_date",
    "model",
    "evaluation_split",
    "actual",
    "p_risk_on",
    "p_transition",
    "p_risk_off",
    "fallback",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cases = (
        ("2024-01-05 21:00:00+00:00", "2024-01-12 21:00:00+00:00", "selection", "risk_on", ("0.7", "0.2", "0.1"), ("0.5", "0.3", "0.2")),
        ("2024-01-12 21:00:00+00:00", "2024-01-19 21:00:00+00:00", "selection", "risk_off", ("0.2", "0.2", "0.6"), ("0.3", "0.3", "0.4")),
        ("2024-01-19 21:00:00+00:00", "2024-01-26 21:00:00+00:00", "holdout", "transition", ("0.2", "0.6", "0.2"), ("0.2", "0.7", "0.1")),
        ("2024-01-26 21:00:00+00:00", "2024-02-02 21:00:00+00:00", "holdout", "risk_on", ("0.6", "0.3", "0.1"), ("0.7", "0.2", "0.1")),
    )
    v4: list[dict[str, object]] = []
    v5: list[dict[str, object]] = []
    for origin, target, split, actual, markov, multiscale in cases:
        base = {
            "origin_date": origin,
            "target_date": target,
            "evaluation_split": split,
            "actual": actual,
            "fallback": "False",
        }
        markov_row = {
            **base,
            "model": "markov",
            **dict(zip(comparison.PROBABILITY_COLUMNS, markov, strict=True)),
        }
        v4.append(dict(markov_row))
        v5.append(dict(markov_row))
        v5.append(
            {
                **base,
                "model": "causal_multiscale_ensemble",
                **dict(zip(comparison.PROBABILITY_COLUMNS, multiscale, strict=True)),
            }
        )
    return v4, v5


def _refresh_frozen_inventory(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = sorted(path for path in directory.iterdir() if path.name != "SHA256SUMS")
    inventory = "".join(f"{_sha256(path)}  {path.name}\n" for path in files).encode("ascii")
    (directory / "SHA256SUMS").write_bytes(inventory)
    monkeypatch.setattr(
        comparison,
        "FROZEN_V4_INVENTORY_SHA256",
        hashlib.sha256(inventory).hexdigest(),
    )
    monkeypatch.setattr(comparison, "FROZEN_V4_INVENTORY_FILE_COUNT", len(files))


def _selection_rows(v5_oos: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(v5_oos)
    frame["fallback"] = frame["fallback"].map(
        lambda value: value if isinstance(value, bool) else value == "True"
    )
    selection = frame.loc[frame["evaluation_split"].eq("selection")].copy()
    leaderboard = evaluate_predictions(selection)
    _, diagnostics = select_champion_with_diagnostics(
        leaderboard,
        selection,
        minimum_log_loss_improvement=0.01,
    )
    return [
        {
            field: "" if pd.isna(value) else value
            for field, value in row.items()
        }
        for row in diagnostics.to_dict(orient="records")
    ]


def _refresh_v5_payload(directory: Path, *, fx_ablation: dict[str, object] | None = None) -> None:
    oos = directory / "oos-predictions.csv"
    diagnostics = directory / "selection-diagnostics.csv"
    research_artifacts: dict[str, dict[str, object]] = {}
    fx_sidecar = directory / comparison.FX_ABLATION_OOS_PATH
    if fx_sidecar.is_file():
        research_artifacts["fx_ablation_oos"] = {
            "path": fx_sidecar.name,
            "row_count": sum(1 for _ in fx_sidecar.open(encoding="utf-8")) - 1,
            "sha256": _sha256(fx_sidecar),
        }
    parsed_diagnostics, _ = comparison._read_selection_diagnostics(diagnostics)
    selection_rows = list(parsed_diagnostics.values())
    selected = [row["model"] for row in selection_rows if row["selected"] is True]
    assert len(selected) == 1
    payload = {
        "meta": {"result_version": "weekly-regime-result-v5", "mode": "live"},
        "model": {
            "champion": selected[0],
            "selection_diagnostics": selection_rows,
            "core_artifacts": {
                "oos_predictions": {
                    "path": oos.name,
                    "row_count": sum(1 for _ in oos.open(encoding="utf-8")) - 1,
                    "sha256": _sha256(oos),
                },
                "selection_diagnostics": {
                    "path": diagnostics.name,
                    "row_count": sum(1 for _ in diagnostics.open(encoding="utf-8")) - 1,
                    "sha256": _sha256(diagnostics),
                },
            },
            "fx_ablation": fx_ablation
            or {
                "status": "unavailable",
                "status_reason": "fx_not_collected",
                "variant_metrics": [],
            },
            "research_artifacts": research_artifacts,
        },
    }
    (directory / "regime-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _evaluated_fx_fixture(directory: Path) -> dict[str, object]:
    origins = (
        ("2024-03-01", "2024-03-08", "risk_on"),
        ("2024-03-08", "2024-03-15", "risk_off"),
    )
    origin_pairs = [[origin, target] for origin, target, _ in origins]
    origin_sha256 = comparison._canonical_sha256(origin_pairs)
    probability_by_variant = {
        "v4_control": (("0.7", "0.2", "0.1"), ("0.2", "0.2", "0.6")),
        "v4_plus_broad_index": (("0.8", "0.1", "0.1"), ("0.1", "0.2", "0.7")),
        "v4_plus_bilateral_panel": (("0.6", "0.3", "0.1"), ("0.2", "0.1", "0.7")),
        "v4_plus_all_fx": (("0.75", "0.15", "0.1"), ("0.1", "0.15", "0.75")),
    }
    rows: list[dict[str, object]] = []
    for position, (origin, target, actual) in enumerate(origins):
        for variant in comparison.FX_VARIANTS:
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": target,
                    "variant": variant,
                    "evaluation_split": "prospective_shadow",
                    "current_state": actual,
                    "actual": actual,
                    **dict(
                        zip(
                            comparison.PROBABILITY_COLUMNS,
                            probability_by_variant[variant][position],
                            strict=True,
                        )
                    ),
                    "train_size": 104,
                    "gap": 1,
                    "last_train_target": "2024-02-23",
                    "purged_origin_count": 1,
                    "fallback": "False",
                    "fallback_reason": "",
                    "common_origins_sha256": origin_sha256,
                }
            )
    sidecar = directory / comparison.FX_ABLATION_OOS_PATH
    _write_csv(sidecar, rows, comparison.FX_ABLATION_OOS_COLUMNS)
    loaded, _, _ = comparison._read_fx_ablation_oos(sidecar)
    manifest: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for index, variant in enumerate(comparison.FX_VARIANTS):
        fx_feature_count = index
        manifest.append(
            {
                "variant": variant,
                "feature_count": fx_feature_count,
                "feature_columns_sha256": hashlib.sha256(variant.encode()).hexdigest(),
            }
        )
        recomputed = comparison._fx_metrics(
            [row for row in loaded if row.model == variant]
        )
        feature_hash = hashlib.sha256(f"core+{variant}".encode()).hexdigest()
        metrics.append(
            {
                "variant": variant,
                "feature_count": 10 + fx_feature_count,
                "fx_feature_count": fx_feature_count,
                "feature_columns_sha256": feature_hash,
                **recomputed,
                "n": recomputed["n_predictions"],
                "fallback": False,
                "fallback_reasons": {},
                "first_origin": origins[0][0],
                "last_origin": origins[-1][0],
                "origin_sha256": origin_sha256,
            }
        )
    metric_index = {row["variant"]: row for row in metrics}
    gate_comparisons = [
        {
            "variant": variant,
            "reference_variant": "v4_control",
            "mean_log_loss_improvement": (
                metric_index["v4_control"]["log_loss"]
                - metric_index[variant]["log_loss"]
            ),
            "brier_difference": (
                metric_index[variant]["brier"]
                - metric_index["v4_control"]["brier"]
            ),
            "control_fallback_count": 0,
            "fallback_count": 0,
        }
        for variant in comparison.FX_VARIANTS[1:]
    ]
    return {
        "status": "evaluated",
        "status_reason": None,
        "manifest": manifest,
        "variant_metrics": metrics,
        "common_evaluation_origins": {
            "count": len(origins),
            "first_origin": origins[0][0],
            "last_origin": origins[-1][0],
            "sha256": origin_sha256,
            "rows": [],
        },
        "gate": {"comparisons": gate_comparisons},
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    v4 = tmp_path / "v4"
    v5 = tmp_path / "v5"
    v4.mkdir()
    v5.mkdir()
    v4_rows, v5_rows = _prediction_rows()
    _write_csv(v4 / "oos-predictions.csv", v4_rows, FIELDS)
    _write_csv(v5 / "oos-predictions.csv", v5_rows, FIELDS)
    selection_rows = _selection_rows(v5 / "oos-predictions.csv")
    _write_csv(
        v5 / "selection-diagnostics.csv",
        selection_rows,
        tuple(selection_rows[0]),
    )
    _refresh_v5_payload(v5)
    _refresh_frozen_inventory(v4, monkeypatch)
    return v5, v4


def test_report_is_deterministic_split_matched_and_derived_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)

    first = comparison.build_comparison(v5, v4)
    second = comparison.build_comparison(v5, v4)

    assert comparison.canonical_json_bytes(first) == comparison.canonical_json_bytes(second)
    assert first["promotion_interpretation"] == "prohibited"
    assert first["provider_or_raw_feature_values_included"] is False
    markov = first["v5_markov_vs_frozen_v4_markov"]
    assert markov["primary_selection"]["common_keys"]["count"] == 2
    assert markov["post_selection_holdout"]["common_keys"]["count"] == 2
    assert markov["primary_selection"]["probability_parity"]["probability_token_bytes"]["exact_parity"] is True
    assert markov["primary_selection"]["probability_parity"]["probability_numeric"]["exact_float_parity"] is True
    assert markov["primary_selection"]["delta_left_minus_right"]["log_loss"] == 0.0
    multiscale = first["v5_causal_multiscale_ensemble_vs_v5_markov"]
    assert multiscale["selection_gate_crosscheck"]["models"]["causal_multiscale_ensemble"]["matched_metric_crosscheck"] is True
    gate = multiscale["selection_gate_crosscheck"]
    assert gate["artifact_role"] == "selection_family_independently_recomputed"
    assert gate["multiscale_gate_against_selection_reference"] is False
    assert first["fx_ablation"]["comparison_status"] == "unavailable"
    assert len(first["inputs"]["v5"]["oos_predictions"]["sha256"]) == 64


def test_selection_gate_crosscheck_reflects_multiscale_gate_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    oos_path = v5 / "oos-predictions.csv"
    oos_rows = list(csv.DictReader(oos_path.open(encoding="utf-8")))
    for row in oos_rows:
        if row["model"] != "causal_multiscale_ensemble" or row["evaluation_split"] != "selection":
            continue
        probabilities = {
            "risk_on": ("0.999", "0.0005", "0.0005"),
            "transition": ("0.0005", "0.999", "0.0005"),
            "risk_off": ("0.0005", "0.0005", "0.999"),
        }[row["actual"]]
        row.update(dict(zip(comparison.PROBABILITY_COLUMNS, probabilities, strict=True)))
    _write_csv(oos_path, oos_rows, FIELDS)
    path = v5 / "selection-diagnostics.csv"
    rows = _selection_rows(oos_path)
    _write_csv(path, rows, tuple(rows[0]))
    _refresh_v5_payload(v5)

    report = comparison.build_comparison(v5, v4)

    gate = report["v5_causal_multiscale_ensemble_vs_v5_markov"][
        "selection_gate_crosscheck"
    ]
    assert gate["multiscale_gate_against_selection_reference"] is True
    assert gate["models"]["causal_multiscale_ensemble"]["selected"] is True


def test_selection_gate_crosscheck_binds_non_markov_champion_and_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    oos_path = v5 / "oos-predictions.csv"
    rows = list(csv.DictReader(oos_path.open(encoding="utf-8")))
    markov_rows = [row for row in rows if row["model"] == "markov"]
    rows.extend(
        {
            **row,
            "model": "xgboost",
            **dict(
                zip(
                    comparison.PROBABILITY_COLUMNS,
                    {
                        "risk_on": ("0.999", "0.0005", "0.0005"),
                        "transition": ("0.0005", "0.999", "0.0005"),
                        "risk_off": ("0.0005", "0.0005", "0.999"),
                    }[row["actual"]],
                    strict=True,
                )
            ),
        }
        for row in markov_rows
    )
    _write_csv(oos_path, rows, FIELDS)

    diagnostics_path = v5 / "selection-diagnostics.csv"
    diagnostics = _selection_rows(oos_path)
    _write_csv(diagnostics_path, diagnostics, tuple(diagnostics[0]))
    _refresh_v5_payload(v5)

    report = comparison.build_comparison(v5, v4)

    models = report["v5_causal_multiscale_ensemble_vs_v5_markov"][
        "selection_gate_crosscheck"
    ]["models"]
    assert set(models) == {"causal_multiscale_ensemble", "markov", "xgboost"}
    assert models["xgboost"]["reference_model"] == "markov"
    assert models["xgboost"]["selected"] is True
    assert models["xgboost"]["gate_passed"] is True
    assert models["xgboost"]["gate_reason"] == "passed"
    assert models["xgboost"]["n_predictions"] == 2
    assert models["xgboost"]["fallback_count"] == 0
    assert models["xgboost"]["log_loss"] < models["markov"]["log_loss"]
    assert models["xgboost"]["brier"] < models["markov"]["brier"]
    assert models["xgboost"]["matched_metric_crosscheck"] is True


def test_probability_token_bytes_and_numeric_parity_are_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader((v4 / "oos-predictions.csv").open(encoding="utf-8")))
    rows[0]["p_risk_on"] = "0.70"
    _write_csv(v4 / "oos-predictions.csv", rows, FIELDS)
    _refresh_frozen_inventory(v4, monkeypatch)

    report = comparison.build_comparison(v5, v4)

    parity = report["v5_markov_vs_frozen_v4_markov"]["primary_selection"]["probability_parity"]
    assert parity["probability_token_bytes"]["exact_parity"] is False
    assert parity["probability_numeric"]["exact_float_parity"] is True


@pytest.mark.parametrize("field,new_value,error", [
    ("actual", "transition", "origins/outcomes differ"),
    ("evaluation_split", "holdout", "origins/outcomes differ"),
])
def test_common_key_label_or_split_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    new_value: str,
    error: str,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader((v5 / "oos-predictions.csv").open(encoding="utf-8")))
    markov = next(row for row in rows if row["model"] == "markov")
    markov[field] = new_value
    _write_csv(v5 / "oos-predictions.csv", rows, FIELDS)
    _refresh_v5_payload(v5)

    with pytest.raises(comparison.ComparisonError, match=error):
        comparison.build_comparison(v5, v4)


def test_duplicate_oos_key_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    rows = list(csv.DictReader((v5 / "oos-predictions.csv").open(encoding="utf-8")))
    rows.append(dict(rows[0]))
    _write_csv(v5 / "oos-predictions.csv", rows, FIELDS)
    _refresh_v5_payload(v5)

    with pytest.raises(comparison.ComparisonError, match="duplicate key"):
        comparison.build_comparison(v5, v4)


def test_v5_manifest_and_frozen_inventory_tampering_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    payload_path = v5 / "regime-results.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["model"]["core_artifacts"]["oos_predictions"]["sha256"] = "0" * 64
    payload_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(comparison.ComparisonError, match="core artifact hash mismatch"):
        comparison.build_comparison(v5, v4)

    _refresh_v5_payload(v5)
    (v4 / "oos-predictions.csv").write_bytes(b"mutated")
    with pytest.raises(comparison.ComparisonError, match="artifact hash mismatch"):
        comparison.build_comparison(v5, v4)


def test_selection_gate_metric_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    path = v5 / "selection-diagnostics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    row = next(value for value in rows if value["model"] == "causal_multiscale_ensemble")
    row["log_loss"] = str(float(row["log_loss"]) + 0.01)
    row["absolute_log_loss_improvement"] = str(
        float(row["reference_log_loss"]) - float(row["log_loss"])
    )
    fields = tuple(rows[0])
    _write_csv(path, rows, fields)
    _refresh_v5_payload(v5)

    with pytest.raises(comparison.ComparisonError, match="log_loss mismatch"):
        comparison.build_comparison(v5, v4)


@pytest.mark.parametrize("field", ("raw_p_value", "holm_adjusted_p_value"))
def test_selection_family_bootstrap_or_holm_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    path = v5 / "selection-diagnostics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    challenger = next(
        row for row in rows if row["model"] == "causal_multiscale_ensemble"
    )
    challenger[field] = str(float(challenger[field]) + 0.0005)
    _write_csv(path, rows, tuple(rows[0]))
    # Refresh both the artifact manifest and payload diagnostics so only an
    # independent OOS-family recomputation can detect the forged evidence.
    _refresh_v5_payload(v5)

    with pytest.raises(comparison.ComparisonError, match=rf"{field} mismatch"):
        comparison.build_comparison(v5, v4)


def test_payload_only_selection_family_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    payload_path = v5 / "regime-results.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    challenger = next(
        row
        for row in payload["model"]["selection_diagnostics"]
        if row["model"] == "causal_multiscale_ensemble"
    )
    challenger["raw_p_value"] += 0.0005
    payload_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        comparison.ComparisonError,
        match="payload selection diagnostics raw_p_value mismatch",
    ):
        comparison.build_comparison(v5, v4)


def test_cli_help_states_exact_match_and_no_promotion() -> None:
    help_text = comparison._parser().format_help()

    assert "exact-common-origin" in help_text
    assert "never makes a promotion decision" in help_text
    assert "artifacts/baselines/v4-20260821" in help_text


def test_cli_writes_canonical_derived_only_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "report.json"

    assert comparison.main(
        [
            "--v5-artifacts",
            str(v5),
            "--v4-artifacts",
            str(v4),
            "--output",
            str(output),
        ]
    ) == 0

    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert b"2024-01-05" not in raw
    assert json.loads(raw)["schema_version"] == "regime-v5-v4-matched-comparison/1"


def test_sibling_cli_build_layout_resolves_payload_from_artifacts_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    artifacts = tmp_path / "v5-live-artifacts"
    v5.rename(artifacts)
    payload_directory = tmp_path / "v5-live"
    payload_directory.mkdir()
    (artifacts / "regime-results.json").rename(
        payload_directory / "regime-results.json"
    )

    report = comparison.build_comparison(artifacts, v4)

    assert report["inputs"]["v5"]["regime_results"]["path"] == "regime-results.json"


def test_fx_sidecar_metrics_are_independently_recomputed_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    fx_ablation = _evaluated_fx_fixture(v5)
    _refresh_v5_payload(v5, fx_ablation=fx_ablation)

    report = comparison.build_comparison(v5, v4)

    fx = report["fx_ablation"]
    assert fx["comparison_status"] == "evaluated"
    assert fx["aggregate_crosscheck"] is True
    assert fx["common_origins"]["count"] == 2
    assert [row["variant"] for row in fx["variants"]] == list(comparison.FX_VARIANTS)
    assert "fx_ablation_oos" in report["inputs"]["v5"]
    assert fx["interpretation"] == "diagnostic_only_not_a_promotion_decision"


def test_fx_sidecar_drift_from_payload_aggregate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    fx_ablation = _evaluated_fx_fixture(v5)
    sidecar = v5 / comparison.FX_ABLATION_OOS_PATH
    rows = list(csv.DictReader(sidecar.open(encoding="utf-8")))
    rows[0]["p_risk_on"] = "0.6"
    rows[0]["p_transition"] = "0.3"
    _write_csv(sidecar, rows, comparison.FX_ABLATION_OOS_COLUMNS)
    _refresh_v5_payload(v5, fx_ablation=fx_ablation)

    with pytest.raises(comparison.ComparisonError, match="independent log_loss mismatch"):
        comparison.build_comparison(v5, v4)


def test_empty_non_evaluated_fx_sidecar_stays_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v5, v4 = _fixture(tmp_path, monkeypatch)
    _write_csv(
        v5 / comparison.FX_ABLATION_OOS_PATH,
        [],
        comparison.FX_ABLATION_OOS_COLUMNS,
    )
    _refresh_v5_payload(v5)

    report = comparison.build_comparison(v5, v4)

    assert report["fx_ablation"]["comparison_status"] == "unavailable"
    assert report["inputs"]["v5"]["fx_ablation_oos"]["row_count"] == 0
