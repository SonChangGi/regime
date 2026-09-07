"""Release-boundary tests; synthetic staging fixtures are not financial evidence."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_comprehensive_release", ROOT / "scripts/prepare_comprehensive_release.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def payloads() -> tuple[dict, dict]:
    source = json.loads((ROOT / "publication/live/regime-results.json").read_text())
    preview = deepcopy(source)
    preview["meta"]["publication_status"] = "unpublished"
    preview["meta"].pop("publication_review")
    preview["meta"].pop("generation_manifest_sha256")
    preview["model"]["lifecycle"]["deployment"] = {"status": "candidate"}
    preview["model"]["lifecycle"]["publication"] = {"status": "unpublished"}
    return source, preview


def test_research_allowlist_preserves_every_other_official_field(payloads) -> None:
    source, preview = payloads
    preview["research"]["decision_research_v2"] = {"test": "synthetic"}
    preview["research"]["prospective_decision_shadow"]["allocation_research_v2"] = {}
    preview["research"]["label_sensitivity"] = {"test": "synthetic"}
    preview["model"]["directional_transition"] = {"test": "synthetic"}
    preview["weekly"][0]["duration_context"] = {"test": "synthetic"}
    preview["weekly"][0]["directional_risk_raw"] = {"test": "synthetic"}
    preview["model"]["execution_parameters"][
        "directional_maximum_selection_origins"
    ] = None
    before = deepcopy(source)
    MODULE.validate_research_only_update(source, preview)
    assert source == before


@pytest.mark.parametrize(
    "path",
    [
        ("forecast",),
        ("meta", "data_as_of"),
        ("meta", "generation_id"),
        ("model", "champion"),
        ("model", "selection_diagnostics"),
        ("model", "selection_status"),
        ("model", "lifecycle", "selection"),
        ("model", "execution_parameters", "profile"),
        ("model", "research_artifacts", "directional_oos_predictions", "path"),
        ("research", "prospective_decision_shadow", "current_signal"),
        ("research", "prospective_decision_shadow", "prospective_ledger"),
        ("weekly", 0, "current"),
        ("weekly", 0, "next_week"),
        ("weekly", 0, "model_forecasts"),
        ("sources",),
    ],
)
def test_official_content_changes_are_rejected_before_candidate_normalization(
    payloads, path
) -> None:
    source, preview = payloads
    target = preview
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "synthetic mutation"
    with pytest.raises(
        MODULE.ReleasePreparationError, match="changed protected content"
    ):
        MODULE.validate_research_only_update(source, preview)


@pytest.mark.parametrize("change", ["duplicate", "removed", "reordered"])
def test_origin_identity_and_order_cannot_be_changed(payloads, change) -> None:
    source, preview = payloads
    if change == "duplicate":
        preview["weekly"][1] = deepcopy(preview["weekly"][0])
    elif change == "removed":
        preview["weekly"].pop()
    else:
        preview["weekly"].reverse()
    with pytest.raises(MODULE.ReleasePreparationError, match="/weekly"):
        MODULE.validate_research_only_update(source, preview)


def test_changed_research_cannot_reuse_an_existing_approval(payloads) -> None:
    source, preview = payloads
    preview["meta"]["publication_review"] = deepcopy(
        source["meta"]["publication_review"]
    )
    with pytest.raises(MODULE.ReleasePreparationError, match="must not inherit"):
        MODULE.validate_research_only_update(source, preview)


def _decision_run(tmp_path: Path) -> tuple[dict, dict, dict]:
    paths = {}
    for name in (
        "source_payload",
        "ledger",
        "oos_predictions",
        "transition_predictions",
        "outcome_rows",
        "canonical_cache",
    ):
        paths[name] = tmp_path / f"{name}.txt"
        paths[name].write_text(name)
    research = tmp_path / "research"
    research.mkdir()
    outputs = {}
    for name in (
        "allocation-v2.json",
        "decision-research-v2.json",
        "operational-diagnostics.json",
    ):
        raw = b'{"synthetic":true}\n'
        (research / name).write_bytes(raw)
        outputs[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    source = {
        "weekly": [{"date": "2026-09-04"}],
        "model": {"champion": "test", "selection_end": "2023-01-01"},
    }
    run = {
        "schema_version": "regime-decision-research-run/1",
        "origin": "2026-09-04",
        "forecast_model": "test",
        "selection_end": "2023-01-01",
        "source_inputs": {
            name: {
                "path": "/previous-machine/private/input",
                "sha256": MODULE._sha256(path),
            }
            for name, path in paths.items()
        },
        "outputs": outputs,
    }
    return paths, source, run


def test_snapshot_relocation_uses_content_identity_without_reading_old_machine_paths(
    tmp_path,
) -> None:
    paths, source, run = _decision_run(tmp_path)
    MODULE.validate_decision_run(
        run, paths=paths, source=source, research_root=tmp_path
    )


@pytest.mark.parametrize(
    "member", ["ledger", "oos_predictions", "source_payload", "canonical_cache"]
)
def test_different_original_inputs_cannot_be_relabelled_as_the_current_run(
    tmp_path, member
) -> None:
    paths, source, run = _decision_run(tmp_path)
    paths[member].write_text("different snapshot")
    with pytest.raises(
        MODULE.ReleasePreparationError, match=f"input mismatch: {member}"
    ):
        MODULE.validate_decision_run(
            run, paths=paths, source=source, research_root=tmp_path
        )


def test_output_tampering_and_old_origin_are_rejected(tmp_path) -> None:
    paths, source, run = _decision_run(tmp_path)
    (tmp_path / "research/allocation-v2.json").write_text("{}")
    with pytest.raises(MODULE.ReleasePreparationError, match="output mismatch"):
        MODULE.validate_decision_run(
            run, paths=paths, source=source, research_root=tmp_path
        )
    run["origin"] = "2026-08-28"
    with pytest.raises(MODULE.ReleasePreparationError, match="different final origin"):
        MODULE.validate_decision_run(
            run, paths=paths, source=source, research_root=tmp_path
        )


def _synthetic_directional_coverage():
    index = pd.date_range("2020-01-03", periods=50, freq="W-FRI", tz="UTC")
    states = pd.Series(
        [("risk_on", "transition", "risk_off")[(n // 7) % 3] for n in range(50)],
        index=index,
    )
    rows = []
    # Explicit expected boundaries: target must finish before selection cutoff;
    # diagnostic origin begins at cutoff. Cross-boundary targets are purged.
    positions = {
        1: [*range(13, 19), *range(20, 49)],
        4: range(20, 46),
        13: range(25, 37),
    }
    for horizon, origins in positions.items():
        for origin in origins:
            future = states.iloc[origin + 1 : origin + horizon + 1]
            changes = future[future.ne(states.iloc[origin])]
            for model in (
                "empirical_first_passage",
                "markov_first_passage",
                "regularized_multinomial",
                "shallow_multiclass_xgboost",
            ):
                rows.append(
                    {
                        "horizon_weeks": horizon,
                        "model": model,
                        "origin_date": index[origin],
                        "target_end": index[origin + horizon],
                        "evaluation_split": (
                            "selection" if origin < 20 else "retrospective_diagnostic"
                        ),
                        "current_state": states.iloc[origin],
                        "actual_outcome": (
                            changes.iloc[0] if len(changes) else "no_departure"
                        ),
                    }
                )
    return states, pd.DataFrame(rows)


def test_full_directional_coverage_counts_purged_targets_and_first_departures() -> None:
    states, rows = _synthetic_directional_coverage()
    assert len(rows) == (35 + 26 + 12) * 4
    MODULE.validate_directional_coverage(
        rows,
        states,
        selection_end=states.index[20].date().isoformat(),
        minimum_train_weeks=12,
    )


@pytest.mark.parametrize(
    "mutation", ["dropped", "duplicate", "target", "split", "outcome", "model"]
)
def test_partial_or_misaligned_directional_research_cannot_claim_full_coverage(
    mutation,
) -> None:
    states, rows = _synthetic_directional_coverage()
    if mutation == "dropped":
        rows = rows.iloc[1:]
    elif mutation == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    elif mutation == "target":
        rows.at[0, "target_end"] = states.index[15]
    elif mutation == "split":
        rows.loc[0, "evaluation_split"] = "retrospective_diagnostic"
    elif mutation == "outcome":
        rows.loc[0, "actual_outcome"] = "unknown"
    else:
        rows.loc[0, "model"] = "unknown"
    with pytest.raises(MODULE.ReleasePreparationError, match="directional research"):
        MODULE.validate_directional_coverage(
            rows,
            states,
            selection_end=states.index[20].date().isoformat(),
            minimum_train_weeks=12,
        )


def _arguments(root: Path) -> dict:
    return {
        "source_path": root / "source/publication/regime-results.json",
        "source_manifest_path": root / "source/publication/generation-manifest.json",
        "source_artifacts": root / "source/artifacts",
        "ledger_path": root / "source/ledger.sqlite3",
        "preview_path": root / "research-run/preview/data/regime-results.json",
        "research_root": root / "research-run",
        "output_root": root / "build/candidate",
        "reviewed_at": datetime(2026, 9, 7, tzinfo=timezone.utc),
    }


def test_publication_directory_or_nested_original_artifacts_are_never_mutation_targets(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(MODULE, "project_root", lambda: tmp_path)
    args = _arguments(tmp_path)
    args["output_root"] = tmp_path / "publication/live"
    with pytest.raises(MODULE.ReleasePreparationError, match="below build"):
        MODULE.prepare_release(**args)
    args["source_artifacts"] = tmp_path / "build/source/artifacts"
    args["output_root"] = args["source_artifacts"] / "candidate"
    with pytest.raises(
        MODULE.ReleasePreparationError, match="overlaps a read-only input"
    ):
        MODULE.prepare_release(**args)
    assert not (tmp_path / "publication").exists()


def test_existing_last_good_candidate_is_preserved(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "project_root", lambda: tmp_path)
    args = _arguments(tmp_path)
    args["output_root"].mkdir(parents=True)
    last_good = args["output_root"] / "last-good.json"
    last_good.write_text("trusted previous result")
    with pytest.raises(MODULE.ReleasePreparationError, match="must not exist"):
        MODULE.prepare_release(**args)
    assert last_good.read_text() == "trusted previous result"


@pytest.mark.parametrize(
    "fail_at",
    ["comparison", "promotion", "packaging", "post_install", "inputs_changed"],
)
def test_release_failure_removes_only_its_own_stage(
    tmp_path, monkeypatch, payloads, fail_at
) -> None:
    """Fault-inject orchestration; actual economic/review checks run in the release drill."""
    source, preview = payloads
    args = _arguments(tmp_path)
    monkeypatch.setattr(MODULE, "project_root", lambda: tmp_path)
    args["source_artifacts"].mkdir(parents=True)
    (args["source_artifacts"] / "selection-family-audit.json").write_text("{}")
    args["source_path"].parent.mkdir(parents=True)
    args["source_path"].write_text("protected source")
    args["preview_path"].parent.mkdir(parents=True)
    args["preview_path"].write_text(json.dumps(preview))
    for name in set(MODULE.RESEARCH_DOCUMENTS) | {
        "input/input-manifest.json",
        "research/decision-research-run.json",
        "model-economics/directional-execution.json",
    }:
        path = args["research_root"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for key in MODULE.DIRECTIONAL_FRAMES:
        (
            args["research_root"]
            / "model-economics"
            / MODULE.V5_RESEARCH_ARTIFACTS[key].path
        ).write_text("synthetic\n")
    source_before = args["source_path"].read_bytes()
    monkeypatch.setattr(
        MODULE,
        "_research_input_paths",
        lambda **kwargs: {"source": args["source_path"]},
    )
    monkeypatch.setattr(MODULE, "_validate_research_inputs", lambda **kwargs: {})
    generation = {
        "payload": source,
        "input_snapshot": {},
        "label_spec": {"path": "label.json"},
        "selection_family": {},
    }
    monkeypatch.setattr(
        MODULE, "validate_generation_manifest", lambda *a, **k: generation
    )
    monkeypatch.setattr(MODULE, "validate_public_live_derived_payload", lambda *a: None)
    monkeypatch.setattr(MODULE, "validate_dashboard_payload", lambda *a: None)
    monkeypatch.setattr(MODULE, "verify_staged_v5_research_artifacts", lambda *a: None)
    monkeypatch.setattr(MODULE, "verify_artifact_inventory", lambda *a: None)
    monkeypatch.setattr(MODULE, "build_generation_manifest", lambda **k: {})
    monkeypatch.setattr(MODULE, "bind_payload_to_generation_manifest", lambda p, m: p)

    def fail():
        raise ValueError("injected release failure")

    def comparison(**kwargs):
        if fail_at == "comparison":
            fail()
        return {}

    def promotion(**kwargs):
        if fail_at == "promotion":
            fail()
        for name in MODULE.PUBLICATION_MEMBERS:
            path = kwargs["output_path"].with_name(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    def packaging(**kwargs):
        assert (
            kwargs["staged_generation_contract_directory"]
            == tmp_path / "publication/live"
        )
        assert kwargs["publication_mode"] == "live-derived"
        if fail_at == "packaging":
            fail()
        if fail_at == "inputs_changed":
            args["source_path"].write_bytes(b"external concurrent update")
        kwargs["output_directory"].mkdir()

    def verification(path):
        if fail_at == "post_install" and path.parent == args["output_root"]:
            fail()
        return {"ok": True}

    monkeypatch.setattr(MODULE, "_build_expected_comparison", comparison)
    monkeypatch.setattr(MODULE, "promote", promotion)
    monkeypatch.setattr(MODULE, "package_public_dashboard", packaging)
    monkeypatch.setattr(MODULE, "verify_public_package", verification)
    message = (
        "inputs changed during preparation"
        if fail_at == "inputs_changed"
        else "injected release failure"
    )
    with pytest.raises(ValueError, match=message):
        MODULE.prepare_release(**args)
    expected = (
        b"external concurrent update" if fail_at == "inputs_changed" else source_before
    )
    assert args["source_path"].read_bytes() == expected
    assert not args["output_root"].exists()
    assert not list((tmp_path / "build").glob(".candidate-*"))
    assert not (tmp_path / "publication").exists()
