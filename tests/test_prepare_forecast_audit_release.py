"""Release boundary failures must preserve issued decisions and last-good files."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.prepare_forecast_audit_release as release


@pytest.fixture
def pair():
    source = {
        "meta": {"publication_status": "reviewed_publication", "publication_review": {"old": True},
                 "generation_manifest_sha256": "a" * 64, "data_as_of": "2026-09-04T20:00:00+00:00", "generation_id": "unchanged"},
        "model": {"champion": "original", "lifecycle": {"selection": {"status": "unchanged"},
            "publication": {"status": "reviewed_publication"}, "deployment": {"status": "operating"}},
            "selection_status": "unchanged", "execution_parameters": {"keep": True}, "research_artifacts": {}},
        "weekly": [{"data_as_of": "2026-09-04T20:00:00+00:00", "current": {"state": "risk_on"},
                    "next_week": {"probabilities": [0.6, 0.3, 0.1]}, "duration_context": {"weeks": 4}}],
        "forecast": {"target": "2026-09-11"}, "selection": {"model": "original"}, "label": {"spec_sha256": "b" * 64},
        "research": {"existing": {"preserved": True}, "extensions": {"build": {
            "compiled_at": "original", "source_payload_sha256": "c" * 64,
            "outputs": {"existing": "d" * 64}}},
            "operational_diagnostics": {
                "as_of": "old", "source_forecast_hashes": ["issued"], "source_evaluation_hashes": ["investment"],
                "probability_scores": {"completed_weeks": 0},
                "continuous_segments": [], "cross_segment_cumulative_return": None,
                "timing": {"issued_entry_count": 1, "deadline_observed_entries": 1,
                    "on_time_entries": 1, "on_time_rate": 1, "median_lead_seconds": 60,
                    "rows": [{"target_week": "2026-09-11", "decision_at": "decision", "scheduled_entry_at": "deadline",
                              "lead_seconds": 60, "on_time": True}]}}},
    }
    candidate = deepcopy(source)
    candidate["meta"]["publication_status"] = "unpublished"
    for name in ("publication_review", "generation_manifest_sha256"):
        candidate["meta"].pop(name)
    candidate["model"]["lifecycle"]["publication"] = {"status": "unpublished"}
    candidate["model"]["lifecycle"]["deployment"] = {"status": "reviewed"}
    for name in release.BLOCKS:
        candidate["research"][name] = {"rows": [{"model": "test", "log_loss": 0.3}],
            "artifacts": [{"label": "전체 결과 JSON", "url": f"./data/{name}.json"}]}
    operational = candidate["research"]["operational_diagnostics"]
    operational["as_of"] = "new"
    operational["probability_scores"] = {"completed_weeks": 1, "log_loss": 0.2, "evaluation_manifest_sha256": "e" * 64}
    operational["timing"].update(on_time_entries=0, on_time_rate=0, median_lead_seconds=-1)
    operational["timing"]["rows"][0].update(issued_at="receipt", lead_seconds=-1, on_time=False, issue_evidence={"eligible": False})
    build = candidate["research"]["extensions"]["build"]
    build.update(forecast_audit_present=True, forecast_audit={"cache_key": "f" * 64},
                 forecast_audit_operational={"scope": "isolated_copy_of_actual_issued_ledger"})
    build["outputs"].update({name: "0" * 64 for name in release.OUTPUT_RECEIPTS})
    return source, candidate


def test_research_probability_evaluation_and_narrow_receipts_preserve_inputs(pair):
    before = deepcopy(pair)
    release.validate_forecast_audit_only_update(*pair)
    assert pair == before


def test_legacy_build_can_add_only_the_exact_official_generation_identity(pair):
    source, candidate = pair
    candidate["research"]["extensions"]["build"].update({
        name: source["meta"][name] for name in ("generation_id", "data_as_of")})
    release.validate_forecast_audit_only_update(source, candidate)


@pytest.mark.parametrize("name", ["generation_id", "data_as_of"])
def test_build_identity_cannot_relabel_the_frozen_generation(pair, name):
    source, candidate = pair
    candidate["research"]["extensions"]["build"][name] = "different generation"
    with pytest.raises(release.ReleasePreparationError, match=f"protected {name}"):
        release.validate_forecast_audit_only_update(source, candidate)


@pytest.mark.parametrize("name", ["generation_id", "data_as_of"])
def test_existing_build_identity_cannot_be_removed(pair, name):
    source, candidate = pair
    source["research"]["extensions"]["build"][name] = source["meta"][name]
    with pytest.raises(release.ReleasePreparationError, match=f"protected {name}"):
        release.validate_forecast_audit_only_update(source, candidate)


@pytest.mark.parametrize("path", [
    ("forecast", "target"), ("selection", "model"), ("label", "spec_sha256"),
    ("meta", "generation_id"), ("meta", "data_as_of"), ("model", "champion"),
    ("model", "selection_status"), ("model", "lifecycle", "selection", "status"),
    ("model", "execution_parameters", "keep"), ("model", "research_artifacts"),
    ("weekly", 0, "current", "state"), ("weekly", 0, "next_week", "probabilities"),
    ("weekly", 0, "duration_context", "weeks"), ("research", "existing"),
    ("research", "extensions", "build", "compiled_at"),
    ("research", "extensions", "build", "source_payload_sha256"),
    ("research", "extensions", "build", "outputs", "existing"),
    ("research", "operational_diagnostics", "source_forecast_hashes"),
    ("research", "operational_diagnostics", "source_evaluation_hashes"),
    ("research", "operational_diagnostics", "continuous_segments"),
    ("research", "operational_diagnostics", "cross_segment_cumulative_return"),
    ("research", "operational_diagnostics", "timing", "issued_entry_count"),
    ("research", "operational_diagnostics", "timing", "rows", 0, "decision_at"),
    ("research", "operational_diagnostics", "timing", "rows", 0, "scheduled_entry_at"),
])
def test_official_and_investment_content_cannot_hide_inside_research_release(pair, path):
    source, candidate = pair
    target = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "tampered"
    with pytest.raises(release.ReleasePreparationError, match="changed protected content"):
        release.validate_forecast_audit_only_update(source, candidate)


@pytest.mark.parametrize("change", ["review", "binding", "missing_block", "issued_candidate"])
def test_previous_approval_or_partial_extension_cannot_be_reused(pair, change):
    source, candidate = pair
    if change == "review":
        candidate["meta"]["publication_review"] = {"old": True}
    elif change == "binding":
        candidate["meta"]["generation_manifest_sha256"] = "a" * 64
    elif change == "missing_block":
        candidate["research"].pop("calibration_audit")
    else:
        candidate["meta"]["publication_status"] = "reviewed_publication"
    with pytest.raises(release.ReleasePreparationError):
        release.validate_forecast_audit_only_update(source, candidate)


def test_preview_results_allow_relocated_sources_and_fresh_clocks_only(pair):
    _, preview = pair
    blocks = {name: deepcopy(preview["research"][name]) for name in release.BLOCKS}
    for name, block in blocks.items():
        block["generated_at"] = "fresh"
        block["publication_provenance"] = {"actual_inputs": "new verified receipt"}
    blocks["forecast_research"].update(source_hashes={"frame": "0" * 64}, source_context={"path": "logical"})
    blocks["calibration_audit"]["sources"] = [{"id": "logical", "sha256": "a" * 64}]
    blocks["forecast_information"].update(sources=[{"id": "optional"}], blocks={"optional": {"status": "not_evaluable"}})
    operational = deepcopy(preview["research"]["operational_diagnostics"])
    operational["as_of"] = "fresh evaluation clock"
    operational["probability_scores"]["evaluation_manifest_sha256"] = "fresh record hash"
    result = release.validate_reproduced_preview(preview, blocks, operational)
    assert set(result) == {*release.BLOCKS, "operational_diagnostics"}
    assert all(row["results_exact"] for row in result.values())


@pytest.mark.parametrize("name", [*release.BLOCKS, "operational_diagnostics"])
def test_changed_numeric_results_cannot_be_explained_away_as_new_provenance(pair, name):
    _, preview = pair
    blocks = {key: deepcopy(preview["research"][key]) for key in release.BLOCKS}
    operational = deepcopy(preview["research"]["operational_diagnostics"])
    if name == "operational_diagnostics":
        operational["probability_scores"]["completed_weeks"] = 99
    else:
        blocks[name]["rows"][0]["log_loss"] = 0.01
    with pytest.raises(release.ReleasePreparationError, match=f"reproduced {name}"):
        release.validate_reproduced_preview(preview, blocks, operational)


def test_frozen_input_validation_checks_hashes_then_issued_labels(tmp_path, pair):
    source, _ = pair
    index = pd.DatetimeIndex([source["meta"]["data_as_of"]])
    states = pd.Series(["risk_on"], index=index)
    canonical = pd.DataFrame({"spy_close": [100.]}, index=index)
    canonical.to_pickle(tmp_path / "canonical.pkl")
    states.to_pickle(tmp_path / "states.pkl")
    manifest = {"data_as_of": source["meta"]["data_as_of"], "frames": {
        "canonical": release.frame_sha256(canonical), "states": release.frame_sha256(states)}}
    (tmp_path / "input-manifest.json").write_text(json.dumps(manifest))
    release._frozen_inputs(source, tmp_path)
    source["weekly"][0]["current"]["state"] = "risk_off"
    with pytest.raises(release.ReleasePreparationError, match="labels differ"):
        release._frozen_inputs(source, tmp_path)
    states.iloc[0] = "risk_off"
    states.to_pickle(tmp_path / "states.pkl")
    with pytest.raises(release.ReleasePreparationError, match="states hash differs"):
        release._frozen_inputs(source, tmp_path)


def test_staging_never_targets_live_original_inputs_or_existing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "project_root", lambda: tmp_path)
    source = tmp_path / "build/original"
    source.mkdir(parents=True)
    for path, message in [(tmp_path / "publication/live", "below build"),
                          (source, "must not exist"), (source / "release", "overlaps"),
                          (tmp_path / "build", "must not exist")]:
        with pytest.raises(release.ReleasePreparationError, match=message):
            release._require_destination(path, [source])


def _staging_args(tmp_path, pair, monkeypatch):
    source, preview = pair
    for name in ("source", "artifacts", "input", "existing", "information"):
        (tmp_path / name).mkdir()
    for name in release.PUBLICATION_MEMBERS:
        (tmp_path / "source" / name).write_text(json.dumps(source if name == "regime-results.json" else {}))
    for name in ("canonical.pkl", "states.pkl", "input-manifest.json"):
        (tmp_path / "input" / name).write_text("protected input")
    for name in ("oos-predictions.csv", "transition-oos-predictions.csv", "transition-candidate-forecasts.csv"):
        (tmp_path / "artifacts" / name).write_text("column\n1\n")
    (tmp_path / "preview.json").write_text(json.dumps(preview))
    (tmp_path / "ledger.sqlite3").write_bytes(b"protected original ledger")
    monkeypatch.setattr(release, "project_root", lambda: tmp_path)
    generation = {"payload": source, "input_snapshot": {}, "label_spec": {"path": "config/label-spec.json"}, "selection_family": {}}
    monkeypatch.setattr(release, "validate_generation_manifest", lambda *a, **k: generation)
    monkeypatch.setattr(release, "validate_public_live_derived_payload", lambda *a: None)
    monkeypatch.setattr(release, "validate_dashboard_payload", lambda *a: None)
    monkeypatch.setattr(release, "validate_research_extensions", lambda *a, **k: None)
    monkeypatch.setattr(release, "reject_raw_provider_material", lambda *a: None)
    monkeypatch.setattr(release, "_frozen_inputs", lambda *a: (pd.DataFrame(), pd.Series(dtype=object)))
    return {"source_path": tmp_path / "source/regime-results.json", "source_artifacts": tmp_path / "artifacts",
        "preview_path": tmp_path / "preview.json", "input_directory": tmp_path / "input", "ledger_path": tmp_path / "ledger.sqlite3",
        "research_cache": tmp_path / "build/cache", "existing_sources": tmp_path / "existing", "information_sources": tmp_path / "information",
        "output_root": tmp_path / "build/release", "reviewed_at": datetime(2026, 9, 7, tzinfo=timezone.utc)}


def test_production_failure_never_creates_reviewed_output(tmp_path, pair, monkeypatch):
    args = _staging_args(tmp_path, pair, monkeypatch)
    import regime_lab.research.forecast_publication as producer

    def fail(*a, **k):
        raise ValueError("normal research failed")

    monkeypatch.setattr(producer, "build_forecast_publication_research", fail)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="normal research failed"):
        release.prepare_forecast_audit_release(**args)
    assert not args["output_root"].exists()
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_failed_promotion_removes_only_own_staging_and_preserves_last_good(tmp_path, pair, monkeypatch):
    args = _staging_args(tmp_path, pair, monkeypatch)
    import regime_lab.research.forecast_publication as producer
    import scripts.audit_outputs as auditor
    source, preview = pair
    blocks = {name: deepcopy(preview["research"][name]) for name in release.BLOCKS}
    monkeypatch.setattr(producer, "build_forecast_publication_research", lambda *a, **k: {"blocks": blocks, "provenance": {}})

    def evaluate(*a):
        a[-1].mkdir()
        (a[-1] / "operational-diagnostics.json").write_text(json.dumps(preview["research"]["operational_diagnostics"]))
        return {"issued_forecasts_unchanged": True, "investment_evaluations_unchanged": True}

    def normalize(value):
        result = deepcopy(preview)
        for name in release.BLOCKS:
            result["research"].pop(name)
        return result

    monkeypatch.setattr(release, "evaluate_copy", evaluate)
    monkeypatch.setattr(release, "reviewed_candidate_payload", normalize)
    monkeypatch.setattr(release, "verify_staged_v5_research_artifacts", lambda *a: None)
    monkeypatch.setattr(release, "build_generation_manifest", lambda **k: {})
    monkeypatch.setattr(release, "bind_payload_to_generation_manifest", lambda payload, manifest: payload)
    monkeypatch.setattr(release, "_build_expected_comparison", lambda **k: {})
    monkeypatch.setattr(auditor, "audit", lambda *a, **k: {"ok": True})

    def fail(**kwargs):
        target = kwargs["output_path"]
        target.parent.mkdir(parents=True)
        target.write_text("partial output")
        raise RuntimeError("independent review rejected candidate")

    monkeypatch.setattr(release, "promote", fail)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="independent review rejected"):
        release.prepare_forecast_audit_release(**args)
    assert not args["output_root"].exists()
    assert not list((tmp_path / "build").glob(".release-*"))
    assert all(p.read_bytes() == raw for p, raw in before.items())
