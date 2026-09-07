"""Weekly research stays generation-bound without collecting in unit tests."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from regime_lab import cli
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.research import additional_sources, publication


CUTOFF = "2026-09-04T20:00:00+00:00"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(additional_sources.requests, "get", lambda *a, **k: pytest.fail("unexpected network request"))


@pytest.fixture
def generation():
    dates = pd.date_range("2026-08-21T20:00:00Z", periods=3, freq="W-FRI")
    canonical = pd.DataFrame({"spy_close": [100., 102., 103.], "spy_adjusted_open": [100., 102., 103.]}, index=dates)
    predictions = pd.DataFrame({"model": ["selected"] * 3, "evaluation_split": ["selection", "holdout", "holdout"], "origin_date": dates})
    dataset = SimpleNamespace(canonical=canonical, features=canonical.copy(), input_vintages=(), availability_basis="reconstructed_market")
    benchmark = SimpleNamespace(champion="selected", predictions=predictions,
        transition_benchmark=SimpleNamespace(predictions=pd.DataFrame({"value": [1]})),
        model_conditioned_asset_outcomes=pd.DataFrame({"value": [2]}),
        feature_ablation=SimpleNamespace(predictions=pd.DataFrame({"value": [3]})))
    payload = {
        "meta": {"mode": "live", "status": "ok", "generation_id": "test-generation", "data_as_of": CUTOFF},
        "model": {"profile": "standard", "champion": "selected", "selection_end": "2023-07-01", "candidate_manifest_sha256": "a" * 64},
        "selection": {"operating_champion": "selected"},
        "label": {"spec_sha256": "b" * 64},
        "forecast": {"origin_at": CUTOFF, "decision_at": "2026-09-05T12:00:00+00:00", "target_at": "2026-09-11T20:00:00+00:00"},
        "weekly": [{"date": "2026-09-04", "next_week": {"state": "risk_on", "probability": .9}}],
        "research": {"prospective_decision_shadow": {"schema_version": "regime-prospective-decision-shadow/2", "current_signal": {"action": "hold"}}, "existing_research": {"keep": True}},
    }
    return payload, dataset, benchmark


def fake_studies(monkeypatch, tmp_path, *, downside_cutoff=CUTOFF, matched=2):
    directory = tmp_path / "source-snapshot"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"sources":{}}')
    seen = {}
    monkeypatch.setattr(publication, "_source_snapshot", lambda cache, cutoff: seen.update(cache=cache, cutoff=cutoff) or directory)
    monkeypatch.setattr(publication, "_additional_research", lambda source, index: (pd.DataFrame({"vvix": 1.}, index=index), {"sources": []}))

    def allocation(weekly, canonical, predictions, **kwargs):
        seen["allocation"] = kwargs
        seen["canonical"] = canonical.copy()
        return {"schema_version": "regime-allocation-research/2", "role": "research_only_no_promotion", "affects_official_forecast": False, "affects_champion_selection": False, "affects_issued_ledger": False, "performance": {"weeks": matched}}

    def decision(predictions, transitions, canonical, **kwargs):
        seen["decision"] = kwargs
        return {"schema_version": "regime-decision-research/2", "diagnostic_origins": matched, "alert_budgets": []}

    def downside(canonical, **kwargs):
        seen["downside"] = kwargs
        return SimpleNamespace(summary={"status": "evaluated", "as_of": downside_cutoff, "latest": [], "comparisons": []})

    monkeypatch.setattr(publication, "build_allocation_shadow_v2", allocation)
    monkeypatch.setattr(publication, "build_decision_research_v2", decision)
    monkeypatch.setattr(publication, "run_downside_research", downside)
    monkeypatch.setattr(publication, "ablation_diagnostics", lambda frame: {"ablation": []})
    monkeypatch.setattr(publication, "read_operational_diagnostics", lambda path: {"schema_version": "regime-operational-diagnostics/1", "as_of": "2026-09-05T12:01:00+00:00", "timing": {"issued_entry_count": 4}})
    return seen


def compose(generation, tmp_path, **kwargs):
    payload, dataset, benchmark = generation
    return publication.compose_live_publication_research(payload, dataset=dataset,
        benchmark=benchmark, contract_version="v5", profile_name=payload["model"]["profile"],
        ledger_path=tmp_path / "ledger.sqlite3", cache_directory=tmp_path / "cache", **kwargs)


def test_composition_uses_generation_inputs_and_preserves_official_decisions(monkeypatch, tmp_path, generation):
    seen = fake_studies(monkeypatch, tmp_path)
    payload, dataset, benchmark = generation
    before = deepcopy(payload)
    frame_before = dataset.canonical.copy()
    messages = []
    result = compose(generation, tmp_path, progress=messages.append)
    assert payload == before
    pd.testing.assert_frame_equal(dataset.canonical, frame_before)
    for field in ("meta", "model", "forecast", "selection", "weekly"):
        assert result[field] == before[field]
    assert result["research"]["existing_research"] == {"keep": True}
    for study in ("allocation", "decision", "downside"):
        assert seen[study]["selection_end"] == "2023-07-01"
    pd.testing.assert_frame_equal(seen["canonical"], frame_before)
    assert seen["decision"]["outcome_rows"].equals(benchmark.model_conditioned_asset_outcomes)
    assert seen["cutoff"] == CUTOFF
    research = result["research"]
    assert research["prospective_decision_shadow"]["current_signal"] == {"action": "hold"}
    assert research["prospective_decision_shadow"]["allocation_research_v2"]["performance"]["weeks"] == 2
    receipt = research["extensions"]["build"]
    assert receipt["generation_id"] == before["meta"]["generation_id"]
    assert receipt["source_payload_sha256"] == canonical_json_sha256_v1(before)
    assert receipt["operational_scope"] == "issued_ledger_before_current_generation_publication"
    assert receipt["outputs"]["decision_research_v2"] == canonical_json_sha256_v1(research["decision_research_v2"])
    assert receipt["input_frames"]["canonical"]["rows"] == 3
    assert len(messages) == 7


@pytest.mark.parametrize("profile,mode,contract", [("quick", "live", "v5"), ("standard", "demo", "v5"), ("full", "replay", "v5"), ("standard", "live", "v4")])
def test_nonproduction_paths_do_not_read_inputs_cache_or_ledger(tmp_path, profile, mode, contract):
    payload = {"meta": {"mode": mode}}
    assert publication.compose_live_publication_research(payload, dataset=None, benchmark=None,
        profile_name=profile, contract_version=contract, ledger_path=tmp_path / "missing",
        cache_directory=tmp_path / "cache") is payload
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("failure", ["wrong_cutoff", "lost_origins", "study_failure"])
def test_failed_or_stale_studies_leave_the_source_payload_untouched(monkeypatch, tmp_path, generation, failure):
    fake_studies(monkeypatch, tmp_path, downside_cutoff="2026-08-28T20:00:00+00:00" if failure == "wrong_cutoff" else CUTOFF, matched=1 if failure == "lost_origins" else 2)
    if failure == "study_failure":
        monkeypatch.setattr(publication, "run_downside_research", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("study failed")))
    before = deepcopy(generation[0])
    with pytest.raises((ValueError, RuntimeError)):
        compose(generation, tmp_path)
    assert generation[0] == before


def test_source_snapshot_refreshes_once_per_cutoff_and_retains_first_seen(monkeypatch, tmp_path):
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(content=url.encode(), raise_for_status=lambda: None)
    monkeypatch.setattr(additional_sources.requests, "get", fetch)
    first = publication._source_snapshot(tmp_path, CUTOFF)
    initial = json.loads((first / "manifest.json").read_text())
    assert publication._source_snapshot(tmp_path, CUTOFF) == first
    assert len(calls) == len(additional_sources.SOURCES)
    second = publication._source_snapshot(tmp_path, "2026-09-11T20:00:00+00:00")
    newer = json.loads((second / "manifest.json").read_text())
    assert len(calls) == 2 * len(additional_sources.SOURCES)
    for key, record in initial["sources"].items():
        assert newer["sources"][key]["first_seen_at"] == record["first_seen_at"]
        assert (first / record["file"]).stat().st_ino == (second / record["file"]).stat().st_ino
    record = initial["sources"]["vix"]
    (first / record["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        publication._source_snapshot(tmp_path, CUTOFF)
    assert len(calls) == 2 * len(additional_sources.SOURCES)


def test_failed_source_collection_does_not_publish_a_cutoff_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(publication, "collect_sources", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("download failed")))
    with pytest.raises(RuntimeError, match="download failed"):
        publication._source_snapshot(tmp_path, CUTOFF)
    assert not (tmp_path / "2026-09-04").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_cli_composes_before_publication_and_does_not_cut_over_on_failure(monkeypatch, tmp_path, generation, fail):
    import regime_lab.data as data_module
    import regime_lab.h10_store as h10_module
    payload, dataset, benchmark = generation
    output, artifacts = tmp_path / "result.json", tmp_path / "artifacts"
    output.write_bytes(b"previous complete result")
    artifacts.mkdir()
    (artifacts / "marker").write_bytes(b"previous complete artifacts")
    events = []
    config = {"model": {"final_holdout_start": "2023-07-01"}}
    collection = SimpleNamespace(cutoffs=(), records=())
    monkeypatch.setattr(cli, "_resolve_contract_write_targets", lambda **k: (output, artifacts))
    monkeypatch.setattr(cli, "_resolve_live_checkpoint_directory", lambda **k: tmp_path / "checkpoint")
    monkeypatch.setattr(cli, "load_config", lambda p: config)
    monkeypatch.setattr(cli, "_mutable_path", lambda value, **k: Path(value))
    monkeypatch.setattr(cli, "providers_for_live_config", lambda c: [])
    monkeypatch.setattr(cli, "verify_provider_rights", lambda *a, **k: None)
    monkeypatch.setattr(cli, "append_run_event", lambda *a, **k: None)
    monkeypatch.setattr(cli, "current_run_status", lambda *a: "analyzing")
    monkeypatch.setattr(cli, "automation_lock", lambda p: nullcontext())
    monkeypatch.setattr(cli, "verify_v5_preflight", lambda **k: {"source_fingerprint_sha256": "c" * 64})
    monkeypatch.setattr(cli, "require_v5_analysis_source_unchanged", lambda *a, **k: events.append("source_check"))
    monkeypatch.setattr(cli, "_backup_database_before_mutation", lambda *a, **k: None)
    monkeypatch.setattr(cli, "collect_live_data", lambda *a, **k: collection)
    monkeypatch.setattr(cli, "validate_collection_for_training", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_require_ac_power_before_analysis", lambda **k: None)
    monkeypatch.setattr(data_module, "SQLiteSnapshotStore", lambda p: nullcontext())
    monkeypatch.setattr(data_module, "H10ArchiveClient", object)
    monkeypatch.setattr(data_module, "H10Client", object)
    monkeypatch.setattr(h10_module, "refresh_h10_archive_store", lambda *a, **k: None)
    monkeypatch.setattr(h10_module, "refresh_h10_store", lambda *a, **k: SimpleNamespace(fx_features=None, fx_context=None, source_row={"id": "frb_h10", "status": "ok", "issues": []}))
    monkeypatch.setattr(cli, "build_weekly_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(cli, "build_dashboard_result", lambda *a, **k: (payload, benchmark))
    monkeypatch.setattr(cli, "_operational_inputs_for_generation", lambda *a, **k: ())
    monkeypatch.setattr(cli, "operational_input_manifest_sha256", lambda *a: "d" * 64)
    monkeypatch.setattr(cli, "_prospective_actual_states", lambda b: pd.Series("risk_on", index=dataset.canonical.index))
    monkeypatch.setattr(cli, "build_research_replay_input_document", lambda **k: {})
    monkeypatch.setattr(cli, "mature_forecast_evaluations", lambda *a, **k: SimpleNamespace(unresolved_due=()))
    ledger = SimpleNamespace(public_summary=lambda **k: {}, list_evaluations=lambda: [])
    monkeypatch.setattr(cli, "ForecastLedger", lambda p: nullcontext(ledger))
    monkeypatch.setattr(cli, "prospective_ledger_shadow_contract", lambda v: {})

    def enrich(actual, **kwargs):
        events.append("compose")
        assert kwargs["dataset"] is dataset and kwargs["benchmark"] is benchmark
        assert kwargs["cache_directory"] == artifacts.parent / "research-cache" / "additional-sources"
        if fail:
            raise RuntimeError("research incomplete")
        return {**actual, "research_complete": True}

    def publish(actual, *args, **kwargs):
        events.append("publish")
        assert actual["research_complete"] is True
        assert callable(kwargs["finalization"])
        return actual

    monkeypatch.setattr(cli, "compose_live_publication_research", enrich)
    monkeypatch.setattr(cli, "_publish_active_generation", publish)
    args = SimpleNamespace(profile="standard", contract="v5", config=tmp_path / "config", database=tmp_path / "db", output=output, artifacts=artifacts, from_env=True, expected_cutoff=datetime.fromisoformat(CUTOFF))
    if fail:
        with pytest.raises(RuntimeError, match="research incomplete"):
            cli.command_build(args)
        assert "publish" not in events
    else:
        assert cli.command_build(args) == 0
        assert events == ["source_check", "compose", "source_check", "publish"]
    assert output.read_bytes() == b"previous complete result"
    assert (artifacts / "marker").read_bytes() == b"previous complete artifacts"
