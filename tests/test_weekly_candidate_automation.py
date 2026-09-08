"""Normal generation files feed a real isolated candidate ledger across weeks."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from regime_lab import automation, candidate_weekly
from regime_lab.forecast_enhancement_publication import (
    DECLARATION, FILENAME, STATES_FILENAME, STATES_MANIFEST_FILENAME,
    bind_document, declaration, encode,
)
from regime_lab.integrity import canonical_json_sha256_v1, canonical_json_sha256_v1_without_generation_binding
from regime_lab.operational_forecast import frame_sha256
from test_weekly_automation import _settings


def weekly_files(settings, weeks):
    dates = pd.date_range("2026-09-04 16:00", periods=weeks, freq="W-FRI", tz="America/New_York").tz_convert("UTC")
    states = pd.Series(["transition"] * weeks, index=dates)
    origin = dates[-1]
    payload = {"meta": {"generation_id": f"g{weeks}", "data_as_of": origin.isoformat(), "publication_status": "unpublished"},
               "model": {"selection_status": "selected_by_gate", "lifecycle": {
                   "selection": {"status": "selected_by_gate"}, "deployment": {"status": "candidate"},
                   "publication": {"status": "unpublished"}}},
               "label": {"spec_sha256": "a" * 64}, "research": {},
               "weekly": [{"data_as_of": origin.isoformat(), "current": {"state": "transition"}}]}
    protocol = {"version": "test"}
    rows = [{"model": "candidate", "origin_date": origin.isoformat(), "horizon_weeks": h,
             "target_date": (origin.tz_convert("America/New_York") + pd.DateOffset(weeks=h)).tz_convert("UTC").isoformat(),
             "target": "endpoint", "current_state": "transition", "actual": None,
             "probabilities": {"risk_on": .7, "transition": .2, "risk_off": .1}} for h in (1, 4, 13)]
    document = {"schema_version": "regime-forecast-enhancements/1", "automatic_promotion": False,
                "source_generation_id": f"g{weeks}", "data_as_of": origin.isoformat(),
                "generated_at": (origin + pd.DateOffset(hours=1)).isoformat(), "protocol": protocol,
                "history": rows, "latest": deepcopy(rows), "model_metrics": [],
                "calibration": {"audit": []}, "alerts": {"history": []},
                "economics": {"incremental": {"history": []}},
                "provenance": {"input_frames": {"states": frame_sha256(states)}, "model_cache_key": {
                    "schema_version": "forecast-model-cache/2", "protocol": protocol,
                    "source_sha256": {"model.py": "b" * 64}, "runtime": {"python": "test"}}}}
    bound = bind_document(document, payload)
    payload["research"][DECLARATION] = declaration(bound)
    manifest = {"schema_version": "regime-generation-manifest/2", "generation_id": f"g{weeks}",
                "payload": {"payload_contract_sha256": canonical_json_sha256_v1_without_generation_binding(payload)}}
    payload["meta"]["generation_manifest_sha256"] = canonical_json_sha256_v1(manifest)
    settings.candidate_path.parent.mkdir(parents=True, exist_ok=True)
    settings.artifacts.mkdir(parents=True, exist_ok=True)
    settings.candidate_path.write_bytes(encode(payload))
    settings.candidate_path.with_name(FILENAME).write_bytes(encode(bound))
    settings.candidate_generation_manifest_path.write_bytes(encode(manifest))
    states.to_pickle(settings.artifacts / STATES_FILENAME)
    (settings.artifacts / STATES_MANIFEST_FILENAME).write_bytes(encode({"frames": {"states": frame_sha256(states)}}))
    paths = (settings.candidate_path, settings.candidate_path.with_name(FILENAME), settings.candidate_generation_manifest_path,
             settings.artifacts / STATES_FILENAME, settings.artifacts / STATES_MANIFEST_FILENAME)
    return {p: p.read_bytes() for p in paths}


def test_enabled_automation_records_two_weeks_and_retry_preserves_first_packet(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path), contract="v5", forecast_enhancements_research=True, forecast_candidates_weekly=True)
    originals = weekly_files(settings, 1)
    monkeypatch.setattr(candidate_weekly, "_now", lambda: pd.Timestamp("2026-09-05T10:00Z").to_pydatetime())
    first = automation._record_weekly_forecast_candidates(settings)
    assert first["issued_packets"] == 1 and first["pending_predictions"] == 3 and first["matured_predictions"] == 0
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    originals = weekly_files(settings, 2)
    monkeypatch.setattr(candidate_weekly, "_now", lambda: pd.Timestamp("2026-09-12T10:00Z").to_pydatetime())
    second = automation._record_weekly_forecast_candidates(settings)
    assert second["issued_packets"] == 2 and second["pending_predictions"] == 5 and second["matured_predictions"] == 1
    assert second["models"][0]["log_loss"] == pytest.approx(-math.log(.2))
    again = automation._record_weekly_forecast_candidates(settings)
    assert again["status"] == "already_issued"
    assert again["issued_packets"] == 2 and again["latest"]["issued_at"] == second["latest"]["issued_at"]
    assert again["models"] == second["models"]
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    workspace = settings.state_directory / "forecast-candidates"
    assert json.loads((workspace / "candidate-summary.json").read_text())["matured_predictions"] == 1
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.glob("*") if p.is_file() and p.suffix != ".lock"}
    settings.candidate_path.with_name(FILENAME).write_text("{}")
    with pytest.raises(automation.AutomationError, match="weekly candidate issuance/evaluation failed"):
        automation._record_weekly_forecast_candidates(settings)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.glob("*") if p.is_file() and p.suffix != ".lock"}


def test_default_does_not_create_or_touch_candidate_ledger(tmp_path):
    settings = _settings(tmp_path)
    assert automation._record_weekly_forecast_candidates(settings) is None
    assert not (settings.state_directory / "forecast-candidates").exists()
    with pytest.raises(automation.AutomationError, match="require V5"):
        automation._record_weekly_forecast_candidates(replace(settings, forecast_candidates_weekly=True))


@pytest.mark.parametrize("candidate,enhancement,valid", [(True, True, True), (False, False, True), (True, False, False), ("true", True, False)])
def test_opt_in_settings_require_an_explicit_boolean_and_current_comparisons(tmp_path, monkeypatch, candidate, enhancement, valid):
    config = json.loads((Path(__file__).resolve().parents[1] / "config/automation.json").read_text())
    config["build"].update(forecast_candidates_weekly=candidate, forecast_enhancements_research=enhancement)
    path = tmp_path / "automation.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(automation, "project_root", lambda: tmp_path)
    if not valid:
        with pytest.raises(automation.AutomationError): automation.AutomationSettings.load(path)
    else:
        settings = automation.AutomationSettings.load(path)
        assert settings.forecast_candidates_weekly is candidate


def test_candidate_record_failure_uses_cached_resume_policy(tmp_path):
    settings = _settings(tmp_path)
    result = automation._failure_policy(automation.AutomationError("weekly candidate issuance/evaluation failed: temporary IO"),
                                        stage="record_forecast_candidates", attempt_started_at=pd.Timestamp("2026-09-06T10:00Z").to_pydatetime(), settings=settings)
    assert result == ("record_forecast_candidates_failed", "resume", None)
