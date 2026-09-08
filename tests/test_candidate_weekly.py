from copy import deepcopy
import json
import hashlib
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import math
import sqlite3

import pandas as pd
import pytest

from regime_lab import candidate_weekly as workflow
from regime_lab.operational_forecast import frame_sha256
from regime_lab.candidate_recipes import candidate_model_recipes


def bundle(weeks=1, start="2026-09-04 16:00"):
    dates = pd.date_range(start, periods=weeks, freq="W-FRI", tz="America/New_York").tz_convert("UTC")
    states = pd.Series(["risk_on"] * (weeks - 1) + ["transition"], index=dates)
    origin = dates[-1]
    payload = {"meta": {"generation_id": f"g{weeks}", "data_as_of": origin.isoformat()},
               "label": {"spec_sha256": "a" * 64},
               "weekly": [{"data_as_of": origin.isoformat(), "current": {"state": states.iloc[-1]}}]}
    document = {"schema_version": "regime-forecast-enhancements/1", "automatic_promotion": False,
                "source_generation_id": f"g{weeks}", "data_as_of": origin.isoformat(),
                "generated_at": (origin + pd.DateOffset(hours=1)).isoformat(), "protocol": {"version": "test"},
                "latest": [{"model": "candidate", "origin_date": origin.isoformat(),
                            "target_date": (origin.tz_convert("America/New_York") + pd.DateOffset(weeks=h)).tz_convert("UTC").isoformat(),
                            "target": "endpoint", "horizon_weeks": h, "current_state": states.iloc[-1],
                            "actual": None, "probabilities": {"risk_on": 0.7, "transition": 0.2, "risk_off": 0.1}}
                           for h in (1, 4, 13)]}
    document["provenance"] = {"model_cache_key": {"schema_version": "forecast-model-cache/2", "protocol": deepcopy(document["protocol"]), "source_sha256": {"model.py": "b" * 64}, "runtime": {"python": "test"}}}
    return {"payload": payload, "enhancement": document, "states": states,
            "state_manifest": {"frames": {"states": frame_sha256(states)}}}


def evidence_for(data):
    payload_raw = json.dumps(data["payload"]).encode()
    buffer = BytesIO()
    data["states"].to_pickle(buffer)
    states_raw = buffer.getvalue()
    hashes = {"payload": hashlib.sha256(payload_raw).hexdigest(), "states": hashlib.sha256(states_raw).hexdigest()}
    data["enhancement"].setdefault("provenance", {})["input_sha256"] = hashes
    enhancement_raw = json.dumps(data["enhancement"]).encode()
    manifest = {"status": "validated_local_research", "source_hashes": hashes,
                "files": {"forecast-enhancements.json": hashlib.sha256(enhancement_raw).hexdigest()}}
    return (*workflow.read_candidate_evidence(payload_raw, enhancement_raw, states_raw, manifest), manifest)


def run(directory, **data):
    data["evidence"] = evidence_for(data)[3]
    return workflow.run_weekly_candidates(directory, **data)


def clock(monkeypatch, at):
    monkeypatch.setattr(workflow, "_now", lambda: pd.Timestamp(at).to_pydatetime())


def test_two_week_issuance_scores_without_copied_origin_receipt(tmp_path, monkeypatch):
    first = bundle()
    clock(monkeypatch, "2026-09-07T23:00Z")  # Labor Day: next session is Tuesday.
    result = run(tmp_path / "candidate", **first)
    assert result["status"] == "issued_local_preview"
    assert result["latest"]["issue_deadline_at"] == "2026-09-08T13:30:00+00:00"
    assert result["pending_predictions"] == 3
    assert result["matured_predictions"] == 0
    second = bundle(2)
    second["states"].iloc[0] = "transition"
    second["state_manifest"]["frames"]["states"] = frame_sha256(second["states"])
    clock(monkeypatch, "2026-09-12T10:00Z")
    result = run(tmp_path / "candidate", **second)
    assert result["issued_packets"] == 2
    assert result["pending_predictions"] == 5
    assert result["matured_predictions"] == 1
    assert result["models"][0]["log_loss"] == pytest.approx(-math.log(0.2))
    assert result["models"][0]["brier"] == pytest.approx(1.14)
    assert result["models"][0]["n"] == 1
    issued_at = result["latest"]["issued_at"]
    # A newer candidate cannot replace an already issued origin.
    second["enhancement"]["latest"][0]["probabilities"] = {"risk_on": .2, "transition": .7, "risk_off": .1}
    rerun = run(tmp_path / "candidate", **second)
    assert rerun["status"] == "already_issued"
    assert rerun["issued_packets"] == 2
    assert rerun["latest"]["issued_at"] == issued_at
    assert rerun["latest"]["predictions"][0]["probabilities"]["risk_on"] == .7
    # First score is immutable when an outcome is revised.
    second["states"].iloc[-1] = "risk_off"
    second["payload"]["weekly"][-1]["current"]["state"] = "risk_off"
    for row in second["enhancement"]["latest"]:
        row["current_state"] = "risk_off"
    second["state_manifest"]["frames"]["states"] = frame_sha256(second["states"])
    revised = run(tmp_path / "candidate", **second)
    assert len(revised["revision_conflicts"]) == 1
    assert revised["models"] == result["models"]


def test_expired_snapshot_cannot_be_backdated_or_count_as_issued(tmp_path, monkeypatch):
    clock(monkeypatch, "2026-09-08T13:30Z")
    result = run(tmp_path / "candidate", **bundle())
    assert result["status"] == "deadline_missed"
    assert result["issued_packets"] == result["matured_predictions"] == 0
    assert result["latest"]["issued_at"] is None


@pytest.mark.parametrize("kind", ["generation", "manifest", "future", "prediction", "horizon", "resolved", "training", "duplicate"])
def test_invalid_generation_is_rejected_before_workspace_creation(tmp_path, monkeypatch, kind):
    data = bundle()
    clock(monkeypatch, "2026-09-05T10:00Z")
    row = data["enhancement"]["latest"][0]
    if kind == "generation": data["enhancement"]["source_generation_id"] = "old"
    if kind == "manifest": data["state_manifest"]["frames"]["states"] = "0" * 64
    if kind == "future": data["enhancement"]["generated_at"] = "2026-09-07T00:00Z"
    if kind == "prediction": row["probabilities"]["risk_on"] = .9
    if kind == "horizon": row["target_date"] = "2026-09-12T20:00Z"
    if kind == "resolved": row["actual"] = "transition"
    if kind == "training": row["last_train_target"] = row["origin_date"]
    if kind == "duplicate": data["enhancement"]["latest"].append(deepcopy(row))
    with pytest.raises(ValueError): run(tmp_path / "candidate", **data)
    assert not (tmp_path / "candidate").exists()


def test_all_horizons_mature_and_recipes_are_scored_separately(tmp_path, monkeypatch):
    clock(monkeypatch, "2026-09-05T10:00Z")
    run(tmp_path / "candidate", **bundle())
    later = bundle(14)
    later["enhancement"]["protocol"]["version"] = "second"
    later["enhancement"]["provenance"]["model_cache_key"]["protocol"]["version"] = "second"
    clock(monkeypatch, "2026-12-05T10:00Z")
    result = run(tmp_path / "candidate", **later)
    assert result["matured_predictions"] == 3
    assert {m["horizon_weeks"] for m in result["models"]} == {1, 4, 13}
    assert all(m["recipe_sha256"] != result["active_recipe_sha256"] for m in result["models"])


def test_missing_target_is_pending_and_tampering_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "candidate"
    clock(monkeypatch, "2026-09-05T10:00Z")
    run(path, **bundle())
    later = bundle(3)
    later["states"] = later["states"].drop(later["states"].index[1])
    later["state_manifest"]["frames"]["states"] = frame_sha256(later["states"])
    clock(monkeypatch, "2026-09-19T10:00Z")
    result = run(path, **later)
    assert result["overdue_predictions"] == 1
    assert result["matured_predictions"] == 0
    with sqlite3.connect(path / workflow.LEDGER_NAME) as db:
        db.execute("UPDATE packets SET content='{}'")
    with pytest.raises(ValueError, match="content changed"):
        run(path, **later)


def test_redirected_candidate_workspace_is_rejected(tmp_path, monkeypatch):
    clock(monkeypatch, "2026-09-05T10:00Z")
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        run(linked, **bundle())
    assert not list(target.iterdir())


def test_bad_run_receipt_and_missing_model_binding_cannot_issue(tmp_path, monkeypatch):
    clock(monkeypatch, "2026-09-05T10:00Z")
    data = bundle()
    _, _, _, evidence, manifest = evidence_for(data)
    with pytest.raises(ValueError, match="completed run"):
        workflow.read_candidate_evidence(b"{}", b"{}", b"bad pickle", manifest)
    del data["enhancement"]["provenance"]["model_cache_key"]
    with pytest.raises(ValueError, match="recipe binding"):
        run(tmp_path / "candidate", **data)
    assert not (tmp_path / "candidate").exists()


def test_source_change_rolls_back_packet_and_scores(tmp_path, monkeypatch):
    path = tmp_path / "candidate"
    clock(monkeypatch, "2026-09-05T10:00Z")
    def fail():
        raise RuntimeError("input changed")
    with pytest.raises(RuntimeError, match="input changed"):
        run(path, **bundle(), verify_inputs=fail)
    with sqlite3.connect(path / workflow.LEDGER_NAME) as db:
        assert db.execute("SELECT count(*) FROM packets").fetchone()[0] == 0
    run(path, **bundle())
    clock(monkeypatch, "2026-09-12T10:00Z")
    with pytest.raises(RuntimeError, match="input changed"):
        run(path, **bundle(2), verify_inputs=fail)
    with sqlite3.connect(path / workflow.LEDGER_NAME) as db:
        assert db.execute("SELECT count(*) FROM packets").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM scores").fetchone()[0] == 0


def test_concurrent_retries_freeze_one_packet_and_keep_summary_order(tmp_path, monkeypatch):
    path = tmp_path / "candidate"
    clock(monkeypatch, "2026-09-05T10:00Z")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(path, **bundle()), range(2)))
    assert {r["status"] for r in results} == {"issued_local_preview", "already_issued"}
    assert len({r["latest"]["issued_at"] for r in results}) == 1
    clock(monkeypatch, "2026-09-12T10:00Z")
    run(path, **bundle(2))
    with pytest.raises(ValueError, match="move backward"):
        run(path, **bundle())


def test_normal_weekly_generation_feeds_same_candidate_ledger(tmp_path, monkeypatch):
    from regime_lab.forecast_enhancement_publication import bind_document, declaration, DECLARATION, encode
    from regime_lab.integrity import canonical_json_sha256_v1_without_generation_binding
    data = bundle()
    payload, document = data["payload"], data["enhancement"]
    payload["meta"]["publication_status"] = "unpublished"
    payload["model"] = {"selection_status": "selected_by_gate", "lifecycle": {
        "selection": {"status": "selected_by_gate"}, "deployment": {"status": "candidate"},
        "publication": {"status": "unpublished"}}}
    payload["research"] = {}
    document.update(history=deepcopy(document["latest"]), model_metrics=[], calibration={"audit": []},
                    alerts={"history": []}, economics={"incremental": {"history": []}})
    document["provenance"]["input_frames"] = {"states": frame_sha256(data["states"])}
    bound = bind_document(document, payload)
    payload["research"][DECLARATION] = declaration(bound)
    manifest = {"schema_version": "regime-generation-manifest/2", "generation_id": "g1",
                "payload": {"payload_contract_sha256": canonical_json_sha256_v1_without_generation_binding(payload)}}
    payload["meta"]["generation_manifest_sha256"] = workflow.digest(manifest)
    buffer = BytesIO()
    data["states"].to_pickle(buffer)
    read = lambda record: workflow.read_candidate_evidence(encode(payload), encode(bound), buffer.getvalue(), record)
    payload, enhancement, states, evidence = read(manifest)
    assert evidence["evidence_type"] == "weekly_generation"
    clock(monkeypatch, "2026-09-05T10:00Z")
    result = workflow.run_weekly_candidates(tmp_path / "candidate", payload=payload, enhancement=enhancement,
                                          states=states, state_manifest=data["state_manifest"], evidence=evidence)
    assert result["pending_predictions"] == 3
    assert result["latest"]["evidence_receipt"]["evidence_type"] == "weekly_generation"
    with pytest.raises(ValueError, match="generation manifest"):
        read({**manifest, "generation_id": "g0"})


def versioned_bundle(weeks=1):
    data = bundle(weeks)
    payload, document = data["payload"], data["enhancement"]
    manifest = {"models": ["causal_dynamic_ensemble"], "half_life": 52}
    payload["model"] = {"candidate_manifest": manifest, "candidate_manifest_sha256": workflow.digest(manifest),
                        "feature_manifest_sha256": "f"*64}
    payload["research"] = {
        "forecast_improvement": {"provenance": {"code_sha256": "boundary-v1", "runtime": {"numpy": "test"}}},
        "forecast_research": {"protocol": {"boundary": {"window": 520}, "hazard": {"window": 520}},
                              "publication_provenance": {"recipe_sha256": "hazard-v1"}}}
    document["provenance"]["operating_model_recipe"] = {
        "schema_version": "regime-operating-numerical-recipe/1", "source_sha256": {"structural_models.py": "v1"},
        "runtime": {"python": "test"}, "candidate_manifest_sha256": workflow.digest(manifest),
        "effective_config_sha256": "c"*64, "feature_manifest_sha256": "f"*64}
    base = document["latest"]
    document["latest"] = [{**row, "model": model} for model in
                         ("causal_dynamic_ensemble", "boundary_filtered_history", "directional_duration_hazard", "evolving_boundary_gjr_skewt")
                         for row in base]
    return data


@pytest.mark.parametrize("change,affected", [
    ("operating_manifest", "causal_dynamic_ensemble"), ("operating_code", "causal_dynamic_ensemble"),
    ("operating_config", "causal_dynamic_ensemble"), ("operating_features", "causal_dynamic_ensemble"),
    ("boundary", "boundary_filtered_history"), ("hazard", "directional_duration_hazard"),
    ("new_code", "evolving_boundary_gjr_skewt")])
def test_effective_versions_change_only_their_model(change, affected):
    data = versioned_bundle()
    payload, document = data["payload"], data["enhancement"]
    before = candidate_model_recipes(payload, document)
    if change == "operating_manifest":
        payload["model"]["candidate_manifest"]["half_life"] = 26
        value = workflow.digest(payload["model"]["candidate_manifest"])
        payload["model"]["candidate_manifest_sha256"] = value
        document["provenance"]["operating_model_recipe"]["candidate_manifest_sha256"] = value
    elif change == "operating_code":
        document["provenance"]["operating_model_recipe"]["source_sha256"]["structural_models.py"] = "v2"
    elif change == "operating_config":
        document["provenance"]["operating_model_recipe"]["effective_config_sha256"] = "d"*64
    elif change == "operating_features":
        payload["model"]["feature_manifest_sha256"] = "e"*64
        document["provenance"]["operating_model_recipe"]["feature_manifest_sha256"] = "e"*64
    elif change == "boundary":
        payload["research"]["forecast_improvement"]["provenance"]["code_sha256"] = "v2"
    elif change == "hazard":
        payload["research"]["forecast_research"]["publication_provenance"]["recipe_sha256"] = "v2"
    else:
        document["provenance"]["model_cache_key"]["source_sha256"]["model.py"] = "v2"
    after = candidate_model_recipes(payload, document)
    assert {name for name in before if before[name] != after[name]} == {affected}


def test_incomplete_historical_config_binding_remains_generation_specific():
    first,second=versioned_bundle(),versioned_bundle(2)
    for data in (first,second):
        del data["enhancement"]["provenance"]["operating_model_recipe"]["effective_config_sha256"]
    a,b=(candidate_model_recipes(d["payload"],d["enhancement"]) for d in (first,second))
    assert a["causal_dynamic_ensemble"]!=b["causal_dynamic_ensemble"]
    assert a["evolving_boundary_gjr_skewt"]==b["evolving_boundary_gjr_skewt"]


def test_weekly_inputs_and_alert_changes_do_not_split_numerical_versions():
    first, second = versioned_bundle(), versioned_bundle(2)
    second["enhancement"]["provenance"]["model_cache_key"]["inputs"] = {"states": "new-week"}
    second["enhancement"]["provenance"]["model_cache_key"]["protocol"]["alert_annual_budget"] = 3
    assert candidate_model_recipes(first["payload"], first["enhancement"]) == candidate_model_recipes(second["payload"], second["enhancement"])
    for data in (first, second):
        del data["enhancement"]["provenance"]["operating_model_recipe"]
    a, b = (candidate_model_recipes(d["payload"], d["enhancement"]) for d in (first, second))
    assert a["causal_dynamic_ensemble"] != b["causal_dynamic_ensemble"]
    assert a["evolving_boundary_gjr_skewt"] == b["evolving_boundary_gjr_skewt"]


def test_model_scores_pool_same_version_and_separate_changed_code(tmp_path, monkeypatch):
    path = tmp_path / "candidate"
    clock(monkeypatch, "2026-09-05T10:00Z")
    first = run(path, **versioned_bundle())
    clock(monkeypatch, "2026-09-12T10:00Z")
    second = versioned_bundle(2)
    second["enhancement"]["provenance"]["operating_model_recipe"]["source_sha256"]["structural_models.py"] = "v2"
    run(path, **second)
    clock(monkeypatch, "2026-09-19T10:00Z")
    result = run(path, **versioned_bundle(3))
    one = [row for row in result["models"] if row["horizon_weeks"] == 1]
    assert sorted(row["n"] for row in one if row["model"] == "causal_dynamic_ensemble") == [1, 1]
    assert [row["n"] for row in one if row["model"] == "evolving_boundary_gjr_skewt"] == [2]
    assert result["latest"]["predictions"][0]["recipe_sha256"]
    with sqlite3.connect(path / workflow.LEDGER_NAME) as db:
        packet = json.loads(db.execute("SELECT content FROM packets ORDER BY origin LIMIT 1").fetchone()[0])
    assert packet["issued_at"] == first["latest"]["issued_at"]
    assert packet["predictions"] == first["latest"]["predictions"]


def test_legacy_packet_stays_immutable_and_is_not_pooled_with_current_version(tmp_path, monkeypatch):
    path = tmp_path / "candidate"
    clock(monkeypatch, "2026-09-05T10:00Z")
    run(path, **bundle())
    with sqlite3.connect(path / workflow.LEDGER_NAME) as db:
        packet = json.loads(db.execute("SELECT content FROM packets").fetchone()[0])
        for row in packet["predictions"]:
            del row["recipe_sha256"]
        raw = workflow._encoded(packet)
        db.execute("UPDATE packets SET content=?, sha256=?", (raw, workflow.digest(packet)))
    clock(monkeypatch, "2026-09-12T10:00Z")
    result = run(path, **bundle(2))
    assert result["models"][0]["recipe_sha256"] != result["active_model_recipe_sha256"]["candidate"]
    with sqlite3.connect(path / workflow.LEDGER_NAME) as db:
        assert db.execute("SELECT content FROM packets ORDER BY origin LIMIT 1").fetchone()[0] == raw
