from copy import deepcopy
import hashlib
import json

import pytest

from regime_lab.forecast_enhancement_publication import (
    DECLARATION, DESTINATION, FILENAME, bind_document, declaration, encode,
    package_sidecar, source_contract_sha256, validate_binding, validate_packaged_sidecar,
)
from regime_lab.publication_contract import PublicContractError


@pytest.fixture
def source():
    payload = {"meta": {"generation_id": "a", "data_as_of": "2026-09-04T20:00:00+00:00", "publication_status": "unpublished"},
               "model": {"selection_status": "selected_by_gate", "lifecycle": {"selection": {"status": "selected_by_gate"}, "deployment": {"status": "candidate"}, "publication": {"status": "unpublished"}}}, "research": {}}
    document = {"schema_version": "regime-forecast-enhancements/1", "automatic_promotion": False,
                "source_generation_id": "a", "data_as_of": payload["meta"]["data_as_of"],
                "history": [], "latest": [], "model_metrics": [], "calibration": {"audit": []},
                "alerts": {"history": []}, "economics": {"incremental": {"history": []}}, "provenance": {}}
    forecast = {"model": "test", "horizon_weeks": 1, "origin_date": payload["meta"]["data_as_of"],
                "target_date": "2026-09-11T20:00:00+00:00", "current_state": "risk_on",
                "probabilities": {"risk_on": .8, "transition": .15, "risk_off": .05}}
    document["history"] = [forecast]
    document["latest"] = [deepcopy(forecast)]
    return payload, document


def test_binding_survives_review_lifecycle_but_rejects_generation_payload_and_content_changes(source):
    payload, document = source
    bound = bind_document(document, payload)
    reviewed = deepcopy(payload)
    reviewed["meta"].update(publication_status="reviewed_publication", generation_manifest_sha256="x", publication_review={"receipt": "x"})
    reviewed["model"]["lifecycle"].update(deployment={"status": "operating"}, publication={"status": "reviewed_publication"})
    assert source_contract_sha256(payload) == source_contract_sha256(reviewed)
    validate_binding(bound, reviewed)
    for mutation in ("generation", "payload", "content"):
        p, d = deepcopy(payload), deepcopy(bound)
        if mutation == "generation": p["meta"]["generation_id"] = "b"
        elif mutation == "payload": p["research"]["different_result"] = True
        else: d["latest"] = [{"changed": True}]
        with pytest.raises(PublicContractError): validate_binding(d, p)


def test_optional_absence_required_absence_and_explicit_omission_are_distinct(source, tmp_path):
    payload, _ = source
    path = tmp_path/"regime-results.json"
    files, meta = package_sidecar(payload, payload_raw=encode(payload), payload_path=path)
    assert meta == {"mode": "auto", "status": "omitted", "reason": "not_configured_for_this_generation"}
    validate_packaged_sidecar(files, meta, payload)
    with pytest.raises(PublicContractError, match="missing"):
        package_sidecar(payload, payload_raw=encode(payload), payload_path=path, mode="required")
    _, meta = package_sidecar(payload, payload_raw=encode(payload), payload_path=path, mode="omit")
    assert meta["reason"] == "explicitly_disabled"


def test_raw_import_requires_exact_source_receipt_and_package_hashes_are_checked(source, tmp_path):
    payload, document = source
    source_raw = encode(payload)
    document["provenance"] = {"input_sha256": {"payload": hashlib.sha256(source_raw).hexdigest()}}
    path = tmp_path/FILENAME
    path.write_bytes(encode(document))
    files, meta = package_sidecar(payload, payload_raw=source_raw, payload_path=tmp_path/"regime-results.json", mode="required")
    validate_packaged_sidecar(files, meta, payload)
    assert set(files) == {DESTINATION}
    with pytest.raises(PublicContractError, match="receipt"):
        package_sidecar(payload, payload_raw=source_raw+b" ", payload_path=tmp_path/"regime-results.json")
    with pytest.raises(PublicContractError): validate_packaged_sidecar({DESTINATION: files[DESTINATION]+b" "}, meta, payload)


def test_required_generation_declaration_cannot_be_omitted_or_redirected(source, tmp_path):
    payload, document = source
    bound = bind_document(document, payload)
    payload["research"][DECLARATION] = declaration(bound)
    validate_binding(bound, payload)
    with pytest.raises(PublicContractError, match="cannot omit"):
        package_sidecar(payload, payload_raw=encode(payload), payload_path=tmp_path/"regime-results.json", mode="omit")
    with pytest.raises(PublicContractError, match="missing"):
        package_sidecar(payload, payload_raw=encode(payload), payload_path=tmp_path/"regime-results.json")
    payload["research"][DECLARATION]["sha256"] = "0"*64
    with pytest.raises(PublicContractError, match="declaration"): validate_binding(bound, payload)


def test_a_sidecar_cannot_override_its_latest_history_row(source):
    payload, document = source
    document["latest"][0]["probabilities"] = {"risk_on": .1, "transition": .8, "risk_off": .1}
    with pytest.raises(PublicContractError, match="latest differs"):
        bind_document(document, payload)


def test_preview_companion_links_become_a_complete_json_download(source, tmp_path):
    payload, document = source
    raw = encode(payload)
    document["provenance"] = {"input_sha256": {"payload": hashlib.sha256(raw).hexdigest()},
                              "artifacts": [{"label": "private CSV", "url": "./data/research/private.csv"}]}
    (tmp_path/FILENAME).write_bytes(encode(document))
    files, meta = package_sidecar(payload, payload_raw=raw, payload_path=tmp_path/"regime-results.json")
    packed = json.loads(files[DESTINATION])
    assert "artifacts" not in packed["provenance"]
    assert packed["provenance"]["package_projection"]["omitted_companion_download_links"] == 1
    validate_packaged_sidecar(files, meta, payload)


@pytest.mark.parametrize("factory_fails", [False, True])
def test_weekly_factory_stages_required_evidence_before_atomic_cutover(source, tmp_path, monkeypatch, factory_fails):
    from types import SimpleNamespace
    import pandas as pd
    from regime_lab import cli
    from regime_lab.forecast_enhancement_publication import STATES_FILENAME, STATES_MANIFEST_FILENAME
    from regime_lab.operational_forecast import frame_sha256
    payload, document = source
    payload["meta"].update(result_version="weekly-regime-result-v5", mode="demo")
    states = pd.Series(["risk_on"], index=pd.DatetimeIndex([payload["meta"]["data_as_of"]], name="cutoff"))
    benchmark = SimpleNamespace(state_label_history=pd.DataFrame({"date": states.index, "state": states.to_numpy()}))
    document["provenance"].update(input_frames={"states": frame_sha256(states)},
                                   state_frame_schema={"series_name": None, "index_name": "cutoff"})
    output, artifacts = tmp_path/"result.json", tmp_path/"artifacts"
    output.write_bytes(b"old result"); artifacts.mkdir(); (artifacts/"old.txt").write_bytes(b"old artifacts")
    monkeypatch.setattr(cli, "validate_dashboard_payload", lambda *_: None)
    def supporting(_, directory, **kwargs):
        directory.mkdir(); (directory/"selection-family-audit.json").write_text("{}")
    monkeypatch.setattr(cli, "_write_supporting_results", supporting)
    for name in ("_verify_staged_evidence_artifacts", "_verify_staged_research_artifacts", "_verify_staged_core_artifacts", "_verify_staged_feature_quality_artifact"):
        monkeypatch.setattr(cli, name, lambda *_: None)
    def manifest(**kwargs):
        stage = kwargs["artifact_directory"]
        enhanced = json.loads((stage/FILENAME).read_text())
        assert kwargs["payload"]["research"][DECLARATION] == declaration(enhanced)
        assert frame_sha256(pd.read_pickle(stage/STATES_FILENAME)) == frame_sha256(states)
        assert json.loads((stage/STATES_MANIFEST_FILENAME).read_text())["frames"]["states"] == frame_sha256(states)
        assert FILENAME in (stage/"SHA256SUMS").read_text()
        return {}
    monkeypatch.setattr(cli, "build_generation_manifest", manifest)
    monkeypatch.setattr(cli, "bind_payload_to_generation_manifest", lambda value, _: value)
    monkeypatch.setattr(cli, "validate_generation_manifest", lambda *_, **__: {})
    monkeypatch.setattr(cli, "write_dashboard_payload", lambda value, path: path.write_bytes(encode(value)))
    def factory(candidate):
        candidate["factory_local_mutation"] = True
        if factory_fails: raise RuntimeError("comparison incomplete")
        return document
    if factory_fails:
        with pytest.raises(RuntimeError, match="comparison incomplete"):
            cli._publish_active_generation(payload, benchmark, output=output, artifacts=artifacts, forecast_enhancements_factory=factory)
        assert output.read_bytes() == b"old result" and (artifacts/"old.txt").read_bytes() == b"old artifacts"
    else:
        cli._publish_active_generation(payload, benchmark, output=output, artifacts=artifacts, forecast_enhancements_factory=factory)
        assert (artifacts/FILENAME).is_file() and not (artifacts/"old.txt").exists()
    assert "factory_local_mutation" not in payload
