"""Generation binding, missing futures, immutable source clocks and cache failures."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.research import forecast_publication as publication
from regime_lab.research import forecast_new_information as information
from regime_lab.research.forecast_information_study import run_information_study


def generation(*, shift=0):
    dates = pd.date_range("2026-04-24T20:00:00Z", periods=20, freq="W-FRI") + pd.Timedelta(weeks=shift)
    canonical = pd.DataFrame({"spy_close": range(100, 120)}, index=dates)
    states = pd.Series("risk_on", index=dates)
    cutoff = dates[-1].isoformat()
    payload = {"meta": {"generation_id": f"week-{shift}", "data_as_of": cutoff},
               "model": {"transition_selection_end": "2023-01-01"},
               "weekly": [{"data_as_of": cutoff, "current": {"state": "risk_on"}}]}
    baseline = pd.DataFrame([{"model": model, "origin_date": origin, "target_date": target,
        "current_state": "risk_on", "actual": "risk_on", "evaluation_split": "holdout",
        "p_risk_on": .7, "p_transition": .2, "p_risk_off": .1}
        for model in publication.audit.BASELINES for origin, target in zip(dates[-4:-1], dates[-3:])])
    resolved, future = [], []
    for horizon in (1, 4, 13):
        for position, origin in enumerate(dates):
            target = dates[position + horizon] if position + horizon < len(dates) else dates[-1] + pd.Timedelta(weeks=position + horizon - len(dates) + 1)
            row = {"model": "calibration_test", "horizon": horizon, "origin_date": origin,
                "target_end": target, "current_state": "risk_on", "raw_p_change": .3,
                "p_change": .2, "actual_change": False if target <= dates[-1] else None,
                "evaluation_split": "retrospective_diagnostic" if target <= dates[-1] else "prospective"}
            (resolved if target <= dates[-1] else future).append(row)
    return payload, canonical, states, baseline, pd.DataFrame(resolved), pd.DataFrame(future)


@pytest.fixture
def studies(monkeypatch):
    calls = {"model": 0, "calibration": 0, "information": []}
    def model(canonical, states, baseline, **kwargs):
        calls["model"] += 1
        reference = baseline.loc[baseline.model.eq(publication.audit.BASELINES[0])]
        rows = [{"origin_date": row.origin_date.isoformat(), "current_state": row.current_state,
                 "next_state": {"risk_on": .7, "transition": .2, "risk_off": .1}, "horizons": {}}
                for row in reference.itertuples()]
        models = []
        for name in (*publication.audit.BASELINES, publication.audit.BOUNDARY_BASELINE,
                     publication.audit.ASYMMETRIC_MODEL, publication.audit.PATH_MODEL,
                     publication.audit.MARKOV_PATH_MODEL):
            item = {"id": name, "label": name, "history": deepcopy(rows)}
            if name not in publication.audit.BASELINES:
                item["latest"] = {**deepcopy(rows[-1]), "origin_date": states.index[-1].isoformat()}
            if name in {publication.audit.PATH_MODEL, publication.audit.MARKOV_PATH_MODEL}:
                from regime_lab.analysis.forecast_paths import project_multistate_paths
                for row in [*item["history"], item["latest"]]:
                    paths = project_multistate_paths(lambda state, age: [.7, .2, .1], "risk_on", 1)
                    for path in paths:
                        path["target_date"] = (pd.Timestamp(row["origin_date"]) + pd.Timedelta(days=path["horizon_weeks"] * 7)).isoformat()
                    row["horizons"] = {f"{p['horizon_weeks']}w": p for p in paths}
            models.append(item)
        return SimpleNamespace(document={"schema_version": "regime-forecast-research/1",
            "data_as_of": states.index[-1].isoformat(), "models": models,
            "selected_model": publication.audit.PATH_MODEL, "automatic_promotion": False})
    def calibration(payload, resolved, future, **kwargs):
        calls["calibration"] += 1
        return {"schema_version": "regime-calibration-audit/1", "data_as_of": payload["meta"]["data_as_of"],
            "rows": [{"model": model, "horizon_weeks": int(horizon), "evaluation_split": split,
                      "n_predictions": len(part), "raw_log_loss": .3, "calibrated_log_loss": .2}
                     for (model, horizon, split), part in resolved.groupby(["model", "horizon", "evaluation_split"])],
            "latest_rows": [{"model": model, "horizon_weeks": int(horizon)} for horizon, model in set(zip(resolved.horizon, resolved.model))],
            "sources": kwargs["sources"]}
    def info(history, *, data_as_of, output, sources_dir, offline, refresh=False, **kwargs):
        calls["information"].append({"cutoff": data_as_of, "offline": offline, "refresh": refresh})
        output.mkdir(parents=True, exist_ok=True)
        sources_dir.mkdir(exist_ok=True)
        if not offline:
            raw = f"DATE,CLOSE\n09/03/2026,{len(calls['information']) + 20}\n".encode()
            seen = pd.Timestamp(data_as_of) + pd.Timedelta(days=2)
            snap = information.Snapshot("vix3m", information.SOURCES["vix3m"], hashlib.sha256(raw).hexdigest(), seen, raw)
            (sources_dir / (snap.sha256 + ".blob")).write_bytes(raw)
            (sources_dir / f"vix3m-{seen.strftime('%Y%m%dT%H%M%S%fZ')}.json").write_text(json.dumps(snap.manifest()))
        summary = {"schema_version": information.SCHEMA_VERSION, "data_as_of": data_as_of,
            "runtime_as_of": (pd.Timestamp(data_as_of) + pd.Timedelta(days=2)).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(), "experiments": {},
            "blocks": {"vix3m": {"status": "unavailable", "reason": "source at /Users/private/file failed"}},
            "sources": {"vix3m": {"status": "unavailable"}}}
        (output / "summary.json").write_text(json.dumps(summary))
        return summary
    monkeypatch.setattr(publication.audit, "run_audit_research", model)
    monkeypatch.setattr(publication.audit, "forecast_research_extension", lambda document: document)
    monkeypatch.setattr(publication, "build_calibration_audit_from_frames", calibration)
    monkeypatch.setattr(publication, "run_information_study", info)
    return calls


def build(values, cache, **kwargs):
    return publication.build_forecast_publication_research(*values, cache, **kwargs)


def test_build_is_complete_public_safe_and_retry_does_not_refit_or_refetch(tmp_path, studies):
    values = generation()
    original = deepcopy(values)
    first = build(values, tmp_path)
    assert set(first["blocks"]) == set(publication.BLOCKS)
    info = first["blocks"]["forecast_information"]
    assert info["status"] == "not_evaluable" and info["rows"] == []
    assert info["blocks"]["vix3m"]["reason"] == "optional_source_unavailable"
    assert "/Users/" not in json.dumps(first["blocks"])
    assert build(values, tmp_path) == first
    assert studies["model"] == studies["calibration"] == 1
    assert len(studies["information"]) == 1
    assert values[0] == original[0]
    for before, after in zip(original[1:], values[1:]):
        assert before.equals(after)


def test_new_cutoff_refreshes_sources_and_keeps_previous_first_seen(tmp_path, studies):
    first = build(generation(), tmp_path)
    old_versions = {p.name: p.read_bytes() for p in (tmp_path / "information-source-versions").glob("*.json")}
    second = build(generation(shift=1), tmp_path)
    assert first["provenance"]["cache_key"] != second["provenance"]["cache_key"]
    assert len(studies["information"]) == 2
    assert all(row["refresh"] and not row["offline"] for row in studies["information"])
    for name, raw in old_versions.items():
        assert (tmp_path / "information-source-versions" / name).read_bytes() == raw
    assert len(information.load_snapshots(tmp_path / "information-source-versions", "vix3m")) == 2
    assert len(list((tmp_path / "information-inputs").iterdir())) == 2


@pytest.mark.parametrize("damage", ["missing_latest", "missing_future", "wrong_target", "wrong_state", "wrong_actual", "unresolved_actual", "bad_probability", "bad_split", "cutoff"])
def test_invalid_generation_stops_before_studies_or_cache(tmp_path, studies, damage):
    values = list(generation())
    resolved, future = values[4:]
    if damage == "missing_latest":
        values[5] = future.loc[~(future.horizon.eq(13) & future.origin_date.eq(values[2].index[-1]))]
    elif damage == "missing_future":
        values[5] = future.drop(future.loc[future.horizon.eq(13)].index[0])
    elif damage == "wrong_target":
        values[5].loc[0, "target_end"] += pd.Timedelta(days=1)
    elif damage == "wrong_state":
        values[3].loc[0, "current_state"] = "risk_off"
    elif damage == "wrong_actual":
        values[4].loc[0, "actual_change"] = True
    elif damage == "unresolved_actual":
        values[5].loc[0, "actual_change"] = False
    elif damage == "bad_probability":
        values[4].loc[0, "p_change"] = float("nan")
    elif damage == "bad_split":
        values[4].loc[0, "evaluation_split"] = "selection"
    else:
        values[0]["meta"]["data_as_of"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        build(values, tmp_path / "cache")
    assert studies["model"] == 0 and not (tmp_path / "cache").exists()


@pytest.mark.parametrize("damage", ["delete_block", "tamper_block", "tamper_source", "partial_cache"])
def test_corrupt_cache_never_silently_rebuilds_or_returns_previous_cutoff(tmp_path, studies, damage):
    result = build(generation(), tmp_path)
    cache = tmp_path / "results" / result["provenance"]["cache_key"]
    if damage in {"delete_block", "partial_cache"}:
        (cache / ("forecast_information.json" if damage == "delete_block" else "receipt.json")).unlink()
    elif damage == "tamper_block":
        path = cache / "calibration_audit.json"
        document = json.loads(path.read_text()); document["rows"][0]["n_predictions"] = 999
        path.write_text(json.dumps(document))
    else:
        next((tmp_path / "information-inputs").rglob("*.blob")).write_bytes(b"corrupt")
    with pytest.raises((ValueError, FileNotFoundError)):
        build(generation(), tmp_path)
    assert studies["model"] == 1 and len(studies["information"]) == 1


def test_failed_required_study_preserves_existing_complete_cache(tmp_path, studies, monkeypatch):
    first = build(generation(), tmp_path)
    cache = tmp_path / "results" / first["provenance"]["cache_key"]
    before = {p.name: p.read_bytes() for p in cache.iterdir()}
    monkeypatch.setattr(publication.audit, "run_audit_research", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        build(generation(shift=1), tmp_path)
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == before
    assert len(list((tmp_path / "results").iterdir())) == 1


def test_recipe_change_invalidates_result_but_reuses_frozen_source_clock(tmp_path, studies, monkeypatch):
    first = build(generation(), tmp_path)
    original_recipe = publication._recipe()
    monkeypatch.setattr(publication, "_recipe", lambda: {**original_recipe, "test_version": 2})
    second = build(generation(), tmp_path)
    assert first["provenance"]["cache_key"] != second["provenance"]["cache_key"]
    assert first["provenance"]["information_snapshot_sha256"] == second["provenance"]["information_snapshot_sha256"]
    assert studies["information"][-1]["offline"] is True
    assert studies["information"][-1]["refresh"] is False


def test_information_library_frozen_year_and_no_backdated_optional_data(tmp_path, monkeypatch):
    values = generation()
    history = values[3].loc[values[3].model.eq(publication.audit.BASELINES[0])]
    attempted = []
    def load(directory, source_id):
        attempted.append(source_id)
        return []
    monkeypatch.setattr(information, "load_snapshots", load)
    monkeypatch.setattr(information.requests, "get", lambda *a, **k: pytest.fail("offline requested network"))
    config = publication._recipe()["information_protocol"]
    summary = run_information_study(history, data_as_of=values[0]["meta"]["data_as_of"], config=config,
        output=tmp_path / "output", sources_dir=tmp_path / "sources", offline=True,
        as_of="2025-12-31T12:00:00Z")
    assert {"tff_2024", "tff_2025"} <= set(attempted) and "tff_2026" not in attempted
    assert not summary["experiments"]
    assert all(row["status"] == "unavailable" for row in summary["blocks"].values())


@pytest.mark.parametrize("stored_version", ["legacy", "transition-calibration/2"])
def test_frame_calibration_accepts_archived_v1_and_normal_v2_without_changing_forecast(stored_version):
    from regime_lab.research.forecast_calibration import build_calibration_audit_from_frames
    from regime_lab.analysis.causal_calibration import TransitionCalibrator
    dates = pd.date_range("2024-01-05T21:00:00Z", periods=70, freq="W-FRI")
    resolved, future = [], []
    raw = {1: .3, 4: .4, 13: .5}
    stored = raw if stored_version.endswith("/2") else {1: .35, 4: .45, 13: .55}
    for horizon in (1, 4, 13):
        for i, origin in enumerate(dates):
            row = {"model": "calibration_test", "horizon": horizon, "origin_date": origin,
                "target_end": origin + pd.Timedelta(days=7 * horizon), "raw_p_change": raw[horizon],
                "p_change": stored[horizon], "actual_change": False if i + horizon < len(dates) else None,
                "evaluation_split": "retrospective_diagnostic" if i + horizon < len(dates) else "prospective"}
            if stored_version.endswith("/2"):
                row["calibration_version"] = stored_version
            (resolved if i + horizon < len(dates) else future).append(row)
    payload = {"meta": {"data_as_of": dates[-1].isoformat(), "generation_id": "calibration-fixture"},
        "model": {"transition_selection_end": "2023-01-01"}, "weekly": []}
    for origin in dates:
        payload["weekly"].append({"data_as_of": origin.isoformat(), "transition_risk": {
            f"{h}w": {"model": "official_anchor" if h == 1 else "calibration_test",
                "probability": .2 if h == 1 else stored[h],
                "target_end": (origin + pd.Timedelta(days=7 * h)).date().isoformat()}
            for h in (1, 4, 13)}})
    before = deepcopy(payload)
    report = build_calibration_audit_from_frames(payload, pd.DataFrame(resolved), pd.DataFrame(future))
    assert payload == before
    assert report["verification"]["one_week_anchor_unchanged"] is True
    assert report["verification"]["coherence_violations"] == 0
    assert report["calibration_version"] == "transition-calibration/2"
    for row in report["rows"]:
        assert row["calibrated_log_loss"] == pytest.approx(row["raw_log_loss"])
        if stored_version.endswith("/2"):
            assert row["calibrated_log_loss"] == pytest.approx(row["previous_calibrated_log_loss"])
        else:
            assert row["calibrated_log_loss"] < row["previous_calibrated_log_loss"]


@pytest.mark.parametrize("damage", ["missing_paths", "missing_latest", "missing_model"])
def test_required_path_model_outputs_cannot_disappear(tmp_path, studies, monkeypatch, damage):
    original = publication.audit.run_audit_research
    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        model = next(m for m in result.document["models"] if m["id"] == publication.audit.PATH_MODEL)
        if damage == "missing_paths":
            model["latest"]["horizons"] = {}
        elif damage == "missing_latest":
            del model["latest"]
        else:
            result.document["models"].remove(model)
            result.document["selected_model"] = publication.audit.BOUNDARY_BASELINE
        return result
    monkeypatch.setattr(publication.audit, "run_audit_research", broken)
    with pytest.raises(ValueError):
        build(generation(), tmp_path)
    assert not (tmp_path / "results").exists()
