import hashlib
import json

import pandas as pd
import pytest

from regime_lab.dataset import evidence_drivers
from regime_lab.research.forecast_new_information import Snapshot
from regime_lab.research.forecast_macro_information import align_features, parse_inflation, parse_reserves


def snap(source, raw):
    return Snapshot(source, "https://www.newyorkfed.org/example", hashlib.sha256(raw).hexdigest(),
                    pd.Timestamp("2026-09-07T00:00:00Z"), raw)


def test_grouping_precedes_display_limit():
    at = pd.Timestamp("2026-09-04T20:00:00Z")
    row = {f"dgs{n}__z_52w": 20-n for n in range(1, 10)}
    row.update({"icsa__z_52w": 4, "ccsa__z_52w": 3, "indpro__z_52w": 2})
    drivers = evidence_drivers(pd.DataFrame([row], index=[at]), at, limit=3)
    assert [d["feature"] for d in drivers] == ["dgs1__z_52w", "icsa__z_52w", "indpro__z_52w"]
    assert drivers[1]["direction"] == "risk_off"
    assert evidence_drivers(pd.DataFrame([row], index=[at]), at, limit=0) == []


def test_inflation_archive_excludes_actual_and_handles_leap_day():
    chart = {"chart": {"subcaption": "2024-2"}, "categories": [{"category": [
        {"label": "02/29"}, {"label": "Release", "vline": "true"}, {"label": "03/01"}]}],
        "dataset": [{"seriesname": "CPI Inflation", "data": [{"value": ".2"}, {"value": ".3"}]},
                    {"seriesname": "Actual CPI Inflation", "data": [{"value": "9"}, {"value": "9"}]}]}
    result = parse_inflation(snap("cleveland_inflation", json.dumps([chart]).encode()))
    assert result.value.tolist() == [.2, .3]
    origins = pd.DatetimeIndex(["2024-02-29T21:00:00Z", "2024-03-01T21:00:00Z"])
    assert align_features(result, origins).isna().all().all()
    reconstructed = align_features(result, origins, reconstructed=True)
    assert pd.isna(reconstructed.iloc[0, 0])
    assert reconstructed.iloc[1, 0] == .2


def test_reserve_lag_never_backdates_or_delays_first_seen():
    raw = ("Date,Elasticity - 50th percentile (main),Elasticity - 2.5th percentile,Elasticity - 97.5th percentile\n"
           "9/1/2026,-.3,-.7,.1\n").encode()
    records = parse_reserves(snap("reserve_elasticity", raw))
    current = align_features(records, pd.DatetimeIndex(["2026-09-07T00:00:00Z"]))
    assert current.reserve_elasticity.iloc[0] == -.3
    assert align_features(records, pd.DatetimeIndex(["2026-09-04T20:00:00Z"])).isna().all().all()
    assert align_features(records, current.index, reconstructed=True).isna().all().all()
    with pytest.raises(ValueError):
        align_features(records, pd.DatetimeIndex(["2026-09-07"]))


def test_future_observation_is_never_a_current_first_seen_value():
    raw = ("Date,Elasticity - 50th percentile (main),Elasticity - 2.5th percentile,Elasticity - 97.5th percentile\n"
           "9/8/2026,-.3,-.7,.1\n").encode()
    with pytest.raises(ValueError, match="future"):
        parse_reserves(snap("reserve_elasticity", raw))
    raw = raw.replace(b"9/8/2026", b"9/1/2026")
    records = parse_reserves(snap("reserve_elasticity", raw))
    records["observed_at"] = pd.Timestamp("2026-09-08T00:00:00Z")
    assert align_features(records, pd.DatetimeIndex(["2026-09-07T00:00:00Z"])).isna().all().all()


def test_cached_pointer_cannot_backdate_the_original_receipt(tmp_path, monkeypatch):
    from regime_lab.research import forecast_macro_information as module
    source = "reserve_elasticity"
    raw = b"original snapshot"
    snapshot = Snapshot(source, module.SOURCES[source]["download"], hashlib.sha256(raw).hexdigest(),
        pd.Timestamp("2026-09-01T00:00:00Z"), raw)
    manifest = snapshot.manifest()
    receipt = tmp_path/f"{snapshot.sha256}.json"
    receipt.write_text(json.dumps(manifest))
    (tmp_path/f"{snapshot.sha256}.blob").write_bytes(raw)
    pointer = tmp_path/f"{source}.json"
    pointer.write_text(json.dumps(manifest))
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("cached audit must not access network"))
    assert module.snapshot(source, tmp_path).retrieved_at == snapshot.retrieved_at
    before = receipt.read_bytes()
    manifest["retrieved_at"] = "2025-01-01T00:00:00Z"
    pointer.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="first-seen receipt"):
        module.snapshot(source, tmp_path)
    assert receipt.read_bytes() == before


def test_failed_run_preserves_last_summary_and_derived_artifacts(tmp_path, monkeypatch):
    from regime_lab.research import forecast_macro_information as module
    (tmp_path/"additional-information.json").write_text('{"previous":true}')
    (tmp_path/"source-predictions.csv").write_text('old complete predictions')
    monkeypatch.setattr(module, "snapshot", lambda source, directory, refresh=False: snap(source, b"stub"))
    def fail(payload, controls, staging, snapshots):
        (staging/"source-predictions.csv").write_text('incomplete new predictions')
        raise ValueError('fit failed')
    monkeypatch.setattr(module, "_compute_macro_information", fail)
    with pytest.raises(ValueError, match="fit failed"):
        module.build_macro_information({}, pd.DataFrame(), tmp_path)
    assert (tmp_path/"additional-information.json").read_text() == '{"previous":true}'
    assert (tmp_path/"source-predictions.csv").read_text() == 'old complete predictions'
    assert not list(tmp_path.glob('.macro-run-*'))


def test_complete_runs_keep_earlier_csv_bytes_and_reject_changed_inputs(tmp_path, monkeypatch):
    from regime_lab.research import forecast_macro_information as module
    monkeypatch.setattr(module, "snapshot", lambda source, directory, refresh=False: snap(source, b"stub"))
    def compute(payload, controls, staging, snapshots):
        (staging/"predictions.csv").write_text(str(payload['value']))
        return payload
    monkeypatch.setattr(module, "_compute_macro_information", compute)
    first = module.build_macro_information({'value':1}, pd.DataFrame(), tmp_path)
    second = module.build_macro_information({'value':2}, pd.DataFrame(), tmp_path)
    assert (tmp_path/first['artifact_manifest']['files'][0]['path']).read_text() == '1'
    assert (tmp_path/second['artifact_manifest']['files'][0]['path']).read_text() == '2'
    before = (tmp_path/"additional-information.json").read_bytes()
    def changed():
        raise ValueError('input changed')
    with pytest.raises(ValueError, match="input changed"):
        module.build_macro_information({'value':3}, pd.DataFrame(), tmp_path, verify_inputs=changed)
    assert (tmp_path/"additional-information.json").read_bytes() == before
