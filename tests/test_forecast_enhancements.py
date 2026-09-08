"""Economic strata, causal adaptive decisions, and coherent simulated paths."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.boundary_forecast import next_scores
from regime_lab.analysis.forecast_audit_research import AuditResearchProtocol, economic_validation, json_safe
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.research.forecast_enhancement_models import EnhancementProtocol, evolve_paths, fit_gjr, scores_from_prices
from regime_lab.research.forecast_enhancement_diagnostics import adaptive_calibration, alert_policies, coherent_departure
from regime_lab.schema import STATE_ORDER
from regime_lab.research.forecast_enhancement_cache import cache_matches, model_cache_key


@pytest.fixture
def market():
    dates=pd.date_range("2012-01-06",periods=620,freq="W-FRI",tz="America/New_York").tz_convert("UTC")
    returns=np.random.default_rng(18).normal(.001,.023,len(dates))
    canonical=pd.DataFrame({"spy_close":100*np.exp(returns.cumsum())},index=dates)
    labeler=CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(canonical.iloc[:520])
    return canonical,labeler


def test_simulated_score_is_exact_official_hypothetical_formula(market):
    canonical,labeler=market
    history=canonical.spy_close.to_numpy()[-53:]
    shocks=np.array([-.12,-.04,0.,.03,.11])
    paths=np.tile(history,(len(shocks),1))
    paths=np.column_stack([paths[:,1:],paths[:,-1]*np.exp(shocks)])
    np.testing.assert_allclose(scores_from_prices(paths,labeler),next_scores(history,shocks,labeler),rtol=1e-11,atol=1e-11)


def test_evolving_paths_preserve_event_identity_and_reentry_semantics(market):
    canonical,labeler=market
    noise=np.random.default_rng(23).normal(size=(128,13))
    rows=evolve_paths(canonical.spy_close.to_numpy()[-53:],"risk_off",labeler,noise,.02**2,(0.,.15,0.,.85))
    assert rows[0]["risk_off_entry_probability"]==pytest.approx(0)
    assert rows[0]["first_departure_probability"]==pytest.approx(1-rows[0]["probability"][2])
    assert all(np.isclose(r["probability"].sum(),1) for r in rows)
    for key in ("first_departure_probability","risk_off_entry_probability","risk_off_occupancy_probability"):
        assert np.diff([r[key] for r in rows]).min()>=-1e-12
    assert all(r["risk_off_entry_probability"]<=r["risk_off_occupancy_probability"]+1e-12 for r in rows)


def calibration_source():
    dates=pd.date_range("2018-01-05",periods=240,freq="W-FRI",tz="UTC")
    rng=np.random.default_rng(10)
    frame=pd.DataFrame({"origin_date":dates,"anchor":.08,"raw_4":rng.uniform(.15,.65,len(dates)),"raw_13":rng.uniform(.4,.9,len(dates)),
                        "stored_4":.4,"stored_13":.7,"end_4":dates+pd.Timedelta(weeks=4),"end_13":dates+pd.Timedelta(weeks=13),
                        "y_4":rng.random(len(dates))<.25,"y_13":rng.random(len(dates))<.65})
    return frame


def test_adaptive_calibration_cannot_consume_current_or_future_outcomes():
    source=calibration_source();protocol=EnhancementProtocol(calibration_min_fit=40,calibration_min_validation=52)
    first=adaptive_calibration(source,protocol)
    date=source.origin_date.iloc[190]
    changed=source.copy()
    for h in (4,13):changed.loc[changed[f"end_{h}"]>=date,f"y_{h}"]=~changed.loc[changed[f"end_{h}"]>=date,f"y_{h}"]
    second=adaptive_calibration(changed,protocol)
    def point(result):
        return json_safe([{k:v for k,v in row.items() if k!="actual"} for row in result["history"] if row["origin_date"]==date])
    assert point(first)==point(second)
    audit=next(x for x in first["audit"] if x["origin_date"]==date)
    assert all(target is None or target<date for target in audit["last_fit_targets"])
    assert audit["last_validation_target"]<date
    for entry in first["audit"]:
        for check in entry["checks"]:
            if check["passed"]:
                assert len(check["horizons"])==2
                assert all(h["delta_log_loss"]<=0 and h["upper_one_sided_95"]<=0 for h in check["horizons"])


def test_joint_coherence_preserves_one_week_anchor():
    np.testing.assert_allclose(coherent_departure(.3,.8,.4),[.6,.6])
    np.testing.assert_allclose(coherent_departure(.7,.2,.4),[.7,.7])


def test_alarm_episode_state_does_not_depend_on_unobserved_truth():
    dates=pd.date_range("2021-01-01",periods=160,freq="W-FRI",tz="UTC")
    frame=pd.DataFrame({"model":"test","origin_date":dates,"target_date":dates+pd.Timedelta(weeks=1),"current_state":"risk_on",
                        "actual":np.where(np.arange(160)%8==0,"transition","risk_on"),"worsening_probability":np.tile([.05,.1,.8,.7,.6,.05,.05,.9],20),"evaluation_split":"retrospective_diagnostic"})
    first=alert_policies(frame)
    date=dates[120];changed=frame.copy();changed.loc[changed.target_date>=date,"actual"]="risk_off"
    second=alert_policies(changed)
    def point(result):return [{k:v for k,v in row.items() if k!="actual"} for row in result["history"] if row["origin_date"]==date]
    assert point(first)==point(second)
    metrics=first["metrics"]
    assert all(r["false_alert_episodes"]<=r["false_alert_weeks"] for r in metrics)


def test_economic_validation_retains_state_and_marks_structural_zero():
    dates=pd.date_range("2024-01-05",periods=32,freq="W-FRI",tz="UTC")
    canonical=pd.DataFrame({"spy_close":100*np.exp(np.sin(np.arange(32))*.03)},index=dates)
    scored=pd.DataFrame({"model":"test","origin_date":dates[:-1],"current_state":["risk_off" if i%2 else "risk_on" for i in range(31)],
                         "worsening_probability":[0 if i%2 else .2 for i in range(31)]})
    rows=economic_validation(scored,canonical,AuditResearchProtocol())
    worst=[r for r in rows if r["stratum"]=="risk_off"]
    assert worst and all(r["spearman_forward_return"] is None for r in worst)
    assert all(r["interpretation"]=="worsening_not_applicable_already_risk_off" for r in worst)
    assert all(r["weeks"]<next(x["weeks"] for x in rows if x["stratum"]=="all" and x["horizon_weeks"]==r["horizon_weeks"]) for r in worst)


def test_nat_serializes_as_missing_evidence():
    assert json_safe({"target":pd.NaT,"observed":pd.NA})=={"target":None,"observed":None}


def test_estimated_gjr_excludes_origin_from_parameter_fit(market):
    pytest.importorskip("arch")
    canonical,_=market
    returns=np.log(canonical.spy_close).diff().to_numpy()
    first=fit_gjr(returns,550)
    changed=returns.copy();changed[550:]=.09
    second=fit_gjr(changed,550)
    assert first[1]==second[1]
    assert first[3]==second[3]
    assert first[0]!=second[0]  # observed origin shock updates next variance
    assert first[4]["fit_last_position"]==549


def test_cache_cold_then_exact_reuse_and_changed_recipe_rejected(tmp_path):
    root=Path(__file__).resolve().parents[1]
    runtime={"numpy":"test-version","python":"test-python"}
    expected=json_safe(model_cache_key(root,{"canonical":"c","states":"s"},EnhancementProtocol().record(),runtime))
    path=tmp_path/"cache-key.json"
    assert cache_matches(path,expected) is False
    path.write_text(json.dumps(expected))
    assert cache_matches(path,expected) is True
    for section,key,value in (
        ("runtime","numpy","changed"),
        ("inputs","states","changed"),
        ("protocol","seed",99),
        ("source_sha256","src/regime_lab/analysis/labels.py","changed"),
        ("source_sha256","src/regime_lab/analysis/boundary_forecast.py","changed"),
        ("source_sha256","src/regime_lab/analysis/forecast_paths.py","changed"),
    ):
        changed=deepcopy(expected);changed[section][key]=value
        with pytest.raises(ValueError,match="recipe/source/runtime changed"):
            cache_matches(path,changed)


def test_numerical_key_covers_label_features_and_fitting_dependencies():
    root=Path(__file__).resolve().parents[1]
    key=model_cache_key(root,{},EnhancementProtocol().record(),{})
    assert {"src/regime_lab/analysis/labels.py","src/regime_lab/analysis/label_spec.py",
            "src/regime_lab/analysis/boundary_forecast.py","src/regime_lab/analysis/forecast_paths.py",
            "src/regime_lab/analysis/models.py","config/label-spec.json"}<=key["source_sha256"].keys()
