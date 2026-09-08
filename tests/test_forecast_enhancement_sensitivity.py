"""Same-label path parity, causal reuse, and matched competitive comparisons."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.research import forecast_enhancement_sensitivity as sensitivity
from regime_lab.research.forecast_enhancement_models import EnhancementProtocol, PROB, run_models


@pytest.fixture(scope="module")
def complete_label_experiment():
    pytest.importorskip("arch")
    dates=pd.date_range("2014-01-03",periods=541,freq="W-FRI",tz="UTC")
    prices=100*np.exp(np.random.default_rng(18).normal(.001,.023,len(dates)).cumsum())
    canonical=pd.DataFrame({"spy_close":prices},index=dates)
    labeler=CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(canonical.iloc[:520])
    states=labeler.transform(canonical)
    protocol=replace(EnhancementProtocol(),simulation_power=4,bootstrap_resamples=49)
    reference,_,_=run_models(canonical,states,protocol)
    # Reference names are fixture aliases solely for join/target verification.
    extras=[]
    for horizon,name in ((1,"boundary_filtered_history"),(4,"directional_duration_hazard")):
        extra=reference.loc[reference.model.eq("markov_endpoint")&reference.horizon_weeks.eq(horizon)].copy()
        extra["model"]=name;extras.append(extra)
    reference=pd.concat([reference,*extras],ignore_index=True)
    calls=[]
    original=sensitivity.fit_gjr
    def recording_fit(*args,**kwargs):
        calls.append(args[1])
        return original(*args,**kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sensitivity,"fit_gjr",recording_fit)
        result,rows=sensitivity.near_label_sensitivity(canonical,states,protocol,reference_predictions=reference)
    return canonical,states,protocol,reference,result,rows,calls


def test_every_definition_has_complete_path_horizons_and_exact_official_parity(complete_label_experiment):
    canonical,states,_,reference,result,rows,calls=complete_label_experiment
    origins=canonical.index[521:]
    paths=rows.loc[rows.model.isin(sensitivity.PATH_MODELS)]
    assert len(paths)==5*2*3*len(origins)
    assert not paths.duplicated(["spec_id","model","origin_date","horizon_weeks"]).any()
    assert len(calls)==len(origins)==result["path_fit_reuse"]["origin_count"]
    for _,group in paths.groupby(["spec_id","model","horizon_weeks"]):
        assert list(group.origin_date)==list(origins)
    for model in sensitivity.PATH_MODELS:
        assert result["official_path_parity"][model]["maximum_probability_difference"]<=1e-12
        assert result["official_path_parity"][model]["n_predictions"]==3*len(origins)
    assert paths[PROB].sum(axis=1).between(1-1e-12,1+1e-12).all()
    assert (paths.last_train_target<paths.origin_date).all()
    assert set(result["models"])==set(rows.model)
    assert len(result["competitive_baselines"]["rows"])==18


def test_alternative_targets_follow_their_own_actual_definition(complete_label_experiment):
    canonical,_,_,_,_,rows,_=complete_label_experiment
    spec="wider_thresholds"
    labeler=CausalRegimeLabeler(RegimeLabelConfig(lower_quantile=.275,upper_quantile=.725,hysteresis_fraction=.15,minimum_fit_observations=260)).fit(canonical.iloc[:520])
    states=labeler.transform(canonical)
    for record in rows.loc[rows.spec_id.eq(spec)&rows.model.isin(sensitivity.PATH_MODELS)].itertuples():
        position=states.index.get_loc(record.origin_date)
        assert record.current_state==states.iloc[position]
        if position+record.horizon_weeks<len(states):
            assert record.actual==states.iloc[position+record.horizon_weeks]
        else:
            assert pd.isna(record.actual)


def test_future_returns_cannot_change_label_independent_path_inputs(complete_label_experiment):
    canonical,_,protocol,_,_,_,_=complete_label_experiment
    def candidates(prices):
        returns=np.log(prices).diff().to_numpy()
        sigma=pd.Series(returns).ewm(span=13,adjust=False).std(bias=True).clip(lower=.003).to_numpy()
        residual=returns/np.r_[np.nan,sigma[:-1]]
        uniforms=qmc.Sobol(d=13,scramble=True,seed=protocol.seed).random_base2(protocol.simulation_power)
        return sensitivity._path_candidates(returns,sigma,residual,525,uniforms,protocol)
    original=candidates(canonical.spy_close)
    changed=canonical.spy_close.copy();changed.iloc[526:]*=3
    altered=candidates(changed)
    for first,second in zip(original,altered):
        assert first[:3]==second[:3]
        np.testing.assert_array_equal(first[3],second[3])
        assert first[4:]==second[4:]


def test_competitive_pairs_reject_missing_or_different_targets(complete_label_experiment):
    _,_,protocol,reference,result,_,_=complete_label_experiment
    row=next(r for r in result["competitive_baselines"]["rows"] if r["horizon_weeks"]==1)
    assert row["baseline_model"]=="boundary_filtered_history"
    changed=reference.copy()
    idx=changed.index[changed.model.eq("boundary_filtered_history")][0]
    with pytest.raises(ValueError,match="identical origins"):
        sensitivity.competitive_baselines(changed.drop(idx),protocol)
    changed.at[idx,"target_date"]=pd.Timestamp(changed.at[idx,"target_date"])+pd.Timedelta(weeks=1)
    with pytest.raises(ValueError,match="targets/states differ"):
        sensitivity.competitive_baselines(changed,protocol)


def test_gjr_failure_keeps_visible_unchanged_ewma_fallback(monkeypatch):
    returns=np.random.default_rng(51).normal(0,.02,540)
    sigma=pd.Series(returns).ewm(span=13,adjust=False).std(bias=True).clip(lower=.003).to_numpy()
    residual=returns/np.r_[np.nan,sigma[:-1]]
    def fail(*args,**kwargs):
        raise ValueError("test optimizer failure")
    monkeypatch.setattr(sensitivity,"fit_gjr",fail)
    rows=sensitivity._path_candidates(returns,sigma,residual,525,np.full((16,13),.5),EnhancementProtocol())
    assert rows[1][0]=="evolving_boundary_gjr_skewt"
    assert rows[1][4:]==(True,"test optimizer failure")
    assert rows[0][1:3]==rows[1][1:3]
    np.testing.assert_array_equal(rows[0][3],rows[1][3])
