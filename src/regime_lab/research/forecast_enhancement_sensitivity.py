"""Local threshold perturbations evaluated on actual final candidate recipes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc

from regime_lab.analysis.boundary_forecast import build_boundary_inputs, next_scores, next_states, state_distribution
from regime_lab.analysis.labels import CausalRegimeLabeler, RegimeLabelConfig
from regime_lab.analysis.forecast_paths import fit_markov_path
from regime_lab.schema import STATE_ORDER
from .forecast_enhancement_models import PROB, EnhancementProtocol, _aligned, evolve_paths, fit_gjr, row_for


PATH_MODELS = ("evolving_boundary_ewma", "evolving_boundary_gjr_skewt")
STRONG_BASELINES = {1: "boundary_filtered_history", 4: "directional_duration_hazard", 13: "markov_endpoint"}


def _path_candidates(returns, sigma, residual, position, uniforms, protocol):
    """Use the unchanged main-model innovations; these do not depend on labels."""
    empirical = residual[max(2, position-protocol.train_window_weeks):position]
    empirical = np.clip(empirical[np.isfinite(empirical)], -8, 8)
    empirical = empirical-empirical.mean()
    empirical = empirical/max(empirical.std(), 1e-6)
    noise = np.quantile(empirical, uniforms)
    candidates = [(PATH_MODELS[0], sigma[position]**2, (0., 2/14, 0., 12/14), noise, False, "")]
    try:
        variance, parameters, distribution, dist_parameters, _ = fit_gjr(returns, position, protocol.train_window_weeks)
        candidates.append((PATH_MODELS[1], variance, parameters, distribution.ppf(uniforms, dist_parameters), False, ""))
    except (ValueError, FloatingPointError) as error:
        candidates.append((PATH_MODELS[1], sigma[position]**2, (0., 2/14, 0., 12/14), noise, True, str(error)))
    return candidates


def competitive_baselines(frame, protocol=EnhancementProtocol()):
    """Official-label comparisons against fixed existing competitors, never a winner selected here."""
    if frame is None:
        return {"status": "unavailable", "reason": "official_reference_predictions_not_supplied", "rows": []}
    from .forecast_enhancement_diagnostics import endpoint_scores
    mature = endpoint_scores(frame.loc[frame.actual.notna() & frame.evaluation_split.isin(["selection", "retrospective_diagnostic"])])
    rows = []
    for (model, h, period), group in mature.loc[mature.model.isin(PATH_MODELS)].groupby(["model", "horizon_weeks", "evaluation_split"]):
        base = mature.loc[mature.model.eq(STRONG_BASELINES[h]) & mature.horizon_weeks.eq(h) & mature.evaluation_split.eq(period)]
        if group.origin_date.duplicated().any() or base.origin_date.duplicated().any():
            raise ValueError("competitive comparison has duplicate origins")
        paired = group.set_index("origin_date").join(base.set_index("origin_date")[["loss", "brier_row", "actual", "target_date", "current_state"]], rsuffix="_base", how="inner").sort_index()
        if len(paired) != len(group) or len(paired) != len(base):
            raise ValueError("competitive comparison needs identical origins")
        if any(not paired[key].equals(paired[key+"_base"]) for key in ("actual", "target_date", "current_state")):
            raise ValueError("competitive comparison targets/states differ")
        delta = (paired.loss-paired.loss_base).to_numpy(float)
        for block in (13, 26, 52):
            width = min(block, max(1, len(delta)//2))
            starts = np.random.default_rng(protocol.seed).integers(0, len(delta), (protocol.bootstrap_resamples, int(np.ceil(len(delta)/width))))
            indices = ((starts[..., None]+np.arange(width)) % len(delta)).reshape(protocol.bootstrap_resamples, -1)[:, :len(delta)]
            lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            rows.append({"model": model, "baseline_model": STRONG_BASELINES[h], "horizon_weeks": int(h),
                         "evaluation_split": period, "spec_id": "official", "target": "endpoint", "n_predictions": len(paired),
                         "log_loss": float(paired.loss.mean()), "baseline_log_loss": float(paired.loss_base.mean()),
                         "delta_log_loss": float(delta.mean()), "delta_brier": float((paired.brier_row-paired.brier_row_base).mean()),
                         "block_weeks": width, "requested_block_weeks": block, "ci_low": float(lo), "ci_high": float(hi),
                         "resamples": protocol.bootstrap_resamples})
    return {"status": "computed", "scope": "official_label_only_fixed_existing_comparators_same_origin_and_target",
            "evaluation_role": "retrospective_diagnostic_not_fresh_holdout", "baselines": STRONG_BASELINES, "rows": rows}


def label_explanations(canonical, states, fit_weeks=520):
    labeler=CausalRegimeLabeler(RegimeLabelConfig(minimum_fit_observations=260)).fit(canonical.iloc[:fit_weeks])
    scores=labeler.score_frame(canonical)
    if not labeler.transform(canonical).equals(states.rename("regime")):
        raise ValueError("official label explanation differs from authoritative state")
    lower,upper=labeler.lower_threshold_,labeler.upper_threshold_
    margin=(upper-lower)*labeler.config.hysteresis_fraction
    rows=[]
    price=canonical.spy_close.to_numpy(float)
    for position in range(fit_weeks+1,len(canonical)):
        score=float(scores.risk_score.iloc[position]);state=str(states.iloc[position])
        boundary=upper-margin if state=="risk_on" else lower+margin if state=="risk_off" else lower if score<(lower+upper)/2 else upper
        zero=float(next_scores(price[position-52:position+1],np.array([0.]),labeler)[0])
        rows.append({"origin_date":canonical.index[position],"current_state":state,"risk_score":score,
                     "trend_score":float(scores.trend_score.iloc[position]),"stress_score":float(scores.stress_score.iloc[position]),
                     "distance_to_boundary":abs(score-boundary),"active_boundary":boundary,
                     "boundary_direction":"down" if state=="risk_on" else "up" if state=="risk_off" else "nearest_of_two",
                     "weekly_score_change":float(scores.risk_score.diff().iloc[position]),"known_zero_return_rolloff_change":zero-score,
                     "lower_threshold":lower,"upper_threshold":upper,"hysteresis_margin":margin,
                     "membership_semantics":"distance_to_anchor_not_posterior"})
    return rows


def near_label_sensitivity(canonical, official_states, protocol=EnhancementProtocol(), progress=None, *, reference_predictions=None):
    """Refit endpoint/boundary candidates and evolve price paths for every label.

    These nearby perturbations are frozen before scoring and never select the
    operating label. Features are causal price transforms plus each alternative's
    actual current state. Full weekly origin coverage is retained.
    """
    variants=[("official",.30,.70,.15),("wider_thresholds",.275,.725,.15),("narrower_thresholds",.325,.675,.15),
              ("less_hysteresis",.30,.70,.10),("more_hysteresis",.30,.70,.20)]
    prices=canonical.spy_close
    returns=np.log(prices).diff()
    sigma=returns.ewm(span=13,adjust=False).std(bias=True).clip(lower=.003)
    residual=returns/sigma.shift(1)
    records=[];stability=[]
    path_cache={}
    uniforms=qmc.Sobol(d=13,scramble=True,seed=protocol.seed).random_base2(protocol.simulation_power)
    return_values,sigma_values,residual_values=returns.to_numpy(),sigma.to_numpy(),residual.to_numpy()
    official_changes=official_states.ne(official_states.shift())
    first=protocol.label_fit_weeks+1
    base_features=build_boundary_inputs(canonical,official_states,fit_weeks=protocol.label_fit_weeks).features
    from scipy.stats import t as student_t
    quantiles=(np.arange(1001)+.5)/1001
    student_shocks=student_t.ppf(quantiles,df=5)*np.sqrt(3/5)
    for spec,low,high,hysteresis in variants:
        labeler=CausalRegimeLabeler(RegimeLabelConfig(lower_quantile=low,upper_quantile=high,hysteresis_fraction=hysteresis,minimum_fit_observations=260)).fit(canonical.iloc[:protocol.label_fit_weeks])
        states=labeler.transform(canonical)
        if spec=="official" and not states.equals(official_states.rename(states.name)):
            raise ValueError("sensitivity official labels differ from authoritative states")
        scores=labeler.score_frame(canonical)
        width=labeler.upper_threshold_-labeler.lower_threshold_
        feature=base_features.copy()
        feature["score"]=scores.risk_score/width
        feature["trend"]=scores.trend_score/width
        feature["stress"]=scores.stress_score/width
        feature["distance_lower"]=(scores.risk_score-labeler.lower_threshold_)/width
        feature["distance_upper"]=(scores.risk_score-labeler.upper_threshold_)/width
        for lag in (1,2,4):feature[f"score_delta_{lag}"]=scores.risk_score.diff(lag)/width
        feature["score_acceleration"]=scores.risk_score.diff().diff()/width
        feature["log_state_age"]=np.log1p(states.groupby(states.ne(states.shift()).cumsum()).cumcount()+1)
        price_values=prices.to_numpy()
        for position in range(52,len(states)):
            history=price_values[position-52:position+1]
            hypothetical=next_scores(history,float(sigma.iloc[position])*np.array([-1.,0.,1.]),labeler)
            date=states.index[position]
            for name,value in zip(("down_return_next_score","zero_return_next_score","up_return_next_score"),hypothetical):feature.loc[date,name]=value/width
            feature.loc[date,"known_rolloff_score_change"]=(hypothetical[1]-scores.risk_score.iloc[position])/width
            mechanical=state_distribution(next_scores(history,float(sigma.iloc[position])*student_shocks,labeler),states.iloc[position],labeler)
            for state,value in zip(STATE_ORDER,mechanical):feature.loc[date,f"mechanical_probability_{state}"]=value
        for state in STATE_ORDER:
            indicator=states.eq(state).astype(float)
            feature[f"state_{state}"]=indicator
            for col in ("score","distance_lower","distance_upper","zero_return_next_score","down_return_next_score","up_return_next_score","score_delta_1"):
                feature[f"{state}__{col}"]=indicator*feature[col]
        estimators={}
        for position in range(first,len(states)):
            markov=fit_markov_path(states,position,train_window_weeks=protocol.train_window_weeks)
            for h in (1,4,13):
                raw=row_for(states,position,h,"markov_endpoint",np.linalg.matrix_power(markov,h)[STATE_ORDER.index(states.iloc[position])])
                raw["spec_id"]=spec;records.append(raw)
                if (position-first)%protocol.direct_refit_every_weeks==0 or h not in estimators:
                    train=np.arange(max(52,position-protocol.train_window_weeks),position-h)
                    estimator=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=1000,tol=1e-6))
                    estimator.fit(feature.iloc[train],states.iloc[train+h])
                    estimators[h]=estimator
                row=row_for(states,position,h,"direct_endpoint_ridge",_aligned(estimators[h],feature.iloc[[position]]))
                row["spec_id"]=spec;records.append(row)
            shocks=residual.iloc[max(2,position-protocol.train_window_weeks):position].dropna().clip(-8,8).to_numpy()*sigma.iloc[position]
            s=next_scores(prices.to_numpy()[position-52:position+1],shocks,labeler)
            next_values=next_states(s,states.iloc[position],labeler)
            probability=.99*np.array([(next_values==state).mean() for state in STATE_ORDER])+.01/3
            row=row_for(states,position,1,"boundary_residual_raw",probability)
            row["spec_id"]=spec;records.append(row)
            if position not in path_cache:
                path_cache[position]=_path_candidates(return_values,sigma_values,residual_values,position,uniforms,protocol)
            for model,variance,parameters,innovations,fallback,reason in path_cache[position]:
                simulated=evolve_paths(price_values[position-52:position+1],str(states.iloc[position]),labeler,innovations,variance,parameters)
                for item in simulated:
                    horizon,probability=item.pop("horizon_weeks"),item.pop("probability")
                    row=row_for(states,position,horizon,model,probability,last_train_target=states.index[position-1],fallback=fallback,fallback_reason=reason,**item)
                    row["spec_id"]=spec;records.append(row)
        for period,mask in (("selection",(states.index< pd.Timestamp("2023-01-01",tz="UTC"))&(np.arange(len(states))>=first)),("retrospective_diagnostic",states.index>=pd.Timestamp("2023-01-01",tz="UTC"))):
            c=states.ne(states.shift()).to_numpy()[mask];o=official_changes.to_numpy()[mask]
            stability.append({"spec_id":spec,"evaluation_split":period,"n_predictions":int(mask.sum()),"state_agreement":float((states.to_numpy()[mask]==official_states.to_numpy()[mask]).mean()) if mask.any() else None,
                              "transition_jaccard":float((c&o).sum()/max(1,(c|o).sum())),"transition_event_count":int(c.sum()),"lower_quantile":low,"upper_quantile":high,"hysteresis_fraction":hysteresis})
        if progress:progress(f"Nearby labels completed: {spec}")
    frame=pd.DataFrame(records)
    parity={}
    if reference_predictions is not None:
        for model in PATH_MODELS:
            columns=["origin_date","horizon_weeks"]
            actual=frame.loc[frame.spec_id.eq("official")&frame.model.eq(model)].set_index(columns).sort_index()
            expected=reference_predictions.loc[reference_predictions.model.eq(model)].set_index(columns).sort_index()
            if not actual.index.equals(expected.index) or any(not actual[key].equals(expected[key]) for key in ("actual","target_date","current_state")):
                raise ValueError("official path sensitivity coverage differs from the main candidate")
            difference=float(np.abs(actual[PROB].to_numpy(float)-expected[PROB].to_numpy(float)).max())
            if difference>1e-12:
                raise ValueError("official path sensitivity changed the main candidate probabilities")
            parity[model]={"n_predictions":len(actual),"maximum_probability_difference":difference}
    from .forecast_enhancement_diagnostics import endpoint_scores
    mature=endpoint_scores(frame.loc[frame.actual.notna()&frame.evaluation_split.isin(["selection","retrospective_diagnostic"])])
    rows=[]
    for (spec,model,h,period),group in mature.groupby(["spec_id","model","horizon_weeks","evaluation_split"]):
        base=mature.loc[mature.spec_id.eq(spec)&mature.model.eq("markov_endpoint")&mature.horizon_weeks.eq(h)&mature.evaluation_split.eq(period)]
        paired=group.set_index("origin_date").join(base.set_index("origin_date")[["loss","actual","target_date"]],rsuffix="_base",how="inner")
        if len(paired)!=len(group) or len(paired)!=len(base) or any(not paired[key].equals(paired[key+"_base"]) for key in ("actual","target_date")):
            raise ValueError("label sensitivity requires matching origins and definition-specific targets")
        rows.append({"spec_id":spec,"model":model,"horizon_weeks":int(h),"evaluation_split":period,"n_predictions":len(group),"log_loss":float(group.loss.mean()),"brier":float(group.brier_row.mean()),"delta_log_loss":float((paired.loss-paired.loss_base).mean()),"baseline_model":"markov_endpoint","fallback_count":int(group.fallback.sum())})
    return {"rows":rows,"stability":stability,"selection_rule":"fixed_nearby_quantile_and_hysteresis_perturbations_no_label_selection",
            "scope":"boundary_residual_and_direct_ridge_refitted_and_ewma_gjr_paths_relabelled_for_every_definition; unchanged_final_model_recipes; operating_ensemble_and_hazard_not_relabelled",
            "path_fit_reuse":{"scope":"label_independent_variance_and_innovations_reused_across_definitions","origin_count":len(path_cache),"gjr_fallback_origins":sum(candidates[1][4] for candidates in path_cache.values()),"fit_rule":"returns_strictly_before_origin_then_assimilate_observed_origin_return","random_draws":len(uniforms)},
            "official_path_parity":parity,
            "competitive_baselines":competitive_baselines(reference_predictions,protocol),
            "models":["boundary_residual_raw","direct_endpoint_ridge","markov_endpoint",*PATH_MODELS]},frame
