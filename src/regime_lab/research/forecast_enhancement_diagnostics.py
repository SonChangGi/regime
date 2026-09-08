"""Matured-only adaptive policies and event-aware forecast diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression

from regime_lab.analysis.forecast_audit_research import budget_threshold
from regime_lab.schema import STATE_ORDER
from .forecast_enhancement_models import EnhancementProtocol, PROB, split


def binary_loss(actual, probability):
    y = np.asarray(actual, float)
    p = np.clip(np.asarray(probability, float), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def endpoint_scores(frame):
    frame = frame.copy()
    p = frame[PROB].to_numpy(float)
    actual = frame.actual.map({s: i for i, s in enumerate(STATE_ORDER)}).to_numpy(int)
    frame["loss"] = -np.log(np.maximum(p[np.arange(len(p)), actual], 1e-9))
    frame["brier_row"] = ((p - np.eye(3)[actual]) ** 2).sum(axis=1)
    frame["correct"] = actual == p.argmax(axis=1)
    return frame


def summarize_models(frame):
    mature = frame.loc[frame.actual.notna() & frame.evaluation_split.isin(["selection", "retrospective_diagnostic"])].copy()
    scored = endpoint_scores(mature)
    rows = []
    for (model, horizon, period), group in scored.groupby(["model", "horizon_weeks", "evaluation_split"]):
        baseline = scored.loc[scored.model.eq("markov_endpoint") & scored.horizon_weeks.eq(horizon) & scored.evaluation_split.eq(period)].set_index("origin_date")
        common = group.set_index("origin_date").join(baseline[["loss"]], rsuffix="_baseline", how="inner")
        if len(common) != len(group):
            raise ValueError(f"incomplete same-origin model comparison: {model}/{horizon}")
        current = group.current_state.map({s: i for i, s in enumerate(STATE_ORDER)}).to_numpy()
        actual = group.actual.map({s: i for i, s in enumerate(STATE_ORDER)}).to_numpy()
        p = group[PROB].to_numpy(float)
        worsening = (p * (np.arange(3) > current[:, None])).sum(axis=1)
        at_risk=current<2
        rows.append({"model": model, "horizon_weeks": int(horizon), "target": "endpoint", "evaluation_split": period,
                     "n_predictions": len(group), "n": len(group), "log_loss": float(group.loss.mean()),
                     "brier": float(group.brier_row.mean()), "accuracy": float(group.correct.mean()),
                     "baseline_model": "markov_endpoint", "delta_log_loss": float((common.loss - common.loss_baseline).mean()),
                     "worsening_brier": float((((actual > current)[at_risk] - worsening[at_risk]) ** 2).mean()) if at_risk.any() else None,
                     "worsening_at_risk_n": int(at_risk.sum()), "worsening_score_population": "current_risk_on_or_transition",
                     "worsening_event_count": int((actual > current).sum()), "fallback_count": int(group.fallback.sum())})
    return rows, scored


def coherent_departure(anchor, p4, p13):
    """Euclidean isotonic projection with an immutable official one-week anchor."""
    values = np.array([max(anchor, p4), max(anchor, p13)], float)
    if values[0] > values[1]:
        values[:] = values.mean()
    return values


def _upper_block_mean(values, block=13):
    values = np.asarray(values, float)
    chunks = [values[start:start + block].mean() for start in range(0, len(values) - block + 1, block)]
    if len(chunks) < 4:
        return float("inf"), len(chunks)
    upper = values.mean() + student_t.ppf(.95, len(chunks) - 1) * np.std(chunks, ddof=1) / np.sqrt(len(chunks))
    return float(upper), len(chunks)


def adaptive_calibration(source, protocol=EnhancementProtocol()):
    """Evaluate fixed shrink candidates prequentially *after* joint coherence.

    source has one row/origin: anchor, raw_4/raw_13, y_4/y_13, end_4/end_13.
    Neither current nor future outcomes enter coefficient fits or policy choices.
    """
    frame = source.sort_values("origin_date").copy()
    if frame.origin_date.duplicated().any():
        raise ValueError("duplicate calibration origin")
    weights = (0., .25, .5)
    candidates, chosen, audits = [], [], []
    for record in frame.itertuples(index=False):
        origin = record.origin_date
        calibrated = []
        fit_targets = []
        for horizon in (4, 13):
            past = frame.loc[(frame[f"end_{horizon}"] < origin) & frame[f"y_{horizon}"].notna()].tail(protocol.calibration_window_weeks)
            raw = float(getattr(record, f"raw_{horizon}"))
            y = past[f"y_{horizon}"].to_numpy(float)
            if len(past) < protocol.calibration_min_fit or len(np.unique(y)) < 2 or min(y.sum(), (1-y).sum()) < 5:
                calibrated.append(raw)
            else:
                estimator = LogisticRegression(C=.1, max_iter=1000, tol=1e-7)
                estimator.fit(logit(np.clip(past[f"raw_{horizon}"].to_numpy(float), 1e-6, 1-1e-6))[:, None], y)
                calibrated.append(float(estimator.predict_proba([[logit(np.clip(raw, 1e-6, 1-1e-6))]])[0, 1]))
            fit_targets.append(past[f"end_{horizon}"].max() if len(past) else None)
        current = {}
        for weight in weights:
            values = coherent_departure(record.anchor, (1-weight)*record.raw_4 + weight*calibrated[0], (1-weight)*record.raw_13 + weight*calibrated[1])
            current[weight] = values
        # Candidate validation is its own previously issued, coherent OOS path.
        eligible = [x for x in candidates if x["end_13"] < origin and x["y_13"] is not None][-protocol.calibration_window_weeks:]
        selected = 0.
        checks = []
        if len(eligible) >= protocol.calibration_min_validation:
            identity = np.array([sum(binary_loss([x[f"y_{h}"]], [x["candidates"][0.][i]])[0] for i,h in enumerate((4,13)))/2 for x in eligible])
            passing = []
            for weight in weights[1:]:
                loss = np.array([sum(binary_loss([x[f"y_{h}"]], [x["candidates"][weight][i]])[0] for i,h in enumerate((4,13)))/2 for x in eligible])
                delta = loss - identity
                upper, blocks = _upper_block_mean(delta)
                horizon_checks=[]
                for i,h in enumerate((4,13)):
                    yd=np.array([x[f"y_{h}"] for x in eligible],float)
                    difference=binary_loss(yd,[x["candidates"][weight][i] for x in eligible])-binary_loss(yd,[x["candidates"][0.][i] for x in eligible])
                    hu,hb=_upper_block_mean(difference)
                    horizon_checks.append({"horizon_weeks":h,"delta_log_loss":float(difference.mean()),"upper_one_sided_95":hu if np.isfinite(hu) else None,"blocks":hb,"passed":bool(difference.mean()<=0 and hu<=0)})
                passed = delta.mean() < -protocol.calibration_material_improvement and upper <= 0 and all(x["passed"] for x in horizon_checks)
                checks.append({"weight": weight, "delta_log_loss": float(delta.mean()), "upper_one_sided_95": upper if np.isfinite(upper) else None, "blocks": blocks, "horizons":horizon_checks,"passed": bool(passed)})
                if passed:
                    passing.append((float(loss.mean()), weight))
            if passing:
                selected = min(passing)[1]
        values = current[selected]
        history_row = {"origin_date": origin, "end_13": record.end_13, "end_4": record.end_4,
                       "y_13": None if pd.isna(record.y_13) else bool(record.y_13),
                       "y_4": None if pd.isna(record.y_4) else bool(record.y_4), "candidates": current}
        candidates.append(history_row)
        audits.append({"origin_date": origin, "selected_weight": selected, "validation_rows": len(eligible),
                       "last_validation_target": eligible[-1]["end_13"] if eligible else None,
                       "last_fit_targets": fit_targets, "checks": checks})
        for i,horizon in enumerate((4,13)):
            target = getattr(record, f"end_{horizon}")
            actual = getattr(record, f"y_{horizon}")
            for model, probability in (("adaptive_shrink_after_coherence", values[i]), ("identity_after_coherence", current[0.][i]), ("frozen_stored_calibration", getattr(record, f"stored_{horizon}"))):
                chosen.append({"origin_date": origin, "target_date": target, "horizon_weeks": horizon,
                               "target": "first_departure", "model": model, "probability": float(probability),
                               "raw_probability": float(getattr(record, f"raw_{horizon}")), "anchor": record.anchor,
                               "actual": None if pd.isna(actual) else bool(actual),
                               "selected_weight": selected if model == "adaptive_shrink_after_coherence" else None,
                               "evaluation_split": split(origin, target, not pd.isna(actual))})
    output = pd.DataFrame(chosen)
    rows = []
    for (model,horizon,period),group in output.loc[output.actual.notna() & output.evaluation_split.isin(["selection","retrospective_diagnostic"])].groupby(["model","horizon_weeks","evaluation_split"]):
        y,p = group.actual.to_numpy(float),group.probability.to_numpy(float)
        rows.append({"model":model,"horizon_weeks":int(horizon),"target":"first_departure","evaluation_split":period,"n_predictions":len(group),"event_count":int(y.sum()),"log_loss":float(binary_loss(y,p).mean()),"brier":float(((p-y)**2).mean())})
    return {"metrics":rows,"history":output.to_dict("records"),"latest":output.loc[output.origin_date.eq(output.origin_date.max())].to_dict("records"),
            "policy":{"choices":[0,.25,.5],"target":"first_departure","no_harm_gate":"each_horizon_paired_coherent_delta_and_95pct_block_upper_bound_nonpositive_plus_joint_material_improvement","scope":"fixed_policy_adapts_only_to_strictly_matured_past_outcomes","min_fit_rows":protocol.calibration_min_fit,"min_validation_rows":protocol.calibration_min_validation},"audit":audits}


def alert_policies(frame, protocol=EnhancementProtocol()):
    """Weekly and episode alarms; outcomes never control episode state."""
    rows = []
    for model, group in frame.groupby("model"):
        group = group.sort_values("origin_date")
        active, cooldown = False, 0
        for record in group.itertuples():
            history = group.loc[(group.target_date < record.origin_date) & group.actual.notna()].tail(156)
            y = np.array([STATE_ORDER.index(a) > STATE_ORDER.index(c) for a,c in zip(history.actual,history.current_state)])
            probabilities = history.worsening_probability.to_numpy(float)
            threshold = budget_threshold(probabilities,y,annual_budget=protocol.alert_annual_budget) if len(history)>=52 else np.nextafter(1.,2.)
            p = float(record.worsening_probability)
            episode_start = False
            if active and p < threshold * protocol.alert_exit_fraction:
                active = False
                cooldown = protocol.alert_cooldown_weeks
            elif not active:
                if cooldown:
                    cooldown -= 1
                elif p >= threshold:
                    active = True
                    episode_start = True
            actual = None if pd.isna(record.actual) else STATE_ORDER.index(record.actual)>STATE_ORDER.index(record.current_state)
            for policy, alert, start in (("weekly_threshold",p>=threshold,p>=threshold),("episode_hysteresis_cooldown",active,episode_start)):
                rows.append({"model":model,"policy":policy,"origin_date":record.origin_date,"target_date":record.target_date,"horizon_weeks":1,
                             "target":"one_week_worsening","probability":p,"threshold":float(threshold),"alert":bool(alert),"episode_start":bool(start),
                             "actual":actual,"current_state":record.current_state,"evaluation_split":record.evaluation_split,
                             "calibration_rows":len(history),"last_policy_target":history.target_date.max() if len(history) else None,
                             "cooldown_remaining":cooldown if policy=="episode_hysteresis_cooldown" else 0})
    result = pd.DataFrame(rows)
    metrics = []
    for (model,policy,period),group in result.loc[result.actual.notna() & result.evaluation_split.isin(["selection","retrospective_diagnostic"])].groupby(["model","policy","evaluation_split"]):
        group=group.sort_values("origin_date")
        truth,alarm = group.actual.to_numpy(bool),group.alert.to_numpy(bool)
        # Alarm episodes are connected active runs, clipped at evaluation boundaries.
        starts=np.flatnonzero(alarm & ~np.r_[False,alarm[:-1]])
        false_episodes=0
        censored_episodes=0
        for start in starts:
            end=start
            while end+1<len(alarm) and alarm[end+1]:end+=1
            if end==len(alarm)-1 and alarm[end]:
                censored_episodes+=1
            elif not truth[start:end+1].any():false_episodes+=1
        count=int(truth.sum());hits=int((truth&alarm).sum());false=int((~truth&alarm).sum())
        lead=[]
        for index in np.flatnonzero(truth&alarm):
            beginning=int(index)
            while beginning>0 and alarm[beginning-1]:beginning-=1
            lead.append(float((group.target_date.iloc[index]-group.origin_date.iloc[beginning]).total_seconds()/604800))
        metrics.append({"model":model,"policy":policy,"evaluation_split":period,"horizon_weeks":1,"n_predictions":len(group),"event_count":count,"hit_count":hits,"recall":hits/count if count else None,
                        "false_alert_weeks":false,"false_alert_episodes":false_episodes,"false_alerts_per_year":false/len(group)*52.1775,
                        "false_alert_episodes_per_year":false_episodes/len(group)*52.1775,"alert_episodes":len(starts),"right_censored_alert_episodes":censored_episodes,
                        "annual_budget":protocol.alert_annual_budget,"false_week_budget_exceeded":false/len(group)*52.1775>protocol.alert_annual_budget})
        metrics[-1].update({"mean_lead_weeks_detected_events":float(np.mean(lead)) if lead else None,
                           "median_lead_weeks_detected_events":float(np.median(lead)) if lead else None,
                           "lead_definition":"first_active_alarm_origin_to_actual_worsening_target; continuing alarms count once per observed worsening",
                           "false_episode_budget_exceeded":false_episodes/len(group)*52.1775>protocol.alert_annual_budget})
    return {"metrics":metrics,"history":result.to_dict("records"),"latest":result.loc[result.origin_date.eq(result.origin_date.max())].to_dict("records"),
            "policy":{"target":"one_week_worsening","budget":protocol.alert_annual_budget,"budget_unit":"historical_false_weeks_per_year","window_weeks":156,"cooldown_weeks":protocol.alert_cooldown_weeks,"exit_threshold_fraction":protocol.alert_exit_fraction,"threshold_training":"strictly_matured_past_only","episode_truth":"any_actual_worsening_within_active_alarm_run"}}


def robustness(scored, states, protocol=EnhancementProtocol()):
    starts=np.flatnonzero(states.eq("risk_off").to_numpy() & states.shift().ne("risk_off").to_numpy())
    episodes=[]
    for position in starts:
        if position==0:continue
        end=position
        while end+1<len(states) and states.iloc[end+1]=="risk_off":end+=1
        episodes.append({"episode_id":states.index[position].isoformat(),"start":states.index[position],"end":states.index[end],"duration_weeks":end-position+1,"evaluation_split":"selection" if states.index[position]<pd.Timestamp("2023-01-01",tz="UTC") else "retrospective_diagnostic","right_censored":end==len(states)-1})
    loo=[];blocks=[]
    for (model,horizon,period),group in scored.groupby(["model","horizon_weeks","evaluation_split"]):
        if model=="markov_endpoint":continue
        base=scored.loc[scored.model.eq("markov_endpoint") & scored.horizon_weeks.eq(horizon)&scored.evaluation_split.eq(period)].set_index("origin_date")
        common=group.set_index("origin_date").join(base[["loss"]],rsuffix="_base",how="inner").sort_index()
        delta=(common.loss-common.loss_base).to_numpy(float)
        for block in (13,26,52):
            count=len(delta);width=min(block,max(1,count//2));starts_draw=np.random.default_rng(protocol.seed).integers(0,count,(protocol.bootstrap_resamples,int(np.ceil(count/width))))
            indices=((starts_draw[...,None]+np.arange(width))%count).reshape(protocol.bootstrap_resamples,-1)[:,:count]
            ci=np.quantile(delta[indices].mean(axis=1),[.025,.975])
            blocks.append({"model":model,"horizon_weeks":int(horizon),"evaluation_split":period,"baseline_model":"markov_endpoint","n_predictions":count,"block_weeks":width,"delta_log_loss":float(delta.mean()),"ci_low":float(ci[0]),"ci_high":float(ci[1]),"resamples":protocol.bootstrap_resamples})
        for episode in episodes:
            if episode["evaluation_split"]!=period:continue
            # Remove every origin whose forecast window touches the spell.
            mask=(common.index<=episode["end"]) & (pd.to_datetime(common.target_date,utc=True)>=episode["start"])
            remaining=delta[~mask]
            if len(remaining):loo.append({"model":model,"horizon_weeks":int(horizon),"evaluation_split":period,"episode_id":episode["episode_id"],"removed_origins":int(mask.sum()),"remaining_origins":len(remaining),"delta_log_loss":float(remaining.mean()),"full_delta_log_loss":float(delta.mean())})
    return {"events":episodes,"leave_one_event_out":loo,"block_sensitivity":blocks,
            "definition":"Overlapping forecast windows touching each Risk-off spell are removed together; descriptive robustness does not restore a fresh holdout."}


def event_and_calibration_metrics(frame,states):
    """Path-event scores and class/state/direction reliability on matched rows."""
    from regime_lab.analysis.forecast_paths import fit_markov_path
    events=[];health=[]
    state_values=states.to_numpy()
    matrix_cache={}
    for record in frame.loc[frame.actual.notna()&frame.evaluation_split.isin(["selection","retrospective_diagnostic"])].itertuples():
        position=int(states.index.get_loc(record.origin_date));h=int(record.horizon_weeks);current=STATE_ORDER.index(record.current_state)
        values=state_values[position+1:position+h+1];previous=state_values[position:position+h]
        probability={"endpoint_risk_off":getattr(record,"p_risk_off")}
        for name,column in (("first_departure","first_departure_probability"),("risk_off_entry","risk_off_entry_probability"),("risk_off_occupancy","risk_off_occupancy_probability")):
            value=getattr(record,column,None)
            if value is not None and pd.notna(value):probability[name]=float(value)
        if record.model=="markov_endpoint":
            if position not in matrix_cache:matrix_cache[position]=fit_markov_path(states,position)
            matrix=matrix_cache[position];start=np.eye(3)[current]
            entry_kernel=matrix.copy();entry_kernel[:2,2]=0
            occupancy_kernel=matrix.copy();occupancy_kernel[:,2]=0
            probability.update({"first_departure":1-matrix[current,current]**h,
                                "risk_off_entry":1-float(start@np.linalg.matrix_power(entry_kernel,h)@np.ones(3)),
                                "risk_off_occupancy":1-float(start@np.linalg.matrix_power(occupancy_kernel,h)@np.ones(3))})
        actual={"endpoint_risk_off":record.actual=="risk_off","first_departure":bool((values!=record.current_state).any()),
                "risk_off_entry":bool(((previous!="risk_off")&(values=="risk_off")).any()),"risk_off_occupancy":bool((values=="risk_off").any())}
        for target,p in probability.items():
            events.append({"model":record.model,"horizon_weeks":h,"origin_date":record.origin_date,"target_date":record.target_date,
                           "evaluation_split":record.evaluation_split,"current_state":record.current_state,"target":target,
                           "actual":actual[target],"probability":float(np.clip(p,0,1))})
    event_frame=pd.DataFrame(events)
    metrics=[]
    for (model,h,target,period),group in event_frame.groupby(["model","horizon_weeks","target","evaluation_split"]):
        y,p=group.actual.to_numpy(float),group.probability.to_numpy(float)
        baseline=event_frame.loc[event_frame.model.eq("markov_endpoint")&event_frame.horizon_weeks.eq(h)&event_frame.target.eq(target)&event_frame.evaluation_split.eq(period)].set_index("origin_date")
        matched=group.set_index("origin_date").join(baseline[["probability"]],rsuffix="_base",how="inner")
        metrics.append({"model":model,"horizon_weeks":int(h),"target":target,"evaluation_split":period,"n_predictions":len(group),"event_count":int(y.sum()),
                        "log_loss":float(binary_loss(y,p).mean()),"brier":float(((p-y)**2).mean()),"baseline_model":"markov_endpoint",
                        "delta_log_loss":float((binary_loss(matched.actual,matched.probability)-binary_loss(matched.actual,matched.probability_base)).mean())})
    for (model,h,period),group in frame.loc[frame.actual.notna()&frame.evaluation_split.isin(["selection","retrospective_diagnostic"])].groupby(["model","horizon_weeks","evaluation_split"]):
        group=group.sort_values("origin_date")
        for window,part in (("all_mature_origins",group),("recent_52_mature_origins",group.tail(52))):
          for stratum,population in [("all",part),*[(s,part.loc[part.current_state.eq(s)]) for s in STATE_ORDER]]:
            if population.empty:continue
            current=population.current_state.map({s:i for i,s in enumerate(STATE_ORDER)}).to_numpy();actual=population.actual.map({s:i for i,s in enumerate(STATE_ORDER)}).to_numpy();p=population[PROB].to_numpy(float)
            targets=[(f"endpoint_{s}",actual==i,p[:,i],np.ones(len(p),bool)) for i,s in enumerate(STATE_ORDER)]
            targets.extend([("worsening",actual>current,(p*(np.arange(3)>current[:,None])).sum(axis=1),current<2),
                            ("recovery",actual<current,(p*(np.arange(3)<current[:,None])).sum(axis=1),current>0)])
            for target,y,q,eligible in targets:
                y,q=y[eligible].astype(float),q[eligible]
                if not len(y):continue
                bins=[]
                for indices in np.array_split(np.argsort(q,kind="stable"),max(1,min(5,len(q)//20))):
                    bins.append({"n":len(indices),"mean_probability":float(q[indices].mean()),"observed_rate":float(y[indices].mean()),"events":int(y[indices].sum())})
                ece=sum(b["n"]/len(q)*abs(b["mean_probability"]-b["observed_rate"]) for b in bins)
                bias=float(q.mean()-y.mean());count=int(y.sum());supported=min(count,len(y)-count)>=8
                health.append({"model":model,"horizon_weeks":int(h),"evaluation_split":period,"window":window,"current_state":stratum,"target":target,
                               "n_predictions":len(y),"event_count":count,"nonevent_count":len(y)-count,"mean_probability":float(q.mean()),"observed_rate":float(y.mean()),"calibration_in_the_large":bias,
                               "binary_ece":ece,"brier":float(((q-y)**2).mean()),"status":"insufficient_events" if not supported else "review_due" if abs(bias)>.05 or ece>.08 else "within_descriptive_thresholds","reliability_bins":bins})
    return metrics,event_frame,health


def economic_incremental(canonical,states,frame,protocol=EnhancementProtocol()):
    """Matched-horizon downside forecasts: current state vs direct market vs OOS forecasts.

    The event is a 5% drawdown from origin within h weeks, not label worsening or
    an executable trade. All mapper inputs are saved OOS endpoint predictions.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from regime_lab.analysis.forecast_audit_research import economic_outcomes
    from regime_lab.research.downside import downside_features,paired_block_interval
    outcomes=economic_outcomes(canonical).rename(columns={"target_date":"outcome_target"})
    features=downside_features(canonical).replace([np.inf,-np.inf],np.nan)
    rows=[]
    chosen=["direct_endpoint_ridge","evolving_boundary_gjr_skewt"]
    for h in (4,13):
        forecasts={m:frame.loc[frame.model.eq(m)&frame.horizon_weeks.eq(h)].set_index("origin_date").sort_index() for m in chosen}
        common=forecasts[chosen[0]].index
        if not forecasts[chosen[1]].index.equals(common):raise ValueError("economic forecast origins must match")
        table=pd.DataFrame(index=common)
        table["current_state"]=states.reindex(common)
        for i,state in enumerate(STATE_ORDER):table[f"state_{state}"]=table.current_state.eq(state).astype(float)
        observed=outcomes.loc[outcomes.horizon_weeks.eq(h)].set_index("origin_date")
        table["actual"]=observed.downside_event.reindex(common)
        table["target_date"]=forecasts[chosen[0]].target_date
        for col in features:table[col]=features[col].reindex(common)
        state_columns=[f"state_{s}" for s in STATE_ORDER]
        variants={"downside_state_only":state_columns,"downside_direct_market":state_columns+list(features.columns)}
        for model in chosen:
            for state in ("risk_on","risk_off"):
                key=f"{model}__p_{state}";table[key]=forecasts[model][f"p_{state}"]
            variants[f"downside_state_plus_{model}"]=state_columns+[f"{model}__p_risk_on",f"{model}__p_risk_off"]
        estimators={}
        for i,(origin,record) in enumerate(table.iterrows()):
            past=table.loc[(table.target_date<origin)&table.actual.notna()].tail(protocol.train_window_weeks)
            y=past.actual.to_numpy(float)
            ready=len(past)>=104 and len(np.unique(y))==2 and min(y.sum(),len(y)-y.sum())>=5
            for model,columns in variants.items():
                if ready and (model not in estimators or i%13==0):
                    estimator=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=1000,tol=1e-6))
                    estimator.fit(past[columns],y)
                    estimators[model]=(estimator,past.target_date.max())
                if model in estimators:
                    estimator,last_target=estimators[model];probability=float(estimator.predict_proba(table.loc[[origin],columns])[0,1]);fallback=False
                else:
                    historical=outcomes.loc[(outcomes.horizon_weeks==h)&(outcomes.outcome_target<origin)]
                    same=historical.loc[states.reindex(historical.origin_date).to_numpy()==record.current_state]
                    probability=float((same.downside_event.sum()+1)/(len(same)+2));last_target=historical.outcome_target.max();fallback=True
                rows.append({"model":model,"horizon_weeks":h,"origin_date":origin,"target_date":record.target_date,"current_state":record.current_state,
                             "actual":None if pd.isna(record.actual) else bool(record.actual),"probability":probability,"last_train_target":last_target,"fallback":fallback,
                             "evaluation_split":split(origin,record.target_date,not pd.isna(record.actual))})
    predictions=pd.DataFrame(rows);metrics=[]
    for (model,h,period),group in predictions.loc[predictions.actual.notna()&predictions.evaluation_split.isin(["selection","retrospective_diagnostic"])].groupby(["model","horizon_weeks","evaluation_split"]):
        baseline=predictions.loc[predictions.model.eq("downside_state_only")&predictions.horizon_weeks.eq(h)&predictions.evaluation_split.eq(period)].set_index("origin_date")
        matched=group.set_index("origin_date").join(baseline[["probability"]],rsuffix="_base",how="inner")
        y,q=matched.actual.to_numpy(float),matched.probability.to_numpy(float)
        delta=binary_loss(y,q)-binary_loss(y,matched.probability_base)
        ci=paired_block_interval(delta,block=max(13,int(h)),resamples=protocol.bootstrap_resamples,seed=protocol.seed)
        metrics.append({"model":model,"horizon_weeks":int(h),"evaluation_split":period,"target":"minimum_origin_relative_return_lte_minus_5pct","n_predictions":len(group),"event_count":int(y.sum()),"log_loss":float(binary_loss(y,q).mean()),"brier":float(((y-q)**2).mean()),
                        "baseline_model":"downside_state_only","delta_log_loss":float(delta.mean()),"ci_low":ci[0],"ci_high":ci[1],"fallback_count":int(group.fallback.sum())})
    return {"metrics":metrics,"history":predictions.to_dict("records"),"latest":predictions.loc[predictions.origin_date.eq(predictions.origin_date.max())].to_dict("records"),
            "definition":"5pct decline from origin at any observed weekly close within the same 4/13week horizon; state-only, direct-market, and state-plus-OOS-forecast comparisons use identical origins and strictly matured past targets."}
