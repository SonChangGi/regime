"""Assemble validated, local-only forecasting enhancement evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from regime_lab.analysis.forecast_audit_research import AuditResearchProtocol, economic_validation, json_safe
from regime_lab.analysis.forecast_research_evaluation import per_week_scores
from regime_lab.schema import STATE_ORDER
from .forecast_enhancement_models import EnhancementProtocol, MODEL_LABELS, PROB, row_for
from .forecast_enhancement_diagnostics import adaptive_calibration, robustness, summarize_models, event_and_calibration_metrics, economic_incremental
from regime_lab.research.forecast_alert_policy import alert_policies
from .forecast_enhancement_sensitivity import label_explanations


def public_forecast(row):
    row = dict(row)
    if "simulation_standard_error_max" in row:
        reference_error = row.pop("simulation_standard_error_max")
        if pd.notna(reference_error):
            row["simulation_iid_reference_standard_error_max"] = reference_error
            row["simulation_error_interpretation"] = "sqrt(0.25/draws) is an IID reference scale, not a confidence interval for the scrambled Sobol estimate"
    p = np.array([row.pop(key) for key in PROB], float)
    current = STATE_ORDER.index(row["current_state"])
    row.update({"probabilities": dict(zip(STATE_ORDER, p)),
                "endpoint_change_probability": float(1-p[current]),
                "worsening_probability": float(p[current+1:].sum()),
                "recovery_probability": float(p[:current].sum())})
    return json_safe(row)


def calibration_inputs(payload, baseline, transition, future):
    all_rows = pd.concat([transition, future], ignore_index=True)
    for name in ("origin_date", "target_end"):
        all_rows[name] = pd.to_datetime(all_rows[name], utc=True, format="mixed")
    champions = {h: payload["weekly"][-1]["transition_risk"][f"{h}w"]["model"] for h in (4,13)}
    selected = {h: all_rows.loc[all_rows.horizon.eq(h)&all_rows.model.eq(champions[h])].set_index("origin_date") for h in (4,13)}
    if any(x.index.has_duplicates for x in selected.values()):
        raise ValueError("duplicate transition calibration inputs")
    anchor = baseline.loc[baseline.model.eq(payload["model"]["champion"])].copy()
    anchor.origin_date = pd.to_datetime(anchor.origin_date,utc=True,format="mixed")
    anchors = {r.origin_date: 1-getattr(r,f"p_{r.current_state}") for r in anchor.itertuples()}
    for week in payload["weekly"]:
        anchors[pd.Timestamp(week["data_as_of"])] = float(week["transition_risk"]["1w"]["probability"])
    from .forecast_enhancement_diagnostics import coherent_departure
    rows=[]
    for origin in sorted(set(selected[4].index)&set(selected[13].index)&set(anchors)):
        a,b = selected[4].loc[origin],selected[13].loc[origin]
        stored=coherent_departure(anchors[origin],float(a.p_change),float(b.p_change))
        rows.append({"origin_date":origin,"anchor":anchors[origin],"raw_4":float(a.raw_p_change),"raw_13":float(b.raw_p_change),
                     "stored_4":stored[0],"stored_13":stored[1],"end_4":a.target_end,"end_13":b.target_end,
                     "y_4":a.actual_change,"y_13":b.actual_change})
    return pd.DataFrame(rows)


def add_existing_paths(frame, states, path_rows):
    rows=[]
    for item in path_rows:
        if item["model"]!="directional_duration_hazard":continue
        origin=pd.Timestamp(item["origin_date"])
        position=int(states.index.get_loc(origin))
        rows.append(row_for(states,position,int(item["horizon_weeks"]),item["model"],[item["endpoint"][s] for s in STATE_ORDER],last_train_target=states.index[position-1],
                            first_departure_probability=1-item["first_departure"]["no_departure"],risk_off_entry_probability=item["any_risk_off_entry"],risk_off_occupancy_probability=item["any_risk_off_occupancy"]))
    return pd.concat([frame,pd.DataFrame(rows)],ignore_index=True)


def assemble(payload, canonical, states, models, volatility_audit, baseline, transition, future, path_rows, protocol=EnhancementProtocol(), provenance=None):
    frame=add_existing_paths(models,states,path_rows)
    # Existing one-week finalists share the identical frozen labels and dates.
    existing=baseline.loc[baseline.model.isin(["causal_dynamic_ensemble","boundary_filtered_history"])].copy()
    for column in ("origin_date","target_date"):
        existing[column]=pd.to_datetime(existing[column],utc=True,format="mixed")
    existing["horizon_weeks"]=1;existing["target"]="endpoint"
    existing.evaluation_split=existing.evaluation_split.replace({"holdout":"retrospective_diagnostic"})
    latest_existing=[]
    last=payload["weekly"][-1]
    latest_existing.append(row_for(states,len(states)-1,1,"causal_dynamic_ensemble",[last["next_week"]["probabilities"][s] for s in STATE_ORDER]))
    for model in payload["research"]["forecast_improvement"]["models"]:
        if model["id"]=="boundary_filtered_history":
            value=model["latest"]
            if pd.Timestamp(value["origin_date"])!=states.index[-1]:raise ValueError("saved boundary latest origin is stale")
            latest_existing.append(row_for(states,len(states)-1,1,model["id"],[value["probabilities"][s] for s in STATE_ORDER]))
    existing=pd.concat([existing,pd.DataFrame(latest_existing)],ignore_index=True)
    frame=pd.concat([frame,existing],ignore_index=True)
    if frame.duplicated(["origin_date","horizon_weeks","model"]).any():
        raise ValueError("duplicate endpoint evidence")
    for column in ("origin_date","target_date"):
        frame[column]=pd.to_datetime(frame[column],utc=True,format="mixed")
    frame["fallback"]=frame.fallback.fillna(False).astype(bool)
    metrics,scored=summarize_models(frame)
    event_metrics,event_history,calibration_health=event_and_calibration_metrics(frame,states)
    calibration=adaptive_calibration(calibration_inputs(payload,baseline,transition,future),protocol)
    one=frame.loc[frame.horizon_weeks.eq(1)].copy()
    p=one[PROB].to_numpy(float); current=one.current_state.map({s:i for i,s in enumerate(STATE_ORDER)}).to_numpy()
    one["worsening_probability"]=(p*(np.arange(3)>current[:,None])).sum(axis=1)
    alerts=alert_policies(one,protocol)
    mature_one=one.loc[one.actual.notna()&one.evaluation_split.isin(["selection","retrospective_diagnostic"])].copy()
    mature_one.evaluation_split=mature_one.evaluation_split.replace({"retrospective_diagnostic":"holdout"})
    economic=economic_validation(per_week_scores(mature_one),canonical,AuditResearchProtocol())
    robust=robustness(scored,states,protocol)
    history=[public_forecast(row) for row in frame.sort_values(["origin_date","model","horizon_weeks"]).to_dict("records")]
    latest=[r for r in history if pd.Timestamp(r["origin_date"])==states.index[-1]]
    document={"schema_version":"regime-forecast-enhancements/1","status":"validated_local_research",
              "data_as_of":payload["meta"]["data_as_of"],"source_generation_id":payload["meta"]["generation_id"],
              "generated_at":datetime.now(timezone.utc).isoformat(),"automatic_promotion":False,
              "protocol":protocol.record(),"model_labels":MODEL_LABELS,"model_metrics":metrics,
              "label_explanations":label_explanations(canonical,states,protocol.label_fit_weeks),
              "event_metrics":event_metrics,"event_history":event_history.to_dict("records"),"calibration_health":calibration_health,
              "history":history,"latest":latest,"calibration":calibration,"alerts":alerts,
              "economics":{"rows":economic,"incremental":economic_incremental(canonical,states,frame,protocol),"definition":"Retains current state. Risk-off worsening is structurally zero, never a low absolute-risk score. Pooled rows retained only as descriptive compatibility evidence."},
              "robustness":robust,"volatility_fit_audit":volatility_audit.to_dict("records"),"provenance":provenance or {}}
    document=json_safe(document)
    validate_enhancements(document)
    return document,frame


def validate_enhancements(document):
    if document.get("schema_version")!="regime-forecast-enhancements/1" or document.get("automatic_promotion") is not False:
        raise ValueError("invalid enhancement identity/promotion boundary")
    json.dumps(document,allow_nan=False)
    seen=set()
    for row in document["history"]:
        key=(row["model"],row["origin_date"],row["horizon_weeks"])
        if key in seen:raise ValueError("duplicate forecast history key")
        seen.add(key)
        origin,target=pd.Timestamp(row["origin_date"]),pd.Timestamp(row["target_date"])
        if target<=origin:raise ValueError("target must follow origin")
        expected=(origin.tz_convert("America/New_York")+pd.DateOffset(weeks=row["horizon_weeks"])).tz_convert("UTC")
        if target!=expected:raise ValueError("horizon/date mismatch")
        p=np.array(list(row["probabilities"].values()),float)
        if len(p)!=3 or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.isclose(p.sum(),1,atol=1e-8):raise ValueError("invalid probabilities")
        train=row.get("last_train_target")
        if train and pd.Timestamp(train)>=origin:raise ValueError("future target in training")
        if row["horizon_weeks"]==1 and row.get("first_departure_probability") is not None and not np.isclose(row["first_departure_probability"],row["endpoint_change_probability"],atol=1e-8):raise ValueError("one-week path/endpoint identity failed")
    for audit in document["calibration"]["audit"]:
        for target in [*audit["last_fit_targets"],audit["last_validation_target"]]:
            if target and pd.Timestamp(target)>=pd.Timestamp(audit["origin_date"]):raise ValueError("calibration consumed an unresolved target")
        if audit["selected_weight"]:
            check=next(x for x in audit["checks"] if x["weight"]==audit["selected_weight"])
            if not check["passed"] or not all(x["passed"] for x in check["horizons"]):raise ValueError("adaptive calibration failed horizon-specific no-harm gate")
    for row in document["alerts"]["history"]:
        if row["last_policy_target"] and pd.Timestamp(row["last_policy_target"])>=pd.Timestamp(row["origin_date"]):raise ValueError("alert threshold consumed unresolved outcome")
    for row in document["economics"]["incremental"]["history"]:
        if row["last_train_target"] and pd.Timestamp(row["last_train_target"])>=pd.Timestamp(row["origin_date"]):raise ValueError("economic mapper consumed unresolved outcome")


def sha256_file(path):
    return sha256(path.read_bytes()).hexdigest()
