#!/usr/bin/env python3
"""Run every eligible origin against frozen local inputs; never mutate live."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import shutil
import platform

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from regime_lab.analysis.forecast_audit_research import json_safe
from regime_lab.operational_forecast import frame_sha256
from regime_lab.research.forecast_enhancement_models import EnhancementProtocol,run_models
from regime_lab.research.forecast_enhancements import assemble,sha256_file
from regime_lab.research.forecast_enhancement_sensitivity import near_label_sensitivity
from regime_lab.research.forecast_enhancement_cache import cache_matches, legacy_binding_evidence, local_source_hashes, model_cache_key


def write(path,value):
    path.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--payload",type=Path,required=True)
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--refresh-derived",action="store_true",help="reuse validated model caches and atomically replace only this owned output, preserving its prior version")
    parser.add_argument("--upgrade-cache-binding",action="store_true",help="explicitly expand a legacy validated cache binding; record the evidence boundary without numerical refitting")
    args=parser.parse_args()
    if args.upgrade_cache_binding and not args.refresh_derived:raise ValueError("legacy binding expansion requires --refresh-derived")
    if args.output.exists() and not args.refresh_derived:raise ValueError("use a new output directory or explicit --refresh-derived")
    work=args.output.with_name(args.output.name+"-work")
    if args.refresh_derived and args.output.exists() and not work.exists():
        previous=json.loads((args.output/"run-manifest.json").read_text())
        for name in ("model-predictions.pkl","volatility-audit.pkl","model-cache-key.json","label-sensitivity-key.json","label-sensitivity.json","label-sensitivity-predictions.csv","forecast-enhancements.json"):
            if sha256_file(args.output/name)!=previous["files"][name]:raise ValueError("validated model cache changed")
        shutil.copytree(args.output,work)
    if work.exists() and not (args.resume or args.refresh_derived):raise ValueError("existing in-progress work requires explicit --resume")
    work.mkdir(parents=True,exist_ok=True)
    tracked={"canonical":args.input/"canonical.pkl","states":args.input/"states.pkl","input_manifest":args.input/"input-manifest.json","payload":args.payload,
             "baseline":args.audit/"oos-predictions.csv","paths":args.audit/"path-predictions.json",
             "transition":args.artifacts/"transition-oos-predictions.csv","future":args.artifacts/"transition-candidate-forecasts.csv"}
    hashes={key:sha256_file(path) for key,path in tracked.items()}
    protocol=EnhancementProtocol()
    runtime={name:version(name) for name in ("numpy","pandas","scipy","scikit-learn","xgboost","arch","statsmodels")}
    runtime.update({"python":platform.python_version(),"platform":sys.platform,"machine":platform.machine()})
    model_key=json_safe(model_cache_key(ROOT,{k:hashes[k] for k in ("canonical","states")},protocol.record(),runtime))
    sensitivity_code=sha256_file(ROOT/"src/regime_lab/research/forecast_enhancement_sensitivity.py")
    sensitivity_key={"model_key":model_key,"code":sensitivity_code,"diagnostics_code":sha256_file(ROOT/"src/regime_lab/research/forecast_enhancement_diagnostics.py"),
                     "reference_inputs":{name:hashes[name] for name in ("payload","baseline","paths","transition","future")}}
    code_entries=["src/regime_lab/research/"+name for name in ("forecast_enhancement_models.py","forecast_enhancement_diagnostics.py","forecast_enhancements.py","forecast_enhancement_sensitivity.py","forecast_enhancement_cache.py")]
    code_hashes=local_source_hashes(ROOT,code_entries)
    code_hashes.update(model_key["source_sha256"])
    code_hashes[str(Path(__file__).resolve().relative_to(ROOT))]=sha256_file(Path(__file__))
    migrations=[]
    old_key=None
    if (work/"forecast-enhancements.json").exists():
        prior_document=json.loads((work/"forecast-enhancements.json").read_text())
        migrations=list(prior_document.get("provenance",{}).get("cache_binding_migrations",[]))
    if args.upgrade_cache_binding:
        old_key=json.loads((work/"model-cache-key.json").read_text())
        if old_key.get("schema_version")!="forecast-model-cache/2":
            evidence=legacy_binding_evidence(ROOT,old_key,model_key,prior_document["provenance"]["runtime"])
            legacy_sensitivity={"model_key":old_key,"code":sensitivity_code}
            if json.loads((work/"label-sensitivity-key.json").read_text())!=legacy_sensitivity:raise ValueError("legacy sensitivity recipe changed")
            evidence.update({"recorded_at":datetime.now(timezone.utc).isoformat(),"preserved_numerical_file_sha256":{name:sha256_file(work/name) for name in ("model-predictions.pkl","volatility-audit.pkl","label-sensitivity-predictions.csv")}})
            migrations.append(evidence)
            write(work/"model-cache-key.json",model_key)
            write(work/"label-sensitivity-key.json",sensitivity_key)
    write(work/"frozen-protocol.json",{"frozen_at":datetime.now(timezone.utc).isoformat(),"protocol":protocol.record(),"input_hashes":hashes,"model_key":model_key})
    canonical,states=pd.read_pickle(tracked["canonical"]),pd.read_pickle(tracked["states"])
    manifest=json.loads(tracked["input_manifest"].read_text());payload=json.loads(tracked["payload"].read_text())
    if states.index[-1]!=pd.Timestamp(payload["meta"]["data_as_of"]) or states.index[-1]!=canonical.index[-1]:raise ValueError("generation/date mismatch")
    for name,frame in (("canonical",canonical),("states",states)):
        if frame_sha256(frame)!=manifest["frames"][name]:raise ValueError("frozen frame hash mismatch")
    started=time.monotonic()
    if cache_matches(work/"model-cache-key.json",model_key):
        models=pd.read_pickle(work/"model-predictions.pkl");audit=pd.read_pickle(work/"volatility-audit.pkl")
        print(f"Validated recipe cache: {len(models)} model rows",flush=True)
    else:
        models,audit,_=run_models(canonical,states,protocol,progress=lambda message:print(message,flush=True))
        models.to_pickle(work/"model-predictions.pkl");audit.to_pickle(work/"volatility-audit.pkl")
        write(work/"model-cache-key.json",model_key)
    baseline=pd.read_csv(tracked["baseline"]);transition=pd.read_csv(tracked["transition"]);future=pd.read_csv(tracked["future"])
    paths=json.loads(tracked["paths"].read_text())
    provenance={"input_sha256":hashes,"code_sha256":code_hashes,"model_cache_key":model_key,"cache_binding_migrations":migrations,"input_manifest_verified":True,"source_generation_id":payload["meta"]["generation_id"],"runtime":runtime,"existing_payload_unchanged":True,"raw_data_or_api_audit":False}
    print("Assembling causal calibration, alerts, economics and event robustness",flush=True)
    document,frame=assemble(payload,canonical,states,models,audit,baseline,transition,future,paths,protocol,provenance)
    # This derived experiment is rebuilt when its scope or references change;
    # the separate numerical-model cache remains strictly immutable.
    sensitivity_exists=(work/"label-sensitivity-key.json").exists()
    sensitivity_matches=sensitivity_exists and json.loads((work/"label-sensitivity-key.json").read_text())==json_safe(sensitivity_key)
    if sensitivity_exists and not sensitivity_matches and not args.refresh_derived:
        raise ValueError("changed label sensitivity requires explicit --refresh-derived")
    if sensitivity_matches:
        sensitivity=json.loads((work/"label-sensitivity.json").read_text())
    else:
        print("Rebuilding all-label sensitivity from the current validation recipe",flush=True)
        sensitivity,label_rows=near_label_sensitivity(canonical,states,protocol,progress=lambda message:print(message,flush=True),reference_predictions=frame)
        write(work/"label-sensitivity.json",sensitivity)
        label_rows.to_csv(work/"label-sensitivity-predictions.csv",index=False)
        write(work/"label-sensitivity-key.json",sensitivity_key)
    document["robustness"]["label_sensitivity"]=json_safe(sensitivity)
    for row in sensitivity["rows"]:
        if row["spec_id"]=="official" and (row["model"] in ("evolving_boundary_ewma","evolving_boundary_gjr_skewt") or row["model"]=="direct_endpoint_ridge" and row["horizon_weeks"] in (4,13)):
            expected=next(x for x in document["model_metrics"] if x["model"]==row["model"] and x["horizon_weeks"]==row["horizon_weeks"] and x["evaluation_split"]==row["evaluation_split"])
            if row["n_predictions"]!=expected["n_predictions"] or abs(row["log_loss"]-expected["log_loss"])>1e-8:raise ValueError("official-label sensitivity refit differs from final candidate recipe")
    write(work/"forecast-enhancements.json",document)
    frame.to_csv(work/"endpoint-predictions.csv",index=False)
    audit.to_csv(work/"volatility-fits.csv",index=False)
    for section in ("calibration","alerts"):
        pd.DataFrame(document[section]["history"]).to_csv(work/f"{section}-history.csv",index=False)
    for key,path in tracked.items():
        if sha256_file(path)!=hashes[key]:raise RuntimeError(f"source changed during run: {key}")
    for relative,digest in code_hashes.items():
        if sha256_file(ROOT/relative)!=digest:raise RuntimeError("research implementation changed during run")
    write(work/"run-manifest.json",{"status":"validated_local_research","elapsed_seconds":time.monotonic()-started,"model_rows":len(models),"complete_rows":len(frame),"source_hashes":hashes,"files":{p.name:sha256_file(p) for p in work.iterdir() if p.is_file() and p.name!="run-manifest.json"}})
    if args.output.exists():
        backup=args.output.with_name(args.output.name+"-previous-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
        args.output.rename(backup)
    work.rename(args.output)
    print(json.dumps({"output":str(args.output.resolve()),"model_rows":len(models),"elapsed_seconds":time.monotonic()-started}),flush=True)


if __name__=="__main__":main()
