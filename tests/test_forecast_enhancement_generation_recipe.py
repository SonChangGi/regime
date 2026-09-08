"""Archived payloads cannot acquire the current operating-code provenance."""
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from regime_lab import candidate_recipes
from regime_lab.research import forecast_enhancement_generation as generation
from regime_lab.integrity import canonical_json_sha256_v1


@pytest.mark.parametrize("current_build", [False, True])
def test_only_current_weekly_build_records_operating_recipe(tmp_path,monkeypatch,current_build):
    dates=pd.date_range("2024-01-05",periods=3,freq="W-FRI",tz="UTC")
    canonical=pd.DataFrame({"spy_close":[100.,101.,102.]},index=dates)
    labels=pd.DataFrame({"date":dates,"state":["risk_on"]*3})
    transition=SimpleNamespace(predictions=pd.DataFrame(),latest_candidate_forecasts=lambda:pd.DataFrame())
    benchmark=SimpleNamespace(state_label_history=labels,transition_benchmark=transition)
    payload={"meta":{"generation_id":"archived-generation","data_as_of":dates[-1].isoformat()}}
    monkeypatch.setattr(generation,"run_models",lambda *a,**k:(pd.DataFrame(),pd.DataFrame(),None))
    monkeypatch.setattr(generation,"publication_baselines",lambda *a:(pd.DataFrame(),[]))
    monkeypatch.setattr(generation,"assemble",lambda *a:({"provenance":a[-1],"robustness":{}},pd.DataFrame()))
    monkeypatch.setattr(generation,"near_label_sensitivity",lambda *a,**k:({"rows":[]},pd.DataFrame()))
    monkeypatch.setattr(generation,"validate_enhancements",lambda *a:None)
    calls=[]
    config={"model":{"custom_window":104},"feature_engineering":{"lags":[1,4]}}
    def operating_recipe(root,value,*,config):
        calls.append(value["meta"]["generation_id"])
        return {"built_now":True,"config":config}
    monkeypatch.setattr(candidate_recipes,"operating_recipe",operating_recipe)
    kwargs={"operating_built_current":True,"operating_config":config} if current_build else {}
    result=generation.build_forecast_enhancement_candidate(payload,dataset=SimpleNamespace(canonical=canonical),benchmark=benchmark,cache_directory=tmp_path,**kwargs)
    assert ("operating_model_recipe" in result["provenance"])==current_build
    assert calls==(["archived-generation"] if current_build else [])
    if current_build:
        assert result["provenance"]["operating_model_recipe"]["config"]==config


def test_actual_custom_configuration_changes_operating_identity_without_exporting_values():
    root=Path(__file__).resolve().parents[1]
    payload={"model":{"candidate_manifest_sha256":"a"*64,"feature_manifest_sha256":"b"*64}}
    config={"model":{"window":520},"feature_engineering":{"lags":[1,4]},"alfred":{"series":["test-series"]}}
    first=candidate_recipes.operating_recipe(root,payload,config=config)
    assert first["effective_config_sha256"]==canonical_json_sha256_v1(config)
    assert first["feature_manifest_sha256"]=="b"*64
    assert "config" not in first
    for section,key,value in (("model","window",260),("feature_engineering","lags",[1,2,4]),("alfred","series",["other-series"])):
        changed=deepcopy(config);changed[section][key]=value
        assert candidate_recipes.operating_recipe(root,payload,config=changed)["effective_config_sha256"]!=first["effective_config_sha256"]
    with pytest.raises(ValueError,match="actual build configuration"):
        candidate_recipes.operating_recipe(root,payload,config=None)
