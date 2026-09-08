"""Per-model numerical identities for first-issued candidate forecasts."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from regime_lab.integrity import canonical_json_sha256_v1 as digest


def operating_recipe(root: Path, payload: dict, *, config: dict) -> dict:
    """Capture code only where the operating forecast was built in this run."""
    if not isinstance(config,dict) or not isinstance(config.get("model"),dict) or not isinstance(config.get("feature_engineering"),dict):
        raise ValueError("current operating recipe needs the actual build configuration")
    from regime_lab.research.forecast_enhancement_cache import local_source_hashes
    sources = local_source_hashes(root, ["src/regime_lab/analysis/validation.py",
        "src/regime_lab/analysis/structural_models.py", "src/regime_lab/analysis/features.py",
        "src/regime_lab/v5.py", "src/regime_lab/pipeline.py"])
    return {"schema_version": "regime-operating-numerical-recipe/1",
            "source_sha256": sources,
            "runtime": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
            "effective_config_sha256": digest(config),
            "feature_manifest_sha256": payload["model"]["feature_manifest_sha256"],
            "candidate_manifest_sha256": payload["model"]["candidate_manifest_sha256"]}


def candidate_model_recipes(payload: dict, enhancement: dict) -> dict:
    """Exclude changing inputs; isolate archived forecasts with unknown code."""
    provenance = enhancement["provenance"]
    key = provenance["model_cache_key"]
    research = payload.get("research", {})
    model = payload.get("model", {})
    result = {}
    for name in sorted({row["model"] for row in enhancement["latest"]}):
        recipe = {"schema_version": "regime-candidate-model-recipe/2", "model": name,
                  "label_spec_sha256": payload["label"]["spec_sha256"]}
        if name == "causal_dynamic_ensemble":
            manifest_hash = model.get("candidate_manifest_sha256")
            if not manifest_hash or (model.get("candidate_manifest") is not None
                                    and digest(model["candidate_manifest"]) != manifest_hash):
                raise ValueError("operating candidate manifest binding differs")
            recipe.update(candidate_manifest_sha256=manifest_hash,
                          feature_manifest_sha256=model.get("feature_manifest_sha256"),
                          model_version=model.get("version"), feature_set_version=model.get("feature_set_version"),
                          structural_preregistration=model.get("structural_preregistration", {}))
            binding = provenance.get("operating_model_recipe")
            if binding:
                if (binding.get("schema_version") != "regime-operating-numerical-recipe/1"
                        or binding.get("candidate_manifest_sha256") != manifest_hash
                        or binding.get("feature_manifest_sha256") is not None and binding["feature_manifest_sha256"] != model.get("feature_manifest_sha256")
                        or not binding.get("source_sha256") or not binding.get("runtime")):
                    raise ValueError("operating numerical recipe binding differs")
                recipe["numerical_recipe"] = binding
                if not binding.get("effective_config_sha256") or not binding.get("feature_manifest_sha256"):
                    recipe["unversioned_config_generation"] = payload["meta"]["generation_id"]
            else:
                recipe["unversioned_archive_generation"] = payload["meta"]["generation_id"]
        elif name == "boundary_filtered_history":
            baseline = research.get("forecast_improvement", {}).get("provenance", {})
            recipe.update(code_sha256=baseline.get("code_sha256"), runtime=baseline.get("runtime"),
                          protocol=research.get("forecast_research", {}).get("protocol", {}).get("boundary"))
            if not recipe["code_sha256"] or not recipe["runtime"] or not recipe["protocol"]:
                recipe["unversioned_archive_generation"] = payload["meta"]["generation_id"]
        elif name == "directional_duration_hazard":
            baseline = research.get("forecast_research", {})
            recipe.update(publication_recipe_sha256=baseline.get("publication_provenance", {}).get("recipe_sha256"),
                          protocol=baseline.get("protocol", {}).get("hazard"))
            if not recipe["publication_recipe_sha256"] or not recipe["protocol"]:
                recipe["unversioned_archive_generation"] = payload["meta"]["generation_id"]
        else:
            numerical_fields = ("version", "label_fit_weeks", "train_window_weeks", "simulation_power",
                                "path_contamination", "seed", "direct_refit_every_weeks")
            recipe.update(model_sources=key["source_sha256"], runtime=key["runtime"],
                          protocol={k: key["protocol"][k] for k in numerical_fields if k in key["protocol"]})
        result[name] = {"recipe_sha256": digest(recipe), "recipe": recipe}
    return result
