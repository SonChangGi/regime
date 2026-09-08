"""Optional, reproducible comparison generation from the current weekly inputs."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import pandas as pd

from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256
from regime_lab.research.forecast_enhancement_cache import cache_matches, local_source_hashes, model_cache_key
from regime_lab.research.forecast_enhancement_models import EnhancementProtocol, run_models
from regime_lab.research.forecast_enhancement_sensitivity import near_label_sensitivity
from regime_lab.research.forecast_enhancements import assemble, sha256_file, validate_enhancements
from regime_lab.analysis.forecast_audit_research import json_safe
from regime_lab.schema import STATE_ORDER


def _write(path: Path, document) -> None:
    from regime_lab.io import write_json_atomic
    write_json_atomic(path, json_safe(document))


def publication_baselines(payload: dict, states: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """Use the current generation's actual frozen one-week/path histories."""
    rows, paths = [], []
    models = payload["research"]["forecast_research"]["models"]
    for model in models:
        name = model["id"]
        if name in ("causal_dynamic_ensemble", "boundary_filtered_history"):
            for row in model["history"]:
                origin = pd.Timestamp(row["origin_date"])
                position = states.index.get_loc(origin)
                if position + 1 >= len(states):
                    continue
                target = states.index[position+1]
                period = "selection" if target < pd.Timestamp("2023-01-01", tz="UTC") else "holdout"
                rows.append({"model": name, "origin_date": origin, "target_date": target,
                             "current_state": row["current_state"], "actual": states.iloc[position+1],
                             "evaluation_split": period,
                             **{f"p_{state}": row["next_state"][state] for state in STATE_ORDER}})
        elif name == "directional_duration_hazard":
            for row in [*model["history"], *([model["latest"]] if model.get("latest") else [])]:
                for horizon in row.get("horizons", {}).values():
                    paths.append({**horizon, "model": name, "origin_date": row["origin_date"]})
    if {row["model"] for row in rows} != {"causal_dynamic_ensemble", "boundary_filtered_history"}:
        raise ValueError("weekly generation lacks current frozen comparison baselines")
    if not paths:
        raise ValueError("weekly generation lacks its frozen directional path forecasts")
    return pd.DataFrame(rows), paths


def build_forecast_enhancement_candidate(payload: dict, *, dataset, benchmark,
                                          cache_directory: Path, progress=None, operating_built_current: bool = False,
                                          operating_config: dict | None = None) -> dict:
    """Run all eligible origins under an explicit opt-in; reuse exact numerical fits.

    No network, live ledger, publication path or schedule is touched. A new cutoff
    binds new canonical/state inputs; an old sidecar is never relabelled current.
    """
    protocol = EnhancementProtocol()
    canonical = dataset.canonical.loc[dataset.canonical.spy_close.notna()].copy()
    labels = benchmark.state_label_history
    states = pd.Series(labels["state"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(labels["date"], utc=True))).reindex(canonical.index)
    if states.isna().any() or states.index[-1] != pd.Timestamp(payload["meta"]["data_as_of"]):
        raise ValueError("weekly enhancement inputs differ from the current generation")
    root = Path(__file__).resolve().parents[3]
    runtime = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "arch", "statsmodels")}
    runtime.update(python=platform.python_version(), platform=sys.platform, machine=platform.machine())
    entries = ["src/regime_lab/research/"+name for name in (
        "forecast_enhancement_generation.py", "forecast_enhancements.py", "forecast_enhancement_models.py",
        "forecast_enhancement_diagnostics.py", "forecast_enhancement_sensitivity.py")]
    sources = local_source_hashes(root, entries)
    inputs = {"canonical": frame_sha256(canonical), "states": frame_sha256(states)}
    key = json_safe(model_cache_key(root, inputs, protocol.record(), runtime))
    numeric = cache_directory/"models"/canonical_json_sha256_v1(key)
    numeric.mkdir(parents=True, exist_ok=True)
    if cache_matches(numeric/"key.json", key):
        receipt = json.loads((numeric/"files.json").read_text())
        if any(sha256_file(numeric/name) != digest for name, digest in receipt.items()):
            raise ValueError("weekly comparison numerical cache changed")
        models, audit = pd.read_pickle(numeric/"predictions.pkl"), pd.read_pickle(numeric/"volatility.pkl")
    else:
        if progress: progress("기간별 국면·동적 경로 비교: 모든 적격 시점 계산")
        models, audit, _ = run_models(canonical, states, protocol, progress=progress)
        models.to_pickle(numeric/"predictions.pkl"); audit.to_pickle(numeric/"volatility.pkl")
        _write(numeric/"files.json", {name: sha256_file(numeric/name) for name in ("predictions.pkl", "volatility.pkl")})
        _write(numeric/"key.json", key)
    baseline, paths = publication_baselines(payload, states)
    transition = benchmark.transition_benchmark.predictions.copy(deep=True)
    future = benchmark.transition_benchmark.latest_candidate_forecasts().copy(deep=True)
    provenance = {"source_generation_id": payload["meta"]["generation_id"], "input_frames": inputs,
                  "state_frame_schema": {"series_name": states.name, "index_name": states.index.name},
                  "source_payload_sha256": canonical_json_sha256_v1(payload), "runtime": runtime,
                  "code_sha256": sources, "model_cache_key": key,
                  "generated_from_current_weekly_inputs": True, "network_access": False}
    if operating_built_current:
        from regime_lab.candidate_recipes import operating_recipe
        provenance["operating_model_recipe"] = operating_recipe(root,payload,config=operating_config)
    document, frame = assemble(payload, canonical, states, models, audit, baseline, transition, future, paths, protocol, provenance)
    sensitivity_key = {"model_key": key, "source": sources["src/regime_lab/research/forecast_enhancement_sensitivity.py"],
                       "diagnostics": sources["src/regime_lab/research/forecast_enhancement_diagnostics.py"],
                       "reference_payload_sha256": canonical_json_sha256_v1(payload)}
    sensitivity_path = cache_directory/("label-sensitivity-"+canonical_json_sha256_v1(sensitivity_key)+".json")
    if sensitivity_path.exists():
        receipt = json.loads(sensitivity_path.read_text())
        sensitivity = receipt["document"]
        if receipt["key"] != sensitivity_key or receipt["document_sha256"] != canonical_json_sha256_v1(sensitivity):
            raise ValueError("weekly comparison label sensitivity cache changed")
    else:
        sensitivity, _ = near_label_sensitivity(canonical, states, protocol, progress=progress, reference_predictions=frame)
        sensitivity = json_safe(sensitivity)
        _write(sensitivity_path, {"key": sensitivity_key, "document": sensitivity, "document_sha256": canonical_json_sha256_v1(sensitivity)})
    document["robustness"]["label_sensitivity"] = json_safe(sensitivity)
    for name, digest in sources.items():
        if sha256_file(root/name) != digest:
            raise ValueError("weekly comparison source changed during generation")
    if inputs != {"canonical": frame_sha256(canonical), "states": frame_sha256(states)}:
        raise ValueError("weekly comparison inputs changed during generation")
    validate_enhancements(document)
    return document
