"""Complete weekly research before the existing atomic generation publication."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import pandas as pd

from regime_lab.allocation.research_v2 import build_allocation_shadow_v2
from regime_lab.analysis.decision_research import build_decision_research_v2
from regime_lab.forecast_ledger import read_operational_diagnostics
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.operational_forecast import frame_sha256
from regime_lab.research.additional_sources import (
    SOURCES,
    ads_vintage_features,
    cboe_weekly,
    cmdi_history,
    collect_sources,
    read_source,
    sloos_history,
    source_summary,
)
from regime_lab.research.contract import validate_research_extensions
from regime_lab.research.diagnostics import ablation_diagnostics
from regime_lab.research.downside import run_downside_research


def _verify_source_snapshot(directory: Path, cutoff: str) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("research_data_as_of") != cutoff
        or set(manifest.get("sources", {})) != set(SOURCES)
    ):
        raise ValueError("additional research snapshot cutoff or sources differ")
    for key, record in manifest["sources"].items():
        if Path(record["file"]).name != record["file"]:
            raise ValueError("additional research snapshot filename is invalid")
        if len(read_source(directory, key)) != record["bytes"]:
            raise ValueError("additional research snapshot byte count differs")
    return manifest


def _source_snapshot(cache: Path, cutoff: str) -> Path:
    """Refresh once per cutoff; retries reuse one verified immutable snapshot.

    The shared content-addressed blobs preserve first-seen clocks across weeks.
    Snapshot hard links avoid copying unchanged downloads into every week.
    The CLI's existing database lock serializes these generation builds.
    """
    destination = cache / pd.Timestamp(cutoff).date().isoformat()
    if destination.exists():
        _verify_source_snapshot(destination, cutoff)
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    blobs = cache / "blobs"
    manifest = collect_sources(blobs, refresh=True)
    staging = Path(tempfile.mkdtemp(prefix=".source-snapshot-", dir=cache))
    try:
        for key, record in manifest["sources"].items():
            if Path(record["file"]).name != record["file"]:
                raise ValueError("additional research snapshot filename is invalid")
            read_source(blobs, key)
            os.link(blobs / record["file"], staging / record["file"])
        manifest = {**manifest, "research_data_as_of": cutoff}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        _verify_source_snapshot(staging, cutoff)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _additional_research(directory: Path, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    features = cboe_weekly(directory, index)
    summary = source_summary(directory)
    parsed = {
        "ads": ads_vintage_features(read_source(directory, "ads"), index),
        "sloos": sloos_history(read_source(directory, "sloos")),
        "cmdi": cmdi_history(read_source(directory, "cmdi")),
    }
    for row in summary["sources"]:
        if row["id"] in parsed:
            row["status"] = "parsed"
            row["parsed_rows"] = len(parsed[row["id"]])
    return features, summary


def compose_live_publication_research(
    payload: dict,
    *,
    dataset: Any,
    benchmark: Any,
    contract_version: str,
    profile_name: str,
    ledger_path: Path,
    cache_directory: Path,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Return a complete research-enriched copy, leaving official inputs intact.

    Only normal live V5 builds run these studies. Quick/demo/replay consumers
    keep their existing behavior and never touch the network or ledger here.
    Exceptions propagate before the caller publishes a generation or appends
    its current forecast to the ledger.
    """
    if (
        contract_version != "v5"
        or profile_name not in {"standard", "full"}
        or payload.get("meta", {}).get("mode") != "live"
    ):
        return payload
    if payload["model"]["profile"] != profile_name:
        raise ValueError("publication research profile differs from generation")
    result = deepcopy(payload)
    source_payload_sha256 = canonical_json_sha256_v1(payload)
    cutoff = str(payload["meta"]["data_as_of"])
    model = str(payload["model"]["champion"])
    selection_end = str(payload["model"]["selection_end"])
    canonical = dataset.canonical.loc[dataset.canonical.spy_close.notna()].copy()
    if canonical.empty or canonical.index[-1] != pd.Timestamp(cutoff):
        raise ValueError("publication research canonical cutoff differs from generation")
    frames = {
        "canonical": canonical,
        "oos_predictions": benchmark.predictions.copy(deep=True),
        "transition_predictions": benchmark.transition_benchmark.predictions.copy(deep=True),
        "model_conditioned_outcomes": benchmark.model_conditioned_asset_outcomes.copy(deep=True),
        "ablation_predictions": benchmark.feature_ablation.predictions.copy(deep=True),
    }
    frame_identity = {
        name: {"sha256": frame_sha256(frame), "rows": len(frame)}
        for name, frame in frames.items()
    }
    predictions = frames["oos_predictions"]
    expected = int((predictions.model.eq(model) & predictions.evaluation_split.eq("holdout")).sum())
    if expected < 1:
        raise ValueError("publication research requires matched holdout origins")
    durations: dict[str, float] = {}

    def run(label: str, function: Callable[[], Any]) -> Any:
        if progress is not None:
            progress(f"확장 연구: {label}")
        started = time.monotonic()
        value = function()
        durations[label] = round(time.monotonic() - started, 3)
        return value

    source_directory = run("추가 자료 스냅샷", lambda: _source_snapshot(cache_directory, cutoff))
    additional, sources = run("추가 자료 해석", lambda: _additional_research(source_directory, canonical.index))
    shadow = result["research"]["prospective_decision_shadow"]
    allocation = run("4주 배분 비교", lambda: build_allocation_shadow_v2(
        result["weekly"], canonical, predictions,
        forecast_model=model, selection_end=selection_end,
        current_signal=shadow["current_signal"],
    ))
    decision = run("조기 경보와 손익", lambda: build_decision_research_v2(
        predictions, frames["transition_predictions"], canonical,
        forecast_model=model, selection_end=selection_end,
        outcome_rows=frames["model_conditioned_outcomes"],
    ))
    downside = run("SPY 하방 시나리오", lambda: run_downside_research(
        canonical, additional=additional, selection_end=selection_end,
    )).summary
    diagnostics = run("특성별 기여", lambda: ablation_diagnostics(frames["ablation_predictions"]))
    # The current issuance is appended only after atomic publication succeeds.
    operational = run("실제 발행 기록", lambda: read_operational_diagnostics(ledger_path))
    if allocation["performance"]["weeks"] != expected or decision["diagnostic_origins"] != expected:
        raise ValueError("publication research lost matched holdout origins")
    if downside.get("as_of") != cutoff:
        raise ValueError("publication downside origin differs from generation")
    protected = ("meta", "model", "selection", "forecast", "weekly")
    if any(result.get(key) != payload.get(key) for key in protected) or shadow["current_signal"] != payload["research"]["prospective_decision_shadow"]["current_signal"]:
        raise ValueError("publication research changed official generation decisions")
    result["research"]["decision_research_v2"] = decision
    result["research"]["operational_diagnostics"] = operational
    shadow["allocation_research_v2"] = allocation
    extensions = {
        "schema_version": "regime-research-extensions/1",
        "downside": downside,
        "diagnostics": diagnostics,
        "additional_data": sources,
    }
    extensions["build"] = {
        "schema_version": "regime-publication-research-build/1",
        "purpose": "weekly_publication_research",
        "generation_id": payload["meta"]["generation_id"],
        "data_as_of": cutoff,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "source_payload_sha256": source_payload_sha256,
        "source_payload_hash_basis": "canonical_json_sha256_v1_before_research_enrichment",
        "forecast_model": model,
        "selection_end": selection_end,
        "matched_holdout_origins": expected,
        "input_frames": frame_identity,
        "additional_source_manifest_sha256": hashlib.sha256((source_directory / "manifest.json").read_bytes()).hexdigest(),
        "operational_scope": "issued_ledger_before_current_generation_publication",
        "operational_as_of": operational["as_of"],
        "issued_forecast_preserved": True,
        "phase_seconds": durations,
        "outputs": {name: canonical_json_sha256_v1(value) for name, value in {
            "allocation_research_v2": allocation,
            "decision_research_v2": decision,
            "operational_diagnostics": operational,
            "downside": downside,
            "diagnostics": diagnostics,
            "additional_data": sources,
        }.items()},
    }
    result["research"]["extensions"] = extensions
    validate_research_extensions(result["research"])
    return result
