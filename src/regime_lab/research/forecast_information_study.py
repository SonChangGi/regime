"""Reusable optional-information study for CLI and atomic weekly publication."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from regime_lab.data.release_archive import ReleaseRecord
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.research import forecast_new_information as research


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_existing_cboe(directory: Path, key: str) -> research.Snapshot:
    metadata = json.loads((directory / "manifest.json").read_text())["sources"][key]
    if Path(metadata["file"]).name != metadata["file"]:
        raise ValueError("existing snapshot filename must be a basename")
    return research.Snapshot(key, metadata["url"], metadata["sha256"],
                             research.utc(metadata["first_seen_at"]),
                             (directory / metadata["file"]).read_bytes())


def evaluate_stored_block(name: str, history: pd.DataFrame, features: pd.DataFrame,
                          columns: list[str], output: Path, config: dict) -> dict:
    """Once prospective snapshots mature, the same script can score the block.

    Empty or not-yet-mature history is an explicit result, never synthetic zeros.
    Existing boundary probabilities/current state are common to both arms.
    """
    if not columns:
        return {"status": "not_evaluable", "reason": "no historically eligible stored features", "metrics": {}}
    prediction, experiment = research.run_same_origin_ablation(
        history, features, control_columns=[], extra_columns=columns,
        **{k: config[k] for k in ("minimum_training_rows", "regularization_c", "mixture_weight")})
    experiment["evidence"] = "first-seen optional features on derived baseline; not issued prospective model predictions"
    if len(prediction):
        prediction.to_csv(output / f"{name}-oos-predictions.csv", index=False)
    return experiment


def run_information_study(history: pd.DataFrame, *, data_as_of: str, config: dict,
                          output: Path, sources_dir: Path,
                          existing_sources: Path | None = None,
                          offline: bool = False, refresh: bool = False,
                          as_of: str | None = None,
                          bls_web_extract: Path | None = None,
                          ebp_records_path: Path | None = None,
                          source_statuses: dict | None = None,
                          progress: Callable[[str], None] | None = None) -> dict:
    """Compute from verified generation history; optional unavailable data is explicit.

    Source versions remain in sources_dir. The caller owns per-cutoff snapshotting
    and transaction boundaries. No database or official forecast is mutated.
    """
    if offline and refresh:
        raise ValueError("offline and refresh are mutually exclusive")
    history = research.validate_history(history)
    if history.empty or history.target_date.max() > research.utc(data_as_of):
        raise ValueError("information history is empty or extends beyond the generation")
    if config["baseline_model"] != "boundary_filtered_history" or research.utc(config["selection_end"]) != research.SELECTION_END:
        raise ValueError("protocol baseline/cutoff disagrees with implementation")
    output.mkdir(parents=True, exist_ok=True)
    started = pd.Timestamp(datetime.now(timezone.utc))
    summary = {"schema_version": research.SCHEMA_VERSION, "started_at": started,
               "data_as_of": data_as_of, "protocol": config,
               "protocol_sha256": canonical_json_sha256_v1(config),
               "promotion": "none", "blocks": {}, "sources": {}, "experiments": {}}
    snapshots = {}
    requests_to_make = dict(research.SOURCES)
    year = research.utc(as_of).year if as_of else started.year
    for y in (year - 1, year):
        requests_to_make[f"tff_{y}"] = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip"
    for key, url in requests_to_make.items():
        try:
            if source_statuses is not None and source_statuses.get(key, {}).get("status") != "available":
                raise ValueError("source unavailable in the frozen cutoff snapshot")
            if existing_sources and key in ("vix", "vix9d", "vvix"):
                snapshot = read_existing_cboe(existing_sources, key)
                reuse = True
            else:
                cached = research.load_snapshots(sources_dir, key)
                if offline:
                    if not cached:
                        raise ValueError("optional source has no stored snapshot (offline)")
                    snapshot = cached[-1]
                else:
                    snapshot = research.fetch_snapshot(key, url, sources_dir, refresh=refresh)
                reuse = False
            snapshots[key] = snapshot
            summary["sources"][key] = {**snapshot.manifest(), "reused_read_only": reuse, "status": "available"}
        except (ValueError, OSError, KeyError, research.requests.RequestException) as error:
            summary["sources"][key] = {"url": url, "status": "unavailable", "reason": str(error)}
        if progress:
            progress(f'{key}: {summary["sources"][key]["status"]}')
    bls_alternates = []
    if "bls" not in snapshots:
        if not offline:
            try:
                alternate = research.collect_bls_html_fallback(sources_dir, refresh=refresh)
                summary["sources"]["bls_html"] = {**alternate.manifest(), "status": "available"}
            except (ValueError, OSError, research.requests.RequestException) as error:
                summary["sources"]["bls_html"] = {"status": "unavailable", "reason": str(error)}
        if bls_web_extract:
            raw = bls_web_extract.read_bytes()
            # Validate before accepting a durable alternate version.
            probe = research.Snapshot("bls_web_extract", "https://www.bls.gov/schedule/",
                hashlib.sha256(raw).hexdigest(), pd.Timestamp(datetime.now(timezone.utc)), raw)
            research.parse_bls_alternate(probe)
            alternate = research.store_observed_snapshot("bls_web_extract", probe.url, raw, sources_dir)
            summary["sources"]["bls_web_extract"] = {**alternate.manifest(), "status": "available",
                "representation": "official_html_rendered_by_web_tool; not raw HTML HTTP response"}
        for source_id in ("bls_html", "bls_web_extract"):
            bls_alternates.extend(research.load_snapshots(sources_dir, source_id))
        if bls_alternates and "bls_web_extract" not in summary["sources"]:
            summary["sources"]["bls_web_extract"] = {**bls_alternates[-1].manifest(), "status": "available",
                "representation": "stored alternate official calendar representation"}
    now = research.utc(as_of) if as_of else pd.Timestamp(datetime.now(timezone.utc))
    runtime_origins = pd.DatetimeIndex([now])
    summary["runtime_as_of"] = now
    origins = pd.DatetimeIndex(history.origin_date)
    if all(k in snapshots for k in ("vix3m", "vix", "vix9d", "vvix")):
        try:
            features, lineage = research.market_features(snapshots, origins, track="reconstructed_market_prior_day")
            features.to_csv(output / "market-features.csv")
            research.write_json(output / "market-lineage.json", lineage.to_dict("records"))
            pred, experiment = research.run_same_origin_ablation(
                history, features, **{k: config[k] for k in ("control_columns", "extra_columns",
                    "minimum_training_rows", "regularization_c", "mixture_weight")})
            pred.to_csv(output / "oos-predictions.csv", index=False)
            runtime, runtime_lineage = research.market_features(snapshots, runtime_origins)
            summary["experiments"]["vix3m"] = experiment
            summary["blocks"]["vix3m"] = {"status": experiment["status"],
                "reason": "same-origin reconstructed market ablation; no automatic promotion",
                "runtime_features": runtime.reset_index().to_dict("records"),
                "runtime_lineage": runtime_lineage.to_dict("records")}
        except (ValueError, KeyError) as error:
            summary["blocks"]["vix3m"] = {"status": "invalid_optional_block", "reason": str(error)}
    else:
        summary["blocks"]["vix3m"] = {"status": "unavailable", "reason": "new quote or existing control snapshot unavailable"}
    versions = []
    for key, parse in (("fomc", research.parse_fomc_calendar), ("bls", research.parse_bls_calendar)):
        try:
            if key not in snapshots and not (key == "bls" and bls_alternates):
                raise ValueError(summary["sources"][key]["reason"])
            source_versions = research.load_snapshots(sources_dir, key)
            parsed_versions = [parse(s) for s in source_versions]
            if key == "bls":
                parsed_versions.extend(research.parse_bls_alternate(s) for s in bls_alternates)
            versions.extend(parsed_versions)
            research.write_json(output / f"{key}-calendar-versions.json",
                                [v.to_dict("records") for v in parsed_versions])
            historical = research.calendar_features(parsed_versions, origins)
            runtime = research.calendar_features(parsed_versions, runtime_origins)
            summary["blocks"][key] = {"status": "prospective_only", "reason": "historical announcement knowledge not certified; first stored full schedule only",
                "historical_origins_with_features": int(historical.notna().any(axis=1).sum()),
                "stored_versions": len(parsed_versions), "runtime_features": runtime.reset_index().to_dict("records"),
                "alternate_representation": key == "bls" and bool(bls_alternates)}
            summary["experiments"][key] = evaluate_stored_block(key, history, historical,
                [c for c in historical if c.endswith(("_known_events_7d", "_days_to_next_known"))], output, config)
        except (ValueError, KeyError) as error:
            summary["blocks"][key] = {"status": "unavailable", "reason": str(error), "historical_origins_with_features": 0}
    research.write_json(output / "runtime-calendar.json", research.calendar_features(versions, runtime_origins).reset_index().to_dict("records"))
    tff_frames = []
    for key, snapshot in snapshots.items():
        if not key.startswith("tff_"):
            continue
        try:
            for version in research.load_snapshots(sources_dir, key):
                tff_frames.append(research.parse_cftc_tff(version, contract_codes=tuple(config["cftc_contract_codes"])))
        except ValueError as error:
            summary["sources"][key].update({"status": "parse_failed", "reason": str(error)})
    if tff_frames:
        records = pd.concat(tff_frames, ignore_index=True)
        records.to_csv(output / "tff-observations.csv", index=False)
        historical = research.positioning_features(records, origins)
        runtime = research.positioning_features(records, runtime_origins)
        research.write_json(output / "runtime-positioning.json", runtime.reset_index().to_dict("records"))
        summary["blocks"]["cftc_tff"] = {"status": "prospective_only", "reason": "official annual reports have no verified historical actual release timestamps; no Tuesday+3 backdating",
            "parsed_rows": len(records), "historical_origins_with_features": int(historical.notna().any(axis=1).sum()),
            "runtime_features": runtime.reset_index().to_dict("records")}
        summary["experiments"]["cftc_tff"] = evaluate_stored_block("cftc_tff", history, historical,
            [c for c in historical if c.endswith(("_net_oi", "_change_4w", "_percentile_52w"))], output, config)
    else:
        summary["blocks"]["cftc_tff"] = {"status": "unavailable", "reason": "no successfully parsed TFF snapshot"}
    ebp_records = []
    if ebp_records_path:
        for item in json.loads(ebp_records_path.read_text()):
            row = dict(item)
            row["observed_period_end"] = date.fromisoformat(row["observed_period_end"])
            for key in ("source_released_at", "provider_first_seen_at", "system_retrieved_at"):
                row[key] = research.utc(row[key]).to_pydatetime()
            ebp_records.append(ReleaseRecord(**row))
    elif "board_ebp" in snapshots:
        try:
            for snapshot in research.load_snapshots(sources_dir, "board_ebp"):
                ebp_records.extend(research.parse_ebp(snapshot))
        except ValueError as error:
            summary["sources"]["board_ebp"].update({"status": "parse_failed", "reason": str(error)})
    if ebp_records:
        ebp = research.ebp_features(ebp_records, runtime_origins)
        summary["blocks"]["board_ebp"] = {"status": "prospective_only", "reason": "existing ReleaseRecord interface; revised monthly history not backdated",
            "parsed_rows": len(ebp_records), "runtime_features": ebp.reset_index().to_dict("records")}
        if ebp_records_path:
            summary["blocks"]["board_ebp"]["input_sha256"] = digest(ebp_records_path)
        historical = research.ebp_features(ebp_records, origins)
        summary["blocks"]["board_ebp"]["historical_origins_with_features"] = int(historical.notna().any(axis=1).sum())
        research.write_json(output / "runtime-ebp.json", ebp.reset_index().to_dict("records"))
        summary["experiments"]["board_ebp"] = evaluate_stored_block("board_ebp", history, historical,
            [c for c in historical if c == "ebp"], output, config)
    else:
        summary["blocks"]["board_ebp"] = {"status": "unavailable", "reason": summary["sources"].get("board_ebp", {}).get("reason", "no usable release records")}
    summary["completed_at"] = pd.Timestamp(datetime.now(timezone.utc))
    adapter = research.build_forecast_information(summary)
    research.write_json(output / "forecast-information.json", adapter)
    research.write_json(output / "summary.json", summary)
    return research.json_safe(summary)
