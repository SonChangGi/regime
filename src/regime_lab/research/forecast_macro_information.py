"""Public macro signals with separate first-seen and reconstructed timelines."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import calendar
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from .forecast_new_information import (Snapshot, json_safe, load_boundary_history,
                                       run_same_origin_ablation, write_json)

SOURCES = {
    "cleveland_inflation": {
        "label": "Cleveland Fed 물가 Nowcast", "unit": "% MoM",
        "url": "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting",
        "download": "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json?sc_lang=en",
    },
    "nyfed_growth": {
        "label": "NY Fed 성장 Nowcast", "unit": "% QoQ SAAR",
        "url": "https://www.newyorkfed.org/research/policy/nowcast/",
        "download": "https://www.newyorkfed.org/medialibrary/Research/Interactives/Data/NowCast/downloads/New-York-Fed-Staff-Nowcast_download_data.xlsx",
    },
    "reserve_elasticity": {
        "label": "NY Fed 준비금 수요 탄력성", "unit": "bp / 은행자산 1%p",
        "url": "https://www.newyorkfed.org/research/reserve-demand-elasticity/",
        "download": "https://www.newyorkfed.org/medialibrary/research/interactives/data/elasticity/elasticity-data.csv",
    },
}
FEATURE_LABELS = {"cpi_mom": "CPI", "core_cpi_mom": "근원 CPI", "pce_mom": "PCE", "core_pce_mom": "근원 PCE",
                  "gdp_nowcast": "이번 분기 성장률", "gdp_next_quarter": "다음 분기 성장률",
                  "reserve_elasticity": "준비금 수요 탄력성", "reserve_lower95": "95% 하한", "reserve_upper95": "95% 상한"}


def _create_once(path: Path, raw: bytes) -> bool:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".snapshot-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
    try:
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)


def _read_snapshot(source: str, directory: Path, meta: dict) -> Snapshot:
    sha = meta.get("sha256", "")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("invalid snapshot identity")
    receipt = json.loads((directory / f"{sha}.json").read_text())
    fields = ("source_id", "url", "sha256", "retrieved_at", "bytes")
    if any(meta.get(key) != receipt.get(key) for key in fields):
        raise ValueError("snapshot pointer differs from its immutable first-seen receipt")
    if receipt["source_id"] != source or receipt["url"] != SOURCES[source]["download"]:
        raise ValueError("snapshot source identity differs")
    raw = (directory / f"{sha}.blob").read_bytes()
    result = Snapshot(source, receipt["url"], sha, pd.Timestamp(receipt["retrieved_at"]), raw)
    if receipt["bytes"] != len(raw) or result.retrieved_at > pd.Timestamp.now(tz="UTC"):
        raise ValueError("snapshot size or first-seen clock is invalid")
    return result


def snapshot(source: str, directory: Path, *, refresh: bool = False) -> Snapshot:
    """Content-addressed downloads; refresh never changes the first-seen time."""
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / f"{source}.json"
    if pointer.exists() and not refresh:
        meta = json.loads(pointer.read_text())
        return _read_snapshot(source, directory, meta)
    url = SOURCES[source]["download"]
    response = requests.get(url, timeout=(15, 60), stream=True)
    response.raise_for_status()
    if urlparse(response.url).hostname not in {"www.clevelandfed.org", "www.newyorkfed.org"}:
        raise ValueError("unexpected public source redirect")
    chunks, size = [], 0
    for chunk in response.iter_content(65536):
        size += len(chunk)
        if size > 25_000_000:
            raise ValueError("public snapshot exceeds size bound")
        chunks.append(chunk)
    raw = b"".join(chunks)
    sha = hashlib.sha256(raw).hexdigest()
    first_seen = directory / f"{sha}.json"
    now = pd.Timestamp(datetime.now(timezone.utc))
    result = Snapshot(source, url, sha, now, raw)
    parsers = {"cleveland_inflation": parse_inflation, "nyfed_growth": parse_growth,
               "reserve_elasticity": parse_reserves}
    if parsers[source](result).empty:
        raise ValueError("public macro snapshot has no usable observations")
    blob = directory / f"{sha}.blob"
    if not _create_once(blob, raw) and blob.read_bytes() != raw:
        raise ValueError("immutable snapshot content changed")
    _create_once(first_seen, (json.dumps(result.manifest(), ensure_ascii=False, sort_keys=True) + "\n").encode())
    receipt = json.loads(first_seen.read_text())
    result = _read_snapshot(source, directory, receipt)
    write_json(pointer, result.manifest())
    return result


def _record(snap: Snapshot, feature: str, value: float, observed: pd.Timestamp,
            reconstructed: pd.Timestamp, target: str = "") -> dict:
    observed, reconstructed = pd.Timestamp(observed), pd.Timestamp(reconstructed)
    if observed.tzinfo is None or reconstructed.tzinfo is None or not np.isfinite(value):
        raise ValueError("finite value and explicit timezones required")
    if observed > snap.retrieved_at or reconstructed < observed:
        raise ValueError("macro observation is in the future or its reconstructed release precedes observation")
    return {"source": snap.source_id, "feature": feature, "value": float(value),
            "observed_at": observed.tz_convert("UTC"), "target_period": target,
            "reconstructed_available_at": reconstructed.tz_convert("UTC"),
            "available_at": snap.retrieved_at,
            "first_seen_at": snap.retrieved_at, "sha256": snap.sha256}


def parse_inflation(snap: Snapshot) -> pd.DataFrame:
    names = {"CPI Inflation": "cpi_mom", "Core CPI Inflation": "core_cpi_mom",
             "PCE Inflation": "pce_mom", "Core PCE Inflation": "core_pce_mom"}
    rows = []
    for chart in json.loads(snap.raw):
        year, month = map(int, chart["chart"]["subcaption"].split("-"))
        target = pd.Timestamp(year=year, month=month, day=1)
        labels = [c["label"] for c in chart["categories"][0]["category"] if c.get("vline") != "true"]
        for series in chart["dataset"]:
            if series["seriesname"] not in names:  # Never ingest the Actual series.
                continue
            if len(labels) != len(series["data"]):
                raise ValueError("inflation archive chart alignment changed")
            for label, point in zip(labels, series["data"]):
                if point.get("value") in ("", None):
                    continue
                m, d = map(int, label.split("/"))
                candidates = [pd.Timestamp(year=y, month=m, day=d) for y in (year-1, year, year+1)
                              if d <= calendar.monthrange(y, m)[1]]
                date = min(candidates, key=lambda t: abs((t-target).days))
                if abs((date-target).days) > 183:
                    raise ValueError("ambiguous inflation archive date")
                # The dated archive is a reconstructed timeline; use the next day.
                date = date.tz_localize("America/New_York")
                if date.tz_convert("UTC") > snap.retrieved_at:
                    continue
                rows.append(_record(snap, names[series["seriesname"]], float(point["value"]),
                                    date, date + pd.DateOffset(days=1), f"{year:04d}-{month:02d}"))
    return pd.DataFrame(rows)


def parse_growth(snap: Snapshot) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(snap.raw), sheet_name="Forecasts By Horizon", header=None)
    header = next(i for i in raw.index if str(raw.iloc[i, 0]).strip().lower() == "forecast date")
    rows = []
    for values in raw.iloc[header+1:].itertuples(index=False, name=None):
        if not isinstance(values[0], (datetime, pd.Timestamp)):
            continue
        observed = pd.Timestamp(values[0]).tz_localize("America/New_York")
        # The current model resumed public releases in September 2023.
        if observed < pd.Timestamp("2023-09-08", tz="America/New_York"):
            continue
        for column, name in ((3, "gdp_nowcast"), (4, "gdp_next_quarter")):
            if pd.notna(values[column]):
                rows.append(_record(snap, name, float(values[column]), observed,
                                    observed + pd.DateOffset(days=1), str(values[1])))
    return pd.DataFrame(rows)


def parse_reserves(snap: Snapshot, lag_days: int = 30) -> pd.DataFrame:
    if lag_days < 1:
        raise ValueError("reserve diagnostic lag must be positive")
    raw = pd.read_csv(io.BytesIO(snap.raw))
    names = {"Elasticity - 50th percentile (main)": "reserve_elasticity",
             "Elasticity - 2.5th percentile": "reserve_lower95",
             "Elasticity - 97.5th percentile": "reserve_upper95"}
    rows = []
    for _, row in raw.iterrows():
        date = pd.Timestamp(row["Date"]).tz_localize("America/New_York")
        for column, name in names.items():
            if pd.notna(row[column]):
                rows.append(_record(snap, name, float(row[column]), date,
                                    date + pd.DateOffset(days=lag_days)))
    return pd.DataFrame(rows)


def align_features(records: pd.DataFrame, origins: pd.DatetimeIndex, *,
                   reconstructed: bool = False, max_age_days: int = 95) -> pd.DataFrame:
    """No forward lookup; target-quarter/month changes never rewrite prior rows."""
    if origins.tz is None or origins.has_duplicates:
        raise ValueError("origins must be unique and timezone-aware")
    timeline = "reconstructed_available_at" if reconstructed else "available_at"
    result = pd.DataFrame(index=origins)
    for feature, group in records.groupby("feature"):
        # At an equal archive date, choose the most recent target period.
        group = group.sort_values([timeline, "observed_at", "target_period"]).drop_duplicates(timeline, keep="last")
        merged = pd.merge_asof(pd.DataFrame({"origin": origins}).sort_values("origin"),
                               group[[timeline, "observed_at", "value"]].sort_values(timeline),
                               left_on="origin", right_on=timeline, direction="backward")
        age = (merged.origin - merged.observed_at).dt.total_seconds()
        fresh = age.between(0, max_age_days*86400)
        values = pd.Series(np.where(fresh, merged.value, np.nan), index=pd.DatetimeIndex(merged.origin))
        result[feature] = values.reindex(origins)
    return result


def _compute_macro_information(payload: dict, controls: pd.DataFrame, output: Path, snaps: dict) -> dict:
    parsers = {"cleveland_inflation": parse_inflation, "nyfed_growth": parse_growth,
               "reserve_elasticity": parse_reserves}
    history = load_boundary_history(payload)
    origins = pd.DatetimeIndex(history.origin_date)
    control_columns = [c for c in ["cpiaucsl__z_52w", "indpro__z_52w", "unrate__z_52w", "walcl__z_52w", "nfci__z_52w"] if c in controls]
    if not control_columns:
        raise ValueError("existing macro control features are required")
    sources, metric_rows, all_features, experiments = [], [], [], {}
    for key, snap in snaps.items():
        records = parsers[key](snap)
        records.to_csv(output/f"{key}-records.csv", index=False)
        archive = align_features(records, origins, reconstructed=True)
        first_seen = align_features(records, origins)
        if first_seen.loc[origins < snap.retrieved_at].notna().any().any():
            raise ValueError("a snapshot became available before its first-seen time")
        first_seen.to_csv(output/f"{key}-weekly-first-seen.csv", index_label="origin_date")
        columns = [c for c in archive if "lower" not in c and "upper" not in c]
        for column in columns.copy():
            archive[column+"_change4w"] = archive[column].diff(4)
            columns.append(column+"_change4w")
        archive.to_csv(output/f"{key}-weekly-reconstructed.csv", index_label="origin_date")
        candidate_features = controls.reindex(origins)[control_columns].join(archive[columns])
        predictions, experiment = run_same_origin_ablation(history, candidate_features,
            control_columns=control_columns, extra_columns=columns, minimum_training_rows=104)
        predictions.to_csv(output/f"{key}-predictions.csv", index=False)
        experiments[key] = experiment
        for split, metrics in experiment["metrics"].items():
            paired = metrics["paired_delta"]
            if (not paired["same_origins"] or not paired["same_training_counts"]
                    or metrics["candidate"]["n"] != metrics["control"]["n"]):
                raise ValueError("macro comparison requires identical origins and training samples")
            metric_rows.append({"source": key, "model": f"boundary_{key}", "horizon_weeks": 1,
                "evaluation_split": "selection" if split == "selection" else "retrospective_diagnostic",
                "n": metrics["candidate"]["n"], "delta_log_loss": metrics["paired_delta"]["candidate_minus_control_log_loss"],
                "delta_brier": metrics["paired_delta"]["candidate_minus_control_brier"],
                "status": "archive_diagnostic" if key != "reserve_elasticity" else "current_vintage_sensitivity"})
        if not experiment["metrics"]:
            metric_rows.append({"source": key, "model": f"boundary_{key}", "horizon_weeks": 1,
                                "n": 0, "delta_log_loss": None, "status": "accumulating"})
        # First-seen latest feature display uses actual collection time.
        now = max(s.retrieved_at for s in snaps.values())
        current = align_features(records, pd.DatetimeIndex([now])).iloc[0].dropna().to_dict()
        most_recent = records.loc[records.observed_at == records.observed_at.max()]
        sources.append({"id": key, **{k: SOURCES[key][k] for k in ("label", "url", "unit")},
            "observed_at": records.observed_at.max(), "available_at": snap.retrieved_at,
            "latest": current, "target_period": most_recent.target_period.iloc[-1],
            "history_rows": int(records.observed_at.nunique()), "availability_mode": "first_seen",
            "snapshot": snap.manifest()})
        all_features.extend({"id": c, "label": FEATURE_LABELS[c], "value": v, "unit": SOURCES[key]["unit"]} for c,v in current.items())
        if key == "reserve_elasticity":
            alternate = align_features(parse_reserves(snap, 60), origins, reconstructed=True)
            for column in columns:
                if column.endswith("_change4w"):
                    alternate[column] = alternate[column.removesuffix("_change4w")].diff(4)
            _, sensitivity = run_same_origin_ablation(history, controls.reindex(origins)[control_columns].join(alternate[columns]),
                control_columns=control_columns, extra_columns=columns, minimum_training_rows=104)
            experiments["reserve_elasticity_lag60"] = sensitivity
    result = {"schema_version": "regime-additional-information/1", "data_as_of": payload["meta"]["data_as_of"],
              "source_generation_id": payload["meta"]["generation_id"],
              "source_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest(),
              "sources": sources, "features": all_features,
              "evaluation": {"rows": metric_rows, "policy": "고정 규칙·동일 시점 비교 · 공개 아카이브 재구성", "experiments": experiments},
              "protocol": {"minimum_training_rows": 104, "mixture_weight": .25, "regularization_c": .1,
                  "historical_timeline": "Cleveland/NYFed dated archives + 1 calendar day; reserve current-vintage lag30/60 sensitivity",
                  "operational_timeline": "immutable first_seen_at; reconstructed lags never backdate availability",
                  "reserve_results_are_predictive_evidence": False, "automatic_promotion": False}}
    write_json(output/"additional-information.json", result)
    return json_safe(result)


def build_macro_information(payload: dict, controls: pd.DataFrame, output: Path, *, refresh=False, verify_inputs=None) -> dict:
    """Publish the summary only after every derived artifact is frozen together."""
    output.mkdir(parents=True, exist_ok=True)
    for path in (output, output/"snapshots", output/"runs", output/"additional-information.json"):
        if path.is_symlink():
            raise ValueError("macro output paths must not be symlinks")
    with ThreadPoolExecutor(max_workers=3) as pool:
        snaps = dict(zip(SOURCES, pool.map(lambda key: snapshot(key, output/"snapshots", refresh=refresh), SOURCES)))
    staging = Path(tempfile.mkdtemp(prefix=".macro-run-", dir=output))
    try:
        result = _compute_macro_information(payload, controls, staging, snaps)
        artifacts = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                     for path in sorted(staging.glob("*.csv"))}
        generation = hashlib.sha256(json.dumps({"result": result, "artifacts": artifacts}, sort_keys=True,
            ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()[:24]
        result["artifact_manifest"] = {"generation": generation,
            "files": [{"path": f"runs/{generation}/{name}", **record} for name, record in artifacts.items()]}
        write_json(staging/"additional-information.json", result)
        if verify_inputs is not None:
            verify_inputs()
        destination = output/"runs"/generation
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if any((destination/path.name).read_bytes() != path.read_bytes() for path in staging.iterdir()):
                raise ValueError("immutable macro run changed")
        else:
            staging.rename(destination)
        if verify_inputs is not None:
            verify_inputs()
        write_json(output/"additional-information.json", result)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
