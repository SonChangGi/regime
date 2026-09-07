"""Small explicit contract for optional comprehensive research blocks."""

from __future__ import annotations
import json


def validate_research_extensions(research: dict, *, data_as_of: str | None = None) -> None:
    _validate_required_forecast_publication(research, data_as_of=data_as_of)
    # JSON itself must remain finite; it is consumed without a Python runtime.
    json.dumps(research, allow_nan=False)
    if data_as_of is not None:
        import pandas as pd
        for name in ("forecast_research", "calibration_audit", "forecast_information"):
            if name in research and pd.Timestamp(research[name].get("data_as_of")) != pd.Timestamp(data_as_of):
                raise ValueError(f"{name} cutoff differs from the dashboard")
    if "forecast_research" in research:
        from regime_lab.analysis.forecast_audit_research import validate_forecast_research_extension
        validate_forecast_research_extension(research["forecast_research"])
    for name, schema in (("calibration_audit", "regime-calibration-audit/1"),
                         ("forecast_information", "regime-forecast-information/1")):
        if name in research:
            _validate_forecast_audit_table(research[name], schema=schema)
    if "forecast_improvement" in research:
        from regime_lab.research.forecast_contract import validate_forecast_improvement
        validate_forecast_improvement(research["forecast_improvement"])
    extensions = research.get("extensions")
    if extensions is not None:
        if extensions.get("schema_version") != "regime-research-extensions/1":
            raise ValueError("research extensions schema invalid")
        if not set(extensions).issubset(
            {"schema_version", "downside", "diagnostics", "additional_data", "build"}
        ):
            raise ValueError("unknown research extension")
        downside = extensions.get("downside", {})
        for row in downside.get("latest", []):
            if (
                row["horizon_weeks"] not in (4, 13)
                or not 0 <= row["loss_probability"] <= 1
                or row["q10"] > row["q25"]
            ):
                raise ValueError("downside prediction invalid")
        for row in downside.get("comparisons", []):
            if row["weeks"] <= 0 or any(
                not 0 <= row[k] <= 1
                for k in ("coverage_q10", "coverage_q25", "brier_loss")
            ):
                raise ValueError("downside comparison invalid")
        for source in extensions.get("additional_data", {}).get("sources", []):
            if not source["url"].startswith("https://") or len(source["sha256"]) != 64:
                raise ValueError("additional source provenance invalid")
    for key, schema in [
        ("decision_research_v2", "regime-decision-research/2"),
        ("operational_diagnostics", "regime-operational-diagnostics/1"),
    ]:
        block = research.get(key)
        if block is not None and block.get("schema_version") != schema:
            raise ValueError(f"{key} schema invalid")
    allocation = research.get("prospective_decision_shadow", {}).get(
        "allocation_research_v2"
    )
    if allocation is not None:
        if (
            allocation.get("schema_version") != "regime-allocation-research/2"
            or allocation.get("role") != "research_only_no_promotion"
        ):
            raise ValueError("allocation research identity invalid")
        if any(
            allocation.get(k) is not False
            for k in (
                "affects_official_forecast",
                "affects_champion_selection",
                "affects_issued_ledger",
            )
        ):
            raise ValueError("allocation research may not affect issued decisions")


def _validate_forecast_audit_table(block: dict, *, schema: str) -> None:
    """Validate optional preview score tables without inventing absent evidence."""
    import math
    import pandas as pd
    if not isinstance(block, dict) or block.get("schema_version") != schema:
        raise ValueError("forecast audit table schema is invalid")
    cutoff = pd.Timestamp(block.get("data_as_of"))
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("forecast audit table requires a zoned cutoff")
    rows = block.get("rows")
    if not isinstance(rows, list):
        raise ValueError("forecast audit table rows are missing")
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            raise ValueError("forecast audit row model is missing")
        if row.get("evaluation_split") not in {"selection", "holdout", "retrospective_diagnostic", "prospective"}:
            raise ValueError("forecast audit row evidence split is invalid")
        sample_key = "n_predictions" if schema == "regime-calibration-audit/1" else "matched_n"
        count = row.get(sample_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("forecast audit row needs a positive matched sample count")
        identity = (row.get("feature_block"), row["model"], row.get("baseline_model"),
                    row.get("horizon_weeks"), row["evaluation_split"])
        if identity in identities:
            raise ValueError("forecast audit table contains duplicate metric rows")
        identities.add(identity)
        for key, value in row.items():
            if "log_loss" not in key and "brier" not in key:
                continue
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("forecast audit scores must be finite or unavailable")
            if not key.startswith("delta_") and value < 0:
                raise ValueError("proper probability scores cannot be negative")
        if schema == "regime-calibration-audit/1":
            if row.get("horizon_weeks") not in (1, 4, 13):
                raise ValueError("calibration audit horizon is invalid")
        else:
            if not isinstance(row.get("feature_block"), str) or not isinstance(row.get("baseline_model"), str):
                raise ValueError("information audit comparison identity is invalid")
            if not math.isclose(row["candidate_log_loss"] - row["baseline_log_loss"], row["delta_log_loss"], abs_tol=1e-10):
                raise ValueError("information audit difference does not match scores")


def _validate_required_forecast_publication(research: dict, *, data_as_of: str | None) -> None:
    """Legacy optional extensions remain readable; completed weekly builds bind all three."""
    import re
    import pandas as pd
    from regime_lab.integrity import canonical_json_sha256_v1
    build = research.get("extensions", {}).get("build", {})
    if "forecast_audit_present" not in build:
        return
    if build["forecast_audit_present"] is not True:
        raise ValueError("forecast publication completion marker must be true")
    names = ("forecast_research", "calibration_audit", "forecast_information")
    if not all(isinstance(research.get(name), dict) for name in names):
        raise ValueError("completed forecast publication requires all three blocks")
    provenance = build.get("forecast_audit")
    if not isinstance(provenance, dict) or provenance.get("schema_version") != "regime-forecast-publication-build/1":
        raise ValueError("forecast publication provenance schema differs")
    if provenance.get("automatic_promotion") is not False or provenance.get("evidence_track") != "reconstructed_market":
        raise ValueError("forecast publication evidence role differs")
    expected = dict(provenance)
    cache_key = expected.pop("cache_key", None)
    if cache_key != canonical_json_sha256_v1(expected):
        raise ValueError("forecast publication provenance checksum differs")
    for key in ("official_payload_sha256", "recipe_sha256", "information_snapshot_sha256"):
        if not isinstance(provenance.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", provenance[key]):
            raise ValueError("forecast publication input hash is invalid")
    if provenance.get("source_generation_id") != build.get("generation_id") or provenance.get("data_as_of") != build.get("data_as_of"):
        raise ValueError("forecast publication source generation differs")
    if data_as_of is not None and pd.Timestamp(provenance["data_as_of"]) != pd.Timestamp(data_as_of):
        raise ValueError("forecast publication cutoff differs")
    for name in names:
        block = research[name]
        if block.get("publication_provenance") != provenance:
            raise ValueError("forecast publication block provenance differs")
        if build.get("outputs", {}).get(name) != canonical_json_sha256_v1(block):
            raise ValueError("forecast publication output checksum differs")
        if block.get("artifacts") != [{"label": "전체 결과 JSON", "url": f"./data/{name}.json"}]:
            raise ValueError("forecast publication artifact link differs")
