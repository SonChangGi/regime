"""Small explicit contract for optional comprehensive research blocks."""

from __future__ import annotations
import json


def validate_research_extensions(research: dict) -> None:
    # JSON itself must remain finite; it is consumed without a Python runtime.
    json.dumps(research, allow_nan=False)
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
