"""Research-only portfolio allocation candidates."""

from regime_lab.allocation.shadow import (
    ALLOCATION_RESULT_SCHEMA_VERSION,
    allocation_calibration_evidence,
    build_allocation_shadow_candidate,
    default_allocation_shadow_spec_path,
    load_allocation_shadow_spec,
    rebase_allocation_candidate_intent,
    split_safe_asset_return_frames,
)

__all__ = [
    "ALLOCATION_RESULT_SCHEMA_VERSION",
    "allocation_calibration_evidence",
    "build_allocation_shadow_candidate",
    "default_allocation_shadow_spec_path",
    "load_allocation_shadow_spec",
    "rebase_allocation_candidate_intent",
    "split_safe_asset_return_frames",
]
