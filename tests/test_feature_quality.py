from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.feature_quality import (
    canonical_feature_quality_json_bytes,
    feature_quality_artifact_manifest,
    feature_quality_document,
    verify_feature_quality_artifact,
)
from regime_lab.contract_v5 import _validate_feature_quality_artifact


def test_feature_quality_flags_missing_constant_and_shift_deterministically() -> None:
    index = pd.date_range("2020-01-03", periods=220, freq="W-FRI", tz="UTC")
    frame = pd.DataFrame(
        {
            "healthy": np.sin(np.arange(220) / 8),
            "missing": [*range(215), *([np.nan] * 5)],
            "constant": [*range(168), *([7.0] * 52)],
            "shift": [*([0.0] * 168), *([10.0] * 52)],
            "empty": [np.nan] * 220,
            "healthy__missing": [0.0] * 220,
        },
        index=index,
    )

    first = feature_quality_document(frame)
    second = feature_quality_document(frame.copy())
    by_name = {row["feature"]: row for row in first["features"]}

    assert first == second
    assert first["status"] == "warning"
    assert len(first["sha256"]) == 64
    assert "latest_missing_streak" in by_name["missing"]["reasons"]
    assert "recent_constant" in by_name["constant"]["reasons"]
    assert "distribution_shift" in by_name["shift"]["reasons"]
    assert by_name["empty"]["status"] == "unavailable"
    assert by_name["healthy__missing"]["status"] == "ok"
    assert by_name["healthy__missing"]["role"] == "availability_or_event_monitor"


def test_feature_quality_rejects_noncausal_index_shape() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=[1, 1])
    with pytest.raises(ValueError, match="unique and increasing"):
        feature_quality_document(frame)


def test_feature_quality_artifact_is_hash_bound_and_tamper_evident(
    tmp_path,
) -> None:
    index = pd.date_range("2024-01-05", periods=60, freq="W-FRI", tz="UTC")
    document = feature_quality_document(
        pd.DataFrame({"x": np.arange(60, dtype=float)}, index=index)
    )
    manifest = feature_quality_artifact_manifest(document)
    path = tmp_path / "feature-quality.json"
    path.write_bytes(canonical_feature_quality_json_bytes(document))

    assert verify_feature_quality_artifact(manifest, tmp_path) == document

    path.write_bytes(path.read_bytes().replace(b'"status": "ok"', b'"status": "warning"'))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_feature_quality_artifact(manifest, tmp_path)


def test_v5_contract_accepts_multiple_feature_quality_warnings() -> None:
    index = pd.date_range("2020-01-03", periods=60, freq="W-FRI", tz="UTC")
    document = feature_quality_document(
        pd.DataFrame(
            {
                "first": [1.0] * 60,
                "second": [2.0] * 60,
            },
            index=index,
        )
    )
    manifest = feature_quality_artifact_manifest(document)

    assert manifest["warning_feature_count"] == 2
    _validate_feature_quality_artifact({"feature_quality_artifact": manifest})
