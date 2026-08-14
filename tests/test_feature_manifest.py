from __future__ import annotations

import copy

import pytest

from regime_lab.feature_manifest import (
    complete_feature_group_manifest,
    feature_manifest_document,
    feature_manifest_sha256,
)


def test_complete_manifest_assigns_pipeline_columns_to_legacy_once() -> None:
    groups = (
        {
            "id": "treasury_curve",
            "description": "curve",
            "feature_count": 1,
            "features": ("treasury_curve__level",),
        },
        {
            "id": "legacy_v3",
            "description": "legacy",
            "feature_count": 1,
            "features": ("spy_close__return_1w",),
        },
    )
    completed = complete_feature_group_manifest(
        (
            "spy_close__return_1w",
            "treasury_curve__level",
            "regime_boundary__risk_score",
        ),
        groups,
    )

    by_id = {row["id"]: row for row in completed}
    assert by_id["legacy_v3"]["features"] == (
        "spy_close__return_1w",
        "regime_boundary__risk_score",
    )
    assigned = [feature for row in completed for feature in row["features"]]
    assert len(assigned) == len(set(assigned)) == 3


def test_feature_manifest_hash_is_stable_and_tamper_evident() -> None:
    document = feature_manifest_document(
        (
            {
                "id": "legacy_v3",
                "description": "legacy",
                "feature_count": 2,
                "features": ("a", "b"),
            },
        ),
        feature_set_version="weekly-pit-structural-v4",
    )
    core = {key: value for key, value in document.items() if key != "sha256"}
    assert document["sha256"] == feature_manifest_sha256(core)

    changed = copy.deepcopy(core)
    changed["groups"][0]["features"][1] = "c"
    assert feature_manifest_sha256(changed) != document["sha256"]


def test_complete_manifest_fails_closed_on_unknown_or_duplicate_assignment() -> None:
    with pytest.raises(ValueError, match="unknown columns"):
        complete_feature_group_manifest(
            ("a",),
            ({"id": "legacy_v3", "features": ("missing",)},),
        )
    with pytest.raises(ValueError, match="belongs to both"):
        complete_feature_group_manifest(
            ("a",),
            (
                {"id": "one", "features": ("a",)},
                {"id": "two", "features": ("a",)},
            ),
        )
