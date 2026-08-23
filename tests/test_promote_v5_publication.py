from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_v5_publication.py"
SPEC = importlib.util.spec_from_file_location("promote_v5_publication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
promote_v5_publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote_v5_publication)


@pytest.mark.parametrize(
    "reasons",
    [
        ["weak_generalization"],
        ["calibration_drift"],
        ["weak_generalization", "calibration_drift"],
    ],
)
def test_publication_health_accepts_only_reviewed_model_warnings(
    reasons: list[str],
) -> None:
    promote_v5_publication._validate_publication_health(
        {"status": "degraded"},
        {"model_health": {"status": "review_due", "reasons": reasons}},
    )


def test_publication_health_accepts_fully_healthy_candidate() -> None:
    promote_v5_publication._validate_publication_health(
        {"status": "ok"},
        {"model_health": {"status": "ok", "reasons": []}},
    )


@pytest.mark.parametrize(
    ("meta_status", "model_status", "reasons"),
    [
        ("failed", "review_due", ["weak_generalization"]),
        ("degraded", "degraded", ["weak_generalization"]),
        ("degraded", "review_due", []),
        ("degraded", "review_due", ["provider_degraded"]),
        (
            "degraded",
            "review_due",
            ["weak_generalization", "provider_degraded"],
        ),
        (
            "degraded",
            "review_due",
            ["weak_generalization", "weak_generalization"],
        ),
        ("ok", "review_due", ["weak_generalization"]),
        ("ok", "ok", ["calibration_drift"]),
    ],
)
def test_publication_health_rejects_unapproved_or_inconsistent_health(
    meta_status: str,
    model_status: str,
    reasons: list[str],
) -> None:
    with pytest.raises(promote_v5_publication.PromotionError):
        promote_v5_publication._validate_publication_health(
            {"status": meta_status},
            {"model_health": {"status": model_status, "reasons": reasons}},
        )


@pytest.mark.parametrize("reasons", [None, "weak_generalization", [1]])
def test_publication_health_rejects_malformed_reason_list(reasons: object) -> None:
    with pytest.raises(
        promote_v5_publication.PromotionError,
        match="reasons are invalid",
    ):
        promote_v5_publication._validate_publication_health(
            {"status": "degraded"},
            {"model_health": {"status": "review_due", "reasons": reasons}},
        )
