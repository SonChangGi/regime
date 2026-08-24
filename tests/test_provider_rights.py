from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from regime_lab.provider_rights import (
    ProviderRightsError,
    providers_for_live_config,
    verify_provider_rights,
)


NOW = datetime(2026, 8, 25, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def _write_policy(path: Path, provider: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "providers": {"source": provider}}),
        encoding="utf-8",
    )


def test_live_config_maps_provider_names_to_policy_ids() -> None:
    assert providers_for_live_config(
        {"alfred": {}, "alpha_vantage": {}, "feature_engineering": {}}
    ) == ("fred_alfred", "alpha_vantage")
    assert providers_for_live_config(
        {
            "alfred": {},
            "provider_rights_providers": ["frb_h10", "ofr_fsi", "frb_h10"],
        }
    ) == ("fred_alfred", "frb_h10", "ofr_fsi")
    assert providers_for_live_config({}) == ()

    with pytest.raises(ProviderRightsError, match="provider_rights_providers"):
        providers_for_live_config({"provider_rights_providers": "frb_h10"})


def test_rights_gate_requires_affirmative_current_capabilities(tmp_path: Path) -> None:
    policy = tmp_path / "rights.json"
    _write_policy(
        policy,
        {
            "status": "allowed",
            "review_after": "2027-01-01T00:00:00+00:00",
            "capabilities": {
                "collection": True,
                "local_storage": True,
                "model_training": True,
                "derived_publication": True,
            },
            "evidence": {
                "source_url": "https://example.gov/terms",
                "basis": "official_terms_review",
            },
        },
    )
    verify_provider_rights(("source",), policy_path=policy, now=NOW)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"status": "unconfirmed"}, "unconfirmed"),
        (
            {
                "status": "allowed",
                "review_after": "2026-01-01T00:00:00+00:00",
                "capabilities": {
                    "collection": True,
                    "local_storage": True,
                    "model_training": True,
                    "derived_publication": True,
                },
                "evidence": {
                    "source_url": "https://example.gov/terms",
                    "basis": "official_terms_review",
                },
            },
            "expired",
        ),
        (
            {
                "status": "allowed",
                "review_after": "2027-01-01T00:00:00+00:00",
                "capabilities": {
                    "collection": True,
                    "local_storage": True,
                    "model_training": False,
                    "derived_publication": True,
                },
                "evidence": {
                    "source_url": "https://example.gov/terms",
                    "basis": "official_terms_review",
                },
            },
            "model_training",
        ),
    ],
)
def test_rights_gate_fails_closed(
    tmp_path: Path,
    entry: dict[str, object],
    message: str,
) -> None:
    policy = tmp_path / "rights.json"
    _write_policy(policy, entry)
    with pytest.raises(ProviderRightsError, match=message):
        verify_provider_rights(("source",), policy_path=policy, now=NOW)


def test_rights_gate_rejects_missing_policy_before_transport(tmp_path: Path) -> None:
    with pytest.raises(ProviderRightsError, match="unavailable"):
        verify_provider_rights(
            ("source",),
            policy_path=tmp_path / "missing.json",
            now=NOW,
        )


def test_current_live_provider_set_allows_approved_derived_publication() -> None:
    config = json.loads((ROOT / "config/series.json").read_text(encoding="utf-8"))
    providers = providers_for_live_config(config)
    verify_provider_rights(
        providers,
        policy_path=ROOT / "config/provider_rights.json",
        now=NOW,
        capabilities=("collection", "local_storage", "model_training"),
    )
    verify_provider_rights(
        providers,
        policy_path=ROOT / "config/provider_rights.json",
        now=NOW,
    )


def test_user_attested_derived_publication_requires_exact_scope(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "rights.json"
    _write_policy(
        policy,
        {
            "status": "allowed",
            "review_after": "2027-01-01T00:00:00+00:00",
            "capabilities": {
                "collection": True,
                "local_storage": True,
                "model_training": True,
                "derived_publication": True,
            },
            "evidence": {
                "source_url": "https://example.com/terms",
                "basis": "user_attested_direct_provider_approval",
                "attested_at": "2026-08-24T00:00:00+00:00",
                "approval_reference": "direct-approval-reference",
                "approval_scope": "personal_noncommercial_derived_results_only",
            },
            "conditions": {
                "project": "regime",
                "raw_publication": False,
                "derived_outputs_only": True,
                "personal_noncommercial_only": True,
                "commercial_publication": False,
                "derived_publication_scope_document_pending": True,
            },
        },
    )

    with pytest.raises(ProviderRightsError, match="scope confirmation"):
        verify_provider_rights(
            ("source",),
            policy_path=policy,
            now=NOW,
            capabilities=("derived_publication",),
        )


def test_v6_market_plan_matches_alpha_derived_publication_approval_boundary() -> None:
    plan = json.loads(
        (ROOT / "config/structural_v6_research.json").read_text(encoding="utf-8")
    )
    market = plan["market_label_source"]

    assert market["provider"] == "Alpha Vantage"
    assert market["status"] == "selected_for_private_research"
    assert market["selection_gate_passed"] is True
    assert market["historical_evidence"]["point_in_time_certified_before_first_collection"] is False
    assert market["historical_evidence"]["eligible_role"] == "retrospective_sensitivity_only"
    assert market["derived_publication"] == (
        "approved_personal_noncommercial_derived_results_only"
    )
    assert plan["publication_boundary"]["derived_publication"] is True
    assert plan["publication_boundary"]["raw_publication"] is False
