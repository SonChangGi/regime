from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
from pathlib import Path

import pytest

from regime_lab.data.release_archive import (
    DEFAULT_RELEASE_SOURCE_CATALOG,
    EvidenceTrack,
    ReleaseCatalogError,
    ReleaseRecord,
    eligible_release_records,
    load_release_source_catalog,
    resolve_source_released_at,
    weekly_decision_at,
)


UTC = timezone.utc
REQUIRED_SOURCE_IDS = {
    "philadelphia_ads",
    "philadelphia_rtdsm",
    "board_h41",
    "board_h8",
    "board_sloos",
    "ofr_fsi",
    "board_ntfs",
    "board_ebp",
    "cboe_vix_1600_control",
    "cboe_vix_1615_sensitivity",
    "cboe_vix_term_structure",
    "dol_weekly_claims",
    "bls",
    "bea",
    "census_eits",
}


def _record(
    *,
    period_end: date = date(2026, 8, 20),
    released_at: datetime = datetime(2026, 8, 21, 12, 30, tzinfo=UTC),
    first_seen_at: datetime = datetime(2026, 8, 21, 20, 1, tzinfo=UTC),
    retrieved_at: datetime = datetime(2026, 8, 21, 20, 2, tzinfo=UTC),
    revision_seq: int = 0,
    digest: str = "a" * 64,
) -> ReleaseRecord:
    return ReleaseRecord(
        source_id="bls",
        series_id="PAYEMS",
        observed_period_end=period_end,
        value=120.0 + revision_seq,
        source_released_at=released_at,
        provider_first_seen_at=first_seen_at,
        system_retrieved_at=retrieved_at,
        revision_seq=revision_seq,
        raw_sha256=digest,
        units="thousands",
    )


def test_default_catalog_covers_priority_sources_without_ingestion_claims() -> None:
    catalog = load_release_source_catalog()

    assert set(catalog.source_ids) == REQUIRED_SOURCE_IDS
    assert catalog.admitted_source_ids == ()
    assert len(catalog.sha256) == 64
    assert all(not source.enabled for source in catalog.sources)
    assert all(not source.ingested for source in catalog.sources)
    assert all(source.official_primary_url.startswith("https://") for source in catalog.sources)


def test_ofr_parser_status_and_cboe_written_license_blocks_are_explicit() -> None:
    catalog = load_release_source_catalog()

    assert catalog.source("ofr_fsi").status.value == "parser_implemented"
    for source_id in (
        "cboe_vix_1600_control",
        "cboe_vix_1615_sensitivity",
        "cboe_vix_term_structure",
    ):
        source = catalog.source(source_id)
        assert source.status.value == "blocked_pending_written_license"
        assert source.rights_profile == "cboe_written_license_required"
        assert source.enabled is False
        assert source.ingested is False


def test_catalog_rejects_false_ingestion_status(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_RELEASE_SOURCE_CATALOG.read_text(encoding="utf-8"))
    payload["sources"][0]["ingested"] = True
    path = tmp_path / "invalid-catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseCatalogError, match="must also be enabled"):
        load_release_source_catalog(path)


def test_catalog_rejects_enabling_a_written_license_block(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_RELEASE_SOURCE_CATALOG.read_text(encoding="utf-8"))
    source = next(
        item
        for item in payload["sources"]
        if item["id"] == "cboe_vix_1600_control"
    )
    source["enabled"] = True
    path = tmp_path / "invalid-cboe-catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseCatalogError, match="rights-blocked source"):
        load_release_source_catalog(path)


def test_catalog_hash_is_independent_of_json_formatting(tmp_path: Path) -> None:
    original = load_release_source_catalog()
    payload = json.loads(DEFAULT_RELEASE_SOURCE_CATALOG.read_text(encoding="utf-8"))
    reformatted = tmp_path / "reformatted-catalog.json"
    reformatted.write_text(
        json.dumps(payload, ensure_ascii=False, indent=7),
        encoding="utf-8",
    )

    assert load_release_source_catalog(reformatted).sha256 == original.sha256


def test_weekly_cutoff_uses_exact_new_york_dst_offset() -> None:
    assert weekly_decision_at(date(2026, 3, 6)) == datetime(
        2026,
        3,
        6,
        21,
        0,
        tzinfo=UTC,
    )
    assert weekly_decision_at(date(2026, 3, 13)) == datetime(
        2026,
        3,
        13,
        20,
        0,
        tzinfo=UTC,
    )
    assert weekly_decision_at(date(2026, 11, 6)) == datetime(
        2026,
        11,
        6,
        21,
        0,
        tzinfo=UTC,
    )
    with pytest.raises(ReleaseCatalogError, match="weekly decision date"):
        weekly_decision_at(date(2026, 3, 12))


def test_date_only_month_end_rolls_forward_without_same_day_assumption() -> None:
    catalog = load_release_source_catalog()
    census = catalog.source("census_eits")
    rtdsm = catalog.source("philadelphia_rtdsm")

    assert resolve_source_released_at(census, date(2026, 1, 31)) == datetime(
        2026,
        2,
        1,
        5,
        0,
        tzinfo=UTC,
    )
    # A Friday date-only RTDSM vintage cannot enter that Friday's decision.
    assert resolve_source_released_at(rtdsm, date(2026, 1, 30)) == datetime(
        2026,
        2,
        6,
        21,
        0,
        tzinfo=UTC,
    )


def test_same_day_release_and_post_cutoff_vix_are_not_mixed() -> None:
    catalog = load_release_source_catalog()
    bls = catalog.source("bls")
    vix_1615 = catalog.source("cboe_vix_1615_sensitivity")
    decision = weekly_decision_at(date(2026, 8, 21))

    assert resolve_source_released_at(bls, date(2026, 8, 21)) == datetime(
        2026,
        8,
        21,
        12,
        30,
        tzinfo=UTC,
    )
    assert resolve_source_released_at(bls, date(2026, 8, 21)) <= decision
    assert resolve_source_released_at(vix_1615, date(2026, 8, 21)) == datetime(
        2026,
        8,
        21,
        20,
        15,
        tzinfo=UTC,
    )
    assert resolve_source_released_at(vix_1615, date(2026, 8, 21)) > decision


def test_exact_timestamp_source_refuses_date_only_backdating() -> None:
    source = load_release_source_catalog().source("cboe_vix_1600_control")

    with pytest.raises(ReleaseCatalogError, match="exact source timestamp"):
        resolve_source_released_at(source, date(2026, 8, 21))

    exact = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
    assert resolve_source_released_at(
        source,
        date(2026, 8, 21),
        exact_timestamp=exact,
    ) == exact


def test_late_provider_response_is_reconstructed_but_not_operational() -> None:
    record = _record()
    decision = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)

    assert record.is_eligible(decision, track=EvidenceTrack.RECONSTRUCTED_OOS)
    assert not record.is_eligible(decision, track=EvidenceTrack.OPERATIONAL_OOS)
    assert record.is_eligible(
        datetime(2026, 8, 21, 20, 1, tzinfo=UTC),
        track=EvidenceTrack.OPERATIONAL_OOS,
    )
    reconstructed = record.to_observation(track=EvidenceTrack.RECONSTRUCTED_OOS)
    operational = record.to_observation(track=EvidenceTrack.OPERATIONAL_OOS)
    assert reconstructed.available_at == record.source_released_at
    assert operational.available_at == record.provider_first_seen_at
    assert operational.metadata["evidence_track"] == "operational_oos"


def test_future_period_is_never_eligible_even_with_bad_early_release_clock() -> None:
    malformed_archive_row = _record(
        period_end=date(2026, 8, 28),
        released_at=datetime(2026, 8, 21, 12, 30, tzinfo=UTC),
    )
    decision = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)

    assert not malformed_archive_row.is_eligible(
        decision,
        track=EvidenceTrack.RECONSTRUCTED_OOS,
    )
    assert eligible_release_records(
        (malformed_archive_row,),
        decision,
        track=EvidenceTrack.RECONSTRUCTED_OOS,
    ) == ()


def test_latest_revision_selection_respects_decision_time() -> None:
    decision = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
    original = _record(
        first_seen_at=datetime(2026, 8, 21, 13, 0, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 21, 13, 0, tzinfo=UTC),
    )
    late_revision = _record(
        first_seen_at=datetime(2026, 8, 22, 13, 0, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 22, 13, 0, tzinfo=UTC),
        revision_seq=1,
        digest="b" * 64,
    )

    assert eligible_release_records(
        (late_revision, original),
        decision,
        track=EvidenceTrack.OPERATIONAL_OOS,
    ) == (original,)
    assert eligible_release_records(
        (late_revision, original),
        datetime(2026, 8, 22, 13, 0, tzinfo=UTC),
        track=EvidenceTrack.OPERATIONAL_OOS,
    ) == (late_revision,)


def test_release_record_rejects_unverifiable_hash_and_clock_order() -> None:
    with pytest.raises(ReleaseCatalogError, match="raw_sha256"):
        _record(digest="not-a-digest")
    with pytest.raises(ReleaseCatalogError, match="first_seen_at must not be after"):
        _record(
            first_seen_at=datetime(2026, 8, 21, 20, 2, tzinfo=UTC),
            retrieved_at=datetime(2026, 8, 21, 20, 1, tzinfo=UTC),
        )


def test_naive_release_timestamp_is_rejected() -> None:
    source = load_release_source_catalog().source("cboe_vix_1600_control")
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_source_released_at(
            source,
            date(2026, 8, 21),
            exact_timestamp=datetime(2026, 8, 21, 16, 0),
        )


def test_weekly_cutoff_accepts_explicit_alternative_time_without_relabeling() -> None:
    assert weekly_decision_at(
        date(2026, 8, 21),
        local_time=time(16, 15),
    ) == datetime(2026, 8, 21, 20, 15, tzinfo=UTC)
