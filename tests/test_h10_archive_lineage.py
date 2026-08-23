from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib

import numpy as np
import pandas as pd

from regime_lab.analysis.fx import (
    build_official_archive_fx_features,
)
from regime_lab.analysis.fx_ablation import fx_ablation_readiness
from regime_lab.data import HealthStatus, Observation
from regime_lab.data import SQLiteSnapshotStore, SnapshotMode, SnapshotProvenance
from regime_lab.data.h10 import (
    BUSINESS_DAY_FREQUENCY,
    FIXED_BILATERAL_PANEL,
    SERIES_CATALOG,
)
from regime_lab.data.h10_archive import (
    H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS,
    H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY,
    H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS,
    H10_ARCHIVE_REVISION_POLICY,
    H10ArchiveRelease,
    H10ArchiveCollection,
    archive_release_available_at,
    detect_h10_archive_correction_equivalents,
    merge_h10_archive_releases,
)
import regime_lab.h10_store as h10_store
from regime_lab.h10_store import (
    H10_ARCHIVE_DATASET,
    H10_LICENSE_CLASS,
    H10_SOURCE,
    h10_collection_receipt_document,
    ingest_h10_archive_store,
    refresh_h10_archive_store,
)


UTC = timezone.utc
RETRIEVED_AT = datetime(2026, 8, 23, 1, tzinfo=UTC)
FX_CODES = ("BRD", "AFE", "EME", *FIXED_BILATERAL_PANEL)
CORRECTION_EQUIVALENT_DATES = (
    date(2024, 8, 7),
    date(2025, 1, 7),
    date(2026, 8, 12),
)


def _normal_records(
    event_date: date,
    *,
    week_position: int,
) -> tuple[Observation, ...]:
    available_at = archive_release_available_at(event_date)
    observed_dates = tuple(
        event_date - timedelta(days=7 - offset) for offset in range(5)
    )
    rows: list[Observation] = []
    for code_position, code in enumerate(FX_CODES):
        spec = SERIES_CATALOG[code]
        for day_position, observed_date in enumerate(observed_dates):
            value = (1.0 + code_position * 0.4) * np.exp(
                0.0004 * week_position
                + 0.00003 * day_position
                + 0.00001 * code_position * week_position
            )
            identity = f"{event_date}|{code}|{observed_date}|{value:.17g}"
            rows.append(
                Observation(
                    source="frb_h10",
                    series_id=(
                        f"H10|FX={code}|FREQ={BUSINESS_DAY_FREQUENCY}"
                    ),
                    observed_period_end=observed_date,
                    value=float(value),
                    released_at=available_at,
                    available_at=available_at,
                    vintage_date=event_date,
                    retrieved_at=RETRIEVED_AT,
                    units=code,
                    license_class=(
                        "federal_reserve_board_public_domain_citation_requested"
                    ),
                    quality_status=HealthStatus.OK,
                    raw_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                    metadata={
                        "fx_code": code,
                        "frequency_code": BUSINESS_DAY_FREQUENCY,
                        "quote_convention": spec.quote_convention.value,
                        "usd_strength_sign": spec.usd_strength_sign,
                        "obs_status": "A",
                        "official_release_archive_ingest": True,
                        "archive_chain_availability_basis": (
                            "official_archive_release_schedule"
                        ),
                        "archive_revision_policy": H10_ARCHIVE_REVISION_POLICY,
                        "availability_basis": (
                            H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS
                        ),
                    },
                )
            )
    return tuple(rows)


def _normal_release(event_date: date, *, week_position: int) -> H10ArchiveRelease:
    return H10ArchiveRelease(
        release_date=event_date,
        last_update_date=event_date,
        available_at=archive_release_available_at(event_date),
        availability_basis=H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS,
        retrieved_at=RETRIEVED_AT,
        source_url=(
            "https://www.federalreserve.gov/releases/h10/"
            f"{event_date.strftime('%Y%m%d')}/"
        ),
        snapshot_sha256=hashlib.sha256(
            f"normal|{event_date}".encode()
        ).hexdigest(),
        records=_normal_records(event_date, week_position=week_position),
    )


def _republication(
    prior: H10ArchiveRelease,
    event_date: date,
    *,
    declared_correction: bool,
    material_revision: bool,
) -> H10ArchiveRelease:
    release_date = prior.last_update_date if declared_correction else event_date
    available_at = archive_release_available_at(
        release_date,
        last_update_date=event_date,
    )
    basis = (
        "date_only_conservative_next_day"
        if declared_correction
        else H10_ARCHIVE_NORMAL_AVAILABILITY_BASIS
    )
    records: list[Observation] = []
    for position, record in enumerate(prior.records):
        revised = material_revision and position % 5 == 4
        value = float(record.value) + (0.01 if revised else 0.0)
        identity = (
            f"republication|{event_date}|{record.series_id}|"
            f"{record.observed_period_end}|{value:.17g}"
        )
        records.append(
            replace(
                record,
                value=value,
                released_at=available_at,
                available_at=available_at,
                vintage_date=event_date,
                raw_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                metadata={
                    **dict(record.metadata),
                    "availability_basis": basis,
                },
            )
        )
    return H10ArchiveRelease(
        release_date=release_date,
        last_update_date=event_date,
        available_at=available_at,
        availability_basis=basis,
        retrieved_at=RETRIEVED_AT,
        source_url=(
            "https://www.federalreserve.gov/releases/h10/"
            f"{event_date.strftime('%Y%m%d')}/"
        ),
        snapshot_sha256=hashlib.sha256(
            f"republished|{event_date}".encode()
        ).hexdigest(),
        records=tuple(records),
    )


def _frozen_245_page_regression_releases() -> tuple[H10ArchiveRelease, ...]:
    """Build a rights-safe replica of the audited private cache lineage."""
    normal_dates = tuple(
        value.date()
        for value in pd.date_range(
            "2022-01-03", "2026-08-17", freq="W-MON"
        )
    )
    event_dates = tuple(sorted((*normal_dates, *CORRECTION_EQUIVALENT_DATES)))
    releases: list[H10ArchiveRelease] = []
    normal_position = 0
    for event_date in event_dates:
        if event_date in CORRECTION_EQUIVALENT_DATES:
            releases.append(
                _republication(
                    releases[-1],
                    event_date,
                    declared_correction=event_date == date(2024, 8, 7),
                    material_revision=event_date == date(2026, 8, 12),
                )
            )
        else:
            releases.append(
                _normal_release(event_date, week_position=normal_position)
            )
            normal_position += 1
    assert len(releases) == 245
    return tuple(releases)


def test_245_page_lineage_detects_only_three_correction_equivalents() -> None:
    releases = _frozen_245_page_regression_releases()
    lineage = detect_h10_archive_correction_equivalents(releases)

    assert H10_ARCHIVE_CORRECTION_EQUIVALENT_POLICY == (
        "declared_or_material_revision_or_complete_republication"
    )
    assert H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS == (
        "declared_correction",
        "material_revision",
        "complete_republication",
    )
    assert lineage.release_count == 245
    assert tuple(event.event_date for event in lineage.events) == (
        CORRECTION_EQUIVALENT_DATES
    )
    assert lineage.declared_correction_event_count == 1
    assert lineage.detected_revision_event_count == 1
    assert lineage.detected_revision_row_count == 12
    assert lineage.complete_republication_event_count == 3
    assert lineage.short_gap_auxiliary_event_count == 3
    assert [event.prior_release_gap_days for event in lineage.events] == [2, 1, 2]
    assert [event.new_series_date_rows for event in lineage.events] == [0, 0, 0]
    assert [event.overlap_series_date_rows for event in lineage.events] == [
        60,
        60,
        60,
    ]
    assert [event.trigger_components for event in lineage.events] == [
        ("declared_correction", "complete_republication"),
        ("complete_republication",),
        ("material_revision", "complete_republication"),
    ]

    records, revision_rows = merge_h10_archive_releases(releases)
    assert len(records) == 14_532
    assert revision_rows == 12


def test_three_event_quarantine_has_exact_frozen_common_origin_contract() -> None:
    releases = _frozen_245_page_regression_releases()
    lineage = detect_h10_archive_correction_equivalents(releases)
    records, _revision_rows = merge_h10_archive_releases(releases)
    result = build_official_archive_fx_features(
        records,
        as_of=datetime(2026, 8, 21, 20, tzinfo=UTC),
        correction_available_at=tuple(
            event.available_at for event in lineage.events
        ),
    )
    cutoffs = pd.date_range("2022-01-07", "2026-08-21", freq="W-FRI")
    readiness = fx_ablation_readiness(result, cutoffs)

    assert int(result.coverage["archive_correction_quarantined"].sum()) == 51
    assert readiness["eligible_common_weeks"] == 165
    assert readiness["first_eligible_cutoff"] == "2022-07-08"
    assert readiness["last_eligible_cutoff"] == "2026-08-07"


def _small_lineage_releases() -> tuple[H10ArchiveRelease, ...]:
    first = _normal_release(date(2024, 8, 5), week_position=1)
    declared = _republication(
        first,
        date(2024, 8, 7),
        declared_correction=True,
        material_revision=False,
    )
    second = _normal_release(date(2025, 1, 6), week_position=2)
    republished = _republication(
        second,
        date(2025, 1, 7),
        declared_correction=False,
        material_revision=False,
    )
    third = _normal_release(date(2026, 8, 10), week_position=3)
    revised = _republication(
        third,
        date(2026, 8, 12),
        declared_correction=False,
        material_revision=True,
    )
    return (first, declared, second, republished, third, revised)


class _ZeroDeltaArchiveClient:
    def __init__(self, release_dates: tuple[date, ...]) -> None:
        self.release_dates = release_dates
        self.calls = 0

    def collect(self, **kwargs: object) -> H10ArchiveCollection:
        self.calls += 1
        cached = dict(kwargs["cached_releases"])
        assert tuple(sorted(cached)) == self.release_dates
        assert tuple(kwargs["known_release_dates"]) == self.release_dates
        requested = kwargs["requested_at"]
        assert isinstance(requested, datetime)
        return H10ArchiveCollection(
            releases=(),
            records=(),
            requested_at=requested,
            retrieved_at=requested + timedelta(minutes=1),
            index_sha256="b" * 64,
            collection_sha256="c" * 64,
            revision_event_count=0,
            discovered_release_dates=self.release_dates,
        )


def test_legacy_provenance_upgrades_from_cache_on_zero_delta_refresh(
    tmp_path,
) -> None:
    releases = _small_lineage_releases()
    release_dates = tuple(release.last_update_date for release in releases)
    records, revision_rows = merge_h10_archive_releases(releases)
    requested = datetime(2026, 8, 22, 23, tzinfo=UTC)
    cutoff = datetime(2026, 8, 21, 20, tzinfo=UTC)
    initial_collection = H10ArchiveCollection(
        releases=releases,
        records=records,
        requested_at=requested,
        retrieved_at=RETRIEVED_AT,
        index_sha256="a" * 64,
        collection_sha256="d" * 64,
        revision_event_count=revision_rows,
        discovered_release_dates=release_dates,
        lineage=None,
    )

    with SQLiteSnapshotStore(tmp_path / "archive-lineage.sqlite3") as store:
        initial = ingest_h10_archive_store(
            store,
            initial_collection,
            requested_at=requested,
            as_of=cutoff,
        )
        for release in releases:
            h10_store._write_archive_release_cache(
                store,
                release,
                requested_at=requested,
                cutoff=cutoff,
            )

        current = store.get_last_good_provenance(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
        )
        assert current is not None
        legacy_params = dict(current.request_params)
        for key in (
            "archive_correction_equivalent_policy",
            "archive_correction_equivalent_components",
            "archive_short_gap_auxiliary_days",
            "archive_declared_correction_count",
            "archive_quarantine_event_count",
            "archive_quarantine_event_dates",
            "archive_quarantine_available_at",
            "archive_detected_revision_event_count",
            "archive_detected_revision_row_count",
            "archive_complete_republication_event_count",
            "archive_short_gap_auxiliary_event_count",
            "archive_index_sha256",
            "archive_lineage_sha256",
            "archive_lineage_complete",
            "archive_correction_equivalent_event_lineage",
        ):
            legacy_params.pop(key)
        legacy_params["snapshot_mode"] = SnapshotMode.DELTA.value
        legacy_time = RETRIEVED_AT + timedelta(minutes=1)
        store.write_snapshot(
            (),
            SnapshotProvenance(
                source=H10_SOURCE,
                dataset=H10_ARCHIVE_DATASET,
                cutoff=cutoff,
                requested_at=legacy_time,
                retrieved_at=legacy_time,
                quality_status=HealthStatus.OK,
                license_class=H10_LICENSE_CLASS,
                request_params=legacy_params,
                response_sha256="e" * 64,
            ),
        )

        client = _ZeroDeltaArchiveClient(release_dates)
        upgraded = refresh_h10_archive_store(
            store,
            client,
            requested_at=legacy_time + timedelta(minutes=1),
            as_of=cutoff,
            end_date=date(2026, 8, 21),
        )
        stable = refresh_h10_archive_store(
            store,
            client,
            requested_at=legacy_time + timedelta(minutes=3),
            as_of=cutoff,
            end_date=date(2026, 8, 21),
        )
        provenance = store.get_last_good_provenance(
            source=H10_SOURCE,
            dataset=H10_ARCHIVE_DATASET,
        )

    assert initial.archive_correction_count == 1
    assert initial.archive_quarantine_event_count == 1
    assert upgraded.prepared.snapshot_mode is SnapshotMode.DELTA
    assert upgraded.prepared.added_count == 0
    assert upgraded.prepared.changed_count == 0
    assert upgraded.effective_records == initial.effective_records
    assert upgraded.archive_release_count == 6
    assert upgraded.archive_correction_count == 1
    assert upgraded.archive_quarantine_event_count == 3
    assert upgraded.archive_quarantine_event_dates == (
        CORRECTION_EQUIVALENT_DATES
    )
    assert upgraded.archive_detected_revision_event_count == 1
    assert upgraded.archive_detected_revision_row_count == 12
    assert upgraded.archive_complete_republication_event_count == 3
    assert upgraded.archive_short_gap_auxiliary_event_count == 3
    assert upgraded.source_row["archive_correction_count"] == 3
    assert upgraded.source_row["archive_correction_available_at"] == [
        value.isoformat() for value in upgraded.archive_quarantine_available_at
    ]
    assert upgraded.archive_index_sha256 == "b" * 64
    assert len(upgraded.archive_lineage_sha256) == 64
    assert stable.archive_lineage_sha256 == upgraded.archive_lineage_sha256
    assert stable.prepared.added_count == stable.prepared.changed_count == 0
    assert provenance is not None
    assert provenance.request_params["archive_lineage_complete"] is True
    assert provenance.request_params["archive_quarantine_event_count"] == 3
    assert provenance.request_params["archive_detected_revision_row_count"] == 12
    receipt = h10_collection_receipt_document(
        upgraded,
        requested_at=legacy_time + timedelta(minutes=1),
        as_of=cutoff,
    )
    assert receipt["archive_declared_correction_count"] == 1
    assert receipt["archive_quarantine_event_count"] == 3
    assert receipt["archive_detected_revision_event_count"] == 1
    assert receipt["archive_detected_revision_row_count"] == 12
    assert receipt["archive_index_sha256"] == "b" * 64
    assert receipt["archive_correction_equivalent_components"] == list(
        H10_ARCHIVE_CORRECTION_EQUIVALENT_COMPONENTS
    )
