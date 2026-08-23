from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from regime_lab.analysis.fx import (
    FXFeatureConfig,
    FX_MAX_OBSERVATION_AGE_DAYS,
    build_fx_features,
    build_official_archive_fx_features,
    fx_context_at,
)
from regime_lab.data.h10 import (
    ByteResponse,
    DEFAULT_ALLOWED_FX,
    FIXED_BILATERAL_PANEL,
    H10Client,
    H10Config,
    H10SchemaError,
    H10TimestampError,
    SERIES_CATALOG,
)
from regime_lab.data import HealthStatus, Observation


UTC = timezone.utc
FIXTURE = Path(__file__).parent / "fixtures" / "h10_fixture.xml"
LAST_MODIFIED = "Mon, 17 Aug 2026 20:15:15 GMT"
FIRST_SEEN = datetime(2026, 8, 17, 20, 20, tzinfo=UTC)
RETRIEVED = datetime(2026, 8, 17, 20, 21, tzinfo=UTC)


def _fixture_zip(xml: bytes | None = None) -> bytes:
    payload = FIXTURE.read_bytes() if xml is None else xml
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("H10_data.xml", payload)
        archive.writestr("H10_H10.xsd", b"<schema/>")
        archive.writestr("H10_struct.xml", b"<structure/>")
        archive.writestr("frb_common.xsd", b"<schema/>")
    return buffer.getvalue()


class _FakeByteTransport:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[tuple[str, float]] = []

    def get_bytes(self, url: str, *, timeout: float) -> ByteResponse:
        self.calls.append((url, timeout))
        return ByteResponse(
            body=self.body,
            headers={"Last-Modified": LAST_MODIFIED, "ETag": '"fixture"'},
            retrieved_at=RETRIEVED,
        )


def test_parser_creates_pit_observations_and_preserves_quote_metadata() -> None:
    transport = _FakeByteTransport(_fixture_zip())
    client = H10Client(
        H10Config(allowed_fx=("BRD", "EUR", "KRW")),
        transport=transport,
    )

    result = client.collect(first_seen_at=FIRST_SEEN)

    assert len(transport.calls) == 1
    assert result.release.message_id == "H10"
    assert result.release.sender_id == "FRB"
    assert result.release.message_prepared_raw == "2026-08-17T00:00:00"
    assert result.release.source_object_modified_at == datetime(
        2026, 8, 17, 20, 15, 15, tzinfo=UTC
    )
    assert result.release.first_seen_at == FIRST_SEEN
    assert result.release.retrieved_at == RETRIEVED
    assert result.release.release_date_et == date(2026, 8, 17)
    assert result.release.etag == '"fixture"'

    metadata = {item.fx_code: item for item in result.series}
    assert set(metadata) == {"BRD", "EUR", "KRW"}
    assert metadata["BRD"].usd_strength_sign == 1
    assert metadata["EUR"].usd_strength_sign == -1
    assert metadata["EUR"].quote_convention.value == "usd_per_foreign"
    assert metadata["KRW"].usd_strength_sign == 1
    assert metadata["KRW"].quote_convention.value == "foreign_per_usd"

    brd = [record for record in result.records if record.metadata["fx_code"] == "BRD"]
    # The FREQ=129 BRD fixture row is not admitted.
    assert [record.observed_period_end for record in brd] == [
        date(2026, 8, 13),
        date(2026, 8, 14),
    ]
    assert brd[-1].value == 102.0
    assert brd[-1].released_at == datetime(
        2026, 8, 17, 20, 15, 15, tzinfo=UTC
    )
    assert brd[-1].available_at == FIRST_SEEN
    assert brd[-1].retrieved_at == RETRIEVED
    assert brd[-1].vintage_date == FIRST_SEEN.date()

    missing = next(
        record
        for record in result.records
        if record.metadata["fx_code"] == "EUR"
        and record.observed_period_end == date(2026, 7, 3)
    )
    assert missing.value is None
    assert missing.quality_status is HealthStatus.UNAVAILABLE
    assert missing.metadata["obs_status"] == "ND"
    assert missing.metadata["raw_value_token"] == "-9999"


def test_parser_default_allowlist_excludes_krw() -> None:
    result = H10Client(
        H10Config(),
        transport=_FakeByteTransport(_fixture_zip()),
    ).collect(first_seen_at=FIRST_SEEN)

    assert {item.fx_code for item in result.series} == set(DEFAULT_ALLOWED_FX)
    assert "KRW" not in {record.metadata["fx_code"] for record in result.records}


def test_live_client_defaults_first_seen_to_retrieval_time() -> None:
    result = H10Client(
        H10Config(allowed_fx=("BRD",)),
        transport=_FakeByteTransport(_fixture_zip()),
    ).collect()

    assert result.release.first_seen_at == RETRIEVED
    assert {record.available_at for record in result.records} == {RETRIEVED}


def test_client_rejects_explicit_first_seen_after_retrieval() -> None:
    client = H10Client(
        H10Config(allowed_fx=("BRD",)),
        transport=_FakeByteTransport(_fixture_zip()),
    )

    with pytest.raises(
        H10TimestampError,
        match="retrieved_at must not precede first_seen_at",
    ):
        client.collect(first_seen_at=RETRIEVED + timedelta(microseconds=1))


def test_parser_fails_closed_when_provider_series_id_changes() -> None:
    changed = FIXTURE.read_bytes().replace(
        b'RXI_N.B.KO"',
        b'RXI_N.B.XX"',
    )
    client = H10Client(
        H10Config(allowed_fx=("KRW",)),
        transport=_FakeByteTransport(_fixture_zip(changed)),
    )

    with pytest.raises(H10SchemaError, match="series name changed"):
        client.collect(first_seen_at=FIRST_SEEN)


def test_parser_rejects_source_timestamp_after_first_seen() -> None:
    with pytest.raises(H10SchemaError, match="Last-Modified is after"):
        H10Client(
            H10Config(allowed_fx=("BRD",)),
            transport=_FakeByteTransport(_fixture_zip()),
        ).collect(
            first_seen_at=datetime(2026, 8, 17, 20, 10, tzinfo=UTC)
        )


_CORE_SLOPES = {"BRD": 0.010, "AFE": 0.005, "EME": 0.015}
_PANEL_SLOPES = dict(
    zip(
        FIXED_BILATERAL_PANEL,
        (-0.008, -0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006, 0.008),
        strict=True,
    )
)


def _available_after(observed: date) -> datetime:
    return datetime.combine(
        observed + timedelta(days=3),
        time(20, 15),
        tzinfo=UTC,
    )


def _synthetic_observation(
    fx_code: str,
    observed: date,
    usd_log_level: float,
    *,
    missing: bool = False,
) -> Observation:
    spec = SERIES_CATALOG[fx_code]
    available = _available_after(observed)
    raw_value = None if missing else math_exp(spec.usd_strength_sign * usd_log_level)
    status = "ND" if missing else "A"
    return Observation(
        source="frb_h10",
        series_id=f"H10|FX={fx_code}|FREQ=9",
        observed_period_end=observed,
        value=raw_value,
        released_at=available,
        available_at=available,
        vintage_date=available.date(),
        retrieved_at=available + timedelta(minutes=1),
        units="Currency" if fx_code not in ("BRD", "AFE", "EME") else "Index",
        license_class="federal_reserve_board_public_domain_citation_requested",
        quality_status=(HealthStatus.UNAVAILABLE if missing else HealthStatus.OK),
        raw_sha256=f"{fx_code}-{observed}-{status}",
        metadata={
            "fx_code": fx_code,
            "frequency_code": "9",
            "series_name": spec.series_name,
            "quote_convention": spec.quote_convention.value,
            "usd_strength_sign": spec.usd_strength_sign,
            "obs_status": status,
        },
    )


def math_exp(value: float) -> float:
    return float(np.exp(value))


def _synthetic_panel(
    *,
    missing_last: frozenset[str] = frozenset(),
    include_krw: bool = False,
) -> tuple[list[Observation], pd.DatetimeIndex]:
    dates = pd.date_range("2025-12-05", periods=36, freq="W-FRI")
    slopes = {**_CORE_SLOPES, **_PANEL_SLOPES}
    if include_krw:
        slopes["KRW"] = 1.0
    records: list[Observation] = []
    for position, timestamp in enumerate(dates):
        for fx_code, slope in slopes.items():
            base = 4.0 if fx_code in _CORE_SLOPES else 0.2
            records.append(
                _synthetic_observation(
                    fx_code,
                    timestamp.date(),
                    base + slope * position,
                    missing=(
                        position == len(dates) - 1 and fx_code in missing_last
                    ),
                )
            )
    return records, dates


def test_fx_features_normalize_direction_and_build_fixed_panel_statistics() -> None:
    records, dates = _synthetic_panel(include_krw=True)

    result = build_fx_features(records)
    at = dates[-1]
    features = result.features

    # EUR is quoted USD per EUR in the source, yet its normalized return has
    # the same positive-USD-strength sign as foreign-per-USD series.
    np.testing.assert_allclose(
        features.loc[at, "fx__eur__usd_log_return_1w"],
        _PANEL_SLOPES["EUR"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features.loc[at, "fx__eme_minus_afe__usd_log_return_13w"],
        13 * (_CORE_SLOPES["EME"] - _CORE_SLOPES["AFE"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features.loc[at, "fx__broad_minus_afe__usd_log_return_4w"],
        4 * (_CORE_SLOPES["BRD"] - _CORE_SLOPES["AFE"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features.loc[at, "fx__bilateral__median_usd_log_return_1w"],
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features.loc[at, "fx__bilateral__usd_appreciating_share_1w"],
        4 / 9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features.loc[at, "fx__bilateral__return_mad_1w"],
        0.004,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features.loc[at, "fx__brd__realized_vol_26w"],
        0.0,
        atol=1e-12,
    )
    assert result.coverage.loc[at, "core_level_count"] == 3
    assert result.coverage.loc[at, "bilateral_level_count"] == 9
    assert result.status.loc[at, "source_status"] == "ok"
    assert result.status.loc[at, "feature_status"] == "ok"
    assert result.status.iloc[0]["feature_status"] == "warming_up"
    assert "fx__krw__usd_log_level" not in features.columns
    assert "KRW" not in result.weekly_usd_log_levels.columns
    assert result.coverage.loc[at, "feature_available_at"] == pd.Timestamp(
        _available_after(at.date())
    )


def test_fx_coverage_fails_closed_below_fixed_panel_minimum() -> None:
    records, dates = _synthetic_panel(
        missing_last=frozenset({"EUR", "JPY", "GBP", "CHF"})
    )

    result = build_fx_features(records)
    at = dates[-1]

    assert result.coverage.loc[at, "bilateral_level_count"] == 5
    assert result.coverage.loc[at, "bilateral_return_1w_count"] == 5
    assert result.status.loc[at, "source_status"] == "insufficient_coverage"
    assert result.status.loc[at, "feature_status"] == "insufficient_coverage"
    assert np.isnan(
        result.features.loc[
            at,
            "fx__bilateral__median_usd_log_return_1w",
        ]
    )


def test_fx_as_of_filter_never_uses_later_first_seen_records() -> None:
    records, dates = _synthetic_panel()
    cutoff_date = dates[29]
    cutoff = _available_after(cutoff_date.date())

    as_of_result = build_fx_features(records, as_of=cutoff)
    prefix_records = [record for record in records if record.available_at <= cutoff]
    prefix_result = build_fx_features(prefix_records)

    assert as_of_result.features.index.max() == cutoff_date
    assert_frame_equal(as_of_result.features, prefix_result.features)
    assert_frame_equal(as_of_result.coverage, prefix_result.coverage)


def test_repeated_full_snapshot_preserves_original_first_seen_time() -> None:
    records, _dates = _synthetic_panel()
    repeated = [
        replace(
            record,
            available_at=record.available_at + timedelta(days=7),
            retrieved_at=record.retrieved_at + timedelta(days=7),
            vintage_date=record.vintage_date + timedelta(days=7),
            raw_sha256=f"repeat-{record.raw_sha256}",
        )
        for record in records
    ]

    original = build_fx_features(records)
    with_repeat = build_fx_features([*records, *repeated])

    assert_frame_equal(with_repeat.features, original.features)
    assert_frame_equal(with_repeat.weekly_availability, original.weekly_availability)
    assert_frame_equal(with_repeat.coverage, original.coverage)


def test_public_fx_context_enforces_availability_cutoff() -> None:
    records, dates = _synthetic_panel()
    result = build_fx_features(records)

    before_first_seen = fx_context_at(
        result,
        cutoff=_available_after(dates[0].date()) - timedelta(seconds=1),
    )
    assert before_first_seen["status"] == "unavailable"

    latest = fx_context_at(
        result,
        cutoff=_available_after(dates[-1].date()),
    )
    assert latest["status"] == "ok"
    assert latest["coverage"] == {
        "available_pairs": 9,
        "required_pairs": 9,
        "available_indexes": 3,
        "required_indexes": 3,
    }
    assert latest["bilateral_panel"] == [
        "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "MXN", "BRL"
    ]
    assert latest["direction"] == "positive_is_usd_appreciation"


def test_public_fx_context_marks_carried_observation_stale() -> None:
    records, dates = _synthetic_panel()
    result = build_fx_features(records)
    stale_cutoff = _available_after(dates[-1].date()) + timedelta(days=8)

    context = fx_context_at(result, cutoff=stale_cutoff)

    assert context["status"] == "stale"
    assert context["observation_week"] == dates[-1].date().isoformat()
    assert context["observation_age_days"] == 11
    assert context["maximum_age_days"] == FX_MAX_OBSERVATION_AGE_DAYS == 10


def test_fx_config_keeps_krw_out_of_fixed_panel() -> None:
    with pytest.raises(ValueError, match="KRW is intentionally excluded"):
        FXFeatureConfig(
            bilateral_panel=(*FIXED_BILATERAL_PANEL[:-1], "KRW")
        )


def test_official_archive_replay_uses_cutoff_vintage_and_quarantines_27_origins() -> None:
    weeks = pd.date_range("2023-01-06", periods=70, freq="W-FRI")
    slopes = {**_CORE_SLOPES, **_PANEL_SLOPES}
    records: list[Observation] = []
    for position, week in enumerate(weeks):
        for code, slope in slopes.items():
            original = _synthetic_observation(
                code,
                week.date(),
                (4.0 if code in _CORE_SLOPES else 0.2) + slope * position,
            )
            metadata = {
                **dict(original.metadata),
                "official_release_archive_ingest": True,
                "archive_chain_availability_basis": (
                    "official_archive_release_schedule"
                ),
                "availability_basis": "archived_release_date_16_15_ET",
                "archive_revision_policy": (
                    "later_official_release_preserved_as_new_vintage"
                ),
            }
            records.append(replace(original, metadata=metadata))

    corrected_week = weeks[30]
    original_eur = next(
        row
        for row in records
        if row.metadata["fx_code"] == "EUR"
        and row.observed_period_end == corrected_week.date()
    )
    correction_available = datetime.combine(
        corrected_week.date() + timedelta(days=6),
        time(4, 0),
        tzinfo=UTC,
    )
    corrected_metadata = {
        **dict(original_eur.metadata),
        "availability_basis": "date_only_conservative_next_day",
    }
    corrected = replace(
        original_eur,
        value=float(original_eur.value) * 1.01,
        released_at=correction_available,
        available_at=correction_available,
        vintage_date=(corrected_week.date() + timedelta(days=5)),
        retrieved_at=correction_available + timedelta(minutes=1),
        raw_sha256=f"correction-{original_eur.raw_sha256}",
        metadata=corrected_metadata,
    )

    result = build_official_archive_fx_features(
        [*records, corrected],
        as_of=datetime.combine(
            (weeks[-1] + timedelta(days=7)).date(),
            time(20, 0),
            tzinfo=UTC,
        ),
        correction_available_at=(correction_available,),
    )

    assert result.official_release_archive_ingest is True
    assert result.availability_basis == "official_archive_release_schedule"
    assert result.coverage.index.equals(
        pd.date_range(weeks[0], weeks[-1], freq="W-FRI")
    )
    quarantine = result.coverage["archive_correction_quarantined"]
    assert int(quarantine.sum()) == 27
    assert list(result.coverage.index[quarantine][[0, -1]]) == [
        corrected_week,
        corrected_week + timedelta(weeks=26),
    ]
    assert result.coverage.loc[
        corrected_week,
        "archive_correction_available_at",
    ] == pd.Timestamp(correction_available)
    assert result.coverage.loc[
        corrected_week,
        "archive_correction_quarantine_until_week",
    ] == corrected_week + timedelta(weeks=27)
    expected_level = (
        SERIES_CATALOG["EUR"].usd_strength_sign
        * np.log(float(corrected.value))
    )
    np.testing.assert_allclose(
        result.weekly_usd_log_levels.loc[corrected_week, "EUR"],
        expected_level,
    )
    context = fx_context_at(result, cutoff=correction_available)
    assert context["status"] == "ok"
    assert context["observation_week"] == corrected_week.date().isoformat()
