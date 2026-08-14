from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from regime_lab.data import HealthStatus, Observation, weekly_asof_join


UTC = timezone.utc
RETRIEVED = datetime(2024, 2, 1, tzinfo=UTC)


def _observation(
    value: float | None,
    *,
    period_end: date,
    available_at: datetime,
    revision_seq: int,
) -> Observation:
    return Observation(
        source="alfred",
        series_id="SERIES",
        observed_period_end=period_end,
        value=value,
        released_at=available_at,
        available_at=available_at,
        vintage_date=available_at.date(),
        retrieved_at=RETRIEVED,
        revision_seq=revision_seq,
        raw_sha256=f"hash-{value}",
    )


def test_weekly_asof_join_never_uses_future_revision_or_release() -> None:
    records = [
        _observation(
            100,
            period_end=date(2024, 1, 1),
            available_at=datetime(2024, 1, 5, tzinfo=UTC),
            revision_seq=0,
        ),
        _observation(
            90,
            period_end=date(2024, 1, 1),
            available_at=datetime(2024, 1, 12, tzinfo=UTC),
            revision_seq=1,
        ),
        _observation(
            110,
            period_end=date(2024, 1, 8),
            available_at=datetime(2024, 1, 15, tzinfo=UTC),
            revision_seq=0,
        ),
    ]
    cutoffs = [
        datetime(2024, 1, 10, tzinfo=UTC),
        datetime(2024, 1, 13, tzinfo=UTC),
        datetime(2024, 1, 16, tzinfo=UTC),
    ]

    joined = weekly_asof_join(cutoffs, records)

    assert [row.value for row in joined] == [100, 90, 110]
    assert [row.revision_seq for row in joined] == [0, 1, 0]
    assert all(row.available_at is not None and row.available_at <= row.cutoff for row in joined)


def test_h8_borrowings_first_populates_only_after_release_cutoff() -> None:
    release = datetime(2019, 8, 9, 22, tzinfo=UTC)
    record = Observation(
        source="alfred",
        series_id="H8B3094NCBA",
        observed_period_end=date(2019, 7, 31),
        value=1_996_718.3,
        released_at=release,
        available_at=release,
        vintage_date=release.date(),
        retrieved_at=datetime(2019, 8, 20, tzinfo=UTC),
        raw_sha256="h8-first-vintage",
    )
    cutoffs = (
        # The H.8 release is after the Friday 16:00 ET model cutoff.
        datetime(2019, 8, 9, 20, tzinfo=UTC),
        datetime(2019, 8, 16, 20, tzinfo=UTC),
    )

    joined = weekly_asof_join(
        cutoffs,
        (record,),
        required_series=(("alfred", "H8B3094NCBA"),),
    )

    assert joined[0].value is None
    assert joined[0].quality_status is HealthStatus.UNAVAILABLE
    assert joined[1].value == 1_996_718.3
    assert joined[1].available_at == release
    assert joined[1].available_at <= joined[1].cutoff


def test_weekly_asof_join_marks_missing_required_and_stale_explicitly() -> None:
    record = _observation(
        100,
        period_end=date(2024, 1, 1),
        available_at=datetime(2024, 1, 5, tzinfo=UTC),
        revision_seq=0,
    )
    cutoff = datetime(2024, 1, 20, tzinfo=UTC)

    joined = weekly_asof_join(
        [cutoff],
        [record],
        required_series=(("alfred", "SERIES"), ("alfred", "MISSING")),
        max_age_by_series={"SERIES": timedelta(days=7)},
    )

    assert joined[0].quality_status is HealthStatus.STALE
    assert joined[0].age_days == 19
    assert joined[0].is_filled
    assert joined[1].value is None
    assert joined[1].quality_status is HealthStatus.UNAVAILABLE


def test_incremental_asof_defers_observation_period_that_is_not_yet_eligible() -> None:
    records = [
        _observation(
            100,
            period_end=date(2024, 1, 1),
            available_at=datetime(2024, 1, 2, tzinfo=UTC),
            revision_seq=0,
        ),
        # Defensive edge case: available timestamp precedes the declared period
        # end.  It must not leak at the first cutoff, but must become eligible
        # without rescanning the complete series at the second cutoff.
        _observation(
            200,
            period_end=date(2024, 1, 20),
            available_at=datetime(2024, 1, 5, tzinfo=UTC),
            revision_seq=0,
        ),
    ]

    joined = weekly_asof_join(
        [
            datetime(2024, 1, 10, tzinfo=UTC),
            datetime(2024, 1, 21, tzinfo=UTC),
        ],
        records,
    )

    assert [row.value for row in joined] == [100, 200]
    assert all(row.available_at is not None and row.available_at <= row.cutoff for row in joined)


def test_latest_null_revision_tombstones_only_its_period_then_forward_fills() -> None:
    records = [
        _observation(
            80,
            period_end=date(2023, 12, 25),
            available_at=datetime(2024, 1, 2, tzinfo=UTC),
            revision_seq=0,
        ),
        _observation(
            100,
            period_end=date(2024, 1, 1),
            available_at=datetime(2024, 1, 5, tzinfo=UTC),
            revision_seq=0,
        ),
        _observation(
            None,
            period_end=date(2024, 1, 1),
            available_at=datetime(2024, 1, 12, tzinfo=UTC),
            revision_seq=1,
        ),
        _observation(
            95,
            period_end=date(2024, 1, 1),
            available_at=datetime(2024, 1, 19, tzinfo=UTC),
            revision_seq=2,
        ),
    ]
    cutoffs = [
        datetime(2024, 1, 10, tzinfo=UTC),
        datetime(2024, 1, 13, tzinfo=UTC),
        datetime(2024, 1, 20, tzinfo=UTC),
    ]

    joined = weekly_asof_join(cutoffs, records)
    audited = weekly_asof_join(
        [cutoffs[1]],
        records,
        include_missing_values=True,
    )
    single_period = weekly_asof_join(
        [cutoffs[1]],
        records[1:3],
    )

    assert [row.value for row in joined] == [100, 80, 95]
    # The same period's old 100 never reappears; model mode may still
    # forward-fill the prior non-null observation period across a missing day.
    assert joined[1].observed_period_end == date(2023, 12, 25)
    assert single_period[0].value is None
    assert single_period[0].quality_status is HealthStatus.UNAVAILABLE
    assert audited[0].value is None
    assert audited[0].observed_period_end == date(2024, 1, 1)
    assert audited[0].vintage_date == date(2024, 1, 12)
    assert audited[0].revision_seq == 1
    assert audited[0].quality_status is HealthStatus.UNAVAILABLE
