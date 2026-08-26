from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from regime_lab.cli import _operational_inputs_for_generation
from regime_lab.data import AsOfValue, HealthStatus, Observation


UTC = timezone.utc
ORIGIN = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
DECISION = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


def _asof(
    *,
    period: date = date(2026, 8, 21),
    first_seen: datetime = datetime(2026, 8, 22, 3, 0, tzinfo=UTC),
) -> AsOfValue:
    return AsOfValue(
        cutoff=ORIGIN,
        source="alpha_vantage",
        series_id="SPY.adjusted_close",
        value=645.0,
        observed_period_end=period,
        released_at=datetime(2026, 8, 21, 20, 0, tzinfo=UTC),
        source_released_at=datetime(2026, 8, 21, 20, 0, tzinfo=UTC),
        available_at=datetime(2026, 8, 21, 20, 0, tzinfo=UTC),
        provider_first_seen_at=first_seen,
        system_retrieved_at=first_seen,
        vintage_date=period,
        revision_seq=0,
        raw_sha256="a" * 64,
        age_days=0,
        release_lag_days=0,
        is_filled=False,
        quality_status=HealthStatus.OK,
    )


def _observation(*, first_seen: datetime) -> Observation:
    return Observation(
        source="frb_h10",
        series_id="DEXUSEU",
        observed_period_end=date(2026, 8, 21),
        value=1.17,
        released_at=datetime(2026, 8, 21, 21, 0, tzinfo=UTC),
        source_released_at=datetime(2026, 8, 21, 21, 0, tzinfo=UTC),
        available_at=datetime(2026, 8, 21, 21, 0, tzinfo=UTC),
        provider_first_seen_at=first_seen,
        vintage_date=date(2026, 8, 21),
        retrieved_at=first_seen,
        system_retrieved_at=first_seen,
        units="USD/EUR",
        adjustment="none",
        license_class="public_domain",
        quality_status=HealthStatus.OK,
        raw_sha256="b" * 64,
    )


def test_current_operational_forecast_can_bind_reconstructed_history() -> None:
    # The market row arrived after Friday's modeled origin but before the real
    # publication decision.  It is valid for this prospective forecast, while
    # the historical walk-forward evidence remains reconstructed OOS.
    dataset = SimpleNamespace(
        availability_basis="source",
        input_vintages=(_asof(),),
    )

    result = _operational_inputs_for_generation(
        dataset,
        origin_at=ORIGIN,
        decision_at=DECISION,
    )

    assert len(result) == 1
    assert result[0].provider_first_seen_at > ORIGIN
    assert result[0].provider_first_seen_at <= DECISION


def test_operational_binding_rejects_dataset_input_first_seen_after_decision() -> None:
    dataset = SimpleNamespace(
        availability_basis="source",
        input_vintages=(
            _asof(first_seen=datetime(2026, 8, 26, 0, 0, tzinfo=UTC)),
        ),
    )

    with pytest.raises(ValueError, match="first seen after the decision"):
        _operational_inputs_for_generation(
            dataset,
            origin_at=ORIGIN,
            decision_at=DECISION,
        )


def test_operational_binding_rejects_future_period_and_skips_late_auxiliary() -> None:
    future_period = SimpleNamespace(
        availability_basis="source",
        input_vintages=(_asof(period=date(2026, 8, 28)),),
    )
    with pytest.raises(ValueError, match="period exceeds"):
        _operational_inputs_for_generation(
            future_period,
            origin_at=ORIGIN,
            decision_at=DECISION,
        )

    valid = SimpleNamespace(
        availability_basis="source",
        input_vintages=(_asof(),),
    )
    result = _operational_inputs_for_generation(
        valid,
        additional_records=(
            _observation(first_seen=datetime(2026, 8, 26, 0, 0, tzinfo=UTC)),
        ),
        origin_at=ORIGIN,
        decision_at=DECISION,
    )
    assert {item.source for item in result} == {"alpha_vantage"}


def test_operational_binding_requires_aware_clocks_and_known_basis() -> None:
    dataset = SimpleNamespace(
        availability_basis="source",
        input_vintages=(_asof(),),
    )
    with pytest.raises(ValueError, match="origin must include a timezone"):
        _operational_inputs_for_generation(
            dataset,
            origin_at=datetime(2026, 8, 21, 20, 0),
            decision_at=DECISION,
        )
    with pytest.raises(ValueError, match="invalid availability basis"):
        _operational_inputs_for_generation(
            SimpleNamespace(availability_basis="unknown", input_vintages=()),
            origin_at=ORIGIN,
            decision_at=DECISION,
        )
