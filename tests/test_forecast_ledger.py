from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import stat

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.decision_shadow import load_decision_shadow_spec
from regime_lab.data import AsOfValue, HealthStatus
from regime_lab.forecast_ledger import (
    ConflictingEvaluationError,
    ConflictingForecastError,
    DuplicateEvaluationError,
    DuplicateForecastError,
    ForecastEvaluationEntry,
    ForecastLedger,
    ForecastLedgerEntry,
    OperationalInput,
    build_research_replay_input_document,
    mature_forecast_evaluations,
    operational_input_manifest_sha256,
)
from regime_lab.integrity import canonical_json_sha256_v1


UTC = timezone.utc
SOURCE_FINAL = datetime(2026, 8, 14, 20, 15, tzinfo=UTC)
FIRST_SEEN = datetime(2026, 8, 20, 1, 33, 35, tzinfo=UTC)
DECISION = datetime(2026, 8, 21, 20, 0, tzinfo=UTC)
TARGET = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)


def _operational_input(
    *,
    source: str = "alpha_vantage",
    series_id: str = "SPY.adjusted_close",
    first_seen: datetime = FIRST_SEEN,
    raw_sha256: str = "d" * 64,
) -> OperationalInput:
    return OperationalInput(
        source=source,
        series_id=series_id,
        observed_period_end=date(2026, 8, 14),
        source_released_at=SOURCE_FINAL,
        provider_first_seen_at=first_seen,
        system_retrieved_at=first_seen,
        revision_seq=0,
        raw_sha256=raw_sha256,
    )


def _entry(
    *,
    decision_at: datetime = DECISION,
    target_at: datetime = TARGET,
    operational_inputs: tuple[OperationalInput, ...] | None = None,
) -> ForecastLedgerEntry:
    inputs = operational_inputs or (_operational_input(),)
    return ForecastLedgerEntry(
        origin_week=date(2026, 8, 14),
        decision_at=decision_at,
        target_at=target_at,
        label_spec_sha256="a" * 64,
        model_manifest_sha256="b" * 64,
        input_snapshot_sha256="c" * 64,
        operational_inputs=inputs,
        forecast={
            "predicted_state": "risk_on",
            "probabilities": {
                "risk_off": 0.2,
                "risk_on": 0.5,
                "transition": 0.3,
            },
        },
    )


def _v2_entry(
    *,
    origin_week: date = date(2026, 8, 21),
    target_week: date = date(2026, 8, 28),
    probabilities: dict[str, float] | None = None,
    input_snapshot_sha256: str = "c" * 64,
) -> ForecastLedgerEntry:
    decision_at = datetime.combine(
        origin_week, datetime.min.time(), tzinfo=UTC
    ).replace(hour=20)
    target_at = datetime.combine(
        target_week, datetime.min.time(), tzinfo=UTC
    ).replace(hour=20)
    scheduled_entry_at = datetime.combine(
        target_week - timedelta(days=4),
        datetime.min.time(),
        tzinfo=UTC,
    ).replace(hour=13, minute=30)
    probability_row = probabilities or {
        "risk_on": 0.6,
        "transition": 0.3,
        "risk_off": 0.1,
    }
    spec = load_decision_shadow_spec()
    spec_identity = {
        "path": "config/decision-shadow-v2.json",
        "sha256": canonical_json_sha256_v1(spec),
        "spec_id": spec["spec_id"],
    }
    signal = {
        "origin_date": origin_week.isoformat(),
        "target_week": target_week.isoformat(),
        "scheduled_entry_at": scheduled_entry_at.isoformat(),
        "decision_at": decision_at.isoformat(),
        "forecast_model": "causal_dynamic_ensemble",
        "status": "scheduled",
        "action": "trade_at_scheduled_open",
    }
    inputs = (_operational_input(),)
    return ForecastLedgerEntry(
        origin_week=origin_week,
        decision_at=decision_at,
        target_at=target_at,
        label_spec_sha256="a" * 64,
        model_manifest_sha256="b" * 64,
        input_snapshot_sha256=input_snapshot_sha256,
        operational_inputs=inputs,
        forecast={
            "schema_version": "regime-operational-forecast-ledger/1",
            "official": {"date": target_week.isoformat()},
            "selection": {"operating_champion": "causal_dynamic_ensemble"},
            "model_forecasts": [
                {
                    "model": "causal_dynamic_ensemble",
                    "date": target_week.isoformat(),
                    "state": max(probability_row, key=probability_row.__getitem__),
                    "probabilities": probability_row,
                }
            ],
            "decision_shadow": {
                "schema_version": "regime-prospective-decision-shadow/2",
                "spec": spec_identity,
                "spec_snapshot": spec,
                "execution_contract": spec["execution"],
                "current_signal": signal,
            },
        },
    )


def _price_panel(*weeks: date) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(list(weeks)))
    rows: list[dict[str, float]] = []
    for offset, _ in enumerate(weeks):
        spy_open = 100.0 + offset
        spy_close = spy_open + 1.0
        tlt_open = 100.0 - 0.5 * offset
        tlt_close = tlt_open + 0.25
        rows.append(
            {
                "spy_close": spy_close,
                "spy_raw_open": spy_open,
                "spy_raw_close": spy_close,
                "spy_dividend_amount": 0.0,
                "tlt_close": tlt_close,
                "tlt_raw_open": tlt_open,
                "tlt_raw_close": tlt_close,
                "tlt_dividend_amount": 0.0,
            }
        )
    return pd.DataFrame(rows, index=index, dtype=float)


def _states(*weeks: date) -> pd.Series:
    return pd.Series(
        ["risk_on" for _ in weeks],
        index=pd.DatetimeIndex(pd.to_datetime(list(weeks))),
        dtype=object,
    )


def test_operational_input_preserves_the_exact_asof_revision_identity() -> None:
    value = AsOfValue(
        cutoff=DECISION,
        source="alpha_vantage",
        series_id="SPY.adjusted_close",
        value=643.0,
        observed_period_end=date(2026, 8, 14),
        released_at=SOURCE_FINAL,
        source_released_at=SOURCE_FINAL,
        available_at=FIRST_SEEN,
        provider_first_seen_at=FIRST_SEEN,
        system_retrieved_at=FIRST_SEEN,
        vintage_date=FIRST_SEEN.date(),
        revision_seq=3,
        raw_sha256="9" * 64,
        age_days=7,
        release_lag_days=6,
        is_filled=True,
        quality_status=HealthStatus.OK,
    )

    operational = OperationalInput.from_asof_value(value)

    assert operational.observed_period_end == value.observed_period_end
    assert operational.source_released_at == value.source_released_at
    assert operational.provider_first_seen_at == value.provider_first_seen_at
    assert operational.system_retrieved_at == value.system_retrieved_at
    assert operational.revision_seq == 3
    assert operational.raw_sha256 == "9" * 64


def test_forecast_ledger_roundtrip_is_append_only_and_private(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    entry = _entry()

    with ForecastLedger(path, clock=lambda: DECISION) as ledger:
        ledger.append(entry)
        assert ledger.read(entry.key) == entry
        assert ledger.list_entries() == (entry,)

        with pytest.raises(DuplicateForecastError, match="already exists"):
            ledger.append(entry)
        with pytest.raises(ConflictingForecastError, match="different content"):
            ledger.append(
                replace(
                    entry,
                    forecast={
                        "predicted_state": "risk_off",
                        "probabilities": {
                            "risk_off": 0.6,
                            "risk_on": 0.2,
                            "transition": 0.2,
                        },
                    },
                )
            )

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger._connection.execute(
                "UPDATE forecast_ledger SET forecast_json = '{}'"
            )
        ledger._connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger._connection.execute("DELETE FROM forecast_ledger")
        ledger._connection.rollback()
        assert ledger.list_entries() == (entry,)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_public_summary_hashes_only_ordered_primary_keys_and_can_bind_pending() -> None:
    first = _entry()
    second = replace(
        _entry(
            decision_at=datetime(2026, 8, 28, 20, 0, tzinfo=UTC),
            target_at=datetime(2026, 9, 4, 20, 0, tzinfo=UTC),
        ),
        origin_week=date(2026, 8, 21),
    )
    with ForecastLedger(":memory:", clock=lambda: DECISION) as ledger:
        empty = ledger.public_summary()
        pending = ledger.public_summary(pending_key=first.key)
        ledger.append(first)
        recorded = ledger.public_summary()
        two = ledger.public_summary(pending_key=second.key)

    assert empty["entry_count"] == 0
    assert pending == recorded
    assert recorded["entry_count"] == 1
    assert recorded["hash_scope"] == "ordered_ledger_primary_keys_only"
    assert two["entry_count"] == 2
    assert two["key_manifest_sha256"] != recorded["key_manifest_sha256"]


def test_2026_08_14_input_cannot_be_claimed_before_its_first_seen_clock() -> None:
    with pytest.raises(
        ValueError,
        match="provider_first_seen_at exceeds decision_at",
    ):
        _entry(
            decision_at=datetime(2026, 8, 14, 20, 0, tzinfo=UTC),
            target_at=datetime(2026, 8, 21, 20, 0, tzinfo=UTC),
        )

    accepted = _entry()
    assert accepted.operational_inputs[0].operating_available_at == FIRST_SEEN
    assert accepted.operational_inputs[0].provider_first_seen_at <= accepted.decision_at


def test_forecast_ledger_rejects_non_forward_target_and_future_system_retrieval() -> None:
    with pytest.raises(ValueError, match="strictly before"):
        _entry(target_at=DECISION)

    future_retrieval = _operational_input(
        first_seen=datetime(2026, 8, 22, 0, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="provider_first_seen_at exceeds decision_at"):
        _entry(operational_inputs=(future_retrieval,))


def test_operational_input_manifest_hash_is_order_independent_and_clock_bound() -> None:
    first = _operational_input()
    second = _operational_input(
        source="alfred",
        series_id="UNRATE",
        raw_sha256="e" * 64,
    )

    forward = operational_input_manifest_sha256((first, second))
    reversed_order = operational_input_manifest_sha256((second, first))
    later_first_seen = operational_input_manifest_sha256(
        (
            replace(
                first,
                provider_first_seen_at=FIRST_SEEN.replace(second=36),
                system_retrieved_at=FIRST_SEEN.replace(second=36),
            ),
            second,
        )
    )

    assert forward == reversed_order
    assert forward != later_first_seen
    assert len(forward) == 64


def test_research_replay_input_binds_distinct_research_and_live_inputs() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-14T20:00:00+00:00"),
            pd.Timestamp("2026-08-21T20:00:00+00:00"),
        ]
    )
    canonical = pd.DataFrame(
        {"spy_close": [640.0, 645.0], "tlt_close": [89.0, 90.0]},
        index=index,
    )
    states = pd.Series(["transition", "risk_on"], index=index, dtype=object)
    source = AsOfValue(
        cutoff=DECISION,
        source="alpha_vantage",
        series_id="SPY.adjusted_close",
        value=645.0,
        observed_period_end=date(2026, 8, 21),
        released_at=DECISION,
        source_released_at=DECISION,
        available_at=DECISION,
        provider_first_seen_at=DECISION,
        system_retrieved_at=DECISION,
        vintage_date=DECISION.date(),
        revision_seq=0,
        raw_sha256="e" * 64,
        age_days=0,
        release_lag_days=0,
        is_filled=False,
        quality_status=HealthStatus.OK,
    )

    document = build_research_replay_input_document(
        input_vintages=(source,),
        availability_basis="reconstructed_market",
        source_observation_count=2,
        canonical=canonical,
        states=states,
        data_as_of=DECISION,
        operational_input_snapshot_sha256="d" * 64,
    )

    assert document["input_vintages"]["count"] == 1
    assert document["input_vintages"]["sha256"] != "d" * 64
    assert document["canonical_panel"]["rows"] == 2
    assert document["state_membership"]["rows"] == 2
    assert document["operational_generation_input_snapshot_sha256"] == "d" * 64


def test_forecast_ledger_validates_all_hashes_and_key_timestamps() -> None:
    with pytest.raises(ValueError, match="raw_sha256"):
        _operational_input(raw_sha256="not-a-hash")
    with pytest.raises(ValueError, match="label_spec_sha256"):
        replace(_entry(), label_spec_sha256="not-a-hash")
    with pytest.raises(ValueError, match="origin_week"):
        replace(_entry(), origin_week=date(2026, 8, 22))


def test_due_forecast_matures_once_with_split_safe_self_financing_accounting(
    tmp_path: Path,
) -> None:
    entry = _v2_entry()
    prices = _price_panel(date(2026, 8, 21), date(2026, 8, 28))
    states = _states(date(2026, 8, 21), date(2026, 8, 28))
    evaluated_at = datetime(2026, 8, 28, 21, 0, tzinfo=UTC)

    with ForecastLedger(tmp_path / "ledger.sqlite3", clock=lambda: evaluated_at) as ledger:
        ledger.append(entry)
        report = mature_forecast_evaluations(
            ledger,
            canonical=prices,
            states=states,
            evaluated_at=evaluated_at,
        )
        assert len(report.appended) == 1
        assert report.unresolved_due == {}
        evaluation = ledger.read_evaluation(entry.key)
        assert evaluation is not None
        assert evaluation.status == "completed"
        document = evaluation.evaluation
        assert document["forecast"]["model"] == "causal_dynamic_ensemble"
        assert document["forecast"]["probabilities"] == {
            "risk_on": 0.6,
            "transition": 0.3,
            "risk_off": 0.1,
        }
        assert document["actual_next_state"] == "risk_on"
        assert document["prices"]["assets"]["SPY"]["raw_open"] == 101.0
        assert document["prices"]["assets"]["SPY"]["raw_close"] == 102.0
        assert document["portfolio"]["prior_source"] == "cash_genesis"
        assert document["execution"]["turnover"] == pytest.approx(1.0)
        assert document["execution"]["one_way_turnover_bps"] == 10.0
        assert document["execution"]["transaction_cost_rate"] == pytest.approx(
            0.001
        )
        assert document["returns"]["net_return"] < document["returns"][
            "gross_return"
        ]

        first_summary = ledger.public_summary()
        repeated = mature_forecast_evaluations(
            ledger,
            canonical=prices,
            states=states,
            evaluated_at=evaluated_at,
        )
        assert repeated.appended == ()
        assert ledger.public_summary() == first_summary
        assert first_summary["status"] == "completed"
        assert first_summary["performance"]["weeks"] == 1
        assert first_summary["performance"]["forecast_hit_count"] == 1

        with pytest.raises(DuplicateEvaluationError, match="already exists"):
            ledger.append_evaluation(evaluation)
        conflicting_document = json.loads(json.dumps(document))
        conflicting_document["actual_next_state"] = "risk_off"
        conflicting = ForecastEvaluationEntry(
            forecast_key=entry.key,
            evaluated_at=evaluated_at,
            status="completed",
            evaluation=conflicting_document,
        )
        with pytest.raises(ConflictingEvaluationError, match="different evaluation"):
            ledger.append_evaluation(conflicting)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger._connection.execute(
                "UPDATE forecast_evaluation_ledger SET status = 'partial'"
            )
        ledger._connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger._connection.execute("DELETE FROM forecast_evaluation_ledger")
        ledger._connection.rollback()


def test_due_missing_target_data_stays_unresolved_and_retries_without_sealing(
    tmp_path: Path,
) -> None:
    entry = _v2_entry()
    complete = _price_panel(date(2026, 8, 21), date(2026, 8, 28))
    missing = complete.copy()
    missing.loc[pd.Timestamp("2026-08-28"), "spy_raw_open"] = np.nan
    states = _states(date(2026, 8, 21), date(2026, 8, 28))
    evaluated_at = datetime(2026, 8, 28, 21, 0, tzinfo=UTC)

    with ForecastLedger(tmp_path / "ledger.sqlite3", clock=lambda: evaluated_at) as ledger:
        ledger.append(entry)
        report = mature_forecast_evaluations(
            ledger,
            canonical=missing,
            states=states,
            evaluated_at=evaluated_at,
        )
        assert report.appended == ()
        assert report.unresolved_due == {entry.key: "target_prices_unavailable"}
        assert ledger.read_evaluation(entry.key) is None
        summary = ledger.public_summary(unresolved_due=report.unresolved_due)
        assert summary["status"] == "partial"
        assert summary["unresolved_due_evaluation_count"] == 1
        assert summary["partial_evaluation_count"] == 0

        retried = mature_forecast_evaluations(
            ledger,
            canonical=complete,
            states=states,
            evaluated_at=evaluated_at,
        )
        assert len(retried.appended) == 1
        assert retried.appended[0].status == "completed"


def test_contiguous_maturity_drifts_prior_weights_and_excludes_target_dividend(
    tmp_path: Path,
) -> None:
    first = _v2_entry()
    second = _v2_entry(
        origin_week=date(2026, 8, 28),
        target_week=date(2026, 9, 4),
        probabilities={"risk_on": 0.1, "transition": 0.2, "risk_off": 0.7},
        input_snapshot_sha256="d" * 64,
    )
    prices = _price_panel(
        date(2026, 8, 21), date(2026, 8, 28), date(2026, 9, 4)
    )
    target = pd.Timestamp("2026-09-04")
    # Exact 2-for-1 split with a target-week distribution.  The adjusted total
    # factor carries both, while the operational holding leg remains raw
    # open-to-close and therefore does not credit an unknown ex-date dividend.
    prices.loc[target, "spy_raw_open"] = 51.5
    prices.loc[target, "spy_raw_close"] = 52.0
    prices.loc[target, "spy_dividend_amount"] = 0.5
    prices.loc[target, "spy_close"] = 105.0
    states = _states(
        date(2026, 8, 21), date(2026, 8, 28), date(2026, 9, 4)
    )
    evaluated_at = datetime(2026, 9, 4, 21, 0, tzinfo=UTC)

    with ForecastLedger(tmp_path / "ledger.sqlite3", clock=lambda: evaluated_at) as ledger:
        ledger.append(first)
        ledger.append(second)
        mature_forecast_evaluations(
            ledger,
            canonical=prices,
            states=states,
            evaluated_at=evaluated_at,
        )
        first_evaluation = ledger.read_evaluation(first.key)
        second_evaluation = ledger.read_evaluation(second.key)
        assert first_evaluation is not None and second_evaluation is not None
        portfolio = second_evaluation.evaluation["portfolio"]
        assert portfolio["prior_source"] == "prior_completed_evaluation"
        assert portfolio["prior_forecast_key"] == first.key.as_dict()
        assert portfolio["gap_relatives"]["SPY"] == pytest.approx(
            2.0 * 51.5 / 102.0
        )
        assert portfolio["pretrade_weights"] != portfolio["prior_close_weights"]
        assert second_evaluation.evaluation["returns"][
            "open_to_close_asset_returns"
        ]["SPY"] == pytest.approx(52.0 / 51.5 - 1.0)
        assert second_evaluation.evaluation["prices"]["assets"]["SPY"][
            "dividend_amount"
        ] == 0.5
        assert second_evaluation.evaluation["execution"]["turnover"] > 0.0


def test_terminal_sequence_gap_does_not_poison_the_next_contiguous_segment(
    tmp_path: Path,
) -> None:
    first = _v2_entry()
    gap = _v2_entry(
        origin_week=date(2026, 9, 4),
        target_week=date(2026, 9, 11),
        input_snapshot_sha256="d" * 64,
    )
    restarted = _v2_entry(
        origin_week=date(2026, 9, 11),
        target_week=date(2026, 9, 18),
        input_snapshot_sha256="e" * 64,
    )
    weeks = tuple(
        date(2026, 8, 21) + timedelta(days=7 * offset)
        for offset in range(5)
    )
    prices = _price_panel(*weeks)
    states = _states(*weeks)
    evaluated_at = datetime(2026, 9, 18, 21, 0, tzinfo=UTC)

    with ForecastLedger(tmp_path / "ledger.sqlite3", clock=lambda: evaluated_at) as ledger:
        for entry in (first, gap, restarted):
            ledger.append(entry)
        report = mature_forecast_evaluations(
            ledger,
            canonical=prices,
            states=states,
            evaluated_at=evaluated_at,
        )
        assert [item.status for item in report.appended] == [
            "completed",
            "partial",
            "completed",
        ]
        gap_evaluation = ledger.read_evaluation(gap.key)
        restarted_evaluation = ledger.read_evaluation(restarted.key)
        assert gap_evaluation is not None
        assert gap_evaluation.evaluation["reason"] == "forecast_sequence_gap"
        assert restarted_evaluation is not None
        assert (
            restarted_evaluation.evaluation["portfolio"]["prior_source"]
            == "cash_segment_restart_after_terminal_partial"
        )


def test_opening_a_forecast_only_v1_database_adds_evaluation_schema_in_place(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    entry = _entry()
    with ForecastLedger(path, clock=lambda: DECISION) as ledger:
        ledger.append(entry)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE forecast_evaluation_ledger")
        connection.commit()

    with ForecastLedger(path, clock=lambda: DECISION) as migrated:
        assert migrated.read(entry.key) == entry
        assert migrated.list_evaluations() == ()
        table = migrated._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'forecast_evaluation_ledger'"
        ).fetchone()
        assert table is not None
