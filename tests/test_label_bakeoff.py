from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.analysis.label_spec import load_label_spec
from regime_lab.data import (
    HealthStatus,
    Observation,
    SnapshotMode,
    SnapshotProvenance,
    SQLiteSnapshotStore,
)
from regime_lab.integrity import canonical_json_sha256_v1
from regime_lab.label_bakeoff import (
    run_label_bakeoff,
    write_label_bakeoff_generation,
)


UTC = timezone.utc
AS_OF = datetime(2025, 1, 10, 12, tzinfo=UTC)


def _raw_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _observation(
    *,
    series_id: str,
    period: date,
    value: float,
    retrieved_at: datetime,
) -> Observation:
    released = datetime.combine(period, time(21, 0), tzinfo=UTC)
    return Observation(
        source="alpha_vantage",
        series_id=series_id,
        observed_period_end=period,
        value=value,
        released_at=released,
        source_released_at=released,
        available_at=released,
        provider_first_seen_at=retrieved_at,
        vintage_date=period,
        retrieved_at=retrieved_at,
        system_retrieved_at=retrieved_at,
        units="test",
        adjustment="weekly",
        license_class="private_test",
        quality_status=HealthStatus.OK,
        raw_sha256=_raw_hash(series_id, period, value),
        metadata={"field": series_id.rsplit(".", 1)[-1]},
    )


def _write_market_fixture(
    path: Path,
    *,
    periods: int = 540,
    include_split_series: bool = True,
    include_all_symbols: bool = True,
) -> None:
    symbols = tuple(
        item.symbol for item in load_label_spec("v2_broad_equity").series
    )
    if not include_all_symbols:
        symbols = ("SPY",)
    dates = pd.date_range("2010-01-01", periods=periods, freq="7D")
    retrieved = datetime(2024, 12, 31, 12, tzinfo=UTC)
    records: list[Observation] = []
    for symbol_position, symbol in enumerate(symbols):
        position = np.arange(periods, dtype=float)
        economic_price = 100.0 * np.exp(
            0.0012 * position
            + 0.10 * np.sin(position / (13.0 + symbol_position * 0.3))
            + 0.03 * np.cos(position / (5.0 + symbol_position * 0.2))
        )
        split_at = 310 + symbol_position % 9
        splits = np.ones(periods, dtype=float)
        share_factor = np.ones(periods, dtype=float)
        if split_at < periods:
            splits[split_at] = 2.0
            share_factor[split_at:] = 2.0
        raw_close = economic_price / share_factor
        dividends = np.where((position.astype(int) + symbol_position) % 13 == 0, 0.08, 0.0)
        adjusted = np.full(periods, 100.0, dtype=float)
        gross = (
            raw_close[1:] * splits[1:] + dividends[1:]
        ) / raw_close[:-1]
        adjusted[1:] = 100.0 * np.cumprod(gross)
        for row, timestamp in enumerate(dates):
            period = timestamp.date()
            values = {
                f"{symbol}.close": raw_close[row],
                f"{symbol}.dividend_amount": dividends[row],
                f"{symbol}.adjusted_close": adjusted[row],
            }
            if include_split_series:
                values[f"{symbol}.split_coefficient"] = splits[row]
            records.extend(
                _observation(
                    series_id=series_id,
                    period=period,
                    value=float(value),
                    retrieved_at=retrieved,
                )
                for series_id, value in values.items()
            )
    provenance = SnapshotProvenance(
        source="alpha_vantage",
        dataset="weekly_adjusted_etf",
        cutoff=datetime(2024, 12, 31, 21, tzinfo=UTC),
        requested_at=retrieved,
        retrieved_at=retrieved,
        quality_status=HealthStatus.OK,
        license_class="private_test",
        request_params={"snapshot_mode": SnapshotMode.FULL.value},
        response_sha256=_raw_hash("response", len(records)),
    )
    with SQLiteSnapshotStore(path) as store:
        store.write_snapshot(records, provenance)


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(map(str, value))
        for child in value.values():
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def test_exact_split_bakeoff_is_matched_derived_only_and_non_promoting(
    tmp_path: Path,
) -> None:
    database = tmp_path / "market.sqlite3"
    _write_market_fixture(database)

    result = run_label_bakeoff(
        database,
        as_of=AS_OF,
        split_policy="exact_split_series",
        generated_at=AS_OF,
    )

    assert result.complete
    label_report = result.label_audit_report
    pit_report = result.pit_replay_report
    assert label_report["matched_origins"]["rows"] == 540
    assert set(label_report["labels"]) == {
        "v1_spy_hysteresis",
        "v2_spy_pit_total_return",
        "v2_broad_equity",
    }
    assert all(
        len(item["origin_records"]) == 540
        for item in label_report["labels"].values()
    )
    assert all(
        item["fit_period"]["fit_rows"] == 520
        for item in label_report["labels"].values()
    )
    assert label_report["automatic_promotion_eligible"] is False
    assert label_report["official_label_unchanged"] is True
    assert label_report["pagan_sossounov"]["canonical_target"] is False
    assert pit_report["evidence_track"] == "reconstructed_oos"
    assert pit_report["operational_oos_claimed"] is False
    assert pit_report["lineage_contract"]["unit_split_fill_allowed"] is False
    assert all(
        item["lineage_kind"]
        == "exact_dated_raw_close_dividend_split_composite"
        for item in pit_report["symbols"].values()
    )
    assert canonical_json_sha256_v1(
        {key: value for key, value in label_report.items() if key != "sha256"}
    ) == label_report["sha256"]
    assert not {
        "raw_close",
        "dividend_amount",
        "split_coefficient",
        "price",
    }.intersection(_walk_keys(label_report))


def test_adjusted_close_composite_is_explicitly_reconstructed_not_operational(
    tmp_path: Path,
) -> None:
    database = tmp_path / "market.sqlite3"
    _write_market_fixture(database, include_split_series=False)

    result = run_label_bakeoff(
        database,
        as_of=AS_OF,
        split_policy="adjusted_close_composite",
        generated_at=AS_OF,
    )

    assert result.complete
    report = result.pit_replay_report
    assert report["split_policy"] == "adjusted_close_composite"
    assert report["operational_oos_claimed"] is False
    assert report["lineage_contract"]["adjusted_close_composite_is_historical_pit"] is False
    assert all(
        item["lineage_kind"]
        == "reconstructed_current_adjusted_close_implied_split_composite"
        for item in report["symbols"].values()
    )
    assert all(
        item["operational_eligible_rows_at_reconstructed_decision_clock"] == 0
        for item in report["symbols"].values()
    )


def test_missing_split_series_is_reported_and_generation_is_fail_closed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "market.sqlite3"
    _write_market_fixture(
        database,
        periods=12,
        include_split_series=False,
        include_all_symbols=False,
    )

    result = run_label_bakeoff(
        database,
        as_of=AS_OF,
        split_policy="exact_split_series",
        generated_at=AS_OF,
    )
    assert not result.complete
    assert result.status == "blocked_input_contract"
    missing = {
        issue["series_id"]
        for issue in result.pit_replay_report["input_issues"]
        if issue["code"] == "missing_required_series"
    }
    assert "SPY.split_coefficient" in missing
    assert "SPY.adjusted_close" not in missing
    assert result.label_audit_report["labels"] == {}
    assert result.pit_replay_report["replay_completed"] is False

    generation = write_label_bakeoff_generation(tmp_path / "output", result)
    assert (generation / "label-audit-report.json").is_file()
    assert (generation / "pit-replay-report.json").is_file()
    manifest = json.loads(
        (generation / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked_input_contract"
    assert manifest["automatic_promotion_eligible"] is False
    latest = json.loads(
        (tmp_path / "output/latest.json").read_text(encoding="utf-8")
    )
    assert latest["status"] == "blocked_input_contract"


def test_reconstructed_run_still_rejects_inputs_released_after_as_of(
    tmp_path: Path,
) -> None:
    database = tmp_path / "market.sqlite3"
    _write_market_fixture(database, periods=12, include_all_symbols=False)

    result = run_label_bakeoff(
        database,
        as_of=datetime(2010, 1, 1, 12, tzinfo=UTC),
        split_policy="exact_split_series",
        generated_at=AS_OF,
    )

    assert not result.complete
    assert result.pit_replay_report["component_coverage"]
    assert all(
        item["finite_value_rows"] == 0
        for item in result.pit_replay_report["component_coverage"]
    )
