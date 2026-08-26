from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import regime_lab.collection as collection_module
from regime_lab.data import (
    AlphaVantageClient,
    AlphaVantageConfig,
    HealthStatus,
    Observation,
    RetryPolicy,
    SQLiteSnapshotStore,
    SnapshotProvenance,
    weekly_asof_join,
)


UTC = timezone.utc
FIRST_SEEN_2026_08_14 = datetime(2026, 8, 20, 1, 33, 35, tzinfo=UTC)
SOURCE_FINAL_2026_08_14 = datetime(2026, 8, 14, 20, 15, tzinfo=UTC)
PIT_REPLAY_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "pit-replay-2026-08-14.json"
)


class _August14AlphaTransport:
    def get_json(
        self,
        _url: str,
        params: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Mapping[str, Any]:
        del params, timeout
        return {
            "Meta Data": {"2. Symbol": "SPY"},
            "Weekly Adjusted Time Series": {
                "2026-08-14": {
                    "1. open": "640.00",
                    "2. high": "645.00",
                    "3. low": "637.00",
                    "4. close": "643.00",
                    "5. adjusted close": "643.00",
                    "6. volume": "123456789",
                    "7. dividend amount": "0.00",
                }
            },
        }


def test_2026_08_14_alpha_row_is_not_operational_before_provider_first_seen() -> None:
    client = AlphaVantageClient(
        AlphaVantageConfig(api_key="test-key"),
        transport=_August14AlphaTransport(),
        retry=RetryPolicy(max_attempts=1),
        clock=lambda: FIRST_SEEN_2026_08_14,
    )

    result = client.fetch_weekly_adjusted(
        ("SPY",),
        # The operating forecast decision preceded the source's modeled final
        # weekly bar by 15 minutes and the provider receipt by almost six days.
        cutoff=datetime(2026, 8, 14, 20, 0, tzinfo=UTC),
        fields=("adjusted_close",),
    )

    assert result.health is HealthStatus.OK
    assert len(result.records) == 1
    record = result.records[0]
    assert record.observed_period_end == date(2026, 8, 14)
    assert record.source_released_at == SOURCE_FINAL_2026_08_14
    assert record.available_at == SOURCE_FINAL_2026_08_14
    assert record.provider_first_seen_at == FIRST_SEEN_2026_08_14
    assert record.system_retrieved_at == FIRST_SEEN_2026_08_14
    assert record.operating_available_at == FIRST_SEEN_2026_08_14
    assert record.metadata["operating_availability_policy"] == (
        "max_source_finalization_provider_first_seen"
    )

    source_replay = weekly_asof_join(
        (datetime(2026, 8, 15, 12, tzinfo=UTC),),
        result.records,
        required_series=(("alpha_vantage", "SPY.adjusted_close"),),
        availability_basis="source",
    )
    reconstructed_replay = weekly_asof_join(
        (datetime(2026, 8, 14, 20, 0, tzinfo=UTC),),
        result.records,
        required_series=(("alpha_vantage", "SPY.adjusted_close"),),
        availability_basis="reconstructed_market",
    )
    operational_replay = weekly_asof_join(
        (
            datetime(2026, 8, 14, 20, 0, tzinfo=UTC),
            datetime(2026, 8, 15, 12, tzinfo=UTC),
            FIRST_SEEN_2026_08_14,
        ),
        result.records,
        required_series=(("alpha_vantage", "SPY.adjusted_close"),),
        availability_basis="operational",
    )

    assert source_replay[0].value == 643.0
    assert reconstructed_replay[0].value == 643.0
    assert reconstructed_replay[0].available_at == datetime(
        2026, 8, 14, 20, 0, tzinfo=UTC
    )
    assert reconstructed_replay[0].source_released_at == SOURCE_FINAL_2026_08_14
    assert reconstructed_replay[0].provider_first_seen_at == FIRST_SEEN_2026_08_14
    assert [row.value for row in operational_replay] == [None, None, 643.0]
    assert operational_replay[-1].available_at == FIRST_SEEN_2026_08_14
    assert operational_replay[-1].provider_first_seen_at == FIRST_SEEN_2026_08_14


def test_reconstructed_market_basis_does_not_backdate_macro_releases() -> None:
    cutoff = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)
    macro_release = Observation(
        source="alfred",
        series_id="INDPRO",
        observed_period_end=date(2026, 7, 31),
        value=101.0,
        released_at=datetime(2026, 8, 14, 21, 0, tzinfo=UTC),
        available_at=datetime(2026, 8, 14, 21, 0, tzinfo=UTC),
        vintage_date=date(2026, 8, 14),
        retrieved_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
    )

    replay = weekly_asof_join(
        (cutoff,),
        (macro_release,),
        required_series=(("alfred", "INDPRO"),),
        availability_basis="reconstructed_market",
    )

    assert replay[0].value is None


def test_2026_08_14_unfavorable_replay_evidence_is_frozen() -> None:
    fixture = json.loads(PIT_REPLAY_FIXTURE.read_text(encoding="utf-8"))
    before = fixture["replays"]["pre_first_seen_generation"]
    after = fixture["replays"]["post_first_seen_reconstruction"]

    assert datetime.fromisoformat(fixture["forecast_cutoff_at"]) < datetime.fromisoformat(
        fixture["source_released_at"]
    ) < datetime.fromisoformat(fixture["provider_first_seen_at"])
    assert before["current_state"] == "transition"
    assert after["current_state"] == "risk_on"
    assert before["display_membership_semantics"] != after[
        "display_membership_semantics"
    ]
    assert after["forecast_risk_on_probability"] - before[
        "forecast_risk_on_probability"
    ] > 0.70
    assert fixture["interpretation"] == (
        "unfavorable_fixed_evidence_of_vintage_sensitive_pseudo_oos_not_metric_identity"
    )


def test_live_collection_does_not_override_alpha_source_finalization_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    captured_config: list[AlphaVantageConfig] = []
    real_config = AlphaVantageConfig
    real_client = AlphaVantageClient

    def capture_from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> AlphaVantageConfig:
        del cls, environ
        captured_kwargs.update(kwargs)
        configured = real_config(api_key="test-key", **kwargs)
        captured_config.append(configured)
        return configured

    monkeypatch.setattr(
        collection_module.AlphaVantageConfig,
        "from_env",
        classmethod(capture_from_env),
    )
    monkeypatch.setattr(
        collection_module,
        "AlphaVantageClient",
        lambda config, retry, budget=None: real_client(
            config,
            transport=_August14AlphaTransport(),
            retry=retry,
            budget=budget,
            clock=lambda: FIRST_SEEN_2026_08_14,
        ),
    )

    collection = collection_module.collect_live_data(
        {
            "alpha_vantage": {
                "base_url": "https://example.invalid/alpha",
                "daily_request_cap": 25,
                "symbols": ["SPY"],
                "fields": ["adjusted_close"],
            },
            "alfred": {
                "base_url": "https://example.invalid/fred",
                "series": [],
            },
        },
        database_path=tmp_path / "collection.sqlite3",
        history_start=date(2026, 8, 14),
        now=datetime(2026, 8, 15, 12, tzinfo=UTC),
    )

    assert "market_available_time_et" not in captured_kwargs
    assert captured_config[0].market_available_time_et == time(16, 15)
    alpha_source = next(
        item for item in collection.sources if item["id"] == "alpha_vantage"
    )
    assert alpha_source["coverage"] == "2026-08-14–2026-08-14"
    assert alpha_source["available_at"] == SOURCE_FINAL_2026_08_14.isoformat()


def _create_legacy_observation_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE observations (
                snapshot_id TEXT NOT NULL,
                source TEXT NOT NULL,
                series_id TEXT NOT NULL,
                observed_period_end TEXT NOT NULL,
                released_at TEXT,
                available_at TEXT NOT NULL,
                vintage_date TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                revision_seq INTEGER NOT NULL,
                value REAL,
                units TEXT NOT NULL,
                adjustment TEXT NOT NULL,
                license_class TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (
                    snapshot_id, source, series_id, observed_period_end,
                    vintage_date, revision_seq, retrieved_at
                )
            );
            """
        )
        connection.execute(
            """
            INSERT INTO observations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "legacy-snapshot",
                "alpha_vantage",
                "SPY.adjusted_close",
                "2026-08-14",
                SOURCE_FINAL_2026_08_14.isoformat(),
                SOURCE_FINAL_2026_08_14.isoformat(),
                "2026-08-14",
                FIRST_SEEN_2026_08_14.isoformat(),
                0,
                643.0,
                "USD",
                "weekly_adjusted",
                "private_research",
                "ok",
                "a" * 64,
                "{}",
            ),
        )


def test_read_only_legacy_database_synthesizes_explicit_clocks_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    _create_legacy_observation_database(path)
    before = path.read_bytes()

    with SQLiteSnapshotStore(path, read_only=True) as store:
        loaded = store.read_observations(snapshot_id="legacy-snapshot")

    assert path.read_bytes() == before
    assert len(loaded) == 1
    assert loaded[0].source_released_at == SOURCE_FINAL_2026_08_14
    assert loaded[0].provider_first_seen_at == FIRST_SEEN_2026_08_14
    assert loaded[0].system_retrieved_at == FIRST_SEEN_2026_08_14
    assert loaded[0].operating_available_at == FIRST_SEEN_2026_08_14


def test_writable_legacy_database_migrates_and_backfills_bitemporal_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-migrated.sqlite3"
    _create_legacy_observation_database(path)

    with SQLiteSnapshotStore(path) as store:
        loaded = store.read_observations(snapshot_id="legacy-snapshot")
        columns = {
            str(row[1])
            for row in store._connection.execute(
                "PRAGMA table_info(observations)"
            ).fetchall()
        }

    assert {
        "source_released_at",
        "provider_first_seen_at",
        "system_retrieved_at",
    }.issubset(columns)
    assert loaded[0].source_released_at == SOURCE_FINAL_2026_08_14
    assert loaded[0].provider_first_seen_at == FIRST_SEEN_2026_08_14
    assert loaded[0].system_retrieved_at == FIRST_SEEN_2026_08_14


def test_snapshot_roundtrip_preserves_explicit_bitemporal_identity(tmp_path: Path) -> None:
    path = tmp_path / "explicit.sqlite3"
    system_retrieved = datetime(2026, 8, 20, 2, 0, tzinfo=UTC)
    record = Observation(
        source="alpha_vantage",
        series_id="SPY.adjusted_close",
        observed_period_end=date(2026, 8, 14),
        value=643.0,
        released_at=SOURCE_FINAL_2026_08_14,
        source_released_at=SOURCE_FINAL_2026_08_14,
        available_at=SOURCE_FINAL_2026_08_14,
        provider_first_seen_at=FIRST_SEEN_2026_08_14,
        vintage_date=date(2026, 8, 14),
        retrieved_at=system_retrieved,
        system_retrieved_at=system_retrieved,
        revision_seq=3,
        raw_sha256="b" * 64,
    )
    provenance = SnapshotProvenance(
        source="alpha_vantage",
        dataset="weekly_adjusted_etf",
        cutoff=datetime(2026, 8, 21, 20, tzinfo=UTC),
        requested_at=system_retrieved,
        retrieved_at=system_retrieved,
        quality_status=HealthStatus.OK,
    )

    with SQLiteSnapshotStore(path) as store:
        snapshot_id = store.write_snapshot((record,), provenance)
        loaded = store.read_observations(snapshot_id=snapshot_id)

    assert loaded == (record,)
    assert loaded[0].observed_period_end == date(2026, 8, 14)
    assert loaded[0].revision_seq == 3
    assert loaded[0].raw_sha256 == "b" * 64
