import json
import numpy as np
import pandas as pd
import pytest
from regime_lab.research.additional_sources import cboe_weekly, read_source
from regime_lab.research.downside import run_downside_research
from regime_lab.research.contract import validate_research_extensions


def prices(n=110):
    rng = np.random.default_rng(8)
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI", tz="UTC")
    return pd.DataFrame(
        {
            "spy_close": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.035, n))),
            "spy_adjusted_open": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.035, n))),
            "tlt_close": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))),
        },
        index=idx,
    )


def test_downside_is_unchanged_by_future_append_and_labels_are_mature():
    frame = prices()
    early = run_downside_research(
        frame.iloc[:90], minimum_train_weeks=30, refit_weeks=13
    )
    full = run_downside_research(frame, minimum_train_weeks=30, refit_weeks=13)
    columns = [
        "origin_date",
        "model",
        "horizon_weeks",
        "q10",
        "q25",
        "loss_probability",
    ]
    old = early.predictions[columns].sort_values(columns[:3]).reset_index(drop=True)
    same = (
        full.predictions[
            full.predictions.origin_date.isin(early.predictions.origin_date)
        ][columns]
        .sort_values(columns[:3])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(old, same)
    for row in full.predictions.itertuples():
        train = frame.index.get_loc(pd.Timestamp(row.train_end_origin))
        origin = frame.index.get_loc(pd.Timestamp(row.origin_date))
        assert train + row.horizon_weeks + 1 <= origin
        assert row.q10 <= row.q25
    assert all(r["weeks"] > 0 for r in full.summary["comparisons"])
    assert full.summary["as_of"] == frame.index[-1].isoformat()
    assert all(
        r["origin_date"] == full.summary["as_of"] for r in full.summary["latest"]
    )


def test_cboe_snapshot_is_immutable_and_same_day_close_not_consumed(tmp_path):
    raw = b"DATE,CLOSE\n08/27/2026,17\n08/28/2026,99\n"
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "snapshot.csv").write_bytes(raw)
    meta = {
        "sources": {
            key: {"file": "snapshot.csv", "sha256": digest}
            for key in ["vix", "vix9d", "vvix"]
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    idx = pd.DatetimeIndex(["2026-08-28T20:00:00Z"])
    weekly = cboe_weekly(tmp_path, idx)
    assert weekly.iloc[0].vix == 17
    (tmp_path / "snapshot.csv").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        read_source(tmp_path, "vix")


def test_research_contract_rejects_bad_probability_and_infinite_values():
    r = {
        "extensions": {
            "schema_version": "regime-research-extensions/1",
            "downside": {
                "latest": [
                    {"horizon_weeks": 4, "loss_probability": 1.1, "q10": -0.1, "q25": 0}
                ]
            },
        }
    }
    with pytest.raises(ValueError, match="prediction"):
        validate_research_extensions(r)
    with pytest.raises(ValueError):
        validate_research_extensions({"x": float("nan")})


def test_downside_selection_never_crosses_the_holdout_boundary():
    frame = prices(145)
    cutoff = frame.index[95].date().isoformat()
    result = run_downside_research(frame, minimum_train_weeks=30, selection_end=cutoff)
    selected = result.predictions[result.predictions.split.eq("selection")]
    assert (
        pd.to_datetime(selected.target_exit, utc=True) < pd.Timestamp(cutoff, tz="UTC")
    ).all()
    assert result.predictions.split.eq("boundary_purged").any()
    baseline = result.predictions[result.predictions.model.eq("volatility_baseline")]
    assert baseline.last_fit_origin.equals(baseline.origin_date)


def test_downside_empty_history_is_a_normal_result():
    result = run_downside_research(prices(22))
    assert result.summary["status"] == "insufficient_history"
    assert result.summary["latest"] == []


def test_missing_optional_source_does_not_erase_baselines():
    frame = prices(100)
    missing = pd.DataFrame({"vvix": np.nan}, index=frame.index)
    result = run_downside_research(frame, additional=missing, minimum_train_weeks=30)
    assert result.summary["comparisons"]
    assert "linear_quantile_cboe" not in result.predictions.model.unique()
