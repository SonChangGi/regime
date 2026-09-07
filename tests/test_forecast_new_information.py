"""Discriminating parser, first-seen, revision and prequential research tests."""
from __future__ import annotations

from datetime import date
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from regime_lab.data.release_archive import ReleaseRecord
from regime_lab.research.forecast_new_information import (
    PROBABILITY_COLUMNS, STATE_ORDER, Snapshot, calendar_features, ebp_features,
    fetch_snapshot, load_boundary_history, market_features, parse_bls_calendar,
    parse_cboe, parse_cftc_tff, parse_fomc_calendar, positioning_features,
    probability_metrics, run_same_origin_ablation, utc, validate_history, build_forecast_information,
    parse_bls_html, parse_bls_alternate, collect_bls_html_fallback, parse_ebp,
)


def snap(raw, key="vix3m", at="2026-09-07T12:00:00Z"):
    raw = raw.encode() if isinstance(raw, str) else raw
    return Snapshot(key, "https://www.cftc.gov/test" if key.startswith("tff") else "https://cdn.cboe.com/test",
                    hashlib.sha256(raw).hexdigest(), utc(at), raw)


def quotes(key="vix3m", at="2026-09-07T12:00:00Z"):
    return snap("DATE,CLOSE\n08/06/2026,20\n08/07/2026,22\n09/03/2026,24\n09/04/2026,30\n", key, at)


@pytest.mark.parametrize("value", ["NaN", "inf", "-2", "0"])
def test_invalid_market_values_rejected(value):
    with pytest.raises(ValueError):
        parse_cboe(snap(f"DATE,CLOSE\n09/03/2026,{value}\n"))


def test_market_prior_day_is_separate_from_first_seen_and_no_same_day_close():
    sources = {k: quotes(k) for k in ("vix", "vix9d", "vvix", "vix3m")}
    sources["vix3m"] = snap("DATE,CLOSE\n08/06/2026,20\n08/07/2026,25\n09/03/2026,36\n09/04/2026,300\n")
    origin = pd.DatetimeIndex(["2026-09-04T20:00:00Z"])
    historical, lineage = market_features(sources, origin, track="reconstructed_market_prior_day")
    assert historical.iloc[0].vix3m_ratio == 1.5
    assert historical.iloc[0].vix3m_ratio_change_4w == .5
    assert lineage.iloc[0].observed_date == "2026-09-03"
    assert lineage.iloc[0].available_at < origin[0]
    first_seen, _ = market_features(sources, origin)
    assert first_seen.isna().all().all()
    runtime, _ = market_features(sources, pd.DatetimeIndex(["2026-09-07T13:00:00Z"]))
    assert runtime.iloc[0].vix3m_ratio == 10
    assert np.isfinite(runtime.iloc[0].vix3m_ratio_change_4w)


def test_market_rejects_duplicate_future_and_unrecognized_columns():
    for raw in ("DATE,CLOSE\n09/03/2026,20\n09/03/2026,20\n",
                "DATE,CLOSE\n09/09/2026,20\n", "DATE,OPEN\n09/03/2026,20\n"):
        with pytest.raises(ValueError):
            parse_cboe(snap(raw))
    assert parse_cboe(snap("DATE,VVIX\n09/03/2026,80\n", "vvix")).value.iloc[0] == 80


def test_market_rejects_stale_quotes():
    sources = {k: snap("DATE,CLOSE\n08/01/2026,20\n", k) for k in ("vix", "vix9d", "vvix", "vix3m")}
    features, _ = market_features(sources, pd.DatetimeIndex(["2026-09-04T20:00:00Z"]), track="reconstructed_market_prior_day")
    assert features.isna().all().all()


def ics(day="20260910", *, timestamp="20250101T000000Z", cancelled=False):
    cancellation = "STATUS:CANCELLED\n" if cancelled else ""
    return ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:cpi-1\nSUMMARY:Consumer Price \n Index\n"
            f"DTSTAMP:{timestamp}\nDTSTART;TZID=America/New_York:{day}T083000\n{cancellation}END:VEVENT\n"
            "BEGIN:VEVENT\nUID:cpi-2\nSUMMARY:Consumer Price Index\nDTSTART:20261014T123000Z\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:jobs-1\nSUMMARY:Employment Situation\nDTSTART;TZID=US/Eastern:20261002T083000\nEND:VEVENT\nEND:VCALENDAR\n")


def test_calendar_dtstamp_cannot_manufacture_historical_knowledge():
    frame = parse_bls_calendar(snap(ics(), "bls"))
    assert frame.iloc[0].event_at == utc("2026-09-10T12:30:00Z")
    assert (frame.known_at == utc("2026-09-07T12:00:00Z")).all()
    features = calendar_features([frame], pd.DatetimeIndex(["2026-09-04T20:00:00Z", "2026-09-07T13:00:00Z"]))
    assert features.iloc[0].isna().all()
    assert features.iloc[1].cpi_known_events_7d == 1
    assert features.iloc[1].jobs_known_events_7d == 0


def test_calendar_revision_and_cancellation_replace_schedule_without_retroactivity():
    first = parse_bls_calendar(snap(ics(), "bls", "2026-09-07T12:00:00Z"))
    revised = parse_bls_calendar(snap(ics("20260918"), "bls", "2026-09-08T12:00:00Z"))
    cancelled = parse_bls_calendar(snap(ics(cancelled=True), "bls", "2026-09-09T12:00:00Z"))
    features = calendar_features([first, revised, cancelled], pd.DatetimeIndex([
        "2026-09-07T13:00:00Z", "2026-09-08T13:00:00Z", "2026-09-09T13:00:00Z"]))
    assert features.cpi_known_events_7d.tolist() == [1, 0, 0]
    assert features.iloc[1].cpi_days_to_next_known > 9
    assert features.iloc[2].cpi_days_to_next_known > 30


def test_calendar_unknown_and_stale_are_not_zero_events():
    frame = parse_bls_calendar(snap(ics(), "bls"))
    assert calendar_features([frame], pd.DatetimeIndex(["2026-10-20T13:00:00Z"])).empty
    with pytest.raises(ValueError, match="timezone"):
        parse_bls_calendar(snap(ics().replace("DTSTART;TZID=America/New_York:", "DTSTART:"), "bls"))
    with pytest.raises(ValueError, match="recurring"):
        parse_bls_calendar(snap(ics().replace("UID:cpi-1", "UID:cpi-1\nRRULE:FREQ=MONTHLY"), "bls"))


def test_fomc_cross_month_and_tentative_dates_keep_first_seen():
    html = ('<h4>2026 FOMC Meetings</h4>'
            '<div class="fomc-meeting__month"><strong>Jan/Feb</strong></div>'
            '<div class="fomc-meeting__date">31-1*</div>'
            '<div class="fomc-meeting__month">September</div>'
            '<div class="fomc-meeting__date">15-16*</div>')
    frame = parse_fomc_calendar(snap(html, "fomc"))
    assert frame.iloc[0].event_at.tz_convert("America/New_York").date() == date(2026, 2, 1)
    assert frame.time_precision.eq("date").all()
    assert frame.known_at.min() == utc("2026-09-07T12:00:00Z")


def tff_csv(*, observed="2026-09-01", code="13874A", oi=1000, combined="FutOnly"):
    return ("Market_and_Exchange_Names,Report_Date_as_YYYY-MM-DD,CFTC_Contract_Market_Code,Open_Interest_All,"
            "Asset_Mgr_Positions_Long_All,Asset_Mgr_Positions_Short_All,Lev_Money_Positions_Long_All,"
            "Lev_Money_Positions_Short_All,FutOnly_or_Combined\n"
            f'E-MINI S&P 500,{observed},{code},{oi},600,300,100,300,{combined}\n')


def test_tff_zip_parser_and_delayed_publication_use_actual_first_seen():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../FinFutYY.txt", tff_csv())
    frame = parse_cftc_tff(snap(buffer.getvalue(), "tff_2026"), contract_codes=("13874A",))
    assert frame.iloc[0].asset_net_oi == .3
    assert frame.iloc[0].leveraged_net_oi == -.2
    assert frame.source_released_at.isna().all()
    features = positioning_features(frame, pd.DatetimeIndex(["2026-09-04T20:00:00Z", "2026-09-07T13:00:00Z"]))
    assert features.iloc[0].isna().all()  # nominal Friday release is not asserted
    assert features.iloc[1].tff_13874A_asset_net_oi == .3


@pytest.mark.parametrize("raw", [tff_csv(oi=0), tff_csv(oi=500), tff_csv(combined="Combined"),
                                 tff_csv(observed="2026-09-08"), tff_csv(code="999999")])
def test_tff_invalid_rows_rejected(raw):
    with pytest.raises(ValueError):
        parse_cftc_tff(snap(raw, "tff_2026"), contract_codes=("13874A",))


def test_tff_missing_exact_four_week_observation_stays_missing_and_stale_expires():
    frame = parse_cftc_tff(snap(tff_csv(), "tff_2026"), contract_codes=("13874A",))
    output = positioning_features(frame, pd.DatetimeIndex(["2026-09-07T13:00:00Z", "2026-10-01T13:00:00Z"]))
    assert np.isnan(output.iloc[0].tff_13874A_asset_change_4w)
    assert output.iloc[1].isna().all()


def test_ebp_uses_system_receipt_even_if_provider_timestamp_backdated():
    record = ReleaseRecord(source_id="board_ebp", series_id="ebp", observed_period_end=date(2026, 8, 31),
        value=.2, source_released_at=utc("2026-09-01T12:00:00Z"),
        provider_first_seen_at=utc("2026-09-01T12:00:00Z"),
        system_retrieved_at=utc("2026-09-07T12:00:00Z"), revision_seq=0, raw_sha256="a" * 64)
    output = ebp_features([record], pd.DatetimeIndex(["2026-09-04T20:00:00Z", "2026-09-07T13:00:00Z"]))
    assert output.iloc[0].isna().all()
    assert output.iloc[1].ebp == .2


def history_fixture(n=70):
    origin = pd.date_range("2022-01-07T21:00:00Z", periods=n, freq="7D")
    actual = [STATE_ORDER[(i + 1) % 3] for i in range(n)]
    frame = pd.DataFrame({"origin_date": origin, "target_date": origin + pd.Timedelta(7, unit="D"),
                          "current_state": [STATE_ORDER[i % 3] for i in range(n)], "actual": actual})
    cutoff = utc("2023-01-01T00:00:00Z")
    frame["evaluation_split"] = np.where(frame.target_date < cutoff, "selection",
                                        np.where(frame.origin_date >= cutoff, "holdout", "boundary_excluded"))
    frame[PROBABILITY_COLUMNS] = [.5, .3, .2]
    features = pd.DataFrame({"old": np.sin(np.arange(n)), "new": np.cos(np.arange(n))}, index=origin)
    return frame, features


def test_prequential_ablation_pairs_training_and_origins_and_has_direction_scores():
    frame, features = history_fixture()
    features.iloc[10, 1] = np.nan
    predictions, summary = run_same_origin_ablation(frame, features, control_columns=["old"], extra_columns=["new"], minimum_training_rows=12)
    assert summary["all_train_targets_strictly_before_origin"]
    assert summary["excluded"]["missing_features"] == 1
    assert predictions.groupby("origin_date").model.nunique().eq(3).all()
    assert predictions.groupby("origin_date").train_rows.nunique().eq(1).all()
    for split in summary["metrics"].values():
        assert split["paired_delta"]["same_origins"]
        assert split["candidate"]["worsening"]["average_precision"] is not None


def test_future_outcome_and_feature_changes_cannot_change_earlier_predictions():
    frame, features = history_fixture()
    first, _ = run_same_origin_ablation(frame, features, control_columns=["old"], extra_columns=["new"], minimum_training_rows=12)
    changed, feature_changed = frame.copy(), features.copy()
    changed.loc[55:, "actual"] = "risk_off"
    feature_changed.iloc[55:] = 10000
    second, _ = run_same_origin_ablation(changed, feature_changed, control_columns=["old"], extra_columns=["new"], minimum_training_rows=12)
    cutoff = frame.iloc[55].origin_date
    pd.testing.assert_frame_equal(first.loc[first.origin_date < cutoff].reset_index(drop=True),
                                  second.loc[second.origin_date < cutoff].reset_index(drop=True))


def test_target_on_origin_excluded_and_split_tampering_rejected():
    frame, features = history_fixture()
    predictions, _ = run_same_origin_ablation(frame, features, control_columns=["old"], extra_columns=["new"], minimum_training_rows=12)
    first = predictions.iloc[0]
    assert first.last_train_target == first.origin_date - pd.Timedelta(7, unit="D")
    frame["evaluation_split"] = "selection"
    with pytest.raises(ValueError, match="cutoff"):
        validate_history(frame)


def test_direction_score_excludes_structural_zero_origins():
    frame, _ = history_fixture(3)
    p = frame[PROBABILITY_COLUMNS].to_numpy()
    score = probability_metrics(frame, p)
    assert score["worsening"]["at_risk_origins"] == 2
    assert score["recovery"]["at_risk_origins"] == 2
    assert score["transition_events"] == 3


def test_checksum_and_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="checksum"):
        Snapshot("vix3m", "https://cdn.cboe.com/test", "a" * 64, utc("2026-09-07T12:00:00Z"), b"bad")
    with pytest.raises(ValueError, match="timezone"):
        utc("2026-09-07")


def test_download_is_cached_and_refresh_does_not_erase_old_calendar(tmp_path):
    class Response:
        url = "https://www.bls.gov/schedule/news_release/bls.ics"
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def raise_for_status(self): pass
        def iter_content(self, _): yield ics().encode()
    class Session:
        calls = 0
        def get(self, *_, **__):
            self.calls += 1
            return Response()
    session = Session()
    first = fetch_snapshot("bls", Response.url, tmp_path, session=session)
    second = fetch_snapshot("bls", Response.url, tmp_path, session=session)
    assert session.calls == 1 and first.retrieved_at == second.retrieved_at
    fetch_snapshot("bls", Response.url, tmp_path, session=session, refresh=True)
    assert len(list(tmp_path.glob("bls-*.json"))) == 2
    assert len(list(tmp_path.glob("*.blob"))) == 1
    with pytest.raises(ValueError, match="official"):
        fetch_snapshot("bad", "https://example.com/paid", tmp_path, session=session)


def test_ui_adapter_has_matched_boundary_and_matched_fit_control_and_rejects_mismatch():
    frame, features = history_fixture(30)
    _, experiment = run_same_origin_ablation(frame, features, control_columns=["old"], extra_columns=["new"], minimum_training_rows=12)
    summary = {"data_as_of": "2026-09-04T20:00:00Z", "experiments": {"vix3m": experiment}}
    adapter = build_forecast_information(summary)
    assert adapter["schema_version"] == "regime-forecast-information/1"
    assert {r["baseline_model"] for r in adapter["rows"]} == {"boundary_filtered_history", "boundary_existing_vol_control"}
    assert len({r["matched_n"] for r in adapter["rows"]}) == 1
    experiment["metrics"]["selection"]["paired_delta"]["same_origins"] = False
    with pytest.raises(ValueError, match="matched"):
        build_forecast_information(summary)


def test_fomc_incomplete_markup_fails_whole_snapshot():
    with pytest.raises(ValueError, match="pairing"):
        parse_fomc_calendar(snap('<h4>2026 FOMC Meetings</h4><div class="fomc-meeting__month">September</div>', "fomc"))


def bls_html(kind="cpi"):
    name = "Consumer Price Index" if kind == "cpi" else "Employment Situation"
    return (f'<html><h1>Schedule of Releases for the {name}</h1><table>'
            '<tr><th>Reference Month</th><th>Release Date</th><th>Release Time</th></tr>'
            '<tr><td>August 2026</td><td>Sep. 11, 2026</td><td>08:30 AM</td></tr>'
            '<tr><td>October 2026</td><td>November 10, 2026</td><td>08:30 AM</td></tr>'
            '</table>Subscribe to the BLS Online Calendar</html>')


def test_bls_html_release_dates_not_reference_months_and_dst():
    frame = parse_bls_html(snap(bls_html(), "bls"), "cpi")
    assert frame.event_at.tolist() == [utc("2026-09-11T12:30:00Z"), utc("2026-11-10T13:30:00Z")]
    assert (frame.known_at == utc("2026-09-07T12:00:00Z")).all()
    with pytest.raises(ValueError, match="heading"):
        parse_bls_html(snap("<html>Access denied</html>", "bls"), "cpi")


def test_bls_html_fallback_fetches_both_official_release_pages(monkeypatch, tmp_path):
    from regime_lab.research import forecast_new_information as module
    calls = []
    def fetch(key, url, directory, **kwargs):
        calls.append(url)
        raw = bls_html("cpi" if "cpi.htm" in url else "jobs").encode()
        return Snapshot(key, url, hashlib.sha256(raw).hexdigest(), utc("2026-09-07T12:00:00Z"), raw)
    monkeypatch.setattr(module, "fetch_snapshot", fetch)
    snapshot = collect_bls_html_fallback(tmp_path)
    frame = parse_bls_alternate(snapshot)
    assert set(frame.event_kind) == {"cpi", "jobs"}
    assert len(calls) == 2 and all(url.startswith("https://www.bls.gov/schedule/news_release/") for url in calls)


def test_saved_official_web_extraction_is_real_current_knowledge_not_historical():
    raw = ('https://www.bls.gov/schedule/news_release/cpi.htm\n'
           'L211: Schedule of Releases for the Consumer Price Index\n'
           'L224: August 2026 Sep. 11, 2026 08:30 AM\n'
           'L225: September 2026 Oct. 14, 2026 08:30 AM\nSubscribe to\n'
           'https://www.bls.gov/schedule/news_release/empsit.htm\n'
           'L211: Schedule of Releases for the Employment Situation\n'
           'L224: August 2026 Sep. 04, 2026 08:30 AM\n'
           'L225: September 2026 Oct. 02, 2026 08:30 AM\nSubscribe to\n')
    frame = parse_bls_alternate(snap(raw, "bls_web_extract"))
    features = calendar_features([frame], pd.DatetimeIndex(["2026-09-04T20:00:00Z", "2026-09-07T13:00:00Z"]))
    assert features.iloc[0].isna().all()
    assert features.iloc[1].cpi_known_events_7d == 1
    assert features.iloc[1].jobs_known_events_7d == 0


def test_ebp_parser_maps_month_to_end_and_preserves_revision_uncertainty():
    records = parse_ebp(snap("date,gz_spread,ebp,est_prob\n7/1/2026,0.84,-0.319,0.10\n", "board_ebp"))
    assert records[0].observed_period_end == date(2026, 7, 31)
    assert records[0].value == -.319
    assert records[0].metadata["actual_source_released_at"] is None
    assert records[0].metadata["timestamp_semantics"] == "first_seen_only"
    features = ebp_features(records, pd.DatetimeIndex(["2026-09-04T20:00:00Z", "2026-09-07T13:00:00Z"]))
    assert features.iloc[0].isna().all()
    assert features.iloc[1].ebp == -.319


@pytest.mark.parametrize("raw", ["date,ebp\n9/1/2026,0.2\n", "date,ebp\n7/1/2026,inf\n",
                                 "date,ebp\n7/1/2026,0.2\n7/1/2026,0.3\n"])
def test_ebp_invalid_or_unfinished_month_rejected(raw):
    with pytest.raises(ValueError):
        parse_ebp(snap(raw, "board_ebp"))
