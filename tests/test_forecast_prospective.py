"""Prospective chronology, path semantics, and immutable isolated evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sqlite3

import pytest

from regime_lab.research import forecast_prospective as prospective


UTC = timezone.utc
LABEL = {"version": "official-label-fixture-v1", "sha256": "a" * 64}
ORIGIN = datetime(2027, 1, 8, 21, tzinfo=UTC)


def protocol(origin_count=8, horizons=(1, 4, 13)):
    return {"schema_version": prospective.PROTOCOL_SCHEMA, "version": "synthetic-forward-v1", "recipe_sha256": "b" * 64,
            "label": LABEL.copy(), "calendar": [(ORIGIN + timedelta(weeks=i)).isoformat() for i in range(origin_count + 13)],
            "issue_deadline_seconds": 48 * 3600, "max_generation_age_seconds": 3600,
            "models": [{"id": "candidate", "version": "v1", "recipe_sha256": "c" * 64,
                        "role": "candidate", "baseline_id": "baseline", "horizons": list(horizons)},
                       {"id": "baseline", "version": "v1", "recipe_sha256": "d" * 64,
                        "role": "baseline", "horizons": list(horizons)}],
            "criteria": {"minimum_paired_origins": 4, "minimum_worsening_events": 1, "minimum_recovery_events": 1,
                         "minimum_risk_off_episodes": 1, "minimum_binary_events": 1, "minimum_binary_nonevents": 1,
                         "block_length": 2, "bootstrap_draws": 199, "maximum_missing_origin_fraction": 0,
                         "maximum_log_loss_delta_ci_high": 0, "maximum_brier_delta": 0,
                         "minimum_worsening_recall": .25, "maximum_false_alarm_fraction": .1}}


def paths(current):
    order = prospective.STATE_ORDER
    if current == "risk_on":
        rows = [(1, [.6, .3, .1], [.6, 0, .3, .1], .1, .1),
                (4, [.4, .3, .3], [.3, 0, .5, .2], .4, .4),
                (13, [.25, .35, .4], [.1, 0, .55, .35], .7, .7)]
    elif current == "risk_off":
        rows = [(1, [.1, .2, .7], [.7, .1, .2, 0], 0, .7),
                (4, [.3, .3, .4], [.2, .3, .5, 0], .3, .9),
                (13, [.3, .3, .4], [.05, .4, .55, 0], .5, .99)]
    else:
        rows = [(1, [.2, .6, .2], [.6, .2, 0, .2], .2, .2),
                (4, [.3, .4, .3], [.3, .35, 0, .35], .45, .45),
                (13, [.3, .4, .3], [.1, .4, 0, .5], .7, .7)]
    return [{"horizon_weeks": h, "endpoint": dict(zip(order, endpoint)),
             "first_departure": dict(zip(prospective.FIRST_DESTINATIONS, departure)),
             "any_risk_off_entry": entry, "any_risk_off_occupancy": occupancy}
            for h, endpoint, departure, entry, occupancy in rows]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    clock = {"now": ORIGIN - timedelta(days=1)}
    monkeypatch.setattr(prospective, "_utc_now", lambda: clock["now"])
    p = protocol()
    ledger = prospective.ForecastResearchLedger(tmp_path / "research.sqlite", create=True)
    registered = ledger.register_protocol(p)
    return clock, p, ledger, registered["protocol_sha256"], tmp_path


def freeze(fixture, *, index=0, current="risk_on", evidence_track="prospective_inputs", one_week=None):
    clock, p, _, _, _ = fixture
    origin = datetime.fromisoformat(p["calendar"][index])
    clock["now"] = origin + timedelta(minutes=5)
    block = {"schema_version": "regime-forecast-audit-research/1", "evidence_track": evidence_track,
             "data_as_of": origin.isoformat(), "generated_at": clock["now"].isoformat(), "latest": []}
    for model in p["models"]:
        row = {"model": model["id"], "origin_date": origin.isoformat(), "current_state": current,
               "version": model["version"], "recipe_sha256": model["recipe_sha256"],
               "training": {"last_train_target": (origin - timedelta(weeks=1)).isoformat()},
               "paths": [path for path in paths(current) if path["horizon_weeks"] in model["horizons"]]}
        if one_week is not None:
            row.pop("paths")
            row["next_state"] = one_week[model["id"]]
        block["latest"].append(row)
    information = {"manifest_sha256": "e" * 64, "sources": [
        {"id": "synthetic-official-states", "sha256": "f" * 64,
         "available_at": (clock["now"] - timedelta(seconds=1)).isoformat(),
         "retrieved_at": (clock["now"] - timedelta(seconds=1)).isoformat(), "availability_basis": "first_seen"}]}
    result = prospective.freeze_research_forecast(block, protocol=p, information_set=information)
    return result, block, information


def snapshot(p, states, clock):
    return {"schema_version": prospective.STATES_SCHEMA, "label": deepcopy(p["label"]),
            "available_at": clock["now"].isoformat(),
            "observations": [{"date": date, "state": state} for date, state in zip(p["calendar"], states)]}


def table_count(ledger, table):
    with sqlite3.connect(ledger.path) as conn:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_pure_preview_freezes_exact_probabilities_and_does_not_issue(fixture):
    clock, p, ledger, _, _ = fixture
    frozen, block, info = freeze(fixture)
    assert frozen["preview_only"] is True
    assert frozen["eligible_for_issue_at_preview"] is True
    assert frozen["forecast"]["models"][0]["paths"][0]["endpoint"] == block["latest"][0]["paths"][0]["endpoint"]
    assert frozen["forecast_sha256"] == prospective.content_sha256(frozen["forecast"])
    info["sources"][0]["sha256"] = "0" * 64
    block["latest"][0]["paths"][0]["endpoint"]["risk_on"] = 0
    assert frozen["forecast"]["information_set"]["sources"][0]["sha256"] == "f" * 64
    assert table_count(ledger, "research_forecasts") == 0
    assert frozen["forecast"]["models"][0]["paths"][2]["target_date"] == p["calendar"][13]


@pytest.mark.parametrize("mutation", ["label_hash", "duplicate_model", "baseline_missing", "horizon_missing", "gap", "naive_date", "duplicate_date", "late_deadline", "negative_events", "no_event_requirement", "too_few_blocks", "nonfinite", "future_allowed_loss"])
def test_protocol_rejects_incomplete_or_incoherent_freeze(mutation):
    p = protocol()
    if mutation == "label_hash": p["label"]["sha256"] = "bad"
    elif mutation == "duplicate_model": p["models"].append(deepcopy(p["models"][0]))
    elif mutation == "baseline_missing": p["models"].pop()
    elif mutation == "horizon_missing": p["models"][1]["horizons"] = [1]
    elif mutation == "gap": p["calendar"].pop(2)
    elif mutation == "naive_date": p["calendar"][0] = "2027-01-08"
    elif mutation == "duplicate_date": p["calendar"][1] = p["calendar"][0]
    elif mutation == "late_deadline": p["issue_deadline_seconds"] = 7 * 86400
    elif mutation == "negative_events": p["criteria"]["minimum_worsening_events"] = -1
    elif mutation == "no_event_requirement": p["criteria"]["minimum_risk_off_episodes"] = 0
    elif mutation == "too_few_blocks": p["criteria"]["block_length"] = 13
    elif mutation == "nonfinite": p["criteria"]["maximum_brier_delta"] = float("nan")
    else: p["criteria"]["maximum_log_loss_delta_ci_high"] = .1
    with pytest.raises(prospective.ResearchLedgerError):
        prospective.validate_protocol(p)


def test_protocol_cannot_be_registered_after_first_origin_or_rewritten(fixture):
    clock, p, ledger, _, _ = fixture
    assert ledger.register_protocol(p)["created"] is False
    p["criteria"]["minimum_worsening_events"] = 2
    with pytest.raises(prospective.ResearchConflictError, match="already frozen"):
        ledger.register_protocol(p)
    p["version"] = "another-protocol"
    clock["now"] = ORIGIN
    with pytest.raises(prospective.ResearchLedgerError, match="before its first"):
        ledger.register_protocol(p)


@pytest.mark.parametrize("mutation", ["nan", "negative", "sum", "missing_class", "missing_model", "duplicate_model", "different_origin", "different_state", "recipe", "future_training", "endpoint_alias", "first_departure_alias", "origin_destination", "horizon_monotonicity", "entry_above_occupancy", "wrong_target", "future_source", "backdated_first_seen", "future_generation"])
def test_preview_rejects_invalid_probabilities_and_model_contracts(fixture, mutation):
    clock, p, _, _, _ = fixture
    _, block, info = freeze(fixture)
    model = block["latest"][0]
    if mutation == "nan": model["paths"][0]["endpoint"]["risk_on"] = float("nan")
    elif mutation == "negative": model["paths"][0]["endpoint"]["risk_on"] = -.1
    elif mutation == "sum": model["paths"][0]["endpoint"]["risk_on"] = .5
    elif mutation == "missing_class": del model["paths"][0]["endpoint"]["risk_on"]
    elif mutation == "missing_model": block["latest"].pop()
    elif mutation == "duplicate_model": block["latest"].append(deepcopy(model))
    elif mutation == "different_origin": model["origin_date"] = p["calendar"][1]
    elif mutation == "different_state": model["current_state"] = "risk_off"
    elif mutation == "recipe": model["recipe_sha256"] = "0" * 64
    elif mutation == "future_training": model["training"]["last_train_target"] = model["origin_date"]
    elif mutation == "endpoint_alias": model["paths"][0]["endpoint_probabilities"] = {"risk_on": 0, "transition": 1, "risk_off": 0}
    elif mutation == "first_departure_alias": model["paths"][0].update(first_destination={"risk_on": 0, "transition": .2, "risk_off": .2}, no_departure=.6)
    elif mutation == "origin_destination": model["paths"][0]["first_departure"].update(risk_on=.1, transition=.2)
    elif mutation == "horizon_monotonicity": model["paths"][2]["first_departure"].update(no_departure=.2, transition=.45)
    elif mutation == "entry_above_occupancy": model["paths"][2]["any_risk_off_entry"] = .8
    elif mutation == "wrong_target": model["paths"][0]["target_date"] = p["calendar"][2]
    elif mutation == "future_source": info["sources"][0]["retrieved_at"] = p["calendar"][1]
    elif mutation == "backdated_first_seen": info["sources"][0]["available_at"] = p["calendar"][0]
    else: block["generated_at"] = (clock["now"] + timedelta(seconds=1)).isoformat()
    with pytest.raises(prospective.ResearchLedgerError):
        prospective.freeze_research_forecast(block, protocol=p, information_set=info)


def test_native_and_split_path_aliases_freeze_identical_distributions(fixture):
    _, p, _, _, _ = fixture
    original, block, info = freeze(fixture)
    for model in block["latest"]:
        for path in model["paths"]:
            path["endpoint_probabilities"] = path.pop("endpoint")
            departure = path.pop("first_departure")
            path["no_departure"] = departure.pop("no_departure")
            path["first_destination"] = departure
    frozen = prospective.freeze_research_forecast(block, protocol=p, information_set=info)
    assert frozen["forecast"]["models"] == original["forecast"]["models"]


@pytest.mark.parametrize("reason", ["reconstructed_market", "replay", "reconstructed_source", "missing_attestation"])
def test_research_preview_is_available_but_not_prospective_issuance(fixture, reason):
    _, p, ledger, _, tmp_path = fixture
    _, block, info = freeze(fixture)
    if reason == "reconstructed_market": block["evidence_track"] = reason
    elif reason == "replay": block["replay"] = True
    elif reason == "reconstructed_source": info["sources"][0]["availability_basis"] = "reconstructed"
    else: del block["latest"][0]["recipe_sha256"]
    frozen = prospective.freeze_research_forecast(block, protocol=p, information_set=info)
    assert frozen["eligible_for_issue_at_preview"] is False
    with pytest.raises(prospective.ResearchLedgerError, match="cannot be issued"):
        ledger.issue(frozen, publication_path=tmp_path / "never-published.json")
    assert table_count(ledger, "research_forecasts") == 0
    assert not (tmp_path / "never-published.json").exists()


def test_storage_and_publication_use_actual_clock_and_are_idempotent(fixture):
    clock, p, ledger, digest, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    clock["now"] += timedelta(seconds=5)
    receipt = ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    assert receipt["stored_at"] == clock["now"].isoformat()
    assert receipt["published_at"] == clock["now"].isoformat()
    assert receipt["publication_sha256"] == hashlib.sha256((tmp_path / "issued.json").read_bytes()).hexdigest()
    assert json.loads((tmp_path / "issued.json").read_text())["forecast"] == frozen["forecast"]
    clock["now"] += timedelta(weeks=3)
    again = ledger.issue(frozen, publication_path=tmp_path / "unused.json")
    assert again["created"] is False
    assert again["published_at"] == receipt["published_at"]
    assert not (tmp_path / "unused.json").exists()
    assert table_count(ledger, "research_forecasts") == table_count(ledger, "research_publications") == 1
    ledger.verify_integrity()


@pytest.mark.parametrize("mutation", ["expired", "stale", "backdated_storage", "hash", "changed_distribution"])
def test_issue_rechecks_clock_and_frozen_content_at_actual_storage(fixture, mutation):
    clock, _, ledger, _, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    if mutation == "expired": clock["now"] += timedelta(weeks=1)
    elif mutation == "stale": clock["now"] += timedelta(hours=2)
    elif mutation == "backdated_storage": clock["now"] -= timedelta(seconds=1)
    elif mutation == "hash": frozen["forecast_sha256"] = "0" * 64
    else:
        frozen["forecast"]["models"][0]["paths"][0]["endpoint"]["risk_on"] = .8
        frozen["forecast_sha256"] = prospective.content_sha256(frozen["forecast"])
    with pytest.raises(prospective.ResearchLedgerError):
        ledger.issue(frozen, publication_path=tmp_path / "rejected.json")
    assert table_count(ledger, "research_forecasts") == 0


def test_late_publication_is_rejected_after_file_write(fixture, monkeypatch):
    clock, p, ledger, _, tmp_path = fixture
    _, block, info = freeze(fixture)
    deadline = ORIGIN + timedelta(seconds=p["issue_deadline_seconds"])
    block["generated_at"] = (deadline - timedelta(seconds=10)).isoformat()
    clock["now"] = deadline - timedelta(seconds=5)
    frozen = prospective.freeze_research_forecast(block, protocol=p, information_set=info)
    ticks = iter([deadline - timedelta(seconds=2), deadline - timedelta(seconds=1), deadline])
    monkeypatch.setattr(prospective, "_utc_now", lambda: next(ticks))
    with pytest.raises(prospective.ResearchLedgerError, match="expired"):
        ledger.issue(frozen, publication_path=tmp_path / "too-late.json")
    assert not (tmp_path / "too-late.json").exists()
    assert table_count(ledger, "research_forecasts") == 1
    assert table_count(ledger, "research_publications") == 0


def test_failed_publication_leaves_unpublished_record_and_can_retry(fixture):
    _, _, ledger, digest, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    destination = tmp_path / "existing.json"
    destination.write_text("keep me")
    with pytest.raises(FileExistsError):
        ledger.issue(frozen, publication_path=destination)
    assert destination.read_text() == "keep me"
    assert ledger.readiness(digest)["unpublished_origins"] == [ORIGIN.isoformat()]
    assert ledger.readiness(digest)["published_origins"] == 0
    receipt = ledger.issue(frozen, publication_path=tmp_path / "new.json")
    assert receipt["created"] is True
    changed = deepcopy(frozen)
    changed["forecast"]["source_block_sha256"] = "0" * 64
    changed["forecast_sha256"] = prospective.content_sha256(changed["forecast"])
    with pytest.raises(prospective.ResearchConflictError):
        ledger.issue(changed, publication_path=tmp_path / "conflicting.json")


def test_matures_endpoint_first_departure_entry_and_occupancy_without_prices(fixture):
    clock, p, ledger, _, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    # risk_on -> transition -> risk_off -> transition -> risk_on; endpoint
    # returns to origin while first departure and risk-off entry remain events.
    states = ["risk_on", "transition", "risk_off", "transition", "risk_on"] + ["risk_on"] * 9
    clock["now"] = datetime.fromisoformat(p["calendar"][13]) + timedelta(minutes=1)
    official = snapshot(p, states, clock)
    result = ledger.evaluate_matured(official)
    assert result["created"] == 3 and not result["pending"]
    with sqlite3.connect(ledger.path) as conn:
        evaluated = json.loads(conn.execute("SELECT content FROM research_evaluations WHERE horizon_weeks=4").fetchone()[0])
    assert evaluated["outcome"] == {"endpoint": "risk_on", "first_departure": "transition", "any_risk_off_entry": True,
                                    "any_risk_off_occupancy": True, "worsening": False, "recovery": False}
    score = evaluated["scores"][0]["targets"]
    assert score["endpoint"]["log_loss"] == pytest.approx(-__import__("math").log(.4))
    assert score["endpoint"]["brier"] == pytest.approx(.36 + .09 + .09)
    assert score["first_departure"]["brier"] == pytest.approx(.09 + .25 + .04)
    assert score["any_risk_off_entry"]["brier"] == pytest.approx(.36)
    assert ledger.evaluate_matured(official)["unchanged"] == 3
    # A later larger snapshot cannot rewrite the original availability/source receipt.
    official["observations"].append({"date": p["calendar"][14], "state": "risk_on"})
    clock["now"] = datetime.fromisoformat(p["calendar"][14]) + timedelta(minutes=1)
    official["available_at"] = clock["now"].isoformat()
    assert ledger.evaluate_matured(official)["unchanged"] == 3


@pytest.mark.parametrize("leave_and_return", [False, True])
def test_risk_off_origin_distinguishes_new_entry_from_future_occupancy(fixture, leave_and_return):
    clock, p, ledger, _, tmp_path = fixture
    frozen, _, _ = freeze(fixture, current="risk_off")
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    states = ["risk_off"] * 14
    if leave_and_return:
        states[1] = "transition"
    clock["now"] = datetime.fromisoformat(p["calendar"][13]) + timedelta(minutes=1)
    ledger.evaluate_matured(snapshot(p, states, clock))
    with sqlite3.connect(ledger.path) as conn:
        result = json.loads(conn.execute("SELECT content FROM research_evaluations WHERE horizon_weeks=4").fetchone()[0])
    assert result["outcome"]["any_risk_off_entry"] is leave_and_return
    assert result["outcome"]["any_risk_off_occupancy"] is True
    assert result["outcome"]["first_departure"] == ("transition" if leave_and_return else "no_departure")


def test_maturity_and_missing_intermediate_official_observations_are_not_silently_skipped(fixture):
    clock, p, ledger, _, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][4]) + timedelta(minutes=1)
    official = snapshot(p, ["risk_on"] * 5, clock)
    official["observations"].pop(2)
    result = ledger.evaluate_matured(official)
    assert result["created"] == 1
    assert [p["reason"] for p in result["pending"]] == ["official_observations_missing", "not_matured"]
    assert result["pending"][0]["missing_dates"] == [p["calendar"][2]]


@pytest.mark.parametrize("mutation", ["label", "future_available", "future_state", "duplicate", "invalid_state", "revised_origin", "revised_path"])
def test_official_state_identity_availability_and_revisions_are_enforced(fixture, mutation):
    clock, p, ledger, _, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][13]) + timedelta(minutes=1)
    official = snapshot(p, ["risk_on"] * 14, clock)
    ledger.evaluate_matured(official)
    if mutation == "label": official["label"]["sha256"] = "0" * 64
    elif mutation == "future_available": official["available_at"] = (clock["now"] + timedelta(seconds=1)).isoformat()
    elif mutation == "future_state": official["observations"].append({"date": p["calendar"][14], "state": "risk_on"})
    elif mutation == "duplicate": official["observations"].append(deepcopy(official["observations"][0]))
    elif mutation == "invalid_state": official["observations"][1]["state"] = "unknown"
    elif mutation == "revised_origin": official["observations"][0]["state"] = "transition"
    else: official["observations"][2]["state"] = "transition"
    with pytest.raises(prospective.ResearchLedgerError):
        ledger.evaluate_matured(official)
    assert table_count(ledger, "research_evaluations") == 3


def test_never_touches_an_existing_unrelated_database(tmp_path):
    path = tmp_path / "not-research.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE important_data (value TEXT)")
        conn.execute("INSERT INTO important_data VALUES ('keep')")
    before = path.read_bytes()
    with pytest.raises(prospective.ResearchLedgerError, match="dedicated"):
        prospective.ForecastResearchLedger(path, create=True)
    assert path.read_bytes() == before
    with pytest.raises(prospective.ResearchLedgerError, match="explicit init"):
        prospective.ForecastResearchLedger(tmp_path / "does-not-exist.sqlite")
    assert not (tmp_path / "does-not-exist.sqlite").exists()


@pytest.mark.parametrize("table", prospective._TABLES)
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_append_only_triggers_protect_all_records(fixture, table, operation):
    clock, p, ledger, _, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][13]) + timedelta(minutes=1)
    ledger.evaluate_matured(snapshot(p, ["risk_on"] * 14, clock))
    sql = f"DELETE FROM {table}" if operation == "DELETE" else f"UPDATE {table} SET content=content"
    with sqlite3.connect(ledger.path) as conn, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)


def test_content_corruption_is_detected_before_reporting_or_issuing(fixture):
    _, _, ledger, digest, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    with sqlite3.connect(ledger.path) as conn:
        conn.execute("DROP TRIGGER research_forecasts_update")
        conn.execute("UPDATE research_forecasts SET content='{}'")
        conn.execute("CREATE TRIGGER research_forecasts_update BEFORE UPDATE ON research_forecasts BEGIN SELECT RAISE(ABORT, 'append-only research ledger'); END")
    with pytest.raises(prospective.ResearchLedgerError, match="hash mismatch"):
        ledger.readiness(digest)


def test_event_counts_and_continuous_coverage_block_elapsed_time_only_readiness(fixture):
    clock, p, ledger, digest, tmp_path = fixture
    for index in range(6):
        frozen, _, _ = freeze(fixture, index=index)
        ledger.issue(frozen, publication_path=tmp_path / f"issued-{index}.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][-1]) + timedelta(minutes=1)
    ledger.evaluate_matured(snapshot(p, ["risk_on"] * len(p["calendar"]), clock))
    result = ledger.readiness(digest)
    candidate = result["candidates"][0]
    assert result["missing_origins"] == p["calendar"][6:8]
    assert candidate["event_counts"] == {"worsening": 0, "recovery": 0, "risk_off_episodes": 0}
    assert candidate["status"] == "collecting_evidence"
    assert candidate["checks"]["13w_paired_origin_count"] is True
    assert candidate["checks"]["issuance_coverage"] is False
    assert candidate["checks"]["worsening_count"] is False
    assert candidate["automatic_promotion"] is False


def test_gap_is_not_collapsed_in_same_origin_block_uncertainty(fixture):
    clock, p, ledger, digest, tmp_path = fixture
    for index in (0, 1, 3, 4, 5):
        frozen, _, _ = freeze(fixture, index=index)
        ledger.issue(frozen, publication_path=tmp_path / f"issued-{index}.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][-1]) + timedelta(minutes=1)
    ledger.evaluate_matured(snapshot(p, ["risk_on"] * len(p["calendar"]), clock))
    candidate = ledger.readiness(digest)["candidates"][0]
    assert not candidate["checks"]["1w_continuous_evidence"]
    assert candidate["comparisons"][0]["paired_origins"] == 5
    assert candidate["comparisons"][0]["log_loss_delta_ci_high"] is None


def test_one_week_models_get_paired_scores_and_manual_review_only(tmp_path, monkeypatch):
    clock = {"now": ORIGIN - timedelta(days=1)}
    monkeypatch.setattr(prospective, "_utc_now", lambda: clock["now"])
    p = protocol(origin_count=8, horizons=(1,))
    ledger = prospective.ForecastResearchLedger(tmp_path / "one-week.sqlite", create=True)
    digest = ledger.register_protocol(p)["protocol_sha256"]
    local = clock, p, ledger, digest, tmp_path
    states = ["risk_on", "transition", "risk_off", "risk_off", "transition", "risk_on", "risk_on", "risk_off", "risk_off"] + ["risk_on"] * 12
    for index in range(8):
        probabilities = {"candidate": {state: .8 if state == states[index + 1] else .1 for state in prospective.STATE_ORDER},
                         "baseline": {state: 1 / 3 for state in prospective.STATE_ORDER}}
        frozen, _, _ = freeze(local, index=index, current=states[index], one_week=probabilities)
        ledger.issue(frozen, publication_path=tmp_path / f"issued-{index}.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][-1]) + timedelta(minutes=1)
    ledger.evaluate_matured(snapshot(p, states, clock))
    summary = ledger.readiness(digest)
    candidate = summary["candidates"][0]
    assert candidate["event_counts"] == {"worsening": 3, "recovery": 2, "risk_off_episodes": 2}
    assert candidate["worsening_recall"] == 1
    assert candidate["false_alarm_fraction_among_stays"] == 0
    assert candidate["comparisons"][0]["candidate_log_loss"] == pytest.approx(-__import__("math").log(.8))
    assert candidate["comparisons"][0]["baseline_log_loss"] == pytest.approx(-__import__("math").log(1 / 3))
    assert candidate["comparisons"][0]["log_loss_delta_ci_high"] < 0
    assert candidate["status"] == "manual_review_ready"
    assert summary["automatic_promotion"] is False
    assert len(list(tmp_path.glob("*.sqlite"))) == 1


def test_unscored_matured_origins_block_readiness(fixture):
    clock, p, ledger, digest, tmp_path = fixture
    frozen, _, _ = freeze(fixture)
    ledger.issue(frozen, publication_path=tmp_path / "issued.json")
    clock["now"] = datetime.fromisoformat(p["calendar"][13]) + timedelta(minutes=1)
    candidate = ledger.readiness(digest)["candidates"][0]
    assert len(candidate["pending_matured_evaluations"]) == 3
    assert candidate["checks"]["13w_evaluation_coverage"] is False


def test_cli_preview_creates_only_requested_artifact_and_has_no_clock_override(fixture, capsys):
    _, p, ledger, _, tmp_path = fixture
    _, block, info = freeze(fixture, evidence_track="reconstructed_market")
    for name, value in (("block", block), ("protocol", p), ("information", info)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    script = Path(__file__).resolve().parents[1] / "scripts/manage_forecast_research_ledger.py"
    main = runpy.run_path(str(script))["main"]
    args = ["preview", "--block", str(tmp_path / "block.json"), "--protocol", str(tmp_path / "protocol.json"),
            "--information-set", str(tmp_path / "information.json"), "--output", str(tmp_path / "preview.json")]
    assert main(args) == 0
    assert json.loads((tmp_path / "preview.json").read_text())["eligible_for_issue_at_preview"] is False
    assert table_count(ledger, "research_forecasts") == 0
    assert main(args) == 2  # preview never replaces an existing artifact
    with pytest.raises(SystemExit):
        main(["issue", "--ledger", str(ledger.path), "--frozen", "unused.json", "--publication", "unused2.json", "--now", "2026-01-01"])
    assert "unrecognized arguments: --now" in capsys.readouterr().err


def audit_family_block(fixture):
    _, block, _ = freeze(fixture, evidence_track="reconstructed_market")
    path_model, baseline = block["latest"]
    path_model["model"] = "directional_duration_hazard"
    baseline["model"] = "markov_duration_path_baseline"
    boundary = deepcopy(path_model)
    boundary["model"] = "boundary_filtered_history"
    boundary["paths"] = boundary["paths"][:1]
    asymmetric = deepcopy(boundary)
    asymmetric["model"] = "boundary_asymmetric_ewma"
    block["latest"] = [path_model, baseline, boundary, asymmetric]
    for model in block["latest"]:
        model.pop("version")
        model.pop("recipe_sha256")
    block["protocol"] = {"version": "synthetic-audit-v1", "sha256": "b" * 64}
    root = Path(__file__).resolve().parents[1]
    block["source_hashes"] = {"input-manifest.json": "a" * 64, "states.pkl": "b" * 64,
                              "src/regime_lab/analysis/labels.py": hashlib.sha256((root / "src/regime_lab/analysis/labels.py").read_bytes()).hexdigest()}
    return block


def template_for(block):
    root = Path(__file__).resolve().parents[1]
    raw = (root / "config/label-spec.json").read_bytes()
    spec = json.loads(raw)
    return prospective.build_protocol_template(block, label_version=spec["specs"][spec["default_spec"]]["version"],
                                                label_sha256=hashlib.sha256(raw).hexdigest(), version="future-fixture-v1")


def test_template_and_reconstructed_information_preserve_dst_and_unknown_availability(fixture):
    clock, _, _, _, _ = fixture
    block = audit_family_block(fixture)
    # Explicitly include the fall DST transition in the same 1/4/13 calendar.
    block["data_as_of"] = "2026-09-04T20:00:00+00:00"
    p = template_for(block)
    assert p["calendar"][1] == "2026-09-11T20:00:00+00:00"
    assert p["calendar"][4] == "2026-10-02T20:00:00+00:00"
    assert p["calendar"][13] == "2026-12-04T21:00:00+00:00"
    assert len(p["calendar"]) == 117
    assert p["criteria"]["minimum_risk_off_episodes"] > 0
    assert [m["horizons"] for m in p["models"]] == [[1, 4, 13], [1, 4, 13], [1], [1]]
    for model in p["models"]:
        assert prospective.content_sha256(model["recipe_components"]) == model["recipe_sha256"]
        assert "scikit-learn" in model["recipe_components"]["runtime_versions"]
    info = prospective.reconstructed_preview_information(block)
    assert all(row["retrieved_at"] is None and row["available_at"] is None for row in info["sources"])
    assert all(row["availability_basis"] == "reconstructed" for row in info["sources"])


def test_template_cli_builds_complete_concrete_bundle_without_a_ledger(fixture, capsys):
    _, _, ledger, _, tmp_path = fixture
    block = audit_family_block(fixture)
    block_path = tmp_path / "research-block.json"
    block_path.write_text(json.dumps(block))
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/manage_forecast_research_ledger.py"))["main"]
    output = tmp_path / "review"
    args = ["template", "--block", str(block_path), "--label-spec", str(root / "config/label-spec.json"),
            "--version", "review-v1", "--output-dir", str(output)]
    assert main(args) == 0
    assert {p.name for p in output.iterdir()} == {"preview.md", "protocol.preview.json", "protocol.future-template.json",
                                               "frozen-preview.json", "information-set.preview.json", "preview-manifest.json"}
    frozen = json.loads((output / "frozen-preview.json").read_text())
    assert not frozen["eligible_for_issue_at_preview"]
    assert frozen["forecast"]["evidence_track"] == "reconstructed_market"
    assert frozen["forecast"]["models"][0]["paths"][0]["endpoint"] == block["latest"][0]["paths"][0]["endpoint"]
    future = json.loads((output / "protocol.future-template.json").read_text())
    assert future["calendar"][0] == "2027-01-15T21:00:00+00:00"
    assert not list(output.rglob("*.sqlite"))
    assert table_count(ledger, "research_forecasts") == 0
    assert main(args) == 2


def captured_input_fixture(fixture, monkeypatch):
    import pandas as pd
    from regime_lab.operational_forecast import frame_sha256
    clock, _, _, _, tmp_path = fixture
    block = audit_family_block(fixture)
    p = template_for(block)
    inputs = tmp_path / "actual-inputs"
    inputs.mkdir()
    index = pd.date_range(end=ORIGIN, periods=3, freq="7D")
    canonical = pd.DataFrame({"spy_close": [98., 99., 100.]}, index=index)
    states = pd.Series(["transition", "risk_on", "risk_on"], index=index)
    canonical.to_pickle(inputs / "canonical.pkl")
    states.to_pickle(inputs / "states.pkl")
    manifest = {"data_as_of": ORIGIN.isoformat(), "frames": {"canonical": frame_sha256(canonical), "states": frame_sha256(states)}}
    (inputs / "input-manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame({"date": [d.isoformat() for d in index], "state": states.to_list()}).to_csv(inputs / "official.csv", index=False)
    pd.DataFrame({"origin_date": [index[0].isoformat()], "model": ["fixture_baseline"]}).to_csv(inputs / "oos.csv", index=False)
    capture = tmp_path / "captured"
    result = prospective.capture_research_inputs(protocol=p, input_directory=inputs, source_oos=inputs / "oos.csv",
                                                 official_state_history=inputs / "official.csv", output_directory=capture)
    calls = []
    def producer(canonical_input, states_input, baseline_input):
        calls.append((canonical_input.copy(), states_input.copy(), baseline_input.copy()))
        return deepcopy(block["protocol"]), deepcopy(block["latest"])
    monkeypatch.setattr(prospective, "_run_fixed_latest", producer)
    return p, capture, calls, block, inputs, result


def test_capture_and_prepare_are_executable_and_attest_only_a_fresh_run(fixture, monkeypatch):
    clock, _, ledger, _, _ = fixture
    p, capture, calls, block, inputs, receipt = captured_input_fixture(fixture, monkeypatch)
    assert receipt["captured_at"] == clock["now"].isoformat()
    clock["now"] += timedelta(seconds=10)
    prepared = prospective.prepare_research_forecast(protocol=p, capture_directory=capture)
    assert len(calls) == 1
    assert prepared["producer_block"]["evidence_track"] == "prospective_inputs"
    assert prepared["producer_block"]["generated_at"] == clock["now"].isoformat()
    assert prepared["producer_block"]["historical_diagnostics_included"] is False
    assert prepared["frozen"]["eligible_for_issue_at_preview"] is True
    assert prepared["issued_forecasts"] == 0
    assert block["evidence_track"] == "reconstructed_market"  # source fixture never relabeled
    for model, contract in zip(prepared["frozen"]["forecast"]["models"], p["models"]):
        assert model["version"] == contract["version"]
        assert model["recipe_sha256"] == contract["recipe_sha256"]
    assert table_count(ledger, "research_forecasts") == 0
    manifest = json.loads((capture / "capture-manifest.json").read_text())
    assert manifest["historical_availability_claim"] is False
    assert all(source["available_at"] == source["retrieved_at"] for source in prepared["information_set"]["sources"])


@pytest.mark.parametrize("mutation", ["expired", "input_bytes", "capture_backdate", "source_backdate", "label", "runtime", "recipe_hash", "official_state_mismatch"])
def test_preparation_rejects_stale_tampered_or_unattested_inputs(fixture, monkeypatch, mutation):
    clock, _, _, _, _ = fixture
    p, capture, calls, _, _, _ = captured_input_fixture(fixture, monkeypatch)
    manifest_path = capture / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "expired": clock["now"] += timedelta(days=3)
    elif mutation == "input_bytes": (capture / "states.pkl").write_bytes(b"not a pickle and must fail before unpickling")
    elif mutation == "capture_backdate": manifest["captured_at"] = (ORIGIN - timedelta(seconds=1)).isoformat()
    elif mutation == "source_backdate": manifest["files"][0]["stored_at"] = (ORIGIN - timedelta(seconds=1)).isoformat()
    elif mutation == "label": p["label"]["sha256"] = "0" * 64
    elif mutation == "runtime":
        p["models"][0]["recipe_components"]["runtime_versions"]["numpy"] = "wrong-version"
        p["models"][0]["recipe_sha256"] = prospective.content_sha256(p["models"][0]["recipe_components"])
    elif mutation == "recipe_hash": p["models"][0]["recipe_sha256"] = "0" * 64
    else:
        path = capture / "official-state-history.csv"
        path.write_text(path.read_text().replace("transition", "risk_off"))
        raw = path.read_bytes()
        for row in manifest["files"]:
            if row["name"] == path.name:
                row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(prospective.ResearchLedgerError):
        prospective.prepare_research_forecast(protocol=p, capture_directory=capture)
    assert not calls


def test_preparation_rechecks_clock_and_input_hash_after_fresh_fit(fixture, monkeypatch):
    clock, _, _, _, _ = fixture
    p, capture, _, block, _, _ = captured_input_fixture(fixture, monkeypatch)
    def producer(*args):
        clock["now"] = ORIGIN + timedelta(seconds=p["issue_deadline_seconds"])
        return block["protocol"], deepcopy(block["latest"])
    monkeypatch.setattr(prospective, "_run_fixed_latest", producer)
    with pytest.raises(prospective.ResearchLedgerError, match="expired"):
        prospective.prepare_research_forecast(protocol=p, capture_directory=capture)


def test_current_historical_block_cannot_be_accepted_by_prepare_cli(fixture, capsys):
    _, _, _, _, tmp_path = fixture
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/manage_forecast_research_ledger.py"))["main"]
    with pytest.raises(SystemExit):
        main(["prepare", "--protocol", "future.json", "--capture-dir", "capture", "--output-dir", str(tmp_path / "never"),
              "--block", "historical-forecast.json"])
    assert "unrecognized arguments: --block" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()
