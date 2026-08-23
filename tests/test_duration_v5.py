from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.duration import (
    SPELL_COLUMNS,
    causal_spell_table,
    conditional_duration_estimates,
    conditional_duration_summary,
    duration_context,
    kaplan_meier_table,
)


def _states(values: list[str], *, start: str = "2020-01-03") -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="7D"),
        dtype="object",
    )


def _spells(
    durations: list[int],
    events: list[bool],
    *,
    state: str = "risk_on",
) -> pd.DataFrame:
    starts = pd.date_range("2010-01-01", periods=len(durations), freq="28D")
    rows = []
    for episode_id, (start, duration, event) in enumerate(
        zip(starts, durations, events, strict=True)
    ):
        end = start + pd.DateOffset(weeks=duration - 1)
        rows.append(
            {
                "episode_id": episode_id,
                "state": state,
                "start_date": start,
                "end_date": end,
                "departure_date": end + pd.DateOffset(weeks=1)
                if event
                else pd.NaT,
                "duration_weeks": duration,
                "event_observed": event,
                "is_current": not event,
            }
        )
    return pd.DataFrame(rows, columns=SPELL_COLUMNS)


def test_causal_spell_table_right_censors_only_the_current_history_end() -> None:
    states = _states(
        ["risk_on"] * 3 + ["transition"] * 2 + ["risk_off"] * 4
    )

    spells = causal_spell_table(states)

    assert spells["state"].tolist() == ["risk_on", "transition", "risk_off"]
    assert spells["duration_weeks"].tolist() == [3, 2, 4]
    assert spells["event_observed"].tolist() == [True, True, False]
    assert spells["is_current"].tolist() == [False, False, True]
    assert spells.loc[0, "departure_date"] == states.index[3]
    assert pd.isna(spells.loc[2, "departure_date"])


def test_spell_table_is_invariant_to_states_after_as_of() -> None:
    observed = _states(["risk_on"] * 2 + ["transition"] * 3 + ["risk_off"] * 2)
    future = _states(
        ["transition", "risk_on", "risk_on"],
        start=(observed.index[-1] + pd.DateOffset(weeks=1)).date().isoformat(),
    )
    extended = pd.concat([observed, future])

    left = causal_spell_table(extended, as_of=observed.index[-1])
    right = causal_spell_table(observed)

    pd.testing.assert_frame_equal(left, right)
    assert not bool(left.iloc[-1]["event_observed"])


def test_kaplan_meier_table_matches_hand_calculation() -> None:
    spells = _spells([1, 2, 2, 3], [True, True, False, True])

    km = kaplan_meier_table(spells, state="risk_on")

    assert km["at_risk"].tolist() == [4, 3, 1]
    assert km["events"].tolist() == [1, 1, 1]
    assert km["censored"].tolist() == [0, 1, 0]
    assert km["survival"].tolist() == pytest.approx([0.75, 0.5, 0.0])


def test_conditional_survival_median_and_discrete_rmst() -> None:
    spells = _spells([2, 4, 6, 8, 10], [True, True, True, True, False])
    km = kaplan_meier_table(spells, state="risk_on")

    estimate = conditional_duration_estimates(
        km,
        elapsed_weeks=2,
        horizons=(2, 4),
        restriction_weeks=4,
    )

    assert estimate is not None
    assert estimate["conditional_survival"] == pytest.approx(
        {"2w": 0.8, "4w": 0.6}
    )
    assert estimate["departure_probability"] == pytest.approx(
        {"2w": 0.2, "4w": 0.4}
    )
    assert estimate["median_remaining_weeks"] == 5
    assert estimate["restricted_mean_remaining_weeks"] == pytest.approx(3.2)


def test_conditioning_keeps_spells_that_depart_on_the_next_observation() -> None:
    spells = _spells([2, 2, 4, 4], [True, True, True, False])
    km = kaplan_meier_table(spells, state="risk_on")

    estimate = conditional_duration_estimates(
        km,
        elapsed_weeks=2,
        horizons=(1,),
        restriction_weeks=2,
    )

    assert estimate is not None
    assert estimate["conditional_survival"] == pytest.approx({"1w": 0.5})
    assert estimate["departure_probability"] == pytest.approx({"1w": 0.5})
    assert estimate["restricted_mean_remaining_weeks"] == pytest.approx(1.5)


def test_episode_bootstrap_is_deterministic() -> None:
    spells = _spells(
        [2, 3, 4, 5, 6, 7, 8],
        [True, True, True, True, True, True, False],
    )
    kwargs = {
        "state": "risk_on",
        "elapsed_weeks": 2,
        "horizons": (4, 13),
        "restriction_weeks": 52,
        "min_completed_spells": 5,
        "bootstrap_resamples": 199,
        "bootstrap_seed": 17,
    }

    first = conditional_duration_summary(spells, **kwargs)
    second = conditional_duration_summary(spells, **kwargs)

    assert first["status"] == "ok"
    assert first["ci95"] == second["ci95"]
    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["valid_resamples"] > 0


def test_minimum_completed_spell_support_fails_closed() -> None:
    spells = _spells([2, 8], [True, False])

    summary = conditional_duration_summary(
        spells,
        state="risk_on",
        elapsed_weeks=3,
        min_completed_spells=2,
        bootstrap_resamples=0,
    )

    assert summary["status"] == "insufficient_history"
    assert summary["conditional_survival"] == {"4w": None, "13w": None}
    assert summary["restricted_mean_remaining_weeks"] is None
    assert summary["ci95"] is None


def test_duration_context_is_causal_under_future_append() -> None:
    observed = _states(
        ["risk_on", "transition"] * 6 + ["risk_off"] * 3
    )
    future = _states(
        ["transition", "risk_on", "risk_off"],
        start=(observed.index[-1] + pd.DateOffset(weeks=1)).date().isoformat(),
    )
    extended = pd.concat([observed, future])
    kwargs = {
        "as_of": observed.index[-1],
        "min_completed_spells": 1,
        "bootstrap_resamples": 0,
    }

    assert duration_context(extended, **kwargs) == duration_context(
        observed, **kwargs
    )


def test_irregular_weekly_index_is_rejected() -> None:
    states = pd.Series(
        ["risk_on", "transition"],
        index=pd.to_datetime(["2020-01-03", "2020-01-17"]),
    )

    with pytest.raises(ValueError, match="consecutive weekly"):
        causal_spell_table(states)


def test_weekly_history_accepts_utc_dst_hour_shift() -> None:
    index = pd.date_range(
        "2026-02-27 16:00",
        periods=6,
        freq="W-FRI",
        tz="America/New_York",
    ).tz_convert("UTC")
    states = pd.Series(
        ["risk_on"] * 3 + ["transition"] * 3,
        index=index,
        dtype="object",
    )

    spells = causal_spell_table(states)

    assert spells["duration_weeks"].tolist() == [3, 3]
