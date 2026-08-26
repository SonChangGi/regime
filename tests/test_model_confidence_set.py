from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from regime_lab.analysis.model_confidence_set import (
    MCS_METHOD,
    model_confidence_set,
    validate_matched_loss_matrix,
)


def _paired_losses() -> pd.DataFrame:
    origins = pd.date_range("2022-01-07", periods=156, freq="W-FRI")
    position = np.arange(len(origins), dtype=float)
    common = 0.45 + 0.04 * np.sin(position / 11.0)
    return pd.DataFrame(
        {
            "best": common,
            "peer": common + 0.015 * np.sin(2.0 * np.pi * position / 13.0),
            "inferior": common + 0.20,
        },
        index=origins,
    )


def test_mcs_is_deterministic_and_retains_indistinguishable_models() -> None:
    losses = _paired_losses()

    first = model_confidence_set(
        losses, alpha=0.10, block_length=13, resamples=499, random_state=17
    )
    second = model_confidence_set(
        losses, alpha=0.10, block_length=13, resamples=499, random_state=17
    )

    assert first == second
    assert first.method == MCS_METHOD
    assert first.retained_models == ("best", "peer")
    assert first.eliminated_models == ("inferior",)
    assert first.termination_reason == "equal_predictive_ability_not_rejected"
    assert first.effective_block_length == 13
    assert [step.rejected for step in first.elimination_path] == [True, False]
    assert first.elimination_path[0].eliminated_model == "inferior"
    assert first.elimination_path[-1].bootstrap_p_value > first.alpha


def test_mcs_eliminates_a_strict_ranking_to_a_singleton() -> None:
    origins = pd.RangeIndex(120, name="origin")
    baseline = 0.3 + 0.02 * np.sin(np.arange(120) / 5.0)
    losses = pd.DataFrame(
        {
            "alpha": baseline,
            "bravo": baseline + 0.1 + 0.02 * np.sin(np.arange(120) / 7.0),
            "charlie": baseline + 0.2 + 0.02 * np.cos(np.arange(120) / 9.0),
        },
        index=origins,
    )

    result = model_confidence_set(losses, resamples=199, random_state=29)

    assert result.retained_models == ("alpha",)
    assert result.eliminated_models == ("charlie", "bravo")
    assert result.termination_reason == "singleton"
    assert tuple(step.eliminated_model for step in result.elimination_path) == (
        "charlie",
        "bravo",
    )


def test_mcs_retains_identical_models_without_elimination() -> None:
    origins = pd.RangeIndex(40, name="origin")
    common = np.linspace(0.2, 0.7, len(origins))
    losses = pd.DataFrame(
        {"zeta": common, "alpha": common, "middle": common}, index=origins
    )

    result = model_confidence_set(losses, resamples=99)

    assert result.retained_models == ("zeta", "alpha", "middle")
    assert result.eliminated_models == ()
    assert len(result.elimination_path) == 1
    assert result.elimination_path[0].test_statistic == 0.0
    assert result.elimination_path[0].bootstrap_p_value == 1.0


def test_validate_matched_loss_matrix_returns_independent_numeric_copy() -> None:
    losses = _paired_losses().iloc[:10].astype(np.float32)

    validated = validate_matched_loss_matrix(losses)

    assert_frame_equal(validated, losses.astype(float))
    validated.iloc[0, 0] = 999.0
    assert losses.iloc[0, 0] != 999.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.assign(peer=np.nan),
            "complete and finite",
        ),
        (
            lambda frame: frame.assign(peer=np.inf),
            "complete and finite",
        ),
        (
            lambda frame: frame.rename(columns={"peer": ""}),
            "non-empty strings",
        ),
        (
            lambda frame: frame.assign(peer=True),
            "not booleans",
        ),
        (
            lambda frame: frame.assign(peer="not-a-loss"),
            "must be numeric",
        ),
        (
            lambda frame: frame.iloc[::-1],
            "must be increasing",
        ),
        (
            lambda frame: frame.set_axis(
                [frame.index[0], *frame.index[1:-1], frame.index[0]], axis=0
            ),
            "must be unique",
        ),
        (
            lambda frame: frame[["best"]],
            "at least two models",
        ),
        (
            lambda frame: frame.iloc[:2],
            "at least three ordered origins",
        ),
    ],
)
def test_loss_matrix_contract_rejects_unmatched_or_invalid_input(
    mutate, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_matched_loss_matrix(mutate(_paired_losses()))


def test_loss_matrix_contract_rejects_duplicate_model_names() -> None:
    losses = _paired_losses()
    losses.columns = ["same", "same", "inferior"]

    with pytest.raises(ValueError, match="model names must be unique"):
        validate_matched_loss_matrix(losses)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"block_length": 0}, "block_length"),
        ({"resamples": 98}, "resamples"),
        ({"random_state": -1}, "random_state"),
        ({"random_state": True}, "random_state"),
    ],
)
def test_mcs_rejects_invalid_bootstrap_configuration(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model_confidence_set(_paired_losses(), **kwargs)
