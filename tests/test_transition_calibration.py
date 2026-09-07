"""Past-block calibration selection, purging, frozen diagnostics and failure paths."""
import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.causal_calibration import (
    TRANSITION_CALIBRATION_VERSION, TransitionCalibrator,
)
from regime_lab.analysis.validation import _calibrate_transition_probability


def history(n=390, horizon=13, raw=0.2):
    origin = pd.date_range("2016-01-08", periods=n, freq="7D", tz="UTC")
    result = pd.DataFrame({
        "origin_date": origin,
        "target_end": origin + pd.Timedelta(7 * horizon, unit="D"),
        "horizon": horizon, "model": "fixture",
        "raw_p_change": raw,
        "actual_change": (np.arange(n) % 5 == 0),
    })
    cutoff = pd.Timestamp("2023-01-01", tz="UTC")
    result["evaluation_split"] = np.where(result.target_end < cutoff, "selection", "retrospective_diagnostic")
    return result.loc[~((result.origin_date < cutoff) & (result.target_end >= cutoff))].reset_index(drop=True)


def test_well_calibrated_identity_is_a_successful_choice_not_a_fallback():
    data = history(raw=.5)
    data["actual_change"] = np.arange(len(data)) % 2 == 0
    fitted = TransitionCalibrator(data).fit("2026-09-04")
    assert fitted.apply(.5) == (.5, "identity", False, "")
    assert fitted.metadata["calibration_version"] == TRANSITION_CALIBRATION_VERSION
    assert 26 <= fitted.metadata["calibration_validation_rows"] <= 78
    assert fitted.metadata["calibration_validation_blocks"] <= 3
    assert fitted.metadata["calibration_identity_log_loss"] <= fitted.metadata["calibration_platt_log_loss"]


def test_platt_is_selected_when_prior_blocks_support_correcting_bias():
    fitted = TransitionCalibrator(history(raw=.8)).fit()
    probability, method, fallback, reason = fitted.apply(.8)
    assert method == "prequential_platt_logit"
    assert probability == pytest.approx(.2, abs=.01)
    assert not fallback and not reason
    assert fitted.metadata["calibration_shrink_weight"] == 1


def test_shrink_candidate_can_win_after_a_change_in_event_frequency():
    data = history(raw=.8)
    # Earlier fits learn a lower frequency; the last three blocks contain a
    # higher frequency. Partial correction can then beat both extremes.
    recent = data.origin_date >= pd.Timestamp("2021-06-01", tz="UTC")
    data.loc[recent, "actual_change"] = (np.arange(int(recent.sum())) % 5 < 3)
    fitted = TransitionCalibrator(data).fit()
    assert fitted.method == "prequential_shrunk_platt_logit"
    assert fitted.metadata["calibration_shrink_weight"] == .25
    assert .6 < fitted.apply(.8)[0] < .8


def test_diagnostic_outcomes_and_future_rows_never_change_selection_or_fit():
    data = history(raw=.8)
    original = TransitionCalibrator(data).fit()
    diagnostic = data.evaluation_split.eq("retrospective_diagnostic")
    assert diagnostic.any()
    data.loc[diagnostic, "actual_change"] = ~data.loc[diagnostic, "actual_change"]
    data.loc[diagnostic, "raw_p_change"] = .01
    extra = data.loc[diagnostic].copy()
    for field in ("origin_date", "target_end"):
        extra[field] += pd.Timedelta(700, unit="D")
    other = TransitionCalibrator(pd.concat([data, extra])).fit("2030-01-04")
    assert original.metadata == other.metadata
    assert original.apply(.65) == other.apply(.65)


def test_block_choice_is_frozen_and_equal_or_future_targets_are_purged():
    data = history(raw=.8, horizon=1)
    origin = pd.Timestamp("2021-06-11", tz="UTC")
    calibrator = TransitionCalibrator(data)
    original = calibrator.fit(origin)
    as_of = pd.Timestamp(original.metadata["calibration_selection_as_of"])
    assert as_of <= origin
    # Includes the target exactly at the refit boundary.
    assert data.target_end.eq(as_of).any()
    changed = data.copy()
    ineligible = changed.target_end >= as_of
    changed.loc[ineligible, "actual_change"] = ~changed.loc[ineligible, "actual_change"]
    changed.loc[ineligible, "raw_p_change"] = .99
    other = TransitionCalibrator(changed).fit(origin)
    assert original.metadata == other.metadata
    assert original.apply(.7) == other.apply(.7)
    assert calibrator.fit(as_of + pd.Timedelta(1, unit="D")) is original
    assert calibrator.fit(as_of + pd.Timedelta(180, unit="D")) is original


def test_each_inner_fit_excludes_overlapping_horizon_targets(monkeypatch):
    data = history(raw=.8)
    calibrator = TransitionCalibrator(data)
    training_frames = []
    original_fit = calibrator._fit

    def record(frame):
        training_frames.append(frame.copy())
        return original_fit(frame)

    monkeypatch.setattr(calibrator, "_fit", record)
    fitted = calibrator.fit()
    selection = data.loc[data.evaluation_split.eq("selection")]
    starts = selection.origin_date.map(calibrator._block_start)
    width = pd.Timedelta(182, unit="D")
    completed = sorted(starts.loc[starts + width <= calibrator.cutoff].unique())[-3:]
    assert len(training_frames) == 4  # three comparisons, one final fit
    for frame, validation_start in zip(training_frames[:3], completed, strict=True):
        assert (frame.target_end < validation_start).all()
        assert frame.target_end.max() < validation_start
    assert (training_frames[-1].target_end < calibrator.cutoff).all()
    assert pd.Timestamp(fitted.metadata["calibration_validation_last_target"]) < calibrator.cutoff


def test_insufficient_evidence_and_fit_failure_have_explicit_identity_fallback(monkeypatch):
    fitted = TransitionCalibrator(history(n=10)).fit()
    assert fitted.apply(.3)[:3] == (.3, "identity", True)
    assert fitted.reason.startswith("insufficient_prequential_rows")
    calibrator = TransitionCalibrator(history(raw=.8))
    monkeypatch.setattr(calibrator, "_fit", lambda _: None)
    fitted = calibrator.fit()
    assert fitted.fallback
    assert fitted.reason == "insufficient_past_validation_blocks"


def test_one_class_validation_cannot_select_a_calibrator():
    data = history(raw=.8)
    data["actual_change"] = False
    fitted = TransitionCalibrator(data).fit()
    assert fitted.fallback and fitted.method == "identity"


@pytest.mark.parametrize("probability", [np.nan, np.inf, -.1, 1.1])
def test_invalid_live_or_history_probability_is_rejected(probability):
    data = history()
    calibrator = TransitionCalibrator(data)
    with pytest.raises(ValueError, match="probability"):
        calibrator.fit().apply(probability)
    data.loc[0, "raw_p_change"] = probability
    with pytest.raises(ValueError, match="probabilities"):
        TransitionCalibrator(data)


@pytest.mark.parametrize("split", ["selection", "retrospective_diagnostic"])
def test_cross_boundary_history_is_rejected_even_if_split_strings_agree(split):
    data = history(n=2, horizon=1)
    data.loc[0, "origin_date"] = pd.Timestamp("2022-12-30", tz="UTC")
    data.loc[0, "target_end"] = pd.Timestamp("2023-01-06", tz="UTC")
    data.loc[0, "evaluation_split"] = split
    with pytest.raises(ValueError, match="cross-boundary"):
        TransitionCalibrator(data)


def test_explicit_v2_operational_interface_matches_frozen_benchmark_fit():
    data = history(raw=.8)
    data = data.loc[data.evaluation_split.eq("selection")]
    data["calibration_version"] = "transition-calibration/2"
    result = _calibrate_transition_probability(.8, data, minimum_rows=12, random_state=17)
    assert len(result) == 4
    assert result == TransitionCalibrator(data).fit("2026-09-04").apply(.8)
