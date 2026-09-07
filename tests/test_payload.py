from __future__ import annotations

import numpy as np
import pytest
from regime_lab.payload import estimate_from_probabilities, normalized_probabilities


def test_probability_normalization_is_state_ordered_and_only_corrects_rounding() -> None:
    probabilities = normalized_probabilities({'risk_off': .5, 'risk_on': .25, 'transition': .25000001})
    assert list(probabilities) == ['risk_on', 'transition', 'risk_off']
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert estimate_from_probabilities([.25, .25, .5])['state'] == 'risk_off'


@pytest.mark.parametrize('values', [[0,0,0], [np.nan,1,0], [np.inf,0,0], [-.1,.6,.5],
    [2,1,1], [.5,.5,.5], {'risk_on':1}, {'risk_on':1,'transition':0,'risk_off':0,'extra':0}])
def test_broken_model_output_cannot_be_silently_published(values) -> None:
    with pytest.raises(ValueError, match='probabilities'):
        estimate_from_probabilities(values)


def test_pipeline_rejects_nonfinite_probability_before_marking_normal_forecast():
    from regime_lab.pipeline import _comparison_forecast_record
    with pytest.raises(ValueError, match='finite'):
        _comparison_forecast_record({'target_date':'2026-09-11','model':'markov',
            'p_risk_on':np.nan, 'p_transition':1., 'p_risk_off':0., 'fallback':False})
