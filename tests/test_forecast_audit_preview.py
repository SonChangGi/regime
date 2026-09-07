from copy import deepcopy
from pathlib import Path
import pytest

from regime_lab.research.contract import validate_research_extensions
from scripts.build_forecast_audit_preview import require_local_output


def _information():
    return {'schema_version':'regime-forecast-information/1','data_as_of':'2026-09-04T20:00:00+00:00',
        'rows':[{'model':'candidate','baseline_model':'baseline','feature_block':'vix3m',
                 'evaluation_split':'holdout','matched_n':191,'baseline_log_loss':.3,
                 'candidate_log_loss':.31,'delta_log_loss':.01,'delta_brier':.002}]}


def test_optional_tables_bind_cutoff_sample_and_score_difference():
    info=_information()
    validate_research_extensions({'forecast_information':info},data_as_of=info['data_as_of'])
    with pytest.raises(ValueError,match='cutoff differs'):
        validate_research_extensions({'forecast_information':info},data_as_of='2026-08-28T20:00:00+00:00')
    for key,value,pattern in [('matched_n',0,'sample count'),('delta_log_loss',-.01,'difference'),
                              ('candidate_log_loss',float('nan'),'JSON compliant')]:
        broken=deepcopy(info);broken['rows'][0][key]=value
        with pytest.raises(ValueError,match=pattern):
            validate_research_extensions({'forecast_information':broken})


def test_preview_cannot_write_to_publication_or_reference_project():
    for output in (Path('publication/live'),Path('../regime-improvements/build/preview'),Path('web')):
        with pytest.raises(ValueError,match='must stay under'):
            require_local_output(output)
    require_local_output(Path('build/forecast-audit-improvements/preview'))
