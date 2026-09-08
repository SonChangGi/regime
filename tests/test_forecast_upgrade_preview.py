from pathlib import Path
import hashlib

import pytest

from scripts.build_forecast_upgrade_preview import require_local_output, read_information_artifacts, validate_weekly_summary


def test_preview_rejects_publication_reference_project_and_symlink_escape(tmp_path):
    for path in (Path("publication/live"), Path("../regime-improvements/build/preview"), tmp_path):
        with pytest.raises(ValueError, match="isolated"):
            require_local_output(path)
    require_local_output(Path("build/forecast-upgrade-20260907/preview"))
    link = tmp_path/"preview"
    link.symlink_to(Path("publication/live").resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="isolated"):
        require_local_output(link)


def test_information_downloads_are_bound_to_one_complete_immutable_run(tmp_path):
    generation = 'a'*24
    path = tmp_path/'runs'/generation/'predictions.csv'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'complete predictions')
    record = {'path':f'runs/{generation}/predictions.csv', 'bytes':len(path.read_bytes()),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    macro = {'artifact_manifest':{'generation':generation,'files':[record]}}
    information = tmp_path/'additional-information.json'
    assert read_information_artifacts(information, macro) == {path:b'complete predictions'}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match="completed run"):
        read_information_artifacts(information, macro)


@pytest.mark.parametrize('relative', ['../outside.csv', '/tmp/outside.csv', 'unversioned.csv', 'runs/other/x.csv'])
def test_information_manifest_cannot_refer_outside_its_frozen_run(tmp_path, relative):
    macro = {'artifact_manifest':{'generation':'a'*24,'files':[{'path':relative,'bytes':0,'sha256':'0'*64}]}}
    with pytest.raises(ValueError, match="immutable run"):
        read_information_artifacts(tmp_path/'additional-information.json', macro)


def test_weekly_summary_cannot_mix_generations_or_score_counts():
    enhancement = {"source_generation_id": "g1", "data_as_of": "2026-09-04T20:00Z"}
    summary = {**enhancement, "schema_version": "regime-weekly-candidates-summary/1",
               "scope": "local_preview_only", "automatic_promotion": False,
               "issued_predictions": 18, "pending_predictions": 18, "matured_predictions": 0}
    validate_weekly_summary(summary, enhancement)
    with pytest.raises(ValueError, match="generation"):
        validate_weekly_summary({**summary, "source_generation_id": "g0"}, enhancement)
    with pytest.raises(ValueError, match="counts"):
        validate_weekly_summary({**summary, "matured_predictions": 1}, enhancement)
