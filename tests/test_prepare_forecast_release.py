"""The release may add one research block; operating decisions stay frozen."""
from copy import deepcopy

import pytest

from scripts.prepare_forecast_release import ReleasePreparationError, _require_staging_destination, validate_forecast_only_update


@pytest.fixture
def pair():
    source = {"meta": {"publication_status": "reviewed_publication", "publication_review": {"old": True}, "generation_manifest_sha256": "a"*64, "data_as_of": "2026-09-04"},
              "model": {"champion": "original", "lifecycle": {"publication": {"status": "reviewed_publication"}, "deployment": {"status": "operating"}}, "execution_parameters": {"keep": True}},
              "weekly": [{"current": {"state": "risk_on"}, "next_week": {"state": "risk_on"}, "duration_context": {"weeks": 4}}],
              "forecast": {"target": "2026-09-11"}, "selection": {"model": "original"},
              "research": {"existing": {"preserved": True}}}
    preview = deepcopy(source)
    preview["meta"] = {"publication_status": "unpublished", "data_as_of": "2026-09-04"}
    preview["model"]["lifecycle"] = {"publication": {"status": "unpublished"}, "deployment": {"status": "candidate"}}
    preview["research"]["forecast_improvement"] = {"models": ["new"]}
    return source, preview


def test_only_forecast_research_is_permitted_and_inputs_are_unchanged(pair):
    before = deepcopy(pair)
    validate_forecast_only_update(*pair)
    assert pair == before


@pytest.mark.parametrize("mutation", ["forecast", "selection", "weekly_prediction", "weekly_duration", "champion", "settings", "existing_research", "cutoff"])
def test_rejects_every_other_update(pair, mutation):
    source, preview = pair
    if mutation in {"forecast", "selection"}:
        preview[mutation]["changed"] = True
    elif mutation == "weekly_prediction":
        preview["weekly"][0]["next_week"]["state"] = "risk_off"
    elif mutation == "weekly_duration":
        preview["weekly"][0]["duration_context"]["weeks"] = 5
    elif mutation == "champion":
        preview["model"]["champion"] = "new"
    elif mutation == "settings":
        preview["model"]["execution_parameters"]["keep"] = False
    elif mutation == "existing_research":
        preview["research"]["existing"]["preserved"] = False
    else:
        preview["meta"]["data_as_of"] = "2026-09-11"
    with pytest.raises(ReleasePreparationError, match="protected"):
        validate_forecast_only_update(source, preview)


def test_candidate_cannot_inherit_approval(pair):
    pair[1]["meta"]["publication_review"] = {"old": True}
    with pytest.raises(ReleasePreparationError, match="previous review"):
        validate_forecast_only_update(*pair)


def test_missing_new_forecast_block_cannot_be_published(pair):
    pair[1]["research"].pop("forecast_improvement")
    with pytest.raises(ReleasePreparationError, match="no new"):
        validate_forecast_only_update(*pair)


def test_output_must_not_replace_read_only_input(tmp_path, monkeypatch):
    import scripts.prepare_forecast_release as release
    monkeypatch.setattr(release, "project_root", lambda: tmp_path)
    (tmp_path / "build").mkdir()
    source = tmp_path / "build" / "source"
    source.mkdir()
    with pytest.raises(ReleasePreparationError, match="overlaps"):
        _require_staging_destination(source / "output", [source])
    with pytest.raises(ReleasePreparationError, match="must not exist"):
        _require_staging_destination(source, [])
