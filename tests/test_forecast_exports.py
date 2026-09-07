import pytest

from regime_lab.forecast_exports import build_forecast_exports
from regime_lab.publication_contract import PublicContractError


@pytest.mark.parametrize(
    "extra",
    [{"source": "/Users/person/private/input.csv"}, {"raw_observations": [1.0]}],
)
def test_download_refuses_private_material(extra):
    block = {
        "artifacts": [{"label": "전체 결과 JSON", "url": "./data/forecast_information.json"}],
        **extra,
    }
    with pytest.raises(PublicContractError):
        build_forecast_exports({"research": {"forecast_information": block}})


@pytest.mark.parametrize("url", ["../private.json", "./data/missing.json", "https://example.com/result.json"])
def test_download_link_must_resolve_to_packaged_payload_block(url):
    block = {"artifacts": [{"label": "전체 결과 JSON", "url": url}]}
    with pytest.raises(PublicContractError, match="payload-bound download link"):
        build_forecast_exports({"research": {"forecast_information": block}})
