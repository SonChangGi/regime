"""Provide the versioned browser-startup formatter in Node renderer tests."""
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def shared_number_startup(monkeypatch):
    preload = Path(__file__).parent / "fixtures" / "quant-number-preload.cjs"
    import os
    current = os.environ.get("NODE_OPTIONS", "")
    monkeypatch.setenv("NODE_OPTIONS", f'{current} --require={preload}'.strip())
