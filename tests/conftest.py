import pytest

from corr_agent import config
from corr_agent.clients import http, open_meteo


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    http.clear_cache()
    open_meteo.geocode.cache_clear()
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
    yield
    http.clear_cache()
