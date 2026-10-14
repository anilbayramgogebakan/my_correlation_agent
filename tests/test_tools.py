import json

import numpy as np
import pandas as pd
import pytest

from corr_agent import config
from corr_agent.clients import fred, open_meteo, wikimedia
from corr_agent.data_store import DatasetStore
from corr_agent.tools import build_tools

PLACE = {"name": "Berlin", "country": "Germany", "latitude": 52.5, "longitude": 13.4, "timezone": "Europe/Berlin"}


def call(tools, name, **kwargs):
    tool = next(t for t in tools if t.name == name)
    return json.loads(tool.invoke(kwargs))


@pytest.fixture
def fake_apis(monkeypatch):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    temp = pd.Series(rng.normal(15, 6, len(idx)), index=idx)
    views = 1000 + 40 * temp + rng.normal(0, 20, len(idx))

    monkeypatch.setattr(open_meteo, "geocode", lambda name: PLACE)
    monkeypatch.setattr(open_meteo, "daily_weather_history", lambda *a, **k: (temp, "°C"))
    monkeypatch.setattr(open_meteo, "daily_weather_forecast",
                        lambda *a, **k: (pd.Series([10.0, 20.0], index=pd.date_range("2025-06-01", periods=2)), "°C"))
    monkeypatch.setattr(wikimedia, "resolve_title", lambda article, lang="en": "Ice cream")
    monkeypatch.setattr(wikimedia, "daily_pageviews", lambda *a, **k: views)


def test_fetch_tools_store_data_and_return_ids(fake_apis):
    store = DatasetStore()
    tools = build_tools(store)
    w = call(tools, "fetch_weather_history", location="Berlin", variable="temperature_2m_mean")
    p = call(tools, "fetch_wikipedia_pageviews", article="ice cream")
    assert (w["dataset_id"], p["dataset_id"]) == ("ds1", "ds2")
    assert w["n_points"] == 120 and w["units"] == "°C"
    assert p["article"] == "Ice cream"
    assert len(store) == 2


def test_analysis_with_forecast(fake_apis):
    store = DatasetStore()
    tools = build_tools(store)
    call(tools, "fetch_weather_history", location="Berlin", variable="temperature_2m_mean")
    call(tools, "fetch_wikipedia_pageviews", article="Ice cream")
    call(tools, "fetch_weather_forecast", location="Berlin", variable="temperature_2m_mean", days=2)
    out = call(tools, "run_correlation_analysis", x_dataset_id="ds1", y_dataset_id="ds2", forecast_dataset_id="ds3")
    assert out["pearson_r"] > 0.9
    assert len(out["forecast"]) == 2
    assert len(store.analyses) == 1


def test_forecast_must_match_x_variable(fake_apis):
    store = DatasetStore()
    tools = build_tools(store)
    call(tools, "fetch_weather_history", location="Berlin", variable="temperature_2m_mean")
    call(tools, "fetch_wikipedia_pageviews", article="Ice cream")
    call(tools, "fetch_weather_forecast", location="Berlin", variable="precipitation_sum")
    out = call(tools, "run_correlation_analysis", x_dataset_id="ds1", y_dataset_id="ds2", forecast_dataset_id="ds3")
    assert "same variable" in out["error"]


def test_errors_are_returned_not_raised(fake_apis):
    tools = build_tools(DatasetStore())
    out = call(tools, "run_correlation_analysis", x_dataset_id="ds9", y_dataset_id="ds1")
    assert "Unknown dataset_id" in out["error"]
    out = call(tools, "fetch_weather_history", location="Berlin", variable="temperature_2m_mean",
               start_date="last year")
    assert "YYYY-MM-DD" in out["error"]


def test_fred_tools_only_with_key(monkeypatch):
    assert "fetch_fred_series" in [t.name for t in build_tools(DatasetStore())]
    monkeypatch.setattr(config, "FRED_API_KEY", None)
    assert "fetch_fred_series" not in [t.name for t in build_tools(DatasetStore())]


def test_fred_series_keeps_frequency(monkeypatch):
    monkeypatch.setattr(fred, "series_info", lambda sid: {"series_id": sid, "title": "Unemployment Rate",
                                                          "frequency": "M", "units": "%"})
    monkeypatch.setattr(fred, "observations", lambda *a: pd.Series(
        [4.0, 4.1], index=pd.to_datetime(["2025-01-01", "2025-02-01"])))
    store = DatasetStore()
    out = call(build_tools(store), "fetch_fred_series", series_id="unrate")
    assert out["frequency"] == "M"
    assert store.get("ds1").meta["series_id"] == "UNRATE"
