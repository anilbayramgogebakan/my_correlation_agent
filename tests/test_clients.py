import re
from datetime import date

import pytest
import responses

from corr_agent.clients import fred, open_meteo, wikimedia
from corr_agent.clients.http import APIError


@responses.activate
def test_geocode_prefers_country_hint():
    responses.get(open_meteo.GEOCODE_URL, json={"results": [
        {"name": "Paris", "country": "France", "country_code": "FR", "latitude": 48.85, "longitude": 2.35,
         "timezone": "Europe/Paris"},
        {"name": "Paris", "country": "United States", "country_code": "US", "latitude": 33.66,
         "longitude": -95.55, "timezone": "America/Chicago"},
    ]})
    assert open_meteo.geocode("Paris, US")["country"] == "United States"


@responses.activate
def test_geocode_not_found():
    responses.get(open_meteo.GEOCODE_URL, json={})
    with pytest.raises(APIError, match="not found"):
        open_meteo.geocode("Atlantis")


@responses.activate
def test_weather_history_converts_sunshine_to_hours():
    responses.get(open_meteo.ARCHIVE_URL, json={
        "daily_units": {"time": "iso8601", "sunshine_duration": "s"},
        "daily": {"time": ["2025-01-01", "2025-01-02", "2025-01-03"], "sunshine_duration": [3600, None, 7200]},
    })
    s, units = open_meteo.daily_weather_history(1, 2, date(2025, 1, 1), date(2025, 1, 3), "sunshine_duration")
    assert units == "h"
    assert s.tolist() == [1.0, 2.0]


@responses.activate
def test_air_quality_hourly_to_daily_mean():
    responses.get(open_meteo.AIR_QUALITY_URL, json={
        "hourly_units": {"pm2_5": "μg/m³"},
        "hourly": {"time": ["2025-01-01T00:00", "2025-01-01T12:00", "2025-01-02T00:00"], "pm2_5": [10, 20, 5]},
    })
    s, units = open_meteo.daily_air_quality(1, 2, date(2025, 1, 1), date(2025, 1, 2), "pm2_5")
    assert s.tolist() == [15.0, 5.0]
    assert units == "μg/m³"


@responses.activate
def test_pageviews_fills_missing_days_with_zero():
    responses.get(re.compile(r"https://wikimedia\.org/api/rest_v1/metrics/pageviews/.*Ice_cream.*"), json={"items": [
        {"timestamp": "2025010100", "views": 100},
        {"timestamp": "2025010300", "views": 300},
    ]})
    s = wikimedia.daily_pageviews("Ice cream", date(2025, 1, 1), date(2025, 1, 3))
    assert s.tolist() == [100.0, 0.0, 300.0]


@responses.activate
def test_resolve_title_follows_redirect():
    responses.get("https://de.wikipedia.org/w/api.php", json={
        "query": {"redirects": [{"from": "Heuschnupfen", "to": "Allergische Rhinitis"}],
                  "pages": [{"title": "Allergische Rhinitis", "pageid": 1}]}})
    assert wikimedia.resolve_title("Heuschnupfen", "de") == "Allergische Rhinitis"


@responses.activate
def test_resolve_title_missing():
    responses.get("https://en.wikipedia.org/w/api.php",
                  json={"query": {"pages": [{"title": "Xyzzy", "missing": True}]}})
    with pytest.raises(APIError, match="does not exist"):
        wikimedia.resolve_title("Xyzzy")


@responses.activate
def test_fred_observations_drop_missing_values():
    responses.get(f"{fred.BASE_URL}/series/observations", json={"observations": [
        {"date": "2025-01-01", "value": "1.5"},
        {"date": "2025-01-02", "value": "."},
        {"date": "2025-01-03", "value": "2.5"},
    ]})
    s = fred.observations("VIXCLS", date(2025, 1, 1), date(2025, 1, 3))
    assert s.tolist() == [1.5, 2.5]


@responses.activate
def test_fred_error_hides_api_key():
    responses.get(f"{fred.BASE_URL}/series", status=400, body="bad api_key=test-key")
    with pytest.raises(APIError) as exc:
        fred.series_info("X")
    assert "test-key" not in str(exc.value)


@pytest.mark.parametrize("code,expected", [("D", "B"), ("W", "W"), ("WEF", "W"), ("BW", "W"),
                                           ("M", "M"), ("Q", "Q"), ("A", "A"), ("SA", "A")])
def test_fred_frequency_normalization(code, expected):
    assert fred.normalize_frequency(code) == expected


@responses.activate
def test_search_articles_translates_english_query_for_other_languages():
    responses.get("https://en.wikipedia.org/w/api.php", json={"query": {"pages": [
        {"title": "Skiing", "index": 1, "description": "recreational activity",
         "langlinks": [{"lang": "de", "title": "Skifahren"}]}]}})
    responses.get("https://de.wikipedia.org/w/api.php", json={"query": {"pages": [
        {"title": "Ski Aggu", "index": 2, "description": "deutscher Rapper"},
        {"title": "Skifahren", "index": 1, "description": "Wintersport"}]}})
    results = wikimedia.search_articles("skiing", lang="de")
    assert [r["title"] for r in results] == ["Skifahren", "Ski Aggu"]
    assert "English article 'Skiing'" in results[0]["description"]
