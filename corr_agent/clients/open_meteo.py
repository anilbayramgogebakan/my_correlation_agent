"""Open-Meteo: geocoding, historical weather (ERA5 archive), air quality/pollen and forecast. No API key."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd

from corr_agent.clients.http import APIError, get_json

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Daily variables supported by both the archive and the forecast API.
WEATHER_VARIABLES = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
    "shortwave_radiation_sum",
)

# Hourly air-quality variables; aggregated to daily means. Pollen is Europe-only.
AIR_QUALITY_VARIABLES = (
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
    "us_aqi",
    "european_aqi",
    "uv_index",
    "grass_pollen",
    "birch_pollen",
    "alder_pollen",
    "ragweed_pollen",
)


@lru_cache(maxsize=256)
def geocode(name: str) -> dict[str, Any]:
    """Resolve a place name to coordinates. Raises APIError if nothing matches."""
    data = get_json(GEOCODE_URL, {"name": name.split(",")[0].strip(), "count": 5, "language": "en"})
    results = data.get("results") or []
    if not results:
        raise APIError(f"Location {name!r} not found. Try a city name like 'Berlin' or 'New York'.")
    # If the caller added a country hint ("Paris, US"), prefer a matching result.
    hint = name.split(",")[1].strip().lower() if "," in name else ""
    best = next((r for r in results if hint and hint in (r.get("country", "") + r.get("country_code", "")).lower()),
                results[0])
    return {
        "name": best["name"],
        "country": best.get("country", ""),
        "latitude": best["latitude"],
        "longitude": best["longitude"],
        "timezone": best.get("timezone", "auto"),
    }


def _to_series(block: dict[str, Any], variable: str) -> pd.Series:
    values = block.get(variable)
    if values is None:
        raise APIError(f"Open-Meteo returned no data for {variable!r}")
    s = pd.Series(values, index=pd.to_datetime(block["time"]), dtype="float64")
    return s.dropna()


def _convert_units(series: pd.Series, units: str) -> tuple[pd.Series, str]:
    # sunshine_duration comes in seconds; hours are easier to read.
    if units == "s":
        return series / 3600.0, "h"
    return series, units


def daily_weather_history(lat: float, lon: float, start: date, end: date, variable: str,
                          timezone: str = "auto") -> tuple[pd.Series, str]:
    data = get_json(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": variable, "timezone": timezone,
    })
    series = _to_series(data["daily"], variable)
    return _convert_units(series, data.get("daily_units", {}).get(variable, ""))


def daily_air_quality(lat: float, lon: float, start: date, end: date, variable: str) -> tuple[pd.Series, str]:
    data = get_json(AIR_QUALITY_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": variable,
    })
    hourly = _to_series(data["hourly"], variable)
    if hourly.empty:
        raise APIError(f"No {variable} data for this location/period (pollen is only available in Europe).")
    daily = hourly.resample("D").mean().dropna()
    return daily, data.get("hourly_units", {}).get(variable, "")


def daily_weather_forecast(lat: float, lon: float, variable: str, days: int,
                           timezone: str = "auto") -> tuple[pd.Series, str]:
    data = get_json(FORECAST_URL, {
        "latitude": lat, "longitude": lon,
        "daily": variable, "forecast_days": days, "timezone": timezone,
    })
    series = _to_series(data["daily"], variable)
    return _convert_units(series, data.get("daily_units", {}).get(variable, ""))
