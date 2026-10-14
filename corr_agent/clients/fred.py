"""FRED (Federal Reserve Economic Data): series search and observations. Requires a free API key."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from corr_agent import config
from corr_agent.clients.http import APIError, get_json

BASE_URL = "https://api.stlouisfed.org/fred"


def _get(endpoint: str, **params: Any) -> dict[str, Any]:
    if not config.FRED_API_KEY:
        raise APIError("FRED_API_KEY is not configured.")
    try:
        return get_json(f"{BASE_URL}/{endpoint}", {**params, "api_key": config.FRED_API_KEY, "file_type": "json"})
    except APIError as e:
        # Never echo the key back through error messages.
        raise APIError(str(e).replace(config.FRED_API_KEY, "***")) from None


def normalize_frequency(code: str) -> str:
    """Map FRED frequency codes (D, W, WEF, BW, M, Q, SA, A, ...) to D/B/W/M/Q/A."""
    code = (code or "").upper()
    if code.startswith("D"):
        return "B"  # FRED daily series are mostly business days (markets, rates)
    if code.startswith(("W", "BW")):
        return "W"
    if code.startswith("M"):
        return "M"
    if code.startswith("Q"):
        return "Q"
    if code.startswith(("A", "SA")):
        return "A"
    return "M"


def search_series(query: str, limit: int = 8) -> list[dict[str, Any]]:
    data = _get("series/search", search_text=query, limit=limit,
                order_by="popularity", sort_order="desc")
    return [{
        "series_id": s["id"],
        "title": s["title"],
        "frequency": s.get("frequency_short", ""),
        "units": s.get("units_short", ""),
        "start": s.get("observation_start"),
        "end": s.get("observation_end"),
    } for s in data.get("seriess", [])]


def series_info(series_id: str) -> dict[str, Any]:
    seriess = _get("series", series_id=series_id).get("seriess", [])
    if not seriess:
        raise APIError(f"FRED series {series_id!r} not found. Use search_fred_series first.")
    s = seriess[0]
    return {
        "series_id": s["id"],
        "title": s["title"],
        "frequency": normalize_frequency(s.get("frequency_short", "")),
        "units": s.get("units_short", ""),
    }


def observations(series_id: str, start: date, end: date) -> pd.Series:
    data = _get("series/observations", series_id=series_id,
                observation_start=start.isoformat(), observation_end=end.isoformat())
    obs = data.get("observations", [])
    # Missing values are encoded as ".".
    series = pd.Series(
        pd.to_numeric([o["value"] for o in obs], errors="coerce"),
        index=pd.to_datetime([o["date"] for o in obs]),
        dtype="float64",
    ).dropna()
    if series.empty:
        raise APIError(f"FRED series {series_id!r} has no observations between {start} and {end}.")
    return series
