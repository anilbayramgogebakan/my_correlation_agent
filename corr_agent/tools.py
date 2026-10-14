"""LangChain tools exposed to the agent.

`build_tools(store)` returns tools bound (via closures) to one session's DatasetStore. Fetch tools
store the series and return only a dataset_id plus summary stats; the analysis tool reads by id.
Every tool returns JSON and never raises: errors come back as {"error": ...} so the model can retry.
"""

from __future__ import annotations

import functools
import json
from datetime import date, timedelta
from typing import Any, Callable, Literal

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from corr_agent import config
from corr_agent.analysis import analyze
from corr_agent.clients import fred, open_meteo, wikimedia
from corr_agent.data_store import DatasetStore

DEFAULT_WINDOW_DAYS = 365


def emit(message: str, **data: Any) -> None:
    """Send a progress line to the Execution Log (no-op outside a LangGraph run)."""
    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer({"message": message, **data})


def _parse_date(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        raise ValueError(f"{name} must be in YYYY-MM-DD format, got {value!r}") from None


def _window(start_date: str | None, end_date: str | None) -> tuple[date, date]:
    yesterday = date.today() - timedelta(days=1)
    end = min(_parse_date(end_date, "end_date") or yesterday, yesterday)
    start = _parse_date(start_date, "start_date") or end - timedelta(days=DEFAULT_WINDOW_DAYS)
    if start >= end:
        raise ValueError(f"start_date {start} must be before end_date {end}")
    return start, end


def _json_tool(fn: Callable[..., Any], *, name: str, args_schema: type[BaseModel]) -> BaseTool:
    @functools.wraps(fn)
    def run(**kwargs: Any) -> str:
        try:
            return json.dumps(fn(**kwargs), default=str, ensure_ascii=False)
        except Exception as e:  # surfaced to the model as data, not as a crash
            emit(f"{name} failed: {e}", level="error")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return StructuredTool.from_function(func=run, name=name, description=fn.__doc__.strip(),
                                        args_schema=args_schema)


# ---------------------------------------------------------------------------- argument schemas

_START = Field(None, description="YYYY-MM-DD. Default: 365 days before end_date.")
_END = Field(None, description="YYYY-MM-DD. Default: yesterday.")
_LOCATION = Field(description="City name, optionally with country, e.g. 'Berlin' or 'Paris, France'.")


class WeatherHistoryArgs(BaseModel):
    location: str = _LOCATION
    variable: Literal[open_meteo.WEATHER_VARIABLES] = Field(
        description="Daily weather variable. sunshine_duration is returned in hours.")
    start_date: str | None = _START
    end_date: str | None = _END


class AirQualityArgs(BaseModel):
    location: str = _LOCATION
    pollutant: Literal[open_meteo.AIR_QUALITY_VARIABLES] = Field(
        description="Air-quality or pollen variable (pollen is Europe-only); aggregated to daily means.")
    start_date: str | None = _START
    end_date: str | None = _END


class WeatherForecastArgs(BaseModel):
    location: str = _LOCATION
    variable: Literal[open_meteo.WEATHER_VARIABLES] = Field(description="Same variable as the historical series.")
    days: int = Field(7, ge=1, le=16, description="Number of forecast days (1-16).")


class WikiSearchArgs(BaseModel):
    query: str = Field(description="Topic to search for, e.g. 'ice cream'.")
    lang: str = Field("en", description="Wikipedia language code, e.g. 'en', 'de', 'fr', 'tr'.")


class PageviewsArgs(BaseModel):
    article: str = Field(description="Exact Wikipedia article title, e.g. 'Ice cream'.")
    lang: str = Field("en", description="Wikipedia language code, e.g. 'en', 'de'.")
    start_date: str | None = _START
    end_date: str | None = _END


class FredSearchArgs(BaseModel):
    query: str = Field(description="What to look for, e.g. 'unemployment rate', 'VIX', '30-year mortgage rate'.")


class FredSeriesArgs(BaseModel):
    series_id: str = Field(description="FRED series id from search_fred_series, e.g. 'UNRATE'.")
    start_date: str | None = _START
    end_date: str | None = _END


class AnalysisArgs(BaseModel):
    x_dataset_id: str = Field(description="dataset_id of the explanatory variable X (likely driver), e.g. 'ds1'.")
    y_dataset_id: str = Field(description="dataset_id of the outcome variable Y, e.g. 'ds2'.")
    max_lag: int = Field(7, ge=0, le=30, description="Max lag (in aligned periods) to scan in both directions.")
    forecast_dataset_id: str | None = Field(
        None, description="Optional dataset_id from fetch_weather_forecast (same variable as X) to predict Y.")


# ---------------------------------------------------------------------------- tools

def build_tools(store: DatasetStore) -> list[BaseTool]:

    def fetch_weather_history(location: str, variable: str, start_date: str | None = None,
                              end_date: str | None = None) -> dict:
        """Fetch a daily historical weather series (Open-Meteo ERA5 archive) for a city.
        Returns a dataset_id and summary statistics."""
        start, end = _window(start_date, end_date)
        place = open_meteo.geocode(location)
        where = f"{place['name']}, {place['country']}"
        emit(f"Calling Open-Meteo archive API: {variable} in {where}, {start} → {end}", source="open-meteo")
        series, units = open_meteo.daily_weather_history(place["latitude"], place["longitude"], start, end,
                                                         variable, place["timezone"])
        ds = store.add(label=f"{variable} in {place['name']}", source="Open-Meteo", series=series, units=units,
                       meta={"location": where, "variable": variable, "kind": "history"})
        emit(f"Stored {ds.id}: {ds.label} ({len(series)} days)", source="open-meteo")
        return ds.summary()

    def fetch_air_quality(location: str, pollutant: str, start_date: str | None = None,
                          end_date: str | None = None) -> dict:
        """Fetch a daily air-quality or pollen series (Open-Meteo, hourly values averaged per day) for a city.
        Returns a dataset_id and summary statistics."""
        start, end = _window(start_date, end_date)
        place = open_meteo.geocode(location)
        where = f"{place['name']}, {place['country']}"
        emit(f"Calling Open-Meteo air-quality API: {pollutant} in {where}, {start} → {end}", source="open-meteo")
        series, units = open_meteo.daily_air_quality(place["latitude"], place["longitude"], start, end, pollutant)
        ds = store.add(label=f"{pollutant} in {place['name']}", source="Open-Meteo", series=series, units=units,
                       meta={"location": where, "variable": pollutant, "kind": "history"})
        emit(f"Stored {ds.id}: {ds.label} ({len(series)} days)", source="open-meteo")
        return ds.summary()

    def fetch_weather_forecast(location: str, variable: str, days: int = 7) -> dict:
        """Fetch a daily weather forecast (up to 16 days) for a city, to be used as forecast_dataset_id
        in run_correlation_analysis. Returns a dataset_id and summary statistics."""
        place = open_meteo.geocode(location)
        where = f"{place['name']}, {place['country']}"
        emit(f"Calling Open-Meteo forecast API: {variable} in {where}, next {days} days", source="open-meteo")
        series, units = open_meteo.daily_weather_forecast(place["latitude"], place["longitude"], variable, days,
                                                          place["timezone"])
        ds = store.add(label=f"{variable} forecast in {place['name']}", source="Open-Meteo", series=series,
                       units=units, meta={"location": where, "variable": variable, "kind": "forecast"})
        emit(f"Stored {ds.id}: {ds.label} ({len(series)} days)", source="open-meteo")
        return ds.summary()

    def search_wikipedia_articles(query: str, lang: str = "en") -> dict:
        """Search Wikipedia for exact article titles matching a topic, with a short description of each.
        The query may be in English for any language edition. Use before fetch_wikipedia_pageviews."""
        emit(f"Searching {lang}.wikipedia for '{query}'", source="wikipedia")
        return {"lang": lang, "results": wikimedia.search_articles(query, lang)}

    def fetch_wikipedia_pageviews(article: str, lang: str = "en", start_date: str | None = None,
                                  end_date: str | None = None) -> dict:
        """Fetch daily pageviews (human traffic) of a Wikipedia article, a proxy for public interest.
        Redirects are resolved automatically. Returns a dataset_id and summary statistics."""
        start, end = _window(start_date, end_date)
        title = wikimedia.resolve_title(article, lang)
        if title != article:
            emit(f"Resolved redirect '{article}' → '{title}'", source="wikipedia")
        emit(f"Calling Wikimedia Pageviews API: {lang}:{title}, {start} → {end}", source="wikipedia")
        series = wikimedia.daily_pageviews(title, start, end, lang)
        ds = store.add(label=f"Wikipedia views: {title} ({lang})", source="Wikimedia", series=series,
                       units="views/day", meta={"article": title, "lang": lang, "kind": "history"})
        emit(f"Stored {ds.id}: {ds.label} ({len(series)} days)", source="wikipedia")
        return ds.summary()

    def search_fred_series(query: str) -> dict:
        """Search FRED for US economic/financial series. Returns series ids with title, frequency and units."""
        emit(f"Searching FRED for '{query}'", source="fred")
        return {"results": fred.search_series(query)}

    def fetch_fred_series(series_id: str, start_date: str | None = None, end_date: str | None = None) -> dict:
        """Fetch observations of a FRED series by id. Returns a dataset_id and summary statistics."""
        start, end = _window(start_date, end_date)
        info = fred.series_info(series_id.strip().upper())
        emit(f"Calling FRED API: {info['series_id']} ({info['title']}), {start} → {end}", source="fred")
        series = fred.observations(info["series_id"], start, end)
        ds = store.add(label=f"{info['title']} [{info['series_id']}]", source="FRED", series=series,
                       freq=info["frequency"], units=info["units"],
                       meta={"series_id": info["series_id"], "kind": "history"})
        emit(f"Stored {ds.id}: {ds.label} ({len(series)} points, freq {ds.freq})", source="fred")
        return ds.summary()

    def run_correlation_analysis(x_dataset_id: str, y_dataset_id: str, max_lag: int = 7,
                                 forecast_dataset_id: str | None = None) -> dict:
        """Align two stored series (resampling to the coarser frequency) and compute Pearson and Spearman
        correlation with p-values, a lag scan, linear-regression and random-forest models (out-of-sample R²),
        and optionally a forecast of Y from forecast X values."""
        x, y = store.get(x_dataset_id), store.get(y_dataset_id)
        forecast_x = None
        if forecast_dataset_id:
            fc = store.get(forecast_dataset_id)
            if fc.meta.get("kind") != "forecast" or fc.meta.get("variable") != x.meta.get("variable"):
                raise ValueError("forecast_dataset_id must come from fetch_weather_forecast with the same "
                                 "variable as X.")
            forecast_x = fc.series
        emit(f"Aligning {x.id} and {y.id} on a common time axis", source="analysis")
        emit("Computing Pearson & Spearman correlation and lag scan", source="analysis")
        emit("Training LinearRegression and RandomForest (TimeSeriesSplit cross-validation)", source="analysis")
        result = analyze(x.series, y.series, x_freq=x.freq, y_freq=y.freq, x_label=x.label, y_label=y.label,
                         x_units=x.units, y_units=y.units, max_lag=max_lag, forecast_x=forecast_x)
        store.analyses.append(result)
        emit(f"Done: r = {result.pearson_r:.2f}, ρ = {result.spearman_rho:.2f}, n = {result.n}", source="analysis")
        return result.to_llm_dict()

    specs: list[tuple[Callable[..., Any], type[BaseModel]]] = [
        (fetch_weather_history, WeatherHistoryArgs),
        (fetch_air_quality, AirQualityArgs),
        (fetch_weather_forecast, WeatherForecastArgs),
        (search_wikipedia_articles, WikiSearchArgs),
        (fetch_wikipedia_pageviews, PageviewsArgs),
    ]
    if config.FRED_API_KEY:
        specs += [(search_fred_series, FredSearchArgs), (fetch_fred_series, FredSeriesArgs)]
    specs.append((run_correlation_analysis, AnalysisArgs))
    return [_json_tool(fn, name=fn.__name__, args_schema=schema) for fn, schema in specs]
