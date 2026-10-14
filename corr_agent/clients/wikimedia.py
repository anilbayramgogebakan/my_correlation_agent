"""Wikipedia article search and Wikimedia per-article daily pageviews. No API key."""

from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import quote

import pandas as pd

from corr_agent.clients.http import APIError, get_json

PAGEVIEWS_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                 "{project}/all-access/user/{article}/daily/{start}/{end}")
PAGEVIEWS_EARLIEST = date(2015, 7, 1)


def _api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _full_text_search(query: str, lang: str, limit: int, **extra: Any) -> list[dict[str, Any]]:
    """Relevance-ranked full-text search; each page comes with its short description."""
    data = get_json(_api_url(lang), {
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
        "gsrnamespace": 0, "prop": "description", "format": "json", "formatversion": 2, **extra,
    })
    return sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))


def search_articles(query: str, lang: str = "en", limit: int = 5) -> list[dict[str, Any]]:
    """Find article titles for a topic, with descriptions so the right one can be told apart.

    Title-prefix search is a trap here: "skiing" on de.wikipedia returns a rapper called "Ski Aggu".
    Full-text relevance search is used instead, and for non-English editions the best English match
    is followed through its interlanguage link (en "Skiing" → de "Skifahren"), since the query is
    often written in English.
    """
    results: list[dict[str, Any]] = []
    if lang != "en":
        top_en = _full_text_search(query, "en", 1, prop="description|langlinks", lllang=lang)
        for page in top_en:
            for link in page.get("langlinks", []):
                results.append({"title": link["title"],
                                "description": f"{lang} version of English article '{page['title']}'"
                                               + (f" ({page['description']})" if page.get("description") else "")})
    for page in _full_text_search(query, lang, limit):
        if all(r["title"] != page["title"] for r in results):
            results.append({"title": page["title"], "description": page.get("description", "")})
    return results[:limit]


def resolve_title(article: str, lang: str = "en") -> str:
    """Follow redirects so pageviews are counted on the canonical article.

    Pageviews are recorded per exact title, so a redirect like de:"Heuschnupfen"
    shows almost no traffic while its target article holds the real numbers.
    """
    data = get_json(_api_url(lang), {
        "action": "query", "titles": article, "redirects": 1,
        "format": "json", "formatversion": 2,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise APIError(f"Wikipedia article {article!r} does not exist on {lang}.wikipedia. "
                       "Use search_wikipedia_articles to find the exact title.")
    return pages[0]["title"]


def daily_pageviews(article: str, start: date, end: date, lang: str = "en") -> pd.Series:
    start = max(start, PAGEVIEWS_EARLIEST)
    url = PAGEVIEWS_URL.format(
        project=f"{lang}.wikipedia",
        article=quote(article.replace(" ", "_"), safe=""),
        start=start.strftime("%Y%m%d"),
        end=end.strftime("%Y%m%d"),
    )
    try:
        data = get_json(url)
    except APIError as e:
        if "HTTP 404" in str(e):
            raise APIError(f"No pageview data for {article!r} on {lang}.wikipedia in this period.") from e
        raise
    items = data.get("items", [])
    if not items:
        raise APIError(f"No pageview data for {article!r} on {lang}.wikipedia in this period.")
    index = pd.to_datetime([it["timestamp"][:8] for it in items], format="%Y%m%d")
    series = pd.Series([it["views"] for it in items], index=index, dtype="float64")
    # The API omits days with zero views; make them explicit.
    return series.reindex(pd.date_range(start, end, freq="D"), fill_value=0.0).loc[: index.max()]
