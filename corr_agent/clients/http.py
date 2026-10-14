"""Shared HTTP session: descriptive User-Agent, retries with backoff and a small TTL cache."""

from __future__ import annotations

import json
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from corr_agent import config

_session: requests.Session | None = None
_cache: dict[str, tuple[float, Any]] = {}


class APIError(RuntimeError):
    """Raised when an upstream API returns an error the agent should see."""


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = config.USER_AGENT
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        _session.mount("https://", HTTPAdapter(max_retries=retry))
    return _session


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET a JSON document; identical requests within HTTP_CACHE_TTL are served from memory."""
    key = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < config.HTTP_CACHE_TTL:
        return hit[1]
    try:
        resp = _get_session().get(url, params=params, timeout=config.HTTP_TIMEOUT)
    except requests.RequestException as e:
        raise APIError(f"Request to {url} failed: {e}") from e
    if resp.status_code >= 400:
        raise APIError(f"{url} returned HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    _cache[key] = (time.monotonic(), data)
    return data


def clear_cache() -> None:
    _cache.clear()
