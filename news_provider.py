"""Unified news provider — orchestrates Google News + Finnhub adapters.

Public API:
    get_stock_news_unified(symbol, market, window_hours=24, ...) -> list[dict]

Each item: {title, url, published_at (ISO UTC), source, summary, provider}.
Items are deduplicated by URL and sorted newest-first.

Failure semantics: if one adapter fails, the other is still returned
(degraded mode — never raises to caller).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from news_finnhub import fetch as finnhub_fetch
from news_google import fetch as google_fetch

logger = logging.getLogger(__name__)

# Module-level cache: {(symbol, market, window_hours): (timestamp, items)}
_CACHE: dict[tuple[str, str, int], tuple[float, list[dict]]] = {}
_CACHE_TTL_SEC = 1800  # 30 min, per plan doc


def _cache_get(key: tuple) -> Optional[list[dict]]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, items = entry
    if time.time() - ts > _CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return items


def _cache_put(key: tuple, items: list[dict]) -> None:
    _CACHE[key] = (time.time(), items)


def get_stock_news_unified(
    symbol: str,
    market: str,
    window_hours: int = 24,
    company_hint: Optional[str] = None,
    use_cache: bool = True,
) -> list[dict]:
    """Fetch news for a stock from all available providers.

    Args:
        symbol: ticker (e.g. 'AAPL', '0700.HK', '600519.SS').
        market: 'US' / 'HK' / 'CN' / 'UK' — affects which providers run.
        window_hours: lookback window. Default 24h.
        company_hint: human-readable name to disambiguate (improves Google News recall).
        use_cache: read/write 30-min in-memory cache.

    Returns:
        List of news dicts, deduplicated by URL, sorted newest-first.
    """
    symbol = (symbol or "").strip().upper()
    market = (market or "US").strip().upper()
    if not symbol:
        return []

    cache_key = (symbol, market, int(window_hours))
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("news cache hit %s/%s", symbol, market)
            return cached

    items: list[dict] = []

    # Finnhub: only US tickers get reliable coverage on free tier; skip for HK/CN
    if market == "US":
        for n in finnhub_fetch(symbol, window_hours=window_hours):
            n = dict(n)
            n["provider"] = "finnhub"
            items.append(n)

    # Google News: works for all markets; fall back to symbol-only query if no hint
    google_query_symbol = symbol
    # Strip exchange suffix for cleaner Google News query (e.g. 0700.HK -> 0700)
    if "." in symbol:
        google_query_symbol = symbol.split(".")[0]
    for n in google_fetch(google_query_symbol, window_hours=window_hours,
                          company_hint=company_hint):
        n = dict(n)
        n["provider"] = "google"
        items.append(n)

    # Dedupe by URL, prefer Finnhub (richer summary) when both present
    seen: dict[str, dict] = {}
    for item in items:
        url = item.get("url") or ""
        if not url:
            continue
        if url in seen and seen[url]["provider"] == "finnhub":
            continue  # keep finnhub
        seen[url] = item

    deduped = sorted(seen.values(), key=lambda x: x.get("published_at") or "", reverse=True)

    if use_cache:
        _cache_put(cache_key, deduped)
    return deduped


def clear_cache() -> int:
    """Clear in-memory news cache. Returns number of entries cleared."""
    n = len(_CACHE)
    _CACHE.clear()
    return n
