"""Finnhub company-news adapter.

Free tier: 60 calls/min. Requires FINNHUB_API_KEY in env.
Docs: https://finnhub.io/docs/api/company-news
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

# Defensive: ensure .env is loaded for standalone-script callers.
# In the main.py path it is already loaded — this is a no-op then.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1/company-news"
_TIMEOUT = 10


def fetch(symbol: str, window_hours: int = 24, api_key: Optional[str] = None) -> list[dict]:
    """Fetch company news for `symbol` from Finnhub within last `window_hours`.

    Returns list of dicts: {title, url, published_at (ISO), source, summary}.
    Empty list on missing key, HTTP error, or non-US market (Finnhub free tier
    only covers US tickers reliably).
    """
    key = api_key or os.getenv("FINNHUB_API_KEY")
    if not key:
        logger.debug("FINNHUB_API_KEY missing; skipping Finnhub fetch for %s", symbol)
        return []

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=window_hours)
    params = {
        "symbol": symbol.upper(),
        "from": since.strftime("%Y-%m-%d"),
        "to": now.strftime("%Y-%m-%d"),
        "token": key,
    }
    try:
        resp = requests.get(_BASE, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        rows = resp.json() or []
    except requests.exceptions.RequestException as e:
        logger.warning("Finnhub fetch failed for %s: %s", symbol, e)
        return []
    except ValueError:
        logger.warning("Finnhub returned non-JSON for %s", symbol)
        return []

    cutoff_ts = since.timestamp()
    out: list[dict] = []
    for row in rows:
        ts = row.get("datetime") or 0
        if ts < cutoff_ts:
            continue
        published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        out.append({
            "title": (row.get("headline") or "").strip(),
            "url": row.get("url") or "",
            "published_at": published,
            "source": row.get("source") or "finnhub",
            "summary": (row.get("summary") or "").strip(),
        })
    return out
