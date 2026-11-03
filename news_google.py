"""Google News RSS adapter (no API key, no extra dependency).

Uses stdlib xml.etree.ElementTree to parse the RSS feed.
"""
from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

logger = logging.getLogger(__name__)

_BASE = "https://news.google.com/rss/search"
_TIMEOUT = 10


def _parse_pub_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def fetch(symbol: str, window_hours: int = 24, lang: str = "en-US",
          country: str = "US", company_hint: str | None = None) -> list[dict]:
    """Fetch Google News RSS for `symbol` within last `window_hours`.

    Returns list of dicts: {title, url, published_at (ISO), source, summary}.
    `company_hint` improves recall for ambiguous tickers (e.g. 'AAPL Apple').
    """
    query_parts = [symbol.upper(), "stock"]
    if company_hint:
        query_parts.append(company_hint)
    query = " ".join(query_parts)

    params = {
        "q": query,
        "hl": lang,
        "gl": country,
        "ceid": f"{country}:{lang.split('-')[0]}",
    }
    url = f"{_BASE}?{urllib.parse.urlencode(params)}"

    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Google News fetch failed for %s: %s", symbol, e)
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        logger.warning("Google News RSS parse failed for %s: %s", symbol, e)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    out: list[dict] = []
    for item in root.iterfind("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_str = item.findtext("pubDate") or ""
        pub_dt = _parse_pub_date(pub_date_str)
        if pub_dt is None or pub_dt < cutoff:
            continue
        # source is in <source> child element
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None and src_el.text else "Google News"
        description = (item.findtext("description") or "").strip()
        out.append({
            "title": title,
            "url": link,
            "published_at": pub_dt.isoformat(),
            "source": source,
            "summary": description[:500],  # description can be HTML-heavy; truncate
        })
    return out
