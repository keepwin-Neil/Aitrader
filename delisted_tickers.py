"""Delisted ticker registry (data-source audit B3).

File-backed registry of tickers that stopped producing new bars for >threshold
days — typically because they got delisted, acquired, bankrupt, or renamed.

Once a ticker is here:
- `update_stock_data`        stops trying to fetch it (saves rate limit quota)
- `mark_to_market` in core_engine marks affected positions to market_value=0
  instead of silently forward-filling the last pre-delist price

Registry file: ``delisted_tickers.json`` at project root.
Shape:
    {
        "US:XYZ": {
            "delisted_at":   "2024-12-01",            # last bar seen
            "last_bar_date": "2024-12-01",            # same; kept for symmetry
            "detected_at":   "2025-01-15T03:30:00Z",  # when we first flagged
            "reason":        "no new bars for 45d"
        },
        ...
    }

Key convention:  UPPER(market) + ":" + UPPER(symbol).
Threadsafe via module-level RLock.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import threading as _threading
from pathlib import Path as _Path
from typing import Any, Dict, Optional

_PATH = _Path(__file__).parent / "delisted_tickers.json"
_LOCK = _threading.RLock()
_CACHE: Optional[Dict[str, Any]] = None


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _key(symbol: str, market: str) -> str:
    return f"{(market or '').upper()}:{(symbol or '').upper()}"


def _load() -> Dict[str, Any]:
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        if _PATH.exists():
            try:
                with open(_PATH, "r", encoding="utf-8") as f:
                    _CACHE = _json.load(f) or {}
                if not isinstance(_CACHE, dict):
                    _CACHE = {}
            except Exception:
                _CACHE = {}
        else:
            _CACHE = {}
        return _CACHE


def _save(data: Dict[str, Any]) -> None:
    """Atomic write via tmp + os.replace (survives crashes mid-write)."""
    with _LOCK:
        tmp = _PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        _os.replace(tmp, _PATH)
        global _CACHE
        _CACHE = data


def is_delisted(symbol: str, market: str) -> bool:
    return _key(symbol, market) in _load()


def get_meta(symbol: str, market: str) -> Optional[Dict[str, Any]]:
    return _load().get(_key(symbol, market))


def mark_delisted(
    symbol: str,
    market: str,
    last_bar_date: str,
    reason: str = "no new bars > threshold",
) -> None:
    with _LOCK:
        d = dict(_load())
        d[_key(symbol, market)] = {
            "delisted_at": last_bar_date,
            "last_bar_date": last_bar_date,
            "detected_at": _utc_now_iso(),
            "reason": reason,
        }
        _save(d)


def unmark_delisted(symbol: str, market: str) -> bool:
    """Return True if ticker was previously delisted (and is now removed)."""
    with _LOCK:
        d = dict(_load())
        k = _key(symbol, market)
        if k in d:
            d.pop(k)
            _save(d)
            return True
        return False


def list_delisted() -> Dict[str, Any]:
    return dict(_load())


def reload() -> None:
    """Force reload from disk (for tests)."""
    global _CACHE
    with _LOCK:
        _CACHE = None
        _load()
