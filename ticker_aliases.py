"""Ticker alias resolver (data-source audit B1).

Some historical tickers (FB, TWTR, ATVI, XLNX, ...) either got renamed or the
company got acquired. Without aliasing, a pool referencing the old symbol will:
  - fetch partial/empty history (yfinance partially redirects but inconsistently)
  - silently drop the acquired company from performance calculations
  - break trade_fills → positions reconciliation across the rename date

This module provides a simple forward mapping:
    resolve(symbol, market) -> canonical_symbol

If `canonical` is null (taken private, no successor), resolve() returns the
original symbol; caller should handle as potentially delisted.

Registry file: ticker_aliases.json at project root.
Threadsafe cached load.
"""
from __future__ import annotations

import json as _json
import threading as _threading
from pathlib import Path as _Path
from typing import Any, Dict, Optional

_PATH = _Path(__file__).parent / "ticker_aliases.json"
_LOCK = _threading.RLock()
_CACHE: Optional[Dict[str, Any]] = None


def _load() -> Dict[str, Any]:
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        if _PATH.exists():
            try:
                with open(_PATH, "r", encoding="utf-8") as f:
                    data = _json.load(f) or {}
                _CACHE = data.get("aliases") or {}
                if not isinstance(_CACHE, dict):
                    _CACHE = {}
            except Exception:
                _CACHE = {}
        else:
            _CACHE = {}
        return _CACHE


def resolve(symbol: str, market: str = "US") -> str:
    """Return canonical symbol if aliased, else the original.

    If alias.canonical is null, returns original (caller treats as delisted).
    """
    if not symbol:
        return symbol
    sym = symbol.upper()
    mkt = (market or "US").upper()
    aliases = _load().get(mkt) or {}
    entry = aliases.get(sym)
    if not isinstance(entry, dict):
        return symbol
    canon = entry.get("canonical")
    if not canon:
        return symbol  # null canonical → no successor; caller handles
    return canon


def get_alias_entry(symbol: str, market: str = "US") -> Optional[Dict[str, Any]]:
    """Return full alias metadata (canonical, effective_date, note) or None."""
    if not symbol:
        return None
    aliases = _load().get((market or "US").upper()) or {}
    entry = aliases.get(symbol.upper())
    return entry if isinstance(entry, dict) else None


def list_aliases(market: str = "US") -> Dict[str, Dict[str, Any]]:
    """All aliases for a market."""
    aliases = _load().get((market or "US").upper()) or {}
    return {k: v for k, v in aliases.items() if isinstance(v, dict)}


def reload() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None
        _load()
