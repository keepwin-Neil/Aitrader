"""Historical market universe resolver (data-source audit A2 P1).

Context: all current scans / optimizations / backtests use the *current*
snapshot of `market_universes/<MARKET>.json` — which is survivorship-biased
for historical studies. This module provides the plumbing to load a
historical snapshot when given a signal/backtest date.

Directory layout:
    market_universes_history/
        US_2024Q1.json
        US_2024Q2.json
        ...
        US_2026Q2.json      ← latest known snapshot (seeded 2026-04-22)

Resolution rules:
    load_universe_at("US", "2025-03-15") → loads the *most recent* snapshot
        whose date ≤ 2025-03-15. If none exists, falls back to current
        `market_universes/US.json`.

This is P1 MVP. For P2 (true survivorship-bias-free history from inception),
we need either:
    - git archaeology of `market_universes/US.json` history → synthesized snapshots
    - Third-party source (Norgate, Polygon, CRSP) with point-in-time index
      constituents

Neither is in scope for this module. Callers should treat pre-P2 history as
"best effort" and prefer the oldest snapshot rather than assuming completeness.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import re as _re
import threading as _threading
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Set

_PROJECT_ROOT = _Path(__file__).parent
_HISTORY_DIR = _PROJECT_ROOT / "market_universes_history"
_CURRENT_DIR = _PROJECT_ROOT / "market_universes"
_LOCK = _threading.RLock()
_CACHE: Dict[str, List[Dict[str, Any]]] = {}  # market → list of {date, path, data}
_FNAME_RE = _re.compile(
    r"^(?P<market>[A-Z]+)_(?P<year>\d{4})(?:Q(?P<q>[1-4])|M(?P<m>0[1-9]|1[0-2])|-(?P<ymd>\d{2}-\d{2}))?\.json$"
)


def _quarter_to_date(year: int, q: int) -> str:
    """End-of-quarter date (YYYY-MM-DD). Q1=Mar 31, Q2=Jun 30, ..."""
    end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[q]
    return f"{year:04d}-{end[0]:02d}-{end[1]:02d}"


def _parse_filename(fname: str) -> Optional[Dict[str, Any]]:
    """Return {market, date} or None if not a valid snapshot filename."""
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    market = m.group("market")
    year = int(m.group("year"))
    if m.group("q"):
        date = _quarter_to_date(year, int(m.group("q")))
    elif m.group("m"):
        mn = int(m.group("m"))
        last_day = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[mn]
        # leap year — good enough for our use case
        if mn == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            last_day = 29
        date = f"{year:04d}-{mn:02d}-{last_day:02d}"
    elif m.group("ymd"):
        mm, dd = m.group("ymd").split("-")
        date = f"{year:04d}-{mm}-{dd}"
    else:
        # naked year → Dec 31
        date = f"{year:04d}-12-31"
    return {"market": market, "date": date}


def _index_history() -> Dict[str, List[Dict[str, Any]]]:
    """Scan history dir; build per-market list sorted ASCENDING by date."""
    global _CACHE
    with _LOCK:
        if _CACHE:
            return _CACHE
        result: Dict[str, List[Dict[str, Any]]] = {}
        if not _HISTORY_DIR.exists():
            _CACHE = result
            return _CACHE
        for p in sorted(_HISTORY_DIR.glob("*.json")):
            meta = _parse_filename(p.name)
            if not meta:
                continue
            result.setdefault(meta["market"], []).append(
                {"date": meta["date"], "path": p}
            )
        for m in result:
            result[m].sort(key=lambda x: x["date"])
        _CACHE = result
        return _CACHE


def _load_snapshot(path: _Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _extract_symbols(data: Any) -> List[str]:
    """Try common shapes for universe files."""
    if isinstance(data, list):
        out: List[str] = []
        for row in data:
            if isinstance(row, str):
                out.append(row.upper())
            elif isinstance(row, dict):
                sym = row.get("symbol") or row.get("ticker") or row.get("code")
                if sym:
                    out.append(str(sym).upper())
        return out
    if isinstance(data, dict):
        # e.g. {"symbols": [...]}  or {"constituents": [...]}
        for k in ("symbols", "constituents", "tickers", "members"):
            if k in data:
                return _extract_symbols(data[k])
    return []


def find_snapshot_at(market: str, date: str) -> Optional[Dict[str, Any]]:
    """Most recent snapshot whose date ≤ target. Returns {date, path} or None."""
    mkt = (market or "US").upper()
    hist = _index_history().get(mkt, [])
    if not hist:
        return None
    target = (date or "")[:10]
    if not target:
        return hist[-1]
    best: Optional[Dict[str, Any]] = None
    for entry in hist:
        if entry["date"] <= target:
            best = entry
        else:
            break
    return best


def load_universe_at(market: str, date: Optional[str] = None) -> List[str]:
    """Return symbol list for (market, date).

    Preference order:
      1. Latest history snapshot with date ≤ target (survivorship-correct-ish)
      2. Latest history snapshot regardless (if target is before first snap)
      3. Current `market_universes/<MARKET>.json` (fallback, survivor-biased)
      4. Empty list
    """
    mkt = (market or "US").upper()
    snap = find_snapshot_at(mkt, date)
    if snap is None:
        hist = _index_history().get(mkt, [])
        if hist:
            snap = hist[0]  # pre-history target → best guess is oldest snapshot
    if snap is not None:
        data = _load_snapshot(snap["path"])
        syms = _extract_symbols(data)
        if syms:
            return syms
    # 3. fallback to current
    cur = _CURRENT_DIR / f"{mkt}.json"
    if cur.exists():
        data = _load_snapshot(cur)
        return _extract_symbols(data)
    return []


def available_snapshots(market: Optional[str] = None) -> Dict[str, List[str]]:
    idx = _index_history()
    if market:
        m = market.upper()
        return {m: [e["date"] for e in idx.get(m, [])]}
    return {m: [e["date"] for e in lst] for m, lst in idx.items()}


def reload() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = {}
