"""Stock symbol → human-readable name lookup, with disk cache.

CN/HK only. US tickers usually convey the brand directly (AAPL, NVDA…), but
6-digit Shenzhen/Shanghai codes and 4-digit HK codes need the Chinese name to
be readable in the UI.

Cache strategy:
- One JSON file per market under .cache/symbol_names/{MARKET}.json
- File has {"updated_at": ISO, "names": {symbol: name}}
- Lazy refresh: if file is missing or > REFRESH_DAYS old, repopulate via
  akshare on first lookup. Subsequent calls hit the in-memory map.

Public API:
  get_name(symbol, market) -> str            single lookup, '' if unknown
  get_names_bulk(symbols, market) -> dict    bulk lookup, {sym: name}
  refresh_market(market) -> int              force refresh, returns count
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / ".cache" / "symbol_names"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REFRESH_DAYS = 7

_lock = threading.Lock()
_in_memory: dict[str, dict[str, str]] = {}  # market → {symbol: name}
_loaded_at: dict[str, float] = {}             # market → epoch seconds


def _cache_path(market: str) -> Path:
    return CACHE_DIR / f"{market.upper()}.json"


def _read_cache(market: str) -> Optional[dict]:
    fp = _cache_path(market)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"stock_names cache read failed {market}: {e}")
        return None


def _write_cache(market: str, names: dict[str, str]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "names": names,
    }
    tmp = _cache_path(market).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(_cache_path(market))


def _is_stale(payload: dict) -> bool:
    ts = payload.get("updated_at")
    if not ts:
        return True
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt) > timedelta(days=REFRESH_DAYS)
    except Exception:
        return True


def _fetch_cn() -> dict[str, str]:
    """Fetch CN A-share name map via akshare (~5500 entries, ~12s)."""
    import akshare as ak
    df = ak.stock_info_a_code_name()
    return {str(row["code"]).zfill(6): str(row["name"]).strip()
            for _, row in df.iterrows() if row.get("code") and row.get("name")}


def _fetch_hk() -> dict[str, str]:
    """Fetch HK name map. Tries ``stock_hk_spot_em`` first (compact, fast)."""
    import akshare as ak
    try:
        df = ak.stock_hk_spot_em()
    except Exception as e:
        logger.warning(f"stock_hk_spot_em failed: {e}")
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row.get("代码") or row.get("symbol") or "").strip()
        name = str(row.get("名称") or row.get("name") or "").strip()
        if not code or not name:
            continue
        out[code.zfill(5)] = name  # HK codes are 5 digits with leading zero
    return out


def _ensure_loaded(market: str) -> dict[str, str]:
    """Return the in-memory map for the market, lazy-loading from cache or
    upstream akshare if needed. Thread-safe."""
    market = market.upper()
    with _lock:
        if market in _in_memory:
            return _in_memory[market]
        cached = _read_cache(market)
        if cached and not _is_stale(cached):
            names = cached.get("names") or {}
            _in_memory[market] = names
            _loaded_at[market] = time.time()
            return names
        # Fetch fresh
        try:
            if market == "CN":
                names = _fetch_cn()
            elif market == "HK":
                names = _fetch_hk()
            else:
                names = {}
        except Exception as e:
            logger.warning(f"stock_names fetch failed {market}: {e}")
            # Use stale cache if available, else empty
            names = (cached.get("names") if cached else {}) or {}
        _in_memory[market] = names
        _loaded_at[market] = time.time()
        if names:
            _write_cache(market, names)
        return names


def get_name(symbol: str, market: str) -> str:
    """Return the human-readable name for (symbol, market), or '' if unknown.
    Only looks up CN/HK; US/CRYPTO/UK return empty (no display name needed)."""
    if not symbol:
        return ""
    market = market.upper()
    if market not in ("CN", "HK"):
        return ""
    sym = symbol.strip()
    if market == "CN":
        sym = sym.zfill(6)
    elif market == "HK":
        sym = sym.zfill(5)
    return _ensure_loaded(market).get(sym, "")


def get_names_bulk(symbols, market: str) -> dict[str, str]:
    """Return {symbol: name} for the given symbols. Symbols not found are absent."""
    if not symbols:
        return {}
    market = market.upper()
    if market not in ("CN", "HK"):
        return {}
    table = _ensure_loaded(market)
    out: dict[str, str] = {}
    for s in symbols:
        if not s:
            continue
        key = s.strip()
        lookup = key.zfill(6) if market == "CN" else key.zfill(5)
        name = table.get(lookup) or table.get(key)
        if name:
            out[s] = name
    return out


def refresh_market(market: str) -> int:
    """Force re-fetch from upstream. Returns the new entry count."""
    market = market.upper()
    with _lock:
        _in_memory.pop(market, None)
    table = _ensure_loaded(market)
    return len(table)
