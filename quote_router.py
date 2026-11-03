"""
Multi-source quote router with load balancing.

Sources, in priority order:
  1. IB Gateway   — paid realtime NBBO, 16-slot client pool (default)
  2. Yahoo Finance — free, ~60 req/min per IP, capped to 8 concurrent here
  3. OHLCV cache  — local parquet, stale (last close)

Load-balancing rule: each request walks the priority list and runs on the
first provider with idle capacity. So when IB pool is saturated, overflow
requests run on Yahoo Finance in parallel rather than queueing for IB.
Combined throughput ≈ 16 + 8 concurrent fetches.

Public API:
    from quote_router import router
    quote  = router.fetch_quote("AAPL", "US")
    quotes = router.fetch_quotes(["AAPL","NVDA"], "US")
    status = router.status()
"""
from __future__ import annotations

import os
import threading
import logging
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

logger = logging.getLogger("uvicorn.error")


# ─── Provider abstraction ────────────────────────────────────────────────────

class QuoteProvider:
    name: str = "abstract"
    supported_markets: set = set()

    def supports(self, market: str) -> bool:
        return str(market).upper() in self.supported_markets

    def available(self) -> int:
        """Approximate idle capacity right now (for routing + telemetry)."""
        return 1

    def try_fetch(self, symbol: str, market: str) -> Optional[dict]:
        """Non-blocking. Returns dict on success or None on
        saturation/failure. The router walks providers in priority order
        until one returns a usable result."""
        raise NotImplementedError

    def force_fetch(self, symbol: str, market: str) -> Optional[dict]:
        """May block until capacity is available. Returns dict on success
        or None on failure. The router falls back to this when every
        provider's try_fetch returned None."""
        return self.try_fetch(symbol, market)


# ─── IB Gateway provider ─────────────────────────────────────────────────────

class IBProvider(QuoteProvider):
    name = "IB Gateway"
    supported_markets = {"US", "HK"}

    def available(self) -> int:
        try:
            from ib_gateway import _ensure_pool, _pool
            if not _is_ib_enabled():
                return 0
            _ensure_pool()
            return _pool.qsize() if _pool else 0
        except Exception:
            return 0

    def try_fetch(self, symbol, market):
        if not self.supports(market) or not _is_ib_enabled():
            return None
        from ib_gateway import try_fetch_quote as _ib_try
        try:
            return _ib_try(symbol, market)
        except Exception as e:
            logger.debug(f"IB try_fetch {symbol}: {e}")
            return None

    def force_fetch(self, symbol, market):
        if not self.supports(market) or not _is_ib_enabled():
            return None
        from ib_gateway import fetch_quote as _ib_fetch
        try:
            return _ib_fetch(symbol, market)
        except Exception as e:
            logger.debug(f"IB force_fetch {symbol}: {e}")
            return None


def _is_ib_enabled() -> bool:
    try:
        from ib_gateway import ib_enabled
        return ib_enabled()
    except Exception:
        return False


# ─── Yahoo Finance provider ──────────────────────────────────────────────────

class YahooProvider(QuoteProvider):
    name = "Yahoo Finance"
    supported_markets = {"US", "HK", "CN", "UK"}

    def __init__(self, max_concurrent: int = 8):
        # Cap concurrent yfinance calls to avoid 429 cascades. Yahoo's
        # documented free-tier limit is ~60 req/min/IP; 8 concurrent leaves
        # enough headroom that brief bursts don't trip the rate limiter.
        self._max = max(1, max_concurrent)
        self._semaphore = threading.Semaphore(self._max)
        self._in_flight = 0
        self._lock = threading.Lock()

    def available(self) -> int:
        return max(0, self._max - self._in_flight)

    def try_fetch(self, symbol, market):
        if not self.supports(market):
            return None
        if not self._semaphore.acquire(blocking=False):
            return None
        with self._lock:
            self._in_flight += 1
        try:
            return self._do_fetch(symbol, market)
        finally:
            with self._lock:
                self._in_flight -= 1
            self._semaphore.release()

    def force_fetch(self, symbol, market):
        if not self.supports(market):
            return None
        self._semaphore.acquire()  # blocking
        with self._lock:
            self._in_flight += 1
        try:
            return self._do_fetch(symbol, market)
        finally:
            with self._lock:
                self._in_flight -= 1
            self._semaphore.release()

    def _do_fetch(self, symbol, market):
        import yfinance as yf
        try:
            tsym = _yf_symbol(symbol, market)
            t = yf.Ticker(tsym)
            info = t.fast_info
            price = _safe_float(getattr(info, "last_price", None))
            prev = _safe_float(getattr(info, "previous_close", None))
            if price is None:
                return None
            change_pct = None
            if price and prev:
                try:
                    change_pct = round((price - prev) / prev * 100, 2)
                except Exception:
                    pass
            return {
                "symbol": symbol.upper(),
                "ticker": tsym,
                "market": market.upper(),
                "price": price,
                "prev_close": prev,
                "change_pct": change_pct,
                "high": _safe_float(getattr(info, "day_high", None)),
                "low": _safe_float(getattr(info, "day_low", None)),
                "market_cap": _safe_float(getattr(info, "market_cap", None)),
                "currency": getattr(info, "currency", "USD"),
                "source": "Yahoo Finance",
                "ts": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.debug(f"YF fetch {symbol}: {e}")
            return None


# ─── OHLCV cache (always-available stale fallback) ──────────────────────────

class CacheProvider(QuoteProvider):
    name = "OHLCV cache"
    supported_markets = {"US", "HK", "CN", "UK", "CRYPTO"}

    def available(self) -> int:
        return 999  # local file read, never saturates

    def try_fetch(self, symbol, market):
        if not self.supports(market):
            return None
        try:
            from backtest import get_ohlcv
            df = get_ohlcv(symbol, market=market, interval="1d",
                           period="3mo", local_only=True)
            if df is None or len(df) == 0:
                return None
            last_close = _safe_float(df["close"].iloc[-1])
            prev_close = _safe_float(df["close"].iloc[-2]) if len(df) >= 2 else None
            if last_close is None:
                return None
            last_dt = str(df.index[-1])[:10]
            change_pct = None
            if last_close and prev_close:
                try:
                    change_pct = round((last_close - prev_close) / prev_close * 100, 2)
                except Exception:
                    pass
            return {
                "symbol": symbol.upper(),
                "market": market.upper(),
                "price": last_close,
                "prev_close": prev_close,
                "change_pct": change_pct,
                "high": None,
                "low": None,
                "market_cap": None,
                "currency": "USD" if market.upper() == "US" else None,
                "source": f"OHLCV cache (close {last_dt})",
                "stale": True,
                "ts": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.debug(f"Cache fetch {symbol}: {e}")
            return None


# ─── Router ─────────────────────────────────────────────────────────────────

class QuoteRouter:
    """Provider chain with load-balanced dispatch.

    fetch_quote: walks providers in priority order, calling try_fetch on
    each. The first provider with idle capacity AND a usable result wins.
    If every try_fetch returns None (all saturated or all transiently
    failed), falls back to force_fetch (which may block) in the same order.

    fetch_quotes: spreads a batch across a ThreadPool — each thread runs
    fetch_quote independently, so providers see real-time saturation and
    naturally balance load (e.g. first 16 requests grab IB slots, the rest
    spill to Yahoo in parallel).
    """

    def __init__(self, providers: List[QuoteProvider]):
        self.providers = providers

    def fetch_quote(self, symbol: str, market: str) -> Optional[dict]:
        market_u = str(market).upper()
        # Pass 1: non-blocking, in priority order
        for p in self.providers:
            if not p.supports(market_u):
                continue
            result = p.try_fetch(symbol, market_u)
            if _is_usable(result):
                return result
        # Pass 2: blocking fallback. Cache is unconditionally available so
        # at minimum we return a stale-but-real number.
        for p in self.providers:
            if not p.supports(market_u):
                continue
            result = p.force_fetch(symbol, market_u)
            if _is_usable(result):
                return result
        return None

    def fetch_quotes(
        self, symbols: List[str], market: str, max_workers: int = 32,
        timeout: float = 15.0,
    ) -> Dict[str, Optional[dict]]:
        if not symbols:
            return {}
        n_workers = min(max_workers, len(symbols))
        result: Dict[str, Optional[dict]] = {sym: None for sym in symbols}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(self.fetch_quote, sym, market): sym for sym in symbols}
            for fut in as_completed(futures, timeout=timeout):
                sym = futures[fut]
                try:
                    result[sym] = fut.result(timeout=0.1)
                except Exception:
                    result[sym] = None
        return result

    def status(self) -> dict:
        return {
            "providers": [
                {
                    "name": p.name,
                    "supports": sorted(p.supported_markets),
                    "available": p.available(),
                }
                for p in self.providers
            ],
        }


# ─── Helpers ────────────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None


def _yf_symbol(symbol: str, market: str) -> str:
    """Internal symbol → Yahoo Finance ticker. Duplicated from main.yf_symbol
    to avoid the circular import (main imports ib_gateway, which would then
    import quote_router which would import main)."""
    sym = symbol.upper()
    market = market.upper()
    if market == "HK":
        return sym.zfill(4) + ".HK"
    if market == "CN":
        # Shanghai prefixes (6/9) → .SS; Shenzhen (others) → .SZ
        if sym.startswith(("6", "9")):
            return sym + ".SS"
        return sym + ".SZ"
    if market == "UK":
        return sym + ".L"
    if market == "CRYPTO":
        return sym + "-USD"
    return sym


def _is_usable(quote: Optional[dict]) -> bool:
    return bool(quote and quote.get("price") is not None)


# ─── Default instance ───────────────────────────────────────────────────────

router = QuoteRouter(providers=[
    IBProvider(),
    YahooProvider(max_concurrent=int(os.getenv("YF_MAX_CONCURRENT", "8"))),
    CacheProvider(),
])
