"""Yahoo Finance pseudo-streaming via 3-second batched polling.

Yahoo doesn't offer a streaming API. We poll yfinance every YAHOO_POLL_INTERVAL_S
seconds for all currently-subscribed symbols, diff against the previous round,
and push tick events for symbols whose price changed. From the frontend's
perspective this looks identical to IB streaming, just lower frequency.

Design choices:
  - One thread, one yfinance.Tickers().history batch per round. yfinance does
    its own internal HTTP fan-out and respects backoff. Batching 50 symbols
    into one call is ~1-2s, well under the 3s interval, with one HTTP fan-out
    (vs 50 separate calls → 429-ed in seconds).
  - Diff is on (price, volume) tuple. We don't push when nothing changed,
    so an idle stream uses ~0 bandwidth.
  - Volume is reported as the cumulative-day total like IB; ib_stream's
    delta logic isn't reused here because Yahoo's volume isn't fine-grained
    enough to be useful as bar input. For bar aggregation, we feed the
    `last` price and let the aggregator track volume=0.

Phase 2 spec acceptance:
  - 50 symbols in one round, no 429 (achieved by single batch HTTP call)
  - <2s switchover IB → Yahoo (handled by sub_manager.switch_source)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from streaming import protocol as proto
from streaming.sub_manager import SubscriptionManager

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_POLL_INTERVAL_S = 3.0
DEFAULT_BATCH_SIZE = 50    # per yfinance call; conservative


# ─────────────────────────────────────────────────────────────────────────────
# YahooPoller
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Snap:
    """Last seen price+volume for a symbol."""
    price: Optional[float] = None
    volume: Optional[float] = None
    ts: int = 0


class YahooPoller:
    """Polls Yahoo for all currently-subscribed (market, symbol) keys.

    Owned by ws_routes.init_streaming alongside IBStreamBridge. Lives in
    its own background thread so the asyncio event loop isn't blocked by
    yfinance HTTP calls.
    """

    def __init__(
        self,
        sub_manager: SubscriptionManager,
        *,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.sub_manager = sub_manager
        self.poll_interval_s = poll_interval_s
        self.batch_size = batch_size
        self._active_keys: dict[proto.StreamKey, _Snap] = {}
        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        # stats
        self.rounds = 0
        self.errors = 0
        self.last_round_duration_s: Optional[float] = None

    def start(self) -> None:
        """Bind to current event loop and spawn the polling thread."""
        if self._thread is not None:
            return
        self._main_loop = asyncio.get_running_loop()
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="yahoo-poller",
        )
        self._thread.start()
        log.info("[yahoo_poller] started (interval=%.1fs, batch=%d)",
                 self.poll_interval_s, self.batch_size)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        log.info("[yahoo_poller] stopped (rounds=%d errors=%d)", self.rounds, self.errors)

    # ── public API (called by sub_manager.switch_source) ────────────────────

    def add_key(self, key: proto.StreamKey) -> None:
        """Start polling this key (idempotent)."""
        with self._lock:
            if key not in self._active_keys:
                self._active_keys[key] = _Snap()
                log.info("[yahoo_poller] added %s (total=%d)", key, len(self._active_keys))

    def remove_key(self, key: proto.StreamKey) -> None:
        with self._lock:
            if self._active_keys.pop(key, None) is not None:
                log.info("[yahoo_poller] removed %s (total=%d)", key, len(self._active_keys))

    def active_keys(self) -> list[proto.StreamKey]:
        with self._lock:
            return list(self._active_keys.keys())

    def stats(self) -> dict:
        return {
            "active": len(self._active_keys),
            "rounds": self.rounds,
            "errors": self.errors,
            "last_round_duration_s": self.last_round_duration_s,
        }

    # ── polling loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                self._one_round()
            except Exception:
                self.errors += 1
                log.exception("[yahoo_poller] round failed")
            self.last_round_duration_s = time.monotonic() - t0
            self.rounds += 1
            # Sleep the remainder of the interval; if a round took longer
            # than the interval, just immediately start the next one.
            slack = self.poll_interval_s - self.last_round_duration_s
            if slack > 0:
                self._stop_evt.wait(slack)

    def _one_round(self) -> None:
        with self._lock:
            keys = list(self._active_keys.keys())
        if not keys:
            return

        # Group keys by market. yfinance suffixes are different per market;
        # Phase 1 supports US only, but we already prep for Phase 3 here.
        # US symbols → "AAPL"; HK → "0700.HK"; CN → "000001.SZ" / "600519.SS".
        # Phase 2: we only really use US. HK/CN suffix logic stays as a hook.
        by_market: dict[str, list[proto.StreamKey]] = {}
        for k in keys:
            by_market.setdefault(k.market, []).append(k)

        for market, mkeys in by_market.items():
            # Batch into chunks of self.batch_size
            for i in range(0, len(mkeys), self.batch_size):
                chunk = mkeys[i:i + self.batch_size]
                try:
                    self._poll_chunk(market, chunk)
                except Exception:
                    self.errors += 1
                    log.exception("[yahoo_poller] chunk failed market=%s n=%d",
                                  market, len(chunk))

    def _poll_chunk(self, market: str, keys: list[proto.StreamKey]) -> None:
        import yfinance as yf

        symbols = [_yf_symbol(k.market, k.symbol) for k in keys]
        # yfinance.Tickers + .history(period='1d', interval='1m')[:-1] would
        # work but is slow & heavy. fast_info is per-ticker; download() is the
        # cheapest batch. We just want last price + volume.
        # yf.download is sync; we're already in our own thread, so blocking
        # the requests is fine.
        df = yf.download(
            tickers=" ".join(symbols),
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker",
            auto_adjust=False,
            threads=False,  # let yfinance pool its own HTTP, don't spawn N more
        )
        if df is None or df.empty:
            return

        now_ms = int(time.time() * 1000)
        for k, ysym in zip(keys, symbols):
            try:
                # yfinance column shape varies by version AND symbol count:
                # sometimes flat (Close/Volume), sometimes multi-level keyed
                # by ticker — even for a single symbol when group_by="ticker".
                # Handle both shapes instead of guessing from len(symbols).
                import pandas as pd
                sub = df
                if isinstance(df.columns, pd.MultiIndex):
                    if ysym in df.columns.get_level_values(0):
                        sub = df[ysym]
                    else:
                        continue
                last_row = sub.tail(1)
                if last_row.empty or "Close" not in sub.columns:
                    continue
                price = float(last_row["Close"].iloc[0])
                volume = float(last_row["Volume"].iloc[0])

                # NaN guard
                if price != price or volume != volume:
                    continue

                self._push_if_changed(k, price=price, volume=volume, ts_ms=now_ms)
            except Exception:
                log.exception("[yahoo_poller] extract failed for %s", ysym)

    def _push_if_changed(
        self, key: proto.StreamKey, *, price: float, volume: float, ts_ms: int,
    ) -> None:
        """Diff against last seen; only call sub_manager if price changed."""
        with self._lock:
            snap = self._active_keys.get(key)
            if snap is None:
                return  # key was removed mid-round
            if snap.price == price and snap.volume == volume:
                return  # no change
            snap.price = price
            snap.volume = volume
            snap.ts = ts_ms

        # We're on the polling thread; hop to main loop.
        main_loop = self._main_loop
        sub_mgr = self.sub_manager
        if main_loop is None:
            return

        if key.channel == proto.CH_TICK:
            def _push():
                sub_mgr.feed_tick(
                    key, ts_ms=ts_ms, last=price, bid=None, ask=None, volume=volume,
                )
            main_loop.call_soon_threadsafe(_push)
        else:
            # bar channel — feed the cumulative-volume DELTA so the bar
            # aggregator accumulates correctly (mirrors the IB path). The
            # previous volume=None meant every Yahoo-sourced bar showed v=0.
            prev = getattr(self, "_bar_vol_seen", None)
            if prev is None:
                prev = {}
                self._bar_vol_seen = prev
            last_cum = prev.get(key)
            delta = 0.0 if last_cum is None else max(0.0, volume - last_cum)
            prev[key] = volume

            def _push_bar():
                sub_mgr.feed_tick_for_bar(key, ts_ms=ts_ms, last=price, volume=delta)
            main_loop.call_soon_threadsafe(_push_bar)


# ─────────────────────────────────────────────────────────────────────────────
# yfinance symbol formatting
# ─────────────────────────────────────────────────────────────────────────────


def _yf_symbol(market: str, symbol: str) -> str:
    """Translate internal symbol to yfinance format.

    US: as-is ("AAPL")
    HK: 4 digits + ".HK" ("00700" -> "0700.HK")
    CN: 6 digits + ".SS" (Shanghai 600xxx) or ".SZ" (Shenzhen 000xxx/002xxx/300xxx)
    """
    m = market.upper()
    s = symbol.upper()
    if m == "US":
        return s
    if m == "HK":
        # yf expects 4-digit code (no leading zero past 4 chars)
        s4 = s.lstrip("0")[:4].zfill(4)
        return f"{s4}.HK"
    if m == "CN":
        if s.startswith(("6", "9")):
            return f"{s}.SS"
        return f"{s}.SZ"
    return s
