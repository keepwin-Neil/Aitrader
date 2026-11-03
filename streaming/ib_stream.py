"""IB Gateway → SubscriptionManager bridge.

Owns the IB slots reserved by ib_gateway (see get_streaming_slots()) and
wires ib_insync callbacks into SubscriptionManager.feed_tick / feed_tick_for_bar.

Threading model:
  - Each _IBSlot runs its own asyncio loop on a dedicated thread (see
    ib_gateway._IBSlot._run).
  - ib_insync callbacks fire on that slot's loop thread.
  - We hop back to the *main* event loop (where SubscriptionManager lives)
    via `main_loop.call_soon_threadsafe(...)`.

Slot allocation strategy (Phase 1, 2 slots):
  - slot 0: reqTickByTickData — true tick-by-tick, used for /ws/quote/tick.
            IB limit: 5 concurrent per client. 1 slot × 5 = 5 streams max
            on tick channel. For Phase 1 (AAPL only) that's plenty.
  - slot 1: reqMktData — conflated last/bid/ask, used for /ws/quote/bar
            aggregation. IB limit: ~100 lines per client. Fan out from here.

Phase 2 will expand to 4 slots (2 tick + 2 mktdata).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from streaming import protocol as proto
from streaming.sub_manager import SubscriptionManager

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IB stream coordinator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _StreamHandle:
    """Per-stream IB-side state."""
    key: proto.StreamKey
    slot: Any  # ib_gateway._IBSlot
    contract: Any  # ib_insync Contract
    ticker: Any = None  # the ib_insync Ticker returned by reqTickByTickData / reqMktData
    callback: Any = None  # the closure registered on ticker.updateEvent (for detach)
    last_seen_volume: float = 0.0  # for volume-delta tracking


class IBStreamBridge:
    """Wires SubscriptionManager.subscribe_fn / unsubscribe_fn to IB.

    Singleton. Created at app startup. Holds the reserved IB slots.

    Phase 2: watches IB connection state and triggers a Yahoo failover when
    the IB connection drops. on_ib_disconnect / on_ib_reconnect callbacks
    can be wired by ws_routes to flip sub_manager source labels and
    add/remove keys from the yahoo poller.
    """

    def __init__(self, sub_manager: SubscriptionManager):
        self.sub_manager = sub_manager
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._tick_slot: Any = None        # for reqTickByTickData
        self._mktdata_slot: Any = None     # for reqMktData
        self._streams: dict[proto.StreamKey, _StreamHandle] = {}
        self._lock = threading.RLock()
        # Source-failover callbacks (set by ws_routes during init)
        self.on_ib_disconnect: Optional[Any] = None  # () -> None, called from main loop
        self.on_ib_reconnect: Optional[Any] = None
        # Start in "disconnected" state. The first health tick after slot
        # binding will fire on_ib_reconnect — harmless when no subscribers
        # exist yet, and avoids a spurious on_ib_disconnect during the
        # startup race where slots take a few seconds to come online.
        self._ib_was_connected = False
        self._health_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Acquire IB slots and bind to current event loop. Idempotent.

        Slot binding can fail if IB Gateway isn't ready at app startup; in
        that case the warning is logged but we don't bail — subscribe() will
        retry binding lazily via _try_bind_slots().
        """
        self._main_loop = asyncio.get_running_loop()
        self._try_bind_slots()
        if self._tick_slot is None:
            log.warning("[ib_stream] no IB streaming slots reserved at startup — "
                        "will retry on first subscribe() call")
        # Start health monitor — polls is_ok() on the streaming slots every
        # second and triggers on_ib_disconnect / on_ib_reconnect on edge
        # transitions. Phase 2 §5.2: <2s failover.
        self._health_task = self._main_loop.create_task(self._health_loop())

    async def _health_loop(self) -> None:
        """Watch IB slot health, emit edge transitions to ws_routes callbacks."""
        try:
            while True:
                await asyncio.sleep(1.0)
                if not self._slots_ok():
                    # Our bound slot objects may be dead while ib_gateway's
                    # maintenance thread has already swapped fresh replacements
                    # into _streaming_slots. Re-pull so recovery is automatic.
                    self._try_bind_slots(rebind=True)
                ok = self._slots_ok()
                if self._ib_was_connected and not ok:
                    self._ib_was_connected = False
                    log.warning("[ib_stream] IB slots no longer healthy — "
                                "triggering Yahoo failover")
                    # Old handles reference dead tickers on a dead connection;
                    # drop them so a later re-subscribe starts clean instead of
                    # early-returning on `key in self._streams` (zombie state).
                    with self._lock:
                        self._streams.clear()
                    if self.on_ib_disconnect is not None:
                        try:
                            self.on_ib_disconnect()
                        except Exception:
                            log.exception("[ib_stream] on_ib_disconnect callback failed")
                elif (not self._ib_was_connected) and ok:
                    self._ib_was_connected = True
                    log.info("[ib_stream] IB slots healthy again — switching back from Yahoo")
                    if self.on_ib_reconnect is not None:
                        try:
                            self.on_ib_reconnect()
                        except Exception:
                            log.exception("[ib_stream] on_ib_reconnect callback failed")
        except asyncio.CancelledError:
            return

    def _slots_ok(self) -> bool:
        """True iff at least one streaming slot reports is_ok()."""
        for s in (self._tick_slot, self._mktdata_slot):
            if s is not None and s.is_ok():
                return True
        return False

    async def stop(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

    async def subscribe(self, key: proto.StreamKey) -> None:
        """Tell IB to start streaming for this key. Called by SubscriptionManager."""
        # Startup race: if IB Gateway wasn't ready when init_streaming() ran,
        # self._tick_slot is None forever. Lazy-rebind: every subscribe()
        # tries to acquire slots again, so the first real WS connection after
        # IB comes online recovers the bridge.
        if self._tick_slot is None:
            self._try_bind_slots()
        if self._tick_slot is None:
            raise UpstreamDisconnected(
                "IB Gateway streaming slots not available yet — retry shortly"
            )
        with self._lock:
            if key in self._streams:
                return  # already streaming
            handle = await self._start_one(key)
            self._streams[key] = handle

    def _try_bind_slots(self, rebind: bool = False) -> None:
        """Attempt to grab the reserved streaming slots from ib_gateway.

        Called at start(), lazily on subscribe(), and from the health loop
        with rebind=True when the currently-bound slot objects have died —
        ib_gateway's maintenance thread may have swapped fresh replacements
        into _streaming_slots, so re-pulling picks up the revived connections.
        """
        from ib_gateway import get_streaming_slots
        slots = get_streaming_slots()
        if not slots:
            return
        new_tick = slots[0]
        new_mkt = slots[1] if len(slots) >= 2 else slots[0]
        if not rebind and self._tick_slot is new_tick and self._mktdata_slot is new_mkt:
            return  # already bound to these exact objects
        if self._tick_slot is new_tick and self._mktdata_slot is new_mkt:
            return  # rebind requested but nothing changed upstream yet
        self._tick_slot = new_tick
        self._mktdata_slot = new_mkt
        log.info(
            "[ib_stream] bound tick_slot=client_id=%d, mktdata_slot=client_id=%d",
            self._tick_slot.client_id, self._mktdata_slot.client_id,
        )

    async def unsubscribe(self, key: proto.StreamKey) -> None:
        """Tell IB to stop streaming for this key."""
        with self._lock:
            handle = self._streams.pop(key, None)
        if handle is None:
            return
        await self._stop_one(handle)

    # ── per-key subscribe/stop ──────────────────────────────────────────────

    async def _start_one(self, key: proto.StreamKey) -> _StreamHandle:
        # ib_insync API requires running on the slot's own loop thread.
        from ib_insync import Stock
        # For now Phase 1: US stocks only (validation already done upstream)
        contract = Stock(key.symbol, "SMART", "USD")

        slot = self._tick_slot if key.channel == proto.CH_TICK else self._mktdata_slot

        async def _do(ib):
            # Qualify the contract (resolves conId) — quick.
            qualified = await ib.qualifyContractsAsync(contract)
            if not qualified:
                raise UnknownSymbol(f"contract not qualified: {key.symbol}")
            c = qualified[0]
            if key.channel == proto.CH_TICK:
                # reqTickByTickData: "Last" gives prints; "BidAsk" gives quote
                # changes. We use Last for Phase 1 (matches what bar aggregator
                # would feed on either path).
                ticker = ib.reqTickByTickData(c, "Last", 0, False)
            else:
                # reqMktData: conflated last/bid/ask
                ticker = ib.reqMktData(c, "", False, False)
            return c, ticker

        contract, ticker = slot.submit(_do, timeout=8.0)

        # Hook the ticker.updateEvent → sub_manager (via main loop). Keep the
        # closure on the handle so _stop_one can detach it — ib_insync reuses
        # Ticker objects per contract, so without an explicit `-=` a
        # re-subscribe would stack a second callback and double-push every
        # tick to sub_manager.
        callback = self._make_callback(key)
        ticker.updateEvent += callback
        log.info("[ib_stream] subscribed: %s slot=%d", key, slot.client_id)
        return _StreamHandle(key=key, slot=slot, contract=contract,
                             ticker=ticker, callback=callback)

    async def _stop_one(self, handle: _StreamHandle) -> None:
        # Detach our callback first — even if the cancel RPC below fails, no
        # further ticks flow into sub_manager for this stream.
        if handle.ticker is not None and handle.callback is not None:
            try:
                handle.ticker.updateEvent -= handle.callback
            except Exception:
                pass

        channel = handle.key.channel

        async def _do(ib):
            # Cancel only the request type this handle actually made.
            if channel == proto.CH_TICK:
                try:
                    ib.cancelTickByTickData(handle.contract, "Last")
                except Exception:
                    pass
            else:
                try:
                    ib.cancelMktData(handle.contract)
                except Exception:
                    pass
        try:
            handle.slot.submit(_do, timeout=4.0)
        except Exception:
            # Slot may already be dead (IB drop) — cancel is moot then.
            log.warning("[ib_stream] cancel skipped/failed for %s (slot down?)", handle.key)
        log.info("[ib_stream] unsubscribed: %s", handle.key)

    # ── callback bridge: IB thread → main loop ──────────────────────────────

    def _make_callback(self, key: proto.StreamKey):
        """Return a closure that handles a ticker update.

        IB fires this on the slot's loop thread. We schedule the actual
        feed_tick call on the main loop via call_soon_threadsafe.
        """
        main_loop = self._main_loop
        sub_mgr = self.sub_manager
        if main_loop is None:
            raise RuntimeError("IBStreamBridge.start() not called")

        # capture last_volume out-of-band so we report delta not cumulative.
        # Sentinel `None` means we haven't seen any volume yet — on first
        # observation we just record the baseline and report delta=0, so
        # the bar aggregator doesn't get the entire day's volume slammed
        # into the first bar.
        state: dict = {"last_volume": None}

        def _on_update(ticker):
            try:
                ts_ms = _now_ms()
                last = _safe_float(getattr(ticker, "last", None))
                bid = _safe_float(getattr(ticker, "bid", None))
                ask = _safe_float(getattr(ticker, "ask", None))
                cum_vol = _safe_float(getattr(ticker, "volume", None))
                # IB volume is cumulative for the day; emit deltas so
                # bar aggregator can sum naturally.
                vol_delta: Optional[float] = None
                if cum_vol is not None:
                    if state["last_volume"] is None:
                        # First-ever observation: set baseline, no delta yet
                        vol_delta = 0.0
                    else:
                        delta = cum_vol - state["last_volume"]
                        if delta >= 0:
                            vol_delta = delta
                    state["last_volume"] = cum_vol

                # call_soon_threadsafe doesn't take kwargs, so wrap in a
                # captured-state closure.
                if key.channel == proto.CH_TICK:
                    def _push():
                        sub_mgr.feed_tick(
                            key, ts_ms=ts_ms, last=last, bid=bid, ask=ask, volume=vol_delta,
                        )
                    main_loop.call_soon_threadsafe(_push)
                else:
                    def _push_bar():
                        sub_mgr.feed_tick_for_bar(
                            key, ts_ms=ts_ms, last=last, volume=vol_delta,
                        )
                    main_loop.call_soon_threadsafe(_push_bar)
            except Exception:
                log.exception("[ib_stream] callback failed for %s", key)

        return _on_update


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(x: Any) -> Optional[float]:
    """ib_insync sets unset fields to nan; treat those as None."""
    if x is None:
        return None
    try:
        f = float(x)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Errors (caught by ws_routes and converted to error frames)
# ─────────────────────────────────────────────────────────────────────────────


class UpstreamDisconnected(Exception):
    pass


class UnknownSymbol(Exception):
    pass
