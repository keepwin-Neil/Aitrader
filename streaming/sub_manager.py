"""SubscriptionManager — the fanout core.

One SubscriptionManager runs per service process (i.e. one in main:app
on port 9000). Multiple WS clients can subscribe to the same
(market, symbol, channel) stream and share a single upstream subscription.

Threading model:
  - SubscriptionManager itself is asyncio-native (`asyncio.Queue` per
    client), but its `feed_tick()` / `feed_bar()` entry points are
    thread-safe so the IB callback thread can call them via
    `loop.call_soon_threadsafe`.
  - Each client has its own send queue (default 1000). A slow client only
    drops its own queue, never blocks the upstream feed or other clients.
  - Reference counting: last unsubscribe drops the upstream after a 5s
    cooldown, so a user flipping between two charts doesn't tear down +
    rebuild the IB subscription.

This module is upstream-agnostic. ib_stream.py wires IB callbacks into
``feed_tick(stream_key, tick)``; sub_manager doesn't know IB exists.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from streaming import protocol as proto
from streaming.bar_aggregator import BarAggregator

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tunables (all overridable for tests)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLIENT_QUEUE_SIZE = 1000          # per-client send buffer
DEFAULT_UNSUB_COOLDOWN_S = 5.0            # delay before tearing down upstream
DEFAULT_SUBSCRIPTIONS_PER_CONN_MAX = 50   # cap to prevent spam


# Upstream callback signatures. ib_stream provides these at startup.
#   subscribe_fn(StreamKey) -> awaitable[None]
#   unsubscribe_fn(StreamKey) -> awaitable[None]
UpstreamFn = Callable[[proto.StreamKey], Awaitable[None]]


# ─────────────────────────────────────────────────────────────────────────────
# Internal state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _ClientHandle:
    """One WS client's queue + lifecycle."""
    client_id: int  # monotonic, assigned by SubscriptionManager
    queue: asyncio.Queue
    keys: set[proto.StreamKey] = field(default_factory=set)
    dropped: int = 0  # cumulative count of messages dropped because queue was full

    def send_nowait(self, msg: dict) -> bool:
        """Put on queue; on overflow, drop the oldest and return False."""
        try:
            self.queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            # Eject oldest, push newest. This favors freshness over completeness;
            # the gap will be reported via gap event by the caller.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # extremely unlikely; queue is asyncio so single-threaded
            self.dropped += 1
            return False


@dataclass
class _StreamState:
    """Shared state for one (market, symbol, channel[, interval]) stream."""
    key: proto.StreamKey
    seq: proto.SeqGen = field(default_factory=proto.SeqGen)
    subscribers: set[int] = field(default_factory=set)   # client_ids
    source: str = proto.SRC_IB                            # current data source
    bar_aggregator: Optional[BarAggregator] = None        # only for CH_BAR
    last_bar_dict: Optional[dict] = None                  # for snapshot replay
    # tear-down cooldown
    pending_unsub_at: Optional[float] = None              # monotonic clock


# ─────────────────────────────────────────────────────────────────────────────
# SubscriptionManager
# ─────────────────────────────────────────────────────────────────────────────


class SubscriptionManager:
    """Multi-client fanout + reference-counted upstream subscription manager."""

    def __init__(
        self,
        *,
        subscribe_fn: Optional[UpstreamFn] = None,
        unsubscribe_fn: Optional[UpstreamFn] = None,
        client_queue_size: int = DEFAULT_CLIENT_QUEUE_SIZE,
        unsub_cooldown_s: float = DEFAULT_UNSUB_COOLDOWN_S,
        subs_per_conn_max: int = DEFAULT_SUBSCRIPTIONS_PER_CONN_MAX,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._subscribe_fn = subscribe_fn
        self._unsubscribe_fn = unsubscribe_fn
        self._client_queue_size = client_queue_size
        self._unsub_cooldown_s = unsub_cooldown_s
        self._subs_per_conn_max = subs_per_conn_max
        self._loop = loop  # set lazily on first asyncio op
        # state
        self._clients: dict[int, _ClientHandle] = {}
        self._streams: dict[proto.StreamKey, _StreamState] = {}
        self._next_client_id = 1
        self._janitor_task: Optional[asyncio.Task] = None

    # ── loop binding ────────────────────────────────────────────────────────

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    def start_janitor(self) -> None:
        """Start the cooldown-expiry sweeper. Call once from app startup."""
        if self._janitor_task is None or self._janitor_task.done():
            loop = self._get_loop()
            self._janitor_task = loop.create_task(self._janitor_loop())

    async def _janitor_loop(self) -> None:
        """Sweep streams whose unsub cooldown has elapsed.

        Sweep cadence is half the cooldown (so worst-case latency is
        cooldown * 1.5). Cap at 2.5s so we don't busy-loop with stupid
        small cooldowns AND don't sleep forever with huge ones.
        """
        try:
            while True:
                sweep_interval = min(2.5, max(0.02, self._unsub_cooldown_s / 2))
                await asyncio.sleep(sweep_interval)
                now = time.monotonic()
                expired = [
                    key for key, st in self._streams.items()
                    if st.pending_unsub_at is not None
                    and not st.subscribers
                    and now >= st.pending_unsub_at
                ]
                for key in expired:
                    # Re-check right before teardown: the awaits in this loop
                    # let a subscribe() interleave after the snapshot above —
                    # if a new subscriber arrived (or cooldown was cancelled),
                    # skip instead of tearing down a live stream.
                    st = self._streams.get(key)
                    if st is None or st.subscribers or st.pending_unsub_at is None:
                        continue
                    await self._tear_down_stream(key)
        except asyncio.CancelledError:
            return

    async def stop_janitor(self) -> None:
        if self._janitor_task is not None:
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except asyncio.CancelledError:
                pass

    # ── client lifecycle ────────────────────────────────────────────────────

    def add_client(self) -> _ClientHandle:
        """Allocate a client handle. WS endpoint calls this on connect."""
        cid = self._next_client_id
        self._next_client_id += 1
        h = _ClientHandle(
            client_id=cid,
            queue=asyncio.Queue(maxsize=self._client_queue_size),
        )
        self._clients[cid] = h
        log.info("[sub_mgr] client %d connected (total=%d)", cid, len(self._clients))
        return h

    async def remove_client(self, h: _ClientHandle) -> None:
        """WS endpoint calls this on disconnect."""
        # Unsubscribe all this client's keys
        for key in list(h.keys):
            await self._unsubscribe_one(h, key)
        self._clients.pop(h.client_id, None)
        log.info("[sub_mgr] client %d disconnected (total=%d, dropped=%d)",
                 h.client_id, len(self._clients), h.dropped)

    # ── subscribe / unsubscribe ─────────────────────────────────────────────

    async def subscribe(self, h: _ClientHandle, key: proto.StreamKey) -> _StreamState:
        """Add a client subscription. Idempotent (re-sub is no-op)."""
        if len(h.keys) >= self._subs_per_conn_max and key not in h.keys:
            raise SubscriptionLimitExceeded(
                f"client {h.client_id} already has {len(h.keys)} subscriptions"
            )
        st = self._streams.get(key)
        if st is None:
            st = _StreamState(key=key)
            if key.channel == proto.CH_BAR and key.interval:
                st.bar_aggregator = BarAggregator(interval=key.interval, market=key.market)
            self._streams[key] = st
            # New upstream subscription needed
            if self._subscribe_fn is not None:
                try:
                    await self._subscribe_fn(key)
                except Exception:
                    log.exception("[sub_mgr] upstream subscribe failed for %s", key)
                    self._streams.pop(key, None)
                    raise
        # Cancel any pending tear-down — a new subscriber arrived
        st.pending_unsub_at = None
        if h.client_id not in st.subscribers:
            st.subscribers.add(h.client_id)
            h.keys.add(key)
        return st

    async def unsubscribe(self, h: _ClientHandle, key: proto.StreamKey) -> None:
        """Idempotent unsubscribe. Triggers cooldown-then-teardown if last."""
        await self._unsubscribe_one(h, key)

    async def _unsubscribe_one(self, h: _ClientHandle, key: proto.StreamKey) -> None:
        st = self._streams.get(key)
        if st is None:
            return
        st.subscribers.discard(h.client_id)
        h.keys.discard(key)
        if not st.subscribers:
            # Last subscriber gone — start cooldown. Janitor will tear down
            # the upstream after _unsub_cooldown_s if no one re-subscribes.
            st.pending_unsub_at = time.monotonic() + self._unsub_cooldown_s

    async def _tear_down_stream(self, key: proto.StreamKey) -> None:
        st = self._streams.pop(key, None)
        if st is None:
            return
        if self._unsubscribe_fn is not None:
            try:
                await self._unsubscribe_fn(key)
            except Exception:
                log.exception("[sub_mgr] upstream unsubscribe failed for %s", key)
        log.info("[sub_mgr] stream torn down: %s", key)

    # ── upstream feed (called from IB callback thread via call_soon_threadsafe) ─

    def feed_tick(
        self,
        key: proto.StreamKey,
        *,
        ts_ms: int,
        last: Optional[float],
        bid: Optional[float],
        ask: Optional[float],
        volume: Optional[float],
    ) -> None:
        """Push a tick into all subscribers' queues.

        IB callback thread calls this via ``loop.call_soon_threadsafe``.
        For pure thread-safety: only inspect the loop, never call asyncio
        ops directly here.
        """
        st = self._streams.get(key)
        if st is None or not st.subscribers:
            return
        seq = st.seq.next()
        msg = proto.msg_tick(
            market=key.market, symbol=key.symbol, source=st.source, seq=seq,
            last=last, bid=bid, ask=ask, volume=volume, ts=ts_ms,
        )
        self._broadcast(st, msg)

    def feed_tick_for_bar(
        self,
        bar_key: proto.StreamKey,
        *,
        ts_ms: int,
        last: Optional[float],
        volume: Optional[float],
    ) -> None:
        """Feed a tick into the bar aggregator and emit bar updates.

        Called by ib_stream when reqMktData delivers a price for a stream
        that has bar-channel subscribers.
        """
        st = self._streams.get(bar_key)
        if st is None or st.bar_aggregator is None or not st.subscribers:
            return
        for action, bar_dict in st.bar_aggregator.feed(ts_ms, last, volume):
            seq = st.seq.next()
            msg = proto.msg_bar(
                market=bar_key.market, symbol=bar_key.symbol,
                interval=bar_key.interval or "", source=st.source, seq=seq,
                action=action,
                o=bar_dict["o"], h=bar_dict["h"], l=bar_dict["l"],
                c=bar_dict["c"], v=bar_dict["v"], ts=bar_dict["ts"],
            )
            st.last_bar_dict = bar_dict
            self._broadcast(st, msg)

    def _broadcast(self, st: _StreamState, msg: dict) -> None:
        for cid in list(st.subscribers):
            h = self._clients.get(cid)
            if h is None:
                continue
            ok = h.send_nowait(msg)
            if not ok:
                # Queue overflowed. Notify client of the gap so it can
                # decide to re-snapshot. We can't know the exact seq range
                # dropped, so we just say from=current-1.
                gap = proto.msg_gap(
                    channel=st.key.channel, market=st.key.market,
                    symbol=st.key.symbol, from_seq=msg.get("seq", 0) - 1,
                    to_seq=msg.get("seq", 0),
                )
                # Best-effort: even gap may be dropped if queue is jammed.
                h.send_nowait(gap)

    # ── source switching (Phase 2) ──────────────────────────────────────────

    def switch_source_all(self, *, from_src: str, to_src: str) -> list[proto.StreamKey]:
        """Switch every stream currently sourced from `from_src` over to `to_src`.

        Returns the list of affected keys (caller wires them up on the new
        provider — e.g. yahoo_poller.add_key). Broadcasts a source_changed
        message to every subscriber.
        """
        affected: list[proto.StreamKey] = []
        for key, st in self._streams.items():
            if st.source == from_src:
                st.source = to_src
                affected.append(key)
                msg = proto.msg_source_changed(
                    market=key.market, symbol=key.symbol,
                    from_src=from_src, to_src=to_src,
                )
                self._broadcast(st, msg)
        if affected:
            log.info("[sub_mgr] source switched %d streams: %s → %s",
                     len(affected), from_src, to_src)
        return affected

    def keys_with_source(self, source: str) -> list[proto.StreamKey]:
        return [k for k, st in self._streams.items() if st.source == source]

    # ── stats (for /ws/stats) ───────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "connections": len(self._clients),
            "streams": len(self._streams),
            "subscriptions_by_symbol": self._subs_by_symbol(),
            "client_drop_counts": {
                cid: h.dropped for cid, h in self._clients.items() if h.dropped
            },
        }

    def _subs_by_symbol(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, st in self._streams.items():
            label = f"{key.market}:{key.symbol}"
            out[label] = out.get(label, 0) + len(st.subscribers)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class SubscriptionLimitExceeded(Exception):
    pass
