"""Unit tests for SubscriptionManager ref-counting + isolation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.sub_manager import (
    SubscriptionManager, SubscriptionLimitExceeded, DEFAULT_CLIENT_QUEUE_SIZE,
)
from streaming import protocol as proto


# Helper: track upstream subscribe / unsubscribe calls.
class _UpstreamProbe:
    def __init__(self):
        self.subs: list[proto.StreamKey] = []
        self.unsubs: list[proto.StreamKey] = []

    async def subscribe(self, k):
        self.subs.append(k)

    async def unsubscribe(self, k):
        self.unsubs.append(k)


def _make_mgr(*, cooldown=0.05, queue_size=4, max_subs=3):
    """Build manager with short cooldown so tests run fast."""
    probe = _UpstreamProbe()
    mgr = SubscriptionManager(
        subscribe_fn=probe.subscribe,
        unsubscribe_fn=probe.unsubscribe,
        unsub_cooldown_s=cooldown,
        client_queue_size=queue_size,
        subs_per_conn_max=max_subs,
    )
    return mgr, probe


def _k_tick(symbol='AAPL'):
    return proto.StreamKey(market='US', symbol=symbol, channel=proto.CH_TICK)


def _k_bar(symbol='AAPL', interval='1m'):
    return proto.StreamKey(market='US', symbol=symbol, channel=proto.CH_BAR, interval=interval)


# ─────────────────────────────────────────────────────────────────────────────
# Ref-counting
# ─────────────────────────────────────────────────────────────────────────────


async def first_subscribe_triggers_upstream():
    mgr, probe = _make_mgr()
    mgr.start_janitor()
    h1 = mgr.add_client()
    await mgr.subscribe(h1, _k_tick())
    assert probe.subs == [_k_tick()]
    assert probe.unsubs == []
    await mgr.stop_janitor()


async def second_subscribe_same_key_does_not_call_upstream_again():
    mgr, probe = _make_mgr()
    mgr.start_janitor()
    h1 = mgr.add_client()
    h2 = mgr.add_client()
    await mgr.subscribe(h1, _k_tick())
    await mgr.subscribe(h2, _k_tick())
    assert len(probe.subs) == 1
    await mgr.stop_janitor()


async def last_unsubscribe_after_cooldown_tears_down():
    mgr, probe = _make_mgr(cooldown=0.05)
    mgr.start_janitor()
    h1 = mgr.add_client()
    await mgr.subscribe(h1, _k_tick())
    await mgr.unsubscribe(h1, _k_tick())
    assert probe.unsubs == []  # cooldown not elapsed yet
    await asyncio.sleep(0.15)  # cooldown + janitor sweep interval
    assert probe.unsubs == [_k_tick()]
    await mgr.stop_janitor()


async def re_subscribe_inside_cooldown_cancels_teardown():
    mgr, probe = _make_mgr(cooldown=0.1)
    mgr.start_janitor()
    h1 = mgr.add_client()
    await mgr.subscribe(h1, _k_tick())
    await mgr.unsubscribe(h1, _k_tick())
    # Re-subscribe before cooldown expires
    h2 = mgr.add_client()
    await mgr.subscribe(h2, _k_tick())
    await asyncio.sleep(0.2)
    assert probe.unsubs == []  # tear-down was cancelled
    assert len(probe.subs) == 1  # upstream still up from the first subscribe
    await mgr.stop_janitor()


async def remove_client_cleans_all_keys():
    mgr, probe = _make_mgr(cooldown=0.05)
    mgr.start_janitor()
    h = mgr.add_client()
    await mgr.subscribe(h, _k_tick('AAPL'))
    await mgr.subscribe(h, _k_tick('NVDA'))
    await mgr.remove_client(h)
    await asyncio.sleep(0.15)
    # Both should have torn down
    assert sorted(k.symbol for k in probe.unsubs) == ['AAPL', 'NVDA']
    await mgr.stop_janitor()


async def subscription_limit_exceeded_raises():
    mgr, _ = _make_mgr(max_subs=2)
    mgr.start_janitor()
    h = mgr.add_client()
    await mgr.subscribe(h, _k_tick('AAA'))
    await mgr.subscribe(h, _k_tick('BBB'))
    try:
        await mgr.subscribe(h, _k_tick('CCC'))
        raise AssertionError("expected SubscriptionLimitExceeded")
    except SubscriptionLimitExceeded:
        pass
    await mgr.stop_janitor()


# ─────────────────────────────────────────────────────────────────────────────
# Tick fanout
# ─────────────────────────────────────────────────────────────────────────────


async def tick_fan_out_to_two_clients():
    mgr, _ = _make_mgr()
    mgr.start_janitor()
    h1 = mgr.add_client()
    h2 = mgr.add_client()
    k = _k_tick()
    await mgr.subscribe(h1, k)
    await mgr.subscribe(h2, k)
    mgr.feed_tick(k, ts_ms=1779025357000, last=100.0, bid=99.9, ask=100.1, volume=10)
    m1 = h1.queue.get_nowait()
    m2 = h2.queue.get_nowait()
    assert m1['type'] == 'tick' and m1['last'] == 100.0
    assert m2['type'] == 'tick' and m2['last'] == 100.0
    assert m1['seq'] == m2['seq']  # shared seq
    await mgr.stop_janitor()


async def slow_client_does_not_block_fast_one():
    """Send queue_size+5 messages to two subscribers. The slow client (we
    never read from its queue) should accumulate drops, the fast client
    (we drain) should see all messages.
    """
    mgr, _ = _make_mgr(queue_size=4)
    mgr.start_janitor()
    fast = mgr.add_client()
    slow = mgr.add_client()
    k = _k_tick()
    await mgr.subscribe(fast, k)
    await mgr.subscribe(slow, k)

    received_fast: list = []

    async def drainer():
        for _ in range(20):
            m = await asyncio.wait_for(fast.queue.get(), timeout=0.5)
            received_fast.append(m)

    drain_task = asyncio.create_task(drainer())
    await asyncio.sleep(0)  # let drainer start

    for i in range(20):
        mgr.feed_tick(k, ts_ms=1779025357000 + i, last=100.0 + i,
                      bid=None, ask=None, volume=1)
        await asyncio.sleep(0)  # yield so the drainer can run

    await drain_task

    # Fast client should have seen all (or nearly all — depending on
    # scheduling we may collect 16-20). At a minimum no dropping on its end.
    assert len(received_fast) >= 16, f"fast client got only {len(received_fast)}"
    # Slow client must have dropped a bunch (queue capped at 4, never drained)
    assert slow.dropped > 0, "slow client should have registered drops"
    await mgr.stop_janitor()


# ─────────────────────────────────────────────────────────────────────────────
# Bar fanout via aggregator
# ─────────────────────────────────────────────────────────────────────────────


async def switch_source_relabels_and_broadcasts():
    """Phase 2: switching source updates each stream's source field and
    pushes a source_changed message to every subscriber."""
    mgr, _ = _make_mgr()
    mgr.start_janitor()
    h1 = mgr.add_client()
    h2 = mgr.add_client()
    k = _k_tick()
    await mgr.subscribe(h1, k)
    await mgr.subscribe(h2, k)
    # Drain the initial state (no msgs yet, but no surprise)
    affected = mgr.switch_source_all(from_src='ib', to_src='yahoo')
    assert affected == [k]
    # Both clients should get a source_changed
    m1 = h1.queue.get_nowait()
    m2 = h2.queue.get_nowait()
    assert m1['type'] == 'source_changed' and m1['from'] == 'ib' and m1['to'] == 'yahoo'
    assert m2['type'] == 'source_changed' and m2['from'] == 'ib' and m2['to'] == 'yahoo'
    # Subsequent ticks should be tagged as the new source
    mgr.feed_tick(k, ts_ms=1779025357000, last=100.0, bid=None, ask=None, volume=None)
    tick = h1.queue.get_nowait()
    assert tick['source'] == 'yahoo'
    await mgr.stop_janitor()


async def switch_source_idempotent_when_no_match():
    mgr, _ = _make_mgr()
    mgr.start_janitor()
    h = mgr.add_client()
    await mgr.subscribe(h, _k_tick())
    # No yahoo-sourced streams to switch back
    affected = mgr.switch_source_all(from_src='yahoo', to_src='ib')
    assert affected == []
    await mgr.stop_janitor()


async def bar_channel_emits_aggregated_messages():
    mgr, _ = _make_mgr()
    mgr.start_janitor()
    h = mgr.add_client()
    k = _k_bar('AAPL', '1m')
    await mgr.subscribe(h, k)
    # Two ticks in same minute → 1 new + 1 update
    mgr.feed_tick_for_bar(k, ts_ms=1779025357000, last=100.0, volume=10)
    mgr.feed_tick_for_bar(k, ts_ms=1779025357500, last=101.0, volume=5)
    m1 = h.queue.get_nowait()
    m2 = h.queue.get_nowait()
    assert m1['type'] == 'bar' and m1['action'] == 'new' and m1['o'] == 100.0
    assert m2['type'] == 'bar' and m2['action'] == 'update' and m2['c'] == 101.0
    assert m2['seq'] == m1['seq'] + 1
    await mgr.stop_janitor()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    import inspect
    tests = [v for k, v in globals().items()
             if inspect.iscoroutinefunction(v) and not k.startswith('_')]
    failed = []
    for t in tests:
        try:
            asyncio.run(t())
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n  {len(tests)-len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
