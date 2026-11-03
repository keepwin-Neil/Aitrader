"""Unit tests for bar_aggregator boundary math + state transitions."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# So `python streaming/tests/test_bar_aggregator.py` works without pytest setup
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.bar_aggregator import BarAggregator, bar_start, bar_end
from streaming import protocol as proto


# ─────────────────────────────────────────────────────────────────────────────
# bar_start() — pure boundary math
# ─────────────────────────────────────────────────────────────────────────────


def test_bar_start_1m_aligns_to_minute():
    # 2026-05-20 13:42:37.123 UTC = 1779025357123 ms
    ts = 1779025357123
    # 13:42:00 UTC = 1779025320000
    assert bar_start(ts, '1m', 'US') == 1779025320000


def test_bar_start_5m_aligns_to_5min():
    ts = 1779025357123  # 13:42:37
    # 13:40:00 UTC = 1779025200000
    assert bar_start(ts, '5m', 'US') == 1779025200000


def test_bar_start_1h_aligns_to_hour():
    ts = 1779025357123  # 13:42:37
    # 13:00:00 UTC = 1779022800000
    assert bar_start(ts, '1h', 'US') == 1779022800000


def test_bar_start_at_exact_boundary_returns_self():
    # 13:42:00.000 UTC
    ts = 1779025320000
    assert bar_start(ts, '1m', 'US') == ts


def test_bar_start_1d_us_uses_eastern_midnight():
    # Pick 2026-05-20 04:00:00 UTC. In ET (UTC-4 DST), that's 2026-05-20 00:00 ET.
    # Both UTC and ET interpretations land on "2026-05-20" so we need a more
    # interesting case: 03:00 UTC = 23:00 ET previous day, so bar should be
    # 2026-05-19 ET midnight = 2026-05-19 04:00 UTC.
    ts_03utc = int(datetime(2026, 5, 20, 3, 0, tzinfo=timezone.utc).timestamp() * 1000)
    expected = int(datetime(2026, 5, 19, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert bar_start(ts_03utc, '1d', 'US') == expected


def test_bar_start_1d_hk_uses_hk_midnight():
    # HK is UTC+8, no DST. 2026-05-20 16:00 UTC = 2026-05-21 00:00 HKT.
    ts = int(datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # The bar containing ts is "HK 2026-05-21", which starts at 16:00 UTC.
    assert bar_start(ts, '1d', 'HK') == ts

    # Just before that, 2026-05-20 15:59 UTC is still HK 2026-05-20, which
    # started at 2026-05-19 16:00 UTC.
    ts_before = int(datetime(2026, 5, 20, 15, 59, tzinfo=timezone.utc).timestamp() * 1000)
    expected = int(datetime(2026, 5, 19, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert bar_start(ts_before, '1d', 'HK') == expected


def test_bar_start_1d_us_dst_spring_forward():
    # DST began 2026-03-08 02:00 ET → jumps to 03:00 ET. So 2026-03-08 ET
    # day starts at 2026-03-08 05:00 UTC (EST UTC-5) and 2026-03-09 ET
    # day starts at 2026-03-09 04:00 UTC (EDT UTC-4).
    ts_mar08_noon_utc = int(datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    expected_mar08_start = int(datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert bar_start(ts_mar08_noon_utc, '1d', 'US') == expected_mar08_start

    ts_mar10_noon_utc = int(datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    expected_mar10_start = int(datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert bar_start(ts_mar10_noon_utc, '1d', 'US') == expected_mar10_start


def test_bar_end_1m_is_start_plus_60s():
    ts = 1779025357123
    assert bar_end(ts, '1m', 'US') == bar_start(ts, '1m', 'US') + 60_000


def test_bar_end_1d_us_dst_spring_forward_is_23h_not_24h():
    # On the spring-forward day, the ET calendar day is only 23 hours long
    # (lost 1 hour to DST). Make sure bar_end - bar_start reflects that.
    ts = int(datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    duration_ms = bar_end(ts, '1d', 'US') - bar_start(ts, '1d', 'US')
    assert duration_ms == 23 * 60 * 60 * 1000


# ─────────────────────────────────────────────────────────────────────────────
# BarAggregator state transitions
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregator_first_tick_opens_new_bar():
    a = BarAggregator(interval='1m', market='US')
    events = a.feed(1779025357123, last=100.0, volume=10)
    assert len(events) == 1
    assert events[0][0] == proto.BAR_NEW
    bar = events[0][1]
    assert bar['o'] == bar['h'] == bar['l'] == bar['c'] == 100.0
    assert bar['v'] == 10


def test_aggregator_same_bar_update():
    a = BarAggregator(interval='1m', market='US')
    a.feed(1779025357123, last=100.0, volume=10)  # opens bar @ 13:42
    events = a.feed(1779025357999, last=101.0, volume=5)  # same minute
    assert len(events) == 1
    assert events[0][0] == proto.BAR_UPDATE
    bar = events[0][1]
    assert bar['o'] == 100.0
    assert bar['h'] == 101.0
    assert bar['l'] == 100.0
    assert bar['c'] == 101.0
    assert bar['v'] == 15


def test_aggregator_boundary_crossing_emits_two_events():
    a = BarAggregator(interval='1m', market='US')
    a.feed(1779025357123, last=100.0, volume=10)  # 13:42:37
    # Push to 13:43:05 — boundary crossed
    events = a.feed(1779025385000, last=102.0, volume=7)
    assert len(events) == 2
    # First: final update on the old bar (close = 100.0, the price from before)
    assert events[0][0] == proto.BAR_UPDATE
    old_bar = events[0][1]
    assert old_bar['o'] == old_bar['h'] == old_bar['l'] == old_bar['c'] == 100.0
    assert old_bar['v'] == 10
    # Second: new bar opens
    assert events[1][0] == proto.BAR_NEW
    new_bar = events[1][1]
    assert new_bar['o'] == new_bar['h'] == new_bar['l'] == new_bar['c'] == 102.0
    assert new_bar['v'] == 7


def test_aggregator_bidonly_tick_emits_nothing():
    a = BarAggregator(interval='1m', market='US')
    events = a.feed(1779025357123, last=None, volume=10)
    assert events == []


def test_aggregator_high_low_tracking():
    a = BarAggregator(interval='1m', market='US')
    a.feed(1779025357000, last=100.0, volume=1)
    a.feed(1779025357100, last=105.0, volume=1)
    a.feed(1779025357200, last=98.0, volume=1)
    events = a.feed(1779025357300, last=102.0, volume=1)
    bar = events[0][1]
    assert bar['o'] == 100.0
    assert bar['h'] == 105.0
    assert bar['l'] == 98.0
    assert bar['c'] == 102.0


if __name__ == '__main__':
    # Mini test runner so this file works without pytest.
    import inspect
    tests = [v for k, v in globals().items() if k.startswith('test_') and inspect.isfunction(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, e))
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"  {len(tests)-len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
