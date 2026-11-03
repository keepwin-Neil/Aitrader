"""Tick → bar aggregation with server-side time authority.

The aggregator owns the bar boundary decision so that all clients subscribed
to the same (symbol, interval) stream see identical bars. Boundaries use
tick.time (the IB upstream timestamp), not the local wall clock — this avoids
NTP drift skewing whose-bar-is-this calls.

Time zones:
  - 1s / 1m / 5m / 15m / 1h: boundaries are UTC multiples (ts // step * step)
  - 1d: boundaries align to the market's local calendar day. US uses ET,
    HK uses Asia/Hong_Kong, CN uses Asia/Shanghai. This means a "1d bar"
    rolls when the *market* sees midnight, not UTC. DST handled by zoneinfo.

See design §3.2a.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

from streaming import protocol as proto


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Interval → seconds. 1d handled separately (calendar-aware).
_INTERVAL_SECONDS = {
    proto.INT_1S: 1,
    proto.INT_1M: 60,
    proto.INT_5M: 5 * 60,
    proto.INT_15M: 15 * 60,
    proto.INT_1H: 60 * 60,
}

# Market → IANA timezone for the 1d boundary.
_MARKET_TZ = {
    proto.MK_US: ZoneInfo("America/New_York"),
    proto.MK_HK: ZoneInfo("Asia/Hong_Kong"),
    proto.MK_CN: ZoneInfo("Asia/Shanghai"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Boundary math
# ─────────────────────────────────────────────────────────────────────────────


def bar_start(ts_ms: int, interval: str, market: str) -> int:
    """Return the UTC ms epoch of the bar that ``ts_ms`` falls into.

    The returned value is itself a valid `ts` for the bar (the bar opens at
    bar_start and closes at bar_start + duration).

    For 1d: the "calendar day" of the market is used. The returned ms epoch
    corresponds to 00:00 local time in the market's timezone, expressed
    back in UTC.
    """
    if interval in _INTERVAL_SECONDS:
        step_ms = _INTERVAL_SECONDS[interval] * 1000
        return (ts_ms // step_ms) * step_ms

    if interval == proto.INT_1D:
        tz = _MARKET_TZ.get(market.upper())
        if tz is None:
            raise ValueError(f"1d aggregation requires a known market; got {market!r}")
        # Convert to market-local time, drop time-of-day, convert back.
        from datetime import datetime, timezone
        dt_local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz)
        midnight_local = dt_local.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight_local.timestamp() * 1000)

    raise ValueError(f"unsupported interval {interval!r}")


def bar_end(ts_ms: int, interval: str, market: str) -> int:
    """Return the (exclusive) UTC ms epoch where the bar containing ts_ms ends.

    For non-1d intervals this is just start + step_ms. For 1d we have to
    compute "next calendar midnight in market timezone" — DST means we can't
    assume 86_400_000.
    """
    start = bar_start(ts_ms, interval, market)
    if interval in _INTERVAL_SECONDS:
        return start + _INTERVAL_SECONDS[interval] * 1000
    if interval == proto.INT_1D:
        tz = _MARKET_TZ[market.upper()]
        from datetime import datetime, timedelta, timezone
        dt_local = datetime.fromtimestamp(start / 1000, tz=timezone.utc).astimezone(tz)
        # +1 calendar day (DST-aware: zoneinfo handles the spring/fall offset jumps)
        next_day_local = (dt_local + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(next_day_local.timestamp() * 1000)
    raise ValueError(f"unsupported interval {interval!r}")


# ─────────────────────────────────────────────────────────────────────────────
# BarAggregator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Bar:
    """In-progress bar state."""
    open_ts: int
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ts": self.open_ts,
            "o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v,
        }


@dataclass
class BarAggregator:
    """Folds a stream of ticks into a stream of bar updates.

    Usage:
        agg = BarAggregator(interval='1m', market='US')
        for tick in upstream:
            for action, bar_dict in agg.feed(tick['ts'], tick['last'], tick['volume']):
                # action ∈ {'update', 'new'}; bar_dict is OHLCV + ts
                ...

    feed() may emit 0, 1, or 2 events:
      - 0:    tick had no price (e.g. bid-only tick), state unchanged
      - 1:    same-bar update OR first tick of a new bar
      - 2:    closing-update on the previous bar (BAR_UPDATE) + opening
              tick on the new bar (BAR_NEW), if the tick crosses a boundary

    Ticks with `last is None` (bid/ask-only) still bump the volume if a
    non-None volume was passed. Bid/ask-only ticks with no price contribute
    nothing — they belong to the tick channel, not the bar channel.
    """
    interval: str
    market: str
    _cur: Optional[_Bar] = field(default=None, init=False, repr=False)

    def feed(
        self,
        ts_ms: int,
        last: Optional[float],
        volume: Optional[float] = None,
    ) -> list[tuple[str, dict]]:
        """Feed one tick. Returns 0–2 (action, bar_dict) events."""
        if last is None:
            return []  # bar channel only cares about prints

        out: list[tuple[str, dict]] = []
        new_start = bar_start(ts_ms, self.interval, self.market)
        v = float(volume) if volume is not None else 0.0

        if self._cur is None:
            # First-ever tick — open a new bar.
            self._cur = _Bar(
                open_ts=new_start, o=last, h=last, l=last, c=last, v=v,
            )
            out.append((proto.BAR_NEW, self._cur.to_dict()))
            return out

        if new_start == self._cur.open_ts:
            # Same bar — update HLCV
            if last > self._cur.h: self._cur.h = last
            if last < self._cur.l: self._cur.l = last
            self._cur.c = last
            self._cur.v += v
            out.append((proto.BAR_UPDATE, self._cur.to_dict()))
            return out

        # Boundary crossed. Emit final update on the old bar (so subscribers
        # see the close), then open a new bar.
        # Note: we do *not* mutate the old bar with this tick's price — the
        # tick belongs to the new bar.
        out.append((proto.BAR_UPDATE, self._cur.to_dict()))
        self._cur = _Bar(
            open_ts=new_start, o=last, h=last, l=last, c=last, v=v,
        )
        out.append((proto.BAR_NEW, self._cur.to_dict()))
        return out

    def snapshot(self) -> Optional[dict]:
        """Return current in-progress bar, or None if no tick seen yet."""
        return self._cur.to_dict() if self._cur else None
