"""WebSocket message protocol v1 (frozen schema).

Wire-level types, validation, error codes, sequence-number generator.
Anything that touches the WS payload format goes through this module —
keep the wire format in one place so it's clear what's stable.

See ``行情Live功能方案.md`` §4 for the full schema.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enums (encoded as plain strings on the wire; defined here so callers can
# import the constants instead of typing literals).
# ─────────────────────────────────────────────────────────────────────────────

# Channels
CH_TICK = "tick"
CH_BAR = "bar"
ALLOWED_CHANNELS = frozenset({CH_TICK, CH_BAR})

# Markets — Phase 1 only US. HK/CN deferred to Phase 2/3 (see design §3.3).
MK_US = "US"
MK_HK = "HK"
MK_CN = "CN"
ALLOWED_MARKETS_PHASE1 = frozenset({MK_US})
ALLOWED_MARKETS_FULL = frozenset({MK_US, MK_HK, MK_CN})

# Intervals
INT_1S = "1s"
INT_1M = "1m"
INT_5M = "5m"
INT_15M = "15m"
INT_1H = "1h"
INT_1D = "1d"
ALLOWED_INTERVALS = frozenset({INT_1S, INT_1M, INT_5M, INT_15M, INT_1H, INT_1D})

# Sources
SRC_IB = "ib"
SRC_YAHOO = "yahoo"
SRC_AKSHARE = "akshare"
ALLOWED_SOURCES = frozenset({SRC_IB, SRC_YAHOO, SRC_AKSHARE})

# Snapshot bar count cap (design §4.1).
SNAPSHOT_MAX_BARS = 500


# Client → server actions
ACT_SUBSCRIBE = "subscribe"
ACT_UNSUBSCRIBE = "unsubscribe"
ACT_PING = "ping"
ACT_RESEND = "resend"  # bar-only; tick has no replay buffer
ALLOWED_ACTIONS = frozenset({ACT_SUBSCRIBE, ACT_UNSUBSCRIBE, ACT_PING, ACT_RESEND})


# Server → client message types
T_SUBSCRIBED = "subscribed"
T_SNAPSHOT = "snapshot"
T_TICK = "tick"
T_BAR = "bar"
T_HB = "hb"
T_PONG = "pong"
T_SOURCE_CHANGED = "source_changed"
T_GAP = "gap"
T_ERROR = "error"


# Error code enumeration (closed set — design §4.3 v3 update).
class ErrorCode:
    UNAUTHORIZED = "unauthorized"
    INVALID_PAYLOAD = "invalid_payload"
    UNSUPPORTED_MARKET = "unsupported_market"
    UNSUPPORTED_INTERVAL = "unsupported_interval"
    UNKNOWN_SYMBOL = "unknown_symbol"
    SUBSCRIPTION_LIMIT_EXCEEDED = "subscription_limit_exceeded"
    UPSTREAM_DISCONNECTED = "upstream_disconnected"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"


ALL_ERROR_CODES = frozenset(
    v for k, v in ErrorCode.__dict__.items() if not k.startswith("_") and isinstance(v, str)
)


# Bar action (used on `type=bar` messages; not a wire-level enum but referenced
# in code).
BAR_UPDATE = "update"  # same bar, more data
BAR_NEW = "new"        # new bar started, old one is final


# ─────────────────────────────────────────────────────────────────────────────
# Sequence-number generator
#
# Per-(market, symbol, channel) monotonic, **shared across all clients** so
# that two clients subscribed to the same stream see the same seq numbers and
# can compare gap detection. SubscriptionManager owns one of these per stream.
# ─────────────────────────────────────────────────────────────────────────────


class SeqGen:
    """Thread-safe monotonic counter starting from 1."""

    __slots__ = ("_it", "_lock", "_last")

    def __init__(self, start: int = 0) -> None:
        # itertools.count is atomic for int next() in CPython but we still
        # need the lock for `last` readback. Pre-skip to ensure first next()
        # returns start+1.
        self._it = itertools.count(start + 1)
        self._lock = Lock()
        self._last = start

    def next(self) -> int:
        with self._lock:
            n = next(self._it)
            self._last = n
            return n

    @property
    def last(self) -> int:
        with self._lock:
            return self._last


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────


def now_ms() -> int:
    """Current wall-clock as 13-digit UTC ms epoch (matches `ts` field)."""
    return int(time.time() * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Payload validation
#
# Strict on inbound (client → server) — we close 401 / send error on bad
# input. Permissive on outbound (server → client) — assume our own code
# emits well-formed messages.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedAction:
    """A validated client request, ready to be dispatched."""
    action: str
    channel: Optional[str] = None
    market: Optional[str] = None
    symbol: Optional[str] = None
    interval: Optional[str] = None
    from_seq: Optional[int] = None
    to_seq: Optional[int] = None


@dataclass
class ValidationError(Exception):
    code: str
    msg: str

    def __str__(self) -> str:
        return f"{self.code}: {self.msg}"


def parse_client_message(
    raw: Any,
    *,
    allowed_markets: frozenset[str] = ALLOWED_MARKETS_PHASE1,
) -> ParsedAction:
    """Validate + normalize a single inbound message.

    Returns ParsedAction on success. Raises ValidationError otherwise.
    Caller is responsible for sending the error back to the client.

    `allowed_markets` lets the test rig flip to ALLOWED_MARKETS_FULL.
    Production callers should keep the Phase 1 default.
    """
    if not isinstance(raw, dict):
        raise ValidationError(ErrorCode.INVALID_PAYLOAD, "payload must be a JSON object")

    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValidationError(
            ErrorCode.INVALID_PAYLOAD,
            f"unknown action {action!r}; must be one of {sorted(ALLOWED_ACTIONS)}",
        )

    if action == ACT_PING:
        return ParsedAction(action=ACT_PING)

    # subscribe / unsubscribe / resend all need (channel, market, symbol)
    channel = raw.get("channel")
    if channel not in ALLOWED_CHANNELS:
        raise ValidationError(
            ErrorCode.INVALID_PAYLOAD,
            f"channel must be one of {sorted(ALLOWED_CHANNELS)}",
        )

    market = raw.get("market")
    if not isinstance(market, str):
        raise ValidationError(ErrorCode.INVALID_PAYLOAD, "market must be a string")
    market = market.upper()
    if market not in allowed_markets:
        raise ValidationError(
            ErrorCode.UNSUPPORTED_MARKET,
            f"market {market} not supported in this phase",
        )

    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValidationError(ErrorCode.INVALID_PAYLOAD, "symbol must be a non-empty string")
    symbol = symbol.strip().upper()

    # bar channel requires interval; tick channel must not have one
    interval = raw.get("interval")
    if channel == CH_BAR:
        if not isinstance(interval, str) or interval not in ALLOWED_INTERVALS:
            raise ValidationError(
                ErrorCode.UNSUPPORTED_INTERVAL,
                f"bar channel requires interval in {sorted(ALLOWED_INTERVALS)}",
            )
    else:  # CH_TICK
        if interval is not None:
            raise ValidationError(
                ErrorCode.INVALID_PAYLOAD, "tick channel must not include interval"
            )

    # resend extra fields
    from_seq = None
    to_seq = None
    if action == ACT_RESEND:
        if channel != CH_BAR:
            raise ValidationError(
                ErrorCode.INVALID_PAYLOAD,
                "resend is only supported on bar channel (tick has no replay buffer)",
            )
        from_seq = raw.get("from_seq")
        to_seq = raw.get("to_seq")
        if not isinstance(from_seq, int) or not isinstance(to_seq, int) or from_seq > to_seq:
            raise ValidationError(
                ErrorCode.INVALID_PAYLOAD,
                "resend requires integer from_seq <= to_seq",
            )

    return ParsedAction(
        action=action,
        channel=channel,
        market=market,
        symbol=symbol,
        interval=interval,
        from_seq=from_seq,
        to_seq=to_seq,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Server → client message constructors
#
# Wrap dict() construction in named functions so wire-format changes are easy
# to grep. All return plain dicts (json-serializable).
# ─────────────────────────────────────────────────────────────────────────────


def msg_subscribed(
    *, channel: str, market: str, symbol: str, source: str, last_seq: int,
    interval: Optional[str] = None,
) -> dict:
    out: dict = {
        "type": T_SUBSCRIBED,
        "channel": channel,
        "market": market,
        "symbol": symbol,
        "source": source,
        "ts": now_ms(),
        "last_seq": last_seq,
    }
    if interval is not None:
        out["interval"] = interval
    return out


def msg_snapshot_bars(
    *, market: str, symbol: str, interval: str, source: str, last_seq: int,
    bars: list[dict],
) -> dict:
    # Caller is responsible for capping bars to SNAPSHOT_MAX_BARS.
    return {
        "type": T_SNAPSHOT,
        "channel": CH_BAR,
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "ts": now_ms(),
        "last_seq": last_seq,
        "bars": bars[-SNAPSHOT_MAX_BARS:],
    }


def msg_tick(
    *, market: str, symbol: str, source: str, seq: int,
    last: Optional[float], bid: Optional[float], ask: Optional[float],
    volume: Optional[float], ts: Optional[int] = None,
) -> dict:
    out: dict = {
        "type": T_TICK,
        "market": market,
        "symbol": symbol,
        "source": source,
        "ts": ts if ts is not None else now_ms(),
        "seq": seq,
    }
    if last is not None: out["last"] = last
    if bid is not None: out["bid"] = bid
    if ask is not None: out["ask"] = ask
    if volume is not None: out["volume"] = volume
    return out


def msg_bar(
    *, market: str, symbol: str, interval: str, source: str, seq: int,
    action: str, o: float, h: float, l: float, c: float, v: float,
    ts: Optional[int] = None,
) -> dict:
    # action: BAR_UPDATE | BAR_NEW
    return {
        "type": T_BAR,
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "ts": ts if ts is not None else now_ms(),
        "seq": seq,
        "action": action,
        "o": o, "h": h, "l": l, "c": c, "v": v,
    }


def msg_hb() -> dict:
    return {"type": T_HB, "ts": now_ms()}


def msg_pong() -> dict:
    return {"type": T_PONG, "ts": now_ms()}


def msg_source_changed(*, market: str, symbol: str, from_src: str, to_src: str) -> dict:
    return {
        "type": T_SOURCE_CHANGED,
        "market": market,
        "symbol": symbol,
        "from": from_src,
        "to": to_src,
        "ts": now_ms(),
    }


def msg_gap(*, channel: str, market: str, symbol: str, from_seq: int, to_seq: int) -> dict:
    return {
        "type": T_GAP,
        "channel": channel,
        "market": market,
        "symbol": symbol,
        "from_seq": from_seq,
        "to_seq": to_seq,
        "ts": now_ms(),
    }


def msg_error(*, code: str, msg: str) -> dict:
    # caller's responsibility to use a valid ErrorCode constant; we still
    # accept arbitrary strings so tests can probe weird inputs
    return {"type": T_ERROR, "code": code, "msg": msg, "ts": now_ms()}


# ─────────────────────────────────────────────────────────────────────────────
# Stream key (market, symbol, channel[, interval])
#
# Used as dict key inside SubscriptionManager. Tuple comparison is O(1) and
# hashable.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StreamKey:
    market: str
    symbol: str
    channel: str
    interval: Optional[str] = None  # only for bar channel

    @classmethod
    def from_parsed(cls, p: ParsedAction) -> "StreamKey":
        # caller guarantees p has channel/market/symbol set
        return cls(
            market=p.market or "",
            symbol=p.symbol or "",
            channel=p.channel or "",
            interval=p.interval,
        )

    def __str__(self) -> str:
        if self.interval:
            return f"{self.market}:{self.symbol}:{self.channel}:{self.interval}"
        return f"{self.market}:{self.symbol}:{self.channel}"
