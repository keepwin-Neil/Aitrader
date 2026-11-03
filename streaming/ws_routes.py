"""WebSocket endpoints and the streaming app-state lifecycle.

Three endpoints (all only attached when LIVE_STREAMING_ENABLED=1):

  - GET /ws/quote/{market}/{symbol}?token=...
      Subscribes the client to tick events for (market, symbol). Client can
      then `unsubscribe` and `subscribe` other (market, symbol) pairs on the
      same connection — the URL path is only an initial convenience.

  - GET /ws/kline/{market}/{symbol}/{interval}?token=...
      Same idea for bar channel.

  - GET /ws/stats
      Localhost-only JSON debug dump.

Auth: ?token= matched against env LIVE_STREAMING_TOKEN (constant-time).
Network: the app binds to 127.0.0.1; /ws/stats additionally checks client.host.

See 行情Live功能方案.md §2.1, §4.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException, status

from streaming import protocol as proto
from streaming.sub_manager import SubscriptionManager, SubscriptionLimitExceeded
from streaming.ib_stream import IBStreamBridge, UnknownSymbol, UpstreamDisconnected
from streaming.yahoo_poller import YahooPoller

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Singletons (initialized via init_streaming() from main.py startup)
# ─────────────────────────────────────────────────────────────────────────────

_sub_manager: Optional[SubscriptionManager] = None
_ib_bridge: Optional[IBStreamBridge] = None
_yahoo_poller: Optional[YahooPoller] = None


def init_streaming() -> SubscriptionManager:
    """Build the singletons. Called from main.py @app.on_event('startup').

    Phase 2: also spawns YahooPoller (idle by default; activates on IB
    failover) and wires connection-state callbacks.
    """
    global _sub_manager, _ib_bridge, _yahoo_poller
    if _sub_manager is not None:
        return _sub_manager
    _sub_manager = SubscriptionManager()
    _ib_bridge = IBStreamBridge(_sub_manager)
    # Wire the upstream fns BEFORE starting the bridge/poller so no early WS
    # subscribe can slip through while _subscribe_fn is still None (silently
    # "subscribed" with no upstream = ghost stream).
    _sub_manager._subscribe_fn = _ib_bridge.subscribe
    _sub_manager._unsubscribe_fn = _ib_bridge.unsubscribe
    _ib_bridge.start()
    _yahoo_poller = YahooPoller(_sub_manager)
    _yahoo_poller.start()

    # Failover wiring: when IB drops, take every IB-sourced key and hand it
    # to the Yahoo poller. When IB comes back, reverse AND re-subscribe on
    # the IB side — the bridge cleared its dead handles at disconnect, so
    # without an explicit re-subscribe the streams would flip back to
    # source="ib" but receive no IB data (zombie streams).
    def _on_ib_disconnect():
        if _sub_manager is None or _yahoo_poller is None:
            return
        switched = _sub_manager.switch_source_all(from_src=proto.SRC_IB, to_src=proto.SRC_YAHOO)
        for k in switched:
            _yahoo_poller.add_key(k)
        log.warning("[streaming] failover: %d streams now sourced from Yahoo", len(switched))

    def _on_ib_reconnect():
        if _sub_manager is None or _yahoo_poller is None or _ib_bridge is None:
            return
        switched = _sub_manager.switch_source_all(from_src=proto.SRC_YAHOO, to_src=proto.SRC_IB)
        for k in switched:
            _yahoo_poller.remove_key(k)

        async def _resub(keys):
            for k in keys:
                try:
                    await _ib_bridge.subscribe(k)
                except Exception:
                    log.exception("[streaming] IB re-subscribe failed for %s", k)

        if switched:
            asyncio.create_task(_resub(switched))
        log.info("[streaming] failover: %d streams switched back to IB (re-subscribing)",
                 len(switched))

    _ib_bridge.on_ib_disconnect = _on_ib_disconnect
    _ib_bridge.on_ib_reconnect = _on_ib_reconnect
    _sub_manager.start_janitor()
    log.info("[streaming] initialized (IB + Yahoo failover armed)")
    return _sub_manager


async def shutdown_streaming() -> None:
    if _sub_manager is not None:
        await _sub_manager.stop_janitor()
    if _ib_bridge is not None:
        await _ib_bridge.stop()
    if _yahoo_poller is not None:
        _yahoo_poller.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────


def _expected_token() -> str:
    return os.environ.get("LIVE_STREAMING_TOKEN", "")


def _check_token(provided: Optional[str]) -> bool:
    expected = _expected_token()
    if not expected:
        # Misconfigured deployment — refuse to accept anyone rather than open up.
        return False
    if not provided:
        return False
    return hmac.compare_digest(expected, provided)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket loop helpers
# ─────────────────────────────────────────────────────────────────────────────

# Heartbeat: server pushes hb every 15s; client should ping every 15s. We
# read with a 30s timeout — beyond that we consider the connection dead.
SERVER_HB_INTERVAL_S = 15.0
CLIENT_READ_TIMEOUT_S = 30.0


async def _send_loop(ws: WebSocket, handle, stop_evt: asyncio.Event) -> None:
    """Drain client.queue → WebSocket send."""
    while not stop_evt.is_set():
        try:
            msg = await asyncio.wait_for(handle.queue.get(), timeout=SERVER_HB_INTERVAL_S)
            await ws.send_text(json.dumps(msg, separators=(",", ":")))
        except asyncio.TimeoutError:
            # No data in 15s — push heartbeat
            try:
                await ws.send_text(json.dumps(proto.msg_hb(), separators=(",", ":")))
            except Exception:
                return
        except Exception:
            log.exception("[ws] send loop error")
            return


async def _recv_loop(
    ws: WebSocket, handle, sub_mgr: SubscriptionManager, stop_evt: asyncio.Event
) -> None:
    """Read frames from client, validate, dispatch."""
    while not stop_evt.is_set():
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=CLIENT_READ_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.info("[ws] client %d read timeout — closing", handle.client_id)
            return
        except WebSocketDisconnect:
            return
        except Exception:
            log.exception("[ws] recv error")
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await _send_error(ws, proto.ErrorCode.INVALID_PAYLOAD, "not valid JSON")
            continue

        try:
            parsed = proto.parse_client_message(payload)
        except proto.ValidationError as ve:
            await _send_error(ws, ve.code, ve.msg)
            continue

        try:
            await _dispatch(ws, handle, sub_mgr, parsed)
        except SubscriptionLimitExceeded as e:
            await _send_error(ws, proto.ErrorCode.SUBSCRIPTION_LIMIT_EXCEEDED, str(e))
        except UnknownSymbol as e:
            await _send_error(ws, proto.ErrorCode.UNKNOWN_SYMBOL, str(e))
        except UpstreamDisconnected as e:
            await _send_error(ws, proto.ErrorCode.UPSTREAM_DISCONNECTED, str(e))
        except Exception:
            log.exception("[ws] dispatch failed")
            await _send_error(ws, proto.ErrorCode.INTERNAL_ERROR, "server bug")


async def _dispatch(
    ws: WebSocket, handle, sub_mgr: SubscriptionManager, p: proto.ParsedAction
) -> None:
    if p.action == proto.ACT_PING:
        handle.send_nowait(proto.msg_pong())
        return

    if p.action == proto.ACT_SUBSCRIBE:
        key = proto.StreamKey.from_parsed(p)
        st = await sub_mgr.subscribe(handle, key)
        # Tell the client the subscription is live + give them last_seq baseline
        handle.send_nowait(proto.msg_subscribed(
            channel=key.channel, market=key.market, symbol=key.symbol,
            source=st.source, last_seq=st.seq.last, interval=key.interval,
        ))
        # For bar channel, send a snapshot if we have one buffered
        if key.channel == proto.CH_BAR and st.last_bar_dict is not None:
            handle.send_nowait(proto.msg_snapshot_bars(
                market=key.market, symbol=key.symbol, interval=key.interval or "",
                source=st.source, last_seq=st.seq.last,
                bars=[st.last_bar_dict],
            ))
        return

    if p.action == proto.ACT_UNSUBSCRIBE:
        key = proto.StreamKey.from_parsed(p)
        await sub_mgr.unsubscribe(handle, key)
        return

    if p.action == proto.ACT_RESEND:
        # Phase 1: we don't keep a bar history buffer — tell the client to
        # walk away and use HTTP /kline. (Doc §4.2.)
        await _send_error(
            ws, proto.ErrorCode.INVALID_PAYLOAD,
            "resend not implemented in Phase 1; fall back to HTTP /kline snapshot",
        )
        return


async def _send_error(ws: WebSocket, code: str, msg: str) -> None:
    try:
        await ws.send_text(json.dumps(proto.msg_error(code=code, msg=msg), separators=(",", ":")))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


router = APIRouter()


async def _ws_session(
    ws: WebSocket, token: Optional[str], initial_action: Optional[proto.ParsedAction],
) -> None:
    """Common WS loop: accept, auth, run send + recv until disconnect."""
    if not _check_token(token):
        # FastAPI WebSocket: must accept before close to send a reason
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="unauthorized")
        return

    if _sub_manager is None:
        await ws.close(code=status.WS_1011_INTERNAL_ERROR, reason="not initialized")
        return

    await ws.accept()
    handle = _sub_manager.add_client()

    # If the URL implied a specific subscription (subscribe via path), do it
    # before entering the loop so the very first thing the client sees is a
    # subscribed/snapshot frame.
    if initial_action is not None:
        try:
            await _dispatch(ws, handle, _sub_manager, initial_action)
        except Exception:
            log.exception("[ws] initial subscribe failed")

    stop_evt = asyncio.Event()
    send_t = asyncio.create_task(_send_loop(ws, handle, stop_evt))
    recv_t = asyncio.create_task(_recv_loop(ws, handle, _sub_manager, stop_evt))

    try:
        # First task to finish ends the session (either disconnect or error)
        await asyncio.wait({send_t, recv_t}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_evt.set()
        for t in (send_t, recv_t):
            t.cancel()
        try:
            await _sub_manager.remove_client(handle)
        except Exception:
            log.exception("[ws] remove_client failed")
        try:
            await ws.close()
        except Exception:
            pass


@router.websocket("/ws/quote/{market}/{symbol}")
async def ws_quote(ws: WebSocket, market: str, symbol: str, token: Optional[str] = None):
    """Subscribe to tick channel for (market, symbol). Connection is upgradable
    to additional subscriptions via JSON messages.
    """
    try:
        initial = proto.parse_client_message({
            "action": "subscribe", "channel": "tick",
            "market": market, "symbol": symbol,
        })
    except proto.ValidationError as ve:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason=ve.code)
        return
    await _ws_session(ws, token, initial)


@router.websocket("/ws/kline/{market}/{symbol}/{interval}")
async def ws_kline(
    ws: WebSocket, market: str, symbol: str, interval: str, token: Optional[str] = None,
):
    """Subscribe to bar channel for (market, symbol, interval)."""
    try:
        initial = proto.parse_client_message({
            "action": "subscribe", "channel": "bar",
            "market": market, "symbol": symbol, "interval": interval,
        })
    except proto.ValidationError as ve:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason=ve.code)
        return
    await _ws_session(ws, token, initial)


@router.get("/ws/stats", include_in_schema=False)
async def ws_stats(request: Request):
    """Debug stats. Localhost-only (LAN attackers shouldn't be able to map
    subscriptions even with a token leak)."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="localhost only")
    if _sub_manager is None:
        return {"initialized": False}
    base = _sub_manager.stats()
    # Annotate with IB streaming slot info. ib_status() already returns the
    # reserved-slot info as public fields; no need to reach into the private
    # _streaming_slots list directly.
    try:
        from ib_gateway import ib_status
        ibs = ib_status()
        base["ib"] = {
            "connected": ibs.get("connected"),
            "streaming_slots_reserved": ibs.get("streaming_slots_reserved"),
            "streaming_slot_client_ids": ibs.get("streaming_slot_client_ids"),
        }
    except Exception:
        base["ib"] = {"error": "ib_gateway introspect failed"}
    # Yahoo poller stats
    if _yahoo_poller is not None:
        base["yahoo"] = _yahoo_poller.stats()
    else:
        base["yahoo"] = {"initialized": False}
    return base
