import logging
import os
import threading
import asyncio
import atexit
import time as _time
import queue as _queue
import multiprocessing as _mp
from datetime import datetime
from typing import Optional, List

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock, util
except Exception:  # pragma: no cover - optional dependency
    IB = None
    Stock = None
    util = None


# IB API market data types (reqMarketDataType): 1=live, 2=frozen,
# 3=delayed (15min), 4=delayed-frozen. We only use 1 and 3.
IB_MARKET_DATA_LIVE = 1
IB_MARKET_DATA_DELAYED = 3


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def ib_config() -> dict:
    primary_host = os.getenv("IB_GATEWAY_HOST", "127.0.0.1")
    fallback_host = os.getenv("IB_GATEWAY_FALLBACK_HOST", "127.0.0.1")
    primary_port = int(os.getenv("IB_GATEWAY_PORT", "4002"))
    fallback_port = int(os.getenv("IB_GATEWAY_FALLBACK_PORT", "4001"))
    # Try both ports on each host (4001 = live, 4002 = paper). readonly=True
    # on the connect call means we can't place orders even on live port —
    # quote-only access is safe. Without this, a misconfigured port on one
    # host kills the connection even when the other port is open.
    endpoints = []
    for host, port in ((primary_host, primary_port),
                       (primary_host, fallback_port),
                       (fallback_host, primary_port),
                       (fallback_host, fallback_port)):
        entry = {"host": host, "port": port}
        if entry not in endpoints:
            endpoints.append(entry)
    return {
        "enabled": _env_bool("IB_GATEWAY_ENABLED", True),
        "host": primary_host,
        "port": primary_port,
        "fallback_host": fallback_host,
        "fallback_port": fallback_port,
        "endpoints": endpoints,
        "client_id_base": int(os.getenv("IB_GATEWAY_CLIENT_ID", "77")),
        # Default live; fetch_quote auto-falls back to delayed when live
        # returns nothing. Set IB_MARKET_DATA_TYPE=3 to force delayed.
        "market_data_type": int(os.getenv("IB_MARKET_DATA_TYPE", str(IB_MARKET_DATA_LIVE))),
    }


def ib_available() -> bool:
    return IB is not None


def ib_enabled() -> bool:
    return ib_available() and ib_config()["enabled"]


def ib_supported_market(market: str) -> bool:
    return str(market or "").upper() in {"US", "HK"}


def _request_timeout(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    try:
        value = float(raw) if raw is not None else float(default)
    except Exception:
        value = float(default)
    return max(1.0, value)


def _contract_for(symbol: str, market: str):
    mkt = str(market or "").upper()
    sym = str(symbol or "").upper()
    if mkt == "US":
        return Stock(sym, "SMART", "USD")
    if mkt == "HK":
        return Stock(sym.zfill(4), "SEHK", "HKD")
    raise ValueError(f"IB market not supported: {mkt}")


# ─────────────────────────────────────────────────────────────────────────────
# IB Client Pool
#
# Each slot owns one IB API connection in a dedicated background thread with
# its own asyncio event loop. Callers acquire a slot via _SlotContext, submit
# coroutines via slot.submit(), then release.
#
# IB Gateway has a hard 32 concurrent client cap shared across all processes
# connecting to it. Default IB_POOL_SIZE=16 leaves headroom for other
# local clients (e.g. TWS itself) plus the scanner subprocesses (8 by default — each
# uses 1 client).
#
# Spawned subprocess workers detect non-main process and force pool_size=1 —
# they rarely need IB and shouldn't burn 16 slots × N_workers.
# ─────────────────────────────────────────────────────────────────────────────


def _default_pool_size() -> int:
    """Pool size for this process. Subprocess workers cap to 1."""
    # 只把 ProcessPoolExecutor 的 worker 降为 1 slot (worker 由 initializer
    # proc_role.mark_pool_worker 显式打标); uvicorn --reload 的应用子进程
    # (name="SpawnProcess-1", 与 executor worker 同名) 跑的是完整 FastAPI
    # 主服务, 需要正常 pool — 所以不能靠进程名猜.
    try:
        import proc_role
        if proc_role.is_pool_worker():
            return 1
    except Exception:
        pass
    return max(1, int(os.getenv("IB_POOL_SIZE", "16")))


def _streaming_slot_count() -> int:
    """How many IB clients to reserve for the live-streaming subsystem.

    Default 0 = no reservation (legacy behavior unchanged). When
    LIVE_STREAMING_ENABLED=1, the streaming module sets
    IB_STREAMING_SLOTS=2 by default (overridable). These slots are
    deducted from the pool size — total IB clients == IB_POOL_SIZE,
    of which (IB_POOL_SIZE - IB_STREAMING_SLOTS) are scan/quote and
    IB_STREAMING_SLOTS are held by streaming.
    """
    # Only the app process; executor workers never do streaming.
    # (uvicorn --reload 的应用子进程也算 app 进程, 见 _default_pool_size)
    try:
        import proc_role
        if proc_role.is_pool_worker():
            return 0
    except Exception:
        pass
    if os.getenv("LIVE_STREAMING_ENABLED", "0").lower() in ("0", "false", "no", "off", ""):
        return 0
    return max(0, int(os.getenv("IB_STREAMING_SLOTS", "2")))


class _IBSlot:
    """One IB connection + its own asyncio loop running in a dedicated thread.

    Callers submit coroutines via .submit(coro_factory) — the factory is
    invoked with the underlying IB instance on the slot's loop thread. The
    IB instance must never be used from outside this slot's loop thread
    (ib_insync binds its asyncio tasks to one loop).
    """

    def __init__(self, client_id: int, endpoints: List[dict], market_data_type: int):
        self.client_id = client_id
        self.endpoints = endpoints
        self.market_data_type = market_data_type
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ib: Optional["IB"] = None
        self.connected_endpoint: Optional[dict] = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"ib-slot-{client_id}",
        )
        self._thread.start()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        ib = IB()
        connected = False
        for ep in self.endpoints:
            try:
                ib.connect(ep["host"], ep["port"], clientId=self.client_id,
                           readonly=True, timeout=2)
                try:
                    ib.reqMarketDataType(self.market_data_type)
                except Exception:
                    pass
                self.connected_endpoint = ep
                self.ib = ib
                connected = True
                break
            except Exception:
                try:
                    if ib.isConnected():
                        ib.disconnect()
                except Exception:
                    pass
        self._ready.set()
        if connected:
            try:
                self.loop.run_forever()
            except Exception:
                pass
        # run_forever returned (stop() called) OR connect failed and we never
        # ran. Either way this thread is ending — CLOSE the loop so its kqueue
        # FD + self-pipe socketpair are released. Not closing here leaked ~1
        # kqueue + 2 unix fds PER slot, which the maintenance thread multiplied
        # into FD exhaustion (Errno 24) when IB was offline and slots were
        # rebuilt every 60s. (2026-07-06)
        try:
            # Neuter IB.__del__ first: if the loop closes before GC reaps the
            # IB instance, its finalizer tries to schedule disconnect on the
            # closed loop and spews tracebacks.
            if self.ib is not None:
                try:
                    self.ib.disconnect = lambda: None
                except Exception:
                    pass
            self.loop.close()
        except Exception:
            pass

    def wait_ready(self, timeout: float = 15.0) -> bool:
        return self._ready.wait(timeout=timeout)

    def is_ok(self) -> bool:
        return self.ib is not None and self.ib.isConnected()

    def submit(self, coro_factory, timeout: float = 10.0):
        """Run a coroutine on this slot's loop and return the result.

        coro_factory: callable taking the IB instance, returning a coroutine.
        """
        if not self.is_ok():
            raise RuntimeError(f"slot {self.client_id} not connected")
        coro = coro_factory(self.ib)
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def stop(self):
        if self.ib and self.loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.ib.disconnectAsync(), self.loop)
                fut.result(timeout=3)
            except Exception:
                pass
        if self.loop:
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass
        # Prevent ib_insync's IB.__del__ from re-running disconnect during
        # interpreter shutdown — that path schedules a callback on the loop,
        # which CPython closes before reaping IB instances, producing dozens
        # of "Event loop is closed" tracebacks at exit time.
        if self.ib is not None:
            try:
                self.ib.disconnect = lambda: None
            except Exception:
                pass


_pool: Optional[_queue.Queue] = None
_slots: List[_IBSlot] = []
_pool_lock = threading.Lock()
_pool_pid: Optional[int] = None
_pool_atexit_registered = False

# 2026-05-20: streaming subsystem reserves a few slots out of the pool so the
# WS endpoints get their own long-held IB connections (reqTickByTickData /
# reqMktData are persistent subscriptions, not request-response). These
# slots are *not* put on the pool queue — scan/quote callers can't see them.
# See 行情Live功能方案.md §2.2.
_streaming_slots: List[_IBSlot] = []


def _shutdown_pool_unlocked():
    global _pool, _slots, _pool_pid
    for s in _slots:
        try:
            s.stop()
        except Exception:
            pass
    _slots = []
    _pool = None
    _pool_pid = None


def _shutdown_pool():
    with _pool_lock:
        _shutdown_pool_unlocked()


def _ensure_pool() -> _queue.Queue:
    global _pool, _slots, _pool_pid, _pool_atexit_registered
    if not ib_enabled():
        raise RuntimeError("IB Gateway is disabled or ib_insync is unavailable")
    pid = os.getpid()
    if _pool is not None and _pool_pid == pid:
        return _pool
    with _pool_lock:
        if _pool is not None and _pool_pid == pid:
            return _pool
        if _pool is not None and _pool_pid != pid:
            # Pool inherited from another pid (post-fork). Reset.
            _shutdown_pool_unlocked()

        cfg = ib_config()
        pool_size = _default_pool_size()
        streaming_count = _streaming_slot_count()
        # 32-wide clientId bucket per pid so workers (pool_size=1) and the
        # main pool never collide, even when they happen to share a pid
        # modulo 1000.
        base = cfg["client_id_base"] + (pid % 1000) * 32

        slots = [
            _IBSlot(
                client_id=base + i,
                endpoints=cfg["endpoints"],
                market_data_type=cfg["market_data_type"],
            )
            for i in range(pool_size)
        ]
        for s in slots:
            s.wait_ready(timeout=15.0)

        ok = [s for s in slots if s.is_ok()]
        if not ok:
            for s in slots:
                try:
                    s.stop()
                except Exception:
                    pass
            raise RuntimeError(
                f"IB pool: 0/{pool_size} slots connected to any of "
                f"{cfg['endpoints']}")

        # Split the OK slots: first `streaming_count` are reserved for the
        # streaming subsystem (held outside the queue), the rest go on the
        # pool. We choose the first slots (lowest client_id) for streaming
        # purely for deterministic logging.
        global _streaming_slots
        if streaming_count > 0:
            reserve = min(streaming_count, len(ok))
            _streaming_slots = ok[:reserve]
            pool_slots = ok[reserve:]
            logger.info(
                "[ib_gateway] reserved %d/%d slots for streaming "
                "(client_ids=%s); %d on pool",
                len(_streaming_slots), pool_size,
                [s.client_id for s in _streaming_slots], len(pool_slots),
            )
        else:
            pool_slots = ok

        q: _queue.Queue = _queue.Queue()
        for s in pool_slots:
            q.put(s)

        _slots = slots
        _pool = q
        _pool_pid = pid
        if not _pool_atexit_registered:
            atexit.register(_shutdown_pool)
            _pool_atexit_registered = True
        _start_pool_maintenance_unlocked()
    return _pool


# ─────────────────────────────────────────────────────────────────────────────
# Pool self-healing (2026-07-06)
#
# Previously, broken slots were simply dropped by _SlotContext.__exit__ and
# never replaced — after enough IB Gateway hiccups (daily restart, network
# blips: we observed 192× Error 1100 in one day) the pool shrank to zero and
# stayed dead until a full process restart. The maintenance thread below scans
# every IB_POOL_MAINTENANCE_S seconds, rebuilds dead slots OUTSIDE the lock
# (connect can take seconds), then swaps them in: pool slots go back on the
# queue, streaming slots are replaced in-place in _streaming_slots (the
# streaming bridge re-binds lazily via get_streaming_slots()).
# ─────────────────────────────────────────────────────────────────────────────

_maintenance_thread: Optional[threading.Thread] = None


def _start_pool_maintenance_unlocked() -> None:
    global _maintenance_thread
    if _maintenance_thread is not None and _maintenance_thread.is_alive():
        return
    _maintenance_thread = threading.Thread(
        target=_pool_maintenance_loop, daemon=True, name="ib-pool-maintenance",
    )
    _maintenance_thread.start()


def _gateway_reachable(cfg: dict, timeout: float = 0.3) -> bool:
    """Cheap TCP probe of the IB Gateway API port — no ib_insync handshake."""
    import socket as _socket
    for host, port in ((cfg["host"], cfg["port"]),
                       (cfg["fallback_host"], cfg["fallback_port"])):
        try:
            with _socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            continue
    return False


def _pool_maintenance_loop() -> None:
    base_interval = max(15.0, float(os.getenv("IB_POOL_MAINTENANCE_S", "60")))
    pid = os.getpid()
    backoff = 1  # multiplier while the gateway is unreachable (capped)
    while True:
        _time.sleep(base_interval * backoff)
        try:
            if _pool is None or _pool_pid != pid:
                continue
            # Snapshot dead slots under the lock, rebuild outside it.
            with _pool_lock:
                dead = [(i, s) for i, s in enumerate(_slots) if not s.is_ok()]
            if not dead:
                backoff = 1
                continue
            cfg = ib_config()
            # If the gateway port isn't even listening (IB restart / weekend
            # maintenance), don't spawn 12 threads × 10s wait_ready to fail —
            # back off exponentially (up to 8× the base interval). Spawning
            # doomed slots was the FD-churn amplifier.
            if not _gateway_reachable(cfg):
                backoff = min(backoff * 2, 8)
                logger.info("[ib_gateway] maintenance: gateway unreachable, "
                            "backing off to %.0fs (%d dead slots waiting)",
                            base_interval * backoff, len(dead))
                continue
            backoff = 1
            logger.warning("[ib_gateway] maintenance: %d dead slot(s), attempting revive: %s",
                           len(dead), [s.client_id for _, s in dead])
            for i, old in dead:
                repl = _IBSlot(
                    client_id=old.client_id,
                    endpoints=cfg["endpoints"],
                    market_data_type=cfg["market_data_type"],
                )
                repl.wait_ready(timeout=10.0)
                if not repl.is_ok():
                    try:
                        repl.stop()
                    except Exception:
                        pass
                    continue  # IB still down; retry next sweep
                with _pool_lock:
                    # Re-check the slot wasn't already swapped by another path.
                    if i >= len(_slots) or _slots[i] is not old:
                        try:
                            repl.stop()
                        except Exception:
                            pass
                        continue
                    try:
                        old.stop()
                    except Exception:
                        pass
                    _slots[i] = repl
                    swapped_streaming = False
                    for j, s in enumerate(_streaming_slots):
                        if s is old:
                            _streaming_slots[j] = repl
                            swapped_streaming = True
                            break
                    if not swapped_streaming and _pool is not None:
                        # Pool slot: old one is either already dropped
                        # (never returned to queue when dead) or will be
                        # dropped by its holder at __exit__; queue the fresh
                        # replacement now.
                        _pool.put(repl)
                logger.info("[ib_gateway] maintenance: revived %s slot client_id=%d",
                            "streaming" if swapped_streaming else "pool", repl.client_id)
        except Exception:
            logger.exception("[ib_gateway] maintenance sweep failed; will retry")


class _SlotContext:
    """Acquire a slot from the pool; return it on exit (unless it died)."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.slot: Optional[_IBSlot] = None

    def __enter__(self) -> _IBSlot:
        pool = _ensure_pool()
        try:
            self.slot = pool.get(timeout=self.timeout)
        except _queue.Empty:
            raise RuntimeError(
                f"IB pool: no slot available within {self.timeout}s")
        return self.slot

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.slot is not None and _pool is not None:
            # Return healthy slot to pool. Broken slots are dropped — the
            # pool just runs at reduced capacity until restart.
            if self.slot.is_ok():
                _pool.put(self.slot)
        return False


def get_streaming_slots() -> List["_IBSlot"]:
    """Return the IB slots reserved for the streaming subsystem.

    Caller (streaming.ib_stream) should call this *after* _ensure_pool()
    has run — usually by calling _ensure_pool() once at app startup. The
    returned list is the same object each call (slots are long-held).

    Returns [] when streaming is disabled or no slots were reserved.
    """
    # Ensure the pool has been built. We don't want to silently return
    # an empty list just because nobody triggered pool init yet.
    try:
        _ensure_pool()
    except Exception as exc:
        logger.warning("[ib_gateway] _ensure_pool failed during get_streaming_slots: %r", exc)
        return []
    return list(_streaming_slots)


def ib_status_light() -> dict:
    """Read-only, side-effect-free pool snapshot for health endpoints.

    ib_status() calls _ensure_pool(), which — when IB Gateway is down —
    synchronously retries building all 12 slots (2s connect timeout × 2
    endpoints × 12 slots ≈ 30-60s). That made /ops/health itself time out
    whenever IB was offline. This variant only inspects existing state and
    does a 200ms TCP probe of the primary endpoint.
    """
    cfg = ib_config()
    ok_slots = [s for s in _slots if s.is_ok()]
    gateway_listening = False
    try:
        import socket as _socket
        with _socket.create_connection((cfg["host"], cfg["port"]), timeout=0.2):
            gateway_listening = True
    except Exception:
        pass
    return {
        "enabled": cfg["enabled"],
        "available": ib_available(),
        "gateway_listening": gateway_listening,
        "pool_initialized": _pool is not None,
        "pool_size_target": _default_pool_size(),
        "pool_size_actual": len(ok_slots),
        "connected": len(ok_slots) > 0,
        "streaming_slots_reserved": len(_streaming_slots),
    }


def ib_status() -> dict:
    cfg = ib_config()
    status = {
        "enabled": cfg["enabled"],
        "available": ib_available(),
        "host": cfg["host"],
        "port": cfg["port"],
        "fallback_host": cfg["fallback_host"],
        "fallback_port": cfg["fallback_port"],
        "endpoints": cfg["endpoints"],
        "client_id_base": cfg["client_id_base"],
        "market_data_type": cfg["market_data_type"],
        "pid": os.getpid(),
        "pool_size_target": _default_pool_size(),
        "connected": False,
    }
    if not ib_enabled():
        return status
    try:
        _ensure_pool()
        ok_slots = [s for s in _slots if s.is_ok()]
        status["connected"] = len(ok_slots) > 0
        status["pool_size_actual"] = len(ok_slots)
        status["pool_idle"] = _pool.qsize() if _pool else 0
        status["streaming_slots_reserved"] = len(_streaming_slots)
        status["streaming_slot_client_ids"] = [s.client_id for s in _streaming_slots]
        if ok_slots:
            status["active_endpoint"] = ok_slots[0].connected_endpoint
            status["slot_client_ids"] = [s.client_id for s in ok_slots]
    except Exception as e:
        status["error"] = str(e)
    return status


def _duration_from_period(period: Optional[str] = None, start: Optional[str] = None) -> str:
    if start:
        start_ts = pd.Timestamp(start)
        days = max(2, int((pd.Timestamp.now() - start_ts).total_seconds() / 86400) + 3)
        return f"{days} D"

    p = str(period or "1mo").strip().lower()
    if p == "max":
        return "15 Y"
    if p == "ytd":
        year_start = pd.Timestamp(datetime.now().year, 1, 1)
        days = max(2, int((pd.Timestamp.now() - year_start).total_seconds() / 86400) + 3)
        return f"{days} D"
    if p.endswith("d"):
        try:
            return f"{max(2, int(p[:-1]) + 2)} D"
        except Exception:
            return "30 D"

    mapping = {
        "1mo": "45 D",
        "3mo": "120 D",
        "6mo": "240 D",
        "1y": "2 Y",
        "2y": "3 Y",
        "5y": "6 Y",
        "10y": "10 Y",
    }
    return mapping.get(p, "30 D")


def _bar_size_for_interval(interval: str) -> str:
    mapping = {
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1h": "1 hour",
        "1d": "1 day",
        "1wk": "1 week",
        "1mo": "1 month",
    }
    if interval not in mapping:
        raise ValueError(f"IB interval not supported: {interval}")
    return mapping[interval]


def fetch_history(symbol: str, market: str, interval: str, period: str = None, start: str = None) -> pd.DataFrame:
    if not ib_supported_market(market):
        return pd.DataFrame()
    contract = _contract_for(symbol, market)
    duration = _duration_from_period(period=period, start=start)
    bar_size = _bar_size_for_interval(interval)
    hist_timeout = _request_timeout("IB_HISTORY_TIMEOUT_SEC", 8)

    with _SlotContext() as slot:
        async def _do(ib):
            await ib.qualifyContractsAsync(contract)
            return await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
                timeout=hist_timeout,
            )
        try:
            bars = slot.submit(_do, timeout=hist_timeout + 2)
        except Exception:
            return pd.DataFrame()

    if not bars:
        return pd.DataFrame()

    df = util.df(bars)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    keep_cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df.index.name = "date"
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[c for c in ("open", "high", "low", "close") if c in df.columns])


def _ticker_has_price(t) -> bool:
    """Has at least one usable price field (not None and not NaN)."""
    for f in (getattr(t, "last", None), getattr(t, "close", None),
              getattr(t, "bid", None), getattr(t, "ask", None)):
        if f is not None and not pd.isna(f):
            return True
    return False


def _fetch_quote_with_slot(slot: "_IBSlot", symbol: str, market: str) -> Optional[dict]:
    """Run the actual reqTickers + delayed-fallback dance against a slot the
    caller already holds. Separated from fetch_quote so the non-blocking
    try_fetch_quote variant (used by quote_router for load balancing) can
    share the body without re-implementing the live→delayed logic."""
    cfg = ib_config()
    contract = _contract_for(symbol, market)
    endpoint = slot.connected_endpoint

    async def _do(ib):
        await ib.qualifyContractsAsync(contract)
        return await ib.reqTickersAsync(contract)
    try:
        tickers = slot.submit(_do, timeout=10)
    except Exception:
        return None
    if not tickers:
        return None
    t = tickers[0]
    try:
        market_price = t.marketPrice()
    except Exception:
        market_price = None

    if cfg["market_data_type"] == IB_MARKET_DATA_LIVE:
        live_empty = (
            not _ticker_has_price(t)
            and (market_price is None or pd.isna(market_price))
        )
        if live_empty:
            async def _fb(ib):
                ib.reqMarketDataType(IB_MARKET_DATA_DELAYED)
                try:
                    return await ib.reqTickersAsync(contract)
                finally:
                    try:
                        ib.reqMarketDataType(cfg["market_data_type"])
                    except Exception:
                        pass
            try:
                tickers2 = slot.submit(_fb, timeout=10)
                if tickers2:
                    t = tickers2[0]
                    try:
                        market_price = t.marketPrice()
                    except Exception:
                        pass
            except Exception:
                pass

    last = float(t.last) if t.last is not None and not pd.isna(t.last) else None
    close = float(t.close) if t.close is not None and not pd.isna(t.close) else None
    bid = float(t.bid) if t.bid is not None and not pd.isna(t.bid) else None
    ask = float(t.ask) if t.ask is not None and not pd.isna(t.ask) else None
    market_price = float(market_price) if market_price is not None and not pd.isna(market_price) else None
    price = last or market_price or bid or ask or close
    change_pct = None
    if price and close:
        try:
            change_pct = round((price - close) / close * 100, 2)
        except Exception:
            change_pct = None
    return {
        "symbol": str(symbol).upper(),
        "market": str(market).upper(),
        "price": price,
        "prev_close": close,
        "change_pct": change_pct,
        "high": float(t.high) if t.high is not None and not pd.isna(t.high) else None,
        "low": float(t.low) if t.low is not None and not pd.isna(t.low) else None,
        "bid": bid,
        "ask": ask,
        "volume": float(t.volume) if t.volume is not None and not pd.isna(t.volume) else None,
        "currency": getattr(contract, "currency", None),
        "source": "IB Gateway",
        "ib_endpoint": endpoint,
        "ts": datetime.utcnow().isoformat(),
    }


def try_fetch_quote(symbol: str, market: str) -> Optional[dict]:
    """Non-blocking variant of fetch_quote. Returns None if the IB pool has
    no idle slot right now (caller should try a different provider). Used
    by quote_router for IB → yfinance overflow when the pool saturates."""
    if not ib_supported_market(market) or not ib_enabled():
        return None
    try:
        pool = _ensure_pool()
    except Exception:
        return None
    try:
        slot = pool.get(block=False)
    except _queue.Empty:
        return None
    try:
        return _fetch_quote_with_slot(slot, symbol, market)
    finally:
        if slot.is_ok():
            pool.put(slot)


def fetch_quote(symbol: str, market: str) -> Optional[dict]:
    if not ib_supported_market(market):
        return None
    with _SlotContext() as slot:
        return _fetch_quote_with_slot(slot, symbol, market)
