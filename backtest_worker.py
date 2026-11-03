"""Background backtest worker daemon (runs on this machine).

Long-running process that polls backtest_queue, claims tasks, runs them
(calling backtest.run_strategy_backtest), and writes results back.

Single Python file, no shell-specific deps. Standard library + pandas +
psycopg2 + requests.

Usage:
    python backtest_worker.py
    python backtest_worker.py --worker-id local-1 --concurrency 4

Env vars (set in .env before launch):
    DATABASE_URL                 PG connection (本机)
    AITRADER_INTERNAL_SECRET        token for /internal/ohlcv
    AITRADER_OHLCV_PROXY            default http://127.0.0.1:9000
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

# Local imports — work regardless of the current working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_queue

# OHLCV proxy URL — the main app on this machine.
_DEFAULT_OHLCV_URL = os.environ.get("AITRADER_OHLCV_PROXY", "http://127.0.0.1:9000")
_OHLCV_TOKEN = os.environ.get("AITRADER_INTERNAL_SECRET", "")

# Polling cadence (seconds)
DEFAULT_POLL_INTERVAL = 1.0
HEARTBEAT_INTERVAL_S = 30.0
RECLAIM_INTERVAL_S = 60.0
RECLAIM_AGE_S = 300

logging.basicConfig(
    level=os.environ.get("BACKTEST_WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backtest_worker")


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV fetcher (TTL cache per worker process)
#
# Was @lru_cache — but lru_cache never expires, so a worker running for days
# kept serving week-old prices to backtests. Now a TTL cache: entries expire
# after BACKTEST_OHLCV_TTL_S (default 1h), capped at 128 entries with
# oldest-fetch eviction.
# ─────────────────────────────────────────────────────────────────────────────

_OHLCV_TTL_S = float(os.environ.get("BACKTEST_OHLCV_TTL_S", "3600"))
_OHLCV_CACHE_MAX = 128
_ohlcv_cache: dict = {}          # key -> (fetched_at_monotonic, df)
_ohlcv_cache_lock = threading.Lock()


def _fetch_ohlcv_cached(
    proxy_url: str, symbol: str, market: str, interval: str, period: str,
):
    """TTL-cached fetch: same (symbol, market, interval, period) won't refetch
    within the TTL window. Per-process cache shared by all worker threads."""
    import pandas as pd
    key = (proxy_url, symbol, market, interval, period)
    now = time.monotonic()
    with _ohlcv_cache_lock:
        hit = _ohlcv_cache.get(key)
        if hit is not None and now - hit[0] < _OHLCV_TTL_S:
            return hit[1]

    if not _OHLCV_TOKEN:
        raise RuntimeError("AITRADER_INTERNAL_SECRET not set; cannot fetch OHLCV")
    resp = requests.get(
        f"{proxy_url}/internal/ohlcv",
        params={"symbol": symbol, "market": market, "interval": interval, "period": period},
        headers={"X-Aitrader-Internal-Token": _OHLCV_TOKEN},
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    log.debug("fetched %s/%s %s rows=%d", market, symbol, interval, len(df))

    with _ohlcv_cache_lock:
        # Purge expired, then evict oldest if still at cap.
        for k in [k for k, (ts, _) in _ohlcv_cache.items() if now - ts >= _OHLCV_TTL_S]:
            del _ohlcv_cache[k]
        if len(_ohlcv_cache) >= _OHLCV_CACHE_MAX:
            oldest = min(_ohlcv_cache, key=lambda k: _ohlcv_cache[k][0])
            del _ohlcv_cache[oldest]
        _ohlcv_cache[key] = (now, df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Single-task execution
# ─────────────────────────────────────────────────────────────────────────────


def _run_backtest_task(payload: dict, proxy_url: str) -> dict:
    """Execute one backtest task. Returns a result dict suitable for
    backtest_queue.complete_task.

    payload schema (see design §3 / §4.1):
        {symbol, market, interval, period, strategy, params, dynamic_pos, batch_id?}

    Validates the payload up front so a malformed enqueue surfaces as a
    clean ValueError rather than a KeyError / TypeError deep in backtest.
    """
    # Type & presence checks. The coordinator should always send valid
    # payloads, but a stray test or future caller might not.
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    symbol = payload.get("symbol")
    strategy_name = payload.get("strategy")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"payload.symbol must be a non-empty string; got {symbol!r}")
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        raise ValueError(f"payload.strategy must be a non-empty string; got {strategy_name!r}")
    market = payload.get("market", "US")
    if not isinstance(market, str):
        raise ValueError(f"payload.market must be a string; got {market!r}")
    interval = payload.get("interval", "1d")
    if not isinstance(interval, str):
        raise ValueError(f"payload.interval must be a string; got {interval!r}")
    period = payload.get("period", "2y")
    if not isinstance(period, str):
        raise ValueError(f"payload.period must be a string; got {period!r}")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"payload.params must be a dict (or null); got {type(params).__name__}")
    dynamic_pos = bool(payload.get("dynamic_pos", False))
    symbol = symbol.strip().upper()

    df = _fetch_ohlcv_cached(proxy_url, symbol, market, interval, period)
    if df is None or df.empty:
        raise ValueError(f"empty OHLCV for {market}/{symbol}")

    # Import inside the function so we don't pay the cost when just polling
    from backtest import run_strategy_backtest
    r = run_strategy_backtest(
        df, strategy_name,
        params_override=params,
        dynamic_position=dynamic_pos,
    )
    summary = r.get("summary") or {}
    # Strip unjsonable bits (numpy types) — psycopg2's Json adapter handles
    # native types fine but not numpy. Apply via main._clean if present.
    try:
        from main import _clean
        summary = _clean(summary)
    except Exception:
        pass
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Worker daemon
# ─────────────────────────────────────────────────────────────────────────────


class BacktestWorker:
    def __init__(
        self,
        worker_id: str,
        *,
        concurrency: int = 4,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        ohlcv_proxy_url: str = _DEFAULT_OHLCV_URL,
    ):
        self.worker_id = worker_id
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.proxy_url = ohlcv_proxy_url
        self._stop = False
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None

    # ── public lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        backtest_queue.register_worker(
            self.worker_id,
            hostname=socket.gethostname(),
            platform=sys.platform,
            max_concurrency=self.concurrency,
        )
        log.info(
            "worker %s started | host=%s plat=%s concurrency=%d proxy=%s",
            self.worker_id, socket.gethostname(), sys.platform,
            self.concurrency, self.proxy_url,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix=f"bt-{self.worker_id}",
        )

        # Background threads
        threading.Thread(target=self._heartbeat_loop, daemon=True,
                         name="heartbeat").start()
        threading.Thread(target=self._reclaim_loop, daemon=True,
                         name="reclaim").start()

        try:
            self._poll_loop()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop:
            return
        self._stop = True
        log.info("worker %s shutting down", self.worker_id)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
        try:
            backtest_queue.shutdown_worker(self.worker_id)
        except Exception:
            log.exception("shutdown_worker call failed")

    # ── poll loop ──────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop:
            try:
                available = self.concurrency - self._inflight_count()
                if available <= 0:
                    time.sleep(self.poll_interval)
                    continue
                tasks = backtest_queue.claim_pending(self.worker_id, limit=available)
                if not tasks:
                    time.sleep(self.poll_interval)
                    continue
                for task in tasks:
                    self._dispatch(task)
            except KeyboardInterrupt:
                self._stop = True
                break
            except Exception:
                log.exception("poll loop error; sleeping before retry")
                time.sleep(self.poll_interval * 2)

    def _dispatch(self, task: dict) -> None:
        self._bump_inflight(1)
        try:
            self._executor.submit(self._process_task, task)
        except RuntimeError:
            # Executor already shutting down (shutdown race): the task was
            # claimed but will never run here — undo the inflight bump so the
            # counter doesn't drift, and leave the task to the stale-processing
            # reclaim (another worker picks it up within RECLAIM_AGE_S).
            self._bump_inflight(-1)
            log.warning("dispatch after shutdown; task %s left for reclaim",
                        task.get("task_uuid"))

    def _process_task(self, task: dict) -> None:
        t0 = time.monotonic()
        try:
            result = _run_backtest_task(task["payload"], self.proxy_url)
            ok = backtest_queue.complete_task(
                task["task_uuid"], task["claim_token"], result,
            )
            if ok:
                backtest_queue.increment_worker_counter(self.worker_id, completed=1)
                elapsed = time.monotonic() - t0
                payload = task["payload"]
                log.info(
                    "✓ %s/%s %s sharpe=%s (%.2fs)",
                    payload.get("market", "?"), payload.get("symbol", "?"),
                    payload.get("strategy", "?"),
                    f"{result.get('sharpe', '?'):.2f}" if isinstance(result.get('sharpe'), (int, float)) else "?",
                    elapsed,
                )
            else:
                log.warning("complete_task no-op (reclaimed?): %s", task["task_uuid"])
        except Exception as exc:
            # Leave full traceback intact; backtest_queue.fail_task does the
            # final truncation to 4000 chars. Double-truncating loses root cause.
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            try:
                backtest_queue.fail_task(task["task_uuid"], task["claim_token"], err)
                backtest_queue.increment_worker_counter(self.worker_id, failed=1)
            except Exception:
                log.exception("fail_task call itself failed; task may be stuck")
            log.warning("✗ %s: %s", task["task_uuid"], exc)
        finally:
            self._bump_inflight(-1)

    # ── inflight counter ────────────────────────────────────────────────────

    def _inflight_count(self) -> int:
        with self._inflight_lock:
            return self._inflight

    def _bump_inflight(self, delta: int) -> None:
        with self._inflight_lock:
            self._inflight += delta

    # ── background loops ────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        while not self._stop:
            time.sleep(HEARTBEAT_INTERVAL_S)
            try:
                backtest_queue.heartbeat(self.worker_id)
            except Exception:
                log.exception("heartbeat failed; will retry next tick")

    def _reclaim_loop(self) -> None:
        # Every worker may run reclaim; only one will get the DB lock, others
        # see 0 reclaimed and continue. Cheap.
        while not self._stop:
            time.sleep(RECLAIM_INTERVAL_S)
            try:
                backtest_queue.reclaim_stale_processing(max_processing_seconds=RECLAIM_AGE_S)
            except Exception:
                log.exception("reclaim failed; will retry next tick")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def main():
    parser = argparse.ArgumentParser(description="Background backtest worker")
    parser.add_argument("--worker-id", default=os.environ.get("BACKTEST_WORKER_ID") or _default_worker_id())
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("BACKTEST_WORKER_CONCURRENCY", "4")))
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--ohlcv-proxy", default=os.environ.get("AITRADER_OHLCV_PROXY") or _DEFAULT_OHLCV_URL)
    parser.add_argument("--log-level", default=os.environ.get("BACKTEST_WORKER_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    worker = BacktestWorker(
        worker_id=args.worker_id,
        concurrency=args.concurrency,
        poll_interval=args.poll_interval,
        ohlcv_proxy_url=args.ohlcv_proxy,
    )

    def _signal_handler(signum, frame):
        log.info("received signal %d; stopping worker", signum)
        worker.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    # SIGTERM doesn't exist on Windows — registering it raises ValueError.
    # NSSM sends Ctrl+Break (≈ SIGBREAK) on stop; Python translates that to
    # SIGINT for us, so the SIGINT handler above is enough on Windows.
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        worker.start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
