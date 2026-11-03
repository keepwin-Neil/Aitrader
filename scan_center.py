"""Scan Center module (v0.5 §3).

Owns per-interval signal caching and the unified scan entry point used by
``core_engine._process_monitor_groups``. This module implements P5 of the
v0.5 refactor: pull per-group scanning out of the engine and replace it with
ONE unified scan per cycle, served via per-interval in-memory caches.

Key design points:
  * C1 fix — ``cycle_id`` is embedded in every ``SignalKey`` so cross-cycle
    audit pollution is impossible; the sweeper also caps how many distinct
    cycles may live in the cache at once.
  * N1 fix — ``SignalCache.get_or_compute`` uses a bounded in-flight retry
    loop so a timed-out waiter re-claims the compute slot rather than looping
    forever.
  * Per-interval partitioning — each interval owns its own ``SignalCache``
    with its own TTL (1d=300s, 1h=180s, 15m=60s, 5m=30s).
  * Dependency injection — the heavy ``_run_realtime_scan`` and
    ``_resolve_scan_universe`` functions live in ``main.py``; we expose setter
    hooks so ``main.py`` can wire them at startup without a circular import.

Self-tests run when executed directly:
    venv/bin/python scan_center.py

Does NOT modify ``core_engine.py`` or ``main.py`` — wiring the engine to
call ``unified_scan`` is P6.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import scan_audit

# §7.2.1: cycle_id producers live in scan_audit (canonical location as of P8).
# We re-export them here so the six pre-existing call sites
# (`core_engine`, `main`, `routers/scan`, `scripts/test_p6_integration`, …)
# keep working without churn. If ``scan_audit`` somehow fails to import
# (e.g. during a test stub) the lazy wrappers below fall back to an inline
# synthesis so the back-compat surface never goes dark.
try:
    from scan_audit import (  # noqa: F401 — back-compat re-export
        new_cycle_id,
        new_cycle_id_safe,
    )
except ImportError:  # pragma: no cover — stubbed-test defensive branch
    def new_cycle_id(source: str = "tick") -> str:  # type: ignore[no-redef]
        return f"{source}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    def new_cycle_id_safe(source: str = "manual") -> str:  # type: ignore[no-redef]
        try:
            return new_cycle_id(source)
        except Exception:  # noqa: BLE001
            return f"{source}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_SIGNAL_SWEEP_INTERVAL = 600   # seconds between sweep passes


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalKey:
    """Cache key for SignalCache.

    ``cycle_id`` is embedded (C1 fix) so an alert emitted in cycle A can never
    be served as a hit in cycle B — each new scan cycle introduces a fresh
    namespace. The sweeper caps the number of distinct ``cycle_id`` values so
    old cycles do not accumulate forever.
    """
    symbol: str
    interval: str
    strategy: str
    cycle_id: str


@dataclass
class CachedSignal:
    """Value stored in ``SignalCache._store``.

    Not frozen because it's stored and returned by reference; only the key
    needs to be hashable. ``data`` holds the raw alert dict emitted by
    ``_run_realtime_scan``.
    """
    data: dict
    timestamp: float


# ---------------------------------------------------------------------------
# SignalCache
# ---------------------------------------------------------------------------


class SignalCache:
    """In-memory signal cache with TTL, keyed by (symbol, interval, strategy, cycle_id).

    Intra-cycle callers share results through ``get`` / ``get_or_compute``;
    ``unified_scan`` uses ``put`` to pre-populate the cache from a single
    batched scan result. The sweeper (``sweep``) is called periodically from
    a shared daemon thread to evict TTL-expired entries and cap the number
    of distinct cycles that may live in the cache (C1).
    """

    _WAIT_TIMEOUT = 60      # seconds a waiter blocks before the retry loop
    _MAX_RETRY_ROUNDS = 3   # N1: bounded retry so we never loop forever
    _MAX_CYCLES = 16        # C1: cap distinct cycle_ids; oldest dropped on overflow

    def __init__(self, ttl_seconds: int = 300):
        self._store: "dict[SignalKey, CachedSignal]" = {}
        self._inflight: "dict[SignalKey, threading.Event]" = {}
        self._cycle_first_seen: "dict[str, float]" = {}   # cycle_id -> first-seen ts
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self.stats: "dict[str, int]" = {
            "hit": 0,
            "miss": 0,
            "inflight_wait": 0,
            "inflight_timeout": 0,
            "cycle_evicted": 0,
        }

    # -- reads ---------------------------------------------------------------

    def get(
        self,
        symbol: str,
        interval: str,
        strategy: str,
        cycle_id: str,
    ) -> Optional[CachedSignal]:
        """Return cached signal if fresh, else ``None``. Read-only, no compute.

        C1: ``cycle_id`` is mandatory — a new cycle_id is a miss by construction,
        preventing cross-cycle audit pollution.
        """
        key = SignalKey(symbol, interval, strategy, cycle_id)
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.time() - entry.timestamp) < self._ttl:
                self.stats["hit"] += 1
                return entry
            self.stats["miss"] += 1
        return None

    # -- writes --------------------------------------------------------------

    def put(
        self,
        symbol: str,
        interval: str,
        strategy: str,
        cycle_id: str,
        data: dict,
    ) -> None:
        """Direct insertion used by ``unified_scan`` to pre-populate the cache
        from a single batched scan result.

        Encapsulates the lock and the ``cycle_first_seen`` bookkeeping so
        callers outside this module never touch ``_store`` directly.
        """
        key = SignalKey(symbol, interval, strategy, cycle_id)
        with self._lock:
            self._store[key] = CachedSignal(data=data, timestamp=time.time())
            self._cycle_first_seen.setdefault(cycle_id, time.time())

    def get_or_compute(
        self,
        symbol: str,
        interval: str,
        strategy: str,
        cycle_id: str,
        compute_fn: Callable[[], dict],
    ) -> CachedSignal:
        """Get cached signal or compute it with in-flight deduplication (Q6, N1, C1).

        Semantics:
          * Same cycle, fresh cache hit → return immediately (intra-cycle dedup).
          * New cycle (different ``cycle_id``) → miss by construction; compute fresh.
          * Another thread is computing same key → wait for its Event, then re-read.
          * Nobody is computing → become the computer, fill cache, signal waiters.
          * Waiter timeout or computer failure → bounded retry; promotes self to
            computer if still empty on the next round.
        """
        key = SignalKey(symbol, interval, strategy, cycle_id)
        with self._lock:
            if cycle_id not in self._cycle_first_seen:
                self._cycle_first_seen[cycle_id] = time.time()

        for _attempt in range(self._MAX_RETRY_ROUNDS):
            # Single-shot claim decision under lock.
            compute_self = False  # N1: always defined before the branch below
            with self._lock:
                entry = self._store.get(key)
                if entry and (time.time() - entry.timestamp) < self._ttl:
                    self.stats["hit"] += 1
                    return entry
                self.stats["miss"] += 1
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    compute_self = True
                else:
                    self.stats["inflight_wait"] += 1

            if not compute_self:
                # Waiter path: wait for the computer, then re-read on next iteration.
                if not event.wait(timeout=self._WAIT_TIMEOUT):
                    self.stats["inflight_timeout"] += 1
                    continue
                with self._lock:
                    entry = self._store.get(key)
                if entry and (time.time() - entry.timestamp) < self._ttl:
                    return entry
                continue

            # Computer path.
            try:
                data = compute_fn()
                with self._lock:
                    self._store[key] = CachedSignal(data=data, timestamp=time.time())
                    result = self._store[key]
                return result
            except Exception:
                # Propagate so the caller sees the real error; still wake waiters
                # in the ``finally`` block so they can retry rather than block.
                raise
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                event.set()

        raise TimeoutError(
            f"SignalCache.get_or_compute: no progress for key={key} "
            f"after {self._MAX_RETRY_ROUNDS} rounds"
        )

    # -- maintenance ---------------------------------------------------------

    def purge(self, cycle_id: Optional[str] = None) -> int:
        """Purge cache entries (§5.2.6 / P7).

        ``cycle_id=None`` → purge all entries; clear every in-flight Event so
        any waiters unblock and retry rather than hang.
        ``cycle_id=str``  → purge only entries whose ``SignalKey.cycle_id``
        matches; signal and drop matching in-flight Events.

        Returns the number of store entries removed. Mirrors the shape of
        ``opinion_cache.purge_opinion_cache`` so operators have a consistent
        purge surface across both caches.
        """
        with self._lock:
            if cycle_id is None:
                count = len(self._store)
                self._store.clear()
                # Signal then drop all in-flight events so waiters don't hang.
                for ev in list(self._inflight.values()):
                    try:
                        ev.set()
                    except Exception:  # noqa: BLE001
                        pass
                self._inflight.clear()
                self._cycle_first_seen.clear()
                return count

            # Targeted purge — by cycle_id.
            keys_to_drop = [
                k for k in self._store.keys() if k.cycle_id == cycle_id
            ]
            for k in keys_to_drop:
                self._store.pop(k, None)

            inflight_keys = [
                k for k in self._inflight.keys() if k.cycle_id == cycle_id
            ]
            for k in inflight_keys:
                ev = self._inflight.pop(k, None)
                if ev is not None:
                    try:
                        ev.set()
                    except Exception:  # noqa: BLE001
                        pass

            self._cycle_first_seen.pop(cycle_id, None)
            return len(keys_to_drop)

    def sweep(self) -> int:
        """Periodic maintenance (C1): evict TTL-expired entries + cap distinct cycles.

        Called by a shared sweeper thread (run side-by-side with the opinion
        sweeper). Returns entries dropped this pass.
        """
        removed = 0
        now = time.time()
        with self._lock:
            # Phase 1: TTL expiry.
            for k in list(self._store.keys()):
                if (now - self._store[k].timestamp) >= self._ttl:
                    del self._store[k]
                    removed += 1

            # Phase 2: cycle_id count cap.
            cycles_in_store = {k.cycle_id for k in self._store}
            if len(cycles_in_store) > self._MAX_CYCLES:
                cycles_sorted = sorted(
                    cycles_in_store,
                    key=lambda c: self._cycle_first_seen.get(c, 0.0),
                )
                to_drop = set(
                    cycles_sorted[: len(cycles_in_store) - self._MAX_CYCLES]
                )
                for k in list(self._store.keys()):
                    if k.cycle_id in to_drop:
                        del self._store[k]
                        removed += 1
                        self.stats["cycle_evicted"] += 1
                for c in to_drop:
                    self._cycle_first_seen.pop(c, None)

            # Phase 3: drop tracking for cycles no longer referenced.
            live = {k.cycle_id for k in self._store}
            for c in list(self._cycle_first_seen):
                if c not in live:
                    self._cycle_first_seen.pop(c, None)

        return removed

    def snapshot(self) -> dict:
        """Return a consistent point-in-time view of this cache's stats + size.

        Holds ``self._lock`` for the whole read so concurrent mutation can't
        produce torn numbers (e.g. size that disagrees with cycles count).
        """
        with self._lock:
            return {
                **self.stats,
                "size": len(self._store),
                "cycles": len(self._cycle_first_seen),
                "ttl_seconds": self._ttl,
            }


# ---------------------------------------------------------------------------
# Per-interval partitioning (§3.2)
# ---------------------------------------------------------------------------


_scan_caches: "dict[str, SignalCache]" = {
    "1d":  SignalCache(ttl_seconds=300),
    "1h":  SignalCache(ttl_seconds=180),
    "15m": SignalCache(ttl_seconds=60),
    "5m":  SignalCache(ttl_seconds=30),
}


# ---------------------------------------------------------------------------
# Dependency injection (avoid circular import with main.py)
# ---------------------------------------------------------------------------


# ``main.py`` owns the real implementations; we call them through these hooks.
# Same pattern as ``opinion_cache.configure_provider_fn`` — keeps this module
# free of heavy engine imports.

_realtime_scan_fn: Optional[Callable] = None
_resolve_scan_universe_fn: Optional[Callable] = None
_positions_universe_fn: Optional[Callable] = None  # Returns SELL universe list


def configure_realtime_scan_fn(fn: Callable) -> None:
    """Inject ``main._run_realtime_scan``. Call once at startup from the module
    that owns the scan implementation, to avoid circular import."""
    global _realtime_scan_fn
    _realtime_scan_fn = fn


def configure_resolve_universe_fn(fn: Callable) -> None:
    """Inject ``main._resolve_scan_universe``. Call once at startup."""
    global _resolve_scan_universe_fn
    _resolve_scan_universe_fn = fn


def configure_positions_universe_fn(fn: Callable) -> None:
    """Inject the positions-universe resolver. Returns
    ``[{"symbol": str, "market": str, "holders": int}, ...]`` (open paper
    positions). Used as the SELL-side universe so SELL signals only scan
    symbols that are actually held.

    Wired at startup with ``main._resolve_positions_universe_union``.
    Tests inject a stub directly. If unset, the SELL universe falls
    back to the BUY universe (current pre-2026-04-25 behavior) so we
    don't accidentally scan zero symbols on a misconfigured deploy.
    """
    global _positions_universe_fn
    _positions_universe_fn = fn


# Per-cycle cache of the positions universe so repeated bucket scans
# within ONE cycle don't re-query the DB. The cache key is the cycle_id
# string. Eviction: we only ever keep the most-recent cycle's entry,
# so the dict has at most 1 key. No TTL needed.
_positions_universe_cache: "dict[str, list[dict]]" = {}
_positions_universe_lock: threading.RLock = threading.RLock()


def _resolve_positions_universe(cycle_id: str) -> "list[dict]":
    """Return the positions universe for ``cycle_id``, cached per-cycle.

    Each entry: ``{"symbol": str, "market": str, "holders": int}``.
    Returns ``[]`` if no resolver is configured or the resolver fails —
    callers must treat this as "no SELL candidates this cycle" (which
    is the correct behavior when the daemon is mid-bootstrap).
    """
    if _positions_universe_fn is None:
        return []
    with _positions_universe_lock:
        cached = _positions_universe_cache.get(cycle_id)
        if cached is not None:
            return cached
    try:
        items = _positions_universe_fn() or []
    except Exception as exc:  # noqa: BLE001 — DB blip
        log.warning(
            "scan_center._resolve_positions_universe: resolver raised %r — "
            "treating SELL universe as empty for cycle %s",
            exc, cycle_id,
        )
        items = []
    # Normalize shape; drop bad rows defensively.
    norm: "list[dict]" = []
    for it in items:
        sym = str(it.get("symbol") or "").strip()
        mkt = str(it.get("market") or "").strip().upper() or "US"
        if not sym:
            continue
        norm.append({"symbol": sym, "market": mkt, "holders": int(it.get("holders") or 0)})
    with _positions_universe_lock:
        # Evict prior entries — only the most-recent cycle is interesting.
        _positions_universe_cache.clear()
        _positions_universe_cache[cycle_id] = norm
    return norm


# ---------------------------------------------------------------------------
# cycle_id helpers are re-exported from scan_audit (see top-of-module import,
# §7.2.1 — canonical location as of P8).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# unified_scan (§3.4)
# ---------------------------------------------------------------------------


# 2026-04-26: cached strategy → owner lookup for Stage 1 audit records.
# Without this, raw_signal payloads would be missing the 'owner' field
# that the UI's owner column needs.
#
# IMPORTANT: We read strategy_llm_registry.json DIRECTLY from disk rather
# than calling ``main._load_strategy_llm_registry``. Importing main has
# heavy module-level side effects (configures realtime_scan_fn, opens DB
# pools, etc.) that break test stubs and slow process startup. The
# registry is a small JSON file with a stable schema, so a flat read is
# both correct and safe.
_strategy_owner_cache: dict = {"map": None, "default": "Claude", "loaded_at": 0.0}


def _strategy_owner_for(strategy_name: str) -> str:
    """Look up the owner for a strategy from strategy_llm_registry.json.

    Cached for 60s. Returns the registry's ``default_owner`` (or 'Claude'
    fallback) for unknown strategies. Never raises — defensive against
    a missing/corrupt registry file.

    Does NOT import ``main`` (would trigger main.py module-level side
    effects that overwrite scan_center's injected ``_realtime_scan_fn``
    and break tests + parallel-process callers).
    """
    import time as _t
    import json as _json
    import os as _os
    cache = _strategy_owner_cache
    if cache["map"] is None or (_t.time() - cache["loaded_at"]) > 60:
        try:
            reg_path = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "strategy_llm_registry.json",
            )
            if _os.path.exists(reg_path):
                with open(reg_path, encoding="utf-8") as _fh:
                    reg = _json.load(_fh) or {}
            else:
                reg = {}
            cache["map"] = dict(reg.get("owners") or {})
            cache["default"] = str(reg.get("default_owner") or "Claude")
            cache["loaded_at"] = _t.time()
        except Exception:  # noqa: BLE001
            if cache["map"] is None:
                cache["map"] = {}
    m = cache["map"] or {}
    return m.get(strategy_name) or cache["default"]


def _bucket_groups(
    monitor_groups: "list[dict]",
) -> "dict[tuple, list[dict]]":
    """Bucket groups by ``(interval, signal_side, is_topn)``.

    top_n groups are segregated because their strategy set is opaque (selected
    by ``_run_realtime_scan``'s top_n logic on the fly). Mixing them into a
    union scan with fixed-strategy groups would either drop the top_n
    semantics or over-scan — both incorrect. Keeping them in their own
    buckets preserves correctness: the union optimization still collapses
    ~N fixed-strategy scans into 1 per ``(interval, signal_side)``, and
    top_n groups continue to fall back to one-shot scans as today.

    2026-04-26 (BUY/SELL universe-split fix): a group with
    ``signal_side="both"`` is split into TWO virtual buckets — one
    side="buy" (universe=merged stock pool) and one side="sell"
    (universe=positions only). Without this split, "both" buckets used
    the BUY pool for both sides → SELL signals fired on non-positions
    (e.g. TRX SELL when only BNB/ETH/SOL are held), violating the
    user-facing design '合并持仓池 用于 SELL 信号监控'. The group's
    ``signal_side`` is overridden on the deepcopied entries placed in
    each bucket so unified_scan / _run_realtime_scan filter alerts
    correctly per side.
    """
    import copy as _copy
    buckets: "dict[tuple, list[dict]]" = {}
    for g in monitor_groups:
        interval = g["interval"]
        side_raw = str(g.get("signal_side", "buy") or "buy").lower()
        is_topn = bool(g.get("top_n") and g.get("owner"))
        if side_raw == "both":
            # Split: produce TWO sub-groups, each tagged with one side.
            for side_split in ("buy", "sell"):
                g_split = _copy.deepcopy(g)
                g_split["signal_side"] = side_split
                # Force a unique signature suffix so unified_scan keeps
                # buy/sell results separate when keying output dicts.
                if g_split.get("signature"):
                    g_split["signature"] = f"{g_split['signature']}::side={side_split}"
                key = (interval, side_split, is_topn)
                buckets.setdefault(key, []).append(g_split)
        else:
            key = (interval, side_raw, is_topn)
            buckets.setdefault(key, []).append(g)
    return buckets


def _collect_universe(
    bucket_groups: "list[dict]",
) -> "list[dict]":
    """Resolve the union of pool items for a bucket, deduped by symbol.

    This is the BUY-side universe (the merged stock pool). For SELL-side
    buckets, callers should use :func:`_collect_universe_for_side` which
    swaps in the positions universe.
    """
    pool_union: "set[str]" = set()
    for g in bucket_groups:
        pool_union.update(g.get("pool_names") or [])

    if not pool_union:
        return []

    if _resolve_scan_universe_fn is None:
        raise RuntimeError(
            "scan_center: resolve_scan_universe_fn not configured — "
            "main.py must call configure_resolve_universe_fn(_resolve_scan_universe) at startup"
        )

    items = _resolve_scan_universe_fn(
        pool_names=sorted(pool_union),
        all_pools=False,
        scope="pool",
    ) or []

    # Dedupe by symbol (resolver may return duplicates when pools overlap).
    seen: "set[str]" = set()
    deduped: "list[dict]" = []
    for it in items:
        sym = it.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        deduped.append(it)
    return deduped


def _collect_universe_for_side(
    bucket_groups: "list[dict]",
    side: str,
    cycle_id: str,
) -> "list[dict]":
    """Side-aware universe resolution (2026-04-25):

    * ``side == "buy"`` → merged stock pool (existing behavior).
    * ``side == "sell"`` → positions universe, filtered to the markets that
      this bucket's monitor groups currently watch. This way the SELL scan
      only considers symbols that someone holds AND that the user is
      monitoring (otherwise we'd surface SELL alerts on positions in
      markets the user has no strategies for).

    Edge cases:
      * No positions universe configured → falls back to BUY universe so
        a misconfigured deploy doesn't silently scan zero symbols.
      * Empty positions universe → returns ``[]`` and the existing
        ``empty_inputs`` scan_summary path handles it (UI shows "empty
        sell pool", which is the correct user message).

    Returned dict shape matches ``_collect_universe``: at least
    ``{"symbol", "market"}``. ``pool`` is omitted for SELL items
    (positions don't belong to a watchlist pool).
    """
    if str(side or "buy").lower() != "sell":
        return _collect_universe(bucket_groups)

    # Determine which markets this SELL bucket's monitor groups watch by
    # peeking at their resolved BUY pools. Cheap because we already pay
    # this resolver call for the stage-0 audit; this is an extra walk
    # over the items (no extra DB hit).
    bucket_markets: "set[str]" = set()
    if _resolve_scan_universe_fn is not None:
        for g in bucket_groups:
            g_pools = g.get("pool_names") or []
            if not g_pools:
                continue
            try:
                g_items = _resolve_scan_universe_fn(
                    pool_names=sorted(set(g_pools)),
                    all_pools=False,
                    scope="pool",
                ) or []
            except Exception:  # noqa: BLE001 — resolver flake; skip
                continue
            for it in g_items:
                mkt = str(it.get("market") or "US").upper()
                bucket_markets.add(mkt)

    positions_items = _resolve_positions_universe(cycle_id)
    if not positions_items:
        # No positions / resolver unset → safe fallback to BUY universe so
        # we don't accidentally scan zero symbols on misconfig. Logged so
        # ops can spot it.
        if _positions_universe_fn is None:
            log.debug(
                "scan_center: SELL bucket falling back to BUY universe — "
                "positions resolver not configured"
            )
        return _collect_universe(bucket_groups)

    # Filter positions to bucket-watched markets. If the bucket has no
    # explicit market set (e.g. all groups have empty pool_names — a config
    # error), surface ALL positions so the operator at least sees signal.
    if not bucket_markets:
        filtered = positions_items
    else:
        filtered = [
            it for it in positions_items
            if str(it.get("market") or "US").upper() in bucket_markets
        ]

    seen: "set[str]" = set()
    deduped: "list[dict]" = []
    for it in filtered:
        sym = it.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        deduped.append({
            "symbol": sym,
            "market": str(it.get("market") or "US").upper(),
        })
    return deduped


def _collect_strategy_union(
    bucket_groups: "list[dict]",
) -> "list[str]":
    """Sorted union of explicit strategy names across the bucket."""
    strategies: "set[str]" = set()
    for g in bucket_groups:
        strategies.update(g.get("strategies") or [])
    return sorted(strategies)


def unified_scan(
    monitor_groups: "list[dict]",
    signal_caches: "dict[str, SignalCache]",
    *,
    cycle_id: str,
) -> "dict[str, list]":
    """Run ONE scan per (interval, signal_side) bucket and route alerts back
    to each group (§3.4).

    Steps:
      1. Bucket groups by ``(interval, signal_side, is_topn)``.
      2. For each fixed-strategy bucket:
         a. Resolve the union universe (``_resolve_scan_universe`` once).
         b. Resolve the strategy union.
         c. Call ``_run_realtime_scan`` ONCE with the union.
         d. Populate ``signal_caches[interval]`` with ``cycle_id`` embedded.
      3. For top_n buckets, fall back to one scan per group (opaque strategy set).
      4. Return ``{group_signature: [relevant_alerts_from_cache]}`` routed via
         the cache lookup so downstream consumers can re-hit the cache freely
         within the same cycle.
    """
    if _realtime_scan_fn is None:
        raise RuntimeError(
            "scan_center: realtime_scan_fn not configured — "
            "main.py must call configure_realtime_scan_fn(_run_realtime_scan) at startup"
        )

    out: "dict[str, list]" = {}
    buckets = _bucket_groups(monitor_groups)

    # §7.2.2 "universe" stage — ONE audit record per cycle summarizing the
    # merged symbol pool. We aggregate across every bucket so the monitor UI
    # can show a single "Stage 0" number. Emit-once-at-the-top contract: this
    # runs before any bucket-level scan, even if all buckets are empty.
    #
    # 2026-04-25: split into BUY pool (merged stock pool from monitor groups)
    # and SELL pool (open paper positions, deduped). Frontend
    # uses these to render per-market tabs with two sub-cards each.
    _audit_universe: "set[str]" = set()
    _audit_by_interval: "dict[str, set[str]]" = {}
    _audit_by_market: "dict[str, set[str]]" = {}     # full universe per market
    _audit_buy_pool: "dict[str, set[str]]" = {}      # BUY (stock pool) per market
    _audit_buy_markets_in_use: "set[str]" = set()    # markets monitor groups care about
    for g in monitor_groups:
        g_interval = g.get("interval", "?")
        g_items: "list[dict]" = []
        g_pools = g.get("pool_names") or []
        if g_pools and _resolve_scan_universe_fn is not None:
            try:
                g_items = _resolve_scan_universe_fn(
                    pool_names=sorted(set(g_pools)),
                    all_pools=False,
                    scope="pool",
                ) or []
            except Exception:  # noqa: BLE001 — audit is best-effort
                g_items = []
        for it in g_items:
            sym = it.get("symbol")
            if not sym:
                continue
            mkt = str(it.get("market") or "US").upper()
            _audit_universe.add(sym)
            _audit_by_interval.setdefault(g_interval, set()).add(sym)
            _audit_by_market.setdefault(mkt, set()).add(sym)
            _audit_buy_pool.setdefault(mkt, set()).add(sym)
            _audit_buy_markets_in_use.add(mkt)

    # SELL pool = open paper positions.
    _audit_sell_pool: "dict[str, set[str]]" = {}
    positions_items = _resolve_positions_universe(cycle_id)
    for pos in positions_items:
        mkt = str(pos.get("market") or "US").upper()
        sym = pos.get("symbol")
        if not sym:
            continue
        # Show every market that has positions, even if no monitor group
        # currently watches it — the UI can show empty BUY card / non-empty
        # SELL card. (Filtering here would hide a real SELL workload.)
        _audit_sell_pool.setdefault(mkt, set()).add(sym)
        # SELL symbols also count toward the union universe + by_market view,
        # so Stage 0's "all markets" totals reflect both pools.
        _audit_universe.add(sym)
        _audit_by_market.setdefault(mkt, set()).add(sym)

    # log_stage is no-raise; wire it at the natural unlocked seam above.
    scan_audit.log_stage(cycle_id, "universe", {
        "symbols": sorted(_audit_universe),
        "by_interval": {k: sorted(v) for k, v in _audit_by_interval.items()},
        # NEW (2026-04-25): per-market split + BUY/SELL pool separation.
        # Keys uppercase ISO market codes ("US", "HK", "CN", "UK", "CRYPTO").
        # Values: sorted symbol lists.
        "by_market": {k: sorted(v) for k, v in _audit_by_market.items()},
        "buy_pool": {k: sorted(v) for k, v in _audit_buy_pool.items()},
        "sell_pool": {k: sorted(v) for k, v in _audit_sell_pool.items()},
    })

    for (interval, side, is_topn), bucket in buckets.items():
        cache = signal_caches.get(interval)
        bucket_key = f"{interval}/{side}{'/topn' if is_topn else ''}"
        if cache is None:
            # Unsupported interval — fall back to per-group isolated scans and
            # do not touch the cache (no cache for this interval exists).
            log.debug(
                "scan_center: no cache registered for interval=%s; "
                "falling back to per-group scans", interval,
            )
            total_alerts = 0
            for g in bucket:
                alerts_for_g = _scan_one_group(g, cycle_id, cache=None)
                out[g["signature"]] = alerts_for_g
                total_alerts += len(alerts_for_g)
            # §7.2.2b Emit scan_summary even for fallback path so empty cycles
            # are distinguishable from "unified_scan never reached this bucket".
            scan_audit.log_stage(cycle_id, "scan_summary", {
                "bucket": bucket_key,
                "interval": interval,
                "side": side,
                "path": "fallback_no_cache",
                "groups": len(bucket),
                "universe_size": 0,  # no universe computed on this path
                "strategies_count": 0,
                "alerts_emitted": total_alerts,
                "error": None,
            })
            continue

        if is_topn:
            # Each top_n group stays isolated: its strategy set is selected
            # dynamically by _run_realtime_scan based on top_n + owner, so it
            # cannot share a scan with other groups.
            total_alerts = 0
            for g in bucket:
                alerts_for_g = _scan_one_group(g, cycle_id, cache=cache)
                out[g["signature"]] = alerts_for_g
                total_alerts += len(alerts_for_g)
            scan_audit.log_stage(cycle_id, "scan_summary", {
                "bucket": bucket_key,
                "interval": interval,
                "side": side,
                "path": "topn_per_group",
                "groups": len(bucket),
                "universe_size": 0,  # per-group universes; aggregate omitted
                "strategies_count": 0,  # dynamic per-group
                "alerts_emitted": total_alerts,
                "error": None,
            })
            continue

        # Fixed-strategy bucket — one batched scan covers the whole bucket.
        # 2026-04-25: side-aware universe — BUY uses the merged stock pool
        # (existing behavior), SELL uses the positions universe filtered
        # to the markets this bucket's groups watch.
        universe = _collect_universe_for_side(bucket, side, cycle_id)
        strategies = _collect_strategy_union(bucket)

        if not universe or not strategies:
            # Nothing to scan; still emit empty entries for every group so
            # callers can blindly iterate the return dict.
            for g in bucket:
                out[g["signature"]] = []
            scan_audit.log_stage(cycle_id, "scan_summary", {
                "bucket": bucket_key,
                "interval": interval,
                "side": side,
                "path": "empty_inputs",
                "groups": len(bucket),
                "universe_size": len(universe),
                "strategies_count": len(strategies),
                "alerts_emitted": 0,
                "error": (
                    "empty universe" if not universe
                    else "empty strategies"
                ),
            })
            continue

        try:
            result = _realtime_scan_fn(
                items=universe,
                interval=interval,
                top_n=0,
                strategy_names=strategies,
                llm_owners=None,
                signal_side=side,
                refresh_sec=None,
                local_only=False,
                filters=None,
                progress_hook=None,
            ) or {}
        except Exception as e:  # noqa: BLE001 — logged and surfaced as empty bucket
            log.exception(
                "scan_center.unified_scan: _run_realtime_scan failed "
                "interval=%s side=%s err=%r", interval, side, e,
            )
            for g in bucket:
                out[g["signature"]] = []
            # §7.2.2b Emit scan_summary with error so UI can show a ❌ red
            # banner instead of a silent "0 signals" (would otherwise be
            # indistinguishable from a legitimate no-signal cycle).
            scan_audit.log_stage(cycle_id, "scan_summary", {
                "bucket": bucket_key,
                "interval": interval,
                "side": side,
                "path": "realtime_scan_exception",
                "groups": len(bucket),
                "universe_size": len(universe),
                "strategies_count": len(strategies),
                "alerts_emitted": 0,
                # `{e}` (not `{e!r}`) — repr duplicates the type name since
                # BaseException.__repr__ already prints it (e.g.
                # "RuntimeError: RuntimeError('x')"). Use the stringified
                # message and prepend the type name once.
                "error": f"{type(e).__name__}: {e}"[:300],
            })
            continue

        alerts = result.get("alerts") or []

        # Populate the cache under cycle_id so every subscriber sees the
        # same snapshot within this cycle.
        for alert in alerts:
            sym = alert.get("symbol")
            strat = alert.get("strategy")
            if not sym or not strat:
                continue
            cache.put(sym, interval, strat, cycle_id, alert)
            # §7.2.2 "raw_signal" stage — one audit record per emitted signal.
            # Skip no-signal iterations (filtered above). Emit OUTSIDE the
            # SignalCache lock by virtue of cache.put() having already
            # returned. scan_audit.log_stage owns never-raise internally.
            scan_audit.log_stage(cycle_id, "raw_signal", {
                "symbol": sym,
                "strategy": strat,
                "owner": _strategy_owner_for(strat),
                "side": alert.get("side"),
                "interval": interval,
                "market": alert.get("market"),
                "price": float(alert.get("price") or 0.0),
            })

        # Route alerts back to each group via the cache (so a later intra-cycle
        # lookup also hits — single source of truth).
        for g in bucket:
            sig = g["signature"]
            group_alerts: "list[dict]" = []
            group_strategies = g.get("strategies") or []
            # Use the universe members this group's pools contain.
            group_pools = set(g.get("pool_names") or [])
            if group_pools and _resolve_scan_universe_fn is not None:
                try:
                    group_items = _resolve_scan_universe_fn(
                        pool_names=sorted(group_pools),
                        all_pools=False,
                        scope="pool",
                    ) or []
                except Exception:  # noqa: BLE001
                    group_items = universe
            else:
                group_items = universe

            seen_syms: "set[str]" = set()
            for it in group_items:
                sym = it.get("symbol")
                if not sym or sym in seen_syms:
                    continue
                seen_syms.add(sym)
                for strat in group_strategies:
                    hit = cache.get(sym, interval, strat, cycle_id)
                    if hit is not None:
                        group_alerts.append(hit.data)
            out[sig] = group_alerts

        # §7.2.2b Fixed-strategy bucket summary (happy path).
        # Emits even when alerts=[] so a silent "no signals" cycle is still
        # distinguishable from an aborted / never-reached bucket.
        scan_audit.log_stage(cycle_id, "scan_summary", {
            "bucket": bucket_key,
            "interval": interval,
            "side": side,
            "path": "fixed_strategy_batch",
            "groups": len(bucket),
            "universe_size": len(universe),
            "strategies_count": len(strategies),
            "alerts_emitted": len(alerts),
            "error": None,
        })

    return out


def _scan_one_group(
    group: dict,
    cycle_id: str,
    cache: Optional[SignalCache],
) -> "list[dict]":
    """Fallback per-group scan for top_n groups (opaque strategy set) and
    unsupported intervals. Still populates the cache when one is available
    so downstream lookups stay consistent within the cycle.

    2026-04-26 (BUY/SELL split fix): for SELL buckets the universe must
    come from the positions resolver (only stocks the user holds), not
    from pool_names (the merged stock pool). Without this guard, top_n
    groups bypassed the side-split because _scan_one_group resolved
    universe straight from pool_names regardless of side — letting SELL
    signals fire on non-positions like BTC. _collect_universe_for_side
    handles non-topn buckets via the fixed-strategy union path; this
    function is the topn-bucket equivalent.
    """
    interval = group["interval"]
    side = group.get("signal_side", "buy")
    pool_names = sorted(group.get("pool_names") or [])

    if _resolve_scan_universe_fn is None:
        raise RuntimeError(
            "scan_center: resolve_scan_universe_fn not configured"
        )

    side_norm = str(side or "buy").lower()
    items: "list[dict]" = []
    if side_norm == "sell":
        # SELL universe = positions only. Filter to the markets this
        # group's pools cover so a CRYPTO group doesn't surface US
        # positions, etc. Mirrors _collect_universe_for_side.
        try:
            pool_items = _resolve_scan_universe_fn(
                pool_names=pool_names, all_pools=False, scope="pool",
            ) or []
        except Exception:  # noqa: BLE001
            pool_items = []
        bucket_markets = {
            str(it.get("market") or "US").upper()
            for it in pool_items
            if it.get("market")
        }
        positions_items = _resolve_positions_universe(cycle_id)
        if not positions_items and _positions_universe_fn is None:
            log.debug(
                "scan_center._scan_one_group: SELL bucket falling back to "
                "BUY universe — positions resolver not configured"
            )
            items = pool_items
        elif not positions_items:
            return []  # no positions → nothing to scan SELL on
        else:
            if bucket_markets:
                items = [
                    p for p in positions_items
                    if str(p.get("market") or "US").upper() in bucket_markets
                ]
            else:
                items = positions_items
    else:
        # BUY (or anything not 'sell') uses pool_names → merged BUY pool.
        try:
            items = _resolve_scan_universe_fn(
                pool_names=pool_names,
                all_pools=False,
                scope="pool",
            ) or []
        except Exception:  # noqa: BLE001
            log.exception(
                "scan_center._scan_one_group: universe resolve failed sig=%s",
                group.get("signature"),
            )
            return []

    if not items:
        return []

    try:
        result = _realtime_scan_fn(
            items=items,
            interval=interval,
            top_n=int(group.get("top_n") or 0),
            strategy_names=group.get("strategies") or None,
            llm_owners=[group["owner"]] if group.get("owner") else None,
            signal_side=side,
            refresh_sec=group.get("refresh_sec"),
            local_only=False,
            filters=None,
            progress_hook=None,
        ) or {}
    except Exception:  # noqa: BLE001
        log.exception(
            "scan_center._scan_one_group: scan failed sig=%s",
            group.get("signature"),
        )
        return []

    alerts = result.get("alerts") or []
    if cache is not None:
        for alert in alerts:
            sym = alert.get("symbol")
            strat = alert.get("strategy")
            if not sym or not strat:
                continue
            cache.put(sym, interval, strat, cycle_id, alert)
    # §7.2.2 "raw_signal" stage — emit for top_n / fallback scans too so the
    # audit trail is consistent with the fixed-strategy path above.
    # scan_audit.log_stage owns never-raise internally; call it naked per
    # alert so one bad payload can't blackhole the rest of the bucket.
    # In top_n / fallback path the group owns exactly one owner — prefer
    # group["owner"], fallback to per-strategy lookup.
    group_owner = group.get("owner")
    for alert in alerts:
        sym = alert.get("symbol")
        strat = alert.get("strategy")
        if not sym or not strat:
            continue
        scan_audit.log_stage(cycle_id, "raw_signal", {
            "symbol": sym,
            "strategy": strat,
            "owner": group_owner or _strategy_owner_for(strat),
            "side": alert.get("side"),
            "interval": interval,
            "market": alert.get("market"),
            "price": float(alert.get("price") or 0.0),
        })
    return alerts


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------


def purge_signal_cache(
    *,
    cycle_id: Optional[str] = None,
    interval: Optional[str] = None,
) -> "dict[str, int]":
    """Module-facing purge used by the HTTP layer (§5.2.6 / P7).

    ``interval=None`` → purge every per-interval cache.
    ``interval=str``  → purge only that cache; unknown interval → empty dict.
    ``cycle_id`` is forwarded to each cache's ``purge()``:
      * ``cycle_id=None`` — clear all entries in the selected cache(s).
      * ``cycle_id=str``  — clear only entries under that cycle_id.

    Returns a ``{interval: count_removed}`` map so the caller can summarize
    the purge by interval.
    """
    out: "dict[str, int]" = {}
    if interval is not None:
        cache = _scan_caches.get(interval)
        if cache is None:
            return out
        out[interval] = cache.purge(cycle_id=cycle_id)
        return out

    for iv, cache in _scan_caches.items():
        out[iv] = cache.purge(cycle_id=cycle_id)
    return out


def snapshot_signal_caches() -> "dict[str, dict]":
    """Per-interval snapshot across all ``_scan_caches``.

    Each interval takes its own lock — intervals aren't locked together, so the
    snapshot is per-interval-consistent but not cross-interval-atomic. That's
    fine for an operational dashboard.
    """
    return {iv: cache.snapshot() for iv, cache in _scan_caches.items()}


def _signal_sweeper_once() -> "dict[str, int]":
    """Run one sweep across all per-interval caches.

    Returns ``{interval: dropped_count}``. Failures on one interval are logged
    but never propagate — the sweeper must keep running across iterations.
    """
    out: "dict[str, int]" = {}
    for interval, cache in _scan_caches.items():
        try:
            out[interval] = cache.sweep()
        except Exception:  # noqa: BLE001 — sweeper must keep running
            log.exception("scan_center sweeper failed for interval=%s", interval)
            out[interval] = 0
    if any(v for v in out.values()):
        log.debug("scan_center sweeper dropped=%s", out)
    return out


def _signal_sweeper_loop() -> None:
    """Long-running thread body. Sleep, sweep, repeat. Never dies on errors."""
    while True:
        time.sleep(_SIGNAL_SWEEP_INTERVAL)
        try:
            _signal_sweeper_once()
        except Exception as e:  # noqa: BLE001 — outer guard
            log.warning(
                "scan_center: sweeper iteration failed err=%r",
                e, exc_info=True,
            )


def start_signal_sweeper() -> threading.Thread:
    """Spawn the daemon sweeper thread. Returns the thread handle (mostly for
    tests that want to assert it is alive)."""
    t = threading.Thread(
        target=_signal_sweeper_loop,
        daemon=True,
        name="scan_center-sweeper",
    )
    t.start()
    return t


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal tests — run: venv/bin/python scan_center.py
    import sys

    passed = 0
    failed = 0

    def check(cond: bool, label: str) -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {label}")
        else:
            failed += 1
            print(f"  FAIL: {label}")

    print("=== scan_center self-test ===")

    # Test 1: basic get miss
    cache = SignalCache(ttl_seconds=5)
    check(
        cache.get("AAPL", "1d", "macd", "tick-1-abc") is None,
        "empty cache returns None",
    )

    # Test 2: get_or_compute miss computes
    calls = [0]

    def compute_one():
        calls[0] += 1
        return {"symbol": "AAPL", "strategy": "macd", "side": "buy", "price": 100.0}

    result = cache.get_or_compute("AAPL", "1d", "macd", "tick-1-abc", compute_one)
    check(calls[0] == 1, "compute_fn called once on miss")
    check(result.data["price"] == 100.0, "computed data returned")

    # Test 3: get_or_compute hit within cycle
    result2 = cache.get_or_compute("AAPL", "1d", "macd", "tick-1-abc", compute_one)
    check(calls[0] == 1, "compute_fn NOT called on hit")
    check(cache.stats["hit"] >= 1, "hit stat incremented")

    # Test 4: C1 — different cycle_id = miss
    cache.get_or_compute("AAPL", "1d", "macd", "tick-2-def", compute_one)
    check(calls[0] == 2, "new cycle_id triggers recompute (C1)")

    # Test 5: put() direct insert
    cache2 = SignalCache(ttl_seconds=5)
    cache2.put("NVDA", "1h", "rsi", "tick-3-ghi", {"symbol": "NVDA"})
    hit = cache2.get("NVDA", "1h", "rsi", "tick-3-ghi")
    check(
        hit is not None and hit.data["symbol"] == "NVDA",
        "put() then get() works",
    )

    # Test 6: TTL expiry on sweep
    cache3 = SignalCache(ttl_seconds=0)  # immediate expiry
    cache3.put("TSLA", "5m", "macd", "tick-4-jkl", {"x": 1})
    time.sleep(0.01)
    dropped = cache3.sweep()
    check(dropped >= 1, f"sweep drops expired entries (got {dropped})")

    # Test 7: cycle cap (C1) — inject many cycles, sweep should cap
    cache4 = SignalCache(ttl_seconds=3600)  # long TTL so TTL phase doesn't evict
    for i in range(SignalCache._MAX_CYCLES + 5):
        cache4.put("AAPL", "1d", "macd", f"tick-{i}-xxx", {"i": i})
    dropped = cache4.sweep()
    remaining_cycles = {k.cycle_id for k in cache4._store}
    check(
        len(remaining_cycles) == SignalCache._MAX_CYCLES,
        f"sweep caps cycles at _MAX_CYCLES (got {len(remaining_cycles)})",
    )
    check(cache4.stats["cycle_evicted"] >= 5, "cycle_evicted stat incremented")

    # Test 8: new_cycle_id format
    cid = new_cycle_id("manual")
    parts = cid.split("-")
    check(
        parts[0] == "manual" and parts[1].isdigit() and len(parts[2]) == 8,
        f"new_cycle_id format correct (got {cid})",
    )

    # Test 8a: new_cycle_id_safe normal path = same shape as new_cycle_id
    cid_safe = new_cycle_id_safe("manual")
    safe_parts = cid_safe.split("-")
    check(
        safe_parts[0] == "manual"
        and safe_parts[1].isdigit()
        and len(safe_parts[2]) == 8
        and all(c in "0123456789abcdef" for c in safe_parts[2]),
        f"new_cycle_id_safe normal path returns {{source}}-{{ts}}-{{hex8}} (got {cid_safe})",
    )

    # Test 8b: new_cycle_id_safe fallback path — when new_cycle_id raises,
    # the fallback synth must still produce a valid 8-char hex suffix
    # (NOT a literal "00000000") so concurrent manual triggers in the
    # same second do not collide.
    import scan_center as _self
    _orig_new_cycle_id = _self.new_cycle_id

    def _raising():
        raise RuntimeError("simulated new_cycle_id failure")

    try:
        _self.new_cycle_id = lambda source="tick": _raising()
        cid_fb = new_cycle_id_safe("manual")
    finally:
        _self.new_cycle_id = _orig_new_cycle_id
    fb_parts = cid_fb.split("-")
    check(
        fb_parts[0] == "manual"
        and fb_parts[1].isdigit()
        and len(fb_parts[2]) == 8
        and all(c in "0123456789abcdef" for c in fb_parts[2])
        and fb_parts[2] != "00000000",
        f"new_cycle_id_safe fallback path returns unique hex8 suffix (got {cid_fb})",
    )

    # Test 9: in-flight dedup — two threads race, only one compute call
    cache5 = SignalCache(ttl_seconds=60)
    slow_calls = [0]
    start = threading.Event()

    def slow_compute():
        start.wait(timeout=1.0)
        slow_calls[0] += 1
        time.sleep(0.05)
        return {"x": "slow"}

    results: "list" = []

    def worker():
        results.append(cache5.get_or_compute("X", "1d", "s", "cid-race", slow_compute))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    time.sleep(0.02)  # let both enter
    start.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    check(
        slow_calls[0] == 1,
        f"in-flight dedup: compute called once (got {slow_calls[0]})",
    )
    check(
        len(results) == 2 and results[0] is results[1],
        "both threads got same CachedSignal",
    )

    # Test 10: unified_scan with stubbed realtime_scan
    called: "list" = []

    def stub_scan(items, interval, top_n=0, strategy_names=None, llm_owners=None,
                  signal_side="buy", refresh_sec=None, local_only=False,
                  filters=None, progress_hook=None):
        called.append({
            "interval": interval,
            "strategies": strategy_names,
            "side": signal_side,
            "item_count": len(items),
        })
        return {
            "alerts": [
                {
                    "symbol": s["symbol"],
                    "strategy": (strategy_names or ["macd"])[0],
                    "side": signal_side,
                    "interval": interval,
                    "price": 100.0,
                    "market": "US",
                }
                for s in items
            ]
        }

    def stub_resolve(pool_names, all_pools=False, scope="pool"):
        return [
            {"symbol": "AAPL", "market": "US"},
            {"symbol": "NVDA", "market": "US"},
        ]

    configure_realtime_scan_fn(stub_scan)
    configure_resolve_universe_fn(stub_resolve)

    # wipe caches from earlier tests
    for c in _scan_caches.values():
        c._store.clear()
        c._cycle_first_seen.clear()

    groups = [
        {
            "signature": "g1",
            "interval": "1d",
            "signal_side": "buy",
            "pool_names": ["us_tech"],
            "strategies": ["macd"],
            "owner": None,
            "top_n": 0,
            "subscribers": [],
            "refresh_sec": 300,
        },
        {
            "signature": "g2",
            "interval": "1d",
            "signal_side": "buy",
            "pool_names": ["us_tech"],
            "strategies": ["macd"],
            "owner": None,
            "top_n": 0,
            "subscribers": [],
            "refresh_sec": 300,
        },
    ]
    out = unified_scan(groups, _scan_caches, cycle_id="cid-test-1")
    check(
        len(called) == 1,
        f"unified_scan called _run_realtime_scan ONCE for 2 identical groups (got {len(called)})",
    )
    check("g1" in out and "g2" in out, "return dict has both group signatures")
    check(
        len(out["g1"]) >= 1 and len(out["g2"]) >= 1,
        "both groups got alerts from cache",
    )

    # Test 11: RuntimeError if not configured
    # (can't easily test without un-configuring — skip)

    print(f"\n  {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
