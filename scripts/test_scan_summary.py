#!/usr/bin/env python3
"""test_scan_summary.py — verify §7.2.2b scan_summary emission + router.

Background
----------
Before this fix, a silent empty cycle (e.g. tick-1777056405 — one path's
_realtime_scan_fn returned alerts=[], exception was caught + logged, alerts
forced to []) looked identical to a legitimate no-signal cycle in the monitor
UI. Four distinct failure modes all rendered as "0 alerts":

  1. realtime_scan_fn exception
  2. universe empty (no symbols in the merged pool for this bucket)
  3. strategies empty
  4. cold interval with no cache → per-group fallback

Fix: unified_scan() now emits ONE scan_summary record per bucket, on every
path, carrying ``bucket / interval / side / path / groups / universe_size /
strategies_count / alerts_emitted / error``. Five distinct ``path`` tags:

    fallback_no_cache, topn_per_group, empty_inputs,
    realtime_scan_exception, fixed_strategy_batch

routers/scan_monitor.py has ``"scan_summary"`` in its ``_STAGES`` whitelist so
records survive the per-stage filter in /scan/monitor/cycle/{cid}.

Coverage
--------
  Backend (unified_scan)
   1.  fallback_no_cache        — interval not in signal_caches
   2.  topn_per_group           — is_topn bucket (top_n + owner set)
   3.  empty_inputs (universe)  — pool resolver returns []
   4.  empty_inputs (strategies)— group has no strategies
   5.  realtime_scan_exception  — _realtime_scan_fn raises
   6.  fixed_strategy_batch     — happy path with alerts
   7.  fixed_strategy_batch (0) — happy path with alerts=[]
   8.  multiple buckets in one cycle — one scan_summary per bucket
   9.  payload shape — every emission carries all 9 required keys

  Router (/scan/monitor/*)
   10. /scan/monitor/cycle/{id} exposes scan_summary stage (not filtered)
   11. /scan/monitor/cycles counts scan_summary per cycle
   12. /scan/monitor/stats aggregates scan_summary across cycles

Fully in-memory. No real DB, no real scans. Uses the same FastAPI TestClient
pattern as test_p8_audit.py — mount only the scan_monitor router to avoid
main.py startup side effects.

Run:
    venv/bin/python \\
        scripts/test_scan_summary.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("DATABASE_URL", "postgres://stub")

import scan_audit  # noqa: E402
import scan_center  # noqa: E402


passed = 0
failed = 0
total_tests = 12


def _check(cond: bool, label: str, detail: str = "") -> None:
    global passed, failed
    prefix = f"[{passed + failed + 1}/{total_tests}]"
    if cond:
        passed += 1
        print(f"  {prefix} {label} ... OK")
    else:
        failed += 1
        msg = f"FAIL: {label}"
        if detail:
            msg += f" ({detail})"
        print(f"  {prefix} {msg}")


def _reset_ring() -> None:
    with scan_audit._audit_lock:
        scan_audit._audit_ring.clear()
        for k in scan_audit._audit_log_errors:
            scan_audit._audit_log_errors[k] = 0


# --------------------------------------------------------------------------
# Fixtures — configure scan_center DI hooks with programmable stubs.
# Each test can override scan_stub / resolve_stub as needed via attributes.
# --------------------------------------------------------------------------


class _Scanner:
    """Programmable stub for _run_realtime_scan."""

    def __init__(self) -> None:
        self.alerts_to_return: "list[dict]" = []
        self.raise_exc: "Exception|None" = None
        self.calls: "list[dict]" = []

    def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"alerts": list(self.alerts_to_return)}


class _Resolver:
    """Programmable stub for _resolve_scan_universe.

    ``items_for_pool[pool_name] = [items]`` lets individual tests control the
    shape of the resolved universe per-pool. Missing pools resolve to [].
    """

    def __init__(self) -> None:
        self.items_for_pool: "dict[str, list[dict]]" = {}
        self.raise_exc: "Exception|None" = None
        self.calls = 0

    def __call__(self, *, pool_names, all_pools=False, scope="pool"):
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        out: "list[dict]" = []
        for p in pool_names or []:
            out.extend(self.items_for_pool.get(p, []))
        return out


scanner = _Scanner()
resolver = _Resolver()
scan_center.configure_realtime_scan_fn(scanner)
scan_center.configure_resolve_universe_fn(resolver)


def _make_group(
    sig: str,
    *,
    interval: str = "1d",
    side: str = "buy",
    pool_names=("TEST_POOL",),
    strategies=("s1",),
    top_n: int = 0,
    owner: "str|None" = None,
) -> dict:
    return {
        "signature": sig,
        "interval": interval,
        "signal_side": side,
        "pool_names": list(pool_names),
        "strategies": list(strategies),
        "refresh_sec": 60,
        "subscribers": [],
        "owner": owner,
        "top_n": top_n,
        "strategy_mode": "exclusive",
    }


def _summaries(cycle_id: str) -> "list[dict]":
    """Pull scan_summary records for one cycle from the audit ring."""
    return [
        r["payload"]
        for r in scan_audit.snapshot_audit_ring()
        if r.get("cycle_id") == cycle_id
        and r.get("stage") == "scan_summary"
    ]


def _reset_stubs() -> None:
    scanner.alerts_to_return = []
    scanner.raise_exc = None
    scanner.calls.clear()
    resolver.items_for_pool = {
        "TEST_POOL": [{"symbol": "AAPL", "market": "US"}],
    }
    resolver.raise_exc = None
    resolver.calls = 0


# --------------------------------------------------------------------------
# Test 1: fallback_no_cache — interval not in signal_caches
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
# Group uses interval "4h" — absent from signal_caches dict → cache is None
# → falls through to fallback_no_cache path, which calls _scan_one_group
# per group. That won't emit alerts because we leave resolver returning
# AAPL + scanner returning []. The point is: the scan_summary record MUST
# still be emitted with path="fallback_no_cache".
g1 = _make_group("g1-fallback", interval="4h")
scan_center.unified_scan([g1], signal_caches={}, cycle_id="test-1-fallback")
sums = _summaries("test-1-fallback")
rec = sums[0] if sums else {}
_check(
    len(sums) == 1
    and rec.get("path") == "fallback_no_cache"
    and rec.get("interval") == "4h"
    and rec.get("side") == "buy"
    and rec.get("bucket") == "4h/buy"
    and rec.get("groups") == 1
    and rec.get("universe_size") == 0  # not computed on this path
    and rec.get("strategies_count") == 0
    and rec.get("alerts_emitted") == 0
    and rec.get("error") is None,
    "fallback_no_cache emits scan_summary with path + bucket + 0 alerts",
    detail=f"sums={sums}",
)


# --------------------------------------------------------------------------
# Test 2: topn_per_group — is_topn bucket (top_n + owner)
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
cache_1d = scan_center.SignalCache()
g2 = _make_group(
    "g2-topn", interval="1d", top_n=5, owner="Claude",
    strategies=[],  # top_n resolves strategies dynamically
)
scan_center.unified_scan(
    [g2], signal_caches={"1d": cache_1d}, cycle_id="test-2-topn",
)
sums = _summaries("test-2-topn")
rec = sums[0] if sums else {}
_check(
    len(sums) == 1
    and rec.get("path") == "topn_per_group"
    and rec.get("bucket") == "1d/buy/topn"
    and rec.get("interval") == "1d"
    and rec.get("groups") == 1
    and rec.get("strategies_count") == 0  # dynamic per-group
    and rec.get("alerts_emitted") == 0
    and rec.get("error") is None,
    "topn_per_group emits scan_summary with /topn suffix in bucket key",
    detail=f"sums={sums}",
)


# --------------------------------------------------------------------------
# Test 3: empty_inputs — empty universe
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
resolver.items_for_pool = {}  # every pool resolves to []
cache_1d = scan_center.SignalCache()
g3 = _make_group("g3-empty-univ")
scan_center.unified_scan(
    [g3], signal_caches={"1d": cache_1d}, cycle_id="test-3-empty-univ",
)
sums = _summaries("test-3-empty-univ")
rec = sums[0] if sums else {}
_check(
    len(sums) == 1
    and rec.get("path") == "empty_inputs"
    and rec.get("universe_size") == 0
    and rec.get("strategies_count") == 1  # strategies provided, universe empty
    and rec.get("alerts_emitted") == 0
    and rec.get("error") == "empty universe"
    and len(scanner.calls) == 0,  # no scan invoked when universe empty
    "empty_inputs (empty universe) emits error='empty universe' + no scan call",
    detail=f"sums={sums} scan_calls={len(scanner.calls)}",
)


# --------------------------------------------------------------------------
# Test 4: empty_inputs — empty strategies
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
cache_1d = scan_center.SignalCache()
g4 = _make_group("g4-empty-strats", strategies=[])
scan_center.unified_scan(
    [g4], signal_caches={"1d": cache_1d}, cycle_id="test-4-empty-strats",
)
sums = _summaries("test-4-empty-strats")
rec = sums[0] if sums else {}
_check(
    len(sums) == 1
    and rec.get("path") == "empty_inputs"
    and rec.get("universe_size") == 1  # universe resolved
    and rec.get("strategies_count") == 0
    and rec.get("alerts_emitted") == 0
    and rec.get("error") == "empty strategies"
    and len(scanner.calls) == 0,
    "empty_inputs (empty strategies) emits error='empty strategies'",
    detail=f"sums={sums} scan_calls={len(scanner.calls)}",
)


# --------------------------------------------------------------------------
# Test 5: realtime_scan_exception — _realtime_scan_fn raises
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
scanner.raise_exc = RuntimeError("simulated scan failure")
cache_1d = scan_center.SignalCache()
g5 = _make_group("g5-exc")
scan_center.unified_scan(
    [g5], signal_caches={"1d": cache_1d}, cycle_id="test-5-exc",
)
sums = _summaries("test-5-exc")
rec = sums[0] if sums else {}
err_str = rec.get("error") or ""
_check(
    len(sums) == 1
    and rec.get("path") == "realtime_scan_exception"
    and rec.get("universe_size") == 1
    and rec.get("strategies_count") == 1
    and rec.get("alerts_emitted") == 0
    and isinstance(err_str, str)
    and err_str.startswith("RuntimeError: ")
    and "simulated scan failure" in err_str
    # Guard against the `{e!r}` regression that duplicates the type name
    # ("RuntimeError: RuntimeError('x')"): type name must appear exactly once.
    and err_str.count("RuntimeError") == 1,
    "realtime_scan_exception captures error type + message (no type-name dup)",
    detail=f"sums={sums}",
)


# --------------------------------------------------------------------------
# Test 6: fixed_strategy_batch — happy path with alerts
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
scanner.alerts_to_return = [
    {"symbol": "AAPL", "strategy": "s1", "side": "BUY",
     "price": 150.0, "interval": "1d"},
]
cache_1d = scan_center.SignalCache()
g6 = _make_group("g6-happy")
scan_center.unified_scan(
    [g6], signal_caches={"1d": cache_1d}, cycle_id="test-6-happy",
)
sums = _summaries("test-6-happy")
rec = sums[0] if sums else {}
_check(
    len(sums) == 1
    and rec.get("path") == "fixed_strategy_batch"
    and rec.get("universe_size") == 1
    and rec.get("strategies_count") == 1
    and rec.get("alerts_emitted") == 1
    and rec.get("error") is None
    and len(scanner.calls) == 1,
    "fixed_strategy_batch happy path: alerts_emitted=1, 1 scan call",
    detail=f"sums={sums} scan_calls={len(scanner.calls)}",
)


# --------------------------------------------------------------------------
# Test 7: fixed_strategy_batch with alerts=[] (no signals but scan ran)
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()  # alerts_to_return = [] by default
cache_1d = scan_center.SignalCache()
g7 = _make_group("g7-no-signals")
scan_center.unified_scan(
    [g7], signal_caches={"1d": cache_1d}, cycle_id="test-7-no-sig",
)
sums = _summaries("test-7-no-sig")
rec = sums[0] if sums else {}
_check(
    len(sums) == 1
    and rec.get("path") == "fixed_strategy_batch"
    and rec.get("alerts_emitted") == 0
    and rec.get("error") is None
    and len(scanner.calls) == 1,  # scan did run, just returned []
    "fixed_strategy_batch with alerts=[] still emits summary (not silent)",
    detail=f"sums={sums} scan_calls={len(scanner.calls)}",
)


# --------------------------------------------------------------------------
# Test 8: multiple buckets in one cycle → one scan_summary per bucket
# --------------------------------------------------------------------------
_reset_ring()
_reset_stubs()
cache_1d = scan_center.SignalCache()
cache_1h = scan_center.SignalCache()
# bucket A: 1d/buy fixed  — happy path
# bucket B: 1h/sell fixed — happy path with different side
# bucket C: 4h/buy        — fallback_no_cache (no cache for 4h)
groups_multi = [
    _make_group("g8a", interval="1d", side="buy"),
    _make_group("g8b", interval="1h", side="sell"),
    _make_group("g8c", interval="4h", side="buy"),
]
scan_center.unified_scan(
    groups_multi,
    signal_caches={"1d": cache_1d, "1h": cache_1h},
    cycle_id="test-8-multi",
)
sums = _summaries("test-8-multi")
buckets_seen = {s.get("bucket") for s in sums}
paths_seen = {s.get("path") for s in sums}
_check(
    len(sums) == 3
    and buckets_seen == {"1d/buy", "1h/sell", "4h/buy"}
    and paths_seen == {"fixed_strategy_batch", "fallback_no_cache"},
    "3 buckets in one cycle → 3 scan_summary records (distinct bucket keys)",
    detail=f"buckets={buckets_seen} paths={paths_seen}",
)


# --------------------------------------------------------------------------
# Test 9: payload shape — every emission has all 9 required keys
# --------------------------------------------------------------------------
required_keys = {
    "bucket", "interval", "side", "path", "groups",
    "universe_size", "strategies_count", "alerts_emitted", "error",
}
# Reuse test-8's 3 records — they cover 2 distinct paths.
missing_by_record: "list[tuple]" = []
for s in sums:
    missing = required_keys - set(s.keys())
    if missing:
        missing_by_record.append((s.get("path"), missing))
_check(
    not missing_by_record,
    "every scan_summary payload carries all 9 required keys",
    detail=f"missing={missing_by_record}",
)


# --------------------------------------------------------------------------
# Router tests — mount only scan_monitor to avoid main.py side effects.
# --------------------------------------------------------------------------
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from routers import scan_monitor as _r_scan_monitor  # noqa: E402

app = FastAPI()
app.include_router(_r_scan_monitor.router)
client = TestClient(app)


# --------------------------------------------------------------------------
# Test 10: /scan/monitor/cycle/{id} exposes scan_summary (not filtered by
# the _STAGES whitelist). Before the router fix, scan_summary records were
# silently dropped by the per-stage filter.
# --------------------------------------------------------------------------
_reset_ring()
scan_audit.log_stage("router-test-1", "universe", {"symbols": ["AAPL"]})
scan_audit.log_stage("router-test-1", "scan_summary", {
    "bucket": "1d/buy", "interval": "1d", "side": "buy",
    "path": "fixed_strategy_batch", "groups": 1, "universe_size": 1,
    "strategies_count": 1, "alerts_emitted": 0, "error": None,
})
scan_audit.log_stage("router-test-1", "scan_summary", {
    "bucket": "1h/sell", "interval": "1h", "side": "sell",
    "path": "empty_inputs", "groups": 1, "universe_size": 0,
    "strategies_count": 1, "alerts_emitted": 0, "error": "empty universe",
})
resp = client.get("/scan/monitor/cycle/router-test-1")
body = resp.json() if resp.status_code == 200 else None
stages = body.get("stages") if isinstance(body, dict) else None
ss_records = stages.get("scan_summary") if isinstance(stages, dict) else None
_check(
    resp.status_code == 200
    and isinstance(ss_records, list)
    and len(ss_records) == 2
    and {r.get("payload", {}).get("path") for r in ss_records}
        == {"fixed_strategy_batch", "empty_inputs"},
    "/scan/monitor/cycle exposes scan_summary stage (2 records, both paths)",
    detail=f"status={resp.status_code} ss_records={ss_records}",
)


# --------------------------------------------------------------------------
# Test 11: /scan/monitor/cycles counts scan_summary per cycle
# --------------------------------------------------------------------------
_reset_ring()
# Cycle A: 1 universe + 2 scan_summary
scan_audit.log_stage("cA", "universe", {"symbols": ["X"]})
scan_audit.log_stage("cA", "scan_summary", {
    "bucket": "1d/buy", "interval": "1d", "side": "buy",
    "path": "fixed_strategy_batch", "groups": 1, "universe_size": 1,
    "strategies_count": 1, "alerts_emitted": 0, "error": None,
})
scan_audit.log_stage("cA", "scan_summary", {
    "bucket": "1h/buy", "interval": "1h", "side": "buy",
    "path": "fixed_strategy_batch", "groups": 1, "universe_size": 1,
    "strategies_count": 1, "alerts_emitted": 1, "error": None,
})
# Cycle B: 1 universe + 1 scan_summary
scan_audit.log_stage("cB", "universe", {"symbols": ["Y"]})
scan_audit.log_stage("cB", "scan_summary", {
    "bucket": "1d/sell", "interval": "1d", "side": "sell",
    "path": "fallback_no_cache", "groups": 1, "universe_size": 0,
    "strategies_count": 0, "alerts_emitted": 0, "error": None,
})

resp = client.get("/scan/monitor/cycles", params={"limit": 50})
body = resp.json() if resp.status_code == 200 else None
cycles = body.get("cycles") if isinstance(body, dict) else []
by_cid = {c["cycle_id"]: c for c in (cycles or [])}
cA_counts = by_cid.get("cA", {}).get("counts", {})
cB_counts = by_cid.get("cB", {}).get("counts", {})
_check(
    resp.status_code == 200
    and cA_counts.get("scan_summary") == 2
    and cB_counts.get("scan_summary") == 1
    and cA_counts.get("universe") == 1
    and cB_counts.get("universe") == 1,
    "/scan/monitor/cycles includes scan_summary in per-cycle counts",
    detail=f"cA={cA_counts} cB={cB_counts}",
)


# --------------------------------------------------------------------------
# Test 12: /scan/monitor/stats aggregates scan_summary across cycles
# --------------------------------------------------------------------------
# Reuse cA + cB seed from Test 11 — 3 scan_summary + 2 universe total.
resp = client.get("/scan/monitor/stats")
body = resp.json() if resp.status_code == 200 else None
by_stage = body.get("by_stage") if isinstance(body, dict) else {}
_check(
    resp.status_code == 200
    and by_stage.get("scan_summary") == 3
    and by_stage.get("universe") == 2
    and body.get("unique_cycles") == 2,
    "/scan/monitor/stats counts scan_summary in by_stage aggregate",
    detail=f"by_stage={by_stage} unique={body.get('unique_cycles') if body else None}",
)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print()
print(f"  Results: {passed}/{total_tests} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
