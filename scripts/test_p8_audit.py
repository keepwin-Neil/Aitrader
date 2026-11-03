#!/usr/bin/env python3
"""test_p8_audit.py — verify P8 scan_audit module + monitor router (§7.2-§7.4).

Covers:
  1.  new_cycle_id format regex for tick/manual/backfill sources
  2.  new_cycle_id generates 1000 unique ids
  3.  new_cycle_id_safe fallback when new_cycle_id raises
  4.  scan_center re-exports both cycle_id producers
  5.  log_stage appends records to the ring in order
  6.  log_stage respects the 10_000 ring bound (drops oldest)
  7.  log_stage swallows non-serializable payloads without raising
  8.  log_stage is thread-safe under 8×1000 concurrent inserts
  9.  GET /scan/monitor/cycles lists distinct cycle_ids with counts
  10. GET /scan/monitor/cycle/{cid} returns all stages for one cycle
  11. GET /scan/monitor/symbol/{sym}?cycle_id=... filters by symbol
  12. GET /scan/monitor/stats aggregates counts across the ring
  13. Empty-string cycle_id / symbol rejections → HTTP 400

We mount a minimal FastAPI stub with only scan_monitor router attached, to
avoid main.py startup side effects (daemon loops, DB pools).

Run:
    venv/bin/python \
        scripts/test_p8_audit.py
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Env stub so main imports don't fail on DATABASE_URL.
os.environ.setdefault("DATABASE_URL", "postgres://stub")

import scan_audit  # noqa: E402
import scan_center  # noqa: E402

# 测试写的 cycle 不能落进真实的审计目录 (.cache/scan_audit), 指到临时目录
import atexit as _atexit  # noqa: E402
import tempfile as _tempfile  # noqa: E402
_TMP_AUDIT = _tempfile.TemporaryDirectory()
scan_audit.set_audit_dir(Path(_TMP_AUDIT.name))
_atexit.register(_TMP_AUDIT.cleanup)


passed = 0
failed = 0
total_tests = 16


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
    """Clear the ring between tests so prior state doesn't bleed in."""
    with scan_audit._audit_lock:
        scan_audit._audit_ring.clear()
        for k in scan_audit._audit_log_errors:
            scan_audit._audit_log_errors[k] = 0


# --------------------------------------------------------------------------
# Test 1: new_cycle_id format regex across sources
# --------------------------------------------------------------------------
_FMT_RE = re.compile(r"^(tick|manual|backfill)-\d+-[0-9a-f]{8}$")
sources_ok = all(
    _FMT_RE.match(scan_audit.new_cycle_id(src))
    for src in ("tick", "manual", "backfill")
)
_check(
    sources_ok,
    "new_cycle_id produces {source}-{ts}-{hex8} for tick/manual/backfill",
    detail="regex failed for one or more sources",
)


# --------------------------------------------------------------------------
# Test 2: Uniqueness across 1000 calls
# --------------------------------------------------------------------------
ids = {scan_audit.new_cycle_id("tick") for _ in range(1000)}
_check(
    len(ids) == 1000,
    "new_cycle_id produces 1000 unique ids in tight loop",
    detail=f"got {len(ids)} distinct",
)


# --------------------------------------------------------------------------
# Test 3: new_cycle_id_safe fallback when new_cycle_id raises
# --------------------------------------------------------------------------
_orig = scan_audit.new_cycle_id
try:
    def _boom(source: str = "tick") -> str:
        raise RuntimeError("simulated new_cycle_id failure")
    scan_audit.new_cycle_id = _boom
    fb = scan_audit.new_cycle_id_safe("manual")
    fb_parts = fb.split("-")
finally:
    scan_audit.new_cycle_id = _orig
_check(
    bool(fb)
    and fb_parts[0] == "manual"
    and fb_parts[1].isdigit()
    and len(fb_parts[2]) == 8
    and all(c in "0123456789abcdef" for c in fb_parts[2])
    and fb_parts[2] != "00000000",
    "new_cycle_id_safe returns valid hex8 suffix when new_cycle_id raises",
    detail=f"got {fb!r}",
)


# --------------------------------------------------------------------------
# Test 4: scan_center re-exports both producers and they work
# --------------------------------------------------------------------------
try:
    from scan_center import new_cycle_id as sc_new  # noqa: F401
    from scan_center import new_cycle_id_safe as sc_safe  # noqa: F401
    reexport_ok = (
        bool(_FMT_RE.match(sc_new("tick")))
        and bool(_FMT_RE.match(sc_safe("manual")))
        # Same function identity preserved across both modules.
        and sc_new is scan_audit.new_cycle_id
        and sc_safe is scan_audit.new_cycle_id_safe
    )
except Exception as e:  # noqa: BLE001
    reexport_ok = False
    detail = f"import error: {e!r}"
else:
    detail = ""
_check(
    reexport_ok,
    "scan_center re-exports new_cycle_id + new_cycle_id_safe from scan_audit",
    detail=detail,
)


# --------------------------------------------------------------------------
# Test 5: log_stage appends in order
# --------------------------------------------------------------------------
_reset_ring()
scan_audit.log_stage("tick-5-abc", "universe", {"symbols": ["AAPL", "NVDA"]})
scan_audit.log_stage("tick-5-abc", "raw_signal", {"symbol": "AAPL"})
scan_audit.log_stage("tick-5-abc", "filter", {"symbol": "AAPL", "passed": True})
snap = scan_audit.snapshot_audit_ring()
_check(
    len(snap) == 3
    and [r["stage"] for r in snap] == ["universe", "raw_signal", "filter"]
    and all(r["cycle_id"] == "tick-5-abc" for r in snap),
    "log_stage appends 3 records in order",
    detail=f"stages={[r['stage'] for r in snap]}",
)


# --------------------------------------------------------------------------
# Test 6: ring is bounded at maxlen=10_000 (oldest dropped)
# --------------------------------------------------------------------------
_reset_ring()
for i in range(10_050):
    scan_audit.log_stage("tick-6-xyz", "universe", {"i": i})
snap = scan_audit.snapshot_audit_ring()
# First surviving record's payload.i should be 50 (indices 0..49 dropped).
_check(
    len(snap) == 10_000
    and snap[0]["payload"]["i"] == 50
    and snap[-1]["payload"]["i"] == 10_049,
    "ring bounded at 10_000 (oldest 50 dropped)",
    detail=f"size={len(snap)} first_i={snap[0]['payload']['i'] if snap else None}",
)


# --------------------------------------------------------------------------
# Test 7: log_stage swallows exceptions (non-serializable + non-dict)
# --------------------------------------------------------------------------
_reset_ring()
errors_pre = dict(scan_audit._audit_log_errors)
# set is not JSON-serializable → drop + counter up
scan_audit.log_stage("tick-7-ser", "filter", {"bad": {1, 2}})
# non-dict payload → drop + counter up
scan_audit.log_stage("tick-7-type", "decision", "not a dict")  # type: ignore[arg-type]
# Legit record after the bad ones — ring must still accept it
scan_audit.log_stage("tick-7-ok", "universe", {"symbols": ["X"]})
errors_post = dict(scan_audit._audit_log_errors)
snap = scan_audit.snapshot_audit_ring()
_check(
    errors_post["serialize"] >= 2
    and len(snap) == 1
    and snap[0]["cycle_id"] == "tick-7-ok",
    "log_stage swallows bad payloads; good record still lands",
    detail=f"errors={errors_post} ring_size={len(snap)}",
)


# --------------------------------------------------------------------------
# Test 8: thread safety — 8 threads × 1000 inserts = 8000 total
# --------------------------------------------------------------------------
_reset_ring()
N_THREADS = 8
N_PER_THREAD = 1000


def _worker(tid: int) -> None:
    for i in range(N_PER_THREAD):
        scan_audit.log_stage(
            f"tick-8-t{tid}", "universe", {"tid": tid, "i": i}
        )


threads = [threading.Thread(target=_worker, args=(t,)) for t in range(N_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=30)
snap = scan_audit.snapshot_audit_ring()
# Ring cap is 10_000 and we inserted 8_000 — none should be dropped.
_check(
    len(snap) == N_THREADS * N_PER_THREAD,
    f"thread-safe: 8×1000={N_THREADS * N_PER_THREAD} records after concurrent insert",
    detail=f"got {len(snap)}",
)


# --------------------------------------------------------------------------
# Helper: build a minimal FastAPI app with only the scan_monitor router.
# Importing main once satisfies the circular `import main as _m` in the
# monitor router module (if any); we avoid main.app to skip its startup.
# --------------------------------------------------------------------------
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from routers import scan_monitor as _r_scan_monitor  # noqa: E402

app = FastAPI()
app.include_router(_r_scan_monitor.router)
client = TestClient(app)


def _seed_two_cycles() -> None:
    """Seed ring with 2 cycles (universe + 2 raw_signal each). Reuse across router tests."""
    _reset_ring()
    for cid in ("tick-A", "tick-B"):
        scan_audit.log_stage(cid, "universe", {
            "symbols": ["AAPL", "NVDA"],
            "by_interval": {"1d": ["AAPL", "NVDA"]},
        })
        scan_audit.log_stage(cid, "raw_signal", {
            "symbol": "AAPL", "strategy": "macd", "side": "BUY",
            "interval": "1d", "price": 100.0,
        })
        scan_audit.log_stage(cid, "raw_signal", {
            "symbol": "NVDA", "strategy": "rsi", "side": "BUY",
            "interval": "1d", "price": 500.0,
        })


# --------------------------------------------------------------------------
# Test 9: GET /scan/monitor/cycles lists distinct cycles with counts
# --------------------------------------------------------------------------
_seed_two_cycles()
resp = client.get("/scan/monitor/cycles", params={"limit": 50})
body9 = resp.json() if resp.status_code == 200 else None
cycles9 = body9.get("cycles") if isinstance(body9, dict) else None
cycle_ids_9 = [c["cycle_id"] for c in cycles9] if cycles9 else []
first9 = cycles9[0] if cycles9 else None
cnt9 = first9.get("counts", {}) if first9 else {}
_check(
    resp.status_code == 200
    and len(cycles9 or []) == 2
    and set(cycle_ids_9) == {"tick-A", "tick-B"}
    and cnt9.get("universe", 0) == 1
    and cnt9.get("raw_signal", 0) == 2
    and first9.get("first_ts", 0) <= first9.get("last_ts", 0),
    "GET /scan/monitor/cycles returns 2 cycles with per-stage counts",
    detail=f"status={resp.status_code} body={body9}",
)


# --------------------------------------------------------------------------
# Test 10: GET /scan/monitor/cycle/{cycle_id} returns all stages
# --------------------------------------------------------------------------
_seed_two_cycles()
resp = client.get("/scan/monitor/cycle/tick-A")
body10 = resp.json() if resp.status_code == 200 else None
stages10 = body10.get("stages") if isinstance(body10, dict) else None
_check(
    resp.status_code == 200
    and body10.get("cycle_id") == "tick-A"
    and isinstance(stages10, dict)
    and len(stages10.get("universe", [])) == 1
    and len(stages10.get("raw_signal", [])) == 2,
    "GET /scan/monitor/cycle/tick-A groups records by stage",
    detail=f"status={resp.status_code} stages={stages10}",
)

# 404 when cycle absent
resp_404 = client.get("/scan/monitor/cycle/no-such-cycle")
_check(
    resp_404.status_code == 404,
    "GET /scan/monitor/cycle/<unknown> returns 404",
    detail=f"status={resp_404.status_code}",
)


# --------------------------------------------------------------------------
# Test 11: GET /scan/monitor/symbol/{sym} filters by symbol within cycle
# --------------------------------------------------------------------------
_seed_two_cycles()
resp = client.get("/scan/monitor/symbol/AAPL", params={"cycle_id": "tick-A"})
body11 = resp.json() if resp.status_code == 200 else None
stages11 = body11.get("stages") if isinstance(body11, dict) else None
_check(
    resp.status_code == 200
    and body11.get("cycle_id") == "tick-A"
    and body11.get("symbol") == "AAPL"
    and isinstance(stages11, dict)
    # AAPL appears in: 1 raw_signal
    and len(stages11.get("raw_signal", [])) == 1
    # universe stage doesn't tag a single symbol → should not appear (or be empty)
    and all(
        r["payload"].get("symbol") == "AAPL"
        for stage_name, stage_rows in stages11.items()
        if stage_name != "universe"
        for r in stage_rows
    ),
    "GET /scan/monitor/symbol/AAPL returns AAPL-tagged records only",
    detail=f"status={resp.status_code} stages={stages11}",
)

# 404 when cycle absent
resp_404_sym = client.get(
    "/scan/monitor/symbol/AAPL", params={"cycle_id": "no-such-cycle"}
)
_check(
    resp_404_sym.status_code == 404,
    "GET /scan/monitor/symbol/{sym}?cycle_id=<unknown> returns 404",
    detail=f"status={resp_404_sym.status_code}",
)

# Empty arrays (not 404) when symbol absent in existing cycle
resp_empty_sym = client.get(
    "/scan/monitor/symbol/GOOG", params={"cycle_id": "tick-A"}
)
body_e = resp_empty_sym.json() if resp_empty_sym.status_code == 200 else None
stages_e = body_e.get("stages") if isinstance(body_e, dict) else None
_check(
    resp_empty_sym.status_code == 200
    and isinstance(stages_e, dict)
    and all(
        isinstance(stages_e.get(s, []), list) and not stages_e.get(s)
        for s in ("raw_signal",)
    ),
    "GET /scan/monitor/symbol/<absent>?cycle_id=<known> returns 200 with empty arrays",
    detail=f"status={resp_empty_sym.status_code} stages={stages_e}",
)


# --------------------------------------------------------------------------
# Test 12: GET /scan/monitor/stats aggregates counts across the ring
# --------------------------------------------------------------------------
_seed_two_cycles()
resp = client.get("/scan/monitor/stats")
body12 = resp.json() if resp.status_code == 200 else None
by_stage = body12.get("by_stage") if isinstance(body12, dict) else None
_check(
    resp.status_code == 200
    and isinstance(body12, dict)
    # 2 cycles × 3 records each = 6
    and body12.get("total_records") == 6
    and body12.get("unique_cycles") == 2
    and isinstance(by_stage, dict)
    and by_stage.get("universe", 0) == 2
    and by_stage.get("raw_signal", 0) == 4
    # cache stats present and numeric (don't assert concrete values)
    and isinstance(body12.get("scans_avoided"), (int, float)),
    "GET /scan/monitor/stats aggregates total/by_stage/unique_cycles + cache metrics",
    detail=f"status={resp.status_code} body={body12}",
)


# --------------------------------------------------------------------------
# Test 13: Empty-string / whitespace path params → HTTP 400
# FastAPI collapses trailing "/" redirects, so we hit the handler directly
# with an explicitly whitespace-only path segment (%20 encodes a space).
# --------------------------------------------------------------------------
_seed_two_cycles()
resp_ws_cycle = client.get("/scan/monitor/cycle/%20")
resp_ws_sym = client.get("/scan/monitor/symbol/%20", params={"cycle_id": "tick-A"})
resp_empty_cid_q = client.get(
    "/scan/monitor/symbol/AAPL", params={"cycle_id": "   "}
)
_check(
    resp_ws_cycle.status_code == 400
    and resp_ws_sym.status_code == 400
    and resp_empty_cid_q.status_code == 400,
    "empty / whitespace path+query params return HTTP 400",
    detail=(
        f"ws_cycle={resp_ws_cycle.status_code} "
        f"ws_sym={resp_ws_sym.status_code} "
        f"ws_query_cid={resp_empty_cid_q.status_code}"
    ),
)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print(f"\n  {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
