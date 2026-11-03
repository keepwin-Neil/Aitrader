#!/usr/bin/env python3
"""test_p12_durability.py — verify P12 M6 atomic per-cycle segments (§7.5).

Covers:
  1.  _segment_path derives YYYY-MM-DD from cycle_id ts + creates parent dir
  2.  _segment_path("unknown") raises ValueError
  3.  _segment_path(malformed) raises ValueError
  4.  log_stage also appends to _audit_buffers (disk queue)
  5.  log_stage with falsy cycle_id skips disk buffer (no "unknown" key)
  6.  _flush_buffer writes JSONL segment with sort_keys format
  7.  _flush_buffer appends on second flush (no overwrite)
  8.  _flush_buffer atomic on crash simulation: tmp cleaned + requeued +
      flush_errors incremented + final file unchanged
  9.  recover_audit_segments removes orphaned *.jsonl.tmp files
  10. recover_audit_segments truncates partial trailing JSON line + bumps
      corrupt_recovered
  11. recover_audit_segments on missing AUDIT_DIR returns 0 (no error)
  12. _audit_flusher background thread flushes periodically + clean shutdown
  13. log_stage schedules immediate flush when buffer hits MAXROWS
  14. GET /scan/monitor/cycle/{cid} falls back to on-disk segment when ring
      is empty
  15. GET /scan/monitor/symbol/{sym}?cycle_id=<cid> falls back to disk
  16. GET /scan/cache/stats exposes audit_flush sub-dict
  17. cron/scan_audit_cleanup.py deletes dirs older than retention window
  18. _flush_buffer drops cycle buffer after N consecutive failures (I2)

Run:
    venv/bin/python \
        scripts/test_p12_durability.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("DATABASE_URL", "postgres://stub")

import scan_audit  # noqa: E402


passed = 0
failed = 0
total_tests = 18


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


# ---------------------------------------------------------------------------
# Test hygiene helpers
# ---------------------------------------------------------------------------


def _reset_audit_state() -> None:
    """Clear ring, buffers, stats, and any lingering flusher thread."""
    # Ensure flusher is stopped before clearing state.
    scan_audit.stop_audit_flusher(timeout=2.0)
    with scan_audit._audit_lock:
        scan_audit._audit_ring.clear()
        for k in scan_audit._audit_log_errors:
            scan_audit._audit_log_errors[k] = 0
    with scan_audit._audit_buffers_lock:
        scan_audit._audit_buffers.clear()
        scan_audit._audit_flush_failures.clear()
        for k in scan_audit._audit_flush_stats:
            scan_audit._audit_flush_stats[k] = 0


# Save the original so tests can restore between scenarios.
_ORIG_AUDIT_DIR = scan_audit._AUDIT_DIR


# ---------------------------------------------------------------------------
# Test 1: _segment_path derives date + creates parent
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    # 1735061400 == 2024-12-24 at some UTC-ish local time; pick a cycle_id
    # with that ts so we get a known date string. The derived date depends
    # on the local timezone, so we compute the expected string ourselves
    # rather than hard-coding.
    ts = 1735061400
    expected_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    cid = f"tick-{ts}-a3f9c2b1"
    p = scan_audit._segment_path(cid)
    expected_path = Path(tmp) / expected_date / f"{cid}.jsonl"
    dir_created = (Path(tmp) / expected_date).is_dir()
    _check(
        p == expected_path and dir_created,
        "_segment_path derives YYYY-MM-DD from ts and creates parent dir",
        detail=f"got={p} expected={expected_path} dir_created={dir_created}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 2: _segment_path("unknown") raises ValueError
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    raised = False
    try:
        scan_audit._segment_path("unknown")
    except ValueError:
        raised = True
    _check(
        raised,
        "_segment_path('unknown') raises ValueError",
        detail=f"raised={raised}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 3: _segment_path(malformed) raises ValueError
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    r1 = r2 = r3 = False
    try:
        scan_audit._segment_path("no-dashes-bad")  # ts portion not int
    except ValueError:
        r1 = True
    try:
        scan_audit._segment_path("single")  # only one part
    except ValueError:
        r2 = True
    try:
        scan_audit._segment_path("")  # empty
    except ValueError:
        r3 = True
    _check(
        r1 and r2 and r3,
        "_segment_path rejects malformed/empty cycle_id",
        detail=f"no-dashes={r1} single={r2} empty={r3}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 4: log_stage also accumulates into _audit_buffers
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-a1b2c3d4"
    scan_audit.log_stage(cid, "universe", {"symbols": ["AAPL"]})
    scan_audit.log_stage(cid, "raw_signal", {"symbol": "AAPL"})
    scan_audit.log_stage(cid, "filter", {"symbol": "AAPL", "passed": True})
    with scan_audit._audit_buffers_lock:
        buf_len = len(scan_audit._audit_buffers.get(cid, []))
        keys = list(scan_audit._audit_buffers.keys())
    _check(
        buf_len == 3 and keys == [cid],
        "log_stage appends 3 records to _audit_buffers[cid]",
        detail=f"buf_len={buf_len} keys={keys}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 5: log_stage with falsy cycle_id doesn't create "unknown" buffer
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    scan_audit.log_stage("", "universe", {"x": 1})
    scan_audit.log_stage(None, "raw_signal", {"y": 2})  # type: ignore[arg-type]
    ring_snap = scan_audit.snapshot_audit_ring()
    with scan_audit._audit_buffers_lock:
        buf_keys = list(scan_audit._audit_buffers.keys())
    _check(
        len(ring_snap) == 2
        and all(r["cycle_id"] == "unknown" for r in ring_snap)
        and "unknown" not in buf_keys
        and buf_keys == [],
        "log_stage with falsy cycle_id: ring gets 'unknown', buffers skip it",
        detail=f"ring_cids={[r['cycle_id'] for r in ring_snap]} "
               f"buf_keys={buf_keys}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 6: _flush_buffer writes JSONL segment with sort_keys format
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-b2c3d4e5"
    scan_audit.log_stage(cid, "universe", {"symbols": ["AAPL", "NVDA"]})
    scan_audit.log_stage(cid, "raw_signal", {"symbol": "AAPL", "price": 100.0})
    pre_flushed = scan_audit._audit_flush_stats["flushed_segments"]
    ok = scan_audit._flush_buffer(cid)
    post_flushed = scan_audit._audit_flush_stats["flushed_segments"]
    seg = scan_audit._segment_path(cid)
    exists = seg.exists()
    lines = seg.read_text(encoding="utf-8").splitlines() if exists else []
    first_row = json.loads(lines[0]) if lines else None
    keys_sorted = list(first_row.keys()) == sorted(first_row.keys()) if first_row else False
    _check(
        ok
        and exists
        and len(lines) == 2
        and post_flushed == pre_flushed + 1
        and first_row.get("cycle_id") == cid
        and first_row.get("stage") == "universe"
        and keys_sorted,
        "_flush_buffer writes 2 lines JSONL with sort_keys + stat incremented",
        detail=f"ok={ok} exists={exists} lines={len(lines)} "
               f"flushed_delta={post_flushed - pre_flushed} "
               f"keys_sorted={keys_sorted}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 7: _flush_buffer appends on second flush (existing content preserved)
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-c3d4e5f6"
    scan_audit.log_stage(cid, "universe", {"symbols": ["AAPL"]})
    scan_audit.log_stage(cid, "raw_signal", {"symbol": "AAPL", "i": 1})
    ok1 = scan_audit._flush_buffer(cid)
    seg = scan_audit._segment_path(cid)
    first_batch = seg.read_text(encoding="utf-8")
    first_batch_lines = first_batch.splitlines()

    # Second batch — new records appended
    scan_audit.log_stage(cid, "filter", {"symbol": "AAPL", "i": 2})
    scan_audit.log_stage(cid, "opinion", {"symbol": "AAPL", "i": 3})
    ok2 = scan_audit._flush_buffer(cid)
    full = seg.read_text(encoding="utf-8")
    full_lines = full.splitlines()
    # First two lines must be unchanged from first batch
    first_preserved = (full_lines[:2] == first_batch_lines)
    stages_in_order = [json.loads(l).get("stage") for l in full_lines]
    _check(
        ok1 and ok2
        and len(full_lines) == 4
        and first_preserved
        and stages_in_order == ["universe", "raw_signal", "filter", "opinion"],
        "_flush_buffer appends on second call; first-batch lines unchanged",
        detail=f"ok1={ok1} ok2={ok2} total_lines={len(full_lines)} "
               f"first_preserved={first_preserved} stages={stages_in_order}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 8: _flush_buffer atomic on crash — os.replace raising leaves final
# unchanged, cleans tmp, requeues records, bumps flush_errors
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-d4e5f6a7"
    # Seed a good first flush so "final" has content we can check is unchanged
    scan_audit.log_stage(cid, "universe", {"symbols": ["AAPL"]})
    scan_audit._flush_buffer(cid)
    seg = scan_audit._segment_path(cid)
    baseline_content = seg.read_text(encoding="utf-8")
    baseline_size = seg.stat().st_size

    # Now buffer 2 more records and simulate a crashed os.replace
    scan_audit.log_stage(cid, "filter", {"symbol": "AAPL", "i": 2})
    scan_audit.log_stage(cid, "opinion", {"symbol": "AAPL", "i": 3})
    pre_errors = scan_audit._audit_flush_stats["flush_errors"]

    _orig_replace = os.replace

    def _boom_replace(src, dst):
        raise OSError("simulated atomic write failure")

    os.replace = _boom_replace
    try:
        ok = scan_audit._flush_buffer(cid)
    finally:
        os.replace = _orig_replace

    post_errors = scan_audit._audit_flush_stats["flush_errors"]
    with scan_audit._audit_buffers_lock:
        requeued = list(scan_audit._audit_buffers.get(cid, []))
    # Verify: final file unchanged, tmp cleaned, 2 records requeued, err++
    final_unchanged = (seg.read_text(encoding="utf-8") == baseline_content
                       and seg.stat().st_size == baseline_size)
    tmp_file = seg.with_suffix(".jsonl.tmp")
    tmp_gone = not tmp_file.exists()
    _check(
        ok is False
        and final_unchanged
        and tmp_gone
        and len(requeued) == 2
        and post_errors == pre_errors + 1,
        "_flush_buffer atomic: tmp cleaned, requeued, flush_errors++, final unchanged",
        detail=f"ok={ok} final_unchanged={final_unchanged} tmp_gone={tmp_gone} "
               f"requeued={len(requeued)} err_delta={post_errors - pre_errors}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 9: recover_audit_segments removes orphaned *.jsonl.tmp
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    date_dir = Path(tmp) / "2024-12-24"
    date_dir.mkdir(parents=True)
    # Create a fake orphan tmp + a normal segment
    orphan = date_dir / "tick-1735000000-abc12345.jsonl.tmp"
    orphan.write_text("partial content", encoding="utf-8")
    normal = date_dir / "tick-1735000001-def67890.jsonl"
    normal.write_text('{"cycle_id": "tick-1735000001-def67890", "timestamp": 1.0, "stage": "universe", "payload": {}}\n', encoding="utf-8")
    cleaned = scan_audit.recover_audit_segments()
    _check(
        cleaned == 1
        and not orphan.exists()
        and normal.exists(),
        "recover_audit_segments removes orphaned tmp files, keeps segments",
        detail=f"cleaned={cleaned} orphan_exists={orphan.exists()} "
               f"normal_exists={normal.exists()}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 10: recover_audit_segments truncates partial trailing line +
# bumps corrupt_recovered stat
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    date_dir = Path(tmp) / "2024-12-24"
    date_dir.mkdir(parents=True)
    seg = date_dir / "tick-1735000002-aaa00001.jsonl"
    # One good line + one truncated partial line (no newline, bad json)
    good_line = json.dumps({
        "cycle_id": "tick-1735000002-aaa00001",
        "timestamp": 1.0,
        "stage": "universe",
        "payload": {"x": 1},
    }, sort_keys=True)
    bad_tail = '{"cycle_id": "tick-1735000002-aaa00001", "timestamp": 2.0, "stage":'
    seg.write_text(good_line + "\n" + bad_tail, encoding="utf-8")
    pre_corrupt = scan_audit._audit_flush_stats["corrupt_recovered"]
    scan_audit.recover_audit_segments()
    post_corrupt = scan_audit._audit_flush_stats["corrupt_recovered"]
    content_after = seg.read_text(encoding="utf-8")
    lines_after = content_after.splitlines()
    trailing_line_ok = False
    if lines_after:
        try:
            json.loads(lines_after[-1])
            trailing_line_ok = True
        except Exception:
            pass
    _check(
        len(lines_after) == 1
        and trailing_line_ok
        and post_corrupt == pre_corrupt + 1,
        "recover_audit_segments truncates partial trailing line + bumps stat",
        detail=f"lines_after={len(lines_after)} trailing_ok={trailing_line_ok} "
               f"corrupt_delta={post_corrupt - pre_corrupt} "
               f"content={content_after!r}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 11: recover_audit_segments on missing dir returns 0 (no error)
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    nonexistent = Path(tmp) / "does" / "not" / "exist"
    scan_audit.set_audit_dir(nonexistent)
    try:
        n = scan_audit.recover_audit_segments()
        ok = (n == 0)
        err = None
    except Exception as exc:
        ok = False
        err = repr(exc)
    _check(
        ok,
        "recover_audit_segments returns 0 on missing dir (no error)",
        detail=f"ok={ok} err={err}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 12: background flusher loop flushes periodically + clean shutdown
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    # Patch MAXAGE so the loop fires fast enough for tests
    _orig_maxage = scan_audit._AUDIT_FLUSH_MAXAGE
    scan_audit._AUDIT_FLUSH_MAXAGE = 0.1
    try:
        cid = f"tick-{int(time.time())}-e5f6a7b8"
        scan_audit.log_stage(cid, "universe", {"symbols": ["AAPL"]})
        scan_audit.log_stage(cid, "raw_signal", {"symbol": "AAPL"})
        scan_audit.start_audit_flusher()
        time.sleep(0.35)  # should fire ~3 times
        thread_alive_pre = (
            scan_audit._audit_flusher_thread is not None
            and scan_audit._audit_flusher_thread.is_alive()
        )
        scan_audit.stop_audit_flusher(timeout=2.0)
        thread_alive_post = (
            scan_audit._audit_flusher_thread is not None
            and scan_audit._audit_flusher_thread.is_alive()
        )
        seg = scan_audit._segment_path(cid)
        seg_exists = seg.exists()
    finally:
        scan_audit._AUDIT_FLUSH_MAXAGE = _orig_maxage
    _check(
        thread_alive_pre
        and seg_exists
        and not thread_alive_post,
        "_audit_flusher loop flushes periodically + stops cleanly on Event",
        detail=f"alive_pre={thread_alive_pre} alive_post={thread_alive_post} "
               f"seg_exists={seg_exists}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 13: log_stage schedules immediate flush when buffer hits MAXROWS
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    _orig_maxrows = scan_audit._AUDIT_FLUSH_MAXROWS
    scan_audit._AUDIT_FLUSH_MAXROWS = 3
    try:
        cid = f"tick-{int(time.time())}-f6a7b8c9"
        # Log exactly MAXROWS records — the 3rd triggers _schedule_flush
        scan_audit.log_stage(cid, "universe", {"i": 0})
        scan_audit.log_stage(cid, "raw_signal", {"i": 1})
        scan_audit.log_stage(cid, "filter", {"i": 2})
        # Wait up to 500ms for the scheduled single-shot thread
        deadline = time.time() + 0.5
        seg = scan_audit._segment_path(cid)
        while time.time() < deadline and not seg.exists():
            time.sleep(0.02)
        seg_exists = seg.exists()
        lines = seg.read_text(encoding="utf-8").splitlines() if seg_exists else []
    finally:
        scan_audit._AUDIT_FLUSH_MAXROWS = _orig_maxrows
    _check(
        seg_exists and len(lines) == 3,
        "log_stage triggers _schedule_flush when buffer hits MAXROWS",
        detail=f"seg_exists={seg_exists} lines={len(lines)}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Router test plumbing: mount scan_monitor + scan routers on a stub app.
# ---------------------------------------------------------------------------
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Import main once so circular `import main as _m` in routers/scan resolves.
import main as _main_mod  # noqa: E402, F401
from routers import scan_monitor as _r_scan_monitor  # noqa: E402
from routers import scan as _r_scan  # noqa: E402

app = FastAPI()
app.include_router(_r_scan_monitor.router)
app.include_router(_r_scan.router)
client = TestClient(app)


def _seed_disk_segment(cid: str, symbol: str = "AAPL") -> None:
    """Write a fake on-disk segment for ``cid`` with a couple of stages."""
    path = scan_audit._segment_path(cid)
    rows = [
        {
            "cycle_id": cid,
            "timestamp": 100.0,
            "stage": "universe",
            "payload": {"symbols": [symbol]},
        },
        {
            "cycle_id": cid,
            "timestamp": 101.0,
            "stage": "raw_signal",
            "payload": {"symbol": symbol, "strategy": "macd", "side": "BUY"},
        },
        {
            "cycle_id": cid,
            "timestamp": 102.0,
            "stage": "scan_summary",
            "payload": {"bucket": "US/1d/buy", "alerts_emitted": 1},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Test 14: GET /scan/monitor/cycle/{cid} falls back to disk
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-ae1b2c3d"
    _seed_disk_segment(cid, "AAPL")
    # Ring is empty (we just reset) — endpoint must fall back to disk.
    resp = client.get(f"/scan/monitor/cycle/{cid}")
    body = resp.json() if resp.status_code == 200 else None
    stages = body.get("stages") if isinstance(body, dict) else None
    _check(
        resp.status_code == 200
        and body.get("cycle_id") == cid
        and isinstance(stages, dict)
        and len(stages.get("universe", [])) == 1
        and len(stages.get("raw_signal", [])) == 1
        and len(stages.get("scan_summary", [])) == 1,
        "GET /scan/monitor/cycle/{cid} falls back to disk when ring empty",
        detail=f"status={resp.status_code} body={body}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 15: GET /scan/monitor/symbol/{sym}?cycle_id=... falls back to disk
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-be2c3d4e"
    _seed_disk_segment(cid, "NVDA")
    resp = client.get(
        f"/scan/monitor/symbol/NVDA", params={"cycle_id": cid},
    )
    body = resp.json() if resp.status_code == 200 else None
    stages = body.get("stages") if isinstance(body, dict) else None
    _check(
        resp.status_code == 200
        and body.get("cycle_id") == cid
        and body.get("symbol") == "NVDA"
        and isinstance(stages, dict)
        and len(stages.get("raw_signal", [])) == 1,
        "GET /scan/monitor/symbol/{sym}?cycle_id=... falls back to disk",
        detail=f"status={resp.status_code} body={body}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Test 16: GET /scan/cache/stats exposes audit_flush key
# ---------------------------------------------------------------------------
_reset_audit_state()
resp = client.get("/scan/cache/stats")
body = resp.json() if resp.status_code == 200 else None
audit_flush = body.get("audit_flush") if isinstance(body, dict) else None
_check(
    resp.status_code == 200
    and isinstance(audit_flush, dict)
    and "flushed_segments" in audit_flush
    and "flush_errors" in audit_flush
    and "corrupt_recovered" in audit_flush
    and all(isinstance(audit_flush[k], int)
            for k in ("flushed_segments", "flush_errors", "corrupt_recovered")),
    "GET /scan/cache/stats exposes audit_flush with 3 int sub-keys",
    detail=f"status={resp.status_code} audit_flush={audit_flush}",
)


# ---------------------------------------------------------------------------
# Test 17: cron/scan_audit_cleanup.py respects retention window
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    audit_root = Path(tmp) / "scan_audit"
    audit_root.mkdir()
    # 10 days old (should be deleted at 7d retention)
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    # 2 days old (should survive)
    young_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    old_dir = audit_root / old_date
    young_dir = audit_root / young_date
    old_dir.mkdir()
    young_dir.mkdir()
    (old_dir / "fake.jsonl").write_text("old\n")
    (young_dir / "fake.jsonl").write_text("young\n")

    script = _PROJECT_ROOT / "cron" / "scan_audit_cleanup.py"
    env = os.environ.copy()
    env["SCAN_AUDIT_DIR"] = str(audit_root)
    env["SCAN_AUDIT_RETENTION_DAYS"] = "7"
    proc = subprocess.run(
        [sys.executable, str(script)],
        env=env, capture_output=True, text=True, timeout=10,
    )
    _check(
        proc.returncode == 0
        and not old_dir.exists()
        and young_dir.exists()
        and "deleted 1" in proc.stdout,
        "cron/scan_audit_cleanup deletes >7d dirs, keeps young ones",
        detail=f"rc={proc.returncode} stdout={proc.stdout!r} "
               f"stderr={proc.stderr!r} old_exists={old_dir.exists()} "
               f"young_exists={young_dir.exists()}",
    )


# ---------------------------------------------------------------------------
# Test 18: _flush_buffer drops cycle buffer after N consecutive failures (I2)
# ---------------------------------------------------------------------------
_reset_audit_state()
with tempfile.TemporaryDirectory() as tmp:
    scan_audit.set_audit_dir(Path(tmp))
    cid = f"tick-{int(time.time())}-fa11ba11"
    scan_audit.log_stage(cid, "universe", {"symbols": ["AAPL"]})
    scan_audit.log_stage(cid, "raw_signal", {"symbol": "AAPL"})

    # Monkey-patch _segment_path to always raise PermissionError — simulates
    # a persistent FS error (e.g. parent dir unwritable).
    _orig_seg_path = scan_audit._segment_path

    def _boom_seg_path(cycle_id):
        raise PermissionError("simulated persistent FS failure")

    scan_audit._segment_path = _boom_seg_path
    pre_dropped = scan_audit._audit_flush_stats["dropped_cycles"]
    pre_errors = scan_audit._audit_flush_stats["flush_errors"]
    requeued_counts: "list[int]" = []
    try:
        # First N-1 calls: should requeue and bump failure counter.
        for i in range(scan_audit._AUDIT_MAX_FLUSH_RETRIES - 1):
            ok = scan_audit._flush_buffer(cid)
            with scan_audit._audit_buffers_lock:
                requeued_counts.append(len(scan_audit._audit_buffers.get(cid, [])))
            if ok is not False:
                break
        # After N-1 calls the buffer should still be present (2 records).
        with scan_audit._audit_buffers_lock:
            buf_before_drop = len(scan_audit._audit_buffers.get(cid, []))
            failures_before = scan_audit._audit_flush_failures.get(cid, 0)

        # Nth call: should drop the buffer + bump dropped_cycles.
        scan_audit._flush_buffer(cid)
        with scan_audit._audit_buffers_lock:
            buf_after_drop = len(scan_audit._audit_buffers.get(cid, []))
            failures_after = scan_audit._audit_flush_failures.get(cid, 0)
        post_dropped = scan_audit._audit_flush_stats["dropped_cycles"]
        post_errors = scan_audit._audit_flush_stats["flush_errors"]

        # A new log_stage for the same cid should start fresh (no stuck
        # counter), and it should NOT have the old failure streak.
        scan_audit.log_stage(cid, "filter", {"symbol": "AAPL"})
        with scan_audit._audit_buffers_lock:
            new_buf_len = len(scan_audit._audit_buffers.get(cid, []))
            new_failures = scan_audit._audit_flush_failures.get(cid, 0)
    finally:
        scan_audit._segment_path = _orig_seg_path

    _check(
        # First N-1 calls kept buffer at 2 records (requeued each time)
        all(c == 2 for c in requeued_counts)
        # Just before the drop, buffer still has 2 records
        and buf_before_drop == 2
        # failures counter was N-1
        and failures_before == scan_audit._AUDIT_MAX_FLUSH_RETRIES - 1
        # After the Nth failure, buffer is dropped
        and buf_after_drop == 0
        # Failure counter reset after drop
        and failures_after == 0
        # dropped_cycles stat incremented by exactly 1
        and post_dropped == pre_dropped + 1
        # flush_errors was bumped once per failed call (including the Nth)
        and post_errors == pre_errors + scan_audit._AUDIT_MAX_FLUSH_RETRIES
        # Fresh log_stage after drop starts clean: 1 record, 0 failures
        and new_buf_len == 1
        and new_failures == 0,
        "_flush_buffer drops buffer after MAX_FLUSH_RETRIES consecutive failures (I2)",
        detail=f"requeued_counts={requeued_counts} buf_before_drop={buf_before_drop} "
               f"failures_before={failures_before} buf_after_drop={buf_after_drop} "
               f"failures_after={failures_after} dropped_delta={post_dropped - pre_dropped} "
               f"err_delta={post_errors - pre_errors} new_buf={new_buf_len} "
               f"new_failures={new_failures}",
    )
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
# Restore AUDIT_DIR + stop any lingering flusher before we exit.
scan_audit.stop_audit_flusher(timeout=2.0)
scan_audit.set_audit_dir(_ORIG_AUDIT_DIR)

print(f"\n  {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
