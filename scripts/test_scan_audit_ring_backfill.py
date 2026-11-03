#!/usr/bin/env python3
"""test_scan_audit_ring_backfill.py — verify ring backfill on startup.

Why this exists: prior to 2026-04-25, the in-memory ``_audit_ring`` was
process-local, so a worker restart left it empty. The scan-monitor UI's
`/scan/monitor/cycles` endpoint reads from the ring, so users saw "0
stocks / 0 cycles" for ~5 minutes after restart until the next scan tick
repopulated the ring — even though every cycle was already on disk.

The fix is :func:`scan_audit.replay_recent_segments_to_ring`, called
from the existing ``startup_scan_audit_durability`` event hook in
``main.py``. This test pins down the contract so the next person who
"refactors" the recovery path can't silently break it.

Run::
    venv/bin/python \\
        scripts/test_scan_audit_ring_backfill.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

# Path setup so we can import scan_audit from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import scan_audit  # noqa: E402  — after sys.path mutation


passed = 0
failed = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {label}")
    else:
        failed += 1
        msg = f"FAIL: {label}"
        if detail:
            msg += f" ({detail})"
        print(f"  {msg}")


def _reset_ring() -> None:
    """Reset module-level state so each test starts clean."""
    with scan_audit._audit_lock:
        scan_audit._audit_ring.clear()
    with scan_audit._audit_buffers_lock:
        scan_audit._audit_buffers.clear()
        scan_audit._audit_flush_failures.clear()


def _make_segment(audit_dir: Path, cycle_id: str, rows: list[dict]) -> Path:
    """Helper: write a JSONL segment under audit_dir/<date>/<cycle_id>.jsonl
    with the given row dicts."""
    parts = cycle_id.split("-")
    if len(parts) < 3:
        raise AssertionError(f"bad test cycle_id: {cycle_id}")
    date_str = datetime.fromtimestamp(int(parts[-2])).strftime("%Y-%m-%d")
    date_dir = audit_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    seg = date_dir / f"{cycle_id}.jsonl"
    with open(seg, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return seg


def _row(cycle_id: str, ts: float, stage: str, payload: dict) -> dict:
    return {"cycle_id": cycle_id, "timestamp": ts, "stage": stage, "payload": payload}


def main() -> int:
    print("=== scan_audit.replay_recent_segments_to_ring ===")

    # ------------------------------------------------------------------
    # 1) Empty audit dir → returns 0, ring stays empty.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        check(
            "1 empty audit dir → 0 records",
            n == 0,
            detail=f"got n={n}",
        )

    # ------------------------------------------------------------------
    # 2) Non-existent audit dir → 0 records (not an error).
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(Path(tmpdir) / "does_not_exist")
        _reset_ring()
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        check(
            "2 non-existent audit dir → 0 records (no exception)",
            n == 0,
            detail=f"got n={n}",
        )

    # ------------------------------------------------------------------
    # 3) Single segment with 3 rows → ring has 3 records, in file order.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        cid = f"tick-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        _make_segment(Path(tmpdir), cid, [
            _row(cid, 1.0, "universe", {"symbols": ["AAPL"]}),
            _row(cid, 2.0, "raw_signal", {"symbol": "AAPL"}),
            _row(cid, 3.0, "decision", {"alerts_emitted": 1}),
        ])
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        check(
            "3 single segment 3 rows → backfilled 3",
            n == 3,
            detail=f"got n={n}",
        )
        snap = scan_audit.snapshot_audit_ring()
        check(
            "3b ring contains all 3 records in file order",
            len(snap) == 3
            and [r["stage"] for r in snap] == ["universe", "raw_signal", "decision"],
            detail=f"snap stages={[r['stage'] for r in snap]}",
        )

    # ------------------------------------------------------------------
    # 4) Multiple cycles across multiple date directories → newest at
    #    the tail of the ring (so /scan/monitor/cycles returns newest first).
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        # Three cycles, oldest to newest by ts.
        ts_old = int(time.time()) - 86400 * 2  # 2 days ago
        ts_mid = int(time.time()) - 86400      # 1 day ago
        ts_new = int(time.time())
        cid_old = f"tick-{ts_old}-{uuid.uuid4().hex[:8]}"
        cid_mid = f"tick-{ts_mid}-{uuid.uuid4().hex[:8]}"
        cid_new = f"tick-{ts_new}-{uuid.uuid4().hex[:8]}"
        for cid, ts in ((cid_old, ts_old), (cid_mid, ts_mid), (cid_new, ts_new)):
            _make_segment(Path(tmpdir), cid, [
                _row(cid, float(ts), "universe", {"symbols": ["AAPL"]}),
                _row(cid, float(ts) + 0.5, "decision", {"alerts_emitted": 0}),
            ])
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        check(
            "4 three cycles across 3 date dirs → 6 records total",
            n == 6,
            detail=f"got n={n}",
        )
        snap = scan_audit.snapshot_audit_ring()
        # Ring tail should be the newest cycle's "decision" record.
        check(
            "4b ring tail is newest cycle's last record",
            len(snap) == 6 and snap[-1]["cycle_id"] == cid_new
            and snap[-1]["stage"] == "decision",
            detail=f"tail={snap[-1] if snap else None}",
        )
        # Ring head should be the OLDEST cycle's first record.
        check(
            "4c ring head is oldest cycle's first record",
            len(snap) == 6 and snap[0]["cycle_id"] == cid_old
            and snap[0]["stage"] == "universe",
            detail=f"head={snap[0] if snap else None}",
        )

    # ------------------------------------------------------------------
    # 5) max_cycles cap honored — only N most-recent cycles backfilled.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        # 5 cycles spread over 5 different days. We ask for max_cycles=2.
        cycle_ids = []
        for i in range(5):
            ts = int(time.time()) - 86400 * (4 - i)
            cid = f"tick-{ts}-{uuid.uuid4().hex[:8]}"
            cycle_ids.append(cid)
            _make_segment(Path(tmpdir), cid, [
                _row(cid, float(ts), "universe", {"i": i}),
            ])
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=2)
        check(
            "5 max_cycles=2 cap honored (only 2 most-recent cycles loaded)",
            n == 2,
            detail=f"got n={n}",
        )
        snap = scan_audit.snapshot_audit_ring()
        loaded = {r["cycle_id"] for r in snap}
        # We expect cycle_ids[3] and cycle_ids[4] (the two most recent).
        expected = set(cycle_ids[-2:])
        check(
            "5b loaded cycles are the two most recent",
            loaded == expected,
            detail=f"loaded={loaded} expected={expected}",
        )

    # ------------------------------------------------------------------
    # 6) Malformed JSON line in the middle of a segment → that line is
    #    skipped, the rest are loaded (read_segment is forgiving).
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        cid = f"tick-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        date_str = datetime.fromtimestamp(int(cid.split('-')[-2])).strftime("%Y-%m-%d")
        date_dir = Path(tmpdir) / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        seg = date_dir / f"{cid}.jsonl"
        with open(seg, "w", encoding="utf-8") as f:
            f.write(json.dumps(_row(cid, 1.0, "universe", {"x": 1})) + "\n")
            f.write("THIS IS NOT VALID JSON\n")
            f.write(json.dumps(_row(cid, 2.0, "decision", {"x": 2})) + "\n")
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        check(
            "6 malformed line skipped → 2 good records loaded",
            n == 2,
            detail=f"got n={n}",
        )
        snap = scan_audit.snapshot_audit_ring()
        check(
            "6b ring has stages [universe, decision] (bad line skipped)",
            [r["stage"] for r in snap] == ["universe", "decision"],
            detail=f"stages={[r['stage'] for r in snap]}",
        )

    # ------------------------------------------------------------------
    # 7) Re-run on a non-empty ring → records are APPENDED (not replaced).
    #    This guards against accidentally clobbering live records.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        # Pre-load one live record into the ring.
        scan_audit.log_stage("live-1-aaaaaaaa", "universe", {"live": True})
        # Now add a disk segment and replay.
        ts = int(time.time())
        cid = f"tick-{ts}-{uuid.uuid4().hex[:8]}"
        _make_segment(Path(tmpdir), cid, [
            _row(cid, float(ts), "universe", {"backfilled": True}),
        ])
        n = scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        snap = scan_audit.snapshot_audit_ring()
        check(
            "7 replay does not clobber live ring entries",
            len(snap) == 2 and any(
                r.get("payload", {}).get("live") for r in snap
            ),
            detail=f"snap={snap}",
        )

    # ------------------------------------------------------------------
    # 8) Routing contract: /scan/monitor/cycles after backfill is non-empty.
    #    We don't spin up FastAPI here; we just call the inner helper that
    #    routers/scan_monitor.py uses (snapshot_audit_ring).
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        scan_audit.set_audit_dir(tmpdir)
        _reset_ring()
        ts = int(time.time())
        cid = f"tick-{ts}-{uuid.uuid4().hex[:8]}"
        _make_segment(Path(tmpdir), cid, [
            _row(cid, float(ts), "universe", {"symbols": ["AAPL"]}),
        ])
        scan_audit.replay_recent_segments_to_ring(max_cycles=10)
        snap = scan_audit.snapshot_audit_ring()
        unique_cycles = {r["cycle_id"] for r in snap}
        check(
            "8 router contract — at least one cycle visible after backfill",
            len(unique_cycles) >= 1,
            detail=f"got cycles={unique_cycles}",
        )

    # Restore the original audit dir so subsequent processes don't pick
    # up our temp paths.
    default_dir = Path(os.environ.get("SCAN_AUDIT_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "scan_audit")))
    scan_audit.set_audit_dir(default_dir)
    _reset_ring()

    print()
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
