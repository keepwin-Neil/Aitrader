"""Scan Audit module (Scan Center v0.5 §7.2 / P8 + §7.5 / P12).

Owns the stage-audit ring buffer, the canonical ``cycle_id`` producer, and
(since P12) per-cycle JSONL segment durability + background flusher + startup
recovery.

Every scan pipeline write point (§7.2.2) calls ``log_stage()`` with a compact
JSON-serializable payload; the monitoring UI (§7.4) reads the ring through
``snapshot_audit_ring()`` via ``routers/scan_monitor``. For cycles that have
aged out of the in-memory ring (10,000 most-recent records — ~100 cycles
at typical 100 records/cycle), the router falls back to
``read_segment()`` which loads the on-disk JSONL segment.

Key design points:
  * ``new_cycle_id`` / ``new_cycle_id_safe`` live here (canonical location per
    spec §7.2.1). ``scan_center`` re-exports them for back-compat with the six
    existing call sites that already import from ``scan_center``.
  * Bounded in-memory ring (``maxlen=10_000``) under an RLock.
  * Per-cycle JSONL segments under ``_AUDIT_DIR/<YYYY-MM-DD>/<cycle_id>.jsonl``
    (P12). Atomic write: tmp → fsync → os.replace → fsync(parent). The
    background flusher drains every ``_AUDIT_FLUSH_MAXAGE`` seconds; a
    ``_schedule_flush`` trigger fires when any cycle's buffer reaches
    ``_AUDIT_FLUSH_MAXROWS``.
  * ``log_stage`` is a no-raise hot-path helper: the outer try/except wraps
    both the ring append AND the new disk-buffer append, so a broken FS must
    never cascade into the scan loop.
  * ``snapshot_audit_ring`` is the supported external reader for recent data.
    ``read_segment(cycle_id)`` is the supported external reader for cold data.

Self-tests run when executed directly::

    venv/bin/python scan_audit.py
"""

from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# cycle_id (§7.2.1 — canonical location)
# ---------------------------------------------------------------------------


def new_cycle_id(source: str = "tick") -> str:
    """Format: ``{source}-{unix_ts}-{hex8}``.

    Example: ``tick-1735061400-a3f9c2b1``. This is the single canonical
    producer; ``scan_center`` re-exports it for back-compat with the six
    existing callers that still import from ``scan_center``.
    """
    return f"{source}-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def new_cycle_id_safe(source: str = "manual") -> str:
    """Fallback-tolerant wrapper around :func:`new_cycle_id`.

    Tries the normal path first; on ANY exception (e.g. someone monkey-patched
    ``new_cycle_id`` in a test, or a transient internal error), synthesizes a
    cycle_id with the SAME ``{source}-{ts}-{hex8}`` shape using a fresh uuid4
    so the fallback stays globally unique. Previously a literal
    ``<source>-<ts>-00000000`` suffix would collide across concurrent
    triggers in the same second.
    """
    try:
        return new_cycle_id(source)
    except Exception:  # noqa: BLE001 — safe fallback
        return f"{source}-{int(time.time())}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# StageRecord + ring buffer (§7.2)
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """One audit event. ``payload`` is a JSON-serializable dict; schema varies
    by ``stage`` (see §7.2.2 call-site map)."""
    cycle_id: str
    timestamp: float
    stage: str          # "universe" | "raw_signal" | "filter" | "opinion" | "decision"
    payload: dict


_AUDIT_RING_MAXLEN = 10_000
_audit_ring: "collections.deque[StageRecord]" = collections.deque(maxlen=_AUDIT_RING_MAXLEN)
_audit_lock: threading.RLock = threading.RLock()

# Counter bumped whenever log_stage internally swallows an error so ops can
# spot silent payload bugs via the monitor UI.
_audit_log_errors: "dict[str, int]" = {"serialize": 0, "other": 0}


# ---------------------------------------------------------------------------
# P12 §7.5 — disk durability: per-cycle JSONL segments, flusher, recovery
# ---------------------------------------------------------------------------

# Directory root is env-configurable (SCAN_AUDIT_DIR). Tests override via
# set_audit_dir() so they can point at a TemporaryDirectory.
_AUDIT_DIR: Path = Path(os.environ.get("SCAN_AUDIT_DIR", str(Path(__file__).resolve().parent / ".cache" / "scan_audit")))

# Flush policy:
#   * _AUDIT_FLUSH_MAXAGE — background flusher wakes every N seconds
#   * _AUDIT_FLUSH_MAXROWS — log_stage schedules immediate flush when a
#     cycle's buffer reaches this size (reduces data loss on crash).
#   * _AUDIT_MAX_FLUSH_RETRIES — after this many consecutive flush failures
#     for a given cycle_id, drop its buffer to prevent unbounded growth
#     (I2 fix). The ring record is still preserved; only the disk record
#     is lost for that cycle.
_AUDIT_FLUSH_MAXAGE: float = 2.0
_AUDIT_FLUSH_MAXROWS: int = 256
_AUDIT_MAX_FLUSH_RETRIES: int = 5

# Per-cycle pending-write buffers. All mutations under _audit_buffers_lock.
_audit_buffers: "dict[str, list[StageRecord]]" = {}
_audit_buffers_lock: threading.RLock = threading.RLock()

# Per-cycle consecutive-flush-failure counter (I2). Incremented on each
# failing _flush_buffer call; cleared on success or when the buffer is
# dropped after hitting _AUDIT_MAX_FLUSH_RETRIES. All mutations under
# _audit_buffers_lock.
_audit_flush_failures: "dict[str, int]" = {}

# Observability: surfaced via /scan/cache/stats → "audit_flush". All writes
# under _audit_buffers_lock (same lesson as P8 I2 — one lock, not two).
_audit_flush_stats: "dict[str, int]" = {
    "flushed_segments": 0,
    "flush_errors": 0,
    "corrupt_recovered": 0,
    "dropped_cycles": 0,
}

# Flusher thread lifecycle (Event-based for deterministic shutdown in tests).
_audit_flusher_stop: threading.Event = threading.Event()
_audit_flusher_thread: "Optional[threading.Thread]" = None


def set_audit_dir(path: "Path | str") -> None:
    """Test-only: override ``_AUDIT_DIR``.

    Callers must ensure no flush is in flight (``stop_audit_flusher()`` and
    drain any scheduled single-shot threads). Production code must NOT call
    this — the env var ``SCAN_AUDIT_DIR`` is the supported knob.
    """
    global _AUDIT_DIR
    _AUDIT_DIR = Path(path)


def _segment_path(cycle_id: str) -> Path:
    """Return the on-disk segment path for ``cycle_id``.

    Format: ``{_AUDIT_DIR}/<YYYY-MM-DD>/<cycle_id>.jsonl``, where the date is
    derived from the ts portion of the cycle_id (``{source}-{ts}-{hex8}``).

    Creates the parent date directory as a side effect (``mkdir(parents=True,
    exist_ok=True)``).

    Raises ``ValueError`` if ``cycle_id`` is malformed (doesn't match the
    expected shape) or the literal sentinel ``"unknown"``. The caller must
    catch and skip the disk side gracefully — this is expected behavior for
    falsy cycle_ids from the hot path.
    """
    if not cycle_id or cycle_id == "unknown":
        raise ValueError(f"invalid cycle_id for disk flush: {cycle_id!r}")
    parts = cycle_id.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"invalid cycle_id for disk flush: {cycle_id!r} — expected "
            f"'source-ts-hex8' shape"
        )
    try:
        ts_int = int(parts[-2])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid cycle_id for disk flush: {cycle_id!r} — ts portion "
            f"not an integer ({exc})"
        ) from None
    date_str = datetime.fromtimestamp(ts_int).strftime("%Y-%m-%d")
    date_dir = _AUDIT_DIR / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir / f"{cycle_id}.jsonl"


def _flush_buffer(cycle_id: str) -> bool:
    """Atomically append all buffered records for ``cycle_id`` to disk.

    Sequence: pop buffer under lock → resolve segment path → read existing
    content → write tmp with (existing + new_lines) → fsync tmp →
    ``os.replace(tmp, final)`` → fsync parent dir.

    Returns True on success OR on an expected skip (malformed cycle_id).
    Returns False on any I/O error; in that case the records are requeued
    onto ``_audit_buffers[cycle_id]`` for retry on the next flusher tick,
    ``_audit_flush_stats['flush_errors']`` is incremented, and the tmp file
    is best-effort cleaned up.

    Never raises — the outer try/except swallows every exception path.
    """
    # DESIGN NOTE (I1): If a _flush_buffer call fails AFTER a concurrent
    # flush from another trigger has already popped + successfully written
    # newer records to the same segment file, the requeued-on-failure
    # records from the first call will be appended to the file in a
    # position AFTER those newer records. This violates strict logical
    # ordering within a cycle's segment. In practice this is rare (requires
    # a specific failure + interleaving), and consumers that need strict
    # order should sort by StageRecord.timestamp rather than file position.
    # A stricter alternative would be a per-cycle serialization lock; we
    # opted for throughput over strict-order under contention.
    with _audit_buffers_lock:
        buf = _audit_buffers.pop(cycle_id, None)
    if not buf:
        return True

    try:
        final = _segment_path(cycle_id)
    except ValueError as exc:
        # Expected skip: cycle_id is "unknown" / malformed. Don't count as
        # error — we deliberately drop the disk side for these.
        logger.warning(
            "scan_audit._flush_buffer: skipping disk flush for %s (%s); "
            "dropping %d buffered record(s)",
            cycle_id, exc, len(buf),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — e.g. PermissionError on mkdir
        # Real I/O failure (can't create the parent dir). Treat as a flush
        # error: requeue + bump stat so ops can spot persistent FS problems.
        with _audit_buffers_lock:
            existing_buf = _audit_buffers.setdefault(cycle_id, [])
            _audit_buffers[cycle_id] = buf + existing_buf
            _audit_flush_stats["flush_errors"] += 1
            # I2: bump per-cycle failure counter; drop buffer if stuck.
            _audit_flush_failures[cycle_id] = (
                _audit_flush_failures.get(cycle_id, 0) + 1
            )
            if _audit_flush_failures[cycle_id] >= _AUDIT_MAX_FLUSH_RETRIES:
                # Persistent failure: drop the buffer, reset counter, bump
                # drop stat. Better to lose N records than OOM the process.
                _audit_buffers.pop(cycle_id, None)
                _audit_flush_failures.pop(cycle_id, None)
                _audit_flush_stats["dropped_cycles"] += 1
                logger.error(
                    "scan_audit: dropping cycle_id=%s buffer after %d "
                    "consecutive flush failures — ring record is still "
                    "preserved; disk record lost",
                    cycle_id, _AUDIT_MAX_FLUSH_RETRIES,
                )
        logger.error(
            "scan_audit._flush_buffer: segment path resolution failed for "
            "cycle_id=%s err=%r (requeued %d record(s))",
            cycle_id, exc, len(buf),
        )
        return False

    tmp = final.with_suffix(".jsonl.tmp")
    existing = ""
    try:
        if final.exists():
            try:
                existing = final.read_text(encoding="utf-8")
            except Exception as read_exc:  # noqa: BLE001
                logger.warning(
                    "scan_audit._flush_buffer: read existing segment failed "
                    "%s (%s) — starting fresh",
                    final, read_exc,
                )
                existing = ""
        # Build the new chunk. ``sort_keys=True`` stabilizes output for tests.
        new_lines = "\n".join(
            json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True)
            for rec in buf
        ) + "\n"
        # Write tmp + fsync. Re-assert the parent dir right before the write:
        # _segment_path() created it, but production showed intermittent
        # FileNotFoundError here (2026-07-06) — a cleanup / date-rollover race
        # can remove the dir between path resolution and this open.
        final.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            if existing:
                f.write(existing)
            f.write(new_lines)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replace.
        os.replace(tmp, final)
        # fsync parent dir (best-effort; some filesystems don't need it but
        # it's cheap insurance on crash durability).
        try:
            fd = os.open(str(final.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as dir_exc:  # noqa: BLE001
            logger.debug(
                "scan_audit._flush_buffer: parent fsync skipped for %s (%s)",
                final.parent, dir_exc,
            )
        with _audit_buffers_lock:
            _audit_flush_stats["flushed_segments"] += 1
            # I2: success clears any prior failure streak for this cycle.
            _audit_flush_failures.pop(cycle_id, None)
        logger.debug(
            "scan_audit._flush_buffer: wrote %d record(s) to %s",
            len(buf), final,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        # Clean up tmp if present.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass
        # Requeue records so next flusher tick retries.
        with _audit_buffers_lock:
            existing_buf = _audit_buffers.setdefault(cycle_id, [])
            # Prepend the unflushed records so they stay in order if new
            # records arrived during our attempt.
            _audit_buffers[cycle_id] = buf + existing_buf
            _audit_flush_stats["flush_errors"] += 1
            # I2: bump per-cycle failure counter; drop buffer if stuck.
            _audit_flush_failures[cycle_id] = (
                _audit_flush_failures.get(cycle_id, 0) + 1
            )
            if _audit_flush_failures[cycle_id] >= _AUDIT_MAX_FLUSH_RETRIES:
                # Persistent failure: drop the buffer, reset counter, bump
                # drop stat. Better to lose N records than OOM the process.
                _audit_buffers.pop(cycle_id, None)
                _audit_flush_failures.pop(cycle_id, None)
                _audit_flush_stats["dropped_cycles"] += 1
                logger.error(
                    "scan_audit: dropping cycle_id=%s buffer after %d "
                    "consecutive flush failures — ring record is still "
                    "preserved; disk record lost",
                    cycle_id, _AUDIT_MAX_FLUSH_RETRIES,
                )
        logger.error(
            "scan_audit._flush_buffer: flush FAILED for cycle_id=%s err=%r "
            "(requeued %d record(s))",
            cycle_id, exc, len(buf),
        )
        return False


def _schedule_flush(cycle_id: str) -> None:
    """Non-blocking trigger to flush a cycle's buffer immediately.

    Spawns a short-lived daemon thread that calls ``_flush_buffer``. Under
    concurrent triggers, ``_flush_buffer`` pops the buffer under the lock so
    only one thread does real work; the others find an empty buffer and
    return.
    """
    try:
        threading.Thread(
            target=_flush_buffer,
            args=(cycle_id,),
            name=f"scan_audit_flush_{cycle_id}",
            daemon=True,
        ).start()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "scan_audit._schedule_flush: spawn failed cycle_id=%s err=%r",
            cycle_id, exc,
        )


def _audit_flusher_loop() -> None:
    """Background flusher body. Terminates cleanly on _audit_flusher_stop.

    Uses ``Event.wait(timeout)`` rather than ``time.sleep`` so tests can
    interrupt the thread immediately. Each tick: snapshot the keys set
    (under lock), release the lock, then call ``_flush_buffer`` per key
    (each acquires the lock for its own pop).
    """
    while not _audit_flusher_stop.wait(_AUDIT_FLUSH_MAXAGE):
        with _audit_buffers_lock:
            pending = list(_audit_buffers.keys())
        for cid in pending:
            try:
                _flush_buffer(cid)
            except Exception as exc:  # noqa: BLE001 — _flush_buffer is
                # already no-raise, but defensive belt.
                logger.error(
                    "scan_audit._audit_flusher_loop: unexpected escape for "
                    "cycle_id=%s err=%r",
                    cid, exc,
                )


def start_audit_flusher() -> None:
    """Idempotently start the background flusher thread.

    Safe to call multiple times; subsequent calls are a no-op when a live
    thread is already running.
    """
    global _audit_flusher_thread
    if _audit_flusher_thread is not None and _audit_flusher_thread.is_alive():
        return
    _audit_flusher_stop.clear()
    _audit_flusher_thread = threading.Thread(
        target=_audit_flusher_loop,
        name="scan_audit_flusher",
        daemon=True,
    )
    _audit_flusher_thread.start()
    logger.info("scan_audit: flusher thread started (dir=%s)", _AUDIT_DIR)


def stop_audit_flusher(timeout: float = 3.0) -> None:
    """Signal the flusher to exit and wait up to ``timeout`` seconds.

    Called by tests between scenarios to ensure deterministic state. In
    production, the thread is a daemon so it dies with the process — no
    explicit shutdown is wired at process-exit time.
    """
    _audit_flusher_stop.set()
    if _audit_flusher_thread is not None:
        _audit_flusher_thread.join(timeout=timeout)


def log_stage(cycle_id: str, stage: str, payload: dict) -> None:
    """Append one stage event to the in-memory ring + disk buffer. Never raises.

    Contract:
      * ``cycle_id`` — caller should pass a valid id produced by
        ``new_cycle_id``; falsy values fall through to ``"unknown"`` so orphan
        events are still visible in the monitor tab (rather than being dropped
        and silently hiding a bug). ``"unknown"`` cycle_ids are NOT written to
        disk — the segment-path derivation requires a parseable ts.
      * ``stage`` — one of the five stage names from §7.2.2.
      * ``payload`` — must be JSON-serializable. If not, we log at WARNING,
        bump ``_audit_log_errors["serialize"]``, and drop the record.

    The entire body is wrapped in try/except Exception so no caller has to
    guard their own call site — the five wiring points (§7.2.2) call
    ``log_stage(...)`` directly. This umbrella also protects the new P12
    disk-buffer append path.
    """
    try:
        cid = cycle_id if cycle_id else "unknown"
        # Payload must be a dict; other types (set, custom object) are a
        # programmer bug — drop + count.
        if not isinstance(payload, dict):
            with _audit_lock:
                _audit_log_errors["serialize"] += 1
            logger.warning(
                "scan_audit.log_stage: payload must be dict, got %s — dropping "
                "cycle_id=%s stage=%s",
                type(payload).__name__, cid, stage,
            )
            return
        # Best-effort serializability probe. We don't store the serialized
        # string — just verify the dict is safe to read later. We use strict
        # json (no ``default=str``) so programmer errors like stuffing a
        # ``set`` into a payload get caught here rather than crashing the
        # monitor endpoint; callers already convert dataclasses / datetimes
        # explicitly before calling us (see §7.2.2 wiring).
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as exc:
            with _audit_lock:
                _audit_log_errors["serialize"] += 1
            logger.warning(
                "scan_audit.log_stage: payload not JSON-serializable "
                "cycle_id=%s stage=%s err=%r",
                cid, stage, exc,
            )
            return
        rec = StageRecord(
            cycle_id=cid,
            timestamp=time.time(),
            stage=stage,
            payload=payload,
        )
        with _audit_lock:
            _audit_ring.append(rec)
        # P12: also buffer for per-cycle JSONL segment. Skip "unknown" — the
        # segment-path derivation needs a parseable ts.
        if cid != "unknown":
            with _audit_buffers_lock:
                _audit_buffers.setdefault(cid, []).append(rec)
                buffer_len = len(_audit_buffers[cid])
            if buffer_len >= _AUDIT_FLUSH_MAXROWS:
                _schedule_flush(cid)
    except Exception as exc:  # noqa: BLE001 — must never raise
        with _audit_lock:
            _audit_log_errors["other"] += 1
        logger.warning(
            "scan_audit.log_stage: unexpected error cycle_id=%s stage=%s err=%r",
            cycle_id, stage, exc,
        )


def snapshot_audit_ring(limit: Optional[int] = None) -> "list[dict]":
    """Return a list-of-dict snapshot of the ring under the lock.

    ``limit=None`` → copy the entire ring (up to 10k entries — cheap).
    ``limit=int``  → return the most-recent ``limit`` entries. Useful for the
    ``/cycles`` endpoint which only needs the tail.

    Each entry is ``asdict(StageRecord)`` — router code must never depend on
    the dataclass instance. This is the only supported external reader.
    """
    with _audit_lock:
        if limit is not None and limit < len(_audit_ring):
            # Slice from the tail. ``deque`` doesn't support slicing, so
            # materialize once then tail.
            items = list(_audit_ring)[-limit:]
        else:
            items = list(_audit_ring)
    return [asdict(rec) for rec in items]


def audit_log_error_counts() -> dict:
    """Return a snapshot of internal drop counters. Exposed for ops visibility
    (e.g. if the monitor UI ever shows a 'dropped payloads' stat)."""
    with _audit_lock:
        return dict(_audit_log_errors)


def snapshot_audit_flush_stats() -> dict:
    """Return a lock-protected copy of the flush stats block.

    Surfaced at ``GET /scan/cache/stats`` under the ``audit_flush`` key.
    Shape::

        {"flushed_segments": N, "flush_errors": N,
         "corrupt_recovered": N, "dropped_cycles": N}

    ``dropped_cycles`` (P12 polish, I2) counts cycles whose buffers were
    abandoned after ``_AUDIT_MAX_FLUSH_RETRIES`` consecutive flush failures
    to prevent unbounded memory growth on persistent FS errors.
    """
    with _audit_buffers_lock:
        return dict(_audit_flush_stats)


def read_segment(cycle_id: str) -> "Optional[list[dict]]":
    """Read the on-disk segment for ``cycle_id`` and return the parsed rows.

    Returns ``None`` if the segment file doesn't exist (or ``cycle_id`` is
    malformed / "unknown" — the monitor router treats that as 404). Otherwise
    returns a list of dicts in write order. Malformed JSON lines (e.g. a
    trailing truncated write that recovery missed) are skipped silently at
    DEBUG.

    This is the router-side fallback path: cycles that aged out of the
    in-memory ring live only on disk.
    """
    try:
        path = _segment_path(cycle_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    rows: "list[dict]" = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "scan_audit.read_segment: skipping bad line %d of "
                        "%s (%s)",
                        lineno, path, exc,
                    )
                    continue
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scan_audit.read_segment: read failed for %s (%s)",
            path, exc,
        )
        return None
    return rows


def replay_recent_segments_to_ring(max_cycles: int = 50) -> int:
    """Backfill the in-memory audit ring from on-disk JSONL segments.

    Why this exists: the in-memory ``_audit_ring`` is process-local, so on
    every restart it starts empty. ``GET /scan/monitor/cycles`` reads from
    the ring, so the scan-monitor UI shows "no cycles" for ~5 min until
    the next scan tick refills the ring. This is jarring (the user's
    Stage 0 merged pool reads "0 stocks") and historically masks the fact
    that **the data is on disk** — just not loaded.

    Strategy:
      * Walk the ``_AUDIT_DIR`` tree, find all date subdirectories
        (``YYYY-MM-DD``), pick the most-recent N (by directory name lex
        order — works because of the ISO-8601 format).
      * Within each, list all ``*.jsonl`` segments by mtime descending.
      * For each segment (up to ``max_cycles`` total across all dates),
        call :func:`read_segment` which returns parsed dicts; convert
        each back to a ``StageRecord`` and ``deque.append`` into
        ``_audit_ring`` under the ring lock.

    Returns the number of records added to the ring.

    Never raises. Per-file errors are logged at WARNING and skipped.
    A fresh install where ``_AUDIT_DIR`` doesn't exist yet returns 0.

    Cycles are appended in cycle-id timestamp order (oldest first) so the
    ring keeps its invariant: ring[-1] is the most-recent record. The
    monitor UI's ``snapshot_audit_ring()`` already returns dicts in ring
    order, so /cycles will report the newest cycle first as expected.

    The cap (``max_cycles``) bounds startup latency. With 100 records per
    cycle and 50 cycles backfilled, that's 5,000 records — half the
    ring's capacity (10,000), leaves room for live cycles.
    """
    if not _AUDIT_DIR.exists():
        return 0

    # Collect (cycle_id, jsonl_path) tuples across the most-recent N date
    # directories. We use directory name (ISO-8601 date) for ordering so a
    # filesystem-stat-free walk is enough.
    try:
        date_dirs = sorted(
            (d for d in _AUDIT_DIR.iterdir() if d.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scan_audit.replay_recent_segments_to_ring: iterdir failed: %r",
            exc,
        )
        return 0

    # Walk newest-date-first, collect segments mtime-desc, capped at
    # ``max_cycles`` total. Stop early once we have enough.
    candidates: "list[tuple[str, Path]]" = []
    for ddir in date_dirs:
        try:
            segs = list(ddir.glob("*.jsonl"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scan_audit.replay_recent_segments_to_ring: glob in %s "
                "failed: %r",
                ddir, exc,
            )
            continue
        # mtime desc — most recently flushed cycle wins ties on the same day.
        try:
            segs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:  # noqa: BLE001 — best-effort ordering
            pass
        for seg in segs:
            cycle_id = seg.stem  # filename without ".jsonl"
            candidates.append((cycle_id, seg))
            if len(candidates) >= max_cycles:
                break
        if len(candidates) >= max_cycles:
            break

    if not candidates:
        return 0

    # We want OLDEST first in the ring (so the ring invariant
    # "most-recent at the tail" holds). candidates is currently
    # newest-first, so reverse before appending.
    candidates.reverse()

    appended = 0
    skipped_files = 0
    for cycle_id, seg_path in candidates:
        try:
            rows = read_segment(cycle_id)
        except Exception as exc:  # noqa: BLE001 — read_segment is no-raise
            # but defensive belt
            logger.warning(
                "scan_audit.replay_recent_segments_to_ring: read_segment(%s) "
                "raised %r",
                cycle_id, exc,
            )
            skipped_files += 1
            continue
        if not rows:
            skipped_files += 1
            continue
        # Within a single segment, rows are already in write order
        # (read_segment preserves file order). Append them as-is.
        with _audit_lock:
            for row in rows:
                try:
                    rec = StageRecord(
                        cycle_id=row.get("cycle_id", cycle_id),
                        timestamp=float(row.get("timestamp", 0.0)),
                        stage=row.get("stage", "unknown"),
                        payload=row.get("payload", {}) if isinstance(
                            row.get("payload"), dict
                        ) else {},
                    )
                    _audit_ring.append(rec)
                    appended += 1
                except Exception as exc:  # noqa: BLE001 — bad row, skip
                    logger.debug(
                        "scan_audit.replay_recent_segments_to_ring: "
                        "skipping bad row in %s: %r",
                        seg_path, exc,
                    )
    logger.info(
        "scan_audit.replay_recent_segments_to_ring: backfilled %d record(s) "
        "from %d cycle segment(s) (%d skipped); ring now %d/%d",
        appended, len(candidates) - skipped_files, skipped_files,
        len(_audit_ring), _AUDIT_RING_MAXLEN,
    )
    return appended


def recover_audit_segments() -> int:
    """Startup pass over ``_AUDIT_DIR``. Returns count of orphaned tmp files cleaned.

    Two responsibilities:
      1. Delete any leftover ``*.jsonl.tmp`` files (crashed mid-flush).
      2. Scan every ``*.jsonl`` segment; if the trailing line fails JSON
         parse (crash between write and fsync), truncate the partial tail
         and bump ``_audit_flush_stats['corrupt_recovered']``.

    Never errors on a fresh install where ``_AUDIT_DIR`` doesn't exist
    yet — returns 0 in that case. Every per-file operation is wrapped in
    try/except so one bad file doesn't abort recovery.
    """
    if not _AUDIT_DIR.exists():
        return 0
    cleaned = 0
    # Pass 1: orphaned tmp files.
    try:
        tmp_iter = list(_AUDIT_DIR.rglob("*.jsonl.tmp"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scan_audit.recover_audit_segments: rglob(*.jsonl.tmp) failed: %r",
            exc,
        )
        tmp_iter = []
    for tmp in tmp_iter:
        try:
            tmp.unlink()
            cleaned += 1
            logger.info(
                "scan_audit.recover_audit_segments: removed orphaned tmp %s",
                tmp,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scan_audit.recover_audit_segments: unlink %s failed: %r",
                tmp, exc,
            )
    # Pass 2: validate trailing JSON of each segment.
    try:
        jsonl_iter = list(_AUDIT_DIR.rglob("*.jsonl"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scan_audit.recover_audit_segments: rglob(*.jsonl) failed: %r",
            exc,
        )
        jsonl_iter = []
    for seg in jsonl_iter:
        try:
            content = seg.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scan_audit.recover_audit_segments: read %s failed: %r",
                seg, exc,
            )
            continue
        if not content:
            continue
        lines = content.split("\n")
        # read_text leaves a trailing "" when the file ends with "\n";
        # drop it for validation.
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if not lines:
            continue
        last = lines[-1]
        try:
            json.loads(last)
        except (TypeError, ValueError):
            # Truncate the partial trailing line and rewrite.
            good = lines[:-1]
            try:
                new_content = ("\n".join(good) + "\n") if good else ""
                seg.write_text(new_content, encoding="utf-8")
                with _audit_buffers_lock:
                    _audit_flush_stats["corrupt_recovered"] += 1
                logger.warning(
                    "scan_audit.recover_audit_segments: truncated partial "
                    "trailing line in %s (kept %d good line(s))",
                    seg, len(good),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "scan_audit.recover_audit_segments: truncate %s failed: %r",
                    seg, exc,
                )
    return cleaned


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
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

    print("=== scan_audit self-test ===")

    # new_cycle_id format
    cid = new_cycle_id("tick")
    parts = cid.split("-")
    check(
        parts[0] == "tick" and parts[1].isdigit() and len(parts[2]) == 8,
        f"new_cycle_id format (got {cid})",
    )

    # log_stage + snapshot_audit_ring roundtrip
    with _audit_lock:
        _audit_ring.clear()
    log_stage("tick-1-abc", "universe", {"symbols": ["AAPL", "NVDA"]})
    log_stage("tick-1-abc", "raw_signal", {"symbol": "AAPL"})
    snap = snapshot_audit_ring()
    check(
        len(snap) == 2 and snap[0]["stage"] == "universe" and snap[1]["stage"] == "raw_signal",
        f"snapshot roundtrip ({len(snap)} records)",
    )

    # log_stage swallows non-dict payloads
    pre = audit_log_error_counts()["serialize"]
    log_stage("tick-2-def", "filter", {"bad": set([1, 2])})  # set isn't JSON-able
    post = audit_log_error_counts()["serialize"]
    check(
        post > pre,
        f"log_stage drops non-serializable payload (err count {pre}→{post})",
    )

    # log_stage falsy cycle_id → "unknown"
    with _audit_lock:
        _audit_ring.clear()
    log_stage("", "universe", {"x": 1})
    log_stage(None, "raw_signal", {"x": 2})  # type: ignore[arg-type]
    snap = snapshot_audit_ring()
    check(
        all(rec["cycle_id"] == "unknown" for rec in snap),
        f"falsy cycle_id folds to 'unknown' (got {[r['cycle_id'] for r in snap]})",
    )

    print(f"\n  {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
