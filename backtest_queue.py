"""PG-backed queue for background backtest tasks.

PostgreSQL SKIP LOCKED queue for backtest payloads.

All functions are thread-safe (psycopg2 connection pool handles that). The only
thing they need is the local PG at DATABASE_URL. Every public function uses the ``_conn()`` context manager
so connections always go back to the pool, even on error.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import time
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json

from core_store import _get_conn, _put_conn

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _conn():
    """Acquire a connection, commit on clean exit, rollback on exception,
    always return it to the pool.

    Lets every consumer skip the boilerplate `try/except: rollback; commit`
    dance — they just open a `with _conn() as c:` block and run queries.
    Read-only callers still get a (cheap, no-op) commit, which is fine.
    """
    c = _get_conn()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        try:
            _put_conn(c)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Enqueue (producer side — coordinator / auto_optimize calls these)
# ─────────────────────────────────────────────────────────────────────────────


def enqueue_task(
    payload: dict,
    *,
    kind: str = "single_backtest",
    batch_id: Optional[str] = None,
    priority: int = 0,
) -> str:
    """Push one task. Returns task_uuid as str."""
    if batch_id and "batch_id" not in payload:
        payload = {**payload, "batch_id": batch_id}
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO backtest_queue (kind, payload, priority) "
            "VALUES (%s, %s, %s) RETURNING task_uuid",
            (kind, Json(payload), priority),
        )
        return str(cur.fetchone()[0])


def enqueue_batch(
    payloads: list[dict],
    *,
    batch_id: str,
    kind: str = "single_backtest",
    priority: int = 0,
) -> list[str]:
    """Bulk insert; returns ordered task_uuids."""
    from psycopg2.extras import execute_values
    if not payloads:
        return []
    enriched = [{**p, "batch_id": batch_id} for p in payloads]
    with _conn() as conn, conn.cursor() as cur:
        uuids = execute_values(
            cur,
            "INSERT INTO backtest_queue (kind, payload, priority) "
            "VALUES %s RETURNING task_uuid",
            [(kind, Json(p), priority) for p in enriched],
            template="(%s, %s, %s)",
            fetch=True,
        )
        return [str(u[0]) for u in uuids]


def wait_for_batch(
    batch_id: str,
    *,
    timeout: float = 3600.0,
    poll_interval: float = 2.0,
    progress_cb=None,
) -> dict:
    """Block until every task in the batch is in a terminal state.

    Returns ``{task_uuid: {'status': ..., 'result': ..., 'error_message': ...}}``.
    Raises TimeoutError if `timeout` elapses with tasks still in flight.
    """
    deadline = time.time() + timeout
    while True:
        progress = batch_progress(batch_id)
        if progress["pending"] == 0 and progress["processing"] == 0:
            return _batch_results(batch_id)
        if progress_cb is not None:
            progress_cb(progress["completed"] + progress["failed"], progress["total"])
        if time.time() > deadline:
            raise TimeoutError(
                f"batch {batch_id} did not finish within {timeout}s "
                f"(progress: {progress})"
            )
        time.sleep(poll_interval)


def cancel_batch(batch_id: str) -> int:
    """Cancel every still-open task of a batch (pending AND processing).

    Coordinator calls this when wait_for_batch times out — otherwise the
    orphan tasks sit in the queue forever, get reclaimed in cycles, and
    churn workers long after anyone cares about the results. Returns the
    number of tasks cancelled.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_queue SET status = 'cancelled', completed_at = now(), "
            "error_message = COALESCE(error_message,'') || ' [batch cancelled by coordinator]' "
            "WHERE payload->>'batch_id' = %s AND status IN ('pending','processing')",
            (batch_id,),
        )
        n = cur.rowcount
    if n:
        log.warning("[backtest_queue] cancelled %d open task(s) of batch %s", n, batch_id)
    return n


def batch_progress(batch_id: str) -> dict:
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT status, COUNT(*) AS n FROM backtest_queue "
                "WHERE payload->>'batch_id' = %s GROUP BY status",
                (batch_id,),
            )
            rows = cur.fetchall()
    out = {"pending": 0, "processing": 0, "completed": 0, "failed": 0,
           "timeout": 0, "cancelled": 0, "total": 0}
    for r in rows:
        out[r["status"]] = int(r["n"])
        out["total"] += int(r["n"])
    return out


def _batch_results(batch_id: str) -> dict:
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT task_uuid, status, result, error_message "
                "FROM backtest_queue WHERE payload->>'batch_id' = %s",
                (batch_id,),
            )
            rows = cur.fetchall()
    return {
        str(r["task_uuid"]): {
            "status": r["status"],
            "result": r["result"],
            "error_message": r["error_message"],
        }
        for r in rows
    }


# ─────────────────────────────────────────────────────────────────────────────
# Claim (consumer side — workers call these)
# ─────────────────────────────────────────────────────────────────────────────


def claim_pending(worker_id: str, *, limit: int = 4) -> list[dict]:
    """Atomically claim up to ``limit`` pending tasks.

    Uses FOR UPDATE SKIP LOCKED so two workers calling concurrently get
    disjoint sets. Returns list of {task_uuid, kind, payload, claim_token}.
    """
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH t AS (
              SELECT id FROM backtest_queue
               WHERE status = 'pending'
               ORDER BY priority DESC, enqueued_at
               LIMIT %s
               FOR UPDATE SKIP LOCKED
            )
            UPDATE backtest_queue q
               SET status = 'processing',
                   claimed_by = %s,
                   claimed_at = now(),
                   claim_token = gen_random_uuid()
              FROM t
             WHERE q.id = t.id
             RETURNING q.task_uuid, q.kind, q.payload, q.claim_token
            """,
            (limit, worker_id),
        )
        rows = cur.fetchall()
    return [
        {
            "task_uuid": str(r["task_uuid"]),
            "kind": r["kind"],
            "payload": r["payload"],
            "claim_token": str(r["claim_token"]),
        }
        for r in rows
    ]


def complete_task(task_uuid: str, claim_token: str, result: dict) -> bool:
    """Mark task completed. Returns False if claim_token mismatched (task was
    reclaimed by another worker)."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_queue SET status = 'completed', result = %s, "
            "completed_at = now() WHERE task_uuid = %s "
            "AND claim_token::text = %s AND status = 'processing'",
            (Json(result), task_uuid, claim_token),
        )
        return cur.rowcount > 0


def fail_task(task_uuid: str, claim_token: str, error_message: str) -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_queue SET status = 'failed', error_message = %s, "
            "completed_at = now() WHERE task_uuid = %s "
            "AND claim_token::text = %s AND status = 'processing'",
            (error_message[:4000], task_uuid, claim_token),
        )
        return cur.rowcount > 0


def reclaim_stale_processing(max_processing_seconds: int = 300) -> int:
    """Re-queue any 'processing' task older than ``max_processing_seconds``.

    Returns count of reclaimed tasks. Workers run this from a background
    thread every 30s — multiple workers racing on this is fine, only one
    gets the row lock.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_queue SET status = 'pending', "
            "claimed_by = NULL, claimed_at = NULL, claim_token = NULL, "
            "error_message = COALESCE(error_message,'') || ' [reclaimed from stale processing]' "
            "WHERE status = 'processing' "
            "AND claimed_at < now() - INTERVAL '1 second' * %s",
            (max_processing_seconds,),
        )
        n = cur.rowcount
    if n > 0:
        log.warning("[backtest_queue] reclaimed %d stale processing tasks", n)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Worker registration (lifecycle)
# ─────────────────────────────────────────────────────────────────────────────


def register_worker(
    worker_id: str,
    *,
    hostname: Optional[str] = None,
    platform: Optional[str] = None,
    max_concurrency: int = 4,
) -> None:
    if hostname is None:
        hostname = socket.gethostname()
    if platform is None:
        import sys
        platform = sys.platform   # 'linux' / 'darwin' / 'win32'
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO backtest_workers
              (worker_id, hostname, platform, max_concurrency, status, last_heartbeat, started_at)
            VALUES (%s, %s, %s, %s, 'online', now(), now())
            ON CONFLICT (worker_id) DO UPDATE
              SET hostname = EXCLUDED.hostname,
                  platform = EXCLUDED.platform,
                  max_concurrency = EXCLUDED.max_concurrency,
                  status = 'online',
                  last_heartbeat = now(),
                  started_at = now()
            """,
            (worker_id, hostname, platform, max_concurrency),
        )


def heartbeat(worker_id: str) -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_workers SET last_heartbeat = now(), status = 'online' "
            "WHERE worker_id = %s",
            (worker_id,),
        )


def shutdown_worker(worker_id: str) -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_workers SET status = 'offline' WHERE worker_id = %s",
            (worker_id,),
        )


def increment_worker_counter(worker_id: str, *, completed: int = 0, failed: int = 0) -> None:
    if completed == 0 and failed == 0:
        return
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE backtest_workers SET tasks_completed = tasks_completed + %s, "
            "tasks_failed = tasks_failed + %s WHERE worker_id = %s",
            (completed, failed, worker_id),
        )


def list_workers(include_offline: bool = False) -> list[dict]:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT worker_id, hostname, platform, max_concurrency, status, "
            "last_heartbeat, started_at, tasks_completed, tasks_failed "
            "FROM backtest_workers WHERE %s OR status = 'online' "
            "ORDER BY last_heartbeat DESC",
            (include_offline,),
        )
        return [dict(r) for r in cur.fetchall()]


def queue_stats() -> dict:
    """Aggregate counts for /backtest/queue/stats."""
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT status, COUNT(*) AS n FROM backtest_queue GROUP BY status")
        by_status = {r["status"]: int(r["n"]) for r in cur.fetchall()}
        cur.execute(
            "SELECT COUNT(*) AS n FROM backtest_queue "
            "WHERE status = 'completed' AND completed_at > now() - INTERVAL '1 hour'"
        )
        completed_1h = int(cur.fetchone()["n"])
        cur.execute("SELECT COUNT(*) AS n FROM backtest_workers WHERE status = 'online'")
        online_workers = int(cur.fetchone()["n"])
    return {
        "by_status": by_status,
        "completed_last_hour": completed_1h,
        "online_workers": online_workers,
    }


def list_recent(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    """Debug helper for /backtest/queue/recent."""
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if status:
            cur.execute(
                "SELECT task_uuid, status, kind, claimed_by, "
                "enqueued_at, claimed_at, completed_at, error_message "
                "FROM backtest_queue WHERE status = %s ORDER BY id DESC LIMIT %s",
                (status, limit),
            )
        else:
            cur.execute(
                "SELECT task_uuid, status, kind, claimed_by, "
                "enqueued_at, claimed_at, completed_at, error_message "
                "FROM backtest_queue ORDER BY id DESC LIMIT %s",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]
