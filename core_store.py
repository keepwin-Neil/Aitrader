from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Optional

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor


logger = logging.getLogger("uvicorn.error")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/aitrader")

_POOL: psycopg2.pool.ThreadedConnectionPool = None
_POOL_LOCK = threading.Lock()

SYSTEM_ACTOR = {"id": 0, "role": "admin", "display_name": "system"}

# Advisory lock key for init_db() cross-process serialization.
_INIT_DB_ADVISORY_LOCK_KEY = -294_705_577  # stable, unique to init_db


def _utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = psycopg2.pool.ThreadedConnectionPool(2, 20, DATABASE_URL)
    return _POOL


def _get_conn() -> Any:
    return _get_pool().getconn()


def _put_conn(conn: Any) -> None:
    _get_pool().putconn(conn)


import contextlib as _contextlib


@_contextlib.contextmanager
def _pooled_conn():
    """Checkout a pooled connection; ALWAYS return it to the pool on exit.

    ⚠️ Never write ``with _get_conn() as conn:`` — psycopg2's ``with conn:``
    only manages the *transaction* (commit/rollback); it does NOT return the
    connection to the pool. That exact mistake leaked one connection per call
    in the user_stock_pools functions and exhausted the 20-conn pool in
    production (2026-07 connection-pool exhaustion incident). Use this
    context manager instead: commit on success, rollback on error, putconn
    always.
    """
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _put_conn(conn)


def _json_dumps(data: Any) -> str:
    return json.dumps(data or {}, ensure_ascii=False, sort_keys=True)


def _parse_json(value: Any) -> Any:
    if value in (None, "", "null"):
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def init_db() -> None:
    """建表 (幂等): 策略组合 / LLM 评审 / 优化记录。"""
    statements = [
        """
        -- 策略组合: 命名的一组策略, 供研究工具 signal pairing 使用
        CREATE TABLE IF NOT EXISTS strategy_combos (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            strategies_json TEXT NOT NULL DEFAULT '[]',
            llm_owners_json TEXT NOT NULL DEFAULT '[]',
            note TEXT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS strategy_llm_reviews (
            id              SERIAL PRIMARY KEY,
            strategy_name   VARCHAR(100) NOT NULL,
            review_round    VARCHAR(50)  NOT NULL,
            signal_type     VARCHAR(20)  DEFAULT 'combined',
            eval_pool       VARCHAR(50),
            eval_interval   VARCHAR(10)  DEFAULT '1d',
            provider        VARCHAR(50)  NOT NULL,
            model           VARCHAR(100) NOT NULL,
            verdict         VARCHAR(20)  NOT NULL,
            confidence      INTEGER      NOT NULL DEFAULT 0,
            overall_score   INTEGER      DEFAULT 0,
            reasoning       TEXT,
            risk_warnings   JSONB,
            suggestions     JSONB,
            input_metrics   JSONB        NOT NULL,
            raw_response    TEXT,
            prompt_hash     VARCHAR(64),
            response_hash   VARCHAR(64),
            action_taken    VARCHAR(20),
            input_tokens    INTEGER      DEFAULT 0,
            output_tokens   INTEGER      DEFAULT 0,
            duration_ms     INTEGER      DEFAULT 0,
            created_at      TIMESTAMP    DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_slr_strategy ON strategy_llm_reviews(strategy_name)",
        "CREATE INDEX IF NOT EXISTS idx_slr_round ON strategy_llm_reviews(review_round)",
        "CREATE INDEX IF NOT EXISTS idx_slr_verdict ON strategy_llm_reviews(verdict)",
        "CREATE INDEX IF NOT EXISTS idx_slr_created ON strategy_llm_reviews(created_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id              SERIAL PRIMARY KEY,
            run_id          VARCHAR(50)  NOT NULL UNIQUE,
            scope           VARCHAR(10)  NOT NULL DEFAULT 'pool',
            pool_name       VARCHAR(50),
            symbol          VARCHAR(20),
            symbols         JSONB,
            signal_type     VARCHAR(20)  NOT NULL DEFAULT 'buy_and_sell',
            method          VARCHAR(20)  NOT NULL DEFAULT 'bayesian',
            interval        VARCHAR(10)  NOT NULL DEFAULT '1d',
            n_trials        INTEGER      DEFAULT 200,
            total_strategies INTEGER     DEFAULT 0,
            robust_count    INTEGER      DEFAULT 0,
            fair_count      INTEGER      DEFAULT 0,
            overfit_count   INTEGER      DEFAULT 0,
            total_combos    INTEGER      DEFAULT 0,
            elapsed_seconds DOUBLE PRECISION DEFAULT 0,
            per_stock       JSONB,
            summary         JSONB,
            created_at      TIMESTAMP    DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_optrun_pool ON optimization_runs(pool_name, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_optrun_signal ON optimization_runs(signal_type, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_optrun_created ON optimization_runs(created_at DESC)",
    ]
    conn = _get_conn()
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            # 多个进程同时启动时串行化建表, 避免并发 DDL 死锁。
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_INIT_DB_ADVISORY_LOCK_KEY,))
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


# ═══════════════════════════════════════════════════════════════
# strategy_llm_reviews CRUD
# ═══════════════════════════════════════════════════════════════

def _row_to_llm_review(row: dict) -> dict:
    import json as _json
    def _pj(v):
        if v is None: return None
        if isinstance(v, (dict, list)): return v
        try: return _json.loads(v)
        except Exception: return v
    return {
        "id": row["id"],
        "strategy_name": row["strategy_name"],
        "review_round": row["review_round"],
        "signal_type": row.get("signal_type", "combined"),
        "eval_pool": row.get("eval_pool"),
        "eval_interval": row.get("eval_interval", "1d"),
        "provider": row["provider"],
        "model": row["model"],
        "verdict": row["verdict"],
        "confidence": row["confidence"],
        "overall_score": row.get("overall_score", 0),
        "reasoning": row.get("reasoning"),
        "risk_warnings": _pj(row.get("risk_warnings")),
        "suggestions": _pj(row.get("suggestions")),
        "input_metrics": _pj(row.get("input_metrics")),
        "action_taken": row.get("action_taken"),
        "prompt_hash": row.get("prompt_hash"),
        "response_hash": row.get("response_hash"),
        "input_tokens": row.get("input_tokens", 0),
        "output_tokens": row.get("output_tokens", 0),
        "duration_ms": row.get("duration_ms", 0),
        "created_at": str(row.get("created_at", "")),
    }


def create_llm_review(*, strategy_name: str, review_round: str, signal_type: str = "combined",
                       eval_pool: Optional[str] = None, eval_interval: str = "1d",
                       provider: str, model: str, verdict: str, confidence: int = 0,
                       overall_score: int = 0, reasoning: Optional[str] = None,
                       risk_warnings=None, suggestions=None, input_metrics: dict = None,
                       raw_response: Optional[str] = None, prompt_hash: Optional[str] = None,
                       response_hash: Optional[str] = None, action_taken: Optional[str] = None,
                       input_tokens: int = 0, output_tokens: int = 0,
                       duration_ms: int = 0) -> dict:
    import json as _json
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO strategy_llm_reviews(
                    strategy_name, review_round, signal_type, eval_pool, eval_interval,
                    provider, model, verdict, confidence, overall_score,
                    reasoning, risk_warnings, suggestions, input_metrics,
                    raw_response, prompt_hash, response_hash, action_taken,
                    input_tokens, output_tokens, duration_ms
                ) VALUES (
                    %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s
                ) RETURNING *
            """, (
                strategy_name, review_round, signal_type, eval_pool, eval_interval,
                provider, model, verdict, confidence, overall_score,
                reasoning,
                _json.dumps(risk_warnings) if risk_warnings else None,
                _json.dumps(suggestions) if suggestions else None,
                _json.dumps(input_metrics or {}),
                raw_response, prompt_hash, response_hash, action_taken,
                input_tokens, output_tokens, duration_ms,
            ))
            row = cur.fetchone()
        conn.commit()
        return _row_to_llm_review(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def list_llm_reviews(strategy_name: Optional[str] = None, review_round: Optional[str] = None,
                     verdict: Optional[str] = None, limit: int = 100) -> list:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            clauses, params = [], []
            if strategy_name:
                clauses.append("strategy_name = %s"); params.append(strategy_name)
            if review_round:
                clauses.append("review_round = %s"); params.append(review_round)
            if verdict:
                clauses.append("verdict = %s"); params.append(verdict)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            cur.execute(f"SELECT * FROM strategy_llm_reviews {where} ORDER BY created_at DESC LIMIT %s",
                        params + [limit])
            return [_row_to_llm_review(r) for r in cur.fetchall()]
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def get_latest_review(strategy_name: str) -> Optional[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT * FROM strategy_llm_reviews
                           WHERE strategy_name = %s ORDER BY created_at DESC LIMIT 1""",
                        (strategy_name,))
            row = cur.fetchone()
        return _row_to_llm_review(row) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


# ═══════════════════════════════════════════════════════════
# optimization_runs CRUD
# ═══════════════════════════════════════════════════════════

def _row_to_opt_run(row: dict) -> dict:
    import json as _json
    def _pj(v):
        if v is None: return None
        if isinstance(v, (dict, list)): return v
        try: return _json.loads(v)
        except: return v
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "scope": row.get("scope", "pool"),
        "pool_name": row.get("pool_name"),
        "symbol": row.get("symbol"),
        "symbols": _pj(row.get("symbols")),
        "signal_type": row.get("signal_type", "buy_and_sell"),
        "method": row.get("method", "bayesian"),
        "interval": row.get("interval", "1d"),
        "n_trials": row.get("n_trials", 200),
        "total_strategies": row.get("total_strategies", 0),
        "robust_count": row.get("robust_count", 0),
        "fair_count": row.get("fair_count", 0),
        "overfit_count": row.get("overfit_count", 0),
        "total_combos": row.get("total_combos", 0),
        "elapsed_seconds": row.get("elapsed_seconds", 0),
        "per_stock": _pj(row.get("per_stock")),
        "summary": _pj(row.get("summary")),
        "created_at": str(row.get("created_at", "")),
    }


def create_optimization_run(*, run_id: str, scope: str = "pool", pool_name: str = None,
                             symbol: str = None, symbols: list = None,
                             signal_type: str = "buy_and_sell", method: str = "bayesian",
                             interval: str = "1d", n_trials: int = 200,
                             total_strategies: int = 0, robust_count: int = 0,
                             fair_count: int = 0, overfit_count: int = 0,
                             total_combos: int = 0, elapsed_seconds: float = 0,
                             per_stock: dict = None, summary: dict = None) -> dict:
    import json as _json
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO optimization_runs(
                    run_id, scope, pool_name, symbol, symbols, signal_type, method,
                    interval, n_trials, total_strategies, robust_count, fair_count,
                    overfit_count, total_combos, elapsed_seconds, per_stock, summary
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING *
            """, (
                run_id, scope, pool_name, symbol,
                _json.dumps(symbols) if symbols else None,
                signal_type, method, interval, n_trials,
                total_strategies, robust_count, fair_count, overfit_count,
                total_combos, elapsed_seconds,
                _json.dumps(per_stock) if per_stock else None,
                _json.dumps(summary) if summary else None,
            ))
            row = cur.fetchone()
        conn.commit()
        return _row_to_opt_run(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def list_optimization_runs(pool_name: str = None, signal_type: str = None,
                           limit: int = 50) -> list:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            clauses, params = [], []
            if pool_name:
                clauses.append("pool_name = %s"); params.append(pool_name)
            if signal_type:
                clauses.append("signal_type = %s"); params.append(signal_type)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            cur.execute(f"""SELECT id, run_id, scope, pool_name, symbol, signal_type, method,
                            interval, n_trials, total_strategies, robust_count, fair_count,
                            overfit_count, total_combos, elapsed_seconds, summary, created_at
                            FROM optimization_runs {where}
                            ORDER BY created_at DESC LIMIT %s""",
                        params + [limit])
            return [_row_to_opt_run(r) for r in cur.fetchall()]
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def get_optimization_run(run_id: str) -> Optional[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM optimization_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        return _row_to_opt_run(row) if row else None
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


# =====================================================================
# strategy_combos — 保存的策略组合
#   * 研究工具 (回测/Sell 评估/策略优化) 引用组合作为 signal pairing 依据
#   * llm_owners_json: 记录组合内策略所涉及的 LLM 品牌 (可含多个)
# =====================================================================
def _row_to_strategy_combo(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "strategies": _parse_json(row.get("strategies_json")) or [],
        "llm_owners": _parse_json(row.get("llm_owners_json")) or [],
        "note": row.get("note"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _normalize_combo_strategies(strategies: Any) -> list[str]:
    if strategies is None:
        return []
    if isinstance(strategies, str):
        strategies = [strategies]
    if not isinstance(strategies, (list, tuple, set)):
        raise ValueError("strategies 必须是字符串数组")
    out: list[str] = []
    seen: set[str] = set()
    for s in strategies:
        if s is None:
            continue
        name = str(s).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _normalize_combo_owners(owners: Any) -> list[str]:
    if owners is None:
        return []
    if isinstance(owners, str):
        owners = [owners]
    if not isinstance(owners, (list, tuple, set)):
        raise ValueError("llm_owners 必须是字符串数组")
    out: list[str] = []
    seen: set[str] = set()
    for o in owners:
        if o is None:
            continue
        name = str(o).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def create_strategy_combo(
    *,
    name: str,
    strategies: list[str],
    llm_owners: Optional[list[str]] = None,
    note: Optional[str] = None,
) -> dict:
    if not name or not name.strip():
        raise ValueError("组合名称不能为空")
    strategies_norm = _normalize_combo_strategies(strategies)
    if not strategies_norm:
        raise ValueError("至少需要选择一个策略")
    owners_norm = _normalize_combo_owners(llm_owners or [])
    now = _utc_now_iso()
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT 1 FROM strategy_combos WHERE name = %s", (name.strip(),))
            if cur.fetchone():
                raise ValueError(f"已存在同名组合：{name}")
            cur.execute(
                """
                INSERT INTO strategy_combos(
                    name, strategies_json, llm_owners_json, note, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (name.strip(), _json_dumps(strategies_norm), _json_dumps(owners_norm),
                 (note or None), now, now),
            )
            row = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)
    return _row_to_strategy_combo(row)


def list_strategy_combos() -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM strategy_combos ORDER BY updated_at DESC")
            rows = cur.fetchall() or []
    finally:
        _put_conn(conn)
    return [_row_to_strategy_combo(r) for r in rows]


def get_strategy_combo(combo_id: int) -> Optional[dict]:
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM strategy_combos WHERE id = %s", (combo_id,))
            row = cur.fetchone()
    finally:
        _put_conn(conn)
    return _row_to_strategy_combo(row) if row else None


def update_strategy_combo(combo_id: int, **changes: Any) -> dict:
    allowed = {"name", "strategies", "llm_owners", "note"}
    payload: dict[str, Any] = {}
    for k, v in changes.items():
        if k not in allowed:
            continue
        if k == "note":
            payload[k] = v  # 允许清空
        elif v is not None:
            payload[k] = v
    if not payload:
        raise ValueError("没有可更新字段")
    if "strategies" in payload:
        payload["strategies"] = _normalize_combo_strategies(payload["strategies"])
        if not payload["strategies"]:
            raise ValueError("至少需要选择一个策略")
    if "llm_owners" in payload:
        payload["llm_owners"] = _normalize_combo_owners(payload["llm_owners"])
    set_parts: list[str] = []
    values: list[Any] = []
    if "name" in payload:
        set_parts.append("name = %s")
        values.append(str(payload["name"]).strip())
    if "strategies" in payload:
        set_parts.append("strategies_json = %s")
        values.append(_json_dumps(payload["strategies"]))
    if "llm_owners" in payload:
        set_parts.append("llm_owners_json = %s")
        values.append(_json_dumps(payload["llm_owners"]))
    if "note" in payload:
        set_parts.append("note = %s")
        values.append(payload["note"] or None)
    values.extend([_utc_now_iso(), combo_id])
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM strategy_combos WHERE id = %s", (combo_id,))
            if not cur.fetchone():
                raise ValueError("组合不存在")
            if "name" in payload:
                cur.execute("SELECT 1 FROM strategy_combos WHERE name = %s AND id <> %s",
                            (str(payload["name"]).strip(), combo_id))
                if cur.fetchone():
                    raise ValueError("已存在同名组合")
            cur.execute(
                f"UPDATE strategy_combos SET {', '.join(set_parts)}, updated_at = %s WHERE id = %s",
                tuple(values),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)
    return get_strategy_combo(combo_id) or {}


def delete_strategy_combo(combo_id: int) -> bool:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM strategy_combos WHERE id = %s", (combo_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)
    return deleted
