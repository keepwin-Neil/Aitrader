-- 2026-05-20: background backtest — queue + workers tables.
--
-- Forward-only DDL. Use 2026_05_20_backtest_queue.rollback.sql to undo.
-- gen_random_uuid() requires the pgcrypto extension; we enable it
-- defensively in case the DB was created without it.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ─── queue ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_queue (
  id                 BIGSERIAL PRIMARY KEY,
  task_uuid          UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
  status             TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending','processing','completed','failed','timeout','cancelled')),
  kind               TEXT NOT NULL DEFAULT 'single_backtest',
  payload            JSONB NOT NULL,

  -- worker assignment
  claimed_by         TEXT,
  claimed_at         TIMESTAMPTZ,
  claim_token        UUID,

  -- result + error
  result             JSONB,
  error_message      TEXT,

  -- timestamps
  enqueued_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at       TIMESTAMPTZ,

  priority           SMALLINT NOT NULL DEFAULT 0
);

-- Index for claim_pending (only scans the small pending set, ignores rest)
CREATE INDEX IF NOT EXISTS idx_bt_queue_claim
  ON backtest_queue (status, priority DESC, enqueued_at)
  WHERE status = 'pending';

-- Index for stale-processing reclaim sweep
CREATE INDEX IF NOT EXISTS idx_bt_queue_claimed
  ON backtest_queue (status, claimed_at)
  WHERE status = 'processing';

-- Index for batch progress queries (auto_optimize.wait_for_batch)
CREATE INDEX IF NOT EXISTS idx_bt_queue_batch
  ON backtest_queue ((payload->>'batch_id'))
  WHERE payload ? 'batch_id';


-- ─── workers ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_workers (
  worker_id          TEXT PRIMARY KEY,
  hostname           TEXT,
  platform           TEXT,
  max_concurrency    INT NOT NULL DEFAULT 4,
  status             TEXT NOT NULL DEFAULT 'online'
                       CHECK (status IN ('online','offline')),
  last_heartbeat     TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

  tasks_completed    BIGINT NOT NULL DEFAULT 0,
  tasks_failed       BIGINT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_bt_workers_status_hb
  ON backtest_workers (status, last_heartbeat);

COMMIT;
