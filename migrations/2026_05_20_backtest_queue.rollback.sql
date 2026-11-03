-- Rollback for 2026_05_20_backtest_queue.sql.

BEGIN;

DROP TABLE IF EXISTS backtest_workers;
DROP TABLE IF EXISTS backtest_queue;
-- pgcrypto is left in place (may be used by other tables)

COMMIT;
