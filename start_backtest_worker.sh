#!/bin/bash
# start_backtest_worker.sh — launcher for the backtest worker (本机).
#
# Env vars expected in .env (alongside this script):
#   DATABASE_URL                 本机 PostgreSQL
#   AITRADER_INTERNAL_SECRET        token for /internal/ohlcv
#   AITRADER_OHLCV_PROXY            http://127.0.0.1:9000
#   BACKTEST_WORKER_ID           optional — defaults to $HOSTNAME-$$
#   BACKTEST_WORKER_CONCURRENCY  optional — default 4

set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

# venv selection: try venv/bin/python (Linux/macOS); fall back to plain python3
if [ -x venv/bin/python ]; then
  PY="venv/bin/python"
elif command -v python3.13 >/dev/null 2>&1; then
  PY="python3.13"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "ERROR: no Python 3.13 found and no venv/" >&2
  exit 1
fi

echo "=== backtest_worker starting at $(date) ==="
echo "  worker_id:   ${BACKTEST_WORKER_ID:-auto}"
echo "  concurrency: ${BACKTEST_WORKER_CONCURRENCY:-4}"
echo "  proxy:       ${AITRADER_OHLCV_PROXY:-http://127.0.0.1:9000}"
echo "  DB:          ${DATABASE_URL:?DATABASE_URL not set}"

exec "$PY" backtest_worker.py \
    --concurrency "${BACKTEST_WORKER_CONCURRENCY:-4}" \
    ${BACKTEST_WORKER_ID:+--worker-id "$BACKTEST_WORKER_ID"} \
    ${AITRADER_OHLCV_PROXY:+--ohlcv-proxy "$AITRADER_OHLCV_PROXY"}
