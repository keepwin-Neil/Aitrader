#!/bin/bash
# 本地开发启动脚本 — 只跑 9000 (dev main app), 用本机 TWS/IB Gateway 作行情源
#
# 只在本机运行: 只用本机的 TWS/IB Gateway + 本机 Postgres, 只监听 127.0.0.1:9000
#
# 用法:
#   ./start_local.sh                 # 前台跑, Ctrl+C 停
#
# 依赖:
#   * venv/bin/python 里装了 requirements.txt
#   * 本机 TWS (port 4001) 或 IB Gateway (port 4001/4002) 在跑
#   * 本机 Postgres (可选) — core_store.py 里 DATABASE_URL 指向本机

set -euo pipefail
cd "$(dirname "$0")"

PORT=9000

# ── IB 覆盖: 指向本机 TWS/Gateway ────────────────────────────────────────────
export IB_GATEWAY_HOST=127.0.0.1
export IB_GATEWAY_FALLBACK_HOST=127.0.0.1
export IB_GATEWAY_PORT=4001                # TWS live 或 IB Gateway live
export IB_GATEWAY_FALLBACK_PORT=4002        # 4002 = Gateway paper
export IB_MARKET_DATA_TYPE=1                # 1=live (你有订阅), 3=delayed
export IB_POOL_SIZE="${IB_POOL_SIZE:-4}"    # 本地开发用不了太多 slot, 4 够了

export IB_GATEWAY_ENABLED=1
export LIVE_STREAMING_ENABLED=0             # 本地开发先不开 streaming

# ── 路径检查 ────────────────────────────────────────────────────────────────
if [[ ! -x venv/bin/python3 ]]; then
  echo "❌ venv/bin/python3 不存在, 先跑: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "═══════════════════════════════════════════════════"
echo "  本地开发模式 — aitrader-9000 @ localhost:${PORT}"
echo "  IB Gateway: ${IB_GATEWAY_HOST}:${IB_GATEWAY_PORT} (pool=${IB_POOL_SIZE})"
echo "  IB market data type: ${IB_MARKET_DATA_TYPE} (1=live)"
echo "═══════════════════════════════════════════════════"

# ── 排除 reload-trigger 的文件 ─────────────────────────────────────────────
# 长跑 in-memory worker (新策略统一验证 / quant 池 daemon / IB pool 等) 存活在
# 主进程内, 任何 .py 改动触发 uvicorn --reload 都会 kill 它们. 为了让"改
# 前端 + 加端点 + 调 stock_report" 不误杀长跑 worker, 把最常改的模块
# 从 reload 触发中排除掉. 修完这些文件需要手动 kill+重启 server 才生效.
#
# 想在 dev 里保留 hot-reload 的模块 (main.py 本身 / 少量高频改的入口)
# 就不在下面列表里.
RELOAD_EXCLUDE=(
  "quant_engine.py"           # quant 60s tick daemon 状态在这
  "quant_store.py"
  "quant_borrow.py"
  "quant_llm.py"
  "stock_report.py"
  "news_scorer.py"
  "news_provider.py"
  "news_pools.py"
  "news_daily_report.py"
  "news_auto_scanner.py"
  "routers/quant.py"
  "routers/stock_report.py"
  "routers/scan.py"
)
EXCLUDE_ARGS=()
for f in "${RELOAD_EXCLUDE[@]}"; do
  EXCLUDE_ARGS+=(--reload-exclude "$f")
done

exec venv/bin/uvicorn main:app --host 127.0.0.1 --port "${PORT}" --reload "${EXCLUDE_ARGS[@]}"
