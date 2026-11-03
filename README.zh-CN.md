# AITrader — AI 量化交易与策略优化平台

[English](README.md) | 中文

基于 **FastAPI + PostgreSQL + LLM** 的量化交易研发系统，**只在本机运行**：
多策略回测、参数优化、Walk-Forward 验证、实时扫描、新闻驱动选股、
模拟交易、LLM 策略评估与 IB Gateway 行情接入。

> 📦 仓库地址：https://github.com/keepwin-Neil/Aitrader （公开仓库）

---

## 目录

- [服务架构](#服务架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [参数分层与存储](#参数分层与存储)
- [核心模块](#核心模块)
- [LLM 调用](#llm-调用)
- [API 概览](#api-概览)
- [运维监控](#运维监控)
- [代码维护指南](#代码维护指南)
- [技术栈](#技术栈)

---

## 服务架构

单主服务 + 可选 worker 的架构，全部围绕 **9000 端口的 main:app**，
所有服务只监听 `127.0.0.1`（没有登录鉴权，不要对外暴露端口）：

| 组件 | 端口/形态 | 启动方式 | 说明 |
|------|----------|---------|------|
| 主服务 | 9000 (HTTP) | `./start_9000.sh` 或 `./start_local.sh` | FastAPI：回测、优化、扫描、运维接口与前端 |
| 回测 Worker | 后台进程 | `./start_backtest_worker.sh` | 消费回测队列 |

回测 Worker 通过主服务的 `/internal/ohlcv` 拉取 K 线数据。只有 9000 一个端口。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/keepwin-Neil/Aitrader.git
cd Aitrader
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# requirements.txt 未覆盖的运行时依赖：
pip install psycopg2-binary python-dotenv
```

### 3. 准备 PostgreSQL

系统需要一个本机 PostgreSQL 数据库（策略组合、LLM 评审、优化记录三张表在首次启动时自动创建）：

```bash
createdb aitrader
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env：
#   DATABASE_URL=postgresql://localhost/aitrader
#   AITRADER_SMTP_*       —— 健康告警邮件（可选）
#   FINNHUB_API_KEY    —— 新闻数据源（可选）
```

### 5. 启动

```bash
# 开发（自动 reload，127.0.0.1:9000）
./start_local.sh

# 不带 reload
./start_9000.sh

# Windows
start.bat
```

浏览器打开 **http://localhost:9000** 即为前端 Dashboard。

### 6. 可选组件

```bash
./start_backtest_worker.sh  # 回测 worker
python3 -u auto_optimize.py --trials 150 --max-rounds 5   # 自动优化流水线
```

### IB Gateway 行情（可选）

本机装有 Interactive Brokers Gateway 时，`US / HK` 行情优先走 IB，
失败自动回退 Yahoo / AKShare。默认连接参数：

```bash
IB_GATEWAY_ENABLED=1
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=4002
IB_GATEWAY_FALLBACK_HOST=127.0.0.1
IB_GATEWAY_FALLBACK_PORT=4001
```

状态查看：`GET /market-data/ib/status`。

---

## 项目结构

```
Aitrader/
├── main.py                  # FastAPI 主服务：路由挂载、启动逻辑、健康检查
├── backtest.py              # 回测引擎：信号、仓位、绩效统计
├── strategies.py            # 策略定义与注册表（全部策略）
├── optimizer.py             # 参数优化器：贝叶斯/穷举/粗搜精搜
├── auto_optimize.py         # 多轮自动优化流水线（并行、断点续跑）
│
├── routers/                 # API 路由模块
│   ├── market.py            #   行情、K线、搜索
│   ├── data.py              #   日线同步、数据管理
│   ├── strategies.py        #   策略列表、仓库管理
│   ├── trading.py           #   回测、对比、组合
│   ├── scan.py              #   实时扫描
│   ├── scan_monitor.py      #   盘中监控
│   ├── pools.py             #   股票池 CRUD
│   ├── quant.py             #   量化分析
│   ├── eval_chain.py        #   策略评估链
│   ├── stock_report.py      #   个股报告
│   ├── llm_admin.py         #   LLM 管理：provider 密钥 / CLI 模式 / 模型名
│   ├── combos.py            #   保存的策略组合
│   └── opslog.py            #   扫描审计 / 基线 / 异常日志查询
│
├── core_engine.py           # LLM 调用基础设施（CLI 优先 / API 兜底）+ 扫描守护进程
├── core_store.py            # PostgreSQL 连接池与数据层（组合、LLM 评审、优化记录）
├── scan_center.py           # 统一扫描（Stage 0 股票池 → Stage 1 原始信号）
├── backtest_queue.py        # 后台回测队列
├── backtest_worker.py       # 后台回测 worker
├── ib_gateway.py            # IB Gateway 连接池
├── news_finnhub.py          # Finnhub 新闻抓取
│
├── admin_auth.py            # 可选的管理 token 门禁（AITRADER_ADMIN_TOKEN）
├── health_watchdog.py  (scripts/)  # 健康监控 + 邮件告警
│
├── static/                  # 前端（index.html 单文件 SPA、ops.html 运维面板等）
├── stock_pools.json         # 股票池定义
├── strategy_warehouse.json  # 策略仓库状态（活跃/归档/收藏）
├── universal_params.json    # 通用参数基准
├── custom_params/           # 个股标准参数覆盖（每股票一个 JSON）
├── optimized_params/        # 个股优化参数（每股票一个 JSON）
├── news_pool_daily.json     # 新闻选股池（日/周）
└── migrations/              # SQL 迁移记录（表结构以 core_store.init_db 为准）
```

---

## 参数分层与存储

页面里的 `通用参数 / 标准参数 / 优化参数` 是三层不同口径：

| 层 | 存储 | 读取逻辑 | 用途 |
|----|------|---------|------|
| **通用参数** | `universal_params.json`，缺省回退 `strategies.py::default_params` | `_get_universal_strategy_params()` | 通用模式、通用 Top、个股标准参数基底 |
| **标准参数** | 通用参数 + `custom_params/<SYMBOL>.json` 个股覆盖 | `_get_standard_strategy_params()` | 标准模式、盘中监测、模拟交易监控 |
| **优化参数** | `optimized_params/<SYMBOL>.json`（含 `strategies` / `method_best` / `wf_validation`） | `_load_optimized()` | 优化模式、标准 vs 优化对比、升级候选 |

标准模式下的取值顺序：`default_params` → `universal_params.json` 覆盖 →
`custom_params/<SYMBOL>.json` 继续覆盖。

---

## 核心模块

### backtest.py — 回测引擎

```python
df = get_ohlcv("AAPL", "US", interval="1d", period="2y")
result = run_strategy_backtest(df, "mean_reversion",
                                params_override={"rsi_period": 12},
                                dynamic_position=True)
# result: summary / trades / equity_curve / chart_data
```

- ATR 动态仓位与止损，支持做空
- 分钟/小时/日线自动检测
- 综合评分：`0.40×收益 + 0.25×Sharpe + 0.20×(1-回撤) + 0.10×胜率 + 0.05×盈亏比`

### strategies.py — 策略定义

所有策略继承 `BaseStrategy`，实现 `add_indicators()` 与 `generate_signals()`。
当前仓库状态（`strategy_warehouse.json`）：**42 个活跃策略、102 个归档、10 个收藏**，
覆盖趋势、均值回归、动量、量价、背离、形态等类型。完整清单以
`GET /strategies` 与 `strategy_warehouse.json` 为准。

### optimizer.py — 参数优化

| 方法 | 说明 |
|------|------|
| `bayesian` | Optuna TPE，默认 200 trials，最高效 |
| `grid` | 全排列穷举，最全面但耗时 |
| `coarse_fine` | 粗搜 + 精搜两阶段 |

全部使用 `ProcessPoolExecutor` 多进程并行。Walk-Forward 验证标准：
一致性 ≥ 50%、过拟合比 ≥ 0.3、全池评分提升 > 0.005 才升级参数。

### auto_optimize.py — 自动优化流水线

多轮收敛：基准测试 → 每股 × 多策略并行优化 → Walk-Forward 验证 →
增量全池评分 → 达标升级；连续收敛即停止。支持断点续跑
（`auto_optimize_results/_checkpoint_v2.json`）。

---

## LLM 调用

LLM 用于策略评估链、个股研报、新闻打分和量化交易否决层（veto prompt 在「LLM Prompt」页维护）。

- provider 配置在「🤖 LLM 管理」页维护，存于本机 `.cache/llm_keys.json`（不进仓库）
- 调用规则：同一 provider 的 CLI 串行；CLI 优先，超时回退 API

---

## API 概览

### 行情与数据

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/quote/{market}/{symbol}` | 实时报价 |
| GET | `/kline/{market}/{symbol}` | K线数据 |
| POST | `/quotes/batch` | 批量报价 |
| GET | `/market-data/providers` / `/market-data/ib/status` | 行情源与 IB 状态 |
| POST | `/data/daily-sync/start` | 日线数据补齐任务 |
| GET | `/search?q=` | 股票搜索 |

### 回测与优化

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/backtest/{market}/{symbol}` | 单策略回测 |
| POST | `/compare/{market}/{symbol}` | 全策略排名 |
| POST | `/portfolio/backtest` | 组合回测 |
| POST | `/walkforward/{market}/{symbol}` | Walk-Forward 验证 |
| POST | `/optimize/{market}/{symbol}` | 参数优化（三方法） |
| POST | `/auto-optimize/start` | 自动优化流水线 |

### 策略与股票池

`/strategies`（列表/仓库/收藏/归档）、`/pools`（CRUD）、`/scan/realtime`
（实时 BUY/SELL 扫描）、`/selector/realtime`（选股）。

### LLM 与运维

`/llm/keys`（LLM provider 密钥）、`/strategy-combos`（策略组合）、
`/scan/monitor/*`（扫描监测）、
`/ops/health`（聚合健康检查）、`/internal/ohlcv`（回测 worker 专用数据接口）。

---

## 运维监控

- **`GET /ops/health`**：聚合健康检查（IB 行情池、回测队列、
  文件描述符压力、扫描审计落盘）
- **`static/ops.html`**：运维面板（http://localhost:9000/static/ops.html）
- **`scripts/health_watchdog.py`**：定时探测健康接口，状态翻转时邮件告警
  （launchd 定时，SMTP 配置见 `.env`）
- **`routers/opslog.py`**：查询扫描审计 JSONL（`.cache/scan_audit/`）、基线快照与未处理异常日志

---

## 代码维护指南

### 添加新策略

1. `strategies.py`：继承 `BaseStrategy` 实现两个抽象方法，设置元信息
2. `strategies.py` 末尾 `_register(...)` 注册
3. `optimizer.py::PARAM_RANGES` 添加参数范围
4. （可选）`main.py::OPTIMIZE_GRIDS` 添加穷举网格
5. 验证：`python3 -c "from strategies import get_strategy; print(get_strategy('name').description)"`

### 修改评分公式

`backtest.py::composite_score()` 与 `auto_optimize.py::compute_composite_score()` 需同步修改。

### 常见问题排查

| 问题 | 排查方向 |
|------|---------|
| 策略 dropna 清空数据 | `shift()` 产生 NaN，改用 `fillna(0)` + `dropna(subset=core_cols)` |
| ProcessPool pickle 错误 | worker 函数必须是模块级（非闭包） |
| 优化无提升 | 参数已收敛，或 Walk-Forward 拒绝了过拟合参数 |

### Git 工作流

```bash
git add <files>
git commit -m "feat/fix/perf: 描述"
git push origin main
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python 3.9+, FastAPI, uvicorn |
| 数据 | PostgreSQL (psycopg2), yfinance, akshare, IB Gateway (ib_insync) |
| 优化 | optuna (贝叶斯), ProcessPoolExecutor (多进程并行) |
| LLM | Claude / OpenAI / Gemini 等（CLI 优先，API 兜底） |
| 前端 | TradingView Lightweight Charts, Chart.js（单文件 SPA），英/中双语（默认英文，右上角 中/EN 按钮切换，`static/i18n.js` 词典 6500 条） |
| 运行 | 本机 uvicorn（127.0.0.1）/ launchd watchdog（可选） |
