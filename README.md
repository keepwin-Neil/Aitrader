# AITrader — AI Quantitative Trading & Strategy Optimization Platform

English | [中文](README.zh-CN.md)

A quantitative trading R&D system built on **FastAPI + PostgreSQL + LLM** that **runs on the local
machine only**: multi-strategy backtesting, parameter optimization, Walk-Forward validation,
real-time scanning, news-driven stock picking, paper trading, LLM strategy evaluation, and IB
Gateway market data integration.

> 📦 Repository: https://github.com/keepwin-Neil/Aitrader (public)

---

## Table of Contents

- [Service Architecture](#service-architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Parameter Layering & Storage](#parameter-layering--storage)
- [Core Modules](#core-modules)
- [LLM Calls](#llm-calls)
- [API Overview](#api-overview)
- [Ops & Monitoring](#ops--monitoring)
- [Maintenance Guide](#maintenance-guide)
- [Tech Stack](#tech-stack)

---

## Service Architecture

A single main service with optional workers, all centered on **main:app at port 9000**.
Every service listens on `127.0.0.1` only (there is no login — do not expose the ports):

| Component | Port / Form | Launch | Description |
|-----------|------------|--------|-------------|
| Main service | 9000 (HTTP) | `./start_9000.sh` or `./start_local.sh` | FastAPI: backtesting, optimization, scanning, ops endpoints and frontend |
| Backtest Worker | background process | `./start_backtest_worker.sh` | Consumes the backtest queue |

Backtest workers pull OHLCV data from the main service's `/internal/ohlcv`
endpoint. Port 9000 is the only port.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/keepwin-Neil/Aitrader.git
cd Aitrader
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Runtime dependencies not covered by requirements.txt:
pip install psycopg2-binary python-dotenv
```

### 3. Prepare PostgreSQL

The system requires a local PostgreSQL database (the strategy combo, LLM review and
optimization run tables are created automatically on first startup):

```bash
createdb aitrader
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env:
#   DATABASE_URL=postgresql://localhost/aitrader
#   AITRADER_SMTP_*       — health alert email (optional)
#   FINNHUB_API_KEY    — news data source (optional)
```

### 5. Start

```bash
# Development (auto-reload, 127.0.0.1:9000)
./start_local.sh

# Without reload
./start_9000.sh

# Windows
start.bat
```

Open **http://localhost:9000** in your browser for the dashboard.

### 6. Optional components

```bash
./start_backtest_worker.sh  # backtest worker
python3 -u auto_optimize.py --trials 150 --max-rounds 5   # auto-optimization pipeline
```

### IB Gateway market data (optional)

With Interactive Brokers Gateway installed locally, `US / HK` market
data goes through IB first and automatically falls back to Yahoo / AKShare on
failure. Default connection parameters:

```bash
IB_GATEWAY_ENABLED=1
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=4002
IB_GATEWAY_FALLBACK_HOST=127.0.0.1
IB_GATEWAY_FALLBACK_PORT=4001
```

Status: `GET /market-data/ib/status`.

---

## Project Structure

```
Aitrader/
├── main.py                  # FastAPI main service: router mounting, startup, health checks
├── backtest.py              # Backtest engine: signals, position sizing, performance stats
├── strategies.py            # Strategy definitions and registry (all strategies)
├── optimizer.py             # Parameter optimizer: Bayesian / grid / coarse-fine
├── auto_optimize.py         # Multi-round auto-optimization pipeline (parallel, resumable)
│
├── routers/                 # API router modules
│   ├── market.py            #   quotes, K-lines, search
│   ├── data.py              #   daily sync, data management
│   ├── strategies.py        #   strategy listing, warehouse management
│   ├── trading.py           #   backtest, comparison, portfolio
│   ├── scan.py              #   real-time scanning
│   ├── scan_monitor.py      #   intraday monitoring
│   ├── pools.py             #   stock pool CRUD
│   ├── quant.py             #   quantitative analysis
│   ├── eval_chain.py        #   strategy evaluation chain
│   ├── stock_report.py      #   per-stock reports
│   ├── llm_admin.py         #   LLM management: provider keys / CLI mode / model names
│   ├── combos.py            #   saved strategy combos
│   └── opslog.py            #   scan-audit / baseline / error-log queries
│
├── core_engine.py           # LLM call infrastructure (CLI first / API fallback) + scan daemon
├── core_store.py            # PostgreSQL pool & data layer (combos, LLM reviews, optimization runs)
├── scan_center.py           # unified scan (Stage 0 stock pools → Stage 1 raw signals)
├── backtest_queue.py        # background backtest queue
├── backtest_worker.py       # background backtest worker
├── ib_gateway.py            # IB Gateway connection pool
├── news_finnhub.py          # Finnhub news fetching
│
├── admin_auth.py            # optional admin token gate (AITRADER_ADMIN_TOKEN)
├── health_watchdog.py  (scripts/)  # health monitoring + email alerts
│
├── static/                  # frontend (index.html single-file SPA, ops.html dashboard, etc.)
├── stock_pools.json         # stock pool definitions
├── strategy_warehouse.json  # strategy warehouse state (active/archived/favorites)
├── universal_params.json    # universal parameter baseline
├── custom_params/           # per-symbol standard parameter overrides (one JSON per symbol)
├── optimized_params/        # per-symbol optimized parameters (one JSON per symbol)
├── news_pool_daily.json     # news-driven stock pools (daily/weekly)
└── migrations/              # SQL migration records (core_store.init_db is the source of truth)
```

---

## Parameter Layering & Storage

The `Universal / Standard / Optimized` parameters shown in the UI are three distinct layers:

| Layer | Storage | Read logic | Usage |
|-------|---------|-----------|-------|
| **Universal** | `universal_params.json`, falls back to `strategies.py::default_params` | `_get_universal_strategy_params()` | Universal mode, universal Top ranking, base for per-symbol standard params |
| **Standard** | Universal params + per-symbol overrides in `custom_params/<SYMBOL>.json` | `_get_standard_strategy_params()` | Standard mode, intraday monitoring, paper-trade monitoring |
| **Optimized** | `optimized_params/<SYMBOL>.json` (contains `strategies` / `method_best` / `wf_validation`) | `_load_optimized()` | Optimized mode, standard-vs-optimized comparison, upgrade candidates |

Resolution order in standard mode: `default_params` → overridden by
`universal_params.json` → further overridden by `custom_params/<SYMBOL>.json`.

---

## Core Modules

### backtest.py — Backtest engine

```python
df = get_ohlcv("AAPL", "US", interval="1d", period="2y")
result = run_strategy_backtest(df, "mean_reversion",
                                params_override={"rsi_period": 12},
                                dynamic_position=True)
# result: summary / trades / equity_curve / chart_data
```

- ATR-based dynamic position sizing and stops, short-selling support
- Automatic minute/hour/daily detection
- Composite score: `0.40×return + 0.25×Sharpe + 0.20×(1-drawdown) + 0.10×win-rate + 0.05×profit-factor`

### strategies.py — Strategy definitions

All strategies inherit from `BaseStrategy`, implementing `add_indicators()` and
`generate_signals()`. Current warehouse state (`strategy_warehouse.json`):
**42 active strategies, 102 archived, 10 favorites**, covering trend, mean
reversion, momentum, volume-price, divergence, chart patterns and more. For the
full list see `GET /strategies` and `strategy_warehouse.json`.

### optimizer.py — Parameter optimization

| Method | Description |
|--------|-------------|
| `bayesian` | Optuna TPE, 200 trials by default, most efficient |
| `grid` | Full exhaustive search, most thorough but slow |
| `coarse_fine` | Two-phase coarse-then-fine grid |

All methods run in parallel via `ProcessPoolExecutor`. Walk-Forward validation
criteria: consistency ≥ 50%, overfit ratio ≥ 0.3, whole-pool score improvement
> 0.005 before a parameter upgrade.

### auto_optimize.py — Auto-optimization pipeline

Multi-round convergence: baseline benchmarking → per-symbol multi-strategy
parallel optimization → Walk-Forward validation → incremental whole-pool scoring
→ upgrade when thresholds are met; stops after consecutive converged rounds.
Supports checkpoint resume (`auto_optimize_results/_checkpoint_v2.json`).

---

## LLM Calls

LLMs are used by the strategy evaluation chain, stock reports, news scoring, and the quant-trading
veto layer (its prompt is edited on the "LLM Prompt" page).

- Provider config is managed on the "🤖 LLM Management" page and stored locally in `.cache/llm_keys.json` (never committed)
- Call rules: CLI calls to the same provider are serialized; CLI first, falling back to the API on timeout

---

## API Overview

### Market data

| Method | Path | Description |
|--------|------|-------------|
| GET | `/quote/{market}/{symbol}` | Real-time quote |
| GET | `/kline/{market}/{symbol}` | K-line data |
| POST | `/quotes/batch` | Batch quotes |
| GET | `/market-data/providers` / `/market-data/ib/status` | Data source & IB status |
| POST | `/data/daily-sync/start` | Daily data backfill task |
| GET | `/search?q=` | Symbol search |

### Backtesting & optimization

| Method | Path | Description |
|--------|------|-------------|
| POST | `/backtest/{market}/{symbol}` | Single-strategy backtest |
| POST | `/compare/{market}/{symbol}` | All-strategy ranking |
| POST | `/portfolio/backtest` | Portfolio backtest |
| POST | `/walkforward/{market}/{symbol}` | Walk-Forward validation |
| POST | `/optimize/{market}/{symbol}` | Parameter optimization (3 methods) |
| POST | `/auto-optimize/start` | Auto-optimization pipeline |

### Strategies & stock pools

`/strategies` (listing / warehouse / favorites / archive), `/pools` (CRUD),
`/scan/realtime` (real-time BUY/SELL scanning), `/selector/realtime` (stock selection).

### LLM & ops

`/llm/keys` (LLM provider keys), `/strategy-combos` (saved combos),
`/scan/monitor/*` (scan monitoring),
`/ops/health` (aggregate health check), `/internal/ohlcv` (data endpoint reserved for backtest workers).

---

## Ops & Monitoring

- **`GET /ops/health`**: aggregate health check (IB quote pool, backtest queue,
  file-descriptor pressure, scan-audit flushing)
- **`static/ops.html`**: ops dashboard (http://localhost:9000/static/ops.html)
- **`scripts/health_watchdog.py`**: periodically probes the health endpoint and
  emails on state transitions (launchd timer, SMTP config in `.env`)
- **`routers/opslog.py`**: query scan-audit JSONL (`.cache/scan_audit/`), baseline snapshots and the unhandled-error log

---

## Maintenance Guide

### Adding a new strategy

1. `strategies.py`: inherit `BaseStrategy`, implement the two abstract methods, set metadata
2. Register at the end of `strategies.py` via `_register(...)`
3. Add parameter ranges in `optimizer.py::PARAM_RANGES`
4. (Optional) add a grid in `main.py::OPTIMIZE_GRIDS`
5. Verify: `python3 -c "from strategies import get_strategy; print(get_strategy('name').description)"`

### Changing the scoring formula

`backtest.py::composite_score()` and `auto_optimize.py::compute_composite_score()`
must be updated together.

### Troubleshooting

| Problem | What to check |
|---------|--------------|
| Strategy dropna empties the data | NaN from `shift()`; use `fillna(0)` + `dropna(subset=core_cols)` |
| ProcessPool pickle errors | Worker functions must be module-level (not closures) |
| Optimization shows no improvement | Parameters already converged, or Walk-Forward rejected overfit params |

### Git workflow

```bash
git add <files>
git commit -m "feat/fix/perf: description"
git push origin main
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, FastAPI, uvicorn |
| Data | PostgreSQL (psycopg2), yfinance, akshare, IB Gateway (ib_insync) |
| Optimization | optuna (Bayesian), ProcessPoolExecutor (multiprocess parallelism) |
| LLM | Claude / OpenAI / Gemini etc. (CLI first, API fallback) |
| Frontend | TradingView Lightweight Charts, Chart.js (single-file SPA), English/Chinese UI — English by default, 中/EN toggle at the top right (6,500-entry dictionary in `static/i18n.js`) |
| Runtime | local uvicorn (127.0.0.1) / launchd watchdog (optional) |
