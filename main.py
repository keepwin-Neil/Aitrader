
# Force multiprocessing to use 'spawn' (not the Linux default 'fork') to avoid
# ProcessPoolExecutor deadlocks: forked children inherit parent locks held by
# threads they don't have, leading to hangs in logging/PG/numpy/optuna code
# paths. Must run BEFORE fastapi/uvicorn imports, which trigger those locks.
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# 启动时加载项目根目录 .env，让各模块在导入时就能读到本地配置。
# override=False（已在 shell export 的 env 变量优先）。
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).parent / ".env", override=False)
except Exception:
    pass  # python-dotenv not installed → 静默退化，行为同改前

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from typing import Optional, Union, List
from pathlib import Path
import json
import yfinance as yf
import akshare as ak
import time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from itertools import product as iter_product
from backtest import (
    get_ohlcv, add_indicators, generate_signals, run_backtest, run_strategy_backtest,
    DEFAULT_PARAMS, bulk_update, _get_data_path, _filter_period,
    init_stock_data, update_stock_data, _MEM_CACHE
)
from strategies import list_strategies, get_strategy, STRATEGY_REGISTRY
import logging, threading, os, hashlib, math, contextlib, tempfile
# LockedDict 抽在独立模块 locked_dict.py (只依赖 threading), worker 脚本和单测
# 都能直接 import, 不必为一个容器类把整个 FastAPI app 拖起来。
# 这里 re-export, 既有的 _m.LockedDict / _m.snapshot_of 调用点继续可用。
from locked_dict import LockedDict, snapshot_of, _detach as _ld_detach
import re
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    TimeoutError as FutureTimeoutError,
)
import proc_role  # ProcessPoolExecutor(initializer=proc_role.mark_pool_worker) 给 worker 打标
from ib_gateway import ib_status as _ib_status, fetch_quote as ib_fetch_quote, ib_enabled as _ib_enabled
from pydantic import BaseModel
from core_store import init_db
from core_engine import start_scan_daemon

logger = logging.getLogger("uvicorn.error")

# ── CPU 自适应并行 ──
_CPU_COUNT = os.cpu_count() or 4
N_WORKERS = max(2, _CPU_COUNT - 2)  # 保留2核给系统
# Realtime scan workers are capped much lower than N_WORKERS because they each
# hit external rate-limited services (yfinance ~60req/min, IB Gateway hard
# 32-connection cap). With N_WORKERS=30 × 4 markets the scanner used to peg IB
# Gateway and starve foreground /quotes calls. 8 per market keeps headroom.
SCAN_WORKERS = int(os.getenv("AITRADER_SCAN_WORKERS", "8"))
logger.info(f"CPU cores={_CPU_COUNT}, parallel workers={N_WORKERS}, scan workers={SCAN_WORKERS}")

# ── 股票池持久化 ──
POOLS_FILE = Path(__file__).parent / "stock_pools.json"
OPTIMIZED_DIR = Path(__file__).parent / "optimized_params"
OPTIMIZED_DIR.mkdir(exist_ok=True)
CUSTOM_PARAMS_DIR = Path(__file__).parent / "custom_params"
CUSTOM_PARAMS_DIR.mkdir(exist_ok=True)
OPT_HISTORY_DIR = Path(__file__).parent / "optimization_history"
OPT_HISTORY_DIR.mkdir(exist_ok=True)
UNIVERSAL_PARAMS_FILE = Path(__file__).parent / "universal_params.json"
HOT_CACHE_FILE = Path(__file__).parent / "hot_stocks_cache.json"
HOT_POOL_NAME = "🔥 US Hot Stocks"
UNIVERSAL_BENCHMARK_POOL = "NASDAQ 100"
PARAM_INTERVALS = ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
MARKET_DATA_CONFIG_FILE = Path(__file__).parent / "market_data_config.json"
CORE_MARKETS = ["US", "CN", "HK", "UK", "CRYPTO"]
MARKET_UNIVERSE_DIR = Path(__file__).parent / "market_universes"
MARKET_UNIVERSE_DIR.mkdir(exist_ok=True)
MARKET_UNIVERSE_EXCLUDES_FILE = Path(__file__).parent / "market_universe_excludes.json"
FUNDAMENTAL_DATA_DIR = Path(__file__).parent / "fundamental_data"
FUNDAMENTAL_DATA_DIR.mkdir(exist_ok=True)
MARKET_SYNC_SYMBOL_TIMEOUT_SEC = int(os.getenv("MARKET_SYNC_SYMBOL_TIMEOUT_SEC", "20"))
STRATEGY_LLM_REGISTRY_FILE = Path(__file__).parent / "strategy_llm_registry.json"
STRATEGY_VALIDATION_FILE = Path(__file__).parent / "strategy_validation_registry.json"
STRATEGY_REPORTS_DIR = Path(__file__).parent / "strategy_reports"
STRATEGY_REPORTS_DIR.mkdir(exist_ok=True)

# 2026-04-25 — Per-market scan-monitor settings.
# Each CORE market (US/HK/CN/CRYPTO/UK) is an independent monitoring
# console with its own scan parameters. Stored on disk so settings
# survive restarts and sync across browsers (vs. localStorage-only).
PER_MARKET_SETTINGS_FILE = Path(__file__).parent / "per_market_scan_settings.json"
_PER_MARKET_SETTINGS_LOCK = threading.RLock()

# Defaults applied when the file is missing OR a market entry is absent.
# pools: 这个市场扫哪些股票池 (空 = 不扫)。
PER_MARKET_DEFAULT = {
    "enabled": True,
    "interval": "1d",
    "refresh_sec": 300,
    "top_n": 5,
    "owners": ["claude", "gpt", "gemini"],
    "pools": [],
}

# Field set persisted/exposed for each market. Anything outside this list is
# dropped to keep the schema tight.
_PER_MARKET_FIELDS = ("enabled", "interval", "refresh_sec", "top_n", "owners", "pools")


def _normalize_per_market_entry(merged: dict) -> dict:
    merged = {k: merged[k] for k in _PER_MARKET_FIELDS if k in merged}
    merged["enabled"] = bool(merged.get("enabled", True))
    try:
        merged["refresh_sec"] = max(30, int(merged.get("refresh_sec") or 300))
    except (TypeError, ValueError):
        merged["refresh_sec"] = 300
    try:
        merged["top_n"] = max(0, int(merged.get("top_n") or 0))
    except (TypeError, ValueError):
        merged["top_n"] = 5
    for key, lower in (("owners", True), ("pools", False)):
        vals = merged.get(key) or []
        if not isinstance(vals, list):
            vals = []
        vals = [str(v).strip() for v in vals if str(v).strip()]
        merged[key] = [v.lower() for v in vals] if lower else vals
    merged["interval"] = str(merged.get("interval") or "1d").strip() or "1d"
    return merged


def load_per_market_settings() -> dict:
    """Read per-market settings from disk. Returns the full {market: {...}} dict.

    Missing file / missing market → returns defaults so callers always get a
    complete view across all CORE_MARKETS. Never raises.
    """
    out: dict = {}
    raw: dict = {}
    if PER_MARKET_SETTINGS_FILE.exists():
        try:
            with _PER_MARKET_SETTINGS_LOCK:
                with open(PER_MARKET_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "load_per_market_settings: parse failed (%s) — falling back to defaults", exc,
            )
            raw = {}
    markets_raw = raw.get("markets") if isinstance(raw, dict) else {}
    if not isinstance(markets_raw, dict):
        markets_raw = {}
    for m in CORE_MARKETS:
        entry = markets_raw.get(m) or markets_raw.get(m.upper()) or {}
        merged = {**PER_MARKET_DEFAULT}
        if isinstance(entry, dict):
            for k in _PER_MARKET_FIELDS:
                if k in entry:
                    merged[k] = entry[k]
        out[m] = _normalize_per_market_entry(merged)
    return out


def save_per_market_settings(market: str, patch: dict) -> dict:
    """Persist a single market's settings, merging with existing values.

    Returns the full updated settings dict (all markets, post-update). The
    saved-on-disk JSON has shape::

        {"markets": {"<MARKET>": {<settings>}, ...}}

    Raises ValueError on bad input (unknown market, malformed types).
    """
    market = str(market or "").upper().strip()
    if market not in CORE_MARKETS:
        raise ValueError(f"unknown market: {market!r} (expected one of {CORE_MARKETS})")
    if not isinstance(patch, dict):
        raise ValueError("patch must be a dict")
    with _PER_MARKET_SETTINGS_LOCK:
        # Read current file or {} if missing.
        raw: dict = {"markets": {}}
        if PER_MARKET_SETTINGS_FILE.exists():
            try:
                with open(PER_MARKET_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {"markets": {}}
            except Exception:  # noqa: BLE001
                raw = {"markets": {}}
        if not isinstance(raw.get("markets"), dict):
            raw["markets"] = {}
        # Merge: existing market settings win over defaults; patch wins over both.
        existing = raw["markets"].get(market) or {}
        merged = {**PER_MARKET_DEFAULT, **existing}
        for k in _PER_MARKET_FIELDS:
            if k in patch:
                merged[k] = patch[k]
        raw["markets"][market] = _normalize_per_market_entry(merged)
        # Atomic write: tmp → fsync → rename.
        tmp = PER_MARKET_SETTINGS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, PER_MARKET_SETTINGS_FILE)
    # Return the full view so callers can echo it back to the UI.
    return load_per_market_settings()


SUPPORTED_STRATEGY_LLM_OWNERS = [
    "Claude", "OpenAI", "Grok", "Gemini", "DeepSeek",
    "Llama", "Mistral", "Qwen", "Cohere",
    "Yi", "Doubao", "Zhipu", "MiniMax", "Moonshot", "StepFun",
]
DEFAULT_STRATEGY_LLM_OWNER = "Claude"
NEW_STRATEGY_MPOOL_THRESHOLD = 20.0
NEW_STRATEGY_READY_THRESHOLD = 50.0

# ═══════════════════════════════════════════════════════════════════
#  个股参数优化 — 精简搜索网格（每策略 ~50-300 组合）
# ═══════════════════════════════════════════════════════════════════
OPTIMIZE_GRIDS = {
    # ── 扩大搜索范围，每策略目标 500~2000 组合 ──
    "claude_multi_indicator": {
        "ema_fast": list(range(5, 21, 3)),           # [5,8,11,14,17,20] 6个
        "ema_slow": list(range(30, 101, 15)),         # [30,45,60,75,90] 5个
        "rsi_buy_min": list(range(35, 56, 5)),        # [35,40,45,50,55] 5个
        "rsi_sell": list(range(70, 86, 5)),           # [70,75,80,85] 4个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 6×5×5×4×5 = 3000
    },
    "claude_dual_ma_cross": {
        "fast_period": list(range(3, 21, 2)),         # [3,5,7,9,11,13,15,17,19] 9个
        "slow_period": list(range(15, 61, 5)),        # [15,20,25,30,35,40,45,50,55,60] 10个
        "ma_type": ["sma", "ema"],                    # 2个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 9×10×2×5 = 900
    },
    "claude_momentum_breakout": {
        "breakout_period": list(range(8, 41, 4)),     # [8,12,16,20,24,28,32,36,40] 9个
        "exit_period": list(range(3, 16, 2)),         # [3,5,7,9,11,13,15] 7个
        "vol_ratio_min": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5],  # 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 9×7×6×5 = 1890
    },
    "claude_mean_reversion": {
        "rsi_period": list(range(5, 21, 3)),          # [5,8,11,14,17,20] 6个
        "rsi_oversold": list(range(20, 41, 5)),       # [20,25,30,35,40] 5个
        "rsi_overbought": list(range(60, 81, 5)),     # [60,65,70,75,80] 5个
        "bb_std": [1.0, 1.5, 2.0, 2.5, 3.0],        # 5个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 6×5×5×5×5 = 3750
    },
    "claude_turtle_trading": {
        "entry_period": list(range(8, 41, 4)),        # [8,12,16,20,24,28,32,36,40] 9个
        "exit_period": list(range(3, 16, 2)),         # [3,5,7,9,11,13,15] 7个
        "atr_stop_mul": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # 6个  → 9×7×6 = 378
    },
    "claude_macd_divergence": {
        "macd_fast": list(range(6, 19, 2)),           # [6,8,10,12,14,16,18] 7个
        "macd_slow": list(range(18, 41, 4)),          # [18,22,26,30,34,38] 6个
        "lookback": list(range(15, 51, 7)),           # [15,22,29,36,43,50] 6个
        "rsi_sell": list(range(70, 91, 5)),           # [70,75,80,85,90] 5个
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0],       # 4个  → 7×6×6×5×4 = 5040
    },
    "claude_kdj_dmi": {
        "kdj_period": list(range(5, 26, 4)),          # [5,9,13,17,21,25] 6个
        "kdj_oversold": list(range(10, 31, 5)),       # [10,15,20,25,30] 5个
        "kdj_overbought": list(range(70, 91, 5)),     # [70,75,80,85,90] 5个
        "adx_threshold": list(range(10, 31, 5)),      # [10,15,20,25,30] 5个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 6×5×5×5×5 = 3750
    },
    "claude_vwap_breakout": {
        "vwap_period": list(range(5, 36, 5)),         # [5,10,15,20,25,30,35] 7个
        "atr_band_mul": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5],  # 7个
        "vol_ratio_min": [0.5, 0.8, 1.0, 1.2, 1.5],  # 5个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 7×7×5×5 = 1225
    },
    "claude_adaptive_ma": {
        "er_period": list(range(5, 31, 5)),           # [5,10,15,20,25,30] 6个
        "fast_sc": [2, 3, 4, 5, 7],                  # 5个
        "slow_sc": [15, 20, 25, 30, 40, 50],         # 6个
        "threshold": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008],  # 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 6×5×6×6×5 = 5400
    },
    "claude_rsi_macd_trend": {
        "rsi_entry": list(range(20, 46, 5)),          # [20,25,30,35,40,45] 6个
        "rsi_exit": list(range(65, 91, 5)),           # [65,70,75,80,85,90] 6个
        "ema_trend": list(range(20, 71, 10)),         # [20,30,40,50,60,70] 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 6×6×6×5 = 1080
    },
    "claude_ensemble_voting": {
        "buy_threshold": [1, 2, 3],                   # 3个
        "sell_threshold": [1, 2, 3],                  # 3个
        "mr_rsi_oversold": list(range(20, 41, 5)),    # [20,25,30,35,40] 5个
        "mr_rsi_overbought": list(range(60, 81, 5)),  # [60,65,70,75,80] 5个
        "dm_fast": list(range(5, 16, 3)),             # [5,8,11,14] 4个
        "dm_slow": list(range(15, 41, 5)),            # [15,20,25,30,35,40] 6个  → 3×3×5×5×4×6 = 5400
    },
    "claude_ichimoku_cloud": {
        "tenkan_period": list(range(5, 16, 2)),       # [5,7,9,11,13,15] 6个
        "kijun_period": list(range(18, 35, 3)),       # [18,21,24,27,30,33] 6个
        "senkou_b_period": list(range(40, 65, 4)),    # [40,44,48,52,56,60,64] 7个
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0, 3.5],  # 5个  → 6×6×7×5 = 1260
    },
    "claude_volatility_breakout": {
        "k_factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 8个
        "ema_period": list(range(20, 81, 10)),        # [20,30,40,50,60,70,80] 7个
        "min_range": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],  # 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 8×7×6×5 = 1680
    },
    "claude_elder_triple_screen": {
        "weekly_ema_period": list(range(8, 21, 2)),   # [8,10,12,14,16,18,20] 7个
        "force_period": [2, 3, 5],                    # 3个
        "ema_short": list(range(8, 21, 2)),           # [8,10,12,14,16,18,20] 7个
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0],       # 4个  → 7×3×7×4 = 588
    },
    "claude_supertrend": {
        "st_period": list(range(7, 26, 3)),           # [7,10,13,16,19,22,25] 7个
        "st_multiplier": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 7×6×5 = 210
    },
    "claude_ross_hook": {
        "fractal_n": [2, 3, 4, 5],                   # 4个
        "ema_period": list(range(10, 61, 10)),        # [10,20,30,40,50,60] 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],  # 6个  → 4×6×6 = 144
    },
    "claude_divergence_combo": {
        "pivot_n": [2, 3, 5],                        # 3个
        "max_pivot_dist": list(range(30, 91, 15)),    # [30,45,60,75,90] 5个
        "div_confirm": [1, 2],                        # 2个
        "rsi_zone_low": list(range(40, 66, 5)),       # [40,45,50,55,60,65] 6个
        "rsi_zone_high": list(range(35, 61, 5)),      # [35,40,45,50,55,60] 6个
        "atr_stop_mul": [1.0, 1.5, 2.0, 2.5, 3.0],  # 5个  → 3×5×2×6×6×5 = 5400
    },
}

def _get_active_strategies() -> list:
    """获取活跃策略列表（从仓库文件读取，回退到全部已注册策略）"""
    try:
        with open("strategy_warehouse.json", encoding="utf-8") as f:
            wh = json.load(f)
        active = wh.get("active", [])
        if active:
            return [s for s in active if s in STRATEGY_REGISTRY]
    except Exception:
        pass
    return list(STRATEGY_REGISTRY.keys())


def _normalize_param_interval(interval: str = "1d") -> str:
    interval = (interval or "1d").strip()
    return interval if interval in PARAM_INTERVALS else "1d"


def _read_json_file(fp: Path, default):
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def _write_json_file(fp: Path, data: dict):
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _df_span_days(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    start = df.index[0]
    end = df.index[-1]
    try:
        return int(max(1, math.ceil((end - start).total_seconds() / 86400)))
    except Exception:
        return 0


SIGNAL_OOS_PERIOD_BY_INTERVAL = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "2y",
    "1d": "5y", "1wk": "max", "1mo": "max",
}


def _signal_oos_period_for_interval(interval: str) -> str:
    """买/卖点「样本外验收」该取多长的历史。

    跟 WF 是同一个理由: _optimization_period_for_interval 给日线只有 1y(251 根)。
    样本外要切 6 段、最后 2 段留着不给优化器看 —— 251 根切完, 持出段只剩 ~84 根,
    实测 NVDA 4 个策略里 3 个是因为"持出段只有 4~9 个信号, 验不出来"被拒收的,
    不是策略不行, 是根本没数据可验。5y(~1250 根)持出段 ~420 根才验得动。
    日线不用 max: 样本外每个 trial 都要全量扫一遍, max 对老股票是上万根, 白等。
    """
    return SIGNAL_OOS_PERIOD_BY_INTERVAL.get(_normalize_param_interval(interval), "2y")


def _optimization_period_for_interval(interval: str) -> str:
    interval = _normalize_param_interval(interval)
    period_map = {
        "1m": "7d",
        "5m": "30d",
        "15m": "30d",
        "30m": "60d",
        "1h": "60d",
        "1d": "1y",
    }
    return period_map.get(interval, "1y")


def _extract_strategy_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}
    return {
        k: v for k, v in data.items()
        if isinstance(v, dict) and (
            "params" in v or "promoted_at" in v or "source_sharpe" in v
            or "source_return" in v or "std_return" in v or "method" in v
            or "sharpe" in v or "total_return" in v or "optimized_keys" in v
            or "combos_tested" in v
            or "signal_params" in v  # buy/sell 独立参数
        )
    }


def _extract_interval_layer(raw: dict, interval: str) -> dict:
    interval = _normalize_param_interval(interval)
    if isinstance(raw, dict) and isinstance(raw.get("intervals"), dict):
        return raw["intervals"].get(interval) or {}
    return raw or {}


def _opt_score_of(entry: dict):
    """比较两次优化结果用的分数。优先 sharpe, 没有就退回 total_return。"""
    if not isinstance(entry, dict):
        return None
    for k in ("sharpe", "total_return", "success_rate"):
        v = entry.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _worse_than_previous(new_entry, prev_entry, new_period, prev_period) -> bool:
    """本次结果是不是"又差又浅", 差到不该覆盖已有参数。

    为什么要这道闸: 重新优化以前是**无条件覆盖**的。用低 n_trials 快跑一轮,
    就会把之前精调的结果悄悄毁掉 —— 实测 RKLB 用 n_trials=20(100 combos)
    重跑一遍, claude_vol_adj_momentum 从 75.72% 掉到 5.39%、
    claude_entropy_momentum 从 60.24% 掉到 4.4%, 而原来那份是 200 combos,
    过程里没有任何提示。
    只在**可比**且**两项都更差**时才拦: 期间不同不拦(那是换了数据窗口重优化,
    新的该赢); 轮数更多不拦(跑得更深, 即便分数低也更可信)。
    """
    if not isinstance(prev_entry, dict) or not prev_entry.get("params"):
        return False
    if new_period and prev_period and str(new_period) != str(prev_period):
        return False        # 优化区间换了, 不可比
    new_combos = new_entry.get("combos_tested") or 0
    prev_combos = prev_entry.get("combos_tested") or 0
    if new_combos >= prev_combos:
        return False        # 跑得一样深或更深, 认新的
    ns, ps = _opt_score_of(new_entry), _opt_score_of(prev_entry)
    if ns is None or ps is None:
        return False
    return ns < ps


def _build_method_best(prev_mb: dict, payload: dict) -> dict:
    """按方法累积每个策略的历史最优 (bayesian / grid / coarse_fine 各一份)。

    这段逻辑原来长在 _save_optimized 里, 但那个函数开头有一句
        if "intervals" in data: _write_json_file(fp, data); return
    而 intervals 分层改造之后送进去的**永远**带 intervals —— 于是整段
    method_best 合并成了死代码。实测 194 个 interval 层里 method_best
    出现 0 次, 后果:
      · 「三种方案对比 ~10min」跑完拿不到任何对比输出
      · WF 的 贝叶斯/穷举/粗搜精搜 三档参数源恒为空 → 接口 400
      · 批量 WF 用这三档时整池跳过
    """
    out = {k: dict(v) for k, v in (prev_mb or {}).items() if isinstance(v, dict)}
    default_method = payload.get("method", "bayesian")
    stamp = datetime.utcnow().isoformat()

    def _better(sname: str, m: str, entry: dict):
        if not isinstance(entry, dict) or entry.get("error") or not entry.get("params"):
            return
        cur = out.setdefault(sname, {}).get(m, {})
        if entry.get("sharpe", -999) > cur.get("sharpe", -999):
            out[sname][m] = {**{k: v for k, v in entry.items() if k != "best_of"},
                             "updated_at": stamp}

    for sname, sdata in (payload.get("strategies") or {}).items():
        if isinstance(sdata, dict):
            _better(sname, sdata.get("method", default_method), sdata)
    # method="all" 时每种方法的原始结果在 method_comparison 里
    for sname, methods in (payload.get("method_comparison") or {}).items():
        if isinstance(methods, dict):
            for m, mdata in methods.items():
                _better(sname, m, mdata)
    return {k: v for k, v in out.items() if v}


def _write_interval_layer(raw: dict, interval: str, payload: dict, strategy_payload: bool = False,
                          merge_strategies: bool = True) -> dict:
    interval = _normalize_param_interval(interval)
    base = dict(raw) if isinstance(raw, dict) else {}
    intervals = dict(base.get("intervals") or {})
    if strategy_payload:
        legacy_payload = _extract_strategy_payload(base)
        if legacy_payload and "1d" not in intervals:
            intervals["1d"] = legacy_payload
        base = {k: v for k, v in base.items() if k == "intervals"}
    else:
        legacy_keys = {"symbol", "market", "period", "dynamic_position", "method",
                       "optimized_at", "total_combos_tested", "elapsed_seconds",
                       "strategies", "method_comparison", "method_best", "wf_validation"}
        legacy_payload = {k: base[k] for k in legacy_keys if k in base}
        if legacy_payload and "intervals" not in base and "1d" not in intervals:
            intervals["1d"] = legacy_payload
        base = {k: v for k, v in base.items() if k not in legacy_keys}
    # 关键: 按策略合并, 不能整层覆盖。
    # 原来是 intervals[interval] = payload —— 而 payload["strategies"] 只装本次
    # 跑出来的那几个。于是"只优化其中几个策略"会把**其余策略已有的优化参数
    # 全部抹掉**, 而且不留任何痕迹。
    # 实测: 先存 13 个, 再只优化 7 个 → 文件里只剩 7 个, s7~s12 没了。
    # 这正是"13 个策略跑完, 优化候选表只有 7 行"的成因之一。
    # 每条策略记上自己的 optimized_at, 免得整层的时间戳把旧条目说成是新跑的。
    kept_previous = []
    if merge_strategies and isinstance(payload, dict):
        prev = intervals.get(interval) or {}
        # method_best 以前没人往 payload 里放(那段逻辑死在 _save_optimized 的
        # early-return 后面), 所以下面那个按策略合并的循环拿不到任何新数据。
        if payload.get("strategies") or payload.get("method_comparison"):
            _mb = _build_method_best(prev.get("method_best"), payload)
            if _mb:
                payload = {**payload, "method_best": _mb}
        merged_payload = dict(payload)
        _stamp = payload.get("optimized_at")
        for key in ("strategies", "method_comparison", "method_best"):
            old_sub = prev.get(key)
            new_sub = payload.get(key)
            if not isinstance(old_sub, dict) and not isinstance(new_sub, dict):
                continue
            combined = dict(old_sub or {})
            if isinstance(new_sub, dict):
                for sname, entry in new_sub.items():
                    if key == "strategies" and isinstance(entry, dict):
                        prev_entry = (old_sub or {}).get(sname)
                        if _worse_than_previous(entry, prev_entry,
                                                payload.get("period"), prev.get("period")):
                            # 本次更差且跑得更浅 → 保留原参数, 别把精调结果毁了
                            kept_previous.append(sname)
                            continue
                        if _stamp:
                            entry = {**entry, "optimized_at": _stamp}
                    combined[sname] = entry
            combined = {k: v for k, v in combined.items() if v}
            if combined:
                merged_payload[key] = combined
        # wf_validation 是两层的 {来源: {策略: 结论}}, 上面那个单层合并套不进来,
        # 单独处理。不合并的话: 只优化 2 个策略 → WF 只算这 2 个 →
        # existing_wf["best"] = 这2个 → 其余 11 个的 WF 结论被整体抹掉,
        # 表里就是一排"—"。实测 13 个那次 WF 明明全算出来了, 之后跑 2 个就没了。
        old_wf = prev.get("wf_validation")
        new_wf = payload.get("wf_validation")
        if isinstance(old_wf, dict) and isinstance(new_wf, dict):
            combined_wf = {}
            for src in set(old_wf) | set(new_wf):
                o, n = old_wf.get(src), new_wf.get(src)
                if isinstance(o, dict) and isinstance(n, dict):
                    m = dict(o)
                    # 注: 调用方(optimize_stock)在跑 WF 之前已经把 kept_previous
                    # 那几个策略的 results 换成了"真正存下来的那份参数", 所以这里
                    # 传进来的 WF 结论跟存下来的参数是配套的, 直接收即可。
                    for sname, verdict in n.items():
                        m[sname] = verdict
                    combined_wf[src] = m
                else:
                    combined_wf[src] = n if n is not None else o
            merged_payload["wf_validation"] = combined_wf
        elif isinstance(old_wf, dict) and new_wf is None:
            # 本次没跑 WF(比如优化前的那次保存) —— 别把已有的 WF 结论丢了
            merged_payload["wf_validation"] = old_wf

        if kept_previous:
            merged_payload["kept_previous_strategies"] = sorted(set(kept_previous))
            logger.warning(
                f"[optimize] {len(set(kept_previous))} 个策略本次结果不如已有的(且轮数更少), "
                f"保留原参数: {', '.join(sorted(set(kept_previous))[:8])}")
        payload = merged_payload
    intervals[interval] = payload
    base["intervals"] = intervals
    return base


def _load_optimized(symbol: str) -> Optional[dict]:
    """加载已保存的个股优化参数"""
    fp = OPTIMIZED_DIR / f"{symbol.upper()}.json"
    return _read_json_file(fp, None)


def _load_optimized_interval(symbol: str, interval: str = "1d") -> Optional[dict]:
    raw = _load_optimized(symbol)
    if raw is None:
        return None
    layer = _extract_interval_layer(raw, interval)
    if not layer:
        return None
    if isinstance(layer, dict) and not layer.get("strategies") and not layer.get("method_best") and not layer.get("wf_validation"):
        return None
    return layer


def _load_custom_params(symbol: str) -> dict:
    """加载个股自定义标准参数 {strategy_name: {params, promoted_at, source_sharpe, ...}}"""
    fp = CUSTOM_PARAMS_DIR / f"{symbol.upper()}.json"
    return _read_json_file(fp, {})


def _load_custom_params_interval(symbol: str, interval: str = "1d") -> dict:
    raw = _load_custom_params(symbol.upper())
    layer = _extract_interval_layer(raw, interval)
    return _extract_strategy_payload(layer) if "intervals" in raw else _extract_strategy_payload(raw)


def _save_custom_params(symbol: str, data: dict):
    """保存个股自定义标准参数"""
    fp = CUSTOM_PARAMS_DIR / f"{symbol.upper()}.json"
    _write_json_file(fp, data)


def _save_custom_params_interval(symbol: str, interval: str, data: dict):
    fp = CUSTOM_PARAMS_DIR / f"{symbol.upper()}.json"
    raw = _load_custom_params(symbol.upper())
    merged = _write_interval_layer(raw, interval, data, strategy_payload=True)
    _write_json_file(fp, merged)


def _save_optimization_run(*, scope: str, pool_name: str = None, symbol: str = None,
                            symbols: list = None, signal_type: str = "buy_and_sell",
                            method: str = "bayesian", interval: str = "1d", n_trials: int = 200,
                            results: dict = None, wf_results: dict = None,
                            total_combos: int = 0, elapsed_seconds: float = 0):
    """保存优化运行记录到 JSON 文件 + PostgreSQL。"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope_label = pool_name or symbol or "unknown"

    # 构建 per_stock 数据
    per_stock = {}
    if results:
        for sym_or_strat, data in results.items():
            if isinstance(data, dict):
                per_stock[sym_or_strat] = {
                    k: v for k, v in data.items()
                    if k not in ("params", "optimized_keys", "buy_params", "sell_params")
                }

    # 把 WF 结论并进每条记录。
    # 原先 wf_results 只用来数 robust/fair/overfit 三个计数, 从不写进 per_stock ——
    # 而「优化历史 → 查看」的明细表有一列 WF, 读的正是 per_stock 里的 wf_verdict。
    # 于是那一列 1777 行全是 '-'(实测), 而配色逻辑是
    #     wf==='robust'?绿 : wf==='fair'?黄 : 红
    # '-' 落到 else 被染成红色 —— 看上去像"历史上每一条都过拟合"。
    if wf_results and per_stock:
        for sname, wf in wf_results.items():
            if not isinstance(wf, dict):
                continue
            entry = per_stock.get(sname)          # 单股 run: 顶层键就是策略名
            if isinstance(entry, dict):
                entry["wf_verdict"] = wf.get("verdict")
                entry["wf_consistency"] = wf.get("consistency")
                continue
            # 整池 run: 顶层键是股票, 逐策略数据在 strategy_details 里
            for _sym, sdata in per_stock.items():
                det = (sdata or {}).get("strategy_details")
                if isinstance(det, dict) and sname in det and isinstance(det[sname], dict):
                    det[sname]["wf_verdict"] = wf.get("verdict")
                    det[sname]["wf_consistency"] = wf.get("consistency")

    # 统计 WF 结果
    robust = fair = overfit = 0
    if wf_results:
        for v in wf_results.values():
            if isinstance(v, dict):
                verdict = v.get("verdict", "")
                if verdict == "robust": robust += 1
                elif verdict == "fair": fair += 1
                elif verdict == "overfit": overfit += 1

    summary = {
        "total_strategies": len(results) if results else 0,
        "robust_count": robust,
        "fair_count": fair,
        "overfit_count": overfit,
    }

    # 1) JSON 文件
    filename = f"{run_id}_{scope_label}_{signal_type}.json".replace(" ", "_")
    record = {
        "run_id": run_id, "scope": scope, "pool_name": pool_name, "symbol": symbol,
        "symbols": symbols, "signal_type": signal_type, "method": method,
        "interval": interval, "n_trials": n_trials,
        "total_combos": total_combos, "elapsed_seconds": round(elapsed_seconds, 1),
        "summary": summary, "per_stock": per_stock,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        fp = OPT_HISTORY_DIR / filename
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"[OptHistory] 已保存: {filename}")
    except Exception as e:
        logger.warning(f"[OptHistory] JSON保存失败: {e}")

    # 2) PostgreSQL
    try:
        from core_store import create_optimization_run
        create_optimization_run(
            run_id=run_id, scope=scope, pool_name=pool_name, symbol=symbol,
            symbols=symbols, signal_type=signal_type, method=method,
            interval=interval, n_trials=n_trials,
            total_strategies=summary["total_strategies"],
            robust_count=robust, fair_count=fair, overfit_count=overfit,
            total_combos=total_combos, elapsed_seconds=elapsed_seconds,
            per_stock=per_stock, summary=summary,
        )
        logger.info(f"[OptHistory] DB写入成功: {run_id}")
    except Exception as e:
        logger.warning(f"[OptHistory] DB写入失败: {e}")


def _save_signal_params(symbol: str, strategy_name: str, signal_type: str,
                        params: dict, metrics: dict = None, interval: str = "1d"):
    """保存 buy/sell 独立优化的参数到 custom_params/{SYMBOL}.json。

    存储结构:
    {
      "strategy_name": {
        "params": {...},               ← combined (buy_and_sell)
        "signal_params": {
          "buy":  {"ema_fast": 10, "rsi_buy": 30, ...},
          "sell": {"ema_fast": 15, "rsi_sell": 75, ...}
        },
        "signal_metrics": {
          "buy":  {"success_rate": 45.3, "avg_mfe": 12.5, ...},
          "sell": {"success_rate": 38.1, "avg_mfe": 8.2, ...}
        }
      }
    }
    """
    if signal_type not in ("buy", "sell"):
        return
    custom = _load_custom_params_interval(symbol.upper(), interval=interval)
    entry = custom.get(strategy_name, {})
    if "signal_params" not in entry:
        entry["signal_params"] = {}
    if "signal_metrics" not in entry:
        entry["signal_metrics"] = {}
    entry["signal_params"][signal_type] = params
    if metrics:
        entry["signal_metrics"][signal_type] = {
            **metrics,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    custom[strategy_name] = entry
    _save_custom_params_interval(symbol, interval, custom)


def _load_universal_params() -> dict:
    """加载通用标准参数 {strategy_name: {params, promoted_at, ...}}"""
    return _read_json_file(UNIVERSAL_PARAMS_FILE, {})


def _load_universal_params_interval(interval: str = "1d") -> dict:
    raw = _load_universal_params()
    layer = _extract_interval_layer(raw, interval)
    return _extract_strategy_payload(layer) if "intervals" in raw else _extract_strategy_payload(raw)


def _save_universal_params(data: dict):
    """保存通用标准参数"""
    _write_json_file(UNIVERSAL_PARAMS_FILE, data)


def _save_universal_params_interval(interval: str, data: dict):
    raw = _load_universal_params()
    merged = _write_interval_layer(raw, interval, data, strategy_payload=True)
    _write_json_file(UNIVERSAL_PARAMS_FILE, merged)


def _get_universal_strategy_params(strategy_name: str, interval: str = "1d") -> dict:
    """获取某策略的通用标准参数（内置默认 + universal_params 覆盖）"""
    strategy = get_strategy(strategy_name)
    params = {**getattr(strategy, "default_params", {})}
    universal = _load_universal_params_interval(interval).get(strategy_name, {})
    if universal.get("params"):
        params.update(universal["params"])
    return params


def _params_source_label(symbol: str, strategy_name: str, interval: str = "1d") -> str:
    """这只股票这条策略, 走的到底是个股标准还是通用标准。

    界面上"参数来源"以前是 11 个 return 站点各自写死的字面量 "个股标准" ——
    但 _get_standard_strategy_params 只有在 custom_params[股票][策略] 真存在时
    才会覆盖通用标准。实测 NASDAQ 100 里 79/100 只根本没有 custom_params,
    它们跑的是通用标准, 界面却一律写"个股标准"。
    """
    try:
        custom = _load_custom_params_interval(symbol.upper(), interval=interval)
        if strategy_name in custom and custom[strategy_name].get("params"):
            return "个股标准"
    except Exception:
        pass
    return "通用标准"


def _params_source_summary(symbols, strategy_names, interval: str = "1d") -> str:
    """批量场景: 一批股票×策略里有几个走个股标准、几个走通用标准。"""
    n_custom = n_univ = 0
    for sym in symbols or []:
        for sn in strategy_names or []:
            if _params_source_label(sym, sn, interval) == "个股标准":
                n_custom += 1
            else:
                n_univ += 1
    if n_custom and n_univ:
        return f"个股标准 {n_custom} / 通用标准 {n_univ}"
    if n_custom:
        return "个股标准"
    return "通用标准"


def _get_standard_strategy_params(symbol: str, strategy_name: str, interval: str = "1d",
                                   signal_type: str = "buy_and_sell") -> dict:
    """获取某股票某策略的个股标准参数（通用标准 + custom_params 覆盖）。

    signal_type:
      - "buy_and_sell": 合并 buy + sell 参数（buy 为底，sell 覆盖）
      - "buy": 优先用 buy 专属参数，fallback 到 combined
      - "sell": 优先用 sell 专属参数，fallback 到 combined
    """
    # 1) 基础参数（内置默认 + universal）
    params = _get_universal_strategy_params(strategy_name, interval=interval)

    # 2) custom_params 的 combined 层覆盖
    custom = _load_custom_params_interval(symbol.upper(), interval=interval)
    if strategy_name in custom and custom[strategy_name].get("params"):
        params.update(custom[strategy_name]["params"])

    # 3) signal_type 层叠加
    if signal_type in ("buy", "sell"):
        # 尝试读 signal_params.{buy|sell}
        sig_params = custom.get(strategy_name, {}).get("signal_params", {}).get(signal_type, {})
        if sig_params:
            params.update(sig_params)
    elif signal_type == "buy_and_sell":
        # 合并 buy + sell：buy 为底层，sell 覆盖（sell 通常调 exit 相关参数）
        sig_all = custom.get(strategy_name, {}).get("signal_params", {})
        buy_p = sig_all.get("buy", {})
        sell_p = sig_all.get("sell", {})
        if buy_p or sell_p:
            params.update(buy_p)
            params.update(sell_p)

    return params


def _get_dual_signal_params(symbol: str, strategy_name: str, interval: str = "1d") -> dict:
    """获取 buy / sell 两套独立参数。

    返回格式:
    {
      "has_dual": True/False,          # 是否存在分离的 buy/sell 参数
      "buy_params":  {...},            # buy 专属参数（base + buy signal_params）
      "sell_params": {...},            # sell 专属参数（base + sell signal_params）
    }
    如果没有分离参数，has_dual=False，buy_params=sell_params=combined params。
    """
    custom = _load_custom_params_interval(symbol.upper(), interval=interval)
    sig_all = custom.get(strategy_name, {}).get("signal_params", {})
    buy_sp = sig_all.get("buy", {})
    sell_sp = sig_all.get("sell", {})

    has_dual = bool(buy_sp or sell_sp)

    # base = 通用默认 + custom combined
    base = _get_universal_strategy_params(strategy_name, interval=interval)
    if strategy_name in custom and custom[strategy_name].get("params"):
        base.update(custom[strategy_name]["params"])

    if has_dual:
        buy_params = {**base}
        if buy_sp:
            buy_params.update(buy_sp)
        sell_params = {**base}
        if sell_sp:
            sell_params.update(sell_sp)
    else:
        buy_params = {**base}
        sell_params = {**base}

    return {
        "has_dual": has_dual,
        "buy_params": buy_params,
        "sell_params": sell_params,
    }


def _pool_avg_return(strategy_name: str, params: dict, pool_name: str = UNIVERSAL_BENCHMARK_POOL,
                     period: str = "2y", market: str = "US", dynamic_pos: bool = False,
                     interval: str = "1d", executor_mode: str = "process") -> dict:
    """评估某策略参数在指定股票池上的平均收益。

    executor_mode 默认 process: 这里是把策略在 NASDAQ100 全池跑一遍 (100 次回测),
    纯 CPU 密集, 之前默认 thread 等于被 GIL 串死。其它调用点本来就显式传了 process,
    只有 _promote_to_universal_if_better 这条链一路用着默认值。
    _resolve_pool_executor_mode 里已有守卫 (任务数 <=1 / 起不了进程池 / 已在 worker 里)
    会自动回退 thread, 所以改默认不会造成进程池嵌套。
    """
    interval = _normalize_param_interval(interval)
    pools = _load_pools()
    pool = pools.get(pool_name)
    if not pool:
        return {"avg_return": None, "count": 0, "pool": pool_name, "error": "pool not found"}
    symbols = pool.get("symbols", [])
    if not symbols:
        return {"avg_return": None, "count": 0, "pool": pool_name, "error": "empty pool"}

    data_mtime = _data_mtime_sig(symbols, market, interval)
    params_sig = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    cache_key = f"pool_avg:{pool_name}:{strategy_name}:{interval}:{period}:{dynamic_pos}:{data_mtime}:{params_sig}"
    cached = cache_get(cache_key, ttl=86400)
    if cached is not None:
        return cached

    tasks = [
        (str(sym).upper(), market, interval, period, strategy_name, params, dynamic_pos)
        for sym in symbols
    ]
    returns = _collect_pool_backtest_returns(tasks, _pool_backtest_return_worker, executor_mode=executor_mode)

    avg_return = round(float(np.mean(returns)), 2) if returns else None
    result = {"avg_return": avg_return, "count": len(returns), "pool": pool_name}
    cache_set(cache_key, result)
    return result


def _can_spawn_process_pool() -> bool:
    """能否再开 ProcessPoolExecutor: executor worker (daemonic) 里不能嵌套;
    uvicorn --reload 的应用子进程可以 (它就是主服务, 只是名字叫 SpawnProcess-1)."""
    import sys
    try:
        import proc_role
        if proc_role.is_pool_worker():
            return False
    except Exception:
        return False
    main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    return not str(main_file).startswith("<")


def _resolve_pool_executor_mode(executor_mode: str, task_count: int) -> str:
    import sys

    mode = str(executor_mode or "thread").lower()
    if task_count <= 1:
        return "thread"
    main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    if str(main_file).startswith("<"):
        return "thread"
    if mode == "process":
        try:
            return "process" if _can_spawn_process_pool() else "thread"
        except Exception:
            return "thread"
    if mode == "auto":
        if _can_spawn_process_pool() and task_count >= 6:
            return "process"
    return "thread"


def _pool_backtest_return_worker(args):
    sym, market, interval, period, strategy_name, params, dynamic_pos = args
    try:
        df = get_ohlcv(sym, market, interval=interval, period=period)
        if len(df) < 60:
            return None
        r = run_strategy_backtest(df, strategy_name, params_override=params, dynamic_position=dynamic_pos)
        return float(r["summary"].get("total_return", 0))
    except Exception:
        return None


def _pool_standard_return_worker(args):
    sym, market, interval, period, strategy_name, dynamic_pos = args
    try:
        df = get_ohlcv(sym, market, interval=interval, period=period)
        if len(df) < 60:
            return None
        params = _get_standard_strategy_params(sym, strategy_name, interval=interval)
        r = run_strategy_backtest(df, strategy_name, params_override=params, dynamic_position=dynamic_pos)
        return float(r["summary"].get("total_return", 0))
    except Exception:
        return None


def _collect_pool_backtest_returns(tasks: list, worker, executor_mode: str = "thread") -> list:
    if not tasks:
        return []
    mode = _resolve_pool_executor_mode(executor_mode, len(tasks))
    executor_cls = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
    n_workers = (
        min(max(2, N_WORKERS), len(tasks))
        if mode == "process"
        else min(max(2, N_WORKERS // 2), len(tasks))
    )
    returns = []
    with executor_cls(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, task) for task in tasks]
        for fut in as_completed(futures):
            try:
                ret = fut.result()
                if ret is not None:
                    returns.append(ret)
            except Exception:
                continue
    return returns


def _promote_to_universal_if_better(symbol: str, strategy_name: str, candidate_params: dict,
                                    period: str = "2y", dynamic_pos: bool = False,
                                    interval: str = "1d", executor_mode: str = "process"):
    """若候选参数在 NASDAQ100 的平均收益优于当前通用标准，则升级通用标准。

    这个函数挂在 _promote_to_standard 里, 而 _promote_to_standard 又挂在
    批量优化的 WF 收割循环上 —— 默认走线程池时, 每升级一个策略就要用 GIL 串行跑
    200 次回测, 直接堵住整个批量任务。默认改 process。
    """
    interval = _normalize_param_interval(interval)
    if not candidate_params:
        return {"updated": False, "reason": "empty_params"}

    universal = _load_universal_params_interval(interval)
    current_info = universal.get(strategy_name, {})
    current_params = _get_universal_strategy_params(strategy_name, interval=interval)

    cand_eval = _pool_avg_return(
        strategy_name, candidate_params, period=period, dynamic_pos=dynamic_pos,
        interval=interval, executor_mode=executor_mode
    )
    curr_eval = _pool_avg_return(
        strategy_name, current_params, period=period, dynamic_pos=dynamic_pos,
        interval=interval, executor_mode=executor_mode
    )
    cand_avg = cand_eval.get("avg_return")
    curr_avg = curr_eval.get("avg_return")
    cand_count = cand_eval.get("count", 0)
    curr_count = curr_eval.get("count", 0)

    if cand_avg is None or curr_avg is None:
        logger.info(f"⏭ universal promote skip {strategy_name}: pool avg unavailable ({cand_count}/{curr_count})")
        return {"updated": False, "reason": "pool_avg_unavailable", "candidate_avg_return": cand_avg, "baseline_avg_return": curr_avg}
    if min(cand_count, curr_count) < 30:
        logger.info(f"⏭ universal promote skip {strategy_name}: insufficient pool coverage ({cand_count}/{curr_count})")
        return {"updated": False, "reason": "insufficient_pool_coverage", "candidate_avg_return": cand_avg, "baseline_avg_return": curr_avg}
    if cand_avg <= curr_avg:
        logger.info(f"⏭ universal keep {strategy_name}: candidate {cand_avg:.2f}% <= current {curr_avg:.2f}%")
        return {"updated": False, "reason": "not_better", "candidate_avg_return": cand_avg, "baseline_avg_return": curr_avg}

    universal[strategy_name] = {
        "params": candidate_params,
        "promoted_at": datetime.utcnow().isoformat(),
        "source_symbol": symbol.upper(),
        "interval": interval,
        "pool": UNIVERSAL_BENCHMARK_POOL,
        "period": period,
        "candidate_avg_return": cand_avg,
        "baseline_avg_return": curr_avg,
        "delta_avg_return": round(cand_avg - curr_avg, 2),
        "replaced_promoted_at": current_info.get("promoted_at"),
    }
    _save_universal_params_interval(interval, universal)
    logger.info(
        f"🌍 universal promoted {strategy_name} from {symbol.upper()}: "
        f"{cand_avg:.2f}% > {curr_avg:.2f}% on {UNIVERSAL_BENCHMARK_POOL}"
    )
    return {
        "updated": True,
        "strategy": strategy_name,
        "source_symbol": symbol.upper(),
        "candidate_avg_return": cand_avg,
        "baseline_avg_return": curr_avg,
        "delta_avg_return": round(cand_avg - curr_avg, 2),
    }


def _promote_to_standard(symbol: str, market: str, wf_validation: dict, opt_strategies: dict,
                         period: str = "2y", dynamic_pos: bool = False,
                         interval: str = "1d", task_id: str = ""):
    """WF 验证稳健的策略 → 写入个股自定义标准参数（永久生效）
    条件：1) WF consistency ≥ 55%  2) 优化收益 > 标准参数收益

    task_id: 传了就上报进度。这一步以前完全不可见, 但实测在 /compare-optimized
    里占到 62%(3.7s / 5.9s) —— 前面的对比回测进度条早就 100% 了, 它还在这儿
    逐个策略跑标准口径回测 + 通用参数晋级, 面板上就是十几秒纹丝不动。

    这里不把外层循环并行化: _promote_to_universal_if_better 内部已经在起进程池,
    外面再套一层就是嵌套池 (本项目出过 336 个 worker 抢 12 核的事故)。
    """
    interval = _normalize_param_interval(interval)
    custom = _load_custom_params_interval(symbol, interval=interval)
    changed = False
    custom_promoted = []
    universal_promoted = []

    # 加载数据用于标准参数回测对比（懒加载，所有策略共用）
    _std_df = None

    _cands = [(sn, wi) for sn, wi in wf_validation.items() if wi.get("promoted")]
    if task_id:
        prog_step(task_id, "promote", total=len(_cands), done=0, unit=" 个策略",
                  workers=1, running=[], note="")
    _pdone = 0

    for sname, wf_info in _cands:
        if task_id:
            # 这一步实测占 /compare-optimized 整个请求的 62%, 没有检查点的话
            # ⏹ 在这儿同样是失效的
            prog_check_cancel(task_id)
            prog_update(task_id, done=_pdone, total=len(_cands), running=[sname])
        _pdone += 1
        if sname not in opt_strategies:
            continue
        opt_data = opt_strategies[sname]
        params = opt_data.get("params")
        if not params:
            continue

        opt_return = opt_data.get("total_return")
        if opt_return is None:
            continue

        # 比较优化收益 vs 标准参数收益（含已有 custom 参数）
        try:
            if _std_df is None:
                _std_df = get_ohlcv(symbol, market, interval=interval, period=period)
                if len(_std_df) < 10:
                    logger.warning(f"_promote_to_standard {symbol}: 数据不足, 跳过收益对比")
                    break

            # 标准参数回测（如果已有 custom 参数则用 custom，否则用全局默认）
            std_result = run_strategy_backtest(
                _std_df, sname, params_override=_get_standard_strategy_params(symbol, sname, interval=interval)
            )
            std_return = std_result.get("summary", {}).get("total_return", 0)

            if opt_return <= std_return:
                logger.info(f"⏭ {symbol}/{sname} 优化收益({opt_return:.2f}%) "
                            f"≤ 标准收益({std_return:.2f}%), 不升级")
                continue
        except Exception as e:
            logger.warning(f"_promote_to_standard {symbol}/{sname} 标准回测失败: {e}, 跳过")
            continue

        custom[sname] = {
            "params": params,
            "promoted_at": datetime.utcnow().isoformat(),
            "source_sharpe": opt_data.get("sharpe", 0),
            "source_return": round(opt_return, 2),
            "std_return": round(std_return, 2),
            "delta_return": round(opt_return - std_return, 2),
            "source_method": opt_data.get("method", "unknown"),
            "wf_consistency": wf_info.get("consistency", 0),
            "interval": interval,
        }
        changed = True
        custom_promoted.append({
            "strategy": sname,
            "optimized_return": round(opt_return, 2),
            "baseline_return": round(std_return, 2),
            "delta_return": round(opt_return - std_return, 2),
            "wf_consistency": round(float(wf_info.get("consistency", 0)), 1),
        })
        logger.info(f"📌 {symbol}/{sname} promoted: 优化{opt_return:.2f}% > "
                    f"标准{std_return:.2f}% (+{opt_return - std_return:.2f}%), "
                    f"WF={wf_info.get('consistency', 0):.1f}%")

        uni_result = _promote_to_universal_if_better(
            symbol, sname, params, period=period, dynamic_pos=dynamic_pos, interval=interval
        )
        if uni_result and uni_result.get("updated"):
            universal_promoted.append(uni_result)

    if task_id:
        prog_update(task_id, done=len(_cands), total=len(_cands), running=[])
    if changed:
        _save_custom_params_interval(symbol, interval, custom)
    return {
        "symbol": symbol.upper(),
        "custom_promoted": custom_promoted,
        "universal_promoted": universal_promoted,
        "custom_count": len(custom_promoted),
        "universal_count": len(universal_promoted),
        "changed": changed or bool(universal_promoted),
    }

def _save_optimized(symbol: str, data: dict):
    """智能保存个股优化参数 — 只保留更好的结果

    存储结构:
    {
      "strategies": {
        "claude_dual_ma_cross": {           # ← "已优化" = 所有方法中最好的
            "params": {...}, "sharpe": 1.8, "method": "grid", ...
        },
        ...
      },
      "method_best": {               # ← 每种方法各自的历史最优
        "claude_dual_ma_cross": {
            "bayesian": {"params": {...}, "sharpe": 1.5, ...},
            "grid":     {"params": {...}, "sharpe": 1.8, ...},
            "coarse_fine": {"params": {...}, "sharpe": 1.6, ...},
        },
        ...
      },
      "method_comparison": {...},     # ← 最近一次 "all" 对比的原始数据
      ...
    }
    """
    fp = OPTIMIZED_DIR / f"{symbol.upper()}.json"

    if isinstance(data, dict) and "intervals" in data:
        _write_json_file(fp, data)
        return

    # 加载已有数据
    old = {}
    if fp.exists():
        try:
            old = json.loads(fp.read_text(encoding="utf-8"))
        except:
            old = {}

    old_strategies = old.get("strategies", {})
    old_method_best = old.get("method_best", {})

    new_strategies = data.get("strategies", {})
    new_method = data.get("method", "bayesian")
    new_mc = data.get("method_comparison", {})

    # 1) 更新每种方法的历史最优 (method_best)
    for sname, sdata in new_strategies.items():
        if sname not in old_method_best:
            old_method_best[sname] = {}

        # 确定本次结果属于哪种方法
        result_method = sdata.get("method", new_method)
        new_sharpe = sdata.get("sharpe", -999)

        # 与该方法的历史最优比较
        prev = old_method_best[sname].get(result_method, {})
        prev_sharpe = prev.get("sharpe", -999)

        if new_sharpe > prev_sharpe:
            old_method_best[sname][result_method] = {
                k: v for k, v in sdata.items()
                if k not in ("best_of",)  # 去掉 best_of 标记
            }
            old_method_best[sname][result_method]["updated_at"] = datetime.utcnow().isoformat()

    # 如果是 method=all，还要处理 method_comparison 中的每种方法
    if new_mc:
        for sname, methods_data in new_mc.items():
            if sname not in old_method_best:
                old_method_best[sname] = {}
            for m, mdata in methods_data.items():
                if mdata.get("error"):
                    continue
                m_sharpe = mdata.get("sharpe", -999)
                prev = old_method_best[sname].get(m, {})
                if m_sharpe > prev.get("sharpe", -999):
                    old_method_best[sname][m] = {**mdata}
                    old_method_best[sname][m]["updated_at"] = datetime.utcnow().isoformat()

    # 2) "已优化" (strategies) = 所有方法中 Sharpe 最高的
    merged_strategies = {}
    for sname in set(list(old_strategies.keys()) + list(new_strategies.keys())):
        candidates = []

        # 候选1: 旧的已优化结果
        if sname in old_strategies:
            candidates.append(old_strategies[sname])

        # 候选2: 本次新结果
        if sname in new_strategies:
            candidates.append(new_strategies[sname])

        # 候选3: method_best 中各方法的最优
        if sname in old_method_best:
            for m, mdata in old_method_best[sname].items():
                if not mdata.get("error") and mdata.get("params"):
                    candidates.append(mdata)

        # 选 Sharpe 最高的
        best = max(candidates, key=lambda x: x.get("sharpe", -999)) if candidates else None
        if best and best.get("sharpe", -999) > -999:
            merged_strategies[sname] = {k: v for k, v in best.items() if k != "updated_at"}

    # 3) 组装最终保存数据
    save = {
        "symbol": data.get("symbol", symbol.upper()),
        "market": data.get("market", old.get("market", "US")),
        "period": data.get("period", old.get("period", "2y")),
        "dynamic_position": data.get("dynamic_position", False),
        "method": new_method,
        "optimized_at": datetime.utcnow().isoformat(),
        "total_combos_tested": data.get("total_combos_tested", 0),
        "elapsed_seconds": data.get("elapsed_seconds", 0),
        "strategies": merged_strategies,
        "method_best": old_method_best,
    }
    if new_mc:
        save["method_comparison"] = new_mc
    elif old.get("method_comparison"):
        save["method_comparison"] = old["method_comparison"]

    # 保留 WF 验证结果（按 param_source 分层存储，合并新旧）
    old_wf = old.get("wf_validation") or {}
    new_wf = data.get("wf_validation") or {}
    # 兼容旧格式：如果旧数据是扁平的（没有 param_source 嵌套），迁移为 "best"
    if old_wf and any(isinstance(v, dict) and "verdict" in v for v in old_wf.values()):
        old_wf = {"best": old_wf}
    if new_wf and any(isinstance(v, dict) and "verdict" in v for v in new_wf.values()):
        new_wf = {"best": new_wf}
    merged_wf = {**old_wf, **new_wf}  # 新的覆盖旧的同名 source
    if merged_wf:
        save["wf_validation"] = merged_wf

    fp.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")

    # 日志
    updated = []
    for sname in new_strategies:
        old_s = old_strategies.get(sname, {}).get("sharpe", -999)
        new_s = merged_strategies.get(sname, {}).get("sharpe", -999)
        if new_s > old_s:
            updated.append(f"{sname}({old_s:.2f}→{new_s:.2f})")
    if updated:
        logger.info(f"[{symbol}] 参数更新: {', '.join(updated)}")
    else:
        logger.info(f"[{symbol}] 本次优化未产生更优参数，保留历史最优")

# ═══════════════════════════════════════════════════════════════════
#  美股热门股 — Yahoo 热度 + 本地量价热度混合更新
# ═══════════════════════════════════════════════════════════════════

def _hot_score_percentile(values: dict) -> dict:
    if not values:
        return {}
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {}
    ranks = s.rank(method="average", pct=True)
    return {str(idx): round(float(val), 4) for idx, val in ranks.items()}


def _run_hot_fetch_with_timeout(fn, timeout_sec: int = 8):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_sec)
    except FutureTimeoutError:
        logger.warning("Hot stock source timed out")
        fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        return None
    except Exception as e:
        logger.warning(f"Hot stock source failed: {e}")
        ex.shutdown(wait=False, cancel_futures=True)
        return None
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _collect_yahoo_hot_candidates() -> tuple:
    from yfinance import EquityQuery
    from collections import Counter

    all_quotes = {}
    source_count = Counter()

    screeners = [
        ("most_actives", "predefined"),
        ("day_gainers", "predefined"),
        ("growth_technology_stocks", "predefined"),
    ]

    for screen_name, _ in screeners:
        try:
            result = _run_hot_fetch_with_timeout(lambda: yf.screen(screen_name, count=25), timeout_sec=8) or {}
            for q in result.get("quotes", []):
                sym = str(q.get("symbol", "")).upper().strip()
                if not sym or "." in sym or "-" in sym:
                    continue
                exchange = q.get("exchange", "")
                if exchange not in ("NMS", "NYQ", "NGM", "NCM", "NAS", "PCX", "ASE"):
                    continue
                mcap = q.get("marketCap", 0) or 0
                if mcap < 2e9:
                    continue
                source_count[sym] += 1
                all_quotes[sym] = q
        except Exception as e:
            logger.warning(f"Screener {screen_name} failed: {e}")

    try:
        custom_q = EquityQuery("and", [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gt", ["dayvolume", 20_000_000]),
            EquityQuery("gt", ["intradaymarketcap", 10_000_000_000]),
        ])
        result = _run_hot_fetch_with_timeout(
            lambda: yf.screen(custom_q, size=25, sortField="dayvolume", sortAsc=False),
            timeout_sec=10
        ) or {}
        for q in result.get("quotes", []):
            sym = str(q.get("symbol", "")).upper().strip()
            if not sym or "." in sym or "-" in sym:
                continue
            exchange = q.get("exchange", "")
            if exchange not in ("NMS", "NYQ", "NGM", "NCM", "NAS", "PCX", "ASE"):
                continue
            mcap = q.get("marketCap", 0) or 0
            if mcap < 2e9:
                continue
            source_count[sym] += 1
            all_quotes[sym] = q
    except Exception as e:
        logger.warning(f"Custom screener failed: {e}")

    return all_quotes, source_count


def _collect_local_hot_candidates(limit: int = 120) -> list:
    us_dir = Path(__file__).parent / "market_data" / "US"
    if not us_dir.exists():
        return []

    rows = []
    for fp in sorted(us_dir.glob("*_1d.parquet")):
        sym = fp.stem.replace("_1d", "").upper()
        try:
            df = pd.read_parquet(fp, columns=["open", "high", "close", "volume"])
            if df is None or len(df) < 25:
                continue
            df = df.dropna(subset=["open", "high", "close", "volume"]).tail(40)
            if len(df) < 25:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = safe_float(last.get("close"))
            prev_close = safe_float(prev.get("close"))
            volume = safe_float(last.get("volume"))
            if close is None or prev_close in (None, 0) or volume is None:
                continue
            avg20_volume = safe_float(df["volume"].tail(20).mean())
            if avg20_volume in (None, 0):
                continue
            dollar_volume = close * volume
            if close < 2 or dollar_volume < 5_000_000:
                continue
            day_change_pct = (close / prev_close - 1.0) * 100.0
            volume_ratio = volume / avg20_volume
            high20 = safe_float(df["high"].tail(20).max())
            high_proximity = (close / high20) if high20 not in (None, 0) else None
            fund = _load_fundamental_snapshot(sym, "US")
            market_cap = safe_float(fund.get("market_cap"))
            if market_cap is not None and market_cap < 2e9:
                continue
            rows.append({
                "symbol": sym,
                "name": fund.get("name") or _get_stock_name(sym, "US"),
                "price": round(close, 2),
                "volume": int(volume),
                "avg20_volume": int(avg20_volume),
                "market_cap": market_cap,
                "change_pct": round(day_change_pct, 2),
                "dollar_volume": dollar_volume,
                "volume_ratio": volume_ratio,
                "abs_move_pct": abs(day_change_pct),
                "high_proximity": high_proximity or 0,
            })
        except Exception:
            continue

    if not rows:
        return []

    dollar_rank = _hot_score_percentile({r["symbol"]: r["dollar_volume"] for r in rows})
    vol_ratio_rank = _hot_score_percentile({r["symbol"]: r["volume_ratio"] for r in rows})
    move_rank = _hot_score_percentile({r["symbol"]: r["abs_move_pct"] for r in rows})
    high_rank = _hot_score_percentile({r["symbol"]: r["high_proximity"] for r in rows})

    for r in rows:
        sym = r["symbol"]
        r["local_hot_score"] = round(
            0.35 * dollar_rank.get(sym, 0) +
            0.30 * vol_ratio_rank.get(sym, 0) +
            0.20 * move_rank.get(sym, 0) +
            0.15 * high_rank.get(sym, 0),
            4
        )
        tags = []
        if r["volume_ratio"] >= 2:
            tags.append("放量")
        if r["abs_move_pct"] >= 5:
            tags.append("异动")
        if r["high_proximity"] >= 0.98:
            tags.append("近20日高点")
        r["tags"] = tags

    rows.sort(key=lambda x: (x["local_hot_score"], x["dollar_volume"]), reverse=True)
    return rows[:limit]


def fetch_hot_us_stocks() -> dict:
    """混合 Yahoo 热门榜 + 本地量价热度，生成当前热门美股 Top 30。"""
    all_quotes, source_count = _collect_yahoo_hot_candidates()
    local_rows = _collect_local_hot_candidates(limit=120)
    local_map = {row["symbol"]: row for row in local_rows}

    yahoo_volume_rank = _hot_score_percentile({
        sym: (q.get("regularMarketVolume", 0) or 0) for sym, q in all_quotes.items()
    })
    yahoo_move_rank = _hot_score_percentile({
        sym: abs(q.get("regularMarketChangePercent", 0) or 0) for sym, q in all_quotes.items()
    })
    yahoo_source_rank = _hot_score_percentile({sym: cnt for sym, cnt in source_count.items()})

    candidate_symbols = sorted(set(all_quotes.keys()) | set(local_map.keys()))
    scored = []
    for sym in candidate_symbols:
        q = all_quotes.get(sym, {})
        local = local_map.get(sym, {})
        yahoo_score = round(
            0.45 * yahoo_source_rank.get(sym, 0) +
            0.35 * yahoo_volume_rank.get(sym, 0) +
            0.20 * yahoo_move_rank.get(sym, 0),
            4
        ) if q else 0.0
        local_score = float(local.get("local_hot_score", 0) or 0)
        final_score = round(0.55 * yahoo_score + 0.45 * local_score, 4)
        if source_count.get(sym, 0) > 0 and local_score > 0:
            final_score = round(final_score + 0.08, 4)

        name = (
            q.get("shortName") or q.get("longName") or
            local.get("name") or sym
        )
        price = q.get("regularMarketPrice")
        if price in (None, 0):
            price = local.get("price")
        change_pct = q.get("regularMarketChangePercent")
        if change_pct is None:
            change_pct = local.get("change_pct")
        volume = q.get("regularMarketVolume")
        if not volume:
            volume = local.get("volume", 0)
        market_cap = q.get("marketCap")
        if not market_cap:
            market_cap = local.get("market_cap")

        reason = []
        if source_count.get(sym, 0):
            reason.append(f"Yahoo热榜{x if (x:=source_count[sym]) else ''}")
        if local.get("volume_ratio", 0) >= 2:
            reason.append("显著放量")
        if abs(local.get("change_pct", 0) or 0) >= 5:
            reason.append("价格异动")
        if local.get("high_proximity", 0) >= 0.98:
            reason.append("接近20日高点")

        scored.append({
            "symbol": sym,
            "name": name,
            "volume": int(volume or 0),
            "change_pct": round(float(change_pct or 0), 2),
            "market_cap": safe_float(market_cap),
            "price": round(float(price or 0), 2),
            "sources": int(source_count.get(sym, 0)),
            "yahoo_score": yahoo_score,
            "local_score": local_score,
            "hot_score": final_score,
            "reason": " · ".join(reason[:3]),
            "tags": local.get("tags", []),
        })

    scored.sort(key=lambda x: (x["hot_score"], x["sources"], x["volume"]), reverse=True)
    top_rows = scored[:30]
    ranked = [row["symbol"] for row in top_rows]
    details = {row["symbol"]: {k: v for k, v in row.items() if k != "symbol"} for row in top_rows}

    return {
        "symbols": ranked,
        "details": details,
        "updated_at": datetime.now().isoformat(),
        "total_screened": len(candidate_symbols),
        "sources": {
            "yahoo_candidates": len(all_quotes),
            "local_candidates": len(local_rows),
            "method": "Yahoo screener + local price/volume heat",
        },
    }


def _update_hot_pool():
    """拉取热门股并更新股票池 + 缓存"""
    try:
        data = fetch_hot_us_stocks()
        if not data["symbols"]:
            logger.warning("No hot stocks fetched, skipping update")
            return data

        # 更新股票池 — 只动热门池这一个 key, 且在锁内重新读盘面,
        # 不会把用户同期的其它池编辑覆盖掉
        def _set_hot(pools):
            prev = pools.get(HOT_POOL_NAME) or {}
            entry = {k: v for k, v in prev.items() if k not in ("market", "symbols")}
            entry["market"] = "US"
            entry["symbols"] = data["symbols"]
            pools[HOT_POOL_NAME] = entry
        _mutate_pools(_set_hot)

        # 保存详情缓存
        HOT_CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"🔥 Hot stocks updated: {len(data['symbols'])} symbols")
        return data
    except Exception as e:
        logger.error(f"Failed to update hot stocks: {e}")
        return None


def _load_hot_cache() -> Optional[dict]:
    if HOT_CACHE_FILE.exists():
        try:
            return json.loads(HOT_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# stock_pools.json 的读-改-写锁. 之前无锁 + 直接覆盖写, 两个后果:
#   1) 热门股后台刷新 (_update_hot_pool) 与用户的加/删/排序撞上时, 后写的那份
#      整体覆盖前一份 → 用户的编辑被悄悄吃掉 (实测出现过)
#   2) write_text 非原子, 写一半被打断会截断整个文件 (12 个池一起没)
# 所有改动一律走 _mutate_pools(), 单纯读用 _load_pools().
_POOLS_LOCK = threading.RLock()


def _load_pools() -> dict:
    with _POOLS_LOCK:
        if POOLS_FILE.exists():
            return json.loads(POOLS_FILE.read_text(encoding="utf-8"))
        # 默认股票池
        default = {
            "US Tech": {"market": "US", "symbols": ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "META", "AMZN", "AMD"]},
            "US Finance": {"market": "US", "symbols": ["JPM", "BAC", "GS", "MS", "V", "MA"]},
            "Crypto": {"market": "CRYPTO", "symbols": ["BTC", "ETH", "SOL", "BNB"]},
        }
        _save_pools(default)
        return default


def _save_pools(pools: dict):
    """原子写 (临时文件 + replace). 单独调用只在"已持锁"的路径里用; 外部改动请用 _mutate_pools."""
    with _POOLS_LOCK:
        tmp = POOLS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pools, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(POOLS_FILE)


def _mutate_pools(fn):
    """锁内 读-改-写: fn(pools) 就地修改 pools, 其返回值原样透传给调用方.

    fn 抛异常则不落盘. 保证同一时刻只有一个写者, 且写者看到的一定是最新盘面
    (不会拿几秒前的快照整体盖回去).
    """
    with _POOLS_LOCK:
        pools = _load_pools()
        out = fn(pools)
        _save_pools(pools)
        return out


def _default_market_data_config() -> dict:
    pools = _load_pools()
    pool_rules = {}
    optimizer_defaults = {
        "1m": "7d",
        "5m": "30d",
        "15m": "30d",
        "30m": "60d",
        "1h": "60d",
        "1d": "2y",
        "1wk": "5y",
        "1mo": "10y",
    }
    # selector 以前只 seed 了 {"1d": "1y"} 一个 key, 而周期下拉有 8 个周期。
    # 前端 `cfg.pools[p][usage][interval] || '1y'` 于是对 1h/5m 显示
    # 「默认同步 -」并把周期设成 1年 —— 用户一点「按规则增量同步」就把 1y
    # 当 period 发下去, 而 Yahoo 的 1h 根本给不了一年。这里补齐成后端
    # 真正的兜底值(_optimization_period_for_interval), 只有 1d 特殊(选股用 1y)。
    selector_defaults = {iv: ("1y" if iv == "1d" else optimizer_defaults.get(iv)
                              or _optimization_period_for_interval(iv))
                         for iv in PARAM_INTERVALS}
    for pool_name, info in pools.items():
        pool_rules[pool_name] = {
            "market": info.get("market", "US"),
            "selector": selector_defaults.copy(),
            "optimizer": optimizer_defaults.copy(),
        }
    markets = {m: {"enabled": True, "selector": selector_defaults.copy()} for m in CORE_MARKETS}
    for info in pools.values():
        market = str(info.get("market", "")).upper().strip()
        if market and market not in markets:
            markets[market] = {"enabled": True, "selector": selector_defaults.copy()}
    return {
        "markets": markets,
        "pools": pool_rules,
        "updated_at": datetime.utcnow().isoformat(),
    }


def _merge_market_data_config(raw: dict) -> dict:
    cfg = _default_market_data_config()
    if not isinstance(raw, dict):
        return cfg
    for market, market_cfg in raw.get("markets", {}).items():
        base = cfg["markets"].get(market, {})
        merged_m = {**base, **market_cfg}
        # selector 必须逐个周期合并, 不能整层覆盖 —— 存过的 {"1d": "7d"} 会把
        # 另外 7 个周期的默认值一起抹掉, 前端在 1h 上就只剩「默认同步 -」。
        # pools 那条路早就这么做了, markets 这条漏了。
        merged_m["selector"] = {**base.get("selector", {}), **(market_cfg.get("selector") or {})}
        cfg["markets"][market] = merged_m
    for pool_name, pool_cfg in raw.get("pools", {}).items():
        merged = {**cfg["pools"].get(pool_name, {}), **pool_cfg}
        merged["selector"] = {**cfg["pools"].get(pool_name, {}).get("selector", {}), **pool_cfg.get("selector", {})}
        merged["optimizer"] = {**cfg["pools"].get(pool_name, {}).get("optimizer", {}), **pool_cfg.get("optimizer", {})}
        cfg["pools"][pool_name] = merged
    cfg["updated_at"] = raw.get("updated_at") or cfg["updated_at"]
    return cfg


def _load_market_data_config() -> dict:
    if MARKET_DATA_CONFIG_FILE.exists():
        try:
            raw = json.loads(MARKET_DATA_CONFIG_FILE.read_text(encoding="utf-8"))
            return _merge_market_data_config(raw)
        except Exception:
            pass
    cfg = _default_market_data_config()
    _save_market_data_config(cfg)
    return cfg


def _save_market_data_config(cfg: dict):
    cfg = _merge_market_data_config(cfg)
    cfg["updated_at"] = datetime.utcnow().isoformat()
    MARKET_DATA_CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _configured_market_codes() -> list:
    cfg = _load_market_data_config()
    codes = []
    seen = set()
    for market in CORE_MARKETS:
        seen.add(market)
        codes.append(market)
    for market in cfg.get("markets", {}).keys():
        market = str(market).upper().strip()
        if market and market not in seen:
            seen.add(market)
            codes.append(market)
    for info in _load_pools().values():
        market = str(info.get("market", "")).upper().strip()
        if market and market not in seen:
            seen.add(market)
            codes.append(market)
    return codes


def _market_universe_file(market: str) -> Path:
    return MARKET_UNIVERSE_DIR / f"{str(market).upper()}.json"


def _load_market_universe_excludes() -> dict:
    return _read_json_file(MARKET_UNIVERSE_EXCLUDES_FILE, {})


def _save_market_universe_excludes(data: dict):
    MARKET_UNIVERSE_EXCLUDES_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _market_universe_excluded_symbols(market: str) -> set:
    data = _load_market_universe_excludes()
    market_data = data.get(str(market).upper(), {})
    if isinstance(market_data, dict):
        return {str(sym).upper() for sym in market_data.keys()}
    if isinstance(market_data, list):
        return {str(sym).upper() for sym in market_data}
    return set()


def _mark_market_universe_excluded(market: str, symbol: str, reason: str):
    market = str(market).upper()
    symbol = str(symbol).upper()
    data = _load_market_universe_excludes()
    bucket = data.setdefault(market, {})
    if not isinstance(bucket, dict):
        bucket = {}
        data[market] = bucket
    bucket[symbol] = {
        "reason": str(reason)[:200],
        "updated_at": datetime.utcnow().isoformat(),
    }
    _save_market_universe_excludes(data)


_US_UNIVERSE_EXCLUDE_NAME_PATTERNS = [
    r"\bwarrant(s)?\b",
    r"\bright(s)?\b",
    r"\bunit(s)?\b",
    r"when issued",
    r"\bpreferred\b",
    r"depositary share",
    r"depository share",
    r"\bacquisition\b",
    r"\bspac\b",
    r"\bblank check\b",
    r"\betf\b",
    r"\betn\b",
    r"\bclosed[- ]end\b",
    r"\bindex fund\b",
    r"\bseries trust\b",
    r"\bbulletshares\b",
    r"\byieldmax\b",
    r"\bdirexion\b",
    r"\bproshares\b",
    r"\binvesco\b",
    r"\bishares\b",
    r"\bvanguard\b",
    r"\broundhill\b",
    r"\bgraniteshares\b",
    r"\bglobal x\b",
    r"\bwisdomtree\b",
    r"\bfirst trust\b",
    r"\bspdr\b",
    r"\bcalamos\b",
    r"\bsimplify\b",
]
_US_UNIVERSE_EXCLUDE_RE = re.compile("|".join(_US_UNIVERSE_EXCLUDE_NAME_PATTERNS), re.I)
_US_UNIVERSE_SYMBOL_EXCLUDE_RE = re.compile(r"[\^/=]")


def _normalize_us_symbol(sym_raw: str) -> str:
    sym_raw = str(sym_raw or "").strip().upper()
    if not sym_raw:
        return ""
    if "." in sym_raw:
        parts = [p for p in sym_raw.split(".") if p]
        if parts and parts[0].isdigit() and len(parts) > 1:
            return "".join(parts[1:]).upper()
        if parts and parts[0].isdigit():
            return ""
    if sym_raw[:1].isdigit() and "." in sym_raw:
        return sym_raw.split(".", 1)[-1].replace(".", "").upper()
    return sym_raw.replace(".", "").upper()


def _should_keep_us_universe_row(symbol: str, name: str) -> bool:
    sym = str(symbol or "").strip().upper()
    nm = str(name or "").strip()
    if not sym or len(sym) > 6 or _US_UNIVERSE_SYMBOL_EXCLUDE_RE.search(sym):
        return False
    if _US_UNIVERSE_EXCLUDE_RE.search(nm):
        return False
    return True


_CN_UNIVERSE_EXCLUDE_RE = re.compile(
    r"(etf|lof|reit|基金|债券|转债|分级|货币|债基|spac|acquisition)",
    re.I,
)
_HK_UNIVERSE_EXCLUDE_RE = re.compile(
    r"(etf|trust|fund|warrant|rights|unit|牛熊|认股证|基金|信托|杠杆|反向|spac|acquisition)",
    re.I,
)
_UK_UNIVERSE_EXCLUDE_RE = re.compile(
    r"(etf|etn|fund|trust|warrant|rights|unit|preferred|acquisition|spac|blank check|when issued|series)",
    re.I,
)


def _should_keep_cn_universe_row(symbol: str, name: str) -> bool:
    sym = str(symbol or "").strip().upper()
    nm = str(name or "").strip()
    if not re.fullmatch(r"\d{6}", sym):
        return False
    if _CN_UNIVERSE_EXCLUDE_RE.search(nm):
        return False
    return True


def _should_keep_hk_universe_row(symbol: str, name: str) -> bool:
    sym = str(symbol or "").strip().upper()
    nm = str(name or "").strip()
    digits = "".join(ch for ch in sym if ch.isdigit())
    if not digits:
        return False
    if _HK_UNIVERSE_EXCLUDE_RE.search(nm):
        return False
    return True


def _should_keep_uk_universe_row(symbol: str, name: str) -> bool:
    sym = str(symbol or "").strip().upper()
    nm = str(name or "").strip()
    if not sym or _US_UNIVERSE_SYMBOL_EXCLUDE_RE.search(sym):
        return False
    if sym.startswith("0") and ".L" not in sym and ".IL" not in sym:
        return False
    if _UK_UNIVERSE_EXCLUDE_RE.search(nm):
        return False
    return True


def _sanitize_market_universe_rows(market: str, rows: list) -> list:
    market = str(market).upper()
    excluded = _market_universe_excluded_symbols(market)
    cleaned = []
    seen = set()
    for item in rows or []:
        sym = str(item.get("symbol", "")).strip().upper()
        name = str(item.get("name", "")).strip()
        if market == "US":
            sym = _normalize_us_symbol(sym)
            if not _should_keep_us_universe_row(sym, name):
                continue
        elif market == "CN":
            if not _should_keep_cn_universe_row(sym, name):
                continue
        elif market == "HK":
            digits = "".join(ch for ch in sym if ch.isdigit())
            sym = digits.zfill(4) if digits else ""
            if not _should_keep_hk_universe_row(sym, name):
                continue
        elif market == "UK":
            if not _should_keep_uk_universe_row(sym, name):
                continue
        if not sym or sym in excluded or sym in seen:
            continue
        seen.add(sym)
        cleaned.append({"symbol": sym, "name": name})
    return cleaned


def _market_universe_needs_refresh(market: str, max_age_hours: int = 24) -> bool:
    fp = _market_universe_file(market)
    if not fp.exists():
        return True
    age_hours = (time.time() - fp.stat().st_mtime) / 3600
    return age_hours >= max_age_hours


def _run_with_timeout(fn, timeout: int, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeoutError:
        fut.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _extract_symbol_name_rows(df: pd.DataFrame, symbol_candidates: list, name_candidates: list) -> list:
    if df is None or df.empty:
        return []
    symbol_col = next((c for c in symbol_candidates if c in df.columns), None)
    if not symbol_col:
        return []
    name_col = next((c for c in name_candidates if c in df.columns), None)
    rows = []
    seen = set()
    for _, row in df.iterrows():
        sym_raw = str(row.get(symbol_col, "")).strip()
        if not sym_raw or sym_raw.lower() == "nan":
            continue
        sym = sym_raw.upper()
        if sym in seen:
            continue
        seen.add(sym)
        rows.append({
            "symbol": sym,
            "name": str(row.get(name_col, "")).strip() if name_col else "",
        })
    return rows


def _fetch_cn_market_universe() -> list:
    funcs = [
        lambda: ak.stock_info_a_code_name(),
        lambda: ak.stock_zh_a_spot_em(),
        lambda: ak.stock_zh_a_spot(),
    ]
    for fn in funcs:
        try:
            df = fn()
            rows = _extract_symbol_name_rows(
                df,
                ["code", "代码", "symbol", "证券代码"],
                ["name", "名称", "证券简称"],
            )
            if rows:
                return rows
        except Exception:
            continue
    return []


def _fetch_hk_market_universe() -> list:
    funcs = [
        lambda: ak.stock_hk_spot_em(),
        lambda: ak.stock_hk_spot(),
    ]
    for fn in funcs:
        try:
            df = fn()
            rows = _extract_symbol_name_rows(
                df,
                ["symbol", "代码", "代码代码", "code"],
                ["name", "名称"],
            )
            normalized = []
            seen = set()
            for item in rows:
                digits = "".join(ch for ch in item["symbol"] if ch.isdigit())
                sym = digits.zfill(4) if digits else item["symbol"].upper()
                if sym in seen:
                    continue
                seen.add(sym)
                normalized.append({**item, "symbol": sym})
            if normalized:
                return normalized
        except Exception:
            continue
    return []


def _fetch_us_market_universe() -> list:
    funcs = [
        lambda: ak.stock_us_spot_em(),
        lambda: ak.stock_us_spot(),
    ]
    for fn in funcs:
        try:
            df = fn()
            rows = _extract_symbol_name_rows(
                df,
                ["symbol", "代码", "code"],
                ["name", "名称"],
            )
            normalized = []
            seen = set()
            for item in rows:
                sym = _normalize_us_symbol(item["symbol"])
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                normalized.append({**item, "symbol": sym})
            if normalized:
                return normalized
        except Exception:
            continue
    return []


def _fetch_uk_market_universe() -> list:
    try:
        from yfinance import EquityQuery
        q = EquityQuery("eq", ["region", "gb"])
        offset = 0
        size = 250
        total = None
        rows = []
        seen = set()
        while total is None or offset < total:
            resp = yf.screen(q, size=size, offset=offset, sortField="ticker", sortAsc=True)
            quotes = resp.get("quotes", []) or []
            total = int(resp.get("total") or len(quotes))
            if not quotes:
                break
            for qd in quotes:
                sym = str(qd.get("symbol", "")).strip().upper()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                rows.append({
                    "symbol": sym,
                    "name": qd.get("longName") or qd.get("shortName") or sym,
                })
            offset += len(quotes)
            if len(quotes) < size:
                break
        return rows
    except Exception:
        return []


def _fetch_market_universe(market: str) -> dict:
    market = str(market).upper().strip()
    fetchers = {
        "CN": _fetch_cn_market_universe,
        "HK": _fetch_hk_market_universe,
        "US": _fetch_us_market_universe,
        "UK": _fetch_uk_market_universe,
    }
    rows = fetchers.get(market, lambda: [])()
    rows = _sanitize_market_universe_rows(market, rows)
    rows = sorted(rows, key=lambda x: x["symbol"])
    return {
        "market": market,
        "symbols": rows,
        "count": len(rows),
        "updated_at": datetime.utcnow().isoformat(),
    }


def _load_market_universe(market: str, refresh: bool = False) -> dict:
    market = str(market).upper().strip()
    fp = _market_universe_file(market)
    if not refresh and fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data.get("symbols"), list):
                sanitized = _sanitize_market_universe_rows(market, data.get("symbols", []))
                if len(sanitized) != len(data.get("symbols", [])):
                    data = {
                        **data,
                        "market": market,
                        "symbols": sanitized,
                        "count": len(sanitized),
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return data
        except Exception:
            pass
    data = _fetch_market_universe(market)
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

app = FastAPI(title="QuantAI Data API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
class _NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    """HTML 文件禁止浏览器强缓存，防止部署后用户拿到旧 JS"""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        if "text/html" in ct:
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response
app.add_middleware(_NoCacheHTMLMiddleware)


@app.on_event("startup")
async def startup_init_db():
    """初始化本地 PostgreSQL 表结构。"""
    init_db()
    logger.info("DB schema initialized")

@app.on_event("startup")
async def startup_sync_warehouse():
    """服务启动时同步仓库状态：将已归档策略从 REGISTRY 移到 WAREHOUSE"""
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE
    wh = _load_warehouse()
    archived_names = {a["name"] for a in wh.get("archived", [])}
    moved = []
    for name in list(archived_names):
        if name in STRATEGY_REGISTRY:
            STRATEGY_WAREHOUSE[name] = STRATEGY_REGISTRY.pop(name)
            moved.append(name)
    if moved:
        logger.info(f"📦 启动同步: {len(moved)} 个归档策略已从 REGISTRY 移至 WAREHOUSE: {moved}")
    logger.info(f"📊 活跃策略: {len(STRATEGY_REGISTRY)} | 归档: {len(STRATEGY_WAREHOUSE)}")

@app.on_event("startup")
async def startup_scan_audit_durability():
    """v0.5 P12 §7.5: recover any orphaned scan_audit segments and start the
    background flusher BEFORE the monitor daemon fires its first tick.

    Also replays the most-recent N segments from disk into the in-memory
    audit ring so the scan-monitor UI is non-empty immediately after a
    restart (otherwise users see "0 stocks" for ~5 min until the first
    scan tick repopulates the ring — the ring is process-local and dies
    with the worker).
    """
    try:
        import scan_audit as _scan_audit
        cleaned = _scan_audit.recover_audit_segments()
        if cleaned:
            logger.info(
                "scan_audit: recovered/cleaned %d orphan segment file(s)",
                cleaned,
            )
        # Backfill ring from disk so /scan/monitor/cycles is non-empty
        # at startup. Cap at 50 cycles (typically ~5,000 records) so we
        # leave headroom in the 10k ring for live cycles.
        try:
            replayed = _scan_audit.replay_recent_segments_to_ring(max_cycles=50)
            if replayed:
                logger.info(
                    "scan_audit: backfilled %d audit record(s) from disk into ring",
                    replayed,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan_audit: ring backfill failed: %r", exc)
        _scan_audit.start_audit_flusher()
        logger.info("scan_audit: background flusher started")
    except Exception as exc:  # noqa: BLE001 — never block startup on audit path
        logger.warning("scan_audit: startup wiring failed: %r", exc)


@app.on_event("startup")
async def startup_scan_daemon():
    started = start_scan_daemon()
    logger.info("scan daemon %s", "started" if started else "already running")

@app.on_event("startup")
async def startup_news_auto_scanner():
    """2026-04-27: news auto-scan scheduler — periodically fires /selector/jobs/start
    so scan_signals.json keeps accumulating data. Without this, "本周报告" can
    never get 7 days of data because rolling buffer is forward-only."""
    try:
        from news_auto_scanner import start_auto_scanner
        started = start_auto_scanner()
        logger.info("news_auto_scanner %s", "started" if started else "already running")
    except Exception as e:
        logger.warning(f"news_auto_scanner failed to start: {e}")

@app.on_event("shutdown")
async def shutdown_news_auto_scanner():
    try:
        from news_auto_scanner import stop_auto_scanner
        stop_auto_scanner()
    except Exception:
        pass

@app.on_event("startup")
async def startup_update_hot_stocks():
    """服务启动时检查热门股是否需要更新（超过12小时才重新拉取）"""
    cache = _load_hot_cache()
    need_update = True
    if cache and "updated_at" in cache:
        try:
            last = datetime.fromisoformat(cache["updated_at"])
            if (datetime.now() - last).total_seconds() < 12 * 3600:
                need_update = False
                logger.info(f"🔥 Hot stocks cache still fresh ({len(cache.get('symbols', []))} symbols)")
        except Exception:
            pass
    if need_update:
        # 后台线程更新，不阻塞启动
        threading.Thread(target=_update_hot_pool, daemon=True).start()

_cache: dict = {}
_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _data_mtime_sig(symbols: list, market: str, interval: str = "1d") -> str:
    """
    计算股票池数据文件的最新修改时间签名。
    数据有更新时签名改变，触发缓存失效——避免旧结果被复用。
    若无任何文件存在，返回 '0'（首次下载后自动失效）。
    """
    max_mtime = 0
    for sym in symbols:
        p = _get_data_path(sym, market, interval)
        if p.exists():
            max_mtime = max(max_mtime, int(p.stat().st_mtime))
    # 精度取整到分钟（避免同一次运行内多次 stat 产生细微差异）
    return str(max_mtime // 60)

def _hash_symbols(symbols: list) -> str:
    """计算股票列表的短 hash（排序后 MD5 前8位），用于缓存 key"""
    return hashlib.md5(','.join(sorted(symbols)).encode()).hexdigest()[:8]

def _cache_file(key: str) -> Path:
    """将cache key转为安全的文件名"""
    safe = key.replace(":", "_").replace("/", "_").replace(" ", "_")
    return _CACHE_DIR / f"{safe}.json"

def cache_get(key: str, ttl: int = 60):
    # 1. 内存缓存
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < ttl:
            return data
    # 2. 文件缓存（用于重启后恢复）
    cf = _cache_file(key)
    if cf.exists():
        try:
            age = time.time() - cf.stat().st_mtime
            if age < ttl:
                data = json.load(open(cf, encoding="utf-8"))
                _cache[key] = (data, cf.stat().st_mtime)
                return data
        except Exception:
            pass
    return None

def stable_hash(obj) -> str:
    """跨进程稳定的短摘要。

    内置 hash() 对 str 是加了 PYTHONHASHSEED 随机化的, 每次启动都不一样 ——
    拿它拼磁盘缓存 key, 重启之后同样的参数会算出不同的文件名, 永远命中不了,
    老文件还一直堆在 .cache 里等 7 天过期。
    """
    return hashlib.md5(str(obj).encode("utf-8")).hexdigest()[:16]


def _json_default(o):
    """json.dump 的兜底转换。

    回测结果里 chart_data/indicators 的 date 是 pandas.Timestamp,
    _clean 只处理带 .item() 的 numpy 标量, 漏掉了它们。
    以前 json.dump 撞上就抛 TypeError, 而它是**边写边抛**的 ——
    异常被 except: pass 吞掉, 磁盘上留下一个从第一个日期处截断的半截文件。
    实测 .cache 里 52/57 个 compare_*.json 和 128/128 个 bt_*.json 都是这么坏的。
    """
    try:
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "item"):
            return o.item()
    except Exception:
        pass
    return str(o)


def _json_sanitize(o, _depth: int = 0):
    """json.dumps 兜不住的两类东西, 在这儿统一擦干净。只在真撞上时才走一趟。

    1. NaN / ±Infinity: 换成 None (指标算不出来, 前端本来就按缺失处理)
    2. 非法 key 类型: json 只认 str/int/float/bool/None, 而 np.int64 / tuple /
       date 做 key 都会 TypeError。旧的 to_jsonable 对每个 key 做过 str(k),
       换成 json.dumps 之后那层保护没了, 这里补回来。
    """
    if _depth > 12:
        return o
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            out[k] = _json_sanitize(v, _depth + 1)
        return out
    if isinstance(o, (list, tuple)):
        return [_json_sanitize(v, _depth + 1) for v in o]
    return o


def job_json_response(job, drop=None):
    """后台任务进度/结果的**统一响应出口** —— 9 个轮询端点都该走这里。

    为什么不是各端点自己 `return JSONResponse(dict(job))`:

    1. numpy 标量。jsonable_encoder 认得 np.float64, 却对 np.int64 / np.bool_
       直接抛 ValueError → HTTP 500。逐个端点各包一层转换意味着以后每加一个
       端点都得记得包一次, 忘了就又是一个只在"恰好出现 np.int64"时才现形的 500。
       这里返回的是 Response 实例, FastAPI 原样透传、**不再过 jsonable_encoder**,
       numpy 由 _json_default 一次性兜住, 根上解决。

    2. 速度。原先一次轮询要把 payload 走四遍 (snapshot → 逐端点的类型转换 →
       jsonable_encoder → json.dumps)。FastAPI 对返回 Response 实例的路径会
       短路掉 serialize_response/jsonable_encoder, 所以现在只有 snapshot +
       dumps 两遍。20000 条 details 实测 244ms → 44ms, 省下的全是**事件循环**
       上的时间 (轮询端点都是 async def)。

    drop: 不下发的字段名集合。进度轮询只看计数器, 没必要每 1.5 秒把
          上万条 details / 1.4MB strategy_details 推一遍再让前端丢掉。
    """
    # 顺序要紧: 先看要丢什么, 再决定拷什么。
    # 原来是 snapshot_of(job) 整份深拷之后再 pop —— 等于把马上要扔掉的
    # details 一条不落地复制了一遍。20000 条 details 实测白烧 28.9ms,
    # 而且是在**事件循环**上(轮询端点都是 async def), 每 1.5 秒一次。
    if drop:
        counts = {}
        keep = {}
        for k, v in (job.items() if hasattr(job, "items") else []):
            if k in drop:
                # 别静默丢信息: 留一个计数, 面板要用随时能看出来。
                # len() 不需要深拷, 拿的是活对象的长度 —— 差一两条无所谓。
                try:
                    if isinstance(v, (dict, list, tuple, set)):
                        counts[f"{k}_count"] = len(v)
                except Exception:
                    pass
                continue
            keep[k] = v
        snap = {k: _ld_detach(v) for k, v in keep.items()}
        snap.update(counts)
    else:
        snap = snapshot_of(job)
    # allow_nan=False: 默认 True 会把 NaN/Infinity 原样写进去, 那是**非法 JSON** ——
    # HTTP 还是 200, 但浏览器 JSON.parse 直接抛 SyntaxError, 前端轮询落进
    # 不 clearInterval 的 catch 分支, 任务永远不会被判定为完成, 面板一直转。
    # (旧的 jsonable_encoder 路径在这里抛 ValueError → 500, 前端 `if (!resp.ok)`
    #  能兜住 —— 所以这次改动如果不管, 是把一个响亮的错误换成了静默的死循环。)
    # 回测里 max_drawdown 在 equity 为空时就会是 NaN。
    try:
        body = json.dumps(snap, ensure_ascii=False, default=_json_default,
                          allow_nan=False)
    except (ValueError, TypeError):
        # ValueError = 撞上 NaN/Inf; TypeError = 撞上非法 key 类型。
        # 都只在真出问题时才多走一趟, 快路径不受影响。
        body = json.dumps(_json_sanitize(snap), ensure_ascii=False,
                          default=_json_default, allow_nan=False)
    return Response(content=body, media_type="application/json")


def cache_set(key: str, data):
    _cache[key] = (data, time.time())
    # 写入文件缓存（仅对eval等重量级缓存）
    # 原子写: 先写临时文件再 replace, 免得写一半被打断(异常/进程被 kill)
    # 就地留下半截文件, 之后每次读都 JSONDecodeError → 缓存永久失效。
    tmp = None
    try:
        cf = _cache_file(key)
        fd, tmp = tempfile.mkstemp(dir=str(Path(cf).parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=_json_default)
        os.replace(tmp, cf)
        tmp = None
    except Exception as e:
        logger.warning(f"[cache] 写文件缓存失败 key={key[:60]}: {e}")
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass

def cache_invalidate_symbol(symbol: str, market: str):
    """清除指定股票的所有回测缓存（内存+文件），供刷新行情后调用"""
    prefix = f"bt:{market.upper()}:{symbol.upper()}:"
    # 内存缓存
    keys_to_remove = [k for k in _cache if k.startswith(prefix)]
    for k in keys_to_remove:
        del _cache[k]
    # 文件缓存（key中的 : 被替换为 _）
    file_prefix = prefix.replace(":", "_")
    try:
        for f in _CACHE_DIR.iterdir():
            if f.name.startswith(file_prefix) and f.is_file():
                f.unlink(missing_ok=True)
    except Exception:
        pass


def _cache_cleanup(max_age_days: int = 7):
    """启动时清理超龄缓存文件"""
    cutoff = time.time() - max_age_days * 86400
    try:
        for f in _CACHE_DIR.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass

# 启动时清理旧缓存
_cache_cleanup()

def safe_float(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except:
        return None


def composite_score(total_return, sharpe, max_drawdown, win_rate, profit_factor):
    """综合评分（偏重收益）: 收益40% + Sharpe25% + 回撤20% + 胜率10% + PF5%"""
    norm_ret = float(np.clip((total_return + 50) / 250, 0, 1))
    norm_sharpe = float(np.clip((sharpe + 1) / 4, 0, 1))
    norm_dd = float(np.clip(1 - max_drawdown / 60, 0, 1))
    norm_wr = float(np.clip(win_rate / 100, 0, 1))
    norm_pf = float(np.clip(profit_factor / 5, 0, 1))
    return round(0.40 * norm_ret + 0.25 * norm_sharpe + 0.20 * norm_dd
                 + 0.10 * norm_wr + 0.05 * norm_pf, 4)

def yf_symbol(symbol: str, market: str) -> str:
    m = market.upper()
    if m == "CN":
        return f"{symbol}.SS" if symbol.startswith("6") else f"{symbol}.SZ"
    if m == "HK":
        return f"{symbol.zfill(4)}.HK"
    if m == "UK":
        return symbol.upper() if "." in symbol else f"{symbol.upper()}.L"
    return symbol.upper()

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ── Router registration ──
from routers import market as _r_market, pools as _r_pools, strategies as _r_strategies
from routers import data as _r_data, trading as _r_trading, scan as _r_scan
from routers import llm_admin as _r_llm_admin, combos as _r_combos
from routers import eval_chain as _r_eval_chain
from routers import scan_monitor as _r_scan_monitor
from routers import opslog as _r_opslog

app.include_router(_r_market.router)
app.include_router(_r_pools.router)
app.include_router(_r_strategies.router)
app.include_router(_r_data.router)
app.include_router(_r_trading.router)
app.include_router(_r_scan.router)
app.include_router(_r_llm_admin.router)
app.include_router(_r_combos.router)
app.include_router(_r_eval_chain.router)
app.include_router(_r_scan_monitor.router)
app.include_router(_r_opslog.router)

# ── 量化交易池 (Phase 1) ───────────────────────────────────────
from routers import quant as _r_quant  # noqa: E402
app.include_router(_r_quant.router)

from routers import stock_report as _r_stock_report  # noqa: E402
app.include_router(_r_stock_report.router)


@app.on_event("startup")
async def startup_quant_daemon():
    """量化交易引擎 daemon: 每 60s tick 所有 enabled 池, 自动开/平仓."""
    import quant_engine as _qe
    started = _qe.start_daemon()
    logger.info(f"quant-engine daemon {'started' if started else 'already running'}")


@app.on_event("startup")
async def startup_resume_validation_jobs():
    """恢复被 uvicorn --reload 中断的新策略验证任务.
    幂等: 已有 registry result 的策略会被跳过."""
    try:
        _resume_validation_jobs_on_startup()
    except Exception as e:
        logger.warning(f"resume validation jobs failed: {e}")


# ── Unified ops health (2026-07-06) ─────────────────────────────────────────
# One endpoint that aggregates every subsystem's health so ops (and the
# health_watchdog launchd timer) don't have to poll four different URLs.
@app.get("/ops/health")
def ops_health(request: Request):

    checks: dict = {}
    worst = "green"

    def _worse(level: str) -> None:
        nonlocal worst
        order = {"green": 0, "yellow": 1, "red": 2}
        if order[level] > order[worst]:
            worst = level

    # 1) IB Gateway pool — yellow when down (markets may simply be closed /
    #    IB weekend maintenance; quote path degrades to Yahoo, not an outage).
    #    Uses ib_status_light(): the full ib_status() re-attempts pool
    #    creation and can block 30-60s while IB is offline, which made this
    #    very health endpoint time out.
    try:
        from ib_gateway import ib_status_light as _ibs
        s = _ibs()
        ib_ok = bool(s.get("connected"))
        checks["ib"] = {
            "status": "green" if ib_ok else "yellow",
            "connected": ib_ok,
            "gateway_listening": s.get("gateway_listening"),
            "pool": f"{s.get('pool_size_actual', 0)}/{s.get('pool_size_target', 0)}",
            "streaming_slots": s.get("streaming_slots_reserved", 0),
        }
        if not ib_ok:
            _worse("yellow")
    except Exception as exc:
        checks["ib"] = {"status": "yellow", "error": str(exc)[:200]}
        _worse("yellow")

    # 2) Backtest queue + workers
    try:
        from backtest_queue import queue_stats as _bq_stats
        qs = _bq_stats()
        stuck = qs["by_status"].get("processing", 0)
        checks["backtest_queue"] = {
            "status": "green",
            "by_status": qs["by_status"],
            "online_workers": qs["online_workers"],
        }
    except Exception as exc:
        checks["backtest_queue"] = {"status": "yellow", "error": str(exc)[:200]}
        _worse("yellow")

    # 3b) Process file-descriptor pressure. An FD leak (2026-07-06: IB slot
    #     loops never closed, amplified by the pool-maintenance thread) can
    #     silently hang the whole app with Errno 24. Catch it early: yellow at
    #     70% of the soft limit, red at 90%.
    try:
        import resource
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        open_fds = len(os.listdir(f"/dev/fd")) if os.path.isdir("/dev/fd") else -1
        if soft > 0 and open_fds >= 0:
            frac = open_fds / soft
            level = "red" if frac >= 0.9 else "yellow" if frac >= 0.7 else "green"
            checks["file_descriptors"] = {
                "status": level, "open": open_fds, "soft_limit": soft,
                "pct": round(frac * 100, 1),
            }
            _worse(level)
    except Exception as exc:
        checks["file_descriptors"] = {"status": "green", "note": str(exc)[:120]}

    # 4) scan_audit flush health — cumulative counters since process start.
    try:
        from scan_audit import _audit_flush_stats
        dropped = _audit_flush_stats.get("dropped_cycles", 0)
        checks["scan_audit"] = {
            "status": "yellow" if dropped else "green",
            **{k: v for k, v in _audit_flush_stats.items()},
        }
        if dropped:
            _worse("yellow")
    except Exception as exc:
        checks["scan_audit"] = {"status": "yellow", "error": str(exc)[:200]}

    return {"status": worst, "checks": checks,
            "ts": datetime.utcnow().isoformat() + "Z"}


# ── Internal OHLCV proxy (background backtest workers use this) ────────────
# Backtest worker processes fetch OHLCV from the main app over HTTP instead of
# opening their own IB Gateway connections. Returns a pandas DataFrame as parquet
# bytes (≈10× smaller than JSON for numeric data).
#
# Auth: X-Aitrader-Internal-Token header matched against AITRADER_INTERNAL_SECRET
# (constant-time compare). Returns 401 if missing/wrong. The app binds to
# 127.0.0.1, so only workers on this machine can reach it.
# Whitelist matches the backtest.get_ohlcv accepted values; rejecting
# malformed input early avoids crash-stacks bubbling up to workers.
_OHLCV_ALLOWED_PERIODS = frozenset({"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"})
_OHLCV_ALLOWED_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "60m", "4h", "1d", "1wk", "1mo"})
_OHLCV_ALLOWED_MARKETS = frozenset({"US", "HK", "CN", "UK", "CRYPTO"})


@app.get("/internal/ohlcv")
def internal_ohlcv(
    request: Request,
    symbol: str,
    market: str = "US",
    interval: str = "1d",
    period: str = "2y",
):
    import hmac
    import io
    expected = os.environ.get("AITRADER_INTERNAL_SECRET", "")
    if not expected:
        raise HTTPException(status_code=500, detail="AITRADER_INTERNAL_SECRET not configured")
    provided = request.headers.get("X-Aitrader-Internal-Token", "")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="invalid internal token")

    # Input validation — fail fast with HTTP 422 rather than crashing
    # downstream in yfinance/akshare.
    market_u = (market or "").upper()
    if market_u not in _OHLCV_ALLOWED_MARKETS:
        raise HTTPException(status_code=422, detail=f"market must be one of {sorted(_OHLCV_ALLOWED_MARKETS)}")
    if interval not in _OHLCV_ALLOWED_INTERVALS:
        raise HTTPException(status_code=422, detail=f"interval must be one of {sorted(_OHLCV_ALLOWED_INTERVALS)}")
    if period not in _OHLCV_ALLOWED_PERIODS:
        raise HTTPException(status_code=422, detail=f"period must be one of {sorted(_OHLCV_ALLOWED_PERIODS)}")
    symbol_u = (symbol or "").strip().upper()
    if not symbol_u or len(symbol_u) > 16:
        raise HTTPException(status_code=422, detail="symbol must be 1–16 chars after upper()")

    # Reuse the backtest get_ohlcv path so we get whatever IB/yfinance/akshare
    # mix is currently best for this (symbol, market). It hits ib_gateway when
    # available, yfinance otherwise — same code path as the in-process backtest.
    from backtest import get_ohlcv
    df = get_ohlcv(symbol_u, market=market_u, interval=interval, period=period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"no data for {market}/{symbol}")

    buf = io.BytesIO()
    df.to_parquet(buf, compression="snappy")
    buf.seek(0)
    from fastapi.responses import Response
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"X-Aitrader-Ohlcv-Rows": str(len(df))},
    )


# ── Backtest queue stats ───────────────────────────────────────────────────
# Lightweight dashboard: queue depth by status, online workers, last heartbeat.
@app.get("/backtest/queue/stats")
def backtest_queue_stats():
    try:
        from backtest_queue import queue_stats, list_workers
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"backtest_queue import failed: {exc}")
    return {
        "queue": queue_stats(),
        "workers": list_workers(include_offline=False),
    }


@app.get("/backtest/queue/recent")
def backtest_queue_recent(limit: int = 50, status: Optional[str] = None):
    try:
        from backtest_queue import list_recent
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"backtest_queue import failed: {exc}")
    limit = max(1, min(500, int(limit)))
    return list_recent(limit=limit, status=status)


# ── Live streaming token issuer ────────────────────────────────────────────
# Returns LIVE_STREAMING_TOKEN so the dashboard SPA can connect to /ws/*
# without prompting. The app binds to 127.0.0.1 only.
@app.get("/live/config")
def live_config():
    tok = os.environ.get("LIVE_STREAMING_TOKEN", "")
    if not tok:
        raise HTTPException(status_code=503, detail="streaming not configured")
    return {
        "token": tok,
        "enabled": os.environ.get("LIVE_STREAMING_ENABLED", "0").lower() not in ("0","false","no","off",""),
    }


# ── Live streaming (WS) ────────────────────────────────────────────────────
# Feature-flagged. When LIVE_STREAMING_ENABLED=1, mount the streaming
# WS routes and bind the SubscriptionManager / IBStreamBridge in startup.
# See 行情Live功能方案.md.
_LIVE_STREAMING_ENABLED = os.environ.get("LIVE_STREAMING_ENABLED", "0").lower() not in (
    "0", "false", "no", "off", "",
)
if _LIVE_STREAMING_ENABLED:
    try:
        from streaming import ws_routes as _r_ws
        app.include_router(_r_ws.router)

        @app.on_event("startup")
        async def _startup_live_streaming():
            try:
                _r_ws.init_streaming()
                logger.info("[live-streaming] WS routes mounted + SubscriptionManager started")
            except Exception:
                logger.exception("[live-streaming] init failed — WS routes mounted but will error")

        @app.on_event("shutdown")
        async def _shutdown_live_streaming():
            try:
                await _r_ws.shutdown_streaming()
            except Exception:
                logger.exception("[live-streaming] shutdown failed")
    except Exception:
        # If streaming module fails to import, log and continue — the rest
        # of the app keeps working.
        logging.getLogger(__name__).exception(
            "[live-streaming] failed to import streaming module — feature disabled")

def _clean(obj):
    """清理 numpy 类型，避免 JSON 序列化报错"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


def _load_warehouse():
    wf = Path(__file__).parent / "strategy_warehouse.json"
    if wf.exists():
        return json.load(open(wf, encoding="utf-8"))
    return {"favourites": [], "active": [], "archived": [], "deleted": [], "history": []}

def _save_warehouse(data):
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d")
    wf = Path(__file__).parent / "strategy_warehouse.json"
    with open(wf, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _default_strategy_llm_registry():
    return {
        "description": "策略归属到各 LLM 团队的元数据注册表。未显式声明的策略默认归属 Claude。",
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
        "default_owner": DEFAULT_STRATEGY_LLM_OWNER,
        "supported_owners": SUPPORTED_STRATEGY_LLM_OWNERS,
        "owners": {},
        "guide_path": "docs/strategy_llm_authoring_guide.md",
    }


def _normalize_strategy_owner(owner: Optional[str]) -> str:
    if not owner:
        return DEFAULT_STRATEGY_LLM_OWNER
    owner = str(owner).strip()
    for candidate in SUPPORTED_STRATEGY_LLM_OWNERS:
        if candidate.lower() == owner.lower():
            return candidate
    return DEFAULT_STRATEGY_LLM_OWNER


def _load_strategy_llm_registry():
    if STRATEGY_LLM_REGISTRY_FILE.exists():
        try:
            data = json.load(open(STRATEGY_LLM_REGISTRY_FILE, encoding="utf-8"))
        except Exception:
            data = _default_strategy_llm_registry()
    else:
        data = _default_strategy_llm_registry()

    merged = _default_strategy_llm_registry()
    merged.update({k: v for k, v in data.items() if k != "owners"})
    owners = data.get("owners", {}) if isinstance(data.get("owners", {}), dict) else {}
    merged["supported_owners"] = SUPPORTED_STRATEGY_LLM_OWNERS
    merged["default_owner"] = _normalize_strategy_owner(merged.get("default_owner"))
    merged["owners"] = {str(k): _normalize_strategy_owner(v) for k, v in owners.items()}
    return merged


def _save_strategy_llm_registry(data):
    payload = _default_strategy_llm_registry()
    payload.update({k: v for k, v in (data or {}).items() if k != "owners"})
    payload["supported_owners"] = SUPPORTED_STRATEGY_LLM_OWNERS
    payload["default_owner"] = _normalize_strategy_owner(payload.get("default_owner"))
    payload["owners"] = {
        str(k): _normalize_strategy_owner(v)
        for k, v in (data or {}).get("owners", {}).items()
        if str(k).strip()
    }
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d")
    with open(STRATEGY_LLM_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _get_strategy_llm_owner(name: str, registry: Optional[dict] = None) -> str:
    registry = registry or _load_strategy_llm_registry()
    return _normalize_strategy_owner((registry.get("owners") or {}).get(name) or registry.get("default_owner"))


def _default_strategy_validation_state():
    return {
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
        "strategies": {},
    }


def _load_strategy_validation_state():
    raw = _read_json_file(STRATEGY_VALIDATION_FILE, {})
    merged = _default_strategy_validation_state()
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if k != "strategies"})
        merged["strategies"] = raw.get("strategies", {}) if isinstance(raw.get("strategies", {}), dict) else {}
    return merged


def _save_strategy_validation_state(data):
    payload = _default_strategy_validation_state()
    if isinstance(data, dict):
        payload.update({k: v for k, v in data.items() if k != "strategies"})
        payload["strategies"] = data.get("strategies", {}) if isinstance(data.get("strategies", {}), dict) else {}
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d")
    _write_json_file(STRATEGY_VALIDATION_FILE, payload)


def _set_strategy_validation_result(strategy_name: str, owner: str, interval: str, payload: dict):
    state = _load_strategy_validation_state()
    strategies = state.setdefault("strategies", {})
    item = strategies.setdefault(strategy_name, {"owner": _normalize_strategy_owner(owner), "intervals": {}})
    item["owner"] = _normalize_strategy_owner(owner)
    item.setdefault("intervals", {})[_normalize_param_interval(interval)] = payload
    _save_strategy_validation_state(state)


def _get_strategy_validation_summary(owner: Optional[str] = None, interval: str = "1d") -> dict:
    state = _load_strategy_validation_state()
    owner = _normalize_strategy_owner(owner) if owner else None
    interval = _normalize_param_interval(interval)
    result = {}
    for sname, info in (state.get("strategies") or {}).items():
        item_owner = _normalize_strategy_owner(info.get("owner"))
        if owner and item_owner != owner:
            continue
        interval_data = (info.get("intervals") or {}).get(interval)
        if interval_data:
            result[sname] = {"owner": item_owner, **interval_data}
    return result


def _ensure_strategy_new_candidate(name: str, reason: str = "平台入库新策略"):
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE

    wh = _load_warehouse()
    changed = False
    if name in STRATEGY_REGISTRY and name not in STRATEGY_WAREHOUSE:
        STRATEGY_WAREHOUSE[name] = STRATEGY_REGISTRY.pop(name)
        changed = True

    if name in wh.get("active", []):
        wh["active"] = [s for s in wh.get("active", []) if s != name]
        changed = True

    archived_before = len(wh.get("archived", []))
    wh["archived"] = [a for a in wh.get("archived", []) if a.get("name") != name]
    if len(wh["archived"]) != archived_before:
        changed = True

    deleted_before = len(wh.get("deleted", []))
    wh["deleted"] = [
        d for d in wh.get("deleted", [])
        if (d.get("name") if isinstance(d, dict) else d) != name
    ]
    if len(wh["deleted"]) != deleted_before:
        changed = True

    new_list = wh.setdefault("new_strategies", [])
    if name not in new_list:
        new_list.append(name)
        changed = True

    if changed:
        wh.setdefault("history", []).append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "action": "new_candidate",
            "strategies": [name],
            "reason": reason,
        })
        _save_warehouse(wh)


def _mark_strategy_archived_state(name: str, reason: str, extra: Optional[dict] = None):
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE

    wh = _load_warehouse()
    if name in STRATEGY_REGISTRY and name not in STRATEGY_WAREHOUSE:
        STRATEGY_WAREHOUSE[name] = STRATEGY_REGISTRY.pop(name)

    wh["active"] = [s for s in wh.get("active", []) if s != name]
    wh["new_strategies"] = [s for s in wh.get("new_strategies", []) if s != name]
    wh["deleted"] = [
        d for d in wh.get("deleted", [])
        if (d.get("name") if isinstance(d, dict) else d) != name
    ]
    wh["archived"] = [a for a in wh.get("archived", []) if a.get("name") != name]

    record = {
        "name": name,
        "archived_at": datetime.now().strftime("%Y-%m-%d"),
        "reason": reason,
    }
    if extra:
        record.update(extra)
    wh.setdefault("archived", []).append(record)
    wh.setdefault("history", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "archived",
        "strategies": [name],
        "reason": reason,
    })
    _save_warehouse(wh)


def _classify_strategy_status(name: str, wh: dict) -> str:
    """Map strategy name → status string by exclusion: deleted > active > archived > new.
    Single source of truth for the eval ranking + cache-update + frontend mirror.
    """
    if name in {(d.get("name") if isinstance(d, dict) else d) for d in wh.get("deleted", [])}:
        return "deleted"
    if name in set(wh.get("active", [])):
        return "active"
    archived_names = {a.get("name") for a in wh.get("archived", []) if isinstance(a, dict)}
    if name in archived_names:
        return "archived"
    return "new"


def _remove_universal_strategy_entry(strategy_name: str, interval: str = "1d") -> bool:
    raw = _load_universal_params()
    interval = _normalize_param_interval(interval)
    changed = False

    if isinstance(raw, dict) and "intervals" in raw:
        intervals = raw.get("intervals", {})
        layer = intervals.get(interval)
        if isinstance(layer, dict):
            target = layer.get("strategies") if isinstance(layer.get("strategies"), dict) else layer
            if isinstance(target, dict) and strategy_name in target:
                target.pop(strategy_name, None)
                changed = True
    elif isinstance(raw, dict) and strategy_name in raw:
        raw.pop(strategy_name, None)
        changed = True

    if changed:
        _save_universal_params(raw)
    return changed


def _list_new_strategy_candidates(owner: Optional[str] = None) -> list[str]:
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE

    owner = _normalize_strategy_owner(owner) if owner else None
    wh = _load_warehouse()
    active = set(wh.get("active", []))
    archived = {a.get("name") for a in wh.get("archived", []) if isinstance(a, dict)}
    deleted = {
        d.get("name") if isinstance(d, dict) else d
        for d in wh.get("deleted", [])
    }
    registry = _load_strategy_llm_registry()
    candidates = []
    all_known = set(STRATEGY_REGISTRY.keys()) | set(STRATEGY_WAREHOUSE.keys()) | set(wh.get("new_strategies", []))
    for name in all_known:
        if name in active or name in archived or name in deleted:
            continue
        item_owner = _get_strategy_llm_owner(name, registry)
        if owner and item_owner != owner:
            continue
        candidates.append(name)
    return sorted(candidates)


def _representative_benchmark_symbols(limit: int = 6) -> list[str]:
    pools = _load_pools()
    nasdaq = pools.get(UNIVERSAL_BENCHMARK_POOL, {})
    symbols = [str(s).upper() for s in nasdaq.get("symbols", []) if str(s).strip()]
    preferred = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "NFLX"]
    ordered = [s for s in preferred if s in symbols]
    ordered.extend([s for s in symbols if s not in ordered])
    return ordered[:max(1, limit)]


def _pool_avg_standard_return(strategy_name: str, pool_name: str = "MYPOOL",
                              period: str = "1y", market: str = "US",
                              dynamic_pos: bool = False, interval: str = "1d",
                              executor_mode: str = "thread") -> dict:
    interval = _normalize_param_interval(interval)
    pools = _load_pools()
    pool = pools.get(pool_name)
    if not pool:
        return {"avg_return": None, "count": 0, "pool": pool_name, "error": "pool not found"}
    symbols = [str(s).upper() for s in pool.get("symbols", []) if str(s).strip()]
    if not symbols:
        return {"avg_return": None, "count": 0, "pool": pool_name, "error": "empty pool"}

    tasks = [
        (str(sym).upper(), market, interval, period, strategy_name, dynamic_pos)
        for sym in symbols
    ]
    returns = _collect_pool_backtest_returns(tasks, _pool_standard_return_worker, executor_mode=executor_mode)

    avg_return = round(float(np.mean(returns)), 2) if returns else None
    return {"avg_return": avg_return, "count": len(returns), "pool": pool_name}


def _optimize_universal_candidate_strategy(strategy_name: str, interval: str = "1d",
                                           n_trials: int = 60, dynamic_pos: bool = False) -> dict:
    interval = _normalize_param_interval(interval)
    baseline = _pool_avg_return(
        strategy_name,
        {**getattr(get_strategy(strategy_name), "default_params", {})},
        pool_name=UNIVERSAL_BENCHMARK_POOL,
        period="1y",
        dynamic_pos=dynamic_pos,
        interval=interval,
        executor_mode="process",
    )
    benchmark_symbols = _representative_benchmark_symbols()
    attempts = []
    optimization_period = _optimization_period_for_interval(interval)
    raw_attempts = {}
    worker_tasks = [
        (sym, strategy_name, interval, optimization_period, n_trials, dynamic_pos)
        for sym in benchmark_symbols
    ]
    n_opt_workers = min(max(1, N_WORKERS // 2), len(worker_tasks))
    if n_opt_workers > 1 and _can_spawn_process_pool():
        with ProcessPoolExecutor(max_workers=n_opt_workers, initializer=proc_role.mark_pool_worker) as pool:
            futures = {pool.submit(_optimize_universal_benchmark_worker, task): task[0] for task in worker_tasks}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    raw_attempts[sym] = fut.result()
                except Exception as e:
                    raw_attempts[sym] = {"symbol": sym, "status": "error", "reason": str(e)[:160]}
    else:
        for task in worker_tasks:
            raw_attempts[task[0]] = _optimize_universal_benchmark_worker(task)

    for sym in benchmark_symbols:
        item = raw_attempts.get(sym) or {"symbol": sym, "status": "error", "reason": "missing result"}
        if item.get("status") != "ok":
            attempts.append(item)
            continue
        full_params = item.get("params") or {}
        promote = _promote_to_universal_if_better(
            sym,
            strategy_name,
            full_params,
            period="1y",
            dynamic_pos=dynamic_pos,
            interval=interval,
            executor_mode="process",
        )
        attempts.append({
            "symbol": sym,
            "status": "ok",
            "method": item.get("method"),
            "combos_tested": item.get("combos_tested"),
            "optimized_symbol_return": item.get("optimized_symbol_return"),
            "optimized_symbol_sharpe": item.get("optimized_symbol_sharpe"),
            "nasdaq_avg_return": promote.get("candidate_avg_return"),
            "promoted": bool(promote.get("updated")),
        })

    optimized_params = _get_universal_strategy_params(strategy_name, interval=interval)
    optimized_eval = _pool_avg_return(
        strategy_name,
        optimized_params,
        pool_name=UNIVERSAL_BENCHMARK_POOL,
        period="1y",
        dynamic_pos=dynamic_pos,
        interval=interval,
        executor_mode="process",
    )
    return {
        "baseline": baseline,
        "optimized": optimized_eval,
        "benchmark_symbols": benchmark_symbols,
        "attempts": attempts,
        "params": optimized_params,
    }


def _optimize_universal_benchmark_worker(args):
    from optimizer import optimize_strategy

    sym, strategy_name, interval, optimization_period, n_trials, dynamic_pos = args
    try:
        strategy = get_strategy(strategy_name)
        default_params = {**getattr(strategy, "default_params", {})}
        df = get_ohlcv(sym, "US", interval=interval, period=optimization_period)
        if len(df) < 60:
            return {"symbol": sym, "status": "skipped", "reason": f"insufficient bars ({len(df)})"}
        best_params, _, best_summary, combos, method_used = optimize_strategy(
            df,
            strategy_name,
            method="bayesian",
            dynamic_pos=dynamic_pos,
            n_trials=n_trials,
        )
        if not best_params or not best_summary:
            return {"symbol": sym, "status": "error", "reason": "optimizer returned empty result"}
        return {
            "symbol": sym,
            "status": "ok",
            "method": method_used,
            "combos_tested": combos,
            "optimized_symbol_return": round(float(best_summary.get("total_return", 0)), 2),
            "optimized_symbol_sharpe": round(float(best_summary.get("sharpe", 0)), 2),
            "params": {**default_params, **best_params},
        }
    except Exception as e:
        return {"symbol": sym, "status": "error", "reason": str(e)[:160]}


def _write_new_strategy_validation_report(strategy_name: str, owner: str, interval: str, summary: dict) -> Path:
    report_path = STRATEGY_REPORTS_DIR / f"{strategy_name}_report.md"
    baseline = summary.get("nasdaq_baseline", {}) or {}
    optimized = summary.get("nasdaq_optimized", {}) or {}
    mypool = summary.get("mypool_validation", {}) or {}
    final_standard = summary.get("final_mypool_standard", {}) or {}
    rounds = summary.get("batch_rounds", []) or []
    attempts = summary.get("optimization_attempts", []) or []

    lines = [
        f"# 平台统一验证报告: {strategy_name}",
        "",
        "## 基本信息",
        f"- Owner: {owner}",
        f"- 周期: {interval}",
        f"- 当前状态: {summary.get('final_status', 'unknown')}",
        f"- 说明: {summary.get('reason', '-')}",
        "",
        "## NASDAQ 基准测试",
        f"- 默认参数平均收益: {baseline.get('avg_return')}",
        f"- 有效股票数: {baseline.get('count')}",
        f"- 优化后平均收益: {optimized.get('avg_return')}",
        "",
        "## 通用参数优化尝试",
    ]
    if attempts:
        for item in attempts:
            if item.get("status") == "ok":
                lines.append(
                    f"- {item['symbol']}: NASDAQ均值 {item.get('nasdaq_avg_return')}%, "
                    f"单股收益 {item.get('optimized_symbol_return')}%, Sharpe {item.get('optimized_symbol_sharpe')}, "
                    f"提升通用标准: {'是' if item.get('promoted') else '否'}"
                )
            else:
                lines.append(f"- {item.get('symbol')}: {item.get('status')} {item.get('reason', '')}".strip())
    else:
        lines.append("- 无有效优化尝试")

    lines.extend([
        "",
        "## MYPOOL 验证",
        f"- 通用标准平均收益: {mypool.get('avg_return')}",
        f"- 有效股票数: {mypool.get('count')}",
        "",
        "## 两轮批量优化 + WF",
    ])
    if rounds:
        for item in rounds:
            lines.append(
                f"- Round {item.get('round')}: 完成 {item.get('completed')}/{item.get('total')} "
                f"| 稳健 {item.get('robust')} | 个股标准升级 {item.get('standard_promoted')} "
                f"| 通用标准升级 {item.get('universal_promoted')}"
            )
    else:
        lines.append("- 未执行")

    lines.extend([
        "",
        "## 最终个股标准 MYPOOL 回测",
        f"- 平均收益: {final_standard.get('avg_return')}",
        f"- 有效股票数: {final_standard.get('count')}",
        f"- 达到候选激活线(>={NEW_STRATEGY_READY_THRESHOLD:.0f}%): {'是' if summary.get('ready_for_activation') else '否'}",
        "",
        "## 生成时间",
        datetime.utcnow().isoformat(),
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _run_new_strategy_batch_round(strategy_name: str, interval: str = "1d", n_trials: int = 80,
                                  round_index: int = 1, pool_name: str = "MYPOOL") -> dict:
    pools = _load_pools()
    pool = pools.get(pool_name) or {}
    symbols = [str(s).upper() for s in pool.get("symbols", []) if str(s).strip()]
    market = str(pool.get("market", "US")).upper()
    if not symbols:
        return {
            "round": round_index,
            "job_id": None,
            "status": "skipped",
            "completed": 0,
            "total": 0,
            "robust": 0,
            "standard_promoted": 0,
            "universal_promoted": 0,
            "message": "empty pool",
        }

    job_id = f"newstr_{strategy_name}_{round_index}_{int(time.time())}"
    register_job(_batch_jobs, job_id, {
        "job_id": job_id,
        "status": "running",
        "pool_name": pool_name,
        "market": market,
        "interval": interval,
        "method": "bayesian",
        "total": len(symbols),
        "completed": 0,
        "total_robust": 0,
        "total_standard_promoted": 0,
        "total_universal_promoted": 0,
        "recent_standard_promotions": [],
        "recent_universal_promotions": [],
        "current_symbol": "",
        "current_index": 0,
        "message": f"{strategy_name} round {round_index} 启动中...",
        "results": {},
        "started_at": datetime.utcnow().isoformat(),
    })
    # try/finally 不能省: _batch_optimize_worker 只在自己跑完时把 status 写成
    # done, 中途逃出来的异常(例如池起不来)会让这条 newstr_* 永远停在 running。
    # 而 /batch-optimize 和 /batch-wf 的入口都会扫 _batch_jobs 找 running ——
    # 于是之后**每一次**批量优化都被 409 挡掉, 只能重启进程才能恢复。
    # (endpoint 那条路径本来就有 _run_batch_and_release 的 finally 兜底, 唯独
    #  这条内部路径没有。)
    try:
        _batch_optimize_worker(
            job_id,
            symbols,
            market,
            _optimization_period_for_interval(interval),
            interval,
            "bayesian",
            False,
            [strategy_name],
            n_trials,
        )
    except Exception as e:
        logger.exception(f"[newstr-round] {strategy_name} round {round_index} 失败")
        _j = _batch_jobs.get(job_id)
        if _j is not None:
            _j.update({"status": "error", "error": str(e)[:500],
                       "message": f"批量轮次失败: {str(e)[:120]}"})
        raise
    finally:
        _j = _batch_jobs.get(job_id)
        if _j is not None and _j.get("status") == "running":
            _j.update({"status": "error",
                       "message": "批量轮次异常结束 (worker 未收尾)"})
    sub_job = _batch_jobs.get(job_id, {})
    return {
        "round": round_index,
        "job_id": job_id,
        "status": sub_job.get("status"),
        "completed": sub_job.get("completed", 0),
        "total": sub_job.get("total", 0),
        "robust": sub_job.get("total_robust", 0),
        "standard_promoted": sub_job.get("total_standard_promoted", 0),
        "universal_promoted": sub_job.get("total_universal_promoted", 0),
        "message": sub_job.get("message", ""),
    }

def _new_strategy_validation_worker(job_id: str, owner: str, interval: str, n_trials: int):
    job = _new_strategy_validation_jobs.setdefault(job_id, LockedDict({
        "job_id": job_id, "owner": owner, "interval": interval,
        "n_trials": n_trials, "total": 0, "done": 0,
        "validated": 0, "archived": 0, "failed": 0,
        "current": "", "message": "", "results": LockedDict(),  # 正常路径下走 register_job, 这里只是兜底
        "started_at": datetime.utcnow().isoformat(),
    }))
    # 整个函数体都要兜住: 逃出去的异常会让线程直接死掉, 而 job["status"]
    # 还停在 "running" —— 更糟的是这个 "running" 已经被 _save_validation_jobs
    # 落盘了, 启动时的恢复逻辑看到 running/queued 就重新拉起 worker。如果失败
    # 是确定性的(某个坏候选策略、registry 里一条脏数据), 这个任务会在**每次**
    # 重启时复活并再次崩掉, 永远清不掉, UI 上也没有任何清除入口。
    try:
        owner = _normalize_strategy_owner(owner)
        interval = _normalize_param_interval(interval)
        candidates = _list_new_strategy_candidates(owner)
        # 可选: 只跑勾选的
        only = job.get("only_strategies") or []
        if only:
            only_set = set(only)
            candidates = [c for c in candidates if c in only_set]
        job["owner"] = owner
        job["interval"] = interval
        job["total"] = len(candidates)
        job["status"] = "running"
        job["message"] = f"开始统一验证 {owner} 新策略..."
        _save_validation_jobs()

        if not candidates:
            job["status"] = "done"
            job["message"] = f"{owner} 暂无待验证新策略"
            job["finished_at"] = datetime.utcnow().isoformat()
            _save_validation_jobs()
            return

        # Per-strategy pipeline has 6 stages; expose to UI so progress doesn't
        # appear stuck while one strategy is grinding through bayes / WF.
        _STAGE_TOTAL = 6
        _STAGE_NASDAQ_BASELINE = 1   # NASDAQ 基准测试
        _STAGE_BAYES_OPT       = 2   # 贝叶斯优化通用参数
        _STAGE_MYPOOL_VALIDATE = 3   # MYPOOL 验证
        _STAGE_BATCH_ROUND_1   = 4   # 第 1 轮批量优化 + WF
        _STAGE_BATCH_ROUND_2   = 5   # 第 2 轮批量优化 + WF
        _STAGE_FINAL_STANDARD  = 6   # 个股标准 MYPOOL 收益
        job["stage_total"] = _STAGE_TOTAL

        def _set_stage(num: int, label: str):
            job["stage"] = num
            job["stage_label"] = label
            job["stage_started_at"] = datetime.utcnow().isoformat()
            job["message"] = f"{job.get('current') or ''}: {label}"

        for idx, strategy_name in enumerate(candidates, start=1):
            if job.get("cancelled"):
                job["status"] = "cancelled"
                job["message"] = f"已取消 ({job['done']}/{job['total']})"
                job["finished_at"] = datetime.utcnow().isoformat()
                _save_validation_jobs()
                return

            # 幂等 (仅服务重启恢复的 job): registry 里已有 validated/archived 结果的
            # 策略跳过, 计数按跳过项重建 (resume 时计数已清零). 新起的 job 不跳 —
            # 用户勾选重跑就是要重跑; 'error' 也不跳 — 上次报错的要再试一次.
            _prior = _has_validation_result_for(strategy_name, interval) if job.get("resumed_at") else None
            if _prior in ("archived", "validated"):
                job["done"] = idx
                if _prior == "archived":
                    job["archived"] += 1
                else:
                    job["validated"] += 1
                job["message"] = f"{strategy_name}: 恢复 · 已有结果 ({_prior}), 跳过"
                _save_validation_jobs()
                continue

            job["current"] = strategy_name
            job["strategy_started_at"] = datetime.utcnow().isoformat()
            summary = {
                "owner": owner,
                "interval": interval,
                "started_at": datetime.utcnow().isoformat(),
                "final_status": "running",
            }

            try:
                default_params = {**getattr(get_strategy(strategy_name), "default_params", {})}
                _set_stage(_STAGE_NASDAQ_BASELINE, "NASDAQ 基准测试")
                baseline = _pool_avg_return(
                    strategy_name,
                    default_params,
                    pool_name=UNIVERSAL_BENCHMARK_POOL,
                    period="1y",
                    dynamic_pos=False,
                    interval=interval,
                    executor_mode="process",
                )
                summary["nasdaq_baseline"] = baseline

                _set_stage(_STAGE_BAYES_OPT, "贝叶斯优化通用参数")
                opt_summary = _optimize_universal_candidate_strategy(
                    strategy_name,
                    interval=interval,
                    n_trials=n_trials,
                    dynamic_pos=False,
                )
                summary["nasdaq_optimized"] = opt_summary.get("optimized", {})
                summary["optimization_attempts"] = opt_summary.get("attempts", [])

                _set_stage(_STAGE_MYPOOL_VALIDATE, "MYPOOL 验证")
                universal_params = _get_universal_strategy_params(strategy_name, interval=interval)
                mypool_eval = _pool_avg_return(
                    strategy_name,
                    universal_params,
                    pool_name="MYPOOL",
                    period="1y",
                    dynamic_pos=False,
                    interval=interval,
                    executor_mode="process",
                )
                summary["mypool_validation"] = mypool_eval

                mypool_avg = mypool_eval.get("avg_return")
                if mypool_avg is None or mypool_avg < NEW_STRATEGY_MPOOL_THRESHOLD:
                    _remove_universal_strategy_entry(strategy_name, interval)
                    reason = f"统一验证淘汰：MYPOOL 平均收益 {mypool_avg if mypool_avg is not None else 'N/A'}% < {NEW_STRATEGY_MPOOL_THRESHOLD:.0f}%"
                    _mark_strategy_archived_state(
                        strategy_name,
                        reason,
                        {"avg_annual_return": round(float(mypool_avg), 2)} if mypool_avg is not None else None,
                    )
                    summary["reason"] = reason
                    summary["batch_rounds"] = []
                    summary["final_mypool_standard"] = {}
                    summary["ready_for_activation"] = False
                    summary["final_status"] = "archived"
                else:
                    rounds = []
                    for round_index in (1, 2):
                        _set_stage(
                            _STAGE_BATCH_ROUND_1 if round_index == 1 else _STAGE_BATCH_ROUND_2,
                            f"第 {round_index} 轮批量优化 + WF",
                        )
                        rounds.append(
                            _run_new_strategy_batch_round(
                                strategy_name,
                                interval=interval,
                                n_trials=max(40, n_trials),
                                round_index=round_index,
                                pool_name="MYPOOL",
                            )
                        )
                    summary["batch_rounds"] = rounds

                    _set_stage(_STAGE_FINAL_STANDARD, "计算个股标准 MYPOOL 收益")
                    final_standard = _pool_avg_standard_return(
                        strategy_name,
                        pool_name="MYPOOL",
                        period="1y",
                        market="US",
                        dynamic_pos=False,
                        interval=interval,
                        executor_mode="process",
                    )
                    summary["final_mypool_standard"] = final_standard
                    final_avg = final_standard.get("avg_return")
                    summary["ready_for_activation"] = bool(final_avg is not None and final_avg >= NEW_STRATEGY_READY_THRESHOLD)
                    summary["reason"] = (
                        f"统一验证通过：通用标准 {mypool_avg:.2f}% ，个股标准 {final_avg if final_avg is not None else 'N/A'}%"
                    )
                    summary["final_status"] = "validated"
                    _ensure_strategy_new_candidate(strategy_name, "统一验证完成，保留在新策略")

                report_path = _write_new_strategy_validation_report(strategy_name, owner, interval, summary)
                summary["report_path"] = str(report_path.relative_to(Path(__file__).parent))
                summary["validated_at"] = datetime.utcnow().isoformat()
                _set_strategy_validation_result(strategy_name, owner, interval, summary)
                job["results"][strategy_name] = summary
                if summary["final_status"] == "archived":
                    job["archived"] += 1
                elif summary["final_status"] == "validated":
                    job["validated"] += 1
            except Exception as e:
                summary["final_status"] = "error"
                summary["reason"] = str(e)[:200]
                summary["validated_at"] = datetime.utcnow().isoformat()
                _set_strategy_validation_result(strategy_name, owner, interval, summary)
                job["results"][strategy_name] = summary
                job["failed"] += 1

            job["done"] = idx
            _save_validation_jobs()

        job["current"] = ""
        job["status"] = "done"
        job["message"] = (
            f"{owner} 新策略统一验证完成：保留 {job['validated']}，归档 {job['archived']}，错误 {job['failed']}"
        )
        job["finished_at"] = datetime.utcnow().isoformat()
        _save_validation_jobs()

    except Exception as e:
        logger.exception(f"[newval] job={job_id} worker 异常退出")
        job.update({
            "status": "error",
            "error": str(e)[:500],
            "message": f"验证任务异常终止: {str(e)[:120]}",
            "finished_at": datetime.utcnow().isoformat(),
        })
        _save_validation_jobs()
# 这 5 张也用 LockedDict: _reap_registry 里写着"values() 在 LockedDict 上返回的是
# 副本, 遍历时别人插入/删除都不会炸" —— 那句话对普通 dict 是不成立的, 而回收
# 恰恰要遍历整张表。留着普通 dict 等于注释在骗人, 且回收确实可能撞上写侧。
_bulk_progress = LockedDict()  # 全局下载进度: {task_id: {total, done, success, failed, current, status, details}}
_daily_sync_progress = LockedDict()
_data_sync_jobs = LockedDict()
_fundamental_sync_jobs = LockedDict()
_selector_jobs = LockedDict()
_new_strategy_validation_jobs = LockedDict()   # 同上: 会被遍历

# ── 新策略验证 · job checkpoint 落盘 (survive uvicorn --reload) ──────
_VALIDATION_JOBS_FILE = Path(__file__).parent / ".cache" / "new_strategy_validation_jobs.json"
_VALIDATION_RESUME_MAX = int(os.getenv("AITRADER_VALIDATION_RESUME_MAX", "3"))


def _save_validation_jobs():
    """全量落盘当前 _new_strategy_validation_jobs 快照."""
    try:
        _VALIDATION_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 只序列化能落盘的字段, 忽略 non-JSON 的 (Thread refs 等)
        snapshot = {}
        for jid, job in _new_strategy_validation_jobs.items():
            snapshot[jid] = {k: v for k, v in snapshot_of(job).items()
                             if k not in ("thread",)}
        tmp = _VALIDATION_JOBS_FILE.with_suffix(".tmp")
        # default=_json_default: job["results"] 里是回测直接产出的原始值,
        # numpy 标量会让 json.dumps 抛 TypeError —— 而下面那个 except 只
        # logger.warning, 于是 checkpoint **每次都静默写不进去**, 直到重启
        # 才发现 in-flight 任务全丢了(这套恢复机制存在的唯一理由就是别丢)。
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2,
                                  default=_json_default),
                       encoding="utf-8")
        tmp.replace(_VALIDATION_JOBS_FILE)
    except Exception as e:
        logger.warning(f"validation jobs save failed: {e}", exc_info=True)


def _load_validation_jobs() -> dict:
    if not _VALIDATION_JOBS_FILE.exists():
        return {}
    try:
        return json.loads(_VALIDATION_JOBS_FILE.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"validation jobs load failed: {e}")
        return {}


def _has_validation_result_for(strategy_name: str, interval: str) -> Optional[str]:
    """返回 registry 里该策略该周期的 final_status ('validated'/'archived'/'error'),
    没有则 None. 用于 worker 恢复时跳过已完成的策略."""
    try:
        state = _load_strategy_validation_state()
        interval = _normalize_param_interval(interval)
        info = (state.get("strategies") or {}).get(strategy_name)
        if not info:
            return None
        item = (info.get("intervals") or {}).get(interval)
        if not item:
            return None
        st = item.get("final_status")
        if st in ("validated", "archived", "error"):
            return st
    except Exception:
        pass
    return None


def _resume_validation_jobs_on_startup():
    """服务启动时读取 in-flight jobs, 把 status 'running'/'queued' 的重新起 worker.
    幂等: 已完成的策略会因为 registry 里已有 result 被跳过."""
    saved = _load_validation_jobs()
    if not saved:
        return
    # saved 是 json.loads 出来的**普通** dict。直接 update 进去, 这些 job 就
    # 永远是无锁的 —— 而 worker 那边的 setdefault 看到 key 已存在就不会再包,
    # 于是"重启后恢复"这条路径上竞态原样复活。逐条过工厂。
    for jid, jdata in list(saved.items()):
        register_job(_new_strategy_validation_jobs, jid, jdata)
    resumed = 0
    for jid in list(saved.keys()):
        # 用 .get: register_job 里的回收可能已经把这条清掉了(saved 条数超上限时)
        job = _new_strategy_validation_jobs.get(jid)
        if job is None:
            continue
        if job.get("status") in ("running", "queued"):
            # 恢复次数上限: worker 里确定性的失败(坏候选策略/脏 registry)会让
            # 这条 job 每次重启都复活、再崩、再落盘成 running …… 无限循环,
            # 而 UI 上没有任何清除入口。给它一个止损位。
            tries = int(job.get("resume_count") or 0) + 1
            if tries > _VALIDATION_RESUME_MAX:
                job.update({
                    "status": "error",
                    "message": f"连续 {tries - 1} 次恢复后仍未完成, 已停止自动恢复",
                    "error": "resume_limit_exceeded",
                    "finished_at": datetime.utcnow().isoformat(),
                })
                logger.warning(f"[validation-resume] {jid} 超过恢复上限, 不再拉起")
                continue
            job["resume_count"] = tries
            job["status"] = "queued"
            job["message"] = f"服务重启, 恢复中... (第 {tries} 次)"
            job["resumed_at"] = datetime.utcnow().isoformat()
            # worker 会按 registry 里已有结果重建计数, 这里先清零避免双算
            for k in ("done", "validated", "archived", "failed"):
                job[k] = 0
            job["current"] = ""
            threading.Thread(
                target=_new_strategy_validation_worker,
                args=(jid, job.get("owner", "OpenAI"),
                      job.get("interval", "1d"),
                      job.get("n_trials", 60)),
                daemon=True,
            ).start()
            resumed += 1
    if resumed:
        logger.info(f"[validation-resume] restarted {resumed} in-flight job(s)")
        _save_validation_jobs()


def _sync_file_max_age_sec(interval: str) -> int:
    interval = str(interval or "1d")
    return {
        "1m": 3600,
        "5m": 6 * 3600,
        "15m": 12 * 3600,
        "30m": 24 * 3600,
        "1h": 2 * 86400,
        "1d": 86400,
        "1wk": 7 * 86400,
        "1mo": 30 * 86400,
    }.get(interval, 86400)


def _configured_market_universe(markets: list = None, pools: list = None) -> list:
    all_pools = _load_pools()
    target_markets = {str(m).upper() for m in (markets or []) if str(m).strip()}
    target_pools = set(pools or [])
    seen = set()
    items = []

    # 股票池模式：沿用明确配置的池子内容
    if target_pools:
        for pool_name, info in all_pools.items():
            market = str(info.get("market", "US")).upper()
            if pool_name not in target_pools:
                continue
            if target_markets and market not in target_markets:
                continue
            for sym in info.get("symbols", []):
                key = f"{market}:{sym.upper()}"
                if key in seen:
                    continue
                seen.add(key)
                items.append({"symbol": sym.upper(), "market": market, "pool": pool_name})
        return items

    # 市场模式：使用全市场 universe，而不是本地股票池
    for market in sorted(target_markets or _configured_market_codes()):
        try:
            universe = _load_market_universe(market, refresh=False)
            for row in universe.get("symbols", []):
                sym = str(row.get("symbol", "")).upper()
                if not sym:
                    continue
                key = f"{market}:{sym}"
                if key in seen:
                    continue
                seen.add(key)
                items.append({"symbol": sym, "market": market, "pool": "__market__"})
        except Exception:
            continue
    return items


def _fundamental_path(symbol: str, market: str) -> Path:
    mkt_dir = FUNDAMENTAL_DATA_DIR / str(market).upper()
    mkt_dir.mkdir(exist_ok=True)
    return mkt_dir / f"{str(symbol).upper()}.json"


def _fetch_fundamental_snapshot(symbol: str, market: str) -> dict:
    market = str(market).upper()
    if market == "CRYPTO":
        raise ValueError("CRYPTO fundamentals not supported")
    ticker_sym = yf_symbol(symbol, market)
    ticker = yf.Ticker(ticker_sym)
    info = ticker.info
    revenue_4q_consecutive = None
    revenue_4q_count = None
    revenue_yoy_latest_quarter = None
    quarterly_financials_2y = []
    qdf = None
    qcf = None

    def _series_from_rows(df, aliases: set[str]) -> pd.Series:
        if df is None or getattr(df, "empty", True):
            return pd.Series(dtype="float64")
        row_name = next((idx for idx in df.index if str(idx).strip().lower() in aliases), None)
        if row_name is None:
            return pd.Series(dtype="float64")
        return pd.to_numeric(df.loc[row_name], errors="coerce").dropna()

    try:
        qdf = getattr(ticker, "quarterly_income_stmt", None)
        if qdf is None or qdf.empty:
            qdf = getattr(ticker, "quarterly_financials", None)
        if qdf is not None and not qdf.empty:
            rev = _series_from_rows(qdf, {"total revenue", "operating revenue", "revenue"})
            if len(rev) >= 8:
                rev = rev.sort_index()
                last8 = rev.tail(8)
                prev4 = last8.iloc[:4].tolist()
                curr4 = last8.iloc[4:].tolist()
                yoy_flags = []
                yoy_pcts = []
                for prev_v, curr_v in zip(prev4, curr4):
                    if prev_v and not pd.isna(prev_v) and not pd.isna(curr_v):
                        yoy_flags.append(curr_v > prev_v)
                        yoy_pcts.append((curr_v / prev_v - 1) * 100.0)
                if len(yoy_flags) == 4:
                    revenue_4q_consecutive = all(yoy_flags)
                    revenue_4q_count = sum(1 for x in yoy_flags if x)
                    revenue_yoy_latest_quarter = yoy_pcts[-1]
            op_income = _series_from_rows(qdf, {"operating income", "operatingincome"})
            net_income_q = _series_from_rows(qdf, {"net income", "net income common stockholders", "netincome"})
            gross_profit_q = _series_from_rows(qdf, {"gross profit", "grossprofit"})
        else:
            rev = pd.Series(dtype="float64")
            op_income = pd.Series(dtype="float64")
            net_income_q = pd.Series(dtype="float64")
            gross_profit_q = pd.Series(dtype="float64")
        qcf = getattr(ticker, "quarterly_cashflow", None)
        op_cf = _series_from_rows(qcf, {"operating cash flow", "cash flow from continuing operating activities", "operatingcashflow"})
        fcf = _series_from_rows(qcf, {"free cash flow", "freecashflow"})
        date_index = sorted({
            *(rev.index if len(rev) else []),
            *(op_income.index if len(op_income) else []),
            *(net_income_q.index if len(net_income_q) else []),
            *(gross_profit_q.index if len(gross_profit_q) else []),
            *(op_cf.index if len(op_cf) else []),
            *(fcf.index if len(fcf) else []),
        })
        if date_index:
            for dt in date_index[-8:]:
                quarterly_financials_2y.append({
                    "quarter_end": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "revenue": safe_float(rev.get(dt)),
                    "gross_profit": safe_float(gross_profit_q.get(dt)),
                    "operating_income": safe_float(op_income.get(dt)),
                    "net_income": safe_float(net_income_q.get(dt)),
                    "operating_cashflow": safe_float(op_cf.get(dt)),
                    "free_cashflow": safe_float(fcf.get(dt)),
                })
    except Exception:
        pass
    snapshot = {
        "symbol": str(symbol).upper(),
        "market": market,
        "name": info.get("longName") or info.get("shortName") or str(symbol).upper(),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency"),
        "market_cap": safe_float(info.get("marketCap")),
        "enterprise_value": safe_float(info.get("enterpriseValue")),
        "shares_outstanding": safe_float(info.get("sharesOutstanding")),
        "pe_ratio": safe_float(info.get("trailingPE")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "pb_ratio": safe_float(info.get("priceToBook")),
        "ps_ratio": safe_float(info.get("priceToSalesTrailing12Months")),
        "ev_to_ebitda": safe_float(info.get("enterpriseToEbitda")),
        "dividend_yield": safe_float(info.get("dividendYield")),
        "profit_margin": safe_float(info.get("profitMargins")),
        "operating_margin": safe_float(info.get("operatingMargins")),
        "gross_margin": safe_float(info.get("grossMargins")),
        "return_on_equity": safe_float(info.get("returnOnEquity")),
        "return_on_assets": safe_float(info.get("returnOnAssets")),
        "revenue_growth": safe_float(info.get("revenueGrowth")),
        "earnings_growth": safe_float(info.get("earningsGrowth")),
        "earnings_quarterly_growth": safe_float(info.get("earningsQuarterlyGrowth")),
        "debt_to_equity": safe_float(info.get("debtToEquity")),
        "current_ratio": safe_float(info.get("currentRatio")),
        "quick_ratio": safe_float(info.get("quickRatio")),
        "operating_cashflow": safe_float(info.get("operatingCashflow")),
        "free_cashflow": safe_float(info.get("freeCashflow")),
        "ebitda": safe_float(info.get("ebitda")),
        "total_revenue": safe_float(info.get("totalRevenue")),
        "net_income": safe_float(info.get("netIncomeToCommon")),
        "beta": safe_float(info.get("beta")),
        "52w_high": safe_float(info.get("fiftyTwoWeekHigh")),
        "52w_low": safe_float(info.get("fiftyTwoWeekLow")),
        "recommendation": info.get("recommendationKey"),
        "target_mean_price": safe_float(info.get("targetMeanPrice")),
        "next_earnings_date": info.get("earningsTimestamp"),
        "revenue_growth_4q_consecutive": revenue_4q_consecutive,
        "revenue_growth_4q_count": revenue_4q_count,
        "revenue_yoy_latest_quarter": safe_float(revenue_yoy_latest_quarter),
        "quarterly_financials_2y": quarterly_financials_2y,
        "source": "Yahoo Finance",
        "updated_at": datetime.utcnow().isoformat(),
    }
    return snapshot


def _save_fundamental_snapshot(symbol: str, market: str, snapshot: dict):
    fp = _fundamental_path(symbol, market)
    fp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_fundamental_snapshot(symbol: str, market: str) -> dict:
    fp = _fundamental_path(symbol, market)
    return _read_json_file(fp, {})


def _selector_filter_requested(filters: dict) -> bool:
    if not filters:
        return False
    return any(v not in (None, "", [], False) for v in filters.values())


def _selector_pct_value(v):
    v = safe_float(v)
    if v is None:
        return None
    return v * 100.0


def _parse_selector_filters(body: dict = None) -> dict:
    body = body or {}
    raw = body.get("filters") or {}
    def _num(key):
        v = raw.get(key)
        if v in (None, ""):
            return None
        return safe_float(v)
    # News filter mode: 'off' | 'must' | 'weight' | None (treated as off)
    news_require = str(raw.get("news_require", "") or "").strip().lower() or None
    if news_require not in (None, "off", "must", "weight"):
        news_require = None
    return {
        "price_min": _num("price_min"),
        "price_max": _num("price_max"),
        "pe_min": _num("pe_min"),
        "pe_max": _num("pe_max"),
        "market_cap_min_b": _num("market_cap_min_b"),
        "market_cap_max_b": _num("market_cap_max_b"),
        "volume_min": _num("volume_min"),
        "avg_volume_20_min": _num("avg_volume_20_min"),
        "revenue_growth_min_pct": _num("revenue_growth_min_pct"),
        "earnings_growth_min_pct": _num("earnings_growth_min_pct"),
        "earnings_quarterly_growth_min_pct": _num("earnings_quarterly_growth_min_pct"),
        "roe_min_pct": _num("roe_min_pct"),
        "profit_margin_min_pct": _num("profit_margin_min_pct"),
        "debt_to_equity_max": _num("debt_to_equity_max"),
        "sector_keyword": str(raw.get("sector_keyword", "")).strip(),
        "require_revenue_4q_consecutive": bool(raw.get("require_revenue_4q_consecutive")),
        "require_positive_fcf": bool(raw.get("require_positive_fcf")),
        # 📰 News-driven (Phase 2, owner=news)
        # window clamped to [1, 168] hours server-side to bound LLM/API cost.
        "news_require": news_require,
        "news_sentiment_min": _num("news_sentiment_min"),
        "news_window_hours": (lambda v: None if v is None
                              else max(1.0, min(float(v), 168.0)))(_num("news_window_hours")),
    }


def _selector_local_metrics(symbol: str, market: str, interval: str) -> dict:
    df = get_ohlcv(symbol, market, interval=interval, period="1y", include_today=True, local_only=True)
    if df is None or df.empty:
        if interval != "1d":
            df = get_ohlcv(symbol, market, interval="1d", period="1y", include_today=True, local_only=True)
        if df is None or df.empty:
            return {}
    close = safe_float(df["close"].iloc[-1]) if "close" in df.columns and len(df) else None
    volume = safe_float(df["volume"].iloc[-1]) if "volume" in df.columns and len(df) else None
    avg20 = safe_float(df["volume"].tail(20).mean()) if "volume" in df.columns and len(df) else None
    return {"price": close, "volume": volume, "avg_volume_20": avg20}


def _selector_filter_one(args):
    item, interval, filters = args
    symbol = item["symbol"]
    market = item["market"]
    try:
        metrics = _selector_local_metrics(symbol, market, interval)
        fund = _load_fundamental_snapshot(symbol, market)
        price = metrics.get("price")
        volume = metrics.get("volume")
        avg_volume_20 = metrics.get("avg_volume_20")
        market_cap = safe_float(fund.get("market_cap"))
        pe_ratio = safe_float(fund.get("pe_ratio"))
        revenue_growth_pct = _selector_pct_value(fund.get("revenue_growth"))
        earnings_growth_pct = _selector_pct_value(fund.get("earnings_growth"))
        earnings_q_growth_pct = _selector_pct_value(fund.get("earnings_quarterly_growth"))
        roe_pct = _selector_pct_value(fund.get("return_on_equity"))
        profit_margin_pct = _selector_pct_value(fund.get("profit_margin"))
        debt_to_equity = safe_float(fund.get("debt_to_equity"))
        sector_text = " ".join([str(fund.get("sector", "")).strip(), str(fund.get("industry", "")).strip()]).strip().lower()

        def _fail(reason):
            return False, reason

        if filters.get("price_min") is not None and (price is None or price < filters["price_min"]):
            return _fail("price_min")
        if filters.get("price_max") is not None and (price is None or price > filters["price_max"]):
            return _fail("price_max")
        if filters.get("volume_min") is not None and (volume is None or volume < filters["volume_min"]):
            return _fail("volume_min")
        if filters.get("avg_volume_20_min") is not None and (avg_volume_20 is None or avg_volume_20 < filters["avg_volume_20_min"]):
            return _fail("avg_volume_20_min")
        if filters.get("market_cap_min_b") is not None and (market_cap is None or market_cap < filters["market_cap_min_b"] * 1e9):
            return _fail("market_cap_min_b")
        if filters.get("market_cap_max_b") is not None and (market_cap is None or market_cap > filters["market_cap_max_b"] * 1e9):
            return _fail("market_cap_max_b")
        if filters.get("pe_min") is not None and (pe_ratio is None or pe_ratio < filters["pe_min"]):
            return _fail("pe_min")
        if filters.get("pe_max") is not None and (pe_ratio is None or pe_ratio > filters["pe_max"]):
            return _fail("pe_max")
        if filters.get("revenue_growth_min_pct") is not None and (revenue_growth_pct is None or revenue_growth_pct < filters["revenue_growth_min_pct"]):
            return _fail("revenue_growth_min_pct")
        if filters.get("earnings_growth_min_pct") is not None and (earnings_growth_pct is None or earnings_growth_pct < filters["earnings_growth_min_pct"]):
            return _fail("earnings_growth_min_pct")
        if filters.get("earnings_quarterly_growth_min_pct") is not None and (earnings_q_growth_pct is None or earnings_q_growth_pct < filters["earnings_quarterly_growth_min_pct"]):
            return _fail("earnings_quarterly_growth_min_pct")
        if filters.get("roe_min_pct") is not None and (roe_pct is None or roe_pct < filters["roe_min_pct"]):
            return _fail("roe_min_pct")
        if filters.get("profit_margin_min_pct") is not None and (profit_margin_pct is None or profit_margin_pct < filters["profit_margin_min_pct"]):
            return _fail("profit_margin_min_pct")
        if filters.get("debt_to_equity_max") is not None and (debt_to_equity is None or debt_to_equity > filters["debt_to_equity_max"]):
            return _fail("debt_to_equity_max")
        if filters.get("sector_keyword") and filters["sector_keyword"].lower() not in sector_text:
            return _fail("sector_keyword")
        if filters.get("require_positive_fcf") and not ((safe_float(fund.get("free_cashflow")) or 0) > 0):
            return _fail("require_positive_fcf")
        if filters.get("require_revenue_4q_consecutive") and fund.get("revenue_growth_4q_consecutive") is not True:
            return _fail("require_revenue_4q_consecutive")
        return True, ""
    except Exception as e:
        return False, str(e)[:80]


# ── 📰 News-driven scoring (Phase 2, owner=news) ─────────────────────────────
# LLM CLI calls must be serial per provider (per llm_call_rules.md).
# Therefore news scoring runs AFTER the parallel technical scan, in main thread,
# annotating each alert with sentiment data and optionally filtering by score.
NEWS_SCORE_MAX_SYMBOLS = 50  # cap to keep one selector run < ~15 min @ 18s/call

def _selector_score_news(alerts: list, filters: dict, progress_hook=None) -> tuple[list, int, int]:
    """Annotate alerts with news sentiment and optionally filter by min score.

    Returns: (annotated_alerts, news_filtered_out_count, n_symbols_scored).

    Mode (filters['news_require']):
      - 'must':   drop alerts whose news_score < news_sentiment_min
      - 'weight': annotate only; sort alerts by news_score desc
      - 'off' / None: no-op (returns alerts unchanged)
    """
    mode = (filters or {}).get("news_require")
    if mode not in ("must", "weight") or not alerts:
        return alerts, 0, 0

    # Lazy imports — avoid loading at module import time
    try:
        from news_provider import get_stock_news_unified
        from news_scorer import score_news as _score_news
    except Exception as e:
        logger.warning(f"news modules unavailable, skip news scoring: {e}")
        return alerts, 0, 0

    window_h = int((filters.get("news_window_hours") or 24) or 24)
    min_score = float(filters.get("news_sentiment_min") or 0)

    # Dedupe by symbol — score once per stock
    sym_to_market: dict[str, str] = {}
    for a in alerts:
        sym = a.get("symbol")
        if sym and sym not in sym_to_market:
            sym_to_market[sym] = a.get("market") or "US"

    syms_to_score = list(sym_to_market.keys())
    if len(syms_to_score) > NEWS_SCORE_MAX_SYMBOLS:
        logger.warning(
            f"news scoring capped: {len(syms_to_score)} symbols → first {NEWS_SCORE_MAX_SYMBOLS}"
        )
        syms_to_score = syms_to_score[:NEWS_SCORE_MAX_SYMBOLS]

    # 2026-04-27: news_pools integration —
    #   "已经在股票池的，不再参与评分" (per user feedback). If a symbol is already
    #   in today's daily pool, skip the costly LLM call and reuse the cached score.
    try:
        import news_pools as _news_pools
    except Exception:
        _news_pools = None  # graceful degrade if module missing

    score_map: dict[str, dict] = {}
    skipped_already_in_pool = 0
    n_total = len(syms_to_score)
    if progress_hook:
        progress_hook("news_scoring", 0, n_total, "拉新闻 + LLM 打分...")

    for idx, sym in enumerate(syms_to_score):
        market_for_sym = sym_to_market[sym]

        # Pool cache hit: skip LLM, reuse stored score
        if _news_pools and _news_pools.already_in_today_pool(sym, market_for_sym):
            cached = _news_pools.get_pool_entry(sym, market_for_sym, source="daily")
            if cached:
                score_map[sym] = {
                    "score": cached.get("score"),
                    "label": cached.get("label") or "neutral",
                    "reasoning": cached.get("reasoning") or "(cached from today's pool)",
                    "n_items": cached.get("n_news") or 0,
                    "top3": [],
                    "from_pool": True,
                }
                skipped_already_in_pool += 1
                if progress_hook:
                    progress_hook("news_scoring", idx + 1, n_total,
                                  f"⚡ {sym} 命中池缓存 (score={cached.get('score')})")
                continue

        try:
            news = get_stock_news_unified(sym, market_for_sym, window_hours=window_h)
            sr = _score_news(sym, news, provider="claude")
            score_map[sym] = {
                "score": sr["score"],
                "label": sr["label"],
                "reasoning": sr["reasoning"],
                "n_items": sr["n_items"],
                "top3": [
                    {
                        "title": (n.get("title") or "")[:200],
                        "url": n.get("url") or "",
                        "source": n.get("source") or "",
                        "published_at": n.get("published_at") or "",
                    }
                    for n in news[:3]
                ],
            }
            # Auto-populate pools if score >= threshold
            if _news_pools:
                try:
                    _news_pools.maybe_add(
                        sym, market_for_sym,
                        score=sr.get("score"),
                        label=sr.get("label"),
                        reasoning=sr.get("reasoning"),
                        n_news=sr.get("n_items"),
                    )
                except Exception as e:
                    logger.debug(f"news_pools.maybe_add failed for {sym}: {e}")
        except Exception as e:
            logger.warning(f"news score failed for {sym}: {e}")
            score_map[sym] = {
                "score": None, "label": "error",
                "reasoning": str(e)[:200], "n_items": 0, "top3": [],
            }
        if progress_hook:
            progress_hook("news_scoring", idx + 1, n_total, f"打分 {sym} ({score_map[sym].get('score')})")

    if skipped_already_in_pool and progress_hook:
        progress_hook("news_scoring", n_total, n_total,
                      f"完成 · {skipped_already_in_pool} 只命中今日池缓存 (省 LLM 调用)")

    # Annotate every alert (overflow beyond NEWS_SCORE_MAX_SYMBOLS marked 'skipped')
    annotated = []
    filtered_out = 0
    for a in alerts:
        sym = a.get("symbol")
        nd = score_map.get(sym)
        if nd is None:
            # In 'must' mode, unscored alerts cannot satisfy min_score → drop them.
            # In 'weight' mode, keep at the bottom of the sort with score=None.
            if mode == "must":
                filtered_out += 1
                continue
            a["news"] = {"score": None, "label": "skipped", "reasoning": "above_score_cap",
                         "n_items": 0, "top3": []}
            annotated.append(a)
            continue
        a["news"] = nd
        if mode == "must":
            score = nd.get("score")
            if score is None or score < min_score:
                filtered_out += 1
                continue
        annotated.append(a)

    if mode == "weight":
        annotated.sort(key=lambda x: ((x.get("news") or {}).get("score") or -1), reverse=True)

    return annotated, filtered_out, n_total


def _fundamental_scope_status(scope: str, market: str = None, pool: str = None) -> dict:
    scope = str(scope or "pool").lower()
    if scope == "market":
        target_market = str(market or "").upper()
        items = _configured_market_universe(markets=[target_market])
    else:
        items = _configured_market_universe(pools=[pool] if pool else None)
        target_market = None

    total = len(items)
    downloaded = 0
    stale = 0
    missing = 0
    rows = []
    now_ts = time.time()
    for item in items[:500]:
        sym = item["symbol"]
        mkt = item["market"]
        fp = _fundamental_path(sym, mkt)
        if not fp.exists():
            missing += 1
            rows.append({"symbol": sym, "market": mkt, "status": "missing", "updated_at": None})
            continue
        downloaded += 1
        age_hours = (now_ts - fp.stat().st_mtime) / 3600
        is_stale = age_hours > 24 * 7
        if is_stale:
            stale += 1
        rows.append({
            "symbol": sym,
            "market": mkt,
            "status": "stale" if is_stale else "fresh",
            "updated_at": datetime.fromtimestamp(fp.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    missing += max(0, total - len(rows) - max(0, downloaded - len([r for r in rows if r["status"] != "missing"])))
    return {
        "scope": scope,
        "market": target_market,
        "pool": pool,
        "total": total,
        "downloaded": downloaded,
        "stale": stale,
        "missing": max(0, total - downloaded),
        "rows": rows,
    }


def _one_year_daily_status(symbol: str, market: str, min_bars: int = 200) -> dict:
    path = _get_data_path(symbol, market, "1d")
    if not path.exists():
        return {"ok": False, "reason": "missing"}
    try:
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return {"ok": False, "reason": "empty"}
        recent = _filter_period(df.copy(), "1y", "1d", include_today=True)
        bars = len(recent)
        return {
            "ok": bars >= min_bars,
            "bars_1y": bars,
            "start": str(recent.index[0].date()) if bars else None,
            "end": str(recent.index[-1].date()) if bars else None,
            "updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)[:80]}


def _configured_period_for_market(market: str, interval: str = "1d") -> str:
    cfg = _load_market_data_config()
    market_cfg = cfg.get("markets", {}).get(str(market).upper(), {})
    # 兜底以前硬写 "1y" —— 分钟/小时线拿 1y 是拿不到的
    return market_cfg.get("selector", {}).get(
        interval, "1y" if interval == "1d" else _optimization_period_for_interval(interval))


def _configured_period_for_pool(pool_name: str, interval: str, usage: str = "optimizer") -> str:
    cfg = _load_market_data_config()
    pool_cfg = cfg.get("pools", {}).get(pool_name, {})
    usage_map = pool_cfg.get(usage, {}) if isinstance(pool_cfg, dict) else {}
    if interval in usage_map:
        return usage_map[interval]
    if usage == "selector":
        return "1y" if interval == "1d" else _optimization_period_for_interval(interval)
    return _optimization_period_for_interval(interval)

def _get_stock_name(symbol: str, market: str) -> str:
    """获取股票名称（带缓存）"""
    cache_key = f"name:{market}:{symbol}"
    cached = cache_get(cache_key, ttl=86400)  # 24h缓存
    if cached:
        return cached
    try:
        ticker_sym = yf_symbol(symbol, market.upper()) if market.upper() != "CRYPTO" else f"{symbol.upper()}-USD"
        info = yf.Ticker(ticker_sym).info
        name = info.get("longName") or info.get("shortName") or symbol
        cache_set(cache_key, name)
        return name
    except Exception:
        return symbol

def _portfolio_bt_one(args):
    """模块级函数：组合回测中单股票回测（可被ProcessPool序列化）"""
    sym, mkt, sname, prd, dyn = args
    try:
        df = get_ohlcv(sym, mkt, interval="1d", period=prd)
        if len(df) < 1:
            return {"symbol": sym, "error": "数据不足"}
        r = run_strategy_backtest(df, sname, dynamic_position=dyn)
        r["symbol"] = sym
        return r
    except Exception as e:
        return {"symbol": sym, "error": str(e)}


def _combo_bt_one(args):
    """多策略组合回测：对单只股票用多策略 OR 逻辑合并信号，全仓买卖"""
    sym, mkt, strategies, prd, interval = args
    try:
        from backtest import get_strategy, get_ohlcv, run_backtest
        import pandas as pd, numpy as np

        df_raw = get_ohlcv(sym, mkt, interval=interval, period=prd)
        if df_raw is None or len(df_raw) < 30:
            return {"symbol": sym, "error": "数据不足"}

        capital = 100000.0
        # IB Tiered commission: $0.0035/share + $0.0002 clearing + $0.003 exchange = $0.0067/share
        # Min $0.35/order, max 1% of trade value
        IB_PER_SHARE = 0.0067
        IB_MIN_ORDER = 0.35
        IB_MAX_PCT = 0.01
        def ib_commission(shares, price):
            raw = shares * IB_PER_SHARE
            trade_value = shares * price
            return max(IB_MIN_ORDER, min(raw, trade_value * IB_MAX_PCT))

        # Run each strategy to get signals
        signal_frames = {}  # strategy_name -> df with buy_signal/sell_signal columns
        indicator_frames = {}  # strategy_name -> full df with indicator columns (for chart display)
        primary_strategy = None  # first successful strategy → its indicators shown on chart
        primary_chart_indicators = None
        primary_indicator_formulas = None
        primary_signal_rules = None
        for sname in strategies:
            try:
                strat = get_strategy(sname)
                params = {**strat.default_params}
                # Try loading per-stock params
                try:
                    sp = _get_standard_strategy_params(sym, sname)
                    if sp:
                        params.update(sp)
                except Exception:
                    pass
                df_s = strat.add_indicators(df_raw.copy(), params)
                df_s = strat.generate_signals(df_s, params)
                signal_frames[sname] = df_s[["buy_signal", "sell_signal"]].copy() if "buy_signal" in df_s.columns else pd.DataFrame(index=df_s.index)
                indicator_frames[sname] = df_s.copy()
                if primary_strategy is None:
                    primary_strategy = sname
                    primary_chart_indicators = getattr(strat, "chart_indicators", None)
                    primary_indicator_formulas = getattr(strat, "indicator_formulas", None)
                    primary_signal_rules = getattr(strat, "signal_rules", None)
            except Exception:
                continue

        if not signal_frames:
            return {"symbol": sym, "error": "所有策略信号生成失败"}

        # Build common index (intersection of all signal frames)
        common_idx = df_raw.index
        for sf in signal_frames.values():
            common_idx = common_idx.intersection(sf.index)
        if len(common_idx) < 10:
            return {"symbol": sym, "error": "共同K线不足"}

        df = df_raw.loc[common_idx].copy()

        # Merge signals: OR logic
        combo_buy = pd.Series(False, index=common_idx)
        combo_sell = pd.Series(False, index=common_idx)
        buy_triggers = pd.Series("", index=common_idx, dtype=str)
        sell_triggers = pd.Series("", index=common_idx, dtype=str)

        for sname, sf in signal_frames.items():
            sf_aligned = sf.reindex(common_idx).fillna(False)
            buy_mask = sf_aligned.get("buy_signal", pd.Series(False, index=common_idx)).astype(bool)
            sell_mask = sf_aligned.get("sell_signal", pd.Series(False, index=common_idx)).astype(bool)
            combo_buy = combo_buy | buy_mask
            combo_sell = combo_sell | sell_mask
            # Track which strategies triggered
            buy_triggers = buy_triggers.where(~buy_mask, buy_triggers + sname + ",")
            sell_triggers = sell_triggers.where(~sell_mask, sell_triggers + sname + ",")

        # Simulate: full-position buy/sell
        cash = capital
        qty = 0.0
        total_commission = 0.0
        trades = []
        equity_curve = []
        chart_data = []
        wins = 0
        losses = 0
        last_buy_cost = 0.0
        is_intra = len(df) >= 2 and (df.index[1] - df.index[0]).total_seconds() < 86400

        # ─────────────────────────────────────────────────────────────────────
        # 信号-成交分离（A1 修复 — 消除 look-ahead bias）
        # 信号在 bar (i-1) 的 close 检出 → 成交在 bar i 的 open
        # 最后一根 bar (i=n-1) 的信号没有下一根 open 可以成交 → 自然丢弃
        # 参考：docs/superpowers/reports/2026-04-22-data-source-audit.md §A1
        # ─────────────────────────────────────────────────────────────────────
        for i, (idx, row) in enumerate(df.iterrows()):
            close = float(row["close"])
            open_price = float(row["open"])
            did_buy = False
            did_sell = False

            # 上一根 bar 的信号（用于本根 bar 的 open 成交）
            prev_sell_signal = bool(combo_sell.iloc[i - 1]) if i >= 1 else False
            prev_buy_signal  = bool(combo_buy.iloc[i - 1])  if i >= 1 else False
            prev_sell_trig   = sell_triggers.iloc[i - 1].rstrip(",") if i >= 1 else ""
            prev_buy_trig    = buy_triggers.iloc[i - 1].rstrip(",")  if i >= 1 else ""

            # Sell first (if holding and previous bar had sell signal)
            if qty > 0 and prev_sell_signal:
                sell_fee = ib_commission(qty, open_price)
                total_commission += sell_fee
                sell_value = qty * open_price - sell_fee
                pnl = sell_value - last_buy_cost
                trades.append({
                    "time": str(idx.date()) if not is_intra else str(idx),
                    "side": "SELL",
                    "price": round(open_price, 2),
                    "qty": round(qty, 2),
                    "value": round(sell_value, 2),
                    "pnl": round(pnl, 2),
                    "trigger_strategy": prev_sell_trig,
                })
                cash += sell_value
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                qty = 0.0
                did_sell = True

            # Buy (if not holding and previous bar had buy signal)
            if qty == 0 and prev_buy_signal and not did_sell:
                # Estimate max shares affordable (cash * 95% / price, then subtract commission)
                est_qty = int(cash * 0.95 / open_price)
                buy_fee = ib_commission(est_qty, open_price) if est_qty > 0 else 0
                buy_qty = int((cash * 0.95 - buy_fee) / open_price)
                if buy_qty > 0:
                    buy_fee = ib_commission(buy_qty, open_price)
                    total_commission += buy_fee
                    cost = buy_qty * open_price + buy_fee
                    cash -= cost
                    qty = buy_qty
                    last_buy_cost = cost  # total cost including commission
                    trades.append({
                        "time": str(idx.date()) if not is_intra else str(idx),
                        "side": "BUY",
                        "price": round(open_price, 2),
                        "qty": buy_qty,
                        "value": round(cost, 2),
                        "pnl": None,
                        "trigger_strategy": prev_buy_trig,
                    })
                    did_buy = True

            equity = cash + qty * close
            date_key = str(idx.date()) if not is_intra else int(idx.timestamp())
            equity_curve.append({"date": str(idx.date()) if not is_intra else str(idx), "equity": round(equity, 2)})

            chart_row = {
                "date": date_key,
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": int(row["volume"]),
                "combo_buy": did_buy,
                "combo_sell": did_sell,
                "buy_triggers": prev_buy_trig if did_buy else "",
                "sell_triggers": prev_sell_trig if did_sell else "",
            }
            # Merge primary strategy indicator columns for chart overlay/subchart display
            if primary_strategy and primary_strategy in indicator_frames:
                prim_df = indicator_frames[primary_strategy]
                if idx in prim_df.index:
                    prim_row = prim_df.loc[idx]
                    # All chart_indicators-referenced columns
                    if primary_chart_indicators:
                        all_inds = (primary_chart_indicators.get("overlay") or []) + (primary_chart_indicators.get("subchart") or [])
                        for ind in all_inds:
                            col = ind.get("col")
                            if col and col in prim_df.columns:
                                val = prim_row.get(col)
                                try:
                                    if val is not None and pd.notna(val):
                                        chart_row[col] = float(val)
                                except Exception:
                                    pass
            chart_data.append(chart_row)

        # Force close at end of period if still holding
        if qty > 0:
            last_close = float(df.iloc[-1]["close"])
            close_fee = ib_commission(qty, last_close)
            total_commission += close_fee
            sell_value = qty * last_close - close_fee
            pnl = sell_value - last_buy_cost
            last_date = str(df.index[-1].date()) if not is_intra else str(df.index[-1])
            trades.append({
                "time": last_date,
                "side": "SELL",
                "price": round(last_close, 2),
                "qty": round(qty, 2),
                "value": round(sell_value, 2),
                "pnl": round(pnl, 2),
                "trigger_strategy": "到期平仓",
            })
            cash += sell_value
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            qty = 0.0
            # Update last chart_data point
            if chart_data:
                chart_data[-1]["combo_sell"] = True
                chart_data[-1]["sell_triggers"] = "到期平仓"

        final_equity = cash
        total_return = (final_equity - capital) / capital * 100
        trade_count = wins + losses

        # Quick per-strategy solo return (reuse signal_frames, simple buy/sell sim)
        # ── Codex#3 (2026-04-22) — 去掉 per_strategy_return 的 look-ahead ──
        # 旧实现用同一根 bar 的 close 同时做信号检测+成交，与 A1 修复后的主 combo 循环
        # (previous-bar signal → current-bar open) 标准不一致，会让 solo 列显得比
        # combo 列更高（systematically 偏乐观），用户按 solo 排序会错误选出其实
        # 有 look-ahead 优势的策略。
        # 修法：与 main combo 循环完全同构 —— 前一根 bar 触发信号，本根 bar open 成交；
        # 最后一根 bar 的信号自然丢弃（没有 T+1 open）。
        per_strategy_return = {}
        for sname, sf in signal_frames.items():
            sf_a = sf.reindex(common_idx).fillna(False)
            s_buy = sf_a.get("buy_signal", pd.Series(False, index=common_idx)).astype(bool)
            s_sell = sf_a.get("sell_signal", pd.Series(False, index=common_idx)).astype(bool)
            s_cash, s_qty = capital, 0.0
            for j in range(len(common_idx)):
                open_p = float(df.iloc[j]["open"])
                prev_buy  = bool(s_buy.iloc[j - 1])  if j >= 1 else False
                prev_sell = bool(s_sell.iloc[j - 1]) if j >= 1 else False
                did_sell = False
                # Sell first (same semantics as combo loop: sell → buy on same bar OK-ish)
                if s_qty > 0 and prev_sell:
                    s_cash += s_qty * open_p - ib_commission(s_qty, open_p)
                    s_qty = 0.0
                    did_sell = True
                # Buy (only if we didn't just close on this bar)
                if s_qty == 0 and prev_buy and not did_sell:
                    sq = int(s_cash * 0.95 / open_p)
                    if sq > 0:
                        fee = ib_commission(sq, open_p)
                        s_cash -= sq * open_p + fee
                        s_qty = sq
            # 期末 mark-to-market 用最后一根 close（与主循环的"到期平仓"收益基本等价）
            s_final = s_cash + s_qty * float(df.iloc[-1]["close"])
            per_strategy_return[sname] = round((s_final - capital) / capital * 100, 2)

        return {
            "symbol": sym,
            "summary": {
                "total_return": round(total_return, 2),
                "final_equity": round(final_equity, 2),
                "trade_count": trade_count,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / trade_count * 100, 1) if trade_count > 0 else 0,
                "capital": capital,
                "total_commission": round(total_commission, 2),
                "per_strategy_return": per_strategy_return,
            },
            "trades": trades,
            "equity_curve": equity_curve,
            "chart_data": chart_data,
            "chart_indicators": primary_chart_indicators or {"overlay": [], "subchart": []},
            "indicator_formulas": primary_indicator_formulas or {},
            "signal_rules": primary_signal_rules or {},
            "primary_strategy": primary_strategy,
        }
    except Exception as e:
        return {"symbol": sym, "error": str(e)}


def _bp_one_task(args):
    """模块级函数：单策略 Buy 点分析的「一只股票 × 一个窗口」(可被 ProcessPool 序列化)。

    /strategies/buy-point-analysis (单策略版) 之前是双层串行:
    主窗口逐股跑一遍, 再对 6 个窗口 × N 只股票又跑一遍, 95 只的池子就是 570 次分析
    全在一个线程里。而同文件的 buy-point-batch (多策略版) 早就并行了。
    主窗口本来就包含在 multi_windows 里 (路由里 sorted(set(...+[window]))),
    所以合成一趟 (窗口 × 股票) 就够, 顺带还省掉主窗口被重复算一次。

    want_chart: 只有主窗口那一轮需要 chart_data, 6 个窗口都带回来太重。
    """
    import pickle as _pk
    from backtest import analyze_buy_points
    # *_extra 是为了兼容还没传新参数的老调用方(判定方式默认走原口径)
    sym, df_bytes, sname, params, w, entry_mode, mfe_target, mae_limit, want_chart, *_extra = args
    judge_mode = _extra[0] if len(_extra) > 0 else "mfe_mae"
    same_bar = _extra[1] if len(_extra) > 1 else "adverse_first"
    try:
        df = _pk.loads(df_bytes)
        r = analyze_buy_points(df, sname, params_override=params, window=w, entry_mode=entry_mode,
                               judge_mode=judge_mode, up_pct=mfe_target, dn_pct=mae_limit,
                               same_bar=same_bar)
        bps = r["buy_points"]
        if judge_mode != "first_touch":
            # 先到先判那条路 analyze_buy_points 里已经按路径填好 success 了,
            # 这里再按 mfe/mae 覆盖一遍会把结论改回跟顺序无关的老口径。
            for bp in bps:
                bp["success"] = (bp["mfe"] >= mfe_target and abs(bp["mae"]) <= mae_limit)
        out = {"symbol": sym, "window": w, "buy_points": bps, "stats": r["stats"], "error": None}
        if want_chart:
            out["chart_data"] = r["chart_data"]
        return out
    except Exception as e:
        return {"symbol": sym, "window": w, "buy_points": [], "stats": None, "error": str(e)[:200]}


def _sp_one_task(args):
    """模块级函数：单策略 Sell 点分析的「一只股票 × 一个窗口」。与 _bp_one_task 对称。"""
    import pickle as _pk
    from backtest import analyze_sell_points
    # *_extra 兼容还没传新参数的老调用方
    sym, df_bytes, sname, params, w, mfe_target, mae_limit, want_chart, *_extra = args
    judge_mode = _extra[0] if len(_extra) > 0 else "mfe_mae"
    same_bar = _extra[1] if len(_extra) > 1 else "adverse_first"
    try:
        df = _pk.loads(df_bytes)
        r = analyze_sell_points(df, sname, params_override=params, window=w,
                                judge_mode=judge_mode, down_pct=mfe_target,
                                up_pct=mae_limit, same_bar=same_bar)
        sps = r["sell_points"]
        if judge_mode != "first_touch":
            for sp in sps:
                # 注意: sell 这边原代码就是不取绝对值 (跟 buy 不同), 照搬
                sp["success"] = (sp["mfe"] >= mfe_target and sp["mae"] <= mae_limit)
        out = {"symbol": sym, "window": w, "sell_points": sps, "stats": r["stats"], "error": None}
        if want_chart:
            out["chart_data"] = r["chart_data"]
        return out
    except Exception as e:
        return {"symbol": sym, "window": w, "sell_points": [], "stats": None, "error": str(e)[:200]}


def _bp_analysis_worker(args):
    """模块级函数：对单个策略做BUY点分析（含多窗口）—— 可被ProcessPool序列化。
    接收 pickled DataFrames 避免重复下载。"""
    import pickle, numpy as np
    from backtest import analyze_buy_points

    strategy_name, stock_data, multi_windows, window, mfe_target, mae_limit, entry_mode, interval, *_extra = args
    judge_mode = _extra[0] if len(_extra) > 0 else "mfe_mae"
    same_bar = _extra[1] if len(_extra) > 1 else "adverse_first"

    stock_cache = {}
    for sym, df_bytes, params in stock_data:
        stock_cache[sym] = (pickle.loads(df_bytes), params)

    per_stock = []
    all_buy_points = []
    for sym, (df, params) in stock_cache.items():
        result = analyze_buy_points(df, strategy_name,
                                    params_override=params,
                                    window=window,
                                    entry_mode=entry_mode,
                                    judge_mode=judge_mode,
                                    up_pct=mfe_target, dn_pct=mae_limit,
                                    same_bar=same_bar)
        bps = result["buy_points"]
        if not bps:
            continue
        if judge_mode != "first_touch":
            for bp in bps:
                bp["success"] = (bp["mfe"] >= mfe_target and abs(bp["mae"]) <= mae_limit)
        success_count = sum(1 for bp in bps if bp["success"])
        success_rate = round(success_count / len(bps) * 100, 1)
        per_stock.append({
            "symbol": sym, "buy_point_count": len(bps),
            "success_rate": success_rate, "stats": result["stats"],
        })
        all_buy_points.extend(bps)

    if not all_buy_points:
        from backtest import _empty_bp_stats
        return {"strategy": strategy_name, "aggregate": _empty_bp_stats(),
                "multi_window": [], "per_stock": [], "stock_count": len(stock_cache)}

    total_n = len(all_buy_points)
    total_success = sum(1 for bp in all_buy_points if bp["success"])
    mfes = [bp["mfe"] for bp in all_buy_points]
    maes = [bp["mae"] for bp in all_buy_points]
    rs = [bp["r_ratio"] for bp in all_buy_points]
    bars_to_mfe = [bp["bars_to_mfe"] for bp in all_buy_points]

    aggregate = {
        "total_buy_points": total_n,
        "success_rate": round(total_success / total_n * 100, 1),
        "avg_mfe": round(float(np.mean(mfes)), 2),
        "avg_mae": round(float(np.mean(maes)), 2),
        "median_mfe": round(float(np.median(mfes)), 2),
        "median_mae": round(float(np.median(maes)), 2),
        "avg_r_ratio": round(float(np.mean(rs)), 2),
        "avg_bars_to_mfe": round(float(np.mean(bars_to_mfe)), 1),
        "mfe_distribution": {
            ">30%": sum(1 for m in mfes if m > 30),
            "20-30%": sum(1 for m in mfes if 20 < m <= 30),
            "10-20%": sum(1 for m in mfes if 10 < m <= 20),
            "<10%": sum(1 for m in mfes if m <= 10),
        },
        "mae_distribution": {
            "<5%": sum(1 for m in maes if abs(m) < 5),
            "5-10%": sum(1 for m in maes if 5 <= abs(m) < 10),
            ">10%": sum(1 for m in maes if abs(m) >= 10),
        },
    }
    # 先到先判: 多少根之内分出胜负。批量这条路的 aggregate 是这里单独拼的,
    # 跟 routers/strategies.py 单策略那条是两份代码, 漏了就是表里一列「—」。
    _decided = [bp.get("bars_to_outcome") for bp in all_buy_points
                if bp.get("bars_to_outcome") is not None]
    if _decided:
        aggregate["decided_count"] = len(_decided)
        aggregate["decided_pct"] = round(len(_decided) / total_n * 100, 1)
        aggregate["median_bars_to_outcome"] = round(float(np.median(_decided)), 1)

    multi_window_results = []
    for w in multi_windows:
        w_bps = []
        for sym, (df, params) in stock_cache.items():
            r = analyze_buy_points(df, strategy_name, params_override=params,
                                   window=w, entry_mode=entry_mode,
                                   judge_mode=judge_mode, up_pct=mfe_target,
                                   dn_pct=mae_limit, same_bar=same_bar)
            if judge_mode != "first_touch":
                for bp in r["buy_points"]:
                    bp["success"] = (bp["mfe"] >= mfe_target and abs(bp["mae"]) <= mae_limit)
            w_bps.extend(r["buy_points"])
        if w_bps:
            w_success = sum(1 for bp in w_bps if bp["success"])
            multi_window_results.append({
                "window": w, "total_signals": len(w_bps),
                "success_rate": round(w_success / len(w_bps) * 100, 1),
                "avg_mfe": round(float(np.mean([bp["mfe"] for bp in w_bps])), 2),
                "avg_mae": round(float(np.mean([bp["mae"] for bp in w_bps])), 2),
                "avg_r_ratio": round(float(np.mean([bp["r_ratio"] for bp in w_bps])), 2),
            })
        else:
            multi_window_results.append({"window": w, "total_signals": 0, "success_rate": 0,
                                         "avg_mfe": 0, "avg_mae": 0, "avg_r_ratio": 0})

    best_w = max(multi_window_results, key=lambda x: x["success_rate"])
    aggregate["best_window"] = best_w["window"]

    return {"strategy": strategy_name, "aggregate": aggregate,
            "multi_window": multi_window_results, "per_stock": per_stock,
            "stock_count": len(stock_cache)}


def _sp_analysis_worker(args):
    """模块级函数：对单个策略做 SELL 点分析（含多窗口）—— ProcessPool 用。"""
    import pickle, numpy as np
    from backtest import analyze_sell_points

    strategy_name, stock_data, multi_windows, window, mfe_target, mae_limit, interval, *_extra = args
    judge_mode = _extra[0] if len(_extra) > 0 else "mfe_mae"
    same_bar = _extra[1] if len(_extra) > 1 else "adverse_first"

    stock_cache = {}
    for sym, df_bytes, params in stock_data:
        stock_cache[sym] = (pickle.loads(df_bytes), params)

    per_stock = []
    all_sell_points = []
    for sym, (df, params) in stock_cache.items():
        result = analyze_sell_points(df, strategy_name, params_override=params, window=window,
                                     judge_mode=judge_mode, down_pct=mfe_target,
                                     up_pct=mae_limit, same_bar=same_bar)
        sps = result["sell_points"]
        if not sps: continue
        if judge_mode != "first_touch":
            for sp in sps:
                sp["success"] = (sp["mfe"] >= mfe_target and sp["mae"] <= mae_limit)
        success_count = sum(1 for sp in sps if sp["success"])
        per_stock.append({"symbol": sym, "sell_point_count": len(sps),
                          "success_rate": round(success_count / len(sps) * 100, 1), "stats": result["stats"]})
        all_sell_points.extend(sps)

    if not all_sell_points:
        from backtest import _empty_sp_stats
        return {"strategy": strategy_name, "aggregate": _empty_sp_stats(),
                "multi_window": [], "per_stock": [], "stock_count": len(stock_cache)}

    total_n = len(all_sell_points)
    total_success = sum(1 for sp in all_sell_points if sp["success"])
    mfes = [sp["mfe"] for sp in all_sell_points]
    maes = [sp["mae"] for sp in all_sell_points]
    rs = [sp["r_ratio"] for sp in all_sell_points]

    aggregate = {
        "total_sell_points": total_n,
        "success_rate": round(total_success / total_n * 100, 1),
        "avg_mfe": round(float(np.mean(mfes)), 2),
        "avg_mae": round(float(np.mean(maes)), 2),
        "avg_r_ratio": round(float(np.mean(rs)), 2),
    }
    # 先到先判: 多少根之内分出胜负 (跟买点那边同一个道理)
    _decided = [sp.get("bars_to_outcome") for sp in all_sell_points
                if sp.get("bars_to_outcome") is not None]
    if _decided:
        aggregate["decided_count"] = len(_decided)
        aggregate["decided_pct"] = round(len(_decided) / total_n * 100, 1)
        aggregate["median_bars_to_outcome"] = round(float(np.median(_decided)), 1)

    multi_window_results = []
    for w in multi_windows:
        w_sps = []
        for sym, (df, params) in stock_cache.items():
            r = analyze_sell_points(df, strategy_name, params_override=params, window=w,
                                    judge_mode=judge_mode, down_pct=mfe_target,
                                    up_pct=mae_limit, same_bar=same_bar)
            if judge_mode != "first_touch":
                for sp in r["sell_points"]:
                    sp["success"] = (sp["mfe"] >= mfe_target and sp["mae"] <= mae_limit)
            w_sps.extend(r["sell_points"])
        if w_sps:
            w_success = sum(1 for sp in w_sps if sp["success"])
            multi_window_results.append({
                "window": w, "total_signals": len(w_sps),
                "success_rate": round(w_success / len(w_sps) * 100, 1),
                "avg_mfe": round(float(np.mean([sp["mfe"] for sp in w_sps])), 2),
                "avg_mae": round(float(np.mean([sp["mae"] for sp in w_sps])), 2),
                "avg_r_ratio": round(float(np.mean([sp["r_ratio"] for sp in w_sps])), 2),
            })
        else:
            multi_window_results.append({"window": w, "total_signals": 0, "success_rate": 0,
                                         "avg_mfe": 0, "avg_mae": 0, "avg_r_ratio": 0})

    best_w = max(multi_window_results, key=lambda x: x["success_rate"])
    aggregate["best_window"] = best_w["window"]

    return {"strategy": strategy_name, "aggregate": aggregate,
            "multi_window": multi_window_results, "per_stock": per_stock,
            "stock_count": len(stock_cache)}


def _combo_opt_worker(args):
    """模块级函数：组合优化单任务（可被ProcessPool序列化）。
    args = (label, strats, symbols, market, period, interval)
    """
    import numpy as np
    label, strats, symbols, market, period, interval = args
    results = []
    for sym in symbols:
        r = _combo_bt_one((sym, market, strats, period, interval))
        r = _clean(r)
        if r.get("summary"):
            results.append(r)
    if not results:
        return {"added": label, "error": "无有效结果"}
    n = len(results)
    total_capital = n * 100000
    total_final = sum(r["summary"]["final_equity"] for r in results)
    total_return = (total_final - total_capital) / total_capital * 100
    total_trades = sum(r["summary"].get("trade_count", 0) for r in results)
    wins = sum(r["summary"].get("wins", 0) for r in results)
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

    eq_map = {}
    for r in results:
        for pt in r.get("equity_curve", []):
            d = pt["date"]
            eq_map[d] = eq_map.get(d, 0) + pt["equity"]
    eq_by_date = [v for _, v in sorted(eq_map.items())]
    sharpe = 0.0
    max_dd = 0.0
    if len(eq_by_date) > 2:
        eq_arr = np.array(eq_by_date, dtype=float)
        daily_rets = np.diff(eq_arr) / eq_arr[:-1]
        std = np.std(daily_rets)
        sharpe = float((np.mean(daily_rets) / std * np.sqrt(252)) if std > 0 else 0)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        max_dd = float(np.min(dd))

    per_stock_summary = []
    for r in results:
        sm = r.get("summary", {})
        per_stock_summary.append({
            "symbol": r.get("symbol", ""),
            "total_return": sm.get("total_return", 0),
            "trade_count": sm.get("trade_count", 0),
            "win_rate": sm.get("win_rate", 0),
            "final_equity": sm.get("final_equity", 0),
            "total_commission": sm.get("total_commission", 0),
        })

    return {
        "added": label,
        "strategies": strats,
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": win_rate,
        "total_trades": total_trades,
        "num_stocks": n,
        "final_equity": round(total_final, 2),
        "per_stock": per_stock_summary,
    }


def _combo_pair_worker(args):
    """模块级函数：两两组合单对任务（可被ProcessPool序列化）。
    args = (strats, symbols, market, period, interval)
    """
    import numpy as np
    strats, symbols, market, period, interval = args
    results = []
    for sym in symbols:
        r = _combo_bt_one((sym, market, strats, period, interval))
        r = _clean(r)
        if r.get("summary"):
            results.append(r)
    if not results:
        return {"pair": strats, "error": "无有效结果"}
    n = len(results)
    total_capital = n * 100000
    total_final = sum(r["summary"]["final_equity"] for r in results)
    total_return = (total_final - total_capital) / total_capital * 100
    total_trades = sum(r["summary"].get("trade_count", 0) for r in results)
    wins = sum(r["summary"].get("wins", 0) for r in results)
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

    eq_map = {}
    for r in results:
        for pt in r.get("equity_curve", []):
            d = pt["date"]
            eq_map[d] = eq_map.get(d, 0) + pt["equity"]
    eq_by_date = [v for _, v in sorted(eq_map.items())]
    sharpe = 0.0
    max_dd = 0.0
    if len(eq_by_date) > 2:
        eq_arr = np.array(eq_by_date, dtype=float)
        daily_rets = np.diff(eq_arr) / eq_arr[:-1]
        std = np.std(daily_rets)
        sharpe = float((np.mean(daily_rets) / std * np.sqrt(252)) if std > 0 else 0)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        max_dd = float(np.min(dd))

    return {
        "pair": strats,
        "strategy_a": strats[0],
        "strategy_b": strats[1],
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": win_rate,
        "total_trades": total_trades,
        "num_stocks": n,
        "final_equity": round(total_final, 2),
    }


def _wf_one_window(args):
    """模块级函数：Walk-Forward单窗口回测（可被ProcessPool序列化）"""
    train_df, test_df, sname, wp, dyn = args
    try:
        tr = run_strategy_backtest(train_df, sname, params_override=wp, dynamic_position=dyn)
        train_sharpe = tr["summary"]["sharpe"]
    except Exception:
        train_sharpe = 0
    try:
        te = run_strategy_backtest(test_df, sname, params_override=wp, dynamic_position=dyn)
        test_summary = te["summary"]
    except Exception:
        test_summary = {"sharpe": 0, "total_return": 0, "max_drawdown": 0, "win_rate": 0, "total_trades": 0}
    period_start = str(test_df.index[0].date())
    period_end = str(test_df.index[-1].date())
    return {
        "period": f"{period_start} → {period_end}",
        "train_sharpe": round(train_sharpe, 2),
        "test_sharpe": round(test_summary["sharpe"], 2),
        "test_return": round(test_summary["total_return"], 2),
        "test_max_dd": round(test_summary["max_drawdown"], 2),
        "test_win_rate": round(test_summary["win_rate"], 2),
        "test_trades": test_summary["total_trades"],
    }


def _wf_one_window_retrain(args):
    """模块级函数：Walk-Forward 单窗口「每窗口重新优化」版 (可被 ProcessPool 序列化)。

    与 _wf_one_window 的区别: 这里先在训练段跑一次参数优化, 再用优化出来的参数
    做训练/测试回测。这是 param_source=retrain_* 三个选项走的路, 也是最贵的一条
    —— 每个窗口一次贝叶斯/穷举。之前这条分支是串行 for 循环, 而紧挨着的
    非 retrain 分支早就用 pool.map 并行了, 并行化时只改了一半。
    """
    from optimizer import optimize_strategy   # main 模块级没导, 跟其它调用点保持一致
    train_df, test_df, sname, method, grid, dyn, n_trials, default_params = args
    window_params, train_combos = None, 0
    try:
        bp, bs, bsum, combos, mu = optimize_strategy(
            train_df, sname, method=method, grid=grid,
            dynamic_pos=dyn, n_trials=n_trials
        )
        if bp:
            window_params = {**default_params, **bp}
            train_combos = combos
    except Exception:
        pass

    try:
        tr = run_strategy_backtest(train_df, sname, params_override=window_params,
                                   dynamic_position=dyn)
        train_sharpe = tr["summary"]["sharpe"]
    except Exception:
        train_sharpe = 0
    try:
        te = run_strategy_backtest(test_df, sname, params_override=window_params,
                                   dynamic_position=dyn)
        test_summary = te["summary"]
    except Exception:
        test_summary = {"sharpe": 0, "total_return": 0, "max_drawdown": 0,
                        "win_rate": 0, "total_trades": 0}

    wr = {
        "period": f"{str(test_df.index[0].date())} → {str(test_df.index[-1].date())}",
        "train_sharpe": round(train_sharpe, 2),
        "test_sharpe": round(test_summary["sharpe"], 2),
        "test_return": round(test_summary["total_return"], 2),
        "test_max_dd": round(test_summary["max_drawdown"], 2),
        "test_win_rate": round(test_summary["win_rate"], 2),
        "test_trades": test_summary["total_trades"],
    }
    if train_combos:
        wr["combos_tested"] = train_combos
    return wr


WF_MAX_WINDOWS = int(os.getenv("AITRADER_WF_MAX_WINDOWS", "20"))  # <=0 表示不限


#: 去过拟合: 每轮搜索的 trial 数。比普通优化少 —— 每个 trial 要在所有搜索窗口
#: 上各跑一次回测, 单次成本是普通优化的 N 倍(N=窗口数)。
DEOVERFIT_TRIALS = int(os.getenv("AITRADER_DEOVERFIT_TRIALS", "40"))
#: 最多重试几轮。"直到稳健"不能真的无限循环 —— 有的策略在这只股票上就是不行,
#: 硬凑只会凑出对 WF 窗口的过拟合。轮数用完还不稳健就如实报告。
DEOVERFIT_MAX_ROUNDS = int(os.getenv("AITRADER_DEOVERFIT_MAX_ROUNDS", "3"))
#: 留几个窗口不给搜索看。**这个不能省**: 拿 WF 窗口当优化目标, 本身就是在对这些
#: 窗口过拟合, 跑出来的"稳健"有自我实现的成分。留最近 N 个窗口作为搜索过程完全
#: 看不见的持出集, 最后用它判定才算数。
DEOVERFIT_HOLDOUT = int(os.getenv("AITRADER_DEOVERFIT_HOLDOUT", "2"))


def _deoverfit_score(test_sharpes) -> tuple:
    """(一致性%, 平均sharpe) —— 去过拟合的评分口径。

    跟 _summarize_wf_results 判 verdict 用的是同一个口径(test_sharpe > 0 的窗口
    占比), 这样"优化目标"和"验收标准"才是一回事。否则优化半天, 判定用的是另一把尺。
    """
    if not test_sharpes:
        return 0.0, 0.0
    cons = sum(1 for x in test_sharpes if x > 0) / len(test_sharpes) * 100.0
    return cons, float(np.mean(test_sharpes))


def _deoverfit_eval(windows_bytes, sname, params, dyn):
    """在给定窗口上只跑**测试段**, 返回每个窗口的 test_sharpe。

    只跑测试段不跑训练段: 目标函数只关心样本外表现, 训练段的 sharpe 是给人看的,
    这里跑它纯属浪费(成本直接翻倍)。
    """
    import pickle as _pk
    windows = _pk.loads(windows_bytes) if isinstance(windows_bytes, (bytes, bytearray)) else windows_bytes
    out = []
    for _tr, te in windows:
        try:
            r = run_strategy_backtest(te, sname, params_override=params, dynamic_position=dyn)
            out.append(float(r["summary"]["sharpe"]))
        except Exception:
            out.append(0.0)
    return out


def _deoverfit_one_strategy(args):
    """对一个策略做"以样本外一致性为目标"的参数搜索。供 ProcessPool 调用。

    跟普通优化的根本区别:
      普通优化 → 目标是**样本内**得分(整段行情上的信号质量/sharpe), WF 只是事后检验。
                 过拟合的本质就是样本内好、样本外崩 —— 用样本内当目标, 天然招过拟合。
      去过拟合 → 目标直接就是**样本外一致性**(WF 各测试段里 sharpe>0 的占比)。
                 选参数的依据从"在见过的数据上多好"换成"在没见过的数据上多稳"。

    方法学上的坑和处理: 拿 WF 窗口当目标, 就把这些窗口变成了训练数据, 结果会
    自我实现。所以最近 DEOVERFIT_HOLDOUT 个窗口搜索全程看不见, 最后单独拿它们
    验一次, 两个数都返回给调用方 —— 差得多说明还是过拟合了, 只是换了个对象。
    """
    import pickle as _pk
    (windows_bytes, sname, dyn, n_trials, holdout_n, base_params, round_idx, prev_best) = args
    windows = _pk.loads(windows_bytes)
    n_all = len(windows)
    if n_all < 3:
        return {"sname": sname, "ok": False, "reason": f"窗口只有 {n_all} 个, 不够拆出持出集"}

    holdout_n = max(1, min(holdout_n, n_all // 3))
    search_w = windows[:-holdout_n]
    hold_w = windows[-holdout_n:]
    search_bytes = _pk.dumps(search_w)
    hold_bytes = _pk.dumps(hold_w)

    try:
        from optimizer import PARAM_RANGES, _infer_param_ranges
        ranges = PARAM_RANGES.get(sname) or _infer_param_ranges(sname)
    except Exception as e:
        return {"sname": sname, "ok": False, "reason": f"取参数范围失败: {e}"}
    if not ranges:
        return {"sname": sname, "ok": False, "reason": "这个策略没有可调参数"}

    # 第 2 轮起围着上一轮的最好参数收窄搜索范围 —— 不收窄的话每轮都是重新乱撞,
    # "多跑几轮"并不会更接近稳健。
    if round_idx > 1 and prev_best:
        narrowed = {}
        shrink = 0.5 ** (round_idx - 1)
        for pname, (low, high, step, ptype) in ranges.items():
            c = prev_best.get(pname)
            if c is None or ptype == "categorical":
                narrowed[pname] = (low, high, step, ptype)
                continue
            half = (high - low) * shrink / 2.0
            nlow, nhigh = max(low, c - half), min(high, c + half)
            if nhigh <= nlow:
                nlow, nhigh = low, high
            if ptype == "int":
                nlow, nhigh = int(round(nlow)), int(round(nhigh))
                if nhigh <= nlow:
                    nlow, nhigh = int(low), int(high)
            narrowed[pname] = (nlow, nhigh, step, ptype)
        ranges = narrowed

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception as e:
        return {"sname": sname, "ok": False, "reason": f"optuna 不可用: {e}"}

    def objective(trial):
        p = dict(base_params or {})
        for pname, (low, high, step, ptype) in ranges.items():
            if ptype == "int":
                p[pname] = trial.suggest_int(pname, int(low), int(high), step=max(1, int(step)))
            elif ptype == "float":
                p[pname] = round(trial.suggest_float(pname, float(low), float(high), step=step), 4)
            elif ptype == "categorical":
                p[pname] = trial.suggest_categorical(pname, low)
        sharpes = _deoverfit_eval(search_bytes, sname, p, dyn)
        cons, avg = _deoverfit_score(sharpes)
        # 一致性为主; 平均 sharpe 只用来打破平局并偏向真赚钱的(截到 3 避免被
        # 某一个窗口的极端值带跑)。
        return cons + min(max(avg, -3.0), 3.0) * 5.0

    try:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42 + round_idx))
        study.optimize(objective, n_trials=max(5, int(n_trials)), show_progress_bar=False)
        best_params = {**dict(base_params or {}), **study.best_params}
    except Exception as e:
        return {"sname": sname, "ok": False, "reason": f"搜索失败: {type(e).__name__}: {e}"}

    s_search = _deoverfit_eval(search_bytes, sname, best_params, dyn)
    s_hold = _deoverfit_eval(hold_bytes, sname, best_params, dyn)
    c_search, a_search = _deoverfit_score(s_search)
    c_hold, a_hold = _deoverfit_score(s_hold)
    s_all = s_search + s_hold
    c_all, a_all = _deoverfit_score(s_all)

    return {
        "sname": sname, "ok": True, "params": best_params,
        "round": round_idx,
        "trials": len(study.trials),
        "search_windows": len(search_w), "holdout_windows": len(hold_w),
        "search_consistency": round(c_search, 1), "search_avg_sharpe": round(a_search, 2),
        # 持出集上的数 —— 搜索全程没见过, 这个才是真正能信的
        "holdout_consistency": round(c_hold, 1), "holdout_avg_sharpe": round(a_hold, 2),
        "all_consistency": round(c_all, 1), "all_avg_sharpe": round(a_all, 2),
    }


def _build_wf_plan(df: pd.DataFrame, train_years: int = 3, test_months: int = 6,
                   max_windows: Optional[int] = None) -> dict:
    """统一构建 WF 窗口。

    数据足够时使用标准滚动窗口；数据偏短时改用 anchored expanding windows，
    避免简单地把训练/测试窗口机械压得过小。

    max_windows: 窗口数上限，只保留**最近的** N 个；None 用全局 WF_MAX_WINDOWS，
    <=0 不限。为什么要有上限：窗口数随历史长度线性增长，而成本也线性增长。
    实测 AMD 有 11720 根日线(46 年)，按 3 年训练 / 6 月测试滚出 87 个窗口，
    retrain 模式下单窗口一次贝叶斯要 146 秒 —— 一个策略就是 3.5 小时，
    前端还要逐个跑 22 个策略。而最早那些窗口的测试段落在 1980 年代，
    拿那个年代的行情验证今天的策略参数，参考价值存疑，成本却照付。
    默认 20 个窗口 = 最近约 10 年的滚动测试覆盖（标准模式下步长为一个测试段）。
    """
    total_bars = len(df)
    ideal_train = max(int(train_years * 252), 60)
    ideal_test = max(int(test_months * 21), 15)

    if total_bars < 90:
        return {
            "ok": False,
            "reason": f"数据不足({total_bars}条)，无法进行可靠的WF验证",
            "total_bars": total_bars,
        }

    windows = []
    mode = "standard"
    auto_adjusted = False
    train_days = ideal_train
    test_days = ideal_test

    if total_bars >= ideal_train + ideal_test * 2:
        idx = 0
        while idx + train_days + test_days <= total_bars:
            windows.append((
                df.iloc[idx:idx + train_days].copy(),
                df.iloc[idx + train_days:idx + train_days + test_days].copy()
            ))
            idx += test_days
    else:
        auto_adjusted = True
        if total_bars >= 252 + 21 * 2:
            mode = "anchored_compact"
            test_days = max(21, min(ideal_test, total_bars // 5))
            train_days = max(252, min(ideal_train, total_bars - test_days * 2))
        else:
            mode = "anchored_micro"
            test_days = max(15, min(21, total_bars // 6))
            train_days = max(90, min(max(total_bars - test_days * 2, 90), ideal_train))

        split = train_days
        while split + test_days <= total_bars:
            windows.append((
                df.iloc[:split].copy(),
                df.iloc[split:split + test_days].copy()
            ))
            split += test_days

    if not windows:
        return {
            "ok": False,
            "reason": f"数据不足({total_bars}条)，无法构建有效WF窗口",
            "total_bars": total_bars,
        }

    # 只留最近的 N 个窗口（WF 是前推验证，近期的市场状态才是要验证的对象）
    cap = WF_MAX_WINDOWS if max_windows is None else int(max_windows)
    windows_available = len(windows)
    windows_trimmed = 0
    if cap > 0 and windows_available > cap:
        windows = windows[-cap:]
        windows_trimmed = windows_available - cap

    num_windows = len(windows)
    if mode == "standard" and num_windows >= 3 and train_days >= 504 and test_days >= 63:
        reliability = "high"
    elif num_windows >= 2 and train_days >= 126 and test_days >= 15:
        reliability = "medium" if mode == "standard" else "low"
    else:
        reliability = "low"

    note = None
    if auto_adjusted:
        note = (
            f"历史仅{total_bars}条，WF改为{mode}模式，"
            f"训练{train_days}天 / 测试{test_days}天 / {num_windows}个窗口"
        )
    if windows_trimmed:
        first_test = windows[0][1]
        since = str(first_test.index[0].date()) if len(first_test) else "?"
        trim_note = (f"历史可切出 {windows_available} 个窗口，只跑最近 {num_windows} 个"
                     f"（测试段自 {since} 起）；早期窗口对当下参数参考有限，"
                     f"且窗口数与耗时成正比。要全部跑设 AITRADER_WF_MAX_WINDOWS=0")
        note = f"{note}；{trim_note}" if note else trim_note

    if reliability == "low":
        low_note = "样本偏短，结果只用于排除明显过拟合，不建议直接升级标准参数"
        note = f"{note}；{low_note}" if note else low_note

    return {
        "ok": True,
        "total_bars": total_bars,
        "ideal_train": ideal_train,
        "ideal_test": ideal_test,
        "train_days": train_days,
        "test_days": test_days,
        "num_windows": num_windows,
        "windows_available": windows_available,
        "windows_trimmed": windows_trimmed,
        "max_windows": cap,
        "mode": mode,
        "reliability": reliability,
        "auto_adjusted": auto_adjusted,
        "note": note,
        "windows": windows,
    }


def _summarize_wf_results(wf_results: list, plan: dict) -> dict:
    """统一汇总 WF 结果并输出 verdict/promoted。"""
    if not wf_results:
        return {
            "verdict": "overfit",
            "consistency": 0.0,
            "avg_test_sharpe": 0.0,
            "avg_test_return": 0.0,
            "avg_test_max_dd": 0.0,
            "num_windows": 0,
            "promoted": False,
            "window_mode": plan.get("mode"),
            "reliability": plan.get("reliability", "low"),
            "note": plan.get("note"),
        }

    consistency = sum(1 for r in wf_results if r["test_sharpe"] > 0) / len(wf_results) * 100
    avg_sharpe = float(np.mean([r["test_sharpe"] for r in wf_results]))
    avg_return = float(np.mean([r["test_return"] for r in wf_results]))
    avg_dd = float(np.mean([r["test_max_dd"] for r in wf_results]))
    reliability = plan.get("reliability", "low")

    if consistency >= 65 and avg_sharpe > 0 and reliability in ("high", "medium"):
        verdict = "robust"
    elif consistency >= 50 or (consistency >= 65 and reliability == "low"):
        verdict = "fair"
    else:
        verdict = "overfit"

    promoted = verdict == "robust" and reliability in ("high", "medium")
    return {
        "verdict": verdict,
        "consistency": round(float(consistency), 1),
        "avg_test_sharpe": round(float(avg_sharpe), 2),
        "avg_test_return": round(float(avg_return), 2),
        "avg_test_max_dd": round(float(avg_dd), 2),
        "num_windows": len(wf_results),
        "promoted": promoted,
        "window_mode": plan.get("mode"),
        "reliability": reliability,
        "note": plan.get("note"),
    }


def _compare_rank_bt_one(args):
    """模块级函数：/compare 排行榜的单次回测 (可被 ProcessPool 序列化)。

    注意别跟 _compare_bt_one 搞混 —— 那个是 /compare-optimized 用的
    「标准 vs 优化参数」对比, 名字像但参数格式和返回都不一样。

    参数在主进程里就解析好了 —— worker 不碰 custom_params/universal_params 这些
    文件, 免得几十个进程同时抢同一批 JSON。df 也在主进程 pickle 一次, 这里只解包。

    kind: "main" = 排行榜那一轮; "orig" = 算「升级前后差多少」的通用标准那一轮。
    """
    kind, sname, df_bytes, buy_params, sell_params, params_override, dyn_pos, allow_short = args
    try:
        import pickle as _pk
        df = _pk.loads(df_bytes)
        if buy_params is not None or sell_params is not None:
            r = run_strategy_backtest(df, sname,
                                      buy_params=buy_params, sell_params=sell_params,
                                      dynamic_position=dyn_pos, allow_short=allow_short)
        else:
            r = run_strategy_backtest(df, sname, params_override=params_override,
                                      dynamic_position=dyn_pos, allow_short=allow_short)
        return kind, sname, _clean(r), None
    except Exception as e:
        return kind, sname, None, str(e)


def _bt_one_applicability(args):
    """模块级函数：策略适用性单任务回测（可被ProcessPool序列化）"""
    sym, df, sname, dyn_pos = args
    try:
        r = run_strategy_backtest(df, sname, dynamic_position=dyn_pos)
        sm = r["summary"]
        return sym, sname, {
            "sharpe": sm["sharpe"],
            "return": sm["total_return"],
            "max_dd": sm["max_drawdown"],
            "trades": sm["total_trades"],
            "win_rate": sm["win_rate"],
            "profit_factor": sm.get("profit_factor", 0),
            "score": composite_score(
                sm["total_return"], sm["sharpe"], sm["max_drawdown"],
                sm["win_rate"], sm.get("profit_factor", 0)
            ),
        }
    except Exception:
        return sym, sname, {"sharpe": 0, "return": 0, "max_dd": 0, "trades": 0,
                            "win_rate": 0, "profit_factor": 0, "score": 0}


def _run_wf_validation_from_plan(symbol: str, opt_strategies: dict, dynamic_pos: bool, wf_plan: dict,
                                 task_id: str = "", slot_stage: str = "wf") -> dict:
    """基于已构建好的 WF 计划执行验证，避免重复拉取/切分数据。

    task_id: 传了就往统一进度表(_prog_tasks)按"策略×窗口"报进度。
    原先只有 opt_slot_progress 一条通道 —— 那是**优化槽位**专用的, 非优化
    调用方(比如 /compare-optimized)根本没持有槽位, 报了等于没报, 于是这一段
    (往往是整个请求里最慢的) 对前端完全是黑的。更要命的是僵尸回收器认的是
    _prog_tasks 里的 updated_at: 5 分钟不动就判 error —— WF 跑满 5 分钟会被
    误判成失败, 而它其实好好的。
    """
    from concurrent.futures import ProcessPoolExecutor as _WfPool, as_completed as _wf_ac
    import gc

    wf_validation = {}
    try:
        if not wf_plan.get("ok"):
            return wf_validation
        wf_windows = wf_plan["windows"]

        # 所有策略×窗口 扁平化到一个 ProcessPool
        all_tasks = []  # [(sname, task_args), ...]
        for sname, sdata in opt_strategies.items():
            opt_params = sdata.get("params")
            if not opt_params:
                continue
            for tr, te in wf_windows:
                all_tasks.append((sname, (tr, te, sname, opt_params, dynamic_pos)))

        if not all_tasks:
            return wf_validation

        # 收集: { sname: [result, ...] }
        wf_raw = {}
        n_wf = min(N_WORKERS, len(all_tasks))
        pool = _WfPool(max_workers=n_wf, initializer=proc_role.mark_pool_worker)
        # 进度: WF 是长尾, 按"策略×窗口"报, 不然用户看不出还要多久
        # stage 以前写死 "wf" —— 这个函数一次优化里会被调两遍(最优参数 +
        # 标准参数对照), 第二遍把 stage 改回 wf 就会让调用方刚分好的步骤失效,
        # 面板上表现为进度条从 40% 掉回 0%。
        opt_slot_progress(stage=slot_stage, wf_total=len(all_tasks), wf_done=0, wf_workers=n_wf)
        _wf_tr = ProgTracker(task_id, [sn for sn, _ in all_tasks], n_wf,
                             unit=" 个窗口") if task_id else None
        _wf_cancelled = False
        try:
            futures = {}
            for sname, task_args in all_tasks:
                fut = pool.submit(_wf_one_window, task_args)
                futures[fut] = sname

            _wf_done = 0
            for fut in _wf_ac(futures):
                # 取消检查点。少了它, ⏹ 在**最慢的这一步**完全失效:
                # 旗子插上了, 但这个循环只调 done_one, 而 done_one 每次推进度
                # 都带 note="" —— 连"正在停止…"那句提示都被立刻擦掉。WF 跑完
                # 之后端点照常 prog_end("done"), 前端看到的是正常结果,
                # progCancelled 守卫也不触发 —— 用户点了停止, 什么反馈都没有。
                if task_id and prog_is_cancelled(task_id):
                    _wf_cancelled = True
                    break
                sname = futures[fut]
                try:
                    res = fut.result()
                    wf_raw.setdefault(sname, []).append(res)
                except Exception:
                    pass
                _wf_done += 1
                opt_slot_progress(wf_done=_wf_done, current=sname)
                if _wf_tr is not None:
                    _wf_tr.done_one(sname)
        finally:
            # 被取消时绝不能 wait=True: 那会一直等到所有在跑的窗口算完
            # (每个窗口几十秒), "停止"变成"再等一分钟"。正常结束才等。
            if _wf_cancelled:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    pool.shutdown(wait=False)
            else:
                pool.shutdown(wait=True)
            del pool, futures
            gc.collect()

        if _wf_cancelled:
            prog_end(task_id, "cancelled", "已按请求停止")
            raise TaskCancelled()

        # 汇总每个策略的结果
        for sname, wf_results in wf_raw.items():
            try:
                wf_summary = _summarize_wf_results(wf_results, wf_plan)
                wf_validation[sname] = wf_summary
                logger.info(
                    f"WF validation {symbol}/{sname}: {wf_summary['verdict']} "
                    f"(consistency={wf_summary['consistency']:.1f}%, "
                    f"reliability={wf_summary['reliability']})"
                )
            except Exception as e:
                logger.warning(f"WF validation for {sname} failed: {e}")

    except TaskCancelled:
        raise          # 用户主动停的, 不是"WF 跳过", 别被下面的兜底吞掉
    except Exception as e:
        logger.warning(f"WF validation skipped for {symbol}: {e}")

    return wf_validation


#: WF 验证专用的取数窗口。不跟优化窗口共用 —— 见 _wf_period_for_interval 的说明。
WF_PERIOD_BY_INTERVAL = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "2y",
    "1d": "max", "1wk": "max", "1mo": "max",
}


def _wf_period_for_interval(interval: str) -> str:
    """WF 验证该取多长的历史。

    为什么不能直接复用优化用的 period: _optimization_period_for_interval 给日线
    返回 "1y"(252 根)。而 WF 的标准窗口是 3 年训练 + 6 月测试, 至少要 756+126 根。
    252 根根本切不出标准窗口, _build_wf_plan 会降级成 anchored_micro
    (训练 210 天 / 测试 21 天 / 只有 2 个窗口), reliability 直接判成 low,
    再加上测试段只有 21 根、常常一笔交易都没有 → test_sharpe=0 →
    consistency=0 → verdict 一律 overfit, 没有任何策略能升级。
    实测 MP: 用 1y 取数是 252 根 2 个窗口全 overfit; 用 max 是 1568 根 6 个窗口,
    夏普 -1.52 到 +2.28 都有, 是真实判定。
    窗口数上限由 WF_MAX_WINDOWS 管, 所以取 max 不会让耗时失控。
    """
    return WF_PERIOD_BY_INTERVAL.get(_normalize_param_interval(interval), "max")


def _run_wf_validation(symbol: str, market: str, opt_strategies: dict, dynamic_pos: bool,
                       interval: str = "1d", period: Optional[str] = None,
                       task_id: str = "") -> dict:
    """
    对每个已优化策略跑 Walk-Forward 验证，返回 {strategy_name: {verdict, consistency, ...}}
    verdict: "robust" (>=65%), "fair" (>=50%), "overfit" (<50%)
    所有策略×窗口 统一提交到一个 ProcessPool，避免逐策略创建/销毁进程池的开销。

    period=None (默认) 表示用 WF 自己的取数窗口, 不跟着优化窗口走。
    """
    interval = _normalize_param_interval(interval)
    wf_period = period or _wf_period_for_interval(interval)
    try:
        try:
            wf_df = get_ohlcv(symbol, market, interval=interval, period=wf_period)
        except Exception:
            wf_df = get_ohlcv(symbol, market, interval=interval, period=wf_period)
        wf_plan = _build_wf_plan(wf_df, train_years=3, test_months=6)
        return _run_wf_validation_from_plan(symbol, opt_strategies, dynamic_pos, wf_plan,
                                           task_id=task_id)
    except TaskCancelled:
        raise
    except Exception as e:
        logger.warning(f"WF validation skipped for {symbol}: {e}")
        return {}


def _signal_baseline(df, sname, baseline_params, signal_type, so):
    """买/卖点专项优化的**优化前基线**: 拿当前在用的参数, 按**同一套口径**
    (同样的前瞻根数/阈值/判定方式)跑一遍, 好让用户看到"优化前 vs 优化后"。

    不算基线的话, 结果表里只有一个孤零零的"成功率 44%", 没法判断这次优化
    到底有没有用 —— 也许原来就是 44%, 也许原来是 60% 被调差了。
    """
    if signal_type not in ("buy", "sell") or not baseline_params:
        return None
    so = so or {}
    jm = so.get("judge_mode", "mfe_mae")
    sb = so.get("same_bar", "adverse_first")
    w = so.get("window", 20)
    try:
        if signal_type == "buy":
            from backtest import analyze_buy_points
            r = analyze_buy_points(df, sname, params_override=baseline_params, window=w,
                                   entry_mode=so.get("entry_mode", "next_open"),
                                   judge_mode=jm, up_pct=so.get("mfe_target", 15),
                                   dn_pct=so.get("mae_limit", 10), same_bar=sb)
            pts = r["buy_points"]
            if jm != "first_touch":
                for b in pts:
                    b["success"] = (b["mfe"] >= so.get("mfe_target", 15)
                                    and abs(b["mae"]) <= so.get("mae_limit", 10))
            key = "total_buy_points"
        else:
            from backtest import analyze_sell_points
            r = analyze_sell_points(df, sname, params_override=baseline_params, window=w,
                                    judge_mode=jm, down_pct=so.get("mfe_target", 10),
                                    up_pct=so.get("mae_limit", 5), same_bar=sb)
            pts = r["sell_points"]
            if jm != "first_touch":
                for b in pts:
                    b["success"] = (b["mfe"] >= so.get("mfe_target", 10)
                                    and b["mae"] <= so.get("mae_limit", 5))
            key = "total_sell_points"
        n = len(pts)
        if not n:
            return {key: 0, "success_rate": 0, "avg_mfe": 0, "avg_mae": 0, "avg_r_ratio": 0}
        import numpy as _np
        return {
            key: n,
            "success_rate": round(sum(1 for b in pts if b.get("success")) / n * 100, 1),
            "avg_mfe": round(float(_np.mean([b["mfe"] for b in pts])), 2),
            "avg_mae": round(float(_np.mean([b["mae"] for b in pts])), 2),
            "avg_r_ratio": round(float(_np.mean([b["r_ratio"] for b in pts])), 2),
        }
    except Exception as e:
        logger.warning(f"[baseline] {sname} 基线计算失败: {e}")
        return None


def _optimize_one_stock_strategy(args):
    """单策略优化 worker — 供个股优化并行调用
    args 元组末尾可选 signal_type（向后兼容）"""
    from optimizer import optimize_strategy
    # 元组末尾两项都是可选的(向后兼容): signal_type, 再往后是 sig_opts
    sig_opts = None
    if len(args) == 10:
        df, sname, method, grid, dynamic_pos, n_trials, run_all, methods_to_run, signal_type, sig_opts = args
    elif len(args) == 9:
        df, sname, method, grid, dynamic_pos, n_trials, run_all, methods_to_run, signal_type = args
    else:
        df, sname, method, grid, dynamic_pos, n_trials, run_all, methods_to_run = args
        signal_type = "buy_and_sell"
    strategy = get_strategy(sname)

    if run_all:
        # 三种方法全跑，取最优
        method_results = {}
        best_method = None
        best_sharpe_all = -999
        combos_total = 0

        for m in methods_to_run:
            if not grid and m in ("grid", "coarse_fine"):
                try:
                    r = run_strategy_backtest(df, sname, dynamic_position=dynamic_pos)
                    r = _clean(r)
                    method_results[m] = {
                        "params": strategy.default_params.copy(),
                        "sharpe": r["summary"]["sharpe"],
                        "total_return": r["summary"]["total_return"],
                        "max_drawdown": r["summary"]["max_drawdown"],
                        "win_rate": r["summary"]["win_rate"],
                        "total_trades": r["summary"]["total_trades"],
                        "combos_tested": 1,
                        "is_default": True,
                    }
                    if r["summary"]["sharpe"] > best_sharpe_all:
                        best_sharpe_all = r["summary"]["sharpe"]
                        best_method = m
                except:
                    method_results[m] = {"sharpe": -999, "error": True}
                continue

            try:
                bp, bs, bsum, combos, mu = optimize_strategy(
                    df, sname, method=m, grid=grid,
                    dynamic_pos=dynamic_pos, n_trials=n_trials,
                    signal_type=signal_type, sig_opts=sig_opts
                )
                combos_total += combos
                if bp and bsum:
                    full_params = {**strategy.default_params, **bp}
                    entry = {
                        "params": full_params,
                        "optimized_keys": list(bp.keys()),
                        "combos_tested": combos,
                        "signal_type": signal_type,
                        **bsum,  # 动态包含所有指标（buy/sell/combined各不同）
                    }
                    # 统一排序键
                    sort_val = bsum.get("sharpe", bsum.get("success_rate", bs or 0))
                    entry["_sort_val"] = sort_val
                    method_results[m] = entry
                    if sort_val > best_sharpe_all:
                        best_sharpe_all = sort_val
                        best_method = m
                else:
                    method_results[m] = {"_sort_val": -999, "error": True}
            except Exception as e:
                logger.error(f"Optimize {sname} [{m}] failed: {e}")
                method_results[m] = {"_sort_val": -999, "error": True}

        result_entry = None
        if best_method and best_method in method_results and not method_results[best_method].get("error"):
            winner = method_results[best_method]
            result_entry = {**winner, "method": best_method, "best_of": "all"}

        _skip = None
        if result_entry is None:
            _errs = [m for m, v in method_results.items() if v.get("error")]
            _skip = f"全部方法都没跑出结果({', '.join(_errs) or 'unknown'})"
            logger.warning(f"Optimize {sname}: {_skip}, 本次不保存")
        return {
            "sname": sname, "mode": "all",
            "result": result_entry,
            "method_comparison": method_results,
            "combos": combos_total,
            "skip_reason": _skip,
        }

    else:
        # 单方法模式
        if not grid and method == "grid":
            try:
                r = run_strategy_backtest(df, sname, dynamic_position=dynamic_pos)
                r = _clean(r)
                return {
                    "sname": sname, "mode": "single",
                    "result": {
                        "params": strategy.default_params.copy(),
                        "sharpe": r["summary"]["sharpe"],
                        "total_return": r["summary"]["total_return"],
                        "max_drawdown": r["summary"]["max_drawdown"],
                        "win_rate": r["summary"]["win_rate"],
                        "total_trades": r["summary"]["total_trades"],
                        "combos_tested": 1,
                        "is_default": True,
                        "method": "default",
                    },
                    "combos": 1,
                }
            except Exception as e:
                logger.warning(f"Optimize {sname}: 默认参数回测失败({type(e).__name__}), 本次不保存")
                return {"sname": sname, "mode": "single", "result": None, "combos": 0,
                        "skip_reason": f"默认参数回测失败: {type(e).__name__}"}

        try:
            best_params, best_score, best_summary, combos, method_used = optimize_strategy(
                df, sname, method=method, grid=grid,
                dynamic_pos=dynamic_pos, n_trials=n_trials,
                signal_type=signal_type, sig_opts=sig_opts
            )
            if best_params and best_summary:
                full_params = {**strategy.default_params, **best_params}
                result_entry = {
                    "params": full_params,
                    "optimized_keys": list(best_params.keys()),
                    "combos_tested": combos,
                    "method": method_used,
                    "signal_type": signal_type,
                    **best_summary,  # 动态包含所有指标
                }
                # 优化前基线 (同口径), 供前端做"优化前 vs 优化后"对比。
                # 样本外模式下 optimize_signal_oos 已经在**持出段**上算好了基线
                # (那才是跟它的 success_rate 可比的口径), 这里不能再用全历史的覆盖掉。
                if not result_entry.get("baseline"):
                    _base = _signal_baseline(df, sname, (sig_opts or {}).get("baseline_params"),
                                             signal_type, sig_opts)
                    if _base:
                        result_entry["baseline"] = _base
                return {
                    "sname": sname, "mode": "single",
                    "result": result_entry,
                    "combos": combos,
                }
        except Exception as e:
            logger.error(f"Optimize {sname} failed: {e}")
            return {"sname": sname, "mode": "single", "result": None,
                    "combos": 0, "skip_reason": f"{type(e).__name__}: {str(e)[:120]}"}

        # 走到这里 = optimize_strategy 正常返回了, 但 best_params 或 best_summary
        # 是空的。原来这条路径**一条日志都没有**, 上层再来一句
        #     if res.get("result"): results[...] = ...
        # 于是策略凭空消失: 进度条报"13 个完成", 存下来只有 7 个, 没人知道少了谁。
        # 样本外优化被闸门拒收时, best_summary 里带着真实原因 —— 别再报
        # 笼统的"优化器没返回参数", 用户会以为是程序出错。
        _rej = (best_summary or {}).get("reject_reason") if isinstance(best_summary, dict) else None
        _why = _rej or (best_summary or {}).get("error") if isinstance(best_summary, dict) else None
        _why = _why or ("优化器没返回参数" if not best_params else "优化器没返回回测指标")
        logger.warning(f"Optimize {sname}: {_why}, 本次不保存 (params={bool(best_params)}, summary={bool(best_summary)})")
        return {"sname": sname, "mode": "single", "result": None,
                "combos": combos if 'combos' in dir() else 0, "skip_reason": _why,
                "rejected_detail": best_summary if isinstance(best_summary, dict) else None}


# 注册表本身也用 LockedDict: 它会被遍历 (trading.py 的 "有没有任务在跑" 检查),
# 而写侧散在多个后台线程里 —— 之前 _run_new_strategy_batch_round 往里插就没拿
# _batch_lock, 单边锁等于没锁。换成容器自持锁, 谁插都安全, 不依赖调用方自觉。
_batch_jobs = LockedDict()   # job_id -> LockedDict
# _batch_lock 仍然需要, 但它管的是**另一件事**: "先查有没有在跑的, 再插一条"
# 这个 check-then-act 要整体互斥, 不能被两个并发请求交错(否则会起两个任务)。
_batch_lock = threading.Lock()



JOB_REGISTRY_MAX = int(os.getenv("AITRADER_JOB_REGISTRY_MAX", "40"))


def _reap_registry(registry, keep=None):
    """注册表只留最近 keep 条已结束的任务。

    在跑的一律不动 —— 哪怕超了上限也不能把活任务扔掉, 前端还在轮询它。
    没有这个回收, 注册表是**只增不减**的: 一条批量优化 job 的 strategy_details
    实测 1.4MB, 一次多市场同步的 details 上万条, 服务在生产机上连开几天就是
    RSS 单调往上爬, 最后 OOM 把正在跑的优化一起带走。
    """
    keep = keep or JOB_REGISTRY_MAX
    try:
        items = registry.items()          # LockedDict: 已经是脱离锁的副本
        if len(items) <= keep:
            return
        # 存活状态要用**白名单**列举已结束的, 不能用"不在 running/queued 里就算完了" ——
        # daily-sync / sync-configured 登记时写的是 "preparing", 按黑名单会被判成
        # 已结束: 表满时 register_job 刚插进去就被同一次回收 pop 掉,
        # 前端轮询立刻 404 "task not found", 而后台线程照跑不误。
        _FINISHED = ("done", "error", "cancelled", "failed", "skipped")
        done = [(jid, j) for jid, j in items
                if (j.get("status") if hasattr(j, "get") else None) in _FINISHED]
        if len(items) - len(done) >= keep:
            drop = done                    # 活任务已经占满, 已结束的全清
        else:
            done.sort(key=lambda kv: str(kv[1].get("finished_at")
                                         or kv[1].get("started_at") or ""))
            # 注: 缺时间戳的排在最前 (空串), 优先回收 —— 不能让 str(None)
            # 这种字符串排到 ISO 日期后面, 否则它们永远轮不到被清。
            drop = done[:len(items) - keep]
        for jid, _ in drop:
            registry.pop(jid, None)
        if drop:
            logger.info(f"[job-reap] 清掉 {len(drop)} 条已结束任务, 余 {len(registry)}")
    except Exception as e:
        logger.warning(f"[job-reap] 回收失败(不影响主流程): {e}")


def register_job(registry, job_id: str, fields: dict, nested=("results", "details")):
    """登记一条后台任务 —— **所有** job 都该从这里进, 不要在调用点手写字面量。

    为什么要这个工厂: 上一轮把 8 张注册表的 job 都包成了 LockedDict, 偏偏漏了
    /strategies/validate-new 这一个创建点(它在 router 里手写字面量), 结果
    worker 那边的 `setdefault(job_id, LockedDict({...}))` 拿到的是已存在的普通
    dict —— 新建的 LockedDict 当场被丢掉, 整串改造在这条路径上完全没生效,
    而且因为读侧有 hasattr 兜底, 失败是**静默**的。
    9 个手写字面量 + 全靠记性 = 第 10 个一定还会漏。这里收成一个门。

    nested: 这些字段如果是普通 dict, 一并包成 LockedDict —— results/details
            是 worker 写得最密的两个容器。
    """
    # fields 本身已经是 LockedDict 时**原样登记**, 不要重新包一层:
    # 包一层会产生一个新对象, 而持有旧对象的那一方(比如 EvalChainRunner
    # 自己的 self.progress)还在往旧对象里写 —— 登记表里的那份就此冻结,
    # 前端轮询看到的进度永远停在创建那一刻。
    job = fields if isinstance(fields, LockedDict) else LockedDict(fields)
    for k in nested:
        v = dict.get(job, k)
        if type(v) is dict:
            dict.__setitem__(job, k, LockedDict(v))
    registry[job_id] = job
    _reap_registry(registry)
    return job


# ── 通用任务进度登记表 ─────────────────────────────────────────────────────────
# 为什么要有: 策略评估下一堆耗时接口 (适用性/组合/买卖点/WF/compare…) 都是
# "一次阻塞请求 + 内部 ProcessPool 并发", 请求本身不回报任何中间状态,
# 前端只能塞一句静态文字干等。这里给一个统一的登记表:
#   前端生成 task_id 随请求带上 → 接口在干活的过程中往这里写 → 前端轮询 /progress/{id}
#
# 并发在跑的是哪几个: ProcessPoolExecutor 是 FIFO 派活的, 所以
# "已提交但还没完成的前 n_workers 个" 就是此刻占着 worker 的那几个。
# 这样不用改 worker 函数签名 (worker 在别的进程里, 没法直接写父进程的 dict)。
_prog_tasks = LockedDict()       # task_id -> LockedDict
#: _prog_lock 看起来跟 LockedDict 重复了, 其实不是, 别当冗余删掉:
#: 容器级的锁只保证**单次**操作原子, 而这里好几处是"读-判断-写"的复合操作 ——
#: prog_update / prog_step 都要先确认 status 还是 running 才写, prog_start 要
#: 先收僵尸、再淘汰、再插入。这几步中间被别的线程插进来, 结果就是
#: 「prog_end 之后又被写回 running」这类错乱, 容器自持锁拦不住。
#: LockedDict 负责"容器不会被写坏", _prog_lock 负责"这几步之间没人插队"。
_prog_lock = threading.RLock()
_PROG_MAX = 40          # 最多留这么多条, 免得长时间运行把内存吃掉


#: 一个 running 条目多久没更新就当它死了 (端点 raise 之后没人收尾, 留下僵尸)。
#: 正常任务至少每 0.33s push 一次进度, 5 分钟不动基本可以判定是异常退出了。
_PROG_STALE_SEC = 300


def _prog_reap_locked() -> None:
    """把明显已死的 running 条目标成 error。调用方必须已持有 _prog_lock。

    为什么需要: 各端点 prog_start 之后随时可能 raise (HTTPException、
    worker 异常、BrokenProcessPool…), 没有统一 finally 的话登记表里就留下
    永远 status='running' 的僵尸, 占着 _PROG_MAX 的名额。
    """
    now = time.time()
    # values() 在 LockedDict 上返回的是副本 list, 遍历时别人插入/删除都不会炸;
    # 拿到的 t 本身仍是登记表里那个对象, 所以下面改它是生效的。
    for t in _prog_tasks.values():
        if t.get("status") == "running" and now - t.get("updated_at", now) > _PROG_STALE_SEC:
            t["status"] = "error"
            t["message"] = t.get("message") or "任务异常结束（长时间无进度更新）"
            t["running"] = []
            t["note"] = ""
            t["ended_at"] = now


def prog_start(task_id: str, title: str = "", scope: str = "", steps=None, **extra) -> None:
    if not task_id:
        return
    with _prog_lock:
        if len(_prog_tasks) >= _PROG_MAX:
            _prog_reap_locked()          # 先把僵尸收掉, 多半就腾出位置了
            # 丢掉最老的已结束任务
            for k in sorted(_prog_tasks, key=lambda k: _prog_tasks[k].get("started_at", 0)):
                if _prog_tasks[k].get("status") != "running":
                    _prog_tasks.pop(k, None)
                    break
            else:
                # 全是 running: 绝不能按插入顺序瞎踢, 那会把一个**正在跑**的任务
                # 踢出登记表, 之后它的 prog_update/prog_end 全部静默失效,
                # 前端就永远收到 'unknown'。挑最久没动静的那个 —— 它最可能已经死了。
                stale = min(_prog_tasks, key=lambda k: _prog_tasks[k].get("updated_at", 0))
                _prog_tasks.pop(stale, None)
        _prog_tasks[task_id] = LockedDict({
            "task_id": task_id, "title": title, "scope": scope,
            "steps": steps or [], "step": (steps or [{}])[0].get("key") if steps else None,
            "status": "running", "done": 0, "total": 0, "unit": "",
            "running": [], "workers": 0, "note": "", "message": "",
            "started_at": time.time(), "updated_at": time.time(),
            **extra,
        })


def prog_update(task_id: str, **fields) -> None:
    if not task_id:
        return
    with _prog_lock:
        t = _prog_tasks.get(task_id)
        if not t or t.get("status") != "running":
            return
        t.update(fields)
        t["updated_at"] = time.time()


def prog_step(task_id: str, step_key: str, **fields) -> None:
    """切到下一步, 顺便把计数清零 (每步的 total 不一样)。"""
    if not task_id:
        return
    with _prog_lock:
        t = _prog_tasks.get(task_id)
        if not t or t.get("status") != "running":
            return
        # note 也要跟着清: 它是"上一步"的说明 (比如进程池启动提示),
        # 留到下一步就成了误导
        t.update({"step": step_key, "done": 0, "total": 0, "running": [], "note": ""})
        t.update(fields)
        t["updated_at"] = time.time()


def prog_end(task_id: str, status: str = "done", message: str = "") -> None:
    if not task_id:
        return
    with _prog_lock:
        t = _prog_tasks.get(task_id)
        if not t:
            return
        t["status"] = status
        t["message"] = message
        t["running"] = []
        t["note"] = ""          # 收尾时清掉, 否则完成/失败的面板上还挂着"正在启动…"
        t["ended_at"] = time.time()
        t["updated_at"] = time.time()


def prog_get(task_id: str):
    with _prog_lock:
        _prog_reap_locked()      # 轮询时顺手收僵尸, 前端才能看到"异常结束"而不是一直转
        t = _prog_tasks.get(task_id)
        if not t:
            return None
        # snapshot 而不是 dict(t): 浅拷会把 running / steps 这两个 list 原样
        # 交出去, 它们还是登记表里的活对象 —— 读侧一边序列化、写侧一边换掉,
        # 就是老问题。snapshot 逐层脱壳。
        cur = snapshot_of(t)
    cur["elapsed_seconds"] = round((cur.get("ended_at") or time.time()) - cur.get("started_at", time.time()), 1)
    return cur


class TaskCancelled(Exception):
    """用户主动停止。各端点捕获后返回一个"已取消"的结果, 不当成错误。"""


def prog_request_cancel(task_id: str) -> bool:
    """请求停止某个任务。返回 False 表示这个任务不存在或已经结束。

    只是插旗子, 不强杀 —— 真正的停在 worker 循环里: 下一次
    prog_is_cancelled() 检查时跳出, 顺手 shutdown(cancel_futures=True)
    把还没开跑的分片撤掉。已经在跑的那几个分片会自己跑完(进程池的
    Future.cancel() 对 RUNNING 状态返回 False, 这是 CPython 的语义,
    想真断只能杀进程, 代价和风险都更大)。
    """
    if not task_id:
        return False
    with _prog_lock:
        t = _prog_tasks.get(task_id)
        if not t or t.get("status") != "running":
            return False
        t["cancel_requested"] = True
        t["note"] = "正在停止…（已经在跑的分片会先跑完）"
        t["updated_at"] = time.time()
        return True


def prog_is_cancelled(task_id: str) -> bool:
    if not task_id:
        return False
    t = _prog_tasks.get(task_id)
    return bool(t and t.get("cancel_requested"))


def prog_check_cancel(task_id: str, pool=None, futures=None) -> None:
    """在 as_completed 循环里调一下。被停了就撤掉待派的分片并抛 TaskCancelled。

    futures: 还没跑完的 future 集合。**优先传它**, 别只传 pool ——

    原先这里只写 `pool.shutdown(wait=False, cancel_futures=True)`, 看着对,
    实际一个分片都撤不掉。CPython 的 ProcessPoolExecutor.shutdown 第一行是
        self._cancel_pending_futures = cancel_futures
    然后叫醒管理线程去读这个旗子 —— 真正的取消是**异步**发生的。而我们下一句
    就抛 TaskCancelled, 外层 `with ProcessPoolExecutor(...)` 的 __exit__ 紧接着
    又 shutdown 一次(cancel_futures 默认 False), 在管理线程读到之前就把旗子
    改回 False。
    实测(4 worker / 24 个 0.8s 分片, 第一个完成即取消, 之后进程继续活着):
        旧写法 24/24 全部跑完  ·  直接 cancel 只剩 10/24
    也就是说"已按请求停止"只是话说得好听, CPU 一点没省。用户以为停了, 转头
    再发一个任务, 两套 worker 叠着抢核。

    Future.cancel() 是同步、就地生效的, 不受之后任何 shutdown 影响。
    已经在跑的分片撤不掉(RUNNING 状态 cancel() 返回 False), 那是 CPython 的
    语义, 想真断只能杀进程, 代价和风险都更大。
    """
    if not prog_is_cancelled(task_id):
        return
    if futures:
        try:
            for f in futures:
                f.cancel()
        except Exception:
            pass
    elif pool is not None:
        # 没有 futures 时的退路。只有在 pool **不是** with 管理时才真的有效
        # (比如 WF 那条是裸 pool + try/finally, 只 shutdown 一次)。
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)       # 老版本没有 cancel_futures
        except Exception:
            pass
    # 自己收尾: 异常一路抛到 FastAPI 的处理器时, 那边拿不到 task_id,
    # 不在这里 prog_end 的话进度条会一直停在 running, 直到 300 秒僵尸回收
    # 才被标成 error —— 用户明明是主动停的, 却看到"异常结束"。
    prog_end(task_id, "cancelled", "已按请求停止")
    raise TaskCancelled()


@contextlib.contextmanager
def prog_task(task_id: str, title: str = "", scope: str = "", steps=None, **extra):
    """登记一个任务并保证收尾。

    为什么要有: 各端点里 prog_start 之后随时可能 raise (HTTPException、
    worker 异常через future.result()、BrokenProcessPool…), 没有 finally 的话
    登记表里就留下一条永远 status='running' 的僵尸。攒够 _PROG_MAX 条之后,
    prog_start 的兜底淘汰会按插入顺序 pop, 有可能把一个**正在跑**的任务踢掉,
    之后它的 prog_update/prog_end 全部静默失效, 前端就一直收到 'unknown'。
    """
    prog_start(task_id, title=title, scope=scope, steps=steps, **extra)
    try:
        yield
    except BaseException as e:
        prog_end(task_id, "error", str(e)[:200] or type(e).__name__)
        raise


class ProgTracker:
    """跟着 ProcessPoolExecutor 用的小助手: 记住派活顺序, 完成一个划掉一个,
    剩下的头 n_workers 个就是此刻在跑的。

        tr = ProgTracker(task_id, [str(x) for x in items], n_workers)
        for fut in as_completed(futs):
            ...
            tr.done_one(label_of(fut))
    """

    #: 进程池版的启动说明。起 worker 要几秒到几十秒 (每个都要 import 一次
    #: backtest/pandas, 实测单次 import 就 5s), 这段时间 done 恒为 0,
    #: 进度条纹丝不动, 看着像卡死 —— 给条说明, 别让用户干猜。
    POOL_START_NOTE = "正在启动 {n} 个并行进程…（每个都要加载一次回测环境）"

    def __init__(self, task_id: str, labels, workers: int, unit: str = "",
                 note: Optional[str] = None):
        """note: None = 用进程池默认说明; "" = 不显示; 其它字符串 = 自定义。

        线程池的调用方(比如拉行情)必须显式传, 否则会显示"正在启动 N 个并行进程",
        而它其实在做网络 I/O, 一个进程都没起。
        """
        self.task_id = task_id
        self.pending = list(labels)
        self.workers = max(1, int(workers or 1))
        self.total = len(self.pending)
        self.count = 0
        self._last_push = 0.0
        if note is None:
            note = self.POOL_START_NOTE.format(n=self.workers)
        # running 初始留空: 这时候一个任务都还没真正开跑, 列出 N 个名字
        # 跟旁边"正在启动"的说明自相矛盾。前端对空 running 会显示"正在派活…"。
        prog_update(task_id, total=self.total, done=0, workers=self.workers,
                    unit=unit, running=[], note=note)

    def done_one(self, label=None, force: bool = False) -> None:
        self.count += 1
        if label is not None:
            try:
                self.pending.remove(label)
            except ValueError:
                if self.pending:
                    self.pending.pop(0)
        elif self.pending:
            self.pending.pop(0)
        now = time.time()
        # 最多 3/秒 往登记表写, 几千个任务时别把锁磨穿
        if force or self.count >= self.total or (now - self._last_push) >= 0.33:
            self._last_push = now
            # 每次都带 note="": 有任务落地就说明池子起好了。不做成"只在 count==1
            # 时清一次", 那样一旦第一次 push 被 _last_push 节流吞掉, 说明就留在屏幕上了。
            prog_update(self.task_id, done=self.count, total=self.total,
                        workers=self.workers, running=self.pending[:self.workers],
                        note="")


# ── 优化占用登记表 ───────────────────────────────────────────────────────────
# 为什么要有: 优化是 CPU 密集的, 一次跑会起满 N_WORKERS 个进程池 worker,
# 跑完还要再起一个 WF 池验证. 重复点「优化参数」会让进程池叠加,
# 12 核的机器叠三层就是 29 个 worker 抢 12 核, 整个服务被拖垮 (实测负载 300+).
# 所以同一时刻只允许一个优化任务, 第二个请求直接 409 带上"谁在跑/跑了多久".
_opt_slot: dict = {}              # 空 = 空闲; 否则 = 正在跑的那个任务的描述
_opt_slot_lock = threading.Lock()


#: 槽位多久没心跳就当它死了。批量任务正常每完成一个策略就会推一次进度,
#: 单股优化在 WF 阶段也会持续上报, 10 分钟纹丝不动基本只剩一种可能: 持有者已经没了。
_OPT_SLOT_STALE_SEC = 600


def _opt_slot_is_stale_locked() -> bool:
    """槽位是不是已经没人了。调用方必须持有 _opt_slot_lock。

    为什么需要: 槽位在 finally 里释放, 但 uvicorn --reload / 进程被 kill /
    机器崩了这些情况根本走不到 finally, 槽位就永久占着 —— 之后所有优化请求
    一律 409「已有优化任务在跑」, 而实际上后台什么都没有。
    实测撞到过: 一个批量任务的后台线程随服务重启没了, 槽位还挂了 55 分钟,
    期间点批量优化全被拒。
    """
    if not _opt_slot:
        return False
    beat = _opt_slot.get("heartbeat_at") or _opt_slot.get("started_at") or 0
    return (time.time() - beat) > _OPT_SLOT_STALE_SEC


def opt_slot_acquire(kind: str, scope: str, **extra) -> Optional[dict]:
    """占用优化槽位。成功返回 None, 已被占用则返回当前占用者的描述。"""
    with _opt_slot_lock:
        if _opt_slot and _opt_slot_is_stale_locked():
            logger.warning(
                f"[opt-slot] 槽位僵死自动释放: {_opt_slot.get('kind')}/{_opt_slot.get('scope')} "
                f"已 {int(time.time() - (_opt_slot.get('started_at') or 0))}s 无心跳"
            )
            _opt_slot.clear()
        if _opt_slot:
            return dict(_opt_slot)
        _opt_slot.clear()
        _opt_slot.update({
            "kind": kind,               # "symbol" | "pool"
            "scope": scope,             # 股票代码 或 股票池名
            "started_at": time.time(),
            **extra,
        })
        return None


def opt_slot_release(scope: Optional[str] = None) -> None:
    """释放槽位。scope 给定时只释放自己占的那个, 免得误放别人的。"""
    with _opt_slot_lock:
        if not _opt_slot:
            return
        if scope is not None and _opt_slot.get("scope") != scope:
            return
        _opt_slot.clear()


def opt_slot_progress(**fields) -> None:
    """把进度并进槽位 (线程安全)。槽位空了就丢弃, 不复活一个已结束的任务。

    每次上报顺便打一次心跳, 供 _opt_slot_is_stale_locked 判断持有者是否还活着。
    """
    with _opt_slot_lock:
        if not _opt_slot:
            return
        _opt_slot.update(fields)
        _opt_slot["heartbeat_at"] = time.time()


def opt_slot_status() -> dict:
    """当前占用情况。顺手把僵死的槽位收掉, 免得前端预检一直显示"有任务在跑"。"""
    with _opt_slot_lock:
        if _opt_slot and _opt_slot_is_stale_locked():
            logger.warning(f"[opt-slot] 槽位僵死自动释放 (status 查询时): {_opt_slot.get('scope')}")
            _opt_slot.clear()
    return _opt_slot_status_inner()


def _opt_slot_status_inner() -> dict:
    """当前占用情况 (供 GET /optimize/running 和前端开跑前预检)。"""
    with _opt_slot_lock:
        if not _opt_slot:
            return {"running": False}
        cur = dict(_opt_slot)
    cur["elapsed_seconds"] = round(time.time() - cur.get("started_at", time.time()), 1)
    cur["running"] = True
    # 总体百分比。三段: 参数优化 → WF(优化参数) → WF(标准参数对照)。
    # 原来只分两段(70/30), 而 WF 其实要跑两遍 —— 第二遍 wf_done 从 0 重新数,
    # 于是总进度在界面上是 99% → 70%, 看着像倒退。按三段各自加权。
    stage = cur.get("stage")
    opt_t, opt_d = cur.get("opt_total") or 0, cur.get("opt_done") or 0
    wf_t, wf_d = cur.get("wf_total") or 0, cur.get("wf_done") or 0
    W_OPT, W_WF, W_STD = 60.0, 25.0, 15.0
    if opt_t:
        if stage == "wf_std":
            # 前两段必然已经走完
            pct = W_OPT + W_WF + (W_STD * wf_d / wf_t if wf_t else 0.0)
        elif stage == "wf":
            pct = W_OPT + (W_WF * wf_d / wf_t if wf_t else 0.0)
        else:
            pct = W_OPT * opt_d / opt_t
        cur["percent"] = round(min(pct, 99.0), 1)
    return cur


def opt_slot_busy_message(cur: dict) -> str:
    """拼给用户看的一句话: 谁在跑 / 跑了多久。"""
    sec = int(time.time() - cur.get("started_at", time.time()))
    when = f"{sec}秒" if sec < 60 else f"{sec // 60}分{sec % 60}秒"
    what = "股票池" if cur.get("kind") == "pool" else "个股"
    return (f"已有优化任务在跑: {what} {cur.get('scope', '?')}"
            f"（{cur.get('method', '?')} · 已跑 {when}）。"
            f"优化很吃 CPU, 同一时刻只允许一个, 请等它跑完或先取消。")


def _wf_one_sym(args):
    """单只股票的完整 WF 验证（模块级函数，供 ProcessPoolExecutor 调用）"""
    sym, mkt, results, dyn_pos, interval, period = args
    wf_val = _run_wf_validation(sym, mkt, results, dyn_pos, interval=interval, period=period)
    return sym, wf_val


def _batch_optimize_worker(job_id: str, symbols: list, market: str,
                           period: str, interval: str, method: str, dynamic_pos: bool,
                           strategy_names: list, n_trials: int,
                           signal_type: str = "buy_and_sell",
                           keep_overfit: bool = True,
                           sig_opts: dict = None):
    """后台线程：所有股票×策略扁平化到一个 ProcessPool，无 GIL 瓶颈"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time as _time
    import gc

    interval = _normalize_param_interval(interval)
    job = _batch_jobs[job_id]
    total = len(symbols)
    active = strategy_names or _get_active_strategies()
    run_all = (method == "all")
    methods_to_run = ["bayesian", "grid", "coarse_fine"] if run_all else [method]

    # ── 第一步：预加载数据，构建全局任务列表 ──
    sym_data = {}  # { "AAPL": df, ... }
    flat_tasks = []  # [(sym, sname, task_args), ...]

    # 取数是 I/O (读 parquet, 必要时还要联网做增量更新), 用线程池并发 ——
    # 当天首次跑时每只股票都要走一次网络, 95 只串行就是好几分钟的死等,
    # 而这段时间进程池一个任务都还没派出去。
    # 注意用线程池不是进程池: get_ohlcv 是 I/O 密集, 而且要写进程内缓存。
    from concurrent.futures import ThreadPoolExecutor as _DlPool, as_completed as _dl_ac
    _dl_workers = max(1, min(8, len(symbols)))
    # 进度写回 job 本身: 批量优化走的是 /batch-optimize/status/{job_id} 这条通道,
    # 不是通用的 /progress/{task_id}, 所以这里不用 ProgTracker。
    job.update({"stage": "fetch", "message": f"拉取行情 0/{total}…",
                "fetch_done": 0, "fetch_total": total, "fetch_workers": _dl_workers})

    def _dl_one(sym):
        try:
            return sym, get_ohlcv(sym, market, interval=interval, period=period), None
        except Exception as e:
            return sym, None, str(e)[:120]

    _fetched = {}
    _dl_cancelled = False
    with _DlPool(max_workers=_dl_workers) as _dlp:
        _dl_futs = {_dlp.submit(_dl_one, s): s for s in symbols}
        for _f in _dl_ac(_dl_futs):
            # 取消检查点。改并行之前, 取数是在带 cancel 检查的循环里逐只做的,
            # 点停止能在一只股票之内停下; 改并行之后整段没有检查, 而这正是
            # "当天首次跑、95 只都要联网"最慢的一段 —— 用户点了停止要等它全部下完。
            if job.get("cancelled"):
                _dl_cancelled = True
                try:
                    _dlp.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    _dlp.shutdown(wait=False)
                break
            _sym, _df, _err = _f.result()
            _fetched[_sym] = (_df, _err)
            _dl_n = len(_fetched)
            job.update({"fetch_done": _dl_n, "current_symbol": str(_sym).upper(),
                        "message": f"拉取行情 {_dl_n}/{total}…"})
    if _dl_cancelled:
        job.update({"status": "cancelled", "message": "已取消（拉取行情阶段）"})
        return

    job.update({"stage": "build", "message": "行情就绪, 正在派活…"})
    # 下面组装完就把 _fetched 放掉: 它跟 sym_data 指向同一批 DataFrame,
    # 不放的话后面那句刻意的 del sym_data + gc.collect() 一个都释放不掉
    # (那句是为了在写 JSON/入库之前腾内存)。
    # 按原始顺序组装, 保证 flat_tasks 的次序跟以前一致 (下游有依赖派活顺序的地方)
    for sym in symbols:
        if job.get("cancelled"):
            job["status"] = "cancelled"
            return
        try:
            df, _err = _fetched.get(sym, (None, "未取到"))
            if df is None:
                raise RuntimeError(_err or "取数失败")
            if len(df) < 10:
                job["results"][sym.upper()] = {"status": "skipped", "reason": f"数据不足({len(df)}条)"}
                job["completed"] += 1
                continue
            sym_data[sym.upper()] = df
            for sname in active:
                task_args = (df, sname, method, OPTIMIZE_GRIDS.get(sname), dynamic_pos,
                             n_trials, run_all, methods_to_run, signal_type, sig_opts)
                flat_tasks.append((sym.upper(), sname, task_args))
        except Exception as e:
            job["results"][sym.upper()] = {"status": "error", "reason": str(e)}
            job["completed"] += 1

    if not flat_tasks:
        job["status"] = "done"
        job["message"] = "批量优化完成: 无有效任务"
        return

    total_flat_tasks = len(flat_tasks)
    n_pool = min(N_WORKERS, total_flat_tasks)
    logger.info(f"🚀 Batch optimize: {len(sym_data)} stocks × {len(active)} strategies "
                f"= {len(flat_tasks)} tasks, {n_pool} workers, method={method}")
    job["message"] = f"优化中: {len(sym_data)}只股票, {len(flat_tasks)}个任务并行..."

    # ── 第二步：流水线 —— 优化与WF并行（一只股票优化完立即提交WF）──
    # 按股票跟踪：每只股票有多少策略任务，完成几个
    sym_results = {sym: {"results": {}, "method_comparison": {}, "combos": 0}
                   for sym in sym_data}
    sym_task_count = {}  # { "AAPL": 5 }  每只股票的策略任务总数
    # 进度: 除了原有的按股票计的 completed/total, 再加一份按"任务"(股票×策略) 计的,
    # 不然 22 个策略 × 95 只的时候百分比会长时间卡 0 然后猛跳
    _pending_labels = [f"{sy} · {sn}" for sy, sn, _ in flat_tasks]
    job["task_total"] = total_flat_tasks
    job["task_done"] = 0
    job["workers"] = n_pool
    job["running"] = _pending_labels[:n_pool]

    for sym, sname, _ in flat_tasks:
        sym_task_count[sym] = sym_task_count.get(sym, 0) + 1
    sym_done_count = {sym: 0 for sym in sym_task_count}  # 每只股票已完成的策略数

    t_start = _time.time()

    # 用两个进程池：优化池 + WF池，WF池用较少 worker 避免争资源
    n_wf_workers = max(1, n_pool // 2)
    job["wf_workers"] = n_wf_workers   # 必须等 n_wf_workers 算出来再写, 早了是 UnboundLocalError
    _cancelled_early = False      # 取消时让 finally 别再 join, 见下面注释
    opt_pool = ProcessPoolExecutor(max_workers=n_pool, initializer=proc_role.mark_pool_worker)
    wf_pool = ProcessPoolExecutor(max_workers=n_wf_workers, initializer=proc_role.mark_pool_worker)

    wf_futures = {}       # { future: sym }
    wf_pending = {}       # 未处理的WF futures（浅拷贝用于边优化边收割）
    sym_save_data = {}    # 已保存的股票数据
    wf_submitted = set()  # 已提交WF的股票
    wf_done = 0

    def _harvest_wf_results():
        """收割已完成的WF结果（非阻塞）"""
        nonlocal wf_done
        finished = [f for f in wf_pending if f.done()]
        for fut in finished:
            sym_wf = wf_pending.pop(fut)
            save_data, results, combos = sym_save_data[sym_wf]
            try:
                _, wf_validation = fut.result()
                promote_summary = {
                    "custom_promoted": [],
                    "universal_promoted": [],
                    "custom_count": 0,
                    "universal_count": 0,
                }
                if wf_validation:
                    # 「保留过拟合参数」没勾: WF 判成 overfit 的策略这次的优化参数不入库
                    if not keep_overfit:
                        _dropped = [sn for sn, wf in wf_validation.items()
                                    if isinstance(wf, dict) and wf.get("verdict") == "overfit"]
                        for sn in _dropped:
                            results.pop(sn, None)
                            (save_data.get("strategies") or {}).pop(sn, None)
                        if _dropped:
                            save_data["dropped_overfit"] = _dropped
                            logger.info(f"[batch-opt] {sym_wf} 未勾「保留过拟合参数」, 丢弃 {len(_dropped)} 个")
                    existing_wf = save_data.get("wf_validation", {})
                    if existing_wf and any(isinstance(v, dict) and "verdict" in v for v in existing_wf.values()):
                        existing_wf = {"best": existing_wf}
                    existing_wf["best"] = wf_validation
                    save_data["wf_validation"] = existing_wf
                    _save_optimized(sym_wf, _write_interval_layer(_load_optimized(sym_wf) or {}, interval, save_data, strategy_payload=False))
                    promote_summary = _promote_to_standard(
                        sym_wf, market.upper(), wf_validation, results,
                        period=period, dynamic_pos=dynamic_pos, interval=interval
                    )
                else:
                    wf_validation = {}
            except Exception as e:
                logger.error(f"Batch WF {sym_wf} error: {e}")
                wf_validation = {}

            n_robust = sum(1 for v in wf_validation.values() if v.get("promoted"))
            n_fair = sum(1 for v in wf_validation.values() if v.get("verdict") == "fair")
            n_overfit = sum(1 for v in wf_validation.values() if v.get("verdict") == "overfit")
            strat_details = {}
            for sn, sd in results.items():
                if not sd: continue
                entry = {k:v for k,v in sd.items() if k not in ("params","optimized_keys","buy_params","sell_params")}
                entry["wf_verdict"] = (wf_validation.get(sn) or {}).get("verdict", "-")
                strat_details[sn] = entry
            job["results"][sym_wf] = {
                "status": "done", "strategies": len(results),
                "robust": n_robust, "fair": n_fair, "overfit": n_overfit,
                "elapsed": round((_time.time() - t_start) / max(len(sym_results), 1), 1),
                "combos": combos,
                "custom_promoted": promote_summary.get("custom_promoted", []),
                "universal_promoted": promote_summary.get("universal_promoted", []),
                "strategy_details": strat_details,
            }
            job["completed"] += 1
            job["total_robust"] += n_robust
            job["total_standard_promoted"] += promote_summary.get("custom_count", 0)
            job["total_universal_promoted"] += promote_summary.get("universal_count", 0)
            if promote_summary.get("custom_promoted"):
                job["recent_standard_promotions"] = (
                    job.get("recent_standard_promotions", []) + [
                        f"{sym_wf}:{item['strategy']}" for item in promote_summary["custom_promoted"]
                    ]
                )[-12:]
            if promote_summary.get("universal_promoted"):
                job["recent_universal_promotions"] = (
                    job.get("recent_universal_promotions", []) + [
                        f"{item['strategy']}@{item['source_symbol']}" for item in promote_summary["universal_promoted"]
                    ]
                )[-12:]
            wf_done += 1
            logger.info(
                f"✅ Batch opt+WF {sym_wf}: {len(results)} strategies, {n_robust} robust, "
                f"std+{promote_summary.get('custom_count', 0)}, uni+{promote_summary.get('universal_count', 0)}"
            )

    try:
        # 提交所有优化任务
        opt_futures = {}
        for sym, sname, task_args in flat_tasks:
            fut = opt_pool.submit(_optimize_one_stock_strategy, task_args)
            opt_futures[fut] = (sym, sname)

        done_count = 0
        for fut in as_completed(opt_futures):
            if job.get("cancelled"):
                # 直接撤 future, 不靠 shutdown(cancel_futures=True):
                # 那个只是置一个旗子等管理线程去读, 而我们马上 return,
                # 下面 finally 里的第二次 shutdown 会在读到之前把旗子改回 False
                # —— 实测取消形同虚设, 剩余分片照跑完。
                for _f in opt_futures:
                    _f.cancel()
                for _f in list(wf_futures):
                    _f.cancel()
                _cancelled_early = True
                job["status"] = "cancelled"
                job["message"] = f"已取消 ({job['completed']}/{total})"
                gc.collect()
                return

            sym, sname = opt_futures[fut]
            try:
                res = fut.result()
                sr = sym_results[sym]
                sr["combos"] += res.get("combos", 0)
                if res.get("result"):
                    sr["results"][res["sname"]] = res["result"]
                if res.get("method_comparison"):
                    sr["method_comparison"][res["sname"]] = res["method_comparison"]
            except Exception as e:
                logger.error(f"Batch opt {sym}/{sname} error: {e}")

            done_count += 1
            sym_done_count[sym] = sym_done_count.get(sym, 0) + 1

            # current_symbol 之前只在初始化时设成 "", 全库没有任何地方赋值,
            # 所以前端进度条里"当前在跑哪只"永远是空的 —— 这里补上
            job["current_symbol"] = sym
            job["task_done"] = done_count
            # 给优化槽位打心跳, 否则跑得久会被 _opt_slot_is_stale_locked 误判成僵死
            opt_slot_progress(stage="batch", done=done_count, total=total_flat_tasks,
                              current=sym, workers=n_pool)
            try:
                _pending_labels.remove(f"{sym} · {sname}")
            except ValueError:
                pass
            job["running"] = _pending_labels[:n_pool]

            # 检查该股票是否所有策略都优化完毕 → 立即保存并提交WF
            if sym not in wf_submitted and sym_done_count[sym] >= sym_task_count[sym]:
                wf_submitted.add(sym)
                sr = sym_results[sym]
                results = sr["results"]
                if not results:
                    job["results"][sym] = {"status": "skipped", "reason": "无有效策略结果"}
                    job["completed"] += 1
                else:
                    elapsed_so_far = _time.time() - t_start
                    save_data = {
                        "symbol": sym, "market": market.upper(),
                        "period": period, "interval": interval, "dynamic_position": dynamic_pos,
                        "method": method,
                        "optimized_at": datetime.utcnow().isoformat(),
                        "total_combos_tested": sr["combos"],
                        "elapsed_seconds": round(elapsed_so_far, 1),
                        "strategies": results,
                    }
                    if sr["method_comparison"]:
                        save_data["method_comparison"] = sr["method_comparison"]
                    _save_optimized(sym, _write_interval_layer(_load_optimized(sym) or {}, interval, save_data, strategy_payload=False))
                    sym_save_data[sym] = (save_data, results, sr["combos"])

                    # ── Buy/Sell 模式：保存到 signal_params，不走 WF ──
                    if signal_type in ("buy", "sell"):
                        for sname, sdata in results.items():
                            if sdata.get("params"):
                                _save_signal_params(sym, sname, signal_type,
                                                    params=sdata["params"],
                                                    metrics={k: v for k, v in sdata.items() if k != "params"},
                                                    interval=interval)
                        job["results"][sym] = {
                            "status": "done", "strategies": len(results),
                            "signal_type": signal_type,
                            "combos": sr["combos"],
                            "strategy_details": {sn: {k:v for k,v in sd.items() if k not in ("params","optimized_keys","buy_params","sell_params")} for sn, sd in results.items() if sd},
                        }
                        job["completed"] += 1
                        wf_submitted.add(sym)  # 标记完成，跳过 WF
                        logger.info(f"📤 {sym} {signal_type}优化完成({len(results)}策略), 已保存到 signal_params")
                    else:
                        # ── Buy&Sell 模式：先保存 buy/sell 独立参数，再走 WF 验证 ──
                        for sname, sdata in results.items():
                            bp = sdata.get("buy_params")
                            sp = sdata.get("sell_params")
                            if bp:
                                _save_signal_params(sym, sname, "buy", params=bp,
                                                    metrics=sdata.get("buy_metrics"),
                                                    interval=interval)
                            if sp:
                                _save_signal_params(sym, sname, "sell", params=sp,
                                                    metrics=sdata.get("sell_metrics"),
                                                    interval=interval)
                        # period 传 None: WF 用自己的取数窗口(完整历史), 不跟优化窗口走。
                        # 之前把优化用的 period(日线=1y=252根) 原样传下来, WF 只能切出
                        # 2 个 anchored_micro 窗口, 测试段 21 根常常零交易 → 全判 overfit。
                        wf_task = (sym, market, results, dynamic_pos, interval, None)
                        wf_fut = wf_pool.submit(_wf_one_sym, wf_task)
                        wf_futures[wf_fut] = sym
                        wf_pending[wf_fut] = sym
                        logger.info(f"📤 {sym} 优化完成({len(results)}策略), buy/sell参数已保存, WF已提交")

            # 每次优化任务完成后，非阻塞收割已完成的WF结果
            _harvest_wf_results()

            if done_count % 10 == 0:
                job["message"] = (
                    f"流水线: 优化 {done_count}/{len(flat_tasks)}, WF {wf_done}/{len(wf_submitted)} 完成, "
                    f"稳健{job['total_robust']} / 个股标准升级{job['total_standard_promoted']} / "
                    f"通用标准升级{job['total_universal_promoted']}"
                )

        # 优化全部完成，关闭优化池
        opt_pool.shutdown(wait=True)
        del opt_pool
        gc.collect()

        # 等待剩余WF完成
        job["message"] = f"优化完成, 等待剩余WF验证 {wf_done}/{len(wf_futures)}..."
        remaining = list(wf_pending.keys())
        for fut in as_completed(remaining):
            _harvest_wf_results()
            job["message"] = (
                f"WF验证: {wf_done}/{len(wf_futures)} 完成, "
                f"个股标准升级{job['total_standard_promoted']} / 通用标准升级{job['total_universal_promoted']}"
            )
    finally:
        try:
            opt_pool.shutdown(wait=False)
        except Exception:
            pass
        # 被取消时不能 wait=True: 那会一直 join 到所有在飞的 WF 窗口算完
        # (每个几十秒), 而 job 状态早就写成 cancelled、UI 也显示已取消了 ——
        # 机器还在满载跑一堆没人要的结果。
        wf_pool.shutdown(wait=not _cancelled_early)
        del wf_pool, wf_futures
        gc.collect()

    elapsed_total = _time.time() - t_start

    # 释放内存
    del sym_data, sym_results, flat_tasks, sym_save_data, _fetched
    gc.collect()

    job["status"] = "done"
    job["message"] = (
        f"批量优化完成: {total}只股票, 个股标准升级{job['total_standard_promoted']}项, "
        f"通用标准升级{job['total_universal_promoted']}项 ({round(elapsed_total)}s)"
    )
    logger.info(f"🏁 Batch optimize job {job_id} finished: {total} stocks in {round(elapsed_total)}s")

    # ── 保存优化运行记录 ──
    try:
        _save_optimization_run(
            scope="pool", pool_name=job.get("pool_name", ""),
            symbols=symbols, signal_type=signal_type,
            method=method, interval=interval, n_trials=n_trials,
            results=job.get("results", {}),
            total_combos=sum(r.get("combos", 0) for r in job.get("results", {}).values() if isinstance(r, dict)),
            elapsed_seconds=elapsed_total,
        )
    except Exception as e:
        logger.warning(f"[OptHistory] batch save failed: {e}")


def _extract_wf_strategies(opt: dict, param_source: str, symbol: str = None, interval: str = "1d") -> dict:
    """根据 param_source 从优化数据中提取用于 WF 验证的策略参数
    param_source: "standard" (通用标准), "custom" (个股标准), "best" (最优), "bayesian", "grid", "coarse_fine"
    """
    if param_source == "standard":
        # 使用通用标准参数
        result = {}
        active = _get_active_strategies()
        for sname in active:
            try:
                params = _get_universal_strategy_params(sname, interval=interval)
                if params:
                    result[sname] = {"params": params}
            except Exception:
                pass
        return result

    if param_source == "custom" and symbol:
        # 个股标准参数 = 通用标准 + custom_params 覆盖
        result = {}
        active = _get_active_strategies()
        for sname in active:
            try:
                params = _get_standard_strategy_params(symbol.upper(), sname, interval=interval)
                if params:
                    result[sname] = {"params": params}
            except Exception:
                pass
        return result

    if param_source == "best" or not param_source:
        # 使用全局最优参数（strategies 字段）
        return opt.get("strategies", {})

    # 使用指定方法的参数（从 method_best 中提取）
    method_best = opt.get("method_best", {})
    result = {}
    for sname, methods in method_best.items():
        method_data = methods.get(param_source)
        if method_data and method_data.get("params"):
            result[sname] = method_data
    return result


def _batch_wf_worker(job_id: str, symbols: list, market: str,
                     dynamic_pos: bool, param_source: str):
    """后台线程：所有股票×策略的 WF 窗口 统一扁平化到一个全局进程池，真正并行"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import gc

    job = _batch_jobs[job_id]
    total = len(symbols)
    src_label = {"standard": "通用标准", "custom": "个股标准", "best": "最优", "bayesian": "贝叶斯", "grid": "穷举", "coarse_fine": "粗搜精搜"}.get(param_source, param_source)

    # ── 第一步：预处理所有股票，构建全局任务列表 ──
    # sym_tasks: { "AAPL": { "claude_dual_ma_cross": [(tr,te,sname,params,dyn), ...], ... }, ... }
    sym_tasks = {}
    sym_opt = {}  # 保存每只股票的 opt 和 opt_strategies 以便后续保存

    for sym in symbols:
        if job.get("cancelled"):
            job["status"] = "cancelled"
            return
        # 必须读 interval 层, 不能读整个文件。
        # 新格式是 {"intervals": {"1d": {"strategies": ...}}}, 直接 .get("strategies")
        # 拿到的是 {} → 整批股票全被判成"无优化参数"跳过。实测 189/189 中招。
        opt = _load_optimized_interval(sym.upper(), "1d") or {}
        if not opt and param_source != "standard":
            job["results"][sym.upper()] = {"status": "skipped", "reason": "无优化参数"}
            job["completed"] += 1
            continue
        opt_strategies = _extract_wf_strategies(opt, param_source, symbol=sym)
        if not opt_strategies:
            job["results"][sym.upper()] = {"status": "skipped", "reason": f"无{src_label}参数"}
            job["completed"] += 1
            continue

        # 获取数据并构建 WF 窗口
        try:
            try:
                wf_df = get_ohlcv(sym, market, interval="1d", period="max")
            except Exception:
                wf_df = get_ohlcv(sym, market, interval="1d", period="10y")
            wf_plan = _build_wf_plan(wf_df, train_years=3, test_months=6)
            if not wf_plan.get("ok"):
                job["results"][sym.upper()] = {"status": "skipped", "reason": wf_plan.get("reason", "窗口不足")}
                job["completed"] += 1
                continue
            wf_windows = wf_plan["windows"]

            tasks_for_sym = {}
            for sname, sdata in opt_strategies.items():
                opt_params = sdata.get("params")
                if not opt_params:
                    continue
                tasks_for_sym[sname] = [
                    (tr, te, sname, opt_params, dynamic_pos) for tr, te in wf_windows
                ]
            if tasks_for_sym:
                sym_tasks[sym.upper()] = {"tasks": tasks_for_sym, "plan": wf_plan}
                sym_opt[sym.upper()] = (opt, opt_strategies)

        except Exception as e:
            job["results"][sym.upper()] = {"status": "error", "reason": str(e)}
            job["completed"] += 1

    if not sym_tasks:
        job["status"] = "done"
        job["message"] = f"WF批量验证完成: 无有效任务"
        return

    # ── 第二步：扁平化所有任务，提交到全局进程池 ──
    flat_tasks = []  # [(sym, sname, task_args), ...]
    for sym, bundle in sym_tasks.items():
        for sname, task_list in bundle["tasks"].items():
            for t in task_list:
                flat_tasks.append((sym, sname, t))

    total_flat_tasks = len(flat_tasks)
    n_pool = min(N_WORKERS, total_flat_tasks)
    logger.info(f"🚀 Batch WF: {len(sym_tasks)} stocks, {total_flat_tasks} tasks, "
                f"{n_pool} workers, param_source={param_source}")
    job["message"] = f"WF验证中: {len(sym_tasks)}只股票, {total_flat_tasks}个任务并行..."

    # 收集结果: { "AAPL": { "claude_dual_ma_cross": [result, ...], ... } }
    wf_raw = {sym: {} for sym in sym_tasks}

    _cancelled_early = False
    pool = ProcessPoolExecutor(max_workers=n_pool, initializer=proc_role.mark_pool_worker)
    try:
        futures = {}
        for sym, sname, task_args in flat_tasks:
            fut = pool.submit(_wf_one_window, task_args)
            futures[fut] = (sym, sname)

        done_count = 0
        for fut in as_completed(futures):
            if job.get("cancelled"):
                for _f in futures:
                    _f.cancel()
                _cancelled_early = True
                job["status"] = "cancelled"
                gc.collect()
                return

            sym, sname = futures[fut]
            try:
                res = fut.result()
                wf_raw[sym].setdefault(sname, []).append(res)
            except Exception as e:
                logger.warning(f"WF window {sym}/{sname} failed: {e}")

            done_count += 1
            if done_count % 50 == 0:
                job["message"] = f"WF验证: {done_count}/{total_flat_tasks} 窗口完成..."
    finally:
        pool.shutdown(wait=not _cancelled_early)
        del pool, futures, flat_tasks
        gc.collect()

    # ── 第三步：汇总结果，保存 ──
    for sym, strats_results in wf_raw.items():
        opt, opt_strategies = sym_opt[sym]
        wf_validation = {}
        wf_plan = sym_tasks[sym]["plan"]
        for sname, wf_results in strats_results.items():
            if not wf_results:
                continue
            wf_validation[sname] = _summarize_wf_results(wf_results, wf_plan)

        if wf_validation:
            # 按 param_source 分层保存 WF 结果
            existing_wf = opt.get("wf_validation", {})
            # 兼容旧格式
            if existing_wf and any(isinstance(v, dict) and "verdict" in v for v in existing_wf.values()):
                existing_wf = {"best": existing_wf}
            existing_wf[param_source] = wf_validation
            opt["wf_validation"] = existing_wf
            _save_optimized(sym, opt)
            # 仅对优化参数（非标准）的 WF 结果进行升级判断
            if param_source != "standard":
                _promote_to_standard(
                    sym, market.upper(), wf_validation, opt_strategies,
                    period=opt.get("period", "2y"), dynamic_pos=dynamic_pos
                )

        n_robust = sum(1 for v in wf_validation.values() if v.get("promoted"))
        n_fair = sum(1 for v in wf_validation.values() if v["verdict"] == "fair" and not v.get("promoted"))
        n_overfit = sum(1 for v in wf_validation.values() if v["verdict"] == "overfit")
        job["results"][sym] = {
            "status": "done", "strategies": len(strats_results),
            "robust": n_robust, "fair": n_fair, "overfit": n_overfit,
        }
        job["completed"] += 1
        job["total_robust"] += n_robust
        logger.info(f"✅ Batch WF {sym}: {len(strats_results)} strategies, {n_robust} robust")

    job["status"] = "done"
    job["message"] = f"WF批量验证完成: {total}只股票, {job['total_robust']}个策略升级"
    logger.info(f"🏁 Batch WF job {job_id} finished: {total} stocks, {total_flat_tasks} tasks")


def _compare_bt_one(args):
    """模块级函数：单策略标准vs优化参数对比回测（可被ProcessPool序列化）。
    args 格式：(sname, df, dyn_pos, std_params, opt_params[, buy_params, sell_params])
    当 buy_params/sell_params 存在时使用双参数分离模式。
    """
    if len(args) >= 7:
        sname, df_, dyn_pos, std_params, opt_params, buy_params, sell_params = args[:7]
    else:
        sname, df_, dyn_pos, std_params, opt_params = args[:5]
        buy_params, sell_params = None, None
    # df 允许是预先 pickle 好的 bytes: 父进程派活时 pandas 的 pickler 只跑一次,
    # 之后每个任务带的都是同一段 bytes (memcpy)。22 个策略共用一份行情时,
    # 这是"序列化 22 次 DataFrame"和"序列化 1 次 + 复制 22 次"的差别。
    if isinstance(df_, (bytes, bytearray)):
        import pickle as _pk
        df_ = _pk.loads(df_)
    row = {"strategy": sname, "description": get_strategy(sname).description}
    try:
        if buy_params is not None and sell_params is not None:
            r_std = run_strategy_backtest(df_, sname,
                                          buy_params=buy_params, sell_params=sell_params,
                                          dynamic_position=dyn_pos)
        else:
            r_std = run_strategy_backtest(df_, sname, params_override=std_params, dynamic_position=dyn_pos)
        sm = r_std["summary"]
        row["std_sharpe"] = sm["sharpe"]
        row["std_return"] = sm["total_return"]
        row["std_max_dd"] = sm["max_drawdown"]
        row["std_win_rate"] = sm["win_rate"]
        row["std_trades"] = sm["total_trades"]
        row["std_score"] = composite_score(
            sm["total_return"], sm["sharpe"], sm["max_drawdown"],
            sm["win_rate"], sm.get("profit_factor", 0)
        )
    except Exception:
        row.update({"std_sharpe": 0, "std_return": 0, "std_max_dd": 0,
                    "std_win_rate": 0, "std_trades": 0, "std_score": 0})
    if opt_params is not None:
        try:
            r_opt = run_strategy_backtest(df_, sname, params_override=opt_params,
                                          dynamic_position=dyn_pos)
            sm = r_opt["summary"]
            row["opt_sharpe"] = sm["sharpe"]
            row["opt_return"] = sm["total_return"]
            row["opt_max_dd"] = sm["max_drawdown"]
            row["opt_win_rate"] = sm["win_rate"]
            row["opt_trades"] = sm["total_trades"]
            row["opt_score"] = composite_score(
                sm["total_return"], sm["sharpe"], sm["max_drawdown"],
                sm["win_rate"], sm.get("profit_factor", 0)
            )
            row["opt_params"] = opt_params
            row["has_optimized"] = True
        except Exception:
            row["has_optimized"] = False
    else:
        row["has_optimized"] = False
    if row.get("has_optimized"):
        row["delta_sharpe"] = round(row["opt_sharpe"] - row["std_sharpe"], 4)
        row["delta_return"] = round(row["opt_return"] - row["std_return"], 2)
        row["delta_score"] = round(row["opt_score"] - row["std_score"], 4)
    return row


import uuid

PAPER_TRADE_FILE = Path(__file__).parent / "paper_trades.json"
PAPER_TRADE_BACKUP_DIR = Path(__file__).parent / "paper_trades_backups"
PAPER_TRADE_BACKUP_DIR.mkdir(exist_ok=True)

def _load_paper_trades() -> dict:
    if PAPER_TRADE_FILE.exists():
        return json.load(open(PAPER_TRADE_FILE, encoding="utf-8"))
    return {"positions": [], "history": [], "next_id": 1}

def _save_paper_trades(data: dict):
    # Keep a rolling local backup before overwriting the main file.
    if PAPER_TRADE_FILE.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_fp = PAPER_TRADE_BACKUP_DIR / f"paper_trades_{ts}.json"
        try:
            backup_fp.write_text(PAPER_TRADE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
            backups = sorted(PAPER_TRADE_BACKUP_DIR.glob("paper_trades_*.json"))
            for old_fp in backups[:-20]:
                old_fp.unlink(missing_ok=True)
        except Exception:
            pass
    with open(PAPER_TRADE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


_auto_opt_status = {"running": False, "progress": "", "result": None, "error": None}

def _scan_one_stock_mp(args):
    """模块级函数：扫描单只股票买入信号（可被ProcessPoolExecutor序列化）"""
    sym, market, interval, period, use_strategies, refresh_sec, signal_side, local_only = args
    stock_alerts = []
    try:
        df = get_ohlcv(
            sym, market, interval=interval, period=period,
            max_age_sec=refresh_sec, include_today=True, local_only=local_only
        )
        if len(df) < 30:
            return sym, stock_alerts, None
    except Exception as e:
        return sym, stock_alerts, str(e)[:50]

    stock_name = _get_stock_name(sym, market)

    for sname in use_strategies:
        try:
            strategy = get_strategy(sname)
            params = _get_standard_strategy_params(sym, sname)
            df_ind = strategy.add_indicators(df.copy(), params)
            if len(df_ind) < 2:
                continue
            df_sig = strategy.generate_signals(df_ind, params)
            last = df_sig.iloc[-1]
            prev = df_sig.iloc[-2] if len(df_sig) >= 2 else None
            if signal_side in ("buy", "both") and last.get("buy_signal", 0) == 1:
                if prev is None or prev.get("buy_signal", 0) == 0:
                    stock_alerts.append({
                        "symbol": sym.upper(), "name": stock_name, "market": market,
                        "strategy": sname,
                        "side": "BUY",
                        "price": round(float(last.get("close", 0)), 2),
                        "time": str(df_sig.index[-1]),
                        "interval": interval,
                    })
            if signal_side in ("sell", "both") and last.get("sell_signal", 0) == 1:
                if prev is None or prev.get("sell_signal", 0) == 0:
                    stock_alerts.append({
                        "symbol": sym.upper(), "name": stock_name, "market": market,
                        "strategy": sname,
                        "side": "SELL",
                        "price": round(float(last.get("close", 0)), 2),
                        "time": str(df_sig.index[-1]),
                        "interval": interval,
                    })
        except Exception:
            continue
    return sym, stock_alerts, None


def _rank_one_stock(args):
    """Per-stock strategy ranking (using 2-year data, annualized return).

    For each stock:
      1. Fetch 2-year daily OHLCV data (or all available if < 2 years)
      2. Run all active strategies using per-stock standard params
      3. Calculate annualized return = total_return / actual_years
      4. Group by llm_owner, each owner independently takes top_n
      5. Merge all owners' top_n into final list

    This ensures each LLM owner has its own best strategies per stock.
    e.g. 3 owners x top 5 = up to 15 strategies per stock.
    """
    sym, market, active_list, top_n, local_only = args
    try:
        df = get_ohlcv(sym, market, interval="1d", period="2y", local_only=local_only)
        if len(df) < 30:
            return sym, active_list[:top_n]
    except Exception:
        return sym, active_list[:top_n]

    actual_days = (df.index[-1] - df.index[0]).days
    actual_years = max(actual_days / 365.25, 0.1)
    llm_registry = _load_strategy_llm_registry()

    # Score all strategies by annualized return on this stock
    scores = []
    for sname in active_list:
        try:
            use_params = _get_standard_strategy_params(sym, sname)
            r = run_strategy_backtest(df.copy(), sname, params_override=use_params)
            s = r["summary"]
            total_return = float(s.get("total_return", 0))
            annualized = total_return / actual_years
            scores.append((sname, annualized))
        except Exception:
            scores.append((sname, 0))

    # Group by owner, each owner takes top_n (ranked by annualized return)
    by_owner = {}
    for sname, sc in scores:
        owner = _get_strategy_llm_owner(sname, llm_registry)
        by_owner.setdefault(owner, []).append((sname, sc))

    merged = []
    for owner, owner_scores in by_owner.items():
        owner_scores.sort(key=lambda x: x[1], reverse=True)
        merged.extend([s[0] for s in owner_scores[:top_n]])

    return sym, merged


def _resolve_scan_universe(pool_names: list = None, all_pools: bool = False,
                           markets: list = None, scope: str = "pool") -> list:
    pools = _load_pools()
    scope = str(scope or "pool").lower()
    if scope == "market":
        use_markets = [str(m).upper() for m in (markets or _configured_market_codes()) if str(m).strip()]
        return _configured_market_universe(markets=use_markets)

    selected = list(pool_names or [])
    if all_pools and not selected:
        selected = list(pools.keys())
    return _configured_market_universe(pools=selected)


def _run_realtime_scan(items: list, interval: str = "5m", top_n: int = 0,
                       strategy_names: list = None,
                       llm_owners: list = None,
                       signal_side: str = "buy", refresh_sec: int = None,
                       local_only: bool = False,
                       filters: dict = None,
                       progress_hook=None,
                       *,
                       cycle_id: Optional[str] = None) -> dict:
    # v0.5 §7.2.1 line 1654: accept optional cycle_id; auto-generate when omitted.
    # cycle_id is threaded for future scan_center audit hooks (P8); kept as a
    # local here so the signature is future-compatible without behavior change.
    if cycle_id is None:
        try:
            import scan_center as _sc  # lazy/guarded
            cycle_id = _sc.new_cycle_id_safe("manual")
        except ImportError:
            import time as _time
            import uuid as _uuid
            cycle_id = f"manual-{int(_time.time())}-{_uuid.uuid4().hex[:8]}"
    _ = cycle_id  # reserved for P8 audit propagation; no-op today
    if not items:
        return {
            "alerts": [],
            "scan_time": datetime.now().isoformat(),
            "stocks_scanned": 0,
            "strategies_used": 0,
            "strategies_desc": "无",
            "errors": [],
            "total_symbols_before_filter": 0,
            "candidates_after_filter": 0,
            "filtered_out": 0,
            "filter_reasons": {},
        }

    filters = filters or {}
    total_symbols_before_filter = len(items)
    filter_reasons = {}
    if _selector_filter_requested(filters):
        filtered_items = []
        n_filter_workers = min(max(2, N_WORKERS * 2), len(items))
        if progress_hook:
            progress_hook("filtering", 0, len(items), "按条件过滤候选...")
        with ThreadPoolExecutor(max_workers=n_filter_workers) as ex:
            futures = {ex.submit(_selector_filter_one, (item, interval, filters)): item for item in items}
            done_filter = 0
            for fut in as_completed(futures):
                item = futures[fut]
                sym = item["symbol"]
                try:
                    ok, reason = fut.result()
                    if ok:
                        filtered_items.append(item)
                    else:
                        reason = reason or "filtered"
                        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                except Exception as e:
                    reason = str(e)[:80] or "filter_error"
                    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                done_filter += 1
                if progress_hook:
                    progress_hook("filtering", done_filter, len(items), f"过滤 {sym}")
        items = filtered_items

    candidates_after_filter = len(items)
    filtered_out = max(0, total_symbols_before_filter - candidates_after_filter)
    if not items:
        return {
            "alerts": [],
            "scan_time": datetime.now().isoformat(),
            "stocks_scanned": 0,
            "signal_side": signal_side,
            "strategies_used": len(strategy_names or []),
            "strategies_desc": (f"选中{len(strategy_names)}个" if strategy_names else "无候选"),
            "errors": [],
            "total_symbols_before_filter": total_symbols_before_filter,
            "candidates_after_filter": candidates_after_filter,
            "filtered_out": filtered_out,
            "filter_reasons": filter_reasons,
            "filters_applied": filters,
        }

    wh = _load_warehouse()
    active_list = wh.get("active", []) or [s.name for s in STRATEGY_REGISTRY.values()]
    llm_registry = _load_strategy_llm_registry()
    owner_filter = {
        _normalize_strategy_owner(owner)
        for owner in (llm_owners or [])
        if str(owner).strip()
    }
    if owner_filter:
        active_list = [
            s for s in active_list
            if _get_strategy_llm_owner(s, llm_registry) in owner_filter
        ]
    selected_strategies = []
    if strategy_names:
        selected_strategies = [s for s in strategy_names if s in active_list]
    if not active_list and not selected_strategies:
        return {
            "alerts": [],
            "scan_time": datetime.now().isoformat(),
            "stocks_scanned": 0,
            "signal_side": signal_side,
            "strategies_used": 0,
            "strategies_desc": "无可用策略",
            "errors": [],
            "total_symbols_before_filter": total_symbols_before_filter,
            "candidates_after_filter": candidates_after_filter,
            "filtered_out": filtered_out,
            "filter_reasons": filter_reasons,
            "filters_applied": filters,
        }
    use_ranked_top = not selected_strategies and bool(top_n and top_n > 0)
    period_map = {
        "1m": "2d", "5m": "10d", "15m": "20d", "30m": "30d",
        "1h": "6mo", "1d": "1y"
    }
    period = period_map.get(interval, "10d")

    scan_symbols = [item["symbol"] for item in items]
    scan_markets = list(set(item["market"] for item in items))
    scan_mtime = max((_data_mtime_sig(scan_symbols, m) for m in scan_markets), default="0")
    scan_scope_key = ",".join(sorted(f"{item['market']}:{item['symbol']}" for item in items[:200]))
    owner_cache_sig = ",".join(sorted(owner_filter)) if owner_filter else "all"
    active_sig = hashlib.md5(",".join(sorted(active_list)).encode()).hexdigest()[:10] if active_list else "none"
    scan_rank_cache_key = (
        f"scan_rank:{hashlib.md5(scan_scope_key.encode()).hexdigest()[:10]}:"
        f"{scan_mtime}:{top_n}:{owner_cache_sig}:{active_sig}"
    )
    stock_top_cache = cache_get(scan_rank_cache_key, ttl=86400 * 7) or {}

    need_rank_items = []
    if use_ranked_top:
        for item in items:
            cached = stock_top_cache.get(item["symbol"])
            # Per-owner ranking: each owner gets top_n, so total > top_n
            if not cached or len(cached) < top_n:
                need_rank_items.append(item)

    if need_rank_items:
        rank_args = [(item["symbol"], item["market"], active_list, top_n, local_only) for item in need_rank_items]
        if progress_hook:
            progress_hook("ranking", 0, len(rank_args), "计算策略排名...")
        # Per-market rotation + capped workers — same rationale as the scan
        # stage below: avoid blowing past IB Gateway's 32-connection limit and
        # yfinance's rate limit.
        rank_by_market: dict[str, list] = {}
        for args in rank_args:
            rank_by_market.setdefault(args[1], []).append(args)
        done_rank = 0
        for market_name, market_rank_args in rank_by_market.items():
            n_rank_workers = min(SCAN_WORKERS, len(market_rank_args))
            with ProcessPoolExecutor(max_workers=n_rank_workers, initializer=proc_role.mark_pool_worker) as ex:
                for sym, top_list in ex.map(_rank_one_stock, market_rank_args):
                    stock_top_cache[sym] = top_list
                    done_rank += 1
                    if progress_hook:
                        progress_hook("ranking", done_rank, len(rank_args), f"{market_name} 排名 {sym}")

    scan_args = []
    for item in items:
        sym = item["symbol"]
        use_strats = selected_strategies or active_list
        if use_ranked_top:
            cached_top = stock_top_cache.get(sym)
            if cached_top:
                use_strats = cached_top  # Already per-owner top_n merged
        scan_args.append((sym, item["market"], interval, period, use_strats, refresh_sec, signal_side, local_only))

    alerts = []
    errors = []
    stocks_scanned = 0
    if progress_hook:
        progress_hook("scanning", 0, len(scan_args), "扫描中...")
    # Per-market rotation: each market gets up to SCAN_WORKERS parallel workers,
    # but only one market is scanned at a time. Prevents the previous behavior
    # of spawning N_WORKERS × N_markets subprocesses simultaneously, which
    # saturated IB Gateway's 32-connection cap and triggered yfinance 429s
    # across all foreground /quotes calls.
    scan_by_market: dict[str, list] = {}
    for args in scan_args:
        scan_by_market.setdefault(args[1], []).append(args)
    done_scan = 0
    for market_name, market_scan_args in scan_by_market.items():
        n_scan_workers = min(SCAN_WORKERS, len(market_scan_args))
        with ProcessPoolExecutor(max_workers=n_scan_workers, initializer=proc_role.mark_pool_worker) as executor:
            futures = {executor.submit(_scan_one_stock_mp, args): args[0] for args in market_scan_args}
            for future in as_completed(futures):
                sym, stock_alerts, err = future.result()
                done_scan += 1
                if err:
                    errors.append(f"{sym}: {err}")
                else:
                    stocks_scanned += 1
                    alerts.extend(stock_alerts)
                if progress_hook:
                    progress_hook("scanning", done_scan, len(scan_args), f"{market_name} 扫描 {sym}")

    if use_ranked_top and stock_top_cache:
        cache_set(scan_rank_cache_key, stock_top_cache)

    # ── 📰 News-driven enrichment (owner=news, Phase 2) ─────────────────────
    # Runs serially after parallel technical scan, honors LLM CLI serialization rule.
    news_filtered_out = 0
    news_symbols_scored = 0
    if alerts and (filters or {}).get("news_require") in ("must", "weight"):
        alerts, news_filtered_out, news_symbols_scored = _selector_score_news(
            alerts, filters, progress_hook=progress_hook
        )
        if news_filtered_out:
            filter_reasons["news_below_min"] = filter_reasons.get("news_below_min", 0) + news_filtered_out

    if use_ranked_top:
        # Per-owner top_n: actual count varies by stock (depends on how many owners have strategies)
        sample_count = len(next(iter(stock_top_cache.values()), [])) if stock_top_cache else top_n
        strategies_desc = f"每Owner各Top{top_n}(约{sample_count}个/股)"
        strats_per_stock = sample_count
    else:
        strats_per_stock = len(selected_strategies or active_list)
        strategies_desc = (
            f"选中{len(selected_strategies)}个" if selected_strategies else f"全部{len(active_list)}个"
        )
    scan_time = datetime.now().isoformat()
    if alerts:
        log_path = Path("scan_signals.json")
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
            for a in alerts:
                existing.append({**a, "scan_time": scan_time})

            # 2026-04-27: 双限 rolling buffer，保留 max_days 天 OR 最多 max_records 条
            # （哪个先到都裁）。配置来自 news_buffer_config.json，由 UI 设置面板维护。
            # 默认 7 天 / 20000 条；用户可调到 90 天 / 200000 条。
            try:
                from routers.scan import _read_news_buffer_config
                _bcfg = _read_news_buffer_config()
                _max_days = _bcfg.get("max_days", 7)
                _max_records = _bcfg.get("max_records", 20000)
            except Exception:
                _max_days, _max_records = 7, 20000

            cutoff = (datetime.now() - timedelta(days=_max_days)).isoformat()
            existing = [a for a in existing
                        if (a.get("scan_time") or a.get("time") or "") >= cutoff]
            if len(existing) > _max_records:
                existing = existing[-_max_records:]

            log_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"scan log write failed: {e}")

    return {
        "alerts": alerts,
        "scan_time": scan_time,
        "stocks_scanned": stocks_scanned,
        "signal_side": signal_side,
        "strategies_used": strats_per_stock,
        "strategies_desc": strategies_desc,
        "errors": errors[:20] if errors else [],
        "total_symbols_before_filter": total_symbols_before_filter,
        "candidates_after_filter": candidates_after_filter,
        "filtered_out": filtered_out,
        "filter_reasons": filter_reasons,
        "filters_applied": filters,
        "news_filtered_out": news_filtered_out,
        "news_symbols_scored": news_symbols_scored,
    }


_cn_name_cache: dict = {}  # {market:symbol -> cn_name}

# ---------------------------------------------------------------------------
# v0.5 P6: Wire scan_center dependency injection.
# Must happen after _run_realtime_scan and _resolve_scan_universe are defined,
# so placed here at the bottom of main.py. scan_center cannot import main
# at module-load time (circular), so main injects the heavy functions here.
# ---------------------------------------------------------------------------


def _resolve_positions_universe_union() -> "list[dict]":
    """SELL 侧扫描用的持仓股票: 模拟交易里 status=='open' 的持仓。

    按 (symbol, market) 去重, 每条 ``{"symbol", "market", "holders"}``,
    holders = 该股票的未平仓模拟单数。从不抛异常。
    """
    seen: "dict[tuple, dict]" = {}

    try:
        data = _load_paper_trades() or {}
        for pos in data.get("positions", []) or []:
            if not isinstance(pos, dict):
                continue
            if str(pos.get("status") or "").lower() != "open":
                continue
            sym = str(pos.get("symbol") or "").strip()
            mkt = str(pos.get("market") or "").strip().upper()
            if not sym:
                continue
            if not mkt:
                mkt = "US"
            key = (sym, mkt)
            entry = seen.setdefault(key, {"symbol": sym, "market": mkt, "holders": 0})
            entry["holders"] += 1  # one paper holder per open paper position
    except Exception as exc:  # noqa: BLE001 — file missing / parse error
        logger.warning(
            "_resolve_positions_universe_union: paper positions read failed: %r", exc,
        )

    return list(seen.values())


try:
    import scan_center as _scan_center
    _scan_center.configure_realtime_scan_fn(_run_realtime_scan)
    _scan_center.configure_resolve_universe_fn(_resolve_scan_universe)
    # SELL 侧扫描只扫模拟交易里实际持有的股票; 没接上时退回 BUY 侧股票池。
    _scan_center.configure_positions_universe_fn(_resolve_positions_universe_union)
except Exception as _sc_wire_err:  # pragma: no cover — best-effort wiring
    import logging as _wire_log
    _wire_log.getLogger(__name__).warning(
        "scan_center wiring failed (unified_scan will error if used): %s",
        _sc_wire_err,
    )




























# ── 未捕获异常: 落盘留痕 ─────────────────────────────────────────────────────
# 为什么要有: 之前排查"进度查询 HTTP 500"时, 真实堆栈只在 uvicorn 的控制台里,
# 而 --reload 一重启就冲掉了, 只能靠猜。落一份到文件, 下次直接看。
#
# 挂在 logger 上而不是在 handler 里手写 open/write, 原因有三:
#   1. handler 是 async def —— 手写的 mkdir + open + write 是**同步**磁盘调用,
#      直接压在事件循环上。而这个 handler 恰恰在"一堆 2-3 秒轮询端点同时 500"
#      时最忙, 一边 500 一边阻塞事件循环, 会把本来健康的轮询也一起拖死。
#      logging 的 handler 自带锁, 且不在协程里同步等 I/O。
#   2. 没有轮转。每条是完整 traceback, 一个每拍都 500 的端点一天能写几百 MB,
#      而 .gitignore 又把 logs/ 挡了, git status 里看不见 —— 第一个症状会是
#      生产机磁盘写满, 然后 parquet 和 .cache 的原子写全线失败。
#   3. 手写那份和下面的 logger.exception 是同一条堆栈的两份拷贝, 格式还不一样;
#      再加上 Starlette 的 ServerErrorMiddleware 返回后还会再抛一次让 uvicorn
#      记一遍 —— 同一个错误记三遍, 反而更难读。
# 现在: 只调一次 logger.exception, 它同时进 uvicorn 控制台和带轮转的文件。
LOG_DIR = Path(os.getenv("AITRADER_LOG_DIR", str(Path(__file__).parent / "logs")))


def _install_error_logfile():
    """给 uvicorn.error 挂一个带轮转的文件 handler (幂等)。"""
    tag = "_aitrader_err_file"
    if any(getattr(h, tag, False) for h in logger.handlers):
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        h = RotatingFileHandler(
            LOG_DIR / "unhandled_errors.log",
            maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        h.setLevel(logging.ERROR)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        setattr(h, tag, True)
        logger.addHandler(h)
    except Exception as e:
        # 目录不可写(只读部署)时不该把服务拖垮, 但要让人知道文件里不会有东西
        logger.warning(f"[errlog] 文件日志挂载失败, 堆栈只进控制台: {e}")


_install_error_logfile()


@app.exception_handler(TaskCancelled)
async def _handle_task_cancelled(request: Request, exc: TaskCancelled):
    """用户主动停止不是错误, 别记进未捕获异常日志, 也别回 500。

    回 200 + cancelled 标记: 前端那几处调用方拿到后直接把面板收成"已取消",
    不用每个端点各写一遍 try/except。
    """
    return JSONResponse(status_code=200,
                        content={"cancelled": True, "message": "已按请求停止"})


@app.exception_handler(Exception)
async def _log_unhandled(request: Request, exc: Exception):
    logger.exception(f"[unhandled] {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务端错误: {type(exc).__name__}. "
                           f"详见 {LOG_DIR}/unhandled_errors.log"},
    )


# ── 轮询端点自检 ─────────────────────────────────────────────────────────────
# 进度/状态类 GET 必须是 async def。同步端点会被 FastAPI 丢进 anyio 线程池,
# 跟正在跑的重活 (/compare、/applicability 这些 sync 端点, 尤其是它们创建
# 进程池那几十秒) 抢名额, 轮询迟迟拿不到响应, 前端面板就冻在上一次拿到的状态,
# 表现为"卡在某一步不动"。这条检查让以后新写的轮询端点漏了 async 能当场看见。
_POLL_PATH_RE = re.compile(r"status|progress|running|/jobs")
#: 确实需要干重活、不适合放事件循环的, 在这里登记豁免
_POLL_ASYNC_EXEMPT = {
    "/wf-status/{market}/{symbol}",      # 要读参数文件并做判定, 不是纯内存读
    "/scan/market-status",               # 算各市场开闭市, 走日历库
    "/market-data/ib/status",            # 探活 IB 连接, 有网络 I/O
    "/data/status",                      # 遍历 market_data 全目录读 parquet 元信息
    "/data/fundamentals/status",         # 扫基本面文件统计覆盖率
    "/news/buffer-status",               # 解析 scan_signals.json (可能几十 MB)
    "/scan/monitor/status",              # 汇总多市场实时状态, 有磁盘/网络读
    "/scan/monitor/daemon-progress",     # 同上, 逐市场快照
}


@app.on_event("startup")
async def _check_poll_endpoints_are_async():
    import inspect as _inspect
    bad = []
    for r in app.routes:
        path = getattr(r, "path", "")
        methods = getattr(r, "methods", set()) or set()
        ep = getattr(r, "endpoint", None)
        if "GET" not in methods or not ep or not _POLL_PATH_RE.search(path):
            continue
        if path in _POLL_ASYNC_EXEMPT:
            continue
        if not _inspect.iscoroutinefunction(ep):
            bad.append(f"{path} -> {getattr(ep, '__name__', ep)}")
    if bad:
        logger.warning(
            "[poll-check] 以下轮询端点还是同步的, 会跟重活抢线程池导致前端进度卡住; "
            "改成 async def, 或在 _POLL_ASYNC_EXEMPT 里登记豁免并写明理由: %s",
            ", ".join(bad))
    else:
        logger.info("[poll-check] 所有进度/状态轮询端点均为 async ✓")
