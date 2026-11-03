#!/usr/bin/env python3
"""
自动优化流水线 v4 (多轮收敛 + 策略级并行)
==========================================
流程:
  1. 基准测试 — NASDAQ100 × 所有策略，标准参数跑回测
  2. 多轮全量优化，每轮:
     a. 逐股（N只），每只股票内所有策略并行贝叶斯优化
     b. 各策略同时 Walk-Forward 验证，筛出无过拟合的
     c. 通过验证的参数在 NASDAQ100 全池重跑
     d. 全池综合评分优于当前基准 → 升级为新基准
     e. 下一只股票，依此类推
  3. 收敛停止: 连续2轮升级数≤1 → 参数已稳定，停止

用法:
  python auto_optimize.py                      # 完整多轮流程
  python auto_optimize.py --resume             # 从断点恢复
  python auto_optimize.py --trials 100         # 贝叶斯试验次数(默认150)
  python auto_optimize.py --max-rounds 5       # 最大轮数(默认10)
  python auto_optimize.py --converge-rounds 3  # 连续N轮低升级视为收敛(默认2)
"""

import sys, io, os, json, time, copy, logging, argparse
import proc_role          # 池 worker 打标 + 父进程死了跟着退(防孤儿)
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY, get_strategy
from optimizer import optimize_strategy, PARAM_RANGES

# 并行数 — 用满 CPU（i9-13900KF 32线程）
N_WORKERS = max(multiprocessing.cpu_count() - 2, 4)       # 主池：留2线程给系统
N_STRATEGY_WORKERS = multiprocessing.cpu_count()           # 策略级：用全部核心

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s   %(message)s')
logger = logging.getLogger("auto_optimize")

# ── 配置 ──────────────────────────────────────────────────────────────
POOL_FILE = Path(__file__).parent / "stock_pools.json"
RESULT_DIR = Path(__file__).parent / "auto_optimize_results"
RESULT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = RESULT_DIR / "_checkpoint_v2.json"

# Walk-Forward 参数
WF_TRAIN_DAYS = 504    # 2年训练
WF_TEST_DAYS  = 126    # 6个月测试
WF_CONSISTENCY_MIN = 50.0
WF_OVERFIT_RATIO_MIN = 0.3

DATA_PERIOD = "5y"
MIN_SCORE_IMPROVEMENT = 0.005  # 全池综合评分至少提升 0.005 才升级
TRIM_N = 5  # 截尾平均：去掉前后各 N 只极端值

# 综合评分权重（偏重收益）
W_RETURN   = 0.40   # 收益率权重
W_SHARPE   = 0.25   # Sharpe 权重
W_DRAWDOWN = 0.20   # 最大回撤权重（越小越好）
W_WINRATE  = 0.10   # 胜率权重
W_PF       = 0.05   # Profit Factor 权重


# ═══════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════

def load_nasdaq100_symbols():
    """从 stock_pools.json 加载 NASDAQ 100 股票池"""
    with open(POOL_FILE) as f:
        pools = json.load(f)
    for name, pool in pools.items():
        if "nasdaq" in name.lower() or "NASDAQ" in name:
            return pool["symbols"], pool.get("market", "US")
    raise ValueError("找不到 NASDAQ 100 股票池")


def load_all_data(symbols, market, period=DATA_PERIOD):
    """批量下载股票数据（并行加速）"""
    datasets = {}
    total = len(symbols)

    def _fetch(sym):
        try:
            df = get_ohlcv(sym, market, interval="1d", period=period)
            return sym, df, None
        except Exception as e:
            return sym, None, str(e)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_fetch, sym): sym for sym in symbols}
        done = 0
        for future in as_completed(futures):
            sym, df, err = future.result()
            done += 1
            if err:
                logger.warning(f"[{done}/{total}] {sym}: 下载失败 - {err}")
            elif df is not None and len(df) >= 200:
                datasets[sym] = df
                logger.info(f"[{done}/{total}] {sym}: {len(df)} bars")
            else:
                bars = len(df) if df is not None else 0
                logger.warning(f"[{done}/{total}] {sym}: 数据不足 ({bars} bars)")
    return datasets


def save_checkpoint(data):
    """保存断点"""
    data["timestamp"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"💾 断点已保存")


def load_checkpoint():
    """加载断点"""
    if CHECKPOINT_FILE.exists():
        return json.load(open(CHECKPOINT_FILE))
    return {}


def safe_float(v, decimals=4):
    """安全转换为 float"""
    if isinstance(v, (int, float, np.floating, np.integer)):
        f = float(v)
        return round(f, decimals) if f == f else 0
    return v


def trimmed_mean(values, trim_n=TRIM_N):
    """截尾平均：排序后去掉前后各 trim_n 个极端值，取中间部分的平均"""
    if len(values) <= trim_n * 2:
        return float(np.mean(values)) if values else 0
    sorted_vals = sorted(values)
    trimmed = sorted_vals[trim_n:-trim_n]
    return float(np.mean(trimmed))


def composite_score(total_return, sharpe, max_drawdown, win_rate, profit_factor):
    """
    综合评分（偏重收益）
    各指标先做归一化映射到 [0, 1] 区间，再加权求和。

    归一化参考区间:
      收益率:       [-50%, +200%] → [0, 1]
      Sharpe:       [-1, +3]     → [0, 1]
      最大回撤:     [0%, 60%]    → [1, 0]  (越小越好)
      胜率:         [0%, 100%]   → [0, 1]
      ProfitFactor: [0, 5]       → [0, 1]
    """
    # 归一化到 [0, 1]，超出范围则 clip
    norm_ret = np.clip((total_return + 50) / 250, 0, 1)        # -50%→0, +200%→1
    norm_sharpe = np.clip((sharpe + 1) / 4, 0, 1)              # -1→0, +3→1
    norm_dd = np.clip(1 - max_drawdown / 60, 0, 1)             # 0%→1, 60%→0
    norm_wr = np.clip(win_rate / 100, 0, 1)                    # 0%→0, 100%→1
    norm_pf = np.clip(profit_factor / 5, 0, 1)                 # 0→0, 5→1

    score = (W_RETURN * norm_ret +
             W_SHARPE * norm_sharpe +
             W_DRAWDOWN * norm_dd +
             W_WINRATE * norm_wr +
             W_PF * norm_pf)
    return round(float(score), 4)


# ═══════════════════════════════════════════════════════════════════════
#  Step 1: 基准测试
# ═══════════════════════════════════════════════════════════════════════

def _backtest_one(args):
    """单只股票单策略回测（用于并行）"""
    sym, df, sname, params = args
    try:
        r = run_strategy_backtest(df, sname, params_override=params)
        s = r["summary"]
        score = composite_score(
            s["total_return"], s["sharpe"], s["max_drawdown"],
            s["win_rate"], s.get("profit_factor", 0)
        )
        return sym, sname, {
            "score": score, "sharpe": s["sharpe"],
            "total_return": s["total_return"], "max_drawdown": s["max_drawdown"],
            "win_rate": s["win_rate"], "total_trades": s["total_trades"],
            "profit_factor": s.get("profit_factor", 0),
        }
    except Exception:
        return sym, sname, {"score": 0, "sharpe": 0, "total_return": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Distributed-mode shim: route baseline through backtest_queue when
# --distributed is on. Falls back to the local ProcessPoolExecutor path.
# ─────────────────────────────────────────────────────────────────────────────


def run_baseline_distributed(symbols, market, strategies, current_params=None,
                             timeout_s=3600):
    """Distributed counterpart of run_baseline().

    Enqueues N_strategies × N_symbols tasks into backtest_queue and waits
    for the workers to finish. Worker pulls OHLCV via /internal/ohlcv —
    coordinator never holds the dataframes in memory.

    Returns the same shape as run_baseline (a per-stock dict per strategy
    with composite_score / sharpe / drawdown / etc).
    """
    import backtest_queue as bq
    import uuid

    t0 = time.time()
    batch_id = f"auto_opt_baseline_{uuid.uuid4().hex[:12]}"

    payloads = []
    payload_index = {}  # task_uuid → (sym, sname) lookup
    for sname in strategies:
        params = current_params.get(sname) if current_params else None
        for sym in symbols:
            payloads.append({
                "symbol": sym, "market": market,
                "interval": "1d", "period": DATA_PERIOD,
                "strategy": sname,
                "params": params or {},
                "dynamic_pos": False,
            })

    logger.info(f"  基准[distributed]: 入队 {len(payloads)} task → batch_id={batch_id}")
    uuids = bq.enqueue_batch(payloads, batch_id=batch_id)
    # Index payloads by task_uuid for result lookup
    for tu, p in zip(uuids, payloads):
        payload_index[tu] = (p["symbol"], p["strategy"])

    def _progress(done, total):
        if done % 100 == 0 or done == total:
            elapsed = time.time() - t0
            logger.info(f"  基准[distributed] 进度: {done}/{total} ({elapsed:.0f}s)")

    try:
        results = bq.wait_for_batch(batch_id, timeout=timeout_s, poll_interval=2.0,
                                    progress_cb=_progress)
    except TimeoutError:
        # Don't leave orphan tasks churning workers forever — cancel the
        # remainder, then salvage whatever completed before the deadline.
        n_cancelled = bq.cancel_batch(batch_id)
        logger.warning(
            f"  基准[distributed] 超时({timeout_s}s)：已取消 {n_cancelled} 个未完成 task，"
            f"使用已完成的部分结果继续")
        results = bq._batch_results(batch_id)
    elapsed = time.time() - t0
    logger.info(f"  基准[distributed] 完成: {len(results)} task in {elapsed:.0f}s")

    # Translate back to run_baseline's output shape.
    # Note: failed tasks are *not* inserted into results_map with zeros —
    # zero-filling would skew trimmed_mean and unfairly penalize strategies
    # whose tasks happened to fail (e.g., transient yfinance 429 or worker
    # crash). Treating a missing entry as "this stock didn't contribute"
    # is the same behavior as the local ProcessPoolExecutor path, which
    # similarly skips errored datasets.
    results_map = {sname: {} for sname in strategies}
    failed_count = {sname: 0 for sname in strategies}
    for tu, entry in results.items():
        if tu not in payload_index:
            continue
        sym, sname = payload_index[tu]
        if entry["status"] != "completed" or entry["result"] is None:
            failed_count[sname] += 1
            continue
        s = entry["result"]
        score = composite_score(
            s.get("total_return", 0), s.get("sharpe", 0),
            s.get("max_drawdown", 0), s.get("win_rate", 0),
            s.get("profit_factor", 0),
        )
        results_map[sname][sym] = {
            "score": score, "sharpe": s.get("sharpe", 0),
            "total_return": s.get("total_return", 0),
            "max_drawdown": s.get("max_drawdown", 0),
            "win_rate": s.get("win_rate", 0),
            "total_trades": s.get("total_trades", 0),
            "profit_factor": s.get("profit_factor", 0),
        }

    # Log failed-task counts so silent flakiness doesn't get hidden
    for sname, n in failed_count.items():
        if n > 0:
            logger.warning(
                "  基准[distributed] %s: %d/%d task 失败（已跳过，不计入均值）",
                sname, n, len(symbols),
            )

    # Wrap in the same avg-aggregate shape as run_baseline returns
    out = {}
    for sname in strategies:
        per_stock = results_map[sname]
        if not per_stock:
            out[sname] = {"avg_score": 0, "avg_sharpe": 0, "positive_pct": 0, "per_stock": {}}
            continue
        scores = [v["score"] for v in per_stock.values()]
        sharpes = [v["sharpe"] for v in per_stock.values()]
        positive_pct = sum(1 for s in sharpes if s > 0) / len(sharpes) * 100 if sharpes else 0
        out[sname] = {
            "avg_score": round(trimmed_mean(scores), 4),
            "avg_sharpe": round(trimmed_mean(sharpes), 4),
            "positive_pct": round(positive_pct, 1),
            "per_stock": per_stock,
        }
    return out


def run_baseline(datasets, strategies, current_params=None):
    """
    用指定参数（或默认参数）跑全池回测（并行加速版）。
    返回: {strategy_name: {"avg_score": x, "avg_sharpe": x, "per_stock": {...}}}
    """
    t0 = time.time()
    # 构建所有任务
    tasks = []
    for sname in strategies:
        params = current_params.get(sname) if current_params else None
        for sym, df in datasets.items():
            tasks.append((sym, df, sname, params))

    logger.info(f"  基准测试: {len(tasks)} 个任务 ({len(strategies)}策略 x {len(datasets)}股), {N_WORKERS}线程并行")

    # 并行执行
    results_map = {}  # {sname: {sym: {...}}}
    for sname in strategies:
        results_map[sname] = {}

    done_count = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=proc_role.mark_pool_worker) as pool:
        futures = [pool.submit(_backtest_one, t) for t in tasks]
        for future in as_completed(futures):
            sym, sname, result = future.result()
            results_map[sname][sym] = result
            done_count += 1
            if done_count % 100 == 0:
                logger.info(f"  基准进度: {done_count}/{len(tasks)} ({time.time()-t0:.0f}s)")

    logger.info(f"  基准测试完成: {len(tasks)} 个任务, 耗时 {time.time()-t0:.0f}s")

    # 汇总
    baseline = {}
    for sname in strategies:
        per_stock = results_map[sname]
        scores = [v["score"] for v in per_stock.values()]
        sharpes = [v["sharpe"] for v in per_stock.values()]
        avg_score = trimmed_mean(scores) if scores else 0
        avg_sharpe = trimmed_mean(sharpes) if sharpes else 0
        positive_pct = sum(1 for s in sharpes if s > 0) / len(sharpes) * 100 if sharpes else 0
        baseline[sname] = {
            "avg_score": round(avg_score, 4),
            "avg_sharpe": round(avg_sharpe, 4),
            "positive_pct": round(positive_pct, 1),
            "per_stock": per_stock,
        }
    return baseline


def print_baseline(baseline, title="基准"):
    """打印基准排名"""
    sorted_b = sorted(baseline.items(), key=lambda x: x[1]["avg_score"], reverse=True)
    print(f"\n  {'─'*70}")
    print(f"  {title}排名 (按综合评分, 权重: 收益40% Sharpe25% 回撤20% 胜率10% PF5%):")
    for i, (sname, b) in enumerate(sorted_b):
        star = " ⭐" if b["avg_score"] > 0.5 else ""
        print(f"    {i+1:>2}. {sname:<22} 综合={b['avg_score']:.4f}  "
              f"Sharpe={b['avg_sharpe']:.3f}  正收益率={b['positive_pct']:.0f}%{star}")


# ═══════════════════════════════════════════════════════════════════════
#  Walk-Forward 验证
# ═══════════════════════════════════════════════════════════════════════

def walk_forward_validate(df, strategy_name, params_override,
                          train_days=WF_TRAIN_DAYS, test_days=WF_TEST_DAYS):
    """用指定参数做 Walk-Forward 验证"""
    windows = []
    i = 0
    while i + train_days + test_days <= len(df):
        train_df = df.iloc[i:i+train_days]
        test_df  = df.iloc[i+train_days:i+train_days+test_days]

        try:
            train_r = run_strategy_backtest(train_df, strategy_name,
                                             params_override=params_override)
            train_sharpe = train_r["summary"]["sharpe"]
        except Exception:
            train_sharpe = 0

        try:
            test_r = run_strategy_backtest(test_df, strategy_name,
                                            params_override=params_override)
            test_sharpe = test_r["summary"]["sharpe"]
        except Exception:
            test_sharpe = 0

        windows.append({"train_sharpe": train_sharpe, "test_sharpe": test_sharpe})
        i += test_days

    if not windows:
        return {"passed": False, "reason": "数据不足",
                "consistency": 0, "overfit_ratio": 0,
                "avg_test_sharpe": 0, "avg_train_sharpe": 0, "num_windows": 0}

    test_sharpes = [w["test_sharpe"] for w in windows]
    train_sharpes = [w["train_sharpe"] for w in windows]

    consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100
    avg_test = float(np.mean(test_sharpes))
    avg_train = float(np.mean(train_sharpes))
    overfit_ratio = avg_test / max(avg_train, 0.01)

    passed = (consistency >= WF_CONSISTENCY_MIN and overfit_ratio >= WF_OVERFIT_RATIO_MIN)

    return {
        "passed": passed,
        "consistency": round(consistency, 1),
        "overfit_ratio": round(overfit_ratio, 3),
        "avg_test_sharpe": round(avg_test, 4),
        "avg_train_sharpe": round(avg_train, 4),
        "num_windows": len(windows),
    }


# ═══════════════════════════════════════════════════════════════════════
#  全池快速回测（单策略）
# ═══════════════════════════════════════════════════════════════════════

def fullpool_score(datasets, strategy_name, params):
    """用指定参数在全池跑一遍，返回截尾平均综合评分（并行加速版）"""
    tasks = [(sym, df, strategy_name, params) for sym, df in datasets.items()]

    scores = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=proc_role.mark_pool_worker) as pool:
        futures = [pool.submit(_backtest_one, t) for t in tasks]
        for future in as_completed(futures):
            _, _, result = future.result()
            scores.append(result["score"])
    return trimmed_mean(scores) if scores else 0


def fullpool_score_incremental(baseline, datasets, strategy_name, new_params, changed_sym):
    """
    增量全池评分：只重跑参数变化的那只股票，其余复用baseline缓存的per_stock分数。
    比fullpool_score快约100倍（1次回测 vs 100次回测）。
    """
    per_stock = baseline[strategy_name].get("per_stock", {})
    scores = []

    for sym in datasets:
        if sym == changed_sym:
            # 只重跑这只股票
            try:
                r = run_strategy_backtest(datasets[sym], strategy_name, params_override=new_params)
                s = r["summary"]
                score = composite_score(
                    s["total_return"], s["sharpe"], s["max_drawdown"],
                    s["win_rate"], s.get("profit_factor", 0)
                )
            except Exception:
                score = 0
        else:
            # 复用缓存分数
            score = per_stock.get(sym, {}).get("score", 0)
        scores.append(score)

    return trimmed_mean(scores) if scores else 0


# ═══════════════════════════════════════════════════════════════════════
#  主流程：逐股迭代优化
# ═══════════════════════════════════════════════════════════════════════

def _optimize_one_strategy(args):
    """单策略优化 worker（用于并行）：贝叶斯优化 + Walk-Forward 验证 + 增量评分"""
    stock_df, sname, n_trials, baseline_score = args
    result = {"strategy": sname, "status": "skip"}

    if sname not in PARAM_RANGES:
        return result

    # ── 贝叶斯优化 ──
    try:
        best_params, best_sharpe, best_summary, combos, method = \
            optimize_strategy(stock_df, sname, method="bayesian",
                              n_trials=n_trials, dynamic_pos=False)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    if best_sharpe <= 0 or best_params is None:
        result["status"] = "no_improvement"
        return result

    # ── Walk-Forward 验证 ──
    wf = walk_forward_validate(stock_df, sname, best_params)

    if not wf["passed"]:
        result["status"] = "wf_fail"
        result["sharpe"] = best_sharpe
        result["wf"] = wf
        return result

    result["status"] = "wf_pass"
    result["sharpe"] = best_sharpe
    result["wf"] = wf
    result["best_params"] = best_params
    result["baseline_score"] = baseline_score
    return result


def run_one_round(datasets, stock_list, strategies, current_params, baseline,
                  n_trials, round_num):
    """
    全量 N股×M策略 并行优化（股票×策略对全部扔进大池子，充分利用多核）。
    架构：
      1. 把所有 (股票, 策略) 对构建为任务列表
      2. 用大池子（N_WORKERS）并行跑贝叶斯优化+WF验证
      3. 主线程串行处理结果：增量全池评分→升级决策
    """
    total_stocks = len(stock_list)
    round_upgrades = 0
    round_log = []

    valid_strategies = [s for s in strategies if s in PARAM_RANGES]
    n_strats = len(valid_strategies)
    total_tasks = total_stocks * n_strats

    print(f"\n{'='*80}")
    print(f"  [Round {round_num}] 全量并行优化: {total_stocks}只 x {n_strats}策略 = {total_tasks}任务")
    print(f"  并行进程数: {N_WORKERS}  (CPU: {multiprocessing.cpu_count()}线程)")
    print(f"{'='*80}")

    # ── Step 1: 构建所有(股票×策略)任务，一次性提交 ──
    all_tasks = [
        (datasets[sym], sname, n_trials, baseline[sname]["avg_score"])
        for sym in stock_list
        for sname in valid_strategies
    ]
    # 记录任务顺序：(sym, sname)
    task_keys = [
        (sym, sname)
        for sym in stock_list
        for sname in valid_strategies
    ]

    # ── Step 2: 并行跑贝叶斯+WF（全部核心一起用） ──
    t0 = time.time()
    completed = 0
    # 按策略分组收集候选升级
    candidates = {sname: [] for sname in valid_strategies}

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=proc_role.mark_pool_worker) as pool:
        future_to_key = {
            pool.submit(_optimize_one_strategy, task): key
            for task, key in zip(all_tasks, task_keys)
        }
        for fut in as_completed(future_to_key):
            sym, sname = future_to_key[fut]
            completed += 1
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"  [{sym}] {sname}: 异常 - {e}")
                continue

            if res["status"] == "wf_pass":
                candidates[sname].append((sym, res))
            elif res["status"] == "wf_fail":
                wf = res.get("wf", {})
                logger.info(f"  [{sym}] {sname}: WF失败 "
                            f"一致性={wf.get('consistency',0):.0f}% "
                            f"过拟合={wf.get('overfit_ratio',0):.2f}")

            if completed % 30 == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                pct = completed / total_tasks * 100
                eta = elapsed / completed * (total_tasks - completed) if completed > 0 else 0
                print(f"  进度: {completed}/{total_tasks} ({pct:.0f}%) | "
                      f"耗时 {elapsed:.0f}s | 预计剩余 {eta:.0f}s")

    elapsed_parallel = time.time() - t0
    print(f"\n  并行阶段完成: {total_tasks}任务, 耗时 {elapsed_parallel:.0f}s "
          f"({elapsed_parallel/total_tasks*1000:.0f}ms/任务)")

    # ── Step 3: 主线程串行处理候选升级（增量评分+确认+应用）──
    print(f"\n  处理候选升级...")
    for sname in valid_strategies:
        cands = candidates[sname]
        if not cands:
            print(f"    {sname:<22}: 无WF通过候选")
            continue

        current_baseline_score = baseline[sname]["avg_score"]
        print(f"    {sname:<22}: {len(cands)}个候选 (基准分={current_baseline_score:.4f})")

        # 对每个候选做增量全池评分，找最佳
        best_delta = MIN_SCORE_IMPROVEMENT
        best_sym = None
        best_res = None
        best_incr = None

        for sym, res in cands:
            incr = fullpool_score_incremental(
                baseline, datasets, sname, res["best_params"], sym)
            delta = incr - current_baseline_score
            if delta > best_delta:
                best_delta = delta
                best_sym = sym
                best_res = res
                best_incr = incr

        if best_res is None:
            print(f"      -> 无候选超越基准")
            continue

        # 增量通过 → 全池确认
        opt_fullpool_full = fullpool_score(datasets, sname, best_res["best_params"])
        delta_full = opt_fullpool_full - current_baseline_score

        if delta_full > MIN_SCORE_IMPROVEMENT:
            current_params[sname] = best_res["best_params"]
            baseline[sname]["avg_score"] = round(opt_fullpool_full, 4)
            # 并行重建per_stock缓存
            rebuild_tasks = [(s2, df2, sname, best_res["best_params"]) for s2, df2 in datasets.items()]
            with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=proc_role.mark_pool_worker) as _pool:
                _futs = [_pool.submit(_backtest_one, t) for t in rebuild_tasks]
                for _f in as_completed(_futs):
                    _sym2, _sn, _r = _f.result()
                    baseline[sname].setdefault("per_stock", {})[_sym2] = _r

            round_upgrades += 1
            wf = best_res["wf"]
            round_log.append({
                "round": round_num, "stock": best_sym, "strategy": sname,
                "old_score": round(current_baseline_score, 4),
                "new_score": round(opt_fullpool_full, 4),
                "delta": round(delta_full, 4),
                "params": best_res["best_params"],
                "wf_consistency": wf["consistency"],
                "wf_overfit_ratio": wf["overfit_ratio"],
                "timestamp": datetime.now().isoformat(),
            })
            print(f"      -> [{best_sym}] 全池 {current_baseline_score:.4f} -> "
                  f"{opt_fullpool_full:.4f} (delta={delta_full:+.4f}) [UPGRADE]")
        else:
            print(f"      -> [{best_sym}] 全池确认未过 ({opt_fullpool_full:.4f}, "
                  f"delta={delta_full:+.4f})")

    # ── 保存断点 ──
    save_checkpoint({
        "current_params": current_params,
        "current_round": round_num,
        "round_upgrades": round_upgrades,
        "round_log": round_log,
        "baseline_snapshot": {s: {"avg_score": b["avg_score"], "avg_sharpe": b.get("avg_sharpe", 0)}
                              for s, b in baseline.items()},
    })

    return round_upgrades, round_log


def main():
    parser = argparse.ArgumentParser(description="自动策略参数优化 v3 (多轮收敛)")
    parser.add_argument("--resume", action="store_true", help="从断点恢复")
    parser.add_argument("--trials", type=int, default=150, help="贝叶斯优化试验次数 (默认150)")
    parser.add_argument("--max-rounds", type=int, default=10, help="最大轮数 (默认10)")
    parser.add_argument("--converge-rounds", type=int, default=2, help="连续N轮升级≤1视为收敛 (默认2)")
    parser.add_argument("--strategies", type=str, default="", help="指定策略名,逗号分隔")
    parser.add_argument("--distributed", action="store_true",
                        help="把 baseline + per-stock 重跑分发到 backtest workers (PG queue)；默认本机 ProcessPoolExecutor")
    parser.add_argument("--distributed-timeout", type=int, default=3600,
                        help="--distributed 时单个 batch 的等待超时（秒，默认 3600）")
    args = parser.parse_args()

    t_start = time.time()

    print(f"\n{'='*80}")
    print(f"  [AI Trader] 自动参数优化 v4 (股票x策略全并行)")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  最大轮数: {args.max_rounds}  收敛条件: 连续{args.converge_rounds}轮升级<=1")
    print(f"  CPU: {multiprocessing.cpu_count()}线程  并行进程: {N_WORKERS}")
    print(f"{'='*80}")

    # ── 加载数据 ──
    symbols, market = load_nasdaq100_symbols()
    print(f"\n  股票池: NASDAQ 100 ({len(symbols)} 只)")

    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies = [name for name in STRATEGY_REGISTRY.keys() if name in PARAM_RANGES]
    print(f"  策略数: {len(strategies)}")
    print(f"  贝叶斯试验数: {args.trials}")

    if args.distributed:
        # Workers fetch OHLCV themselves via /internal/ohlcv. Coordinator
        # doesn't need datasets in memory, and a full load_all_data takes
        # 5-10 min on 100 symbols even with the IB-backed path. Skip it
        # and trust that worker-level errors will be surfaced via
        # failed_count logging in run_baseline_distributed.
        print(f"\n  [distributed] 跳过 coordinator OHLCV 下载；worker 端会自行拉取")
        datasets = {}  # kept for downstream code that references it; not used in --distributed
        stock_list = list(symbols)
    else:
        print(f"\n  下载股票数据 ({DATA_PERIOD})...")
        datasets = load_all_data(symbols, market)
        print(f"  有效股票: {len(datasets)}/{len(symbols)}")
        if len(datasets) < 10:
            print("  错误: 有效股票不足10只，终止")
            return
        stock_list = list(datasets.keys())

    # ── 加载断点 ──
    checkpoint = load_checkpoint() if args.resume else {}
    current_params = checkpoint.get("current_params", {})
    all_upgrade_log = checkpoint.get("all_upgrade_log", [])
    round_history = checkpoint.get("round_history", [])  # [(round_num, upgrades), ...]
    start_round = checkpoint.get("next_round", 1)

    # ══════════════════════════════════════════════════════════════
    #  Step 1: 基准测试
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  基准测试 — {len(datasets)}只股票 x {len(strategies)}个策略")
    print(f"{'='*80}")

    if args.distributed:
        # Skip pre-loaded datasets — workers will pull OHLCV via /internal/ohlcv.
        # We still need the stock_list to enqueue tasks.
        baseline = run_baseline_distributed(
            stock_list, market, strategies,
            current_params if current_params else None,
            timeout_s=args.distributed_timeout,
        )
    else:
        baseline = run_baseline(datasets, strategies, current_params if current_params else None)
    print_baseline(baseline, "当前基准")

    # In --distributed we skipped load_all_data above (workers fetch OHLCV
    # themselves for the baseline). But Round 1+ per-strategy bayesian
    # optimization is still a local ProcessPoolExecutor path that needs
    # datasets in memory. Lazy-load them now, after the cheap baseline.
    if args.distributed and not datasets:
        print(f"\n  [distributed] baseline 完成，加载 datasets 给 Round 1+ 本地优化用 ({DATA_PERIOD})...")
        datasets = load_all_data(symbols, market)
        print(f"  有效股票: {len(datasets)}/{len(symbols)}")
        if len(datasets) < 10:
            print("  错误: 有效股票不足10只，终止")
            return
        # Re-derive stock_list from the loaded set; some baseline-only symbols
        # might now be filtered out by load_all_data's "≥200 bars" check.
        stock_list = list(datasets.keys())

    # ══════════════════════════════════════════════════════════════
    #  多轮优化直到收敛
    # ══════════════════════════════════════════════════════════════
    consecutive_low = 0  # 连续低升级轮数

    for round_num in range(start_round, args.max_rounds + 1):
        round_t0 = time.time()

        round_upgrades, round_log = run_one_round(
            datasets, stock_list, strategies, current_params, baseline,
            args.trials, round_num
        )

        round_elapsed = time.time() - round_t0
        all_upgrade_log.extend(round_log)
        round_history.append((round_num, round_upgrades))

        # ── 本轮报告 ──
        print(f"\n{'─'*80}")
        print(f"  Round {round_num} 完成: 升级 {round_upgrades} 次, "
              f"耗时 {round_elapsed/60:.1f} 分钟")
        print(f"  历史: {' -> '.join(f'R{r}:{u}' for r, u in round_history)}")
        print(f"{'─'*80}")

        # ── 收敛检查 ──
        if round_upgrades <= 1:
            consecutive_low += 1
            print(f"  低升级轮数: {consecutive_low}/{args.converge_rounds}")
        else:
            consecutive_low = 0

        if consecutive_low >= args.converge_rounds:
            print(f"\n  收敛! 连续 {args.converge_rounds} 轮升级<=1，停止优化。")
            break

        # 保存断点供下一轮恢复
        save_checkpoint({
            "current_params": current_params,
            "all_upgrade_log": all_upgrade_log,
            "round_history": round_history,
            "next_round": round_num + 1,
            "baseline_snapshot": {s: {"avg_score": b["avg_score"], "avg_sharpe": b.get("avg_sharpe", 0)}
                                for s, b in baseline.items()},
        })

        print(f"\n  ▶️  进入 Round {round_num + 1}...")

    else:
        print(f"\n  ⚠️  达到最大轮数 {args.max_rounds}，停止优化。")

    # ══════════════════════════════════════════════════════════════
    #  最终报告
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    print(f"\n{'═'*80}")
    print(f"  📋 最终报告")
    print(f"{'═'*80}")
    print(f"  总轮数:      {len(round_history)}")
    print(f"  收敛曲线:    {' → '.join(f'R{r}:{u}次' for r, u in round_history)}")
    print(f"  总升级次数:  {sum(u for _, u in round_history)}")
    print(f"  总耗时:      {elapsed/60:.1f} 分钟")

    # 每个策略最终状态
    print(f"\n  {'策略':<22} {'综合评分':>10} {'已优化':>8}")
    print(f"  {'─'*45}")
    for sname in sorted(strategies, key=lambda s: baseline[s]["avg_score"], reverse=True):
        has_opt = "✅" if sname in current_params else "—"
        print(f"  {sname:<22} {baseline[sname]['avg_score']:>10.4f} {has_opt:>8}")

    # 升级日志
    if all_upgrade_log:
        print(f"\n  📝 全部升级日志 ({len(all_upgrade_log)} 次):")
        print(f"  {'轮':>3} {'股票':<8} {'策略':<22} {'旧分':>8} {'新分':>8} {'Δ':>8}")
        print(f"  {'─'*62}")
        for entry in all_upgrade_log:
            print(f"  R{entry['round']:<2} {entry['stock']:<8} {entry['strategy']:<22} "
                  f"{entry['old_score']:>8.4f} {entry['new_score']:>8.4f} {entry['delta']:>+8.4f}")

    # 保存报告
    report = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed / 60, 1),
        "total_rounds": len(round_history),
        "convergence_curve": round_history,
        "total_upgrades": sum(u for _, u in round_history),
        "final_params": current_params,
        "final_baseline": {s: {"avg_score": b["avg_score"], "avg_sharpe": b.get("avg_sharpe", 0)}
                          for s, b in baseline.items()},
        "all_upgrade_log": all_upgrade_log,
    }
    report_file = RESULT_DIR / f"report_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📄 报告已保存: {report_file.name}")

    # 更新 strategies.py
    if current_params:
        print(f"\n  🔄 更新 strategies.py 默认参数...")
        update_strategies_file(current_params)

    print(f"\n  ⏱️  总耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f}s)")
    print(f"  🎉 自动优化 v3 完成！\n")


def update_strategies_file(updated_params):
    """将优化后的参数保存到 JSON（strategies.py 已迁移为 strategies/ 包）"""
    strategies_file = Path(__file__).parent / "strategies.py"

    # 如果旧的 strategies.py 仍存在（单文件时代），尝试更新
    if strategies_file.exists() and strategies_file.is_file():
        try:
            content = strategies_file.read_text(encoding="utf-8")
            updated_count = 0
            for sname, new_params in updated_params.items():
                try:
                    strat = get_strategy(sname)
                    old_params = strat.default_params
                    merged = {**old_params, **new_params}
                    old_dict_str = repr(old_params)
                    new_dict_str = repr(merged)
                    if old_dict_str in content:
                        content = content.replace(old_dict_str, new_dict_str)
                        updated_count += 1
                        print(f"    ✅ {sname}: 已更新 default_params")
                    else:
                        print(f"    ⚠️  {sname}: 无法自动替换 (格式不匹配), 参数已保存到JSON")
                except Exception as e:
                    print(f"    ❌ {sname}: 更新失败 - {e}")
            if updated_count > 0:
                strategies_file.write_text(content, encoding="utf-8")
                print(f"    📝 strategies.py 已更新 {updated_count} 个策略")
        except Exception as e:
            print(f"    ⚠️  strategies.py 更新跳过: {e}")
    else:
        print(f"    ℹ️  strategies.py 已迁移为 strategies/ 包，跳过源文件更新，仅保存JSON参数")

    # 保存 JSON 备份（始终执行）
    backup_file = RESULT_DIR / f"params_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_file, "w") as f:
        json.dump(updated_params, f, indent=2, ensure_ascii=False, default=str)
    print(f"    💾 参数备份: {backup_file.name}")


if __name__ == "__main__":
    main()
