"""
通用策略参数优化器 —— 对所有 top 策略做网格搜索
输出 JSON 文件以便后续分析
"""
import sys, json, time
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest
import numpy as np
from itertools import product

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]

# ═══════════════════════════════════════════════════════════════════
#  各策略的参数搜索空间
# ═══════════════════════════════════════════════════════════════════

STRATEGY_GRIDS = {
    "claude_dual_ma_cross": {
        "fast_period":  [5, 8, 10, 13, 15, 20],
        "slow_period":  [20, 25, 30, 40, 50, 60],
        "ma_type":      ["sma", "ema"],
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0],
    },
    "claude_rsi_macd_trend": {
        "rsi_entry":    [30, 35, 40, 45],
        "rsi_exit":     [70, 75, 80, 85],
        "ema_trend":    [30, 40, 50, 60, 80],
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0],
    },
    "claude_vwap_breakout": {
        "vwap_period":   [10, 15, 20, 30],
        "atr_band_mul":  [0.5, 0.8, 1.0, 1.2, 1.5],
        "vol_ratio_min": [0.8, 1.0, 1.2, 1.5],
        "atr_stop_mul":  [1.5, 2.0, 2.5, 3.0],
    },
    "claude_momentum_breakout": {
        "breakout_period": [10, 15, 20, 25, 30],
        "exit_period":     [5, 8, 10, 15],
        "vol_ratio_min":   [0.8, 1.0, 1.2, 1.5],
        "atr_stop_mul":    [1.5, 2.0, 2.5, 3.0],
    },
    "claude_ensemble_voting": {
        "buy_threshold":    [1, 2],
        "sell_threshold":   [2, 3, 4],
        "mr_rsi_oversold":  [25, 30, 35],
        "mr_rsi_overbought":[70, 75, 80],
        "mr_bb_std":        [2.0, 2.5, 3.0],
        "dm_fast":          [10, 13, 15],
        "dm_slow":          [25, 30, 40],
    },
    "claude_adaptive_ma": {
        "er_period":    [8, 10, 15, 20],
        "fast_sc":      [2, 3, 5],
        "slow_sc":      [20, 30, 40],
        "threshold":    [0.001, 0.002, 0.003, 0.005],
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0],
    },
    "claude_turtle_trading": {
        "entry_period": [10, 15, 20, 25, 30],
        "exit_period":  [5, 8, 10, 15],
        "atr_stop_mul": [1.5, 2.0, 2.5, 3.0],
    },
}


def load_data(period="2y"):
    print(f"加载 {period} 数据...")
    datasets = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, interval="1d", period=period)
            datasets[(symbol, market)] = df
            print(f"  {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  {symbol}: FAIL - {e}")
    return datasets


def optimize_strategy(strategy_name, param_grid, datasets, top_n=10):
    """对单个策略做网格搜索，返回排序后的结果列表"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    total = len(combos)
    print(f"\n{'='*70}")
    print(f"  优化 {strategy_name}: {total} 组合 x {len(datasets)} 股票")
    print(f"{'='*70}")

    results = []
    t0 = time.time()
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        sharpes = []
        returns = []
        drawdowns = []
        for (symbol, market), df in datasets.items():
            try:
                r = run_strategy_backtest(df, strategy_name, params_override=params)
                sharpes.append(r["summary"]["sharpe"])
                returns.append(r["summary"]["total_return"])
                drawdowns.append(r["summary"]["max_drawdown"])
            except:
                sharpes.append(0)
                returns.append(0)
                drawdowns.append(-100)

        avg_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes))
        avg_return = float(np.mean(returns))
        avg_dd = float(np.mean(drawdowns))
        min_sharpe = float(np.min(sharpes))

        results.append({
            "params": params,
            "avg_sharpe": round(avg_sharpe, 4),
            "std_sharpe": round(std_sharpe, 4),
            "min_sharpe": round(min_sharpe, 4),
            "avg_return": round(avg_return, 2),
            "avg_drawdown": round(avg_dd, 2),
            "per_stock": {sym: round(s, 3) for (sym, _), s in zip(datasets.keys(), sharpes)},
        })

        if (idx + 1) % max(1, total // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"  进度: {idx+1}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # 按平均夏普排序
    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    elapsed = time.time() - t0
    print(f"\n  完成! 耗时 {elapsed:.1f}s")
    print(f"\n  TOP {top_n}:")
    print(f"  {'排名':<4} {'AvgSharpe':>9} {'Std':>6} {'MinS':>6} {'AvgRet%':>8} {'AvgDD%':>7} | 参数")
    print(f"  {'-'*80}")
    for i, r in enumerate(results[:top_n]):
        p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  {i+1:<4} {r['avg_sharpe']:>9.3f} {r['std_sharpe']:>6.3f} {r['min_sharpe']:>6.2f} "
              f"{r['avg_return']:>8.1f} {r['avg_drawdown']:>7.1f} | {p_str}")

    return results


def main():
    all_results = {}

    for period in ["2y", "5y"]:
        datasets = load_data(period)
        if len(datasets) < 4:
            print(f"数据不足，跳过 {period}")
            continue

        period_results = {}
        for strategy_name, grid in STRATEGY_GRIDS.items():
            results = optimize_strategy(strategy_name, grid, datasets)
            period_results[strategy_name] = {
                "top10": results[:10],
                "best_params": results[0]["params"],
                "best_sharpe": results[0]["avg_sharpe"],
                "total_combos": len(results),
            }

        all_results[period] = period_results

    # 保存结果
    output_file = "optimization_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n结果已保存到 {output_file}")

    # 最终排行榜
    print(f"\n{'='*70}")
    print(f"  最终排行榜（各周期最优参数）")
    print(f"{'='*70}")
    for period, strategies in all_results.items():
        print(f"\n  ── {period} ──")
        ranking = sorted(strategies.items(), key=lambda x: x[1]["best_sharpe"], reverse=True)
        for rank, (name, data) in enumerate(ranking, 1):
            p_str = ", ".join(f"{k}={v}" for k, v in data["best_params"].items())
            print(f"  {rank}. {name:<20} Sharpe={data['best_sharpe']:.3f}  | {p_str}")


if __name__ == "__main__":
    main()
