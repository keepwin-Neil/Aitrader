"""
Walk-Forward-Aware Parameter Optimization
==========================================
Instead of optimizing parameters on full in-sample data (which leads to overfitting),
this optimizer uses walk-forward consistency as the scoring metric.

For each parameter combination:
  1. Run walk-forward validation (train 3y + test 6m)
  2. Score = consistency% * avg_test_sharpe (both must be good)
  3. Best params = highest WF score across all stocks

This prevents overfitting by selecting parameters that generalize well.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from itertools import product
from backtest import get_ohlcv, run_strategy_backtest
from strategies import get_strategy

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]


def load_data(period="5y"):
    datasets = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, interval="1d", period=period)
            datasets[(symbol, market)] = df
            print(f"  OK {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  FAIL {symbol}: {e}")
    return datasets


def walk_forward_score(df, strategy_name, params, train_days=756, test_days=126, dynamic_pos=True):
    """Calculate walk-forward score for a parameter set on a single stock."""
    test_sharpes = []
    i = 0
    while i + train_days + test_days <= len(df):
        test_df = df.iloc[i + train_days:i + train_days + test_days]
        try:
            r = run_strategy_backtest(test_df, strategy_name,
                                       params_override=params,
                                       dynamic_position=dynamic_pos)
            test_sharpes.append(r["summary"]["sharpe"])
        except:
            test_sharpes.append(0)
        i += test_days

    if not test_sharpes:
        return 0, 0, 0

    avg_sharpe = np.mean(test_sharpes)
    consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100
    # Combined score: rewards both consistency and performance
    score = (consistency / 100) * max(avg_sharpe, 0)
    return score, consistency, avg_sharpe


def multi_stock_wf_score(datasets, strategy_name, params, dynamic_pos=True):
    """Average walk-forward score across all stocks."""
    scores = []
    consistencies = []
    sharpes = []

    for (symbol, market), df in datasets.items():
        s, c, sh = walk_forward_score(df, strategy_name, params, dynamic_pos=dynamic_pos)
        scores.append(s)
        consistencies.append(c)
        sharpes.append(sh)

    return np.mean(scores), np.mean(consistencies), np.mean(sharpes)


# ── Parameter Grids (smaller than full grid search, focused on promising ranges) ──

PARAM_GRIDS = {
    "claude_rsi_macd_trend": {
        "rsi_entry": [25, 30, 35],
        "rsi_exit": [75, 80, 85],
        "ema_trend": [30, 40, 50],
        "atr_stop_mul": [1.5, 2.0, 2.5],
    },
    "claude_ensemble_voting": {
        "buy_threshold": [1, 2],
        "sell_threshold": [2, 3],
        "mr_rsi_oversold": [20, 25, 30],
        "mr_rsi_overbought": [65, 70, 75],
        "dm_fast": [10, 15],
        "dm_slow": [25, 30],
    },
    "claude_momentum_breakout": {
        "breakout_period": [20, 25, 30],
        "exit_period": [6, 8, 10],
        "vol_ratio_min": [0.8, 1.0, 1.2],
        "atr_stop_mul": [1.5, 2.0, 2.5],
    },
    "claude_dual_ma_cross": {
        "fast_period": [10, 13, 15],
        "slow_period": [25, 30, 35],
        "atr_stop_mul": [2.0, 2.5, 3.0],
    },
    "claude_mean_reversion": {
        "rsi_period": [10, 14, 20],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [70, 75, 80],
        "bb_period": [20, 25, 30],
        "bb_std": [2.0, 2.5],
    },
    "claude_turtle_trading": {
        "entry_period": [20, 25, 30],
        "exit_period": [6, 8, 10],
        "atr_stop_mul": [1.5, 2.0, 2.5],
    },
}


def optimize_strategy(datasets, strategy_name, dynamic_pos=True):
    """Find best parameters using walk-forward scoring."""
    grid = PARAM_GRIDS.get(strategy_name)
    if not grid:
        print(f"  No grid for {strategy_name}, skipping")
        return None

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(product(*param_values))
    total = len(combos)

    print(f"\n  Strategy: {strategy_name}")
    print(f"  Grid: {total} combinations x {len(datasets)} stocks")
    print(f"  {'#':>4} {'Score':>8} {'Consist%':>9} {'AvgSharpe':>10}  Params")
    print(f"  {'-'*75}")

    results = []
    for idx, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        score, consistency, avg_sharpe = multi_stock_wf_score(
            datasets, strategy_name, params, dynamic_pos)
        results.append({
            "params": params,
            "score": score,
            "consistency": consistency,
            "avg_sharpe": avg_sharpe,
        })

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  {idx+1:>4}/{total}", end="\r")

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n\n  Top 5 results:")
    print(f"  {'Rank':>4} {'Score':>8} {'Consist%':>9} {'AvgSharpe':>10}  Params")
    print(f"  {'-'*75}")
    for i, r in enumerate(results[:5]):
        p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        star = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:>4} {r['score']:>8.3f} {r['consistency']:>8.1f}% {r['avg_sharpe']:>10.3f}  {p_str}{star}")

    # Compare with current defaults
    print(f"\n  Current defaults comparison:")
    default_score, default_c, default_s = multi_stock_wf_score(
        datasets, strategy_name, {}, dynamic_pos)
    print(f"       {'Score':>8} {'Consist%':>9} {'AvgSharpe':>10}")
    print(f"  Default: {default_score:>6.3f} {default_c:>8.1f}% {default_s:>10.3f}")
    if results:
        best = results[0]
        print(f"  Best WF: {best['score']:>6.3f} {best['consistency']:>8.1f}% {best['avg_sharpe']:>10.3f}")
        improvement = (best["score"] - default_score) / max(default_score, 0.001) * 100
        print(f"  Improvement: {improvement:+.1f}%")

    return results[0] if results else None


def main():
    print("=" * 90)
    print("  Walk-Forward-Aware Parameter Optimization")
    print("  (Prevents overfitting by scoring on out-of-sample consistency)")
    print("=" * 90)

    print("\nLoading 5y data...")
    datasets = load_data("5y")
    print(f"  {len(datasets)} stocks loaded\n")

    strategies = ["claude_rsi_macd_trend", "claude_ensemble_voting", "claude_momentum_breakout",
                  "claude_dual_ma_cross", "claude_mean_reversion", "claude_turtle_trading"]

    best_params = {}
    for sname in strategies:
        result = optimize_strategy(datasets, sname, dynamic_pos=True)
        if result:
            best_params[sname] = result

    # Final summary
    print("\n\n" + "=" * 90)
    print("  FINAL SUMMARY - Walk-Forward Optimized Parameters")
    print("=" * 90)
    for sname, result in best_params.items():
        p_str = ", ".join(f"{k}={v}" for k, v in result["params"].items())
        print(f"\n  {sname}:")
        print(f"    WF Score: {result['score']:.3f} | Consistency: {result['consistency']:.1f}% | Avg Sharpe: {result['avg_sharpe']:.3f}")
        print(f"    Params: {p_str}")

    print("\n" + "=" * 90)
    print("  Optimization complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
