"""
用优化后的默认参数验证所有策略表现
"""
import sys
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest
import numpy as np

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]

STRATEGIES = [
    "claude_mean_reversion", "claude_ensemble_voting", "claude_dual_ma_cross",
    "claude_rsi_macd_trend", "claude_vwap_breakout", "claude_momentum_breakout",
    "claude_turtle_trading", "claude_adaptive_ma",
    "claude_multi_indicator", "claude_macd_divergence", "claude_kdj_dmi",  # bottom 3 for reference
]

for period in ["2y", "5y"]:
    print(f"\n{'='*80}")
    print(f"  验证优化后参数 — {period} 周期")
    print(f"{'='*80}")

    datasets = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, interval="1d", period=period)
            datasets[(symbol, market)] = df
        except Exception as e:
            print(f"  {symbol}: FAIL - {e}")

    results = []
    for strategy_name in STRATEGIES:
        sharpes = []
        returns = []
        trades_list = []
        win_rates = []
        drawdowns = []
        per_stock = {}

        for (symbol, market), df in datasets.items():
            try:
                r = run_strategy_backtest(df, strategy_name)
                s = r["summary"]
                sharpes.append(s["sharpe"])
                returns.append(s["total_return"])
                trades_list.append(s["total_trades"])
                win_rates.append(s["win_rate"])
                drawdowns.append(s["max_drawdown"])
                per_stock[symbol] = f"S={s['sharpe']:.2f} R={s['total_return']:.1f}%"
            except Exception as e:
                sharpes.append(0)
                returns.append(0)
                per_stock[symbol] = "ERROR"

        results.append({
            "name": strategy_name,
            "avg_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "min_sharpe": np.min(sharpes),
            "avg_return": np.mean(returns),
            "avg_trades": np.mean(trades_list) if trades_list else 0,
            "avg_winrate": np.mean(win_rates) if win_rates else 0,
            "avg_drawdown": np.mean(drawdowns) if drawdowns else 0,
            "per_stock": per_stock,
        })

    # 排序
    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    print(f"\n  {'Rank':<5} {'Strategy':<22} {'AvgSharpe':>9} {'Std':>6} {'MinS':>6} {'AvgRet%':>9} {'Trades':>7} {'WinRate':>8} {'MaxDD%':>7}")
    print(f"  {'-'*85}")
    for i, r in enumerate(results, 1):
        marker = " ★" if i <= 3 else ""
        print(f"  {i:<5} {r['name']:<22} {r['avg_sharpe']:>9.3f} {r['std_sharpe']:>6.3f} {r['min_sharpe']:>6.2f} "
              f"{r['avg_return']:>9.2f} {r['avg_trades']:>7.1f} {r['avg_winrate']:>7.1f}% {r['avg_drawdown']:>7.1f}{marker}")

    # Top 3 详情
    print(f"\n  Top 3 各股详情:")
    for r in results[:3]:
        print(f"\n    {r['name']}:")
        for sym, detail in r["per_stock"].items():
            print(f"      {sym:<6} {detail}")
