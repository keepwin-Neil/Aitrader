"""
测试不同K线周期（30min, 1h, 1d）对策略收益的影响
yfinance 限制: intraday 数据最多 730 天, 30m 最多 60天
"""
import sys
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest
import numpy as np
import json

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]

# yfinance 限制：
# 1m: 最多 7 天
# 5m/15m/30m: 最多 60 天
# 60m/1h: 最多 730 天
# 1d: 无限制

STRATEGIES = ["claude_rsi_macd_trend", "claude_ensemble_voting", "claude_momentum_breakout", "claude_dual_ma_cross", "claude_turtle_trading", "claude_mean_reversion"]

INTERVALS = [
    ("30m", "60d"),   # 30分钟线, 最多60天数据
    ("1h",  "2y"),    # 1小时线, 最多730天
    ("1d",  "2y"),    # 日线, 2年
]

all_results = {}

for interval, period in INTERVALS:
    print(f"\n{'='*70}")
    print(f"  测试 {interval} K线 (数据范围: {period})")
    print(f"{'='*70}")

    # 加载数据
    datasets = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, interval=interval, period=period)
            datasets[(symbol, market)] = df
            print(f"  {symbol}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  {symbol}: FAIL - {e}")

    if len(datasets) < 4:
        print(f"  数据不足，跳过")
        continue

    interval_results = {}
    for strategy_name in STRATEGIES:
        sharpes = []
        returns = []
        trades_count = []
        details = {}
        for (symbol, market), df in datasets.items():
            try:
                r = run_strategy_backtest(df, strategy_name)
                s = r["summary"]
                sharpes.append(s["sharpe"])
                returns.append(s["total_return"])
                trades_count.append(s["total_trades"])
                details[symbol] = {
                    "sharpe": s["sharpe"],
                    "return": s["total_return"],
                    "trades": s["total_trades"],
                    "max_dd": s["max_drawdown"],
                    "win_rate": s["win_rate"],
                }
            except Exception as e:
                print(f"    {strategy_name}/{symbol}: ERROR - {e}")
                sharpes.append(0)
                returns.append(0)
                trades_count.append(0)
                details[symbol] = {"sharpe": 0, "return": 0, "trades": 0}

        avg_sharpe = float(np.mean(sharpes))
        avg_return = float(np.mean(returns))
        avg_trades = float(np.mean(trades_count))

        interval_results[strategy_name] = {
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_return": round(avg_return, 2),
            "avg_trades": round(avg_trades, 1),
            "per_stock": details,
        }

    # 打印排行
    print(f"\n  ── {interval} 策略排行 ──")
    print(f"  {'排名':<4} {'策略':<22} {'AvgSharpe':>10} {'AvgReturn%':>11} {'AvgTrades':>10}")
    print(f"  {'-'*65}")
    ranking = sorted(interval_results.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True)
    for i, (name, data) in enumerate(ranking, 1):
        print(f"  {i:<4} {name:<22} {data['avg_sharpe']:>10.3f} {data['avg_return']:>11.2f} {data['avg_trades']:>10.1f}")

    # 打印各股票详情
    for strategy_name, data in ranking[:3]:
        print(f"\n    {strategy_name} 详情:")
        for sym, d in data["per_stock"].items():
            print(f"      {sym:<6} Sharpe={d['sharpe']:>6.2f}  Return={d['return']:>7.2f}%  Trades={d['trades']:>3}  MaxDD={d.get('max_dd',0):>6.2f}%  WinRate={d.get('win_rate',0):>5.1f}%")

    all_results[interval] = interval_results

# 跨周期对比
print(f"\n\n{'='*70}")
print(f"  跨 K线周期 对比 (同一策略不同interval)")
print(f"{'='*70}")
for strategy_name in STRATEGIES:
    print(f"\n  {strategy_name}:")
    for interval, period in INTERVALS:
        if interval in all_results and strategy_name in all_results[interval]:
            d = all_results[interval][strategy_name]
            print(f"    {interval:>4} ({period}):  Sharpe={d['avg_sharpe']:>6.3f}  Return={d['avg_return']:>8.2f}%  Trades={d['avg_trades']:>6.1f}")

# 保存
with open("interval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到 interval_results.json")
