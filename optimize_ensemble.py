"""
优化 ensemble_voting 投票阈值
"""
import sys
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest
import numpy as np
from itertools import product

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]

stock_data = {}
for symbol, market in TARGETS:
    try:
        df = get_ohlcv(symbol, market, period="2y")
        stock_data[symbol] = df
    except:
        pass

stock_data_5y = {}
for symbol, market in TARGETS:
    try:
        df = get_ohlcv(symbol, market, period="5y")
        stock_data_5y[symbol] = df
    except:
        pass

print(f"2y data: {len(stock_data)} stocks, 5y data: {len(stock_data_5y)} stocks\n")

for buy_th, sell_th in [(1,2), (1,3), (2,2), (2,3), (3,2), (3,3)]:
    params = {"buy_threshold": buy_th, "sell_threshold": sell_th}

    # 2y
    sharpes_2y = []
    for symbol, df in stock_data.items():
        try:
            r = run_strategy_backtest(df, "claude_ensemble_voting", params_override=params)
            sharpes_2y.append(r["summary"]["sharpe"])
        except:
            pass

    # 5y
    sharpes_5y = []
    for symbol, df in stock_data_5y.items():
        try:
            r = run_strategy_backtest(df, "claude_ensemble_voting", params_override=params)
            sharpes_5y.append(r["summary"]["sharpe"])
        except:
            pass

    avg_2y = np.mean(sharpes_2y) if sharpes_2y else -999
    avg_5y = np.mean(sharpes_5y) if sharpes_5y else -999
    print(f"  buy≥{buy_th} sell≥{sell_th} | 2y avg={avg_2y:.3f} | 5y avg={avg_5y:.3f}")

# 用最优参数查看每只股票的详细结果
print(f"\n\n=== 最优组合策略 详细结果 ===")
best_params = {"buy_threshold": 2, "sell_threshold": 2}
for period_label, data in [("2y", stock_data), ("5y", stock_data_5y)]:
    print(f"\n  --- {period_label} ---")
    results = []
    for symbol, df in data.items():
        try:
            r = run_strategy_backtest(df, "claude_ensemble_voting", params_override=best_params)
            s = r["summary"]
            results.append((symbol, s["sharpe"], s["total_return"], s["max_drawdown"], s["win_rate"], s["total_trades"]))
        except Exception as e:
            results.append((symbol, -999, 0, 0, 0, 0))
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Stock':<8} {'Sharpe':>7} {'Return%':>8} {'MaxDD%':>8} {'WinR%':>7} {'Trades':>6}")
    for sym, sh, ret, dd, wr, tr in results:
        print(f"  {sym:<8} {sh:>7.2f} {ret:>8.2f} {dd:>8.2f} {wr:>7.1f} {tr:>6}")
    avg_sh = np.mean([x[1] for x in results if x[1] > -999])
    print(f"  {'AVG':<8} {avg_sh:>7.2f}")
