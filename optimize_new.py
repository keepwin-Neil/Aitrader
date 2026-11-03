"""
优化 vwap_breakout 和 rsi_macd_trend 参数
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

# 获取数据
stock_data = {}
for symbol, market in TARGETS:
    try:
        df = get_ohlcv(symbol, market, period="2y")
        stock_data[symbol] = df
    except:
        pass
print(f"数据: {len(stock_data)} stocks")

def optimize(strategy_name, param_grid):
    combos = list(product(*param_grid.values()))
    keys = list(param_grid.keys())
    print(f"\n=== 优化 {strategy_name} ({len(combos)} 组合) ===")

    results = []
    for idx, vals in enumerate(combos):
        if (idx + 1) % 30 == 0:
            print(f"  进度: {idx+1}/{len(combos)}")
        params = dict(zip(keys, vals))
        sharpes = []
        for symbol, df in stock_data.items():
            try:
                r = run_strategy_backtest(df, strategy_name, params_override=params)
                sharpes.append(r["summary"]["sharpe"])
            except:
                pass
        if len(sharpes) >= 6:
            results.append({
                "params": params,
                "avg_sharpe": round(np.mean(sharpes), 3),
                "median": round(np.median(sharpes), 3),
                "std": round(np.std(sharpes), 3),
                "min": round(min(sharpes), 3),
                "n": len(sharpes),
            })

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    print(f"\n  TOP 10:")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. avg={r['avg_sharpe']:.3f} med={r['median']:.3f} std={r['std']:.3f} min={r['min']:.3f} | {r['params']}")
    return results

# 1. VWAP Breakout
optimize("claude_vwap_breakout", {
    "vwap_period": [10, 15, 20, 30],
    "atr_band_mul": [1.0, 1.5, 2.0, 2.5],
    "vol_ratio_min": [0.8, 1.0, 1.2],
    "atr_stop_mul": [1.5, 2.0, 2.5],
})

# 2. RSI + MACD Trend
optimize("claude_rsi_macd_trend", {
    "rsi_entry": [30, 35, 40, 45],
    "rsi_exit": [65, 70, 75, 80],
    "ema_trend": [30, 50, 60],
    "atr_stop_mul": [1.5, 2.0, 2.5],
})
