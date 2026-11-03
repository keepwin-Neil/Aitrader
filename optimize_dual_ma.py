"""
dual_ma_cross 参数优化
网格搜索：fast_period x slow_period x ma_type
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

# 参数空间
FAST_PERIODS = [5, 8, 10, 13, 15, 20]
SLOW_PERIODS = [20, 30, 40, 50, 60]
MA_TYPES = ["ema", "sma"]
ATR_MULS = [1.5, 2.0, 2.5, 3.0]

combos = [(f, s, m, a) for f, s, m, a in product(FAST_PERIODS, SLOW_PERIODS, MA_TYPES, ATR_MULS) if s > f * 1.5]
print(f"参数组合数: {len(combos)}")

# 获取数据
stock_data = {}
for symbol, market in TARGETS:
    try:
        df = get_ohlcv(symbol, market, period="2y")
        stock_data[symbol] = df
        print(f"  {symbol}: {len(df)} bars")
    except Exception as e:
        print(f"  {symbol}: FAILED - {e}")

results = []
total = len(combos)
for idx, (fast, slow, ma_type, atr_mul) in enumerate(combos):
    if (idx + 1) % 20 == 0:
        print(f"  进度: {idx+1}/{total}")
    params = {"fast_period": fast, "slow_period": slow, "ma_type": ma_type, "atr_stop_mul": atr_mul}
    sharpes = []
    for symbol, df in stock_data.items():
        try:
            r = run_strategy_backtest(df, "claude_dual_ma_cross", params_override=params)
            sharpes.append(r["summary"]["sharpe"])
        except:
            pass
    if sharpes:
        results.append({
            "params": params,
            "avg_sharpe": round(np.mean(sharpes), 3),
            "median_sharpe": round(np.median(sharpes), 3),
            "std": round(np.std(sharpes), 3),
            "min": round(min(sharpes), 3),
            "max": round(max(sharpes), 3),
            "n": len(sharpes),
        })

results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

print(f"\n{'='*80}")
print(f"  dual_ma_cross 参数优化结果 TOP 20")
print(f"{'='*80}")
print(f"  {'排名':<4} {'fast':>4} {'slow':>4} {'type':<4} {'ATR':>4} {'avg':>7} {'med':>6} {'std':>6} {'min':>6} {'max':>6}")
print(f"  {'-'*60}")
for i, r in enumerate(results[:20]):
    p = r["params"]
    print(f"  {i+1:<4} {p['fast_period']:>4} {p['slow_period']:>4} {p['ma_type']:<4} {p['atr_stop_mul']:>4.1f} {r['avg_sharpe']:>7.3f} {r['median_sharpe']:>6.3f} {r['std']:>6.3f} {r['min']:>6.3f} {r['max']:>6.3f}")

print(f"\n最优参数: {results[0]['params']}")
print(f"平均夏普: {results[0]['avg_sharpe']}")
