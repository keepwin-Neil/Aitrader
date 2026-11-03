"""
对 mean_reversion 策略做参数网格搜索，以平均夏普比率为优化目标
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

# 先加载所有数据
print("加载数据...")
datasets = {}
for symbol, market in TARGETS:
    try:
        df = get_ohlcv(symbol, market, interval="1d", period="2y")
        datasets[(symbol, market)] = df
        print(f"  {symbol}: {len(df)} 条")
    except Exception as e:
        print(f"  {symbol}: 失败 - {e}")

# 参数网格
param_grid = {
    "rsi_period":     [10, 14, 20],
    "rsi_oversold":   [25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "bb_period":      [15, 20, 25],
    "bb_std":         [1.5, 2.0, 2.5],
}

keys = list(param_grid.keys())
values = list(param_grid.values())
combos = list(product(*values))
print(f"\n网格搜索: {len(combos)} 种参数组合 x {len(datasets)} 只股票")

results = []
for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    sharpes = []
    for (symbol, market), df in datasets.items():
        try:
            r = run_strategy_backtest(df, "claude_mean_reversion", params_override=params)
            sharpes.append(r["summary"]["sharpe"])
        except:
            sharpes.append(0)
    avg_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    results.append((avg_sharpe, std_sharpe, params, sharpes))
    if (idx + 1) % 50 == 0:
        print(f"  进度: {idx+1}/{len(combos)}")

# 按平均夏普排序
results.sort(key=lambda x: x[0], reverse=True)

print(f"\n{'='*70}")
print(f"  TOP 10 参数组合（按平均夏普比率）")
print(f"{'='*70}")
print(f"\n  {'排名':<4} {'平均夏普':>8} {'标准差':>6} {'参数'}")
print(f"  {'-'*65}")
for i, (avg, std, params, sharpes) in enumerate(results[:10]):
    p_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  {i+1:<4} {avg:>8.3f} {std:>6.2f}  {p_str}")

# 最优参数的详细表现
print(f"\n\n最优参数在各标的上的表现:")
best_avg, best_std, best_params, best_sharpes = results[0]
print(f"  参数: {best_params}")
for (symbol, market), sharpe in zip(datasets.keys(), best_sharpes):
    print(f"  {symbol}: sharpe = {sharpe:.2f}")

# 也看看最差的
print(f"\n\n最差 5 个:")
for i, (avg, std, params, _) in enumerate(results[-5:]):
    p_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  {len(results)-4+i:<4} {avg:>8.3f} {std:>6.2f}  {p_str}")
