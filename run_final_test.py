"""
最终全策略对比（11个策略），2y + 5y
"""
import sys
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY
import numpy as np

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]

strategies = list(STRATEGY_REGISTRY.keys())
print(f"策略 ({len(strategies)}): {strategies}\n")

for period in ["2y", "5y"]:
    print(f"\n{'='*70}")
    print(f"  {period} 综合排名")
    print(f"{'='*70}")

    strategy_sharpes = {s: [] for s in strategies}
    strategy_returns = {s: [] for s in strategies}
    strategy_drawdowns = {s: [] for s in strategies}

    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, period=period)
        except:
            continue

        for sname in strategies:
            try:
                r = run_strategy_backtest(df, sname)
                s = r["summary"]
                strategy_sharpes[sname].append(s["sharpe"])
                strategy_returns[sname].append(s["total_return"])
                strategy_drawdowns[sname].append(s["max_drawdown"])
            except:
                pass

    ranking = []
    for sname in strategies:
        sh = strategy_sharpes[sname]
        ret = strategy_returns[sname]
        dd = strategy_drawdowns[sname]
        if sh:
            ranking.append((sname, np.mean(sh), np.median(sh), np.std(sh),
                           min(sh), max(sh), np.mean(ret), np.mean(dd), len(sh)))
    ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Rank':<4} {'Strategy':<22} {'AvgSharpe':>9} {'Median':>7} {'Std':>6} {'Min':>6} {'Max':>6} {'AvgRet%':>8} {'AvgDD%':>7}")
    print(f"  {'-'*78}")
    for i, (s, avg, med, std, mn, mx, ret, dd, n) in enumerate(ranking):
        print(f"  {i+1:<4} {s:<22} {avg:>9.3f} {med:>7.3f} {std:>6.3f} {mn:>6.2f} {mx:>6.2f} {ret:>8.2f} {dd:>7.2f}")
