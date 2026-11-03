"""
全策略 x 全周期 对比测试（10个策略）
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

PERIODS = ["2y", "5y"]

# 获取所有数据
print("=== 获取数据 ===")
all_data = {}
for period in PERIODS:
    all_data[period] = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, period=period)
            all_data[period][symbol] = df
            print(f"  {symbol} [{period}]: {len(df)} bars, {df.index[0].date()} ~ {df.index[-1].date()}")
        except Exception as e:
            print(f"  {symbol} [{period}]: FAILED - {e}")

strategies = list(STRATEGY_REGISTRY.keys())
print(f"\n策略列表 ({len(strategies)}): {strategies}")

for period in PERIODS:
    print(f"\n\n{'#'*70}")
    print(f"#  周期: {period}")
    print(f"{'#'*70}")

    all_results = {}
    for symbol in all_data[period]:
        df = all_data[period][symbol]
        results = []
        for sname in strategies:
            try:
                r = run_strategy_backtest(df, sname)
                s = r["summary"]
                results.append({
                    "strategy": sname,
                    "sharpe": round(s["sharpe"], 2),
                    "return": round(s["total_return"], 2),
                    "max_dd": round(s["max_drawdown"], 2),
                    "win_rate": round(s["win_rate"], 2),
                    "trades": s["total_trades"],
                    "pf": round(s["profit_factor"], 2),
                    "bh_return": round(s["bh_return"], 2),
                })
            except Exception as e:
                results.append({"strategy": sname, "error": str(e)})
        results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        all_results[symbol] = results

    # 每只股票的排名
    for symbol, results in all_results.items():
        print(f"\n  --- {symbol} [{period}] ---")
        print(f"  {'排名':<4} {'策略':<22} {'夏普':>6} {'收益%':>8} {'买持%':>7} {'最大回撤%':>9} {'胜率%':>6} {'交易':>4}")
        print(f"  {'-'*70}")
        for i, r in enumerate(results):
            if "error" in r:
                print(f"  {i+1:<4} {r['strategy']:<22} ERROR: {r['error'][:30]}")
            else:
                print(f"  {i+1:<4} {r['strategy']:<22} {r['sharpe']:>6.2f} {r['return']:>8.2f} {r.get('bh_return',0):>7.1f} {r['max_dd']:>9.2f} {r['win_rate']:>6.1f} {r['trades']:>4}")

    # 综合排名
    strategy_sharpes = {sname: [] for sname in strategies}
    for symbol, results in all_results.items():
        for r in results:
            if "sharpe" in r:
                strategy_sharpes[r["strategy"]].append(r["sharpe"])

    avg_ranking = []
    for sname, sharpes in strategy_sharpes.items():
        if sharpes:
            avg_ranking.append((sname, np.mean(sharpes), np.std(sharpes), np.median(sharpes), min(sharpes), max(sharpes), len(sharpes)))
    avg_ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  === {period} 综合排名 ===")
    print(f"  {'排名':<4} {'策略':<22} {'平均夏普':>8} {'中位数':>6} {'标准差':>6} {'最差':>6} {'最好':>6}")
    print(f"  {'-'*60}")
    for i, (sname, avg, std, med, mn, mx, n) in enumerate(avg_ranking):
        print(f"  {i+1:<4} {sname:<22} {avg:>8.2f} {med:>6.2f} {std:>6.2f} {mn:>6.2f} {mx:>6.2f}")
