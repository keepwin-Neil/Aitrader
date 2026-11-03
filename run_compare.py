"""
多策略 x 多周期 对比测试
对多个标的、多个历史周期运行全部策略，记录夏普比率排名
"""
import sys, json
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY
import numpy as np

TARGETS = [
    ("AAPL", "US"),
    ("NVDA", "US"),
    ("MSFT", "US"),
    ("TSLA", "US"),
    ("GOOGL", "US"),
    ("META", "US"),
    ("AMZN", "US"),
    ("AMD", "US"),
]

PERIODS = ["2y", "5y", "10y"]

def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

all_period_results = {}

for period in PERIODS:
    print(f"\n\n{'#'*70}")
    print(f"#  周期: {period}")
    print(f"{'#'*70}")

    all_results = {}
    for symbol, market in TARGETS:
        print(f"\n  --- {symbol} ({market}) [{period}] ---")
        try:
            df = get_ohlcv(symbol, market, interval="1d", period=period)
            print(f"  数据: {len(df)} 条K线, {df.index[0].date()} ~ {df.index[-1].date()}")
        except Exception as e:
            print(f"  获取数据失败: {e}")
            continue

        results = []
        for sname in STRATEGY_REGISTRY:
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
                print(f"    {sname}: 错误 - {e}")
                results.append({"strategy": sname, "error": str(e)})

        results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        all_results[symbol] = results

        print(f"  {'排名':<4} {'策略':<22} {'夏普':>6} {'收益%':>8} {'买持%':>7} {'最大回撤%':>9} {'胜率%':>6} {'交易':>4} {'盈亏比':>6}")
        print(f"  {'-'*78}")
        for i, r in enumerate(results):
            if "error" in r:
                print(f"  {i+1:<4} {r['strategy']:<22} ERROR")
            else:
                print(f"  {i+1:<4} {r['strategy']:<22} {r['sharpe']:>6.2f} {r['return']:>8.2f} {r.get('bh_return',0):>7.1f} {r['max_dd']:>9.2f} {r['win_rate']:>6.1f} {r['trades']:>4} {r['pf']:>6.2f}")

    # 汇总该周期
    strategy_sharpes = {sname: [] for sname in STRATEGY_REGISTRY}
    for symbol, results in all_results.items():
        for r in results:
            if "sharpe" in r:
                strategy_sharpes[r["strategy"]].append(r["sharpe"])

    avg_ranking = []
    for sname, sharpes in strategy_sharpes.items():
        if sharpes:
            avg_ranking.append((sname, np.mean(sharpes), np.std(sharpes), len(sharpes)))
    avg_ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  === {period} 综合排名 ===")
    print(f"  {'排名':<4} {'策略':<22} {'平均夏普':>8} {'标准差':>6} {'样本':>4}")
    print(f"  {'-'*50}")
    for i, (sname, avg, std, n) in enumerate(avg_ranking):
        print(f"  {i+1:<4} {sname:<22} {avg:>8.2f} {std:>6.2f} {n:>4}")

    all_period_results[period] = {
        "per_stock": all_results,
        "ranking": [{"rank": i+1, "strategy": s, "avg_sharpe": round(a, 2), "std": round(d, 2)}
                    for i, (s, a, d, _) in enumerate(avg_ranking)]
    }

# 跨周期汇总
print(f"\n\n{'='*70}")
print(f"  跨周期综合排名")
print(f"{'='*70}")

strategy_all = {sname: [] for sname in STRATEGY_REGISTRY}
for period, data in all_period_results.items():
    for symbol, results in data["per_stock"].items():
        for r in results:
            if "sharpe" in r:
                strategy_all[r["strategy"]].append(r["sharpe"])

final_ranking = []
for sname, sharpes in strategy_all.items():
    if sharpes:
        final_ranking.append((sname, np.mean(sharpes), np.std(sharpes),
                              np.median(sharpes), min(sharpes), max(sharpes), len(sharpes)))
final_ranking.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'排名':<4} {'策略':<22} {'平均夏普':>8} {'中位数':>6} {'标准差':>6} {'最差':>6} {'最好':>6} {'样本':>4}")
print(f"  {'-'*68}")
for i, (sname, avg, std, med, mn, mx, n) in enumerate(final_ranking):
    print(f"  {i+1:<4} {sname:<22} {avg:>8.2f} {med:>6.2f} {std:>6.2f} {mn:>6.2f} {mx:>6.2f} {n:>4}")

# 保存
output = clean({"periods": all_period_results, "final_ranking": [
    {"rank": i+1, "strategy": s, "avg_sharpe": round(a,2), "median": round(med,2),
     "std": round(d,2), "min": round(mn,2), "max": round(mx,2), "samples": n}
    for i, (s, a, d, med, mn, mx, n) in enumerate(final_ranking)
]})
with open("compare_results.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\n结果已保存到 compare_results.json")
