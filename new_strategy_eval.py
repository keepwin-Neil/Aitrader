#!/usr/bin/env python3
"""新策略评估脚本 — 步骤2: NASDAQ基准测试"""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import get_ohlcv, run_strategy_backtest
from strategies import get_strategy
import numpy as np

def test_pool(strategy_name, pool_name, pool_data, capital=100000, period="1y"):
    """在指定股票池中测试策略，返回平均收益率"""
    symbols = pool_data["symbols"]
    market = pool_data["market"]
    results = []
    errors = []

    for sym in symbols:
        try:
            df = get_ohlcv(sym, market, interval="1d", period=period)
            if df is None or len(df) < 50:
                errors.append(f"{sym}: 数据不足")
                continue
            res = run_strategy_backtest(df, strategy_name, capital=capital)
            ret = res["summary"]["total_return"]
            results.append({"symbol": sym, "return": ret})
            print(f"  {sym}: {ret:.2f}%")
        except Exception as e:
            errors.append(f"{sym}: {e}")
            print(f"  {sym}: ERROR - {e}")

    if not results:
        return {"avg_return": 0, "results": results, "errors": errors}

    returns = [r["return"] for r in results]
    avg = np.mean(returns)
    best = max(results, key=lambda x: x["return"])
    worst = min(results, key=lambda x: x["return"])

    return {
        "strategy": strategy_name,
        "pool": pool_name,
        "count": len(results),
        "avg_return": round(avg, 2),
        "median_return": round(np.median(returns), 2),
        "best": best,
        "worst": worst,
        "errors": errors,
    }


if __name__ == "__main__":
    with open("stock_pools.json") as f:
        pools = json.load(f)

    new_strategies = ["claude_hurst_exponent", "claude_entropy_momentum", "claude_kaufman_efficiency"]

    pool_name = sys.argv[1] if len(sys.argv) > 1 else "NASDAQ 100"
    pool_data = pools[pool_name]

    print(f"\n{'='*60}")
    print(f"基准测试: {pool_name} ({len(pool_data['symbols'])}只股票)")
    print(f"{'='*60}")

    all_results = {}
    for sn in new_strategies:
        print(f"\n--- 策略: {sn} ---")
        try:
            # verify strategy exists
            get_strategy(sn)
            result = test_pool(sn, pool_name, pool_data)
            all_results[sn] = result
            print(f"\n✅ {sn}: 平均收益 {result['avg_return']:.2f}%, "
                  f"最高 {result['best']['symbol']} ({result['best']['return']:.2f}%), "
                  f"最低 {result['worst']['symbol']} ({result['worst']['return']:.2f}%)")
        except Exception as e:
            print(f"\n❌ {sn}: 策略加载失败 - {e}")
            all_results[sn] = {"error": str(e)}

    # Save results
    out_path = f"/tmp/eval_{pool_name.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")
