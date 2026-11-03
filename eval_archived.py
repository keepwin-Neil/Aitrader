#!/usr/bin/env python3
"""Evaluate all archived strategies against MYPOOL"""
import json, time, numpy as np, multiprocessing
import proc_role          # 池 worker 打标 + 父进程死了跟着退(防孤儿)
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY

def _bt_one(args):
    sym, sname, df = args
    try:
        result = run_strategy_backtest(df.copy(), sname, {})
        if result and result.get("summary"):
            s = result["summary"]
            ret = s.get("total_return", 0)
            sharpe = s.get("sharpe_ratio", 0)
            dd = s.get("max_drawdown", 1)
            wr = s.get("win_rate", 0)
            pf = s.get("profit_factor", 0)
            score = 0.40*ret + 0.25*max(sharpe,0)/3 + 0.20*(1-dd) + 0.10*wr + 0.05*min(pf,5)/5
            return (sym, sname, score)
    except:
        pass
    return (sym, sname, 0)

if __name__ == "__main__":
    with open("stock_pools.json") as f:
        pools = json.load(f)
    mypool = pools["MYPOOL"]["symbols"]
    market = pools["MYPOOL"].get("market", "US")
    print(f"MYPOOL: {len(mypool)} stocks, market={market}")

    with open("strategy_warehouse.json") as f:
        wh = json.load(f)
    archived = [s["name"] for s in wh.get("archived", []) if s["name"] in STRATEGY_REGISTRY]
    print(f"Archived strategies (in registry): {len(archived)}")

    print("Downloading data...")
    datasets = {}
    for i, sym in enumerate(mypool):
        try:
            df = get_ohlcv(sym, market=market, period="2y", interval="1d")
            if df is not None and len(df) > 60:
                datasets[sym] = df
        except:
            pass
        if (i+1) % 20 == 0:
            print(f"  Downloaded {i+1}/{len(mypool)}...")
    print(f"Got data for {len(datasets)}/{len(mypool)} stocks")

    tasks = []
    for sname in archived:
        for sym, df in datasets.items():
            tasks.append((sym, sname, df))
    print(f"Total tasks: {len(tasks)} ({len(archived)} strategies × {len(datasets)} stocks)")

    n_workers = min(multiprocessing.cpu_count() - 2, 10)
    print(f"Running backtests with {n_workers} workers...")
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        futures = {pool.submit(_bt_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            sym, sname, score = fut.result()
            if sname not in results:
                results[sname] = []
            results[sname].append(score)
            done += 1
            if done % 500 == 0:
                pct = done / len(tasks) * 100
                elapsed = time.time() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"  Progress: {done}/{len(tasks)} ({pct:.0f}%) ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print(f"{'#':>3} {'策略':<30} {'均分':>8} {'Trimmed':>8} {'正收益%':>8} {'股票数':>6}")
    print(f"{'='*70}")

    ranking = []
    for sname in archived:
        scores = results.get(sname, [])
        if not scores:
            continue
        avg = np.mean(scores)
        s = sorted(scores)
        trim = max(1, len(s)//10)
        trimmed = np.mean(s[trim:-trim]) if len(s) > 2*trim else avg
        positive = sum(1 for x in scores if x > 0.2) / len(scores) * 100
        ranking.append((sname, avg, trimmed, positive, len(scores)))

    ranking.sort(key=lambda x: x[2], reverse=True)
    for i, (sname, avg, trimmed, pos, n) in enumerate(ranking):
        marker = "⭐" if trimmed > 0.35 else "  "
        print(f"{marker} {i+1:>2}. {sname:<28} {avg:>8.4f} {trimmed:>8.4f} {pos:>7.1f}% {n:>5}")

    print(f"\n⭐ = Trimmed Mean > 0.35 (potential reactivation candidates)")
