#!/usr/bin/env python3
"""Evaluate all active strategies on MYPOOL + NASDAQ100 dual pool"""
import json, time, numpy as np, os, pickle, multiprocessing
import proc_role          # 池 worker 打标 + 父进程死了跟着退(防孤儿)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY

N_WORKERS = max(multiprocessing.cpu_count() - 2, 4)


def _bt_one(args):
    sym, sname, df_bytes = args
    df = pickle.loads(df_bytes)
    try:
        params = {}
        path = f"optimized_params/{sym}.json"
        if os.path.exists(path):
            with open(path) as f:
                all_p = json.load(f)
            if sname in all_p and "params" in all_p[sname]:
                params = all_p[sname]["params"]

        result = run_strategy_backtest(df.copy(), sname, params)
        if result and result.get("summary"):
            s = result["summary"]
            ret = s.get("total_return", 0) or 0          # percentage, e.g. 25.5 = 25.5%
            sharpe = s.get("sharpe", s.get("sharpe_ratio", 0)) or 0  # raw number
            dd = abs(s.get("max_drawdown", 0) or 0)       # percentage, make positive
            wr = s.get("win_rate", 0) or 0                 # percentage, e.g. 75.0
            pf = s.get("profit_factor", 0) or 0
            trades = s.get("total_trades", 0) or 0
            # Normalize to 0~1 scale for scoring
            ret_n = min(max(ret / 100, -1), 3)             # -1 ~ 3
            sharpe_n = max(sharpe, 0) / 3                  # 0 ~ 1
            dd_n = 1 - min(dd / 100, 1)                    # 0 ~ 1 (lower dd = higher score)
            wr_n = wr / 100                                # 0 ~ 1
            pf_n = min(pf, 5) / 5                          # 0 ~ 1
            score = 0.40 * ret_n + 0.25 * sharpe_n + 0.20 * dd_n + 0.10 * wr_n + 0.05 * pf_n
            return (sym, sname, score, ret, sharpe, dd, wr, pf, trades)
    except Exception:
        pass
    return (sym, sname, 0, 0, 0, 1, 0, 0, 0)


def trimmed_mean(arr):
    s = sorted(arr)
    t = max(1, len(s) // 10)
    return float(np.mean(s[t:-t])) if len(s) > 2 * t else float(np.mean(s))


if __name__ == "__main__":
    # Load pools
    with open("stock_pools.json") as f:
        pools = json.load(f)
    mypool = pools["MYPOOL"]["symbols"]
    nasdaq = pools["NASDAQ 100"]["symbols"]
    all_syms = sorted(set(mypool + nasdaq))
    print(f"MYPOOL: {len(mypool)}, NASDAQ100: {len(nasdaq)}, Total unique: {len(all_syms)}")

    # Active strategies
    with open("strategy_warehouse.json", encoding="utf-8") as f:
        wh = json.load(f)
    active = wh.get("active", [])
    print(f"Active strategies: {len(active)}")

    # Download data (并行)
    print("Downloading data (parallel)...")

    def _fetch_sym(sym):
        try:
            df = get_ohlcv(sym, period="2y", interval="1d", market="US")
            if df is not None and len(df) > 60:
                return sym, df
        except Exception:
            pass
        return sym, None

    datasets = {}
    n_dl = min(10, len(all_syms))
    with ThreadPoolExecutor(max_workers=n_dl) as dl_pool:
        dl_futures = {dl_pool.submit(_fetch_sym, sym): sym for sym in all_syms}
        for fut in as_completed(dl_futures):
            sym, df = fut.result()
            if df is not None:
                datasets[sym] = df
    print(f"Got data for {len(datasets)}/{len(all_syms)} stocks")

    mypool_syms = [s for s in mypool if s in datasets]
    nasdaq_syms = [s for s in nasdaq if s in datasets]
    all_eval_syms = sorted(set(mypool_syms + nasdaq_syms))

    # Build tasks
    tasks = []
    for sname in active:
        for sym in all_eval_syms:
            df_bytes = pickle.dumps(datasets[sym])
            tasks.append((sym, sname, df_bytes))
    print(f"Total tasks: {len(tasks)} ({len(active)} strategies × {len(all_eval_syms)} stocks)")

    # Run parallel
    print("Running backtests...")
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=proc_role.mark_pool_worker) as pool:
        futures = {pool.submit(_bt_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            sym, sname = r[0], r[1]
            results[(sname, sym)] = r[2:]
            done += 1
            if done % 500 == 0:
                print(f"  Progress: {done}/{len(tasks)}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Compute per-strategy stats
    output = []
    for sname in active:
        my_data = [results.get((sname, s), (0, 0, 0, 1, 0, 0, 0)) for s in mypool_syms]
        nq_data = [results.get((sname, s), (0, 0, 0, 1, 0, 0, 0)) for s in nasdaq_syms]
        all_data = [results.get((sname, s), (0, 0, 0, 1, 0, 0, 0)) for s in all_eval_syms]

        my_scores = [d[0] for d in my_data]
        nq_scores = [d[0] for d in nq_data]
        all_rets = [d[1] for d in all_data]
        all_sharpes = [d[2] for d in all_data]
        all_dds = [d[3] for d in all_data]
        all_wrs = [d[4] for d in all_data]

        my_tm = trimmed_mean(my_scores)
        nq_tm = trimmed_mean(nq_scores)
        combined = 0.5 * my_tm + 0.5 * nq_tm

        avg_ret = float(np.mean(all_rets))       # already in %
        avg_sharpe = float(np.mean(all_sharpes))
        avg_dd = float(np.mean(all_dds))          # already in %
        avg_wr = float(np.mean(all_wrs))           # already in %
        pos_pct = sum(1 for x in all_rets if x > 0) / len(all_rets) * 100

        output.append({
            "name": sname,
            "mypool": my_tm,
            "nasdaq": nq_tm,
            "combined": combined,
            "avg_ret": avg_ret,
            "avg_sharpe": avg_sharpe,
            "avg_dd": avg_dd,
            "avg_wr": avg_wr,
            "pos_pct": pos_pct,
        })

    output.sort(key=lambda x: x["combined"], reverse=True)

    # Save
    with open("auto_optimize_results/eval_20active.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print table
    print(f"\n{'=' * 120}")
    print(f"{'#':>2} {'策略':<30} {'MYPOOL':>8} {'NASDAQ':>8} {'综合':>8} {'平均收益%':>10} {'Sharpe':>8} {'回撤%':>8} {'胜率%':>7} {'正收益%':>8}")
    print(f"{'=' * 120}")
    for i, s in enumerate(output):
        print(f"{i+1:>2} {s['name']:<30} {s['mypool']:>8.4f} {s['nasdaq']:>8.4f} {s['combined']:>8.4f} {s['avg_ret']:>9.1f}% {s['avg_sharpe']:>8.3f} {s['avg_dd']:>7.1f}% {s['avg_wr']:>6.1f}% {s['pos_pct']:>7.1f}%")
    print(f"{'=' * 120}")
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
