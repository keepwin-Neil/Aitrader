#!/usr/bin/env python3
"""
Multi-process evaluation worker for strategy ranking.
Called by main.py's /strategies/evaluate endpoint.
Reads tasks from _eval_tasks.pkl (including pre-downloaded data),
writes results to _eval_results.pkl.
"""
import pickle, json, os, sys, time, tempfile
import proc_role          # 池 worker 打标 + 父进程死了跟着退(防孤儿)
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE

_cpu = os.cpu_count() or 4
N_WORKERS = max(2, _cpu - 2)


def _bt_one(args):
    sym, sname, df_bytes = args
    try:
        df = pickle.loads(df_bytes)
        # Load optimized params
        params = {}
        opt_path = Path(__file__).parent / "optimized_params" / f"{sym}.json"
        if opt_path.exists():
            with open(opt_path) as f:
                all_p = json.load(f)
            if sname in all_p and "params" in all_p[sname]:
                params = all_p[sname]["params"]

        result = run_strategy_backtest(df.copy(), sname, params)
        if result and result.get("summary"):
            s = result["summary"]
            ret = s.get("total_return", 0)
            sharpe = s.get("sharpe_ratio", 0)
            dd = s.get("max_drawdown", 0)
            wr = s.get("win_rate", 0)
            pf = s.get("profit_factor", 0)
            trades = s.get("total_trades", 0)
            wins = s.get("win_trades", 0)
            score = (0.40 * (ret / 100)
                     + 0.25 * max(sharpe, 0) / 3
                     + 0.20 * (1 - abs(dd) / 100)
                     + 0.10 * (wr / 100)
                     + 0.05 * min(pf, 5) / 5)
            return (sym, sname, score, ret, sharpe, dd, wr, pf, trades, wins)
    except Exception:
        pass
    return (sym, sname, 0, 0, 0, 0, 0, 0, 0, 0)


# ── 评测进度文件: 路径和读写都只定义在这里 ────────────────────────────────────
# 之前路径写了 3 份 (这里 / routers/strategies.py / eval_chain.py),
# "存在就读、坏了给默认" 的读法抄了 4 份, 连默认值都不一致 ({} vs {"stage":"idle"})。
# 改一处漏三处的典型。这个模块所有执行代码都在 __main__ guard 里, import 安全。
PROGRESS_FILE = Path(__file__).parent / "_eval_progress.json"
IDLE = {"stage": "idle"}


def read_progress(default=None):
    """读进度文件。文件不存在/内容坏了都返回 default。

    不做 .exists() 预检: read_text 的 FileNotFoundError 本来就落进同一个 except,
    多一次预检只是把 TOCTOU 窗口拉大 (eval_chain 会在跑之前 unlink 这个文件)。
    """
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8")) or (default or {})
    except Exception:
        return default if default is not None else {}


def write_progress(**fields):
    """原子写进度文件 (供 /strategies/evaluate/progress 读)。失败不影响评测。

    fields 里带 run_id 时会覆盖原有的 —— 新一轮评测必须先写 run_id,
    否则上一轮的残留会被前端当成本轮进度画出来。
    """
    try:
        prev = read_progress()
        prev.update(fields)
        prev["updated_at"] = time.time()
        fd, tmp = tempfile.mkstemp(dir=str(PROGRESS_FILE.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(prev, f, ensure_ascii=False)
        os.replace(tmp, PROGRESS_FILE)
    except Exception:
        pass


if __name__ == "__main__":
    task_file = Path(__file__).parent / "_eval_tasks.pkl"
    result_file = Path(__file__).parent / "_eval_results.pkl"

    with open(task_file, "rb") as f:
        task_data = pickle.load(f)

    strategies = task_data["strategies"]
    datasets = task_data["datasets"]  # {sym: df_bytes} pre-downloaded & pickled

    print(f"eval_worker: CPU={_cpu}, workers={N_WORKERS}, {len(strategies)} strategies × {len(datasets)} stocks = {len(strategies)*len(datasets)} tasks", file=sys.stderr)
    t0 = time.time()

    # Build task list with pre-serialized data
    tasks = []
    for sname in strategies:
        for sym, df_bytes in datasets.items():
            tasks.append((sym, sname, df_bytes))

    # Run with ProcessPoolExecutor
    results = {}
    total = len(tasks)
    done = 0
    write_progress(stage="backtest", done=0, total=total,
                   strategies=len(strategies), symbols=len(datasets), workers=N_WORKERS)
    _last_flush = 0.0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=proc_role.mark_pool_worker) as pool:
        futures = {pool.submit(_bt_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            sym, sname = r[0], r[1]
            if sname not in results:
                results[sname] = []
            results[sname].append(r[2:])
            done += 1
            # 最多每 0.4s 落一次盘, 免得几千个任务把磁盘写爆
            now = time.time()
            if done == total or (now - _last_flush) >= 0.4:
                _last_flush = now
                write_progress(stage="backtest", done=done, total=total, current=sname)

    with open(result_file, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.time() - t0
    print(f"eval_worker: done in {elapsed:.1f}s, {len(results)} strategies evaluated", file=sys.stderr)
