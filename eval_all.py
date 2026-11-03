#!/usr/bin/env python3
"""Evaluate ALL strategies (active + archived) against MYPOOL, output HTML table"""
import json, time, numpy as np, multiprocessing, csv
import proc_role          # 池 worker 打标 + 父进程死了跟着退(防孤儿)
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY

def composite_score(total_return, sharpe, max_drawdown, win_rate, profit_factor):
    """与 auto_optimize.py 完全一致的归一化评分，结果在 [0, 1]"""
    norm_ret = np.clip((total_return + 50) / 250, 0, 1)
    norm_sharpe = np.clip((sharpe + 1) / 4, 0, 1)
    norm_dd = np.clip(1 - max_drawdown / 60, 0, 1)
    norm_wr = np.clip(win_rate / 100, 0, 1)
    norm_pf = np.clip(profit_factor / 5, 0, 1)
    score = (0.40 * norm_ret + 0.25 * norm_sharpe +
             0.20 * norm_dd + 0.10 * norm_wr + 0.05 * norm_pf)
    return round(float(score), 4)

def _bt_one(args):
    sym, sname, df = args
    try:
        result = run_strategy_backtest(df.copy(), sname, {})
        if result and result.get("summary"):
            s = result["summary"]
            score = composite_score(
                s.get("total_return", 0), s.get("sharpe_ratio", 0),
                s.get("max_drawdown", 100), s.get("win_rate", 0),
                s.get("profit_factor", 0)
            )
            ret = s.get("total_return", 0)
            return (sym, sname, score, ret)
    except:
        pass
    return (sym, sname, 0, 0)

if __name__ == "__main__":
    with open("stock_pools.json") as f:
        pools = json.load(f)
    mypool = pools["MYPOOL"]["symbols"]
    market = pools["MYPOOL"].get("market", "US")

    with open("strategy_warehouse.json") as f:
        wh = json.load(f)
    # active can be list of strings or list of dicts
    raw_active = wh.get("active", [])
    if raw_active and isinstance(raw_active[0], dict):
        active_names = set(s["name"] for s in raw_active)
    else:
        active_names = set(raw_active)
    raw_archived = wh.get("archived", [])
    if raw_archived and isinstance(raw_archived[0], dict):
        archived_names = set(s["name"] for s in raw_archived if s["name"] in STRATEGY_REGISTRY)
    else:
        archived_names = set(s for s in raw_archived if s in STRATEGY_REGISTRY)
    all_strategies = list(active_names | archived_names)
    print(f"MYPOOL: {len(mypool)} stocks | Active: {len(active_names)} | Archived: {len(archived_names)} | Total: {len(all_strategies)}")

    # Download data (并行)
    print("Downloading data (parallel)...")

    def _fetch_sym(sym):
        try:
            df = get_ohlcv(sym, market=market, period="2y", interval="1d")
            if df is not None and len(df) > 60:
                return sym, df
        except:
            pass
        return sym, None

    datasets = {}
    n_dl = min(10, len(mypool))
    with ThreadPoolExecutor(max_workers=n_dl) as dl_pool:
        dl_futures = {dl_pool.submit(_fetch_sym, sym): sym for sym in mypool}
        for fut in as_completed(dl_futures):
            sym, df = fut.result()
            if df is not None:
                datasets[sym] = df
    print(f"Got data for {len(datasets)}/{len(mypool)} stocks")

    # Build tasks
    tasks = []
    for sname in all_strategies:
        for sym, df in datasets.items():
            tasks.append((sym, sname, df))
    print(f"Total tasks: {len(tasks)} ({len(all_strategies)} strategies × {len(datasets)} stocks)")

    # Run
    n_workers = max(multiprocessing.cpu_count() - 2, 4)
    print(f"Running backtests with {n_workers} workers...")
    t0 = time.time()
    results = {}
    returns = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        futures = {pool.submit(_bt_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            sym, sname, score, ret = fut.result()
            if sname not in results:
                results[sname] = []
                returns[sname] = []
            results[sname].append(score)
            returns[sname].append(ret)
            done += 1
            if done % 500 == 0:
                pct = done / len(tasks) * 100
                elapsed = time.time() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"  Progress: {done}/{len(tasks)} ({pct:.0f}%) ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Compute ranking
    ranking = []
    for sname in all_strategies:
        scores = results.get(sname, [])
        if not scores:
            continue
        avg = np.mean(scores)
        s = sorted(scores)
        trim = max(1, len(s)//10)
        trimmed = np.mean(s[trim:-trim]) if len(s) > 2*trim else avg
        positive = sum(1 for x in scores if x > 0.2) / len(scores) * 100
        avg_ret = np.mean(returns.get(sname, [0]))
        is_active = sname in active_names
        ranking.append((sname, avg, trimmed, positive, len(scores), is_active, avg_ret))

    ranking.sort(key=lambda x: x[2], reverse=True)

    # Print console
    print(f"\n{'='*90}")
    print(f"{'#':>3} {'状态':<6} {'策略':<30} {'均分':>8} {'Trimmed':>8} {'平均收益':>8} {'正收益%':>8} {'股票数':>6}")
    print(f"{'='*90}")
    for i, (sname, avg, trimmed, pos, n, active, avg_ret) in enumerate(ranking):
        tag = "🟢活跃" if active else "⬜归档"
        print(f"{i+1:>3} {tag} {sname:<30} {avg:>8.4f} {trimmed:>8.4f} {avg_ret:>7.1f}% {pos:>7.1f}% {n:>5}")

    # Generate HTML
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>策略评测 - MYPOOL</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
h1 { color: #00d4ff; text-align: center; }
.info { text-align: center; color: #888; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; max-width: 900px; margin: 0 auto; }
th { background: #16213e; color: #00d4ff; padding: 10px 14px; text-align: left; border-bottom: 2px solid #0f3460; }
td { padding: 8px 14px; border-bottom: 1px solid #333; }
tr:hover { background: #1a1a3e !important; }
.active { background: rgba(0, 212, 255, 0.12); font-weight: 600; }
.active td:first-child::after { content: ''; }
.tag-active { background: #00d4ff; color: #000; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 700; }
.tag-archived { background: #555; color: #ccc; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
.score-high { color: #00ff88; }
.score-mid { color: #ffcc00; }
.score-low { color: #ff6666; }
.rank { color: #888; }
</style></head><body>
<h1>📊 全部策略 MYPOOL 评测排名</h1>
<div class="info">MYPOOL """ + str(len(datasets)) + """ 只股票 | 评测时间: """ + time.strftime("%Y-%m-%d %H:%M") + """ | 耗时: """ + f"{elapsed:.1f}s" + """</div>
<table>
<tr><th>#</th><th>状态</th><th>策略名称</th><th>均分</th><th>Trimmed均分</th><th>平均收益</th><th>正收益%</th><th>股票数</th></tr>
"""
    for i, (sname, avg, trimmed, pos, n, active, avg_ret) in enumerate(ranking):
        row_class = 'class="active"' if active else ''
        tag = '<span class="tag-active">🟢 活跃</span>' if active else '<span class="tag-archived">⬜ 归档</span>'
        if trimmed > 0.40:
            score_class = "score-high"
        elif trimmed > 0.25:
            score_class = "score-mid"
        else:
            score_class = "score-low"
        ret_class = "score-high" if avg_ret > 20 else ("score-mid" if avg_ret > 0 else "score-low")
        html += f'<tr {row_class}><td class="rank">{i+1}</td><td>{tag}</td><td>{sname}</td>'
        html += f'<td class="{score_class}">{avg:.4f}</td><td class="{score_class}">{trimmed:.4f}</td>'
        html += f'<td class="{ret_class}">{avg_ret:.1f}%</td>'
        html += f'<td>{pos:.1f}%</td><td>{n}</td></tr>\n'

    html += """</table>
<div class="info" style="margin-top:20px;">评分公式: 0.40×收益率 + 0.25×Sharpe/3 + 0.20×(1-最大回撤) + 0.10×胜率 + 0.05×盈亏比/5</div>
</body></html>"""

    with open("static/eval_all.html", "w") as f:
        f.write(html)
    print(f"\n✅ HTML报告已保存: static/eval_all.html")
    print(f"   访问: http://localhost:9000/static/eval_all.html")

    # 保存 JSON 结果
    results_dir = Path("auto_optimize_results")
    results_dir.mkdir(exist_ok=True)
    run_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    result_data = {
        "run_time": run_ts,
        "elapsed_s": round(elapsed, 1),
        "stocks_count": len(datasets),
        "strategies_count": len(all_strategies),
        "ranking": [
            {
                "rank": i + 1,
                "name": sname,
                "status": "active" if active else "archived",
                "avg_score": round(avg, 4),
                "trimmed_score": round(trimmed, 4),
                "avg_return_pct": round(avg_ret, 2),
                "positive_pct": round(pos, 1),
                "stocks": n,
            }
            for i, (sname, avg, trimmed, pos, n, active, avg_ret) in enumerate(ranking)
        ],
    }
    json_path = results_dir / "eval_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON结果已保存: {json_path}")

    # 保存 CSV 结果
    csv_path = results_dir / "eval_all.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_time", "rank", "status", "strategy", "avg_score",
                    "trimmed_score", "avg_return_pct", "positive_pct", "stocks"])
        for i, (sname, avg, trimmed, pos, n, active, avg_ret) in enumerate(ranking):
            w.writerow([run_ts, i + 1, "active" if active else "archived",
                        sname, f"{avg:.4f}", f"{trimmed:.4f}",
                        f"{avg_ret:.2f}", f"{pos:.1f}", n])
    print(f"✅ CSV结果已保存: {csv_path}")
