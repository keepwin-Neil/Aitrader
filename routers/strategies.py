"""
Strategies router: /strategies, /strategies/*, /strategies/evaluate
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pathlib import Path
from datetime import datetime
import threading

import main as _m
import proc_role

router = APIRouter()



def _slice_history(df, n=None, date_from=None, date_to=None):
    """按「最近 n 根」或「指定时间段」裁剪行情。都不给就是全部历史。

    买点评测以前日线固定取 period="max" —— NVDA 能一路翻到 1999 年的买点。
    行情结构二十年前跟现在完全是两回事, 拿那段去评超短线买点意义不大。
    三种口径:
      · 全部历史        n/date 都不给
      · 最近 n 根       n>0
      · 指定时间段      date_from / date_to (任一给一个即可, 闭区间)
    裁剪都在指标计算**之前**, 跟「📊 评测排名」的 N 根口径一致。
    """
    if df is None or len(df) == 0:
        return df
    if date_from or date_to:
        try:
            import pandas as _pd
            idx = df.index
            if date_from:
                lo = _pd.Timestamp(date_from)
                if getattr(idx, "tz", None) is not None and lo.tzinfo is None:
                    lo = lo.tz_localize(idx.tz)
                df = df[idx >= lo]
                idx = df.index
            if date_to:
                # 闭区间: 结束日当天也算进来, 所以取那天的 23:59:59
                hi = _pd.Timestamp(date_to) + _pd.Timedelta(days=1) - _pd.Timedelta(seconds=1)
                if getattr(idx, "tz", None) is not None and hi.tzinfo is None:
                    hi = hi.tz_localize(idx.tz)
                df = df[idx <= hi]
        except Exception:
            return df      # 日期给得不对就当没给, 别把整个请求搞挂
        return df
    if n and n > 0:
        return df.tail(int(n)) if len(df) > int(n) else df
    return df


def _tail_bars(df, n):
    """兼容旧名。"""
    return _slice_history(df, n=n)


def _preload_stock_cache(symbols, market, dl_period, interval, strategy_name, signal_type,
                         history_bars=None, date_from=None, date_to=None):
    """并发预取 {sym: (df, params)}。

    取数是 I/O (读 parquet, 当天首次还要联网做增量更新), 串行跑 N 只就是
    N 次网络往返的死等, 而后面的并行回测一个都还没开始。用线程池不用进程池:
    get_ohlcv 是 I/O 密集, 且要落进程内缓存, 换进程反而每个 worker 各缓存一份。
    """
    from concurrent.futures import ThreadPoolExecutor as _TP, as_completed as _ac
    from backtest import get_ohlcv     # 本模块里 backtest 都是函数内 import

    def _one(sym):
        try:
            df = get_ohlcv(sym, period=dl_period, interval=interval, market=market)
            df = _slice_history(df, history_bars, date_from, date_to)
            if df is None or len(df) < 30:
                return sym, None
            params = _m._get_standard_strategy_params(sym, strategy_name,
                                                      interval=interval, signal_type=signal_type)
            return sym, (df, params)
        except Exception:
            return sym, None

    # 按**入参顺序**回填, 不按完成顺序: 这个 dict 的迭代顺序会一路流到
    # per_stock, 而前端对 per_stock 是按 success_rate 做**稳定**排序的 ——
    # 成功率相同的股票(100.0 / 0.0 很常见)会每次刷新换位置。
    got = {}
    with _TP(max_workers=max(1, min(8, len(symbols)))) as ex:
        for f in _ac([ex.submit(_one, s) for s in symbols]):
            sym, val = f.result()
            if val is not None:
                got[sym] = val
    return {s: got[s] for s in symbols if s in got}


@router.get("/strategies")
def get_strategies():
    """列出所有可用策略（含收藏和仓库状态）。以 warehouse.json 为唯一状态源。"""
    warehouse_data = _m._load_warehouse()
    llm_registry = _m._load_strategy_llm_registry()
    favourites = set(warehouse_data.get("favourites", []))
    active_set = set(warehouse_data.get("active", []))
    new_set = set(warehouse_data.get("new_strategies", []))
    archived_names = {a["name"] for a in warehouse_data.get("archived", [])}
    deleted_names = {d["name"] if isinstance(d, dict) else d for d in warehouse_data.get("deleted", [])}

    from strategies import STRATEGY_WAREHOUSE
    result = []
    for s in _m.list_strategies():
        name = s["name"]
        if name in archived_names or name in deleted_names:
            continue
        s["favourite"] = name in favourites
        s["llm_owner"] = _m._get_strategy_llm_owner(name, llm_registry)
        if name in new_set:
            s["status"] = "new"
        elif name in active_set:
            s["status"] = "active"
        else:
            s["status"] = "new"
        result.append(s)
    for name, strat in STRATEGY_WAREHOUSE.items():
        if name in archived_names or name in deleted_names:
            continue
        if any(r["name"] == name for r in result):
            continue
        result.append({
            "name": name, "description": strat.description,
            "status": "active" if name in active_set else "new" if name in new_set else "new",
            "favourite": name in favourites,
            "llm_owner": _m._get_strategy_llm_owner(name, llm_registry),
        })
    for a in warehouse_data.get("archived", []):
        result.append({
            "name": a["name"], "description": f"[已归档] {a.get('reason','')}",
            "status": "archived", "favourite": a["name"] in favourites,
            "archived_at": a.get("archived_at"),
            "avg_annual_return": a.get("avg_annual_return"),
            "llm_owner": _m._get_strategy_llm_owner(a["name"], llm_registry),
        })
    for d in warehouse_data.get("deleted", []):
        dname = d["name"] if isinstance(d, dict) else d
        result.append({
            "name": dname, "description": f"[已删除] {d.get('reason','') if isinstance(d, dict) else ''}",
            "status": "deleted", "favourite": False,
            "deleted_at": d.get("deleted_at") if isinstance(d, dict) else None,
            "llm_owner": _m._get_strategy_llm_owner(dname, llm_registry),
        })
    return {
        "strategies": result,
        "llm_registry": {
            "default_owner": llm_registry.get("default_owner", _m.DEFAULT_STRATEGY_LLM_OWNER),
            "supported_owners": llm_registry.get("supported_owners", _m.SUPPORTED_STRATEGY_LLM_OWNERS),
            "guide_path": llm_registry.get("guide_path", "docs/strategy_llm_authoring_guide.md"),
        },
    }


@router.post("/strategies/{name}/favourite")
def toggle_favourite(name: str):
    """收藏/取消收藏策略。收藏的策略不会被自动归档；仓库中的策略被收藏后自动恢复"""
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE
    wh = _m._load_warehouse()
    favs = wh.get("favourites", [])
    restored = False
    if name in favs:
        favs.remove(name)
        action = "unfavourited"
    else:
        favs.append(name)
        action = "favourited"
        archived_names = [a["name"] for a in wh.get("archived", [])]
        if name in archived_names:
            wh["archived"] = [a for a in wh["archived"] if a["name"] != name]
            if name not in wh.get("active", []):
                wh["active"].append(name)
            if name in STRATEGY_WAREHOUSE and name not in STRATEGY_REGISTRY:
                STRATEGY_REGISTRY[name] = STRATEGY_WAREHOUSE.pop(name)
            wh.setdefault("history", []).append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "action": "restored",
                "strategies": [name],
                "reason": "收藏后自动从仓库恢复",
            })
            restored = True
    wh["favourites"] = favs
    _m._save_warehouse(wh)
    return {"strategy": name, "action": action, "restored": restored, "favourites": favs}


@router.post("/strategies/{name}/archive")
def archive_strategy(name: str, body: dict = None):
    """将策略归档入库"""
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE
    wh = _m._load_warehouse()
    favs = set(wh.get("favourites", []))
    if name in favs:
        raise HTTPException(400, f"策略 {name} 已被收藏，不能归档。请先取消收藏。")
    if name in wh.get("active", []):
        wh["active"].remove(name)
    if name in wh.get("new_strategies", []):
        wh["new_strategies"].remove(name)
    if name in STRATEGY_REGISTRY:
        STRATEGY_WAREHOUSE[name] = STRATEGY_REGISTRY.pop(name)
    reason = (body or {}).get("reason", "手动归档")
    wh.setdefault("archived", []).append({
        "name": name, "archived_at": datetime.now().strftime("%Y-%m-%d"),
        "reason": reason,
    })
    wh.setdefault("history", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "archived", "strategies": [name], "reason": reason,
    })
    _m._save_warehouse(wh)
    return {"archived": name, "reason": reason}


@router.post("/strategies/{name}/restore")
def restore_strategy(name: str):
    """激活策略：从归档/新策略恢复为活跃状态"""
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE
    wh = _m._load_warehouse()
    archived_names = [a["name"] for a in wh.get("archived", [])]
    if name in archived_names:
        wh["archived"] = [a for a in wh["archived"] if a["name"] != name]
    if name in wh.get("new_strategies", []):
        wh["new_strategies"].remove(name)
    if name not in wh.get("active", []):
        wh.setdefault("active", []).append(name)
    if name in STRATEGY_WAREHOUSE and name not in STRATEGY_REGISTRY:
        STRATEGY_REGISTRY[name] = STRATEGY_WAREHOUSE.pop(name)
    wh.setdefault("history", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "activated", "strategies": [name], "reason": "手动激活",
    })
    _m._save_warehouse(wh)
    return {"restored": name}


@router.post("/strategies/{name}/delete")
def delete_strategy(name: str, body: dict = None):
    """将策略移到已删除（从归档中移出，不再参与评测但保留作为负面参考）"""
    from strategies import STRATEGY_REGISTRY, STRATEGY_WAREHOUSE
    wh = _m._load_warehouse()
    archived_names = [a["name"] for a in wh.get("archived", [])]
    if name not in archived_names:
        if name in wh.get("active", []):
            wh["active"].remove(name)
        if name in wh.get("new_strategies", []):
            wh["new_strategies"].remove(name)
        if name in STRATEGY_REGISTRY:
            STRATEGY_WAREHOUSE[name] = STRATEGY_REGISTRY.pop(name)
    else:
        wh["archived"] = [a for a in wh["archived"] if a["name"] != name]

    deleted = wh.setdefault("deleted", [])
    if not any(d["name"] == name for d in deleted):
        reason = (body or {}).get("reason", "手动删除")
        deleted.append({
            "name": name,
            "deleted_at": datetime.now().strftime("%Y-%m-%d"),
            "reason": reason,
        })
    wh.setdefault("history", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "deleted", "strategies": [name],
        "reason": (body or {}).get("reason", "手动删除"),
    })
    _m._save_warehouse(wh)
    return {"deleted": name}


@router.post("/strategies/{name}/undelete")
def undelete_strategy(name: str):
    """从已删除恢复到归档"""
    wh = _m._load_warehouse()
    deleted = wh.get("deleted", [])
    target = None
    for d in deleted:
        if d["name"] == name:
            target = d
            break
    if not target:
        raise HTTPException(404, f"策略 {name} 不在已删除列表中")
    wh["deleted"] = [d for d in deleted if d["name"] != name]
    wh.setdefault("archived", []).append({
        "name": name,
        "archived_at": datetime.now().strftime("%Y-%m-%d"),
        "reason": f"从已删除恢复: {target.get('reason', '')}",
    })
    wh.setdefault("history", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "undeleted", "strategies": [name], "reason": "从已删除恢复到归档",
    })
    _m._save_warehouse(wh)
    return {"undeleted": name}


@router.get("/strategies/warehouse")
def get_warehouse():
    """获取策略仓库状态"""
    return _m._load_warehouse()


@router.get("/strategies/validate-new/summary")
def get_new_strategy_validation_summary(owner: Optional[str] = Query(None), interval: str = Query("1d")):
    owner_norm = _m._normalize_strategy_owner(owner) if owner else None
    interval_norm = _m._normalize_param_interval(interval)
    strategies = _m._get_strategy_validation_summary(owner_norm, interval_norm)
    registry = _m._load_strategy_llm_registry()
    for name in _m._list_new_strategy_candidates(owner_norm):
        if name not in strategies:
            strategies[name] = {
                "owner": _m._get_strategy_llm_owner(name, registry),
                "interval": interval_norm,
                "final_status": "pending",
            }
    return {
        "owner": owner_norm,
        "interval": interval_norm,
        "strategies": strategies,
    }


@router.post("/strategies/validate-new/start")
def start_new_strategy_validation(body: dict = None):
    body = body or {}
    owner = _m._normalize_strategy_owner(body.get("owner") or "OpenAI")
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    n_trials = int(body.get("n_trials") or 60)
    # 可选: 只验证这几只 (前端勾选) · 空/未传 = 全部待验证新策略
    only = body.get("strategies")
    only_list = [str(s).strip() for s in only if str(s).strip()] if isinstance(only, list) else []

    running = [jid for jid, info in _m._new_strategy_validation_jobs.items() if info.get("status") == "running"]
    if running:
        raise HTTPException(409, f"已有新策略验证任务运行中 ({running[0]})")

    import time
    job_id = f"newval_{owner.lower()}_{int(time.time())}"
    # 走工厂而不是手写字面量: 手写的是普通 dict, worker 里的
    # setdefault(job_id, LockedDict(...)) 会因为 key 已存在而把新建的
    # LockedDict 丢掉, 这条路径就完全没锁 (上一轮就是这么漏掉的)。
    _m.register_job(_m._new_strategy_validation_jobs, job_id, {
        "job_id": job_id,
        "status": "queued",
        "owner": owner,
        "interval": interval,
        "n_trials": n_trials,
        "only_strategies": only_list,
        "total": 0,
        "done": 0,
        "validated": 0,
        "archived": 0,
        "failed": 0,
        "current": "",
        "message": "排队中...",
        "results": {},
        "started_at": datetime.utcnow().isoformat(),
    })
    threading.Thread(
        target=_m._new_strategy_validation_worker,
        args=(job_id, owner, interval, n_trials),
        daemon=True,
    ).start()
    return {"job_id": job_id, "owner": owner, "interval": interval, "only_strategies": only_list}


@router.get("/strategies/validate-new/status/{job_id}")
async def get_new_strategy_validation_status(job_id: str):
    job = _m._new_strategy_validation_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return _m.job_json_response(job)


@router.post("/strategies/validate-new/cancel/{job_id}")
def cancel_new_strategy_validation(job_id: str):
    """请求取消运行中的任务。worker 在下一次循环开始前（每个策略之间）检查
    cancelled 标志退出。如果当前正在某个策略的优化阶段，会等到那个策略完成
    才退出（最多几分钟）。"""
    job = _m._new_strategy_validation_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.get("status") not in ("running", "queued"):
        return {"job_id": job_id, "status": job.get("status"), "note": "任务已结束，无需取消"}
    job["cancelled"] = True
    return {"job_id": job_id, "status": "cancel_requested",
            "note": "已请求取消。worker 完成当前策略后会退出。"}


# 进度文件的路径和读写都在 eval_worker 里, 这里只复用, 不再各写一份
from eval_worker import PROGRESS_FILE as _EVAL_PROGRESS_FILE, IDLE as _EVAL_IDLE, \
    read_progress as _eval_read_progress, write_progress as _eval_write_progress


@router.get("/strategies/evaluate/progress")
async def evaluate_progress():
    """当前 (或最近一次) 评测的进度. 前端轮询这个端点画进度条.

    async: 同步端点会跟正在跑的重活抢 anyio 线程池, 轮询拿不到响应, 面板就卡住。
    这里只读一个几百字节的 JSON 文件, 放事件循环上不阻塞。

    带 run_id: 同一份文件会被「UI 评测排名」和「LLM 评估链」同时写
    (后者还会先 unlink 再起自己的 eval_worker), 没有身份标识的话,
    前端会把另一轮的 done/total 当成自己的画出来, 上一轮的终态也会
    在新一轮启动的空档里闪一下。前端按 run_id 过滤即可。"""
    return _eval_read_progress(default=dict(_EVAL_IDLE))


@router.post("/strategies/evaluate")
async def evaluate_strategies(body: dict):
    """对指定策略池评测所有策略（含活跃/新/归档），返回排名表。当日结果缓存复用。"""
    import asyncio
    import json
    import pickle
    import subprocess
    import numpy as np
    from pathlib import Path
    from backtest import get_ohlcv

    llm_registry = _m._load_strategy_llm_registry()
    pool_name = body.get("pool", "MYPOOL")
    interval = body.get("interval", "1d")
    limit = body.get("limit", 500)
    force = body.get("force", False)
    # 前端传来的策略清单 = 当前 tab 选中的那些. 只评测它们:
    # 以前无论在哪个 tab 点, 后端都固定跑全部 164 个策略 × 全池 —— MYPOOL 37 只
    # 就是 6068 次回测, 会把机器压满并撞上 subprocess 600s 超时 → 前端只看到
    # "评测失败: Internal Server Error". 传了清单就只跑清单.
    req_strategies = body.get("strategies")
    req_strategies = [str(x) for x in req_strategies if str(x).strip()] if isinstance(req_strategies, list) else []

    # 第一件事就占住进度文件: 前端的轮询在 POST 之前就起来了 (tick() 先于 fetch),
    # 不先覆盖的话它会读到上一轮的终态 —— 上次失败就闪一下红色错误,
    # 上次是评估链跑的就闪一个 100% 的满进度条。run_id 让前端能认出是不是自己这轮。
    import time as _t
    import uuid as _uuid
    _run_id = _uuid.uuid4().hex[:12]
    _eval_write_progress(run_id=_run_id, stage="start", done=0, total=0,
                         sym_done=0, sym_total=0, current="", workers=0,
                         pool=pool_name, interval=interval,
                         started_at=_t.time(), finished_at=None, error=None)

    pool_path = Path(__file__).parent.parent / "stock_pools.json"
    if not pool_path.exists():
        raise HTTPException(404, "stock_pools.json not found")
    with open(pool_path, encoding="utf-8") as f:
        _pools_all = json.load(f)
    _pool_info_pre = _pools_all.get(pool_name)
    if not _pool_info_pre:
        raise HTTPException(404, f"Pool '{pool_name}' not found")
    _syms_pre = _pool_info_pre["symbols"]

    _syms_hash = _m._hash_symbols(_syms_pre)
    _mkt_pre = _pool_info_pre.get("market", "US")
    _mtime_sig = _m._data_mtime_sig(_syms_pre, _mkt_pre, interval)

    _eval_write_progress(sym_total=len(_syms_pre))

    _strat_sig = _m._hash_symbols(sorted(req_strategies)) if req_strategies else "ALL"
    eval_cache_key = f"eval:{pool_name}:{interval}:{limit}:{_syms_hash}:{_mtime_sig}:{_strat_sig}"
    if not force:
        cached = _m.cache_get(eval_cache_key, ttl=86400)
        if cached is not None:
            wh = _m._load_warehouse()
            fav_set = set(wh.get("favourites", []))
            for r in cached.get("ranking", []):
                r["status"] = _m._classify_strategy_status(r["name"], wh)
                r["favourite"] = r["name"] in fav_set
                r["llm_owner"] = _m._get_strategy_llm_owner(r["name"], llm_registry)
            cached["from_cache"] = True
            _eval_write_progress(stage="done", from_cache=True, finished_at=_t.time())
            return cached

    pool_info = _pool_info_pre
    symbols = _syms_pre
    market = pool_info.get("market", "US")

    wh = _m._load_warehouse()
    fav_set = set(wh.get("favourites", []))
    deleted_set = {d["name"] if isinstance(d, dict) else d for d in wh.get("deleted", [])}

    from strategies import STRATEGY_WAREHOUSE
    _known = set(list(_m.STRATEGY_REGISTRY.keys()) + list(STRATEGY_WAREHOUSE.keys()))
    if req_strategies:
        all_strats = [s for s in req_strategies if s in _known and s not in deleted_set]
        if not all_strats:
            raise HTTPException(400, "请求的策略都不存在或已删除")
    else:
        all_strats = [s for s in _known if s not in deleted_set]

    loop = asyncio.get_event_loop()

    def _download_and_run():
        _period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
            "1h": "2y", "1d": "max", "1wk": "max", "1mo": "max",
        }
        dl_period = _period_map.get(interval, "2y")

        datasets = {}
        _eval_write_progress(stage="download", sym_done=0, sym_total=len(symbols))
        for _i, sym in enumerate(symbols, 1):
            _eval_write_progress(stage="download", sym_done=_i, sym_total=len(symbols), current=sym)
            try:
                df = get_ohlcv(sym, period=dl_period, interval=interval, market=market)
                if df is not None and len(df) > 30:
                    if limit and len(df) > limit:
                        df = df.iloc[-limit:]
                    datasets[sym] = pickle.dumps(df)
            except Exception:
                pass
        if not datasets:
            raise RuntimeError("No data available for evaluation")

        task_file = Path(__file__).parent.parent / "_eval_tasks.pkl"
        result_file = Path(__file__).parent.parent / "_eval_results.pkl"
        task_data = {
            "strategies": all_strats,
            "datasets": datasets,
        }
        with open(task_file, "wb") as f:
            pickle.dump(task_data, f)

        venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        # 超时按任务量给: 每 1000 组回测 200s, 下限 300s 上限 1800s.
        # (固定 600s 时, MYPOOL×全部策略 的 6068 组正好卡在边界, 超时后前端只看到
        #  一句 Internal Server Error, 完全不知道发生了什么)
        _n_tasks = len(all_strats) * len(datasets)
        _timeout = max(300, min(1800, int(_n_tasks / 1000 * 200) + 120))
        try:
            proc = subprocess.run(
                [python_cmd, str(Path(__file__).parent.parent / "eval_worker.py")],
                cwd=str(Path(__file__).parent.parent),
                capture_output=True, timeout=_timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"评测超时: {len(all_strats)} 策略 × {len(datasets)} 只股票 = {_n_tasks} 组回测, "
                f"超过 {_timeout}s 未完成. 建议减少策略数 (切到具体 tab) 或换小一些的股票池.")
        if proc.returncode != 0:
            raise RuntimeError(f"eval_worker failed: {proc.stderr.decode()[-500:]}")
        with open(result_file, "rb") as f:
            return pickle.load(f)

    try:
        results = await loop.run_in_executor(None, _download_and_run)
    except Exception as e:
        _eval_write_progress(stage="error", error=str(e)[:300], finished_at=_t.time())
        raise HTTPException(500, str(e)[:500] or "评测失败")
    _eval_write_progress(stage="rank", finished_at=None)

    ranking = []
    for sname in all_strats:
        scores_data = results.get(sname, [])
        if not scores_data:
            continue
        scores = [d[0] for d in scores_data]
        rets = [d[1] for d in scores_data]
        sharpes = [d[2] for d in scores_data]
        dds = [d[3] for d in scores_data]
        wrs = [d[4] for d in scores_data]
        trades_list = [d[6] if len(d) > 6 else 0 for d in scores_data]
        wins_list = [d[7] if len(d) > 7 else 0 for d in scores_data]

        avg_score = float(np.mean(scores))
        s_sorted = sorted(scores)
        trim = max(1, len(s_sorted) // 10)
        trimmed = float(np.mean(s_sorted[trim:-trim])) if len(s_sorted) > 2 * trim else avg_score

        avg_ret = float(np.mean(rets))
        avg_sharpe = float(np.mean(sharpes))
        avg_dd = float(np.mean(dds))
        avg_wr = float(np.mean(wrs))
        pos_pct = sum(1 for r in rets if r > 0) / len(rets) * 100
        total_trades = int(sum(trades_list))
        total_wins = int(sum(wins_list))

        status = _m._classify_strategy_status(sname, wh)

        ranking.append({
            "name": sname,
            "status": status,
            "favourite": sname in fav_set,
            "llm_owner": _m._get_strategy_llm_owner(sname, llm_registry),
            "score": round(avg_score, 4),
            "trimmed_score": round(trimmed, 4),
            "avg_return": round(avg_ret, 1),
            "max_drawdown": round(avg_dd, 1),
            "win_rate": round(avg_wr, 1),
            "positive_pct": round(pos_pct, 1),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "stock_count": len(scores_data),
        })

    ranking.sort(key=lambda x: x["trimmed_score"], reverse=True)

    for i, r in enumerate(ranking):
        r["rank"] = i + 1

    new_summary = []
    for r in ranking:
        if r["status"] == "new":
            new_summary.append(f"{r['name']} 排第{r['rank']}名")

    result = {
        "pool": pool_name,
        "stock_count": len(symbols),
        "strategy_count": len(ranking),
        "ranking": ranking,
        # new_summary: 前端 0 处引用, 且实测 13 份真实响应全是空数组
        # (warehouse 的 new_strategies 一直为空)。留着当死字段。
        "new_summary": new_summary,
    }
    _m.cache_set(eval_cache_key, result)
    _eval_write_progress(stage="done", from_cache=False, finished_at=_t.time())
    return result


# ── BUY 点验证 (MFE/MAE 分析) ──────────────────────────────────

@router.post("/strategies/buy-point-analysis")
async def buy_point_analysis(body: dict):
    """对单个策略做 BUY 点质量分析（MFE/MAE），支持单股或股票池。
    使用个股标准参数（通用标准 + 个股 custom_params 覆盖）。"""
    import asyncio, json, pickle, numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from backtest import get_ohlcv, analyze_buy_points

    strategy_name = body.get("strategy")
    if not strategy_name:
        raise HTTPException(400, "缺少 strategy 参数")

    market = body.get("market", "US")
    period = body.get("period", "2y")
    interval = body.get("interval", "1d")
    window = int(body.get("window", 20))
    mfe_target = float(body.get("mfe_target", 20))
    mae_limit = float(body.get("mae_limit", 10))
    entry_mode = body.get("entry_mode", "next_open")
    # judge_mode: "mfe_mae"(原有, 跟先后顺序无关) | "first_touch"(超短线, 先到先判)
    # first_touch 下 mfe_target / mae_limit 的含义变成「涨幅阈值 / 回撤阈值」
    judge_mode = body.get("judge_mode", "mfe_mae")
    same_bar = body.get("same_bar", "adverse_first")
    # 用多少历史: "all" 全部 / "bars" 最近 N 根 / "range" 指定时间段
    _hmode = str(body.get("history_mode", "all"))
    history_bars = None
    date_from = date_to = None
    if _hmode == "bars":
        try:
            history_bars = max(50, int(body.get("history_bars", 500)))
        except (TypeError, ValueError):
            history_bars = 500
    elif _hmode == "range":
        date_from = (body.get("date_from") or "").strip() or None
        date_to = (body.get("date_to") or "").strip() or None
    pool_name = body.get("pool")
    symbol = body.get("symbol")

    # 解析股票列表
    if pool_name:
        pool_path = Path(__file__).parent.parent / "stock_pools.json"
        if not pool_path.exists():
            raise HTTPException(404, "stock_pools.json not found")
        with open(pool_path, encoding="utf-8") as f:
            pools = json.load(f)
        pool_info = pools.get(pool_name)
        if not pool_info:
            raise HTTPException(404, f"Pool '{pool_name}' not found")
        symbols = pool_info["symbols"]
        market = pool_info.get("market", market)
    elif symbol:
        symbols = [symbol.upper()]
    else:
        raise HTTPException(400, "需要 pool 或 symbol 参数")

    # 多窗口列表
    multi_windows = [5, 10, 20, 40, 60]
    if window not in multi_windows:
        multi_windows = sorted(set(multi_windows + [window]))

    loop = asyncio.get_event_loop()

    def _run():
        _period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
            "1h": "2y", "1d": "max", "1wk": "max", "1mo": "max",
        }
        dl_period = _period_map.get(interval, "2y")

        # ── 1) 一次性下载全部股票数据 + 参数（不在窗口循环中重复下载）──
        stock_cache = _preload_stock_cache(symbols, market, dl_period, interval,
                                           strategy_name, "buy", history_bars=history_bars,
                                           date_from=date_from, date_to=date_to)

        if not stock_cache:
            from backtest import _empty_bp_stats
            return {
                "strategy": strategy_name,
                "pool": pool_name or symbols[0],
                "window": window, "mfe_target": mfe_target, "mae_limit": mae_limit,
                "entry_mode": entry_mode, "judge_mode": judge_mode, "same_bar": same_bar,
            "history_bars": history_bars, "date_from": date_from, "date_to": date_to,
            "params_source": _m._params_source_summary(symbols, [strategy_name], interval),
                "aggregate": _empty_bp_stats(), "multi_window": [], "per_stock": [],
                "stock_count": len(symbols),
            }

        # ── 2) 主窗口 + 多窗口 一趟并行 ──
        # 之前是双层串行: 主窗口逐股跑一遍, 再 6 个窗口 × N 只股票又跑一遍。
        # 95 只的池子 = 570 次分析全在一个线程里, 而同文件的 buy-point-batch
        # (多策略版) 早就并行了。主窗口本来就在 multi_windows 里
        # (上面 sorted(set(multi_windows + [window]))), 所以合成一趟就够,
        # 顺带省掉主窗口被重复算一次。
        _pk_cache = {sym: pickle.dumps(df) for sym, (df, _p) in stock_cache.items()}
        _bp_tasks = [
            (sym, _pk_cache[sym], strategy_name, params, w, entry_mode,
             mfe_target, mae_limit, (w == window), judge_mode, same_bar)
            for w in multi_windows
            for sym, (df, params) in stock_cache.items()
        ]
        _n_workers = max(1, min(_m.N_WORKERS, len(_bp_tasks)))
        _by_window = {w: [] for w in multi_windows}
        with ProcessPoolExecutor(max_workers=_n_workers, initializer=proc_role.mark_pool_worker) as _pool:
            for _r in _pool.map(_m._bp_one_task, _bp_tasks):
                _by_window[_r["window"]].append(_r)

        per_stock = []
        all_buy_points = []
        for _r in _by_window.get(window, []):
            bps = _r["buy_points"]
            if not bps:
                continue
            success_count = sum(1 for bp in bps if bp["success"])
            per_stock.append({
                "symbol": _r["symbol"], "buy_points": bps,
                "stats": _r["stats"],
                "success_rate": round(success_count / len(bps) * 100, 1),
                "chart_data": _r.get("chart_data"),
            })
            all_buy_points.extend(bps)

        if not all_buy_points:
            from backtest import _empty_bp_stats
            return {
                "strategy": strategy_name,
                "pool": pool_name or symbols[0],
                "window": window, "mfe_target": mfe_target, "mae_limit": mae_limit,
                "entry_mode": entry_mode, "judge_mode": judge_mode, "same_bar": same_bar,
            "history_bars": history_bars, "date_from": date_from, "date_to": date_to,
            "params_source": _m._params_source_summary(symbols, [strategy_name], interval),
                "aggregate": _empty_bp_stats(), "multi_window": [], "per_stock": [],
                "stock_count": len(symbols),
            }

        # ── 3) 汇总统计 ──
        total_success = sum(1 for bp in all_buy_points if bp["success"])
        total_n = len(all_buy_points)
        success_rate_all = round(total_success / total_n * 100, 1)

        mfes = [bp["mfe"] for bp in all_buy_points]
        maes = [bp["mae"] for bp in all_buy_points]
        rs = [bp["r_ratio"] for bp in all_buy_points]
        bars_to_mfe = [bp["bars_to_mfe"] for bp in all_buy_points]

        mfe_dist = {
            ">30%": sum(1 for m in mfes if m > 30),
            "20-30%": sum(1 for m in mfes if 20 < m <= 30),
            "10-20%": sum(1 for m in mfes if 10 < m <= 20),
            "<10%": sum(1 for m in mfes if m <= 10),
        }
        mae_dist = {
            "<5%": sum(1 for m in maes if abs(m) < 5),
            "5-10%": sum(1 for m in maes if 5 <= abs(m) < 10),
            ">10%": sum(1 for m in maes if abs(m) >= 10),
        }

        aggregate = {
            "total_buy_points": total_n,
            "success_rate": success_rate_all,
            "avg_mfe": round(float(np.mean(mfes)), 2),
            "avg_mae": round(float(np.mean(maes)), 2),
            "median_mfe": round(float(np.median(mfes)), 2),
            "median_mae": round(float(np.median(maes)), 2),
            "avg_r_ratio": round(float(np.mean(rs)), 2),
            "avg_bars_to_mfe": round(float(np.mean(bars_to_mfe)), 1),
            "mfe_distribution": mfe_dist,
            "mae_distribution": mae_dist,
        }
        # 先到先判: 多少根之内分出胜负 —— 比「最优窗口」有用得多。
        # (最优窗口在这个口径下恒等于试过的最长窗口, 见 _calc_bp_stats 的注释)
        _decided = [bp.get("bars_to_outcome") for bp in all_buy_points
                    if bp.get("bars_to_outcome") is not None]
        if _decided:
            aggregate["decided_count"] = len(_decided)
            aggregate["decided_pct"] = round(len(_decided) / total_n * 100, 1)
            aggregate["median_bars_to_outcome"] = round(float(np.median(_decided)), 1)

        # ── 4) 多窗口汇总（结果在上面那趟并行里已经算好了）──
        multi_window_results = []
        for w in multi_windows:
            w_bps = []
            for _r in _by_window.get(w, []):
                w_bps.extend(_r["buy_points"])
            if w_bps:
                w_success = sum(1 for bp in w_bps if bp["success"])
                multi_window_results.append({
                    "window": w,
                    "total_signals": len(w_bps),
                    "success_rate": round(w_success / len(w_bps) * 100, 1),
                    "avg_mfe": round(float(np.mean([bp["mfe"] for bp in w_bps])), 2),
                    "avg_mae": round(float(np.mean([bp["mae"] for bp in w_bps])), 2),
                    "avg_r_ratio": round(float(np.mean([bp["r_ratio"] for bp in w_bps])), 2),
                })
            else:
                multi_window_results.append({
                    "window": w, "total_signals": 0, "success_rate": 0,
                    "avg_mfe": 0, "avg_mae": 0, "avg_r_ratio": 0,
                })

        best_w = max(multi_window_results, key=lambda x: x["success_rate"])
        aggregate["best_window"] = best_w["window"]

        per_stock_slim = [{
            "symbol": ps["symbol"],
            "buy_point_count": len(ps["buy_points"]),
            "success_rate": ps["success_rate"],
            "stats": ps["stats"],
        } for ps in per_stock]

        return {
            "strategy": strategy_name,
            "pool": pool_name or symbols[0],
            "window": window, "mfe_target": mfe_target, "mae_limit": mae_limit,
            "entry_mode": entry_mode, "judge_mode": judge_mode, "same_bar": same_bar,
            "history_bars": history_bars, "date_from": date_from, "date_to": date_to,
            "params_source": _m._params_source_summary(symbols, [strategy_name], interval),
            "aggregate": aggregate, "multi_window": multi_window_results,
            "per_stock": per_stock_slim, "stock_count": len(symbols),
        }

    result = await loop.run_in_executor(None, _run)
    return result


@router.post("/strategies/buy-point-batch")
async def buy_point_batch(body: dict):
    """多策略 BUY 点批量分析 — ProcessPoolExecutor 真并行。
    一次请求，所有策略同时跑，充分利用多核CPU。"""
    import asyncio, json, pickle, numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout
    from backtest import get_ohlcv

    strategies = body.get("strategies", [])
    if not strategies:
        raise HTTPException(400, "缺少 strategies 参数")

    market = body.get("market", "US")
    interval = body.get("interval", "1d")
    window = int(body.get("window", 20))
    mfe_target = float(body.get("mfe_target", 20))
    mae_limit = float(body.get("mae_limit", 10))
    entry_mode = body.get("entry_mode", "next_open")
    # judge_mode: "mfe_mae"(原有, 跟先后顺序无关) | "first_touch"(超短线, 先到先判)
    # first_touch 下 mfe_target / mae_limit 的含义变成「涨幅阈值 / 回撤阈值」
    judge_mode = body.get("judge_mode", "mfe_mae")
    same_bar = body.get("same_bar", "adverse_first")
    # 用多少历史: "all" 全部 / "bars" 最近 N 根 / "range" 指定时间段
    _hmode = str(body.get("history_mode", "all"))
    history_bars = None
    date_from = date_to = None
    if _hmode == "bars":
        try:
            history_bars = max(50, int(body.get("history_bars", 500)))
        except (TypeError, ValueError):
            history_bars = 500
    elif _hmode == "range":
        date_from = (body.get("date_from") or "").strip() or None
        date_to = (body.get("date_to") or "").strip() or None
    pool_name = body.get("pool")
    symbol = body.get("symbol")

    if pool_name:
        pool_path = Path(__file__).parent.parent / "stock_pools.json"
        if not pool_path.exists():
            raise HTTPException(404, "stock_pools.json not found")
        with open(pool_path, encoding="utf-8") as f:
            pools = json.load(f)
        pool_info = pools.get(pool_name)
        if not pool_info:
            raise HTTPException(404, f"Pool '{pool_name}' not found")
        symbols = pool_info["symbols"]
        market = pool_info.get("market", market)
    elif symbol:
        symbols = [symbol.upper()]
    else:
        raise HTTPException(400, "需要 pool 或 symbol 参数")

    multi_windows = [5, 10, 20, 40, 60]
    if window not in multi_windows:
        multi_windows = sorted(set(multi_windows + [window]))

    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="Buy 买点评估",
                  scope=f"{pool_name or (symbols[0] if symbols else '')} · {len(symbols)} 只 · {len(strategies)} 策略 · {interval}",
                  steps=[{"key": "dl", "label": "拉取行情"},
                         {"key": "bt", "label": "逐策略并行评估"}])

    loop = asyncio.get_event_loop()

    def _run_batch():
        _period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
            "1h": "2y", "1d": "max", "1wk": "max", "1mo": "max",
        }
        dl_period = _period_map.get(interval, "2y")

        # 1) 主进程一次性下载全部股票数据 + pickle 序列化
        stock_data = []  # [(sym, df_bytes, params), ...]
        _m.prog_step(task_id, "dl", total=len(symbols), unit=" 只", workers=1)
        for _i, sym in enumerate(symbols):
            _m.prog_update(task_id, done=_i, running=[sym])
            try:
                df = _slice_history(get_ohlcv(sym, period=dl_period, interval=interval, market=market),
                                    history_bars, date_from, date_to)
                if df is None or len(df) < 30:
                    continue
                params = _m._get_standard_strategy_params(sym, strategies[0], interval=interval, signal_type="buy")
                stock_data.append((sym, pickle.dumps(df), params))
            except Exception:
                continue

        if not stock_data:
            from backtest import _empty_bp_stats
            return [{"strategy": s, "aggregate": _empty_bp_stats(),
                     "multi_window": [], "per_stock": [], "stock_count": 0,
                     "pool": pool_name or symbols[0], "window": window,
                     "mfe_target": mfe_target, "mae_limit": mae_limit,
                     "entry_mode": entry_mode, "params_source": _m._params_source_summary(symbols, [s], interval)}
                    for s in strategies]

        # 2) 为每个策略构建 per-stock 参数（不同策略的 custom_params 不同）
        worker_tasks = []
        for strat in strategies:
            strat_stock_data = []
            for sym, df_bytes, _ in stock_data:
                params = _m._get_standard_strategy_params(sym, strat, interval=interval, signal_type="buy")
                strat_stock_data.append((sym, df_bytes, params))
            worker_tasks.append(
                (strat, strat_stock_data, multi_windows, window, mfe_target, mae_limit, entry_mode, interval,
                 judge_mode, same_bar)
            )

        # 3) ProcessPoolExecutor 真并行
        n_workers = min(_m.N_WORKERS, len(worker_tasks))
        results_map = {}
        _m.prog_step(task_id, "bt")
        _tr = _m.ProgTracker(task_id, [t[0] for t in worker_tasks], n_workers, unit=" 策略")
        # 超时: 原来写的是 future.result(timeout=120), 但那是死代码 ——
        # as_completed 只会吐出「已经 done」的 future, .result() 立即返回, 120s 永不触发,
        # 等于完全没有超时保护 (worker 真卡住就在这儿无限等)。
        # 改成给 as_completed 设真超时, 并按任务量算: 每个策略要跑 股票数 × 6 个窗口,
        # 95 只的池子单策略就可能几分钟, 写死 120s 本来也不合理。
        _to = max(300, min(3600, int(len(worker_tasks) * len(symbols) * 1.2) + 120))
        _timed_out = set(t[0] for t in worker_tasks)
        # 边完成边处理, 不要 list(as_completed(...)) 先耗尽再循环 ——
        # 那样 done_one 一次都不会在跑的过程中执行, 进度恒为 0/N 且
        # ProgTracker 的"正在启动进程"说明会挂满全程, 最后一瞬间跳到 N/N。
        with ProcessPoolExecutor(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
            future_to_strat = {pool.submit(_m._bp_analysis_worker, t): t[0] for t in worker_tasks}
            try:
                for future in as_completed(future_to_strat, timeout=_to):
                    strat = future_to_strat[future]
                    _timed_out.discard(strat)
                    _tr.done_one(strat)
                    try:
                        r = future.result()
                        r.update({"pool": pool_name or symbols[0], "window": window,
                                  "mfe_target": mfe_target, "mae_limit": mae_limit,
                                  "entry_mode": entry_mode, "params_source": _m._params_source_summary(symbols, [strat], interval)})
                        results_map[strat] = r
                    except Exception as e:
                        from backtest import _empty_bp_stats
                        results_map[strat] = {
                            "strategy": strat, "aggregate": _empty_bp_stats(),
                            "multi_window": [], "per_stock": [],
                            "pool": pool_name or symbols[0], "window": window,
                            "mfe_target": mfe_target, "mae_limit": mae_limit,
                            "entry_mode": entry_mode, "params_source": _m._params_source_summary(symbols, [strat], interval),
                            "error": str(e)[:200],
                        }
            except FuturesTimeout:
                _left = [f for f in future_to_strat if not f.done()]
                _m.logger.warning(f"buy-point-batch 超时 ({_to}s): {len(_left)} 个策略未完成, 强杀 worker")
                # future.cancel() 对已经在跑的任务返回 False(是 no-op), 而离开这个
                # with 会调 shutdown(wait=True) 去 join 管理线程 —— 只要还有 RUNNING
                # 的任务, 那个 join 永远不返回, 整个请求就此挂死, 连 500 都发不出去。
                # 必须直接干掉 worker 进程: 池子随即进入 BROKEN, shutdown 立刻返回。
                for _p in list(getattr(pool, "_processes", {}).values()):
                    try:
                        _p.kill()
                    except Exception:
                        pass

        # 没跑出结果的分两种: 超时被取消的 / 压根没排上的, 说清楚哪种
        return [results_map.get(
            s,
            {"strategy": s, "error": f"超时未完成 (上限 {_to}s)" if s in _timed_out else "未执行"}
        ) for s in strategies]

    try:
        results = await loop.run_in_executor(None, _run_batch)
    except Exception as e:
        _m.prog_end(task_id, "error", str(e)[:200])
        raise
    _m.prog_end(task_id, "done", f"{len(strategies)} 策略 × {len(symbols)} 只")
    return {"results": results, "strategy_count": len(strategies),
            "pool": pool_name or (symbols[0] if symbols else ""),
            "stock_count": len(symbols)}


@router.post("/strategies/sell-point-analysis")
async def sell_point_analysis(body: dict):
    """对单个策略做 SELL 点质量分析（MFE/MAE），支持单股或股票池。"""
    import asyncio, json, pickle, numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from backtest import get_ohlcv, analyze_sell_points, _empty_sp_stats

    strategy_name = body.get("strategy")
    if not strategy_name:
        raise HTTPException(400, "缺少 strategy 参数")

    market = body.get("market", "US")
    interval = body.get("interval", "1d")
    window = int(body.get("window", 20))
    mfe_target = float(body.get("mfe_target", 10))
    mae_limit = float(body.get("mae_limit", 5))
    # 跟买点镜像: first_touch 下 mfe_target/mae_limit 的含义变成
    # 「跌幅阈值(先到算卖对了) / 反弹阈值(先到算卖早了)」
    judge_mode = body.get("judge_mode", "mfe_mae")
    same_bar = body.get("same_bar", "adverse_first")
    _hmode = str(body.get("history_mode", "all"))
    history_bars = None
    date_from = date_to = None
    if _hmode == "bars":
        try:
            history_bars = max(50, int(body.get("history_bars", 500)))
        except (TypeError, ValueError):
            history_bars = 500
    elif _hmode == "range":
        date_from = (body.get("date_from") or "").strip() or None
        date_to = (body.get("date_to") or "").strip() or None
    pool_name = body.get("pool")
    symbol = body.get("symbol")

    if pool_name:
        pool_path = Path(__file__).parent.parent / "stock_pools.json"
        with open(pool_path, encoding="utf-8") as f:
            pools = json.load(f)
        pool_info = pools.get(pool_name)
        if not pool_info:
            raise HTTPException(404, f"Pool '{pool_name}' not found")
        symbols = pool_info["symbols"]
        market = pool_info.get("market", market)
    elif symbol:
        symbols = [symbol.upper()]
    else:
        raise HTTPException(400, "需要 pool 或 symbol 参数")

    multi_windows = [5, 10, 20, 40, 60]
    if window not in multi_windows:
        multi_windows = sorted(set(multi_windows + [window]))

    loop = asyncio.get_event_loop()

    def _run():
        _period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "2y", "1d": "max", "1wk": "max", "1mo": "max"}
        dl_period = _period_map.get(interval, "2y")

        stock_cache = _preload_stock_cache(symbols, market, dl_period, interval,
                                           strategy_name, "sell", history_bars=history_bars,
                                           date_from=date_from, date_to=date_to)

        if not stock_cache:
            return {"strategy": strategy_name, "pool": pool_name or symbols[0],
                    "window": window, "mfe_target": mfe_target, "mae_limit": mae_limit,
                    "judge_mode": judge_mode, "same_bar": same_bar, "history_bars": history_bars,
                    "date_from": date_from, "date_to": date_to,
                    "params_source": _m._params_source_summary(symbols, [strategy_name], interval), "aggregate": _empty_sp_stats(),
                    "multi_window": [], "per_stock": [], "stock_count": len(symbols)}

        # ── 主窗口 + 多窗口 一趟并行 (同 buy-point-analysis, 原来也是双层串行) ──
        _pk_cache = {sym: pickle.dumps(df) for sym, (df, _p) in stock_cache.items()}
        _sp_tasks = [
            (sym, _pk_cache[sym], strategy_name, params, w, mfe_target, mae_limit, (w == window),
             judge_mode, same_bar)
            for w in multi_windows
            for sym, (df, params) in stock_cache.items()
        ]
        _n_workers = max(1, min(_m.N_WORKERS, len(_sp_tasks)))
        _by_window = {w: [] for w in multi_windows}
        with ProcessPoolExecutor(max_workers=_n_workers, initializer=proc_role.mark_pool_worker) as _pool:
            for _r in _pool.map(_m._sp_one_task, _sp_tasks):
                _by_window[_r["window"]].append(_r)

        per_stock = []
        all_sell_points = []
        for _r in _by_window.get(window, []):
            sps = _r["sell_points"]
            if not sps: continue
            success_count = sum(1 for sp in sps if sp["success"])
            per_stock.append({"symbol": _r["symbol"], "sell_point_count": len(sps),
                              "success_rate": round(success_count / len(sps) * 100, 1),
                              "stats": _r["stats"]})
            all_sell_points.extend(sps)

        if not all_sell_points:
            return {"strategy": strategy_name, "pool": pool_name or symbols[0],
                    "window": window, "mfe_target": mfe_target, "mae_limit": mae_limit,
                    "judge_mode": judge_mode, "same_bar": same_bar, "history_bars": history_bars,
                    "date_from": date_from, "date_to": date_to,
                    "params_source": _m._params_source_summary(symbols, [strategy_name], interval), "aggregate": _empty_sp_stats(),
                    "multi_window": [], "per_stock": [], "stock_count": len(symbols)}

        total_success = sum(1 for sp in all_sell_points if sp["success"])
        total_n = len(all_sell_points)
        mfes = [sp["mfe"] for sp in all_sell_points]
        maes = [sp["mae"] for sp in all_sell_points]
        rs = [sp["r_ratio"] for sp in all_sell_points]

        aggregate = {
            "total_sell_points": total_n,
            "success_rate": round(total_success / total_n * 100, 1),
            "avg_mfe": round(float(np.mean(mfes)), 2),
            "avg_mae": round(float(np.mean(maes)), 2),
            "avg_r_ratio": round(float(np.mean(rs)), 2),
        }
        # 先到先判: 多少根之内分出胜负。跟买点一样, 单策略和批量是两份
        # aggregate 拼装代码, 两边都要加 —— 买点那边第一次就漏了批量那份。
        _decided = [sp.get("bars_to_outcome") for sp in all_sell_points
                    if sp.get("bars_to_outcome") is not None]
        if _decided:
            aggregate["decided_count"] = len(_decided)
            aggregate["decided_pct"] = round(len(_decided) / total_n * 100, 1)
            aggregate["median_bars_to_outcome"] = round(float(np.median(_decided)), 1)

        # 多窗口汇总（结果在上面那趟并行里已经算好了）
        multi_window_results = []
        for w in multi_windows:
            w_sps = []
            for _r in _by_window.get(w, []):
                w_sps.extend(_r["sell_points"])
            if w_sps:
                w_success = sum(1 for sp in w_sps if sp["success"])
                multi_window_results.append({
                    "window": w, "total_signals": len(w_sps),
                    "success_rate": round(w_success / len(w_sps) * 100, 1),
                    "avg_mfe": round(float(np.mean([sp["mfe"] for sp in w_sps])), 2),
                    "avg_mae": round(float(np.mean([sp["mae"] for sp in w_sps])), 2),
                    "avg_r_ratio": round(float(np.mean([sp["r_ratio"] for sp in w_sps])), 2),
                })
            else:
                multi_window_results.append({"window": w, "total_signals": 0, "success_rate": 0,
                                             "avg_mfe": 0, "avg_mae": 0, "avg_r_ratio": 0})

        best_w = max(multi_window_results, key=lambda x: x["success_rate"])
        aggregate["best_window"] = best_w["window"]

        return {"strategy": strategy_name, "pool": pool_name or symbols[0],
                "window": window, "mfe_target": mfe_target, "mae_limit": mae_limit,
                "judge_mode": judge_mode, "same_bar": same_bar, "history_bars": history_bars,
                "date_from": date_from, "date_to": date_to,
                "params_source": _m._params_source_summary(symbols, [strategy_name], interval), "aggregate": aggregate,
                "multi_window": multi_window_results, "per_stock": per_stock,
                "stock_count": len(symbols)}

    return await loop.run_in_executor(None, _run)


@router.post("/strategies/sell-point-batch")
async def sell_point_batch(body: dict):
    """多策略 SELL 点批量分析 — ProcessPoolExecutor 真并行。"""
    import asyncio, json, pickle, numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout
    from backtest import get_ohlcv, _empty_sp_stats

    strategies = body.get("strategies", [])
    if not strategies:
        raise HTTPException(400, "缺少 strategies 参数")

    market = body.get("market", "US")
    interval = body.get("interval", "1d")
    window = int(body.get("window", 20))
    mfe_target = float(body.get("mfe_target", 10))
    mae_limit = float(body.get("mae_limit", 5))
    # 跟买点镜像: first_touch 下 mfe_target/mae_limit 的含义变成
    # 「跌幅阈值(先到算卖对了) / 反弹阈值(先到算卖早了)」
    judge_mode = body.get("judge_mode", "mfe_mae")
    same_bar = body.get("same_bar", "adverse_first")
    _hmode = str(body.get("history_mode", "all"))
    history_bars = None
    date_from = date_to = None
    if _hmode == "bars":
        try:
            history_bars = max(50, int(body.get("history_bars", 500)))
        except (TypeError, ValueError):
            history_bars = 500
    elif _hmode == "range":
        date_from = (body.get("date_from") or "").strip() or None
        date_to = (body.get("date_to") or "").strip() or None
    pool_name = body.get("pool")
    symbol = body.get("symbol")

    if pool_name:
        pool_path = Path(__file__).parent.parent / "stock_pools.json"
        with open(pool_path, encoding="utf-8") as f:
            pools = json.load(f)
        pool_info = pools.get(pool_name)
        if not pool_info:
            raise HTTPException(404, f"Pool '{pool_name}' not found")
        symbols = pool_info["symbols"]
        market = pool_info.get("market", market)
    elif symbol:
        symbols = [symbol.upper()]
    else:
        raise HTTPException(400, "需要 pool 或 symbol 参数")

    multi_windows = [5, 10, 20, 40, 60]
    if window not in multi_windows:
        multi_windows = sorted(set(multi_windows + [window]))

    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="Sell 卖点评估",
                  scope=f"{pool_name or (symbols[0] if symbols else '')} · {len(symbols)} 只 · {len(strategies)} 策略 · {interval}",
                  steps=[{"key": "dl", "label": "拉取行情"},
                         {"key": "bt", "label": "逐策略并行评估"}])

    loop = asyncio.get_event_loop()

    def _run_batch():
        _period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "2y", "1d": "max", "1wk": "max", "1mo": "max"}
        dl_period = _period_map.get(interval, "2y")

        stock_data = []
        _m.prog_step(task_id, "dl", total=len(symbols), unit=" 只", workers=1)
        for _i, sym in enumerate(symbols):
            _m.prog_update(task_id, done=_i, running=[sym])
            try:
                df = _slice_history(get_ohlcv(sym, period=dl_period, interval=interval, market=market),
                                    history_bars, date_from, date_to)
                if df is None or len(df) < 30: continue
                params = _m._get_standard_strategy_params(sym, strategies[0], interval=interval, signal_type="sell")
                stock_data.append((sym, pickle.dumps(df), params))
            except Exception: continue

        if not stock_data:
            return [{"strategy": s, "aggregate": _empty_sp_stats(), "multi_window": [], "per_stock": [],
                     "pool": pool_name or symbols[0], "window": window, "mfe_target": mfe_target,
                     "mae_limit": mae_limit, "params_source": _m._params_source_summary(symbols, [s], interval)} for s in strategies]

        worker_tasks = []
        for strat in strategies:
            strat_stock_data = []
            for sym, df_bytes, _ in stock_data:
                params = _m._get_standard_strategy_params(sym, strat, interval=interval, signal_type="sell")
                strat_stock_data.append((sym, df_bytes, params))
            worker_tasks.append((strat, strat_stock_data, multi_windows, window, mfe_target, mae_limit, interval,
                                 judge_mode, same_bar))

        n_workers = min(_m.N_WORKERS, len(worker_tasks))
        results_map = {}
        _m.prog_step(task_id, "bt")
        _tr = _m.ProgTracker(task_id, [t[0] for t in worker_tasks], n_workers, unit=" 策略")
        # 超时: 原来写的是 future.result(timeout=120), 但那是死代码 ——
        # as_completed 只会吐出「已经 done」的 future, .result() 立即返回, 120s 永不触发,
        # 等于完全没有超时保护 (worker 真卡住就在这儿无限等)。
        # 改成给 as_completed 设真超时, 并按任务量算: 每个策略要跑 股票数 × 6 个窗口,
        # 95 只的池子单策略就可能几分钟, 写死 120s 本来也不合理。
        _to = max(300, min(3600, int(len(worker_tasks) * len(symbols) * 1.2) + 120))
        _timed_out = set(t[0] for t in worker_tasks)
        # 边完成边处理, 不要 list(as_completed(...)) 先耗尽再循环 ——
        # 那样 done_one 一次都不会在跑的过程中执行, 进度恒为 0/N 且
        # ProgTracker 的"正在启动进程"说明会挂满全程, 最后一瞬间跳到 N/N。
        with ProcessPoolExecutor(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
            future_to_strat = {pool.submit(_m._sp_analysis_worker, t): t[0] for t in worker_tasks}
            try:
                for future in as_completed(future_to_strat, timeout=_to):
                    strat = future_to_strat[future]
                    _timed_out.discard(strat)
                    _tr.done_one(strat)
                    try:
                        r = future.result()
                        r.update({"pool": pool_name or symbols[0], "window": window,
                                  "mfe_target": mfe_target, "mae_limit": mae_limit,
                                  "judge_mode": judge_mode, "same_bar": same_bar, "history_bars": history_bars,
                                  "date_from": date_from, "date_to": date_to,
                                  "params_source": _m._params_source_summary(symbols, [strat], interval)})
                        results_map[strat] = r
                    except Exception as e:
                        results_map[strat] = {"strategy": strat, "aggregate": _empty_sp_stats(),
                                              "multi_window": [], "per_stock": [],
                                              "pool": pool_name or symbols[0], "error": str(e)[:200]}
            except FuturesTimeout:
                _left = [f for f in future_to_strat if not f.done()]
                _m.logger.warning(f"sell-point-batch 超时 ({_to}s): {len(_left)} 个策略未完成, 强杀 worker")
                # 同 buy: cancel() 拦不住 RUNNING 的任务, 不杀进程就会卡死在 shutdown
                for _p in list(getattr(pool, "_processes", {}).values()):
                    try:
                        _p.kill()
                    except Exception:
                        pass

        # 没跑出结果的分两种: 超时被取消的 / 压根没排上的, 说清楚哪种
        return [results_map.get(
            s,
            {"strategy": s, "error": f"超时未完成 (上限 {_to}s)" if s in _timed_out else "未执行"}
        ) for s in strategies]

    try:
        results = await loop.run_in_executor(None, _run_batch)
    except Exception as e:
        _m.prog_end(task_id, "error", str(e)[:200])
        raise
    _m.prog_end(task_id, "done", f"{len(strategies)} 策略 × {len(symbols)} 只")
    return {"results": results, "strategy_count": len(strategies),
            "pool": pool_name or (symbols[0] if symbols else ""), "stock_count": len(symbols)}


@router.post("/strategies/buy-point-detail/{symbol}")
async def buy_point_detail(symbol: str, body: dict):
    """获取单只股票的 BUY 点详情（含 chart_data，用于 K 线渲染）。
    使用个股标准参数（通用标准 + 个股 custom_params 覆盖）。"""
    import asyncio
    from backtest import get_ohlcv, analyze_buy_points

    strategy_name = body.get("strategy")
    if not strategy_name:
        raise HTTPException(400, "缺少 strategy 参数")

    market = body.get("market", "US")
    interval = body.get("interval", "1d")
    window = int(body.get("window", 20))
    mfe_target = float(body.get("mfe_target", 20))
    mae_limit = float(body.get("mae_limit", 10))
    entry_mode = body.get("entry_mode", "next_open")
    # judge_mode: "mfe_mae"(原有, 跟先后顺序无关) | "first_touch"(超短线, 先到先判)
    # first_touch 下 mfe_target / mae_limit 的含义变成「涨幅阈值 / 回撤阈值」
    judge_mode = body.get("judge_mode", "mfe_mae")
    same_bar = body.get("same_bar", "adverse_first")
    # 用多少历史: "all" 全部 / "bars" 最近 N 根 / "range" 指定时间段
    _hmode = str(body.get("history_mode", "all"))
    history_bars = None
    date_from = date_to = None
    if _hmode == "bars":
        try:
            history_bars = max(50, int(body.get("history_bars", 500)))
        except (TypeError, ValueError):
            history_bars = 500
    elif _hmode == "range":
        date_from = (body.get("date_from") or "").strip() or None
        date_to = (body.get("date_to") or "").strip() or None

    _period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "2y", "1d": "max", "1wk": "max", "1mo": "max",
    }
    dl_period = _period_map.get(interval, "2y")

    loop = asyncio.get_event_loop()

    def _run():
        sym = symbol.upper()
        df = _slice_history(get_ohlcv(sym, period=dl_period, interval=interval, market=market),
                            history_bars, date_from, date_to)
        if df is None or len(df) < 30:
            raise HTTPException(400, f"{sym} 数据不足")

        # 使用个股标准参数（优先使用 buy 独立参数）
        params = _m._get_standard_strategy_params(sym, strategy_name, interval=interval, signal_type="buy")

        result = analyze_buy_points(df, strategy_name,
                                    params_override=params,
                                    window=window,
                                    entry_mode=entry_mode,
                                    judge_mode=judge_mode,
                                    up_pct=mfe_target, dn_pct=mae_limit,
                                    same_bar=same_bar)

        for bp in result["buy_points"]:
            if judge_mode != "first_touch":
                bp["success"] = (bp["mfe"] >= mfe_target and abs(bp["mae"]) <= mae_limit)

        if result["buy_points"]:
            success_count = sum(1 for bp in result["buy_points"] if bp["success"])
            result["success_rate"] = round(success_count / len(result["buy_points"]) * 100, 1)
        else:
            result["success_rate"] = 0

        result["symbol"] = sym
        result["strategy"] = strategy_name
        result["window"] = window
        result["mfe_target"] = mfe_target
        result["mae_limit"] = mae_limit
        # 口径回显: 不写出来的话, 看着一份数据不知道它是按哪段历史、哪套判定算的
        result["judge_mode"] = judge_mode
        result["same_bar"] = same_bar
        result["history_bars"] = history_bars
        result["date_from"] = date_from
        result["date_to"] = date_to
        return result

    return await loop.run_in_executor(None, _run)
