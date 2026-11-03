"""
Data router: /data/*, /data/bulk-update, /data/config, /data/status,
             /data/market-universe, /data/fundamentals, /data/daily-sync,
             /data/sync-configured, /data/refresh, /data/update
"""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
import threading
import time

import main as _m

router = APIRouter()


@router.post("/data/bulk-update")
def data_bulk_update(body: dict):
    """批量下载/增量更新股票池行情数据（后台线程执行，返回task_id用于轮询进度）"""
    import json
    from pathlib import Path
    from backtest import get_ohlcv

    pool_name = body.get("pool", "MYPOOL")
    interval = body.get("interval", "1d")
    pool_path = Path(__file__).parent.parent / "stock_pools.json"
    with open(pool_path, encoding="utf-8") as f:
        pools = json.load(f)
    pool_info = pools.get(pool_name)
    if not pool_info:
        raise HTTPException(404, f"Pool '{pool_name}' not found")
    symbols = pool_info["symbols"]
    market = pool_info.get("market", "US")

    task_id = f"{pool_name}_{interval}_{int(time.time())}"
    _m.register_job(_m._bulk_progress, task_id, {
        "total": len(symbols), "done": 0, "success": 0, "failed": 0,
        "current": "", "status": "running", "details": _m.LockedDict()
    })

    def _run():
        prog = _m._bulk_progress[task_id]
        try:
            for i, sym in enumerate(symbols):
                prog["current"] = sym
                ok = False
                last_err = ""
                for attempt in range(3):
                    try:
                        df = get_ohlcv(sym, market=market, interval=interval, period="max")
                        rows = len(df) if df is not None and not df.empty else 0
                        prog["details"][sym] = rows
                        if rows > 0:
                            ok = True
                            break
                        else:
                            last_err = "返回空数据"
                    except Exception as e:
                        last_err = str(e)[:100]
                        _m.logger.warning(f"[bulk] {sym} attempt {attempt+1}: {e}")
                        if attempt < 2:
                            time.sleep(2)
                prog["done"] = i + 1
                if ok:
                    prog["success"] += 1
                else:
                    prog["failed"] += 1
                    prog["details"][sym] = f"❌ {last_err}"
                    prog.setdefault("errors", []).append(f"{sym}: {last_err}")
                if i < len(symbols) - 1:
                    time.sleep(0.5)
            prog["status"] = "done"
            prog["current"] = ""
        except Exception as e:
            prog["status"] = "error"
            prog["current"] = f"线程异常: {e}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"task_id": task_id, "total": len(symbols)}


@router.get("/data/config")
def data_config():
    import json
    cfg = _m._load_market_data_config()
    market_universe_meta = {}
    for market in _m._configured_market_codes():
        fp = _m._market_universe_file(market)
        if not fp.exists():
            market_universe_meta[market] = {"count": 0, "updated_at": None}
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            market_universe_meta[market] = {
                "count": len(data.get("symbols", []) or []),
                "updated_at": data.get("updated_at"),
            }
        except Exception:
            market_universe_meta[market] = {"count": 0, "updated_at": None}
    return {
        **cfg,
        "supported_markets": _m._configured_market_codes(),
        "supported_intervals": list(_m.PARAM_INTERVALS),
        "market_universes": market_universe_meta,
        "pools": {**cfg.get("pools", {}), **{
            name: {**cfg.get("pools", {}).get(name, {}), "market": info.get("market", "US"), "symbols_count": len(info.get("symbols", []))}
            for name, info in _m._load_pools().items()
        }},
    }


@router.post("/data/config")
def save_data_config(body: dict = None):
    if body is None:
        body = {}
    current = _m._load_market_data_config()
    merged = _m._merge_market_data_config({**current, **body})
    _m._save_market_data_config(merged)
    return {"ok": True, "updated_at": merged.get("updated_at"), "config": merged}


@router.get("/data/bulk-progress/{task_id}")
async def data_bulk_progress(task_id: str):
    """查询批量下载进度"""
    prog = _m._bulk_progress.get(task_id)
    if not prog:
        raise HTTPException(404, "task not found")
    # drop details: 前端这几个面板只读计数器(total/done/success/failed/current/
    # status), 从不读 details。而多市场同步时它有上万条, 每 1.5 秒轮询一次就是
    # 近 1MB 的序列化 + 传输, 全压在事件循环上, 到前端又被直接丢掉。
    # 留 details_count 以便需要时看出规模。
    return _m.job_json_response(prog, drop=("details",))


@router.get("/data/status")
def data_status():
    """查看本地行情数据状态"""
    import pandas as pd
    from backtest import _DATA_DIR
    status = {}
    for mkt_dir in sorted(_DATA_DIR.iterdir()):
        if not mkt_dir.is_dir():
            continue
        market = mkt_dir.name
        files = sorted(mkt_dir.glob("*.parquet"))
        stocks = {}
        for f in files:
            name = f.stem
            parts = name.rsplit("_", 1)
            sym = parts[0]
            interval = parts[1] if len(parts) > 1 else "1d"
            key = name
            try:
                df = pd.read_parquet(f)
                stocks[key] = {
                    "symbol": sym,
                    "interval": interval,
                    "bars": len(df),
                    "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()),
                    "file_size_kb": round(f.stat().st_size / 1024, 1),
                    "updated": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            except Exception:
                stocks[key] = {"symbol": sym, "interval": interval, "error": "read failed"}
        status[market] = {"count": len(stocks), "stocks": stocks}
    return status


@router.get("/data/daily-universe")
def data_daily_universe(markets: str = Query("", description="逗号分隔市场，如 US,CN,HK,UK")):
    market_list = [m.strip().upper() for m in markets.split(",") if m.strip()] if markets else []
    items = _m._configured_market_universe(markets=market_list or None)
    grouped = {}
    for item in items:
        grouped.setdefault(item["market"], []).append(item["symbol"])
    return {
        "markets": market_list or sorted(grouped.keys()),
        "counts": {m: len(v) for m, v in grouped.items()},
        "total": len(items),
        "symbols": grouped,
    }


@router.get("/data/market-universe/{market}")
def data_market_universe(market: str, refresh: bool = Query(False), limit: int = Query(500, ge=1, le=5000)):
    data = _m._load_market_universe(market.upper(), refresh=refresh)
    symbols = data.get("symbols", []) or []
    return {
        "market": market.upper(),
        "count": len(symbols),
        "updated_at": data.get("updated_at"),
        "symbols": symbols[:limit],
        "truncated": len(symbols) > limit,
    }


@router.get("/data/fundamentals/status")
def data_fundamentals_status(scope: str = Query("pool"), market: str = Query(None), pool: str = Query(None)):
    return _m._fundamental_scope_status(scope=scope, market=market, pool=pool)


@router.post("/data/fundamentals/sync/start")
def data_fundamentals_sync_start(body: dict = None):
    if body is None:
        body = {}
    scope = str(body.get("scope", "pool")).lower()
    target_market = str(body.get("market", "")).upper() or None
    target_pool = body.get("pool") or None
    refresh = bool(body.get("refresh", False))

    if scope == "market":
        if not target_market:
            raise HTTPException(400, "market is required when scope=market")
        items = [i for i in _m._configured_market_universe(markets=[target_market]) if i["market"] != "CRYPTO"]
    elif scope == "pool":
        if not target_pool:
            raise HTTPException(400, "pool is required when scope=pool")
        items = [i for i in _m._configured_market_universe(pools=[target_pool]) if i["market"] != "CRYPTO"]
    else:
        raise HTTPException(400, "scope must be market or pool")

    task_id = f"fund_{scope}_{int(time.time())}"
    _m.register_job(_m._fundamental_sync_jobs, task_id, {
        "task_id": task_id,
        "status": "running",
        "scope": scope,
        "market": target_market,
        "pool": target_pool,
        "total": len(items),
        "done": 0,
        "success": 0,
        "skipped": 0,      # 24h 内的缓存命中。以前算进 success, 于是
                           # 「更新 91」里绝大多数其实一个字节都没下。
        "failed": 0,
        "current": "",
        "details": _m.LockedDict(),
        "started_at": datetime.utcnow().isoformat(),
    })

    def _run():
        prog = _m._fundamental_sync_jobs[task_id]
        try:
            _sync_fundamentals_loop(prog, items, refresh)
        except Exception as e:
            # 同文件另外三个 _run 都有这层兜底, 唯独这个没有。少了它,
            # 一个畸形 items 条目(KeyError)或 exists()/stat() 之间被删掉的
            # 文件(FileNotFoundError)就会让线程直接死掉, status 永远停在
            # running, 前端转圈转到天荒地老, 而 job 又从不回收。
            _m.logger.exception(f"[fundamentals-sync] task={task_id} 异常终止")
            prog.update({"status": "error", "error": str(e)[:300],
                         "current": "",
                         "finished_at": datetime.utcnow().isoformat()})

    def _sync_fundamentals_loop(prog, items, refresh):
        for idx, item in enumerate(items, start=1):
            sym = item["symbol"]
            mkt = item["market"]
            key = f"{mkt}:{sym}"
            prog["current"] = key
            fp = _m._fundamental_path(sym, mkt)
            if fp.exists() and not refresh:
                age_hours = (time.time() - fp.stat().st_mtime) / 3600
                if age_hours <= 24:
                    prog["skipped"] += 1
                    prog["done"] = idx
                    prog["details"][key] = {"status": "cached"}
                    continue
            try:
                snapshot = _m._fetch_fundamental_snapshot(sym, mkt)
                _m._save_fundamental_snapshot(sym, mkt, snapshot)
                prog["success"] += 1
                prog["details"][key] = {"status": "ok", "name": snapshot.get("name")}
            except Exception as e:
                prog["failed"] += 1
                prog["details"][key] = {"status": "error", "error": str(e)[:120]}
            prog["done"] = idx
        prog["status"] = "done"
        prog["current"] = ""
        prog["finished_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "total": len(items)}


@router.get("/data/fundamentals/sync/status/{task_id}")
async def data_fundamentals_sync_status(task_id: str):
    prog = _m._fundamental_sync_jobs.get(task_id)
    if not prog:
        raise HTTPException(404, "task not found")
    # drop details: 前端这几个面板只读计数器(total/done/success/failed/current/
    # status), 从不读 details。而多市场同步时它有上万条, 每 1.5 秒轮询一次就是
    # 近 1MB 的序列化 + 传输, 全压在事件循环上, 到前端又被直接丢掉。
    # 留 details_count 以便需要时看出规模。
    return _m.job_json_response(prog, drop=("details",))


@router.post("/data/daily-sync/start")
def data_daily_sync_start(body: dict = None):
    """补齐并每日更新本地日线数据，目标是每个市场至少有1年日K。"""
    from backtest import get_ohlcv
    from concurrent.futures import Future

    if body is None:
        body = {}
    markets = [str(m).upper() for m in body.get("markets", _m._configured_market_codes())]
    pools = body.get("pools") or None
    min_bars = int(body.get("min_bars", 200))
    refresh_universe = bool(body.get("refresh_universe", True))

    initial_total = 0
    if pools:
        initial_total = len(_m._configured_market_universe(markets=markets, pools=pools))
    else:
        for market in markets:
            if _m._market_universe_file(market).exists():
                try:
                    initial_total += len(_m._load_market_universe(market, refresh=False).get("symbols", []) or [])
                except Exception:
                    pass

    task_id = f"daily_{int(time.time())}"
    _m.register_job(_m._daily_sync_progress, task_id, {
        "task_id": task_id,
        "status": "preparing",
        "markets": markets,
        "total": initial_total,
        "done": 0,
        "success": 0,
        "failed": 0,
        "current": "准备日线同步标的列表...",
        "min_bars": min_bars,
        "details": _m.LockedDict(),
        "started_at": datetime.utcnow().isoformat(),
    })

    def _run():
        prog = _m._daily_sync_progress[task_id]
        try:
            if not pools and refresh_universe:
                for market in markets:
                    prog["current"] = f"刷新 {market} 市场主列表..."
                    if _m._market_universe_needs_refresh(market):
                        _m._load_market_universe(market, refresh=True)
            universe = _m._configured_market_universe(markets=markets, pools=pools)
            if not universe:
                prog["status"] = "error"
                prog["current"] = "没有可同步的股票，请先配置对应市场或股票池"
                prog["finished_at"] = datetime.utcnow().isoformat()
                return
            prog["total"] = len(universe)
            prog["status"] = "running"
        except Exception as e:
            prog["status"] = "error"
            prog["current"] = f"准备失败: {str(e)[:120]}"
            prog["finished_at"] = datetime.utcnow().isoformat()
            return

        for idx, item in enumerate(universe, start=1):
            sym = item["symbol"]
            market = item["market"]
            prog["current"] = f"{market}:{sym}"
            try:
                df = _m._run_with_timeout(
                    get_ohlcv,
                    _m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC if not pools else max(30, _m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC),
                    sym,
                    market=market,
                    interval="1d",
                    period="1y",
                    include_today=True,
                )
                bars = len(df) if df is not None else 0
                status = _m._one_year_daily_status(sym, market, min_bars=min_bars)
                prog["details"][f"{market}:{sym}"] = {"bars": bars, **status}
                if status.get("ok"):
                    prog["success"] += 1
                else:
                    prog["failed"] += 1
            except _m.FutureTimeoutError:
                prog["failed"] += 1
                prog["details"][f"{market}:{sym}"] = {"ok": False, "reason": f"timeout>{_m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC}s"}
            except Exception as e:
                prog["failed"] += 1
                prog["details"][f"{market}:{sym}"] = {"ok": False, "reason": str(e)[:100]}
            prog["done"] = idx
        prog["status"] = "done"
        prog["current"] = ""
        prog["finished_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "total": initial_total, "markets": markets, "refresh_universe": refresh_universe}


@router.get("/data/daily-sync/status/{task_id}")
async def data_daily_sync_status(task_id: str):
    prog = _m._daily_sync_progress.get(task_id)
    if not prog:
        raise HTTPException(404, "task not found")
    # drop details: 前端这几个面板只读计数器(total/done/success/failed/current/
    # status), 从不读 details。而多市场同步时它有上万条, 每 1.5 秒轮询一次就是
    # 近 1MB 的序列化 + 传输, 全压在事件循环上, 到前端又被直接丢掉。
    # 留 details_count 以便需要时看出规模。
    return _m.job_json_response(prog, drop=("details",))


@router.post("/data/sync-configured/start")
def data_sync_configured_start(body: dict = None):
    """按市场规则或股票池规则增量同步本地数据。"""
    from backtest import get_ohlcv

    if body is None:
        body = {}
    scope = str(body.get("scope", "market")).lower()
    usage = str(body.get("usage", "selector")).lower()
    interval = str(body.get("interval", "1d"))
    target_market = str(body.get("market", "")).upper() or None
    target_pool = body.get("pool") or None
    custom_period = body.get("period")
    custom_symbols = [str(s).upper() for s in body.get("symbols", [])]

    pools = _m._load_pools()
    if scope == "market":
        if not target_market:
            raise HTTPException(400, "market is required when scope=market")
        period = custom_period or _m._configured_period_for_market(target_market, interval=interval)
        total_hint = (_m._load_market_universe(target_market, refresh=False).get("count", 0)
                      if _m._market_universe_file(target_market).exists() else 0)
    elif scope == "pool":
        if not target_pool or target_pool not in pools:
            raise HTTPException(400, "valid pool is required when scope=pool")
        pool = pools[target_pool]
        period = custom_period or _m._configured_period_for_pool(target_pool, interval=interval, usage=usage)
        total_hint = len(custom_symbols or [str(s).upper() for s in pool.get("symbols", [])])
    else:
        raise HTTPException(400, "scope must be market or pool")

    task_id = f"synccfg_{scope}_{int(time.time())}"
    _m.register_job(_m._data_sync_jobs, task_id, {
        "task_id": task_id,
        "status": "preparing",
        "scope": scope,
        "usage": usage,
        "market": target_market,
        "pool": target_pool,
        "interval": interval,
        "period": period,
        "total": total_hint,
        "done": 0,
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "current": "准备同步标的列表...",
        "details": _m.LockedDict(),
        "started_at": datetime.utcnow().isoformat(),
    })

    def _run():
        prog = _m._data_sync_jobs[task_id]
        tasks = []

        def _should_exclude_failed_symbol(market_: str, detail: str) -> bool:
            if str(market_).upper() != "US":
                return False
            text = str(detail or "").lower()
            return any(key in text for key in [
                "no security definition",
                "unknown contract",
                "possibly delisted",
                "no timezone found",
                "historical market data service error",
            ])

        try:
            if scope == "market":
                prog["current"] = f"加载 {target_market} 全市场主列表..."
                need_refresh = _m._market_universe_needs_refresh(target_market)
                universe_data = _m._load_market_universe(target_market, refresh=False)
                if not universe_data.get("symbols") or need_refresh:
                    prog["current"] = f"重建 {target_market} 全市场主列表..."
                    universe_data = _m._load_market_universe(target_market, refresh=True)
                universe = [
                    {"symbol": str(row.get("symbol", "")).upper(), "market": target_market, "pool": "__market__"}
                    for row in (universe_data.get("symbols", []) or [])
                    if str(row.get("symbol", "")).strip()
                ]
                for item in universe:
                    tasks.append({"symbol": item["symbol"], "market": item["market"], "interval": interval, "period": period})
            else:
                pool = pools[target_pool]
                for sym in (custom_symbols or [str(s).upper() for s in pool.get("symbols", [])]):
                    tasks.append({"symbol": sym, "market": str(pool.get("market", "US")).upper(), "interval": interval, "period": period})

            if not tasks:
                prog["status"] = "error"
                prog["current"] = "没有可同步的标的"
                prog["finished_at"] = datetime.utcnow().isoformat()
                return

            prog["total"] = len(tasks)
            prog["status"] = "running"
        except Exception as e:
            prog["status"] = "error"
            prog["current"] = f"准备失败: {str(e)[:120]}"
            prog["finished_at"] = datetime.utcnow().isoformat()
            return

        for idx, item in enumerate(tasks, start=1):
            sym = item["symbol"]
            market = item["market"]
            interval_ = item["interval"]
            period_ = item["period"]
            key = f"{market}:{sym}:{interval_}"
            prog["current"] = key
            try:
                path = _m._get_data_path(sym, market, interval_)
                max_age_sec = _m._sync_file_max_age_sec(interval_)
                if path.exists():
                    age = time.time() - path.stat().st_mtime
                    if age < max_age_sec:
                        try:
                            df = get_ohlcv(
                                sym,
                                market=market,
                                interval=interval_,
                                period=period_,
                                max_age_sec=max_age_sec,
                                include_today=True,
                                local_only=True,
                            )
                        except Exception:
                            import pandas as pd
                            df = pd.DataFrame()
                        bars = len(df) if df is not None else 0
                        prog["details"][key] = {
                            "bars": bars,
                            "start": str(df.index[0]) if bars else None,
                            "end": str(df.index[-1]) if bars else None,
                            "status": "skipped",
                        }
                        prog["skipped"] += 1
                        prog["done"] = idx
                        continue
                timeout = _m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC if scope == "market" else max(30, _m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC)
                df = _m._run_with_timeout(
                    get_ohlcv,
                    timeout,
                    sym,
                    market=market,
                    interval=interval_,
                    period=period_,
                    max_age_sec=0,
                    include_today=True,
                )
                bars = len(df) if df is not None else 0
                prog["details"][key] = {
                    "bars": bars,
                    "start": str(df.index[0]) if bars else None,
                    "end": str(df.index[-1]) if bars else None,
                    "status": "updated",
                }
                if bars > 0:
                    prog["success"] += 1
                else:
                    prog["failed"] += 1
                    if scope == "market" and market == "US":
                        _m._mark_market_universe_excluded(market, sym, "empty_history")
            except _m.FutureTimeoutError:
                prog["failed"] += 1
                prog["details"][key] = {"error": f"timeout>{_m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC}s"}
                if scope == "market":
                    _m._mark_market_universe_excluded(market, sym, f"timeout>{_m.MARKET_SYNC_SYMBOL_TIMEOUT_SEC}s")
            except Exception as e:
                prog["failed"] += 1
                err = str(e)[:120]
                prog["details"][key] = {"error": err}
                if scope == "market" and _should_exclude_failed_symbol(market, err):
                    _m._mark_market_universe_excluded(market, sym, err)
            prog["done"] = idx
        prog["current"] = ""
        prog["status"] = "done"
        prog["finished_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "total": total_hint, "interval": interval, "period": period}


@router.get("/data/sync-configured/status/{task_id}")
async def data_sync_configured_status(task_id: str):
    prog = _m._data_sync_jobs.get(task_id)
    if not prog:
        raise HTTPException(404, "task not found")
    # drop details: 前端这几个面板只读计数器(total/done/success/failed/current/
    # status), 从不读 details。而多市场同步时它有上万条, 每 1.5 秒轮询一次就是
    # 近 1MB 的序列化 + 传输, 全压在事件循环上, 到前端又被直接丢掉。
    # 留 details_count 以便需要时看出规模。
    return _m.job_json_response(prog, drop=("details",))


@router.post("/data/refresh/{market}/{symbol}")
def data_refresh_one(market: str, symbol: str, interval: str = Query("1d")):
    """强制刷新单只股票当前周期的行情数据"""
    from backtest import get_ohlcv, init_stock_data, _MEM_CACHE
    market = market.upper()
    symbol = symbol.upper()
    path = _m._get_data_path(symbol, market, interval)
    cache_key = f"{market}_{symbol}_{interval}"
    try:
        _MEM_CACHE.pop(cache_key, None)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        df = init_stock_data(symbol, market, interval)
        if df is None or df.empty:
            _MEM_CACHE.pop(cache_key, None)
            df = get_ohlcv(symbol, market=market, interval=interval, period="max", max_age_sec=0)
        # 清除该股票的回测结果缓存，确保下次回测使用新数据
        _m.cache_invalidate_symbol(symbol, market)
        rows = len(df) if df is not None else 0
        updated_at = None
        if path.exists():
            updated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "ok": rows > 0,
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "rows": rows,
            "updated_at": updated_at,
        }
    except Exception as e:
        raise HTTPException(500, f"refresh failed: {e}")


@router.post("/data/update/{market}/{symbol}")
def data_update_one(market: str, symbol: str, interval: str = Query("1d")):
    """增量更新单只股票当前周期的行情数据"""
    from backtest import init_stock_data, update_stock_data, _MEM_CACHE
    market = market.upper()
    symbol = symbol.upper()
    path = _m._get_data_path(symbol, market, interval)
    cache_key = f"{market}_{symbol}_{interval}"
    try:
        _MEM_CACHE.pop(cache_key, None)
        if path.exists():
            df = update_stock_data(symbol, market, interval)
        else:
            df = init_stock_data(symbol, market, interval)
        # 清除该股票的回测结果缓存，确保下次回测使用新数据
        _m.cache_invalidate_symbol(symbol, market)
        rows = len(df) if df is not None else 0
        updated_at = None
        if path.exists():
            updated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "ok": rows > 0,
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "rows": rows,
            "updated_at": updated_at,
        }
    except Exception as e:
        raise HTTPException(500, f"update failed: {e}")
