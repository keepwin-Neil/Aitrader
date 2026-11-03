"""
Scan/selector router: /scan/*, /selector/*
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime
from typing import Optional
import json
import threading
from pathlib import Path

# scan_center / scan_audit 不 import main, 模块顶层导入安全。
import scan_center
import scan_audit
from admin_auth import require_admin

import main as _m

router = APIRouter()


@router.post("/scan/realtime")
def realtime_scan(body: dict = None):
    """
    扫描指定股票池，用每只股票排名前top_n的策略检测最新 BUY/SELL 信号（并行加速版）
    body: {
      "pools": ["MYPOOL", "US Tech"],
      "interval": "5m",
      "top_n": 6,
      "signal_side": "buy|sell|both",
      "markets": ["US"]   # 2026-04-25 — optional market filter; when set,
                          # the universe is filtered to symbols whose market
                          # is in this list (deterministic per-market scan)
    }
    """
    if body is None:
        body = {}
    pool_names = body.get("pools", [])
    if not pool_names and body.get("all_pools"):
        body["all_pools"] = True
    interval = body.get("interval", "5m")
    top_n = body.get("top_n", 0)
    strategy_names = [str(s).strip() for s in body.get("strategies", []) if str(s).strip()]
    llm_owners = [str(owner).strip() for owner in body.get("llm_owners", []) if str(owner).strip()]
    signal_side = str(body.get("signal_side", "buy")).strip().lower()
    if signal_side not in ("buy", "sell", "both"):
        signal_side = "buy"
    # 2026-04-25: per-market filter. When set, the resolved universe is
    # filtered to items whose market is in the list. Empty/missing = no
    # filter (legacy behavior). Markets are case-insensitive on input but
    # compared upper-case (item["market"] is already upper-case from the
    # configured-market resolver).
    markets_filter_raw = body.get("markets") or []
    if not isinstance(markets_filter_raw, list):
        markets_filter_raw = []
    markets_filter = {str(m).strip().upper() for m in markets_filter_raw if str(m).strip()}
    refresh_sec = body.get("refresh_sec")
    try:
        refresh_sec = int(refresh_sec) if refresh_sec is not None else None
    except Exception:
        refresh_sec = None
    if refresh_sec is not None:
        refresh_sec = max(0, refresh_sec)
    filters = _m._parse_selector_filters(body)
    items = _m._resolve_scan_universe(pool_names=pool_names, all_pools=body.get("all_pools", False), scope="pool")
    if markets_filter:
        # Filter to symbols in the requested markets. Item shape is
        # {"symbol": str, "market": str, "pool": str} from
        # _configured_market_universe / _resolve_scan_universe.
        items = [it for it in (items or []) if str(it.get("market") or "").upper() in markets_filter]
    if not items:
        return {"alerts": [], "error": "no pools",
                "scan_time": datetime.now().isoformat(), "stocks_scanned": 0, "strategies_used": 0,
                "markets_filter": sorted(markets_filter) if markets_filter else None}
    # 2026-04-26: cycle source uses underscore-suffixed market name when a
    # single-market filter is active so the audit cycle id is parseable
    # ("manual_us-<ts>-<hex8>"). When emit_audit_cycle is set, the UI can
    # group by market just like the daemon's tick_us cycles.
    if len(markets_filter) == 1:
        cid_source = f"manual_{next(iter(markets_filter)).lower()}"
    else:
        cid_source = "manual"
    cid = scan_center.new_cycle_id_safe(cid_source)

    # 2026-04-26 (revised): audit emission is now opt-in via
    # ``emit_audit_cycle`` to stop the legacy "实时监测" page + monitor-box
    # auto-poll timers (which call /scan/realtime every refresh_sec) from
    # spamming the audit ring with manual-… cycles. Only the scan-monitor
    # page should request audit emission, and as of this commit it does so
    # via /scan/monitor/run-full-pipeline (which routes through
    # _process_monitor_groups for full Stage 2-4). /scan/realtime stays as
    # a pure raw-scan endpoint by default.
    emit_audit_cycle = bool(body.get("emit_audit_cycle", False))

    if emit_audit_cycle:
        try:
            symbols_only = sorted({str(it.get("symbol")) for it in items if it.get("symbol")})
            by_mkt: dict = {}
            for it in items:
                m = str(it.get("market") or "").upper()
                if m and it.get("symbol"):
                    by_mkt.setdefault(m, []).append(str(it["symbol"]))
            scan_audit.log_stage(cid, "universe", {
                "symbols": symbols_only,
                "by_interval": {interval: symbols_only},
                "by_market": {k: sorted(v) for k, v in by_mkt.items()},
                "buy_pool": {k: sorted(v) for k, v in by_mkt.items()},
                "sell_pool": {},
                "manual_scan": True,
                "markets_filter": sorted(markets_filter) if markets_filter else None,
            })
        except Exception:  # noqa: BLE001
            pass

    result = _m._run_realtime_scan(
        items=items,
        interval=interval,
        top_n=top_n,
        strategy_names=strategy_names,
        llm_owners=llm_owners,
        signal_side=signal_side,
        refresh_sec=refresh_sec,
        filters=filters,
        cycle_id=cid,
    )

    if emit_audit_cycle:
        try:
            alerts = (result or {}).get("alerts") or []
            for a in alerts:
                try:
                    scan_audit.log_stage(cid, "raw_signal", a)
                except Exception:  # noqa: BLE001
                    pass
            bucket = (
                f"{next(iter(markets_filter))}/{interval}/{signal_side}"
                if len(markets_filter) == 1
                else f"{interval}/{signal_side}"
            )
            scan_audit.log_stage(cid, "scan_summary", {
                "bucket": bucket,
                "interval": interval,
                "side": signal_side,
                "market": (next(iter(markets_filter)) if len(markets_filter) == 1 else None),
                "path": "manual_scan_realtime",
                "groups": 1,
                "universe_size": len(items),
                "strategies_count": int((result or {}).get("strategies_used") or 0),
                "alerts_emitted": len(alerts),
                "error": None,
            })
        except Exception:  # noqa: BLE001
            pass

    # Stamp the cycle_id into the response so callers (frontend) can
    # link directly to the cycle in the monitoring UI.
    if isinstance(result, dict):
        result.setdefault("cycle_id", cid)
        if markets_filter:
            result.setdefault("markets_filter", sorted(markets_filter))
    return result


@router.post("/selector/realtime")
def selector_realtime(body: dict = None):
    """选股服务别名：支持 BUY / SELL / BOTH，并可扫描全部已配置市场股票池。"""
    if body is None:
        body = {}
    if "all_pools" not in body and not body.get("pools"):
        body["all_pools"] = True
    body["local_only"] = True
    return realtime_scan(body)


@router.post("/selector/jobs/start")
def selector_job_start(body: dict = None):
    """后台选股服务：支持按市场或股票池发起 BUY/SELL/BOTH 扫描任务。"""
    if body is None:
        body = {}
    scope = str(body.get("scope", "market")).lower()
    interval = str(body.get("interval", "1d"))
    top_n = int(body.get("top_n", 0) or 0)
    strategy_names = [str(s).strip() for s in body.get("strategies", []) if str(s).strip()]
    llm_owners = [str(owner).strip() for owner in body.get("llm_owners", []) if str(owner).strip()]
    signal_side = str(body.get("signal_side", "buy")).strip().lower()
    local_only = bool(body.get("local_only", True))
    filters = _m._parse_selector_filters(body)
    refresh_sec = body.get("refresh_sec")
    try:
        refresh_sec = int(refresh_sec) if refresh_sec is not None else None
    except Exception:
        refresh_sec = None
    markets = [str(m).upper() for m in body.get("markets", []) if str(m).strip()]
    pool_names = [str(p) for p in body.get("pools", []) if str(p).strip()]

    items = _m._resolve_scan_universe(
        pool_names=pool_names,
        all_pools=bool(body.get("all_pools", False)),
        markets=markets,
        scope=scope,
    )
    if not items:
        raise HTTPException(400, "没有可扫描的标的")

    import time
    task_id = f"selector_{int(time.time())}"
    _m.register_job(_m._selector_jobs, task_id, {
        "task_id": task_id,
        "status": "running",
        "scope": scope,
        "markets": sorted({i["market"] for i in items}),
        "pools": sorted({i.get("pool") for i in items if i.get("pool") and i.get("pool") != "__market__"}),
        "interval": interval,
        "top_n": top_n,
        "strategies": strategy_names,
        "llm_owners": llm_owners,
        "signal_side": signal_side,
        "local_only": local_only,
        "refresh_sec": refresh_sec,
        "filters": filters,
        "total_symbols": len(items),
        "progress_stage": "preparing",
        "progress_done": 0,
        "progress_total": len(items),
        "current": "准备选股任务...",
        "alerts": [],
        "errors": [],
        "started_at": datetime.utcnow().isoformat(),
    })

    def _progress(stage, done, total, current):
        job = _m._selector_jobs[task_id]
        job["progress_stage"] = stage
        job["progress_done"] = done
        job["progress_total"] = total
        job["current"] = current

    def _run():
        job = _m._selector_jobs[task_id]
        # v0.5 §7.2.1 line 1648: background selector job is also a "manual"
        # cycle source — generate cycle_id at the job-start boundary.
        cid = scan_center.new_cycle_id_safe("manual")
        try:
            result = _m._run_realtime_scan(
                items=items,
                interval=interval,
                top_n=top_n,
                strategy_names=strategy_names,
                llm_owners=llm_owners,
                signal_side=signal_side,
                refresh_sec=refresh_sec,
                local_only=local_only,
                filters=filters,
                progress_hook=_progress,
                cycle_id=cid,
            )
            job.update(result)
            job["status"] = "done"
            job["finished_at"] = datetime.utcnow().isoformat()
            job["current"] = ""
        except Exception as e:
            job["status"] = "error"
            job["current"] = str(e)[:160]
            job["finished_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return {
        "task_id": task_id,
        "status": "running",
        "total_symbols": len(items),
        "scope": scope,
        "markets": sorted({i["market"] for i in items}),
    }


@router.get("/selector/jobs/status/{task_id}")
async def selector_job_status(task_id: str):
    job = _m._selector_jobs.get(task_id)
    if not job:
        raise HTTPException(404, "task not found")
    # 轮询阶段前端只用 alerts.length(真数据走 /selector/jobs/result), 而这里
    # 每 2.5 秒把整个 alerts 数组序列化一遍 —— 按 1336 条 × ~150B 算约 200KB/次。
    # 同文件其它几个进度接口都传了 drop, 只有这个漏了。job_json_response 会
    # 自动补一个 alerts_count。
    return _m.job_json_response(job, drop=("alerts", "details"))


@router.get("/selector/jobs/result/{task_id}")
async def selector_job_result(task_id: str):
    job = _m._selector_jobs.get(task_id)
    if not job:
        raise HTTPException(404, "task not found")
    # 先取一次快照再拼: 下面 16 个 job.get() 是分开执行的, 后台线程的
    # job.update(result) 和紧随其后的 job["status"]="done" 是两次写, 读侧
    # 夹在中间就会拿到**两代混合**的结果 —— 比如 status 已经 done 但 alerts
    # 还是空的, 前端渲染出"扫描完成, 0 条结果"。
    # (同文件上面的 selector_job_status 早就是这么写的, 这里漏了。)
    job = _m.snapshot_of(job)
    return _m.job_json_response({
        "task_id": task_id,
        "status": job.get("status"),
        "alerts": job.get("alerts", []),
        "errors": job.get("errors", []),
        "scan_time": job.get("scan_time"),
        "stocks_scanned": job.get("stocks_scanned", 0),
        "signal_side": job.get("signal_side"),
        "strategies_desc": job.get("strategies_desc"),
        "markets": job.get("markets", []),
        "pools": job.get("pools", []),
        "filters": job.get("filters", {}),
        "filters_applied": job.get("filters_applied", {}),
        "total_symbols_before_filter": job.get("total_symbols_before_filter", job.get("total_symbols", 0)),
        "candidates_after_filter": job.get("candidates_after_filter", job.get("total_symbols", 0)),
        "filtered_out": job.get("filtered_out", 0),
        "filter_reasons": job.get("filter_reasons", {}),
    })


@router.get("/scan/signals-log")
def scan_signals_log(limit: int = 200, symbol: str = None, date: str = None):
    """查询历史扫描信号日志。"""
    log_path = Path("scan_signals.json")
    if not log_path.exists():
        return {"logs": [], "total": 0}
    try:
        logs = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return {"logs": [], "total": 0}
    if symbol:
        logs = [l for l in logs if l.get("symbol", "").upper() == symbol.upper()]
    if date:
        logs = [l for l in logs if l.get("scan_time", "").startswith(date)]
    total = len(logs)
    return {"logs": logs[-limit:], "total": total}


@router.post(
    "/scan/cache/signal/purge",
    dependencies=[Depends(require_admin)],
)
def scan_cache_signal_purge(body: dict = None):
    """Purge SignalCache entries across per-interval partitions (§5.2.6 / P7).

    Body:
        ``{}``                                  → purge every interval + all cycles
        ``{"interval": "1d"}``                  → purge only the 1d cache
        ``{"cycle_id": "tick-..."}``            → purge that cycle across all intervals
        ``{"interval": "1d", "cycle_id": "x"}`` → purge that cycle in the 1d cache only

    Returns ``{"removed_by_interval": {interval: N, ...}, "total": N}``.

    ``cycle_id`` must be omitted or a non-empty (post-trim) string; empty or
    whitespace-only cycle_id returns HTTP 400 to prevent an ops typo from
    silently falling through to purge-all. ``interval`` is similarly validated
    for whitespace-only strings. Non-existent intervals are a soft no-op.

    Admin-gated: when ``AITRADER_ADMIN_TOKEN`` env is set, requires a matching
    ``X-Admin-Token`` header; 401 otherwise. Env unset → gate is a no-op.
    """
    if body is None:
        body = {}
    cycle_id_raw = body.get("cycle_id")
    if cycle_id_raw is not None:
        cycle_id_str = str(cycle_id_raw).strip()
        if not cycle_id_str:
            raise HTTPException(
                400,
                "cycle_id must be non-empty string; omit key to purge all cycles",
            )
        cycle_id = cycle_id_str
    else:
        cycle_id = None

    interval_raw = body.get("interval")
    if interval_raw is not None:
        interval_str = str(interval_raw).strip()
        if not interval_str:
            raise HTTPException(
                400,
                "interval must be non-empty string; omit key to purge all intervals",
            )
        interval = interval_str
    else:
        interval = None

    removed_map = scan_center.purge_signal_cache(cycle_id=cycle_id, interval=interval)
    total = sum(removed_map.values())
    return {
        "removed_by_interval": removed_map,
        "total": int(total),
        "cycle_id": cycle_id,
        "interval": interval,
    }


@router.get("/scan/cache/stats")
def scan_cache_stats():
    """Operational stats for all Scan Center caches (§5.2.6 / P7 + §7.5 / P12).

    Returns a snapshot of:
      * ``signal``  — per-interval SignalCache stats + current ``size`` + ``cycles``
      * ``audit_flush`` — scan_audit durability counters
        (``flushed_segments`` / ``flush_errors`` / ``corrupt_recovered``)
    """
    return {
        "signal": scan_center.snapshot_signal_caches(),
        "audit_flush": scan_audit.snapshot_audit_flush_stats(),
    }


@router.get("/scan/market-status")
def market_status():
    """检查美股市场是否在交易时间"""
    import pytz
    now_et = datetime.now(pytz.timezone("US/Eastern"))
    weekday = now_et.weekday()
    hour = now_et.hour
    minute = now_et.minute
    t = hour * 60 + minute

    is_open = weekday < 5 and 570 <= t <= 960
    is_premarket = weekday < 5 and 240 <= t < 570
    is_afterhours = weekday < 5 and 960 < t <= 1200

    return {
        "is_open": is_open,
        "is_premarket": is_premarket,
        "is_afterhours": is_afterhours,
        "eastern_time": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "weekday": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][weekday],
    }


# ── 📰 News-driven daily report (Phase 3) ────────────────────────────────────
@router.get("/news/daily_report")
def news_daily_report(market: str = "ALL", window_hours: int = 24, top_n: int = 10):
    """Generate news-driven daily report from scan_signals.json.

    Reads alerts persisted by the selector pipeline; aggregates those carrying
    a `news` payload (added by Phase 2 _selector_score_news). Returns markdown
    plus stats.

    Query params:
      market:       ALL / US / HK / CN / UK (default ALL)
      window_hours: lookback window 1-168 (default 24)
      top_n:        max stocks to detail (default 10, max 50)
    """
    market = (market or "ALL").upper()
    window_hours = max(1, min(int(window_hours or 24), 168))
    top_n = max(1, min(int(top_n or 10), 50))

    try:
        from news_daily_report import generate_daily_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"news_daily_report module unavailable: {e}")

    try:
        return generate_daily_report(market=market, window_hours=window_hours, top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"report generation failed: {e}")


# scan_signals.json rolling buffer, configurable via news_buffer_config.json.
# Owns max_days/max_records; auto_scan and auto_pools blocks live in the same
# file but are owned by news_auto_scanner / news_pools respectively.

NEWS_BUFFER_CONFIG_PATH = Path(__file__).parent.parent / "news_buffer_config.json"


def _read_news_buffer_config() -> dict:
    """Read max_days/max_records from the shared config file (delegates to news_auto_scanner)."""
    from news_auto_scanner import get_config
    full = get_config()
    return {"max_days": int(full.get("max_days") or 7),
            "max_records": int(full.get("max_records") or 20000)}


@router.get("/news/buffer-status")
def news_buffer_status():
    """Current scan_signals.json buffer state — total records, oldest/newest scan_time, span."""
    import json as _json
    from news_daily_report import _parse_dt

    log_path = Path(__file__).parent.parent / "scan_signals.json"
    if not log_path.exists():
        return {"total_records": 0, "oldest": None, "newest": None, "span_hours": 0.0,
                "config": _read_news_buffer_config(), "file_size_bytes": 0, "exists": False}
    try:
        data = _json.loads(log_path.read_text(encoding="utf-8")) or []
    except Exception as e:
        raise HTTPException(500, f"buffer read failed: {e}")

    times = sorted(t for t in (a.get("scan_time") or a.get("time") for a in data) if t)
    oldest = times[0] if times else None
    newest = times[-1] if times else None
    t0 = _parse_dt(oldest) if oldest else None
    t1 = _parse_dt(newest) if newest else None
    span_h = round(max(0.0, (t1 - t0).total_seconds() / 3600.0), 2) if (t0 and t1) else 0.0

    return {
        "total_records": len(data),
        "oldest": oldest,
        "newest": newest,
        "span_hours": span_h,
        "config": _read_news_buffer_config(),
        "file_size_bytes": log_path.stat().st_size,
        "exists": True,
    }


@router.post("/news/buffer-config")
def news_buffer_config(max_days: int = 7, max_records: int = 20000):
    """Update rolling-buffer config. Takes effect on next scan write."""
    from news_auto_scanner import update_root_config
    max_days = max(1, min(int(max_days), 90))
    max_records = max(1000, min(int(max_records), 500000))
    update_root_config(max_days=max_days, max_records=max_records)
    return {"saved": True, "config": {"max_days": max_days, "max_records": max_records},
            "note": "下次扫描写入 scan_signals.json 时生效"}


# ── 🤖 Auto-scan scheduler API (2026-04-27) ──────────────────────────────────
#
# 用户反馈："你要开自动扫描，设定扫描频率，必须在日新闻报告增加自动扫描，
# 这个默认是后台一直运行" — 周报概念依赖持续累积数据，没有自动调度永远
# 拿不到 7 天数据。news_auto_scanner.py 实现的是个后台线程定时调
# /selector/jobs/start，这里暴露 GET 状态 + POST 配置 给前端用。

@router.get("/news/auto-scan-status")
async def news_auto_scan_status():
    """Return current auto-scan config + runtime state."""
    try:
        from news_auto_scanner import get_config, is_running
    except Exception as e:
        raise HTTPException(500, f"news_auto_scanner unavailable: {e}")
    cfg = get_config()
    return {
        "running": is_running(),
        "auto_scan": cfg.get("auto_scan") or {},
    }


@router.post("/news/auto-scan-config")
def news_auto_scan_config(
    enabled: Optional[bool] = None,
    interval_minutes: Optional[int] = None,
    market: Optional[str] = None,
    news_mode: Optional[str] = None,
    interval_bar: Optional[str] = None,
    signal_side: Optional[str] = None,
):
    """Partial-update of auto-scan config. None = leave unchanged."""
    try:
        from news_auto_scanner import update_config
    except Exception as e:
        raise HTTPException(500, f"news_auto_scanner unavailable: {e}")

    updates = {}
    if enabled is not None:
        updates["enabled"] = bool(enabled)
    if interval_minutes is not None:
        updates["interval_minutes"] = max(5, min(int(interval_minutes), 1440))  # 5min ~ 24h
    if market is not None:
        updates["market"] = (market or "US").upper()
    if news_mode is not None:
        nm = (news_mode or "weight").lower()
        if nm not in ("off", "weight", "must"):
            raise HTTPException(400, "news_mode must be off / weight / must")
        updates["news_mode"] = nm
    if interval_bar is not None:
        updates["interval_bar"] = interval_bar
    if signal_side is not None:
        ss = (signal_side or "buy").lower()
        if ss not in ("buy", "sell"):
            raise HTTPException(400, "signal_side must be buy / sell")
        updates["signal_side"] = ss

    cfg = update_config(**updates)
    return {"saved": True, "auto_scan": cfg.get("auto_scan") or {},
            "note": "配置已保存。下一次 60s 调度循环醒来后立即生效。"}


@router.post("/news/auto-scan-trigger")
def news_auto_scan_trigger_now():
    """Force a one-off scan immediately (independent of schedule).
    Resets last_run_at to None so the loop fires on its next 60s tick.
    """
    try:
        from news_auto_scanner import update_config
    except Exception as e:
        raise HTTPException(500, f"news_auto_scanner unavailable: {e}")
    update_config(last_run_at=None, next_run_at=None)
    return {"triggered": True, "note": "下一次调度循环（≤ 60s）会立即触发一次扫描"}


# ── 📅 News stock pools (auto-populated by scoring, 2026-04-27) ──────────────
#
# 用户需求："建立每日新闻股票池，将打分通过的放入。建立每周股票池，从周一到周五，
# 打分通过的股票放入股票池。已经在股票池的，不再参与评分"
#
# 实现：news_pools 模块 + main.py:_selector_score_news 集成。两个 JSON 文件：
# news_pool_daily.json (每日重置) + news_pool_weekly.json (每周一重置)。
# score >= threshold (默认 70) 自动入池；已经在今日池的 symbol 在下次扫描时
# 跳过 LLM 调用直接复用缓存分数。

@router.get("/news/pools")
def news_get_pools():
    """Return current daily + weekly pools (auto-resets on date change)."""
    try:
        import news_pools
    except Exception as e:
        raise HTTPException(500, f"news_pools unavailable: {e}")
    daily, weekly = news_pools.get_pools()
    return {
        "daily": daily,
        "weekly": weekly,
        "threshold": news_pools.get_threshold(),
    }


@router.post("/news/pools/clear")
def news_clear_pool(which: str = Query(..., pattern="^(daily|weekly|both)$")):
    """Manually wipe a pool. which: daily / weekly / both"""
    try:
        import news_pools
    except Exception as e:
        raise HTTPException(500, f"news_pools unavailable: {e}")
    if which == "both":
        n_d = news_pools.clear_pool("daily")
        n_w = news_pools.clear_pool("weekly")
        return {"cleared": {"daily": n_d, "weekly": n_w}}
    n = news_pools.clear_pool(which)
    return {"cleared": {which: n}}


@router.post("/news/pools/threshold")
def news_set_pool_threshold(threshold: int = Query(..., ge=0, le=100)):
    """Set the score threshold (0-100) for pool inclusion."""
    try:
        import news_pools
    except Exception as e:
        raise HTTPException(500, f"news_pools unavailable: {e}")
    saved = news_pools.set_threshold(threshold)
    return {"saved": True, "threshold": saved}


# ── 📊 On-demand news scoring trigger (2026-04-27) ───────────────────────────
@router.post("/news/trigger-scoring")
def trigger_news_scoring(
    market: str = "ALL",
    window_hours: int = 24,
    max_symbols: int = 30,
    provider: str = "claude",
):
    """Score in-window alerts that lack news scores.

    Lets the user trigger LLM news scoring directly from the news report panel
    without going through 选股服务 → 高级筛选 → 📰 新闻热点驱动 → 跑一次完整选股。

    Pipeline (mirrors main.py:_selector_score_news):
      1. Read scan_signals.json
      2. Filter alerts: in window + market match + no existing news.score
      3. Dedupe symbols, cap at max_symbols (default 30)
      4. For each symbol → fetch news (Finnhub + Google) → LLM score → write back
      5. Save scan_signals.json

    Returns:
      {scored: N, skipped_no_alerts: 0, skipped_already_scored: N,
       capped: N, errors: [{symbol, error}, ...], duration_sec: float}
    """
    import json
    import time
    from datetime import datetime, timedelta
    from pathlib import Path

    market = (market or "ALL").upper()
    window_hours = max(1, min(int(window_hours or 24), 168))
    max_symbols = max(1, min(int(max_symbols or 30), 50))  # hard cap = 50

    SCAN_LOG = Path(__file__).parent.parent / "scan_signals.json"
    try:
        from news_daily_report import _load_alerts, _parse_dt
        from news_provider import get_stock_news_unified
        from news_scorer import score_news as _score_news
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"news modules unavailable: {e}")

    t0 = time.time()
    alerts = _load_alerts()
    if not alerts:
        return {"scored": 0, "skipped_no_alerts": True, "skipped_already_scored": 0,
                "capped": 0, "errors": [], "duration_sec": 0.0,
                "message": "scan_signals.json 为空 — 先跑一次选股扫描"}

    cutoff = datetime.now() - timedelta(hours=window_hours)

    # Find in-window alerts that lack news scores
    needs_scoring: list[dict] = []
    already_scored = 0
    for a in alerts:
        scan_ts = _parse_dt(a.get("scan_time") or "") or _parse_dt(a.get("time") or "")
        if scan_ts is None or scan_ts < cutoff:
            continue
        if market != "ALL" and (a.get("market") or "").upper() != market:
            continue
        n = a.get("news")
        if isinstance(n, dict) and (n.get("score") is not None):
            already_scored += 1
            continue
        needs_scoring.append(a)

    if not needs_scoring:
        return {"scored": 0, "skipped_no_alerts": False,
                "skipped_already_scored": already_scored,
                "capped": 0, "errors": [], "duration_sec": round(time.time() - t0, 2),
                "message": (f"窗口内 {already_scored} 只已有打分，0 只待打分"
                            if already_scored else
                            f"窗口（{window_hours}h, 市场 {market}）内无 alerts")}

    # Dedupe symbols
    sym_to_market: dict[str, str] = {}
    for a in needs_scoring:
        sym = a.get("symbol")
        if sym and sym not in sym_to_market:
            sym_to_market[sym] = a.get("market") or "US"
    syms = list(sym_to_market.keys())
    capped = max(0, len(syms) - max_symbols)
    syms = syms[:max_symbols]

    # Score each
    score_map: dict[str, dict] = {}
    errors: list[dict] = []
    for sym in syms:
        try:
            news_items = get_stock_news_unified(
                sym, sym_to_market[sym], window_hours=window_hours
            )
            sr = _score_news(sym, news_items, provider=provider)
            score_map[sym] = {
                "score": sr.get("score"),
                "label": sr.get("label"),
                "reasoning": sr.get("reasoning"),
                "n_items": sr.get("n_items"),
                "top3": [
                    {
                        "title": (n.get("title") or "")[:200],
                        "url": n.get("url") or "",
                        "source": n.get("source") or "",
                        "published_at": n.get("published_at") or "",
                    }
                    for n in news_items[:3]
                ],
            }
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)[:200]})

    # Write back into scan_signals.json — update ALL alerts of these symbols,
    # not just the first one (so future _selector_score_news doesn't re-score).
    if score_map:
        for a in alerts:
            sym = a.get("symbol")
            if sym in score_map:
                a["news"] = score_map[sym]
        try:
            SCAN_LOG.write_text(
                json.dumps(alerts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"persist failed: {e}")

    return {
        "scored": len(score_map),
        "skipped_no_alerts": False,
        "skipped_already_scored": already_scored,
        "capped": capped,
        "errors": errors,
        "duration_sec": round(time.time() - t0, 2),
        "provider": provider,
        "max_symbols": max_symbols,
    }
