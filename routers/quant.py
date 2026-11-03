"""
Quant Trading HTTP endpoints.
Mounts under /quant/*.
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Optional

import quant_store as _store
import quant_engine as _engine
import quant_llm as _llm
import quant_borrow as _borrow

router = APIRouter()


# ─── Pool CRUD ──────────────────────────────────────────────────

@router.get("/quant/pools")
def list_pools():
    """List all quant pools with basic runtime snapshot."""
    pools = _store.load_pools()
    # Enrich with equity snapshot per pool (no live prices for perf; use entry px)
    out = []
    for p in pools:
        eq = _store.compute_pool_equity(p, {})
        summary = dict(p)
        # equity_history 是整条权益曲线(实测占这个响应 88% 体积, 227KB/256KB),
        # 而量化 tab 拿它一次都没用过 —— 曲线走 /quant/pools/{id}/equity。
        # 每次 tick / 启停 / 平仓 / 保存都会重拉这个接口, 池一多就卡。
        _hist = summary.pop("equity_history", None)
        summary["equity_history_count"] = len(_hist) if isinstance(_hist, list) else 0
        summary["equity_snapshot"] = eq
        summary["open_positions_count"] = sum(
            1 for pos in (p.get("positions") or []) if pos.get("status") == "open")
        out.append(summary)
    return {"pools": out}


@router.get("/quant/pools/{pool_id}")
def get_pool(pool_id: str):
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    eq = _store.compute_pool_equity(pool, {})
    pool["equity_snapshot"] = eq
    return pool


_POOL_DEFAULTS = {
    "auto_buy": {"enabled": True, "min_strategies_agree": 1},
    "auto_sell": {"sell_signal": True, "stop_loss_pct": -5, "take_profit_pct": 15},
    "short_config": {"stop_loss_pct": 5, "take_profit_pct": -15, "borrow_cost_pct_annual": 2.5},
    "capital_allocation": {"mode": "equal_weight", "sizing_mode": "pct_cash",
                           "max_positions_concurrent": 10, "position_size_pct": 10,
                           "use_guardrails": False,
                           "min_position_size": 0, "max_position_size": 0},
    "strategy_selection": {"mode": "per_stock_top_n", "candidate_owners": [],
                           "top_n": 3, "min_annual_return_pct": 15,
                           "rebalance_frequency_days": 7},
    "risk": {"market_hours_only": True, "max_daily_trades": 20, "max_drawdown_stop_pct": -20},
    "fees": {"model": "ib_tiered", "per_share_usd": 0.0067,
             "min_per_order_usd": 0.35, "max_pct_of_notional": 0.01},
    "llm_veto": {"providers": [], "aggregate": "all_agree", "min_confidence": 0.7},
}
_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}


def _num(section: dict, key: str, default, lo=None, hi=None, label: str = ""):
    """取数字字段: 缺失 / null / NaN / 非数字 → default; 越界 → 400."""
    import math
    v = section.get(key)
    if v is None or v == "":
        v = default
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{label or key} 必须是数字 (got {v!r})")
    if math.isnan(v) or math.isinf(v):
        v = float(default)
    if lo is not None and v < lo:
        raise HTTPException(400, f"{label or key} 不能小于 {lo} (got {v})")
    if hi is not None and v > hi:
        raise HTTPException(400, f"{label or key} 不能大于 {hi} (got {v})")
    section[key] = v
    return v


def _validate_pool_body(body: dict) -> dict:
    """填默认值 + 校验模式约束. 返回清洗后的 body, 非法 → HTTPException(400)."""
    if not str(body.get("name") or "").strip():
        raise HTTPException(400, "name is required")
    if not body.get("stock_pool"):
        raise HTTPException(400, "stock_pool is required")
    body["name"] = str(body["name"]).strip()
    body["market"] = str(body.get("market") or "US").upper()
    body["interval"] = str(body.get("interval") or "1d")
    if body["interval"] not in _INTERVALS:
        raise HTTPException(400, f"interval 非法: {body['interval']} (可选 {sorted(_INTERVALS)})")
    body["allow_short"] = bool(body.get("allow_short", False))
    body["strategies"] = [str(s).strip() for s in (body.get("strategies") or []) if str(s).strip()]
    _num(body, "initial_capital", 100000, lo=1, label="初始资金")

    # 嵌套段: 缺的 key 用默认值补齐 (前端只传部分字段也能存)
    for section, defaults in _POOL_DEFAULTS.items():
        cur = body.get(section)
        if not isinstance(cur, dict):
            cur = {}
        body[section] = {**defaults, **cur}

    # 策略选择模式
    mode = str(body.get("strategy_mode") or "top_n").lower()
    if mode not in ("top_n", "manual"):
        raise HTTPException(400, f"strategy_mode 非法: {mode} (top_n | manual)")
    body["strategy_mode"] = mode
    sel = body["strategy_selection"]
    sel["candidate_owners"] = [str(o).strip() for o in (sel.get("candidate_owners") or []) if str(o).strip()]
    _num(sel, "top_n", 3, lo=1, hi=50, label="Top-N")
    sel["top_n"] = int(sel["top_n"])
    _num(sel, "min_annual_return_pct", 15, label="最低年化")
    _num(sel, "rebalance_frequency_days", 7, lo=1, label="重排周期(天)")
    sel["rebalance_frequency_days"] = int(sel["rebalance_frequency_days"])
    if mode == "manual" and not body["strategies"]:
        raise HTTPException(400, "手动模式: 至少选一个策略")
    if mode == "top_n" and not sel["candidate_owners"]:
        raise HTTPException(400, "Top-N 模式: 至少勾选一个 owner")

    # 资金分配
    alloc = body["capital_allocation"]
    sizing = str(alloc.get("sizing_mode") or "pct_cash").lower()
    if sizing not in ("pct_cash", "llm"):
        raise HTTPException(400, f"sizing_mode 非法: {sizing} (pct_cash | llm)")
    alloc["sizing_mode"] = sizing
    _num(alloc, "position_size_pct", 10, lo=0.1, hi=100, label="单仓 %")
    _num(alloc, "max_positions_concurrent", 10, lo=1, label="最大同时持仓")
    alloc["max_positions_concurrent"] = int(alloc["max_positions_concurrent"])
    alloc["use_guardrails"] = bool(alloc.get("use_guardrails", False))
    _num(alloc, "min_position_size", 0, lo=0, label="单仓最低 $")
    _num(alloc, "max_position_size", 0, lo=0, label="单仓最高 $")
    if alloc["use_guardrails"] and alloc["max_position_size"] > 0 \
            and alloc["min_position_size"] > alloc["max_position_size"]:
        raise HTTPException(400, "单仓最低 $ 不能大于单仓最高 $")

    # LLM veto: providers 决定是否启用; sizing=llm 必须有 provider
    veto = body["llm_veto"]
    provs = veto.get("providers")
    if not isinstance(provs, list):
        provs = [veto["provider"]] if veto.get("enabled") and veto.get("provider") else []
    provs = [str(p).lower().strip() for p in provs if str(p).strip()]
    bad = [p for p in provs if p not in _llm.SUPPORTED_PROVIDERS]
    if bad:
        raise HTTPException(400, f"LLM provider 非法: {bad} (可选 {_llm.SUPPORTED_PROVIDERS})")
    veto["providers"] = provs
    veto["enabled"] = bool(provs)
    veto["aggregate"] = str(veto.get("aggregate") or "all_agree").lower()
    if veto["aggregate"] not in ("all_agree", "majority"):
        raise HTTPException(400, f"aggregate 非法: {veto['aggregate']} (all_agree | majority)")
    _num(veto, "min_confidence", 0.7, lo=0, hi=1, label="最低 confidence")
    if sizing == "llm" and not provs:
        raise HTTPException(400, "仓位模式=LLM 决定 需要至少勾选一个 LLM 参与决策")

    # 自动卖 / 卖空 / 风控 / 费率
    _num(body["auto_buy"], "min_strategies_agree", 1, lo=1, label="最少策略同意数")
    body["auto_buy"]["min_strategies_agree"] = int(body["auto_buy"]["min_strategies_agree"])
    _num(body["auto_sell"], "stop_loss_pct", -5, hi=0, label="多头止损 %")
    _num(body["auto_sell"], "take_profit_pct", 15, lo=0, label="多头止盈 %")
    _num(body["short_config"], "stop_loss_pct", 5, lo=0, label="空头止损 %")
    _num(body["short_config"], "take_profit_pct", -15, hi=0, label="空头止盈 %")
    _num(body["short_config"], "borrow_cost_pct_annual", 2.5, lo=0, hi=100, label="借券年化 %")
    _num(body["risk"], "max_daily_trades", 20, lo=0, label="每日最多交易数")
    body["risk"]["max_daily_trades"] = int(body["risk"]["max_daily_trades"])
    _num(body["risk"], "max_drawdown_stop_pct", -20, hi=0, label="回撤停止 %")
    body["risk"]["market_hours_only"] = bool(body["risk"].get("market_hours_only", True))
    _num(body["fees"], "per_share_usd", 0.0067, lo=0, label="每股手续费")
    _num(body["fees"], "min_per_order_usd", 0.35, lo=0, label="每单最低手续费")
    _num(body["fees"], "max_pct_of_notional", 0.01, lo=0, hi=1, label="手续费上限(比例)")
    return body


@router.post("/quant/pools")
def upsert_pool(body: dict = Body(...)):
    """Create or update pool. Body is the full pool config; server preserves
    runtime state (cash / positions / history / enabled / equity_history) on update.
    模式约束在这里校验 (manual 需策略, top_n 需 owner, sizing=llm 需 provider ...)."""
    body = _validate_pool_body(body)
    if not body.get("id"):
        body["enabled"] = False   # 新建默认停用; 更新时 enabled 由 store 保留 (/toggle 单独控制)
    else:
        body.pop("enabled", None)
    saved = _store.upsert_pool(body)
    return {"ok": True, "pool": saved}


@router.get("/quant/llm-providers")
def llm_providers():
    """各 LLM provider 可用状态 (只读; 编辑去 系统 · LLM 管理 tab)."""
    return _llm.provider_status()


@router.delete("/quant/pools/{pool_id}")
def delete_pool(pool_id: str):
    ok = _store.delete_pool(pool_id)
    if not ok:
        raise HTTPException(404, f"pool {pool_id} not found")
    return {"ok": True}


@router.post("/quant/pools/{pool_id}/toggle")
def toggle_pool(pool_id: str):
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    new_state = not bool(pool.get("enabled"))
    _store.update_pool_fields(pool_id, enabled=new_state)
    return {"ok": True, "enabled": new_state}


@router.post("/quant/pools/{pool_id}/rebalance")
def manual_rebalance(pool_id: str):
    """Manually trigger strategy rebalance for this pool (Phase 2).
    Re-runs owner→strategies filter + per-stock top-N ranking."""
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    if pool.get("strategy_mode") == "manual":
        raise HTTPException(400, "手动指定策略模式不需要 rebalance")
    try:
        result = _engine.rebalance_pool(pool)
    except Exception as e:
        raise HTTPException(500, str(e))
    if result.get("error"):
        # 0 命中 / 无候选策略: 不是服务器错误, 但也不能算成功
        raise HTTPException(409, result["error"])
    return {"ok": True, "result": result}


@router.post("/quant/pools/{pool_id}/tick")
def manual_tick(pool_id: str):
    """Run one immediate tick for this pool (bypass daemon)."""
    try:
        stats = _engine.tick_once(pool_id)
        return {"ok": True, "stats": stats}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))   # daemon 正在处理该池
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Positions / Trades ──────────────────────────────────────────

@router.get("/quant/pools/{pool_id}/positions")
def get_positions(pool_id: str):
    """Return open positions with live prices + unrealized PnL."""
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    open_pos = [p for p in (pool.get("positions") or []) if p.get("status") == "open"]
    # Fetch live prices in parallel
    from quote_router import router as _qr
    from concurrent.futures import ThreadPoolExecutor
    price_map = {}
    if open_pos:
        def _fetch(sym_mkt):
            sym, mkt = sym_mkt
            try:
                q = _qr.fetch_quote(sym, mkt)
                return sym, float(q.get("price", 0)) if q else 0.0
            except Exception:
                return sym, 0.0
        unique = list({(p["symbol"], p.get("market", pool.get("market", "US")))
                       for p in open_pos})
        with ThreadPoolExecutor(max_workers=min(8, len(unique))) as ex:
            for sym, price in ex.map(_fetch, unique):
                price_map[sym] = price
    enriched = []
    for pos in open_pos:
        cur = price_map.get(pos["symbol"], 0)
        pnl = _store.compute_position_pnl(pos, cur)
        enriched.append({**pos, "current_price": cur, **pnl})
    equity = _store.compute_pool_equity(pool, price_map)
    return {"positions": enriched, "equity": equity}


@router.get("/quant/pools/{pool_id}/equity")
def get_equity_history(pool_id: str, limit: int = 1000):
    """Return equity_history for equity curve rendering."""
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    hist = pool.get("equity_history") or []
    if len(hist) > limit:
        hist = hist[-limit:]
    return {"equity_history": hist,
            "initial_capital": pool.get("initial_capital", 0)}


@router.get("/quant/pools/{pool_id}/history")
def get_history(pool_id: str, limit: int = 200):
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    hist = list(reversed(pool.get("history") or []))[:max(1, min(int(limit), 500))]
    return {"history": hist}


@router.post("/quant/pools/{pool_id}/close")
def manual_close(pool_id: str, body: dict = Body(...)):
    """Manual close (full or partial) for a specific position.
    Body: {position_id, price?, quantity?, reason?}
    If price omitted, fetches live via quote_router."""
    pos_id = body.get("position_id")
    if not pos_id:
        raise HTTPException(400, "position_id required")
    pool = _store.get_pool(pool_id)
    if not pool:
        raise HTTPException(404, f"pool {pool_id} not found")
    pos = next((p for p in (pool.get("positions") or [])
               if p["id"] == pos_id and p.get("status") == "open"), None)
    if not pos:
        raise HTTPException(404, "position not found or already closed")
    price = body.get("price")
    if not price or float(price) <= 0:
        # 拉现价
        from quote_router import router as _qr
        q = _qr.fetch_quote(pos["symbol"], pos.get("market", pool.get("market", "US")))
        price = float(q.get("price", 0)) if q else 0
        if not price or price <= 0:
            raise HTTPException(400, "price fetch failed, provide manually")
    trade = _store.close_position(pool_id, pos_id, float(price),
                                  quantity=body.get("quantity"),
                                  reason=body.get("reason", "manual"))
    if not trade:
        raise HTTPException(500, "close failed")
    return {"ok": True, "trade": trade}


# ─── Events / Log ────────────────────────────────────────────────

@router.get("/quant/events")
def list_events(pool_id: Optional[str] = None, limit: int = 100):
    events = _store.load_events()
    if pool_id:
        events = [e for e in events if e.get("pool_id") == pool_id]
    events = list(reversed(events))[:max(1, min(int(limit), 500))]
    return {"events": events}


# ─── Daemon control ──────────────────────────────────────────────

@router.get("/quant/borrow-rates")
def list_borrow_rates():
    """Return all seeded + user-override borrow rates."""
    return _borrow.list_all()


@router.get("/quant/borrow-rates/{symbol}")
def get_borrow_rate(symbol: str, probe_ib: bool = True):
    rate, source = _borrow.get_borrow_rate(symbol, probe_ib=probe_ib)
    return {"symbol": symbol.upper(), "rate_pct_annual": rate, "source": source}


@router.get("/quant/borrow-rates/{symbol}/ib-probe")
def probe_ib_shortable(symbol: str):
    """Return IB Gateway's current shortable status for the symbol."""
    result = _borrow._probe_ib_shortable(symbol)
    if result is None:
        return {"symbol": symbol.upper(), "ib_available": False,
                "shortable": None, "note": "IB unavailable or probe failed"}
    return {"symbol": symbol.upper(), "ib_available": True,
            "shortable": result,
            "would_use_rate": (_borrow.HTB_RATE if not result else _borrow.get_borrow_rate(symbol, probe_ib=False)[0])}


@router.post("/quant/borrow-rates/{symbol}")
def set_borrow_rate(symbol: str, body: dict = Body(...)):
    """Override a symbol's borrow rate."""
    try:
        rate = float(body.get("rate_pct_annual", 0))
        if rate < 0 or rate > 100:
            raise ValueError("rate must be 0-100")
        _borrow.save_rate(symbol, rate)
        return {"ok": True, "symbol": symbol.upper(), "rate_pct_annual": rate}
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@router.get("/quant/llm-prompt")
def get_llm_prompt():
    """返回当前用户覆盖的 system prompt (or default) 及默认 prompt 参考."""
    current = _llm.get_system_prompt()
    default = _llm.get_default_system_prompt()
    return {"system_prompt": current, "default_prompt": default,
            "is_default": current == default}


@router.post("/quant/llm-prompt")
def set_llm_prompt(body: dict = Body(...)):
    """保存 system prompt. 传空字符串恢复默认."""
    text = str(body.get("system_prompt") or "")
    result = _llm.save_system_prompt(text)
    return {"ok": True, **result}


@router.get("/quant/llm-prompt/preview")
def preview_llm_prompt(pool_id: Optional[str] = None):
    """
    渲染一条 BUY 决策的完整 LLM 调用内容 (system + user prompt) 用于 UI 预览.
    如果 pool_id 给定, 用真实池数据; 否则用假池.
    """
    system_prompt = _llm.get_system_prompt()
    if pool_id:
        pool = _store.get_pool(pool_id) or {}
    else:
        pool = {}
    # 若无 pool 或池空, 造假样例
    if not pool.get("name"):
        pool = {
            "id": "qp-preview", "name": "SAMPLE_POOL",
            "initial_capital": 100000, "cash": 92341.50,
            "market": "US", "interval": "1d",
            "positions": [
                {"id": "sample1", "status": "open", "side": "LONG",
                 "symbol": "AAPL", "quantity": 20, "entry_price": 180.0},
            ],
            "equity_history": [
                {"equity": 98500, "return_pct": -1.5},
                {"equity": 101200, "return_pct": 1.2},
            ],
        }
    # 拿池里第一只股票 (或 fallback NVDA) 做样例
    stock_pool_symbols = []
    try:
        if pool.get("stock_pool"):
            import main as _m
            sp = _m._load_pools().get(pool["stock_pool"]) or {}
            stock_pool_symbols = sp.get("symbols") or []
    except Exception:
        pass
    sample_symbol = stock_pool_symbols[0] if stock_pool_symbols else "NVDA"
    decision = {
        "action": "BUY", "symbol": sample_symbol,
        "quantity": 42, "price": 178.35,
        "strategy": "dual_ma_cross", "existing_position": None,
    }
    user_prompt = _llm._build_user_prompt(decision, pool)
    return {"system_prompt": system_prompt, "user_prompt": user_prompt,
            "sample_decision": decision, "used_pool": pool.get("name")}


@router.get("/quant/daemon/status")
async def daemon_status():
    return _engine.get_daemon_status()


@router.post("/quant/daemon/start")
def daemon_start():
    started = _engine.start_daemon()
    return {"started": started, "status": _engine.get_daemon_status()}


@router.post("/quant/daemon/stop")
def daemon_stop():
    stopped = _engine.stop_daemon()
    return {"stopped": stopped, "status": _engine.get_daemon_status()}
