"""
Quant Trading Engine — background daemon + 5-stage pipeline.

Phase 1 scope:
- daemon 60s tick 循环所有 enabled 池
- 5 阶段 pipeline (universe → raw_signal → filter → decision → execute)
- 支持 LONG 开/平 + SHORT 开/平
- 触发条件: 策略 BUY/SELL edge, stop_loss, take_profit
- IB Tiered fee 每次执行扣
- 借券成本 SHORT 平仓时按持有天数结算
- 无 LLM veto (Phase 3)
- 无 top-N 策略排名 (Phase 2)

参考: core_engine._process_monitor_groups 的架构骨架 + 竞技池 5 阶段模型.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import quant_store as _store
import quant_llm as _llm

logger = logging.getLogger(__name__)

TICK_SEC = 60          # daemon tick 周期
DEFAULT_REBALANCE_DAYS = 7   # 策略重排默认周期
REBALANCE_RETRY_SEC = 3600   # rebalance 失败/0 命中后的重试间隔 (避免每 tick 重跑 2 年回测)
_DAEMON_THREAD: Optional[threading.Thread] = None

# 每池一把锁: daemon tick 与 UI [▶ 手动 tick] 不能同时处理同一个池
# (否则两边各自读到"无持仓"→ 同一信号开两次仓)
_POOL_LOCKS: dict[str, threading.Lock] = {}
_POOL_LOCKS_META = threading.Lock()


def _pool_lock(pool_id: str) -> threading.Lock:
    with _POOL_LOCKS_META:
        lock = _POOL_LOCKS.get(pool_id)
        if lock is None:
            lock = threading.Lock()
            _POOL_LOCKS[pool_id] = lock
        return lock
_DAEMON_STOP = threading.Event()
_DAEMON_STATE = {
    "running": False,
    "started_at": None,
    "last_tick_at": None,
    "last_error": None,
    "processed_pools_last_tick": 0,
    "next_tick_at": None,
}
_DAEMON_LOCK = threading.Lock()


# ─── 市场时段判定 ─────────────────────────────────────────────────────

def _in_market_hours(market: str) -> bool:
    """
    交易时段判定. 复用 core_engine._is_market_open (各市场本地时区,
    自动处理夏令时; 之前这里硬编码 UTC 偏移, 美股夏令时会差 1 小时).
    CRYPTO: 24/7. 未知市场: 不锁.
    """
    m = (market or "").upper()
    if m == "CRYPTO":
        return True
    try:
        from core_engine import _is_market_open
        return bool(_is_market_open(m))
    except Exception as e:
        logger.debug(f"market hours check {m}: {e}")
        return True


# ─── 拉行情 (通过 quote_router, 走三级 fallback) ───────────────────

def _get_current_price(symbol: str, market: str) -> Optional[float]:
    """Return live price via quote_router. Cache via router's own layer."""
    try:
        from quote_router import router as _qr
        q = _qr.fetch_quote(symbol, market)
        if q and q.get("price"):
            return float(q["price"])
    except Exception as e:
        logger.debug(f"quote fetch {symbol}/{market}: {e}")
    return None


# ─── 策略信号 ────────────────────────────────────────────────────────

# ─── Phase 2: Owner → strategies + Top-N ranking ────────────────────

def _get_active_strategies_for_owners(candidate_owners: list) -> list:
    """
    从当前 active 策略里过滤出属于 candidate_owners 之一的.
    如果 candidate_owners 为空或含 'ALL' → 全部 active 策略.
    """
    try:
        import main as _m
        all_active = _m._get_active_strategies()
        if not candidate_owners or "ALL" in [str(o).upper() for o in candidate_owners]:
            return list(all_active)
        wanted = {str(o).strip() for o in candidate_owners}
        try:
            llm_registry = _m._load_strategy_llm_registry()
        except Exception:
            llm_registry = {}
        return [
            name for name in all_active
            if _m._get_strategy_llm_owner(name, llm_registry) in wanted
        ]
    except Exception as e:
        logger.warning(f"list owner strategies failed: {e}")
        return []


def _needs_rebalance(pool: dict) -> bool:
    """判断是否到期需要重排. 仅 top_n 模式需要."""
    if pool.get("strategy_mode") == "manual":
        return False
    sel = pool.get("strategy_selection") or {}
    if not sel.get("mode"):  # 未启用 top-N (兼容 Phase 1 老池)
        return False
    freq_days = int(sel.get("rebalance_frequency_days", DEFAULT_REBALANCE_DAYS))
    now = datetime.now(timezone.utc)

    def _age_sec(ts) -> Optional[float]:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return (now - dt).total_seconds()
        except Exception:
            return None

    # 上次 rebalance 失败 / 0 命中: 不刷 last_rebalance_at, 但按 REBALANCE_RETRY_SEC 退避
    attempt_age = _age_sec(pool.get("last_rebalance_attempt_at"))
    success_age = _age_sec(pool.get("last_rebalance_at"))
    if success_age is None:
        return attempt_age is None or attempt_age >= REBALANCE_RETRY_SEC
    if success_age >= freq_days * 86400:
        return attempt_age is None or attempt_age >= REBALANCE_RETRY_SEC
    return False


def rebalance_pool(pool: dict) -> dict:
    """
    重排整个池的策略: 每 symbol 跑候选策略 → 排序 → 存 top-N.
    复用 core_engine._rank_strategies_for_stock (2年 OHLCV + 年化收益).
    """
    pool_id = pool["id"]
    sel = pool.get("strategy_selection") or {}
    # 先记"尝试过", 失败路径只刷这个, 成功才刷 last_rebalance_at
    _store.update_pool_fields(pool_id, last_rebalance_attempt_at=_store._now_iso())
    candidates = _get_active_strategies_for_owners(sel.get("candidate_owners") or [])
    if not candidates:
        return {"error": "no active strategies for candidate owners",
                "candidates": 0, "symbols_ranked": 0}

    top_n = int(sel.get("top_n", 3))
    min_return = float(sel.get("min_annual_return_pct", 15))

    # 拉股票池 symbols
    try:
        import main as _m
        stock_pool = _m._load_pools().get(pool.get("stock_pool"))
        symbols = [s.upper() for s in (stock_pool.get("symbols") or [])] if stock_pool else []
    except Exception as e:
        return {"error": f"load stock_pool failed: {e}"}

    market = pool.get("market", "US").upper()

    # 复用 core_engine._rank_strategies_for_stock 逐个跑 (可并行)
    try:
        import core_engine as _me
    except Exception as e:
        return {"error": f"core_engine import failed: {e}"}

    ranks_cache = {}
    skip_reasons = {}   # {symbol: 为什么没排上} —— 界面上的「未排」要能说出原因
    # 原先是 ThreadPoolExecutor: _rank_strategies_for_stock 是 CPU 密集的
    # (一次取数 + N 个策略回测), 线程池被 GIL 串死, 实测 6只×7策略
    # 线程池 1.01s vs 串行 0.94s —— 比串行还慢。换成自适应派发。
    args_list = [(sym, market, candidates, min_return, top_n) for sym in symbols]
    _ranked_syms = set()
    for sym, mkt, selected, *_r in _me.rank_strategies_parallel(args_list, max_workers=8):
        _ranked_syms.add(sym)
        if selected:
            ranks_cache[sym] = [
                {"strategy": s["strategy_name"],
                 "annual_return": s.get("annual_return"),
                 "sharpe": s.get("sharpe"),
                 "rank": s.get("selection_rank")}
                for s in selected
            ]
        else:
            skip_reasons[sym] = (_r[0] if _r else "") or "unknown"
    # 进程池分片整个失败时, 这只股票连结果都回不来 —— 也要标出来,
    # 否则跟"门槛没过"一样显示成光秃秃的「未排」。
    for sym in symbols:
        if sym not in _ranked_syms and sym not in ranks_cache:
            skip_reasons.setdefault(sym, "rank_missing")

    if not ranks_cache:
        # 0 命中: 保留上一次的 cache, 不刷 last_rebalance_at (让 _needs_rebalance
        # 按 REBALANCE_RETRY_SEC 退避重试), 记一条事件方便排查.
        _store.append_event({
            "type": "REBALANCE", "pool_id": pool_id,
            "reason": (f"0/{len(symbols)} symbols ranked · candidates={len(candidates)} "
                       f"· min_return={min_return}% — 保留旧 cache, {REBALANCE_RETRY_SEC//60}min 后重试"),
            "candidates_count": len(candidates), "symbols_ranked": 0, "ok": False,
        })
        return {"error": f"no symbol ranked (candidates={len(candidates)}, min_return={min_return}%)",
                "symbols_ranked": 0, "candidates_count": len(candidates),
                "top_n": top_n, "min_return": min_return}

    # 写回 pool
    _store.update_pool_fields(
        pool_id,
        strategy_ranks_cache=ranks_cache,
        strategy_rank_skips=skip_reasons,
        last_rebalance_at=_store._now_iso(),
    )
    _store.append_event({
        "type": "REBALANCE", "pool_id": pool_id,
        "reason": f"ranked {len(ranks_cache)}/{len(symbols)} symbols · candidates={len(candidates)}",
        "candidates_count": len(candidates), "symbols_ranked": len(ranks_cache),
    })
    return {"ok": True, "symbols_ranked": len(ranks_cache),
            "candidates_count": len(candidates), "top_n": top_n,
            "min_return": min_return, "skipped": len(skip_reasons),
            "skip_reasons": skip_reasons}


def _get_effective_strategies(pool: dict, symbol: str) -> list:
    """
    返回该 symbol 该 pool 实际用来跑信号的策略列表.

    严格按 pool.strategy_mode 分流 (不再 fallback 混跑):
      - "top_n"  → 只用 strategy_ranks_cache[symbol]. rebalance 没跑该 sym
                   就返回 []; symbol 该轮不出信号.
      - "manual" → 只用 pool.strategies 硬编码列.
      - 未设 (Phase 1 老池兼容): 若有 ranks_cache 用 top_n, 否则 manual.
    """
    mode = pool.get("strategy_mode")
    cache = pool.get("strategy_ranks_cache") or {}
    manual_list = list(pool.get("strategies") or [])

    if mode == "top_n":
        return [r["strategy"] for r in (cache.get(symbol) or [])]
    if mode == "manual":
        return manual_list
    # 未指定 mode: 老池兼容路径
    if cache.get(symbol):
        return [r["strategy"] for r in cache[symbol]]
    return manual_list


def _get_signal_edge(symbol: str, market: str, strategy_name: str,
                    interval: str = "1d") -> tuple[Optional[str], float, str]:
    """
    Run one strategy on the symbol, return the LAST-BAR signal edge:
      - "BUY"  = fresh buy_signal edge (last bar True, prev bar False)
      - "SELL" = fresh sell_signal edge
      - None   = no fresh signal on last bar
    Returns (edge, last_bar_close, last_bar_ts).
    last_bar_ts 用来去重: 同一根 bar 上的同一个 edge 只能触发一次动作
    (1d 周期时 get_ohlcv 不含今日 bar, 昨天的 edge 会在今天每个 tick 反复出现).
    参数来源: 个股标准参数 (_get_standard_strategy_params, 按 symbol + 策略 + 周期).
    """
    try:
        from backtest import get_ohlcv
        from strategies import get_strategy
        import main as _m
    except Exception as e:
        logger.warning(f"import failed for signal check: {e}")
        return None, 0.0, ""

    period_map = {"1m": "5d", "5m": "20d", "15m": "60d", "30m": "60d",
                  "1h": "6mo", "1d": "1y", "1wk": "5y", "1mo": "10y"}
    try:
        df = get_ohlcv(symbol, market, interval=interval,
                       period=period_map.get(interval, "1y"), local_only=False)
        if df is None or len(df) < 30:
            return None, 0.0, ""
        strategy = get_strategy(strategy_name)
        if not strategy:
            return None, 0.0, ""
        params = _m._get_standard_strategy_params(symbol, strategy_name, interval)
        df_ind = strategy.add_indicators(df.copy(), params)
        df_sig = strategy.generate_signals(df_ind, params)
        if len(df_sig) < 2:
            return None, 0.0, ""
        last = df_sig.iloc[-1]
        prev = df_sig.iloc[-2]
        idx = df_sig.index[-1]
        bar_ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        current_price = float(last.get("close", 0)) or 0.0
        buy_edge = bool(last.get("buy_signal", False)) and not bool(prev.get("buy_signal", False))
        sell_edge = bool(last.get("sell_signal", False)) and not bool(prev.get("sell_signal", False))
        if buy_edge:
            return "BUY", current_price, bar_ts
        if sell_edge:
            return "SELL", current_price, bar_ts
        return None, current_price, bar_ts
    except Exception as e:
        logger.debug(f"signal check {symbol}/{strategy_name}: {e}")
        return None, 0.0, ""


# ─── 单个 pool 的 tick ─────────────────────────────────────────────

def _process_pool(pool: dict, wait_sec: Optional[float] = None) -> dict:
    """
    Full 5-stage pipeline for ONE pool (串行: 同一池同一时间只跑一份).
    wait_sec=None → 阻塞等锁 (daemon); 给数值 → 等不到就抛 RuntimeError (手动 tick).
    Returns stats: {actions, skipped, errors, ...}
    """
    lock = _pool_lock(pool["id"])
    if wait_sec is None:
        acquired = lock.acquire()
    else:
        acquired = lock.acquire(timeout=float(wait_sec))
    if not acquired:
        raise RuntimeError(f"pool {pool['id']} 正在被 daemon 处理, 稍后再试")
    try:
        return _process_pool_unlocked(pool)
    finally:
        lock.release()


def _process_pool_unlocked(pool: dict) -> dict:
    pool_id = pool["id"]
    market = pool.get("market", "US").upper()
    stock_pool_name = pool.get("stock_pool")
    interval = pool.get("interval", "1d")

    stats = {"actions": 0, "skipped": 0, "errors": 0,
             "buys": 0, "sells": 0, "shorts": 0, "covers": 0,
             "stop_loss_closes": 0, "take_profit_closes": 0,
             "rebalanced": False}

    if not stock_pool_name:
        return stats

    # Phase 2: 到期就 rebalance
    if _needs_rebalance(pool):
        try:
            rr = rebalance_pool(pool)
            if rr.get("ok"):
                stats["rebalanced"] = True
                # 拿最新 pool 数据 (含新 ranks cache)
                pool = _store.get_pool(pool_id) or pool
            else:
                logger.warning(f"[quant] rebalance {pool_id} failed: {rr}")
        except Exception as e:
            logger.warning(f"[quant] rebalance {pool_id} error: {e}")

    # 至少要有一个策略配置 (硬编码或已 rebalance 后的 cache)
    if not (pool.get("strategies") or pool.get("strategy_ranks_cache")):
        return stats

    # ── (1) universe: 从股票池取 symbols
    try:
        import main as _m
        pools_map = _m._load_pools()
        stock_pool = pools_map.get(stock_pool_name)
        if not stock_pool:
            logger.warning(f"[quant] pool {pool_id}: stock_pool '{stock_pool_name}' not found")
            return stats
        symbols = [s.upper() for s in (stock_pool.get("symbols") or [])]
    except Exception as e:
        logger.warning(f"[quant] pool {pool_id} universe error: {e}")
        return stats

    if not symbols:
        return stats

    # ── (3a) global filter: market hours
    if pool.get("risk", {}).get("market_hours_only", True) and not _in_market_hours(market):
        stats["skipped"] += len(symbols)
        return stats

    # ── (3b) global filter: drawdown stop
    # dd_stopped=True 只停新开仓 (BUY/SHORT), 平仓 (SELL/COVER/止损止盈) 照走
    equity_snapshot = _store.compute_pool_equity(pool, {})
    max_dd_pct = float(pool.get("risk", {}).get("max_drawdown_stop_pct", -100))
    dd_stopped = equity_snapshot["return_pct"] <= max_dd_pct

    # ── (3c) daily trade cap — exit_time / entry_time 都是 UTC ISO, 这里也用 UTC 日期
    today_iso = datetime.now(timezone.utc).date().isoformat()
    todays_history = [t for t in (pool.get("history") or [])
                     if str(t.get("exit_time", "")).startswith(today_iso)]
    todays_opens = [p for p in (pool.get("positions") or [])
                    if str(p.get("entry_time", "")).startswith(today_iso)]
    daily_trades_used = len(todays_history) + len(todays_opens)
    max_daily = int(pool.get("risk", {}).get("max_daily_trades", 20))

    open_positions = [p for p in (pool.get("positions") or []) if p.get("status") == "open"]
    concurrent_used = len(open_positions)
    max_concurrent = int(pool.get("capital_allocation", {}).get("max_positions_concurrent", 10))

    allow_short = bool(pool.get("allow_short", False))
    auto_buy_enabled = bool(pool.get("auto_buy", {}).get("enabled", True))
    auto_sell = pool.get("auto_sell", {}) or {}
    short_cfg = pool.get("short_config", {}) or {}

    # ── 首先: 检查所有 open 仓的 stop_loss / take_profit
    for pos in list(open_positions):
        if pos.get("status") != "open":
            continue
        sym = pos["symbol"]
        cur_price = _get_current_price(sym, pos.get("market", market))
        if cur_price is None or cur_price <= 0:
            continue
        entry = float(pos["entry_price"])
        side = pos.get("side", "LONG")

        if side == "LONG":
            sl_pct = float(auto_sell.get("stop_loss_pct", -100))  # -5 表示 -5%
            tp_pct = float(auto_sell.get("take_profit_pct", 10000))
            change = (cur_price - entry) / entry * 100
            trigger_reason = None
            if change <= sl_pct:
                trigger_reason = f"stop_loss ({change:.2f}% ≤ {sl_pct}%)"
            elif change >= tp_pct:
                trigger_reason = f"take_profit ({change:.2f}% ≥ {tp_pct}%)"
        else:  # SHORT
            sl_pct = float(short_cfg.get("stop_loss_pct", 100))
            tp_pct = float(short_cfg.get("take_profit_pct", -10000))
            # SHORT PnL 是 (entry - cur), stop_loss 是 cur 涨 5% → change=+5
            change = (cur_price - entry) / entry * 100
            trigger_reason = None
            if change >= sl_pct:
                trigger_reason = f"short_stop_loss ({change:.2f}% ≥ {sl_pct}%)"
            elif change <= tp_pct:
                trigger_reason = f"short_take_profit ({change:.2f}% ≤ {tp_pct}%)"

        if trigger_reason:
            trade = _store.close_position(pool_id, pos["id"], cur_price, reason=trigger_reason)
            if trade:
                stats["actions"] += 1
                if "stop_loss" in trigger_reason: stats["stop_loss_closes"] += 1
                if "take_profit" in trigger_reason: stats["take_profit_closes"] += 1
                daily_trades_used += 1
                concurrent_used -= 1
                # 重新从 store 拿最新 pool (因为 close_position 已改数据)
                pool = _store.get_pool(pool_id) or pool

    # ── (2) + (3d + 4): 对每只 symbol × 每个策略跑, 找信号 edge, 决策, 执行
    # 每只 symbol 只处理一个决策 (第一个策略触发 edge 就决定, 避免多策略互冲)
    for sym in symbols:
        if daily_trades_used >= max_daily:
            stats["skipped"] += 1
            continue

        # 该 symbol 当前的持仓 (open)
        pos_on_sym = next((p for p in (pool.get("positions") or [])
                          if p["symbol"] == sym and p.get("status") == "open"), None)

        # 跑策略拿信号 — Phase 2: 用 top-N 有序列表, min_strategies_agree 门槛
        effective_strategies = _get_effective_strategies(pool, sym)
        if not effective_strategies:
            continue
        buy_hits = []
        sell_hits = []
        last_price = 0.0
        last_bar_ts = ""
        for strategy_name in effective_strategies:
            edge, price, bar_ts = _get_signal_edge(sym, market, strategy_name, interval)
            if price:
                last_price = price
            if bar_ts:
                last_bar_ts = bar_ts
            if edge == "BUY":
                buy_hits.append(strategy_name)
            elif edge == "SELL":
                sell_hits.append(strategy_name)

        min_agree = int(pool.get("auto_buy", {}).get("min_strategies_agree", 1))
        signal_edge = None
        signal_strategy = None
        if len(buy_hits) >= min_agree:
            signal_edge = "BUY"
            signal_strategy = ",".join(buy_hits[:min_agree])
        elif len(sell_hits) >= min_agree:
            signal_edge = "SELL"
            signal_strategy = ",".join(sell_hits[:min_agree])

        if not signal_edge:
            continue

        # 同一根 bar 的同一个 edge 只动作一次 (否则 1d 周期下: 止损平掉 → 下个
        # tick 同一个 BUY edge 又开仓 → 再止损 ... 直到 max_daily_trades 耗尽)
        acted = pool.get("acted_signals") or {}
        signal_key = f"{signal_edge}@{last_bar_ts}"
        if last_bar_ts and acted.get(sym) == signal_key:
            continue

        # 成交价: 用实时价 (1d 周期时 last_price 是昨收, 不能当今天的成交价);
        # 拉不到实时价才退回 bar 收盘价
        signal_price = _get_current_price(sym, market) or last_price or 0.0
        if signal_price <= 0:
            stats["errors"] += 1
            continue

        # ── (4) decision + (5) LLM veto + execute
        pending_action = None  # ('BUY'/'SELL'/'SHORT'/'COVER', qty)
        if signal_edge == "BUY":
            if pos_on_sym and pos_on_sym["side"] == "SHORT":
                pending_action = ("COVER", pos_on_sym["quantity"])
            elif auto_buy_enabled and not dd_stopped and not pos_on_sym and concurrent_used < max_concurrent:
                qty = _compute_position_size(pool, signal_price)
                if qty > 0:
                    pending_action = ("BUY", qty)
        elif signal_edge == "SELL":
            if pos_on_sym and pos_on_sym["side"] == "LONG":
                pending_action = ("SELL", pos_on_sym["quantity"])
            elif allow_short and not dd_stopped and not pos_on_sym and concurrent_used < max_concurrent:
                qty = _compute_position_size(pool, signal_price)
                if qty > 0:
                    pending_action = ("SHORT", qty)

        if not pending_action:
            if concurrent_used >= max_concurrent:
                stats["skipped"] += 1
            continue

        action, qty = pending_action

        # 先记下"这根 bar 的这个 edge 已处理" — 不管最终 veto/执行结果, 都不再重复触发
        if last_bar_ts:
            acted = dict(pool.get("acted_signals") or {})
            acted[sym] = signal_key
            _store.update_pool_fields(pool_id, acted_signals=acted)
            pool["acted_signals"] = acted

        # ── (5) LLM veto (可选, 默认关) ────────────────────────
        veto = _llm.veto_decision(
            {
                "action": action, "symbol": sym, "quantity": qty,
                "price": signal_price, "strategy": signal_strategy,
                "existing_position": (
                    f"{pos_on_sym.get('side')} {pos_on_sym.get('quantity')} @ ${pos_on_sym.get('entry_price')}"
                    if pos_on_sym else None
                ),
            }, pool)
        if veto["verdict"] == "SKIP":
            _store.append_event({
                "type": "LLM_VETO", "pool_id": pool_id, "symbol": sym,
                "action": action, "reason": f"llm_skip: {veto.get('reason','')}",
                "confidence": veto.get("confidence"),
                "provider": veto.get("provider"),
            })
            stats["skipped"] += 1
            continue

        # LLM 决定仓位模式: veto 若返回 suggested_size_pct (0-100) 就替换 qty
        alloc = pool.get("capital_allocation") or {}
        if alloc.get("sizing_mode") == "llm" and action in ("BUY", "SHORT"):
            llm_pct = veto.get("suggested_size_pct")
            if llm_pct is not None:
                try:
                    llm_pct = float(llm_pct)
                    if 0 < llm_pct <= 100:
                        new_qty = _compute_position_size(pool, signal_price,
                                                         override_pct=llm_pct / 100)
                        if new_qty > 0:
                            qty = new_qty
                            pending_action = (action, qty)
                except (TypeError, ValueError):
                    pass

        # ── execute
        if action == "COVER":
            _store.close_position(pool_id, pos_on_sym["id"], signal_price,
                                  reason=f"buy_signal_covers_short ({signal_strategy})")
            stats["actions"] += 1; stats["covers"] += 1
            daily_trades_used += 1; concurrent_used -= 1
        elif action == "SELL":
            _store.close_position(pool_id, pos_on_sym["id"], signal_price,
                                  reason=f"sell_signal ({signal_strategy})")
            stats["actions"] += 1; stats["sells"] += 1
            daily_trades_used += 1; concurrent_used -= 1
        elif action == "BUY":
            _store.open_position(pool_id, sym, market, "LONG", qty,
                                 signal_price, signal_strategy,
                                 reason=f"buy_signal ({signal_strategy})")
            stats["actions"] += 1; stats["buys"] += 1
            daily_trades_used += 1; concurrent_used += 1
        elif action == "SHORT":
            _store.open_position(pool_id, sym, market, "SHORT", qty,
                                 signal_price, signal_strategy,
                                 reason=f"sell_signal_opens_short ({signal_strategy})")
            stats["actions"] += 1; stats["shorts"] += 1
            daily_trades_used += 1; concurrent_used += 1

        pool = _store.get_pool(pool_id) or pool

    # 收尾: update last_run_at + 采样 equity snapshot (Phase 4)
    _store.update_pool_fields(pool_id, last_run_at=_store._now_iso())
    try:
        # 拉每仓现价算 equity, snapshot 存
        pool = _store.get_pool(pool_id) or pool
        price_map = {}
        for pos in pool.get("positions", []) or []:
            if pos.get("status") == "open":
                cp = _get_current_price(pos["symbol"], pos.get("market", market))
                if cp:
                    price_map[pos["symbol"]] = cp
        eq = _store.compute_pool_equity(pool, price_map)
        _store.append_equity_snapshot(pool_id, {
            "equity": eq["equity"], "cash": eq["cash"],
            "long_mv": eq["long_mv"], "short_mv": eq["short_mv"],
            "unrealized": eq["unrealized"], "return_pct": eq["return_pct"],
        })
    except Exception as e:
        logger.debug(f"[quant] equity snapshot {pool_id} failed: {e}")
    return stats


def _compute_position_size(pool: dict, price: float,
                           override_pct: float | None = None) -> int:
    """按 capital_allocation 算单仓股数.
    override_pct: 若给 (0-1), 覆盖 pool 里的 position_size_pct (给 LLM 决定模式用).
    use_guardrails=False 时不做 min/max clamp.
    """
    if price <= 0:
        return 0
    alloc = pool.get("capital_allocation") or {}
    cash = float(pool.get("cash", 0))
    if override_pct is not None:
        size_pct = max(0.0, min(1.0, float(override_pct)))
    else:
        size_pct = float(alloc.get("position_size_pct", 10)) / 100
    budget = cash * size_pct
    if alloc.get("use_guardrails"):
        min_size = float(alloc.get("min_position_size", 0))
        max_size = float(alloc.get("max_position_size", 0))
        if min_size > 0:
            budget = max(min_size, budget)
        if max_size > 0:
            budget = min(budget, max_size)
    qty = int(budget / price)
    return max(0, qty)


# ─── Daemon ─────────────────────────────────────────────────────────

def _daemon_loop():
    """Main daemon loop — tick every TICK_SEC, iterate enabled pools."""
    with _DAEMON_LOCK:
        _DAEMON_STATE["running"] = True
        _DAEMON_STATE["started_at"] = _store._now_iso()
        _DAEMON_STATE["last_error"] = None
    logger.info("quant-engine daemon started")
    while not _DAEMON_STOP.is_set():
        try:
            pools = _store.load_pools()
            enabled = [p for p in pools if p.get("enabled")]
            processed = 0
            for pool in enabled:
                try:
                    stats = _process_pool(pool)
                    processed += 1
                    if stats["actions"] > 0:
                        logger.info(f"[quant] pool {pool['id']} tick: {stats}")
                except Exception as e:
                    logger.warning(f"[quant] pool {pool['id']} tick error: {e}")
            now_ts = time.time()
            with _DAEMON_LOCK:
                _DAEMON_STATE["last_tick_at"] = _store._now_iso()
                _DAEMON_STATE["processed_pools_last_tick"] = processed
                _DAEMON_STATE["last_error"] = None
                _DAEMON_STATE["next_tick_at"] = datetime.fromtimestamp(
                    now_ts + TICK_SEC, tz=timezone.utc).isoformat()
        except Exception as e:
            logger.warning(f"[quant] daemon tick error: {e}")
            with _DAEMON_LOCK:
                _DAEMON_STATE["last_error"] = str(e)
        _DAEMON_STOP.wait(TICK_SEC)
    with _DAEMON_LOCK:
        _DAEMON_STATE["running"] = False
    logger.info("quant-engine daemon stopped")


def start_daemon() -> bool:
    """Idempotent start. Returns True if newly started."""
    global _DAEMON_THREAD
    with _DAEMON_LOCK:
        if _DAEMON_THREAD and _DAEMON_THREAD.is_alive():
            return False
        _DAEMON_STOP.clear()
        _DAEMON_THREAD = threading.Thread(target=_daemon_loop,
                                          name="quant-engine",
                                          daemon=True)
        _DAEMON_THREAD.start()
    return True


def stop_daemon() -> bool:
    """Stop daemon (best-effort). Returns True if it was running."""
    global _DAEMON_THREAD
    with _DAEMON_LOCK:
        if not _DAEMON_THREAD or not _DAEMON_THREAD.is_alive():
            return False
        _DAEMON_STOP.set()
    return True


def get_daemon_status() -> dict:
    with _DAEMON_LOCK:
        s = dict(_DAEMON_STATE)
    s["alive"] = bool(_DAEMON_THREAD and _DAEMON_THREAD.is_alive())
    s["tick_interval_sec"] = TICK_SEC
    return s


def tick_once(pool_id: str) -> dict:
    """Manual single-pool tick (bypasses daemon). For UI [▶ 手动 tick] button.
    与 daemon 共用每池锁: daemon 正在处理该池时最多等 5s, 等不到抛 RuntimeError."""
    pool = _store.get_pool(pool_id)
    if not pool:
        raise ValueError(f"pool {pool_id} not found")
    return _process_pool(pool, wait_sec=5)
