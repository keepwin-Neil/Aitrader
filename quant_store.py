"""
Quant Trading Pool store + fee/borrow calculators.

Data model (JSON, keep simple):
- quant_pools.json  : list of quant pools (accounts + rules + cash + positions)
- quant_events.json : rolling 200-entry event log for audit

Each quant pool = 一个独立交易账户 + 一套自动交易规则.
采用 top-N 策略选择 + LLM veto,
另外支持:
- 卖空 (allow_short + borrow cost)
- IB Tiered 手续费
- 单机 JSON 存储 (无 Postgres 依赖)
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
POOLS_FILE = BASE_DIR / "quant_pools.json"
EVENTS_FILE = BASE_DIR / "quant_events.json"

_EVENTS_MAX = 200  # rolling cap
_lock = threading.RLock()

# 运行时状态字段: 由 engine / toggle 维护, UI 编辑池配置 (upsert) 时原样保留
RUNTIME_FIELDS = (
    "cash", "positions", "history", "strategy_ranks_cache", "strategy_rank_skips", "equity_history",
    "acted_signals", "enabled", "last_run_at", "last_rebalance_at",
    "last_rebalance_attempt_at", "created_at",
)


# ─── ID + Time ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_pool_id() -> str:
    return "qp-" + uuid.uuid4().hex[:8]


def _new_pos_id(pool_id: str) -> str:
    return f"{pool_id}-pos-{uuid.uuid4().hex[:6]}"


def _new_event_id() -> str:
    return "ev-" + uuid.uuid4().hex[:10]


# ─── Persistence ────────────────────────────────────────────────────────

def _read_json(fp: Path, default):
    if not fp.exists():
        return default
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"quant_store read failed {fp.name}: {e}")
        return default


def _write_json(fp: Path, data) -> None:
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(fp)


def load_pools() -> list[dict]:
    with _lock:
        data = _read_json(POOLS_FILE, {"pools": []})
    return data.get("pools") or []


def save_pools(pools: list[dict]) -> None:
    with _lock:
        _write_json(POOLS_FILE, {"pools": pools, "updated_at": _now_iso()})


def load_events() -> list[dict]:
    with _lock:
        data = _read_json(EVENTS_FILE, {"events": []})
    return data.get("events") or []


def append_event(evt: dict) -> None:
    """Append event with rolling cap. Called from engine on every decision."""
    with _lock:
        data = _read_json(EVENTS_FILE, {"events": []})
        evts = data.get("events") or []
        evt.setdefault("id", _new_event_id())
        evt.setdefault("ts", _now_iso())
        evts.append(evt)
        # rolling cap: keep last N
        if len(evts) > _EVENTS_MAX:
            evts = evts[-_EVENTS_MAX:]
        _write_json(EVENTS_FILE, {"events": evts, "updated_at": _now_iso()})


def get_pool(pool_id: str) -> Optional[dict]:
    for p in load_pools():
        if p.get("id") == pool_id:
            return p
    return None


def upsert_pool(pool: dict) -> dict:
    """Create-or-update. If pool has no 'id', generate one + init defaults."""
    with _lock:
        pools = load_pools()
        if not pool.get("id"):
            pool["id"] = _new_pool_id()
            pool.setdefault("created_at", _now_iso())
            # Initial account state
            pool.setdefault("cash", float(pool.get("initial_capital", 100000)))
            pool.setdefault("positions", [])
            pool.setdefault("history", [])
            pool.setdefault("strategy_ranks_cache", {})
            pool.setdefault("strategy_rank_skips", {})
            pool.setdefault("equity_history", [])
            pool.setdefault("acted_signals", {})
            pool.setdefault("enabled", False)
            pool.setdefault("last_run_at", None)
            pool.setdefault("last_rebalance_at", None)
            pools.append(pool)
        else:
            idx = next((i for i, p in enumerate(pools) if p["id"] == pool["id"]), -1)
            if idx < 0:
                pools.append(pool)
            else:
                # Merge — preserve account state, replace configuration fields
                existing = pools[idx]
                pool_dict = dict(pool)
                # 保护运行时状态字段不被 UI 编辑覆盖
                # (enabled 由 /toggle 单独控制; equity_history / acted_signals 是
                #  daemon 累积的运行时数据, 编辑配置不能清掉)
                for keep in RUNTIME_FIELDS:
                    if keep in existing:
                        pool_dict[keep] = existing[keep]
                pool_dict["updated_at"] = _now_iso()
                pools[idx] = pool_dict
                pool = pool_dict
        save_pools(pools)
    return pool


def delete_pool(pool_id: str) -> bool:
    with _lock:
        pools = load_pools()
        new_pools = [p for p in pools if p.get("id") != pool_id]
        if len(new_pools) == len(pools):
            return False
        save_pools(new_pools)
    return True


_EQUITY_SNAPSHOT_MAX = 1000  # 每池最多存 1000 个快照 (滚动)


def append_equity_snapshot(pool_id: str, snapshot: dict) -> None:
    """Append equity snapshot to pool.equity_history (rolling cap)."""
    with _lock:
        pools = load_pools()
        for p in pools:
            if p.get("id") == pool_id:
                hist = p.setdefault("equity_history", [])
                snapshot.setdefault("ts", _now_iso())
                hist.append(snapshot)
                if len(hist) > _EQUITY_SNAPSHOT_MAX:
                    hist[:] = hist[-_EQUITY_SNAPSHOT_MAX:]
                save_pools(pools)
                return


def update_pool_fields(pool_id: str, **fields) -> Optional[dict]:
    """Update specific runtime fields (cash, positions, ...) atomically."""
    with _lock:
        pools = load_pools()
        for p in pools:
            if p.get("id") == pool_id:
                p.update(fields)
                p["updated_at"] = _now_iso()
                save_pools(pools)
                return p
    return None


# ─── 手续费: IB Tiered ────────────────────────────────────────────────

def compute_fee(price: float, quantity: int, config: dict) -> float:
    """
    IB Tiered (来自 CLAUDE.md feedback_backtest_rules):
      - $0.0067 / 股
      - 最低 $0.35 / 单
      - 最高 1% of notional
    可通过 config 覆盖任一值.

    Args:
      price: 成交单价
      quantity: 股数 (取绝对值, side 不影响费)
      config: fees 配置 dict, 可含:
        model: 'ib_tiered' | 'flat' | 'percentage'
        per_share_usd: 0.0067
        min_per_order_usd: 0.35
        max_pct_of_notional: 0.01
        flat_fee_usd: (仅 flat 模式)
        percentage_rate: (仅 percentage 模式)
    """
    model = (config or {}).get("model", "ib_tiered")
    qty = abs(int(quantity))
    if qty <= 0 or price <= 0:
        return 0.0
    notional = price * qty
    if model == "flat":
        return round(float((config or {}).get("flat_fee_usd", 1.0)), 4)
    if model == "percentage":
        rate = float((config or {}).get("percentage_rate", 0.001))
        return round(notional * rate, 4)
    # ib_tiered 默认
    per_share = float((config or {}).get("per_share_usd", 0.0067))
    min_fee = float((config or {}).get("min_per_order_usd", 0.35))
    max_pct = float((config or {}).get("max_pct_of_notional", 0.01))
    raw = qty * per_share
    max_fee = notional * max_pct
    return round(max(min_fee, min(raw, max_fee)), 4)


# ─── 卖空借券成本 ─────────────────────────────────────────────────────

def compute_borrow_cost(short_notional: float, days_held: float,
                        borrow_cost_pct_annual: float) -> float:
    """
    按年化借券利率算持有 N 天的成本.
    Args:
      short_notional: 卖空持仓的市值 (正数 = |quantity| × current_price)
      days_held: 已持有天数 (可小数, 如 0.5 天)
      borrow_cost_pct_annual: 年化借券费率 (%, 如 2.5 表示 2.5%)
    Returns:
      持有 days_held 天的借券费 (USD)
    """
    if short_notional <= 0 or days_held <= 0:
        return 0.0
    daily_rate = float(borrow_cost_pct_annual) / 100.0 / 365.0
    return round(float(short_notional) * daily_rate * float(days_held), 4)


# ─── PnL Accounting ───────────────────────────────────────────────────

def compute_position_pnl(pos: dict, current_price: float) -> dict:
    """
    Both LONG and SHORT PnL calculation.
    LONG:  pnl = (current - entry) × qty
    SHORT: pnl = (entry - current) × qty
    """
    side = pos.get("side", "LONG")
    entry = float(pos.get("entry_price") or pos.get("avg_price") or 0)
    qty = int(pos.get("quantity") or 0)
    if qty <= 0 or entry <= 0:
        return {"pnl": 0.0, "pnl_pct": 0.0, "market_value": 0.0}
    if side == "SHORT":
        pnl = (entry - current_price) * qty
        pnl_pct = (entry - current_price) / entry * 100
        market_value = current_price * qty  # short 侧仍按 current 计市值负载
    else:
        pnl = (current_price - entry) * qty
        pnl_pct = (current_price - entry) / entry * 100
        market_value = current_price * qty
    return {
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "market_value": round(market_value, 2),
    }


def compute_pool_equity(pool: dict, price_map: dict) -> dict:
    """
    权益 = 现金 + 所有 open LONG 市值 - 所有 open SHORT 回补负债 (现价 × 股数)

    记账口径 (与 open_position / close_position 一致):
      - LONG 开仓已从 cash 扣掉 entry×qty, 所以 cash + 现价×qty 自带未实现盈亏
      - SHORT 开仓把 entry×qty 计入 cash (借券卖出所得), 回补要花 现价×qty,
        所以 cash - 现价×qty 自带未实现盈亏
    unrealized 仅作展示字段, 不再重复加进 equity (以前双算了).
    已 closed 的仓不参与 (它们的盈亏已经结算进 cash).
    Return snapshot: cash / long_mv / short_mv / unrealized / equity
    """
    cash = float(pool.get("cash", 0))
    long_mv = 0.0
    short_mv = 0.0     # 卖空的回补负债 (现价 × 股数)
    unrealized = 0.0
    for pos in pool.get("positions") or []:
        if pos.get("status") != "open":
            continue
        sym = pos.get("symbol")
        p = float(price_map.get(sym) or pos.get("entry_price") or 0)
        pnl_info = compute_position_pnl(pos, p)
        unrealized += pnl_info["pnl"]
        if pos.get("side") == "SHORT":
            short_mv += pnl_info["market_value"]
        else:
            long_mv += pnl_info["market_value"]
    equity = cash + long_mv - short_mv
    return {
        "cash": round(cash, 2),
        "long_mv": round(long_mv, 2),
        "short_mv": round(short_mv, 2),
        "unrealized": round(unrealized, 2),
        "equity": round(equity, 2),
        "initial_capital": float(pool.get("initial_capital", 0)),
        "return_pct": round(
            (equity - float(pool.get("initial_capital", 0)))
            / max(float(pool.get("initial_capital", 1)), 1) * 100, 2),
    }


# ─── Position Ops (mutating) ────────────────────────────────────────

def open_position(pool_id: str, symbol: str, market: str, side: str,
                  quantity: int, price: float, strategy: str,
                  reason: str = "auto") -> Optional[dict]:
    """
    Open a new LONG or SHORT position.
    Deducts cash for LONG. For SHORT, no cash out — but proceeds are held
    (加入 cash), and settlement happens on close (需要回补).
    """
    side = side.upper()
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"invalid side: {side}")
    if int(quantity) <= 0 or float(price) <= 0:
        return None
    symbol = symbol.upper()
    # SHORT: 开仓时就把借券利率定下来并存到仓位上 (平仓直接用, 不再探测).
    # IB 探测可能阻塞数秒, 必须放在拿 _lock 之前.
    borrow_rate = None
    borrow_source = None
    if side == "SHORT":
        try:
            import quant_borrow
            pool0 = get_pool(pool_id) or {}
            pool_default = (pool0.get("short_config") or {}).get("borrow_cost_pct_annual")
            pool_default = float(pool_default) if pool_default is not None else None
            borrow_rate, borrow_source = quant_borrow.get_borrow_rate(
                symbol, pool_default, market=market)
        except Exception as e:
            logger.debug(f"borrow rate lookup {symbol}: {e}")
    with _lock:
        pool = get_pool(pool_id)
        if not pool:
            return None
        # 同一 symbol 已有 open 仓 → 不重复开 (daemon 与手动 tick 并发 / 同一信号重复触发)
        dup = next((p for p in (pool.get("positions") or [])
                    if p.get("symbol") == symbol and p.get("status") == "open"), None)
        if dup:
            append_event({
                "type": "SKIP", "pool_id": pool_id, "symbol": symbol, "market": market,
                "action": "BUY" if side == "LONG" else "SHORT",
                "reason": f"already_open {dup.get('side')} {dup.get('quantity')} @ {dup.get('entry_price')} ({dup.get('id')})",
            })
            return None
        fee = compute_fee(price, quantity, pool.get("fees") or {})
        notional = price * quantity
        if side == "LONG":
            need = notional + fee
            if pool.get("cash", 0) < need:
                append_event({
                    "type": "SKIP",
                    "pool_id": pool_id, "symbol": symbol, "market": market,
                    "action": "BUY", "reason": f"insufficient_cash need={need:.2f} have={pool.get('cash',0):.2f}",
                })
                return None
            pool["cash"] = round(pool["cash"] - need, 2)
        else:  # SHORT — 卖空: 现金入账, 但欠券, 到平时回补
            pool["cash"] = round(pool.get("cash", 0) + notional - fee, 2)
        pos = {
            "id": _new_pos_id(pool_id),
            "symbol": symbol.upper(),
            "market": market.upper(),
            "side": side,
            "quantity": int(quantity),
            "entry_price": float(price),
            "entry_time": _now_iso(),
            "strategy": strategy or "",
            "fees_paid": fee,
            "status": "open",
        }
        if side == "SHORT":
            pos["borrow_rate_pct"] = borrow_rate
            pos["borrow_source"] = borrow_source
        pool.setdefault("positions", []).append(pos)
        pool["last_run_at"] = _now_iso()
        pools = load_pools()
        for i, p in enumerate(pools):
            if p["id"] == pool_id:
                pools[i] = pool
                break
        save_pools(pools)
        append_event({
            "type": "FILL", "pool_id": pool_id, "position_id": pos["id"],
            "symbol": symbol, "market": market,
            "action": "BUY" if side == "LONG" else "SHORT",
            "side": side, "quantity": quantity, "price": price,
            "fee": fee, "reason": reason, "strategy": strategy,
        })
    return pos


def close_position(pool_id: str, position_id: str, price: float,
                   quantity: Optional[int] = None,
                   reason: str = "manual") -> Optional[dict]:
    """
    Close (fully or partial). LONG: sell → cash in. SHORT: buy to cover → cash out.
    Returns the trade record.
    """
    with _lock:
        pool = get_pool(pool_id)
        if not pool:
            return None
        pos = next((p for p in pool.get("positions") or []
                    if p["id"] == position_id and p.get("status") == "open"), None)
        if not pos:
            return None
        side = pos.get("side", "LONG")
        close_qty = int(quantity) if quantity else int(pos["quantity"])
        close_qty = min(close_qty, int(pos["quantity"]))
        entry = float(pos["entry_price"])
        fee = compute_fee(price, close_qty, pool.get("fees") or {})
        # 借券成本 (仅 SHORT)
        borrow_fee = 0.0
        if side == "SHORT":
            entry_dt = pos.get("entry_time", _now_iso())
            try:
                entered = datetime.fromisoformat(entry_dt.replace("Z", "+00:00"))
                days = max(0, (datetime.now(timezone.utc) - entered).total_seconds() / 86400)
            except Exception:
                days = 0
            # 借券利率: 优先用开仓时锁定在仓位上的值; 老仓位没有就查
            # seeded / user cache (probe_ib=False: 持 _lock 期间不做 IB 网络往返)
            borrow_rate = pos.get("borrow_rate_pct")
            borrow_source = pos.get("borrow_source") or "position"
            if borrow_rate is None:
                pool_default = (pool.get("short_config") or {}).get("borrow_cost_pct_annual")
                pool_default = float(pool_default) if pool_default is not None else None
                try:
                    import quant_borrow
                    borrow_rate, borrow_source = quant_borrow.get_borrow_rate(
                        pos["symbol"], pool_default, probe_ib=False)
                except Exception:
                    borrow_rate = pool_default if pool_default is not None else 2.5
                    borrow_source = "pool_default"
            borrow_rate = float(borrow_rate)
            borrow_fee = compute_borrow_cost(price * close_qty, days, borrow_rate)
        if side == "LONG":
            pnl = round((price - entry) * close_qty, 2)
            pool["cash"] = round(pool.get("cash", 0) + price * close_qty - fee, 2)
        else:  # SHORT — 回补: 花现金买回
            pnl = round((entry - price) * close_qty - borrow_fee, 2)
            pool["cash"] = round(pool.get("cash", 0) - price * close_qty - fee - borrow_fee, 2)
        pnl_pct = round((pnl / (entry * close_qty)) * 100, 2) if entry * close_qty > 0 else 0.0
        trade = {
            "id": pos["id"],
            "symbol": pos["symbol"], "market": pos["market"],
            "side": side, "strategy": pos.get("strategy", ""),
            "quantity": close_qty, "entry_price": entry, "entry_time": pos["entry_time"],
            "exit_price": float(price), "exit_time": _now_iso(),
            "fees_paid": round(pos.get("fees_paid", 0) + fee, 4),
            "borrow_fee": borrow_fee,
            "pnl": pnl, "pnl_pct": pnl_pct,
            "reason": reason,
        }
        if side == "SHORT":
            trade["borrow_rate_pct"] = borrow_rate
            trade["borrow_source"] = borrow_source
        pool.setdefault("history", []).append(trade)
        # 数量减少 / 全平
        if close_qty >= int(pos["quantity"]):
            pos["status"] = "closed"
            pos["exit_price"] = price
            pos["exit_time"] = trade["exit_time"]
            pos["pnl"] = pnl
        else:
            pos["quantity"] = int(pos["quantity"]) - close_qty
        pools = load_pools()
        for i, p in enumerate(pools):
            if p["id"] == pool_id:
                pools[i] = pool
                break
        save_pools(pools)
        append_event({
            "type": "FILL", "pool_id": pool_id, "position_id": pos["id"],
            "symbol": pos["symbol"], "market": pos["market"],
            "action": "SELL" if side == "LONG" else "COVER",
            "side": side, "quantity": close_qty, "price": price,
            "fee": fee, "borrow_fee": borrow_fee,
            "pnl": pnl, "pnl_pct": pnl_pct,
            "reason": reason,
        })
    return trade
