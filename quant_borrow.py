"""
Stock borrow cost (short interest) lookup + cache.

Data source strategy:
  1. Per-symbol cache from .cache/borrow_rates.json (allows manual override)
  2. Static seeded rates for common heavy-short names (biotech, meme stocks, EV)
  3. Fallback = pool.short_config.borrow_cost_pct_annual (default 2.5%)

IB Gateway 实盘会返回真实借券率 (reqShortableShares + rebate rate),
但 quant 是 paper 模拟, IB Gateway 拉的 rate 无法反映"如果真交易"的价格,
用户偏好的 config 值优先; cache 只提供 sensible defaults 让不同股票
有不同成本 (SPY 便宜, meme 贵).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent / ".cache" / "borrow_rates.json"

# 常见"贵借"名单 — 年化 %. 数据来自公开做空成本报告 (2025 年样本估算).
# 不是所有历史数据, 定期人工/半自动更新.
_SEEDED_RATES = {
    # ── ETF / 大盘 (超便宜) ──────────────
    "SPY": 0.20, "QQQ": 0.20, "IWM": 0.30, "DIA": 0.20,
    # ── 大型成熟股 ~ 0.25%
    "AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25,
    "META": 0.25, "NVDA": 0.25, "JPM": 0.25, "V": 0.25, "MA": 0.25,
    # ── 中型科技/AI (~1-2%)
    "AMD": 0.75, "ARM": 1.5, "PLTR": 1.0, "SNOW": 1.2, "COIN": 2.0,
    "SHOP": 0.8, "CRWV": 3.5, "AVGO": 0.6,
    # ── 电动车 / 太阳能 (~2-4%)
    "TSLA": 1.5, "RIVN": 3.5, "LCID": 5.0, "NIO": 4.0, "XPEV": 4.5,
    # ── AI / 高波动小盘 (~4-8%)
    "SMCI": 3.5, "IONQ": 4.0, "RGTI": 6.0, "QBTS": 8.0,
    "TEM": 5.0, "RXRX": 6.0, "GEMI": 8.0,
    # ── Meme / 热门空头 (~10-30%)
    "GME": 15.0, "AMC": 20.0, "BB": 3.5, "BBBY": 25.0,
    # ── 生物医药 (~5-15%)
    "MRNA": 3.5, "BIIB": 2.0, "REGN": 1.5,
    # ── 中概股 (~2-6%)
    "BABA": 2.5, "BIDU": 3.0, "JD": 3.0, "PDD": 3.0, "NIU": 4.5,
    "BILI": 5.0, "TCEHY": 2.0,
    # ── 稀土/矿产/核能 (~3-8%)
    "USAR": 6.0, "UUUU": 5.0, "UAMY": 8.0, "CCJ": 2.5, "LEU": 4.0,
    "MP": 3.5, "TMQ": 6.0, "CRML": 8.0,
    # ── 能源 (~1-3%)
    "VST": 2.0, "CEG": 1.5, "NEE": 1.0, "BE": 4.0, "OKLO": 6.0,
    # ── 新股 IPO 高热度 (~10-20%)
    "CRCL": 15.0, "KLAR": 12.0, "AMBQ": 18.0, "AUR": 6.0, "AVEX": 10.0,
}

DEFAULT_RATE = 2.5
HTB_RATE = 25.0                # Hard-to-borrow 兜底率 (IB 标记 shortable=False)
_cache_loaded = False
_in_memory: dict = {}
_ib_shortable_cache: dict = {}  # {symbol: (shortable_bool, ts)}
_IB_CACHE_TTL_SEC = 3600         # IB 探测缓存 1 小时


def _load_cache() -> None:
    global _cache_loaded, _in_memory
    if _cache_loaded:
        return
    _cache_loaded = True
    _in_memory = dict(_SEEDED_RATES)
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            rates = data.get("rates") or {}
            # user cache overrides seed
            _in_memory.update({k.upper(): float(v) for k, v in rates.items() if v is not None})
        except Exception as e:
            logger.warning(f"borrow_rates cache read failed: {e}")


HTB_SHARES_THRESHOLD = 1000   # IB 可借股数低于此视为 hard-to-borrow


def _probe_ib_shortable(symbol: str, market: str = "US") -> Optional[bool]:
    """
    Query IB Gateway for whether the stock is currently shortable.
    Returns True if broadly shortable, False if HTB (hard-to-borrow), None if
    IB unavailable / query failed / 非美股 (目前只探测 US: SMART/USD 合约).

    数据口径 (ib_insync): generic tick 236 "Shortable" 回来的是 tick 89
    shortableShares = 当前可借股数 (不是 0-3 的等级码). 可借股数 < HTB_SHARES_THRESHOLD
    视为 HTB. 拿不到 (NaN) → None 让上层退到 seeded (best-effort).
    Cache 1h.
    """
    import time as _time
    sym = symbol.upper().strip()
    if (market or "US").upper() != "US":
        return None
    cached = _ib_shortable_cache.get(sym)
    if cached and (_time.time() - cached[1]) < _IB_CACHE_TTL_SEC:
        return cached[0]
    try:
        from ib_gateway import ib_enabled, _SlotContext
        if not ib_enabled():
            return None
        from ib_insync import Stock
        contract = Stock(sym, "SMART", "USD")

        async def _do(ib):
            import asyncio
            await ib.qualifyContractsAsync(contract)
            # generic tick 236 = Shortable — 需 reqMktData 显式请求,
            # snapshot 不包含 generic ticks. 用 streaming + 短等待.
            t = ib.reqMktData(contract, genericTickList="236",
                              snapshot=False, regulatorySnapshot=False)
            # 给 tick 236 到达时间 (通常 <500ms), 最多等 2s
            for _ in range(20):
                await asyncio.sleep(0.1)
                import math as _m
                v = getattr(t, "shortableShares", float("nan"))
                try:
                    if not _m.isnan(float(v)):
                        break
                except (TypeError, ValueError):
                    continue
            # 拿到 (或超时) 后取消 streaming
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
            return t

        with _SlotContext(timeout=5) as slot:
            t = slot.submit(_do, timeout=10)
        if t is None:
            return None
        try:
            shortable_shares = float(getattr(t, "shortableShares", "nan"))
        except (TypeError, ValueError):
            return None
        import math
        if math.isnan(shortable_shares):
            return None
        result = shortable_shares >= HTB_SHARES_THRESHOLD
    except Exception as e:
        logger.debug(f"IB shortable probe {sym} failed: {e}")
        return None
    _ib_shortable_cache[sym] = (result, _time.time())
    return result


def get_borrow_rate(symbol: str, pool_default: Optional[float] = None,
                    probe_ib: bool = True, market: str = "US") -> tuple[float, str]:
    """
    Return (annual_rate_pct, source).
    Lookup order:
      1. IB probe (if probe_ib, 仅 US): shortable → seeded/default, HTB → HTB_RATE
      2. User cache (Phase 5 覆盖)
      3. Seeded static (Phase 5 表)
      4. pool_default (显式给了就用, 包括 0.0) / DEFAULT_RATE
    source: 'ib_htb' / 'user_cache' / 'seeded' / 'pool_default' / 'global_default'
    """
    _load_cache()
    sym = (symbol or "").upper().strip()
    if pool_default is None:
        fallback = (DEFAULT_RATE, "global_default")
    else:
        fallback = (float(pool_default), "pool_default")
    if not sym:
        return fallback

    # (1) IB probe HTB check — 只影响 HTB 判定
    if probe_ib:
        ib_shortable = _probe_ib_shortable(sym, market=market)
        if ib_shortable is False:  # 明确 HTB
            return (HTB_RATE, "ib_htb")

    # (2) User override / seeded
    if sym in _in_memory:
        if sym in _SEEDED_RATES and _in_memory[sym] == _SEEDED_RATES[sym]:
            return (_in_memory[sym], "seeded")
        return (_in_memory[sym], "user_cache")

    # (3+4) fallback
    return fallback


def save_rate(symbol: str, rate_pct: float) -> None:
    """Persist user override to .cache/borrow_rates.json."""
    _load_cache()
    sym = symbol.upper().strip()
    _in_memory[sym] = float(rate_pct)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # user_cache = 内存 minus seeded (只存 override 部分)
    user_overrides = {k: v for k, v in _in_memory.items()
                      if k not in _SEEDED_RATES or _SEEDED_RATES[k] != v}
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rates": user_overrides,
        "seeded_count": len(_SEEDED_RATES),
    }
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CACHE_FILE)


def list_all() -> dict:
    """Return snapshot of all rates (seeded + overrides) for UI.

    sources: 每个 symbol 是内置 seed 还是用户改过的 override。
    以前只有 seeded_count / override_count 两个聚合数, 逐 symbol 的来源
    只在 GET /quant/borrow-rates/{symbol} 里才有 —— 于是利率表的「来源」列
    没有数据可读, 前端拿 symbol 字符串长度硬凑了个占位符, 65 行全印「存」。
    """
    _load_cache()
    sources = {
        k: ("override" if (k not in _SEEDED_RATES or _SEEDED_RATES[k] != v) else "seeded")
        for k, v in _in_memory.items()
    }
    return {
        "rates": dict(_in_memory),
        "sources": sources,
        "seeded_count": len(_SEEDED_RATES),
        "override_count": sum(1 for v in sources.values() if v == "override"),
        "default_pct_annual": DEFAULT_RATE,
    }
