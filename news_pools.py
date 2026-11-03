"""News-driven stock pools.

Two persistent pools auto-populated by news scoring:

- 📅 **Daily pool** (`news_pool_daily.json`): stocks scoring above the threshold
  on the current local date. Resets at 00:00 local time.
- 📆 **Weekly pool** (`news_pool_weekly.json`): stocks scoring above the threshold
  during the current Monday-Sunday week. Resets every Monday 00:00.

Behaviour:
- Each pool dedupes by (symbol, market). Once a stock is in the pool, it stays
  for the rest of the period regardless of subsequent scores.
- The scorer can call `already_in_today_pool(symbol, market)` BEFORE making a
  costly LLM call — if true, it can reuse the cached score and skip the call.
  This is the "已经在股票池的，不再参与评分" UX requested by the user.

Threshold default: 70 (corresponds to "bullish" or stronger). Configurable via
`news_buffer_config.json -> auto_pools.threshold`.

NOT USED FOR TRADING. Pure tracking. Strategy-side selector pipeline reads
these pools as candidate filters if explicitly configured.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DAILY_PATH = Path(__file__).parent / "news_pool_daily.json"
WEEKLY_PATH = Path(__file__).parent / "news_pool_weekly.json"
CONFIG_PATH = Path(__file__).parent / "news_buffer_config.json"

DEFAULT_THRESHOLD = 70  # score >= this → goes into pool
_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────
# date helpers
# ──────────────────────────────────────────────────────────────────────────
def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _week_start() -> str:
    """Monday of current week (local)."""
    d = datetime.now()
    return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def _read_threshold() -> int:
    """Read threshold from news_buffer_config.json (or default 70)."""
    if not CONFIG_PATH.exists():
        return DEFAULT_THRESHOLD
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        ap = cfg.get("auto_pools") or {}
        return int(ap.get("threshold") or DEFAULT_THRESHOLD)
    except Exception:
        return DEFAULT_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────
# pool I/O (with auto-reset on date change)
# ──────────────────────────────────────────────────────────────────────────
def _read_pool(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"news_pools read {path.name} failed: {e}")
        return {}


def _write_pool(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _ensure_daily_fresh(pool: dict) -> dict:
    """Reset if pool's date != today."""
    if pool.get("date") != _today():
        return {"date": _today(), "items": []}
    return pool


def _ensure_weekly_fresh(pool: dict) -> dict:
    """Reset if pool's week_start != current Monday."""
    if pool.get("week_start") != _week_start():
        return {"week_start": _week_start(), "items": []}
    return pool


def get_pools() -> tuple[dict, dict]:
    """Return (daily, weekly) pools, auto-resetting if date passed."""
    with _lock:
        raw_daily = _read_pool(DAILY_PATH)
        daily = _ensure_daily_fresh(raw_daily)
        if daily is not raw_daily:  # actually reset
            _write_pool(DAILY_PATH, daily)
        raw_weekly = _read_pool(WEEKLY_PATH)
        weekly = _ensure_weekly_fresh(raw_weekly)
        if weekly is not raw_weekly:
            _write_pool(WEEKLY_PATH, weekly)
        return daily, weekly


def _pool_has(pool: dict, symbol: str, market: str) -> bool:
    sym = (symbol or "").upper()
    mkt = (market or "").upper()
    for it in pool.get("items") or []:
        if (it.get("symbol") or "").upper() == sym and \
           (it.get("market") or "").upper() == mkt:
            return True
    return False


def already_in_today_pool(symbol: str, market: str) -> bool:
    """Cheap cache check used by scorer to skip LLM calls.

    Returns True if (symbol, market) is already in today's daily pool.
    """
    daily, _ = get_pools()
    return _pool_has(daily, symbol, market)


def get_pool_entry(symbol: str, market: str, *, source: str = "daily") -> Optional[dict]:
    """Look up a single (symbol, market) entry in daily or weekly pool."""
    daily, weekly = get_pools()
    pool = daily if source == "daily" else weekly
    sym = (symbol or "").upper()
    mkt = (market or "").upper()
    for it in pool.get("items") or []:
        if (it.get("symbol") or "").upper() == sym and \
           (it.get("market") or "").upper() == mkt:
            return it
    return None


def maybe_add(symbol: str, market: str, score: Optional[int],
              label: Optional[str], reasoning: str = "",
              n_news: int = 0) -> bool:
    """Add to both daily and weekly pools if score >= threshold AND not present.

    Returns True if at least one pool was updated.
    """
    if score is None:
        return False
    threshold = _read_threshold()
    if int(score) < threshold:
        return False
    sym = (symbol or "").upper()
    mkt = (market or "").upper()
    if not sym:
        return False

    item = {
        "symbol": sym,
        "market": mkt,
        "score": int(score),
        "label": label or "",
        "reasoning": (reasoning or "")[:200],
        "n_news": int(n_news or 0),
        "added_at": datetime.now().isoformat(),
    }

    updated = False
    with _lock:
        # Reload + rewrite atomically
        daily = _ensure_daily_fresh(_read_pool(DAILY_PATH))
        weekly = _ensure_weekly_fresh(_read_pool(WEEKLY_PATH))
        if not _pool_has(daily, sym, mkt):
            daily.setdefault("items", []).append(item)
            _write_pool(DAILY_PATH, daily)
            updated = True
        if not _pool_has(weekly, sym, mkt):
            weekly.setdefault("items", []).append(item)
            _write_pool(WEEKLY_PATH, weekly)
            updated = True
    if updated:
        logger.info(f"news_pools: added {sym}/{mkt} score={score} label={label}")
    return updated


def clear_pool(which: str = "daily") -> int:
    """Manually wipe a pool. Returns count of items removed."""
    with _lock:
        if which == "daily":
            pool = _read_pool(DAILY_PATH)
            n = len(pool.get("items") or [])
            _write_pool(DAILY_PATH, {"date": _today(), "items": []})
            return n
        elif which == "weekly":
            pool = _read_pool(WEEKLY_PATH)
            n = len(pool.get("items") or [])
            _write_pool(WEEKLY_PATH, {"week_start": _week_start(), "items": []})
            return n
        else:
            raise ValueError(f"unknown pool: {which}")


def get_threshold() -> int:
    return _read_threshold()


def set_threshold(threshold: int) -> int:
    """Persist threshold to news_buffer_config.json -> auto_pools.threshold.
    Delegates to news_auto_scanner's atomic writer so concurrent updaters of
    auto_scan/max_days don't get clobbered."""
    threshold = max(0, min(int(threshold), 100))
    from news_auto_scanner import _lock as _cfg_lock, _read_config, _write_config
    with _cfg_lock:
        cfg = _read_config()
        cfg.setdefault("auto_pools", {})["threshold"] = threshold
        _write_config(cfg)
    return threshold
