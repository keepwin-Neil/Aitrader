"""Stock research reports HTTP endpoints.
Mounts under /stock-reports/*.
"""
from fastapi import APIRouter, HTTPException, Body, Query
from typing import Optional

import stock_report as _sr

router = APIRouter()


@router.get("/stock-reports/summary")
def list_summary():
    """Return the roster of all cached reports (weekly + daily) for the UI table."""
    weekly = _sr.load_weekly()
    daily = _sr.load_daily()
    # 合并 symbol 键
    symbols = set(weekly.keys()) | set(daily.keys())
    out = []
    for sym in sorted(symbols):
        w = weekly.get(sym)
        sym_daily = daily.get(sym) or {}
        latest_daily = sym_daily.get(max(sym_daily.keys())) if sym_daily else None
        out.append({
            "symbol": sym,
            "weekly": ({
                "generated_at": w.get("generated_at"),
                "bias": w.get("bias"),
                "confidence": w.get("confidence"),
                "time_horizon": w.get("time_horizon"),
                "fresh": _sr.is_weekly_fresh(w),
                "provider": w.get("provider"),
            } if w else None),
            "daily_latest": ({
                "generated_at": latest_daily.get("generated_at"),
                "for_date": latest_daily.get("for_date"),
                "thesis_still_valid": latest_daily.get("thesis_still_valid"),
                "flag": latest_daily.get("flag"),
                "fresh": _sr.is_daily_fresh(latest_daily),
                "provider": latest_daily.get("provider"),
            } if latest_daily else None),
            "daily_history_count": len(sym_daily),
        })
    return {"reports": out, "n_weekly": len(weekly), "n_daily": len(daily)}


@router.get("/stock-reports/{symbol}")
def get_symbol(symbol: str):
    """Full report bundle for one symbol (weekly + all daily history)."""
    weekly = _sr.get_weekly(symbol)
    daily_all = _sr.load_daily().get(symbol.upper()) or {}
    return {
        "symbol": symbol.upper(),
        "weekly": weekly,
        "daily": daily_all,
    }


@router.post("/stock-reports/generate")
def generate(body: dict = Body(...)):
    """Batch-generate reports.

    Body:
      {
        "symbols": ["NVDA", "AAPL", ...],   REQUIRED
        "market": "US",                      default "US"
        "kind": "weekly" | "daily" | "both", default "weekly"
        "provider": "claude" | "openai" | "gemini",  default "claude"
        "concurrency": 1-8                   default 4
      }
    """
    symbols = body.get("symbols") or []
    if not isinstance(symbols, list) or not symbols:
        raise HTTPException(400, "symbols must be non-empty list")
    symbols = [str(s).upper().strip() for s in symbols if str(s).strip()]
    market = str(body.get("market") or "US").upper()
    kind = str(body.get("kind") or "weekly").lower()
    provider = str(body.get("provider") or "claude").lower()
    concurrency = int(body.get("concurrency") or 4)

    if kind not in ("weekly", "daily", "both"):
        raise HTTPException(400, f"invalid kind: {kind}")

    if kind == "both":
        weekly_result = _sr.generate_batch(symbols, market, "weekly",
                                            provider, concurrency)
        daily_result = _sr.generate_batch(symbols, market, "daily",
                                           provider, concurrency)
        return {"ok": True, "weekly": weekly_result, "daily": daily_result}

    result = _sr.generate_batch(symbols, market, kind, provider, concurrency)
    return {"ok": True, **result}


@router.delete("/stock-reports/{symbol}/weekly")
def delete_symbol_weekly(symbol: str):
    ok = _sr.delete_weekly(symbol)
    return {"ok": ok}


@router.delete("/stock-reports/{symbol}/daily")
def delete_symbol_daily(symbol: str, date: Optional[str] = Query(None)):
    ok = _sr.delete_daily(symbol, date)
    return {"ok": ok}
