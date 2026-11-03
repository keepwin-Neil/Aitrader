"""Stock research reports · weekly thesis + daily update.

Two report kinds:
- Weekly: 500-800 字 结构性论点, 有效期 7 天. 存 .cache/stock_reports_weekly.json
- Daily : 100-200 字 状态更新, 引用最近周报. 存 .cache/stock_reports_daily.json

调用路径 (CLI-first, API fallback, 与 llm_evaluator 保持一致):
  build_prompt → core_engine._call_via_cli / _call_via_api → parse JSON → save

Data sources (backend prefetch, LLM 不联网):
- K 线: backtest.get_ohlcv (local_only)
- 财务/行业/下次财报: yfinance Ticker.info / .calendar
- 新闻 headlines: news_provider.get_stock_news_unified
- 元信号: news_pools 里是否在热点池 (加一句"该股本周/今日在热点池")
"""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WEEKLY_PATH = Path(__file__).parent / ".cache" / "stock_reports_weekly.json"
DAILY_PATH = Path(__file__).parent / ".cache" / "stock_reports_daily.json"
_lock = threading.Lock()

WEEKLY_VALID_DAYS = 7
DAILY_VALID_HOURS = 24


# ─── System prompts ─────────────────────────────────────────────────

_WEEKLY_SYSTEM_PROMPT = """You are a stock research analyst producing a weekly thesis for a quant fund's
LLM veto layer. You get: symbol, price history, fundamentals, recent news.

Write a structured thesis. Return ONLY JSON, no prose:
{
  "thesis": "500-800 char narrative covering current situation, why hold/short, expected path",
  "drivers": ["3-5 bullet key upside/downside drivers"],
  "risks": ["3-5 bullet key risks that could invalidate the thesis"],
  "valuation": "one-line: P/E, growth, sector-relative",
  "sector_context": "one-line: sector regime, peers behavior",
  "bias": "long" | "short" | "neutral",
  "confidence": 0.0 to 1.0,
  "time_horizon": "3-6mo" | "1-2mo" | "2-4wk"
}

Guidelines:
- If news is thin, say so in the thesis and lower confidence.
- Bias is your overall directional call over the horizon.
- Confidence reflects data quality + conviction, not certainty."""


_DAILY_SYSTEM_PROMPT = """You are a stock daily-update writer for a quant fund. You get: today's price
action, news, and the prior weekly thesis. Write a short daily update.

Return ONLY JSON:
{
  "delta_summary": "80-150 char: what happened today and why",
  "thesis_still_valid": true | false,
  "notes": "100-200 char: any tomorrow signals, sector changes, revised outlook",
  "flag": null | "earnings_tmr" | "ex_div_tmr" | "news_spike" | "thesis_broken" | "gap_move"
}

Guidelines:
- If thesis is broken by today's action (e.g. earnings miss when thesis was long), set thesis_still_valid=false and flag='thesis_broken'.
- If tomorrow has earnings, ex-div, or a scheduled event, flag it.
- Keep notes actionable — what should the veto layer weigh."""


# ─── Storage ─────────────────────────────────────────────────────────

def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"stock_report read {path.name}: {e}")
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_weekly() -> dict:
    with _lock:
        return _read(WEEKLY_PATH)


def load_daily() -> dict:
    with _lock:
        return _read(DAILY_PATH)


def get_weekly(symbol: str) -> Optional[dict]:
    return load_weekly().get(symbol.upper())


def get_latest_daily(symbol: str) -> Optional[dict]:
    """Return the newest daily note for symbol, or None."""
    sym_daily = load_daily().get(symbol.upper()) or {}
    if not sym_daily:
        return None
    latest_date = max(sym_daily.keys())
    return sym_daily.get(latest_date)


def is_weekly_fresh(report: Optional[dict]) -> bool:
    if not report:
        return False
    try:
        gen = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - gen).total_seconds() / 86400
        return age_days < WEEKLY_VALID_DAYS
    except Exception:
        return False


def is_daily_fresh(report: Optional[dict]) -> bool:
    if not report:
        return False
    try:
        gen = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - gen).total_seconds() / 3600
        return age_hours < DAILY_VALID_HOURS
    except Exception:
        return False


# ─── Data fetchers ──────────────────────────────────────────────────

def _fetch_context(symbol: str, market: str) -> dict:
    """Prefetch all context for the LLM report."""
    ctx = {"symbol": symbol.upper(), "market": market.upper()}

    # K 线 (最近 60 根日线用于周报, 5 根用于日报)
    # 注: backtest.get_ohlcv 返回小写列名 (open/high/low/close/volume).
    def _col(row, *names):
        for n in names:
            if n in row:
                return row[n]
        return None
    try:
        import backtest as _bt
        df = _bt.get_ohlcv(symbol, market, interval="1d",
                           period="3mo", local_only=True)
        if df is not None and not df.empty:
            tail60 = df.tail(60)
            ctx["klines"] = [
                {
                    "date": (idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]),
                    "o": round(float(_col(row, "open", "Open") or 0), 2),
                    "h": round(float(_col(row, "high", "High") or 0), 2),
                    "l": round(float(_col(row, "low", "Low") or 0), 2),
                    "c": round(float(_col(row, "close", "Close") or 0), 2),
                    "v": int(_col(row, "volume", "Volume") or 0),
                }
                for idx, row in tail60.iterrows()
            ]
            # 20 日相对 SPY 强度
            try:
                spy = _bt.get_ohlcv("SPY", "US", interval="1d",
                                    period="3mo", local_only=True)
                if spy is not None and not spy.empty:
                    close_col = "close" if "close" in df.columns else "Close"
                    spy_close = "close" if "close" in spy.columns else "Close"
                    sym_ret = df.tail(20)[close_col].iloc[-1] / df.tail(20)[close_col].iloc[0] - 1
                    spy_ret = spy.tail(20)[spy_close].iloc[-1] / spy.tail(20)[spy_close].iloc[0] - 1
                    ctx["relative_strength_20d"] = round((sym_ret - spy_ret) * 100, 2)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"kline fetch {symbol} failed: {e}")

    # yfinance 财务/行业/下次财报
    try:
        import yfinance as yf
        ticker_sym = symbol if market.upper() == "US" else symbol
        t = yf.Ticker(ticker_sym)
        info = t.info or {}
        ctx["fundamentals"] = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "beta": info.get("beta"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
        }
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    earnings = cal.get("Earnings Date") or cal.get("earnings_date") or []
                elif hasattr(cal, "loc"):
                    earnings = cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
                else:
                    earnings = []
                if earnings:
                    first = earnings[0]
                    if hasattr(first, "strftime"):
                        ctx["next_earnings"] = first.strftime("%Y-%m-%d")
                    else:
                        ctx["next_earnings"] = str(first)[:10]
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"yfinance fetch {symbol} failed: {e}")

    # 新闻 (最近 7 天用于周报, 24h 用于日报; 都拿 7d, prompt 侧筛)
    try:
        from news_provider import get_stock_news_unified
        news = get_stock_news_unified(symbol, market, window_hours=168) or []
        ctx["news"] = [
            {
                "title": (n.get("title") or "").strip()[:200],
                "source": n.get("source") or "",
                "published_at": n.get("published_at") or "",
                "summary": (n.get("summary") or "").strip()[:300],
            }
            for n in news[:20]
        ]
    except Exception as e:
        logger.debug(f"news fetch {symbol} failed: {e}")
        ctx["news"] = []

    # news_pool 元信号 (今日池 / 本周池)
    try:
        import news_pools
        ctx["in_daily_news_pool"] = bool(news_pools.get_pool_entry(symbol, market, source="daily"))
        ctx["in_weekly_news_pool"] = bool(news_pools.get_pool_entry(symbol, market, source="weekly"))
    except Exception as e:
        logger.debug(f"news_pools lookup {symbol} failed: {e}")
        ctx["in_daily_news_pool"] = False
        ctx["in_weekly_news_pool"] = False

    return ctx


# ─── Prompt builders ────────────────────────────────────────────────

def _build_weekly_user_prompt(ctx: dict) -> str:
    lines = [
        f"Symbol: {ctx['symbol']} · Market: {ctx['market']}",
        f"Report date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]
    if ctx.get("fundamentals"):
        f = ctx["fundamentals"]
        lines.append("Fundamentals:")
        if f.get("sector"):
            lines.append(f"  Sector: {f['sector']} · Industry: {f.get('industry','?')}")
        if f.get("market_cap"):
            lines.append(f"  Market cap: ${f['market_cap']/1e9:.1f}B")
        if f.get("pe_trailing"):
            lines.append(f"  P/E trailing: {f['pe_trailing']:.1f} · forward: {f.get('pe_forward') or '?'}")
        if f.get("revenue_growth_yoy") is not None:
            lines.append(f"  Revenue growth YoY: {f['revenue_growth_yoy']*100:+.1f}%")
        if f.get("profit_margin") is not None:
            lines.append(f"  Profit margin: {f['profit_margin']*100:+.1f}%")
        if f.get("beta") is not None:
            lines.append(f"  Beta: {f['beta']:.2f}")
        if f.get("short_percent_of_float") is not None:
            lines.append(f"  Short interest: {f['short_percent_of_float']*100:.1f}% of float")
        lines.append("")
    if ctx.get("next_earnings"):
        lines.append(f"Next earnings: {ctx['next_earnings']}")
        lines.append("")
    if ctx.get("klines"):
        klines = ctx["klines"]
        last = klines[-1]
        first_60 = klines[0]
        ret60 = (last["c"] - first_60["c"]) / first_60["c"] * 100
        recent20 = klines[-20:]
        hi20 = max(k["h"] for k in recent20)
        lo20 = min(k["l"] for k in recent20)
        lines.append(f"Price ({len(klines)} daily bars):")
        lines.append(f"  Current: ${last['c']:.2f}")
        lines.append(f"  60d return: {ret60:+.1f}%")
        lines.append(f"  20d range: ${lo20:.2f} - ${hi20:.2f}")
        if "relative_strength_20d" in ctx:
            lines.append(f"  20d relative-to-SPY: {ctx['relative_strength_20d']:+.2f}%")
        lines.append("  Last 10 bars OHLC:")
        for k in klines[-10:]:
            lines.append(f"    {k['date']}: {k['o']}, {k['h']}, {k['l']}, {k['c']} · vol {k['v']/1e6:.1f}M")
        lines.append("")
    news = ctx.get("news") or []
    if news:
        lines.append(f"Recent {len(news)} headlines (newest first):")
        for i, n in enumerate(news[:15], 1):
            lines.append(f"  {i}. [{n['published_at'][:10]}] [{n['source']}] {n['title']}")
        lines.append("")
    else:
        lines.append("Recent news: NONE FOUND. Lower your confidence to reflect low information.")
        lines.append("")
    if ctx.get("in_weekly_news_pool"):
        lines.append("Meta: this symbol is in the current WEEKLY news-hotness pool (had ≥70 score at some point this week).")
    if ctx.get("in_daily_news_pool"):
        lines.append("Meta: this symbol is in the current DAILY news-hotness pool.")
    lines.append("")
    lines.append("Produce the JSON weekly thesis.")
    return "\n".join(lines)


def _build_daily_user_prompt(ctx: dict, weekly: Optional[dict]) -> str:
    lines = [
        f"Symbol: {ctx['symbol']} · Market: {ctx['market']}",
        f"Report date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]
    if weekly:
        thesis_snippet = (weekly.get("thesis") or "")[:400]
        lines.append(f"Prior weekly thesis ({weekly.get('generated_at','?')[:10]}, bias={weekly.get('bias')}, conf={weekly.get('confidence')}):")
        lines.append(f"  {thesis_snippet}")
        drivers = weekly.get("drivers") or []
        if drivers:
            lines.append(f"  Key drivers: {'; '.join(drivers[:3])}")
        lines.append("")
    else:
        lines.append("Prior weekly thesis: NONE (no weekly generated yet).")
        lines.append("")

    if ctx.get("klines"):
        klines = ctx["klines"]
        last = klines[-1]
        prev = klines[-2] if len(klines) >= 2 else None
        lines.append(f"Today ({last['date']}):")
        lines.append(f"  OHLC: {last['o']}, {last['h']}, {last['l']}, {last['c']} · vol {last['v']/1e6:.1f}M")
        if prev:
            chg = (last["c"] - prev["c"]) / prev["c"] * 100
            lines.append(f"  Change from prev close: {chg:+.2f}%")
        lines.append("  Recent 5 days OHLC:")
        for k in klines[-5:]:
            lines.append(f"    {k['date']}: c ${k['c']:.2f}")
        lines.append("")
    news = ctx.get("news") or []
    fresh_news = [n for n in news if _is_within_hours(n.get("published_at"), 30)]
    if fresh_news:
        lines.append(f"News in last 30h ({len(fresh_news)} items):")
        for i, n in enumerate(fresh_news[:8], 1):
            lines.append(f"  {i}. [{n['published_at'][:16]}] [{n['source']}] {n['title']}")
        lines.append("")
    else:
        lines.append("News in last 30h: NONE.")
        lines.append("")
    if ctx.get("next_earnings"):
        lines.append(f"Next earnings: {ctx['next_earnings']}")
        lines.append("")
    lines.append("Produce the JSON daily update.")
    return "\n".join(lines)


def _is_within_hours(published_at: str, hours: int) -> bool:
    if not published_at:
        return False
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - pub).total_seconds() / 3600 <= hours
    except Exception:
        return False


# ─── LLM call (CLI first, API fallback) ────────────────────────────

# 与 quant_llm 共用同一个异常类 (call_provider 抛的就是它), 前端据此引导到 LLM 管理 tab
from quant_llm import ProviderNotConfigured  # noqa: E402  (re-export)


def _call_llm(provider: str, system_prompt: str, user_prompt: str,
              max_tokens: int = 1200) -> tuple[Optional[dict], str]:
    """Return (parsed_dict_or_None, raw_text).
    走 quant_llm.call_provider: 读 LLM 管理 tab 的配置 (use_cli / api_key / model_name),
    CLI 优先且同 provider 串行 (_CLI_MUTEX), 失败回退 API.
    抛 ProviderNotConfigured 如果 provider 完全没配 (无 API key 也没 CLI);
    其他调用失败抛 RuntimeError.
    """
    from quant_llm import call_provider
    return call_provider(provider, system_prompt, user_prompt, max_tokens=max_tokens)


def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    # 直接 json
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 找 fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 找第一个 {...} balanced
    m2 = re.search(r"\{[\s\S]*\}", raw)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
    return None


# ─── Generators ─────────────────────────────────────────────────────

def generate_weekly(symbol: str, market: str, provider: str = "claude") -> dict:
    """Generate a weekly thesis for a symbol. Returns the stored report dict.
    抛 RuntimeError / ProviderNotConfigured 如 LLM 调用失败."""
    symbol = symbol.upper()
    ctx = _fetch_context(symbol, market)
    user_prompt = _build_weekly_user_prompt(ctx)
    parsed, raw = _call_llm(provider, _WEEKLY_SYSTEM_PROMPT, user_prompt, max_tokens=1200)
    if parsed is None:
        parsed = _parse_json(raw) or {}
    if not parsed and not raw:
        raise RuntimeError(f"LLM 返回空, 检查 {provider} 配置")
    if not parsed.get("thesis"):
        raise RuntimeError(f"LLM 返回未包含 thesis · raw: {(raw or '')[:200]}")
    report = {
        "symbol": symbol,
        "market": market.upper(),
        "kind": "weekly",
        "provider": provider,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thesis": str(parsed.get("thesis") or "")[:1500],
        "drivers": list(parsed.get("drivers") or [])[:8],
        "risks": list(parsed.get("risks") or [])[:8],
        "valuation": str(parsed.get("valuation") or ""),
        "sector_context": str(parsed.get("sector_context") or ""),
        "bias": str(parsed.get("bias") or "neutral").lower(),
        "confidence": float(parsed.get("confidence") or 0),
        "time_horizon": str(parsed.get("time_horizon") or "3-6mo"),
        "raw_snippet": (raw or "")[:2000],
        "context_summary": {
            "n_klines": len(ctx.get("klines") or []),
            "n_news": len(ctx.get("news") or []),
            "has_fundamentals": bool(ctx.get("fundamentals")),
            "next_earnings": ctx.get("next_earnings"),
            "in_weekly_news_pool": ctx.get("in_weekly_news_pool"),
        },
    }
    with _lock:
        data = _read(WEEKLY_PATH)
        data[symbol] = report
        _write(WEEKLY_PATH, data)
    return report


def generate_daily(symbol: str, market: str, provider: str = "claude") -> dict:
    """Generate a daily update for a symbol, referencing latest weekly if any.
    抛 RuntimeError / ProviderNotConfigured 如 LLM 调用失败."""
    symbol = symbol.upper()
    ctx = _fetch_context(symbol, market)
    weekly = get_weekly(symbol)
    user_prompt = _build_daily_user_prompt(ctx, weekly)
    parsed, raw = _call_llm(provider, _DAILY_SYSTEM_PROMPT, user_prompt, max_tokens=600)
    if parsed is None:
        parsed = _parse_json(raw) or {}
    if not parsed and not raw:
        raise RuntimeError(f"LLM 返回空, 检查 {provider} 配置")
    if not parsed.get("delta_summary"):
        raise RuntimeError(f"LLM 返回未包含 delta_summary · raw: {(raw or '')[:200]}")
    today = datetime.now().strftime("%Y-%m-%d")
    report = {
        "symbol": symbol,
        "market": market.upper(),
        "kind": "daily",
        "provider": provider,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "for_date": today,
        "delta_summary": str(parsed.get("delta_summary") or "")[:400],
        "thesis_still_valid": bool(parsed.get("thesis_still_valid", True)),
        "notes": str(parsed.get("notes") or "")[:500],
        "flag": (str(parsed.get("flag")) if parsed.get("flag") else None),
        "raw_snippet": (raw or "")[:1500],
        "has_weekly_context": weekly is not None,
    }
    with _lock:
        data = _read(DAILY_PATH)
        sym_key = symbol
        sym_data = data.get(sym_key) or {}
        sym_data[today] = report
        # 只留最近 30 天
        if len(sym_data) > 30:
            for old in sorted(sym_data.keys())[:-30]:
                sym_data.pop(old, None)
        data[sym_key] = sym_data
        _write(DAILY_PATH, data)
    return report


def generate_batch(symbols: list, market: str, kind: str = "weekly",
                   provider: str = "claude", concurrency: int = 4) -> dict:
    """Batch generate weekly OR daily reports for a list of symbols.
    Concurrency capped [1, 8] to avoid API rate limits.
    Returns {"ok": [...], "failed": [{symbol, error}, ...]}.
    """
    concurrency = max(1, min(int(concurrency), 8))
    fn = generate_weekly if kind == "weekly" else generate_daily
    ok, failed = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(fn, s, market, provider): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r = fut.result(timeout=180)
                ok.append({"symbol": sym, "bias": r.get("bias"),
                           "confidence": r.get("confidence"),
                           "flag": r.get("flag")})
            except Exception as e:
                logger.warning(f"batch {kind} {sym} failed: {e}")
                failed.append({"symbol": sym, "error": str(e)[:200]})
    return {"ok": ok, "failed": failed, "kind": kind, "provider": provider,
            "concurrency": concurrency}


def delete_weekly(symbol: str) -> bool:
    with _lock:
        data = _read(WEEKLY_PATH)
        if symbol.upper() in data:
            data.pop(symbol.upper())
            _write(WEEKLY_PATH, data)
            return True
    return False


def delete_daily(symbol: str, date: Optional[str] = None) -> bool:
    with _lock:
        data = _read(DAILY_PATH)
        sym = symbol.upper()
        if sym not in data:
            return False
        if date:
            if date in data[sym]:
                data[sym].pop(date)
                if not data[sym]:
                    data.pop(sym)
                _write(DAILY_PATH, data)
                return True
            return False
        # 无 date → 全删该 symbol
        data.pop(sym)
        _write(DAILY_PATH, data)
        return True
