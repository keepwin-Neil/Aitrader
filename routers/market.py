"""
Market data router: /quote, /kline, /quotes, /search, /info,
                    /market-data/*, /health, /
"""
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from typing import Optional
from pathlib import Path
from datetime import datetime
import requests as _requests
import yfinance as yf

# Import shared state / helpers from main
import main as _m

router = APIRouter()


@router.get("/quote/{market}/{symbol}")
def get_quote(market: str, symbol: str):
    cache_key = f"quote:{market}:{symbol}"
    cached = _m.cache_get(cache_key, ttl=30)
    if cached:
        return cached
    market = market.upper()
    if market == "CRYPTO":
        sym = symbol.upper() + "USDT"
        try:
            r = _requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}", timeout=5)
            d = r.json()
            if "code" in d:
                raise HTTPException(400, d["msg"])
            result = {
                "symbol": symbol.upper(), "market": "CRYPTO",
                "price": _m.safe_float(d["lastPrice"]),
                "change_pct": _m.safe_float(d["priceChangePercent"]),
                "high": _m.safe_float(d["highPrice"]),
                "low": _m.safe_float(d["lowPrice"]),
                "volume_usdt": _m.safe_float(d["quoteVolume"]),
                "currency": "USDT", "source": "Binance",
                "ts": datetime.utcnow().isoformat(),
            }
            _m.cache_set(cache_key, result)
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Binance error: {e}")
    # Equity/HK/UK/CN quotes go through the multi-source router. Priority is
    # IB Gateway (paid realtime NBBO, 16-slot pool) → Yahoo Finance (free,
    # capped at 8 concurrent) → OHLCV cache (local, stale). When IB pool is
    # saturated overflow runs on Yahoo in parallel — see quote_router.py.
    from quote_router import router as _qr
    result = _qr.fetch_quote(symbol, market)
    if result is None or result.get("price") is None:
        raise HTTPException(500, "all quote sources failed")
    _m.cache_set(cache_key, result)
    return result


@router.get("/market-data/providers")
def market_data_providers():
    from ib_gateway import ib_status as _ib_status
    return {
        "primary_history_source": "Yahoo Finance",
        "fallback_sources": ["IB Gateway", "AKShare"],
        "ib": _ib_status(),
        "ts": datetime.utcnow().isoformat(),
    }


@router.get("/market-data/ib/status")
def market_data_ib_status():
    from ib_gateway import ib_status as _ib_status
    return _ib_status()


@router.get("/kline/{market}/{symbol}")
def get_kline(market: str, symbol: str,
              interval: str = Query("1d"),
              limit: int = Query(120, ge=10, le=1000)):
    from backtest import get_ohlcv
    today_str = datetime.now().strftime("%Y%m%d") if interval in ("1d","1wk","1mo") else ""
    cache_key = f"kline:{market}:{symbol}:{interval}:{limit}:{today_str}"
    ttl = 60 if interval in ("1d", "1wk", "1mo") else 30
    cached = _m.cache_get(cache_key, ttl)
    if cached:
        return cached
    market = market.upper()
    if market == "CRYPTO":
        bi_map = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
                  "1h":"1h","4h":"4h","1d":"1d","1wk":"1w","1mo":"1M"}
        bi = bi_map.get(interval, "1d")
        sym = symbol.upper() + "USDT"
        try:
            r = _requests.get(
                f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={bi}&limit={limit}",
                timeout=8)
            raw = r.json()
            klines = [{"t": datetime.fromtimestamp(k[0]/1000).isoformat(),
                        "o": _m.safe_float(k[1]), "h": _m.safe_float(k[2]),
                        "l": _m.safe_float(k[3]), "c": _m.safe_float(k[4]),
                        "v": _m.safe_float(k[5])} for k in raw]
            result = {"symbol": symbol, "market": market, "interval": interval,
                      "count": len(klines), "source": "Binance", "klines": klines}
            _m.cache_set(cache_key, result)
            return result
        except Exception as e:
            raise HTTPException(500, f"Binance kline error: {e}")
    if market == "CN":
        try:
            cn_period = {"1d": "2y", "1wk": "5y", "1mo": "10y"}.get(interval, "2y")
            df = get_ohlcv(symbol, period=cn_period, interval=interval, market="CN")
            if df is not None and len(df) > 0:
                df = df.tail(limit)
                klines = [{"t": idx.isoformat(), "o": _m.safe_float(row.get("open", row.get("Open", 0))),
                            "h": _m.safe_float(row.get("high", row.get("High", 0))), "l": _m.safe_float(row.get("low", row.get("Low", 0))),
                            "c": _m.safe_float(row.get("close", row.get("Close", 0))), "v": _m.safe_float(row.get("volume", row.get("Volume", 0)))}
                           for idx, row in df.iterrows()]
                result = {"symbol": symbol, "market": "CN", "interval": interval,
                          "count": len(klines), "source": "Local+Incremental", "klines": klines}
                _m.cache_set(cache_key, result)
                return result
            raise Exception("No data")
        except Exception as e:
            raise HTTPException(500, f"CN kline error: {e}")
    period_map = {"1d": "2y", "1wk": "5y", "1mo": "10y"}
    if interval in period_map:
        try:
            period_needed = period_map[interval]
            df = get_ohlcv(symbol, period=period_needed, interval=interval, market=market)
            if df is not None and len(df) > 0:
                df = df.tail(limit)
                klines = [{"t": idx.isoformat(), "o": _m.safe_float(row.get("open", row.get("Open", 0))),
                            "h": _m.safe_float(row.get("high", row.get("High", 0))), "l": _m.safe_float(row.get("low", row.get("Low", 0))),
                            "c": _m.safe_float(row.get("close", row.get("Close", 0))), "v": _m.safe_float(row.get("volume", row.get("Volume", 0)))}
                           for idx, row in df.iterrows()]
                result = {"symbol": symbol, "market": market, "interval": interval,
                          "count": len(klines), "source": "Local+Incremental", "klines": klines}
                _m.cache_set(cache_key, result)
                return result
        except Exception:
            pass
    days_needed = limit * {"1m":1,"5m":1,"15m":2,"30m":3,"1h":7,
                           "4h":14,"1d":1,"1wk":7,"1mo":30}.get(interval, 1)
    intraday_period = f"{max(days_needed, 5)}d" if days_needed <= 59 else f"{min((days_needed//30)+1,60)}mo"
    try:
        df = get_ohlcv(symbol, market=market, interval=interval, period=intraday_period)
        if df is not None and len(df) > 0:
            df = df.tail(limit)
            klines = [{"t": idx.isoformat(), "o": _m.safe_float(row["open"]),
                        "h": _m.safe_float(row["high"]), "l": _m.safe_float(row["low"]),
                        "c": _m.safe_float(row["close"]), "v": _m.safe_float(row["volume"])}
                       for idx, row in df.iterrows()]
            result = {"symbol": symbol, "market": market, "interval": interval,
                      "count": len(klines), "source": "Local+Incremental", "klines": klines}
            _m.cache_set(cache_key, result)
            return result
    except Exception:
        pass
    ticker_sym = _m.yf_symbol(symbol, market)
    try:
        t = yf.Ticker(ticker_sym)
        df = t.history(interval=interval, period=intraday_period).tail(limit)
        klines = [{"t": idx.isoformat(), "o": _m.safe_float(row["Open"]),
                    "h": _m.safe_float(row["High"]), "l": _m.safe_float(row["Low"]),
                    "c": _m.safe_float(row["Close"]), "v": _m.safe_float(row["Volume"])}
                   for idx, row in df.iterrows()]
        result = {"symbol": symbol, "market": market, "interval": interval,
                  "count": len(klines), "source": "Yahoo Finance", "klines": klines}
        _m.cache_set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(500, f"kline error: {e}")


@router.post("/quotes/batch")
def batch_quotes(body: dict):
    """Batch quotes. Accepts both symbol shapes:
      {"symbols": [{"market":"US","symbol":"AAPL"}, ...]}
      {"symbols": ["AAPL","NVDA"], "market": "US"}   ← market defaults to US
    Previously only the dict shape worked; plain strings crashed with
    `TypeError: string indices must be integers` (seen in prod logs).
    """
    from concurrent.futures import ThreadPoolExecutor as _TP
    symbols = body.get("symbols", [])[:20]
    default_market = str(body.get("market", "US")).upper()
    if not symbols:
        return {"count": 0, "data": []}

    def _fetch(s):
        if isinstance(s, dict):
            market, symbol = s.get("market", default_market), s.get("symbol", "")
        else:
            market, symbol = default_market, str(s)
        try:
            return get_quote(market, symbol)
        except Exception as e:
            return {"symbol": symbol, "market": market, "error": str(e)}

    with _TP(max_workers=min(len(symbols), 8)) as ex:
        results = list(ex.map(_fetch, symbols))
    return {"count": len(results), "data": results}


@router.get("/search")
def search_symbol(q: str, market: Optional[str] = None):
    try:
        results = yf.Search(q, max_results=10)
        hits = []
        for r in (results.quotes or []):
            m = "US"
            ex = r.get("exchange", "")
            if ex in ("HKG","HKSE"): m = "HK"
            elif ex in ("SHH","SHZ","SHA","SHE"): m = "CN"
            elif ex in ("LSE", "IOB", "LON"): m = "UK"
            if market and m != market.upper(): continue
            hits.append({"symbol": r.get("symbol",""),
                          "name": r.get("longname") or r.get("shortname",""),
                          "market": m, "exchange": ex, "type": r.get("quoteType","")})
        return {"query": q, "results": hits}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/info/{market}/{symbol}")
def get_info(market: str, symbol: str):
    if market.upper() == "CRYPTO":
        raise HTTPException(400, "Crypto info not supported")
    ticker_sym = _m.yf_symbol(symbol, market.upper())
    cache_key = f"info:{ticker_sym}"
    cached = _m.cache_get(cache_key, ttl=3600)
    if cached: return cached
    try:
        info = yf.Ticker(ticker_sym).info
        result = {
            "symbol": symbol, "name": info.get("longName") or info.get("shortName"),
            "market": market.upper(), "sector": info.get("sector"),
            "industry": info.get("industry"),
            "pe_ratio": _m.safe_float(info.get("trailingPE")),
            "pb_ratio": _m.safe_float(info.get("priceToBook")),
            "market_cap": _m.safe_float(info.get("marketCap")),
            "dividend_yield": _m.safe_float(info.get("dividendYield")),
            "52w_high": _m.safe_float(info.get("fiftyTwoWeekHigh")),
            "52w_low": _m.safe_float(info.get("fiftyTwoWeekLow")),
            "currency": info.get("currency"), "source": "Yahoo Finance",
        }
        _m.cache_set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "cache_keys": len(_m._cache)}


@router.post("/quotes")
def get_batch_quotes(body: dict):
    """批量获取股票行情（价格/涨跌幅/成交量/市值），5分钟缓存

    Cache miss路径走 quote_router: IB Gateway pool 优先，满了溢出到 Yahoo
    Finance，两源并行均衡负载。最坏情况 fallback 到 OHLCV cache 返回
    stale-but-real 价格。详见 quote_router.py。
    """
    from quote_router import router as _qr
    # 上限 50 → 200: MYPOOL 60 / NASDAQ 100 / 中国 100 等常见池能一次调用装下.
    # 前端 loadQuoteDetails 也加了 chunk 循环兜底 (每批 100), 超 200 也不丢.
    symbols = body.get("symbols", [])[:200]
    market = body.get("market", "US").upper()
    result = {}
    to_fetch = []

    for sym in symbols:
        key = f"quote:{market}:{sym}"
        cached = _m.cache_get(key, ttl=300)
        if cached:
            result[sym] = cached
        else:
            to_fetch.append(sym)

    if to_fetch:
        fetched = _qr.fetch_quotes(to_fetch, market, max_workers=32, timeout=15.0)
        for sym in to_fetch:
            q = fetched.get(sym)
            if q:
                result[sym] = {
                    "name": q.get("name") or sym,
                    "price": q.get("price"),
                    "change_pct": q.get("change_pct"),
                    "volume": q.get("volume"),
                    "market_cap": q.get("market_cap"),
                    "source": q.get("source"),
                }
                _m.cache_set(f"quote:{market}:{sym}", result[sym])
            else:
                result[sym] = None

    return {"details": result}


@router.get("/")
def root():
    content = (Path(__file__).parent.parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


_US_NAMES_CACHE_FILE = Path(__file__).parent.parent / ".cache" / "symbol_names" / "US.json"
_us_disk_loaded = False


def _load_us_names_from_disk():
    """一次性把 US.json 磁盘缓存读进内存 (_cn_name_cache), 幂等."""
    global _us_disk_loaded
    if _us_disk_loaded:
        return
    _us_disk_loaded = True
    try:
        import json
        if _US_NAMES_CACHE_FILE.exists():
            payload = json.loads(_US_NAMES_CACHE_FILE.read_text(encoding="utf-8"))
            names = payload.get("names", {})
            for sym, name in names.items():
                if name:
                    _m._cn_name_cache[f"US:{sym}"] = name
    except Exception as e:
        import logging
        logging.warning(f"[stock-names] US disk cache load failed: {e}")


def _save_us_names_to_disk():
    """把当前内存里所有 US:* 名字写回磁盘, 供下次启动复用."""
    try:
        import json
        _US_NAMES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        us_names = {k[3:]: v for k, v in _m._cn_name_cache.items()
                    if k.startswith("US:") and v}
        payload = {"updated_at": datetime.utcnow().isoformat(), "names": us_names}
        tmp = _US_NAMES_CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_US_NAMES_CACHE_FILE)
    except Exception as e:
        import logging
        logging.warning(f"[stock-names] US disk cache save failed: {e}")


def _fetch_us_name(symbol: str) -> str:
    """单个 US 股票公司名: yfinance longName > shortName > 空."""
    try:
        info = yf.Ticker(symbol).info
        name = (info.get("longName") or info.get("shortName") or "").strip()
        return name
    except Exception:
        return ""


@router.get("/stock-names")
def get_stock_names(symbols: str, market: str = "CN"):
    """公司/股票名字查询. CN/HK 用新浪, US 用 yfinance + 磁盘缓存."""
    import re
    import logging
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    result = {}
    to_fetch = []

    # US 走磁盘缓存, 先把磁盘的加载进内存
    if market.upper() == "US":
        _load_us_names_from_disk()

    for s in sym_list:
        key = f"{market}:{s}"
        if key in _m._cn_name_cache:
            result[s] = _m._cn_name_cache[key]
        else:
            to_fetch.append(s)

    if not to_fetch:
        return result

    # ── US: yfinance 并行拉, 缓存写盘 ──────────────────────────────
    if market.upper() == "US":
        from concurrent.futures import ThreadPoolExecutor as _TP
        n_workers = min(8, len(to_fetch))
        with _TP(max_workers=n_workers) as ex:
            fetched = dict(zip(to_fetch, ex.map(_fetch_us_name, to_fetch)))
        for sym, name in fetched.items():
            # 空字符串也缓存 (标记"已查过没有"), 避免每次重复调 yfinance
            _m._cn_name_cache[f"US:{sym}"] = name
            if name:
                result[sym] = name
        _save_us_names_to_disk()
        return result

    sina_syms = []
    if market == "CN":
        for s in to_fetch:
            prefix = "sh" if s.startswith(("6", "9")) else "sz"
            sina_syms.append(f"{prefix}{s}")
    elif market == "HK":
        for s in to_fetch:
            sina_syms.append(f"hk{s.zfill(5)}")
    else:
        return result

    try:
        url = f"http://hq.sinajs.cn/list={','.join(sina_syms)}"
        resp = _requests.get(url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=8)
        text = resp.content.decode("gbk", errors="replace")

        for line in text.strip().split("\n"):
            m = re.match(r'var hq_str_(\w+)="([^"]*)"', line)
            if not m:
                continue
            sina_sym = m.group(1)
            fields = m.group(2).split(",")
            if not fields or not fields[0]:
                continue

            if market == "CN":
                cn_name = fields[0]
                orig = sina_sym[2:]
            else:
                cn_name = fields[1] if len(fields) > 1 else fields[0]
                padded = sina_sym[2:]
                orig = next((s for s in to_fetch if s.zfill(5) == padded), padded)

            _m._cn_name_cache[f"{market}:{orig}"] = cn_name
            result[orig] = cn_name

    except Exception as e:
        logging.warning(f"[stock-names] Sina API error: {e}")

    return result
