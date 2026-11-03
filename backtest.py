import pandas as pd
import numpy as np
import yfinance as yf
import ta
import akshare as ak
import time, pickle, os, logging, tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from strategies import get_strategy, list_strategies, STRATEGY_REGISTRY
from ib_gateway import ib_enabled, ib_supported_market, fetch_history as ib_fetch_history

_logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "ema_fast": 10, "ema_mid": 20, "ema_slow": 60,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "rsi_period": 14, "rsi_buy_min": 45, "rsi_buy_max": 70, "rsi_sell": 78,
    "bb_period": 20, "bb_std": 2,
    "vol_ma": 20, "vol_ratio_min": 1.2,
    "capital": 100000, "commission": 0.001,
    "position_pct": 0.95, "atr_stop_mul": 2.0,
}

# ── 本地行情数据管理 ──
# 目录结构: market_data/{market}/{symbol}_{interval}.parquet
_DATA_DIR = Path(__file__).parent / "market_data"
_DATA_DIR.mkdir(exist_ok=True)
_MEM_CACHE = {}   # 内存缓存: key -> (timestamp, df)
_MEM_TTL = 120    # 内存缓存2分钟（避免同一请求重复读盘）
_MEM_CACHE_MAX = 512  # 上限：过期条目只在命中时才检查，长跑进程扫过几百只股票后
                      # dict 会无限增长（每个 df 数百 KB → GB 级）。触顶时先清
                      # 过期，仍超则踢最老的。


#: 保护 _MEM_CACHE 的淘汰逻辑。
#: 触顶淘汰要遍历整张表(min(...) 扫一遍找最老的), 而 get_ohlcv 现在有多处
#: 并发调用(买/卖点批量的 8 线程预取、批量优化的预取、适用性分析的取数池)。
#: 无锁的话, 一个线程在 min() 里遍历、另一个线程往里塞, 直接
#:   RuntimeError: dictionary keys changed during iteration
#: 或者 lambda 里 _MEM_CACHE[k][0] 撞上别人的 pop → KeyError。
#: 实测 8 线程 3 秒触发 197 次。而调用方(_preload_stock_cache 的 _one)是裸
#: except, 异常被吞掉 → 那只股票静默从结果里消失, 前端只看到数量少了一只。
_MEM_CACHE_LOCK = threading.Lock()


_TMP_SWEPT = set()


def _sweep_stale_tmp(d: Path) -> None:
    """清掉上次被砍掉的进程留下的 .parquet.tmp (每个进程每个目录只扫一次)。

    它们不会被 glob("*.parquet") 当成行情文件(后缀对不上), 所以不影响正确性,
    但 --reload 一次就留几个, 攒起来白占磁盘。
    """
    key = str(d)
    if key in _TMP_SWEPT:
        return
    _TMP_SWEPT.add(key)
    try:
        cutoff = time.time() - 3600
        for f in d.glob("*.parquet.tmp"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _atomic_to_parquet(df, path) -> None:
    """原子写 parquet: 先写临时文件, 再 os.replace 换过去。

    直写最终路径有两个风险, 两个都真会发生:
      1. 进程被中途砍掉 —— 池 worker 现在带了"父进程没了就 os._exit(0)"的
         看门狗(proc_role), uvicorn --reload 一重启, 正在写缓存的 worker 就
         当场没了, 磁盘上留一个没有 footer 的半截 parquet。之后每次读该
         symbol 都抛 ArrowInvalid, **不会自愈**。
      2. 同一 symbol 被两个线程并发写(模拟盘监控改成线程池之后, 同股多笔
         持仓会同时进来)。
    os.replace 在同一文件系统上是原子的: 要么看到旧文件, 要么看到完整的新
    文件, 不会看到写了一半的。
    """
    path = path if isinstance(path, Path) else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _sweep_stale_tmp(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _mem_cache_put(cache_key: str, df) -> None:
    now = time.time()
    with _MEM_CACHE_LOCK:
        if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
            for k in [k for k, (ts, _) in list(_MEM_CACHE.items()) if now - ts >= _MEM_TTL]:
                _MEM_CACHE.pop(k, None)
            while len(_MEM_CACHE) >= _MEM_CACHE_MAX:
                # 拿 items() 的快照再挑, 别在 min() 里边遍历边让别人改
                snap = list(_MEM_CACHE.items())
                if not snap:
                    break
                oldest = min(snap, key=lambda kv: kv[1][0])[0]
                _MEM_CACHE.pop(oldest, None)
        _MEM_CACHE[cache_key] = (now, df)

# Yahoo Finance ticker 转换
_TICKER_MAP = {
    "CN":     lambda s: f"{s}.SS" if s.startswith("6") else f"{s}.SZ",
    "HK":     lambda s: f"{int(s):04d}.HK",   # "00981" → "0981.HK"（去多余前导零）
    "UK":     lambda s: s.upper() if "." in str(s) else f"{str(s).upper()}.L",
    "US":     lambda s: s.upper(),
    "CRYPTO": lambda s: f"{s.upper()}-USD",
}


def _get_data_path(symbol: str, market: str, interval: str) -> Path:
    """本地数据文件路径"""
    mkt_dir = _DATA_DIR / market.upper()
    mkt_dir.mkdir(exist_ok=True)
    return mkt_dir / f"{symbol}_{interval}.parquet"


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        _logger.warning(f"[parquet] {path.name} damaged: {e}")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _download_yf(symbol: str, market: str, interval: str,
                 start: str = None, period: str = None) -> pd.DataFrame:
    """从 Yahoo Finance 下载数据，含自动重试和指数退避"""
    # B1 data-source audit: ticker alias resolution (FB→META, TWTR→X, etc.)
    try:
        from ticker_aliases import resolve as _resolve_alias
        resolved = _resolve_alias(symbol, market)
        if resolved != symbol:
            _logger.debug(f"[alias] {market}:{symbol} → {resolved}")
            symbol = resolved
    except Exception:
        pass
    ticker = _TICKER_MAP.get(market.upper(), lambda s: s)(symbol)
    kwargs = {"interval": interval}
    if start:
        kwargs["start"] = start
    else:
        kwargs["period"] = period or "max"
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(**kwargs)
            if df.empty:
                return pd.DataFrame()
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(kw in err_str for kw in ("429", "rate", "too many", "throttl"))
            if attempt < 2:
                wait = (2 ** attempt) * (2.0 if is_rate_limit else 0.5)
                _logger.warning(f"[yf] {market}:{symbol} {interval} attempt {attempt+1}/3: {e}. retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                _logger.warning(f"[yf] {market}:{symbol} {interval} failed after 3 attempts: {e}")
    return pd.DataFrame()


def _download_ib(symbol: str, market: str, interval: str,
                 start: str = None, period: str = None) -> pd.DataFrame:
    """从 IB Gateway 下载数据，失败则返回空表。"""
    if not ib_enabled() or not ib_supported_market(market):
        return pd.DataFrame()
    try:
        return ib_fetch_history(symbol, market, interval, period=period, start=start)
    except Exception as e:
        _logger.warning(f"[ib] {market}:{symbol} {interval} failed: {e}")
        return pd.DataFrame()


def _download_primary(symbol: str, market: str, interval: str,
                      start: str = None, period: str = None) -> pd.DataFrame:
    """主数据源优先：Yahoo Finance → IB Gateway。"""
    df = _download_yf(symbol, market, interval, start=start, period=period)
    if df is not None and not df.empty:
        return df
    return _download_ib(symbol, market, interval, start=start, period=period)


def _download_ak_hk(symbol: str, start: str = None) -> pd.DataFrame:
    """用 akshare 下载港股日线数据（Yahoo Finance 对新股数据不全时的备用源），含3次重试"""
    start_date = (start or "20100101").replace("-", "")
    end_date = datetime.now().strftime("%Y%m%d")
    for attempt in range(3):
        try:
            df = ak.stock_hk_hist(
                symbol=symbol.zfill(5),   # akshare 需要 5 位代码，如 "00100"
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df.index.name = "date"
            return df.astype({"open": float, "high": float, "low": float,
                               "close": float, "volume": float})
        except Exception as e:
            _logger.warning(f"[akshare_hk] {symbol} attempt {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(3)
    return pd.DataFrame()


def _resample_from_daily(symbol: str, market: str, interval: str) -> pd.DataFrame:
    if interval not in ("1wk", "1mo"):
        return pd.DataFrame()
    day_path = _get_data_path(symbol, market, "1d")
    if not day_path.exists():
        return pd.DataFrame()
    try:
        df = _safe_read_parquet(day_path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    rule = "W-FRI" if interval == "1wk" else "MS"
    agg = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    agg.index.name = "date"
    return agg


def _rebuild_current_daily_bar(symbol: str, market: str) -> pd.DataFrame:
    """用分时数据重建当日日线，避免 Yahoo 当天日线 OHLC 偶发异常。"""
    try:
        intraday = _download_primary(symbol, market, "5m", period="5d")
    except Exception:
        intraday = pd.DataFrame()
    if intraday is None or intraday.empty:
        return pd.DataFrame()
    intraday = intraday.sort_index()
    today = pd.Timestamp.now().normalize()
    today_df = intraday[intraday.index.normalize() == today]
    if today_df.empty:
        return pd.DataFrame()
    row = pd.DataFrame([{
        "open": float(today_df["open"].iloc[0]),
        "high": float(today_df["high"].max()),
        "low": float(today_df["low"].min()),
        "close": float(today_df["close"].iloc[-1]),
        "volume": float(today_df["volume"].sum()),
    }], index=[today])
    row.index.name = "date"
    return row


def _repair_daily_with_intraday(symbol: str, market: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        latest_idx = df.index[-1]
        latest = df.iloc[-1]
        invalid_ohlc = (
            latest["low"] > min(latest["open"], latest["close"]) or
            latest["high"] < max(latest["open"], latest["close"])
        )
        is_today = latest_idx.normalize() == pd.Timestamp.now().normalize()
        if not (is_today or invalid_ohlc):
            return df
        rebuilt = _rebuild_current_daily_bar(symbol, market)
        if rebuilt is None or rebuilt.empty:
            return df
        combined = pd.concat([df.iloc[:-1] if is_today else df, rebuilt])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined
    except Exception:
        return df


def init_stock_data(symbol: str, market: str, interval: str = "1d") -> pd.DataFrame:
    """首次下载：获取最大历史数据并保存到本地文件"""
    path = _get_data_path(symbol, market, interval)
    _logger.info(f"[init] {market}:{symbol} {interval} → {path}")

    if interval in ("1wk", "1mo"):
        df_local = _resample_from_daily(symbol, market, interval)
        if not df_local.empty:
            _atomic_to_parquet(df_local, path)
            _logger.info(f"[init] {market}:{symbol} {interval} rebuilt from local 1d ({len(df_local)} bars)")
            return df_local

    df = _download_primary(symbol, market, interval, period="max")

    # 若 period="max" 返回空（部分新股 Yahoo 不支持该参数），
    # 改用近3年起始日期重试（适合新上市股票）
    if df.empty:
        start_3y = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")
        _logger.info(f"[init] {market}:{symbol} period=max 为空，改用 start={start_3y} 重试")
        df = _download_primary(symbol, market, interval, start=start_3y)

    # HK 日线：若 Yahoo Finance 数据仍不足30行，尝试 akshare 补充（新股/小市值股）
    if market.upper() == "HK" and interval == "1d" and len(df) < 30:
        _logger.info(f"[init] {market}:{symbol} Yahoo数据不足({len(df)}行)，尝试 akshare 补充")
        df_ak = _download_ak_hk(symbol)
        if len(df_ak) > len(df):
            df = df_ak
            _logger.info(f"[init] {market}:{symbol} akshare 补充 {len(df)} 行")

    # 不足1行则放弃（真正无数据）
    if df.empty:
        _logger.warning(f"[init] {market}:{symbol} 无法获取任何数据，跳过保存")
        return df

    if interval == "1d":
        df = _repair_daily_with_intraday(symbol, market, df)

    # 有数据就保存（即使不足60根），保存实际拥有的所有行
    _atomic_to_parquet(df, path)
    _logger.info(f"[init] {market}:{symbol} saved {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


def update_stock_data(symbol: str, market: str, interval: str = "1d") -> pd.DataFrame:
    """增量更新：读取本地数据，仅下载缺失部分并追加"""
    path = _get_data_path(symbol, market, interval)

    if not path.exists():
        return init_stock_data(symbol, market, interval)

    # 读取已有数据
    try:
        existing = _safe_read_parquet(path)
    except Exception:
        return init_stock_data(symbol, market, interval)
    if existing.empty:
        return init_stock_data(symbol, market, interval)

    last_date = existing.index[-1]
    now = pd.Timestamp.now()
    is_intraday = interval not in ("1d", "1wk", "1mo")

    if is_intraday:
        # 分钟线：Yahoo限制历史深度(1m=7d, 5m/15m/30m=60d, 1h=730d)
        # 如果本地数据太旧超出历史深度限制，需要重新全量下载
        max_days = {"1m": 6, "5m": 55, "15m": 55, "30m": 55,
                    "1h": 700, "60m": 700, "4h": 700}
        limit = max_days.get(interval, 55)
        age_days = (now - last_date).total_seconds() / 86400
        if age_days > limit:
            _logger.info(f"[update] {market}:{symbol} {interval} 本地数据过旧({age_days:.0f}天)，重新下载")
            return init_stock_data(symbol, market, interval)
        # 从最后数据前1天开始下载增量
        start = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # 日线/周线/月线
        today = now.normalize()
        if last_date.normalize() >= today - timedelta(days=1):
            start = (last_date - timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            start = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")

    # 下载增量数据
    new_df = _download_primary(symbol, market, interval, start=start)
    # HK 日线：Yahoo 返回空时用 akshare 补充
    if new_df.empty and market.upper() == "HK" and interval == "1d":
        _logger.info(f"[update] {market}:{symbol} Yahoo增量为空，尝试 akshare 补充")
        new_df = _download_ak_hk(symbol, start=start)
    if new_df.empty:
        # B3 data-source audit: delisted detection
        # 只对日线做这个检测（分钟线天然波动大）；超过 10 天无新 bar → 标记退市
        if interval == "1d":
            try:
                age_days = int(
                    (pd.Timestamp.now().normalize() - last_date.normalize()).days
                )
                if age_days > 10:
                    from delisted_tickers import mark_delisted as _mark_delisted
                    last_str = str(last_date)[:10]
                    reason = f"no new 1d bars for {age_days}d (last={last_str})"
                    _mark_delisted(symbol, market, last_str, reason)
                    _logger.warning(
                        f"[delisted] {market}:{symbol} marked — {reason}"
                    )
            except Exception as e:
                _logger.debug(f"[delisted] detect failed for {market}:{symbol}: {e}")
        return existing

    # 合并：去重（新数据覆盖旧数据）
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # B3: 如果之前标过退市但现在又有新数据 → 反标
    if interval == "1d":
        try:
            from delisted_tickers import unmark_delisted as _unmark_delisted
            if _unmark_delisted(symbol, market):
                _logger.info(
                    f"[delisted] {market}:{symbol} un-marked (new bars detected)"
                )
        except Exception:
            pass

    if interval == "1d":
        combined = _repair_daily_with_intraday(symbol, market, combined)

    # 分钟线：裁剪保留最大深度范围，避免文件无限增长
    if is_intraday:
        max_keep = {"1m": 7, "5m": 60, "15m": 60, "30m": 60,
                    "1h": 730, "60m": 730, "4h": 730}
        keep_days = max_keep.get(interval, 60)
        cutoff = now - timedelta(days=keep_days)
        combined = combined[combined.index >= cutoff]

    # 保存
    _atomic_to_parquet(combined, path)
    added = len(combined) - len(existing)
    if added > 0:
        _logger.info(f"[update] {market}:{symbol} {interval} +{added} bars → {len(combined)} total")

    return combined


def get_ohlcv(symbol: str, market: str, interval: str = "1d", period: str = "2y",
              max_age_sec: int = None, include_today: bool = False,
              local_only: bool = False) -> pd.DataFrame:
    """获取行情数据：本地文件优先 + 增量更新"""
    cache_key = f"{market}_{symbol}_{interval}"
    is_intraday = interval not in ("1d", "1wk", "1mo")
    mem_ttl = _MEM_TTL
    if is_intraday and max_age_sec is not None:
        mem_ttl = max(0, int(max_age_sec))

    # 1. 内存缓存（2分钟）
    if cache_key in _MEM_CACHE:
        ts, df = _MEM_CACHE[cache_key]
        if time.time() - ts < mem_ttl:
            return _filter_period(df.copy(), period, interval, include_today=include_today)

    # 2. 仅本地读取（不触发任何远端刷新）
    path = _get_data_path(symbol, market, interval)
    if local_only:
        if not path.exists():
            return pd.DataFrame()
        try:
            df = _safe_read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        _mem_cache_put(cache_key, df)
        return _filter_period(df.copy(), period, interval, include_today=include_today)

    # 3. 本地文件 + 增量更新

    if path.exists():
        if interval in ("1d", "1wk", "1mo"):
            # 日线/周线/月线：本地文件较新时直接使用，避免频繁触发远端限频
            age = time.time() - path.stat().st_mtime
            default_age = 3600 if interval == "1d" else 21600
            file_max_age = default_age if max_age_sec is None else max(0, int(max_age_sec))
            if age < file_max_age:
                try:
                    df = _safe_read_parquet(path)
                except Exception:
                    if interval in ("1wk", "1mo"):
                        df = _resample_from_daily(symbol, market, interval)
                        if df.empty:
                            df = init_stock_data(symbol, market, interval)
                        else:
                            _atomic_to_parquet(df, path)
                    else:
                        df = init_stock_data(symbol, market, interval)
            else:
                if interval in ("1wk", "1mo"):
                    df = _resample_from_daily(symbol, market, interval)
                    if df.empty:
                        df = update_stock_data(symbol, market, interval)
                    else:
                        _atomic_to_parquet(df, path)
                else:
                    df = update_stock_data(symbol, market, interval)
        else:
            # 分钟线：本地文件 + 增量更新；可由调用方指定更严格的最大陈旧时间
            age = time.time() - path.stat().st_mtime
            file_max_age = 300 if max_age_sec is None else max(0, int(max_age_sec))
            if age < file_max_age:
                try:
                    df = _safe_read_parquet(path)
                except Exception:
                    df = init_stock_data(symbol, market, interval)
            else:
                df = update_stock_data(symbol, market, interval)
    else:
        df = init_stock_data(symbol, market, interval)

    if df is None or df.empty:
        return pd.DataFrame()

    # 写入内存缓存
    _mem_cache_put(cache_key, df)
    return _filter_period(df.copy(), period, interval, include_today=include_today)


def _filter_period(df: pd.DataFrame, period: str, interval: str = "1d",
                   include_today: bool = False) -> pd.DataFrame:
    """根据 period 参数截取数据范围。日线以昨日为终点（确保当日结果不变），分钟线保留今日数据"""
    if not period or period == "max" or df.empty:
        return df
    # 日线/周线/月线：以昨日为终点，排除今天的数据
    if interval in ("1d", "1wk", "1mo") and not include_today:
        yesterday = pd.Timestamp(datetime.now().date()) - timedelta(days=1)
        df = df[df.index <= yesterday]
        if df.empty:
            return df
    period_map = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
        "ytd": None,
    }
    if period == "ytd":
        year_start = pd.Timestamp(datetime.now().year, 1, 1)
        return df[df.index >= year_start]
    days = period_map.get(period)
    if days is None and isinstance(period, str) and period.endswith("d"):
        try:
            days = int(period[:-1])
        except Exception:
            days = None
    if days:
        # 日线用 yesterday 作为终点，分时线用当前时间
        anchor = (
            pd.Timestamp.now()
            if include_today or interval not in ("1d", "1wk", "1mo")
            else yesterday
        )
        cutoff = anchor - timedelta(days=days)
        return df[df.index >= cutoff]
    # Custom date range: "2025-01-01:2025-06-30"
    if isinstance(period, str) and ":" in period:
        parts = period.split(":", 1)
        try:
            start = pd.Timestamp(parts[0].strip())
            end = pd.Timestamp(parts[1].strip()) + timedelta(days=1)  # inclusive end
            return df[(df.index >= start) & (df.index < end)]
        except Exception:
            pass
    return df


def bulk_update(symbols: list, market: str = "US", interval: str = "1d") -> dict:
    """批量下载/更新多只股票数据，返回 {symbol: row_count}。并行下载，最多10线程。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    def _fetch_one(sym):
        for attempt in range(3):
            try:
                df = get_ohlcv(sym, market=market, interval=interval, period="max")
                rows = len(df) if df is not None and not df.empty else 0
                return sym, rows
            except Exception as e:
                _logger.warning(f"[bulk] {sym} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        return sym, 0

    total = len(symbols)
    results = {}
    n_threads = min(10, total)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        done = 0
        for fut in _as_completed(futures):
            sym, rows = fut.result()
            results[sym] = rows
            done += 1
            if rows > 0:
                _logger.info(f"[bulk] {done}/{total} {sym} ✅ {rows} bars")
            else:
                _logger.warning(f"[bulk] {done}/{total} {sym} ⚠️ empty data")
            if done % 10 == 0:
                _logger.info(f"[bulk] progress: {done}/{total}")
    _logger.info(f"[bulk] complete: {sum(1 for v in results.values() if v > 0)}/{total} succeeded")
    return results


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
    df["ema_mid"]  = ta.trend.ema_indicator(c, window=params["ema_mid"])
    df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])

    macd = ta.trend.MACD(c,
        window_slow=params["macd_slow"],
        window_fast=params["macd_fast"],
        window_sign=params["macd_signal"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])

    bb = ta.volatility.BollingerBands(c,
        window=params["bb_period"],
        window_dev=params["bb_std"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["vol_ma"]    = v.rolling(params["vol_ma"]).mean()
    df["vol_ratio"] = v / df["vol_ma"]
    df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)

    return df.dropna()


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    cond_ema   = (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"])
    cond_macd  = (df["macd"] > df["macd_signal"]) & (df["macd_hist"] > 0)
    cond_rsi   = (df["rsi"] > params["rsi_buy_min"]) & (df["rsi"] < params["rsi_buy_max"])
    cond_price = df["close"] > df["bb_mid"]
    cond_vol   = df["vol_ratio"] > params["vol_ratio_min"]

    sell_ema  = df["ema_fast"] < df["ema_mid"]
    sell_macd = (df["macd"] < df["macd_signal"]) & (df["macd_hist"] < 0)
    sell_rsi  = df["rsi"] > params["rsi_sell"]

    df["buy_signal"]  = cond_ema & cond_macd & cond_rsi & cond_price & cond_vol
    df["sell_signal"] = sell_ema | sell_macd | sell_rsi

    return df


def _ib_commission(shares: float, price: float) -> float:
    """IB Tiered: $0.0067/share (commission + clearing + exchange), min $0.35, max 1% of trade value"""
    raw = abs(shares) * 0.0067
    trade_value = abs(shares) * price
    return max(0.35, min(raw, trade_value * 0.01))

def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    capital      = params["capital"]
    commission   = params["commission"]  # legacy rate, kept for compatibility
    atr_stop_mul = params["atr_stop_mul"]
    use_ib_commission = params.get("use_ib_commission", True)  # default to IB model
    dynamic_pos  = params.get("dynamic_position", False)  # ATR动态仓位开关
    target_risk  = params.get("target_risk", 0.02)        # 目标风险 2%
    allow_short  = params.get("allow_short", False)        # 允许做空开关

    # 判断是否分钟/小时级数据
    _is_intraday = False
    if len(df) >= 2:
        _diff = (df.index[1] - df.index[0]).total_seconds()
        _is_intraday = _diff < 86400
    def _dt_key(idx):
        """生成日期key：日线用date，分钟线用完整timestamp"""
        return str(idx)[:19] if _is_intraday else str(idx.date())

    # 若开启动态仓位，预计算仓位比例
    if dynamic_pos and "atr" in df.columns:
        atr_pct = df["atr"] / df["close"]
        df = df.copy()
        df["_pos_size"] = (target_risk / atr_pct).clip(0.3, 1.0)

    # Commission helpers: IB per-share model or legacy rate model
    if use_ib_commission:
        def _sell_proceeds(shares, price):
            return shares * price - _ib_commission(shares, price)
        def _buy_cost(shares, price):
            return shares * price + _ib_commission(shares, price)
        def _max_shares(available_cash, price, pct=1.0):
            est = int(available_cash * pct / price)
            if est <= 0: return 0
            fee = _ib_commission(est, price)
            return max(0, int((available_cash * pct - fee) / price))
    else:
        def _sell_proceeds(shares, price):
            return shares * price * (1 - commission)
        def _buy_cost(shares, price):
            return shares * price * (1 + commission)
        def _max_shares(available_cash, price, pct=1.0):
            return int(available_cash * pct / (price * (1 + commission)))

    cash        = capital
    position    = 0      # 正数=多头，负数=空头
    entry_price = 0.0
    entry_cost  = 0.0    # total cost at entry (for PnL calc)
    stop_price  = 0.0
    trades      = []
    equity_curve = []
    actual_buys  = set()
    actual_sells = set()
    actual_shorts = set()
    actual_covers = set()

    # ─────────────────────────────────────────────────────────────────────────
    # 信号-成交分离（A1 修复 — 消除 look-ahead bias）
    #   entry_mode="next_open"（默认，正确）：信号在 bar T 的 close 检出 → 成交在 bar T+1 的 open
    #   entry_mode="signal_close"（opt-in 兼容模式，含 look-ahead）：信号与成交同在 bar T 的 close
    # "next_open" 模式下，最后一根 bar 的信号被丢弃（没有 T+1 open 可以成交）。
    # 止损分支仍用当前 bar 的 close（止损精度是独立问题，不在本次修复范围内）。
    # 参考：docs/superpowers/reports/2026-04-22-data-source-audit.md § A1
    # ─────────────────────────────────────────────────────────────────────────
    entry_mode = params.get("entry_mode", "next_open")
    _opens_arr = df["open"].values
    _dates_arr = df.index.tolist()
    _n_bars = len(df)

    def _resolve_signal_fill(pos_i: int, current_price: float, current_idx):
        """按 entry_mode 返回 (fill_price, fill_idx)。
        next_open 模式在最后一根 bar 返回 (None, None) 表示"信号必须丢弃"。"""
        if entry_mode == "signal_close":
            return current_price, current_idx
        if pos_i + 1 >= _n_bars:
            return None, None
        return float(_opens_arr[pos_i + 1]), _dates_arr[pos_i + 1]

    for pos_i, (i, row) in enumerate(df.iterrows()):
        price  = row["close"]
        # 计算权益：多头=cash+position*price，空头=cash+position*price（position为负）
        equity = cash + position * price
        equity_curve.append({"date": _dt_key(i), "equity": round(equity, 2)})

        # ── 多头止损 ──（用当前 bar close 做检查与成交；止损精度是独立问题）
        if position > 0 and price <= stop_price:
            proceeds = _sell_proceeds(position, price)
            cash += proceeds
            pnl = proceeds - entry_cost
            trades.append({"type": "SELL", "date": _dt_key(i), "price": round(price, 2),
                           "shares": abs(position), "pnl": round(pnl, 2), "reason": "多头止损"})
            actual_sells.add(_dt_key(i))
            position = 0

        # ── 空头止损（价格上涨超过止损价）──
        elif position < 0 and price >= stop_price:
            # 平空：买回股票
            cost = _buy_cost(abs(position), price)
            cash -= cost
            pnl = _sell_proceeds(abs(position), entry_price) - cost
            trades.append({"type": "COVER", "date": _dt_key(i), "price": round(price, 2),
                           "shares": abs(position), "pnl": round(pnl, 2), "reason": "空头止损"})
            actual_covers.add(_dt_key(i))
            position = 0

        # ── 无仓位 + 买入信号 → 开多（默认成交在 T+1 open）──
        elif position == 0 and row["buy_signal"]:
            fp, fi = _resolve_signal_fill(pos_i, price, i)
            if fp is None:
                continue  # 最后一根 bar，next_open 模式下丢弃信号
            pos_pct = row.get("_pos_size", params["position_pct"]) if dynamic_pos else params["position_pct"]
            shares = _max_shares(cash, fp, pos_pct)
            if shares > 0:
                cost = _buy_cost(shares, fp)
                cash -= cost
                position    = shares
                entry_price = fp
                entry_cost  = cost
                stop_price  = fp - atr_stop_mul * row["atr"]
                trades.append({"type": "BUY", "date": _dt_key(fi), "price": round(fp, 2),
                               "shares": shares, "pnl": 0, "reason": f"信号买入(仓位{pos_pct:.0%})" if dynamic_pos else "信号买入"})
                actual_buys.add(_dt_key(fi))

        # ── 空头持仓 + 买入信号 → 平空 ──
        elif position < 0 and row["buy_signal"]:
            fp, fi = _resolve_signal_fill(pos_i, price, i)
            if fp is None:
                continue
            cost = _buy_cost(abs(position), fp)
            cash -= cost
            pnl = _sell_proceeds(abs(position), entry_price) - cost
            trades.append({"type": "COVER", "date": _dt_key(fi), "price": round(fp, 2),
                           "shares": abs(position), "pnl": round(pnl, 2), "reason": "信号平空"})
            actual_covers.add(_dt_key(fi))
            position = 0

        # ── 多头持仓 + 卖出信号 → 平多（+ 可选开空）──
        elif position > 0 and row["sell_signal"]:
            fp, fi = _resolve_signal_fill(pos_i, price, i)
            if fp is None:
                continue
            proceeds = _sell_proceeds(position, fp)
            cash += proceeds
            pnl = proceeds - entry_cost
            trades.append({"type": "SELL", "date": _dt_key(fi), "price": round(fp, 2),
                           "shares": position, "pnl": round(pnl, 2), "reason": "信号卖出"})
            actual_sells.add(_dt_key(fi))
            position = 0

            # 平多后立即开空（同一根成交 bar）
            if allow_short:
                pos_pct = row.get("_pos_size", params["position_pct"]) if dynamic_pos else params["position_pct"]
                short_shares = _max_shares(cash, fp, pos_pct)
                if short_shares > 0:
                    proceeds = _sell_proceeds(short_shares, fp)
                    cash += proceeds  # 做空收到资金
                    position    = -short_shares
                    entry_price = fp
                    entry_cost  = proceeds  # short entry proceeds
                    stop_price  = fp + atr_stop_mul * row["atr"]  # 空头止损在上方
                    trades.append({"type": "SHORT", "date": _dt_key(fi), "price": round(fp, 2),
                                   "shares": short_shares, "pnl": 0, "reason": "信号做空"})
                    actual_shorts.add(_dt_key(fi))

        # ── 无仓位 + 卖出信号 + 允许做空 → 开空 ──
        elif position == 0 and row["sell_signal"] and allow_short:
            fp, fi = _resolve_signal_fill(pos_i, price, i)
            if fp is None:
                continue
            pos_pct = row.get("_pos_size", params["position_pct"]) if dynamic_pos else params["position_pct"]
            short_shares = _max_shares(cash, fp, pos_pct)
            if short_shares > 0:
                proceeds = _sell_proceeds(short_shares, fp)
                cash += proceeds
                position    = -short_shares
                entry_price = fp
                entry_cost  = proceeds
                stop_price  = fp + atr_stop_mul * row["atr"]
                trades.append({"type": "SHORT", "date": _dt_key(fi), "price": round(fp, 2),
                               "shares": short_shares, "pnl": 0, "reason": "信号做空"})
                actual_shorts.add(_dt_key(fi))

    # ── 回测结束：平掉所有仓位（仅参与资金计算，不在K线图上标记）──
    if position > 0:
        price = df["close"].iloc[-1]
        proceeds = _sell_proceeds(position, price)
        cash += proceeds
        pnl = proceeds - entry_cost
        trades.append({"type": "SELL", "date": _dt_key(df.index[-1]), "price": round(price, 2),
                       "shares": position, "pnl": round(pnl, 2), "reason": "回测结束平仓"})
        # 不加入 actual_sells，避免最后一根K线同时出现 BUY+SELL
    elif position < 0:
        price = df["close"].iloc[-1]
        cost = _buy_cost(abs(position), price)
        cash -= cost
        pnl = _sell_proceeds(abs(position), entry_price) - cost
        trades.append({"type": "COVER", "date": _dt_key(df.index[-1]), "price": round(price, 2),
                       "shares": abs(position), "pnl": round(pnl, 2), "reason": "回测结束平空"})
        # 不加入 actual_covers

    final_equity = cash
    # 补充最终权益点，确保 equity_curve 末值与 final_equity 一致
    if equity_curve:
        equity_curve[-1]["equity"] = round(final_equity, 2)
    total_return = (final_equity - capital) / capital * 100
    eq_series    = pd.Series([e["equity"] for e in equity_curve])
    peak         = eq_series.cummax()
    drawdown     = (eq_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    # 统计所有平仓交易（SELL=平多, COVER=平空）
    close_trades = [t for t in trades if t["type"] in ("SELL", "COVER")]
    sell_trades  = close_trades  # 兼容旧变量名
    win_trades   = [t for t in close_trades if t["pnl"] > 0]
    lose_trades  = [t for t in close_trades if t["pnl"] <= 0]
    win_rate     = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
    avg_win      = np.mean([t["pnl"] for t in win_trades])  if win_trades  else 0
    avg_loss     = np.mean([t["pnl"] for t in lose_trades]) if lose_trades else 0
    total_win_pnl  = sum(t["pnl"] for t in win_trades)  if win_trades  else 0
    total_loss_pnl = sum(t["pnl"] for t in lose_trades) if lose_trades else 0
    profit_factor  = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else 0

    daily_returns = eq_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    bh_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    # 构建完整图表数据（OHLCV + 所有指标列）
    chart_cols = ["open", "high", "low", "close", "volume", "buy_signal", "sell_signal"]
    # 包含所有数值型指标列
    for col in df.columns:
        if col not in chart_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            chart_cols.append(col)
    chart_df = df[chart_cols].copy()
    # 判断是否分钟/小时级数据（用于时间格式选择）
    idx = chart_df.index
    is_intraday = False
    if len(idx) >= 2:
        diff = (idx[1] - idx[0]).total_seconds()
        is_intraday = diff < 86400  # 小于1天 = 分钟/小时级

    if is_intraday:
        # 分钟/小时K线：用Unix时间戳（TradingView需要）
        chart_df["_ts"] = chart_df.index.map(lambda x: int(x.timestamp()))
    else:
        chart_df.index = chart_df.index.strftime("%Y-%m-%d")

    chart_records = []
    for i, (date_idx, row) in enumerate(chart_df.iterrows()):
        if is_intraday:
            rec = {"date": int(row["_ts"])}
            date_key = str(chart_df.index[i])[:19]  # for actual_buys matching
        else:
            rec = {"date": date_idx}
            date_key = date_idx
        for col in chart_cols:
            v = row[col]
            if isinstance(v, (bool, np.bool_)):
                rec[col] = bool(v)
            elif pd.isna(v):
                rec[col] = None
            else:
                rec[col] = round(float(v), 4)
        # 添加实际成交标记
        rec["actual_buy"] = date_key in actual_buys
        rec["actual_sell"] = date_key in actual_sells
        rec["actual_short"] = date_key in actual_shorts
        rec["actual_cover"] = date_key in actual_covers
        chart_records.append(rec)

    # 边缘检测：只保留 False→True 转换点，避免状态型条件连续标记
    for idx in range(len(chart_records)):
        rec = chart_records[idx]
        if rec.get("buy_signal"):
            prev_buy = chart_records[idx - 1].get("buy_signal") if idx > 0 else False
            if prev_buy:
                rec["buy_signal"] = False
        if rec.get("sell_signal"):
            prev_sell = chart_records[idx - 1].get("sell_signal") if idx > 0 else False
            if prev_sell:
                rec["sell_signal"] = False

    # 持仓状态标记：基于实际成交追踪持仓
    # 持仓中再次触发买入  → buy_again  (图上显示小 B)
    # 空仓中再次触发卖出  → sell_again (图上显示小 S)
    _sim_pos = 0  # 0=空仓, 1=多头, -1=空头
    for rec in chart_records:
        has_buy      = rec.get("buy_signal",  False)
        has_sell     = rec.get("sell_signal", False)
        is_actual_buy  = rec.get("actual_buy",  False)
        is_actual_sell = rec.get("actual_sell", False)
        is_actual_short = rec.get("actual_short", False)
        is_actual_cover = rec.get("actual_cover", False)

        if has_buy:
            if _sim_pos == 1 and not is_actual_buy:
                # 多头持仓中再次触发买入 → buy_again
                rec["buy_signal"] = False
                rec["buy_again"]  = True
            # 空仓/空头首次买入：buy_signal 保持

        if has_sell:
            # is_actual_buy 这一项不能少: _sim_pos 反映的是**上一根**收盘后的持仓
            # (下面那段更新写在这个判断之后), 而这一根开盘已经买进去了。不排除的话,
            # 一根既 actual_buy 又恰好触发卖出条件的 K 线会被标成"空仓中再次触发
            # 卖出", 图上就是同一根同时出现绿色 BUY 箭头和橙色 S —— 看着像指标自相
            # 矛盾。(864 行那句"不加入 actual_sells，避免最后一根K线同时出现
            #  BUY+SELL"堵的是 actual_sell 那扇门, sell_again 这扇当时没堵。)
            if (_sim_pos in (0, -1) and not is_actual_buy
                    and not is_actual_sell and not is_actual_short):
                # 空仓或空头中再次触发卖出 → sell_again
                rec["sell_signal"] = False
                rec["sell_again"]  = True
            elif is_actual_buy:
                # 刚买进就触发卖出条件: 不画重复标记, 也不当成本根的卖出信号
                # (真要卖, 下一根的 sell_signal 会自己冒出来)
                rec["sell_signal"] = False
            # 多头首次卖出：sell_signal 保持

        # 持仓状态由实际成交驱动
        if is_actual_buy or is_actual_cover:
            _sim_pos = 0 if is_actual_cover and not is_actual_buy else 1
        if is_actual_sell:
            _sim_pos = 0
        if is_actual_short:
            _sim_pos = -1

    return {
        "summary": {
            "capital":        capital,
            "final_equity":   round(final_equity, 2),
            "total_return":   round(total_return, 2),
            "bh_return":      round(bh_return, 2),
            "max_drawdown":   round(max_drawdown, 2),
            "sharpe":         round(sharpe, 2),
            "win_rate":       round(win_rate, 2),
            "profit_factor":  round(profit_factor, 2),
            "total_trades":   len(sell_trades),
            "win_trades":     len(win_trades),
            "lose_trades":    len(lose_trades),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
        },
        "equity_curve": equity_curve[-200:],
        "trades":        trades[-50:],
        "chart_data":    chart_records,
        # 原来这里还有一份 "indicators": df[...].tail(100) —— 前端**全文件 0 处
        # 引用**(它读的 chart_indicators 是另一个字段, 存的是画图配置)。
        # 实测每次 ~19-20KB, 而 chart_data 里已经有同样的 OHLCV+信号列且更全。
        # /backtest 和 /compare 各走一遍, 远程访问时很明显。
    }


def run_strategy_backtest(df_raw: pd.DataFrame, strategy_name: str,
                          params_override: dict = None,
                          capital: float = 100000,
                          commission: float = 0.001,
                          position_pct: float = 0.95,
                          dynamic_position: bool = False,
                          allow_short: bool = False,
                          buy_params: dict = None,
                          sell_params: dict = None) -> dict:
    """用指定策略跑回测，返回结果 dict
    dynamic_position: 是否启用ATR动态仓位管理（波动大减仓，波动小加仓）
    allow_short: 是否允许做空（卖出信号时开空头仓位）
    buy_params/sell_params: 当两者都提供时，buy信号用buy_params生成，sell信号用sell_params生成（双参数分离模式）
    """
    strategy = get_strategy(strategy_name)

    # ── 双参数分离模式 ──
    dual_mode = buy_params is not None and sell_params is not None

    if dual_mode:
        # 用 buy_params 跑一遍生成 buy_signal
        bp = {**strategy.default_params, **buy_params}
        bp.setdefault("capital", capital)
        bp.setdefault("commission", commission)
        bp.setdefault("position_pct", position_pct)
        bp.setdefault("atr_stop_mul", bp.get("atr_stop_mul", 2.0))
        if dynamic_position or bp.get("dynamic_position", False):
            bp["dynamic_position"] = True
        if allow_short or bp.get("allow_short", False):
            bp["allow_short"] = True

        df_buy = strategy.add_indicators(df_raw, bp)
        if len(df_buy) < 2:
            raise ValueError(f"数据不足：{strategy_name} 指标需要更多历史数据（仅剩{len(df_buy)}根）")
        df_buy = strategy.generate_signals(df_buy, bp)

        # 用 sell_params 跑一遍生成 sell_signal
        sp = {**strategy.default_params, **sell_params}
        sp.setdefault("capital", capital)
        sp.setdefault("commission", commission)
        sp.setdefault("position_pct", position_pct)
        sp.setdefault("atr_stop_mul", sp.get("atr_stop_mul", 2.0))
        if dynamic_position or sp.get("dynamic_position", False):
            sp["dynamic_position"] = True
        if allow_short or sp.get("allow_short", False):
            sp["allow_short"] = True

        df_sell = strategy.add_indicators(df_raw, sp)
        df_sell = strategy.generate_signals(df_sell, sp)

        # 合并：以 buy 运行结果为主体 DataFrame（指标列来自 buy_params），
        # 仅替换 sell_signal 为 sell_params 生成的版本
        common_idx = df_buy.index.intersection(df_sell.index)
        df = df_buy.loc[common_idx].copy()
        df["sell_signal"] = df_sell.loc[common_idx, "sell_signal"]

        # 回测用的 params 以 buy_params 为主（控制 capital/commission/position 等交易参数）
        params = bp
        result = run_backtest(df, params)
    else:
        # ── 传统单参数模式 ──
        params = {**strategy.default_params, **(params_override or {})}
        params.setdefault("capital", capital)
        params.setdefault("commission", commission)
        params.setdefault("position_pct", position_pct)
        params.setdefault("atr_stop_mul", params.get("atr_stop_mul", 2.0))
        if dynamic_position or params.get("dynamic_position", False):
            params["dynamic_position"] = True
        if allow_short or params.get("allow_short", False):
            params["allow_short"] = True

        df = strategy.add_indicators(df_raw, params)
        if len(df) < 2:
            raise ValueError(f"数据不足：{strategy_name} 指标需要更多历史数据（仅剩{len(df)}根）")
        df = strategy.generate_signals(df, params)
        result = run_backtest(df, params)

    # 补回预热期K线（指标计算前被dropna丢弃的数据），确保图表显示完整历史
    # 注意：只补回 chart_data 第一条日期之前的数据，避免 dropna 删除中间行导致日期乱序
    warmup_idx = df_raw.index.difference(df.index)
    if len(warmup_idx) > 0:
        # 确定 chart_data 中最早的日期，只补这之前的数据
        first_chart_date = df.index.min() if len(df) > 0 else None
        warmup_records = []
        _is_intra = False
        if len(df_raw) >= 2:
            _is_intra = (df_raw.index[1] - df_raw.index[0]).total_seconds() < 86400
        for idx in sorted(warmup_idx):
            if first_chart_date is not None and idx >= first_chart_date:
                continue  # 跳过 chart_data 日期范围内被 dropna 删除的中间行
            row = df_raw.loc[idx]
            if _is_intra:
                rec = {"date": int(idx.timestamp())}
            else:
                rec = {"date": str(idx.date())}
            rec.update({
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": int(row["volume"]),
                "buy_signal": False, "sell_signal": False,
                "actual_buy": False, "actual_sell": False,
                "actual_short": False, "actual_cover": False,
            })
            warmup_records.append(rec)
        result["chart_data"] = warmup_records + result["chart_data"]

    result["strategy"] = strategy_name
    result["strategy_desc"] = strategy.description
    result["params"] = params
    result["chart_indicators"] = strategy.chart_indicators
    result["indicator_formulas"] = strategy.indicator_formulas
    result["signal_rules"] = strategy.signal_rules
    if dual_mode:
        result["dual_signal_params"] = True
        result["buy_params_used"] = buy_params
        result["sell_params_used"] = sell_params
    return result


# ── BUY 点 MFE/MAE 分析 ──────────────────────────────────────────

def analyze_buy_points(df_raw: pd.DataFrame, strategy_name: str,
                       params_override: dict = None,
                       window: int = 20,
                       entry_mode: str = "next_open",
                       judge_mode: str = "mfe_mae",
                       up_pct: float = None,
                       dn_pct: float = None,
                       same_bar: str = "adverse_first") -> dict:
    """对策略的 BUY 信号做前瞻 MFE/MAE 分析，不执行 SELL 逻辑。

    judge_mode — 买点算不算"成功"的两种口径:

      "mfe_mae"(原有, 默认)
          mfe >= 目标 且 |mae| <= 上限。**跟先后顺序无关** —— 只要窗口内
          最高点够高、最低点没破下限就算成功。

      "first_touch"(超短线, 先到先判)
          逐根往后扫, 谁先碰到谁说了算:
            · 先涨到 +up_pct%  → 成功(后面再怎么回撤都不改判)
            · 先跌到 -dn_pct%  → 失败(后面再怎么涨也不改判)
            · 窗口内两边都没碰到 → 失败
          这跟 mfe_mae 会给出**不同**结论: 一个买点先跌 12% 再涨 30%,
          mfe_mae 看 mae 破了上限判失败, first_touch 也判失败(先跌);
          但先涨 6% 再跌 20% 的, mfe_mae 判失败(mae 破限), first_touch
          判成功(先涨到位)。超短线关心的是后者。

    same_bar — 同一根 K 线里 high 和 low **两边都碰到**时算谁先。
        OHLC 只有四个价, **真实先后顺序无从得知**, 只能约定:
          "adverse_first"(默认) 算先跌 → 失败。悲观, 不会高估成功率。
          "favorable_first"     算先涨 → 成功。乐观, 会系统性高估。
        默认取悲观的那个 —— 回测宁可低估也不要骗自己。

    Parameters
    ----------
    df_raw : OHLCV DataFrame (DatetimeIndex)
    strategy_name : 策略名
    params_override : 参数覆盖
    window : 前瞻 K 线数（默认 20）
    entry_mode : "next_open" = 次日开盘价入场（推荐）；"signal_close" = 信号日收盘价

    Returns
    -------
    dict  { "buy_points": [...], "stats": {...}, "chart_data": [...] }
    """
    strategy = get_strategy(strategy_name)
    params = {**strategy.default_params, **(params_override or {})}

    # 1) 计算指标 + 信号
    df = strategy.add_indicators(df_raw.copy(), params)
    if len(df) < 2:
        return {"buy_points": [], "stats": _empty_bp_stats(), "chart_data": []}
    df = strategy.generate_signals(df, params)

    # 2) 边缘检测：只保留 False→True 的买入信号
    buy_mask = df["buy_signal"].astype(bool).values
    edges = []
    for i in range(len(buy_mask)):
        if buy_mask[i] and (i == 0 or not buy_mask[i - 1]):
            edges.append(i)

    # 3) 判断日线 vs 分钟线
    _is_intra = False
    if len(df) >= 2:
        _is_intra = (df.index[1] - df.index[0]).total_seconds() < 86400

    def _dt_str(idx):
        return str(idx)[:19] if _is_intra else str(idx.date())

    # 4) 对每个买入信号做前瞻分析
    buy_points = []
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    for sig_i in edges:
        # 入场价
        if entry_mode == "next_open":
            if sig_i + 1 >= len(df):
                continue  # 最后一根K线无法次日入场
            entry_bar = sig_i + 1
            entry_price = float(opens[entry_bar])
        else:  # signal_close
            entry_bar = sig_i
            entry_price = float(closes[entry_bar])

        if entry_price <= 0:
            continue

        # 前瞻窗口
        end_bar = min(entry_bar + window, len(df) - 1)
        actual_window = end_bar - entry_bar
        if actual_window < 1:
            continue
        truncated = (actual_window < window)

        # 从 entry_bar+1 开始看（入场当天不算前瞻）
        fwd_start = entry_bar + 1
        fwd_end = end_bar + 1  # exclusive
        if fwd_start >= len(df):
            continue

        fwd_highs = highs[fwd_start:fwd_end]
        fwd_lows = lows[fwd_start:fwd_end]
        fwd_closes = closes[fwd_start:fwd_end]

        if len(fwd_highs) == 0:
            continue

        # MFE / MAE
        mfe_abs = float(np.max(fwd_highs) - entry_price)
        mae_abs = float(np.min(fwd_lows) - entry_price)
        mfe_pct = round(mfe_abs / entry_price * 100, 2)
        mae_pct = round(mae_abs / entry_price * 100, 2)

        # MFE/MAE 对应的日期和 bar 偏移
        mfe_idx = int(np.argmax(fwd_highs))
        mae_idx = int(np.argmin(fwd_lows))
        bars_to_mfe = mfe_idx + 1  # 从入场开始算第几根
        bars_to_mae = mae_idx + 1

        mfe_date = _dt_str(df.index[fwd_start + mfe_idx])
        mae_date = _dt_str(df.index[fwd_start + mae_idx])

        # R 倍数
        r_ratio = round(abs(mfe_pct / mae_pct), 2) if mae_pct != 0 else 99.99

        # 窗口结束时收益
        end_return = round((float(fwd_closes[-1]) - entry_price) / entry_price * 100, 2)

        # ── 先到先判 (超短线口径) ──────────────────────────────────
        _ft = {}
        if judge_mode == "first_touch" and up_pct is not None and dn_pct is not None:
            up_line = entry_price * (1.0 + float(up_pct) / 100.0)
            dn_line = entry_price * (1.0 - float(dn_pct) / 100.0)
            _iu = np.nonzero(fwd_highs >= up_line)[0]
            _id = np.nonzero(fwd_lows <= dn_line)[0]
            iu = int(_iu[0]) if len(_iu) else None
            idn = int(_id[0]) if len(_id) else None
            if iu is None and idn is None:
                ok, outcome, at = False, "none", None            # 窗口内没碰到任何一边
            elif idn is None:
                ok, outcome, at = True, "up", iu
            elif iu is None:
                ok, outcome, at = False, "down", idn
            elif iu < idn:
                ok, outcome, at = True, "up", iu                 # 先涨到位
            elif idn < iu:
                ok, outcome, at = False, "down", idn             # 先跌破
            else:
                # 同一根里两边都碰到, OHLC 看不出先后 —— 按 same_bar 约定
                ok = (same_bar == "favorable_first")
                outcome, at = "same_bar", iu
            _ft = {"success": ok, "outcome": outcome,
                   "bars_to_outcome": (at + 1) if at is not None else None}

        buy_points.append({
            "date": _dt_str(df.index[sig_i]),
            "entry_date": _dt_str(df.index[entry_bar]),
            "entry_price": round(entry_price, 4),
            "mfe": mfe_pct,
            "mae": mae_pct,
            "r_ratio": r_ratio,
            "mfe_date": mfe_date,
            "mae_date": mae_date,
            "bars_to_mfe": bars_to_mfe,
            "bars_to_mae": bars_to_mae,
            "close_at_end": round(float(fwd_closes[-1]), 4),
            "end_return": end_return,
            "truncated": truncated,
            **_ft,
        })

    # 5) 统计
    stats = _calc_bp_stats(buy_points)

    # 6) chart_data（带买点标记的 K 线数据）
    bp_dates = {bp["date"] for bp in buy_points}
    chart_data = []
    for i in range(len(df)):
        row = df.iloc[i]
        dt = _dt_str(df.index[i])
        rec = {
            "date": int(df.index[i].timestamp()) if _is_intra else dt,
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row["volume"]),
            "buy_signal": bool(buy_mask[i] and (i == 0 or not buy_mask[i - 1])),
        }
        # 附加买点详情（前端用于渲染标记）
        if dt in bp_dates:
            bp = next(b for b in buy_points if b["date"] == dt)
            rec["bp_mfe"] = bp["mfe"]
            rec["bp_mae"] = bp["mae"]
            rec["bp_r_ratio"] = bp["r_ratio"]
            rec["bp_success"] = None  # 由前端根据阈值判定
        chart_data.append(rec)

    return {
        "buy_points": buy_points,
        "stats": stats,
        "chart_data": chart_data,
    }


def analyze_sell_points(df_raw: pd.DataFrame, strategy_name: str,
                        params_override: dict = None,
                        window: int = 20,
                        judge_mode: str = "mfe_mae",
                        down_pct: float = None,
                        up_pct: float = None,
                        same_bar: str = "adverse_first") -> dict:
    """对策略的 SELL 信号做前瞻 MFE/MAE 分析（镜像 analyze_buy_points）。

    MFE = 卖出后最大跌幅（有利方向 = 价格继续下跌，说明卖对了）
    MAE = 卖出后最大反弹（不利方向 = 价格反弹上涨，说明卖早了）

    judge_mode — 卖点算不算"成功"的两种口径 (跟买点那边镜像):

      "mfe_mae"(原有, 默认)
          mfe >= 目标 且 mae <= 上限。**跟先后顺序无关**。

      "first_touch"(超短线, 先到先判)
          卖出后逐根往后扫, 谁先碰到谁说了算:
            · 先跌到 -down_pct% → 成功 (卖对了, 后面再怎么反弹都不改判)
            · 先涨到 +up_pct%   → 失败 (卖早了, 后面再怎么跌也不改判)
            · 窗口内两边都没碰到 → 失败
          注意方向跟买点是**反的**: 买点是"先涨算成功", 卖点是"先跌算成功"。

    same_bar — 同一根里 high 和 low 两边都碰到时算谁先。OHLC 看不出真实
        先后, 只能约定。默认 "adverse_first" 取悲观的那个(卖点的不利方向是
        反弹, 所以悲观 = 算先反弹 = 判失败)。
    """
    strategy = get_strategy(strategy_name)
    params = {**strategy.default_params, **(params_override or {})}

    df = strategy.add_indicators(df_raw.copy(), params)
    if len(df) < 2:
        return {"sell_points": [], "stats": _empty_sp_stats(), "chart_data": []}
    df = strategy.generate_signals(df, params)

    # 边缘检测：sell_signal 的 False→True
    sell_mask = df["sell_signal"].astype(bool).values
    edges = []
    for i in range(len(sell_mask)):
        if sell_mask[i] and (i == 0 or not sell_mask[i - 1]):
            edges.append(i)

    _is_intra = False
    if len(df) >= 2:
        _is_intra = (df.index[1] - df.index[0]).total_seconds() < 86400

    def _dt_str(idx):
        return str(idx)[:19] if _is_intra else str(idx.date())

    sell_points = []
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for sig_i in edges:
        exit_price = float(closes[sig_i])  # 卖出价 = 信号日收盘
        if exit_price <= 0:
            continue

        end_bar = min(sig_i + window, len(df) - 1)
        actual_window = end_bar - sig_i
        if actual_window < 1:
            continue
        truncated = (actual_window < window)

        fwd_start = sig_i + 1
        fwd_end = end_bar + 1
        if fwd_start >= len(df):
            continue

        fwd_highs = highs[fwd_start:fwd_end]
        fwd_lows = lows[fwd_start:fwd_end]
        fwd_closes = closes[fwd_start:fwd_end]

        if len(fwd_lows) == 0:
            continue

        # MFE = 卖出后最大跌幅（有利方向）= (exit_price - min_low) / exit_price
        mfe_abs = float(exit_price - np.min(fwd_lows))
        # MAE = 卖出后最大反弹（不利方向）= (max_high - exit_price) / exit_price
        mae_abs = float(np.max(fwd_highs) - exit_price)

        mfe_pct = round(mfe_abs / exit_price * 100, 2)  # 正值 = 有利
        mae_pct = round(mae_abs / exit_price * 100, 2)   # 正值 = 不利

        mfe_idx = int(np.argmin(fwd_lows))
        mae_idx = int(np.argmax(fwd_highs))
        bars_to_mfe = mfe_idx + 1
        bars_to_mae = mae_idx + 1

        mfe_date = _dt_str(df.index[fwd_start + mfe_idx])
        mae_date = _dt_str(df.index[fwd_start + mae_idx])

        r_ratio = round(mfe_pct / mae_pct, 2) if mae_pct > 0 else 99.99

        end_return = round((exit_price - float(fwd_closes[-1])) / exit_price * 100, 2)

        # ── 先到先判 (超短线口径, 方向跟买点相反) ──────────────────
        _ft = {}
        if judge_mode == "first_touch" and down_pct is not None and up_pct is not None:
            dn_line = exit_price * (1.0 - float(down_pct) / 100.0)   # 跌到这 = 卖对了
            up_line = exit_price * (1.0 + float(up_pct) / 100.0)     # 涨到这 = 卖早了
            _id = np.nonzero(fwd_lows <= dn_line)[0]
            _iu = np.nonzero(fwd_highs >= up_line)[0]
            idn = int(_id[0]) if len(_id) else None
            iu = int(_iu[0]) if len(_iu) else None
            if idn is None and iu is None:
                ok, outcome, at = False, "none", None
            elif iu is None:
                ok, outcome, at = True, "down", idn        # 只跌到 → 卖对了
            elif idn is None:
                ok, outcome, at = False, "up", iu          # 只反弹 → 卖早了
            elif idn < iu:
                ok, outcome, at = True, "down", idn        # 先跌
            elif iu < idn:
                ok, outcome, at = False, "up", iu          # 先反弹
            else:
                # 同一根里两边都碰到: 悲观 = 算先反弹(不利方向) = 失败
                ok = (same_bar == "favorable_first")
                outcome, at = "same_bar", idn
            _ft = {"success": ok, "outcome": outcome,
                   "bars_to_outcome": (at + 1) if at is not None else None}

        sell_points.append({
            "date": _dt_str(df.index[sig_i]),
            "exit_price": round(exit_price, 4),
            "mfe": mfe_pct,
            "mae": mae_pct,
            "r_ratio": r_ratio,
            "mfe_date": mfe_date,
            "mae_date": mae_date,
            "bars_to_mfe": bars_to_mfe,
            "bars_to_mae": bars_to_mae,
            "close_at_end": round(float(fwd_closes[-1]), 4),
            "end_return": end_return,
            "truncated": truncated,
            **_ft,
        })

    stats = _calc_sp_stats(sell_points)

    sp_dates = {sp["date"] for sp in sell_points}
    chart_data = []
    for i in range(len(df)):
        row = df.iloc[i]
        dt = _dt_str(df.index[i])
        rec = {
            "date": int(df.index[i].timestamp()) if _is_intra else dt,
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row["volume"]),
            "sell_signal": bool(sell_mask[i] and (i == 0 or not sell_mask[i - 1])),
        }
        if dt in sp_dates:
            sp = next(s for s in sell_points if s["date"] == dt)
            rec["sp_mfe"] = sp["mfe"]
            rec["sp_mae"] = sp["mae"]
            rec["sp_r_ratio"] = sp["r_ratio"]
        chart_data.append(rec)

    return {"sell_points": sell_points, "stats": stats, "chart_data": chart_data}


def _empty_sp_stats() -> dict:
    """空的 SELL 点统计。"""
    return {
        "total_signals": 0,
        "avg_mfe": 0, "avg_mae": 0,
        "median_mfe": 0, "median_mae": 0,
        "avg_r_ratio": 0, "avg_bars_to_mfe": 0,
        "mfe_distribution": {">20%": 0, "10-20%": 0, "5-10%": 0, "<5%": 0},
        "mae_distribution": {"<3%": 0, "3-5%": 0, ">5%": 0},
    }


def _calc_sp_stats(sell_points: list) -> dict:
    """汇总 SELL 点统计指标。"""
    if not sell_points:
        return _empty_sp_stats()
    mfes = [sp["mfe"] for sp in sell_points]
    maes = [sp["mae"] for sp in sell_points]
    rs = [sp["r_ratio"] for sp in sell_points]
    bars = [sp["bars_to_mfe"] for sp in sell_points]
    n = len(sell_points)
    out = {
        "total_signals": len(sell_points),
        "avg_mfe": round(float(np.mean(mfes)), 2),
        "avg_mae": round(float(np.mean(maes)), 2),
        "median_mfe": round(float(np.median(mfes)), 2),
        "median_mae": round(float(np.median(maes)), 2),
        "avg_r_ratio": round(float(np.mean(rs)), 2),
        "avg_bars_to_mfe": round(float(np.mean(bars)), 1),
        "mfe_distribution": {
            ">20%": sum(1 for m in mfes if m > 20),
            "10-20%": sum(1 for m in mfes if 10 < m <= 20),
            "5-10%": sum(1 for m in mfes if 5 < m <= 10),
            "<5%": sum(1 for m in mfes if m <= 5),
        },
        "mae_distribution": {
            "<3%": sum(1 for m in maes if m < 3),
            "3-5%": sum(1 for m in maes if 3 <= m < 5),
            ">5%": sum(1 for m in maes if m >= 5),
        },
    }
    # 先到先判专用: 多少根之内分出胜负。跟买点那边同一个道理 ——
    # 「最优窗口」在这个口径下恒等于试过的最长窗口, 没有参考价值。
    decided = [sp.get("bars_to_outcome") for sp in sell_points
               if sp.get("bars_to_outcome") is not None]
    if decided:
        out["decided_count"] = len(decided)
        out["decided_pct"] = round(len(decided) / n * 100, 1)
        out["median_bars_to_outcome"] = round(float(np.median(decided)), 1)
    return out


def _empty_bp_stats() -> dict:
    """空的 BUY 点统计。"""
    return {
        "total_signals": 0,
        "avg_mfe": 0, "avg_mae": 0,
        "median_mfe": 0, "median_mae": 0,
        "avg_r_ratio": 0, "avg_bars_to_mfe": 0,
        "mfe_distribution": {">30%": 0, "20-30%": 0, "10-20%": 0, "<10%": 0},
        "mae_distribution": {"<5%": 0, "5-10%": 0, ">10%": 0},
    }


def _calc_bp_stats(buy_points: list) -> dict:
    """汇总 BUY 点统计指标。"""
    if not buy_points:
        return _empty_bp_stats()

    mfes = [bp["mfe"] for bp in buy_points]
    maes = [bp["mae"] for bp in buy_points]
    rs = [bp["r_ratio"] for bp in buy_points]
    bars = [bp["bars_to_mfe"] for bp in buy_points]

    n = len(buy_points)
    # MFE 分布
    mfe_dist = {
        ">30%": sum(1 for m in mfes if m > 30),
        "20-30%": sum(1 for m in mfes if 20 < m <= 30),
        "10-20%": sum(1 for m in mfes if 10 < m <= 20),
        "<10%": sum(1 for m in mfes if m <= 10),
    }
    # MAE 分布（mae 是负值，用 abs）
    mae_dist = {
        "<5%": sum(1 for m in maes if abs(m) < 5),
        "5-10%": sum(1 for m in maes if 5 <= abs(m) < 10),
        ">10%": sum(1 for m in maes if abs(m) >= 10),
    }

    out = {
        "total_signals": n,
        "avg_mfe": round(float(np.mean(mfes)), 2),
        "avg_mae": round(float(np.mean(maes)), 2),
        "median_mfe": round(float(np.median(mfes)), 2),
        "median_mae": round(float(np.median(maes)), 2),
        "avg_r_ratio": round(float(np.mean(rs)), 2),
        "avg_bars_to_mfe": round(float(np.mean(bars)), 1),
        "mfe_distribution": mfe_dist,
        "mae_distribution": mae_dist,
    }
    # 先到先判专用: 多少根之内分出胜负。
    # 「最优窗口」那一列在先到先判下是**没有意义**的 —— 成功率随窗口单调递增
    # (涨阈值比跌阈值近, 窗口越长, 原本"两边都没碰到"的失败越容易变成"先涨到"),
    # 所以它永远等于试过的最长窗口, 跟用户设的窗口无关。实测 5/10/20/40/60 →
    # 52.1 / 68.8 / 75.0 / 75.0 / 77.1%, 恒选 60。
    # 真正有用的是这个: 分出胜负的那些买点, 中位数第几根出的结果。
    decided = [bp.get("bars_to_outcome") for bp in buy_points
               if bp.get("bars_to_outcome") is not None]
    if decided:
        out["decided_count"] = len(decided)
        out["decided_pct"] = round(len(decided) / n * 100, 1)
        out["median_bars_to_outcome"] = round(float(np.median(decided)), 1)
    return out
