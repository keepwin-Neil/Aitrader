"""
Trading router: /backtest, /compare, /walkforward, /optimize, /batch-optimize,
                /batch-wf, /compare-optimized, /applicability, /portfolio,
                /paper-trade, /auto-optimize, /wf-status
"""
from fastapi import APIRouter, HTTPException, Query, Request
from datetime import datetime
import threading

import main as _m
import proc_role

router = APIRouter()


def _prepend_history(result: dict, symbol: str, market: str, interval: str,
                     period: str, chart_period, bt_df):
    """把 chart_period 这一段更早的历史当作**纯背景K线**贴到 chart_data 前面。

    只有 OHLCV, 没有指标也没有信号 —— 指标必须和回测用同一段数据算, 否则
    递归型指标在同一个日期上会给出不同的值, 图上的信号就跟排名表对不上了。
    额外返回 signal_start: 信号区间的第一根日期, 前端可以据此画一条分界线,
    让用户一眼看出"这条线左边只是历史行情, 没跑过信号"。
    """
    if not chart_period or chart_period == period:
        return result
    cd = result.get("chart_data") or []
    if not cd:
        return result
    try:
        from backtest import get_ohlcv
        long_df = get_ohlcv(symbol, market, interval=interval, period=chart_period)
    except Exception as e:
        _m.logger.warning(f"[backtest] 长历史取数失败 {symbol} {chart_period}: {e}")
        return result
    if long_df is None or len(long_df) <= len(bt_df):
        return result        # 不比回测区间长, 没必要拼

    is_intraday = interval not in ("1d", "1wk", "1mo")
    first = cd[0].get("date")
    extra = []
    for ts, r in long_df.iterrows():
        d = int(ts.timestamp()) if is_intraday else ts.strftime("%Y-%m-%d")
        if (d >= first) if not is_intraday else (d >= first):
            break        # 到回测区间了, 后面用带信号的那份
        extra.append({"date": d,
                      "open": round(float(r["open"]), 4),
                      "high": round(float(r["high"]), 4),
                      "low": round(float(r["low"]), 4),
                      "close": round(float(r["close"]), 4),
                      "volume": int(r["volume"])})
    if not extra:
        return result
    result["chart_data"] = extra + cd
    result["signal_start"] = first
    result["history_bars"] = len(extra)
    return result


@router.post("/backtest/{market}/{symbol}")
def backtest(market: str, symbol: str, body: dict = None):
    """运行回测，支持指定策略和K线周期"""
    from backtest import get_ohlcv
    from backtest import run_strategy_backtest

    if body is None:
        body = {}
    strategy_name = body.pop("strategy", "multi_indicator")
    period = body.pop("period", "2y")
    # chart_period: 只影响**画多少根K线**, 不影响回测/信号。
    # 为什么要分开: 递归型指标(McGinley/自适应均线这类)每根的值依赖前一根,
    # 喂 1460 根和喂 500 根, 同一个日期上的指标值就是两个数 —— 图表要是自作主张
    # 用 max 去算, 画出来的信号跟排名表(按所选 period 算)根本不是一回事。
    # 所以: 信号按 period 算, 前面那段更早的历史只当背景K线贴上去, 不带信号。
    chart_period = str(body.pop("chart_period", "") or "").strip() or None
    interval = body.pop("interval", "1d")
    dynamic_position = body.pop("dynamic_position", False)
    allow_short = body.pop("allow_short", False)
    # param_source="standard" → 用个股标准(有则)/通用标准, 与 /compare 排名口径一致.
    # 不传 (默认 "") → 保持老行为: body 里剩下的键当 params_override, 空则策略 default_params.
    param_source = str(body.pop("param_source", "") or "").strip()
    # 与 /compare 对齐的实时口径 (含当日 bar), 否则弹出的 K 线会比面板少一根
    include_today = bool(body.pop("include_today", False))
    try:
        _bt_max_age = int(body.pop("max_age_sec")) if body.get("max_age_sec") is not None else None
    except (TypeError, ValueError):
        _bt_max_age = None
    body.pop("max_age_sec", None)
    INTERVAL_MAX_PERIOD = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "2y", "60m": "2y", "4h": "2y",
    }
    if interval in INTERVAL_MAX_PERIOD:
        period = INTERVAL_MAX_PERIOD[interval]
    # ── 双参数分离：检查是否有独立的 buy/sell 参数 ──
    dual = _m._get_dual_signal_params(symbol.upper(), strategy_name, interval=interval)
    _bt_mtime = _m._data_mtime_sig([symbol], market, interval)
    _dual_tag = "dual" if dual["has_dual"] else "single"
    _ps_tag = param_source or "raw"
    _ps_mtime = ""
    if param_source == "standard":
        # 参数文件变了要让缓存失效
        _cust_fp = _m.CUSTOM_PARAMS_DIR / f"{symbol.upper()}.json"
        _univ_fp = _m.UNIVERSAL_PARAMS_FILE
        _ps_mtime = str(int(_cust_fp.stat().st_mtime) // 60) if _cust_fp.exists() else "0"
        if _univ_fp.exists():
            _ps_mtime += "-" + str(int(_univ_fp.stat().st_mtime) // 60)
    _rt_tag = "0"
    _bt_ttl = 86400 * 7
    df_rt = None
    if include_today:
        _bt_ttl = 60
        try:
            df_rt = get_ohlcv(symbol, market, interval=interval, period=period,
                              include_today=True, max_age_sec=_bt_max_age)
            _rt_tag = (f"{df_rt.index[-1].isoformat()}@{round(float(df_rt['close'].iloc[-1]), 4)}"
                       if len(df_rt) else "rt")
        except Exception:
            _rt_tag = "rt"
    # period 必须进 key: 它在上面已经被 pop 出 body 了, 而 body 的 hash 是 key
    # 的最后一段 —— 不单独带上的话, 同一个策略换周期会命中上一次的结果。
    # 实测: 先请求 2y 拿到 500 根, 再请求 max 仍返回 500 根, 而 TTL 是 7 天,
    # 也就是谁先请求谁说了算、错一周。
    # 对递归型指标(McGinley 之类)这不只是"少几根K线", 信号本身就是错的。
    cache_key = f"bt:{market}:{symbol}:{strategy_name}:{interval}:{period}:{chart_period or '-'}:{dynamic_position}:{allow_short}:{_dual_tag}:{_ps_tag}:{_ps_mtime}:{_rt_tag}:{_bt_mtime}:{_m.stable_hash(sorted(body.items()))}"
    cached = _m.cache_get(cache_key, ttl=_bt_ttl)
    if cached:
        return cached
    try:
        df = df_rt if df_rt is not None else get_ohlcv(symbol, market, interval=interval, period=period)
        min_bars = 1
        if len(df) < min_bars:
            raise HTTPException(400, f"数据不足，只有 {len(df)} 条K线（需要至少 {min_bars} 条）")
        try:
            if dual["has_dual"] and not body:
                # 双参数分离模式：buy信号用buy_params，sell信号用sell_params
                result = run_strategy_backtest(df, strategy_name,
                                               buy_params=dual["buy_params"],
                                               sell_params=dual["sell_params"],
                                               dynamic_position=dynamic_position,
                                               allow_short=allow_short)
            elif param_source == "standard" and not body:
                # 个股标准(有则) → 通用标准, 与 /compare 的排名口径完全一致
                result = run_strategy_backtest(
                    df, strategy_name,
                    params_override=_m._get_standard_strategy_params(
                        symbol.upper(), strategy_name, interval=interval),
                    dynamic_position=dynamic_position,
                    allow_short=allow_short)
            else:
                # 传统单参数模式（用户手动传参 或 没有分离参数）
                result = run_strategy_backtest(df, strategy_name, params_override=body,
                                               dynamic_position=dynamic_position,
                                               allow_short=allow_short)
        except (IndexError, ValueError, KeyError) as e:
            _m.logger.warning(f"Strategy indicators failed for {symbol}/{interval} ({len(df)} bars): {e}")
            chart_data = []
            for d, r in df.iterrows():
                if interval in ("1d", "1wk", "1mo"):
                    date_val = d.strftime("%Y-%m-%d")
                else:
                    date_val = int(d.timestamp())
                chart_data.append({"date": date_val,
                    "open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "volume": int(r["volume"])})
            stock_name = _m._get_stock_name(symbol, market)
            # 顶层的 total_return / sharpe_ratio / max_drawdown / win_rate /
            # num_trades 是**老字段名的残留**: 正常路径根本不产出它们, 前端一律
            # 读 summary.*。留着只会让人以为有这条路径。indicator_formulas 这里
            # 是列表而正常路径是 dict, 类型也对不上 —— 一并对齐。
            result = {"strategy": strategy_name, "chart_data": chart_data,
                      "equity_curve": [], "trades": [],
                      "summary": {"capital": 100000, "final_equity": 100000, "total_return": 0, "bh_return": 0, "max_drawdown": 0, "sharpe": 0, "win_rate": 0, "profit_factor": 0, "total_trades": 0, "win_trades": 0, "lose_trades": 0, "avg_win": 0, "avg_loss": 0},
                      "chart_indicators": {"overlay": [], "subchart": []},
                      "indicator_formulas": {}, "signal_rules": {},
                      "symbol": symbol.upper(), "market": market.upper(),
                      "period": period, "interval": interval,
                      "stock_name": stock_name, "data_len": len(df),
                      "warning": f"数据不足以计算策略指标（{len(df)}条），仅显示K线"}
            _m.cache_set(cache_key, result)
            return result
        result = _m._clean(result)
        result = _prepend_history(result, symbol, market, interval, period, chart_period, df)
        stock_name = _m._get_stock_name(symbol, market)
        result.update({"symbol": symbol.upper(), "market": market.upper(),
                       "period": period, "interval": interval,
                       "chart_period": chart_period or period,
                       "stock_name": stock_name, "data_len": len(df)})
        _m.cache_set(cache_key, result)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"回测失败: {str(e)}")


@router.post("/compare/{market}/{symbol}")
def compare_strategies(market: str, symbol: str, body: dict = None):
    """多策略对比，按夏普比率排序"""
    import pickle as _pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed as _as_completed
    from backtest import get_ohlcv
    from backtest import run_strategy_backtest

    if body is None:
        body = {}
    period = body.get("period", "2y")
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    strategy_names = body.get("strategies") or _m._get_active_strategies()
    dynamic_position = body.get("dynamic_position", False)
    allow_short = body.get("allow_short", False)
    use_universal = body.get("use_universal", False)
    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="多策略对比", scope=f"{symbol.upper()} · {len(strategy_names)} 策略 · {period}",
                  steps=[{"key": "dl", "label": "拉取行情"}, {"key": "bt", "label": "逐策略回测"}])
    _m.prog_step(task_id, "dl", total=1, unit=" 只", workers=1, running=[symbol.upper()])
    # 实时口径: 含当日 (可能未走完) 的 bar, 并把本地行情文件的最大陈旧时间压到 max_age_sec.
    # 默认 False = 老口径 (日线截到昨天, 当日内结果稳定), 不影响其它调用方.
    include_today = bool(body.get("include_today", False))
    try:
        max_age_sec = int(body.get("max_age_sec")) if body.get("max_age_sec") is not None else None
    except (TypeError, ValueError):
        max_age_sec = None

    # 实时模式先取数 (会按 max_age_sec 触发增量更新), 再用"真正用到的最后一根 bar"
    # 当缓存键 — 否则缓存键里的 mtime 是刷新前的, 会把旧结果直接返回、永远不刷新.
    df_rt = None
    if include_today:
        try:
            df_rt = get_ohlcv(symbol, market, interval=interval, period=period,
                              include_today=True, max_age_sec=max_age_sec)
        except Exception as e:
            _m.prog_end(task_id, "error", f"获取数据失败: {str(e)[:120]}")
            raise HTTPException(500, f"获取数据失败: {str(e)}")
    _cmp_mtime = _m._data_mtime_sig([symbol], market, interval)
    _cmp_opt_fp = _m.OPTIMIZED_DIR / f"{symbol.upper()}.json"
    _cmp_opt_mtime = str(int(_cmp_opt_fp.stat().st_mtime) // 60) if _cmp_opt_fp.exists() else "0"
    _cmp_cust_fp = _m.CUSTOM_PARAMS_DIR / f"{symbol.upper()}.json"
    _cmp_cust_mtime = str(int(_cmp_cust_fp.stat().st_mtime) // 60) if _cmp_cust_fp.exists() else "0"
    _univ_flag = "1" if use_universal else "0"
    # 实时模式: 最后一根 bar 的时间+收盘价进 key, 价格一动就重算; TTL 也压到 60s
    _rt_tag = "0"
    _cmp_ttl = 86400 * 7
    if include_today:
        _cmp_ttl = 60
        if df_rt is not None and len(df_rt):
            try:
                _rt_tag = f"{df_rt.index[-1].isoformat()}@{round(float(df_rt['close'].iloc[-1]), 4)}"
            except Exception:
                _rt_tag = "rt"
        else:
            _rt_tag = "rt"
    cmp_cache_key = f"compare:{market}:{symbol}:{interval}:{period}:{dynamic_position}:{allow_short}:{_cmp_mtime}:{_cmp_opt_mtime}:{_cmp_cust_mtime}:{_univ_flag}:{_rt_tag}:{_m.stable_hash(sorted(strategy_names))}"
    cached = _m.cache_get(cmp_cache_key, ttl=_cmp_ttl)
    if cached is not None:
        _m.prog_end(task_id, "done", "命中缓存")
        return cached

    try:
        df = df_rt if df_rt is not None else get_ohlcv(symbol, market, interval=interval, period=period)
        if len(df) < 1:
            raise HTTPException(400, f"数据不足，只有 {len(df)} 条K线")
    except HTTPException:
        raise
    except Exception as e:
        _m.prog_end(task_id, "error", f"获取数据失败: {str(e)[:120]}")
        raise HTTPException(500, f"获取数据失败: {str(e)}")

    # 只 pickle 一次, 几十个任务共用这一份字节 (比每个任务各自序列化一遍 DataFrame 便宜)
    df_bytes = _pickle.dumps(df)

    custom = _m._load_custom_params_interval(symbol.upper(), interval=interval) if not use_universal else {}

    # ── 两轮回测合并成一个进程池 ────────────────────────────────────────────
    # 原来这里是两个串行 for 循环, 一个接一个:
    #   ① 排行榜那一轮 (选中的策略, 个股标准参数)
    #   ② 算「升级前后差多少」那一轮 (该股所有有个股参数的策略, 改用通用标准重算)
    # 实测 NBIS 22 策略时 ①22 次 + ②27 次 = 49 次回测, 串行 23 秒。
    # 这是 2026-03-19 那次「全面并行化」漏掉的一处 (这段代码来自首次提交)。
    # 现在合成一个任务表丢进同一个池: 池只起一次, 两轮互相填满 worker 的空档,
    # 进度也变成一条连续的 0→49 而不是跑到 22/22 就钉在 100% 干等。
    #
    # 参数一律在主进程解析好再传进去 —— worker 不碰 JSON 文件, 免得抢锁。
    _tasks = []
    for sname in strategy_names:
        if not use_universal:
            dual = _m._get_dual_signal_params(symbol.upper(), sname, interval=interval)
            if dual["has_dual"]:
                _tasks.append(("main", sname, df_bytes, dual["buy_params"], dual["sell_params"],
                               None, dynamic_position, allow_short))
                continue
            po = _m._get_standard_strategy_params(symbol.upper(), sname, interval=interval)
        else:
            po = _m._get_universal_strategy_params(sname, interval=interval)
        _tasks.append(("main", sname, df_bytes, None, None, po, dynamic_position, allow_short))

    for sname in custom:
        po = _m._get_universal_strategy_params(sname, interval=interval)
        _tasks.append(("orig", sname, df_bytes, None, None, po, dynamic_position, allow_short))

    n_main = len(strategy_names)
    n_workers = max(1, min(_m.N_WORKERS, len(_tasks)))
    _m.prog_step(task_id, "bt")
    _tr = _m.ProgTracker(
        task_id,
        [f"{sn}" if kd == "main" else f"{sn} (通用标准对照)" for kd, sn, *_ in _tasks],
        n_workers, unit=" 次回测",
    )

    _main_map, promoted_delta = {}, {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as _pool:
        _futs = {_pool.submit(_m._compare_rank_bt_one, t): t for t in _tasks}
        for _f in _as_completed(_futs):
            _m.prog_check_cancel(task_id, _pool, _futs)   # 点了停止就撤掉还没派出去的分片
            kd, sn, r, err = _f.result()
            _tr.done_one(sn if kd == "main" else f"{sn} (通用标准对照)")
            if kd == "main":
                if r is None:
                    _m.logger.warning(f"Strategy {sn} failed for {symbol} ({len(df)} bars): {err}")
                    _main_map[sn] = {
                        "strategy": sn, "error": err,
                        "summary": {"capital": 100000, "final_equity": 100000,
                                    "total_return": 0, "bh_return": 0,
                                    "max_drawdown": 0, "sharpe": 0, "win_rate": 0,
                                    "profit_factor": 0, "total_trades": 0,
                                    "win_trades": 0, "lose_trades": 0,
                                    "avg_win": 0, "avg_loss": 0, "score": 0},
                        "warning": f"数据不足({len(df)}条)，该策略无法计算"}
                else:
                    if not use_universal and sn in custom and custom[sn].get("params"):
                        r["_promoted"] = True
                    _main_map[sn] = r
            elif r is not None:
                so = r["summary"]
                promoted_delta[sn] = {
                    "orig_score": _m.composite_score(so["total_return"], so["sharpe"], so["max_drawdown"],
                                                     so["win_rate"], so.get("profit_factor", 0)),
                    "orig_return": so["total_return"],
                    "orig_sharpe": so["sharpe"],
                }

    # 保持原来的顺序 (按 strategy_names), 免得下游依赖入参次序的地方出意外
    results = [_main_map[sn] for sn in strategy_names if sn in _main_map]

    # 回测失败的策略在上面被**补了一份全 0 的 summary**(为了下游不 KeyError),
    # 所以 `"summary" not in r` 永远是 False —— errors 恒为空数组, 而那些
    # 全 0 的行会混进排名表, 看起来跟"真的不赚钱"一模一样。改用 error 标记分流。
    valid = [r for r in results if "summary" in r and not r.get("error")]
    errors = [{"strategy": r.get("strategy"),
               "error": r.get("error"),
               "warning": r.get("warning")}
              for r in results if r.get("error") or "summary" not in r]
    for r in valid:
        s = r["summary"]
        s["score"] = _m.composite_score(
            s["total_return"], s["sharpe"], s["max_drawdown"],
            s["win_rate"], s.get("profit_factor", 0)
        )
    valid.sort(key=lambda r: r["summary"]["score"], reverse=True)

    ranking = []
    for i, r in enumerate(valid):
        entry = {
            "rank": i + 1,
            "strategy": r["strategy"],
            "description": r.get("strategy_desc", ""),
            "score": r["summary"]["score"],
            "sharpe": r["summary"]["sharpe"],
            "total_return": r["summary"]["total_return"],
            "max_drawdown": r["summary"]["max_drawdown"],
            "win_rate": r["summary"]["win_rate"],
            "total_trades": r["summary"]["total_trades"],
            "profit_factor": r["summary"].get("profit_factor", 0),
        }
        if r.get("_promoted"):
            entry["promoted"] = True
            pd_ = promoted_delta.get(r["strategy"])
            if pd_:
                entry["delta_return"] = round(entry["total_return"] - pd_["orig_return"], 2)
                entry["delta_score"] = round(entry["score"] - pd_["orig_score"], 4)
                entry["orig_return"] = pd_["orig_return"]
                entry["orig_score"] = pd_["orig_score"]
        ranking.append(entry)

    best = valid[0] if valid else None
    result = {
        "symbol": symbol.upper(),
        "market": market.upper(),
        "period": period,
        "interval": interval,
        "data_len": len(df),
        "data_start": (df.index[0].isoformat() if len(df) else None),
        "data_end": (df.index[-1].isoformat() if len(df) else None),
        "data_days": _m._df_span_days(df),
        "include_today": include_today,
        "ranking": ranking,
        "errors": errors,
        "best_strategy": best,
    }
    _m.cache_set(cmp_cache_key, result)
    _m.prog_end(task_id, "done", f"{len(results)} 个策略回测完成")
    return result


@router.get("/backtest/params/default")
def get_default_params():
    """获取默认参数（原有策略）"""
    return _m.DEFAULT_PARAMS


@router.post("/portfolio/backtest")
def portfolio_backtest(body: dict):
    """多股票组合回测：等权/优化权重分配"""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor as _ProcPool

    market = body.get("market", "US")
    symbols = body.get("symbols", ["AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "AMD", "TSLA"])
    strategy_name = body.get("strategy", "rsi_macd_trend")
    period = body.get("period", "5y")
    dynamic_pos = body.get("dynamic_position", False)
    weight_mode = body.get("weights", "equal")

    _pf_mtime = _m._data_mtime_sig(symbols, market, "1d")
    pf_cache_key = f"portfolio:{market}:{strategy_name}:{period}:{dynamic_pos}:{weight_mode}:{_pf_mtime}:{_m.stable_hash(sorted(symbols))}"
    cached = _m.cache_get(pf_cache_key, ttl=86400 * 7)
    if cached is not None:
        return cached

    bt_args = [(sym, market, strategy_name, period, dynamic_pos) for sym in symbols]
    n_workers = min(_m.N_WORKERS, len(symbols))
    stock_results = []
    with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        for r in pool.map(_m._portfolio_bt_one, bt_args):
            stock_results.append(_m._clean(r))

    valid = [r for r in stock_results if "summary" in r]
    if not valid:
        raise HTTPException(400, "没有有效回测结果")

    n = len(valid)
    if isinstance(weight_mode, list):
        weights = weight_mode[:n]
    elif weight_mode == "sharpe_weighted":
        sharpes = [max(r["summary"]["sharpe"], 0.01) for r in valid]
        total = sum(sharpes)
        weights = [s / total for s in sharpes]
    else:
        weights = [1.0 / n] * n

    eq_map = {}
    for w, r in zip(weights, valid):
        capital = r["summary"]["capital"]
        for pt in r.get("equity_curve", []):
            d = pt["date"]
            norm_eq = pt["equity"] / capital
            eq_map[d] = eq_map.get(d, 0) + w * norm_eq

    combined_equity = [{"date": d, "equity": round(v * 100000, 2)} for d, v in sorted(eq_map.items())]

    portfolio_return = sum(w * r["summary"]["total_return"] for w, r in zip(weights, valid))
    portfolio_wr = sum(w * r["summary"]["win_rate"] for w, r in zip(weights, valid))
    if len(combined_equity) > 2:
        eq_vals = np.array([pt["equity"] for pt in combined_equity], dtype=float)
        daily_rets = np.diff(eq_vals) / eq_vals[:-1]
        std = np.std(daily_rets)
        portfolio_sharpe = float((np.mean(daily_rets) / std * np.sqrt(252)) if std > 0 else 0)
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak * 100
        portfolio_dd = float(np.min(dd))
    else:
        portfolio_sharpe = sum(w * r["summary"]["sharpe"] for w, r in zip(weights, valid))
        portfolio_dd = sum(w * r["summary"]["max_drawdown"] for w, r in zip(weights, valid))

    allocation = [{"symbol": r["symbol"], "weight": round(w * 100, 1),
                    "sharpe": r["summary"]["sharpe"],
                    "return": r["summary"]["total_return"],
                    "max_dd": r["summary"]["max_drawdown"]}
                   for w, r in zip(weights, valid)]

    result = {
        "strategy": strategy_name,
        "period": period,
        "dynamic_position": dynamic_pos,
        "weight_mode": weight_mode if isinstance(weight_mode, str) else "custom",
        "portfolio": {
            "total_return": round(portfolio_return, 2),
            "sharpe": round(portfolio_sharpe, 2),
            "max_drawdown": round(portfolio_dd, 2),
            "win_rate": round(portfolio_wr, 2),
            "num_stocks": n,
        },
        "allocation": allocation,
        "equity_curve": combined_equity[-200:],
    }
    _m.cache_set(pf_cache_key, result)
    return result


@router.post("/portfolio/combo-backtest")
def portfolio_combo_backtest(body: dict):
    """多策略组合回测：对股票池中每只股票，用多策略共享信号（OR逻辑）跑回测"""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor as _ProcPool, as_completed as _as_completed

    market = body.get("market", "US")
    pool_name = body.get("pool", "")
    direct_symbols = body.get("symbols", [])
    strategies = body.get("strategies", [])
    period = body.get("period", "1y")
    interval = body.get("interval", "1d")

    if not strategies:
        raise HTTPException(400, "请选择至少一个策略")

    # Support both pool name and direct symbols list
    if direct_symbols:
        symbols = [str(s).upper() for s in direct_symbols if str(s).strip()]
        pool_name = pool_name or ", ".join(symbols)
    elif pool_name:
        pools = _m._load_pools()
        pool = pools.get(pool_name)
        if not pool:
            raise HTTPException(400, f"股票池 '{pool_name}' 不存在")
        symbols = [str(s).upper() for s in pool.get("symbols", []) if str(s).strip()]
    else:
        raise HTTPException(400, "请选择股票池或提供股票列表")

    if not symbols:
        raise HTTPException(400, f"股票池 '{pool_name}' 为空")

    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="组合回测",
                  scope=f"{pool_name or f'{len(symbols)} 只'} · {len(strategies)} 策略 · {period}",
                  steps=[{"key": "bt", "label": "逐股并行回测"},
                         {"key": "agg", "label": "组合汇总"}])

    bt_args = [(sym, market, strategies, period, interval) for sym in symbols]
    n_workers = min(_m.N_WORKERS, max(1, len(symbols)))
    # 原来是 executor.map, 连完成了几个都拿不到; 改 submit+as_completed 后
    # 按下标回填, 结果顺序跟 map 一致, 但多了进度
    stock_results = [None] * len(bt_args)
    _m.prog_step(task_id, "bt")
    _tr = _m.ProgTracker(task_id, list(symbols), n_workers, unit=" 只")
    with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as executor:
        _futs = {executor.submit(_m._combo_bt_one, a): i for i, a in enumerate(bt_args)}
        for _f in _as_completed(_futs):
            _m.prog_check_cancel(task_id, executor, _futs)
            _i = _futs[_f]
            stock_results[_i] = _m._clean(_f.result())
            _tr.done_one(symbols[_i])
    _m.prog_step(task_id, "agg")

    valid = [r for r in stock_results if r.get("summary")]
    if not valid:
        _m.prog_end(task_id, "error", "没有有效回测结果")
        raise HTTPException(400, "没有有效回测结果")

    # Aggregate portfolio metrics
    n = len(valid)
    total_capital = n * 100000
    total_final = sum(r["summary"]["final_equity"] for r in valid)
    total_return = (total_final - total_capital) / total_capital * 100

    # Combined equity curve (sum of all stocks' equity)
    eq_map = {}
    for r in valid:
        for pt in r.get("equity_curve", []):
            d = pt["date"]
            eq_map[d] = eq_map.get(d, 0) + pt["equity"]
    combined_equity = [{"date": d, "equity": round(v, 2)} for d, v in sorted(eq_map.items())]

    # Portfolio Sharpe & drawdown from combined curve
    portfolio_sharpe = 0.0
    portfolio_dd = 0.0
    if len(combined_equity) > 2:
        eq_vals = np.array([pt["equity"] for pt in combined_equity], dtype=float)
        daily_rets = np.diff(eq_vals) / eq_vals[:-1]
        std = np.std(daily_rets)
        portfolio_sharpe = float((np.mean(daily_rets) / std * np.sqrt(252)) if std > 0 else 0)
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak * 100
        portfolio_dd = float(np.min(dd))

    total_trades = sum(r["summary"].get("trade_count", 0) for r in valid)
    wins = sum(r["summary"].get("wins", 0) for r in valid)
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

    # Sort by return descending
    valid.sort(key=lambda r: r["summary"].get("total_return", 0), reverse=True)

    # 列表响应只要 summary —— 点某只股票时前端会另外调 /portfolio/combo-stock-detail
    # 把明细拿回来。以前这里只 pop 了 chart_data, 但 equity_curve / trades /
    # chart_indicators / indicator_formulas / signal_rules 全都原样留着:
    # 实测 {date,equity} 44B/条 × 2y 日线 500 根 ≈ 21.5KB/只, 95 只池 ≈ 2MB 纯废载荷
    # (trades 另计)。对照普通回测是 equity_curve[-200:] / trades[-50:] 截过的。
    for r in valid:
        for _k in ("chart_data", "equity_curve", "trades", "chart_indicators",
                   "indicator_formulas", "signal_rules", "indicators"):
            r.pop(_k, None)

    _m.prog_end(task_id, "done", f"{len(valid)}/{len(symbols)} 只有效")
    return {
        "portfolio": {
            "total_return": round(total_return, 2),
            "sharpe": round(portfolio_sharpe, 2),
            "max_drawdown": round(portfolio_dd, 2),
            "win_rate": win_rate,
            "total_trades": total_trades,
            "num_stocks": n,
            "total_capital": total_capital,
            "final_equity": round(total_final, 2),
        },
        "per_stock": valid,
        "equity_curve": combined_equity[-500:],
        "strategies_used": strategies,
        "pool": pool_name,
        "period": period,
    }


@router.post("/portfolio/combo-stock-detail")
def portfolio_combo_stock_detail(body: dict):
    """获取单只股票的组合回测详情（K线+信号标记+成交记录）"""
    market = body.get("market", "US")
    symbol = body.get("symbol", "")
    strategies = body.get("strategies", [])
    period = body.get("period", "1y")
    interval = body.get("interval", "1d")

    if not symbol or not strategies:
        raise HTTPException(400, "缺少参数")

    result = _m._combo_bt_one((symbol, market, strategies, period, interval))
    return _m._clean(result)


@router.post("/portfolio/combo-optimize")
def portfolio_combo_optimize(body: dict):
    """组合优化：在基础策略上逐一叠加每个候选策略，找出最佳组合"""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor as _ProcPool, as_completed as _as_completed

    market = body.get("market", "US")
    pool_name = body.get("pool", "")
    direct_symbols = body.get("symbols", [])
    base_strategies = body.get("base_strategies", [])
    all_candidates = body.get("all_candidates", [])
    period = body.get("period", "1y")
    interval = body.get("interval", "1d")

    if not base_strategies:
        raise HTTPException(400, "请选择基础策略")
    if not all_candidates:
        raise HTTPException(400, "没有候选策略")

    # Resolve symbols
    if direct_symbols:
        symbols = [str(s).upper() for s in direct_symbols if str(s).strip()]
    elif pool_name:
        pools = _m._load_pools()
        pool = pools.get(pool_name)
        if not pool:
            raise HTTPException(400, f"股票池 '{pool_name}' 不存在")
        symbols = [str(s).upper() for s in pool.get("symbols", []) if str(s).strip()]
    else:
        raise HTTPException(400, "请选择股票池或提供股票")
    if not symbols:
        raise HTTPException(400, "股票列表为空")

    # Filter candidates: exclude those already in base
    base_set = set(base_strategies)
    candidates = [s for s in all_candidates if s not in base_set]

    # Build tasks: baseline + one per candidate
    tasks = [("__baseline__", base_strategies)]
    for c in candidates:
        tasks.append((c, base_strategies + [c]))

    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="组合优化",
                  scope=f"基础 {len(base_strategies)} 策略 + {len(candidates)} 个候选 · {len(symbols)} 只",
                  steps=[{"key": "opt", "label": "基准 + 逐候选并行回测"}])

    # 并行执行：每个任务 = 1个候选策略 × 所有股票（使用个股标准参数）
    # 用ProcessPoolExecutor，max_workers自适应服务器CPU数（N_WORKERS = cpu_count - 2）
    n_workers = min(_m.N_WORKERS, max(1, len(tasks)))
    opt_args = [(label, strats, symbols, market, period, interval) for label, strats in tasks]
    _labels = ['基准组合' if lb == '__baseline__' else lb for lb, _ in tasks]
    rows = [None] * len(opt_args)
    _m.prog_step(task_id, "opt")
    _tr = _m.ProgTracker(task_id, list(_labels), n_workers, unit=" 组")
    with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        _futs = {pool.submit(_m._combo_opt_worker, a): i for i, a in enumerate(opt_args)}
        for _f in _as_completed(_futs):
            _m.prog_check_cancel(task_id, pool, _futs)
            _i = _futs[_f]
            rows[_i] = _f.result()
            _tr.done_one(_labels[_i])

    baseline = rows[0] if rows else {}
    _m.prog_end(task_id, "done", f"{len(candidates)} 个候选评估完成")
    return {
        "baseline": baseline,
        "results": rows,
        "base_strategies": base_strategies,
        "candidates_tested": len(candidates),
    }


@router.post("/portfolio/combo-pairs")
def portfolio_combo_pairs(body: dict):
    """两两组合：测试所有候选策略的两两组合"""
    import numpy as np
    from itertools import combinations
    from concurrent.futures import ProcessPoolExecutor as _ProcPool, as_completed as _as_completed

    market = body.get("market", "US")
    pool_name = body.get("pool", "")
    direct_symbols = body.get("symbols", [])
    all_candidates = body.get("all_candidates", [])
    period = body.get("period", "1y")
    interval = body.get("interval", "1d")

    if not all_candidates or len(all_candidates) < 2:
        raise HTTPException(400, "至少需要2个候选策略")

    # Resolve symbols
    if direct_symbols:
        symbols = [str(s).upper() for s in direct_symbols if str(s).strip()]
    elif pool_name:
        pools = _m._load_pools()
        pool = pools.get(pool_name)
        if not pool:
            raise HTTPException(400, f"股票池 '{pool_name}' 不存在")
        symbols = [str(s).upper() for s in pool.get("symbols", []) if str(s).strip()]
    else:
        raise HTTPException(400, "请选择股票池或提供股票")
    if not symbols:
        raise HTTPException(400, "股票列表为空")

    pairs = list(combinations(all_candidates, 2))
    # 按本机核数并行执行，无硬限制；超大规模给出预估时间提示
    # C(n,2)对，每对跑全部股票；N_WORKERS=cpu_count-2 进程并发
    n_pairs = len(pairs)
    n_sym = len(symbols)
    # 粗估：每股≈0.3s（本地parquet缓存），32CPU并行
    est_sec = max(1, n_pairs // max(1, _m.N_WORKERS)) * n_sym * 0.3
    import math
    est_min = math.ceil(est_sec / 60)

    # 并行执行：每个任务 = 1对策略 × 所有股票（使用个股标准参数）
    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="两两组合",
                  scope=f"{n_pairs} 对 · {n_sym} 只 · {period}",
                  steps=[{"key": "pairs", "label": "逐对并行回测"}])

    pair_args = [(list(p), symbols, market, period, interval) for p in pairs]
    n_workers = min(_m.N_WORKERS, max(1, n_pairs))
    _labels = [' + '.join(list(p)) for p in pairs]
    rows = [None] * len(pair_args)
    _m.prog_step(task_id, "pairs")
    _tr = _m.ProgTracker(task_id, list(_labels), n_workers, unit=" 对")
    with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        _futs = {pool.submit(_m._combo_pair_worker, a): i for i, a in enumerate(pair_args)}
        for _f in _as_completed(_futs):
            _m.prog_check_cancel(task_id, pool, _futs)
            _i = _futs[_f]
            rows[_i] = _f.result()
            _tr.done_one(_labels[_i])

    _m.prog_end(task_id, "done", f"{n_pairs} 对回测完成")
    return {
        "results": rows,
        "pairs_tested": n_pairs,
        "candidates": all_candidates,
        "est_min": est_min,
    }


@router.post("/walkforward/{market}/{symbol}")
def walk_forward(market: str, symbol: str, body: dict = None):
    """Walk-Forward 验证：滚动窗口训练+样本外测试"""
    import numpy as np
    from backtest import get_ohlcv
    from backtest import run_strategy_backtest
    from optimizer import optimize_strategy
    from concurrent.futures import ProcessPoolExecutor as _ProcPool
    from strategies import get_strategy

    if body is None:
        body = {}
    strategy_name = body.get("strategy", "rsi_macd_trend")
    train_years = body.get("train_years", 3)
    test_months = body.get("test_months", 6)
    dynamic_pos = body.get("dynamic_position", False)
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    param_source = body.get("param_source", "standard")
    if param_source == "default":
        param_source = "standard"
    if param_source == "optimized":
        param_source = "best"

    _wf_data_mtime = _m._data_mtime_sig([symbol], market, interval)
    _wf_opt_mtime = "0"
    _opt_fp = _m.OPTIMIZED_DIR / f"{symbol.upper()}.json"
    if _opt_fp.exists():
        _wf_opt_mtime = str(int(_opt_fp.stat().st_mtime) // 60)
    wf_cache_key = f"wf:{market}:{symbol}:{strategy_name}:{interval}:{train_years}:{test_months}:{dynamic_pos}:{param_source}:{_wf_data_mtime}:{_wf_opt_mtime}"
    cached = _m.cache_get(wf_cache_key, ttl=86400 * 7)
    if cached is not None:
        return cached

    strategy = get_strategy(strategy_name)
    grid = _m.OPTIMIZE_GRIDS.get(strategy_name)

    saved_params = None
    if param_source == "standard":
        saved_params = _m._get_universal_strategy_params(strategy_name, interval=interval)
    elif param_source == "custom":
        saved_params = _m._get_standard_strategy_params(symbol.upper(), strategy_name, interval=interval)
    elif param_source == "best":
        opt_data = _m._load_optimized_interval(symbol.upper(), interval)
        if opt_data and "strategies" in opt_data:
            strat_data = opt_data["strategies"].get(strategy_name)
            if strat_data and "params" in strat_data:
                saved_params = strat_data["params"]
        if not saved_params:
            raise HTTPException(400, f"未找到 {symbol.upper()} 的 {strategy_name} 最优优化参数，请先运行优化")
    elif param_source in ("bayesian", "grid", "coarse_fine"):
        opt_data = _m._load_optimized_interval(symbol.upper(), interval)
        if opt_data:
            method_best = opt_data.get("method_best", {})
            method_data = method_best.get(strategy_name, {}).get(param_source)
            if method_data and method_data.get("params"):
                saved_params = method_data["params"]
        if not saved_params:
            raise HTTPException(400, f"未找到 {symbol.upper()} 的 {strategy_name} {param_source}参数")

    try:
        df = get_ohlcv(symbol, market, interval=interval, period="max")
    except Exception:
        df = get_ohlcv(symbol, market, interval=interval, period="10y")

    wf_plan = _m._build_wf_plan(df, train_years=train_years, test_months=test_months)
    if not wf_plan.get("ok"):
        raise HTTPException(400, wf_plan.get("reason", "数据不足，无法执行walk-forward"))
    total_bars = wf_plan["total_bars"]
    train_days = wf_plan["train_days"]
    test_days = wf_plan["test_days"]
    auto_adjusted = wf_plan.get("auto_adjusted", False)

    results = []
    retrain_method = None
    if param_source.startswith("retrain_"):
        retrain_method = param_source.replace("retrain_", "")

    n_trials_wf = 50
    windows = wf_plan["windows"]

    if retrain_method:
        # 每窗口重新优化 —— 全函数最贵的一条路 (每个窗口一次贝叶斯/穷举)。
        # 之前这里是串行 for 循环, 而下面的 else 分支早就 pool.map 并行了,
        # 2026-03-19 那次「全面并行化」只改了一半。窗口之间没有任何依赖,
        # 所以直接照搬 else 分支的做法, 换成 retrain 版 worker。
        wf_tasks = [(tr_df, te_df, strategy_name, retrain_method, grid,
                     dynamic_pos, n_trials_wf, strategy.default_params)
                    for tr_df, te_df in windows]
        n_workers = max(1, min(_m.N_WORKERS, len(wf_tasks)))
        with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
            results = list(pool.map(_m._wf_one_window_retrain, wf_tasks))
    else:
        wf_params = saved_params
        wf_tasks = [(tr_df, te_df, strategy_name, wf_params, dynamic_pos)
                     for tr_df, te_df in windows]
        n_workers = min(_m.N_WORKERS, len(wf_tasks))
        with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
            results = list(pool.map(_m._wf_one_window, wf_tasks))

    if not results:
        raise HTTPException(400, "数据不够执行walk-forward")

    wf_summary = _m._summarize_wf_results(results, wf_plan)

    param_source_labels = {
        "standard": "通用标准", "custom": "个股标准", "best": "最优参数",
        "bayesian": "贝叶斯", "grid": "穷举", "coarse_fine": "粗搜精搜",
        "retrain_bayesian": "每窗口贝叶斯优化", "retrain_grid": "每窗口穷举优化",
        "retrain_coarse_fine": "每窗口粗搜+精搜",
    }
    wf_save_key_map = {
        "standard": "standard", "custom": "custom",
        "best": "best", "bayesian": "bayesian", "grid": "grid", "coarse_fine": "coarse_fine",
        "retrain_bayesian": "bayesian", "retrain_grid": "grid", "retrain_coarse_fine": "coarse_fine",
    }
    wf_save_key = wf_save_key_map.get(param_source, param_source)

    resp = {
        "symbol": symbol.upper(),
        "market": market.upper(),
        "strategy": strategy_name,
        "param_source": param_source,
        "param_source_label": param_source_labels.get(param_source, param_source),
        "dynamic_position": dynamic_pos,
        "total_bars": total_bars,
        "train_days": train_days,
        "test_days": test_days,
        "num_windows": len(results),
        # 让调用方看得见"只跑了最近 N 个", 别以为验的是全部历史
        "windows_available": wf_plan.get("windows_available", len(results)),
        "windows_trimmed": wf_plan.get("windows_trimmed", 0),
        "max_windows": wf_plan.get("max_windows", 0),
        "window_mode": wf_plan["mode"],
        "reliability": wf_plan["reliability"],
        "summary": {
            "avg_test_sharpe": wf_summary["avg_test_sharpe"],
            "avg_test_return": wf_summary["avg_test_return"],
            "avg_test_max_dd": wf_summary["avg_test_max_dd"],
            "consistency": wf_summary["consistency"],
            # verdict/promoted/reliability 以前只落盘不外发, 于是前端只能
            # 按 consistency 单因素自己重算, 跟落盘的结论打架 (robust 还要求
            # avg_sharpe>0 且 reliability 非 low)。
            "verdict": wf_summary["verdict"],
            "promoted": wf_summary["promoted"],
            "reliability": wf_summary["reliability"],
            "window_mode": wf_summary.get("window_mode"),
        },
        "windows": results,
    }
    if auto_adjusted:
        resp["auto_adjusted"] = True
    if wf_plan.get("note"):
        resp["note"] = wf_plan["note"]
    else:
        resp["train_years"] = train_years
        resp["test_months"] = test_months

    wf_entry = {
        strategy_name: {
            "verdict": wf_summary["verdict"],
            "consistency": wf_summary["consistency"],
            "avg_test_sharpe": wf_summary["avg_test_sharpe"],
            "avg_test_return": wf_summary["avg_test_return"],
            "avg_test_max_dd": wf_summary["avg_test_max_dd"],
            "num_windows": wf_summary["num_windows"],
            "promoted": wf_summary["promoted"],
            "window_mode": wf_summary["window_mode"],
            "reliability": wf_summary["reliability"],
            "note": wf_summary.get("note"),
        }
    }
    opt_data = _m._load_optimized(symbol.upper()) or {}
    existing_wf = opt_data.get("wf_validation", {})
    if existing_wf and any(isinstance(v, dict) and "verdict" in v for v in existing_wf.values()):
        existing_wf = {"best": existing_wf}
    layer = existing_wf.get(wf_save_key, {})
    layer[strategy_name] = wf_entry[strategy_name]
    existing_wf[wf_save_key] = layer
    opt_data["wf_validation"] = existing_wf
    _m._save_optimized(symbol.upper(), opt_data)

    resp = _m._clean(resp)
    resp["wf_save_key"] = wf_save_key
    _m.cache_set(wf_cache_key, resp)
    return resp


@router.post("/applicability")
def strategy_applicability(body: dict = None):
    """策略适用性分析：在股票池内所有股票上跑全部策略"""
    import numpy as np
    from backtest import get_ohlcv
    from strategies import get_strategy
    from concurrent.futures import ProcessPoolExecutor as _ProcPool, ThreadPoolExecutor as _ThreadPool, as_completed as _as_completed

    if body is None:
        body = {}
    pool_name = body.get("pool", "US Tech")
    period = body.get("period", "2y")
    dynamic_pos = body.get("dynamic_position", False)
    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="策略适用性", scope=f"{pool_name} · {period}",
                  steps=[{"key": "fetch", "label": "拉取行情"},
                         {"key": "bt", "label": "逐股×逐策略回测"},
                         {"key": "rank", "label": "汇总排名"}])

    pools = _m._load_pools()
    if pool_name not in pools:
        _m.prog_end(task_id, "error", f"股票池 '{pool_name}' 不存在")
        raise HTTPException(400, f"股票池 '{pool_name}' 不存在")

    pool = pools[pool_name]
    market = pool["market"]
    symbols = pool["symbols"]

    symbols_hash = _m._hash_symbols(symbols)
    mtime_sig = _m._data_mtime_sig(symbols, market)
    cache_key = f"applicability:{pool_name}:{period}:{dynamic_pos}:{symbols_hash}:{mtime_sig}"
    cached = _m.cache_get(cache_key, ttl=86400 * 7)
    if cached is not None:
        _m.prog_end(task_id, "done", "命中缓存")
        return cached
    strategy_names = body.get("strategies") or _m._get_active_strategies()

    heatmap = {sname: {} for sname in strategy_names}
    errors = []

    datasets = {}

    def _fetch_sym(sym):
        try:
            df = get_ohlcv(sym, market, interval="1d", period=period)
            if len(df) < 10:
                return sym, None, f"数据不足({len(df)}条)"
            return sym, df, None
        except Exception as e:
            return sym, None, str(e)

    # Executor.map 严格按提交顺序 yield: 第 1 个慢, 后面全都完成了进度也一格不动
    # (第一只股票撞上 yfinance 429 重试就是几秒, 面板就"卡在拉取行情")。
    # 换 submit + as_completed, 谁先完成谁先记。
    _n_fetch = max(1, min(_m.N_WORKERS, len(symbols)))
    _m.prog_step(task_id, "fetch")
    # 这一步是线程池拉网络, 不是起进程池 —— 必须把默认那条"正在启动 N 个并行进程"关掉
    _tr = _m.ProgTracker(task_id, list(symbols), _n_fetch, unit=" 只", note="")
    with _ThreadPool(max_workers=_n_fetch) as _tp:
        _ffuts = {_tp.submit(_fetch_sym, sy): sy for sy in symbols}
        for _f in _as_completed(_ffuts):
            _m.prog_check_cancel(task_id, _tp, _ffuts)
            sym, df, err = _f.result()
            if err:
                errors.append({"symbol": sym, "error": err})
            elif df is not None:
                datasets[sym] = df
            _tr.done_one(sym)

    _app_tasks = []
    for sym, df in datasets.items():
        for sname in strategy_names:
            _app_tasks.append((sym, df, sname, dynamic_pos))

    n_workers = max(1, min(_m.N_WORKERS, len(_app_tasks)))
    _m.prog_step(task_id, "bt")
    _tr2 = _m.ProgTracker(task_id, [f"{t[0]} · {t[2]}" for t in _app_tasks], n_workers, unit=" 组")
    with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool_exec:
        _bfuts = {pool_exec.submit(_m._bt_one_applicability, t): t for t in _app_tasks}
        for _f in _as_completed(_bfuts):
            _m.prog_check_cancel(task_id, pool_exec, _bfuts)
            sym, sname, result = _f.result()
            heatmap[sname][sym] = result
            _tr2.done_one(f"{sym} · {sname}")

    _m.prog_step(task_id, "rank")

    valid_symbols = [s for s in symbols if any(s in heatmap[sn] for sn in strategy_names)]

    strategy_ranking = []
    for sname in strategy_names:
        vals = heatmap[sname]
        if not vals:
            continue
        scores = [v["score"] for v in vals.values()]
        sharpes = [v["sharpe"] for v in vals.values()]
        returns = [v["return"] for v in vals.values()]
        dds = [v["max_dd"] for v in vals.values()]
        positive = sum(1 for s in sharpes if s > 0)
        strategy_ranking.append({
            "strategy": sname,
            "description": get_strategy(sname).description,
            "avg_score": round(float(np.mean(scores)), 4),
            "avg_sharpe": round(float(np.mean(sharpes)), 2),
            "avg_return": round(float(np.mean(returns)), 2),
            "avg_max_dd": round(float(np.mean(dds)), 2),
            "positive_count": positive,
            "total_count": len(sharpes),
        })
    strategy_ranking.sort(key=lambda x: x["avg_score"], reverse=True)

    symbol_best = {}
    for sym in valid_symbols:
        best_s, best_sharpe = "", -999
        for sname in strategy_names:
            if sym in heatmap[sname]:
                s = heatmap[sname][sym]["sharpe"]
                if s > best_sharpe:
                    best_sharpe = s
                    best_s = sname
        symbol_best[sym] = best_s

    result = _m._clean({
        "pool": pool_name,
        "market": market,
        "period": period,
        "dynamic_position": dynamic_pos,
        "num_symbols": len(valid_symbols),
        "num_strategies": len(strategy_names),
        "strategy_ranking": strategy_ranking,
        "heatmap": heatmap,
        "symbols": valid_symbols,
        "symbol_best": symbol_best,
        "errors": errors,
    })
    _m.cache_set(cache_key, result)
    _m.prog_end(task_id, "done",
                f"{len(valid_symbols)}/{len(symbols)} 只 × {len(strategy_names)} 策略")
    return result


@router.post("/optimize/{market}/{symbol}")
def optimize_stock(market: str, symbol: str, body: dict = None):
    """对单只股票逐策略做参数优化（多策略并行）"""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from backtest import get_ohlcv

    if body is None:
        body = {}
    period = body.get("period", "2y")
    dynamic_pos = body.get("dynamic_position", False)
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    period = _m._optimization_period_for_interval(interval)
    strategy_filter = body.get("strategies")
    method = body.get("method", "bayesian")
    n_trials = body.get("n_trials", 200)
    signal_type = body.get("signal_type", "buy_and_sell")
    # 样本外验收要留 1/3 的数据不给优化器看 —— 日线默认的 1y(251根)切完,
    # 持出段只剩 ~84 根, 信号数常常个位数, 全卡在"验不出来"上。换长历史。
    if body.get("opt_objective") == "oos" and signal_type in ("buy", "sell"):
        period = _m._signal_oos_period_for_interval(interval)
    # 弹窗里那个「保留过拟合参数」复选框以前全仓库 0 处读取 —— 勾不勾对
    # 请求体没有任何影响。默认 True 保持原有行为(无条件保留)。
    body["keep_overfit"] = bool(body.get("keep_overfit", True))

    # 同一时刻只允许一个优化任务 (见 main.opt_slot_acquire 的注释)
    busy = _m.opt_slot_acquire("symbol", symbol.upper(), method=method, interval=interval)
    if busy:
        raise HTTPException(409, _m.opt_slot_busy_message(busy))

    try:
        return _optimize_stock_inner(
            market, symbol, body, period, dynamic_pos, interval,
            strategy_filter, method, n_trials, signal_type,
        )
    finally:
        _m.opt_slot_release(symbol.upper())


def _optimize_stock_inner(market, symbol, body, period, dynamic_pos, interval,
                          strategy_filter, method, n_trials, signal_type):
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from backtest import get_ohlcv

    # 策略清单不依赖行情, 先算出来 —— 好在拉行情之前就把总数报出去。
    # 以前第一次 opt_slot_progress 在两次 get_ohlcv + _build_wf_plan 之后,
    # 这段(冷缓存时几十秒到几分钟)前端拿到的是 total=0, 进度面板里只有转圈和
    # 计时、没有进度条本体, 看着像"这个功能没有进度条"。
    # 买点/卖点专项优化的口径。opt_objective="success" = **只优化成功率**,
    # 不看收益幅度(默认 composite 是原有的 成功率×0.6+均MFE×0.3−均MAE×0.1)。
    # judge_mode 跟买/卖点评测同一套, 这样"优化出来的成功率"和"评测里看到的"
    # 是同一个数, 不会一个先到先判一个 MFE/MAE。
    sig_opts = {
        "objective_mode": body.get("opt_objective", "composite"),
        "judge_mode": body.get("judge_mode", "first_touch"),
        "same_bar": body.get("same_bar", "adverse_first"),
        "window": int(body.get("sig_window", 10)),
        "mfe_target": float(body.get("sig_mfe_target", 5)),
        "mae_limit": float(body.get("sig_mae_limit", 10)),
        "entry_mode": body.get("entry_mode", "next_open"),
    }
    if sig_opts["objective_mode"] == "oos":
        # 样本外验收的三道门槛做成可调 —— 默认这套(持出段>=40%, 落差<=15pp)
        # 是按"宁可不改也不写进只在搜索段好看的参数"定的, 偏严。
        sig_opts.update({
            "n_segments": int(body.get("oos_segments", 6)),
            "holdout_segments": int(body.get("oos_holdout", 2)),
            "holdout_min_rate": float(body.get("oos_min_rate", 40.0)),
            "max_gap_pp": float(body.get("oos_max_gap", 15.0)),
        })

    strategy_names = strategy_filter or _m._get_active_strategies()
    _m.opt_slot_progress(stage="optimize", opt_total=len(strategy_names), opt_done=0,
                         current="拉取行情…", workers=0)

    try:
        df = get_ohlcv(symbol, market, interval=interval, period=period)
        if len(df) < 1:
            raise HTTPException(400, f"数据不足，只有 {len(df)} 条K线")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"获取数据失败: {str(e)}")

    results = {}
    method_comparison = {}
    total_combos = 0
    t_start = _time.time()

    run_all = (method == "all")
    methods_to_run = ["bayesian", "grid", "coarse_fine"] if run_all else [method]

    # WF 计划要用**完整历史**单独取一份, 不能复用上面优化用的 df ——
    # 优化窗口日线只有 1y(252根), 而 WF 标准窗口要 3年训练+6月测试(756+126根),
    # 拿 252 根去切只会降级成 anchored_micro(训练210/测试21/2个窗口),
    # 测试段 21 根常常零交易 → test_sharpe=0 → consistency=0 → 一律判 overfit。
    _wf_period = _m._wf_period_for_interval(interval)
    if _wf_period == period:
        wf_df = df
    else:
        _m.opt_slot_progress(stage="optimize", current="拉取 WF 长历史…")
        try:
            wf_df = get_ohlcv(symbol, market, interval=interval, period=_wf_period)
        except Exception as e:
            _m.logger.warning(f"[WF] {symbol} 取完整历史失败, 退回优化窗口的数据: {e}")
            wf_df = df
    wf_plan = _m._build_wf_plan(wf_df, train_years=3, test_months=6)

    n_workers = min(_m.N_WORKERS, len(strategy_names))
    n_workers = max(n_workers, 1)
    # 每个策略的 sig_opts 里带上**它当前在用的参数**当基线 —— 优化完好让前端
    # 显示"优化前 vs 优化后"。没有基线的话结果表里只有一个孤零零的成功率,
    # 判断不了这次优化到底有没有用。
    def _sig_opts_for(sname):
        if signal_type not in ("buy", "sell"):
            return sig_opts
        try:
            bp = _m._get_standard_strategy_params(symbol.upper(), sname,
                                                  interval=interval, signal_type=signal_type)
        except Exception:
            bp = None
        return {**sig_opts, "baseline_params": bp}

    tasks = [
        (df, sname, method, _m.OPTIMIZE_GRIDS.get(sname), dynamic_pos,
         n_trials, run_all, methods_to_run, signal_type, _sig_opts_for(sname))
        for sname in strategy_names
    ]

    skipped = {}          # {策略名: 没保存的原因}
    _m.logger.info(f"个股优化 {symbol}: {len(strategy_names)} 个策略并行 (workers={n_workers}, method={method})")

    # 进度: 优化阶段 (每完成一个策略上报一次, 前端轮询 /optimize/running 拿)
    _m.opt_slot_progress(stage="optimize", opt_total=len(strategy_names), opt_done=0,
                         current="", workers=n_workers)

    with ProcessPoolExecutor(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        futures = {pool.submit(_m._optimize_one_stock_strategy, t): t[1] for t in tasks}
        _opt_done = 0
        for fut in as_completed(futures):
            sname = futures[fut]
            try:
                res = fut.result()
                total_combos += res.get("combos", 0)
                if res.get("result"):
                    results[res["sname"]] = res["result"]
                else:
                    # 别再静默丢: 以前这里没有 else, 策略凭空消失 —— 进度条报
                    # "13 个完成", 存下来只有 7 个, 用户无从知道少了谁、为什么少。
                    skipped[res["sname"]] = res.get("skip_reason") or "优化未产出结果"
                if res.get("method_comparison"):
                    method_comparison[res["sname"]] = res["method_comparison"]
            except Exception as e:
                _m.logger.error(f"Optimize {sname} parallel error: {e}")
                skipped[sname] = f"{type(e).__name__}: {str(e)[:120]}"
            _opt_done += 1
            _m.opt_slot_progress(stage="optimize", opt_done=_opt_done, current=sname,
                                 combos=total_combos)

    elapsed = _time.time() - t_start

    save_data = {
        "symbol": symbol.upper(),
        "market": market.upper(),
        "period": period,
        "interval": interval,
        "data_len": int(len(df)),
        "data_start": (df.index[0].isoformat() if len(df) else None),
        "data_end": (df.index[-1].isoformat() if len(df) else None),
        "dynamic_position": dynamic_pos,
        "method": method,
        "optimized_at": datetime.utcnow().isoformat(),
        "total_combos_tested": total_combos,
        "elapsed_seconds": round(elapsed, 1),
        "strategies": results,
        # 存下来: 事后能查"这次到底请求了哪些、成了哪些、哪些没成、为什么"。
        # 以前只存 strategies(幸存者), 少了谁完全查不出来。
        "requested_strategies": list(strategy_names),
        "skipped_strategies": skipped,
    }
    if skipped:
        _m.logger.warning(
            f"个股优化 {symbol} {interval}: 请求 {len(strategy_names)} 个, "
            f"成功 {len(results)} 个, 未产出 {len(skipped)} 个 -> "
            + "; ".join(f"{k}({v})" for k, v in list(skipped.items())[:6]))
    if method_comparison:
        save_data["method_comparison"] = method_comparison
    _merged = _m._write_interval_layer(_m._load_optimized(symbol.upper()) or {}, interval, save_data, strategy_payload=False)
    # 把合并时的判断结果捎回给调用方: _write_interval_layer 改的是内部副本,
    # save_data 本身拿不到, 不回填的话前端永远看不到"哪些保留了原参数"。
    _kept = (_merged.get("intervals", {}).get(interval) or {}).get("kept_previous_strategies")
    if _kept:
        save_data["kept_previous_strategies"] = _kept
        # WF 必须验**真正存下来的那份参数**。
        # 这几个策略本次结果更差且更浅, 存的是旧参数; 而下面的 WF 是拿
        # results(本次的新参数)去跑的 —— 不换回来的话, 算出来的结论描述的是
        # 一份已经被丢弃的参数, 贴到旧参数上就是一条错的结论。
        # (合并逻辑因此会跳过它们, 结果就是这些策略永远拿不到 WF, 表里一排"—"。)
        _stored = (_merged.get("intervals", {}).get(interval) or {}).get("strategies") or {}
        for _sn in _kept:
            if _sn in _stored:
                results[_sn] = _stored[_sn]
        _m.logger.info(f"[WF] {symbol}: {len(_kept)} 个策略改用已存参数做验证 ({', '.join(_kept[:6])})")
    _m._save_optimized(symbol.upper(), _merged)

    # ── Buy/Sell 模式：保存到 signal_params，跳过 WF ──
    if signal_type in ("buy", "sell"):
        for sname, sdata in results.items():
            if sdata.get("params"):
                _m._save_signal_params(symbol.upper(), sname, signal_type,
                                       params=sdata["params"],
                                       metrics={k: v for k, v in sdata.items() if k != "params"},
                                       interval=interval)
        save_data["signal_type"] = signal_type
        # 前端要靠这个判断表里该不该出「搜索段/持出段/落差」三列 ——
        # 以前是从结果行里嗅探的, 结果全被拒收(0 行)时就嗅不到了,
        # 恰好"全拒"才是最需要把验收口径写清楚的时候。
        save_data["objective_mode"] = sig_opts.get("objective_mode")
        save_data["sig_opts"] = dict(sig_opts)
    else:
        # ── Buy&Sell 模式：先保存 buy/sell 独立参数，再走 WF ──
        for sname, sdata in results.items():
            bp = sdata.get("buy_params")
            sp = sdata.get("sell_params")
            if bp:
                _m._save_signal_params(symbol.upper(), sname, "buy", params=bp,
                                       metrics=sdata.get("buy_metrics"), interval=interval)
            if sp:
                _m._save_signal_params(symbol.upper(), sname, "sell", params=sp,
                                       metrics=sdata.get("sell_metrics"), interval=interval)
        wf_validation = {}
        if results:
            # 这里必须把 wf_total 一起报。以前只发 stage=wf + wf_done=0,
            # 前端 total 归 0 → 进度条从优化阶段的百分比**倒退成一片空白**,
            # 要等 _run_wf_validation_from_plan 内部跑到才补上。
            # 口径和那边的 len(all_tasks) 一致: 有参数的策略数 × 窗口数。
            _wf_n_win = len((wf_plan or {}).get("windows") or [])
            _wf_n_str = sum(1 for _s in results.values()
                            if isinstance(_s, dict) and _s.get("params"))
            _m.opt_slot_progress(stage="wf", current="", wf_done=0,
                                 wf_total=_wf_n_win * _wf_n_str)
            wf_validation = _m._run_wf_validation_from_plan(symbol, results, dynamic_pos, wf_plan)
            if not wf_validation:
                wf_validation = _m._run_wf_validation(symbol, market, results, dynamic_pos, interval=interval, period=period)
        # 顺便把**标准参数**也验一遍。
        # 「对比模式」里有 WF标准 / WF优化 两列, 但自动流程一直只写 best(优化参数)
        # 那一份 —— standard 这一档只有手动点 Walk-Forward 按钮、逐个策略、且把
        # 参数源切成"标准"时才会写。没人会为 13 个策略逐个手点, 所以那一整列
        # 永远是"—"。而这列恰恰是对比模式的意义所在: 优化到底有没有让它更稳,
        # 得拿"优化前也做同样的 WF"当对照才看得出来。
        wf_standard = {}
        if results:
            std_for_wf = {}
            for sname in results:
                try:
                    sp = _m._get_standard_strategy_params(symbol.upper(), sname, interval=interval)
                    if sp:
                        std_for_wf[sname] = {"params": sp}
                except Exception as e:
                    _m.logger.warning(f"[WF-std] {symbol}/{sname} 取标准参数失败: {e}")
            if std_for_wf:
                # 独立成一个 stage: 同一条 wf 进度条被重置会看成"倒退",
                # 换成新步骤前端就会把上一步打勾、这一步从头开始。
                _m.opt_slot_progress(stage="wf_std", current="标准参数对照", wf_done=0,
                                     wf_total=_wf_n_win * len(std_for_wf))
                try:
                    wf_standard = _m._run_wf_validation_from_plan(
                        symbol, std_for_wf, dynamic_pos, wf_plan,
                        slot_stage="wf_std") or {}
                except _m.TaskCancelled:
                    raise
                except Exception as e:
                    _m.logger.warning(f"[WF-std] {symbol} 标准参数 WF 失败: {e}")

        # 「保留过拟合参数」没勾: WF 判定 overfit 的策略, 优化参数不入库
        # (signal_params 里的 buy/sell 参数已经单独存过, 这里只管 best 这一份)
        if wf_validation and not body.get("keep_overfit", True):
            _dropped = [sn for sn, wf in wf_validation.items()
                        if isinstance(wf, dict) and wf.get("verdict") == "overfit"]
            for sn in _dropped:
                results.pop(sn, None)
                save_data.get("strategies", {}).pop(sn, None)
            if _dropped:
                save_data["dropped_overfit"] = _dropped
                _m.logger.info(f"[optimize] {symbol} 未勾「保留过拟合参数」, 丢弃 {len(_dropped)} 个: {', '.join(_dropped[:8])}")

        if wf_validation or wf_standard:
            existing_wf = save_data.get("wf_validation", {})
            if existing_wf and any(isinstance(v, dict) and "verdict" in v for v in existing_wf.values()):
                existing_wf = {"best": existing_wf}
            if wf_standard:
                existing_wf["standard"] = wf_standard
                _m.logger.info(f"[WF-std] {symbol} {interval}: 标准参数验了 {len(wf_standard)} 个策略")
            if wf_validation:
                existing_wf["best"] = wf_validation
            save_data["wf_validation"] = existing_wf
            _merged2 = _m._write_interval_layer(_m._load_optimized(symbol.upper()) or {}, interval, save_data, strategy_payload=False)
            _kept2 = (_merged2.get("intervals", {}).get(interval) or {}).get("kept_previous_strategies")
            if _kept2:
                save_data["kept_previous_strategies"] = _kept2
            _m._save_optimized(symbol.upper(), _merged2)
            _m._promote_to_standard(
                symbol.upper(), market.upper(), wf_validation or {}, results,
                period=period, dynamic_pos=dynamic_pos, interval=interval
            )

    # ── 保存优化运行记录 ──
    try:
        _m._save_optimization_run(
            scope="symbol", symbol=symbol.upper(),
            symbols=[symbol.upper()], signal_type=signal_type,
            method=method, interval=interval, n_trials=n_trials,
            results=results, wf_results=wf_validation if signal_type == "buy_and_sell" else None,
            total_combos=total_combos, elapsed_seconds=elapsed,
        )
    except Exception as e:
        _m.logger.warning(f"[OptHistory] single save failed: {e}")

    return _m._clean(save_data)


@router.post("/optimize/deoverfit/{market}/{symbol}")
def deoverfit_optimize(market: str, symbol: str, body: dict = None):
    """去过拟合优化: 只针对 WF 判定为「一般/过拟合」的策略, 重新找参数, 直到稳健。

    跟「⚡ 优化参数」的根本区别在**优化目标**:
      优化参数   → 目标是样本内得分(整段行情上的信号质量), WF 只是事后检验。
                   过拟合的本质就是样本内好、样本外崩 —— 用样本内当目标天然招它。
      去过拟合   → 目标直接是样本外一致性(WF 各测试段里 sharpe>0 的占比),
                   跟判定 verdict 用的是同一把尺。

    多轮: 一轮之后还不稳健, 就围着这一轮的最好参数收窄范围再来一轮
    (不收窄的话多跑几轮只是重新乱撞)。轮数上限 DEOVERFIT_MAX_ROUNDS ——
    "直到稳健"不能真的无限循环, 有的策略在这只股票上就是不行, 硬凑只会凑出
    对 WF 窗口的过拟合。用完还不稳健就如实报告, 不改动它原来的参数。

    持出集: 最近几个窗口搜索全程看不见, 最后单独验一次。搜索集和持出集的数
    都返回 —— 差得多就说明只是换了个对象过拟合, 这种不落盘。
    """
    import pickle as _pickle
    from concurrent.futures import ProcessPoolExecutor as _ProcPool, as_completed as _as_completed
    from backtest import get_ohlcv

    if body is None:
        body = {}
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    dynamic_pos = bool(body.get("dynamic_position", False))
    n_trials = int(body.get("n_trials") or _m.DEOVERFIT_TRIALS)
    max_rounds = int(body.get("max_rounds") or _m.DEOVERFIT_MAX_ROUNDS)
    holdout_n = int(body.get("holdout") or _m.DEOVERFIT_HOLDOUT)
    task_id = body.get("task_id") or ""
    only = body.get("strategies") or None

    opt_layer = _m._load_optimized_interval(symbol.upper(), interval) or {}
    opt_strategies = opt_layer.get("strategies") or {}
    if not opt_strategies:
        raise HTTPException(400, f"{symbol.upper()} {interval} 还没优化过, 先跑一次「⚡ 优化参数」")

    wf_all = opt_layer.get("wf_validation") or {}
    wf_best = wf_all.get("best") or {}

    # 选目标: WF 判定不是 robust 的。没有 WF 结论的也算(等于没验过, 不能当稳健)。
    targets, reasons = [], {}
    for sname in opt_strategies:
        if only and sname not in only:
            continue
        v = (wf_best.get(sname) or {}).get("verdict")
        if v == "robust":
            continue
        targets.append(sname)
        reasons[sname] = v or "未验证"
    if not targets:
        return {"symbol": symbol.upper(), "interval": interval, "targets": [],
                "message": "这些策略的 WF 结论都已经是「稳健」, 没有需要去过拟合的"}

    _m.prog_start(task_id, title="去过拟合优化",
                  scope=f"{symbol.upper()} · {len(targets)} 个策略 · 最多 {max_rounds} 轮",
                  steps=[{"key": "plan", "label": "切分 WF 窗口"},
                         {"key": "search", "label": "按样本外一致性搜参"},
                         {"key": "verify", "label": "完整 WF 复验"}])
    try:
        _m.prog_step(task_id, "plan", total=1, done=0, unit=" 步", workers=1,
                     running=[symbol.upper()], note="")
        wf_period = _m._wf_period_for_interval(interval)
        df = get_ohlcv(symbol, market, interval=interval, period=wf_period)
        plan = _m._build_wf_plan(df, train_years=3, test_months=6)
        if not plan.get("ok"):
            _m.prog_end(task_id, "error", plan.get("reason", "WF 窗口切不出来"))
            raise HTTPException(400, plan.get("reason", "数据不足, 切不出 WF 窗口"))
        windows = plan["windows"]
        win_bytes = _pickle.dumps(windows)

        results, rejected = {}, {}
        still = list(targets)
        prev_best = {}
        n_workers = max(1, min(_m.N_WORKERS, len(still)))

        for rnd in range(1, max_rounds + 1):
            if not still:
                break
            _m.prog_step(task_id, "search",
                         note=f"第 {rnd}/{max_rounds} 轮 · 每个 trial 要在 {len(windows) - holdout_n} 个窗口上各回测一次")
            tr = _m.ProgTracker(task_id, list(still), min(n_workers, len(still)),
                                unit=f" 个策略(第{rnd}轮)")
            tasks = [(win_bytes, sn, dynamic_pos, n_trials, holdout_n,
                      (opt_strategies.get(sn) or {}).get("params") or {}, rnd, prev_best.get(sn))
                     for sn in still]
            round_out = {}
            with _ProcPool(max_workers=min(n_workers, len(tasks)),
                           initializer=proc_role.mark_pool_worker) as pool:
                futs = {pool.submit(_m._deoverfit_one_strategy, t): t[1] for t in tasks}
                for f in _as_completed(futs):
                    _m.prog_check_cancel(task_id, pool, futs)
                    sn = futs[f]
                    try:
                        round_out[sn] = f.result()
                    except Exception as e:
                        round_out[sn] = {"sname": sn, "ok": False,
                                         "reason": f"{type(e).__name__}: {str(e)[:120]}"}
                    tr.done_one(sn)

            nxt = []
            for sn, r in round_out.items():
                if not r.get("ok"):
                    rejected[sn] = r.get("reason", "搜索未产出")
                    continue
                prev_best[sn] = r.get("params")
                # 收不收: 搜索集和持出集**都**要过线。只看搜索集就是自欺欺人。
                ok_search = r["search_consistency"] >= 65 and r["search_avg_sharpe"] > 0
                ok_hold = r["holdout_consistency"] >= 65 and r["holdout_avg_sharpe"] > 0
                if ok_search and ok_hold:
                    results[sn] = r
                elif rnd < max_rounds:
                    nxt.append(sn)
                else:
                    rejected[sn] = (
                        f"{max_rounds} 轮都没跑到稳健(搜索集一致性 {r['search_consistency']}%, "
                        f"持出集 {r['holdout_consistency']}%), 原参数保持不动")
            still = nxt

        # ── 复验 + 落盘: 拿完整 WF 重新判一次 verdict, 不信搜索过程自报的数 ──
        saved = {}
        if results:
            _m.prog_step(task_id, "verify", total=len(results), done=0, unit=" 个策略",
                         workers=1, running=list(results)[:1], note="")
            new_strats = {}
            for sn, r in results.items():
                base = dict(opt_strategies.get(sn) or {})
                base.update({
                    "params": r["params"],
                    "method": "deoverfit",
                    "deoverfit": {
                        "round": r["round"], "trials": r["trials"],
                        "search_windows": r["search_windows"], "holdout_windows": r["holdout_windows"],
                        "search_consistency": r["search_consistency"],
                        "holdout_consistency": r["holdout_consistency"],
                        "prev_verdict": reasons.get(sn),
                    },
                })
                new_strats[sn] = base
            wf_new = _m._run_wf_validation_from_plan(symbol, new_strats, dynamic_pos, plan,
                                                    task_id=task_id) or {}
            # 复验没到 robust 的不落盘 —— 宁可保持原样, 也不要换上一份"看着更好"
            # 但实际没验住的参数。
            for sn in list(new_strats):
                v = (wf_new.get(sn) or {}).get("verdict")
                if v != "robust":
                    rejected[sn] = f"完整 WF 复验只到 {v or '未知'}, 不落盘"
                    new_strats.pop(sn, None)
                    wf_new.pop(sn, None)
            if new_strats:
                save_data = dict(opt_layer)
                save_data["strategies"] = new_strats
                save_data["optimized_at"] = datetime.utcnow().isoformat()
                existing_wf = {k: dict(v) for k, v in wf_all.items() if isinstance(v, dict)}
                existing_wf.setdefault("best", {})
                existing_wf["best"] = {**existing_wf["best"], **wf_new}
                save_data["wf_validation"] = existing_wf
                _merged = _m._write_interval_layer(
                    _m._load_optimized(symbol.upper()) or {}, interval, save_data,
                    strategy_payload=False)
                _m._save_optimized(symbol.upper(), _merged)
                saved = {sn: wf_new[sn] for sn in new_strats if sn in wf_new}
                _m._promote_to_standard(symbol.upper(), market.upper(), wf_new, new_strats,
                                        period=opt_layer.get("period", "1y"),
                                        dynamic_pos=dynamic_pos, interval=interval)

        _m.prog_end(task_id, "done",
                    f"{len(saved)}/{len(targets)} 个转为稳健" + (f", {len(rejected)} 个没成" if rejected else ""))
        return _m._clean({
            "symbol": symbol.upper(), "market": market.upper(), "interval": interval,
            "targets": targets, "target_reasons": reasons,
            "max_rounds": max_rounds, "n_trials": n_trials, "holdout": holdout_n,
            "total_windows": len(windows),
            "succeeded": saved, "details": results, "rejected": rejected,
        })
    except _m.TaskCancelled:
        raise
    except HTTPException:
        raise
    except Exception as e:
        _m.prog_end(task_id, "error", f"去过拟合失败: {str(e)[:150]}")
        _m.logger.exception(f"[deoverfit] {symbol} 失败")
        raise HTTPException(500, f"去过拟合失败: {str(e)}")


@router.get("/optimized-params/{market}/{symbol}")
def get_optimized_params(market: str, symbol: str, interval: str = Query("1d")):
    """获取已保存的个股优化参数"""
    data = _m._load_optimized_interval(symbol.upper(), interval)
    if data is None:
        raise HTTPException(404, f"{symbol.upper()} 尚未进行参数优化")
    return data


@router.get("/custom-params/{symbol}")
def get_custom_params(symbol: str, interval: str = Query("1d")):
    """获取个股自定义标准参数（WF 稳健后永久保存的）"""
    data = _m._load_custom_params_interval(symbol.upper(), interval)
    if not data:
        raise HTTPException(404, f"{symbol.upper()} 尚无自定义标准参数")
    return {"symbol": symbol.upper(), "interval": _m._normalize_param_interval(interval), "strategies": data}


@router.post("/batch-optimize/{pool_name}")
def batch_optimize_pool(pool_name: str, body: dict = None):
    """启动批量优化任务（后台运行）"""
    import uuid
    import json

    if body is None:
        body = {}

    pools = json.loads(_m.POOLS_FILE.read_text(encoding="utf-8")) if _m.POOLS_FILE.exists() else {}
    pool_info = pools.get(pool_name)
    if not pool_info:
        raise HTTPException(404, f"股票池 '{pool_name}' 不存在")

    symbols = pool_info.get("symbols", [])
    market = pool_info.get("market", "US")
    if not symbols:
        raise HTTPException(400, "股票池为空")

    method = body.get("method", "bayesian")
    period = body.get("period", "2y")
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    period = _m._optimization_period_for_interval(interval)
    dynamic_pos = body.get("dynamic_position", False)
    strategy_names = body.get("strategies")
    n_trials = body.get("n_trials", 200)
    signal_type = body.get("signal_type", "buy_and_sell")

    job_id = uuid.uuid4().hex[:8]

    with _m._batch_lock:
        running = [jid for jid, j in _m._batch_jobs.items() if j.get("status") == "running"]
        if running:
            raise HTTPException(409, f"已有批量优化任务运行中 ({running[0]})")

    # 个股优化也占同一个槽位: 两边互斥, 免得进程池叠加把机器压垮
    busy = _m.opt_slot_acquire("pool", pool_name, method=method, interval=interval)
    if busy:
        raise HTTPException(409, _m.opt_slot_busy_message(busy))

    with _m._batch_lock:

        _m.register_job(_m._batch_jobs, job_id, {
            "job_id": job_id,
            "status": "running",
            "pool_name": pool_name,
            "market": market,
            "interval": interval,
            "method": method,
            "total": len(symbols),
            "completed": 0,
            "total_robust": 0,
            "total_standard_promoted": 0,
            "total_universal_promoted": 0,
            "recent_standard_promotions": [],
            "recent_universal_promotions": [],
            "current_symbol": "",
            "current_index": 0,
            "message": "启动中...",
            "results": _m.LockedDict(),
            "started_at": datetime.utcnow().isoformat(),
        })

    def _run_batch_and_release():
        # 槽位在上面的 opt_slot_acquire 里占的, 必须在后台线程结束时释放,
        # 不能在接口返回时释放 —— 接口是立刻返回的, 活儿还在这个线程里跑.
        try:
            _m._batch_optimize_worker(job_id, symbols, market, period, interval,
                                      method, dynamic_pos, strategy_names, n_trials, signal_type,
                                      keep_overfit=bool(body.get("keep_overfit", True)),
                                      sig_opts={
                                          "objective_mode": body.get("opt_objective", "composite"),
                                          "judge_mode": body.get("judge_mode", "first_touch"),
                                          "same_bar": body.get("same_bar", "adverse_first"),
                                          "window": int(body.get("sig_window", 10)),
                                          "mfe_target": float(body.get("sig_mfe_target", 5)),
                                          "mae_limit": float(body.get("sig_mae_limit", 10)),
                                          "entry_mode": body.get("entry_mode", "next_open"),
                                      })
        finally:
            _m.opt_slot_release(pool_name)
            job = _m._batch_jobs.get(job_id)
            if job and job.get("status") == "running":
                job["status"] = "error"
                job["error"] = job.get("error") or "任务异常结束"

    t = threading.Thread(target=_run_batch_and_release, daemon=True)
    t.start()

    return {"job_id": job_id, "total": len(symbols), "message": f"批量优化已启动: {len(symbols)}只股票"}


#: 跑完之后才需要给前端的重字段 (renderStratOptResults 只在 status=='done' 时读)
_BATCH_HEAVY_KEYS = ("results",)


@router.get("/batch-optimize/status/{job_id}")
async def batch_optimize_status(job_id: str):
    """查询批量优化进度 (async: 别跟重活抢线程池, 见 get_task_progress 的注释)。

    running 阶段不把 results 整包吐出去: 95 只 × 22 策略实测 1.4MB, 序列化要
    几百毫秒, 而这活是在事件循环上做的 —— 每 2 秒一次, 正好堵住刚改成 async
    的那批进度轮询。所以 running 只给计数和摘要, 跑完再给全量。

    (读写竞态由 LockedDict 自己保证, 这里不用再自己加锁或退避重试。)
    """
    job = _m._batch_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    try:
        snap = _m.snapshot_of(job)
        if snap.get("status") == "running":
            # 剔掉最重的 strategy_details (见上面 docstring 里的 1.4MB)
            snap["results"] = {
                sym: {k: v for k, v in (r or {}).items() if k != "strategy_details"}
                for sym, r in (snap.get("results") or {}).items()
            }
            snap["results_partial"] = True
        return _m.job_json_response(snap)
    except Exception:
        # 任何意外都不许变成 500: 这个端点每 2-3 秒被轮询一次, 一次 500 会让前端
        # 判定任务失败并停止跟踪, 而后台其实还在跑。记堆栈 + 返回降级快照。
        _m.logger.exception(f"[batch-status] 快照失败 job={job_id}, 返回降级快照")
        degraded = {"job_id": job_id, "results": {},
                    "message": "进度读取中…", "snapshot_degraded": True}
        # status/error 必须照实带出去, 不能像以前那样硬写 "running"。
        # 硬写的后果: 序列化要是**稳定**失败(某个值每次都转不动), 每一拍都回
        # "running", 已经 done 或 error 的任务就永远停在进度条上 —— 前端只在
        # 读到 done/cancelled/error 才收尾, 于是按钮一直禁用, 结果一直不渲染。
        for k in ("status", "error", "total", "completed", "task_total", "task_done",
                  "workers", "total_robust", "total_standard_promoted",
                  "total_universal_promoted", "current_symbol"):
            try:
                degraded[k] = job.get(k)
            except Exception:
                pass
        degraded.setdefault("status", "running")
        return _m.job_json_response(degraded)


@router.get("/progress/{task_id}")
async def get_task_progress(task_id: str):
    """通用任务进度 (前端用自己生成的 task_id 轮询)。

    没这条记录时返回 {"status":"unknown"} 而不是 404 —— 任务可能还没来得及登记,
    前端第一次轮询就撞 404 会白白报一次错。

    必须是 async def: 同步端点会被 FastAPI 丢进 anyio 线程池, 跟正在跑的重活
    (/compare、/applicability 这些 sync 端点, 尤其是它们创建进程池那几秒) 抢
    线程池名额和 GIL, 于是轮询迟迟拿不到响应, 面板就卡在上一次拿到的阶段上
    —— 表现就是"一直显示拉取行情"。这里只是加锁读个 dict, 微秒级,
    放事件循环上不会阻塞它。
    """
    t = _m.prog_get(task_id)
    if not t:
        return {"task_id": task_id, "status": "unknown"}
    return _m.job_json_response(t)


@router.post("/progress/{task_id}/cancel")
async def cancel_task_progress(task_id: str):
    """请求停止一个通用进度任务 (面板上的 ⏹ 停止)。

    以前只有批量优化和新策略验证能停, /compare、/compare-optimized、
    /applicability 这些一跑十几秒到几分钟的都停不了 —— 只能等, 或者上机器
    kill 进程。现在挂在统一进度表上, 凡是走 prog_start 的都能停。

    不强杀: 只插旗子, worker 循环下一拍自己跳出去, 顺手把还没开跑的分片撤掉。
    已经在跑的分片会先跑完 (进程池的 Future 对 RUNNING 状态 cancel 不动)。
    """
    ok = _m.prog_request_cancel(task_id)
    return {"task_id": task_id, "accepted": ok,
            "message": "已请求停止" if ok else "任务不存在或已结束"}


@router.get("/optimize/running")
async def optimize_running():
    """当前有没有优化在跑 (前端开跑前预检, 避免重复发请求)。

    async: 别跟重活抢线程池, 见 get_task_progress 的注释。"""
    return _m.opt_slot_status()


@router.post("/optimize/force-release")
def optimize_force_release():
    """强制释放优化槽位。

    只有在服务端槽位卡住 (比如上一次优化的进程被外部 kill 掉、
    finally 没能跑到) 时才用；正常取消请走 /batch-optimize/cancel。
    """
    cur = _m.opt_slot_status()
    _m.opt_slot_release()
    return {"released": cur.get("running", False), "was": cur}


@router.post("/batch-optimize/cancel/{job_id}")
def batch_optimize_cancel(job_id: str):
    """取消批量优化任务"""
    job = _m._batch_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    job["cancelled"] = True
    return {"message": "取消请求已发送"}


@router.get("/wf-status/{market}/{symbol}")
def get_wf_status(market: str, symbol: str, interval: str = Query("1d")):
    """获取个股的 WF 验证状态"""
    opt = _m._load_optimized_interval(symbol.upper(), interval)
    wf = opt.get("wf_validation", {}) if opt else {}
    if wf and any(isinstance(v, dict) and "verdict" in v for v in wf.values()):
        wf = {"best": wf}
    custom = _m._load_custom_params_interval(symbol.upper(), interval)
    return {
        "symbol": symbol.upper(),
        "wf_validation": wf,
        "custom_params": {k: {"promoted_at": v.get("promoted_at"), "source_sharpe": v.get("source_sharpe")}
                          for k, v in custom.items()},
    }


@router.post("/batch-wf/{pool_name}")
def batch_wf_pool(pool_name: str, body: dict = None):
    """启动批量 WF 验证（后台运行）"""
    import uuid
    import json

    if body is None:
        body = {}

    pools = json.loads(_m.POOLS_FILE.read_text(encoding="utf-8")) if _m.POOLS_FILE.exists() else {}
    pool_info = pools.get(pool_name)
    if not pool_info:
        raise HTTPException(404, f"股票池 '{pool_name}' 不存在")

    symbols = pool_info.get("symbols", [])
    market = pool_info.get("market", "US")
    if not symbols:
        raise HTTPException(400, "股票池为空")

    dynamic_pos = body.get("dynamic_position", False)
    param_source = body.get("param_source", "best")

    no_opt = [s for s in symbols if not (_m._load_optimized(s.upper()))]
    if len(no_opt) == len(symbols):
        raise HTTPException(400, "该池所有股票均未优化，请先进行批量优化")

    src_label = {"standard": "通用标准", "custom": "个股标准", "best": "最优参数", "bayesian": "贝叶斯", "grid": "穷举", "coarse_fine": "粗搜精搜"}.get(param_source, param_source)

    job_id = "wf_" + uuid.uuid4().hex[:8]

    with _m._batch_lock:
        running = [jid for jid, j in _m._batch_jobs.items() if j.get("status") == "running"]
        if running:
            raise HTTPException(409, f"已有任务运行中 ({running[0]})")

        _m.register_job(_m._batch_jobs, job_id, {
            "job_id": job_id,
            "status": "running",
            "type": "wf",
            "pool_name": pool_name,
            "param_source": param_source,
            "total": len(symbols),
            "completed": 0,
            "total_robust": 0,
            "message": f"启动WF批量验证 [{src_label}]...",
            "results": _m.LockedDict(),
            "started_at": datetime.utcnow().isoformat(),
        })

    t = threading.Thread(
        target=_m._batch_wf_worker,
        args=(job_id, symbols, market, dynamic_pos, param_source),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "total": len(symbols),
            "skipped": len(no_opt),
            "message": f"WF批量验证已启动 [{src_label}]: {len(symbols)}只股票 ({len(no_opt)}只无优化参数将跳过)"}


@router.post("/compare-optimized/{market}/{symbol}")
def compare_with_optimized(market: str, symbol: str, body: dict = None):
    """标准参数 vs 优化参数对比。

    这条路径以前完全没有进度上报, 但它可能很慢: N 个策略的对比回测之外,
    还藏着一步 _run_wf_validation —— Walk-Forward 要重跑十几个窗口, 比前面
    所有对比加起来还久。前端只能对着一个不动的转圈干等, 分不清是在跑还是死了。
    """
    import pickle as _pickle
    from backtest import get_ohlcv
    from concurrent.futures import ProcessPoolExecutor as _ProcPool, as_completed as _as_completed

    if body is None:
        body = {}
    period = body.get("period", "2y")
    dynamic_pos = body.get("dynamic_position", False)
    interval = _m._normalize_param_interval(body.get("interval", "1d"))
    strategy_names = body.get("strategies") or _m._get_active_strategies()
    task_id = body.get("task_id") or ""
    _m.prog_start(task_id, title="标准 vs 优化参数对比",
                  scope=f"{symbol.upper()} · {len(strategy_names)} 策略 · {period}",
                  steps=[{"key": "dl", "label": "拉取行情"},
                         {"key": "bt", "label": "对比回测"},
                         {"key": "wf", "label": "Walk-Forward 验证"},
                         {"key": "promote", "label": "参数晋级"}])

    _co_data_mtime = _m._data_mtime_sig([symbol], market, interval)
    _co_opt_fp = _m.OPTIMIZED_DIR / f"{symbol.upper()}.json"
    _co_opt_mtime = str(int(_co_opt_fp.stat().st_mtime) // 60) if _co_opt_fp.exists() else "0"
    co_cache_key = f"compare_opt:{market}:{symbol}:{interval}:{period}:{dynamic_pos}:{_co_data_mtime}:{_co_opt_mtime}:{_m.stable_hash(sorted(strategy_names))}"
    cached = _m.cache_get(co_cache_key, ttl=86400 * 7)
    if cached is not None:
        _m.prog_end(task_id, "done", "命中缓存")
        return cached

    _m.prog_step(task_id, "dl", total=1, unit=" 只", workers=1,
                 running=[symbol.upper()], note="")
    try:
        df = get_ohlcv(symbol, market, interval=interval, period=period)
        if len(df) < 1:
            _m.prog_end(task_id, "error", f"数据不足，只有 {len(df)} 条K线")
            raise HTTPException(400, f"数据不足，只有 {len(df)} 条K线")
    except HTTPException:
        raise
    except Exception as e:
        _m.prog_end(task_id, "error", f"获取数据失败: {str(e)[:120]}")
        raise HTTPException(500, f"获取数据失败: {str(e)}")

    optimized_data = _m._load_optimized_interval(symbol.upper(), interval)
    opt_strategies = optimized_data.get("strategies", {}) if optimized_data else {}

    # df 先 pickle 一次: 下面每个任务带的都是同一段 bytes, 父进程派活时不用
    # 把 DataFrame 重新序列化 N 遍 (22 个策略就是 22 遍)。
    df_bytes = _pickle.dumps(df)
    compare_tasks = []
    for sname in strategy_names:
        std_p = _m._get_standard_strategy_params(symbol.upper(), sname, interval=interval)
        opt_p = opt_strategies[sname].get("params", {}) if sname in opt_strategies else None
        dual = _m._get_dual_signal_params(symbol.upper(), sname, interval=interval)
        if dual["has_dual"]:
            compare_tasks.append((sname, df_bytes, dynamic_pos, std_p, opt_p,
                                  dual["buy_params"], dual["sell_params"]))
        else:
            compare_tasks.append((sname, df_bytes, dynamic_pos, std_p, opt_p))
    n_workers = max(1, min(_m.N_WORKERS, len(compare_tasks)))
    _m.prog_step(task_id, "bt")
    _tr = _m.ProgTracker(task_id, [t[0] for t in compare_tasks], n_workers,
                         unit=" 次回测")
    # map -> submit/as_completed: map 只能等全部跑完才返回, 中间一个进度都报不出来。
    # 按索引回填而不是 append: as_completed 是完成顺序, 而下面 sort 是**稳定**排序,
    # 同分策略(零交易的 composite_score 恒为 0.3425, 回测失败的 std_score 恒为 0,
    # 一次对比里通常好几个)的 rank 就会每次抽签。更麻烦的是结果会被 cache_set
    # 冻进 7 天缓存 —— 之后一周前端看到的都是那次抽签结果。
    results = [None] * len(compare_tasks)
    with _ProcPool(max_workers=n_workers, initializer=proc_role.mark_pool_worker) as pool:
        _futs = {pool.submit(_m._compare_bt_one, t): i for i, t in enumerate(compare_tasks)}
        for _f in _as_completed(_futs):
            _m.prog_check_cancel(task_id, pool, _futs)
            _i = _futs[_f]
            results[_i] = _f.result()
            _tr.done_one(compare_tasks[_i][0])
    results = [r for r in results if r is not None]

    results.sort(key=lambda r: r.get("std_score", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    method_best = optimized_data.get("method_best", {}) if optimized_data else {}

    raw_wf = optimized_data.get("wf_validation", {}) if optimized_data else {}
    if raw_wf and any(isinstance(v, dict) and "verdict" in v for v in raw_wf.values()):
        raw_wf = {"best": raw_wf}
    if optimized_data and opt_strategies and "best" not in raw_wf:
        # 这一步能比前面整段对比还慢(要重跑十几个 WF 窗口), 单独给它一个阶段,
        # 否则用户看到的是"对比回测 22/22 完成"之后长时间毫无动静。
        _m.prog_step(task_id, "wf",
                     note="正在切分 Walk-Forward 窗口…")
        best_wf = _m._run_wf_validation(symbol, market, opt_strategies, dynamic_pos,
                                        interval=interval, period=period, task_id=task_id)
        if best_wf:
            raw_wf["best"] = best_wf
            optimized_data["wf_validation"] = raw_wf
            _m._save_optimized(symbol.upper(), _m._write_interval_layer(_m._load_optimized(symbol.upper()) or {}, interval, optimized_data, strategy_payload=False))

    best_wf = raw_wf.get("best", {})
    if best_wf and opt_strategies:
        # 实测这一步占整个请求的 62%(3.7s/5.9s), 以前完全不上报 ——
        # 前面进度条早就 100% 了它还在跑, 看着就是卡死。
        _m._promote_to_standard(
            symbol.upper(), market.upper(), best_wf, opt_strategies,
            period=period, dynamic_pos=dynamic_pos, interval=interval,
            task_id=task_id
        )

    result = _m._clean({
        "symbol": symbol.upper(),
        "market": market.upper(),
        "period": period,
        "interval": interval,
        "data_len": len(df),
        "data_start": (df.index[0].isoformat() if len(df) else None),
        "data_end": (df.index[-1].isoformat() if len(df) else None),
        "data_days": _m._df_span_days(df),
        "dynamic_position": dynamic_pos,
        "has_optimized": optimized_data is not None,
        "optimized_at": optimized_data.get("optimized_at") if optimized_data else None,
        "optimized_period": optimized_data.get("period") if optimized_data else None,
        "ranking": results,
        "method_best": method_best,
        "wf_validation": raw_wf,
    })
    _m.cache_set(co_cache_key, result)
    _m.prog_end(task_id, "done", f"完成 {len(results)} 个策略对比")
    return result


@router.post("/paper-trade/open")
def paper_trade_open(body: dict):
    """开仓：买入建仓"""
    symbol = body.get("symbol", "").upper()
    market = body.get("market", "US").upper()
    strategy = body.get("strategy", "")
    quantity = int(body.get("quantity", 100))
    price = float(body.get("price", 0))
    interval = body.get("interval", "1d")
    notes = body.get("notes", "")

    if not symbol or price <= 0 or quantity <= 0:
        raise HTTPException(400, "symbol/price/quantity 必填")

    data = _m._load_paper_trades()
    pos_id = f"PT{data['next_id']:04d}"
    data["next_id"] = data.get("next_id", 1) + 1

    position = {
        "id": pos_id,
        "symbol": symbol,
        "market": market,
        "strategy": strategy,
        "interval": interval,
        "quantity": quantity,
        "entry_price": price,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cost": round(price * quantity, 2),
        "status": "open",
        "notes": notes,
        "stop_loss": body.get("stop_loss"),
        "take_profit": body.get("take_profit"),
    }
    data["positions"].append(position)
    _m._save_paper_trades(data)
    return {"success": True, "position": position}


@router.post("/paper-trade/close")
def paper_trade_close(body: dict):
    """平仓：卖出平仓"""
    pos_id = body.get("id", "")
    price = float(body.get("price", 0))
    quantity = body.get("quantity")
    reason = body.get("reason", "手动平仓")

    if not pos_id or price <= 0:
        raise HTTPException(400, "id/price 必填")

    data = _m._load_paper_trades()
    pos = None
    for p in data["positions"]:
        if p["id"] == pos_id and p["status"] == "open":
            pos = p
            break
    if not pos:
        raise HTTPException(404, f"持仓 {pos_id} 不存在或已平仓")

    pre_close_qty = pos["quantity"]
    close_qty = int(quantity) if quantity else pos["quantity"]
    close_qty = min(close_qty, pos["quantity"])

    pnl = round((price - pos["entry_price"]) * close_qty, 2)
    pnl_pct = round((price - pos["entry_price"]) / pos["entry_price"] * 100, 2)

    trade_record = {
        "id": pos["id"],
        "symbol": pos["symbol"],
        "market": pos["market"],
        "strategy": pos["strategy"],
        "interval": pos.get("interval", "1d"),
        "quantity": close_qty,
        "entry_price": pos["entry_price"],
        "entry_time": pos["entry_time"],
        "exit_price": price,
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "hold_days": (datetime.now() - datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S")).days,
    }
    data["history"].append(trade_record)

    if close_qty >= pos["quantity"]:
        pos["status"] = "closed"
        pos["exit_price"] = price
        pos["exit_time"] = trade_record["exit_time"]
        pos["pnl"] = pnl
        pos["pnl_pct"] = pnl_pct
    else:
        pos["quantity"] -= close_qty
        pos["cost"] = round(pos["entry_price"] * pos["quantity"], 2)

    _m._save_paper_trades(data)
    return {"success": True, "trade": trade_record}


@router.get("/paper-trade/positions")
def paper_trade_positions():
    """获取所有持仓和历史，含浮动盈亏"""
    from routers.market import get_quote
    from concurrent.futures import ThreadPoolExecutor

    data = _m._load_paper_trades()
    open_positions = [p for p in data["positions"] if p["status"] == "open"]

    # Parallel quote fetch: previously this loop was sequential and a single
    # yfinance rate-limit retry × 10 positions made the endpoint take 30s.
    # I/O-bound (HTTP), so threads are fine; cap at 8 to stay polite to upstreams.
    def _fetch(p):
        try:
            q = get_quote(p.get("market", "US"), p["symbol"])
            return p["id"], q.get("price", 0) or 0
        except Exception:
            return p["id"], 0
    price_map: dict[str, float] = {}
    if open_positions:
        with ThreadPoolExecutor(max_workers=min(8, len(open_positions))) as ex:
            for pid, cur in ex.map(_fetch, open_positions):
                price_map[pid] = cur

    total_unrealized = 0.0
    total_market_value = 0.0
    for p in open_positions:
        cur_price = price_map.get(p["id"], 0)
        p["current_price"] = cur_price
        if cur_price > 0:
            cost = p["entry_price"] * p["quantity"]
            mkt_val = cur_price * p["quantity"]
            unrealized = round(mkt_val - cost, 2)
            unrealized_pct = round((cur_price / p["entry_price"] - 1) * 100, 2)
            p["unrealized_pnl"] = unrealized
            p["unrealized_pct"] = unrealized_pct
            total_unrealized += unrealized
            total_market_value += mkt_val
        else:
            p["unrealized_pnl"] = 0
            p["unrealized_pct"] = 0
            total_market_value += p["entry_price"] * p["quantity"]

    total_cost = sum(p["entry_price"] * p["quantity"] for p in open_positions)
    history = data.get("history", [])
    total_pnl = sum(t.get("pnl", 0) for t in history)
    win_count = sum(1 for t in history if t.get("pnl", 0) > 0)
    win_rate = round(win_count / len(history) * 100, 1) if history else 0

    return {
        "open": open_positions,
        "history": history,
        "summary": {
            "open_count": len(open_positions),
            "total_cost": round(total_cost, 2),
            "total_market_value": round(total_market_value, 2),
            "total_unrealized": round(total_unrealized, 2),
            "closed_count": len(history),
            "total_pnl": round(total_pnl, 2),
            "win_rate": win_rate,
        }
    }


@router.post("/paper-trade/monitor")
def paper_trade_monitor():
    """监控所有持仓的SELL信号"""
    from backtest import get_ohlcv
    from strategies import get_strategy

    data = _m._load_paper_trades()
    open_positions = [p for p in data["positions"] if p["status"] == "open"]

    if not open_positions:
        return {"alerts": [], "positions_checked": 0}

    # 逐个持仓串行跑: 每个都是"一次取数(I/O) + add_indicators/generate_signals(CPU)"。
    # 持仓一多就是线性叠加, 而前端这个按钮是同步等结果的。
    # 用线程池: 取数那半段是纯 I/O 能真并发; 指标计算那半段虽有 GIL,
    # 但 pandas/numpy 的重算子内部会放开 GIL, 所以仍有收益。
    # 不用进程池 —— 每个 worker 要重新 import backtest(实测 5s), 比省下的还多。
    from concurrent.futures import ThreadPoolExecutor as _TP, as_completed as _ac

    def _check_one(pos):
        try:
            interval = pos.get("interval", "1d")
            period_map = {"1m": "2d", "5m": "10d", "15m": "20d", "30m": "30d",
                          "1h": "6mo", "1d": "1y"}
            period = period_map.get(interval, "1y")
            df = get_ohlcv(pos["symbol"], pos["market"], interval=interval, period=period)
            if df is None or len(df) < 30:
                return None

            strategy = get_strategy(pos["strategy"])
            if not strategy:
                return None
            # 尝试双参数分离模式
            dual = _m._get_dual_signal_params(pos["symbol"], pos["strategy"])
            if dual["has_dual"]:
                # 双参数分离：buy信号用buy_params，sell信号用sell_params
                bp = {**strategy.default_params, **dual["buy_params"]}
                sp_dict = {**strategy.default_params, **dual["sell_params"]}
                df_buy = strategy.add_indicators(df.copy(), bp)
                df_buy = strategy.generate_signals(df_buy, bp)
                df_sell_run = strategy.add_indicators(df.copy(), sp_dict)
                df_sell_run = strategy.generate_signals(df_sell_run, sp_dict)
                common_idx = df_buy.index.intersection(df_sell_run.index)
                df_sig = df_buy.loc[common_idx].copy()
                df_sig["sell_signal"] = df_sell_run.loc[common_idx, "sell_signal"]
            else:
                params = _m._get_standard_strategy_params(pos["symbol"], pos["strategy"])
                params_source = "个股标准/通用标准"
                # 必须走 _load_optimized_interval。以前这里是 json.load 整个文件再
                # .get("strategies"), 新格式(strategies 在 intervals 层里)一律拿到 {},
                # 于是监控**静默**用标准参数跑信号, 界面却不会说一个字。
                opt_layer = _m._load_optimized_interval(pos["symbol"], pos.get("interval") or "1d") or {}
                _opt_entry = (opt_layer.get("strategies") or {}).get(pos["strategy"])
                if isinstance(_opt_entry, dict) and _opt_entry.get("params"):
                    params.update(_opt_entry["params"])
                    params_source = "最优参数"
                df_ind = strategy.add_indicators(df.copy(), params)
                df_sig = strategy.generate_signals(df_ind, params)
            if len(df_sig) < 2:
                return None

            last = df_sig.iloc[-1]
            current_price = float(last.get("close", 0))
            pnl = round((current_price - pos["entry_price"]) * pos["quantity"], 2)
            pnl_pct = round((current_price - pos["entry_price"]) / pos["entry_price"] * 100, 2)

            alert_info = {
                "id": pos["id"],
                "symbol": pos["symbol"],
                "market": pos["market"],
                "strategy": pos["strategy"],
                "quantity": pos["quantity"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "sell_signal": bool(last.get("sell_signal", 0)),
                "time": str(df_sig.index[-1]),
                "params_source": ("Buy/Sell 分离参数" if dual["has_dual"] else params_source),
            }

            if pos.get("stop_loss") and current_price <= pos["stop_loss"]:
                alert_info["stop_loss_triggered"] = True
            if pos.get("take_profit") and current_price >= pos["take_profit"]:
                alert_info["take_profit_triggered"] = True

            return alert_info
        except Exception as e:
            _m.logger.warning(f"Monitor {pos['id']} {pos['symbol']} failed: {e}")
            return None

    alerts = []
    with _TP(max_workers=max(1, min(8, len(open_positions)))) as _ex:
        for _f in _ac([_ex.submit(_check_one, p) for p in open_positions]):
            _a = _f.result()
            if _a is not None:
                alerts.append(_a)


    from datetime import datetime
    return {
        "alerts": alerts,
        "positions_checked": len(open_positions),
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.delete("/paper-trade/position/{pos_id}")
def paper_trade_delete(pos_id: str):
    """删除一条持仓记录（仅已平仓）"""
    data = _m._load_paper_trades()
    found = False
    for p in data["positions"]:
        if p["id"] == pos_id:
            if p["status"] == "open":
                raise HTTPException(400, "不能删除未平仓的持仓，请先平仓")
            data["positions"] = [x for x in data["positions"] if x["id"] != pos_id]
            found = True
            break
    if not found:
        before = len(data["history"])
        data["history"] = [h for h in data["history"] if h["id"] != pos_id]
        found = len(data["history"]) < before
    if not found:
        raise HTTPException(404, "记录不存在")
    _m._save_paper_trades(data)
    return {"deleted": pos_id}


@router.post("/auto-optimize/start")
async def auto_optimize_start(request: Request):
    """启动自动优化流水线（后台运行）"""
    from pathlib import Path
    global _auto_opt_status_ref

    if _m._auto_opt_status["running"]:
        return {"status": "already_running", "progress": _m._auto_opt_status["progress"]}

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    n_trials = body.get("trials", 150)
    top_n = body.get("top", 0)
    strategies_filter = body.get("strategies", "")

    def _run():
        _m._auto_opt_status.update({"running": True, "progress": "initializing", "result": None, "error": None})
        try:
            import subprocess, sys
            cmd = [sys.executable, "auto_optimize.py", f"--trials={n_trials}"]
            if top_n:
                cmd.append(f"--top={top_n}")
            if strategies_filter:
                cmd.append(f"--strategies={strategies_filter}")

            proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent.parent),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                if "Step" in line or "📊" in line or "🔬" in line or "🔍" in line or "🏁" in line:
                    _m._auto_opt_status["progress"] = line.strip()
            proc.wait()

            _m._auto_opt_status["running"] = False
            if proc.returncode == 0:
                _m._auto_opt_status["progress"] = "completed"
                _m._auto_opt_status["result"] = {"output_lines": len(lines), "return_code": 0}
            else:
                _m._auto_opt_status["progress"] = "failed"
                _m._auto_opt_status["error"] = "\n".join(lines[-20:])
        except Exception as e:
            _m._auto_opt_status["running"] = False
            _m._auto_opt_status["progress"] = "error"
            _m._auto_opt_status["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "trials": n_trials, "top": top_n}


@router.get("/auto-optimize/status")
async def auto_optimize_status():
    """查询自动优化进度"""
    return _m._auto_opt_status


@router.get("/auto-optimize/report")
async def auto_optimize_report():
    """获取最新优化报告"""
    import json
    from pathlib import Path
    result_dir = Path(__file__).parent.parent / "auto_optimize_results"
    if not result_dir.exists():
        return {"error": "no reports"}

    reports = sorted(result_dir.glob("report_*.json"), reverse=True)
    if not reports:
        return {"error": "no reports"}

    with open(reports[0]) as f:
        return json.load(f)
