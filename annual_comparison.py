#!/usr/bin/env python3
"""
年度策略收益 vs NASDAQ 指数收益对比
===================================
对 NASDAQ100 股票池，按年计算每个策略的平均收益，与 QQQ(纳指ETF) 收益对比
"""

import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from backtest import get_ohlcv, run_strategy_backtest
from strategies import STRATEGY_REGISTRY

# ── 配置 ──
POOL_FILE = Path(__file__).parent / "stock_pools.json"
YEARS = [2021, 2022, 2023, 2024, 2025]  # 分析的年份


def load_nasdaq100_symbols():
    pools = json.load(open(POOL_FILE, encoding="utf-8"))
    for name, pool in pools.items():
        if "nasdaq" in name.lower() or "100" in name:
            return pool["symbols"], pool.get("market", "US")
    return [], "US"


def get_index_annual_returns():
    """获取 QQQ (NASDAQ 100 ETF) 的年度收益（使用本地+增量数据系统）"""
    print("  ⬇️  加载 QQQ 指数数据...")
    qqq = get_ohlcv("QQQ", period="10y", interval="1d", market="US")
    if qqq is None or qqq.empty:
        return {}

    returns = {}
    for year in YEARS:
        year_data = qqq[qqq.index.year == year]
        if len(year_data) < 10:
            continue
        start_price = float(year_data['Close'].iloc[0])
        end_price = float(year_data['Close'].iloc[-1])
        ret = (end_price - start_price) / start_price * 100
        returns[year] = round(ret, 2)
    return returns


def run_annual_analysis():
    t0 = time.time()

    print(f"\n{'═'*90}")
    print(f"  📊 年度策略收益 vs NASDAQ 指数对比")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*90}")

    # 1. 获取 NASDAQ 指数年度收益
    index_returns = get_index_annual_returns()
    print(f"  QQQ 年度收益: {index_returns}")

    # 2. 加载股票池
    symbols, market = load_nasdaq100_symbols()
    strategies = list(STRATEGY_REGISTRY.keys())
    print(f"\n  股票池: {len(symbols)} 只 | 策略: {len(strategies)} 个 | 年份: {YEARS}")

    # 3. 下载所有股票数据
    print(f"\n  ⬇️  下载股票数据...")
    all_data = {}
    for i, sym in enumerate(symbols):
        try:
            df = get_ohlcv(sym, market, interval="1d", period="10y")
            if df is not None and len(df) > 50:
                all_data[sym] = df
                print(f"    [{i+1}/{len(symbols)}] {sym}: {len(df)} bars ✓")
        except Exception as e:
            print(f"    [{i+1}/{len(symbols)}] {sym}: ❌ {e}")

    print(f"  📦 有效股票: {len(all_data)}/{len(symbols)}")

    # 4. 按年、按策略计算收益
    # results[year][strategy] = [list of returns for each stock]
    results = {year: {s: [] for s in strategies} for year in YEARS}
    # Buy & Hold
    bh_results = {year: [] for year in YEARS}

    total = len(all_data)
    for idx, (sym, full_df) in enumerate(all_data.items()):
        print(f"\n  📌 [{idx+1}/{total}] {sym}", end="", flush=True)

        for year in YEARS:
            # 取该年数据
            year_df = full_df[full_df.index.year == year].copy()
            if len(year_df) < 20:
                continue

            # Buy & Hold 收益
            bh_ret = (float(year_df['close'].iloc[-1]) - float(year_df['close'].iloc[0])) / float(year_df['close'].iloc[0]) * 100
            bh_results[year].append(bh_ret)

            for sname in strategies:
                try:
                    r = run_strategy_backtest(year_df, sname)
                    ret = r["summary"]["total_return"]
                    results[year][sname].append(ret)
                except Exception:
                    pass

        print(f" ✓", end="", flush=True)

    print(f"\n\n{'═'*90}")
    print(f"  📋 年度对比报告")
    print(f"{'═'*90}")

    # 5. 打印汇总表
    # Header
    header = f"  {'策略':<24}"
    for year in YEARS:
        header += f" {year:>8}"
    header += f" {'平均':>8}"
    print(header)
    print(f"  {'─'*80}")

    # QQQ 指数
    row = f"  {'📈 QQQ (NASDAQ指数)':<24}"
    qqq_vals = []
    for year in YEARS:
        val = index_returns.get(year, None)
        if val is not None:
            row += f" {val:>7.1f}%"
            qqq_vals.append(val)
        else:
            row += f" {'N/A':>8}"
    avg_qqq = np.mean(qqq_vals) if qqq_vals else 0
    row += f" {avg_qqq:>7.1f}%"
    print(row)

    # Buy & Hold 平均
    row = f"  {'📊 Buy&Hold (均值)':<24}"
    bh_avgs = []
    for year in YEARS:
        if bh_results[year]:
            avg = np.mean(bh_results[year])
            row += f" {avg:>7.1f}%"
            bh_avgs.append(avg)
        else:
            row += f" {'N/A':>8}"
    avg_bh = np.mean(bh_avgs) if bh_avgs else 0
    row += f" {avg_bh:>7.1f}%"
    print(row)

    print(f"  {'─'*80}")

    # 每个策略
    strategy_avgs = {}
    for sname in strategies:
        row = f"  {sname:<24}"
        yearly_avgs = []
        for year in YEARS:
            rets = results[year][sname]
            if rets:
                # 截尾平均（去掉前后各5%）
                sorted_rets = sorted(rets)
                trim = max(1, len(sorted_rets) // 20)  # 5%
                trimmed = sorted_rets[trim:-trim] if len(sorted_rets) > 2*trim else sorted_rets
                avg = np.mean(trimmed)
                row += f" {avg:>7.1f}%"
                yearly_avgs.append(avg)
            else:
                row += f" {'N/A':>8}"

        total_avg = np.mean(yearly_avgs) if yearly_avgs else 0
        strategy_avgs[sname] = total_avg
        row += f" {total_avg:>7.1f}%"

        # 标记 vs QQQ
        if total_avg > avg_qqq:
            row += " ✅"
        else:
            row += " ❌"

        print(row)

    # 排名
    print(f"\n  {'─'*80}")
    print(f"\n  🏆 策略排名（按年均收益）:")
    ranked = sorted(strategy_avgs.items(), key=lambda x: x[1], reverse=True)
    for i, (sname, avg) in enumerate(ranked):
        beat = "🟢 超越指数" if avg > avg_qqq else "🔴 落后指数"
        diff = avg - avg_qqq
        print(f"    {i+1:>2}. {sname:<24} {avg:>7.1f}%  ({diff:>+6.1f}% vs QQQ)  {beat}")

    # 统计
    beat_count = sum(1 for _, v in strategy_avgs.items() if v > avg_qqq)
    print(f"\n  📊 统计: {beat_count}/{len(strategies)} 个策略超越 QQQ 指数")
    print(f"  ⏱️  总耗时: {(time.time()-t0)/60:.1f} 分钟")

    # 保存结果
    report = {
        "generated_at": datetime.now().isoformat(),
        "qqq_returns": index_returns,
        "bh_avg_returns": {str(y): round(np.mean(bh_results[y]), 2) if bh_results[y] else None for y in YEARS},
        "strategy_annual_returns": {},
        "strategy_avg_returns": strategy_avgs,
    }
    for sname in strategies:
        report["strategy_annual_returns"][sname] = {}
        for year in YEARS:
            rets = results[year][sname]
            if rets:
                sorted_rets = sorted(rets)
                trim = max(1, len(sorted_rets) // 20)
                trimmed = sorted_rets[trim:-trim] if len(sorted_rets) > 2*trim else sorted_rets
                report["strategy_annual_returns"][sname][str(year)] = round(float(np.mean(trimmed)), 2)

    report_file = Path(__file__).parent / "annual_comparison_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📄 报告已保存: {report_file}")


if __name__ == "__main__":
    run_annual_analysis()
