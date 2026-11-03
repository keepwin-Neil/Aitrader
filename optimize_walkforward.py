"""
Walk-Forward + Monte Carlo + Portfolio Optimization
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from backtest import get_ohlcv, run_strategy_backtest, run_backtest
from strategies import get_strategy, STRATEGY_REGISTRY

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]

TOP_STRATEGIES = ["claude_rsi_macd_trend", "claude_ensemble_voting", "claude_momentum_breakout",
                  "claude_dual_ma_cross", "claude_mean_reversion", "claude_turtle_trading"]


def load_data(period="5y"):
    datasets = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, interval="1d", period=period)
            datasets[(symbol, market)] = df
            print(f"  OK {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  FAIL {symbol}: {e}")
    return datasets


# ═══════════════════════════════════════════════════════════════════════
#  Part 1: Walk-Forward 验证
# ═══════════════════════════════════════════════════════════════════════

def walk_forward_single(df, strategy_name, train_days=756, test_days=126, dynamic_pos=False):
    """单股票Walk-Forward"""
    windows = []
    i = 0
    while i + train_days + test_days <= len(df):
        train_df = df.iloc[i:i+train_days]
        test_df = df.iloc[i+train_days:i+train_days+test_days]

        try:
            train_r = run_strategy_backtest(train_df, strategy_name, dynamic_position=dynamic_pos)
            train_sharpe = train_r["summary"]["sharpe"]
        except:
            train_sharpe = 0

        try:
            test_r = run_strategy_backtest(test_df, strategy_name, dynamic_position=dynamic_pos)
            ts = test_r["summary"]
        except:
            ts = {"sharpe": 0, "total_return": 0, "max_drawdown": 0, "win_rate": 0, "total_trades": 0}

        windows.append({
            "period": f"{test_df.index[0].date()} → {test_df.index[-1].date()}",
            "train_sharpe": train_sharpe,
            "test_sharpe": ts["sharpe"],
            "test_return": ts["total_return"],
            "test_dd": ts["max_drawdown"],
            "test_wr": ts["win_rate"],
            "test_trades": ts["total_trades"],
        })
        i += test_days

    return windows


def run_walk_forward(datasets, strategies, dynamic_pos=False):
    """全股票Walk-Forward汇总"""
    print(f"\n{'='*90}")
    print(f"  Walk-Forward 验证 (训练3年 + 测试6个月)  {'+ 动态仓位' if dynamic_pos else ''}")
    print(f"{'='*90}")

    all_results = {}

    for sname in strategies:
        all_windows = []
        for (symbol, market), df in datasets.items():
            windows = walk_forward_single(df, sname, dynamic_pos=dynamic_pos)
            all_windows.extend(windows)

        if not all_windows:
            continue

        test_sharpes = [w["test_sharpe"] for w in all_windows]
        test_returns = [w["test_return"] for w in all_windows]
        test_dds = [w["test_dd"] for w in all_windows]
        train_sharpes = [w["train_sharpe"] for w in all_windows]

        consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100
        avg_ts = np.mean(test_sharpes)
        std_ts = np.std(test_sharpes)
        avg_tr = np.mean(test_returns)
        avg_dd = np.mean(test_dds)
        overfit_ratio = np.mean(test_sharpes) / max(np.mean(train_sharpes), 0.01)

        all_results[sname] = {
            "avg_test_sharpe": round(avg_ts, 3),
            "std_test_sharpe": round(std_ts, 3),
            "avg_test_return": round(avg_tr, 2),
            "avg_test_dd": round(avg_dd, 2),
            "consistency": round(consistency, 1),
            "overfit_ratio": round(overfit_ratio, 2),
            "num_windows": len(all_windows),
        }

    # 按一致性排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["consistency"], reverse=True)

    print(f"\n  {'Strategy':<20} {'AvgSharpe':>10} {'StdSharpe':>10} {'AvgRet%':>8} {'AvgDD%':>7} {'Consist%':>9} {'OvfitR':>7} {'Windows':>8}")
    print(f"  {'-'*84}")
    for sname, r in sorted_results:
        star = " *" if r["consistency"] >= 65 and r["avg_test_sharpe"] > 0.3 else ""
        print(f"  {sname:<20} {r['avg_test_sharpe']:>10.3f} {r['std_test_sharpe']:>10.3f} {r['avg_test_return']:>8.2f} {r['avg_test_dd']:>7.1f} {r['consistency']:>8.1f}% {r['overfit_ratio']:>7.2f} {r['num_windows']:>8}{star}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  Part 2: Monte Carlo 模拟
# ═══════════════════════════════════════════════════════════════════════

def monte_carlo_simulation(df, strategy_name, n_simulations=500, dynamic_pos=False):
    """
    蒙特卡洛模拟：随机打乱每日收益率序列
    评估策略收益的统计分布和置信区间
    """
    try:
        result = run_strategy_backtest(df, strategy_name, dynamic_position=dynamic_pos)
    except:
        return None

    eq = pd.Series([e["equity"] for e in result["equity_curve"]])
    daily_returns = eq.pct_change().dropna().values
    actual_return = result["summary"]["total_return"]
    actual_sharpe = result["summary"]["sharpe"]

    sim_returns = []
    sim_sharpes = []
    sim_max_dds = []

    for _ in range(n_simulations):
        # 随机打乱日收益序列
        shuffled = np.random.permutation(daily_returns)
        sim_equity = 100000 * np.cumprod(1 + shuffled)

        total_ret = (sim_equity[-1] - 100000) / 100000 * 100
        sim_returns.append(total_ret)

        if np.std(shuffled) > 0:
            sim_sharpes.append(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252))
        else:
            sim_sharpes.append(0)

        peak = np.maximum.accumulate(sim_equity)
        dd = (sim_equity - peak) / peak * 100
        sim_max_dds.append(np.min(dd))

    # 统计
    p_value = sum(1 for r in sim_returns if r >= actual_return) / n_simulations
    sharpe_pvalue = sum(1 for s in sim_sharpes if s >= actual_sharpe) / n_simulations

    return {
        "actual_return": round(actual_return, 2),
        "actual_sharpe": round(actual_sharpe, 2),
        "actual_dd": round(result["summary"]["max_drawdown"], 2),
        "sim_return_mean": round(np.mean(sim_returns), 2),
        "sim_return_p5": round(np.percentile(sim_returns, 5), 2),
        "sim_return_p95": round(np.percentile(sim_returns, 95), 2),
        "sim_sharpe_mean": round(np.mean(sim_sharpes), 2),
        "sim_dd_mean": round(np.mean(sim_max_dds), 2),
        "sim_dd_p5": round(np.percentile(sim_max_dds, 5), 2),
        "return_p_value": round(p_value, 3),
        "sharpe_p_value": round(sharpe_pvalue, 3),
        "n_simulations": n_simulations,
    }


def run_monte_carlo(datasets, strategies, dynamic_pos=False):
    print(f"\n{'='*90}")
    print(f"  Monte Carlo 蒙特卡洛模拟 (500次打乱)  {'+ 动态仓位' if dynamic_pos else ''}")
    print(f"{'='*90}")

    all_results = {}

    for sname in strategies:
        mc_list = []
        for (symbol, market), df in datasets.items():
            mc = monte_carlo_simulation(df, sname, 500, dynamic_pos)
            if mc:
                mc_list.append((symbol, mc))

        if not mc_list:
            continue

        avg_pvalue = np.mean([mc["return_p_value"] for _, mc in mc_list])
        avg_sharpe_pv = np.mean([mc["sharpe_p_value"] for _, mc in mc_list])
        significant = sum(1 for _, mc in mc_list if mc["return_p_value"] < 0.1) / len(mc_list) * 100

        all_results[sname] = {
            "avg_return_pvalue": round(avg_pvalue, 3),
            "avg_sharpe_pvalue": round(avg_sharpe_pv, 3),
            "significant_pct": round(significant, 1),
            "details": mc_list,
        }

    print(f"\n  {'Strategy':<20} {'AvgRetP':>8} {'AvgShpP':>8} {'Signif%':>8} {'Verdict':>12}")
    print(f"  {'-'*58}")

    for sname, r in sorted(all_results.items(), key=lambda x: x[1]["significant_pct"], reverse=True):
        if r["avg_return_pvalue"] < 0.1:
            verdict = "OK 显著"
        elif r["avg_return_pvalue"] < 0.2:
            verdict = "~ 边缘"
        else:
            verdict = "FAIL 不显著"

        print(f"  {sname:<20} {r['avg_return_pvalue']:>8.3f} {r['avg_sharpe_pvalue']:>8.3f} {r['significant_pct']:>7.1f}% {verdict:>12}")

    # 打印各股票详情
    print(f"\n  {'Strategy':<18} {'Symbol':<8} {'ActRet%':>8} {'SimMeanRet%':>12} {'SimP5%':>7} {'SimP95%':>8} {'P-val':>6} {'DD':>7}")
    print(f"  {'-'*80}")
    for sname in strategies:
        if sname not in all_results:
            continue
        for symbol, mc in all_results[sname]["details"]:
            sig = "*" if mc["return_p_value"] < 0.1 else ""
            print(f"  {sname:<18} {symbol:<8} {mc['actual_return']:>8.1f} {mc['sim_return_mean']:>12.1f} {mc['sim_return_p5']:>7.1f} {mc['sim_return_p95']:>8.1f} {mc['return_p_value']:>6.3f} {mc['actual_dd']:>7.1f}{sig}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  Part 3: 组合配置优化
# ═══════════════════════════════════════════════════════════════════════

def portfolio_optimization(datasets, strategy_name, dynamic_pos=False):
    """等权 vs Sharpe加权 vs MinVar 组合"""
    print(f"\n{'='*90}")
    print(f"  组合配置优化 - {strategy_name} {'+ 动态仓位' if dynamic_pos else ''}")
    print(f"{'='*90}")

    # 收集每只股票的日收益率序列
    returns_dict = {}
    sharpes = {}
    for (symbol, market), df in datasets.items():
        try:
            r = run_strategy_backtest(df, strategy_name, dynamic_position=dynamic_pos)
            eq = pd.Series([e["equity"] for e in r["equity_curve"]])
            daily_ret = eq.pct_change().dropna()
            returns_dict[symbol] = daily_ret.values
            sharpes[symbol] = r["summary"]["sharpe"]
        except:
            pass

    if len(returns_dict) < 3:
        print("  股票数不足，跳过")
        return

    symbols = list(returns_dict.keys())
    n = len(symbols)

    # 对齐长度
    min_len = min(len(v) for v in returns_dict.values())
    ret_matrix = np.array([returns_dict[s][:min_len] for s in symbols])

    # 相关性矩阵
    corr = np.corrcoef(ret_matrix)
    print(f"\n  相关性矩阵 ({n} stocks):")
    print(f"  {'':>8}", end="")
    for s in symbols:
        print(f"{s:>8}", end="")
    print()
    for i, s in enumerate(symbols):
        print(f"  {s:>8}", end="")
        for j in range(n):
            print(f"{corr[i][j]:>8.2f}", end="")
        print()

    avg_corr = (corr.sum() - n) / (n * (n - 1))
    print(f"\n  平均相关性: {avg_corr:.3f}")

    # 几种配置方案
    configs = {}

    # 1. 等权
    w_equal = np.ones(n) / n
    port_ret_equal = ret_matrix.T @ w_equal
    eq_sharpe = np.mean(port_ret_equal) / np.std(port_ret_equal) * np.sqrt(252) if np.std(port_ret_equal) > 0 else 0
    eq_return = (np.cumprod(1 + port_ret_equal)[-1] - 1) * 100
    eq_dd = np.min((np.cumprod(1 + port_ret_equal) - np.maximum.accumulate(np.cumprod(1 + port_ret_equal))) / np.maximum.accumulate(np.cumprod(1 + port_ret_equal)) * 100)
    configs["等权 (Equal)"] = {"sharpe": eq_sharpe, "return": eq_return, "dd": eq_dd, "weights": w_equal}

    # 2. Sharpe加权
    s_values = np.array([max(sharpes[s], 0.01) for s in symbols])
    w_sharpe = s_values / s_values.sum()
    port_ret_sharpe = ret_matrix.T @ w_sharpe
    sp_sharpe = np.mean(port_ret_sharpe) / np.std(port_ret_sharpe) * np.sqrt(252) if np.std(port_ret_sharpe) > 0 else 0
    sp_return = (np.cumprod(1 + port_ret_sharpe)[-1] - 1) * 100
    sp_dd = np.min((np.cumprod(1 + port_ret_sharpe) - np.maximum.accumulate(np.cumprod(1 + port_ret_sharpe))) / np.maximum.accumulate(np.cumprod(1 + port_ret_sharpe)) * 100)
    configs["Sharpe加权"] = {"sharpe": sp_sharpe, "return": sp_return, "dd": sp_dd, "weights": w_sharpe}

    # 3. 逆波动率加权
    vols = np.array([np.std(returns_dict[s]) for s in symbols])
    inv_vols = 1.0 / np.maximum(vols, 1e-8)
    w_inv_vol = inv_vols / inv_vols.sum()
    port_ret_iv = ret_matrix.T @ w_inv_vol
    iv_sharpe = np.mean(port_ret_iv) / np.std(port_ret_iv) * np.sqrt(252) if np.std(port_ret_iv) > 0 else 0
    iv_return = (np.cumprod(1 + port_ret_iv)[-1] - 1) * 100
    iv_dd = np.min((np.cumprod(1 + port_ret_iv) - np.maximum.accumulate(np.cumprod(1 + port_ret_iv))) / np.maximum.accumulate(np.cumprod(1 + port_ret_iv)) * 100)
    configs["逆波动率"] = {"sharpe": iv_sharpe, "return": iv_return, "dd": iv_dd, "weights": w_inv_vol}

    # 4. 随机搜索最优权重 (简化版)
    best_sharpe = -999
    best_w = w_equal.copy()
    for _ in range(10000):
        w = np.random.dirichlet(np.ones(n))
        port_ret = ret_matrix.T @ w
        s = np.mean(port_ret) / np.std(port_ret) * np.sqrt(252) if np.std(port_ret) > 0 else 0
        if s > best_sharpe:
            best_sharpe = s
            best_w = w
    port_ret_opt = ret_matrix.T @ best_w
    opt_return = (np.cumprod(1 + port_ret_opt)[-1] - 1) * 100
    opt_dd = np.min((np.cumprod(1 + port_ret_opt) - np.maximum.accumulate(np.cumprod(1 + port_ret_opt))) / np.maximum.accumulate(np.cumprod(1 + port_ret_opt)) * 100)
    configs["最优搜索(10K)"] = {"sharpe": best_sharpe, "return": opt_return, "dd": opt_dd, "weights": best_w}

    # 打印结果
    print(f"\n  {'配置方案':<18} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8}  权重分配")
    print(f"  {'-'*80}")
    for name, c in configs.items():
        w_str = " ".join(f"{symbols[i]}:{c['weights'][i]:.1%}" for i in range(n))
        star = " *" if c["sharpe"] == max(cc["sharpe"] for cc in configs.values()) else ""
        print(f"  {name:<18} {c['sharpe']:>8.3f} {c['return']:>9.1f} {c['dd']:>8.1f}  {w_str}{star}")

    return configs


def main():
    print("=" * 90)
    print("  量化策略深度验证 - Walk-Forward + Monte Carlo + 组合优化")
    print("=" * 90)

    print("\n加载 5y 数据...")
    datasets = load_data("5y")
    print(f"  {len(datasets)} 只股票加载完成")

    # ── Part 1: Walk-Forward 验证 ──
    print("\n\n" + "#" * 90)
    print("  PART 1: Walk-Forward 验证")
    print("#" * 90)
    wf_base = run_walk_forward(datasets, TOP_STRATEGIES, dynamic_pos=False)
    wf_dyn = run_walk_forward(datasets, TOP_STRATEGIES, dynamic_pos=True)

    # 对比
    print(f"\n  动态仓位 vs 固定仓位 Walk-Forward 对比:")
    print(f"  {'Strategy':<20} {'Base Consist%':>13} {'Dyn Consist%':>13} {'Base Sharpe':>12} {'Dyn Sharpe':>11}")
    print(f"  {'-'*70}")
    for sname in TOP_STRATEGIES:
        b = wf_base.get(sname, {})
        d = wf_dyn.get(sname, {})
        bc = b.get("consistency", 0)
        dc = d.get("consistency", 0)
        bs = b.get("avg_test_sharpe", 0)
        ds = d.get("avg_test_sharpe", 0)
        better = " UP" if dc > bc else ""
        print(f"  {sname:<20} {bc:>12.1f}% {dc:>12.1f}% {bs:>12.3f} {ds:>11.3f}{better}")

    # ── Part 2: Monte Carlo ──
    print("\n\n" + "#" * 90)
    print("  PART 2: Monte Carlo 蒙特卡洛模拟")
    print("#" * 90)
    mc_base = run_monte_carlo(datasets, TOP_STRATEGIES[:4], dynamic_pos=False)
    mc_dyn = run_monte_carlo(datasets, TOP_STRATEGIES[:4], dynamic_pos=True)

    # ── Part 3: 组合优化 ──
    print("\n\n" + "#" * 90)
    print("  PART 3: 多股票组合配置优化")
    print("#" * 90)
    for sname in ["claude_rsi_macd_trend", "claude_ensemble_voting", "claude_momentum_breakout"]:
        portfolio_optimization(datasets, sname, dynamic_pos=False)
        portfolio_optimization(datasets, sname, dynamic_pos=True)

    print("\n\n" + "=" * 90)
    print("  全部验证完成！")
    print("=" * 90)


if __name__ == "__main__":
    main()
