"""
高级优化实验：
1. 测试动态仓位管理（ATR自适应仓位）
2. 测试趋势过滤器（200日均线过滤）
3. 测试回撤控制（最大回撤止损）
4. 测试组合配置（多策略分散投资）
"""
import sys
sys.path.insert(0, ".")
from backtest import get_ohlcv, run_strategy_backtest, run_backtest
from strategies import get_strategy
import numpy as np
import pandas as pd
import ta

TARGETS = [
    ("AAPL", "US"), ("NVDA", "US"), ("MSFT", "US"), ("TSLA", "US"),
    ("GOOGL", "US"), ("META", "US"), ("AMZN", "US"), ("AMD", "US"),
]


def load_data(period="5y"):
    datasets = {}
    for symbol, market in TARGETS:
        try:
            df = get_ohlcv(symbol, market, interval="1d", period=period)
            datasets[(symbol, market)] = df
        except:
            pass
    return datasets


# ═══════════════════════════════════════════════════════════════════
#  实验1: 200日均线趋势过滤器
#  只在价格在200日均线上方时允许买入，下方时强制清仓
# ═══════════════════════════════════════════════════════════════════

def run_with_trend_filter(df_raw, strategy_name, ma_period=200, params_override=None):
    """在策略信号基础上叠加长期趋势过滤"""
    strategy = get_strategy(strategy_name)
    params = {**strategy.default_params, **(params_override or {})}
    params.setdefault("capital", 100000)
    params.setdefault("commission", 0.001)
    params.setdefault("position_pct", 0.95)

    df = strategy.add_indicators(df_raw, params)
    df = strategy.generate_signals(df, params)

    # 叠加趋势过滤
    df["ma_trend"] = df["close"].rolling(ma_period).mean()
    df = df.dropna()

    # 在下降趋势中禁止买入，强制卖出
    uptrend = df["close"] > df["ma_trend"]
    df["buy_signal"] = df["buy_signal"] & uptrend
    df["sell_signal"] = df["sell_signal"] | (~uptrend)

    result = run_backtest(df, params)
    result["strategy"] = f"{strategy_name}+MA{ma_period}filter"
    return result


# ═══════════════════════════════════════════════════════════════════
#  实验2: ATR自适应仓位
#  波动大时减仓，波动小时加仓
# ═══════════════════════════════════════════════════════════════════

def run_with_dynamic_position(df_raw, strategy_name, params_override=None):
    """ATR自适应仓位管理"""
    strategy = get_strategy(strategy_name)
    params = {**strategy.default_params, **(params_override or {})}
    params.setdefault("capital", 100000)
    params.setdefault("commission", 0.001)

    df = strategy.add_indicators(df_raw, params)
    df = strategy.generate_signals(df, params)

    # ATR百分比 = ATR / Close
    df["atr_pct"] = df["atr"] / df["close"]
    atr_median = df["atr_pct"].rolling(60).median()

    # 仓位 = target_risk / ATR_pct, 限制在 [0.3, 1.0]
    target_risk = 0.02  # 2%目标风险
    df["position_size"] = (target_risk / df["atr_pct"]).clip(0.3, 1.0)

    # 自定义回测
    capital = params["capital"]
    commission = params["commission"]
    atr_stop_mul = params.get("atr_stop_mul", 2.0)

    cash = capital
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    trades = []
    equity_curve = []

    for i, row in df.iterrows():
        price = row["close"]
        equity = cash + position * price
        equity_curve.append({"date": str(i.date()), "equity": round(equity, 2)})

        if position > 0 and price <= stop_price:
            proceeds = position * price * (1 - commission)
            cash += proceeds
            pnl = proceeds - position * entry_price * (1 + commission)
            trades.append({"type": "SELL", "price": round(price, 2), "pnl": round(pnl, 2), "reason": "stop"})
            position = 0

        elif position == 0 and row["buy_signal"]:
            pos_pct = row["position_size"]
            shares = int(cash * pos_pct / (price * (1 + commission)))
            if shares > 0:
                cash -= shares * price * (1 + commission)
                position = shares
                entry_price = price
                stop_price = price - atr_stop_mul * row["atr"]
                trades.append({"type": "BUY", "price": round(price, 2), "pnl": 0, "reason": f"pos={pos_pct:.0%}"})

        elif position > 0 and row["sell_signal"]:
            proceeds = position * price * (1 - commission)
            cash += proceeds
            pnl = proceeds - position * entry_price * (1 + commission)
            trades.append({"type": "SELL", "price": round(price, 2), "pnl": round(pnl, 2), "reason": "signal"})
            position = 0

    if position > 0:
        price = df["close"].iloc[-1]
        proceeds = position * price * (1 - commission)
        cash += proceeds

    final_equity = cash
    total_return = (final_equity - capital) / capital * 100
    eq_series = pd.Series([e["equity"] for e in equity_curve])
    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    sell_trades = [t for t in trades if t["type"] == "SELL"]
    win_trades = [t for t in sell_trades if t["pnl"] > 0]
    daily_returns = eq_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    return {
        "summary": {
            "sharpe": round(float(sharpe), 2),
            "total_return": round(total_return, 2),
            "max_drawdown": round(float(max_drawdown), 2),
            "total_trades": len(sell_trades),
            "win_rate": round(len(win_trades) / len(sell_trades) * 100, 2) if sell_trades else 0,
        },
        "strategy": f"{strategy_name}+dynamic_pos",
    }


# ═══════════════════════════════════════════════════════════════════
#  实验3: 最大回撤控制
#  当组合回撤超过阈值时暂停交易N天
# ═══════════════════════════════════════════════════════════════════

def run_with_drawdown_control(df_raw, strategy_name, max_dd_threshold=-15, cooldown_days=20, params_override=None):
    """最大回撤控制：回撤超阈值暂停交易"""
    strategy = get_strategy(strategy_name)
    params = {**strategy.default_params, **(params_override or {})}
    params.setdefault("capital", 100000)
    params.setdefault("commission", 0.001)
    params.setdefault("position_pct", 0.95)

    df = strategy.add_indicators(df_raw, params)
    df = strategy.generate_signals(df, params)

    capital = params["capital"]
    commission = params["commission"]
    atr_stop_mul = params.get("atr_stop_mul", 2.0)

    cash = capital
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    trades = []
    equity_curve = []
    cooldown = 0

    for i, row in df.iterrows():
        price = row["close"]
        equity = cash + position * price
        equity_curve.append({"date": str(i.date()), "equity": round(equity, 2)})

        # 检查回撤
        eq_series = pd.Series([e["equity"] for e in equity_curve])
        peak = eq_series.max()
        current_dd = (equity - peak) / peak * 100

        if current_dd < max_dd_threshold:
            cooldown = cooldown_days
            if position > 0:
                proceeds = position * price * (1 - commission)
                cash += proceeds
                pnl = proceeds - position * entry_price * (1 + commission)
                trades.append({"type": "SELL", "price": round(price, 2), "pnl": round(pnl, 2), "reason": "dd_control"})
                position = 0
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        if position > 0 and price <= stop_price:
            proceeds = position * price * (1 - commission)
            cash += proceeds
            pnl = proceeds - position * entry_price * (1 + commission)
            trades.append({"type": "SELL", "price": round(price, 2), "pnl": round(pnl, 2), "reason": "stop"})
            position = 0

        elif position == 0 and row["buy_signal"]:
            shares = int(cash * params["position_pct"] / (price * (1 + commission)))
            if shares > 0:
                cash -= shares * price * (1 + commission)
                position = shares
                entry_price = price
                stop_price = price - atr_stop_mul * row["atr"]
                trades.append({"type": "BUY", "price": round(price, 2), "pnl": 0})

        elif position > 0 and row["sell_signal"]:
            proceeds = position * price * (1 - commission)
            cash += proceeds
            pnl = proceeds - position * entry_price * (1 + commission)
            trades.append({"type": "SELL", "price": round(price, 2), "pnl": round(pnl, 2), "reason": "signal"})
            position = 0

    if position > 0:
        price = df["close"].iloc[-1]
        proceeds = position * price * (1 - commission)
        cash += proceeds

    final_equity = cash
    total_return = (final_equity - capital) / capital * 100
    eq_series = pd.Series([e["equity"] for e in equity_curve])
    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    sell_trades = [t for t in trades if t["type"] == "SELL"]
    win_trades = [t for t in sell_trades if t["pnl"] > 0]
    daily_returns = eq_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    return {
        "summary": {
            "sharpe": round(float(sharpe), 2),
            "total_return": round(total_return, 2),
            "max_drawdown": round(float(max_drawdown), 2),
            "total_trades": len(sell_trades),
            "win_rate": round(len(win_trades) / len(sell_trades) * 100, 2) if sell_trades else 0,
        },
        "strategy": f"{strategy_name}+dd_control",
    }


def main():
    print("加载 5y 数据...")
    datasets = load_data("5y")
    print(f"  {len(datasets)} 只股票加载完成\n")

    top_strategies = ["claude_ensemble_voting", "claude_rsi_macd_trend", "claude_momentum_breakout", "claude_dual_ma_cross"]

    experiments = [
        ("baseline", lambda df, s: run_strategy_backtest(df, s)),
        ("MA200_filter", lambda df, s: run_with_trend_filter(df, s, 200)),
        ("MA100_filter", lambda df, s: run_with_trend_filter(df, s, 100)),
        ("dynamic_pos", lambda df, s: run_with_dynamic_position(df, s)),
        ("dd_control_15", lambda df, s: run_with_drawdown_control(df, s, -15, 20)),
        ("dd_control_20", lambda df, s: run_with_drawdown_control(df, s, -20, 15)),
    ]

    for strategy_name in top_strategies:
        print(f"\n{'='*80}")
        print(f"  策略: {strategy_name}")
        print(f"{'='*80}")
        print(f"  {'实验':<20} {'AvgSharpe':>10} {'AvgReturn%':>11} {'AvgMaxDD%':>10} {'AvgTrades':>10} {'AvgWR%':>8}")
        print(f"  {'-'*70}")

        for exp_name, exp_func in experiments:
            sharpes, returns, dds, trades_list, wrs = [], [], [], [], []
            for (symbol, market), df in datasets.items():
                try:
                    r = exp_func(df, strategy_name)
                    s = r["summary"]
                    sharpes.append(s["sharpe"])
                    returns.append(s["total_return"])
                    dds.append(s["max_drawdown"])
                    trades_list.append(s["total_trades"])
                    wrs.append(s["win_rate"])
                except Exception as e:
                    sharpes.append(0)
                    returns.append(0)
                    dds.append(-50)

            avg_s = np.mean(sharpes)
            avg_r = np.mean(returns)
            avg_dd = np.mean(dds)
            avg_t = np.mean(trades_list) if trades_list else 0
            avg_wr = np.mean(wrs) if wrs else 0

            marker = " ★" if avg_s > 0.7 else ""
            print(f"  {exp_name:<20} {avg_s:>10.3f} {avg_r:>11.2f} {avg_dd:>10.1f} {avg_t:>10.1f} {avg_wr:>7.1f}%{marker}")

    # 2y 验证
    print(f"\n\n加载 2y 数据进行验证...")
    datasets_2y = load_data("2y")
    print(f"  {len(datasets_2y)} 只股票\n")

    for strategy_name in top_strategies:
        print(f"\n  {strategy_name} (2y):")
        print(f"  {'实验':<20} {'AvgSharpe':>10} {'AvgReturn%':>11} {'AvgMaxDD%':>10}")
        print(f"  {'-'*55}")

        for exp_name, exp_func in experiments:
            sharpes, returns, dds = [], [], []
            for (symbol, market), df in datasets_2y.items():
                try:
                    r = exp_func(df, strategy_name)
                    s = r["summary"]
                    sharpes.append(s["sharpe"])
                    returns.append(s["total_return"])
                    dds.append(s["max_drawdown"])
                except:
                    sharpes.append(0)
                    returns.append(0)
                    dds.append(-50)
            print(f"  {exp_name:<20} {np.mean(sharpes):>10.3f} {np.mean(returns):>11.2f} {np.mean(dds):>10.1f}")


if __name__ == "__main__":
    main()
