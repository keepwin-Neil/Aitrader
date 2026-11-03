#!/usr/bin/env python3
"""新策略贝叶斯优化 — 步骤3: 在NASDAQ子集上优化通用参数"""
import sys, json, os, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from backtest import get_ohlcv, run_strategy_backtest
from strategies import get_strategy

# 优化参数范围
PARAM_RANGES = {
    "claude_hurst_exponent": {
        "hurst_window":   (50, 150),
        "hurst_trend":    (0.50, 0.65),
        "hurst_mr":       (0.35, 0.50),
        "ema_fast":       (5, 20),
        "ema_slow":       (20, 60),
        "bb_period":      (15, 30),
        "bb_std":         (1.5, 3.0),
        "rsi_period":     (10, 20),
        "rsi_oversold":   (20, 40),
        "rsi_overbought": (60, 85),
        "atr_stop_mul":   (1.5, 4.0),
    },
    "claude_entropy_momentum": {
        "entropy_window":  (10, 40),
        "entropy_bins":    (5, 20),
        "entropy_smooth":  (3, 10),
        "entropy_low":     (1.5, 3.5),
        "entropy_high":    (2.5, 4.0),
        "mom_period":      (5, 20),
        "mom_exit":        (1.0, 8.0),
        "ema_period":      (15, 50),
        "rsi_period":      (10, 20),
        "rsi_sell":        (70, 90),
        "atr_stop_mul":    (1.5, 4.0),
    },
    "claude_kaufman_efficiency": {
        "er_period":      (5, 25),
        "er_threshold":   (0.15, 0.55),
        "er_exit":        (0.05, 0.25),
        "kama_fast":      (2, 5),
        "kama_slow":      (20, 50),
        "atr_period":     (10, 25),
        "atr_mul":        (1.0, 3.0),
        "vol_period":     (10, 30),
        "vol_min":        (0.5, 1.5),
        "rsi_period":     (10, 20),
        "rsi_sell":       (70, 90),
        "atr_stop_mul":   (1.5, 4.0),
    },
}

INT_PARAMS = {
    "hurst_window", "ema_fast", "ema_slow", "bb_period", "rsi_period",
    "entropy_window", "entropy_bins", "entropy_smooth", "mom_period",
    "er_period", "kama_fast", "kama_slow", "atr_period", "vol_period",
}


def optimize_strategy(strategy_name, symbols, market, n_trials=80):
    """Bayesian optimization of strategy params on given stock set"""
    ranges = PARAM_RANGES[strategy_name]
    strat = get_strategy(strategy_name)

    # Pre-load data
    data_cache = {}
    for sym in symbols:
        try:
            df = get_ohlcv(sym, market, interval="1d", period="1y")
            if df is not None and len(df) >= 50:
                data_cache[sym] = df
        except:
            pass
    print(f"  数据加载完成: {len(data_cache)}/{len(symbols)} stocks")

    def objective(trial):
        params = dict(strat.default_params)  # start from defaults
        for k, (lo, hi) in ranges.items():
            if k in INT_PARAMS:
                params[k] = trial.suggest_int(k, int(lo), int(hi))
            else:
                params[k] = trial.suggest_float(k, lo, hi)

        # Ensure constraints
        if "ema_fast" in params and "ema_slow" in params:
            if params["ema_fast"] >= params["ema_slow"]:
                params["ema_slow"] = params["ema_fast"] + 10
        if "entropy_low" in params and "entropy_high" in params:
            if params["entropy_low"] >= params["entropy_high"]:
                params["entropy_high"] = params["entropy_low"] + 0.5
        if "er_threshold" in params and "er_exit" in params:
            if params["er_exit"] >= params["er_threshold"]:
                params["er_exit"] = params["er_threshold"] - 0.1

        returns = []
        for sym, df in data_cache.items():
            try:
                res = run_strategy_backtest(df, strategy_name,
                                            params_override=params, capital=100000)
                returns.append(res["summary"]["total_return"])
            except:
                returns.append(0)

        avg_ret = np.mean(returns) if returns else -999
        return avg_ret

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    # Apply constraints to best params too
    if "ema_fast" in best and "ema_slow" in best:
        if best["ema_fast"] >= best["ema_slow"]:
            best["ema_slow"] = best["ema_fast"] + 10
    if "entropy_low" in best and "entropy_high" in best:
        if best["entropy_low"] >= best["entropy_high"]:
            best["entropy_high"] = best["entropy_low"] + 0.5
    if "er_threshold" in best and "er_exit" in best:
        if best["er_exit"] >= best["er_threshold"]:
            best["er_exit"] = best["er_threshold"] - 0.1

    # Fill non-optimized params from defaults
    full_params = dict(strat.default_params)
    full_params.update(best)

    return {
        "best_params": full_params,
        "best_return": round(study.best_value, 2),
        "n_trials": n_trials,
    }


if __name__ == "__main__":
    with open("stock_pools.json") as f:
        pools = json.load(f)

    nasdaq = pools["NASDAQ 100"]
    # Use 30 representative stocks for optimization
    random.seed(42)
    opt_symbols = random.sample(nasdaq["symbols"], min(30, len(nasdaq["symbols"])))
    print(f"优化子集: {len(opt_symbols)} stocks from NASDAQ 100")

    strategies = ["claude_hurst_exponent", "claude_entropy_momentum", "claude_kaufman_efficiency"]
    results = {}

    for sn in strategies:
        print(f"\n{'='*50}")
        print(f"贝叶斯优化: {sn}")
        print(f"{'='*50}")
        res = optimize_strategy(sn, opt_symbols, "US", n_trials=80)
        results[sn] = res
        print(f"✅ {sn} 最优平均收益: {res['best_return']:.2f}%")
        print(f"   最优参数: {json.dumps(res['best_params'], indent=2)}")

    with open("/tmp/optimize_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n优化结果已保存: /tmp/optimize_results.json")
