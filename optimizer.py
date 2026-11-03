"""
Multi-method parameter optimizer for strategy backtesting.
Methods: grid (exhaustive), bayesian (Optuna), coarse_fine (two-stage).
All methods support multiprocessing parallelization.
"""
import logging
import time
from itertools import product as iter_product
from concurrent.futures import ProcessPoolExecutor, as_completed
import proc_role


def _opt_map(worker, tasks, max_workers, chunksize=None):
    """并行跑 tasks; 但如果本进程已经是别人的 pool worker, 就老老实实串行。

    为什么要这个守卫: optimize_strategy 会被放进上层的 ProcessPool 里跑
    (比如 /walkforward 的 retrain 分支、批量优化), 这时候再往下建一层池就是
    嵌套 —— 实测 12 核机器上瞬间起到 336 个进程, 直接打死。
    proc_role 的标记由上层池的 initializer 打上, 这里只管查。
    """
    if proc_role.is_pool_worker() or max_workers <= 1 or len(tasks) <= 1:
        return [worker(t) for t in tasks]
    kw = {}
    if chunksize:
        kw["chunksize"] = chunksize
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=proc_role.mark_pool_worker) as ex:
        return list(ex.map(worker, tasks, **kw))


from functools import partial

import optuna
# Suppress Optuna logs (very verbose by default)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Parameter range definitions for Bayesian/Coarse-Fine search
#  format: {param_name: (low, high, step, type)}
#  type: "int", "float", "categorical"
# ──────────────────────────────────────────────────────────────

PARAM_RANGES = {
    "claude_multi_indicator": {
        "ema_fast": (5, 20, 1, "int"),
        "ema_slow": (30, 100, 5, "int"),
        "rsi_buy_min": (30, 60, 1, "int"),
        "rsi_sell": (70, 90, 1, "int"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_dual_ma_cross": {
        "fast_period": (3, 20, 1, "int"),
        "slow_period": (15, 60, 1, "int"),
        "ma_type": (["sma", "ema"], None, None, "categorical"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_momentum_breakout": {
        "breakout_period": (5, 45, 1, "int"),
        "exit_period": (3, 20, 1, "int"),
        "vol_ratio_min": (0.2, 2.0, 0.1, "float"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_mean_reversion": {
        "rsi_period": (5, 25, 1, "int"),
        "rsi_oversold": (15, 45, 1, "int"),
        "rsi_overbought": (55, 85, 1, "int"),
        "bb_std": (1.0, 3.5, 0.25, "float"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_turtle_trading": {
        "entry_period": (5, 50, 1, "int"),
        "exit_period": (3, 20, 1, "int"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_macd_divergence": {
        "macd_fast": (5, 20, 1, "int"),
        "macd_slow": (15, 45, 1, "int"),
        "lookback": (10, 60, 1, "int"),
        "rsi_sell": (65, 95, 1, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_kdj_dmi": {
        "kdj_period": (5, 30, 1, "int"),
        "kdj_oversold": (5, 35, 1, "int"),
        "kdj_overbought": (65, 95, 1, "int"),
        "adx_threshold": (10, 35, 1, "int"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_vwap_breakout": {
        "vwap_period": (5, 40, 1, "int"),
        "atr_band_mul": (0.2, 3.0, 0.1, "float"),
        "vol_ratio_min": (0.3, 2.0, 0.1, "float"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_adaptive_ma": {
        "er_period": (5, 35, 1, "int"),
        "fast_sc": (2, 8, 1, "int"),
        "slow_sc": (15, 60, 5, "int"),
        "threshold": (0.0003, 0.01, 0.0005, "float"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_rsi_macd_trend": {
        "rsi_entry": (15, 50, 1, "int"),
        "rsi_exit": (60, 95, 1, "int"),
        "ema_trend": (15, 80, 1, "int"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_ensemble_voting": {
        "buy_threshold": (1, 3, 1, "int"),
        "sell_threshold": (1, 3, 1, "int"),
        "mr_rsi_oversold": (15, 45, 1, "int"),
        "mr_rsi_overbought": (55, 85, 1, "int"),
        "dm_fast": (3, 18, 1, "int"),
        "dm_slow": (15, 50, 1, "int"),
    },
    "claude_ichimoku_cloud": {
        "tenkan_period": (5, 18, 1, "int"),
        "kijun_period": (15, 40, 1, "int"),
        "senkou_b_period": (35, 70, 1, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_volatility_breakout": {
        "k_factor": (0.05, 1.0, 0.05, "float"),
        "ema_period": (15, 100, 5, "int"),
        "min_range": (0.05, 1.0, 0.05, "float"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_elder_triple_screen": {
        "weekly_ema_period": (6, 25, 1, "int"),
        "force_period": (2, 5, 1, "int"),
        "ema_short": (6, 25, 1, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_supertrend": {
        "st_period": (5, 30, 1, "int"),
        "st_multiplier": (1.0, 5.0, 0.5, "float"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_ross_hook": {
        "fractal_n": (2, 6, 1, "int"),
        "ema_period": (10, 70, 5, "int"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    "claude_divergence_combo": {
        "pivot_n": (2, 6, 1, "int"),
        "max_pivot_dist": (25, 100, 5, "int"),
        "div_confirm": (1, 2, 1, "int"),
        "rsi_zone_low": (20, 45, 1, "int"),
        "rsi_zone_high": (55, 80, 1, "int"),
        "atr_stop_mul": (0.5, 4.0, 0.5, "float"),
    },
    # ── 新策略 ──
    "claude_trend_hold": {
        "ema_short": (10, 30, 5, "int"),
        "ema_mid": (30, 80, 10, "int"),
        "ema_long": (100, 250, 50, "int"),
        "rsi_entry_min": (30, 50, 5, "int"),
        "rsi_exit_low": (25, 45, 5, "int"),
        "pullback_atr_mul": (0.5, 3.0, 0.5, "float"),
        "sell_confirm_days": (2, 5, 1, "int"),
        "atr_stop_mul": (2.0, 5.0, 0.5, "float"),
    },
    "claude_momentum_reversion_hybrid": {
        "adx_period": (10, 20, 2, "int"),
        "adx_threshold": (15, 35, 5, "int"),
        "ema_fast": (8, 20, 4, "int"),
        "ema_slow": (30, 65, 5, "int"),
        "rsi_oversold": (25, 40, 5, "int"),
        "rsi_overbought": (65, 85, 5, "int"),
        "bb_period": (15, 30, 5, "int"),
        "bb_std": (1.5, 2.5, 0.5, "float"),
        "atr_stop_mul": (2.0, 4.0, 0.5, "float"),
    },
    "claude_breakout_trailing": {
        "breakout_period": (10, 40, 5, "int"),
        "trail_atr_mul": (2.0, 5.0, 0.5, "float"),
        "ema_period": (30, 80, 10, "int"),
        "vol_ratio_min": (0.5, 1.5, 0.25, "float"),
        "atr_stop_mul": (2.0, 5.0, 0.5, "float"),
    },
    "claude_multi_timeframe_trend": {
        "weekly_ema_period": (40, 100, 10, "int"),
        "daily_ema": (10, 30, 5, "int"),
        "daily_ema_fast": (5, 12, 1, "int"),
        "pullback_tolerance": (0.01, 0.04, 0.005, "float"),
        "atr_stop_mul": (2.5, 5.0, 0.5, "float"),
    },
    "claude_adaptive_regime": {
        "adx_period": (10, 20, 2, "int"),
        "adx_trend_threshold": (15, 30, 5, "int"),
        "ema_long": (50, 150, 25, "int"),
        "rsi_oversold": (25, 40, 5, "int"),
        "rsi_overbought": (60, 80, 5, "int"),
        "bull_rsi_floor": (20, 40, 5, "int"),
        "bb_period": (15, 30, 5, "int"),
        "bb_std": (1.5, 2.5, 0.5, "float"),
        "atr_stop_mul": (2.5, 5.0, 0.5, "float"),
    },
    "claude_obv_momentum": {
        "obv_ema_period": (10, 40, 5, "int"),
        "ema_trend_period": (30, 80, 10, "int"),
        "obv_slope_len": (3, 10, 1, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_overbought": (65, 85, 5, "int"),
        "rsi_oversold": (20, 40, 5, "int"),
        "volume_ratio_min": (0.8, 2.0, 0.2, "float"),
    },
    "claude_keltner_breakout": {
        "kelt_period": (15, 35, 5, "int"),
        "kelt_atr_mul": (1.0, 3.0, 0.5, "float"),
        "atr_period": (10, 20, 2, "int"),
        "ema_trend_period": (30, 80, 10, "int"),
        "atr_filter_period": (15, 30, 5, "int"),
        "stop_atr_mul": (1.5, 3.5, 0.5, "float"),
        "trail_atr_mul": (2.0, 4.0, 0.5, "float"),
    },
    "claude_zscore_reversion": {
        "lookback": (20, 60, 5, "int"),
        "entry_z": (1.0, 2.5, 0.25, "float"),
        "exit_z": (0.0, 0.5, 0.1, "float"),
        "stop_z": (2.5, 4.0, 0.5, "float"),
        "ema_filter_period": (60, 150, 10, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_floor": (15, 35, 5, "int"),
    },
    "claude_donchian_breakout": {
        "dc_period": (10, 50, 5, "int"),
        "adx_period": (10, 20, 2, "int"),
        "adx_min": (15, 35, 5, "int"),
        "vol_ratio_min": (0.5, 2.0, 0.25, "float"),
        "trail_atr_mul": (1.5, 4.0, 0.5, "float"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_williams_r_adx": {
        "wr_period": (7, 21, 1, "int"),
        "wr_oversold": (-90, -60, 5, "int"),
        "wr_overbought": (-40, -10, 5, "int"),
        "adx_period": (10, 20, 2, "int"),
        "adx_threshold": (15, 35, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_parabolic_sar": {
        "psar_step": (0.01, 0.05, 0.005, "float"),
        "psar_max": (0.1, 0.4, 0.05, "float"),
        "macd_fast": (8, 16, 2, "int"),
        "macd_slow": (20, 35, 3, "int"),
        "macd_signal": (7, 12, 1, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_overbought": (70, 90, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_heikin_ashi": {
        "ema_period": (30, 100, 10, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_min": (25, 50, 5, "int"),
        "rsi_overbought": (65, 90, 5, "int"),
        "consec_bull": (1, 4, 1, "int"),
        "consec_bear": (1, 4, 1, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_stochastic_momentum": {
        "smi_period": (8, 21, 1, "int"),
        "smooth_r": (3, 8, 1, "int"),
        "smooth_s": (2, 5, 1, "int"),
        "signal_period": (5, 15, 2, "int"),
        "ema_period": (30, 80, 10, "int"),
        "oversold": (-60, -20, 5, "int"),
        "overbought": (20, 60, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_chaikin_money_flow": {
        "cmf_period": (10, 30, 5, "int"),
        "ema_fast": (10, 30, 5, "int"),
        "ema_slow": (30, 80, 10, "int"),
        "cmf_buy_threshold": (0.02, 0.15, 0.01, "float"),
        "cmf_sell_threshold": (-0.15, -0.02, 0.01, "float"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_elder_force": {
        "short_period": (2, 5, 1, "int"),
        "long_period": (8, 20, 2, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_trix_momentum": {
        "trix_period": (8, 25, 1, "int"),
        "signal_period": (5, 15, 2, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_connors_rsi": {
        "rsi_period": (2, 5, 1, "int"),
        "streak_rsi_period": (2, 5, 1, "int"),
        "pct_rank_period": (50, 200, 25, "int"),
        "ema_period": (30, 80, 10, "int"),
        "oversold": (5, 25, 5, "int"),
        "overbought": (75, 95, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_coppock_curve": {
        "long_roc": (10, 20, 2, "int"),
        "short_roc": (8, 15, 1, "int"),
        "wma_period": (5, 15, 2, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (2.0, 5.0, 0.5, "float"),
    },
    "claude_mass_index": {
        "ema_period": (30, 80, 10, "int"),
        "bulge_threshold": (25, 29, 0.5, "float"),
        "trigger_threshold": (24, 27, 0.5, "float"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_aroon_oscillator": {
        "aroon_period": (15, 35, 5, "int"),
        "buy_threshold": (30, 70, 10, "int"),
        "sell_threshold": (-70, -30, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_squeeze_momentum": {
        "bb_period": (15, 30, 5, "int"),
        "bb_mult": (1.5, 2.5, 0.5, "float"),
        "kc_period": (15, 30, 5, "int"),
        "kc_mult": (1.0, 2.0, 0.5, "float"),
        "mom_period": (10, 25, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_acc_dist_trend": {
        "ad_ema_fast": (5, 15, 2, "int"),
        "ad_ema_slow": (20, 50, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_ehlers_fisher": {
        "fisher_period": (5, 20, 1, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_vol_weighted_mom": {
        "vwm_period": (8, 25, 1, "int"),
        "signal_period": (5, 15, 2, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_price_channel_vol": {
        "channel_period": (10, 30, 5, "int"),
        "exit_period": (5, 15, 5, "int"),
        "vol_ma_period": (10, 30, 5, "int"),
        "vol_threshold": (1.2, 2.5, 0.1, "float"),
        "atr_stop_mul": (2.0, 5.0, 0.5, "float"),
    },
    "claude_meanrev_bb_rsi": {
        "bb_period": (15, 30, 5, "int"),
        "bb_std": (1.5, 2.5, 0.5, "float"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (20, 35, 5, "int"),
        "rsi_overbought": (65, 85, 5, "int"),
        "adx_period": (10, 20, 2, "int"),
        "adx_max": (20, 35, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_dual_thrust": {
        "lookback": (2, 6, 1, "int"),
        "k1": (0.3, 0.8, 0.05, "float"),
        "k2": (0.3, 0.8, 0.05, "float"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_rsi_divergence": {
        "rsi_period": (10, 20, 2, "int"),
        "lookback": (8, 20, 2, "int"),
        "ema_period": (30, 80, 10, "int"),
        "rsi_buy_zone": (30, 45, 5, "int"),
        "rsi_sell_zone": (55, 70, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_ma_ribbon": {
        "ema_1": (5, 10, 1, "int"),
        "ema_2": (10, 16, 1, "int"),
        "ema_3": (18, 26, 2, "int"),
        "ema_4": (28, 40, 2, "int"),
        "ema_5": (45, 65, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_vwap_mean_reversion": {
        "std_period": (10, 30, 5, "int"),
        "std_mult": (1.5, 3.0, 0.5, "float"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (25, 40, 5, "int"),
        "rsi_overbought": (60, 80, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_trend_strength": {
        "adx_period": (10, 20, 2, "int"),
        "adx_threshold": (15, 30, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_gap_and_go": {
        "gap_threshold": (1.0, 4.0, 0.5, "float"),
        "vol_ma_period": (10, 30, 5, "int"),
        "vol_threshold": (1.2, 2.5, 0.1, "float"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_triple_ema_cross": {
        "fast_period": (5, 12, 1, "int"),
        "mid_period": (15, 30, 3, "int"),
        "slow_period": (40, 70, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_momentum_accel": {
        "roc_period": (8, 20, 2, "int"),
        "accel_period": (3, 10, 1, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_pivot_bounce": {
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (25, 40, 5, "int"),
        "rsi_overbought": (60, 80, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_chandelier_exit": {
        "lookback": (15, 30, 5, "int"),
        "atr_mult": (2.0, 4.0, 0.5, "float"),
        "atr_period": (14, 28, 7, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_bb_squeeze_breakout": {
        "bb_period": (15, 30, 5, "int"),
        "bb_std": (1.5, 2.5, 0.5, "float"),
        "width_lookback": (60, 180, 30, "int"),
        "width_percentile": (10, 30, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_volume_climax": {
        "vol_ma_period": (10, 30, 5, "int"),
        "vol_spike": (2.0, 4.0, 0.5, "float"),
        "body_pct": (0.01, 0.04, 0.005, "float"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (25, 40, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_adaptive_bollinger": {
        "bb_period": (15, 30, 5, "int"),
        "base_mult": (1.5, 3.0, 0.5, "float"),
        "atr_period": (10, 20, 2, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (20, 35, 5, "int"),
        "rsi_overbought": (65, 85, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_pullback_buy": {
        "ema_fast": (15, 30, 5, "int"),
        "ema_slow": (40, 70, 10, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_pullback_low": (30, 45, 5, "int"),
        "rsi_pullback_high": (45, 60, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_double_bottom_rsi": {
        "rsi_period": (10, 20, 2, "int"),
        "rsi_low": (25, 40, 5, "int"),
        "lookback": (10, 30, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_atr_channel_trend": {
        "ema_period": (15, 30, 5, "int"),
        "atr_period": (10, 20, 2, "int"),
        "atr_mult": (1.5, 3.5, 0.5, "float"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_stoch_rsi": {
        "rsi_period": (10, 20, 2, "int"),
        "stoch_period": (10, 20, 2, "int"),
        "k_smooth": (2, 5, 1, "int"),
        "d_smooth": (2, 5, 1, "int"),
        "oversold": (10, 30, 5, "int"),
        "overbought": (70, 90, 5, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_combo_meanrev_plus": {
        "bb_period": (15, 30, 5, "int"),
        "bb_std": (1.5, 2.5, 0.5, "float"),
        "rsi_period": (10, 20, 2, "int"),
        "stoch_period": (10, 20, 2, "int"),
        "rsi_oversold": (20, 35, 5, "int"),
        "rsi_overbought": (65, 85, 5, "int"),
        "stoch_oversold": (10, 30, 5, "int"),
        "adx_max": (20, 35, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_range_expansion": {
        "atr_period": (10, 20, 2, "int"),
        "atr_ma_period": (30, 60, 10, "int"),
        "contraction": (0.5, 0.8, 0.1, "float"),
        "expansion": (1.1, 1.8, 0.1, "float"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_multi_osc_consensus": {
        "rsi_period": (10, 20, 2, "int"),
        "cci_period": (14, 30, 2, "int"),
        "stoch_period": (10, 20, 2, "int"),
        "rsi_oversold": (20, 35, 5, "int"),
        "rsi_overbought": (65, 85, 5, "int"),
        "cci_oversold": (-150, -80, 10, "int"),
        "cci_overbought": (80, 150, 10, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_adaptive_rsi_meanrev": {
        "rsi_period": (10, 20, 2, "int"),
        "base_oversold": (25, 35, 5, "int"),
        "base_overbought": (65, 80, 5, "int"),
        "atr_period": (10, 20, 2, "int"),
        "atr_ma_period": (30, 60, 10, "int"),
        "ema_period": (30, 80, 10, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_keltner_meanrev": {
        "kc_period": (15, 30, 5, "int"),
        "kc_mult": (1.5, 3.0, 0.5, "float"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (25, 40, 5, "int"),
        "rsi_overbought": (60, 80, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_volume_profile": {
        "lookback": (30, 90, 15, "int"),
        "bins": (10, 30, 5, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (30, 45, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    "claude_swing_failure": {
        "swing_lookback": (5, 20, 5, "int"),
        "rsi_period": (10, 20, 2, "int"),
        "rsi_oversold": (30, 45, 5, "int"),
        "atr_stop_mul": (1.5, 3.5, 0.5, "float"),
    },
    # ── 新增策略 (2026-03-19) ──
    "claude_momentum_quality": {
        "mom_period": (5, 25, 1, "int"),
        "quality_days": (3, 10, 1, "int"),
        "quality_threshold": (2, 6, 1, "int"),
        "ema_trend": (20, 80, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_adaptive_ensemble_v2": {
        "rolling_window": (20, 80, 10, "int"),
        "min_weight": (0.01, 0.15, 0.02, "float"),
        "buy_threshold": (0.2, 0.6, 0.05, "float"),
        "sell_threshold": (0.3, 0.7, 0.05, "float"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "claude_volatility_regime": {
        "vol_period": (10, 30, 5, "int"),
        "vol_lookback": (30, 100, 10, "int"),
        "high_vol_pct": (60, 85, 5, "int"),
        "low_vol_pct": (15, 40, 5, "int"),
        "ema_fast": (5, 15, 1, "int"),
        "ema_slow": (20, 50, 5, "int"),
        "rsi_oversold": (20, 40, 5, "int"),
        "atr_stop_mul": (1.5, 4.0, 0.5, "float"),
    },
    "candlepattern_signal": {
        "ema_period": (20, 100, 5, "int"),
        "body_ratio": (0.2, 0.5, 0.05, "float"),
        "shadow_ratio": (1.2, 3.0, 0.2, "float"),
        "vol_avg": (10, 30, 5, "int"),
        "vol_confirm": (0.6, 1.5, 0.1, "float"),
        "rsi_buy_max": (60, 85, 5, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_fft_cycle_trend": {
        "fft_window": (20, 60, 5, "int"),
        "min_period": (2, 8, 1, "int"),
        "max_period": (8, 30, 2, "int"),
        "ema_period": (15, 60, 5, "int"),
        "buy_threshold": (-0.8, 0.1, 0.1, "float"),
        "sell_threshold": (0.2, 1.2, 0.1, "float"),
        "atr_stop_mul": (1.5, 5.0, 0.5, "float"),
    },
    "ou_mean_reversion": {
        "window": (30, 90, 5, "int"),
        "entry_z": (0.8, 2.5, 0.1, "float"),
        "exit_z": (-0.5, 0.5, 0.1, "float"),
        "min_halflife": (1, 8, 1, "int"),
        "max_halflife": (10, 50, 5, "int"),
        "adx_max": (15, 40, 5, "int"),
        "ema_period": (50, 150, 10, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "hurst_regime": {
        "hurst_window": (40, 100, 10, "int"),
        "hurst_trend": (0.52, 0.72, 0.02, "float"),
        "hurst_mr": (0.28, 0.48, 0.02, "float"),
        "ema_period": (10, 40, 5, "int"),
        "rsi_period": (7, 21, 1, "int"),
        "rsi_oversold": (25, 45, 5, "int"),
        "rsi_overbought": (55, 80, 5, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_linreg_trend": {
        "linreg_period": (15, 50, 5, "int"),
        "slope_min": (0.0, 0.15, 0.01, "float"),
        "r2_min": (0.3, 0.8, 0.05, "float"),
        "band_std": (1.0, 3.0, 0.25, "float"),
        "ema_period": (20, 80, 5, "int"),
        "rsi_max": (60, 85, 5, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_vol_adj_momentum": {
        "mom_period": (10, 40, 5, "int"),
        "vol_period": (10, 40, 5, "int"),
        "signal_ema": (5, 20, 1, "int"),
        "buy_threshold": (0.1, 1.5, 0.1, "float"),
        "sell_threshold": (-0.5, 0.5, 0.1, "float"),
        "ema_period": (20, 80, 5, "int"),
        "atr_stop_mul": (1.0, 4.0, 0.5, "float"),
    },
    "claude_seasonal_calendar": {
        "ema_period":    (20, 80, 5, "int"),
        "rsi_period":    (10, 21, 1, "int"),
        "rsi_sell":      (65, 85, 5, "int"),
        "buy_score":     (0.1, 0.7, 0.05, "float"),
        "sell_score":    (-0.2, 0.3, 0.05, "float"),
        "month_boost":   (0.5, 3.0, 0.25, "float"),
        "dow_boost":     (0.5, 2.0, 0.25, "float"),
        "atr_stop_mul":  (1.5, 4.0, 0.5, "float"),
    },
    "claude_garch_vol_timing": {
        "garch_omega":      (0.00001, 0.001, 0.0001, "float"),
        "garch_alpha":      (0.05, 0.3, 0.05, "float"),
        "garch_beta":       (0.6, 0.95, 0.05, "float"),
        "vol_ma":           (20, 120, 10, "int"),
        "vol_buy_thresh":   (0.6, 1.2, 0.1, "float"),
        "vol_sell_thresh":  (1.0, 2.0, 0.1, "float"),
        "ema_fast":         (5, 20, 5, "int"),
        "ema_slow":         (20, 60, 5, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "claude_candle_pattern_vol": {
        "ema_period":        (20, 80, 5, "int"),
        "vol_period":        (10, 30, 5, "int"),
        "vol_min":           (1.0, 2.5, 0.1, "float"),
        "body_ratio_min":    (0.2, 0.7, 0.05, "float"),
        "hammer_ratio":      (1.5, 4.0, 0.5, "float"),
        "rsi_period":        (10, 21, 1, "int"),
        "rsi_sell":          (65, 85, 5, "int"),
        "atr_stop_mul":      (1.5, 4.0, 0.5, "float"),
    },
    "openai_vortex_trend_pulse": {
        "vi_period":        (8, 20, 1, "int"),
        "ema_period":       (20, 80, 5, "int"),
        "adx_period":       (10, 30, 2, "int"),
        "adx_min":          (8.0, 25.0, 1.0, "float"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_relative_vigor_reversal": {
        "rvi_period":       (8, 24, 1, "int"),
        "ema_period":       (8, 40, 2, "int"),
        "rvi_floor":        (0.0, 0.15, 0.01, "float"),
        "rvi_take_profit":  (0.2, 0.9, 0.05, "float"),
        "atr_stop_mul":     (1.5, 4.5, 0.5, "float"),
    },
    "openai_kst_trend_signal": {
        "roc1":             (5, 15, 1, "int"),
        "roc2":             (8, 20, 1, "int"),
        "roc3":             (12, 30, 2, "int"),
        "roc4":             (20, 45, 5, "int"),
        "window1":          (3, 10, 1, "int"),
        "window2":          (4, 12, 1, "int"),
        "window3":          (4, 12, 1, "int"),
        "window4":          (5, 14, 1, "int"),
        "nsig":             (3, 9, 1, "int"),
        "ema_period":       (30, 100, 5, "int"),
        "atr_stop_mul":     (1.5, 4.5, 0.5, "float"),
    },
    "openai_schaff_cycle_trend": {
        "window_fast":      (10, 30, 1, "int"),
        "window_slow":      (30, 60, 2, "int"),
        "cycle":            (6, 18, 1, "int"),
        "smooth1":          (2, 6, 1, "int"),
        "smooth2":          (2, 6, 1, "int"),
        "ema_period":       (20, 80, 5, "int"),
        "stc_buy":          (15, 40, 1, "int"),
        "stc_sell":         (55, 85, 1, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_ppo_trend_pulse": {
        "window_fast":      (5, 15, 1, "int"),
        "window_slow":      (18, 40, 2, "int"),
        "window_sign":      (5, 15, 1, "int"),
        "ema_period":       (20, 80, 5, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_balance_power_breakout": {
        "bop_smooth":       (3, 10, 1, "int"),
        "breakout_period":  (10, 30, 2, "int"),
        "ema_period":       (15, 60, 5, "int"),
        "bop_min":          (0.05, 0.4, 0.05, "float"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_cci_trend_reclaim": {
        "cci_period":       (10, 30, 1, "int"),
        "ema_period":       (20, 80, 5, "int"),
        "cci_exit":         (-120, -10, 5, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_dpo_trend_reversal": {
        "dpo_period":       (10, 30, 1, "int"),
        "ema_period":       (15, 60, 5, "int"),
        "rsi_period":       (8, 20, 1, "int"),
        "rsi_cap":          (55, 80, 1, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_atr_expansion_breakout": {
        "atr_period":       (10, 20, 1, "int"),
        "atr_ma_period":    (10, 30, 2, "int"),
        "breakout_period":  (10, 25, 1, "int"),
        "ema_period":       (15, 60, 5, "int"),
        "atr_ratio_min":    (0.9, 1.4, 0.05, "float"),
        "atr_ratio_exit":   (0.7, 1.0, 0.05, "float"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_rsi_range_shift": {
        "rsi_period":       (8, 20, 1, "int"),
        "ema_period":       (15, 60, 5, "int"),
        "breakout_period":  (10, 30, 2, "int"),
        "rsi_buy":          (48, 65, 1, "int"),
        "rsi_exit":         (35, 52, 1, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_keltner_midline_resume": {
        "ema_period":       (10, 35, 1, "int"),
        "atr_period":       (10, 20, 1, "int"),
        "kc_mult":          (1.0, 2.5, 0.1, "float"),
        "rsi_period":       (8, 20, 1, "int"),
        "rsi_floor":        (45, 60, 1, "int"),
        "rsi_exit":         (35, 50, 1, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_vhf_regime_breakout": {
        "vhf_period":       (12, 40, 2, "int"),
        "breakout_period":  (10, 35, 2, "int"),
        "exit_period":      (5, 20, 1, "int"),
        "vhf_min":          (0.12, 0.45, 0.02, "float"),
        "vhf_exit":         (0.05, 0.3, 0.01, "float"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_adx_donchian_thrust": {
        "breakout_period":  (10, 35, 2, "int"),
        "exit_period":      (5, 20, 1, "int"),
        "adx_period":       (8, 24, 1, "int"),
        "adx_min":          (10, 30, 1, "int"),
        "adx_exit":         (8, 22, 1, "int"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
    "openai_ema_slope_breakout": {
        "ema_period":       (15, 55, 2, "int"),
        "slope_lookback":   (2, 10, 1, "int"),
        "breakout_period":  (10, 30, 2, "int"),
        "slope_min":        (0.02, 0.3, 0.01, "float"),
        "slope_exit":       (-0.15, 0.0, 0.01, "float"),
        "atr_stop_mul":     (1.5, 4.0, 0.5, "float"),
    },
}


def _infer_param_ranges(strategy_name):
    """Fallback ranges inferred from a strategy's default params.
    This keeps newly-added candidate strategies optimizable even before
    bespoke PARAM_RANGES are authored.
    """
    try:
        from strategies import get_strategy
        defaults = dict(getattr(get_strategy(strategy_name), "default_params", {}) or {})
    except Exception:
        return None

    ranges = {}
    for pname, value in defaults.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            continue

        key = pname.lower()
        if isinstance(value, int):
            if any(token in key for token in ["buy", "sell", "entry", "exit", "floor", "cap", "threshold", "level", "min", "max"]):
                low = max(-100, value - max(5, int(abs(value) * 0.4)))
                high = min(100, value + max(5, int(abs(value) * 0.4)))
                step = 1
            else:
                span = max(2, int(abs(value) * 0.5))
                low = max(2, value - span)
                high = max(low + 2, value + span)
                step = 1 if value < 25 else max(1, int(round(value / 10)))
            ranges[pname] = (int(low), int(high), int(step), "int")
            continue

        if isinstance(value, float):
            span = max(abs(value) * 0.5, 0.1)
            low = value - span
            high = value + span
            if any(token in key for token in ["mul", "mult", "ratio", "std", "band", "gap"]):
                low = max(0.05, low)
            step = 0.05 if max(abs(low), abs(high)) <= 2 else 0.1
            if abs(high - low) < step:
                high = low + step
            ranges[pname] = (round(low, 4), round(high, 4), step, "float")

    return ranges or None


def _run_single_backtest(args):
    """Worker function for multiprocessing — must be top-level for pickling."""
    df_pickle, strategy_name, override, dynamic_pos = args
    import pandas as pd
    from backtest import run_strategy_backtest
    df = pd.read_pickle(df_pickle)
    try:
        r = run_strategy_backtest(df, strategy_name, params_override=override,
                                  dynamic_position=dynamic_pos)
        return override, r["summary"]["sharpe"], r["summary"]
    except Exception:
        return override, -999, {"sharpe": -999, "total_return": 0, "max_drawdown": 0, "win_rate": 0, "total_trades": 0, "profit_factor": 0}


# ──────────────────────────────────────────────────────────────
#  Method 1: Grid Search (exhaustive)
# ──────────────────────────────────────────────────────────────

def optimize_grid(df, strategy_name, grid, dynamic_pos=False, max_workers=1):
    """Exhaustive grid search. Returns (best_params, best_sharpe, best_summary, combos_tested)."""
    from backtest import run_strategy_backtest

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combos = list(iter_product(*param_values))

    best_sharpe = -999
    best_params = None
    best_summary = None

    if max_workers > 1 and len(combos) > 50:
        # Parallel execution
        import tempfile, pandas as pd
        tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        df.to_pickle(tmp.name)
        tmp.close()
        try:
            tasks = [(tmp.name, strategy_name, dict(zip(param_names, combo)), dynamic_pos)
                     for combo in combos]
            for override, sharpe, summary in _opt_map(
                    _run_single_backtest, tasks, max_workers,
                    chunksize=max(1, len(tasks) // max(1, max_workers))):
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = override
                    best_summary = summary
        finally:
            import os
            os.unlink(tmp.name)
    else:
        # Sequential
        for combo in combos:
            override = dict(zip(param_names, combo))
            try:
                r = run_strategy_backtest(df, strategy_name, params_override=override,
                                          dynamic_position=dynamic_pos)
                sharpe = r["summary"]["sharpe"]
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = override
                    best_summary = r["summary"]
            except:
                continue

    return best_params, best_sharpe, best_summary, len(combos)


# ──────────────────────────────────────────────────────────────
#  Method 2: Bayesian Optimization (Optuna)
# ──────────────────────────────────────────────────────────────

def optimize_bayesian(df, strategy_name, n_trials=200, dynamic_pos=False):
    """Bayesian optimization using Optuna TPE sampler.
    Returns (best_params, best_sharpe, best_summary, n_trials_run)."""
    from backtest import run_strategy_backtest

    ranges = PARAM_RANGES.get(strategy_name) or _infer_param_ranges(strategy_name)
    if not ranges:
        return None, -999, None, 0

    best_summary_holder = [None]  # mutable container for closure

    def objective(trial):
        params = {}
        for pname, (low, high, step, ptype) in ranges.items():
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, low, high, step=step)
            elif ptype == "float":
                # Round to step precision
                val = trial.suggest_float(pname, low, high, step=step)
                params[pname] = round(val, 4)
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, low)  # low is the list

        try:
            r = run_strategy_backtest(df, strategy_name, params_override=params,
                                      dynamic_position=dynamic_pos)
            sharpe = r["summary"]["sharpe"]
            # Track best summary
            try:
                prev_best = trial.study.best_value
            except ValueError:
                prev_best = -999  # 首个 trial 尚无已完成记录
            if sharpe > prev_best:
                best_summary_holder[0] = r["summary"]
            return sharpe
        except Exception:
            return -999

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_sharpe = study.best_value

    # Get best summary (re-run to be safe)
    try:
        r = run_strategy_backtest(df, strategy_name, params_override=best_params,
                                  dynamic_position=dynamic_pos)
        best_summary = r["summary"]
    except:
        best_summary = best_summary_holder[0]

    return best_params, best_sharpe, best_summary, len(study.trials)


# ──────────────────────────────────────────────────────────────
#  Method 3: Coarse-to-Fine (two-stage grid search)
# ──────────────────────────────────────────────────────────────

def _run_combo_worker(args):
    """模块级 worker：粗搜/精搜并行回测单个参数组合（ProcessPoolExecutor 要求可 pickle）"""
    from backtest import run_strategy_backtest
    combo, pnames, sname, df_, dyn = args
    override = dict(zip(pnames, combo))
    try:
        r = run_strategy_backtest(df_, sname, params_override=override,
                                  dynamic_position=dyn)
        return combo, r["summary"]["sharpe"], r["summary"]
    except Exception:
        return combo, -999, None


def optimize_coarse_fine(df, strategy_name, grid, dynamic_pos=False):
    """Two-stage: coarse grid → fine grid around best region.
    Returns (best_params, best_sharpe, best_summary, total_combos)."""

    param_names = list(grid.keys())
    param_values = list(grid.values())

    # Stage 1: Coarse — use every other value (skip elements)
    coarse_values = [v[::2] if len(v) > 4 else v for v in param_values]
    coarse_combos = list(iter_product(*coarse_values))

    best_sharpe = -999
    best_combo = None
    best_summary = None

    from concurrent.futures import ProcessPoolExecutor
    import os

    n_workers = min(os.cpu_count() or 4, 12)
    coarse_tasks = [(c, param_names, strategy_name, df, dynamic_pos) for c in coarse_combos]
    for combo, sharpe, summary in _opt_map(_run_combo_worker, coarse_tasks, n_workers):
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_combo = combo
            best_summary = summary

    if best_combo is None:
        return None, -999, None, len(coarse_combos)

    # Stage 2: Fine — search neighborhood around best coarse params
    fine_values = []
    for i, (pname, vals) in enumerate(zip(param_names, param_values)):
        best_val = best_combo[i]
        if best_val in vals:
            idx = vals.index(best_val)
        else:
            idx = min(range(len(vals)), key=lambda j: abs(vals[j] - best_val) if isinstance(vals[j], (int, float)) else 0)
        lo = max(0, idx - 2)
        hi = min(len(vals), idx + 3)
        fine_values.append(vals[lo:hi])

    fine_combos = list(iter_product(*fine_values))
    coarse_set = set(coarse_combos)
    fine_combos = [c for c in fine_combos if c not in coarse_set]

    # 并行精搜
    fine_tasks = [(c, param_names, strategy_name, df, dynamic_pos) for c in fine_combos]
    if fine_tasks:
        for combo, sharpe, summary in _opt_map(_run_combo_worker, fine_tasks, n_workers):
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_combo = combo
                best_summary = summary

    best_params = dict(zip(param_names, best_combo)) if best_combo else None
    total = len(coarse_combos) + len(fine_combos)
    return best_params, best_sharpe, best_summary, total


# ──────────────────────────────────────────────────────────────
#  Unified optimizer interface
# ──────────────────────────────────────────────────────────────

# 纯成功率目标下的样本折扣阈值。没有它, "只有 2 个买点且都成功" 会拿到
# 100 分直接夺冠 —— 优化器会主动把参数调到几乎不出信号。
_SIG_MIN_N = 30


def _sig_seg_of(dates, bounds):
    """把每个信号的日期映射到第几段。bounds 是各段的右边界(含)。"""
    import bisect
    return [bisect.bisect_left(bounds, d) for d in dates]


def optimize_signal_oos(df, strategy_name, signal_type="buy", n_trials=200,
                        window=10, up_pct=5.0, dn_pct=10.0, entry_mode="next_open",
                        judge_mode="first_touch", same_bar="adverse_first",
                        n_segments=6, holdout=2,
                        holdout_min_rate=40.0, max_gap_pp=15.0,
                        baseline_params=None):
    """**直接优化样本外成功率** —— 跟普通买/卖点优化的根本区别:

      普通优化   目标是整段行情上的成功率。优化器有 200 次机会去迎合这一段
                 历史, 样本内当然好看。实测 NVDA 训练集 61~75%、持出集只有
                 40~62%, **平均落差 18.6pp** —— 报给你的数字有一半是幻觉。
      本函数     把数据切成 n_segments 段, 最后 holdout 段**搜索全程看不见**;
                 目标函数只看搜索段的表现, 而且是
                     平均成功率 − 0.5×段间标准差
                 减标准差是关键: 只看平均的话, "某一段特别好、其余稀烂" 会赢过
                 "每段都稳定 45%", 而前者换个行情就崩。

    方法学上的坑: 拿搜索段当目标, 就把它们变成了训练数据, 结论会自我实现。
    所以持出段单独验一次, 两个数都返回 —— **差得多说明只是换了个对象过拟合**。

    为什么不按段切 df 再逐段跑: 那样每段开头都丢掉指标预热期(200 周期均线
    需要 200 根历史), 段越短失真越大。改成整段跑一次、再按日期把信号分桶。
    """
    import numpy as _np
    from backtest import analyze_buy_points, analyze_sell_points

    ranges = PARAM_RANGES.get(strategy_name) or _infer_param_ranges(strategy_name)
    if not ranges:
        return None, -999, None, 0

    n = len(df)
    if n < 120 or n_segments < 3:
        return None, -999, {"error": f"数据只有 {n} 根, 不够切 {n_segments} 段"}, 0
    holdout = max(1, min(holdout, n_segments // 3))
    # 各段右边界(按位置等分), 用于把信号日期分桶
    edges = [df.index[min(int(n * (i + 1) / n_segments), n - 1)] for i in range(n_segments)]
    n_search = n_segments - holdout

    def _eval(params):
        """返回 (搜索段各段成功率, 搜索段信号数, 持出段成功率, 持出段信号数)"""
        if signal_type == "buy":
            r = analyze_buy_points(df, strategy_name, params_override=params, window=window,
                                   entry_mode=entry_mode, judge_mode=judge_mode,
                                   up_pct=up_pct, dn_pct=dn_pct, same_bar=same_bar)
            pts = r["buy_points"]
            if judge_mode != "first_touch":
                for x in pts:
                    x["success"] = (x["mfe"] >= up_pct and abs(x["mae"]) <= dn_pct)
            dkey = "entry_date"
        else:
            r = analyze_sell_points(df, strategy_name, params_override=params, window=window,
                                    judge_mode=judge_mode, down_pct=up_pct,
                                    up_pct=dn_pct, same_bar=same_bar)
            pts = r["sell_points"]
            if judge_mode != "first_touch":
                for x in pts:
                    x["success"] = (x["mfe"] >= up_pct and x["mae"] <= dn_pct)
            dkey = "date"
        if not pts:
            return [], 0, 0.0, 0, [], []
        import pandas as _pd
        segs = _sig_seg_of([_pd.Timestamp(x[dkey]) for x in pts], edges)
        buckets = [[] for _ in range(n_segments)]
        for x, g in zip(pts, segs):
            if 0 <= g < n_segments:
                buckets[g].append(bool(x.get("success")))
        srates, s_n, s_cnt = [], 0, []
        for g in range(n_search):
            b = buckets[g]
            # 一段里少于 3 个信号, 那一段的"成功率"是噪声(1 个信号就是 0% 或 100%),
            # 不参与平均, 免得把 100% 这种数拉进目标函数
            if len(b) >= 3:
                srates.append(sum(b) / len(b) * 100.0)
                s_n += len(b)
                s_cnt.append(len(b))
        hpts = [x for x, g in zip(pts, segs) if n_search <= g < n_segments]
        hb = [bool(x.get("success")) for x in hpts]
        h_rate = round(sum(hb) / len(hb) * 100.0, 1) if hb else 0.0
        return srates, s_n, h_rate, len(hb), s_cnt, hpts

    best = {"score": -999, "params": None, "detail": None}

    def objective(trial):
        params = {}
        for pname, (low, high, step, ptype) in ranges.items():
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, low, high, step=step)
            elif ptype == "float":
                params[pname] = round(trial.suggest_float(pname, low, high, step=step), 4)
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, low)
        try:
            srates, s_n, h_rate, h_n, s_cnt, _ = _eval(params)
            # 覆盖率: 搜索段里有信号的段数太少就不算数(比如只有 1 段出信号,
            # 那个"平均"和"标准差"都没有意义)
            if len(srates) < max(2, n_search // 2) or s_n == 0:
                return -999
            avg = float(_np.mean(srates))
            sd = float(_np.std(srates))
            score = (avg - 0.5 * sd) * min(s_n / _SIG_MIN_N, 1.0)
            if score > best["score"]:
                best.update({"score": score, "params": dict(params), "detail": {
                    "seg_rates": [round(x, 1) for x in srates],
                    "seg_points": s_cnt,
                    "search_success_rate": round(avg, 1),
                    "search_std": round(sd, 1),
                    "search_points": s_n,
                    "holdout_success_rate": h_rate,
                    "holdout_points": h_n,
                    "sample_discount": round(min(s_n / _SIG_MIN_N, 1.0), 3),
                }})
            return score
        except Exception:
            return -999

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if not best["params"]:
        return None, -999, {"error": "没有一组参数能在足够多的段上出信号"}, len(study.trials)

    def _aggr(pts):
        """把一批信号点压成表里那几列。**只喂持出段的点** —— 表里 success_rate
        报的是持出段, 旁边几列要是拿全历史算的, 就是两个口径混在一行里。"""
        key = "total_buy_points" if signal_type == "buy" else "total_sell_points"
        if not pts:
            return {key: 0, "success_rate": 0.0, "avg_mfe": 0.0, "avg_mae": 0.0, "avg_r_ratio": 0.0}
        m = len(pts)
        _mae = [x["mae"] for x in pts]
        return {
            key: m,
            "success_rate": round(sum(1 for x in pts if x.get("success")) / m * 100.0, 1),
            "avg_mfe": round(float(_np.mean([x["mfe"] for x in pts])), 2),
            "avg_mae": round(float(_np.mean(_mae)), 2),
            "avg_r_ratio": round(float(_np.mean([x.get("r_ratio", 0) for x in pts])), 2),
        }

    d = best["detail"]
    # 最优参数在持出段上的完整指标(买点数/均MFE/均MAE/R倍数)
    try:
        d.update(_aggr(_eval(best["params"])[5]))
    except Exception:
        pass
    # 优化前基线也必须只在**持出段**上算。拿全历史的基线去比持出段的结果,
    # 是拿"样本内、1254 根"比"样本外、14 个信号" —— 实测会出现
    # 基线 61.6% → 优化后 57.1% 这种看着像调坏了、其实根本不可比的数。
    if baseline_params:
        try:
            _bs, _, _bh, _bhn, _, _bhp = _eval(baseline_params)
            d["baseline"] = {**_aggr(_bhp), "holdout_points": _bhn,
                             "search_success_rate": (round(float(_np.mean(_bs)), 1) if _bs else None),
                             "scope": "holdout"}
        except Exception:
            pass
    d["objective_mode"] = "oos"
    d["judge_mode"] = judge_mode
    d["n_segments"] = n_segments
    d["holdout_segments"] = holdout
    # 搜索段 vs 持出段的落差 —— 这才是"还过不过拟合"的答案
    d["oos_gap"] = round(d["search_success_rate"] - d["holdout_success_rate"], 1)
    d["success_rate"] = d["holdout_success_rate"]   # 对外报**持出集**的数, 不报好看的那个

    # ── 验收闸门 ──────────────────────────────────────────────────
    # 光把目标换成样本外还不够: 搜索段一旦成为目标, 优化器就会改成对着它过拟合。
    # 实测 NVDA/claude_dual_ma_cross 搜索段 75% 但持出段只有 28.6% ——
    # 比普通优化的持出 41.7% **还差**。所以持出集必须当**闸门**, 不只是报个数。
    reasons = []
    if d["holdout_points"] < _SIG_MIN_N // 3:
        reasons.append(f"持出段只有 {d['holdout_points']} 个信号, 验不出来")
    if d["holdout_success_rate"] < holdout_min_rate:
        reasons.append(f"持出段成功率 {d['holdout_success_rate']}% < 门槛 {holdout_min_rate}%")
    if d["oos_gap"] > max_gap_pp:
        reasons.append(f"搜索段比持出段高 {d['oos_gap']}pp, 超过上限 {max_gap_pp}pp(还是过拟合)")
    d["accepted"] = not reasons
    d["reject_reason"] = " · ".join(reasons) if reasons else None
    if reasons:
        # 不收 = 不返回参数, 调用方那边 best_params 为空就不会落盘,
        # 原来的参数一个字节都不动。
        return None, best["score"], d, len(study.trials)
    return best["params"], best["score"], d, len(study.trials)


def optimize_buy_signal(df, strategy_name, n_trials=200, window=20, mfe_target=15, mae_limit=10,
                        entry_mode="next_open", objective_mode="composite",
                        judge_mode="mfe_mae", same_bar="adverse_first"):
    """Buy-only 优化。

    objective_mode:
      "composite"(原有, 默认) — 成功率×0.6 + 均MFE×0.3 − 均MAE×0.1
      "success"              — **只优化成功率**, 不看收益幅度。
            score = 成功率 × min(买点数/30, 1)
            那个折扣是必须的: 不加的话, 优化器会把参数调到几乎不出信号 ——
            2 个买点全中就是 100 分, 稳赢任何 200 个买点 60% 的解。
    judge_mode: 跟买点评测同一套口径("first_touch" 先到先判 / "mfe_mae" 原有),
            这样"优化出来的成功率"和"评测里看到的成功率"是同一个数。
    Returns: (best_params, best_score, best_summary, n_trials_run)"""
    from backtest import analyze_buy_points

    ranges = PARAM_RANGES.get(strategy_name) or _infer_param_ranges(strategy_name)
    if not ranges:
        return None, -999, None, 0

    best_summary_holder = [None]

    def objective(trial):
        params = {}
        for pname, (low, high, step, ptype) in ranges.items():
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, low, high, step=step)
            elif ptype == "float":
                params[pname] = round(trial.suggest_float(pname, low, high, step=step), 4)
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, low)
        try:
            r = analyze_buy_points(df, strategy_name, params_override=params,
                                   window=window, entry_mode=entry_mode,
                                   judge_mode=judge_mode, up_pct=mfe_target,
                                   dn_pct=mae_limit, same_bar=same_bar)
            bps = r["buy_points"]
            if not bps:
                return -999
            if judge_mode != "first_touch":
                for bp in bps:
                    bp["success"] = (bp["mfe"] >= mfe_target and abs(bp["mae"]) <= mae_limit)
            n = len(bps)
            success_rate = sum(1 for bp in bps if bp["success"]) / n * 100
            avg_mfe = sum(bp["mfe"] for bp in bps) / n
            avg_mae = sum(abs(bp["mae"]) for bp in bps) / n
            if objective_mode == "success":
                score = success_rate * min(n / _SIG_MIN_N, 1.0)
            else:
                # 综合得分: 成功率权重高 + MFE奖励 - MAE惩罚
                score = success_rate * 0.6 + avg_mfe * 0.3 - avg_mae * 0.1
            summary = {
                "total_buy_points": n, "success_rate": round(success_rate, 1),
                "avg_mfe": round(avg_mfe, 2), "avg_mae": round(-avg_mae, 2),
                "avg_r_ratio": round(sum(bp["r_ratio"] for bp in bps) / n, 2),
            }
            try:
                prev = trial.study.best_value
            except ValueError:
                prev = -999
            if score > prev:
                best_summary_holder[0] = summary
            return score
        except Exception:
            return -999

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    # Re-run for final summary
    try:
        r = analyze_buy_points(df, strategy_name, params_override=best_params,
                               window=window, entry_mode=entry_mode,
                               judge_mode=judge_mode, up_pct=mfe_target,
                               dn_pct=mae_limit, same_bar=same_bar)
        bps = r["buy_points"]
        if bps:
            if judge_mode != "first_touch":
                for bp in bps:
                    bp["success"] = (bp["mfe"] >= mfe_target and abs(bp["mae"]) <= mae_limit)
            n = len(bps)
            best_summary = {
                "total_buy_points": n,
                "success_rate": round(sum(1 for bp in bps if bp["success"]) / n * 100, 1),
                "avg_mfe": round(sum(bp["mfe"] for bp in bps) / n, 2),
                "avg_mae": round(-sum(abs(bp["mae"]) for bp in bps) / n, 2),
                "avg_r_ratio": round(sum(bp["r_ratio"] for bp in bps) / n, 2),
                "best_window": window,
                "objective_mode": objective_mode,
                "judge_mode": judge_mode,
                # 样本不足时 score 是被折扣过的, 写出来免得看着跟成功率对不上
                "sample_discount": round(min(n / _SIG_MIN_N, 1.0), 3) if objective_mode == "success" else None,
            }
        else:
            best_summary = best_summary_holder[0]
    except Exception:
        best_summary = best_summary_holder[0]

    return best_params, best_score, best_summary, len(study.trials)


def optimize_sell_signal(df, strategy_name, n_trials=200, window=20, mfe_target=10, mae_limit=5,
                         objective_mode="composite", judge_mode="mfe_mae", same_bar="adverse_first"):
    """Sell-only 优化。objective_mode / judge_mode 含义同 optimize_buy_signal,
    只是方向相反(先跌算卖对)。"success" 模式同样带 min(卖点数/30,1) 样本折扣。
    Returns: (best_params, best_score, best_summary, n_trials_run)"""
    from backtest import analyze_sell_points

    ranges = PARAM_RANGES.get(strategy_name) or _infer_param_ranges(strategy_name)
    if not ranges:
        return None, -999, None, 0

    best_summary_holder = [None]

    def objective(trial):
        params = {}
        for pname, (low, high, step, ptype) in ranges.items():
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, low, high, step=step)
            elif ptype == "float":
                params[pname] = round(trial.suggest_float(pname, low, high, step=step), 4)
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, low)
        try:
            r = analyze_sell_points(df, strategy_name, params_override=params, window=window,
                                    judge_mode=judge_mode, down_pct=mfe_target,
                                    up_pct=mae_limit, same_bar=same_bar)
            sps = r["sell_points"]
            if not sps:
                return -999
            for sp in sps:
                if judge_mode != "first_touch":
                    sp["success"] = (sp["mfe"] >= mfe_target and sp["mae"] <= mae_limit)
            n = len(sps)
            success_rate = sum(1 for sp in sps if sp["success"]) / n * 100
            avg_mfe = sum(sp["mfe"] for sp in sps) / n
            avg_mae = sum(sp["mae"] for sp in sps) / n
            if objective_mode == "success":
                score = success_rate * min(n / _SIG_MIN_N, 1.0)
            else:
                score = success_rate * 0.6 + avg_mfe * 0.3 - avg_mae * 0.1
            summary = {
                "total_sell_points": n, "success_rate": round(success_rate, 1),
                "avg_mfe": round(avg_mfe, 2), "avg_mae": round(avg_mae, 2),
                "avg_r_ratio": round(sum(sp["r_ratio"] for sp in sps) / n, 2),
            }
            try:
                prev = trial.study.best_value
            except ValueError:
                prev = -999
            if score > prev:
                best_summary_holder[0] = summary
            return score
        except Exception:
            return -999

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    try:
        r = analyze_sell_points(df, strategy_name, params_override=best_params, window=window,
                                judge_mode=judge_mode, down_pct=mfe_target,
                                up_pct=mae_limit, same_bar=same_bar)
        sps = r["sell_points"]
        if sps:
            for sp in sps:
                if judge_mode != "first_touch":
                    sp["success"] = (sp["mfe"] >= mfe_target and sp["mae"] <= mae_limit)
            n = len(sps)
            best_summary = {
                "total_sell_points": n,
                "success_rate": round(sum(1 for sp in sps if sp["success"]) / n * 100, 1),
                "avg_mfe": round(sum(sp["mfe"] for sp in sps) / n, 2),
                "avg_mae": round(sum(sp["mae"] for sp in sps) / n, 2),
                "avg_r_ratio": round(sum(sp["r_ratio"] for sp in sps) / n, 2),
                "best_window": window,
            }
        else:
            best_summary = best_summary_holder[0]
    except Exception:
        best_summary = best_summary_holder[0]

    return best_params, best_score, best_summary, len(study.trials)


def optimize_strategy(df, strategy_name, method="bayesian", grid=None,
                      dynamic_pos=False, n_trials=200, max_workers=1,
                      signal_type="buy_and_sell", sig_opts=None):
    """
    Unified optimizer interface.
    method: "grid" | "bayesian" | "coarse_fine"
    signal_type: "buy_and_sell" | "buy" | "sell"
    Returns: (best_params, best_score, best_summary, combos_tested, method_used)
    """
    t0 = time.time()

    # ── Buy-only / Sell-only 模式 → 使用信号质量目标函数 ──
    # sig_opts: buy/sell 专项优化的口径 —— objective_mode / judge_mode /
    # same_bar / window / mfe_target / mae_limit。不给就是原有行为。
    _so = dict(sig_opts or {})
    # objective_mode="oos" → 直接优化样本外成功率(最后几段全程不给搜索看)
    if signal_type in ("buy", "sell") and _so.get("objective_mode") == "oos":
        params, score, summary, combos = optimize_signal_oos(
            df, strategy_name, signal_type=signal_type, n_trials=n_trials,
            window=_so.get("window", 10),
            up_pct=_so.get("mfe_target", 5), dn_pct=_so.get("mae_limit", 10),
            entry_mode=_so.get("entry_mode", "next_open"),
            judge_mode=_so.get("judge_mode", "first_touch"),
            same_bar=_so.get("same_bar", "adverse_first"),
            n_segments=int(_so.get("n_segments", 6)),
            holdout=int(_so.get("holdout_segments", 2)),
            holdout_min_rate=float(_so.get("holdout_min_rate", 40.0)),
            max_gap_pp=float(_so.get("max_gap_pp", 15.0)),
            baseline_params=_so.get("baseline_params"))
        dt = time.time() - t0
        _d = summary or {}
        logger.info(f"  [{signal_type}-oos] {strategy_name}: 搜索段 {_d.get('search_success_rate')}% / "
                    f"持出段 {_d.get('holdout_success_rate')}% (落差 {_d.get('oos_gap')}pp), "
                    f"combos={combos}, time={dt:.1f}s")
        return params, score, summary, combos, f"{signal_type}_oos"

    if signal_type == "buy":
        params, score, summary, combos = optimize_buy_signal(
            df, strategy_name, n_trials,
            window=_so.get("window", 20),
            mfe_target=_so.get("mfe_target", 15),
            mae_limit=_so.get("mae_limit", 10),
            entry_mode=_so.get("entry_mode", "next_open"),
            objective_mode=_so.get("objective_mode", "composite"),
            judge_mode=_so.get("judge_mode", "mfe_mae"),
            same_bar=_so.get("same_bar", "adverse_first"))
        dt = time.time() - t0
        logger.info(f"  [buy-opt] {strategy_name}: score={score:.2f}, combos={combos}, time={dt:.1f}s")
        return params, score, summary, combos, "buy_bayesian"

    if signal_type == "sell":
        params, score, summary, combos = optimize_sell_signal(
            df, strategy_name, n_trials,
            window=_so.get("window", 20),
            mfe_target=_so.get("mfe_target", 10),
            mae_limit=_so.get("mae_limit", 5),
            objective_mode=_so.get("objective_mode", "composite"),
            judge_mode=_so.get("judge_mode", "mfe_mae"),
            same_bar=_so.get("same_bar", "adverse_first"))
        dt = time.time() - t0
        logger.info(f"  [sell-opt] {strategy_name}: score={score:.2f}, combos={combos}, time={dt:.1f}s")
        return params, score, summary, combos, "sell_bayesian"

    # ── Buy & Sell 模式 → 分别优化 buy/sell，合并参数，回测验证 ──
    from backtest import run_strategy_backtest
    from strategies import get_strategy

    # Step 1: 优化 Buy 信号
    buy_trials = max(n_trials // 2, 50)
    buy_params, buy_score, buy_summary, buy_combos = optimize_buy_signal(
        df, strategy_name, buy_trials)
    logger.info(f"  [buy&sell] {strategy_name} buy阶段: score={buy_score:.2f}, combos={buy_combos}")

    # Step 2: 优化 Sell 信号
    sell_trials = max(n_trials // 2, 50)
    sell_params, sell_score, sell_summary, sell_combos = optimize_sell_signal(
        df, strategy_name, sell_trials)
    logger.info(f"  [buy&sell] {strategy_name} sell阶段: score={sell_score:.2f}, combos={sell_combos}")

    # Step 3: 双参数分离模式回测（buy信号用buy_params，sell信号用sell_params）
    strategy = get_strategy(strategy_name)

    # 构建完整参数集（各自以 default_params 为底层）
    full_buy_params = {**strategy.default_params}
    if buy_params:
        full_buy_params.update(buy_params)
    full_sell_params = {**strategy.default_params}
    if sell_params:
        full_sell_params.update(sell_params)

    # 兼容：merged_params 仍然保存（用于不支持双参数的场景）
    merged_params = {**strategy.default_params}
    if buy_params:
        merged_params.update(buy_params)
    if sell_params:
        merged_params.update(sell_params)

    # Step 4: 用双参数分离模式跑完整回测
    try:
        r = run_strategy_backtest(df, strategy_name,
                                  buy_params=full_buy_params,
                                  sell_params=full_sell_params,
                                  dynamic_position=dynamic_pos)
        summary = r["summary"]
        sharpe = summary["sharpe"]
    except Exception:
        summary = buy_summary or {}
        sharpe = -999

    # 将 buy/sell 独立指标附加到 summary
    summary["buy_params"] = buy_params
    summary["sell_params"] = sell_params
    summary["buy_metrics"] = buy_summary
    summary["sell_metrics"] = sell_summary

    total_combos = buy_combos + sell_combos
    dt = time.time() - t0
    logger.info(f"  [buy&sell] {strategy_name}: sharpe={sharpe:.3f}, "
                f"combos={total_combos}, time={dt:.1f}s")

    return merged_params, sharpe, summary, total_combos, "buy_and_sell"
