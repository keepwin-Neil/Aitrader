import pandas as pd
import numpy as np
import ta

from .base import BaseStrategy

# ─── 1. 多指标共振（原有策略）────────────────────────────────────────────

class MultiIndicator(BaseStrategy):
    name = "claude_multi_indicator"
    description = "多指标共振：EMA多头排列 + MACD金叉 + RSI区间 + 布林带 + 放量"
    indicator_formulas = {
        "ema_fast": {"name": "EMA Fast", "formula": "EMA(close, N)", "params": ["ema_fast"]},
        "ema_mid": {"name": "EMA Mid", "formula": "EMA(close, N)", "params": ["ema_mid"]},
        "ema_slow": {"name": "EMA Slow", "formula": "EMA(close, N)", "params": ["ema_slow"]},
        "macd": {"name": "MACD", "formula": "EMA(close, fast) - EMA(close, slow)", "params": ["macd_fast", "macd_slow"]},
        "macd_signal": {"name": "MACD Signal", "formula": "EMA(MACD, N)", "params": ["macd_signal"]},
        "macd_hist": {"name": "MACD Hist", "formula": "MACD - Signal", "params": []},
        "rsi": {"name": "RSI", "formula": "100 - 100/(1 + avgGain/avgLoss)", "params": ["rsi_period"]},
        "bb_upper": {"name": "BB Upper", "formula": "SMA(close, N) + K × StdDev(close, N)", "params": ["bb_period", "bb_std"]},
        "bb_lower": {"name": "BB Lower", "formula": "SMA(close, N) - K × StdDev(close, N)", "params": ["bb_period", "bb_std"]},
        "vol_ratio": {"name": "Vol Ratio", "formula": "Volume / SMA(Volume, N)", "params": ["vol_ma"]},
    }
    signal_rules = {
        "buy": "EMA_fast > EMA_mid > EMA_slow AND MACD > Signal AND MACD_Hist > 0 AND rsi_buy_min < RSI < rsi_buy_max AND Close > BB_Mid AND Vol_Ratio > vol_ratio_min",
        "sell": "EMA_fast < EMA_mid OR (MACD < Signal AND MACD_Hist < 0) OR RSI > rsi_sell",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA10", "color": "#ff9800"},
            {"col": "ema_mid", "label": "EMA20", "color": "#2196f3"},
            {"col": "ema_slow", "label": "EMA60", "color": "#9c27b0"},
            {"col": "bb_upper", "label": "BB Upper", "color": "#555555", "type": "line", "dash": True},
            {"col": "bb_lower", "label": "BB Lower", "color": "#555555", "type": "line", "dash": True},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"},
        ],
    }
    default_params = {
        "ema_fast": 10, "ema_mid": 20, "ema_slow": 60,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "rsi_period": 14, "rsi_buy_min": 45, "rsi_buy_max": 70, "rsi_sell": 78,
        "bb_period": 20, "bb_std": 2,
        "vol_ma": 20, "vol_ratio_min": 1.2,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_mid"]  = ta.trend.ema_indicator(c, window=params["ema_mid"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        macd = ta.trend.MACD(c, window_slow=params["macd_slow"],
                             window_fast=params["macd_fast"],
                             window_sign=params["macd_signal"])
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["vol_ma"]    = v.rolling(params["vol_ma"]).mean()
        df["vol_ratio"] = v / df["vol_ma"]
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = ((df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"])
               & (df["macd"] > df["macd_signal"]) & (df["macd_hist"] > 0)
               & (df["rsi"] > params["rsi_buy_min"]) & (df["rsi"] < params["rsi_buy_max"])
               & (df["close"] > df["bb_mid"])
               & (df["vol_ratio"] > params["vol_ratio_min"]))
        sell = ((df["ema_fast"] < df["ema_mid"])
                | ((df["macd"] < df["macd_signal"]) & (df["macd_hist"] < 0))
                | (df["rsi"] > params["rsi_sell"]))
        df["buy_signal"]  = buy
        df["sell_signal"] = sell
        return df


# ─── 2. 双均线交叉 ─────────────────────────────────────────────────────

class DualMaCross(BaseStrategy):
    name = "claude_dual_ma_cross"
    description = "双均线交叉：快线上穿慢线买入，下穿卖出"
    indicator_formulas = {
        "ma_fast": {"name": "MA Fast", "formula": "SMA/EMA(close, N)", "params": ["fast_period", "ma_type"]},
        "ma_slow": {"name": "MA Slow", "formula": "SMA/EMA(close, N)", "params": ["slow_period", "ma_type"]},
    }
    signal_rules = {
        "buy": "MA_Fast 上穿 MA_Slow (金叉)",
        "sell": "MA_Fast 下穿 MA_Slow (死叉)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ma_fast", "label": "MA Fast", "color": "#ff9800"},
            {"col": "ma_slow", "label": "MA Slow", "color": "#2196f3"},
        ],
        "subchart": [],
    }
    default_params = {
        "fast_period": 10, "slow_period": 25, "ma_type": "sma",
        "atr_stop_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        if params.get("ma_type", "sma") == "sma":
            df["ma_fast"] = c.rolling(params["fast_period"]).mean()
            df["ma_slow"] = c.rolling(params["slow_period"]).mean()
        else:
            df["ma_fast"] = ta.trend.ema_indicator(c, window=params["fast_period"])
            df["ma_slow"] = ta.trend.ema_indicator(c, window=params["slow_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_fast = df["ma_fast"].shift(1)
        prev_slow = df["ma_slow"].shift(1)
        df["buy_signal"]  = (prev_fast <= prev_slow) & (df["ma_fast"] > df["ma_slow"])
        df["sell_signal"] = (prev_fast >= prev_slow) & (df["ma_fast"] < df["ma_slow"])
        df["buy_signal"]  = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 3. 动量突破 ───────────────────────────────────────────────────────

class MomentumBreakout(BaseStrategy):
    name = "claude_momentum_breakout"
    description = "动量突破：N日新高突破买入，跌破M日低点卖出"
    indicator_formulas = {
        "highest": {"name": "N日最高价", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
        "lowest": {"name": "M日最低价", "formula": "Min(Low[-1], M)", "params": ["exit_period"]},
        "vol_ratio": {"name": "成交量比", "formula": "Volume / SMA(Volume, N)", "params": ["vol_ma"]},
    }
    signal_rules = {
        "buy": "Close > N日最高价 AND Vol_Ratio > vol_ratio_min",
        "sell": "Close < M日最低价",
    }
    chart_indicators = {
        "overlay": [
            {"col": "highest", "label": "N日高点", "color": "#f44336", "dash": True},
            {"col": "lowest", "label": "M日低点", "color": "#4caf50", "dash": True},
        ],
        "subchart": [
            {"col": "vol_ratio", "label": "Vol Ratio", "color": "#ff9800"},
        ],
    }
    default_params = {
        "breakout_period": 25, "exit_period": 10,
        "vol_ma": 20, "vol_ratio_min": 0.8,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        # shift(1) 避免用当前bar的high/low，确保是"突破前N日"
        df["highest"]  = h.shift(1).rolling(params["breakout_period"]).max()
        df["lowest"]   = l.shift(1).rolling(params["exit_period"]).min()
        df["vol_ma"]   = v.rolling(params["vol_ma"]).mean()
        df["vol_ratio"] = v / df["vol_ma"]
        df["atr"]      = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"]  = ((df["close"] > df["highest"])
                             & (df["vol_ratio"] > params["vol_ratio_min"]))
        df["sell_signal"] = df["close"] < df["lowest"]
        return df


# ─── 4. 均值回归 ───────────────────────────────────────────────────────

class MeanReversion(BaseStrategy):
    name = "claude_mean_reversion"
    description = "均值回归：RSI超卖+布林带下轨买入，RSI超买+上轨卖出"
    indicator_formulas = {
        "rsi": {"name": "RSI", "formula": "100 - 100/(1 + avgGain/avgLoss)", "params": ["rsi_period"]},
        "bb_upper": {"name": "BB Upper", "formula": "SMA(close, N) + K × StdDev(close, N)", "params": ["bb_period", "bb_std"]},
        "bb_mid": {"name": "BB Mid", "formula": "SMA(close, N)", "params": ["bb_period"]},
        "bb_lower": {"name": "BB Lower", "formula": "SMA(close, N) - K × StdDev(close, N)", "params": ["bb_period", "bb_std"]},
    }
    signal_rules = {
        "buy": "RSI < rsi_oversold AND Close <= BB_Lower",
        "sell": "RSI > rsi_overbought OR Close >= BB_Upper",
    }
    chart_indicators = {
        "overlay": [
            {"col": "bb_upper", "label": "BB Upper", "color": "#ef5350", "dash": True},
            {"col": "bb_mid", "label": "BB Mid", "color": "#78909c"},
            {"col": "bb_lower", "label": "BB Lower", "color": "#66bb6a", "dash": True},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "rsi_period": 10, "rsi_oversold": 35, "rsi_overbought": 70,
        "bb_period": 25, "bb_std": 2.0,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["atr"]      = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"]  = ((df["rsi"] < params["rsi_oversold"])
                             & (df["close"] <= df["bb_lower"]))
        df["sell_signal"] = ((df["rsi"] > params["rsi_overbought"])
                             | (df["close"] >= df["bb_upper"]))
        return df


# ─── 5. 海龟交易 ───────────────────────────────────────────────────────

class TurtleTrading(BaseStrategy):
    name = "claude_turtle_trading"
    description = "海龟交易：20日突破入场，10日突破离场，ATR仓位管理"
    indicator_formulas = {
        "entry_high": {"name": "入场高点", "formula": "Max(High[-1], N)", "params": ["entry_period"]},
        "exit_low": {"name": "离场低点", "formula": "Min(Low[-1], M)", "params": ["exit_period"]},
        "atr": {"name": "ATR", "formula": "ATR(high, low, close, 20)", "params": []},
    }
    signal_rules = {
        "buy": "Close > N日最高价 (突破入场)",
        "sell": "Close < M日最低价 (突破离场)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "entry_high", "label": "入场高点", "color": "#f44336", "dash": True},
            {"col": "exit_low", "label": "离场低点", "color": "#4caf50", "dash": True},
        ],
        "subchart": [],
    }
    default_params = {
        "entry_period": 30, "exit_period": 10,
        "atr_stop_mul": 1.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["entry_high"] = h.shift(1).rolling(params["entry_period"]).max()
        df["exit_low"]   = l.shift(1).rolling(params["exit_period"]).min()
        df["atr"]        = ta.volatility.average_true_range(h, l, c, window=20)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"]  = df["close"] > df["entry_high"]
        df["sell_signal"] = df["close"] < df["exit_low"]
        return df


# ─── 6. MACD 背离 ──────────────────────────────────────────────────────

class MacdDivergence(BaseStrategy):
    name = "claude_macd_divergence"
    description = "MACD背离：价格新低但MACD不创新低时买入（底背离）"
    indicator_formulas = {
        "macd": {"name": "MACD", "formula": "EMA(close, fast) - EMA(close, slow)", "params": ["macd_fast", "macd_slow"]},
        "macd_signal": {"name": "Signal", "formula": "EMA(MACD, N)", "params": ["macd_signal_w"]},
        "macd_hist": {"name": "MACD Hist", "formula": "MACD - Signal", "params": []},
        "rsi": {"name": "RSI", "formula": "100 - 100/(1 + avgGain/avgLoss)", "params": ["rsi_period"]},
        "price_low": {"name": "价格低点", "formula": "Min(Close, lookback)", "params": ["lookback"]},
        "macd_low": {"name": "MACD低点", "formula": "Min(MACD, lookback)", "params": ["lookback"]},
    }
    signal_rules = {
        "buy": "Close <= 近期低点×1.05 AND MACD > MACD低点(背离) AND MACD_Hist由负转正(金叉)",
        "sell": "RSI > rsi_sell OR Close < EMA(20)",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "macd", "label": "MACD", "color": "#2196f3"},
            {"col": "macd_signal", "label": "Signal", "color": "#ff9800"},
            {"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"},
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "macd_fast": 12, "macd_slow": 26, "macd_signal_w": 9,
        "lookback": 30, "rsi_period": 14, "rsi_sell": 80,
        "atr_stop_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        macd = ta.trend.MACD(c, window_slow=params["macd_slow"],
                             window_fast=params["macd_fast"],
                             window_sign=params["macd_signal_w"])
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()
        df["rsi"]         = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"]         = ta.volatility.average_true_range(h, l, c, window=14)
        lb = params["lookback"]
        df["price_low"]   = c.shift(1).rolling(lb).min()
        df["macd_low"]    = df["macd"].shift(1).rolling(lb).min()
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 底背离：价格接近近期低点，但 MACD 比近期低点高（背离），且 MACD 金叉
        price_near_low = df["close"] <= df["price_low"] * 1.05  # 5% 容差
        macd_higher    = df["macd"] > df["macd_low"]
        macd_cross_up  = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
        df["buy_signal"]  = price_near_low & macd_higher & macd_cross_up
        # 卖出：RSI超买 或 价格跌破布林中轨（趋势反转）
        ema_mid = ta.trend.ema_indicator(df["close"], window=20)
        df["sell_signal"] = (df["rsi"] > params["rsi_sell"]) | (df["close"] < ema_mid)
        df["buy_signal"]  = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 7. KDJ + DMI 趋势过滤 ──────────────────────────────────────────────

class KdjDmi(BaseStrategy):
    name = "claude_kdj_dmi"
    description = "KDJ超卖金叉买入 + DMI趋势过滤，ADX确认趋势强度"
    indicator_formulas = {
        "k": {"name": "K线", "formula": "Stochastic %K(H, L, C, N)", "params": ["kdj_period"]},
        "d": {"name": "D线", "formula": "SMA(%K, 3)", "params": ["kdj_period"]},
        "j": {"name": "J线", "formula": "3K - 2D", "params": []},
        "adx": {"name": "ADX", "formula": "ADX(H, L, C, N)", "params": ["adx_period"]},
        "di_plus": {"name": "DI+", "formula": "+DM / ATR(N)", "params": ["adx_period"]},
        "di_minus": {"name": "DI-", "formula": "-DM / ATR(N)", "params": ["adx_period"]},
    }
    signal_rules = {
        "buy": "K上穿D(金叉) AND J[-1] < kdj_oversold AND ADX > adx_threshold AND DI+ > DI-",
        "sell": "K下穿D(死叉) OR J > kdj_overbought",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "k", "label": "K", "color": "#2196f3"},
            {"col": "d", "label": "D", "color": "#ff9800"},
            {"col": "j", "label": "J", "color": "#e91e63"},
            {"col": "adx", "label": "ADX", "color": "#9c27b0"},
        ],
    }
    default_params = {
        "kdj_period": 14, "kdj_oversold": 20, "kdj_overbought": 80,
        "adx_period": 14, "adx_threshold": 20,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        stoch = ta.momentum.StochasticOscillator(
            h, l, c, window=params["kdj_period"], smooth_window=3)
        df["k"] = stoch.stoch()
        df["d"] = stoch.stoch_signal()
        df["j"] = 3 * df["k"] - 2 * df["d"]
        df["adx"] = ta.trend.adx(h, l, c, window=params["adx_period"])
        df["di_plus"]  = ta.trend.adx_pos(h, l, c, window=params["adx_period"])
        df["di_minus"] = ta.trend.adx_neg(h, l, c, window=params["adx_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：K线上穿D线（金叉）+ J值从超卖区回升 + ADX > 阈值（趋势存在）+ DI+>DI-
        k_cross_d = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1))
        j_oversold = df["j"].shift(1) < params["kdj_oversold"]
        trend_up = (df["adx"] > params["adx_threshold"]) & (df["di_plus"] > df["di_minus"])
        df["buy_signal"] = k_cross_d & j_oversold & trend_up
        # 卖出：K线下穿D线（死叉）+ J值超买
        k_cross_d_down = (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1))
        df["sell_signal"] = k_cross_d_down | (df["j"] > params["kdj_overbought"])
        df["buy_signal"]  = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 8. VWAP 突破策略 ─────────────────────────────────────────────────

class VwapBreakout(BaseStrategy):
    name = "claude_vwap_breakout"
    description = "VWAP突破：价格突破VWAP+ATR通道买入，跌破VWAP卖出"
    indicator_formulas = {
        "vwap": {"name": "VWAP", "formula": "Σ(TP×Vol, N) / Σ(Vol, N), TP=(H+L+C)/3", "params": ["vwap_period"]},
        "vwap_upper": {"name": "VWAP Upper", "formula": "VWAP + K × ATR(N)", "params": ["atr_band_mul", "atr_period"]},
        "vwap_lower": {"name": "VWAP Lower", "formula": "VWAP - K × ATR(N)", "params": ["atr_band_mul", "atr_period"]},
        "ema_trend": {"name": "EMA50", "formula": "EMA(close, 50)", "params": []},
        "vol_ratio": {"name": "成交量比", "formula": "Volume / SMA(Volume, N)", "params": ["vol_ma"]},
    }
    signal_rules = {
        "buy": "Close[-1] <= VWAP_Upper AND Close > VWAP_Upper AND Vol_Ratio > vol_ratio_min AND Close > EMA50",
        "sell": "Close < VWAP",
    }
    chart_indicators = {
        "overlay": [
            {"col": "vwap", "label": "VWAP", "color": "#ff9800"},
            {"col": "vwap_upper", "label": "VWAP Upper", "color": "#ef5350", "dash": True},
            {"col": "vwap_lower", "label": "VWAP Lower", "color": "#66bb6a", "dash": True},
            {"col": "ema_trend", "label": "EMA50", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "vol_ratio", "label": "Vol Ratio", "color": "#ff9800"},
        ],
    }
    default_params = {
        "vwap_period": 20, "atr_period": 14,
        "atr_band_mul": 1.5, "atr_stop_mul": 2.0,
        "vol_ma": 20, "vol_ratio_min": 1.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        # 滚动VWAP（典型价格 * 成交量 的累计均值）
        tp = (h + l + c) / 3
        df["vwap"] = (tp * v).rolling(params["vwap_period"]).sum() / v.rolling(params["vwap_period"]).sum()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["vwap_upper"] = df["vwap"] + params["atr_band_mul"] * df["atr"]
        df["vwap_lower"] = df["vwap"] - params["atr_band_mul"] * df["atr"]
        df["vol_ma"] = v.rolling(params["vol_ma"]).mean()
        df["vol_ratio"] = v / df["vol_ma"]
        df["ema_trend"] = ta.trend.ema_indicator(c, window=50)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：价格从下方突破VWAP上轨 + 放量 + 趋势向上
        prev_below = df["close"].shift(1) <= df["vwap_upper"]
        df["buy_signal"] = (prev_below & (df["close"] > df["vwap_upper"])
                            & (df["vol_ratio"] > params["vol_ratio_min"])
                            & (df["close"] > df["ema_trend"]))
        # 卖出：跌破VWAP
        df["sell_signal"] = df["close"] < df["vwap"]
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 9. 自适应均线策略（Kaufman AMA）──────────────────────────────────

class AdaptiveMA(BaseStrategy):
    name = "claude_adaptive_ma"
    description = "Kaufman自适应均线：AMA变化率加速时顺势交易"
    indicator_formulas = {
        "ama": {"name": "AMA", "formula": "AMA[i] = AMA[i-1] + SC² × (Close - AMA[i-1])\nER = |Close-Close[-N]| / Σ|ΔClose|\nSC = ER×(fast_sc - slow_sc) + slow_sc", "params": ["er_period", "fast_sc", "slow_sc"]},
        "ama_diff": {"name": "AMA变化率", "formula": "AMA.pct_change()", "params": []},
    }
    signal_rules = {
        "buy": "AMA变化率 > threshold AND AMA变化率 > MA(AMA变化率,5) AND Close > AMA",
        "sell": "AMA变化率 < 0 OR Close < AMA",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ama", "label": "AMA", "color": "#ff9800"},
        ],
        "subchart": [
            {"col": "ama_diff", "label": "AMA Diff", "color": "#2196f3", "type": "histogram"},
        ],
    }
    default_params = {
        "er_period": 20, "fast_sc": 3, "slow_sc": 30,
        "atr_stop_mul": 2.5, "threshold": 0.003,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        er_period = params["er_period"]
        fast_sc = 2.0 / (params["fast_sc"] + 1)
        slow_sc = 2.0 / (params["slow_sc"] + 1)

        # Efficiency Ratio
        direction = abs(c - c.shift(er_period))
        volatility = c.diff().abs().rolling(er_period).sum()
        er = direction / volatility.replace(0, np.nan)
        er = er.fillna(0)

        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # AMA
        ama = pd.Series(index=df.index, dtype=float)
        ama.iloc[er_period] = c.iloc[er_period]
        for i in range(er_period + 1, len(df)):
            ama.iloc[i] = ama.iloc[i-1] + sc.iloc[i] * (c.iloc[i] - ama.iloc[i-1])
        df["ama"] = ama
        df["ama_diff"] = df["ama"].pct_change()
        df["ama_diff_ma"] = df["ama_diff"].rolling(5).mean()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        threshold = params["threshold"]
        # 买入：AMA加速上行（变化率超过阈值且加速）
        df["buy_signal"] = ((df["ama_diff"] > threshold)
                            & (df["ama_diff"] > df["ama_diff_ma"])
                            & (df["close"] > df["ama"]))
        # 卖出：AMA转向或价格跌破AMA
        df["sell_signal"] = ((df["ama_diff"] < 0) | (df["close"] < df["ama"]))
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 10. RSI+MACD混合趋势策略 ────────────────────────────────────────

class RsiMacdTrend(BaseStrategy):
    name = "claude_rsi_macd_trend"
    description = "RSI+MACD趋势：RSI区间 + MACD金叉 + EMA趋势确认"
    indicator_formulas = {
        "rsi": {"name": "RSI", "formula": "100 - 100/(1 + avgGain/avgLoss)", "params": ["rsi_period"]},
        "macd_hist": {"name": "MACD Hist", "formula": "MACD(fast,slow) - Signal(N)", "params": ["macd_fast", "macd_slow", "macd_signal"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_trend"]},
    }
    signal_rules = {
        "buy": "rsi_entry < RSI < rsi_exit AND MACD_Hist由负转正 AND Close > EMA_Trend",
        "sell": "RSI > rsi_exit OR (MACD_Hist < 0 AND Close < EMA_Trend)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"},
        ],
    }
    default_params = {
        "rsi_period": 14, "rsi_entry": 30, "rsi_exit": 80,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "ema_trend": 40, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        macd = ta.trend.MACD(c, window_slow=params["macd_slow"],
                             window_fast=params["macd_fast"],
                             window_sign=params["macd_signal"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_trend"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # RSI在合适区间 + MACD柱状图转正 + 价格在趋势线上方
        rsi_ok = (df["rsi"] > params["rsi_entry"]) & (df["rsi"] < params["rsi_exit"])
        macd_cross = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
        trend_up = df["close"] > df["ema_trend"]
        df["buy_signal"] = rsi_ok & macd_cross & trend_up
        # 卖出：RSI超买 或 MACD死叉+跌破趋势线
        df["sell_signal"] = ((df["rsi"] > params["rsi_exit"])
                             | ((df["macd_hist"] < 0) & (df["close"] < df["ema_trend"])))
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 11. 组合投票策略 ──────────────────────────────────────────────────

class EnsembleVoting(BaseStrategy):
    name = "claude_ensemble_voting"
    description = "组合投票：top4策略任一买入信号入场，多数卖出信号(≥3/4)离场"
    indicator_formulas = {
        "mr_rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["mr_rsi_period"]},
        "mr_bb_upper": {"name": "BB Upper", "formula": "SMA(close,N) + K×StdDev", "params": ["mr_bb_period", "mr_bb_std"]},
        "mr_bb_lower": {"name": "BB Lower", "formula": "SMA(close,N) - K×StdDev", "params": ["mr_bb_period", "mr_bb_std"]},
        "dm_fast": {"name": "MA Fast", "formula": "SMA(close, N)", "params": ["dm_fast"]},
        "dm_slow": {"name": "MA Slow", "formula": "SMA(close, N)", "params": ["dm_slow"]},
        "rm_macd_hist": {"name": "MACD Hist", "formula": "MACD(12,26) - Signal(9)", "params": []},
    }
    signal_rules = {
        "buy": "投票数 >= buy_threshold (4个子策略: MeanReversion/DualMA/RSI_MACD/VWAP)",
        "sell": "投票数 >= sell_threshold (4个子策略的卖出信号投票)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "dm_fast", "label": "MA Fast", "color": "#ff9800"},
            {"col": "dm_slow", "label": "MA Slow", "color": "#2196f3"},
            {"col": "mr_bb_upper", "label": "BB Upper", "color": "#555555", "dash": True},
            {"col": "mr_bb_lower", "label": "BB Lower", "color": "#555555", "dash": True},
        ],
        "subchart": [
            {"col": "mr_rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "rm_macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"},
        ],
    }
    default_params = {
        "atr_stop_mul": 2.0,
        # mean_reversion params
        "mr_rsi_period": 14, "mr_rsi_oversold": 30, "mr_rsi_overbought": 65,
        "mr_bb_period": 25, "mr_bb_std": 2.0,
        # dual_ma params
        "dm_fast": 10, "dm_slow": 25,
        # rsi_macd params
        "rm_rsi_entry": 30, "rm_rsi_exit": 80, "rm_ema_trend": 40,
        # vwap params
        "vw_period": 15, "vw_band_mul": 1.0,
        # voting
        "buy_threshold": 1, "sell_threshold": 2,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # Mean Reversion indicators
        df["mr_rsi"] = ta.momentum.rsi(c, window=params["mr_rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["mr_bb_period"],
                                          window_dev=params["mr_bb_std"])
        df["mr_bb_upper"] = bb.bollinger_hband()
        df["mr_bb_lower"] = bb.bollinger_lband()

        # Dual MA indicators
        df["dm_fast"] = c.rolling(params["dm_fast"]).mean()
        df["dm_slow"] = c.rolling(params["dm_slow"]).mean()

        # RSI+MACD indicators
        df["rm_rsi"] = df["mr_rsi"]  # reuse
        macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        df["rm_macd_hist"] = macd.macd_diff()
        df["rm_ema"] = ta.trend.ema_indicator(c, window=params["rm_ema_trend"])

        # VWAP indicators
        tp = (h + l + c) / 3
        df["vw_vwap"] = (tp * v).rolling(params["vw_period"]).sum() / v.rolling(params["vw_period"]).sum()
        atr = ta.volatility.average_true_range(h, l, c, window=14)
        df["vw_upper"] = df["vw_vwap"] + params["vw_band_mul"] * atr
        df["vw_ema50"] = ta.trend.ema_indicator(c, window=50)
        df["vol_ma"] = v.rolling(20).mean()
        df["vol_ratio"] = v / df["vol_ma"]

        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()

        # Strategy 1: Mean Reversion
        mr_buy = (df["mr_rsi"] < params["mr_rsi_oversold"]) & (df["close"] <= df["mr_bb_lower"])
        mr_sell = (df["mr_rsi"] > params["mr_rsi_overbought"]) | (df["close"] >= df["mr_bb_upper"])

        # Strategy 2: Dual MA Cross
        prev_fast = df["dm_fast"].shift(1)
        prev_slow = df["dm_slow"].shift(1)
        dm_buy = (prev_fast <= prev_slow) & (df["dm_fast"] > df["dm_slow"])
        dm_sell = (prev_fast >= prev_slow) & (df["dm_fast"] < df["dm_slow"])

        # Strategy 3: RSI+MACD Trend
        rm_rsi_ok = (df["rm_rsi"] > params["rm_rsi_entry"]) & (df["rm_rsi"] < params["rm_rsi_exit"])
        rm_macd_cross = (df["rm_macd_hist"] > 0) & (df["rm_macd_hist"].shift(1) <= 0)
        rm_buy = rm_rsi_ok & rm_macd_cross & (df["close"] > df["rm_ema"])
        rm_sell = (df["rm_rsi"] > params["rm_rsi_exit"]) | ((df["rm_macd_hist"] < 0) & (df["close"] < df["rm_ema"]))

        # Strategy 4: VWAP Breakout
        vw_buy = ((df["close"].shift(1) <= df["vw_upper"]) & (df["close"] > df["vw_upper"])
                  & (df["vol_ratio"] > 1.2) & (df["close"] > df["vw_ema50"]))
        vw_sell = df["close"] < df["vw_vwap"]

        # Voting
        buy_votes = mr_buy.astype(int) + dm_buy.astype(int) + rm_buy.astype(int) + vw_buy.astype(int)
        sell_votes = mr_sell.astype(int) + dm_sell.astype(int) + rm_sell.astype(int) + vw_sell.astype(int)

        df["buy_signal"] = buy_votes >= params["buy_threshold"]
        df["sell_signal"] = sell_votes >= params["sell_threshold"]
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 12. Ichimoku Cloud 一目均衡表 ─────────────────────────────────────

class IchimokuCloud(BaseStrategy):
    name = "claude_ichimoku_cloud"
    description = "一目均衡表：转换线/基准线交叉 + 云层确认 + 延迟线过滤"
    indicator_formulas = {
        "tenkan": {"name": "转换线", "formula": "(HH(N)+LL(N))/2", "params": ["tenkan_period"]},
        "kijun": {"name": "基准线", "formula": "(HH(N)+LL(N))/2", "params": ["kijun_period"]},
        "senkou_a": {"name": "先行A", "formula": "(Tenkan+Kijun)/2, shifted 26", "params": []},
        "senkou_b": {"name": "先行B", "formula": "(HH(52)+LL(52))/2, shifted 26", "params": ["senkou_b_period"]},
    }
    signal_rules = {
        "buy": "转换线>基准线 AND Close>云层上沿 AND Close>延迟线参考价",
        "sell": "转换线<基准线 OR Close<云层下沿",
    }
    chart_indicators = {
        "overlay": [
            {"col": "tenkan", "label": "转换线", "color": "#2196f3"},
            {"col": "kijun", "label": "基准线", "color": "#f44336"},
            {"col": "senkou_a", "label": "先行A", "color": "#4caf50", "dash": True},
            {"col": "senkou_b", "label": "先行B", "color": "#ff9800", "dash": True},
        ],
        "subchart": [],
    }
    default_params = {
        "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        tp = params["tenkan_period"]
        kp = params["kijun_period"]
        sp = params["senkou_b_period"]

        df["tenkan"] = (h.rolling(tp).max() + l.rolling(tp).min()) / 2
        df["kijun"]  = (h.rolling(kp).max() + l.rolling(kp).min()) / 2
        # 先行线不做 shift（因为回测用当前bar判断），直接用当前值
        df["senkou_a"] = (df["tenkan"] + df["kijun"]) / 2
        df["senkou_b"] = (h.rolling(sp).max() + l.rolling(sp).min()) / 2
        df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["cloud_bot"] = df[["senkou_a", "senkou_b"]].min(axis=1)
        # 延迟线参考：26日前的收盘价
        df["chikou_ref"] = c.shift(26)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：转换线>基准线 + 价格在云层上方 + 价格>26日前价格（延迟线确认）
        tk_cross = (df["tenkan"] > df["kijun"]) & (df["tenkan"].shift(1) <= df["kijun"].shift(1))
        above_cloud = df["close"] > df["cloud_top"]
        chikou_ok = df["close"] > df["chikou_ref"]
        df["buy_signal"] = tk_cross & above_cloud & chikou_ok
        # 卖出：转换线<基准线 或 价格跌破云层下沿
        df["sell_signal"] = ((df["tenkan"] < df["kijun"]) |
                             (df["close"] < df["cloud_bot"]))
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 13. Larry Williams 波动率突破 ────────────────────────────────────

class VolatilityBreakout(BaseStrategy):
    name = "claude_volatility_breakout"
    description = "波动率突破：Open + 昨日Range × K系数，趋势过滤"
    indicator_formulas = {
        "upper_band": {"name": "突破线", "formula": "Open + (High[-1]-Low[-1]) × K", "params": ["k_factor"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
        "range_ratio": {"name": "波幅比", "formula": "Range / ATR", "params": []},
    }
    signal_rules = {
        "buy": "Close > Open + Range[-1] × K AND Close > EMA_Trend AND Range_Ratio > min_range",
        "sell": "Close < EMA_Trend OR Close < 入场价 × (1 - stop_pct)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "upper_band", "label": "突破线", "color": "#f44336", "dash": True},
            {"col": "ema_trend", "label": "EMA Trend", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "range_ratio", "label": "Range Ratio", "color": "#ff9800"},
        ],
    }
    default_params = {
        "k_factor": 0.3, "ema_period": 50, "min_range": 0.3,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, o = df["high"], df["low"], df["close"], df["open"]
        prev_range = (h.shift(1) - l.shift(1))
        df["upper_band"] = o + prev_range * params["k_factor"]
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        df["range_ratio"] = (h - l) / df["atr"]
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：收盘突破波动率上轨 + 趋势向上 + 波幅够大
        df["buy_signal"] = ((df["close"] > df["upper_band"])
                            & (df["close"] > df["ema_trend"])
                            & (df["range_ratio"] > params["min_range"]))
        # 卖出：跌破趋势线
        df["sell_signal"] = df["close"] < df["ema_trend"]
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 14. Elder 三重滤网 ──────────────────────────────────────────────

class ElderTripleScreen(BaseStrategy):
    name = "claude_elder_triple_screen"
    description = "Elder三重滤网：周趋势(EMA13w) + 日Force Index + 入场确认"
    indicator_formulas = {
        "weekly_ema": {"name": "周趋势线", "formula": "EMA(Close, N×5)", "params": ["weekly_ema_period"]},
        "force_index": {"name": "Force Index", "formula": "EMA(ΔClose × Volume, N)", "params": ["force_period"]},
        "elder_ray_bull": {"name": "Bull Power", "formula": "High - EMA(Close, 13)", "params": []},
        "elder_ray_bear": {"name": "Bear Power", "formula": "Low - EMA(Close, 13)", "params": []},
    }
    signal_rules = {
        "buy": "Close > 周EMA(上升趋势) AND Force Index < 0(回调) AND Bull Power > 0",
        "sell": "Close < 周EMA OR Bear Power < prev Bear Power(恶化)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "weekly_ema", "label": "周EMA", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "force_index", "label": "Force Index", "color": "#2196f3", "type": "histogram"},
            {"col": "elder_ray_bull", "label": "Bull", "color": "#4caf50"},
            {"col": "elder_ray_bear", "label": "Bear", "color": "#f44336"},
        ],
    }
    default_params = {
        "weekly_ema_period": 13, "force_period": 2,
        "ema_short": 13,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        # 周级别趋势用日线EMA(65)模拟（13周 × 5交易日）
        weekly_len = params["weekly_ema_period"] * 5
        df["weekly_ema"] = ta.trend.ema_indicator(c, window=weekly_len)
        df["weekly_ema_prev"] = df["weekly_ema"].shift(1)
        # Force Index = EMA( (Close - Close[-1]) * Volume, N )
        raw_force = c.diff() * v
        df["force_index"] = raw_force.ewm(span=params["force_period"], adjust=False).mean()
        # Elder Ray
        ema_short = ta.trend.ema_indicator(c, window=params["ema_short"])
        df["elder_ray_bull"] = h - ema_short
        df["elder_ray_bear"] = l - ema_short
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 第一屏：周趋势上升（weekly_ema上升）
        weekly_up = df["weekly_ema"] > df["weekly_ema_prev"]
        price_above = df["close"] > df["weekly_ema"]
        # 第二屏：Force Index < 0（回调买入机会）
        force_pullback = df["force_index"] < 0
        # 第三屏：Bull Power > 0 确认
        bull_ok = df["elder_ray_bull"] > 0

        df["buy_signal"] = weekly_up & price_above & force_pullback & bull_ok
        # 卖出：周趋势转跌 或 Bear Power恶化
        df["sell_signal"] = ((df["close"] < df["weekly_ema"]) |
                             ((df["elder_ray_bear"] < df["elder_ray_bear"].shift(1)) &
                              (df["elder_ray_bear"] < 0)))
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 15. SuperTrend 策略 ─────────────────────────────────────────────

class SuperTrendStrategy(BaseStrategy):
    name = "claude_supertrend"
    description = "SuperTrend：ATR动态趋势带，突破翻转交易"
    indicator_formulas = {
        "supertrend": {"name": "SuperTrend", "formula": "HL2 ± Multiplier × ATR(N)", "params": ["st_period", "st_multiplier"]},
        "st_direction": {"name": "趋势方向", "formula": "1=上涨, -1=下跌", "params": []},
    }
    signal_rules = {
        "buy": "Close > SuperTrend (趋势翻转向上) AND Volume > MA(Vol)",
        "sell": "Close < SuperTrend (趋势翻转向下)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "supertrend", "label": "SuperTrend", "color": "#ff9800"},
        ],
        "subchart": [
            {"col": "st_direction", "label": "Direction", "color": "#2196f3", "type": "histogram"},
        ],
    }
    default_params = {
        "st_period": 14, "st_multiplier": 3.0,
        "vol_ma": 20,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        period = params["st_period"]
        mult = params["st_multiplier"]

        atr = ta.volatility.average_true_range(h, l, c, window=period)
        hl2 = (h + l) / 2
        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr

        # SuperTrend 计算
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        supertrend.iloc[period] = upper_band.iloc[period]
        direction.iloc[period] = -1

        for i in range(period + 1, len(df)):
            if direction.iloc[i-1] == 1:  # 之前是上涨
                # 下轨只能上移
                lb = max(lower_band.iloc[i], supertrend.iloc[i-1]) if not np.isnan(supertrend.iloc[i-1]) else lower_band.iloc[i]
                if c.iloc[i] < lb:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lb
                    direction.iloc[i] = 1
            else:  # 之前是下跌
                # 上轨只能下移
                ub = min(upper_band.iloc[i], supertrend.iloc[i-1]) if not np.isnan(supertrend.iloc[i-1]) else upper_band.iloc[i]
                if c.iloc[i] > ub:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = ub
                    direction.iloc[i] = -1

        df["supertrend"] = supertrend
        df["st_direction"] = direction
        df["atr"] = atr
        df["vol_ma"] = v.rolling(params["vol_ma"]).mean()
        df["vol_ratio"] = v / df["vol_ma"]
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：方向从-1翻转为1（趋势转多）
        dir_flip_up = (df["st_direction"] == 1) & (df["st_direction"].shift(1) == -1)
        df["buy_signal"] = dir_flip_up & (df["vol_ratio"] > 0.8)
        # 卖出：方向从1翻转为-1（趋势转空）
        df["sell_signal"] = (df["st_direction"] == -1) & (df["st_direction"].shift(1) == 1)
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 16. Ross Hook 分形策略 ──────────────────────────────────────────

class RossHook(BaseStrategy):
    name = "claude_ross_hook"
    description = "Ross Hook：1-2-3反转形态 + 钩子突破二次入场，分形识别"
    indicator_formulas = {
        "swing_high": {"name": "分形高点", "formula": "N-bar fractal high (左右各N根)", "params": ["fractal_n"]},
        "swing_low": {"name": "分形低点", "formula": "N-bar fractal low (左右各N根)", "params": ["fractal_n"]},
        "hook_level": {"name": "Hook Level", "formula": "看涨1-2-3的Point2值(突破目标)", "params": []},
        "ema_filter": {"name": "EMA趋势", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "看涨1-2-3形态确立(P1低→P2高→P3更高低) + Close突破P2 + EMA趋势向上",
        "sell": "Close < 最近分形低点 OR Close < EMA趋势线",
    }
    chart_indicators = {
        "overlay": [
            {"col": "swing_high_val", "label": "分形高点", "color": "#f44336", "dash": True},
            {"col": "swing_low_val", "label": "分形低点", "color": "#4caf50", "dash": True},
            {"col": "hook_level", "label": "Hook Level", "color": "#ff9800"},
            {"col": "ema_filter", "label": "EMA Trend", "color": "#9c27b0"},
        ],
        "subchart": [],
    }
    default_params = {
        "fractal_n": 2,          # 分形窗口：左右各N根K线
        "ema_period": 30,        # 趋势过滤EMA (WF optimized)
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        n = params["fractal_n"]
        highs = h.values
        lows = l.values
        closes = c.values
        length = len(df)

        # ── 步骤1: 分形识别 ──
        fractal_high = np.full(length, np.nan)
        fractal_low = np.full(length, np.nan)
        for i in range(n, length - n):
            is_fh = all(highs[i] > highs[i - j] and highs[i] > highs[i + j] for j in range(1, n + 1))
            if is_fh:
                fractal_high[i] = highs[i]
            is_fl = all(lows[i] < lows[i - j] and lows[i] < lows[i + j] for j in range(1, n + 1))
            if is_fl:
                fractal_low[i] = lows[i]

        # 前向填充用于图表
        sh_series = pd.Series(fractal_high, index=df.index).ffill()
        sl_series = pd.Series(fractal_low, index=df.index).ffill()
        df["swing_high_val"] = sh_series
        df["swing_low_val"] = sl_series

        # ── 步骤2: 状态机检测1-2-3 + Hook ──
        # 收集已确认的分形点序列
        swing_points = []  # list of (bar_index, value, 'H'|'L')
        for i in range(length):
            if not np.isnan(fractal_high[i]):
                swing_points.append((i, fractal_high[i], 'H'))
            if not np.isnan(fractal_low[i]):
                swing_points.append((i, fractal_low[i], 'L'))

        # 用滑动窗口在分形序列中找 L-H-L 模式 (看涨1-2-3)
        hook_level = np.full(length, np.nan)
        active_hook = np.nan  # 当前有效的hook水平

        sp_idx = 0  # swing_points cursor
        for i in range(length):
            # 推进cursor到当前bar已确认的分形
            while sp_idx < len(swing_points) and swing_points[sp_idx][0] <= i:
                sp_idx += 1

            # 在已确认的分形点中搜索最近的L-H-L看涨形态
            confirmed = swing_points[:sp_idx]  # 到目前为止确认的分形
            if len(confirmed) >= 3:
                # 检查最近的几组连续分形
                for k in range(len(confirmed) - 1, 1, -1):
                    p3_idx, p3_val, p3_type = confirmed[k]
                    p2_idx, p2_val, p2_type = confirmed[k - 1]
                    p1_idx, p1_val, p1_type = confirmed[k - 2]

                    # 看涨 1-2-3: Low(P1) → High(P2) → Higher-Low(P3)
                    if (p1_type == 'L' and p2_type == 'H' and p3_type == 'L'
                            and p3_val > p1_val    # P3 > P1 (更高低)
                            and p2_val > p3_val):  # P2是高点
                        active_hook = p2_val
                        break

                    # 只检查最近3组，避免太旧的形态
                    if k < len(confirmed) - 6:
                        break

            if not np.isnan(active_hook):
                hook_level[i] = active_hook
                # 如果价格已大幅超过hook，失效（防止陈旧信号）
                if closes[i] > active_hook * 1.15:
                    active_hook = np.nan

        df["hook_level"] = hook_level

        # EMA趋势过滤
        df["ema_filter"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()

        # 买入：Close突破hook_level + 前一日在hook下方 + EMA趋势向上
        has_hook = df["hook_level"].notna()
        breakout = df["close"] > df["hook_level"]
        was_below = df["high"].shift(1) <= df["hook_level"]
        trend_up = df["close"] > df["ema_filter"]

        df["buy_signal"] = has_hook & breakout & was_below & trend_up

        # 卖出：跌破最近分形低点 或 跌破EMA
        df["sell_signal"] = ((df["close"] < df["swing_low_val"]) |
                             (df["close"] < df["ema_filter"]))
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 17. Swing-Point Divergence (RSI + MACD + Stochastic) ──────────────

class DivergenceCombo(BaseStrategy):
    name = "claude_divergence_combo"
    description = "摆动点背离：在真实Pivot高/低点之间检测RSI/MACD/Stoch背离，精确买卖"
    indicator_formulas = {
        "rsi": {"name": "RSI", "formula": "RSI(C, rsi_period)", "params": ["rsi_period"]},
        "macd_hist": {"name": "MACD Hist", "formula": "MACD(fast,slow) - Signal(signal_w)", "params": ["macd_fast", "macd_slow"]},
        "stoch_k": {"name": "Stoch %K", "formula": "STO(high,low,close,k_period)", "params": ["stoch_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(C, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "底背离：Pivot Low更低但RSI/MACD/Stoch不更低(≥div_confirm确认) + MACD柱翻升 + RSI<rsi_zone_low",
        "sell": "顶背离：Pivot High更高但RSI/MACD/Stoch不更高(≥div_confirm确认) + MACD柱翻降 + RSI>rsi_zone_high 或ATR止损",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"},
            {"col": "pivot_low_price", "label": "Pivot Low", "color": "#4caf50", "dash": True},
            {"col": "pivot_high_price", "label": "Pivot High", "color": "#f44336", "dash": True},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"},
        ],
    }
    default_params = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal_w": 9,
        "stoch_period": 14,
        "pivot_n": 3,            # Pivot检测：左右各N根K线
        "max_pivot_dist": 60,    # 两个Pivot之间最大间隔(bars)
        "ema_period": 30,        # 趋势过滤EMA
        "div_confirm": 1,        # 需要几个指标同时确认背离 (1-3)
        "rsi_zone_low": 50,      # 买入时RSI需低于此值
        "rsi_zone_high": 50,     # 卖出时RSI需高于此值
        "atr_stop_mul": 2.0,
    }

    @staticmethod
    def _find_pivots(series_low, series_high, n):
        """找到所有Pivot Low和Pivot High的索引和值"""
        pivot_lows = {}   # idx -> value
        pivot_highs = {}  # idx -> value
        vals_low = series_low.values
        vals_high = series_high.values
        length = len(vals_low)

        for i in range(n, length - n):
            # Pivot Low: low[i] 是左右各n根中最低
            is_pivot_low = True
            for j in range(1, n + 1):
                if vals_low[i] > vals_low[i - j] or vals_low[i] > vals_low[i + j]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows[i] = vals_low[i]

            # Pivot High: high[i] 是左右各n根中最高
            is_pivot_high = True
            for j in range(1, n + 1):
                if vals_high[i] < vals_high[i - j] or vals_high[i] < vals_high[i + j]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs[i] = vals_high[i]

        return pivot_lows, pivot_highs

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]

        # 基础指标
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        macd_ind = ta.trend.MACD(c, window_slow=params["macd_slow"],
                                  window_fast=params["macd_fast"],
                                  window_sign=params["macd_signal_w"])
        df["macd_hist"] = macd_ind.macd_diff()
        df["macd_line"] = macd_ind.macd()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(h, l, c, window=params["stoch_period"],
                                                  smooth_window=3)
        df["stoch_k"] = stoch.stoch()

        # 趋势过滤
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)

        # 找Pivot点并编码为列
        n = params["pivot_n"]
        pivot_lows, pivot_highs = self._find_pivots(l, h, n)

        # 标记pivot位置 (0=不是pivot, 价格值=是pivot)
        df["_is_pivot_low"] = 0.0
        df["_is_pivot_high"] = 0.0
        for idx, val in pivot_lows.items():
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc("_is_pivot_low")] = val
        for idx, val in pivot_highs.items():
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc("_is_pivot_high")] = val

        # 图表显示用
        df["pivot_low_price"] = df["_is_pivot_low"].replace(0.0, np.nan).ffill()
        df["pivot_high_price"] = df["_is_pivot_high"].replace(0.0, np.nan).ffill()

        # 只删除核心指标（RSI/MACD/EMA）的NaN行（即预热期），不要因为pivot列的NaN删除中间行
        core_cols = ["rsi", "macd_hist", "stoch_k", "ema_trend"]
        return df.dropna(subset=core_cols)

    def generate_signals(self, df, params):
        df = df.copy()
        need = params["div_confirm"]
        max_dist = params["max_pivot_dist"]
        rsi_lo = params["rsi_zone_low"]
        rsi_hi = params["rsi_zone_high"]

        # Reconstruct pivot dicts from encoded columns
        pivot_lows = {}
        pivot_highs = {}
        plow_vals = df["_is_pivot_low"].values
        phigh_vals = df["_is_pivot_high"].values
        for i in range(len(plow_vals)):
            if plow_vals[i] != 0.0:
                pivot_lows[i] = plow_vals[i]
            if phigh_vals[i] != 0.0:
                pivot_highs[i] = phigh_vals[i]

        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)

        rsi_vals = df["rsi"].values
        macd_vals = df["macd_hist"].values
        stoch_vals = df["stoch_k"].values
        close_vals = df["close"].values
        low_vals = df["low"].values
        high_vals = df["high"].values

        # 把pivot索引转换为排序列表
        pl_indices = sorted(pivot_lows.keys())
        ph_indices = sorted(pivot_highs.keys())

        # ═══ 底背离：连续两个Pivot Low，价格更低但指标更高 ═══
        for i in range(1, len(pl_indices)):
            curr_idx = pl_indices[i]
            prev_idx = pl_indices[i - 1]

            if curr_idx - prev_idx > max_dist:
                continue

            # 价格创新低（当前pivot low < 前一个）
            if pivot_lows[curr_idx] >= pivot_lows[prev_idx]:
                continue

            # 检查各指标是否"没有创新低"（底背离）
            # 取pivot附近±2范围的指标极值来更准确
            def get_indicator_at_pivot(vals, idx, radius=2):
                start = max(0, idx - radius)
                end = min(len(vals), idx + radius + 1)
                return np.min(vals[start:end])  # 低点取最小

            curr_rsi = get_indicator_at_pivot(rsi_vals, curr_idx)
            prev_rsi = get_indicator_at_pivot(rsi_vals, prev_idx)
            curr_macd = get_indicator_at_pivot(macd_vals, curr_idx)
            prev_macd = get_indicator_at_pivot(macd_vals, prev_idx)
            curr_stoch = get_indicator_at_pivot(stoch_vals, curr_idx)
            prev_stoch = get_indicator_at_pivot(stoch_vals, prev_idx)

            bull_count = 0
            if curr_rsi > prev_rsi:
                bull_count += 1
            if curr_macd > prev_macd:
                bull_count += 1
            if curr_stoch > prev_stoch:
                bull_count += 1

            if bull_count >= need:
                # 在pivot之后的几根bar中寻找MACD柱翻升作为入场点
                n = params["pivot_n"]
                for bar in range(curr_idx + n, min(curr_idx + n + 8, len(df))):
                    if bar >= len(macd_vals) or bar >= len(rsi_vals):
                        break
                    # MACD柱翻升 + RSI在低区
                    if (macd_vals[bar] > macd_vals[bar - 1] and
                        rsi_vals[bar] < rsi_lo and
                        not buy_signals.iloc[bar]):
                        buy_signals.iloc[bar] = True
                        break

        # ═══ 顶背离：连续两个Pivot High，价格更高但指标更低 ═══
        for i in range(1, len(ph_indices)):
            curr_idx = ph_indices[i]
            prev_idx = ph_indices[i - 1]

            if curr_idx - prev_idx > max_dist:
                continue

            # 价格创新高
            if pivot_highs[curr_idx] <= pivot_highs[prev_idx]:
                continue

            def get_indicator_at_pivot_high(vals, idx, radius=2):
                start = max(0, idx - radius)
                end = min(len(vals), idx + radius + 1)
                return np.max(vals[start:end])  # 高点取最大

            curr_rsi = get_indicator_at_pivot_high(rsi_vals, curr_idx)
            prev_rsi = get_indicator_at_pivot_high(rsi_vals, prev_idx)
            curr_macd = get_indicator_at_pivot_high(macd_vals, curr_idx)
            prev_macd = get_indicator_at_pivot_high(macd_vals, prev_idx)
            curr_stoch = get_indicator_at_pivot_high(stoch_vals, curr_idx)
            prev_stoch = get_indicator_at_pivot_high(stoch_vals, prev_idx)

            bear_count = 0
            if curr_rsi < prev_rsi:
                bear_count += 1
            if curr_macd < prev_macd:
                bear_count += 1
            if curr_stoch < prev_stoch:
                bear_count += 1

            if bear_count >= need:
                n = params["pivot_n"]
                for bar in range(curr_idx + n, min(curr_idx + n + 8, len(df))):
                    if bar >= len(macd_vals) or bar >= len(rsi_vals):
                        break
                    # MACD柱翻降 + RSI在高区
                    if (macd_vals[bar] < macd_vals[bar - 1] and
                        rsi_vals[bar] > rsi_hi and
                        not sell_signals.iloc[bar]):
                        sell_signals.iloc[bar] = True
                        break

        df["buy_signal"] = buy_signals
        df["sell_signal"] = sell_signals
        return df


# ─── 18. 趋势持有（高持仓率趋势跟随）──────────────────────────────────

class TrendHold(BaseStrategy):
    name = "claude_trend_hold"
    description = "趋势持有：EMA200趋势过滤，回调至EMA20买入，仅在强反转信号时卖出（高持仓率）"
    indicator_formulas = {
        "ema_short": {"name": "EMA20", "formula": "EMA(close, 20)", "params": ["ema_short"]},
        "ema_mid": {"name": "EMA50", "formula": "EMA(close, 50)", "params": ["ema_mid"]},
        "ema_long": {"name": "EMA200", "formula": "EMA(close, 200)", "params": ["ema_long"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, 14)", "params": ["rsi_period"]},
    }
    signal_rules = {
        "buy": "EMA50>EMA200 AND Close>EMA200 AND 回调至EMA20附近(1.5×ATR) AND RSI>40",
        "sell": "连续3日收盘低于EMA50 AND RSI<35",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_short", "label": "EMA20", "color": "#ff9800"},
            {"col": "ema_mid", "label": "EMA50", "color": "#2196f3"},
            {"col": "ema_long", "label": "EMA200", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "ema_short": 20, "ema_mid": 50, "ema_long": 200,
        "rsi_period": 14, "rsi_entry_min": 40, "rsi_exit_low": 35,
        "pullback_atr_mul": 1.5,
        "sell_confirm_days": 3,
        "atr_stop_mul": 3.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["ema_short"] = ta.trend.ema_indicator(c, window=params["ema_short"])
        df["ema_mid"] = ta.trend.ema_indicator(c, window=params["ema_mid"])
        df["ema_long"] = ta.trend.ema_indicator(c, window=params["ema_long"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        below_mid = (c < df["ema_mid"]).astype(int)
        consec = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            consec.iloc[i] = consec.iloc[i-1] + 1 if below_mid.iloc[i] == 1 else 0
        df["below_mid_consec"] = consec
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        uptrend = (df["ema_mid"] > df["ema_long"]) & (df["close"] > df["ema_long"])
        near_ema20 = (df["close"] - df["ema_short"]).abs() < params["pullback_atr_mul"] * df["atr"]
        rsi_ok = df["rsi"] > params["rsi_entry_min"]
        df["buy_signal"] = uptrend & near_ema20 & rsi_ok
        df["sell_signal"] = ((df["below_mid_consec"] >= params["sell_confirm_days"])
                             & (df["rsi"] < params["rsi_exit_low"]))
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 19. 动量均值回归混合 ─────────────────────────────────────────────

class MomentumReversionHybrid(BaseStrategy):
    name = "claude_momentum_reversion_hybrid"
    description = "动量均值回归混合：ADX判断趋势/震荡，趋势市追涨，震荡市低吸高抛"
    indicator_formulas = {
        "adx": {"name": "ADX", "formula": "ADX(H,L,C,N)", "params": ["adx_period"]},
        "ema_fast": {"name": "EMA Fast", "formula": "EMA(close, N)", "params": ["ema_fast"]},
        "ema_slow": {"name": "EMA Slow", "formula": "EMA(close, N)", "params": ["ema_slow"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
        "bb_upper": {"name": "BB Upper", "formula": "SMA+K*Std", "params": ["bb_period", "bb_std"]},
        "bb_lower": {"name": "BB Lower", "formula": "SMA-K*Std", "params": ["bb_period", "bb_std"]},
    }
    signal_rules = {
        "buy": "趋势(ADX>25): EMA快>EMA慢+DI+>DI- | 震荡: RSI<35+Close≤BB下轨",
        "sell": "趋势: Close<EMA慢 | 震荡: RSI>75或Close≥BB上轨",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#ff9800"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#2196f3"},
            {"col": "bb_upper", "label": "BB Upper", "color": "#555555"},
            {"col": "bb_lower", "label": "BB Lower", "color": "#555555"},
        ],
        "subchart": [
            {"col": "adx", "label": "ADX", "color": "#9c27b0"},
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "adx_period": 14, "adx_threshold": 25,
        "ema_fast": 12, "ema_slow": 50,
        "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 75,
        "bb_period": 20, "bb_std": 2.0,
        "atr_stop_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["adx"] = ta.trend.adx(h, l, c, window=params["adx_period"])
        df["di_plus"] = ta.trend.adx_pos(h, l, c, window=params["adx_period"])
        df["di_minus"] = ta.trend.adx_neg(h, l, c, window=params["adx_period"])
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        trending = df["adx"] > params["adx_threshold"]
        ranging = ~trending
        mom_buy = (trending & (df["ema_fast"] > df["ema_slow"])
                   & (df["close"] > df["ema_fast"]) & (df["di_plus"] > df["di_minus"]))
        mom_sell = trending & (df["close"] < df["ema_slow"])
        mr_buy = (ranging & (df["rsi"] < params["rsi_oversold"])
                  & (df["close"] <= df["bb_lower"]))
        mr_sell = (ranging & ((df["rsi"] > params["rsi_overbought"])
                              | (df["close"] >= df["bb_upper"])))
        df["buy_signal"] = (mom_buy | mr_buy).fillna(False)
        df["sell_signal"] = (mom_sell | mr_sell).fillna(False)
        return df


# ─── 20. 突破追踪止损 ────────────────────────────────────────────────

class BreakoutTrailing(BaseStrategy):
    name = "claude_breakout_trailing"
    description = "突破追踪：N日新高突破入场，ATR追踪止损紧跟趋势，仅止损触发时卖出"
    indicator_formulas = {
        "highest": {"name": "N日最高价", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
        "trailing_stop": {"name": "追踪止损", "formula": "RollingMax(Close,N) - trail_mul×ATR", "params": ["trail_atr_mul"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "Close > N日最高价 AND Close > EMA_Trend AND 成交量比>0.8",
        "sell": "Close < 追踪止损（ATR trailing stop触发）",
    }
    chart_indicators = {
        "overlay": [
            {"col": "highest", "label": "N日高点", "color": "#f44336"},
            {"col": "trailing_stop", "label": "Trailing Stop", "color": "#4caf50"},
            {"col": "ema_trend", "label": "EMA Trend", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "vol_ratio", "label": "Vol Ratio", "color": "#ff9800"},
        ],
    }
    default_params = {
        "breakout_period": 20, "trail_atr_mul": 3.0,
        "ema_period": 50, "vol_ma": 20, "vol_ratio_min": 0.8,
        "atr_stop_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        df["highest"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        df["vol_ma"] = v.rolling(params["vol_ma"]).mean()
        df["vol_ratio"] = v / df["vol_ma"]
        rolling_high = c.rolling(params["breakout_period"]).max()
        df["trailing_stop"] = rolling_high - params["trail_atr_mul"] * df["atr"]
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"] = ((df["close"] > df["highest"])
                            & (df["close"] > df["ema_trend"])
                            & (df["vol_ratio"] > params["vol_ratio_min"]))
        df["sell_signal"] = df["close"] < df["trailing_stop"]
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 21. 多周期趋势 ─────────────────────────────────────────────────

class MultiTimeframeTrend(BaseStrategy):
    name = "claude_multi_timeframe_trend"
    description = "多周期趋势：周线趋势定方向(EMA65)，日线EMA20回调定入场，周趋势翻空才卖出"
    indicator_formulas = {
        "weekly_ema": {"name": "周EMA", "formula": "EMA(close, 65)", "params": ["weekly_ema_period"]},
        "daily_ema": {"name": "日EMA20", "formula": "EMA(close, 20)", "params": ["daily_ema"]},
        "daily_ema_fast": {"name": "日EMA8", "formula": "EMA(close, 8)", "params": ["daily_ema_fast"]},
        "macd_hist": {"name": "MACD Hist", "formula": "MACD-Signal", "params": []},
    }
    signal_rules = {
        "buy": "周EMA上升 AND Close>周EMA AND 回调至日EMA20附近 AND MACD转正或快EMA>慢EMA",
        "sell": "周EMA下降 AND Close<周EMA（周趋势破位）",
    }
    chart_indicators = {
        "overlay": [
            {"col": "weekly_ema", "label": "周EMA65", "color": "#9c27b0"},
            {"col": "daily_ema", "label": "日EMA20", "color": "#2196f3"},
            {"col": "daily_ema_fast", "label": "日EMA8", "color": "#ff9800"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4"},
        ],
    }
    default_params = {
        "weekly_ema_period": 65, "daily_ema": 20, "daily_ema_fast": 8,
        "rsi_period": 14, "pullback_tolerance": 0.02,
        "atr_stop_mul": 3.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["weekly_ema"] = ta.trend.ema_indicator(c, window=params["weekly_ema_period"])
        df["weekly_ema_prev"] = df["weekly_ema"].shift(5)
        df["daily_ema"] = ta.trend.ema_indicator(c, window=params["daily_ema"])
        df["daily_ema_fast"] = ta.trend.ema_indicator(c, window=params["daily_ema_fast"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        df["macd_hist"] = macd.macd_diff()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        tol = params["pullback_tolerance"]
        weekly_up = (df["weekly_ema"] > df["weekly_ema_prev"]) & (df["close"] > df["weekly_ema"])
        pullback_zone = ((df["close"] - df["daily_ema"]) / df["daily_ema"]).abs() < tol
        bounce = (df["low"] <= df["daily_ema"] * (1 + tol)) & (df["close"] > df["daily_ema"])
        momentum_ok = ((df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)) | (df["daily_ema_fast"] > df["daily_ema"])
        df["buy_signal"] = weekly_up & (pullback_zone | bounce) & momentum_ok
        weekly_broken = (df["weekly_ema"] < df["weekly_ema_prev"]) & (df["close"] < df["weekly_ema"])
        df["sell_signal"] = weekly_broken
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df


# ─── 22. 自适应市场状态 ──────────────────────────────────────────────

class AdaptiveRegime(BaseStrategy):
    name = "claude_adaptive_regime"
    description = "自适应市场状态：ADX/DI检测牛/熊/震荡，牛市持有不动，熊市空仓，震荡均值回归"
    indicator_formulas = {
        "adx": {"name": "ADX", "formula": "ADX(H,L,C,N)", "params": ["adx_period"]},
        "di_plus": {"name": "DI+", "formula": "+DM/ATR", "params": ["adx_period"]},
        "di_minus": {"name": "DI-", "formula": "-DM/ATR", "params": ["adx_period"]},
        "ema_long": {"name": "EMA100", "formula": "EMA(close, 100)", "params": ["ema_long"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
        "regime": {"name": "Regime", "formula": "1=Bull, 0=Sideways, -1=Bear", "params": []},
    }
    signal_rules = {
        "buy": "牛市: Close>EMA100(持有不动) | 震荡: RSI<35+Close<BB下轨",
        "sell": "熊市: DI->DI++ADX>阈值 | 震荡: RSI>70或Close≥BB上轨",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_long", "label": "EMA100", "color": "#9c27b0"},
            {"col": "bb_upper", "label": "BB Upper", "color": "#555555"},
            {"col": "bb_lower", "label": "BB Lower", "color": "#555555"},
        ],
        "subchart": [
            {"col": "adx", "label": "ADX", "color": "#2196f3"},
            {"col": "regime", "label": "Regime", "color": "#ff9800"},
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "adx_period": 14, "adx_trend_threshold": 20,
        "ema_long": 100,
        "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 70,
        "bb_period": 20, "bb_std": 2.0,
        "bull_rsi_floor": 30,
        "atr_stop_mul": 4.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["adx"] = ta.trend.adx(h, l, c, window=params["adx_period"])
        df["di_plus"] = ta.trend.adx_pos(h, l, c, window=params["adx_period"])
        df["di_minus"] = ta.trend.adx_neg(h, l, c, window=params["adx_period"])
        df["ema_long"] = ta.trend.ema_indicator(c, window=params["ema_long"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        # 市场状态识别
        adx_t = params["adx_trend_threshold"]
        regime = pd.Series(0, index=df.index, dtype=int)
        bull = (df["adx"] > adx_t) & (df["di_plus"] > df["di_minus"]) & (c > df["ema_long"])
        bear = (df["adx"] > adx_t) & (df["di_minus"] > df["di_plus"]) & (c < df["ema_long"])
        regime[bull] = 1
        regime[bear] = -1
        # 平滑：连续3天同方向才切换
        smoothed = regime.copy()
        for i in range(3, len(regime)):
            if regime.iloc[i] != smoothed.iloc[i-1]:
                if regime.iloc[i] == regime.iloc[i-1] == regime.iloc[i-2]:
                    smoothed.iloc[i] = regime.iloc[i]
                else:
                    smoothed.iloc[i] = smoothed.iloc[i-1]
        df["regime"] = smoothed
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        is_bull = df["regime"] == 1
        is_bear = df["regime"] == -1
        is_sideways = df["regime"] == 0
        bull_buy = is_bull & (df["close"] > df["ema_long"])
        bull_sell = is_bull & (df["rsi"] < params["bull_rsi_floor"])
        bear_sell = is_bear
        side_buy = is_sideways & (df["rsi"] < params["rsi_oversold"]) & (df["close"] <= df["bb_lower"])
        side_sell = is_sideways & ((df["rsi"] > params["rsi_overbought"]) | (df["close"] >= df["bb_upper"]))
        df["buy_signal"] = (bull_buy | side_buy).fillna(False)
        df["sell_signal"] = (bull_sell | bear_sell | side_sell).fillna(False)
        return df


# ─── 23. OBV量价动量 ────────────────────────────────────────────────

class ObvMomentum(BaseStrategy):
    name = "claude_obv_momentum"
    description = "OBV量价动量：OBV突破均线+价格趋势确认，量价齐升买入，量价背离卖出"
    indicator_formulas = {
        "obv": {"name": "OBV", "formula": "累计(volume × sign(close-prev_close))", "params": []},
        "obv_ema": {"name": "OBV EMA", "formula": "EMA(OBV, N)", "params": ["obv_ema_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_trend_period"]},
    }
    signal_rules = {
        "buy": "OBV > OBV_EMA 且 OBV斜率>0 且 Close > EMA趋势线 且 RSI < 超买线",
        "sell": "OBV < OBV_EMA 或 Close < EMA趋势线 或 RSI > 超卖线",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#2196f3", "type": "line"},
        ],
        "subchart": [
            {"col": "obv_norm", "label": "OBV", "color": "#ff9800", "type": "line"},
            {"col": "obv_ema_norm", "label": "OBV EMA", "color": "#9c27b0", "type": "line"},
            {"col": "rsi", "label": "RSI", "color": "#e91e63", "type": "line"},
        ],
    }
    default_params = {
        "obv_ema_period": 20,
        "ema_trend_period": 50,
        "obv_slope_len": 5,
        "rsi_period": 14,
        "rsi_overbought": 75,
        "rsi_oversold": 30,
        "volume_ratio_min": 1.2,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, v = df["close"], df["volume"]
        # OBV
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_ema"] = ta.trend.ema_indicator(obv, window=params["obv_ema_period"])
        # OBV斜率 (N日变化率)
        slope_len = params["obv_slope_len"]
        df["obv_slope"] = obv.diff(slope_len) / (obv.rolling(slope_len).mean().abs() + 1e-10)
        # 归一化OBV用于图表显示
        obv_min, obv_max = obv.rolling(100, min_periods=20).min(), obv.rolling(100, min_periods=20).max()
        df["obv_norm"] = (obv - obv_min) / (obv_max - obv_min + 1e-10) * 100
        df["obv_ema_norm"] = (df["obv_ema"] - obv_min) / (obv_max - obv_min + 1e-10) * 100
        # 趋势线
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_trend_period"])
        # RSI
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        # 成交量比率
        df["vol_ratio"] = v / v.rolling(20).mean()
        # ATR (回测引擎需要)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        obv_above = df["obv"] > df["obv_ema"]
        obv_rising = df["obv_slope"] > 0
        price_above = df["close"] > df["ema_trend"]
        rsi_ok = df["rsi"] < params["rsi_overbought"]
        vol_ok = df["vol_ratio"] > params["volume_ratio_min"]

        buy = obv_above & obv_rising & price_above & rsi_ok & vol_ok
        sell = (~obv_above) | (~price_above) | (df["rsi"] > params["rsi_overbought"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── 24. Keltner通道突破 ────────────────────────────────────────────

class KeltnerBreakout(BaseStrategy):
    name = "claude_keltner_breakout"
    description = "Keltner通道突破：价格突破上轨+ATR过滤假突破+EMA趋势方向确认"
    indicator_formulas = {
        "kelt_mid": {"name": "Keltner Mid", "formula": "EMA(close, N)", "params": ["kelt_period"]},
        "kelt_upper": {"name": "Keltner Upper", "formula": "EMA + K × ATR", "params": ["kelt_period", "kelt_atr_mul"]},
        "kelt_lower": {"name": "Keltner Lower", "formula": "EMA - K × ATR", "params": ["kelt_period", "kelt_atr_mul"]},
        "atr": {"name": "ATR", "formula": "ATR(H,L,C, N)", "params": ["atr_period"]},
    }
    signal_rules = {
        "buy": "Close突破Keltner上轨 且 EMA趋势向上 且 ATR > ATR均线(波动放大)",
        "sell": "Close跌破Keltner中线 或 Close < 入场价 - stop_atr_mul × ATR",
    }
    chart_indicators = {
        "overlay": [
            {"col": "kelt_upper", "label": "Keltner Upper", "color": "#ff5722", "type": "line"},
            {"col": "kelt_mid", "label": "Keltner Mid", "color": "#2196f3", "type": "line"},
            {"col": "kelt_lower", "label": "Keltner Lower", "color": "#4caf50", "type": "line"},
            {"col": "ema_trend", "label": "EMA Trend", "color": "#9c27b0", "type": "line"},
        ],
        "subchart": [
            {"col": "atr", "label": "ATR", "color": "#ff9800", "type": "line"},
            {"col": "atr_ma", "label": "ATR MA", "color": "#795548", "type": "line"},
        ],
    }
    default_params = {
        "kelt_period": 20,
        "kelt_atr_mul": 2.0,
        "atr_period": 14,
        "ema_trend_period": 50,
        "atr_filter_period": 20,
        "stop_atr_mul": 2.5,
        "trail_atr_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        # Keltner Channel
        kelt_mid = ta.trend.ema_indicator(c, window=params["kelt_period"])
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["kelt_mid"] = kelt_mid
        df["kelt_upper"] = kelt_mid + params["kelt_atr_mul"] * atr
        df["kelt_lower"] = kelt_mid - params["kelt_atr_mul"] * atr
        df["atr"] = atr
        df["atr_ma"] = atr.rolling(params["atr_filter_period"]).mean()
        # 趋势线
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_trend_period"])
        df["ema_slope"] = df["ema_trend"].diff(5) / df["ema_trend"].shift(5) * 100
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        # 突破上轨
        breakout = (c > df["kelt_upper"]) & (c.shift(1) <= df["kelt_upper"].shift(1))
        # 趋势向上
        trend_up = df["ema_slope"] > 0
        # 波动放大 (ATR高于均值)
        vol_expand = df["atr"] > df["atr_ma"]
        buy = breakout & trend_up & vol_expand
        # 卖出: 跌破中线
        sell = c < df["kelt_mid"]
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── 25. Z-Score均值回归 ────────────────────────────────────────────

class ZScoreReversion(BaseStrategy):
    name = "claude_zscore_reversion"
    description = "Z-Score均值回归：价格偏离均值超过阈值时逆向交易，趋势过滤避免逆势"
    indicator_formulas = {
        "zscore": {"name": "Z-Score", "formula": "(Close - SMA) / StdDev", "params": ["lookback"]},
        "sma": {"name": "SMA", "formula": "SMA(close, N)", "params": ["lookback"]},
        "ema_filter": {"name": "EMA Filter", "formula": "EMA(close, N)", "params": ["ema_filter_period"]},
    }
    signal_rules = {
        "buy": "Z-Score < -entry_z (超卖) 且 Close > EMA趋势线 (非熊市过滤)",
        "sell": "Z-Score > exit_z (回归均值) 或 Z-Score > stop_z (极端超买止损)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "sma", "label": "SMA", "color": "#2196f3", "type": "line"},
            {"col": "ema_filter", "label": "EMA Filter", "color": "#9c27b0", "type": "line"},
            {"col": "upper_band", "label": "Entry Band+", "color": "#ff5722", "type": "line"},
            {"col": "lower_band", "label": "Entry Band-", "color": "#4caf50", "type": "line"},
        ],
        "subchart": [
            {"col": "zscore", "label": "Z-Score", "color": "#ff9800", "type": "line"},
        ],
    }
    default_params = {
        "lookback": 40,
        "entry_z": 1.5,
        "exit_z": 0.3,
        "stop_z": 3.0,
        "ema_filter_period": 100,
        "rsi_period": 14,
        "rsi_floor": 25,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        lb = params["lookback"]
        sma = c.rolling(lb).mean()
        std = c.rolling(lb).std()
        df["sma"] = sma
        df["zscore"] = (c - sma) / (std + 1e-10)
        df["upper_band"] = sma + params["entry_z"] * std
        df["lower_band"] = sma - params["entry_z"] * std
        df["ema_filter"] = ta.trend.ema_indicator(c, window=params["ema_filter_period"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        # ATR (回测引擎需要)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        z = df["zscore"]
        # 买入: 超卖回归 + 趋势过滤
        trend_ok = df["close"] > df["ema_filter"]
        rsi_ok = df["rsi"] > params["rsi_floor"]
        buy = (z < -params["entry_z"]) & trend_ok & rsi_ok
        # 卖出: 回归均值 或 极端超买
        sell = (z > params["exit_z"]) | (z > params["stop_z"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── 26. 唐奇安通道突破 ─────────────────────────────────────────────

class DonchianBreakout(BaseStrategy):
    name = "claude_donchian_breakout"
    description = "唐奇安通道突破：N日最高/最低价三带通道，ADX趋势过滤+ATR追踪止损，区别于海龟策略的精细化版本"
    indicator_formulas = {
        "dc_upper": {"name": "唐奇安上轨", "formula": "Max(High[-1], N)", "params": ["dc_period"]},
        "dc_mid": {"name": "唐奇安中线", "formula": "(dc_upper + dc_lower) / 2", "params": []},
        "dc_lower": {"name": "唐奇安下轨", "formula": "Min(Low[-1], N)", "params": ["dc_period"]},
        "adx": {"name": "ADX", "formula": "ADX(H,L,C, adx_period)", "params": ["adx_period"]},
        "atr": {"name": "ATR", "formula": "ATR(H,L,C, 14)", "params": []},
    }
    signal_rules = {
        "buy": "Close突破dc_upper 且 ADX > adx_min(趋势确认) 且 成交量比 > vol_ratio_min",
        "sell": "Close跌破dc_mid(通道中线止损) 或 Close < 追踪止损(peak - trail_atr_mul × ATR)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "dc_upper", "label": "唐奇安上轨", "color": "#f44336", "dash": True},
            {"col": "dc_mid",   "label": "唐奇安中线", "color": "#78909c"},
            {"col": "dc_lower", "label": "唐奇安下轨", "color": "#4caf50", "dash": True},
            {"col": "trail_stop", "label": "追踪止损", "color": "#ff9800"},
        ],
        "subchart": [
            {"col": "adx", "label": "ADX", "color": "#9c27b0"},
            {"col": "vol_ratio", "label": "Vol Ratio", "color": "#2196f3"},
        ],
    }
    default_params = {
        "dc_period": 20,
        "adx_period": 14,
        "adx_min": 20,
        "vol_ma": 20,
        "vol_ratio_min": 0.8,
        "trail_atr_mul": 2.5,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        # 唐奇安三带：用shift(1)避免当前bar泄露
        df["dc_upper"] = h.shift(1).rolling(params["dc_period"]).max()
        df["dc_lower"] = l.shift(1).rolling(params["dc_period"]).min()
        df["dc_mid"]   = (df["dc_upper"] + df["dc_lower"]) / 2
        # ADX趋势强度
        df["adx"]      = ta.trend.adx(h, l, c, window=params["adx_period"])
        df["di_plus"]  = ta.trend.adx_pos(h, l, c, window=params["adx_period"])
        df["di_minus"] = ta.trend.adx_neg(h, l, c, window=params["adx_period"])
        # ATR与追踪止损
        df["atr"]      = ta.volatility.average_true_range(h, l, c, window=14)
        rolling_peak   = c.rolling(params["dc_period"]).max()
        df["trail_stop"] = rolling_peak - params["trail_atr_mul"] * df["atr"]
        # 成交量
        df["vol_ma"]   = v.rolling(params["vol_ma"]).mean()
        df["vol_ratio"] = v / df["vol_ma"]
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：突破上轨 + ADX趋势有效 + DI+领先 + 放量
        breakout   = df["close"] > df["dc_upper"]
        adx_ok     = (df["adx"] > params["adx_min"]) & (df["di_plus"] > df["di_minus"])
        vol_ok     = df["vol_ratio"] > params["vol_ratio_min"]
        df["buy_signal"] = (breakout & adx_ok & vol_ok).fillna(False)
        # 卖出：跌破中线 或 追踪止损触发
        df["sell_signal"] = ((df["close"] < df["dc_mid"]) |
                             (df["close"] < df["trail_stop"])).fillna(False)
        return df


# ─── 27. Williams %R + ADX 趋势过滤 ─────────────────────────────────

class WilliamsRAdx(BaseStrategy):
    name = "claude_williams_r_adx"
    description = "Williams %R + ADX趋势过滤：%R超卖区域配合ADX强趋势买入，%R超买区域卖出，EMA方向确认"
    indicator_formulas = {
        "williams_r": {"name": "Williams %R", "formula": "100 × (HH(N) - Close) / (HH(N) - LL(N))", "params": ["wr_period"]},
        "adx": {"name": "ADX", "formula": "ADX(H,L,C, N)", "params": ["adx_period"]},
        "di_plus": {"name": "DI+", "formula": "+DM/ATR", "params": ["adx_period"]},
        "di_minus": {"name": "DI-", "formula": "-DM/ATR", "params": ["adx_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "Williams_%R < wr_oversold(超卖) 且 ADX > adx_threshold 且 DI+ > DI- 且 Close > EMA趋势线",
        "sell": "Williams_%R > wr_overbought(超买) 或 Close < EMA趋势线 或 DI- > DI+",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "williams_r", "label": "Williams %R", "color": "#2196f3"},
            {"col": "adx", "label": "ADX", "color": "#ff9800"},
            {"col": "di_plus",  "label": "DI+", "color": "#4caf50"},
            {"col": "di_minus", "label": "DI-", "color": "#f44336"},
        ],
    }
    default_params = {
        "wr_period": 14,
        "wr_oversold": -80,
        "wr_overbought": -20,
        "adx_period": 14,
        "adx_threshold": 20,
        "ema_period": 50,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["wr_period"]
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        # Williams %R 范围 [-100, 0]，-80以下为超卖，-20以上为超买
        df["williams_r"] = -100 * (hh - c) / (hh - ll + 1e-10)
        df["adx"]        = ta.trend.adx(h, l, c, window=params["adx_period"])
        df["di_plus"]    = ta.trend.adx_pos(h, l, c, window=params["adx_period"])
        df["di_minus"]   = ta.trend.adx_neg(h, l, c, window=params["adx_period"])
        df["ema_trend"]  = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"]        = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        wr = df["williams_r"]
        oversold  = wr < params["wr_oversold"]          # e.g. < -80
        overbought = wr > params["wr_overbought"]        # e.g. > -20
        adx_ok    = (df["adx"] > params["adx_threshold"]) & (df["di_plus"] > df["di_minus"])
        above_ema = df["close"] > df["ema_trend"]
        # 买入：%R超卖 + ADX强趋势 + 价格在趋势线上方
        df["buy_signal"] = (oversold & adx_ok & above_ema).fillna(False)
        # 卖出：%R超买 或 价格跌破趋势线 或 DI-反超DI+
        df["sell_signal"] = (overbought | (~above_ema) |
                             (df["di_minus"] > df["di_plus"])).fillna(False)
        return df


# ─── 28. Parabolic SAR 趋势跟随 ─────────────────────────────────────

class ParabolicSarTrend(BaseStrategy):
    name = "claude_parabolic_sar"
    description = "Parabolic SAR趋势跟随：SAR翻转信号入场，MACD方向确认，RSI过滤极端区域"
    indicator_formulas = {
        "psar": {"name": "Parabolic SAR", "formula": "SAR = SAR[-1] + AF × (EP - SAR[-1]), AF=[step,max]", "params": ["psar_step", "psar_max"]},
        "psar_up": {"name": "SAR多头", "formula": "价格在SAR上方 = 多头趋势", "params": []},
        "macd_hist": {"name": "MACD Hist", "formula": "MACD(fast,slow) - Signal(signal_w)", "params": ["macd_fast", "macd_slow"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
    }
    signal_rules = {
        "buy": "SAR翻转为多头(price > SAR) 且 MACD_Hist > 0 且 RSI < rsi_overbought",
        "sell": "SAR翻转为空头(price < SAR) 或 RSI > rsi_overbought",
    }
    chart_indicators = {
        "overlay": [
            {"col": "psar", "label": "Parabolic SAR", "color": "#ff5722"},
            {"col": "ema_filter", "label": "EMA Filter", "color": "#9c27b0"},
        ],
        "subchart": [
            {"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"},
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "psar_step": 0.02,
        "psar_max": 0.2,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_overbought": 80,
        "ema_period": 50,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        # Parabolic SAR（ta库实现）
        psar_ind = ta.trend.PSARIndicator(
            h, l, c,
            step=params["psar_step"],
            max_step=params["psar_max"]
        )
        df["psar"]    = psar_ind.psar()
        df["psar_up"] = psar_ind.psar_up()    # 多头SAR（NaN表示当前为空头）
        df["psar_dn"] = psar_ind.psar_down()  # 空头SAR（NaN表示当前为多头）
        # MACD
        macd = ta.trend.MACD(c,
                             window_slow=params["macd_slow"],
                             window_fast=params["macd_fast"],
                             window_sign=params["macd_signal"])
        df["macd_hist"] = macd.macd_diff()
        # RSI
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        # 趋势过滤EMA
        df["ema_filter"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        # ATR
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        # psar_up非NaN表示多头状态
        df["psar_is_bull"] = df["psar_up"].notna().astype(int)
        core_cols = ["psar", "macd_hist", "rsi", "ema_filter", "atr"]
        return df.dropna(subset=core_cols)

    def generate_signals(self, df, params):
        df = df.copy()
        bull_now  = df["psar_is_bull"] == 1
        bull_prev = df["psar_is_bull"].shift(1) == 0
        bear_now  = df["psar_is_bull"] == 0
        bear_prev = df["psar_is_bull"].shift(1) == 1
        # SAR翻转为多头
        sar_flip_bull = bull_now & bull_prev
        # SAR翻转为空头
        sar_flip_bear = bear_now & bear_prev
        macd_pos  = df["macd_hist"] > 0
        rsi_ok    = df["rsi"] < params["rsi_overbought"]
        above_ema = df["close"] > df["ema_filter"]
        # 买入：SAR翻多 + MACD正柱 + RSI未超买 + 价格在EMA上方
        df["buy_signal"]  = (sar_flip_bull & macd_pos & rsi_ok & above_ema).fillna(False)
        # 卖出：SAR翻空 或 RSI超买
        df["sell_signal"] = (sar_flip_bear | (df["rsi"] > params["rsi_overbought"])).fillna(False)
        return df


# ─── 29. Heikin-Ashi 平滑趋势 ────────────────────────────────────────

class HeikinAshiTrend(BaseStrategy):
    name = "claude_heikin_ashi"
    description = "Heikin-Ashi平滑趋势：HA蜡烛过滤噪音，连续同色K线确认趋势，EMA+RSI辅助"
    indicator_formulas = {
        "ha_close": {"name": "HA收盘", "formula": "(Open+High+Low+Close)/4", "params": []},
        "ha_open": {"name": "HA开盘", "formula": "(HA_Open[-1] + HA_Close[-1]) / 2", "params": []},
        "ha_high": {"name": "HA最高", "formula": "Max(High, HA_Open, HA_Close)", "params": []},
        "ha_low": {"name": "HA最低", "formula": "Min(Low, HA_Open, HA_Close)", "params": []},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "连续consec_bull根HA阳线(HA_Close>HA_Open 且 HA_Low无下影) 且 Close>EMA趋势线 且 RSI在适中区间",
        "sell": "连续consec_bear根HA阴线(HA_Close<HA_Open 且 HA_High无上影) 或 RSI>rsi_overbought",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#9c27b0"},
            {"col": "ha_close",  "label": "HA Close",  "color": "#ff9800"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "ha_bull_streak", "label": "Bull Streak", "color": "#4caf50", "type": "histogram"},
        ],
    }
    default_params = {
        "ema_period": 50,
        "rsi_period": 14,
        "rsi_min": 35,
        "rsi_overbought": 80,
        "consec_bull": 2,    # 需要连续N根HA阳线才买入
        "consec_bear": 2,    # 需要连续N根HA阴线才卖出
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        # Heikin-Ashi蜡烛
        ha_close = (o + h + l + c) / 4
        ha_open  = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
        ha_low  = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
        df["ha_open"]  = ha_open
        df["ha_high"]  = ha_high
        df["ha_low"]   = ha_low
        df["ha_close"] = ha_close
        # HA阳线：HA_Close > HA_Open
        df["ha_bullish"] = (ha_close > ha_open).astype(int)
        # 连续HA阳线计数
        streak_bull = pd.Series(0, index=df.index)
        streak_bear = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df["ha_bullish"].iloc[i] == 1:
                streak_bull.iloc[i] = streak_bull.iloc[i-1] + 1
                streak_bear.iloc[i] = 0
            else:
                streak_bear.iloc[i] = streak_bear.iloc[i-1] + 1
                streak_bull.iloc[i] = 0
        df["ha_bull_streak"] = streak_bull
        df["ha_bear_streak"] = streak_bear
        # 辅助指标
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"]       = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)
        core_cols = ["ha_close", "ha_open", "ema_trend", "rsi", "atr"]
        return df.dropna(subset=core_cols)

    def generate_signals(self, df, params):
        df = df.copy()
        # 买入：连续N根HA阳线 + 价格在EMA上方 + RSI在合理区间
        bull_streak = df["ha_bull_streak"] >= params["consec_bull"]
        above_ema   = df["close"] > df["ema_trend"]
        rsi_ok      = (df["rsi"] > params["rsi_min"]) & (df["rsi"] < params["rsi_overbought"])
        df["buy_signal"] = (bull_streak & above_ema & rsi_ok).fillna(False)
        # 卖出：连续N根HA阴线 或 RSI超买
        bear_streak = df["ha_bear_streak"] >= params["consec_bear"]
        rsi_extreme = df["rsi"] > params["rsi_overbought"]
        df["sell_signal"] = (bear_streak | rsi_extreme).fillna(False)
        return df


# ─── 27. Stochastic Momentum Index ─────────────────────────────────────

class StochasticMomentumIndex(BaseStrategy):
    name = "claude_stochastic_momentum"
    description = "随机动量指数SMI：双重平滑的随机振荡器，比KDJ更稳定，EMA趋势确认"
    indicator_formulas = {
        "smi": {"name": "SMI", "formula": "EMA(EMA(Close - (HH+LL)/2, R), S) / EMA(EMA((HH-LL)/2, R), S) × 100", "params": ["smi_period", "smooth_r", "smooth_s"]},
        "smi_signal": {"name": "SMI Signal", "formula": "EMA(SMI, signal_period)", "params": ["signal_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "SMI crosses above Signal AND SMI < oversold AND Close > EMA",
        "sell": "SMI crosses below Signal AND SMI > overbought",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [
            {"col": "smi", "label": "SMI", "color": "#2196f3"},
            {"col": "smi_signal", "label": "SMI Signal", "color": "#ff5722"},
        ],
    }
    default_params = {
        "smi_period": 13, "smooth_r": 5, "smooth_s": 3,
        "signal_period": 9, "ema_period": 50,
        "oversold": -40, "overbought": 40,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["smi_period"]
        r = params["smooth_r"]
        s = params["smooth_s"]
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        diff = c - (hh + ll) / 2
        range_hl = (hh - ll) / 2
        # Double smoothing
        diff_s = diff.ewm(span=r).mean().ewm(span=s).mean()
        range_s = range_hl.ewm(span=r).mean().ewm(span=s).mean()
        df["smi"] = (diff_s / range_s * 100).replace([np.inf, -np.inf], 0)
        df["smi_signal"] = df["smi"].ewm(span=params["signal_period"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        smi, sig = df["smi"], df["smi_signal"]
        prev_smi, prev_sig = smi.shift(1), sig.shift(1)
        cross_up = (prev_smi <= prev_sig) & (smi > sig)
        cross_dn = (prev_smi >= prev_sig) & (smi < sig)
        df["buy_signal"] = (cross_up & (smi < params["oversold"]) & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = (cross_dn & (smi > params["overbought"])).fillna(False)
        return df


# ─── 28. Chaikin Money Flow ────────────────────────────────────────────

class ChaikinMoneyFlow(BaseStrategy):
    name = "claude_chaikin_money_flow"
    description = "蔡金资金流CMF：衡量资金流入流出压力，配合价格突破确认趋势"
    indicator_formulas = {
        "cmf": {"name": "CMF", "formula": "SUM(((C-L)-(H-C))/(H-L)*V, N) / SUM(V, N)", "params": ["cmf_period"]},
        "ema_fast": {"name": "EMA Fast", "formula": "EMA(Close, fast)", "params": ["ema_fast"]},
        "ema_slow": {"name": "EMA Slow", "formula": "EMA(Close, slow)", "params": ["ema_slow"]},
    }
    signal_rules = {
        "buy": "CMF > threshold AND EMA Fast > EMA Slow (资金流入+趋势向上)",
        "sell": "CMF < -threshold OR EMA Fast < EMA Slow",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#4caf50"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#f44336"},
        ],
        "subchart": [{"col": "cmf", "label": "CMF", "color": "#9c27b0"}],
    }
    default_params = {
        "cmf_period": 20, "ema_fast": 20, "ema_slow": 50,
        "cmf_buy_threshold": 0.05, "cmf_sell_threshold": -0.05,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
        mfv = mfm * v
        n = params["cmf_period"]
        df["cmf"] = mfv.rolling(n).sum() / v.rolling(n).sum()
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        trend_up = df["ema_fast"] > df["ema_slow"]
        trend_dn = df["ema_fast"] < df["ema_slow"]
        cmf_strong = df["cmf"] > params["cmf_buy_threshold"]
        cmf_weak = df["cmf"] < params["cmf_sell_threshold"]
        # 买入：资金流入+均线多头排列
        prev_cmf_weak = df["cmf"].shift(1) <= params["cmf_buy_threshold"]
        df["buy_signal"] = (cmf_strong & prev_cmf_weak & trend_up).fillna(False)
        # 卖出：资金流出或均线死叉
        df["sell_signal"] = (cmf_weak | trend_dn).fillna(False)
        return df


# ─── 29. Elder Force Index ─────────────────────────────────────────────

class ElderForceIndex(BaseStrategy):
    name = "claude_elder_force"
    description = "Elder力度指数：价格变化×成交量衡量多空力量，EMA平滑后捕捉趋势反转"
    indicator_formulas = {
        "efi_short": {"name": "EFI Short", "formula": "EMA((Close-Close[-1])*Volume, short_period)", "params": ["short_period"]},
        "efi_long": {"name": "EFI Long", "formula": "EMA((Close-Close[-1])*Volume, long_period)", "params": ["long_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "EFI Short crosses above 0 AND EFI Long > 0 AND Close > EMA (力量转多+大趋势向上)",
        "sell": "EFI Short crosses below 0 AND EFI Long < 0",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [
            {"col": "efi_short", "label": "EFI Short", "color": "#4caf50"},
            {"col": "efi_long", "label": "EFI Long", "color": "#f44336"},
        ],
    }
    default_params = {
        "short_period": 2, "long_period": 13, "ema_period": 50,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        force = (c - c.shift(1)) * v
        df["efi_short"] = force.ewm(span=params["short_period"]).mean()
        df["efi_long"] = force.ewm(span=params["long_period"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        efi_s, efi_l = df["efi_short"], df["efi_long"]
        prev_efi_s = efi_s.shift(1)
        cross_up = (prev_efi_s <= 0) & (efi_s > 0)
        cross_dn = (prev_efi_s >= 0) & (efi_s < 0)
        df["buy_signal"] = (cross_up & (efi_l > 0) & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = (cross_dn & (efi_l < 0)).fillna(False)
        return df


# ─── 30. TRIX Momentum ────────────────────────────────────────────────

class TrixMomentum(BaseStrategy):
    name = "claude_trix_momentum"
    description = "TRIX三重指数平滑动量：三次EMA的变化率，极度平滑的动量指标，过滤噪音捕捉趋势"
    indicator_formulas = {
        "trix": {"name": "TRIX", "formula": "ROC(EMA(EMA(EMA(Close, N), N), N), 1) × 100", "params": ["trix_period"]},
        "trix_signal": {"name": "TRIX Signal", "formula": "SMA(TRIX, signal_period)", "params": ["signal_period"]},
    }
    signal_rules = {
        "buy": "TRIX crosses above Signal (三重平滑动量转多)",
        "sell": "TRIX crosses below Signal",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "trix", "label": "TRIX", "color": "#2196f3"},
            {"col": "trix_signal", "label": "Signal", "color": "#ff5722"},
        ],
    }
    default_params = {
        "trix_period": 15, "signal_period": 9,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["trix_period"]
        ema1 = c.ewm(span=n).mean()
        ema2 = ema1.ewm(span=n).mean()
        ema3 = ema2.ewm(span=n).mean()
        df["trix"] = ema3.pct_change() * 100
        df["trix_signal"] = df["trix"].rolling(params["signal_period"]).mean()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        t, s = df["trix"], df["trix_signal"]
        pt, ps = t.shift(1), s.shift(1)
        df["buy_signal"] = ((pt <= ps) & (t > s)).fillna(False)
        df["sell_signal"] = ((pt >= ps) & (t < s)).fillna(False)
        return df


# ─── 31. Connors RSI ──────────────────────────────────────────────────

class ConnorsRsi(BaseStrategy):
    name = "claude_connors_rsi"
    description = "Connors RSI：融合传统RSI+连涨连跌RSI+百分位排名，短期均值回归利器"
    indicator_formulas = {
        "crsi": {"name": "Connors RSI", "formula": "(RSI(Close,rsi_period) + RSI(Streak,streak_rsi) + PercentRank(ROC,pct_rank_period)) / 3", "params": ["rsi_period", "streak_rsi_period", "pct_rank_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "CRSI < oversold AND Close > EMA (超卖+趋势向上)",
        "sell": "CRSI > overbought",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [{"col": "crsi", "label": "Connors RSI", "color": "#e91e63"}],
    }
    default_params = {
        "rsi_period": 3, "streak_rsi_period": 2, "pct_rank_period": 100,
        "ema_period": 50, "oversold": 15, "overbought": 85,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        # Component 1: Standard RSI
        rsi1 = ta.momentum.rsi(c, window=params["rsi_period"])
        # Component 2: Streak RSI (consecutive up/down days)
        streak = pd.Series(0, index=df.index, dtype=float)
        changes = c.diff()
        for i in range(1, len(df)):
            if changes.iloc[i] > 0:
                streak.iloc[i] = max(streak.iloc[i-1], 0) + 1
            elif changes.iloc[i] < 0:
                streak.iloc[i] = min(streak.iloc[i-1], 0) - 1
            else:
                streak.iloc[i] = 0
        rsi2 = ta.momentum.rsi(streak, window=params["streak_rsi_period"])
        # Component 3: Percent Rank of 1-day ROC
        roc = c.pct_change()
        n = params["pct_rank_period"]
        pct_rank = roc.rolling(n).apply(lambda x: (x[-1:].values[0] > x[:-1]).sum() / (len(x)-1) * 100 if len(x) > 1 else 50, raw=False)
        df["crsi"] = (rsi1 + rsi2 + pct_rank) / 3
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_crsi = df["crsi"].shift(1)
        # 边缘检测：从非超卖进入超卖
        df["buy_signal"] = ((df["crsi"] < params["oversold"]) & (prev_crsi >= params["oversold"]) & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = ((df["crsi"] > params["overbought"]) & (prev_crsi <= params["overbought"])).fillna(False)
        return df


# ─── 32. Coppock Curve ────────────────────────────────────────────────

class CoppockCurve(BaseStrategy):
    name = "claude_coppock_curve"
    description = "Coppock曲线：长期底部识别器，双ROC加权移动平均，专门捕捉大级别底部反转"
    indicator_formulas = {
        "coppock": {"name": "Coppock", "formula": "WMA(ROC(Close,long_roc) + ROC(Close,short_roc), wma_period)", "params": ["long_roc", "short_roc", "wma_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "Coppock crosses above 0 from below (底部反转信号)",
        "sell": "Coppock crosses below 0 OR Close < EMA (趋势翻空)",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [{"col": "coppock", "label": "Coppock", "color": "#00bcd4"}],
    }
    default_params = {
        "long_roc": 14, "short_roc": 11, "wma_period": 10,
        "ema_period": 50, "atr_stop_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        roc_l = c.pct_change(params["long_roc"]) * 100
        roc_s = c.pct_change(params["short_roc"]) * 100
        combined = roc_l + roc_s
        # WMA
        n = params["wma_period"]
        weights = np.arange(1, n + 1, dtype=float)
        df["coppock"] = combined.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        cop = df["coppock"]
        prev = cop.shift(1)
        df["buy_signal"] = ((prev <= 0) & (cop > 0)).fillna(False)
        df["sell_signal"] = ((prev >= 0) & (cop < 0) | (df["close"] < df["ema_trend"])).fillna(False)
        return df


# ─── 33. Mass Index Reversal ──────────────────────────────────────────

class MassIndexReversal(BaseStrategy):
    name = "claude_mass_index"
    description = "Mass Index膨胀反转：通过高低价差EMA比率检测趋势膨胀-收缩，捕捉趋势反转"
    indicator_formulas = {
        "mass_idx": {"name": "Mass Index", "formula": "SUM(EMA(H-L, 9) / EMA(EMA(H-L, 9), 9), 25)", "params": []},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "Mass Index rises above 27 then drops below 26.5 (reversal bulge) AND Close > EMA",
        "sell": "Mass Index reversal bulge AND Close < EMA",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [{"col": "mass_idx", "label": "Mass Index", "color": "#9c27b0"}],
    }
    default_params = {
        "ema_period": 50, "bulge_threshold": 27, "trigger_threshold": 26.5,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        hl_range = h - l
        ema9 = hl_range.ewm(span=9).mean()
        ema_ema9 = ema9.ewm(span=9).mean()
        ratio = ema9 / ema_ema9
        df["mass_idx"] = ratio.rolling(25).sum()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        mi = df["mass_idx"]
        prev_mi = mi.shift(1)
        bulge = params["bulge_threshold"]
        trigger = params["trigger_threshold"]
        # Detect reversal bulge: mass index was above bulge, now drops below trigger
        was_above = mi.rolling(10).max() >= bulge
        drops_below = (prev_mi >= trigger) & (mi < trigger)
        reversal = was_above & drops_below
        df["buy_signal"] = (reversal & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = (reversal & (df["close"] < df["ema_trend"])).fillna(False)
        return df


# ─── 34. Aroon Oscillator ─────────────────────────────────────────────

class AroonOscillator(BaseStrategy):
    name = "claude_aroon_oscillator"
    description = "Aroon振荡器：衡量趋势开始与结束，Aroon Up/Down交叉捕捉新趋势启动"
    indicator_formulas = {
        "aroon_up": {"name": "Aroon Up", "formula": "(N - periods_since_high) / N × 100", "params": ["aroon_period"]},
        "aroon_down": {"name": "Aroon Down", "formula": "(N - periods_since_low) / N × 100", "params": ["aroon_period"]},
        "aroon_osc": {"name": "Aroon Osc", "formula": "Aroon Up - Aroon Down", "params": ["aroon_period"]},
    }
    signal_rules = {
        "buy": "Aroon Osc crosses above +50 (strong uptrend starting)",
        "sell": "Aroon Osc crosses below -50 (strong downtrend)",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "aroon_up", "label": "Aroon Up", "color": "#4caf50"},
            {"col": "aroon_down", "label": "Aroon Down", "color": "#f44336"},
            {"col": "aroon_osc", "label": "Aroon Osc", "color": "#2196f3"},
        ],
    }
    default_params = {
        "aroon_period": 25, "buy_threshold": 50, "sell_threshold": -50,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["aroon_period"]
        df["aroon_up"] = ta.trend.aroon_up(h, l, window=n)
        df["aroon_down"] = ta.trend.aroon_down(h, l, window=n)
        df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        osc = df["aroon_osc"]
        prev = osc.shift(1)
        bt = params["buy_threshold"]
        st = params["sell_threshold"]
        df["buy_signal"] = ((prev <= bt) & (osc > bt)).fillna(False)
        df["sell_signal"] = ((prev >= st) & (osc < st)).fillna(False)
        return df


# ─── 35. Squeeze Momentum ─────────────────────────────────────────────

class SqueezeMomentum(BaseStrategy):
    name = "claude_squeeze_momentum"
    description = "挤压动量：布林带收缩进入Keltner通道时为挤压态，释放时趋势爆发，配合动量方向交易"
    indicator_formulas = {
        "squeeze_on": {"name": "Squeeze", "formula": "BB_Lower > KC_Lower AND BB_Upper < KC_Upper", "params": ["bb_period", "kc_period"]},
        "momentum": {"name": "Momentum", "formula": "LinReg(Close - Mean(Close,N), N)", "params": ["mom_period"]},
    }
    signal_rules = {
        "buy": "Squeeze releases (off) AND Momentum > 0 AND increasing",
        "sell": "Momentum turns negative OR new squeeze starts",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "momentum", "label": "Squeeze Mom", "color": "#00bcd4"},
        ],
    }
    default_params = {
        "bb_period": 20, "bb_mult": 2.0,
        "kc_period": 20, "kc_mult": 1.5,
        "mom_period": 20,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"], window_dev=params["bb_mult"])
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        # Keltner Channels
        kc_mid = c.rolling(params["kc_period"]).mean()
        atr = ta.volatility.average_true_range(h, l, c, window=params["kc_period"])
        kc_upper = kc_mid + params["kc_mult"] * atr
        kc_lower = kc_mid - params["kc_mult"] * atr
        # Squeeze detection
        df["squeeze_on"] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
        # Momentum: linear regression of (close - mean)
        mid = c.rolling(params["mom_period"]).mean()
        delta = c - mid
        # Simple momentum approximation using linear regression slope
        n = params["mom_period"]
        def linreg_val(x):
            if len(x) < n:
                return 0
            y = np.array(x)
            xr = np.arange(len(y))
            slope = (len(y) * np.sum(xr * y) - np.sum(xr) * np.sum(y)) / (len(y) * np.sum(xr**2) - np.sum(xr)**2 + 1e-10)
            intercept = (np.sum(y) - slope * np.sum(xr)) / len(y)
            return slope * (len(y) - 1) + intercept
        df["momentum"] = delta.rolling(n).apply(linreg_val, raw=True)
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        sq = df["squeeze_on"]
        mom = df["momentum"]
        prev_sq = sq.shift(1)
        prev_mom = mom.shift(1)
        # Buy: squeeze releases (was on, now off) AND momentum positive and increasing
        squeeze_off = (prev_sq == 1) & (sq == 0)
        mom_pos = mom > 0
        mom_rising = mom > prev_mom
        df["buy_signal"] = (squeeze_off & mom_pos & mom_rising).fillna(False)
        # Sell: momentum turns negative or new squeeze
        df["sell_signal"] = ((prev_mom >= 0) & (mom < 0) | ((prev_sq == 0) & (sq == 1) & (mom < 0))).fillna(False)
        return df


# ─── 36. Accumulation/Distribution Trend ──────────────────────────────

class AccDistTrend(BaseStrategy):
    name = "claude_acc_dist_trend"
    description = "AD线趋势：累积/派发线与价格趋势的背离和确认，EMA交叉入场"
    indicator_formulas = {
        "ad_line": {"name": "A/D Line", "formula": "CUMSUM(((C-L)-(H-C))/(H-L)*V)", "params": []},
        "ad_ema_fast": {"name": "AD EMA Fast", "formula": "EMA(AD, fast)", "params": ["ad_ema_fast"]},
        "ad_ema_slow": {"name": "AD EMA Slow", "formula": "EMA(AD, slow)", "params": ["ad_ema_slow"]},
    }
    signal_rules = {
        "buy": "AD EMA Fast crosses above AD EMA Slow AND Close > EMA (资金流入加速+价格趋势向上)",
        "sell": "AD EMA Fast crosses below AD EMA Slow",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [
            {"col": "ad_ema_fast", "label": "AD Fast", "color": "#4caf50"},
            {"col": "ad_ema_slow", "label": "AD Slow", "color": "#f44336"},
        ],
    }
    default_params = {
        "ad_ema_fast": 10, "ad_ema_slow": 30, "ema_period": 50,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        clv = ((c - l) - (h - c)) / (h - l + 1e-10)
        df["ad_line"] = (clv * v).cumsum()
        df["ad_ema_fast"] = df["ad_line"].ewm(span=params["ad_ema_fast"]).mean()
        df["ad_ema_slow"] = df["ad_line"].ewm(span=params["ad_ema_slow"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        fast, slow = df["ad_ema_fast"], df["ad_ema_slow"]
        pf, ps = fast.shift(1), slow.shift(1)
        cross_up = (pf <= ps) & (fast > slow)
        cross_dn = (pf >= ps) & (fast < slow)
        df["buy_signal"] = (cross_up & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = cross_dn.fillna(False)
        return df


# ─── 37. Ehlers Fisher Transform ─────────────────────────────────────

class EhlersFisher(BaseStrategy):
    name = "claude_ehlers_fisher"
    description = "Ehlers Fisher变换：将价格转换为近似高斯分布，产生清晰的转折信号"
    indicator_formulas = {
        "fisher": {"name": "Fisher", "formula": "0.5*ln((1+X)/(1-X)), X=normalized price", "params": ["fisher_period"]},
        "fisher_signal": {"name": "Fisher Signal", "formula": "Fisher[-1]", "params": []},
    }
    signal_rules = {
        "buy": "Fisher crosses above Fisher Signal from negative territory",
        "sell": "Fisher crosses below Fisher Signal from positive territory",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [
            {"col": "fisher", "label": "Fisher", "color": "#2196f3"},
            {"col": "fisher_signal", "label": "Signal", "color": "#ff5722"},
        ],
    }
    default_params = {
        "fisher_period": 10, "ema_period": 50,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["fisher_period"]
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        # Normalized value clamped to (-0.999, 0.999)
        raw = 2 * ((c - ll) / (hh - ll + 1e-10) - 0.5)
        val = raw.clip(-0.999, 0.999)
        # Smoothing
        smoothed = val.ewm(span=5).mean().clip(-0.999, 0.999)
        fisher = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            fisher.iloc[i] = 0.5 * np.log((1 + smoothed.iloc[i]) / (1 - smoothed.iloc[i] + 1e-10))
            fisher.iloc[i] = 0.5 * fisher.iloc[i] + 0.5 * fisher.iloc[i-1]
        df["fisher"] = fisher
        df["fisher_signal"] = fisher.shift(1).fillna(0)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        core_cols = ["fisher", "ema_trend", "atr"]
        return df.dropna(subset=core_cols)

    def generate_signals(self, df, params):
        df = df.copy()
        f, s = df["fisher"], df["fisher_signal"]
        cross_up = (f.shift(1) <= s.shift(1)) & (f > s) & (f < 0)
        cross_dn = (f.shift(1) >= s.shift(1)) & (f < s) & (f > 0)
        df["buy_signal"] = (cross_up & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = cross_dn.fillna(False)
        return df


# ─── 38. Volume Weighted Momentum ────────────────────────────────────

class VolumeWeightedMomentum(BaseStrategy):
    name = "claude_vol_weighted_mom"
    description = "成交量加权动量：用成交量加权价格变动率，量价齐升时信号更强"
    indicator_formulas = {
        "vwm": {"name": "VWM", "formula": "SUM(ROC*V, N) / SUM(V, N)", "params": ["vwm_period"]},
        "vwm_signal": {"name": "VWM Signal", "formula": "EMA(VWM, signal_period)", "params": ["signal_period"]},
    }
    signal_rules = {
        "buy": "VWM crosses above Signal AND VWM > 0 (量价齐升)",
        "sell": "VWM crosses below Signal AND VWM < 0",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [{"col": "vwm", "label": "VWM", "color": "#00bcd4"}, {"col": "vwm_signal", "label": "Signal", "color": "#ff5722"}],
    }
    default_params = {
        "vwm_period": 14, "signal_period": 9, "ema_period": 50,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        roc = c.pct_change()
        n = params["vwm_period"]
        df["vwm"] = (roc * v).rolling(n).sum() / (v.rolling(n).sum() + 1e-10) * 100
        df["vwm_signal"] = df["vwm"].ewm(span=params["signal_period"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        vwm, sig = df["vwm"], df["vwm_signal"]
        pv, ps = vwm.shift(1), sig.shift(1)
        cross_up = (pv <= ps) & (vwm > sig)
        cross_dn = (pv >= ps) & (vwm < sig)
        df["buy_signal"] = (cross_up & (vwm > 0) & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = (cross_dn & (vwm < 0)).fillna(False)
        return df


# ─── 39. Price Channel Breakout with Volume ──────────────────────────

class PriceChannelVolume(BaseStrategy):
    name = "claude_price_channel_vol"
    description = "价格通道+量能突破：N日高低通道突破配合成交量放大确认，减少假突破"
    indicator_formulas = {
        "channel_high": {"name": "Channel High", "formula": "MAX(High, N)", "params": ["channel_period"]},
        "channel_low": {"name": "Channel Low", "formula": "MIN(Low, N)", "params": ["channel_period"]},
        "vol_ratio": {"name": "Vol Ratio", "formula": "V / SMA(V, vol_ma)", "params": ["vol_ma_period"]},
    }
    signal_rules = {
        "buy": "Close breaks above Channel High AND Volume > vol_threshold × avg_volume",
        "sell": "Close breaks below Channel Low OR trailing stop hit",
    }
    chart_indicators = {
        "overlay": [
            {"col": "channel_high", "label": "Channel High", "color": "#4caf50"},
            {"col": "channel_low", "label": "Channel Low", "color": "#f44336"},
        ],
        "subchart": [{"col": "vol_ratio", "label": "Vol Ratio", "color": "#9c27b0"}],
    }
    default_params = {
        "channel_period": 20, "exit_period": 10,
        "vol_ma_period": 20, "vol_threshold": 1.5,
        "atr_stop_mul": 3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        n = params["channel_period"]
        # Use shifted values to avoid look-ahead
        df["channel_high"] = h.shift(1).rolling(n).max()
        df["channel_low"] = l.shift(1).rolling(params["exit_period"]).min()
        vol_ma = v.rolling(params["vol_ma_period"]).mean()
        df["vol_ratio"] = v / (vol_ma + 1e-10)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        breakout = df["close"] > df["channel_high"]
        prev_no_breakout = df["close"].shift(1) <= df["channel_high"].shift(1)
        vol_confirm = df["vol_ratio"] > params["vol_threshold"]
        df["buy_signal"] = (breakout & prev_no_breakout & vol_confirm).fillna(False)
        breakdown = df["close"] < df["channel_low"]
        df["sell_signal"] = breakdown.fillna(False)
        return df


# ─── 40. Mean Reversion Bollinger RSI ────────────────────────────────

class MeanRevBollingerRsi(BaseStrategy):
    name = "claude_meanrev_bb_rsi"
    description = "布林带RSI均值回归增强版：BB%B+RSI双重超卖确认，ADX低值过滤趋势市"
    indicator_formulas = {
        "pct_b": {"name": "%B", "formula": "(Close - BB_Lower) / (BB_Upper - BB_Lower)", "params": ["bb_period"]},
        "rsi": {"name": "RSI", "formula": "RSI(Close, rsi_period)", "params": ["rsi_period"]},
        "adx": {"name": "ADX", "formula": "ADX(H,L,C, adx_period)", "params": ["adx_period"]},
    }
    signal_rules = {
        "buy": "%B < 0.05 AND RSI < rsi_oversold AND ADX < adx_max (震荡市超卖)",
        "sell": "%B > 0.95 OR RSI > rsi_overbought",
    }
    chart_indicators = {
        "overlay": [
            {"col": "bb_upper", "label": "BB Upper", "color": "#f44336"},
            {"col": "bb_lower", "label": "BB Lower", "color": "#4caf50"},
            {"col": "bb_mid", "label": "BB Mid", "color": "#9e9e9e"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#2196f3"},
            {"col": "adx", "label": "ADX", "color": "#ff9800"},
        ],
    }
    default_params = {
        "bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
        "rsi_oversold": 30, "rsi_overbought": 70,
        "adx_period": 14, "adx_max": 25,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"], window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["pct_b"] = bb.bollinger_pband()
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["adx"] = ta.trend.adx(h, l, c, window=params["adx_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        pct_b = df["pct_b"]
        rsi = df["rsi"]
        adx = df["adx"]
        prev_pct_b = pct_b.shift(1)
        # Buy: deep oversold + ranging market
        oversold = (pct_b < 0.05) & (prev_pct_b >= 0.05)
        rsi_low = rsi < params["rsi_oversold"]
        ranging = adx < params["adx_max"]
        df["buy_signal"] = (oversold & rsi_low & ranging).fillna(False)
        # Sell: overbought
        df["sell_signal"] = ((pct_b > 0.95) | (rsi > params["rsi_overbought"])).fillna(False)
        return df


# ─── 41. Dual Thrust ─────────────────────────────────────────────────

class DualThrust(BaseStrategy):
    name = "claude_dual_thrust"
    description = "Dual Thrust日内突破：基于前N日振幅构建动态Range，开盘价±K×Range确定突破阈值"
    indicator_formulas = {
        "upper_band": {"name": "Upper", "formula": "Open + K1 × max(HH-LC, HC-LL)", "params": ["lookback", "k1"]},
        "lower_band": {"name": "Lower", "formula": "Open - K2 × max(HH-LC, HC-LL)", "params": ["lookback", "k2"]},
    }
    signal_rules = {
        "buy": "Close breaks above Upper Band",
        "sell": "Close breaks below Lower Band",
    }
    chart_indicators = {
        "overlay": [
            {"col": "upper_band", "label": "DT Upper", "color": "#4caf50"},
            {"col": "lower_band", "label": "DT Lower", "color": "#f44336"},
        ],
        "subchart": [],
    }
    default_params = {
        "lookback": 4, "k1": 0.5, "k2": 0.5,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["lookback"]
        hh = h.shift(1).rolling(n).max()
        ll = l.shift(1).rolling(n).min()
        hc = c.shift(1).rolling(n).max()
        lc = c.shift(1).rolling(n).min()
        range_val = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
        df["upper_band"] = df["open"] + params["k1"] * range_val
        df["lower_band"] = df["open"] - params["k2"] * range_val
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        prev_c = c.shift(1)
        df["buy_signal"] = ((c > df["upper_band"]) & (prev_c <= df["upper_band"].shift(1))).fillna(False)
        df["sell_signal"] = ((c < df["lower_band"]) & (prev_c >= df["lower_band"].shift(1))).fillna(False)
        return df


# ─── 42. RSI Divergence ──────────────────────────────────────────────

class RsiDivergence(BaseStrategy):
    name = "claude_rsi_divergence"
    description = "RSI背离策略：价格创新低但RSI未创新低(看涨背离)，精确底部反转捕捉"
    indicator_formulas = {
        "rsi": {"name": "RSI", "formula": "RSI(Close, rsi_period)", "params": ["rsi_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "Price makes lower low BUT RSI makes higher low (bullish divergence) AND RSI < 40",
        "sell": "Price makes higher high BUT RSI makes lower high (bearish divergence) AND RSI > 60",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "rsi_period": 14, "lookback": 14, "ema_period": 50,
        "rsi_buy_zone": 40, "rsi_sell_zone": 60,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        rsi = df["rsi"]
        n = params["lookback"]
        # Price lower low but RSI higher low (bullish divergence)
        price_ll = c < c.rolling(n).min().shift(1)
        rsi_hl = rsi > rsi.rolling(n).min().shift(1)
        bull_div = price_ll & rsi_hl & (rsi < params["rsi_buy_zone"])
        # Price higher high but RSI lower high (bearish divergence)
        price_hh = c > c.rolling(n).max().shift(1)
        rsi_lh = rsi < rsi.rolling(n).max().shift(1)
        bear_div = price_hh & rsi_lh & (rsi > params["rsi_sell_zone"])
        # Edge detection
        prev_bull = (c.shift(1) < c.shift(1).rolling(n).min().shift(1)) & (rsi.shift(1) > rsi.shift(1).rolling(n).min().shift(1))
        df["buy_signal"] = (bull_div & ~prev_bull.fillna(False)).fillna(False)
        prev_bear = (c.shift(1) > c.shift(1).rolling(n).max().shift(1)) & (rsi.shift(1) < rsi.shift(1).rolling(n).max().shift(1))
        df["sell_signal"] = (bear_div & ~prev_bear.fillna(False)).fillna(False)
        return df


# ─── 43. Moving Average Ribbon ───────────────────────────────────────

class MaRibbon(BaseStrategy):
    name = "claude_ma_ribbon"
    description = "均线带状策略：多条EMA形成带状排列，全部多头排列时趋势强烈，扭转时反转"
    indicator_formulas = {
        "ema_8": {"name": "EMA 8", "formula": "EMA(C, 8)", "params": []},
        "ema_13": {"name": "EMA 13", "formula": "EMA(C, 13)", "params": []},
        "ema_21": {"name": "EMA 21", "formula": "EMA(C, 21)", "params": []},
        "ema_34": {"name": "EMA 34", "formula": "EMA(C, 34)", "params": []},
        "ema_55": {"name": "EMA 55", "formula": "EMA(C, 55)", "params": []},
    }
    signal_rules = {
        "buy": "All EMAs in bullish order (8>13>21>34>55) AND price above all",
        "sell": "EMA 8 crosses below EMA 21 (ribbon twist)",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_8", "label": "EMA 8", "color": "#4caf50"},
            {"col": "ema_13", "label": "EMA 13", "color": "#8bc34a"},
            {"col": "ema_21", "label": "EMA 21", "color": "#ff9800"},
            {"col": "ema_34", "label": "EMA 34", "color": "#ff5722"},
            {"col": "ema_55", "label": "EMA 55", "color": "#f44336"},
        ],
        "subchart": [],
    }
    default_params = {
        "ema_1": 8, "ema_2": 13, "ema_3": 21, "ema_4": 34, "ema_5": 55,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_8"] = ta.trend.ema_indicator(c, window=params["ema_1"])
        df["ema_13"] = ta.trend.ema_indicator(c, window=params["ema_2"])
        df["ema_21"] = ta.trend.ema_indicator(c, window=params["ema_3"])
        df["ema_34"] = ta.trend.ema_indicator(c, window=params["ema_4"])
        df["ema_55"] = ta.trend.ema_indicator(c, window=params["ema_5"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        e8, e13, e21, e34, e55 = df["ema_8"], df["ema_13"], df["ema_21"], df["ema_34"], df["ema_55"]
        # Bullish ribbon: all EMAs in order
        bullish = (e8 > e13) & (e13 > e21) & (e21 > e34) & (e34 > e55)
        prev_not_bullish = ~((e8.shift(1) > e13.shift(1)) & (e13.shift(1) > e21.shift(1)) & (e21.shift(1) > e34.shift(1)) & (e34.shift(1) > e55.shift(1)))
        df["buy_signal"] = (bullish & prev_not_bullish & (df["close"] > e8)).fillna(False)
        # Sell: ribbon twist - fast EMA crosses below medium
        twist = (e8 < e21) & (e8.shift(1) >= e21.shift(1))
        df["sell_signal"] = twist.fillna(False)
        return df


# ─── 44. VWAP Mean Reversion ─────────────────────────────────────────

class VwapMeanReversion(BaseStrategy):
    name = "claude_vwap_mean_reversion"
    description = "VWAP均值回归：价格偏离VWAP超过N个标准差后回归，配合RSI确认超卖/超买"
    chart_indicators = {
        "overlay": [{"col": "vwap", "label": "VWAP", "color": "#ff9800"}],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "std_period": 20, "std_mult": 2.0, "rsi_period": 14,
        "rsi_oversold": 35, "rsi_overbought": 70, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        tp = (h + l + c) / 3
        df["vwap"] = (tp * v).cumsum() / (v.cumsum() + 1e-10)
        df["vwap_std"] = (c - df["vwap"]).rolling(params["std_period"]).std()
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c, vwap, std, rsi = df["close"], df["vwap"], df["vwap_std"], df["rsi"]
        k = params["std_mult"]
        below = c < (vwap - k * std)
        prev_above = c.shift(1) >= (vwap.shift(1) - k * std.shift(1))
        df["buy_signal"] = (below & prev_above & (rsi < params["rsi_oversold"])).fillna(False)
        df["sell_signal"] = ((c > (vwap + k * std)) | (rsi > params["rsi_overbought"])).fillna(False)
        return df


# ─── 45. Trend Strength ADX-DI ───────────────────────────────────────

class TrendStrengthAdxDi(BaseStrategy):
    name = "claude_trend_strength"
    description = "趋势强度ADX+DI：ADX上升确认趋势强化，+DI/-DI交叉定方向，只在强趋势中交易"
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "adx", "label": "ADX", "color": "#9c27b0"},
            {"col": "di_plus", "label": "+DI", "color": "#4caf50"},
            {"col": "di_minus", "label": "-DI", "color": "#f44336"},
        ],
    }
    default_params = {"adx_period": 14, "adx_threshold": 20, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["adx_period"]
        df["adx"] = ta.trend.adx(h, l, c, window=n)
        df["di_plus"] = ta.trend.adx_pos(h, l, c, window=n)
        df["di_minus"] = ta.trend.adx_neg(h, l, c, window=n)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        adx, dip, dim = df["adx"], df["di_plus"], df["di_minus"]
        pdip, pdim = dip.shift(1), dim.shift(1)
        strong = adx > params["adx_threshold"]
        adx_rising = adx > adx.shift(1)
        df["buy_signal"] = ((pdip <= pdim) & (dip > dim) & strong & adx_rising).fillna(False)
        df["sell_signal"] = ((pdip >= pdim) & (dip < dim) & strong).fillna(False)
        return df


# ─── 46. Gap and Go ──────────────────────────────────────────────────

class GapAndGo(BaseStrategy):
    name = "claude_gap_and_go"
    description = "跳空追涨：开盘跳空高开且成交量放大，顺势追涨，ATR追踪止损"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [{"col": "gap_pct", "label": "Gap %", "color": "#4caf50"}, {"col": "vol_ratio", "label": "Vol Ratio", "color": "#9c27b0"}],
    }
    default_params = {"gap_threshold": 2.0, "vol_ma_period": 20, "vol_threshold": 1.5, "ema_period": 50, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v, o = df["high"], df["low"], df["close"], df["volume"], df["open"]
        df["gap_pct"] = (o - c.shift(1)) / (c.shift(1) + 1e-10) * 100
        df["vol_ratio"] = v / (v.rolling(params["vol_ma_period"]).mean() + 1e-10)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        gap_up = df["gap_pct"] > params["gap_threshold"]
        vol_ok = df["vol_ratio"] > params["vol_threshold"]
        close_gt_open = df["close"] > df["open"]
        above_ema = df["close"] > df["ema_trend"]
        prev_no = df["gap_pct"].shift(1) <= params["gap_threshold"]
        df["buy_signal"] = (gap_up & vol_ok & close_gt_open & above_ema & prev_no).fillna(False)
        df["sell_signal"] = ((df["gap_pct"] < -params["gap_threshold"]) | (df["close"] < df["ema_trend"])).fillna(False)
        return df


# ─── 47. Triple EMA Crossover ────────────────────────────────────────

class TripleEmaCross(BaseStrategy):
    name = "claude_triple_ema_cross"
    description = "三重EMA交叉：快/中/慢三条EMA形成多头排列确认强趋势，比双均线更可靠"
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#4caf50"},
            {"col": "ema_mid", "label": "EMA Mid", "color": "#ff9800"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#f44336"},
        ],
        "subchart": [],
    }
    default_params = {"fast_period": 8, "mid_period": 21, "slow_period": 55, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["fast_period"])
        df["ema_mid"] = ta.trend.ema_indicator(c, window=params["mid_period"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["slow_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        ef, em, es = df["ema_fast"], df["ema_mid"], df["ema_slow"]
        pef, pem = ef.shift(1), em.shift(1)
        df["buy_signal"] = ((pef <= pem) & (ef > em) & (ef > es) & (em > es)).fillna(False)
        df["sell_signal"] = ((pef >= pem) & (ef < em)).fillna(False)
        return df


# ─── 48. Momentum Acceleration ──────────────────────────────────────

class MomentumAcceleration(BaseStrategy):
    name = "claude_momentum_accel"
    description = "动量加速度：ROC的ROC，检测动量加速拐点，在动量开始加速时入场"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [{"col": "roc1", "label": "ROC", "color": "#2196f3"}, {"col": "roc2", "label": "Accel", "color": "#e91e63"}],
    }
    default_params = {"roc_period": 12, "accel_period": 6, "ema_period": 50, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["roc1"] = c.pct_change(params["roc_period"]) * 100
        df["roc2"] = df["roc1"].diff(params["accel_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        roc, accel = df["roc1"], df["roc2"]
        prev_accel = accel.shift(1)
        df["buy_signal"] = ((roc > 0) & (prev_accel <= 0) & (accel > 0) & (df["close"] > df["ema_trend"])).fillna(False)
        df["sell_signal"] = ((roc < 0) & (roc.shift(1) >= 0)).fillna(False)
        return df


# ─── 49. Pivot Point Bounce ─────────────────────────────────────────

class PivotPointBounce(BaseStrategy):
    name = "claude_pivot_bounce"
    description = "枢轴点反弹：经典Pivot点+支撑阻力位，价格触及支撑位反弹买入"
    chart_indicators = {
        "overlay": [
            {"col": "pivot", "label": "Pivot", "color": "#9e9e9e"},
            {"col": "support1", "label": "S1", "color": "#4caf50"},
            {"col": "resist1", "label": "R1", "color": "#f44336"},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {"rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 70, "atr_stop_mul": 2.0}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)
        df["pivot"] = (ph + pl + pc) / 3
        df["support1"] = 2 * df["pivot"] - ph
        df["resist1"] = 2 * df["pivot"] - pl
        df["support2"] = df["pivot"] - (ph - pl)
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        touch_s1 = (df["low"] <= df["support1"]) & (c > df["support1"])
        prev_above = df["low"].shift(1) > df["support1"].shift(1)
        rsi_ok = df["rsi"] < params["rsi_oversold"] + 10
        df["buy_signal"] = (touch_s1 & rsi_ok & prev_above).fillna(False)
        df["sell_signal"] = ((c >= df["resist1"]) | (c < df["support2"]) | (df["rsi"] > params["rsi_overbought"])).fillna(False)
        return df


# ─── 50. Chandelier Exit ────────────────────────────────────────────

class ChandelierExit(BaseStrategy):
    name = "claude_chandelier_exit"
    description = "吊灯止损趋势跟随：基于ATR的动态止损线，紧密跟随最高价，趋势反转时精准离场"
    chart_indicators = {
        "overlay": [
            {"col": "chand_long", "label": "Chand Long", "color": "#4caf50"},
            {"col": "chand_short", "label": "Chand Short", "color": "#f44336"},
        ],
        "subchart": [],
    }
    default_params = {"lookback": 22, "atr_mult": 3.0, "atr_period": 22, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["lookback"]
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["chand_long"] = h.rolling(n).max() - params["atr_mult"] * atr
        df["chand_short"] = l.rolling(n).min() + params["atr_mult"] * atr
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c, pc = df["close"], df["close"].shift(1)
        cl, cs = df["chand_long"], df["chand_short"]
        pcl, pcs = cl.shift(1), cs.shift(1)
        df["buy_signal"] = ((pc <= pcs) & (c > cs)).fillna(False)
        df["sell_signal"] = ((pc >= pcl) & (c < cl)).fillna(False)
        return df


# ─── 51. BB Squeeze Breakout ─────────────────────────────────────────

class BbSqueezeBreakout(BaseStrategy):
    name = "claude_bb_squeeze_breakout"
    description = "布林带收缩突破：波动率极低时(BB带宽最窄)蓄力待发，突破方向入场，趋势过滤"
    chart_indicators = {
        "overlay": [
            {"col": "bb_upper", "label": "BB Upper", "color": "#f44336"},
            {"col": "bb_lower", "label": "BB Lower", "color": "#4caf50"},
            {"col": "ema_trend", "label": "EMA", "color": "#ff9800"},
        ],
        "subchart": [{"col": "bb_width", "label": "BB Width", "color": "#9c27b0"}],
    }
    default_params = {
        "bb_period": 20, "bb_std": 2.0, "width_lookback": 120,
        "width_percentile": 20, "ema_period": 50, "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"], window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        n = params["width_lookback"]
        df["width_pct"] = df["bb_width"].rolling(n).apply(
            lambda x: (x[-1:].values[0] <= np.percentile(x, params["width_percentile"])), raw=False)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        squeeze = df["width_pct"] == 1
        breakout_up = c > df["bb_upper"]
        above_ema = c > df["ema_trend"]
        prev_no_breakout = c.shift(1) <= df["bb_upper"].shift(1)
        df["buy_signal"] = (squeeze & breakout_up & above_ema & prev_no_breakout).fillna(False)
        breakout_dn = c < df["bb_lower"]
        df["sell_signal"] = (breakout_dn | (c < df["ema_trend"])).fillna(False)
        return df


# ─── 52. Volume Climax Reversal ──────────────────────────────────────

class VolumeClimaxReversal(BaseStrategy):
    name = "claude_volume_climax"
    description = "量能高潮反转：极端成交量+大阴线后的反转买入，恐慌抛售后的底部捕捉"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [{"col": "vol_ratio", "label": "Vol Ratio", "color": "#9c27b0"}, {"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "vol_ma_period": 20, "vol_spike": 2.5, "body_pct": 0.02,
        "rsi_period": 14, "rsi_oversold": 35, "ema_period": 50, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v, o = df["high"], df["low"], df["close"], df["volume"], df["open"]
        vol_ma = v.rolling(params["vol_ma_period"]).mean()
        df["vol_ratio"] = v / (vol_ma + 1e-10)
        df["body_change"] = (c - o) / (o + 1e-10)
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # Climax selling: huge volume + big red candle yesterday, then reversal today
        prev_vol_spike = df["vol_ratio"].shift(1) > params["vol_spike"]
        prev_big_red = df["body_change"].shift(1) < -params["body_pct"]
        today_green = df["close"] > df["open"]
        rsi_low = df["rsi"] < params["rsi_oversold"] + 10
        df["buy_signal"] = (prev_vol_spike & prev_big_red & today_green & rsi_low).fillna(False)
        # Sell: RSI overbought or below EMA
        df["sell_signal"] = ((df["rsi"] > 75) | (df["close"] < df["ema_trend"])).fillna(False)
        return df


# ─── 53. Adaptive Bollinger ─────────────────────────────────────────

class AdaptiveBollinger(BaseStrategy):
    name = "claude_adaptive_bollinger"
    description = "自适应布林带：根据ATR动态调整布林带倍数，高波动时放宽通道，低波动时收紧"
    chart_indicators = {
        "overlay": [
            {"col": "abb_upper", "label": "AdaBB Upper", "color": "#f44336"},
            {"col": "abb_lower", "label": "AdaBB Lower", "color": "#4caf50"},
            {"col": "abb_mid", "label": "AdaBB Mid", "color": "#9e9e9e"},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "bb_period": 20, "base_mult": 2.0, "atr_period": 14,
        "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["bb_period"]
        sma = c.rolling(n).mean()
        std = c.rolling(n).std()
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        atr_ma = atr.rolling(n).mean()
        # Adaptive multiplier: higher when ATR is above its average
        atr_ratio = atr / (atr_ma + 1e-10)
        mult = params["base_mult"] * atr_ratio.clip(0.5, 2.0)
        df["abb_upper"] = sma + mult * std
        df["abb_lower"] = sma - mult * std
        df["abb_mid"] = sma
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        pc = c.shift(1)
        rsi = df["rsi"]
        # Buy: price crosses below adaptive lower band + RSI oversold
        touch_lower = (c <= df["abb_lower"]) & (pc > df["abb_lower"].shift(1))
        df["buy_signal"] = (touch_lower & (rsi < params["rsi_oversold"])).fillna(False)
        # Sell: price hits upper band or RSI overbought
        touch_upper = c >= df["abb_upper"]
        df["sell_signal"] = (touch_upper | (rsi > params["rsi_overbought"])).fillna(False)
        return df


# ─── 54. Pullback Buy ───────────────────────────────────────────────

class PullbackBuy(BaseStrategy):
    name = "claude_pullback_buy"
    description = "趋势回调买入：在强上升趋势中等待回调至支撑位(EMA/Fib)，低点企稳后买入"
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#4caf50"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#f44336"},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "ema_fast": 20, "ema_slow": 50, "rsi_period": 14,
        "rsi_pullback_low": 40, "rsi_pullback_high": 55,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        ef, es = df["ema_fast"], df["ema_slow"]
        rsi = df["rsi"]
        # Uptrend: EMA fast > EMA slow
        uptrend = ef > es
        # Pullback: price touches EMA fast from above, RSI in pullback zone
        touch_ema = (c <= ef * 1.01) & (c > es)
        rsi_pb = (rsi > params["rsi_pullback_low"]) & (rsi < params["rsi_pullback_high"])
        # Bounce: today close > open (green candle after pullback)
        bounce = c > df["open"]
        prev_not_touch = c.shift(1) > ef.shift(1) * 1.01
        df["buy_signal"] = (uptrend & touch_ema & rsi_pb & bounce & prev_not_touch).fillna(False)
        # Sell: trend reversal
        df["sell_signal"] = ((ef < es) & (ef.shift(1) >= es.shift(1))).fillna(False)
        return df


# ─── 55. Double Bottom RSI ──────────────────────────────────────────

class DoubleBottomRsi(BaseStrategy):
    name = "claude_double_bottom_rsi"
    description = "双底RSI：RSI在超卖区形成双底(W形)后反弹买入，价格确认的经典底部形态"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "rsi_period": 14, "rsi_low": 35, "lookback": 20,
        "ema_period": 50, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        rsi = df["rsi"]
        n = params["lookback"]
        rsi_low = params["rsi_low"]
        # Detect RSI double bottom: current RSI was recently low, bounced, dipped again, now rising
        rsi_min_recent = rsi.rolling(n).min()
        rsi_was_low = rsi_min_recent < rsi_low
        rsi_bounced = rsi > rsi_low  # currently above threshold
        rsi_rising = rsi > rsi.shift(1)
        # Second dip near first: current low within 5 points of recent minimum
        near_first_bottom = (rsi - rsi_min_recent).abs() < 8
        prev_not = ~((rsi.shift(1) > rsi.shift(2)) & (rsi.shift(1) > rsi_low))
        df["buy_signal"] = (rsi_was_low & rsi_bounced & rsi_rising & near_first_bottom & prev_not).fillna(False)
        df["sell_signal"] = ((rsi > 70) | (df["close"] < df["ema_trend"])).fillna(False)
        return df


# ─── 56. ATR Channel Trend ──────────────────────────────────────────

class AtrChannelTrend(BaseStrategy):
    name = "claude_atr_channel_trend"
    description = "ATR通道趋势：以EMA为中轴，ATR倍数为通道宽度，突破通道追踪趋势"
    chart_indicators = {
        "overlay": [
            {"col": "atr_upper", "label": "ATR Upper", "color": "#f44336"},
            {"col": "atr_lower", "label": "ATR Lower", "color": "#4caf50"},
            {"col": "ema_mid", "label": "EMA Mid", "color": "#9e9e9e"},
        ],
        "subchart": [],
    }
    default_params = {
        "ema_period": 20, "atr_period": 14, "atr_mult": 2.0,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_mid"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["atr_upper"] = df["ema_mid"] + params["atr_mult"] * atr
        df["atr_lower"] = df["ema_mid"] - params["atr_mult"] * atr
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c, pc = df["close"], df["close"].shift(1)
        # Buy: price breaks above upper channel
        df["buy_signal"] = ((c > df["atr_upper"]) & (pc <= df["atr_upper"].shift(1))).fillna(False)
        # Sell: price breaks below lower channel
        df["sell_signal"] = ((c < df["atr_lower"]) & (pc >= df["atr_lower"].shift(1))).fillna(False)
        return df


# ─── 57. Stochastic RSI ─────────────────────────────────────────────

class StochasticRsi(BaseStrategy):
    name = "claude_stoch_rsi"
    description = "StochRSI：对RSI再做随机指标运算，灵敏度更高的超买超卖信号，EMA趋势确认"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [
            {"col": "stoch_rsi_k", "label": "StochRSI K", "color": "#2196f3"},
            {"col": "stoch_rsi_d", "label": "StochRSI D", "color": "#ff5722"},
        ],
    }
    default_params = {
        "rsi_period": 14, "stoch_period": 14, "k_smooth": 3, "d_smooth": 3,
        "oversold": 20, "overbought": 80, "ema_period": 50,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        rsi = ta.momentum.rsi(c, window=params["rsi_period"])
        n = params["stoch_period"]
        rsi_min = rsi.rolling(n).min()
        rsi_max = rsi.rolling(n).max()
        stoch_rsi_raw = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
        df["stoch_rsi_k"] = stoch_rsi_raw.rolling(params["k_smooth"]).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(params["d_smooth"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"] = rsi
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        k, d = df["stoch_rsi_k"], df["stoch_rsi_d"]
        pk, pd_ = k.shift(1), d.shift(1)
        os_lvl = params["oversold"]
        ob_lvl = params["overbought"]
        # Buy: K crosses above D in oversold zone, price above EMA
        cross_up = (pk <= pd_) & (k > d) & (k < os_lvl + 10)
        above_ema = df["close"] > df["ema_trend"]
        df["buy_signal"] = (cross_up & above_ema).fillna(False)
        # Sell: K crosses below D in overbought zone
        cross_dn = (pk >= pd_) & (k < d) & (k > ob_lvl - 10)
        df["sell_signal"] = cross_dn.fillna(False)
        return df


# ─── 58. Combo Mean Reversion Plus ──────────────────────────────────

class ComboMeanRevPlus(BaseStrategy):
    name = "claude_combo_meanrev_plus"
    description = "组合均值回归增强：BB%B+RSI+StochRSI三重超卖确认，ADX+ATR趋势过滤，高胜率均值回归"
    chart_indicators = {
        "overlay": [
            {"col": "bb_upper", "label": "BB Upper", "color": "#f44336"},
            {"col": "bb_lower", "label": "BB Lower", "color": "#4caf50"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#2196f3"},
            {"col": "stoch_rsi_k", "label": "StochRSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
        "stoch_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
        "stoch_oversold": 20, "adx_max": 30,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"], window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["pct_b"] = bb.bollinger_pband()
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        # StochRSI
        rsi = df["rsi"]
        n = params["stoch_period"]
        rsi_min = rsi.rolling(n).min()
        rsi_max = rsi.rolling(n).max()
        df["stoch_rsi_k"] = ((rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100).rolling(3).mean()
        df["adx"] = ta.trend.adx(h, l, c, window=14)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # Triple oversold confirmation
        bb_low = df["pct_b"] < 0.1
        rsi_low = df["rsi"] < params["rsi_oversold"]
        stoch_low = df["stoch_rsi_k"] < params["stoch_oversold"]
        ranging = df["adx"] < params["adx_max"]
        # At least 2 of 3 oversold signals + ranging market
        oversold_count = bb_low.astype(int) + rsi_low.astype(int) + stoch_low.astype(int)
        buy_cond = (oversold_count >= 2) & ranging
        prev_not = (df["pct_b"].shift(1) >= 0.1) | (df["rsi"].shift(1) >= params["rsi_oversold"])
        df["buy_signal"] = (buy_cond & prev_not).fillna(False)
        # Sell: overbought
        df["sell_signal"] = ((df["pct_b"] > 0.95) | (df["rsi"] > params["rsi_overbought"])).fillna(False)
        return df


# ─── 59. Range Contraction Expansion ────────────────────────────────

class RangeContractionExpansion(BaseStrategy):
    name = "claude_range_expansion"
    description = "波幅收缩扩张：ATR收缩至极低值后突然扩张，配合趋势方向捕捉大行情起点"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [{"col": "atr_ratio", "label": "ATR Ratio", "color": "#9c27b0"}],
    }
    default_params = {
        "atr_period": 14, "atr_ma_period": 50, "contraction": 0.7,
        "expansion": 1.3, "ema_period": 50, "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        atr_ma = atr.rolling(params["atr_ma_period"]).mean()
        df["atr_ratio"] = atr / (atr_ma + 1e-10)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        ratio = df["atr_ratio"]
        prev_ratio = ratio.shift(1)
        c = df["close"]
        # Was contracted, now expanding + price above EMA
        was_contracted = prev_ratio < params["contraction"]
        now_expanding = ratio > params["expansion"]
        above_ema = c > df["ema_trend"]
        df["buy_signal"] = (was_contracted & now_expanding & above_ema).fillna(False)
        # Sell: expansion in bearish direction
        below_ema = c < df["ema_trend"]
        df["sell_signal"] = ((was_contracted & now_expanding & below_ema) | (c < df["ema_trend"])).fillna(False)
        return df


# ─── 60. Multi Oscillator Consensus ─────────────────────────────────

class MultiOscConsensus(BaseStrategy):
    name = "claude_multi_osc_consensus"
    description = "多振荡器共识：RSI+StochRSI+CCI三重振荡器投票，至少2个超卖/超买才触发信号"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#2196f3"},
            {"col": "cci", "label": "CCI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "rsi_period": 14, "cci_period": 20, "stoch_period": 14,
        "rsi_oversold": 30, "rsi_overbought": 70,
        "cci_oversold": -100, "cci_overbought": 100,
        "ema_period": 50, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["cci"] = ta.trend.cci(h, l, c, window=params["cci_period"])
        # StochRSI
        rsi = df["rsi"]
        n = params["stoch_period"]
        rsi_min = rsi.rolling(n).min()
        rsi_max = rsi.rolling(n).max()
        df["stoch_rsi"] = ((rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100).rolling(3).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        rsi_low = df["rsi"] < params["rsi_oversold"]
        cci_low = df["cci"] < params["cci_oversold"]
        stoch_low = df["stoch_rsi"] < 20
        oversold_votes = rsi_low.astype(int) + cci_low.astype(int) + stoch_low.astype(int)
        above_ema = df["close"] > df["ema_trend"]
        prev_fewer = (df["rsi"].shift(1) >= params["rsi_oversold"]).astype(int) + (df["cci"].shift(1) >= params["cci_oversold"]).astype(int)
        df["buy_signal"] = ((oversold_votes >= 2) & above_ema & (prev_fewer >= 1)).fillna(False)
        rsi_hi = df["rsi"] > params["rsi_overbought"]
        cci_hi = df["cci"] > params["cci_overbought"]
        overbought_votes = rsi_hi.astype(int) + cci_hi.astype(int)
        df["sell_signal"] = (overbought_votes >= 2).fillna(False)
        return df


# ─── 61. Adaptive RSI Mean Reversion ────────────────────────────────

class AdaptiveRsiMeanRev(BaseStrategy):
    name = "claude_adaptive_rsi_meanrev"
    description = "自适应RSI均值回归：根据波动率动态调整RSI阈值，高波动时放宽，低波动时收紧，提高胜率"
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA", "color": "#ff9800"}],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#2196f3"},
            {"col": "rsi_lower", "label": "RSI Lower", "color": "#4caf50", "type": "line"},
            {"col": "rsi_upper", "label": "RSI Upper", "color": "#f44336", "type": "line"},
        ],
    }
    default_params = {
        "rsi_period": 14, "base_oversold": 30, "base_overbought": 70,
        "atr_period": 14, "atr_ma_period": 50,
        "ema_period": 50, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        atr_ma = atr.rolling(params["atr_ma_period"]).mean()
        vol_ratio = (atr / (atr_ma + 1e-10)).clip(0.5, 2.0)
        # High vol -> widen thresholds (lower oversold, higher overbought)
        # Low vol -> tighten thresholds
        df["rsi_lower"] = params["base_oversold"] - (vol_ratio - 1) * 10
        df["rsi_upper"] = params["base_overbought"] + (vol_ratio - 1) * 10
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        rsi = df["rsi"]
        prev_rsi = rsi.shift(1)
        # Buy: RSI crosses below adaptive lower threshold
        below = rsi < df["rsi_lower"]
        prev_above = prev_rsi >= df["rsi_lower"].shift(1)
        above_ema = df["close"] > df["ema_trend"]
        df["buy_signal"] = (below & prev_above & above_ema).fillna(False)
        # Sell: RSI crosses above adaptive upper threshold
        above = rsi > df["rsi_upper"]
        prev_below = prev_rsi <= df["rsi_upper"].shift(1)
        df["sell_signal"] = (above & prev_below).fillna(False)
        return df


# ─── 62. Keltner Mean Reversion ─────────────────────────────────────

class KeltnerMeanRev(BaseStrategy):
    name = "claude_keltner_meanrev"
    description = "Keltner通道均值回归：价格触及Keltner下轨后回归中轨，ATR自适应通道+RSI确认"
    chart_indicators = {
        "overlay": [
            {"col": "kc_upper", "label": "KC Upper", "color": "#f44336"},
            {"col": "kc_lower", "label": "KC Lower", "color": "#4caf50"},
            {"col": "kc_mid", "label": "KC Mid", "color": "#9e9e9e"},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "kc_period": 20, "kc_mult": 2.0, "rsi_period": 14,
        "rsi_oversold": 35, "rsi_overbought": 70,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["kc_period"]
        df["kc_mid"] = ta.trend.ema_indicator(c, window=n)
        atr = ta.volatility.average_true_range(h, l, c, window=n)
        df["kc_upper"] = df["kc_mid"] + params["kc_mult"] * atr
        df["kc_lower"] = df["kc_mid"] - params["kc_mult"] * atr
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c, pc = df["close"], df["close"].shift(1)
        # Buy: price touches/crosses below lower band and bounces
        touch_lower = (c <= df["kc_lower"]) & (pc > df["kc_lower"].shift(1))
        rsi_low = df["rsi"] < params["rsi_oversold"]
        df["buy_signal"] = (touch_lower & rsi_low).fillna(False)
        # Sell: price reaches upper band or RSI overbought
        df["sell_signal"] = ((c >= df["kc_upper"]) | (df["rsi"] > params["rsi_overbought"])).fillna(False)
        return df


# ─── 63. Volume Profile Support ─────────────────────────────────────

class VolumeProfileSupport(BaseStrategy):
    name = "claude_volume_profile"
    description = "成交量密集区支撑：检测历史高成交量价格区域作为支撑，价格回落至支撑区反弹买入"
    chart_indicators = {
        "overlay": [{"col": "vol_poc", "label": "Vol POC", "color": "#9c27b0"}],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "lookback": 60, "bins": 20, "rsi_period": 14,
        "rsi_oversold": 40, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        n = params["lookback"]
        bins = params["bins"]
        # Calculate rolling POC (Point of Control - price with highest volume)
        poc = pd.Series(np.nan, index=df.index)
        for i in range(n, len(df)):
            window_c = c.iloc[i-n:i].values
            window_v = v.iloc[i-n:i].values
            hist, edges = np.histogram(window_c, bins=bins, weights=window_v)
            max_bin = np.argmax(hist)
            poc.iloc[i] = (edges[max_bin] + edges[max_bin+1]) / 2
        df["vol_poc"] = poc
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c, pc = df["close"], df["close"].shift(1)
        poc = df["vol_poc"]
        # Buy: price approaches POC from above and bounces
        near_poc = (c - poc).abs() / (poc + 1e-10) < 0.02  # within 2%
        bounce = c > df["open"]  # green candle
        rsi_ok = df["rsi"] < params["rsi_oversold"]
        prev_not_near = (pc - poc.shift(1)).abs() / (poc.shift(1) + 1e-10) >= 0.02
        df["buy_signal"] = (near_poc & bounce & rsi_ok & prev_not_near).fillna(False)
        # Sell: price moves far above POC
        far_above = (c - poc) / (poc + 1e-10) > 0.08
        df["sell_signal"] = (far_above | (df["rsi"] > 75)).fillna(False)
        return df


# ─── 64. Swing Failure Pattern ──────────────────────────────────────

class SwingFailure(BaseStrategy):
    name = "claude_swing_failure"
    description = "摆动失败形态：价格突破前高/低后快速回撤(假突破)，反向交易捕捉流动性猎取后的反转"
    chart_indicators = {
        "overlay": [
            {"col": "swing_high", "label": "Swing High", "color": "#f44336"},
            {"col": "swing_low", "label": "Swing Low", "color": "#4caf50"},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#2196f3"}],
    }
    default_params = {
        "swing_lookback": 10, "rsi_period": 14,
        "rsi_oversold": 40, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        n = params["swing_lookback"]
        df["swing_high"] = h.shift(1).rolling(n).max()
        df["swing_low"] = l.shift(1).rolling(n).min()
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        # Bullish SFP: price briefly drops below swing low but closes above it
        sfp_bull = (l < df["swing_low"]) & (c > df["swing_low"])
        rsi_ok = df["rsi"] < params["rsi_oversold"]
        prev_above = df["low"].shift(1) >= df["swing_low"].shift(1)
        df["buy_signal"] = (sfp_bull & rsi_ok & prev_above).fillna(False)
        # Bearish SFP: price briefly exceeds swing high but closes below it
        sfp_bear = (h > df["swing_high"]) & (c < df["swing_high"])
        df["sell_signal"] = (sfp_bear & (df["rsi"] > 60)).fillna(False)
        return df


# ─── 波动率政权切换 ──────────────────────────────────────────────────────

class VolatilityRegime(BaseStrategy):
    name = "claude_volatility_regime"
    description = "波动率政权切换：高波动→均值回归(RSI+BB)，低波动→趋势跟踪(EMA交叉)，自适应市况"
    indicator_formulas = {
        "vol_percentile": {"name": "Vol Percentile", "formula": "Rank(StdDev(Return, N), lookback)", "params": ["vol_period", "vol_lookback"]},
        "ema_fast": {"name": "EMA Fast", "formula": "EMA(close, N)", "params": ["ema_fast"]},
        "ema_slow": {"name": "EMA Slow", "formula": "EMA(close, N)", "params": ["ema_slow"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
        "bb_upper": {"name": "BB Upper", "formula": "SMA + K×Std", "params": ["bb_period", "bb_std"]},
        "bb_lower": {"name": "BB Lower", "formula": "SMA - K×Std", "params": ["bb_period", "bb_std"]},
    }
    signal_rules = {
        "buy": "高波动(>70pct): RSI<30 AND Close<BB_Lower; 低波动(<30pct): EMA_Fast上穿EMA_Slow",
        "sell": "高波动: RSI>70 OR Close>BB_Upper; 低波动: EMA_Fast下穿EMA_Slow",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#ff9800"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#2196f3"},
            {"col": "bb_upper", "label": "BB Upper", "color": "#555555", "dash": True},
            {"col": "bb_lower", "label": "BB Lower", "color": "#555555", "dash": True},
        ],
        "subchart": [
            {"col": "vol_percentile", "label": "Vol %ile", "color": "#9c27b0"},
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
        ],
    }
    default_params = {
        "vol_period": 15, "vol_lookback": 40,
        "high_vol_pct": 60, "low_vol_pct": 25,
        "ema_fast": 6, "ema_slow": 45,
        "rsi_period": 14, "rsi_oversold": 40,
        "bb_period": 20, "bb_std": 2.0,
        "atr_stop_mul": 4.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        # Realized volatility
        daily_ret = c.pct_change()
        realized_vol = daily_ret.rolling(params["vol_period"]).std()
        # Percentile rank over lookback
        lookback = params["vol_lookback"]
        df["vol_percentile"] = realized_vol.rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c = df["close"]
        vp = df["vol_percentile"]
        high_vol = vp > params["high_vol_pct"]
        low_vol = vp < params["low_vol_pct"]
        # High vol: mean reversion
        mr_buy = high_vol & (df["rsi"] < params["rsi_oversold"]) & (c < df["bb_lower"])
        mr_sell = high_vol & ((df["rsi"] > 70) | (c > df["bb_upper"]))
        # Low vol: trend following (EMA crossover)
        prev_fast = df["ema_fast"].shift(1)
        prev_slow = df["ema_slow"].shift(1)
        trend_buy = low_vol & (prev_fast <= prev_slow) & (df["ema_fast"] > df["ema_slow"])
        trend_sell = low_vol & (prev_fast >= prev_slow) & (df["ema_fast"] < df["ema_slow"])
        df["buy_signal"] = (mr_buy | trend_buy).fillna(False)
        df["sell_signal"] = (mr_sell | trend_sell).fillna(False)
        return df


# ─── 动量质量过滤 ────────────────────────────────────────────────────────

class MomentumQuality(BaseStrategy):
    name = "claude_momentum_quality"
    description = "动量质量过滤：仅在动量方向一致且连续性好时入场，过滤假突破"
    indicator_formulas = {
        "momentum": {"name": "Momentum", "formula": "ROC(close, N)", "params": ["mom_period"]},
        "quality": {"name": "Quality", "formula": "Count(return>0, N)", "params": ["quality_days"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_trend"]},
    }
    signal_rules = {
        "buy": "Momentum > 0 AND Quality >= threshold AND Close > EMA_trend",
        "sell": "Momentum < 0 AND Neg_Quality >= threshold OR ATR stop",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#2196f3"},
        ],
        "subchart": [
            {"col": "momentum", "label": "Momentum", "color": "#ff9800"},
            {"col": "quality", "label": "Quality", "color": "#4caf50", "type": "histogram"},
        ],
    }
    default_params = {
        "mom_period": 10, "quality_days": 5, "quality_threshold": 3,
        "ema_trend": 50, "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["momentum"] = ta.momentum.roc(c, window=params["mom_period"])
        daily_ret = c.pct_change()
        df["quality"] = (daily_ret > 0).rolling(params["quality_days"]).sum()
        df["neg_quality"] = (daily_ret < 0).rolling(params["quality_days"]).sum()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_trend"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        thresh = params["quality_threshold"]
        df["buy_signal"] = (
            (df["momentum"] > 0) &
            (df["quality"] >= thresh) &
            (df["close"] > df["ema_trend"])
        ).fillna(False)
        df["sell_signal"] = (
            (df["momentum"] < 0) &
            (df["neg_quality"] >= thresh)
        ).fillna(False)
        return df


# ─── ADX趋势强度策略 ────────────────────────────────────────────────────

class AdxTrendStrength(BaseStrategy):
    name = "claude_adx_trend_strength"
    description = "ADX趋势强度：仅在强趋势(ADX>25)中跟随+DI/-DI方向，弱趋势回避"
    indicator_formulas = {
        "adx": {"name": "ADX", "formula": "ADX(H, L, C, N)", "params": ["adx_period"]},
        "plus_di": {"name": "+DI", "formula": "+DI(H, L, C, N)", "params": ["adx_period"]},
        "minus_di": {"name": "-DI", "formula": "-DI(H, L, C, N)", "params": ["adx_period"]},
        "ema": {"name": "EMA", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "ADX > 25 AND +DI > -DI AND +DI上穿-DI AND Close > EMA",
        "sell": "-DI > +DI AND ADX > 20, OR ADX < 15",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema", "label": "EMA", "color": "#2196f3"},
        ],
        "subchart": [
            {"col": "adx", "label": "ADX", "color": "#ff9800"},
            {"col": "plus_di", "label": "+DI", "color": "#4caf50"},
            {"col": "minus_di", "label": "-DI", "color": "#f44336"},
        ],
    }
    default_params = {
        "adx_period": 14, "adx_strong": 25, "adx_weak": 15,
        "ema_period": 50, "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        adx_ind = ta.trend.ADXIndicator(h, l, c, window=params["adx_period"])
        df["adx"] = adx_ind.adx()
        df["plus_di"] = adx_ind.adx_pos()
        df["minus_di"] = adx_ind.adx_neg()
        df["ema"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        adx_strong = params["adx_strong"]
        adx_weak = params["adx_weak"]
        # +DI crosses above -DI
        prev_plus = df["plus_di"].shift(1)
        prev_minus = df["minus_di"].shift(1)
        di_cross_up = (prev_plus <= prev_minus) & (df["plus_di"] > df["minus_di"])
        df["buy_signal"] = (
            (df["adx"] > adx_strong) &
            (df["plus_di"] > df["minus_di"]) &
            di_cross_up &
            (df["close"] > df["ema"])
        ).fillna(False)
        df["sell_signal"] = (
            ((df["minus_di"] > df["plus_di"]) & (df["adx"] > 20)) |
            (df["adx"] < adx_weak)
        ).fillna(False)
        return df


# ─── 动量过滤均值回归 ────────────────────────────────────────────────────

class MeanReversionFiltered(BaseStrategy):
    name = "claude_mean_reversion_filtered"
    description = "动量过滤均值回归：ADX<25确认盘整 + 反转K线确认，修复原mean_reversion过拟合"
    indicator_formulas = {
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
        "bb_upper": {"name": "BB Upper", "formula": "SMA + K×Std", "params": ["bb_period", "bb_std"]},
        "bb_lower": {"name": "BB Lower", "formula": "SMA - K×Std", "params": ["bb_period", "bb_std"]},
        "bb_mid": {"name": "BB Mid", "formula": "SMA(close, N)", "params": ["bb_period"]},
        "adx": {"name": "ADX", "formula": "ADX(H, L, C, N)", "params": ["adx_period"]},
    }
    signal_rules = {
        "buy": "RSI < oversold AND Close <= BB_Lower AND ADX < 25 AND 近2根有阳线",
        "sell": "RSI > overbought OR Close >= BB_Upper OR ADX > 30",
    }
    chart_indicators = {
        "overlay": [
            {"col": "bb_upper", "label": "BB Upper", "color": "#555555", "dash": True},
            {"col": "bb_lower", "label": "BB Lower", "color": "#555555", "dash": True},
            {"col": "bb_mid", "label": "BB Mid", "color": "#9e9e9e", "dash": True},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "adx", "label": "ADX", "color": "#ff9800"},
        ],
    }
    default_params = {
        "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
        "bb_period": 20, "bb_std": 2.0,
        "adx_period": 14, "adx_max": 25,
        "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        adx_ind = ta.trend.ADXIndicator(h, l, c, window=params["adx_period"])
        df["adx"] = adx_ind.adx()
        # Green candle detection (close > open)
        df["green_candle"] = df["close"] > df["open"]
        df["green_recent"] = df["green_candle"] | df["green_candle"].shift(1)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"] = (
            (df["rsi"] < params["rsi_oversold"]) &
            (df["close"] <= df["bb_lower"]) &
            (df["adx"] < params["adx_max"]) &
            df["green_recent"]
        ).fillna(False)
        df["sell_signal"] = (
            (df["rsi"] > params["rsi_overbought"]) |
            (df["close"] >= df["bb_upper"]) |
            (df["adx"] > 30)
        ).fillna(False)
        return df


# ─── 短期反转 ────────────────────────────────────────────────────────────

class ShortTermReversal(BaseStrategy):
    name = "claude_short_term_reversal"
    description = "短期反转：大幅下跌后买入反弹（学术验证的短期反转效应），限时持仓"
    indicator_formulas = {
        "daily_return": {"name": "Daily Return", "formula": "(Close - Close[-1]) / Close[-1] × 100", "params": []},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period=14"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_trend"]},
    }
    signal_rules = {
        "buy": "Daily Return < -drop_pct% AND RSI < rsi_threshold AND Close > EMA_trend",
        "sell": "持仓超过hold_days OR RSI > 50 OR ATR stop",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#2196f3"},
        ],
        "subchart": [
            {"col": "rsi", "label": "RSI", "color": "#e91e63"},
            {"col": "daily_return", "label": "Daily Ret%", "color": "#ff9800", "type": "histogram"},
        ],
    }
    default_params = {
        "drop_pct": 3.0, "rsi_threshold": 25, "hold_days": 5,
        "ema_trend": 100, "atr_stop_mul": 2.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["daily_return"] = c.pct_change() * 100
        df["rsi"] = ta.momentum.rsi(c, window=14)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_trend"])
        # Rolling counter for days since buy signal
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        drop_pct = params["drop_pct"]
        # Buy on large single-day drops with RSI filter and trend filter
        raw_buy = (
            (df["daily_return"] < -drop_pct) &
            (df["rsi"] < params["rsi_threshold"]) &
            (df["close"] > df["ema_trend"])
        )
        # Sell: track days since buy and sell after hold_days or RSI recovery
        hold_days = params["hold_days"]
        buy_signal = pd.Series(False, index=df.index)
        sell_signal = pd.Series(False, index=df.index)
        days_held = 0
        in_position = False
        for i in range(len(df)):
            if not in_position:
                if raw_buy.iloc[i]:
                    buy_signal.iloc[i] = True
                    in_position = True
                    days_held = 0
            else:
                days_held += 1
                if days_held >= hold_days or df["rsi"].iloc[i] > 50:
                    sell_signal.iloc[i] = True
                    in_position = False
                    days_held = 0
        df["buy_signal"] = buy_signal
        df["sell_signal"] = sell_signal
        return df


# ─── 自适应加权集成V2 ────────────────────────────────────────────────────

class AdaptiveEnsembleV2(BaseStrategy):
    name = "claude_adaptive_ensemble_v2"
    description = "自适应加权集成V2：基于滚动Sharpe动态调整子策略权重，优于固定投票"
    indicator_formulas = {
        "weighted_score": {"name": "Weighted Buy Score", "formula": "Σ(weight_i × signal_i)", "params": ["rolling_window"]},
        "ema_fast": {"name": "EMA Fast", "formula": "EMA(close, 10)", "params": []},
        "ema_slow": {"name": "EMA Slow", "formula": "EMA(close, 25)", "params": []},
    }
    signal_rules = {
        "buy": "Weighted buy score > buy_threshold (子策略: DualMA/SuperTrend/KeltnerMR/MomentumBO)",
        "sell": "Weighted sell score > sell_threshold",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA10", "color": "#ff9800"},
            {"col": "ema_slow", "label": "EMA25", "color": "#2196f3"},
        ],
        "subchart": [
            {"col": "weighted_buy_score", "label": "Buy Score", "color": "#4caf50"},
        ],
    }
    default_params = {
        "rolling_window": 40, "min_weight": 0.05,
        "buy_threshold": 0.4, "sell_threshold": 0.5,
        "atr_stop_mul": 2.5,
    }

    def _compute_sub_signals(self, df):
        """Compute 4 sub-strategy signals inline (no circular dependency)."""
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # 1. DualMA: EMA10/EMA25 crossover
        ema10 = ta.trend.ema_indicator(c, window=10)
        ema25 = ta.trend.ema_indicator(c, window=25)
        dma_buy = (ema10.shift(1) <= ema25.shift(1)) & (ema10 > ema25)
        dma_sell = (ema10.shift(1) >= ema25.shift(1)) & (ema10 < ema25)

        # 2. SuperTrend: simplified ATR-based
        atr14 = ta.volatility.average_true_range(h, l, c, window=14)
        hl2 = (h + l) / 2
        st_upper = hl2 + 3.0 * atr14
        st_lower = hl2 - 3.0 * atr14
        st_dir = pd.Series(0, index=df.index, dtype=float)
        st_line = pd.Series(np.nan, index=df.index, dtype=float)
        start = atr14.first_valid_index()
        if start is not None:
            start_idx = df.index.get_loc(start)
            st_line.iloc[start_idx] = st_upper.iloc[start_idx]
            st_dir.iloc[start_idx] = -1
            for i in range(start_idx + 1, len(df)):
                if st_dir.iloc[i-1] == 1:
                    lb = max(st_lower.iloc[i], st_line.iloc[i-1]) if not np.isnan(st_line.iloc[i-1]) else st_lower.iloc[i]
                    if c.iloc[i] < lb:
                        st_line.iloc[i] = st_upper.iloc[i]
                        st_dir.iloc[i] = -1
                    else:
                        st_line.iloc[i] = lb
                        st_dir.iloc[i] = 1
                else:
                    ub = min(st_upper.iloc[i], st_line.iloc[i-1]) if not np.isnan(st_line.iloc[i-1]) else st_upper.iloc[i]
                    if c.iloc[i] > ub:
                        st_line.iloc[i] = st_lower.iloc[i]
                        st_dir.iloc[i] = 1
                    else:
                        st_line.iloc[i] = ub
                        st_dir.iloc[i] = -1
        st_buy = (st_dir == 1) & (st_dir.shift(1) == -1)
        st_sell = (st_dir == -1) & (st_dir.shift(1) == 1)

        # 3. KeltnerMR: close < keltner_lower and RSI < 35
        kc = ta.volatility.KeltnerChannel(h, l, c, window=20, window_atr=14)
        kc_lower = kc.keltner_channel_lband()
        kc_upper = kc.keltner_channel_hband()
        rsi14 = ta.momentum.rsi(c, window=14)
        km_buy = (c < kc_lower) & (rsi14 < 35)
        km_sell = (c > kc_upper) | (rsi14 > 70)

        # 4. MomentumBO: close > 20-day high and volume > 1.2x avg
        high20 = h.shift(1).rolling(20).max()
        vol_avg = v.rolling(20).mean()
        mb_buy = (c > high20) & (v > 1.2 * vol_avg)
        low10 = l.shift(1).rolling(10).min()
        mb_sell = c < low10

        return (
            [dma_buy, st_buy, km_buy, mb_buy],
            [dma_sell, st_sell, km_sell, mb_sell],
            ema10, ema25
        )

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        buy_sigs, sell_sigs, ema10, ema25 = self._compute_sub_signals(df)

        df["ema_fast"] = ema10
        df["ema_slow"] = ema25
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)

        # Store sub-strategy signal returns for rolling Sharpe weighting
        daily_ret = c.pct_change()
        window = params["rolling_window"]
        min_w = params["min_weight"]

        # For each sub-strategy, compute rolling performance
        weights = []
        for sig_buy in buy_sigs:
            # Simple: return after buy signals
            sig_ret = daily_ret.where(sig_buy.shift(1).fillna(False), 0)
            roll_mean = sig_ret.rolling(window, min_periods=10).mean()
            roll_std = sig_ret.rolling(window, min_periods=10).std()
            roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)).fillna(0)
            weights.append(roll_sharpe.clip(lower=0) + min_w)

        # Normalize weights
        total_w = sum(weights)
        total_w = total_w.replace(0, 1)  # avoid div by zero
        norm_weights = [w / total_w for w in weights]

        # Compute weighted buy/sell scores
        buy_score = sum(w * sig.astype(float).fillna(0) for w, sig in zip(norm_weights, buy_sigs))
        sell_score = sum(w * sig.astype(float).fillna(0) for w, sig in zip(norm_weights, sell_sigs))

        df["weighted_buy_score"] = buy_score
        df["weighted_sell_score"] = sell_score
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"] = (df["weighted_buy_score"] > params["buy_threshold"]).fillna(False)
        df["sell_signal"] = (df["weighted_sell_score"] > params["sell_threshold"]).fillna(False)
        return df


# ─── New 1. FFT Dominant Cycle ───────────────────────────────────────────────

class FftDominantCycle(BaseStrategy):
    name = "claude_fft_cycle_trend"
    description = "FFT主导周期：快速傅里叶变换识别价格主导波动周期，归一化周期值在低谷买入高峰卖出，EMA趋势过滤"
    indicator_formulas = {
        "cycle_value": {"name": "Cycle Value (normalized)", "formula": "IFFT(dominant_band) / std", "params": ["fft_window", "min_period", "max_period"]},
        "ema_trend":   {"name": "EMA Trend", "formula": "EMA(Close, ema_period)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy":  "cycle_value crosses up through buy_threshold (周期低谷) AND close > EMA趋势",
        "sell": "cycle_value crosses down through sell_threshold (周期高峰)",
    }
    chart_indicators = {
        "overlay":  [{"col": "ema_trend",   "label": "EMA Trend",   "color": "#ff9800"}],
        "subchart": [{"col": "cycle_value", "label": "Cycle Value", "color": "#9c27b0"}],
    }
    default_params = {
        "fft_window":     35,    # rolling FFT window (bars)
        "min_period":      3,    # shortest cycle to extract (bars)
        "max_period":     15,    # longest cycle to extract (bars)
        "ema_period":     30,
        "buy_threshold":  -0.1,  # cycle crosses UP through this → buy
        "sell_threshold":  0.7,  # cycle crosses DOWN through this → sell
        "atr_stop_mul":    3.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]

        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)

        n        = params["fft_window"]
        min_p    = params["min_period"]
        max_p    = params["max_period"]
        close_arr = c.values.astype(float)
        cycle_vals = np.full(len(close_arr), np.nan)

        freqs_ref = np.fft.rfftfreq(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            periods_ref = np.where(freqs_ref > 0, 1.0 / freqs_ref, np.inf)
        valid_mask = (periods_ref >= min_p) & (periods_ref <= max_p)

        hann = np.hanning(n)
        for i in range(n, len(close_arr)):
            seg = close_arr[i - n:i]
            # Detrend by subtracting linear trend
            trend_line = np.linspace(seg[0], seg[-1], n)
            detrended  = (seg - trend_line) * hann
            # FFT → keep only dominant-period band → IFFT
            fft_vals = np.fft.rfft(detrended)
            filtered = np.zeros_like(fft_vals)
            filtered[valid_mask] = fft_vals[valid_mask]
            sig = np.fft.irfft(filtered, n)
            std = sig.std()
            cycle_vals[i] = sig[-1] / std if std > 1e-10 else 0.0

        df["cycle_value"] = cycle_vals
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        cv      = df["cycle_value"]
        prev_cv = cv.shift(1)
        in_up   = df["close"] > df["ema_trend"]

        df["buy_signal"]  = (
            (cv > params["buy_threshold"]) &
            (prev_cv <= params["buy_threshold"]) &
            in_up
        ).fillna(False)
        df["sell_signal"] = (
            (cv < params["sell_threshold"]) &
            (prev_cv >= params["sell_threshold"])
        ).fillna(False)
        return df

# ─── New B. Linear Regression Trend Quality ──────────────────────────────────

class LinearRegressionTrend(BaseStrategy):
    name = "claude_linreg_trend"
    description = "线性回归趋势质量：滚动OLS拟合斜率和R²，高R²+正斜率确认优质趋势，回调至回归中线以下后反弹入场"
    indicator_formulas = {
        "linreg_mid":   {"name": "LinReg Mid",   "formula": "OLS_predict(Close[-n:])",              "params": ["linreg_period"]},
        "linreg_upper": {"name": "LinReg Upper", "formula": "LinReg + band_std × resid_std",        "params": ["band_std"]},
        "linreg_lower": {"name": "LinReg Lower", "formula": "LinReg - band_std × resid_std",        "params": ["band_std"]},
        "linreg_r2":    {"name": "R² Quality",   "formula": "1 - SS_res/SS_tot",                   "params": []},
        "slope_pct":    {"name": "Slope (%/bar)","formula": "OLS_slope / Close × 100",              "params": []},
    }
    signal_rules = {
        "buy":  "R²>r2_min AND slope_pct>slope_min AND close穿越回归中线向上 (回调后确认)",
        "sell": "slope_pct由正转负 (趋势反转)",
    }
    chart_indicators = {
        "overlay":  [
            {"col": "linreg_mid",   "label": "LinReg",       "color": "#2196f3"},
            {"col": "linreg_upper", "label": "LinReg Upper", "color": "#555555", "dash": True},
            {"col": "linreg_lower", "label": "LinReg Lower", "color": "#555555", "dash": True},
        ],
        "subchart": [{"col": "linreg_r2", "label": "R²", "color": "#4caf50"}],
    }
    default_params = {
        "linreg_period": 20,
        "slope_min":      0.11,  # min slope (%/bar) to confirm uptrend
        "r2_min":         0.35,  # min R² for quality trend
        "band_std":        3.0,
        "ema_period":     30,    # secondary trend filter
        "rsi_period":     14,
        "rsi_max":        75,    # not overbought on buy
        "atr_stop_mul":    3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]

        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"]       = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)

        n = params["linreg_period"]
        prices = c.values.astype(float)
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        ss_xx  = ((x - x_mean) ** 2).sum()

        band_std = params["band_std"]
        lr_mid   = np.full(len(prices), np.nan)
        lr_upper = np.full(len(prices), np.nan)
        lr_lower = np.full(len(prices), np.nan)
        lr_slope = np.full(len(prices), np.nan)
        lr_r2    = np.full(len(prices), np.nan)

        for i in range(n - 1, len(prices)):
            y      = prices[i - n + 1:i + 1]
            y_mean = y.mean()
            ss_xy  = ((x - x_mean) * (y - y_mean)).sum()
            ss_yy  = ((y - y_mean) ** 2).sum()
            slope  = ss_xy / ss_xx
            intercept = y_mean - slope * x_mean
            y_hat  = intercept + slope * (n - 1)   # predict at current bar
            y_pred = intercept + slope * x
            resid  = y - y_pred
            ss_res = (resid ** 2).sum()
            r2     = max(0.0, 1.0 - ss_res / ss_yy) if ss_yy > 1e-10 else 0.0
            r_std  = resid.std()
            lr_mid[i]   = y_hat
            lr_slope[i] = slope
            lr_r2[i]    = r2
            lr_upper[i] = y_hat + band_std * r_std
            lr_lower[i] = y_hat - band_std * r_std

        df["linreg_mid"]   = lr_mid
        df["linreg_upper"] = lr_upper
        df["linreg_lower"] = lr_lower
        df["linreg_r2"]    = lr_r2
        # Slope as % per bar
        df["slope_pct"]    = pd.Series(lr_slope, index=df.index) / c * 100
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        c         = df["close"]
        good_trend = (df["linreg_r2"] > params["r2_min"]) & (df["slope_pct"] > params["slope_min"])
        rsi_ok     = df["rsi"] < params["rsi_max"]

        above_mid  = c > df["linreg_mid"]
        prev_above = above_mid.shift(1).fillna(False)

        # Buy: good-quality uptrend AND price just crossed back above LinReg midline
        df["buy_signal"]  = (
            good_trend & above_mid & ~prev_above & rsi_ok
        ).fillna(False)
        # Sell: slope just turned negative
        df["sell_signal"] = (
            (df["slope_pct"] < 0) & (df["slope_pct"].shift(1) >= 0)
        ).fillna(False)
        return df


# ─── New C. Volatility-Adjusted Momentum ────────────────────────────────────

class VolAdjMomentum(BaseStrategy):
    name = "claude_vol_adj_momentum"
    description = "波动率调整动量：N日收益率除以已实现波动率，得到Sharpe式动量评分，高分穿越阈值买入，信号EMA平滑，趋势+低RSI过滤"
    indicator_formulas = {
        "vol_adj_signal": {"name": "Vol-Adj Mom", "formula": "EMA(N_day_return / realized_vol, signal_ema)", "params": ["mom_period", "vol_period", "signal_ema"]},
        "ema_trend":      {"name": "EMA Trend",   "formula": "EMA(Close, ema_period)",                       "params": ["ema_period"]},
    }
    signal_rules = {
        "buy":  "vol_adj_signal穿越buy_threshold向上 AND close>EMA趋势",
        "sell": "vol_adj_signal穿越sell_threshold向下",
    }
    chart_indicators = {
        "overlay":  [{"col": "ema_trend",      "label": "EMA Trend",    "color": "#ff9800"}],
        "subchart": [{"col": "vol_adj_signal", "label": "Vol-Adj Mom",  "color": "#2196f3"}],
    }
    default_params = {
        "mom_period":      10,
        "vol_period":      15,
        "signal_ema":      11,
        "buy_threshold":  0.2,   # vol-adj momentum to trigger buy
        "sell_threshold": 0.5,   # vol-adj momentum to trigger sell (protect profit)
        "ema_period":     25,
        "atr_stop_mul":    3.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]

        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)

        log_ret = np.log(c / c.shift(1))
        # N-day return
        mom = c.pct_change(params["mom_period"])
        # Realized vol (annualized std of daily log returns)
        vol = log_ret.rolling(params["vol_period"]).std() * np.sqrt(252)
        # Vol-adjusted momentum
        vol_adj = mom / vol.replace(0, np.nan)
        # Smooth with EMA
        df["vol_adj_signal"] = vol_adj.ewm(span=params["signal_ema"], adjust=False).mean()
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        vas      = df["vol_adj_signal"]
        prev_vas = vas.shift(1)
        in_up    = df["close"] > df["ema_trend"]

        df["buy_signal"]  = (
            (vas > params["buy_threshold"]) &
            (prev_vas <= params["buy_threshold"]) &
            in_up
        ).fillna(False)
        df["sell_signal"] = (
            (vas < params["sell_threshold"]) &
            (prev_vas >= params["sell_threshold"])
        ).fillna(False)
        return df


# ─── New C. Seasonality Calendar Effect ─────────────────────────────────────

class SeasonalityCalendar(BaseStrategy):
    name = "claude_seasonal_calendar"
    description = "季节性日历效应：月度强弱+星期效应+最佳6个月窗口，日历得分穿越阈值配合EMA趋势+RSI过滤"
    indicator_formulas = {
        "season_score": {"name": "Season Score", "formula": "month_weight + dow_weight (normalized)", "params": ["month_boost", "dow_boost"]},
        "ema_trend":    {"name": "EMA Trend",    "formula": "EMA(Close, ema_period)",                  "params": ["ema_period"]},
        "rsi":          {"name": "RSI",           "formula": "RSI(Close, rsi_period)",                  "params": ["rsi_period"]},
    }
    signal_rules = {
        "buy":  "season_score > buy_score AND close > EMA趋势 AND RSI < rsi_sell AND season_score穿越阈值",
        "sell": "season_score < sell_score OR close < EMA趋势 OR RSI > rsi_sell",
    }
    chart_indicators = {
        "overlay":  [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [
            {"col": "season_score", "label": "Season Score", "color": "#4caf50"},
            {"col": "rsi",          "label": "RSI",          "color": "#e91e63"},
        ],
    }
    default_params = {
        "ema_period":   30,    # trend EMA period
        "rsi_period":   21,
        "rsi_sell":     80,    # RSI overbought exit
        "buy_score":    0.35,  # season score threshold to buy
        "sell_score":   0.15,  # season score threshold to exit
        "month_boost":  0.75,  # multiplier for strong-month weights
        "dow_boost":    1.75,  # multiplier for strong-dow weights
        "atr_stop_mul": 2.0,
    }

    # Month weights: 1-indexed; 1月/11月/12月/4月最强
    _MONTH_W = {1:1.2, 2:0.6, 3:0.9, 4:1.1, 5:0.5, 6:0.8,
                7:0.7, 8:0.6, 9:0.4, 10:0.9, 11:1.3, 12:1.2}
    # DoW weights: 0=Mon…4=Fri; 周四/周五略强
    _DOW_W   = {0:0.6, 1:0.8, 2:0.9, 3:1.1, 4:1.1}

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]

        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"]       = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)

        mb = params["month_boost"]
        db = params["dow_boost"]

        # Raw calendar score per bar
        months = df.index.month
        dows   = df.index.dayofweek

        month_raw = np.array([self._MONTH_W.get(int(m), 0.8) for m in months])
        dow_raw   = np.array([self._DOW_W.get(int(d), 0.8)   for d in dows])

        # Apply boosts and normalize to [0,1] range
        month_raw = month_raw * mb
        dow_raw   = dow_raw   * db

        # Combined score: weighted average, normalize by max possible
        raw_max   = max(self._MONTH_W.values()) * mb
        raw_dow_max = max(self._DOW_W.values()) * db
        m_norm    = (month_raw - month_raw.min()) / (month_raw.max() - month_raw.min() + 1e-9)
        d_norm    = (dow_raw   - dow_raw.min())   / (dow_raw.max()   - dow_raw.min()   + 1e-9)

        # Smooth with 5-day EMA to reduce daily noise
        raw_score = pd.Series(0.6 * m_norm + 0.4 * d_norm, index=df.index)
        df["season_score"] = raw_score.ewm(span=5).mean()
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        sc       = df["season_score"]
        prev_sc  = sc.shift(1)
        in_trend = df["close"] > df["ema_trend"]
        rsi_ok   = df["rsi"] < params["rsi_sell"]

        # Buy: score crosses UP through buy_score threshold + trend + rsi filter
        df["buy_signal"]  = (
            (sc > params["buy_score"]) &
            (prev_sc <= params["buy_score"]) &
            in_trend & rsi_ok
        ).fillna(False)
        # Sell: score drops below sell_score OR trend breaks
        df["sell_signal"] = (
            ((sc < params["sell_score"]) & (prev_sc >= params["sell_score"])) |
            (~in_trend & (df["close"] < df["ema_trend"]))
        ).fillna(False)
        return df


# ─── New D. GARCH(1,1) Volatility Timing ─────────────────────────────────────

class GarchVolTiming(BaseStrategy):
    name = "claude_garch_vol_timing"
    description = "GARCH波动率择时：GARCH(1,1)预测条件波动率，低波动期跟随EMA趋势顺势入场，高波动期空仓规避风险"
    indicator_formulas = {
        "garch_vol":   {"name": "GARCH Cond.Vol (%)", "formula": "sqrt(GARCH(1,1) variance) × sqrt(252)", "params": ["garch_omega", "garch_alpha", "garch_beta"]},
        "vol_ratio":   {"name": "Vol Ratio",           "formula": "garch_vol / rolling_mean(garch_vol, N)", "params": ["vol_ma"]},
        "ema_fast":    {"name": "EMA Fast",            "formula": "EMA(Close, ema_fast)",                   "params": ["ema_fast"]},
        "ema_slow":    {"name": "EMA Slow",            "formula": "EMA(Close, ema_slow)",                   "params": ["ema_slow"]},
    }
    signal_rules = {
        "buy":  "vol_ratio < vol_buy_thresh AND ema_fast穿越ema_slow向上 (低波动期趋势追随)",
        "sell": "vol_ratio > vol_sell_thresh OR ema_fast跌破ema_slow (高波动期空仓)",
    }
    chart_indicators = {
        "overlay":  [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#ff9800"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#2196f3"},
        ],
        "subchart": [
            {"col": "vol_ratio", "label": "Vol Ratio", "color": "#9c27b0"},
            {"col": "garch_vol", "label": "GARCH Vol", "color": "#f44336"},
        ],
    }
    default_params = {
        "garch_omega":     0.00001, # GARCH ω (long-run var component)
        "garch_alpha":     0.05,    # GARCH α (ARCH effect)
        "garch_beta":      0.80,    # GARCH β (GARCH persistence)
        "vol_ma":          20,      # rolling window for vol baseline
        "vol_buy_thresh":  1.2,     # enter only when vol_ratio < this
        "vol_sell_thresh": 1.9,     # exit when vol_ratio > this
        "ema_fast":        5,
        "ema_slow":        20,
        "atr_stop_mul":    3.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]

        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        df["atr"]      = ta.volatility.average_true_range(h, l, c, window=14)

        # Log returns
        log_ret = np.log(c / c.shift(1)).fillna(0.0).values
        n       = len(log_ret)

        omega = max(params["garch_omega"], 1e-9)
        alpha = max(params["garch_alpha"], 0.0)
        beta  = max(params["garch_beta"],  0.0)
        # Clamp to ensure stationarity
        if alpha + beta >= 1.0:
            total = alpha + beta + 1e-6
            alpha = alpha / total * 0.99
            beta  = beta  / total * 0.99

        # Initialise variance with unconditional variance
        init_var = np.var(log_ret[:20]) if n > 20 else 1e-4
        init_var = max(init_var, 1e-8)

        cond_var = np.empty(n)
        cond_var[0] = init_var
        for t in range(1, n):
            cond_var[t] = omega + alpha * log_ret[t-1]**2 + beta * cond_var[t-1]

        # Annualised conditional volatility (%)
        garch_vol = np.sqrt(np.maximum(cond_var, 0)) * np.sqrt(252) * 100.0
        df["garch_vol"] = garch_vol

        # Rolling mean of vol for ratio
        df["vol_baseline"] = pd.Series(garch_vol, index=df.index).rolling(params["vol_ma"]).mean()
        df["vol_ratio"]    = df["garch_vol"] / (df["vol_baseline"] + 1e-9)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        vr        = df["vol_ratio"]
        ef        = df["ema_fast"]
        es        = df["ema_slow"]
        prev_ef   = ef.shift(1)
        prev_es   = es.shift(1)

        low_vol   = vr < params["vol_buy_thresh"]
        high_vol  = vr > params["vol_sell_thresh"]
        ema_cross_up   = (ef > es) & (prev_ef <= prev_es)
        ema_cross_down = (ef < es) & (prev_ef >= prev_es)

        df["buy_signal"]  = (ema_cross_up & low_vol).fillna(False)
        df["sell_signal"] = (ema_cross_down | high_vol).fillna(False)
        return df


# ─── New E. Candlestick Pattern + Volume Confirmation ────────────────────────

class CandlePatternVol(BaseStrategy):
    name = "claude_candle_pattern_vol"
    description = "K线组合形态+量能确认：看涨吞噬、锤子线、晨星、刺穿线，成交量放大+EMA趋势过滤，纯价格行为策略"
    indicator_formulas = {
        "body":        {"name": "K线实体", "formula": "abs(close - open)",                      "params": []},
        "upper_wick":  {"name": "上影线",  "formula": "high - max(open, close)",                "params": []},
        "lower_wick":  {"name": "下影线",  "formula": "min(open, close) - low",                 "params": []},
        "vol_ratio":   {"name": "成交量比", "formula": "volume / SMA(volume, vol_period)",       "params": ["vol_period"]},
        "pattern_score":{"name":"形态得分","formula": "bullish_engulfing | hammer | morning_star | piercing", "params": []},
        "ema_trend":   {"name": "EMA Trend","formula": "EMA(close, ema_period)",                "params": ["ema_period"]},
    }
    signal_rules = {
        "buy":  "识别到看涨形态(吞噬/锤子/晨星/刺穿) AND 成交量放大 AND 收盘>EMA趋势",
        "sell": "识别到看跌吞噬形态 OR close跌破EMA趋势 OR RSI超买",
    }
    chart_indicators = {
        "overlay":  [{"col": "ema_trend", "label": "EMA Trend", "color": "#ff9800"}],
        "subchart": [
            {"col": "vol_ratio",    "label": "Vol Ratio",    "color": "#2196f3"},
            {"col": "pattern_score","label": "Pattern Score","color": "#4caf50", "type": "histogram"},
        ],
    }
    default_params = {
        "ema_period":       25,    # trend filter EMA
        "vol_period":       10,    # volume baseline window
        "vol_min":          1.4,   # min volume ratio for confirmation
        "body_ratio_min":   0.25,  # min body / total range for engulfing
        "hammer_ratio":     1.5,   # lower wick / body ratio for hammer
        "rsi_period":       17,
        "rsi_sell":         85,
        "atr_stop_mul":     3.0,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"]       = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"]       = ta.volatility.average_true_range(h, l, c, window=14)

        body       = (c - o).abs()
        total_rng  = (h - l).replace(0, np.nan)
        upper_wick = h - c.where(c > o, o)
        lower_wick = c.where(c < o, o) - l
        vol_ma     = v.rolling(params["vol_period"]).mean()
        df["vol_ratio"] = v / (vol_ma + 1e-9)

        prev_o = o.shift(1); prev_c = c.shift(1); prev_h = h.shift(1); prev_l = l.shift(1)
        prev_body  = (prev_c - prev_o).abs()
        prev_total = (prev_h - prev_l).replace(0, np.nan)
        br_min     = params["body_ratio_min"]

        # ── Pattern 1: Bullish Engulfing ──────────────────────────────────────
        # prev: bearish candle; current: bullish candle that fully covers prev body
        prev_bearish  = prev_c < prev_o
        curr_bullish  = c > o
        engulf        = curr_bullish & prev_bearish & (o <= prev_c) & (c >= prev_o)
        body_ok       = (body / total_rng) >= br_min
        bullish_engulf = engulf & body_ok

        # ── Pattern 2: Hammer ────────────────────────────────────────────────
        # Small body at top, lower_wick >= hammer_ratio * body, upper_wick small
        small_body    = (body / total_rng) < 0.4
        long_lower    = lower_wick >= params["hammer_ratio"] * (body + 1e-9)
        small_upper   = upper_wick <= 0.3 * total_rng
        hammer        = small_body & long_lower & small_upper

        # ── Pattern 3: Morning Star (3-bar) ──────────────────────────────────
        # Bar-1: big bearish; Bar0 (prev): small body (indecision); current: big bullish
        bar2_o = o.shift(2); bar2_c = c.shift(2)
        bar1_body = prev_body
        bar2_body = (bar2_c - bar2_o).abs()
        bar2_total = ((h.shift(2) - l.shift(2)).replace(0, np.nan))
        bar2_bearish = bar2_c < bar2_o
        small_middle = (bar1_body / prev_total) < 0.3
        morning_star = (
            bar2_bearish &
            small_middle &
            curr_bullish &
            (body / total_rng >= br_min) &
            (c > (bar2_o + bar2_c) / 2)   # closes above midpoint of bar-2
        )

        # ── Pattern 4: Piercing Line ─────────────────────────────────────────
        # prev: bearish; current: opens below prev_low, closes > midpoint of prev body
        prev_mid  = (prev_o + prev_c) / 2
        piercing  = (
            prev_bearish &
            curr_bullish &
            (o < prev_l) &
            (c > prev_mid) &
            (c < prev_o)
        )

        # ── Bearish Engulfing (for sell) ──────────────────────────────────────
        prev_bullish  = prev_c > prev_o
        curr_bearish  = c < o
        bearish_engulf = curr_bearish & prev_bullish & (o >= prev_c) & (c <= prev_o)

        # Combined pattern score
        bull_pattern = bullish_engulf | hammer | morning_star | piercing
        df["pattern_score"]    = bull_pattern.astype(int)
        df["bearish_engulf"]   = bearish_engulf.astype(int)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        bull = df["pattern_score"].astype(bool)
        vol_ok     = df["vol_ratio"] >= params["vol_min"]
        in_trend   = df["close"] > df["ema_trend"]
        rsi_ok     = df["rsi"] < params["rsi_sell"]

        df["buy_signal"]  = (bull & vol_ok & in_trend & rsi_ok).fillna(False)
        df["sell_signal"] = (
            df["bearish_engulf"].astype(bool) |
            (~in_trend & (df["close"] < df["ema_trend"])) |
            (df["rsi"] > params["rsi_sell"])
        ).fillna(False)
        return df


# ─── 新研发策略: 分形/信息论/市场效率 ────────────────────────────────

class HurstExponentStrategy(BaseStrategy):
    name = "claude_hurst_exponent"
    description = "分形分析：滚动Hurst指数判断市场状态，H>0.5趋势跟随+EMA确认，H<0.5均值回归+布林带，自适应切换"
    indicator_formulas = {
        "hurst":     {"name": "Hurst Exponent", "formula": "R/S rescaled range over N periods", "params": ["hurst_window"]},
        "ema_fast":  {"name": "EMA Fast",       "formula": "EMA(close, N)",                    "params": ["ema_fast"]},
        "ema_slow":  {"name": "EMA Slow",       "formula": "EMA(close, N)",                    "params": ["ema_slow"]},
        "bb_upper":  {"name": "BB Upper",       "formula": "SMA(close, N) + K × Std",          "params": ["bb_period", "bb_std"]},
        "bb_lower":  {"name": "BB Lower",       "formula": "SMA(close, N) - K × Std",          "params": ["bb_period", "bb_std"]},
    }
    signal_rules = {
        "buy":  "H>0.55: EMA_fast > EMA_slow (趋势); H<0.45: close < BB_lower (均值回归)",
        "sell": "H>0.55: EMA_fast < EMA_slow; H<0.45: close > BB_upper; 或 ATR止损",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_fast", "label": "EMA Fast", "color": "#ff9800"},
            {"col": "ema_slow", "label": "EMA Slow", "color": "#2196f3"},
            {"col": "bb_upper", "label": "BB Upper", "color": "#999", "dash": True},
            {"col": "bb_lower", "label": "BB Lower", "color": "#999", "dash": True},
        ],
        "subchart": [
            {"col": "hurst", "label": "Hurst Exponent", "color": "#e91e63"},
        ],
    }
    default_params = {
        "hurst_window":   50,    # R/S analysis window (optimized)
        "hurst_trend":    0.535, # threshold for trending regime
        "hurst_mr":       0.43,  # threshold for mean-reversion regime
        "ema_fast":       20,
        "ema_slow":       40,
        "bb_period":      30,
        "bb_std":         1.83,
        "rsi_period":     13,
        "rsi_oversold":   38,
        "rsi_overbought": 84,
        "atr_stop_mul":   2.36,
    }

    @staticmethod
    def _rolling_hurst(series, window):
        """Compute rolling Hurst exponent using R/S method"""
        hurst_vals = np.full(len(series), np.nan)
        vals = series.values
        for i in range(window, len(vals)):
            segment = vals[i - window:i]
            returns = np.diff(segment) / (segment[:-1] + 1e-10)
            if len(returns) < 20:
                continue
            mean_r = np.mean(returns)
            deviate = np.cumsum(returns - mean_r)
            R = np.max(deviate) - np.min(deviate)
            S = np.std(returns, ddof=1)
            if S < 1e-10:
                continue
            rs = R / S
            if rs > 0:
                hurst_vals[i] = np.log(rs) / np.log(window)
        return pd.Series(hurst_vals, index=series.index)

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["hurst"] = self._rolling_hurst(c, params["hurst_window"])
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"],
                                          window_dev=params["bb_std"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["rsi"]      = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"]      = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        H = df["hurst"]
        trending   = H > params["hurst_trend"]
        mean_rev   = H < params["hurst_mr"]

        # Trend-following signals
        ema_bull = df["ema_fast"] > df["ema_slow"]
        ema_bear = df["ema_fast"] < df["ema_slow"]

        # Mean-reversion signals
        below_bb = df["close"] < df["bb_lower"]
        above_bb = df["close"] > df["bb_upper"]
        rsi_os   = df["rsi"] < params["rsi_oversold"]
        rsi_ob   = df["rsi"] > params["rsi_overbought"]

        df["buy_signal"] = (
            (trending & ema_bull) |
            (mean_rev & below_bb & rsi_os)
        ).fillna(False)

        df["sell_signal"] = (
            (trending & ema_bear) |
            (mean_rev & above_bb & rsi_ob)
        ).fillna(False)
        return df


class EntropyMomentum(BaseStrategy):
    name = "claude_entropy_momentum"
    description = "信息论策略：Shannon熵衡量市场随机性，低熵=可预测=顺势交易，高熵=混沌=规避，结合动量过滤"
    indicator_formulas = {
        "entropy":      {"name": "Shannon Entropy", "formula": "-Σ(p_i × log2(p_i)) over N bins of returns", "params": ["entropy_window", "entropy_bins"]},
        "momentum":     {"name": "Momentum",        "formula": "ROC(close, N)",                              "params": ["mom_period"]},
        "ema_trend":    {"name": "EMA Trend",        "formula": "EMA(close, N)",                              "params": ["ema_period"]},
        "entropy_ma":   {"name": "Entropy MA",       "formula": "SMA(entropy, N)",                            "params": ["entropy_smooth"]},
    }
    signal_rules = {
        "buy":  "Entropy < entropy_low (可预测市场) AND Momentum > 0 AND close > EMA_trend",
        "sell": "Entropy > entropy_high (高随机性) OR Momentum < -mom_exit OR close < EMA_trend",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#2196f3"},
        ],
        "subchart": [
            {"col": "entropy_ma", "label": "Entropy (smoothed)", "color": "#e91e63"},
            {"col": "momentum",   "label": "Momentum",           "color": "#4caf50"},
        ],
    }
    default_params = {
        "entropy_window": 12,    # window to compute entropy (optimized)
        "entropy_bins":   7,     # histogram bins for return distribution
        "entropy_smooth": 5,     # smooth entropy with MA
        "entropy_low":    2.95,  # low entropy threshold (structured)
        "entropy_high":   3.22,  # high entropy threshold (chaotic)
        "mom_period":     12,    # momentum (ROC) period
        "mom_exit":       6.69,  # negative momentum exit threshold (%)
        "ema_period":     17,    # trend filter EMA
        "rsi_period":     10,
        "rsi_sell":       90,
        "atr_stop_mul":   2.47,
    }

    @staticmethod
    def _rolling_entropy(series, window, bins):
        """Compute rolling Shannon entropy of return distribution"""
        returns = series.pct_change()
        entropy_vals = np.full(len(series), np.nan)
        ret_vals = returns.values
        for i in range(window + 1, len(ret_vals)):
            seg = ret_vals[i - window:i]
            seg = seg[~np.isnan(seg)]
            if len(seg) < bins:
                continue
            counts, _ = np.histogram(seg, bins=bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy_vals[i] = -np.sum(probs * np.log2(probs))
        return pd.Series(entropy_vals, index=series.index)

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l = df["close"], df["high"], df["low"]
        df["entropy"] = self._rolling_entropy(c, params["entropy_window"], params["entropy_bins"])
        df["entropy_ma"] = df["entropy"].rolling(params["entropy_smooth"]).mean()
        df["momentum"] = ta.momentum.roc(c, window=params["mom_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        low_entropy  = df["entropy_ma"] < params["entropy_low"]
        high_entropy = df["entropy_ma"] > params["entropy_high"]
        mom_up       = df["momentum"] > 0
        mom_crash    = df["momentum"] < -params["mom_exit"]
        in_trend     = df["close"] > df["ema_trend"]

        df["buy_signal"] = (low_entropy & mom_up & in_trend).fillna(False)
        df["sell_signal"] = (
            high_entropy | mom_crash | (df["rsi"] > params["rsi_sell"])
        ).fillna(False)
        return df


class KaufmanEfficiencyBreakout(BaseStrategy):
    name = "claude_kaufman_efficiency"
    description = "市场效率策略：Kaufman效率比率(ER)识别强趋势(ER高=信噪比高)，结合ATR通道突破和成交量确认"
    indicator_formulas = {
        "efficiency_ratio": {"name": "Kaufman ER",  "formula": "abs(close-close[N]) / Σ|close[i]-close[i-1]|", "params": ["er_period"]},
        "kama":             {"name": "KAMA",         "formula": "Adaptive MA based on ER",                      "params": ["er_period", "kama_fast", "kama_slow"]},
        "atr_upper":        {"name": "ATR Upper",    "formula": "KAMA + atr_mul × ATR(N)",                      "params": ["atr_period", "atr_mul"]},
        "atr_lower":        {"name": "ATR Lower",    "formula": "KAMA - atr_mul × ATR(N)",                      "params": ["atr_period", "atr_mul"]},
        "vol_ratio":        {"name": "Vol Ratio",    "formula": "Volume / SMA(Volume, N)",                      "params": ["vol_period"]},
    }
    signal_rules = {
        "buy":  "ER > er_threshold (强趋势) AND close > ATR_upper (突破) AND vol_ratio > vol_min (放量)",
        "sell": "ER < er_exit (趋势减弱) OR close < KAMA (跌破自适应均线) OR close < ATR_lower",
    }
    chart_indicators = {
        "overlay": [
            {"col": "kama",      "label": "KAMA",      "color": "#ff9800"},
            {"col": "atr_upper", "label": "ATR Upper",  "color": "#4caf50", "dash": True},
            {"col": "atr_lower", "label": "ATR Lower",  "color": "#f44336", "dash": True},
        ],
        "subchart": [
            {"col": "efficiency_ratio", "label": "Efficiency Ratio", "color": "#2196f3"},
            {"col": "vol_ratio",        "label": "Vol Ratio",        "color": "#9c27b0"},
        ],
    }
    default_params = {
        "er_period":     18,    # efficiency ratio lookback (optimized)
        "er_threshold":  0.20,  # min ER for trend confirmation
        "er_exit":       0.10,  # ER below this = no trend
        "kama_fast":     5,     # KAMA fast constant
        "kama_slow":     38,    # KAMA slow constant
        "atr_period":    12,    # ATR period for channel
        "atr_mul":       1.62,  # ATR multiplier for channel
        "vol_period":    27,    # volume MA period
        "vol_min":       0.54,  # min volume ratio
        "rsi_period":    14,
        "rsi_sell":      73,
        "atr_stop_mul":  2.36,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        n = params["er_period"]

        # Kaufman Efficiency Ratio
        direction = (c - c.shift(n)).abs()
        volatility = c.diff().abs().rolling(n).sum()
        df["efficiency_ratio"] = direction / (volatility + 1e-10)

        # KAMA (Kaufman Adaptive Moving Average)
        er = df["efficiency_ratio"].values
        fast_sc = 2.0 / (params["kama_fast"] + 1)
        slow_sc = 2.0 / (params["kama_slow"] + 1)
        kama = np.full(len(c), np.nan)
        close_vals = c.values
        # Initialize KAMA at first valid ER
        start_idx = n
        if start_idx < len(close_vals):
            kama[start_idx] = close_vals[start_idx]
        for i in range(start_idx + 1, len(close_vals)):
            if np.isnan(er[i]):
                kama[i] = kama[i - 1] if not np.isnan(kama[i - 1]) else close_vals[i]
                continue
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            prev = kama[i - 1] if not np.isnan(kama[i - 1]) else close_vals[i]
            kama[i] = prev + sc * (close_vals[i] - prev)
        df["kama"] = kama

        # ATR channel around KAMA
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["atr"] = atr
        df["atr_upper"] = df["kama"] + params["atr_mul"] * atr
        df["atr_lower"] = df["kama"] - params["atr_mul"] * atr

        # Volume ratio
        vol_ma = v.rolling(params["vol_period"]).mean()
        df["vol_ratio"] = v / (vol_ma + 1e-9)

        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        er_strong    = df["efficiency_ratio"] > params["er_threshold"]
        er_weak      = df["efficiency_ratio"] < params["er_exit"]
        above_upper  = df["close"] > df["atr_upper"]
        below_kama   = df["close"] < df["kama"]
        below_lower  = df["close"] < df["atr_lower"]
        vol_ok       = df["vol_ratio"] >= params["vol_min"]

        df["buy_signal"] = (er_strong & above_upper & vol_ok).fillna(False)
        df["sell_signal"] = (
            er_weak | below_kama | (df["rsi"] > params["rsi_sell"])
        ).fillna(False)
        return df


