import pandas as pd
import numpy as np
import ta

from .base import BaseStrategy
from .core import (
    EhlersFisher, ElderForceIndex, SuperTrendStrategy, AdaptiveMA,
    ConnorsRsi, RossHook, AdaptiveRegime, VolatilityRegime,
    BbSqueezeBreakout, DivergenceCombo, StochasticRsi,
    VolumeWeightedMomentum, MomentumAcceleration, VolumeProfileSupport,
    MacdDivergence, AdaptiveBollinger, HeikinAshiTrend,
    ChandelierExit, ZScoreReversion,
)

# ─── Gemini 20 策略 (find-new-str 研发系列) ──────────────────────────

class GeminiVortexTrend(BaseStrategy):
    name = "gemini_vortex_trend"
    description = "Gemini 涡流趋势：Vortex Indicator (VI+/VI-) 交叉确认趋势方向"
    indicator_formulas = {
        "vi_plus": {"name": "VI+", "formula": "Sum(abs(High - Low[-1]), N) / Sum(TR, N)", "params": ["vortex_period"]},
        "vi_minus": {"name": "VI-", "formula": "Sum(abs(Low - High[-1]), N) / Sum(TR, N)", "params": ["vortex_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, 50)", "params": []},
    }
    signal_rules = {
        "buy": "VI+ 上穿 VI- AND VI+ > 1.0 AND Close > EMA_Trend",
        "sell": "VI+ 下穿 VI- OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA50", "color": "#9c27b0"}],
        "subchart": [
            {"col": "vi_plus", "label": "VI+", "color": "#4caf50"},
            {"col": "vi_minus", "label": "VI-", "color": "#f44336"},
        ],
    }
    default_params = {"vortex_period": 14, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        vortex = ta.trend.VortexIndicator(df["high"], df["low"], df["close"], window=params["vortex_period"])
        df["vi_plus"] = vortex.vortex_indicator_pos()
        df["vi_minus"] = vortex.vortex_indicator_neg()
        df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=50)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["vi_plus"] > df["vi_minus"]) & (df["vi_plus"].shift(1) <= df["vi_minus"].shift(1)) & (df["vi_plus"] > 1.0) & (df["close"] > df["ema_trend"])
        sell = (df["vi_plus"] < df["vi_minus"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiSchaffTrendCycle(BaseStrategy):
    name = "gemini_schaff_trend"
    description = "Gemini 舒夫趋势周期：STC 结合 MACD 与随机指标，比传统指标更早识别趋势"
    indicator_formulas = {
        "stc": {"name": "STC", "formula": "Schaff Trend Cycle", "params": ["fast", "slow", "cycle"]},
    }
    signal_rules = {
        "buy": "STC 上穿 25 AND STC 上行",
        "sell": "STC 下穿 75 OR STC 下行",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [{"col": "stc", "label": "STC", "color": "#2196f3"}],
    }
    default_params = {"fast": 23, "slow": 50, "cycle": 10, "atr_stop_mul": 2.2}

    def add_indicators(self, df, params):
        df = df.copy()
        # STC manual calculation (ta-lib style)
        macd = ta.trend.ema_indicator(df["close"], window=params["fast"]) - ta.trend.ema_indicator(df["close"], window=params["slow"])
        def get_stoch(series, window):
            low_val = series.rolling(window).min()
            high_val = series.rolling(window).max()
            return 100 * (series - low_val) / (high_val - low_val).replace(0, np.nan)
        
        stoch_macd = get_stoch(macd, params["cycle"])
        pf = stoch_macd.ewm(span=3, adjust=False).mean()
        stoch_pf = get_stoch(pf, params["cycle"])
        df["stc"] = stoch_pf.ewm(span=3, adjust=False).mean()
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["stc"] > 25) & (df["stc"].shift(1) <= 25)
        sell = (df["stc"] < 75) & (df["stc"].shift(1) >= 75)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiEhlersCyberCycle(BaseStrategy):
    name = "gemini_ehlers_cycle"
    description = "Gemini 埃勒斯赛博周期：利用数字信号处理识别价格循环周期"
    indicator_formulas = {
        "cycle": {"name": "Cyber Cycle", "formula": "DSP Cycle", "params": ["alpha", "window"]},
        "trigger": {"name": "Trigger", "formula": "Cycle[-1]", "params": []},
    }
    signal_rules = {
        "buy": "Cycle 上穿 Trigger",
        "sell": "Cycle 下穿 Trigger",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [
            {"col": "cycle", "label": "Cycle", "color": "#2196f3"},
            {"col": "trigger", "label": "Trigger", "color": "#ff9800"},
        ],
    }
    default_params = {"alpha": 0.07, "window": 14, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        price = (df["high"] + df["low"]) / 2
        alpha = params["alpha"]
        smooth = (price + 2 * price.shift(1) + 2 * price.shift(2) + price.shift(3)) / 6
        cycle = pd.Series(0.0, index=df.index)
        for i in range(7, len(df)):
            cycle.iloc[i] = (1 - 0.5 * alpha) ** 2 * (smooth.iloc[i] - 2 * smooth.iloc[i-1] + smooth.iloc[i-2]) + \
                             2 * (1 - alpha) * cycle.iloc[i-1] - (1 - alpha) ** 2 * cycle.iloc[i-2]
        df["cycle"] = cycle
        df["trigger"] = cycle.shift(1)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["cycle"] > df["trigger"]) & (df["cycle"].shift(1) <= df["trigger"].shift(1))
        sell = (df["cycle"] < df["trigger"]) & (df["cycle"].shift(1) >= df["trigger"].shift(1))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiMcGinleyDynamic(BaseStrategy):
    name = "gemini_mcginley_dynamic"
    description = "Gemini 麦金利动态指标：自适应价格变动速度的移动平均线，减少滞后"
    indicator_formulas = {
        "md": {"name": "McGinley", "formula": "MD + (Price - MD) / (N * (Price/MD)^4)", "params": ["window"]},
    }
    signal_rules = {
        "buy": "Close 上穿 McGinley AND McGinley 上行",
        "sell": "Close 下穿 McGinley",
    }
    chart_indicators = {
        "overlay": [{"col": "md", "label": "McGinley", "color": "#ff9800"}],
        "subchart": [],
    }
    default_params = {"window": 14, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        n = params["window"]
        md = pd.Series(index=df.index, dtype=float)
        md.iloc[0] = c.iloc[0]
        for i in range(1, len(df)):
            md.iloc[i] = md.iloc[i-1] + (c.iloc[i] - md.iloc[i-1]) / (n * (c.iloc[i] / md.iloc[i-1])**4)
        df["md"] = md
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["close"] > df["md"]) & (df["close"].shift(1) <= df["md"].shift(1)) & (df["md"] > df["md"].shift(1))
        sell = (df["close"] < df["md"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiHullMaCross(BaseStrategy):
    name = "gemini_hull_ma"
    description = "Gemini 赫尔均线交叉：极低滞后的 HMA 均线金叉策略"
    indicator_formulas = {
        "hma_fast": {"name": "HMA Fast", "formula": "WMA(2*WMA(n/2) - WMA(n), sqrt(n))", "params": ["fast_window"]},
        "hma_slow": {"name": "HMA Slow", "formula": "WMA(2*WMA(n/2) - WMA(n), sqrt(n))", "params": ["slow_window"]},
    }
    signal_rules = {
        "buy": "HMA Fast 上穿 HMA Slow",
        "sell": "HMA Fast 下穿 HMA Slow",
    }
    chart_indicators = {
        "overlay": [
            {"col": "hma_fast", "label": "HMA Fast", "color": "#ff9800"},
            {"col": "hma_slow", "label": "HMA Slow", "color": "#2196f3"},
        ],
        "subchart": [],
    }
    default_params = {"fast_window": 9, "slow_window": 21, "atr_stop_mul": 2.5}

    def hma(self, series, window):
        def wma(s, w):
            weights = np.arange(1, w + 1)
            return s.rolling(w).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        half_w = window // 2
        sqrt_w = int(np.sqrt(window))
        raw_hma = 2 * wma(series, half_w) - wma(series, window)
        return wma(raw_hma, sqrt_w)

    def add_indicators(self, df, params):
        df = df.copy()
        df["hma_fast"] = self.hma(df["close"], params["fast_window"])
        df["hma_slow"] = self.hma(df["close"], params["slow_window"])
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["hma_fast"] > df["hma_slow"]) & (df["hma_fast"].shift(1) <= df["hma_slow"].shift(1))
        sell = (df["hma_fast"] < df["hma_slow"]) & (df["hma_fast"].shift(1) >= df["hma_slow"].shift(1))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiTrendPersistence(BaseStrategy):
    name = "gemini_trend_persistence"
    description = "Gemini 趋势持久度：利用效率比 (ER) 过滤低质量震荡，仅在强趋势中持仓"
    indicator_formulas = {
        "er": {"name": "Efficiency Ratio", "formula": "Change / Volatility", "params": ["window"]},
        "ema": {"name": "EMA Trend", "formula": "EMA(close, 20)", "params": []},
    }
    signal_rules = {
        "buy": "ER > threshold AND Close > EMA",
        "sell": "ER < threshold OR Close < EMA",
    }
    chart_indicators = {
        "overlay": [{"col": "ema", "label": "EMA", "color": "#9c27b0"}],
        "subchart": [{"col": "er", "label": "ER", "color": "#4caf50"}],
    }
    default_params = {"window": 20, "threshold": 0.3, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        change = (df["close"] - df["close"].shift(params["window"])).abs()
        volatility = (df["close"] - df["close"].shift(1)).abs().rolling(params["window"]).sum()
        df["er"] = change / volatility.replace(0, np.nan)
        df["ema"] = ta.trend.ema_indicator(df["close"], window=20)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["er"] > params["threshold"]) & (df["close"] > df["ema"])
        sell = (df["er"] < params["threshold"] * 0.7) | (df["close"] < df["ema"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiAroonOscillator(BaseStrategy):
    name = "gemini_aroon_osc"
    description = "Gemini 阿隆振荡器：衡量趋势强度与潜在转折点"
    indicator_formulas = {
        "aroon_osc": {"name": "Aroon Osc", "formula": "Aroon Up - Aroon Down", "params": ["window"]},
    }
    signal_rules = {
        "buy": "Aroon Osc 上穿 0 AND Aroon Up > 70",
        "sell": "Aroon Osc 下穿 0 OR Aroon Down > 70",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [{"col": "aroon_osc", "label": "Aroon Osc", "color": "#2196f3"}],
    }
    default_params = {"window": 25, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        aroon = ta.trend.AroonIndicator(df["close"], window=params["window"])
        df["aroon_up"] = aroon.aroon_up()
        df["aroon_down"] = aroon.aroon_down()
        df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["aroon_osc"] > 0) & (df["aroon_osc"].shift(1) <= 0) & (df["aroon_up"] > 70)
        sell = (df["aroon_osc"] < 0) & (df["aroon_osc"].shift(1) >= 0)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiCoppockMomentum(BaseStrategy):
    name = "gemini_coppock"
    description = "Gemini 估波曲线：识别长期底部与动量转折，经典多头策略"
    indicator_formulas = {
        "coppock": {"name": "Coppock", "formula": "WMA(ROC(14) + ROC(11), 10)", "params": []},
    }
    signal_rules = {
        "buy": "Coppock 穿越 0 向上",
        "sell": "Coppock 见顶回落 OR Coppock < 0",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [{"col": "coppock", "label": "Coppock", "color": "#9c27b0"}],
    }
    default_params = {"atr_stop_mul": 3.0}

    def add_indicators(self, df, params):
        df = df.copy()
        roc14 = ta.momentum.roc(df["close"], window=14)
        roc11 = ta.momentum.roc(df["close"], window=11)
        combined = roc14 + roc11
        weights = np.arange(1, 11)
        df["coppock"] = combined.rolling(10).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["coppock"] > 0) & (df["coppock"].shift(1) <= 0)
        sell = (df["coppock"] < df["coppock"].shift(1)) & (df["coppock"].shift(1) > df["coppock"].shift(2)) & (df["coppock"] > 0)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiKeltnerReversion(BaseStrategy):
    name = "gemini_keltner_meanrev"
    description = "Gemini 肯特纳均值回归：当价格极度偏离 EMA 时博取反弹"
    indicator_formulas = {
        "k_mid": {"name": "EMA Mid", "formula": "EMA(close, 20)", "params": []},
        "k_upper": {"name": "K Upper", "formula": "EMA + 2*ATR", "params": []},
        "k_lower": {"name": "K Lower", "formula": "EMA - 2*ATR", "params": []},
    }
    signal_rules = {
        "buy": "Close < K Lower AND RSI < 30",
        "sell": "Close > K Mid OR RSI > 70",
    }
    chart_indicators = {
        "overlay": [
            {"col": "k_upper", "label": "K Upper", "color": "#555555", "dash": True},
            {"col": "k_mid", "label": "K Mid", "color": "#ff9800"},
            {"col": "k_lower", "label": "K Lower", "color": "#555555", "dash": True},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#e91e63"}],
    }
    default_params = {"ema_window": 20, "atr_multiplier": 2.0, "atr_stop_mul": 1.5}

    def add_indicators(self, df, params):
        df = df.copy()
        df["k_mid"] = ta.trend.ema_indicator(df["close"], window=params["ema_window"])
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        df["k_upper"] = df["k_mid"] + params["atr_multiplier"] * df["atr"]
        df["k_lower"] = df["k_mid"] - params["atr_multiplier"] * df["atr"]
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["close"] < df["k_lower"]) & (df["rsi"] < 30)
        sell = (df["close"] > df["k_mid"]) | (df["rsi"] > 70)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiZScoreSpread(BaseStrategy):
    name = "gemini_zscore_spread"
    description = "Gemini Z-Score 价差：价格相对于其均线的标准差水平，捕捉极端偏离"
    indicator_formulas = {
        "zscore": {"name": "Z-Score", "formula": "(Close - Mean) / StdDev", "params": ["window"]},
    }
    signal_rules = {
        "buy": "Z-Score < -2.0 AND Z-Score 上行",
        "sell": "Z-Score > 0 OR Z-Score > 2.0",
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [{"col": "zscore", "label": "Z-Score", "color": "#2196f3"}],
    }
    default_params = {"window": 30, "atr_stop_mul": 2.0}

    def add_indicators(self, df, params):
        df = df.copy()
        mean = df["close"].rolling(params["window"]).mean()
        std = df["close"].rolling(params["window"]).std()
        df["zscore"] = (df["close"] - mean) / std
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["zscore"] < -2.0) & (df["zscore"] > df["zscore"].shift(1))
        sell = (df["zscore"] > 0) | (df["zscore"] > 2.0)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiBullishEngulfingPlus(BaseStrategy):
    name = "gemini_bullish_engulfing"
    description = "Gemini 看涨吞噬增强：结合成交量突破与 EMA 趋势确认"
    indicator_formulas = {
        "is_engulfing": {"name": "Engulfing", "formula": "Close > Open[-1] AND Open < Close[-1]", "params": []},
        "vol_ratio": {"name": "Vol Ratio", "formula": "Vol / MeanVol", "params": []},
    }
    signal_rules = {
        "buy": "看涨吞噬形态 AND Vol Ratio > 1.2 AND Close > EMA50",
        "sell": "Close < EMA50 OR RSI > 75",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA50", "color": "#ff9800"}],
        "subchart": [{"col": "vol_ratio", "label": "Vol Ratio", "color": "#4caf50"}],
    }
    default_params = {"vol_window": 20, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        c, o, v = df["close"], df["open"], df["volume"]
        prev_c, prev_o = c.shift(1), o.shift(1)
        df["bullish_engulfing"] = (c > prev_o) & (o < prev_c) & (prev_c < prev_o)
        df["vol_ratio"] = v / v.rolling(params["vol_window"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=50)
        df["rsi"] = ta.momentum.rsi(c, window=14)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = df["bullish_engulfing"] & (df["vol_ratio"] > 1.2) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["ema_trend"]) | (df["rsi"] > 75)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiMorningStarPlus(BaseStrategy):
    name = "gemini_morning_star"
    description = "Gemini 晨星增强：三根 K 线筑底形态 + RSI 低位金叉"
    indicator_formulas = {
        "morning_star": {"name": "Morning Star", "formula": "3-bar pattern", "params": []},
    }
    signal_rules = {
        "buy": "识别晨星形态 AND RSI < 40",
        "sell": "Close < EMA20 OR RSI > 70",
    }
    chart_indicators = {
        "overlay": [{"col": "ema20", "label": "EMA20", "color": "#2196f3"}],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#e91e63"}],
    }
    default_params = {"atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        c, o = df["close"], df["open"]
        # Simplified Morning Star: Bar1 down, Bar2 small body, Bar3 up
        body = (c - o).abs()
        bar1_down = (c.shift(2) < o.shift(2)) & ((o.shift(2) - c.shift(2)) > body.rolling(10).mean().shift(2))
        bar2_small = body.shift(1) < body.rolling(10).mean().shift(1) * 0.5
        bar3_up = (c > o) & (c > (o.shift(2) + c.shift(2))/2)
        df["morning_star"] = bar1_down & bar2_small & bar3_up
        df["rsi"] = ta.momentum.rsi(c, window=14)
        df["ema20"] = ta.trend.ema_indicator(c, window=20)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = df["morning_star"] & (df["rsi"] < 45)
        sell = (df["close"] < df["ema20"]) | (df["rsi"] > 70)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiHammerBounce(BaseStrategy):
    name = "gemini_hammer_bounce"
    description = "Gemini 锤子线反弹：在近期低点出现的锤子线，捕捉探底回升"
    indicator_formulas = {
        "is_hammer": {"name": "Hammer", "formula": "Small Body & Long Lower Wick", "params": []},
    }
    signal_rules = {
        "buy": "锤子线形态 AND 处于近期低位(Close < BB Lower)",
        "sell": "Close > BB Mid OR RSI > 65",
    }
    chart_indicators = {
        "overlay": [{"col": "bb_mid", "label": "BB Mid", "color": "#78909c"}],
        "subchart": [],
    }
    default_params = {"atr_stop_mul": 2.0}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, o = df["high"], df["low"], df["close"], df["open"]
        body = (c - o).abs()
        lower_wick = (df[["open", "close"]].min(axis=1) - l)
        upper_wick = (h - df[["open", "close"]].max(axis=1))
        df["is_hammer"] = (lower_wick > 2 * body) & (upper_wick < body) & (body > 0)
        bb = ta.volatility.BollingerBands(c, window=20)
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["rsi"] = ta.momentum.rsi(c, window=14)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = df["is_hammer"] & (df["close"] < df["bb_low"] * 1.02)
        sell = (df["close"] > df["bb_mid"]) | (df["rsi"] > 65)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiShootingStarReversal(BaseStrategy):
    name = "gemini_shooting_star"
    description = "Gemini 射击之星反转：用于高位逃顶或反向信号确认"
    indicator_formulas = {
        "is_shooting_star": {"name": "Shooting Star", "formula": "Small Body & Long Upper Wick", "params": []},
    }
    signal_rules = {
        "buy": "MACD 金叉 AND Close > EMA20 (基础趋势)",
        "sell": "射击之星形态 AND 处于近期高位(Close > BB Upper)",
    }
    chart_indicators = {
        "overlay": [{"col": "bb_upper", "label": "BB Upper", "color": "#ef5350"}],
        "subchart": [],
    }
    default_params = {"atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, o = df["high"], df["low"], df["close"], df["open"]
        body = (c - o).abs()
        upper_wick = (h - df[["open", "close"]].max(axis=1))
        lower_wick = (df[["open", "close"]].min(axis=1) - l)
        df["is_shooting_star"] = (upper_wick > 2 * body) & (lower_wick < body) & (body > 0)
        bb = ta.volatility.BollingerBands(c, window=20)
        df["bb_upper"] = bb.bollinger_hband()
        macd = ta.trend.MACD(c)
        df["macd_hist"] = macd.macd_diff()
        df["ema20"] = ta.trend.ema_indicator(c, window=20)
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0) & (df["close"] > df["ema20"])
        sell = df["is_shooting_star"] & (df["close"] > df["bb_upper"] * 0.98)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiMondayReversal(BaseStrategy):
    name = "gemini_monday_reversal"
    description = "Gemini 周一效应：捕捉'周一跌、周二起'的日历效应，配合成交量"
    indicator_formulas = {
        "weekday": {"name": "Weekday", "formula": "0=Monday, 1=Tuesday...", "params": []},
    }
    signal_rules = {
        "buy": "Today is Tuesday AND Yesterday (Monday) was Down AND Vol > Vol_MA",
        "sell": "Hold 3 days OR Close < EMA20",
    }
    chart_indicators = {
        "overlay": [{"col": "ema20", "label": "EMA20", "color": "#2196f3"}],
        "subchart": [],
    }
    default_params = {"vol_window": 10, "atr_stop_mul": 2.0}

    def add_indicators(self, df, params):
        df = df.copy()
        df["weekday"] = df.index.weekday
        df["prev_close"] = df["close"].shift(1)
        df["prev_open"] = df["open"].shift(1)
        df["vol_ma"] = df["volume"].rolling(params["vol_window"]).mean()
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # Buy on Tuesday morning if Monday was a down day
        monday_down = (df["weekday"].shift(1) == 0) & (df["close"].shift(1) < df["open"].shift(1))
        buy = (df["weekday"] == 1) & monday_down & (df["volume"] > df["vol_ma"])
        sell = (df["weekday"] == 4) | (df["close"] < df["ema20"]) # Exit on Friday or trend break
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiMonthlyWindow(BaseStrategy):
    name = "gemini_monthly_window"
    description = "Gemini 月末年初效应：捕捉每个月最后 2 天和前 3 天的超额收益"
    indicator_formulas = {
        "day": {"name": "Day of Month", "formula": "1-31", "params": []},
    }
    signal_rules = {
        "buy": "Month Day in [28, 29, 30, 31, 1, 2, 3] AND Close > EMA50",
        "sell": "Month Day == 5 OR Close < EMA50",
    }
    chart_indicators = {
        "overlay": [{"col": "ema50", "label": "EMA50", "color": "#ff9800"}],
        "subchart": [],
    }
    default_params = {"atr_stop_mul": 3.0}

    def add_indicators(self, df, params):
        df = df.copy()
        df["day"] = df.index.day
        df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        is_window = df["day"].isin([28, 29, 30, 31, 1, 2, 3])
        buy = is_window & (df["close"] > df["ema50"]) & (~is_window.shift(1).fillna(False))
        sell = (df["day"] == 5) | (df["close"] < df["ema50"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiGarchTrendFilter(BaseStrategy):
    name = "gemini_garch_filter"
    description = "Gemini GARCH 趋势过滤：利用 GARCH(1,1) 预测波动率，规避高风险波动期"
    indicator_formulas = {
        "garch_vol": {"name": "GARCH Vol", "formula": "Estimated Volatility", "params": []},
    }
    signal_rules = {
        "buy": "EMA Fast > EMA Slow AND GARCH Vol < Threshold",
        "sell": "EMA Fast < EMA Slow OR GARCH Vol > Threshold * 1.5",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_slow", "label": "EMA Slow", "color": "#2196f3"}],
        "subchart": [{"col": "garch_vol", "label": "GARCH Vol", "color": "#f44336"}],
    }
    default_params = {"fast": 12, "slow": 26, "vol_limit": 0.02, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        returns = df["close"].pct_change().dropna()
        # Simple GARCH(1,1) approximation: sigma^2 = omega + alpha*r^2 + beta*sigma_prev^2
        omega, alpha, beta = 0.00001, 0.1, 0.85
        vol = pd.Series(index=df.index, dtype=float)
        vol.iloc[0] = returns.std() if len(returns)>0 else 0.01
        for i in range(1, len(df)):
            ret_sq = (df["close"].iloc[i] / df["close"].iloc[i-1] - 1)**2
            vol.iloc[i] = np.sqrt(omega + alpha * ret_sq + beta * vol.iloc[i-1]**2)
        df["garch_vol"] = vol
        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=params["fast"])
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=params["slow"])
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["ema_fast"] > df["ema_slow"]) & (df["garch_vol"] < params["vol_limit"])
        sell = (df["ema_fast"] < df["ema_slow"]) | (df["garch_vol"] > params["vol_limit"] * 1.5)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiVwapAnchoredTrend(BaseStrategy):
    name = "gemini_anchored_vwap"
    description = "Gemini 锚定 VWAP：从近期波段低点锚定 VWAP，作为动态支撑位"
    indicator_formulas = {
        "avwap": {"name": "Anchored VWAP", "formula": "Sum(P*V)/Sum(V) from Anchor", "params": ["lookback"]},
    }
    signal_rules = {
        "buy": "Close 上穿 Anchored VWAP AND Vol Ratio > 1.2",
        "sell": "Close 下穿 Anchored VWAP",
    }
    chart_indicators = {
        "overlay": [{"col": "avwap", "label": "AVWAP", "color": "#ff9800"}],
        "subchart": [],
    }
    default_params = {"lookback": 60, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        # Find local low as anchor
        low_idx = df["low"].rolling(params["lookback"]).apply(np.argmin, raw=True)
        avwap = pd.Series(index=df.index, dtype=float)
        for i in range(params["lookback"], len(df)):
            start_pos = int(i - params["lookback"] + low_idx.iloc[i])
            window_df = df.iloc[start_pos : i+1]
            avwap.iloc[i] = (window_df["close"] * window_df["volume"]).sum() / window_df["volume"].sum()
        df["avwap"] = avwap
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["close"] > df["avwap"]) & (df["close"].shift(1) <= df["avwap"].shift(1)) & (df["vol_ratio"] > 1.2)
        sell = (df["close"] < df["avwap"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiVolumeProfileSva(BaseStrategy):
    name = "gemini_volume_sva"
    description = "Gemini 成交量密集区突破：识别显著成交量价位 (SVA)，跟随机构突破"
    indicator_formulas = {
        "poc": {"name": "POC", "formula": "Price of Control (Highest Vol Price)", "params": ["window"]},
    }
    signal_rules = {
        "buy": "Close 突破近 30 日成交密集区上沿 AND Vol > Vol_MA",
        "sell": "Close 跌回密集区 OR RSI > 75",
    }
    chart_indicators = {
        "overlay": [{"col": "poc", "label": "POC", "color": "#ffa726", "dash": True}],
        "subchart": [],
    }
    default_params = {"window": 30, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        w = params["window"]
        poc = pd.Series(index=df.index, dtype=float)
        for i in range(w, len(df)):
            window_df = df.iloc[i-w:i]
            # Simple volume profile: bin prices
            bins = pd.cut(window_df["close"], bins=10)
            poc_bin = window_df.groupby(bins, observed=True)["volume"].sum().idxmax()
            poc.iloc[i] = poc_bin.mid
        df["poc"] = poc
        df["vol_ma"] = df["volume"].rolling(20).mean()
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["close"] > df["poc"] * 1.02) & (df["close"].shift(1) <= df["poc"].shift(1) * 1.02) & (df["volume"] > df["vol_ma"])
        sell = (df["close"] < df["poc"]) | (df["rsi"] > 75)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

class GeminiPeadMomentum(BaseStrategy):
    name = "gemini_pead_momentum"
    description = "Gemini 财报漂移模拟：利用巨量跳空确认盈余冲击后的持续动量"
    indicator_formulas = {
        "gap": {"name": "Gap %", "formula": "(Open - Close[-1]) / Close[-1]", "params": []},
    }
    signal_rules = {
        "buy": "Gap > 3% AND Vol > 3 * Vol_MA (跳空高开高走)",
        "sell": "Hold 10 days OR Close < EMA20",
    }
    chart_indicators = {
        "overlay": [{"col": "ema20", "label": "EMA20", "color": "#2196f3"}],
        "subchart": [],
    }
    default_params = {"vol_window": 20, "atr_stop_mul": 3.0}

    def add_indicators(self, df, params):
        df = df.copy()
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["vol_ma"] = df["volume"].rolling(params["vol_window"]).mean()
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        # Buy if price gaps up significantly on huge volume and stays above open
        buy = (df["gap"] > 0.03) & (df["volume"] > df["vol_ma"] * 3) & (df["close"] > df["open"])
        # Simple exit: trend break or 10 bars
        sell = (df["close"] < df["ema20"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

# ─── Gemini 21-40: 候选别名类（保留为新策略，逻辑复用近似母策略） ───────────────

class GeminiFractalTrend(GeminiTrendPersistence):
    name = "gemini_fractal_trend"
    description = "Gemini 分形趋势：利用分形突破与趋势延续的近似实现"


class GeminiFisherTransform(EhlersFisher):
    name = "gemini_fisher_transform"
    description = "Gemini Fisher变换：利用 Fisher 变换捕捉趋势拐点"


class GeminiKlingerVolume(ElderForceIndex):
    name = "gemini_klinger_volume"
    description = "Gemini Klinger成交量摆动：以量价动能近似 Klinger 摆动"


class GeminiSuperTrendV2(SuperTrendStrategy):
    name = "gemini_supertrend_v2"
    description = "Gemini SuperTrend V2：双通道趋势追踪的近似实现"


class GeminiChandeMomentum(ConnorsRsi):
    name = "gemini_chande_momentum"
    description = "Gemini 钱德动量：以 CMO 风格的短线动量近似实现"


class GeminiAdaptiveAlpha(AdaptiveMA):
    name = "gemini_adaptive_alpha"
    description = "Gemini 自适应Alpha：以自适应均线趋势近似实现"


class GeminiPriceActionLiquidity(RossHook):
    name = "gemini_price_action_liquidity"
    description = "Gemini 价格行为流动性：以价格行为突破近似订单块回收"


class GeminiRegimeSwitching(AdaptiveRegime):
    name = "gemini_regime_switching"
    description = "Gemini 状态切换：趋势/震荡切换的近似实现"


class GeminiFractalDimension(VolatilityRegime):
    name = "gemini_fractal_dimension"
    description = "Gemini 分形维数：用波动状态近似分形维数过滤"


class GeminiVolatilityBurst(BbSqueezeBreakout):
    name = "gemini_volatility_burst"
    description = "Gemini 波动率爆发：以挤压后突破近似实现"


class GeminiMomentumExhaustion(DivergenceCombo):
    name = "gemini_momentum_exhaustion"
    description = "Gemini 动量衰竭：以背离反转逻辑近似实现"


class GeminiStochasticRsiBurst(StochasticRsi):
    name = "gemini_stochastic_rsi_burst"
    description = "Gemini StochRSI 爆发：利用 StochRSI 脉冲近似实现"


class GeminiVolumePriceDiv(VolumeWeightedMomentum):
    name = "gemini_volume_price_div"
    description = "Gemini 价量背离：以量价动量差异近似实现"


class GeminiTripleScreenPulse(MomentumAcceleration):
    name = "gemini_triple_screen_pulse"
    description = "Gemini 三重滤网脉冲：以趋势+动量加速度近似实现"


class GeminiSupportResistanceSva(VolumeProfileSupport):
    name = "gemini_support_resistance_sva"
    description = "Gemini 阻力位SVA：以成交量支撑阻力近似实现"


class GeminiMacdHistDiv(MacdDivergence):
    name = "gemini_macd_hist_div"
    description = "Gemini MACD柱背离：以 MACD 背离近似实现"


class GeminiBollingerImpulse(AdaptiveBollinger):
    name = "gemini_bollinger_impulse"
    description = "Gemini 布林带脉冲：以布林带脉冲近似实现"


class GeminiHeikinAshiBalance(HeikinAshiTrend):
    name = "gemini_heikin_ashi_balance"
    description = "Gemini Heikin Ashi平衡：以平均K线趋势近似实现"


class GeminiAtrVolatilityStop(ChandelierExit):
    name = "gemini_atr_volatility_stop"
    description = "Gemini ATR波动止损：以 ATR 追踪止损近似实现"


class GeminiStdDevChannel(ZScoreReversion):
    name = "gemini_std_dev_channel"
    description = "Gemini 标准差通道：以标准差偏离回归近似实现"


# ─── OpenAI 第二批候选策略辅助函数 ───────────────────────────────────────

def _atr14(high, low, close, window: int = 14):
    return ta.volatility.average_true_range(high, low, close, window=window)


def _true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def _cross_up(a, b):
    prev_b = b.shift(1) if hasattr(b, "shift") else b
    return (a > b) & (a.shift(1) <= prev_b)


def _cross_down(a, b):
    prev_b = b.shift(1) if hasattr(b, "shift") else b
    return (a < b) & (a.shift(1) >= prev_b)


def _vwma(close, volume, window: int):
    vol_sum = volume.rolling(window).sum().replace(0, np.nan)
    return (close * volume).rolling(window).sum() / vol_sum


def _dema(close, window: int):
    ema1 = ta.trend.ema_indicator(close, window=window)
    ema2 = ta.trend.ema_indicator(ema1, window=window)
    return 2 * ema1 - ema2


def _tema(close, window: int):
    ema1 = ta.trend.ema_indicator(close, window=window)
    ema2 = ta.trend.ema_indicator(ema1, window=window)
    ema3 = ta.trend.ema_indicator(ema2, window=window)
    return 3 * ema1 - 3 * ema2 + ema3


def _rolling_zscore(series, window: int):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _vol_ratio(volume, window: int):
    return volume / volume.rolling(window).mean().replace(0, np.nan)


def _intraday_intensity(high, low, close, volume, window: int):
    raw = ((2 * close - high - low) / (high - low).replace(0, np.nan)) * volume
    vol_sum = volume.rolling(window).sum().replace(0, np.nan)
    return raw.rolling(window).sum() / vol_sum


def _chaikin_volatility(high, low, ema_window: int, change_window: int):
    hl_ema = (high - low).abs().ewm(span=ema_window, adjust=False).mean()
    return hl_ema.pct_change(change_window) * 100


def _body_ratio(open_, high, low, close):
    return (close - open_).abs() / (high - low).replace(0, np.nan)


# ─── OpenAI 第二批候选策略（1-30） ─────────────────────────────────────

class OpenAiTsiTrendline(BaseStrategy):
    name = "openai_tsi_trendline"
    description = "TSI 趋势线：TSI 上穿信号线且价格位于趋势 EMA 上方时买入"
    default_params = {"slow": 25, "fast": 13, "signal_period": 7, "ema_period": 55, "buy_floor": 5.0, "exit_floor": -5.0, "atr_stop_mul": 2.6}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["tsi"] = ta.momentum.tsi(c, window_slow=params["slow"], window_fast=params["fast"])
        df["tsi_signal"] = df["tsi"].ewm(span=params["signal_period"], adjust=False).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["tsi"], df["tsi_signal"]) & (df["tsi"] > params["buy_floor"]) & (df["close"] > df["ema_trend"])
        sell = _cross_down(df["tsi"], df["tsi_signal"]) | (df["tsi"] < params["exit_floor"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiDemaReclaim(BaseStrategy):
    name = "openai_dema_reclaim"
    description = "DEMA 回收：价格重新站上 DEMA 并突破短期高点时买入"
    default_params = {"dema_period": 21, "ema_period": 55, "breakout_period": 12, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["dema_fast"] = _dema(c, params["dema_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["close"], df["dema_fast"]) & (df["dema_fast"] > df["ema_trend"]) & (df["close"] > df["breakout_high"])
        sell = (df["close"] < df["dema_fast"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiTemaImpulse(BaseStrategy):
    name = "openai_tema_impulse"
    description = "TEMA 脉冲：TEMA 上升且价格突破近期高点时买入"
    default_params = {"tema_period": 18, "breakout_period": 14, "slope_lookback": 4, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["tema_fast"] = _tema(c, params["tema_period"])
        df["tema_slope"] = df["tema_fast"] - df["tema_fast"].shift(params["slope_lookback"])
        df["breakout_high"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["tema_slope"] > 0) & (df["close"] > df["tema_fast"]) & (df["close"] > df["breakout_high"])
        sell = (df["tema_slope"] < 0) | (df["close"] < df["tema_fast"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiVwmaBreakout(BaseStrategy):
    name = "openai_vwma_breakout"
    description = "VWMA 突破：量价加权均线配合放量突破时买入"
    default_params = {"vwma_period": 24, "vol_period": 20, "breakout_period": 18, "vol_ratio_min": 1.1, "atr_stop_mul": 2.7}

    def add_indicators(self, df, params):
        df = df.copy()
        c, v = df["close"], df["volume"]
        df["vwma"] = _vwma(c, v, params["vwma_period"])
        df["vol_ratio"] = _vol_ratio(v, params["vol_period"])
        df["breakout_high"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["close"] > df["vwma"]) & (df["close"] > df["breakout_high"]) & (df["vol_ratio"] > params["vol_ratio_min"])
        sell = (df["close"] < df["vwma"]) | (df["vol_ratio"] < 0.8)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiEnvelopePullback(BaseStrategy):
    name = "openai_envelope_pullback"
    description = "包络回调：上升趋势中价格重新站上下包络时买入"
    default_params = {"ema_period": 34, "band_pct": 0.03, "rsi_period": 14, "rsi_floor": 48, "rsi_exit": 40, "atr_stop_mul": 2.4}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["ema_mid"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["env_upper"] = df["ema_mid"] * (1 + params["band_pct"])
        df["env_lower"] = df["ema_mid"] * (1 - params["band_pct"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["close"], df["env_lower"]) & (df["ema_mid"] > df["ema_mid"].shift(2)) & (df["rsi"] > params["rsi_floor"])
        sell = (df["close"] < df["env_lower"]) | (df["rsi"] < params["rsi_exit"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiAtrRatioTrend(BaseStrategy):
    name = "openai_atr_ratio_trend"
    description = "ATR 比率趋势：波动扩张与均线趋势共振时买入"
    default_params = {"ema_fast": 21, "ema_slow": 55, "atr_period": 14, "atr_ma_period": 24, "breakout_period": 16, "atr_ratio_min": 1.05, "atr_ratio_exit": 0.88, "atr_stop_mul": 2.8}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_fast"] = ta.trend.ema_indicator(c, window=params["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(c, window=params["ema_slow"])
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["atr_ratio"] = atr / atr.rolling(params["atr_ma_period"]).mean().replace(0, np.nan)
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = atr
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["ema_fast"] > df["ema_slow"]) & (df["atr_ratio"] > params["atr_ratio_min"]) & (df["close"] > df["breakout_high"])
        sell = (df["ema_fast"] < df["ema_slow"]) | (df["atr_ratio"] < params["atr_ratio_exit"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiUltimateBreakout(BaseStrategy):
    name = "openai_ultimate_breakout"
    description = "UO 突破：终极振荡器转强并突破高点时买入"
    default_params = {"window1": 7, "window2": 14, "window3": 28, "buy_level": 55, "exit_level": 45, "ema_period": 50, "breakout_period": 14, "atr_stop_mul": 2.6}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["uo"] = ta.momentum.ultimate_oscillator(h, l, c, window1=params["window1"], window2=params["window2"], window3=params["window3"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(h, l, c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["uo"], params["buy_level"]) & (df["close"] > df["ema_trend"]) & (df["close"] > df["breakout_high"])
        sell = (df["uo"] < params["exit_level"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiRsiFailureSwing(BaseStrategy):
    name = "openai_rsi_failure_swing"
    description = "RSI 失败摆动：RSI 从弱势区反抽并回到趋势线上方时买入"
    default_params = {"rsi_period": 14, "oversold": 32, "rebound": 46, "exit_level": 42, "ema_period": 34, "atr_stop_mul": 2.3}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["rsi"].shift(1) < params["oversold"]) & (df["rsi"] > params["rebound"]) & (df["close"] > df["ema_trend"])
        sell = (df["rsi"] < params["exit_level"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiMfiPressurePop(BaseStrategy):
    name = "openai_mfi_pressure_pop"
    description = "MFI 资金弹升：资金流指数转强并放量突破时买入"
    default_params = {"mfi_period": 14, "buy_level": 55, "exit_level": 45, "breakout_period": 16, "vol_period": 20, "vol_ratio_min": 1.0, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        df["mfi"] = ta.volume.money_flow_index(h, l, c, v, window=params["mfi_period"])
        df["vol_ratio"] = _vol_ratio(v, params["vol_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["mfi"], params["buy_level"]) & (df["vol_ratio"] > params["vol_ratio_min"]) & (df["close"] > df["breakout_high"])
        sell = (df["mfi"] < params["exit_level"]) | (df["vol_ratio"] < 0.8)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiRocTrendShift(BaseStrategy):
    name = "openai_roc_trend_shift"
    description = "ROC 趋势换挡：ROC 翻正并突破短期高点时买入"
    default_params = {"roc_period": 12, "buy_level": 1.0, "exit_level": -1.0, "ema_period": 50, "breakout_period": 14, "atr_stop_mul": 2.6}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["roc"] = ta.momentum.roc(c, window=params["roc_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["roc"], params["buy_level"]) & (df["close"] > df["ema_trend"]) & (df["close"] > df["breakout_high"])
        sell = (df["roc"] < params["exit_level"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiDisparityRebound(BaseStrategy):
    name = "openai_disparity_rebound"
    description = "乖离回弹：价格对均线负乖离收敛后重回基线时买入"
    default_params = {"base_period": 34, "entry_neg": -0.05, "recover_level": -0.01, "take_profit": 0.05, "atr_stop_mul": 2.1}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["base_ma"] = ta.trend.sma_indicator(c, window=params["base_period"])
        df["disparity"] = c / df["base_ma"].replace(0, np.nan) - 1.0
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["disparity"].shift(1) < params["entry_neg"]) & (df["disparity"] > params["recover_level"]) & (df["close"] > df["base_ma"])
        sell = (df["disparity"] > params["take_profit"]) | (df["close"] < df["base_ma"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiPercentBTrend(BaseStrategy):
    name = "openai_percent_b_trend"
    description = "%B 趋势：布林带 %B 回到中轴上方并伴随带宽扩张时买入"
    default_params = {"bb_period": 20, "bb_std": 2.0, "buy_level": 0.55, "exit_level": 0.35, "bandwidth_min": 0.08, "atr_stop_mul": 2.3}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"], window_dev=params["bb_std"])
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        band = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["percent_b"] = (c - df["bb_lower"]) / band
        df["bandwidth"] = band / df["bb_mid"].replace(0, np.nan)
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["percent_b"], params["buy_level"]) & (df["close"] > df["bb_mid"]) & (df["bandwidth"] > params["bandwidth_min"])
        sell = (df["percent_b"] < params["exit_level"]) | (df["close"] < df["bb_mid"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiBandwidthExpansion(BaseStrategy):
    name = "openai_bandwidth_expansion"
    description = "带宽扩张：布林带带宽从压缩转扩张并突破高点时买入"
    default_params = {"bb_period": 20, "bb_std": 2.0, "bandwidth_min": 0.1, "bandwidth_exit": 0.06, "ema_period": 34, "breakout_period": 16, "atr_stop_mul": 2.7}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        bb = ta.volatility.BollingerBands(c, window=params["bb_period"], window_dev=params["bb_std"])
        df["bb_mid"] = bb.bollinger_mavg()
        band = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan)
        df["bandwidth"] = band / df["bb_mid"].replace(0, np.nan)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["bandwidth"] > params["bandwidth_min"]) & (df["bandwidth"] > df["bandwidth"].shift(1)) & (df["close"] > df["breakout_high"]) & (df["close"] > df["ema_trend"])
        sell = (df["bandwidth"] < params["bandwidth_exit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiKeltnerBandPulse(BaseStrategy):
    name = "openai_keltner_band_pulse"
    description = "Keltner 带脉冲：价格上破通道上轨且 RSI 转强时买入"
    default_params = {"ema_period": 20, "atr_period": 14, "kc_mult": 1.6, "rsi_period": 14, "rsi_floor": 52, "rsi_exit": 44, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["kc_mid"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        atr = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["kc_upper"] = df["kc_mid"] + params["kc_mult"] * atr
        df["kc_lower"] = df["kc_mid"] - params["kc_mult"] * atr
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = atr
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["close"], df["kc_upper"]) & (df["rsi"] > params["rsi_floor"])
        sell = (df["close"] < df["kc_mid"]) | (df["rsi"] < params["rsi_exit"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiZscoreTrendSnap(BaseStrategy):
    name = "openai_zscore_trend_snap"
    description = "Z 分数趋势回弹：价格在趋势中从负偏离快速修复时买入"
    default_params = {"z_window": 30, "entry_low": -1.2, "recover_level": -0.3, "take_profit": 1.0, "ema_period": 50, "atr_stop_mul": 2.2}

    def add_indicators(self, df, params):
        df = df.copy()
        c = df["close"]
        df["zscore"] = _rolling_zscore(c, params["z_window"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["zscore"].shift(1) < params["entry_low"]) & (df["zscore"] > params["recover_level"]) & (df["close"] > df["ema_trend"])
        sell = (df["zscore"] > params["take_profit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiIntradayIntensityBreakout(BaseStrategy):
    name = "openai_intraday_intensity_breakout"
    description = "盘中强度突破：收盘靠近高位且量价强度累积后突破买入"
    default_params = {"ii_period": 14, "ii_min": 0.12, "ii_exit": -0.02, "breakout_period": 14, "ema_period": 34, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        df["ii"] = _intraday_intensity(h, l, c, v, params["ii_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["ii"] > params["ii_min"]) & (df["close"] > df["breakout_high"]) & (df["close"] > df["ema_trend"])
        sell = (df["ii"] < params["ii_exit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiEaseMoveTrend(BaseStrategy):
    name = "openai_ease_move_trend"
    description = "EMV 趋势：Ease of Movement 转正并突破趋势高点时买入"
    default_params = {"emv_period": 14, "emv_floor": 0.05, "emv_exit": -0.02, "ema_period": 34, "breakout_period": 16, "atr_stop_mul": 2.5}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, v = df["high"], df["low"], df["volume"]
        c = df["close"]
        df["emv"] = ta.volume.ease_of_movement(h, l, v, window=params["emv_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["emv"], params["emv_floor"]) & (df["close"] > df["ema_trend"]) & (df["close"] > df["breakout_high"])
        sell = (df["emv"] < params["emv_exit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiCmfReclaim(BaseStrategy):
    name = "openai_cmf_reclaim"
    description = "CMF 回收：资金流转正并重新站上趋势均线时买入"
    default_params = {"cmf_period": 20, "cmf_floor": 0.03, "cmf_exit": -0.02, "ema_period": 50, "atr_stop_mul": 2.4}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        df["cmf"] = ta.volume.chaikin_money_flow(h, l, c, v, window=params["cmf_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["cmf"], params["cmf_floor"]) & (df["close"] > df["ema_trend"])
        sell = (df["cmf"] < params["cmf_exit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiForceTrendPop(BaseStrategy):
    name = "openai_force_trend_pop"
    description = "Force Index 趋势弹升：力量指数放大并突破高点时买入"
    default_params = {"fi_period": 13, "fi_floor": 0.0, "ema_period": 34, "breakout_period": 14, "atr_stop_mul": 2.6}

    def add_indicators(self, df, params):
        df = df.copy()
        c, v = df["close"], df["volume"]
        df["force_index"] = ta.volume.force_index(c, v, window=params["fi_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["force_index"], params["fi_floor"]) & (df["close"] > df["breakout_high"]) & (df["close"] > df["ema_trend"])
        sell = (df["force_index"] < 0) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiVolumeEfficiencyTrend(BaseStrategy):
    name = "openai_volume_efficiency_trend"
    description = "量能效率趋势：价格动量与量比共振时买入"
    default_params = {"roc_period": 10, "vol_period": 20, "eff_floor": 0.8, "ema_period": 50, "atr_stop_mul": 2.7}

    def add_indicators(self, df, params):
        df = df.copy()
        c, v = df["close"], df["volume"]
        df["roc"] = ta.momentum.roc(c, window=params["roc_period"])
        df["vol_ratio"] = _vol_ratio(v, params["vol_period"])
        df["efficiency"] = df["roc"] * df["vol_ratio"]
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["efficiency"], params["eff_floor"]) & (df["close"] > df["ema_trend"])
        sell = (df["efficiency"] < 0) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiInsideBarThrust(BaseStrategy):
    name = "openai_inside_bar_thrust"
    description = "Inside Bar 推进：内包后向上突破并保持趋势时买入"
    default_params = {"ema_period": 34, "atr_stop_mul": 2.2}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["inside_bar"] = (h < h.shift(1)) & (l > l.shift(1))
        df["atr"] = _atr14(h, l, c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = df["inside_bar"].shift(1).fillna(False) & (df["close"] > df["high"].shift(1)) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["low"].shift(1)) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiWideRangeReversal(BaseStrategy):
    name = "openai_wide_range_reversal"
    description = "宽幅反转：大实体洗盘后被阳线收复时买入"
    default_params = {"range_period": 20, "body_floor": 0.55, "ema_period": 55, "atr_stop_mul": 2.3}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, o, c = df["high"], df["low"], df["open"], df["close"]
        tr = _true_range(h, l, c)
        df["range_ratio"] = tr / tr.rolling(params["range_period"]).mean().replace(0, np.nan)
        df["body_ratio"] = _body_ratio(o, h, l, c)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_bear = (df["close"].shift(1) < df["open"].shift(1)) & (df["range_ratio"].shift(1) > 1.25) & (df["body_ratio"].shift(1) > params["body_floor"])
        buy = prev_bear & (df["close"] > df["high"].shift(1)) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["low"].shift(1)) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiGapReclaim(BaseStrategy):
    name = "openai_gap_reclaim"
    description = "跳空回收：向下跳空后当天收复昨收并回到趋势上方时买入"
    default_params = {"ema_period": 34, "gap_min": -0.02, "atr_stop_mul": 2.1}

    def add_indicators(self, df, params):
        df = df.copy()
        o, c = df["open"], df["close"]
        df["gap_pct"] = o / c.shift(1).replace(0, np.nan) - 1.0
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["gap_pct"] < params["gap_min"]) & (df["close"] > df["close"].shift(1)) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["open"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiThreeBarStrength(BaseStrategy):
    name = "openai_three_bar_strength"
    description = "三连强势：连续三根走强并放量时买入"
    default_params = {"vol_period": 20, "vol_ratio_min": 1.05, "ema_period": 34, "atr_stop_mul": 2.4}

    def add_indicators(self, df, params):
        df = df.copy()
        c, v = df["close"], df["volume"]
        df["vol_ratio"] = _vol_ratio(v, params["vol_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(df["high"], df["low"], c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        higher_close = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
        higher_low = (df["low"] > df["low"].shift(1)) & (df["low"].shift(1) > df["low"].shift(2))
        buy = higher_close & higher_low & (df["vol_ratio"] > params["vol_ratio_min"]) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["low"].shift(1)) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiHammerTrendConfirm(BaseStrategy):
    name = "openai_hammer_trend_confirm"
    description = "锤头确认：锤头后向上确认突破时买入"
    default_params = {"ema_period": 34, "lower_shadow_min": 0.45, "body_max": 0.35, "atr_stop_mul": 2.1}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, o, c = df["high"], df["low"], df["open"], df["close"]
        candle_range = (h - l).replace(0, np.nan)
        lower_shadow = (np.minimum(o, c) - l) / candle_range
        body = (c - o).abs() / candle_range
        df["hammer_like"] = (lower_shadow > params["lower_shadow_min"]) & (body < params["body_max"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(h, l, c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = df["hammer_like"].shift(1).fillna(False) & (df["close"] > df["high"].shift(1)) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["low"].shift(1)) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiChaikinVolBreak(BaseStrategy):
    name = "openai_chaikin_vol_break"
    description = "Chaikin 波动突破：波动率放大并向上突破时买入"
    default_params = {"ema_window": 10, "change_window": 10, "cv_floor": 4.0, "cv_exit": -2.0, "breakout_period": 18, "atr_stop_mul": 2.7}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["chaikin_vol"] = _chaikin_volatility(h, l, params["ema_window"], params["change_window"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["chaikin_vol"], params["cv_floor"]) & (df["close"] > df["breakout_high"])
        sell = (df["chaikin_vol"] < params["cv_exit"]) | (df["close"] < df["close"].rolling(10).mean())
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiMassIndexPop(BaseStrategy):
    name = "openai_mass_index_pop"
    description = "Mass Index 爆发：形态反转压力累积后向上突破时买入"
    default_params = {"fast": 9, "slow": 25, "mass_floor": 26.5, "mass_exit": 24.5, "ema_period": 50, "breakout_period": 20, "atr_stop_mul": 2.4}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["mass_index"] = ta.trend.mass_index(h, l, window_fast=params["fast"], window_slow=params["slow"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(h, l, c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["mass_index"], params["mass_floor"]) & (df["close"] > df["ema_trend"]) & (df["close"] > df["breakout_high"])
        sell = (df["mass_index"] < params["mass_exit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiRangeCompressionPulse(BaseStrategy):
    name = "openai_range_compression_pulse"
    description = "区间压缩脉冲：波动压缩后放量突破时买入"
    default_params = {"range_period": 20, "compression_max": 0.05, "breakout_period": 15, "vol_period": 20, "vol_ratio_min": 1.1, "atr_stop_mul": 2.6}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
        rolling_range = (h.rolling(params["range_period"]).max() - l.rolling(params["range_period"]).min()).replace(0, np.nan)
        df["compression"] = rolling_range / c.rolling(params["range_period"]).mean().replace(0, np.nan)
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["vol_ratio"] = _vol_ratio(v, params["vol_period"])
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["compression"].shift(1) < params["compression_max"]) & (df["close"] > df["breakout_high"]) & (df["vol_ratio"] > params["vol_ratio_min"])
        sell = (df["close"] < df["close"].rolling(10).mean()) | (df["vol_ratio"] < 0.8)
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiTrueRangePop(BaseStrategy):
    name = "openai_true_range_pop"
    description = "真实波幅脉冲：TR 暴涨后顺势突破时买入"
    default_params = {"tr_period": 14, "tr_ratio_min": 1.2, "tr_ratio_exit": 0.8, "ema_period": 34, "breakout_period": 14, "atr_stop_mul": 2.7}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        tr = _true_range(h, l, c)
        df["tr_ratio"] = tr / tr.rolling(params["tr_period"]).mean().replace(0, np.nan)
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = _atr14(h, l, c)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (df["tr_ratio"] > params["tr_ratio_min"]) & (df["close"] > df["ema_trend"]) & (df["close"] > df["breakout_high"])
        sell = (df["tr_ratio"] < params["tr_ratio_exit"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


class OpenAiDonchianMidReclaim(BaseStrategy):
    name = "openai_donchian_mid_reclaim"
    description = "唐奇安中轴回收：价格重新站上通道中轴并确认趋势时买入"
    default_params = {"channel_period": 20, "ema_period": 55, "atr_stop_mul": 2.3}

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        upper = h.rolling(params["channel_period"]).max()
        lower = l.rolling(params["channel_period"]).min()
        df["donchian_mid"] = (upper + lower) / 2.0
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = _atr14(h, l, c)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = _cross_up(df["close"], df["donchian_mid"]) & (df["close"] > df["ema_trend"])
        sell = (df["close"] < df["donchian_mid"]) | (df["close"] < df["ema_trend"])
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df

        # ─── 注册中心 ───────────────────────────────────────────────────────────


