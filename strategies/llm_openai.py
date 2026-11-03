import pandas as pd
import numpy as np
import ta

from .base import BaseStrategy

# ─── OpenAI 候选策略：VHF Regime Breakout ─────────────────────────────

class VhfRegimeBreakout(BaseStrategy):
    name = "openai_vhf_regime_breakout"
    description = "VHF 趋势过滤：高 VHF 趋势状态下突破近期高点时买入"
    indicator_formulas = {
        "vhf": {"name": "VHF", "formula": "(Max(Close,N)-Min(Close,N))/Sum(abs(diff(Close)),N)", "params": ["vhf_period"]},
        "breakout_high": {"name": "Breakout High", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
        "breakout_low": {"name": "Breakout Low", "formula": "Min(Low[-1], N)", "params": ["exit_period"]},
    }
    signal_rules = {
        "buy": "VHF > vhf_min AND Close > Breakout_High",
        "sell": "Close < Breakout_Low OR VHF < vhf_exit",
    }
    chart_indicators = {
        "overlay": [
            {"col": "breakout_high", "label": "Breakout High", "color": "#ef5350", "dash": True},
            {"col": "breakout_low", "label": "Breakout Low", "color": "#66bb6a", "dash": True},
        ],
        "subchart": [{"col": "vhf", "label": "VHF", "color": "#42a5f5"}],
    }
    default_params = {
        "vhf_period": 28,
        "breakout_period": 25,
        "exit_period": 12,
        "vhf_min": 0.28,
        "vhf_exit": 0.18,
        "atr_stop_mul": 2.7,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        highest_close = c.rolling(params["vhf_period"]).max()
        lowest_close = c.rolling(params["vhf_period"]).min()
        path = c.diff().abs().rolling(params["vhf_period"]).sum()
        df["vhf"] = (highest_close - lowest_close) / path.replace(0, np.nan)
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["breakout_low"] = l.shift(1).rolling(params["exit_period"]).min()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = ((df["vhf"] > params["vhf_min"]) & (df["close"] > df["breakout_high"]))
        sell = ((df["close"] < df["breakout_low"]) | (df["vhf"] < params["vhf_exit"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：ADX Donchian Thrust ──────────────────────────────

class AdxDonchianThrust(BaseStrategy):
    name = "openai_adx_donchian_thrust"
    description = "ADX Donchian 推进：高 ADX 趋势状态下突破唐奇安上轨时买入"
    indicator_formulas = {
        "donchian_high": {"name": "Donchian High", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
        "donchian_low": {"name": "Donchian Low", "formula": "Min(Low[-1], N)", "params": ["exit_period"]},
        "adx": {"name": "ADX", "formula": "ADX(high, low, close, N)", "params": ["adx_period"]},
    }
    signal_rules = {
        "buy": "ADX > adx_min AND Close > Donchian_High",
        "sell": "Close < Donchian_Low OR ADX < adx_exit",
    }
    chart_indicators = {
        "overlay": [
            {"col": "donchian_high", "label": "Donchian High", "color": "#ef5350", "dash": True},
            {"col": "donchian_low", "label": "Donchian Low", "color": "#66bb6a", "dash": True},
        ],
        "subchart": [{"col": "adx", "label": "ADX", "color": "#42a5f5"}],
    }
    default_params = {
        "breakout_period": 24,
        "exit_period": 12,
        "adx_period": 14,
        "adx_min": 18,
        "adx_exit": 14,
        "atr_stop_mul": 2.8,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["donchian_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["donchian_low"] = l.shift(1).rolling(params["exit_period"]).min()
        df["adx"] = ta.trend.ADXIndicator(high=h, low=l, close=c, window=params["adx_period"]).adx()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = ((df["adx"] > params["adx_min"]) & (df["close"] > df["donchian_high"]))
        sell = ((df["close"] < df["donchian_low"]) | (df["adx"] < params["adx_exit"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：EMA Slope Breakout ──────────────────────────────

class EmaSlopeBreakout(BaseStrategy):
    name = "openai_ema_slope_breakout"
    description = "EMA 斜率突破：EMA 斜率转强且价格突破近期高点时买入"
    indicator_formulas = {
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
        "ema_slope": {"name": "EMA Slope", "formula": "EMA(t)-EMA(t-slope_lookback)", "params": ["slope_lookback"]},
        "breakout_high": {"name": "Breakout High", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
    }
    signal_rules = {
        "buy": "EMA_Slope > slope_min AND Close > EMA_Trend AND Close > Breakout_High",
        "sell": "EMA_Slope < slope_exit OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"},
            {"col": "breakout_high", "label": "Breakout High", "color": "#ef5350", "dash": True},
        ],
        "subchart": [{"col": "ema_slope", "label": "EMA Slope", "color": "#42a5f5"}],
    }
    default_params = {
        "ema_period": 34,
        "slope_lookback": 5,
        "breakout_period": 20,
        "slope_min": 0.15,
        "slope_exit": -0.05,
        "atr_stop_mul": 2.6,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["ema_slope"] = df["ema_trend"] - df["ema_trend"].shift(params["slope_lookback"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (
            (df["ema_slope"] > params["slope_min"])
            & (df["close"] > df["ema_trend"])
            & (df["close"] > df["breakout_high"])
        )
        sell = (
            (df["ema_slope"] < params["slope_exit"])
            | (df["close"] < df["ema_trend"])
        )
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：Vortex Trend Pulse ───────────────────────────────

class VortexTrendPulse(BaseStrategy):
    name = "openai_vortex_trend_pulse"
    description = "Vortex 趋势脉冲：VI+ 上穿 VI- 且价格位于趋势 EMA 上方时买入"
    indicator_formulas = {
        "vi_plus": {"name": "VI+", "formula": "Sum(VM+, N) / Sum(TR, N)", "params": ["vi_period"]},
        "vi_minus": {"name": "VI-", "formula": "Sum(VM-, N) / Sum(TR, N)", "params": ["vi_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
        "adx": {"name": "ADX", "formula": "ADX(high, low, close, N)", "params": ["adx_period"]},
    }
    signal_rules = {
        "buy": "VI+ 上穿 VI- AND Close > EMA_Trend AND ADX > adx_min",
        "sell": "VI+ 下穿 VI- OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [
            {"col": "vi_plus", "label": "VI+", "color": "#66bb6a"},
            {"col": "vi_minus", "label": "VI-", "color": "#ef5350"},
            {"col": "adx", "label": "ADX", "color": "#42a5f5"},
        ],
    }
    default_params = {
        "vi_period": 12,
        "ema_period": 56,
        "adx_period": 20,
        "adx_min": 10.72,
        "atr_stop_mul": 2.35,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)
        vm_plus = (h - l.shift(1)).abs()
        vm_minus = (l - h.shift(1)).abs()
        tr_sum = tr.rolling(params["vi_period"]).sum().replace(0, np.nan)
        df["vi_plus"] = vm_plus.rolling(params["vi_period"]).sum() / tr_sum
        df["vi_minus"] = vm_minus.rolling(params["vi_period"]).sum() / tr_sum
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["adx"] = ta.trend.ADXIndicator(high=h, low=l, close=c, window=params["adx_period"]).adx()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_plus = df["vi_plus"].shift(1)
        prev_minus = df["vi_minus"].shift(1)
        buy = (
            (prev_plus <= prev_minus)
            & (df["vi_plus"] > df["vi_minus"])
            & (df["close"] > df["ema_trend"])
            & (df["adx"] > params["adx_min"])
        )
        sell = (
            ((prev_plus >= prev_minus) & (df["vi_plus"] < df["vi_minus"]))
            | (df["close"] < df["ema_trend"])
        )
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：Relative Vigor Reversal ─────────────────────────

class RelativeVigorReversal(BaseStrategy):
    name = "openai_relative_vigor_reversal"
    description = "Relative Vigor 反转：RVI 上穿信号线并站上趋势 EMA 时买入"
    indicator_formulas = {
        "rvi": {"name": "RVI", "formula": "SMA((Close-Open)/(High-Low), N)", "params": ["rvi_period"]},
        "rvi_signal": {"name": "RVI Signal", "formula": "SMA(RVI, 4)", "params": []},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "RVI 上穿 Signal AND RVI > rvi_floor AND Close > EMA_Trend",
        "sell": "RVI 下穿 Signal OR RVI > rvi_take_profit OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [
            {"col": "rvi", "label": "RVI", "color": "#42a5f5"},
            {"col": "rvi_signal", "label": "Signal", "color": "#ef5350"},
        ],
    }
    default_params = {
        "rvi_period": 15,
        "ema_period": 15,
        "rvi_floor": 0.023,
        "rvi_take_profit": 0.653,
        "atr_stop_mul": 3.61,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        strength = (c - o) / (h - l).replace(0, np.nan)
        df["rvi"] = strength.rolling(params["rvi_period"]).mean()
        df["rvi_signal"] = df["rvi"].rolling(4).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_rvi = df["rvi"].shift(1)
        prev_sig = df["rvi_signal"].shift(1)
        buy = (
            (prev_rvi <= prev_sig)
            & (df["rvi"] > df["rvi_signal"])
            & (df["rvi"] > params["rvi_floor"])
            & (df["close"] > df["ema_trend"])
        )
        sell = (
            ((prev_rvi >= prev_sig) & (df["rvi"] < df["rvi_signal"]))
            | (df["rvi"] > params["rvi_take_profit"])
            | (df["close"] < df["ema_trend"])
        )
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：KST Trend Signal ────────────────────────────────

class KstTrendSignal(BaseStrategy):
    name = "openai_kst_trend_signal"
    description = "KST 趋势信号：KST 上穿信号线且价格位于趋势 EMA 上方买入"
    indicator_formulas = {
        "kst": {"name": "KST", "formula": "Weighted ROC SMA blend", "params": ["roc1", "roc2", "roc3", "roc4"]},
        "kst_signal": {"name": "KST Signal", "formula": "SMA(KST, N)", "params": ["nsig"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "KST 上穿 Signal AND Close > EMA_Trend",
        "sell": "KST 下穿 Signal OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [
            {"col": "kst", "label": "KST", "color": "#42a5f5"},
            {"col": "kst_signal", "label": "Signal", "color": "#ef5350"},
        ],
    }
    default_params = {
        "roc1": 10, "roc2": 13, "roc3": 18, "roc4": 38,
        "window1": 8, "window2": 10, "window3": 10, "window4": 11,
        "nsig": 5, "ema_period": 75, "atr_stop_mul": 3.32,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]

        def _roc(series, window):
            return series.pct_change(window) * 100

        rcma1 = _roc(c, params["roc1"]).rolling(params["window1"]).mean()
        rcma2 = _roc(c, params["roc2"]).rolling(params["window2"]).mean()
        rcma3 = _roc(c, params["roc3"]).rolling(params["window3"]).mean()
        rcma4 = _roc(c, params["roc4"]).rolling(params["window4"]).mean()
        df["kst"] = rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4
        df["kst_signal"] = df["kst"].rolling(params["nsig"]).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_kst = df["kst"].shift(1)
        prev_sig = df["kst_signal"].shift(1)
        buy = (
            (prev_kst <= prev_sig)
            & (df["kst"] > df["kst_signal"])
            & (df["close"] > df["ema_trend"])
        )
        sell = (
            ((prev_kst >= prev_sig) & (df["kst"] < df["kst_signal"]))
            | (df["close"] < df["ema_trend"])
        )
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：Schaff Cycle Trend ──────────────────────────────

class SchaffCycleTrend(BaseStrategy):
    name = "openai_schaff_cycle_trend"
    description = "STC 金叉并处于上升区时买入"
    indicator_formulas = {
        "stc": {"name": "STC", "formula": "双重随机化 MACD 周期振荡器", "params": ["window_fast", "window_slow", "cycle"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "STC 上穿 stc_buy 且向上 AND Close > EMA_Trend",
        "sell": "STC 从高位回落 OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [{"col": "stc", "label": "STC", "color": "#42a5f5"}],
    }
    default_params = {
        "window_fast": 23,
        "window_slow": 50,
        "cycle": 10,
        "smooth1": 3,
        "smooth2": 3,
        "ema_period": 55,
        "stc_buy": 25,
        "stc_sell": 70,
        "atr_stop_mul": 2.8,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        ema_fast = ta.trend.ema_indicator(c, window=params["window_fast"])
        ema_slow = ta.trend.ema_indicator(c, window=params["window_slow"])
        macd = ema_fast - ema_slow
        macd_min = macd.rolling(params["cycle"]).min()
        macd_max = macd.rolling(params["cycle"]).max()
        fast_k = 100 * (macd - macd_min) / (macd_max - macd_min).replace(0, np.nan)
        fast_d = fast_k.ewm(span=params["smooth1"], adjust=False).mean()
        d_min = fast_d.rolling(params["cycle"]).min()
        d_max = fast_d.rolling(params["cycle"]).max()
        df["stc"] = (100 * (fast_d - d_min) / (d_max - d_min).replace(0, np.nan)).ewm(
            span=params["smooth2"], adjust=False
        ).mean()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_stc = df["stc"].shift(1)
        buy = (
            (prev_stc <= params["stc_buy"])
            & (df["stc"] > params["stc_buy"])
            & (df["stc"] > prev_stc)
            & (df["close"] > df["ema_trend"])
        )
        sell = (
            ((prev_stc >= params["stc_sell"]) & (df["stc"] < prev_stc))
            | (df["close"] < df["ema_trend"])
        )
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：PPO Trend Pulse ────────────────────────────────

class PpoTrendPulse(BaseStrategy):
    name = "openai_ppo_trend_pulse"
    description = "PPO 柱线翻正且价格位于趋势 EMA 之上买入"
    indicator_formulas = {
        "ppo": {"name": "PPO", "formula": "100*(EMAfast-EMAslow)/EMAslow", "params": ["window_fast", "window_slow"]},
        "ppo_signal": {"name": "PPO Signal", "formula": "EMA(PPO, N)", "params": ["window_sign"]},
        "ppo_hist": {"name": "PPO Hist", "formula": "PPO - Signal", "params": []},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "PPO Hist 上穿 0 AND Close > EMA_Trend",
        "sell": "PPO Hist 下穿 0 OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [{"col": "ppo_hist", "label": "PPO Hist", "color": "#42a5f5", "type": "histogram"}],
    }
    default_params = {
        "window_fast": 10,
        "window_slow": 28,
        "window_sign": 9,
        "ema_period": 55,
        "atr_stop_mul": 2.7,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        ema_fast = ta.trend.ema_indicator(c, window=params["window_fast"])
        ema_slow = ta.trend.ema_indicator(c, window=params["window_slow"]).replace(0, np.nan)
        df["ppo"] = 100 * (ema_fast - ema_slow) / ema_slow
        df["ppo_signal"] = df["ppo"].ewm(span=params["window_sign"], adjust=False).mean()
        df["ppo_hist"] = df["ppo"] - df["ppo_signal"]
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_hist = df["ppo_hist"].shift(1)
        buy = (
            (prev_hist <= 0)
            & (df["ppo_hist"] > 0)
            & (df["close"] > df["ema_trend"])
        )
        sell = (
            ((prev_hist >= 0) & (df["ppo_hist"] < 0))
            | (df["close"] < df["ema_trend"])
        )
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：Balance Power Breakout ──────────────────────────

class BalancePowerBreakout(BaseStrategy):
    name = "openai_balance_power_breakout"
    description = "多头收盘压力增强并突破近期高点时买入"
    indicator_formulas = {
        "bop": {"name": "BOP", "formula": "(Close-Open)/(High-Low)", "params": ["bop_smooth"]},
        "breakout_high": {"name": "Breakout High", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "BOP > bop_min AND Close > Breakout_High AND Close > EMA_Trend",
        "sell": "Close < EMA_Trend OR BOP < 0",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"},
            {"col": "breakout_high", "label": "Breakout High", "color": "#ef5350", "dash": True},
        ],
        "subchart": [{"col": "bop", "label": "BOP", "color": "#42a5f5"}],
    }
    default_params = {
        "bop_smooth": 5,
        "breakout_period": 20,
        "ema_period": 34,
        "bop_min": 0.2,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        bop_raw = (c - o) / (h - l).replace(0, np.nan)
        df["bop"] = bop_raw.rolling(params["bop_smooth"]).mean()
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (
            (df["bop"] > params["bop_min"])
            & (df["close"] > df["breakout_high"])
            & (df["close"] > df["ema_trend"])
        )
        sell = ((df["close"] < df["ema_trend"]) | (df["bop"] < 0))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：CCI Trend Reclaim ──────────────────────────────

class CciTrendReclaim(BaseStrategy):
    name = "openai_cci_trend_reclaim"
    description = "CCI 上穿零轴且价格位于趋势 EMA 上方买入"
    indicator_formulas = {
        "cci": {"name": "CCI", "formula": "CCI(high, low, close, N)", "params": ["cci_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "CCI 上穿 0 AND Close > EMA_Trend",
        "sell": "CCI < cci_exit OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [{"col": "cci", "label": "CCI", "color": "#42a5f5"}],
    }
    default_params = {
        "cci_period": 20,
        "ema_period": 50,
        "cci_exit": -50,
        "atr_stop_mul": 2.6,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["cci"] = ta.trend.cci(h, l, c, window=params["cci_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_cci = df["cci"].shift(1)
        buy = (
            (prev_cci <= 0)
            & (df["cci"] > 0)
            & (df["close"] > df["ema_trend"])
        )
        sell = ((df["cci"] < params["cci_exit"]) | (df["close"] < df["ema_trend"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：DPO Trend Reversal ─────────────────────────────

class DpoTrendReversal(BaseStrategy):
    name = "openai_dpo_trend_reversal"
    description = "DPO 翻正且价格收复趋势 EMA 时买入"
    indicator_formulas = {
        "dpo": {"name": "DPO", "formula": "Close[shift]-SMA(Close,N)", "params": ["dpo_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
    }
    signal_rules = {
        "buy": "DPO 上穿 0 AND Close > EMA_Trend AND RSI < rsi_cap",
        "sell": "DPO < 0 OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [{"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"}],
        "subchart": [
            {"col": "dpo", "label": "DPO", "color": "#42a5f5"},
            {"col": "rsi", "label": "RSI", "color": "#ab47bc"},
        ],
    }
    default_params = {
        "dpo_period": 20,
        "ema_period": 34,
        "rsi_period": 14,
        "rsi_cap": 68,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        sma = c.rolling(params["dpo_period"]).mean()
        shift = params["dpo_period"] // 2 + 1
        df["dpo"] = c.shift(shift) - sma
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_dpo = df["dpo"].shift(1)
        buy = (
            (prev_dpo <= 0)
            & (df["dpo"] > 0)
            & (df["close"] > df["ema_trend"])
            & (df["rsi"] < params["rsi_cap"])
        )
        sell = ((df["dpo"] < 0) | (df["close"] < df["ema_trend"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：ATR Expansion Breakout ────────────────────────

class AtrExpansionBreakout(BaseStrategy):
    name = "openai_atr_expansion_breakout"
    description = "波动扩张配合高点突破时买入"
    indicator_formulas = {
        "atr_ratio": {"name": "ATR Ratio", "formula": "ATR / SMA(ATR, N)", "params": ["atr_period", "atr_ma_period"]},
        "breakout_high": {"name": "Breakout High", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
    }
    signal_rules = {
        "buy": "ATR Ratio > atr_ratio_min AND Close > Breakout_High AND Close > EMA_Trend",
        "sell": "ATR Ratio < atr_ratio_exit OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"},
            {"col": "breakout_high", "label": "Breakout High", "color": "#ef5350", "dash": True},
        ],
        "subchart": [{"col": "atr_ratio", "label": "ATR Ratio", "color": "#42a5f5"}],
    }
    default_params = {
        "atr_period": 14,
        "atr_ma_period": 20,
        "breakout_period": 18,
        "ema_period": 34,
        "atr_ratio_min": 1.05,
        "atr_ratio_exit": 0.9,
        "atr_stop_mul": 2.8,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["atr_ratio"] = df["atr"] / df["atr"].rolling(params["atr_ma_period"]).mean().replace(0, np.nan)
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        buy = (
            (df["atr_ratio"] > params["atr_ratio_min"])
            & (df["close"] > df["breakout_high"])
            & (df["close"] > df["ema_trend"])
        )
        sell = ((df["atr_ratio"] < params["atr_ratio_exit"]) | (df["close"] < df["ema_trend"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：RSI Range Shift ────────────────────────────────

class RsiRangeShift(BaseStrategy):
    name = "openai_rsi_range_shift"
    description = "RSI 上穿强势区并站上趋势 EMA 时买入"
    indicator_formulas = {
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
        "ema_trend": {"name": "EMA Trend", "formula": "EMA(close, N)", "params": ["ema_period"]},
        "breakout_high": {"name": "Breakout High", "formula": "Max(High[-1], N)", "params": ["breakout_period"]},
    }
    signal_rules = {
        "buy": "RSI 上穿 rsi_buy AND Close > EMA_Trend AND Close > Breakout_High",
        "sell": "RSI < rsi_exit OR Close < EMA_Trend",
    }
    chart_indicators = {
        "overlay": [
            {"col": "ema_trend", "label": "EMA Trend", "color": "#ffa726"},
            {"col": "breakout_high", "label": "Breakout High", "color": "#ef5350", "dash": True},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#42a5f5"}],
    }
    default_params = {
        "rsi_period": 14,
        "ema_period": 34,
        "breakout_period": 20,
        "rsi_buy": 55,
        "rsi_exit": 45,
        "atr_stop_mul": 2.5,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        df["ema_trend"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["breakout_high"] = h.shift(1).rolling(params["breakout_period"]).max()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_rsi = df["rsi"].shift(1)
        buy = (
            (prev_rsi <= params["rsi_buy"])
            & (df["rsi"] > params["rsi_buy"])
            & (df["close"] > df["ema_trend"])
            & (df["close"] > df["breakout_high"])
        )
        sell = ((df["rsi"] < params["rsi_exit"]) | (df["close"] < df["ema_trend"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── OpenAI 候选策略：Keltner Midline Resume ─────────────────────────

class KeltnerMidlineResume(BaseStrategy):
    name = "openai_keltner_midline_resume"
    description = "价格回到 Keltner 中轴上方并转强时买入"
    indicator_formulas = {
        "kc_mid": {"name": "KC Mid", "formula": "EMA(close, N)", "params": ["ema_period"]},
        "kc_upper": {"name": "KC Upper", "formula": "KC Mid + ATR×K", "params": ["kc_mult"]},
        "kc_lower": {"name": "KC Lower", "formula": "KC Mid - ATR×K", "params": ["kc_mult"]},
        "rsi": {"name": "RSI", "formula": "RSI(close, N)", "params": ["rsi_period"]},
    }
    signal_rules = {
        "buy": "Close 重新站上 KC Mid AND RSI > rsi_floor",
        "sell": "Close < KC Mid OR RSI < rsi_exit",
    }
    chart_indicators = {
        "overlay": [
            {"col": "kc_mid", "label": "KC Mid", "color": "#ffa726"},
            {"col": "kc_upper", "label": "KC Upper", "color": "#9e9e9e", "dash": True},
            {"col": "kc_lower", "label": "KC Lower", "color": "#9e9e9e", "dash": True},
        ],
        "subchart": [{"col": "rsi", "label": "RSI", "color": "#42a5f5"}],
    }
    default_params = {
        "ema_period": 20,
        "atr_period": 14,
        "kc_mult": 1.8,
        "rsi_period": 14,
        "rsi_floor": 52,
        "rsi_exit": 44,
        "atr_stop_mul": 2.4,
    }

    def add_indicators(self, df, params):
        df = df.copy()
        h, l, c = df["high"], df["low"], df["close"]
        df["kc_mid"] = ta.trend.ema_indicator(c, window=params["ema_period"])
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=params["atr_period"])
        df["kc_upper"] = df["kc_mid"] + params["kc_mult"] * df["atr"]
        df["kc_lower"] = df["kc_mid"] - params["kc_mult"] * df["atr"]
        df["rsi"] = ta.momentum.rsi(c, window=params["rsi_period"])
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        prev_close = df["close"].shift(1)
        prev_mid = df["kc_mid"].shift(1)
        buy = (
            (prev_close <= prev_mid)
            & (df["close"] > df["kc_mid"])
            & (df["rsi"] > params["rsi_floor"])
        )
        sell = ((df["close"] < df["kc_mid"]) | (df["rsi"] < params["rsi_exit"]))
        df["buy_signal"] = buy.fillna(False)
        df["sell_signal"] = sell.fillna(False)
        return df


# ─── 策略仓库（已归档的策略仍保留代码，但不注册）──────────────────────
# 入库策略: MacdDivergence, KdjDmi, ElderTripleScreen
# 原因: 年均收益为负，表现持续落后指数

