import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from backtest import add_indicators, generate_signals, run_backtest, DEFAULT_PARAMS
from strategies import get_strategy


def make_ohlcv(n=200, base_price=100.0):
    """生成模拟OHLCV数据用于测试"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = base_price + np.cumsum(np.random.randn(n) * 2)
    close = np.maximum(close, 10)  # 保证价格为正
    df = pd.DataFrame({
        "open":   close + np.random.randn(n) * 0.5,
        "high":   close + abs(np.random.randn(n)) * 1.5,
        "low":    close - abs(np.random.randn(n)) * 1.5,
        "close":  close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    }, index=dates)
    return df


class TestAddIndicators:
    def test_all_columns_present(self):
        df = make_ohlcv()
        result = add_indicators(df, DEFAULT_PARAMS)
        expected_cols = ["ema_fast", "ema_mid", "ema_slow", "macd", "macd_signal",
                         "macd_hist", "rsi", "bb_upper", "bb_mid", "bb_lower",
                         "bb_width", "vol_ma", "vol_ratio", "atr"]
        for col in expected_cols:
            assert col in result.columns, f"缺少指标列: {col}"

    def test_no_nan_after_dropna(self):
        df = make_ohlcv()
        result = add_indicators(df, DEFAULT_PARAMS)
        assert result.isna().sum().sum() == 0

    def test_does_not_mutate_input(self):
        df = make_ohlcv()
        original_cols = list(df.columns)
        add_indicators(df, DEFAULT_PARAMS)
        assert list(df.columns) == original_cols


class TestGenerateSignals:
    def test_signal_columns_present(self):
        df = make_ohlcv()
        df = add_indicators(df, DEFAULT_PARAMS)
        result = generate_signals(df, DEFAULT_PARAMS)
        assert "buy_signal" in result.columns
        assert "sell_signal" in result.columns

    def test_signals_are_boolean(self):
        df = make_ohlcv()
        df = add_indicators(df, DEFAULT_PARAMS)
        result = generate_signals(df, DEFAULT_PARAMS)
        assert result["buy_signal"].dtype == bool
        assert result["sell_signal"].dtype == bool


class TestRunBacktest:
    def _prepare_df(self):
        df = make_ohlcv()
        df = add_indicators(df, DEFAULT_PARAMS)
        df = generate_signals(df, DEFAULT_PARAMS)
        return df

    def test_summary_keys(self):
        df = self._prepare_df()
        result = run_backtest(df, DEFAULT_PARAMS)
        expected_keys = ["capital", "final_equity", "total_return", "bh_return",
                         "max_drawdown", "sharpe", "win_rate", "profit_factor",
                         "total_trades", "win_trades", "lose_trades", "avg_win", "avg_loss"]
        for key in expected_keys:
            assert key in result["summary"], f"summary 缺少: {key}"

    def test_capital_preserved_or_changed(self):
        df = self._prepare_df()
        result = run_backtest(df, DEFAULT_PARAMS)
        assert result["summary"]["final_equity"] > 0

    def test_equity_curve_not_empty(self):
        df = self._prepare_df()
        result = run_backtest(df, DEFAULT_PARAMS)
        assert len(result["equity_curve"]) > 0

    def test_max_drawdown_non_positive(self):
        df = self._prepare_df()
        result = run_backtest(df, DEFAULT_PARAMS)
        assert result["summary"]["max_drawdown"] <= 0

    def test_no_trades_when_no_signals(self):
        df = self._prepare_df()
        df["buy_signal"] = False
        df["sell_signal"] = False
        result = run_backtest(df, DEFAULT_PARAMS)
        assert result["summary"]["total_trades"] == 0
        assert result["summary"]["final_equity"] == DEFAULT_PARAMS["capital"]


class TestStrategyLookup:
    def test_short_strategy_name_resolves_unique_prefixed_strategy(self):
        strategy = get_strategy("trix_momentum")

        assert strategy.name == "claude_trix_momentum"
