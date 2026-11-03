"""
多策略注册中心
每个策略实现 add_indicators + generate_signals，输出 buy_signal / sell_signal 列，
由通用 run_backtest 引擎执行回测。
"""

import pandas as pd
import numpy as np
import ta
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    name: str = ""
    description: str = ""
    default_params: dict = {}
    # 图表指标配置: {"overlay": [...], "subchart": [...]}
    # overlay: 叠加在K线主图上的指标 (如均线、布林带)
    # subchart: 显示在副图上的指标 (如RSI、MACD)
    # 每个指标: {"col": "列名", "label": "显示名", "color": "#颜色", "type": "line|area|histogram"}
    chart_indicators: dict = {"overlay": [], "subchart": []}
    # 指标公式定义: {"col_name": {"name": "显示名", "formula": "公式", "params": ["参数名"]}}
    indicator_formulas: dict = {}
    # 买卖信号规则: {"buy": "买入条件描述", "sell": "卖出条件描述"}
    signal_rules: dict = {"buy": "", "sell": ""}

    @abstractmethod
    def add_indicators(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        ...
