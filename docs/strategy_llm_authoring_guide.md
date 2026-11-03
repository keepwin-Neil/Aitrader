# LLM 策略开发简介

这份文档给后续参与策略研发的 LLM 使用，目标是：

- 不必反复通读整份 [strategies.py](../strategies.py)
- 快速知道一个新策略最少需要实现什么
- 知道新增策略后还要同步哪些地方

## 1. 策略文件结构

核心文件是 [strategies.py](../strategies.py)。

每个策略都继承 `BaseStrategy`，最少包含：

- `name`
- `description`
- `default_params`
- `add_indicators(self, df, params)`
- `generate_signals(self, df, params)`

可选但建议补齐：

- `chart_indicators`
- `indicator_formulas`
- `signal_rules`

## 2. 最小实现模板

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    description = "一句话说明策略逻辑"
    default_params = {
        "fast_period": 10,
        "slow_period": 30,
    }
    chart_indicators = {
        "overlay": [],
        "subchart": [],
    }
    indicator_formulas = {}
    signal_rules = {
        "buy": "买入条件说明",
        "sell": "卖出条件说明",
    }

    def add_indicators(self, df, params):
        df = df.copy()
        # 计算技术指标列
        return df.dropna()

    def generate_signals(self, df, params):
        df = df.copy()
        df["buy_signal"] = ...
        df["sell_signal"] = ...
        df["buy_signal"] = df["buy_signal"].fillna(False)
        df["sell_signal"] = df["sell_signal"].fillna(False)
        return df
```

## 3. 编写规则

### 3.1 必须遵守

- `name` 必须唯一，使用小写蛇形命名，例如 `entropy_momentum`
- `add_indicators()` 必须返回带指标列的 `DataFrame`
- `generate_signals()` 必须输出：
  - `buy_signal`
  - `sell_signal`
- 信号列必须是布尔值，最后最好 `fillna(False)`

### 3.2 强烈建议

- 一开始就 `df = df.copy()`
- 避免未来函数/偷看未来数据
- 突破类策略通常要用 `shift(1)`，不要直接用当前 bar 的高低点
- 如果策略依赖止损，尽量补 `atr` 列，现有回测链路很多地方默认会用到
- `description` 要写成可读的一句话，而不是仅写缩写

## 4. 如何避免未来函数

最常见错误：

- 用当前 K 线的 `high/low` 去定义当前是否突破
- 用未移位的 rolling 高低点做信号

正确示例：

```python
df["highest"] = df["high"].shift(1).rolling(params["breakout_period"]).max()
df["buy_signal"] = df["close"] > df["highest"]
```

## 5. 图表配置怎么写

### 主图叠加

`chart_indicators["overlay"]`

适合放：

- 均线
- 通道
- 布林带
- VWAP

### 副图

`chart_indicators["subchart"]`

适合放：

- RSI
- MACD 柱
- 振荡器
- 量价派生指标

单个指标配置示例：

```python
{"col": "ema_fast", "label": "EMA10", "color": "#ff9800"}
{"col": "macd_hist", "label": "MACD Hist", "color": "#00bcd4", "type": "histogram"}
```

## 6. 新策略加入后还要改哪里

### 6.1 注册到策略中心

在 [strategies.py](../strategies.py) 底部：

- 想直接启用：加入 `_register(...)`
- 想先入仓库待验证：加入 `_archive(...)`

### 6.2 前端中文名和分组

在 [static/index.html](../static/index.html) 的 `ALL_STRATEGIES` 中补一项：

```javascript
{ name: 'my_strategy', label: '中文显示名', group: '趋势' }
```

否则前端仍能运行，但会只显示原始英文名，分组也会为空。

### 6.3 归属到哪个 LLM

策略归属定义在 [strategy_llm_registry.json](../strategy_llm_registry.json)。

当前规则：

- 未显式声明的策略，默认归属 `Claude`
- 如果是 OpenAI / Grok / Gemini / DeepSeek 新增的策略，要显式登记

示例：

```json
{
  "owners": {
    "my_strategy": "OpenAI"
  }
}
```

## 7. 参数设计建议

### 参数不要太多

第一版尽量控制在 `2-6` 个核心参数。

常见可优化参数：

- 周期长度
- 阈值
- 止损倍数
- 成交量过滤阈值

### 参数要有交易含义

不要为了让贝叶斯优化“有更多空间”而硬塞很多参数。  
参数越多，越容易过拟合。

## 8. 研发顺序建议

推荐流程：

1. 先写出最小可运行版本
2. 在单只股票上看图检查买卖点是否合理
3. 再跑优化
4. 再跑 Walk-Forward
5. 再决定是：
   - 进入活跃
   - 留作新策略
   - 直接归档

## 9. 现有系统默认口径

- 策略管理现在支持按 LLM 团队查看：
  - `Claude`
  - `OpenAI`
  - `Grok`
  - `Gemini`
  - `DeepSeek`
- 当前历史策略默认都归属 `Claude`

## 10. 给后续 LLM 的最短行动清单

如果你要新增一个策略，最少做这 4 件事：

1. 在 [strategies.py](../strategies.py) 新增策略类
2. 把策略加到 `_register(...)` 或 `_archive(...)`
3. 在 [static/index.html](../static/index.html) 的 `ALL_STRATEGIES` 里补中文名和分组
4. 在 [strategy_llm_registry.json](../strategy_llm_registry.json) 里登记归属团队

做到这 4 步，基本就能进入现有平台的策略管理、回测、优化和分类展示链路。
