---
name: find-new-str
description: 寻找新的投资策略：自动研发、回测验证、优化、WF验证、筛选入库的完整流水线。当用户想要发现新策略、扩充策略库时使用。
argument-hint: "[策略数量(默认3)]"
allowed-tools: Read, Write, Edit, Grep, Glob, Bash(*), Agent, WebSearch, WebFetch
effort: max
---

# 寻找新投资策略 — 完整工作流

你是一位量化策略研发专家。按以下 8 个步骤寻找并验证新的交易策略，全程自动化执行，遇到错误自行修复。

**项目路径**: 本仓库根目录
**策略文件**: `strategies/core.py`
**回测引擎**: `backtest.py` 中的 `run_strategy_backtest(df, strategy_name, params_override, capital=100000, buy_params=None, sell_params=None)`
**数据获取**: `backtest.py` 中的 `get_ohlcv(symbol, market, interval="1d", period="1y")`
**股票池配置**: `stock_pools.json`
**WEB API**: `http://localhost:9000`

### Buy/Sell 双参数分离架构

系统支持 buy 和 sell 信号使用**完全独立的参数**。这是核心架构要求，新策略必须遵循。

**原理**：策略的 `generate_signals(df, params)` 会被调用两次 —— 一次用 buy_params 提取 buy_signal，一次用 sell_params 提取 sell_signal，两个信号合并后用于回测和K线图显示。

**参数存储结构** (`custom_params/{SYMBOL}.json`):
```json
{
  "intervals": {
    "1d": {
      "strategy_name": {
        "params": {...},
        "signal_params": {
          "buy":  {"param_a": 10, "param_b": 30},
          "sell": {"param_a": 15, "param_b": 75}
        }
      }
    }
  }
}
```

**关键函数**:
- `main._get_dual_signal_params(symbol, strategy, interval)` → 返回 `{has_dual, buy_params, sell_params}`
- `main._save_signal_params(symbol, strategy, signal_type, params)` → 分别保存 buy/sell 参数
- `run_strategy_backtest(df, name, buy_params=bp, sell_params=sp)` → 双参数分离回测

**策略设计要求**：新策略的 buy 和 sell 信号应尽量使用**不同的参数名**来控制，以避免合并时冲突。例如：
- 好: `rsi_buy_threshold=30`, `rsi_sell_threshold=70`
- 差: 买卖共用一个 `rsi_period`（优化 buy 和 sell 时会互相覆盖）

---

## 步骤 1: 分析现有策略，研发新策略

1. 读取 `strategies/core.py`，总结当前所有策略的**类别分布**（趋势跟踪、均值回归、动量、波动率、成交量、振荡器等）
2. 分析哪些**类别缺失或薄弱**
3. 通过 WebSearch 搜索学术论文、量化社区中**不同于现有策略类别**的经典交易策略
4. 设计 $0 个（默认 3 个）新策略，每个策略必须：
   - 属于与现有策略**不同的类别或组合方式**
   - 继承 `BaseStrategy`，实现 `add_indicators()` 和 `generate_signals()`
   - 定义合理的 `default_params`、`chart_indicators`、`indicator_formulas`、`signal_rules`
   - 命名遵循 PascalCase 类名 + snake_case 的 `name` 属性
   - **Buy/Sell 参数分离设计**：`default_params` 中 buy 和 sell 相关参数必须使用不同的参数名，例如 `rsi_buy_threshold` 和 `rsi_sell_threshold`，不能共用同一个参数控制买卖。这样后续优化时 buy_params 和 sell_params 可以各自独立调优而不冲突
5. 将新策略代码追加到 `strategies/core.py` 的 `_archive()` 调用之前
6. 先用 `_archive()` 注册（不立即激活），等验证通过再决定

---

## 步骤 2: NASDAQ 股票池基准测试

对每个新策略，用 **通用默认参数** 在 NASDAQ 股票池中测试：

```python
# 伪代码逻辑
from backtest import get_ohlcv, run_strategy_backtest

pool = load_stock_pools()["NASDAQ"]  # 或实际的nasdaq池名
symbols = pool["symbols"]
results = []
for sym in symbols:
    df = get_ohlcv(sym, "US", interval="1d", period="1y")
    res = run_strategy_backtest(df, strategy_name, capital=100000)
    results.append(res["summary"]["total_return"])
avg_return = mean(results)
```

通过 API 或直接 Python 脚本执行，记录每个新策略在 NASDAQ 池中的 **1 年平均收益率**。

---

## 步骤 3: 贝叶斯优化通用标准参数

对每个新策略，通过 API 或脚本进行贝叶斯优化：

1. 在 NASDAQ 池中选取代表性股票（如随机 20-30 只）
2. 用贝叶斯搜索最优参数（最大化平均 Sharpe / 收益）
3. 将优化后的参数作为该策略的 `default_params`（通用标准参数）
4. 更新 `strategies.py` 中的 `default_params`
5. 再用优化后的通用参数重跑 NASDAQ 池，记录优化后的平均收益率

---

## 步骤 4: MYPOOL 验证

用优化后的通用标准参数，在 MYPOOL 股票池中验证：

1. 读取 `stock_pools.json` 中的 MYPOOL
2. 对每只股票用 `run_strategy_backtest` 计算 1 年收益
3. 计算 MYPOOL 的 **平均收益率**

---

## 步骤 5: 策略筛选（淘汰 / 入库）

对每个新策略：
- **平均收益率 < 20%** → 从 `strategies.py` 中删除该策略代码，记录淘汰原因
- **平均收益率 ≥ 20%** → 保留在 `_archive()` 中作为新策略候选

---

## 步骤 6: MYPOOL 批量优化（两轮，Buy/Sell 分离）

对通过筛选的策略，执行两轮批量优化，**必须使用 `signal_type: "buy_and_sell"` 模式**：

每轮流程：
1. 调用批量优化 API: `POST /batch-optimize/MYPOOL`，body 中指定：
   ```json
   {
     "strategies": ["new_strategy_1", "new_strategy_2"],
     "method": "bayesian",
     "signal_type": "buy_and_sell"
   }
   ```
   `signal_type: "buy_and_sell"` 会先用 buy 目标函数（success_rate + avg_mfe）优化 buy 参数，再用 sell 目标函数优化 sell 参数，两套参数独立存储到 `signal_params.buy` 和 `signal_params.sell`
2. 等待优化完成（轮询 `GET /batch-optimize/status/{job_id}`）
3. 优化完成后 WF 验证会自动执行（流水线并行）
4. WF 通过的策略参数自动升级到个股标准参数（`_promote_to_standard`），同时 buy/sell 分离参数保存到 `_save_signal_params`

两轮都要完成。

---

## 步骤 7: 最终验收（双参数分离回测）

用优化后的**个股标准参数**，以 **Buy/Sell 双参数分离模式**重跑 MYPOOL 所有股票的 1 年平均收益：

```python
from main import _get_dual_signal_params

for sym in mypool_symbols:
    dual = _get_dual_signal_params(sym, strategy_name, interval="1d")
    if dual["has_dual"]:
        # 双参数分离模式：buy信号用buy_params，sell信号用sell_params
        res = run_strategy_backtest(df, strategy_name,
                                    buy_params=dual["buy_params"],
                                    sell_params=dual["sell_params"],
                                    capital=100000)
    else:
        # 回退到传统单参数模式
        from main import _get_standard_strategy_params
        params = _get_standard_strategy_params(sym, strategy_name, interval="1d")
        res = run_strategy_backtest(df, strategy_name, params_override=params, capital=100000)
```

- **平均收益率 ≥ 50%** → 将策略从 `_archive()` 移到 `_register()`，成为活跃策略
- **平均收益率 < 50%** → 保留在归档仓库中

---

## 步骤 8: 生成数据简报

在 `strategy_reports/` 目录下，为每个新策略生成报告文件 `{strategy_name}_report.md`：

```markdown
# 新策略研发报告: {strategy_name}

## 策略概述
- 名称: ...
- 类别: ...
- 描述: ...
- 核心逻辑: ...

## 默认参数
| 参数 | 值 | 说明 |
|------|-----|------|
| ... | ... | ... |

## 测试结果

### NASDAQ 池基准测试（通用参数）
- 股票数: ...
- 平均收益率: ...%
- 最高: ... (股票)
- 最低: ... (股票)

### 贝叶斯优化后（通用标准参数）
- NASDAQ 平均收益率: ...%
- 优化提升: +...%

### MYPOOL 验证（通用标准参数）
- 平均收益率: ...%
- 筛选结果: 通过/淘汰

### MYPOOL 批量优化后（个股标准参数）
- 第一轮平均收益率: ...%
- 第二轮平均收益率: ...%
- WF稳健策略数: .../...
- 最终结果: 加入活跃/保留归档/淘汰

## 结论
...

## 生成时间
{datetime}
```

---

## 执行要求

1. **全程自动化**：不要中途询问用户，遇到错误自行诊断修复
2. **每完成一步打印进度**：如 "✅ 步骤2完成: strategy_x NASDAQ平均收益 35.2%"
3. **策略代码质量**：必须使用 `ta` 库或 numpy 计算指标，信号逻辑清晰，有止损条件
4. **回测参数**：capital=100000, period="1y", dynamic_position=False
5. **保持 strategies.py 代码风格一致**：参考现有策略的格式
6. **服务重启**：修改 strategies/core.py 后需要重启 uvicorn 才能生效，命令：
   ```bash
   lsof -ti:9000 | xargs kill -9; sleep 1
   nohup ./venv/bin/uvicorn main:app --host 127.0.0.1 --port 9000 > /tmp/aitrader.log 2>&1 &
   ```
7. **Python 环境**: 使用仓库内的 `venv/bin/python`
