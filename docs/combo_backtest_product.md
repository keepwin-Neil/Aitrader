# 组合回测 & 组合优化 — 产品文档

> 最后更新: 2026-04-06

## 1. 功能概述

### 1.1 组合回测
对股票池或单只股票，使用用户自选的多策略组合进行信号融合回测。核心逻辑：**任一策略发出 BUY 信号则全仓买入，任一策略发出 SELL 信号则全部卖出**（OR 信号合并）。每只股票独立 10 万初始资金。

### 1.2 组合优化
在已选的基础策略组合上，逐一叠加每个候选策略，找出加入哪个新策略能最大提升组合表现。输出可排序对比表格。

### 1.3 两两组合
测试所有候选策略的两两配对组合，找出最佳策略对。

---

## 2. 页面入口

**位置**: 策略评估标签页 → 第二行控制栏

```
[WF:通用标准 ▾] [Walk-Forward] [?] [-- 选择池 --] [选策略(N)] [组合回测] [组合优化] [两两组合] [策略适用性]
```

### 2.1 前置条件

| 功能 | 必须选池？ | 必须选策略？ | 说明 |
|------|----------|-----------|------|
| 组合回测 | 否（用 Symbol 栏单股） | 是（选策略按钮） | 池或 Symbol 二选一 |
| 组合优化 | 否（用 Symbol 栏单股） | 是（作为基础策略） | 候选策略自动取全部活跃策略 |
| 两两组合 | 否（用 Symbol 栏单股） | 否（自动取 Owner 筛选下活跃策略） | 候选不超过 20 个 |

---

## 3. 策略选择器

### 3.1 弹窗 UI
点击「选策略」按钮打开弹窗 (`#combo-strat-overlay`)：
- **快捷按钮**: 全选 / 全不选 / 只选收藏 / 选 Top5
- **策略列表**: 所有活跃策略，⭐ 标记收藏，复选框多选
- **组合管理**: 保存/加载/删除命名组合（存 localStorage `_savedCombos`）
- **底部**: 已选数量 + 确定按钮

### 3.2 数据结构
```javascript
// 全局变量
let _comboSelectedStrategies = new Set();  // 当前选中的策略名
let _savedCombos = JSON.parse(localStorage.getItem('_savedCombos') || '{}');
// _savedCombos = { "我的组合1": ["adaptive_ma", "fft_cycle_trend"], ... }
```

---

## 4. 组合回测

### 4.1 流程

```
用户操作                          前端                              后端
──────                          ────                              ────
选策略 → 选池/输入Symbol
点击「组合回测」            → POST /portfolio/combo-backtest
                                                          → 解析 pool 或 symbols
                                                          → ProcessPool 并行: 每只股票调 _combo_bt_one()
                                                          ← 返回 portfolio + per_stock + equity_curve
                            ← renderComboResults(data)
                               显示汇总卡片 + 个股表 + 权益曲线
点击某只股票行              → POST /portfolio/combo-stock-detail
                                                          → _combo_bt_one() 含 chart_data
                            ← renderComboKline()
                               显示 K线图 + 买卖标记 + 成交表
```

### 4.2 后端 API

#### `POST /portfolio/combo-backtest`

**请求体**:
```json
{
  "market": "US",
  "pool": "US Tech",           // 股票池名（与 symbols 二选一）
  "symbols": ["NVDA"],         // 直接传股票列表（与 pool 二选一）
  "strategies": ["adaptive_ma", "fft_cycle_trend"],
  "period": "1y",              // 支持: 1y, 2y, 5y, max, 自定义 "2025-01-01:2025-12-31"
  "interval": "1d"
}
```

**响应体**:
```json
{
  "portfolio": {
    "total_return": 14.16,
    "sharpe": 1.37,
    "max_drawdown": -12.91,
    "win_rate": 39.3,
    "total_trades": 56,
    "num_stocks": 8,
    "total_capital": 800000,
    "final_equity": 913295.00
  },
  "per_stock": [
    {
      "symbol": "AMD",
      "summary": {
        "total_return": 55.25,
        "final_equity": 155248,
        "trade_count": 7,
        "wins": 4,
        "losses": 3,
        "win_rate": 57.1,
        "capital": 100000,
        "total_commission": 47.89
      },
      "trades": [...],        // 完整交易记录
      "equity_curve": [...]   // 权益曲线（组合回测列表接口不含 chart_data）
    }
  ],
  "equity_curve": [...],       // 组合总权益曲线
  "strategies_used": ["adaptive_ma", "fft_cycle_trend"],
  "pool": "US Tech",
  "period": "1y"
}
```

#### `POST /portfolio/combo-stock-detail`

**请求体**: 同上 + `"symbol": "NVDA"`

**响应体**: 单只股票的完整数据，包含 `chart_data`（K 线数据 + 买卖信号标记）

### 4.3 核心函数: `_combo_bt_one()`

**文件**: `main.py`

**逻辑**:
1. `get_ohlcv()` 获取 K 线数据（自动按 period 过滤）
2. 对每个策略调用 `strategy.add_indicators()` + `strategy.generate_signals()` 获取信号列
3. OR 合并信号: `combo_buy = strat1.buy | strat2.buy | ...`
4. 模拟交易循环:
   - BUY: 全仓买入（cash × 95% / price - 佣金）
   - SELL: 全仓卖出（qty × price - 佣金）
   - 到期强制平仓（trigger = "到期平仓"）
5. 输出: summary, trades, equity_curve, chart_data

### 4.4 交易成本模型

采用 Interactive Brokers (盈透证券) Tiered 阶梯费率:

```python
IB_PER_SHARE = 0.0067   # $0.0035佣金 + $0.0002清算 + $0.003交易所
IB_MIN_ORDER = 0.35      # 每单最低 $0.35
IB_MAX_PCT = 0.01        # 每单最高 = 交易金额的 1%

def ib_commission(shares, price):
    raw = shares * IB_PER_SHARE
    trade_value = shares * price
    return max(IB_MIN_ORDER, min(raw, trade_value * IB_MAX_PCT))
```

每笔交易的佣金累计到 `summary.total_commission`。

### 4.5 到期强制平仓规则

回测截止日如仍有持仓，按当日收盘价强制卖出:
- `trigger_strategy = "到期平仓"`
- K 线图最后一根标记卖出
- 盈亏计入胜率/交易数统计
- `final_equity` = 全部变现后金额（不含浮盈）

---

## 5. 组合优化

### 5.1 流程

```
用户操作                          前端                              后端
──────                          ────                              ────
选策略（作为基础组合）
点击「组合优化」            → POST /portfolio/combo-optimize
                                                          → 基准: _combo_bt_one(base_strategies)
                                                          → 逐一: _combo_bt_one(base + candidate_i)
                                                          ← 返回 baseline + results[]
                            ← renderComboOptimizeResults(data)
                               显示可排序表格（基准行高亮 + vs基准列）
点击某行                    → POST /portfolio/combo-backtest (优化组合)
                            → POST /portfolio/combo-backtest (基础组合) ← 并行
                            ← renderComboOptDetailInline()
                               显示详情卡片（基准收益 → 优化收益 + delta）
                               显示个股表（基准收益 / 优化收益 / 交易成本）
点击某只股票                → POST /portfolio/combo-stock-detail
                            ← renderComboKline()
                               K 线图 + 策略颜色图例 + 成交表
```

### 5.2 后端 API

#### `POST /portfolio/combo-optimize`

**请求体**:
```json
{
  "market": "US",
  "pool": "US Tech",
  "base_strategies": ["adaptive_ma", "fft_cycle_trend"],
  "all_candidates": ["adaptive_ma", "fft_cycle_trend", "entropy_momentum", "dual_ma_cross", ...],
  "period": "1y",
  "interval": "1d"
}
```

**响应体**:
```json
{
  "baseline": {
    "added": "__baseline__",
    "strategies": ["adaptive_ma", "fft_cycle_trend"],
    "total_return": 19.92,
    "sharpe": 2.14,
    "max_drawdown": -10.39,
    "win_rate": 65.7,
    "total_trades": 67,
    "num_stocks": 8,
    "final_equity": 959360.00
  },
  "results": [
    { "added": "__baseline__", ... },
    { "added": "entropy_momentum", "strategies": ["adaptive_ma", "fft_cycle_trend", "entropy_momentum"], "total_return": 16.18, ... },
    { "added": "dual_ma_cross", ... }
  ],
  "base_strategies": ["adaptive_ma", "fft_cycle_trend"],
  "candidates_tested": 36
}
```

### 5.3 组合详情面板

点击结果表中的某行后，展开详情面板:

**标题行（一行紧凑布局）**:
```
组合详情  基准收益 19.92%  →  优化收益 16.18%  (-3.74%)  Sharpe 2.11  回撤 -14.57%  胜率 55.0%  交易 131  [adaptive_ma] [fft_cycle_trend] [entropy_momentum]  [关闭]
```

**个股表格**:
| 股票 | 基准收益% | 优化收益% | 交易数 | 胜率 | 交易成本 | 权益 |
|------|---------|---------|--------|------|--------|------|

- **基准收益%**: 并行请求基础策略的 per_stock 数据
- **交易成本**: 从 `summary.total_commission` 读取（IB Tiered 费率）

---

## 6. 两两组合

### 6.1 后端 API

#### `POST /portfolio/combo-pairs`

**请求体**:
```json
{
  "market": "US",
  "pool": "US Tech",
  "all_candidates": ["adaptive_ma", "fft_cycle_trend", "entropy_momentum", ...],
  "period": "1y",
  "interval": "1d"
}
```

**限制**: 候选策略 ≤ 20 个（C(20,2) = 190 对），超过则报错。

**响应体**:
```json
{
  "results": [
    {
      "pair": ["adaptive_ma", "fft_cycle_trend"],
      "strategy_a": "adaptive_ma",
      "strategy_b": "fft_cycle_trend",
      "total_return": 19.92,
      "sharpe": 2.14,
      "max_drawdown": -10.39,
      "win_rate": 65.7,
      "total_trades": 67,
      "num_stocks": 8,
      "final_equity": 959360.00
    }
  ],
  "pairs_tested": 190,
  "candidates": [...]
}
```

---

## 7. K 线图渲染

### 7.1 共享渲染函数

**函数**: `renderComboKline(area, symbol, data, closeTarget)`

**特性**:
- 使用 LightweightCharts 库
- 默认高度 750px，`resize: vertical` 可手动拖拽调整
- `ResizeObserver` 自动适应容器宽高变化

### 7.2 策略颜色系统

每个策略自动分配独立颜色:

```javascript
const _STRAT_COLORS = [
  '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4',
  '#84cc16', '#f97316', '#14b8a6', '#e879f9', '#facc15',
  '#22d3ee', '#a78bfa', '#fb923c', '#34d399', '#f472b6',
];
```

### 7.3 标记规则

- **标记文字**: 只显示 `B`（买入）/ `S`（卖出），简洁不遮挡 K 线
- **标记颜色**: 按触发策略分配（参考标题栏图例）
- **标题栏图例**: 每个策略显示圆点 + 策略名，颜色匹配标记
- **成交记录表**: 策略列也用对应颜色显示

---

## 8. Period 自定义区间

### 8.1 前端

Period 下拉框新增 `custom` 选项，选中后显示两个日期输入框:
```html
<select id="period" onchange="onPeriodChange()">
  ...
  <option value="custom">自定义区间</option>
</select>
<input type="date" id="periodStart">
<span>→</span>
<input type="date" id="periodEnd">
```

`getSelectedPeriod()` 返回 `"2025-01-01:2025-12-31"` 格式。

### 8.2 后端

`backtest.py:_filter_period()` 解析 `start:end` 格式:
```python
if ":" in period:
    start = pd.Timestamp(parts[0])
    end = pd.Timestamp(parts[1]) + timedelta(days=1)  # inclusive
    return df[(df.index >= start) & (df.index < end)]
```

---

## 9. 关键文件

| 文件 | 职责 |
|------|------|
| `static/index.html` | 前端 UI（策略选择器、结果渲染、K 线图） |
| `routers/trading.py` | 后端 API 路由（combo-backtest, combo-optimize, combo-pairs, combo-stock-detail） |
| `main.py:_combo_bt_one()` | 核心回测函数（信号合并、交易模拟、佣金计算） |
| `backtest.py:run_backtest()` | 标准单策略回测引擎（也已更新 IB 佣金） |
| `backtest.py:_filter_period()` | 数据时间范围过滤（支持自定义日期区间） |

---

## 10. 策略参数来源

### 10.1 参数体系

系统中有多套策略参数，调用策略时**必须明确注明**使用的是哪套：

| 参数来源 | 文件位置 | 说明 |
|---------|---------|------|
| 默认参数 | `strategy.default_params` | 策略代码内置原始默认值 |
| 通用标准 | `universal_params.json` | 对所有股票通用的优化参数 |
| 个股标准 | `custom_params/{SYMBOL}.json` | 通用标准 + 个股优化覆盖 |
| 最优参数 | `optimized_params/{SYMBOL}.json` | 各优化方法（贝叶斯/穷举/粗搜精搜）的最佳参数 |

### 10.2 当前各功能使用的参数

| 功能 | 参数来源 | 加载函数 |
|------|---------|---------|
| 组合回测 / 组合优化 | **个股标准** | `_get_standard_strategy_params(symbol, strategy)` |
| Run Comparison (标准模式) | **个股标准** | 同上 |
| Run Comparison (通用模式) | **通用标准** | `_get_universal_strategy_params(strategy)` |
| Run Comparison (优化模式) | **最优参数** | 从 `optimized_params/{SYMBOL}.json` 加载 |
| Walk-Forward | **取决于参数来源下拉框** | 支持全部来源切换 |

### 10.3 开发规则

**必须遵守**：
1. **UI 标注**: 凡调用策略的界面，必须显示当前参数来源（如 "参数: 个股标准"）
2. **代码注释**: 调用策略的代码必须注释参数来源
3. **理想设计**: 提供参数来源选择（至少在策略选择弹窗中注明）

### 10.4 待办（TODO）
- [ ] 组合回测策略选择弹窗：标注当前使用"个股标准参数"
- [ ] 组合回测结果：注明参数来源
- [ ] 允许用户切换参数来源（通用标准 / 个股标准 / 最优参数）

---

## 11. 注意事项

1. **性能**: 组合优化需要 N+1 轮回测（N=候选策略数），单股约需 1-2 秒/轮，8 只股票 × 38 个候选 ≈ 60-80 秒
2. **竞态防护**: 前端使用 `_comboOptReqId` 防止旧请求覆盖新结果
3. **缓存**: `_combo_bt_one` 内部调用 `get_ohlcv` 有 2 分钟内存缓存，不会重复下载数据
4. **策略预热**: 策略信号需要历史数据预热（如 MA 需要 N 根 K 线），自定义短区间可能导致前期无信号
5. **ProcessPool vs 顺序**: combo-optimize 和 combo-pairs 目前**顺序执行**（因嵌套函数不可 pickle），后续可优化为将 worker 函数提至模块级别
