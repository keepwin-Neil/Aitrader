# 行情主站多路由（IB ↔ yfinance ↔ Cache）— 设计与运维（2026-05-15）

> **目标**：每个行情请求都自动选择当前空闲容量最大的数据源，IB Gateway 池满时**并行**溢出到 yfinance，而不是排队等 IB。
> **承载位置**：[quote_router.py](../quote_router.py) + [ib_gateway.py](../ib_gateway.py) + [routers/market.py](../routers/market.py)。

---

## 一、为什么要做

### 旧设计的两个问题

1. **`ib_gateway.py` 单 client + 全局 `threading.Lock()`**：整个进程所有 `fetch_quote` 调用都被锁串行化，即使 IB Gateway 本身支持 32 并发。30 线程同时 `/quotes` 会全堵在锁后 → 单次 `reqTickers` ~3s × 串行 = 50s+，前端 chip 刷新动辄秒级到分钟级挂起。
2. **`routers/market.py` 单线程优先级链**：`yfinance → IB → cache` 写死在 `/quote` 端点里。yfinance 一旦被限流（429），所有线程都等 IB；IB 也忙时全部落到 cache。**没有横向扩展能力**。

### 用户决策

> "主用 IB Gateway，yfinance 备用。当 IB 并发不够用时，yfinance 同时用，相当于两路并行，均衡负载。"

→ 抽象出多源 router + 各源容量感知 + 非阻塞 try_fetch。

---

## 二、架构

```
┌──────────────────────────────────────────────────────────┐
│  HTTP                                                    │
│  /quote/{m}/{sym}    /quotes (batch)                     │
│       │                    │                             │
│       ▼                    ▼                             │
│  ┌─────────────────────────────────────┐                 │
│  │ quote_router.py:router              │                 │
│  │                                     │                 │
│  │  fetch_quote(sym, mkt):             │                 │
│  │    for p in providers:              │  Pass 1: try    │
│  │      r = p.try_fetch(sym, mkt)      │  (非阻塞)        │
│  │      if r usable: return r          │                 │
│  │    for p in providers:              │  Pass 2: force  │
│  │      r = p.force_fetch(sym, mkt)    │  (可阻塞)        │
│  │      if r usable: return r          │                 │
│  │                                     │                 │
│  │  fetch_quotes(syms, mkt):           │                 │
│  │    ThreadPool(32) parallel dispatch │                 │
│  └────┬──────────────┬──────────────┬──┘                 │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  ┌─────────┐   ┌────────────┐   ┌──────────┐             │
│  │  IB     │   │  Yahoo     │   │  OHLCV   │             │
│  │ pool=N  │   │ Finance    │   │  cache   │             │
│  │ slots   │   │ Semaphore  │   │ parquet  │             │
│  │ (US/HK) │   │   (8)      │   │ (stale)  │             │
│  │         │   │ US/HK/CN/UK│   │   全市场  │             │
│  └────┬────┘   └────────────┘   └──────────┘             │
│       │                                                  │
│       ▼                                                  │
│  ib_gateway.py: 16 _IBSlot                               │
│   每 slot = 1 IB client + 自己的 asyncio loop +          │
│   dedicated thread (ib_insync 不允许跨线程共享 loop)       │
└──────────────────────────────────────────────────────────┘
```

---

## 三、Provider 三件套

定义在 [quote_router.py](../quote_router.py)，统一 `QuoteProvider` 接口：

```python
class QuoteProvider:
    name: str
    supported_markets: set
    def available(self) -> int           # 当前空闲容量（路由 + telemetry）
    def try_fetch(sym, mkt) -> dict|None  # 非阻塞, 饱和返回 None
    def force_fetch(sym, mkt) -> dict|None  # 可阻塞, 失败返回 None
```

### 3.1 IBProvider（优先级 1）

- **支持市场**：`{"US", "HK"}`（IB Gateway 对其他市场不支持或要额外订阅）
- **容量**：由各服务的 `IB_POOL_SIZE` 决定（见第五节预算）
- **非阻塞**：[ib_gateway.py:try_fetch_quote()](../ib_gateway.py)，`pool.get(block=False)`，池空立刻返回 None（不等）
- **数据**：实时 NBBO（订阅 USD 4.50/月 美股扩展实时数据组 + USD 10/月 NP Bundle + USD 1/月 Cboe One）

### 3.2 YahooProvider（优先级 2）

- **支持市场**：`{"US", "HK", "CN", "UK"}`
- **容量**：`threading.Semaphore(8)`（env `YF_MAX_CONCURRENT`），避免 yfinance ~60 req/min/IP 限制下批量 429
- **数据**：免费，正常延迟 ~15-20 分钟（market hours 内 fast_info 也基本实时）

### 3.3 CacheProvider（优先级 3，永远可用）

- **支持市场**：`{"US", "HK", "CN", "UK", "CRYPTO"}`
- **容量**：999（本地 parquet 读，永不饱和）
- **数据**：本地 OHLCV 缓存最近收盘价，`stale=True` 标识，给前端显示 `OHLCV cache (close YYYY-MM-DD)`

---

## 四、IB 池实现细节

[ib_gateway.py](../ib_gateway.py) 的核心是 `_IBSlot` 类。

### 4.1 为什么每 slot 一个独立线程 + asyncio loop

`ib_insync` 把异步 I/O 包装成同步 API，每个 `IB()` 实例**强绑定**一个 asyncio event loop。两个线程共享同一个 `IB()` 会乱（asyncio task 调度跨线程行为未定义）。

合法的并发方式只剩一种：N 个独立 `IB()` 实例，每个在自己的 daemon thread 里跑自己的 `loop.run_forever()`，调用方通过 `asyncio.run_coroutine_threadsafe(coro, slot.loop)` 跨线程提交协程。

```python
class _IBSlot:
    def _run(self):  # 在 daemon thread 里跑
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.ib = IB()
        self.ib.connect(host, port, clientId=...)
        self.loop.run_forever()  # 阻塞直至 stop()

    def submit(self, coro_factory, timeout=10):
        fut = asyncio.run_coroutine_threadsafe(coro_factory(self.ib), self.loop)
        return fut.result(timeout=timeout)
```

### 4.2 clientId 分配（关键）

IB Gateway 要求每个 client 连接的 clientId 全局唯一。如果两个进程用同一个 ID 连，第二个会把第一个踢掉。

我们的方案（[ib_gateway.py:_ensure_pool()](../ib_gateway.py)）：

```python
base = cfg["client_id_base"] + (pid % 1000) * 32
# slot i 的 clientId = base + i
# 例: pid=200 → base=77+200*32=6477, 16 slot = [6477..6492]
# 例: pid=201 → base=77+201*32=6509, 16 slot = [6509..6524]
```

每个 pid 占 32-wide bucket，slot 用前 N 个。子进程 worker（强制 `pool_size=1`）只用第 0 个，永不和主池碰撞。

### 4.3 子进程检测

[ib_gateway.py:_default_pool_size()](../ib_gateway.py)：

```python
def _default_pool_size():
    if mp.current_process().name != "MainProcess":
        return 1  # 子进程 worker (scanner / _scan_one_stock_mp 等) 只要 1 个 client
    return int(os.getenv("IB_POOL_SIZE", "16"))
```

避免 8 个 scanner 子进程各开 16 slot → 128 connections 把 IB Gateway 32 上限轰穿。

### 4.4 进程退出清理

`atexit.register(_shutdown_pool)` 在解释器退出前调 `slot.stop()` 给每个 IB 做 graceful disconnect。

**坑**：`ib_insync.IB.__del__` 在 GC 时还会再调 `disconnect`，那时 CPython 已经关闭了 event loop → 抛 `RuntimeError: Event loop is closed` 一堆 traceback。`stop()` 里把 `self.ib.disconnect = lambda: None` 替换掉来压噪音。

---

## 五、连接预算

**IB Gateway 硬上限 32 并发 client**（gateway 进程内 Java 队列限制），本机所有进程共享：

| 进程 | `IB_POOL_SIZE` | 用途 |
|------|----------------|------|
| main:app (9000) | `start_local.sh` 默认 **4** | /quotes、模拟交易取价 |
| scanner 子进程 | 1（自动） | `_scan_one_stock_mp`，8 个并行（[main.py:SCAN_WORKERS=8](../main.py)） |

加开回测 worker 等其它连 IB 的进程时，各自的 `IB_POOL_SIZE` 加总不要超过 31（留 1 余量）。在 `.env` 或启动脚本里设置。

---

## 六、路由策略（重要）

`fetch_quote(sym, mkt)` 走两遍 provider 列表：

### Pass 1（非阻塞）

按 IB → YF → Cache 顺序调 `try_fetch`：
- **IB 池满**（`pool.get(block=False)` 拿不到 slot）→ 返回 None，**不等**
- **YF Semaphore 满**（`acquire(blocking=False)` 失败）→ 返回 None，**不等**
- **Cache 不存在**（本地 parquet 没数据）→ 返回 None

第一个返回 usable（`price is not None`）的就赢，**典型场景这一遍就结束**。

### Pass 2（阻塞）

如果 Pass 1 全 None（极端情况：IB 全忙 + YF 全忙），按同顺序调 `force_fetch`（可阻塞）。Cache 几乎总能返回 stale-but-real 价格，所以最坏情况兜底 OK。

### 批量调用（关键并发点）

`fetch_quotes(symbols, market)` 用 `ThreadPoolExecutor(32)` 并行跑 `fetch_quote`，**每个 thread 独立调 `pick provider`**，自动看 IB 池剩多少。

**实测例子**：

| Batch | 拆分 | 时延 |
|-------|------|------|
| 20 US cold | 前 16 → IB, 后 4 → YF（并行）| **3.15s** |
| 5 CN cold | 全 YF（IB 不支持 CN） | 1.01s |

之前同样 20 US cold + 池满场景下，会挂 30-60s 直到 urllib 超时。

---

## 七、外部调用方

代码里**只**两种用 IB 行情的方式：

### 7.1 通过 router（推荐）

```python
from quote_router import router
q = router.fetch_quote("AAPL", "US")        # 单
qs = router.fetch_quotes(["AAPL"], "US")   # 批量
```

当前用户：[routers/market.py:get_quote()](../routers/market.py)、[routers/market.py:get_batch_quotes()](../routers/market.py)。

### 7.2 直接调 IB（要 IB 实时, 不要 fallback）

```python
from ib_gateway import fetch_quote as ib_fetch_quote
q = ib_fetch_quote("AAPL", "US")  # 阻塞等 slot, IB 失败返回 None
```

当前用户：[main.py:49](../main.py) 导入但**实际只用作 fallback 链中的一段**（已迁移到 router）。

> ⚠️ 未来 paper trade 模拟交易等"必须用 live"的场景应改成显式调 `ib_fetch_quote`，而不是走 router——router 会在 IB 不可用时降级到 yfinance 或 cache，那时拿到的就不是真实时价了。

---

## 八、Env 变量速查

| Env | 默认 | 影响 | 注释 |
|-----|------|------|------|
| `IB_POOL_SIZE` | 16 | 每进程 IB pool 大小 | 在 .env 或启动脚本里按进程设 |
| `IB_GATEWAY_HOST` | 127.0.0.1 | 主 host | 本机 IBKR Gateway |
| `IB_GATEWAY_PORT` | 4002 | 主 port | 4002=paper, 4001=live, 我们 readonly 都行 |
| `IB_GATEWAY_FALLBACK_HOST` | 127.0.0.1 | 备 host | 偶发本机也跑 gateway |
| `IB_GATEWAY_FALLBACK_PORT` | 4001 | 备 port | |
| `IB_GATEWAY_CLIENT_ID` | 77 | clientId 基址 | 一般不动 |
| `IB_MARKET_DATA_TYPE` | 1 | 1=live, 3=delayed | 订阅生效后默认 1，失败自动 fallback 到 3 |
| `IB_HISTORY_TIMEOUT_SEC` | 8 | reqHistoricalData 超时 | |
| `YF_MAX_CONCURRENT` | 8 | yfinance 并发上限 | 防 429 |
| `AITRADER_SCAN_WORKERS` | 8 | scanner ProcessPool 大小 | [main.py:SCAN_WORKERS](../main.py) |

---

## 九、调试 / 运维

### 9.1 看 pool 实时状态

```bash
curl -sS http://localhost:9000/market-data/ib/status | python3 -m json.tool
```

返回：
```json
{
  "connected": true,
  "pool_size_target": 12,
  "pool_size_actual": 12,    // 实际连上的 slot 数（连不上会少）
  "pool_idle": 12,           // 当前空闲（用完会回到 idle）
  "active_endpoint": {"host": "127.0.0.1", "port": 4001},
  "slot_client_ids": [...]
}
```

`pool_size_actual < pool_size_target` 说明有 slot 连不上 IB Gateway——查 Gateway 端连接数是不是撞 32 上限、或 host:port 不通。

### 9.2 看后台 ProcessPool 子进程数

```bash
ps -ef | grep multiprocessing | grep -v grep | wc -l
```

scanner 高峰应 ≤ 8（单 market），市场切换瞬间可能短暂到 16（旧池在 tear-down，新池在启动）。

### 9.3 看 IB Gateway 端实际并发

打开 IBKR Gateway GUI → "连接状态" → "API 客户端" 那一行：
- 显示数字 = 当前活跃 client 连接数
- 应该 ≤ 31（我们的预算）
- 标签栏 history 是会话内连过的所有 ID 列表，**不是当前活跃**——单独看右侧数字

### 9.4 yfinance 429 排查

```bash
grep "Too Many Requests" logs/uvicorn_9000.log | tail
```

短时间大量 429 → 后台扫描器并发太高，调小 `AITRADER_SCAN_WORKERS`，或 router 的 yf Semaphore 太宽，调小 `YF_MAX_CONCURRENT`。

### 9.5 Cache 触发（说明 IB+YF 都挂了）

```bash
grep "OHLCV cache" logs/uvicorn_9000.log | tail
```

如果 cache 频繁触发，说明 IB 池真的不够用 + yfinance 也连不上——查网络、查 IB Gateway 服务状态。

---

## 十、已知坑 / Future Work

### 10.1 Pool 初始化要 5 秒

冷启动时 16 个 slot 各自尝试 4 个 endpoint（每个 2s 超时），并行执行总耗时 ≈ 5s。第一个 `/market-data/ib/status` 请求会阻塞这 5s。服务启动时一次性付出，影响不大，调试时要意识到。

### 10.2 Failed slots 不自动重连

slot 连接掉了后从池里"消失"，capacity 永久减少直到 service restart。考虑：每个 slot 加心跳，掉线后自动重连。**当前不做**——connect 本身偶发失败是常态，重试逻辑容易引入复杂度，restart 服务成本很低。

### 10.3 yfinance Semaphore 是进程内的

同时开多个进程（main + 回测 worker）时每个都各有 8 个 yfinance 并发，加起来可能触 yahoo 速率限制。**当前不做**——yfinance 是 backup，主流量在 IB；如果发现 yfinance 429 频繁，按进程分别调小 `YF_MAX_CONCURRENT`。

### 10.4 历史数据没走 router

`fetch_history()` ([ib_gateway.py](../ib_gateway.py)) 还是直接走 IB slot pool，没有 yfinance 备份。`backtest.get_ohlcv()` 自己有 yfinance + parquet cache 优先级链，独立于 router。后续可考虑统一。

### 10.5 paper trade 走 router 会拿到 cache stale 值

第 7.2 节提到的：模拟交易"必须 live"场景目前走的是 `get_quote()` → router → cache fallback。当 IB+YF 全挂时会返回 stale 价格做 PnL，可能会让用户看到"假"盈利。要修要么改 paper trade 调用方明确要 live，要么 router 加个 `force_live` 参数禁用 cache fallback。

---
