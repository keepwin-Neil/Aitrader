# LLM 调用规范

**Status:** 强制  
**Owner:** core_engine.py · routers/llm_admin.py  
**Last-updated:** 2026-04-20

所有走 `core_engine._call_via_cli()` / `_call_via_api()` 的 LLM 调用，必须遵守以下三条铁律。

---

## 1. CLI 优先，失败回退 API

### 流程

```
┌─────────────────────────┐
│  request(provider, ...) │
└───────────┬─────────────┘
            ▼
     use_cli = true ?
        ├─ no ──► 直接走 API 模式
        └─ yes ──► 进入 per-provider CLI 队列
                    │
                    ▼
           CLI 子进程（限 120s）
              ├─ 成功 ──► 返回结果
              └─ 失败 / 超时 ──► kill + 解锁 + **API 重试同一 request**
```

### 判定「CLI 失败」的触发
- 子进程 `return code != 0`
- 输出无法解析为 JSON
- 子进程执行时间超过 **120 秒**
- CLI bin 不可用 / 不存在

### 代码落点
- 调用方（`quant_llm.py` / `stock_report.py` / `llm_evaluator.py` / `news_scorer.py`）— CLI 失败后回退 `_call_via_api()`
- `core_engine.py::_call_via_cli()` — CLI 执行器（需要加 timeout + fallback）
- fallback 在调用方一次完成，保证 **单次 request 对上层语义只成功/失败一次**

---

## 2. 同 provider 的 CLI 调用串行（禁并发）

### 规则
- 以 `provider_name`（已通过 `_normalize_provider` 归一化）为 key 做互斥
- 同一 provider 同时只允许 **一个 CLI 子进程**在执行
- 后到的请求必须排队，等前一个完成或超时后才能启动自己的子进程
- 不同 provider 之间互相独立，可并行

### 为什么
- CLI 工具（Claude CLI / Codex / Gemini CLI）共享本地凭证 / 会话文件 / token 配额
- 并发子进程会互相破坏会话状态、随机失败、吃光账号额度
- 实际观测到同 provider 并发造成的随机 `429 / stale session / config write race`

### 实现约束
- 用 `threading.Lock` / `asyncio.Lock` **按 provider** 建互斥（不是全局锁）
- 建议封装为 `_CliMutexRegistry` 单例；key 为 provider，value 为 `Lock`
- 锁的获取/释放必须放 `try/finally`，子进程超时 kill 之后也要释放
- 不要在持锁时做 HTTP / 数据库调用，只包住 `subprocess.run()`

### 反例（禁止）
```python
# ❌ 全局 ThreadPoolExecutor(max_workers=8) 并行跑多个同 provider CLI
# ❌ asyncio.gather(*[cli_call("claude", ...) for _ in range(10)])
# ❌ 在 FastAPI endpoint 里 async def 然后 await 多个 CLI（默认线程池并发）
```

---

## 3. CLI 超时 120 秒 → 切 API

### 规则
- `subprocess` 设置 `timeout=120`
- 超时触发 `TimeoutExpired` 异常
- 异常处理必须：
  1. `process.kill()` + `process.wait(timeout=3)` 确保子进程死透（不留僵尸）
  2. 释放该 provider 的 CLI 锁（`finally` 块）
  3. 打 log：`logger.warning("CLI timeout provider=%s model=%s elapsed=%.1fs → fallback API", provider, model, elapsed)`
  4. 调 API 模式重试同一 prompt（system_prompt + user_prompt 不变）
  5. 返回的结果结构和 CLI 成功时一致，`metadata.fallback_reason = "cli_timeout"` 作 trace

### 120 秒的选择
- 经验值：Claude CLI 首次执行含 `--ephemeral` warmup 约 15-40s，长 prompt 偶尔 60-90s
- 120s 足够覆盖健康情况，又不会让失败 CLI 卡死队列超过 2 分钟
- 不要调高到 300s+，会拖垮整个回测（N 个超时 × 300s = 灾难）

### 不要吞超时
```python
# ❌ 禁止
try:
    run_cli(...)
except Exception:
    return api_call(...)  # 无 log，无标记，无法排查

# ✅ 正确
try:
    return run_cli(...)
except subprocess.TimeoutExpired:
    logger.warning("CLI timeout ... fallback API")
    return api_call(..., fallback_reason="cli_timeout")
except Exception as e:
    logger.warning("CLI error ... fallback API", exc_info=True)
    return api_call(..., fallback_reason=f"cli_err:{type(e).__name__}")
```

---

## 适用场景

- 策略评估 `/strategies/{name}/llm-analyze`
- 量化池 LLM veto、个股研报、新闻打分
- 任何通过 `_call_via_cli` / `_call_via_api` 的调用

---

## 验证清单（PR 必答）

改动涉及 `_call_via_cli` / `_call_via_api` 时，至少回答：

- [ ] CLI 调用是否走 `_CliMutexRegistry.get(provider)` 取锁？
- [ ] `subprocess.run` / `subprocess.Popen + communicate` 是否设置了 `timeout=120`？
- [ ] `TimeoutExpired` 是否 kill + finally 解锁 + log + API fallback？
- [ ] API fallback 后返回体是否带 `fallback_reason`？
- [ ] 对同 provider 发 10 个并发请求，是否只观察到 1 个 CLI 子进程在运行（其他排队）？
