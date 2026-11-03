"""
Quant LLM veto layer (复用 core_engine 的 provider 基础设施).

- provider 配置 (api_key / use_cli / model_name) 只读 core_engine 那一份
  (.cache/llm_keys.json, 经 _LLM_CONFIG_LOCK), 本模块不再另存/另写.
  编辑入口统一在 系统 · LLM 管理 tab.
- 调用规则同 docs/llm_call_rules.md: CLI 优先 (同 provider 串行, 走 _CLI_MUTEX),
  CLI 未启用/失败 → HTTP API (需 api_key). 两条路都没有 → ProviderNotConfigured.
- veto 决策: 给 LLM 一个精简的 payload → 期望回 EXECUTE/SKIP + confidence 0-1
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ["claude", "openai", "gemini"]
PROMPT_FILE = Path(__file__).parent / ".cache" / "quant_llm_prompt.json"
# veto 是 tick 内同步调用, CLI 互斥锁等太久就直接走 API / fail-open
CLI_MUTEX_WAIT_SEC = float(os.getenv("QUANT_CLI_MUTEX_WAIT_SEC", "20"))


class ProviderNotConfigured(RuntimeError):
    """provider 既没 API key 也没启用 CLI."""


# ─── Provider 配置 (只读 core_engine 的存储) ──────────────────────

def provider_status() -> dict:
    """各 provider 可用状态 (脱敏, 只读). {provider: {configured, use_cli, has_api_key, model}}"""
    try:
        import core_engine as _me
        cfgs = _me._read_llm_provider_configs()
    except Exception as e:
        logger.debug(f"read llm provider configs failed: {e}")
        cfgs, _me = {}, None
    out = {}
    for p in SUPPORTED_PROVIDERS:
        cfg = cfgs.get(p) or {}
        api_key = str(cfg.get("api_key") or "").strip()
        use_cli = bool(cfg.get("use_cli", False))
        model = ""
        if _me is not None:
            try:
                model = _me._effective_model_name(p, "")
            except Exception:
                model = str(cfg.get("model_name") or "")
        out[p] = {
            "configured": bool(api_key) or use_cli,
            "use_cli": use_cli,
            "has_api_key": bool(api_key),
            "model": model,
        }
    return out


def call_provider(provider: str, system_prompt: str, user_prompt: str,
                  max_tokens: int = 300) -> tuple[Optional[dict], str]:
    """
    统一的 LLM 调用入口 (quant veto / stock_report 共用).
    CLI 优先 (拿 core_engine._CLI_MUTEX 同 provider 串行), 失败回退 API.
    Returns (parsed_dict_or_None, raw_text).
    Raises ProviderNotConfigured / RuntimeError.
    """
    provider = str(provider or "").lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderNotConfigured(f"unsupported provider: {provider}")
    import core_engine as _me
    st = provider_status().get(provider) or {}
    if not st.get("configured"):
        raise ProviderNotConfigured(
            f"provider '{provider}' 未配 API key, 也未启用 CLI. "
            f"请到 系统 · LLM 管理 tab 配置 (勾 CLI 或填 API key).")
    model = st.get("model") or _me._effective_model_name(provider, "")
    last_err = None
    if st.get("use_cli"):
        lock = _me._CLI_MUTEX.get(provider)
        if lock.acquire(timeout=CLI_MUTEX_WAIT_SEC):
            try:
                parsed, raw, _ = _me._call_via_cli(provider, model, system_prompt,
                                                    user_prompt, max_tokens=max_tokens)
                if parsed is not None or raw:
                    return (parsed if isinstance(parsed, dict) else None), (raw or "")
                last_err = "CLI 返回空"
            except Exception as e:
                last_err = f"CLI 调用: {e}"
                logger.warning(f"CLI call {provider} failed: {e}")
            finally:
                lock.release()
        else:
            last_err = f"CLI 互斥锁等待 {CLI_MUTEX_WAIT_SEC:.0f}s 超时"
    if st.get("has_api_key"):
        try:
            parsed, raw, _ = _me._call_via_api(provider, model, system_prompt, user_prompt)
            if parsed is not None or raw:
                return (parsed if isinstance(parsed, dict) else None), (raw or "")
            last_err = last_err or "API 返回空"
        except Exception as e:
            last_err = last_err or f"API 调用: {e}"
            logger.error(f"API call {provider} failed: {e}")
    raise RuntimeError(f"{provider} 调用失败: {last_err or '未知'}")


# ─── Veto decision ───────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = """You are a quant trading risk validator. You will receive a proposed
BUY / SELL / SHORT / COVER decision and must decide whether to allow it.

Return ONLY a JSON object, no prose:
{
  "verdict": "EXECUTE" | "SKIP",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence",
  "suggested_size_pct": 0.5 to 50.0    // OPTIONAL: your recommended % of current cash for this position; omit if you don't want to change sizing
}

Guidelines:
- If price/quantity looks reasonable given the signal → EXECUTE with high confidence.
- If signal contradicts obvious price action (e.g. BUY on -20% gap down) → SKIP.
- If confidence < 0.5, prefer SKIP.
- Keep reason under 20 words.
- suggested_size_pct: only include for BUY/SHORT (new position opening). Sensible range 2-15. Higher = more conviction. Omit entirely for SELL/COVER or when you have no view."""


def get_system_prompt() -> str:
    """读当前 system prompt (用户覆盖优先, fallback default)."""
    if PROMPT_FILE.exists():
        try:
            data = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
            text = (data.get("system_prompt") or "").strip()
            if text:
                return text
        except Exception:
            pass
    return _DEFAULT_SYSTEM_PROMPT


def save_system_prompt(text: str) -> dict:
    """持久化用户覆盖的 system prompt. 空串/None 恢复默认."""
    PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    text = (text or "").strip()
    if not text:
        # 删掉文件恢复默认
        if PROMPT_FILE.exists():
            PROMPT_FILE.unlink()
        return {"is_default": True, "system_prompt": _DEFAULT_SYSTEM_PROMPT}
    payload = {"system_prompt": text}
    tmp = PROMPT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PROMPT_FILE)
    return {"is_default": False, "system_prompt": text}


def get_default_system_prompt() -> str:
    return _DEFAULT_SYSTEM_PROMPT


def _pool_state_block(pool: dict) -> str:
    """池当前状态摘要 (cash / 持仓 / 未实现盈亏 / 回撤)."""
    lines = ["Pool state:"]
    lines.append(f"- Name: {pool.get('name', '?')}")
    lines.append(f"- Initial capital: ${float(pool.get('initial_capital', 0)):.0f}")
    lines.append(f"- Current cash: ${float(pool.get('cash', 0)):.2f}")
    positions = [p for p in (pool.get('positions') or []) if p.get('status') == 'open']
    if positions:
        long_n = sum(1 for p in positions if p.get('side') == 'LONG')
        short_n = sum(1 for p in positions if p.get('side') == 'SHORT')
        lines.append(f"- Open positions: {len(positions)} ({long_n} LONG + {short_n} SHORT)")
    else:
        lines.append("- Open positions: 0")
    eq_hist = pool.get('equity_history') or []
    if eq_hist:
        last = eq_hist[-1]
        eq = last.get('equity') or 0
        peak = max(e.get('equity', 0) for e in eq_hist)
        dd = ((eq - peak) / peak * 100) if peak else 0
        ret = last.get('return_pct')
        if ret is not None:
            lines.append(f"- Return from init: {ret:+.2f}%")
        lines.append(f"- Drawdown from peak: {dd:+.2f}%")
    return "\n".join(lines)


def _kline_summary(symbol: str, market: str, interval: str, n: int = 10) -> str:
    """最近 n 根 K 线 + 简单指标, 用来给 LLM 一点价格上下文."""
    try:
        import backtest as _bt
        df = _bt.get_ohlcv(symbol, market, interval=interval,
                           period="3mo", local_only=True)
    except Exception:
        return ""
    if df is None or df.empty:
        return ""
    # backtest.get_ohlcv 返回小写列 (open/high/low/close/volume); 兼容大写以防万一
    cols = {str(c).lower(): c for c in df.columns}

    def _col(name: str):
        return cols.get(name)

    if not all(_col(c) for c in ("open", "high", "low", "close")):
        return ""
    tail = df.tail(n)
    lines = [f"Recent {len(tail)} {interval} bars for {symbol} (O,H,L,C,V):"]
    for idx, row in tail.iterrows():
        try:
            ts = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            o = float(row[_col("open")]); h = float(row[_col("high")])
            lo = float(row[_col("low")]); c = float(row[_col("close")])
            v = float(row[_col("volume")]) if _col("volume") else 0.0
            v_str = f"{v/1e6:.1f}M" if v > 1e6 else f"{v:.0f}"
            lines.append(f"  {ts}: {o:.2f}, {h:.2f}, {lo:.2f}, {c:.2f}, {v_str}")
        except Exception:
            continue
    # 简单指标: 最近 n 根收益 + 20 根 hi/lo
    try:
        first_close = float(tail.iloc[0][_col("close")])
        last_close = float(tail.iloc[-1][_col("close")])
        ret_pct = (last_close - first_close) / first_close * 100
        lines.append(f"Return over last {len(tail)} bars: {ret_pct:+.2f}%")
    except Exception:
        pass
    try:
        recent = df.tail(20)
        hi20 = float(recent[_col("high")].max())
        lo20 = float(recent[_col("low")].min())
        lines.append(f"20-bar range: ${lo20:.2f} - ${hi20:.2f}")
    except Exception:
        pass
    return "\n".join(lines)


def _build_user_prompt(decision: dict, pool: dict, market_snapshot: Optional[str] = None) -> str:
    action = decision.get("action", "")
    symbol = decision.get("symbol", "")
    qty = decision.get("quantity", 0)
    price = decision.get("price", 0)
    try:
        notional = float(qty) * float(price)
    except Exception:
        notional = 0
    market = pool.get("market", "US")
    interval = pool.get("interval", "1d")

    blocks = [
        _pool_state_block(pool),
        "",
        "Proposed decision:",
        f"- Action: {action}",
        f"- Symbol: {symbol}",
        f"- Quantity: {qty}",
        f"- Price: ${price}",
        f"- Notional: ${notional:.2f}",
        f"- Strategy triggered: {decision.get('strategy', '?')}",
        f"- Existing position on {symbol}: {decision.get('existing_position') or 'none'}",
    ]
    kline = _kline_summary(symbol, market, interval, n=10)
    if kline:
        blocks.append("")
        blocks.append(kline)
    if market_snapshot:
        blocks.append("")
        blocks.append(f"Recent market: {market_snapshot}")
    blocks.append("")
    blocks.append("Respond with the JSON verdict only.")
    return "\n".join(blocks)


def _call_one_provider(provider: str, decision: dict, pool: dict,
                       market_snapshot: Optional[str], min_conf: float) -> dict:
    """单个 provider 调用 (CLI 优先 → API). 全部内部错误路径都返回 EXECUTE (fail-open)."""
    if provider not in SUPPORTED_PROVIDERS:
        return {"verdict": "EXECUTE", "confidence": 1.0,
                "reason": f"unknown provider {provider}", "provider": provider,
                "ok": False}
    system_prompt = get_system_prompt()
    user_prompt = _build_user_prompt(decision, pool, market_snapshot)
    try:
        parsed, raw = call_provider(provider, system_prompt, user_prompt, max_tokens=300)
    except ProviderNotConfigured as e:
        return {"verdict": "EXECUTE", "confidence": 1.0,
                "reason": f"not configured: {str(e)[:80]}", "provider": provider,
                "ok": False}
    except Exception as e:
        logger.warning(f"llm veto call failed {provider}: {e}")
        return {"verdict": "EXECUTE", "confidence": 1.0,
                "reason": f"llm call failed: {str(e)[:60]}", "provider": provider,
                "ok": False}
    raw_text = (raw or "").strip()
    verdict = None
    confidence = 0.0
    reason = ""
    suggested_size_pct = None
    try:
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).upper()
            confidence = float(parsed.get("confidence", 0))
            reason = str(parsed.get("reason", ""))
            if "suggested_size_pct" in parsed and parsed["suggested_size_pct"] is not None:
                try:
                    suggested_size_pct = float(parsed["suggested_size_pct"])
                except (TypeError, ValueError):
                    pass
        if not verdict and raw_text:
            m = re.search(r'"verdict"\s*:\s*"(EXECUTE|SKIP)"', raw_text, re.IGNORECASE)
            if m:
                verdict = m.group(1).upper()
            m2 = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_text)
            if m2:
                confidence = float(m2.group(1))
            m3 = re.search(r'"reason"\s*:\s*"([^"]{1,200})"', raw_text)
            if m3:
                reason = m3.group(1)
            if suggested_size_pct is None:
                m4 = re.search(r'"suggested_size_pct"\s*:\s*([0-9.]+)', raw_text)
                if m4:
                    try:
                        suggested_size_pct = float(m4.group(1))
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"llm veto parse failed: {e}, raw={raw_text[:200]}")
    if verdict not in ("EXECUTE", "SKIP"):
        return {"verdict": "EXECUTE", "confidence": 1.0,
                "reason": "llm output unparseable, fail-open", "provider": provider,
                "raw_snippet": raw_text[:120], "ok": False,
                "suggested_size_pct": None}
    # 单个 provider 自己的置信度阈值判定
    if verdict == "EXECUTE" and confidence < min_conf:
        return {"verdict": "SKIP", "confidence": confidence,
                "reason": f"confidence {confidence:.2f} < min {min_conf}", "provider": provider,
                "llm_reason": reason, "ok": True, "suggested_size_pct": suggested_size_pct}
    return {"verdict": verdict, "confidence": confidence,
            "reason": reason or "ok", "provider": provider, "ok": True,
            "suggested_size_pct": suggested_size_pct}


def veto_decision(decision: dict, pool: dict,
                  market_snapshot: Optional[str] = None) -> dict:
    """
    多 LLM 联合裁决:
    - providers = pool.llm_veto.providers[]  (或旧字段 provider: str)
    - aggregate = pool.llm_veto.aggregate    ("all_agree" | "majority", 默认 all_agree)
    - all_agree: 所有 provider 都 EXECUTE 才 EXECUTE, 任一 SKIP → SKIP
    - majority : 过半 EXECUTE 才 EXECUTE
    Returns: {verdict, confidence: 均值, reason: 汇总, providers: 各家明细, provider: 兼容字段}
    异常/未配 key: 单家 fail-open EXECUTE, 聚合逻辑照跑.
    """
    veto_cfg = pool.get("llm_veto") or {}
    # 启用与否由 providers 列表决定 (UI 里勾选 = 启用). 兼容旧字段:
    #   {enabled: true, provider: "claude"} → ["claude"]; enabled=false 且无 providers → 关
    providers = veto_cfg.get("providers")
    if isinstance(providers, list):
        providers = [str(p).lower().strip() for p in providers if p]
    else:
        providers = []
    if not providers and veto_cfg.get("enabled") and veto_cfg.get("provider"):
        providers = [str(veto_cfg.get("provider")).lower().strip()]
    if not providers:
        return {"verdict": "EXECUTE", "confidence": 1.0, "reason": "llm_veto disabled (no providers)",
                "provider": None, "providers": {}}

    min_conf = float(veto_cfg.get("min_confidence", 0.7))
    aggregate = str(veto_cfg.get("aggregate", "all_agree")).lower()

    details = {}
    for p in providers:
        details[p] = _call_one_provider(p, decision, pool, market_snapshot, min_conf)

    # 聚合
    verdicts = [d["verdict"] for d in details.values()]
    execs = sum(1 for v in verdicts if v == "EXECUTE")
    skips = sum(1 for v in verdicts if v == "SKIP")
    if aggregate == "majority":
        final = "EXECUTE" if execs > skips else "SKIP"
    else:  # all_agree
        final = "EXECUTE" if skips == 0 else "SKIP"
    avg_conf = sum(d.get("confidence", 0) for d in details.values()) / max(1, len(details))
    reasons = "; ".join(f"{p}:{d['verdict']}({d.get('confidence', 0):.2f})" for p, d in details.items())
    # 聚合 suggested_size_pct: 只看投 EXECUTE 且给了数字的那些, 取平均
    exec_sizes = [d["suggested_size_pct"] for d in details.values()
                  if d.get("verdict") == "EXECUTE" and d.get("suggested_size_pct") is not None
                  and 0 < float(d["suggested_size_pct"]) <= 100]
    agg_size = round(sum(exec_sizes) / len(exec_sizes), 2) if exec_sizes else None
    return {"verdict": final, "confidence": avg_conf,
            "reason": f"[{aggregate}] {reasons}",
            "provider": providers[0],  # 兼容字段: 第一家
            "providers": details,
            "suggested_size_pct": agg_size}
