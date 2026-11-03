"""
LLM Strategy Evaluator — 调用 LLM 对策略评估指标做结构化分析
复用 core_engine.py 的 LLM 基础设施，但覆盖 max_tokens 以支持长分析
"""
import json
import logging
import os
import re
import time
import hashlib
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger("uvicorn.error")

# ── 从 core_engine.py 复用基础设施 ──
from core_engine import (
    _provider_key,
    _provider_base_url,
    _extract_json_candidate,
    _call_via_cli,
    _provider_saved_config,
    LLM_API_CONFIG,
)

# ── 常量 ──
EVAL_TIMEOUT_SEC = 60  # 策略评估需要更长超时（vs 交易决策的 25s）
EVAL_MAX_TOKENS = 1500  # 策略分析需要更长输出（vs 交易决策的 200）
EVAL_TEMPERATURE = 0.3

VALID_VERDICTS = {"ACTIVATE", "ARCHIVE", "WATCH"}

# ── Prompt 模板 ──

EVAL_SYSTEM_PROMPT = """You are a senior quantitative strategy analyst. Evaluate trading strategies based on their backtest performance metrics.

Your evaluation must be objective, data-driven, and actionable. You will receive structured metrics for a single strategy and must return a JSON verdict.

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation outside the JSON."""

EVAL_USER_PROMPT_TEMPLATE = """Strategy: {strategy_name}
Signal Type: {signal_type}
Current Status: {status} {favourite_tag}
Evaluation Pool: {pool_name} ({stock_count} symbols)
Period: {interval}

=== PERFORMANCE METRICS ===
Composite Score: {score:.4f} (trimmed: {trimmed_score:.4f})
Rank: #{rank} of {total_strategies}
Average Return: {avg_return:.1f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_drawdown:.1f}%
Win Rate: {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}
Total Trades (avg): {avg_trades:.0f}
Winning Stocks: {positive_pct:.0f}% ({positive_stocks}/{stock_count})

=== EVALUATION CRITERIA ===
ACTIVATE: Score top 30%, Return > 10%, Sharpe > 0.8, MaxDD < 30%, WinRate > 45%
WATCH: Shows potential but fails 1-2 criteria, or inconsistent across stocks
ARCHIVE: Bottom 30% score, Return < 0%, Sharpe < 0.3, or MaxDD > 40%

Output strictly as JSON (no markdown):
{{"verdict": "ACTIVATE|ARCHIVE|WATCH", "confidence": 0-100, "overall_score": 0-100, "reasoning": "2-3 sentence analysis", "risk_warnings": ["warning1", "..."], "suggestions": ["suggestion1", "..."]}}"""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════
# LLM 调用封装（覆盖 max_tokens / timeout）
# ═══════════════════════════════════════════════════════════

def _default_model(provider: str) -> str:
    """默认模型名统一从 core_engine.LLM_PROVIDER_DEFAULTS 取, 本文件不再自带字面量。

    延迟 import: core_engine 很重, 且它反过来 import opinion_cache 等模块,
    放模块级容易成环。
    """
    try:
        from core_engine import default_model_for
        return default_model_for(provider) or ""
    except Exception:
        return ""


def _call_llm_for_eval(provider: str, model: str, system_prompt: str, user_prompt: str) -> dict:
    """调用 LLM 进行策略评估。返回 {parsed, raw_text, token_usage, duration_ms}
    优先使用 CLI 模式（如果已配置），否则使用 API Key 直接调用。"""
    t0 = time.time()

    # ── 优先检查 CLI 模式 ──
    saved = _provider_saved_config(provider)
    if saved.get("use_cli", False):
        logger.info(f"[LLM-Eval] 使用 CLI 模式: {provider}/{model}")
        try:
            parsed, raw_text, token_usage = _call_via_cli(
                provider, model, system_prompt, user_prompt, max_tokens=EVAL_MAX_TOKENS
            )
            duration_ms = int((time.time() - t0) * 1000)
            return {
                "parsed": parsed,
                "raw_text": raw_text,
                "token_usage": token_usage if isinstance(token_usage, dict) else {},
                "duration_ms": duration_ms,
                "prompt_hash": _hash_text(user_prompt),
                "response_hash": _hash_text(raw_text or ""),
            }
        except Exception as e:
            logger.warning(f"[LLM-Eval] CLI 模式失败: {e}, 尝试 API 模式")
            # CLI 失败后 fallthrough 到 API 模式

    key = _provider_key(provider)
    if not key:
        raise RuntimeError(f"missing_api_key:{provider}")

    provider_lower = provider.lower()

    if provider_lower == "claude":
        resp = requests.post(
            _provider_base_url("claude"),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": EVAL_MAX_TOKENS,
                "temperature": EVAL_TEMPERATURE,
                "system": [{"type": "text", "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=EVAL_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("content") or []
        text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        usage = data.get("usage") or {}
        token_usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }

    elif provider_lower == "gemini":
        base = _provider_base_url("gemini")
        url = f"{base}/{model}:generateContent?key={key}"
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": EVAL_TEMPERATURE,
                    "maxOutputTokens": EVAL_MAX_TOKENS,
                    "responseMimeType": "application/json",
                },
            },
            timeout=EVAL_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join(p.get("text", "") for p in parts).strip()
        um = data.get("usageMetadata") or {}
        token_usage = {
            "input_tokens": um.get("promptTokenCount", 0),
            "output_tokens": um.get("candidatesTokenCount", 0),
            "total_tokens": um.get("totalTokenCount", 0),
        }

    else:
        # OpenAI-compatible (OpenAI, Grok, DeepSeek, etc.)
        config = LLM_API_CONFIG.get(provider_lower, {})
        base = _provider_base_url(provider_lower)
        chat_path = config.get("chat_path", "/chat/completions")
        url = f"{base}{chat_path}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "max_tokens": EVAL_MAX_TOKENS,
                "temperature": EVAL_TEMPERATURE,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=EVAL_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        um = data.get("usage") or {}
        token_usage = {
            "input_tokens": um.get("prompt_tokens", 0),
            "output_tokens": um.get("completion_tokens", 0),
            "total_tokens": um.get("total_tokens", 0),
        }

    duration_ms = int((time.time() - t0) * 1000)
    parsed = _extract_json_candidate(text)
    return {
        "parsed": parsed,
        "raw_text": text,
        "token_usage": token_usage,
        "duration_ms": duration_ms,
        "prompt_hash": _hash_text(user_prompt),
        "response_hash": _hash_text(text),
    }


# ═══════════════════════════════════════════════════════════
# 结构化 verdict 校验
# ═══════════════════════════════════════════════════════════

def _normalize_eval_verdict(parsed: Optional[dict], is_favourite: bool) -> dict:
    """校验 + 标准化 LLM 返回的 verdict。Favourite 策略不可被 ARCHIVE。"""
    default = {
        "verdict": "WATCH",
        "confidence": 50,
        "overall_score": 50,
        "reasoning": "LLM response could not be parsed",
        "risk_warnings": [],
        "suggestions": [],
    }
    if not isinstance(parsed, dict):
        return default

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        verdict = "WATCH"

    # ── Favourite 保护：永远不被 ARCHIVE ──
    if is_favourite and verdict == "ARCHIVE":
        verdict = "WATCH"

    confidence = max(0, min(100, int(parsed.get("confidence", 50))))
    overall_score = max(0, min(100, int(parsed.get("overall_score", 50))))
    reasoning = str(parsed.get("reasoning", ""))[:500]
    warnings = parsed.get("risk_warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    suggestions = parsed.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "overall_score": overall_score,
        "reasoning": reasoning,
        "risk_warnings": warnings[:10],
        "suggestions": suggestions[:10],
    }


# ═══════════════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════════════

def evaluate_strategy(
    strategy_name: str,
    metrics: dict,
    warehouse_status: str = "active",
    is_favourite: bool = False,
    pool_name: str = "",
    stock_count: int = 0,
    interval: str = "1d",
    rank: int = 0,
    total_strategies: int = 0,
    signal_type: str = "combined",
    provider: str = "claude",
    model: Optional[str] = None,      # None = 按 provider 取默认, 见 _default_model()
) -> dict:
    """对单个策略调用 LLM 分析，返回结构化评估结果。

    Parameters
    ----------
    metrics : dict — 来自 /strategies/evaluate 的排名数据，包含:
        score, trimmed_score, avg_return, sharpe, max_drawdown,
        win_rate, profit_factor, avg_trades, positive_pct, positive_stocks
    """
    model = model or _default_model(provider)
    user_prompt = EVAL_USER_PROMPT_TEMPLATE.format(
        strategy_name=strategy_name,
        signal_type=signal_type,
        status=warehouse_status,
        favourite_tag="⭐ (favourite — cannot be archived)" if is_favourite else "",
        pool_name=pool_name,
        stock_count=stock_count,
        interval=interval,
        score=metrics.get("score", 0),
        trimmed_score=metrics.get("trimmed_score", 0),
        rank=rank,
        total_strategies=total_strategies,
        avg_return=metrics.get("avg_return", 0),
        sharpe=metrics.get("sharpe", 0),
        max_drawdown=metrics.get("max_drawdown", 0),
        win_rate=metrics.get("win_rate", 0),
        profit_factor=metrics.get("profit_factor", 0),
        avg_trades=metrics.get("avg_trades", 0),
        positive_pct=metrics.get("positive_pct", 0),
        positive_stocks=metrics.get("positive_stocks", 0),
    )

    try:
        llm_result = _call_llm_for_eval(provider, model, EVAL_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error(f"[LLM-Eval] {strategy_name} 调用失败: {e}")
        return {
            "strategy_name": strategy_name,
            "verdict": "WATCH",
            "confidence": 0,
            "overall_score": 0,
            "reasoning": f"LLM call failed: {str(e)[:200]}",
            "risk_warnings": ["LLM evaluation failed"],
            "suggestions": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "duration_ms": 0,
            "prompt_hash": _hash_text(user_prompt),
            "response_hash": "",
            "raw_response": "",
            "error": str(e)[:200],
        }

    verdict_data = _normalize_eval_verdict(llm_result["parsed"], is_favourite)

    return {
        "strategy_name": strategy_name,
        **verdict_data,
        "token_usage": llm_result["token_usage"],
        "duration_ms": llm_result["duration_ms"],
        "prompt_hash": llm_result["prompt_hash"],
        "response_hash": llm_result["response_hash"],
        "raw_response": llm_result["raw_text"],
    }


def evaluate_batch(
    strategies: list,
    pool_name: str = "",
    stock_count: int = 0,
    interval: str = "1d",
    signal_type: str = "combined",
    provider: str = "claude",
    model: Optional[str] = None,      # None = 按 provider 取默认, 见 _default_model()
    max_concurrent: int = 4,
    on_progress=None,
) -> dict:
    """批量评估多个策略。

    Parameters
    ----------
    strategies : list of dict — 每个元素包含:
        name, metrics, status, is_favourite, rank
    on_progress : callable(done, total, strategy_name, verdict) — 进度回调
    """
    model = model or _default_model(provider)
    total = len(strategies)
    reviews = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_duration = 0

    def _eval_one(item):
        return evaluate_strategy(
            strategy_name=item["name"],
            metrics=item.get("metrics", {}),
            warehouse_status=item.get("status", "active"),
            is_favourite=item.get("is_favourite", False),
            pool_name=pool_name,
            stock_count=stock_count,
            interval=interval,
            rank=item.get("rank", 0),
            total_strategies=total,
            signal_type=signal_type,
            provider=provider,
            model=model,
        )

    # ThreadPoolExecutor 控制 LLM API 并发（避免 rate limit）
    done = 0
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_map = {pool.submit(_eval_one, s): s for s in strategies}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                result = future.result(timeout=EVAL_TIMEOUT_SEC + 10)
            except Exception as e:
                result = {
                    "strategy_name": item["name"],
                    "verdict": "WATCH",
                    "confidence": 0,
                    "overall_score": 0,
                    "reasoning": f"Evaluation timeout/error: {str(e)[:100]}",
                    "risk_warnings": ["Evaluation failed"],
                    "suggestions": [],
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "duration_ms": 0,
                    "prompt_hash": "",
                    "response_hash": "",
                    "raw_response": "",
                    "error": str(e)[:200],
                }
            reviews.append(result)
            tu = result.get("token_usage", {})
            total_tokens["input_tokens"] += tu.get("input_tokens", 0)
            total_tokens["output_tokens"] += tu.get("output_tokens", 0)
            total_tokens["total_tokens"] += tu.get("total_tokens", 0)
            total_duration += result.get("duration_ms", 0)
            done += 1
            if on_progress:
                try:
                    on_progress(done, total, result["strategy_name"], result["verdict"])
                except Exception:
                    pass

    # 按 overall_score 降序排
    reviews.sort(key=lambda r: r.get("overall_score", 0), reverse=True)

    activate_count = sum(1 for r in reviews if r["verdict"] == "ACTIVATE")
    archive_count = sum(1 for r in reviews if r["verdict"] == "ARCHIVE")
    watch_count = sum(1 for r in reviews if r["verdict"] == "WATCH")

    return {
        "reviews": reviews,
        "summary": {
            "total": total,
            "activate_count": activate_count,
            "archive_count": archive_count,
            "watch_count": watch_count,
            "avg_confidence": round(sum(r["confidence"] for r in reviews) / max(1, len(reviews)), 1),
            "avg_score": round(sum(r["overall_score"] for r in reviews) / max(1, len(reviews)), 1),
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration,
        },
    }


# ═══════════════════════════════════════════════════════════
# Markdown 报告生成
# ═══════════════════════════════════════════════════════════

def generate_report(reviews: list, summary: dict, pool_name: str = "",
                    round_id: str = "", provider: str = "", model: str = "") -> str:
    """生成 Markdown 格式的评估报告。"""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# LLM Strategy Evaluation Report",
        f"",
        f"**Round:** {round_id}  |  **Pool:** {pool_name}  |  **Date:** {now}",
        f"**Provider:** {provider}  |  **Model:** {model}",
        f"",
        f"## Executive Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Strategies | {summary['total']} |",
        f"| ✅ ACTIVATE | {summary['activate_count']} |",
        f"| ❌ ARCHIVE | {summary['archive_count']} |",
        f"| ⏳ WATCH | {summary['watch_count']} |",
        f"| Avg Confidence | {summary['avg_confidence']}% |",
        f"| Avg Score | {summary['avg_score']} |",
        f"| Total Tokens | {summary['total_tokens'].get('total_tokens', 0):,} |",
        f"",
    ]

    # ── ACTIVATE 策略 ──
    activate = [r for r in reviews if r["verdict"] == "ACTIVATE"]
    if activate:
        lines.append("## ✅ ACTIVATE Strategies")
        lines.append("")
        lines.append("| # | Strategy | Score | Confidence | Reasoning |")
        lines.append("|---|----------|-------|------------|-----------|")
        for i, r in enumerate(activate, 1):
            lines.append(f"| {i} | {r['strategy_name']} | {r['overall_score']} | {r['confidence']}% | {r['reasoning'][:80]} |")
        lines.append("")

    # ── ARCHIVE 策略 ──
    archive = [r for r in reviews if r["verdict"] == "ARCHIVE"]
    if archive:
        lines.append("## ❌ ARCHIVE Strategies")
        lines.append("")
        lines.append("| # | Strategy | Score | Confidence | Reasoning |")
        lines.append("|---|----------|-------|------------|-----------|")
        for i, r in enumerate(archive, 1):
            lines.append(f"| {i} | {r['strategy_name']} | {r['overall_score']} | {r['confidence']}% | {r['reasoning'][:80]} |")
        lines.append("")

    # ── WATCH 策略 ──
    watch = [r for r in reviews if r["verdict"] == "WATCH"]
    if watch:
        lines.append("## ⏳ WATCH Strategies")
        lines.append("")
        lines.append("| # | Strategy | Score | Confidence | Reasoning |")
        lines.append("|---|----------|-------|------------|-----------|")
        for i, r in enumerate(watch, 1):
            lines.append(f"| {i} | {r['strategy_name']} | {r['overall_score']} | {r['confidence']}% | {r['reasoning'][:80]} |")
        lines.append("")

    # ── Risk Warnings ──
    all_warnings = []
    for r in reviews:
        for w in (r.get("risk_warnings") or []):
            all_warnings.append(f"- **{r['strategy_name']}**: {w}")
    if all_warnings:
        lines.append("## ⚠️ Risk Warnings")
        lines.append("")
        lines.extend(all_warnings[:20])
        lines.append("")

    lines.append(f"---")
    lines.append(f"*Generated by AITrader LLM Eval Chain | {now}*")

    return "\n".join(lines)
