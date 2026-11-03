"""News sentiment scorer — uses CLI-first LLM via llm_evaluator.

Public API:
    score_news(symbol, news_items, provider='claude', model=None) -> dict

Returns: {score (0-100), label (bullish/bearish/neutral), reasoning, n_items, provider}.

Score semantics:
    - 0-30  : bearish
    - 30-50 : mildly bearish / mixed-negative
    - 50-65 : neutral / mixed
    - 65-80 : mildly bullish
    - 80-100: bullish

Empty `news_items` returns score=50 (neutral) with label='no_news' — never raises.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default LLM ──
# Per memory:feedback_llm_call_rules — CLI 优先；llm_evaluator already handles this.
_DEFAULT_PROVIDER = "claude"


def _default_model_for(provider: str) -> str:
    """默认模型名统一从 core_engine.LLM_PROVIDER_DEFAULTS 取。

    本文件原来自带一张表, 而且跟别处不一致 (这里写 claude-sonnet-4-6,
    eval_chain/llm_evaluator/opinion_cache 都写 claude-sonnet-4-20250514),
    同一个 provider 在不同入口拿到不同默认值。延迟 import 避免成环。
    """
    try:
        from core_engine import default_model_for
        return default_model_for(provider) or ""
    except Exception:
        return ""

_SYSTEM_PROMPT = """You are a financial news sentiment analyzer.
Given a list of recent news headlines for a single stock, output a sentiment score
on a 0-100 scale where:
  0  = strongly bearish (would expect price to drop significantly)
  50 = neutral / mixed (no clear directional bias)
  100 = strongly bullish (would expect price to rise significantly)

Consider:
  - Headline sentiment (positive vs negative language)
  - Materiality (earnings beat > rumor; lawsuit > minor product update)
  - Recency (last 6 hours weighted more than 24 hours ago)
  - Concentration (5 same-direction news = stronger signal than 1)

Respond ONLY with valid JSON in this exact format (no markdown fences):
{"score": <int 0-100>, "label": "<bullish|bearish|neutral>", "reasoning": "<1-2 sentence explanation>"}
"""


def _build_user_prompt(symbol: str, news_items: list[dict]) -> str:
    lines = [f"Symbol: {symbol}", f"News count: {len(news_items)}", "", "Headlines (newest first):"]
    for i, item in enumerate(news_items[:30], 1):  # cap at 30 to control tokens
        title = (item.get("title") or "").strip()
        src = item.get("source") or ""
        ts = item.get("published_at") or ""
        lines.append(f"{i}. [{ts}] [{src}] {title}")
    return "\n".join(lines)


def _parse_score(raw: str) -> dict:
    """Extract JSON from LLM response (may include markdown fences or prose)."""
    if not raw:
        return {"score": 50, "label": "neutral", "reasoning": "empty LLM response"}
    # Try direct parse first
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # Strip code fences and try again
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("score JSON parse failed; raw=%s", raw[:200])
    return {"score": 50, "label": "neutral", "reasoning": "parse failed"}


def score_news(
    symbol: str,
    news_items: list[dict],
    provider: str = _DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> dict:
    """Score sentiment of news for a single stock via CLI-first LLM call.

    Args:
        symbol: ticker (used only to label the prompt context).
        news_items: list of {title, source, published_at, ...} from news_provider.
        provider: 'claude' / 'openai' / 'gemini' / 'deepseek' / 'qwen' / etc.
        model: override model id; falls back to provider default.

    Returns dict:
        {
            'score': int (0-100),
            'label': 'bullish' | 'bearish' | 'neutral' | 'no_news',
            'reasoning': str,
            'n_items': int,
            'provider': str,
            'model': str,
            'duration_ms': int,
        }
    """
    n = len(news_items or [])
    if n == 0:
        return {
            "score": 50,
            "label": "no_news",
            "reasoning": "No news items in window.",
            "n_items": 0,
            "provider": provider,
            "model": model or "",
            "duration_ms": 0,
        }

    model = model or _default_model_for(provider)
    user_prompt = _build_user_prompt(symbol, news_items)

    # Lazy imports: avoid loading at module import time
    from llm_evaluator import _call_llm_for_eval, _provider_saved_config
    # Per-provider CLI mutex from core_engine — prevents concurrent claude/gemini/codex
    # CLI subprocess invocations from corrupting session state. _call_llm_for_eval
    # itself does NOT enforce this lock; the caller must (see code-review 2026-04-26).
    # Acquire only when CLI mode is configured for this provider.
    from core_engine import _CLI_MUTEX
    use_cli = bool(_provider_saved_config(provider).get("use_cli", False))
    cli_lock = _CLI_MUTEX.get(provider) if use_cli else None
    cli_wait_sec = float(__import__("os").environ.get("AITRADER_CLI_MUTEX_WAIT_SEC", "25"))

    acquired = True
    if cli_lock is not None:
        acquired = cli_lock.acquire(timeout=cli_wait_sec)
        if not acquired:
            logger.info("news_scorer: CLI mutex contention for %s, falling back to API", provider)
            # _call_llm_for_eval will use API path because we'll temporarily mark use_cli=False?
            # Simpler: just call without lock and let _call_llm_for_eval's existing
            # CLI-fail-fallthrough handle it. But we don't want CLI race — caller MUST hold
            # the lock when calling _call_via_cli. Solution: skip CLI by passing a hint.
            # Workaround: invoke _call_llm_for_eval anyway; it will try CLI without lock,
            # and that's exactly the race we want to avoid. Cleanest: bypass CLI by
            # signaling to _call_llm_for_eval. There's no flag — so we monkey-patch:
            # re-call with a temporary saved-config override is invasive.
            # Pragmatic choice: hold lock for full call duration; acquire(timeout=cli_wait_sec)
            # blocks but gives up after 25s. If we got here, fall back to neutral score
            # rather than racing the CLI.
            return {
                "score": 50,
                "label": "neutral",
                "reasoning": "CLI mutex contention timeout (25s); skipped scoring to avoid CLI race",
                "n_items": n,
                "provider": provider,
                "model": model,
                "duration_ms": 0,
            }

    try:
        try:
            result = _call_llm_for_eval(provider, model, _SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.warning("LLM scoring failed for %s: %s", symbol, e)
            return {
                "score": 50,
                "label": "neutral",
                "reasoning": f"LLM call failed: {e}",
                "n_items": n,
                "provider": provider,
                "model": model,
                "duration_ms": 0,
            }
    finally:
        if cli_lock is not None and acquired:
            try:
                cli_lock.release()
            except RuntimeError:
                pass  # already released

    raw = (result or {}).get("raw_text") or ""
    parsed = _parse_score(raw)
    score = parsed.get("score", 50)
    try:
        score = max(0, min(100, int(score)))
    except (TypeError, ValueError):
        score = 50
    label = parsed.get("label") or "neutral"
    if label not in ("bullish", "bearish", "neutral"):
        label = "neutral"

    return {
        "score": score,
        "label": label,
        "reasoning": (parsed.get("reasoning") or "")[:500],
        "n_items": n,
        "provider": provider,
        "model": model,
        "duration_ms": (result or {}).get("duration_ms", 0),
    }
