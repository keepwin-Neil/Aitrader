"""News daily report generator (Phase 3).

Reads scan_signals.json — flat list of alerts persisted by the selector. Alerts
that came from the 9000「🎯 选股服务」tab with news mode enabled carry a
`news` payload (score / label / reasoning / top3). This generator aggregates
those into a daily markdown report.

Public API:
    generate_daily_report(market='ALL', window_hours=24, top_n=10) -> dict
        Returns {markdown, stats, generated_at}

Design:
- Pure aggregation, no LLM calls (data already scored upstream).
- Empty news → returns "no news-annotated alerts" notice (with how-to-fix tip).
- Sorted by news.score descending; ties broken by recency.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCAN_LOG = Path(__file__).parent / "scan_signals.json"


def _load_alerts() -> list[dict]:
    if not _SCAN_LOG.exists():
        return []
    try:
        return json.loads(_SCAN_LOG.read_text(encoding="utf-8")) or []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("scan_signals.json load failed: %s", e)
        return []


def _parse_dt(s: str) -> Optional[datetime]:
    """Parse alert timestamps. Returns naive datetime in local time.

    NOTE: scan_signals.json producer (main.py:_run_realtime_scan) uses
    datetime.now().isoformat() — naive local time. We strip timezone info to
    keep both sides consistent. If the producer is ever changed to UTC, also
    update this function (or normalize both to UTC) — otherwise window_hours
    filter will be off by N hours of timezone offset.
    """
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.split("+")[0].rstrip("Z"), fmt)
        except (ValueError, TypeError):
            continue
    return None


def _label_emoji(label: str) -> str:
    return {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪",
            "no_news": "⚫", "error": "⚠️"}.get(str(label or "").lower(), "⚪")


def _score_color_md(score: int) -> str:
    if score is None:
        return "—"
    s = int(score)
    if s >= 70:
        return f"**🟢 {s}**"
    if s >= 50:
        return f"🟡 {s}"
    return f"🔴 {s}"


def generate_daily_report(
    market: str = "ALL",
    window_hours: int = 24,
    top_n: int = 10,
) -> dict:
    """Generate news-driven daily report from scan_signals.json.

    Args:
        market: 'ALL' / 'US' / 'HK' / 'CN' / 'UK' — filter by alert market.
        window_hours: lookback window from "now".
        top_n: cap top stocks shown by news score.

    Returns:
        {
          'markdown':    str (the report),
          'stats':       {'total_alerts', 'with_news', 'unique_symbols',
                          'bullish_count', 'bearish_count', 'window_hours'},
          'generated_at': ISO timestamp,
        }
    """
    alerts = _load_alerts()
    market = (market or "ALL").upper()
    cutoff = datetime.now() - timedelta(hours=window_hours)

    # Filter by window + market
    in_window: list[dict] = []
    for a in alerts:
        scan_ts = _parse_dt(a.get("scan_time") or "") or _parse_dt(a.get("time") or "")
        if scan_ts is None or scan_ts < cutoff:
            continue
        if market != "ALL" and (a.get("market") or "").upper() != market:
            continue
        in_window.append(a)

    # Dedupe by symbol — keep highest news.score per symbol (or latest if no news)
    by_sym: dict[str, dict] = {}
    for a in in_window:
        sym = a.get("symbol")
        if not sym:
            continue
        prev = by_sym.get(sym)
        cur_score = ((a.get("news") or {}).get("score")) if isinstance(a.get("news"), dict) else None
        prev_score = ((prev.get("news") or {}).get("score")) if (prev and isinstance(prev.get("news"), dict)) else None
        if prev is None:
            by_sym[sym] = a
        elif cur_score is not None and (prev_score is None or cur_score > prev_score):
            by_sym[sym] = a

    with_news = [a for a in by_sym.values()
                 if isinstance(a.get("news"), dict) and (a["news"].get("score") is not None)]
    no_news = [a for a in by_sym.values() if a not in with_news]

    bullish = [a for a in with_news if a["news"].get("label") == "bullish"]
    bearish = [a for a in with_news if a["news"].get("label") == "bearish"]

    # Sort with_news by score desc, then by scan_time desc
    def _sort_key(a):
        return (
            -(a["news"].get("score") or 0),
            -(_parse_dt(a.get("scan_time") or "") or datetime.min).timestamp(),
        )
    with_news.sort(key=_sort_key)
    top = with_news[:top_n]

    now = datetime.now()
    # 2026-04-27: 根据窗口动态选标题，避免本周报告显示「每日报告」。
    if window_hours <= 24:
        period_label = "今日报告"
        period_emoji = "📰"
    elif window_hours <= 72:
        period_label = f"近 {window_hours} 小时报告"
        period_emoji = "📰"
    elif window_hours <= 168:
        period_label = "本周报告"
        period_emoji = "📅"
    else:
        period_label = f"近 {window_hours // 24} 天报告"
        period_emoji = "📅"

    # Real data range can be << requested window when scan_signals.json
    # rolling buffer is short. Surface actual span + warn when <80% of window.
    in_window_times = []
    for a in in_window:
        t = _parse_dt(a.get("scan_time") or "") or _parse_dt(a.get("time") or "")
        if t is not None:
            in_window_times.append(t)
    if in_window_times:
        actual_start = min(in_window_times)
        actual_end = max(in_window_times)
        actual_span_h = max(0.0, (actual_end - actual_start).total_seconds() / 3600.0)
    else:
        actual_start = actual_end = None
        actual_span_h = 0.0

    md_lines: list[str] = []
    md_lines.append(f"# {period_emoji} 新闻驱动选股 · {period_label}")
    if actual_start and actual_end:
        # 真实起止日期 — 用户能直接看到数据覆盖的实际范围
        md_lines.append(
            f"_生成时间 {now.strftime('%Y-%m-%d %H:%M')} · 市场 {market} · "
            f"请求窗口 {window_hours}h · "
            f"**实际数据范围 {actual_start.strftime('%Y-%m-%d %H:%M')} ~ "
            f"{actual_end.strftime('%Y-%m-%d %H:%M')} ({actual_span_h:.1f}h)**_\n"
        )
        # 真实范围 < 请求窗口 80% 时，前端会渲染一个可操作的设置面板（读 stats 里的
        # data_coverage_insufficient 标志），所以 markdown 里只放最简短的提示。
        # 详细操作（调 buffer 配置、立即扫描）走 UI 而非文字。
    else:
        md_lines.append(
            f"_生成时间 {now.strftime('%Y-%m-%d %H:%M')} · 市场 {market} · 请求窗口 {window_hours}h · 无数据_\n"
        )

    md_lines.append("## 📊 概览")
    md_lines.append(f"- 时间窗内总 alerts：**{len(in_window)}**")
    md_lines.append(f"- 去重股票数：**{len(by_sym)}**")
    md_lines.append(f"- 有新闻打分：**{len(with_news)}**（{len(no_news)} 只缺打分）")
    if with_news:
        avg = sum(a["news"]["score"] for a in with_news) / len(with_news)
        md_lines.append(
            f"- 情绪分布：🟢 {len(bullish)} bullish · 🔴 {len(bearish)} bearish · ⚪ "
            f"{len(with_news) - len(bullish) - len(bearish)} neutral · 平均分 {avg:.1f}"
        )

    if not with_news:
        md_lines.append("\n## ℹ️ 暂无新闻打分数据")
        md_lines.append(
            "近期未发现带新闻打分的 alerts。打开「🎯 选股服务」tab，"
            "在「高级筛选 → 📰 新闻热点驱动」选择 `仅打分+排序` 或 `严格过滤` 后跑一次扫描即可生成数据。"
        )
        return {
            "markdown": "\n".join(md_lines),
            "stats": {
                "total_alerts": len(in_window),
                "with_news": 0,
                "unique_symbols": len(by_sym),
                "bullish_count": 0,
                "bearish_count": 0,
                "window_hours": window_hours,
                "actual_data_start": actual_start.isoformat() if actual_start else None,
                "actual_data_end": actual_end.isoformat() if actual_end else None,
                "actual_span_hours": round(actual_span_h, 2),
                "data_coverage_insufficient": bool(actual_start) and (actual_span_h < window_hours * 0.8),
            },
            "generated_at": now.isoformat(),
        }

    md_lines.append(f"\n## 🏆 TOP {min(top_n, len(top))} 新闻催化股")
    md_lines.append("")
    md_lines.append("| 排名 | 代码 | 名称 | 市场 | 策略 | 价格 | 情绪 | 标签 | 新闻数 |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, a in enumerate(top, 1):
        n = a["news"]
        sym = a.get("symbol") or "-"
        name = (a.get("name") or "")[:30]
        mkt = a.get("market") or "-"
        strat = (a.get("strategy") or "-")[:25]
        price = a.get("price")
        price_str = f"${price}" if price is not None else "—"
        score_str = _score_color_md(n.get("score"))
        label = f"{_label_emoji(n.get('label'))} {n.get('label', '')}"
        n_items = n.get("n_items", 0)
        md_lines.append(
            f"| {i} | **{sym}** | {name} | {mkt} | {strat} | {price_str} | "
            f"{score_str} | {label} | {n_items} |"
        )

    # Per-stock detail with reasoning + top3 news
    md_lines.append("\n## 📋 逐股详情（TOP {0}）".format(min(top_n, len(top))))
    for i, a in enumerate(top, 1):
        n = a["news"]
        sym = a.get("symbol") or "-"
        md_lines.append(
            f"\n### {i}. {sym} {_label_emoji(n.get('label'))} score={n.get('score')}"
        )
        reason = (n.get("reasoning") or "").strip()
        if reason:
            md_lines.append(f"\n> {reason}\n")
        top3 = n.get("top3") or []
        if top3:
            md_lines.append("**近 3 条新闻：**")
            for j, t in enumerate(top3[:3], 1):
                title = (t.get("title") or "").strip()
                src = t.get("source") or ""
                url = t.get("url") or ""
                pub = (t.get("published_at") or "")[:16]
                if url:
                    md_lines.append(f"{j}. [{title}]({url}) — _{src} · {pub}_")
                else:
                    md_lines.append(f"{j}. {title} — _{src} · {pub}_")

    return {
        "markdown": "\n".join(md_lines),
        "stats": {
            "total_alerts": len(in_window),
            "with_news": len(with_news),
            "unique_symbols": len(by_sym),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "window_hours": window_hours,
            "actual_data_start": actual_start.isoformat() if actual_start else None,
            "actual_data_end": actual_end.isoformat() if actual_end else None,
            "actual_span_hours": round(actual_span_h, 2),
            "data_coverage_insufficient": bool(actual_start) and (actual_span_h < window_hours * 0.8),
        },
        "generated_at": now.isoformat(),
    }
