#!/usr/bin/env python3
"""compare_baseline.py — diff two baseline_overlap snapshots.

Usage:
    python scripts/compare_baseline.py before.json after.json

Exit codes:
    0 = success
    1 = missing or invalid file (or not JSON)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: {path} is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict):
        print(f"ERROR: {path} does not contain a JSON object", file=sys.stderr)
        sys.exit(1)
    return data


def _g(d: dict, key: str, default=0):
    val = d.get(key, default)
    return default if val is None else val


def _fmt_delta(delta: float, digits: int = 4) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{round(delta, digits)}"


def _fmt_pct(before: int, after: int) -> str:
    if before <= 0:
        return "n/a (before=0)"
    reduction = (before - after) / before * 100.0
    return f"{reduction:+.1f}%"


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: compare_baseline.py <before.json> <after.json>", file=sys.stderr)
        return 1

    before = _load(sys.argv[1])
    after = _load(sys.argv[2])

    print("=== Scan Center v0.5 — baseline diff ===")
    print(f"  before: {sys.argv[1]}  measured_at={before.get('measured_at_utc', 'n/a')}")
    print(f"  after:  {sys.argv[2]}  measured_at={after.get('measured_at_utc', 'n/a')}")
    print(f"  groups: {_g(before, 'group_count')} -> {_g(after, 'group_count')}")
    print()

    # Stock overlap
    s_b = float(_g(before, "stock_overlap_ratio", 0.0))
    s_a = float(_g(after, "stock_overlap_ratio", 0.0))
    print("-- Stock overlap ratio --")
    print(f"  before:  {s_b}")
    print(f"  after:   {s_a}")
    print(f"  delta:   {_fmt_delta(s_a - s_b)}")
    print(f"  unique_symbols: {_g(before, 'unique_symbols_total')} -> {_g(after, 'unique_symbols_total')}")
    print()

    # Strategy overlap
    t_b = float(_g(before, "strategy_overlap_ratio", 0.0))
    t_a = float(_g(after, "strategy_overlap_ratio", 0.0))
    print("-- Strategy overlap ratio --")
    print(f"  before:  {t_b}")
    print(f"  after:   {t_a}")
    print(f"  delta:   {_fmt_delta(t_a - t_b)}")
    print(f"  unique_strategies: {_g(before, 'unique_strategies_total')} -> {_g(after, 'unique_strategies_total')}")
    print()

    # LLM calls
    l_b = int(_g(before, "llm_calls_yesterday", 0))
    l_a = int(_g(after, "llm_calls_yesterday", 0))
    print("-- LLM calls (yesterday) --")
    print(f"  before:  {l_b}")
    print(f"  after:   {l_a}")
    print(f"  delta:   {_fmt_delta(l_a - l_b, digits=0)}")
    print(f"  reduction: {_fmt_pct(l_b, l_a)}")
    print()

    # Duplicates
    d_b = int(_g(before, "duplicate_llm_calls_yesterday", 0))
    d_a = int(_g(after, "duplicate_llm_calls_yesterday", 0))
    print("-- Duplicate LLM calls (yesterday) --")
    print(f"  before:  {d_b}")
    print(f"  after:   {d_a}")
    print(f"  delta:   {_fmt_delta(d_a - d_b, digits=0)}")
    print()

    # Savings summary
    projected = int(_g(before, "projected_savings_llm_calls", 0))
    actual = max(0, l_b - l_a)
    print("-- Savings summary --")
    print(f"  projected reduction (from before.projected_savings): {projected} calls")
    print(f"  actual reduction   (before.llm_calls - after.llm_calls): {actual} calls")
    if projected > 0:
        coverage = actual / projected * 100.0
        print(f"  actual / projected: {coverage:.1f}%")
    else:
        print("  actual / projected: n/a (projected=0)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
