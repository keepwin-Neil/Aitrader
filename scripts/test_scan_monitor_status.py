#!/usr/bin/env python3
"""test_scan_monitor_status.py — verify /scan/monitor/status endpoint.

Pins down the contract for the trading-hours-aware UI (2026-04-25):
the scan-monitor page polls this every 30s to switch between
"🟢 自动扫描中" and "⏸ 非交易时段 · 立即扫描" modes.

Run::
    venv/bin/python \\
        scripts/test_scan_monitor_status.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Set audit dir before importing main (avoids /var/lib write at import time).
os.environ.setdefault("SCAN_AUDIT_DIR", "/tmp/test_scan_audit_status")

from routers.scan_monitor import (  # noqa: E402
    get_trading_status, _market_status_one,
)

passed = 0
failed = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {label}")
    else:
        failed += 1
        msg = f"FAIL: {label}"
        if detail:
            msg += f" ({detail})"
        print(f"  {msg}")


def main_test() -> int:
    print("=== /scan/monitor/status ===")

    j = get_trading_status()

    # 1) Top-level shape contract.
    expected_top = {"checked_at", "any_open", "open_markets", "markets"}
    check(
        "1 top-level keys = {checked_at, any_open, open_markets, markets}",
        set(j.keys()) >= expected_top,
        detail=f"got {sorted(j.keys())}",
    )

    # 2) markets is a list of dicts with the required per-market keys.
    required_market_keys = {"market", "is_open", "tz", "next_open", "next_close", "is_24x7"}
    bad = []
    for m in j["markets"]:
        missing = required_market_keys - set(m.keys())
        if missing:
            bad.append((m.get("market"), missing))
    check(
        "2 every market dict has required keys",
        not bad,
        detail=f"missing keys per market: {bad}",
    )

    # 3) CORE_MARKETS includes US/CN/HK/UK/CRYPTO — they should all appear.
    market_codes = {m["market"] for m in j["markets"]}
    expected_markets = {"US", "CN", "HK", "UK", "CRYPTO"}
    check(
        "3 markets list covers US/CN/HK/UK/CRYPTO",
        expected_markets <= market_codes,
        detail=f"missing: {expected_markets - market_codes}",
    )

    # 4) CRYPTO must be marked is_24x7=True with is_open=True (always-on).
    crypto = next((m for m in j["markets"] if m["market"] == "CRYPTO"), None)
    check(
        "4 CRYPTO is is_24x7=True + is_open=True",
        crypto is not None and crypto.get("is_24x7") is True and crypto.get("is_open") is True,
        detail=f"got crypto={crypto}",
    )

    # 5) any_open / open_markets must EXCLUDE CRYPTO so the badge doesn't
    # always glow green (per the comment in get_trading_status).
    open_set = set(j.get("open_markets", []))
    check(
        "5 open_markets excludes CRYPTO (24/7 doesn't count)",
        "CRYPTO" not in open_set,
        detail=f"open_markets contains CRYPTO ({open_set}) — badge would always be green",
    )

    # 6) any_open is consistent with len(open_markets) > 0.
    check(
        "6 any_open consistent with len(open_markets) > 0",
        bool(j.get("any_open")) == bool(j.get("open_markets")),
        detail=f"any_open={j.get('any_open')}, open_markets={j.get('open_markets')}",
    )

    # 7) For closed markets, next_open is non-null (so the UI can render
    # a countdown). Closed CORE markets should always have a next_open.
    closed_no_next = [
        m["market"] for m in j["markets"]
        if not m["is_open"] and not m.get("is_24x7") and not m.get("next_open")
    ]
    check(
        "7 every closed CORE market carries a next_open ISO timestamp",
        not closed_no_next,
        detail=f"missing next_open for: {closed_no_next}",
    )

    # 8) For currently-open markets, next_close is non-null.
    open_no_close = [
        m["market"] for m in j["markets"]
        if m["is_open"] and not m.get("is_24x7") and not m.get("next_close")
    ]
    check(
        "8 every currently-open CORE market carries a next_close ISO",
        not open_no_close,
        detail=f"missing next_close for: {open_no_close}",
    )

    # 9) tz strings are real-looking IANA names for CORE markets.
    expected_tz = {
        "US": "America/New_York",
        "CN": "Asia/Shanghai",
        "HK": "Asia/Hong_Kong",
        "UK": "Europe/London",
    }
    bad_tz = []
    for m in j["markets"]:
        exp = expected_tz.get(m["market"])
        if exp and m.get("tz") != exp:
            bad_tz.append((m["market"], m.get("tz"), exp))
    check(
        "9 CORE market tz strings match expected IANA names",
        not bad_tz,
        detail=f"mismatched tz: {bad_tz}",
    )

    # 10) Spot-check _market_status_one for an unknown market — should
    # return is_open=True + is_24x7=True (treated as always-on).
    unknown = _market_status_one("FOREX")
    check(
        "10 unknown market → is_24x7=True, is_open=True",
        unknown.get("is_24x7") is True and unknown.get("is_open") is True,
        detail=f"got {unknown}",
    )

    print()
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main_test())
