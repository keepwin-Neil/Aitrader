#!/usr/bin/env python3
"""test_per_market_settings.py — verify per-market scan settings store.

Pins down the contract for the per-market monitoring console (2026-04-25):
each CORE market has its own server-side settings, settable via PUT
/scan/monitor/settings/<market> and read by the daemon to schedule
per-market scans at each market's refresh_sec.

What's covered:
  1. load_per_market_settings returns all 5 CORE markets even on fresh install.
  2. Defaults match the page-level defaults (1d / 5min / Top5 / Claude+GPT+Gemini).
  3. save_per_market_settings persists to disk and round-trips.
  4. Partial updates merge with existing values (don't clobber unrelated keys).
  5. Unknown market raises ValueError.
  6. refresh_sec clamped to >= 30 (so users can't accidentally DDoS the daemon).
  7. owners normalized to lowercase strings, empties dropped.
  8. enabled coerced to bool.
  9. GET /scan/monitor/settings returns the canonical shape via FastAPI router.
  10. PUT /scan/monitor/settings/<market> writes + returns full updated dict.

Run::
    venv/bin/python \\
        scripts/test_per_market_settings.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("SCAN_AUDIT_DIR", "/tmp/test_audit_per_market")

import main  # noqa: E402
from routers.scan_monitor import (  # noqa: E402
    get_per_market_settings,
    put_per_market_settings,
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
    print("=== per_market_scan_settings ===")

    # Use a temp file so we don't trash the dev settings.
    orig_path = main.PER_MARKET_SETTINGS_FILE
    with tempfile.TemporaryDirectory() as tmpdir:
        main.PER_MARKET_SETTINGS_FILE = Path(tmpdir) / "per_market_scan_settings.json"

        # 1) Fresh install (no file) → load returns all CORE markets with defaults.
        loaded = main.load_per_market_settings()
        check(
            "1 fresh install returns all CORE markets",
            set(loaded.keys()) == set(main.CORE_MARKETS),
            detail=f"got keys={sorted(loaded.keys())}",
        )

        # 2) Defaults match the spec.
        us = loaded.get("US", {})
        check(
            "2 default US settings: 1d / 300s / top_n=5 / claude+gpt+gemini",
            us.get("interval") == "1d"
            and us.get("refresh_sec") == 300
            and us.get("top_n") == 5
            and set(us.get("owners", [])) == {"claude", "gpt", "gemini"},
            detail=f"got us={us}",
        )

        # 3) Save → load round-trip.
        full = main.save_per_market_settings("US", {"top_n": 10})
        check(
            "3 save round-trip: US top_n=10 persisted to disk",
            full.get("US", {}).get("top_n") == 10
            and main.load_per_market_settings().get("US", {}).get("top_n") == 10,
            detail=f"got US.top_n={full.get('US',{}).get('top_n')}",
        )

        # 4) Partial update merges (refresh_sec unchanged when only top_n sent).
        before_freq = main.load_per_market_settings()["US"]["refresh_sec"]
        main.save_per_market_settings("US", {"interval": "5m"})
        after = main.load_per_market_settings()["US"]
        check(
            "4 partial update merges (interval changed, refresh_sec preserved)",
            after.get("interval") == "5m" and after.get("refresh_sec") == before_freq,
            detail=f"got after={after}",
        )

        # 5) Unknown market → ValueError.
        raised = False
        try:
            main.save_per_market_settings("UNKNOWN", {"top_n": 5})
        except ValueError:
            raised = True
        check(
            "5 unknown market raises ValueError",
            raised,
            detail="silently saved unknown market",
        )

        # 6) refresh_sec clamped to >= 30.
        main.save_per_market_settings("US", {"refresh_sec": 1})
        clamped = main.load_per_market_settings()["US"]["refresh_sec"]
        check(
            "6 refresh_sec=1 clamped to >= 30",
            clamped >= 30,
            detail=f"got refresh_sec={clamped}",
        )

        # 7) Owners normalized: case + whitespace + empties.
        main.save_per_market_settings("US", {"owners": ["Claude", " GPT ", "", "GEMINI"]})
        owners = main.load_per_market_settings()["US"]["owners"]
        check(
            "7 owners normalized to lowercase, empties dropped",
            owners == ["claude", "gpt", "gemini"],
            detail=f"got owners={owners}",
        )

        # 8) enabled coerced to bool.
        main.save_per_market_settings("US", {"enabled": 0})
        check(
            "8a enabled=0 coerced to False",
            main.load_per_market_settings()["US"]["enabled"] is False,
        )
        main.save_per_market_settings("US", {"enabled": "yes"})
        check(
            "8b enabled='yes' coerced to True",
            main.load_per_market_settings()["US"]["enabled"] is True,
        )

        # 9) GET endpoint returns {markets: {...}} with all CORE markets.
        resp = get_per_market_settings()
        check(
            "9 GET /scan/monitor/settings returns {markets: {US/HK/CN/CRYPTO/UK}}",
            isinstance(resp, dict)
            and "markets" in resp
            and set(resp["markets"].keys()) == set(main.CORE_MARKETS),
            detail=f"got keys={sorted((resp.get('markets') or {}).keys())}",
        )

        # 10) PUT endpoint returns full updated dict.
        put_resp = put_per_market_settings("CN", {"top_n": 7, "interval": "5m"})
        check(
            "10a PUT returns {markets: ..., saved: 'CN'}",
            isinstance(put_resp, dict)
            and put_resp.get("saved") == "CN"
            and put_resp.get("markets", {}).get("CN", {}).get("top_n") == 7,
            detail=f"got resp={put_resp}",
        )
        # 10b PUT with unknown market → HTTPException 400.
        from fastapi import HTTPException
        raised = False
        try:
            put_per_market_settings("FOOBAR", {"top_n": 5})
        except HTTPException as e:
            raised = (e.status_code == 400)
        check(
            "10b PUT unknown market → HTTP 400",
            raised,
            detail="should return 400 for unknown market",
        )

    # Restore original path so subsequent imports of main don't see temp.
    main.PER_MARKET_SETTINGS_FILE = orig_path

    print()
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main_test())
