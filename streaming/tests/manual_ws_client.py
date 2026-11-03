"""Manual end-to-end WS smoke test.

Connects to ws://localhost:9000/ws/quote/US/AAPL?token=...
Subscribes to tick + bar(1m). Prints whatever comes in for N seconds.

Usage:
    ./venv/bin/python streaming/tests/manual_ws_client.py [seconds=30]

This file is intentionally NOT in the automated test runner — it requires
the 9000 service running + IB Gateway connected, and during market-closed
hours it won't produce ticks.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Load .env so LIVE_STREAMING_TOKEN is available
ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
if ENV_FILE.is_file():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

import websockets


HOST = os.environ.get("WS_HOST", "127.0.0.1:9000")
TOKEN = os.environ.get("LIVE_STREAMING_TOKEN")
if not TOKEN:
    raise SystemExit("LIVE_STREAMING_TOKEN not set in env")

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SYMBOL = os.environ.get("WS_SYMBOL", "AAPL")
MARKET = os.environ.get("WS_MARKET", "US")
INTERVAL = os.environ.get("WS_INTERVAL", "1m")


async def watch(url: str, label: str, duration: float) -> dict:
    """Connect to url, collect frames for duration seconds, return stats."""
    stats = {
        "label": label,
        "url": url,
        "frames_received": 0,
        "by_type": {},
        "first_frame_after_s": None,
        "tick_latency_p50_ms": None,
        "tick_latency_p99_ms": None,
        "errors": [],
    }
    latencies = []
    start = time.monotonic()
    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            print(f"[{label}] connected")
            while True:
                elapsed = time.monotonic() - start
                remaining = duration - elapsed
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    stats["errors"].append("non-json frame")
                    continue
                t = msg.get("type", "?")
                stats["frames_received"] += 1
                stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
                if stats["first_frame_after_s"] is None:
                    stats["first_frame_after_s"] = round(elapsed, 3)
                # Compute push latency for tick frames (now_ms - msg.ts)
                if t == "tick":
                    now = int(time.time() * 1000)
                    msg_ts = msg.get("ts")
                    if isinstance(msg_ts, int):
                        latencies.append(now - msg_ts)
                # Print first few of each type so we can see what's happening
                if stats["by_type"][t] <= 3:
                    print(f"  [{label}] {t}: {json.dumps(msg, ensure_ascii=False)[:200]}")
    except Exception as e:
        stats["errors"].append(f"{type(e).__name__}: {e}")
        print(f"[{label}] ERROR: {e}")

    if latencies:
        latencies.sort()
        n = len(latencies)
        stats["tick_latency_p50_ms"] = latencies[n // 2]
        stats["tick_latency_p99_ms"] = latencies[min(n - 1, int(n * 0.99))]
    return stats


async def main():
    qurl = f"ws://{HOST}/ws/quote/{MARKET}/{SYMBOL}?token={TOKEN}"
    kurl = f"ws://{HOST}/ws/kline/{MARKET}/{SYMBOL}/{INTERVAL}?token={TOKEN}"
    print(f"Duration: {DURATION}s  Symbol: {MARKET}/{SYMBOL}  Interval: {INTERVAL}")
    print(f"  tick URL: {qurl[:80]}…")
    print(f"  kline URL: {kurl[:80]}…")
    print()
    tick_stats, bar_stats = await asyncio.gather(
        watch(qurl, "tick", DURATION),
        watch(kurl, "bar ", DURATION),
    )
    print()
    print("=" * 60)
    for s in (tick_stats, bar_stats):
        print(f"[{s['label']}] frames={s['frames_received']}, "
              f"first={s['first_frame_after_s']}s, by_type={s['by_type']}")
        if s["tick_latency_p50_ms"] is not None:
            print(f"  push latency p50={s['tick_latency_p50_ms']}ms "
                  f"p99={s['tick_latency_p99_ms']}ms (n={s['frames_received']})")
        if s["errors"]:
            print(f"  errors: {s['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
