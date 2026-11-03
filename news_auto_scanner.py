"""News auto-scan scheduler.

后台调度器：定时调用本地 /selector/jobs/start 端点，让选股扫描+新闻打分
持续在后台运行，scan_signals.json 自然累积出"本周报告"所需的 7+ 天数据。

设计：
- 单线程（threading.Thread + Event）。每 60s 醒一次，读 news_buffer_config.json
  里的 auto_scan 配置，决定是否到点该跑。
- 跑的时候直接 HTTP POST 到 localhost 自己的 /selector/jobs/start，避免在
  scheduler 线程里复制选股管道逻辑。等于是模拟用户点「开始选股」。
- 跑完后更新 last_run_at / last_run_status / last_run_alerts，前端能查询。
- 优雅关闭：app shutdown 时 set Event 让 loop 退出。

配置（news_buffer_config.json 里的 auto_scan 块）：
{
  "enabled": true,
  "interval_minutes": 60,
  "market": "US",
  "news_mode": "weight",        # off / weight / must
  "interval_bar": "1d",
  "signal_side": "buy",
  "last_run_at": "<iso>",       # 由 scheduler 写
  "last_run_status": "ok",      # ok / error
  "last_run_alerts": 12,
  "last_run_message": "...",
  "next_run_at": "<iso>",
}

启动方式：在 main.py 启动时 import + start_auto_scanner()。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "news_buffer_config.json"
LOOPBACK_BASE = "http://127.0.0.1:9000"  # 本地 9000 自己

DEFAULTS = {
    "max_days": 7,
    "max_records": 20000,
    "auto_scan": {
        "enabled": False,
        "interval_minutes": 60,
        "market": "US",
        "news_mode": "weight",      # off / weight / must
        "interval_bar": "1d",
        "signal_side": "buy",
        "last_run_at": None,
        "last_run_status": None,
        "last_run_alerts": 0,
        "last_run_message": None,
        "next_run_at": None,
    },
}

_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_lock = threading.Lock()


def _read_config() -> dict:
    """Read full config with defaults filled in. Preserves any unknown
    sibling blocks (e.g. ``auto_pools`` owned by news_pools.py)."""
    cfg = {}
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"news_buffer_config read failed: {e}")
    # Apply defaults for known root keys; coerce types
    cfg.setdefault("max_days", DEFAULTS["max_days"])
    cfg.setdefault("max_records", DEFAULTS["max_records"])
    cfg["max_days"] = int(cfg["max_days"])
    cfg["max_records"] = int(cfg["max_records"])
    auto = dict(DEFAULTS["auto_scan"])
    if isinstance(cfg.get("auto_scan"), dict):
        auto.update(cfg["auto_scan"])
    cfg["auto_scan"] = auto
    return cfg


def _write_config(cfg: dict) -> None:
    """Persist config (atomic via temp file + rename)."""
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def update_config(**kwargs) -> dict:
    """Thread-safe partial update of auto_scan block."""
    with _lock:
        cfg = _read_config()
        auto = cfg.setdefault("auto_scan", dict(DEFAULTS["auto_scan"]))
        for k, v in kwargs.items():
            auto[k] = v
        _write_config(cfg)
        return cfg


def update_root_config(**kwargs) -> dict:
    """Thread-safe partial update of root keys (max_days, max_records, ...)."""
    with _lock:
        cfg = _read_config()
        for k, v in kwargs.items():
            cfg[k] = v
        _write_config(cfg)
        return cfg


def get_config() -> dict:
    return _read_config()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _due_to_run(auto: dict) -> bool:
    """Decide if a scan should fire now."""
    if not auto.get("enabled"):
        return False
    last = auto.get("last_run_at")
    interval = max(5, int(auto.get("interval_minutes") or 60))
    if not last:
        return True  # never ran → run now
    try:
        last_dt = datetime.fromisoformat(str(last).split("+")[0].rstrip("Z"))
    except (ValueError, TypeError):
        return True
    return datetime.now() >= last_dt + timedelta(minutes=interval)


_ALL_MARKETS = ["US", "HK", "CN", "UK"]


def _build_selector_payload(auto: dict) -> dict:
    """Construct the POST body for /selector/jobs/start.

    Defaults aim for "lightweight but useful":
    - scope=market with one or more markets
    - empty strategies = use all active strategies (selector handles default)
    - filters: news_require according to auto.news_mode

    Market rules:
      "ALL"  → expand to ['US','HK','CN','UK'] — true multi-market scan
      single → ['US']
      comma-separated 'US,HK' → ['US','HK']
    """
    market_raw = (auto.get("market") or "US").upper().strip()
    if market_raw == "ALL":
        markets = list(_ALL_MARKETS)
    elif "," in market_raw:
        markets = [m.strip() for m in market_raw.split(",") if m.strip()]
        # Validate against known set; keep order
        markets = [m for m in markets if m in _ALL_MARKETS]
        if not markets:
            markets = ["US"]
    else:
        markets = [market_raw if market_raw in _ALL_MARKETS else "US"]

    interval_bar = auto.get("interval_bar") or "1d"
    signal_side = auto.get("signal_side") or "buy"
    news_mode = (auto.get("news_mode") or "weight").lower()
    if news_mode not in ("off", "weight", "must"):
        news_mode = "weight"

    filters: dict = {}
    if news_mode != "off":
        filters["news_require"] = news_mode
        filters["news_window_hours"] = 24
        filters["news_sentiment_min"] = 0  # weight mode doesn't drop, must mode would

    return {
        "scope": "market",
        "markets": markets,
        "interval": interval_bar,
        "strategies": [],   # empty = all active
        "signal_side": signal_side,
        "refresh_sec": 600,
        "filters": filters,
    }


def _trigger_scan(auto: dict) -> dict:
    """Fire a scan via local HTTP. Returns {ok, alerts_count, message}."""
    payload = _build_selector_payload(auto)
    try:
        # Start the job (fire & forget — selector runs async on the server side)
        r = requests.post(
            f"{LOOPBACK_BASE}/selector/jobs/start",
            json=payload,
            timeout=15,
        )
        if r.status_code != 200:
            return {"ok": False, "alerts_count": 0,
                    "message": f"start HTTP {r.status_code}: {r.text[:200]}"}
        task_id = r.json().get("task_id")
        if not task_id:
            return {"ok": False, "alerts_count": 0, "message": "no task_id in response"}
    except Exception as e:
        return {"ok": False, "alerts_count": 0, "message": f"start failed: {e}"}

    # Poll until done (max 25 min — selector usually finishes < 15 min for one market)
    deadline = time.time() + 25 * 60
    last_status = None
    while time.time() < deadline:
        if _stop_event.is_set():
            return {"ok": False, "alerts_count": 0,
                    "message": "scheduler shutdown during scan"}
        try:
            r = requests.get(f"{LOOPBACK_BASE}/selector/jobs/status/{task_id}",
                             timeout=10)
            if r.ok:
                data = r.json()
                last_status = data.get("status")
                if last_status in ("done", "error"):
                    if last_status == "error":
                        return {"ok": False, "alerts_count": 0,
                                "message": f"job error: {data.get('error', '')[:200]}"}
                    # Pull result count
                    res = requests.get(
                        f"{LOOPBACK_BASE}/selector/jobs/result/{task_id}",
                        timeout=10,
                    )
                    if res.ok:
                        rj = res.json()
                        n_alerts = len(rj.get("alerts") or [])
                        return {"ok": True, "alerts_count": n_alerts,
                                "message": f"task {task_id[:8]}, {n_alerts} alerts"}
                    return {"ok": True, "alerts_count": 0,
                            "message": f"task {task_id[:8]} done, result fetch failed"}
        except Exception as e:
            logger.debug(f"auto_scan poll error (will retry): {e}")
        # Poll every 5s
        _stop_event.wait(timeout=5)

    return {"ok": False, "alerts_count": 0,
            "message": f"timeout after 25 min (last status: {last_status})"}


def _loop():
    """Main scheduler loop. Wakes every 60s, decides whether to run."""
    logger.info("news_auto_scanner: thread started")
    # On startup: small grace period so the FastAPI app finishes booting before we
    # start firing HTTP requests at it
    _stop_event.wait(timeout=30)

    while not _stop_event.is_set():
        try:
            cfg = _read_config()
            auto = cfg.get("auto_scan") or {}
            if _due_to_run(auto):
                logger.info(f"news_auto_scanner: firing scan (market={auto.get('market')}, "
                            f"news_mode={auto.get('news_mode')}, interval="
                            f"{auto.get('interval_minutes')}m)")
                t0 = time.time()
                result = _trigger_scan(auto)
                dt = time.time() - t0
                # Update config with run metadata
                interval = int(auto.get("interval_minutes") or 60)
                next_run = (datetime.now() + timedelta(minutes=interval)).isoformat()
                update_config(
                    last_run_at=_now_iso(),
                    last_run_status="ok" if result["ok"] else "error",
                    last_run_alerts=int(result.get("alerts_count") or 0),
                    last_run_message=str(result.get("message") or "")[:300],
                    next_run_at=next_run,
                )
                logger.info(f"news_auto_scanner: scan done in {dt:.1f}s, "
                            f"ok={result['ok']}, alerts={result['alerts_count']}")
        except Exception as e:
            logger.warning(f"news_auto_scanner loop error: {e}")
        # Sleep 60s, but wake early on shutdown
        _stop_event.wait(timeout=60)

    logger.info("news_auto_scanner: thread exiting")


def start_auto_scanner() -> bool:
    """Start the background scheduler. Idempotent."""
    global _thread
    if _thread and _thread.is_alive():
        return False
    _stop_event.clear()
    _thread = threading.Thread(target=_loop, name="news_auto_scanner", daemon=True)
    _thread.start()
    logger.info("news_auto_scanner: start requested")
    return True


def stop_auto_scanner() -> None:
    """Signal the loop to exit. Won't block."""
    _stop_event.set()


def is_running() -> bool:
    return bool(_thread and _thread.is_alive())
