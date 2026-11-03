"""Scan monitor router (Scan Center v0.5 §7.3 / P8 + §7.5 / P12).

Exposes four read-only endpoints over the audit ring owned by ``scan_audit``:

  * ``GET /scan/monitor/cycles``              — list distinct cycle_ids
  * ``GET /scan/monitor/cycle/{cycle_id}``    — full stage dump for one cycle
  * ``GET /scan/monitor/symbol/{symbol}``     — symbol-scoped slice
  * ``GET /scan/monitor/stats``               — aggregate metrics

``/cycles`` and ``/stats`` are ring-only (recent activity). ``/cycle/{cid}``
and ``/symbol/{sym}?cycle_id=...`` fall back to ``scan_audit.read_segment()``
when the cycle isn't in the ring (P12): the ring holds the 10,000 most-recent
records (~100 cycles at typical 100 records/cycle), older cycles live only
on disk as per-cycle JSONL segments.

All ring reads go through ``snapshot_audit_ring()`` — router code never
touches ``_audit_ring`` directly so the lock discipline stays inside
``scan_audit`` itself.

No auth (read-only ops endpoints).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Body, HTTPException, Path, Query

import scan_audit
import scan_center

router = APIRouter(prefix="/scan/monitor", tags=["scan-monitor"])


_STAGES = (
    "universe",
    "raw_signal",
    # §7.2.2b: emitted by unified_scan() to make "silent 0 alerts" cycles
    # (e.g. realtime_scan_fn exception → alerts=[], fetch returned empty,
    # universe/strategies empty) distinguishable from a legitimate no-signal
    # cycle in the UI. Each record carries:
    #   bucket, interval, side, path, groups, universe_size,
    #   strategies_count, alerts_emitted, error
    # One record per bucket (not per symbol), so the count stays ≤ number
    # of buckets per cycle (typically 1-5).
    "scan_summary",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_stage_counts() -> "dict[str, int]":
    return {s: 0 for s in _STAGES}


def _empty_stage_rows() -> "dict[str, list]":
    return {s: [] for s in _STAGES}


def _group_cycles(snapshot: "list[dict]") -> "dict[str, dict]":
    """Build {cycle_id: {first_ts, last_ts, counts}} from a snapshot list.

    Unknown stage names (not in ``_STAGES``) still update the cycle's
    first_ts/last_ts, but are omitted from the per-stage counts — this keeps
    the response shape stable even if a new stage name leaks in from an older
    ring before ``counts.keys()`` catches up.
    """
    out: "dict[str, dict]" = {}
    for rec in snapshot:
        cid = rec.get("cycle_id", "unknown")
        stage = rec.get("stage", "")
        ts = float(rec.get("timestamp") or 0.0)
        entry = out.get(cid)
        if entry is None:
            entry = {
                "cycle_id": cid,
                "first_ts": ts,
                "last_ts": ts,
                "counts": _empty_stage_counts(),
            }
            out[cid] = entry
        else:
            if ts < entry["first_ts"]:
                entry["first_ts"] = ts
            if ts > entry["last_ts"]:
                entry["last_ts"] = ts
        if stage in entry["counts"]:
            entry["counts"][stage] += 1
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/cycles")
def list_cycles(limit: int = Query(50, ge=1, le=10_000)) -> dict:
    """List distinct cycle_ids in the ring, most recent first.

    For each cycle: first_ts/last_ts (seconds epoch) and per-stage counts.
    ``limit`` caps the number of cycles returned (not records scanned; the
    ring is small enough to scan in full).
    """
    snapshot = scan_audit.snapshot_audit_ring()
    grouped = _group_cycles(snapshot)
    # Sort by last_ts desc so the newest cycle is on top.
    sorted_cycles = sorted(
        grouped.values(), key=lambda c: c["last_ts"], reverse=True,
    )
    return {
        "cycles": sorted_cycles[:limit],
        "total": len(sorted_cycles),
    }


@router.get("/cycle/{cycle_id}")
def get_cycle(cycle_id: str) -> dict:
    """All records for one cycle, grouped by stage.

    In-memory ring is checked first. If the cycle isn't in the ring (aged
    out; ring holds 10,000 most-recent records — ~100 cycles at typical
    100 records/cycle), falls back to ``scan_audit.read_segment(cycle_id)``
    for the on-disk JSONL segment (P12 §7.5). Returns 404 only if neither
    source has the cycle.
    """
    if not cycle_id or not cycle_id.strip():
        raise HTTPException(400, detail="cycle_id must not be empty")
    snapshot = scan_audit.snapshot_audit_ring()
    stages = _empty_stage_rows()
    found = False
    for rec in snapshot:
        if rec.get("cycle_id") != cycle_id:
            continue
        found = True
        stage = rec.get("stage", "")
        if stage in stages:
            stages[stage].append(rec)
    if not found:
        # NOTE (I3): If the ring has ANY records for this cycle, we trust
        # them as authoritative and don't consult disk. If the cycle has
        # been partially evicted (some stages in ring, others flushed &
        # dropped from ring), the response will be missing the evicted
        # stages. Acceptable trade-off: ring holds 10k records, typical
        # cycle has <100.
        # P12 disk fallback: try the on-disk segment before 404'ing.
        disk_rows = scan_audit.read_segment(cycle_id)
        if disk_rows:
            for rec in disk_rows:
                if rec.get("cycle_id") != cycle_id:
                    continue
                found = True
                stage = rec.get("stage", "")
                if stage in stages:
                    stages[stage].append(rec)
    if not found:
        raise HTTPException(404, detail=f"cycle_id not found: {cycle_id}")
    return {"cycle_id": cycle_id, "stages": stages}


@router.get("/symbol/{symbol}")
def get_symbol(
    symbol: str,
    cycle_id: str = Query(...),
) -> dict:
    """All records in the given cycle whose payload has a matching ``symbol``.

    In-memory ring is checked first. If the cycle isn't in the ring (aged
    out), falls back to ``scan_audit.read_segment(cycle_id)`` (P12 §7.5).
    404 only if neither source has the cycle. Empty arrays (not 404) when
    the cycle exists but the symbol is absent from every payload.
    """
    if not symbol or not symbol.strip():
        raise HTTPException(400, detail="symbol must not be empty")
    if not cycle_id or not cycle_id.strip():
        raise HTTPException(400, detail="cycle_id must not be empty")
    snapshot = scan_audit.snapshot_audit_ring()
    stages = _empty_stage_rows()
    cycle_has_any = False
    for rec in snapshot:
        if rec.get("cycle_id") != cycle_id:
            continue
        cycle_has_any = True
        payload = rec.get("payload") or {}
        if payload.get("symbol") != symbol:
            continue
        stage = rec.get("stage", "")
        if stage in stages:
            stages[stage].append(rec)
    if not cycle_has_any:
        # NOTE (I3): If the ring has ANY records for this cycle, we trust
        # them as authoritative and don't consult disk. If the cycle has
        # been partially evicted (some stages in ring, others flushed &
        # dropped from ring), the response will be missing the evicted
        # stages. Acceptable trade-off: ring holds 10k records, typical
        # cycle has <100.
        # P12 disk fallback.
        disk_rows = scan_audit.read_segment(cycle_id)
        if disk_rows:
            for rec in disk_rows:
                if rec.get("cycle_id") != cycle_id:
                    continue
                cycle_has_any = True
                payload = rec.get("payload") or {}
                if payload.get("symbol") != symbol:
                    continue
                stage = rec.get("stage", "")
                if stage in stages:
                    stages[stage].append(rec)
    if not cycle_has_any:
        raise HTTPException(404, detail=f"cycle_id not found: {cycle_id}")
    return {"cycle_id": cycle_id, "symbol": symbol, "stages": stages}


@router.get("/stats")
def get_stats() -> dict:
    """Aggregate ring metrics + cache-derived counters.

    ``scans_avoided`` — ``SignalCache.stats['hit']`` summed across intervals.
    """
    snapshot = scan_audit.snapshot_audit_ring()
    by_stage = _empty_stage_counts()
    cycles_seen: "set[str]" = set()
    for rec in snapshot:
        cid = rec.get("cycle_id", "unknown")
        cycles_seen.add(cid)
        stage = rec.get("stage", "")
        if stage in by_stage:
            by_stage[stage] += 1

    # Cache-derived counters. Same source as /scan/cache/stats.
    sig_snapshot = scan_center.snapshot_signal_caches()

    scans_avoided = 0
    for iv_stats in sig_snapshot.values():
        scans_avoided += int(iv_stats.get("hit", 0) or 0)

    return {
        "total_records": len(snapshot),
        "by_stage": by_stage,
        "unique_cycles": len(cycles_seen),
        "scans_avoided": scans_avoided,
    }


# ---------------------------------------------------------------------------
# Trading-hours status (2026-04-25 — UI uses this to switch between
# "auto-scan in progress" mode and "manual 立即扫描" mode)
# ---------------------------------------------------------------------------


def _market_status_one(market: str) -> dict:
    """Return is_open + next_open + next_close for a single market.

    Uses ``core_engine.MARKET_TRADING_HOURS`` (the same source of truth
    that gates per-alert filtering). Unknown markets (e.g. CRYPTO, which
    is 24/7) fall through to ``is_open=True``, ``next_open=None``,
    ``next_close=None``.
    """
    try:
        from core_engine import MARKET_TRADING_HOURS, _is_market_open
    except Exception:  # pragma: no cover — startup race
        return {"market": market, "is_open": True, "tz": None,
                "next_open": None, "next_close": None}

    mkt = MARKET_TRADING_HOURS.get(market.upper())
    if not mkt:
        # Unknown market (CRYPTO / forex / unsupported) — treat as 24/7 open.
        return {
            "market": market, "is_open": True,
            "tz": None, "next_open": None, "next_close": None,
            "is_24x7": True,
        }

    tz = mkt["tz"]
    now = datetime.now(tz)
    is_open = _is_market_open(market)

    # Build today's open/close datetimes in the market's local tz.
    today_open = now.replace(
        hour=mkt["open_hour"], minute=mkt["open_minute"],
        second=0, microsecond=0,
    )
    today_close = now.replace(
        hour=mkt["close_hour"], minute=mkt["close_minute"],
        second=0, microsecond=0,
    )

    # Compute next_open: if currently before today's open AND today is a
    # weekday, next_open = today's open. Otherwise advance day-by-day until
    # we hit a weekday.
    next_open = None
    next_close = None
    if is_open:
        # We're inside trading hours → next_close is today's close.
        next_close = today_close
        # next_open is the next session — advance to next weekday's open.
        candidate = today_open + timedelta(days=1)
        while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
            candidate += timedelta(days=1)
        next_open = candidate
    else:
        # Closed. Could be (a) before today's open on a weekday, (b) after
        # today's close, (c) weekend.
        if now.weekday() < 5 and now < today_open:
            next_open = today_open
        else:
            # Advance to the next weekday's open.
            candidate = today_open + timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate += timedelta(days=1)
            next_open = candidate

    return {
        "market": market,
        "is_open": is_open,
        "tz": str(tz),
        "now_local": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "open_at": f"{mkt['open_hour']:02d}:{mkt['open_minute']:02d}",
        "close_at": f"{mkt['close_hour']:02d}:{mkt['close_minute']:02d}",
        "next_open": next_open.isoformat() if next_open else None,
        "next_close": next_close.isoformat() if next_close else None,
        "is_24x7": False,
    }


@router.get("/status")
def get_trading_status() -> dict:
    """Per-market trading-hours snapshot for the scan-monitor UI.

    Returns:
        {
          "checked_at": float (unix epoch),
          "any_open": bool — at least one CORE market is currently open,
          "open_markets": ["US", ...] — convenience list,
          "markets": [
            {market, is_open, tz, now_local, open_at, close_at,
             next_open (ISO), next_close (ISO), is_24x7},
            ...
          ]
        }

    The UI polls this every 30s when the scan-monitor tab is active to
    decide whether to render "🟢 自动扫描中" or "⏸ 非交易时段 · 立即扫描".

    CRYPTO markets (and any other unknown market in MARKET_TRADING_HOURS)
    return is_open=True with is_24x7=true so the UI can label them
    distinctly.
    """
    # Query every market that's in CORE_MARKETS (US/CN/HK/UK/CRYPTO as
    # of 2026-04-25). Pulling from main.CORE_MARKETS keeps this in sync
    # with the rest of the app's market list.
    try:
        from main import CORE_MARKETS
        markets_to_check = list(CORE_MARKETS)
    except Exception:  # pragma: no cover — fallback if main import fails
        markets_to_check = ["US", "CN", "HK", "UK", "CRYPTO"]

    statuses = [_market_status_one(m) for m in markets_to_check]
    open_markets = [s["market"] for s in statuses if s["is_open"] and not s.get("is_24x7")]
    any_open = bool(open_markets)
    return {
        "checked_at": time.time(),
        "any_open": any_open,
        "open_markets": open_markets,
        "markets": statuses,
    }


# ---------------------------------------------------------------------------
# Per-market scan settings (2026-04-25)
# ---------------------------------------------------------------------------
# Each CORE market (US/HK/CN/CRYPTO/UK) is an independent monitoring console
# with its own scan parameters. The settings are stored server-side
# (per_market_scan_settings.json) so they survive restarts and sync across
# browsers. The UI's left-tab navigation reads settings on tab switch and
# writes via PUT when the user clicks 💾 保存生效.


@router.get("/settings")
def get_per_market_settings() -> dict:
    """Return all CORE market settings.

    Returned shape::

        {
          "markets": {
            "US":     {"enabled": true, "interval": "1d", "refresh_sec": 300,
                        "top_n": 5, "owners": ["claude", "gpt", "gemini"]},
            "HK":     {...},
            "CN":     {...},
            "CRYPTO": {...},
            "UK":     {...}
          }
        }

    Defaults are applied for any market entry missing from disk. Always
    returns the full canonical CORE list so the UI doesn't need to know
    which markets are "configured" vs default.
    """
    try:
        from main import load_per_market_settings
    except Exception as exc:  # pragma: no cover — startup race
        raise HTTPException(status_code=503, detail=f"settings store unavailable: {exc}")
    return {"markets": load_per_market_settings()}


@router.post("/run-scan")
def run_scan(body: dict = Body(default={})) -> dict:
    """手动立即扫一次 (只到 Stage 1 出信号)。

    Body (都可选)::

        {
          "markets":  ["US"],        # 不给 = 所有启用的市场
          "interval": "1d",          # 覆盖设置里的 K 线周期
          "top_n":    5,             # 覆盖设置里的 Top N
          "owners":   ["claude"]     # 覆盖设置里的 owner
        }

    单市场时 cycle_id 前缀是 ``manual_<market>``, 会出现在该市场的 cycle 下拉里。
    """
    try:
        from core_engine import build_market_scan_groups
        from main import load_per_market_settings
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"engine unavailable: {exc}")
    if not isinstance(body, dict):
        body = {}
    settings = load_per_market_settings()
    markets_raw = body.get("markets") or []
    if not isinstance(markets_raw, list):
        markets_raw = []
    markets = [str(m).strip().upper() for m in markets_raw if str(m).strip()]
    if not markets:
        markets = [m for m, cfg in settings.items() if cfg.get("enabled")]
    unknown = [m for m in markets if m not in settings]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown market(s): {unknown}")

    overrides = {k: body[k] for k in ("interval", "top_n", "owners") if body.get(k) not in (None, "", [])}
    groups = []
    for m in markets:
        groups += build_market_scan_groups(m, {**settings[m], **overrides})

    cycle_source = f"manual_{markets[0].lower()}" if len(markets) == 1 else "manual"
    cycle_id = scan_center.new_cycle_id(cycle_source)
    if not groups:
        return {"ok": True, "cycle_id": cycle_id, "markets": markets,
                "groups": 0, "raw_alert_count": 0,
                "note": "所选市场没配股票池, 或没有匹配的 owner"}
    try:
        routed = scan_center.unified_scan(groups, scan_center._scan_caches, cycle_id=cycle_id) or {}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"scan failed: {exc}")
    return {
        "ok": True,
        "cycle_id": cycle_id,
        "markets": markets,
        "groups": len(groups),
        "raw_alert_count": sum(len(v or []) for v in routed.values()),
    }


@router.get("/aggregate")
def get_stage_aggregate(
    market: str = Query(..., description="market filter, e.g. CRYPTO"),
    limit: int = Query(200, ge=1, le=2000),
    stages: str = Query("raw_signal",
                        description="comma-separated stage names"),
) -> dict:
    """把一个市场最近若干 cycle 的 Stage 1 信号 (raw_signal) 拍平成一张列表。

    匹配条件: payload.market == <market>, 或 symbol 能映射到该市场, 或
    cycle_id 前缀是 tick_<m>- / manual_<m>-。每个 stage 取最新 ``limit`` 条。
    """
    market_upper = str(market or "").strip().upper()
    if not market_upper:
        raise HTTPException(status_code=400, detail="market required")
    wanted_stages = {s.strip() for s in stages.split(",") if s.strip()}
    if not wanted_stages:
        wanted_stages = {"raw_signal"}

    snapshot = scan_audit.snapshot_audit_ring()
    # Group by stage, in reverse-time order so we can take the newest N.
    snapshot.sort(key=lambda r: float(r.get("timestamp") or 0), reverse=True)

    # Pre-build symbol→market lookup from the most recent universe payload
    # we can find for THIS market. Symbol → market resolution lets us
    # include filter/opinion/decision records that don't carry market in
    # their own payload.
    sym_to_market: "dict[str, str]" = {}
    for rec in snapshot:
        if rec.get("stage") != "universe":
            continue
        payload = rec.get("payload") or {}
        bm = payload.get("by_market") or {}
        for m, syms in bm.items():
            mu = str(m or "").upper()
            for s in syms or []:
                sym_to_market.setdefault(str(s), mu)
        if sym_to_market:
            break  # one universe is enough — fresh enough for the audit ring

    out: "dict[str, list[dict]]" = {s: [] for s in wanted_stages}
    cycle_seen: "dict[str, str]" = {}  # cycle_id → market source for tagging
    for rec in snapshot:
        stage = rec.get("stage")
        if stage not in wanted_stages:
            continue
        payload = rec.get("payload") or {}
        # Match on payload.market (preferred) OR symbol → market lookup
        # OR cycle_id prefix (tick_<m>- / manual_<m>-).
        rec_market = str(payload.get("market") or "").upper()
        if not rec_market:
            sym = payload.get("symbol")
            if sym:
                rec_market = sym_to_market.get(str(sym), "")
        if not rec_market:
            cid = str(rec.get("cycle_id") or "")
            for prefix in (f"tick_{market_upper.lower()}-", f"manual_{market_upper.lower()}-"):
                if cid.startswith(prefix):
                    rec_market = market_upper
                    break
        if rec_market != market_upper:
            continue
        if len(out[stage]) >= limit:
            continue
        # Stamp the cycle_id source for clarity in the UI table.
        cid = str(rec.get("cycle_id") or "")
        if cid not in cycle_seen:
            if cid.startswith(f"tick_{market_upper.lower()}-"):
                cycle_seen[cid] = "auto"
            elif cid.startswith(f"manual_{market_upper.lower()}-"):
                cycle_seen[cid] = "manual"
            elif cid.startswith("tick-"):
                cycle_seen[cid] = "global"
            elif cid.startswith("manual-"):
                cycle_seen[cid] = "global-manual"
            else:
                cycle_seen[cid] = "?"
        out[stage].append({
            "cycle_id": cid,
            "cycle_source": cycle_seen[cid],
            "timestamp": rec.get("timestamp"),
            "payload": payload,
        })
    return {
        "market": market_upper,
        "stages": {s: out[s] for s in wanted_stages},
        "counts": {s: len(out[s]) for s in wanted_stages},
        "limit": limit,
    }


@router.get("/daemon-progress")
def get_daemon_progress() -> dict:
    """Per-market daemon progress snapshot for the UI's progress widget.

    Returns each CORE market's state:
        {
          "now": float (server epoch),
          "markets": {
            "US": {
              "enabled": true,
              "refresh_sec": 300,
              "last_run": float | null,        # ts of last successful start
              "last_cycle": str | null,         # cycle_id of last scan
              "next_run": float | null,         # last_run + refresh_sec
              "in_progress": bool,              # currently running?
              "in_progress_since": float | null # ts of in-flight start (if running)
            },
            ...
          }
        }

    Polled every few seconds when the user is on the scan-monitor tab so
    the UI can render '⏳ next scan in T-N sec' / '🟢 scanning since X sec'
    / '⚪ idle' for each market.
    """
    try:
        from main import load_per_market_settings
        from core_engine import _DAEMON_LOCK, _DAEMON_STATE
    except Exception as exc:  # pragma: no cover — startup race
        raise HTTPException(status_code=503, detail=f"daemon state unavailable: {exc}")

    settings = load_per_market_settings()
    now_ts = time.time()
    out: dict = {}
    with _DAEMON_LOCK:
        last_run = dict(_DAEMON_STATE.get("per_market_last_run") or {})
        last_cycle = dict(_DAEMON_STATE.get("per_market_last_cycle") or {})
        in_flight = dict(_DAEMON_STATE.get("per_market_in_progress") or {})
    for market, cfg in settings.items():
        try:
            refresh_sec = max(30, int(cfg.get("refresh_sec") or 300))
        except (TypeError, ValueError):
            refresh_sec = 300
        lr = last_run.get(market)
        ip = in_flight.get(market)
        out[market] = {
            "enabled": bool(cfg.get("enabled", True)),
            "refresh_sec": refresh_sec,
            "last_run": float(lr) if lr is not None else None,
            "last_cycle": last_cycle.get(market),
            "next_run": (float(lr) + refresh_sec) if lr is not None else None,
            "in_progress": ip is not None,
            "in_progress_since": float(ip) if ip is not None else None,
        }
    return {"now": now_ts, "markets": out}


@router.put("/settings/{market}")
def put_per_market_settings(
    market: str = Path(..., description="Uppercase market code (US/HK/CN/CRYPTO/UK)"),
    body: dict = Body(...),
) -> dict:
    """Save one market's settings (partial update merged with existing).

    Body shape (all keys optional)::

        {
          "enabled":              true | false,
          "interval":             "1d" | "5m" | "1h" | ...,
          "refresh_sec":          int (clamped to >= 30),
          "top_n":                int (>= 0; 0 means all strategies),
          "owners":               ["claude", "gpt", "gemini"],
          "pools":                ["NASDAQ 100", ...]   (要扫的股票池; 空 = 不扫)
        }

    Returns the FULL updated settings dict (all 5 markets) so the UI can
    replace its local cache atomically.
    """
    try:
        from main import save_per_market_settings, CORE_MARKETS
    except Exception as exc:  # pragma: no cover — startup race
        raise HTTPException(status_code=503, detail=f"settings store unavailable: {exc}")
    market_upper = str(market or "").upper().strip()
    if market_upper not in CORE_MARKETS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown market: {market!r} (expected one of {CORE_MARKETS})",
        )
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    try:
        full = save_per_market_settings(market_upper, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"save failed: {exc}")
    return {"markets": full, "saved": market_upper}
