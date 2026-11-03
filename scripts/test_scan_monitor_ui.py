#!/usr/bin/env python3
"""test_scan_monitor_ui.py — static HTML/JS markers for scan monitor UX.

We can't run the browser in CI, but we CAN assert that every functional
marker the three new UX features depend on is present in index.html and that
no orphan references to removed controls survive. Fails loudly on regressions.

Covers:
  A. Scan-settings panel rewrite (commit 4ab6e5d)
     - "扫描目标：Stage 0 合并股票池" merged-pool header line
     - #scanOptOwnerList multi-select checkbox group with 4 LLM owner boxes
     - scanMonitorTriggerScan sends all_pools:true (merged-pool scan)
     - #scanOptPool SELECT element is GONE (was the misleading single-pool dropdown)
     - scanMonitorLoadPools function is GONE (only the removal comment remains)

  B. Stage 1 per-row 📊 K线 action (commit 4ab6e5d)
     - 时间 / 操作 column headers in Stage 1 table
     - scanMonitorOpenKline / scanMonitorCloseKline function definitions
     - #scanMonitorKlineModal overlay + #scanKlineMountPoint inside it
     - event.stopPropagation on the K线 button (row-highlight toggle
       must still work elsewhere on the row)
     - renderComboKline reference (re-uses strategy-opt chart renderer)

  C. Stage 1-4 tabbed layout (commit 4ab6e5d)
     - #scanMonitorStageTabs container
     - .sm-stage-tab buttons with data-sm-stage="1..4"
     - scanMonitorSwitchStage function definition
     - tab panel divs #scanMonitorStage1Panel .. Stage4Panel
     - Stage 0 (universe) + Scan Summary stay as standalone cards ABOVE the tabs

  D. scan_summary tri-state aggregate badge (commit 9613945)
     - #scanSummaryCard container with per-bucket table

No web server spin-up; reads the served index.html via Python's filesystem.
Run:
    venv/bin/python \\
        scripts/test_scan_monitor_ui.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INDEX = _PROJECT_ROOT / "static" / "index.html"

passed = 0
failed = 0
# Keep a running, stable label list so any future drift (insertions) gets
# flagged by the total counter.
checks: "list[tuple[str, bool, str]]" = []


def check(label: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    checks.append((label, bool(cond), detail))
    if cond:
        passed += 1
    else:
        failed += 1


def report() -> None:
    total = len(checks)
    for i, (label, ok, detail) in enumerate(checks, 1):
        if ok:
            print(f"  [{i}/{total}] {label} ... OK")
        else:
            msg = f"FAIL: {label}"
            if detail:
                msg += f" ({detail})"
            print(f"  [{i}/{total}] {msg}")
    print()
    print(f"  Results: {passed}/{total} passed, {failed} failed")


if not _INDEX.exists():
    print(f"FATAL: index.html not found at {_INDEX}")
    sys.exit(2)

html = _INDEX.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# A. Scan-settings panel rewrite
# --------------------------------------------------------------------------

check(
    "A1 merged-pool header line visible",
    "扫描目标" in html and "Stage 0 合并股票池" in html,
    detail="missing '扫描目标：Stage 0 合并股票池' hint",
)

check(
    "A2 #scanOptOwnerList multi-select container present",
    'id="scanOptOwnerList"' in html,
    detail="#scanOptOwnerList div missing",
)

# Exactly 4 LLM owner checkboxes (claude / gpt / gemini / owner).
owner_cbs = re.findall(
    r'class="scan-owner-cb"\s+value="([^"]+)"', html,
)
check(
    "A3 four LLM owner checkboxes (claude/gpt/gemini/owner)",
    set(owner_cbs) == {"claude", "gpt", "gemini", "owner"}
    and len(owner_cbs) == 4,
    detail=f"found={owner_cbs}",
)

check(
    "A4 立即扫描 routes through /scan/monitor/run-scan",
    "function scanMonitorTriggerScan" in html
    and "${API}/scan/monitor/run-scan" in html,
    detail="立即扫描 must call the scan endpoint so Stage 0/1 audit records are emitted",
)

# #scanOptPool SELECT was removed — absence is the assertion.
check(
    "A5 #scanOptPool SELECT element removed (was misleading)",
    "<select" not in html.split('id="scanOptPool"', 1)[-1][:200]
    if 'id="scanOptPool"' in html else True,
    detail="#scanOptPool still has a <select> sibling (dropdown resurfaced)",
)
# Stronger: the id itself should be gone entirely. The commit removed it.
check(
    "A5b #scanOptPool id is gone entirely",
    'id="scanOptPool"' not in html,
    detail="#scanOptPool id leaked back in",
)

# scanMonitorLoadPools function body was removed (only a removal comment
# remains); the function should no longer be defined.
check(
    "A6 scanMonitorLoadPools function removed",
    "function scanMonitorLoadPools" not in html,
    detail="scanMonitorLoadPools function definition still present",
)


# --------------------------------------------------------------------------
# B. Stage 1 per-row 📊 K线 action
# --------------------------------------------------------------------------

check(
    "B1 Stage 1 table has 时间 column header",
    "时间" in html,
    detail="Stage 1 table missing 时间 column",
)

check(
    "B2 Stage 1 table has 操作 column header",
    "操作" in html,
    detail="Stage 1 table missing 操作 column",
)

check(
    "B3 scanMonitorOpenKline function defined",
    "async function scanMonitorOpenKline" in html
    or "function scanMonitorOpenKline" in html,
    detail="scanMonitorOpenKline function body missing",
)

check(
    "B4 scanMonitorCloseKline function defined",
    "function scanMonitorCloseKline" in html,
    detail="scanMonitorCloseKline function body missing",
)

check(
    "B5 #scanMonitorKlineModal overlay present",
    'id="scanMonitorKlineModal"' in html,
    detail="Pro K-line modal overlay missing",
)

# event.stopPropagation() on the per-row 📊 K线 button — required so row
# highlight toggle still works when clicking elsewhere on the row.
check(
    "B6 📊 K线 button uses event.stopPropagation (row highlight still works)",
    "event.stopPropagation()" in html and "📊 K线" in html,
    detail="stopPropagation missing or 📊 K线 button gone",
)

check(
    "B7 renderComboKline reused from strategy-opt chart flow",
    "renderComboKline" in html,
    detail="renderComboKline reference missing (Pro K线 chart would be blank)",
)


# --------------------------------------------------------------------------
# C. Stage 1-4 tabbed layout
# --------------------------------------------------------------------------

check(
    "C1 #scanMonitorStageTabs container present",
    'id="scanMonitorStageTabs"' in html,
    detail="tab container missing",
)

# Scan-only pipeline: exactly one stage tab (Stage 1 raw signals).
tab_stages = re.findall(r'data-sm-stage="(\d)"', html)
check(
    "C2 single data-sm-stage tab button (Stage 1 only)",
    tab_stages == ["1"],
    detail=f"data-sm-stage values={tab_stages}",
)

check(
    "C3 scanMonitorSwitchStage function defined",
    "function scanMonitorSwitchStage" in html,
    detail="tab switch handler missing",
)

panels_found = [
    f'id="scanMonitorStage{i}Panel"' in html for i in (1, 2, 3, 4)
]
check(
    "C4 Stage 1 panel present, Stage 2-4 panels gone",
    panels_found == [True, False, False, False],
    detail=f"panel presence={panels_found}",
)

# Stage 0 (universe) card should remain separate from the tabs — it's a
# cycle-summary card, not a tab panel. We assert it's NOT inside the tab
# container (indirect check: it has its own scanMonitorStage0Card id).
# Stage 0 card uses inline id="scanMonitorStage0Count" / "scanMonitorStage0Body"
# (not a wrapper id). Presence of both means the card body is wired in.
check(
    "C5 Stage 0 card kept ABOVE tabs (Count+Body markers present)",
    'id="scanMonitorStage0Count"' in html
    and 'id="scanMonitorStage0Body"' in html,
    detail="Stage 0 universe card markers missing (merged-pool summary gone)",
)


# --------------------------------------------------------------------------
# E. Per-market stock-pool picker (scan settings)
# --------------------------------------------------------------------------

check(
    "E1 pool picker container present",
    'id="scanOptPoolList"' in html,
    detail="#scanOptPoolList missing — user can't choose which pools a market scans",
)

check(
    "E2 pool picker renderer defined and fed from settings",
    "function _scanMonitorRenderPoolList" in html
    and "_scanMonitorRenderPoolList(_scanMonitorActiveMarket || 'US', s.pools || [])" in html,
    detail="pool picker not rendered from the market's saved pools",
)

check(
    "E3 save payload includes pools",
    "pools: formSettings.pools," in html,
    detail="saved settings drop the pool selection",
)

check(
    "E4 symbol links carry data-sm-pool-symbol (no truncation)",
    "data-sm-pool-symbol" in html
    and "symbols.slice(0, 40)" not in html  # old truncation pattern
    and "+ ${symbols.length-40} more" not in html,
    detail="symbols still truncated or pool-symbol attr missing",
)

check(
    "E5 per-group pool chips removed",
    "_scanMonitorPoolByComp" not in html
    and "scanMonitorHighlightPool(" not in html,
    detail="per-group chip code still present",
)


# --------------------------------------------------------------------------
# D. scan_summary aggregate badge + per-bucket table
# --------------------------------------------------------------------------

check(
    "D1 #scanMonitorSummaryCard aggregate panel present",
    'id="scanMonitorSummaryCard"' in html
    and 'id="scanMonitorSummaryBody"' in html
    and 'id="scanMonitorSummaryBadge"' in html,
    detail="scan_summary card/body/badge missing — silent cycles would still look like no-signal",
)

# The aggregate table + state badge must be wired in. The renderer uses
# scanMonitorSummary* ids; body gets populated by scanMonitorRenderStages.
check(
    "D2 scan_summary renderer wired in (scanMonitorSummary* refs)",
    "scanMonitorSummaryBody" in html
    and "scanMonitorSummaryBadge" in html,
    detail="no renderer references; aggregate card will stay empty",
)


# --------------------------------------------------------------------------
# F. Scan-settings DEFAULTS (2026-04-25 — user spec'd)
# --------------------------------------------------------------------------
# Defaults the user requested in the screenshot:
#   K线周期 = 日线 (1d)
#   刷新频率 = 5分钟 (300)
#   信号方向 = 两者 Both (both)
#   策略    = Top 5 (top5)
#   LLM Owner: ✓claude ✓gpt ✓gemini  ☐owner (人工)
#   Auto refresh checkbox: checked at page-load
# Also: initScanMonitorTab() must kick off the timer when Auto is pre-checked.
# Together this gives "在交易时间一直扫描" without requiring any clicks.

# Match the SELECTED option of each select. Use a tight regex anchored to the
# select-id so we don't accidentally match a different control with the same
# value.
def _selected_value(select_id: str, html_text: str) -> str | None:
    m = re.search(
        rf'<select[^>]*id="{re.escape(select_id)}"[^>]*>(.*?)</select>',
        html_text, re.DOTALL,
    )
    if not m:
        return None
    block = m.group(1)
    sel = re.search(r'<option\s+value="([^"]+)"\s+selected\b', block)
    return sel.group(1) if sel else None


check(
    "F1 K线周期 default = 1d (日线)",
    _selected_value("scanOptInterval", html) == "1d",
    detail=f"got selected={_selected_value('scanOptInterval', html)}",
)
check(
    "F2 刷新频率 default = 300s (5分钟)",
    _selected_value("scanOptAutoFreq", html) == "300",
    detail=f"got selected={_selected_value('scanOptAutoFreq', html)}",
)
check(
    "F3 信号方向 dropdown REMOVED (now implicit via BUY/SELL pool split)",
    'id="scanOptSide"' not in html,
    detail="#scanOptSide dropdown still present — it should be gone",
)
# 立即扫描 routes through /run-scan; BUY/SELL split is handled by
# scan_center._collect_universe_for_side. No signal_side knob in the request body.
check(
    "F3b 立即扫描 routes through /run-scan (BUY/SELL split is structural)",
    "${API}/scan/monitor/run-scan" in html
    and "function scanMonitorTriggerScan" in html,
    detail="立即扫描 must use /run-scan so BUY/SELL split is handled by _collect_universe_for_side",
)
check(
    "F4 策略 default = top5 (Top 5)",
    _selected_value("scanOptStrategyMode", html) == "top5",
    detail=f"got selected={_selected_value('scanOptStrategyMode', html)}",
)

# LLM Owner: claude/gpt/gemini default-checked, owner default-unchecked.
def _owner_checked(value: str, html_text: str) -> bool:
    # Match: <input ... class="scan-owner-cb" value="<v>" ... checked ... />
    # Or:    <input ... value="<v>" ... class="scan-owner-cb" ... checked />
    m = re.search(
        rf'<input[^>]*class="scan-owner-cb"[^>]*value="{re.escape(value)}"[^>]*?>',
        html_text,
    )
    if not m:
        return False
    return "checked" in m.group(0)


check(
    "F5 LLM Owner default — Claude checked",
    _owner_checked("claude", html),
    detail="claude checkbox not checked at page-load",
)
check(
    "F6 LLM Owner default — GPT checked",
    _owner_checked("gpt", html),
    detail="gpt checkbox not checked at page-load",
)
check(
    "F7 LLM Owner default — Gemini checked",
    _owner_checked("gemini", html),
    detail="gemini checkbox not checked at page-load",
)
check(
    "F8 LLM Owner default — Owner (人工) UNchecked",
    not _owner_checked("owner", html),
    detail="owner checkbox is pre-checked (should default to off — manual is opt-in)",
)

# Auto refresh checkbox must be CHECKED at page load so polling starts
# immediately when the user opens the tab.
check(
    "F9 Auto refresh checkbox checked at page-load",
    bool(re.search(
        r'<input[^>]*id="scanMonitorAuto"[^>]*\bchecked\b',
        html,
    )),
    detail="#scanMonitorAuto missing 'checked' attribute (Auto would default off)",
)

# initScanMonitorTab must start the timer when checkbox is pre-checked
# (otherwise pre-checking does nothing until the user clicks).
check(
    "F10 initScanMonitorTab kicks off timer when Auto is pre-checked",
    "if (autoCb && autoCb.checked && !_scanMonitorAutoTimer)" in html,
    detail="initScanMonitorTab does not start timer when checkbox is pre-checked",
)


# --------------------------------------------------------------------------
# G. Per-market tabs + dual Stage 0 sub-cards (BUY pool / SELL positions)
#    Spec'd 2026-04-25 — split the merged universe into market tabs
#    (US/HK/CN/Crypto/...) where each tab shows TWO pools:
#      • 合并股票池 (BUY 监控) — universe[market].buy_pool
#      • 合并持仓池 (SELL 监控) — paper+live positions union
# --------------------------------------------------------------------------

check(
    "G1 #scanMonitorMarketTabs container present",
    'id="scanMonitorMarketTabs"' in html,
    detail="market tab bar div missing",
)

# G1b (2026-04-25): tab bar must be the TOPMOST element in the scan-monitor
# tab content — even before 扫描设置 — because each market is its own
# independent monitoring console (own settings/status/cycles).
check(
    "G1b market tabs are TOPMOST in tab content (before 扫描设置 card)",
    "📍 市场监测台:" in html
    and html.find("📍 市场监测台:") < html.find("⚙️ 扫描设置"),
    detail="market tabs not above scan-settings card — should be the very first element",
)

# G1c: per-market settings persistence — settings form must save/load
# per-market via localStorage so each market is a distinct console.
check(
    "G1c per-market settings persistence (localStorage scanMonitorSettings)",
    'localStorage.getItem(\'scanMonitorSettings\')' in html
    and 'localStorage.setItem(\'scanMonitorSettings\'' in html
    and 'function _scanMonitorReadFormSettings' in html
    and 'function _scanMonitorWriteFormSettings' in html,
    detail="missing per-market localStorage round-trip — settings won't be market-specific",
)

# G1d: scanMonitorSetActiveMarket must LOAD new market's settings into the
# form on click. Backend canonicalization removed the "save old form on
# switch" step (settings only persist on explicit 💾 保存生效 click), but
# the new market's settings must still be loaded — converted from backend
# shape ({enabled, interval, refresh_sec, top_n, owners}) to UI form shape.
check(
    "G1d scanMonitorSetActiveMarket loads new market's settings on click",
    "function scanMonitorSetActiveMarket" in html
    and "_scanMonitorWriteFormSettings({" in html
    and "freq: String(s.refresh_sec" in html,
    detail="market switch doesn't load that market's saved settings into form",
)

# G1e: tabs always render the CORE markets, not just markets with cycle data.
check(
    "G1e SCAN_MONITOR_CORE_MARKETS canonical list with US/HK/CN/CRYPTO/UK",
    "SCAN_MONITOR_CORE_MARKETS = ['US', 'HK', 'CN', 'CRYPTO', 'UK']" in html,
    detail="canonical CORE markets list missing — tabs only render markets with data",
)

# G1f: status bar must filter to ACTIVE market only (not any-open).
check(
    "G1f status bar renders for the ACTIVE market only",
    "const target = (j.markets || []).find(m => String(m.market || '').toUpperCase() === activeM);" in html,
    detail="status bar still shows 'any-open' across all markets — should filter to active",
)

check(
    "G2 active-market state vars declared",
    "_scanMonitorActiveMarket" in html
    and "_scanMonitorBuyPool" in html
    and "_scanMonitorSellPool" in html,
    detail="market state vars missing — tab switch would have nothing to track",
)

check(
    "G3 scanMonitorSetActiveMarket function defined",
    "function scanMonitorSetActiveMarket" in html,
    detail="market-tab click handler missing",
)

check(
    "G4 scanMonitorApplyMarketFilter function defined",
    "function scanMonitorApplyMarketFilter" in html,
    detail="Stage 1-4 row dimmer missing — cross-market noise won't be hidden",
)

check(
    "G5 universe payload reads buy_pool/sell_pool",
    "payload.buy_pool" in html and "payload.sell_pool" in html,
    detail="render block doesn't consume the new payload keys",
)

# Both sub-cards (BUY/SELL labels) must be rendered.
check(
    "G6 Stage 0 renders 合并股票池（BUY）sub-card",
    "合并股票池（BUY 监控）" in html,
    detail="BUY sub-card label missing",
)
check(
    "G7 Stage 0 renders 合并持仓池（SELL）sub-card",
    "合并持仓池（SELL 监控）" in html,
    detail="SELL sub-card label missing",
)

# Tab markup: button per market with data-sm-market and onclick handler.
check(
    "G8 market tab buttons carry data-sm-market + onclick",
    "data-sm-market=" in html
    and 'onclick="scanMonitorSetActiveMarket(' in html,
    detail="tab buttons not wired to setActiveMarket()",
)

# CRYPTO must appear in the canonical CORE markets list (otherwise Crypto
# tab won't render in the right slot — covered by G1e). Keep G9 as a
# defense-in-depth check that the literal "CRYPTO" string appears in the
# CORE_MARKETS declaration.
check(
    "G9 CRYPTO included in SCAN_MONITOR_CORE_MARKETS",
    "SCAN_MONITOR_CORE_MARKETS" in html and "'CRYPTO'" in html.split("SCAN_MONITOR_CORE_MARKETS", 1)[1][:200],
    detail="CRYPTO missing from canonical CORE list — Crypto tab won't render",
)

# Filter is wired at the END of scanMonitorRenderStages (after Stage 1-4
# panels exist in DOM) — calling it earlier finds no rows to dim.
check(
    "G10 scanMonitorApplyMarketFilter called from end of scanMonitorRenderStages",
    "// After the Stage 1 table is rendered, dim rows that don't belong to the" in html
    and html.find("scanMonitorApplyMarketFilter()") < html.find("function scanMonitorRenderTable"),
    detail="market filter not invoked after Stage 1-4 render — dimming would no-op",
)


# --------------------------------------------------------------------------
# H. Trading-hours-aware status bar + 保存生效 button (2026-04-25)
#    During market hours: 🟢 自动扫描中 + dim manual button.
#    Outside market hours: ⏸ 非交易时段 + prominent 立即扫描 button.
#    Save button applies pending dropdown/checkbox changes; freq dropdown
#    no longer auto-applies on change (would defeat the explicit save UX).
# --------------------------------------------------------------------------

check(
    "H1 #scanOptStatusBadge present in scan-settings panel",
    'id="scanOptStatusBadge"' in html,
    detail="status badge missing — UI can't show trading-hours state",
)

check(
    "H2 #scanOptStatusDetail present (next-open/close hint container)",
    'id="scanOptStatusDetail"' in html,
    detail="status detail text container missing",
)

check(
    "H3 立即扫描 button has stable id #scanOptManualBtn",
    'id="scanOptManualBtn"' in html
    and "▶️ 立即扫描" in html,
    detail="manual scan button id missing — status renderer can't promote/dim it",
)

check(
    "H4 #scanOptSaveBtn 保存生效 button present",
    'id="scanOptSaveBtn"' in html
    and "💾 保存生效" in html,
    detail="save button missing",
)

check(
    "H5 scanMonitorMarkSettingsDirty function defined",
    "function scanMonitorMarkSettingsDirty" in html,
    detail="dirty marker missing — save button won't highlight on change",
)

check(
    "H6 scanMonitorSaveSettings function defined (applies pending changes)",
    "function scanMonitorSaveSettings" in html
    and "scanMonitorToggleAuto()" in html,  # save reseeds the auto timer
    detail="save handler missing or doesn't reseed timer",
)

check(
    "H7 freq dropdown delegates to MarkSettingsDirty (not auto-apply)",
    'onchange="scanMonitorMarkSettingsDirty()"' in html
    and 'id="scanOptAutoFreq"' in html,
    detail="freq dropdown still auto-applies on change — defeats explicit save UX",
)

# Status polling — must start when the tab opens.
check(
    "H8 scanMonitorStartStatusPoll wired into initScanMonitorTab",
    "scanMonitorStartStatusPoll();" in html,
    detail="status poll never starts — badge stays at '加载状态…'",
)

check(
    "H9 scanMonitorLoadStatus calls /scan/monitor/status endpoint",
    "/scan/monitor/status" in html
    and "function scanMonitorLoadStatus" in html,
    detail="status fetcher missing or wrong URL",
)

# Render branch: trading hours → 🟢 ; non-trading → ⏸.
check(
    "H10 status renderer has both 🟢 and ⏸ branches",
    "🟢 自动扫描中" in html
    and "⏸ 非交易时段" in html,
    detail="status renderer missing one of the two branches",
)


# --------------------------------------------------------------------------
# I. Backend integration for per-market settings (2026-04-25)
#    Settings are now server-side canonical (PUT /scan/monitor/settings/<m>)
#    instead of localStorage-only. localStorage is a fallback cache.
#    立即扫描 sends markets=[active] so the backend filters universe.
# --------------------------------------------------------------------------

check(
    "I1 _scanMonitorFetchSettings calls GET /scan/monitor/settings",
    "_scanMonitorFetchSettings" in html
    and "/scan/monitor/settings" in html,
    detail="settings GET endpoint not wired",
)

check(
    "I2 _scanMonitorPutSettings calls PUT /scan/monitor/settings/<market>",
    "_scanMonitorPutSettings" in html
    and "method: 'PUT'" in html,
    detail="settings PUT endpoint not wired",
)

check(
    "I3 saveSettings sends backend payload {enabled, interval, refresh_sec, top_n, owners}",
    "_scanMonitorPutSettings(activeM, backendPayload)" in html
    and "refresh_sec: parseInt(formSettings.freq, 10)" in html,
    detail="save handler doesn't translate UI form to backend shape",
)

check(
    "I4 initScanMonitorTab fetches server settings on tab open",
    "await _scanMonitorFetchSettings()" in html,
    detail="init does not fetch server settings — UI would only show localStorage cache",
)

check(
    "I5 立即扫描 sends markets=[active_market] filter",
    "markets: activeMarket ? [activeMarket] : []" in html,
    detail="manual scan body missing markets filter — would scan globally",
)


# --------------------------------------------------------------------------
# J. 2026-04-26 morning fixes — bug 2 (cycle dropdown filter) + bug 3
#    (K-line modal market resolution).
# --------------------------------------------------------------------------

check(
    "J1 cycle dropdown filtered by active market (per-tab subset)",
    "function _scanMonitorCycleMatchesMarket" in html
    and ".filter(c => _scanMonitorCycleMatchesMarket(" in html,
    detail="cycle dropdown still shows ALL cycles regardless of market tab",
)

check(
    "J2 _scanMonitorCycleMatchesMarket excludes cross-market cycles from per-market tabs (revised PM)",
    "return cycleId.startsWith(`tick_${m}-`) || cycleId.startsWith(`manual_${m}-`);" in html,
    detail="filter still includes legacy tick- / cross-market manual- cycles — pollutes per-market dropdowns with cross-market data",
)

check(
    "J3 market switch reloads cycle dropdown",
    "scanMonitorLoadCycles().then(()" in html
    and "scanMonitorLoadCycle(sel.value)" in html,
    detail="dropdown does not refresh on market switch — stale options remain",
)

check(
    "J4 周期 tooltip explains the metric to users",
    "什么是 Cycle / 周期" in html
    and "tick_us-…" in html,
    detail="cycle tooltip missing — user has no in-UI hint about what 周期 means",
)

# Bug 3: K-line modal must accept and prefer the row's market argument.
check(
    "J5 scanMonitorOpenKline accepts market parameter",
    "function scanMonitorOpenKline(symbol, strategy, interval, market)" in html,
    detail="scanMonitorOpenKline signature missing market param — falls back to global #market",
)

check(
    "J6 K线 button passes p.market from alert payload",
    "scanMonitorOpenKline('${_esc(sym)}','${_esc(strat)}','${_esc(iv)}','${_esc(mkt)}')" in html
    and "const mkt = p.market != null" in html,
    detail="K-line button doesn't pass row's market — modal falls back to wrong market",
)

check(
    "J7 K-line modal uses resolvedMarket (not raw global selector) in fetch URL",
    "${API}/backtest/${resolvedMarket}/${symbol}" in html
    and "let resolvedMarket = (market && String(market).trim()) || ''" in html,
    detail="K-line fetch URL still uses unconditional global market selector",
)


# --------------------------------------------------------------------------
# K. 2026-04-26 PM: Stage 1-4 must FILTER by active market (not just dim)
#    so cross-market noise (e.g., US symbols on the CRYPTO tab) is gone
#    from the rendered table — opacity dimming was visible-but-faded,
#    user complained the rows shouldn't appear at all.
# --------------------------------------------------------------------------

check(
    "K1 Stage 1 records filtered by active market (rawAll exists)",
    "const rawAll = stages.raw_signal" in html,
    detail="Pre-filter bucket missing — counts/dim cannot show full vs filtered",
)

check(
    "K2 _matchActive helper checks payload.market THEN symbol→market lookup",
    "const _matchActive = (rec) =>" in html
    and "if (p.market) return String(p.market).toUpperCase() === activeM;" in html
    and "return _activeMarketSet.has(String(sym));" in html,
    detail="market match logic missing — records without market need symbol lookup",
)

check(
    "K3 cycleByMarket built from THIS cycle's universe payload",
    "const cycleByMarket = (uni.length && (uni[0].payload || {}).by_market) || {};" in html,
    detail="must use cycle's own by_market (not stale module global) for filtering",
)

check(
    "K4 raws is the filter()'d version of rawAll",
    "const raws = rawAll.filter(_matchActive);" in html,
    detail="filter step missing on Stage 1",
)

# --------------------------------------------------------------------------
# L. 2026-04-26 PM: scan-target label per-market context + daemon progress
#    widget so user sees whether the backend is actually working.
# --------------------------------------------------------------------------

check(
    "L1 scan-settings header shows active market label",
    'id="scanOptSettingsActiveMarket"' in html
    and "scanMonitorMarketLabel(market)" in html,
    detail="scan-settings header doesn't dynamically show active market — old 'Stage 0 合并股票池' label is misleading per-market",
)

check(
    "L2 scan target hint reflects per-market scope (not 'all merged pools')",
    "扫描目标：下面勾选的本市场股票池" in html,
    detail="target hint still says 'Stage 0 合并股票池 / 所有监测组的 pools 取并集' — wrong for per-market consoles",
)

check(
    "L3 daemon progress widget present (badge + detail + tooltip)",
    'id="scanOptProgressBadge"' in html
    and 'id="scanOptProgressDetail"' in html
    and "daemon 进度指示器" in html,
    detail="progress widget missing — user can't tell if backend is working",
)

check(
    "L4 scanMonitorStartProgressPoll wired into initScanMonitorTab + 3s cadence",
    "scanMonitorStartProgressPoll();" in html
    and "setInterval(scanMonitorLoadProgress, 3 * 1000)" in html,
    detail="progress poll never starts — badge stays at 'loading…'",
)

check(
    "L5 progress renderer has 3 branches (in_progress / countdown / idle)",
    "🟢 ${activeM} 扫描中" in html
    and "⏳ ${activeM} 下次扫描" in html
    and "⚪ ${activeM} daemon" in html,
    detail="progress renderer missing one of running/countdown/idle states",
)

check(
    "L6 progress fetches /scan/monitor/daemon-progress",
    "${API}/scan/monitor/daemon-progress" in html
    and "function scanMonitorLoadProgress" in html,
    detail="progress fetcher missing or wrong URL",
)

check(
    "L7 market switch re-renders progress badge with current market",
    "if (_scanMonitorProgressCache) {" in html
    and "scanMonitorRenderProgressBadge(_scanMonitorProgressCache);" in html,
    detail="progress badge stuck on prior market after tab switch",
)


# --------------------------------------------------------------------------
# M. 2026-04-26 PM: '走完整流程' button to push manual signals through
#    Stage 2/3/4 for testing.
# --------------------------------------------------------------------------

# After 2026-04-26 consolidation, the dedicated 走完整流程 button is GONE
# (merged into 立即扫描). M1/M2/M3 below cover the consolidated 立即扫描.

check(
    "M1 立即扫描 button is the single scan trigger (consolidated)",
    'id="scanOptManualBtn"' in html
    and "立即扫描" in html,
    detail="立即扫描 button missing — should be the single scan trigger after consolidation",
)

check(
    "M2 run-full-pipeline / refresh-llm endpoints no longer referenced",
    "run-full-pipeline" not in html and "refresh-llm" not in html,
    detail="frontend still calls removed pipeline endpoints",
)

check(
    "M3 button disabled while scan runs (avoids double-click + visual cue)",
    "btn.disabled = true;" in html
    and "btn.textContent = '⏳ 扫描中…'" in html
    and "btn.disabled = false;" in html,
    detail="button doesn't toggle disabled state — user could spam click the slow endpoint",
)

check(
    "M4 result message echoes raw_alert_count / groups",
    "原始信号 ${j.raw_alert_count}" in html
    and "${j.groups} 组" in html,
    detail="result message doesn't show scan stats",
)

check(
    "M5 立即扫描 calls /run-scan scoped to the active market",
    "markets: activeMarket ? [activeMarket] : []," in html
    and "${API}/scan/monitor/run-scan" in html
    and "function scanMonitorTriggerScan" in html,
    detail="scan call doesn't scope by market",
)

# After consolidation, the dedicated 走完整流程 button is gone — only one
# button (立即扫描) which does the full pipeline. Verify the obsolete
# button/function-id is no longer present.
check(
    "M5b 走完整流程 button removed (consolidated into 立即扫描)",
    'id="scanOptFullPipelineBtn"' not in html,
    detail="standalone 走完整流程 button still present — should be merged into 立即扫描",
)

check(
    "M5c 立即扫描 button label has no stale '5 stage' wording",
    "▶️ 立即扫描" in html and "5 stage" not in html,
    detail="button label still advertises the removed 5-stage pipeline",
)


# --------------------------------------------------------------------------
# N. 2026-04-26 audit: comprehensive cross-market leak prevention.
#    Every data render path must scope to active market — no exceptions.
# --------------------------------------------------------------------------

check(
    "N1 stats summary shows per-market records + cycles (not just global)",
    "perMarketRecords" in html
    and "perMarketCycles" in html
    and "信号缓存" in html,
    detail="stats summary line still shows pure global totals — leaks cross-market record counts",
)

check(
    "N2 scan_summary panel filters by payload.market (active-market only)",
    "summaryAll = stages.scan_summary || [];" in html
    and "summaryAll.filter(r =>" in html
    and "pm === _activeMUpper" in html,
    detail="scan_summary panel doesn't filter by payload.market — could show other markets' buckets if cross-market cycle leaked",
)

check(
    "N3 every fetch URL with market context routes through filter (no naked global call)",
    # All these endpoints accept or filter by market; spot-check that the
    # frontend is using them correctly.
    "${API}/scan/monitor/run-scan" in html
    and "markets: activeMarket ? [activeMarket] : []" in html
    and "${API}/scan/monitor/settings/${encodeURIComponent(market)}" in html,
    detail="some fetch is hitting an unscoped endpoint",
)

# 走完整流程 must also pass interval / top_n / owners from the UI form
# so the pipeline honors what the user sees, not the monitor groups'
# DB-default settings (which usually default to 1d).
check(
    "N4 走完整流程 passes interval / top_n / owners from form (UI overrides)",
    "interval: intervalVal," in html
    and "top_n: topN," in html
    and "owners: ownerVals," in html,
    detail="pipeline call doesn't pass UI form values — backend uses DB-default 1d/etc.",
)

check(
    "N5 立即扫描 result echoes used interval / top_n / owners",
    "${intervalVal}/Top${topN}/${ownersLabel}" in html,
    detail="result message doesn't echo the actually-used settings",
)


# --------------------------------------------------------------------------
report()
sys.exit(0 if failed == 0 else 1)
