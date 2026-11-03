"""Live streaming subsystem.

WebSocket-based real-time market data push. Sits alongside the existing
HTTP /quote and /kline endpoints (which remain as snapshot + fallback).

Phase 1: AAPL only, IB tick-by-tick + 1m bar aggregation, single-process
SubscriptionManager in main:app (port 9000), localhost-only.

See ``行情Live功能方案.md`` for the design doc.
"""
from __future__ import annotations

__all__ = []  # public API re-exports added by submodules as they appear
