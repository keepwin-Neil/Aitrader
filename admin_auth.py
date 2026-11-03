"""Optional admin-token gate for Scan Center ops endpoints (§7.3).

Protects write-side operations — ``POST /policy/reload``,
``POST /scan/cache/opinion/purge``, ``POST /scan/cache/signal/purge`` —
with a shared-secret token when ``AITRADER_ADMIN_TOKEN`` is set in the
environment. When the env var is unset the gate is a no-op.

Usage:
    from fastapi import Depends
    from admin_auth import require_admin

    @router.post(
        "/policy/reload",
        dependencies=[Depends(require_admin)],
    )
    def reload_policy_config(): ...

Threat model: the app runs on localhost only. The gate is optional;
default unset. Opt in by setting ``AITRADER_ADMIN_TOKEN`` in ``.env``.

Constant-time header compare via :func:`hmac.compare_digest` to avoid
timing-oracle leaks.
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

from fastapi import Header, HTTPException, status

log = logging.getLogger(__name__)

_ENV_TOKEN_NAME = "AITRADER_ADMIN_TOKEN"


def require_admin(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """FastAPI dependency.

    Behavior:
      * ``AITRADER_ADMIN_TOKEN`` unset → allow all (backward compat).
      * ``AITRADER_ADMIN_TOKEN`` set → require ``X-Admin-Token`` header
        matching it (constant-time compare). Missing / mismatched /
        empty header → HTTP 401.

    Does NOT leak whether the env var is set via response body; 401
    detail is intentionally generic.
    """
    expected = os.environ.get(_ENV_TOKEN_NAME)
    if not expected:
        # Gate disabled — env var not configured for this deploy.
        return

    if not x_admin_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="admin token required",
        )

    # Constant-time compare prevents timing attacks.
    if not hmac.compare_digest(expected, x_admin_token):
        log.warning("admin_auth: X-Admin-Token mismatch")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="admin token invalid",
        )
